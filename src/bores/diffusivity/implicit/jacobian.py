import functools
import itertools
import typing

import numba
import numpy as np
from scipy.sparse import lil_matrix

from bores.diffusivity.base import to_1D_index_interior_only
from bores.models import FluidProperties, RockFluidProperties, RockProperties
from bores.pvt.core import compute_harmonic_mean
from bores.types import (
    CapillaryPressureTable,
    RelativePermeabilityTable,
    ThreeDimensionalGrid,
    ThreeDimensions,
)


@numba.njit(cache=True)
def assmeble_accumulation_derivatives(
    porosity: float,
    cell_volume: float,
    oil_saturation: float,
    gas_saturation: float,
    water_saturation: float,
    oil_compressibility: float,
    gas_compressibility: float,
    water_compressibility: float,
    dtype: np.typing.DTypeLike,
) -> np.ndarray:
    """
    Compute and assemble all accumulation term derivatives.

    Accumulation: A_α = φ·V·S_α

    Derivatives:
    ∂A_α/∂P = φ·V·S_α·c_α
    ∂A_oil/∂S_o = φ·V
    ∂A_gas/∂S_g = φ·V
    ∂A_water/∂S_o = -φ·V (since ∂S_w/∂S_o = -1)
    ∂A_water/∂S_g = -φ·V (since ∂S_w/∂S_g = -1)
    """
    derivatives = np.zeros((3, 3), dtype=dtype)
    pore_volume = porosity * cell_volume

    # ∂R/∂P (column 0)
    # Derivatives w.r.t. Pressure
    # Accumulation terms: φ·V·Sα·cα
    derivatives[0, 0] = pore_volume * oil_saturation * oil_compressibility  # ∂R_oil/∂P
    derivatives[1, 0] = pore_volume * gas_saturation * gas_compressibility  # ∂R_gas/∂P
    derivatives[2, 0] = (
        pore_volume * water_saturation * water_compressibility
    )  # ∂R_water/∂P

    # ∂R/∂S_o (column 1)
    # Derivatives w.r.t. So
    # Only accumulation terms (mobility frozen in quasi-Newton)
    derivatives[0, 1] = pore_volume  # ∂R_oil/∂So
    derivatives[1, 1] = 0.0  # ∂R_gas/∂So
    derivatives[2, 1] = -pore_volume  # ∂R_water/∂So (since ∂Sw/∂So = -1)

    # ∂R/∂S_g (column 2)
    # Derivatives w.r.t. Sg
    derivatives[0, 2] = 0.0  # ∂R_oil/∂Sg
    derivatives[1, 2] = pore_volume  # ∂R_gas/∂Sg
    derivatives[2, 2] = -pore_volume  # ∂R_water/∂Sg (since ∂Sw/∂Sg = -1)
    return derivatives


@numba.njit(cache=True)
def compute_phase_flux_derivative(
    cell_index: typing.Tuple[int, int, int],
    neighbour_index: typing.Tuple[int, int, int],
    mobility_grid: ThreeDimensionalGrid,
    geometric_factor: float,
) -> float:
    """
    Compute flux derivative w.r.t. the current pressure for a single phase between a cell and its neighbour.

    Since flux F = T·(P_cell - P_neighbour + capillary_terms + gravity_terms),
    flux derivative w.r.t. current pressure is simply the transmissibility T.

    :param cell_index: Index of the current cell (i, j, k)
    :param neighbour_index: Index of the neighbouring cell (i, j, k)
    :param mobility_grid: Mobility grid for the phase
    :param geometric_factor: Geometric factor for the face between the cells
    :return: Flux derivative w.r.t. current pressure
    """
    harmonic_mobility = compute_harmonic_mean(
        mobility_grid[neighbour_index], mobility_grid[cell_index]
    )
    transmissibility = harmonic_mobility * geometric_factor
    return transmissibility


@numba.njit(cache=True)
def compute_effective_saturation(
    saturation: float,
    residual_saturation: float,
    irreducible_saturation: float,
) -> float:
    """
    Compute effective saturation.

    S_eff = (S - S_r) / (1 - S_r - S_irr)

    :param saturation: Phase saturation
    :param residual_saturation: Residual saturation of the phase
    :param irreducible_saturation: Irreducible saturation of the phase
    :return: Effective saturation
    """
    denominator = 1.0 - residual_saturation - irreducible_saturation
    if denominator <= 0.0:
        return 0.0
    effective_saturation = (saturation - residual_saturation) / denominator
    return effective_saturation


@numba.njit(cache=True)
def compute_saturation_perturbation(
    saturation: float,
    saturation_min: float,
    saturation_max: float,
    absolute_perturbation: float = 1e-8,
    relative_perturbation: float = 1e-6,
) -> float:
    span = max(
        saturation - saturation_min,
        saturation_max - saturation,
    )
    return max(absolute_perturbation, relative_perturbation * span)


def compute_phase_mobility_derivative(
    saturation: float,
    saturation_min: float,
    saturation_max: float,
    viscosity: float,
    kr_func: typing.Callable[
        [float], float
    ],  # callable passed via numba-compatible closure
    absolute_perturbation: float,
    relative_perturbation: float,
    phase_appearance_tolerance: float = 1e-6,
) -> float:
    # Phase not present → no mobility, no derivative
    if saturation <= saturation_min + phase_appearance_tolerance:
        return 0.0

    delta = compute_saturation_perturbation(
        saturation=saturation,
        saturation_min=saturation_min,
        saturation_max=saturation_max,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
    )

    # Decide differencing scheme
    can_forward = saturation + delta <= saturation_max
    can_backward = saturation - delta >= saturation_min

    if can_forward and can_backward:
        kr_plus = kr_func(saturation + delta)
        kr_minus = kr_func(saturation - delta)
        return (kr_plus - kr_minus) / (2.0 * delta * viscosity)

    elif can_forward:
        kr_plus = kr_func(saturation + delta)
        kr_0 = kr_func(saturation)
        return (kr_plus - kr_0) / (delta * viscosity)

    elif can_backward:
        kr_0 = kr_func(saturation)
        kr_minus = kr_func(saturation - delta)
        return (kr_0 - kr_minus) / (delta * viscosity)
    return 0.0


def compute_mobility_derivatives(
    cell_index: typing.Tuple[int, int, int],
    oil_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    residual_oil_saturation_water_grid: ThreeDimensionalGrid,
    residual_oil_saturation_gas_grid: ThreeDimensionalGrid,
    residual_gas_saturation_grid: ThreeDimensionalGrid,
    irreducible_water_saturation_grid: ThreeDimensionalGrid,
    oil_viscosity_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    relperm_table: RelativePermeabilityTable,
    absolute_perturbation: float = 1e-8,
    relative_perturbation: float = 1e-6,
    phase_appearance_tolerance: float = 1e-6,
) -> typing.Tuple[float, float, float]:
    i, j, k = cell_index

    So = oil_saturation_grid[i, j, k]
    Sg = gas_saturation_grid[i, j, k]
    Sw = water_saturation_grid[i, j, k]

    Sorw = residual_oil_saturation_water_grid[i, j, k]
    Sorg = residual_oil_saturation_gas_grid[i, j, k]
    Sgr = residual_gas_saturation_grid[i, j, k]
    Swirr = irreducible_water_saturation_grid[i, j, k]

    mu_o = oil_viscosity_grid[i, j, k]
    mu_w = water_viscosity_grid[i, j, k]
    mu_g = gas_viscosity_grid[i, j, k]

    # Admissible saturation bounds
    So_min = max(Sorw, Sorg)
    So_max = 1.0 - Swirr - Sgr

    Sg_min = Sgr
    Sg_max = 1.0 - Swirr - Sorg

    Sw_min = Swirr
    Sw_max = 1.0 - Sorw - Sgr

    # Oil
    def kr_oil(S):
        return relperm_table(
            oil_saturation=S,
            gas_saturation=Sg,
            water_saturation=Sw,
            residual_oil_saturation_water=Sorw,
            residual_oil_saturation_gas=Sorg,
            residual_gas_saturation=Sgr,
            irreducible_water_saturation=Swirr,
        )["oil"]

    dλo_dSo = compute_phase_mobility_derivative(
        saturation=So,
        saturation_min=So_min,
        saturation_max=So_max,
        viscosity=mu_o,
        kr_func=kr_oil,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
        phase_appearance_tolerance=phase_appearance_tolerance,
    )

    # Gas
    def kr_gas(S):
        return relperm_table(
            oil_saturation=So,
            gas_saturation=S,
            water_saturation=Sw,
            residual_oil_saturation_water=Sorw,
            residual_oil_saturation_gas=Sorg,
            residual_gas_saturation=Sgr,
            irreducible_water_saturation=Swirr,
        )["gas"]

    dλg_dSg = compute_phase_mobility_derivative(
        saturation=Sg,
        saturation_min=Sg_min,
        saturation_max=Sg_max,
        viscosity=mu_g,
        kr_func=kr_gas,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
        phase_appearance_tolerance=phase_appearance_tolerance,
    )

    # Water
    def kr_water(S):
        return relperm_table(
            oil_saturation=So,
            gas_saturation=Sg,
            water_saturation=S,
            residual_oil_saturation_water=Sorw,
            residual_oil_saturation_gas=Sorg,
            residual_gas_saturation=Sgr,
            irreducible_water_saturation=Swirr,
        )["water"]

    dλw_dSw = compute_phase_mobility_derivative(
        saturation=Sw,
        saturation_min=Sw_min,
        saturation_max=Sw_max,
        viscosity=mu_w,
        kr_func=kr_water,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
        phase_appearance_tolerance=phase_appearance_tolerance,
    )
    return dλo_dSo, dλg_dSg, dλw_dSw


def assemble_jacobian(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    time_step_size: float,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
    water_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_well_rate_grid: ThreeDimensionalGrid,
    gas_well_rate_grid: ThreeDimensionalGrid,
    water_well_rate_grid: ThreeDimensionalGrid,
    well_damping_factor: float = 0.01,
    dtype: np.typing.DTypeLike = np.float64,
    phase_appearance_tolerance: float = 1e-6,
) -> lil_matrix:
    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape
    interior_count_x = cell_count_x - 2
    interior_count_y = cell_count_y - 2
    interior_count_z = cell_count_z - 2
    total_unknowns = 3 * interior_count_x * interior_count_y * interior_count_z

    jacobian = lil_matrix((total_unknowns, total_unknowns), dtype=dtype)

    dx, dy = cell_dimension
    porosity_grid = rock_properties.porosity_grid

    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid
    water_compressibility_grid = fluid_properties.water_compressibility_grid

    _to_1d_index = functools.partial(
        to_1D_index_interior_only,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )

    # Neighbor offsets
    neighbour_offsets = [
        (1, 0, 0, "x"),  # East
        (-1, 0, 0, "x"),  # West
        (0, -1, 0, "y"),  # North
        (0, 1, 0, "y"),  # South
        (0, 0, -1, "z"),  # Top
        (0, 0, 1, "z"),  # Bottom
    ]

    # Main assembly loop
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_1d_index = _to_1d_index(i=i, j=j, k=k)
        if cell_1d_index < 0:
            continue

        thickness = thickness_grid[i, j, k]
        cell_volume = dx * dy * thickness
        porosity = porosity_grid[i, j, k]

        pressure = pressure_grid[i, j, k]
        oil_saturation = oil_saturation_grid[i, j, k]
        gas_saturation = gas_saturation_grid[i, j, k]
        water_saturation = water_saturation_grid[i, j, k]

        oil_compressibility = oil_compressibility_grid[i, j, k]
        gas_compressibility = gas_compressibility_grid[i, j, k]
        water_compressibility = water_compressibility_grid[i, j, k]

        # =====================================================================
        # DIAGONAL BLOCK: ∂R/∂x at current cell
        # =====================================================================
        diagonal_block = assmeble_accumulation_derivatives(
            porosity=porosity,
            cell_volume=cell_volume,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            water_saturation=water_saturation,
            oil_compressibility=oil_compressibility,
            gas_compressibility=gas_compressibility,
            water_compressibility=water_compressibility,
            dtype=dtype,
        )

        # Add flux derivatives
        for di, dj, dk, direction in neighbour_offsets:
            ni = i + di
            nj = j + dj
            nk = k + dk

            if not (
                0 <= ni < cell_count_x
                and 0 <= nj < cell_count_y
                and 0 <= nk < cell_count_z
            ):
                continue

            # Geometric factor and select correct mobility grid based on direction
            if direction == "x":
                geometric_factor = dy * thickness / dx
                oil_mobility_grid = oil_mobility_grid_x
                gas_mobility_grid = gas_mobility_grid_x
                water_mobility_grid = water_mobility_grid_x
            elif direction == "y":
                geometric_factor = dx * thickness / dy
                oil_mobility_grid = oil_mobility_grid_y
                gas_mobility_grid = gas_mobility_grid_y
                water_mobility_grid = water_mobility_grid_y
            else:  # z
                geometric_factor = dx * dy / thickness
                oil_mobility_grid = oil_mobility_grid_z
                gas_mobility_grid = gas_mobility_grid_z
                water_mobility_grid = water_mobility_grid_z

            # Compute flux derivatives for each phase
            oil_dF_dP = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=oil_mobility_grid,
                geometric_factor=geometric_factor,
            )

            gas_dF_dP = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=gas_mobility_grid,
                geometric_factor=geometric_factor,
            )

            water_dF_dP = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=water_mobility_grid,
                geometric_factor=geometric_factor,
            )

            # Add to diagonal block (residual = Acc - Δt·[Flux + Well])
            # ∂R/∂P includes -Δt·∂Flux/∂P
            diagonal_block[0, 0] += time_step_size * oil_dF_dP
            diagonal_block[1, 0] += time_step_size * gas_dF_dP
            diagonal_block[2, 0] += time_step_size * water_dF_dP

        # Add well derivatives
        oil_well_rate = oil_well_rate_grid[i, j, k]
        gas_well_rate = gas_well_rate_grid[i, j, k]
        water_well_rate = water_well_rate_grid[i, j, k]

        if abs(oil_well_rate) > 1e-10 and pressure > 1.0:
            oil_well_derivative = well_damping_factor * oil_well_rate / pressure
            diagonal_block[0, 0] -= time_step_size * oil_well_derivative

        if abs(gas_well_rate) > 1e-10 and pressure > 1.0:
            gas_well_derivative = well_damping_factor * gas_well_rate / pressure
            diagonal_block[1, 0] -= time_step_size * gas_well_derivative

        if abs(water_well_rate) > 1e-10 and pressure > 1.0:
            water_well_derivative = well_damping_factor * water_well_rate / pressure
            diagonal_block[2, 0] -= time_step_size * water_well_derivative

        # Assign diagonal block
        for row_offset in range(3):
            for column_offset in range(3):
                row_index = 3 * cell_1d_index + row_offset
                column_index = 3 * cell_1d_index + column_offset
                jacobian[row_index, column_index] = diagonal_block[
                    row_offset, column_offset
                ]

        # =====================================================================
        # OFF-DIAGONAL BLOCKS: ∂R/∂x at neighbour cells
        # =====================================================================
        for di, dj, dk, direction in neighbour_offsets:
            ni = i + di
            nj = j + dj
            nk = k + dk

            neighbour_1d_index = _to_1d_index(i=ni, j=nj, k=nk)
            if neighbour_1d_index < 0:
                continue

            off_diagonal_block = np.zeros((3, 3), dtype=dtype)

            # Geometric factor and select correct mobility grid based on direction
            if direction == "x":
                geometric_factor = dy * thickness / dx
                oil_mobility_grid = oil_mobility_grid_x
                gas_mobility_grid = gas_mobility_grid_x
                water_mobility_grid = water_mobility_grid_x
            elif direction == "y":
                geometric_factor = dx * thickness / dy
                oil_mobility_grid = oil_mobility_grid_y
                gas_mobility_grid = gas_mobility_grid_y
                water_mobility_grid = water_mobility_grid_y
            else:
                geometric_factor = dx * dy / thickness
                oil_mobility_grid = oil_mobility_grid_z
                gas_mobility_grid = gas_mobility_grid_z
                water_mobility_grid = water_mobility_grid_z

            # Compute derivatives w.r.t. neighbour pressure
            oil_dF_dP = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=oil_mobility_grid,
                geometric_factor=geometric_factor,
            )

            gas_dF_dP = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=gas_mobility_grid,
                geometric_factor=geometric_factor,
            )

            water_dF_dP = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=water_mobility_grid,
                geometric_factor=geometric_factor,
            )

            # Off-diagonal pressure derivatives
            off_diagonal_block[0, 0] = -time_step_size * oil_dF_dP
            off_diagonal_block[1, 0] = -time_step_size * gas_dF_dP
            off_diagonal_block[2, 0] = -time_step_size * water_dF_dP

            # Saturation derivatives are zero in quasi-Newton (mobility frozen)
            off_diagonal_block[0, 1] = 0.0
            off_diagonal_block[0, 2] = 0.0
            off_diagonal_block[1, 1] = 0.0
            off_diagonal_block[1, 2] = 0.0
            off_diagonal_block[2, 1] = 0.0
            off_diagonal_block[2, 2] = 0.0

            # Assign off-diagonal block
            for row_offset in range(3):
                for column_offset in range(3):
                    row_index = 3 * cell_1d_index + row_offset
                    column_index = 3 * neighbour_1d_index + column_offset
                    jacobian[row_index, column_index] = float(
                        off_diagonal_block[row_offset, column_offset]
                    )

    return jacobian
