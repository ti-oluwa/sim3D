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
def assemble_accumulation_derivatives(
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
    min_saturation: float,
    max_saturation: float,
    absolute_perturbation: float = 1e-8,
    relative_perturbation: float = 1e-6,
) -> float:
    """Compute saturation perturbation for numerical differentiation."""
    span = max(
        saturation - min_saturation,
        max_saturation - saturation,
    )
    return max(absolute_perturbation, relative_perturbation * span)


def compute_phase_mobility_derivative(
    saturation: float,
    min_saturation: float,
    max_saturation: float,
    viscosity: float,
    kr_func: typing.Callable[
        [float], float
    ],  # callable passed via numba-compatible closure
    absolute_perturbation: float,
    relative_perturbation: float,
    phase_appearance_tolerance: float = 1e-6,
) -> float:
    """
    Compute derivative of phase mobility w.r.t. its saturation.

    :param saturation: Phase saturation
    :param min_saturation: Minimum admissible saturation for the phase
    :param max_saturation: Maximum admissible saturation for the phase
    :param viscosity: Phase viscosity
    :param kr_func: Relative permeability function for the phase. Takes saturation as input and returns relative permeability.
    :param absolute_perturbation: Absolute perturbation for numerical differentiation
    :param relative_perturbation: Relative perturbation for numerical differentiation
    :param phase_appearance_tolerance: Tolerance to consider phase as present
    :return: Derivative of phase mobility w.r.t. saturation
    """
    # Phase not present → no mobility, no derivative
    if saturation <= min_saturation + phase_appearance_tolerance:
        return 0.0

    delta = compute_saturation_perturbation(
        saturation=saturation,
        min_saturation=min_saturation,
        max_saturation=max_saturation,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
    )

    # Decide differencing scheme
    can_forward = saturation + delta <= max_saturation
    can_backward = saturation - delta >= min_saturation

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
    """
    Compute derivatives of phase mobilities w.r.t. their saturations.

    :param cell_index: Index of the current cell (i, j, k)
    :param oil_saturation_grid: Oil saturation grid
    :param water_saturation_grid: Water saturation grid
    :param gas_saturation_grid: Gas saturation grid
    :param residual_oil_saturation_water_grid: Residual oil saturation (water) grid
    :param residual_oil_saturation_gas_grid: Residual oil saturation (gas) grid
    :param residual_gas_saturation_grid: Residual gas saturation grid
    :param irreducible_water_saturation_grid: Irreducible water saturation grid
    :param oil_viscosity_grid: Oil viscosity grid
    :param water_viscosity_grid: Water viscosity grid
    :param gas_viscosity_grid: Gas viscosity grid
    :param relperm_table: Relative permeability table
    :param absolute_perturbation: Absolute perturbation for numerical differentiation
    :param relative_perturbation: Relative perturbation for numerical differentiation
    :param phase_appearance_tolerance: Tolerance to consider phase as present
    :return: Tuple of (dλo/dSo, dλg/dSg, dλw/dSw)
    """
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
        min_saturation=So_min,
        max_saturation=So_max,
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
        min_saturation=Sg_min,
        max_saturation=Sg_max,
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
        min_saturation=Sw_min,
        max_saturation=Sw_max,
        viscosity=mu_w,
        kr_func=kr_water,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
        phase_appearance_tolerance=phase_appearance_tolerance,
    )
    return dλo_dSo, dλg_dSg, dλw_dSw


def compute_capillary_pressure_derivative(
    saturation: float,
    min_saturation: float,
    max_saturation: float,
    pc_func: typing.Callable[[float], float],
    absolute_perturbation: float,
    relative_perturbation: float,
    phase_appearance_tolerance: float = 1e-6,
) -> float:
    """
    Compute the derivative of the capillary pressure with respect to saturation.

    :param saturation: Phase saturation
    :param min_saturation: Minimum admissible saturation for the phase
    :param max_saturation: Maximum admissible saturation for the phase
    :param pc_func: Capillary pressure function. Takes saturation as input and returns capillary pressure.
    :param absolute_perturbation: Absolute perturbation for numerical differentiation
    :param relative_perturbation: Relative perturbation for numerical differentiation
    :param phase_appearance_tolerance: Tolerance to consider phase as present
    :return: Derivative of capillary pressure w.r.t. saturation
    """
    # Phase absent → no capillary gradient
    if saturation <= min_saturation + phase_appearance_tolerance:
        return 0.0

    delta = compute_saturation_perturbation(
        saturation=saturation,
        min_saturation=min_saturation,
        max_saturation=max_saturation,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
    )

    can_forward = saturation + delta <= max_saturation
    can_backward = saturation - delta >= min_saturation

    if can_forward and can_backward:
        pc_plus = pc_func(saturation + delta)
        pc_minus = pc_func(saturation - delta)
        return (pc_plus - pc_minus) / (2.0 * delta)

    elif can_forward:
        pc_plus = pc_func(saturation + delta)
        pc_0 = pc_func(saturation)
        return (pc_plus - pc_0) / delta

    elif can_backward:
        pc_0 = pc_func(saturation)
        pc_minus = pc_func(saturation - delta)
        return (pc_0 - pc_minus) / delta

    return 0.0


def compute_capillary_pressure_derivatives(
    cell_index: typing.Tuple[int, int, int],
    oil_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    residual_oil_saturation_water_grid: ThreeDimensionalGrid,
    residual_oil_saturation_gas_grid: ThreeDimensionalGrid,
    residual_gas_saturation_grid: ThreeDimensionalGrid,
    irreducible_water_saturation_grid: ThreeDimensionalGrid,
    capillary_pressure_table: CapillaryPressureTable,
    absolute_perturbation: float = 1e-8,
    relative_perturbation: float = 1e-6,
    phase_appearance_tolerance: float = 1e-6,
) -> typing.Tuple[float, float]:
    """
    Compute derivatives of capillary pressures w.r.t. saturations.

    :param cell_index: Index of the current cell (i, j, k)
    :param oil_saturation_grid: Oil saturation grid
    :param water_saturation_grid: Water saturation grid
    :param gas_saturation_grid: Gas saturation grid
    :param residual_oil_saturation_water_grid: Residual oil saturation (water) grid
    :param residual_oil_saturation_gas_grid: Residual oil saturation (gas) grid
    :param residual_gas_saturation_grid: Residual gas saturation grid
    :param irreducible_water_saturation_grid: Irreducible water saturation grid
    :param capillary_pressure_table: Capillary pressure table
    :param absolute_perturbation: Absolute perturbation for numerical differentiation
    :param relative_perturbation: Relative perturbation for numerical differentiation
    :param phase_appearance_tolerance: Tolerance to consider phase as present
    :return: Tuple of (dPcow/dSw, dPcgo/dSg)
    """
    i, j, k = cell_index

    Sw = water_saturation_grid[i, j, k]
    Sg = gas_saturation_grid[i, j, k]
    So = oil_saturation_grid[i, j, k]

    Sorw = residual_oil_saturation_water_grid[i, j, k]
    Sorg = residual_oil_saturation_gas_grid[i, j, k]
    Sgr = residual_gas_saturation_grid[i, j, k]
    Swirr = irreducible_water_saturation_grid[i, j, k]

    # Saturation bounds
    Sw_min = Swirr
    Sw_max = 1.0 - Sorw - Sgr

    Sg_min = Sgr
    Sg_max = 1.0 - Swirr - Sorg

    # Pcow(Sw)
    def pcow(S):
        return capillary_pressure_table(
            water_saturation=S,
            oil_saturation=So,
            gas_saturation=Sg,
            irreducible_water_saturation=Swirr,
            residual_oil_saturation_water=Sorw,
            residual_oil_saturation_gas=Sorg,
            residual_gas_saturation=Sgr,
        )["oil_water"]

    dPcow_dSw = compute_capillary_pressure_derivative(
        saturation=Sw,
        min_saturation=Sw_min,
        max_saturation=Sw_max,
        pc_func=pcow,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
        phase_appearance_tolerance=phase_appearance_tolerance,
    )

    # Pcgo(Sg)
    def pcgo(S):
        return capillary_pressure_table(
            water_saturation=Sw,
            oil_saturation=So,
            gas_saturation=S,
            irreducible_water_saturation=Swirr,
            residual_oil_saturation_water=Sorw,
            residual_oil_saturation_gas=Sorg,
            residual_gas_saturation=Sgr,
        )["gas_oil"]

    dPcgo_dSg = compute_capillary_pressure_derivative(
        saturation=Sg,
        min_saturation=Sg_min,
        max_saturation=Sg_max,
        pc_func=pcgo,
        absolute_perturbation=absolute_perturbation,
        relative_perturbation=relative_perturbation,
        phase_appearance_tolerance=phase_appearance_tolerance,
    )
    return dPcow_dSw, dPcgo_dSg


@numba.njit(cache=True)
def compute_harmonic_mean_derivative_wrt_first(
    value_first: float, value_second: float
) -> float:
    """
    Compute derivative of harmonic mean w.r.t. first value.

    ∂(harmonic_mean)/∂value_first = (value_second)² / (value_first + value_second)²
    """
    if value_first + value_second > 1e-20:
        return (value_second**2) / ((value_first + value_second) ** 2)
    return 0.0


@numba.njit(cache=True)
def compute_gravity_potential(
    harmonic_density: float, elevation_difference: float, gravity_ft_per_s2: float
) -> float:
    """Compute gravity potential: ρ·g·Δz / 144.0 (convert to psi)"""
    return (harmonic_density * gravity_ft_per_s2 * elevation_difference) / 144.0


@numba.njit(cache=True)
def compute_phase_potentials(
    pressure_difference: float,
    elevation_difference: float,
    pcow_difference: float,
    pcgo_difference: float,
    harmonic_oil_density: float,
    harmonic_water_density: float,
    harmonic_gas_density: float,
    gravity_ft_per_s2: float,
) -> typing.Tuple[float, float, float]:
    """
    Compute phase potentials for oil, water, and gas.

    :param pressure_difference: Pressure difference between the two cells
    :param elevation_difference: Elevation difference between the two cells
    :param pcow_difference: Capillary pressure difference for oil-water
    :param pcgo_difference: Capillary pressure difference for gas-oil
    :param harmonic_oil_density: Harmonic mean oil density between the two cells
    :param harmonic_water_density: Harmonic mean water density between the two cells
    :param harmonic_gas_density: Harmonic mean gas density between the two cells
    :param gravity_ft_per_s2: Gravitational acceleration in ft/s²
    :return: Tuple of (oil_phase_potential, water_phase_potential, gas_phase_potential)
    """
    oil_gravity_potential = compute_gravity_potential(
        harmonic_oil_density, elevation_difference, gravity_ft_per_s2
    )
    water_gravity_potential = compute_gravity_potential(
        harmonic_water_density, elevation_difference, gravity_ft_per_s2
    )
    gas_gravity_potential = compute_gravity_potential(
        harmonic_gas_density, elevation_difference, gravity_ft_per_s2
    )

    oil_phase_potential = pressure_difference + oil_gravity_potential
    water_phase_potential = (
        pressure_difference - pcow_difference + water_gravity_potential
    )
    gas_phase_potential = pressure_difference + pcgo_difference + gas_gravity_potential
    return oil_phase_potential, water_phase_potential, gas_phase_potential


def assemble_jacobian_with_frozen_mobility(
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
    """
    Assemble Jacobian with frozen mobilities.

    Only includes:
    1. Accumulation derivatives (w.r.t. P, So, Sg)
    2. Flux pressure derivatives (transmissibility)
    3. Well derivatives

    Saturation derivatives of flux terms are OMITTED (mobilities frozen).
    This is faster but less accurate than full Newton.
    """
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

        # DIAGONAL BLOCK: ∂R/∂x at current cell
        diagonal_block = assemble_accumulation_derivatives(
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

        # OFF-DIAGONAL BLOCKS: ∂R/∂x at neighbour cells
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


def assemble_jacobian(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
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
    pcow_grid: ThreeDimensionalGrid,
    pcgo_grid: ThreeDimensionalGrid,
    oil_well_rate_grid: ThreeDimensionalGrid,
    gas_well_rate_grid: ThreeDimensionalGrid,
    water_well_rate_grid: ThreeDimensionalGrid,
    acceleration_due_to_gravity_ft_per_s2: float,
    well_damping_factor: float = 0.01,
    dtype: np.typing.DTypeLike = np.float64,
    phase_appearance_tolerance: float = 1e-6,
    absolute_perturbation: float = 1e-8,
    relative_perturbation: float = 1e-6,
) -> lil_matrix:
    """
    Assemble full Newton Jacobian with all derivatives.

    Includes:
    1. Accumulation derivatives (w.r.t. P, So, Sg)
    2. Flux pressure derivatives (transmissibility)
    3. Flux saturation derivatives at CURRENT cell (mobility + capillary)
    4. Flux saturation derivatives at neighbour cells (mobility coupling)
    5. Well derivatives

    This is the most accurate formulation but requires careful damping.
    """
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
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid
    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    water_viscosity_grid = fluid_properties.water_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    residual_oil_saturation_water_grid = (
        rock_properties.residual_oil_saturation_water_grid
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )

    capillary_pressure_table = rock_fluid_properties.capillary_pressure_table
    relative_permeability_table = rock_fluid_properties.relative_permeability_table

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

        # DIAGONAL BLOCK: ∂R/∂x at current cell
        # Start with accumulation derivatives
        diagonal_block = assemble_accumulation_derivatives(
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
        # Compute mobility derivatives at current cell
        (dλo_dSo_cell, dλg_dSg_cell, dλw_dSw_cell) = compute_mobility_derivatives(
            cell_index=(i, j, k),
            oil_saturation_grid=oil_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            oil_viscosity_grid=oil_viscosity_grid,
            water_viscosity_grid=water_viscosity_grid,
            gas_viscosity_grid=gas_viscosity_grid,
            relperm_table=relative_permeability_table,
            absolute_perturbation=absolute_perturbation,
            relative_perturbation=relative_perturbation,
            phase_appearance_tolerance=phase_appearance_tolerance,
        )
        # Compute capillary pressure derivatives at current cell
        dPcow_dSw_cell, dPcgo_dSg_cell = compute_capillary_pressure_derivatives(
            cell_index=(i, j, k),
            oil_saturation_grid=oil_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            capillary_pressure_table=capillary_pressure_table,
            absolute_perturbation=absolute_perturbation,
            relative_perturbation=relative_perturbation,
            phase_appearance_tolerance=phase_appearance_tolerance,
        )

        # Add flux derivatives from all neighbours
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

            thickness_neighbour = thickness_grid[ni, nj, nk]

            # Geometric factor based on direction
            if direction == "x":
                thickness_harmonic = compute_harmonic_mean(
                    thickness, thickness_neighbour
                )
                geometric_factor = dy * thickness_harmonic / dx
                oil_mobility_grid = oil_mobility_grid_x
                gas_mobility_grid = gas_mobility_grid_x
                water_mobility_grid = water_mobility_grid_x
            elif direction == "y":
                thickness_harmonic = compute_harmonic_mean(
                    thickness, thickness_neighbour
                )
                geometric_factor = dx * thickness_harmonic / dy
                oil_mobility_grid = oil_mobility_grid_y
                gas_mobility_grid = gas_mobility_grid_y
                water_mobility_grid = water_mobility_grid_y
            else:  # z
                geometric_factor = dx * dy / thickness
                oil_mobility_grid = oil_mobility_grid_z
                gas_mobility_grid = gas_mobility_grid_z
                water_mobility_grid = water_mobility_grid_z

            # PRESSURE DERIVATIVES (Transmissibility)
            oil_transmissibility = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=oil_mobility_grid,
                geometric_factor=geometric_factor,
            )
            gas_transmissibility = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=gas_mobility_grid,
                geometric_factor=geometric_factor,
            )
            water_transmissibility = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=water_mobility_grid,
                geometric_factor=geometric_factor,
            )

            # Add pressure derivatives to diagonal
            diagonal_block[0, 0] += time_step_size * oil_transmissibility
            diagonal_block[1, 0] += time_step_size * gas_transmissibility
            diagonal_block[2, 0] += time_step_size * water_transmissibility

            # SATURATION DERIVATIVES AT CURRENT CELL
            # Get pressure and elevation differences
            pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
            elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]
            pcow_difference = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
            pcgo_difference = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]

            # Compute harmonic mean densities for gravity
            harmonic_oil_density = compute_harmonic_mean(
                oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
            )
            harmonic_water_density = compute_harmonic_mean(
                water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
            )
            harmonic_gas_density = compute_harmonic_mean(
                gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
            )

            # Compute phase potentials using helper function
            oil_phase_potential, water_phase_potential, gas_phase_potential = (
                compute_phase_potentials(
                    pressure_difference=pressure_difference,
                    elevation_difference=elevation_difference,
                    pcow_difference=pcow_difference,
                    pcgo_difference=pcgo_difference,
                    harmonic_oil_density=harmonic_oil_density,
                    harmonic_water_density=harmonic_water_density,
                    harmonic_gas_density=harmonic_gas_density,
                    gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                )
            )

            # Get current cell mobilities
            oil_mobility_cell = oil_mobility_grid[i, j, k]
            gas_mobility_cell = gas_mobility_grid[i, j, k]
            water_mobility_cell = water_mobility_grid[i, j, k]

            oil_mobility_neighbour = oil_mobility_grid[ni, nj, nk]
            gas_mobility_neighbour = gas_mobility_grid[ni, nj, nk]
            water_mobility_neighbour = water_mobility_grid[ni, nj, nk]

            # Compute harmonic mean derivatives for current cell
            dλo_harmonic_d_cell = compute_harmonic_mean_derivative_wrt_first(
                oil_mobility_cell, oil_mobility_neighbour
            )
            dλg_harmonic_d_cell = compute_harmonic_mean_derivative_wrt_first(
                gas_mobility_cell, gas_mobility_neighbour
            )
            dλw_harmonic_d_cell = compute_harmonic_mean_derivative_wrt_first(
                water_mobility_cell, water_mobility_neighbour
            )

            #  ∂R_oil/∂So (current cell)
            # F_o = T_o * pot_o = (λ_o_harmonic * geom) * pot_o
            # ∂F_o/∂So_cell = geom * pot_o * ∂λ_harmonic_o/∂λ_cell * ∂λ_cell/∂So
            dFo_dSo = (
                geometric_factor
                * oil_phase_potential
                * dλo_harmonic_d_cell
                * dλo_dSo_cell
            )
            diagonal_block[0, 1] -= time_step_size * dFo_dSo

            # ∂R_water/∂So (current cell)
            # Water flux affected through:
            # 1. Capillary pressure: ∂pot_w/∂So = -∂Pcow/∂So = -∂Pcow/∂Sw * ∂Sw/∂So = -(-dPcow_dSw_cell)*(-1) = -dPcow_dSw_cell
            # 2. Harmonic mobility: ∂λ_harmonic_w/∂Sw * ∂Sw/∂So = -∂λ_harmonic_w/∂Sw
            dFw_dSo_capillary = -water_transmissibility * dPcow_dSw_cell
            dFw_dSo_mobility = (
                -geometric_factor
                * water_phase_potential
                * dλw_harmonic_d_cell
                * dλw_dSw_cell
            )
            dFw_dSo = dFw_dSo_capillary + dFw_dSo_mobility
            diagonal_block[2, 1] -= time_step_size * dFw_dSo

            # ∂R_gas/∂Sg (current cell)
            # Gas flux affected through:
            # 1. Mobility derivative: ∂λ_harmonic_g/∂Sg
            # 2. Capillary pressure: ∂pot_g/∂Sg = ∂Pcgo/∂Sg = dPcgo_dSg_cell
            dFg_dSg_capillary = gas_transmissibility * dPcgo_dSg_cell
            dFg_dSg_mobility = (
                geometric_factor
                * gas_phase_potential
                * dλg_harmonic_d_cell
                * dλg_dSg_cell
            )
            dFg_dSg = dFg_dSg_mobility + dFg_dSg_capillary
            diagonal_block[1, 2] -= time_step_size * dFg_dSg

            # ∂R_water/∂Sg (current cell)
            # Water flux affected through:
            # 1. Capillary pressure: ∂pot_w/∂Sg = -∂Pcow/∂Sg = -∂Pcow/∂Sw * ∂Sw/∂Sg = -(-dPcow_dSw_cell)*(-1) = -dPcow_dSw_cell
            # 2. Harmonic mobility: ∂λ_harmonic_w/∂Sw * ∂Sw/∂Sg = -∂λ_harmonic_w/∂Sw
            dFw_dSg_capillary = -water_transmissibility * dPcow_dSw_cell
            dFw_dSg_mobility = (
                -geometric_factor
                * water_phase_potential
                * dλw_harmonic_d_cell
                * dλw_dSw_cell
            )
            dFw_dSg = dFw_dSg_capillary + dFw_dSg_mobility
            diagonal_block[2, 2] -= time_step_size * dFw_dSg

        # Add well derivatives
        oil_well_rate = oil_well_rate_grid[i, j, k]
        gas_well_rate = gas_well_rate_grid[i, j, k]
        water_well_rate = water_well_rate_grid[i, j, k]

        if abs(oil_well_rate) > 1e-10 and pressure > 1.0:
            diagonal_block[0, 0] -= (
                time_step_size * well_damping_factor * oil_well_rate / pressure
            )

        if abs(gas_well_rate) > 1e-10 and pressure > 1.0:
            diagonal_block[1, 0] -= (
                time_step_size * well_damping_factor * gas_well_rate / pressure
            )

        if abs(water_well_rate) > 1e-10 and pressure > 1.0:
            diagonal_block[2, 0] -= (
                time_step_size * well_damping_factor * water_well_rate / pressure
            )

        # Assign diagonal block
        for row_offset in range(3):
            for column_offset in range(3):
                row_index = 3 * cell_1d_index + row_offset
                column_index = 3 * cell_1d_index + column_offset
                jacobian[row_index, column_index] = diagonal_block[
                    row_offset, column_offset
                ]

        # OFF-DIAGONAL BLOCKS: ∂R_current_cell/∂x_neighbour_cell
        for di, dj, dk, direction in neighbour_offsets:
            ni = i + di
            nj = j + dj
            nk = k + dk

            neighbour_1d_index = _to_1d_index(i=ni, j=nj, k=nk)
            if neighbour_1d_index < 0:
                continue

            off_diagonal_block = np.zeros((3, 3), dtype=dtype)

            thickness_neighbour = thickness_grid[ni, nj, nk]

            # Geometric factor
            if direction == "x":
                thickness_harmonic = compute_harmonic_mean(
                    thickness, thickness_neighbour
                )
                geometric_factor = dy * thickness_harmonic / dx
                oil_mobility_grid = oil_mobility_grid_x
                gas_mobility_grid = gas_mobility_grid_x
                water_mobility_grid = water_mobility_grid_x
            elif direction == "y":
                thickness_harmonic = compute_harmonic_mean(
                    thickness, thickness_neighbour
                )
                geometric_factor = dx * thickness_harmonic / dy
                oil_mobility_grid = oil_mobility_grid_y
                gas_mobility_grid = gas_mobility_grid_y
                water_mobility_grid = water_mobility_grid_y
            else:
                geometric_factor = dx * dy / thickness
                oil_mobility_grid = oil_mobility_grid_z
                gas_mobility_grid = gas_mobility_grid_z
                water_mobility_grid = water_mobility_grid_z

            # Transmissibilities
            oil_transmissibility = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=oil_mobility_grid,
                geometric_factor=geometric_factor,
            )
            gas_transmissibility = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=gas_mobility_grid,
                geometric_factor=geometric_factor,
            )
            water_transmissibility = compute_phase_flux_derivative(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=water_mobility_grid,
                geometric_factor=geometric_factor,
            )

            # PRESSURE DERIVATIVES (w.r.t. neighbour pressure)
            off_diagonal_block[0, 0] = -time_step_size * oil_transmissibility
            off_diagonal_block[1, 0] = -time_step_size * gas_transmissibility
            off_diagonal_block[2, 0] = -time_step_size * water_transmissibility

            # SATURATION DERIVATIVES (w.r.t. neighbour saturations)
            # Compute mobility derivatives at neighbour cell
            dλo_dSo_neighbour, dλg_dSg_neighbour, dλw_dSw_neighbour = (
                compute_mobility_derivatives(
                    cell_index=(ni, nj, nk),
                    oil_saturation_grid=oil_saturation_grid,
                    water_saturation_grid=water_saturation_grid,
                    gas_saturation_grid=gas_saturation_grid,
                    residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
                    residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
                    residual_gas_saturation_grid=residual_gas_saturation_grid,
                    irreducible_water_saturation_grid=irreducible_water_saturation_grid,
                    oil_viscosity_grid=oil_viscosity_grid,
                    water_viscosity_grid=water_viscosity_grid,
                    gas_viscosity_grid=gas_viscosity_grid,
                    relperm_table=relative_permeability_table,
                    absolute_perturbation=absolute_perturbation,
                    relative_perturbation=relative_perturbation,
                    phase_appearance_tolerance=phase_appearance_tolerance,
                )
            )
            # Compute capillary pressure derivatives at neighbour cell
            dPcow_dSw_neighbour, dPcgo_dSg_neighbour = (
                compute_capillary_pressure_derivatives(
                    cell_index=(ni, nj, nk),
                    oil_saturation_grid=oil_saturation_grid,
                    water_saturation_grid=water_saturation_grid,
                    gas_saturation_grid=gas_saturation_grid,
                    residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
                    residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
                    residual_gas_saturation_grid=residual_gas_saturation_grid,
                    irreducible_water_saturation_grid=irreducible_water_saturation_grid,
                    capillary_pressure_table=capillary_pressure_table,
                    absolute_perturbation=absolute_perturbation,
                    relative_perturbation=relative_perturbation,
                    phase_appearance_tolerance=phase_appearance_tolerance,
                )
            )

            # Get mobilities for harmonic mean derivatives
            oil_mobility_cell = oil_mobility_grid[i, j, k]
            gas_mobility_cell = gas_mobility_grid[i, j, k]
            water_mobility_cell = water_mobility_grid[i, j, k]

            oil_mobility_neighbour = oil_mobility_grid[ni, nj, nk]
            gas_mobility_neighbour = gas_mobility_grid[ni, nj, nk]
            water_mobility_neighbour = water_mobility_grid[ni, nj, nk]

            # Get potentials
            pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
            elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]
            pcow_difference = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
            pcgo_difference = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]

            # Compute harmonic mean densities for gravity
            harmonic_oil_density = compute_harmonic_mean(
                oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
            )
            harmonic_water_density = compute_harmonic_mean(
                water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
            )
            harmonic_gas_density = compute_harmonic_mean(
                gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
            )

            # Compute phase potentials using helper function
            oil_phase_potential, water_phase_potential, gas_phase_potential = (
                compute_phase_potentials(
                    pressure_difference=pressure_difference,
                    elevation_difference=elevation_difference,
                    pcow_difference=pcow_difference,
                    pcgo_difference=pcgo_difference,
                    harmonic_oil_density=harmonic_oil_density,
                    harmonic_water_density=harmonic_water_density,
                    harmonic_gas_density=harmonic_gas_density,
                    gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                )
            )

            # ∂R_oil/∂So_neighbour (column 1)
            # F_o depends on λ_o_neighbour through harmonic mean
            dFo_dSo_neighbour = (
                geometric_factor
                * oil_phase_potential
                * compute_harmonic_mean_derivative_wrt_first(
                    oil_mobility_neighbour, oil_mobility_cell
                )
                * dλo_dSo_neighbour
            )
            off_diagonal_block[0, 1] = -time_step_size * dFo_dSo_neighbour

            # ∂R_water/∂So_neighbour (column 1)
            # Water flux affected by:
            # 1. Capillary at neighbour: ∂pot_w/∂So_neighbour = ∂Pcow_neighbour/∂Sw_neighbour * ∂Sw/∂So = dPcow_dSw_neighbour * (-1)
            # 2. Mobility at neighbour: ∂λ_w_neighbour/∂Sw_neighbour * ∂Sw/∂So = -dλw_dSw_neighbour
            dFw_dSo_neighbour_capillary = -water_transmissibility * dPcow_dSw_neighbour
            dFw_dSo_neighbour_mobility = (
                -geometric_factor
                * water_phase_potential
                * compute_harmonic_mean_derivative_wrt_first(
                    water_mobility_neighbour, water_mobility_cell
                )
                * dλw_dSw_neighbour
            )
            dFw_dSo_neighbour = dFw_dSo_neighbour_capillary + dFw_dSo_neighbour_mobility
            off_diagonal_block[2, 1] = -time_step_size * dFw_dSo_neighbour

            # ∂R_gas/∂Sg_neighbour (column 2)
            # Gas flux affected by:
            # 1. Mobility at neighbour
            # 2. Capillary pressure at neighbour
            dFg_dSg_neighbour_mobility = (
                geometric_factor
                * gas_phase_potential
                * compute_harmonic_mean_derivative_wrt_first(
                    gas_mobility_neighbour, gas_mobility_cell
                )
                * dλg_dSg_neighbour
            )
            dFg_dSg_neighbour_capillary = gas_transmissibility * dPcgo_dSg_neighbour
            dFg_dSg_neighbour = dFg_dSg_neighbour_mobility + dFg_dSg_neighbour_capillary
            off_diagonal_block[1, 2] = -time_step_size * dFg_dSg_neighbour

            # ∂R_water/∂Sg_neighbour (column 2)
            # Water flux affected through Sw_neighbour coupling
            dFw_dSg_neighbour_capillary = -water_transmissibility * dPcow_dSw_neighbour
            dFw_dSg_neighbour_mobility = (
                -geometric_factor
                * water_phase_potential
                * compute_harmonic_mean_derivative_wrt_first(
                    water_mobility_neighbour, water_mobility_cell
                )
                * dλw_dSw_neighbour
            )
            dFw_dSg_neighbour = dFw_dSg_neighbour_capillary + dFw_dSg_neighbour_mobility
            off_diagonal_block[2, 2] = -time_step_size * dFw_dSg_neighbour

            # Cross-coupling terms (explicitly zero)
            off_diagonal_block[1, 1] = 0.0  # ∂R_gas/∂So_neighbour
            off_diagonal_block[0, 2] = 0.0  # ∂R_oil/∂Sg_neighbour

            # Assign off-diagonal block
            for row_offset in range(3):
                for column_offset in range(3):
                    row_index = 3 * cell_1d_index + row_offset
                    column_index = 3 * neighbour_1d_index + column_offset
                    jacobian[row_index, column_index] = float(
                        off_diagonal_block[row_offset, column_offset]
                    )

    return jacobian
