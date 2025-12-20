import functools
import itertools
import typing

import numba
import numpy as np
from scipy.sparse import lil_matrix

from bores.diffusivity.base import to_1D_index_interior_only
from bores.models import FluidProperties, RockProperties
from bores.pvt.core import compute_harmonic_mean
from bores.types import ThreeDimensionalGrid, ThreeDimensions


@numba.njit(cache=True)
def compute_accumulation_derivatives(
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
    Compute all accumulation term derivatives.

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
    derivatives[0, 0] = pore_volume * oil_saturation * oil_compressibility
    derivatives[1, 0] = pore_volume * gas_saturation * gas_compressibility
    derivatives[2, 0] = pore_volume * water_saturation * water_compressibility

    # ∂R/∂S_o (column 1)
    derivatives[0, 1] = pore_volume
    derivatives[1, 1] = 0.0
    derivatives[2, 1] = -pore_volume

    # ∂R/∂S_g (column 2)
    derivatives[0, 2] = 0.0
    derivatives[1, 2] = pore_volume
    derivatives[2, 2] = -pore_volume
    return derivatives


@numba.njit(cache=True)
def compute_phase_flux_derivatives(
    cell_index: typing.Tuple[int, int, int],
    neighbour_index: typing.Tuple[int, int, int],
    mobility_grid: ThreeDimensionalGrid,
    geometric_factor: float,
) -> typing.Tuple[float, float]:
    """
    Compute flux derivatives for a single phase including gravity and capillary effects.

    Flux: F = T_α · (ΔP + ΔP_cap + ΔP_grav)
    where:
        T_α = λ̄_α · ρ̄_α · geometric_factor (transmissibility)
        ΔP = P_neighbour - P_current (pressure driving force)
        ΔP_cap = capillary_pressure_difference (for water: -ΔP_cow, for gas: +ΔP_cgo, for oil: 0)
        ΔP_grav = (ρ̄_α · g · Δz) / 144.0 (gravity driving force)

    Derivatives (mobility and density frozen):
    ∂F/∂P_current = -T_α
    ∂F/∂P_neighbour = +T_α

    Parameters:
    -----------
    elevation_difference : float
        elevation_grid[neighbour] - elevation_grid[current]
    capillary_pressure_difference : float
        For oil: 0.0 (reference phase)
        For gas: pcgo_grid[neighbour] - pcgo_grid[current]
        For water: -(pcow_grid[neighbour] - pcow_grid[current])

    Returns:
    --------
    (dF_dP_cell, dF_dP_neighbour)
    """
    harmonic_mobility = compute_harmonic_mean(
        mobility_grid[neighbour_index], mobility_grid[cell_index]
    )
    transmissibility = harmonic_mobility * geometric_factor

    dF_dP_cell = -transmissibility
    dF_dP_neighbour = transmissibility
    return dF_dP_cell, dF_dP_neighbour


def assemble_jacobian(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
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
    pcow_grid: ThreeDimensionalGrid,
    pcgo_grid: ThreeDimensionalGrid,
    time_step_size: float,
    acceleration_due_to_gravity_ft_per_s2: float,
    well_damping_factor: float = 0.01,
    dtype: np.typing.DTypeLike = np.float64,
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
        diagonal_block = compute_accumulation_derivatives(
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
            oil_dF_dP_current, _ = compute_phase_flux_derivatives(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=oil_mobility_grid,
                geometric_factor=geometric_factor,
            )

            gas_dF_dP_current, _ = compute_phase_flux_derivatives(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=gas_mobility_grid,
                geometric_factor=geometric_factor,
            )

            water_dF_dP_current, _ = compute_phase_flux_derivatives(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=water_mobility_grid,
                geometric_factor=geometric_factor,
            )

            # Add to diagonal block (residual = Acc - Δt·[Flux + Well])
            # ∂R/∂P includes -Δt·∂Flux/∂P
            diagonal_block[0, 0] -= time_step_size * oil_dF_dP_current
            diagonal_block[1, 0] -= time_step_size * gas_dF_dP_current
            diagonal_block[2, 0] -= time_step_size * water_dF_dP_current

        # Add well derivatives
        oil_well_rate = oil_well_rate_grid[i, j, k]
        gas_well_rate = gas_well_rate_grid[i, j, k]
        water_well_rate = water_well_rate_grid[i, j, k]

        if abs(oil_well_rate) > 1e-10 and pressure > 1.0:
            oil_well_derivative = -well_damping_factor * oil_well_rate / pressure
            diagonal_block[0, 0] -= time_step_size * oil_well_derivative

        if abs(gas_well_rate) > 1e-10 and pressure > 1.0:
            gas_well_derivative = -well_damping_factor * gas_well_rate / pressure
            diagonal_block[1, 0] -= time_step_size * gas_well_derivative

        if abs(water_well_rate) > 1e-10 and pressure > 1.0:
            water_well_derivative = -well_damping_factor * water_well_rate / pressure
            diagonal_block[2, 0] -= time_step_size * water_well_derivative

        # Assign diagonal block
        for row_offset in range(3):
            for column_offset in range(3):
                row_index = 3 * cell_1d_index + row_offset
                column_index = 3 * cell_1d_index + column_offset
                jacobian[row_index, column_index] = float(
                    diagonal_block[row_offset, column_offset]
                )

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
            _, oil_dF_dP_neighbour = compute_phase_flux_derivatives(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=oil_mobility_grid,
                geometric_factor=geometric_factor,
            )

            _, gas_dF_dP_neighbour = compute_phase_flux_derivatives(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=gas_mobility_grid,
                geometric_factor=geometric_factor,
            )

            _, water_dF_dP_neighbour = compute_phase_flux_derivatives(
                cell_index=(i, j, k),
                neighbour_index=(ni, nj, nk),
                mobility_grid=water_mobility_grid,
                geometric_factor=geometric_factor,
            )

            # Off-diagonal pressure derivatives
            off_diagonal_block[0, 0] = -time_step_size * oil_dF_dP_neighbour
            off_diagonal_block[1, 0] = -time_step_size * gas_dF_dP_neighbour
            off_diagonal_block[2, 0] = -time_step_size * water_dF_dP_neighbour

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
