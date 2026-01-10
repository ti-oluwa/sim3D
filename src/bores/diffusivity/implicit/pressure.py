import functools
import itertools
import logging
import typing

import numba
import numpy as np
from scipy.sparse import lil_matrix

from bores._precision import get_dtype
from bores.config import Config
from bores.constants import c
from bores.diffusivity.base import (
    EvolutionResult,
    _warn_injection_pressure_is_low,
    _warn_production_pressure_is_high,
    compute_mobility_grids,
    solve_linear_system,
    to_1D_index_interior_only,
)
from bores.errors import SolverError, PreconditionerError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.pvt import build_total_fluid_compressibility_grid
from bores.models import FluidProperties, RockProperties
from bores.pvt.core import compute_harmonic_mean
from bores.types import (
    FluidPhase,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.wells import Wells

logger = logging.getLogger(__name__)


def evolve_pressure_implicitly(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
) -> EvolutionResult[ThreeDimensionalGrid, None]:
    """
    Solves the fully implicit finite-difference pressure equation for a slightly compressible,
    three-phase flow system in a 3D reservoir.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet
    :param thickness_grid: 3D grid of cell thicknesses in feet
    :param elevation_grid: 3D grid of cell elevations in feet
    :param time_step: Current time step number (for logging/debugging)
    :param time_step_size: Time step size in seconds
    :param rock_properties: `RockProperties` object containing model rock properties
    :param fluid_properties: `FluidProperties` object containing model fluid properties
    :param relative_mobility_grids: Tuple of relative mobility grids for (water, oil, gas)
    :param capillary_pressure_grids: Tuple of capillary pressure grids for (oil-water, gas-oil)
    :param wells: `Wells` object containing well definitions and properties
    :param config: `Config` object containing simulation config
    :return: `EvolutionResult` containing the new pressure grid and scheme used
    """
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Determine grid dimensions and cell sizes
    cell_count_x, cell_count_y, cell_count_z = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Compute total fluid system compressibility for each cell
    total_fluid_compressibility_grid = build_total_fluid_compressibility_grid(
        oil_saturation_grid=current_oil_saturation_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        water_saturation_grid=current_water_saturation_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_saturation_grid=current_gas_saturation_grid,
        gas_compressibility_grid=gas_compressibility_grid,
    )
    total_compressibility_grid = total_fluid_compressibility_grid + rock_compressibility
    # Clamp the compressibility within range
    total_compressibility_grid = config.total_compressibility_range.clip(
        total_compressibility_grid
    )

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    # Compute mobility grids for x, y, z directions using njitted function
    x_mobilities, y_mobilities, z_mobilities = compute_mobility_grids(
        absolute_permeability_x=absolute_permeability.x,
        absolute_permeability_y=absolute_permeability.y,
        absolute_permeability_z=absolute_permeability.z,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        md_per_cp_to_ft2_per_psi_per_day=c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY,
    )

    # Unpack mobility grids by direction
    water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x = x_mobilities
    water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y = y_mobilities
    water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z = z_mobilities

    time_step_size_in_days = time_step_size * c.DAYS_PER_SECOND

    # Initialize sparse coefficient matrix and RHS vector
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    dtype = get_dtype()
    A = lil_matrix((interior_cell_count, interior_cell_count), dtype=dtype)
    b = np.zeros(interior_cell_count, dtype=dtype, order="C")

    # FIRST PASS: Initialize accumulation terms for all cells
    add_accumulation_terms(
        A=A,
        b=b,
        porosity_grid=porosity_grid,
        total_compressibility_grid=total_compressibility_grid,
        thickness_grid=thickness_grid,
        current_oil_pressure_grid=current_oil_pressure_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step_size_in_days=time_step_size_in_days,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        dtype=dtype,
    )
    # SECOND PASS: Add face transmissibilities and fluxes
    add_face_transmissibilities_and_fluxes(
        A=A,
        b=b,
        thickness_grid=thickness_grid,
        water_mobility_grid_x=water_mobility_grid_x,
        oil_mobility_grid_x=oil_mobility_grid_x,
        gas_mobility_grid_x=gas_mobility_grid_x,
        water_mobility_grid_y=water_mobility_grid_y,
        oil_mobility_grid_y=oil_mobility_grid_y,
        gas_mobility_grid_y=gas_mobility_grid_y,
        water_mobility_grid_z=water_mobility_grid_z,
        oil_mobility_grid_z=oil_mobility_grid_z,
        gas_mobility_grid_z=gas_mobility_grid_z,
        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
        oil_density_grid=oil_density_grid,
        water_density_grid=water_density_grid,
        gas_density_grid=gas_density_grid,
        elevation_grid=elevation_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        acceleration_due_to_gravity_ft_per_s2=c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2,
        dtype=dtype,
    )
    # THIRD PASS: Compute well flow rates and add to RHS using extracted function
    add_well_contributions(
        A=A,
        b=b,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        current_oil_pressure_grid=current_oil_pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        absolute_permeability=absolute_permeability,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        water_compressibility_grid=water_compressibility_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        fluid_properties=fluid_properties,
        wells=wells,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step=time_step,
        time_step_size=time_step_size,
        config=config,
        md_per_cp_to_ft2_per_psi_per_day=c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY,
    )

    # Solve the linear system A·pⁿ⁺¹ = b
    try:
        new_1D_pressure_grid, _ = solve_linear_system(
            A_csr=A.tocsr(),
            b=b,
            max_iterations=config.max_iterations,
            solver=config.iterative_solver,
            preconditioner=config.preconditioner,
            fallback_to_direct=True,
        )
    except (SolverError, PreconditionerError) as exc:
        logger.error(f"Pressure solve failed at time step {time_step}: {exc}")
        return EvolutionResult(
            value=current_oil_pressure_grid.astype(dtype, copy=False),
            success=False,
            scheme="implicit",
            message=str(exc),
        )

    # Map solution back to 3D grid
    new_pressure_grid = map_1D_solution_to_grid(
        solution_1D=new_1D_pressure_grid,
        current_grid=current_oil_pressure_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )
    return EvolutionResult(
        value=new_pressure_grid.astype(dtype, copy=False),
        success=True,
        scheme="implicit",
        message=f"Implicit pressure evolution for time step {time_step} successful.",
    )


@numba.njit(parallel=True, cache=True)
def map_1D_solution_to_grid(
    solution_1D: OneDimensionalGrid,
    current_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> ThreeDimensionalGrid:
    """
    Map the 1D solution array back to a 3D grid, preserving boundary values.

    This function takes the solution from the linear system solver (which only contains
    interior cell values) and fills them into a 3D grid while preserving the boundary
    values from the current grid.

    :param solution_1D: 1D array containing solution for interior cells only
    :param current_grid: Current 3D grid (used to preserve boundary values)
    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :return: 3D grid with solution mapped to interior cells and boundaries preserved
    """
    # Initialize with current grid (preserves boundary values)
    new_grid = current_grid.copy()

    # Fill interior cells with solution using parallel processing
    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                # Convert 3D indices to 1D index for solution array
                idx = to_1D_index_interior_only(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                new_grid[i, j, k] = solution_1D[idx]
    return new_grid


@numba.njit(parallel=True, cache=True)
def compute_accumulation_arrays(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    current_oil_pressure_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_size_in_days: float,
    dtype: np.typing.DTypeLike,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Compute accumulation terms for all interior cells and return as dense arrays.

    This function computes the diagonal coefficients and RHS values that will be used
    to initialize the sparse matrix A and vector b. The accumulation term represents
    the storage capacity of each cell: accumulation_coefficient = (φ * c_t * V) / Δt

    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param porosity_grid: Cell porosity grid (fraction)
    :param total_compressibility_grid: Total compressibility grid (1/psi)
    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step_size_in_days: Time step size (days)
    :param dtype: Data type for arrays (np.float32 or np.float64)
    :return: Tuple of (diagonal_values, rhs_values) both 1D arrays of length `interior_cell_count`
    """
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    diagonal_values = np.zeros(interior_cell_count, dtype=dtype)
    rhs_values = np.zeros(interior_cell_count, dtype=dtype)

    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_1D_index = to_1D_index_interior_only(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )

                cell_thickness = thickness_grid[i, j, k]
                cell_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_total_compressibility = total_compressibility_grid[i, j, k]
                cell_oil_pressure = current_oil_pressure_grid[i, j, k]

                # Accumulation term coefficient
                accumulation_coefficient = (
                    cell_porosity * cell_total_compressibility * cell_volume
                ) / time_step_size_in_days

                diagonal_values[cell_1D_index] = accumulation_coefficient
                rhs_values[cell_1D_index] = accumulation_coefficient * cell_oil_pressure

    return diagonal_values, rhs_values


def add_accumulation_terms(
    A: lil_matrix,
    b: np.ndarray,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    current_oil_pressure_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_size_in_days: float,
    dtype: np.typing.DTypeLike,
) -> typing.Tuple[lil_matrix, np.ndarray]:
    """
    Initialize accumulation terms then populate sparse matrix.

    :param A: Sparse coefficient matrix (lil_matrix format) to modify in place
    :param b: RHS vector to modify in place
    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param porosity_grid: Cell porosity grid (fraction)
    :param total_compressibility_grid: Total compressibility grid (1/psi)
    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step_size_in_days: Time step size (days)
    :param dtype: Data type for arrays (np.float32 or np.float64)
    :return: Tuple of (A, b) with accumulation terms initialized (same objects passed in, modified in place)
    """
    diagonal_values, rhs_values = compute_accumulation_arrays(
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        porosity_grid=porosity_grid,
        total_compressibility_grid=total_compressibility_grid,
        current_oil_pressure_grid=current_oil_pressure_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step_size_in_days=time_step_size_in_days,
        dtype=dtype,
    )

    # Populate sparse matrix
    A.setdiag(diagonal_values)
    b[:] = rhs_values
    return A, b


# NOTE: Do not usage parallel=True here. Prone to race conditions.
@numba.njit(cache=True)
def compute_face_transmissibility_arrays(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    acceleration_due_to_gravity_ft_per_s2: float,
    dtype: np.typing.DTypeLike,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute face transmissibility arrays for all interior cell faces.

    This function processes each interior cell and computes transmissibilities
    for faces in the positive x, y, and z directions only (to avoid double-counting).
    Each face connects two cells, so we add symmetric entries to the sparse matrix.

    The resulting arrays can be used to construct the sparse matrix A and update
    the diagonal and RHS vectors.

    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param water_mobility_grid_x: Water mobility grid for x-direction faces (1/(cp*ft))
    :param oil_mobility_grid_x: Oil mobility grid for x-direction faces (1/(cp*ft))
    :param gas_mobility_grid_x: Gas mobility grid for x-direction faces (1/(cp*ft))
    :param water_mobility_grid_y: Water mobility grid for y-direction faces (1/(cp*ft))
    :param oil_mobility_grid_y: Oil mobility grid for y-direction faces (1/(cp*ft))
    :param gas_mobility_grid_y: Gas mobility grid for y-direction faces (1/(cp*ft))
    :param water_mobility_grid_z: Water mobility grid for z-direction faces (1/(cp*ft))
    :param oil_mobility_grid_z: Oil mobility grid for z-direction faces (1/(cp*ft))
    :param gas_mobility_grid_z: Gas mobility grid for z-direction faces (1/(cp*ft))
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure grid (psi)
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure grid (psi)
    :param oil_density_grid: Oil density grid (lb/ft^3)
    :param water_density_grid: Water density grid (lb/ft^3)
    :param gas_density_grid: Gas density grid (lb/ft^3)
    :param elevation_grid: Cell elevation grid (ft)
    :param acceleration_due_to_gravity_ft_per_s2: Acceleration due to gravity (ft/s^2)
    :param dtype: Data type for arrays (np.float32 or np.float64)
    :return: Tuple of (rows, cols, off_diag_values, diagonal_additions, rhs_additions)
        - rows: Row indices for off-diagonal entries
        - cols: Column indices for off-diagonal entries
        - off_diag_values: Values for off-diagonal entries (-T_face)
        - diagonal_additions: Array to add to diagonal (indexed by cell_1D_index)
        - rhs_additions: Array to add to RHS (indexed by cell_1D_index)
    """
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)

    # Maximum number of faces: 3 directions per interior cell
    max_faces = interior_cell_count * 3

    # Arrays for off-diagonal entries (each face creates 2 entries: A[i,j] and A[j,i])
    rows = np.zeros(max_faces * 2, dtype=np.int32)
    cols = np.zeros(max_faces * 2, dtype=np.int32)
    off_diag_values = np.zeros(max_faces * 2, dtype=dtype)

    # Diagonal and RHS additions (indexed by cell_1D_index)
    diagonal_additions = np.zeros(interior_cell_count, dtype=dtype)
    rhs_additions = np.zeros(interior_cell_count, dtype=dtype)

    # Counter for entries
    entry_idx = 0

    for i in range(1, cell_count_x - 1):  # Regular loop, not prange
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_1D_index = to_1D_index_interior_only(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                cell_thickness = thickness_grid[i, j, k]

                # X-DIRECTION: Process East face (i+1, j, k) only to avoid double-counting
                ni, nj, nk = i + 1, j, k
                if ni < cell_count_x - 1:  # Check neighbor is interior
                    neighbour_1D_index = to_1D_index_interior_only(
                        i=ni,
                        j=nj,
                        k=nk,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    geometric_factor = cell_size_y * cell_thickness / cell_size_x

                    harmonic_mobility, cap_flux, grav_flux = (
                        compute_pseudo_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(ni, nj, nk),
                            water_mobility_grid=water_mobility_grid_x,
                            oil_mobility_grid=oil_mobility_grid_x,
                            gas_mobility_grid=gas_mobility_grid_x,
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                        )
                    )

                    T_face = harmonic_mobility * geometric_factor
                    rhs_face_term = (cap_flux + grav_flux) * geometric_factor

                    # Store off-diagonal entries (symmetric)
                    rows[entry_idx] = cell_1D_index
                    cols[entry_idx] = neighbour_1D_index
                    off_diag_values[entry_idx] = -T_face

                    rows[entry_idx + 1] = neighbour_1D_index
                    cols[entry_idx + 1] = cell_1D_index
                    off_diag_values[entry_idx + 1] = -T_face

                    entry_idx += 2

                    # Update diagonals (both cells)
                    diagonal_additions[cell_1D_index] += T_face
                    diagonal_additions[neighbour_1D_index] += T_face

                    # Update RHS (opposite signs)
                    rhs_additions[cell_1D_index] += rhs_face_term
                    rhs_additions[neighbour_1D_index] -= rhs_face_term

                # Y-DIRECTION: Process South face (i, j+1, k) only
                ni, nj, nk = i, j + 1, k
                if nj < cell_count_y - 1:
                    neighbour_1D_index = to_1D_index_interior_only(
                        i=ni,
                        j=nj,
                        k=nk,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    geometric_factor = cell_size_x * cell_thickness / cell_size_y

                    harmonic_mobility, cap_flux, grav_flux = (
                        compute_pseudo_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(ni, nj, nk),
                            water_mobility_grid=water_mobility_grid_y,
                            oil_mobility_grid=oil_mobility_grid_y,
                            gas_mobility_grid=gas_mobility_grid_y,
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                        )
                    )

                    T_face = harmonic_mobility * geometric_factor
                    rhs_face_term = (cap_flux + grav_flux) * geometric_factor

                    rows[entry_idx] = cell_1D_index
                    cols[entry_idx] = neighbour_1D_index
                    off_diag_values[entry_idx] = -T_face

                    rows[entry_idx + 1] = neighbour_1D_index
                    cols[entry_idx + 1] = cell_1D_index
                    off_diag_values[entry_idx + 1] = -T_face

                    entry_idx += 2

                    diagonal_additions[cell_1D_index] += T_face
                    diagonal_additions[neighbour_1D_index] += T_face
                    rhs_additions[cell_1D_index] += rhs_face_term
                    rhs_additions[neighbour_1D_index] -= rhs_face_term

                # Z-DIRECTION: Process Bottom face (i, j, k+1) only
                ni, nj, nk = i, j, k + 1
                if nk < cell_count_z - 1:
                    neighbour_1D_index = to_1D_index_interior_only(
                        i=ni,
                        j=nj,
                        k=nk,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    geometric_factor = cell_size_x * cell_size_y / cell_thickness

                    harmonic_mobility, cap_flux, grav_flux = (
                        compute_pseudo_fluxes_from_neighbour(
                            cell_indices=(i, j, k),
                            neighbour_indices=(ni, nj, nk),
                            water_mobility_grid=water_mobility_grid_z,
                            oil_mobility_grid=oil_mobility_grid_z,
                            gas_mobility_grid=gas_mobility_grid_z,
                            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                            oil_density_grid=oil_density_grid,
                            water_density_grid=water_density_grid,
                            gas_density_grid=gas_density_grid,
                            elevation_grid=elevation_grid,
                            acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                        )
                    )

                    T_face = harmonic_mobility * geometric_factor
                    rhs_face_term = (cap_flux + grav_flux) * geometric_factor

                    rows[entry_idx] = cell_1D_index
                    cols[entry_idx] = neighbour_1D_index
                    off_diag_values[entry_idx] = -T_face

                    rows[entry_idx + 1] = neighbour_1D_index
                    cols[entry_idx + 1] = cell_1D_index
                    off_diag_values[entry_idx + 1] = -T_face

                    entry_idx += 2

                    diagonal_additions[cell_1D_index] += T_face
                    diagonal_additions[neighbour_1D_index] += T_face
                    rhs_additions[cell_1D_index] += rhs_face_term
                    rhs_additions[neighbour_1D_index] -= rhs_face_term

    # Trim arrays to actual size
    rows = rows[:entry_idx]
    cols = cols[:entry_idx]
    off_diag_values = off_diag_values[:entry_idx]

    return rows, cols, off_diag_values, diagonal_additions, rhs_additions


def add_face_transmissibilities_and_fluxes(
    A: lil_matrix,
    b: np.ndarray,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    acceleration_due_to_gravity_ft_per_s2: float,
    dtype: np.typing.DTypeLike,
) -> typing.Tuple[lil_matrix, np.ndarray]:
    """
    Add face transmissibilities then populate sparse matrix.

    :param A: Sparse coefficient matrix (lil_matrix format) to modify in place
    :param b: RHS vector to modify in place
    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param water_mobility_grid_x: Water mobility grid in x-direction (ft²/psi·day)
    :param oil_mobility_grid_x: Oil mobility grid in x-direction (ft²/psi·day)
    :param gas_mobility_grid_x: Gas mobility grid in x-direction (ft²/psi·day)
    :param water_mobility_grid_y: Water mobility grid in y-direction (ft²/psi·day)
    :param oil_mobility_grid_y: Oil mobility grid in y-direction (ft²/psi·day)
    :param gas_mobility_grid_y: Gas mobility grid in y-direction (ft²/psi·day)
    :param water_mobility_grid_z: Water mobility grid in z-direction (ft²/psi·day)
    :param oil_mobility_grid_z: Oil mobility grid in z-direction (ft²/psi·day)
    :param gas_mobility_grid_z: Gas mobility grid in z-direction (ft²/psi·day)
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure (psi)
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure (psi)
    :param oil_density_grid: Oil density grid (lb/ft³)
    :param water_density_grid: Water density grid (lb/ft³)
    :param gas_density_grid: Gas density grid (lb/ft³)
    :param elevation_grid: Elevation grid (ft)
    :param acceleration_due_to_gravity_ft_per_s2: Gravitational acceleration (ft/s²)
    :param dtype: Data type for arrays (np.float32 or np.float64)
    :return: Tuple of (A, b) with face contributions added (same objects passed in, modified in place)
    """
    # Compute using njitted function
    rows, cols, off_diag_values, diagonal_additions, rhs_additions = (
        compute_face_transmissibility_arrays(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            thickness_grid=thickness_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            water_mobility_grid_x=water_mobility_grid_x,
            oil_mobility_grid_x=oil_mobility_grid_x,
            gas_mobility_grid_x=gas_mobility_grid_x,
            water_mobility_grid_y=water_mobility_grid_y,
            oil_mobility_grid_y=oil_mobility_grid_y,
            gas_mobility_grid_y=gas_mobility_grid_y,
            water_mobility_grid_z=water_mobility_grid_z,
            oil_mobility_grid_z=oil_mobility_grid_z,
            gas_mobility_grid_z=gas_mobility_grid_z,
            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
            oil_density_grid=oil_density_grid,
            water_density_grid=water_density_grid,
            gas_density_grid=gas_density_grid,
            elevation_grid=elevation_grid,
            acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
            dtype=dtype,
        )
    )

    # Add to diagonal (element-wise addition)
    current_diagonal = A.diagonal()
    new_diagonal = current_diagonal + diagonal_additions
    A.setdiag(new_diagonal)

    # Add off-diagonal entries
    for i in range(len(rows)):
        A[rows[i], cols[i]] = off_diag_values[i]

    # Add to RHS
    b[:] += rhs_additions
    return A, b


def add_well_contributions(
    A: lil_matrix,
    b: np.ndarray,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    current_oil_pressure_grid: ThreeDimensionalGrid,
    temperature_grid: ThreeDimensionalGrid,
    absolute_permeability: typing.Any,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    water_compressibility_grid: ThreeDimensionalGrid,
    oil_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    cell_size_x: float,
    cell_size_y: float,
    time_step: int,
    time_step_size: float,
    config: Config,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[lil_matrix, np.ndarray]:
    """
    Compute well flow rates and add contributions to the RHS vector b.

    Well treatment is semi-implicit in time.

    :param A: Sparse coefficient matrix (lil_matrix format) - not modified, returned for consistency
    :param b: RHS vector to modify in place
    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param temperature_grid: Temperature grid (°F or °R)
    :param absolute_permeability: Absolute permeability in x, y, z directions (mD)
    :param water_relative_mobility_grid: Water relative mobility (1/cP)
    :param oil_relative_mobility_grid: Oil relative mobility (1/cP)
    :param gas_relative_mobility_grid: Gas relative mobility (1/cP)
    :param water_compressibility_grid: Water compressibility grid (1/psi)
    :param oil_compressibility_grid: Oil compressibility grid (1/psi)
    :param gas_compressibility_grid: Gas compressibility grid (1/psi)
    :param fluid_properties: Fluid properties container
    :param wells: Wells grid containing injection and production wells
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step: Current time step number
    :param time_step_size: Time step size (seconds)
    :param config: Simulation config
    :param md_per_cp_to_ft2_per_psi_per_day: Conversion factor from mD·ft/cP to ft²/psi·day
    :return: Tuple of (A, b) with well contributions added to b (A unchanged, returned for consistency)
    """
    _to_1D_index = functools.partial(
        to_1D_index_interior_only,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )

    # Compute well flow rates and add to RHS
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_1D_index = _to_1D_index(i, j, k)
        cell_thickness = thickness_grid[i, j, k]
        cell_temperature = temperature_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]

        injection_well, production_well = wells[i, j, k]
        interval_thickness = (cell_size_x, cell_size_y, cell_thickness)

        permeability = (
            absolute_permeability.x[i, j, k],
            absolute_permeability.y[i, j, k],
            absolute_permeability.z[i, j, k],
        )
        if (
            injection_well is not None
            and injection_well.is_open
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            injected_phase = injected_fluid.phase
            phase_fvf = injected_fluid.get_formation_volume_factor(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
            )
            # Get phase mobility
            if injected_phase == FluidPhase.GAS:
                phase_mobility = gas_relative_mobility_grid[i, j, k]
                compressibility_kwargs = {}
            else:  # Water injection
                phase_mobility = water_relative_mobility_grid[i, j, k]
                compressibility_kwargs = {
                    "bubble_point_pressure": fluid_properties.oil_bubble_point_pressure_grid[
                        i, j, k
                    ],
                    "gas_formation_volume_factor": phase_fvf,
                    "gas_solubility_in_water": fluid_properties.gas_solubility_in_water_grid[
                        i, j, k
                    ],
                }

            # Skip if no mobility
            if phase_mobility <= 0.0:
                continue

            # Get fluid properties
            phase_compressibility = injected_fluid.get_compressibility(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                **compressibility_kwargs,
            )

            use_pseudo_pressure = (
                config.use_pseudo_pressure and injected_phase == FluidPhase.GAS
            )
            well_index = injection_well.get_well_index(
                interval_thickness=interval_thickness,
                permeability=permeability,
                skin_factor=injection_well.skin_factor,
            )
            effective_bhp = injection_well.get_bottom_hole_pressure(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=injected_fluid,
                formation_volume_factor=phase_fvf,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=phase_compressibility,
                pvt_tables=config.pvt_tables,
            )

            # PI = mD·ft/cP * conversion = ft³/psi·day
            phase_productivity_index = (
                well_index * phase_mobility * md_per_cp_to_ft2_per_psi_per_day
            )
            # Check for backflow (cell pressure > BHP)
            if cell_oil_pressure > effective_bhp and config.warn_well_anomalies:
                _warn_injection_pressure_is_low(
                    bhp=effective_bhp,
                    cell_pressure=cell_oil_pressure,
                    well_name=injection_well.name,
                    time=time_step * time_step_size,
                    cell=(i, j, k),
                )

            # Semi-implicit coupling: q = PI * (p_wf - p_cell)
            # Rearranging: PI * p_cell = PI * p_wf - q
            # In pressure equation: ... = ... + q
            # So: A[i,i] += PI, b[i] += PI * p_wf
            A[cell_1D_index, cell_1D_index] += phase_productivity_index
            b[cell_1D_index] += phase_productivity_index * effective_bhp

        if production_well is not None and production_well.is_open:
            water_formation_volume_factor_grid = (
                fluid_properties.water_formation_volume_factor_grid
            )
            oil_formation_volume_factor_grid = (
                fluid_properties.oil_formation_volume_factor_grid
            )
            gas_formation_volume_factor_grid = (
                fluid_properties.gas_formation_volume_factor_grid
            )
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase

                # Get phase-specific properties
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = gas_relative_mobility_grid[i, j, k]
                    phase_compressibility = gas_compressibility_grid[i, j, k]
                    phase_fvf = gas_formation_volume_factor_grid[i, j, k]
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = water_relative_mobility_grid[i, j, k]
                    phase_compressibility = water_compressibility_grid[i, j, k]
                    phase_fvf = water_formation_volume_factor_grid[i, j, k]
                else:  # Oil
                    phase_mobility = oil_relative_mobility_grid[i, j, k]
                    phase_compressibility = oil_compressibility_grid[i, j, k]
                    phase_fvf = oil_formation_volume_factor_grid[i, j, k]

                # Skip if no mobility
                if phase_mobility <= 0.0:
                    continue

                use_pseudo_pressure = (
                    config.use_pseudo_pressure and produced_phase == FluidPhase.GAS
                )
                well_index = production_well.get_well_index(
                    interval_thickness=interval_thickness,
                    permeability=permeability,
                    skin_factor=production_well.skin_factor,
                )
                effective_bhp = production_well.get_bottom_hole_pressure(
                    pressure=cell_oil_pressure,
                    temperature=cell_temperature,
                    phase_mobility=phase_mobility,
                    well_index=well_index,
                    fluid=produced_fluid,
                    formation_volume_factor=phase_fvf,
                    use_pseudo_pressure=use_pseudo_pressure,
                    fluid_compressibility=phase_compressibility,
                    pvt_tables=config.pvt_tables,
                )

                # Check for backflow (positive production = injection)
                if cell_oil_pressure < effective_bhp and config.warn_well_anomalies:
                    _warn_production_pressure_is_high(
                        bhp=effective_bhp,
                        cell_pressure=cell_oil_pressure,
                        well_name=production_well.name,
                        time=time_step * time_step_size,
                        cell=(i, j, k),
                    )

                # Compute productivity index
                phase_productivity_index = (
                    well_index * phase_mobility * md_per_cp_to_ft2_per_psi_per_day
                )

                # Semi-implicit coupling (same form for production)
                A[cell_1D_index, cell_1D_index] += phase_productivity_index
                b[cell_1D_index] += phase_productivity_index * effective_bhp
    return A, b


@numba.njit(cache=True)
def compute_pseudo_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    water_mobility_grid: ThreeDimensionalGrid,
    oil_mobility_grid: ThreeDimensionalGrid,
    gas_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    acceleration_due_to_gravity_ft_per_s2: float,
) -> typing.Tuple[float, float, float]:
    """
    Computes and returns a tuple of the total harmonic mobility of the phases, the capillary pseudo flux,
    and the gravity pseudo flux from the neighbour to the current cell.

    Pseudo flux comes about from the fact that the fluxes returned are not actual fluxes with units of ft³/day,
    but rather pseudo fluxes with units of ft²/day, which are then used to compute the actual fluxes and subsequently,
    the flow rates in the implicit pressure evolution scheme.

    :param cell_indices: Indices of the current cell (i, j, k)
    :param neighbour_indices: Indices of the neighbouring cell (i±1, j, k) or (i, j±1, k) or (i, j, k±1)
    :param oil_pressure_grid: 3D grid of oil pressures (psi)
    :param water_mobility_grid: 3D grid of water mobilities (ft²/psi.day)
    :param oil_mobility_grid: 3D grid of oil mobilities (ft²/psi.day)
    :param gas_mobility_grid: 3D grid of gas mobilities (ft²/psi.day)
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi)
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi)
    :param oil_density_grid: 3D grid of oil densities (lb/ft³)
    :param water_density_grid: 3D grid of water densities (lb/ft³)
    :param gas_density_grid: 3D grid of gas densities (lb/ft³)
    :param elevation_grid: 3D grid of elevations (ft)
    :return: A tuple containing:
        - Total harmonic mobility (ft²/psi.day)
        - Total capillary pseudo flux (ft²/day)
        - Total gravity pseudo flux (ft²/day)
    """
    # Calculate pressure differences relative to current cell (Neighbour - Current)
    # These represent the gradients driving flow from neighbour to current cell, or vice versa
    oil_water_capillary_pressure_difference = (
        oil_water_capillary_pressure_grid[neighbour_indices]
        - oil_water_capillary_pressure_grid[cell_indices]
    )
    gas_oil_capillary_pressure_difference = (
        gas_oil_capillary_pressure_grid[neighbour_indices]
        - gas_oil_capillary_pressure_grid[cell_indices]
    )

    # Calculate the elevation difference between the neighbour and current cell
    elevation_difference = (
        elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
    )
    # Determine the harmonic densities for each phase across the face
    harmonic_water_density = compute_harmonic_mean(
        water_density_grid[neighbour_indices], water_density_grid[cell_indices]
    )
    harmonic_oil_density = compute_harmonic_mean(
        oil_density_grid[neighbour_indices], oil_density_grid[cell_indices]
    )
    harmonic_gas_density = compute_harmonic_mean(
        gas_density_grid[neighbour_indices], gas_density_grid[cell_indices]
    )

    # Calculate harmonic mobilities for each phase across the face
    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid[neighbour_indices], water_mobility_grid[cell_indices]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid[neighbour_indices], oil_mobility_grid[cell_indices]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid[neighbour_indices], gas_mobility_grid[cell_indices]
    )
    total_harmonic_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )
    if total_harmonic_mobility <= 0.0:
        # No flow can occur if there is no mobility
        return 0.0, 0.0, 0.0

    # λ_w * (P_cow_{n+1} - P_cow_{n}) (ft²/psi.day * psi = ft²/day)
    water_capillary_pseudo_flux = (
        water_harmonic_mobility * oil_water_capillary_pressure_difference
    )
    # λ_g * (P_cgo_{n+1} - P_cgo_{n}) (ft²/psi.day * psi = ft²/day)
    gas_capillary_pseudo_flux = (
        gas_harmonic_mobility * gas_oil_capillary_pressure_difference
    )
    # Total capillary flux from the neighbour (ft²/day)
    total_capillary_pseudo_flux = (
        water_capillary_pseudo_flux + gas_capillary_pseudo_flux
    )

    # Calculate the phase gravity potentials (hydrostatic/gravity head)
    water_gravity_potential = (
        harmonic_water_density
        * acceleration_due_to_gravity_ft_per_s2
        * elevation_difference
    ) / 144.0
    oil_gravity_potential = (
        harmonic_oil_density
        * acceleration_due_to_gravity_ft_per_s2
        * elevation_difference
    ) / 144.0
    gas_gravity_potential = (
        harmonic_gas_density
        * acceleration_due_to_gravity_ft_per_s2
        * elevation_difference
    ) / 144.0
    # Total gravity pseudo flux (ft²/day)
    water_gravity_pseudo_flux = water_harmonic_mobility * water_gravity_potential
    oil_gravity_pseudo_flux = oil_harmonic_mobility * oil_gravity_potential
    gas_gravity_pseudo_flux = gas_harmonic_mobility * gas_gravity_potential
    total_gravity_pseudo_flux = (
        water_gravity_pseudo_flux + oil_gravity_pseudo_flux + gas_gravity_pseudo_flux
    )
    return (
        total_harmonic_mobility,
        total_capillary_pseudo_flux,
        total_gravity_pseudo_flux,
    )


"""
Implicit finite difference formulation for pressure diffusion in a 3D reservoir
(slightly compressible fluid):

The governing equation for pressure evolution is the linear-flow diffusivity equation:

    ∂p/∂t * (φ·c_t) * V = ∇·(λ·∇p) * A + q * V

Where:
    ∂p/∂t * (φ·c_t) * V — Accumulation term
    ∇·(λ·∇p) * A — Diffusion term (Darcy's law)
    q * V — Source/sink term

Assumptions:
    - Constant porosity (φ), compressibility (c_t), and density (ρ)
    - No reaction or advection terms (pressure-only evolution)
    - Capillary effects optional, appear in source term via pressure corrections

Diffusion term expanded in 3D:

    ∇·(λ·∇p) = ∂/∂x(λ·∂p/∂x) + ∂/∂y(λ·∂p/∂y) + ∂/∂z(λ·∂p/∂z)

Discretization:

Using Backward Euler in time:

    ∂p/∂t ≈ (pⁿ⁺¹_ijk - pⁿ_ijk) / Δt

Using central differences in space:

    ∂/∂x(λ·∂p/∂x) ≈ [λ_{i+½,j,k}·(pⁿ⁺¹_{i+1,j,k} - pⁿ⁺¹_{i,j,k}) - λ_{i⁻½,j,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i⁻1,j,k})] / Δx²
    ∂/∂y(λ·∂p/∂y) ≈ [λ_{i,j+½,k}·(pⁿ⁺¹_{i,j+1,k} - pⁿ⁺¹_{i,j,k}) - λ_{i,j⁻½,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j⁻1,k})] / Δy²
    ∂/∂z(λ·∂p/∂z) ≈ [λ_{i,j,k+½}·(pⁿ⁺¹_{i,j,k+1} - pⁿ⁺¹_{i,j,k}) - λ_{i,j,k⁻½}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j,k⁻1})] / Δz²

Putting it all together:

    (pⁿ⁺¹_ijk - pⁿ_ijk) * (φ·c_t·V) / Δt =
        A/Δx · [λ_{i+½,j,k}·(pⁿ⁺¹_{i+1,j,k} - pⁿ⁺¹_{i,j,k}) - λ_{i⁻½,j,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i⁻1,j,k})] +
        A/Δy · [λ_{i,j+½,k}·(pⁿ⁺¹_{i,j+1,k} - pⁿ⁺¹_{i,j,k}) - λ_{i,j⁻½,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j⁻1,k})] +
        A/Δz · [λ_{i,j,k+½}·(pⁿ⁺¹_{i,j,k+1} - pⁿ⁺¹_{i,j,k}) - λ_{i,j,k⁻½}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j,k⁻1})] +
        qⁿ⁺¹_ijk * V

Matrix form:

Let:
    Tx⁺ = λ_{i+½,j,k}·A / Δx
    Tx⁻ = λ_{i⁻½,j,k}·A / Δx
    Ty⁺ = λ_{i,j+½,k}·A / Δy
    Ty⁻ = λ_{i,j⁻½,k}·A / Δy
    Tz⁺ = λ_{i,j,k+½}·A / Δz
    Tz⁻ = λ_{i,j,k⁻½}·A / Δz
    β   = φ·c_t·V / Δt

Then:

    A_{ijk,ijk}     = β + Tx⁺ + Tx⁻ + Ty⁺ + Ty⁻ + Tz⁺ + Tz⁻
    A_{ijk,i+1jk}   = -Tx⁺
    A_{ijk,i-1jk}   = -Tx⁻
    A_{ijk,ij+1k}   = -Ty⁺
    A_{ijk,ij-1k}   = -Ty⁻
    A_{ijk,ijk+1}   = -Tz⁺
    A_{ijk,ijk-1}   = -Tz⁻

RHS vector: (Contains terms that actually drive flow)

    b_{ijk} = (β * pⁿ_{ijk}) + (q_{ijk} * V) + Total Capillary Driven Flow + Gravity Driven Flow/Segregation

Capillary pressure driven flow term (if multiphase):

    total_capillary_flow = sum of directional contributions:
        For each direction:
            [(λ_w * ∇P_cow) + (λ_g * ∇P_cgo)] * A / (Δx, Δy, Δz)

Gravity driven segregation (only in effect in the z-direction):

    total_gravity_flow = (
            [λ_w * (upwind_ρ_w * g * ∆z) / 144] 
            + [λ_g * (upwind_ρ_g * g * ∆z) / 144] 
            + [λ_o * (upwind_ρ_o * g * ∆z) / 144]
    ) * A / Δz

    Where;
    g is the gravitational acceleration (32.174 ft/s²),
    upwind_ρ_w, upwind_ρ_g, upwind_ρ_o are the densities of water, gas, and oil

This results in a 7-point stencil sparse matrix (in 3D) for solving A·pⁿ⁺¹ = b.

Notes:
    - Harmonic averaging is used for λ at cell interfaces
    - Capillary pressure is optional but included via ∇P_cow and ∇P_cgo terms
    - The system is solved each time step to advance pressure implicitly
"""
