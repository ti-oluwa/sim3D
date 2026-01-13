"""
Fully implicit solver for simultaneous pressure and saturation evolution.
"""

import itertools
import logging
import typing

import attrs
import numba
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import LinearOperator

from bores._precision import get_dtype
from bores.boundary_conditions import BoundaryConditions
from bores.config import Config
from bores.constants import c
from bores.diffusivity.base import (
    EvolutionResult,
    compute_mobility_grids,
    solve_linear_system,
    to_1D_index_interior_only,
)
from bores.diffusivity.implicit.jacobian import (
    assemble_jacobian,
    assemble_jacobian_with_frozen_mobility,
)
from bores.errors import SolverError, PreconditionerError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.boundary_conditions import apply_boundary_conditions
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.grids.updates import update_pvt_grids
from bores.models import FluidProperties, RockFluidProperties, RockProperties
from bores.pvt.core import compute_harmonic_mean
from bores.types import (
    FluidPhase,
    SupportsSetItem,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.utils import clip
from bores.wells import Wells


logger = logging.getLogger(__name__)


@attrs.frozen(slots=True)
class ImplicitSolution:
    """Result from fully implicit solver."""

    pressure_grid: ThreeDimensionalGrid
    oil_saturation_grid: ThreeDimensionalGrid
    gas_saturation_grid: ThreeDimensionalGrid
    water_saturation_grid: ThreeDimensionalGrid
    converged: bool
    newton_iterations: int
    final_residual_norm: float


@attrs.frozen(slots=True)
class IterationState:
    """Current state during a Newton-Raphson iteration."""

    pressure_grid: ThreeDimensionalGrid
    oil_saturation_grid: ThreeDimensionalGrid
    gas_saturation_grid: ThreeDimensionalGrid
    iteration: int
    max_saturation_change: float
    max_pressure_change: float


def evolve_fully_implicit(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
) -> EvolutionResult[ImplicitSolution, None]:
    """
    Solve fully implicit finite-difference equations for pressure and saturations.

    :param cell_dimension: (dx, dy) in feet
    :param thickness_grid: Cell thickness (dz) grid in feet
    :param elevation_grid: Cell center elevation grid in feet
    :param time_step: Current time step number
    :param time_step_size: Time step size in seconds
    :param rock_properties: Rock properties including porosity and permeability
    :param fluid_properties: Fluid properties at current and previous time steps
    :param rock_fluid_properties: Relative permeability and capillary pressure models
    :param relative_mobility_grids: Mobility grids (kr/mu) for each phase
    :param capillary_pressure_grids: Capillary pressure grids
    :param wells: Well objects with controls
    :param config: Solver config including tolerance and max iterations
    :param boundary_conditions: Boundary conditions
    :param injection_grid: Object supporting setitem to set cell injection rates for each phase in ft³/day.
    :param production_grid: Object supporting setitem to set cell production rates for each phase in ft³/day.
    :return: `EvolutionResult` containing `ImplicitSolution` with updated pressure and saturations
    """
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    old_oil_saturation_grid = fluid_properties.oil_saturation_grid.copy()
    old_gas_saturation_grid = fluid_properties.gas_saturation_grid.copy()
    old_water_saturation_grid = fluid_properties.water_saturation_grid.copy()

    pressure_grid = fluid_properties.pressure_grid.copy()
    oil_saturation_grid = fluid_properties.oil_saturation_grid.copy()
    gas_saturation_grid = fluid_properties.gas_saturation_grid.copy()
    water_saturation_grid = fluid_properties.water_saturation_grid.copy()

    porosity_grid = rock_properties.porosity_grid
    absolute_permeability_x = rock_properties.absolute_permeability.x
    absolute_permeability_y = rock_properties.absolute_permeability.y
    absolute_permeability_z = rock_properties.absolute_permeability.z

    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape
    # Get grid shape without padding
    original_grid_shape = (cell_count_x - 2, cell_count_y - 2, cell_count_z - 2)

    max_newton_iterations = config.max_iterations
    convergence_tolerance = config.convergence_tolerance
    converged = False
    iteration = 0
    previous_residual_norm = 0.0
    current_residual_norm = 0.0

    cached_jacobian = None
    cached_preconditioner = None
    needs_reassembly = True
    jacobian_reuse_count = 0
    dtype = get_dtype()

    mD_per_cP_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    acceleration_due_to_gravity_ft_per_s2 = c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
    phase_appearance_tolerance = config.phase_appearance_tolerance
    max_jacobian_reuses = config.max_jacobian_reuses
    damping_controller = config.damping_controller_factory()

    for iteration in range(max_newton_iterations):
        logger.debug(
            f"Computing mobility grids and well rates for iteration {iteration}..."
        )
        mobility_grids = compute_mobility_grids(
            absolute_permeability_x=absolute_permeability_x,
            absolute_permeability_y=absolute_permeability_y,
            absolute_permeability_z=absolute_permeability_z,
            water_relative_mobility_grid=relative_mobility_grids.water_relative_mobility,
            oil_relative_mobility_grid=relative_mobility_grids.oil_relative_mobility,
            gas_relative_mobility_grid=relative_mobility_grids.gas_relative_mobility,
            md_per_cp_to_ft2_per_psi_per_day=mD_per_cP_to_ft2_per_psi_per_day,
        )
        well_rate_grids = compute_well_rate_grids(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            wells=wells,
            pressure_grid=pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            absolute_permeability_x=absolute_permeability_x,
            absolute_permeability_y=absolute_permeability_y,
            absolute_permeability_z=absolute_permeability_z,
            fluid_properties=fluid_properties,
            relative_mobility_grids=relative_mobility_grids,
            config=config,
            injection_grid=injection_grid,
            production_grid=production_grid,
            dtype=dtype,
        )

        (
            (water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x),
            (water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y),
            (water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z),
        ) = mobility_grids
        oil_well_rate_grid, gas_well_rate_grid, water_well_rate_grid = well_rate_grids

        logger.debug(f"Assembling residual vector for iteration {iteration}...")
        residual_vector = assemble_residual_vector(
            pressure_grid=pressure_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            old_oil_saturation_grid=old_oil_saturation_grid,
            old_gas_saturation_grid=old_gas_saturation_grid,
            old_water_saturation_grid=old_water_saturation_grid,
            oil_density_grid=fluid_properties.oil_effective_density_grid,
            gas_density_grid=fluid_properties.gas_density_grid,
            water_density_grid=fluid_properties.water_density_grid,
            porosity_grid=porosity_grid,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_in_days=time_step_in_days,
            pcow_grid=capillary_pressure_grids.oil_water_capillary_pressure,
            pcgo_grid=capillary_pressure_grids.gas_oil_capillary_pressure,
            oil_mobility_grid_x=oil_mobility_grid_x,
            oil_mobility_grid_y=oil_mobility_grid_y,
            oil_mobility_grid_z=oil_mobility_grid_z,
            gas_mobility_grid_x=gas_mobility_grid_x,
            gas_mobility_grid_y=gas_mobility_grid_y,
            gas_mobility_grid_z=gas_mobility_grid_z,
            water_mobility_grid_x=water_mobility_grid_x,
            water_mobility_grid_y=water_mobility_grid_y,
            water_mobility_grid_z=water_mobility_grid_z,
            oil_well_rate_grid=oil_well_rate_grid,
            gas_well_rate_grid=gas_well_rate_grid,
            water_well_rate_grid=water_well_rate_grid,
            acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
            dtype=dtype,
        )
        current_residual_norm = float(np.linalg.norm(residual_vector))

        # Log first iteration residual for diagnostics
        if iteration == 0:
            logger.info("-" * 60)
            logger.info("INITIAL STATE DIAGNOSTICS")
            logger.info("-" * 60)
            logger.info(f"Residual norm: {current_residual_norm:.6e}")
            logger.info(f"Max residual: {np.max(np.abs(residual_vector)):.6e}")
            logger.info(f"Min residual: {np.min(np.abs(residual_vector)):.6e}")
            logger.info(f"Mean residual: {np.mean(np.abs(residual_vector)):.6e}")
            logger.info(
                f"Number of residuals > 1.0: {np.sum(np.abs(residual_vector) > 1.0)}"
            )
            logger.info(
                f"Number of residuals > 0.1: {np.sum(np.abs(residual_vector) > 0.1)}"
            )
            logger.info(
                f"Number of residuals > 0.01: {np.sum(np.abs(residual_vector) > 0.01)}"
            )

            logger.info("First 10 residuals:")
            for idx in range(min(10, len(residual_vector))):
                logger.info(f"  residual[{idx}] = {residual_vector[idx]:.6e}")

            logger.info(
                f"Pressure range: [{np.min(pressure_grid):.2f}, {np.max(pressure_grid):.2f}] psi"
            )
            logger.info(
                f"Oil saturation range: [{np.min(oil_saturation_grid):.4f}, {np.max(oil_saturation_grid):.4f}]"
            )
            logger.info(
                f"Gas saturation range: [{np.min(gas_saturation_grid):.4f}, {np.max(gas_saturation_grid):.4f}]"
            )
            logger.info("-" * 60)

        # Check for NaN or Inf in residuals
        if not np.isfinite(current_residual_norm):
            logger.error(
                f"Non-finite residual norm at iteration {iteration}: {current_residual_norm}. "
                f"NaN count: {np.isnan(residual_vector).sum()}, "
                f"Inf count: {np.isinf(residual_vector).sum()}"
            )
            break

        relative_reduction = (
            current_residual_norm / previous_residual_norm
            if previous_residual_norm > 0
            else 0
        )
        logger.info(
            f"Iteration {iteration}: Residual norm = {current_residual_norm:.4e}, "
            f"Relative reduction = {relative_reduction:.4e}"
        )
        if current_residual_norm < convergence_tolerance:
            logger.info(
                f"Converged after {iteration} iterations with residual norm {current_residual_norm:.4e}"
            )
            converged = True
            break

        # Decrease damping factor if residual norm increased
        if iteration > 0 and current_residual_norm > (previous_residual_norm * 1.05):
            damping_controller.decrease()

        if jacobian_reuse_count >= max_jacobian_reuses or relative_reduction > 0.85:
            needs_reassembly = True

        if needs_reassembly or cached_jacobian is None:
            logger.debug(f"Assembling Jacobian for iteration {iteration}...")
            # jacobian = assemble_jacobian(
            #     cell_dimension=cell_dimension,
            #     thickness_grid=thickness_grid,
            #     pressure_grid=pressure_grid,
            #     elevation_grid=elevation_grid,
            #     oil_saturation_grid=oil_saturation_grid,
            #     gas_saturation_grid=gas_saturation_grid,
            #     water_saturation_grid=water_saturation_grid,
            #     time_step_size=time_step_in_days,
            #     rock_properties=rock_properties,
            #     fluid_properties=fluid_properties,
            #     rock_fluid_properties=rock_fluid_properties,
            #     oil_mobility_grid_x=oil_mobility_grid_x,
            #     oil_mobility_grid_y=oil_mobility_grid_y,
            #     oil_mobility_grid_z=oil_mobility_grid_z,
            #     gas_mobility_grid_x=gas_mobility_grid_x,
            #     gas_mobility_grid_y=gas_mobility_grid_y,
            #     gas_mobility_grid_z=gas_mobility_grid_z,
            #     water_mobility_grid_x=water_mobility_grid_x,
            #     water_mobility_grid_y=water_mobility_grid_y,
            #     water_mobility_grid_z=water_mobility_grid_z,
            #     oil_well_rate_grid=oil_well_rate_grid,
            #     gas_well_rate_grid=gas_well_rate_grid,
            #     water_well_rate_grid=water_well_rate_grid,
            #     pcow_grid=capillary_pressure_grids.oil_water_capillary_pressure,
            #     pcgo_grid=capillary_pressure_grids.gas_oil_capillary_pressure,
            #     dtype=dtype,
            #     well_damping_factor=1.0,
            #     phase_appearance_tolerance=phase_appearance_tolerance,
            #     acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
            # )
            jacobian = assemble_jacobian_with_frozen_mobility(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                pressure_grid=pressure_grid,
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
                time_step_size=time_step_in_days,
                rock_properties=rock_properties,
                fluid_properties=fluid_properties,
                rock_fluid_properties=rock_fluid_properties,
                oil_mobility_grid_x=oil_mobility_grid_x,
                oil_mobility_grid_y=oil_mobility_grid_y,
                oil_mobility_grid_z=oil_mobility_grid_z,
                gas_mobility_grid_x=gas_mobility_grid_x,
                gas_mobility_grid_y=gas_mobility_grid_y,
                gas_mobility_grid_z=gas_mobility_grid_z,
                water_mobility_grid_x=water_mobility_grid_x,
                water_mobility_grid_y=water_mobility_grid_y,
                water_mobility_grid_z=water_mobility_grid_z,
                oil_well_rate_grid=oil_well_rate_grid,
                gas_well_rate_grid=gas_well_rate_grid,
                water_well_rate_grid=water_well_rate_grid,
                dtype=dtype,
                well_damping_factor=1.0,
                phase_appearance_tolerance=phase_appearance_tolerance,
            )
            cached_jacobian = jacobian
            # Reset reassembly flags, cached preconditioner and counters
            needs_reassembly = False
            jacobian_reuse_count = 0
            cached_preconditioner = None
        else:
            logger.debug(
                f"Reusing cached Jacobian for iteration {iteration} (reuse count={jacobian_reuse_count})"
            )
            jacobian = cached_jacobian
            jacobian_reuse_count += 1

        jacobian_csr = jacobian.tocsr()
        diag = jacobian_csr.diagonal()
        logger.info(
            f"Jacobian diagonal: min={np.min(np.abs(diag[diag != 0])):.4e}, max={np.max(np.abs(diag)):.4e}"
        )
        logger.info(
            f"Jacobian: {np.sum(diag == 0)} zero diagonal entries out of {len(diag)}"
        )
        logger.info(f"Jacobian nnz: {jacobian_csr.nnz}")

        try:
            if cached_preconditioner is None:
                update_vector, preconditioner = solve_newton_update_system(
                    jacobian=jacobian,
                    residual_vector=residual_vector,
                    config=config,
                    preconditioner=None,
                )
                cached_preconditioner = preconditioner
            else:
                update_vector, _ = solve_newton_update_system(
                    jacobian=jacobian,
                    residual_vector=residual_vector,
                    config=config,
                    preconditioner=cached_preconditioner,
                )
        except (SolverError, PreconditionerError) as exc:
            logger.error(
                f"Linear solver failed at iteration {iteration} with error: {exc}"
            )
            return EvolutionResult(
                success=False,
                value=ImplicitSolution(
                    pressure_grid=pressure_grid,
                    oil_saturation_grid=oil_saturation_grid,
                    gas_saturation_grid=gas_saturation_grid,
                    water_saturation_grid=water_saturation_grid,
                    converged=False,
                    newton_iterations=iteration + 1,
                    final_residual_norm=current_residual_norm,
                ),
                scheme="implicit",
                message=f"Linear solver failure during Newton iteration. {exc}",
            )

        # Apply Newton update and get iteration state
        iteration_state = apply_newton_update(
            pressure_grid=pressure_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            update_vector=update_vector,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            iteration=iteration,
            damping_factor=damping_controller.get(),
        )
        max_saturation_change = iteration_state.max_saturation_change
        max_pressure_change = iteration_state.max_pressure_change

        # If the update is too large, reassemble the Jacobian next iteration
        # Also decrease damping factor
        if max_saturation_change > 0.2 or max_pressure_change > 500.0:
            needs_reassembly = True
            damping_controller.decrease()
        else:
            damping_controller.increase()

        pressure_grid = iteration_state.pressure_grid
        oil_saturation_grid = iteration_state.oil_saturation_grid
        gas_saturation_grid = iteration_state.gas_saturation_grid
        water_saturation_grid = 1.0 - oil_saturation_grid - gas_saturation_grid

        # Update fluid properties for next iteration with new pressure and saturations
        logger.debug("Updating rock and fluid properties for next iteration...")
        fluid_properties = attrs.evolve(
            fluid_properties,
            pressure_grid=pressure_grid,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
        )
        # Re-apply boundary conditions after update
        fluid_properties, rock_properties = apply_boundary_conditions(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            boundary_conditions=boundary_conditions,
            cell_dimension=cell_dimension,
            grid_shape=original_grid_shape,
            thickness_grid=thickness_grid,
            time=time_step * time_step_size,
        )
        # Update other PVT properties based on new pressure and saturations
        fluid_properties = update_pvt_grids(
            fluid_properties=fluid_properties,
            wells=wells,
            miscibility_model=config.miscibility_model,
            pvt_tables=config.pvt_tables,
        )
        # Rebuild rock-fluid property grids with updated saturations
        (
            _,
            relative_mobility_grids,
            capillary_pressure_grids,
        ) = build_rock_fluid_properties_grids(
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            irreducible_water_saturation_grid=rock_properties.irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=rock_properties.residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=rock_properties.residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=rock_properties.residual_gas_saturation_grid,
            water_viscosity_grid=fluid_properties.water_viscosity_grid,
            oil_viscosity_grid=fluid_properties.oil_viscosity_grid,
            gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
            relative_permeability_table=rock_fluid_properties.relative_permeability_table,
            capillary_pressure_table=rock_fluid_properties.capillary_pressure_table,
            disable_capillary_effects=config.disable_capillary_effects,
            capillary_strength_factor=config.capillary_strength_factor,
            relative_mobility_range=config.relative_mobility_range,
        )
        # Update previous residual norm for next iteration
        previous_residual_norm = current_residual_norm

    solution = ImplicitSolution(
        pressure_grid=pressure_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        water_saturation_grid=water_saturation_grid,
        converged=converged,
        newton_iterations=iteration + 1,
        final_residual_norm=current_residual_norm,
    )
    return EvolutionResult(value=solution, scheme="implicit", success=converged)


@numba.njit(cache=True)
def compute_accumulation_terms(
    porosity: float,
    oil_saturation: float,
    gas_saturation: float,
    water_saturation: float,
    cell_volume: float,
) -> typing.Tuple[float, float, float]:
    """
    Compute accumulation terms for oil, gas, and water phases in reservoir conditions.

    Computes φ·S·ρ·V for each phase, representing the volume of each phase in the pore space.

    :param porosity: Porosity at cell (dimensionless)
    :param oil_saturation: Oil saturation at cell
    :param gas_saturation: Gas saturation at cell
    :param water_saturation: Water saturation at cell
    :param cell_volume: Cell volume in ft³
    :return: (oil_accumulation, gas_accumulation, water_accumulation) in ft³
    """
    # Accumulation in reservoir conditions: φ·S·V (ft³)
    oil_accumulation = porosity * oil_saturation * cell_volume
    gas_accumulation = porosity * gas_saturation * cell_volume
    water_accumulation = porosity * water_saturation * cell_volume
    return oil_accumulation, gas_accumulation, water_accumulation


def compute_well_rate_grids(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    wells: Wells[ThreeDimensions],
    pressure_grid: ThreeDimensionalGrid,
    temperature_grid: ThreeDimensionalGrid,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    absolute_permeability_x: ThreeDimensionalGrid,
    absolute_permeability_y: ThreeDimensionalGrid,
    absolute_permeability_z: ThreeDimensionalGrid,
    fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    config: Config,
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    dtype: np.typing.DTypeLike = np.float64,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Compute well rate contributions for all cells.

    :param cell_count_x: Number of cells in x direction
    :param cell_count_y: Number of cells in y direction
    :param cell_count_z: Number of cells in z direction
    :param wells: Wells object containing injection and production wells
    :param pressure_grid: Pressure values (psi)
    :param temperature_grid: Temperature values (°R)
    :param cell_dimension: (dx, dy) in feet
    :param thickness_grid: Cell thickness values (ft)
    :param absolute_permeability_x: Absolute permeability x component (md)
    :param absolute_permeability_y: Absolute permeability y component (md)
    :param absolute_permeability_z: Absolute permeability z component (md)
    :param fluid_properties: Fluid properties object
    :param relative_mobility_grids: Relative mobility grids for each phase
    :param config: Evolution config
    :param injection_grid: Optional grid to store injection rates (ft³/day)
    :param production_grid: Optional grid to store production rates (ft³/day)
    :param dtype: Data type for output arrays
    :return: Tuple of (oil_well_rate_grid, gas_well_rate_grid, water_well_rate_grid)
             where rates are in ft³/day (positive for injection, negative for production)
    """
    # Initialize well rate grids
    oil_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    gas_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    water_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )

    dx, dy = cell_dimension

    for i, j, k in itertools.product(
        range(cell_count_x),
        range(cell_count_y),
        range(cell_count_z),
    ):
        pressure = pressure_grid[i, j, k]
        temperature = temperature_grid[i, j, k]
        thickness = thickness_grid[i, j, k]
        permeability = (
            absolute_permeability_x[i, j, k],
            absolute_permeability_y[i, j, k],
            absolute_permeability_z[i, j, k],
        )

        oil_rate = 0.0
        gas_rate = 0.0
        water_rate = 0.0

        cell_oil_injection_rate = 0.0
        cell_water_injection_rate = 0.0
        cell_gas_injection_rate = 0.0
        cell_oil_production_rate = 0.0
        cell_water_production_rate = 0.0
        cell_gas_production_rate = 0.0

        injection_well, production_well = wells[i, j, k]

        # Handle injection well
        if (
            injection_well is not None
            and injection_well.is_open
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            injected_phase = injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                phase_mobility = relative_mobility_grids.gas_relative_mobility[i, j, k]
                compressibility_kwargs = {}
            else:
                phase_mobility = relative_mobility_grids.water_relative_mobility[
                    i, j, k
                ]
                gas_solubility_in_water = fluid_properties.gas_solubility_in_water_grid[
                    i, j, k
                ]
                compressibility_kwargs = {
                    "bubble_point_pressure": fluid_properties.oil_bubble_point_pressure_grid[
                        i, j, k
                    ],
                    "gas_formation_volume_factor": fluid_properties.gas_formation_volume_factor_grid[
                        i, j, k
                    ],
                    "gas_solubility_in_water": gas_solubility_in_water,
                }

            fluid_compressibility = injected_fluid.get_compressibility(
                pressure=pressure,
                temperature=temperature,
                **compressibility_kwargs,
            )
            fluid_formation_volume_factor = injected_fluid.get_formation_volume_factor(
                pressure=pressure,
                temperature=temperature,
            )

            use_pseudo_pressure = (
                config.use_pseudo_pressure and injected_phase == FluidPhase.GAS
            )
            well_index = injection_well.get_well_index(
                interval_thickness=(dx, dy, thickness),
                permeability=permeability,
                skin_factor=injection_well.skin_factor,
            )
            cell_injection_rate = injection_well.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                well_index=well_index,
                phase_mobility=phase_mobility,
                fluid=injected_fluid,
                fluid_compressibility=fluid_compressibility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=fluid_formation_volume_factor,
                pvt_tables=config.pvt_tables,
            )
            if injected_phase != FluidPhase.GAS:
                cell_injection_rate *= c.BBL_TO_FT3

            # Volumetric rates at reservoir conditions (ft³/day)
            if injected_phase == FluidPhase.GAS:
                gas_rate += cell_injection_rate
                cell_gas_injection_rate = cell_injection_rate
            else:
                water_rate += cell_injection_rate
                cell_water_injection_rate = cell_injection_rate

            # Record injection rate for the cell
            if injection_grid is not None:
                injection_grid[i, j, k] = (
                    cell_oil_injection_rate,
                    cell_water_injection_rate,
                    cell_gas_injection_rate,
                )

        # Handle production well
        if production_well is not None and production_well.is_open:
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase

                if produced_phase == FluidPhase.GAS:
                    phase_mobility = relative_mobility_grids.gas_relative_mobility[
                        i, j, k
                    ]
                    fluid_compressibility = fluid_properties.gas_compressibility_grid[
                        i, j, k
                    ]
                    fluid_formation_volume_factor = (
                        fluid_properties.gas_formation_volume_factor_grid[i, j, k]
                    )
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = relative_mobility_grids.water_relative_mobility[
                        i, j, k
                    ]
                    fluid_compressibility = fluid_properties.water_compressibility_grid[
                        i, j, k
                    ]
                    fluid_formation_volume_factor = (
                        fluid_properties.water_formation_volume_factor_grid[i, j, k]
                    )
                else:
                    phase_mobility = relative_mobility_grids.oil_relative_mobility[
                        i, j, k
                    ]
                    fluid_compressibility = fluid_properties.oil_compressibility_grid[
                        i, j, k
                    ]
                    fluid_formation_volume_factor = (
                        fluid_properties.oil_formation_volume_factor_grid[i, j, k]
                    )

                use_pseudo_pressure = (
                    config.use_pseudo_pressure and produced_phase == FluidPhase.GAS
                )
                well_index = production_well.get_well_index(
                    interval_thickness=(dx, dy, thickness),
                    permeability=permeability,
                    skin_factor=production_well.skin_factor,
                )
                production_rate = production_well.get_flow_rate(
                    pressure=pressure,
                    temperature=temperature,
                    well_index=well_index,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                    fluid_compressibility=fluid_compressibility,
                    use_pseudo_pressure=use_pseudo_pressure,
                    formation_volume_factor=fluid_formation_volume_factor,
                    pvt_tables=config.pvt_tables,
                )
                if produced_phase != FluidPhase.GAS:
                    production_rate *= c.BBL_TO_FT3

                # Volumetric rates at reservoir conditions (ft³/day)
                if produced_phase == FluidPhase.GAS:
                    gas_rate += production_rate
                    cell_gas_production_rate += production_rate
                elif produced_phase == FluidPhase.WATER:
                    water_rate += production_rate
                    cell_water_production_rate += production_rate
                else:
                    oil_rate += production_rate
                    cell_oil_production_rate += production_rate

            # Record total production rate for the cell (all phases)
            if production_grid is not None:
                production_grid[i, j, k] = (
                    cell_oil_production_rate,
                    cell_water_production_rate,
                    cell_gas_production_rate,
                )

        # Store well rates for this cell in ft³/day
        oil_well_rate_grid[i, j, k] = oil_rate
        gas_well_rate_grid[i, j, k] = gas_rate
        water_well_rate_grid[i, j, k] = water_rate

    return oil_well_rate_grid, gas_well_rate_grid, water_well_rate_grid


@numba.njit(cache=True)
def compute_phase_fluxes_from_neighbour(
    i: int,
    j: int,
    k: int,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
    water_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    pcow_grid: ThreeDimensionalGrid,
    pcgo_grid: ThreeDimensionalGrid,
    acceleration_due_to_gravity_ft_per_s2: float,
) -> typing.Tuple[float, float, float]:
    """
    Compute flux divergence for oil, gas, and water phases at a cell in reservoir conditions.

    Computes ∇·vα for each phase using two-point flux approximation.
    Includes transmissibility, capillary pressure, and gravity effects.
    All calculations in reservoir conditions (volumetric fluxes in ft³/day).

    :param i: x index
    :param j: y index
    :param k: z index
    :param cell_dimension: (dx, dy) in feet
    :param thickness_grid: Thickness grid
    :param elevation_grid: Elevation grid
    :param pressure_grid: Pressure grid
    :param oil_mobility_grid_x: Oil mobility in x direction (ft²/(psi·day))
    :param oil_mobility_grid_y: Oil mobility in y direction (ft²/(psi·day))
    :param oil_mobility_grid_z: Oil mobility in z direction (ft²/(psi·day))
    :param gas_mobility_grid_x: Gas mobility in x direction (ft²/(psi·day))
    :param gas_mobility_grid_y: Gas mobility in y direction (ft²/(psi·day))
    :param gas_mobility_grid_z: Gas mobility in z direction (ft²/(psi·day))
    :param water_mobility_grid_x: Water mobility in x direction (ft²/(psi·day))
    :param water_mobility_grid_y: Water mobility in y direction (ft²/(psi·day))
    :param water_mobility_grid_z: Water mobility in z direction (ft²/(psi·day))
    :param oil_density_grid: Oil density grid (lbm/ft³)
    :param gas_density_grid: Gas density grid (lbm/ft³)
    :param water_density_grid: Water density grid (lbm/ft³)
    :param pcow_grid: Oil-water capillary pressure grid (psi)
    :param pcgo_grid: Gas-oil capillary pressure grid (psi)
    :param acceleration_due_to_gravity_ft_per_s2: Acceleration due to gravity (ft/s²)
    :return: (oil_flux_div, gas_flux_div, water_flux_div) in ft³/day
    """
    dx, dy = cell_dimension
    dz = thickness_grid[i, j, k]

    oil_flux_div = 0.0
    gas_flux_div = 0.0
    water_flux_div = 0.0

    # Define neighbors: (neighbor_index, geometric_factor, (oil_mob, water_mob, gas_mob))
    # East: (i+1, j, k)
    ni, nj, nk = i + 1, j, k
    geometric_factor = dy * dz / dx

    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid_x[ni, nj, nk], water_mobility_grid_x[i, j, k]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid_x[ni, nj, nk], oil_mobility_grid_x[i, j, k]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid_x[ni, nj, nk], gas_mobility_grid_x[i, j, k]
    )

    total_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )
    if total_mobility > 0.0:
        pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
        pcow_difference = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
        pcgo_difference = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]
        elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]

        harmonic_water_density = compute_harmonic_mean(
            water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
        )
        harmonic_oil_density = compute_harmonic_mean(
            oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
        )
        harmonic_gas_density = compute_harmonic_mean(
            gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
        )

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

        water_potential_difference = (
            pressure_difference - pcow_difference + water_gravity_potential
        )
        oil_potential_difference = pressure_difference + oil_gravity_potential
        gas_potential_difference = (
            pressure_difference + pcgo_difference + gas_gravity_potential
        )

        water_flux = (
            water_harmonic_mobility * water_potential_difference * geometric_factor
        )
        oil_flux = oil_harmonic_mobility * oil_potential_difference * geometric_factor
        gas_flux = gas_harmonic_mobility * gas_potential_difference * geometric_factor

        # Accumulate fluxes multiplied by density to get mass flux divergence
        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # West: (i-1, j, k)
    ni, nj, nk = i - 1, j, k
    geometric_factor = dy * dz / dx

    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid_x[ni, nj, nk], water_mobility_grid_x[i, j, k]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid_x[ni, nj, nk], oil_mobility_grid_x[i, j, k]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid_x[ni, nj, nk], gas_mobility_grid_x[i, j, k]
    )

    total_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )
    if total_mobility > 0.0:
        pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
        pcow_difference = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
        pcgo_difference = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]
        elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]

        harmonic_water_density = compute_harmonic_mean(
            water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
        )
        harmonic_oil_density = compute_harmonic_mean(
            oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
        )
        harmonic_gas_density = compute_harmonic_mean(
            gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
        )

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

        water_potential_difference = (
            pressure_difference - pcow_difference + water_gravity_potential
        )
        oil_potential_difference = pressure_difference + oil_gravity_potential
        gas_potential_difference = (
            pressure_difference + pcgo_difference + gas_gravity_potential
        )

        water_flux = (
            water_harmonic_mobility * water_potential_difference * geometric_factor
        )
        oil_flux = oil_harmonic_mobility * oil_potential_difference * geometric_factor
        gas_flux = gas_harmonic_mobility * gas_potential_difference * geometric_factor

        # Accumulate fluxes multiplied by density to get mass flux divergence
        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # North: (i, j-1, k)
    ni, nj, nk = i, j - 1, k
    geometric_factor = dx * dz / dy

    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid_y[ni, nj, nk], water_mobility_grid_y[i, j, k]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid_y[ni, nj, nk], oil_mobility_grid_y[i, j, k]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid_y[ni, nj, nk], gas_mobility_grid_y[i, j, k]
    )

    total_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )
    if total_mobility > 0.0:
        pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
        pcow_difference = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
        pcgo_difference = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]
        elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]

        harmonic_water_density = compute_harmonic_mean(
            water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
        )
        harmonic_oil_density = compute_harmonic_mean(
            oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
        )
        harmonic_gas_density = compute_harmonic_mean(
            gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
        )

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

        water_potential_difference = (
            pressure_difference - pcow_difference + water_gravity_potential
        )
        oil_potential_difference = pressure_difference + oil_gravity_potential
        gas_potential_difference = (
            pressure_difference + pcgo_difference + gas_gravity_potential
        )

        water_flux = (
            water_harmonic_mobility * water_potential_difference * geometric_factor
        )
        oil_flux = oil_harmonic_mobility * oil_potential_difference * geometric_factor
        gas_flux = gas_harmonic_mobility * gas_potential_difference * geometric_factor

        # Accumulate fluxes multiplied by density to get mass flux divergence
        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # South: (i, j+1, k)
    ni, nj, nk = i, j + 1, k
    geometric_factor = dx * dz / dy

    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid_y[ni, nj, nk], water_mobility_grid_y[i, j, k]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid_y[ni, nj, nk], oil_mobility_grid_y[i, j, k]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid_y[ni, nj, nk], gas_mobility_grid_y[i, j, k]
    )

    total_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )
    if total_mobility > 0.0:
        pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
        pcow_difference = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
        pcgo_difference = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]
        elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]

        harmonic_water_density = compute_harmonic_mean(
            water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
        )
        harmonic_oil_density = compute_harmonic_mean(
            oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
        )
        harmonic_gas_density = compute_harmonic_mean(
            gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
        )

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

        water_potential_difference = (
            pressure_difference - pcow_difference + water_gravity_potential
        )
        oil_potential_difference = pressure_difference + oil_gravity_potential
        gas_potential_difference = (
            pressure_difference + pcgo_difference + gas_gravity_potential
        )

        water_flux = (
            water_harmonic_mobility * water_potential_difference * geometric_factor
        )
        oil_flux = oil_harmonic_mobility * oil_potential_difference * geometric_factor
        gas_flux = gas_harmonic_mobility * gas_potential_difference * geometric_factor

        # Accumulate fluxes multiplied by density to get mass flux divergence
        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # Top: (i, j, k-1)
    ni, nj, nk = i, j, k - 1
    geometric_factor = dx * dy / dz

    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid_z[ni, nj, nk], water_mobility_grid_z[i, j, k]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid_z[ni, nj, nk], oil_mobility_grid_z[i, j, k]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid_z[ni, nj, nk], gas_mobility_grid_z[i, j, k]
    )

    total_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )
    if total_mobility > 0.0:
        pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
        pcow_difference = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
        pcgo_difference = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]
        elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]

        harmonic_water_density = compute_harmonic_mean(
            water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
        )
        harmonic_oil_density = compute_harmonic_mean(
            oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
        )
        harmonic_gas_density = compute_harmonic_mean(
            gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
        )

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

        water_potential_difference = (
            pressure_difference - pcow_difference + water_gravity_potential
        )
        oil_potential_difference = pressure_difference + oil_gravity_potential
        gas_potential_difference = (
            pressure_difference + pcgo_difference + gas_gravity_potential
        )

        water_flux = (
            water_harmonic_mobility * water_potential_difference * geometric_factor
        )
        oil_flux = oil_harmonic_mobility * oil_potential_difference * geometric_factor
        gas_flux = gas_harmonic_mobility * gas_potential_difference * geometric_factor

        # Accumulate fluxes multiplied by density to get mass flux divergence
        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # Bottom: (i, j, k+1)
    ni, nj, nk = i, j, k + 1
    geometric_factor = dx * dy / dz

    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid_z[ni, nj, nk], water_mobility_grid_z[i, j, k]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid_z[ni, nj, nk], oil_mobility_grid_z[i, j, k]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid_z[ni, nj, nk], gas_mobility_grid_z[i, j, k]
    )

    total_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )
    if total_mobility > 0.0:
        pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
        pcow_difference = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
        pcgo_difference = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]
        elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]

        harmonic_water_density = compute_harmonic_mean(
            water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
        )
        harmonic_oil_density = compute_harmonic_mean(
            oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
        )
        harmonic_gas_density = compute_harmonic_mean(
            gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
        )

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

        water_potential_difference = (
            pressure_difference - pcow_difference + water_gravity_potential
        )
        oil_potential_difference = pressure_difference + oil_gravity_potential
        gas_potential_difference = (
            pressure_difference + pcgo_difference + gas_gravity_potential
        )

        water_flux = (
            water_harmonic_mobility * water_potential_difference * geometric_factor
        )
        oil_flux = oil_harmonic_mobility * oil_potential_difference * geometric_factor
        gas_flux = gas_harmonic_mobility * gas_potential_difference * geometric_factor

        # Accumulate fluxes multiplied by density to get mass flux divergence
        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    return oil_flux_div, gas_flux_div, water_flux_div


@numba.njit(cache=True)
def compute_residuals(
    i: int,
    j: int,
    k: int,
    pressure_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    old_oil_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
    water_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    pcow_grid: ThreeDimensionalGrid,
    pcgo_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    oil_well_rate_grid: ThreeDimensionalGrid,
    gas_well_rate_grid: ThreeDimensionalGrid,
    water_well_rate_grid: ThreeDimensionalGrid,
    acceleration_due_to_gravity_ft_per_s2: float,
) -> typing.Tuple[float, float, float]:
    """
    Compute residuals for oil, gas, and water mass balance equations at a cell.

    Residual = (Mass_new - Mass_old) - Δt x (Flow_in - Flow_out + Wells)

    When residual = 0, the equation is satisfied.

    :param i: x index
    :param j: y index
    :param k: z index
    :param pressure_grid: Pressure grid (psi)
    :param oil_saturation_grid: Current oil saturation grid
    :param gas_saturation_grid: Current gas saturation grid
    :param water_saturation_grid: Current water saturation grid
    :param old_oil_saturation_grid: Previous oil saturation grid
    :param old_gas_saturation_grid: Previous gas saturation grid
    :param old_water_saturation_grid: Previous water saturation grid
    :param oil_mobility_grid_x: Oil mobility in x direction (ft²/(psi·day))
    :param oil_mobility_grid_y: Oil mobility in y direction (ft²/(psi·day))
    :param oil_mobility_grid_z: Oil mobility in z direction (ft²/(psi·day))
    :param gas_mobility_grid_x: Gas mobility in x direction (ft²/(psi·day))
    :param gas_mobility_grid_y: Gas mobility in y direction (ft²/(psi·day))
    :param gas_mobility_grid_z: Gas mobility in z direction (ft²/(psi·day))
    :param water_mobility_grid_x: Water mobility in x direction (ft²/(psi·day))
    :param water_mobility_grid_y: Water mobility in y direction (ft²/(psi·day))
    :param water_mobility_grid_z: Water mobility in z direction (ft²/(psi·day))
    :param oil_density_grid: Oil density grid (lbm/ft³)
    :param gas_density_grid: Gas density grid (lbm/ft³)
    :param water_density_grid: Water density grid (lbm/ft³)
    :param pcow_grid: Oil-water capillary pressure grid (psi)
    :param pcgo_grid: Gas-oil capillary pressure grid (psi)
    :param elevation_grid: Elevation grid (ft)
    :param porosity_grid: Porosity grid (-)
    :param cell_dimension: (dx, dy) in feet
    :param thickness_grid: Thickness grid (ft)
    :param time_step_in_days: Time step size in days
    :param oil_well_rate_grid: Pre-computed oil well rates (ft³/day)
    :param gas_well_rate_grid: Pre-computed gas well rates (ft³/day)
    :param water_well_rate_grid: Pre-computed water well rates (ft³/day)
    :param acceleration_due_to_gravity_ft_per_s2: Acceleration due to gravity (ft/s²)
    :return: (oil_residual, gas_residual, water_residual) in ft³
    """
    dx, dy = cell_dimension
    dz = thickness_grid[i, j, k]
    cell_volume = dx * dy * dz
    porosity = porosity_grid[i, j, k]

    # Get current state accumulation (φ·S·V) (ft³)
    oil_saturation = oil_saturation_grid[i, j, k]
    gas_saturation = gas_saturation_grid[i, j, k]
    water_saturation = water_saturation_grid[i, j, k]
    accumulation_new = compute_accumulation_terms(
        porosity=porosity,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        water_saturation=water_saturation,
        cell_volume=cell_volume,
    )

    # Get old state accumulation (φ·S·V) (ft³)
    oil_saturation_old = old_oil_saturation_grid[i, j, k]
    gas_saturation_old = old_gas_saturation_grid[i, j, k]
    water_saturation_old = old_water_saturation_grid[i, j, k]
    accumulation_old = compute_accumulation_terms(
        porosity=porosity,
        oil_saturation=oil_saturation_old,
        gas_saturation=gas_saturation_old,
        water_saturation=water_saturation_old,
        cell_volume=cell_volume,
    )

    # Get flow terms (ft³/day)
    fluxes = compute_phase_fluxes_from_neighbour(
        i=i,
        j=j,
        k=k,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        pressure_grid=pressure_grid,
        oil_mobility_grid_x=oil_mobility_grid_x,
        oil_mobility_grid_y=oil_mobility_grid_y,
        oil_mobility_grid_z=oil_mobility_grid_z,
        gas_mobility_grid_x=gas_mobility_grid_x,
        gas_mobility_grid_y=gas_mobility_grid_y,
        gas_mobility_grid_z=gas_mobility_grid_z,
        water_mobility_grid_x=water_mobility_grid_x,
        water_mobility_grid_y=water_mobility_grid_y,
        water_mobility_grid_z=water_mobility_grid_z,
        oil_density_grid=oil_density_grid,
        gas_density_grid=gas_density_grid,
        water_density_grid=water_density_grid,
        pcow_grid=pcow_grid,
        pcgo_grid=pcgo_grid,
        acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
    )

    # Get well rates from pre-computed grids
    well_rates = (
        oil_well_rate_grid[i, j, k],
        gas_well_rate_grid[i, j, k],
        water_well_rate_grid[i, j, k],
    )

    # Change in pore volume occupied = time x (net flow rate)
    # (φ·S·V)^new - (φ·S·V)^old = Δt x (Flow_in - Flow_out + Wells)
    # Residual = (φ·S·V)^new - (φ·S·V)^old - Δt x (Flow_in - Flow_out + Wells)

    # Note: flux is (Flow_in - Flow_out) in ft³/day
    # Note: well_rates is (Injection - Production) in ft³/day
    oil_residual = (accumulation_new[0] - accumulation_old[0]) - (
        time_step_in_days * (fluxes[0] + well_rates[0])
    )
    gas_residual = (accumulation_new[1] - accumulation_old[1]) - (
        time_step_in_days * (fluxes[1] + well_rates[1])
    )
    water_residual = (accumulation_new[2] - accumulation_old[2]) - (
        time_step_in_days * (fluxes[2] + well_rates[2])
    )
    return oil_residual, gas_residual, water_residual


@numba.njit(parallel=True, cache=True)
def assemble_residual_vector(
    pressure_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    old_oil_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    pcow_grid: ThreeDimensionalGrid,
    pcgo_grid: ThreeDimensionalGrid,
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
    acceleration_due_to_gravity_ft_per_s2: float,
    dtype: np.typing.DTypeLike = np.float64,
) -> np.ndarray:
    """
    Assemble residual vector for all interior cells.

    :param pressure_grid: Pressure grid (psi)
    :param oil_saturation_grid: Current oil saturation grid
    :param gas_saturation_grid: Current gas saturation grid
    :param water_saturation_grid: Current water saturation grid
    :param old_oil_saturation_grid: Previous oil saturation grid
    :param old_gas_saturation_grid: Previous gas saturation grid
    :param old_water_saturation_grid: Previous water saturation grid
    :param oil_density_grid: Oil density grid (lbm/ft³)
    :param gas_density_grid: Gas density grid (lbm/ft³)
    :param water_density_grid: Water density grid (lbm/ft³)
    :param porosity_grid: Porosity grid
    :param cell_dimension: (dx, dy) in feet
    :param thickness_grid: Thickness grid (ft)
    :param elevation_grid: Elevation grid (ft)
    :param time_step_in_days: Time step size in days
    :param pcow_grid: Oil-water capillary pressure grid (psi)
    :param pcgo_grid: Gas-oil capillary pressure grid (psi)
    :param oil_mobility_grid_x: oil mobility in x direction
    :param oil_mobility_grid_y: oil mobility in y direction
    :param oil_mobility_grid_z: oil mobility in z direction
    :param gas_mobility_grid_x: gas mobility in x direction
    :param gas_mobility_grid_y: gas mobility in y direction
    :param gas_mobility_grid_z: gas mobility in z direction
    :param water_mobility_grid_x: water mobility in x direction
    :param water_mobility_grid_y: water mobility in y direction
    :param water_mobility_grid_z: water mobility in z direction
    :param oil_well_rate_grid: oil well rates (ft³/day)
    :param gas_well_rate_grid: gas well rates (ft³/day)
    :param water_well_rate_grid: water well rates (ft³/day)
    :param acceleration_due_to_gravity_ft_per_s2: Gravitational acceleration constant
    :return: Residual vector of length 3 * num_interior_cells
    """
    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape
    interior_count_x = cell_count_x - 2
    interior_count_y = cell_count_y - 2
    interior_count_z = cell_count_z - 2
    total_unknowns = 3 * interior_count_x * interior_count_y * interior_count_z

    residual_vector = np.zeros(total_unknowns, dtype=dtype)

    # Parallel loop over all interior cells
    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                residuals = compute_residuals(
                    i=i,
                    j=j,
                    k=k,
                    pressure_grid=pressure_grid,
                    oil_saturation_grid=oil_saturation_grid,
                    gas_saturation_grid=gas_saturation_grid,
                    water_saturation_grid=water_saturation_grid,
                    old_oil_saturation_grid=old_oil_saturation_grid,
                    old_gas_saturation_grid=old_gas_saturation_grid,
                    old_water_saturation_grid=old_water_saturation_grid,
                    oil_mobility_grid_x=oil_mobility_grid_x,
                    oil_mobility_grid_y=oil_mobility_grid_y,
                    oil_mobility_grid_z=oil_mobility_grid_z,
                    gas_mobility_grid_x=gas_mobility_grid_x,
                    gas_mobility_grid_y=gas_mobility_grid_y,
                    gas_mobility_grid_z=gas_mobility_grid_z,
                    water_mobility_grid_x=water_mobility_grid_x,
                    water_mobility_grid_y=water_mobility_grid_y,
                    water_mobility_grid_z=water_mobility_grid_z,
                    oil_density_grid=oil_density_grid,
                    gas_density_grid=gas_density_grid,
                    water_density_grid=water_density_grid,
                    pcow_grid=pcow_grid,
                    pcgo_grid=pcgo_grid,
                    elevation_grid=elevation_grid,
                    porosity_grid=porosity_grid,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    time_step_in_days=time_step_in_days,
                    oil_well_rate_grid=oil_well_rate_grid,
                    gas_well_rate_grid=gas_well_rate_grid,
                    water_well_rate_grid=water_well_rate_grid,
                    acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                )

                # Convert 3D indices to 1D index for residual vector
                idx = to_1D_index_interior_only(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                residual_vector[3 * idx] = residuals[0]
                residual_vector[3 * idx + 1] = residuals[1]
                residual_vector[3 * idx + 2] = residuals[2]

    return residual_vector


def solve_newton_update_system(
    jacobian: lil_matrix,
    residual_vector: np.ndarray,
    config: Config,
    preconditioner: typing.Optional[LinearOperator] = None,
) -> typing.Tuple[np.typing.NDArray, typing.Optional[LinearOperator]]:
    """
    Solve the Newton update system J·δx = -R for the update vector δx.

    :param jacobian: Jacobian matrix J
    :param residual_vector: Residual vector R
    :param config: Config
    :param preconditioner: Optional preconditioner to use
    :return: A tuple of the update vector/array, δx and the preconditioner, M.
    """
    jacobian_csr = jacobian.tocsr()
    rhs = -residual_vector

    if preconditioner is not None:
        precon = preconditioner
    else:
        precon = config.preconditioner

    solution, M = solve_linear_system(
        A_csr=jacobian_csr,
        b=rhs,
        max_iterations=config.max_iterations,
        solver=config.iterative_solver,
        preconditioner=precon,
        rtol=1e-3,  # Use a looser tolerance for the inner linear solve (Truncated Newton approach)
    )
    return solution, M


@numba.njit(parallel=True, cache=True)
def _apply_newton_update(
    pressure_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    update_vector: np.ndarray,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    damping_factor: float,
    min_valid_pressure: float,
    max_valid_pressure: float,
) -> typing.Tuple[
    ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid, float, float
]:
    """
    Apply Newton update with saturation and pressure clamping to ensure physical constraints (parallelized).

    :param pressure_grid: Current pressure grid
    :param oil_saturation_grid: Current oil saturation grid
    :param gas_saturation_grid: Current gas saturation grid
    :param update_vector: Update vector δx from linear solve
    :param cell_count_x: Number of cells in x (including padding)
    :param cell_count_y: Number of cells in y (including padding)
    :param cell_count_z: Number of cells in z (including padding)
    :param damping_factor: Damping factor for update (0 < α ≤ 1)
    :param min_valid_pressure: Minimum valid pressure (psi)
    :param max_valid_pressure: Maximum valid pressure (psi)
    :return: Tuple of (new_pressure_grid, new_oil_saturation_grid, new_gas_saturation_grid, max_pressure_change, max_saturation_change)
    """
    new_pressure_grid = pressure_grid.copy()
    new_oil_saturation_grid = oil_saturation_grid.copy()
    new_gas_saturation_grid = gas_saturation_grid.copy()

    max_pressure_change = 0.0
    max_saturation_change = 0.0

    # Parallel over interior cells
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
                if cell_1D_index < 0:
                    continue

                # Get damped updates
                pressure_update = damping_factor * update_vector[3 * cell_1D_index]
                oil_saturation_update = (
                    damping_factor * update_vector[3 * cell_1D_index + 1]
                )
                gas_saturation_update = (
                    damping_factor * update_vector[3 * cell_1D_index + 2]
                )

                # Apply updates
                new_pressure = pressure_grid[i, j, k] + pressure_update
                new_oil_saturation = (
                    oil_saturation_grid[i, j, k] + oil_saturation_update
                )
                new_gas_saturation = (
                    gas_saturation_grid[i, j, k] + gas_saturation_update
                )

                # Clamp to physical bounds
                new_pressure = clip(
                    new_pressure, min_valid_pressure, max_valid_pressure
                )
                new_oil_saturation = clip(new_oil_saturation, 0.0, 1.0)
                new_gas_saturation = clip(new_gas_saturation, 0.0, 1.0)

                # Enforce saturation constraint: So + Sg + Sw = 1
                # If So + Sg > 1, scale them down proportionally
                saturation_sum = new_oil_saturation + new_gas_saturation
                if saturation_sum > 1.0:
                    scale_factor = 1.0 / saturation_sum
                    new_oil_saturation *= scale_factor
                    new_gas_saturation *= scale_factor

                new_pressure_grid[i, j, k] = new_pressure
                new_oil_saturation_grid[i, j, k] = new_oil_saturation
                new_gas_saturation_grid[i, j, k] = new_gas_saturation

                # Track maximum changes - use consistent binary max for reduction
                abs_pressure_update = abs(pressure_update)
                abs_oil_sat_update = abs(oil_saturation_update)
                abs_gas_sat_update = abs(gas_saturation_update)

                max_pressure_change = max(max_pressure_change, abs_pressure_update)
                max_saturation_change = max(max_saturation_change, abs_oil_sat_update)
                max_saturation_change = max(max_saturation_change, abs_gas_sat_update)

    return (
        new_pressure_grid,
        new_oil_saturation_grid,
        new_gas_saturation_grid,
        max_pressure_change,
        max_saturation_change,
    )


def apply_newton_update(
    pressure_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    update_vector: np.ndarray,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    iteration: int,
    damping_factor: float = 1.0,
) -> IterationState:
    """
    Apply Newton update with saturation and pressure clamping to ensure physical constraints.

    :param pressure_grid: Current pressure grid
    :param oil_saturation_grid: Current oil saturation grid
    :param gas_saturation_grid: Current gas saturation grid
    :param update_vector: Update vector δx from linear solve
    :param cell_count_x: Number of cells in x (including padding)
    :param cell_count_y: Number of cells in y (including padding)
    :param cell_count_z: Number of cells in z (including padding)
    :param iteration: Current Newton iteration number
    :param damping_factor: Damping factor for update (0 < α ≤ 1).
        This can be used to improve convergence in challenging scenarios.
    :return: `IterationState` with updated grids
    """
    (
        new_pressure_grid,
        new_oil_saturation_grid,
        new_gas_saturation_grid,
        max_pressure_change,
        max_saturation_change,
    ) = _apply_newton_update(
        pressure_grid=pressure_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        update_vector=update_vector,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        damping_factor=damping_factor,
        min_valid_pressure=c.MIN_VALID_PRESSURE,
        max_valid_pressure=c.MAX_VALID_PRESSURE,
    )
    logger.info(
        f"Iteration {iteration}: damping={damping_factor:.4f}, "
        f"max_dP={max_pressure_change:.4e} psi, "
        f"max_dS={max_saturation_change:.4e}"
    )
    return IterationState(
        pressure_grid=new_pressure_grid,
        oil_saturation_grid=new_oil_saturation_grid,
        gas_saturation_grid=new_gas_saturation_grid,
        iteration=iteration,
        max_saturation_change=max_saturation_change,
        max_pressure_change=max_pressure_change,
    )
