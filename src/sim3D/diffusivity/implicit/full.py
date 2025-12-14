"""
Fully implicit solver for simultaneous pressure and saturation evolution.
"""

import functools
import itertools
import logging
import typing

import attrs
import numba
import numpy as np
from scipy.sparse import lil_matrix

from sim3D._precision import get_dtype
from sim3D.boundaries import BoundaryConditions
from sim3D.constants import c
from sim3D.diffusivity.base import (
    EvolutionResult,
    compute_mobility_grids,
    solve_linear_system,
    to_1D_index_interior_only,
)
from sim3D.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from sim3D.helpers import (
    apply_boundary_conditions,
    build_rock_fluid_properties_grids,
    update_phase_densities,
    update_pvt_properties,
)
from sim3D.models import FluidProperties, RockFluidProperties, RockProperties
from sim3D.pvt.core import compute_harmonic_mean, compute_harmonic_mobility
from sim3D.types import (
    FluidPhase,
    Options,
    SupportsSetItem,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from sim3D.utils import clip
from sim3D.wells import Wells

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
    """Current state during Newton-Raphson iteration."""

    pressure_grid: ThreeDimensionalGrid
    oil_saturation_grid: ThreeDimensionalGrid
    gas_saturation_grid: ThreeDimensionalGrid
    iteration: int
    residual_norm: float


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
    options: Options,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
) -> EvolutionResult[ImplicitSolution]:
    """
    Solve fully implicit finite-difference equations for pressure and saturations (optimized with Numba JIT).

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
    :param options: Solver options including tolerance and max iterations
    :param boundary_conditions: Boundary conditions
    :param injection_grid: Object supporting setitem to set cell injection rates for each phase in ft³/day.
    :param production_grid: Object supporting setitem to set cell production rates for each phase in ft³/day.
    :return: `EvolutionResult` containing `ImplicitSolution` with updated pressure and saturations
    """
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    old_fluid_properties = fluid_properties
    pressure_grid = fluid_properties.pressure_grid.copy()
    oil_saturation_grid = fluid_properties.oil_saturation_grid.copy()
    gas_saturation_grid = fluid_properties.gas_saturation_grid.copy()
    water_saturation_grid = fluid_properties.water_saturation_grid.copy()

    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape

    max_newton_iterations = options.max_iterations
    convergence_tolerance = 1e-6
    converged = False
    iteration = 0
    initial_residual_norm = 0.0
    final_residual_norm = 0.0
    dtype = get_dtype()

    if not np.all(np.isfinite(pressure_grid)):
        logger.error("Initial pressure grid contains non-finite values")
        raise ValueError("Initial pressure grid contains NaN or Inf values")

    if not np.all((oil_saturation_grid >= 0) & (oil_saturation_grid <= 1)):
        logger.warning(
            f"Oil saturation out of bounds: min={oil_saturation_grid.min()}, max={oil_saturation_grid.max()}"
        )

    if not np.all((gas_saturation_grid >= 0) & (gas_saturation_grid <= 1)):
        logger.warning(
            f"Gas saturation out of bounds: min={gas_saturation_grid.min()}, max={gas_saturation_grid.max()}"
        )

    md_per_centipoise_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    acceleration_due_to_gravity_ft_per_s2 = c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
    for iteration in range(max_newton_iterations):
        logger.debug(
            "Pre-computing mobility grids and well rates for iteration %d...", iteration
        )
        mobility_grids = compute_mobility_grids(
            absolute_permeability_x=rock_properties.absolute_permeability.x,
            absolute_permeability_y=rock_properties.absolute_permeability.y,
            absolute_permeability_z=rock_properties.absolute_permeability.z,
            water_relative_mobility_grid=relative_mobility_grids.water_relative_mobility,
            oil_relative_mobility_grid=relative_mobility_grids.oil_relative_mobility,
            gas_relative_mobility_grid=relative_mobility_grids.gas_relative_mobility,
            millidarcies_per_centipoise_to_ft2_per_psi_per_day=md_per_centipoise_to_ft2_per_psi_per_day,
        )
        well_rate_grids = compute_well_rate_grids(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            wells=wells,
            pressure_grid=fluid_properties.pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            absolute_permeability_x=rock_properties.absolute_permeability.x,
            absolute_permeability_y=rock_properties.absolute_permeability.y,
            absolute_permeability_z=rock_properties.absolute_permeability.z,
            fluid_properties=fluid_properties,
            relative_mobility_grids=relative_mobility_grids,
            options=options,
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

        residual_vector = assemble_residual_vector(
            pressure_grid=fluid_properties.pressure_grid,
            oil_saturation_grid=fluid_properties.oil_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            old_oil_saturation_grid=old_fluid_properties.oil_saturation_grid,
            old_gas_saturation_grid=old_fluid_properties.gas_saturation_grid,
            old_water_saturation_grid=old_fluid_properties.water_saturation_grid,
            oil_density_grid=fluid_properties.oil_effective_density_grid,
            gas_density_grid=fluid_properties.gas_density_grid,
            water_density_grid=fluid_properties.water_density_grid,
            porosity_grid=rock_properties.porosity_grid,
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
        final_residual_norm = float(np.linalg.norm(residual_vector))

        # Log first iteration residual for diagnostics
        if iteration == 0:
            initial_residual_norm = final_residual_norm
            logger.info("=" * 60)
            logger.info("INITIAL STATE DIAGNOSTICS")
            logger.info("=" * 60)
            logger.info(f"Residual norm: {final_residual_norm:.6e}")
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
            logger.info("=" * 60)

        # Check for NaN or Inf in residuals
        if not np.isfinite(final_residual_norm):
            logger.error(
                f"Non-finite residual norm at iteration {iteration}: {final_residual_norm}. "
                f"NaN count: {np.isnan(residual_vector).sum()}, "
                f"Inf count: {np.isinf(residual_vector).sum()}"
            )
            break

        # Use both absolute and relative criteria
        relative_reduction = (
            final_residual_norm / initial_residual_norm
            if initial_residual_norm > 0
            else 0
        )
        logger.info(
            f"Iteration {iteration}: Residual norm = {final_residual_norm:.2e}, "
            f"Relative reduction = {relative_reduction:.2e}"
        )
        if (final_residual_norm < convergence_tolerance) or (relative_reduction < 1e-3):
            converged = True
            break

        jacobian = assemble_jacobian(
            pressure_grid=pressure_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_size=time_step_in_days,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            old_fluid_properties=old_fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
            capillary_pressure_grids=capillary_pressure_grids,
            wells=wells,
            options=options,
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
        )
        jacobian_csr = jacobian.tocsr()
        diag = jacobian_csr.diagonal()
        logger.info(
            f"Jacobian diagonal: min={np.min(np.abs(diag[diag != 0])):.2e}, max={np.max(np.abs(diag)):.2e}"
        )
        logger.info(
            f"Jacobian: {np.sum(diag == 0)} zero diagonal entries out of {len(diag)}"
        )
        logger.info(f"Jacobian nnz: {jacobian_csr.nnz}")

        update_vector = solve_newton_update_system(
            jacobian=jacobian,
            residual_vector=residual_vector,
            options=options,
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
        )

        pressure_grid = iteration_state.pressure_grid
        oil_saturation_grid = iteration_state.oil_saturation_grid
        gas_saturation_grid = iteration_state.gas_saturation_grid
        water_saturation_grid = 1.0 - oil_saturation_grid - gas_saturation_grid

        # Update fluid properties for next iteration with new pressure and saturations
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
            grid_shape=pressure_grid.shape,
            thickness_grid=thickness_grid,
            time=time_step * time_step_size,
        )
        # Update other PVT properties based on new pressure and saturations
        fluid_properties = update_pvt_properties(
            fluid_properties=fluid_properties,
            wells=wells,
            miscibility_model=options.miscibility_model,
        )
        # Rebuild rock-fluid property grids with updated saturations
        (
            _,
            relative_mobility_grids,
            capillary_pressure_grids,
        ) = build_rock_fluid_properties_grids(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            rock_fluid_properties=rock_fluid_properties,
            options=options,
        )

    solution = ImplicitSolution(
        pressure_grid=pressure_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        water_saturation_grid=water_saturation_grid,
        converged=converged,
        newton_iterations=iteration + 1,
        final_residual_norm=final_residual_norm,
    )
    return EvolutionResult(value=solution, scheme="implicit")


@numba.njit(cache=True, fastmath=True)
def make_saturation_perturbation(
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    i: int,
    j: int,
    k: int,
    oil_saturation_delta: float = 0.0,
    gas_saturation_delta: float = 0.0,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Perturbs oil and/or gas saturation at cell (i, j, k) and updates water
    saturation to maintain Sw = 1 - So - Sg constraint. Clamps all saturations
    to [eps, 1-eps] and renormalizes if needed.

    :param oil_saturation_grid: Oil saturation grid (not modified in-place)
    :param gas_saturation_grid: Gas saturation grid (not modified in-place)
    :param water_saturation_grid: Water saturation grid (not modified in-place)
    :param i: x-index of cell to perturb
    :param j: y-index of cell to perturb
    :param k: z-index of cell to perturb
    :param oil_saturation_delta: Perturbation to add to oil saturation
    :param gas_saturation_delta: Perturbation to add to gas saturation
    :return: Tuple of (perturbed_so_grid, perturbed_sg_grid, perturbed_sw_grid)
    """
    eps = 1e-12
    # Create copies
    so_perturbed = oil_saturation_grid.copy()
    sg_perturbed = gas_saturation_grid.copy()
    sw_perturbed = water_saturation_grid.copy()

    # Apply perturbations
    so_perturbed[i, j, k] += oil_saturation_delta
    sg_perturbed[i, j, k] += gas_saturation_delta

    # Update water saturation to maintain constraint
    sw_perturbed[i, j, k] = 1.0 - so_perturbed[i, j, k] - sg_perturbed[i, j, k]

    # Clamp individual saturations
    so_perturbed[i, j, k] = clip(so_perturbed[i, j, k], eps, 1.0 - eps)
    sg_perturbed[i, j, k] = clip(sg_perturbed[i, j, k], eps, 1.0 - eps)
    sw_perturbed[i, j, k] = clip(sw_perturbed[i, j, k], eps, 1.0 - eps)

    # Renormalize if sum != 1 due to clamping
    saturation_sum = (
        so_perturbed[i, j, k] + sg_perturbed[i, j, k] + sw_perturbed[i, j, k]
    )
    if abs(saturation_sum - 1.0) > 1e-10:
        so_perturbed[i, j, k] /= saturation_sum
        sg_perturbed[i, j, k] /= saturation_sum
        sw_perturbed[i, j, k] /= saturation_sum
    return so_perturbed, sg_perturbed, sw_perturbed


@numba.njit(cache=True, fastmath=True)
def compute_perturbation_magnitude_for_saturation(
    value: float, min_delta: float = 1e-8, rel_step: float = 1e-6
) -> float:
    """
    Computes adaptive perturbation magnitude for saturation finite differences.

    Uses relative perturbation scaled by value magnitude with floor for small values.
    Returns a perturbation that is neither too tiny (round-off) nor too large
    (nonlinear truncation error).

    :param value: Current saturation value
    :param min_delta: Minimum absolute perturbation
    :param rel_step: Relative step size for larger values
    :return: Perturbation magnitude
    """
    # relative step ~ 1e-6 of typical saturation scale, but saturations in [0,1]
    step = max(min_delta, rel_step * max(abs(value), 1.0))
    return float(step)


@numba.njit(cache=True, fastmath=True)
def compute_perturbation_magnitude_for_pressure(
    value: float, min_delta: float = 1e-3, rel_step: float = 1e-6
) -> float:
    """
    Computes adaptive perturbation magnitude for pressure finite differences (psi).

    Uses relative perturbation scaled by magnitude with a sensible floor. For
    small pressures we still take a small absolute step.

    :param value: Current pressure value
    :param min_delta: Minimum absolute perturbation
    :param rel_step: Relative step size for larger values
    :return: Perturbation magnitude
    """
    step = max(min_delta, rel_step * max(abs(value), 1.0))
    return float(step)


@numba.njit(cache=True, fastmath=True)
def compute_accumulation_terms(
    porosity: float,
    oil_saturation: float,
    gas_saturation: float,
    water_saturation: float,
    cell_volume: float = 1.0,
) -> typing.Tuple[float, float, float]:
    """
    Compute accumulation terms for oil, gas, and water phases in reservoir conditions.

    Computes φ·S·V for each phase, representing the volume of each phase in the pore space.

    :param porosity: Porosity at cell (dimensionless)
    :param oil_saturation: Oil saturation at cell
    :param gas_saturation: Gas saturation at cell
    :param water_saturation: Water saturation at cell
    :param cell_volume: Cell volume in ft³ (default 1.0)
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
    options: Options,
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    dtype: np.typing.DTypeLike = np.float64,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Pre-compute well rate contributions for all cells.

    This function handles injection and production wells

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
    :param options: Evolution options
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
            elif injected_phase == FluidPhase.WATER:
                phase_mobility = relative_mobility_grids.water_relative_mobility[
                    i, j, k
                ]
                compressibility_kwargs = {
                    "bubble_point_pressure": fluid_properties.oil_bubble_point_pressure_grid[
                        i, j, k
                    ],
                    "gas_formation_volume_factor": fluid_properties.gas_formation_volume_factor_grid[
                        i, j, k
                    ],
                    "gas_solubility_in_water": fluid_properties.gas_solubility_in_water_grid[
                        i, j, k
                    ],
                }
            else:
                phase_mobility = relative_mobility_grids.oil_relative_mobility[i, j, k]
                compressibility_kwargs = {}

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
                options.use_pseudo_pressure and injected_phase == FluidPhase.GAS
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
            )
            if injected_phase != FluidPhase.GAS:
                cell_injection_rate *= c.BBL_TO_FT3

            # Volumetric rates at reservoir conditions (ft³/day)
            if injected_phase == FluidPhase.GAS:
                gas_rate += cell_injection_rate
                cell_gas_injection_rate = cell_injection_rate
            elif injected_phase == FluidPhase.WATER:
                water_rate += cell_injection_rate
                cell_water_injection_rate = cell_injection_rate
            else:
                oil_rate += cell_injection_rate
                cell_oil_injection_rate = cell_injection_rate

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
                    options.use_pseudo_pressure and produced_phase == FluidPhase.GAS
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

        # Store well rates for this cell
        oil_well_rate_grid[i, j, k] = oil_rate
        gas_well_rate_grid[i, j, k] = gas_rate
        water_well_rate_grid[i, j, k] = water_rate

    return oil_well_rate_grid, gas_well_rate_grid, water_well_rate_grid


@numba.njit(cache=True, fastmath=True)
def compute_flux_divergence_for_cell(
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

    water_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=water_mobility_grid_x
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=oil_mobility_grid_x
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=gas_mobility_grid_x
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

        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # West: (i-1, j, k)
    ni, nj, nk = i - 1, j, k
    geometric_factor = dy * dz / dx

    water_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=water_mobility_grid_x
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=oil_mobility_grid_x
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=gas_mobility_grid_x
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

        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # North: (i, j-1, k)
    ni, nj, nk = i, j - 1, k
    geometric_factor = dx * dz / dy

    water_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=water_mobility_grid_y
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=oil_mobility_grid_y
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=gas_mobility_grid_y
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

        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # South: (i, j+1, k)
    ni, nj, nk = i, j + 1, k
    geometric_factor = dx * dz / dy

    water_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=water_mobility_grid_y
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=oil_mobility_grid_y
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=gas_mobility_grid_y
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

        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # Top: (i, j, k-1)
    ni, nj, nk = i, j, k - 1
    geometric_factor = dx * dy / dz

    water_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=water_mobility_grid_z
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=oil_mobility_grid_z
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=gas_mobility_grid_z
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

        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    # Bottom: (i, j, k+1)
    ni, nj, nk = i, j, k + 1
    geometric_factor = dx * dy / dz

    water_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=water_mobility_grid_z
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=oil_mobility_grid_z
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=(i, j, k), index2=(ni, nj, nk), mobility_grid=gas_mobility_grid_z
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

        oil_flux_div += oil_flux
        gas_flux_div += gas_flux
        water_flux_div += water_flux

    return oil_flux_div, gas_flux_div, water_flux_div


@numba.njit(cache=True, fastmath=True)
def compute_residuals_for_cell(
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
    flux_divergence = compute_flux_divergence_for_cell(
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

    # Get well rates from pre-computed grids (no method calls!)
    well_rates = (
        oil_well_rate_grid[i, j, k],
        gas_well_rate_grid[i, j, k],
        water_well_rate_grid[i, j, k],
    )

    # Change in pore volume occupied = time x (net flow rate)
    # (φ·S·V)^new - (φ·S·V)^old = Δt x (Flow_in - Flow_out + Wells)
    # Residual = (φ·S·V)^new - (φ·S·V)^old - Δt x (Flow_in - Flow_out + Wells)

    # Note: flux_divergence is (Flow_in - Flow_out) in ft³/day
    # Note: well_rates is (Injection - Production) in ft³/day
    oil_residual = (accumulation_new[0] - accumulation_old[0]) - (
        time_step_in_days * (flux_divergence[0] + well_rates[0])
    )
    gas_residual = (accumulation_new[1] - accumulation_old[1]) - (
        time_step_in_days * (flux_divergence[1] + well_rates[1])
    )
    water_residual = (accumulation_new[2] - accumulation_old[2]) - (
        time_step_in_days * (flux_divergence[2] + well_rates[2])
    )
    return oil_residual, gas_residual, water_residual


@numba.njit(parallel=True, cache=True, fastmath=True)
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

    Uses parallel processing for maximum performance.

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
                residuals = compute_residuals_for_cell(
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


def assemble_jacobian(
    pressure_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    old_fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    options: Options,
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
    max_retries: int = 6,
) -> lil_matrix:
    """
    Assemble the Jacobian matrix J = ∂R/∂x using numerical differentiation.
    Includes off-diagonal terms for neighbor cell coupling.

    :param pressure_grid: Pressure grid
    :param oil_saturation_grid: Oil saturation grid
    :param gas_saturation_grid: Gas saturation grid
    :param water_saturation_grid: Water saturation grid
    :param cell_dimension: (dx, dy) in feet
    :param thickness_grid: Thickness grid
    :param elevation_grid: Elevation grid
    :param time_step_size: Time step size in days
    :param rock_properties: Rock properties
    :param fluid_properties: Fluid properties at current time
    :param old_fluid_properties: Fluid properties at previous time
    :param rock_fluid_properties: Rock-fluid properties
    :param relative_mobility_grids: Relative mobility grids
    :param capillary_pressure_grids: Capillary pressure grids
    :param wells: Wells object
    :param options: Options
    :param max_retries: Maximum retries for finite difference computation
    :return: Jacobian matrix as sparse `lil_matrix`
    """
    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape
    interior_count_x = cell_count_x - 2
    interior_count_y = cell_count_y - 2
    interior_count_z = cell_count_z - 2
    total_unknowns = 3 * interior_count_x * interior_count_y * interior_count_z

    dtype = get_dtype()
    jacobian = lil_matrix((total_unknowns, total_unknowns), dtype=dtype)
    acceleration_due_to_gravity_ft_per_s2 = c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2

    def _compute_residual(
        i: int,
        j: int,
        k: int,
        perturbed_pressure_grid: ThreeDimensionalGrid,
        perturbed_oil_saturation_grid: ThreeDimensionalGrid,
        perturbed_gas_saturation_grid: ThreeDimensionalGrid,
        perturbed_water_saturation_grid: ThreeDimensionalGrid,
        perturbed_oil_density_grid: ThreeDimensionalGrid,
        perturbed_gas_density_grid: ThreeDimensionalGrid,
        perturbed_water_density_grid: ThreeDimensionalGrid,
    ) -> typing.Tuple[float, float, float]:
        """
        Compute residuals using PRE-COMPUTED mobility and well rate grids.

        Uses frozen mobility approximation (quasi-Newton) for both pressure and saturation perturbations.
        This is mathematically valid and dramatically faster than recomputing mobility grids.

        Uses capillary pressure grids from outer scope (not perturbed for quasi-Newton approximation).
        """
        return compute_residuals_for_cell(
            i=i,
            j=j,
            k=k,
            pressure_grid=perturbed_pressure_grid,
            oil_saturation_grid=perturbed_oil_saturation_grid,
            gas_saturation_grid=perturbed_gas_saturation_grid,
            water_saturation_grid=perturbed_water_saturation_grid,
            old_oil_saturation_grid=old_fluid_properties.oil_saturation_grid,
            old_gas_saturation_grid=old_fluid_properties.gas_saturation_grid,
            old_water_saturation_grid=old_fluid_properties.water_saturation_grid,
            oil_mobility_grid_x=oil_mobility_grid_x,
            oil_mobility_grid_y=oil_mobility_grid_y,
            oil_mobility_grid_z=oil_mobility_grid_z,
            gas_mobility_grid_x=gas_mobility_grid_x,
            gas_mobility_grid_y=gas_mobility_grid_y,
            gas_mobility_grid_z=gas_mobility_grid_z,
            water_mobility_grid_x=water_mobility_grid_x,
            water_mobility_grid_y=water_mobility_grid_y,
            water_mobility_grid_z=water_mobility_grid_z,
            oil_density_grid=perturbed_oil_density_grid,
            gas_density_grid=perturbed_gas_density_grid,
            water_density_grid=perturbed_water_density_grid,
            pcow_grid=capillary_pressure_grids.oil_water_capillary_pressure,
            pcgo_grid=capillary_pressure_grids.gas_oil_capillary_pressure,
            elevation_grid=elevation_grid,
            porosity_grid=rock_properties.porosity_grid,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time_step_in_days=time_step_size,
            oil_well_rate_grid=oil_well_rate_grid,
            gas_well_rate_grid=gas_well_rate_grid,
            water_well_rate_grid=water_well_rate_grid,
            acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
        )

    # Get neighbor offsets
    neighbor_offsets = [
        (1, 0, 0),
        (-1, 0, 0),  # x-direction
        (0, -1, 0),
        (0, 1, 0),  # y-direction
        (0, 0, -1),
        (0, 0, 1),  # z-direction
    ]
    _to_1D_index = functools.partial(
        to_1D_index_interior_only,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )

    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_1D_index = _to_1D_index(i=i, j=j, k=k)
        if cell_1D_index < 0:
            continue

        # DIAGONAL TERMS: Derivatives with respect to current cell variables
        # Perturb pressure at current cell (central difference, retries)
        current_pressure = pressure_grid[i, j, k]
        delta_p = compute_perturbation_magnitude_for_pressure(current_pressure)

        def _compute_pressure_residuals(delta: float):
            # Plus
            pressure_plus = pressure_grid.copy()
            pressure_plus[i, j, k] = current_pressure + delta
            fp_plus = attrs.evolve(fluid_properties, pressure_grid=pressure_plus)
            fp_plus = update_phase_densities(
                fluid_properties=fp_plus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_plus = _compute_residual(
                i=i,
                j=j,
                k=k,
                perturbed_pressure_grid=fp_plus.pressure_grid,
                perturbed_oil_saturation_grid=fp_plus.oil_saturation_grid,
                perturbed_gas_saturation_grid=fp_plus.gas_saturation_grid,
                perturbed_water_saturation_grid=fp_plus.water_saturation_grid,
                perturbed_oil_density_grid=fp_plus.oil_effective_density_grid,
                perturbed_gas_density_grid=fp_plus.gas_density_grid,
                perturbed_water_density_grid=fp_plus.water_density_grid,
            )

            # Minus
            pressure_minus = pressure_grid.copy()
            pressure_minus[i, j, k] = current_pressure - delta
            fp_minus = attrs.evolve(fluid_properties, pressure_grid=pressure_minus)
            fp_minus = update_phase_densities(
                fluid_properties=fp_minus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_minus = _compute_residual(
                i=i,
                j=j,
                k=k,
                perturbed_pressure_grid=fp_minus.pressure_grid,
                perturbed_oil_saturation_grid=fp_minus.oil_saturation_grid,
                perturbed_gas_saturation_grid=fp_minus.gas_saturation_grid,
                perturbed_water_saturation_grid=fp_minus.water_saturation_grid,
                perturbed_oil_density_grid=fp_minus.oil_effective_density_grid,
                perturbed_gas_density_grid=fp_minus.gas_density_grid,
                perturbed_water_density_grid=fp_minus.water_density_grid,
            )
            r_plus = np.asarray(r_plus, dtype=dtype).ravel()
            r_minus = np.asarray(r_minus, dtype=dtype).ravel()
            if r_plus.shape != r_minus.shape:
                raise ValueError(
                    f"Shape mismatch: r_plus={r_plus.shape}, r_minus={r_minus.shape}"
                )
            if r_plus.size != 3:
                raise ValueError(f"Expected 3 residuals, got {r_plus.size}")
            return r_plus, r_minus

        attempt = 0
        success = False
        derivs = None
        while attempt < max_retries and not success:
            r_plus, r_minus = _compute_pressure_residuals(delta_p)
            derivs = (r_plus - r_minus) / (2.0 * delta_p)
            if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                success = True
            else:
                delta_p *= 0.5
                attempt += 1

        if derivs is None or not success:
            # fail fast and log context
            logger.error(
                "Pressure FD failed at cell (%d,%d,%d): last delta=%.3e, max(|deriv|)=%.3e",
                i,
                j,
                k,
                delta_p,
                np.max(np.abs(derivs)) if derivs is not None else np.nan,
            )
            raise RuntimeError(
                f"Failed to compute finite-difference pressure derivatives at ({i},{j},{k})"
            )

        # assign diagonal derivatives
        for idx in range(3):
            jacobian[3 * cell_1D_index + idx, 3 * cell_1D_index] = float(derivs[idx])

        # Perturb oil saturation at current cell (central FD, keep So+Sg+Sw constraint)
        current_oil_saturation = oil_saturation_grid[i, j, k]
        delta_so = compute_perturbation_magnitude_for_saturation(current_oil_saturation)

        def _compute_oil_saturation_residuals(delta: float):
            # Plus
            so_p, sg_for_so_p, sw_for_so_p = make_saturation_perturbation(
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
                i=i,
                j=j,
                k=k,
                oil_saturation_delta=delta,
                gas_saturation_delta=0.0,
            )
            fp_plus = attrs.evolve(
                fluid_properties,
                oil_saturation_grid=so_p,
                gas_saturation_grid=sg_for_so_p,
                water_saturation_grid=sw_for_so_p,
            )
            fp_plus = update_phase_densities(
                fluid_properties=fp_plus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_plus = _compute_residual(
                i=i,
                j=j,
                k=k,
                perturbed_pressure_grid=fp_plus.pressure_grid,
                perturbed_oil_saturation_grid=fp_plus.oil_saturation_grid,
                perturbed_gas_saturation_grid=fp_plus.gas_saturation_grid,
                perturbed_water_saturation_grid=fp_plus.water_saturation_grid,
                perturbed_oil_density_grid=fp_plus.oil_effective_density_grid,
                perturbed_gas_density_grid=fp_plus.gas_density_grid,
                perturbed_water_density_grid=fp_plus.water_density_grid,
            )

            # Minus
            so_m, sg_for_so_m, sw_for_so_m = make_saturation_perturbation(
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
                i=i,
                j=j,
                k=k,
                oil_saturation_delta=-delta,
                gas_saturation_delta=0.0,
            )
            fp_minus = attrs.evolve(
                fluid_properties,
                oil_saturation_grid=so_m,
                gas_saturation_grid=sg_for_so_m,
                water_saturation_grid=sw_for_so_m,
            )
            fp_minus = update_phase_densities(
                fluid_properties=fp_minus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_minus = _compute_residual(
                i=i,
                j=j,
                k=k,
                perturbed_pressure_grid=fp_minus.pressure_grid,
                perturbed_oil_saturation_grid=fp_minus.oil_saturation_grid,
                perturbed_gas_saturation_grid=fp_minus.gas_saturation_grid,
                perturbed_water_saturation_grid=fp_minus.water_saturation_grid,
                perturbed_oil_density_grid=fp_minus.oil_effective_density_grid,
                perturbed_gas_density_grid=fp_minus.gas_density_grid,
                perturbed_water_density_grid=fp_minus.water_density_grid,
            )
            r_plus = np.asarray(r_plus, dtype=dtype).ravel()
            r_minus = np.asarray(r_minus, dtype=dtype).ravel()
            if r_plus.shape != r_minus.shape:
                raise ValueError(
                    f"Shape mismatch: r_plus={r_plus.shape}, r_minus={r_minus.shape}"
                )
            if r_plus.size != 3:
                raise ValueError(f"Expected 3 residuals, got {r_plus.size}")
            return r_plus, r_minus

        attempt = 0
        success = False
        while attempt < max_retries and not success:
            r_plus, r_minus = _compute_oil_saturation_residuals(delta_so)
            derivs = (r_plus - r_minus) / (2.0 * delta_so)
            if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                success = True
            else:
                delta_so *= 0.5
                attempt += 1

        if not success:
            raise RuntimeError(
                f"Failed to compute finite-difference oil saturation derivatives at ({i},{j},{k})"
            )

        for idx in range(3):
            jacobian[3 * cell_1D_index + idx, 3 * cell_1D_index + 1] = float(
                derivs[idx]
            )

        # Perturb gas saturation at current cell (central FD, enforce saturations)
        current_gas_saturation = gas_saturation_grid[i, j, k]
        delta_sg = compute_perturbation_magnitude_for_saturation(current_gas_saturation)

        def _compute_gas_saturation_residuals(delta: float):
            # Plus
            so_for_sg_p, sg_p, sw_for_sg_p = make_saturation_perturbation(
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
                i=i,
                j=j,
                k=k,
                oil_saturation_delta=0.0,
                gas_saturation_delta=delta,
            )
            fp_plus = attrs.evolve(
                fluid_properties,
                oil_saturation_grid=so_for_sg_p,
                gas_saturation_grid=sg_p,
                water_saturation_grid=sw_for_sg_p,
            )
            fp_plus = update_phase_densities(
                fluid_properties=fp_plus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_plus = _compute_residual(
                i=i,
                j=j,
                k=k,
                perturbed_pressure_grid=fp_plus.pressure_grid,
                perturbed_oil_saturation_grid=fp_plus.oil_saturation_grid,
                perturbed_gas_saturation_grid=fp_plus.gas_saturation_grid,
                perturbed_water_saturation_grid=fp_plus.water_saturation_grid,
                perturbed_oil_density_grid=fp_plus.oil_effective_density_grid,
                perturbed_gas_density_grid=fp_plus.gas_density_grid,
                perturbed_water_density_grid=fp_plus.water_density_grid,
            )

            # Minus
            so_for_sg_m, sg_m, sw_for_sg_m = make_saturation_perturbation(
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
                i=i,
                j=j,
                k=k,
                oil_saturation_delta=0.0,
                gas_saturation_delta=-delta,
            )
            fp_minus = attrs.evolve(
                fluid_properties,
                oil_saturation_grid=so_for_sg_m,
                gas_saturation_grid=sg_m,
                water_saturation_grid=sw_for_sg_m,
            )
            fp_minus = update_phase_densities(
                fluid_properties=fp_minus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_minus = _compute_residual(
                i=i,
                j=j,
                k=k,
                perturbed_pressure_grid=fp_minus.pressure_grid,
                perturbed_oil_saturation_grid=fp_minus.oil_saturation_grid,
                perturbed_gas_saturation_grid=fp_minus.gas_saturation_grid,
                perturbed_water_saturation_grid=fp_minus.water_saturation_grid,
                perturbed_oil_density_grid=fp_minus.oil_effective_density_grid,
                perturbed_gas_density_grid=fp_minus.gas_density_grid,
                perturbed_water_density_grid=fp_minus.water_density_grid,
            )
            r_plus = np.asarray(r_plus, dtype=dtype).ravel()
            r_minus = np.asarray(r_minus, dtype=dtype).ravel()
            if r_plus.shape != r_minus.shape:
                raise ValueError(
                    f"Shape mismatch: r_plus={r_plus.shape}, r_minus={r_minus.shape}"
                )
            if r_plus.size != 3:
                raise ValueError(f"Expected 3 residuals, got {r_plus.size}")
            return r_plus, r_minus

        attempt = 0
        success = False
        while attempt < max_retries and not success:
            r_plus, r_minus = _compute_gas_saturation_residuals(delta_sg)
            derivs = (r_plus - r_minus) / (2.0 * delta_sg)
            if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                success = True
            else:
                delta_sg *= 0.5
                attempt += 1

        if not success:
            raise RuntimeError(
                f"Failed to compute finite-difference gas saturation derivatives at ({i},{j},{k})"
            )

        for idx in range(3):
            jacobian[3 * cell_1D_index + idx, 3 * cell_1D_index + 2] = float(
                derivs[idx]
            )

        # OFF-DIAGONAL TERMS: Derivatives with respect to neighbor cells
        for di, dj, dk in neighbor_offsets:
            ni, nj, nk = i + di, j + dj, k + dk

            # Check if neighbor is in interior region
            if not (
                1 <= ni < cell_count_x - 1
                and 1 <= nj < cell_count_y - 1
                and 1 <= nk < cell_count_z - 1
            ):
                continue

            neighbor_1D_index = _to_1D_index(i=ni, j=nj, k=nk)
            if neighbor_1D_index < 0:
                continue

            # Perturb neighbor pressure
            neighbor_pressure = pressure_grid[ni, nj, nk]
            delta_p_neighbor = compute_perturbation_magnitude_for_pressure(
                neighbor_pressure
            )

            def _compute_neighbor_pressure_residuals(delta: float):
                # Plus
                pressure_plus = pressure_grid.copy()
                pressure_plus[ni, nj, nk] = neighbor_pressure + delta
                fp_plus = attrs.evolve(fluid_properties, pressure_grid=pressure_plus)
                fp_plus = update_phase_densities(
                    fluid_properties=fp_plus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_plus = _compute_residual(
                    i=i,
                    j=j,
                    k=k,
                    perturbed_pressure_grid=fp_plus.pressure_grid,
                    perturbed_oil_saturation_grid=fp_plus.oil_saturation_grid,
                    perturbed_gas_saturation_grid=fp_plus.gas_saturation_grid,
                    perturbed_water_saturation_grid=fp_plus.water_saturation_grid,
                    perturbed_oil_density_grid=fp_plus.oil_effective_density_grid,
                    perturbed_gas_density_grid=fp_plus.gas_density_grid,
                    perturbed_water_density_grid=fp_plus.water_density_grid,
                )

                # Minus
                pressure_minus = pressure_grid.copy()
                pressure_minus[ni, nj, nk] = neighbor_pressure - delta
                fp_minus = attrs.evolve(fluid_properties, pressure_grid=pressure_minus)
                fp_minus = update_phase_densities(
                    fluid_properties=fp_minus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_minus = _compute_residual(
                    i=i,
                    j=j,
                    k=k,
                    perturbed_pressure_grid=fp_minus.pressure_grid,
                    perturbed_oil_saturation_grid=fp_minus.oil_saturation_grid,
                    perturbed_gas_saturation_grid=fp_minus.gas_saturation_grid,
                    perturbed_water_saturation_grid=fp_minus.water_saturation_grid,
                    perturbed_oil_density_grid=fp_minus.oil_effective_density_grid,
                    perturbed_gas_density_grid=fp_minus.gas_density_grid,
                    perturbed_water_density_grid=fp_minus.water_density_grid,
                )
                r_plus = np.asarray(r_plus, dtype=dtype).ravel()
                r_minus = np.asarray(r_minus, dtype=dtype).ravel()
                return r_plus, r_minus

            attempt = 0
            success = False
            derivs = None
            while attempt < max_retries and not success:
                r_plus, r_minus = _compute_neighbor_pressure_residuals(delta_p_neighbor)
                derivs = (r_plus - r_minus) / (2.0 * delta_p_neighbor)
                if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                    success = True
                else:
                    delta_p_neighbor *= 0.5
                    attempt += 1

            if success and derivs is not None:
                for idx in range(3):
                    jacobian[3 * cell_1D_index + idx, 3 * neighbor_1D_index] = float(
                        derivs[idx]
                    )

            # Perturb neighbor oil saturation
            neighbor_oil_saturation = oil_saturation_grid[ni, nj, nk]
            delta_so_neighbor = compute_perturbation_magnitude_for_saturation(
                neighbor_oil_saturation
            )

            def _compute_neighbor_oil_saturation_residuals(delta: float):
                # Plus
                so_p, sg_for_so_p, sw_for_so_p = make_saturation_perturbation(
                    oil_saturation_grid=oil_saturation_grid,
                    gas_saturation_grid=gas_saturation_grid,
                    water_saturation_grid=water_saturation_grid,
                    i=ni,
                    j=nj,
                    k=nk,
                    oil_saturation_delta=delta,
                    gas_saturation_delta=0.0,
                )
                fp_plus = attrs.evolve(
                    fluid_properties,
                    oil_saturation_grid=so_p,
                    gas_saturation_grid=sg_for_so_p,
                    water_saturation_grid=sw_for_so_p,
                )
                fp_plus = update_phase_densities(
                    fluid_properties=fp_plus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_plus = _compute_residual(
                    i=i,
                    j=j,
                    k=k,
                    perturbed_pressure_grid=fp_plus.pressure_grid,
                    perturbed_oil_saturation_grid=fp_plus.oil_saturation_grid,
                    perturbed_gas_saturation_grid=fp_plus.gas_saturation_grid,
                    perturbed_water_saturation_grid=fp_plus.water_saturation_grid,
                    perturbed_oil_density_grid=fp_plus.oil_effective_density_grid,
                    perturbed_gas_density_grid=fp_plus.gas_density_grid,
                    perturbed_water_density_grid=fp_plus.water_density_grid,
                )

                # Minus
                so_m, sg_for_so_m, sw_for_so_m = make_saturation_perturbation(
                    oil_saturation_grid=oil_saturation_grid,
                    gas_saturation_grid=gas_saturation_grid,
                    water_saturation_grid=water_saturation_grid,
                    i=ni,
                    j=nj,
                    k=nk,
                    oil_saturation_delta=-delta,
                    gas_saturation_delta=0.0,
                )
                fp_minus = attrs.evolve(
                    fluid_properties,
                    oil_saturation_grid=so_m,
                    gas_saturation_grid=sg_for_so_m,
                    water_saturation_grid=sw_for_so_m,
                )
                fp_minus = update_phase_densities(
                    fluid_properties=fp_minus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_minus = _compute_residual(
                    i=i,
                    j=j,
                    k=k,
                    perturbed_pressure_grid=fp_minus.pressure_grid,
                    perturbed_oil_saturation_grid=fp_minus.oil_saturation_grid,
                    perturbed_gas_saturation_grid=fp_minus.gas_saturation_grid,
                    perturbed_water_saturation_grid=fp_minus.water_saturation_grid,
                    perturbed_oil_density_grid=fp_minus.oil_effective_density_grid,
                    perturbed_gas_density_grid=fp_minus.gas_density_grid,
                    perturbed_water_density_grid=fp_minus.water_density_grid,
                )
                r_plus = np.asarray(r_plus, dtype=dtype).ravel()
                r_minus = np.asarray(r_minus, dtype=dtype).ravel()
                return r_plus, r_minus

            attempt = 0
            success = False
            derivs = None
            while attempt < max_retries and not success:
                r_plus, r_minus = _compute_neighbor_oil_saturation_residuals(
                    delta_so_neighbor
                )
                derivs = (r_plus - r_minus) / (2.0 * delta_so_neighbor)
                if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                    success = True
                else:
                    delta_so_neighbor *= 0.5
                    attempt += 1

            if success and derivs is not None:
                for idx in range(3):
                    jacobian[3 * cell_1D_index + idx, 3 * neighbor_1D_index + 1] = (
                        float(derivs[idx])
                    )

            # Perturb neighbor gas saturation
            neighbor_gas_saturation = gas_saturation_grid[ni, nj, nk]
            delta_sg_neighbor = compute_perturbation_magnitude_for_saturation(
                neighbor_gas_saturation
            )

            def _compute_neighbor_gas_saturation_residuals(delta: float):
                # Plus
                so_for_sg_p, sg_p, sw_for_sg_p = make_saturation_perturbation(
                    oil_saturation_grid=oil_saturation_grid,
                    gas_saturation_grid=gas_saturation_grid,
                    water_saturation_grid=water_saturation_grid,
                    i=ni,
                    j=nj,
                    k=nk,
                    oil_saturation_delta=0.0,
                    gas_saturation_delta=delta,
                )
                fp_plus = attrs.evolve(
                    fluid_properties,
                    oil_saturation_grid=so_for_sg_p,
                    gas_saturation_grid=sg_p,
                    water_saturation_grid=sw_for_sg_p,
                )
                fp_plus = update_phase_densities(
                    fluid_properties=fp_plus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_plus = _compute_residual(
                    i=i,
                    j=j,
                    k=k,
                    perturbed_pressure_grid=fp_plus.pressure_grid,
                    perturbed_oil_saturation_grid=fp_plus.oil_saturation_grid,
                    perturbed_gas_saturation_grid=fp_plus.gas_saturation_grid,
                    perturbed_water_saturation_grid=fp_plus.water_saturation_grid,
                    perturbed_oil_density_grid=fp_plus.oil_effective_density_grid,
                    perturbed_gas_density_grid=fp_plus.gas_density_grid,
                    perturbed_water_density_grid=fp_plus.water_density_grid,
                )

                # Minus
                so_for_sg_m, sg_m, sw_for_sg_m = make_saturation_perturbation(
                    oil_saturation_grid=oil_saturation_grid,
                    gas_saturation_grid=gas_saturation_grid,
                    water_saturation_grid=water_saturation_grid,
                    i=ni,
                    j=nj,
                    k=nk,
                    oil_saturation_delta=0.0,
                    gas_saturation_delta=-delta,
                )
                fp_minus = attrs.evolve(
                    fluid_properties,
                    oil_saturation_grid=so_for_sg_m,
                    gas_saturation_grid=sg_m,
                    water_saturation_grid=sw_for_sg_m,
                )
                fp_minus = update_phase_densities(
                    fluid_properties=fp_minus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_minus = _compute_residual(
                    i=i,
                    j=j,
                    k=k,
                    perturbed_pressure_grid=fp_minus.pressure_grid,
                    perturbed_oil_saturation_grid=fp_minus.oil_saturation_grid,
                    perturbed_gas_saturation_grid=fp_minus.gas_saturation_grid,
                    perturbed_water_saturation_grid=fp_minus.water_saturation_grid,
                    perturbed_oil_density_grid=fp_minus.oil_effective_density_grid,
                    perturbed_gas_density_grid=fp_minus.gas_density_grid,
                    perturbed_water_density_grid=fp_minus.water_density_grid,
                )
                r_plus = np.asarray(r_plus, dtype=dtype).ravel()
                r_minus = np.asarray(r_minus, dtype=dtype).ravel()
                return r_plus, r_minus

            attempt = 0
            success = False
            derivs = None
            while attempt < max_retries and not success:
                r_plus, r_minus = _compute_neighbor_gas_saturation_residuals(
                    delta_sg_neighbor
                )
                derivs = (r_plus - r_minus) / (2.0 * delta_sg_neighbor)
                if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                    success = True
                else:
                    delta_sg_neighbor *= 0.5
                    attempt += 1

            if success and derivs is not None:
                for idx in range(3):
                    jacobian[3 * cell_1D_index + idx, 3 * neighbor_1D_index + 2] = (
                        float(derivs[idx])
                    )

    return jacobian


def solve_newton_update_system(
    jacobian: lil_matrix,
    residual_vector: np.ndarray,
    options: Options,
) -> np.typing.NDArray:
    """
    Solve the Newton update system J·δx = -R for the update vector δx.

    :param jacobian: Jacobian matrix J
    :param residual_vector: Residual vector R
    :param options: Options
    :return: Update vector δx
    """
    jacobian_csr = jacobian.tocsr()
    rhs = -residual_vector
    return solve_linear_system(
        A_csr=jacobian_csr,
        b=rhs,
        max_iterations=options.max_iterations,
        solver=options.iterative_solver,
        preconditioner=options.preconditioner,
        rtol=1e-3,  # Use a looser tolerance for the inner linear solve (Truncated Newton approach)
    )


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
    :return: `IterationState` with updated grids
    """
    damping_factor: float = 1.0
    # Optionally implement iteration-dependent damping:
    # if iteration == 0:
    #     damping_factor = 0.1  # Only 10% of update
    # elif iteration < 10:
    #     damping_factor = 0.2  # 20% for early iterations
    # elif iteration < 20:
    #     damping_factor = 0.5  # 50% for mid iterations
    # else:
    #     damping_factor = 0.8  # 80% for later iterations (never full 1.0)

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
        f"Iteration {iteration}: damping={damping_factor:.2f}, "
        f"max_dP={max_pressure_change:.2e} psi, "
        f"max_dS={max_saturation_change:.2e}"
    )
    residual_norm = float(np.linalg.norm(update_vector))
    return IterationState(
        pressure_grid=new_pressure_grid,
        oil_saturation_grid=new_oil_saturation_grid,
        gas_saturation_grid=new_gas_saturation_grid,
        iteration=iteration,
        residual_norm=residual_norm,
    )
