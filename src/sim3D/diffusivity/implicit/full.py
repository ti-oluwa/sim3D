"""
Fully implicit solver for simultaneous pressure and saturation evolution.
"""

import functools
import itertools
import logging
import typing

import attrs
import numpy as np
from scipy.sparse import csr_array, csr_matrix, lil_matrix

from sim3D._precision import get_dtype
from sim3D.boundaries import BoundaryConditions
from sim3D.constants import c
from sim3D.diffusivity.base import (
    EvolutionResult,
    solve_linear_system,
    to_1D_index_interior_only,
)
from sim3D.helpers import (
    apply_boundary_conditions,
    build_rock_fluid_properties_grids,
    update_pvt_properties,
)
from sim3D.models import FluidProperties, RockFluidProperties, RockProperties
from sim3D.pvt import compute_harmonic_mean, compute_harmonic_mobility
from sim3D.types import (
    CapillaryPressureGrids,
    FluidPhase,
    Options,
    RelativeMobilityGrids,
    SupportsSetItem,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from sim3D.wells import Wells

logger = logging.getLogger(__name__)


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
    so_perturbed[i, j, k] = np.clip(so_perturbed[i, j, k], eps, 1.0 - eps)
    sg_perturbed[i, j, k] = np.clip(sg_perturbed[i, j, k], eps, 1.0 - eps)
    sw_perturbed[i, j, k] = np.clip(sw_perturbed[i, j, k], eps, 1.0 - eps)

    # Renormalize if sum != 1 due to clamping
    saturation_sum = (
        so_perturbed[i, j, k] + sg_perturbed[i, j, k] + sw_perturbed[i, j, k]
    )
    if abs(saturation_sum - 1.0) > 1e-10:
        so_perturbed[i, j, k] /= saturation_sum
        sg_perturbed[i, j, k] /= saturation_sum
        sw_perturbed[i, j, k] /= saturation_sum
    return so_perturbed, sg_perturbed, sw_perturbed


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
    :param options: Solver options including tolerance and max iterations
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
    interior_count_x = cell_count_x - 2
    interior_count_y = cell_count_y - 2
    interior_count_z = cell_count_z - 2
    total_unknowns = 3 * interior_count_x * interior_count_y * interior_count_z

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

    _to_1D_index = functools.partial(
        to_1D_index_interior_only,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )
    for iteration in range(max_newton_iterations):
        residual_vector = np.zeros(total_unknowns, dtype=dtype)

        for i, j, k in itertools.product(
            range(1, cell_count_x - 1),
            range(1, cell_count_y - 1),
            range(1, cell_count_z - 1),
        ):
            cell_1D_index = _to_1D_index(i=i, j=j, k=k)
            if cell_1D_index < 0:
                continue

            residuals = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_in_days,
                rock_properties=rock_properties,
                fluid_properties=fluid_properties,
                old_fluid_properties=old_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
                injection_grid=injection_grid,
                production_grid=production_grid,
            )
            residual_vector[3 * cell_1D_index] = residuals[0]
            residual_vector[3 * cell_1D_index + 1] = residuals[1]
            residual_vector[3 * cell_1D_index + 2] = residuals[2]

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
                logger.info(f"  Residual[{idx}] = {residual_vector[idx]:.6e}")

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
            relative_mobility_grids=relative_mobility_grids,
            capillary_pressure_grids=capillary_pressure_grids,
            wells=wells,
            options=options,
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
        newton_state = apply_newton_update(
            pressure_grid=pressure_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            update_vector=update_vector,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            iteration=iteration,
        )
        pressure_grid = newton_state.pressure_grid
        oil_saturation_grid = newton_state.oil_saturation_grid
        gas_saturation_grid = newton_state.gas_saturation_grid
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
            _,  # relative_permeability_grids unused
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


def compute_residuals_for_cell(
    i: int,
    j: int,
    k: int,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    old_fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    options: Options,
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
) -> typing.Tuple[float, float, float]:
    """
    Compute residuals for oil, gas, and water mass balance equations at a cell.

    Residual = (Mass_new - Mass_old) - Δt x (Flow_in - Flow_out + Wells)

    When residual = 0, the equation is satisfied.

    :param i: x index
    :param j: y index
    :param k: z index
    :param cell_dimension: (dx, dy) in feet
    :param thickness_grid: Thickness grid
    :param elevation_grid: Elevation grid
    :param time_step_size: Time step size in days
    :param rock_properties: Rock properties at current time/iteration
    :param fluid_properties: Fluid properties at current time/iteration
    :param old_fluid_properties: Fluid properties at previous time/iteration
    :param rock_fluid_properties: Rock-fluid properties
    :param relative_mobility_grids: Relative mobility grids for each phase
    :param capillary_pressure_grids: Capillary pressure grids
    :param wells: Wells object
    :param options: Options
    :return: (oil_residual, gas_residual, water_residual) in ft³
    """
    dx, dy = cell_dimension
    dz = thickness_grid[i, j, k]
    cell_volume = dx * dy * dz
    porosity = rock_properties.porosity_grid[i, j, k]
    pressure_grid = fluid_properties.pressure_grid

    # Get current state accumulation (φ·S·V) (ft³)
    oil_saturation = fluid_properties.oil_saturation_grid[i, j, k]
    gas_saturation = fluid_properties.gas_saturation_grid[i, j, k]
    water_saturation = fluid_properties.water_saturation_grid[i, j, k]
    accumulation_new = compute_accumulation_terms(
        porosity=porosity,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        water_saturation=water_saturation,
        cell_volume=cell_volume,
    )

    # Get old state accumulation (φ·S·V) (ft³)
    oil_saturation_old = old_fluid_properties.oil_saturation_grid[i, j, k]
    gas_saturation_old = old_fluid_properties.gas_saturation_grid[i, j, k]
    water_saturation_old = old_fluid_properties.water_saturation_grid[i, j, k]
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
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        pressure_grid=pressure_grid,
    )
    permeability = (
        rock_properties.absolute_permeability.x[i, j, k],
        rock_properties.absolute_permeability.y[i, j, k],
        rock_properties.absolute_permeability.z[i, j, k],
    )
    well_rates = compute_volumetric_rates_for_cell(
        i=i,
        j=j,
        k=k,
        wells=wells,
        pressure=pressure_grid[i, j, k],
        temperature=fluid_properties.temperature_grid[i, j, k],
        cell_dimension=cell_dimension,
        thickness=thickness_grid[i, j, k],
        permeability=permeability,
        relative_mobility_grids=relative_mobility_grids,
        fluid_properties=fluid_properties,
        options=options,
        injection_grid=injection_grid,
        production_grid=production_grid,
    )

    # Change in pore volume occupied = time x (net flow rate)
    # (φ·S·V)^new - (φ·S·V)^old = Δt x (Flow_in - Flow_out + Wells)
    # Residual = (φ·S·V)^new - (φ·S·V)^old - Δt x (Flow_in - Flow_out + Wells)

    # Note: flux_divergence is (Flow_in - Flow_out) in ft³/day
    # Note: well_rates is (Injection - Production) in ft³/day
    oil_residual = (accumulation_new[0] - accumulation_old[0]) - (
        time_step_size * (flux_divergence[0] + well_rates[0])
    )
    gas_residual = (accumulation_new[1] - accumulation_old[1]) - (
        time_step_size * (flux_divergence[1] + well_rates[1])
    )
    water_residual = (accumulation_new[2] - accumulation_old[2]) - (
        time_step_size * (flux_divergence[2] + well_rates[2])
    )

    # Divide by porosity to ensure that the residuals represent 'saturation error'
    # rather than 'volume error', improving conditioning when porosity is low.
    oil_residual /= porosity
    gas_residual /= porosity
    water_residual /= porosity
    return oil_residual, gas_residual, water_residual


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


def compute_flux_divergence_for_cell(
    i: int,
    j: int,
    k: int,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    pressure_grid: ThreeDimensionalGrid,
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
    :param fluid_properties: Fluid properties
    :param oil_mobility_grid: Oil mobility grid (k_abs * kr_oil / mu_oil)
    :param water_mobility_grid: Water mobility grid (k_abs * kr_water / mu_water)
    :param gas_mobility_grid: Gas mobility grid (k_abs * kr_gas / mu_gas)
    :param capillary_pressure_grids: Capillary pressure grids
    :param pressure_grid: Pressure grid
    :return: (oil_flux_div, gas_flux_div, water_flux_div) in ft³/day
    """
    dx, dy = cell_dimension
    dz = thickness_grid[i, j, k]

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    absolute_permeability = rock_properties.absolute_permeability

    # Compute mobility grids for x, y, z directions
    water_mobility_grid_x = (
        absolute_permeability.x
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_x = (
        absolute_permeability.x
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_x = (
        absolute_permeability.x
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    x_mobility_grids = (oil_mobility_grid_x, water_mobility_grid_x, gas_mobility_grid_x)

    water_mobility_grid_y = (
        absolute_permeability.y
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_y = (
        absolute_permeability.y
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_y = (
        absolute_permeability.y
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    y_mobility_grids = (oil_mobility_grid_y, water_mobility_grid_y, gas_mobility_grid_y)

    water_mobility_grid_z = (
        absolute_permeability.z
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_z = (
        absolute_permeability.z
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_z = (
        absolute_permeability.z
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    z_mobility_grids = (oil_mobility_grid_z, water_mobility_grid_z, gas_mobility_grid_z)

    oil_flux_div = 0.0
    gas_flux_div = 0.0
    water_flux_div = 0.0

    oil_density_grid = fluid_properties.oil_density_grid
    gas_density_grid = fluid_properties.gas_density_grid
    water_density_grid = fluid_properties.water_density_grid

    pcow_grid = capillary_pressure_grids.oil_water_capillary_pressure
    pcgo_grid = capillary_pressure_grids.gas_oil_capillary_pressure

    neighbors = [
        ((i + 1, j, k), dy * dz / dx, x_mobility_grids),
        ((i - 1, j, k), dy * dz / dx, x_mobility_grids),
        ((i, j - 1, k), dx * dz / dy, y_mobility_grids),
        ((i, j + 1, k), dx * dz / dy, y_mobility_grids),
        ((i, j, k - 1), dx * dy / dz, z_mobility_grids),
        ((i, j, k + 1), dx * dy / dz, z_mobility_grids),
    ]

    for neighbor_idx, geometric_factor, mobility_grids in neighbors:
        ni, nj, nk = neighbor_idx
        oil_mobility_grid, water_mobility_grid, gas_mobility_grid = mobility_grids

        water_harmonic_mobility = compute_harmonic_mobility(
            index1=(i, j, k), index2=neighbor_idx, mobility_grid=water_mobility_grid
        )
        oil_harmonic_mobility = compute_harmonic_mobility(
            index1=(i, j, k), index2=neighbor_idx, mobility_grid=oil_mobility_grid
        )
        gas_harmonic_mobility = compute_harmonic_mobility(
            index1=(i, j, k), index2=neighbor_idx, mobility_grid=gas_mobility_grid
        )
        total_mobility = (
            water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
        )
        if total_mobility <= 0.0:
            continue

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
            * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
            * elevation_difference
        ) / 144.0
        oil_gravity_potential = (
            harmonic_oil_density
            * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
            * elevation_difference
        ) / 144.0
        gas_gravity_potential = (
            harmonic_gas_density
            * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
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
        gas_flux_free = (
            gas_harmonic_mobility * gas_potential_difference * geometric_factor
        )

        # Volumetric fluxes in reservoir conditions (ft³/day)
        oil_flux_div += oil_flux
        gas_flux_div += gas_flux_free
        water_flux_div += water_flux

    return oil_flux_div, gas_flux_div, water_flux_div


def compute_volumetric_rates_for_cell(
    i: int,
    j: int,
    k: int,
    wells: Wells[ThreeDimensions],
    pressure: float,
    temperature: float,
    cell_dimension: typing.Tuple[float, float],
    thickness: float,
    permeability: typing.Tuple[float, float, float],
    fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    options: Options,
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
) -> typing.Tuple[float, float, float]:
    """
    Compute well source terms for oil, gas, and water phases at a cell.

    Returns volumetric rates (ft³/day) at reservoir conditions to match volumetric formulation.

    :param i: x index
    :param j: y index
    :param k: z index
    :param wells: Wells object
    :param pressure: Pressure at cell
    :param temperature: Temperature at cell
    :param cell_dimension: (dx, dy) in feet
    :param thickness: Cell thickness in feet
    :param permeability: (kx, ky, kz) in mD
    :param fluid_properties: Fluid properties
    :param relative_mobility_grids: Relative mobility grids
    :param options: Options
    :param injection_grid: Object supporting setitem to set cell injection rates for each phase in ft³/day.
    :param production_grid: Object supporting setitem to set cell production rates for each phase in ft³/day.
    :return: (oil_rate, gas_rate, water_rate) in ft³/day - positive for injection, negative for production
    """
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
    dx, dy = cell_dimension

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
            phase_mobility = relative_mobility_grids.water_relative_mobility[i, j, k]
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

    if production_well is not None and production_well.is_open:
        for produced_fluid in production_well.produced_fluids:
            produced_phase = produced_fluid.phase

            if produced_phase == FluidPhase.GAS:
                phase_mobility = relative_mobility_grids.gas_relative_mobility[i, j, k]
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
                phase_mobility = relative_mobility_grids.oil_relative_mobility[i, j, k]
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

    return oil_rate, gas_rate, water_rate


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
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    options: Options,
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
    :return: Jacobian matrix as sparse lil_matrix
    """
    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape
    interior_count_x = cell_count_x - 2
    interior_count_y = cell_count_y - 2
    interior_count_z = cell_count_z - 2
    total_unknowns = 3 * interior_count_x * interior_count_y * interior_count_z

    dtype = get_dtype()
    jacobian = lil_matrix((total_unknowns, total_unknowns), dtype=dtype)

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
            fp_plus = update_pvt_properties(
                fluid_properties=fp_plus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_plus = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fp_plus,
                old_fluid_properties=old_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
            )

            # Minus
            pressure_minus = pressure_grid.copy()
            pressure_minus[i, j, k] = current_pressure - delta
            fp_minus = attrs.evolve(fluid_properties, pressure_grid=pressure_minus)
            fp_minus = update_pvt_properties(
                fluid_properties=fp_minus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_minus = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fp_minus,
                old_fluid_properties=old_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
            )
            r_plus = np.asarray(r_plus, dtype=dtype).ravel()
            r_minus = np.asarray(r_minus, dtype=dtype).ravel()
            if r_plus.shape != r_minus.shape:
                raise RuntimeError(
                    "Finite-difference residual shape mismatch: "
                    f"r_plus.shape={r_plus.shape}, r_minus.shape={r_minus.shape}"
                )
            if r_plus.size != 3:
                raise RuntimeError(
                    "Unexpected residual size from compute_residuals_for_cell: "
                    f"expected 3 (Ro,Rg,Rw), got {r_plus.size}"
                )
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
                delta_p *= 10.0
                attempt += 1

        if derivs is None or not success:
            # fail fast and log context (better than zeroing diagonal silently)
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
            fp_plus = update_pvt_properties(
                fluid_properties=fp_plus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_plus = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fp_plus,
                old_fluid_properties=old_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
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
            fp_minus = update_pvt_properties(
                fluid_properties=fp_minus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_minus = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fp_minus,
                old_fluid_properties=old_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
            )
            r_plus = np.asarray(r_plus, dtype=dtype).ravel()
            r_minus = np.asarray(r_minus, dtype=dtype).ravel()
            if r_plus.shape != r_minus.shape:
                raise RuntimeError(
                    "Finite-difference residual shape mismatch: "
                    f"r_plus.shape={r_plus.shape}, r_minus.shape={r_minus.shape}"
                )
            if r_plus.size != 3:
                raise RuntimeError(
                    "Unexpected residual size from compute_residuals_for_cell: "
                    f"expected 3 (Ro,Rg,Rw), got {r_plus.size}"
                )
            return r_plus, r_minus

        attempt = 0
        success = False
        while attempt < max_retries and not success:
            r_plus, r_minus = _compute_oil_saturation_residuals(delta_so)
            derivs = (r_plus - r_minus) / (2.0 * delta_so)
            if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                success = True
            else:
                delta_so *= 10.0
                attempt += 1

        if not success:
            logger.error("Oil saturation FD failed at cell (%d,%d,%d)", i, j, k)
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
            fp_plus = update_pvt_properties(
                fluid_properties=fp_plus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_plus = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fp_plus,
                old_fluid_properties=old_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
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
            fp_minus = update_pvt_properties(
                fluid_properties=fp_minus,
                wells=wells,
                miscibility_model=options.miscibility_model,
            )
            r_minus = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fp_minus,
                old_fluid_properties=old_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
            )
            r_plus = np.asarray(r_plus, dtype=dtype).ravel()
            r_minus = np.asarray(r_minus, dtype=dtype).ravel()
            if r_plus.shape != r_minus.shape:
                raise RuntimeError(
                    "Finite-difference residual shape mismatch: "
                    f"r_plus.shape={r_plus.shape}, r_minus.shape={r_minus.shape}"
                )
            if r_plus.size != 3:
                raise RuntimeError(
                    "Unexpected residual size from compute_residuals_for_cell: "
                    f"expected 3 (Ro,Rg,Rw), got {r_plus.size}"
                )
            return r_plus, r_minus

        attempt = 0
        success = False
        while attempt < max_retries and not success:
            r_plus, r_minus = _compute_gas_saturation_residuals(delta_sg)
            derivs = (r_plus - r_minus) / (2.0 * delta_sg)
            if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                success = True
            else:
                delta_sg *= 10.0
                attempt += 1

        if not success:
            logger.error("Gas saturation FD failed at cell (%d,%d,%d)", i, j, k)
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

            # Check if neighbor is in interior (not a boundary cell)
            if not (
                0 < ni < cell_count_x - 1
                and 0 < nj < cell_count_y - 1
                and 0 < nk < cell_count_z - 1
            ):
                continue

            neighbor_1D_index = _to_1D_index(i=ni, j=nj, k=nk)
            if neighbor_1D_index < 0:
                continue

            # Perturb neighbor pressure with central FD and retry
            neighbor_pressure = pressure_grid[ni, nj, nk]
            delta_np = compute_perturbation_magnitude_for_pressure(neighbor_pressure)

            def _compute_neighbor_pressure_residuals(delta: float):
                # Plus
                p_plus = pressure_grid.copy()
                p_plus[ni, nj, nk] = neighbor_pressure + delta
                fp_plus = attrs.evolve(fluid_properties, pressure_grid=p_plus)
                fp_plus = update_pvt_properties(
                    fluid_properties=fp_plus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_plus = compute_residuals_for_cell(
                    i=i,
                    j=j,
                    k=k,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    elevation_grid=elevation_grid,
                    time_step_size=time_step_size,
                    rock_properties=rock_properties,
                    fluid_properties=fp_plus,
                    old_fluid_properties=old_fluid_properties,
                    relative_mobility_grids=relative_mobility_grids,
                    capillary_pressure_grids=capillary_pressure_grids,
                    wells=wells,
                    options=options,
                    injection_grid=None,
                    production_grid=None,
                )

                # Minus
                p_minus = pressure_grid.copy()
                p_minus[ni, nj, nk] = neighbor_pressure - delta
                fp_minus = attrs.evolve(fluid_properties, pressure_grid=p_minus)
                fp_minus = update_pvt_properties(
                    fluid_properties=fp_minus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_minus = compute_residuals_for_cell(
                    i=i,
                    j=j,
                    k=k,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    elevation_grid=elevation_grid,
                    time_step_size=time_step_size,
                    rock_properties=rock_properties,
                    fluid_properties=fp_minus,
                    old_fluid_properties=old_fluid_properties,
                    relative_mobility_grids=relative_mobility_grids,
                    capillary_pressure_grids=capillary_pressure_grids,
                    wells=wells,
                    options=options,
                    injection_grid=None,
                    production_grid=None,
                )
                r_plus = np.asarray(r_plus, dtype=dtype).ravel()
                r_minus = np.asarray(r_minus, dtype=dtype).ravel()
                if r_plus.shape != r_minus.shape:
                    raise RuntimeError(
                        "Finite-difference residual shape mismatch: "
                        f"r_plus.shape={r_plus.shape}, r_minus.shape={r_minus.shape}"
                    )
                if r_plus.size != 3:
                    raise RuntimeError(
                        "Unexpected residual size from compute_residuals_for_cell: "
                        f"expected 3 (Ro,Rg,Rw), got {r_plus.size}"
                    )
                return r_plus, r_minus

            attempt = 0
            success = False
            while attempt < max_retries and not success:
                r_plus, r_minus = _compute_neighbor_pressure_residuals(delta_np)
                derivs = (r_plus - r_minus) / (2.0 * delta_np)
                if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                    success = True
                else:
                    delta_np *= 10.0
                    attempt += 1

            if not success:
                logger.error(
                    "Neighbor pressure FD failed for neighbor (%d,%d,%d) when computing derivatives for cell (%d,%d,%d)",
                    ni,
                    nj,
                    nk,
                    i,
                    j,
                    k,
                )
                raise RuntimeError("Neighbor pressure FD failed.")

            # write off-diagonal entries for neighbor - ∂R[i,j,k]/∂P[neighbor]
            for idx in range(3):
                jacobian[3 * cell_1D_index + idx, 3 * neighbor_1D_index] = float(
                    derivs[idx]
                )

            # Perturb neighbor oil saturation FD with central difference and retry
            neighbor_so = oil_saturation_grid[ni, nj, nk]
            delta_so = compute_perturbation_magnitude_for_saturation(neighbor_so)

            def _compute_neighbor_so_residuals(delta: float):
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
                fp_plus = update_pvt_properties(
                    fluid_properties=fp_plus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_plus = compute_residuals_for_cell(
                    i=i,
                    j=j,
                    k=k,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    elevation_grid=elevation_grid,
                    time_step_size=time_step_size,
                    rock_properties=rock_properties,
                    fluid_properties=fp_plus,
                    old_fluid_properties=old_fluid_properties,
                    relative_mobility_grids=relative_mobility_grids,
                    capillary_pressure_grids=capillary_pressure_grids,
                    wells=wells,
                    options=options,
                    injection_grid=None,
                    production_grid=None,
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
                fp_minus = update_pvt_properties(
                    fluid_properties=fp_minus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_minus = compute_residuals_for_cell(
                    i=i,
                    j=j,
                    k=k,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    elevation_grid=elevation_grid,
                    time_step_size=time_step_size,
                    rock_properties=rock_properties,
                    fluid_properties=fp_minus,
                    old_fluid_properties=old_fluid_properties,
                    relative_mobility_grids=relative_mobility_grids,
                    capillary_pressure_grids=capillary_pressure_grids,
                    wells=wells,
                    options=options,
                    injection_grid=None,
                    production_grid=None,
                )

                r_plus = np.asarray(r_plus, dtype=dtype).ravel()
                r_minus = np.asarray(r_minus, dtype=dtype).ravel()
                if r_plus.shape != r_minus.shape:
                    raise RuntimeError(
                        "Finite-difference residual shape mismatch: "
                        f"r_plus.shape={r_plus.shape}, r_minus.shape={r_minus.shape}"
                    )
                if r_plus.size != 3:
                    raise RuntimeError(
                        "Unexpected residual size from compute_residuals_for_cell: "
                        f"expected 3 (Ro,Rg,Rw), got {r_plus.size}"
                    )
                return r_plus, r_minus

            attempt = 0
            success = False
            while attempt < max_retries and not success:
                r_plus, r_minus = _compute_neighbor_so_residuals(delta_so)
                derivs = (r_plus - r_minus) / (2.0 * delta_so)

                if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                    success = True
                else:
                    delta_so *= 10.0
                    attempt += 1

            if not success:
                raise RuntimeError("Neighbor oil saturation FD failed")

            # write ∂R/∂So_neighbor - ∂R[i,j,k]/∂So[neighbor]
            for idx in range(3):
                jacobian[3 * cell_1D_index + idx, 3 * neighbor_1D_index + 1] = float(
                    derivs[idx]
                )

            # Perturb neighbor gas saturation FD with central difference and retry
            neighbor_sg = gas_saturation_grid[ni, nj, nk]
            delta_sg = compute_perturbation_magnitude_for_saturation(neighbor_sg)

            def _compute_neighbor_sg_residuals(delta: float):
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
                fp_plus = update_pvt_properties(
                    fluid_properties=fp_plus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_plus = compute_residuals_for_cell(
                    i=i,
                    j=j,
                    k=k,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    elevation_grid=elevation_grid,
                    time_step_size=time_step_size,
                    rock_properties=rock_properties,
                    fluid_properties=fp_plus,
                    old_fluid_properties=old_fluid_properties,
                    relative_mobility_grids=relative_mobility_grids,
                    capillary_pressure_grids=capillary_pressure_grids,
                    wells=wells,
                    options=options,
                    injection_grid=None,
                    production_grid=None,
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
                fp_minus = update_pvt_properties(
                    fluid_properties=fp_minus,
                    wells=wells,
                    miscibility_model=options.miscibility_model,
                )
                r_minus = compute_residuals_for_cell(
                    i=i,
                    j=j,
                    k=k,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    elevation_grid=elevation_grid,
                    time_step_size=time_step_size,
                    rock_properties=rock_properties,
                    fluid_properties=fp_minus,
                    old_fluid_properties=old_fluid_properties,
                    relative_mobility_grids=relative_mobility_grids,
                    capillary_pressure_grids=capillary_pressure_grids,
                    wells=wells,
                    options=options,
                    injection_grid=None,
                    production_grid=None,
                )

                r_plus = np.asarray(r_plus, dtype=dtype).ravel()
                r_minus = np.asarray(r_minus, dtype=dtype).ravel()
                if r_plus.shape != r_minus.shape:
                    raise RuntimeError(
                        "Finite-difference residual shape mismatch: "
                        f"r_plus.shape={r_plus.shape}, r_minus.shape={r_minus.shape}"
                    )
                if r_plus.size != 3:
                    raise RuntimeError(
                        "Unexpected residual size from compute_residuals_for_cell: "
                        f"expected 3 (Ro,Rg,Rw), got {r_plus.size}"
                    )
                return r_plus, r_minus

            attempt = 0
            success = False
            while attempt < max_retries and not success:
                r_plus, r_minus = _compute_neighbor_sg_residuals(delta_sg)
                derivs = (r_plus - r_minus) / (2.0 * delta_sg)

                if np.all(np.isfinite(derivs)) and np.max(np.abs(derivs)) < 1e12:
                    success = True
                else:
                    delta_sg *= 10.0
                    attempt += 1

            if not success:
                raise RuntimeError("Neighbor gas saturation FD failed")

            # write ∂R/∂Sg_neighbor - ∂R[i,j,k]/∂Sg[neighbor]
            for idx in range(3):
                jacobian[3 * cell_1D_index + idx, 3 * neighbor_1D_index + 2] = float(
                    derivs[idx]
                )

    return jacobian


def jacobian_finite_difference_check(
    A_csr: typing.Union[csr_matrix, csr_array],
    compute_residuals_fn: typing.Callable[[np.typing.NDArray], np.typing.NDArray],
    x: np.typing.NDArray,
    tol: float = 1e-6,
) -> float:
    """
    Quick check: pick a random direction v and compare J·v vs FD directional derivative.
    Raises/logs if mismatch large.
    """
    v = np.random.randn(A_csr.shape[0])  # type: ignore
    Jv = A_csr.dot(v)
    eps = 1e-6
    R0 = compute_residuals_fn(x)
    x_eps = x + eps * v
    R1 = compute_residuals_fn(x_eps)
    fd = (R1 - R0) / eps
    rel_err = np.linalg.norm(fd - Jv) / max(1e-14, np.linalg.norm(Jv))
    logger.info("Jacobian FD check rel_err = %.3e", rel_err)
    return float(rel_err)


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
    new_pressure_grid = pressure_grid.copy()
    new_oil_saturation_grid = oil_saturation_grid.copy()
    new_gas_saturation_grid = gas_saturation_grid.copy()
    max_pressure_change = 0.0
    max_saturation_change = 0.0

    damping_factor: float = 1.0
    # if iteration == 0:
    #     damping_factor = 0.1  # Only 10% of update
    # elif iteration < 10:
    #     damping_factor = 0.2  # 20% for early iterations
    # elif iteration < 20:
    #     damping_factor = 0.5  # 50% for mid iterations
    # else:
    #     damping_factor = 0.8  # 80% for later iterations (never full 1.0)

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

        # Get damped updates
        pressure_update = damping_factor * update_vector[3 * cell_1D_index]
        oil_saturation_update = damping_factor * update_vector[3 * cell_1D_index + 1]
        gas_saturation_update = damping_factor * update_vector[3 * cell_1D_index + 2]

        # Apply updates
        new_pressure = pressure_grid[i, j, k] + pressure_update
        new_oil_saturation = oil_saturation_grid[i, j, k] + oil_saturation_update
        new_gas_saturation = gas_saturation_grid[i, j, k] + gas_saturation_update

        # Clamp to physical bounds
        new_pressure = np.clip(new_pressure, c.MIN_VALID_PRESSURE, c.MAX_VALID_PRESSURE)
        new_oil_saturation = np.clip(new_oil_saturation, 0.0, 1.0)
        new_gas_saturation = np.clip(new_gas_saturation, 0.0, 1.0)

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

        # Track maximum changes
        max_pressure_change = max(max_pressure_change, abs(pressure_update))
        max_saturation_change = max(
            max_saturation_change,
            abs(oil_saturation_update),
            abs(gas_saturation_update),
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


"""
FULLY IMPLICIT 3D THREE-PHASE FLOW EQUATIONS - COMPLETE REFERENCE

This solver simultaneously solves for three unknowns per cell:
    - P  (oil phase pressure, psi)
    - So (oil saturation, dimensionless fraction)
    - Sg (gas saturation, dimensionless fraction)
    
Water saturation is computed from the constraint: Sw = 1 - So - Sg

================================================================================
THE THREE COUPLED MASS BALANCE EQUATIONS
================================================================================

We solve these three equations simultaneously for each cell (i,j,k):

1. OIL MASS BALANCE:
   ∂/∂t(φ·ρo·So) = -∇·(ρo·vo) + qo
   
   In words: "Rate of oil accumulation = Oil flowing in - Oil flowing out + Oil from wells"

2. GAS MASS BALANCE:
   ∂/∂t(φ·ρg·Sg) = -∇·(ρg·vg) + qg
   
   In words: "Rate of gas accumulation = Gas flowing in - Gas flowing out + Gas from wells"

3. WATER MASS BALANCE:
   ∂/∂t(φ·ρw·Sw) = -∇·(ρw·vw) + qw
   
   In words: "Rate of water accumulation = Water flowing in - Water flowing out + Water from wells"

================================================================================
SIMPLIFIED VOLUMETRIC FORM (Black Oil, No Dissolved Gas)
================================================================================

For computational simplicity, we use a volumetric formulation that ignores 
density variations and dissolved gas effects:

OIL EQUATION:
    ∂/∂t(φ·So) = -∇·vo + qo/ρo
    
GAS EQUATION:
    ∂/∂t(φ·Sg) = -∇·vg + qg/ρg
    
WATER EQUATION:
    ∂/∂t(φ·Sw) = -∇·vw + qw/ρw

Where:
    φ  = porosity (fraction of rock that is pore space)
    So = oil saturation (fraction of pore space occupied by oil)
    Sg = gas saturation (fraction of pore space occupied by gas)
    Sw = water saturation (fraction of pore space occupied by water)
    vo, vg, vw = phase velocities from Darcy's law (ft/day)
    qo, qg, qw = well source terms (ft³/day per unit bulk volume)

================================================================================
DARCY'S LAW FOR MULTIPHASE FLOW
================================================================================

Each phase flows according to its own mobility and pressure gradient:

    vo = -λo·∇Po = -λo·∇P                    (oil velocity)
    vg = -λg·∇Pg = -λg·(∇P + ∇Pcgo)         (gas velocity)
    vw = -λw·∇Pw = -λw·(∇P - ∇Pcow)         (water velocity)

Where:
    λo = k·kro/μo  (oil mobility, ft²/(psi·day))
    λg = k·krg/μg  (gas mobility, ft²/(psi·day))
    λw = k·krw/μw  (water mobility, ft²/(psi·day))
    
    k   = absolute permeability (mD) - intrinsic rock property
    kro = relative permeability to oil (0 to 1) - depends on So
    krg = relative permeability to gas (0 to 1) - depends on Sg
    krw = relative permeability to water (0 to 1) - depends on Sw
    μo  = oil viscosity (cP)
    μg  = gas viscosity (cP)
    μw  = water viscosity (cP)
    
    P    = oil phase pressure (our primary unknown)
    Pcow = Po - Pw (oil-water capillary pressure, function of Sw)
    Pcgo = Pg - Po (gas-oil capillary pressure, function of Sg)

PHYSICAL MEANING:
- Higher mobility → faster flow
- Capillary pressure creates additional driving force beyond pressure gradient
- Each phase "sees" a different pressure due to interfacial tension effects

================================================================================
DISCRETE RESIDUAL FORM (What We Actually Solve)
================================================================================

For cell (i,j,k) at time step n+1, the residuals are:

OIL RESIDUAL:
    Ro = (φ·So)^(n+1) - (φ·So)^n - (Δt/V)·[Σ(Flux_oil) + Qo]

GAS RESIDUAL:
    Rg = (φ·Sg)^(n+1) - (φ·Sg)^n - (Δt/V)·[Σ(Flux_gas) + Qg]

WATER RESIDUAL:
    Rw = (φ·Sw)^(n+1) - (φ·Sw)^n - (Δt/V)·[Σ(Flux_water) + Qw]

INTERPRETATION:
- When Ro = 0: Oil mass is conserved (accumulation = flow + wells)
- When Rg = 0: Gas mass is conserved
- When Rw = 0: Water mass is conserved

TERMS EXPLAINED:
    (φ·So)^(n+1)     = Oil pore volume fraction at NEW time (what we're solving for)
    (φ·So)^n         = Oil pore volume fraction at OLD time (known from previous step)
    Δt               = Time step size (days)
    V                = Cell bulk volume = Δx·Δy·Δz (ft³)
    Σ(Flux_oil)      = Sum of oil fluxes from all 6 neighbor faces (ft³/day)
    Qo               = Oil well rate: (+) for injection, (-) for production (ft³/day)

================================================================================
FLUX CALCULATION ACROSS CELL FACES
================================================================================

For the face between current cell (i,j,k) and its East neighbor (i+1,j,k):

    Flux_oil = λo_harmonic · (P[i+1,j,k] - P[i,j,k]) · (Δy·Δz/Δx)

    Flux_gas = λg_harmonic · [(P[i+1,j,k] + Pcgo[i+1,j,k]) - 
                               (P[i,j,k] + Pcgo[i,j,k])] · (Δy·Δz/Δx)

    Flux_water = λw_harmonic · [(P[i+1,j,k] - Pcow[i+1,j,k]) - 
                                 (P[i,j,k] - Pcow[i,j,k])] · (Δy·Δz/Δx)

WHERE:
    λ_harmonic      = 2·λ1·λ2/(λ1 + λ2)  (harmonic mean ensures flux continuity)
    (Δy·Δz/Δx)      = Transmissibility geometric factor
                    = (Face area / Flow length)
    Positive flux   = Flow INTO current cell (from neighbor)
    Negative flux   = Flow OUT OF current cell (to neighbor)

GRAVITY EFFECTS (for vertical flows):
    Add gravity term: ± λ_harmonic · ρ_phase · g · Δz / 144.0
    Where: g = 32.174 ft/s², Δz = elevation difference, 144 = unit conversion

TOTAL FLUX FOR A CELL:
    Σ(Flux_oil) = Flux_East + Flux_West + Flux_North + Flux_South + Flux_Top + Flux_Bottom

================================================================================
NEWTON-RAPHSON SOLUTION METHOD
================================================================================

We want to find [P, So, Sg] such that all residuals = 0.

Newton's method iteratively solves:
    
    J·δx = -R          (solve linear system for update δx)
    x^(n+1) = x^n + α·δx   (apply damped update)

WHERE:
    x  = [P1, So1, Sg1, P2, So2, Sg2, ..., PN, SoN, SgN]  (all unknowns)
    R  = [Ro1, Rg1, Rw1, Ro2, Rg2, Rw2, ..., RoN, RgN, RwN]  (all residuals)
    J  = Jacobian matrix = ∂R/∂x  (how residuals change with unknowns)
    δx = Update vector (change in unknowns)
    α  = Damping factor (0.1 to 0.5) - prevents overshooting

JACOBIAN STRUCTURE (for 3D problem):
    - Size: (3N x 3N) where N = number of cells
    - Sparse: Only ~21 non-zeros per row (3 diagonal + 6 neighbors x 3 equations)
    - Block structure: Each cell couples to itself and 6 neighbors

CONVERGENCE CRITERION:
    ||R|| < tolerance  (typically 1e-6)
    
    This means: "All mass balance errors are negligible"

================================================================================
EXAMPLE: 5x5x10 GRID
================================================================================

Grid dimensions: 5 cells (x) x 5 cells (y) x 10 cells (z) = 250 cells

Total unknowns: 250 cells x 3 unknowns/cell = 750 unknowns
    [P1, So1, Sg1, P2, So2, Sg2, ..., P250, So250, Sg250]

Total equations: 250 cells x 3 equations/cell = 750 equations
    [Ro1, Rg1, Rw1, Ro2, Rg2, Rw2, ..., Ro250, Rg250, Rw250]

Jacobian matrix: 750 x 750 (but only ~15,750 non-zero entries due to sparsity)

Linear solver: BiCGSTAB or LGMRES with ILU preconditioner

Typical convergence: 5-15 Newton iterations per time step

================================================================================
KEY PHYSICAL INSIGHTS
================================================================================

1. COUPLING:
   - Pressure equation couples ALL three phases (they all flow when P changes)
   - Saturation equations couple through constraint: So + Sg + Sw = 1
   - Relative permeabilities create strong nonlinearity (small Sw → tiny krw)

2. IMPLICIT VS EXPLICIT:
   - IMPLICIT: Solve all equations together → unconditionally stable → larger Δt
   - EXPLICIT: Solve sequentially → conditionally stable → tiny Δt required
   - Trade-off: Implicit is more work per step but takes bigger steps

3. CONVERGENCE CHALLENGES:
   - Strong nonlinearity in relative permeability curves
   - Capillary pressure can be very steep near residual saturations
   - Wells create localized strong sources/sinks
   - Solution: Damping (take small steps) + good preconditioner

4. WHEN IT WORKS WELL:
   - Smooth saturation profiles
   - Moderate pressure gradients
   - Reasonable time steps (not too large)
   - Well-conditioned Jacobian

5. WHEN IT STRUGGLES:
   - Sharp saturation fronts (water breakthrough, gas coning)
   - Near residual saturations (kr → 0, mobility → 0)
   - Large time steps with strong wells
   - Solution: Reduce Δt, increase damping, improve initial guess

================================================================================
SIMPLIFICATIONS IN CURRENT IMPLEMENTATION
================================================================================

What we INCLUDE:
✓ Three-phase flow (oil, water, gas)
✓ Capillary pressure effects
✓ Gravity segregation
✓ Relative permeability
✓ Wells (injection and production)
✓ Heterogeneous rock properties

What we EXCLUDE (for simplicity):
✗ Dissolved gas in oil (Rs) - creates circular dependencies
✗ Dissolved gas in water (Rsw)
✗ Hysteresis in relative permeability
✗ Compositional effects

================================================================================
RECOMMENDED WORKFLOW
================================================================================

1. Start with IMPES (implicit pressure, explicit saturation)
   → Get stable pressure field first

2. Switch to simplified fully implicit (no dissolved gas)
   → Learn Newton-Raphson behavior

This incremental approach makes debugging much easier!

================================================================================
"""
