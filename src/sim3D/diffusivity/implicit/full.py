"""
Fully implicit solver for simultaneous pressure and saturation evolution.
"""

import functools
import itertools
import logging
import typing

import attrs
import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import bicgstab, lgmres

from sim3D._precision import get_dtype, get_floating_point_info
from sim3D.boundaries import BoundaryConditions
from sim3D.constants import c
from sim3D.diffusivity.base import EvolutionResult
from sim3D.diffusivity.implicit.pressure import to_1D_index_interior_only
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
    delta_oil_saturation: float = 0.0,
    delta_gas_saturation: float = 0.0,
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
    :param delta_oil_saturation: Perturbation to add to oil saturation
    :param delta_gas_saturation: Perturbation to add to gas saturation
    :return: Tuple of (perturbed_so_grid, perturbed_sg_grid, perturbed_sw_grid)
    """
    eps = 1e-12
    # Create copies
    so_perturbed = oil_saturation_grid.copy()
    sg_perturbed = gas_saturation_grid.copy()
    sw_perturbed = water_saturation_grid.copy()

    # Apply perturbations
    so_perturbed[i, j, k] += delta_oil_saturation
    sg_perturbed[i, j, k] += delta_gas_saturation

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


def compute_perturbation_magnitude_for_saturation(value: float) -> float:
    """
    Computes adaptive perturbation magnitude for saturation finite differences.
    Uses relative perturbation scaled by value magnitude with floor for small values.

    :param value: Current saturation value
    :return: Perturbation magnitude (dimensionless)
    """
    return max(1e-8, 1e-6 * max(1.0, abs(value)))


def compute_perturbation_magnitude_for_pressure(value: float) -> float:
    """
    Computes adaptive perturbation magnitude for pressure finite differences.
    Uses relative perturbation scaled by value magnitude with floor for small values.

    :param value: Current pressure value (psi)
    :return: Perturbation magnitude (psi)
    """
    return max(1e-6, 1e-6 * max(1.0, abs(value)))


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

    max_newton_iterations = min(options.max_iterations, 50)
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

    for iteration in range(max_newton_iterations):
        residual_vector = np.zeros(total_unknowns, dtype=dtype)

        for i, j, k in itertools.product(
            range(1, cell_count_x - 1),
            range(1, cell_count_y - 1),
            range(1, cell_count_z - 1),
        ):
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
                rock_fluid_properties=rock_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                pressure_grid=pressure_grid,
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
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
            logger.info(
                f"Initial residual norm: {final_residual_norm:.2e}, "
                f"Max residual: {np.max(np.abs(residual_vector)):.2e}, "
                f"Mean residual: {np.mean(np.abs(residual_vector)):.2e}"
            )

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

        jacobian_matrix = assemble_jacobian_matrix(
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
        update_vector = solve_newton_update_system(
            jacobian_matrix=jacobian_matrix,
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
    rock_fluid_properties: RockFluidProperties,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    pressure_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
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
    Compute "scaled" residuals for oil, gas, and water mass balance equations at a cell.

    :param i: x index
    :param j: y index
    :param k: z index
    :param cell_dimension: (dx, dy) in feet
    :param thickness_grid: Thickness grid
    :param elevation_grid: Elevation grid
    :param time_step_size: Time step size in days
    :param rock_properties: Rock properties at current time
    :param fluid_properties: Fluid properties at current time
    :param old_fluid_properties: Fluid properties at previous time
    :param rock_fluid_properties: Rock-fluid properties
    :param relative_mobility_grids: Relative mobility grids
    :param capillary_pressure_grids: Capillary pressure grids
    :param pressure_grid: Pressure grid (current iteration)
    :param oil_saturation_grid: Oil saturation grid (current iteration)
    :param gas_saturation_grid: Gas saturation grid (current iteration)
    :param water_saturation_grid: Water saturation grid (current iteration)
    :param wells: Wells object
    :param options: Options
    :return: (oil_residual, gas_residual, water_residual)
    """
    oil_saturation = oil_saturation_grid[i, j, k]
    gas_saturation = gas_saturation_grid[i, j, k]
    water_saturation = water_saturation_grid[i, j, k]
    accumulation_new = compute_accumulation_terms(
        i=i,
        j=j,
        k=k,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        water_saturation=water_saturation,
    )

    oil_saturation_old = old_fluid_properties.oil_saturation_grid[i, j, k]
    gas_saturation_old = old_fluid_properties.gas_saturation_grid[i, j, k]
    water_saturation_old = old_fluid_properties.water_saturation_grid[i, j, k]
    accumulation_old = compute_accumulation_terms(
        i=i,
        j=j,
        k=k,
        rock_properties=rock_properties,
        fluid_properties=old_fluid_properties,
        oil_saturation=oil_saturation_old,
        gas_saturation=gas_saturation_old,
        water_saturation=water_saturation_old,
    )
    flux_divergence = compute_flux_divergence_for_cell(
        i=i,
        j=j,
        k=k,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        fluid_properties=fluid_properties,
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
        fluid_properties=fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        options=options,
        injection_grid=injection_grid,
        production_grid=production_grid,
    )

    dx, dy = cell_dimension
    dz = thickness_grid[i, j, k]
    cell_volume = dx * dy * dz

    oil_residual = (
        (accumulation_new[0] - accumulation_old[0]) * cell_volume / time_step_size
        + flux_divergence[0]
        - well_rates[0]
    )
    gas_residual = (
        (accumulation_new[1] - accumulation_old[1]) * cell_volume / time_step_size
        + flux_divergence[1]
        - well_rates[1]
    )
    water_residual = (
        (accumulation_new[2] - accumulation_old[2]) * cell_volume / time_step_size
        + flux_divergence[2]
        - well_rates[2]
    )

    # Scale residuals by cell volume
    # This converts from ft³/day to 1/day (mass balance per unit volume)
    # Makes residuals dimensionless and O(1) magnitude
    oil_residual /= cell_volume
    gas_residual /= cell_volume
    water_residual /= cell_volume
    return oil_residual, gas_residual, water_residual


def compute_accumulation_terms(
    i: int,
    j: int,
    k: int,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    oil_saturation: float,
    gas_saturation: float,
    water_saturation: float,
) -> typing.Tuple[float, float, float]:
    """
    Compute accumulation terms for oil, gas, and water phases in reservoir conditions.

    Computes φ·S for each phase, representing volume fraction (dimensionless).
    Gas accumulation includes dissolved gas in oil phase.

    :param i: x index
    :param j: y index
    :param k: z index
    :param rock_properties: Rock properties
    :param fluid_properties: Fluid properties
    :param oil_saturation: Oil saturation at cell
    :param gas_saturation: Gas saturation at cell
    :param water_saturation: Water saturation at cell
    :return: (oil_accumulation, gas_accumulation, water_accumulation) - dimensionless
    """
    porosity = rock_properties.porosity_grid[i, j, k]
    solution_gor_scf_stb = fluid_properties.solution_gas_to_oil_ratio_grid[i, j, k]
    oil_fvf = fluid_properties.oil_formation_volume_factor_grid[i, j, k]
    gas_fvf = fluid_properties.gas_formation_volume_factor_grid[i, j, k]

    # Convert Rs from scf/STB to dimensionless (RCF/RCF)
    # Rs_reservoir [ft³/ft³] = Rs [scf/STB] × gas_fvf [ft³/scf] / (oil_fvf [bbl/STB] × BBL_TO_FT3 [ft³/bbl])
    solution_gor_reservoir = (
        solution_gor_scf_stb * (gas_fvf / (oil_fvf * c.BBL_TO_FT3))
        if oil_fvf > 0
        else 0.0
    )

    # Accumulation in reservoir conditions: φ·S (dimensionless)
    oil_accumulation = porosity * oil_saturation
    gas_accumulation = porosity * (
        gas_saturation + (oil_saturation * solution_gor_reservoir)
    )
    water_accumulation = porosity * water_saturation
    return oil_accumulation, gas_accumulation, water_accumulation


def compute_flux_divergence_for_cell(
    i: int,
    j: int,
    k: int,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    fluid_properties: FluidProperties[ThreeDimensions],
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
    :param relative_mobility_grids: Relative mobility grids
    :param capillary_pressure_grids: Capillary pressure grids
    :param pressure_grid: Pressure grid
    :return: (oil_flux_div, gas_flux_div, water_flux_div) in ft³/day
    """
    dx, dy = cell_dimension
    dz = thickness_grid[i, j, k]

    oil_flux_div = 0.0
    gas_flux_div = 0.0
    water_flux_div = 0.0

    oil_mobility_grid = relative_mobility_grids.oil_relative_mobility
    gas_mobility_grid = relative_mobility_grids.gas_relative_mobility
    water_mobility_grid = relative_mobility_grids.water_relative_mobility

    oil_density_grid = fluid_properties.oil_density_grid
    gas_density_grid = fluid_properties.gas_density_grid
    water_density_grid = fluid_properties.water_density_grid

    solution_gor_grid = fluid_properties.solution_gas_to_oil_ratio_grid

    pcow_grid = capillary_pressure_grids.oil_water_capillary_pressure
    pcgo_grid = capillary_pressure_grids.gas_oil_capillary_pressure

    neighbor_offsets = [
        ((i + 1, j, k), dy * dz / dx),
        ((i - 1, j, k), dy * dz / dx),
        ((i, j + 1, k), dx * dz / dy),
        ((i, j - 1, k), dx * dz / dy),
        ((i, j, k + 1), dx * dy / dz),
        ((i, j, k - 1), dx * dy / dz),
    ]

    for neighbor_idx, geometric_factor in neighbor_offsets:
        ni, nj, nk = neighbor_idx

        if not (
            0 <= ni < pressure_grid.shape[0]
            and 0 <= nj < pressure_grid.shape[1]
            and 0 <= nk < pressure_grid.shape[2]
        ):
            continue

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

        pressure_diff = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
        pcow_diff = pcow_grid[ni, nj, nk] - pcow_grid[i, j, k]
        pcgo_diff = pcgo_grid[ni, nj, nk] - pcgo_grid[i, j, k]
        elevation_diff = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]

        water_density_harmonic = compute_harmonic_mean(
            water_density_grid[i, j, k], water_density_grid[ni, nj, nk]
        )
        oil_density_harmonic = compute_harmonic_mean(
            oil_density_grid[i, j, k], oil_density_grid[ni, nj, nk]
        )
        gas_density_harmonic = compute_harmonic_mean(
            gas_density_grid[i, j, k], gas_density_grid[ni, nj, nk]
        )

        water_gravity_potential = (
            water_density_harmonic
            * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
            * elevation_diff
        ) / 144.0
        oil_gravity_potential = (
            oil_density_harmonic
            * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
            * elevation_diff
        ) / 144.0
        gas_gravity_potential = (
            gas_density_harmonic
            * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
            * elevation_diff
        ) / 144.0

        water_potential_diff = pressure_diff + pcow_diff + water_gravity_potential
        oil_potential_diff = pressure_diff + oil_gravity_potential
        gas_potential_diff = pressure_diff - pcgo_diff + gas_gravity_potential

        water_flux = water_harmonic_mobility * water_potential_diff * geometric_factor
        oil_flux = oil_harmonic_mobility * oil_potential_diff * geometric_factor
        gas_flux_free = gas_harmonic_mobility * gas_potential_diff * geometric_factor

        # Interface-averaged solution GOR (upwind based on oil flow direction)
        # Also get FVFs for unit conversion
        if oil_flux > 0:
            # Flow from neighbor to cell; use neighbor's Rs
            solution_gor_scf_stb = solution_gor_grid[ni, nj, nk]
            oil_fvf = fluid_properties.oil_formation_volume_factor_grid[ni, nj, nk]
            gas_fvf = fluid_properties.gas_formation_volume_factor_grid[ni, nj, nk]
        elif oil_flux < 0:
            # Flow from cell to neighbor; use cell's Rs
            solution_gor_scf_stb = solution_gor_grid[i, j, k]
            oil_fvf = fluid_properties.oil_formation_volume_factor_grid[i, j, k]
            gas_fvf = fluid_properties.gas_formation_volume_factor_grid[i, j, k]
        else:
            # No flow, use arithmetic average
            solution_gor_scf_stb = 0.5 * (
                solution_gor_grid[i, j, k] + solution_gor_grid[ni, nj, nk]
            )
            oil_fvf = 0.5 * (
                fluid_properties.oil_formation_volume_factor_grid[i, j, k]
                + fluid_properties.oil_formation_volume_factor_grid[ni, nj, nk]
            )
            gas_fvf = 0.5 * (
                fluid_properties.gas_formation_volume_factor_grid[i, j, k]
                + fluid_properties.gas_formation_volume_factor_grid[ni, nj, nk]
            )

        # Convert Rs from scf/STB to dimensionless (RCF/RCF) for reservoir volumetric balance
        # Rs_reservoir [ft³/ft³] = Rs [scf/STB] × gas_fvf [RB/scf] / (oil_fvf [RB/STB] × BBL_TO_FT3 [ft³/bbl])
        # Units: (scf/STB) × (RB/scf) / (RB/STB × BBL_TO_FT3) = ft³/ft³ (dimensionless)
        solution_gor_reservoir = (
            solution_gor_scf_stb * (gas_fvf / (oil_fvf * c.BBL_TO_FT3))
            if oil_fvf > 0
            else 0.0
        )

        # Volumetric fluxes in reservoir conditions (ft³/day)
        oil_flux_total = oil_flux
        gas_flux_total = gas_flux_free + (oil_flux * solution_gor_reservoir)
        water_flux_total = water_flux

        oil_flux_div += oil_flux_total
        gas_flux_div += gas_flux_total
        water_flux_div += water_flux_total

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


def assemble_jacobian_matrix(
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

        # Compute base residuals for this cell
        base_residuals = compute_residuals_for_cell(
            i=i,
            j=j,
            k=k,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            old_fluid_properties=old_fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
            relative_mobility_grids=relative_mobility_grids,
            capillary_pressure_grids=capillary_pressure_grids,
            pressure_grid=pressure_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            wells=wells,
            options=options,
            injection_grid=None,
            production_grid=None,
        )

        # DIAGONAL TERMS: Derivatives with respect to current cell variables

        # Perturb pressure at current cell
        current_pressure = pressure_grid[i, j, k]
        delta_p = compute_perturbation_magnitude_for_pressure(current_pressure)

        pressure_perturbed = pressure_grid.copy()
        pressure_perturbed[i, j, k] += delta_p
        perturbed_residuals = compute_residuals_for_cell(
            i=i,
            j=j,
            k=k,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            old_fluid_properties=old_fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
            relative_mobility_grids=relative_mobility_grids,
            capillary_pressure_grids=capillary_pressure_grids,
            pressure_grid=pressure_perturbed,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            wells=wells,
            options=options,
            injection_grid=None,
            production_grid=None,
        )

        # ∂R/∂P (diagonal)
        jacobian[3 * cell_1D_index, 3 * cell_1D_index] = (
            perturbed_residuals[0] - base_residuals[0]
        ) / delta_p
        jacobian[3 * cell_1D_index + 1, 3 * cell_1D_index] = (
            perturbed_residuals[1] - base_residuals[1]
        ) / delta_p
        jacobian[3 * cell_1D_index + 2, 3 * cell_1D_index] = (
            perturbed_residuals[2] - base_residuals[2]
        ) / delta_p

        # Perturb oil saturation at current cell
        # Perturb oil saturation at current cell (holding gas saturation constant)
        # Use helper to enforce Sw = 1 - So - Sg constraint
        current_oil_saturation = oil_saturation_grid[i, j, k]
        delta_oil_saturation = compute_perturbation_magnitude_for_saturation(
            current_oil_saturation
        )
        (
            oil_saturation_perturbed,
            gas_saturation_perturbed_for_oil,
            water_saturation_perturbed_for_oil,
        ) = make_saturation_perturbation(
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            i=i,
            j=j,
            k=k,
            delta_oil_saturation=delta_oil_saturation,
            delta_gas_saturation=0.0,
        )
        # Recompute rock-fluid properties at perturbed state
        fluid_properties_perturbed = attrs.evolve(
            fluid_properties,
            oil_saturation_grid=oil_saturation_perturbed,
            gas_saturation_grid=gas_saturation_perturbed_for_oil,
            water_saturation_grid=water_saturation_perturbed_for_oil,
        )
        (
            _,  # relative_permeability_grids unused
            relative_mobility_grids_perturbed,
            capillary_pressure_grids_perturbed,
        ) = build_rock_fluid_properties_grids(
            fluid_properties=fluid_properties_perturbed,
            rock_properties=rock_properties,
            rock_fluid_properties=rock_fluid_properties,
            options=options,
        )
        perturbed_residuals = compute_residuals_for_cell(
            i=i,
            j=j,
            k=k,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties_perturbed,
            old_fluid_properties=old_fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
            relative_mobility_grids=relative_mobility_grids_perturbed,
            capillary_pressure_grids=capillary_pressure_grids_perturbed,
            pressure_grid=pressure_grid,
            oil_saturation_grid=oil_saturation_perturbed,
            gas_saturation_grid=gas_saturation_perturbed_for_oil,
            water_saturation_grid=water_saturation_perturbed_for_oil,
            wells=wells,
            options=options,
            injection_grid=None,
            production_grid=None,
        )

        # ∂R/∂So (diagonal)
        jacobian[3 * cell_1D_index, 3 * cell_1D_index + 1] = (
            perturbed_residuals[0] - base_residuals[0]
        ) / delta_oil_saturation
        jacobian[3 * cell_1D_index + 1, 3 * cell_1D_index + 1] = (
            perturbed_residuals[1] - base_residuals[1]
        ) / delta_oil_saturation
        jacobian[3 * cell_1D_index + 2, 3 * cell_1D_index + 1] = (
            perturbed_residuals[2] - base_residuals[2]
        ) / delta_oil_saturation

        # Perturb gas saturation at current cell
        # FIXED: Use helper to enforce Sw = 1 - So - Sg constraint
        current_gas_saturation = gas_saturation_grid[i, j, k]
        delta_gas_saturation = compute_perturbation_magnitude_for_saturation(
            current_gas_saturation
        )

        (
            oil_saturation_perturbed_for_gas,
            gas_saturation_perturbed,
            water_saturation_perturbed_for_gas,
        ) = make_saturation_perturbation(
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            i=i,
            j=j,
            k=k,
            delta_oil_saturation=0.0,
            delta_gas_saturation=delta_gas_saturation,
        )

        # Recompute rock-fluid properties at perturbed state
        fluid_properties_perturbed = attrs.evolve(
            fluid_properties,
            oil_saturation_grid=oil_saturation_perturbed_for_gas,
            gas_saturation_grid=gas_saturation_perturbed,
            water_saturation_grid=water_saturation_perturbed_for_gas,
        )
        (
            _,  # relative_permeability_grids unused
            relative_mobility_grids_perturbed,
            capillary_pressure_grids_perturbed,
        ) = build_rock_fluid_properties_grids(
            fluid_properties=fluid_properties_perturbed,
            rock_properties=rock_properties,
            rock_fluid_properties=rock_fluid_properties,
            options=options,
        )
        perturbed_residuals = compute_residuals_for_cell(
            i=i,
            j=j,
            k=k,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties_perturbed,
            old_fluid_properties=old_fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
            relative_mobility_grids=relative_mobility_grids_perturbed,
            capillary_pressure_grids=capillary_pressure_grids_perturbed,
            pressure_grid=pressure_grid,
            oil_saturation_grid=oil_saturation_perturbed_for_gas,
            gas_saturation_grid=gas_saturation_perturbed,
            water_saturation_grid=water_saturation_perturbed_for_gas,
            wells=wells,
            options=options,
            injection_grid=None,
            production_grid=None,
        )

        # ∂R/∂Sg (diagonal)
        jacobian[3 * cell_1D_index, 3 * cell_1D_index + 2] = (
            perturbed_residuals[0] - base_residuals[0]
        ) / delta_gas_saturation
        jacobian[3 * cell_1D_index + 1, 3 * cell_1D_index + 2] = (
            perturbed_residuals[1] - base_residuals[1]
        ) / delta_gas_saturation
        jacobian[3 * cell_1D_index + 2, 3 * cell_1D_index + 2] = (
            perturbed_residuals[2] - base_residuals[2]
        ) / delta_gas_saturation

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

            # Perturb neighbor pressure
            neighbor_pressure = pressure_grid[ni, nj, nk]
            delta_p_neighbor = compute_perturbation_magnitude_for_pressure(
                neighbor_pressure
            )
            pressure_perturbed_neighbor = pressure_grid.copy()
            pressure_perturbed_neighbor[ni, nj, nk] += delta_p_neighbor
            perturbed_residuals_neighbor = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,  # Still computing residuals for cell (i,j,k)
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fluid_properties,
                old_fluid_properties=old_fluid_properties,
                rock_fluid_properties=rock_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                pressure_grid=pressure_perturbed_neighbor,
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
            )

            # ∂R[i,j,k]/∂P[neighbor]
            jacobian[3 * cell_1D_index, 3 * neighbor_1D_index] = (
                perturbed_residuals_neighbor[0] - base_residuals[0]
            ) / delta_p_neighbor
            jacobian[3 * cell_1D_index + 1, 3 * neighbor_1D_index] = (
                perturbed_residuals_neighbor[1] - base_residuals[1]
            ) / delta_p_neighbor
            jacobian[3 * cell_1D_index + 2, 3 * neighbor_1D_index] = (
                perturbed_residuals_neighbor[2] - base_residuals[2]
            ) / delta_p_neighbor

            # Perturb neighbor oil saturation
            # Use helper to enforce Sw = 1 - So - Sg constraint
            neighbor_oil_saturation = oil_saturation_grid[ni, nj, nk]
            delta_oil_saturation_neighbor = (
                compute_perturbation_magnitude_for_saturation(neighbor_oil_saturation)
            )
            (
                oil_saturation_perturbed_neighbor,
                gas_saturation_perturbed_neighbor_for_oil,
                water_saturation_perturbed_neighbor_for_oil,
            ) = make_saturation_perturbation(
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
                i=ni,
                j=nj,
                k=nk,
                delta_oil_saturation=delta_oil_saturation_neighbor,
                delta_gas_saturation=0.0,
            )
            # Recompute rock-fluid properties at perturbed state
            fluid_properties_perturbed_neighbor = attrs.evolve(
                fluid_properties,
                oil_saturation_grid=oil_saturation_perturbed_neighbor,
                gas_saturation_grid=gas_saturation_perturbed_neighbor_for_oil,
                water_saturation_grid=water_saturation_perturbed_neighbor_for_oil,
            )
            (
                _,  # relative_permeability_grids unused
                relative_mobility_grids_perturbed_neighbor,
                capillary_pressure_grids_perturbed_neighbor,
            ) = build_rock_fluid_properties_grids(
                fluid_properties=fluid_properties_perturbed_neighbor,
                rock_properties=rock_properties,
                rock_fluid_properties=rock_fluid_properties,
                options=options,
            )
            perturbed_residuals_neighbor = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fluid_properties_perturbed_neighbor,
                old_fluid_properties=old_fluid_properties,
                rock_fluid_properties=rock_fluid_properties,
                relative_mobility_grids=relative_mobility_grids_perturbed_neighbor,
                capillary_pressure_grids=capillary_pressure_grids_perturbed_neighbor,
                pressure_grid=pressure_grid,
                oil_saturation_grid=oil_saturation_perturbed_neighbor,
                gas_saturation_grid=gas_saturation_perturbed_neighbor_for_oil,
                water_saturation_grid=water_saturation_perturbed_neighbor_for_oil,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
            )

            # ∂R[i,j,k]/∂So[neighbor]
            jacobian[3 * cell_1D_index, 3 * neighbor_1D_index + 1] = (
                perturbed_residuals_neighbor[0] - base_residuals[0]
            ) / delta_oil_saturation_neighbor
            jacobian[3 * cell_1D_index + 1, 3 * neighbor_1D_index + 1] = (
                perturbed_residuals_neighbor[1] - base_residuals[1]
            ) / delta_oil_saturation_neighbor
            jacobian[3 * cell_1D_index + 2, 3 * neighbor_1D_index + 1] = (
                perturbed_residuals_neighbor[2] - base_residuals[2]
            ) / delta_oil_saturation_neighbor

            # Perturb neighbor gas saturation
            # Use helper to enforce Sw = 1 - So - Sg constraint
            neighbor_gas_saturation = gas_saturation_grid[ni, nj, nk]
            delta_gas_saturation_neighbor = (
                compute_perturbation_magnitude_for_saturation(neighbor_gas_saturation)
            )
            (
                oil_saturation_perturbed_neighbor_for_gas,
                gas_saturation_perturbed_neighbor,
                water_saturation_perturbed_neighbor_for_gas,
            ) = make_saturation_perturbation(
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                water_saturation_grid=water_saturation_grid,
                i=ni,
                j=nj,
                k=nk,
                delta_oil_saturation=0.0,
                delta_gas_saturation=delta_gas_saturation_neighbor,
            )
            # Recompute rock-fluid properties at perturbed state
            fluid_properties_perturbed_neighbor = attrs.evolve(
                fluid_properties,
                oil_saturation_grid=oil_saturation_perturbed_neighbor_for_gas,
                gas_saturation_grid=gas_saturation_perturbed_neighbor,
                water_saturation_grid=water_saturation_perturbed_neighbor_for_gas,
            )
            (
                _,  # relative_permeability_grids unused
                relative_mobility_grids_perturbed_neighbor,
                capillary_pressure_grids_perturbed_neighbor,
            ) = build_rock_fluid_properties_grids(
                fluid_properties=fluid_properties_perturbed_neighbor,
                rock_properties=rock_properties,
                rock_fluid_properties=rock_fluid_properties,
                options=options,
            )
            perturbed_residuals_neighbor = compute_residuals_for_cell(
                i=i,
                j=j,
                k=k,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step_size=time_step_size,
                rock_properties=rock_properties,
                fluid_properties=fluid_properties_perturbed_neighbor,
                old_fluid_properties=old_fluid_properties,
                rock_fluid_properties=rock_fluid_properties,
                relative_mobility_grids=relative_mobility_grids_perturbed_neighbor,
                capillary_pressure_grids=capillary_pressure_grids_perturbed_neighbor,
                pressure_grid=pressure_grid,
                oil_saturation_grid=oil_saturation_perturbed_neighbor_for_gas,
                gas_saturation_grid=gas_saturation_perturbed_neighbor,
                water_saturation_grid=water_saturation_perturbed_neighbor_for_gas,
                wells=wells,
                options=options,
                injection_grid=None,
                production_grid=None,
            )

            # ∂R[i,j,k]/∂Sg[neighbor]
            jacobian[3 * cell_1D_index, 3 * neighbor_1D_index + 2] = (
                perturbed_residuals_neighbor[0] - base_residuals[0]
            ) / delta_gas_saturation_neighbor
            jacobian[3 * cell_1D_index + 1, 3 * neighbor_1D_index + 2] = (
                perturbed_residuals_neighbor[1] - base_residuals[1]
            ) / delta_gas_saturation_neighbor
            jacobian[3 * cell_1D_index + 2, 3 * neighbor_1D_index + 2] = (
                perturbed_residuals_neighbor[2] - base_residuals[2]
            ) / delta_gas_saturation_neighbor

    return jacobian


def solve_newton_update_system(
    jacobian_matrix: lil_matrix,
    residual_vector: np.ndarray,
    options: Options,
) -> np.ndarray:
    """
    Solve the Newton update system J·δx = -R for the update vector δx.

    :param jacobian_matrix: Jacobian matrix J
    :param residual_vector: Residual vector R
    :param options: Options
    :return: Update vector δx
    """
    jacobian_csr = jacobian_matrix.tocsr()
    rhs = -residual_vector

    diag_elements = jacobian_csr.diagonal()
    diag_elements = np.where(np.abs(diag_elements) < 1e-10, 1.0, diag_elements)
    preconditioner_diag = diags(1.0 / diag_elements, format="csr")

    rhs_norm = np.linalg.norm(rhs)
    epsilon = get_floating_point_info().eps
    rtol = float(epsilon * 50)
    atol = float(max(1e-8, 20 * epsilon * rhs_norm))

    bicgstab_max_iter = min(150, options.max_iterations)
    update_vector, info = bicgstab(
        A=jacobian_csr,
        b=rhs,
        M=preconditioner_diag,
        rtol=rtol,
        atol=atol,
        maxiter=bicgstab_max_iter,
    )

    if info != 0:
        lgmres_max_iter = min(250, options.max_iterations)
        update_vector, info = lgmres(
            A=jacobian_csr,
            b=rhs,
            M=preconditioner_diag,
            rtol=rtol,
            atol=atol,
            maxiter=lgmres_max_iter,
            inner_m=30,
            outer_k=3,
        )
        if info != 0:
            raise RuntimeError(
                f"Failed to solve Newton update system. BiCGSTAB and LGMRES both failed with code {info}"
            )
    return update_vector


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

    # Adaptive damping: We start conservative, relax as we converge
    if iteration == 0:
        damping_factor = 0.5  # First iteration: 50% of full update
    elif iteration < 3:
        damping_factor = 0.7  # Early iterations: 70%
    else:
        damping_factor = 1.0  # Later iterations: full update

    max_pressure_change = 0.0
    max_saturation_change = 0.0

    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_1D_index = to_1D_index_interior_only(
            i, j, k, cell_count_x, cell_count_y, cell_count_z
        )

        if cell_1D_index < 0:
            continue

        # Apply damped updates
        pressure_update = damping_factor * update_vector[3 * cell_1D_index]
        oil_saturation_update = damping_factor * update_vector[3 * cell_1D_index + 1]
        gas_saturation_update = damping_factor * update_vector[3 * cell_1D_index + 2]

        new_pressure = pressure_grid[i, j, k] + pressure_update
        new_oil_saturation = oil_saturation_grid[i, j, k] + oil_saturation_update
        new_gas_saturation = gas_saturation_grid[i, j, k] + gas_saturation_update

        # Clamp to physical bounds
        new_pressure = max(1.0, new_pressure)
        new_oil_saturation = np.clip(new_oil_saturation, 0.0, 1.0)
        new_gas_saturation = np.clip(new_gas_saturation, 0.0, 1.0)

        # Enforce saturation constraint
        saturation_sum = new_oil_saturation + new_gas_saturation
        if saturation_sum > 1.0:
            scale_factor = 1.0 / saturation_sum
            new_oil_saturation *= scale_factor
            new_gas_saturation *= scale_factor

        new_pressure_grid[i, j, k] = new_pressure
        new_oil_saturation_grid[i, j, k] = new_oil_saturation
        new_gas_saturation_grid[i, j, k] = new_gas_saturation

        # Track maximum changes for diagnostics
        max_pressure_change = max(max_pressure_change, abs(pressure_update))
        max_saturation_change = max(
            max_saturation_change,
            abs(oil_saturation_update),
            abs(gas_saturation_update),
        )

    logger.info(
        f"Iteration {iteration}: damping={damping_factor:.2f}, "
        f"max_dP={max_pressure_change:.2e}, "
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
