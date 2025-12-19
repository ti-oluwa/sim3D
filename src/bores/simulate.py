"""Run a simulation workflow on a 3-Dimensional reservoir model."""

import copy
import logging
import typing

import attrs
import numpy as np

from bores.boundaries import BoundaryConditions, default_bc
from bores.config import Config
from bores.diffusivity import (
    evolve_fully_implicit,
    evolve_miscible_saturation_explicitly,
    evolve_pressure_explicitly,
    evolve_pressure_implicitly,
    evolve_saturation_explicitly,
)
from bores.errors import SimulationError, TimingError
from bores.grids.base import (
    CapillaryPressureGrids,
    RateGrids,
    RelPermGrids,
    RelativeMobilityGrids,
    _RateGridsProxy,
    build_uniform_grid,
    pad_grid,
)
from bores.grids.pvt import build_three_phase_relative_mobilities_grids
from bores.helpers import (
    _mirror_neighbour,
    apply_boundary_conditions,
    build_rock_fluid_properties_grids,
    update_pvt_properties,
)
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockFluidProperties,
    RockProperties,
)
from bores.states import ModelState
from bores.timing import Timer
from bores.types import MiscibilityModel, NDimension, NDimensionalGrid, ThreeDimensions
from bores.wells import Wells


__all__ = ["run"]

logger = logging.getLogger(__name__)


NEGATIVE_PRESSURE_ERROR = """
Negative pressure encountered in the pressure grid at the following indices:
{indices}
This indicates a likely issue with the simulation setup, numerical stability, or physical parameters.

Potential causes include:
1. Boundary conditions that allow for unphysical pressure drops.
2. Incompatible or unrealistic rock/fluid properties.
3. Time step size too large for explicit schemes, leading to instability.
4. Incorrect initial conditions or pressure distributions.
5. Unrealistic/improperly configured wells (e.g., injection/production rates or pressures).
6. Numerical issues due to discretization choices or solver settings.

Suggested actions:
- Validate boundary conditions and ensure fixed-pressure constraints are properly applied.
- Check permeability, porosity, and compressibility values.
- Cell dimensions and bulk volume should be appropriate for the physical scale of the reservoir.
- Use smaller time steps if using explicit updates.
- Clamp pressure updates to a minimum floor (e.g., 1.45 psi) to prevent blow-up.
- Cross-check well source/sink terms for sign and magnitude correctness.

Simulation aborted to avoid propagation of unphysical results.
"""


@attrs.frozen(slots=True)
class StepResult(typing.Generic[NDimension]):
    """
    Result from executing one time step of the simulation.
    """

    fluid_properties: FluidProperties[NDimension]
    """Updated fluid properties after the time step."""
    oil_injection_grid: NDimensionalGrid[NDimension]
    """Grid of oil injection rates during the time step."""
    water_injection_grid: NDimensionalGrid[NDimension]
    """Grid of water injection rates during the time step."""
    gas_injection_grid: NDimensionalGrid[NDimension]
    """Grid of gas injection rates during the time step."""
    oil_production_grid: NDimensionalGrid[NDimension]
    """Grid of oil production rates during the time step."""
    water_production_grid: NDimensionalGrid[NDimension]
    """Grid of water production rates during the time step."""
    gas_production_grid: NDimensionalGrid[NDimension]
    """Grid of gas production rates during the time step."""
    success: bool = True
    """Whether the time step evolution was successful."""
    message: typing.Optional[str] = None
    """Optional message providing additional information about the time step result."""
    accept_kwargs: typing.Dict[str, typing.Any] = attrs.field(factory=dict)


def _run_implicit_step(
    time_step: int,
    zeros_grid: NDimensionalGrid[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    padded_thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    padded_rock_properties: RockProperties[ThreeDimensions],
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using fully implicit solver (simultaneous pressure-saturation).

    Returns: `StepResult` containing updated rates and fluid properties.
    """
    logger.debug("Evolving pressure and saturation simultaneously (fully implicit)...")
    # Build zeros grids to track production and injection at each time step
    oil_injection_grid = zeros_grid.copy()
    water_injection_grid = zeros_grid.copy()
    gas_injection_grid = zeros_grid.copy()
    oil_production_grid = zeros_grid.copy()
    water_production_grid = zeros_grid.copy()
    gas_production_grid = zeros_grid.copy()

    result = evolve_fully_implicit(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        rock_fluid_properties=rock_fluid_properties,
        relative_mobility_grids=padded_relative_mobility_grids,
        capillary_pressure_grids=padded_capillary_pressure_grids,
        wells=wells,
        config=config,
        boundary_conditions=boundary_conditions,
        # Wrap the grids in a proxy to allow item assignment
        injection_grid=_RateGridsProxy(
            oil=oil_injection_grid,
            water=water_injection_grid,
            gas=gas_injection_grid,
        ),
        production_grid=_RateGridsProxy(
            oil=oil_production_grid,
            water=water_production_grid,
            gas=gas_production_grid,
        ),
    )
    if not result.success:
        logger.error(
            f"Fully implicit evolution failed at time step {time_step}: \n{result.message}"
        )
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
            fluid_properties=padded_fluid_properties,
            success=False,
            message=result.message,
        )

    solution = result.value
    padded_pressure_grid = solution.pressure_grid
    padded_oil_saturation_grid = solution.oil_saturation_grid
    padded_gas_saturation_grid = solution.gas_saturation_grid
    padded_water_saturation_grid = solution.water_saturation_grid

    logger.debug(
        f"Fully implicit evolution completed! "
        f"Converged: {solution.converged}, "
        f"Iterations: {solution.newton_iterations}, "
        f"Residual: {solution.final_residual_norm:.2e}"
    )
    if not solution.converged:
        logger.warning(
            f"Fully implicit solver did not converge at time step {time_step}. "
            f"Final residual norm: {solution.final_residual_norm:.2e}"
        )

    # Update fluid properties with new pressure and saturation
    logger.debug("Updating fluid properties with new pressure and saturation grids...")
    padded_fluid_properties = attrs.evolve(
        padded_fluid_properties,
        pressure_grid=padded_pressure_grid,
        water_saturation_grid=padded_water_saturation_grid,
        oil_saturation_grid=padded_oil_saturation_grid,
        gas_saturation_grid=padded_gas_saturation_grid,
    )

    # Update PVT properties with new state
    logger.debug("Updating PVT fluid properties after fully implicit solve...")
    padded_fluid_properties = update_pvt_properties(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
    )
    return StepResult(
        oil_injection_grid=oil_injection_grid,
        water_injection_grid=water_injection_grid,
        gas_injection_grid=gas_injection_grid,
        oil_production_grid=oil_production_grid,
        water_production_grid=water_production_grid,
        gas_production_grid=gas_production_grid,
        fluid_properties=padded_fluid_properties,
        success=True,
        message=result.message,
        accept_kwargs={
            "max_cfl_encountered": None,
            "newton_iterations": solution.newton_iterations,
        },
    )


def _run_impes_step(
    time_step: int,
    zeros_grid: NDimensionalGrid[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    padded_thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    padded_rock_properties: RockProperties[ThreeDimensions],
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    padded_relperm_grids: RelPermGrids[ThreeDimensions],
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    evolve_saturation: typing.Callable,
    config: Config,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using IMPES (Implicit Pressure, Explicit Saturation).

    Returns: `StepResult` containing updated rates and fluid properties.
    """
    logger.debug("Evolving pressure (implicit)...")
    pressure_result = evolve_pressure_implicitly(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        relative_mobility_grids=padded_relative_mobility_grids,
        capillary_pressure_grids=padded_capillary_pressure_grids,
        wells=wells,
        config=config,
    )
    if not pressure_result.success:
        logger.error(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            oil_injection_grid=zeros_grid,
            water_injection_grid=zeros_grid,
            gas_injection_grid=zeros_grid,
            oil_production_grid=zeros_grid,
            water_production_grid=zeros_grid,
            gas_production_grid=zeros_grid,
            fluid_properties=padded_fluid_properties,
            success=False,
            message=pressure_result.message,
        )

    logger.debug("Pressure evolution completed!")
    padded_pressure_grid = pressure_result.value

    if (negative_pressure_indices := np.argwhere(padded_pressure_grid < 0)).size > 0:
        logger.warning(
            f"Negative pressure detected at {negative_pressure_indices.size} grid cells"
        )
        logger.warning(f"Minimum pressure: {np.min(padded_pressure_grid):.8f} psi")
        logger.warning(
            f"First few negative pressure indices: {negative_pressure_indices.tolist()[:10]}"
        )
        message = (
            NEGATIVE_PRESSURE_ERROR.format(indices=negative_pressure_indices.tolist())
            + f"\nAt Time Step {time_step}."
        )
        return StepResult(
            oil_injection_grid=zeros_grid,
            water_injection_grid=zeros_grid,
            gas_injection_grid=zeros_grid,
            oil_production_grid=zeros_grid,
            water_production_grid=zeros_grid,
            gas_production_grid=zeros_grid,
            fluid_properties=padded_fluid_properties,
            success=False,
            message=message,
        )

    # Update fluid properties with new pressure grid
    logger.debug("Updating fluid properties with new pressure grid...")
    padded_fluid_properties = attrs.evolve(
        padded_fluid_properties, pressure_grid=padded_pressure_grid
    )

    # For IMPES, we need to update the fluid properties
    # before proceeding to saturation evolution.
    logger.debug("Updating PVT fluid properties for saturation evolution")
    padded_fluid_properties = update_pvt_properties(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
    )

    # Recompute relative mobility grids with updated fluid properties
    # Since relative mobility depends on fluid viscosities which change with pressure
    logger.debug("Rebuilding relative mobility grids for saturation evolution...")
    (
        padded_water_relative_mobility_grid,
        padded_oil_relative_mobility_grid,
        padded_gas_relative_mobility_grid,
    ) = build_three_phase_relative_mobilities_grids(
        oil_relative_permeability_grid=padded_relperm_grids.kro,
        water_relative_permeability_grid=padded_relperm_grids.krw,
        gas_relative_permeability_grid=padded_relperm_grids.krg,
        water_viscosity_grid=padded_fluid_properties.water_viscosity_grid,
        oil_viscosity_grid=padded_fluid_properties.oil_effective_viscosity_grid,
        gas_viscosity_grid=padded_fluid_properties.gas_viscosity_grid,
    )
    # Clamp relative mobility grids to avoid numerical issues
    padded_water_relative_mobility_grid = config.relative_mobility_range[
        "water"
    ].arrayclip(padded_water_relative_mobility_grid)
    padded_oil_relative_mobility_grid = config.relative_mobility_range["oil"].arrayclip(
        padded_oil_relative_mobility_grid
    )
    padded_gas_relative_mobility_grid = config.relative_mobility_range["gas"].arrayclip(
        padded_gas_relative_mobility_grid
    )
    padded_relative_mobility_grids = RelativeMobilityGrids(
        water_relative_mobility=padded_water_relative_mobility_grid,
        oil_relative_mobility=padded_oil_relative_mobility_grid,
        gas_relative_mobility=padded_gas_relative_mobility_grid,
    )
    logger.debug("Relative mobility grids rebuilt for saturation evolution.")

    # Saturation evolution (explicit)
    logger.debug("Evolving saturation (explicit)...")
    # Build zeros grids to track production and injection at each time step
    oil_injection_grid = zeros_grid.copy()
    water_injection_grid = zeros_grid.copy()
    gas_injection_grid = zeros_grid.copy()
    oil_production_grid = zeros_grid.copy()
    water_production_grid = zeros_grid.copy()
    gas_production_grid = zeros_grid.copy()

    saturation_result = evolve_saturation(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        relative_mobility_grids=padded_relative_mobility_grids,
        capillary_pressure_grids=padded_capillary_pressure_grids,
        wells=wells,
        config=config,
        # Wrap the grids in a proxy to allow item assignment
        injection_grid=_RateGridsProxy(
            oil=oil_injection_grid,
            water=water_injection_grid,
            gas=gas_injection_grid,
        ),
        production_grid=_RateGridsProxy(
            oil=oil_production_grid,
            water=water_production_grid,
            gas=gas_production_grid,
        ),
    )
    if not saturation_result.success:
        logger.error(
            f"Explicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
            fluid_properties=padded_fluid_properties,
            success=False,
            message=saturation_result.message,
        )
    logger.debug("Saturation evolution completed!")

    logger.debug("Updating fluid properties with new saturation grids...")
    (
        padded_water_saturation_grid,
        padded_oil_saturation_grid,
        *other_padded_saturation_grids,
    ) = saturation_result.value  # type: ignore
    other_grids_size = len(other_padded_saturation_grids)
    if other_grids_size == 1:
        padded_gas_saturation_grid = other_padded_saturation_grids[0]
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
        )
    elif other_grids_size == 2:
        (
            padded_gas_saturation_grid,
            padded_solvent_concentration_grid,
        ) = other_padded_saturation_grids
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            solvent_concentration_grid=padded_solvent_concentration_grid,
        )
    else:
        raise RuntimeError(
            f"Unexpected number of saturation grids returned: {other_grids_size + 2}"
        )

    return StepResult(
        oil_injection_grid=oil_injection_grid,
        water_injection_grid=water_injection_grid,
        gas_injection_grid=gas_injection_grid,
        oil_production_grid=oil_production_grid,
        water_production_grid=water_production_grid,
        gas_production_grid=gas_production_grid,
        fluid_properties=padded_fluid_properties,
        success=True,
        message=saturation_result.message,
        accept_kwargs={
            "max_cfl_encountered": saturation_result.metadata.cfl_info.max_cfl_encountered,
            "newton_iterations": None,
        },
    )


def _run_explicit_step(
    time_step: int,
    zeros_grid: NDimensionalGrid[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    padded_thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    padded_rock_properties: RockProperties[ThreeDimensions],
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    evolve_saturation: typing.Callable,
    config: Config,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using fully explicit scheme (explicit pressure and saturation).

    Returns: `StepResult` containing updated rates and fluid properties.
    """
    logger.debug("Evolving pressure (explicit)...")
    pressure_result = evolve_pressure_explicitly(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        relative_mobility_grids=padded_relative_mobility_grids,
        capillary_pressure_grids=padded_capillary_pressure_grids,
        wells=wells,
        config=config,
    )
    if not pressure_result.success:
        logger.error(
            f"Explicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            oil_injection_grid=zeros_grid,
            water_injection_grid=zeros_grid,
            gas_injection_grid=zeros_grid,
            oil_production_grid=zeros_grid,
            water_production_grid=zeros_grid,
            gas_production_grid=zeros_grid,
            fluid_properties=padded_fluid_properties,
            success=False,
            message=pressure_result.message,
        )

    logger.debug("Pressure evolution completed!")
    padded_pressure_grid = pressure_result.value

    if (negative_pressure_indices := np.argwhere(padded_pressure_grid < 0)).size > 0:
        logger.warning(
            f"Negative pressure detected at {negative_pressure_indices.size} grid cells"
        )
        logger.warning(f"Minimum pressure: {np.min(padded_pressure_grid):.8f} psi")
        logger.warning(
            f"First few negative pressure indices: {negative_pressure_indices.tolist()[:10]}"
        )
        message = (
            NEGATIVE_PRESSURE_ERROR.format(indices=negative_pressure_indices.tolist())
            + f"\nAt Time Step {time_step}."
        )
        return StepResult(
            oil_injection_grid=zeros_grid,
            water_injection_grid=zeros_grid,
            gas_injection_grid=zeros_grid,
            oil_production_grid=zeros_grid,
            water_production_grid=zeros_grid,
            gas_production_grid=zeros_grid,
            fluid_properties=padded_fluid_properties,
            success=False,
            message=message,
        )

    # For explicit schemes, we can re-use the current fluid properties
    # in the saturation evolution step.
    # Explicit pressure is fully decoupled from saturation update.
    logger.debug(
        "Using current PVT fluid properties for saturation evolution (explicit scheme)"
    )

    # Saturation evolution (explicit)
    logger.debug("Evolving saturation (explicit)...")
    # Build zeros grids to track production and injection at each time step
    oil_injection_grid = zeros_grid.copy()
    water_injection_grid = zeros_grid.copy()
    gas_injection_grid = zeros_grid.copy()
    oil_production_grid = zeros_grid.copy()
    water_production_grid = zeros_grid.copy()
    gas_production_grid = zeros_grid.copy()

    saturation_result = evolve_saturation(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        relative_mobility_grids=padded_relative_mobility_grids,
        capillary_pressure_grids=padded_capillary_pressure_grids,
        wells=wells,
        config=config,
        # Wrap the grids in a proxy to allow item assignment
        injection_grid=_RateGridsProxy(
            oil=oil_injection_grid,
            water=water_injection_grid,
            gas=gas_injection_grid,
        ),
        production_grid=_RateGridsProxy(
            oil=oil_production_grid,
            water=water_production_grid,
            gas=gas_production_grid,
        ),
    )
    if not saturation_result.success:
        logger.error(
            f"Explicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
            fluid_properties=padded_fluid_properties,
            success=False,
            message=saturation_result.message,
        )

    logger.debug("Saturation evolution completed!")

    # Update fluid properties with new pressure after saturation update
    logger.debug(
        "Updating fluid properties with new pressure grid (explicit scheme)..."
    )
    padded_fluid_properties = attrs.evolve(
        padded_fluid_properties, pressure_grid=padded_pressure_grid
    )
    logger.debug("Fluid properties updated with new pressure grid.")

    logger.debug("Updating fluid properties with new saturation grids...")
    (
        padded_water_saturation_grid,
        padded_oil_saturation_grid,
        *other_padded_saturation_grids,
    ) = saturation_result.value  # type: ignore
    other_grids_size = len(other_padded_saturation_grids)
    if other_grids_size == 1:
        padded_gas_saturation_grid = other_padded_saturation_grids[0]
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
        )
    elif other_grids_size == 2:
        (
            padded_gas_saturation_grid,
            padded_solvent_concentration_grid,
        ) = other_padded_saturation_grids
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            solvent_concentration_grid=padded_solvent_concentration_grid,
        )
    else:
        raise RuntimeError(
            f"Unexpected number of saturation grids returned: {other_grids_size + 2}"
        )

    # Update PVT properties with new state (pressure and saturations)
    logger.debug("Updating PVT fluid properties after explicit solve...")
    padded_fluid_properties = update_pvt_properties(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
    )
    return StepResult(
        fluid_properties=padded_fluid_properties,
        oil_injection_grid=oil_injection_grid,
        water_injection_grid=water_injection_grid,
        gas_injection_grid=gas_injection_grid,
        oil_production_grid=oil_production_grid,
        water_production_grid=water_production_grid,
        gas_production_grid=gas_production_grid,
        success=True,
        message=saturation_result.message,
        accept_kwargs={
            "max_cfl_encountered": saturation_result.metadata.cfl_info.max_cfl_encountered,
            "newton_iterations": None,
        },
    )


def log_progress(
    step: int,
    step_size: float,
    time_elapsed: float,
    total_time: float,
    is_last_step: bool = False,
    interval: int = 3,
):
    """Logs the simulation progress at specified intervals."""
    if step <= 1 or step % interval == 0 or is_last_step:
        percent_complete = (time_elapsed / total_time) * 100.0
        logger.info(
            f"Time Step {step} with Δt = {step_size:.4f}s - "
            f"({percent_complete:.4f}%) - "
            f"Elapsed Time: {time_elapsed:.4f}s / {total_time:.4f}s"
        )


def run(
    model: ReservoirModel[ThreeDimensions],
    timer: Timer,
    wells: typing.Optional[Wells[ThreeDimensions]] = None,
    config: typing.Optional[Config] = None,
) -> typing.Generator[ModelState[ThreeDimensions], None, None]:
    """
    Runs a dynamic simulation on a 3D reservoir model with specified properties and wells.

    The 3D simulation evolves pressure and saturation over time using the specified evolution scheme.
    3D simulations are computationally intensive and may require significant memory and processing power.

    :param model: The reservoir model containing grid, rock, and fluid properties.
    :param timer: The time manager for controlling simulation time steps.
    :param wells: The wells configuration for the simulation.
    :param config: Simulation run configuration and parameters.
    :yield: Yields the model state at specified output intervals.
    """
    if config is None:
        config = Config()
    if wells is None:
        wells = Wells()

    logger.info("Starting reservoir simulation workflow...")

    cell_dimension = model.cell_dimension
    boundary_conditions = model.boundary_conditions
    grid_shape = model.grid_shape
    has_wells = wells.exists()
    output_frequency = config.output_frequency
    convergence_tolerance = config.convergence_tolerance
    scheme = config.scheme
    miscibility_model = config.miscibility_model

    logger.debug(f"Grid dimensions: {grid_shape}")
    logger.debug(f"Cell dimensions: {cell_dimension}")
    logger.debug(f"Evolution scheme: {scheme}")
    logger.debug(f"Total simulation time: {timer.simulation_time} seconds")
    logger.debug(f"Output frequency: every {output_frequency} steps")
    logger.debug(f"Convergence tolerance: {convergence_tolerance}")
    logger.debug(f"Has wells: {has_wells}")
    if has_wells:
        logger.debug("Checking well locations against grid shape")
        wells.check_location(grid_shape=grid_shape)

    # Use the config context manager to ensure that constants defined in config are utilized
    # throughout the simulation run
    with config.constants():
        # Pad fluid and rock properties grids and other necesary grids with ghost cells
        # for boundary condition application
        # Ensure ghost cells mirror neighbour values by default
        padded_fluid_properties = model.fluid_properties.pad(pad_width=1)
        padded_rock_properties = model.rock_properties.pad(pad_width=1)
        rock_fluid_properties = model.rock_fluid_properties
        thickness_grid = model.thickness_grid
        padded_thickness_grid = pad_grid(thickness_grid, pad_width=1)
        padded_thickness_grid = _mirror_neighbour(padded_thickness_grid)
        elevation_grid = model.get_elevation_grid(
            apply_dip=not config.disable_structural_dip
        )
        padded_elevation_grid = pad_grid(elevation_grid, pad_width=1)
        padded_elevation_grid = _mirror_neighbour(padded_elevation_grid)

        # Apply boundary conditions to relevant padded grids
        logger.debug("Applying boundary conditions to initial padded grids")
        padded_fluid_properties, padded_rock_properties = apply_boundary_conditions(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            boundary_conditions=boundary_conditions,
            cell_dimension=cell_dimension,
            grid_shape=grid_shape,
            thickness_grid=thickness_grid,
            time=0.0,
        )

        if miscibility_model != "immiscible":
            logger.debug(
                f"Using explicit miscible saturation evolution: '{miscibility_model}'"
            )
            evolve_saturation = evolve_miscible_saturation_explicitly
        else:
            logger.debug("Using explicit immiscible saturation evolution")
            evolve_saturation = evolve_saturation_explicitly

        # Initialize fluid properties before starting the simulation
        # To ensure all dependent properties are consistent with initial pressure and saturation conditions
        logger.debug("Initializing PVT fluid properties for simulation start")
        padded_fluid_properties = update_pvt_properties(
            fluid_properties=padded_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
        )

        # Unpad the fluid properties back to the original grid shape for model state snapshots
        model = attrs.evolve(
            model, fluid_properties=padded_fluid_properties.unpad(pad_width=1)
        )
        (
            padded_relperm_grids,
            padded_relative_mobility_grids,
            padded_capillary_pressure_grids,
        ) = build_rock_fluid_properties_grids(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            rock_fluid_properties=rock_fluid_properties,
            disable_capillary_effects=config.disable_capillary_effects,
            capillary_strength_factor=config.capillary_strength_factor,
            relative_mobility_range=config.relative_mobility_range,
        )
        relative_mobility_grids = padded_relative_mobility_grids.unpad(pad_width=1)
        relperm_grids = padded_relperm_grids.unpad(pad_width=1)
        capillary_pressure_grids = padded_capillary_pressure_grids.unpad(pad_width=1)
        zeros_grid = build_uniform_grid(model.grid_shape, value=0.0)
        oil_injection_grid = zeros_grid
        water_injection_grid = zeros_grid.copy()
        gas_injection_grid = zeros_grid.copy()
        oil_production_grid = zeros_grid.copy()
        water_production_grid = zeros_grid.copy()
        gas_production_grid = zeros_grid.copy()
        state = ModelState(
            step=0,
            step_size=timer.step_size,
            model=model,
            wells=wells,
            config=config,
            relative_mobilities=relative_mobility_grids,
            relative_permeabilities=relperm_grids,
            capillary_pressures=capillary_pressure_grids,
            injection=RateGrids(
                oil=oil_injection_grid,
                water=water_injection_grid,
                gas=gas_injection_grid,
            ),
            production=RateGrids(
                oil=oil_production_grid,
                water=water_production_grid,
                gas=gas_production_grid,
            ),
        )

        # Yield the Initial model state
        logger.debug("Yielding initial model state")
        yield state

        no_flow_pressure_bc = isinstance(
            boundary_conditions["pressure"], type(default_bc)
        )

        while not timer.done():
            # WE FIRST PROPOSE THE TIME STEP SIZE FOR THE NEXT STEP
            # `timer.step` is still the last accepted step
            # since we have not accepted the new step size proposal and hence, the step yet
            # So we use `timer.next_step` to indicate the new step we are attempting
            new_step = timer.next_step
            step_size = timer.propose_step_size()
            logger.debug(
                f"Attempting time step {new_step} with size {step_size} seconds..."
            )
            try:
                if has_wells:
                    logger.debug(
                        f"Updating wells configuration for time step {new_step}"
                    )
                    wells.evolve(state)
                    logger.debug("Wells updated.")

                if scheme == "implicit":
                    result = _run_implicit_step(
                        time_step=new_step,
                        zeros_grid=zeros_grid,
                        cell_dimension=cell_dimension,
                        padded_thickness_grid=padded_thickness_grid,
                        padded_elevation_grid=padded_elevation_grid,
                        time_step_size=step_size,
                        padded_rock_properties=padded_rock_properties,
                        padded_fluid_properties=padded_fluid_properties,
                        rock_fluid_properties=rock_fluid_properties,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        boundary_conditions=boundary_conditions,
                        miscibility_model=miscibility_model,
                        config=config,
                    )
                elif scheme == "impes":
                    result = _run_impes_step(
                        time_step=new_step,
                        zeros_grid=zeros_grid,
                        cell_dimension=cell_dimension,
                        padded_thickness_grid=padded_thickness_grid,
                        padded_elevation_grid=padded_elevation_grid,
                        time_step_size=step_size,
                        padded_rock_properties=padded_rock_properties,
                        padded_fluid_properties=padded_fluid_properties,
                        padded_relperm_grids=padded_relperm_grids,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        miscibility_model=miscibility_model,
                        evolve_saturation=evolve_saturation,
                        config=config,
                    )
                else:
                    result = _run_explicit_step(
                        time_step=new_step,
                        zeros_grid=zeros_grid,
                        cell_dimension=cell_dimension,
                        padded_thickness_grid=padded_thickness_grid,
                        padded_elevation_grid=padded_elevation_grid,
                        time_step_size=step_size,
                        padded_rock_properties=padded_rock_properties,
                        padded_fluid_properties=padded_fluid_properties,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        miscibility_model=miscibility_model,
                        evolve_saturation=evolve_saturation,
                        config=config,
                    )

                # IF THE STEP WAS SUCCESSFUL, ACCEPT THAT STEP PROPOSAL
                if result.success:
                    # Now we can accept the proposed time step size
                    # and we now agree that this is a new step
                    logger.debug(f"Time step {new_step} completed successfully.")
                    actual_step_size = timer.accept_step(
                        step_size=step_size, **result.accept_kwargs
                    )
                    log_progress(
                        step=timer.step,
                        step_size=actual_step_size,
                        time_elapsed=timer.elapsed_time,
                        total_time=timer.simulation_time,
                        is_last_step=timer.is_last_step,
                        interval=config.log_interval,
                    )
                else:
                    # REJECT AND ADJUST THE TIME STEP SIZE AND RETRY
                    logger.warning(
                        f"Time step {new_step} failed. Retrying with smaller time step."
                    )
                    try:
                        timer.reject_step(
                            step_size=step_size, aggressive=timer.rejection_count > 10
                        )
                    except TimingError as exc:
                        raise SimulationError(
                            f"Simulation failed at time step {new_step} and cannot reduce time step further. {exc}."
                            f"\n{result.message}"
                        ) from exc
                    continue  # Retry the time step with a smaller size

                # Get the updated fluid properties, which will also be used for the next time step
                padded_fluid_properties = result.fluid_properties

                # Take a snapshot of the model state at specified intervals and at the last time step
                if (timer.step % output_frequency == 0) or timer.is_last_step:
                    logger.debug(f"Capturing model state at time step {timer.step}")
                    # The production rates are negative in the evolution
                    # so we need to negate them to report positive production values
                    logger.debug(
                        "Preparing injection and production rate grids for output"
                    )
                    # Unpack results
                    oil_injection_grid = result.oil_injection_grid
                    water_injection_grid = result.water_injection_grid
                    gas_injection_grid = result.gas_injection_grid
                    oil_production_grid = result.oil_production_grid
                    water_production_grid = result.water_production_grid
                    gas_production_grid = result.gas_production_grid

                    oil_production_grid = typing.cast(
                        NDimensionalGrid[ThreeDimensions],
                        np.negative(oil_production_grid),
                    )
                    water_production_grid = typing.cast(
                        NDimensionalGrid[ThreeDimensions],
                        np.negative(water_production_grid),
                    )
                    gas_production_grid = typing.cast(
                        NDimensionalGrid[ThreeDimensions],
                        np.negative(gas_production_grid),
                    )
                    injection_rates = RateGrids(
                        oil=oil_injection_grid,
                        water=water_injection_grid,
                        gas=gas_injection_grid,
                    )
                    production_rates = RateGrids(
                        oil=oil_production_grid,
                        water=water_production_grid,
                        gas=gas_production_grid,
                    )
                    logger.debug("Creating model state snapshot")
                    # Capture the current state of the wells
                    wells_snapshot = copy.deepcopy(wells)
                    # Capture the current model with updated fluid properties
                    model_snapshot = attrs.evolve(
                        model,
                        fluid_properties=padded_fluid_properties.unpad(pad_width=1),
                    )
                    relative_mobility_grids = padded_relative_mobility_grids.unpad(
                        pad_width=1
                    )
                    relperm_grids = padded_relperm_grids.unpad(pad_width=1)
                    capillary_pressure_grids = padded_capillary_pressure_grids.unpad(
                        pad_width=1
                    )
                    state = ModelState(
                        step=timer.step,
                        step_size=timer.step_size,
                        model=model_snapshot,
                        wells=wells_snapshot,
                        config=config,
                        relative_mobilities=relative_mobility_grids,
                        relative_permeabilities=relperm_grids,
                        capillary_pressures=capillary_pressure_grids,
                        injection=injection_rates,
                        production=production_rates,
                    )
                    logger.debug("Yielding model state snapshot")
                    yield state

                if timer.is_last_step:
                    # Break the loop if we've reached the final time step
                    # to avoid unnecessary computations
                    break

                next_step = timer.next_step
                # Apply boundary conditions before pressure update for the next time step
                logger.debug(
                    f"Applying boundary conditions for time step {next_step}..."
                )
                padded_fluid_properties, padded_rock_properties = (
                    apply_boundary_conditions(
                        fluid_properties=padded_fluid_properties,
                        rock_properties=padded_rock_properties,
                        boundary_conditions=boundary_conditions,
                        cell_dimension=cell_dimension,
                        grid_shape=grid_shape,
                        thickness_grid=thickness_grid,
                        time=next_step * step_size,
                    )
                )
                logger.debug("Boundary conditions applied.")
                # If the pressure boundary condition is not no-flow, Then apply PVT update before pressure evolution
                # since most PVT properties depend on pressure. This is skipped for no-flow BCs to save computation.
                # because mirroring neighbour values for PVT properties is sufficient for no-flow BCs.
                if no_flow_pressure_bc is False:
                    logger.debug(
                        "Updating PVT fluid properties due to boundary condition changes..."
                    )
                    padded_fluid_properties = update_pvt_properties(
                        fluid_properties=padded_fluid_properties,
                        wells=wells,
                        miscibility_model=miscibility_model,
                    )
                    logger.debug("PVT fluid properties updated")

                # Build relative permeability, relative mobility, and capillary pressure grids
                logger.debug(
                    f"Rebuilding rock-fluid property grids for time step {next_step}..."
                )
                (
                    padded_relperm_grids,
                    padded_relative_mobility_grids,
                    padded_capillary_pressure_grids,
                ) = build_rock_fluid_properties_grids(
                    fluid_properties=padded_fluid_properties,
                    rock_properties=padded_rock_properties,
                    rock_fluid_properties=rock_fluid_properties,
                    disable_capillary_effects=config.disable_capillary_effects,
                    capillary_strength_factor=config.capillary_strength_factor,
                    relative_mobility_range=config.relative_mobility_range,
                )

            except Exception as exc:
                raise SimulationError(
                    f"Simulation failed at time step {timer.step} due to error: {exc}"
                ) from exc

    logger.info(f"Simulation completed successfully after {timer.step} time steps")
