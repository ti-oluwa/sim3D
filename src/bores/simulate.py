"""Run a simulation workflow on a 3-Dimensional reservoir model."""

import copy
import logging
import typing

import attrs
import numpy as np

from bores._precision import get_dtype
from bores.boundary_conditions import default_bc
from bores.config import Config
from bores.constants import c
from bores.diffusivity import (
    evolve_miscible_saturation_explicitly,
    evolve_pressure_explicitly,
    evolve_pressure_implicitly,
    evolve_saturation_explicitly,
)
from bores.errors import SimulationError, StopSimulation, TimingError
from bores.grids.base import (
    CapillaryPressureGrids,
    RateGrids,
    RelPermGrids,
    RelativeMobilityGrids,
    _RateGridsProxy,
    build_uniform_grid,
    pad_grid,
)
from bores.grids.boundary_conditions import (
    apply_boundary_conditions,
    mirror_neighbour_cells,
)
from bores.grids.pvt import build_three_phase_relative_mobilities_grids
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.grids.updates import update_pvt_grids, update_residual_saturation_grids
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockProperties,
    SaturationHistory,
)
from bores.states import ModelState
from bores.timing import Timer
from bores.types import MiscibilityModel, NDimension, NDimensionalGrid, ThreeDimensions
from bores.utils import clip
from bores.wells import Wells


__all__ = ["run"]

logger = logging.getLogger(__name__)


UNPHYSICAL_PRESSURE_ERROR_MSG = """
Unphysical pressure encountered in the pressure grid at the following indices:

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
    rock_properties: RockProperties[NDimension]
    """Updated rock properties after the time step."""
    saturation_history: SaturationHistory[NDimension]
    """Updated saturation history after the time step."""
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
    timer_kwargs: typing.Dict[str, typing.Any] = attrs.field(factory=dict)
    """Kwargs that should be passed to the simulation timer on accepting or rejecting a step."""


@attrs.frozen
class SaturationChangeCheckResult:
    violated: bool
    max_phase_saturation_change: float
    max_allowed_phase_saturation_change: float
    message: typing.Optional[str] = None


def _check_saturation_changes(
    max_oil_saturation_change: float,
    max_water_saturation_change: float,
    max_gas_saturation_change: float,
    max_allowed_oil_saturation_change: float,
    max_allowed_water_saturation_change: float,
    max_allowed_gas_saturation_change: float,
    tolerance: float = 1e-4,
) -> SaturationChangeCheckResult:
    """
    Check if the saturation changes for each phase exceed their allowed maximums.
    If any phase exceeds its limit, return a violation result. If multiple phases exceed
    their limits, return the maximum phase saturation change and the corresponding allowed maximum.
    If no violations occur, report a non-violated result with zeros for the max changes.

    :param max_oil_saturation_change: Maximum oil saturation change observed.
    :param max_water_saturation_change: Maximum water saturation change observed.
    :param max_gas_saturation_change: Maximum gas saturation change observed.
    :param max_allowed_oil_saturation_change: Maximum allowed oil saturation change.
    :param max_allowed_water_saturation_change: Maximum allowed water saturation change.
    :param max_allowed_gas_saturation_change: Maximum allowed gas saturation change.
    :param tolerance: Relative tolerance for saturation change checks.
    :return: `SaturationChangeCheckResult` indicating if any saturation change limits were violated.
    """
    violated = False
    messages = []
    max_phase_saturation_change = 0.0
    max_allowed_phase_saturation_change = 0.0

    oil_tolerance = max(tolerance, 0.005 * max_allowed_oil_saturation_change)
    max_tolerable_oil_saturation_change = max_allowed_oil_saturation_change * (
        1 + oil_tolerance
    )
    if max_oil_saturation_change > max_tolerable_oil_saturation_change:
        violated = True
        # Start assuming oil saturation change is the largest so far
        max_phase_saturation_change = max_oil_saturation_change
        max_allowed_phase_saturation_change = max_allowed_oil_saturation_change
        messages.append(
            f"Oil saturation change {max_oil_saturation_change:.6f} exceeded maximum allowed {max_allowed_oil_saturation_change:.6f}."
        )

    water_tolerance = max(tolerance, 0.005 * max_allowed_water_saturation_change)
    max_tolerable_water_saturation_change = max_allowed_water_saturation_change * (
        1 + water_tolerance
    )
    if max_water_saturation_change > max_tolerable_water_saturation_change:
        violated = True
        # If water saturation change is the largest so far, update the max values
        if max_water_saturation_change > max_phase_saturation_change:
            max_phase_saturation_change = max_water_saturation_change
            max_allowed_phase_saturation_change = max_allowed_water_saturation_change
        messages.append(
            f"Water saturation change {max_water_saturation_change:.6f} exceeded maximum allowed {max_allowed_water_saturation_change:.6f}."
        )

    gas_tolerance = max(tolerance, 0.005 * max_allowed_gas_saturation_change)
    max_tolerable_gas_saturation_change = max_allowed_gas_saturation_change * (
        1 + gas_tolerance
    )
    if max_gas_saturation_change > max_tolerable_gas_saturation_change:
        violated = True
        # If gas saturation change is the largest so far, update the max values
        if max_gas_saturation_change > max_phase_saturation_change:
            max_phase_saturation_change = max_gas_saturation_change
            max_allowed_phase_saturation_change = max_allowed_gas_saturation_change
        messages.append(
            f"Gas saturation change {max_gas_saturation_change:.6f} exceeded maximum allowed {max_allowed_gas_saturation_change:.6f}."
        )

    message = "\n".join(messages) if messages else None
    return SaturationChangeCheckResult(
        violated=violated,
        max_phase_saturation_change=max_phase_saturation_change,
        max_allowed_phase_saturation_change=max_allowed_phase_saturation_change,
        message=message,
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
    padded_saturation_history: SaturationHistory[ThreeDimensions],
    padded_relperm_grids: RelPermGrids[ThreeDimensions],
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using IMPES (Implicit Pressure, Explicit Saturation).

    :param time_step: Current time step index.
    :param zeros_grid: Grid of zeros for rate tracking.
    :param cell_dimension: Tuple of cell dimensions (dx, dy).
    :param padded_thickness_grid: Padded thickness grid.
    :param padded_elevation_grid: Padded elevation grid.
    :param time_step_size: Size of the current time step.
    :param padded_rock_properties: Padded rock properties.
    :param padded_fluid_properties: Padded fluid properties.
    :param padded_saturation_history: Padded saturation history.
    :param padded_relperm_grids: Padded relative permeability grids.
    :param padded_relative_mobility_grids: Padded relative mobility grids.
    :param padded_capillary_pressure_grids: Padded capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :return: `StepResult` containing updated rates and fluid properties.
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
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    padded_pressure_grid = pressure_solution.pressure_grid
    max_pressure_change = pressure_solution.max_pressure_change
    max_allowed_pressure_change = config.max_pressure_change
    if max_pressure_change > max_allowed_pressure_change:
        message = (
            f"Pressure change {max_pressure_change:.6f} psi "
            f"exceeded maximum allowed {max_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.error(message)
        return StepResult(
            oil_injection_grid=zeros_grid,
            water_injection_grid=zeros_grid,
            gas_injection_grid=zeros_grid,
            oil_production_grid=zeros_grid,
            water_production_grid=zeros_grid,
            gas_production_grid=zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "max_pressure_change": max_pressure_change,
                "max_allowed_pressure_change": max_allowed_pressure_change,
            },
        )

    # Check for any out-of-range pressures
    min_allowable_pressure = c.MIN_VALID_PRESSURE - 1e-3
    max_allowable_pressure = c.MAX_VALID_PRESSURE + 1e-3
    out_of_range_mask = (padded_pressure_grid < min_allowable_pressure) | (
        padded_pressure_grid > max_allowable_pressure
    )
    out_of_range_indices = np.argwhere(out_of_range_mask)
    if out_of_range_indices.size > 0:
        min_pressure = np.min(padded_pressure_grid)
        max_pressure = np.max(padded_pressure_grid)
        logger.warning(
            f"Unphysical pressure detected at {out_of_range_indices.size} cells. "
            f"Range: [{min_pressure:.4f}, {max_pressure:.4f}] psi. Allowed: [{min_allowable_pressure}, {max_allowable_pressure}]."
        )
        message = ""
        if min_pressure < min_allowable_pressure:
            message += f"Pressure dropped below {min_allowable_pressure} psi (Min: {min_pressure:.4f}).\n"
        if max_pressure > max_allowable_pressure:
            message += f"Pressure exceeded {max_allowable_pressure} psi (Max: {max_pressure:.4f}).\n"

        message += (
            UNPHYSICAL_PRESSURE_ERROR_MSG.format(indices=out_of_range_indices.tolist())
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
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
        )

    dtype = get_dtype()
    # Clamp pressures to valid range just for additional safety and to remove numerical noise
    padded_pressure_grid = clip(
        padded_pressure_grid, c.MIN_VALID_PRESSURE, c.MAX_VALID_PRESSURE
    ).astype(dtype, copy=False)

    # Update fluid properties with new pressure grid
    logger.debug("Updating fluid properties with new pressure grid...")
    padded_fluid_properties = attrs.evolve(
        padded_fluid_properties, pressure_grid=padded_pressure_grid
    )
    logger.debug("Pressure evolution completed!")

    # For IMPES, we need to update the fluid properties
    # before proceeding to saturation evolution.
    logger.debug("Updating PVT fluid properties for saturation evolution")
    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
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
    # NOTE: Important design decision! We would normally apply these clamps to active
    # phases only, i.e where "S > Sirr + phase tolerance". This respects the physics but leads to numerical
    # instability as phase mobility can become zero and hence transmissibilities, and hence diagonals in the
    # the sparse matrix can be zeroed out making the matrix singular. Therefore, we clamp all to a very small
    # non-zero value to ensure numerical stability.
    padded_water_relative_mobility_grid = config.relative_mobility_range["water"].clip(
        padded_water_relative_mobility_grid
    )
    padded_oil_relative_mobility_grid = config.relative_mobility_range["oil"].clip(
        padded_oil_relative_mobility_grid
    )
    padded_gas_relative_mobility_grid = config.relative_mobility_range["gas"].clip(
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

    if miscibility_model == "immiscible":
        evolve_saturation = evolve_saturation_explicitly
    else:
        evolve_saturation = evolve_miscible_saturation_explicitly

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
    saturation_solution = saturation_result.value
    saturation_change_result = _check_saturation_changes(
        max_oil_saturation_change=saturation_solution.max_oil_saturation_change,
        max_water_saturation_change=saturation_solution.max_water_saturation_change,
        max_gas_saturation_change=saturation_solution.max_gas_saturation_change,
        max_allowed_oil_saturation_change=config.max_oil_saturation_change,
        max_allowed_water_saturation_change=config.max_water_saturation_change,
        max_allowed_gas_saturation_change=config.max_gas_saturation_change,
    )
    timer_kwargs = {
        "max_cfl_encountered": saturation_solution.max_cfl_encountered,
        "cfl_threshold": saturation_solution.cfl_threshold,
        "max_pressure_change": max_pressure_change,
        "max_allowed_pressure_change": max_allowed_pressure_change,
        "max_saturation_change": saturation_change_result.max_phase_saturation_change
        or None,
        "max_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
        or None,
    }

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
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=saturation_result.message,
            timer_kwargs=timer_kwargs,
        )

    if saturation_change_result.violated:
        message = f"""
        At time step {time_step}, saturation change limits were violated:
        {saturation_change_result.message} 

        Oil saturation change: {saturation_solution.max_oil_saturation_change:.6f}, 
        Water saturation change: {saturation_solution.max_water_saturation_change:.6f}, 
        Gas saturation change: {saturation_solution.max_gas_saturation_change:.6f}.
        """
        logger.debug(message)
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
        )

    logger.debug("Updating fluid properties with new saturation grids...")
    padded_water_saturation_grid = saturation_solution.water_saturation_grid.astype(
        dtype, copy=False
    )
    padded_oil_saturation_grid = saturation_solution.oil_saturation_grid.astype(
        dtype, copy=False
    )
    padded_gas_saturation_grid = saturation_solution.gas_saturation_grid.astype(
        dtype, copy=False
    )
    padded_solvent_concentration_grid = saturation_solution.solvent_concentration_grid

    if padded_solvent_concentration_grid is None:
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
        )
    else:
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            solvent_concentration_grid=padded_solvent_concentration_grid.astype(
                dtype, copy=False
            ),
        )

    # Update residual saturation grids based on new saturations
    padded_rock_properties, padded_saturation_history = (
        update_residual_saturation_grids(
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            water_saturation_grid=padded_water_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )
    )
    logger.debug("Saturation evolution completed!")
    return StepResult(
        oil_injection_grid=oil_injection_grid,
        water_injection_grid=water_injection_grid,
        gas_injection_grid=gas_injection_grid,
        oil_production_grid=oil_production_grid,
        water_production_grid=water_production_grid,
        gas_production_grid=gas_production_grid,
        fluid_properties=padded_fluid_properties,
        rock_properties=padded_rock_properties,
        saturation_history=padded_saturation_history,
        success=True,
        message=saturation_result.message,
        timer_kwargs=timer_kwargs,
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
    padded_saturation_history: SaturationHistory[ThreeDimensions],
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using fully explicit scheme (explicit pressure and saturation).

    :param time_step: Current time step index.
    :param zeros_grid: Grid of zeros for rate tracking.
    :param cell_dimension: Tuple of cell dimensions (dx, dy).
    :param padded_thickness_grid: Padded thickness grid.
    :param padded_elevation_grid: Padded elevation grid.
    :param time_step_size: Size of the current time step.
    :param padded_rock_properties: Padded rock properties.
    :param padded_fluid_properties: Padded fluid properties.
    :param padded_saturation_history: Padded saturation history.
    :param padded_relperm_grids: Padded relative permeability grids.
    :param padded_relative_mobility_grids: Padded relative mobility grids.
    :param padded_capillary_pressure_grids: Padded capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :return: `StepResult` containing updated rates and fluid properties.
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
    pressure_solution = pressure_result.value
    max_pressure_change = pressure_solution.max_pressure_change
    max_allowed_pressure_change = config.max_pressure_change
    timer_kwargs = {
        "max_cfl_encountered": pressure_solution.max_cfl_encountered,
        "cfl_threshold": pressure_solution.cfl_threshold,
        "max_pressure_change": max_pressure_change,
        "max_allowed_pressure_change": max_allowed_pressure_change,
    }

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
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=pressure_result.message,
            timer_kwargs=timer_kwargs,
        )

    if max_pressure_change > max_allowed_pressure_change:
        message = (
            f"Pressure change {max_pressure_change:.6f} psi "
            f"exceeded maximum allowed {max_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.debug(message)
        return StepResult(
            oil_injection_grid=zeros_grid,
            water_injection_grid=zeros_grid,
            gas_injection_grid=zeros_grid,
            oil_production_grid=zeros_grid,
            water_production_grid=zeros_grid,
            gas_production_grid=zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "max_pressure_change": max_pressure_change,
                "max_allowed_pressure_change": max_allowed_pressure_change,
            },
        )

    padded_pressure_grid = pressure_solution.pressure_grid

    # Check for any out-of-range pressures
    min_allowable_pressure = c.MIN_VALID_PRESSURE - 1e-3
    max_allowable_pressure = c.MAX_VALID_PRESSURE + 1e-3
    out_of_range_mask = (padded_pressure_grid < min_allowable_pressure) | (
        padded_pressure_grid > max_allowable_pressure
    )
    out_of_range_indices = np.argwhere(out_of_range_mask)
    if out_of_range_indices.size > 0:
        min_pressure = np.min(padded_pressure_grid)
        max_pressure = np.max(padded_pressure_grid)
        logger.warning(
            f"Unphysical pressure detected at {out_of_range_indices.size} cells. "
            f"Range: [{min_pressure:.4f}, {max_pressure:.4f}] psi. Allowed: [{min_allowable_pressure}, {max_allowable_pressure}]."
        )
        message = ""
        if min_pressure < min_allowable_pressure:
            message += f"Pressure dropped below {min_allowable_pressure} psi (Min: {min_pressure:.4f}).\n"
        if max_pressure > max_allowable_pressure:
            message += f"Pressure exceeded {max_allowable_pressure} psi (Max: {max_pressure:.4f}).\n"

        message += (
            UNPHYSICAL_PRESSURE_ERROR_MSG.format(indices=out_of_range_indices.tolist())
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
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
        )

    dtype = get_dtype()
    # Clamp pressures to valid range just for additional safety and to remove numerical noise
    padded_pressure_grid = clip(
        padded_pressure_grid, c.MIN_VALID_PRESSURE, c.MAX_VALID_PRESSURE
    ).astype(dtype, copy=False)
    logger.debug("Pressure evolution completed!")

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

    if miscibility_model == "immiscible":
        evolve_saturation = evolve_saturation_explicitly
    else:
        evolve_saturation = evolve_miscible_saturation_explicitly

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
    saturation_solution = saturation_result.value
    saturation_change_result = _check_saturation_changes(
        max_oil_saturation_change=saturation_solution.max_oil_saturation_change,
        max_water_saturation_change=saturation_solution.max_water_saturation_change,
        max_gas_saturation_change=saturation_solution.max_gas_saturation_change,
        max_allowed_oil_saturation_change=config.max_oil_saturation_change,
        max_allowed_water_saturation_change=config.max_water_saturation_change,
        max_allowed_gas_saturation_change=config.max_gas_saturation_change,
    )
    timer_kwargs = {
        "max_cfl_encountered": saturation_solution.max_cfl_encountered,
        "cfl_threshold": saturation_solution.cfl_threshold,
        "max_pressure_change": max_pressure_change,
        "max_allowed_pressure_change": max_allowed_pressure_change,
        "max_saturation_change": saturation_change_result.max_phase_saturation_change
        or None,
        "max_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
        or None,
    }

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
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=saturation_result.message,
            timer_kwargs=timer_kwargs,
        )

    if saturation_change_result.violated:
        message = f"""
        At time step {time_step}, saturation change limits were violated:
        {saturation_change_result.message} 

        Oil saturation change: {saturation_solution.max_oil_saturation_change:.6f}, 
        Water saturation change: {saturation_solution.max_water_saturation_change:.6f}, 
        Gas saturation change: {saturation_solution.max_gas_saturation_change:.6f}.
        """
        logger.error(message)
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
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

    padded_water_saturation_grid = saturation_solution.water_saturation_grid.astype(
        dtype, copy=False
    )
    padded_oil_saturation_grid = saturation_solution.oil_saturation_grid.astype(
        dtype, copy=False
    )
    padded_gas_saturation_grid = saturation_solution.gas_saturation_grid.astype(
        dtype, copy=False
    )
    padded_solvent_concentration_grid = saturation_solution.solvent_concentration_grid
    if padded_solvent_concentration_grid is None:
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
        )
    else:
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            solvent_concentration_grid=padded_solvent_concentration_grid.astype(
                dtype, copy=False
            ),
        )

    # Update PVT properties with new state (pressure and saturations)
    logger.debug("Updating PVT fluid properties after explicit solve...")
    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
    )
    # Update residual saturation grids based on new saturations
    padded_rock_properties, padded_saturation_history = (
        update_residual_saturation_grids(
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            water_saturation_grid=padded_water_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )
    )
    return StepResult(
        fluid_properties=padded_fluid_properties,
        rock_properties=padded_rock_properties,
        saturation_history=padded_saturation_history,
        oil_injection_grid=oil_injection_grid,
        water_injection_grid=water_injection_grid,
        gas_injection_grid=gas_injection_grid,
        oil_production_grid=oil_production_grid,
        water_production_grid=water_production_grid,
        gas_production_grid=gas_production_grid,
        success=True,
        message=saturation_result.message,
        timer_kwargs=timer_kwargs,
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
    Run a simulation on a 3D static reservoir model and wells.

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

    logger.info("Starting simulation workflow...")

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
        padded_saturation_history = model.saturation_history.pad(pad_width=1)
        thickness_grid = model.thickness_grid
        padded_thickness_grid = pad_grid(thickness_grid, pad_width=1)
        padded_thickness_grid = mirror_neighbour_cells(padded_thickness_grid)
        elevation_grid = model.get_elevation_grid(
            apply_dip=not config.disable_structural_dip
        )
        padded_elevation_grid = pad_grid(elevation_grid, pad_width=1)
        padded_elevation_grid = mirror_neighbour_cells(padded_elevation_grid)

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

        # Initialize fluid properties before starting the simulation
        # To ensure all dependent properties are consistent with initial pressure and saturation conditions
        logger.debug("Initializing PVT fluid properties for simulation start")
        padded_fluid_properties = update_pvt_grids(
            fluid_properties=padded_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
        )

        # Unpad the fluid properties back to the original grid shape for model state snapshots
        model = model.evolve(
            fluid_properties=padded_fluid_properties.unpad(pad_width=1)
        )
        padded_water_saturation_grid = padded_fluid_properties.water_saturation_grid
        padded_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid
        padded_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid
        padded_irreducible_water_saturation_grid = (
            padded_rock_properties.irreducible_water_saturation_grid
        )
        padded_residual_oil_saturation_water_grid = (
            padded_rock_properties.residual_oil_saturation_water_grid
        )
        padded_residual_oil_saturation_gas_grid = (
            padded_rock_properties.residual_oil_saturation_gas_grid
        )
        padded_residual_gas_saturation_grid = (
            padded_rock_properties.residual_gas_saturation_grid
        )
        padded_water_viscosity_grid = padded_fluid_properties.water_viscosity_grid
        padded_oil_viscosity_grid = padded_fluid_properties.oil_effective_viscosity_grid
        padded_gas_viscosity_grid = padded_fluid_properties.gas_viscosity_grid
        relative_permeability_table = rock_fluid_properties.relative_permeability_table
        capillary_pressure_table = rock_fluid_properties.capillary_pressure_table
        (
            padded_relperm_grids,
            padded_relative_mobility_grids,
            padded_capillary_pressure_grids,
        ) = build_rock_fluid_properties_grids(
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            water_viscosity_grid=padded_water_viscosity_grid,
            oil_viscosity_grid=padded_oil_viscosity_grid,
            gas_viscosity_grid=padded_gas_viscosity_grid,
            relative_permeability_table=relative_permeability_table,
            capillary_pressure_table=capillary_pressure_table,
            disable_capillary_effects=config.disable_capillary_effects,
            capillary_strength_factor=config.capillary_strength_factor,
            relative_mobility_range=config.relative_mobility_range,
            phase_appearance_tolerance=config.phase_appearance_tolerance,
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
            step=timer.step,
            step_size=timer.step_size,
            time=timer.elapsed_time,
            model=model,
            wells=wells,
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
            timer_state=timer.dump_state(),
        )

        # Yield the initial model state
        logger.debug("Yielding initial model state")
        yield state

        no_flow_pressure_bc = isinstance(
            boundary_conditions["pressure"], type(default_bc)
        )
        while not timer.done():
            # WE FIRST PROPOSE THE TIME STEP SIZE FOR THE NEXT STEP
            # `timer.step` is still the last accepted step
            # since we have not accepted the new step size proposal and hence, the step yet.
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

                if new_step > 1:
                    # Apply boundary conditions before pressure update for the new time step
                    logger.debug(
                        f"Applying boundary conditions for time step {new_step}..."
                    )
                    padded_fluid_properties, padded_rock_properties = (
                        apply_boundary_conditions(
                            fluid_properties=padded_fluid_properties,
                            rock_properties=padded_rock_properties,
                            boundary_conditions=boundary_conditions,
                            cell_dimension=cell_dimension,
                            grid_shape=grid_shape,
                            thickness_grid=thickness_grid,
                            time=timer.elapsed_time + step_size,
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
                        padded_fluid_properties = update_pvt_grids(
                            fluid_properties=padded_fluid_properties,
                            wells=wells,
                            miscibility_model=miscibility_model,
                            pvt_tables=config.pvt_tables,
                        )
                        logger.debug("PVT fluid properties updated")

                    # Build relative permeability, relative mobility, and capillary pressure grids
                    logger.debug(
                        f"Rebuilding rock-fluid property grids for time step {new_step}..."
                    )
                    padded_water_saturation_grid = (
                        padded_fluid_properties.water_saturation_grid
                    )
                    padded_oil_saturation_grid = (
                        padded_fluid_properties.oil_saturation_grid
                    )
                    padded_gas_saturation_grid = (
                        padded_fluid_properties.gas_saturation_grid
                    )
                    padded_irreducible_water_saturation_grid = (
                        padded_rock_properties.irreducible_water_saturation_grid
                    )
                    padded_residual_oil_saturation_water_grid = (
                        padded_rock_properties.residual_oil_saturation_water_grid
                    )
                    padded_residual_oil_saturation_gas_grid = (
                        padded_rock_properties.residual_oil_saturation_gas_grid
                    )
                    padded_residual_gas_saturation_grid = (
                        padded_rock_properties.residual_gas_saturation_grid
                    )
                    padded_water_viscosity_grid = (
                        padded_fluid_properties.water_viscosity_grid
                    )
                    padded_oil_viscosity_grid = (
                        padded_fluid_properties.oil_effective_viscosity_grid
                    )
                    padded_gas_viscosity_grid = (
                        padded_fluid_properties.gas_viscosity_grid
                    )
                    (
                        padded_relperm_grids,
                        padded_relative_mobility_grids,
                        padded_capillary_pressure_grids,
                    ) = build_rock_fluid_properties_grids(
                        water_saturation_grid=padded_water_saturation_grid,
                        oil_saturation_grid=padded_oil_saturation_grid,
                        gas_saturation_grid=padded_gas_saturation_grid,
                        irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
                        residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
                        residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
                        residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
                        water_viscosity_grid=padded_water_viscosity_grid,
                        oil_viscosity_grid=padded_oil_viscosity_grid,
                        gas_viscosity_grid=padded_gas_viscosity_grid,
                        relative_permeability_table=relative_permeability_table,
                        capillary_pressure_table=capillary_pressure_table,
                        disable_capillary_effects=config.disable_capillary_effects,
                        capillary_strength_factor=config.capillary_strength_factor,
                        relative_mobility_range=config.relative_mobility_range,
                        phase_appearance_tolerance=config.phase_appearance_tolerance,
                    )

                if scheme == "implicit":
                    logger.warning(
                        "Implicit scheme selected but not yet supported. Falling back to IMPES scheme.",
                        UserWarning,
                    )
                    scheme = "impes"

                if scheme == "impes":
                    result = _run_impes_step(
                        time_step=new_step,
                        zeros_grid=zeros_grid,
                        cell_dimension=cell_dimension,
                        padded_thickness_grid=padded_thickness_grid,
                        padded_elevation_grid=padded_elevation_grid,
                        time_step_size=step_size,
                        padded_rock_properties=padded_rock_properties,
                        padded_fluid_properties=padded_fluid_properties,
                        padded_saturation_history=padded_saturation_history,
                        padded_relperm_grids=padded_relperm_grids,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        miscibility_model=miscibility_model,
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
                        padded_saturation_history=padded_saturation_history,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        miscibility_model=miscibility_model,
                        config=config,
                    )

                # IF THE STEP WAS SUCCESSFUL, ACCEPT THAT STEP PROPOSAL
                if result.success:
                    # Now we can accept the proposed time step size and we now agree that this is a new step
                    logger.debug(f"Time step {new_step} completed successfully.")
                    timer.accept_step(step_size=step_size, **result.timer_kwargs)
                    if config.log_interval:
                        log_progress(
                            step=timer.step,
                            step_size=step_size,
                            time_elapsed=timer.elapsed_time,
                            total_time=timer.simulation_time,
                            is_last_step=timer.is_last_step,
                            interval=config.log_interval,
                        )
                else:
                    # REJECT AND ADJUST THE TIME STEP SIZE AND RETRY
                    logger.debug(
                        f"Time step {new_step} failed with step size {step_size}. Retrying with smaller step size."
                    )
                    try:
                        timer.reject_step(
                            step_size=step_size,
                            aggressive=timer.rejection_count > 5,
                            **result.timer_kwargs,
                        )
                    except TimingError as exc:
                        raise SimulationError(
                            f"Simulation failed at time step {new_step} and cannot reduce step size further. {exc}."
                            f"\n{result.message}"
                        ) from exc
                    continue  # Retry the time step with a smaller size

                # Get the updated fluid properties, which will also be used for the next time step
                padded_fluid_properties = result.fluid_properties
                padded_rock_properties = result.rock_properties
                padded_saturation_history = result.saturation_history

                # Take a snapshot of the model state at start, at specified intervals and at the last time step
                if (
                    timer.step == 1
                    or (timer.step % output_frequency == 0)
                    or timer.is_last_step
                ):
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
                    logger.debug("Taking model state snapshot")
                    # Capture the current state of the wells
                    wells_snapshot = copy.deepcopy(wells)
                    # Capture the current model with updated fluid properties
                    model_snapshot = model.evolve(
                        fluid_properties=padded_fluid_properties.unpad(pad_width=1),
                        rock_properties=padded_rock_properties.unpad(pad_width=1),
                        saturation_history=padded_saturation_history.unpad(pad_width=1),
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
                        time=timer.elapsed_time,
                        model=model_snapshot,
                        wells=wells_snapshot,
                        relative_mobilities=relative_mobility_grids,
                        relative_permeabilities=relperm_grids,
                        capillary_pressures=capillary_pressure_grids,
                        injection=injection_rates,
                        production=production_rates,
                        timer_state=timer.dump_state(),
                    )
                    logger.debug("Yielding model state")
                    yield state

            except StopSimulation as exc:
                logger.info(f"Stopping simulation on request: {exc}")
                break
            except Exception as exc:
                raise SimulationError(
                    f"Simulation failed while attempting time step {new_step} due to error: {exc}"
                ) from exc

    logger.info(f"Simulation completed successfully after {timer.step} time steps")
