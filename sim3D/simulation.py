"""Dynamic simulation on a 3-Dimensional reservoir model with specified properties and wells."""

import copy
import logging
import math
import typing

from attrs import evolve
import numpy as np

from sim3D.boundaries import BoundaryConditions, BoundaryMetadata
from sim3D.diffusivity import (
    evolve_miscible_saturation_explicitly,
    evolve_pressure_explicitly,
    evolve_pressure_implicitly,
    evolve_saturation_explicitly,
)
from sim3D.grids.base import build_uniform_grid, pad_grid
from sim3D.grids.properties import (
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_solubility_in_water_grid,
    build_gas_viscosity_grid,
    build_live_oil_density_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_compressibility_grid,
    build_oil_effective_viscosity_grid,
    build_oil_formation_volume_factor_grid,
    build_oil_viscosity_grid,
    build_solution_gas_to_oil_ratio_grid,
    build_three_phase_capillary_pressure_grids,
    build_three_phase_relative_mobilities_grids,
    build_three_phase_relative_permeabilities_grids,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from sim3D.models import (
    FluidProperties,
    ReservoirModel,
    RockFluidProperties,
    RockProperties,
)
from sim3D.states import ModelState
from sim3D.types import (
    CapillaryPressureGrids,
    MiscibilityModel,
    NDimension,
    NDimensionalGrid,
    Options,
    RateGrids,
    RelPermGrids,
    RelativeMobilityGrids,
    ThreeDimensions,
    _RateGridsProxy,
)
from sim3D.wells import Wells


__all__ = ["run", "build_rock_fluid_properties_grids"]

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

# Default boundary condition instance for mirroring neighbours
# NoFlowBoundary is returned by default for all properties
default_bc = BoundaryConditions()["__default__"]


def _mirror_neighbour(
    grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """Mirrors the neighbour cells for boundary padding."""
    default_bc.apply(grid)
    return grid


def _apply_boundary_conditions(
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
) -> typing.Tuple[FluidProperties, RockProperties]:
    """
    Applies boundary conditions to the fluid property grids.

    :param fluid_properties: The padded fluid properties.
    :param rock_properties: The padded rock properties.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param time: The current simulation time.
    """
    boundary_conditions["pressure"].apply(
        fluid_properties.pressure_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="pressure",
        ),
    )
    boundary_conditions["oil_saturation"].apply(
        fluid_properties.oil_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="oil_saturation",
        ),
    )
    boundary_conditions["water_saturation"].apply(
        fluid_properties.water_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="water_saturation",
        ),
    )
    boundary_conditions["gas_saturation"].apply(
        fluid_properties.gas_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="gas_saturation",
        ),
    )
    boundary_conditions["temperature"].apply(
        fluid_properties.temperature_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="temperature",
        ),
    )
    excluded_fluid_properties = (
        "pressure_grid",
        "oil_saturation_grid",
        "water_saturation_grid",
        "gas_saturation_grid",
        "temperature_grid",
    )
    fluid_properties = fluid_properties.apply_hook(
        hook=_mirror_neighbour, exclude=excluded_fluid_properties
    )
    rock_properties = rock_properties.apply_hook(hook=_mirror_neighbour)
    return fluid_properties, rock_properties


def build_rock_fluid_properties_grids(
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    options: Options,
) -> typing.Tuple[
    RelPermGrids[ThreeDimensions],
    RelativeMobilityGrids[ThreeDimensions],
    CapillaryPressureGrids[ThreeDimensions],
]:
    """Builds the rock-fluid properties grids required for simulation."""
    logger.debug("Building relative permeability grids...")
    krw_grid, kro_grid, krg_grid = build_three_phase_relative_permeabilities_grids(
        water_saturation_grid=fluid_properties.water_saturation_grid,
        oil_saturation_grid=fluid_properties.oil_saturation_grid,
        gas_saturation_grid=fluid_properties.gas_saturation_grid,
        irreducible_water_saturation_grid=rock_properties.irreducible_water_saturation_grid,
        residual_oil_saturation_water_grid=rock_properties.residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=rock_properties.residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=rock_properties.residual_gas_saturation_grid,
        relative_permeability_table=rock_fluid_properties.relative_permeability_table,
    )
    logger.debug("Building relative mobility grids...")
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_three_phase_relative_mobilities_grids(
        oil_relative_permeability_grid=kro_grid,
        water_relative_permeability_grid=krw_grid,
        gas_relative_permeability_grid=krg_grid,
        water_viscosity_grid=fluid_properties.water_viscosity_grid,
        oil_viscosity_grid=fluid_properties.oil_effective_viscosity_grid,
        gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
    )
    # Clamp relative mobility grids to avoid numerical issues
    water_relative_mobility_grid = options.relative_mobility_range["water"].arrayclip(
        water_relative_mobility_grid
    )
    oil_relative_mobility_grid = options.relative_mobility_range["oil"].arrayclip(
        oil_relative_mobility_grid
    )
    gas_relative_mobility_grid = options.relative_mobility_range["gas"].arrayclip(
        gas_relative_mobility_grid
    )

    logger.debug("Building capillary pressure grids...")
    (
        oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid,
    ) = build_three_phase_capillary_pressure_grids(
        water_saturation_grid=fluid_properties.water_saturation_grid,
        gas_saturation_grid=fluid_properties.gas_saturation_grid,
        irreducible_water_saturation_grid=rock_properties.irreducible_water_saturation_grid,
        residual_oil_saturation_water_grid=rock_properties.residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=rock_properties.residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=rock_properties.residual_gas_saturation_grid,
        capillary_pressure_table=rock_fluid_properties.capillary_pressure_table,
    )
    relperm_grids = RelPermGrids[ThreeDimensions](
        kro=kro_grid, krw=krw_grid, krg=krg_grid
    )
    relative_mobility_grids = RelativeMobilityGrids(
        water_relative_mobility=water_relative_mobility_grid,
        oil_relative_mobility=oil_relative_mobility_grid,
        gas_relative_mobility=gas_relative_mobility_grid,
    )
    relative_mobility_grids = typing.cast(
        RelativeMobilityGrids[ThreeDimensions], relative_mobility_grids
    )
    capillary_pressure_grids = CapillaryPressureGrids[ThreeDimensions](
        oil_water_capillary_pressure=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure=gas_oil_capillary_pressure_grid,
    )
    return relperm_grids, relative_mobility_grids, capillary_pressure_grids


def run(
    model: ReservoirModel[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    options: Options,
) -> typing.Generator[ModelState[ThreeDimensions], None, None]:
    """
    Runs a dynamic simulation on a 3D reservoir model with specified properties and wells.

    The 3D simulation evolves pressure and saturation over time using the specified evolution scheme.
    3D simulations are computationally intensive and may require significant memory and processing power.

    :param model: The reservoir model containing grid, rock, and fluid properties.
    :param wells: The wells configuration for the simulation.
    :param options: Simulation run options and parameters.
    :yield: Yields the model state at specified output intervals.
    """
    logger.info("Starting 3D reservoir simulation")
    logger.debug(f"Grid dimensions: {model.grid_shape}")
    logger.debug(f"Cell dimensions: {model.cell_dimension}")
    logger.debug(f"Evolution scheme: {options.scheme}")
    logger.debug(f"Time step size: {options.time_step_size} seconds")
    logger.debug(f"Total simulation time: {options.total_time} seconds")

    # Use constants from options within the simulation
    with options.constants():
        cell_dimension = model.cell_dimension
        boundary_conditions = model.boundary_conditions
        grid_shape = model.grid_shape
        time_step_size = options.time_step_size
        # Pad fluid and rock properties grids and other necesary grids with ghost cells
        # for boundary condition application
        # Ensure ghost cells mirror neighbour values by default
        padded_fluid_properties = model.fluid_properties.pad(pad_width=1)
        padded_rock_properties = model.rock_properties.pad(pad_width=1)
        rock_fluid_properties = model.rock_fluid_properties
        thickness_grid = model.thickness_grid
        padded_thickness_grid = pad_grid(thickness_grid, pad_width=1)
        padded_thickness_grid = _mirror_neighbour(padded_thickness_grid)
        elevation_grid = model.get_elevation_grid(apply_dip=options.apply_dip)
        padded_elevation_grid = pad_grid(elevation_grid, pad_width=1)
        padded_elevation_grid = _mirror_neighbour(padded_elevation_grid)

        # Apply boundary conditions to relevant padded grids
        padded_fluid_properties, padded_rock_properties = _apply_boundary_conditions(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            boundary_conditions=boundary_conditions,
            cell_dimension=cell_dimension,
            grid_shape=grid_shape,
            thickness_grid=thickness_grid,
            time=0.0,
        )

        logger.debug("Checking well locations against grid dimensions")
        wells.check_location(grid_dimensions=grid_shape)
        wells = copy.deepcopy(wells)
        scheme = options.scheme
        if scheme == "impes":
            logger.debug(
                "Using implicit pressure, explicit saturation evolution scheme"
            )
            evolve_pressure = evolve_pressure_implicitly
        else:
            logger.debug("Using fully explicit evolution scheme")
            evolve_pressure = evolve_pressure_explicitly

        miscibility_model = options.miscibility_model
        if miscibility_model != "immiscible":
            logger.debug(
                f"Using miscible saturation evolution scheme: '{miscibility_model}'"
            )
            evolve_saturation = evolve_miscible_saturation_explicitly
        else:
            logger.debug("Using immiscible saturation evolution scheme")
            evolve_saturation = evolve_saturation_explicitly

        # Initialize fluid properties before starting the simulation
        # To ensure all dependent properties are consistent with initial pressure and saturation conditions
        logger.debug("Initializing PVT fluid properties for simulation start")
        padded_fluid_properties = update_pvt_properties(
            fluid_properties=padded_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
        )
        # Unpad the fluid properties back to original grid for model state snapshots
        model = evolve(
            model, fluid_properties=padded_fluid_properties.unpad(pad_width=1)
        )

        zeros_grid = build_uniform_grid(model.grid_shape, value=0.0)
        oil_injection_grid = zeros_grid
        water_injection_grid = zeros_grid.copy()
        gas_injection_grid = zeros_grid.copy()
        oil_production_grid = zeros_grid.copy()
        water_production_grid = zeros_grid.copy()
        gas_production_grid = zeros_grid.copy()
        relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
            build_rock_fluid_properties_grids(
                fluid_properties=padded_fluid_properties,
                rock_properties=padded_rock_properties,
                rock_fluid_properties=rock_fluid_properties,
                options=options,
            )
        )
        model_state = ModelState[ThreeDimensions](
            time_step=0,
            time_step_size=time_step_size,
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
        )

        num_of_time_steps = min(
            math.ceil(options.total_time // time_step_size), options.max_time_steps
        )
        logger.debug(f"Simulating for {num_of_time_steps} time steps")
        output_frequency = options.output_frequency

        logger.debug(f"Number of time steps to simulate: {int(num_of_time_steps)}")
        logger.debug(f"Output frequency: every {output_frequency} steps")
        logger.debug(f"Convergence tolerance: {options.convergence_tolerance}")

        # Yield the Initial model state
        logger.debug("Yielding initial model state")
        yield model_state

        no_flow_pressure_boundary_condition = isinstance(
            boundary_conditions["pressure"], type(default_bc)
        )

        for time_step in range(1, num_of_time_steps + 1):
            logger.debug(f"Running time step {time_step} of {num_of_time_steps}...")

            # Log simulation progress at regular intervals
            if time_step % max(1, int(num_of_time_steps // 10)) == 0:
                progress_percent = (time_step / num_of_time_steps) * 100
                logger.info(
                    f"Simulation progress: {progress_percent:.1f}% (step {time_step}/{int(num_of_time_steps)})"
                )

            # Apply boundary conditions before pressure update
            logger.debug("Applying boundary conditions before pressure evolution...")
            padded_fluid_properties, padded_rock_properties = (
                _apply_boundary_conditions(
                    fluid_properties=padded_fluid_properties,
                    rock_properties=padded_rock_properties,
                    boundary_conditions=boundary_conditions,
                    cell_dimension=cell_dimension,
                    grid_shape=grid_shape,
                    thickness_grid=thickness_grid,
                    time=time_step * time_step_size,
                )
            )
            # If the pressure boundary condition is not no-flow,
            # Then apply PVT update before pressure evolution
            # Since most PVT properties depend on pressure
            # This is skipped for no-flow BCs to save computation.
            # because mirroring neighbour values for PVT properties is sufficient for no-flow BCs.
            if not no_flow_pressure_boundary_condition:
                logger.debug(
                    "Updating PVT fluid properties before pressure evolution..."
                )
                padded_fluid_properties = update_pvt_properties(
                    fluid_properties=padded_fluid_properties,
                    wells=wells,
                    miscibility_model=miscibility_model,
                )

            # Build relative permeability, relative mobility, and capillary pressure grids
            relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
                build_rock_fluid_properties_grids(
                    fluid_properties=padded_fluid_properties,
                    rock_properties=padded_rock_properties,
                    rock_fluid_properties=rock_fluid_properties,
                    options=options,
                )
            )

            logger.debug("Evolving pressure...")
            # Evolvers also return padded grids as output since padded grids were given as input
            pressure_result = evolve_pressure(
                cell_dimension=cell_dimension,
                thickness_grid=padded_thickness_grid,
                elevation_grid=padded_elevation_grid,
                time_step=time_step,
                time_step_size=time_step_size,
                rock_properties=padded_rock_properties,
                fluid_properties=padded_fluid_properties,
                rock_fluid_properties=rock_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
            )
            padded_pressure_grid = pressure_result.value
            logger.debug(
                f"Pressure evolution completed! ({pressure_result.scheme} scheme)"
            )

            if (
                negative_pressure_indices := np.argwhere(padded_pressure_grid < 0)
            ).size > 0:
                logger.error(
                    f"Negative pressure detected at {negative_pressure_indices.size} grid cells"
                )
                logger.error(
                    f"Minimum pressure: {np.min(padded_pressure_grid):.2f} psi"
                )
                logger.error(
                    f"First few negative pressure indices: {negative_pressure_indices.tolist()[:10]}"
                )
                raise RuntimeError(
                    NEGATIVE_PRESSURE_ERROR.format(
                        indices=negative_pressure_indices.tolist()
                    )
                    + f"\nAt Time Step {time_step}."
                )

            # Update fluid properties with new pressure grid
            if pressure_result.scheme == "implicit":
                logger.debug("Updating fluid properties with new pressure grid...")
                padded_fluid_properties = evolve(
                    padded_fluid_properties, pressure_grid=padded_pressure_grid
                )
                # For implicit schemes, we need to update the fluid properties
                # before proceeding to saturation evolution.
                # Explicit pressure is strongly coupled with saturation update.
                logger.debug(
                    "Updating PVT fluid properties for saturation evolution (implicit scheme)"
                )
                padded_fluid_properties = update_pvt_properties(
                    fluid_properties=padded_fluid_properties,
                    wells=wells,
                    miscibility_model=miscibility_model,
                )

                # Recompute relative mobility grids with updated fluid properties
                # Since relative mobility depends on fluid viscosities which change with pressure
                logger.debug(
                    "Rebuilding relative mobility grids for saturation evolution (implicit scheme)..."
                )
                (
                    water_relative_mobility_grid,
                    oil_relative_mobility_grid,
                    gas_relative_mobility_grid,
                ) = build_three_phase_relative_mobilities_grids(
                    oil_relative_permeability_grid=relperm_grids.kro,
                    water_relative_permeability_grid=relperm_grids.krw,
                    gas_relative_permeability_grid=relperm_grids.krg,
                    water_viscosity_grid=padded_fluid_properties.water_viscosity_grid,
                    oil_viscosity_grid=padded_fluid_properties.oil_effective_viscosity_grid,
                    gas_viscosity_grid=padded_fluid_properties.gas_viscosity_grid,
                )
                # Clamp relative mobility grids to avoid numerical issues
                water_relative_mobility_grid = options.relative_mobility_range[
                    "water"
                ].arrayclip(water_relative_mobility_grid)
                oil_relative_mobility_grid = options.relative_mobility_range[
                    "oil"
                ].arrayclip(oil_relative_mobility_grid)
                gas_relative_mobility_grid = options.relative_mobility_range[
                    "gas"
                ].arrayclip(gas_relative_mobility_grid)
                relative_mobility_grids = RelativeMobilityGrids(
                    water_relative_mobility=water_relative_mobility_grid,
                    oil_relative_mobility=oil_relative_mobility_grid,
                    gas_relative_mobility=gas_relative_mobility_grid,
                )
                relative_mobility_grids = typing.cast(
                    RelativeMobilityGrids[ThreeDimensions], relative_mobility_grids
                )

            else:
                # For explicit schemes, we can re-use the current fluid properties
                # in the saturation evolution step.
                # Explicit pressure is fully decoupled from saturation update.
                logger.debug(
                    "Using current PVT fluid properties for saturation evolution (explicit scheme)"
                )

            # Saturation evolution
            logger.debug("Evolving saturation...")
            # Build zeros grids to track production and injection at each time step
            oil_injection_grid = zeros_grid.copy()  # No initial injection
            water_injection_grid = zeros_grid.copy()
            gas_injection_grid = zeros_grid.copy()
            oil_production_grid = zeros_grid.copy()  # No initial production
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
                rock_fluid_properties=rock_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
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
            logger.debug("Saturation evolution completed!")

            # Update fluid properties with new pressure after saturation update, for explicit-explicit scheme
            if pressure_result.scheme == "explicit":
                logger.debug(
                    "Updating fluid properties with new pressure grid (explicit scheme)..."
                )
                padded_fluid_properties = evolve(
                    padded_fluid_properties, pressure_grid=padded_pressure_grid
                )

            logger.debug("Updating fluid properties with new saturation grids...")
            (
                padded_water_saturation_grid,
                padded_oil_saturation_grid,
                *other_padded_saturation_grids,
            ) = saturation_result.value
            other_grids_size = len(other_padded_saturation_grids)
            if other_grids_size == 1:
                padded_gas_saturation_grid = other_padded_saturation_grids[0]
                padded_fluid_properties = evolve(
                    padded_fluid_properties,
                    water_saturation_grid=padded_water_saturation_grid,
                    oil_saturation_grid=padded_oil_saturation_grid,
                    gas_saturation_grid=padded_gas_saturation_grid,
                )
            elif other_grids_size == 2:
                padded_gas_saturation_grid, padded_solvent_concentration_grid = (
                    other_padded_saturation_grids
                )
                padded_fluid_properties = evolve(
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

            # Take a snapshot of the model state at specified intervals and at the last time step
            if (time_step % output_frequency == 0) or (time_step == num_of_time_steps):
                logger.debug(f"Capturing model state at time step {time_step}")
                # The production rates are negative in the evolution
                # so we need to negate them to report positive production values
                logger.debug("Preparing injection and production rate grids for output")
                oil_production_grid = typing.cast(
                    NDimensionalGrid[ThreeDimensions], np.negative(oil_production_grid)
                )
                water_production_grid = typing.cast(
                    NDimensionalGrid[ThreeDimensions],
                    np.negative(water_production_grid),
                )
                gas_production_grid = typing.cast(
                    NDimensionalGrid[ThreeDimensions], np.negative(gas_production_grid)
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
                model_snapshot = evolve(
                    model, fluid_properties=padded_fluid_properties.unpad(pad_width=1)
                )
                model_state = ModelState(
                    time_step=time_step,
                    time_step_size=time_step_size,
                    model=model_snapshot,
                    wells=wells_snapshot,
                    relative_mobilities=relative_mobility_grids,
                    relative_permeabilities=relperm_grids,
                    capillary_pressures=capillary_pressure_grids,
                    injection=injection_rates,
                    production=production_rates,
                )
                yield model_state

            # PVT update for next time step (or only update for explicit-explicit scheme)
            if pressure_result.scheme == "explicit":
                logger.debug("Updating PVT fluid properties for next time step")
                padded_fluid_properties = update_pvt_properties(
                    fluid_properties=padded_fluid_properties,
                    wells=wells,
                    miscibility_model=miscibility_model,
                )

            # Update the wells for the next time step
            logger.debug("Updating wells for next time step")
            wells.evolve(model_state)

    # Log completion outside the loop to avoid unbound variable issues
    logger.info(
        f"Simulation completed successfully after {num_of_time_steps} time steps"
    )


def update_pvt_properties(
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
) -> FluidProperties[ThreeDimensions]:
    """
    Updates PVT fluid properties across the simulation grid using the current pressure and temperature values.
    This function recalculates the fluid PVT properties in a physically consistent sequence:

    ```
    ┌────────────┐
    │  PRESSURE  │
    └────┬───────┘
         ▼
    ┌─────────────┐
    │  TEMPERATURE│
    └────┬───────┘
         ▼
    ┌──────────────┐
    │ GAS PROPERTIES│
    └────┬─────────┘
         ▼
    ┌──────────────────┐
    │ WATER PROPERTIES │
    └────┬─────────────┘
         ▼
    ┌────────────────────────────────────────────────────────────┐
    │ OIL PROPERTIES                                             │
    │  • Compute oil specific gravity and API gravity            │
    │  • Recalculate bubble point pressure (Pb)                  │
    │  • If pressure < Pb: recompute GOR (Rs) using Vazquez-Beggs│
    │  • Compute FVF using pressure, Rs, Pb                      │
    │  • Then compute oil compressibility, density, viscosity    │
    └────────────────────────────────────────────────────────────┘

    === GAS PROPERTIES ===
    - Computes gas gravity from density.
    - Uses gas gravity to derive molecular weight.
    - Computes gas z-factor (compressibility factor) from pressure, temperature, and gas gravity.
    - Updates:
        - Formation volume factor (Bg)
        - Compressibility (Cg)
        - Density (ρg)
        - Viscosity (μg)

    === WATER PROPERTIES ===
    - Computes gas solubility in water (Rs_w) based on salinity, pressure, and temperature.
    - Determines water bubble point pressure from solubility and salinity.
    - Computes gas-free water FVF (for use in density calculation).
    - Updates:
        - Water compressibility (Cw) considering dissolved gas
        - Water FVF (Bw)
        - Water density (ρw)
        - Water viscosity (μw)

    === OIL PROPERTIES ===
    - Computes oil specific gravity and API gravity from base density.
    - Recalculates bubble point pressure (Pb) using API, temperature, and gas gravity.
    - Determines GOR:
        • If current pressure < Pb: compute GOR using Vazquez-Beggs correlation.
        • If current pressure ≥ Pb: GOR = GOR at Pb (Rs = Rs_b).
    - Computes:
        - Oil formation volume factor (Bo) using pressure, Pb, GOR, and gravity.
        - Oil compressibility (Co) using updated GOR and Pb.
        - Oil density (ρo) using GOR and Bo.
        - Oil viscosity (μo) using Rs, Pb, and API.
    ```

    :param fluid_properties: Current fluid property grids (pressure, temperature, salinity, densities, etc.)
    :return: Updated FluidProperties object with recalculated gas, water, and oil properties.
    """
    # GAS PROPERTIES
    gas_compressibility_factor_grid = build_gas_compressibility_factor_grid(
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
    )
    new_gas_formation_volume_factor_grid = build_gas_formation_volume_factor_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )
    new_gas_compressibility_grid = build_gas_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )
    new_gas_density_grid = build_gas_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )
    new_gas_viscosity_grid = build_gas_viscosity_grid(
        temperature_grid=fluid_properties.temperature_grid,
        gas_density_grid=new_gas_density_grid,
        gas_molecular_weight_grid=fluid_properties.gas_molecular_weight_grid,
    )

    # WATER PROPERTIES
    gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        gas=fluid_properties.reservoir_gas,
    )
    new_water_bubble_point_pressure_grid = build_water_bubble_point_pressure_grid(
        temperature_grid=fluid_properties.temperature_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
    )
    gas_free_water_formation_volume_factor_grid = (
        build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=fluid_properties.pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
        )
    )
    new_water_compressibility_grid = build_water_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
        gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )
    new_water_density_grid = build_water_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )
    new_water_formation_volume_factor_grid = build_water_formation_volume_factor_grid(
        water_density_grid=new_water_density_grid,  # Use new density here
        salinity_grid=fluid_properties.water_salinity_grid,
    )
    new_water_viscosity_grid = build_water_viscosity_grid(
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        pressure_grid=fluid_properties.pressure_grid,
    )

    # OIL PROPERTIES (tricky due to bubble point)
    # Make sure to always compute the oil bubble point pressure grid
    # before the gas to oil ratio grid, as the latter depends on the former.

    # Step 1: Compute NEW bubble point using CURRENT Rs
    new_oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        temperature_grid=fluid_properties.temperature_grid,
        solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
    )

    # Step 2: Compute Rs at NEW bubble point
    gor_at_bubble_point_pressure_grid = build_solution_gas_to_oil_ratio_grid(
        pressure_grid=new_oil_bubble_point_pressure_grid,  # New bubble point here
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,  # Use same NEW bubble point here
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
    )
    # Step 3: Compute NEW Rs at current pressure
    new_solution_gas_to_oil_ratio_grid = build_solution_gas_to_oil_ratio_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,  # New bubble point here
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,  # GOR at new bubble point here
    )
    # Step 4: Compute oil FVF (may use lagged compressibility - acceptable)
    # Oil FVF does not depend necessarily on the new compressibility grid,
    # so we can use the old one (compressibility changes are small, hence, it can be lagged).
    # FVF is a function of pressure and phase behavior. Only when pressure changes,
    # does FVF need to be recalculated.
    new_oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        oil_compressibility_grid=fluid_properties.oil_compressibility_grid,
    )
    # Step 5: Compute oil compressibility
    new_oil_compressibility_grid = build_oil_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
        gas_formation_volume_factor_grid=new_gas_formation_volume_factor_grid,
        oil_formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
    )
    # Step 6: Compute oil density and viscosity
    new_oil_density_grid = build_live_oil_density_grid(
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
    )
    new_oil_viscosity_grid = build_oil_viscosity_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
    )
    new_oil_effective_viscosity_grid = new_oil_viscosity_grid
    # If there are miscible injections, update the effective oil viscosity grid
    if miscibility_model != "immiscible":
        for well in wells.injection_wells:
            injected_fluid = well.injected_fluid
            if injected_fluid and injected_fluid.is_miscible:
                injected_fluid_viscosity_grid = np.vectorize(
                    injected_fluid.get_viscosity, otypes=[np.float64]
                )(
                    pressure=fluid_properties.pressure_grid,
                    temperature=fluid_properties.temperature_grid,
                )
                # Update effective oil viscosity grid using Todd-Longstaff model
                new_oil_effective_viscosity_grid = build_oil_effective_viscosity_grid(
                    oil_viscosity_grid=new_oil_effective_viscosity_grid,
                    solvent_viscosity_grid=injected_fluid_viscosity_grid,
                    solvent_concentration_grid=fluid_properties.solvent_concentration_grid,
                    omega=injected_fluid.todd_longstaff_omega,
                    pressure_grid=fluid_properties.pressure_grid,
                    minimum_miscibility_pressure=injected_fluid.minimum_miscibility_pressure,
                )

    # Finally, update the fluid properties with all the new grids
    updated_fluid_properties = evolve(
        fluid_properties,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        oil_bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        water_bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
        oil_viscosity_grid=new_oil_viscosity_grid,
        oil_effective_viscosity_grid=new_oil_effective_viscosity_grid,
        water_viscosity_grid=new_water_viscosity_grid,
        gas_viscosity_grid=new_gas_viscosity_grid,
        oil_formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
        water_formation_volume_factor_grid=new_water_formation_volume_factor_grid,
        gas_formation_volume_factor_grid=new_gas_formation_volume_factor_grid,
        oil_compressibility_grid=new_oil_compressibility_grid,
        water_compressibility_grid=new_water_compressibility_grid,
        gas_compressibility_grid=new_gas_compressibility_grid,
        oil_density_grid=new_oil_density_grid,
        water_density_grid=new_water_density_grid,
        gas_density_grid=new_gas_density_grid,
    )
    return updated_fluid_properties
