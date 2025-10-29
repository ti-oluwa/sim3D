"""Dynamic simulation on a 3-Dimensional reservoir model with specified properties and wells."""

import copy
import logging
import math
import typing

from attrs import evolve
import numpy as np

from sim3D.diffusivity import (
    evolve_pressure_adaptively,
    evolve_pressure_explicitly,
    evolve_pressure_implicitly,
    evolve_saturation_explicitly,
)
from sim3D.grids.base import build_uniform_grid
from sim3D.grids.properties import (
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_solubility_in_water_grid,
    build_solution_gas_to_oil_ratio_grid,
    build_gas_viscosity_grid,
    build_live_oil_density_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_compressibility_grid,
    build_oil_formation_volume_factor_grid,
    build_oil_viscosity_grid,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from sim3D.states import ModelState
from sim3D.statics import FluidProperties, ReservoirModel
from sim3D.types import (
    NDimensionalGrid,
    Options,
    RateGrids,
    ThreeDimensions,
    _RateGridsProxy,
)
from sim3D.wells import Wells


__all__ = ["run_simulation"]

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


def run_simulation(
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
    logger.debug(f"Evolution scheme: {options.evolution_scheme}")
    logger.debug(f"Time step size: {options.time_step_size} seconds")
    logger.debug(f"Total simulation time: {options.total_time} seconds")

    # print("Before copying model properties...")
    # print(
    #     model.fluid_properties.oil_saturation_grid + model.fluid_properties.water_saturation_grid + model.fluid_properties.gas_saturation_grid
    # )
    cell_dimension = model.cell_dimension
    fluid_properties = model.fluid_properties
    rock_properties = model.rock_properties
    rock_fluid_properties = model.rock_fluid_properties
    thickness_grid = model.thickness_grid
    elevation_grid = model.get_elevation_grid(apply_dip=options.apply_dip)

    # print("After copying model properties...")
    # print(
    #     fluid_properties.oil_saturation_grid + fluid_properties.water_saturation_grid + fluid_properties.gas_saturation_grid
    # )

    logger.debug("Checking well locations against grid dimensions")
    wells.check_location(model.grid_shape)
    wells = copy.deepcopy(wells)

    if (method := options.evolution_scheme.lower()) == "adaptive_explicit":
        logger.debug("Using adaptive pressure, explicit saturation evolution scheme")
        evolve_pressure = evolve_pressure_adaptively
        evolve_saturation = evolve_saturation_explicitly
        logger.debug(
            f"Diffusion number threshold: {options.diffusion_number_threshold}"
        )
    elif method == "implicit_explicit":
        logger.debug("Using implicit pressure, explicit saturation evolution scheme")
        evolve_pressure = evolve_pressure_implicitly
        evolve_saturation = evolve_saturation_explicitly
    else:
        logger.debug("Using fully explicit evolution scheme")
        evolve_pressure = evolve_pressure_explicitly
        evolve_saturation = evolve_saturation_explicitly

    time_step_size = options.time_step_size
    # Initialize fluid properties before starting the simulation
    # To ensure all dependent properties are consistent with initial pressure/saturation
    logger.debug("Initializing PVT fluid properties for simulation start")
    fluid_properties = update_pvt_properties(fluid_properties)
    model = evolve(model, fluid_properties=fluid_properties)

    zeros_grid = build_uniform_grid(model.grid_shape, value=0.0)
    oil_injection_grid = zeros_grid
    water_injection_grid = zeros_grid.copy()
    gas_injection_grid = zeros_grid.copy()
    oil_production_grid = zeros_grid.copy()
    water_production_grid = zeros_grid.copy()
    gas_production_grid = zeros_grid.copy()
    model_state = ModelState(
        time_step=0,
        time_step_size=time_step_size,
        model=model,
        wells=wells,
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

    boundary_conditions = model.boundary_conditions
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

    for time_step in range(1, num_of_time_steps + 1):
        logger.debug(f"Running time step {time_step} of {num_of_time_steps}...")

        # Log simulation progress at regular intervals
        if time_step % max(1, int(num_of_time_steps // 10)) == 0:
            progress_percent = (time_step / num_of_time_steps) * 100
            logger.info(
                f"Simulation progress: {progress_percent:.1f}% (step {time_step}/{int(num_of_time_steps)})"
            )

        logger.debug("Evolving pressure...")
        pressure_evolution_result = evolve_pressure(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
            wells=wells,
            options=options,
        )
        pressure_grid = pressure_evolution_result.value
        logger.debug(
            f"Pressure evolution completed! ({pressure_evolution_result.scheme} scheme)"
        )

        if (negative_pressure_indices := np.argwhere(pressure_grid < 0)).size > 0:
            logger.error(
                f"Negative pressure detected at {negative_pressure_indices.size} grid cells"
            )
            logger.error(f"Minimum pressure: {np.min(pressure_grid):.2f} psi")
            logger.error(
                f"First few negative pressure indices: {negative_pressure_indices[:10].tolist()}"
            )
            raise RuntimeError(
                NEGATIVE_PRESSURE_ERROR.format(
                    indices=negative_pressure_indices.tolist()
                )
                + f"\nAt Time Step {time_step}."
            )
        # Update fluid properties with new pressure grid
        if pressure_evolution_result.scheme == "implicit":
            logger.debug("Updating fluid properties with new pressure grid...")
            fluid_properties = evolve(fluid_properties, pressure_grid=pressure_grid)
            # For implicit schemes, we need to update the fluid properties
            # before proceeding to saturation evolution.
            # Explicit pressure is strongly coupled with saturation update.
            logger.debug(
                "Updating PVT fluid properties for saturation evolution (implicit scheme)"
            )
            fluid_properties = update_pvt_properties(fluid_properties)
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
        saturation_evolution_result = evolve_saturation(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
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
        if (
            pressure_evolution_result.scheme == "explicit"
            and saturation_evolution_result.scheme == "explicit"
        ):
            logger.debug(
                "Updating fluid properties with new pressure grid (explicit scheme)..."
            )
            fluid_properties = evolve(fluid_properties, pressure_grid=pressure_grid)

        logger.debug("Updating fluid properties with new saturation grids...")
        water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
            saturation_evolution_result.value
        )
        fluid_properties = evolve(
            fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
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
                NDimensionalGrid[ThreeDimensions], np.negative(water_production_grid)
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
            model_snapshot = evolve(model, fluid_properties=fluid_properties)
            model_state = ModelState(
                time_step=time_step,
                time_step_size=time_step_size,
                model=model_snapshot,
                wells=wells_snapshot,
                injection=injection_rates,
                production=production_rates,
            )
            yield model_state

        # PVT update for next time step (or only update for explicit-explicit scheme)
        if (
            pressure_evolution_result.scheme == "explicit"
            and saturation_evolution_result.scheme == "explicit"
        ):
            logger.debug("Updating PVT fluid properties for next time step")
            fluid_properties = update_pvt_properties(fluid_properties)

        # Update the wells for the next time step
        logger.debug("Updating wells for next time step")
        wells.evolve(model_state)

    # Log completion outside the loop to avoid unbound variable issues
    logger.info(
        f"Simulation completed successfully after {num_of_time_steps} time steps"
    )


def update_pvt_properties(
    fluid_properties: FluidProperties[ThreeDimensions],
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
        gas=fluid_properties.reservoir_gas_name,
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

    # Finally, update the fluid properties with all the new grids
    updated_fluid_properties = evolve(
        fluid_properties,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        oil_bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        water_bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
        oil_viscosity_grid=new_oil_viscosity_grid,
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
