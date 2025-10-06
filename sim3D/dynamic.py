"""Dynamic simulation on a 3-Dimensional reservoir model with specified properties and wells."""

import copy
import logging
import typing

from attrs import define, evolve
import numpy as np

from sim3D.flow import (
    evolve_pressure_adaptively,
    evolve_pressure_explicitly,
    evolve_pressure_implicitly,
    evolve_saturation_explicitly,
    evolve_saturation_implicitly,
)
from sim3D.grids import (
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_solubility_in_water_grid,
    build_gas_to_oil_ratio_grid,
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
from sim3D.static import FluidProperties, ReservoirModel
from sim3D.types import NDimension, Options, ThreeDimensions
from sim3D.wells import Wells


__all__ = [
    "ModelState",
    "run_simulation",
]

logger = logging.getLogger(__name__)


@define(frozen=True, slots=True)
class ModelState(typing.Generic[NDimension]):
    """
    Represents the state of the reservoir model at a specific time step during a simulation.
    """

    time_step: int
    """The time step index of the model state."""
    time_step_size: float
    """The time step size in seconds."""
    model: ReservoirModel[NDimension]
    """The reservoir model at this state."""
    wells: Wells[NDimension]
    """The wells configuration at this state."""

    @property
    def time(self) -> float:
        """
        Returns the total simulation time at this state.
        """
        return self.time_step * self.time_step_size


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
    logger.info("Starting 3D reservoir simulation")
    logger.debug(f"Grid dimensions: {model.grid_dimension}")
    logger.debug(f"Cell dimensions: {model.cell_dimension}")
    logger.debug(f"Evolution scheme: {options.evolution_scheme}")
    logger.debug(f"Time step size: {options.time_step_size} seconds")
    logger.debug(f"Total simulation time: {options.total_time} seconds")

    cell_dimension = model.cell_dimension
    fluid_properties = copy.deepcopy(model.fluid_properties)
    rock_properties = copy.deepcopy(model.rock_properties)
    rock_fluid_properties = copy.deepcopy(model.rock_fluid_properties)
    thickness_grid = model.thickness_grid
    elevation_grid = model.get_elevation_grid(direction="downward")

    logger.debug("Checking well locations against grid dimensions")
    wells.check_location(model.grid_dimension)
    wells = copy.deepcopy(wells)

    if (method := options.evolution_scheme.lower()) == "adaptive_explicit":
        logger.debug("Using adaptive pressure, explicit saturation evolution scheme")
        evolve_pressure = evolve_pressure_adaptively
        evolve_saturation = evolve_saturation_explicitly
        logger.debug(
            f"Diffusion number threshold: {options.diffusion_number_threshold}"
        )
    elif method == "explicit_implicit":
        logger.debug("Using explicit pressure, implicit saturation evolution scheme")
        evolve_pressure = evolve_pressure_explicitly
        evolve_saturation = evolve_saturation_implicitly
    elif method == "implicit_explicit":
        logger.debug("Using implicit pressure, explicit saturation evolution scheme")
        evolve_pressure = evolve_pressure_implicitly
        evolve_saturation = evolve_saturation_explicitly
    elif method == "fully_implicit":
        logger.debug("Using fully implicit evolution scheme")
        evolve_pressure = evolve_pressure_implicitly
        evolve_saturation = evolve_saturation_implicitly
    else:
        logger.debug("Using fully explicit evolution scheme")
        evolve_pressure = evolve_pressure_explicitly
        evolve_saturation = evolve_saturation_explicitly

    time_step_size = options.time_step_size
    initial_state = ModelState(
        time_step=0,
        time_step_size=time_step_size,
        model=model,
        wells=wells,
    )

    boundary_conditions = model.boundary_conditions
    num_of_time_steps = min(
        (options.total_time // time_step_size), options.max_iterations
    )
    output_frequency = options.output_frequency

    logger.debug(f"Number of time steps to simulate: {int(num_of_time_steps)}")
    logger.debug(f"Output frequency: every {output_frequency} steps")
    logger.debug(f"Convergence tolerance: {options.convergence_tolerance}")

    # Yield the Initial model state
    logger.debug("Yielding initial model state")
    yield initial_state

    for time_step in range(1, int(num_of_time_steps + 1)):
        logger.debug(f"Running time step {time_step} of {num_of_time_steps}...")

        # Log simulation progress at regular intervals
        if time_step % max(1, int(num_of_time_steps // 10)) == 0:
            progress_percent = (time_step / num_of_time_steps) * 100
            logger.info(
                f"Simulation progress: {progress_percent:.1f}% (step {time_step}/{int(num_of_time_steps)})"
            )

        logger.debug("Computing pressure evolution...")
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
        # import rich
        # rich.print(pressure_grid)
        logger.debug(
            f"Pressure evolution completed using {pressure_evolution_result.scheme} scheme"
        )

        if (negative_pressure_indices := np.argwhere(pressure_grid < 0)).size > 0:
            logger.error(
                f"Negative pressure detected at {negative_pressure_indices.size} grid cells"
            )
            logger.error(f"Minimum pressure: {np.min(pressure_grid):.2f} psi")
            logger.error(
                f"First few negative pressure indices: {negative_pressure_indices[:5].tolist()}"
            )
            raise RuntimeError(
                NEGATIVE_PRESSURE_ERROR.format(
                    indices=negative_pressure_indices.tolist()
                )
                + f"\nAt Time Step {time_step}."
            )

        logger.debug("Updating fluid properties with new pressure grid...")
        updated_fluid_properties = evolve(
            fluid_properties,
            pressure_grid=pressure_grid,
        )
        if pressure_evolution_result.scheme == "implicit":
            # For implicit schemes, we need to update the fluid properties
            # with the new pressure grid before computing saturation evolution.
            logger.debug(
                "Using updated fluid properties for saturation evolution (implicit scheme)"
            )
            saturation_fluid_properties = copy.deepcopy(updated_fluid_properties)
        else:
            # For explicit schemes, we can use the current fluid properties
            # as the saturation fluid properties, since they are not updated
            # until after the saturation evolution.
            logger.debug(
                "Using current fluid properties for saturation evolution (explicit scheme)"
            )
            saturation_fluid_properties = copy.deepcopy(fluid_properties)

        # Saturation evolution
        logger.debug("Computing saturation evolution...")
        saturation_evolution_result = evolve_saturation(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=saturation_fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
            wells=wells,
            options=options,
        )
        water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
            saturation_evolution_result.value
        )
        logger.debug("Saturation evolution completed")

        logger.debug("Updating fluid properties with new saturation grids...")
        fluid_properties = evolve(
            updated_fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
        )
        # Update the static fluid properties
        logger.debug("Updating static fluid properties (PVT properties)...")
        fluid_properties = update_static_fluid_properties(fluid_properties)
        logger.debug("Static fluid properties update completed")

        # Capture the model state at specified intervals and at the last time step
        if (time_step % output_frequency == 0) or (time_step == num_of_time_steps):
            logger.debug(f"Capturing model state at time step {time_step}")
            model_state = ModelState(
                time_step=time_step,
                time_step_size=time_step_size,
                # Record the model state with updated fluid properties
                model=evolve(model, fluid_properties=copy.deepcopy(fluid_properties)),
                # Capture the current state of the wells
                wells=copy.deepcopy(wells),
            )
            yield model_state

        if time_step > options.max_iterations:
            logger.warning(
                f"Reached maximum number of iterations: {options.max_iterations}. Stopping simulation."
            )
            break
        # Update the wells for the next time step
        logger.debug("Updating wells for next time step")
        wells.evolve(time_step=time_step)

    # Log completion outside the loop to avoid unbound variable issues
    final_time_step = min(int(num_of_time_steps), options.max_iterations)
    logger.info(f"Simulation completed successfully after {final_time_step} time steps")


def update_static_fluid_properties(
    fluid_properties: FluidProperties[ThreeDimensions],
) -> FluidProperties[ThreeDimensions]:
    """
    Updates static fluid properties across the simulation grid using the current pressure and temperature values.
    This function recalculates the fluid PVT properties in a physically consistent sequence:

    ```
    ┌────────────┐
    │  PRESSURE  │
    └────┬───────┘
         ▼
    ┌────────────┐
    │  TEMPERATURE│
    └────┬───────┘
         ▼
    ┌──────────────┐
    │ GAS PROPERTIES│
    └────┬─────────┘
         ▼
    ┌─────────────────────┐
    │ WATER PROPERTIES     │
    └────┬────────────────┘
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
    # === Derived Grids ===
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
    gas_free_water_formation_volume_factor_grid = (
        build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=fluid_properties.pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
        )
    )
    new_water_bubble_point_pressure_grid = build_water_bubble_point_pressure_grid(
        temperature_grid=fluid_properties.temperature_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
    )
    new_water_compressibility_grid = build_water_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
        gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )
    new_water_formation_volume_factor_grid = build_water_formation_volume_factor_grid(
        water_density_grid=fluid_properties.water_density_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
    )
    new_water_density_grid = build_water_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )
    new_water_viscosity_grid = build_water_viscosity_grid(
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        pressure_grid=fluid_properties.pressure_grid,
    )

    # Make sure to always compute the oil bubble point pressure grid
    # before the gas to oil ratio grid, as the latter depends on the former.
    new_oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_to_oil_ratio_grid=fluid_properties.gas_to_oil_ratio_grid,
    )
    gor_at_bubble_point_pressure_grid = build_gas_to_oil_ratio_grid(
        pressure_grid=fluid_properties.oil_bubble_point_pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
    )
    new_gas_to_oil_ratio_grid = build_gas_to_oil_ratio_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=fluid_properties.oil_bubble_point_pressure_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
    )
    # Oil FVF does not depend necessarily on the new compressibility grid,
    # so we can use the old one.
    # FVF is a function of pressure and phase behavior. Only when pressure changes,
    # does FVF need to be recalculated.
    new_oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
        oil_compressibility_grid=fluid_properties.oil_compressibility_grid,
    )
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
    new_oil_density_grid = build_live_oil_density_grid(
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
        formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
    )
    new_oil_viscosity_grid = build_oil_viscosity_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
    )
    # Finally, evolve the fluid properties with all the new grids
    updated_fluid_properties = evolve(
        fluid_properties,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
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
