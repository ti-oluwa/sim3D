"""Run a simulation on a 3-Dimensional reservoir model with specified fluid properties and wells."""

import typing
import copy
from functools import partial
from attrs import define, evolve
import rich
import logging
import numpy as np

from sim3D.models import FluidProperties, ReservoirModel
from sim3D.grids import (
    build_gas_compressibility_factor_grid,
    build_gas_solubility_in_water_grid,
    build_gas_to_oil_ratio_grid,
    build_oil_bubble_point_pressure_grid,
    build_water_bubble_point_pressure_grid,
    build_oil_viscosity_grid,
    build_water_viscosity_grid,
    build_gas_viscosity_grid,
    build_oil_formation_volume_factor_grid,
    build_water_formation_volume_factor_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_compressibility_grid,
    build_oil_compressibility_grid,
    build_water_compressibility_grid,
    build_live_oil_density_grid,
    build_water_density_grid,
    build_gas_density_grid,
)
from sim3D.flow_evolution import (
    compute_adaptive_pressure_evolution,
    compute_explicit_pressure_evolution,
    compute_implicit_pressure_evolution,
    compute_saturation_evolution,
)
from sim3D.wells import Wells
from sim3D.types import EvolutionScheme, FluidMiscibility, NDimension, ThreeDimensions


__all__ = [
    "SimulationParameters",
    "ModelState",
    "run_3D_simulation",
]

logger = logging.getLogger(__name__)


@define(slots=True, frozen=True)
class SimulationParameters:
    """
    Represents the simulation parameters for the reservoir model.
    """

    time_step_size: float = 3600.0
    """Time step for the simulation in seconds (default is 1 hour)."""
    total_time: float = 86400.0
    """Total simulation time in seconds (default is 1 day)."""
    max_iterations: int = 1000
    """Maximum number of iterations for the simulation."""
    convergence_tolerance: float = 1e-6
    """Convergence tolerance for the simulation."""
    output_frequency: int = 10
    """Frequency of output results during the simulation."""
    evolution_scheme: typing.Union[EvolutionScheme, typing.Literal["adaptive"]] = (
        "adaptive"
    )
    """Discretization method for the simulation (e.g., 'adaptive', 'explicit', 'implicit')."""
    fluid_miscibility: typing.Optional[FluidMiscibility] = None
    """Fluid miscibility model to use (e.g., 'harmonic', 'linear', ...). If None, no miscibility model is applied."""
    pressure_decay_constant: float = 1e-8
    """Pressure decay constant for the simulation (default is 1e-8)."""
    saturation_mixing_factor: float = 0.5
    """Saturation mixing factor for the simulation (default is 0.5)."""
    diffusion_number_threshold: float = 0.24
    """Threshold for the diffusion number to determine stability of the simulation (default is 0.24)."""


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
- Use smaller time steps if using explicit updates.
- Clamp pressure updates to a minimum floor (e.g., 1.45 psi) to prevent blow-up.
- Cross-check well source/sink terms for sign and magnitude correctness.

Simulation aborted to avoid propagation of unphysical results.
"""


def run_3D_simulation(
    model: ReservoirModel[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    params: SimulationParameters,
    output: typing.Optional[typing.Callable[[ModelState], typing.Any]] = None,
) -> typing.List[ModelState[ThreeDimensions]]:
    cell_dimension = model.cell_dimension
    fluid_properties = copy.deepcopy(model.fluid_properties)
    rock_properties = copy.deepcopy(model.rock_properties)
    height_grid = model.height_grid
    wells.check_location(model.grid_dimension)
    wells = copy.deepcopy(wells)

    if (method := params.evolution_scheme.lower()) == "adaptive":
        compute_pressure_evolution = partial(
            compute_adaptive_pressure_evolution,
            diffusion_number_threshold=params.diffusion_number_threshold,
        )
    elif method == "implicit":
        compute_pressure_evolution = compute_implicit_pressure_evolution
    else:
        compute_pressure_evolution = compute_explicit_pressure_evolution

    time_step_size = params.time_step_size
    initial_state = ModelState(
        time_step=0,
        time_step_size=time_step_size,
        model=model,
        wells=wells,
    )
    model_states = [initial_state]

    boundary_conditions = model.boundary_conditions
    num_of_time_steps = min(
        (params.total_time // time_step_size), params.max_iterations
    )
    output_frequency = params.output_frequency

    for time_step in range(1, int(num_of_time_steps + 1)):
        print(f"TIME STEP {time_step}")
        # Pressure evolution
        # rich.print("[bold cyan]Computing pressure evolution...[/bold cyan]")

        # rich.print(
        #     "Old Pressure Grid",
        #     fluid_properties.pressure_grid.min(),
        #     fluid_properties.pressure_grid.max(),
        # )
        pressure_evolution_result = compute_pressure_evolution(
            cell_dimension=cell_dimension,
            height_grid=height_grid,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            wells=wells,
        )
        pressure_grid = pressure_evolution_result.value

        # rich.print("New Pressure Grid", pressure_grid.min(), pressure_grid.max())
        if (negative_pressure_indices := np.argwhere(pressure_grid < 0)).size > 0:
            raise RuntimeError(
                NEGATIVE_PRESSURE_ERROR.format(
                    indices=negative_pressure_indices.tolist()
                )
                + f"\nAt Time Step {time_step}."
            )

        updated_fluid_properties = evolve(
            fluid_properties,
            pressure_grid=pressure_grid,
        )
        if pressure_evolution_result.scheme == "implicit":
            # For implicit schemes, we need to update the fluid properties
            # with the new pressure grid before computing saturation evolution.
            saturation_fluid_properties = copy.deepcopy(updated_fluid_properties)
        else:
            # For explicit schemes, we can use the current fluid properties
            # as the saturation fluid properties, since they are not updated
            # until after the saturation evolution.
            saturation_fluid_properties = copy.deepcopy(fluid_properties)

        # rich.print("[bold cyan]Updating fluid properties...[/bold cyan]")
        # rich.print(
        #     "Old Water Saturation Grid",
        #     fluid_properties.water_saturation_grid.min(),
        #     fluid_properties.water_saturation_grid.max(),
        # )
        # rich.print(
        #     "Old Oil Saturation Grid",
        #     fluid_properties.oil_saturation_grid.min(),
        #     fluid_properties.oil_saturation_grid.max(),
        # )
        # rich.print(
        #     "Old Gas Saturation Grid",
        #     fluid_properties.gas_saturation_grid.min(),
        #     fluid_properties.gas_saturation_grid.max(),
        # )
        # Saturation evolution
        saturation_evolution_result = compute_saturation_evolution(
            cell_dimension=cell_dimension,
            height_grid=height_grid,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=saturation_fluid_properties,
            wells=wells,
        )
        water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
            saturation_evolution_result.value
        )
        # rich.print(
        #     "New Water Saturation Grid",
        #     water_saturation_grid.min(),
        #     water_saturation_grid.max(),
        # )
        # rich.print(
        #     "New Oil Saturation Grid",
        #     oil_saturation_grid.min(),
        #     oil_saturation_grid.max(),
        # )
        # rich.print(
        #     "New Gas Saturation Grid",
        #     gas_saturation_grid.min(),
        #     gas_saturation_grid.max(),
        # )

        # Update the fluid properties for the next iteration
        #####################################################
        # Update the fluid properties with the new saturation grids
        fluid_properties = evolve(
            updated_fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
        )
        # Update the static fluid properties
        fluid_properties = update_static_fluid_properties(fluid_properties)

        # Capture the model state at specified intervals and at the last time step
        if (time_step % output_frequency == 0) or (time_step == num_of_time_steps):
            model_state = ModelState(
                time_step=time_step,
                time_step_size=time_step_size,
                # Record the model state with updated fluid properties
                model=evolve(model, fluid_properties=copy.deepcopy(fluid_properties)),
                # Capture the current state of the wells
                wells=copy.deepcopy(wells),
            )
            model_states.append(model_state)
            if output:
                output(model_state)

        if time_step > params.max_iterations:
            logger.debug(
                f"Reached maximum number of iterations: {params.max_iterations}. Stopping simulation."
            )
            break

        # Update the wells for the next time step
        wells.evolve(time_step=time_step)
    return model_states


def update_static_fluid_properties(
    fluid_properties: FluidProperties,
) -> FluidProperties:
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
    # rich.print("[bold cyan]Updating static fluid properties...[/bold cyan]")

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

    # # Print updated gas properties
    # rich.print(
    #     "[bold]Gas Gravity:[/bold]",
    #     fluid_properties.gas_gravity_grid.min(),
    #     fluid_properties.gas_gravity_grid.max(),
    # )
    # rich.print(
    #     "Gas Compressibility Factor:",
    #     gas_compressibility_factor_grid.min(),
    #     gas_compressibility_factor_grid.max(),
    # )
    # rich.print(
    #     "New Gas Compressibility:",
    #     new_gas_compressibility_grid.min(),
    #     new_gas_compressibility_grid.max(),
    # )
    # rich.print(
    #     "New Gas Formation Volume Factor:",
    #     new_gas_formation_volume_factor_grid.min(),
    #     new_gas_formation_volume_factor_grid.max(),
    # )
    # rich.print(
    #     "New Gas Density:", new_gas_density_grid.min(), new_gas_density_grid.max()
    # )
    # rich.print(
    #     "New Gas Viscosity:", new_gas_viscosity_grid.min(), new_gas_viscosity_grid.max()
    # )

    # WATER PROPERTIES
    gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
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

    # # Print updated water properties
    # rich.print(
    #     "Gas Solubility in Water:",
    #     gas_solubility_in_water_grid.min(),
    #     gas_solubility_in_water_grid.max(),
    # )
    # rich.print(
    #     "New Water Bubble Point Pressure:",
    #     new_water_bubble_point_pressure_grid.min(),
    #     new_water_bubble_point_pressure_grid.max(),
    # )
    # rich.print(
    #     "New Water Compressibility:",
    #     new_water_compressibility_grid.min(),
    #     new_water_compressibility_grid.max(),
    # )
    # rich.print(
    #     "New Water Formation Volume Factor:",
    #     new_water_formation_volume_factor_grid.min(),
    #     new_water_formation_volume_factor_grid.max(),
    # )
    # rich.print(
    #     "New Water Density:", new_water_density_grid.min(), new_water_density_grid.max()
    # )
    # rich.print(
    #     "New Water Viscosity:",
    #     new_water_viscosity_grid.min(),
    #     new_water_viscosity_grid.max(),
    # )

    # # OIL PROPERTIES
    # print(
    #     "Old GOR Grid:",
    #     fluid_properties.gas_to_oil_ratio_grid.min(),
    #     fluid_properties.gas_to_oil_ratio_grid.max(),
    # )
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

    # # Print updated oil properties
    # rich.print(
    #     "Oil Specific Gravity:",
    #     fluid_properties.oil_specific_gravity_grid.min(),
    #     fluid_properties.oil_specific_gravity_grid.max(),
    # )
    # rich.print(
    #     "Oil API Gravity:",
    #     fluid_properties.oil_api_gravity_grid.min(),
    #     fluid_properties.oil_api_gravity_grid.max(),
    # )
    # rich.print(
    #     "New Oil Bubble Point Pressure:",
    #     new_oil_bubble_point_pressure_grid.min(),
    #     new_oil_bubble_point_pressure_grid.max(),
    # )
    # rich.print(
    #     "New Oil Formation Volume Factor:",
    #     new_oil_formation_volume_factor_grid.min(),
    #     new_oil_formation_volume_factor_grid.max(),
    # )
    # rich.print(
    #     "New Gas to Oil Ratio:",
    #     new_gas_to_oil_ratio_grid.min(),
    #     new_gas_to_oil_ratio_grid.max(),
    # )
    # rich.print(
    #     "New Oil Compressibility:",
    #     new_oil_compressibility_grid.min(),
    #     new_oil_compressibility_grid.max(),
    # )
    # rich.print(
    #     "New Oil Density:", new_oil_density_grid.min(), new_oil_density_grid.max()
    # )
    # rich.print(
    #     "New Oil Viscosity:", new_oil_viscosity_grid.min(), new_oil_viscosity_grid.max()
    # )
    updated_fluid_properties = evolve(
        fluid_properties,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
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
