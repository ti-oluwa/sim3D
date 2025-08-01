"""Run a simulation on a 2D reservoir model with specified fluid properties and wells."""

import typing
import copy
from functools import partial
from dataclasses import dataclass, replace

import numpy as np


from _sim2D.models import FluidProperties, TwoDimensionalReservoirModel
from _sim2D.grids import (
    build_2D_gas_compressibility_factor_grid,
    build_2D_gas_gravity_from_density_grid,
    build_2D_gas_molecular_weight_grid,
    build_2D_oil_api_gravity_grid,
    build_2D_oil_specific_gravity_grid,
    build_2D_gas_solubility_in_water_grid,
    build_2D_gas_to_oil_ratio_grid,
    build_2D_oil_bubble_point_pressure_grid,
    build_2D_water_bubble_point_pressure_grid,
    build_2D_oil_viscosity_grid,
    build_2D_water_viscosity_grid,
    build_2D_gas_viscosity_grid,
    build_2D_oil_formation_volume_factor_grid,
    build_2D_water_formation_volume_factor_grid,
    build_2D_gas_formation_volume_factor_grid,
    build_2D_gas_free_water_formation_volume_factor_grid,
    build_2D_gas_compressibility_grid,
    build_2D_oil_compressibility_grid,
    build_2D_water_compressibility_grid,
    build_2D_live_oil_density_grid,
    build_2D_water_density_grid,
    build_2D_gas_density_grid,
)
from _sim2D.flow_evolution import (
    compute_adaptive_pressure_evolution,
    compute_explicit_pressure_evolution,
    compute_implicit_pressure_evolution,
    compute_saturation_evolution,
)
from _sim2D.wells import Wells
from _sim2D.types import DiscretizationMethod, FluidMiscibility


__all__ = [
    "SimulationParameters",
    "ModelTimeState",
    "run_simulation",
]


@dataclass(slots=True, frozen=True)
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
    discretization_method: DiscretizationMethod = "adaptive"
    """Discretization method for the simulation (e.g., 'adaptive', 'explicit', 'implicit')."""
    fluid_miscibility: typing.Optional[FluidMiscibility] = None
    """Fluid miscibility model to use (e.g., 'harmonic', 'linear', ...). If None, no miscibility model is applied."""
    pressure_decay_constant: float = 1e-8
    """Pressure decay constant for the simulation (default is 1e-8)."""
    saturation_mixing_factor: float = 0.5
    """Saturation mixing factor for the simulation (default is 0.5)."""
    diffusion_number_threshold: float = 0.24
    """Threshold for the diffusion number to determine stability of the simulation (default is 0.24)."""


@dataclass(frozen=True, slots=True)
class ModelTimeState:
    """
    Represents the state of the reservoir model at a specific time step during a simulation.
    """

    time: float
    """The time in seconds taken to reach this state."""
    fluid_properties: FluidProperties
    """The fluid properties of the reservoir at this time step."""
    wells: Wells
    """The well parameters at this time step."""


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
- Clamp pressure updates to a minimum floor (e.g., 1e4 Pa) to prevent blow-up.
- Cross-check well source/sink terms for sign and magnitude correctness.

Simulation aborted to avoid propagation of unphysical results.
"""


def run_simulation(
    model: TwoDimensionalReservoirModel,
    wells: Wells,
    params: SimulationParameters,
) -> typing.List[ModelTimeState]:
    cell_dimension = model.cell_dimension
    fluid_properties = copy.deepcopy(model.fluid_properties)
    rock_properties = copy.deepcopy(model.rock_properties)
    height_grid = model.height_grid
    wells = copy.deepcopy(wells)

    if (method := params.discretization_method.lower()) == "adaptive":
        compute_pressure_evolution = partial(
            compute_adaptive_pressure_evolution,
            diffusion_number_threshold=params.diffusion_number_threshold,
        )
    elif method == "implicit":
        compute_pressure_evolution = compute_implicit_pressure_evolution
    else:
        compute_pressure_evolution = compute_explicit_pressure_evolution

    initial_state = ModelTimeState(
        time=0.0,
        fluid_properties=fluid_properties,
        wells=wells,
    )
    model_time_states = [initial_state]

    boundary_conditions = model.boundary_conditions
    time_step_size = params.time_step_size
    num_of_time_steps = min(
        (params.total_time // time_step_size), params.max_iterations
    )
    output_frequency = params.output_frequency

    current_fluid_properties = fluid_properties
    for time_step in range(1, int(num_of_time_steps + 1)):
        print(f"TIME STEP {time_step}")
        # Pressure evolution
        pressure_grid = compute_pressure_evolution(
            cell_dimension=cell_dimension,
            height_grid=height_grid,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=current_fluid_properties,
            wells=wells,
        )
        if (negative_pressure_indices := np.argwhere(pressure_grid < 0)).size > 0:
            raise RuntimeError(
                NEGATIVE_PRESSURE_ERROR.format(
                    indices=negative_pressure_indices.tolist()
                )
                + f"\nAt Time Step {time_step}."
            )

        updated_fluid_properties = replace(
            current_fluid_properties,
            pressure_grid=pressure_grid,
        )

        # Saturation evolution
        water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
            compute_saturation_evolution(
                cell_dimension=cell_dimension,
                height_grid=height_grid,
                time_step_size=time_step_size,
                boundary_conditions=boundary_conditions,
                rock_properties=rock_properties,
                fluid_properties=updated_fluid_properties,
                wells=wells,
            )
        )

        # Update the fluid properties for the next iteration
        # Update the fluid properties with the new saturation grids
        current_fluid_properties = replace(
            updated_fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
        )
        # Update the static fluid properties
        current_fluid_properties = update_static_fluid_properties(
            fluid_properties=current_fluid_properties,
        )

        # Capture the model state at specified intervals and at the last time step
        if (time_step % output_frequency == 0) or (time_step == num_of_time_steps):
            model_time_state = ModelTimeState(
                time=time_step * time_step_size,
                fluid_properties=current_fluid_properties,
                wells=wells,
            )
            model_time_states.append(model_time_state)

    return model_time_states


def update_static_fluid_properties(
    fluid_properties: FluidProperties,
) -> FluidProperties:
    """
    Update the static fluid properties based on the current pressure and temperature grids.
    This function recalculates various fluid properties such as oil specific gravity,
    gas solubility in water, gas formation volume factor, and compressibility factors.
    """
    print(
        "GOR",
        fluid_properties.gas_to_oil_ratio_grid.min(),
        fluid_properties.gas_to_oil_ratio_grid.max(),
    )
    print(
        "Pb",
        fluid_properties.oil_bubble_point_pressure_grid.min(),
        fluid_properties.oil_bubble_point_pressure_grid.max(),
    )
    print(
        "C_oil",
        fluid_properties.oil_compressibility_grid.min(),
        fluid_properties.oil_compressibility_grid.max(),
    )
    gas_gravity_grid = build_2D_gas_gravity_from_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        density_grid=fluid_properties.gas_density_grid,
    )
    oil_specific_gravity_grid = build_2D_oil_specific_gravity_grid(
        oil_density_grid=fluid_properties.oil_density_grid,
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        oil_compressibility_grid=fluid_properties.oil_compressibility_grid,
    )
    print(
        "Oil Density",
        fluid_properties.oil_density_grid.min(),
        fluid_properties.oil_density_grid.max(),
    )
    oil_api_gravity_grid = build_2D_oil_api_gravity_grid(
        oil_specific_gravity_grid=oil_specific_gravity_grid,
    )
    gas_solubility_in_water_grid = build_2D_gas_solubility_in_water_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
    )
    gas_free_water_formation_volume_factor_grid = (
        build_2D_gas_free_water_formation_volume_factor_grid(
            pressure_grid=fluid_properties.pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
        )
    )
    gas_molecular_weight_grid = build_2D_gas_molecular_weight_grid(
        gas_gravity_grid=gas_gravity_grid
    )

    # Updated Gas to Oil Ratio Grid
    gor_at_bubble_point_pressure_grid = build_2D_gas_to_oil_ratio_grid(
        pressure_grid=fluid_properties.oil_bubble_point_pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=fluid_properties.oil_bubble_point_pressure_grid,
        gas_gravity_grid=gas_gravity_grid,
        oil_api_gravity_grid=oil_api_gravity_grid,
    )
    new_gas_to_oil_ratio_grid = build_2D_gas_to_oil_ratio_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=fluid_properties.oil_bubble_point_pressure_grid,
        gas_gravity_grid=gas_gravity_grid,
        oil_api_gravity_grid=oil_api_gravity_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
    )

    # Updated Bubble Point Pressure Grids
    new_oil_bubble_point_pressure_grid = build_2D_oil_bubble_point_pressure_grid(
        gas_gravity_grid=gas_gravity_grid,
        oil_api_gravity_grid=oil_api_gravity_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
    )
    new_water_bubble_point_pressure_grid = build_2D_water_bubble_point_pressure_grid(
        temperature_grid=fluid_properties.temperature_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
    )

    # Updated Fluid Compressibility Grids
    new_oil_compressibility_grid = build_2D_oil_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_api_gravity_grid=oil_api_gravity_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
    )
    new_water_compressibility_grid = build_2D_water_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
        gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )
    gas_compressibility_factor_grid = build_2D_gas_compressibility_factor_grid(
        gas_gravity_grid=gas_gravity_grid,
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
    )
    new_gas_compressibility_grid = build_2D_gas_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )

    # Updated Formation Volume Factor Grids
    new_oil_formation_volume_factor_grid = build_2D_oil_formation_volume_factor_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=oil_specific_gravity_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
        oil_compressibility_grid=new_oil_compressibility_grid,
    )
    new_water_formation_volume_factor_grid = (
        build_2D_water_formation_volume_factor_grid(
            water_density_grid=fluid_properties.water_density_grid,
            salinity_grid=fluid_properties.water_salinity_grid,
        )
    )
    new_gas_formation_volume_factor_grid = build_2D_gas_formation_volume_factor_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )

    # Updated Density Grids
    new_oil_density_grid = build_2D_live_oil_density_grid(
        oil_api_gravity_grid=oil_api_gravity_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
        formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
    )
    new_water_density_grid = build_2D_water_density_grid(
        gas_gravity_grid=gas_gravity_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )
    new_gas_density_grid = build_2D_gas_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )

    # Updated Fluid Viscosity Grids
    new_gor_at_bubble_point_pressure_grid = build_2D_gas_to_oil_ratio_grid(
        pressure_grid=new_oil_bubble_point_pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        gas_gravity_grid=gas_gravity_grid,
        oil_api_gravity_grid=oil_api_gravity_grid,
    )
    new_oil_viscosity_grid = build_2D_oil_viscosity_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=oil_specific_gravity_grid,
        gas_to_oil_ratio_grid=new_gas_to_oil_ratio_grid,
        gor_at_bubble_point_pressure_grid=new_gor_at_bubble_point_pressure_grid,
    )
    new_water_viscosity_grid = build_2D_water_viscosity_grid(
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        pressure_grid=fluid_properties.pressure_grid,
    )
    new_gas_viscosity_grid = build_2D_gas_viscosity_grid(
        temperature_grid=fluid_properties.temperature_grid,
        gas_density_grid=new_gas_density_grid,
        gas_molecular_weight_grid=gas_molecular_weight_grid,
    )

    updated_fluid_properties = replace(
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
