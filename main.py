import numpy as np

from sim2D.models import build_2D_reservoir_model
from sim2D.simulation import (
    TwoDimensionalModelSimulator,
    plot_model_time_state,
    animate_model_states,
)
from sim2D.grids import build_2D_layered_grid, build_2D_uniform_grid
from sim2D import boundary_conditions as bc


np.set_printoptions(threshold=np.inf)  # type: ignore


def simulate():
    cell_dimension = (10.0, 10.0)  # Each cell is 10m x 10m
    grid_dimension = (50, 50)
    pressure_range = (1e3, 10e3)  # Pressure range from 1 kPa to 10 kPa
    viscosity_range = (5e-3, 1e-2) # Pressure range 0.005 Pa.s to 0.01 Pa.s
    pressure_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        # layer_values=np.linspace(
        #     pressure_range[0], pressure_range[1], grid_dimension[1]
        # ), # Linear pressure gradient across the grid
        layer_values=np.random.uniform(
            low=pressure_range[0], high=pressure_range[1], size=grid_dimension[1]
        ),  # Random initial pressure in Pa
    )
    fluid_saturation_grid = build_2D_uniform_grid(
        grid_dimension=grid_dimension, value=1.0
    )
    fluid_viscosity_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            viscosity_range[0], viscosity_range[1], grid_dimension[1]
        )
    )
    permeability_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(100, 500, grid_dimension[1]),  # Permeability in mD
        # layer_values=np.random.uniform(
        #     low=100, high=500, size=grid_dimension[1]
        # ),  # Random permeability in mD
    )
    porosity_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.1, 0.3, grid_dimension[1]),  # Porosity in fraction
    )
    temperature_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(280, 320, grid_dimension[1]),  # Temperature in K
    )
    rock_compressibility = 1e-5  # Rock compressibility in 1/Pa
    model = build_2D_reservoir_model(
        grid_dimension=grid_dimension,
        cell_dimension=cell_dimension,
        pressure_grid=pressure_grid,
        fluid_saturation_grid=fluid_saturation_grid,
        fluid_viscosity_grid=fluid_viscosity_grid,
        permeability_grid=permeability_grid,
        porosity_grid=porosity_grid,
        temperature_grid=temperature_grid,
        rock_compressibility=rock_compressibility,
    )

    injected_fluid = "Water"  # Type of injected fluid
    injectors_positions = [
        (10, 5),
        (45, 10),
    ]  # Injectors at bottom left and near the right edge
    producers_positions = [
        (5, 45),
        (40, 40),
    ]  # Producers at top left and near the bottom right edge
    injection_rates = [100.0, 80.0]  # Injection rates in m^3/s
    production_rates = [50.0, 70.0]  # Production rates in m^3/s
    num_of_time_steps = 150  # Number of time steps to simulate
    time_step_size = 1.0 * 3600  # Time step size in seconds (1 hour)
    discretization_method = "explicit"  # Choose between 'implicit' or 'explicit'
    fluid_miscibility = None
    capture_interval = 1  # Capture every time step

    simulator = TwoDimensionalModelSimulator(
        model=model,
        fluid_miscibility=fluid_miscibility,
    )
    boundary_conditions = bc.BoundaryConditions(
        conditions={
            "pressure": bc.GridBoundaryCondition(
                north=bc.NoFlowBoundary(),
                south=bc.NoFlowBoundary(),
                east=bc.ConstantBoundary(1000),
                west=bc.ConstantBoundary(1000),
            )
        }
    )
    model_time_states = simulator.run_simulation(
        num_of_time_steps=num_of_time_steps,
        time_step_size=time_step_size,
        boundary_conditions=boundary_conditions,
        producers_positions=producers_positions,
        production_rates=production_rates,
        injectors_positions=injectors_positions,
        injection_rates=injection_rates,
        injected_fluid=injected_fluid,
        discretization_method=discretization_method,
        capture_interval=capture_interval,
    )

    for i in range(0, len(model_time_states), 4):
        plot_model_time_state(model_time_state=model_time_states[i])

    animate_model_states(
        model_states=model_time_states,
        interval_ms=400,
        save_path=f"./simulation{injected_fluid}{np.random.randint(1000, 12345)}.mp4",
    )
