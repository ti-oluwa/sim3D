import numpy as np

import sim2D
from sim2D.grids import build_2D_layered_grid, build_2D_uniform_grid
from sim2D.properties import (
    compute_fluid_compressibility,
    compute_fluid_viscosity,
    compute_fluid_density,
    compute_water_formation_volume_factor,
    compute_gas_formation_volume_factor,
    compute_gas_compressibility_factor,
)

np.set_printoptions(threshold=np.inf)  # type: ignore


# Ensure that wellbore radius is less than the cell dimension


def simulate() -> None:
    cell_dimension = (100, 100)
    grid_dimension = (50, 50)
    pressure_range = (145, 1000)
    oil_viscosity_range = (5, 10)
    oil_compressibility_range = (7e-7, 1e-5)
    oil_density_range = (55, 58)
    pressure_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        # layer_values=np.linspace(
        #     pressure_range[0], pressure_range[1], grid_dimension[1]
        # ), # Linear pressure gradient across the grid
        layer_values=np.random.uniform(
            low=pressure_range[0], high=pressure_range[1], size=grid_dimension[1]
        ),
    )
    oil_bubble_point_pressure_grid = build_2D_uniform_grid(
        grid_dimension=grid_dimension,
        value=800.0,  # Bubble point pressure in psi
    )
    gas_to_oil_ratio_grid = build_2D_uniform_grid(
        grid_dimension=grid_dimension,
        value=300.0,  # Gas to oil ratio in scf/STB
    )
    oil_saturation_grid = build_2D_uniform_grid(
        grid_dimension=grid_dimension, value=0.8
    )
    residual_oil_saturation_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.05, 0.2, grid_dimension[1]),
    )
    irreducible_water_saturation_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.05, 0.14, grid_dimension[1]),
    )  # Irreducible water saturation in fraction
    oil_viscosity_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_viscosity_range[0], oil_viscosity_range[1], grid_dimension[1]
        ),
    )
    oil_compressibility_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_compressibility_range[0],
            oil_compressibility_range[1],
            grid_dimension[1],
        ),
    )
    oil_density_grid = build_2D_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_density_range[0], oil_density_range[1], grid_dimension[1]
        ),
    )
    absolute_permeability_grid = build_2D_layered_grid(
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
        layer_values=np.linspace(260, 300, grid_dimension[1]),  # Temperature in °F
    )
    rock_compressibility = 1.45e-13  # Rock compressibility in 1/psi

    boundary_conditions = sim2D.BoundaryConditions(
        conditions={
            "pressure": sim2D.GridBoundaryCondition(
                north=sim2D.NoFlowBoundary(),
                south=sim2D.NoFlowBoundary(),
                east=sim2D.ConstantBoundary(1000),
                west=sim2D.ConstantBoundary(1000),
            )
        }
    )
    model = sim2D.build_2D_reservoir_model(
        grid_dimension=grid_dimension,
        cell_dimension=cell_dimension,
        pressure_grid=pressure_grid,
        oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
        absolute_permeability_grid=absolute_permeability_grid,
        porosity_grid=porosity_grid,
        temperature_grid=temperature_grid,
        rock_compressibility=rock_compressibility,
        oil_saturation_grid=oil_saturation_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        oil_density_grid=oil_density_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        residual_oil_saturation_grid=residual_oil_saturation_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        gas_to_oil_ratio_grid=gas_to_oil_ratio_grid,
        boundary_conditions=boundary_conditions,
    )

    # INJECTORS
    fluid_A = "Water"
    fluid_A_density = compute_fluid_density(
        pressure=sim2D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim2D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_A,
    )
    fluid_A_viscosity = compute_fluid_viscosity(
        pressure=sim2D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim2D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_A,
    )
    fluid_A_compressibility = compute_fluid_compressibility(
        pressure=sim2D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim2D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_A,
    )
    fluid_A_formation_volume_factor = compute_water_formation_volume_factor(
        water_density=fluid_A_density,
        salinity=10_000.0,  # Salinity in ppm
    )
    injector_A = sim2D.build_injection_well(
        well_name="Injector A",
        location=(10, 5),  # Position in grid coordinates
        radius=0.3281,
        injected_fluid=sim2D.InjectedFluid(
            name=fluid_A,
            phase=sim2D.FluidPhase.WATER,
            volumetric_flow_rate=300.0,  # STB/day
            density=fluid_A_density,
            viscosity=fluid_A_viscosity,
            compressibility=fluid_A_compressibility,
            formation_volume_factor=fluid_A_formation_volume_factor,
            salinity=10_000.0,  # Salinity in ppm
        ),
    )

    fluid_B = "CO2"
    fluid_B_density = compute_fluid_density(
        pressure=sim2D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim2D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_B,
    )
    fluid_B_viscosity = compute_fluid_viscosity(
        pressure=sim2D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim2D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_B,
    )
    fluid_B_compressibility = compute_fluid_compressibility(
        pressure=sim2D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim2D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_B,
    )
    fluid_B_compressibility_factor = compute_gas_compressibility_factor(
        pressure=sim2D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim2D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_gravity=fluid_B_density / sim2D.constants.STANDARD_WATER_DENSITY,
    )
    fluid_B_formation_volume_factor = compute_gas_formation_volume_factor(
        pressure=sim2D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim2D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_compressibility_factor=fluid_B_compressibility_factor,
    )
    injector_B = sim2D.build_injection_well(
        well_name="Injector B",
        location=(45, 10),  # Position in grid coordinates
        radius=0.3281,
        injected_fluid=sim2D.InjectedFluid(
            name=fluid_B,
            phase=sim2D.FluidPhase.GAS,
            volumetric_flow_rate=1700.0,  # SCF/day
            viscosity=fluid_B_viscosity,
            density=fluid_B_density,
            compressibility=fluid_B_compressibility,
            formation_volume_factor=fluid_B_formation_volume_factor,
        ),
    )

    # PRODUCERS
    producer_A = sim2D.build_production_well(
        well_name="Producer A",
        location=(5, 45),  # Position in grid coordinates
        radius=0.3281,
        produced_fluids=(
            sim2D.ProducedFluid(
                name="Oil",
                phase=sim2D.FluidPhase.OIL,
                volumetric_flow_rate=700.0,  # STB/day
            ),
            sim2D.ProducedFluid(
                name="Gas",
                phase=sim2D.FluidPhase.GAS,
                volumetric_flow_rate=1000.0,  # SCF/day
            ),
            sim2D.ProducedFluid(
                name="Water",
                phase=sim2D.FluidPhase.WATER,
                volumetric_flow_rate=150.0,  # STB/day
            ),
        ),
        skin_factor=0.2,  # Skin factor for the well
    )
    producer_B = sim2D.build_production_well(
        well_name="Producer B",
        location=(40, 40),  # Position in grid coordinates
        radius=0.3281,
        produced_fluids=(
            sim2D.ProducedFluid(
                name="Oil",
                phase=sim2D.FluidPhase.OIL,
                volumetric_flow_rate=800.0,  # STB/day
            ),
            sim2D.ProducedFluid(
                name="Gas",
                phase=sim2D.FluidPhase.GAS,
                volumetric_flow_rate=1500.0,  # SCF/day
            ),
            sim2D.ProducedFluid(
                name="Water", phase=sim2D.FluidPhase.WATER, volumetric_flow_rate=15
            ),
        ),
        skin_factor=0.1,  # Skin factor for the well
    )
    wells = sim2D.Wells(
        injection_wells=[injector_A, injector_B],
        production_wells=[producer_A, producer_B],
    )

    simulation_params = sim2D.SimulationParameters(
        time_step_size=10,  # Time step in seconds (1 hour)
        total_time=86400,  # Total simulation time in seconds (1 day)
        max_iterations=100,  # Maximum number of iterations per time step
        convergence_tolerance=1e-5,  # Convergence tolerance for the simulation
        output_frequency=4,  # Output every 4 time steps
        discretization_method="explicit",
    )
    model_time_states = sim2D.run_simulation(
        model=model,
        wells=wells,
        params=simulation_params,
    )

    # for i in range(0, len(model_time_states), 4):
    #     plot_model_time_state(model_time_state=model_time_states[i])

    # animate_model_states(
    #     model_states=model_time_states,
    #     interval_ms=400,
    #     save_path=f"./simulation{injected_fluid}{np.random.randint(1000, 12345)}.mp4",
    # )
