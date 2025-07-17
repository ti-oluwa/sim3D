import typing
import numpy as np

import sim3D
from sim3D.grids import build_layered_grid, build_uniform_grid
from sim3D.properties import (
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
    cell_dimension = (10.0, 10.0)
    grid_dimension = typing.cast(
        sim3D.ThreeDimensions, (50, 50, 5)
    )  # (x, y, z) dimensions of the grid in cells
    pressure_range = (145.0, 1000.0)
    oil_viscosity_range = (5.0, 10.0)
    oil_compressibility_range = (7e-7, 1e-5)
    oil_density_range = (55.0, 58.0)
    pressure_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        # layer_values=np.linspace(
        #     pressure_range[0], pressure_range[1], grid_dimension[1]
        # ), # Linear pressure gradient across the grid
        layer_values=np.random.uniform(
            low=pressure_range[0], high=pressure_range[1], size=grid_dimension[1]
        ),
        orientation=sim3D.Orientation.Z,
    )
    oil_bubble_point_pressure_grid = build_uniform_grid(
        grid_dimension=grid_dimension,
        value=800.0,  # Bubble point pressure in psi
    )
    gas_to_oil_ratio_grid = build_uniform_grid(
        grid_dimension=grid_dimension,
        value=300.0,  # Gas to oil ratio in scf/STB
    )
    oil_saturation_grid = build_uniform_grid(grid_dimension=grid_dimension, value=0.8)
    residual_oil_saturation_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.05, 0.2, grid_dimension[1]),
        orientation=sim3D.Orientation.Z,
    )
    irreducible_water_saturation_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.05, 0.14, grid_dimension[1]),
        orientation=sim3D.Orientation.Z,
    )  # Irreducible water saturation in fraction
    oil_viscosity_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_viscosity_range[0], oil_viscosity_range[1], grid_dimension[1]
        ),
        orientation=sim3D.Orientation.Z,
    )
    oil_compressibility_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_compressibility_range[0],
            oil_compressibility_range[1],
            grid_dimension[1],
        ),
        orientation=sim3D.Orientation.Z,
    )
    oil_density_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_density_range[0], oil_density_range[1], grid_dimension[1]
        ),
        orientation=sim3D.Orientation.Z,
    )
    x_permeability_grid = build_uniform_grid(
        grid_dimension=grid_dimension,
        value=300,  # Permeability in mD
    )
    y_permeability_grid = build_uniform_grid(
        grid_dimension=grid_dimension,
        value=150,  # Permeability in mD
    )
    z_permeability_grid = build_uniform_grid(
        grid_dimension=grid_dimension,
        value=200,  # Permeability in mD
    )
    absolute_permeability = sim3D.RockPermeability(
        x=x_permeability_grid,
        y=y_permeability_grid,
        z=z_permeability_grid,
    )
    porosity_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.1, 0.3, grid_dimension[1]),  # Porosity in fraction
        orientation=sim3D.Orientation.Z,
    )
    temperature_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(260, 300, grid_dimension[1]),  # Temperature in °F
        orientation=sim3D.Orientation.Z,
    )
    rock_compressibility = 1.45e-13  # Rock compressibility in 1/psi

    boundary_conditions = sim3D.BoundaryConditions(
        conditions={
            "pressure": sim3D.GridBoundaryCondition(
                x_minus=sim3D.NoFlowBoundary(),
                x_plus=sim3D.NoFlowBoundary(),
                y_minus=sim3D.ConstantBoundary(1000),
                y_plus=sim3D.ConstantBoundary(1000),
            )
        }
    )
    height_grid = build_uniform_grid(
        grid_dimension=grid_dimension,
        value=20.0,  # Height of the reservoir in feet
    )
    model = sim3D.build_reservoir_model(
        grid_dimension=grid_dimension,
        cell_dimension=cell_dimension,
        height_grid=height_grid,
        pressure_grid=pressure_grid,
        oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
        absolute_permeability=absolute_permeability,
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
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_A,
    )
    fluid_A_viscosity = compute_fluid_viscosity(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_A,
    )
    fluid_A_compressibility = compute_fluid_compressibility(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_A,
    )
    fluid_A_formation_volume_factor = compute_water_formation_volume_factor(
        water_density=fluid_A_density,
        salinity=10_000.0,  # Salinity in ppm
    )
    injector_A = sim3D.build_injection_well(
        well_name="Injector A",
        radius=0.3281,
        perforating_interval=(
            (0, 0, 0),
            (49, 49, 0),
        ),  # Perforating interval in grid coordinates
        bottom_hole_pressure=1000.0,  # Bottom hole pressure in psi
        injected_fluid=sim3D.InjectedFluid(
            name=fluid_A,
            phase=sim3D.FluidPhase.WATER,
            density=fluid_A_density,
            viscosity=fluid_A_viscosity,
            compressibility=fluid_A_compressibility,
            formation_volume_factor=fluid_A_formation_volume_factor,
            salinity=10_000.0,  # Salinity in ppm
        ),
    )

    fluid_B = "CO2"
    fluid_B_density = compute_fluid_density(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_B,
    )
    fluid_B_viscosity = compute_fluid_viscosity(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_B,
    )
    fluid_B_compressibility = compute_fluid_compressibility(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        fluid=fluid_B,
    )
    fluid_B_compressibility_factor = compute_gas_compressibility_factor(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_gravity=fluid_B_density / sim3D.constants.STANDARD_WATER_DENSITY,
    )
    fluid_B_formation_volume_factor = compute_gas_formation_volume_factor(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_compressibility_factor=fluid_B_compressibility_factor,
    )
    injector_B = sim3D.build_injection_well(
        well_name="Injector B",
        perforating_interval=(
            (45, 10, 0),
            (45, 10, 49),
        ),  # Perforating interval in grid coordinates
        radius=0.3281,
        bottom_hole_pressure=1000.0,  # Bottom hole pressure in psi
        injected_fluid=sim3D.InjectedFluid(
            name=fluid_B,
            phase=sim3D.FluidPhase.GAS,
            viscosity=fluid_B_viscosity,
            density=fluid_B_density,
            compressibility=fluid_B_compressibility,
            formation_volume_factor=fluid_B_formation_volume_factor,
        ),
    )

    # PRODUCERS
    producer_A = sim3D.build_production_well(
        well_name="Producer A",
        perforating_interval=(
            (5, 45, 0),
            (5, 45, 49),
        ),  # Perforating interval in grid coordinates
        radius=0.3281,
        bottom_hole_pressure=1000.0,  # Bottom hole pressure in psi
        produced_fluids=(
            sim3D.ProducedFluid(
                name="Oil",
                phase=sim3D.FluidPhase.OIL,
            ),
            sim3D.ProducedFluid(
                name="Gas",
                phase=sim3D.FluidPhase.GAS,
            ),
            sim3D.ProducedFluid(
                name="Water",
                phase=sim3D.FluidPhase.WATER,
            ),
        ),
        skin_factor=0.2,  # Skin factor for the well
    )
    producer_B = sim3D.build_production_well(
        well_name="Producer B",
        perforating_interval=(
            (40, 40, 0),
            (40, 40, 49),
        ),  # Perforating interval in grid coordinates
        radius=0.3281,
        bottom_hole_pressure=1000.0,  # Bottom hole pressure in psi
        produced_fluids=(
            sim3D.ProducedFluid(
                name="Oil",
                phase=sim3D.FluidPhase.OIL,
            ),
            sim3D.ProducedFluid(
                name="Gas",
                phase=sim3D.FluidPhase.GAS,
            ),
            sim3D.ProducedFluid(name="Water", phase=sim3D.FluidPhase.WATER),
        ),
        skin_factor=0.1,  # Skin factor for the well
    )
    producer_A.update_schedule(
        sim3D.ProductionWellScheduledEvent(
            time_step=9, bottom_hole_pressure=800.0 # Change bottom hole pressure after 9 time steps
        )
    )
    producer_B.update_schedule(
        sim3D.ProductionWellScheduledEvent(
            time_step=5, is_active=False
        )  # Shut down producer B after 5 time steps
    )
    wells = sim3D.Wells(
        injection_wells=[injector_A, injector_B],
        production_wells=[producer_A, producer_B],
    )

    simulation_params = sim3D.SimulationParameters(
        time_step_size=10,  # Time step in seconds (1 hour)
        total_time=86400,  # Total simulation time in seconds (1 day)
        max_iterations=100,  # Maximum number of iterations per time step
        convergence_tolerance=1e-5,  # Convergence tolerance for the simulation
        output_frequency=4,  # Output every 4 time steps
        discretization_method="explicit",
    )
    model_states = sim3D.run_3D_simulation(
        model=model,
        wells=wells,
        params=simulation_params,
    )

    # for i in range(0, len(model_states), 4):
    #     plot_model_time_state(model_time_state=model_states[i])

    # animate_model_states(
    #     model_states=model_states,
    #     interval_ms=400,
    #     save_path=f"./simulation{injected_fluid}{np.random.randint(1000, 12345)}.mp4",
    # )
