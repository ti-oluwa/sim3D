import typing
import numpy as np

import sim3D
from sim3D.grids import build_layered_grid, build_uniform_grid
from sim3D.properties import (
    compute_gas_compressibility,
    compute_gas_molecular_weight,
    compute_gas_viscosity,
    compute_gas_formation_volume_factor,
    compute_gas_compressibility_factor,
    compute_gas_gravity,
    compute_gas_density,
)

np.set_printoptions(threshold=np.inf)  # type: ignore


def simulate():
    cell_dimension = (100.0, 100.0)
    grid_dimension = typing.cast(
        sim3D.ThreeDimensions, (10, 10, 5)
    )  # (x, y, z) dimensions of the grid in cells
    # Height of each cell in the z-direction (in feet)
    thickness_grid = build_uniform_grid(grid_dimension=grid_dimension, value=70.0)
    pressure_range = (1500.0, 2000.0)
    oil_viscosity_range = (5.0, 10.0)
    oil_compressibility_range = (7e-7, 1e-5)
    oil_specific_gravity_range = (0.8, 0.9)

    # pressures in psi, viscosities in cP, compressibilities in 1/psi, densities in lb/ft^3
    pressure_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.random.uniform(
            low=pressure_range[0], high=pressure_range[1], size=grid_dimension[2]
        ),
        orientation=sim3D.Orientation.Z,
    )
    oil_bubble_point_pressure_grid = build_uniform_grid(
        grid_dimension=grid_dimension, value=800.0
    )
    oil_saturation_grid = build_uniform_grid(grid_dimension=grid_dimension, value=0.8)
    residual_oil_saturation_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.05, 0.2, grid_dimension[2]),
        orientation=sim3D.Orientation.Z,
    )
    irreducible_water_saturation_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.05, 0.14, grid_dimension[2]),
        orientation=sim3D.Orientation.Z,
    )  # Irreducible water saturation in fraction
    oil_viscosity_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_viscosity_range[0], oil_viscosity_range[1], grid_dimension[2]
        ),
        orientation=sim3D.Orientation.Z,
    )
    oil_compressibility_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_compressibility_range[0],
            oil_compressibility_range[1],
            grid_dimension[2],
        ),
        orientation=sim3D.Orientation.Z,
    )
    oil_specific_gravity_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(
            oil_specific_gravity_range[0],
            oil_specific_gravity_range[1],
            grid_dimension[2],
        ),
        orientation=sim3D.Orientation.Z,
    )
    # Permeabilities in mD
    x_permeability_grid = build_uniform_grid(grid_dimension=grid_dimension, value=50)
    y_permeability_grid = build_uniform_grid(grid_dimension=grid_dimension, value=30)
    z_permeability_grid = build_uniform_grid(grid_dimension=grid_dimension, value=5)
    absolute_permeability = sim3D.RockPermeability(
        x=x_permeability_grid,
        y=y_permeability_grid,
        z=z_permeability_grid,
    )
    porosity_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(0.1, 0.3, grid_dimension[2]),  # Porosity in fraction
        orientation=sim3D.Orientation.Z,
    )
    temperature_grid = build_layered_grid(
        grid_dimension=grid_dimension,
        layer_values=np.linspace(100, 300, grid_dimension[2]),  # Temperature in °F
        orientation=sim3D.Orientation.Z,
    )
    rock_compressibility = 1.45e-13  # Rock compressibility in 1/psi

    boundary_conditions = sim3D.BoundaryConditions(
        # conditions={
        #     "pressure": sim3D.GridBoundaryCondition(
        #         x_minus=sim3D.NoFlowBoundary(),
        #         x_plus=sim3D.NoFlowBoundary(),
        #         y_minus=sim3D.ConstantBoundary(1000),
        #         y_plus=sim3D.ConstantBoundary(1000),
        #     )
        # }
    )
    methane_gravity = compute_gas_gravity(gas="Methane")
    gas_gravity_grid = build_uniform_grid(
        grid_dimension=grid_dimension, value=methane_gravity
    )
    model = sim3D.build_reservoir_model(
        grid_dimension=grid_dimension,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        pressure_grid=pressure_grid,
        oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
        absolute_permeability=absolute_permeability,
        porosity_grid=porosity_grid,
        temperature_grid=temperature_grid,
        rock_compressibility=rock_compressibility,
        oil_saturation_grid=oil_saturation_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        oil_specific_gravity_grid=oil_specific_gravity_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        gas_gravity_grid=gas_gravity_grid,
        residual_oil_saturation_grid=residual_oil_saturation_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        boundary_conditions=boundary_conditions,
        reservoir_gas_name="Methane",
    )

    fluid_A = "N2"
    fluid_A_gravity = compute_gas_gravity(gas=fluid_A)
    fluid_A_compressibility_factor = compute_gas_compressibility_factor(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_gravity=fluid_A_gravity,
    )
    fluid_A_density = compute_gas_density(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_gravity=fluid_A_gravity,
        gas_compressibility_factor=fluid_A_compressibility_factor,
    )
    fluid_A_molecular_weight = compute_gas_molecular_weight(gas_gravity=fluid_A_gravity)
    fluid_A_viscosity = compute_gas_viscosity(
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_density=fluid_A_density,
        gas_molecular_weight=fluid_A_molecular_weight,
    )
    fluid_A_compressibility = compute_gas_compressibility(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_gravity=fluid_A_gravity,
        gas_compressibility_factor=fluid_A_compressibility_factor,
    )
    fluid_A_formation_volume_factor = compute_gas_formation_volume_factor(
        pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
        temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
        gas_compressibility_factor=fluid_A_compressibility_factor,
    )
    injector_A = sim3D.build_injection_well(
        well_name="Injector A",
        perforating_interval=(
            (8, 9, 0),
            (8, 9, 4),
        ),  # Perforating interval in grid coordinates
        radius=0.3281,
        bottom_hole_pressure=2300.0,  # Bottom hole pressure in psi
        injected_fluid=sim3D.InjectedFluid(
            name=fluid_A,
            phase=sim3D.FluidPhase.GAS,
            viscosity=fluid_A_viscosity,
            density=fluid_A_density,
            compressibility=fluid_A_compressibility,
            formation_volume_factor=fluid_A_formation_volume_factor,
        ),
    )

    producer_A = sim3D.build_production_well(
        well_name="Producer A",
        perforating_interval=(
            (1, 1, 0),
            (1, 1, 4),
        ),  # Perforating interval in grid coordinates
        radius=0.3281,
        bottom_hole_pressure=1520.0,  # Bottom hole pressure in psi
        produced_fluids=(
            sim3D.ProducedFluid(
                name="Oil",
                phase=sim3D.FluidPhase.OIL,
            ),
            # sim3D.ProducedFluid(
            #     name="Gas",
            #     phase=sim3D.FluidPhase.GAS,
            # ),
            sim3D.ProducedFluid(
                name="Water",
                phase=sim3D.FluidPhase.WATER,
            ),
        ),
        skin_factor=3.2,  # Skin factor for the well
    )
    producer_A.update_schedule(
        event=sim3D.ProductionWellScheduledEvent(
            time_step=9,
            # Change bottom hole pressure after 9 time steps
            bottom_hole_pressure=2000.0,
        )
    )
    wells = sim3D.Wells(
        injection_wells=[injector_A],
        production_wells=[producer_A],
    )

    simulation_params = sim3D.SimulationParameters(
        time_step_size=3600,
        total_time=72000 * 6,
        max_iterations=100,
        convergence_tolerance=1e-5,
        output_frequency=1,
        evolution_scheme="implicit",
    )
    model_states = sim3D.run_simulation(
        model=model,
        wells=wells,
        params=simulation_params,
    )
    return list(model_states)
