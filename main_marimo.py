import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full", app_title="SIM3D")


@app.cell
def _():
    import logging
    import typing

    import numpy as np

    import sim3D
    from sim3D.properties import (
        compute_gas_compressibility,
        compute_gas_compressibility_factor,
        compute_gas_formation_volume_factor,
        compute_gas_gravity,
        compute_gas_molecular_weight,
    )

    logging.getLogger("sim3D").setLevel(logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore


    def main():
        cell_dimension = (300.0, 300.0)
        grid_shape = typing.cast(
            sim3D.ThreeDimensions, (10, 10, 7)
        )  # (x, y, z) dimensions of the grid in cells
        # Height of each cell in the z-direction (in feet)
        thickness_grid = sim3D.uniform_grid(grid_shape=grid_shape, value=150.0)
        thickness_grid[:, :, 2] = 90.0  # Making the top layer thinner
        pressure_range = (500.0, 800.0)
        oil_viscosity_range = (5.0, 10.0)
        oil_compressibility_range = (7e-7, 1e-5)
        oil_specific_gravity_range = (0.8, 0.9)

        # pressures in psi, viscosities in cP, compressibilities in 1/psi, densities in lb/ft^3
        pressure_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.random.uniform(
                low=pressure_range[0],
                high=pressure_range[1],
                size=grid_shape[2],
            ),
            orientation=sim3D.Orientation.Z,
        )
        oil_bubble_point_pressure_grid = sim3D.uniform_grid(
            grid_shape=grid_shape, value=800.0
        )
        oil_saturation_grid = sim3D.uniform_grid(grid_shape=grid_shape, value=0.64)
        water_saturation_grid = sim3D.uniform_grid(
            grid_shape=grid_shape, value=0.23
        )
        gas_saturation_grid = typing.cast(
            sim3D.ThreeDimensionalGrid,
            1.0 - oil_saturation_grid - water_saturation_grid,
        )
        residual_oil_saturation_water_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(0.05, 0.2, grid_shape[2]),
            orientation=sim3D.Orientation.Z,
        )
        residual_oil_saturation_gas_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(0.1, 0.25, grid_shape[2]),
            orientation=sim3D.Orientation.Z,
        )
        irreducible_water_saturation_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(0.05, 0.14, grid_shape[2]),
            orientation=sim3D.Orientation.Z,
        )
        residual_gas_saturation_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(0.05, 0.1, grid_shape[2]),
            orientation=sim3D.Orientation.Z,
        )
        oil_viscosity_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(
                oil_viscosity_range[0], oil_viscosity_range[1], grid_shape[2]
            ),
            orientation=sim3D.Orientation.Z,
        )
        oil_compressibility_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(
                oil_compressibility_range[0],
                oil_compressibility_range[1],
                grid_shape[2],
            ),
            orientation=sim3D.Orientation.Z,
        )
        oil_specific_gravity_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(
                oil_specific_gravity_range[0],
                oil_specific_gravity_range[1],
                grid_shape[2],
            ),
            orientation=sim3D.Orientation.Z,
        )
        # Permeabilities in mD
        x_permeability_grid = sim3D.uniform_grid(grid_shape=grid_shape, value=50)
        y_permeability_grid = sim3D.uniform_grid(grid_shape=grid_shape, value=30)
        z_permeability_grid = typing.cast(
            sim3D.ThreeDimensionalGrid, x_permeability_grid * 0.2
        )
        absolute_permeability = sim3D.RockPermeability(
            x=x_permeability_grid,
            y=y_permeability_grid,
            z=z_permeability_grid,
        )
        # relative_permeability_func = sim3D.BrooksCoreyThreePhaseRelPermModel(
        #     irreducible_water_saturation=float(
        #         np.nanmin(irreducible_water_saturation_grid)
        #     ),
        #     residual_oil_saturation_water=float(
        #         np.nanmin(residual_oil_saturation_water_grid)
        #     ),
        #     residual_oil_saturation_gas=float(
        #         np.nanmin(residual_oil_saturation_gas_grid)
        #     ),
        #     residual_gas_saturation=0.05,
        #     mixing_rule=sim3D.stone_II_rule,
        # )
        relative_permeability_func = sim3D.ThreePhaseRelPermTable(
            oil_water_table=sim3D.TwoPhaseRelPermTable(
                phase1=sim3D.FluidPhase.OIL,
                phase2=sim3D.FluidPhase.WATER,
                saturation=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                phase1_relative_permeability=np.array(
                    [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                ),
                phase2_relative_permeability=np.array(
                    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ),
            ),
            oil_gas_table=sim3D.TwoPhaseRelPermTable(
                phase1=sim3D.FluidPhase.OIL,
                phase2=sim3D.FluidPhase.GAS,
                saturation=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                phase1_relative_permeability=np.array(
                    [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
                ),
                phase2_relative_permeability=np.array(
                    [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
                ),
            ),
            mixing_rule=sim3D.stone_II_rule,
        )
        capillary_pressure_params = sim3D.CapillaryPressureParameters(
            wettability=sim3D.Wettability.WATER_WET,
            oil_water_entry_pressure_oil_wet=1.5,  # in psi
            oil_water_pore_size_distribution_index_oil_wet=2.0,
            gas_oil_entry_pressure=0.5,  # in psi
            gas_oil_pore_size_distribution_index=2.0,
        )
        porosity_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(
                0.1, 0.2, grid_shape[2]
            ),  # Porosity in fraction
            orientation=sim3D.Orientation.Z,
        )
        temperature_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=np.linspace(100, 300, grid_shape[2]),  # Temperature in °F
            orientation=sim3D.Orientation.Z,
        )
        rock_compressibility = 1.45e-13  # Rock compressibility in 1/psi

        boundary_conditions = sim3D.BoundaryConditions(
            conditions={
                "pressure": sim3D.GridBoundaryCondition(
                    y_minus=sim3D.DirichletBoundary(1500),
                    y_plus=sim3D.DirichletBoundary(1500),
                    # Fixed bottom boundary pressure to simulate aquifer support
                    z_minus=sim3D.DirichletBoundary(2300),
                )
            }
        )
        methane_gravity = compute_gas_gravity(gas="Methane")
        gas_gravity_grid = sim3D.uniform_grid(
            grid_shape=grid_shape, value=methane_gravity
        )
        fault_network = [
            sim3D.Fault(
                id="main_fault",
                orientation="x",
                fault_index=4,
                slope=0.1,
                transmissibility_scale=1e-4,
                fault_permeability=0.1,
                geometric_throw_cells=2,
                preserve_grid_data=True,
            ),
            sim3D.Fault(
                id="cross_fault",
                orientation="y",
                fault_index=5,
                transmissibility_scale=1e-3,
                fault_permeability=1.0,
            ),
            sim3D.Fault(
                id="subsidiary_fault",
                orientation="x",
                fault_index=4,
                slope=-0.2,  # Opposite dip
                transmissibility_scale=1e-2,
                fault_permeability=10.0,
                conductive=True,  # Fractured zone
            ),
        ]
        model = sim3D.reservoir_model(
            grid_shape=grid_shape,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            pressure_grid=pressure_grid,
            oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            absolute_permeability=absolute_permeability,
            porosity_grid=porosity_grid,
            temperature_grid=temperature_grid,
            rock_compressibility=rock_compressibility,
            oil_saturation_grid=oil_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            oil_viscosity_grid=oil_viscosity_grid,
            oil_specific_gravity_grid=oil_specific_gravity_grid,
            oil_compressibility_grid=oil_compressibility_grid,
            gas_gravity_grid=gas_gravity_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            boundary_conditions=boundary_conditions,
            faults=fault_network,
            relative_permeability_func=relative_permeability_func,
            capillary_pressure_params=capillary_pressure_params,
            reservoir_gas_name="CO2",
        )

        fluid_A = "CO2"
        fluid_A_gravity = compute_gas_gravity(gas=fluid_A)
        fluid_A_compressibility_factor = compute_gas_compressibility_factor(
            pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
            temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
            gas_gravity=fluid_A_gravity,
        )
        fluid_A_molecular_weight = compute_gas_molecular_weight(
            gas_gravity=fluid_A_gravity
        )
        fluid_A_compressibility = compute_gas_compressibility(
            pressure=sim3D.constants.STANDARD_PRESSURE_IMPERIAL,
            temperature=sim3D.constants.STANDARD_TEMPERATURE_IMPERIAL,
            gas_gravity=fluid_A_gravity,
            gas_compressibility_factor=fluid_A_compressibility_factor,
        )
        injector_A = sim3D.injection_well(
            well_name="Injector A",
            perforating_interval=(
                (4, 3, 3),
                (4, 3, 6),
            ),  # Perforating interval in grid coordinates
            radius=0.9281,
            bottom_hole_pressure=1200.0,  # Bottom hole pressure in psi
            injected_fluid=sim3D.WellFluid(
                name=fluid_A,
                phase=sim3D.FluidPhase.GAS,
                specific_gravity=fluid_A_gravity,
                molecular_weight=fluid_A_molecular_weight,
                compressibility=fluid_A_compressibility,
            ),
        )

        injector_B = sim3D.injection_well(
            well_name="Injector B",
            perforating_interval=(
                (2, 7, 3),
                (2, 7, 6),
            ),  # Perforating interval in grid coordinates
            radius=0.9281,
            bottom_hole_pressure=1000.0,  # Bottom hole pressure in psi
            injected_fluid=sim3D.WellFluid(
                name=fluid_A,
                phase=sim3D.FluidPhase.GAS,
                specific_gravity=fluid_A_gravity,
                molecular_weight=fluid_A_molecular_weight,
                compressibility=fluid_A_compressibility,
            ),
        )

        producer_A = sim3D.production_well(
            well_name="Producer A",
            perforating_interval=(
                (7, 7, 3),
                (7, 7, 6),
            ),  # Perforating interval in grid coordinates
            radius=0.5281,
            bottom_hole_pressure=425.0,  # Bottom hole pressure in psi
            produced_fluids=(
                sim3D.WellFluid(
                    name="Oil",
                    phase=sim3D.FluidPhase.OIL,
                    specific_gravity=0.85,  # Average specific gravity for oil
                    molecular_weight=170.0,  # Average molecular weight for oil
                ),
                sim3D.WellFluid(
                    name="Gas",
                    phase=sim3D.FluidPhase.GAS,
                    specific_gravity=1.5,  # Specific gravity for gas (e.g., CO2)
                    molecular_weight=44.01,  # Molecular weight for CO2
                ),
                sim3D.WellFluid(
                    name="Water",
                    phase=sim3D.FluidPhase.WATER,
                    specific_gravity=1.0,  # Specific gravity for water
                    molecular_weight=18.015,  # Molecular weight for water
                ),
            ),
            skin_factor=3.2,  # Skin factor for the well
        )
        producer_A.update_schedule(
            sim3D.WellEvent(
                hook=sim3D.well_time_hook(time_step=200),
                action=sim3D.well_props_action(bottom_hole_pressure=300.0),
            )
        )
        wells = sim3D.wells(
            injectors=[injector_A, injector_B], producers=[producer_A]
        )

        options = sim3D.Options(
            time_step_size=300,
            total_time=72000 * 4,
            max_iterations=1500,
            convergence_tolerance=1e-3,
            output_frequency=1,
            evolution_scheme="implicit_explicit",
        )
        model_states = sim3D.run_simulation(
            model=model,
            wells=wells,
            options=options,
        )
        return list(model_states)
    return main, np, sim3D


@app.cell
def _(main):
    model_states = main()
    return (model_states,)


@app.cell
def _(model_states, np, sim3D):
    analysis = sim3D.ModelAnalysis(model_states)
    oil_production_history = analysis.oil_production_history(interval=30, incremental=True)
    water_production_history = analysis.water_production_history(interval=30, incremental=True)
    gas_production_history = analysis.gas_production_history(interval=30, incremental=True)
    oil_in_place_history = analysis.oil_in_place_history(interval=10)
    gas_in_place_history = analysis.gas_in_place_history(interval=10)
    water_in_place_history = analysis.water_in_place_history(interval=10)
    ipr_method = analysis.recommend_ipr_method()
    productivity_history = analysis.productivity_history(
        ipr_method=ipr_method, interval=10
    )

    oil_saturation_history = []
    water_saturation_history = []
    gas_saturation_history = []
    ipr_data = []
    productivity_data = []

    for i in range(0, len(model_states), 10):
        state = model_states[i]
        model = state.model
        time_step = state.time_step
        avg_oil_sat = np.mean(model.fluid_properties.oil_saturation_grid)
        avg_water_sat = np.mean(model.fluid_properties.water_saturation_grid)
        avg_gas_sat = np.mean(model.fluid_properties.gas_saturation_grid)
        oil_saturation_history.append((time_step, avg_oil_sat))
        water_saturation_history.append((time_step, avg_water_sat))
        gas_saturation_history.append((time_step, avg_gas_sat))

    for state in model_states[1:]:
        avg_pressure = np.mean(state.model.fluid_properties.pressure_grid)
        avg_production = np.mean(state.production.oil)
        ipr_data.append((avg_pressure, avg_production))

    for time_step, result in productivity_history:
        productivity_data.append((time_step, result.productivity_index))

    sim3D.plotly2d.make_series_plot(
        data={
            # "Oil Production": np.array(list(oil_production_history)),
            # "Water Production": np.array(list(water_production_history)),
            # "Gas Production": np.array(list(gas_production_history)),
            # "Avg. Oil Saturation": np.array(oil_saturation_history),
            # "Avg. Water Saturation": np.array(water_saturation_history),
            # "Avg. Gas Saturation": np.array(gas_saturation_history),
            # "IPR": np.array(ipr_data),
            # "Oil In Place": np.array(list(oil_in_place_history)),
            # "Gas In Place": np.array(list(gas_in_place_history)),
            # "Water In Place": np.array(list(water_in_place_history)),
            "Productivity Index": np.array(productivity_data),
        },
        title="Historical Analyses",
        x_label="Time Step",
        width=1080,
        height=580,
    )
    return


@app.cell
def _(model_states, sim3D):
    wells = model_states[0].wells
    injector_locations, producer_locations = wells.locations
    injector_names, producer_names = wells.names
    well_positions = [*injector_locations, *producer_locations]
    well_names = [*injector_names, *producer_names]
    labels = sim3D.plotly3d.Labels()
    labels.add_well_labels(well_positions, well_names)
    sim3D.plotly3d.viz.make_plot(
        model_states[200],
        property="oil-pressure",
        plot_type=sim3D.plotly3d.PlotType.VOLUME_RENDER,
        width=960,
        height=600,
        # isomin=0.13,
        # cmin=600,
        # cmax=2700,
        opacity=0.5,
        use_opacity_scaling=False,
        # subsampling_factor=2,
        # downsampling_factor=1,
        # x_slice=(6, 9),
        # y_slice=(6, 9),
        z_slice=(1, 7),
        labels=labels,
        aspect_mode="data",
        # marker_size=12,
        # notebook=True,
    )
    return


if __name__ == "__main__":
    app.run()
