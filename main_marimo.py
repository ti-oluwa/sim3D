import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full", app_title="SIM3D")


@app.cell
def _():
    import logging
    import typing
    from rich import print as pprint

    import numpy as np

    import sim3D
    from sim3D.properties import (
        compute_gas_compressibility,
        compute_gas_compressibility_factor,
        compute_gas_formation_volume_factor,
        compute_gas_gravity,
        compute_gas_molecular_weight,
    )

    np.set_printoptions(threshold=np.inf)  # type: ignore

    def main():
        # More realistic grid dimensions - typical field scale
        cell_dimension = (500.0, 500.0)  # 500ft x 500ft cells
        grid_shape = typing.cast(
            sim3D.ThreeDimensions,
            (15, 15, 5),  # 15x15 cells, 5 layers
        )

        # Realistic thickness distribution - typical reservoir layers
        # Thicker in the middle, thinner at top/bottom
        thickness_values = np.array([25.0, 40.0, 50.0, 35.0, 20.0])  # feet
        thickness_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=thickness_values,
            orientation=sim3D.Orientation.Z,
        )

        # Realistic pressure gradient: ~0.433 psi/ft for water, slightly less for oil
        # Assuming reservoir top at 8000 ft depth
        reservoir_top_depth = 8000.0  # ft
        pressure_gradient = 0.38  # psi/ft (typical for oil reservoirs)
        layer_depths = reservoir_top_depth + np.cumsum(
            np.concatenate([[0], thickness_values[:-1]])
        )
        layer_pressures = 14.7 + (
            layer_depths * pressure_gradient
        )  # Add atmospheric pressure

        pressure_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=layer_pressures,  # Ranges from ~3055 to ~3117 psi
            orientation=sim3D.Orientation.Z,
        )

        # Bubble point pressure slightly below initial pressure (undersaturated oil)
        oil_bubble_point_pressure_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=layer_pressures - 200.0,  # 200 psi below formation pressure
            orientation=sim3D.Orientation.Z,
        )

        # Realistic saturation endpoints - typical for sandstone reservoirs
        residual_oil_saturation_water_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.25,  # Sor to water
        )
        residual_oil_saturation_gas_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.15,  # Sor to gas
        )
        irreducible_water_saturation_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.20,  # Swi
        )
        connate_water_saturation_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.18,  # Slightly less than Swi
        )
        residual_gas_saturation_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.05,  # Sgr
        )

        # Realistic porosity - decreasing with depth (compaction trend)
        porosity_values = np.linspace(0.28, 0.18, grid_shape[2])  # 28% to 18%
        porosity_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=porosity_values,
            orientation=sim3D.Orientation.Z,
        )

        # Realistic fluid contacts
        # GOC at 8050 ft, OWC at 8150 ft (realistic spacing)
        goc_depth = 8050.0
        owc_depth = 8150.0

        depth_grid = sim3D.depth_grid(thickness_grid)
        water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
            sim3D.build_saturation_grids(
                depth_grid=depth_grid,
                gas_oil_contact=goc_depth - reservoir_top_depth,  # 50 ft below top
                oil_water_contact=owc_depth - reservoir_top_depth,  # 150 ft below top
                connate_water_saturation_grid=connate_water_saturation_grid,
                residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
                residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
                residual_gas_saturation_grid=residual_gas_saturation_grid,
                porosity_grid=porosity_grid,
                use_transition_zones=True,
                oil_water_transition_thickness=15.0,  # Realistic transition zone
                gas_oil_transition_thickness=8.0,
                transition_curvature_exponent=1.5,
            )
        )

        # Realistic oil viscosity - increases slightly with depth (heavier oil)
        oil_viscosity_values = np.linspace(1.2, 2.5, grid_shape[2])  # cP
        oil_viscosity_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=oil_viscosity_values,
            orientation=sim3D.Orientation.Z,
        )

        # Realistic oil compressibility - typical range for crude oil
        oil_compressibility_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=1.2e-5,  # 1/psi
        )

        # Realistic oil specific gravity (API ~35-40 degrees)
        oil_specific_gravity_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.845,  # ~36 API
        )

        # Realistic permeability distribution
        # Higher permeability in middle layers (better reservoir quality)
        # Anisotropy ratio kv/kh ~ 0.1 (typical for layered sandstone)
        x_perm_values = np.array([80.0, 200.0, 350.0, 180.0, 60.0])  # mD
        x_permeability_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=x_perm_values,
            orientation=sim3D.Orientation.Z,
        )

        # Slight directional permeability difference
        y_permeability_grid = typing.cast(
            sim3D.ThreeDimensionalGrid, x_permeability_grid * 0.8
        )

        # Vertical permeability much lower (layering effect)
        z_permeability_grid = typing.cast(
            sim3D.ThreeDimensionalGrid, x_permeability_grid * 0.1
        )

        absolute_permeability = sim3D.RockPermeability(
            x=x_permeability_grid,
            y=y_permeability_grid,
            z=z_permeability_grid,
        )

        # Using tabular relative permeability (more realistic than analytical models)
        relative_permeability_func = sim3D.ThreePhaseRelPermTable(
            oil_water_table=sim3D.TwoPhaseRelPermTable(
                phase1=sim3D.FluidPhase.OIL,
                phase2=sim3D.FluidPhase.WATER,
                saturation=np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]),
                phase1_relative_permeability=np.array(
                    [1.0, 0.65, 0.40, 0.22, 0.10, 0.03, 0.0]
                ),
                phase2_relative_permeability=np.array(
                    [0.0, 0.005, 0.02, 0.06, 0.15, 0.30, 0.55]
                ),
            ),
            oil_gas_table=sim3D.TwoPhaseRelPermTable(
                phase1=sim3D.FluidPhase.OIL,
                phase2=sim3D.FluidPhase.GAS,
                saturation=np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]),
                phase1_relative_permeability=np.array(
                    [1.0, 0.70, 0.48, 0.30, 0.16, 0.06, 0.0]
                ),
                phase2_relative_permeability=np.array(
                    [0.0, 0.02, 0.08, 0.20, 0.40, 0.65, 0.90]
                ),
            ),
            mixing_rule=sim3D.stone_I_rule,
        )

        # Realistic capillary pressure parameters
        capillary_pressure_params = sim3D.CapillaryPressureParameters(
            wettability=sim3D.Wettability.WATER_WET,
            oil_water_entry_pressure_oil_wet=2.5,  # psi
            oil_water_pore_size_distribution_index_oil_wet=2.5,
            gas_oil_entry_pressure=0.8,  # psi
            gas_oil_pore_size_distribution_index=2.2,
        )

        # Realistic temperature gradient (~1.5°F per 100 ft)
        surface_temp = 60.0  # °F
        temp_gradient = 0.015  # °F/ft
        layer_temps = surface_temp + (layer_depths * temp_gradient)
        temperature_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=layer_temps,  # ~180-182°F
            orientation=sim3D.Orientation.Z,
        )

        # Realistic rock compressibility for sandstone
        rock_compressibility = 4.5e-6  # 1/psi

        # Net-to-gross ratio (accounting for shale layers)
        net_to_gross_grid = sim3D.uniform_grid(grid_shape=grid_shape, value=0.85)

        # Realistic boundary conditions - aquifer support from bottom
        boundary_conditions = sim3D.BoundaryConditions(
            conditions={
                "pressure": sim3D.GridBoundaryCondition(
                    z_minus=sim3D.DirichletBoundary(
                        layer_pressures[-1] + 50
                    ),  # Aquifer pressure
                )
            }
        )

        # Natural gas (associated gas) properties
        gas_gravity = 0.65  # Typical for associated gas
        gas_gravity_grid = sim3D.uniform_grid(grid_shape=grid_shape, value=gas_gravity)

        # Realistic fault network - simplified
        fault_network = [
            sim3D.Fault(
                id="sealing_fault",
                orientation="x",
                fault_index=7,
                slope=0.05,
                transmissibility_scale=1e-5,  # Nearly sealing
                fault_permeability=0.01,
                geometric_throw_cells=1,
                preserve_grid_data=True,
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
            connate_water_saturation_grid=connate_water_saturation_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            net_to_gross_ratio_grid=net_to_gross_grid,
            boundary_conditions=boundary_conditions,
            # faults=fault_network,
            relative_permeability_func=relative_permeability_func,
            capillary_pressure_params=capillary_pressure_params,
            reservoir_gas_name="Methane",
        )

        # Water injection well (realistic waterflood scenario)
        water_injector = sim3D.injection_well(
            well_name="Water Injector 1",
            perforating_intervals=[
                ((3, 3, 2), (3, 3, 3))  # Perforated in high perm layers
            ],
            radius=0.3542,  # 8.5 inch wellbore
            bottom_hole_pressure=layer_pressures[2] + 500,  # 500 psi above formation
            injected_fluid=sim3D.WellFluid(
                name="Water",
                phase=sim3D.FluidPhase.WATER,
                specific_gravity=1.05,  # Slightly saline
                molecular_weight=18.015,
                compressibility=3.0e-6,
                salinity=30_000,
            ),
        )

        # Producer well
        producer = sim3D.production_well(
            well_name="Producer 1",
            perforating_intervals=[
                ((7, 7, 2), (7, 7, 3))  # Multiple layers
            ],
            radius=0.3542,
            bottom_hole_pressure=1200.0,  # ~500 psi drawdown
            produced_fluids=(
                sim3D.WellFluid(
                    name="Oil",
                    phase=sim3D.FluidPhase.OIL,
                    specific_gravity=0.845,
                    molecular_weight=180.0,
                ),
                sim3D.WellFluid(
                    name="Gas",
                    phase=sim3D.FluidPhase.GAS,
                    specific_gravity=0.65,
                    molecular_weight=20.0,
                ),
                sim3D.WellFluid(
                    name="Water",
                    phase=sim3D.FluidPhase.WATER,
                    specific_gravity=1.05,
                    molecular_weight=18.015,
                    salinity=30_000,
                ),
            ),
            skin_factor=2.5,
        )
        # producer.update_schedule(
        #     sim3D.WellEvent(
        #         hook=sim3D.well_time_hook(time_step=4),
        #         action=sim3D.well_props_action(bottom_hole_pressure=200),
        #     )
        # )

        wells = sim3D.wells(injectors=[], producers=[producer])

        # Realistic simulation options
        options = sim3D.Options(
            time_step_size=3600 * 20,  # 1 day
            total_time=3600 * 24 * 600,
            max_iterations=5000,
            convergence_tolerance=1e-4,
            output_frequency=1,  # Monthly output
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
    analyst = sim3D.ModelAnalyst(model_states)
    oil_production_history = analyst.oil_production_history(
        interval=3, cumulative=False, from_time_step=1
    )
    water_injection_history = analyst.water_injection_history(
        interval=3, cumulative=False, from_time_step=1
    )
    gas_production_history = analyst.gas_production_history(
        interval=3, cumulative=False, from_time_step=1
    )
    oil_in_place_history = analyst.oil_in_place_history(interval=1)
    gas_in_place_history = analyst.gas_in_place_history(interval=1)
    water_in_place_history = analyst.water_in_place_history(interval=1)
    ipr_method = analyst.recommend_ipr_method()
    productivity_history = analyst.productivity_history(
        ipr_method=ipr_method, interval=1
    )

    oil_saturation_history = []
    water_saturation_history = []
    gas_saturation_history = []
    ipr_data = []
    productivity_data = []

    for i in range(0, len(model_states), 1):
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
            # "Water Injection": np.array(list(water_injection_history)),
            # "Gas Production": np.array(list(gas_production_history)),
            "Avg. Oil Saturation": np.array(oil_saturation_history),
            "Avg. Water Saturation": np.array(water_saturation_history),
            "Avg. Gas Saturation": np.array(gas_saturation_history),
            # "IPR": np.array(ipr_data),
            # "Oil In Place": np.array(list(oil_in_place_history)),
            # "Water In Place": np.array(list(water_in_place_history)),
            # "Gas In Place": np.array(list(gas_in_place_history)),
            # "Productivity Index": np.array(productivity_data),
        },
        title="Model Analyses",
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
    data = model_states[360]
    shared_kwargs = dict(
        plot_type=sim3D.plotly3d.PlotType.ISOSURFACE,
        width=960,
        height=600,
        # isomin=0.13,
        # cmin=600,
        # cmax=2700,
        opacity=0.6,
        use_opacity_scaling=False,
        # subsampling_factor=2,
        # downsampling_factor=1,
        # x_slice=(6, 13),
        # y_slice=(6, 9),
        # z_slice=(0, 4),
        labels=labels,
        aspect_mode="cube",
        # marker_size=12,
        # notebook=True,
    )
    # Oil Plot
    oil_fig = sim3D.plotly3d.viz.make_plot(data, property="oil-sat", **shared_kwargs)
    # Water Plot
    water_fig = sim3D.plotly3d.viz.make_plot(
        data, property="water-sat", **shared_kwargs
    )
    # Gas Plot
    gas_fig = sim3D.plotly3d.viz.make_plot(data, property="gas-sat", **shared_kwargs)
    fig = sim3D.merge_plots(
        oil_fig, water_fig, gas_fig, cols=1, rows=3, width=1080, height=1080
    )
    fig.show()
    return


if __name__ == "__main__":
    app.run()
