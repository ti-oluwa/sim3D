import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full", app_title="SIM3D")


@app.cell
def _():
    import logging
    import typing
    from rich import print as pprint

    import numpy as np

    import sim3D

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore


    def main():
        # Grid dimensions - typical field scale
        cell_dimension = (500.0, 500.0)  # 500ft x 500ft cells
        grid_shape = typing.cast(
            sim3D.ThreeDimensions,
            (15, 15, 5),  # 15x15 cells, 5 layers
        )

        # Thickness distribution - typical reservoir layers
        # Thicker in the middle, thinner at top/bottom
        thickness_values = np.array([25.0, 40.0, 50.0, 35.0, 20.0])  # feet
        thickness_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=thickness_values,
            orientation=sim3D.Orientation.Z,
        )

        # Pressure gradient: ~0.433 psi/ft for water, slightly less for oil
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
            layer_values=layer_pressures
            - 200.0,  # 400 psi below formation pressure
            orientation=sim3D.Orientation.Z,
        )

        # Saturation endpoints - typical for sandstone reservoirs
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
            value=0.15,  # Swi
        )
        connate_water_saturation_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.12,  # Slightly less than Swi
        )
        residual_gas_saturation_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.045,  # Sgr
        )

        # Porosity - decreasing with depth (compaction trend)
        porosity_values = np.linspace(0.28, 0.18, grid_shape[2])  # 28% to 18%
        porosity_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=porosity_values,
            orientation=sim3D.Orientation.Z,
        )

        # Fluid contacts
        # GOC at 8050 ft, OWC at 8150 ft (realistic spacing)
        goc_depth = 8020.0
        owc_depth = 8120.0

        depth_grid = sim3D.depth_grid(thickness_grid)
        water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
            sim3D.build_saturation_grids(
                depth_grid=depth_grid,
                gas_oil_contact=goc_depth - reservoir_top_depth,  # 50 ft below top
                oil_water_contact=owc_depth
                - reservoir_top_depth,  # 150 ft below top
                connate_water_saturation_grid=connate_water_saturation_grid,
                residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
                residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
                residual_gas_saturation_grid=residual_gas_saturation_grid,
                porosity_grid=porosity_grid,
                use_transition_zones=True,
                oil_water_transition_thickness=12.0,  # transition zone
                gas_oil_transition_thickness=8.0,
                transition_curvature_exponent=1.5,
            )
        )

        # Oil viscosity - increases slightly with depth (heavier oil)
        oil_viscosity_values = np.linspace(1.2, 2.5, grid_shape[2])  # cP
        oil_viscosity_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=oil_viscosity_values,
            orientation=sim3D.Orientation.Z,
        )

        # Oil compressibility - typical range for crude oil
        oil_compressibility_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=1.2e-5,  # 1/psi
        )

        # Oil specific gravity (API ~35-40 degrees)
        oil_specific_gravity_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.845,  # ~36 API
        )

        # Permeability distribution
        # Higher permeability in middle layers (better reservoir quality)
        # Anisotropy ratio kv/kh ~ 0.1 (typical for layered sandstone)
        x_perm_values = np.array([18.0, 10.0, 15.0, 18.0, 16.0])  # mD
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

        # RelPerm table
        relative_permeability_table = sim3D.BrooksCoreyThreePhaseRelPermModel(
            irreducible_water_saturation=0.15,
            residual_oil_saturation_gas=0.15,
            residual_oil_saturation_water=0.25,
            residual_gas_saturation=0.045,
            wettability=sim3D.WettabilityType.WATER_WET,
            mixing_rule=sim3D.eclipse_rule,
        )

        # Capillary pressure table
        capillary_pressure_table = sim3D.BrooksCoreyCapillaryPressureModel(
            oil_water_entry_pressure_water_wet=2.0,
            gas_oil_entry_pressure=2.8,
            wettability=sim3D.Wettability.WATER_WET,
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
        # Rock compressibility for sandstone
        rock_compressibility = 4.5e-6  # 1/psi
        # Net-to-gross ratio (accounting for shale layers)
        net_to_gross_grid = sim3D.uniform_grid(grid_shape=grid_shape, value=0.85)

        # Boundary conditions - aquifer support from bottom
        boundary_conditions = sim3D.BoundaryConditions(
            conditions={
                "pressure": sim3D.GridBoundaryCondition(
                    bottom=sim3D.ConstantBoundary(3500),  # Aquifer pressure
                ),
            }
        )

        # Natural gas (associated gas) properties
        gas_gravity = 0.65  # Typical for associated gas
        gas_gravity_grid = sim3D.uniform_grid(
            grid_shape=grid_shape, value=gas_gravity
        )

        # Realistic fault network - simplified
        fault_network = [
            sim3D.Fault(
                id="sealing_fault",
                orientation="x",
                fault_index=7,
                slope=0.4,
                transmissibility_scale=1e-3,  # Nearly sealing
                fault_permeability=0.01,
                geometric_throw_cells=1,
                conductive=True,
                preserve_grid_data=False,
                property_defaults=sim3D.FaultPropertyDefaults(),
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
            relative_permeability_table=relative_permeability_table,
            capillary_pressure_table=capillary_pressure_table,
            reservoir_gas="Methane",
        )

        # Water injection well
        water_injector = sim3D.injection_well(
            well_name="WI-1",
            perforating_intervals=[
                ((3, 3, 3), (3, 3, 4))  # Inject in acquifer zone
            ],
            radius=0.3542,  # 8.5 inch wellbore
            bottom_hole_pressure=3500,
            injected_fluid=sim3D.InjectedFluid(
                name="Water",
                phase=sim3D.FluidPhase.WATER,
                specific_gravity=1.05,
                molecular_weight=18.015,
                salinity=30_000,
            ),
            is_active=False,
        )
        gas_injector = sim3D.injection_well(
            well_name="GI-1",
            perforating_intervals=[
                ((7, 3, 2), (7, 3, 3))  # Inject in gas cap zone
            ],
            radius=0.3542,  # 8.5 inch wellbore
            bottom_hole_pressure=3600,
            injected_fluid=sim3D.InjectedFluid(
                name="CO2",
                phase=sim3D.FluidPhase.GAS,
                specific_gravity=0.65,
                molecular_weight=44.0,
                minimum_miscibility_pressure=2500,
                is_miscible=True,
            ),
            is_active=True,
        )
        gas_injector.schedule_event(
            sim3D.WellEvent(
                hook=sim3D.well_time_hook(time_step=200),
                action=sim3D.well_update_action(is_active=True),  # Activate well
            )
        )
        injectors = [gas_injector]

        # Producer well
        producer = sim3D.production_well(
            well_name="P-1",
            perforating_intervals=[
                ((7, 11, 3), (7, 11, 3))  # Multiple layers
            ],
            radius=0.3542,
            bottom_hole_pressure=2700.0,
            produced_fluids=(
                sim3D.ProducedFluid(
                    name="Oil",
                    phase=sim3D.FluidPhase.OIL,
                    specific_gravity=0.845,
                    molecular_weight=180.0,
                ),
                sim3D.ProducedFluid(
                    name="Gas",
                    phase=sim3D.FluidPhase.GAS,
                    specific_gravity=0.65,
                    molecular_weight=44.0,
                ),
                sim3D.ProducedFluid(
                    name="Water",
                    phase=sim3D.FluidPhase.WATER,
                    specific_gravity=1.05,
                    molecular_weight=18.015,
                ),
            ),
            skin_factor=2.5,
        )
        # producer.schedule_event(
        #     sim3D.WellEvent(
        #         hook=sim3D.well_time_hook(time_step=800),
        #         action=sim3D.update_well_action(bottom_hole_pressure=1450),
        #     )
        # )
        # producer.schedule_event(
        #     sim3D.WellEvent(
        #         hook=sim3D.well_time_hook(time_step=1200),
        #         action=sim3D.update_well_action(bottom_hole_pressure=800),
        #     )
        # )

        producers = [producer]
        wells = sim3D.wells(injectors=injectors, producers=producers)

        options = sim3D.Options(
            scheme="impes",
            total_time=sim3D.Time(days=sim3D.c.DAYS_PER_YEAR * 4),
            time_step_size=sim3D.Time(hours=24),
            max_time_steps=400,
            output_frequency=1,
            miscibility_model="todd_longstaff",
            use_pseudo_pressure=True,
        )
        model_states = sim3D.run(model=model, wells=wells, options=options)
        return list(model_states)
    return main, np, pprint, sim3D


@app.cell
def _(main):
    model_states = main()
    return (model_states,)


@app.cell
def _(model_states, np, sim3D):
    analyst = sim3D.ProductionAnalyst(model_states)
    oil_production_history = analyst.oil_production_history(
        interval=5, cumulative=False, from_time_step=7
    )
    water_production_history = analyst.water_production_history(
        interval=5, cumulative=False, from_time_step=7
    )
    gas_production_history = analyst.free_gas_production_history(
        interval=5, cumulative=False, from_time_step=7
    )
    water_injection_history = analyst.water_injection_history(
        interval=5, cumulative=False, from_time_step=7
    )
    gas_injection_history = analyst.gas_injection_history(
        interval=5, cumulative=False, from_time_step=7
    )
    oil_in_place_history = analyst.oil_in_place_history(
        interval=5, from_time_step=7
    )
    gas_in_place_history = analyst.gas_in_place_history(
        interval=5, from_time_step=7
    )
    water_in_place_history = analyst.water_in_place_history(
        interval=5, from_time_step=7
    )
    sweep_efficiency_history = analyst.sweep_efficiency_history(
        interval=5, from_time_step=7
    )


    oil_saturation_history = []
    water_saturation_history = []
    gas_saturation_history = []
    avg_pressure_history = []
    oil_water_capillary_pressure_history = []
    gas_oil_capillary_pressure_history = []
    krw_history = []
    kro_history = []
    krg_history = []
    krw_saturation_history = []
    kro_saturation_history = []
    krg_saturation_history = []
    volumetric_sweep_efficiency_history = []
    displacement_efficiency_history = []
    recovery_efficiency_history = []


    for i in range(7, len(model_states), 5):
        state = model_states[i]
        model = state.model
        time_step = state.time_step
        avg_oil_sat = np.mean(model.fluid_properties.oil_saturation_grid)
        avg_water_sat = np.mean(model.fluid_properties.water_saturation_grid)
        avg_gas_sat = np.mean(model.fluid_properties.gas_saturation_grid)
        avg_pressure = np.mean(model.fluid_properties.pressure_grid)
        avg_pcow = np.mean(state.capillary_pressures.oil_water_capillary_pressure)
        avg_pcgo = np.mean(state.capillary_pressures.gas_oil_capillary_pressure)
        avg_krw = np.mean(state.relative_permeabilities.krw)
        avg_kro = np.mean(state.relative_permeabilities.kro)
        avg_krg = np.mean(state.relative_permeabilities.krg)
        oil_saturation_history.append((time_step, avg_oil_sat))
        water_saturation_history.append((time_step, avg_water_sat))
        gas_saturation_history.append((time_step, avg_gas_sat))
        avg_pressure_history.append((time_step, avg_pressure))
        oil_water_capillary_pressure_history.append((time_step, avg_pcow))
        gas_oil_capillary_pressure_history.append((time_step, avg_pcgo))
        krw_history.append((time_step, avg_krw))
        kro_history.append((time_step, avg_kro))
        krg_history.append((time_step, avg_krg))
        krg_saturation_history.append((avg_krg, avg_gas_sat))
        kro_saturation_history.append((avg_kro, avg_oil_sat))
        krw_saturation_history.append((avg_krw, avg_water_sat))

    for time_step, result in sweep_efficiency_history:
        volumetric_sweep_efficiency_history.append(
            (time_step, result.volumetric_sweep_efficiency)
        )
        displacement_efficiency_history.append(
            (time_step, result.displacement_efficiency)
        )
        recovery_efficiency_history.append((time_step, result.recovery_efficiency))


    # Production & Injection
    oil_production_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Oil Production": np.array(list(oil_production_history)),
        },
        title="Oil Production Analysis",
        x_label="Time Step",
        y_label="Production (STB/day)",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    water_production_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Water Production": np.array(list(water_production_history)),
        },
        title="Water Production Analysis",
        x_label="Time Step",
        y_label="Production (STB/day)",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    gas_production_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Gas Production": np.array(list(gas_production_history)),
        },
        title="Gas Production Analysis",
        x_label="Time Step",
        y_label="Production (SCF/day)",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    gas_injection_fig = sim3D.plotly2d.make_series_plot(
        data={
            # "Water Injection": np.array(list(water_injection_history)),
            "Gas Injection": np.array(list(gas_injection_history)),
        },
        title="Gas Injection Analysis",
        x_label="Time Step",
        y_label="Injection (SCF/day)",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    # Saturation
    saturation_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Avg. Oil Saturation": np.array(oil_saturation_history),
            "Avg. Water Saturation": np.array(water_saturation_history),
            "Avg. Gas Saturation": np.array(gas_saturation_history),
        },
        title="Saturation Analysis",
        x_label="Time Step",
        y_label="Saturation",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    # Capillary Pressure
    capillary_pressure_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Pcow": np.array(oil_water_capillary_pressure_history),
            "Pcgo": np.array(gas_oil_capillary_pressure_history),
        },
        title="Capillary Pressure Analysis",
        x_label="Time Step",
        y_label="Capillary Pressure",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    # Rel Perm
    relperm_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Krw": np.array(krw_history),
            "Kro": np.array(kro_history),
            "Krg": np.array(krg_history),
        },
        title="RelPerm Analysis",
        x_label="Time Step",
        y_label="Relative Permeability",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    # RelPerm-Saturation
    water_relperm_saturation_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Krw/Sw": np.array(krg_saturation_history),
        },
        title="Water RelPerm-Saturation Analysis",
        x_label="Water Saturation",
        y_label="Relative Permeability",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    oil_relperm_saturation_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Kro/So": np.array(kro_saturation_history),
        },
        title="Oil RelPerm-Saturation Analysis",
        x_label="Oil Saturation",
        y_label="Relative Permeability",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    gas_relperm_saturation_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Krg/Sg": np.array(krg_saturation_history),
        },
        title="Gas RelPerm-Saturation Analysis",
        x_label="Gas Saturation",
        y_label="Relative Permeability",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    # Pressure
    pressure_fig = sim3D.plotly2d.make_series_plot(
        data={"Avg. Reservoir Pressure": np.array(avg_pressure_history)},
        title="Pressure Analysis",
        x_label="Time Step",
        y_label="Avg. Pressure (psia)",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    # Reserves
    reserves_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Oil In Place": np.array(list(oil_in_place_history)),
            "Water In Place": np.array(list(water_in_place_history)),
            # "Gas In Place": np.array(list(gas_in_place_history)),
        },
        title="Reserves Analysis",
        x_label="Time Step",
        y_label="HCIP (STB or SCF)",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    # Sweep Efficiency
    vol_sweep_efficiency_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Volumetric Sweep Efficiency": np.array(
                volumetric_sweep_efficiency_history
            ),
        },
        title="Volumetric Sweep Efficiency Analysis",
        x_label="Time Step",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    displacement_efficiency_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Displacement Efficiency": np.array(displacement_efficiency_history),
        },
        title="Displacement Efficiency Analysis",
        x_label="Time Step",
        width=1080,
        height=580,
        marker_sizes=4,
    )
    recovery_efficiency_fig = sim3D.plotly2d.make_series_plot(
        data={
            "Recovery Efficiency": np.array(recovery_efficiency_history),
        },
        title="Recovery Efficiency Analysis",
        x_label="Time Step",
        width=1080,
        height=580,
        marker_sizes=4,
    )

    # Merge all Plots
    analyses_fig = sim3D.merge_plots(
        oil_production_fig,
        water_production_fig,
        gas_production_fig,
        gas_injection_fig,
        saturation_fig,
        relperm_fig,
        oil_relperm_saturation_fig,
        water_relperm_saturation_fig,
        gas_relperm_saturation_fig,
        capillary_pressure_fig,
        pressure_fig,
        reserves_fig,
        vol_sweep_efficiency_fig,
        displacement_efficiency_fig,
        recovery_efficiency_fig,
        cols=2,
        hovermode="x unified",
    )
    analyses_fig.show()
    return (analyst,)


@app.cell
def _(analyst, np, pprint, sim3D):
    recommended_model, results = analyst.recommend_decline_model(phase="oil")
    pprint("Recommended Decline Model: ", recommended_model)
    decline_curve = results[recommended_model]
    if error := decline_curve.error:
        print(error)
    else:
        production_forecast = analyst.forecast_production(
            decline_result=decline_curve, time_steps=100
        )
        production_forecast_fig = sim3D.plotly2d.make_series_plot(
            data={
                "Production": np.array(production_forecast),
            },
            title="Production Forecast",
            x_label="Time Step",
            y_label="Production (STB/day or SCF/day)",
            width=1080,
            height=580,
            marker_sizes=4,
        )
        production_forecast_fig.show()
    return


@app.cell
def _(model_states, sim3D):
    wells = model_states[0].wells
    injector_locations, producer_locations = wells.locations
    injector_names, producer_names = wells.names
    well_positions = [*injector_locations, *producer_locations]
    well_names = [*injector_names, *producer_names]
    labels = sim3D.plotly3d.Labels()
    # labels.add_well_labels(well_positions, well_names)
    data = model_states[300]
    shared_kwargs = dict(
        plot_type="volume_render",
        width=960,
        height=960,
        # isomin=0.13,
        # cmin=600,
        # cmax=2700,
        opacity=0.6,
        surface_count=100,
        # subsampling_factor=2,
        # downsampling_factor=1,
        # x_slice=(6, 13),
        # y_slice=(6, 9),
        # z_slice=(2, 5),
        labels=labels,
        aspect_mode="data",
        # marker_size=12,
        # notebook=True,
        z_scale=3,
        show_wells=True,
        show_surface_marker=True,
        show_perforations=False,
    )

    # Animation
    # sim3D.plotly3d.viz.animate(
    #     model_states, property="oil-sat", **shared_kwargs, z_slice=(2, 4)
    # )

    # Oil Plot
    oil_pressure_fig = sim3D.plotly3d.viz.make_plot(data, property="pressure", **shared_kwargs)
    oil_sat_fig = sim3D.plotly3d.viz.make_plot(data, property="gas-sat", **shared_kwargs)

    # # Water Plot
    # water_fig = sim3D.plotly3d.viz.make_plot(
    #     data, property="water-sat", **shared_kwargs
    # )
    # # Gas Plot
    # gas_fig = sim3D.plotly3d.viz.make_plot(
    #     data, property="gas-sat", **shared_kwargs
    # )
    # viscosity_fig = sim3D.plotly3d.viz.make_plot(
    #     data, property="solvent_conc", **shared_kwargs
    # )
    fig = sim3D.merge_plots(
        oil_pressure_fig,
        oil_sat_fig,
        cols=1,
        width=1080,
        height=2400,
    )
    fig.show()
    return


if __name__ == "__main__":
    app.run()
