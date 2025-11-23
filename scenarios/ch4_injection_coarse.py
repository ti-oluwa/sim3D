import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full", app_title="SIM3D")


@app.cell
def _():
    import logging
    from pathlib import Path

    import numpy as np
    import sim3D

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    sim3D.use_32bit_precision()
    sim3D.image_config(scale=3)

    DEPLETED_MODEL_STATE = (
        Path.cwd() / "scenarios/states/primary_depleted_coarse.pkl.xz"
    )

    def main():
        # Load last model state of primary depletion
        state = sim3D.ModelState.load(filepath=DEPLETED_MODEL_STATE)
        model = state.model
        del state

        minimum_miscibility_pressure = 4000.0
        omega = 0.33

        # Gas injection wells, 3-spot pattern
        injection_clamp = sim3D.InjectionClamp()
        control = sim3D.AdaptiveBHPRateControl(
            target_rate=1_000_000,
            target_phase="gas",
            minimum_bottom_hole_pressure=1500,
            clamp=injection_clamp,
        )
        gas_injector_1 = sim3D.injection_well(
            well_name="GI-1",
            perforating_intervals=[((16, 3, 1), (16, 3, 3))],
            radius=0.3542,  # 8.5 inch wellbore
            control=control,
            injected_fluid=sim3D.InjectedFluid(
                name="Methane",
                phase=sim3D.FluidPhase.GAS,
                specific_gravity=0.65,
                molecular_weight=16.04,
                minimum_miscibility_pressure=minimum_miscibility_pressure,
                todd_longstaff_omega=omega,
                is_miscible=True,
            ),
            is_active=True,
            skin_factor=2.0,
        )

        # Create other gas wells as duplicates
        gas_injector_2 = gas_injector_1.duplicate(
            name="GI-2", perforating_intervals=[((16, 16, 1), (16, 16, 3))]
        )
        injectors = [gas_injector_1, gas_injector_2]

        # Production well
        production_clamp = sim3D.ProductionClamp()
        control = sim3D.MultiPhaseRateControl(
            oil_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-100,
                target_phase="oil",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
            gas_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-500,
                target_phase="gas",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
            water_control=sim3D.AdaptiveBHPRateControl(
                target_rate=-10,
                target_phase="water",
                minimum_bottom_hole_pressure=1200,
                clamp=production_clamp,
            ),
        )
        producer = sim3D.production_well(
            well_name="P-1",
            perforating_intervals=[((14, 10, 3), (14, 10, 4))],
            radius=0.3542,
            control=control,
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
                    molecular_weight=16.04,
                ),
                sim3D.ProducedFluid(
                    name="Water",
                    phase=sim3D.FluidPhase.WATER,
                    specific_gravity=1.05,
                    molecular_weight=18.015,
                ),
            ),
            skin_factor=2.5,
            is_active=False,
        )
        producer.schedule_event(
            sim3D.WellEvent(
                hook=sim3D.well_time_hook(time_step=100),
                action=sim3D.well_update_action(is_active=True),
            )
        )
        producers = [producer]

        wells = sim3D.wells_(injectors=injectors, producers=producers)
        options = sim3D.Options(
            scheme="impes",
            total_time=sim3D.Time(days=sim3D.c.DAYS_PER_YEAR * 6),  # 6 years
            time_step_size=sim3D.Time(hours=30),
            max_time_steps=2000,
            output_frequency=1,
            miscibility_model="todd_longstaff",
            use_pseudo_pressure=True,
        )
        states = sim3D.run(model=model, wells=wells, options=options)
        return list(states)

    return Path, main, np, sim3D


@app.cell
def _(main):
    states = main()
    return (states,)


@app.cell
def _(Path, sim3D, states):
    sim3D.dump_states(
        states,
        filepath=Path.cwd() / "scenarios/states/ch4_injection_coarse.pkl",
        exist_ok=True,
        compression="lzma",
    )
    return


@app.cell
def _(model_states, np, sim3D):
    analyst = sim3D.ProductionAnalyst(model_states)
    oil_production_history = analyst.oil_production_history(
        interval=1, cumulative=False, from_time_step=1
    )
    water_production_history = analyst.water_production_history(
        interval=1, cumulative=False, from_time_step=1
    )
    gas_production_history = analyst.free_gas_production_history(
        interval=1, cumulative=False, from_time_step=1
    )
    gas_injection_history = analyst.gas_injection_history(
        interval=1, cumulative=False, from_time_step=1
    )
    cumulative_oil_production_history = analyst.oil_production_history(
        interval=1, cumulative=True, from_time_step=1
    )
    cumulative_water_production_history = analyst.water_production_history(
        interval=1, cumulative=True, from_time_step=1
    )
    cumulative_gas_production_history = analyst.free_gas_production_history(
        interval=1, cumulative=True, from_time_step=1
    )
    cumulative_gas_injection_history = analyst.gas_injection_history(
        interval=1, cumulative=True, from_time_step=1
    )
    oil_in_place_history = analyst.oil_in_place_history(interval=1, from_time_step=1)
    gas_in_place_history = analyst.gas_in_place_history(interval=1, from_time_step=1)
    water_in_place_history = analyst.water_in_place_history(
        interval=1, from_time_step=1
    )
    sweep_efficiency_history = analyst.sweep_efficiency_history(
        interval=1, from_time_step=1
    )
    oil_recovery_factor_history = analyst.oil_recovery_factor_history(
        interval=1, from_time_step=1
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

    for i in range(1, len(model_states), 1):
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

    # Pressure
    pressure_fig = sim3D.make_series_plot(
        data={"Avg. Reservoir Pressure": np.array(avg_pressure_history)},
        title="Pressure Analysis",
        x_label="Time Step",
        y_label="Avg. Pressure (psia)",
        marker_sizes=4,
    )
    # Saturation
    saturation_fig = sim3D.make_series_plot(
        data={
            "Avg. Oil Saturation": np.array(oil_saturation_history),
            "Avg. Water Saturation": np.array(water_saturation_history),
            "Avg. Gas Saturation": np.array(gas_saturation_history),
        },
        title="Saturation Analysis",
        x_label="Time Step",
        y_label="Saturation",
        marker_sizes=4,
    )
    # Production & Injection
    oil_production_fig = sim3D.make_series_plot(
        data={
            "Oil Production": np.array(list(oil_production_history)),
        },
        title="Oil Production Analysis",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=4,
    )
    water_production_fig = sim3D.make_series_plot(
        data={
            "Water Production": np.array(list(water_production_history)),
        },
        title="Water Production Analysis",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=4,
    )
    gas_production_fig = sim3D.make_series_plot(
        data={
            "Gas Production": np.array(list(gas_production_history)),
        },
        title="Gas Production Analysis",
        x_label="Time Step",
        y_label="Production (SCF)",
        marker_sizes=4,
    )
    gas_injection_fig = sim3D.make_series_plot(
        data={
            # "Water Injection": np.array(list(water_injection_history)),
            "Gas Injection": np.array(list(gas_injection_history)),
        },
        title="Gas Injection Analysis",
        x_label="Time Step",
        y_label="Injection (SCF)",
        marker_sizes=4,
    )
    # Cumulative production & injection
    cumulative_oil_production_fig = sim3D.make_series_plot(
        data={
            "Cumulative Oil Production": np.array(
                list(cumulative_oil_production_history)
            ),
        },
        title="Cumulative Oil Production Analysis",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=4,
    )
    cumulative_water_production_fig = sim3D.make_series_plot(
        data={
            "Cumulative Water Production": np.array(
                list(cumulative_water_production_history)
            ),
        },
        title="Cumulative Water Production Analysis",
        x_label="Time Step",
        y_label="Production (STB)",
        marker_sizes=4,
    )
    cumulative_gas_production_fig = sim3D.make_series_plot(
        data={
            "Cumulative Gas Production": np.array(
                list(cumulative_gas_production_history)
            ),
        },
        title="Cumulative Gas Production Analysis",
        x_label="Time Step",
        y_label="Production (SCF)",
        marker_sizes=4,
    )
    cumulative_gas_injection_fig = sim3D.make_series_plot(
        data={
            "Cumulative Gas Injection": np.array(
                list(cumulative_gas_injection_history)
            ),
        },
        title="Cumulative Gas Injection Analysis",
        x_label="Time Step",
        y_label="Injection (SCF)",
        marker_sizes=4,
    )
    # Capillary Pressure
    capillary_pressure_fig = sim3D.make_series_plot(
        data={
            "Pcow": np.array(oil_water_capillary_pressure_history),
            "Pcgo": np.array(gas_oil_capillary_pressure_history),
        },
        title="Capillary Pressure Analysis",
        x_label="Time Step",
        y_label="Avg. Capillary Pressure",
        marker_sizes=4,
    )
    # Rel Perm
    relperm_fig = sim3D.make_series_plot(
        data={
            "Krw": np.array(krw_history),
            "Kro": np.array(kro_history),
            "Krg": np.array(krg_history),
        },
        title="RelPerm Analysis",
        x_label="Time Step",
        y_label="Avg. Relative Permeability",
        marker_sizes=4,
    )
    # RelPerm-Saturation
    water_relperm_saturation_fig = sim3D.make_series_plot(
        data={
            "Krw/Sw": np.array(krw_saturation_history),
        },
        title="Water RelPerm-Saturation Analysis",
        x_label="Avg. Water Saturation",
        y_label="Avg. Relative Permeability",
        marker_sizes=4,
    )
    oil_relperm_saturation_fig = sim3D.make_series_plot(
        data={
            "Kro/So": np.array(kro_saturation_history),
        },
        title="Oil RelPerm-Saturation Analysis",
        x_label="Avg. Oil Saturation",
        y_label="Avg. Relative Permeability",
        marker_sizes=4,
    )
    gas_relperm_saturation_fig = sim3D.make_series_plot(
        data={
            "Krg/Sg": np.array(krg_saturation_history),
        },
        title="Gas RelPerm-Saturation Analysis",
        x_label="Avg. Gas Saturation",
        y_label="Avg. Relative Permeability",
        marker_sizes=4,
    )
    # Reserves
    oil_water_reserves_fig = sim3D.make_series_plot(
        data={
            "Oil In Place": np.array(list(oil_in_place_history)),
            "Water In Place": np.array(list(water_in_place_history)),
        },
        title="Oil & Water Reserves Analysis",
        x_label="Time Step",
        y_label="OIP/WIP (STB)",
        marker_sizes=4,
    )
    gas_reserve_fig = sim3D.make_series_plot(
        data={
            "Gas In Place": np.array(list(gas_in_place_history)),
        },
        title="Gas Reserve Analysis",
        x_label="Time Step",
        y_label="GIP (SCF)",
        marker_sizes=4,
    )
    # Recovery
    recovery_efficiency_fig = sim3D.make_series_plot(
        data={
            "Recovery Efficiency": np.array(recovery_efficiency_history),
        },
        title="Recovery Efficiency Analysis",
        x_label="Time Step",
        marker_sizes=4,
    )
    recovery_factor_fig = sim3D.make_series_plot(
        data={
            "Oil Recovery Factor": np.array(list(oil_recovery_factor_history)),
        },
        title="Recovery Factor Analysis",
        x_label="Time Step",
        marker_sizes=4,
    )

    # Merge all Plots
    analyses_plots = sim3D.merge_plots(
        pressure_fig,
        saturation_fig,
        oil_production_fig,
        water_production_fig,
        gas_production_fig,
        gas_injection_fig,
        cumulative_oil_production_fig,
        cumulative_water_production_fig,
        cumulative_gas_production_fig,
        cumulative_gas_injection_fig,
        relperm_fig,
        oil_relperm_saturation_fig,
        water_relperm_saturation_fig,
        gas_relperm_saturation_fig,
        capillary_pressure_fig,
        oil_water_reserves_fig,
        gas_reserve_fig,
        recovery_efficiency_fig,
        recovery_factor_fig,
        cols=2,
        hovermode="x unified",
    )
    analyses_plots.show()
    return (analyst,)


@app.cell
def _(analyst, np, sim3D):
    recommended_model, results = analyst.recommend_decline_model(phase="oil")
    print("Recommended Decline Model: ", recommended_model)
    decline_curve = results[recommended_model]
    if error := decline_curve.error:
        print(error)
    else:
        production_forecast = analyst.forecast_production(
            decline_result=decline_curve, time_steps=100
        )
        production_forecast_fig = sim3D.make_series_plot(
            data={
                "Production Rate": np.array(production_forecast),
            },
            title="Production Forecast",
            x_label="Time Step",
            y_label="Production Rate (STB/day)",
            marker_sizes=4,
        )
        production_forecast_fig.show()
    return


@app.cell
def _(sim3D, states):
    wells = states[0].wells
    injector_locations, producer_locations = wells.locations
    injector_names, producer_names = wells.names
    well_positions = [*injector_locations, *producer_locations]
    well_names = [*injector_names, *producer_names]
    labels = sim3D.plotly3d.Labels()
    labels.add_well_labels(well_positions, well_names)

    shared_kwargs = dict(
        plot_type="scatter_3d",
        width=960,
        height=960,
        opacity=0.7,
        labels=labels,
        aspect_mode="cube",
        z_scale=1.1,
        show_wells=True,
        show_surface_marker=True,
        show_perforations=True,
    )
    viz = sim3D.plotly3d.DataVisualizer()

    property = "oil-saturation"
    figures = []
    timesteps = [86]
    for timestep in timesteps:
        fig = viz.make_plot(
            states[timestep],
            property=property,
            title=f"{property.strip('-').title()} Profile at Timestep {timestep}",
            **shared_kwargs,
        )
        figures.append(fig)

    if len(figures) > 1:
        plots = sim3D.merge_plots(*figures, cols=2)
        plots.show()
    else:
        figures[0].show()
    return


if __name__ == "__main__":
    app.run()
