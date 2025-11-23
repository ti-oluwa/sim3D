import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full", app_title="SIM3D")


@app.cell
def _():
    import logging
    import typing

    import numpy as np
    import sim3D

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    sim3D.use_32bit_precision()
    sim3D.image_config(scale=3)

    def main():
        # Grid dimensions - typical field scale
        cell_dimension = (50.0, 50.0)  # 100ft x 100ft cells
        grid_shape = typing.cast(
            sim3D.ThreeDimensions,
            (40, 40, 10),  # 40x40 cells, 10 layers
        )
        dip_angle = 5.0
        dip_azimuth = 90.0

        # Thickness distribution - typical reservoir layers
        # Thicker in the middle, thinner at top/bottom
        thickness_values = np.array(
            [30.0, 20.0, 25.0, 30.0, 25.0, 30.0, 20.0, 25.0, 30.0, 25.0]
        )  # feet
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
            layer_values=layer_pressures - 400.0,  # 400 psi below formation pressure
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
        porosity_values = np.array(
            [0.04, 0.07, 0.09, 0.1, 0.08, 0.12, 0.14, 0.16, 0.11, 0.08]
        )  # fraction
        porosity_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=porosity_values,
            orientation=sim3D.Orientation.Z,
        )

        # Fluid contacts
        # GOC at 8060 ft, OWC at 8220 ft
        goc_depth = 8060.0
        owc_depth = 8220.0

        depth_grid = sim3D.depth_grid(thickness_grid)
        # Apply structural dip
        depth_grid = sim3D.apply_structural_dip(
            elevation_grid=depth_grid,
            elevation_direction="downward",
            cell_dimension=cell_dimension,
            dip_angle=dip_angle,
            dip_azimuth=dip_azimuth,
        )
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
                oil_water_transition_thickness=12.0,  # transition zone
                gas_oil_transition_thickness=8.0,
                transition_curvature_exponent=1.2,
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
        x_perm_values = np.array([12, 25, 40, 18, 55, 70, 90, 35, 48, 22])  # mD
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
            water_exponent=2.0,
            oil_exponent=2.0,
            gas_exponent=2.0,
            mixing_rule=sim3D.stone_II_rule,
        )

        # Capillary pressure table
        capillary_pressure_table = sim3D.BrooksCoreyCapillaryPressureModel(
            oil_water_entry_pressure_water_wet=2.0,
            oil_water_pore_size_distribution_index_water_wet=2.0,
            gas_oil_entry_pressure=2.8,
            gas_oil_pore_size_distribution_index=2.0,
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
        net_to_gross_grid = sim3D.layered_grid(
            grid_shape=grid_shape,
            layer_values=[
                0.42,
                0.55,
                0.68,
                0.35,
                0.60,
                0.72,
                0.80,
                0.50,
                0.63,
                0.47,
            ],
            orientation=sim3D.Orientation.Z,
        )

        # Boundary conditions - aquifer support from bottom
        boundary_conditions = sim3D.BoundaryConditions(
            conditions={
                "pressure": sim3D.GridBoundaryCondition(
                    bottom=sim3D.ConstantBoundary(3500),  # Aquifer pressure
                ),
            }
        )

        # Natural gas (associated gas) properties
        gas_gravity_grid = sim3D.uniform_grid(
            grid_shape=grid_shape,
            value=0.65,  # Typical for associated gas
        )

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
            relative_permeability_table=relative_permeability_table,  # type: ignore
            capillary_pressure_table=capillary_pressure_table,  # type: ignore
            reservoir_gas="methane",
            dip_angle=dip_angle,
            dip_azimuth=dip_azimuth,
        )

        options = sim3D.Options(
            scheme="impes",
            total_time=sim3D.Time(days=sim3D.c.DAYS_PER_YEAR * 5),  # 3 years
            time_step_size=sim3D.Time(hours=24),
            max_time_steps=30,
            output_frequency=1,
            miscibility_model="immiscible",
            use_pseudo_pressure=True,
        )
        states = sim3D.run(model=model, wells=None, options=options)
        return list(states)

    return main, np, sim3D


@app.cell
def _(main):
    states = main()
    return (states,)


@app.cell
def _(sim3D, states):
    from pathlib import Path

    sim3D.dump_states(
        states,
        filepath=Path.cwd() / "scenarios/states/stabilization_refined.pkl",
        exist_ok=True,
        compression="lzma",
    )

    # Save last stabilized state
    stabilized_state = states[-1]
    stabilized_state.dump(
        filepath=Path.cwd() / "scenarios/states/stabilized_refined.pkl",
        compression="lzma",
        exist_ok=True,
    )
    return


@app.cell
def _(np, sim3D, states):
    analyst = sim3D.ProductionAnalyst(states)
    oil_production_history = analyst.oil_production_history(
        interval=2, cumulative=False, from_time_step=1
    )
    water_production_history = analyst.water_production_history(
        interval=2, cumulative=False, from_time_step=1
    )
    gas_production_history = analyst.free_gas_production_history(
        interval=2, cumulative=False, from_time_step=1
    )
    water_injection_history = analyst.water_injection_history(
        interval=2, cumulative=False, from_time_step=1
    )
    gas_injection_history = analyst.gas_injection_history(
        interval=2, cumulative=False, from_time_step=1
    )
    oil_in_place_history = analyst.oil_in_place_history(interval=2, from_time_step=1)
    gas_in_place_history = analyst.gas_in_place_history(interval=2, from_time_step=1)
    water_in_place_history = analyst.water_in_place_history(
        interval=2, from_time_step=1
    )
    sweep_efficiency_history = analyst.sweep_efficiency_history(
        interval=2, from_time_step=1
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

    for i in range(1, len(states), 2):
        state = states[i]
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

    # Pressure
    pressure_fig = sim3D.make_series_plot(
        data={"Avg. Reservoir Pressure": np.array(avg_pressure_history)},
        title="Pressure Analysis",
        x_label="Time Step",
        y_label="Avg. Pressure (psia)",
        marker_sizes=4,
    )
    # Production & Injection
    oil_production_fig = sim3D.make_series_plot(
        data={
            "Oil Production": np.array(list(oil_production_history)),
        },
        title="Oil Production Analysis",
        x_label="Time Step",
        y_label="Production (STB/day)",
        marker_sizes=4,
    )
    water_production_fig = sim3D.make_series_plot(
        data={
            "Water Production": np.array(list(water_production_history)),
        },
        title="Water Production Analysis",
        x_label="Time Step",
        y_label="Production (STB/day)",
        marker_sizes=4,
    )
    gas_production_fig = sim3D.make_series_plot(
        data={
            "Gas Production": np.array(list(gas_production_history)),
        },
        title="Gas Production Analysis",
        x_label="Time Step",
        y_label="Production (SCF/day)",
        marker_sizes=4,
    )
    gas_injection_fig = sim3D.make_series_plot(
        data={
            # "Water Injection": np.array(list(water_injection_history)),
            "Gas Injection": np.array(list(gas_injection_history)),
        },
        title="Gas Injection Analysis",
        x_label="Time Step",
        y_label="Injection (SCF/day)",
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

    # Merge all Plots
    analyses_plots = sim3D.merge_plots(
        pressure_fig,
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
        oil_water_reserves_fig,
        gas_reserve_fig,
        cols=2,
        hovermode="x unified",
    )
    analyses_plots.show()
    return


@app.cell
def _(sim3D, states):
    shared_kwargs = dict(
        plot_type="scatter_3d",
        width=800,
        height=600,
        opacity=1,
        aspect_mode="data",
        z_scale=3,
        use_opacity_scaling=False,
        render_mode="webgl",
    )

    viz = sim3D.plotly3d.DataVisualizer()
    property = "oil-pressure"
    figures = []
    timesteps = [30]
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
