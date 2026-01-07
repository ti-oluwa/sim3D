import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", app_title="bores")


@app.cell
def _():
    import logging
    import typing
    from pathlib import Path
    import numpy as np
    import bores


    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    bores.use_32bit_precision()


    def main():
        cell_dimension = (100.0, 100.0)  # 100ft x 100ft cells
        grid_shape = typing.cast(
            bores.ThreeDimensions,
            (20, 20, 10),  # 30x30 cells, 10 layers
        )
        dip_angle = 5.0
        dip_azimuth = 90.0

        # Thickness distribution - typical reservoir layers
        # Thicker in the middle, thinner at top/bottom
        thickness_values = bores.array(
            [30.0, 20.0, 25.0, 30.0, 25.0, 30.0, 20.0, 25.0, 30.0, 25.0]
        )  # feet
        thickness_grid = bores.layered_grid(
            grid_shape=grid_shape,
            layer_values=thickness_values,
            orientation=bores.Orientation.Z,
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

        pressure_grid = bores.layered_grid(
            grid_shape=grid_shape,
            layer_values=layer_pressures,  # Ranges from ~3055 to ~3117 psi
            orientation=bores.Orientation.Z,
        )

        # Bubble point pressure slightly below initial pressure (undersaturated oil)
        oil_bubble_point_pressure_grid = bores.layered_grid(
            grid_shape=grid_shape,
            layer_values=layer_pressures
            - 400.0,  # 400 psi below formation pressure
            orientation=bores.Orientation.Z,
        )

        # Saturation endpoints - typical for sandstone reservoirs
        residual_oil_saturation_water_grid = bores.uniform_grid(
            grid_shape=grid_shape,
            value=0.25,  # Sor to water
        )
        residual_oil_saturation_gas_grid = bores.uniform_grid(
            grid_shape=grid_shape,
            value=0.15,  # Sor to gas
        )
        irreducible_water_saturation_grid = bores.uniform_grid(
            grid_shape=grid_shape,
            value=0.15,  # Swi
        )
        connate_water_saturation_grid = bores.uniform_grid(
            grid_shape=grid_shape,
            value=0.12,  # Slightly less than Swi
        )
        residual_gas_saturation_grid = bores.uniform_grid(
            grid_shape=grid_shape,
            value=0.045,  # Sgr
        )

        # Porosity - decreasing with depth (compaction trend)
        porosity_values = bores.array(
            [0.04, 0.07, 0.09, 0.1, 0.08, 0.12, 0.14, 0.16, 0.11, 0.08]
        )  # fraction
        porosity_grid = bores.layered_grid(
            grid_shape=grid_shape,
            layer_values=porosity_values,
            orientation=bores.Orientation.Z,
        )

        # Fluid contacts
        # GOC at 8060 ft, OWC at 8220 ft
        goc_depth = 8060.0
        owc_depth = 8220.0

        depth_grid = bores.depth_grid(thickness_grid)
        # Apply structural dip
        depth_grid = bores.apply_structural_dip(
            elevation_grid=depth_grid,
            elevation_direction="downward",
            cell_dimension=cell_dimension,
            dip_angle=dip_angle,
            dip_azimuth=dip_azimuth,
        )
        water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
            bores.build_saturation_grids(
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
                transition_curvature_exponent=1.2,
            )
        )

        # Oil viscosity - increases slightly with depth (heavier oil)
        oil_viscosity_values = np.linspace(1.2, 2.5, grid_shape[2])  # cP
        oil_viscosity_grid = bores.layered_grid(
            grid_shape=grid_shape,
            layer_values=oil_viscosity_values,
            orientation=bores.Orientation.Z,
        )

        # Oil compressibility - typical range for crude oil
        oil_compressibility_grid = bores.uniform_grid(
            grid_shape=grid_shape,
            value=1.2e-5,  # 1/psi
        )

        # Oil specific gravity (API ~35-40 degrees)
        oil_specific_gravity_grid = bores.uniform_grid(
            grid_shape=grid_shape,
            value=0.845,  # ~36 API
        )

        # Permeability distribution
        # Higher permeability in middle layers (better reservoir quality)
        # Anisotropy ratio kv/kh ~ 0.1 (typical for layered sandstone)
        x_perm_values = bores.array([12, 25, 40, 18, 55, 70, 90, 35, 48, 22])  # mD
        x_permeability_grid = bores.layered_grid(
            grid_shape=grid_shape,
            layer_values=x_perm_values,
            orientation=bores.Orientation.Z,
        )
        # Slight directional permeability difference
        y_permeability_grid = typing.cast(
            bores.ThreeDimensionalGrid, x_permeability_grid * 0.8
        )
        # Vertical permeability much lower (layering effect)
        z_permeability_grid = typing.cast(
            bores.ThreeDimensionalGrid, x_permeability_grid * 0.1
        )

        absolute_permeability = bores.RockPermeability(
            x=x_permeability_grid,
            y=y_permeability_grid,
            z=z_permeability_grid,
        )

        # RelPerm table
        relative_permeability_table = bores.BrooksCoreyThreePhaseRelPermModel(
            irreducible_water_saturation=0.15,
            residual_oil_saturation_gas=0.15,
            residual_oil_saturation_water=0.25,
            residual_gas_saturation=0.045,
            wettability=bores.WettabilityType.WATER_WET,
            water_exponent=2.0,
            oil_exponent=2.0,
            gas_exponent=2.0,
            mixing_rule=bores.stone_II_rule,
        )

        # Capillary pressure table
        capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
            oil_water_entry_pressure_water_wet=2.0,
            oil_water_pore_size_distribution_index_water_wet=2.0,
            gas_oil_entry_pressure=2.8,
            gas_oil_pore_size_distribution_index=2.0,
            wettability=bores.Wettability.WATER_WET,
        )

        # Realistic temperature gradient (~1.5°F per 100 ft)
        surface_temp = 60.0  # °F
        temp_gradient = 0.015  # °F/ft
        layer_temps = surface_temp + (layer_depths * temp_gradient)
        temperature_grid = bores.layered_grid(
            grid_shape=grid_shape,
            layer_values=layer_temps,  # ~180-182°F
            orientation=bores.Orientation.Z,
        )
        # Rock compressibility for sandstone
        rock_compressibility = 4.5e-6  # 1/psi
        # Net-to-gross ratio (accounting for shale layers)
        net_to_gross_grid = bores.layered_grid(
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
            orientation=bores.Orientation.Z,
        )

        # Boundary conditions - aquifer support from bottom
        boundary_conditions = bores.BoundaryConditions(
            conditions={
                "pressure": bores.GridBoundaryCondition(
                    bottom=bores.ConstantBoundary(3500),  # Aquifer pressure
                ),
            }
        )

        # Natural gas (associated gas) properties
        gas_gravity_grid = bores.uniform_grid(
            grid_shape=grid_shape,
            value=0.65,  # Typical for associated gas
        )

        model = bores.reservoir_model(
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

        pvt_table_data = bores.build_pvt_table_data(
            pressures=bores.array(
                [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
            ),
            temperatures=bores.array([120, 140, 160, 180, 200, 220]),
            salinities=bores.array([30000, 32000, 36000, 40000]),  # ppm
            oil_specific_gravity=0.845,
            gas_gravity=0.65,
            reservoir_gas="methane",
        )
        pvt_tables = bores.PVTTables(
            table_data=pvt_table_data,
            interpolation_method="cubic",
        )
        timer = bores.Timer(
            initial_step_size=bores.Time(hours=4.5),
            max_step_size=bores.Time(days=1.0),
            min_step_size=bores.Time(hours=1.0),
            simulation_time=bores.Time(days=30),  # 5 years
            max_cfl_number=0.9,
            ramp_up_factor=1.2,
            backoff_factor=0.5,
            aggressive_backoff_factor=0.25,
        )
        config = bores.Config(
            scheme="impes",
            output_frequency=1,
            miscibility_model="immiscible",
            use_pseudo_pressure=True,
            max_iterations=500,
            iterative_solver="bicgstab",
            preconditioner="ilu",
            log_interval=2,
            pvt_tables=pvt_tables,
        )
        states = bores.run(model=model, timer=timer, wells=None, config=config)
        return states
    return Path, bores, main


@app.cell
def _(Path, bores):
    stabilization_store = bores.ZarrStore(
        store=Path.cwd() / "scenarios/states/stabilization.zarr",
        metadata_dir=Path.cwd()
        / "scenarios/states/stabilization_metadata",
    )
    return (stabilization_store,)


@app.cell
def _(Path, bores, main, stabilization_store):
    stream = bores.StateStream(
        main(),
        store=stabilization_store,
        checkpoint_interval=10,
        checkpoint_dir=Path.cwd() / "scenarios/states/checkpoints",
        auto_replay=True,
    )

    last_state = None
    with stream:
        for state in stream:
            last_state = state
        
    return (last_state,)


@app.cell
def _(Path, bores, last_state):
    stabilized_store = bores.ZarrStore(
        store=Path.cwd() / "scenarios/states/stabilized.zarr",
        metadata_dir=Path.cwd()
        / "scenarios/states/stabilized_metadata",
    )
    stabilized_store.dump([last_state])
    return


if __name__ == "__main__":
    app.run()
