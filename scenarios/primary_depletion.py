import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", app_title="bores")


@app.cell
def _():
    import logging
    from pathlib import Path
    import numpy as np
    import bores

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    stabilized_store = bores.ZarrStore(
        store=Path.cwd() / "scenarios/states/stabilized.zarr",
    )

    def main():
        state = list(stabilized_store.load(bores.ModelState, lazy=False))[-1]
        model = state.model
        del state

        # Production well
        clamp = bores.ProductionClamp()
        control = bores.MultiPhaseRateControl(
            oil_control=bores.AdaptiveBHPRateControl(
                target_rate=-100,
                target_phase="oil",
                bhp_limit=800,
                clamp=clamp,
            ),
            gas_control=bores.AdaptiveBHPRateControl(
                target_rate=-500,
                target_phase="gas",
                bhp_limit=800,
                clamp=clamp,
            ),
            water_control=bores.AdaptiveBHPRateControl(
                target_rate=-10,
                target_phase="water",
                bhp_limit=800,
                clamp=clamp,
            ),
        )
        producer = bores.production_well(
            well_name="P-1",
            perforating_intervals=[((14, 10, 3), (14, 10, 4))],
            radius=0.3542,
            control=control,
            produced_fluids=(
                bores.ProducedFluid(
                    name="Oil",
                    phase=bores.FluidPhase.OIL,
                    specific_gravity=0.845,
                    molecular_weight=180.0,
                ),
                bores.ProducedFluid(
                    name="Gas",
                    phase=bores.FluidPhase.GAS,
                    specific_gravity=0.65,
                    molecular_weight=16.04,
                ),
                bores.ProducedFluid(
                    name="Water",
                    phase=bores.FluidPhase.WATER,
                    specific_gravity=1.05,
                    molecular_weight=18.015,
                ),
            ),
            skin_factor=2.5,
            is_active=True,
        )

        producers = [producer]
        wells = bores.wells_(injectors=None, producers=producers)
        timer = bores.Timer(
            initial_step_size=bores.Time(hours=20),
            max_step_size=bores.Time(days=5),
            min_step_size=bores.Time(minutes=10.0),
            simulation_time=bores.Time(days=2 * bores.c.DAYS_PER_YEAR),  # 5 years
            max_cfl_number=0.9,
            ramp_up_factor=1.2,
            backoff_factor=0.5,
            aggressive_backoff_factor=0.25,
            max_rejects=20,
        )
        pvt_table_data = bores.build_pvt_table_data(
            pressures=bores.array(
                [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
            ),
            temperatures=bores.array([120, 140, 160, 180, 200, 220]),
            salinities=bores.array([30000, 32000, 33500, 35000]),  # ppm
            oil_specific_gravity=0.845,
            gas_gravity=0.65,
            reservoir_gas="methane",
        )
        pvt_tables = bores.PVTTables(
            table_data=pvt_table_data, interpolation_method="linear"
        )
        del pvt_table_data

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
            mixing_rule=bores.eclipse_rule,
        )

        # Capillary pressure table
        capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
            oil_water_entry_pressure_water_wet=2.0,
            oil_water_pore_size_distribution_index_water_wet=2.0,
            gas_oil_entry_pressure=2.8,
            gas_oil_pore_size_distribution_index=2.0,
            wettability=bores.Wettability.WATER_WET,
        )
        rock_fluid_tables = bores.RockFluidTables(
            relative_permeability_table=relative_permeability_table,
            capillary_pressure_table=capillary_pressure_table,
        )

        config = bores.Config(
            timer=timer,
            rock_fluid_tables=rock_fluid_tables,
            wells=wells,
            scheme="impes",
            output_frequency=1,
            miscibility_model="immiscible",
            pressure_solver="bicgstab",
            pressure_preconditioner="ilu",
            pvt_tables=pvt_tables,
            max_gas_saturation_change=0.85,
        )
        states = bores.run(model=model, config=config)
        return states
    return Path, bores, main


@app.cell
def _(Path, bores):
    depletion_store = bores.HDF5Store(
        filepath=Path.cwd() / "scenarios/states/primary_depletion.h5",
        group_name_gen=bores.state_group_name_gen,
    )
    return (depletion_store,)


@app.cell
def _(bores, depletion_store, main):
    stream = bores.StateStream(main(), store=depletion_store, batch_size=10)

    with stream:
        last_state = stream.last()
    return (last_state,)


@app.cell
def _(Path, bores, last_state):
    # Dump last state for use in EOR in next stages
    depleted_store = bores.ZarrStore(
        store=Path.cwd() / "scenarios/states/primary_depleted.zarr"
    )
    depleted_store.dump([last_state])
    return


if __name__ == "__main__":
    app.run()
