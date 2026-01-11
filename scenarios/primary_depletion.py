import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", app_title="bores")


@app.cell
def _():
    import logging
    from pathlib import Path
    import numpy as np
    import bores
    import os

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    bores.use_32bit_precision()
    stabilized_store = bores.ZarrStore(
        store=Path.cwd() / "scenarios/states/stabilized.zarr",
        metadata_dir=Path.cwd() / "scenarios/states/stabilized_metadata/",
    )


    def main():
        state = list(stabilized_store.load(validate=False))[0]
        model = state.model
        del state

        # Production well
        clamp = bores.ProductionClamp()
        control = bores.MultiPhaseRateControl(
            oil_control=bores.AdaptiveBHPRateControl(
                target_rate=-100,
                target_phase="oil",
                bhp_limit=1200,
                clamp=clamp,
            ),
            gas_control=bores.AdaptiveBHPRateControl(
                target_rate=-500,
                target_phase="gas",
                bhp_limit=1200,
                clamp=clamp,
            ),
            water_control=bores.AdaptiveBHPRateControl(
                target_rate=-10,
                target_phase="water",
                bhp_limit=1200,
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
            simulation_time=bores.Time(days=bores.c.DAYS_PER_YEAR * 5),  # 5 years
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
            table_data=pvt_table_data,
            interpolation_method="linear",
        )
        config = bores.Config(
            scheme="impes",
            output_frequency=1,
            miscibility_model="immiscible",
            use_pseudo_pressure=True,
            log_interval=5,
            iterative_solver="bicgstab",
            preconditioner="ilu",
            pvt_tables=pvt_tables,
        )
        states = bores.run(model=model, timer=timer, wells=wells, config=config)
        return states
    return Path, bores, main


@app.cell
def _(Path, bores):

    depletion_store = bores.HDF5Store(
        filepath=Path.cwd() / "scenarios/states/primary_depletion.h5",
        metadata_dir=Path.cwd() / "scenarios/states/primary_depletion_metadata/",
    )
    return (depletion_store,)


@app.cell
def _(bores, depletion_store, main):
    stream = bores.StateStream(
        main(),
        store=depletion_store,
        batch_size=50,
        # checkpoint_interval=20,
        # checkpoint_dir=Path.cwd() / "scenarios/states/checkpoints",
    )
    last_state = None
    with stream:
        for state in stream:
            last_state = state
    return (last_state,)


@app.cell
def _(Path, bores, last_state):
    # Dump last state for use in EOR in next stages
    depleted_store = bores.ZarrStore(
        store=Path.cwd() / "scenarios/states/primary_depleted.zarr",
        metadata_dir=Path.cwd() / "scenarios/states/primary_depleted_metadata/",
    )
    depleted_store.dump([last_state], validate=False)
    return


if __name__ == "__main__":
    app.run()
