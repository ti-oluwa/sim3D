import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full", app_title="bores")


@app.cell
def setup_run():
    import logging
    from pathlib import Path
    import numpy as np
    import bores

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    bores.use_32bit_precision()


    ilu_preconditioner = bores.CachedPreconditionerFactory(
        factory="ilu",
        name="cached_ilu",
        update_frequency=10,
        recompute_threshold=0.3,
    )
    ilu_preconditioner.register(override=True)
    # Load the new run with the resulting model state from the stabilization run
    run = bores.Run.from_files(
        model_path=Path("./scenarios/runs/stabilization/results/model.h5"),
        config_path=Path("./scenarios/runs/setup/config.yaml"),
        pvt_table_path=Path("./scenarios/runs/setup/pvt.h5"),
    )

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
            target_rate=-100,
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

    wells = bores.wells_(injectors=None, producers=[producer])
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
    run.config = run.config.with_updates(wells=wells, timer=timer)
    return Path, bores, run


@app.cell
def save_run(Path, run):
    run.to_file(Path("./scenarios/runs/primary_depletion/run.h5"))
    return


@app.cell
def create_store(Path, bores):
    store = bores.ZarrStore(
        store=Path("./scenarios/runs/primary_depletion/results/primary_depletion.zarr"),
    )
    return (store,)


@app.cell
def execute_run(bores, run, store):
    stream = bores.StateStream(
        run(),
        store=store,
        batch_size=30,
        async_io=True,
    )
    with stream:
        last_state = stream.last()
    return (last_state,)


@app.cell
def capture_last_model_state(Path, last_state):
    last_state.model.to_file(
        Path("./scenarios/runs/primary_depletion/results/model.h5")
    )
    return


if __name__ == "__main__":
    app.run()
