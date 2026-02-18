import marimo

__generated_with = "0.19.6"
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

    # Load the new run with the resulting model state from the primary depletion run
    run = bores.Run.from_files(
        model_path=Path("./scenarios/runs/primary_depletion/results/model.h5"),
        config_path=Path("./scenarios/runs/setup/config.yaml"),
        pvt_table_path=Path("./scenarios/runs/setup/pvt.h5"),
    )

    # Gas injection wells, 5-spot pattern
    injection_clamp = bores.InjectionClamp()
    control = bores.AdaptiveBHPRateControl(
        target_rate=1_000_000,
        target_phase="gas",
        bhp_limit=3500,
        clamp=injection_clamp,
    )
    gas_injector_1 = bores.injection_well(
        well_name="GI-1",
        perforating_intervals=[((16, 3, 1), (16, 3, 3))],
        radius=0.3542,  # 8.5 inch wellbore
        control=control,
        injected_fluid=bores.InjectedFluid(
            name="CO2",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=44.0,
            minimum_miscibility_pressure=3250.0,
            todd_longstaff_omega=0.67,
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

    # Producer well
    production_clamp = bores.ProductionClamp()
    control = bores.MultiPhaseRateControl(
        oil_control=bores.AdaptiveBHPRateControl(
            target_rate=-150,
            target_phase="oil",
            bhp_limit=1200,
            clamp=production_clamp,
        ),
        gas_control=bores.AdaptiveBHPRateControl(
            target_rate=-500,
            target_phase="gas",
            bhp_limit=1200,
            clamp=production_clamp,
        ),
        water_control=bores.AdaptiveBHPRateControl(
            target_rate=-10,
            target_phase="water",
            bhp_limit=1200,
            clamp=production_clamp,
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
        is_active=False,
    )
    # We use a well schedule to activate the producer after some time
    well_schedule = bores.WellSchedule()
    well_schedule.add(
        id_="open_well",
        event=bores.WellEvent(
            predicate=bores.time_predicate(time=bores.Time(days=100)),
            action=bores.update_well(is_active=True),
        ),
    )
    well_schedules = bores.WellSchedules()
    well_schedules.add(well_name=producer.name, schedule=well_schedule)

    producers = [producer]

    wells = bores.wells_(injectors=injectors, producers=producers)
    timer = bores.Timer(
        initial_step_size=bores.Time(hours=30.0),
        max_step_size=bores.Time(days=5.0),
        min_step_size=bores.Time(minutes=10),
        simulation_time=bores.Time(days=(bores.c.DAYS_PER_YEAR * 2) + 100),
        max_cfl_number=0.9,
        ramp_up_factor=1.2,
        backoff_factor=0.5,
        aggressive_backoff_factor=0.25,
        max_rejects=20,
    )
    run.config.update(
        wells=wells,
        well_schedules=well_schedules,
        timer=timer,
        # miscibility_model="todd_longstaff",
    )
    run.config.to_file(Path("./scenarios/runs/co2_injection/config.yaml"))
    return Path, bores, run


@app.cell
def save_run(Path, run):
    run.to_file(Path("./scenarios/runs/co2_injection/run.h5"))
    return


@app.cell
def create_store(Path, bores):
    store = bores.ZarrStore(
        store=Path("./scenarios/runs/co2_injection/results/co2_injection.zarr")
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
        stream.consume()
    return


if __name__ == "__main__":
    app.run()
