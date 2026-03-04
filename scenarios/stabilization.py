import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", app_title="bores")


@app.cell
def build_run_from_setup():
    import logging
    from pathlib import Path

    import numpy as np

    import bores

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    bores.use_32bit_precision()

    preconditioner_factory = bores.CachedPreconditionerFactory(
        factory="ilu",
        name="cached_ilu",
        update_frequency=10,
        recompute_threshold=0.3,
    )
    preconditioner_factory.register(override=True)

    run = bores.Run.from_files(
        model_path=Path("./scenarios/runs/setup/model.h5"),
        config_path=Path("./scenarios/runs/setup/config.yaml"),
        pvt_table_path=Path("./scenarios/runs/setup/pvt.h5"),
    )
    return Path, bores, run


@app.cell
def save_run(Path, run):
    run.save(Path("./scenarios/runs/stabilization/run.h5"))
    return


@app.cell
def make_store(Path, bores):
    store = bores.ZarrStore(
        store=Path("./scenarios/runs/stabilization/results/stabilization.zarr")
    )
    return (store,)


@app.cell
def run_simulation(bores, run, store):
    with bores.StateStream(
        run,
        store=store,
        background_io=True,
    ) as stream:
        last_state = stream.last()
    return (last_state,)


@app.cell
def capture_last_state(Path, last_state):
    last_state.model.save(Path("./scenarios/runs/stabilization/results/model.h5"))
    return


if __name__ == "__main__":
    app.run()
