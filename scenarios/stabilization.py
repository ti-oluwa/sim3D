import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full", app_title="bores")


@app.cell
def build_run_from_base_setup():
    import logging
    from pathlib import Path
    import numpy as np
    import bores

    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(threshold=np.inf)  # type: ignore
    bores.use_32bit_precision()

    run = bores.Run.from_files(
        model_path=Path("./scenarios/runs/setup/model.h5"),
        config_path=Path("./scenarios/runs/setup/config.yaml"),
        pvt_table_path=Path("./scenarios/runs/setup/pvt.h5"),
    )
    return Path, bores, run


@app.cell
def save_run(Path, run):
    run.to_file(Path("./scenarios/runs/stabilization/run.h5"))
    return


@app.cell
def create_store(Path, bores):
    store = bores.ZarrStore(
        store=Path("./scenarios/runs/stabilization/results/stabilization.zarr"),
        group_name_gen=bores.state_group_name_gen,
    )
    return (store,)


@app.cell
def execute_run(bores, run, store):
    stream = bores.StateStream(
        run(),
        store=store,
        async_io=True,
    )
    with stream:
        last_state = stream.last()
    return (last_state,)


@app.cell
def capture_last_model_state(Path, last_state):
    last_state.model.to_file(Path("./scenarios/runs/stabilization/results/model.h5"))
    return


if __name__ == "__main__":
    app.run()
