import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full", app_title="SIM3D")


@app.cell
def _():
    import main
    import sim3D
    import logging

    logging.getLogger("sim3D").setLevel(logging.INFO)

    model_states = main.example()
    return model_states, sim3D


@app.cell
def _(model_states, sim3D):
    wells = model_states[0].wells
    injector_locations, producer_locations = wells.locations
    injector_names, producer_names = wells.names
    well_positions = [*injector_locations, *producer_locations]
    well_names = [*injector_names, *producer_names]
    labels = sim3D.Labels()
    labels.add_well_labels(well_positions, well_names)
    sim3D.viz.make_plot(
        model_states[250],
        property="gas-saturation",
        plot_type=sim3D.PlotType.VOLUME_RENDER,
        width=1080,
        height=600,
        # isomin=0.13,
        # cmin=600,
        # cmax=2700,
        # opacity=0.5,
        # use_opacity_scaling=False,
        # subsampling_factor=2,
        # downsampling_factor=1,
        # x_slice=(6, 9),
        # y_slice=(6, 9),
        z_slice=(2, 5),
        labels=labels,
        aspect_mode="data",
        # marker_size=12,
        # notebook=True,
    )
    return


if __name__ == "__main__":
    app.run()
