import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full", app_title="SIM3D")


@app.cell
def _():
    import main
    import sim3D

    model_states = main.simulate()
    return model_states, sim3D


@app.cell
def _(model_states, sim3D):
    wells = model_states[0].wells
    injector_locations, producer_locations = wells.locations
    injector_names, producer_names = wells.names
    well_positions = [injector_locations[0], producer_locations[0]]
    well_names = [*injector_names, *producer_names]
    labels = sim3D.LabelManager()
    labels.add_well_labels(well_positions, well_names)
    sim3D.viz.animate_property(
        model_states,
        property="pressure",
        plot_type=sim3D.PlotType.CELL_BLOCKS,
        width=960,
        height=600,
        # x_slice=(2, 9),
        # y_slice=(2, 7),
        # z_slice=(2, 5),
        # labels=labels,
    )
    return


if __name__ == "__main__":
    app.run()
