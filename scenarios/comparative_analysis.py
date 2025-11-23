import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np

    import sim3D

    sim3D.image_config(scale=3)

    viz = sim3D.plotly1d.DataVisualizer()
    return np, viz


@app.cell
def _(np, viz):
    # Cumulative Recovery
    cumulative_recovery_bar_chart = viz.make_plot(
        data=[
            np.array([[2, 201.0105]]),
            np.array([[3, 321.9279]]),
            np.array([[4, 343.5733]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="stack",
        bar_width=0.5,
        y_label="Cumulative Oil Production (MSTB)",
        x_label="Case",
        title="Cumulative Recovery",
        series_names=["Primary Depletion", "CO2 Injection", "CH4 Injection"],
    )
    cumulative_recovery_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Incremental Recovery

    incremental_recovery_bar_chart = viz.make_plot(
        data={
            "Primary Depeletion": np.array([[0, 201.0105]]),
            "CO2 Injection": np.array([[0, 321.9279]]),
            "CH4 Injection": np.array([[0, 343.5733]]),
        },
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="stack",
        bar_width=0.3,
        y_label="Cumulative Oil Production (MSTB)",
        x_label="Case",
        title="Incremental Recovery",
    )
    incremental_recovery_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Recovery factor
    recovery_factor_bar_chart = viz.make_plot(
        data=[
            np.array([[2, 0.0461]]),
            np.array([[3, 0.0731]]),
            np.array([[4, 0.0780]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="group",
        bar_width=0.5,
        y_label="Recovery Factor",
        x_label="Case",
        title="Recovery Factors",
        series_names=["Primary Depletion", "CO2 Injection", "CH4 Injection"],
        # y_range=[0, 1],
    )
    recovery_factor_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Recovery efficiency
    recovery_efficiency_bar_chart = viz.make_plot(
        data=[
            np.array([[2, 15.4823]]),
            np.array([[3, 17.1992]]),
            np.array([[4, 14.4480]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="group",
        bar_width=0.5,
        y_label="Recovery Efficiency (%)",
        x_label="Case",
        title="Recovery Efficiencies",
        series_names=["Primary Depletion", "CO2 Injection", "CH4 Injection"],
        # y_range=[0, 100],
    )
    recovery_efficiency_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Displacement efficiency
    displacement_efficiency_bar_chart = viz.make_plot(
        data=[
            np.array([[2, 45.4038]]),
            np.array([[3, 20.9611]]),
            np.array([[4, 23.5072]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="group",
        bar_width=0.5,
        y_label="Displacement Efficiency (%)",
        x_label="Case",
        title="Displacement Efficiencies",
        series_names=["Primary Depletion (Aquifer Driven)", "CO2 Injection", "CH4 Injection"],
        y_range=[0, 100],
    )
    displacement_efficiency_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Volumetric Sweep efficiency
    vol_sweep_efficiency_bar_chart = viz.make_plot(
        data=[
            np.array([[2, 34.0992]]),
            np.array([[3, 82.0530]]),
            np.array([[4, 61.4613]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="group",
        bar_width=0.5,
        y_label="Volumetric Sweep Efficiency (%)",
        x_label="Case",
        title="Volumetric Sweep Efficiencies",
        series_names=["Primary Depletion", "CO2 Injection", "CH4 Injection"],
        y_range=[0, 100],
    )
    vol_sweep_efficiency_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Mobility Ratio
    mobility_ratio_bar_chart = viz.make_plot(
        data=[
            np.array([[3, 2.0688]]),
            np.array([[4, 2.0147]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="group",
        bar_width=0.5,
        y_label="Mobility Ratio (MR)",
        x_label="Case",
        title="Mobility Ratios",
        series_names=["CO2 Injection", "CH4 Injection"],
    )
    mobility_ratio_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # VRR
    vrr_bar_chart = viz.make_plot(
        data=[
            np.array([[3, 10.5350]]),
            np.array([[4, 10.8614]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="group",
        bar_width=0.5,
        y_label="Voidage Replacement Ratio (VRR)",
        x_label="Case",
        title="Voidage Replacement Ratios",
        series_names=["CO2 Injection", "CH4 Injection"],
    )
    vrr_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Solvent Concentration
    solvent_conc_bar_chart = viz.make_plot(
        data=[
            np.array([[3, 14.16]]),
            np.array([[4, 9.86]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="group",
        bar_width=0.5,
        y_label="Dissolved Solvent Concentration (%)",
        x_label="Case",
        title="Dissolved Solvent Concentrations",
        series_names=["CO2 Injection", "CH4 Injection"],
        # y_range=[0, 100],
    )
    solvent_conc_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Viscosity Changes
    viscosity_bar_chart = viz.make_plot(
        data=[
            np.array([[2, 19.73]]),
            np.array([[3, -19.55]]),
            np.array([[4, -12.84]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="relative",
        bar_width=0.7,
        y_label="Avg. Oil Viscosity Change (%)",
        x_label="Case",
        title="Average Oil Viscosity Percentage Changes",
        series_names=["Primary Depletion", "CO2 Injection", "CH4 Injection"],
    )
    viscosity_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Density Changes
    density_bar_chart = viz.make_plot(
        data=[
            np.array([[2, 3.73]]),
            np.array([[3, -15.99]]),
            np.array([[4, -11.19]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="relative",
        bar_width=0.5,
        y_label="Avg. Density Change (%)",
        x_label="Case",
        title="Average Oil Density Percentage Changes",
        series_names=["Primary Depletion", "CO2 Injection", "CH4 Injection"],
    )
    density_bar_chart.show()
    return


@app.cell
def _(np, viz):
    # Oil Relative Mobility Changes
    oil_rel_mobility_bar_chart = viz.make_plot(
        data=[
            np.array([[2, -22.66]]),
            np.array([[3, 44.49]]),
            np.array([[4, 28.89]]),
        ],
        plot_type="bar",
        width=960,
        height=600,
        bar_mode="relative",
        bar_width=0.5,
        y_label="Avg. Oil Relative Mobility Change (%)",
        x_label="Case",
        title="Average Oil Relative Mobility Percentage Changes",
        series_names=["Primary Depletion", "CO2 Injection", "CH4 Injection"],
    )
    oil_rel_mobility_bar_chart.show()
    return


if __name__ == "__main__":
    app.run()
