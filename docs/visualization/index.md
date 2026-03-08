# Visualization

BORES includes a built-in visualization system based on Plotly for creating interactive plots of simulation results. The system covers three levels of visualization: 1D time series plots for production histories and performance metrics, 2D maps for spatial property distributions across reservoir layers, and 3D volume renderings for full-field property visualization with well overlays.

All visualization is handled through `DataVisualizer` classes, one for each dimensionality. Each visualizer uses a registry of renderers that map plot types to rendering strategies. You pass in your data (typically from a `ModelState`) and get back a Plotly `Figure` object that you can display interactively, save as an image, or embed in a report.

The visualization modules are designed to work directly with BORES data structures. The 3D visualizer accepts `ModelState` or `ReservoirModel` objects and can extract any registered property by name. The 1D and 2D visualizers work with numpy arrays, which you extract from states yourself. This gives you full control over data selection and transformation while keeping the plotting API clean.

## Visualization Modules

| Module | Import | Use Case |
| --- | --- | --- |
| 1D Series | `bores.visualization.plotly1d` | Production rates, pressure decline, recovery factors |
| 2D Maps | `bores.visualization.plotly2d` | Layer slices, areal distributions, cross-sections |
| 3D Volumes | `bores.visualization.plotly3d` | Full-field properties, isosurfaces, well locations |
| Base | `bores.visualization.base` | Color schemes, property registry, plot merging |

## Quick Start

```python
import bores
from bores.visualization import plotly1d, plotly2d, plotly3d

# Run a simulation
states = list(bores.run(model, config))

# 1D: Plot pressure decline over time
import numpy as np
time_pressure = np.column_stack([
    [s.time_in_days for s in states],
    [s.model.fluid_properties.pressure_grid.mean() for s in states],
])
viz1d = plotly1d.DataVisualizer()
fig = viz1d.make_plot(time_pressure, x_label="Time (days)", y_label="Pressure (psi)")
fig.show()

# 2D: Heatmap of oil saturation in a layer
state = states[-1]
So_layer = state.model.fluid_properties.oil_saturation_grid[:, :, 0]
viz2d = plotly2d.DataVisualizer()
fig = viz2d.make_plot(So_layer, plot_type="heatmap")
fig.show()

# 3D: Volume rendering of pressure
viz3d = plotly3d.DataVisualizer()
fig = viz3d.make_plot(state, "pressure")
fig.show()
```

## Color Schemes

BORES provides colorblind-friendly color schemes through the `ColorScheme` enum. The default schemes are chosen to be perceptually uniform and accessible:

| Scheme | Type | Best For |
| --- | --- | --- |
| `VIRIDIS` | Sequential | Pressure, permeability (recommended default) |
| `CIVIDIS` | Sequential | Oil saturation, porosity (colorblind-optimized) |
| `PLASMA` | Sequential | Density, concentration |
| `INFERNO` | Sequential | Temperature, viscosity |
| `MAGMA` | Sequential | Gas saturation |
| `RdBu` | Diverging | Pressure changes, saturation differences |
| `RdYlBu` | Diverging | Multi-value diverging data |

The `ColorbarPresets` class provides pre-configured colorbar settings for common reservoir properties with appropriate scales and formatting.

## Property Registry

The 3D visualizer uses a `PropertyRegistry` that maps property names to metadata including display names, units, color schemes, and scaling options. When you call `viz3d.make_plot(state, "pressure")`, the registry looks up the property metadata to determine how to label, scale, and color the visualization.

You can access the registry directly:

```python
from bores.visualization.base import property_registry

# List all registered properties
for name, meta in property_registry.items():
    print(f"{name}: {meta.display_name} ({meta.unit})")
```

Properties with `log_scale=True` (like viscosity and compressibility) are automatically log-transformed for visualization while showing the original physical values in hover text and colorbar labels.

## Merging Plots

The `merge_plots()` function combines multiple Plotly figures into a single subplot grid. This is useful for creating multi-panel comparison views:

```python
from bores.visualization.base import merge_plots

fig_pressure = viz3d.make_plot(state, "pressure")
fig_oil_sat = viz3d.make_plot(state, "oil_saturation")
fig_water_sat = viz3d.make_plot(state, "water_saturation")

combined = merge_plots(
    fig_pressure, fig_oil_sat, fig_water_sat,
    cols=3,
    subplot_titles=["Pressure", "Oil Saturation", "Water Saturation"],
    title="Reservoir State at Final Time Step",
)
combined.show()
```

## Exporting Images

Configure image export settings globally with `image_config()`:

```python
from bores.visualization.base import image_config

# Set default export format and resolution
image_config(fmt="png", scale=3)

# Save a figure
fig.write_image("pressure_map.png")
```

## Environment Configuration

Performance limits and styling defaults can be customized through environment variables. Set these before importing BORES:

```bash
export BORES_MAX_VOLUME_CELLS=2000000      # Max cells for 3D volume rendering
export BORES_COLORBAR_THICKNESS=20         # Colorbar thickness in pixels
export BORES_RECOMMENDED_VOLUME_CELLS=512000  # Target cells after auto-coarsening
```

To see the current configuration:

```python
from bores.visualization.config import config_summary
print(config_summary())
```
