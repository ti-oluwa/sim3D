"""
3D Visualization Suite for Reservoir Simulation

This module provides a comprehensive visualization factory for 3D reservoir simulation data,
including interactive 3D plots, volume rendering, cross-sections, and animations.
"""

import numpy as np
import typing
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation

from sim3D.simulation import ModelState
from sim3D.types import ThreeDimensions, ThreeDimensionalGrid


class PlotType(str, Enum):
    """Types of 3D plots available."""

    VOLUME_RENDER = "volume_render"
    ISOSURFACE = "isosurface"
    SLICE_PLANES = "slice_planes"
    SCATTER_3D = "scatter_3d"
    SURFACE_3D = "surface_3d"
    CONTOUR_3D = "contour_3d"


class ColorScheme(str, Enum):
    """Professional color schemes for reservoir visualization."""

    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    TURBO = "turbo"
    RdYlBu = "RdYlBu"
    RdBu = "RdBu"
    SPECTRAL = "Spectral"
    COOLWARM = "coolwarm"
    SEISMIC = "seismic"


@dataclass
class Plot3DConfig:
    """Configuration for 3D plotting."""

    width: int = 1200
    height: int = 800
    plot_type: PlotType = PlotType.VOLUME_RENDER
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    opacity: float = 0.1
    show_colorbar: bool = True
    show_axes: bool = True
    camera_position: typing.Optional[typing.Dict[str, typing.Any]] = None
    title: str = ""

    def __post_init__(self):
        if self.camera_position is None:
            self.camera_position = {
                "eye": dict(x=1.5, y=1.5, z=1.5),
                "center": dict(x=0, y=0, z=0),
                "up": dict(x=0, y=0, z=1),
            }


# =====================================================================================
# PROPERTY MAPPING AND METADATA
# =====================================================================================


@dataclass
class PropertyMetadata:
    """Metadata for fluid properties."""

    name: str
    display_name: str
    unit: str
    color_scheme: ColorScheme
    log_scale: bool = False
    min_val: typing.Optional[float] = None
    max_val: typing.Optional[float] = None


class FluidPropertyRegistry:
    """Registry of all available fluid properties for visualization."""

    PROPERTIES = {
        # Pressure and Temperature
        "pressure": PropertyMetadata(
            name="pressure_grid",
            display_name="Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
        ),
        "temperature": PropertyMetadata(
            name="temperature_grid",
            display_name="Temperature",
            unit="°F",
            color_scheme=ColorScheme.INFERNO,
        ),
        # Oil Properties
        "oil_saturation": PropertyMetadata(
            name="oil_saturation_grid",
            display_name="Oil Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            min_val=0,
            max_val=1,
        ),
        "oil_viscosity": PropertyMetadata(
            name="oil_viscosity_grid",
            display_name="Oil Viscosity",
            unit="cP",
            color_scheme=ColorScheme.MAGMA,
            log_scale=True,
        ),
        "oil_density": PropertyMetadata(
            name="oil_density_grid",
            display_name="Oil Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.PLASMA,
        ),
        "oil_compressibility": PropertyMetadata(
            name="oil_compressibility_grid",
            display_name="Oil Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.TURBO,
            log_scale=True,
        ),
        "oil_formation_volume_factor": PropertyMetadata(
            name="oil_formation_volume_factor_grid",
            display_name="Oil FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.RdYlBu,
        ),
        "oil_bubble_point_pressure": PropertyMetadata(
            name="oil_bubble_point_pressure_grid",
            display_name="Oil Bubble Point Pressure",
            unit="psi",
            color_scheme=ColorScheme.SPECTRAL,
        ),
        # Water Properties
        "water_saturation": PropertyMetadata(
            name="water_saturation_grid",
            display_name="Water Saturation",
            unit="fraction",
            color_scheme=ColorScheme.RdBu,
            min_val=0,
            max_val=1,
        ),
        "water_viscosity": PropertyMetadata(
            name="water_viscosity_grid",
            display_name="Water Viscosity",
            unit="cP",
            color_scheme=ColorScheme.COOLWARM,
            log_scale=True,
        ),
        "water_density": PropertyMetadata(
            name="water_density_grid",
            display_name="Water Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.SEISMIC,
        ),
        "water_compressibility": PropertyMetadata(
            name="water_compressibility_grid",
            display_name="Water Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.VIRIDIS,
            log_scale=True,
        ),
        "water_formation_volume_factor": PropertyMetadata(
            name="water_formation_volume_factor_grid",
            display_name="Water FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.PLASMA,
        ),
        "water_bubble_point_pressure": PropertyMetadata(
            name="water_bubble_point_pressure_grid",
            display_name="Water Bubble Point Pressure",
            unit="psi",
            color_scheme=ColorScheme.INFERNO,
        ),
        "water_salinity": PropertyMetadata(
            name="water_salinity_grid",
            display_name="Water Salinity",
            unit="ppm NaCl",
            color_scheme=ColorScheme.CIVIDIS,
        ),
        # Gas Properties
        "gas_saturation": PropertyMetadata(
            name="gas_saturation_grid",
            display_name="Gas Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            min_val=0,
            max_val=1,
        ),
        "gas_viscosity": PropertyMetadata(
            name="gas_viscosity_grid",
            display_name="Gas Viscosity",
            unit="cP",
            color_scheme=ColorScheme.TURBO,
            log_scale=True,
        ),
        "gas_density": PropertyMetadata(
            name="gas_density_grid",
            display_name="Gas Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.RdYlBu,
        ),
        "gas_compressibility": PropertyMetadata(
            name="gas_compressibility_grid",
            display_name="Gas Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.SPECTRAL,
            log_scale=True,
        ),
        "gas_formation_volume_factor": PropertyMetadata(
            name="gas_formation_volume_factor_grid",
            display_name="Gas FVF",
            unit="ft³/SCF",
            color_scheme=ColorScheme.RdBu,
        ),
        "gas_to_oil_ratio": PropertyMetadata(
            name="gas_to_oil_ratio_grid",
            display_name="Gas-Oil Ratio",
            unit="SCF/STB",
            color_scheme=ColorScheme.COOLWARM,
        ),
        "gas_gravity": PropertyMetadata(
            name="gas_gravity_grid",
            display_name="Gas Gravity",
            unit="dimensionless",
            color_scheme=ColorScheme.SEISMIC,
        ),
        "gas_molecular_weight": PropertyMetadata(
            name="gas_molecular_weight_grid",
            display_name="Gas Molecular Weight",
            unit="g/mol",
            color_scheme=ColorScheme.VIRIDIS,
        ),
        # API Gravity
        "oil_api_gravity": PropertyMetadata(
            name="oil_api_gravity_grid",
            display_name="Oil API Gravity",
            unit="°API",
            color_scheme=ColorScheme.PLASMA,
        ),
        "oil_specific_gravity": PropertyMetadata(
            name="oil_specific_gravity_grid",
            display_name="Oil Specific Gravity",
            unit="dimensionless",
            color_scheme=ColorScheme.INFERNO,
        ),
    }

    def __init__(self):
        """
        Initialize the property registry.

        This class is a singleton and should not be instantiated directly.
        Use the class methods to access properties and metadata.
        """
        self.__properties = type(self).PROPERTIES.copy()

    def get_available_properties(self) -> typing.List[str]:
        """
        Get list of all available property names.

        :return: List of property names that can be used for visualization
        """
        return list(self.PROPERTIES.keys())

    def get_metadata(self, property: str) -> PropertyMetadata:
        """
        Get metadata for a specific property.

        :param property: Name of the property to get metadata for
        :return: `PropertyMetadata` object containing display information
        :raises ValueError: If property is not found in the registry
        """
        if property not in self.PROPERTIES:
            raise ValueError(
                f"Unknown property: {property}. Available: {', '.join(self.get_available_properties())}"
            )
        return self.PROPERTIES[property]

    def __getitem__(self, name: str, /) -> PropertyMetadata:
        return self.get_metadata(name)

    def __setitem__(self, name: str, value: PropertyMetadata, /) -> None:
        if not isinstance(value, PropertyMetadata):
            raise TypeError("Value must be a `PropertyMetadata` instance")
        self.PROPERTIES[name] = value


fluid_property_registry = FluidPropertyRegistry()

# =====================================================================================
# BASE PLOTTER CLASSES
# =====================================================================================


class Base3DPlotter(ABC):
    """Base class for 3D plotters."""

    def __init__(self, config: Plot3DConfig) -> None:
        self.config = config

    @abstractmethod
    def plot(self, data: ThreeDimensionalGrid, metadata: PropertyMetadata) -> go.Figure:
        """
        Create a 3D plot from data.

        Subclasses should override this with their specific keyword arguments.
        """
        pass

    def get_colorscale(self, color_scheme: ColorScheme) -> str:
        """
        Get plotly colorscale string from ColorScheme enum.

        :param color_scheme: The color scheme enum to convert
        :return: Plotly colorscale string
        """
        return color_scheme.value

    def normalize_data(
        self, data: ThreeDimensionalGrid, metadata: PropertyMetadata
    ) -> ThreeDimensionalGrid:
        """
        Prepare data for plotting (handle log scale, clipping, etc.).

        :param data: Input data array to prepare
        :param metadata: Property metadata containing scaling and range information
        :return: Prepared data array ready for plotting
        """
        normalized_data = data.astype(np.float64)
        if metadata.log_scale:
            # Handle zero/negative values for log scale
            normalized_data = np.where(data <= 0, np.nanmin(data[data > 0]) * 0.1, data)
            normalized_data = np.log10(normalized_data)

        if metadata.min_val is not None and metadata.max_val is not None:
            normalized_data = np.clip(
                normalized_data, metadata.min_val, metadata.max_val, dtype=np.float64
            )
        return typing.cast(ThreeDimensionalGrid, normalized_data)


class VolumeRenderer(Base3DPlotter):
    """3D Volume renderer for scalar fields."""

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        surface_count: int = 15,
    ) -> go.Figure:
        """
        Create a volume rendering plot.

        :param data: 3D data array to render
        :param metadata: Property metadata for labeling and scaling
        :param surface_count: Number of isosurfaces to generate for volume rendering
        :return: Plotly figure object with volume rendering
        """
        if data.ndim != 3:
            raise ValueError("Volume rendering requires 3D data")

        normalized_data = self.normalize_data(data, metadata)
        fig = go.Figure(
            data=go.Volume(
                x=np.arange(data.shape[0]),
                y=np.arange(data.shape[1]),
                z=np.arange(data.shape[2]),
                value=normalized_data.flatten(),
                opacity=self.config.opacity,
                surface_count=surface_count,
                colorscale=self.get_colorscale(metadata.color_scheme),
                showscale=self.config.show_colorbar,
                colorbar=dict(
                    title=f"{metadata.display_name} ({metadata.unit})",
                )
                if self.config.show_colorbar
                else None,
            )
        )
        self.update_layout(fig, metadata)
        return fig

    def update_layout(self, fig: go.Figure, metadata: PropertyMetadata):
        """
        Update figure layout with title, dimensions, and scene configuration.

        :param fig: Plotly figure to update
        :param metadata: Property metadata for title generation
        """
        fig.update_layout(
            title=self.config.title or f"3D Volume: {metadata.display_name}",
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title="X Index",
                yaxis_title="Y Index",
                zaxis_title="Z Index",
                camera=self.config.camera_position,
                aspectmode="cube",
            ),
        )


class IsosurfacePlotter(Base3DPlotter):
    """3D Isosurface plotter."""

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        isomin: typing.Optional[float] = None,
        isomax: typing.Optional[float] = None,
        surface_count: int = 3,
    ) -> go.Figure:
        """
        Create an isosurface plot.

        :param data: 3D data array to create isosurfaces from
        :param metadata: Property metadata for labeling and scaling
        :param isomin: Minimum isovalue (defaults to 10th percentile)
        :param isomax: Maximum isovalue (defaults to 90th percentile)
        :param surface_count: Number of isosurfaces to generate
        :return: Plotly figure object with isosurface plot
        """
        if data.ndim != 3:
            raise ValueError("Isosurface plotting requires 3D data")

        normalized_data = self.normalize_data(data, metadata)
        # Calculate isosurface value (default to median)
        if isomin is None:
            isomin = float(np.percentile(normalized_data, 10))
        if isomax is None:
            isomax = float(np.percentile(normalized_data, 90))

        fig = go.Figure(
            data=go.Isosurface(
                x=np.arange(data.shape[0]),
                y=np.arange(data.shape[1]),
                z=np.arange(data.shape[2]),
                value=normalized_data.flatten(),
                isomin=isomin,
                isomax=isomax,
                opacity=self.config.opacity * 2,  # Make isosurfaces more visible
                surface_count=surface_count,
                colorscale=self.get_colorscale(metadata.color_scheme),
                showscale=self.config.show_colorbar,
                colorbar=dict(
                    title=f"{metadata.display_name} ({metadata.unit})",
                )
                if self.config.show_colorbar
                else None,
            )
        )
        self.update_layout(fig, metadata)
        return fig

    def update_layout(self, fig: go.Figure, metadata: PropertyMetadata):
        """
        Update figure layout with title, dimensions, and scene configuration for isosurface plots.

        :param fig: Plotly figure to update
        :param metadata: Property metadata for title generation
        """
        fig.update_layout(
            title=self.config.title or f"3D Isosurface: {metadata.display_name}",
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title="X Index",
                yaxis_title="Y Index",
                zaxis_title="Z Index",
                camera=self.config.camera_position,
                aspectmode="cube",
            ),
        )


class SlicePlotter(Base3DPlotter):
    """3D Slice plane plotter."""

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        x_slice: typing.Optional[int] = None,
        y_slice: typing.Optional[int] = None,
        z_slice: typing.Optional[int] = None,
        show_x_slice: bool = True,
        show_y_slice: bool = True,
        show_z_slice: bool = True,
    ) -> go.Figure:
        """
        Create slice plane plots showing cross-sections through 3D data.

        :param data: 3D data array to slice through
        :param metadata: Property metadata for labeling and scaling
        :param x_slice: X-coordinate for YZ plane slice (defaults to middle)
        :param y_slice: Y-coordinate for XZ plane slice (defaults to middle)
        :param z_slice: Z-coordinate for XY plane slice (defaults to middle)
        :param show_x_slice: Whether to show the YZ plane slice
        :param show_y_slice: Whether to show the XZ plane slice
        :param show_z_slice: Whether to show the XY plane slice
        :return: Plotly figure object with slice plane plots
        """
        if data.ndim != 3:
            raise ValueError("Slice plotting requires 3D data")

        normalized_data = self.normalize_data(data, metadata)

        # Default slice positions
        if x_slice is None:
            x_slice = data.shape[0] // 2
        if y_slice is None:
            y_slice = data.shape[1] // 2
        if z_slice is None:
            z_slice = data.shape[2] // 2

        fig = go.Figure()

        # X slice (YZ plane)
        if show_x_slice:
            fig.add_trace(
                go.Surface(
                    x=np.full((data.shape[1], data.shape[2]), x_slice),
                    y=np.arange(data.shape[1])[:, np.newaxis],
                    z=np.arange(data.shape[2])[np.newaxis, :],
                    surfacecolor=normalized_data[x_slice, :, :],
                    colorscale=self.get_colorscale(metadata.color_scheme),
                    showscale=False,
                    name=f"X={x_slice}",
                )
            )

        # Y slice (XZ plane)
        if show_y_slice:
            fig.add_trace(
                go.Surface(
                    x=np.arange(data.shape[0])[:, np.newaxis],
                    y=np.full((data.shape[0], data.shape[2]), y_slice),
                    z=np.arange(data.shape[2])[np.newaxis, :],
                    surfacecolor=normalized_data[:, y_slice, :],
                    colorscale=self.get_colorscale(metadata.color_scheme),
                    showscale=False,
                    name=f"Y={y_slice}",
                )
            )

        # Z slice (XY plane)
        if show_z_slice:
            fig.add_trace(
                go.Surface(
                    x=np.arange(data.shape[0])[:, np.newaxis],
                    y=np.arange(data.shape[1])[np.newaxis, :],
                    z=np.full((data.shape[0], data.shape[1]), z_slice),
                    surfacecolor=normalized_data[:, :, z_slice],
                    colorscale=self.get_colorscale(metadata.color_scheme),
                    showscale=self.config.show_colorbar,
                    colorbar=dict(
                        title=f"{metadata.display_name} ({metadata.unit})",
                    )
                    if self.config.show_colorbar
                    else None,
                    name=f"Z={z_slice}",
                )
            )

        self.update_layout(fig, metadata)
        return fig

    def update_layout(self, fig: go.Figure, metadata: PropertyMetadata):
        """
        Update figure layout with title, dimensions, and scene configuration for slice plots.

        :param fig: Plotly figure to update
        :param metadata: Property metadata for title generation
        """
        fig.update_layout(
            title=self.config.title or f"3D Slices: {metadata.display_name}",
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title="X Index",
                yaxis_title="Y Index",
                zaxis_title="Z Index",
                camera=self.config.camera_position,
                aspectmode="cube",
            ),
        )


class Scatter3DPlotter(Base3DPlotter):
    """3D Scatter plotter for sparse data."""

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        threshold: typing.Optional[float] = None,
        sample_rate: float = 0.1,
        marker_size: int = 3,
    ) -> go.Figure:
        """
        Create a 3D scatter plot for sparse data visualization.

        :param data: 3D data array to create scatter plot from
        :param metadata: Property metadata for labeling and scaling
        :param threshold: Value threshold for point inclusion (defaults to 95th percentile)
        :param sample_rate: Fraction of points to sample for performance (0.0-1.0)
        :param marker_size: Size of scatter plot markers
        :return: Plotly figure object with 3D scatter plot
        """
        if data.ndim != 3:
            raise ValueError("Scatter3D plotting requires 3D data")

        normalized_data = self.normalize_data(data, metadata)

        # Subsample data for performance
        if threshold is None:
            threshold = float(np.percentile(normalized_data, 95))

        # Get indices where data exceeds threshold
        mask = normalized_data > threshold
        indices = np.where(mask)

        # Subsample
        n_points = len(indices[0])
        if n_points > 10000:  # Limit number of points for performance
            sample_indices = np.random.choice(
                n_points, int(n_points * sample_rate), replace=False
            )
            x_coords = indices[0][sample_indices]
            y_coords = indices[1][sample_indices]
            z_coords = indices[2][sample_indices]
            values = normalized_data[x_coords, y_coords, z_coords]
        else:
            x_coords, y_coords, z_coords = indices
            values = normalized_data[mask]

        fig = go.Figure(
            data=go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=values,
                    colorscale=self.get_colorscale(metadata.color_scheme),
                    opacity=self.config.opacity * 3,  # Make markers more visible
                    colorbar=dict(
                        title=f"{metadata.display_name} ({metadata.unit})",
                    )
                    if self.config.show_colorbar
                    else None,
                ),
                text=[f"{metadata.display_name}: {v:.3f}" for v in values],
                hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>%{text}<extra></extra>",
            )
        )
        self.update_layout(fig, metadata)
        return fig

    def update_layout(self, fig: go.Figure, metadata: PropertyMetadata):
        """
        Update figure layout with title, dimensions, and scene configuration for scatter plots.

        :param fig: Plotly figure to update
        :param metadata: Property metadata for title generation
        """
        fig.update_layout(
            title=self.config.title or f"3D Scatter: {metadata.display_name}",
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title="X Index",
                yaxis_title="Y Index",
                zaxis_title="Z Index",
                camera=self.config.camera_position,
                aspectmode="cube",
            ),
        )


# =====================================================================================
# MAIN VISUALIZATION FACTORY
# =====================================================================================


class ReservoirModelVisualizer3D:
    """
    3D visualizaer for reservoir model simulation results/data.

    This class provides a comprehensive suite of 3D plotting capabilities
    specifically designed for reservoir engineering workflows.
    """

    def __init__(
        self,
        config: typing.Optional[Plot3DConfig] = None,
        registry: typing.Optional[FluidPropertyRegistry] = None,
    ) -> None:
        """
        Initialize the visualizer with optional configuration.

        :param config: Optional configuration for 3D plotting (uses defaults if None)
        :param registry: Optional fluid property registry (uses default if None)
        """
        self.config = config or Plot3DConfig()
        self.plotters: typing.Dict[PlotType, Base3DPlotter] = {
            PlotType.VOLUME_RENDER: VolumeRenderer(self.config),
            PlotType.ISOSURFACE: IsosurfacePlotter(self.config),
            PlotType.SLICE_PLANES: SlicePlotter(self.config),
            PlotType.SCATTER_3D: Scatter3DPlotter(self.config),
        }
        self.registry = registry or fluid_property_registry

    def plot_property(
        self,
        model_state: ModelState[ThreeDimensions],
        property: str,
        plot_type: typing.Optional[PlotType] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot a specific fluid property in 3D.

        :param model_state: The model state containing the reservoir model data
        :param property_name: Name of the property to plot (from `FluidPropertyRegistry`)
        :param plot_type: Type of 3D plot to create (volume, isosurface, slice, scatter)
        :param kwargs: Additional plotting parameters specific to the plot type
        :return: Plotly figure object containing the 3D visualization
        """
        metadata = self.registry[property]
        data = self._get_property(model_state, metadata.name)
        plot_type = plot_type or self.config.plot_type
        plotter = self.plotters[plot_type]
        return plotter.plot(data, metadata, **kwargs)

    def _get_property(
        self, model_state: ModelState[ThreeDimensions], name: str
    ) -> ThreeDimensionalGrid:
        """
        Get property data from model state.

        :param model_state: The model state containing fluid properties
        :param name: Name of the property to extract as defined by `PropertyMetadata.name`
        :return: A three-dimensional numpy array containing the property data
        :raises AttributeError: If property is not found in fluid_properties
        :raises TypeError: If property is not a numpy array
        """
        # Access fluid properties through model
        fluid_properties = model_state.model.fluid_properties
        if not hasattr(fluid_properties, name):
            available_props = [
                attr for attr in dir(fluid_properties) if not attr.startswith("_")
            ]
            raise AttributeError(
                f"Property '{name}' not found in fluid_properties. "
                f"Available properties: {available_props}"
            )

        data = getattr(fluid_properties, name)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Property '{name}' is not a numpy array")
        return data

    def plot_properties(
        self,
        model_state: ModelState[ThreeDimensions],
        properties: typing.Sequence[str],
        plot_type: typing.Optional[PlotType] = None,
        subplot_columns: int = 2,
    ) -> go.Figure:
        """
        Create a subplot figure with multiple properties.

        :param model_state: The model state containing the reservoir data
        :param properties: Sequence of property names to plot
        :param plot_type: Type of 3D plot to create for each property
        :param subplot_columns: Number of columns in the subplot grid
        :return: Plotly figure with multiple property subplots
        """
        n_props = len(properties)
        subplot_rows = (n_props + subplot_columns - 1) // subplot_columns

        # Create subplots
        fig = make_subplots(
            rows=subplot_rows,
            cols=subplot_columns,
            specs=[
                [{"type": "scene"} for _ in range(subplot_columns)]
                for _ in range(subplot_rows)
            ],
            subplot_titles=[self.registry[prop].display_name for prop in properties],
        )

        for i, prop in enumerate(properties):
            row = (i // subplot_columns) + 1
            col = (i % subplot_columns) + 1

            # Get individual plot
            individual_fig = self.plot_property(model_state, prop, plot_type)

            # Add traces to subplot
            for trace in individual_fig.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            title="Multiple Reservoir Properties",
            width=self.config.width * subplot_columns,
            height=self.config.height * subplot_rows,
        )
        return fig

    def create_dashboard(
        self,
        model_state: ModelState[ThreeDimensions],
        properties: typing.Optional[typing.Sequence[str]] = None,
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with key reservoir properties.

        :param model_state: The model state containing the reservoir data
        :param properties: Sequence of key properties to display (uses defaults if None)
        :return: Dashboard figure with multiple reservoir property visualizations
        """
        if properties is None:
            properties = (
                "pressure",
                "oil_saturation",
                "water_saturation",
                "gas_saturation",
                "oil_viscosity",
                "water_viscosity",
            )
        return self.plot_properties(
            model_state,
            properties=properties,
            plot_type=PlotType.SLICE_PLANES,
            subplot_columns=3,
        )

    def animate_property(
        self,
        model_states: typing.Sequence[ModelState[ThreeDimensions]],
        property: str,
        plot_type: typing.Optional[PlotType] = None,
        frame_duration: int = 500,
    ) -> go.Figure:
        """
        Create an animated plot showing property evolution over time.

        :param model_states: Sequence of model states representing time steps
        :param property: Name of the property to animate
        :param plot_type: Type of 3D plot for animation frames
        :param frame_duration: Duration of each frame in milliseconds
        :return: Animated Plotly figure with time controls
        """
        if not model_states:
            raise ValueError("No model states provided")

        metadata = self.registry[property]
        plot_type = plot_type or PlotType.SLICE_PLANES

        # Create base figure from first state
        base_fig = self.plot_property(model_states[0], property, plot_type)

        # Create frames for animation
        frames = []
        for i, state in enumerate(model_states):
            frame_fig = self.plot_property(state, property, plot_type)
            frame = go.Frame(
                data=frame_fig.data,
                name=f"frame_{i}",
                layout=dict(
                    title=f"{metadata.display_name} at t={state.time / 3600:.2f} hours"
                    if state.time >= 600  # Only switch to hours after 10 minutes
                    else f"{metadata.display_name} at t={state.time:.2f} seconds",
                ),
            )
            frames.append(frame)

        # Add frames to figure
        base_fig.frames = frames

        # Add animation controls
        base_fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Time Step:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame_{i}"],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": f"{i}",
                            "method": "animate",
                        }
                        for i in range(len(model_states))
                    ],
                }
            ],
        )
        return base_fig


def plot_3d_property(
    model_state: ModelState[ThreeDimensions],
    property: str,
    plot_type: PlotType = PlotType.VOLUME_RENDER,
    config: typing.Optional[Plot3DConfig] = None,
    **kwargs,
) -> go.Figure:
    """
    Convenience function to quickly plot a 3D property.

    :param model_state: Model state containing reservoir data to plot
    :param property: Name of the property to visualize
    :param plot_type: Type of 3D plot to create
    :param config: Plot configuration settings
    :param kwargs: Additional plotting parameters specific to plot type
    :return: Plotly figure object with the 3D visualization
    """
    visualizer = ReservoirModelVisualizer3D(config)
    return visualizer.plot_property(model_state, property, plot_type, **kwargs)


def create_reservoir_dashboard(
    model_state: ModelState[ThreeDimensions],
    config: typing.Optional[Plot3DConfig] = None,
) -> go.Figure:
    """
    Convenience function to create a reservoir dashboard.

    :param model_state: Model state containing reservoir data to visualize
    :param config: Plot configuration settings
    :return: Dashboard figure with multiple key reservoir properties
    """
    visualizer = ReservoirModelVisualizer3D(config)
    return visualizer.create_dashboard(model_state)


def animate_3d_property(
    model_states: typing.List[ModelState[ThreeDimensions]],
    property: str,
    plot_type: PlotType = PlotType.SLICE_PLANES,
    config: typing.Optional[Plot3DConfig] = None,
) -> go.Figure:
    """
    Convenience function to animate a 3D property over time.

    :param model_states: List of model states representing time series data
    :param property: Name of the property to animate
    :param plot_type: Type of 3D plot for animation frames
    :param config: Plot configuration settings
    :return: Animated figure with time controls and slider
    """
    visualizer = ReservoirModelVisualizer3D(config)
    return visualizer.animate_property(model_states, property, plot_type)


# =====================================================================================
# LEGACY MATPLOTLIB FUNCTIONS (for backward compatibility)
# =====================================================================================


def plot_model_state(
    model_state: ModelState,
) -> None:
    """
    Legacy 2D matplotlib plotting function for backward compatibility.
    Creates a 2x2 grid of plots showing the reservoir pressure, oil saturation,
    water saturation, and oil viscosity distributions.

    :param model_state: Simulated reservoir model time state containing the grids
    """
    # Extract data from model state
    fluid_properties = model_state.model.fluid_properties

    pressure_grid = fluid_properties.pressure_grid
    oil_saturation_grid = fluid_properties.oil_saturation_grid
    water_saturation_grid = fluid_properties.water_saturation_grid
    oil_viscosity_grid = fluid_properties.oil_viscosity_grid

    total_time_in_hrs = model_state.time / 3600

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2x2 array of axes for easier iteration

    # For 3D data, take middle slice
    if pressure_grid.ndim == 3:
        z_mid = pressure_grid.shape[2] // 2
        pressure_slice = pressure_grid[:, :, z_mid]
        oil_saturation_slice = oil_saturation_grid[:, :, z_mid]
        water_saturation_slice = water_saturation_grid[:, :, z_mid]
        oil_viscosity_slice = oil_viscosity_grid[:, :, z_mid]
    else:
        pressure_slice = pressure_grid
        oil_saturation_slice = oil_saturation_grid
        water_saturation_slice = water_saturation_grid
        oil_viscosity_slice = oil_viscosity_grid

    # Pressure Plot
    pcm1 = axes[0].pcolormesh(pressure_slice.T, cmap="viridis", shading="auto")
    axes[0].set_title("Reservoir Pressure Distribution")
    axes[0].set_xlabel("X cell index")
    axes[0].set_ylabel("Y cell index")
    axes[0].set_aspect("equal")
    fig.colorbar(pcm1, ax=axes[0], label="Pressure (psi)")

    # Oil Saturation Plot
    pcm2 = axes[1].pcolormesh(
        oil_saturation_slice.T,
        cmap="plasma",
        shading="auto",
        norm=Normalize(vmin=0, vmax=1),
    )
    axes[1].set_title("Oil Saturation Distribution")
    axes[1].set_xlabel("X cell index")
    axes[1].set_ylabel("Y cell index")
    axes[1].set_aspect("equal")
    fig.colorbar(pcm2, ax=axes[1], label="Saturation")

    # Water Saturation Plot
    pcm3 = axes[2].pcolormesh(
        water_saturation_slice.T,
        cmap="cividis",
        shading="auto",
        norm=Normalize(vmin=0, vmax=1),
    )
    axes[2].set_title("Water Saturation Distribution")
    axes[2].set_xlabel("X cell index")
    axes[2].set_ylabel("Y cell index")
    axes[2].set_aspect("equal")
    fig.colorbar(pcm3, ax=axes[2], label="Saturation")

    # Oil Viscosity Plot
    viscosity_min = oil_viscosity_slice.min()
    viscosity_max = oil_viscosity_slice.max()

    pcm4 = axes[3].pcolormesh(
        oil_viscosity_slice.T,
        cmap="magma",
        shading="auto",
        norm=Normalize(vmin=viscosity_min, vmax=viscosity_max),
    )
    axes[3].set_title("Oil Viscosity Distribution")
    axes[3].set_xlabel("X cell index")
    axes[3].set_ylabel("Y cell index")
    axes[3].set_aspect("equal")
    fig.colorbar(pcm4, ax=axes[3], label="Viscosity (cP)")

    fig.suptitle(
        f"Reservoir Simulation at {total_time_in_hrs:.2f} hr(s)",
        fontsize=16,
    )
    plt.show()


def animate_model_states(
    model_states: typing.Sequence[ModelState],
    interval_ms: int = 100,
    save_path: typing.Optional[str] = None,
) -> None:
    """
    Legacy matplotlib animation function for backward compatibility.
    Animates the reservoir properties over multiple time states.

    :param model_states: A sequence of ModelState objects representing reservoir snapshots over time
    :param interval_ms: Delay between animation frames in milliseconds
    :param save_path: Optional file path to save the animation (supports .mp4 and .gif)
    """
    if not model_states:
        print("No model states provided for animation.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    initial_state = model_states[0]
    fluid_properties = initial_state.model.fluid_properties

    # Get all data for consistent color scaling
    all_pressures = np.array(
        [s.model.fluid_properties.pressure_grid for s in model_states]
    )
    all_oil_viscosities = np.array(
        [s.model.fluid_properties.oil_viscosity_grid for s in model_states]
    )

    min_pressure, max_pressure = all_pressures.min(), all_pressures.max()
    min_viscosity, max_viscosity = all_oil_viscosities.min(), all_oil_viscosities.max()

    # Handle 3D data by taking middle slice
    if fluid_properties.pressure_grid.ndim == 3:
        z_mid = fluid_properties.pressure_grid.shape[2] // 2
        pressure_slice = fluid_properties.pressure_grid[:, :, z_mid]
        oil_saturation_slice = fluid_properties.oil_saturation_grid[:, :, z_mid]
        water_saturation_slice = fluid_properties.water_saturation_grid[:, :, z_mid]
        oil_viscosity_slice = fluid_properties.oil_viscosity_grid[:, :, z_mid]
    else:
        pressure_slice = fluid_properties.pressure_grid
        oil_saturation_slice = fluid_properties.oil_saturation_grid
        water_saturation_slice = fluid_properties.water_saturation_grid
        oil_viscosity_slice = fluid_properties.oil_viscosity_grid

    # Initial plots
    pcm1 = axes[0].pcolormesh(
        pressure_slice.T,
        cmap="viridis",
        shading="auto",
        norm=Normalize(vmin=min_pressure, vmax=max_pressure),
    )
    axes[0].set_title("Reservoir Pressure Distribution")
    axes[0].set_xlabel("X cell index")
    axes[0].set_ylabel("Y cell index")
    axes[0].set_aspect("equal")
    fig.colorbar(pcm1, ax=axes[0], label="Pressure (psi)")

    pcm2 = axes[1].pcolormesh(
        oil_saturation_slice.T,
        cmap="plasma",
        shading="auto",
        norm=Normalize(vmin=0, vmax=1),
    )
    axes[1].set_title("Oil Saturation Distribution")
    axes[1].set_xlabel("X cell index")
    axes[1].set_ylabel("Y cell index")
    axes[1].set_aspect("equal")
    fig.colorbar(pcm2, ax=axes[1], label="Saturation")

    pcm3 = axes[2].pcolormesh(
        water_saturation_slice.T,
        cmap="cividis",
        shading="auto",
        norm=Normalize(vmin=0, vmax=1),
    )
    axes[2].set_title("Water Saturation Distribution")
    axes[2].set_xlabel("X cell index")
    axes[2].set_ylabel("Y cell index")
    axes[2].set_aspect("equal")
    fig.colorbar(pcm3, ax=axes[2], label="Saturation")

    pcm4 = axes[3].pcolormesh(
        oil_viscosity_slice.T,
        cmap="magma",
        shading="auto",
        norm=Normalize(vmin=min_viscosity, vmax=max_viscosity),
    )
    axes[3].set_title("Oil Viscosity Distribution")
    axes[3].set_xlabel("X cell index")
    axes[3].set_ylabel("Y cell index")
    axes[3].set_aspect("equal")
    fig.colorbar(pcm4, ax=axes[3], label="Viscosity (cP)")

    current_time_hrs = initial_state.time / 3600
    fig_suptitle = fig.suptitle(
        f"Reservoir Simulation at {current_time_hrs:.2f} hr(s)",
        fontsize=16,
    )

    def update(frame_index: int):
        """Updates the plot data for each frame of the animation."""
        state = model_states[frame_index]
        fluid_props = state.model.fluid_properties

        # Handle 3D data
        if fluid_props.pressure_grid.ndim == 3:
            z_mid = fluid_props.pressure_grid.shape[2] // 2
            pressure_data = fluid_props.pressure_grid[:, :, z_mid]
            oil_sat_data = fluid_props.oil_saturation_grid[:, :, z_mid]
            water_sat_data = fluid_props.water_saturation_grid[:, :, z_mid]
            oil_visc_data = fluid_props.oil_viscosity_grid[:, :, z_mid]
        else:
            pressure_data = fluid_props.pressure_grid
            oil_sat_data = fluid_props.oil_saturation_grid
            water_sat_data = fluid_props.water_saturation_grid
            oil_visc_data = fluid_props.oil_viscosity_grid

        # Update data for all four plots
        pcm1.set_array(pressure_data.T.ravel())
        pcm2.set_array(oil_sat_data.T.ravel())
        pcm3.set_array(water_sat_data.T.ravel())
        pcm4.set_array(oil_visc_data.T.ravel())

        # Update title
        current_time_hrs = state.time / 3600
        fig_suptitle.set_text(f"Reservoir Simulation at {current_time_hrs:.2f} hr(s)")

        return (pcm1, pcm2, pcm3, pcm4, fig_suptitle)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(model_states),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        Writer = (
            animation.writers["ffmpeg"]
            if save_path.endswith(".mp4")
            else animation.writers["pillow"]
        )
        writer = Writer(
            fps=1000 // interval_ms, metadata=dict(artist="sim3D"), bitrate=1800
        )
        anim.save(save_path, writer=writer, dpi=100)
        print("Animation saved.")

    plt.show()
    plt.close(fig)
