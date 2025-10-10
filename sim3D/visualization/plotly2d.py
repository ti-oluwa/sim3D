"""
Plotly-based 2D Visualization Suite for Reservoir Simulation Data and Results.
"""

from abc import ABC, abstractmethod
import collections.abc
from enum import Enum
import typing

import attrs
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sim3D.types import TwoDimensionalGrid
from sim3D.visualization.base import ColorScheme, PropertyMetadata


@attrs.define(slots=True, frozen=True)
class PlotConfig:
    """Configuration for 2D plots."""

    # Dimensions
    width: int = 800
    """Figure width in pixels"""

    height: int = 600
    """Figure height in pixels"""

    # Display options
    show_colorbar: bool = True
    """Whether to display color scale bar"""

    title: typing.Optional[str] = None
    """Optional title for plots"""

    # Styling
    color_scheme: str = "viridis"
    """Default colorscale for plots (viridis, plasma, inferno, etc.)"""

    opacity: float = 0.8
    """Default opacity for plot elements"""

    # Grid and axes
    show_grid: bool = True
    """Whether to show grid lines"""

    grid_color: str = "lightgray"
    """Color of grid lines"""

    axis_line_color: str = "black"
    """Color of axis lines"""

    axis_line_width: float = 1.0
    """Width of axis lines"""

    # Text and labels
    font_family: str = "Arial, sans-serif"
    """Font family for text elements"""

    font_size: int = 12
    """Default font size"""

    title_font_size: int = 16
    """Font size for titles"""

    axis_title_font_size: int = 14
    """Font size for axis titles"""

    # Colorbar settings
    colorbar_thickness: int = 20
    """Thickness of colorbar in pixels"""

    colorbar_len: float = 0.8
    """Length of colorbar as fraction of plot height"""

    # Plot-specific defaults
    contour_line_width: float = 1.5
    """Default width for contour lines"""

    contour_levels: int = 20
    """Default number of contour levels"""

    scatter_marker_size: int = 6
    """Default size for scatter plot markers"""

    line_width: float = 2.0
    """Default width for line plots"""

    # Layout margins
    margin_left: int = 80
    """Left margin in pixels"""

    margin_right: int = 80
    """Right margin in pixels"""

    margin_top: int = 80
    """Top margin in pixels"""

    margin_bottom: int = 80
    """Bottom margin in pixels"""

    # Background
    plot_bgcolor: str = "white"
    """Background color of plot area"""

    paper_bgcolor: str = "white"
    """Background color of entire figure"""


class PlotType(str, Enum):
    """Enumeration of available 2D plot types."""

    HEATMAP = "heatmap"
    CONTOUR = "contour"
    CONTOUR_FILLED = "contour_filled"
    SCATTER = "scatter"
    LINE = "line"
    SURFACE = "surface"


class BaseRenderer(ABC):
    """
    Abstract base class for 2D renderers.

    All 2D renderers must implement the render method to create
    Plotly visualizations for 2D reservoir data.
    """

    def __init__(self, config: PlotConfig) -> None:
        """
        Initialize the renderer with configuration.

        :param config: Plot configuration settings
        """
        self.config = config

    @abstractmethod
    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMetadata,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render the 2D data onto the provided figure.

        :param figure: Plotly figure to add the plot to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array (defaults to indices)
        :param y_coords: Optional y-coordinate array (defaults to indices)
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param kwargs: Additional plotting parameters
        :return: Updated Plotly figure
        """
        pass

    def normalize_data(
        self,
        data: TwoDimensionalGrid,
        metadata: PropertyMetadata,
        normalize_range: bool = True,
    ) -> typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid]:
        """
        Normalize data for consistent visualization.

        :param data: Input 2D data array
        :param metadata: Property metadata containing scaling information
        :param normalize_range: Whether to normalize to 0-1 range
        :return: Tuple of (normalized_data, display_data)
        """
        display_data = data.copy()
        if metadata.log_scale:
            # Apply log transformation, handling zeros and negatives
            display_data = np.where(display_data > 0, display_data, np.nan)
            display_data = np.log10(display_data)

        normalized_data = display_data.copy()
        if normalize_range:
            data_min = np.nanmin(display_data)
            data_max = np.nanmax(display_data)
            if data_max > data_min:
                normalized_data = (display_data - data_min) / (data_max - data_min)
            else:
                normalized_data = np.zeros_like(display_data)

        return typing.cast(TwoDimensionalGrid, normalized_data), typing.cast(
            TwoDimensionalGrid, display_data
        )

    def get_colorscale(self, color_scheme: typing.Optional[str] = None) -> str:
        """
        Get the appropriate colorscale for the plot.

        :param color_scheme: Optional color scheme override
        :return: Plotly colorscale name
        """
        if color_scheme:
            return ColorScheme(color_scheme).value
        return ColorScheme(self.config.color_scheme).value or "viridis"

    def format_value(
        self, value: float, metadata: PropertyMetadata, precision: int = 3
    ) -> str:
        """
        Format a data value for display.

        :param value: Value to format
        :param metadata: Property metadata for formatting context
        :param precision: Number of decimal places
        :return: Formatted string
        """
        if np.isnan(value):
            return "NaN"

        if metadata.log_scale:
            # Convert back from log space for display
            original_value = 10**value
            return f"{original_value:.{precision}g}"
        else:
            return f"{value:.{precision}g}"


class HeatmapRenderer(BaseRenderer):
    """2D Heatmap renderer using Plotly's heatmap trace."""

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMetadata,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a heatmap visualization.

        :param figure: Plotly figure to add the heatmap to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param cmin: Minimum value for color mapping
        :param cmax: Maximum value for color mapping
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValueError("Heatmap plotting requires 2D data")

        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        # Create hover text
        hover_text = []
        for i in range(data.shape[0]):
            hover_row = []
            for j in range(data.shape[1]):
                hover_row.append(
                    f"{x_label}: {x_coords[j]:.2f}<br>"
                    f"{y_label}: {y_coords[i]:.2f}<br>"
                    f"{metadata.display_name}: {self.format_value(display_data[i, j], metadata)} {metadata.unit}"
                )
            hover_text.append(hover_row)

        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        figure.add_trace(
            go.Heatmap(
                z=display_data,
                x=x_coords,
                y=y_coords,
                colorscale=colorscale,
                zmin=cmin,
                zmax=cmax,
                showscale=self.config.show_colorbar,
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                colorbar=dict(
                    title=f"{metadata.display_name} ({metadata.unit})"
                    + (" - Log Scale" if metadata.log_scale else "")
                )
                if self.config.show_colorbar
                else None,
            )
        )
        self.update_layout(figure, x_label, y_label, metadata.display_name)
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for heatmap plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=title,
        )


class ContourRenderer(BaseRenderer):
    """2D Contour renderer for contour and filled contour plots."""

    def __init__(self, config: PlotConfig, filled: bool = False) -> None:
        """
        Initialize contour renderer.

        :param config: Plot configuration
        :param filled: Whether to create filled contours
        """
        super().__init__(config)
        self.filled = filled

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMetadata,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        contour_levels: typing.Optional[int] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a contour visualization.

        :param figure: Plotly figure to add the contour to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param contour_levels: Number of contour levels
        :param cmin: Minimum value for color mapping
        :param cmax: Maximum value for color mapping
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValueError("Contour plotting requires 2D data")

        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        levels = contour_levels or 20

        # Calculate proper contour levels
        data_min = cmin if cmin is not None else np.nanmin(display_data)
        data_max = cmax if cmax is not None else np.nanmax(display_data)
        level_step = (data_max - data_min) / levels if data_max > data_min else 1.0

        contour_trace = go.Contour(
            z=display_data,
            x=x_coords,
            y=y_coords,
            colorscale=colorscale,
            zmin=data_min,
            zmax=data_max,
            showscale=self.config.show_colorbar,
            contours=dict(
                start=data_min,
                end=data_max,
                size=level_step,
                showlabels=True,
                labelfont=dict(size=10),
            ),
            line=dict(width=self.config.contour_line_width),
            colorbar=dict(
                title=f"{metadata.display_name} ({metadata.unit})"
                + (" - Log Scale" if metadata.log_scale else "")
            )
            if self.config.show_colorbar
            else None,
        )

        if self.filled:
            contour_trace.update(contours_coloring="fill")

        figure.add_trace(contour_trace)
        self.update_layout(figure, x_label, y_label, metadata.display_name)
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for contour plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=title,
        )


class ScatterRenderer(BaseRenderer):
    """2D Scatter renderer for sparse data visualization."""

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMetadata,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        threshold: float = 0.0,
        marker_size: int = 6,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a scatter plot visualization.

        :param figure: Plotly figure to add the scatter plot to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param threshold: Minimum value threshold for point inclusion
        :param marker_size: Size of scatter markers
        :param cmin: Minimum value for color mapping
        :param cmax: Maximum value for color mapping
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValueError("Scatter plotting requires 2D data")

        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        # Create coordinate meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)

        # Apply threshold filter
        mask = display_data > threshold
        if not np.any(mask):
            raise ValueError("No data points above threshold for scatter plot")

        x_scatter = X[mask]
        y_scatter = Y[mask]
        values = display_data[mask]

        # Create hover text
        hover_text = [
            f"{x_label}: {x:.2f}<br>"
            f"{y_label}: {y:.2f}<br>"
            f"{metadata.display_name}: {self.format_value(v, metadata)} {metadata.unit}"
            for x, y, v in zip(x_scatter, y_scatter, values)
        ]

        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )

        figure.add_trace(
            go.Scatter(
                x=x_scatter,
                y=y_scatter,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=values,
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    opacity=self.config.opacity,
                    colorbar=dict(
                        title=f"{metadata.display_name} ({metadata.unit})"
                        + (" - Log Scale" if metadata.log_scale else "")
                    )
                    if self.config.show_colorbar
                    else None,
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        self.update_layout(figure, x_label, y_label, metadata.display_name)
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for scatter plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=title,
        )


class LineRenderer(BaseRenderer):
    """2D Line renderer for line plots and cross-sections."""

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMetadata,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        line_mode: typing.Literal["horizontal", "vertical", "both"] = "horizontal",
        line_indices: typing.Optional[typing.Union[int, typing.List[int]]] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render line plots (cross-sections) of 2D data.

        :param figure: Plotly figure to add the line plots to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param line_mode: Direction of lines to plot
        :param line_indices: Specific indices to plot (defaults to middle)
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValueError("Line plotting requires 2D data")

        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        # Determine line indices
        if line_indices is None:
            line_indices = [data.shape[0] // 2, data.shape[1] // 2]
        elif isinstance(line_indices, int):
            line_indices = [line_indices]

        # Plot horizontal lines (constant y, varying x)
        if line_mode in ["horizontal", "both"]:
            if len(line_indices) >= 1 and 0 <= line_indices[0] < data.shape[0]:
                idx = line_indices[0]
                figure.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=display_data[idx, :],
                        mode="lines+markers",
                        name=f"{y_label}={y_coords[idx]:.2f}",
                        line=dict(width=self.config.line_width),
                        marker=dict(
                            size=8,
                            symbol="circle",
                            line=dict(width=1, color="white"),
                        ),
                    )
                )

        # Plot vertical lines (constant x, varying y)
        if line_mode in ["vertical", "both"]:
            if len(line_indices) >= 2 and 0 <= line_indices[1] < data.shape[1]:
                idx = line_indices[1]
                figure.add_trace(
                    go.Scatter(
                        x=y_coords,
                        y=display_data[:, idx],
                        mode="lines+markers",
                        name=f"{x_label}={x_coords[idx]:.2f}",
                        line=dict(width=self.config.line_width),
                        marker=dict(
                            size=8,
                            symbol="circle",
                            line=dict(width=1, color="white"),
                        ),
                    )
                )
            elif (
                line_mode == "vertical"
                and len(line_indices) >= 1
                and 0 <= line_indices[0] < data.shape[1]
            ):
                idx = line_indices[0]
                figure.add_trace(
                    go.Scatter(
                        x=y_coords,
                        y=display_data[:, idx],
                        mode="lines+markers",
                        name=f"{x_label}={x_coords[idx]:.2f}",
                        line=dict(width=self.config.line_width),
                        marker=dict(
                            size=8,
                            symbol="circle",
                            line=dict(width=1, color="white"),
                        ),
                    )
                )

        self.update_layout(
            figure,
            x_label,
            f"{metadata.display_name} ({metadata.unit})",
            metadata.display_name,
        )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for line plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            title=title,
            showlegend=True,
        )


class SurfaceRenderer(BaseRenderer):
    """3D Surface renderer for 2D data."""

    def render(
        self,
        figure: go.Figure,
        data: TwoDimensionalGrid,
        metadata: PropertyMetadata,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a 3D surface plot of 2D data.

        :param figure: Plotly figure to add the surface to
        :param data: 2D data array to visualize
        :param metadata: Property metadata for labeling and scaling
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param cmin: Minimum value for color mapping
        :param cmax: Maximum value for color mapping
        :return: Updated figure
        """
        if data.ndim != 2:
            raise ValueError("Surface plotting requires 2D data")

        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Handle coordinate arrays
        if x_coords is None:
            x_coords = np.arange(data.shape[1])
        if y_coords is None:
            y_coords = np.arange(data.shape[0])

        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        figure.add_trace(
            go.Surface(
                z=display_data,
                x=x_coords,
                y=y_coords,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=self.config.show_colorbar,
                colorbar=dict(
                    title=f"{metadata.display_name} ({metadata.unit})"
                    + (" - Log Scale" if metadata.log_scale else "")
                )
                if self.config.show_colorbar
                else None,
            )
        )
        self.update_layout(figure, x_label, y_label, metadata.display_name)
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """Update figure layout for surface plots."""
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=title,
                aspectmode="auto",
            ),
            title=title,
        )


PLOT_TYPE_NAMES: typing.Dict[PlotType, str] = {
    PlotType.HEATMAP: "Heatmap",
    PlotType.CONTOUR: "Contour",
    PlotType.CONTOUR_FILLED: "Filled Contour",
    PlotType.SCATTER: "Scatter",
    PlotType.LINE: "Line Plot",
    PlotType.SURFACE: "3D Surface",
}


class DataVisualizer:
    """
    2D visualizer for two-dimensional (reservoir) data.
    """

    def __init__(self, config: typing.Optional[PlotConfig] = None) -> None:
        """
        Initialize the visualizer with optional configuration.

        :param config: Optional configuration for 2D rendering (uses defaults if None)
        """
        self.config = config or PlotConfig()
        self._renderers: typing.Dict[PlotType, BaseRenderer] = {
            PlotType.HEATMAP: HeatmapRenderer(self.config),
            PlotType.CONTOUR: ContourRenderer(self.config, filled=False),
            PlotType.CONTOUR_FILLED: ContourRenderer(self.config, filled=True),
            PlotType.SCATTER: ScatterRenderer(self.config),
            PlotType.LINE: LineRenderer(self.config),
            PlotType.SURFACE: SurfaceRenderer(self.config),
        }

    def add_renderer(
        self,
        plot_type: PlotType,
        renderer_type: typing.Type[BaseRenderer],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """
        Add a custom renderer for a specific plot type.

        :param plot_type: The type of plot to add
        :param renderer_type: The class implementing the BaseRenderer interface
        :param args: Initialization arguments for the renderer class
        :param kwargs: Initialization keyword arguments for the renderer class
        """
        if not isinstance(plot_type, PlotType):
            raise ValueError(f"Invalid plot type: {plot_type}")
        self._renderers[plot_type] = renderer_type(self.config, *args, **kwargs)

    def get_renderer(self, plot_type: PlotType) -> BaseRenderer:
        """
        Get the renderer for a specific plot type.

        :param plot_type: The type of plot to get
        :return: The renderer instance for the specified plot type
        """
        if not isinstance(plot_type, PlotType):
            raise ValueError(f"Invalid plot type: {plot_type}")
        renderer = self._renderers.get(plot_type, None)
        if renderer is None:
            raise ValueError(f"No renderer registered for plot type: {plot_type}")
        return renderer

    def make_plot(
        self,
        data: typing.Union[TwoDimensionalGrid, np.typing.NDArray[np.floating]],
        plot_type: PlotType = PlotType.HEATMAP,
        metadata: typing.Optional[PropertyMetadata] = None,
        figure: typing.Optional[go.Figure] = None,
        title: typing.Optional[str] = None,
        x_coords: typing.Optional[np.typing.NDArray] = None,
        y_coords: typing.Optional[np.typing.NDArray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        as_series: bool = False,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create a 2D plot of the provided data.

        :param data: 2D numpy array to visualize
        :param plot_type: Type of 2D plot to create
        :param metadata: Optional property metadata for labeling and scaling
        :param figure: Optional existing figure to add to
        :param title: Custom title for the plot
        :param x_coords: Optional x-coordinate array (defaults to indices)
        :param y_coords: Optional y-coordinate array (defaults to indices)
        :param x_label: Label for the x-axis
        :param y_label: Label for the y-axis
        :param width: Custom width for the plot
        :param height: Custom height for the plot
        :param as_series: If True, treat 2D data as time-series (x,y) pairs
        :param kwargs: Additional plotting parameters specific to the plot type
        :return: Plotly figure object containing the 2D visualization
        """
        if data.ndim != 2:
            raise ValueError("2D visualization requires 2D data array")

        data = typing.cast(TwoDimensionalGrid, data)

        # Handle series data conversion
        if as_series:
            if data.shape[1] != 2:
                raise ValueError(
                    "Series mode requires data with shape (n, 2) for (x, y) pairs"
                )

            # Extract x and y values from the data
            x_values = data[:, 0]  # First column = x coordinates
            y_values = data[:, 1]  # Second column = y values

            # Override x_coords with extracted x values
            x_coords = x_values

            # Reshape y_values to be a 1xN array for line plotting
            data = y_values.reshape(1, -1)
            # Force plot_type to LINE for series data
            if plot_type == PlotType.LINE:
                import warnings

                warnings.warn(
                    f"Plot type {plot_type} not suitable for series data. Using LINE instead.",
                    UserWarning,
                )
                plot_type = PlotType.LINE

            # Set default line parameters for series
            kwargs.setdefault("line_mode", "horizontal")
            kwargs.setdefault("line_indices", [0])

        # Create default metadata if not provided
        if metadata is None:
            metadata = PropertyMetadata(
                name="data",
                display_name="Data",
                unit="",
                log_scale=False,
                color_scheme=ColorScheme.VIRIDIS,
            )

        # Get renderer and create plot
        renderer = self.get_renderer(plot_type)
        fig = figure or go.Figure()
        fig = renderer.render(
            fig,
            data,
            metadata,
            x_coords=x_coords,
            y_coords=y_coords,
            x_label=x_label,
            y_label=y_label,
            **kwargs,
        )

        # Apply final layout updates
        layout_updates: typing.Dict[str, typing.Any] = {}

        if title:
            layout_updates["title"] = title
        elif not title and self.config.title:
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "2D Plot")
            layout_updates["title"] = (
                f"{self.config.title}<br><sub>{plot_name}: {metadata.display_name}</sub>"
            )
        elif not title:
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "2D Plot")
            layout_updates["title"] = f"{plot_name}: {metadata.display_name}"

        if width is not None:
            layout_updates["width"] = width
        if height is not None:
            layout_updates["height"] = height

        if layout_updates:
            fig.update_layout(**layout_updates)

        return fig

    def make_plots(
        self,
        data_list: typing.Sequence[
            typing.Union[TwoDimensionalGrid, np.typing.NDArray[np.floating]]
        ],
        plot_types: typing.Union[PlotType, typing.Sequence[PlotType]],
        metadata_list: typing.Optional[typing.Sequence[PropertyMetadata]] = None,
        titles: typing.Optional[typing.Sequence[str]] = None,
        rows: int = 1,
        cols: int = 1,
        x_coords: typing.Optional[np.ndarray] = None,
        y_coords: typing.Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        subplot_titles: typing.Optional[typing.Sequence[str]] = None,
        as_series: bool = False,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create subplots with multiple 2D visualizations.

        :param data_list: Sequence of 2D data arrays to visualize
        :param plot_types: Plot type(s) to use (single type or list)
        :param metadata_list: Optional list of property metadata
        :param titles: Optional list of plot titles
        :param rows: Number of subplot rows
        :param cols: Number of subplot columns
        :param x_coords: Optional x-coordinate array
        :param y_coords: Optional y-coordinate array
        :param x_label: Label for x-axes
        :param y_label: Label for y-axes
        :param subplot_titles: Optional titles for each subplot
        :param as_series: If True, treat 2D data as time-series (x,y) pairs
        :param kwargs: Additional plotting parameters
        :return: Plotly figure with subplots
        """
        if len(data_list) > rows * cols:
            raise ValueError(
                f"Too many plots ({len(data_list)}) for {rows}x{cols} subplot grid"
            )

        # Handle plot types
        if isinstance(plot_types, PlotType):
            plot_types = [plot_types] * len(data_list)
        elif len(plot_types) != len(data_list):
            raise ValueError("Number of plot types must match number of data arrays")

        # Create subplot figure
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
        )

        # Add each plot to its subplot
        for idx, (data, plot_type) in enumerate(zip(data_list, plot_types)):
            row = (idx // cols) + 1
            col = (idx % cols) + 1

            if data.ndim != 2:
                raise ValueError("2D visualization requires 2D data array")
            data = typing.cast(TwoDimensionalGrid, data)

            # Handle series data conversion for this subplot
            current_x_coords = x_coords
            if as_series:
                if data.shape[1] != 2:
                    raise ValueError(
                        f"Series mode requires data with shape (n, 2) for (x, y) pairs. "
                        f"Data {idx} has shape {data.shape}"
                    )

                # Extract x and y values from the data
                x_values = data[:, 0]  # First column = x coordinates
                y_values = data[:, 1]  # Second column = y values

                # Override x_coords with extracted x values for this subplot
                current_x_coords = x_values

                # Reshape y_values to be a 1xN array for line plotting
                data = y_values.reshape(1, -1)

                # Force plot_type to LINE for series data
                if plot_type == PlotType.LINE:
                    import warnings

                    warnings.warn(
                        f"Plot type {plot_type} not suitable for series data. Using LINE instead for subplot {idx}.",
                        UserWarning,
                    )
                    plot_type = PlotType.LINE

            # Create temporary figure for this subplot
            temp_fig = go.Figure()

            # Get metadata
            metadata = (
                metadata_list[idx]
                if metadata_list
                else PropertyMetadata(
                    name=f"data_{idx}",
                    display_name=f"Data {idx}",
                    unit="",
                    log_scale=False,
                    color_scheme=ColorScheme.VIRIDIS,
                )
            )

            # Prepare kwargs for this subplot
            subplot_kwargs = kwargs.copy()
            if as_series:
                subplot_kwargs.setdefault("line_mode", "horizontal")
                subplot_kwargs.setdefault("line_indices", [0])

            # Render on temporary figure
            renderer = self.get_renderer(plot_type)
            temp_fig = renderer.render(
                temp_fig,
                data,
                metadata,
                x_coords=current_x_coords,
                y_coords=y_coords,
                x_label=x_label,
                y_label=y_label,
                **subplot_kwargs,
            )

            # Add traces to main figure
            for trace in temp_fig.data:
                fig.add_trace(trace, row=row, col=col)

        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            title=self.config.title or "2D Visualization Subplots",
        )
        return fig


viz = DataVisualizer()
"""Default 2D visualizer instance for reservoir models."""


def make_series_plot(
    data: typing.Union[
        TwoDimensionalGrid,
        np.typing.NDArray[np.floating],
        typing.Sequence[
            typing.Union[TwoDimensionalGrid, np.typing.NDArray[np.floating]]
        ],
        typing.Mapping[
            str, typing.Union[TwoDimensionalGrid, np.typing.NDArray[np.floating]]
        ],
    ],
    title: str = "Series Plot",
    x_label: str = "X",
    y_label: str = "Y",
    line_colors: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
    line_widths: typing.Optional[typing.Union[float, typing.Sequence[float]]] = None,
    line_styles: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
    show_markers: bool = True,
    marker_sizes: typing.Optional[typing.Union[int, typing.Sequence[int]]] = None,
    marker_symbols: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
    marker_colors: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
    mode: typing.Literal["lines", "markers", "lines+markers"] = "lines+markers",
    series_names: typing.Optional[typing.Sequence[str]] = None,
    width: int = 800,
    height: int = 600,
    show_legend: bool = True,
    legend_position: typing.Literal["top", "bottom", "left", "right"] = "right",
    grid: bool = True,
    background_color: str = "white",
    plot_background_color: str = "white",
    x_range: typing.Optional[typing.Tuple[float, float]] = None,
    y_range: typing.Optional[typing.Tuple[float, float]] = None,
    log_x: bool = False,
    log_y: bool = False,
    show_hover: bool = True,
    hover_template: typing.Optional[str] = None,
    fill_area: typing.Optional[typing.Union[bool, typing.Sequence[bool]]] = None,
    fill_colors: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
    fill_opacity: float = 0.3,
    **kwargs: typing.Any,
) -> go.Figure:
    """
    Create a series plot (line plot) for time-series data.

    Supports single or multiple series with extensive customization options
    including colors, markers, line styles, fills, and interactive features.

    :param data: Series data in various formats:
        - Single 2D array: shape (n, 2) for (x, y) pairs
        - List of 2D arrays: multiple series
        - Mapping of 2D arrays: {series_name: data_array}
    :param title: Title for the plot
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param line_colors: Color(s) for lines (single color or list per series)
    :param line_widths: Width(s) for lines (single width or list per series)
    :param line_styles: Style(s) for lines ('solid', 'dash', 'dot', 'dashdot')
    :param show_markers: Whether to show markers on data points
    :param marker_sizes: Size(s) for markers (single size or list per series)
    :param marker_symbols: Symbol(s) for markers ('circle', 'square', 'diamond', etc.)
    :param marker_colors: Color(s) for markers (can differ from line colors)
    :param mode: Plot mode ('lines', 'markers', or 'lines+markers')
    :param series_names: Names for each series (for legend)
    :param width: Figure width in pixels
    :param height: Figure height in pixels
    :param show_legend: Whether to show the legend
    :param legend_position: Position of the legend
    :param grid: Whether to show grid lines
    :param background_color: Background color of the entire figure
    :param plot_background_color: Background color of the plot area
    :param x_range: Custom x-axis range as (min, max)
    :param y_range: Custom y-axis range as (min, max)
    :param log_x: Whether to use logarithmic x-axis
    :param log_y: Whether to use logarithmic y-axis
    :param show_hover: Whether to show hover information
    :param hover_template: Custom hover template
    :param fill_area: Whether to fill area under curves
    :param fill_colors: Colors for filled areas
    :param fill_opacity: Opacity for filled areas
    :param kwargs: Additional parameters passed to go.Scatter
    :return: Plotly figure for the series plot

    Usage Examples:

    ```python
    # Single series
    data = np.array([(0, 100), (1, 150), (2, 120)])
    fig = make_series_plot(data, title="Production", x_label="Time", y_label="Rate")

    # Multiple series as list
    production = np.array([(0, 100), (1, 150), (2, 120)])
    injection = np.array([(0, 50), (1, 75), (2, 60)])
    fig = make_series_plot(
        [production, injection],
        title="Well Performance",
        series_names=["Production", "Injection"],
        line_colors=["blue", "red"],
        fill_area=[True, False]
    )

    # Multiple series as dict
    data_dict = {
        "Production": production,
        "Injection": injection
    }
    fig = make_series_plot(data_dict, title="Well Performance")
    ```
    """
    series_list = []
    names_list = []

    if isinstance(data, collections.abc.Mapping):
        # Mapping input: {name: data_array}
        for name, series_data in data.items():
            if hasattr(series_data, "ndim") and series_data.ndim != 2:
                raise ValueError(f"Series '{name}' requires 2D data array")
            if hasattr(series_data, "shape") and series_data.shape[1] != 2:
                raise ValueError(
                    f"Series '{name}' requires shape (n, 2) for (x, y) pairs"
                )
            series_list.append(typing.cast(TwoDimensionalGrid, series_data))
            names_list.append(name)
    elif isinstance(data, collections.abc.Sequence):
        # Sequence input: [data_array1, data_array2, ...]
        for i, series_data in enumerate(data):
            if hasattr(series_data, "ndim") and series_data.ndim != 2:
                raise ValueError(f"Series {i} requires 2D data array")
            if hasattr(series_data, "shape") and series_data.shape[1] != 2:
                raise ValueError(f"Series {i} requires shape (n, 2) for (x, y) pairs")
            series_list.append(typing.cast(TwoDimensionalGrid, series_data))
            names_list.append(f"Series {i + 1}")
    else:
        # Single array input
        if hasattr(data, "ndim") and data.ndim != 2:  # type: ignore[attr-defined]
            raise ValueError("Series data requires 2D data array")
        if hasattr(data, "shape") and data.shape[1] != 2:  # type: ignore[attr-defined]
            raise ValueError("Series data requires shape (n, 2) for (x, y) pairs")
        series_list.append(typing.cast(TwoDimensionalGrid, data))
        names_list.append("Series")

    num_series = len(series_list)

    # Override with user-provided names if available
    if series_names is not None:
        if len(series_names) != num_series:
            raise ValueError(
                f"Number of series names ({len(series_names)}) must match number of series ({num_series})"
            )
        names_list = series_names

    # Helper function to normalize parameters
    def normalize_param(param, default_value, param_name):
        if param is None:
            return [default_value] * num_series
        elif isinstance(param, (str, int, float, bool)):
            return [param] * num_series
        elif isinstance(param, list):
            if len(param) != num_series:
                raise ValueError(
                    f"Length of {param_name} ({len(param)}) must match number of series ({num_series})"
                )
            return param
        else:
            raise ValueError(f"Invalid type for {param_name}")

    # Normalize all parameters
    colors = normalize_param(line_colors, None, "line_colors")
    widths = normalize_param(line_widths, 2.0, "line_widths")
    styles = normalize_param(line_styles, "solid", "line_styles")
    m_sizes = normalize_param(marker_sizes, 8, "marker_sizes")
    m_symbols = normalize_param(marker_symbols, "circle", "marker_symbols")
    m_colors = normalize_param(marker_colors, None, "marker_colors")
    fills = normalize_param(fill_area, False, "fill_area")
    f_colors = normalize_param(fill_colors, None, "fill_colors")

    # Default color palette if not specified
    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    fig = go.Figure()

    # Add each series to the figure
    for i, (series_data, name) in enumerate(zip(series_list, names_list)):
        x_values = series_data[:, 0]
        y_values = series_data[:, 1]

        # Determine colors
        line_color = (
            colors[i]
            if colors[i] is not None
            else default_colors[i % len(default_colors)]
        )
        line_color = typing.cast(str, line_color)
        marker_color = m_colors[i] if m_colors[i] is not None else line_color
        fill_color = f_colors[i] if f_colors[i] is not None else line_color
        fill_color = typing.cast(str, fill_color)

        # Configure line style
        line_dash = {
            "solid": "solid",
            "dash": "dash",
            "dot": "dot",
            "dashdot": "dashdot",
        }.get(styles[i], "solid")

        # Configure fill
        fill_mode = None
        if fills[i]:
            fill_mode = "tonexty" if i > 0 else "tozeroy"

        # Create hover template
        if hover_template is None:
            custom_hover = (
                f"<b>{name}</b><br>"
                f"{x_label}: %{{x}}<br>"
                f"{y_label}: %{{y}}<extra></extra>"
            )
        else:
            custom_hover = hover_template

        # Add trace
        trace = go.Scatter(
            x=x_values,
            y=y_values,
            mode=mode,
            name=name,
            line=dict(color=line_color, width=widths[i], dash=line_dash),
            marker=dict(
                color=marker_color,
                size=m_sizes[i],
                symbol=m_symbols[i],
                line=dict(width=1, color="white") if show_markers else None,
            )
            if show_markers or "markers" in mode
            else None,
            fill=fill_mode,
            fillcolor=f"rgba({int(fill_color[1:3], 16)}, {int(fill_color[3:5], 16)}, {int(fill_color[5:7], 16)}, {fill_opacity})"
            if fills[i] and fill_color.startswith("#")
            else None,
            hovertemplate=custom_hover if show_hover else None,
            hoverinfo="skip" if not show_hover else None,
            **kwargs,
        )
        fig.add_trace(trace)

    # Configure layout
    legend_config = dict(
        orientation="v" if legend_position in ["left", "right"] else "h",
        x=1.02
        if legend_position == "right"
        else (0 if legend_position == "left" else 0.5),
        y=0.5
        if legend_position in ["left", "right"]
        else (1.02 if legend_position == "top" else -0.1),
        xanchor="left"
        if legend_position == "right"
        else ("right" if legend_position == "left" else "center"),
        yanchor="middle"
        if legend_position in ["left", "right"]
        else ("bottom" if legend_position == "top" else "top"),
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        showlegend=show_legend and num_series > 1,
        legend=legend_config if show_legend and num_series > 1 else None,
        plot_bgcolor=plot_background_color,
        paper_bgcolor=background_color,
        xaxis=dict(
            showgrid=grid,
            gridwidth=1,
            gridcolor="lightgray",
            range=x_range,
            type="log" if log_x else "linear",
        ),
        yaxis=dict(
            showgrid=grid,
            gridwidth=1,
            gridcolor="lightgray",
            range=y_range,
            type="log" if log_y else "linear",
        ),
        hovermode="x unified" if show_hover else False,
    )
    return fig
