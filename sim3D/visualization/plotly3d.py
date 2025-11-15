"""
Plotly-based 3D Visualization Suite for Reservoir Simulation Data and Results.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import itertools
import logging
import typing

from attrs import asdict
import numpy as np
import plotly.graph_objects as go
from typing_extensions import TypedDict, Unpack

from sim3D._precision import get_dtype
from sim3D.grids.base import coarsen_grid
from sim3D.states import ModelState
from sim3D.types import (
    NDimension,
    NDimensionalGrid,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from sim3D.visualization.base import (
    ColorScheme,
    PropertyMetadata,
    PropertyRegistry,
    property_registry,
)
from sim3D.wells import Wells

logger = logging.getLogger(__name__)


class PlotType(str, Enum):
    """Types of 3D plots available."""

    VOLUME = "volume"
    """Volume rendering for scalar fields, showing continuous data distribution."""
    ISOSURFACE = "isosurface"
    """Isosurface plots for discrete value thresholds, showing surfaces at specific data values."""
    SCATTER_3D = "scatter_3d"
    """3D scatter plots for visualizing point data in 3D space."""
    CELL_BLOCKS = "cell_blocks"
    """Cell block plots for visualizing individual reservoir cells as blocks in 3D space."""


class CameraPosition(TypedDict):
    """Camera position and orientation for 3D plots."""

    eye: dict[str, float]
    """Camera position in 3D space (x, y, z coordinates)."""

    center: dict[str, float]
    """Point in 3D space that the camera is looking at (x, y, z coordinates)."""

    up: dict[str, float]
    """Up direction vector for the camera (x, y, z coordinates)."""


class Lighting(TypedDict, total=False):
    """Lighting configuration for 3D plots."""

    ambient: float
    """Ambient light intensity (0.0 to 1.0)."""

    diffuse: float
    """Diffuse light intensity (0.0 to 1.0)."""

    specular: float
    """Specular light intensity (0.0 to 1.0)."""

    roughness: float
    """Surface roughness (0.0 to 1.0)."""

    fresnel: float
    """Fresnel effect intensity (0.0 to 1.0)."""

    facenormalsepsilon: float
    """Epsilon value for face normals (to avoid numerical issues)."""


class LightPosition(TypedDict):
    """Position of the light source in 3D space."""

    x: float
    """X coordinate of the light source."""

    y: float
    """Y coordinate of the light source."""

    z: float
    """Z coordinate of the light source."""


class WellKwargs(TypedDict, total=False):
    """
    Configuration options for well visualization in 3D plots.

    All fields are optional. Default values are used if not specified.
    """

    show_wellbore: bool
    """Whether to show wellbore trajectory as colored tubes (default: True)."""

    show_surface_marker: bool
    """Whether to show arrows at surface location (default: True)."""

    show_perforations: bool
    """Whether to highlight perforated intervals with thicker lines (default: False)."""

    injection_color: str
    """Color for injection wells - CSS color, hex, or rgb (default: "#ff4444")."""

    production_color: str
    """Color for production wells - CSS color, hex, or rgb (default: "#44dd44")."""

    shut_in_color: str
    """Color for shut-in/inactive wells - CSS color, hex, or rgb (default: "#888888")."""

    wellbore_width: float
    """Width of wellbore line representation in pixels (default: 15.0)."""

    surface_marker_size: float
    """Size scaling factor for surface markers (default: 2.0)."""


DEFAULT_CAMERA_POSITION = CameraPosition(
    eye=dict(x=2.2, y=2.2, z=1.8),  # type: ignore
    center=dict(x=0.0, y=0.0, z=0.0),  # type: ignore
    up=dict(x=0.0, y=0.0, z=1.0),  # type: ignore
)

DEFAULT_OPACITY_SCALE_VALUES = [
    [0, 0.8],
    [0.5, 0.9],
    [1, 1.0],
]

DEFAULT_LIGHTING = Lighting(
    ambient=0.5,
    diffuse=0.8,
    specular=0.2,
    roughness=0.5,
    fresnel=0.2,
    facenormalsepsilon=0.000001,
)


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for 3D plots."""

    width: int = 1200
    """Plot width in pixels. Larger values provide higher resolution but may impact performance."""

    height: int = 960
    """Plot height in pixels. Larger values provide higher resolution but may impact performance."""

    plot_type: PlotType = PlotType.VOLUME
    """Default plot type to use when no specific type is requested. Different plot types offer 
    different visualization perspectives of the same 3D data."""

    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    """Default color scheme for data visualization. Professional color schemes are optimized
    for scientific data visualization and accessibility."""

    opacity: float = 0.85
    """Default opacity level for plot elements (0.0 = transparent, 1.0 = opaque).
    Lower values allow better visualization of internal structures."""

    show_colorbar: bool = True
    """Whether to display the colorbar legend showing the data value to color mapping.
    Useful for quantitative analysis of the visualization."""

    show_axes: bool = True
    """Whether to display 3D axes labels and grid lines. Helps with spatial orientation
    and coordinate reference."""

    camera_position: CameraPosition = field(
        default_factory=lambda: DEFAULT_CAMERA_POSITION
    )
    """3D camera position and orientation. If None, uses default isometric view.
    Dict should contain 'eye', 'center', and 'up' keys with x,y,z coordinates."""

    title: str = ""
    """Plot title to display above the visualization. Empty string shows no title."""

    show_cell_outlines: bool = False
    """Whether to show wireframe outlines around individual cell blocks in cell block plots.
    Helps distinguish individual cells but may impact performance with large datasets."""

    cell_outline_color: str = "#404040"
    """Color for cell block outlines when show_cell_outlines is True. 
    Default is dark gray for better definition. Can be CSS color name, hex code, or rgb() string."""

    cell_outline_width: float = 1.0
    """Width/thickness of cell outline wireframes in pixels when show_cell_outlines is True.
    Default 1.0 provides good visibility. Thicker lines are more visible but may obscure data details."""

    use_opacity_scaling: bool = False
    """Whether to apply data-driven opacity scaling for better depth perception.
    Higher data values become more opaque, lower values more transparent."""

    opacity_scale_values: typing.Sequence[typing.Sequence[float]] = field(
        default_factory=lambda: DEFAULT_OPACITY_SCALE_VALUES
    )
    """Custom opacity scaling values for volume rendering. List of [data_fraction, opacity] pairs.
    If None, uses default scaling optimized for reservoir data visualization.
    Example: [[0, 0.05], [0.2, 0.3], [0.5, 0.6], [0.8, 0.8], [1, 1.0]]"""

    aspect_mode: typing.Optional[typing.Literal["cube", "data", "auto"]] = None
    """Aspect mode for the 3D plot. Determines how the plot is scaled and displayed.
    Options are:
    - "cube": Equal scaling for all axes (default).
    - "data": Automatic scaling based on data extents.
    - "auto": Automatic aspect ratio adjustment."""

    lighting: Lighting = field(default_factory=lambda: DEFAULT_LIGHTING)
    """Lighting configuration for 3D plots. Controls how light interacts with surfaces,

    including ambient, diffuse, and specular reflections, roughness, and fresnel effects.
    This affects how surfaces appear under different lighting conditions, enhancing realism."""

    light_position: typing.Optional[LightPosition] = None
    """Position of the light source in 3D space. Controls where the light comes from,
    affecting shadows and highlights on surfaces. This can be adjusted to simulate different
    lighting conditions, such as overhead sunlight or side lighting for dramatic effects."""

    paper_bgcolor: str = "#ffffff"
    """Background color of entire figure (outer area)"""

    scene_bgcolor: str = "#f8f9fa"
    """Background color of the 3D scene (light gray for subtle contrast)"""

    show_labels: bool = True
    """Global toggle for all labels in plots. When False, disables labels even if labels parameter is provided.
    Useful for clean visualizations without text annotations."""


@dataclass(frozen=True)
class LabelCoordinate:
    """Represents a 3D position for placing labels."""

    x: int
    y: int
    z: int

    def as_physical(
        self,
        cell_dimension: typing.Tuple[float, float],
        thickness_grid: ThreeDimensionalGrid,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> typing.Tuple[float, float, float]:
        """
        Convert index coordinates to physical coordinates.

        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param coordinate_offsets: Optional cell index offsets to apply to the physical coordinates
        :return: Tuple of (x_physical, y_physical, z_physical) coordinates
        """
        offsets = coordinate_offsets or (0, 0, 0)
        dx, dy = cell_dimension

        # For Z, we need to calculate cumulative depth
        actual_x = offsets[0] + self.x
        actual_y = offsets[1] + self.y
        actual_z = offsets[2] + self.z
        # Convert to physical coordinates
        x_physical = actual_x * dx
        y_physical = actual_y * dy

        if (
            actual_x < thickness_grid.shape[0]
            and actual_y < thickness_grid.shape[1]
            and actual_z < thickness_grid.shape[2]
        ):
            # Calculate cumulative depth from surface
            z_physical = -np.sum(thickness_grid[actual_x, actual_y, :actual_z])
        else:
            # Fallback to simple index-based positioning
            z_physical = -(offsets[2] + self.z) * 10.0  # Assume 10 ft average thickness

        return x_physical, y_physical, typing.cast(float, z_physical)


class _SafeValuesDict(dict):
    """A dictionary that safely handles missing keys by returning a formatted string."""

    def __missing__(self, key) -> str:
        return f"{{{key}}}"


class LabelFormatValues(TypedDict):
    """Format values for label text generation."""

    x_index: typing.Union[int, float]
    y_index: typing.Union[int, float]
    z_index: typing.Union[int, float]
    x_physical: typing.Optional[float]
    y_physical: typing.Optional[float]
    z_physical: typing.Optional[float]
    value: typing.Union[int, float, str]
    formatted_value: typing.Optional[str]
    property_name: typing.Optional[str]
    unit: typing.Optional[str]


@dataclass(slots=True)
class Label:
    """
    A flexible label that can be positioned in 3D space on a 3D grid and extract data dynamically.
    """

    position: LabelCoordinate
    text_template: str = """
    {name}
    <br>
    Cell Coordinates: ({x_index}, {y_index}, {z_index})
    <br>
    Physical Coordinates: ({x_physical:.2f}, {y_physical:.2f}, {z_physical:.2f}) ft
    <br>
    Value: {value}
    <br>
    Formatted Value: {formatted_value}
    <br>
    Property: {property_name}
    <br>
    Unit: {unit}
    <br>
    """

    font_size: int = 12
    font_color: str = "#333333"  # Default dark gray
    background_color: typing.Optional[str] = "rgba(240, 240, 240, 0.9)"
    border_color: typing.Optional[str] = None
    border_width: int = 1
    offset: typing.Tuple[float, float, float] = (0, 0, 0)
    visible: bool = True
    name: typing.Optional[str] = None

    def get_text(
        self,
        data_grid: typing.Optional[ThreeDimensionalGrid] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMetadata] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> str:
        """
        Generate the label text based on data at the label position.

        :param data_grid: 3D data array to extract values from
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param metadata: Property metadata for formatting
        :param format_kwargs: Additional values for string formatting
        :return: Formatted label text
        """
        values = {
            "x_index": self.position.x,
            "y_index": self.position.y,
            "z_index": self.position.z,
            "x_physical": None,
            "y_physical": None,
            "z_physical": None,
            "value": "N/A",
            "formatted_value": "N/A",
            "property_name": "N/A",
            "unit": "N/A",
        }

        raw_value = None
        formatted_value = None
        x_index = int(self.position.x)
        y_index = int(self.position.y)
        z_index = int(self.position.z)

        # Extract data value if grid is provided
        if data_grid is not None:
            raw_value = data_grid[x_index, y_index, z_index]

        if cell_dimension is not None and thickness_grid is not None:
            x_physical, y_physical, z_physical = self.position.as_physical(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                coordinate_offsets=None,  # Don't pass offset here, handle separately
            )
            # Apply the label's offset to the calculated physical coordinates
            x_physical += self.offset[0]
            y_physical += self.offset[1]
            z_physical += self.offset[2]

            values["x_physical"] = x_physical
            values["y_physical"] = y_physical
            values["z_physical"] = z_physical

        if metadata is not None:
            if raw_value is not None:
                formatted_value = BaseRenderer.format_value(raw_value, metadata)
            values["unit"] = metadata.unit
            values["property_name"] = metadata.display_name

        values["value"] = raw_value if raw_value is not None else "N/A"
        values["formatted_value"] = (
            formatted_value if formatted_value is not None else "N/A"
        )

        # Add any additional format kwargs
        if format_kwargs:
            values.update(format_kwargs)
        return self.text_template.format(**_SafeValuesDict(values))

    def as_annotation(
        self,
        data_grid: typing.Optional[ThreeDimensionalGrid] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMetadata] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> typing.Dict[str, typing.Any]:
        """
        Convert label to Plotly 3D annotation format.

        :param data_grid: 3D data array for value extraction
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param metadata: Property metadata for formatting
        :param format_kwargs: Additional formatting values
        :param coordinate_offsets: Coordinate offsets for sliced data
        :return: Plotly annotation dictionary
        """
        if not self.visible:
            return {}

        # Determine if we should use physical coordinates
        use_physical = cell_dimension is not None and thickness_grid is not None

        if use_physical:
            cell_dimension = typing.cast(typing.Tuple[float, float], cell_dimension)
            thickness_grid = typing.cast(ThreeDimensionalGrid, thickness_grid)
            # Convert to physical coordinates
            try:
                x_physical, y_physical, z_physical = self.position.as_physical(
                    cell_dimension, thickness_grid, coordinate_offsets
                )
                # Apply offset in physical space
                x_position = x_physical + self.offset[0]
                y_position = y_physical + self.offset[1]
                z_position = z_physical + self.offset[2]

                # Log physical coordinates for debugging
                logger.debug(
                    f"Label at index ({self.position.x}, {self.position.y}, {self.position.z}) "
                    f"-> physical ({x_position:.2f}, {y_position:.2f}, {z_position:.2f})"
                )

            except Exception as exc:
                logger.warning(
                    f"Failed to convert to physical coordinates: {exc}. Using index coordinates."
                )
                use_physical = False

        if not use_physical:
            # Use index coordinates with offsets
            offsets = coordinate_offsets or (0, 0, 0)
            x_position = self.position.x + offsets[0] + self.offset[0]
            y_position = self.position.y + offsets[1] + self.offset[1]
            z_position = self.position.z + offsets[2] + self.offset[2]

            logger.debug(
                f"Label using index coordinates: ({x_position}, {y_position}, {z_position})"
            )

        # Create a minimal annotation first to test if basic rendering works
        annotation = {
            "x": float(x_position),  # type: ignore
            "y": float(y_position),  # type: ignore
            "z": float(z_position),  # type: ignore
            "text": self.get_text(
                data_grid=data_grid,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                metadata=metadata,
                format_kwargs=format_kwargs,
            ),
            "showarrow": True,
            "font": {"size": self.font_size, "color": self.font_color},
        }

        # Add optional styling only if values are provided
        if self.background_color:
            annotation["bgcolor"] = self.background_color
        if self.border_color:
            annotation["bordercolor"] = self.border_color
        if self.border_width > 0:
            annotation["borderwidth"] = self.border_width

        logger.debug(f"Created annotation: {annotation}")
        return annotation


class Labels:
    """
    A collection of labels for 3D plots, allowing dynamic label creation and management.
    """

    def __init__(self, labels: typing.Optional[typing.Iterable[Label]] = None):
        self._labels = list(labels) if labels is not None else []

    def add(self, label: Label) -> None:
        """Add a label to the collection."""
        self._labels.append(label)

    def add_grid_labels(
        self,
        data_shape: typing.Tuple[int, int, int],
        spacing: typing.Tuple[int, int, int] = (10, 10, 5),
        template: str = "({x_index}, {y_index}, {z_index})",
        **label_kwargs,
    ) -> None:
        """
        Add grid labels at regular intervals.

        :param data_shape: Shape of the 3D data grid
        :param spacing: Spacing between labels in each dimension
        :param template: Text template for labels
        :param label_kwargs: Additional Label constructor arguments
        """
        nx, ny, nz = data_shape
        x_spacing, y_spacing, z_spacing = spacing

        for x, y, z in itertools.product(
            range(0, nx, x_spacing), range(0, ny, y_spacing), range(0, nz, z_spacing)
        ):
            position = LabelCoordinate(x, y, z)
            label = Label(position=position, text_template=template, **label_kwargs)
            self.add(label)

    def add_boundary_labels(
        self,
        data_shape: typing.Tuple[int, int, int],
        template: str = "Boundary ({x_index}, {y_index}, {z_index})",
        **label_kwargs,
    ) -> None:
        """
        Add labels at the boundaries of the data grid.

        :param data_shape: Shape of the 3D data grid
        :param template: Text template for boundary labels
        :param label_kwargs: Additional Label constructor arguments
        """
        nx, ny, nz = data_shape

        # Corner positions
        corners = [
            (0, 0, 0),
            (nx - 1, 0, 0),
            (0, ny - 1, 0),
            (0, 0, nz - 1),
            (nx - 1, ny - 1, 0),
            (nx - 1, 0, nz - 1),
            (0, ny - 1, nz - 1),
            (nx - 1, ny - 1, nz - 1),
        ]

        for x, y, z in corners:
            position = LabelCoordinate(x, y, z)
            label = Label(
                position=position,
                text_template=template,
                name=f"corner_{x}_{y}_{z}",
                **label_kwargs,
            )
            self.add(label)

    def add_well_labels(
        self,
        well_positions: typing.List[typing.Tuple[int, int, int]],
        well_names: typing.Optional[typing.List[str]] = None,
        template: str = "Well - '{name}' ({x_index}, {y_index}, {z_index}): {formatted_value} ({unit})",
        **label_kwargs,
    ) -> None:
        """
        Add labels for well positions.

        :param well_positions: List of (x, y, z) well positions
        :param well_names: Optional well names (defaults to Well_1, Well_2, etc.)
        :param template: Text template for well labels
        :param label_kwargs: Additional Label constructor arguments
        """
        for i, (x, y, z) in enumerate(well_positions):
            name = (
                well_names[i] if well_names and i < len(well_names) else f"Well_{i + 1}"
            )
            position = LabelCoordinate(x, y, z)
            label_kwargs.setdefault("font_size", 12)
            label_kwargs.setdefault("font_color", "#333")
            label = Label(
                position=position,
                text_template=template,
                name=name,
                **label_kwargs,
            )
            self.add(label)

    def visible(self) -> typing.Generator[Label, None, None]:
        """Return only visible labels."""
        return (label for label in self._labels if label.visible)

    def as_annotations(
        self,
        data_grid: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMetadata] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Convert all visible labels to Plotly annotations.

        :param data_grid: 3D data array for value extraction
        :param metadata: Property metadata for formatting
        :param cell_dimension: Physical cell dimensions for coordinate conversion
        :param thickness_grid: Height grid for physical coordinate conversion
        :param coordinate_offsets: Coordinate offsets for sliced data
        :param format_kwargs: Additional formatting values
        :return: List of Plotly annotation dictionaries
        """
        annotations = []

        for label in self.visible():
            try:
                # Check if label position is within the sliced data bounds
                if coordinate_offsets is not None and data_grid is not None:
                    # Check if label falls within the current slice bounds
                    # Label position should be relative to the sliced data (0-indexed)
                    if (
                        label.position.x < 0
                        or label.position.x >= data_grid.shape[0]
                        or label.position.y < 0
                        or label.position.y >= data_grid.shape[1]
                        or label.position.z < 0
                        or label.position.z >= data_grid.shape[2]
                    ):
                        # Label is outside the sliced data bounds, skip it
                        continue

                # Add custom values for well names, etc.
                kwargs = {"name": label.name} if label.name else {}
                kwargs.update(format_kwargs or {})
                annotation = label.as_annotation(
                    data_grid=data_grid,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    metadata=metadata,
                    format_kwargs=kwargs,
                    coordinate_offsets=coordinate_offsets,
                )

                if annotation:
                    annotations.append(annotation)

            except Exception as exc:
                logger.warning(
                    f"Failed to create annotation for label {label.name}: {exc}"
                )
                continue
        return annotations

    def clear(self) -> None:
        """Clear all labels."""
        self._labels.clear()

    def __len__(self) -> int:
        """Get the number of labels."""
        return len(self._labels)

    def __iter__(self) -> typing.Iterator[Label]:
        """Iterate over all labels."""
        return iter(self._labels)

    def __getitem__(self, index: int) -> Label:
        """Get a label by index."""
        return self._labels[index]


@typing.overload
def coarsen_grid_and_coords(
    data: NDimensionalGrid[NDimension],
    x_coords: OneDimensionalGrid,
    y_coords: OneDimensionalGrid,
    z_coords: None,
    batch_size: typing.Optional[NDimension],
    method: typing.Literal["mean", "sum", "max", "min"],
) -> typing.Tuple[
    NDimensionalGrid[NDimension], OneDimensionalGrid, OneDimensionalGrid, None
]: ...


@typing.overload
def coarsen_grid_and_coords(
    data: NDimensionalGrid[NDimension],
    x_coords: OneDimensionalGrid,
    y_coords: OneDimensionalGrid,
    z_coords: NDimensionalGrid[NDimension],
    batch_size: typing.Optional[NDimension],
    method: typing.Literal["mean", "sum", "max", "min"],
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    OneDimensionalGrid,
    OneDimensionalGrid,
    NDimensionalGrid[NDimension],
]: ...


def coarsen_grid_and_coords(
    data: NDimensionalGrid[NDimension],
    x_coords: OneDimensionalGrid,
    y_coords: OneDimensionalGrid,
    z_coords: typing.Optional[NDimensionalGrid[NDimension]] = None,
    batch_size: typing.Optional[NDimension] = None,
    method: typing.Literal["mean", "sum", "max", "min"] = "mean",
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    OneDimensionalGrid,
    OneDimensionalGrid,
    typing.Optional[NDimensionalGrid[NDimension]],
]:
    """
    Coarsen a 2D or 3D grid and compute coarsened coordinates using padding instead of trimming.
    Cell coordinates represent **cell boundaries**.

    :param data: 2D or 3D numpy array to coarsen.
    :param x_coords: 1D array of x-axis cell boundaries.
    :param y_coords: 1D array of y-axis cell boundaries.
    :param z_coords: 1D or 3D array of z-axis cell boundaries for 3D grid. Required if data.ndim==3.
    :param batch_size: Coarsening factor per dimension.
    :param method: Aggregation method for data blocks.
    :return: Tuple (coarsened_data, x_batch, y_batch, z_batch)
    """
    if batch_size is None:
        batch_size = typing.cast(NDimension, (2,) * data.ndim)
    # ---------------------------
    # 1. Pad data to be divisible by batch_size
    # ---------------------------
    pad_width = []
    for dim, b in zip(data.shape, batch_size):
        remainder = dim % b
        pad_width.append((0, b - remainder if remainder > 0 else 0))

    if method == "mean":
        pad_value = np.nan
    elif method == "sum":
        pad_value = 0
    elif method == "max":
        pad_value = -np.inf
    elif method == "min":
        pad_value = np.inf
    else:
        raise ValueError(f"Unsupported method '{method}'")

    data_padded = np.pad(
        data, pad_width=pad_width, mode="constant", constant_values=pad_value
    )

    # ---------------------------
    # 2. Coarsen data
    # ---------------------------
    coarsened_data = coarsen_grid(data_padded, batch_size=batch_size, method=method)

    # ---------------------------
    # 3. Coarsen 1D coordinates (x, y)
    # ---------------------------
    def coarsen_1d(coords: np.ndarray, b: int) -> np.ndarray:
        remainder = len(coords) % b
        if remainder > 0:
            coords = np.pad(coords, (0, b - remainder), mode="edge")
        # Coarsen using mean to preserve boundary positions
        return coords.reshape(-1, b).mean(axis=1)

    x_batch = coarsen_1d(x_coords, batch_size[0])
    y_batch = coarsen_1d(y_coords, batch_size[1])

    # ---------------------------
    # 4. Coarsen z-coordinates
    # ---------------------------
    z_batch = None
    if z_coords is not None:
        if data.ndim != 3:
            raise ValueError("z_coords is only valid for 3D data")

        bx, by, bz = batch_size

        if z_coords.ndim == 1:
            # 1D vertical boundaries
            z_batch = coarsen_1d(z_coords, bz)
            # Append last boundary to maintain nz+1
            z_batch = np.append(z_batch, z_batch[-1] + (z_batch[-1] - z_batch[-2]))
        elif z_coords.ndim == 3:
            # Full 3D boundaries: (nx, ny, nz+1)
            nx, ny, nzp1 = z_coords.shape
            pad_x = bx - nx % bx if nx % bx > 0 else 0
            pad_y = by - ny % by if ny % by > 0 else 0
            pad_z = bz - nzp1 % bz if nzp1 % bz > 0 else 0

            z_padded = np.pad(
                z_coords, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="edge"
            )
            nxp, nyp, nzp1p = z_padded.shape

            # Reshape and coarsen along all axes
            z_reshaped = z_padded.reshape(nxp // bx, bx, nyp // by, by, nzp1p // bz, bz)
            z_batch = z_reshaped.mean(axis=(1, 3, 5))

            # Append last boundary layer along Z to maintain nz+1
            z_batch = np.concatenate([z_batch, z_batch[:, :, -1:]], axis=2)
        else:
            raise ValueError("z_coords must be 1D or 3D array for 3D data")

    return (
        typing.cast(NDimensionalGrid[NDimension], coarsened_data),
        typing.cast(OneDimensionalGrid, x_batch),
        typing.cast(OneDimensionalGrid, y_batch),
        typing.cast(typing.Optional[NDimensionalGrid[NDimension]], z_batch),
    )


class BaseRenderer(ABC):
    """Base class for 3D renders."""

    supports_physical_dimensions: typing.ClassVar[bool] = False
    """Whether this renderer supports physical cell dimensions and height grids."""

    def __init__(self, config: PlotConfig) -> None:
        self.config = config

    @abstractmethod
    def render(
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Render a 3D plot from data.

        Subclasses should override this with their specific keyword arguments.
        """
        pass

    def get_colorscale(self, color_scheme: typing.Union[ColorScheme, str]) -> str:
        """
        Get plotly colorscale string from ColorScheme enum.

        :param color_scheme: The color scheme enum to convert
        :return: Plotly colorscale string
        """
        return ColorScheme(color_scheme).value

    @staticmethod
    def format_value(value: float, metadata: PropertyMetadata) -> str:
        """
        Format a value for display with appropriate precision and scientific notation.

        :param value: The numeric value to format
        :param metadata: Property metadata for context
        :return: Formatted string representation
        """
        if np.isnan(value) or np.isinf(value):
            return "N/A"

        # Convert to absolute value to check decimal places
        abs_val = abs(value)

        # If value is very small or would have more than 6 decimal places, use scientific notation
        if abs_val == 0:
            return "0.000"
        elif abs_val < 1e-6 or (
            abs_val < 1 and len(f"{abs_val:.10f}".rstrip("0").split(".")[1]) > 6
        ):
            return f"{value:.4e}"
        elif abs_val >= 1e6:
            return f"{value:.4e}"
        elif abs_val >= 1000:
            return f"{value:.1f}"
        elif abs_val >= 1:
            return f"{value:.3f}"
        # For values between 0 and 1, show up to 6 decimal places
        formatted = f"{value:.6f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"

    def normalize_data(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        normalize_range: bool = False,
    ) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid]:
        """
        Prepare data for plotting (handle log scale, clipping, etc.).

        :param data: Input data array to prepare
        :param metadata: Property metadata containing scaling and range information
        :param normalize_range: Whether to normalize data to 0-1 range (only for volume rendering)
        :return: Tuple of (processed_data_for_plotting, original_data_for_display)
        """
        # Keep original data for display purposes (hover text, colorbar)
        dtype = get_dtype()
        display_data = data.astype(dtype)

        # Process data for plotting
        processed_data = data.astype(dtype)

        if metadata.log_scale:
            # Handle zero/negative values for log scale
            min_positive = np.nanmin(data[data > 0])
            processed_data = np.where(data <= 0, min_positive * 0.1, data)
            processed_data = np.log10(processed_data)

        # Apply clipping if specified
        if metadata.min_val is not None and metadata.max_val is not None:
            if metadata.log_scale:
                # Clip original data, then apply log to processed data
                display_data = np.clip(display_data, metadata.min_val, metadata.max_val)
                processed_data = np.log10(
                    np.where(display_data <= 0, metadata.min_val * 0.1, display_data)
                )
            else:
                processed_data = np.clip(
                    processed_data, metadata.min_val, metadata.max_val
                )
                display_data = processed_data.copy()

        elif metadata.min_val is not None:
            if metadata.log_scale:
                display_data = np.clip(display_data, metadata.min_val, None)
                processed_data = np.log10(
                    np.where(display_data <= 0, metadata.min_val * 0.1, display_data)
                )
            else:
                processed_data = np.clip(processed_data, metadata.min_val, None)
                display_data = processed_data.copy()

        elif metadata.max_val is not None:
            if metadata.log_scale:
                display_data = np.clip(display_data, None, metadata.max_val)
                processed_data = np.log10(
                    np.where(display_data <= 0, metadata.max_val * 0.1, display_data)
                )
            else:
                processed_data = np.clip(processed_data, None, metadata.max_val)
                display_data = processed_data.copy()

        if normalize_range:
            # Only normalize to 0-1 range when explicitly requested (e.g for volume rendering)
            data_min = float(np.nanmin(processed_data))
            data_max = float(np.nanmax(processed_data))
            if data_max > data_min:
                processed_data = (processed_data - data_min) / (data_max - data_min)

        processed_data = self.invert_z_axis(processed_data)
        display_data = self.invert_z_axis(display_data)
        return typing.cast(ThreeDimensionalGrid, processed_data), typing.cast(
            ThreeDimensionalGrid, display_data
        )

    def create_physical_coordinates(
        self,
        cell_dimension: typing.Tuple[float, float],
        thickness_grid: ThreeDimensionalGrid,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> typing.Tuple[OneDimensionalGrid, OneDimensionalGrid, ThreeDimensionalGrid]:
        """
        Create physical coordinate arrays based on cell dimensions and height grid.

        Implements numpy array indexing convention:
        - data[:, :, 0] (first array layer) appears at the TOP of the visualization
        - data[:, :, k] layers stack downward as k increases
        - data[:, :, -1] (last array layer) appears at the BOTTOM of the visualization

        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param coordinate_offsets: Optional cell index offsets for sliced data (x_offset, y_offset, z_offset)
        :return: Tuple of (X, Y, Z) coordinate arrays in physical units
        """
        nx, ny, nz = thickness_grid.shape
        dx, dy = cell_dimension

        # Calculate physical coordinate offsets from cell index offsets
        physical_offsets = (0.0, 0.0, 0.0)
        if coordinate_offsets is not None:
            x_index_offset, y_index_offset, z_index_offset = coordinate_offsets
            x_start_offset = x_index_offset * dx
            y_start_offset = y_index_offset * dy

            # Calculate Z offset from height grid
            if z_index_offset > 0:
                # Calculate cumulative depth to the starting Z slice
                z_start_offset = -np.sum(
                    thickness_grid[:, :, :z_index_offset], axis=2
                ).mean()
            else:
                z_start_offset = 0.0

            physical_offsets = (x_start_offset, y_start_offset, z_start_offset)

        # Apply physical offsets (for sliced data)
        x_offset, y_offset, z_offset = physical_offsets
        # Create base coordinate grids with offsets
        x_coords = x_offset + np.arange(nx + 1) * dx  # Cell boundaries
        y_coords = y_offset + np.arange(ny + 1) * dy  # Cell boundaries
        z_coords = np.zeros((nx, ny, nz + 1), dtype=np.float64)

        # Build coordinates downward - each layer is lower in Z, starting from z_offset
        z_coords[:, :, 0] = z_offset  # Start from the Z offset
        for k in range(nz):
            z_coords[:, :, k + 1] = z_coords[:, :, k] - thickness_grid[:, :, k]

        return (
            typing.cast(OneDimensionalGrid, x_coords),
            typing.cast(OneDimensionalGrid, y_coords),
            typing.cast(ThreeDimensionalGrid, z_coords),
        )

    def apply_labels(
        self,
        figure: go.Figure,
        labels: typing.Optional[
            typing.Union["Labels", typing.Iterable["Label"]]
        ] = None,
        data: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMetadata] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> None:
        """
        Apply labels to a 3D plot using the Label system.

        :param figure: Plotly figure to add labels to
        :param labels: Labels to apply (Labels or list of Label objects)
        :param data: 3D data array for value extraction
        :param metadata: Property metadata for formatting
        :param cell_dimension: Physical cell dimensions for coordinate conversion
        :param thickness_grid: Height grid for physical coordinate conversion
        :param coordinate_offsets: Coordinate offsets for sliced data
        :param format_kwargs: Additional formatting values
        """
        if not self.config.show_labels:
            return

        if labels is None:
            return

        # Handle both Labels and list of Labels
        if isinstance(labels, Labels):
            label_collection = labels
        else:
            label_collection = Labels(labels)

        # Get Plotly annotations
        annotations = label_collection.as_annotations(
            data_grid=data,
            metadata=metadata,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            coordinate_offsets=coordinate_offsets,
            format_kwargs=format_kwargs,
        )
        # Add annotations to the figure
        if annotations:
            # Get existing annotations or create empty list
            existing_annotations = []
            try:
                scene = getattr(figure.layout, "scene", None)
                if scene and hasattr(scene, "annotations"):
                    existing_annotations = list(scene.annotations or [])
            except (AttributeError, TypeError):
                pass

            existing_annotations.extend(annotations)
            figure.update_layout(scene=dict(annotations=existing_annotations))

            # Debug information
            logger.debug(f"Applied {len(annotations)} labels to 3D plot")
        else:
            logger.debug(
                "No annotations to apply - labels may be filtered out or invisible"
            )

    @staticmethod
    def invert_z_axis(arr: np.ndarray) -> np.ndarray:
        """
        Invert the Z axis (last axis) of a 3D array so that data[:,:,0] becomes data[:,:,nz-1].
        This ensures numpy convention (top layer as k=0) matches plotly's rendering (bottom as k=0).
        """
        if arr.ndim == 3:
            return arr[:, :, ::-1]
        return arr

    def render_wells(
        self,
        figure: go.Figure,
        wells: typing.Optional[Wells[ThreeDimensions]] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[np.ndarray] = None,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
        z_scale: float = 1.0,
        **kwargs: Unpack[WellKwargs],
    ) -> None:
        """
        Add visual representation of wells to 3D plot.

        This method renders a comprehensive 3D visualizations of wells on the figure including:
        - Wellbore trajectories as colored lines/tubes
        - Surface location markers with directional arrows
        - Perforated interval highlights (optional)
        - Interactive hover information

        :param figure: Plotly figure to add well visualizations to
        :param wells: Wells object containing injection and production wells
        :param cell_dimension: Physical size of each cell in x and y directions (feet).
            Required for physical coordinate conversion. If None, uses grid indices.
        :param thickness_grid: 3D array with height of each cell (feet).
            Required for physical coordinate conversion. If None, uses grid indices.
        :param coordinate_offsets: Coordinate offsets for sliced data (i_offset, j_offset, k_offset).
            Used to adjust well locations when visualizing a subset of the full grid.
        :param z_scale: Scale factor for Z-axis to match volume rendering (default: 1.0).
            Wells will be scaled to match the stretched/compressed volume visualization.
        :param show_wellbore: Whether to show wellbore trajectory as colored tubes (default: True)
        :param show_surface_marker: Whether to show arrows at surface location (default: True)
        :param show_perforations: Whether to highlight perforated intervals with thicker lines (default: False)
        :param injection_color: Color for injection wells (CSS color, hex, or rgb) (default: red)
        :param production_color: Color for production wells (CSS color, hex, or rgb) (default: green)
        :param shut_in_color: Color for shut-in/inactive wells (CSS color, hex, or rgb) (default: gray)
        :param wellbore_width: Width of wellbore line representation in pixels (default: 15.0)
        :param surface_marker_size: Size scaling factor for surface markers (default: 5.0)

        **Usage Example:**

        ```python
        # Basic usage with default styling
        renderer.render_wells(fig, model_state.wells)

        # Custom styling
        renderer.render_wells(
            fig,
            model_state.wells,
            cell_dimension=(100, 100),
            thickness_grid=thickness,
            injection_color='#ff6b6b',
            production_color='#51cf66',
            wellbore_width=8.0,
            show_surface_marker=True
        )
        ```

        **Visual Elements:**
        - Injection wells: Red/warm colors with upward flow indication
        - Production wells: Green/cool colors with downward flow indication
        - Shut-in wells: Gray/muted colors, partially transparent
        - Surface markers: Cone/arrow shapes pointing into reservoir
        - Wellbore: Continuous line through all perforating intervals

        **Interactive Features:**
        - Hover over wellbore: Shows well name, type, BHP, radius, skin factor
        - Hover over surface marker: Shows well name and surface location
        - Legend groups: Wells can be toggled on/off in plot legend
        """
        if wells is None:
            logger.warning("render_wells called but wells parameter is None")
            return

        if not wells.has_wells():
            logger.warning("`render_wells` called but wells.has_wells() returned False")
            return

        logger.debug(
            f"Rendering {len(wells.injection_wells)} injection wells and {len(wells.production_wells)} production wells"
        )

        # Extract kwargs with defaults
        show_wellbore = kwargs.get("show_wellbore", True)
        show_surface_marker = kwargs.get("show_surface_marker", True)
        show_perforations = kwargs.get("show_perforations", False)
        injection_color = kwargs.get("injection_color", "#ff4444")
        production_color = kwargs.get("production_color", "#44dd44")
        shut_in_color = kwargs.get("shut_in_color", "#888888")
        wellbore_width = kwargs.get("wellbore_width", 15.0)
        surface_marker_size = kwargs.get("surface_marker_size", 5.0)

        # Calculate Z mean for scaling (if z_scale != 1.0 and we have thickness data)
        z_mean = 0.0
        if z_scale != 1.0 and thickness_grid is not None:
            # Calculate the mean Z position of the grid
            # This matches how volume rendering scales Z coordinates
            z_offset_val = coordinate_offsets[2] if coordinate_offsets else 0
            # Sum thickness in ONE vertical column (all layers at i=0, j=0)
            # This gives total vertical extent of the reservoir
            total_depth = float(np.sum(thickness_grid[0, 0, :]))
            # Mean Z is halfway down from surface
            z_mean = z_offset_val - (total_depth / 2.0)  # Negative because Z goes down

        def grid_to_physical(
            i: int, j: int, k: int
        ) -> typing.Optional[typing.Tuple[float, float, float]]:
            """
            Convert grid indices to physical coordinates.
            Returns None if coordinates contain NaN.
            Note: Z-axis is negative (depth increases downward in Plotly convention)
            Z-scale is applied around z_mean to match volume rendering.
            """
            if cell_dimension is not None and thickness_grid is not None:
                dx, dy = cell_dimension
                x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)

                # Physical X and Y
                x_phys = (x_offset + i) * dx
                y_phys = (y_offset + j) * dy

                # Physical Z (depth) - calculate to cell center
                # Z is NEGATIVE (increases downward), matching volume rendering
                # k=0 is top layer, larger k = deeper
                z_phys = z_offset

                if k > 0:
                    # Sum thickness from top to this layer (excluding current)
                    thickness_above = float(np.sum(thickness_grid[i, j, :k]))
                    if np.isnan(thickness_above):
                        return None
                    # Subtract to go deeper (negative Z)
                    z_phys -= thickness_above

                # Add half of current cell thickness (to get to center)
                cell_thickness = float(thickness_grid[i, j, k])
                if np.isnan(cell_thickness):
                    return None
                # Subtract half thickness to get to cell center
                z_phys -= cell_thickness / 2.0

                # Apply z_scale around z_mean (same as volume rendering)
                if z_scale != 1.0:
                    z_phys = z_mean + (z_phys - z_mean) * z_scale

                return x_phys, y_phys, z_phys
            else:
                # Use grid indices directly
                return float(i), float(j), float(k)

        # Process all injection wells
        for well in wells.injection_wells:
            # Determine color based on well state using is_shut_in() method
            if well.is_shut_in:
                color = shut_in_color
                well_type = "Injection (Shut-in)"
                opacity = 0.5
            else:
                color = injection_color
                well_type = "Injection"
                opacity = 1.0

            # Extract well trajectory points from perforating intervals
            if show_wellbore and well.perforating_intervals:
                x_points, y_points, z_points = [], [], []

                # Get the shallowest perforation (first one) for surface extension
                first_perf_start = well.perforating_intervals[0][0]
                first_perf_coords = grid_to_physical(*first_perf_start)

                # Add surface-to-perforation segment (neutral color)
                if first_perf_coords is not None and cell_dimension is not None:
                    x_surf, y_surf, z_perf = first_perf_coords
                    # Surface should be at z_offset (the top of the grid)
                    # coordinate_offsets contains the base Z coordinate
                    x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)
                    z_surface = float(z_offset)  # Top of volume

                    # Apply z_scale to surface coordinate
                    if z_scale != 1.0:
                        z_surface = z_mean + (z_surface - z_mean) * z_scale

                    # Add neutral-colored segment from surface to perforation
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_surf, x_surf],
                            y=[y_surf, y_surf],
                            z=[z_surface, z_perf],
                            mode="lines",
                            line=dict(
                                color="#999999",  # Neutral gray
                                width=wellbore_width * 0.7,  # Slightly thinner
                                dash="dot",  # Dotted to distinguish from perforations
                            ),
                            opacity=0.6,
                            name=f"{well.name} casing",
                            showlegend=False,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} Casing</b><br>"
                                f"Non-perforated section<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

                for start_loc, end_loc in well.perforating_intervals:
                    # Convert start and end to physical coordinates
                    start_coords = grid_to_physical(*start_loc)
                    end_coords = grid_to_physical(*end_loc)

                    # Skip if coordinates contain NaN
                    if start_coords is None or end_coords is None:
                        continue

                    x_start, y_start, z_start = start_coords
                    x_end, y_end, z_end = end_coords

                    # Handle single-cell perforations (start == end)
                    # Create a small vertical line segment for visibility
                    if start_loc == end_loc:
                        if cell_dimension is not None and thickness_grid is not None:
                            # Use half the cell thickness for the vertical extent
                            cell_thickness = float(
                                thickness_grid[start_loc[0], start_loc[1], start_loc[2]]
                            )
                            # Apply z_scale to the extension amount (already applied in grid_to_physical, but extension is raw)
                            extension = (cell_thickness / 4.0) * z_scale
                            z_start -= extension  # Extend upward
                            z_end += extension  # Extend downward
                        else:
                            # For grid coordinates, extend by 0.5 in Z direction
                            z_start -= 0.5
                            z_end += 0.5

                    x_points.extend([x_start, x_end])
                    y_points.extend([y_start, y_end])
                    z_points.extend([z_start, z_end])

                # Add wellbore trajectory only if we have valid points
                if x_points:
                    figure.add_trace(
                        go.Scatter3d(
                            x=x_points,
                            y=y_points,
                            z=z_points,
                            mode="lines",
                            line=dict(
                                color=color,
                                width=wellbore_width,
                                dash="solid" if well.is_open else "dash",
                            ),
                            opacity=opacity,
                            name=f"{well.name} ({well_type})",
                            showlegend=True,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name}</b><br>"
                                f"Type: {well_type}<br>"
                                f"BHP: {well.bottom_hole_pressure:.1f} psi<br>"
                                f"Radius: {well.radius:.2f} ft<br>"
                                f"Skin: {well.skin_factor:.2f}<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

            # Add perforation highlights if requested (injection wells)
            if show_perforations and well.perforating_intervals and well.is_open:
                for start_loc, end_loc in well.perforating_intervals:
                    # Convert to physical coordinates
                    start_coords = grid_to_physical(*start_loc)
                    end_coords = grid_to_physical(*end_loc)

                    # Skip if coordinates contain NaN
                    if start_coords is None or end_coords is None:
                        continue

                    x_start, y_start, z_start = start_coords
                    x_end, y_end, z_end = end_coords

                    # Add perforation interval as thicker, more opaque line with markers
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_start, x_end],
                            y=[y_start, y_end],
                            z=[z_start, z_end],
                            mode="lines+markers",
                            line=dict(
                                color=color,
                                width=wellbore_width * 1.5,  # 50% thicker
                            ),
                            marker=dict(
                                size=4,
                                color=color,
                                symbol="diamond",
                            ),
                            opacity=min(opacity * 1.2, 1.0),  # More opaque
                            name=f"{well.name} perforations",
                            showlegend=False,  # Don't clutter legend
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} Perforation</b><br>"
                                f"Start: ({start_loc[0]}, {start_loc[1]}, {start_loc[2]})<br>"
                                f"End: ({end_loc[0]}, {end_loc[1]}, {end_loc[2]})<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

            # Add surface marker with directional indicator
            if show_surface_marker and well.perforating_intervals:
                # Get surface location (X, Y from first perforation, but Z at surface)
                start_loc = well.perforating_intervals[0][0]
                surf_coords = grid_to_physical(*start_loc)

                # Only add surface marker if coordinates are valid
                if surf_coords is not None:
                    x_surf, y_surf, _ = surf_coords

                    # Surface should be at z_offset (the top of the grid)
                    # coordinate_offsets contains the base Z coordinate
                    x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)
                    z_surf = float(z_offset)  # Top of volume

                    # Apply z_scale to surface coordinate
                    if z_scale != 1.0:
                        z_surf = z_mean + (z_surf - z_mean) * z_scale

                    # Add cone/arrow pointing into reservoir at the SURFACE
                    # Build fluid info for hover text
                    if well.injected_fluid is not None:
                        fluid_info = f"Injected Fluid: <b>{well.injected_fluid.name}</b>  ({well.injected_fluid.phase.value})"
                    else:
                        fluid_info = "Injected Fluid: N/A"

                    status = "Open" if well.is_open else "Shut-in"

                    # Calculate appropriate cone size based on grid dimensions
                    # For aspectmode="data", scale relative to cell size
                    cone_size_multiplier = 2.0  # Default multiplier for cube mode
                    if cell_dimension is not None:
                        dx, dy = cell_dimension
                        # Scale cone size to be a reasonable fraction of cell size
                        # Typical cone should be about 5% of cell dimension
                        cone_size_multiplier = (
                            max(dx, dy) * 0.05 * z_scale / surface_marker_size
                        )

                    # Arrow sizing - base it on actual grid thickness for proportionality
                    if thickness_grid is not None:
                        # Use average layer thickness as base unit
                        avg_layer_thickness = float(np.mean(thickness_grid))
                        # Make arrow about 1.5x average layer thickness (reasonable visibility)
                        base_arrow_length = avg_layer_thickness * 1.5
                    else:
                        # Fallback to marker size
                        base_arrow_length = surface_marker_size

                    # Apply z_scale
                    total_arrow_length = base_arrow_length * z_scale

                    # Split: 60% stem, 40% cone
                    stem_length = total_arrow_length * 0.6
                    cone_length = total_arrow_length * 0.4

                    # For injection wells, arrow points DOWN but starts ABOVE surface
                    # Arrow tip should reach the surface, so we start above
                    arrow_top = z_surf + total_arrow_length  # Start above surface
                    arrow_stem_bottom = (
                        z_surf + cone_length
                    )  # Stem ends where cone starts

                    # Add cylindrical stem (arrow shaft) starting from above
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_surf, x_surf],
                            y=[y_surf, y_surf],
                            z=[arrow_top, arrow_stem_bottom],
                            mode="lines",
                            line=dict(
                                color=color,
                                width=wellbore_width * 0.8,  # Make stem more visible
                            ),
                            opacity=opacity,
                            name=f"{well.name} arrow stem",
                            showlegend=False,
                            legendgroup=well.name,
                            hoverinfo="skip",
                        )
                    )

                    # Add cone (arrowhead) at the bottom of stem, pointing down to surface
                    figure.add_trace(
                        go.Cone(
                            x=[x_surf],
                            y=[y_surf],
                            z=[arrow_stem_bottom],  # Cone starts where stem ends
                            u=[0],
                            v=[0],
                            w=[-cone_length],  # Points down, tip reaches surface
                            sizemode="absolute",
                            sizeref=surface_marker_size * cone_size_multiplier,
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            opacity=opacity,
                            name=f"{well.name} surface",
                            showlegend=False,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} - Surface Location</b><br>"
                                f"Type: {well_type}<br>"
                                f"Status: {status}<br>"
                                f"{fluid_info}<br>"
                                f"BHP: {well.bottom_hole_pressure:.1f} psi<br>"
                                f"Wellbore Radius: {well.radius:.2f} ft<br>"
                                f"Skin Factor: {well.skin_factor:.2f}<br>"
                                f"<br>"
                                f"Surface Coords:<br>"
                                f"  X: {x_surf:.1f} ft<br>"
                                f"  Y: {y_surf:.1f} ft<br>"
                                f"  Z: {z_surf:.1f} ft<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

        # Process all production wells (similar logic)
        for well in wells.production_wells:
            logger.debug(
                f"Processing production well: {well.name}, is_active={well.is_active}, perforations={well.perforating_intervals}"
            )
            # Determine color based on well state using is_shut_in() method
            if well.is_shut_in:
                color = shut_in_color
                well_type = "Production (Shut-in)"
                opacity = 0.5
            else:
                color = production_color
                well_type = "Production"
                opacity = 1.0

            # Extract well trajectory points from perforating intervals
            if show_wellbore and well.perforating_intervals:
                x_points, y_points, z_points = [], [], []
                logger.debug(
                    f"  Extracting wellbore trajectory for {well.name}, show_wellbore={show_wellbore}"
                )

                # Get the shallowest perforation (first one) for surface extension
                first_perf_start = well.perforating_intervals[0][0]
                first_perf_coords = grid_to_physical(*first_perf_start)

                # Add surface-to-perforation segment (neutral color)
                if first_perf_coords is not None and cell_dimension is not None:
                    x_surf, y_surf, z_perf = first_perf_coords
                    # Surface should be at z_offset (the top of the grid)
                    # coordinate_offsets contains the base Z coordinate
                    x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)
                    z_surface = float(z_offset)  # Top of volume

                    # Apply z_scale to surface coordinate
                    if z_scale != 1.0:
                        z_surface = z_mean + (z_surface - z_mean) * z_scale

                    # Add neutral-colored segment from surface to perforation
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_surf, x_surf],
                            y=[y_surf, y_surf],
                            z=[z_surface, z_perf],
                            mode="lines",
                            line=dict(
                                color="#999999",  # Neutral gray
                                width=wellbore_width * 0.7,  # Slightly thinner
                                dash="dot",  # Dotted to distinguish from perforations
                            ),
                            opacity=0.6,
                            name=f"{well.name} casing",
                            showlegend=False,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} Casing</b><br>"
                                f"Non-perforated section<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

                for start_loc, end_loc in well.perforating_intervals:
                    # Convert start and end to physical coordinates
                    start_coords = grid_to_physical(*start_loc)
                    end_coords = grid_to_physical(*end_loc)

                    logger.debug(
                        f"    Production perforation: {start_loc} -> {end_loc}, coords: {start_coords} -> {end_coords}"
                    )

                    # Skip if coordinates contain NaN
                    if start_coords is None or end_coords is None:
                        logger.debug("    Skipping perforation due to None coordinates")
                        continue

                    x_start, y_start, z_start = start_coords
                    x_end, y_end, z_end = end_coords

                    # Handle single-cell perforations (start == end)
                    # Create a small vertical line segment for visibility
                    if start_loc == end_loc:
                        logger.debug(
                            f"    Single-cell perforation detected at {start_loc}, extending vertically"
                        )
                        if cell_dimension is not None and thickness_grid is not None:
                            # Use half the cell thickness for the vertical extent
                            cell_thickness = float(
                                thickness_grid[start_loc[0], start_loc[1], start_loc[2]]
                            )
                            # Apply z_scale to the extension amount
                            extension = (cell_thickness / 4.0) * z_scale
                            z_start -= extension  # Extend upward
                            z_end += extension  # Extend downward
                        else:
                            # For grid coordinates, extend by 0.5 in Z direction
                            z_start -= 0.5
                            z_end += 0.5

                    x_points.extend([x_start, x_end])
                    y_points.extend([y_start, y_end])
                    z_points.extend([z_start, z_end])

                # Add wellbore trajectory only if we have valid points
                if x_points:
                    logger.debug(
                        f"  Adding production wellbore trace for {well.name} with {len(x_points)} points"
                    )
                    figure.add_trace(
                        go.Scatter3d(
                            x=x_points,
                            y=y_points,
                            z=z_points,
                            mode="lines",
                            line=dict(
                                color=color,
                                width=wellbore_width,
                                dash="solid" if well.is_open else "dash",
                            ),
                            opacity=opacity,
                            name=f"{well.name} ({well_type})",
                            showlegend=True,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name}</b><br>"
                                f"Type: {well_type}<br>"
                                f"BHP: {well.bottom_hole_pressure:.1f} psi<br>"
                                f"Radius: {well.radius:.2f} ft<br>"
                                f"Skin: {well.skin_factor:.2f}<br>"
                                "<extra></extra>"
                            ),
                        )
                    )
                else:
                    logger.warning(
                        f"  No valid points found for production well {well.name} wellbore trajectory"
                    )

            # Add perforation highlights if requested (production wells)
            if show_perforations and well.perforating_intervals and well.is_open:
                for start_loc, end_loc in well.perforating_intervals:
                    # Convert to physical coordinates
                    start_coords = grid_to_physical(*start_loc)
                    end_coords = grid_to_physical(*end_loc)

                    # Skip if coordinates contain NaN
                    if start_coords is None or end_coords is None:
                        continue

                    x_start, y_start, z_start = start_coords
                    x_end, y_end, z_end = end_coords

                    # Add perforation interval as thicker, more opaque line with markers
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_start, x_end],
                            y=[y_start, y_end],
                            z=[z_start, z_end],
                            mode="lines+markers",
                            line=dict(
                                color=color,
                                width=wellbore_width * 1.5,  # 50% thicker
                            ),
                            marker=dict(
                                size=4,
                                color=color,
                                symbol="diamond",
                            ),
                            opacity=min(opacity * 1.2, 1.0),  # More opaque
                            name=f"{well.name} perforations",
                            showlegend=False,  # Don't clutter legend
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} Perforation</b><br>"
                                f"Start: ({start_loc[0]}, {start_loc[1]}, {start_loc[2]})<br>"
                                f"End: ({end_loc[0]}, {end_loc[1]}, {end_loc[2]})<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

            # Add surface marker with directional indicator (PRODUCTION WELLS)
            if show_surface_marker and well.perforating_intervals:
                # Get surface location (X, Y from first perforation, but Z at surface)
                start_loc = well.perforating_intervals[0][0]
                surf_coords = grid_to_physical(*start_loc)

                # Only add surface marker if coordinates are valid
                if surf_coords is not None:
                    x_surf, y_surf, _ = surf_coords

                    # Surface should be at z_offset (the top of the grid)
                    # coordinate_offsets contains the base Z coordinate
                    x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)
                    z_surf = float(z_offset)  # Top of volume

                    # Apply z_scale to surface coordinate
                    if z_scale != 1.0:
                        z_surf = z_mean + (z_surf - z_mean) * z_scale

                    # Add cone/arrow pointing OUT OF reservoir at the SURFACE (production = outflow)
                    # Build fluid info for hover text (production wells can have multiple fluids)
                    fluid_names = []
                    if well.produced_fluids:
                        for fluid in well.produced_fluids:
                            fluid_name = f"<b>{fluid.name}</b> ({fluid.phase.value})"
                            fluid_names.append(fluid_name)

                    fluid_info = (
                        "Produced Fluids: " + ", ".join(fluid_names)
                        if fluid_names
                        else "Produced Fluids: N/A"
                    )
                    status = "Open" if well.is_open else "Shut-in"

                    # Calculate appropriate cone size based on grid dimensions
                    # For aspectmode="data", scale relative to cell size
                    cone_size_multiplier = 2.0  # Default multiplier for cube mode
                    if cell_dimension is not None:
                        dx, dy = cell_dimension
                        # Scale cone size to be a reasonable fraction of cell size
                        # Typical cone should be about 5% of cell dimension
                        cone_size_multiplier = (
                            max(dx, dy) * 0.05 * z_scale / surface_marker_size
                        )

                    # Arrow sizing - base it on actual grid thickness for proportionality
                    if thickness_grid is not None:
                        # Use average layer thickness as base unit
                        avg_layer_thickness = float(np.mean(thickness_grid))
                        # Make arrow about 1.5x average layer thickness (reasonable visibility)
                        base_arrow_length = avg_layer_thickness * 1.5
                    else:
                        # Fallback to marker size
                        base_arrow_length = surface_marker_size

                    # Apply z_scale
                    total_arrow_length = base_arrow_length * z_scale

                    # Split: 60% stem, 40% cone
                    stem_length = total_arrow_length * 0.6
                    cone_length = total_arrow_length * 0.4

                    # Add cylindrical stem (arrow shaft) pointing up from surface
                    figure.add_trace(
                        go.Scatter3d(
                            x=[x_surf, x_surf],
                            y=[y_surf, y_surf],
                            z=[z_surf, z_surf + stem_length],
                            mode="lines",
                            line=dict(
                                color=color,
                                width=wellbore_width * 0.8,  # Make stem more visible
                            ),
                            opacity=opacity,
                            name=f"{well.name} arrow stem",
                            showlegend=False,
                            legendgroup=well.name,
                            hoverinfo="skip",
                        )
                    )

                    # Add cone (arrowhead) at the end of stem pointing up
                    figure.add_trace(
                        go.Cone(
                            x=[x_surf],
                            y=[y_surf],
                            z=[z_surf + stem_length],  # Position at end of stem
                            u=[0],
                            v=[0],
                            w=[cone_length],  # Arrow pointing UP (production/outflow)
                            sizemode="absolute",
                            sizeref=surface_marker_size * cone_size_multiplier,
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            opacity=opacity,
                            name=f"{well.name} surface",
                            showlegend=False,
                            legendgroup=well.name,
                            hovertemplate=(
                                f"<b>{well.name} - Surface Location</b><br>"
                                f"Type: {well_type}<br>"
                                f"Status: {status}<br>"
                                f"{fluid_info}<br>"
                                f"BHP: {well.bottom_hole_pressure:.1f} psi<br>"
                                f"Wellbore Radius: {well.radius:.2f} ft<br>"
                                f"Skin Factor: {well.skin_factor:.2f}<br>"
                                f"<br>"
                                f"Surface Coords:<br>"
                                f"  X: {x_surf:.1f} ft<br>"
                                f"  Y: {y_surf:.1f} ft<br>"
                                f"  Z: {z_surf:.1f} ft<br>"
                                "<extra></extra>"
                            ),
                        )
                    )

        # Position legend to avoid overlap with colorbar
        # Check if there's a colorbar (most volume/isosurface plots have one on the right)
        has_colorbar = any(
            hasattr(trace, "colorbar")
            and trace.colorbar is not None
            or (hasattr(trace, "showscale") and trace.showscale)
            for trace in figure.data
        )

        if has_colorbar:
            # Colorbar is typically on the right, position legend on the left
            figure.update_layout(
                legend=dict(
                    x=0.01,  # Left side
                    y=0.99,  # Top
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                    bordercolor="gray",
                    borderwidth=1,
                )
            )
        else:
            # No colorbar, use default right side position
            figure.update_layout(
                legend=dict(
                    x=0.99,  # Right side
                    y=0.99,  # Top
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                )
            )


class VolumeRenderer(BaseRenderer):
    """3D Volume renderer for scalar fields."""

    supports_physical_dimensions: typing.ClassVar[bool] = True

    def render(
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        surface_count: int = 50,
        opacity: typing.Optional[float] = None,
        isomin: typing.Optional[float] = None,
        isomax: typing.Optional[float] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        use_opacity_scaling: typing.Optional[bool] = None,
        aspect_mode: typing.Optional[str] = "auto",
        z_scale: float = 1.0,
        labels: typing.Optional["Labels"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Render a volume rendering plot.

        :param figure: Plotly figure to add the volume rendering to
        :param data: 3D data array to render
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param surface_count: Number of isosurfaces to generate for volume rendering
        :param opacity: Opacity of the volume rendering (defaults to config value). Lower values allow better visualization of internal structures.
        :param isomin: Minimum isovalue for isosurface in ORIGINAL data units. Only values >= isomin will be shown.
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param isomax: Maximum isovalue for isosurface in ORIGINAL data units. Only values <= isomax will be shown.
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).
        :param cmin: Minimum data value for color mapping in ORIGINAL data units (defaults to data min).
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param cmax: Maximum data value for color mapping in ORIGINAL data units (defaults to data max).
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).

        Use cmin/cmax to control value to color mapping in the colorbar.

        :param use_opacity_scaling: Whether to use built-in opacity scaling for better depth perception (defaults to config)
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        :param z_scale: Scale factor for Z-axis (thickness) to make layers appear thicker.
            Values > 1.0 exaggerate vertical thickness, < 1.0 compress it. Default is 1.0 (true scale).
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with volume rendering
        """
        if data.ndim != 3:
            raise ValueError("Volume rendering requires 3D data")

        use_opacity_scaling = (
            use_opacity_scaling
            if use_opacity_scaling is not None
            else self.config.use_opacity_scaling
        )

        # Store original user-provided values in original units
        original_cmin = cmin
        original_cmax = cmax
        original_isomin = isomin
        original_isomax = isomax

        # For volume rendering, we need both normalized (for volume) and display (for colorbar/hover)
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Convert user-provided values to internal processed units
        if metadata.log_scale:
            # Validate that user provided positive values for log scale
            if original_cmin is not None and original_cmin <= 0:
                raise ValueError("cmin must be > 0 for log scale data")
            if original_cmax is not None and original_cmax <= 0:
                raise ValueError("cmax must be > 0 for log scale data")
            if original_isomin is not None and original_isomin <= 0:
                raise ValueError("isomin must be > 0 for log scale data")
            if original_isomax is not None and original_isomax <= 0:
                raise ValueError("isomax must be > 0 for log scale data")

            # Convert to log₁₀ space for internal use
            cmin = np.log10(original_cmin) if original_cmin is not None else None
            cmax = np.log10(original_cmax) if original_cmax is not None else None
            isomin = np.log10(original_isomin) if original_isomin is not None else None
            isomax = np.log10(original_isomax) if original_isomax is not None else None
        else:
            # For non-log scale, use values as-is
            cmin = original_cmin
            cmax = original_cmax
            isomin = original_isomin
            isomax = original_isomax

        # Get data range for colorbar mapping using ORIGINAL values
        data_min = (
            original_cmin
            if original_cmin is not None
            else float(np.nanmin(display_data))
        )
        data_max = (
            original_cmax
            if original_cmax is not None
            else float(np.nanmax(display_data))
        )

        # Create coordinate grids - use physical coordinates if available
        if cell_dimension is not None and thickness_grid is not None:
            coordinate_offsets = kwargs.get("coordinate_offsets", None)
            x_coords, y_coords, z_coords = self.create_physical_coordinates(
                cell_dimension, thickness_grid, coordinate_offsets
            )

            # Apply Z-scaling to make layers appear thicker/thinner
            if z_scale != 1.0:
                # Scale Z coordinates relative to their mean to preserve overall position
                z_mean = np.mean(z_coords)
                z_coords = z_mean + (z_coords - z_mean) * z_scale

            # For volume rendering, we need cell centers
            nx, ny, nz = data.shape
            x_centers = (x_coords[:-1] + x_coords[1:]) / 2
            y_centers = (y_coords[:-1] + y_coords[1:]) / 2
            z_centers = np.zeros((nx, ny, nz))
            for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
                z_centers[i, j, k] = (z_coords[i, j, k] + z_coords[i, j, k + 1]) / 2

            # Create meshgrid with physical coordinates - reverse Z to show 50-300 downwards
            x, y, z = np.meshgrid(
                x_centers, y_centers, z_centers[0, 0, ::-1], indexing="ij"
            )
            x_title = "X Distance (ft)"
            y_title = "Y Distance (ft)"
            z_title = "Z Distance (ft)"
        else:
            # Fallback to index-based coordinates - reverse Z to show k=0 at top
            nx, ny, nz = data.shape
            x, y, z = np.meshgrid(
                np.arange(nx), np.arange(ny), np.arange(nz)[::-1], indexing="ij"
            )
            x_title = "X Index"
            y_title = "Y Index"
            z_title = "Z Index"

        # Extract index offsets to show original dataset indices in hover text
        coordinate_offsets = kwargs.get("coordinate_offsets", None)
        x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (0, 0, 0)

        # Create custom hover text with ORIGINAL physical values
        display_values = display_data.flatten()

        # Create hover text
        hover_text = []
        for i, j, k in itertools.product(
            range(data.shape[0]), range(data.shape[1]), range(data.shape[2])
        ):
            flat_index = i * data.shape[1] * data.shape[2] + j * data.shape[2] + k
            # Apply index offsets to show original dataset positions
            absolute_i = x_index_offset + i
            absolute_j = y_index_offset + j
            absolute_k = z_index_offset + k
            # For hover text, map k to show k=0 at top (highest Z coordinate)
            display_k = data.shape[2] - 1 - k
            absolute_display_k = (
                z_index_offset + display_k
            )  # Apply offset to display k as well
            if cell_dimension is not None and thickness_grid is not None:
                hover_text.append(
                    f"Cell: ({absolute_i}, {absolute_j}, {absolute_display_k})<br>"  # Show absolute indices
                    f"X: {x.flatten()[flat_index]:.1f} ft<br>"
                    f"Y: {y.flatten()[flat_index]:.1f} ft<br>"
                    f"Z: {z.flatten()[flat_index]:.1f} ft<br>"
                    f"{metadata.display_name}: {self.format_value(display_values[flat_index], metadata)} {metadata.unit}"
                    + (" (log scale)" if metadata.log_scale else "")
                )
            else:
                hover_text.append(
                    f"Cell: ({absolute_i}, {absolute_j}, {absolute_k})<br>"  # Show absolute indices for fallback
                    f"X: {x.flatten()[flat_index]:.1f}<br>"
                    f"Y: {y.flatten()[flat_index]:.1f}<br>"
                    f"Z: {z.flatten()[flat_index]:.1f}<br>"
                    f"{metadata.display_name}: {self.format_value(display_values[flat_index], metadata)} {metadata.unit}"
                    + (" (log scale)" if metadata.log_scale else "")
                )

        # Use the normalized data for the scale values
        # For log scale, create evenly-spaced values in LOG space for better visual distribution
        if metadata.log_scale and data_min > 0 and data_max > 0:
            # Create values evenly spaced in log space
            log_min = np.log10(data_min)
            log_max = np.log10(data_max)
            log_scale_values = np.linspace(log_min, log_max, num=6)
            scale_values = 10**log_scale_values  # Convert back to original units
        else:
            # For linear scale, use evenly spaced values
            scale_values = np.linspace(data_min, data_max, num=6)

        scale_text = [self.format_value(val, metadata=metadata) for val in scale_values]
        scale_title = f"{metadata.display_name} ({metadata.unit})" + (
            " - Log Scale" if metadata.log_scale else ""
        )
        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        volume_opacity = opacity if opacity is not None else self.config.opacity
        opacity_scale = (
            self.config.opacity_scale_values if use_opacity_scaling else None
        )
        figure.add_trace(
            go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=normalized_data.flatten(),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                opacity=volume_opacity,
                surface_count=surface_count,
                colorscale=colorscale,
                showscale=self.config.show_colorbar,
                caps=dict(x_show=True, y_show=True, z_show=True),  # Show all 6 faces
                opacityscale=opacity_scale,
                cmin=cmin,  # Use converted values for internal plotting
                cmax=cmax,  # Use converted values for internal plotting
                isomin=isomin,  # Use converted values for internal plotting
                isomax=isomax,  # Use converted values for internal plotting
                lighting=self.config.lighting,
                lightposition=self.config.light_position,
                colorbar=dict(
                    title=scale_title,
                    tickmode="array",
                    tickvals=scale_values,
                    ticktext=scale_text,
                )
                if self.config.show_colorbar
                else None,
            )
        )
        self.update_layout(
            figure,
            metadata=metadata,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
        )

        if labels is not None:
            self.apply_labels(
                figure,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=coordinate_offsets,
            )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Index",
        y_title: str = "Y Index",
        z_title: str = "Z Index",
        aspect_mode: typing.Optional[str] = None,
    ):
        """
        Update figure layout with dimensions and scene configuration.

        :param figure: Plotly figure to update
        :param metadata: Property metadata
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        """
        aspect_mode = aspect_mode or self.config.aspect_mode or "auto"
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor=self.config.paper_bgcolor,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode=aspect_mode,
                dragmode="orbit",  # Allow orbital rotation around all axes
                bgcolor=self.config.scene_bgcolor,
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
            ),
        )


class IsosurfaceRenderer(BaseRenderer):
    """3D Isosurface renderer."""

    supports_physical_dimensions: typing.ClassVar[bool] = True

    def render(
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        isomin: typing.Optional[float] = None,
        isomax: typing.Optional[float] = None,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        surface_count: int = 50,
        opacity: typing.Optional[float] = None,
        aspect_mode: typing.Optional[str] = "auto",
        z_scale: float = 1.0,
        labels: typing.Optional["Labels"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Render an isosurface plot.

        :param figure: Plotly figure to add the isosurface to
        :param data: 3D data array to create isosurfaces from
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param isomin: Minimum isovalue for isosurface in ORIGINAL data units. Only values >= isomin will be shown.
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param isomax: Maximum isovalue for isosurface in ORIGINAL data units. Only values <= isomax will be shown.
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).
        :param cmin: Minimum data value for color mapping in ORIGINAL data units (defaults to data min).
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param cmax: Maximum data value for color mapping in ORIGINAL data units (defaults to data max).
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).

        Use cmin/cmax to control value to color mapping in the colorbar.

        :param surface_count: Number of isosurfaces to generate
        :param opacity: Opacity of the isosurface (defaults to config value). Lower values allow better visualization of internal structures.
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        :param z_scale: Scale factor for Z-axis (thickness) to make layers appear thicker.
            Values > 1.0 exaggerate vertical thickness, < 1.0 compress it. Default is 1.0 (true scale).
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with isosurface plot
        """
        if data.ndim != 3:
            raise ValueError("Isosurface plotting requires 3D data")

        # Store original user-provided values in original units
        original_cmin = cmin
        original_cmax = cmax
        original_isomin = isomin
        original_isomax = isomax

        # Normalized data for isosurface calculation
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Convert user-provided values to internal processed units
        if metadata.log_scale:
            # Validate that user provided positive values for log scale
            if original_cmin is not None and original_cmin <= 0:
                raise ValueError("cmin must be > 0 for log scale data")
            if original_cmax is not None and original_cmax <= 0:
                raise ValueError("cmax must be > 0 for log scale data")
            if original_isomin is not None and original_isomin <= 0:
                raise ValueError("isomin must be > 0 for log scale data")
            if original_isomax is not None and original_isomax <= 0:
                raise ValueError("isomax must be > 0 for log scale data")

            # Convert to log₁₀ space for internal use
            cmin = np.log10(original_cmin) if original_cmin is not None else None
            cmax = np.log10(original_cmax) if original_cmax is not None else None
            isomin = np.log10(original_isomin) if original_isomin is not None else None
            isomax = np.log10(original_isomax) if original_isomax is not None else None
        else:
            # For non-log scale, use values as-is
            cmin = original_cmin
            cmax = original_cmax
            isomin = original_isomin
            isomax = original_isomax

        # Get data range for colorbar mapping using ORIGINAL values
        data_min = (
            original_cmin
            if original_cmin is not None
            else float(np.nanmin(display_data))
        )
        data_max = (
            original_cmax
            if original_cmax is not None
            else float(np.nanmax(display_data))
        )

        # Create coordinate grids - use physical coordinates if available
        if cell_dimension is not None and thickness_grid is not None:
            coordinate_offsets = kwargs.get("coordinate_offsets", None)
            x_coords, y_coords, z_coords = self.create_physical_coordinates(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                coordinate_offsets=coordinate_offsets,
            )

            # Apply Z-scaling to make layers appear thicker/thinner
            if z_scale != 1.0:
                # Scale Z coordinates relative to their mean to preserve overall position
                z_mean = np.mean(z_coords)
                z_coords = z_mean + (z_coords - z_mean) * z_scale

            # For isosurface, we need cell centers
            nx, ny, nz = data.shape
            x_centers = (x_coords[:-1] + x_coords[1:]) / 2
            y_centers = (y_coords[:-1] + y_coords[1:]) / 2
            z_centers = np.zeros((nx, ny, nz))
            for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
                z_centers[i, j, k] = (z_coords[i, j, k] + z_coords[i, j, k + 1]) / 2

            # Create meshgrid with physical coordinates - reverse Z to show 50-300 downwards
            x, y, z = np.meshgrid(
                x_centers, y_centers, z_centers[0, 0, ::-1], indexing="ij"
            )
            x_flat = x.flatten()
            y_flat = y.flatten()
            z_flat = z.flatten()

            x_title = "X Distance (ft)"
            y_title = "Y Distance (ft)"
            z_title = "Z Distance (ft)"
        else:
            # Fallback to index-based coordinates
            x = np.arange(data.shape[0])
            y = np.arange(data.shape[1])
            z = np.arange(data.shape[2])[::-1]  # Reverse to put k=0 at top
            x_flat = np.repeat(x, data.shape[1] * data.shape[2])
            y_flat = np.tile(np.repeat(y, data.shape[2]), data.shape[0])
            z_flat = np.tile(z, data.shape[0] * data.shape[1])

            x_title = "X Index"
            y_title = "Y Index"
            z_title = "Z Index"

        # Extract index offsets to show original dataset indices in hover text
        coordinate_offsets = kwargs.get("coordinate_offsets", None)
        x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (0, 0, 0)

        display_values = display_data.flatten()
        hover_text = []
        for i in range(len(x_flat)):
            # Calculate original dataset indices for this flattened index
            # Need to map back to 3D indices first
            i_3d = i // (data.shape[1] * data.shape[2])
            j_3d = (i % (data.shape[1] * data.shape[2])) // data.shape[2]
            k_3d = i % data.shape[2]

            # Apply index offsets to show original dataset positions
            absolute_i = x_index_offset + i_3d
            absolute_j = y_index_offset + j_3d
            # For k, we need to reverse the index since we reversed Z for visualization
            # k_3d=0 corresponds to the highest Z (top layer in original data)
            # k_3d=nz-1 corresponds to the lowest Z (bottom layer in original data)
            absolute_k = z_index_offset + (data.shape[2] - 1 - k_3d)

            hover_text.append(
                f"Cell: ({absolute_i}, {absolute_j}, {absolute_k})<br>"  # Show absolute indices
                f"X: {x_flat[i]:.2f}<br>"
                f"Y: {y_flat[i]:.2f}<br>"
                f"Z: {z_flat[i]:.2f}<br>"
                f"{metadata.display_name}: {self.format_value(display_values[i], metadata)} {metadata.unit}"
                + (" (log scale)" if metadata.log_scale else "")
            )

        # Use the normalized data for the scale values
        # For log scale, create evenly-spaced values in LOG space for better visual distribution
        if metadata.log_scale and data_min > 0 and data_max > 0:
            # Create values evenly spaced in log space
            log_min = np.log10(data_min)
            log_max = np.log10(data_max)
            log_scale_values = np.linspace(log_min, log_max, num=6)
            scale_values = 10**log_scale_values  # Convert back to original units
        else:
            # For linear scale, use evenly spaced values
            scale_values = np.linspace(data_min, data_max, num=6)

        scale_text = [self.format_value(val, metadata=metadata) for val in scale_values]
        scale_title = f"{metadata.display_name} ({metadata.unit})" + (
            " - Log Scale" if metadata.log_scale else ""
        )
        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        isosurface_opacity = opacity if opacity is not None else self.config.opacity
        figure.add_trace(
            go.Isosurface(
                x=x_flat,
                y=y_flat,
                z=z_flat,
                value=normalized_data.flatten(),
                isomin=isomin,
                isomax=isomax,
                cmin=cmin,
                cmax=cmax,
                opacity=isosurface_opacity,
                surface_count=surface_count,
                colorscale=colorscale,
                showscale=self.config.show_colorbar,
                lighting=self.config.lighting,
                lightposition=self.config.light_position,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                colorbar=dict(
                    title=scale_title,
                    tickmode="array",
                    tickvals=scale_values,
                    ticktext=scale_text,
                )
                if self.config.show_colorbar
                else None,
            )
        )

        self.update_layout(
            figure,
            metadata=metadata,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
        )

        if labels is not None:
            self.apply_labels(
                figure,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=coordinate_offsets,
            )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Index",
        y_title: str = "Y Index",
        z_title: str = "Z Index",
        aspect_mode: typing.Optional[str] = None,
    ) -> None:
        """
        Update figure layout with dimensions and scene configuration for isosurface plots.

        :param figure: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        """
        aspect_mode = aspect_mode or self.config.aspect_mode or "auto"
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor=self.config.paper_bgcolor,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode=aspect_mode,
                dragmode="orbit",  # Allow orbital rotation around all axes
                bgcolor=self.config.scene_bgcolor,
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
            ),
        )


def interpolate_opacity(
    x: np.ndarray, scale_values: typing.Sequence[typing.Sequence[float]]
) -> np.typing.NDArray[np.floating]:
    # scale_values: list of [fraction, opacity]
    fractions, opacities = zip(*scale_values)
    return np.interp(x, fractions, opacities)


class CellBlockRenderer(BaseRenderer):
    """
    3D Cell block renderer that renders each cell as an individual voxel/block.

    May be resource-intensive for large datasets; consider using `subsampling_factor` to reduce
    the number of rendered cells.
    """

    supports_physical_dimensions: typing.ClassVar[bool] = True

    def render(
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Tuple[float, float],
        thickness_grid: ThreeDimensionalGrid,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        subsampling_factor: int = 1,
        downsampling_factor: int = 1,
        downsampling_method: typing.Literal["mean", "max", "min"] = "mean",
        opacity: typing.Optional[float] = None,
        show_outline: typing.Optional[bool] = None,
        outline_color: typing.Optional[str] = None,
        outline_width: typing.Optional[float] = None,
        use_opacity_scaling: bool = True,
        z_scale: float = 1.0,
        labels: typing.Optional["Labels"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Render a 3D cell block plot showing each cell as an individual voxel/block.

        :param figure: Plotly figure to add the cell blocks to
        :param data: 3D data array to render
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param cmin: Minimum data value for color mapping (defaults to data min).
            This should mosttimes be the minimum of the original data range (not normalized).
            When using log scale, cmin should be > 0.
        :param cmax: Maximum data value for color mapping (defaults to data max).
            This should mosttimes be the maximum of the original data range (not normalized).
            When using log scale, cmax should be > 0.
            Use cmin/cmax to control value to color mapping in the colorbar.
        :param subsampling_factor: Factor to reduce the number of cells rendered for performance.
            A value of 1 renders every cell, 2 renders every 2nd cell in each dimension
            (reducing total cells by ~87.5%), 3 renders every 3rd cell (~96% reduction), etc.
            For example, with a 100x100x50 grid: factor=1 shows 500K cells, factor=2 shows
            ~62.5K cells, factor=5 shows ~8K cells. Use higher values for large datasets.
        :param downsampling_factor: Factor to downsample (coarsen) the data and coordinates before rendering.
            If >1, the data will be coarsened by this factor in each dimension using the specified method.
        :param downsampling_method: Method to use for downsampling: "mean", "max", or "min"
        :param opacity: Base opacity of the cell blocks (defaults to config value).
        :param show_outline: Whether to show wireframe outlines around each cell block (defaults to config)
        :param outline_color: Color of the cell block outlines (CSS color name, hex, or rgb, defaults to config)
        :param outline_width: Width/thickness of the outline wireframes in pixels (defaults to config)
        :param use_opacity_scaling: Whether to apply data-based opacity scaling for better depth perception
        :param z_scale: Scale factor for Z-axis (thickness) to make layers appear thicker.
            Values > 1.0 exaggerate vertical thickness, < 1.0 compress it. Default is 1.0 (true scale).
            Useful with aspect_mode="data" when layers appear too thin.
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with cell block visualization
        """
        if data.ndim != 3:
            raise ValueError("Cell block plotting requires 3D data")
        if subsampling_factor < 1:
            raise ValueError("Subsample factor must be at least 1")

        show_outline = (
            show_outline if show_outline is not None else self.config.show_cell_outlines
        )
        outline_color = (
            outline_color
            if outline_color is not None
            else self.config.cell_outline_color
        )
        outline_width = (
            outline_width
            if outline_width is not None
            else self.config.cell_outline_width
        )

        # Create physical coordinate grids
        coordinate_offsets = kwargs.get("coordinate_offsets", None)
        x_coords, y_coords, z_coords = self.create_physical_coordinates(
            cell_dimension, thickness_grid, coordinate_offsets
        )

        # Apply Z-scaling to make layers appear thicker/thinner
        if z_scale != 1.0:
            # Scale Z coordinates relative to their mean to preserve overall position
            z_mean = np.mean(z_coords)
            z_coords = z_mean + (z_coords - z_mean) * z_scale

        normalized_data, display_data = self.normalize_data(
            data, metadata=metadata, normalize_range=False
        )
        dx, dy = cell_dimension

        # Apply downsampling if requested. Coarsen both data and coordinates.
        if downsampling_factor > 1:
            batch_size = (downsampling_factor,) * data.ndim
            display_data, x_coords, y_coords, z_coords = coarsen_grid_and_coords(
                display_data,
                x_coords=x_coords,
                y_coords=y_coords,
                z_coords=z_coords,
                batch_size=batch_size,
                method=downsampling_method,
            )

        nx, ny, nz = display_data.shape

        # Calculate opacity scaling based on data values if enabled
        base_opacity = opacity if opacity is not None else self.config.opacity
        if use_opacity_scaling and base_opacity < 1.0:
            # Normalize data for opacity calculation (0-1 range)
            data_min = float(np.nanmin(normalized_data))
            data_max = float(np.nanmax(normalized_data))
            if data_max > data_min:
                cell_opacities = (normalized_data - data_min) / (data_max - data_min)
                # Scale the opacity values to the range [base_opacity * 0.3, base_opacity]
                # This ensures opacity scales with data while respecting the base value
                cell_opacities = interpolate_opacity(
                    cell_opacities, self.config.opacity_scale_values
                )
                # Scale to user's desired opacity range
                cell_opacities = cell_opacities * base_opacity
            else:
                cell_opacities = np.full_like(display_data, base_opacity)
        else:
            cell_opacities = np.full_like(display_data, base_opacity)

        # Extract index offsets to show original dataset indices in hover text
        x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (0, 0, 0)

        # Subsample for performance
        i_indices = range(0, nx, subsampling_factor)
        j_indices = range(0, ny, subsampling_factor)
        k_indices = range(0, nz, subsampling_factor)

        # Store original user-provided values in original units
        original_cmin = cmin
        original_cmax = cmax

        # Convert user-provided values to internal processed units
        if metadata.log_scale:
            # Validate that user provided positive values for log scale
            if original_cmin is not None and original_cmin <= 0:
                raise ValueError("cmin must be > 0 for log scale data")
            if original_cmax is not None and original_cmax <= 0:
                raise ValueError("cmax must be > 0 for log scale data")

            # Convert to log₁₀ space for internal use
            cmin = np.log10(original_cmin) if original_cmin is not None else None
            cmax = np.log10(original_cmax) if original_cmax is not None else None
        else:
            # For non-log scale, use values as-is
            cmin = original_cmin
            cmax = original_cmax

        # If cmin/cmax not provided, calculate from normalized data
        if cmin is None:
            cmin = float(np.nanmin(normalized_data))
        if cmax is None:
            cmax = float(np.nanmax(normalized_data))

        # Get data range for colorbar mapping using ORIGINAL values
        data_min = (
            original_cmin
            if original_cmin is not None
            else float(np.nanmin(display_data))
        )
        data_max = (
            original_cmax
            if original_cmax is not None
            else float(np.nanmax(display_data))
        )
        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )

        for i, j, k in itertools.product(i_indices, j_indices, k_indices):
            # Get cell boundaries
            x_min, x_max = x_coords[i], x_coords[i + 1]
            y_min, y_max = y_coords[j], y_coords[j + 1]
            z_min, z_max = z_coords[i, j, k], z_coords[i, j, k + 1]

            # Calculate cell center for hover text
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            z_center = (z_min + z_max) / 2

            # Create vertices for the cell block (cuboid)
            # In our coordinate system: k=0 starts at visual top, Z increases downward
            # So z_min is the top face (smaller Z value), z_max is the bottom face (larger Z value)
            vertices = [
                # Bottom face (z_min - higher elevation)
                [x_min, y_min, z_min],  # 0
                [x_max, y_min, z_min],  # 1
                [x_max, y_max, z_min],  # 2
                [x_min, y_max, z_min],  # 3
                # Top face (z_max - lower elevation)
                [x_min, y_min, z_max],  # 4
                [x_max, y_min, z_max],  # 5
                [x_max, y_max, z_max],  # 6
                [x_min, y_max, z_max],  # 7
            ]

            # Convert to mesh3d format
            vertices = np.array(vertices)
            x_verts = vertices[:, 0]
            y_verts = vertices[:, 1]
            z_verts = vertices[:, 2]

            # Define triangular faces with proper winding for correct normals
            # Each face needs to be defined with counter-clockwise winding when viewed from outside
            i_faces = []
            j_faces = []
            k_faces = []

            # Bottom face (z_min) - normal pointing down (away from reservoir)
            i_faces.extend([0, 0])
            j_faces.extend([2, 3])
            k_faces.extend([1, 0])

            # Top face (z_max) - normal pointing up (into reservoir)
            i_faces.extend([4, 4])
            j_faces.extend([5, 6])
            k_faces.extend([6, 7])

            # Front face (y_min)
            i_faces.extend([0, 0])
            j_faces.extend([1, 4])
            k_faces.extend([4, 5])
            i_faces.extend([1, 1])
            j_faces.extend([5, 4])
            k_faces.extend([4, 0])

            # Back face (y_max)
            i_faces.extend([2, 2])
            j_faces.extend([6, 7])
            k_faces.extend([7, 3])
            i_faces.extend([6, 6])
            j_faces.extend([3, 7])
            k_faces.extend([7, 2])

            # Left face (x_min)
            i_faces.extend([0, 0])
            j_faces.extend([3, 7])
            k_faces.extend([7, 4])
            i_faces.extend([3, 3])
            j_faces.extend([4, 7])
            k_faces.extend([7, 0])

            # Right face (x_max)
            i_faces.extend([1, 1])
            j_faces.extend([2, 6])
            k_faces.extend([6, 5])
            i_faces.extend([2, 2])
            j_faces.extend([5, 6])
            k_faces.extend([6, 1])

            normalized_cell_value = normalized_data[i, j, k]  # For color mapping
            cell_value = display_data[i, j, k]  # For hover text
            cell_opacity = cell_opacities[i, j, k]

            # Create hover text for this cell
            hover_text = (
                f"Cell ({x_index_offset + i}, {y_index_offset + j}, {z_index_offset + k})<br>"
                f"X: {x_center:.2f} ft<br>"
                f"Y: {y_center:.2f} ft<br>"
                f"Z: {z_center:.2f} ft<br>"
                f"{metadata.display_name}: {self.format_value(cell_value, metadata)} {metadata.unit}"
                + (" (log scale)" if metadata.log_scale else "")
                + "<br>"
                f"Cell Size: {dx:.1f} x {dy:.1f} x {thickness_grid[i, j, k]:.2f} ft<br>"
                f"Opacity: {cell_opacity:.2f}"
            )

            # Create the main cell block with hover enabled directly on the mesh
            # Apply hover text to all vertices so hover works on all faces
            vertex_hover_text = [hover_text] * len(x_verts)

            figure.add_trace(
                go.Mesh3d(
                    x=x_verts,
                    y=y_verts,
                    z=z_verts,
                    i=i_faces,
                    j=j_faces,
                    k=k_faces,
                    intensity=np.full(len(x_verts), normalized_cell_value),
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    opacity=cell_opacity,
                    showscale=False,  # Only show colorbar on the last trace
                    lighting=self.config.lighting,
                    lightposition=self.config.light_position,
                    text=vertex_hover_text,  # Enables hover on all vertices
                    hovertemplate="%{text}<extra></extra>",
                    # Properties for better rendering, sharper, more defined blocks
                    flatshading=True,  # Flat shading for more defined edges
                    alphahull=0,  # Disable alphahull to reduce haloing
                )
            )

            # Add wireframe outline if requested
            if show_outline:
                # Define edges of the cuboid for wireframe
                edges = [
                    # Bottom face edges
                    (
                        [x_min, x_max],
                        [y_min, y_min],
                        [z_min, z_min],
                    ),  # front bottom
                    (
                        [x_max, x_max],
                        [y_min, y_max],
                        [z_min, z_min],
                    ),  # right bottom
                    (
                        [x_max, x_min],
                        [y_max, y_max],
                        [z_min, z_min],
                    ),  # back bottom
                    (
                        [x_min, x_min],
                        [y_max, y_min],
                        [z_min, z_min],
                    ),  # left bottom
                    # Top face edges
                    (
                        [x_min, x_max],
                        [y_min, y_min],
                        [z_max, z_max],
                    ),  # front top
                    (
                        [x_max, x_max],
                        [y_min, y_max],
                        [z_max, z_max],
                    ),  # right top
                    (
                        [x_max, x_min],
                        [y_max, y_max],
                        [z_max, z_max],
                    ),  # back top
                    (
                        [x_min, x_min],
                        [y_max, y_min],
                        [z_max, z_max],
                    ),  # left top
                    # Vertical edges
                    (
                        [x_min, x_min],
                        [y_min, y_min],
                        [z_min, z_max],
                    ),  # front left
                    (
                        [x_max, x_max],
                        [y_min, y_min],
                        [z_min, z_max],
                    ),  # front right
                    (
                        [x_max, x_max],
                        [y_max, y_max],
                        [z_min, z_max],
                    ),  # back right
                    (
                        [x_min, x_min],
                        [y_max, y_max],
                        [z_min, z_max],
                    ),  # back left
                ]

                # Add each edge as a line
                for x_edge, y_edge, z_edge in edges:
                    figure.add_trace(
                        go.Scatter3d(
                            x=x_edge,
                            y=y_edge,
                            z=z_edge,
                            mode="lines",
                            line=dict(
                                color=outline_color,
                                width=outline_width,
                            ),
                            showlegend=False,
                            hoverinfo="skip",  # Don't show hover for outline
                        )
                    )

        # Add a dummy trace for the colorbar
        if self.config.show_colorbar:
            # Use the normalized data for the scale values
            # For log scale, create evenly-spaced values in LOG space for better visual distribution
            if metadata.log_scale and data_min > 0 and data_max > 0:
                # Create values evenly spaced in log space
                log_min = np.log10(data_min)
                log_max = np.log10(data_max)
                log_scale_values = np.linspace(log_min, log_max, num=6)
                scale_values = 10**log_scale_values  # Convert back to original units
            else:
                # For linear scale, use evenly spaced values
                scale_values = np.linspace(data_min, data_max, num=6)

            scale_text = [
                self.format_value(val, metadata=metadata) for val in scale_values
            ]
            scale_title = f"{metadata.display_name} ({metadata.unit})" + (
                " - Log Scale" if metadata.log_scale else ""
            )

            figure.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(
                        size=0.1,
                        color=[cmin, cmax],  # Use normalized range (already set above)
                        colorscale=colorscale,
                        cmin=cmin,  # Match cell blocks
                        cmax=cmax,  # Match cell blocks
                        colorbar=dict(
                            title=scale_title,
                            tickmode="array",
                            tickvals=scale_values,  # Positions within the normalized range
                            ticktext=scale_text,
                        ),
                        opacity=0,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        self.update_layout(
            figure,
            metadata,
            x_title="X Distance (ft)",
            y_title="Y Distance (ft)",
            z_title="Z Distance (ft)",
        )

        if labels is not None:
            self.apply_labels(
                figure,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=kwargs.get("coordinate_offsets", None),
            )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Distance (ft)",
        y_title: str = "Y Distance (ft)",
        z_title: str = "Z Distance (ft)",
    ):
        """
        Update figure layout with dimensions and scene configuration for cell block plots.

        :param figure: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        """
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor=self.config.paper_bgcolor,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode="data",  # Preserve actual aspect ratios
                dragmode="orbit",  # Allow orbital rotation around all axes
                bgcolor=self.config.scene_bgcolor,
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
            ),
        )


class Scatter3DRenderer(BaseRenderer):
    """3D Scatter renderer for sparse data."""

    supports_physical_dimensions = True

    def render(
        self,
        figure: go.Figure,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        threshold: float = 0.0,
        sample_rate: float = 1.0,
        marker_size: int = 4,
        cmin: typing.Optional[float] = None,
        cmax: typing.Optional[float] = None,
        opacity: typing.Optional[float] = None,
        aspect_mode: typing.Optional[str] = "auto",
        z_scale: float = 1.0,
        labels: typing.Optional["Labels"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Render a 3D scatter plot for sparse data visualization.

        :param figure: Plotly figure to add the scatter plot to
        :param data: 3D data array to create scatter plot from
        :param metadata: Property metadata for labeling and scaling
        :param threshold: Value threshold for point inclusion (in normalized range 0.0-1.0)
        :param sample_rate: Fraction of points to sample for performance (0.0-1.0)

        Both threshold and sample_rate are applied to the normalized data to control point inclusion and rendering.
        This can be useful for large datasets where you want to visualize only significant points and reduce rendering load.

        :param marker_size: Size of scatter plot markers
        :param cmin: Minimum data value for color mapping in ORIGINAL data units (defaults to data min).
            For log-scale properties, provide the original value (e.g., 0.1 cP), not log₁₀(0.1).
        :param cmax: Maximum data value for color mapping in ORIGINAL data units (defaults to data max).
            For log-scale properties, provide the original value (e.g., 1000 cP), not log₁₀(1000).

        Use cmin/cmax to control value to color mapping in the colorbar.

        :param opacity: Opacity of the markers (defaults to config value)
        :param aspect_mode: Aspect mode for the 3D plot (default is "auto"). Could be any of "cube", "auto", or "data".
        :param z_scale: Scale factor for Z-axis (thickness) to make layers appear thicker.
            Values > 1.0 exaggerate vertical thickness, < 1.0 compress it. Default is 1.0 (true scale).
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with 3D scatter plot
        """
        if data.ndim != 3:
            raise ValueError("Scatter3D plotting requires 3D data")

        # Store original user-provided values in original units
        original_cmin = cmin
        original_cmax = cmax

        # Normalize data for consistent scaling and thresholding
        # Original data will be used for hover text and colorbar
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Convert user-provided values to internal processed units
        if metadata.log_scale:
            # Validate that user provided positive values for log scale
            if original_cmin is not None and original_cmin <= 0:
                raise ValueError("cmin must be > 0 for log scale data")
            if original_cmax is not None and original_cmax <= 0:
                raise ValueError("cmax must be > 0 for log scale data")

            # Convert to log₁₀ space for internal use
            cmin = np.log10(original_cmin) if original_cmin is not None else None
            cmax = np.log10(original_cmax) if original_cmax is not None else None
        else:
            # For non-log scale, use values as-is
            cmin = original_cmin
            cmax = original_cmax

        # Get data range for colorbar mapping using ORIGINAL values
        data_min = (
            original_cmin
            if original_cmin is not None
            else float(np.nanmin(display_data))
        )
        data_max = (
            original_cmax
            if original_cmax is not None
            else float(np.nanmax(display_data))
        )

        # Subsample data for performance
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
            z_coords_raw = indices[2][sample_indices]
            values = display_data[
                x_coords, y_coords, z_coords_raw
            ]  # Use original values for display
        else:
            x_coords, y_coords, z_coords_raw = indices
            values = display_data[mask]  # Use original values for display

        if values.size == 0:
            raise ValueError(
                "No data points selected for Scatter3D plot. "
                "Check your threshold, sample rate or data values."
            )

        # Apply numpy convention: z=0 should be at top, z increases downward
        z_coords = data.shape[2] - 1 - z_coords_raw

        if cell_dimension is not None and thickness_grid is not None:
            # Convert indices to physical coordinates
            dx, dy = cell_dimension
            coordinate_offsets = kwargs.get("coordinate_offsets", None)
            x_offset, y_offset, _ = coordinate_offsets or (0, 0, 0)

            x_physical = x_offset * dx + x_coords * dx
            y_physical = y_offset * dy + y_coords * dy

            _, _, z_boundaries = self.create_physical_coordinates(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                coordinate_offsets=coordinate_offsets,
            )

            # Apply Z-scaling to make layers appear thicker/thinner
            if z_scale != 1.0:
                # Scale Z coordinates relative to their mean to preserve overall position
                z_mean = np.mean(z_boundaries)
                z_boundaries = z_mean + (z_boundaries - z_mean) * z_scale

            z_physical = np.array(
                [
                    (z_boundaries[x, y, z] + z_boundaries[x, y, z + 1]) / 2
                    for x, y, z in zip(x_coords, y_coords, z_coords)
                ]
            )

            # Extract index offsets to show original dataset indices in hover text
            x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (
                0,
                0,
                0,
            )

            hover_text = [
                f"Cell: ({x_index_offset + x_coords[i]}, {y_index_offset + y_coords[i]}, {z_index_offset + z_coords[i]})<br>"  # Show absolute indices
                f"X: {x_physical[i]:.1f} ft<br>"
                f"Y: {y_physical[i]:.1f} ft<br>"
                f"Z: {z_physical[i]:.1f} ft<br>"
                f"{metadata.display_name}: {self.format_value(v, metadata)}"
                + (" (log scale)" if metadata.log_scale else "")
                for i, v in enumerate(values)
            ]
            # Use physical coordinates for axis titles
            x_title = "X Distance (ft)"
            y_title = "Y Distance (ft)"
            z_title = "Z Distance (ft)"
        else:
            # Use index coordinates directly
            x_physical = x_coords
            y_physical = y_coords
            z_physical = z_coords_raw

            # Extract index offsets to show original dataset indices in hover text
            x_index_offset, y_index_offset, z_index_offset = kwargs.get(
                "coordinate_offsets", None
            ) or (0, 0, 0)

            hover_text = [
                f"Cell: ({x_index_offset + x_coords[i]}, {y_index_offset + y_coords[i]}, {z_index_offset + z_coords[i]})<br>"  # Show absolute indices
                f"{metadata.display_name}: {self.format_value(v, metadata)}"
                + (" (log scale)" if metadata.log_scale else "")
                for i, v in enumerate(values)
            ]

            x_title = "X Cell Index"
            y_title = "Y Cell Index"
            z_title = "Z Cell Index"

        # Use the normalized data for the scale values
        # For log scale, create evenly-spaced values in LOG space for better visual distribution
        if metadata.log_scale and data_min > 0 and data_max > 0:
            # Create values evenly spaced in log space
            log_min = np.log10(data_min)
            log_max = np.log10(data_max)
            log_scale_values = np.linspace(log_min, log_max, num=6)
            scale_values = 10**log_scale_values  # Convert back to original units
        else:
            # For linear scale, use evenly spaced values
            scale_values = np.linspace(data_min, data_max, num=6)

        scale_text = [self.format_value(val, metadata=metadata) for val in scale_values]
        scale_title = f"{metadata.display_name} ({metadata.unit})" + (
            " - Log Scale" if metadata.log_scale else ""
        )
        colorscale = self.get_colorscale(
            kwargs.get("color_scheme", metadata.color_scheme)
        )
        scatter_opacity = opacity if opacity is not None else self.config.opacity
        figure.add_trace(
            go.Scatter3d(
                x=x_physical,
                y=y_physical,
                z=z_physical,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=values,
                    colorscale=colorscale,
                    opacity=scatter_opacity,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(
                        title=scale_title,
                        tickmode="array",
                        tickvals=scale_values,
                        ticktext=scale_text,
                    )
                    if self.config.show_colorbar
                    else None,
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        self.update_layout(
            figure,
            metadata=metadata,
            aspect_mode=aspect_mode,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
        )

        if labels is not None:
            self.apply_labels(
                figure,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                coordinate_offsets=kwargs.get("coordinate_offsets", None),
                format_kwargs=kwargs.get("format_kwargs", None),
            )
        return figure

    def update_layout(
        self,
        figure: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Cell Index",
        y_title: str = "Y Cell Index",
        z_title: str = "Z Cell Index",
        aspect_mode: typing.Optional[str] = None,
    ) -> None:
        """
        Update figure layout with dimensions and scene configuration for scatter plots.

        :param figure: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param aspect_mode: Aspect mode for the 3D plot (default is "auto"). Could be any of "cube", "auto", or "data".
        """
        figure.update_layout(
            width=self.config.width,
            height=self.config.height,
            paper_bgcolor=self.config.paper_bgcolor,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode=aspect_mode or self.config.aspect_mode or "auto",
                dragmode="orbit",  # Allow orbital rotation around all axes
                bgcolor=self.config.scene_bgcolor,
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="gray",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="lightgray",
                    showbackground=True,
                    zerolinecolor="white",
                ),
            ),
        )


PLOT_TYPE_NAMES: typing.Dict[PlotType, str] = {
    PlotType.VOLUME: "3D Volume",
    PlotType.ISOSURFACE: "3D Isosurface",
    PlotType.SCATTER_3D: "3D Scatter",
    PlotType.CELL_BLOCKS: "3D Cell Blocks",
}

_missing = object()


class ModelVisualizer:
    """
    3D visualizer for reservoir model state and properties.
    """

    default_dashboard_title: typing.ClassVar[str] = "Model Properties"

    def __init__(
        self,
        config: typing.Optional[PlotConfig] = None,
        registry: typing.Optional[PropertyRegistry] = None,
    ) -> None:
        """
        Initialize the visualizer with optional configuration.

        :param config: Optional configuration for 3D rendering (uses defaults if None)
        :param registry: Optional fluid property registry (uses default if None)
        """
        self.config = config or PlotConfig()
        self._renderers: typing.Dict[PlotType, BaseRenderer] = {
            PlotType.VOLUME: VolumeRenderer(self.config),
            PlotType.ISOSURFACE: IsosurfaceRenderer(self.config),
            PlotType.SCATTER_3D: Scatter3DRenderer(self.config),
            PlotType.CELL_BLOCKS: CellBlockRenderer(self.config),
        }
        self.registry = registry or property_registry

    def add_renderer(
        self,
        plot_type: PlotType,
        renderer_type: typing.Type[BaseRenderer],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """
        Add a custom renderer for a specific plot type.

        :param plot_type: The type of plot to add (must be a valid PlotType)
        :param renderer_type: The class implementing the ``BaseRenderer`` interface
        :param args: Initialization arguments for the renderer class
        :param kwargs: Initialization keyword arguments for the renderer class
        :raises ValueError: If plot_type is not a valid PlotType
        """
        if not isinstance(plot_type, PlotType):
            raise ValueError(f"Invalid plot type: {plot_type}")
        self._renderers[plot_type] = renderer_type(self.config, *args, **kwargs)

    def get_renderer(self, plot_type: PlotType) -> BaseRenderer:
        """
        Get the renderer for a specific plot type.

        :param plot_type: The type of plot to get (must be a valid PlotType)
        :return: The renderer instance for the specified plot type
        :raises ValueError: If plot_type is not a valid PlotType or no renderer is registered
        """
        if not isinstance(plot_type, PlotType):
            raise ValueError(f"Invalid plot type: {plot_type}")
        renderer = self._renderers.get(plot_type, None)
        if renderer is None:
            raise ValueError(f"No renderer registered for plot type: {plot_type}")
        return renderer

    def apply_slice(
        self,
        data: ThreeDimensionalGrid,
        x_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        y_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        z_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
    ) -> typing.Tuple[ThreeDimensionalGrid, typing.Tuple[slice, slice, slice]]:
        """
        Validate and apply slicing to 3D data, ensuring result is still 3D.

        :param data: Input 3D data array
        :param x_slice: X dimension slice specification
        :param y_slice: Y dimension slice specification
        :param z_slice: Z dimension slice specification
        :return: Tuple of (sliced_data, actual_slices_used)
        :raises ValueError: If slicing would result in non-3D data
        """
        nx, ny, nz = data.shape

        def normalize_slice_spec(
            spec: typing.Any, dim_size: int, dim_name: str
        ) -> slice:
            """Convert various slice specifications to a slice object."""
            if spec is None:
                return slice(None)  # Full dimension
            elif isinstance(spec, int):
                # Single index - convert to slice to maintain dimension
                if spec < 0:
                    spec = dim_size + spec  # Handle negative indexing
                if spec < 0 or spec >= dim_size:
                    raise ValueError(
                        f"{dim_name}_slice index {spec} out of range [0, {dim_size - 1}]"
                    )
                return slice(spec, spec + 1)  # Single element slice
            elif isinstance(spec, tuple) and len(spec) == 2:
                start, stop = spec
                if start < 0:
                    start = dim_size + start
                if stop < 0:
                    stop = dim_size + stop
                return slice(start, stop)
            elif isinstance(spec, slice):
                return spec
            else:
                raise ValueError(f"Invalid {dim_name}_slice specification: {spec}")

        # Normalize all slice specifications
        x_slice_obj = normalize_slice_spec(x_slice, nx, "x")
        y_slice_obj = normalize_slice_spec(y_slice, ny, "y")
        z_slice_obj = normalize_slice_spec(z_slice, nz, "z")

        # Apply slicing
        sliced_data = data[x_slice_obj, y_slice_obj, z_slice_obj]

        # Validate result is still 3D
        if sliced_data.ndim != 3:
            raise ValueError(
                f"Slicing resulted in {sliced_data.ndim}D data. "
                f"All slice specifications must preserve 3D structure. "
                f"Result shape: {sliced_data.shape}"
            )

        # Check minimum size requirements
        if any(dim < 1 for dim in sliced_data.shape):
            raise ValueError(
                f"Slicing resulted in empty dimension(s). Shape: {sliced_data.shape}"
            )
        return typing.cast(ThreeDimensionalGrid, sliced_data), (
            x_slice_obj,
            y_slice_obj,
            z_slice_obj,
        )

    def _get_state(
        self, model_state: ModelState[ThreeDimensions], name: str
    ) -> ThreeDimensionalGrid:
        """
        Get property data from model state.

        :param model_state: The model state containing reservoir model
        :param name: Name of the property to extract as defined by the `PropertyMetadata.name`
        :return: A three-dimensional numpy array containing the state data
        :raises AttributeError: If property is not found in reservoir model properties
        :raises TypeError: If property is not a numpy array
        """
        state = asdict(model_state, recurse=False)
        name_parts = name.split(".")
        data = None
        if len(name_parts) == 1:
            data = state.get(name_parts[0], _missing)
        else:
            # Nested property access (e.g., "permeability.x")
            data = state
            for part in name_parts:
                if isinstance(data, Mapping):
                    value = data.get(part, _missing)
                else:
                    value = getattr(data, part, _missing)

                if value is not _missing:
                    data = value
                    continue
                raise AttributeError(f"Property '{name}' not found in model state.")

        if data is None or data is _missing:
            raise AttributeError(
                f"Property '{name}' not found on model state or property value is invalid."
            )
        elif isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float64)

        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise TypeError(f"Property '{name}' is not a 3 dimensional array.")
        return typing.cast(ThreeDimensionalGrid, data)

    def get_title(
        self,
        plot_type: PlotType,
        metadata: PropertyMetadata,
        custom_title: typing.Optional[str] = None,
    ) -> str:
        """
        Smart title determination logic that includes plot type information.

        :param custom_title: Custom title provided by user
        :param plot_type: Type of plot being created
        :param metadata: Data metadata
        :return: Final title to use
        """
        # If the custom title is providd and it is not some kind of suffix (starts with "-"), use it directly
        if custom_title is not None and not custom_title.strip().startswith("-"):
            return custom_title
        if self.config.title:
            # If config has a title, add plot type as subtitle info
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "3D Plot")
            return f"{self.config.title}{custom_title or ''}<br><sub>{plot_name}: {metadata.display_name}</sub>"
        plot_name = PLOT_TYPE_NAMES.get(plot_type, "3D Plot")
        return f"{plot_name}{custom_title or ''}: {metadata.display_name}<br><sub>Interactive 3D Visualization</sub>"

    def make_plot(
        self,
        model_state: ModelState[ThreeDimensions],
        property: str,
        plot_type: typing.Optional[typing.Union[PlotType, str]] = None,
        figure: typing.Optional[go.Figure] = None,
        title: typing.Optional[str] = None,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        x_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        y_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        z_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        labels: typing.Optional["Labels"] = None,
        show_wells: bool = False,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Plot a specific model property available on the model state in 3D with optional data slicing.

        :param model_state: The model state containing the reservoir model data
        :param property: Name of the property to plot (from `PropertyRegistry`)
        :param plot_type: Type of 3D plot to create (volume, isosurface, slice, scatter, cell_blocks)
        :param figure: Optional existing Plotly figure to add to (creates new if None)
        :param title: Custom title for this plot (overrides config title)
        :param width: Custom width for this plot (overrides config width)
        :param height: Custom height for this plot (overrides config height)
        :param x_slice: X dimension slice specification:
            - int: Single index (e.g., 5 for cells[5:6, :, :])
            - tuple: Range (e.g., (10, 20) for cells[10:20, :, :])
            - slice: Full slice object (e.g., slice(10, 20, 2))
            - None: Use full dimension
        :param y_slice: Y dimension slice specification (same format as x_slice)
        :param z_slice: Z dimension slice specification (same format as x_slice)
        :param labels: Optional collection of labels to add to the plot
        :param show_wells: Whether to add well visualizations to the plot (default: False)
        :param kwargs: Additional plotting parameters specific to the plot type. Can also include
            well visualization kwargs: injection_color, production_color, shut_in_color,
            wellbore_width, surface_marker_size, show_wellbore, show_surface_marker, show_perforations.
            See WellKwargs TypedDict for details.
        :return: Plotly figure object containing the 3D visualization

        Usage Examples:

        ```python
        # Plot only cells 10-20 in X direction
        viz.make_plot(state, "pressure", x_slice=(10, 20))

        # Plot with string plot type (converted internally)
        viz.make_plot(state, "pressure", plot_type="volume_render")
        viz.make_plot(state, "pressure", plot_type="cell_blocks")

        # Plot single layer at Z index 5
        viz.make_plot(state, "oil_saturation", z_slice=5)

        # Plot corner section
        viz.make_plot(state, "temperature", x_slice=(0, 25), y_slice=(0, 25), z_slice=(0, 10))

        # Use slice objects for advanced slicing
        viz.make_plot(state, "viscosity", x_slice=slice(10, 50, 2))  # Every 2nd cell

        # Add well visualization with default styling
        viz.make_plot(state, "pressure", show_wells=True)

        # Customize well visualization using kwargs
        viz.make_plot(
            state, "oil_saturation",
            show_wells=True,
            injection_color='#ff6b6b',
            production_color='#51cf66',
            wellbore_width=8.0,
            show_surface_marker=True
        )
        ```
        """
        if isinstance(plot_type, str):
            try:
                plot_type = PlotType(plot_type)
            except ValueError:
                raise ValueError(
                    f"Invalid plot_type string: '{plot_type}'. "
                    f"Valid options are: {', '.join(pt.value for pt in PlotType)}"
                )

        metadata = self.registry[property]
        data = self._get_state(model_state, metadata.name)

        # Get original cell dimensions and height grid
        cell_dimension = model_state.model.cell_dimension
        thickness_grid = model_state.model.thickness_grid

        # Apply slicing if any slice parameters are provided
        coordinate_offsets = None
        if any(s is not None for s in [x_slice, y_slice, z_slice]):
            data, normalized_slices = self.apply_slice(data, x_slice, y_slice, z_slice)
            x_slice_obj, y_slice_obj, z_slice_obj = normalized_slices

            # Update title to indicate slicing
            slice_info = []
            if x_slice is not None:
                slice_info.append(
                    f"X[{x_slice_obj.start or 0}:{x_slice_obj.stop or data.shape[0]}]"
                )
            if y_slice is not None:
                slice_info.append(
                    f"Y[{y_slice_obj.start or 0}:{y_slice_obj.stop or data.shape[1]}]"
                )
            if z_slice is not None:
                slice_info.append(
                    f"Z[{z_slice_obj.start or 0}:{z_slice_obj.stop or data.shape[2]}]"
                )

            slice_suffix = f" - Slice: {', '.join(slice_info)}"
            if title:
                title += slice_suffix
            else:
                title = slice_suffix  # Will be used by get_title

            # Calculate coordinate offsets for sliced data (cell index offsets)
            coordinate_offsets = (
                x_slice_obj.start or 0,
                y_slice_obj.start or 0,
                z_slice_obj.start or 0,
            )

            # If we sliced the data, we need to slice the thickness_grid as well for physical coordinates
            if thickness_grid is not None:
                thickness_grid = thickness_grid[x_slice_obj, y_slice_obj, z_slice_obj]  # type: ignore

        plot_type = plot_type or self.config.plot_type
        renderer = self.get_renderer(plot_type)

        # Pass coordinate offsets to renderers
        if coordinate_offsets is not None:
            kwargs["coordinate_offsets"] = coordinate_offsets

        # Pass labels to renderers that support them
        if labels is not None:
            kwargs["labels"] = labels

        fig = figure or go.Figure()
        # Pass physical dimensions to renderers that support them
        if renderer.supports_physical_dimensions:
            fig = renderer.render(
                fig,
                data,
                metadata,
                cell_dimension=kwargs.pop("cell_dimension", cell_dimension),
                thickness_grid=kwargs.pop("thickness_grid", thickness_grid),
                **kwargs,
            )
        else:
            fig = renderer.render(fig, data, metadata, **kwargs)

        # Add well visualization if requested
        if show_wells and model_state.has_wells():
            # Extract z_scale for well rendering (default to 1.0 if not specified)
            z_scale = kwargs.get("z_scale", 1.0)

            # Extract well visualization kwargs from the kwargs dict using TypedDict keys
            well_kwargs: WellKwargs = {}
            for key in WellKwargs.__annotations__.keys():
                if key in kwargs:
                    well_kwargs[key] = kwargs.pop(key)  # type: ignore

            # Add wells to the figure using the renderer's method
            logger.debug(
                f"Rendering wells: {len(model_state.wells.injection_wells)} injection, "
                f"{len(model_state.wells.production_wells)} production"
            )
            renderer.render_wells(
                figure=fig,
                wells=model_state.wells,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                coordinate_offsets=coordinate_offsets,
                z_scale=z_scale,
                **well_kwargs,
            )

        final_title = self.get_title(plot_type, metadata, title)

        # Apply final layout updates
        layout_updates: typing.Dict[str, typing.Any] = {"title": final_title}
        if width is not None:
            layout_updates["width"] = width
        if height is not None:
            layout_updates["height"] = height

        fig.update_layout(**layout_updates)
        return fig

    def animate(
        self,
        model_states: typing.Sequence[ModelState[ThreeDimensions]],
        property: str,
        plot_type: typing.Optional[typing.Union[PlotType, str]] = None,
        frame_duration: int = 200,
        step_size: int = 1,
        width: typing.Optional[int] = None,
        height: typing.Optional[int] = None,
        title: typing.Optional[str] = None,
        x_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        y_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        z_slice: typing.Optional[
            typing.Union[int, slice, typing.Tuple[int, int]]
        ] = None,
        labels: typing.Optional["Labels"] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create an animated plot showing property evolution over time.

        :param model_states: Sequence of model states representing time steps
        :param property: Name of the property to animate
        :param plot_type: Type of 3D plot for animation frames
        :param frame_duration: Duration of each frame in milliseconds
        :param step_size: Step size to skip frames for performance (1 = every frame)
        :param width: Custom width for the animation
        :param height: Custom height for the animation
        :param title: Custom title for the animation
        :param x_slice: X dimension slice specification:
            - int: Single index (e.g., 5 for cells[5:6, :, :])
            - tuple: Range (e.g., (10, 20) for cells[10:20, :, :])
            - slice: Full slice object (e.g., slice(10, 20, 2))
            - None: Use full dimension
        :param y_slice: Y dimension slice specification (same format as x_slice)
        :param z_slice: Z dimension slice specification (same format as x_slice)
        :param labels: Optional collection of labels to add to the animation
        :param kwargs: Additional parameters for individual property plots.
        :return: Animated Plotly figure with time controls
        """
        if not model_states:
            raise ValueError("No model states provided")

        # Convert string plot_type to PlotType enum if needed
        if isinstance(plot_type, str):
            try:
                plot_type = PlotType(plot_type)
            except ValueError:
                raise ValueError(
                    f"Invalid plot_type string: '{plot_type}'. "
                    f"Valid options are: {', '.join(pt.value for pt in PlotType)}"
                )

        metadata = self.registry[property]
        plot_type = plot_type or PlotType.VOLUME

        if "cmin" not in kwargs:
            # Add cmin/cmax to kwargs for consistent color mapping across frames
            data_list = [
                self._get_state(state, metadata.name) for state in model_states
            ]
            cmin = float(np.nanmin([np.nanmin(data) for data in data_list]))
            cmax = float(np.nanmax([np.nanmax(data) for data in data_list]))
            kwargs["cmin"] = cmin
            kwargs["cmax"] = cmax
            logger.debug(f"Animation cmin: {cmin}, cmax: {cmax}")

        # Create base figure from first state
        base_fig = self.make_plot(
            model_states[0],
            property=property,
            plot_type=plot_type,
            width=width,
            height=height,
            x_slice=x_slice,
            y_slice=y_slice,
            z_slice=z_slice,
            labels=labels,
            **kwargs,
        )

        # Create frames for animation. Ensure we get all states
        frames = []
        state_count = len(model_states)
        for i in range(0, state_count, step_size):
            state = model_states[i]
            frame_fig = self.make_plot(
                state,
                property=property,
                plot_type=plot_type,
                width=width,
                height=height,
                x_slice=x_slice,
                y_slice=y_slice,
                z_slice=z_slice,
                labels=labels,
                **kwargs,
            )

            # Create frame title with appropriate time units
            if state.time >= 3600:  # More than 1 hour
                time_str = f"t={state.time / 3600:.2f} hours"
            elif state.time >= 60:  # More than 1 minute
                time_str = f"t={state.time / 60:.2f} minutes"
            else:
                time_str = f"t={state.time:.2f} seconds"

            # Extract annotations from the frame figure if they exist
            frame_layout: typing.Dict[str, typing.Any] = {
                "title": f"{metadata.display_name} at {time_str}"
            }

            if labels is not None:
                try:
                    # Extract annotations from the frame figure's scene layout
                    scene_layout = getattr(frame_fig.layout, "scene", None)
                    if scene_layout and hasattr(scene_layout, "annotations"):
                        annotations = list(scene_layout.annotations or [])
                        if annotations:
                            frame_layout["scene"] = {"annotations": annotations}
                            logger.debug(
                                f"Extracted {len(annotations)} annotations for frame {i}"
                            )
                        else:
                            frame_layout["scene"] = {"annotations": []}
                    else:
                        frame_layout["scene"] = {"annotations": []}
                except Exception as exc:
                    logger.warning(
                        f"Failed to extract annotations for frame {i}: {exc}"
                    )
                    frame_layout["scene"] = {"annotations": []}

            frame = go.Frame(
                data=frame_fig.data,
                name=f"frame_{i}",
                layout=frame_layout,
            )
            frames.append(frame)

        # Add frames to figure
        base_fig.frames = frames

        # Set initial annotations for the base figure (from first frame)
        if frames and labels is not None:
            try:
                if scene_layout := getattr(frames[0].layout, "scene", None) is not None:
                    initial_annotations = getattr(scene_layout, "annotations", [])
                    if initial_annotations:
                        base_fig.update_layout(
                            scene=dict(annotations=initial_annotations)
                        )
                        logger.debug(
                            f"Set {len(initial_annotations)} initial annotations on base figure"
                        )
            except Exception as exc:
                logger.warning(f"Failed to set initial annotations: {exc}")

        # Set final title for animation
        if title:
            animation_title = title
        elif self.config.title:
            animation_title = self.config.title
        else:
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "3D")
            animation_title = f"{plot_name} Animation: {metadata.display_name}"

        # Add animation controls
        base_fig.update_layout(
            title=animation_title,
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
                        "font": {"size": 18},
                        "prefix": "Time Step:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 30, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [f"frame_{i}"],
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": True,
                                    },
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


viz = ModelVisualizer()
"""Default 3D reservoir model visualizer instance."""
