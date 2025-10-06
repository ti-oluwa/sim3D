"""
PyVista-based 3D Visualization Suite for Reservoir Simulation

This module provides a comprehensive PyVista-based visualization factory for 3D reservoir simulation data,
including interactive 3D plots, volume rendering, cross-sections, animations, and advanced visual effects.
"""

import itertools
import numpy as np
import logging
from abc import ABC, abstractmethod
from enum import Enum
from attrs import define, field, asdict
from typing import (
    ClassVar,
    Mapping,
    Optional,
    Type,
    Union,
    Tuple,
    List,
    Dict,
    Any,
    Sequence,
    Iterator,
    cast,
    Literal,
)
from typing_extensions import TypedDict

import pyvista as pv

from sim3D.models import ReservoirModel
from sim3D.simulation import ModelState
from sim3D.types import ThreeDimensions, ThreeDimensionalGrid
from sim3D.visualization.base import (
    ColorScheme,
    PropertyMetadata,
    PropertyRegistry,
    property_registry,
)

logger = logging.getLogger(__name__)

# Constants for reusable values
DEFAULT_GRID_SPACING: Tuple[int, int, int] = (10, 10, 5)
DEFAULT_N_CONTOURS: int = 5
DEFAULT_N_STREAMLINES: int = 50
DEFAULT_SAMPLE_PERCENTILES: Tuple[float, float] = (1.0, 99.0)
DEFAULT_OPACITY_VALUES: List[float] = [0.0, 0.2, 0.4, 0.7, 1.0]
DEFAULT_LOG_SCALE_OPACITY_VALUES: List[float] = [0.0, 0.1, 0.3, 0.6, 0.9]
DEFAULT_SMALL_POSITIVE_VALUE: float = 1e-10
DEFAULT_SCALAR_BAR_N_LABELS: int = 5
DEFAULT_SCALAR_BAR_TITLE_FONT_SIZE: int = 12
DEFAULT_SCALAR_BAR_LABEL_FONT_SIZE: int = 10
DEFAULT_UNIT_THICKNESS: float = 1.0
DEFAULT_COORDINATE_OFFSET_VALUE: int = 0
DEFAULT_MAX_SUBPLOT_COLS: int = 3
DEFAULT_SCALAR_BAR_LABEL_FONT_SIZE: int = 10
DEFAULT_POINT_SIZE_THRESHOLD: float = 0.0
DEFAULT_MAX_INTEGRATION_TIME: float = 1000.0
DEFAULT_ANIMATION_QUALITY: int = 5
DEFAULT_ANIMATION_FRAMERATE: int = 10
DEFAULT_MAX_SUBPLOT_COLS: int = 3
DEFAULT_POINT_CLOUD_REPLACE: bool = False
DEFAULT_UNIT_THICKNESS: float = 1.0
DEFAULT_TUBE_RADIUS_AUTO: bool = False

__all__ = [
    "PlotType",
    "PlotConfig",
    "CameraPosition",
    "Lighting",
    "Label",
    "Labels",
    "BaseRenderer",
    "VolumeRenderer",
    "IsosurfaceRenderer",
    "SliceRenderer",
    "ModelVisualizer3D",
    "viz",
]


Mapper = Literal["fixed_point", "gpu", "open_gl", "smart", "ugrid"]

pv.set_jupyter_backend("trame")


class PlotType(str, Enum):
    """Types of PyVista supported 3D plots available."""

    VOLUME_RENDER = "volume_render"
    """Volume rendering for scalar fields using PyVista's volume rendering capabilities."""

    ISOSURFACE = "isosurface"
    """Isosurface plots using PyVista's contour filtering."""

    SLICE_PLANES = "slice_planes"
    """Slice planes for cross-sectional views using PyVista's slice filters."""


class RenderingMode(str, Enum):
    """PyVista rendering modes for different visual effects."""

    SURFACE = "surface"
    """Standard surface rendering."""

    WIREFRAME = "wireframe"
    """Wireframe rendering showing mesh structure."""

    POINTS = "points"
    """Point cloud rendering."""

    VOLUME = "volume"
    """Volume rendering for translucent effects."""


class CameraPosition(TypedDict):
    """Camera position configuration for PyVista plots."""

    position: Tuple[float, float, float]
    """Camera position in 3D space (x, y, z coordinates)."""

    focal_point: Tuple[float, float, float]
    """Point in 3D space that the camera is looking at (x, y, z coordinates)."""

    view_up: Tuple[float, float, float]
    """Up direction vector for the camera (x, y, z coordinates)."""

    zoom: Optional[float]
    """Camera zoom factor. Values > 1 zoom in, < 1 zoom out."""


class Lighting(TypedDict, total=False):
    """Lighting configuration for PyVista plots."""

    ambient: float
    """Ambient light intensity (0.0 to 1.0)."""

    diffuse: float
    """Diffuse light intensity (0.0 to 1.0)."""

    specular: float
    """Specular light intensity (0.0 to 1.0)."""

    specular_power: float
    """Specular power for shininess control."""

    enable_shadows: bool
    """Whether to enable shadow mapping."""

    shadow_attenuation: float
    """Shadow attenuation factor (0.0 to 1.0)."""


DEFAULT_PYVISTA_CAMERA = CameraPosition(
    position=(1.5, 1.5, 1.8),
    focal_point=(0.0, 0.0, 0.0),
    view_up=(0.0, 0.0, 1.0),
    zoom=1.0,
)

DEFAULT_PYVISTA_LIGHTING = Lighting(
    ambient=0.3,
    diffuse=0.6,
    specular=0.3,
    specular_power=20.0,
    enable_shadows=False,
    shadow_attenuation=0.5,
)


@define(frozen=True)
class PlotConfig:
    """Configuration for PyVista 3D plotting."""

    width: int = 1200
    """Plot window width in pixels."""

    height: int = 800
    """Plot window height in pixels."""

    plot_type: PlotType = PlotType.VOLUME_RENDER
    """Default plot type for PyVista visualizations."""

    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    """Default color scheme for data visualization."""

    opacity: Union[float, Sequence[float]] = 0.65
    """Default opacity level for plot elements (0.0 = transparent, 1.0 = opaque)."""

    rendering_mode: RenderingMode = RenderingMode.SURFACE
    """Rendering mode for PyVista plots."""

    show_scalar_bar: bool = True
    """Whether to display the scalar bar (colorbar) legend."""

    show_axes: bool = True
    """Whether to display 3D axes."""

    show_grid: bool = False
    """Whether to display grid lines."""

    camera_position: CameraPosition = field(factory=lambda: DEFAULT_PYVISTA_CAMERA)
    """Camera position and orientation."""

    lighting: Lighting = field(factory=lambda: DEFAULT_PYVISTA_LIGHTING)
    """Lighting configuration."""

    background_color: Union[str, Sequence[float]] = "white"
    """Background color for the plot."""

    off_screen: bool = False
    """Whether to render off-screen (for saving images without displaying)."""

    jupyter_backend: str = "trame"
    """Backend for Jupyter notebook rendering ('trame', 'pythreejs', 'static')."""

    smooth_shading: bool = True
    """Whether to use smooth shading for surfaces."""

    edge_color: Optional[str] = "black"
    """Color for mesh edges. If None, edges are not shown."""

    line_width: float = 1.0
    """Width of lines and edges."""

    show_edges: bool = True
    """Whether to show cell edges/wireframe."""

    point_size: float = 5.0
    """Size of points in scatter plots."""

    volume_mapper: Mapper = "smart"
    """Volume mapper type ('gpu', 'fixed_point', 'smart')."""

    enable_anti_aliasing: bool = False
    """Whether to enable anti-aliasing for smoother edges."""

    multi_samples: int = 4
    """Number of samples for multi-sample anti-aliasing."""

    font_family: Literal["courier", "arial", "times"] = "arial"
    """Font family for text elements."""

    font_size: int = 12
    """Base font size for text elements."""


@define(frozen=True)
class LabelCoordinate:
    """3D coordinate specification for labels."""

    x: Union[int, float]
    """X coordinate (grid index or physical coordinate)."""

    y: Union[int, float]
    """Y coordinate (grid index or physical coordinate)."""

    z: Union[int, float]
    """Z coordinate (grid index or physical coordinate)."""

    use_physical_coords: bool = False
    """Whether coordinates are physical (True) or grid indices (False)."""


@define(slots=True)
class Label:
    """
    A 3D text label for PyVista plots with enhanced positioning and styling options.

    :param text: The text content of the label. Supports format strings with property values
    :param coordinate: 3D position of the label
    :param visible: Whether the label should be displayed
    :param font_size: Font size for the label text
    :param font_color: Color of the label text
    :param font_family: Font family for the label text
    :param bold: Whether the text should be bold
    :param italic: Whether the text should be italic
    :param background_color: Background color for the label. If None, no background
    :param background_opacity: Opacity of the label background (0.0 to 1.0)
    :param shadow: Whether to add a shadow to the text
    :param shadow_offset: Shadow offset in pixels (x, y)
    :param always_visible: Whether the label should always be visible (not occluded by geometry)
    :param scale_with_zoom: Whether the label size should scale with camera zoom
    """

    text: str
    coordinate: LabelCoordinate
    visible: bool = True
    font_size: int = 12
    font_color: str = "black"
    font_family: str = "arial"
    bold: bool = False
    italic: bool = False
    background_color: Optional[Union[str, Sequence[float]]] = None
    background_opacity: float = 0.8
    shadow: bool = False
    shadow_offset: Tuple[float, float] = (1.0, 1.0)
    always_visible: bool = False
    scale_with_zoom: bool = True


class Labels:
    """
    Collection manager for PyVista 3D labels with advanced positioning and styling.
    """

    def __init__(self, labels: Optional[List[Label]] = None):
        """Initialize the labels collection."""
        self._labels: List[Label] = list(labels) if labels else []

    def add(self, label: Label) -> None:
        """Add a label to the collection."""
        self._labels.append(label)

    def add_text_label(
        self,
        text: str,
        x: Union[int, float],
        y: Union[int, float],
        z: Union[int, float],
        use_physical_coords: bool = False,
        **label_kwargs,
    ) -> None:
        """
        Add a text label at the specified coordinates.

        :param text: The text content for the label
        :param x: X coordinate position
        :param y: Y coordinate position
        :param z: Z coordinate position
        :param use_physical_coords: Whether to use physical coordinates instead of grid indices
        :param label_kwargs: Additional keyword arguments for Label
        """
        coordinate = LabelCoordinate(
            x=x, y=y, z=z, use_physical_coords=use_physical_coords
        )
        label = Label(text=text, coordinate=coordinate, **label_kwargs)
        self.add(label=label)

    def add_grid_labels(
        self,
        data_shape: Tuple[int, int, int],
        spacing: Tuple[int, int, int] = DEFAULT_GRID_SPACING,
        template: str = "({x}, {y}, {z})",
        use_physical_coords: bool = False,
        **label_kwargs,
    ) -> None:
        """
        Add grid labels at regular intervals.

        :param data_shape: Shape of the data grid as (nx, ny, nz)
        :param spacing: Spacing between labels as (sx, sy, sz)
        :param template: Template string for label text with {x}, {y}, {z} placeholders
        :param use_physical_coords: Whether to use physical coordinates instead of grid indices
        :param label_kwargs: Additional keyword arguments for Label
        """
        nx, ny, nz = data_shape
        sx, sy, sz = spacing

        for i, j, k in itertools.product(
            range(0, nx, sx), range(0, ny, sy), range(0, nz, sz)
        ):
            text = template.format(x=i, y=j, z=k)
            self.add_text_label(
                text=text,
                x=i,
                y=j,
                z=k,
                use_physical_coords=use_physical_coords,
                **label_kwargs,
            )

    def add_boundary_labels(
        self,
        data_shape: Tuple[int, int, int],
        template: str = "Boundary ({x}, {y}, {z})",
        use_physical_coords: bool = False,
        **label_kwargs,
    ) -> None:
        """Add labels at the boundaries of the data grid."""
        nx, ny, nz = data_shape

        # Corner points
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
            text = template.format(x=x, y=y, z=z)
            self.add_text_label(
                text=text,
                x=x,
                y=y,
                z=z,
                use_physical_coords=use_physical_coords,
                **label_kwargs,
            )

    def add_well_labels(
        self,
        well_positions: List[Tuple[int, int, int]],
        well_names: Optional[List[str]] = None,
        template: str = "Well {name} ({x}, {y}, {z})",
        use_physical_coords: bool = False,
        **label_kwargs,
    ) -> None:
        """Add labels for well locations."""
        if well_names is None:
            well_names = [f"W{i + 1}" for i in range(len(well_positions))]

        for i, (x, y, z) in enumerate(well_positions):
            name = well_names[i] if i < len(well_names) else f"W{i + 1}"
            text = template.format(name=name, x=x, y=y, z=z)
            self.add_text_label(
                text=text,
                x=x,
                y=y,
                z=z,
                use_physical_coords=use_physical_coords,
                **label_kwargs,
            )

    def visible(self) -> List[Label]:
        """Get all visible labels."""
        return [label for label in self._labels if label.visible]

    def clear(self) -> None:
        """Clear all labels."""
        self._labels.clear()

    def __len__(self) -> int:
        """Get the number of labels."""
        return len(self._labels)

    def __iter__(self) -> Iterator[Label]:
        """Iterate over all labels."""
        return iter(self._labels)

    def __getitem__(self, index: int) -> Label:
        """Get a label by index."""
        return self._labels[index]


class BaseRenderer(ABC):
    """Base class for PyVista 3D renderers with common functionality."""

    supports_physical_dimensions: ClassVar[bool] = False
    """Whether this renderer supports physical cell dimensions and thickness grids."""

    def __init__(self, config: PlotConfig) -> None:
        """Initialize the base renderer with configuration."""
        self.config = config
        self._plotter: Optional[pv.Plotter] = None
        self.colormap = {
            ColorScheme.VIRIDIS: "viridis",
            ColorScheme.PLASMA: "plasma",
            ColorScheme.INFERNO: "inferno",
            ColorScheme.MAGMA: "magma",
            ColorScheme.CIVIDIS: "cividis",
            ColorScheme.TURBO: "turbo",
            ColorScheme.RdYlBu: "RdYlBu_r",
            ColorScheme.RdBu: "RdBu_r",
            ColorScheme.SPECTRAL: "Spectral_r",
            ColorScheme.BALANCE: "balance",
            ColorScheme.EARTH: "gist_earth",
        }

    @abstractmethod
    def render(
        self, plotter: pv.Plotter, data: Any, metadata: PropertyMetadata, **kwargs: Any
    ) -> None:
        """Render visualization for the given data to the provided PyVista plotter."""
        raise NotImplementedError("Subclasses must implement the `render(..)` method")

    def build_plotter(
        self,
        notebook: bool = False,
        theme: Optional[Union[pv.themes.Theme, str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs: Any
    ) -> pv.Plotter:
        """
        Create and configure a PyVista plotter instance.

        :param notebook: Whether to configure for Jupyter notebook rendering
        :param theme: Optional PyVista theme or theme name to apply
        :param width: Optional width override for the plot window
        :param height: Optional height override for the plot window
        :param kwargs: Additional initialization keyword arguments for `pv.Plotter`
        :return: Configured PyVista Plotter instance
        """
        window_width = width or self.config.width
        window_height = height or self.config.height
        window_size = [window_width, window_height]
        plotter = pv.Plotter(
            window_size=window_size,
            off_screen=self.config.off_screen,
            notebook=notebook,
            theme=theme,  # type: ignore[arg-type]
            image_scale=1,
            **kwargs,
        )

        # Set background
        plotter.set_background(self.config.background_color)  # type: ignore[misc]

        # Configure camera
        if self.config.camera_position:
            plotter.camera.position = self.config.camera_position["position"]
            plotter.camera.focal_point = self.config.camera_position["focal_point"]
            plotter.camera.up = self.config.camera_position["view_up"]
            if "zoom" in self.config.camera_position:
                plotter.camera.zoom(self.config.camera_position["zoom"])

        # Configure rendering
        if self.config.enable_anti_aliasing:
            plotter.enable_anti_aliasing()

        self._plotter = plotter
        return plotter

    def get_colormap(self, color_scheme: ColorScheme) -> str:
        """Convert ColorScheme enum to PyVista/matplotlib colormap name."""
        return self.colormap.get(color_scheme, "viridis")

    def normalize_data(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        clip_outliers: bool = True,
    ) -> Tuple[ThreeDimensionalGrid, Dict[str, float]]:
        """
        Normalize and prepare data for visualization.

        :param data: 3D numpy array of shape (nx, ny, nz) with cell-centered data values
        :param metadata: `PropertyMetadata` with visualization settings
        :param clip_outliers: Whether to clip outliers based on percentiles.
            If True, clips to 1st and 99th percentiles, by default.
            To change percentiles, modify `DEFAULT_SAMPLE_PERCENTILES`.
        :return: Tuple of (processed_data, stats) where processed_data is the normalized data
        """
        processed_data = data.copy()

        # Handle log scaling
        if metadata.log_scale:
            # Replace zeros and negatives with small positive values
            processed_data = np.where(
                processed_data <= 0, DEFAULT_SMALL_POSITIVE_VALUE, processed_data
            )
            processed_data = np.log10(processed_data)

        # Apply min/max clipping
        if metadata.min_val is not None:
            processed_data = np.maximum(processed_data, metadata.min_val)
        if metadata.max_val is not None:
            processed_data = np.minimum(processed_data, metadata.max_val)

        # Calculate statistics
        stats = {
            "min": float(np.min(processed_data)),
            "max": float(np.max(processed_data)),
            "mean": float(np.mean(processed_data)),
            "std": float(np.std(processed_data)),
        }

        # Clip outliers if requested
        if clip_outliers:
            q_min, q_max = np.percentile(a=processed_data, q=DEFAULT_SAMPLE_PERCENTILES)
            processed_data = np.clip(a=processed_data, a_min=q_min, a_max=q_max)
            stats["clipped_min"] = q_min
            stats["clipped_max"] = q_max

        return cast(ThreeDimensionalGrid, processed_data), stats

    def build_grid(
        self,
        data: ThreeDimensionalGrid,
        cell_dimension: Optional[Tuple[float, float]] = None,
        thickness_grid: Optional[ThreeDimensionalGrid] = None,
        coordinate_offsets: Optional[Tuple[int, int, int]] = None,
    ) -> pv.UnstructuredGrid:
        """
        Create a PyVista unstructured grid from 3D data with individual hexahedral cells.

        :param data: 3D numpy array of shape (nx, ny, nz) with cell-centered data values
        :param cell_dimension: Tuple of (dx, dy) cell dimensions in physical units. If None, uses unit spacing.
        :param thickness_grid: 3D numpy array of shape (nx, ny, nz) with cell thickness values. If None, uses unit thickness.
        :param coordinate_offsets: Tuple of (x_offset, y_offset, z_offset) to offset the grid coordinates. Defaults to (0, 0, 0).
        :return: PyVista UnstructuredGrid with hexahedral cells and associated data
        """
        nx, ny, nz = data.shape

        if cell_dimension is not None and thickness_grid is not None:
            x_coords, y_coords, z_coords = self.create_physical_coordinates(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                coordinate_offsets=coordinate_offsets,
            )
        else:
            # Use grid indices
            offsets = coordinate_offsets or (0, 0, 0)
            dx, dy = cell_dimension or (1.0, 1.0)
            x_coords = (np.arange(nx + 1) + offsets[0]) * dx
            y_coords = (np.arange(ny + 1) + offsets[1]) * dy
            z_coords = np.arange(nz + 1) + offsets[2]

        # Create points array for unstructured grid
        points = []
        point_ids = {}

        # Create points based on whether we have variable thickness
        if (
            cell_dimension is not None
            and thickness_grid is not None
            and z_coords.ndim == 3
        ):
            # Variable thickness case - each cell can have different Z coordinates
            point_counter = 0
            for i, j, k in itertools.product(
                range(nx + 1), range(ny + 1), range(nz + 1)
            ):
                # Handle boundary extensions for z_coords
                actual_i = min(i, nx - 1)
                actual_j = min(j, ny - 1)

                x = x_coords[i]
                y = y_coords[j]
                z = z_coords[actual_i, actual_j, k]

                points.append([x, y, z])
                point_ids[(i, j, k)] = point_counter
                point_counter += 1
        else:
            # Uniform thickness case
            if isinstance(z_coords, np.ndarray) and z_coords.ndim == 1:
                z_coord_array = z_coords
            else:
                z_coord_array = np.arange(nz + 1) + (coordinate_offsets or (0, 0, 0))[2]

            point_counter = 0
            for i, j, k in itertools.product(
                range(nx + 1), range(ny + 1), range(nz + 1)
            ):
                x = x_coords[i]
                y = y_coords[j]
                z = z_coord_array[k]

                points.append([x, y, z])
                point_ids[(i, j, k)] = point_counter
                point_counter += 1

        points = np.array(points)

        # Create hexahedral cells
        cells = []
        cell_data_values = []

        for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
            # Define the 8 vertices of a hexahedron in VTK order
            # Bottom face (k), then top face (k+1)
            hex_points = [
                point_ids[(i, j, k)],  # 0: bottom-left-front
                point_ids[(i + 1, j, k)],  # 1: bottom-right-front
                point_ids[(i + 1, j + 1, k)],  # 2: bottom-right-back
                point_ids[(i, j + 1, k)],  # 3: bottom-left-back
                point_ids[(i, j, k + 1)],  # 4: top-left-front
                point_ids[(i + 1, j, k + 1)],  # 5: top-right-front
                point_ids[(i + 1, j + 1, k + 1)],  # 6: top-right-back
                point_ids[(i, j + 1, k + 1)],  # 7: top-left-back
            ]

            # Add cell (8 points for hexahedron + cell type)
            cells.extend([8] + hex_points)

            # Add cell data value
            cell_data_values.append(data[i, j, k])

        # Create unstructured grid
        cell_types = np.full(nx * ny * nz, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, cell_types, points)

        # Add data as cell data
        grid.cell_data["values"] = np.array(cell_data_values)
        return grid

    def build_grid(
        self,
        data: ThreeDimensionalGrid,
        cell_dimension: Optional[Tuple[float, float]] = None,
        thickness_grid: Optional[ThreeDimensionalGrid] = None,
        coordinate_offsets: Optional[Tuple[int, int, int]] = None,
    ) -> pv.UnstructuredGrid:
        """
        Create a PyVista unstructured grid from 3D data with hexahedral cells.

        :param data: 3D scalar array
        :param cell_dimension: Optional (dx, dy) for physical spacing
        :param thickness_grid: Optional 3D thickness array for z spacing
        :param coordinate_offsets: Optional (x0, y0, z0) offset
        :return: PyVista UnstructuredGrid
        """
        nx, ny, nz = data.shape

        # Create coordinate arrays
        if cell_dimension and thickness_grid is not None:
            x_coords, y_coords, z_coords = self.create_physical_coordinates(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                coordinate_offsets=coordinate_offsets,
            )
        else:
            offsets = coordinate_offsets or (0, 0, 0)
            dx, dy = cell_dimension or (1.0, 1.0)
            x_coords = (np.arange(nx + 1) + offsets[0]) * dx
            y_coords = (np.arange(ny + 1) + offsets[1]) * dy
            z_coords = np.arange(nz + 1) + offsets[2]

        # Build points
        points = []
        point_ids = {}
        counter = 0

        if cell_dimension and thickness_grid is not None and z_coords.ndim == 3:
            for i, j, k in itertools.product(
                range(nx + 1), range(ny + 1), range(nz + 1)
            ):
                xi, yj = min(i, nx - 1), min(j, ny - 1)
                points.append([x_coords[i], y_coords[j], z_coords[xi, yj, k]])
                point_ids[(i, j, k)] = counter
                counter += 1
        else:
            zc = z_coords if isinstance(z_coords, np.ndarray) else np.arange(nz + 1)
            for i, j, k in itertools.product(
                range(nx + 1), range(ny + 1), range(nz + 1)
            ):
                points.append([x_coords[i], y_coords[j], zc[k]])
                point_ids[(i, j, k)] = counter
                counter += 1

        points = np.array(points, dtype=np.float32)

        # VTK9-style cells array
        n_cells = nx * ny * nz
        cells = np.empty(n_cells * 9, dtype=np.int64)  # 8 points + 1 size per hex
        cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        values = []

        w = 0
        for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
            hex_points = [
                point_ids[(i, j, k)],
                point_ids[(i + 1, j, k)],
                point_ids[(i + 1, j + 1, k)],
                point_ids[(i, j + 1, k)],
                point_ids[(i, j, k + 1)],
                point_ids[(i + 1, j, k + 1)],
                point_ids[(i + 1, j + 1, k + 1)],
                point_ids[(i, j + 1, k + 1)],
            ]
            cells[w : w + 9] = [8, *hex_points]
            w += 9
            values.append(data[i, j, k])

        # Create UnstructuredGrid
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        grid.cell_data["values"] = np.array(values)
        return grid

    def create_physical_coordinates(
        self,
        cell_dimension: Tuple[float, float],
        thickness_grid: ThreeDimensionalGrid,
        coordinate_offsets: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create physical coordinate arrays from cell dimensions and thickness."""
        dx, dy = cell_dimension
        nx, ny, nz = thickness_grid.shape
        offsets = coordinate_offsets or (0, 0, 0)

        # Create X coordinates (node positions, so nx+1 points)
        x_coords = (np.arange(nx + 1) * dx) + (offsets[0] * dx)

        # Create Y coordinates (node positions, so ny+1 points)
        y_coords = (np.arange(ny + 1) * dy) + (offsets[1] * dy)

        # Create Z coordinates from thickness (variable thickness per cell)
        # z_coords[i,j,k] represents the Z coordinate at the top of layer k for cell (i,j)
        z_coords = np.zeros((nx, ny, nz + 1))
        z_coords[:, :, 0] = offsets[2]  # Top surface (or starting depth)

        # Accumulate thickness going downward
        for k in range(nz):
            z_coords[:, :, k + 1] = z_coords[:, :, k] + thickness_grid[:, :, k]

        return x_coords, y_coords, z_coords

    def apply_labels(
        self,
        plotter: pv.Plotter,
        labels: Optional[Labels] = None,
        data: Optional[ThreeDimensionalGrid] = None,
        metadata: Optional[PropertyMetadata] = None,
        cell_dimension: Optional[Tuple[float, float]] = None,
        thickness_grid: Optional[ThreeDimensionalGrid] = None,
        coordinate_offsets: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Apply labels to the PyVista plotter."""
        if not labels:
            return

        for label in labels.visible():
            try:
                # Convert coordinates if needed
                if (
                    label.coordinate.use_physical_coords
                    and cell_dimension
                    and thickness_grid is not None
                ):
                    x_coords, y_coords, z_coords = self.create_physical_coordinates(
                        cell_dimension=cell_dimension,
                        thickness_grid=thickness_grid,
                        coordinate_offsets=coordinate_offsets,
                    )
                    # Find closest grid point to physical coordinate
                    x_idx = int(np.argmin(np.abs(x_coords - label.coordinate.x)))
                    y_idx = int(np.argmin(np.abs(y_coords - label.coordinate.y)))
                    z_idx = int(getattr(label.coordinate, "z", 0))

                    # Skip labels that would be out of bounds instead of clamping
                    if (
                        x_idx >= len(x_coords)
                        or y_idx >= len(y_coords)
                        or (
                            z_coords.ndim == 3
                            and (
                                x_idx >= z_coords.shape[0]
                                or y_idx >= z_coords.shape[1]
                                or z_idx >= z_coords.shape[2]
                            )
                        )
                    ):
                        continue

                    x_pos, y_pos = x_coords[x_idx], y_coords[y_idx]
                    z_pos = (
                        z_coords[x_idx, y_idx, z_idx]
                        if z_coords.ndim == 3
                        else label.coordinate.z
                    )
                else:
                    # Use grid indices directly
                    x_idx, y_idx, z_idx = (
                        int(label.coordinate.x),
                        int(label.coordinate.y),
                        int(getattr(label.coordinate, "z", 0)),
                    )
                    offsets = coordinate_offsets or (0, 0, 0)
                    x_pos, y_pos, z_pos = (
                        x_idx + offsets[0],
                        y_idx + offsets[1],
                        z_idx + offsets[2],
                    )

                text = label.text
                if data is not None and "{value}" in text:
                    dx, dy, dz = (
                        min(x_idx, data.shape[0] - 1),
                        min(y_idx, data.shape[1] - 1),
                        min(z_idx, data.shape[2] - 1),
                    )
                    value = data[dx, dy, dz]
                    formatted_value = self.format_value(value, metadata)
                    text = text.replace("{value}", formatted_value)
                    if metadata:
                        text = text.replace("{unit}", metadata.unit)
                        text = text.replace("{property}", metadata.display_name)

                plotter.add_point_labels(
                    [[x_pos, y_pos, z_pos]],
                    [text],
                    point_size=0,
                    font_size=label.font_size,
                    text_color=label.font_color,
                    always_visible=label.always_visible,
                    shape_color=label.background_color or "gray",
                    shape_opacity=label.background_opacity
                    if label.background_color
                    else 0,
                    bold=label.bold,
                    italic=label.italic,
                    shadow=label.shadow,
                )
            except Exception:
                continue

    @staticmethod
    def format_value(
        value: Union[int, float], metadata: Optional[PropertyMetadata] = None
    ) -> str:
        """Format a numerical value for display."""
        if metadata and metadata.log_scale:
            # Show original value, not log-transformed
            original_value = 10**value if value > -10 else 0
            return f"{original_value:.3g}"

        if isinstance(value, (int, np.integer)):
            return str(value)
        elif abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.2e}"
        return f"{value:.3f}"


class VolumeRenderer(BaseRenderer):
    """PyVista volume renderer for 3D scalar field visualization."""

    supports_physical_dimensions: ClassVar[bool] = True

    def render(
        self,
        plotter: pv.Plotter,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: Optional[Tuple[float, float]] = None,
        thickness_grid: Optional[ThreeDimensionalGrid] = None,
        opacity: Optional[Union[float, Sequence[float]]] = None,
        transfer_function: Optional[Dict[str, Any]] = None,
        volume_mapper: Optional[Mapper] = None,
        shade: bool = True,
        ambient: float = 0.3,
        diffuse: float = 0.6,
        specular: float = 0.3,
        specular_power: float = 20.0,
        labels: Optional[Labels] = None,
        coordinate_offsets: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add volume rendered visualization of 3D scalar data to the provided plotter.

        :param plotter: PyVista plotter to add the visualization to
        :param data: 3D grid of scalar values
        :param metadata: Property metadata for visualization
        :param cell_dimension: Physical cell dimensions (dx, dy) in feet
        :param thickness_grid: 3D grid of cell thickness values
        :param opacity: Single opacity value or list of opacity values for transfer function
        :param transfer_function: Custom transfer function dictionary
        :param volume_mapper: Volume mapper type ('gpu', 'fixed_point', 'smart')
        :param shade: Whether to apply shading
        :param ambient: Ambient lighting coefficient
        :param diffuse: Diffuse lighting coefficient
        :param specular: Specular lighting coefficient
        :param specular_power: Specular power for shininess
        :param labels: Labels to add to the plot
        :param coordinate_offsets: Offset for coordinate system
        :param kwargs: Additional arguments
        """
        # Normalize and prepare data
        processed_data, stats = self.normalize_data(data, metadata)

        # Build grid
        grid = self.build_grid(
            data=processed_data,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            coordinate_offsets=coordinate_offsets,
        )

        # Select volume mapper
        mapper_type = volume_mapper or self.config.volume_mapper

        # Create or use provided transfer function
        if transfer_function is None:
            transfer_function = self._create_default_transfer_function(
                data=processed_data, metadata=metadata, opacity=opacity
            )

        # Handle opacity correctly (don’t override valid 0.0)
        if opacity is None:
            opacity = self.config.opacity

        # Extract mapping for PyVista
        opacity_mapping = transfer_function.get("opacity", opacity)
        # transfer_function opacity is list of (value, opacity) tuples
        # we need just the opacity values in a numpy array
        opacity_values = np.array([v for _, v in opacity_mapping], dtype=float)
        color_map = transfer_function.get(
            "color", self.get_colormap(metadata.color_scheme)
        )

        # Add volume with transfer function
        plotter.add_volume(
            grid,
            scalars="values",
            cmap=color_map,
            opacity=opacity_values,
            shade=shade,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=specular_power,
            mapper=mapper_type,
            **kwargs,
        )

        # # Add wireframe overlay if requested
        # if self.config.show_edges and self.config.edge_color:
        #     # For unstructured grids, extract surface and add wireframe
        #     surface = grid.extract_surface()
        #     if surface.n_points > 0:
        #         plotter.add_mesh(
        #             surface,  # type: ignore
        #             style="wireframe",
        #             color=self.config.edge_color,
        #             line_width=self.config.line_width,
        #             opacity=0.3,
        #             name="cell_edges",
        #         )

        # Scalar bar with safe title
        if self.config.show_scalar_bar:
            unit_str = f" ({metadata.unit})" if getattr(metadata, "unit", None) else ""
            plotter.add_scalar_bar(
                title=f"{metadata.display_name}{unit_str}",  # type: ignore
                n_labels=DEFAULT_SCALAR_BAR_N_LABELS,  # type: ignore
                italic=False,  # type: ignore
                bold=False,  # type: ignore
                title_font_size=DEFAULT_SCALAR_BAR_TITLE_FONT_SIZE,  # type: ignore
                label_font_size=DEFAULT_SCALAR_BAR_LABEL_FONT_SIZE,  # type: ignore
            )

        # Add axes
        if self.config.show_axes:
            if cell_dimension:
                xlabel, ylabel, zlabel = (
                    "X Distance (ft)",
                    "Y Distance (ft)",
                    "Z Distance (ft)",
                )
            else:
                xlabel, ylabel, zlabel = "X Index", "Y Index", "Z Index"
            plotter.add_axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)  # type: ignore

        # Apply labels
        self.apply_labels(
            plotter=plotter,
            labels=labels,
            data=processed_data,
            metadata=metadata,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            coordinate_offsets=coordinate_offsets,
        )

        # Reset camera once
        plotter.reset_camera_clipping_range()

    def _create_default_transfer_function(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        opacity: Optional[Union[float, Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """Create a default transfer function for volume rendering."""
        data_min, data_max = np.min(data), np.max(data)
        data_range = data_max - data_min

        if opacity is None:
            # Create opacity ramp based on data values
            if metadata.log_scale:
                # For log scale data, emphasize higher values
                opacity_values = DEFAULT_LOG_SCALE_OPACITY_VALUES
            else:
                # Standard opacity ramp
                opacity_values = DEFAULT_OPACITY_VALUES
        elif isinstance(opacity, (int, float)):
            # Single opacity value - create linear ramp
            opacity_values = [0.0, opacity * 0.3, opacity * 0.6, opacity * 0.8, opacity]
        else:
            # Use provided opacity list
            opacity_values = list(opacity)

        # Create value positions for opacity mapping
        n_values = len(opacity_values)
        value_positions = [
            data_min + i * data_range / (n_values - 1) for i in range(n_values)
        ]

        return {
            "opacity": list(zip(value_positions, opacity_values)),
            "color": self.get_colormap(color_scheme=metadata.color_scheme),
        }


class IsosurfaceRenderer(BaseRenderer):
    """PyVista isosurface renderer for 3D contour visualization."""

    supports_physical_dimensions: ClassVar[bool] = True

    def render(
        self,
        plotter: pv.Plotter,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: Optional[Tuple[float, float]] = None,
        thickness_grid: Optional[ThreeDimensionalGrid] = None,
        isovalues: Optional[Union[float, List[float]]] = None,
        n_contours: int = DEFAULT_N_CONTOURS,
        opacity: Optional[float] = None,
        smooth_shading: bool = True,
        labels: Optional[Labels] = None,
        coordinate_offsets: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add isosurface plots at specified data values to the provided plotter.

        :param plotter: PyVista plotter to add the visualization to
        :param data: 3D grid of scalar values
        :param metadata: Property metadata for visualization
        :param cell_dimension: Physical cell dimensions (dx, dy) in feet
        :param thickness_grid: 3D grid of cell thickness values
        :param isovalues: Specific values for isosurfaces, or None for automatic
        :param n_contours: Number of contour levels if isovalues not specified
        :param opacity: Opacity for the isosurfaces
        :param smooth_shading: Whether to apply smooth shading
        :param labels: Labels to add to the plot
        :param coordinate_offsets: Offset for coordinate system
        :param kwargs: Additional arguments
        """
        # Normalize and prepare data
        processed_data, stats = self.normalize_data(data=data, metadata=metadata)

        # Create structured grid
        grid = self.build_grid(
            data=processed_data,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            coordinate_offsets=coordinate_offsets,
        )

        # Determine isovalues
        if isovalues is None:
            data_min, data_max = np.min(processed_data), np.max(processed_data)
            isovalues = np.linspace(
                start=data_min, stop=data_max, num=n_contours
            ).tolist()
        elif isinstance(isovalues, (int, float)):
            isovalues = [float(isovalues)]
        elif isinstance(isovalues, np.ndarray):
            isovalues = isovalues.tolist()

        isovalues = cast(List[float], isovalues)
        # Create isosurfaces
        for i, isovalue in enumerate(isovalues):
            contour = grid.contour(isosurfaces=[isovalue], scalars="values")

            if contour.n_points > 0:
                plotter.add_mesh(
                    mesh=contour,  # type: ignore[arg-type]
                    scalars="values",
                    cmap=self.get_colormap(color_scheme=metadata.color_scheme),
                    opacity=opacity or self.config.opacity,
                    smooth_shading=smooth_shading,
                    name=f"isosurface_{i}",
                    show_edges=self.config.show_edges,
                    edge_color=self.config.edge_color,
                    line_width=self.config.line_width,
                    **{k: v for k, v in kwargs.items() if k not in ["notebook"]},
                )

        # Configure scalar bar
        if self.config.show_scalar_bar:
            plotter.add_scalar_bar(
                title=f"{metadata.display_name} ({metadata.unit})",  # type: ignore
                n_labels=DEFAULT_SCALAR_BAR_N_LABELS,  # type: ignore
            )
        # Add axes
        if self.config.show_axes:
            if cell_dimension:
                plotter.add_axes(
                    xlabel="X Distance (ft)",  # type: ignore
                    ylabel="Y Distance (ft)",  # type: ignore
                    zlabel="Z Distance (ft)",  # type: ignore
                )
            else:
                plotter.add_axes(
                    xlabel="X Index",  # type: ignore
                    ylabel="Y Index",  # type: ignore
                    zlabel="Z Index",  # type: ignore
                )

        self.apply_labels(
            plotter=plotter,
            labels=labels,
            data=processed_data,
            metadata=metadata,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            coordinate_offsets=coordinate_offsets,
        )

        # Ensure plot is visible by resetting camera bounds
        plotter.reset_camera_clipping_range()


class SliceRenderer(BaseRenderer):
    """PyVista slice renderer for cross-sectional views."""

    supports_physical_dimensions: ClassVar[bool] = True

    def render(
        self,
        plotter: pv.Plotter,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: Optional[Tuple[float, float]] = None,
        thickness_grid: Optional[ThreeDimensionalGrid] = None,
        x_slices: Optional[List[int]] = None,
        y_slices: Optional[List[int]] = None,
        z_slices: Optional[List[int]] = None,
        opacity: Optional[float] = None,
        labels: Optional[Labels] = None,
        coordinate_offsets: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """Add slice plane visualizations to the provided plotter."""
        # Normalize and prepare data
        processed_data, stats = self.normalize_data(data, metadata)

        # Create structured grid
        grid = self.build_grid(
            processed_data, cell_dimension, thickness_grid, coordinate_offsets
        )

        # Default slice positions
        nx, ny, nz = data.shape
        if x_slices is None:
            x_slices = [nx // 2]
        if y_slices is None:
            y_slices = [ny // 2]
        if z_slices is None:
            z_slices = [nz // 2]

        slice_count = 0

        # Create X slices (YZ planes)
        for x_idx in x_slices:
            if 0 <= x_idx < nx:
                x_coord = grid.points[x_idx * (ny + 1) * (nz + 1), 0]
                slice_mesh = grid.slice(normal=(1, 0, 0), origin=(x_coord, 0, 0))
                plotter.add_mesh(
                    mesh=slice_mesh,
                    scalars="values",
                    cmap=self.get_colormap(metadata.color_scheme),
                    opacity=1.0,  # Solid slices, not transparent
                    name=f"x_slice_{slice_count}",
                    show_edges=True,  # Always show edges for slices
                    edge_color="black",
                    line_width=self.config.line_width,
                    **{k: v for k, v in kwargs.items() if k not in ["notebook"]},
                )
                slice_count += 1

        # Create Y slices (XZ planes)
        for y_idx in y_slices:
            if 0 <= y_idx < ny:
                y_coord = grid.points[y_idx * (nz + 1), 1]
                slice_mesh = grid.slice(normal=(0, 1, 0), origin=(0, y_coord, 0))
                plotter.add_mesh(
                    mesh=slice_mesh,
                    scalars="values",
                    cmap=self.get_colormap(metadata.color_scheme),
                    opacity=1.0,  # Solid slices
                    name=f"y_slice_{slice_count}",
                    show_edges=True,
                    edge_color="black",
                    line_width=self.config.line_width,
                    **{k: v for k, v in kwargs.items() if k not in ["notebook"]},
                )
                slice_count += 1

        # Create Z slices (XY planes)
        for z_idx in z_slices:
            if 0 <= z_idx < nz:
                z_coord = grid.points[z_idx, 2]
                slice_mesh = grid.slice(normal=(0, 0, 1), origin=(0, 0, z_coord))
                plotter.add_mesh(
                    mesh=slice_mesh,
                    scalars="values",
                    cmap=self.get_colormap(metadata.color_scheme),
                    opacity=1.0,  # Solid slices
                    name=f"z_slice_{slice_count}",
                    show_edges=True,
                    edge_color="black",
                    line_width=self.config.line_width,
                    **{k: v for k, v in kwargs.items() if k not in ["notebook"]},
                )
                slice_count += 1

        # Configure scalar bar
        if self.config.show_scalar_bar:
            plotter.add_scalar_bar(
                title=f"{metadata.display_name} ({metadata.unit})",
                n_labels=5,  # type: ignore
            )

        # Add axes
        if self.config.show_axes:
            plotter.add_axes(
                xlabel="X Distance (ft)",  # type: ignore
                ylabel="Y Distance (ft)",  # type: ignore
                zlabel="Z Distance (ft)",  # type: ignore
            )

        # Apply labels
        self.apply_labels(
            plotter=plotter,
            labels=labels,
            data=processed_data,
            metadata=metadata,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            coordinate_offsets=coordinate_offsets,
        )

        # Ensure plot is visible by resetting camera bounds
        plotter.reset_camera_clipping_range()


_missing = object()


class ModelVisualizer3D:
    """
    PyVista-based 3D visualizer for reservoir model simulation results/data.

    This class provides a comprehensive suite of PyVista-based 3D plotting capabilities
    specifically designed for reservoir engineering workflows with enhanced performance
    and advanced visualization features.
    """

    plot_type_names = {
        PlotType.VOLUME_RENDER: "3D Volume",
        PlotType.ISOSURFACE: "3D Isosurface",
        PlotType.SLICE_PLANES: "3D Slices",
    }

    def __init__(
        self,
        config: Optional[PlotConfig] = None,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        """Initialize the PyVista 3D visualizer."""
        self.config = config or PlotConfig()
        self.registry = registry or property_registry
        self._renderers: Dict[PlotType, BaseRenderer] = {}
        self._init_default_renderers()

    def _init_default_renderers(self) -> None:
        """Initialize the default renderer types."""
        self._renderers[PlotType.VOLUME_RENDER] = VolumeRenderer(self.config)
        self._renderers[PlotType.ISOSURFACE] = IsosurfaceRenderer(self.config)
        self._renderers[PlotType.SLICE_PLANES] = SliceRenderer(self.config)

    def add_renderer(
        self, plot_type: PlotType, renderer_type: Type[BaseRenderer]
    ) -> None:
        """Add or replace a renderer for a specific plot type."""
        self._renderers[plot_type] = renderer_type(self.config)

    def get_renderer(self, plot_type: PlotType) -> BaseRenderer:
        """Get the renderer for a specific plot type."""
        if plot_type not in self._renderers:
            raise ValueError(f"Renderer for {plot_type} not found")
        return self._renderers[plot_type]

    def apply_slice(
        self,
        data: ThreeDimensionalGrid,
        x_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
        y_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
        z_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
    ) -> Tuple[ThreeDimensionalGrid, Tuple[slice, slice, slice]]:
        """Apply slicing to 3D data and return sliced data with slice objects."""
        dx, dy, dz = data.shape

        def normalize_slice_spec(spec: Any, dim_size: int, dim_name: str) -> slice:
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

        x_slice_obj = normalize_slice_spec(x_slice, dx, "x")
        y_slice_obj = normalize_slice_spec(y_slice, dy, "y")
        z_slice_obj = normalize_slice_spec(z_slice, dz, "z")

        sliced_data = data[x_slice_obj, y_slice_obj, z_slice_obj]
        return cast(ThreeDimensionalGrid, sliced_data), (
            x_slice_obj,
            y_slice_obj,
            z_slice_obj,
        )

    def _get_property(
        self, model: ReservoirModel[ThreeDimensions], name: str
    ) -> ThreeDimensionalGrid:
        """
        Get property data from the reservoir model by name.

        :param model: Reservoir model containing properties
        :param name: Name of the property to extract as defined by the `PropertyMetadata.name`
        :return: A three-dimensional numpy array containing the property data
        :raises AttributeError: If property is not found in reservoir model properties
        :raises TypeError: If property is not a numpy array
        """
        model_properties = asdict(model, recurse=False)
        name_parts = name.split(".")
        data = None
        if len(name_parts) == 1:
            data = model_properties.get(name_parts[0], _missing)
        else:
            # Nested property access (e.g., "permeability.x")
            data = model_properties
            for part in name_parts:
                if isinstance(data, Mapping):
                    value = data.get(part, _missing)
                else:
                    value = getattr(data, part, _missing)

                if value is not _missing:
                    data = value
                    continue
                raise AttributeError(
                    f"Property '{name}' not found in model properties."
                )

        if data is None or data is _missing:
            raise AttributeError(
                f"Property '{name}' not in model properties or Property value is invalid."
            )
        elif isinstance(data, (list, tuple)):
            data = np.array(data, dtype=np.float32)

        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise TypeError(f"Property '{name}' is not a 3 dimensional array.")
        return cast(ThreeDimensionalGrid, data)

    def get_title(
        self,
        plot_type: PlotType,
        metadata: PropertyMetadata,
        custom_title: Optional[str] = None,
    ) -> str:
        """
        Generate a title for the plot.

        :param plot_type: Type of plot being generated
        :param metadata: Property metadata for the property being visualized
        :param custom_title: Optional custom title to use instead of default
        :return: Generated or custom title string
        """
        if custom_title:
            return custom_title
        plot_name = self.plot_type_names.get(plot_type, "3D Plot")
        return f"{plot_name}: {metadata.display_name} ({metadata.unit})"

    def make_plot(
        self,
        model_state: ModelState[ThreeDimensions],
        property: str,
        plotter: Optional[pv.Plotter] = None,
        plot_type: Optional[PlotType] = None,
        title: Optional[str] = None,
        x_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
        y_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
        z_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
        labels: Optional[Labels] = None,
        notebook: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        show: bool = True,
        **kwargs: Any,
    ) -> pv.Plotter:
        """
        Plot a single model property using PyVista.

        :param model_state: The model state containing reservoir model
        :param property: The property to visualize (as defined in PropertyMetadata.name)
        :param plotter: Optional PyVista plotter to use (creates new if None)
        :param plot_type: Type of plot to create (volume, isosurface, slices)
        :param title: Optional title for the plot
        :param x_slice: Optional slice specification for X dimension
        :param y_slice: Optional slice specification for Y dimension
        :param z_slice: Optional slice specification for Z dimension
        :param labels: Optional labels to add to the plot
        :param notebook: Whether running in a notebook environment
        :param show: Whether to display the plot immediately
        :param kwargs: Additional arguments passed to the renderer
        :return: PyVista plotter with the rendered plot
        """
        metadata = self.registry.get_metadata(property)
        model = model_state.model
        data = self._get_property(model, metadata.name)
        title_suffix = ""

        # Apply slicing if requested
        coordinate_offsets = None
        slices = None
        if any(s is not None for s in [x_slice, y_slice, z_slice]):
            data, slices = self.apply_slice(data, x_slice, y_slice, z_slice)
            x_slice_obj, y_slice_obj, z_slice_obj = slices

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

            title_suffix += f" - Subset: {', '.join(slice_info)}"

            # Calculate coordinate offsets for proper positioning
            x_start = x_slice_obj.start or 0
            y_start = y_slice_obj.start or 0
            z_start = z_slice_obj.start or 0
            coordinate_offsets = (x_start, y_start, z_start)

        cell_dimension = model.cell_dimension
        thickness_grid = model.thickness_grid
        if coordinate_offsets is not None and slices is not None:
            thickness_grid = thickness_grid[slices[0], slices[1], slices[2]]

        # Select plot type
        plot_type = plot_type or self.config.plot_type
        renderer = self.get_renderer(plot_type)
        if plotter is None:
            plotter = renderer.build_plotter(
                notebook=notebook,
                width=width,
                height=height,
            )
        else:
            plotter.notebook = notebook

        # Add visualization to the plotter
        renderer.render(
            plotter=plotter,
            data=data,
            metadata=metadata,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            labels=labels,
            coordinate_offsets=coordinate_offsets,
            **kwargs,
        )

        # # Set title
        # # Update title with time information
        # # Create frame title with appropriate time units
        # if model_state.time >= 3600:  # More than 1 hour
        #     time_str = f"{model_state.time / 3600:.2f}hrs"
        # elif model_state.time >= 60:  # More than 1 minute
        #     time_str = f"{model_state.time / 60:.2f}mins"
        # else:
        #     time_str = f"{model_state.time:.2f}secs"

        # title_suffix += f" - Time: {time_str}"
        # title_suffix += f", Step: {model_state.time_step}"
        # title_suffix = title_suffix.strip().removeprefix("- ")
        # if title:
        #     plotter.add_text(
        #         f"{title} - {title_suffix}",
        #         font_size=self.config.font_size,
        #         font=self.config.font_family,
        #     )
        # else:
        #     default_title = f"{self.get_title(plot_type, metadata)} - {title_suffix}"
        #     plotter.add_text(
        #         default_title,
        #         font_size=self.config.font_size,
        #         font=self.config.font_family,
        #     )

        # Show plot if requested
        if show:
            plotter.show()
        return plotter

    def animate(
        self,
        model_states: Sequence[ModelState[ThreeDimensions]],
        property: str,
        plot_type: Optional[PlotType] = None,
        filename: Optional[str] = None,
        framerate: int = DEFAULT_ANIMATION_FRAMERATE,
        quality: int = DEFAULT_ANIMATION_QUALITY,
        notebook: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Create an animation from multiple model states.

        :param model_states: Sequence of model states to animate
        :param property: Property to visualize
        :param plot_type: Type of plot for animation
        :param filename: Output filename for animation (MP4 or GIF)
        :param framerate: Frames per second
        :param quality: Quality setting (1-10, higher is better)
        :param notebook: Whether running in notebook
        :param width: Width of the animation frames
        :param height: Height of the animation frames
        :param kwargs: Additional arguments for plotting
        :return: Path to saved animation file, or None if not saved
        """
        if not model_states:
            raise ValueError("No model states provided for animation")

        plot_type = plot_type or self.config.plot_type
        renderer = self.get_renderer(plot_type)

        # Create plotter for animation
        plotter = renderer.build_plotter(
            notebook=notebook,
            width=width,
            height=height,
        )

        if filename:
            plotter.open_movie(filename, framerate=framerate, quality=quality)

        # Add frames for each time step
        for i, model_state in enumerate(model_states):
            # Clear the plotter for new frame
            if i > 0:
                plotter.clear()

            kwargs["show"] = False  # Prevent showing each frame
            self.make_plot(
                model_state=model_state,
                property=property,
                plotter=plotter,
                plot_type=plot_type,
                notebook=notebook,
                **kwargs,
            )

            # Write frame
            if filename:
                plotter.write_frame()

        # Finalize movie
        if filename:
            plotter.close()

        plotter.show()
        return None

    def compare_properties(
        self,
        model_states: List[ModelState[ThreeDimensions]],
        properties: List[str],
        plot_type: Optional[PlotType] = None,
        titles: Optional[List[str]] = None,
        subplot_shape: Optional[Tuple[int, int]] = None,
        notebook: bool = False,
        **kwargs: Any,
    ) -> pv.Plotter:
        """
        Create a comparison plot with multiple properties or time states.

        :param model_states: List of model states (can be same state repeated for different properties)
        :param properties: List of properties to compare
        :param plot_type: Type of plot to use
        :param titles: Custom titles for each subplot
        :param subplot_shape: Shape of subplot grid (rows, cols)
        :param notebook: Whether running in notebook
        :param kwargs: Additional plotting arguments
        :return: PyVista plotter with subplots
        """
        n_plots = len(model_states) * len(properties)
        if n_plots == 0:
            raise ValueError("No plots to create")

        # Determine subplot layout
        if subplot_shape is None:
            cols = min(n_plots, DEFAULT_MAX_SUBPLOT_COLS)
            rows = (n_plots + cols - 1) // cols
            subplot_shape = (rows, cols)

        # Create subplot plotter
        plotter = pv.Plotter(
            shape=subplot_shape,
            notebook=notebook,
            window_size=[self.config.width, self.config.height],
        )

        plot_idx = 0
        plot_type = plot_type or self.config.plot_type
        renderer = self.get_renderer(plot_type)

        for state_idx, model_state in enumerate(model_states):
            for prop_idx, property in enumerate(properties):
                if plot_idx >= subplot_shape[0] * subplot_shape[1]:
                    break

                row = plot_idx // subplot_shape[1]
                col = plot_idx % subplot_shape[1]

                # Set the current subplot
                plotter.subplot(row, col)

                # Get data and metadata
                metadata = self.registry.get_metadata(property)
                data = self._get_property(model_state.model, metadata.name)

                # Use renderer to add visualization to current subplot
                renderer.render(
                    plotter=plotter,
                    data=data,
                    metadata=metadata,
                    cell_dimension=model_state.model.cell_dimension,
                    thickness_grid=model_state.model.thickness_grid,
                    **kwargs,
                )

                # Set subplot title
                if titles and plot_idx < len(titles):
                    title = titles[plot_idx]
                else:
                    title = f"{metadata.display_name}"
                    if len(model_states) > 1:
                        title += f" (State {state_idx + 1})"

                plotter.add_title(title)
                plot_idx += 1

        return plotter


viz = ModelVisualizer3D()
"""
Default 3D visualizer instance for reservoir models.
"""
