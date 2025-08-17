"""
3D Visualization Suite for Reservoir Simulation

This module provides a comprehensive visualization factory for 3D reservoir simulation data,
including interactive 3D plots, volume rendering, cross-sections, and animations.
"""

from collections.abc import Mapping
from attrs import asdict
import numpy as np
import typing
import itertools
from typing_extensions import TypedDict
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import ClassVar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation

from sim3D.simulation import ModelState
from sim3D.types import ThreeDimensions, ThreeDimensionalGrid


__all__ = [
    "PlotType",
    "ColorScheme",
    "PlotConfig",
    "CameraPosition",
    "Lighting",
    "LightPosition",
    "PropertyMetadata",
    "PropertyRegistry",
    "Base3DPlotter",
    "VolumeRenderer",
    "IsosurfacePlotter",
    "SlicePlotter",
    "CellBlockPlotter",
    "Scatter3DPlotter",
    "Label",
    "LabelCoordinate3D",
    "LabelManager",
    "ReservoirModelVisualizer3D",
    "viz",
]


class PlotType(str, Enum):
    """Types of 3D plots available."""

    VOLUME_RENDER = "volume_render"
    """Volume rendering for scalar fields, showing continuous data distribution."""
    ISOSURFACE = "isosurface"
    """Isosurface plots for discrete value thresholds, showing surfaces at specific data values."""
    SLICE_PLANES = "slice_planes"
    """Slice planes for cross-sectional views of 3D data, useful for examining internal structures."""
    SCATTER_3D = "scatter_3d"
    """3D scatter plots for visualizing point data in 3D space."""
    CELL_BLOCKS = "cell_blocks"
    """Cell block plots for visualizing individual reservoir cells as blocks in 3D space."""


class ColorScheme(str, Enum):
    """Professional color schemes for reservoir visualization."""

    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    TURBO = "turbo"
    RdYlBu = "rdylbu"
    RdBu = "rdbu"
    SPECTRAL = "spectral"
    BALANCE = "balance"
    EARTH = "earth"


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


DEFAULT_CAMERA_POSITION = CameraPosition(
    eye=dict(x=1.5, y=1.5, z=1.8),
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1),
)

DEFAULT_OPACITY_SCALE_VALUES = [
    [0, 0.5],
    [0.2, 0.65],
    [0.5, 0.75],
    [0.8, 0.95],
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
    """Configuration for 3D plotting."""

    width: int = 1200
    """Plot width in pixels. Larger values provide higher resolution but may impact performance."""

    height: int = 800
    """Plot height in pixels. Larger values provide higher resolution but may impact performance."""

    plot_type: PlotType = PlotType.VOLUME_RENDER
    """Default plot type to use when no specific type is requested. Different plot types offer 
    different visualization perspectives of the same 3D data."""

    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    """Default color scheme for data visualization. Professional color schemes are optimized
    for scientific data visualization and accessibility."""

    opacity: float = 0.65
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

    cell_outline_color: str = "gray"
    """Color for cell block outlines when show_cell_outlines is True. 
    Can be CSS color name, hex code, or rgb() string."""

    cell_outline_width: float = 0.2
    """Width/thickness of cell outline wireframes in pixels when show_cell_outlines is True.
    Thicker lines are more visible but may obscure data details."""

    use_opacity_scaling: bool = True
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


@dataclass(frozen=True)
class PropertyMetadata:
    """Metadata for model properties used in property registry."""

    name: str
    """Internal grid property name in the ModelState object (e.g., 'pressure_grid', 'oil_viscosity_grid').
    This should match the actual attribute name in the simulation data structure."""

    display_name: str
    """Human-readable name shown in plot titles, legends, and hover text (e.g., 'Pressure', 'Oil Viscosity').
    This is what users see in the visualization interface."""

    unit: str
    """Physical unit of measurement displayed with values (e.g., 'psi', 'cP', '°F', 'fraction').
    Used in hover text, colorbar labels, and axis titles for proper scientific notation."""

    color_scheme: ColorScheme
    """Default color scheme for visualizing this property. Different properties use different color schemes
    optimized for their typical value ranges and physical meaning (e.g., pressure uses 'viridis',
    temperature uses 'inferno' for heat-like appearance)."""

    log_scale: bool = False
    """Whether to apply logarithmic scaling (base 10) to the data before visualization.
    
    Use log_scale=True for properties that vary over many orders of magnitude, such as:
    - Viscosity (0.1 to 10,000+ cP)
    - Compressibility (1e-6 to 1e-3 psi⁻¹)
    - Permeability (0.001 to 10,000+ mD)
    
    When True:
    - Data values are transformed using log₁₀(value) for plotting
    - Color mapping and isosurfaces use log-transformed values
    - Hover text and colorbar show ORIGINAL physical values (not log values)
    - Zero/negative values are handled by replacing with small positive values
    
    Example: 0.5 cP viscosity becomes log₁₀(0.5) = -0.301 for plotting,
    but hover text still shows "0.5 cP" to the user."""

    min_val: typing.Optional[float] = None
    """Minimum value for data clipping and normalization. If specified along with max_val,
    all data values will be clipped to this range before visualization.
    
    Useful for:
    - Saturations (min_val=0, max_val=1) to ensure physically meaningful bounds
    - Removing outliers that might distort color scaling
    - Focusing visualization on a specific value range of interest
    
    Set to None for no minimum clipping."""

    max_val: typing.Optional[float] = None
    """Maximum value for data clipping and normalization. If specified along with min_val,
    all data values will be clipped to this range before visualization.
    
    Useful for:
    - Saturations (min_val=0, max_val=1) to ensure physically meaningful bounds
    - Removing outliers that might distort color scaling
    - Focusing visualization on a specific value range of interest
    
    Set to None for no maximum clipping."""


class PropertyRegistry:
    """Registry of all available model properties available for visualization."""

    PROPERTIES = {
        # Pressure and Temperature
        "pressure": PropertyMetadata(
            name="fluid_properties.pressure_grid",
            display_name="Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
        ),
        "oil_pressure": PropertyMetadata(
            name="fluid_properties.pressure_grid",
            display_name="Oil Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
        ),
        "temperature": PropertyMetadata(
            name="fluid_properties.temperature_grid",
            display_name="Temperature",
            unit="°F",
            color_scheme=ColorScheme.INFERNO,
        ),
        # Oil Properties
        "oil_saturation": PropertyMetadata(
            name="fluid_properties.oil_saturation_grid",
            display_name="Oil Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            min_val=0,
            max_val=1,
        ),
        "oil_viscosity": PropertyMetadata(
            name="fluid_properties.oil_viscosity_grid",
            display_name="Oil Viscosity",
            unit="cP",
            color_scheme=ColorScheme.MAGMA,
            log_scale=True,
        ),
        "oil_density": PropertyMetadata(
            name="fluid_properties.oil_density_grid",
            display_name="Oil Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.PLASMA,
        ),
        "oil_compressibility": PropertyMetadata(
            name="fluid_properties.oil_compressibility_grid",
            display_name="Oil Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.TURBO,
            log_scale=True,
        ),
        "oil_formation_volume_factor": PropertyMetadata(
            name="fluid_properties.oil_formation_volume_factor_grid",
            display_name="Oil FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.RdYlBu,
        ),
        "oil_fvf": PropertyMetadata(
            name="fluid_properties.oil_formation_volume_factor_grid",
            display_name="Oil FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.RdYlBu,
        ),
        "oil_bubble_point_pressure": PropertyMetadata(
            name="fluid_properties.oil_bubble_point_pressure_grid",
            display_name="Oil Bubble Point Pressure",
            unit="psi",
            color_scheme=ColorScheme.SPECTRAL,
        ),
        # Water Properties
        "water_saturation": PropertyMetadata(
            name="fluid_properties.water_saturation_grid",
            display_name="Water Saturation",
            unit="fraction",
            color_scheme=ColorScheme.RdBu,
            min_val=0,
            max_val=1,
        ),
        "water_viscosity": PropertyMetadata(
            name="fluid_properties.water_viscosity_grid",
            display_name="Water Viscosity",
            unit="cP",
            color_scheme=ColorScheme.BALANCE,
            log_scale=True,
        ),
        "water_density": PropertyMetadata(
            name="fluid_properties.water_density_grid",
            display_name="Water Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.EARTH,
        ),
        "water_compressibility": PropertyMetadata(
            name="fluid_properties.water_compressibility_grid",
            display_name="Water Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.VIRIDIS,
            log_scale=True,
        ),
        "water_formation_volume_factor": PropertyMetadata(
            name="fluid_properties.water_formation_volume_factor_grid",
            display_name="Water FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.PLASMA,
        ),
        "water_fvf": PropertyMetadata(
            name="fluid_properties.water_formation_volume_factor_grid",
            display_name="Water FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.PLASMA,
        ),
        "water_bubble_point_pressure": PropertyMetadata(
            name="fluid_properties.water_bubble_point_pressure_grid",
            display_name="Water Bubble Point Pressure",
            unit="psi",
            color_scheme=ColorScheme.INFERNO,
        ),
        "water_salinity": PropertyMetadata(
            name="fluid_properties.water_salinity_grid",
            display_name="Water Salinity",
            unit="ppm NaCl",
            color_scheme=ColorScheme.CIVIDIS,
        ),
        "gas_saturation": PropertyMetadata(
            name="fluid_properties.gas_saturation_grid",
            display_name="Gas Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            min_val=0,
            max_val=1,
        ),
        "gas_viscosity": PropertyMetadata(
            name="fluid_properties.gas_viscosity_grid",
            display_name="Gas Viscosity",
            unit="cP",
            color_scheme=ColorScheme.TURBO,
            log_scale=True,
        ),
        "gas_density": PropertyMetadata(
            name="fluid_properties.gas_density_grid",
            display_name="Gas Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.RdYlBu,
        ),
        "gas_compressibility": PropertyMetadata(
            name="fluid_properties.gas_compressibility_grid",
            display_name="Gas Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.SPECTRAL,
            log_scale=True,
        ),
        "gas_formation_volume_factor": PropertyMetadata(
            name="fluid_properties.gas_formation_volume_factor_grid",
            display_name="Gas FVF",
            unit="ft³/SCF",
            color_scheme=ColorScheme.RdBu,
        ),
        "gas_fvf": PropertyMetadata(
            name="fluid_properties.gas_formation_volume_factor_grid",
            display_name="Gas FVF",
            unit="ft³/SCF",
            color_scheme=ColorScheme.RdBu,
        ),
        "gas_to_oil_ratio": PropertyMetadata(
            name="fluid_properties.gas_to_oil_ratio_grid",
            display_name="Gas-Oil Ratio",
            unit="SCF/STB",
            color_scheme=ColorScheme.BALANCE,
        ),
        "gor": PropertyMetadata(
            name="fluid_properties.gas_to_oil_ratio_grid",
            display_name="Gas-Oil Ratio",
            unit="SCF/STB",
            color_scheme=ColorScheme.BALANCE,
        ),
        "gas_gravity": PropertyMetadata(
            name="fluid_properties.gas_gravity_grid",
            display_name="Gas Gravity",
            unit="dimensionless",
            color_scheme=ColorScheme.EARTH,
        ),
        "gas_molecular_weight": PropertyMetadata(
            name="fluid_properties.gas_molecular_weight_grid",
            display_name="Gas Molecular Weight",
            unit="g/mol",
            color_scheme=ColorScheme.VIRIDIS,
        ),
        # API Gravity
        "oil_api_gravity": PropertyMetadata(
            name="fluid_properties.oil_api_gravity_grid",
            display_name="Oil API Gravity",
            unit="°API",
            color_scheme=ColorScheme.PLASMA,
        ),
        "oil_specific_gravity": PropertyMetadata(
            name="fluid_properties.oil_specific_gravity_grid",
            display_name="Oil Specific Gravity",
            unit="dimensionless",
            color_scheme=ColorScheme.INFERNO,
        ),
        "thickness": PropertyMetadata(
            name="thickness_grid",
            display_name="Cell Thickness",
            unit="ft",
            color_scheme=ColorScheme.CIVIDIS,
        ),
        "permeability_x": PropertyMetadata(
            name="rock_properties.absolute_permeability.x",
            display_name="Permeability X",
            unit="mD",
            color_scheme=ColorScheme.MAGMA,
        ),
        "permeability_y": PropertyMetadata(
            name="rock_properties.absolute_permeability.y",
            display_name="Permeability Y",
            unit="mD",
            color_scheme=ColorScheme.TURBO,
        ),
        "permeability_z": PropertyMetadata(
            name="rock_properties.absolute_permeability.z",
            display_name="Permeability Z",
            unit="mD",
            color_scheme=ColorScheme.RdYlBu,
        ),
        "porosity": PropertyMetadata(
            name="rock_properties.porosity_grid",
            display_name="Porosity",
            unit="fraction",
            color_scheme=ColorScheme.PLASMA,
        ),
        "net_to_gross_ratio": PropertyMetadata(
            name="rock_properties.net_to_gross_ratio_grid",
            display_name="Net to Gross Ratio",
            unit="fraction",
            color_scheme=ColorScheme.VIRIDIS,
        ),
        "irreducible_water_saturation": PropertyMetadata(
            name="rock_properties.irreducible_water_saturation_grid",
            display_name="Irreducible Water Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
        ),
        "residual_oil_saturation": PropertyMetadata(
            name="rock_properties.residual_oil_saturation_grid",
            display_name="Residual Oil Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
        ),
        "residual_gas_saturation": PropertyMetadata(
            name="rock_properties.residual_gas_saturation_grid",
            display_name="Residual Gas Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
        ),
    }

    def __init__(self) -> None:
        """
        Initialize the property registry.

        This class is a singleton and should not be instantiated directly.
        Use the class methods to access properties and metadata.
        """
        self._properties = type(self).PROPERTIES.copy()

    def get_available_properties(self) -> typing.List[str]:
        """
        Get list of all available property names.

        :return: List of property names that can be used for visualization
        """
        return list(self._properties.keys())

    def get_metadata(self, property: str) -> PropertyMetadata:
        """
        Get metadata for a specific property.

        :param property: Name of the property to get metadata for
        :return: `PropertyMetadata` object containing display information
        :raises ValueError: If property is not found in the registry
        """
        if property not in self._properties:
            raise ValueError(
                f"Unknown property: {property}. Available: {', '.join(self.get_available_properties())}"
            )
        return self._properties[property]

    def __getitem__(self, name: str, /) -> PropertyMetadata:
        return self.get_metadata(name)

    def __setitem__(self, name: str, value: PropertyMetadata, /) -> None:
        if not isinstance(value, PropertyMetadata):
            raise TypeError("Value must be a `PropertyMetadata` instance")
        self._properties[name] = value

    def __contains__(self, name: str, /) -> bool:
        """Check if a property exists in the registry."""
        return name in self._properties

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over the property names."""
        return iter(self._properties.keys())


property_registry = PropertyRegistry()


class Base3DPlotter(ABC):
    """Base class for 3D plotters."""

    supports_physical_dimensions: typing.ClassVar[bool] = False
    """Whether this plotter supports physical cell dimensions and height grids."""

    def __init__(self, config: PlotConfig) -> None:
        self.config = config

    @abstractmethod
    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> go.Figure:
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
        original_data = data.astype(np.float64)

        # Process data for plotting
        processed_data = data.astype(np.float64)

        if metadata.log_scale:
            # Handle zero/negative values for log scale
            min_positive = np.nanmin(data[data > 0])
            processed_data = np.where(data <= 0, min_positive * 0.1, data)
            processed_data = np.log10(processed_data)

        # Apply clipping if specified
        if metadata.min_val is not None and metadata.max_val is not None:
            if metadata.log_scale:
                # Clip original data, then apply log to processed data
                original_data = np.clip(
                    original_data, metadata.min_val, metadata.max_val
                )
                processed_data = np.log10(
                    np.where(original_data <= 0, metadata.min_val * 0.1, original_data)
                )
            else:
                processed_data = np.clip(
                    processed_data, metadata.min_val, metadata.max_val
                )
                original_data = processed_data.copy()
        elif normalize_range:
            # Only normalize to 0-1 range when explicitly requested (for volume rendering)
            data_min = float(np.nanmin(processed_data))
            data_max = float(np.nanmax(processed_data))
            if data_max > data_min:
                processed_data = (processed_data - data_min) / (data_max - data_min)

        processed_data = self.invert_z_axis(processed_data)
        original_data = self.invert_z_axis(original_data)
        return typing.cast(ThreeDimensionalGrid, processed_data), typing.cast(
            ThreeDimensionalGrid, original_data
        )

    def create_physical_coordinates(
        self,
        cell_dimension: typing.Tuple[float, float],
        thickness_grid: ThreeDimensionalGrid,
        coordinate_offsets: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        return x_coords, y_coords, z_coords

    def apply_labels(
        self,
        fig: go.Figure,
        labels: typing.Optional[
            typing.Union["LabelManager", typing.Iterable["Label"]]
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

        :param fig: Plotly figure to add labels to
        :param labels: Labels to apply (LabelManager or list of Label objects)
        :param data: 3D data array for value extraction
        :param metadata: Property metadata for formatting
        :param cell_dimension: Physical cell dimensions for coordinate conversion
        :param thickness_grid: Height grid for physical coordinate conversion
        :param coordinate_offsets: Coordinate offsets for sliced data
        :param format_kwargs: Additional formatting values
        """
        if labels is None:
            return

        # Handle both LabelManager and list of Labels
        if isinstance(labels, LabelManager):
            label_collection = labels
        else:
            label_collection = LabelManager(labels)

        # Get Plotly annotations
        annotations = label_collection.to_plotly_annotations(
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
                scene = getattr(fig.layout, "scene", None)
                if scene and hasattr(scene, "annotations"):
                    existing_annotations = list(scene.annotations or [])
            except (AttributeError, TypeError):
                pass

            existing_annotations.extend(annotations)
            fig.update_layout(scene=dict(annotations=existing_annotations))

    @staticmethod
    def invert_z_axis(arr: np.ndarray) -> np.ndarray:
        """
        Invert the Z axis (last axis) of a 3D array so that data[:,:,0] becomes data[:,:,nz-1].
        This ensures numpy convention (top layer is k=0) matches plotly's rendering (bottom is k=0).
        """
        if arr.ndim == 3:
            return arr[:, :, ::-1]
        return arr


class VolumeRenderer(Base3DPlotter):
    """3D Volume renderer for scalar fields."""

    supports_physical_dimensions: ClassVar[bool] = True

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        surface_count: int = 50,
        opacity: typing.Optional[float] = None,
        isomin: typing.Optional[float] = None,
        isomax: typing.Optional[float] = None,
        use_opacity_scaling: typing.Optional[bool] = None,
        aspect_mode: typing.Optional[str] = "cube",
        labels: typing.Optional["LabelManager"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Create a volume rendering plot.

        :param data: 3D data array to render
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param surface_count: Number of isosurfaces to generate for volume rendering
        :param opacity: Opacity of the volume rendering (defaults to config value). Lower values allow better visualization of internal structures.
        :param isomin: Minimum value for isosurface clipping (optional)
        :param isomax: Maximum value for isosurface clipping (optional)
        :param use_opacity_scaling: Whether to use built-in opacity scaling for better depth perception (defaults to config)
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
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

        # For volume rendering, we need both normalized (for volume) and display (for colorbar/hover)
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=True
        )

        # Get data range for colorbar mapping using original values
        data_min = float(np.nanmin(display_data))
        data_max = float(np.nanmax(display_data))

        # Create coordinate grids - use physical coordinates if available
        if cell_dimension is not None and thickness_grid is not None:
            coordinate_offsets = kwargs.get("coordinate_offsets", None)
            x_coords, y_coords, z_coords = self.create_physical_coordinates(
                cell_dimension, thickness_grid, coordinate_offsets
            )
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

        fig = go.Figure(
            data=go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=normalized_data.flatten(),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                opacity=opacity if opacity is not None else self.config.opacity,
                surface_count=surface_count,
                colorscale=self.get_colorscale(metadata.color_scheme),
                showscale=self.config.show_colorbar,
                caps=dict(x_show=True, y_show=True, z_show=True),  # Show all 6 faces
                opacityscale=self.config.opacity_scale_values
                if use_opacity_scaling
                else None,
                # Since we use normalized data (0-1), map colorbar to actual values
                cmin=0.0,  # Normalized minimum
                cmax=1.0,  # Normalized maximum
                isomin=isomin,
                isomax=isomax,
                lighting=self.config.lighting,
                lightposition=self.config.light_position,
                colorbar=dict(
                    title=f"{metadata.display_name} ({metadata.unit})"
                    + (" - Log Scale" if metadata.log_scale else ""),
                    # Custom tick labels showing actual values but positioned at normalized values
                    tickmode="array",
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Normalized positions
                    ticktext=[
                        self.format_value(
                            data_min + val * (data_max - data_min)
                            if not metadata.log_scale
                            else 10
                            ** (
                                np.log10(data_min)
                                + val * (np.log10(data_max) - np.log10(data_min))
                            ),
                            metadata=metadata,
                        )
                        for val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    ],  # Show actual physical values (not log-transformed)
                )
                if self.config.show_colorbar
                else None,
            )
        )
        self.update_layout(
            fig,
            metadata=metadata,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
        )

        if labels is not None:
            self.apply_labels(
                fig,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=coordinate_offsets,
            )
        return fig

    def update_layout(
        self,
        fig: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Index",
        y_title: str = "Y Index",
        z_title: str = "Z Index",
        aspect_mode: typing.Optional[str] = None,
    ):
        """
        Update figure layout with dimensions and scene configuration.

        :param fig: Plotly figure to update
        :param metadata: Property metadata
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        """
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode=aspect_mode or self.config.aspect_mode or "auto",
                dragmode="orbit",  # Allow orbital rotation around all axes
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
            ),
        )


class IsosurfacePlotter(Base3DPlotter):
    """3D Isosurface plotter."""

    supports_physical_dimensions: typing.ClassVar[bool] = True

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        isomin: typing.Optional[float] = None,
        isomax: typing.Optional[float] = None,
        surface_count: int = 50,
        opacity: typing.Optional[float] = None,
        aspect_mode: typing.Optional[str] = "cube",
        labels: typing.Optional["LabelManager"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Create an isosurface plot.

        :param data: 3D data array to create isosurfaces from
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param isomin: Minimum isovalue for isosurface
        :param isomax: Maximum isovalue for isosurface
        :param surface_count: Number of isosurfaces to generate
        :param opacity: Opacity of the isosurface (defaults to config value). Lower values allow better visualization of internal structures.
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with isosurface plot
        """
        if data.ndim != 3:
            raise ValueError("Isosurface plotting requires 3D data")

        # Normalized data for isosurface calculation
        normalized_data, original_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Create coordinate grids - use physical coordinates if available
        if cell_dimension is not None and thickness_grid is not None:
            coordinate_offsets = kwargs.get("coordinate_offsets", None)
            x_coords, y_coords, z_coords = self.create_physical_coordinates(
                cell_dimension, thickness_grid, coordinate_offsets
            )
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

        display_values = original_data.flatten()
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
            absolute_k = z_index_offset + k_3d

            hover_text.append(
                f"Cell: ({absolute_i}, {absolute_j}, {absolute_k})<br>"  # Show absolute indices
                f"X: {x_flat[i]:.2f}<br>"
                f"Y: {y_flat[i]:.2f}<br>"
                f"Z: {z_flat[i]:.2f}<br>"
                f"{metadata.display_name}: {self.format_value(display_values[i], metadata)} {metadata.unit}"
                + (" (log scale)" if metadata.log_scale else "")
            )

        fig = go.Figure(
            data=go.Isosurface(
                x=x_flat,
                y=y_flat,
                z=z_flat,
                value=normalized_data.flatten(),
                isomin=isomin,
                isomax=isomax,
                opacity=opacity if opacity is not None else self.config.opacity,
                surface_count=surface_count,
                colorscale=self.get_colorscale(metadata.color_scheme),
                showscale=self.config.show_colorbar,
                lighting=self.config.lighting,
                lightposition=self.config.light_position,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                colorbar=dict(
                    title=f"{metadata.display_name} ({metadata.unit})"
                    + (" - Log Scale" if metadata.log_scale else ""),
                    tickmode="array",
                    tickvals=[
                        np.nanmin(normalized_data)
                        + frac
                        * (np.nanmax(normalized_data) - np.nanmin(normalized_data))
                        for frac in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    ],
                    ticktext=[
                        self.format_value(
                            np.nanmin(original_data)
                            + frac
                            * (np.nanmax(original_data) - np.nanmin(original_data))
                            if not metadata.log_scale
                            else 10
                            ** (
                                np.log10(np.nanmin(original_data))
                                + frac
                                * (
                                    np.log10(np.nanmax(original_data))
                                    - np.log10(np.nanmin(original_data))
                                )
                            ),
                            metadata,
                        )
                        for frac in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    ],
                )
                if self.config.show_colorbar
                else None,
            )
        )

        self.update_layout(
            fig,
            metadata=metadata,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
        )

        if labels is not None:
            self.apply_labels(
                fig,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=coordinate_offsets,
            )
        return fig

    def update_layout(
        self,
        fig: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Index",
        y_title: str = "Y Index",
        z_title: str = "Z Index",
        aspect_mode: typing.Optional[str] = None,
    ) -> None:
        """
        Update figure layout with dimensions and scene configuration for isosurface plots.

        :param fig: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        """
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode=aspect_mode or self.config.aspect_mode or "cube",
                dragmode="orbit",  # Allow orbital rotation around all axes
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
            ),
        )


class SlicePlotter(Base3DPlotter):
    """3D Slice plane plotter."""

    supports_physical_dimensions: typing.ClassVar[bool] = True

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        slice_plane_x: typing.Optional[int] = None,
        slice_plane_y: typing.Optional[int] = None,
        slice_plane_z: typing.Optional[int] = None,
        show_x_slice: bool = True,
        show_y_slice: bool = True,
        show_z_slice: bool = True,
        opacity: typing.Optional[float] = None,
        aspect_mode: typing.Optional[str] = "cube",
        labels: typing.Optional["LabelManager"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Create slice plane plots showing cross-sections through 3D data.

        .. warning::
            SlicePlotter should not be used with pre-sliced data from ReservoirModelVisualizer3D's
            slicing parameters (x_slice, y_slice, z_slice) from plot_property(). The combination creates confusing
            results where slice positions don't correspond to meaningful locations in the
            original dataset. Use other plot types (Volume, Isosurface, CellBlocks, Scatter3D)
            for visualizing sliced data subsets.

        :param data: 3D data array to slice through (should be full dataset, not pre-sliced)
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param slice_plane_x: X-coordinate for YZ plane slice (defaults to middle)
        :param slice_plane_y: Y-coordinate for XZ plane slice (defaults to middle)
        :param slice_plane_z: Z-coordinate for XY plane slice (defaults to middle)
        :param show_x_slice: Whether to show the YZ plane slice
        :param show_y_slice: Whether to show the XZ plane slice
        :param show_z_slice: Whether to show the XY plane slice
        :param opacity: Opacity of the slice planes (defaults to config value)
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with slice plane plots
        """
        if data.ndim != 3:
            raise ValueError("Slice plotting requires 3D data")

        # Warning for data slicing usage
        index_offsets = kwargs.get("index_offsets", None)
        if index_offsets is not None:
            import warnings

            warnings.warn(
                "SlicePlotter is being used with pre-sliced data. This creates confusing "
                "results where slice positions don't correspond to meaningful locations "
                "in the original dataset. Consider using other plot types (Volume, "
                "Isosurface, CellBlocks, Scatter3D) for visualizing sliced data subsets.",
                UserWarning,
                stacklevel=3,
            )

        # Normalized data for surface color mapping, display(original) data for hover text and colorbar
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )

        # Default slice positions
        if slice_plane_x is None:
            slice_plane_x = data.shape[0] // 2
        if slice_plane_y is None:
            slice_plane_y = data.shape[1] // 2
        if slice_plane_z is None:
            slice_plane_z = data.shape[2] // 2

        if slice_plane_x < 0 or slice_plane_x >= data.shape[0]:
            raise ValueError(f"slice_plane_x must be between 0 and {data.shape[0] - 1}")
        if slice_plane_y < 0 or slice_plane_y >= data.shape[1]:
            raise ValueError(f"slice_plane_y must be between 0 and {data.shape[1] - 1}")
        if slice_plane_z < 0 or slice_plane_z >= data.shape[2]:
            raise ValueError(f"slice_plane_z must be between 0 and {data.shape[2] - 1}")

        fig = go.Figure()

        # Extract coordinate offsets from kwargs (used by both physical and index coordinates)
        coordinate_offsets = kwargs.get("coordinate_offsets", None)

        # Extract index offsets to show original dataset indices in hover text
        x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (0, 0, 0)

        # Extract individual coordinate offsets for use throughout the method
        x_offset, y_offset, z_offset = coordinate_offsets or (0, 0, 0)

        # Create coordinate grids - use physical coordinates if available
        if cell_dimension is not None and thickness_grid is not None:
            x_coords, y_coords, z_coords = self.create_physical_coordinates(
                cell_dimension, thickness_grid, coordinate_offsets
            )
            x_title = "X Distance (ft)"
            y_title = "Y Distance (ft)"
            z_title = "Z Distance (ft)"
        else:
            # Fallback to index-based coordinates
            # Apply numpy convention: k=0 at top, k increases downward
            x_coords = x_offset + np.arange(data.shape[0] + 1)
            y_coords = y_offset + np.arange(data.shape[1] + 1)
            z_coords = z_offset + np.arange(
                data.shape[2] + 1
            )  # Normal order for direct mapping
            x_title = "X Index"
            y_title = "Y Index"
            z_title = "Z Index"

        # X slice (YZ plane)
        if show_x_slice:
            if cell_dimension is not None and thickness_grid is not None:
                # Use physical coordinates
                x_pos = x_coords[slice_plane_x]
                y_grid, z_grid = np.meshgrid(
                    y_coords[:-1], np.arange(data.shape[2]), indexing="ij"
                )
                z_physical = np.zeros_like(z_grid)
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        # Map k=0 (data index) to highest Z coordinate (visual top)
                        z_index = (
                            data.shape[2] - 1 - k
                        )  # Reverse k for top-down display
                        z_physical[j, k] = (
                            z_coords[slice_plane_x, j, z_index]
                            + z_coords[slice_plane_x, j, z_index + 1]
                        ) / 2

                # Create hover text for X slice
                hover_text = []
                for j in range(data.shape[1]):
                    hover_row = []
                    for k in range(data.shape[2]):
                        # k here corresponds to the display order (0=top, increasing downward)
                        original_k = (
                            data.shape[2] - 1 - k
                        )  # Map back to original data index
                        # Apply index offsets to show original dataset positions
                        absolute_x = x_index_offset + slice_plane_x
                        absolute_j = y_index_offset + j
                        absolute_k = z_index_offset + original_k
                        hover_row.append(
                            f"Cell: ({absolute_x}, {absolute_j}, {absolute_k})<br>"  # Show absolute indices in original dataset
                            f"X: {x_pos:.1f} ft<br>"
                            f"Y: {y_grid[j, k]:.1f} ft<br>"
                            f"Z: {z_physical[j, k]:.1f} ft<br>"
                            f"{metadata.display_name}: {self.format_value(display_data[slice_plane_x, j, original_k], metadata)} {metadata.unit}"
                        )
                    hover_text.append(hover_row)

                fig.add_trace(
                    go.Surface(
                        x=np.full_like(y_grid, x_pos),
                        y=y_grid,
                        z=z_physical,
                        surfacecolor=normalized_data[
                            slice_plane_x, :, ::-1
                        ],  # Reverse Z dimension to match coordinates
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=self.get_colorscale(metadata.color_scheme),
                        showscale=False,  # Don't show colorbar for this slice
                        opacity=opacity if opacity is not None else self.config.opacity,
                        cmin=0,  # Ensure consistent color mapping
                        cmax=1,  # Ensure consistent color mapping
                        name=f"X={x_pos:.1f} ft",
                    )
                )
            else:
                # Fallback to index coordinates
                # Apply numpy convention: k=0 at top, k increases downward

                # Create hover text for X slice fallback
                hover_text = []
                for j in range(data.shape[1]):
                    hover_row = []
                    for k in range(data.shape[2]):
                        # k corresponds to display index (0=top), but we want original data index in hover
                        z_display = z_offset + (
                            data.shape[2] - 1 - k
                        )  # Apply z_offset and reverse for downward scaling
                        # Apply index offsets to show original dataset positions
                        absolute_x = x_index_offset + slice_plane_x
                        absolute_j = y_index_offset + j
                        absolute_k = z_index_offset + k
                        hover_row.append(
                            f"Cell: ({absolute_x}, {absolute_j}, {absolute_k})<br>"  # Show absolute indices in original dataset
                            f"X: {x_offset + slice_plane_x}<br>"
                            f"Y: {y_offset + j}<br>"
                            f"Z: {z_display}<br>"
                            f"{metadata.display_name}: {self.format_value(display_data[slice_plane_x, j, k], metadata)} {metadata.unit}"
                        )
                    hover_text.append(hover_row)

                fig.add_trace(
                    go.Surface(
                        x=np.full(
                            (data.shape[1], data.shape[2]), x_offset + slice_plane_x
                        ),
                        y=(y_offset + np.arange(data.shape[1]))[:, np.newaxis],
                        z=(z_offset + np.arange(data.shape[2])[::-1])[
                            np.newaxis, :
                        ],  # Apply z_offset and reverse Z
                        surfacecolor=normalized_data[slice_plane_x, :, :],
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=self.get_colorscale(metadata.color_scheme),
                        showscale=False,  # Don't show colorbar for this slice
                        opacity=opacity if opacity is not None else self.config.opacity,
                        cmin=0,  # Ensure consistent color mapping
                        cmax=1,  # Ensure consistent color mapping
                        name=f"X={slice_plane_x}",
                    )
                )

        # Y slice (XZ plane)
        if show_y_slice:
            if cell_dimension is not None and thickness_grid is not None:
                # Use physical coordinates
                y_pos = y_coords[slice_plane_y]
                x_grid, z_grid = np.meshgrid(
                    x_coords[:-1], np.arange(data.shape[2]), indexing="ij"
                )
                z_physical = np.zeros_like(z_grid)
                for i, k in itertools.product(
                    range(data.shape[0]), range(data.shape[2])
                ):
                    # Map k=0 (data index) to highest Z coordinate (visual top)
                    z_index = k  # Use direct mapping since we want k=0 at top
                    z_physical[i, k] = (
                        z_coords[i, slice_plane_y, z_index]
                        + z_coords[i, slice_plane_y, z_index + 1]
                    ) / 2

                # Create hover text for Y slice - accounting for coordinate reversal
                hover_text = []
                for i in range(data.shape[0]):
                    hover_row = []
                    for k in range(data.shape[2]):
                        # Since we'll reverse the array, pre-calculate the indices that will be shown
                        # after reversal: k=0 will end up at position (data.shape[2]-1), etc.
                        final_k_index = (
                            data.shape[2] - 1 - k
                        )  # What k index will be after reversal
                        # Apply index offsets to show original dataset positions
                        absolute_i = x_index_offset + i
                        absolute_y = y_index_offset + slice_plane_y
                        absolute_k = z_index_offset + final_k_index
                        hover_row.append(
                            f"Cell: ({absolute_i}, {absolute_y}, {absolute_k})<br>"  # Show absolute indices in original dataset
                            f"X: {x_grid[i, k]:.1f} ft<br>"
                            f"Y: {y_pos:.1f} ft<br>"
                            f"Z: {z_physical[i, k]:.1f} ft<br>"
                            f"{metadata.display_name}: {self.format_value(display_data[i, slice_plane_y, final_k_index], metadata)} {metadata.unit}"
                        )
                    hover_text.append(hover_row)

                # Now reverse Z coordinates and surface data for proper scaling (50-300 downwards)
                z_physical = z_physical[:, ::-1]
                surface_data = normalized_data[
                    :, slice_plane_y, ::-1
                ]  # Also reverse the surface data
                hover_text = [
                    row[::-1] for row in hover_text
                ]  # Also reverse hover text to match

                fig.add_trace(
                    go.Surface(
                        x=x_grid,
                        y=np.full_like(x_grid, y_pos),
                        z=z_physical,
                        surfacecolor=surface_data,  # Use the reversed surface data
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=self.get_colorscale(metadata.color_scheme),
                        showscale=False,  # Don't show colorbar for this slice
                        opacity=opacity if opacity is not None else self.config.opacity,
                        cmin=0,  # Ensure consistent color mapping
                        cmax=1,  # Ensure consistent color mapping
                        name=f"Y={y_pos:.1f} ft",
                    )
                )
            else:
                # Fallback to index coordinates
                # Apply numpy convention: k=0 at top, k increases downward

                # Create hover text for Y slice fallback
                hover_text = []
                for i in range(data.shape[0]):
                    hover_row = []
                    for k in range(data.shape[2]):
                        z_display = z_offset + (
                            data.shape[2] - 1 - k
                        )  # Apply z_offset and reverse for downward scaling
                        # Apply index offsets to show original dataset positions
                        absolute_i = x_index_offset + i
                        absolute_y = y_index_offset + slice_plane_y
                        absolute_k = z_index_offset + k
                        hover_row.append(
                            f"Cell: ({absolute_i}, {absolute_y}, {absolute_k})<br>"  # Show absolute indices in original dataset
                            f"X: {x_offset + i}<br>"
                            f"Y: {y_offset + slice_plane_y}<br>"
                            f"Z: {z_display}<br>"
                            f"{metadata.display_name}: {self.format_value(display_data[i, slice_plane_y, k], metadata)} {metadata.unit}"
                        )
                    hover_text.append(hover_row)

                fig.add_trace(
                    go.Surface(
                        x=(x_offset + np.arange(data.shape[0]))[:, np.newaxis],
                        y=np.full(
                            (data.shape[0], data.shape[2]), y_offset + slice_plane_y
                        ),
                        z=(z_offset + np.arange(data.shape[2])[::-1])[
                            np.newaxis, :
                        ],  # Apply z_offset and reverse Z
                        surfacecolor=normalized_data[:, slice_plane_y, :],
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=self.get_colorscale(metadata.color_scheme),
                        showscale=False,  # Don't show colorbar for this slice
                        opacity=opacity if opacity is not None else self.config.opacity,
                        cmin=0,  # Ensure consistent color mapping
                        cmax=1,  # Ensure consistent color mapping
                        name=f"Y={slice_plane_y}",
                    )
                )

        # Z slice (XY plane)
        if show_z_slice:
            if cell_dimension is not None and thickness_grid is not None:
                # Use physical coordinates - need to get the z position for this slice
                x_grid, y_grid = np.meshgrid(
                    x_coords[:-1], y_coords[:-1], indexing="ij"
                )
                z_physical = np.zeros_like(x_grid)
                for i, j in itertools.product(
                    range(data.shape[0]), range(data.shape[1])
                ):
                    # Map slice_plane_z (data index) to correct Z coordinate
                    # For proper scaling, we need to map to the reversed coordinate system
                    z_index = (
                        data.shape[2] - 1 - slice_plane_z
                    )  # Reverse mapping for 50-300 downward scaling
                    z_physical[i, j] = (
                        z_coords[i, j, z_index] + z_coords[i, j, z_index + 1]
                    ) / 2

                # Create custom hover text for Z slice to show correct indices and proper Z values
                hover_text = []
                for i in range(data.shape[0]):
                    hover_row = []
                    for j in range(data.shape[1]):
                        # Apply index offsets to show original dataset positions
                        absolute_i = x_index_offset + i
                        absolute_j = y_index_offset + j
                        absolute_z = z_index_offset + slice_plane_z
                        hover_row.append(
                            f"Cell: ({absolute_i}, {absolute_j}, {absolute_z})<br>"  # Show absolute indices in original dataset
                            f"X: {x_grid[i, j]:.1f} ft<br>"
                            f"Y: {y_grid[i, j]:.1f} ft<br>"
                            f"Z: {z_physical[i, j]:.1f} ft<br>"
                            f"{metadata.display_name}: {self.format_value(display_data[i, j, slice_plane_z], metadata)} {metadata.unit}"
                        )
                    hover_text.append(hover_row)

                fig.add_trace(
                    go.Surface(
                        x=x_grid,
                        y=y_grid,
                        z=z_physical,
                        surfacecolor=normalized_data[:, :, slice_plane_z],
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=self.get_colorscale(metadata.color_scheme),
                        showscale=self.config.show_colorbar,
                        opacity=opacity if opacity is not None else self.config.opacity,
                        cmin=0,  # Ensure consistent color mapping with other slices
                        cmax=1,  # Ensure consistent color mapping with other slices
                        colorbar=dict(
                            title=f"{metadata.display_name} ({metadata.unit})"
                            + (" - Log Scale" if metadata.log_scale else ""),
                            # Custom tick labels showing actual values but positioned at normalized values
                            tickmode="array",
                            tickvals=[
                                0,
                                0.2,
                                0.4,
                                0.6,
                                0.8,
                                1.0,
                            ],  # Normalized positions
                            ticktext=[
                                self.format_value(
                                    np.nanmin(display_data)
                                    + val
                                    * (
                                        np.nanmax(display_data)
                                        - np.nanmin(display_data)
                                    )
                                    if not metadata.log_scale
                                    else 10
                                    ** (
                                        np.log10(np.nanmin(display_data))
                                        + val
                                        * (
                                            np.log10(np.nanmax(display_data))
                                            - np.log10(np.nanmin(display_data))
                                        )
                                    ),
                                    metadata,
                                )
                                for val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                            ],  # Show actual physical values (not log-transformed)
                        )
                        if self.config.show_colorbar
                        else None,
                        name=f"Z={z_physical.mean():.1f} ft",
                    )
                )
            else:
                # Fallback to index coordinates
                # Apply numpy convention: k=0 at top, k increases downward with proper scaling
                z_display = z_offset + (
                    data.shape[2] - 1 - slice_plane_z
                )  # Apply z_offset and reverse for proper downward scaling

                # Create custom hover text for Z slice to show correct indices
                hover_text = []
                for i in range(data.shape[0]):
                    hover_row = []
                    for j in range(data.shape[1]):
                        # Apply index offsets to show original dataset positions
                        absolute_i = x_index_offset + i
                        absolute_j = y_index_offset + j
                        absolute_z = z_index_offset + slice_plane_z
                        hover_row.append(
                            f"Cell: ({absolute_i}, {absolute_j}, {absolute_z})<br>"  # Show absolute indices in original dataset
                            f"X: {x_offset + i}<br>"
                            f"Y: {y_offset + j}<br>"
                            f"Z: {z_display}<br>"
                            f"{metadata.display_name}: {self.format_value(display_data[i, j, slice_plane_z], metadata)} {metadata.unit}"
                        )
                    hover_text.append(hover_row)

                fig.add_trace(
                    go.Surface(
                        x=(x_offset + np.arange(data.shape[0]))[:, np.newaxis],
                        y=(y_offset + np.arange(data.shape[1]))[np.newaxis, :],
                        z=np.full((data.shape[0], data.shape[1]), z_display),
                        surfacecolor=normalized_data[:, :, slice_plane_z],
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=self.get_colorscale(metadata.color_scheme),
                        showscale=self.config.show_colorbar,
                        opacity=opacity if opacity is not None else self.config.opacity,
                        cmin=0,  # Ensure consistent color mapping with other slices
                        cmax=1,  # Ensure consistent color mapping with other slices
                        colorbar=dict(
                            title=f"{metadata.display_name} ({metadata.unit})"
                            + (" - Log Scale" if metadata.log_scale else ""),
                            # Custom tick labels showing actual values but positioned at normalized values
                            tickmode="array",
                            tickvals=[
                                0,
                                0.2,
                                0.4,
                                0.6,
                                0.8,
                                1.0,
                            ],  # Normalized positions
                            ticktext=[
                                self.format_value(
                                    np.nanmin(display_data)
                                    + val
                                    * (
                                        np.nanmax(display_data)
                                        - np.nanmin(display_data)
                                    )
                                    if not metadata.log_scale
                                    else 10
                                    ** (
                                        np.log10(np.nanmin(display_data))
                                        + val
                                        * (
                                            np.log10(np.nanmax(display_data))
                                            - np.log10(np.nanmin(display_data))
                                        )
                                    ),
                                    metadata,
                                )
                                for val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                            ],  # Show actual physical values (not log-transformed)
                        )
                        if self.config.show_colorbar
                        else None,
                        name=f"Z={z_display}",
                    )
                )
        self.update_layout(
            fig,
            metadata=metadata,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
            aspect_mode=aspect_mode,
        )

        if labels is not None:
            self.apply_labels(
                fig,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=coordinate_offsets,
            )
        return fig

    def update_layout(
        self,
        fig: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Index",
        y_title: str = "Y Index",
        z_title: str = "Z Index",
        aspect_mode: typing.Optional[str] = None,
    ) -> None:
        """
        Update figure layout with dimensions and scene configuration for slice plots.

        :param fig: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param aspect_mode: Aspect mode for the 3D plot (default is "cube"). Could be any of "cube", "auto", or "data".
        """
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode=aspect_mode or self.config.aspect_mode or "auto",
                dragmode="orbit",  # Allow orbital rotation around all axes
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
            ),
        )


class CellBlockPlotter(Base3DPlotter):
    """3D Cell block plotter that renders each cell as an individual block."""

    supports_physical_dimensions: typing.ClassVar[bool] = True

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Tuple[float, float],
        thickness_grid: ThreeDimensionalGrid,
        subsample_factor: int = 1,
        opacity: typing.Optional[float] = None,
        show_outline: typing.Optional[bool] = None,
        outline_color: typing.Optional[str] = None,
        outline_width: typing.Optional[float] = None,
        use_opacity_scaling: bool = True,
        labels: typing.Optional["LabelManager"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Create a plot where each cell is rendered as an individual 3D block.

        :param data: 3D data array to render
        :param metadata: Property metadata for labeling and scaling
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param subsample_factor: Factor to reduce the number of cells rendered for performance.
            A value of 1 renders every cell, 2 renders every 2nd cell in each dimension
            (reducing total cells by ~87.5%), 3 renders every 3rd cell (~96% reduction), etc.
            For example, with a 100x100x50 grid: factor=1 shows 500K cells, factor=2 shows
            ~62.5K cells, factor=5 shows ~8K cells. Use higher values for large datasets.
        :param opacity: Base opacity of the cell blocks (defaults to config value).
        :param show_outline: Whether to show wireframe outlines around each cell block (defaults to config)
        :param outline_color: Color of the cell block outlines (CSS color name, hex, or rgb, defaults to config)
        :param outline_width: Width/thickness of the outline wireframes in pixels (defaults to config)
        :param use_opacity_scaling: Whether to apply data-based opacity scaling for better depth perception
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with cell block visualization
        """
        if data.ndim != 3:
            raise ValueError("Cell block plotting requires 3D data")
        if subsample_factor < 1:
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

        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
        )
        nx, ny, nz = data.shape
        dx, dy = cell_dimension

        # Calculate opacity scaling based on data values if enabled
        base_opacity = opacity if opacity is not None else self.config.opacity
        if use_opacity_scaling:
            # Normalize data for opacity calculation (0-1 range)
            data_min = float(np.nanmin(display_data))
            data_min = float(np.nanmin(display_data))
            data_max = float(np.nanmax(display_data))
            if data_max > data_min:
                opacity_values = (display_data - data_min) / (data_max - data_min)

                def interpolate_opacity(
                    x: np.ndarray, scale_values: typing.Sequence[typing.Sequence[float]]
                ):
                    # scale_values: list of [fraction, opacity]
                    fractions, opacities = zip(*scale_values)
                    return np.interp(x, fractions, opacities)

                opacity_values = interpolate_opacity(
                    opacity_values, self.config.opacity_scale_values
                )
            else:
                opacity_values = np.full_like(display_data, base_opacity)
        else:
            opacity_values = np.full_like(display_data, base_opacity)

        fig = go.Figure()

        # Create physical coordinate grids
        coordinate_offsets = kwargs.get("coordinate_offsets", None)
        x_coords, y_coords, z_coords = self.create_physical_coordinates(
            cell_dimension, thickness_grid, coordinate_offsets
        )

        # Extract index offsets to show original dataset indices in hover text
        x_index_offset, y_index_offset, z_index_offset = coordinate_offsets or (0, 0, 0)

        # Subsample for performance
        i_indices = range(0, nx, subsample_factor)
        j_indices = range(0, ny, subsample_factor)
        k_indices = range(0, nz, subsample_factor)

        # Get global min/max for consistent color mapping across all cells
        global_min = float(np.nanmin(normalized_data))
        global_max = float(np.nanmax(normalized_data))

        for i, j, k in itertools.product(i_indices, j_indices, k_indices):
            # Get cell boundaries
            x_min, x_max = x_coords[i], x_coords[i + 1]
            y_min, y_max = y_coords[j], y_coords[j + 1]
            z_min, z_max = z_coords[i, j, k], z_coords[i, j, k + 1]

            # Create vertices for the cell block (cuboid)
            # In our coordinate system: k=0 starts at visual top, Z increases downward
            # So z_min is the top face (smaller Z value), z_max is the bottom face (larger Z value)
            vertices = [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],  # top face (smaller Z value)
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],  # bottom face (larger Z value)
            ]

            # Define faces of the cuboid (each face defined by 4 vertices)
            faces = [
                [0, 1, 2, 3],  # top face (smaller Z values)
                [4, 7, 6, 5],  # bottom face (larger Z values)
                [0, 4, 5, 1],  # front
                [2, 6, 7, 3],  # back
                [0, 3, 7, 4],  # left
                [1, 5, 6, 2],  # right
            ]

            # Convert to mesh3d format
            vertices = np.array(vertices)
            x_verts = vertices[:, 0]
            y_verts = vertices[:, 1]
            z_verts = vertices[:, 2]

            # Create triangular faces for mesh3d
            i_faces, j_faces, k_faces = [], [], []
            for face in faces:
                # Split quadrilateral into two triangles
                i_faces.extend([face[0], face[0]])
                j_faces.extend([face[1], face[2]])
                k_faces.extend([face[2], face[3]])

            normalized_cell_value = normalized_data[i, j, k]  # For color mapping
            cell_value = display_data[i, j, k]  # For hover text
            cell_opacity = opacity_values[i, j, k]

            # Create the main cell block
            fig.add_trace(
                go.Mesh3d(
                    x=x_verts,
                    y=y_verts,
                    z=z_verts,
                    i=i_faces,
                    j=j_faces,
                    k=k_faces,
                    intensity=np.full(len(x_verts), normalized_cell_value),
                    colorscale=self.get_colorscale(metadata.color_scheme),
                    cmin=global_min,  # Set global minimum for color scale
                    cmax=global_max,  # Set global maximum for color scale
                    opacity=cell_opacity,
                    showscale=False,  # Only show colorbar on the last trace
                    lighting=self.config.lighting,
                    lightposition=self.config.light_position,
                    hovertemplate=(
                        f"Cell ({x_index_offset + i}, {y_index_offset + j}, {z_index_offset + k})<br>"  # Show absolute indices
                        f"X: {(x_min + x_max) / 2:.2f} ft<br>"
                        f"Y: {(y_min + y_max) / 2:.2f} ft<br>"
                        f"Z: {(z_min + z_max) / 2:.2f} ft<br>"
                        f"{metadata.display_name}: {self.format_value(cell_value, metadata)} {metadata.unit}"
                        + (" (log scale)" if metadata.log_scale else "")
                        + "<br>"
                        f"Cell Size: {dx:.1f} x {dy:.1f} x {thickness_grid[i, j, k]:.2f} ft<br>"
                        f"Opacity: {cell_opacity:.2f}"
                        "<extra></extra>"
                    ),
                    flatshading=True,  # Use flat shading for better visibility
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
                    fig.add_trace(
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
            # Use the same global min/max that the cell blocks use for consistency
            global_min = float(np.nanmin(normalized_data))
            global_max = float(np.nanmax(normalized_data))
            # But get original data range for colorbar labels
            data_min = float(np.nanmin(display_data))
            data_max = float(np.nanmax(display_data))

            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(
                        size=0.1,
                        color=[global_min, global_max],  # Use normalized range
                        colorscale=self.get_colorscale(metadata.color_scheme),
                        cmin=global_min,  # Match cell blocks
                        cmax=global_max,  # Match cell blocks
                        colorbar=dict(
                            title=f"{metadata.display_name} ({metadata.unit})"
                            + (" - Log Scale" if metadata.log_scale else ""),
                            # Map tick positions to actual normalized range used by cell blocks
                            tickmode="array",
                            tickvals=[
                                global_min + val * (global_max - global_min)
                                for val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                            ],  # Positions within the normalized range
                            ticktext=[
                                self.format_value(
                                    data_min + val * (data_max - data_min)
                                    if not metadata.log_scale
                                    else 10
                                    ** (
                                        np.log10(data_min)
                                        + val
                                        * (np.log10(data_max) - np.log10(data_min))
                                    ),
                                    metadata,
                                )
                                for val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                            ],  # Show actual physical values (not log-transformed)
                        ),
                        opacity=0,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        self.update_layout(
            fig,
            metadata,
            x_title="X Distance (ft)",
            y_title="Y Distance (ft)",
            z_title="Z Distance (ft)",
        )

        if labels is not None:
            self.apply_labels(
                fig,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                format_kwargs=kwargs.get("format_kwargs", None),
                coordinate_offsets=kwargs.get("coordinate_offsets", None),
            )
        return fig

    def update_layout(
        self,
        fig: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Distance (ft)",
        y_title: str = "Y Distance (ft)",
        z_title: str = "Z Distance (ft)",
    ):
        """
        Update figure layout with dimensions and scene configuration for cell block plots.

        :param fig: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        """
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode="data",  # Preserve actual aspect ratios
                dragmode="orbit",  # Allow orbital rotation around all axes
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
            ),
        )


class Scatter3DPlotter(Base3DPlotter):
    """3D Scatter plotter for sparse data."""

    supports_physical_dimensions = True

    def plot(
        self,
        data: ThreeDimensionalGrid,
        metadata: PropertyMetadata,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        threshold: float = 0.0,
        sample_rate: float = 1.0,
        marker_size: int = 3,
        opacity: typing.Optional[float] = None,
        aspect_mode: typing.Optional[str] = "cube",
        labels: typing.Optional["LabelManager"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Create a 3D scatter plot for sparse data visualization.

        :param data: 3D data array to create scatter plot from
        :param metadata: Property metadata for labeling and scaling
        :param threshold: Value threshold for point inclusion (in normalized range 0.0-1.0)
        :param sample_rate: Fraction of points to sample for performance (0.0-1.0)

        Both threshold and sample_rate are applied to the normalized data to control point inclusion and rendering.
        This can be useful for large datasets where you want to visualize only significant points and reduce rendering load.

        :param marker_size: Size of scatter plot markers
        :param opacity: Opacity of the markers (defaults to config value)
        :param labels: Optional collection of labels to add to the plot
        :return: Plotly figure object with 3D scatter plot
        """
        if data.ndim != 3:
            raise ValueError("Scatter3D plotting requires 3D data")

        # Normalize data for consistent scaling and thresholding
        # Original data will be used for hover text and colorbar
        normalized_data, display_data = self.normalize_data(
            data, metadata, normalize_range=False
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

        global_min = float(np.nanmin(values))
        global_max = float(np.nanmax(values))
        fig = go.Figure(
            data=go.Scatter3d(
                x=x_physical,
                y=y_physical,
                z=z_physical,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=values,
                    colorscale=self.get_colorscale(metadata.color_scheme),
                    opacity=opacity if opacity is not None else self.config.opacity,
                    cmin=global_min,
                    cmax=global_max,
                    colorbar=dict(
                        title=f"{metadata.display_name} ({metadata.unit})"
                        + (" - Log Scale" if metadata.log_scale else ""),
                        tickmode="array",
                        tickvals=[
                            global_min + frac * (global_max - global_min)
                            for frac in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                        ],
                        ticktext=[
                            self.format_value(
                                global_min + frac * (global_max - global_min),
                                metadata,
                            )
                            for frac in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                        ],
                    )
                    if self.config.show_colorbar
                    else None,
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        self.update_layout(
            fig,
            metadata=metadata,
            aspect_mode=aspect_mode,
            x_title=x_title,
            y_title=y_title,
            z_title=z_title,
        )

        if labels is not None:
            self.apply_labels(
                fig,
                labels,
                data=data,
                metadata=metadata,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                coordinate_offsets=kwargs.get("coordinate_offsets", None),
                format_kwargs=kwargs.get("format_kwargs", None),
            )
        return fig

    def update_layout(
        self,
        fig: go.Figure,
        metadata: PropertyMetadata,
        x_title: str = "X Cell Index",
        y_title: str = "Y Cell Index",
        z_title: str = "Z Cell Index",
        aspect_mode: typing.Optional[str] = None,
    ) -> None:
        """
        Update figure layout with dimensions and scene configuration for scatter plots.

        :param fig: Plotly figure to update
        :param metadata: Property metadata for title generation
        :param x_title: Title for X axis
        :param y_title: Title for Y axis
        :param z_title: Title for Z axis
        :param aspect_mode: Aspect mode for the 3D plot (default is "auto"). Could be any of "cube", "auto", or "data".
        """
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            scene=dict(
                xaxis_title=x_title,
                yaxis_title=y_title,
                zaxis_title=z_title,
                camera=self.config.camera_position,
                aspectmode=aspect_mode or self.config.aspect_mode or "auto",
                dragmode="orbit",  # Allow orbital rotation around all axes
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",
                ),
            ),
        )


PLOT_TYPE_NAMES: typing.Dict[PlotType, str] = {
    PlotType.VOLUME_RENDER: "3D Volume",
    PlotType.ISOSURFACE: "3D Isosurface",
    PlotType.SLICE_PLANES: "3D Slices",  # needs fix
    PlotType.SCATTER_3D: "3D Scatter",
    PlotType.CELL_BLOCKS: "3D Cell Blocks",
}


@dataclass(frozen=True)
class LabelCoordinate3D:
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

        # Convert to physical coordinates
        x_physical = (offsets[0] + self.x) * dx
        y_physical = (offsets[1] + self.y) * dy

        # For Z, we need to calculate cumulative depth
        if isinstance(self.z, int) and self.z < thickness_grid.shape[2]:
            z_physical = -np.sum(thickness_grid[self.x, self.y, : self.z])
            if offsets[2] > 0:
                # Add offset from previous layers
                z_physical -= np.sum(thickness_grid[:, :, : offsets[2]], axis=2).mean()
        else:
            z_physical = offsets[2] + self.z  # Fallback
        return x_physical, y_physical, z_physical


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

    position: LabelCoordinate3D
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
                formatted_value = Base3DPlotter.format_value(raw_value, metadata)
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

    def to_plotly_annotation(
        self,
        data_grid: typing.Optional[ThreeDimensionalGrid] = None,
        cell_dimension: typing.Optional[typing.Tuple[float, float]] = None,
        thickness_grid: typing.Optional[ThreeDimensionalGrid] = None,
        metadata: typing.Optional[PropertyMetadata] = None,
        format_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> typing.Dict[str, typing.Any]:
        """
        Convert label to Plotly 3D annotation format.

        :param data_grid: 3D data array for value extraction
        :param cell_dimension: Physical size of each cell in x and y directions (feet)
        :param thickness_grid: 3D array with height of each cell (feet)
        :param metadata: Property metadata for formatting
        :param format_kwargs: Additional formatting values
        :return: Plotly annotation dictionary
        """
        if not self.visible:
            return {}

        # Apply offset to position
        x_position = self.position.x + self.offset[0]
        y_position = self.position.y + self.offset[1]
        z_position = self.position.z + self.offset[2]

        annotation = {
            "x": x_position,
            "y": y_position,
            "z": z_position,
            "text": self.get_text(
                data_grid=data_grid,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                metadata=metadata,
                format_kwargs=format_kwargs,
            ),
            "showarrow": True,
            "font": {
                "size": self.font_size,
                "color": self.font_color,
            },
            "align": "left",
            "bgcolor": self.background_color,
            "bordercolor": self.border_color,
            "borderwidth": self.border_width,
        }
        return {k: v for k, v in annotation.items() if v is not None}


class LabelManager:
    """
    A collection of labels for 3D plots, allowing dynamic label creation and management.
    """

    def __init__(self, labels: typing.Optional[typing.Iterable[Label]] = None):
        self.labels = list(labels) if labels is not None else []

    def add_label(self, label: Label) -> None:
        """Add a label to the collection."""
        self.labels.append(label)

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

        for x in range(0, nx, x_spacing):
            for y in range(0, ny, y_spacing):
                for z in range(0, nz, z_spacing):
                    position = LabelCoordinate3D(x, y, z)
                    label = Label(
                        position=position, text_template=template, **label_kwargs
                    )
                    self.add_label(label)

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
            position = LabelCoordinate3D(x, y, z)
            label = Label(
                position=position,
                text_template=template,
                name=f"corner_{x}_{y}_{z}",
                **label_kwargs,
            )
            self.add_label(label)

    def add_well_labels(
        self,
        well_positions: typing.List[typing.Tuple[int, int, int]],
        well_names: typing.Optional[typing.List[str]] = None,
        template: str = "Well - '{name}': {formatted_value} ({unit})",
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
            position = LabelCoordinate3D(x, y, z)
            label_kwargs.setdefault("font_size", 12)  # Larger font for wells
            label_kwargs.setdefault("font_color", "#333")  # Make wells stand out
            label = Label(
                position=position,
                text_template=template,
                name=name,
                **label_kwargs,
            )
            self.add_label(label)

    def visible_labels(self) -> typing.Generator[Label, None, None]:
        """Return only visible labels."""
        return (label for label in self.labels if label.visible)

    def to_plotly_annotations(
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

        for label in self.visible_labels():
            try:
                # Check if label position is within the sliced data bounds
                if coordinate_offsets and data_grid is not None:
                    x_offset, y_offset, z_offset = coordinate_offsets

                    # Check if label falls within the original slice range
                    original_x = label.position.x + x_offset
                    original_y = label.position.y + y_offset
                    original_z = label.position.z + z_offset

                    # If label is outside the slice range, skip it
                    if (
                        original_x < x_offset
                        or original_x >= x_offset + data_grid.shape[0]
                        or original_y < y_offset
                        or original_y >= y_offset + data_grid.shape[1]
                        or original_z < z_offset
                        or original_z >= z_offset + data_grid.shape[2]
                    ):
                        continue

                # Add custom values for well names, etc.
                kwargs = {"name": label.name} if label.name else {}
                kwargs.update(format_kwargs or {})
                annotation = label.to_plotly_annotation(
                    data_grid=data_grid,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    metadata=metadata,
                    format_kwargs=kwargs,
                )

                if annotation:
                    annotations.append(annotation)

            except Exception as exc:
                print(
                    f"Warning: Failed to create annotation for label {label.name}: {exc}"
                )
                continue

        return annotations


_missing = object()


class ReservoirModelVisualizer3D:
    """
    3D visualizaer for reservoir model simulation results/data.

    This class provides a comprehensive suite of 3D plotting capabilities
    specifically designed for reservoir engineering workflows.
    """

    default_dashboard_title: str = "Reservoir Model Properties Dashboard"
    default_dashboard_properties: typing.Tuple[str, ...] = (
        "pressure",
        "oil_saturation",
        "water_saturation",
        "gas_saturation",
        "oil_viscosity",
        "water_viscosity",
    )

    def __init__(
        self,
        config: typing.Optional[PlotConfig] = None,
        registry: typing.Optional[PropertyRegistry] = None,
    ) -> None:
        """
        Initialize the visualizer with optional configuration.

        :param config: Optional configuration for 3D plotting (uses defaults if None)
        :param registry: Optional fluid property registry (uses default if None)
        """
        self.config = config or PlotConfig()
        self.plotters: typing.Dict[PlotType, Base3DPlotter] = {
            PlotType.VOLUME_RENDER: VolumeRenderer(self.config),
            PlotType.ISOSURFACE: IsosurfacePlotter(self.config),
            PlotType.SLICE_PLANES: SlicePlotter(self.config),
            PlotType.SCATTER_3D: Scatter3DPlotter(self.config),
            PlotType.CELL_BLOCKS: CellBlockPlotter(self.config),
        }
        self.registry = registry or property_registry

    def add_plotter(
        self,
        plot_type: PlotType,
        plotter_type: typing.Type[Base3DPlotter],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        """
        Add a custom plotter for a specific plot type.

        :param plot_type: The type of plot to add (must be a valid PlotType)
        :param plotter_type: The class implementing the ``Base3DPlotter`` interface
        :param args: Initialization arguments for the plotter class
        :param kwargs: Initialization keyword arguments for the plotter class
        :raises ValueError: If plot_type is not a valid PlotType
        """
        if not isinstance(plot_type, PlotType):
            raise ValueError(f"Invalid plot type: {plot_type}")
        self.plotters[plot_type] = plotter_type(self.config, *args, **kwargs)

    def _validate_and_apply_slice(
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

    def _get_property(
        self, model_state: ModelState[ThreeDimensions], name: str
    ) -> ThreeDimensionalGrid:
        """
        Get property data from model state.

        :param model_state: The model state containing reservoir model
        :param name: Name of the property to extract as defined by the `PropertyMetadata.name`
        :return: A three-dimensional numpy array containing the property data
        :raises AttributeError: If property is not found in reservoir model properties
        :raises TypeError: If property is not a numpy array
        """
        model_properties = asdict(model_state.model, recurse=False)
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
            data = np.array(data, dtype=np.float64)

        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise TypeError(f"Property '{name}' is not a 3 dimensional array.")
        return typing.cast(ThreeDimensionalGrid, data)

    def plot_property(
        self,
        model_state: ModelState[ThreeDimensions],
        property: str,
        plot_type: typing.Optional[PlotType] = None,
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
        labels: typing.Optional["LabelManager"] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Plot a specific fluid property in 3D with optional data slicing.

        :param model_state: The model state containing the reservoir model data
        :param property: Name of the property to plot (from `PropertyRegistry`)
        :param plot_type: Type of 3D plot to create (volume, isosurface, slice, scatter, cell_blocks)
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
        :param kwargs: Additional plotting parameters specific to the plot type
        :return: Plotly figure object containing the 3D visualization

        Usage Examples:

        ```python
            # Plot only cells 10-20 in X direction
            viz.plot_property(state, "pressure", x_slice=(10, 20))

            # Plot single layer at Z index 5
            viz.plot_property(state, "oil_saturation", z_slice=5)

            # Plot corner section
            viz.plot_property(state, "temperature", x_slice=(0, 25), y_slice=(0, 25), z_slice=(0, 10))

            # Use slice objects for advanced slicing
            viz.plot_property(state, "viscosity", x_slice=slice(10, 50, 2))  # Every 2nd cell
        ```
        """
        metadata = self.registry[property]
        data = self._get_property(model_state, metadata.name)

        # Get original cell dimensions and height grid
        cell_dimension = model_state.model.cell_dimension
        thickness_grid = model_state.model.thickness_grid

        # Apply slicing if any slice parameters are provided
        coordinate_offsets = None
        if any(s is not None for s in [x_slice, y_slice, z_slice]):
            data, normalized_slices = self._validate_and_apply_slice(
                data, x_slice, y_slice, z_slice
            )
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

            slice_suffix = f" - Subset: {', '.join(slice_info)}"
            if title:
                title += slice_suffix
            else:
                title = slice_suffix  # Will be used by _determine_title

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
        plotter = self.plotters[plot_type]

        # Pass coordinate offsets to plotters
        if coordinate_offsets is not None:
            kwargs["coordinate_offsets"] = coordinate_offsets

        # Pass labels to plotters that support them
        if labels is not None:
            kwargs["labels"] = labels

        # Pass physical dimensions to plotters that support them
        if plotter.supports_physical_dimensions:
            fig = plotter.plot(
                data,
                metadata,
                cell_dimension=kwargs.pop("cell_dimension", cell_dimension),
                thickness_grid=kwargs.pop("thickness_grid", thickness_grid),
                **kwargs,
            )
        else:
            fig = plotter.plot(data, metadata, **kwargs)

        final_title = self._determine_title(title, plot_type, metadata)

        # Apply final layout updates
        layout_updates: typing.Dict[str, typing.Any] = {"title": final_title}
        if width is not None:
            layout_updates["width"] = width
        if height is not None:
            layout_updates["height"] = height

        fig.update_layout(**layout_updates)
        return fig

    def _determine_title(
        self,
        custom_title: typing.Optional[str],
        plot_type: PlotType,
        metadata: PropertyMetadata,
    ) -> str:
        """
        Smart title determination logic that includes plot type information.

        :param custom_title: Custom title provided by user
        :param plot_type: Type of plot being created
        :param metadata: Property metadata
        :return: Final title to use
        """
        if custom_title is not None:
            return custom_title
        if self.config.title:
            # If config has a title, add plot type as subtitle info
            plot_name = PLOT_TYPE_NAMES.get(plot_type, "3D Plot")
            return f"{self.config.title}<br><sub>{plot_name}: {metadata.display_name}</sub>"
        plot_name = PLOT_TYPE_NAMES.get(plot_type, "3D Plot")
        return f"{plot_name}: {metadata.display_name}<br><sub>Interactive 3D Visualization</sub>"

    def plot_properties(
        self,
        model_state: ModelState[ThreeDimensions],
        properties: typing.Sequence[str],
        plot_type: typing.Optional[PlotType] = None,
        subplot_columns: int = 2,
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
        labels: typing.Optional["LabelManager"] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create a subplot figure with multiple properties.

        :param model_state: The model state containing the reservoir data
        :param properties: Sequence of property names to plot
        :param plot_type: Type of 3D plot to create for each property
        :param subplot_columns: Number of columns in the subplot grid
        :param width: Custom width for the entire subplot figure
        :param height: Custom height for the entire subplot figure
        :param title: Custom title for the entire subplot figure
        :param x_slice: X dimension slice specification:
            - int: Single index (e.g., 5 for cells[5:6, :, :])
            - tuple: Range (e.g., (10, 20) for cells[10:20, :, :])
            - slice: Full slice object (e.g., slice(10, 20, 2))
            - None: Use full dimension
        :param y_slice: Y dimension slice specification (same format as x_slice)
        :param z_slice: Z dimension slice specification (same format as x_slice)
        :param labels: Optional collection of labels to add to each subplot
        :param kwargs: Additional parameters for individual property plots. This will be
            passed to the plotter's plot method, allowing customization of each subplot.
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

            # Get individual plot - no custom size here since it's a subplot
            individual_fig = self.plot_property(
                model_state,
                prop,
                plot_type=plot_type,
                x_slice=x_slice,
                y_slice=y_slice,
                z_slice=z_slice,
                labels=labels,
                **kwargs,
            )
            # Add traces to subplot
            for trace in individual_fig.data:
                fig.add_trace(trace, row=row, col=col)

        # Determine final dimensions and title
        final_width = width or (self.config.width * subplot_columns)
        final_height = height or (self.config.height * subplot_rows)
        final_title = title or "Reservoir Properties"

        fig.update_layout(
            title=final_title,
            width=final_width,
            height=final_height,
        )
        return fig

    def create_dashboard(
        self,
        model_state: ModelState[ThreeDimensions],
        properties: typing.Optional[typing.Sequence[str]] = None,
        plot_type: typing.Optional[PlotType] = None,
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
        labels: typing.Optional["LabelManager"] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with key reservoir properties.

        :param model_state: The model state containing the reservoir data
        :param properties: Sequence of key properties to display (uses defaults if None)
        :param plot_type: Type of 3D plot to create for each property
        :param width: Custom width for the dashboard
        :param height: Custom height for the dashboard
        :param title: Custom title for the dashboard
        :param x_slice: X dimension slice specification:
            - int: Single index (e.g., 5 for cells[5:6, :, :])
            - tuple: Range (e.g., (10, 20) for cells[10:20, :, :])
            - slice: Full slice object (e.g., slice(10, 20, 2))
            - None: Use full dimension
        :param y_slice: Y dimension slice specification (same format as x_slice)
        :param z_slice: Z dimension slice specification (same format as x_slice)
        :param labels: Optional collection of labels to add to the dashboard
        :param kwargs: Additional parameters for individual property plots.
        :return: Dashboard figure with multiple reservoir property visualizations
        """
        return self.plot_properties(
            model_state,
            properties=properties or type(self).default_dashboard_properties,
            plot_type=plot_type,
            subplot_columns=3,
            width=width,
            height=height,
            title=title or type(self).default_dashboard_title,
            x_slice=x_slice,
            y_slice=y_slice,
            z_slice=z_slice,
            labels=labels,
            **kwargs,
        )

    def animate_property(
        self,
        model_states: typing.Sequence[ModelState[ThreeDimensions]],
        property: str,
        plot_type: typing.Optional[PlotType] = None,
        frame_duration: int = 200,
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
        labels: typing.Optional["LabelManager"] = None,
        **kwargs: typing.Any,
    ) -> go.Figure:
        """
        Create an animated plot showing property evolution over time.

        :param model_states: Sequence of model states representing time steps
        :param property: Name of the property to animate
        :param plot_type: Type of 3D plot for animation frames
        :param frame_duration: Duration of each frame in milliseconds
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

        metadata = self.registry[property]
        plot_type = plot_type or PlotType.VOLUME_RENDER
        # Create base figure from first state
        base_fig = self.plot_property(
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

        # Create frames for animation - ensure we get ALL states
        frames = []
        for i, state in enumerate(model_states):
            frame_fig = self.plot_property(
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

            frame = go.Frame(
                data=frame_fig.data,
                name=f"frame_{i}",
                layout=dict(title=f"{metadata.display_name} at {time_str}"),
            )
            frames.append(frame)

        # Add frames to figure
        base_fig.frames = frames

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


viz = ReservoirModelVisualizer3D()
"""Default 3D visualizer instance for reservoir models."""


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
