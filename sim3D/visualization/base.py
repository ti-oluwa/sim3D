from dataclasses import dataclass
from enum import Enum
import typing

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


__all__ = [
    "property_registry",
    "PropertyRegistry",
    "PropertyMetadata",
    "ColorScheme",
    "merge_plots",
]


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

    aliases: typing.Optional[typing.Sequence[str]] = None
    """Alternative names that can be used to refer to this property (e.g., 'pressure' for 'oil_pressure')."""

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
        "oil_pressure": PropertyMetadata(
            name="model.fluid_properties.pressure_grid",
            display_name="Oil Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["pressure"],
        ),
        "temperature": PropertyMetadata(
            name="model.fluid_properties.temperature_grid",
            display_name="Temperature",
            unit="°F",
            color_scheme=ColorScheme.INFERNO,
        ),
        # Oil Properties
        "oil_saturation": PropertyMetadata(
            name="model.fluid_properties.oil_saturation_grid",
            display_name="Oil Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            min_val=0,
            max_val=1,
            aliases=["oil_sat"],
        ),
        "oil_viscosity": PropertyMetadata(
            name="model.fluid_properties.oil_viscosity_grid",
            display_name="Oil Viscosity",
            unit="cP",
            color_scheme=ColorScheme.INFERNO,
            log_scale=True,
        ),
        "oil_density": PropertyMetadata(
            name="model.fluid_properties.oil_density_grid",
            display_name="Oil Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.PLASMA,
        ),
        "oil_compressibility": PropertyMetadata(
            name="model.fluid_properties.oil_compressibility_grid",
            display_name="Oil Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.TURBO,
            log_scale=True,
        ),
        "oil_formation_volume_factor": PropertyMetadata(
            name="model.fluid_properties.oil_formation_volume_factor_grid",
            display_name="Oil FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["oil_fvf"],
        ),
        "oil_bubble_point_pressure": PropertyMetadata(
            name="model.fluid_properties.oil_bubble_point_pressure_grid",
            display_name="Oil Bubble Point Pressure",
            unit="psi",
            color_scheme=ColorScheme.SPECTRAL,
            aliases=["oil_bpp"],
        ),
        # Water Properties
        "water_saturation": PropertyMetadata(
            name="model.fluid_properties.water_saturation_grid",
            display_name="Water Saturation",
            unit="fraction",
            color_scheme=ColorScheme.RdBu,
            min_val=0,
            max_val=1,
            aliases=["water_sat"],
        ),
        "water_viscosity": PropertyMetadata(
            name="model.fluid_properties.water_viscosity_grid",
            display_name="Water Viscosity",
            unit="cP",
            color_scheme=ColorScheme.BALANCE,
            log_scale=True,
        ),
        "water_density": PropertyMetadata(
            name="model.fluid_properties.water_density_grid",
            display_name="Water Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.EARTH,
        ),
        "water_compressibility": PropertyMetadata(
            name="model.fluid_properties.water_compressibility_grid",
            display_name="Water Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.VIRIDIS,
            log_scale=True,
        ),
        "water_formation_volume_factor": PropertyMetadata(
            name="model.fluid_properties.water_formation_volume_factor_grid",
            display_name="Water FVF",
            unit="bbl/STB",
            color_scheme=ColorScheme.PLASMA,
            aliases=["water_fvf"],
        ),
        "water_bubble_point_pressure": PropertyMetadata(
            name="model.fluid_properties.water_bubble_point_pressure_grid",
            display_name="Water Bubble Point Pressure",
            unit="psi",
            color_scheme=ColorScheme.INFERNO,
            aliases=["water_bpp"],
        ),
        "water_salinity": PropertyMetadata(
            name="model.fluid_properties.water_salinity_grid",
            display_name="Water Salinity",
            unit="ppm NaCl",
            color_scheme=ColorScheme.CIVIDIS,
            aliases=["salinity"],
        ),
        "gas_saturation": PropertyMetadata(
            name="model.fluid_properties.gas_saturation_grid",
            display_name="Gas Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            min_val=0,
            max_val=1,
            aliases=["gas_sat"],
        ),
        "gas_viscosity": PropertyMetadata(
            name="model.fluid_properties.gas_viscosity_grid",
            display_name="Gas Viscosity",
            unit="cP",
            color_scheme=ColorScheme.TURBO,
            log_scale=True,
        ),
        "gas_density": PropertyMetadata(
            name="model.fluid_properties.gas_density_grid",
            display_name="Gas Density",
            unit="lbm/ft³",
            color_scheme=ColorScheme.RdYlBu,
        ),
        "gas_compressibility": PropertyMetadata(
            name="model.fluid_properties.gas_compressibility_grid",
            display_name="Gas Compressibility",
            unit="psi⁻¹",
            color_scheme=ColorScheme.SPECTRAL,
            log_scale=True,
        ),
        "gas_formation_volume_factor": PropertyMetadata(
            name="model.fluid_properties.gas_formation_volume_factor_grid",
            display_name="Gas FVF",
            unit="ft³/SCF",
            color_scheme=ColorScheme.RdBu,
            aliases=["gas_fvf"],
        ),
        "gas_to_oil_ratio": PropertyMetadata(
            name="model.fluid_properties.gas_to_oil_ratio_grid",
            display_name="Gas-Oil Ratio",
            unit="SCF/STB",
            color_scheme=ColorScheme.BALANCE,
            aliases=[
                "gor",
                "gas_oil_ratio",
                "solution_gas_oil_ratio",
                "solution_gor",
                "solution_gas_to_oil_ratio",
            ],
        ),
        "gas_gravity": PropertyMetadata(
            name="model.fluid_properties.gas_gravity_grid",
            display_name="Gas Gravity",
            unit="dimensionless",
            color_scheme=ColorScheme.EARTH,
            aliases=["gas_specific_gravity", "gas_sg"],
        ),
        "gas_molecular_weight": PropertyMetadata(
            name="model.fluid_properties.gas_molecular_weight_grid",
            display_name="Gas Molecular Weight",
            unit="g/mol",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["gas_mw"],
        ),
        # API Gravity
        "oil_api_gravity": PropertyMetadata(
            name="model.fluid_properties.oil_api_gravity_grid",
            display_name="Oil API Gravity",
            unit="°API",
            color_scheme=ColorScheme.PLASMA,
            aliases=["api_gravity", "api"],
        ),
        "oil_specific_gravity": PropertyMetadata(
            name="model.fluid_properties.oil_specific_gravity_grid",
            display_name="Oil Specific Gravity",
            unit="dimensionless",
            color_scheme=ColorScheme.INFERNO,
            aliases=["oil_sg", "oil_gravity"],
        ),
        "thickness": PropertyMetadata(
            name="model.thickness_grid",
            display_name="Cell Thickness",
            unit="ft",
            color_scheme=ColorScheme.CIVIDIS,
        ),
        "permeability_x": PropertyMetadata(
            name="model.rock_properties.absolute_permeability.x",
            display_name="Permeability X",
            unit="mD",
            color_scheme=ColorScheme.MAGMA,
            aliases=["kx", "perm_x"],
        ),
        "permeability_y": PropertyMetadata(
            name="model.rock_properties.absolute_permeability.y",
            display_name="Permeability Y",
            unit="mD",
            color_scheme=ColorScheme.TURBO,
            aliases=["ky", "perm_y"],
        ),
        "permeability_z": PropertyMetadata(
            name="model.rock_properties.absolute_permeability.z",
            display_name="Permeability Z",
            unit="mD",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["kz", "perm_z"],
        ),
        "porosity": PropertyMetadata(
            name="model.rock_properties.porosity_grid",
            display_name="Porosity",
            unit="fraction",
            color_scheme=ColorScheme.PLASMA,
        ),
        "net_to_gross_ratio": PropertyMetadata(
            name="model.rock_properties.net_to_gross_ratio_grid",
            display_name="Net to Gross Ratio",
            unit="fraction",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["ngr", "ntg", "net_to_gross", "ntg_ratio"],
        ),
        "irreducible_water_saturation": PropertyMetadata(
            name="model.rock_properties.irreducible_water_saturation_grid",
            display_name="Irreducible Water Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            aliases=["swc", "irreducible_water_sat"],
        ),
        "residual_oil_saturation_water": PropertyMetadata(
            name="model.rock_properties.residual_oil_saturation_water_grid",
            display_name="Residual Oil Saturation (Water Flooded)",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            aliases=["sorw", "sor", "residual_oil_sat", "residual_oil_saturation"],
        ),
        "residual_oil_saturation_gas": PropertyMetadata(
            name="model.rock_properties.residual_oil_saturation_gas_grid",
            display_name="Residual Oil Saturation (Gas Flooded)",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            aliases=["sorg"],
        ),
        "residual_gas_saturation": PropertyMetadata(
            name="model.rock_properties.residual_gas_saturation_grid",
            display_name="Residual Gas Saturation",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
            aliases=["sgr", "residual_gas_sat"],
        ),
        "oil_injection_rate": PropertyMetadata(
            name="injection.oil",
            display_name="Oil Injection Rate",
            unit="ft³/day",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["oil_injection"],
        ),
        "water_injection_rate": PropertyMetadata(
            name="injection.water",
            display_name="Water Injection Rate",
            unit="ft³/day",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["water_injection"],
        ),
        "gas_injection_rate": PropertyMetadata(
            name="injection.gas",
            display_name="Gas Injection Rate",
            unit="ft³/day",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["gas_injection"],
        ),
        "total_injection_rate": PropertyMetadata(
            name="injection.total",
            display_name="Total Injection Rate",
            unit="ft³/day",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["total_injection"],
        ),
        "oil_production_rate": PropertyMetadata(
            name="production.oil",
            display_name="Oil Production Rate",
            unit="ft³/day",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["oil_production"],
        ),
        "water_production_rate": PropertyMetadata(
            name="production.water",
            display_name="Water Production Rate",
            unit="ft³/day",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["water_production"],
        ),
        "gas_production_rate": PropertyMetadata(
            name="production.gas",
            display_name="Gas Production Rate",
            unit="ft³/day",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["gas_production"],
        ),
        "total_production_rate": PropertyMetadata(
            name="production.total",
            display_name="Total Production Rate",
            unit="ft³/day",
            color_scheme=ColorScheme.RdYlBu,
            aliases=["total_production"],
        ),
    }

    def __init__(self) -> None:
        """
        Initialize the property registry.

        This class is a singleton and should not be instantiated directly.
        Use the class methods to access properties and metadata.
        """
        self._properties = _get_expanded_aliases(type(self).PROPERTIES)

    def get_available_properties(self) -> typing.List[str]:
        """
        Get list of all available property names.

        :return: List of property names that can be used for visualization
        """
        return list(self._properties.keys())

    @staticmethod
    def clean_property_name(name: str) -> str:
        """Clean and standardize property name for lookup."""
        return name.strip().replace("-", "_").replace(" ", "_").lower()

    def get_metadata(self, name: str) -> PropertyMetadata:
        """
        Get metadata for a specific property.

        :param property: Name of the property to get metadata for
        :return: `PropertyMetadata` object containing display information
        :raises ValueError: If property is not found in the registry
        """
        name = self.clean_property_name(name)
        if name not in self._properties:
            raise ValueError(
                f"Unknown property: {name}. Available: {', '.join(self.get_available_properties())}"
            )
        return self._properties[name]

    def __getitem__(self, name: str, /) -> PropertyMetadata:
        return self.get_metadata(name)

    def __setitem__(self, name: str, value: PropertyMetadata, /) -> None:
        if not isinstance(value, PropertyMetadata):
            raise TypeError("Value must be a `PropertyMetadata` instance")

        name = self.clean_property_name(name)
        self._properties[name] = value

    def __contains__(self, name: str, /) -> bool:
        """Check if a property exists in the registry."""
        name = self.clean_property_name(name)
        return name in self._properties

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over the property names."""
        return iter(self._properties.keys())


def _get_expanded_aliases(
    properties: typing.Dict[str, PropertyMetadata],
) -> typing.Dict[str, PropertyMetadata]:
    """Return a new dict with all aliases expanded to point to the same metadata."""
    expanded = {}
    for prop_name, metadata in properties.items():
        expanded[prop_name] = metadata
        if metadata.aliases:
            for alias in metadata.aliases:
                alias_clean = PropertyRegistry.clean_property_name(alias)
                if alias_clean in properties or alias_clean in expanded:
                    raise ValueError(
                        f"Alias '{alias}' for property '{prop_name}' conflicts with existing property or alias."
                    )
                expanded[alias_clean] = metadata
    return expanded


property_registry = PropertyRegistry()


def _copy_scene_properties(
    source_fig: go.Figure,
    target_fig: go.Figure,
    scene_key: str,
) -> None:
    """
    Copy all 3D scene properties from source figure to target figure.

    This function performs a comprehensive copy of all scene-related properties including
    camera settings, axis configurations, annotations, backgrounds, and interaction modes.

    :param source_fig: Source Plotly Figure to copy scene properties from
    :param target_fig: Target Plotly Figure to copy scene properties to
    :param scene_key: Scene key in the target layout (e.g., 'scene', 'scene2', 'scene3')
    """
    if not hasattr(source_fig, "layout") or not hasattr(source_fig.layout, "scene"):
        return

    source_scene = source_fig.layout.scene  # type: ignore[attr-defined]
    if not source_scene:
        return

    target_scene = target_fig.layout[scene_key]

    # Camera settings (viewpoint, projection)
    if hasattr(source_scene, "camera") and source_scene.camera:
        target_scene.camera = source_scene.camera

    # Axis properties (labels, ranges, gridlines, etc.)
    if hasattr(source_scene, "xaxis") and source_scene.xaxis:
        target_scene.xaxis = source_scene.xaxis

    if hasattr(source_scene, "yaxis") and source_scene.yaxis:
        target_scene.yaxis = source_scene.yaxis

    if hasattr(source_scene, "zaxis") and source_scene.zaxis:
        target_scene.zaxis = source_scene.zaxis

    # Aspect ratio and scaling
    if hasattr(source_scene, "aspectmode") and source_scene.aspectmode:
        target_scene.aspectmode = source_scene.aspectmode

    if hasattr(source_scene, "aspectratio") and source_scene.aspectratio:
        target_scene.aspectratio = source_scene.aspectratio

    # Background and grid settings
    if hasattr(source_scene, "bgcolor") and source_scene.bgcolor:
        target_scene.bgcolor = source_scene.bgcolor

    # Interaction modes
    if hasattr(source_scene, "dragmode") and source_scene.dragmode:
        target_scene.dragmode = source_scene.dragmode

    if hasattr(source_scene, "hovermode") and source_scene.hovermode:
        target_scene.hovermode = source_scene.hovermode

    # Domain (position in the layout)
    if hasattr(source_scene, "domain") and source_scene.domain:
        # Note: Domain is usually managed by make_subplots, so we may skip this
        # to avoid conflicts, but include it for completeness
        pass

    # Annotations specific to this scene
    if hasattr(source_scene, "annotations") and source_scene.annotations:
        target_scene.annotations = source_scene.annotations

    # Uirevision (for preserving user interactions)
    if hasattr(source_scene, "uirevision") and source_scene.uirevision:
        target_scene.uirevision = source_scene.uirevision


def merge_plots(
    *figures: go.Figure,
    rows: typing.Optional[int] = None,
    cols: typing.Optional[int] = None,
    subplot_titles: typing.Optional[typing.Sequence[str]] = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    vertical_spacing: typing.Optional[float] = None,
    horizontal_spacing: typing.Optional[float] = None,
    height: typing.Optional[int] = None,
    width: typing.Optional[int] = None,
    title: typing.Optional[str] = None,
    show_legend: bool = True,
) -> go.Figure:
    """
    Display multiple Plotly figures in a single unified plot with subplots.

    This function combines multiple independent Plotly figures into a single figure
    with subplots arranged in a grid layout. It intelligently handles figure layout,
    traces, annotations, and preserves original styling while creating a cohesive
    multi-panel visualization.

    **Layout Strategy:**
    - If `rows` and `cols` are not specified, automatically calculates optimal grid dimensions
    - Arranges figures left-to-right, top-to-bottom in the subplot grid
    - Preserves individual figure properties (colorbars, annotations, axes labels)
    - Adjusts spacing based on number of subplots for optimal viewing

    **Use Cases:**
    - Compare multiple reservoir properties side-by-side
    - Create multi-panel diagnostic plots
    - Combine pressure, saturation, and production rate visualizations
    - Build comprehensive reservoir analysis dashboards

    :param figures: Variable number of Plotly Figure objects to display together.
        Each figure is treated as an independent subplot. Empty figures are skipped.
    :param rows: Number of rows in the subplot grid. If None, automatically calculated
        based on number of figures and cols parameter.
    :param cols: Number of columns in the subplot grid. If None, automatically calculated
        to create a roughly square grid layout.
    :param subplot_titles: Sequence of titles for each subplot. Length must match number
        of figures. If None, uses original figure titles if available.
    :param shared_xaxes: If True, all subplots share the same x-axis range. Useful for
        comparing data over the same spatial or temporal domain (default: False).
    :param shared_yaxes: If True, all subplots share the same y-axis range. Useful for
        comparing magnitudes across different properties (default: False).
    :param vertical_spacing: Vertical spacing between subplot rows as a fraction of
        plot height (0 to 1). If None, automatically calculated based on grid size.
        Smaller values create tighter layouts.
    :param horizontal_spacing: Horizontal spacing between subplot columns as a fraction
        of plot width (0 to 1). If None, automatically calculated based on grid size.
        Smaller values create tighter layouts.
    :param height: Total height of the combined figure in pixels. If None, scales
        automatically based on number of rows (default: 400 * rows).
    :param width: Total width of the combined figure in pixels. If None, scales
        automatically based on number of columns (default: 600 * cols).
    :param title: Main title for the entire combined figure. Displayed at the top
        of the visualization. If None, no main title is shown.
    :param show_legend: If True, displays legends for all traces. If False, hides
        all legends to reduce clutter (default: True).
    :return: Combined Plotly Figure object with all input figures as subplots.
        Can be displayed with .show() or saved with .write_html() / .write_image().
    :raises ValueError: If subplot_titles length doesn't match number of figures,
        or if rows * cols is insufficient for the number of figures.
    :raises TypeError: If any input is not a valid Plotly Figure object.

    **Example Usage:**

    ```python
    # Basic usage with automatic layout:
    fig1 = create_pressure_plot(...)
    fig2 = create_saturation_plot(...)
    combined = merge_plots(fig1, fig2)
    combined.show()

    # Custom grid with titles:
    merge_plots(
        pressure_fig,
        oil_sat_fig,
        water_sat_fig,
        gas_sat_fig,
        rows=2,
        cols=2,
        subplot_titles=["Pressure", "Oil Sat", "Water Sat", "Gas Sat"],
        title="Reservoir Properties at Time Step 100",
        height=800,
        width=1200,
    )

    # Shared axes for comparison:
    merge_plots(
        timestep_0_fig,
        timestep_50_fig,
        timestep_100_fig,
        cols=3,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=["Initial", "Mid", "Final"],
    )
    ```
    """
    # Validate inputs
    if not figures:
        raise ValueError("At least one figure must be provided")

    # Filter out None values and validate figure types
    valid_figures = []
    for idx, fig in enumerate(figures):
        if fig is None:
            continue
        if not isinstance(fig, go.Figure):
            raise TypeError(
                f"Argument at position {idx} is not a Plotly Figure object. "
                f"Got type: {type(fig).__name__}"
            )
        # Skip empty figures (no traces)
        if len(fig.data) > 0:  # type: ignore
            valid_figures.append(fig)

    if not valid_figures:
        raise ValueError("No valid figures with data to display")

    num_figures = len(valid_figures)

    # Calculate optimal grid dimensions if not specified
    if rows is None and cols is None:
        # Create roughly square grid, slightly favoring more columns
        cols = int(np.ceil(np.sqrt(num_figures)))
        rows = int(np.ceil(num_figures / cols))
    elif rows is None:
        if cols is None or cols <= 0:
            raise ValueError("cols must be a positive integer")
        rows = int(np.ceil(num_figures / cols))
    elif cols is None:
        if rows <= 0:
            raise ValueError("rows must be a positive integer")
        cols = int(np.ceil(num_figures / rows))

    # Validate grid can accommodate all figures
    if rows * cols < num_figures:
        raise ValueError(
            f"Grid size ({rows} rows x {cols} cols = {rows * cols} subplots) "
            f"is insufficient for {num_figures} figures. "
            f"Need at least {num_figures} subplot positions."
        )

    # Validate subplot_titles if provided
    if subplot_titles is not None:
        if len(subplot_titles) != num_figures:
            raise ValueError(
                f"Length of subplot_titles ({len(subplot_titles)}) must match "
                f"number of valid figures ({num_figures})"
            )
    else:
        # Try to extract titles from original figures
        subplot_titles = []
        for fig in valid_figures:
            fig_title = ""
            if hasattr(fig, "layout") and hasattr(fig.layout, "title"):
                if isinstance(fig.layout.title, dict):
                    fig_title = fig.layout.title.get("text", "")
                elif hasattr(fig.layout.title, "text"):
                    fig_title = fig.layout.title.text or ""
                else:
                    fig_title = str(fig.layout.title) if fig.layout.title else ""
            subplot_titles.append(fig_title)

    # Calculate automatic spacing if not specified
    if vertical_spacing is None:
        # More spacing for fewer rows, less for many rows
        vertical_spacing = max(0.05, min(0.15, 0.3 / rows))

    if horizontal_spacing is None:
        # More spacing for fewer columns, less for many columns
        horizontal_spacing = max(0.05, min(0.15, 0.3 / cols))

    # Calculate automatic dimensions if not specified
    if height is None:
        height = int(400 * rows)

    if width is None:
        width = int(600 * cols)

    # Determine subplot types - check if any figures have 3D traces
    specs = [[{"type": "xy"} for _ in range(cols)] for _ in range(rows)]
    for idx, fig in enumerate(valid_figures):
        row = idx // cols
        col = idx % cols

        # Check if figure contains 3D traces (volume, isosurface, scatter3d, surface, mesh3d, cone)
        has_3d = any(
            hasattr(trace, "type")
            and str(trace.type).lower()
            in (
                "volume",
                "isosurface",
                "scatter3d",
                "surface",
                "mesh3d",
                "cone",
                "streamtube",
            )
            for trace in fig.data
        )
        if has_3d:
            specs[row][col] = {"type": "scene"}

    # Create subplot figure
    try:
        combined_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(subplot_titles),
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            specs=specs,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to create subplot layout: {exc}. "
            f"Check that your grid dimensions ({rows}x{cols}) and spacing parameters are valid."
        ) from exc

    # Add traces from each figure to the combined figure
    for idx, fig in enumerate(valid_figures):
        row = idx // cols + 1
        col = idx % cols + 1

        # Add all traces from the figure
        for trace in fig.data:
            # Create a copy to avoid modifying original
            trace_copy = trace.__class__(trace)

            # Preserve legend group if it exists
            if not show_legend:
                trace_copy.showlegend = False
            elif hasattr(trace, "legendgroup") and trace.legendgroup:
                # Append subplot index to legend group to avoid conflicts
                trace_copy.legendgroup = f"{trace.legendgroup}_subplot{idx}"

            combined_fig.add_trace(trace_copy, row=row, col=col)

        # Copy axis properties from original figure
        if hasattr(fig, "layout"):
            # Check if this subplot is a 3D scene
            is_scene = specs[row - 1][col - 1].get("type") == "scene"

            if is_scene:
                # Handle 3D scene properties - comprehensive copy of all scene settings
                scene_key = (
                    "scene"
                    if row == 1 and col == 1
                    else f"scene{(row - 1) * cols + col}"
                )
                _copy_scene_properties(fig, combined_fig, scene_key)
            else:
                # Handle 2D x-axis and y-axis properties
                if hasattr(fig.layout, "xaxis") and fig.layout.xaxis:
                    xaxis_key = (
                        "xaxis"
                        if row == 1 and col == 1
                        else f"xaxis{(row - 1) * cols + col}"
                    )
                    if hasattr(fig.layout.xaxis, "title") and fig.layout.xaxis.title:
                        combined_fig.layout[xaxis_key].title = fig.layout.xaxis.title
                    if hasattr(fig.layout.xaxis, "type"):
                        combined_fig.layout[xaxis_key].type = fig.layout.xaxis.type

                if hasattr(fig.layout, "yaxis") and fig.layout.yaxis:
                    yaxis_key = (
                        "yaxis"
                        if row == 1 and col == 1
                        else f"yaxis{(row - 1) * cols + col}"
                    )
                    if hasattr(fig.layout.yaxis, "title") and fig.layout.yaxis.title:
                        combined_fig.layout[yaxis_key].title = fig.layout.yaxis.title
                    if hasattr(fig.layout.yaxis, "type"):
                        combined_fig.layout[yaxis_key].type = fig.layout.yaxis.type

            # Copy colorbar settings for heatmaps/contours with smart positioning
            for trace_idx, trace in enumerate(fig.data):
                if hasattr(trace, "colorbar") and trace.colorbar:
                    # Find corresponding trace in combined figure
                    combined_trace_idx = (
                        sum(len(valid_figures[i].data) for i in range(idx)) + trace_idx
                    )
                    if combined_trace_idx < len(combined_fig.data):  # type: ignore
                        # Create a copy of colorbar to avoid modifying original
                        colorbar_dict = (
                            dict(trace.colorbar)
                            if isinstance(trace.colorbar, dict)
                            else {}
                        )

                        # Position colorbar to the right of its specific subplot with gaps
                        # Calculate horizontal position based on column
                        col_fraction = col / cols
                        colorbar_x_position = col_fraction - 0.5 / cols + 0.95 / cols

                        # Calculate vertical position based on row (for multi-row layouts)
                        row_fraction = (
                            rows - row + 1
                        ) / rows  # Inverted because plotly counts from bottom
                        subplot_height = 1.0 / rows
                        colorbar_y_position = row_fraction - subplot_height / 2

                        # Add vertical padding to create gaps between colorbars
                        vertical_padding = 0.05  # 5% padding on each side
                        colorbar_length = max(
                            0.3, (0.8 / rows) - (2 * vertical_padding)
                        )

                        # Set colorbar position and size to fit within subplot bounds with gaps
                        colorbar_dict.update(
                            {
                                "x": colorbar_x_position,
                                "y": colorbar_y_position,
                                "len": colorbar_length,  # Reduced length to create vertical gaps
                                "thickness": 15,  # Consistent thickness
                                "xanchor": "left",
                                "yanchor": "middle",
                            }
                        )
                        combined_fig.data[combined_trace_idx].colorbar = colorbar_dict

    # Apply global layout settings
    combined_fig.update_layout(
        height=height,
        width=width,
        showlegend=show_legend,
    )
    # Add main title if provided
    if title:
        combined_fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
    return combined_fig
