import typing
from dataclasses import dataclass
from enum import Enum


__all__ = [
    "property_registry",
    "PropertyRegistry",
    "PropertyMetadata",
    "ColorScheme",
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
            color_scheme=ColorScheme.INFERNO,
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
        "api_gravity": PropertyMetadata(
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
            name="rock_properties.residual_oil_saturation_water_grid",
            display_name="Residual Oil Saturation (Water Flooded)",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
        ),
        "residual_oil_saturation_water": PropertyMetadata(
            name="rock_properties.residual_oil_saturation_water_grid",
            display_name="Residual Oil Saturation (Water Flooded)",
            unit="fraction",
            color_scheme=ColorScheme.MAGMA,
        ),
        "residual_oil_saturation_gas": PropertyMetadata(
            name="rock_properties.residual_oil_saturation_gas_grid",
            display_name="Residual Oil Saturation (Gas Flooded)",
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


property_registry = PropertyRegistry()
