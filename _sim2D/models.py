"""Data models and schemas for the 2D reservoir."""

import typing
import enum
from dataclasses import dataclass, field

from _sim2D.types import TwoDimensionalGrid
from _sim2D.boundary_conditions import BoundaryConditions


__all__ = [
    "RelativePermeabilityParameters",
    "CapillaryPressureParameters",
    "WettabilityType",
    "FluidProperties",
    "RockProperties",
    "TwoDimensionalReservoirModel",
]


@dataclass(slots=True, frozen=True)
class RelativePermeabilityParameters:
    """Parameters defining the relative permeability curves (e.g., Corey exponents)."""

    water_exponent: float = 2.0  # Nw
    """Corey exponent for water relative permeability."""
    oil_exponent: float = 2.0  # No
    """Corey exponent for oil relative permeability."""
    gas_exponent: float = 2.0  # Ng
    """Corey exponent for gas relative permeability."""


class WettabilityType(enum.Enum):
    """Enum representing the wettability type of the reservoir rock."""

    WATER_WET = "water_wet"
    OIL_WET = "oil_wet"


@dataclass(slots=True, frozen=True)
class CapillaryPressureParameters:
    """Parameters defining the capillary pressure curve (e.g., Brooks-Corey)."""

    wettability: WettabilityType = WettabilityType.WATER_WET
    """
    Wettability type of the reservoir rock.

    This determines how fluids interact with the rock surface, affecting capillary pressure.
    """
    oil_water_entry_pressure_oil_wet: float = 14.5037
    """
    Pressure (psi) at which oil starts to displace water in an oil-wet reservoir.

    This is the entry pressure for oil in an oil-wet system.
    """
    oil_water_pore_size_distribution_index_oil_wet: float = 2.0
    """
    Pore size distribution index for oil in an oil-wet reservoir.

    This parameter characterizes the distribution of pore sizes in the rock that affects capillary pressure.
    The value is typically lower than that for water-wet reservoirs, indicating a different pore structure.
    The lower the value, the more varied the pore sizes are, which affects how fluids interact with the rock.
    """
    oil_water_entry_pressure_water_wet: float = 7.2519
    """
    Pressure (psi) at which oil starts to displace water in a water-wet reservoir.

    This is the entry pressure for oil in a water-wet system.
    """
    oil_water_pore_size_distribution_index_water_wet: float = 3.0
    """
    Pore size distribution index for oil in a water-wet reservoir.

    This parameter characterizes the distribution of pore sizes in the rock that affects capillary pressure.
    The value is typically higher than that for oil-wet reservoirs, indicating a different pore structure.
    The higher the value, the more uniform the pore sizes are, which affects how fluids interact with the rock.
    """
    gas_oil_entry_pressure: float = 29.0075
    """
    Pressure (psi) at which gas starts to displace oil in the reservoir.

    This is the entry pressure for gas in an oil system.
    """
    gas_oil_pore_size_distribution_index: float = 2.0
    """
    Pore size distribution index for gas in an oil system.

    This parameter characterizes the distribution of pore sizes in the rock that affects capillary pressure.
    The value is typically lower than that for water-wet reservoirs, indicating a different pore structure.
    The lower the value, the more varied the pore sizes are, which affects how fluids interact with the rock.
    """


@dataclass(slots=True, frozen=True)
class FluidProperties:
    """
    Represents the fluid properties of the reservoir.

    These properties are liable to change over time due to flow.
    """

    pressure_grid: TwoDimensionalGrid
    """2D numpy array representing the pressure distribution in the reservoir (psi)."""
    temperature_grid: TwoDimensionalGrid
    """2D numpy array representing the temperature distribution across the reservoir (°F)."""
    oil_bubble_point_pressure_grid: TwoDimensionalGrid
    """2D numpy array representing the bubble point pressure distribution in the reservoir (psi)."""
    oil_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the reservoir fluid (Oil) saturation distribution in the reservoir (fraction)."""
    oil_viscosity_grid: TwoDimensionalGrid
    """2D numpy array representing the reservoir fluid (Oil) viscosity distribution in the reservoir in (cP)."""
    oil_compressibility_grid: TwoDimensionalGrid
    """2D numpy array representing the oil compressibility distribution in the reservoir (psi⁻¹)."""
    oil_density_grid: TwoDimensionalGrid
    """2D numpy array representing the oil density distribution in the reservoir (lbm/ft³)."""
    water_bubble_point_pressure_grid: TwoDimensionalGrid
    """2D numpy array representing the bubble point pressure distribution for water in the reservoir (psi)."""
    water_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the reservoir fluid (Water) saturation distribution in the reservoir (fraction)."""
    water_viscosity_grid: TwoDimensionalGrid
    """2D numpy array representing the reservoir fluid (Water) viscosity distribution in the reservoir in (cP)."""
    water_compressibility_grid: TwoDimensionalGrid
    """2D numpy array representing the water compressibility distribution in the reservoir (psi⁻¹)."""
    water_density_grid: TwoDimensionalGrid
    """2D numpy array representing the water density distribution in the reservoir (lbm/ft³)."""
    gas_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the reservoir fluid (Gas) saturation distribution in the reservoir (fraction)."""
    gas_viscosity_grid: TwoDimensionalGrid
    """2D numpy array representing the reservoir fluid (Gas) viscosity distribution in the reservoir in (cP)."""
    gas_compressibility_grid: TwoDimensionalGrid
    """2D numpy array representing the gas compressibility distribution in the reservoir (psi⁻¹)."""
    gas_density_grid: TwoDimensionalGrid
    """2D numpy array representing the gas density distribution in the reservoir (lbm/ft³)."""
    gas_to_oil_ratio_grid: TwoDimensionalGrid
    """2D numpy array representing the gas-to-oil ratio distribution at standard conditions (SCF/STB)."""
    oil_formation_volume_factor_grid: TwoDimensionalGrid
    """2D numpy array representing the oil formation volume factor distribution (bbl/STB)."""
    gas_formation_volume_factor_grid: TwoDimensionalGrid
    """2D numpy array representing the gas formation volume factor distribution (ft³/SCF)."""
    water_formation_volume_factor_grid: TwoDimensionalGrid
    """2D numpy array representing the water formation volume factor distribution (bbl/STB)."""
    water_salinity_grid: TwoDimensionalGrid
    """2D numpy array representing the water salinity distribution (ppm NaCl)."""


@dataclass(slots=True, frozen=True)
class RockProperties:
    """
    Represents the rock properties of the reservoir.

    These properties remain constant over time.
    """

    compressibility: float
    """Reservoir rock compressibility in (psi⁻¹)"""
    absolute_permeability_grid: TwoDimensionalGrid
    """2D numpy array representing the permeability distribution across the reservoir rock (mD)."""
    net_to_gross_ratio_grid: TwoDimensionalGrid
    """2D numpy array representing the net-to-gross ratio distribution across the reservoir rock (fraction)."""
    porosity_grid: TwoDimensionalGrid
    """2D numpy array representing the porosity distribution across the reservoir rock (fraction)."""
    irreducible_water_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the irreducible water saturation distribution (fraction)."""
    residual_oil_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the residual oil saturation distribution (fraction)."""
    residual_gas_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the residual gas saturation distribution (fraction)."""
    relative_permeability_params: RelativePermeabilityParameters = field(
        default_factory=RelativePermeabilityParameters
    )
    """Parameters for relative permeability curves."""
    capillary_pressure_params: CapillaryPressureParameters = field(
        default_factory=CapillaryPressureParameters
    )
    """Parameters for capillary pressure curve."""


@dataclass(slots=True, frozen=True)
class TwoDimensionalReservoirModel:
    """Represents a 2D reservoir model with its properties and state."""

    grid_dimension: typing.Tuple[int, int]
    """Number of cells in the grid. A tuple of number of cells in x and y directions (cell_count_x, cell_count_y)."""
    cell_dimension: typing.Tuple[float, float]
    """Size of each cell in the grid (cell_size_x, cell_size_y) in ft."""
    height_grid: TwoDimensionalGrid
    """2D numpy array representing the height of each cell in the reservoir (ft)."""
    fluid_properties: FluidProperties
    """Fluid properties of the reservoir model."""
    rock_properties: RockProperties
    """Rock properties of the reservoir model."""
    boundary_conditions: BoundaryConditions = field(default_factory=BoundaryConditions)
    """Boundary conditions for the simulation (e.g., no-flow, constant pressure)."""

