"""Data models and schemas for the N-dimensional reservoir."""

import typing
import enum
from attrs import define, field
import numpy as np

from sim3D.types import NDimension, NDimensionalGrid
from sim3D.boundary_conditions import BoundaryConditions


__all__ = [
    "RelativePermeabilityParameters",
    "CapillaryPressureParameters",
    "WettabilityType",
    "FluidProperties",
    "RockProperties",
    "ReservoirModel",
    "RockPermeability",
]


@define(slots=True, frozen=True)
class RelativePermeabilityParameters:
    """Parameters defining the relative permeability curves (e.g., Corey exponents)."""

    water_exponent: float = 2.0  # Nw
    """Corey exponent for water relative permeability."""
    oil_exponent: float = 2.0  # No
    """Corey exponent for oil relative permeability."""
    gas_exponent: float = 2.0  # Ng
    """Corey exponent for gas relative permeability."""


class WettabilityType(str, enum.Enum):
    """Enum representing the wettability type of the reservoir rock."""

    WATER_WET = "water_wet"
    OIL_WET = "oil_wet"


@define(slots=True, frozen=True)
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


@define(slots=True, frozen=True)
class FluidProperties(typing.Generic[NDimension]):
    """
    Fluid properties of a reservoir model.

    Some of these properties are liable to change over time due to flow.

    Changing properties include:
    - Pressure
    - Temperature
    - Oil saturation
    - Oil viscosity
    - Oil compressibility
    - Oil density
    - Water saturation
    - Water viscosity
    - Water compressibility
    - Water density
    - Gas saturation
    - Gas viscosity
    - Gas compressibility
    - Gas density
    - Gas-to-oil ratio
    - Oil formation volume factor
    - Gas formation volume factor
    - Water formation volume factor
    - Oil bubble point pressure
    - Water bubble point pressure

    Constant properties include:
    - Oil specific gravity
    - Oil API gravity
    - Water salinity
    - Gas gravity
    - Gas molecular weight
    These properties are typically constant for a given fluid type, e.g., light oil, seawater, methane.
    """

    pressure_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the pressure distribution in the reservoir (psi)."""
    temperature_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the temperature distribution across the reservoir (°F)."""
    oil_bubble_point_pressure_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the bubble point pressure distribution in the reservoir (psi)."""
    oil_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Oil) saturation distribution in the reservoir (fraction)."""
    oil_viscosity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Oil) viscosity distribution in the reservoir in (cP)."""
    oil_compressibility_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil compressibility distribution in the reservoir (psi⁻¹)."""
    oil_specific_gravity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil specific gravity distribution in the reservoir (dimensionless). Should be constant for a given oil, e.g., 0.85 for light oil)."""
    oil_api_gravity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil API gravity distribution in the reservoir (°API). should be constant for a given oil, e.g., 35°API for light oil)."""
    oil_density_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil density distribution (usually live-oil) in the reservoir (lbm/ft³)."""
    water_bubble_point_pressure_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the bubble point pressure distribution for water in the reservoir (psi)."""
    water_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Water) saturation distribution in the reservoir (fraction)."""
    water_viscosity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Water) viscosity distribution in the reservoir in (cP)."""
    water_compressibility_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water compressibility distribution in the reservoir (psi⁻¹)."""
    water_density_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water density distribution in the reservoir (lbm/ft³)."""
    gas_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Gas) saturation distribution in the reservoir (fraction)."""
    gas_viscosity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the reservoir fluid (Gas) viscosity distribution in the reservoir in (cP)."""
    gas_compressibility_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas compressibility distribution in the reservoir (psi⁻¹)."""
    gas_gravity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas gravity distribution in the reservoir (dimensionless). Should be constant for a given gas, e.g., Methane = 0.556)."""
    gas_molecular_weight_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas molecular weight distribution in the reservoir (g/mol). Should be constant for a given gas, e.g., Methane = 16.04 g/mol)."""
    gas_density_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas density distribution in the reservoir (lbm/ft³)."""
    gas_to_oil_ratio_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas-to-oil ratio distribution at standard conditions (SCF/STB)."""
    oil_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil formation volume factor distribution (bbl/STB)."""
    gas_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas formation volume factor distribution (ft³/SCF)."""
    water_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water formation volume factor distribution (bbl/STB)."""
    water_salinity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water salinity distribution (ppm NaCl). Should be constant for a given water type, e.g., seawater = 35,000 ppm NaCl)."""


@define(slots=True, frozen=True)
class RockPermeability(typing.Generic[NDimension]):
    x: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the permeability distribution in the x-direction (mD)."""
    y: NDimensionalGrid[NDimension] = field(factory=lambda: np.empty((0, 0)))  # type: ignore[assignment]
    """N-dimensional numpy array representing the permeability distribution in the y-direction (mD)."""
    z: NDimensionalGrid[NDimension] = field(factory=lambda: np.empty((0, 0)))  # type: ignore[assignment]
    """N-dimensional numpy array representing the permeability distribution in the z-direction (mD)."""

    def __attrs_post_init__(self) -> None:
        if self.y.size == 0:
            object.__setattr__(self, "y", self.x)
        if self.z.size == 0:
            object.__setattr__(self, "z", self.x)


@define(slots=True, frozen=True)
class RockProperties(typing.Generic[NDimension]):
    """
    Rock properties of a reservoir model.

    These properties remain constant over time.
    """

    compressibility: float
    """Reservoir rock compressibility in (psi⁻¹)"""
    absolute_permeability: RockPermeability[NDimension]
    """Rock permeability in the reservoir, in milliDarcy (mD)."""
    net_to_gross_ratio_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the net-to-gross ratio distribution across the reservoir rock (fraction)."""
    porosity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the porosity distribution across the reservoir rock (fraction)."""
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the irreducible water saturation distribution (fraction)."""
    residual_oil_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the residual oil saturation distribution (fraction)."""
    residual_gas_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the residual gas saturation distribution (fraction)."""
    relative_permeability_params: RelativePermeabilityParameters = field(
        factory=RelativePermeabilityParameters
    )
    """Parameters for relative permeability curves."""
    capillary_pressure_params: CapillaryPressureParameters = field(
        factory=CapillaryPressureParameters
    )
    """Parameters for capillary pressure curve."""


@define(slots=True, frozen=True)
class ReservoirModel(typing.Generic[NDimension]):
    """Models a reservoir in N-dimensional space for simulation."""

    grid_dimension: NDimension
    """Number of cells in the grid. A tuple of number of cells in x and y directions (cell_count_x, cell_count_y)."""
    cell_dimension: typing.Tuple[float, float]
    """Size of each cell in the grid (cell_size_x, cell_size_y) in ft."""
    thickness_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the thickness of each cell in the reservoir (ft)."""
    fluid_properties: FluidProperties[NDimension]
    """Fluid properties of the reservoir model."""
    rock_properties: RockProperties[NDimension]
    """Rock properties of the reservoir model."""
    boundary_conditions: BoundaryConditions = field(factory=BoundaryConditions)
    """Boundary conditions for the simulation (e.g., no-flow, constant pressure)."""

    def get_elevation_grid(
        self, direction: typing.Literal["downward", "upward"] = "downward"
    ) -> NDimensionalGrid[NDimension]:
        """
        Generate an elevation grid based on the thickness grid and specified direction.

        The elevation grid is generated based on the thickness of each cell, starting from the top or bottom
        of the reservoir, depending on the specified direction.

        :param direction: Direction to generate the elevation grid. Can be "downward" (from top to bottom) or "upward" (from bottom to top).
        :return: N-dimensional numpy array representing the elevation of each cell in the reservoir (ft).
        """
        return generate_elevation_grid(self.thickness_grid, direction)


def generate_elevation_grid(
    thickness_grid: NDimensionalGrid[NDimension],
    direction: typing.Literal["downward", "upward"] = "downward",
) -> NDimensionalGrid[NDimension]:
    """
    Convert a cell thickness (height) grid into an absolute elevation grid (cell center z-coordinates).

    The elevation grid is generated based on the thickness of each cell, starting from the top or bottom
    of the reservoir, depending on the specified direction.

    :param thickness_grid: N-dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :param direction: Direction to generate the elevation grid. Can be "downward" (from top to bottom) or "upward" (from bottom to top).
    :return: N-dimensional numpy array representing the elevation of each cell in the reservoir (ft).
    """
    if direction not in {"downward", "upward"}:
        raise ValueError("direction must be 'downward' or 'upward'")

    nz, _, _ = thickness_grid.shape
    elevation_grid = np.zeros_like(thickness_grid, dtype=float)

    if direction == "downward":
        # Start from top layer, move downward
        elevation_grid[0] = thickness_grid[0] / 2
        for k in range(1, nz):
            elevation_grid[k] = (
                elevation_grid[k - 1]
                + thickness_grid[k - 1] / 2
                + thickness_grid[k] / 2
            )
    else:
        # Start from bottom layer, move upward
        elevation_grid[-1] = thickness_grid[-1] / 2
        for k in range(nz - 2, -1, -1):
            elevation_grid[k] = (
                elevation_grid[k + 1]
                + thickness_grid[k + 1] / 2
                + thickness_grid[k] / 2
            )

    return typing.cast(NDimensionalGrid[NDimension], elevation_grid)
