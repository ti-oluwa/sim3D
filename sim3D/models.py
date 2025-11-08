"""Static data models and schemas for the N-dimensional reservoir."""

import typing
from typing_extensions import Self

import attrs
import numpy as np

from sim3D.boundaries import BoundaryConditions
from sim3D.types import (
    NDimension,
    NDimensionalGrid,
    WettabilityType,
    RelativePermeabilityFunc,
)
from sim3D.grids.base import (
    build_elevation_grid,
    build_depth_grid,
    apply_structural_dip,
    pad_grid,
    unpad_grid,
)


__all__ = [
    "CapillaryPressureParameters",
    "FluidProperties",
    "RockProperties",
    "ReservoirModel",
    "RockPermeability",
]


class PadMixin(typing.Generic[NDimension]):
    """Mixin class to add padding functionality to attrs classes with numpy array fields."""

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        """Return iterable of attrs fields that can be padded."""
        raise NotImplementedError

    def pad(
        self,
        pad_width: int,
        hook: typing.Optional[
            typing.Callable[
                [NDimensionalGrid[NDimension]], NDimensionalGrid[NDimension]
            ]
        ] = None,
        exclude: typing.Optional[typing.Iterable[str]] = None,
    ) -> Self:
        """
        Pad all numpy array fields in the attrs class.

        :param pad_width: Number of cells to pad on each side of each dimension.
        :param hook: Optional callable to apply additional processing to each padded grid.
        :param exclude: Optional iterable of field names to exclude from hooking.
        :return: New instance of the attrs class with padded numpy array fields.
        """
        if not attrs.has(type(self)):
            raise TypeError(
                f"{self.__class__.__name__} can only be used with attrs classes"
            )

        target_fields = self.get_paddable_fields()
        padded_fields = {}
        for field in target_fields:
            value = getattr(self, field.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Field '{field.name}' is not a numpy array and cannot be padded"
                )
            padded_value = pad_grid(grid=value, pad_width=pad_width)
            if hook and (not exclude or field.name not in exclude):
                padded_value = hook(padded_value)
            padded_fields[field.name] = padded_value
        return attrs.evolve(self, **padded_fields)

    def unpad(self, pad_width: int) -> Self:
        """
        Remove padding from all numpy array fields in the attrs class.

        :param pad_width: Number of cells to remove from each side of each dimension.
        :return: New instance of the attrs class with unpadded numpy array fields.
        """
        if not attrs.has(type(self)):
            raise TypeError(
                f"{self.__class__.__name__} can only be used with attrs classes"
            )

        target_fields = self.get_paddable_fields()
        unpadded_fields = {}
        for field in target_fields:
            value = getattr(self, field.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Field '{field.name}' is not a numpy array and cannot be padded"
                )
            padded_value = unpad_grid(grid=value, pad_width=pad_width)
            unpadded_fields[field.name] = padded_value
        return attrs.evolve(self, **unpadded_fields)

    def apply_hook(
        self,
        hook: typing.Callable[
            [NDimensionalGrid[NDimension]], NDimensionalGrid[NDimension]
        ],
        exclude: typing.Optional[typing.Iterable[str]] = None,
    ) -> Self:
        """
        Apply a hook function to all numpy array fields in the attrs class.

        :param hook: Callable to apply to each numpy array field.
        :param exclude: Optional iterable of field names to exclude from hooking.
        :return: New instance of the attrs class with hooked numpy array fields.
        """
        if not attrs.has(type(self)):
            raise TypeError(
                f"{self.__class__.__name__} can only be used with attrs classes"
            )

        target_fields = self.get_paddable_fields()
        hooked_fields = {}
        for field in target_fields:
            if exclude and field.name in exclude:
                continue
            value = getattr(self, field.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Field '{field.name}' is not a numpy array and cannot be padded"
                )
            hooked_value = hook(value)
            hooked_fields[field.name] = hooked_value
        return attrs.evolve(self, **hooked_fields)


@attrs.define(slots=True, frozen=True)
class CapillaryPressureParameters:
    """Parameters defining the capillary pressure curve (e.g., Brooks-Corey)."""

    wettability: WettabilityType = WettabilityType.WATER_WET
    """
    Wettability type of the reservoir rock.
    
    Typical values:
    - WATER_WET: Most sandstones, clean carbonates (60-70% of reservoirs)
    - MIXED_WET: Aged reservoirs, some carbonates (20-30% of reservoirs)
    - OIL_WET: Rare, some carbonates, heavy oil reservoirs (5-10% of reservoirs)

    This determines how fluids interact with the rock surface, affecting capillary pressure.
    """
    oil_water_entry_pressure_oil_wet: float = 0.5
    """
    Oil-water entry pressure (psi) for water-wet rock.
    
    Typical ranges by rock type:
    - High permeability sandstone (>500 mD): 0.1-0.5 psi
    - Medium permeability sandstone (100-500 mD): 0.5-2.0 psi
    - Low permeability sandstone (10-100 mD): 2.0-10.0 psi
    - Tight sandstone (<10 mD): 10.0-50.0 psi
    - High permeability carbonate: 0.2-1.0 psi
    - Low permeability carbonate: 5.0-20.0 psi
    
    Default: 0.5 psi (good quality sandstone)

    It is the pressure (psi) at which oil starts to displace water in an oil-wet reservoir.

    This is the entry pressure for oil in an oil-wet system.
    Large entry pressure means stronger capillary forces are needed for oil to enter the pores.
    """
    oil_water_pore_size_distribution_index_water_wet: float = 2.0
    """
    Pore size distribution index (λ) for water-wet rock.
    
    Physical meaning:
    - Higher λ (3.0-5.0): Uniform pore sizes, sharp capillary pressure transition
    - Medium λ (2.0-3.0): Moderate pore size variation (most common)
    - Lower λ (1.0-2.0): Wide pore size distribution, gradual transition
    
    Typical ranges:
    - Well-sorted sandstone: 2.5-4.0
    - Moderately sorted sandstone: 2.0-2.5
    - Poorly sorted sandstone: 1.5-2.0
    - Vuggy carbonate: 1.0-2.0
    - Tight rock: 1.5-2.5
    
    Default: 2.0 (moderate sorting, typical sandstone)

    This parameter characterizes the distribution of pore sizes in the rock that affects capillary pressure.
    The value is typically higher than that for oil-wet reservoirs, indicating a different pore structure.
    The higher the value, the more uniform the pore sizes are, which affects how fluids interact with the rock.
    """
    oil_water_entry_pressure_water_wet: float = 0.3
    """
    Oil-water entry pressure (psi) for oil-wet rock.
    
    Note: Oil-wet systems typically have LOWER entry pressures than water-wet
    because oil preferentially wets the rock surface.
    
    Typical ranges:
    - High permeability: 0.1-0.3 psi
    - Medium permeability: 0.3-1.0 psi
    - Low permeability: 1.0-5.0 psi
    
    Default: 0.3 psi (30-50% lower than water-wet equivalent)

    It is the pressure (psi) at which oil starts to displace water in a water-wet reservoir.

    This is the entry pressure for oil in a water-wet system.
    Large entry pressure means stronger capillary forces are needed for oil to enter the pores.
    """
    oil_water_pore_size_distribution_index_oil_wet: float = 1.8
    """
    Pore size distribution index for oil-wet rock.
    
    Note: Oil-wet systems typically have LOWER λ than water-wet
    because wettability alteration often occurs in larger pores first.
    
    Typical ranges:
    - Well-sorted: 2.0-3.0
    - Moderately sorted: 1.5-2.0
    - Poorly sorted: 1.0-1.5
    
    Default: 1.8 (slightly lower than water-wet)

    This parameter characterizes the distribution of pore sizes in the rock that affects capillary pressure.
    The value is typically lower than that for water-wet reservoirs, indicating a different pore structure.
    The lower the value, the more varied the pore sizes are, which affects how fluids interact with the rock.
    """
    gas_oil_entry_pressure: float = 0.2
    """
    Gas-oil entry pressure (psi).
    
    Gas-oil capillary pressure is typically MUCH LOWER than oil-water
    because:
    - Lower interfacial tension (gas-oil: ~20-30 dyne/cm vs oil-water: ~30-50 dyne/cm)
    - Gas is non-wetting phase in both water-wet and oil-wet systems
    
    Typical ranges by rock type:
    - High permeability (>500 mD): 0.05-0.2 psi
    - Medium permeability (100-500 mD): 0.2-1.0 psi
    - Low permeability (10-100 mD): 1.0-5.0 psi
    - Tight rock (<10 mD): 5.0-20.0 psi
    
    Rule of thumb: P_entry(gas-oil) ≈ 0.3-0.5 x P_entry(oil-water)
    
    Default: 0.2 psi (good quality reservoir)

    This is the pressure (psi) at which gas starts to displace oil in the reservoir in a oil-wet or water-wet system.

    This is the entry pressure for gas in an oil system.
    Large entry pressure means stronger capillary forces are needed for gas to enter the pores.
    """
    gas_oil_pore_size_distribution_index: float = 2.0
    """
    Pore size distribution index for gas-oil system.
    
    Typically similar to oil-water λ since it reflects the same pore structure.
    May be slightly lower due to gas accessing smaller pores more easily.
    
    Typical ranges:
    - Well-sorted: 2.0-3.5
    - Moderately sorted: 1.8-2.5
    - Poorly sorted: 1.5-2.0
    
    Default: 2.0 (moderate sorting)

    This parameter characterizes the distribution of pore sizes in the rock that affects capillary pressure.
    The value is typically lower than that for water-wet reservoirs, indicating a different pore structure.
    The lower the value, the more varied the pore sizes are, which affects how fluids interact with the rock.
    """
    mixed_wet_water_fraction: float = 0.5
    """
    Fraction of pore space that is water-wet in mixed-wet systems (0 to 1).
    
    Physical interpretation:
    - 0.0: Fully oil-wet (all pores prefer oil contact)
    - 0.3: Predominantly oil-wet (fractured carbonates)
    - 0.5: Neutral mixed-wet (equal water/oil preference) - DEFAULT
    - 0.7: Predominantly water-wet (aged sandstones)
    - 1.0: Fully water-wet (clean sandstones)
    
    Typical values by reservoir type:
    - Fresh sandstone: 0.9-1.0 (strongly water-wet)
    - Aged sandstone with some oil: 0.6-0.8 (moderately water-wet)
    - Carbonate with oil aging: 0.3-0.6 (mixed-wet)
    - Heavy oil carbonate: 0.1-0.3 (predominantly oil-wet)
    
    Effect on capillary pressure:
    - Higher fraction → stronger water imbibition
    - Lower fraction → weaker water imbibition, may even repel water
    - 0.5 → balanced system with minimal net capillary drive

    Default: 0.5 (neutral mixed-wet)
    """


@attrs.define(slots=True, frozen=True)
class FluidProperties(PadMixin[NDimension]):
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
    solution_gas_to_oil_ratio_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the solution gas-to-oil ratio distribution at standard conditions (SCF/STB)."""
    gas_solubility_in_water_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas solubility in water distribution at standard conditions (SCF/STB)."""
    oil_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the oil formation volume factor distribution (bbl/STB)."""
    gas_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the gas formation volume factor distribution (ft³/SCF)."""
    water_formation_volume_factor_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water formation volume factor distribution (bbl/STB)."""
    water_salinity_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the water salinity distribution (ppm NaCl). Should be constant for a given water type, e.g., seawater = 35,000 ppm NaCl)."""
    solvent_concentration_grid: NDimensionalGrid[NDimension]
    """Solvent concentration in oil phase (0=pure oil, 1=pure solvent)"""
    oil_effective_viscosity_grid: NDimensionalGrid[NDimension]
    """
    Effective oil-solvent mixture viscosity using miscible model (e.g Todd Longstaff) (cP).

    This will be same as `oil_viscosity_grid` for immiscible flow.
    """
    reservoir_gas: str = "Methane"
    """Name of the reservoir gas (e.g., Methane, Ethane, CO2, N2). Can also be the name of the gas injected into the reservoir."""

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        excluded_fields = ("reservoir_gas",)
        return (
            field
            for field in attrs.fields(type(self))
            if field.name not in excluded_fields
        )


@attrs.define(slots=True, frozen=True)
class RockPermeability(PadMixin[NDimension]):
    """
    Rock permeability in the reservoir, in milliDarcy (mD).

    Permeability can be anisotropic, meaning it can vary in different directions (x, y, z).
    If only the x-direction permeability is provided, it is assumed that the y and z directions have the same permeability (isotropic).
    """

    x: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the permeability distribution in the x-direction (mD)."""
    y: NDimensionalGrid[NDimension] = attrs.field(factory=lambda: np.empty((0, 0)))  # type: ignore[assignment]
    """N-dimensional numpy array representing the permeability distribution in the y-direction (mD)."""
    z: NDimensionalGrid[NDimension] = attrs.field(factory=lambda: np.empty((0, 0)))  # type: ignore[assignment]
    """N-dimensional numpy array representing the permeability distribution in the z-direction (mD)."""

    def __attrs_post_init__(self) -> None:
        if self.y.size == 0:
            object.__setattr__(self, "y", self.x)
        if self.z.size == 0:
            object.__setattr__(self, "z", self.x)

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        return attrs.fields(type(self))


@attrs.define(slots=True, frozen=True)
class RockProperties(PadMixin[NDimension]):
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
    connate_water_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the connate water saturation distribution (fraction)."""
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the irreducible water saturation distribution (fraction)."""
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the residual oil saturation distribution during water flooding (fraction)."""
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the residual oil saturation distribution during gas flooding (fraction)."""
    residual_gas_saturation_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the residual gas saturation distribution (fraction)."""

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        excluded_fields = ("compressibility", "absolute_permeability")
        return (
            field
            for field in attrs.fields(type(self))
            if field.name not in excluded_fields
        )

    def pad(
        self,
        pad_width: int,
        hook: typing.Optional[
            typing.Callable[
                [NDimensionalGrid[NDimension]], NDimensionalGrid[NDimension]
            ]
        ] = None,
        exclude: typing.Optional[typing.Iterable[str]] = None,
    ) -> Self:
        padded = super().pad(pad_width=pad_width, hook=hook, exclude=exclude)
        padded_absolute_permeability = self.absolute_permeability.pad(
            pad_width=pad_width, hook=hook, exclude=exclude
        )
        return attrs.evolve(padded, absolute_permeability=padded_absolute_permeability)


@attrs.define(slots=True, frozen=True)
class RockFluidProperties:
    """
    Combined rock and fluid properties of a reservoir model.
    These properties include both rock and fluid characteristics necessary for reservoir simulation.
    """

    relative_permeability_func: RelativePermeabilityFunc
    """Callable that evaluates the relative permeability curves based on fluid saturations."""
    capillary_pressure_params: CapillaryPressureParameters = attrs.field(
        factory=CapillaryPressureParameters
    )
    """Parameters for capillary pressure curve."""


@attrs.define(slots=True, frozen=True)
class ReservoirModel(typing.Generic[NDimension]):
    """Models a reservoir in N-dimensional space for simulation."""

    grid_shape: NDimension
    """Shape of the reservoir grid (num_cells_x, num_cells_y, num_cells_z)."""
    cell_dimension: typing.Tuple[float, float]
    """Size of each cell in the grid (cell_size_x, cell_size_y) in ft."""
    thickness_grid: NDimensionalGrid[NDimension]
    """N-dimensional numpy array representing the thickness of each cell in the reservoir (ft)."""
    fluid_properties: FluidProperties[NDimension]
    """Fluid properties of the reservoir model."""
    rock_properties: RockProperties[NDimension]
    """Rock properties of the reservoir model."""
    rock_fluid_properties: RockFluidProperties
    """Rock-fluid properties of the reservoir model."""
    boundary_conditions: BoundaryConditions = attrs.field(factory=BoundaryConditions)
    """Boundary conditions for the simulation (e.g., no-flow, constant pressure)."""
    dip_angle: float = 0.0
    """Dip angle of the reservoir in degrees (0 = horizontal, positive = dipping downward in dip_direction)."""
    dip_direction: typing.Literal["N", "S", "E", "W"] = "N"
    """
    Dip direction of the reservoir:
    - 'N': Reservoir dips toward North (positive y-direction)
    - 'S': Reservoir dips toward South (negative y-direction)
    - 'E': Reservoir dips toward East (positive x-direction)
    - 'W': Reservoir dips toward West (negative x-direction)
    """

    def get_elevation_grid(
        self, apply_dip: bool = False
    ) -> NDimensionalGrid[NDimension]:
        """
        Generate an elevation grid of the reservoir cells.

        The elevation grid is generated based on the thickness of each cell, starting from the base elevation (0 ft).

        :param apply_dip: If True, applies the reservoir dip angle and direction to create
            a tilted elevation grid. If False, generates a flat (horizontal) elevation grid.
        :return: N-dimensional numpy array representing the elevation of each cell in the reservoir (ft).

        Example:
        ```python
        # Flat reservoir
        elevation = model.get_elevation_grid(apply_dip=False)

        # Dipping reservoir (5° toward North)
        model = ReservoirModel(dip_angle=5.0, dip_direction="N", ...)
        elevation = model.get_elevation_grid(apply_dip=True)
        ```
        """
        base_elevation_grid = build_elevation_grid(self.thickness_grid)
        # If no dip is requested or dip angle is zero, return flat grid
        if not apply_dip or self.dip_angle == 0.0:
            return base_elevation_grid

        return apply_structural_dip(
            elevation_grid=base_elevation_grid,
            cell_dimension=self.cell_dimension,
            elevation_direction="upward",
            dip_angle=self.dip_angle,
            dip_direction=self.dip_direction,
        )

    def get_depth_grid(self, apply_dip: bool = False) -> NDimensionalGrid[NDimension]:
        """
        Generate a depth grid of the reservoir cells.

        The depth grid is generated based on the thickness of each cell, starting from the surface (0 ft).

        :param apply_dip: If True, applies the reservoir dip angle and direction to create
            a tilted depth grid. If False, generates a flat (horizontal) depth grid.
        :return: N-dimensional numpy array representing the depth of each cell in the reservoir (ft).

        Example:
        ```python
        # Flat reservoir
        depth = model.get_depth_grid(apply_dip=False)

        # Dipping reservoir (5° toward North)
        model = ReservoirModel(dip_angle=5.0, dip_direction="N", ...)
        depth = model.get_depth_grid(apply_dip=True)
        ```
        """
        base_depth_grid = build_depth_grid(self.thickness_grid)
        # If no dip is requested or dip angle is zero, return flat grid
        if not apply_dip or self.dip_angle == 0.0:
            return base_depth_grid

        return apply_structural_dip(
            elevation_grid=base_depth_grid,
            cell_dimension=self.cell_dimension,
            elevation_direction="downward",
            dip_angle=self.dip_angle,
            dip_direction=self.dip_direction,
        )
