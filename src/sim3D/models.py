"""Static data models and schemas for an N-dimensional reservoir model."""

import typing

import attrs
import numpy as np
from typing_extensions import Self

from sim3D.boundaries import BoundaryConditions
from sim3D.grids.base import (
    apply_structural_dip,
    build_depth_grid,
    build_elevation_grid,
)
from sim3D.types import (
    CapillaryPressureTable,
    NDimension,
    NDimensionalGrid,
    PadMixin,
    RelativePermeabilityTable,
)


__all__ = ["FluidProperties", "RockProperties", "ReservoirModel", "RockPermeability"]


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
    - Gas solubility in water
    - Solvent concentration in oil phase
    - Effective oil-solvent mixture viscosity
    - Effective oil-solvent mixture density

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
    oil_effective_density_grid: NDimensionalGrid[NDimension]
    """
    Effective oil-solvent mixture density using miscible model (lbm/ft³).

    This will be same as `oil_density_grid` for immiscible flow.
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

    relative_permeability_table: RelativePermeabilityTable
    """Callable that evaluates the relative permeability curves based on fluid saturations."""
    capillary_pressure_table: CapillaryPressureTable
    """Callable that evaluates the capillary pressure curves based on fluid saturations."""


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
    dip_angle: float = attrs.field(
        default=0.0,
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0), attrs.validators.le(90.0)
        ),
    )
    """Dip angle of the reservoir in degrees (0 = horizontal, 90 = vertical)."""
    dip_azimuth: float = attrs.field(
        default=0.0,
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0), attrs.validators.lt(360.0)
        ),
    )
    """Dip azimuth of the reservoir in degrees (0 = North, 90 = East, 180 = South, 270 = West)."""

    @property
    def dimensions(self) -> int:
        """Return the number of dimensions of the reservoir model."""
        return len(self.grid_shape)

    @property
    def volume(self) -> float:
        """Return the total volume of the reservoir model."""
        return (
            np.prod(self.grid_shape)
            * np.prod(self.cell_dimension)
            * self.thickness_grid.sum()
        )

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
            dip_azimuth=self.dip_azimuth,
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
            dip_azimuth=self.dip_azimuth,
        )
