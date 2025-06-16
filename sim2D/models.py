import typing
from dataclasses import dataclass

from sim2D.typing import TwoDimensionalGrid


@dataclass(slots=True, frozen=True)
class FluidProperties:
    """
    Represents the fluid properties of the reservoir.

    These properties are liable to change over time due to flow.
    """

    pressure_grid: TwoDimensionalGrid
    """2D numpy array representing the pressure distribution in the reservoir (Pa)."""
    fluid_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the reservoir fluid (Oil) saturation distribution in the reservoir (fraction)."""
    fluid_viscosity_grid: TwoDimensionalGrid
    """2D numpy array representing the reservoir fluid (Oil) viscosity distribution in the reservoir in Pa.s."""


@dataclass(slots=True, frozen=True)
class RockProperties:
    """
    Represents the rock properties of the reservoir.

    These properties remain constant over time.
    """

    compressibility: float
    """Reservoir rock compressibility in (1/Pa)"""
    permeability_grid: TwoDimensionalGrid
    """2D numpy array representing the permeability distribution across the reservoir rock (mD)."""
    porosity_grid: TwoDimensionalGrid
    """2D numpy array representing the porosity distribution across the reservoir rock (fraction)."""


@dataclass(slots=True, frozen=True)
class TwoDimensionalReservoirModel:
    """Represents a 2D reservoir model with its properties and state."""

    cell_dimension: typing.Tuple[float, float]
    """Size of each cell in the grid (cell_size_x, cell_size_y) in meters."""
    grid_dimension: typing.Tuple[int, int]
    """Number of cells in the grid. A tuple of number of cells in x and y directions (cell_count_x, cell_count_y)."""
    fluid_properties: FluidProperties
    """Fluid properties of the reservoir model."""
    rock_properties: RockProperties
    """Fluid properties of the reservoir model."""
    temperature_grid: TwoDimensionalGrid
    """2D numpy array representing the temperature distribution across the reservoir (K)."""


def build_2D_reservoir_model(
    grid_dimension: typing.Tuple[int, int],
    cell_dimension: typing.Tuple[float, float],
    pressure_grid: TwoDimensionalGrid,
    fluid_saturation_grid: TwoDimensionalGrid,
    fluid_viscosity_grid: TwoDimensionalGrid,
    permeability_grid: TwoDimensionalGrid,
    porosity_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    rock_compressibility: float,
) -> TwoDimensionalReservoirModel:
    """
    Constructs a 2D reservoir model with the given parameters.

    :param grid_dimension: Tuple of number of cells in x and y directions (cell_count_x, cell_count_y)
    :param cell_dimension: Tuple of size of each cell in x and y directions (cell_size_x, cell_size_y)
    :param pressure_grid: Initial reservoir pressure distribution as a 2D numpy array
    :param fluid_saturation_grid: Initial reservoir fluid saturation distribution as a 2D numpy array
    :param fluid_viscosity_grid: Initial reservoir fluid saturation distribution as a 2D numpy array
    :param permeability_grid: Reservoir permeability distribution as a 2D numpy array
    :param porosity_grid: Reservoir porosity distribution as a 2D numpy array
    :param temperature_grid: Reservoir temperature distribution as a 2D numpy array
    :param rock_compressibility: Rock compressibility (1/Pa)
    :return: TwoDimensionalReservoirModel instance
    """
    if pressure_grid.shape != grid_dimension:
        raise ValueError("Initial pressure shape does not match grid dimensions.")
    if fluid_saturation_grid.shape != grid_dimension:
        raise ValueError("Initial saturation shape does not match grid dimensions.")
    if fluid_viscosity_grid.shape != grid_dimension:
        raise ValueError("Initial saturation shape does not match grid dimensions.")
    if permeability_grid.shape != grid_dimension:
        raise ValueError("Permeability shape does not match grid dimensions.")
    if porosity_grid.shape != grid_dimension:
        raise ValueError("Porosity shape does not match grid dimensions.")
    if temperature_grid.shape != grid_dimension:
        raise ValueError("Temperature shape does not match grid dimensions.")
    if rock_compressibility <= 0:
        raise ValueError("Rock compressibility must be a positive value.")

    return TwoDimensionalReservoirModel(
        cell_dimension=cell_dimension,
        grid_dimension=grid_dimension,
        fluid_properties=FluidProperties(
            pressure_grid=pressure_grid,
            fluid_saturation_grid=fluid_saturation_grid,
            fluid_viscosity_grid=fluid_viscosity_grid,
        ),
        rock_properties=RockProperties(
            permeability_grid=permeability_grid,
            porosity_grid=porosity_grid,
            compressibility=rock_compressibility,
        ),
        temperature_grid=temperature_grid,
    )
