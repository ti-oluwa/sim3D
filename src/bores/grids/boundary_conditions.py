import logging
import typing

from bores._precision import get_dtype
from bores.boundary_conditions import BoundaryConditions, BoundaryMetadata, default_bc
from bores.models import FluidProperties, RockProperties
from bores.types import NDimension, NDimensionalGrid, ThreeDimensions

logger = logging.getLogger(__name__)


def mirror_neighbour_cells(
    grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """Mirrors the neighbour cells for boundary padding."""
    default_bc.apply(grid)
    return grid


def apply_boundary_conditions(
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
) -> typing.Tuple[FluidProperties, RockProperties]:
    """
    Applies boundary conditions to the fluid property grids.

    :param fluid_properties: The padded fluid properties.
    :param rock_properties: The padded rock properties.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param time: The current simulation time.
    """
    boundary_conditions["pressure"].apply(
        fluid_properties.pressure_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="pressure",
        ),
    )
    boundary_conditions["oil_saturation"].apply(
        fluid_properties.oil_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="oil_saturation",
        ),
    )
    boundary_conditions["water_saturation"].apply(
        fluid_properties.water_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="water_saturation",
        ),
    )
    boundary_conditions["gas_saturation"].apply(
        fluid_properties.gas_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="gas_saturation",
        ),
    )
    boundary_conditions["temperature"].apply(
        fluid_properties.temperature_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="temperature",
        ),
    )
    # Clamp saturations to [0, 1] after applying BCs
    dtype = get_dtype()
    fluid_properties.oil_saturation_grid.clip(
        min=0.0, max=1.0, out=fluid_properties.oil_saturation_grid, dtype=dtype
    )
    fluid_properties.water_saturation_grid.clip(
        min=0.0, max=1.0, out=fluid_properties.water_saturation_grid, dtype=dtype
    )
    fluid_properties.gas_saturation_grid.clip(
        min=0.0, max=1.0, out=fluid_properties.gas_saturation_grid, dtype=dtype
    )
    excluded_fluid_properties = (
        "pressure_grid",
        "oil_saturation_grid",
        "water_saturation_grid",
        "gas_saturation_grid",
        "temperature_grid",
    )
    fluid_properties = fluid_properties.apply_hook(
        hook=mirror_neighbour_cells, exclude=excluded_fluid_properties
    )
    rock_properties = rock_properties.apply_hook(hook=mirror_neighbour_cells)
    return fluid_properties, rock_properties
