"""Utils for defining boundary conditions for a 2D reservoir model grid."""

import typing
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from _sim2D.types import TwoDimensionalGrid, OneDimensionalGrid


__all__ = [
    "NoFlowBoundary",
    "ConstantBoundary",
    "VariableBoundary",
    "GridBoundaryCondition",
    "BoundaryConditions",
]

class BoundaryCondition(typing.Protocol):
    """
    Protocol for defining boundary conditions.

    Each boundary condition type must implement an 'apply' method.
    """

    def apply(
        self,
        *,
        boundary_grid: OneDimensionalGrid,
        neighboring_grid: typing.Optional[OneDimensionalGrid] = ...,
    ) -> None:
        """
        Applies the boundary condition to the given boundary grid.

        Modifies the boundary grid in place.

        :param boundary_grid: 1D array representing the boundary grid, usually with ghost cells.
        :param neighboring_grid: Optional 1D array of neighboring grid values.
        """
        ...


class NoFlowBoundary:
    """Implements a no-flow boundary condition."""

    def apply(
        self,
        *,
        boundary_grid: OneDimensionalGrid,
        neighboring_grid: typing.Optional[OneDimensionalGrid] = None,
    ) -> None:
        """
        Applies a no-flow condition by setting the boundary values to zero.

        :param boundary_grid: The (pressure or saturation) grid at the boundary, usually with ghost cells.
        :param neighboring_grid: Optional 1D array of neighboring grid values.
        """
        boundary_grid[:] = (
            neighboring_grid.copy() if neighboring_grid is not None else 0.0
        )


@dataclass(slots=True, frozen=True)
class ConstantBoundary:
    """
    Implements a constant boundary condition.
    """

    constant: typing.Any
    """The constant value to apply at the boundary."""

    def apply(
        self,
        *,
        boundary_grid: OneDimensionalGrid,
        neighboring_grid: typing.Optional[OneDimensionalGrid] = None,
    ) -> None:
        """
        Applies a constant boundary condition.
        """
        boundary_grid[:] = np.full_like(
            boundary_grid, self.constant, dtype=boundary_grid.dtype
        )


@dataclass(slots=True, frozen=True)
class VariableBoundary:
    """Implements a variable boundary condition using a callable function."""

    func: typing.Callable[[typing.Any, typing.Optional[typing.Any]], typing.Any]
    """
    Takes (boundary_cell, neighboring_cell) from (boundary_grid, neighboring_grid) 
    and returns the new value for the boundary cell.

    The function should accept two parameters:
    - boundary_cell: The current value of the boundary cell.
    - neighboring_cell: Optional value of the neighboring cell (if applicable).
    The function should return the new value for the boundary cell.
    """

    def __post_init__(self):
        if not callable(self.func):
            raise ValueError("func must be a callable function.")

    def apply(
        self,
        boundary_grid: OneDimensionalGrid,
        neighboring_grid: typing.Optional[OneDimensionalGrid] = None,
    ) -> None:
        """
        Applies a variable boundary condition to the boundary grid using the provided function.

        :param boundary_grid: 1D array representing the boundary grid, usually with ghost cells.
            The function receives this as the first argument.

        :param neighboring_grid: Optional 1D array of neighboring grid values.
            If provided, the function will receive this as the second argument.
        """
        vectorized_func = np.vectorize(self.func, otypes=[boundary_grid.dtype])
        boundary_grid[:] = vectorized_func(boundary_grid, neighboring_grid)


@dataclass(slots=True, frozen=True)
class GridBoundaryCondition:
    """
    Container for defining boundary conditions for a grid.
    Each side (north, south, east, west) can have its own boundary condition.

    Defaults to no-flow boundary for all sides if not specified.
    """

    north: BoundaryCondition = field(default_factory=NoFlowBoundary)
    """Boundary condition for the northern edge of the grid."""
    south: BoundaryCondition = field(default_factory=NoFlowBoundary)
    """Boundary condition for the southern edge of the grid."""
    east: BoundaryCondition = field(default_factory=NoFlowBoundary)
    """Boundary condition for the eastern edge of the grid."""
    west: BoundaryCondition = field(default_factory=NoFlowBoundary)
    """Boundary condition for the western edge of the grid."""

    def apply(self, padded_grid: TwoDimensionalGrid) -> None:
        """
        Applies each defined boundary condition to the padded grid.

        A padded grid is one that includes ghost cells around the main grid.
        You can think of it as a grid with an extra row and column on each side,
        where the first row/column corresponds to the north/west boundary and
        the last row/column corresponds to the south/east boundary.

        Pad a grid by using numpy's `pad` function or similar methods to create a grid
        Example:
        ```
        import numpy as np
        original_grid = np.random.rand(50, 50)  # Example original grid

        # Pad the grid with ghost cells (1 cell wide on each side)
        padded_grid = np.pad(
            original_grid,
            pad_width=((1, 1), (1, 1)),  # Add one row/column on each side
            mode='edge'  # Use edge values for padding
        )
        ```
        This method modifies the padded grid in place by applying the boundary conditions

        :param padded_grid: 2D numpy array representing the padded grid with ghost cells.
            The shape should be (cell_count_x + 2, cell_count_y + 2).
        """
        if padded_grid.ndim != 2:
            raise ValueError(
                "padded_grid must be a 2D numpy array. Ensure it has a dimension of 2."
            )

        self.north.apply(
            boundary_grid=padded_grid[
                0, :
            ],  # North boundary (first row of padded grid)
            neighboring_grid=padded_grid[
                1, :
            ],  # Neighboring cells (second row, if exists, otherwise None)
        )
        self.south.apply(
            boundary_grid=padded_grid[
                -1, :
            ],  # South boundary (last row of padded grid)
            neighboring_grid=padded_grid[
                -2, :
            ],  # Neighboring cells (second last row, if exists, otherwise None)
        )
        self.east.apply(
            boundary_grid=padded_grid[
                :, -1
            ],  # East boundary (last column of padded grid)
            neighboring_grid=padded_grid[
                :, -2
            ],  # Neighboring cells (second last column, if exists, otherwise None)
        )
        self.west.apply(
            boundary_grid=padded_grid[
                :, 0
            ],  # West boundary (first column of padded grid)
            neighboring_grid=padded_grid[
                :, 1
            ],  # Neighboring cells (second column, if exists, otherwise None)
        )


class BoundaryConditions(defaultdict[str, GridBoundaryCondition]):
    """
    A dictionary-like container for managing boundary conditions for different properties.

    This class allows you to define boundary conditions for various properties
    in a two-dimensional grid, with a default factory to create conditions


    Example usage:
    ```python
    import numpy as np

    boundary_conditions = BoundaryConditions(
        conditions={
            "pressure": GridBoundaryCondition(
                north=ConstantBoundary(constant=2000.0),
                south=NoFlowBoundary(),
                east=VariableBoundary(func=lambda x, y: x * 1.2),
                west=NoFlowBoundary(),
            )
        },
        default_factory=lambda: GridBoundaryCondition(
            north=ConstantBoundary(constant=1000.0),
            south=NoFlowBoundary(),
            east=VariableBoundary(func=lambda x, y: x * 1.1),
            west=NoFlowBoundary(),
        ),
    )

    pressure_grid = np.full((52, 52), 1000.0)
    temperature_grid = np.full((52, 52), 300.0)
    padded_pressure_grid = np.pad(
        pressure_grid, pad_width=1, mode='edge'
    )
    padded_temperature_grid = np.pad(
        temperature_grid, pad_width=1, mode='edge'
    )
    pressure_bc = boundary_conditions["pressure"]
    temperature_bc = boundary_conditions["temperature"]  # Uses default boundary conditions returned by `default_factory`

    pressure_bc.apply(padded_pressure_grid)
    temperature_bc.apply(padded_temperature_grid)
    ```
    """

    def __init__(
        self,
        conditions: typing.Optional[typing.Mapping[str, GridBoundaryCondition]] = None,
        default_factory: typing.Optional[
            typing.Callable[[], GridBoundaryCondition]
        ] = GridBoundaryCondition,
    ) -> None:
        """
        Initializes the `BoundaryConditions`.

        :param default_factory: Optional callable to provide default boundary conditions.
            If not provided, defaults to NoFlowBoundary for all sides.

        :param conditions: Optional mapping of property names to their respective boundary conditions.
        """
        super().__init__(default_factory, conditions or {})
