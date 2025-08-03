"""Utils for defining boundary conditions for a N-Dimensional reservoir model grid."""

import typing
from attrs import define, field
from collections import defaultdict
import numpy as np

from sim3D.types import NDimension, NDimensionalGrid


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
        boundary_grid: NDimensionalGrid[NDimension],
        neighboring_grid: typing.Optional[NDimensionalGrid[NDimension]] = ...,
    ) -> None: ...


class NoFlowBoundary(typing.Generic[NDimension]):
    """Implements a no-flow boundary condition."""

    def apply(
        self,
        *,
        boundary_grid: NDimensionalGrid[NDimension],
        neighboring_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    ) -> None:
        boundary_grid[:] = (
            neighboring_grid.copy() if neighboring_grid is not None else 0.0
        )


@define(slots=True, frozen=True)
class ConstantBoundary(typing.Generic[NDimension]):
    """
    Implements a constant boundary condition.
    """

    constant: typing.Any

    def apply(
        self,
        *,
        boundary_grid: NDimensionalGrid[NDimension],
        neighboring_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    ) -> None:
        boundary_grid[:] = np.full_like(
            boundary_grid, self.constant, dtype=boundary_grid.dtype
        )


@define(slots=True, frozen=True)
class VariableBoundary(typing.Generic[NDimension]):
    """Implements a variable boundary condition using a callable function."""

    func: typing.Callable[
        [NDimensionalGrid[NDimension], typing.Optional[NDimensionalGrid[NDimension]]],
        NDimensionalGrid[NDimension],
    ]
    """Function to compute boundary values based on the grid and an optional neighboring grid."""

    def __attrs_post_init__(self) -> None:
        if not callable(self.func):
            raise ValueError("func must be a callable function.")

    def apply(
        self,
        boundary_grid: NDimensionalGrid[NDimension],
        neighboring_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    ) -> None:
        vectorized_func = np.vectorize(self.func, otypes=[boundary_grid.dtype])
        boundary_grid[:] = vectorized_func(boundary_grid, neighboring_grid)


@define(slots=True, frozen=True)
class GridBoundaryCondition(typing.Generic[NDimension]):
    """
    Container for defining boundary conditions for a grid.
    Each face in a 3D or 2D grid (x-, x+, y-, y+, z-, z+) can have its own boundary condition.

    In 2D:
    - Only x- (left/west), x+ (right/east), y- (bottom/south), y+ (top/north) are applied.

    In 3D:
    - All six faces are applied.

    ```python
                            z+
                ↑
               ┌───────────────┐
              /|              /|
             / |             / |
            /  |            /  |
        y+ /   |           /   |  x+
          ┌───────────────┐    |
          |   |           |    |
          |   |           |    |
          |   |           |    |
          |   └───────────|────┘
          |  /            |  /
          | /             | /
          |/              |/
          └───────────────┘
          ↑               ↑
          z-              y-
        (bottom)        (front)
    ```

    Left face  → x-
    Right face → x+
    Front face → y-
    Back face  → y+
    Bottom     → z-
    Top        → z+

    Defaults to no-flow boundary for all sides if not specified.
    """

    x_minus: BoundaryCondition = field(factory=NoFlowBoundary)
    """Boundary condition for the left face (x-)."""
    x_plus: BoundaryCondition = field(factory=NoFlowBoundary)
    """Boundary condition for the right face (x+)."""
    y_minus: BoundaryCondition = field(factory=NoFlowBoundary)
    """Boundary condition for the bottom face (y-)."""
    y_plus: BoundaryCondition = field(factory=NoFlowBoundary)
    """Boundary condition for the top face (y+)."""
    z_minus: BoundaryCondition = field(factory=NoFlowBoundary)
    """Boundary condition for the front face (z-)."""
    z_plus: BoundaryCondition = field(factory=NoFlowBoundary)
    """Boundary condition for the back face (z+)."""

    def apply(self, padded_grid: NDimensionalGrid[NDimension]) -> None:
        """
        Applies each defined boundary condition to the padded grid with ghost cells.

        - For 2D grids (shape: [nx+2, ny+2]), x and y boundaries are applied.
        - For 3D grids (shape: [nx+2, ny+2, nz+2]), x, y, and z boundaries are applied.
        """
        if padded_grid.ndim == 2:
            self.x_minus.apply(
                boundary_grid=padded_grid[0, :],
                neighboring_grid=padded_grid[1, :],
            )
            self.x_plus.apply(
                boundary_grid=padded_grid[-1, :],
                neighboring_grid=padded_grid[-2, :],
            )
            self.y_minus.apply(
                boundary_grid=padded_grid[:, 0],
                neighboring_grid=padded_grid[:, 1],
            )
            self.y_plus.apply(
                boundary_grid=padded_grid[:, -1],
                neighboring_grid=padded_grid[:, -2],
            )
        elif padded_grid.ndim == 3:
            self.x_minus.apply(
                boundary_grid=padded_grid[0, :, :],
                neighboring_grid=padded_grid[1, :, :],
            )
            self.x_plus.apply(
                boundary_grid=padded_grid[-1, :, :],
                neighboring_grid=padded_grid[-2, :, :],
            )
            self.y_minus.apply(
                boundary_grid=padded_grid[:, 0, :],
                neighboring_grid=padded_grid[:, 1, :],
            )
            self.y_plus.apply(
                boundary_grid=padded_grid[:, -1, :],
                neighboring_grid=padded_grid[:, -2, :],
            )
            self.z_minus.apply(
                boundary_grid=padded_grid[:, :, 0],
                neighboring_grid=padded_grid[:, :, 1],
            )
            self.z_plus.apply(
                boundary_grid=padded_grid[:, :, -1],
                neighboring_grid=padded_grid[:, :, -2],
            )
        else:
            raise ValueError(
                "`padded_grid` must be a 2D or 3D numpy array with ghost cells."
            )


class BoundaryConditions(defaultdict[str, GridBoundaryCondition[NDimension]]):
    """
    A dictionary-like container for managing reservoir model boundary conditions for different properties.

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
        factory=lambda: GridBoundaryCondition(
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
    # Uses default boundary conditions returned by `factory`
    temperature_bc = boundary_conditions["temperature"]

    pressure_bc.apply(padded_pressure_grid)
    temperature_bc.apply(padded_temperature_grid)
    ```
    """

    def __init__(
        self,
        conditions: typing.Optional[
            typing.Mapping[str, GridBoundaryCondition[NDimension]]
        ] = None,
        factory: typing.Optional[
            typing.Callable[[], GridBoundaryCondition[NDimension]]
        ] = GridBoundaryCondition[NDimension],
    ) -> None:
        """
        Initializes the `BoundaryConditions`.

        :param factory: Optional callable to provide default boundary conditions.
            If not provided, defaults to NoFlowBoundary for all sides.

        :param conditions: Optional mapping of property names to their respective boundary conditions.
        """
        super().__init__(factory, conditions or {})
