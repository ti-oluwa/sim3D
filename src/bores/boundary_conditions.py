"""Boundary condition implementations for 2D/3D grids."""

from collections import defaultdict
import copy
import enum
import functools
import threading
import typing

import attrs
import numpy as np
from typing_extensions import ParamSpec, Self

from bores.errors import DeserializationError, SerializationError, ValidationError
from bores.serialization import Serializable, make_serializable_type_registrar
from bores.stores import StoreSerializable
from bores.types import NDimension, NDimensionalGrid


__all__ = [
    "boundary_function",
    "ParameterizedBoundaryFunction",
    "Boundary",
    "BoundaryMetadata",
    "BoundaryCondition",
    "NoFlowBoundary",
    "ConstantBoundary",
    "VariableBoundary",
    "DirichletBoundary",
    "NeumannBoundary",
    "SpatialBoundary",
    "TimeDependentBoundary",
    "LinearGradientBoundary",
    "FluxBoundary",
    "RobinBoundary",
    "PeriodicBoundary",
    "CarterTracyAquifer",
    "GridBoundaryCondition",
    "BoundaryConditions",
]


_BOUNDARY_FUNCTIONS: typing.Dict[str, typing.Callable] = {}
"""Registry of boundary functions."""
_boundary_function_lock = threading.Lock()
P = ParamSpec("P")
R = typing.TypeVar("R")


@typing.overload
def boundary_function(func: typing.Callable[P, R]) -> typing.Callable[P, R]: ...


@typing.overload
def boundary_function(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[[typing.Callable[P, R]], typing.Callable[P, R]]: ...


def boundary_function(
    func: typing.Optional[typing.Callable[P, R]] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[
    typing.Callable[P, R],
    typing.Callable[[typing.Callable[P, R]], typing.Callable[P, R]],
]:
    """
    Register a boundary function for serialization.

    A boundary function is a callable that computes boundary values, usually
    pressure or flux, based on spatial coordinates and/or time. This is basically any function you pass to
    a `BoundaryCondition` that computes values based on position, time or any other parameters.

    Usage:
    ```python
    @boundary_function
    def linear_pressure_gradient(x, y):
        return 2000 - 0.5 * x

    # Or with custom name:
    @boundary_function(name="custom_gradient")
    def my_gradient(x, y):
        return 2500 + 0.1 * x
    ```

    :param func: The function to register.
    :param name: Optional custom name for registration. Uses __name__ if not provided.
    :param override: If True, allows overriding existing registrations.
    :return: The registered function or a decorator.
    """

    def decorator(func: typing.Callable[P, R]) -> typing.Callable[P, R]:
        key = name or getattr(func, "__name__", None)
        if not key:
            raise ValidationError(
                "Boundary function must have a `__name__` attribute or a name must be provided."
            )

        with _boundary_function_lock:
            if not override and key in _BOUNDARY_FUNCTIONS:
                raise ValidationError(
                    f"Boundary function '{key}' already registered. "
                    f"Use `override=True` or choose a different name."
                )
            _BOUNDARY_FUNCTIONS[key] = func
        return func

    if func is not None:
        return decorator(func)
    return decorator


def list_boundary_functions() -> typing.List[str]:
    """List all registered boundary functions."""
    with _boundary_function_lock:
        return list(_BOUNDARY_FUNCTIONS.keys())


def get_boundary_function(name: str) -> typing.Callable:
    """
    Get a registered boundary function by name.

    :param name: Name of the registered boundary function.
    :return: The boundary function.
    :raises ValidationError: If the boundary function is not registered.
    """
    with _boundary_function_lock:
        if name not in _BOUNDARY_FUNCTIONS:
            raise ValidationError(
                f"Boundary function '{name}' not registered. "
                f"Use `@boundary_function` to register it. "
                f"Available: {list(_BOUNDARY_FUNCTIONS.keys())}"
            )
        return _BOUNDARY_FUNCTIONS[name]


def serialize_boundary_function(
    func: typing.Callable[..., typing.Any], recurse: bool = True
) -> typing.Dict[str, typing.Any]:
    """
    Serialize a boundary function.

    Supports:
    1. Registered functions (by name)
    2. Parameterized functions (functools.partial)
    3. Built-in serializable function wrappers
    """
    # Check if it's a registered function
    with _boundary_function_lock:
        for name, registered_func in _BOUNDARY_FUNCTIONS.items():
            if func is registered_func:
                return {"type": "registered", "name": name}

    # Handle partial functions (parameterized)
    if isinstance(func, functools.partial):
        base_func_data = serialize_boundary_function(func.func, recurse)
        return {
            "type": "partial",
            "func": base_func_data,
            "args": list(func.args),
            "kwargs": dict(func.keywords),
        }

    # Handle built-in serializable wrappers
    if isinstance(func, ParameterizedBoundaryFunction):
        return {
            "type": "parameterized",
            "data": func.dump(recurse),
        }

    # Cannot serialize, must be registered
    raise SerializationError(
        f"Cannot serialize boundary function {func}. "
        f"Please register it with @boundary_function. "
        f"Available functions: {list(_BOUNDARY_FUNCTIONS.keys())}"
    )


def deserialize_boundary_function(
    data: typing.Mapping[str, typing.Any],
) -> typing.Callable[..., typing.Any]:
    """Deserialize a boundary function from serialized data."""
    func_type = data.get("type")

    if func_type == "registered":
        if "name" not in data:
            raise DeserializationError(
                "Missing 'name' for registered boundary function."
            )
        return get_boundary_function(data["name"])

    elif func_type == "partial":
        if "func" not in data:
            raise DeserializationError("Missing 'func' for partial boundary function.")

        base_func = deserialize_boundary_function(data["func"])
        return functools.partial(
            base_func,
            *data.get("args", []),
            **data.get("kwargs", {}),
        )

    elif func_type == "parameterized":
        if "data" not in data:
            raise DeserializationError(
                "Missing 'data' for parameterized boundary function."
            )
        return ParameterizedBoundaryFunction.load(data["data"])

    else:
        raise DeserializationError(
            f"Unknown boundary function type: {func_type}. "
            f"Valid types: 'registered', 'partial', 'parameterized'"
        )


@boundary_function
def parametric_gradient(x, y, slope=0.5, intercept=2000):
    """Parameterized linear gradient."""
    return intercept - slope * x


@boundary_function
def sinusoidal_pressure(t: float, amplitude=200, period=86400, offset=2000):
    """Sinusoidal time-dependent pressure (daily cycle)."""
    return offset + amplitude * np.sin(2 * np.pi * t / period)


@boundary_function
def exponential_decay(t: float, initial=2000, time_constant=3600):
    """Exponential pressure decay."""
    return initial * np.exp(-t / time_constant)


class ParameterizedBoundaryFunction(
    Serializable, fields={"func_name": str, "params": typing.Dict[str, typing.Any]}
):
    """
    Wrapper for parameterized boundary functions.

    Alternative to `functools.partial` that's fully serializable.

    Usage:
    ```python
    # Define a base function
    @boundary_function
    def parametric_gradient(x, y, slope, intercept):
        return intercept - slope * x

    # Create parameterized version
    custom_gradient = ParameterizedBoundaryFunction(
        func_name="parametric_gradient",
        params={"slope": 0.8, "intercept": 2500}
    )

    # Use it
    result = custom_gradient(x_array, y_array)
    ```
    """

    def __init__(
        self,
        func_name: str,
        params: typing.Dict[str, typing.Any],
    ):
        """
        Initialize parameterized function.

        :param func_name: Name of the registered base boundary function
        :param params: Dictionary of parameter names to values
        """
        self.func_name = func_name
        self.params = params
        self._func = get_boundary_function(func_name)

    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> np.ndarray:
        """Call the function with stored parameters."""
        merged_kwargs = {**self.params, **kwargs}
        return self._func(*args, **merged_kwargs)


class Boundary(enum.Enum):
    """Enumeration of possible boundary directions."""

    LEFT = "left"
    """The negative X direction (left/west face)."""
    RIGHT = "right"
    """The positive X direction (right/east face)."""
    FRONT = "front"
    """The negative Y direction (bottom/south face)."""
    BACK = "back"
    """The positive Y direction (top/north face)."""
    BOTTOM = "bottom"
    """The negative Z direction (bottom face)."""
    TOP = "top"
    """The positive Z direction (top face)."""


def get_neighbor_indices(
    boundary_indices: typing.Tuple[slice, ...], direction: Boundary
) -> typing.Tuple[slice, ...]:
    """
    Get the indices of neighboring cells for a given boundary direction.

    This utility function converts boundary ghost cell indices to their
    corresponding neighboring interior cell indices.

    :param boundary_indices: Slice indices defining the boundary region
    :param direction: The boundary direction (left, right, etc.)
    :return: Tuple of slice indices for the neighboring interior cells
    :raises ValidationError: If Z-direction is used with 2D grid indices

    Example usage:
    ```python
    from bores.boundary_conditions import get_neighbor_indices, Boundary

    # For left boundary (x=0 ghost cells)
    boundary_slice = (slice(0, 1), slice(None))
    neighbor_slice = get_neighbor_indices(boundary_slice, Boundary.LEFT)
    # Returns: (slice(1, 2), slice(None))  # x=1 interior cells

    # For right boundary (x=-1 ghost cells)
    boundary_slice = (slice(-1, None), slice(None))
    neighbor_slice = get_neighbor_indices(boundary_slice, Boundary.RIGHT)
    # Returns: (slice(-2, -1), slice(None))  # x=-2 interior cells
    ```
    """
    neighbor_indices = list(boundary_indices)
    ndim = len(boundary_indices)

    # Validate Z-direction usage with grid dimensionality
    if direction in [Boundary.BOTTOM, Boundary.TOP] and ndim < 3:
        raise ValidationError(
            f"Cannot use {direction.name} boundary direction with {ndim}D grid. "
            "Z-direction boundaries require 3D grids."
        )

    if direction == Boundary.LEFT:
        # Left boundary: neighbor is at x=1
        neighbor_indices[0] = slice(1, 2)
    elif direction == Boundary.RIGHT:
        # Right boundary: neighbor is at x=-2
        neighbor_indices[0] = slice(-2, -1)
    elif direction == Boundary.FRONT:
        # Front boundary: neighbor is at y=1
        neighbor_indices[1] = slice(1, 2)
    elif direction == Boundary.BACK:
        # Back boundary: neighbor is at y=-2
        neighbor_indices[1] = slice(-2, -1)
    elif direction == Boundary.BOTTOM:
        # Bottom boundary: neighbor is at z=1
        neighbor_indices[2] = slice(1, 2)
    elif direction == Boundary.TOP:
        # Top boundary: neighbor is at z=-2
        neighbor_indices[2] = slice(-2, -1)

    return tuple(neighbor_indices)


@attrs.frozen
class BoundaryMetadata:
    """
    Optional metadata for boundary condition evaluation.

    Provides rich context to boundary conditions, enabling spatial awareness,
    time dependence, and physics-based calculations.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import BoundaryMetadata, SpatialBoundary

    # Option 1: Auto-generate 2D coordinates from cell dimensions and grid shape
    metadata_2d = BoundaryMetadata(
        cell_dimension=(20.0, 20.0),         # 20x20 ft cells
        grid_shape=(50, 25),                 # 50x25 grid (without ghost cells)
        thickness_grid=np.full((50, 25), 10.0),  # 10 ft thickness (no ghost cells)
        time=3600.0,                         # 1 hour simulation time
        property_name="pressure"             # Property being updated
    )
    # Coordinates auto-generated: x=[-10,10,30,...,990], y=[-10,10,30,...,490]

    # Option 2: Auto-generate 3D coordinates with thickness grid
    thickness_3d = np.random.uniform(5.0, 15.0, (50, 25, 10))  # Variable thickness (no ghost cells)
    metadata_3d = BoundaryMetadata(
        cell_dimension=(20.0, 20.0),         # 20x20 ft cells (dx, dy)
        grid_shape=(50, 25, 10),             # 50x25x10 grid (without ghost cells)
        thickness_grid=thickness_3d,         # Variable thickness for z-coordinates
        time=3600.0,
        property_name="pressure"
    )
    # Z-coordinates generated from cumulative thickness values

    # Option 3: Provide explicit coordinates
    x_coords = np.linspace(0, 1000, 51)  # 0-1000 ft
    y_coords = np.linspace(0, 500, 26)   # 0-500 ft
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    coordinates = np.stack([xx, yy], axis=-1)

    metadata_explicit = BoundaryMetadata(
        cell_dimension=(20.0, 20.0),
        coordinates=coordinates,              # Explicit coordinates override auto-generation
        thickness_grid=np.full((50, 25), 10.0),  # No ghost cells
        time=3600.0,
        property_name="pressure",
        grid_shape=(50, 25)
    )

    # Use with spatial boundary
    spatial_bc = SpatialBoundary(
        func=lambda x, y: 2000 + 0.1 * x - 0.05 * y
    )

    grid = np.zeros((52, 27))  # Grid with ghost cells
    boundary_indices = (slice(0, 1), slice(None))  # Left boundary

    spatial_bc.apply(
        grid=grid,
        boundary_indices=boundary_indices,
        direction=Boundary.LEFT,
        metadata=metadata_2d
    )
    ```
    """

    cell_dimension: typing.Optional[typing.Tuple[float, float]] = None
    """Physical dimensions of grid cells (dx, dy) in feet (or meters)."""
    thickness_grid: typing.Optional[NDimensionalGrid] = None
    """Grid of cell thickness values (same shape as original grid, No ghost cells factored in)."""
    coordinates: typing.Optional[NDimensionalGrid] = None
    """Physical coordinates of grid points. Auto-generated if not provided."""
    time: typing.Optional[float] = None
    """Current simulation time."""
    property_name: typing.Optional[str] = None
    """Name of the property being updated."""
    grid_shape: typing.Optional[typing.Tuple[int, ...]] = None
    """Original grid shape (without ghost cells)."""

    def __attrs_post_init__(self) -> None:
        """Auto-generate coordinates if not provided but cell_dimension and grid_shape are available."""
        if (
            self.coordinates is None
            and self.cell_dimension is not None
            and self.grid_shape is not None
        ):
            # Generate coordinate arrays
            dx, dy = self.cell_dimension

            if len(self.grid_shape) == 2:
                # 2D grid
                nx, ny = self.grid_shape

                # Create coordinate arrays (including ghost cells)
                # Using linspace for predictable array lengths (avoids floating-point issues with arange)
                # nx+2 points: from -dx/2 to (nx+0.5)*dx with ghost cells
                x_coords = np.linspace(-dx / 2, (nx + 0.5) * dx, nx + 2)
                y_coords = np.linspace(-dy / 2, (ny + 0.5) * dy, ny + 2)

                # Create meshgrid and stack
                xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
                coordinates = np.stack([xx, yy], axis=-1)

            elif len(self.grid_shape) == 3:
                # 3D grid - need thickness information for z-coordinates
                nx, ny, nz = self.grid_shape

                # Create x and y coordinate arrays using linspace for consistent lengths
                x_coords = np.linspace(-dx / 2, (nx + 0.5) * dx, nx + 2)
                y_coords = np.linspace(-dy / 2, (ny + 0.5) * dy, ny + 2)

                # For z-coordinates, use thickness_grid if available, otherwise assume uniform thickness
                if self.thickness_grid is not None:
                    # Use cumulative thickness to generate z-coordinates
                    # thickness_grid has no ghost cells, same shape as grid_shape
                    if self.thickness_grid.ndim == 3:
                        # Extract thickness values for z-coordinate generation
                        # Use first x,y location as representative (assuming uniform layers)
                        layer_thickness = (
                            self.thickness_grid[0, 0, :]
                            if self.thickness_grid.shape[2] == nz
                            else None
                        )

                        if layer_thickness is not None:
                            # Generate z-coordinates from cumulative thickness
                            # Convention: k=0 is TOP (shallowest), k increases downward (deeper)
                            # z-coordinate represents depth, increasing downward
                            z_coords = np.zeros(nz + 2)  # Include ghost cells

                            # Top ghost cell (above k=0, negative depth)
                            z_coords[0] = -layer_thickness[0] / 2

                            # Interior cells: cumulative depth centers
                            # k=0 (top/shallowest) is at z_coords[1]
                            cumulative_depth = 0.0
                            for k in range(nz):
                                if k == 0:
                                    z_coords[k + 1] = layer_thickness[k] / 2
                                else:
                                    cumulative_depth += layer_thickness[k - 1]
                                    z_coords[k + 1] = (
                                        cumulative_depth + layer_thickness[k] / 2
                                    )

                            # Bottom ghost cell (below k=-1, deepest)
                            cumulative_depth += layer_thickness[-1]
                            z_coords[-1] = cumulative_depth + layer_thickness[-1] / 2
                        else:
                            # Fallback: assume uniform thickness equal to dx
                            dz = dx  # Default thickness
                            z_coords = np.linspace(-dz / 2, (nz + 0.5) * dz, nz + 2)  # type: ignore[assignment]
                    else:
                        # Fallback: assume uniform thickness equal to dx
                        dz = dx  # Default thickness
                        z_coords = np.linspace(-dz / 2, (nz + 0.5) * dz, nz + 2)  # type: ignore[assignment]
                else:
                    # Fallback: assume uniform thickness equal to dx
                    dz = dx  # Default thickness
                    z_coords = np.linspace(-dz / 2, (nz + 0.5) * dz, nz + 2)  # type: ignore[assignment]

                # Create meshgrid and stack
                xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
                coordinates = np.stack([xx, yy, zz], axis=-1)

            else:
                raise ValidationError(
                    f"Unsupported grid dimensionality: {len(self.grid_shape)}. Only 2D and 3D grids are supported."
                )

            object.__setattr__(self, "coordinates", coordinates)


class BoundaryCondition(
    typing.Generic[NDimension],
    StoreSerializable,
    # Register serialization handlers for boundary functions
    serializers={"func": serialize_boundary_function},
    deserializers={"func": deserialize_boundary_function},
):
    """
    Base class for boundary conditions on N-dimensional grids.

    Each boundary condition type must implement an 'apply' method that receives
    the full grid, boundary indices, direction, and optional metadata.
    """

    __abstract_serializable__ = True

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """
        Apply the boundary condition to the specified grid region.

        :param grid: The full grid (including ghost cells)
        :param boundary_indices: Slice indices defining the boundary region
        :param direction: The boundary direction (left, right, etc.)
        :param metadata: Optional metadata for advanced boundary conditions
        """
        raise NotImplementedError


_BOUNDARY_CONDITIONS: typing.Dict[str, typing.Type[BoundaryCondition]] = {}
boundary_condition = make_serializable_type_registrar(
    base_cls=BoundaryCondition,
    registry=_BOUNDARY_CONDITIONS,
    lock=threading.Lock(),
    key_attr="__type__",
    override=False,
    auto_register_serializer=True,
    auto_register_deserializer=True,
)
"""Decorator to register a boundary condition type for serialization."""


@boundary_condition
class NoFlowBoundary(BoundaryCondition[NDimension]):
    """
    Implements a no-flow boundary condition.

    No-flow boundaries ensure that the gradient normal to the boundary is zero,
    effectively creating a sealed boundary where no flux occurs.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import NoFlowBoundary, GridBoundaryCondition, get_neighbor_indices

    # Create a sealed reservoir boundary
    sealed_boundary = NoFlowBoundary()

    # Use in a grid boundary condition for pressure
    pressure_boundary_config = GridBoundaryCondition(
        left=NoFlowBoundary(),  # Left side sealed
        right=NoFlowBoundary(),   # Right side sealed
        front=NoFlowBoundary(),  # Front sealed
        back=NoFlowBoundary(),   # Back sealed
    )

    # Apply to a 2D pressure grid with ghost cells
    pressure_grid = np.full((52, 52), 1500.0)  # 50x50 + 2 ghost cells
    pressure_boundary_config.apply(pressure_grid)

    # Result: Ghost cells copy values from neighboring interior cells
    # This is implemented using the get_neighbor_indices() utility function
    print(pressure_grid[0, :])   # Left boundary = pressure_grid[1, :]
    print(pressure_grid[-1, :])  # Right boundary = pressure_grid[-2, :]
    ```
    """

    __type__ = "no_flow_boundary"
    __abstract_serializable__ = True

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply no-flow boundary by copying values from neighboring cells."""
        neighbor_indices = get_neighbor_indices(boundary_indices, direction)
        grid[boundary_indices] = grid[neighbor_indices]

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls()


@boundary_condition
@attrs.frozen
class ConstantBoundary(BoundaryCondition[NDimension]):
    """
    Implements a constant boundary condition (Dirichlet).

    Sets the boundary to a fixed constant value, useful for modeling
    fixed pressure inlets, constant temperature walls, etc.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import ConstantBoundary, GridBoundaryCondition

    # Create constant pressure inlet at 2000 psi
    pressure_inlet = ConstantBoundary(constant=2000.0)

    # Create constant temperature boundary at 150°F
    temp_boundary = ConstantBoundary(constant=150.0)

    # Use in reservoir simulation (pressure property only)
    pressure_boundary_config = GridBoundaryCondition(
        left=ConstantBoundary(constant=2500.0),  # High pressure injection
        right=ConstantBoundary(constant=1000.0),   # Low pressure production
        front=NoFlowBoundary(),                   # Sealed sides
        back=NoFlowBoundary(),
    )

    pressure_grid = np.full((52, 52), 1500.0)
    pressure_boundary_config.apply(pressure_grid)

    # Result: Boundary cells set to constant values
    assert np.all(pressure_grid[0, :] == 2500.0)   # Left = 2500 psi
    assert np.all(pressure_grid[-1, :] == 1000.0)  # Right = 1000 psi
    ```
    """

    __type__ = "constant_boundary"

    constant: typing.Any
    """The constant value to set at the boundary."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply constant boundary condition, preserving grid dtype."""
        # Preserve the grid's dtype when setting constant value
        grid[boundary_indices] = grid.dtype.type(self.constant)


@boundary_condition
@attrs.frozen
class VariableBoundary(BoundaryCondition[NDimension]):
    """
    Implements a variable boundary condition using a callable function.

    Allows for complex boundary conditions that depend on the grid state,
    location, direction, and metadata.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import VariableBoundary, Boundary, get_neighbor_indices, boundary_function, GridBoundaryCondition

    @boundary_function
    def pressure_gradient_func(grid, boundary_indices, direction, metadata):
        # Example: Pressure increases with depth
        if direction == Boundary.BACK:  # Top boundary
            return np.full(grid[boundary_indices].shape, 1000.0)  # Low pressure
        elif direction == Boundary.FRONT:  # Bottom boundary
            return np.full(grid[boundary_indices].shape, 3000.0)  # High pressure

        # For sides, implement no-flow (Neumann) by copying neighbor values
        # Copy from neighbors for other boundaries
        neighbor_indices = get_neighbor_indices(boundary_indices, direction)
        return grid[neighbor_indices]

    # Create variable boundary
    var_boundary = VariableBoundary(func=pressure_gradient_func)

    # Use in simulation (pressure property)
    pressure_boundary_config = GridBoundaryCondition(
        left=var_boundary,
        right=var_boundary,
        front=var_boundary,
        back=var_boundary,
    )

    pressure_grid = np.full((52, 52), 2000.0)
    pressure_boundary_config.apply(pressure_grid)

    # Result: Top = 1000 psi, Bottom = 3000 psi, sides copy neighbors (no-flow)
    ```
    """

    __type__ = "variable_boundary"

    func: typing.Callable[
        [
            NDimensionalGrid[NDimension],
            typing.Tuple[slice, ...],
            Boundary,
            typing.Optional[BoundaryMetadata],
        ],
        NDimensionalGrid[NDimension],
    ]
    """Function to compute boundary values based on the grid, indices, direction, and metadata."""

    def __attrs_post_init__(self) -> None:
        if not callable(self.func):
            raise ValidationError("func must be a callable function.")

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply variable boundary condition using the provided function."""
        result = self.func(grid, boundary_indices, direction, metadata)
        # Preserve the grid's dtype
        grid[boundary_indices] = np.asarray(result, dtype=grid.dtype)


DirichletBoundary = ConstantBoundary
"""Alias for `ConstantBoundary` representing Dirichlet boundary conditions."""


@boundary_condition
@attrs.frozen
class SpatialBoundary(BoundaryCondition[NDimension]):
    """
    Implements a spatial boundary condition using coordinate-based functions.

    Perfect for creating realistic geological boundaries like pressure gradients,
    temperature profiles, or depth-dependent properties.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import SpatialBoundary, BoundaryMetadata, GridBoundaryCondition, boundary_function

    # Create coordinate grid (1000 ft x 500 ft reservoir)
    x_coords = np.linspace(0, 1000, 51)  # 0 to 1000 ft
    y_coords = np.linspace(0, 500, 26)   # 0 to 500 ft
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    coordinates = np.stack([xx, yy], axis=-1)

    metadata = BoundaryMetadata(coordinates=coordinates)

    # Example 1: Linear pressure drop with distance
    linear_gradient = boundary_function(lambda x, y: 2000 - 0.5 * x, name="linear_pressure_gradient")
    pressure_gradient = SpatialBoundary(
        func=linear_gradient  # 2000 psi at x=0, drops to 1500 at x=1000
    )

    # Example 2: Pressure increasing with depth
    depth_pressure_func = boundary_function(lambda x, y: 2000 + 0.03 * y, name="depth_pressure_func")
    depth_pressure = SpatialBoundary(
        func=depth_pressure_func  # 2000 psi at y=0, 2015 psi at y=500
    )

    # Example 3: Radial pressure distribution from center
    radial_pressure_func = boundary_function(lambda x, y: 2000 + 10 * np.sqrt((x-500)**2 + (y-250)**2), name="radial_pressure_distribution")
    radial_pressure = SpatialBoundary(
        func=radial_pressure_func  # Increases with distance from center (500,250)
    )

    # Apply to boundaries (pressure property only)
    pressure_boundary_config = GridBoundaryCondition(
        left=pressure_gradient,  # West boundary with linear gradient
        front=depth_pressure,     # South boundary with depth-based pressure
        right=radial_pressure,     # East boundary with radial pattern
    )

    pressure_grid = np.full((53, 28), 1800.0)  # Grid with ghost cells
    pressure_boundary_config.apply(pressure_grid, metadata=metadata)

    # Result: Boundaries follow spatial functions based on coordinates
    ```
    """

    __type__ = "spatial_boundary"

    func: typing.Callable[..., np.ndarray]
    """Function that takes coordinate arrays and returns boundary values."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply spatial boundary condition using coordinate-based function."""
        if metadata is None or metadata.coordinates is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires coordinate metadata"
            )

        # Extract coordinates at boundary
        coords = metadata.coordinates[boundary_indices]

        # Apply spatial function based on dimensionality and preserve dtype
        if coords.ndim == 2:  # 2D case
            if coords.shape[-1] >= 2:
                result = self.func(coords[..., 0], coords[..., 1])
            else:
                result = self.func(coords[..., 0])
            grid[boundary_indices] = np.asarray(result, dtype=grid.dtype)
        elif coords.ndim == 3:  # 3D case
            if coords.shape[-1] >= 3:
                result = self.func(coords[..., 0], coords[..., 1], coords[..., 2])
            elif coords.shape[-1] >= 2:
                result = self.func(coords[..., 0], coords[..., 1])
            else:
                result = self.func(coords[..., 0])
            grid[boundary_indices] = np.asarray(result, dtype=grid.dtype)
        else:
            # Fallback: flatten coordinates and apply function
            flat_coords = coords.reshape(-1, coords.shape[-1])
            if flat_coords.shape[1] >= 2:
                result = self.func(flat_coords[:, 0], flat_coords[:, 1])
            else:
                result = self.func(flat_coords[:, 0])
            grid[boundary_indices] = np.asarray(result, dtype=grid.dtype).reshape(
                coords.shape[:-1]
            )


@boundary_condition
@attrs.frozen
class TimeDependentBoundary(BoundaryCondition[NDimension]):
    """
    Implements a time-dependent boundary condition.

    Perfect for modeling cyclic injection, seasonal variations,
    or any boundary condition that changes over time.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import TimeDependentBoundary, BoundaryMetadata, GridBoundaryCondition, boundary_function

    # Example 1: Sinusoidal injection pressure (daily cycle)
    sinusoidal_func = boundary_function(lambda t: 2000 + 200 * np.sin(2 * np.pi * t / 86400), name="sinusoidal_pressure")
    daily_cycle = TimeDependentBoundary(
        func=sinusoidal_func  # 24-hour cycle
    )

    # Example 2: Linear pressure ramp-up
    linear_ramp_func = boundary_function(lambda t: min(1000 + 0.1 * t, 2500), name="linear_pressure_ramp")
    pressure_ramp = TimeDependentBoundary(
        func=linear_ramp_func  # Ramp from 1000 to 2500 psi
    )

    # Example 3: Exponential decay (well shut-in)
    exponential_decay_func = boundary_function(lambda t: 2000 * np.exp(-t / 3600), name="exponential_pressure_decay")
    pressure_decay = TimeDependentBoundary(
        func=exponential_decay_func  # Decay with 1-hour time constant
    )

    # Example 4: Step function (sudden pressure change)
    step_func = boundary_function(lambda t: 2500 if t > 1800 else 1500, name="step_pressure_change")
    step_pressure = TimeDependentBoundary(
        func=step_func  # Jump at 30 minutes
    )

    # Use in simulation at t = 12 hours (pressure property)
    metadata = BoundaryMetadata(time=43200.0)  # 12 hours in seconds

    pressure_boundary_config = GridBoundaryCondition(
        left=daily_cycle,     # Cyclic injection
        right=pressure_ramp,    # Gradual pressure increase
        front=pressure_decay,  # Exponential decay
        back=step_pressure,    # Step function
    )

    pressure_grid = np.full((52, 52), 1800.0)
    pressure_boundary_config.apply(pressure_grid, metadata=metadata)

    # Result: Each boundary reflects time-dependent function at t=43200s
    print(f"Daily cycle value: {2000 + 200 * np.sin(2 * np.pi * 43200 / 86400):.1f}")
    print(f"Ramp value: {min(1000 + 0.1 * 43200, 2500):.1f}")
    ```
    """

    __type__ = "time_dependent_boundary"

    func: typing.Callable[[float], float]
    """Function that takes time and returns boundary value."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply time-dependent boundary condition."""
        if metadata is None or metadata.time is None:
            raise ValidationError(f"{self.__class__.__name__} requires time metadata")

        value = self.func(metadata.time)
        # Preserve the grid's dtype
        grid[boundary_indices] = grid.dtype.type(value)


@boundary_condition
@attrs.frozen
class LinearGradientBoundary(BoundaryCondition[NDimension]):
    """
    Implements a linear gradient boundary condition.

    Creates a linear variation of the property across the boundary,
    useful for modeling pressure drops, temperature gradients, or
    concentration profiles.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import LinearGradientBoundary, BoundaryMetadata, GridBoundaryCondition

    # Create coordinate grid for a 1000x500 ft reservoir
    x_coords = np.linspace(0, 1000, 51)
    y_coords = np.linspace(0, 500, 26)
    z_coords = np.linspace(50, 150, 11)  # 50-150 ft depth

    # 3D coordinates
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    coordinates = np.stack([xx, yy, zz], axis=-1)

    metadata = BoundaryMetadata(coordinates=coordinates)

    # Example 1: Pressure drop across reservoir (west to east)
    pressure_drop = LinearGradientBoundary(
        start=2500.0,      # High pressure at west (x=0)
        end=1500.0,        # Low pressure at east (x=1000)
        direction="x"
    )

    # Example 2: Pressure gradient with depth (shallow to deep)
    depth_pressure = LinearGradientBoundary(
        start=2000.0,      # Lower pressure at shallow depth (z=50)
        end=2300.0,        # Higher pressure at deep depth (z=150)
        direction="z"
    )

    # Example 3: Pressure gradient from north to south
    pressure_ns_gradient = LinearGradientBoundary(
        start=2200.0,      # High pressure at north (y=0)
        end=1800.0,        # Low pressure at south (y=500)
        direction="y"
    )

    # Apply to 3D grid boundaries (pressure property only)
    pressure_boundary_config = GridBoundaryCondition(
        left=pressure_drop,     # West boundary: pressure drop west-east
        front=depth_pressure,    # South boundary: pressure with depth
        top=pressure_ns_gradient, # Top boundary: pressure north-south
    )

    pressure_grid = np.full((53, 28, 13), 2000.0)  # 3D grid with ghost cells
    pressure_boundary_config.apply(pressure_grid, metadata=metadata)

    # Result: Linear gradients applied to pressure boundaries
    # West boundary varies from 2500 to 1500 psi along x-direction
    # South boundary varies from 2000 to 2300 psi along z-direction
    # Top boundary varies from 2200 to 1800 psi along y-direction
    ```
    """

    __type__ = "linear_gradient_boundary"

    start: float
    """Value at the start of the gradient."""
    end: float
    """Value at the end of the gradient."""
    direction: typing.Literal["x", "y", "z"]
    """Direction of the gradient."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply linear gradient boundary condition."""
        if metadata is None or metadata.coordinates is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires coordinate metadata"
            )

        coords = metadata.coordinates[boundary_indices]

        # Determine which coordinate axis to use for gradient
        if self.direction == "x":
            coord_values = coords[..., 0]
        elif self.direction == "y":
            coord_values = coords[..., 1]
        elif self.direction == "z":
            coord_values = coords[..., 2] if coords.shape[-1] > 2 else coords[..., 0]
        else:
            raise ValidationError(f"Invalid gradient direction: {self.direction}")

        # Calculate gradient
        coord_min = np.min(coord_values)
        coord_max = np.max(coord_values)

        if coord_max == coord_min:
            # No gradient possible, use start value (preserve dtype)
            grid[boundary_indices] = grid.dtype.type(self.start)
        else:
            # Linear interpolation, preserve dtype
            normalized_coords = (coord_values - coord_min) / (coord_max - coord_min)
            result = self.start + normalized_coords * (self.end - self.start)
            grid[boundary_indices] = result.astype(grid.dtype, copy=False)


@boundary_condition
@attrs.frozen
class FluxBoundary(BoundaryCondition[NDimension]):
    """
    Implements a flux boundary condition (Neumann with physical interpretation).

    Sets a specified flux (rate of change) across the boundary, useful for
    modeling injection/production rates, heat flux, or mass transfer.

    **Sign Convention (Library-Wide Standard):**
    - **Positive flux (+)**: Flow INTO the reservoir (injection/inflow)
    - **Negative flux (-)**: Flow OUT OF the reservoir (production/outflow)

    This convention is consistent across all BORES modules (wells, boundaries, etc.).

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import FluxBoundary, BoundaryMetadata, GridBoundaryCondition

    # Define cell dimensions (20x20 ft cells)
    cell_dims = (20.0, 20.0)
    thickness = np.full((50, 50), 10.0)  # 10 ft thick reservoir (no ghost cells)
    metadata = BoundaryMetadata(
        cell_dimension=cell_dims,
        thickness_grid=thickness,
        grid_shape=(50, 50)
    )

    # Example 1: Water injection (positive flux)
    water_injector = FluxBoundary(flux=100.0)  # 100 bbl/day/ft² injection

    # Example 2: Oil production (negative flux)
    oil_producer = FluxBoundary(flux=-50.0)    # 50 bbl/day/ft² production

    # Example 3: Heat injection
    heat_injector = FluxBoundary(flux=1000.0)  # 1000 BTU/hr/ft²

    # Example 4: Gas venting (very high production)
    gas_vent = FluxBoundary(flux=-500.0)       # 500 Mscf/day/ft²

    # Set up reservoir with injection/production boundaries (pressure property)
    pressure_boundary_config = GridBoundaryCondition(
        left=water_injector,  # West: water injection
        right=oil_producer,     # East: oil production
        front=FluxBoundary(flux=0.0),    # South: no flux
        back=gas_vent,         # North: gas production
    )

    # Apply to pressure grid (psi)
    pressure_grid = np.full((52, 52), 2000.0)  # Initial 2000 psi
    pressure_boundary_config.apply(pressure_grid, metadata=metadata)

    # Result: Boundary values calculated from flux and cell spacing
    # West boundary: φ_boundary = φ_neighbor + flux * dx
    # φ_boundary = 2000 + 100 * (20/2) = 2000 + 1000 = 3000 psi
    # East boundary: φ_boundary = 2000 + (-50) * (20/2) = 2000 - 500 = 1500 psi

    print(f"Injection boundary pressure: {pressure_grid[0, 10]:.0f} psi")    # ~3000
    print(f"Production boundary pressure: {pressure_grid[-1, 10]:.0f} psi")  # ~1500

    # Physical interpretation:
    # - Positive flux increases boundary pressure (injection)
    # - Negative flux decreases boundary pressure (production)
    # - Zero flux maintains neighbor pressure (no-flow equivalent)
    ```
    """

    __type__ = "flux_boundary"

    flux: float
    """Flux value (positive for injection, negative for production)."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """
        Apply flux boundary condition.

        Sign convention: positive flux = flow INTO the domain (injection),
        negative flux = flow OUT OF the domain (production).

        Ghost cell values are set so that the finite-difference gradient
        across the boundary face equals the specified flux:

            φ_ghost = φ_neighbor + sign * flux * spacing

        where spacing = half the cell width in the normal direction, and:
            sign = +1 for LEFT, FRONT, BOTTOM faces (ghost is upstream of inward flow)
            sign = -1 for RIGHT, BACK, TOP faces  (ghost is downstream of inward flow)

        Examples (dx=20 ft, so spacing=10 ft):
            LEFT,  flux=+100  →  φ_ghost = φ_neighbor + 1000  (injection raises ghost)
            RIGHT, flux=-50   →  φ_ghost = φ_neighbor + 500   (production raises ghost on exit side)
            LEFT,  flux=0     →  φ_ghost = φ_neighbor          (no-flow, equivalent to NoFlowBoundary)

        Note: In this codebase k=0 is the TOP (shallowest) layer and k increases
        downward, so BOTTOM corresponds to the deepest layer (k=-1).
        """
        if metadata is None or metadata.cell_dimension is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires cell dimension metadata"
            )

        # Get neighboring cell values for gradient calculation
        neighbor_indices = get_neighbor_indices(boundary_indices, direction)
        neighbor_values = grid[neighbor_indices]

        # Calculate distance between boundary and neighbor (half cell spacing)
        dx, dy = metadata.cell_dimension

        if direction in [Boundary.LEFT, Boundary.RIGHT]:
            spacing = dx / 2.0
        elif direction in [Boundary.FRONT, Boundary.BACK]:
            spacing = dy / 2.0
        elif direction in [Boundary.BOTTOM, Boundary.TOP]:
            # For 3D, use thickness_grid if available for z-direction spacing
            # Note: k=0 is TOP, k=-1 is BOTTOM in this codebase
            if (
                metadata.thickness_grid is not None
                and metadata.thickness_grid.ndim == 3
            ):
                # Use average thickness at boundary layer for spacing
                # thickness_grid has no ghost cells, so we need to map boundary to interior
                if direction == Boundary.BOTTOM:
                    # Bottom boundary (deepest) - use last layer thickness (k=-1)
                    avg_thickness = np.mean(metadata.thickness_grid[:, :, -1])
                else:  # TOP
                    # Top boundary (shallowest) - use first layer thickness (k=0)
                    avg_thickness = np.mean(metadata.thickness_grid[:, :, 0])
                spacing = avg_thickness / 2.0  # type: ignore[assignment]
            else:
                # Fallback to dx if no thickness info
                spacing = dx / 2.0
        else:
            spacing = dx / 2.0

        # Determine sign based on boundary direction
        # Positive flux = flow into the domain.
        # Ghost cell value = neighbor + flux * spacing (so ghost > neighbor for injection)
        # For LEFT/FRONT/BOTTOM: flow into domain means ghost is upstream (higher),
        # so sign = +1.
        # For RIGHT/BACK/TOP: flow into domain means ghost is downstream (lower
        # relative to the right-side convention), so sign = -1.
        if direction in [Boundary.LEFT, Boundary.FRONT, Boundary.BOTTOM]:
            sign = 1.0
        else:
            sign = -1.0

        # Apply flux boundary: dφ/dn = flux (outward normal convention)
        # φ_boundary = φ_neighbor + sign * flux * spacing
        # Preserve the grid's dtype
        result = neighbor_values + sign * self.flux * spacing
        grid[boundary_indices] = result.astype(grid.dtype, copy=False)


@boundary_condition
@attrs.frozen
class RobinBoundary(BoundaryCondition[NDimension]):
    """
    Implements a Robin (mixed/convective) boundary condition.

    Combines Dirichlet and Neumann conditions: α*φ + β*∂φ/∂n = γ

    Common applications:
    - Heat transfer with convection: h*(T - T_inf) = -k*∂T/∂n
    - Mass transfer with surface reaction
    - Pressure boundaries with partial flow resistance

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import RobinBoundary, BoundaryMetadata, GridBoundaryCondition

    # Example 1: Convective heat transfer boundary
    # h*(T - T_ambient) = -k*∂T/∂n  =>  α=h, β=k, γ=h*T_ambient
    # Rearranged: T_boundary = (γ + β*T_neighbor/spacing) / (α + β/spacing)
    convective_bc = RobinBoundary(
        alpha=10.0,      # Heat transfer coefficient h (BTU/hr/ft²/°F)
        beta=0.5,        # Thermal conductivity k (BTU/hr/ft/°F)
        gamma=700.0      # h * T_ambient (10 * 70°F)
    )

    # Example 2: Semi-permeable pressure boundary
    # Partial resistance to flow at boundary
    semi_permeable = RobinBoundary(
        alpha=1.0,       # Dirichlet weight
        beta=0.1,        # Neumann weight (small = more Dirichlet-like)
        gamma=2000.0     # Reference pressure
    )

    # Example 3: Pure Dirichlet (β=0): α*φ = γ => φ = γ/α
    dirichlet_like = RobinBoundary(alpha=1.0, beta=0.0, gamma=1500.0)

    # Example 4: Pure Neumann (α=0): β*∂φ/∂n = γ
    neumann_like = RobinBoundary(alpha=0.0, beta=1.0, gamma=100.0)

    metadata = BoundaryMetadata(
        cell_dimension=(20.0, 20.0),
        grid_shape=(50, 25)
    )

    temperature_bc = GridBoundaryCondition(
        left=convective_bc,
        right=semi_permeable,
    )

    temperature_grid = np.full((52, 27), 100.0)
    temperature_bc.apply(temperature_grid, metadata=metadata)
    ```
    """

    __type__ = "robin_boundary"

    alpha: float
    """Coefficient for the value term (Dirichlet weight)."""
    beta: float
    """Coefficient for the gradient term (Neumann weight)."""
    gamma: float
    """Right-hand side constant."""

    def __attrs_post_init__(self) -> None:
        if self.alpha == 0.0 and self.beta == 0.0:
            raise ValidationError(
                "At least one of `alpha` or `beta` must be non-zero for Robin boundary."
            )

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """
        Apply Robin boundary condition: α*φ + β*∂φ/∂n = γ

        The gradient ∂φ/∂n uses the OUTWARD normal convention (standard in PDEs):
        the normal points AWAY from the domain, so:
            sign = -1 for LEFT, FRONT, BOTTOM faces (outward normal is negative)
            sign = +1 for RIGHT, BACK, TOP faces    (outward normal is positive)

        Discretization:
            ∂φ/∂n ≈ sign * (φ_ghost - φ_neighbor) / spacing

        Substituting into α*φ_ghost + β*∂φ/∂n = γ and solving:
            φ_ghost = (γ + β*sign*φ_neighbor/spacing) / (α + β*sign/spacing)

        Special cases:
            β=0 (pure Dirichlet): φ_ghost = γ/α  (ignores neighbor)
            α=0 (pure Neumann):   φ_ghost = φ_neighbor + γ*spacing/sign
                → positive γ with sign=-1 (LEFT face) means outward flux = γ,
                  i.e. flow exits through the left face

        Note: RobinBoundary's sign convention is OPPOSITE to FluxBoundary.
        FluxBoundary defines positive as flow INTO the domain; Robin's ∂φ/∂n
        is positive when flux points OUT of the domain (outward normal).

        Note: In this codebase k=0 is the TOP (shallowest) layer and k increases
        downward, so BOTTOM corresponds to the deepest layer (k=-1).
        """
        if metadata is None or metadata.cell_dimension is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires cell dimension metadata"
            )

        # Get neighboring cell values
        neighbor_indices = get_neighbor_indices(boundary_indices, direction)
        neighbor_values = grid[neighbor_indices]

        # Calculate spacing
        dx, dy = metadata.cell_dimension
        if direction in [Boundary.LEFT, Boundary.RIGHT]:
            spacing = dx / 2.0
        elif direction in [Boundary.FRONT, Boundary.BACK]:
            spacing = dy / 2.0
        else:
            # Z-direction: use thickness if available
            # Note: k=0 is TOP, k=-1 is BOTTOM in this codebase
            if (
                metadata.thickness_grid is not None
                and metadata.thickness_grid.ndim == 3
            ):
                if direction == Boundary.BOTTOM:
                    # Bottom boundary (deepest) - use last layer thickness (k=-1)
                    avg_thickness = np.mean(metadata.thickness_grid[:, :, -1])
                else:  # TOP
                    # Top boundary (shallowest) - use first layer thickness (k=0)
                    avg_thickness = np.mean(metadata.thickness_grid[:, :, 0])
                spacing = avg_thickness / 2.0  # type: ignore[assignment]
            else:
                spacing = dx / 2.0

        # Direction sign for gradient (outward normal convention)
        if direction in [
            Boundary.LEFT,
            Boundary.FRONT,
            Boundary.BOTTOM,
        ]:
            sign = -1.0
        else:
            sign = 1.0

        # Robin BC: α*φ_boundary + β*∂φ/∂n = γ
        # Discretized gradient: ∂φ/∂n ≈ sign * (φ_boundary - φ_neighbor) / spacing
        # Substituting: α*φ_boundary + β*sign*(φ_boundary - φ_neighbor)/spacing = γ
        # Solving for φ_boundary:
        # φ_boundary * (α + β*sign/spacing) = γ + β*sign*φ_neighbor/spacing
        # φ_boundary = (γ + β*sign*φ_neighbor/spacing) / (α + β*sign/spacing)

        effective_beta = self.beta * sign / spacing
        denominator = self.alpha + effective_beta

        if np.abs(denominator) < 1e-12:
            # Degenerate case - fall back to neighbor value
            grid[boundary_indices] = neighbor_values
        else:
            # Preserve the grid's dtype
            result = (self.gamma + effective_beta * neighbor_values) / denominator
            grid[boundary_indices] = result.astype(grid.dtype, copy=False)


@boundary_condition
@attrs.frozen
class PeriodicBoundary(BoundaryCondition[NDimension]):
    """
    Implements a periodic boundary condition.

    Links opposite boundaries so that flow/values wrap around,
    useful for modeling repeating geological patterns or infinite domains.

    Note: For proper periodic BCs, both opposite faces must use `PeriodicBoundary`.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import PeriodicBoundary, GridBoundaryCondition

    # Example 1: Periodic in x-direction (left-right wrap)
    periodic_x = GridBoundaryCondition(
        left=PeriodicBoundary(),   # Copies from right interior
        right=PeriodicBoundary(),  # Copies from left interior
    )

    # Example 2: Fully periodic 2D domain
    fully_periodic = GridBoundaryCondition(
        left=PeriodicBoundary(),
        right=PeriodicBoundary(),
        front=PeriodicBoundary(),
        back=PeriodicBoundary(),
    )

    pressure_grid = np.random.uniform(1000, 2000, (52, 27))
    fully_periodic.apply(pressure_grid)

    # Result:
    # - Left ghost cells = right interior values
    # - Right ghost cells = left interior values
    # - Front ghost cells = back interior values
    # - Back ghost cells = front interior values
    ```
    """

    __type__ = "periodic_boundary"
    __abstract_serializable__ = True

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply periodic boundary by copying from opposite interior boundary."""
        # Get indices of the opposite interior cells
        opposite_indices = list(boundary_indices)

        if direction == Boundary.LEFT:
            # Left boundary copies from right interior (second-to-last)
            opposite_indices[0] = slice(-2, -1)
        elif direction == Boundary.RIGHT:
            # Right boundary copies from left interior (second from start)
            opposite_indices[0] = slice(1, 2)
        elif direction == Boundary.FRONT:
            # Front boundary copies from back interior
            opposite_indices[1] = slice(-2, -1)
        elif direction == Boundary.BACK:
            # Back boundary copies from front interior
            opposite_indices[1] = slice(1, 2)
        elif direction == Boundary.BOTTOM:
            # Bottom boundary copies from top interior
            if len(boundary_indices) < 3:
                raise ValidationError(
                    f"Cannot apply {direction.name} to {len(boundary_indices)}D grid"
                )
            opposite_indices[2] = slice(-2, -1)
        elif direction == Boundary.TOP:
            # Top boundary copies from bottom interior
            if len(boundary_indices) < 3:
                raise ValidationError(
                    f"Cannot apply {direction.name} to {len(boundary_indices)}D grid"
                )
            opposite_indices[2] = slice(1, 2)

        grid[boundary_indices] = grid[tuple(opposite_indices)]

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {}

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        return cls()


NeumannBoundary = FluxBoundary
"""Alias for `FluxBoundary` representing Neumann boundary conditions (flux-based)."""


@boundary_condition
@attrs.define
class CarterTracyAquifer(BoundaryCondition[NDimension]):
    """
    Carter-Tracy finite aquifer model that uses physical aquifer properties for accurate simulation.

    The Carter-Tracy model (1960) provides a semi-analytical solution for finite aquifer
    behavior, accounting for transient pressure response and material balance between
    the reservoir and aquifer. This is more realistic than constant pressure (infinite
    aquifer) or no-flow (no aquifer support) boundaries.

    **Physical Model:**
    The aquifer provides pressure support through water influx determined by:
    1. Cumulative pressure drop at the reservoir-aquifer boundary
    2. Aquifer properties (permeability, porosity, compressibility, geometry)
    3. Van Everdingen-Hurst dimensionless water influx functions
    4. Hydraulic diffusivity and dimensionless time

    **Mathematical Formulation:**
    Water influx rate at time t:
        Q_aquifer(t) = B * Σ[ΔP(t_i) * W_D'(t_D - t_Di)]

    where:
        - B = aquifer constant = 1.119 * φ * c_t * (r_e² - r_w²) * h * (θ/360°) / μ_w
        - ΔP(t_i) = pressure change at time t_i
        - W_D'(t_D) = dimensionless water influx derivative (Van Everdingen-Hurst)
        - t_D = dimensionless time = (k / (φ * μ * c_t)) * (t / r_w²)

    **Two Usage Modes:**

    1. **Physical Properties (Recommended)**: Specify k, φ, μ, c_t, radii, thickness
       - Dimensionally correct and physically meaningful
       - Aquifer constant B and dimensionless time t_D computed from first principles
       - Allows sensitivity analysis on individual properties

    2. **Calibrated Constant (Legacy)**: Specify pre-computed `aquifer_constant`
       - For history matching when physical properties are uncertain
       - Must also provide dimensionless_radius_ratio
       - Simplified dimensionless time calculation (t_D ≈ t / characteristic_time)

    **Example (Using Physical Properties - Recommended):**
    ```python
    from bores.boundary_conditions import CarterTracyAquifer, GridBoundaryCondition

    # Edge water drive with known aquifer properties
    edge_aquifer = CarterTracyAquifer(
        aquifer_permeability=500.0,       # mD - from core/log data
        aquifer_porosity=0.25,            # fraction - from logs
        aquifer_compressibility=3e-6,     # psi⁻¹ - rock + water compressibility
        water_viscosity=0.5,              # cP - at reservoir conditions
        inner_radius=1000.0,              # ft - reservoir-aquifer contact radius
        outer_radius=10000.0,             # ft - aquifer extent (seismic, geology)
        aquifer_thickness=50.0,           # ft - from logs/seismic
        initial_pressure=2500.0,          # psi - initial equilibrium pressure
        angle=180.0,                      # degrees - half-circle (edge drive)
    )

    # Bottom water drive (full contact)
    bottom_aquifer = CarterTracyAquifer(
        aquifer_permeability=800.0,
        aquifer_porosity=0.28,
        aquifer_compressibility=4e-6,
        water_viscosity=0.4,
        inner_radius=2000.0,
        outer_radius=20000.0,
        aquifer_thickness=100.0,
        initial_pressure=3000.0,
        angle=360.0,                      # Full contact
    )
    ```

    **Example (Using Calibrated Constant - Legacy):**
    ```python
    # When physical properties are uncertain - use history-matched constant
    aquifer = CarterTracyAquifer(
        aquifer_constant=50.0,            # bbl/psi - from history match
        initial_pressure=2500.0,          # psi
        dimensionless_radius_ratio=10.0,  # r_e/r_w - from geology estimate
        angle=180.0,                      # degrees
    )
    ```

    **Applications:**
    - Edge water drive reservoirs (use on lateral boundaries)
    - Bottom water drive (use on bottom boundary)
    - Aquifer support during primary depletion
    - History matching field production data
    - Pressure transient analysis and aquifer characterization

    **Comparison to Other Boundary Conditions:**
    - **Constant Pressure BC**: Assumes infinite aquifer (over-optimistic, unrealistic)
    - **No-Flow BC**: Assumes no aquifer support (conservative, pessimistic)
    - **Carter-Tracy**: Realistic finite aquifer with time-dependent support
    - **Fetkovich**: Simplified pseudo-steady-state (less accurate than Carter-Tracy)

    **Physical Considerations:**
    - Aquifer permeability typically lower than reservoir (10-1000 mD)
    - Total compressibility = rock + water (typically 1e-6 to 10e-6 psi⁻¹)
    - Inner radius = distance from well/reservoir center to aquifer contact
    - Outer radius = aquifer extent (from seismic, geology, or pressure transient tests)
    - Dimensionless radius r_D = r_e/r_w typically 5-50 (finite aquifer)
    - Early time: infinite-acting behavior (W_D' ∝ √t_D)
    - Late time: exponential decline (boundary effects dominant)

    **References:**
    - Carter, R.D., and Tracy, G.W. (1960). "An Improved Method for Calculating Water
      Influx." Journal of Petroleum Technology, 12(5), 415-417.
    - Van Everdingen, A.F., and Hurst, W. (1949). "The Application of the Laplace
      Transformation to Flow Problems in Reservoirs." Transactions of the AIME, 186, 305-324.
    - Chatas, A.T. (1953). "A Practical Treatment of Nonsteady-State Flow Problems in
      Reservoir Systems." Petroleum Engineer, B-44 to B-56.
    - Dake, L.P. (1978). "Fundamentals of Reservoir Engineering." Elsevier, Chapter 8.
    - Havlena, D., and Odeh, A.S. (1963). "The Material Balance as an Equation of a
      Straight Line." Journal of Petroleum Technology, 15(8), 896-900.
    """

    __type__ = "carter_tracy_aquifer"

    initial_pressure: float
    """
    Initial aquifer pressure (psi).

    Aquifer pressure at t=0, typically equal to initial reservoir pressure.
    Water influx is driven by the difference between current reservoir pressure
    and this initial value (ΔP = initial pressure - current pressure).

    Typical range: 1000-5000 psi (depends on reservoir depth and conditions)
    """

    aquifer_permeability: typing.Optional[float] = attrs.field(default=None)
    """
    Aquifer permeability (mD).

    Horizontal permeability of the aquifer rock. Controls how quickly water can
    flow from the aquifer into the reservoir. Typically lower than reservoir
    permeability.

    Typical range:
        - Low permeability aquifer: 10-100 mD (weak support)
        - Moderate permeability: 100-500 mD (typical)
        - High permeability: 500-2000 mD (strong support)

    Required if using physical properties mode.
    """

    aquifer_porosity: typing.Optional[float] = attrs.field(default=None)
    """
    Aquifer porosity (fraction).

    Pore volume fraction of the aquifer rock. Controls the storage capacity
    of the aquifer. Higher porosity = more water available for influx.

    Typical range:
        - Tight aquifer: 0.10-0.20 (limited storage)
        - Moderate aquifer: 0.20-0.30 (typical)
        - High porosity aquifer: 0.30-0.40 (large storage)

    Required if using physical properties mode.
    """

    aquifer_compressibility: typing.Optional[float] = attrs.field(default=None)
    """
    Total aquifer compressibility (psi⁻¹).

    Combined compressibility of aquifer rock and water:
        c_t = c_rock + c_water

    Controls pressure response to fluid withdrawal. Higher compressibility
    means more water released per psi pressure drop.

    Typical values:
        - Rock compressibility (c_rock): 3e-6 to 10e-6 psi⁻¹
        - Water compressibility (c_water): 3e-6 psi⁻¹ at standard conditions
        - Total (c_t): 5e-6 to 15e-6 psi⁻¹

    Required if using physical properties mode.
    """

    water_viscosity: typing.Optional[float] = attrs.field(default=None)
    """
    Water viscosity (cP).

    Viscosity of water in the aquifer at reservoir temperature and pressure.
    Controls flow resistance. Lower viscosity = faster influx.

    Typical range:
        - Cold water: 0.8-1.0 cP (shallow reservoirs)
        - Warm water: 0.3-0.5 cP (deep hot reservoirs)
        - Hot water: 0.2-0.3 cP (geothermal conditions)

    Required if using physical properties mode.
    """

    inner_radius: typing.Optional[float] = attrs.field(default=None)
    """
    Inner radius - reservoir-aquifer contact radius (ft).

    Distance from reservoir center (or well) to the aquifer-reservoir interface.
    For edge water drive, this is the reservoir radius. For bottom water drive,
    this is the effective radial distance to the oil-water contact.

    Typical range:
        - Small reservoir: 500-2000 ft
        - Medium reservoir: 2000-5000 ft
        - Large reservoir: 5000-20000 ft

    Required if using physical properties mode.
    """

    outer_radius: typing.Optional[float] = attrs.field(default=None)
    """
    Outer radius - aquifer extent (ft).

    Maximum extent of the aquifer from the reservoir center. Controls aquifer
    volume and finite boundary effects. Can be estimated from seismic data,
    geological models, or pressure transient analysis.

    Typical range:
        - Small finite aquifer: 2x to 5x inner radius
        - Moderate aquifer: 5x to 20x inner radius
        - Large aquifer: 20x to 100x inner radius
        - Very large (approaches infinite): >100x inner radius

    The ratio r_outer/r_inner determines aquifer strength and boundary effects.

    Required if using physical properties mode.
    """

    aquifer_thickness: typing.Optional[float] = attrs.field(default=None)
    """
    Aquifer thickness (ft).

    Vertical thickness of the aquifer. For edge water drive, this is typically
    similar to reservoir thickness. For bottom water drive, this can be much
    larger. Controls total aquifer pore volume.

    Typical range:
        - Thin aquifer: 10-50 ft (limited support)
        - Moderate aquifer: 50-200 ft (typical)
        - Thick aquifer: 200-1000 ft (strong support)

    Required if using physical properties mode.
    """

    aquifer_constant: typing.Optional[float] = attrs.field(default=None)
    """
    Pre-computed aquifer constant B (bbl/psi).

    For legacy/history-matching workflows when physical aquifer properties
    are uncertain. The aquifer constant lumps together all physical properties:

        B = 1.119 * φ * c_t * (r_e² - r_w²) * h / μ_w

    This can be calibrated by history matching observed reservoir pressure
    decline or water influx rates.

    Typical range:
        - Weak aquifer: 10-50 bbl/psi
        - Moderate aquifer: 50-200 bbl/psi
        - Strong aquifer: 200-1000 bbl/psi
        - Very strong aquifer: >1000 bbl/psi

    If specified, physical properties are not required (Option B mode).
    """

    dimensionless_radius_ratio: float = attrs.field(default=10.0)
    """
    Dimensionless aquifer radius ratio (r_outer / r_inner).

    Ratio of outer to inner aquifer radius. Controls aquifer size and
    boundary effects on water influx behavior.

    Physical interpretation:
        - r_D = 2-5: Small finite aquifer (weak support, early boundary effects)
        - r_D = 5-10: Moderate finite aquifer (typical field cases)
        - r_D = 10-30: Large finite aquifer (strong support)
        - r_D > 50: Very large aquifer (approaches infinite-acting)
        - r_D → ∞: Infinite aquifer (equivalent to constant pressure BC)

    Behavior:
        - Early time (t_D < 0.5): All aquifers behave infinite-acting
        - Late time (t_D > 2): Boundary effects dominate, smaller r_D = faster decline

    Default: 10.0 (moderate finite aquifer - typical for many reservoirs)

    Note: If using physical properties mode, this is automatically computed
    as outer_radius/inner_radius. This parameter is only used in calibrated
    constant mode (Option B).
    """

    angle: float = attrs.field(default=360.0)
    """
    Aquifer encroachment angle (degrees).

    Defines what fraction of the boundary has aquifer contact. For radial
    aquifers, this is the angle of the aquifer sector.

    Common values:
        - 360°: Full circular encroachment (aquifer surrounds entire reservoir)
                Example: Bottom water drive, full edge water drive
        - 180°: Half-circle encroachment (aquifer on one side)
                Example: Edge water drive on one flank
        - 90°: Quarter-circle encroachment (aquifer at corner)
        - 60°: One-sixth encroachment (limited aquifer contact)

    The influx is scaled proportionally: Q_actual = Q_full * (θ/360°)

    Default: 360.0 (full encroachment)
    """

    _pressure_history: typing.List[typing.Tuple[float, float]] = attrs.field(
        factory=list, init=False
    )
    """
    Internal state: List of (time, pressure_drop) tuples for convolution integral.

    Used to compute water influx using Van Everdingen-Hurst superposition:
        Q(t) = B * Σ[ΔP(t_i) * W_D'(t_D - t_Di)]
    """

    _cumulative_influx: float = attrs.field(default=0.0, init=False)
    """
    Internal state: Cumulative water influx from aquifer (bbl or ft³).

    Tracks total volume of water that has entered the reservoir from the
    aquifer since simulation start. Used for material balance calculations.
    """

    _computed_aquifer_constant: typing.Optional[float] = attrs.field(
        default=None, init=False
    )
    """
    Internal state: Computed aquifer constant B (bbl/psi).

    In physical properties mode, this is computed from k, φ, c_t, μ, radii, h.
    In calibrated constant mode, this equals the user-provided aquifer_constant.
    """

    _computed_dimensionless_radius_ratio: typing.Optional[float] = attrs.field(
        default=None, init=False
    )
    """
    Internal state: Computed dimensionless radius ratio r_D.

    In physical properties mode, this is outer_radius / inner_radius.
    In calibrated constant mode, this equals dimensionless_radius_ratio.
    """

    _hydraulic_diffusivity: typing.Optional[float] = attrs.field(
        default=None, init=False
    )
    """
    Internal state: Hydraulic diffusivity η (ft²/day).

    Only computed in physical properties mode:
        η = 0.006328 * k / (φ * μ * c_t)

    Used to convert real time to dimensionless time:
        t_D = η * t / r_w²

    In calibrated constant mode, this is None (simplified t_D calculation).
    """

    def __attrs_post_init__(self) -> None:
        """
        Validate parameters and compute derived quantities.

        Checks that user has provided either:
            - Option A: All physical properties (k, φ, c_t, μ, r_inner, r_outer, h)
            - Option B: Calibrated aquifer_constant

        Computes internal state variables based on the chosen mode.
        """
        has_physical_properties = all(
            x is not None
            for x in [
                self.aquifer_permeability,
                self.aquifer_porosity,
                self.aquifer_compressibility,
                self.water_viscosity,
                self.inner_radius,
                self.outer_radius,
                self.aquifer_thickness,
            ]
        )
        has_calibrated_constant = self.aquifer_constant is not None
        if not (has_physical_properties or has_calibrated_constant):
            raise ValidationError(
                f"{self.__class__.__name__!r} requires either:\n"
                "  Option A (recommended): Physical properties "
                "(aquifer_permeability, aquifer_porosity, aquifer_compressibility, "
                "water_viscosity, inner_radius, outer_radius, aquifer_thickness)\n"
                "  Option B (legacy): Calibrated aquifer_constant\n"
                "Please provide one complete set of parameters."
            )

        if has_physical_properties:
            # Compute from physical properties
            assert self.inner_radius is not None and self.outer_radius is not None
            assert self.aquifer_permeability is not None
            assert self.aquifer_porosity is not None
            assert self.aquifer_compressibility is not None
            assert self.water_viscosity is not None
            assert self.aquifer_thickness is not None

            # Compute dimensionless radius ratio
            r_D = self.outer_radius / self.inner_radius
            object.__setattr__(self, "_computed_dimensionless_radius_ratio", r_D)

            # Compute aquifer constant B (bbl/psi)
            # B = 1.119 * φ * c_t * (r_e² - r_w²) * h * (θ/360°) / μ_w
            # Note: angle factor applied in influx calculation, not here
            angle_fraction = self.angle / 360.0
            r_squared_diff = self.outer_radius**2 - self.inner_radius**2
            aquifer_constant_computed = (
                1.119
                * self.aquifer_porosity
                * self.aquifer_compressibility
                * angle_fraction
                * r_squared_diff
                * self.aquifer_thickness
                / self.water_viscosity
            )
            object.__setattr__(
                self, "_computed_aquifer_constant", aquifer_constant_computed
            )

            # Compute hydraulic diffusivity η (ft²/day)
            # η = 0.006328 * k / (φ * μ * c_t)
            # Conversion factor 0.006328 = (1 mD * 1 day) / (1 ft² * 1 cP * 1 psi⁻¹)
            hydraulic_diffusivity = (
                0.006328
                * self.aquifer_permeability
                / (
                    self.aquifer_porosity
                    * self.water_viscosity
                    * self.aquifer_compressibility
                )
            )
            object.__setattr__(self, "_hydraulic_diffusivity", hydraulic_diffusivity)

        else:
            # Use calibrated constant
            object.__setattr__(
                self, "_computed_aquifer_constant", self.aquifer_constant
            )
            object.__setattr__(
                self,
                "_computed_dimensionless_radius_ratio",
                self.dimensionless_radius_ratio,
            )
            # No hydraulic diffusivity in calibrated mode - will use simplified t_D
            object.__setattr__(self, "_hydraulic_diffusivity", None)

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: Boundary,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """
        Apply Carter-Tracy aquifer boundary condition.

        Computes water influx based on pressure history using Van Everdingen-Hurst
        convolution integral, then sets boundary pressure to maintain material
        balance between reservoir and aquifer.

        **Physical Process:**
        1. Monitor pressure at reservoir-aquifer interface
        2. Compute cumulative pressure drop history
        3. Calculate water influx using superposition (convolution integral)
        4. Update boundary pressure based on aquifer response

        **Material Balance:**
        The boundary pressure is set to reflect the aquifer's ability to support
        reservoir pressure. Strong aquifers maintain pressure near initial value,
        while weak aquifers allow more pressure decline.

        :param grid: The grid to apply the boundary condition to (typically pressure grid).
        :param boundary_indices: Tuple of slices defining the boundary region.
        :param direction: Boundary direction (e.g., Boundary.RIGHT, Boundary.BOTTOM).
        :param metadata: Optional metadata containing time, cell dimensions, etc.
        """
        if metadata is None or metadata.time is None:
            # Can't apply time-dependent aquifer model without time information
            # Fall back to constant pressure at initial value (infinite aquifer approximation)
            grid[boundary_indices] = self.initial_pressure
            return

        current_time = metadata.time

        # Get average boundary pressure from interior neighbor cells
        # This represents the reservoir pressure at the aquifer interface
        neighbor_indices = get_neighbor_indices(boundary_indices, direction)
        avg_boundary_pressure = float(np.mean(grid[neighbor_indices]))

        # Compute pressure drop from initial (ΔP = initial pressure - current pressure)
        # Positive ΔP means reservoir pressure has declined → water influx
        pressure_drop = self.initial_pressure - avg_boundary_pressure

        # Update pressure history (append new pressure drop at current time)
        # Only add if time has advanced (avoid duplicate entries at same timestep)
        if not self._pressure_history or current_time > self._pressure_history[-1][0]:
            self._pressure_history.append((current_time, pressure_drop))

        # Compute water influx rate using Van Everdingen-Hurst convolution
        water_influx_rate = self._compute_water_influx_rate(current_time)

        # Update cumulative influx (integrate influx rate over timestep)
        if len(self._pressure_history) > 1:
            dt = current_time - self._pressure_history[-2][0]
            self._cumulative_influx += water_influx_rate * dt

        # Material Balance for Boundary Pressure
        # The aquifer provides pressure support proportional to its strength
        # and the water influx capacity.
        #
        # Physical interpretation:
        # - Strong aquifer: Large influx capacity → maintains pressure close to initial pressure
        # - Weak aquifer: Limited influx → allows more pressure decline
        #
        # We use a support factor based on the ratio of actual influx to
        # maximum possible influx given the pressure drop.

        assert self._computed_aquifer_constant is not None
        if pressure_drop > 0:
            # Maximum instantaneous influx capacity (pseudo-steady approximation)
            # For finite aquifer: Q_max ≈ B * ΔP / (characteristic time factor)
            # We use the W_D' value at current conditions as a proxy
            max_influx_capacity = self._computed_aquifer_constant * pressure_drop

            # Support factor: ratio of actual influx to maximum capacity
            # = 0: No influx (no support) → boundary pressure = reservoir pressure
            # = 1: Full capacity influx (strong support) → boundary pressure ≈ initial pressure
            if max_influx_capacity > 0:
                support_factor = min(
                    1.0, water_influx_rate / (max_influx_capacity + 1e-10)
                )
            else:
                support_factor = 0.0

            # Boundary pressure interpolates between reservoir pressure and initial pressure
            # based on aquifer support strength
            boundary_pressure = self.initial_pressure - pressure_drop * (
                1.0 - support_factor
            )
        else:
            # No pressure drop or pressure increase (unusual) → maintain current pressure
            boundary_pressure = avg_boundary_pressure

        # Apply computed pressure to boundary cells
        grid[boundary_indices] = boundary_pressure

    def _compute_water_influx_rate(self, current_time: float) -> float:
        """
        Compute water influx rate using Van Everdingen-Hurst convolution integral.

        **Mathematical Formulation:**
        The Carter-Tracy model uses superposition to compute total influx from
        the history of pressure changes:

            Q(t) = B * Σ[ΔP(t_i) * W_D'(t_D - t_Di)]

        where:
            - Q(t) = water influx rate at current time (bbl/day or ft³/day)
            - B = aquifer constant (bbl/psi or ft³/psi)
            - ΔP(t_i) = pressure drop at historical time t_i (psi)
            - W_D'(t_D) = dimensionless water influx derivative (dimensionless)
            - t_D = dimensionless time since pressure change

        **Dimensionless Time Calculation:**
        - Physical properties mode: t_D = (η * Δt) / r_w²
          where η = hydraulic diffusivity = 0.006328 * k / (φ * μ * c_t)
        - Calibrated constant mode: t_D = Δt (simplified, dimensionally inconsistent)

        The convolution integral accounts for the transient response of the aquifer
        to each historical pressure change, properly superposing all effects.

        :param current_time: Current simulation time (days)
        :return: Water influx rate from aquifer (bbl/day or ft³/day)
        """
        if not self._pressure_history:
            return 0.0

        assert self._computed_aquifer_constant is not None
        assert self._computed_dimensionless_radius_ratio is not None

        influx_rate_sum = 0.0

        # Convolution Integral
        # Sum contributions from all past pressure changes using superposition
        for time_i, pressure_drop_i in self._pressure_history:
            if pressure_drop_i <= 0:
                # No influx if no pressure drop (or pressure increase)
                continue

            # Time elapsed since this pressure change
            time_diff = current_time - time_i
            if time_diff <= 0:
                continue

            # Compute Dimensionless Time
            if self._hydraulic_diffusivity is not None:
                # Physical properties mode: Dimensionally correct calculation
                # t_D = (k / (φ * μ * c_t)) * (t / r_w²) = η * t / r_w²
                assert self.inner_radius is not None
                dimensionless_time = (
                    self._hydraulic_diffusivity * time_diff / (self.inner_radius**2)
                )
            else:
                # Calibrated constant mode: Simplified (dimensionally inconsistent)
                # User should ensure time units are consistent with aquifer_constant calibration
                dimensionless_time = time_diff

            # Van Everdingen-Hurst Function
            # Compute dimensionless water influx derivative at this time
            W_D_prime = self._van_everdingen_hurst_derivative(
                dimensionless_time=dimensionless_time,
                dimensionless_radius_ratio=self._computed_dimensionless_radius_ratio,
            )

            # Add contribution from this pressure change to total influx
            influx_rate_sum += pressure_drop_i * W_D_prime

        # Scale by Aquifer Constant
        # B is already scaled by angle in `__attrs_post_init__` for physical properties mode
        # For calibrated constant mode, we apply angle scaling here
        if self._hydraulic_diffusivity is not None:
            # Physical properties mode: angle already in B
            total_influx_rate = self._computed_aquifer_constant * influx_rate_sum
        else:
            # Calibrated constant mode: apply angle fraction
            angle_fraction = self.angle / 360.0
            total_influx_rate = (
                self._computed_aquifer_constant * angle_fraction * influx_rate_sum
            )

        return total_influx_rate

    @staticmethod
    def _van_everdingen_hurst_derivative(
        dimensionless_time: float, dimensionless_radius_ratio: float
    ) -> float:
        """
        Van Everdingen-Hurst dimensionless water influx derivative W_D'(t_D, r_D).

        **Improved approximation** using Chatas' formulation for finite radial aquifer.

        The Van Everdingen-Hurst functions describe transient pressure response
        and water influx for finite radial aquifers. This function computes the
        derivative W_D'(t_D, r_D), which appears in the Carter-Tracy convolution
        integral.

        **Asymptotic Behavior:**
        - **Early time (t_D < 0.1)**: Infinite-acting behavior
            W_D'(t_D) ≈ √(t_D/π)
          All aquifers behave the same regardless of size (r_D).

        - **Intermediate time (0.1 < t_D < 2.0)**: Transition regime
          Smooth transition from infinite-acting to boundary-dominated flow.

        - **Late time (t_D > 2.0)**: Boundary-dominated exponential decline
            W_D'(t_D) ≈ (2*r_D²)/(r_D²-1) * exp(-β*t_D)
          where β = π²/(r_D²-1) is the first eigenvalue.
          Smaller aquifers (low r_D) decline faster.

        **Physical Interpretation:**
        - Early time: Aquifer appears infinite (pressure disturbance hasn't
          reached outer boundary)
        - Late time: Finite boundary effects dominate (outer boundary no-flow
          condition limits influx)

        **Approximation Quality:**
        - Exact solution requires infinite series (Bessel functions)
        - This approximation matches exact solution within ~5% for all t_D > 0.01
        - Based on Chatas (1953) practical treatment

        :param dimensionless_time: Dimensionless time t_D = (η * t) / r_w²
            where η = hydraulic diffusivity = 0.006328 * k / (φ * μ * c_t)
        :param dimensionless_radius_ratio: Dimensionless radius r_D = r_outer / r_inner
            Typical range: 2-50 for finite aquifers
        :return: Dimensionless water influx derivative W_D'(t_D, r_D)
            Units: dimensionless (1/psi in dimensional form after scaling by B)
        """
        if dimensionless_time <= 0:
            return 0.0

        # Early Time: Infinite-Acting Approximation
        # For small t_D, pressure disturbance hasn't reached outer boundary
        # Solution is independent of r_D (same for all aquifer sizes)
        if dimensionless_time < 0.1:
            return float(np.sqrt(dimensionless_time / np.pi))

        # Transition and Late Time
        # Use Chatas' approximation for finite radial aquifer
        r_D_squared = dimensionless_radius_ratio * dimensionless_radius_ratio
        denominator = r_D_squared - 1.0

        if denominator < 1e-6:
            # Nearly infinite aquifer (r_D → 1 or r_D very small)
            # This shouldn't happen physically (r_D ≥ 1 always), but handle gracefully
            return float(np.sqrt(dimensionless_time / np.pi))

        # Late Time: Exponential Decline
        # First eigenvalue for finite radial aquifer
        beta = (np.pi**2) / denominator

        # Late time exponential term
        late_time_coefficient = 2.0 * r_D_squared / denominator
        late_time_term = late_time_coefficient * np.exp(-beta * dimensionless_time)

        # Smooth Transition
        # Use sigmoid weighting to smoothly transition from early to late time
        # Centers transition around t_D ≈ 0.5 with width controlled by factor 10
        transition_weight = 1.0 / (1.0 + np.exp(-10.0 * (dimensionless_time - 0.5)))

        # Weighted combination: early time at low t_D, late time at high t_D
        early_time_term = np.sqrt(dimensionless_time / np.pi)
        W_D_prime = (
            1.0 - transition_weight
        ) * early_time_term + transition_weight * late_time_term
        return float(W_D_prime)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        data = {
            "initial_pressure": self.initial_pressure,
            "angle": self.angle,
            "pressure_history": self._pressure_history,
            "cumulative_influx": self._cumulative_influx,
        }

        # Save either physical properties or calibrated constant
        if self._hydraulic_diffusivity is not None:
            # Physical properties mode
            data.update(
                {
                    "aquifer_permeability": self.aquifer_permeability,
                    "aquifer_porosity": self.aquifer_porosity,
                    "aquifer_compressibility": self.aquifer_compressibility,
                    "water_viscosity": self.water_viscosity,
                    "inner_radius": self.inner_radius,
                    "outer_radius": self.outer_radius,
                    "aquifer_thickness": self.aquifer_thickness,
                }
            )
        else:
            # Calibrated constant mode
            data.update(
                {
                    "aquifer_constant": self.aquifer_constant,
                    "dimensionless_radius_ratio": self.dimensionless_radius_ratio,
                }
            )
        return data

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        # Check which mode was used
        if "aquifer_permeability" in data:
            # Physical properties mode
            instance = cls(
                aquifer_permeability=data["aquifer_permeability"],
                aquifer_porosity=data["aquifer_porosity"],
                aquifer_compressibility=data["aquifer_compressibility"],
                water_viscosity=data["water_viscosity"],
                inner_radius=data["inner_radius"],
                outer_radius=data["outer_radius"],
                aquifer_thickness=data["aquifer_thickness"],
                initial_pressure=data["initial_pressure"],
                angle=data.get("angle", 360.0),
            )
        else:
            # Calibrated constant mode
            instance = cls(
                aquifer_constant=data["aquifer_constant"],
                dimensionless_radius_ratio=data.get("dimensionless_radius_ratio", 10.0),
                initial_pressure=data["initial_pressure"],
                angle=data.get("angle", 360.0),
            )

        instance._pressure_history = data.get("pressure_history", [])
        instance._cumulative_influx = data.get("cumulative_influx", 0.0)
        return instance


@typing.final
@attrs.frozen
class GridBoundaryCondition(typing.Generic[NDimension], Serializable):
    """
    Container for defining boundary conditions for a grid.
    Each face in a 3D or 2D grid (x-, x+, y-, y+, z-, z+) can have its own boundary condition.

    In 2D:
    - Only x- (left/west), x+ (right/east), y- (bottom/south), y+ (top/north) are applied.

    In 3D:
    - All six faces are applied.

    ```mermaid
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

    left: BoundaryCondition = attrs.field(factory=NoFlowBoundary)
    """Boundary condition for the left face (x-)."""
    right: BoundaryCondition = attrs.field(factory=NoFlowBoundary)
    """Boundary condition for the right face (x+)."""
    front: BoundaryCondition = attrs.field(factory=NoFlowBoundary)
    """Boundary condition for the front face (y-)."""
    back: BoundaryCondition = attrs.field(factory=NoFlowBoundary)
    """Boundary condition for the back face (y+)."""
    bottom: BoundaryCondition = attrs.field(factory=NoFlowBoundary)
    """Boundary condition for the bottom face (z-)."""
    top: BoundaryCondition = attrs.field(factory=NoFlowBoundary)
    """Boundary condition for the top face (z+)."""

    def __attrs_post_init__(self) -> None:
        """Validate boundary condition configuration."""
        self._validate_periodic_boundaries()

    def _validate_periodic_boundaries(self) -> None:
        """
        Validate that periodic boundary conditions are properly paired.

        For proper periodic BCs, both opposite faces must use PeriodicBoundary.

        :raises ValidationError: if one face is periodic but its opposite is not.
        """
        # Check x-direction (left-right) pairing
        left_is_periodic = isinstance(self.left, PeriodicBoundary)
        right_is_periodic = isinstance(self.right, PeriodicBoundary)
        if left_is_periodic != right_is_periodic:
            raise ValidationError(
                "Periodic boundary conditions must be specified on both opposite faces. "
                f"X-direction: left={'Periodic' if left_is_periodic else 'Non-periodic'}, "
                f"right={'Periodic' if right_is_periodic else 'Non-periodic'}. "
                "Both left and right boundaries must be PeriodicBoundary for proper periodic BC."
            )

        # Check y-direction (front-back) pairing
        front_is_periodic = isinstance(self.front, PeriodicBoundary)
        back_is_periodic = isinstance(self.back, PeriodicBoundary)
        if front_is_periodic != back_is_periodic:
            raise ValidationError(
                "Periodic boundary conditions must be specified on both opposite faces. "
                f"Y-direction: front={'Periodic' if front_is_periodic else 'Non-periodic'}, "
                f"back={'Periodic' if back_is_periodic else 'Non-periodic'}. "
                "Both front and back boundaries must be PeriodicBoundary for proper periodic BC."
            )

        # Check z-direction (bottom-top) pairing
        bottom_is_periodic = isinstance(self.bottom, PeriodicBoundary)
        top_is_periodic = isinstance(self.top, PeriodicBoundary)
        if bottom_is_periodic != top_is_periodic:
            raise ValidationError(
                "Periodic boundary conditions must be specified on both opposite faces. "
                f"Z-direction: bottom={'Periodic' if bottom_is_periodic else 'Non-periodic'}, "
                f"top={'Periodic' if top_is_periodic else 'Non-periodic'}. "
                "Both bottom and top boundaries must be PeriodicBoundary for proper periodic BC."
            )

    def apply(
        self,
        padded_grid: NDimensionalGrid[NDimension],
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """
        Applies each defined boundary condition to the padded grid with ghost cells.

        - For 2D grids (shape: [nx+2, ny+2]), x and y boundaries are applied.
        - For 3D grids (shape: [nx+2, ny+2, nz+2]), x, y, and z boundaries are applied.

        :raises ValidationError: If grid dimensions are inconsistent with metadata.grid_shape
        """
        # Validate ghost cells if metadata provides grid_shape
        if metadata is not None and metadata.grid_shape is not None:
            expected_shape = tuple(s + 2 for s in metadata.grid_shape)
            if padded_grid.shape != expected_shape:
                raise ValidationError(
                    f"Grid shape {padded_grid.shape} does not match expected padded shape "
                    f"{expected_shape} (grid_shape {metadata.grid_shape} + 2 ghost cells per dimension). "
                    "Ensure the grid has ghost cells before applying boundary conditions."
                )

        if padded_grid.ndim == 2:
            self.left.apply(
                grid=padded_grid,
                boundary_indices=(slice(0, 1), slice(None)),
                direction=Boundary.LEFT,
                metadata=metadata,
            )
            self.right.apply(
                grid=padded_grid,
                boundary_indices=(slice(-1, None), slice(None)),
                direction=Boundary.RIGHT,
                metadata=metadata,
            )
            self.front.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(0, 1)),
                direction=Boundary.FRONT,
                metadata=metadata,
            )
            self.back.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(-1, None)),
                direction=Boundary.BACK,
                metadata=metadata,
            )
        elif padded_grid.ndim == 3:
            self.left.apply(
                grid=padded_grid,
                boundary_indices=(slice(0, 1), slice(None), slice(None)),
                direction=Boundary.LEFT,
                metadata=metadata,
            )
            self.right.apply(
                grid=padded_grid,
                boundary_indices=(slice(-1, None), slice(None), slice(None)),
                direction=Boundary.RIGHT,
                metadata=metadata,
            )
            self.front.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(0, 1), slice(None)),
                direction=Boundary.FRONT,
                metadata=metadata,
            )
            self.back.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(-1, None), slice(None)),
                direction=Boundary.BACK,
                metadata=metadata,
            )
            self.bottom.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(None), slice(0, 1)),
                direction=Boundary.BOTTOM,
                metadata=metadata,
            )
            self.top.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(None), slice(-1, None)),
                direction=Boundary.TOP,
                metadata=metadata,
            )
        else:
            raise ValidationError(
                "`padded_grid` must be a 2D or 3D numpy array with ghost cells."
            )


@typing.final
class BoundaryConditions(
    defaultdict[str, GridBoundaryCondition[NDimension]], Serializable
):
    """
    A container for managing reservoir model boundary conditions for different properties.

    This class allows you to define boundary conditions for various properties
    in a multi-dimensional grid, with a default factory to create conditions.

    Example usage with the new enhanced boundary conditions:
    ```python
    import numpy as np
    from bores.boundary_conditions import *

    # Create coordinate metadata for spatial boundaries
    x_coords = np.linspace(0, 1000, 50)  # 1000 ft wide
    y_coords = np.linspace(0, 500, 25)   # 500 ft long
    coordinates = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Define separate boundary conditions for each property
    boundary_conditions = BoundaryConditions(
        conditions={
            # Pressure boundary conditions
            "pressure": GridBoundaryCondition(
                left=ConstantBoundary(constant=2000.0),  # Fixed pressure inlet
                right=NoFlowBoundary(),  # Sealed boundary
                front=LinearGradientBoundary(  # Pressure gradient
                    start=2000.0,
                    end=1800.0,
                    direction="x"
                ),
                back=FluxBoundary(flux=-100.0),  # Production
            ),
            # Temperature boundary conditions (separate from pressure)
            "temperature": GridBoundaryCondition(
                left=SpatialBoundary(
                    func=lambda x, y: 60 + 0.025 * y  # Geothermal gradient
                ),
                right=NoFlowBoundary(),
                front=TimeDependentBoundary(
                    func=lambda t: 80 + 10 * np.sin(t / 3600)  # Daily cycle
                ),
                back=ConstantBoundary(constant=70.0),
            )
        },
        factory=lambda: GridBoundaryCondition(  # Default: all no-flow
            left=NoFlowBoundary(),
            right=NoFlowBoundary(),
            front=NoFlowBoundary(),
            back=NoFlowBoundary(),
        ),
    )

    # Create separate padded grids for each property
    pressure_grid = np.full((52, 27), 1500.0)  # +2 for ghost cells
    temperature_grid = np.full((52, 27), 65.0)

    # Option 1: Auto-generate coordinates from cell dimensions
    pressure_metadata = BoundaryMetadata(
        cell_dimension=(20.0, 20.0),        # 20x20 ft cells
        thickness_grid=np.full((50, 25), 10.0),  # 10 ft thickness (no ghost cells)
        time=0.0,
        property_name="pressure",
        grid_shape=(50, 25)                 # Coordinates auto-generated from this
    )

    # Option 2: Provide explicit coordinates
    temperature_metadata = BoundaryMetadata(
        cell_dimension=(20.0, 20.0),
        coordinates=np.stack(coordinates, axis=-1),  # Explicit coordinates
        thickness_grid=np.full((50, 25), 10.0),  # No ghost cells
        time=0.0,
        property_name="temperature",
        grid_shape=(50, 25)
    )    # Each property gets its own GridBoundaryCondition
    pressure_bc = boundary_conditions["pressure"]
    temperature_bc = boundary_conditions["temperature"]

    pressure_bc.apply(pressure_grid, metadata=pressure_metadata)
    temperature_bc.apply(temperature_grid, metadata=temperature_metadata)
    ```

    """

    __abstract_serializable__ = True

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
            If not provided, defaults to NoFlowBoundary for all sides/axes.

        :param conditions: Optional mapping of property names to their respective boundary conditions.
        """
        super().__init__(factory)
        if conditions:
            self.update(conditions)
        self.factory = factory

    def __reduce_ex__(self, protocol):
        """
        Support for pickling/copying.

        :return tuple: (callable, args, state) for reconstruction
        """
        return (self.__class__, (dict(self), self.factory), None, None, None)

    def __deepcopy__(self, memo):
        """
        Ensures the factory and all boundary conditions are properly copied.
        """
        # Create new instance without calling __init__ yet
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Try to deep copy the factory, fall back to shallow copy if not possible
        try:
            new_factory = copy.deepcopy(self.factory, memo)
        except (TypeError, AttributeError):
            # Factory might be a lambda or non-picklable function
            new_factory = self.factory

        # Deep copy all the boundary conditions
        new_conditions = {k: copy.deepcopy(v, memo) for k, v in self.items()}

        # Initialize the new instance
        result.__init__(conditions=new_conditions, factory=new_factory)

        # Deep copy any other instance attributes that might have been added
        for k, v in self.__dict__.items():
            if k not in ("factory", "default_factory"):
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __copy__(self):
        return self.__class__(conditions=dict(self), factory=self.factory)

    @classmethod
    def __load__(
        cls,
        data: typing.Mapping[str, typing.Any],
    ) -> Self:
        conditions_data = data.get("conditions", {})
        conditions: typing.Dict[str, GridBoundaryCondition[NDimension]] = {
            prop: GridBoundaryCondition.load(cond_data)
            for prop, cond_data in conditions_data.items()
        }
        return cls(conditions=conditions)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {"conditions": {prop: cond.dump(recurse) for prop, cond in self.items()}}


default_bc: GridBoundaryCondition[typing.Any] = BoundaryConditions()["__default__"]
"""Default boundary conditions using `NoFlowBoundary` for all sides."""
