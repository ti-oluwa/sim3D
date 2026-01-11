"""Boundary condition implementations for 2D/3D grids."""

from collections import defaultdict
import copy
import enum
import typing

import attrs
import numpy as np

from bores.errors import ValidationError
from bores.types import NDimension, NDimensionalGrid


__all__ = [
    "BoundaryDirection",
    "BoundaryMetadata",
    "BoundaryCondition",
    "get_neighbor_indices",
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
    "GridBoundaryCondition",
    "BoundaryConditions",
]


class BoundaryDirection(enum.Enum):
    """Enumeration for boundary directions."""

    X_MINUS = "left"
    """The negative X direction (left/west face)."""
    X_PLUS = "right"
    """The positive X direction (right/east face)."""
    Y_MINUS = "front"
    """The negative Y direction (bottom/south face)."""
    Y_PLUS = "back"
    """The positive Y direction (top/north face)."""
    Z_MINUS = "bottom"
    """The negative Z direction (bottom face)."""
    Z_PLUS = "top"
    """The positive Z direction (top face)."""


def get_neighbor_indices(
    boundary_indices: typing.Tuple[slice, ...], direction: BoundaryDirection
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
    from bores.boundary_conditions import get_neighbor_indices, BoundaryDirection

    # For left boundary (x=0 ghost cells)
    boundary_slice = (slice(0, 1), slice(None))
    neighbor_slice = get_neighbor_indices(boundary_slice, BoundaryDirection.X_MINUS)
    # Returns: (slice(1, 2), slice(None))  # x=1 interior cells

    # For right boundary (x=-1 ghost cells)
    boundary_slice = (slice(-1, None), slice(None))
    neighbor_slice = get_neighbor_indices(boundary_slice, BoundaryDirection.X_PLUS)
    # Returns: (slice(-2, -1), slice(None))  # x=-2 interior cells
    ```
    """
    neighbor_indices = list(boundary_indices)
    ndim = len(boundary_indices)

    # Validate Z-direction usage with grid dimensionality
    if direction in [BoundaryDirection.Z_MINUS, BoundaryDirection.Z_PLUS] and ndim < 3:
        raise ValidationError(
            f"Cannot use {direction.name} boundary direction with {ndim}D grid. "
            "Z-direction boundaries require 3D grids."
        )

    if direction == BoundaryDirection.X_MINUS:
        # Left boundary: neighbor is at x=1
        neighbor_indices[0] = slice(1, 2)
    elif direction == BoundaryDirection.X_PLUS:
        # Right boundary: neighbor is at x=-2
        neighbor_indices[0] = slice(-2, -1)
    elif direction == BoundaryDirection.Y_MINUS:
        # Front boundary: neighbor is at y=1
        neighbor_indices[1] = slice(1, 2)
    elif direction == BoundaryDirection.Y_PLUS:
        # Back boundary: neighbor is at y=-2
        neighbor_indices[1] = slice(-2, -1)
    elif direction == BoundaryDirection.Z_MINUS:
        # Bottom boundary: neighbor is at z=1
        neighbor_indices[2] = slice(1, 2)
    elif direction == BoundaryDirection.Z_PLUS:
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
        spatial_func=lambda x, y: 2000 + 0.1 * x - 0.05 * y
    )

    grid = np.zeros((52, 27))  # Grid with ghost cells
    boundary_indices = (slice(0, 1), slice(None))  # Left boundary

    spatial_bc.apply(
        grid=grid,
        boundary_indices=boundary_indices,
        direction=BoundaryDirection.X_MINUS,
        metadata=metadata_2d
    )
    ```
    """

    cell_dimension: typing.Optional[typing.Tuple[float, float]] = None
    """Physical dimensions of grid cells (dx, dy) in feet (or meters)."""
    thickness_grid: typing.Optional[NDimensionalGrid] = None
    """Grid of cell thickness values (same shape as original grid, NO ghost cells)."""
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
                            if self.thickness_grid.shape[2] == nz  # type: ignore
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
                            z_coords = np.linspace(-dz / 2, (nz + 0.5) * dz, nz + 2)
                    else:
                        # Fallback: assume uniform thickness equal to dx
                        dz = dx  # Default thickness
                        z_coords = np.linspace(-dz / 2, (nz + 0.5) * dz, nz + 2)
                else:
                    # Fallback: assume uniform thickness equal to dx
                    dz = dx  # Default thickness
                    z_coords = np.linspace(-dz / 2, (nz + 0.5) * dz, nz + 2)

                # Create meshgrid and stack
                xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
                coordinates = np.stack([xx, yy, zz], axis=-1)

            else:
                raise ValidationError(
                    f"Unsupported grid dimensionality: {len(self.grid_shape)}. Only 2D and 3D grids are supported."
                )

            object.__setattr__(self, "coordinates", coordinates)


@typing.runtime_checkable
class BoundaryCondition(typing.Protocol):
    """
    Protocol for defining boundary conditions with enhanced context.

    Each boundary condition type must implement an 'apply' method that receives
    the full grid, boundary indices, direction, and optional metadata.
    """

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """
        Apply the boundary condition to the specified grid region.

        :param grid: The full grid (including ghost cells)
        :param boundary_indices: Slice indices defining the boundary region
        :param direction: The boundary direction (left, right, etc.)
        :param metadata: Optional metadata for advanced boundary conditions
        """
        ...


@attrs.frozen
class NoFlowBoundary(typing.Generic[NDimension]):
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

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply no-flow boundary by copying values from neighboring cells."""
        neighbor_indices = get_neighbor_indices(boundary_indices, direction)
        grid[boundary_indices] = grid[neighbor_indices]


@attrs.frozen
class ConstantBoundary(typing.Generic[NDimension]):
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

    constant: typing.Any
    """The constant value to set at the boundary."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply constant boundary condition, preserving grid dtype."""
        # Preserve the grid's dtype when setting constant value
        grid[boundary_indices] = grid.dtype.type(self.constant)


@attrs.frozen
class VariableBoundary(typing.Generic[NDimension]):
    """
    Implements a variable boundary condition using a callable function.

    Allows for complex boundary conditions that depend on the grid state,
    location, direction, and metadata.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import VariableBoundary, BoundaryDirection, get_neighbor_indices

    def pressure_gradient_func(grid, boundary_indices, direction, metadata):
        # Example: Pressure increases with depth
        if direction == BoundaryDirection.Y_PLUS:  # Top boundary
            return np.full(grid[boundary_indices].shape, 1000.0)  # Low pressure
        elif direction == BoundaryDirection.Y_MINUS:  # Bottom boundary
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

    func: typing.Callable[
        [
            NDimensionalGrid[NDimension],
            typing.Tuple[slice, ...],
            BoundaryDirection,
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
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply variable boundary condition using the provided function."""
        result = self.func(grid, boundary_indices, direction, metadata)
        # Preserve the grid's dtype
        grid[boundary_indices] = np.asarray(result, dtype=grid.dtype)


DirichletBoundary = ConstantBoundary
"""Alias for `ConstantBoundary` representing Dirichlet boundary conditions."""


@attrs.frozen
class SpatialBoundary(typing.Generic[NDimension]):
    """
    Implements a spatial boundary condition using coordinate-based functions.

    Perfect for creating realistic geological boundaries like pressure gradients,
    temperature profiles, or depth-dependent properties.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import SpatialBoundary, BoundaryMetadata, GridBoundaryCondition

    # Create coordinate grid (1000 ft x 500 ft reservoir)
    x_coords = np.linspace(0, 1000, 51)  # 0 to 1000 ft
    y_coords = np.linspace(0, 500, 26)   # 0 to 500 ft
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    coordinates = np.stack([xx, yy], axis=-1)

    metadata = BoundaryMetadata(coordinates=coordinates)

    # Example 1: Linear pressure drop with distance
    pressure_gradient = SpatialBoundary(
        spatial_func=lambda x, y: 2000 - 0.5 * x  # 2000 psi at x=0, drops to 1500 at x=1000
    )

    # Example 2: Pressure increasing with depth
    depth_pressure = SpatialBoundary(
        spatial_func=lambda x, y: 2000 + 0.03 * y  # 2000 psi at y=0, 2015 psi at y=500
    )

    # Example 3: Radial pressure distribution from center
    radial_pressure = SpatialBoundary(
        spatial_func=lambda x, y: 2000 + 10 * np.sqrt((x-500)**2 + (y-250)**2)
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

    spatial_func: typing.Callable[..., np.ndarray]
    """Function that takes coordinate arrays and returns boundary values."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: BoundaryDirection,
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
                result = self.spatial_func(coords[..., 0], coords[..., 1])
            else:
                result = self.spatial_func(coords[..., 0])
            grid[boundary_indices] = np.asarray(result, dtype=grid.dtype)
        elif coords.ndim == 3:  # 3D case
            if coords.shape[-1] >= 3:
                result = self.spatial_func(
                    coords[..., 0], coords[..., 1], coords[..., 2]
                )
            elif coords.shape[-1] >= 2:
                result = self.spatial_func(coords[..., 0], coords[..., 1])
            else:
                result = self.spatial_func(coords[..., 0])
            grid[boundary_indices] = np.asarray(result, dtype=grid.dtype)
        else:
            # Fallback: flatten coordinates and apply function
            flat_coords = coords.reshape(-1, coords.shape[-1])
            if flat_coords.shape[1] >= 2:
                result = self.spatial_func(flat_coords[:, 0], flat_coords[:, 1])
            else:
                result = self.spatial_func(flat_coords[:, 0])
            grid[boundary_indices] = np.asarray(result, dtype=grid.dtype).reshape(
                coords.shape[:-1]
            )


@attrs.frozen
class TimeDependentBoundary(typing.Generic[NDimension]):
    """
    Implements a time-dependent boundary condition.

    Perfect for modeling cyclic injection, seasonal variations,
    or any boundary condition that changes over time.

    Example usage:
    ```python
    import numpy as np
    from bores.boundary_conditions import TimeDependentBoundary, BoundaryMetadata, GridBoundaryCondition

    # Example 1: Sinusoidal injection pressure (daily cycle)
    daily_cycle = TimeDependentBoundary(
        time_func=lambda t: 2000 + 200 * np.sin(2 * np.pi * t / 86400)  # 24-hour cycle
    )

    # Example 2: Linear pressure ramp-up
    pressure_ramp = TimeDependentBoundary(
        time_func=lambda t: min(1000 + 0.1 * t, 2500)  # Ramp from 1000 to 2500 psi
    )

    # Example 3: Exponential decay (well shut-in)
    pressure_decay = TimeDependentBoundary(
        time_func=lambda t: 2000 * np.exp(-t / 3600)  # Decay with 1-hour time constant
    )

    # Example 4: Step function (sudden pressure change)
    step_pressure = TimeDependentBoundary(
        time_func=lambda t: 2500 if t > 1800 else 1500  # Jump at 30 minutes
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

    time_func: typing.Callable[[float], float]
    """Function that takes time and returns boundary value."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply time-dependent boundary condition."""
        if metadata is None or metadata.time is None:
            raise ValidationError(f"{self.__class__.__name__} requires time metadata")

        value = self.time_func(metadata.time)
        # Preserve the grid's dtype
        grid[boundary_indices] = grid.dtype.type(value)


@attrs.frozen
class LinearGradientBoundary(typing.Generic[NDimension]):
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
        start_value=2500.0,      # High pressure at west (x=0)
        end_value=1500.0,        # Low pressure at east (x=1000)
        gradient_direction="x"
    )

    # Example 2: Pressure gradient with depth (shallow to deep)
    depth_pressure = LinearGradientBoundary(
        start_value=2000.0,      # Lower pressure at shallow depth (z=50)
        end_value=2300.0,        # Higher pressure at deep depth (z=150)
        gradient_direction="z"
    )

    # Example 3: Pressure gradient from north to south
    pressure_ns_gradient = LinearGradientBoundary(
        start_value=2200.0,      # High pressure at north (y=0)
        end_value=1800.0,        # Low pressure at south (y=500)
        gradient_direction="y"
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

    start_value: float
    """Value at the start of the gradient."""
    end_value: float
    """Value at the end of the gradient."""
    gradient_direction: typing.Literal["x", "y", "z"]
    """Direction of the gradient."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply linear gradient boundary condition."""
        if metadata is None or metadata.coordinates is None:
            raise ValidationError(
                f"{self.__class__.__name__} requires coordinate metadata"
            )

        coords = metadata.coordinates[boundary_indices]

        # Determine which coordinate axis to use for gradient
        if self.gradient_direction == "x":
            coord_values = coords[..., 0]
        elif self.gradient_direction == "y":
            coord_values = coords[..., 1]
        elif self.gradient_direction == "z":
            coord_values = coords[..., 2] if coords.shape[-1] > 2 else coords[..., 0]
        else:
            raise ValidationError(
                f"Invalid gradient direction: {self.gradient_direction}"
            )

        # Calculate gradient
        coord_min = np.min(coord_values)
        coord_max = np.max(coord_values)

        if coord_max == coord_min:
            # No gradient possible, use start value (preserve dtype)
            grid[boundary_indices] = grid.dtype.type(self.start_value)
        else:
            # Linear interpolation, preserve dtype
            normalized_coords = (coord_values - coord_min) / (coord_max - coord_min)
            result = self.start_value + normalized_coords * (
                self.end_value - self.start_value
            )
            grid[boundary_indices] = result.astype(grid.dtype, copy=False)


@attrs.frozen
class FluxBoundary(typing.Generic[NDimension]):
    """
    Implements a flux boundary condition (Neumann with physical interpretation).

    Sets a specified flux (rate of change) across the boundary, useful for
    modeling injection/production rates, heat flux, or mass transfer.

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
    water_injector = FluxBoundary(flux_value=100.0)  # 100 bbl/day/ft² injection

    # Example 2: Oil production (negative flux)
    oil_producer = FluxBoundary(flux_value=-50.0)    # 50 bbl/day/ft² production

    # Example 3: Heat injection
    heat_injector = FluxBoundary(flux_value=1000.0)  # 1000 BTU/hr/ft²

    # Example 4: Gas venting (very high production)
    gas_vent = FluxBoundary(flux_value=-500.0)       # 500 Mscf/day/ft²

    # Set up reservoir with injection/production boundaries (pressure property)
    pressure_boundary_config = GridBoundaryCondition(
        left=water_injector,  # West: water injection
        right=oil_producer,     # East: oil production
        front=FluxBoundary(flux_value=0.0),    # South: no flux
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

    flux_value: float
    """Flux value (positive for injection, negative for production)."""

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply flux boundary condition.

        The sign convention follows the outward normal:
        - For MINUS faces (left, front, bottom): outward normal points negative
        - For PLUS faces (right, back, top): outward normal points positive
        - Positive flux = flow INTO the domain (injection)
        - Negative flux = flow OUT OF the domain (production)

        Note: In this codebase, k=0 is the TOP (shallowest) layer, and k increases
        downward. So Z_PLUS (top) corresponds to k=0, and Z_MINUS (bottom) to k=-1.
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

        if direction in [BoundaryDirection.X_MINUS, BoundaryDirection.X_PLUS]:
            spacing = dx / 2.0
        elif direction in [BoundaryDirection.Y_MINUS, BoundaryDirection.Y_PLUS]:
            spacing = dy / 2.0
        elif direction in [BoundaryDirection.Z_MINUS, BoundaryDirection.Z_PLUS]:
            # For 3D, use thickness_grid if available for z-direction spacing
            # Note: k=0 is TOP, k=-1 is BOTTOM in this codebase
            if (
                metadata.thickness_grid is not None
                and metadata.thickness_grid.ndim == 3
            ):
                # Use average thickness at boundary layer for spacing
                # thickness_grid has no ghost cells, so we need to map boundary to interior
                if direction == BoundaryDirection.Z_MINUS:
                    # Bottom boundary (deepest) - use last layer thickness (k=-1)
                    avg_thickness = np.mean(metadata.thickness_grid[:, :, -1])
                else:  # Z_PLUS
                    # Top boundary (shallowest) - use first layer thickness (k=0)
                    avg_thickness = np.mean(metadata.thickness_grid[:, :, 0])
                spacing = avg_thickness / 2.0
            else:
                # Fallback to dx if no thickness info
                spacing = dx / 2.0
        else:
            spacing = dx / 2.0

        # Determine sign based on boundary direction
        # For MINUS faces, outward normal is negative, so we need to flip the sign
        # For PLUS faces, outward normal is positive
        # φ_boundary = φ_neighbor + flux * spacing * sign
        # where sign accounts for the direction of the outward normal
        if direction in [
            BoundaryDirection.X_MINUS,
            BoundaryDirection.Y_MINUS,
            BoundaryDirection.Z_MINUS,
        ]:
            # Outward normal points in negative direction
            # Positive flux (into domain) means gradient points inward (negative outward)
            sign = -1.0
        else:
            # Outward normal points in positive direction
            sign = 1.0

        # Apply flux boundary: dφ/dn = flux (outward normal convention)
        # φ_boundary = φ_neighbor + sign * flux * spacing
        # Preserve the grid's dtype
        result = neighbor_values + sign * self.flux_value * spacing
        grid[boundary_indices] = result.astype(grid.dtype, copy=False)


@attrs.frozen
class RobinBoundary(typing.Generic[NDimension]):
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
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply Robin boundary condition: α*φ + β*∂φ/∂n = γ

        Note: In this codebase, k=0 is the TOP (shallowest) layer, and k increases
        downward. So Z_PLUS (top) corresponds to k=0, and Z_MINUS (bottom) to k=-1.
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
        if direction in [BoundaryDirection.X_MINUS, BoundaryDirection.X_PLUS]:
            spacing = dx / 2.0
        elif direction in [BoundaryDirection.Y_MINUS, BoundaryDirection.Y_PLUS]:
            spacing = dy / 2.0
        else:
            # Z-direction: use thickness if available
            # Note: k=0 is TOP, k=-1 is BOTTOM in this codebase
            if (
                metadata.thickness_grid is not None
                and metadata.thickness_grid.ndim == 3
            ):
                if direction == BoundaryDirection.Z_MINUS:
                    # Bottom boundary (deepest) - use last layer thickness (k=-1)
                    avg_thickness = np.mean(metadata.thickness_grid[:, :, -1])
                else:  # Z_PLUS
                    # Top boundary (shallowest) - use first layer thickness (k=0)
                    avg_thickness = np.mean(metadata.thickness_grid[:, :, 0])
                spacing = avg_thickness / 2.0
            else:
                spacing = dx / 2.0

        # Direction sign for gradient (outward normal convention)
        if direction in [
            BoundaryDirection.X_MINUS,
            BoundaryDirection.Y_MINUS,
            BoundaryDirection.Z_MINUS,
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


@attrs.frozen
class PeriodicBoundary(typing.Generic[NDimension]):
    """
    Implements a periodic boundary condition.

    Links opposite boundaries so that flow/values wrap around,
    useful for modeling repeating geological patterns or infinite domains.

    Note: For proper periodic BCs, both opposite faces must use PeriodicBoundary.

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

    def apply(
        self,
        *,
        grid: NDimensionalGrid[NDimension],
        boundary_indices: typing.Tuple[slice, ...],
        direction: BoundaryDirection,
        metadata: typing.Optional[BoundaryMetadata] = None,
    ) -> None:
        """Apply periodic boundary by copying from opposite interior boundary."""
        # Get indices of the opposite interior cells
        opposite_indices = list(boundary_indices)

        if direction == BoundaryDirection.X_MINUS:
            # Left boundary copies from right interior (second-to-last)
            opposite_indices[0] = slice(-2, -1)
        elif direction == BoundaryDirection.X_PLUS:
            # Right boundary copies from left interior (second from start)
            opposite_indices[0] = slice(1, 2)
        elif direction == BoundaryDirection.Y_MINUS:
            # Front boundary copies from back interior
            opposite_indices[1] = slice(-2, -1)
        elif direction == BoundaryDirection.Y_PLUS:
            # Back boundary copies from front interior
            opposite_indices[1] = slice(1, 2)
        elif direction == BoundaryDirection.Z_MINUS:
            # Bottom boundary copies from top interior
            if len(boundary_indices) < 3:
                raise ValidationError(
                    f"Cannot apply {direction.name} to {len(boundary_indices)}D grid"
                )
            opposite_indices[2] = slice(-2, -1)
        elif direction == BoundaryDirection.Z_PLUS:
            # Top boundary copies from bottom interior
            if len(boundary_indices) < 3:
                raise ValidationError(
                    f"Cannot apply {direction.name} to {len(boundary_indices)}D grid"
                )
            opposite_indices[2] = slice(1, 2)

        grid[boundary_indices] = grid[tuple(opposite_indices)]


NeumannBoundary = FluxBoundary
"""Alias for `FluxBoundary` representing Neumann boundary conditions (flux-based)."""


@attrs.frozen
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
                direction=BoundaryDirection.X_MINUS,
                metadata=metadata,
            )
            self.right.apply(
                grid=padded_grid,
                boundary_indices=(slice(-1, None), slice(None)),
                direction=BoundaryDirection.X_PLUS,
                metadata=metadata,
            )
            self.front.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(0, 1)),
                direction=BoundaryDirection.Y_MINUS,
                metadata=metadata,
            )
            self.back.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(-1, None)),
                direction=BoundaryDirection.Y_PLUS,
                metadata=metadata,
            )
        elif padded_grid.ndim == 3:
            self.left.apply(
                grid=padded_grid,
                boundary_indices=(slice(0, 1), slice(None), slice(None)),
                direction=BoundaryDirection.X_MINUS,
                metadata=metadata,
            )
            self.right.apply(
                grid=padded_grid,
                boundary_indices=(slice(-1, None), slice(None), slice(None)),
                direction=BoundaryDirection.X_PLUS,
                metadata=metadata,
            )
            self.front.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(0, 1), slice(None)),
                direction=BoundaryDirection.Y_MINUS,
                metadata=metadata,
            )
            self.back.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(-1, None), slice(None)),
                direction=BoundaryDirection.Y_PLUS,
                metadata=metadata,
            )
            self.bottom.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(None), slice(0, 1)),
                direction=BoundaryDirection.Z_MINUS,
                metadata=metadata,
            )
            self.top.apply(
                grid=padded_grid,
                boundary_indices=(slice(None), slice(None), slice(-1, None)),
                direction=BoundaryDirection.Z_PLUS,
                metadata=metadata,
            )
        else:
            raise ValidationError(
                "`padded_grid` must be a 2D or 3D numpy array with ghost cells."
            )


class BoundaryConditions(defaultdict[str, GridBoundaryCondition[NDimension]]):
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
                    start_value=2000.0,
                    end_value=1800.0,
                    gradient_direction="x"
                ),
                back=FluxBoundary(flux_value=-100.0),  # Production
            ),
            # Temperature boundary conditions (separate from pressure)
            "temperature": GridBoundaryCondition(
                left=SpatialBoundary(
                    spatial_func=lambda x, y: 60 + 0.025 * y  # Geothermal gradient
                ),
                right=NoFlowBoundary(),
                front=TimeDependentBoundary(
                    time_func=lambda t: 80 + 10 * np.sin(t / 3600)  # Daily cycle
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
        Custom deep copy implementation for better control.

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


default_bc = BoundaryConditions()["__default__"]
"""Default boundary conditions using `NoFlowBoundary` for all sides."""
