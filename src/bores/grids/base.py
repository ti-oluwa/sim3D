import typing
import warnings

import attrs
import numba  # type: ignore[import-untyped]
import numpy as np
from typing_extensions import Self

from bores._precision import get_dtype
from bores.errors import ValidationError
from bores.serialization import Serializable
from bores.types import (
    ArrayLike,
    NDimension,
    NDimensionalGrid,
    Orientation,
    ThreeDimensionalGrid,
    ThreeDimensions,
    TwoDimensionalGrid,
    TwoDimensions,
)

__all__ = [
    "CapillaryPressureGrids",
    "RateGrids",
    "RelativeMobilityGrids",
    "apply_structural_dip",
    "array",
    "build_depth_grid",
    "build_elevation_grid",
    "build_layered_grid",
    "build_uniform_grid",
    "coarsen_grid",
    "coarsen_permeability_grids",
    "depth_grid",
    "elevation_grid",
    "flatten_multilayer_grid_to_surface",
    "get_pad_mask",
    "layered_grid",
    "pad_grid",
    "uniform_grid",
    "unpad_grid",
]


def array(obj: typing.Any, **kwargs: typing.Any):
    """
    Wrapper around np.array to enforce global dtype.

    :param obj: Object to convert to numpy array
    :param kwargs: Additional keyword arguments for `np.array`
    :return: return value of `np.array`
    """
    kwargs.setdefault("dtype", get_dtype())
    return np.array(obj, **kwargs)


def build_uniform_grid(
    grid_shape: NDimension,
    value: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Constructs a N-Dimensional uniform grid with the specified initial value.

    :param grid_shape: Tuple of number of cells in all directions (x, y, z).
    :param value: Initial value to fill the grid with
    :return: Numpy array representing the grid
    """
    return np.full(  # type: ignore
        grid_shape,
        fill_value=value,
        dtype=get_dtype(),
        order="C",
    )


uniform_grid = build_uniform_grid  # Alias for convenience


def build_layered_grid(
    grid_shape: NDimension,
    layer_values: ArrayLike[float],
    orientation: typing.Union[Orientation, typing.Literal["x", "y", "z"]],
) -> NDimensionalGrid[NDimension]:
    """
    Constructs a N-Dimensional layered grid with specified layer values.

    :param grid_shape: Tuple of number of cells in x, y, and z directions (cell_count_x, cell_count_y, cell_count_z)
    :param orientation: Direction or axis along which layers are defined ('x', 'y', or 'z')
    :param layer_values: Values for each layer (must match number of layers).
        The number of values should match the number of cells in that direction.
        If the grid NDimension is (50, 30, 10) and orientation is 'horizontal',
        then values should have exactly 50 values.
        If orientation is 'vertical', then values should have exactly 30 values.

    :return: N-Dimensional numpy array representing the grid
    """
    if len(layer_values) < 1:
        raise ValidationError("At least one layer value must be provided.")

    orientation = (
        Orientation(orientation) if isinstance(orientation, str) else orientation
    )
    dtype = get_dtype()
    layered_grid = build_uniform_grid(grid_shape=grid_shape, value=0.0)
    if orientation == Orientation.X:  # Layering along x-axis
        if len(layer_values) != grid_shape[0]:
            raise ValidationError(
                "Number of layer values must match number of cells in x direction."
            )

        for i, layer_value in enumerate(layer_values):
            layered_grid[i, :, :] = layer_value
        return layered_grid.astype(dtype, copy=False)

    elif orientation == Orientation.Y:  # Layering along y-axis
        if len(layer_values) != grid_shape[1]:
            raise ValidationError(
                "Number of layer values must match number of cells in y direction."
            )

        for j, layer_value in enumerate(layer_values):
            layered_grid[:, j, :] = layer_value
        return layered_grid.astype(dtype, copy=False)

    elif orientation == Orientation.Z:  # Layering along z-axis
        if len(grid_shape) != 3:
            raise ValidationError(
                "Grid dimension must be N-Dimensional for z-direction layering."
            )

        if len(layer_values) != grid_shape[2]:
            raise ValidationError(
                "Number of layer values must match number of cells in z direction."
            )

        for k, layer_value in enumerate(layer_values):
            layered_grid[:, :, k] = layer_value
        return layered_grid.astype(dtype, copy=False)

    raise ValidationError(
        "Invalid layering direction. Must be one of 'x', 'y', or 'z'."
    )


layered_grid = build_layered_grid  # Alias for convenience


@numba.njit(cache=True)
def _compute_elevation_downward(
    thickness_grid: NDimensionalGrid[NDimension],
    dtype: np.typing.DTypeLike,
    datum: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Compute elevation grid in downward direction (depth from top).

    :param thickness_grid: 3D array of cell thicknesses (ft)
    :param dtype: NumPy dtype for array allocation
    :param datum: Reference elevation/depth for the bottom/top of the grid (ft).
    :return: 3D elevation grid (ft)
    """
    _, _, nz = thickness_grid.shape
    elevation_grid = np.zeros_like(thickness_grid, dtype=dtype)

    # Start from top layer
    elevation_grid[:, :, 0] = thickness_grid[:, :, 0] / 2
    for k in range(1, nz):
        elevation_grid[:, :, k] = (
            elevation_grid[:, :, k - 1]
            + thickness_grid[:, :, k - 1] / 2
            + thickness_grid[:, :, k] / 2
        )

    return elevation_grid + datum  # type: ignore


@numba.njit(cache=True)
def _compute_elevation_upward(
    thickness_grid: NDimensionalGrid[NDimension],
    dtype: np.typing.DTypeLike,
    datum: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Compute elevation grid in upward direction (elevation from bottom).

    :param thickness_grid: 3D array of cell thicknesses (ft)
    :param dtype: NumPy dtype for array allocation
    :param datum: Reference elevation/depth for the bottom/top of the grid (ft).
    :return: 3D elevation grid (ft)
    """
    _, _, nz = thickness_grid.shape
    elevation_grid = np.zeros_like(thickness_grid, dtype=dtype)

    # Start from bottom layer
    elevation_grid[:, :, -1] = thickness_grid[:, :, -1] / 2
    for k in range(nz - 2, -1, -1):
        elevation_grid[:, :, k] = (
            elevation_grid[:, :, k + 1]
            + thickness_grid[:, :, k + 1] / 2
            + thickness_grid[:, :, k] / 2
        )

    return elevation_grid + datum  # type: ignore


def _build_elevation_grid(
    thickness_grid: NDimensionalGrid[NDimension],
    direction: typing.Literal["downward", "upward"] = "downward",
    datum: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Convert a cell thickness (height) grid into an absolute elevation grid (cell center z-coordinates).

    The elevation grid is generated based on the thickness of each cell, starting from the top or bottom
    of the reservoir, depending on the specified direction.

    :param thickness_grid: N-dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :param direction: Direction to generate the elevation grid.
        Can be "downward" (from top to bottom) or "upward" (from bottom to top).
        "downward" basically gives a depth grid from the top of the reservoir.
    :param datum: Reference elevation/depth for the bottom/top of the grid (ft).
    :return: N-dimensional numpy array representing the elevation of each cell in the reservoir (ft).
    """
    if direction not in {"downward", "upward"}:
        raise ValidationError("direction must be 'downward' or 'upward'")

    dtype = get_dtype()

    if direction == "downward":
        return _compute_elevation_downward(thickness_grid, dtype=dtype, datum=datum)
    return _compute_elevation_upward(thickness_grid, dtype=dtype, datum=datum)


def build_elevation_grid(
    thickness_grid: NDimensionalGrid[NDimension], datum: float = 0.0
) -> NDimensionalGrid[NDimension]:
    """
    Build an upward elevation grid from a thickness grid.

    Elevation is measured upward from a datum level, where positive elevation
    means above the datum. The datum typically represents:
    - Sea level (datum = 0)
    - Base of reservoir (datum = base elevation)
    - Reference surface (datum = reference elevation)

    :param thickness_grid: N-dimensional numpy array representing the thickness
        of each cell in the reservoir (ft).
    :param datum: Reference elevation for the bottom of the grid (ft). Elevation of the bottom surface of the grid.
        - datum = 0: Bottom layer starts at elevation 0 (e.g., sea level)
        - datum > 0: Bottom layer starts above the reference (e.g., above sea level)
        - datum < 0: Bottom layer starts below the reference (e.g., subsea)
        Default is 0.0 (bottom layer at reference level).
        - Negative if bottom is subsea (most common)
        - Positive if bottom is above sea level
        - Zero if bottom is exactly at sea level
    :return: N-dimensional numpy array representing the elevation of each cell
        center in the reservoir (ft), measured upward from datum.

    Example:
    ```python
    # Reservoir from -2000 to -1000 ft (subsea)
    thickness = np.full((10, 10, 20), 50.0)  # 20 layers, 50 ft each

    # Datum at base of reservoir
    elev_grid = build_elevation_grid(thickness, datum=-2000.0)
    # elev_grid[0,0,-1] = -1975.0 ft (center of bottom 50-ft layer)
    # elev_grid[0,0,0] = -1025.0 ft  (center of top layer)

    # Datum at sea level (bottom at -1000 ft from top)
    elev_grid = build_elevation_grid(thickness, datum=-1000.0)
    # elev_grid[0,0,-1] = -975.0 ft
    # elev_grid[0,0,0] = -25.0 ft
    ```

    Notes:
        - Elevation increases upward (k=-1 is lowest, k=0 is highest)
        - For depth (downward-positive), use `build_depth_grid()` instead
        - Datum represents the elevation of the BOTTOM of the grid
    """
    return _build_elevation_grid(thickness_grid, direction="upward", datum=datum)


elevation_grid = build_elevation_grid  # Alias for convenience


def build_depth_grid(
    thickness_grid: NDimensionalGrid[NDimension], datum: float = 0.0
) -> NDimensionalGrid[NDimension]:
    """
    Build a downward depth grid from a thickness grid.

    Depth is measured downward from a datum level, where positive depth means
    below the datum. The datum typically represents:
    - Sea level (datum = 0)
    - Ground surface (datum = surface elevation)
    - Top of reservoir (datum = top depth)

    :param thickness_grid: N-dimensional numpy array representing the thickness
        of each cell in the reservoir (ft).
    :param datum: Reference depth for the top of the grid (ft). Depth of the top surface of the grid.
        - datum = 0: Top layer starts at depth 0 (e.g., sea level)
        - datum > 0: Top layer starts below the reference (e.g., subsea depth)
        - datum < 0: Top layer starts above the reference (e.g., above sea level)
        Default is 0.0 (top layer at reference level).
        - Always positive (depth increases downward)
        - datum = 1000.0 means top is at 1000 ft depth
    :return: N-dimensional numpy array representing the depth of each cell
        center in the reservoir (ft), measured downward from datum.

    Example:
    ```python
    # Reservoir 1000-2000 ft subsea depth
    thickness = np.full((10, 10, 20), 50.0)  # 20 layers, 50 ft each

    # Option 1: Datum at sea level, specify top depth
    depth_grid = build_depth_grid(thickness, datum=1000.0)
    # depth_grid[0,0,0] = 1025.0 ft  (center of first 50-ft layer)
    # depth_grid[0,0,-1] = 1975.0 ft (center of last layer)

    # Option 2: Datum at top of reservoir
    depth_grid = build_depth_grid(thickness, datum=0.0)
    # depth_grid[0,0,0] = 25.0 ft  (relative to top)
    # depth_grid[0,0,-1] = 975.0 ft
    ```

    Notes:
        - Depth increases downward (k=0 is shallowest, k=-1 is deepest)
        - For elevation (upward-positive), use `build_elevation_grid()` instead
        - Datum represents the depth/elevation of the TOP of the grid
    """
    return _build_elevation_grid(thickness_grid, direction="downward", datum=datum)


depth_grid = build_depth_grid  # Alias for convenience


@numba.njit(parallel=True, cache=True)
def _apply_dip_upward(
    dipped_elevation_grid: NDimensionalGrid[NDimension],
    grid_dimensions: typing.Tuple[int, int],
    cell_dimensions: typing.Tuple[float, float],
    dip_components: typing.Tuple[float, float, float],
) -> NDimensionalGrid[NDimension]:
    """
    Apply structural dip for upward elevation convention (parallel).

    Each (i,j) column is processed independently, allowing parallelization.

    :param dipped_elevation_grid: Grid to modify in-place
    :param grid_dimensions: (nx, ny) - number of cells in x and y directions
    :param cell_dimensions: (cell_size_x, cell_size_y) - cell sizes in feet
    :param dip_components: (dx_component, dy_component, tan_dip_angle) - pre-computed dip parameters
    :return: Modified elevation grid
    """
    nx, ny = grid_dimensions
    cell_size_x, cell_size_y = cell_dimensions
    dx_component, dy_component, tan_dip_angle = dip_components

    for i in numba.prange(nx):  # type: ignore  # Parallel outer loop
        for j in range(ny):
            x_distance = i * cell_size_x
            y_distance = j * cell_size_y
            distance_along_dip = (x_distance * dx_component) + (
                y_distance * dy_component
            )
            dip_offset = distance_along_dip * tan_dip_angle
            # Upward: moving in dip direction decreases elevation
            dipped_elevation_grid[i, j, :] -= dip_offset

    return dipped_elevation_grid


@numba.njit(parallel=True, cache=True)
def _apply_dip_downward(
    dipped_elevation_grid: NDimensionalGrid[NDimension],
    grid_dimensions: typing.Tuple[int, int],
    cell_dimensions: typing.Tuple[float, float],
    dip_components: typing.Tuple[float, float, float],
) -> NDimensionalGrid[NDimension]:
    """
    Apply structural dip for downward depth convention (parallel).

    Each (i,j) column is processed independently, allowing parallelization.

    :param dipped_elevation_grid: Grid to modify in-place
    :param grid_dimensions: (nx, ny) - number of cells in x and y directions
    :param cell_dimensions: (cell_size_x, cell_size_y) - cell sizes in feet
    :param dip_components: (dx_component, dy_component, tan_dip_angle) - pre-computed dip parameters
    :return: Modified elevation grid
    """
    nx, ny = grid_dimensions
    cell_size_x, cell_size_y = cell_dimensions
    dx_component, dy_component, tan_dip_angle = dip_components

    for i in numba.prange(nx):  # type: ignore  # Parallel outer loop
        for j in range(ny):
            x_distance = i * cell_size_x
            y_distance = j * cell_size_y
            distance_along_dip = (x_distance * dx_component) + (
                y_distance * dy_component
            )
            dip_offset = distance_along_dip * tan_dip_angle
            # Downward: moving in dip direction increases depth
            dipped_elevation_grid[i, j, :] += dip_offset

    return dipped_elevation_grid


def apply_structural_dip(
    elevation_grid: NDimensionalGrid[NDimension],
    cell_dimension: typing.Tuple[float, float],
    elevation_direction: typing.Literal["downward", "upward"],
    dip_angle: float,
    dip_azimuth: float,
) -> NDimensionalGrid[NDimension]:
    """
    Apply structural dip to a base elevation grid using azimuth convention.

    The dip is applied by adding a planar gradient in the specified azimuth direction.
    The dip angle represents the angle of the reservoir surface from horizontal.

    ---

    ## 🧭 **Azimuth Convention:**
    ```
    Grid Coordinate System:
    North (0°/360°)
         ↑ (+y)
         |
         |
    West ←─────┼─────→ East (90°)
    (270°)  |    (+x)
         |
         ↓ (-y)
    South (180°)
    ```

    Azimuth Examples:
    - 0° (North): Dips toward North
    - 90° (East): Dips toward East
    - 180° (South): Dips toward South
    - 270° (West): Dips toward West
    - 45° (NE): Dips toward Northeast

    The surface tilts DOWN in the azimuth direction, meaning elevation
    DECREASES in that direction (or depth INCREASES for downward convention).

    :param elevation_grid: Base flat elevation grid (shape: [nx, ny, nz])
    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet
    :param elevation_direction: Whether elevation is "upward" (elevation) or "downward" (depth)
    :param dip_angle: Dip angle in degrees (0-90)
    :param dip_azimuth: Dip azimuth in degrees (0-360), measured clockwise from North
    :return: Elevation grid with structural dip applied
    """
    if elevation_direction not in {"downward", "upward"}:
        raise ValidationError("`elevation_direction` must be 'downward' or 'upward'")

    if not (0.0 <= dip_angle <= 90.0):
        raise ValidationError("`dip_angle` must be between 0 and 90 degrees")

    if not (0.0 <= dip_azimuth < 360.0):
        raise ValidationError("`dip_azimuth` must be between 0 and 360 degrees")

    dtype = get_dtype()
    dipped_elevation_grid = elevation_grid.copy().astype(dtype, copy=False)
    dip_angle_radians = np.radians(dip_angle, dtype=dtype)
    dip_azimuth_radians = np.radians(dip_azimuth, dtype=dtype)

    grid_shape = elevation_grid.shape
    nx, ny = grid_shape[0], grid_shape[1]

    # Convert azimuth to directional components
    # Azimuth: 0° = North (+y), 90° = East (+x), 180° = South (-y), 270° = West (-x)
    dx_component = np.sin(dip_azimuth_radians)  # Positive = East
    dy_component = np.cos(dip_azimuth_radians)  # Positive = North
    tan_dip_angle = np.tan(dip_angle_radians)

    grid_dimensions = (nx, ny)
    dip_components = (dx_component, dy_component, tan_dip_angle)

    if elevation_direction == "upward":
        return _apply_dip_upward(
            dipped_elevation_grid=dipped_elevation_grid,
            grid_dimensions=grid_dimensions,
            cell_dimensions=cell_dimension,
            dip_components=dip_components,
        )
    return _apply_dip_downward(
        dipped_elevation_grid=dipped_elevation_grid,
        grid_dimensions=grid_dimensions,
        cell_dimensions=cell_dimension,
        dip_components=dip_components,
    )


def pad_grid(
    grid: NDimensionalGrid[NDimension], pad_width: int = 1
) -> NDimensionalGrid[NDimension]:
    """
    Pads a N-Dimensional grid with the edge values to create a border around the grid.

    This is useful for finite difference methods where boundary conditions are applied.

    :param grid: N-Dimensional numpy array representing the grid to be padded
    :param pad_width: Width of the padding to be applied on all sides of the grid
    :return: Padded N-Dimensional numpy array
    """
    return np.pad(grid, pad_width=pad_width, mode="edge")  # type: ignore[return-value]


@numba.njit(cache=True)
def get_pad_mask(grid_shape: typing.Tuple[int, ...], pad_width: int = 1) -> np.ndarray:
    """
    Generate a boolean mask for the padded grid indicating the padded regions.

    :param grid_shape: Shape of the original grid before padding
    :param pad_width: Width of the padding applied on all sides of the grid
    :return: Boolean mask numpy array where True indicates padded regions
    """
    padded_shape = tuple(dim + 2 * pad_width for dim in grid_shape)
    mask = np.zeros(padded_shape, dtype=bool)

    # Set padded regions to True
    slices = tuple(
        slice(0, pad_width)
        if i == 0
        else slice(-pad_width, None)
        if i == 1
        else slice(pad_width, -pad_width)
        for i, dim in enumerate(padded_shape)
    )
    mask[slices] = True
    return mask


@numba.njit(cache=True)
def unpad_grid(
    grid: NDimensionalGrid[NDimension], pad_width: int = 1
) -> NDimensionalGrid[NDimension]:
    """
    Remove padding from a N-Dimensional grid.

    :param grid: Padded N-Dimensional numpy array representing the grid
    :param pad_width: Width of the padding to be removed from all sides of the grid
    :return: N-Dimensional numpy array with padding removed
    """
    # Build slices list explicitly (generator expressions not supported in Numba)
    ndim = grid.ndim
    if ndim == 2:
        unpadded_grid = grid[pad_width:-pad_width, pad_width:-pad_width]
    elif ndim == 3:
        unpadded_grid = grid[
            pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width
        ]
    else:
        raise ValueError(
            f"Unsupported grid dimension: {ndim}. Only 2D and 3D grids are supported."
        )

    return unpadded_grid  # type: ignore


def coarsen_grid(
    data: np.ndarray,
    batch_size: typing.Tuple[int, ...],
    method: typing.Literal["mean", "sum", "max", "min", "harmonic"] = "mean",
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Coarsen (downsample) a 2D or 3D grid by aggregating blocks of adjacent cells.

    Pads the grid if necessary to make dimensions divisible by `batch_size`.

    :param data: 2D or 3D numpy array to coarsen. Shape can be (nx, ny) or (nx, ny, nz).
    :param batch_size: Tuple of ints representing the coarsening factor along each dimension.
        Length must match `data.ndim`. Example: (2,2) for 2D, (2,2,2) for 3D.
    :param method: Aggregation method to use on each block.
        - 'mean': Arithmetic mean (for porosity, saturation)
        - 'sum': Sum (for total volume, pore volume)
        - 'max': Maximum value in block
        - 'min': Minimum value in block
        - 'harmonic': Harmonic mean (WARNING: only valid for isotropic averaging)
    :param epsilon: Small value to avoid division by zero in harmonic mean (default: 1e-10)
    :return: Coarsened numpy array.
    :raises ValidationError: if `batch_size` length does not match `data.ndim` or if method is unsupported.

    Note:
        For permeability coarsening, use `coarsen_permeability_grids()` instead, which
        applies direction-appropriate averaging (harmonic in flow direction, arithmetic
        perpendicular).

    Example:
    ```python
    data2d = np.arange(16, dtype=float).reshape(4,4)
    coarsen_grid(data2d, batch_size=(2,2))
    # array([[ 2.5,  4.5],
    #        [10.5, 12.5]])

    data3d = np.arange(64, dtype=float).reshape(4,4,4)
    coarsen_grid(data3d, batch_size=(2,2,2), method='max')
    # array([[[ 5.,  7.],
    #         [13., 15.]],
    #        [[21., 23.],
    #         [29., 31.]]])
    ```
    """
    if len(batch_size) != data.ndim:
        raise ValidationError(
            f"batch_size length {len(batch_size)} must match data.ndim {data.ndim}"
        )

    # Validate method
    valid_methods = ("mean", "sum", "max", "min", "harmonic")
    if method not in valid_methods:
        raise ValidationError(
            f"Unsupported method '{method}'. Must be one of {valid_methods}"
        )

    # Calculate padding needed
    pad_width = []
    for dim, b in zip(data.shape, batch_size):
        remainder = dim % b
        if remainder == 0:
            pad_width.append((0, 0))
        else:
            pad_width.append((0, b - remainder))

    # Pad with appropriate value based on method
    # Use NaN for methods that support it (will be ignored in aggregation)
    if method in ("mean", "max", "min", "harmonic"):
        pad_value = np.nan
    elif method == "sum":
        pad_value = 0.0

    data_padded = np.pad(
        data, pad_width=pad_width, mode="constant", constant_values=pad_value
    )

    # Reshape to group blocks along each dimension
    # E.g., (100, 50) with batch (2, 5) → (50, 2, 10, 5) → aggregate over axes (1, 3)
    reshape_shape = []
    for dim, b in zip(data_padded.shape, batch_size):
        reshape_shape.extend([dim // b, b])

    data_reshaped = data_padded.reshape(reshape_shape)

    # Axes to aggregate over: every second axis (the block dimensions)
    agg_axes = tuple(range(1, data_reshaped.ndim, 2))

    dtype = get_dtype()

    # Apply aggregation
    if method == "mean":
        coarsened = np.nanmean(data_reshaped, axis=agg_axes).astype(dtype)

    elif method == "sum":
        coarsened = data_reshaped.sum(axis=agg_axes, dtype=dtype)

    elif method == "max":
        coarsened = np.nanmax(data_reshaped, axis=agg_axes).astype(dtype)

    elif method == "min":
        coarsened = np.nanmin(data_reshaped, axis=agg_axes).astype(dtype)

    elif method == "harmonic":
        # Harmonic mean: H = n / sum(1/x_i)
        # Add epsilon to avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            reciprocals = 1.0 / (data_reshaped + epsilon)
            reciprocals = np.where(np.isnan(data_reshaped), np.nan, reciprocals)

            # Count non-NaN values per block
            counts = np.sum(~np.isnan(data_reshaped), axis=agg_axes)

            # Sum of reciprocals, ignoring NaNs
            sum_reciprocals = np.nansum(reciprocals, axis=agg_axes)

            # Harmonic mean = n / sum(1/x)
            coarsened = counts / sum_reciprocals

            # Handle edge cases
            coarsened = np.where(counts == 0, np.nan, coarsened)
            coarsened = np.where(np.isinf(coarsened), 0.0, coarsened)
            coarsened = coarsened.astype(dtype)

    return coarsened


def _coarsen_2d_permeability_grids(
    kx: TwoDimensionalGrid,
    ky: TwoDimensionalGrid,
    batch_size: TwoDimensions,
    epsilon: float = 1e-10,
) -> typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid]:
    """
    Coarsen 2D permeability grids using direction-appropriate averaging.

    Uses Cardwell-Parsons averaging:
    - k_x: harmonic mean in x-direction, arithmetic mean in y-direction
    - k_y: harmonic mean in y-direction, arithmetic mean in x-direction

    This preserves the effective permeability for flow in each direction.

    :param kx: X-direction permeability grid (mD), shape (nx, ny)
    :param ky: Y-direction permeability grid (mD), shape (nx, ny)
    :param batch_size: Coarsening factors (bx, by) for each direction
    :param epsilon: Small value to avoid division by zero (default: 1e-10)
    :return: Tuple of (coarsened_kx, coarsened_ky)
    :raises ValidationError: If grids have different shapes or batch_size is invalid

    Example:
    ```python
    # 4x4 grid with layered permeability
    kx = np.array([[100, 100, 100, 100],
                    [  1,   1,   1,   1],
                    [100, 100, 100, 100],
                    [  1,   1,   1,   1]], dtype=float)
    ky = kx.copy()

    kx_c, ky_c = _coarsen_2d_permeability_grids(kx, ky, batch_size=(2, 2))
    # kx_c uses harmonic mean in x (across layers), arithmetic in y
    # ky_c uses harmonic mean in y (across layers), arithmetic in x
    ```

    References:
        Cardwell, W. T., & Parsons, R. L. (1945). "Average Permeabilities of
        Heterogeneous Oil Sands." Transactions of the AIME, 160(01), 34-42.
    """
    if kx.shape != ky.shape:
        raise ValidationError(
            f"Permeability grids must have same shape. Got kx: {kx.shape}, ky: {ky.shape}"
        )

    if kx.ndim != 2:
        raise ValidationError(
            f"Expected 2D grids, got {kx.ndim}D. Use _coarsen_3d_permeability_grids instead."
        )

    if len(batch_size) != 2:
        raise ValidationError(
            f"batch_size must have 2 elements for 2D grids, got {len(batch_size)}"
        )

    bx, by = batch_size
    if bx < 1 or by < 1:
        raise ValidationError(f"batch_size elements must be >= 1, got ({bx}, {by})")

    nx, ny = kx.shape

    # Compute padding
    pad_x = (bx - nx % bx) % bx
    pad_y = (by - ny % by) % by

    # Pad grids with NaN
    if pad_x > 0 or pad_y > 0:
        kx_padded = np.pad(
            kx, ((0, pad_x), (0, pad_y)), mode="constant", constant_values=np.nan
        )
        ky_padded = np.pad(
            ky, ((0, pad_x), (0, pad_y)), mode="constant", constant_values=np.nan
        )
    else:
        kx_padded = kx
        ky_padded = ky

    nx_new, ny_new = kx_padded.shape
    nx_coarse = nx_new // bx
    ny_coarse = ny_new // by

    dtype = get_dtype()

    # Initialize output arrays
    kx_coarse = np.zeros((nx_coarse, ny_coarse), dtype=dtype)
    ky_coarse = np.zeros((nx_coarse, ny_coarse), dtype=dtype)

    # Coarsen k_x: harmonic in x, arithmetic in y
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            # Extract block
            block_kx = kx_padded[i * bx : (i + 1) * bx, j * by : (j + 1) * by]

            # First, take harmonic mean along x-direction (axis=0)
            kx_harmonic_x = _axis_harmonic_mean(block_kx, axis=0, epsilon=epsilon)

            # Then, take arithmetic mean along y-direction
            kx_coarse[i, j] = np.nanmean(kx_harmonic_x)

    # Coarsen k_y: harmonic in y, arithmetic in x
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            # Extract block
            block_ky = ky_padded[i * bx : (i + 1) * bx, j * by : (j + 1) * by]

            # First, take harmonic mean along y-direction (axis=1)
            ky_harmonic_y = _axis_harmonic_mean(block_ky, axis=1, epsilon=epsilon)

            # Then, take arithmetic mean along x-direction
            ky_coarse[i, j] = np.nanmean(ky_harmonic_y)

    return kx_coarse, ky_coarse


def _coarsen_3d_permeability_grids(
    kx: ThreeDimensionalGrid,
    ky: ThreeDimensionalGrid,
    kz: ThreeDimensionalGrid,
    batch_size: ThreeDimensions,
    epsilon: float = 1e-10,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Coarsen 3D permeability grids using direction-appropriate averaging.

    Uses Cardwell-Parsons averaging extended to 3D:
    - k_x: harmonic mean in x, arithmetic mean in y and z
    - k_y: harmonic mean in y, arithmetic mean in x and z
    - k_z: harmonic mean in z, arithmetic mean in x and y

    This preserves the effective permeability for flow in each direction.

    :param kx: X-direction permeability grid (mD), shape (nx, ny, nz)
    :param ky: Y-direction permeability grid (mD), shape (nx, ny, nz)
    :param kz: Z-direction permeability grid (mD), shape (nx, ny, nz)
    :param batch_size: Coarsening factors (bx, by, bz) for each direction
    :param epsilon: Small value to avoid division by zero (default: 1e-10)
    :return: Tuple of (coarsened_kx, coarsened_ky, coarsened_kz)
    :raises ValidationError: If grids have different shapes or batch_size is invalid

    Example:
    ```python
    # 4x4x4 grid
    kx = np.random.uniform(10, 100, (4, 4, 4))
    ky = np.random.uniform(10, 100, (4, 4, 4))
    kz = np.random.uniform(1, 10, (4, 4, 4))  # Lower vertical perm

    kx_c, ky_c, kz_c = _coarsen_3d_permeability_grids(
        kx, ky, kz,
        batch_size=(2, 2, 2)
    )
    # Result: 2x2x2 coarsened grids
    ```

    References:
        Cardwell, W. T., & Parsons, R. L. (1945). "Average Permeabilities of
        Heterogeneous Oil Sands." Transactions of the AIME, 160(01), 34-42.

        Deutsch, C. V. (1989). "Calculating Effective Absolute Permeability in
        Sandstone/Shale Sequences." SPE Formation Evaluation, 4(03), 343-348.
    """
    if not (kx.shape == ky.shape == kz.shape):
        raise ValidationError(
            f"All permeability grids must have same shape. "
            f"Got kx: {kx.shape}, ky: {ky.shape}, kz: {kz.shape}"
        )

    if kx.ndim != 3:
        raise ValidationError(
            f"Expected 3D grids, got {kx.ndim}D. Use _coarsen_2d_permeability_grids instead."
        )

    if len(batch_size) != 3:
        raise ValidationError(
            f"batch_size must have 3 elements for 3D grids, got {len(batch_size)}"
        )

    bx, by, bz = batch_size
    if bx < 1 or by < 1 or bz < 1:
        raise ValidationError(
            f"All batch_size elements must be >= 1, got ({bx}, {by}, {bz})"
        )

    nx, ny, nz = kx.shape

    # Compute padding
    pad_x = (bx - nx % bx) % bx
    pad_y = (by - ny % by) % by
    pad_z = (bz - nz % bz) % bz

    # Pad grids with NaN
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        pad_width = ((0, pad_x), (0, pad_y), (0, pad_z))
        kx_padded = np.pad(kx, pad_width, mode="constant", constant_values=np.nan)
        ky_padded = np.pad(ky, pad_width, mode="constant", constant_values=np.nan)
        kz_padded = np.pad(kz, pad_width, mode="constant", constant_values=np.nan)
    else:
        kx_padded = kx
        ky_padded = ky
        kz_padded = kz

    nx_new, ny_new, nz_new = kx_padded.shape
    nx_coarse = nx_new // bx
    ny_coarse = ny_new // by
    nz_coarse = nz_new // bz

    dtype = get_dtype()

    # Initialize output arrays
    kx_coarse = np.zeros((nx_coarse, ny_coarse, nz_coarse), dtype=dtype)
    ky_coarse = np.zeros((nx_coarse, ny_coarse, nz_coarse), dtype=dtype)
    kz_coarse = np.zeros((nx_coarse, ny_coarse, nz_coarse), dtype=dtype)

    # Coarsen k_x: harmonic in x, arithmetic in y and z
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            for k in range(nz_coarse):
                # Extract block
                block_kx = kx_padded[
                    i * bx : (i + 1) * bx, j * by : (j + 1) * by, k * bz : (k + 1) * bz
                ]

                # Harmonic mean in x (axis=0), then arithmetic in y and z
                kx_harmonic_x = _axis_harmonic_mean(block_kx, axis=0, epsilon=epsilon)
                kx_coarse[i, j, k] = np.nanmean(kx_harmonic_x)

    # Coarsen k_y: harmonic in y, arithmetic in x and z
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            for k in range(nz_coarse):
                # Extract block
                block_ky = ky_padded[
                    i * bx : (i + 1) * bx, j * by : (j + 1) * by, k * bz : (k + 1) * bz
                ]

                # Harmonic mean in y (axis=1), then arithmetic in x and z
                ky_harmonic_y = _axis_harmonic_mean(block_ky, axis=1, epsilon=epsilon)
                ky_coarse[i, j, k] = np.nanmean(ky_harmonic_y)

    # Coarsen k_z: harmonic in z, arithmetic in x and y
    for i in range(nx_coarse):
        for j in range(ny_coarse):
            for k in range(nz_coarse):
                # Extract block
                block_kz = kz_padded[
                    i * bx : (i + 1) * bx, j * by : (j + 1) * by, k * bz : (k + 1) * bz
                ]

                # Harmonic mean in z (axis=2), then arithmetic in x and y
                kz_harmonic_z = _axis_harmonic_mean(block_kz, axis=2, epsilon=epsilon)
                kz_coarse[i, j, k] = np.nanmean(kz_harmonic_z)

    return kx_coarse, ky_coarse, kz_coarse


def _axis_harmonic_mean(
    data: np.typing.NDArray, axis: int, epsilon: float = 1e-10
) -> np.typing.NDArray:
    """
    Compute harmonic mean along a specific axis, handling NaN values.

    Harmonic mean: H = n / sum(1/x_i) where n = count of non-NaN values

    :param data: Input array
    :param axis: Axis along which to compute harmonic mean
    :param epsilon: Small value added to denominator to avoid division by zero
    :return: Array with harmonic mean computed along specified axis
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # Compute reciprocals, preserving NaN
        reciprocals = np.where(np.isnan(data), np.nan, 1.0 / (data + epsilon))

        # Count non-NaN values along axis
        counts = np.sum(~np.isnan(data), axis=axis)

        # Sum reciprocals, ignoring NaN
        sum_reciprocals = np.nansum(reciprocals, axis=axis)

        # Harmonic mean = n / sum(1/x)
        result = counts / (sum_reciprocals + epsilon)

        # Handle edge cases
        result = np.where(counts == 0, np.nan, result)
        result = np.where(np.isinf(result), 0.0, result)

    return result


def coarsen_permeability_grids(
    kx: typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid],
    ky: typing.Union[TwoDimensionalGrid, ThreeDimensionalGrid],
    kz: typing.Optional[ThreeDimensionalGrid] = None,
    batch_size: typing.Union[TwoDimensions, ThreeDimensions, None] = None,
    epsilon: float = 1e-10,
) -> typing.Union[
    typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid],
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
]:
    """
    Coarsen permeability grids using direction-appropriate averaging.

    Automatically dispatches to 2D or 3D version based on input dimensions.

    :param kx: X-direction permeability grid (mD)
    :param ky: Y-direction permeability grid (mD)
    :param kz: Z-direction permeability grid (mD), required for 3D
    :param batch_size: Coarsening factors for each direction
    :param epsilon: Small value to avoid division by zero (default: 1e-10)
    :return: Coarsened permeability grids (kx, ky) for 2D or (kx, ky, kz) for 3D

    Example:
    ```python
    # 2D case
    kx, ky = coarsen_permeability_grids(kx_2d, ky_2d, batch_size=(2, 2))

    # 3D case
    kx, ky, kz = coarsen_permeability_grids(kx_3d, ky_3d, kz_3d, batch_size=(2, 2, 2))
    ```
    """
    if batch_size is None:
        raise ValidationError("batch_size must be provided")

    if kx.ndim == 2:
        if kz is not None:
            raise ValidationError("kz should not be provided for 2D grids")
        if len(batch_size) != 2:
            raise ValidationError(
                f"`batch_size` must have 2 elements for 2D grids, got {len(batch_size)}"
            )
        return _coarsen_2d_permeability_grids(
            kx=kx,  # type: ignore[arg-type]
            ky=ky,  # type: ignore[arg-type]
            batch_size=batch_size,  # type: ignore[arg-type]
            epsilon=epsilon,
        )

    elif kx.ndim == 3:
        if kz is None:
            raise ValidationError("kz must be provided for 3D grids")
        if len(batch_size) != 3:
            raise ValidationError(
                f"`batch_size` must have 3 elements for 3D grids, got {len(batch_size)}"
            )
        return _coarsen_3d_permeability_grids(
            kx=kx,  # type: ignore[arg-type]
            ky=ky,  # type: ignore[arg-type]
            kz=kz,
            batch_size=batch_size,  # type: ignore[arg-type]
            epsilon=epsilon,
        )

    raise ValidationError(f"Permeability grids must be 2D or 3D, got {kx.ndim}D")


FlattenStrategy = typing.Union[
    typing.Callable[
        [np.typing.NDArray], typing.Union[float, np.floating, np.typing.NDArray]
    ],
    typing.Literal["max", "min", "mean", "sum", "top", "bottom", "weighted_mean"],
]


def flatten_multilayer_grid_to_surface(
    multilayer_grid: ThreeDimensionalGrid,
    strategy: FlattenStrategy = "max",
    weights: typing.Optional[ThreeDimensionalGrid] = None,
    ignore_nan: bool = True,
) -> TwoDimensionalGrid:
    """
    Flatten a 3D multilayer grid to a 2D surface by collapsing the z-axis (depth).

    This is useful for creating 2D property maps from 3D reservoir grids, such as:
    - Maximum saturation across all layers
    - Average pressure over reservoir thickness
    - Top-of-reservoir property maps

    The z-axis (axis=2) corresponds to depth, where k=0 is the top layer and
    k=nz-1 is the bottom layer.

    :param multilayer_grid: 3D grid with shape (nx, ny, nz) to flatten
    :param strategy: Flattening method to use:
        - "max": Maximum value across layers (useful for saturation, pressure)
        - "min": Minimum value across layers
        - "mean": Arithmetic mean across layers (useful for average properties)
        - "sum": Sum across layers (useful for volumes, totals)
        - "top": Value at top layer (k=0)
        - "bottom": Value at bottom layer (k=nz-1)
        - "weighted_mean": Weighted average (requires `weights` parameter)
        - callable: Custom function that takes 1D array and returns scalar
    :param weights: Optional 3D weight grid (same shape as multilayer_grid).
        Only used when strategy="weighted_mean". Typically layer thickness.
    :param ignore_nan: If True, use NaN-aware operations (nanmax, nanmean, etc.).
        If False, NaN values will propagate to the output. Default: True.
    :return: 2D grid with shape (nx, ny) after flattening
    :raises ValidationError: If input is not 3D, weights shape mismatch, or invalid strategy

    Examples:
    ```python
    # Maximum oil saturation across all layers
    so_max = flatten_multilayer_grid_to_surface(so_grid, strategy="max")

    # Average pressure (thickness-weighted)
    p_avg = flatten_multilayer_grid_to_surface(
        pressure_grid,
        strategy="weighted_mean",
        weights=thickness_grid
    )

    # Top-of-reservoir porosity
    phi_top = flatten_multilayer_grid_to_surface(porosity_grid, strategy="top")

    # Custom strategy: 90th percentile
    p90 = flatten_multilayer_grid_to_surface(
        perm_grid,
        strategy=lambda z: np.nanpercentile(z, 90)
    )
    ```

    Notes:
    - For permeability, consider using directional averaging instead of simple flattening
    - NaN values are handled appropriately if `ignore_nan=True`
    - Weighted mean normalizes weights automatically (no need to pre-normalize)
    """
    if multilayer_grid.ndim != 3:
        raise ValidationError(
            f"`multilayer_grid` must be 3D with shape (nx, ny, nz), got {multilayer_grid.ndim}D"
        )

    nx, ny, nz = multilayer_grid.shape
    dtype = get_dtype()

    # Handle weighted mean separately
    if strategy == "weighted_mean":
        if weights is None:
            raise ValidationError(
                "`weights` parameter is required when `strategy='weighted_mean'`"
            )
        if weights.shape != multilayer_grid.shape:
            raise ValidationError(
                f"`weights` shape {weights.shape} must match `multilayer_grid` shape {multilayer_grid.shape}"
            )

        # Weighted average: sum(w * x) / sum(w)
        if ignore_nan:
            # Handle NaN in both data and weights
            weighted_sum = np.nansum(weights * multilayer_grid, axis=2)
            weight_sum = np.nansum(weights, axis=2)  # type: ignore[arg-type]
        else:
            weighted_sum = np.sum(weights * multilayer_grid, axis=2)
            weight_sum = np.sum(weights, axis=2)  # type: ignore[arg-type]

        # Avoid division by zero
        result = np.divide(
            weighted_sum,
            weight_sum,
            out=np.full((nx, ny), np.nan, dtype=dtype),
            where=(weight_sum != 0),
        )
        return result.astype(dtype)  # type: ignore[return-value]

    if isinstance(strategy, str):
        if strategy == "max":
            func = np.nanmax if ignore_nan else np.max
            return func(multilayer_grid, axis=2).astype(dtype)

        elif strategy == "min":
            func = np.nanmin if ignore_nan else np.min
            return func(multilayer_grid, axis=2).astype(dtype)

        elif strategy == "mean":
            func = np.nanmean if ignore_nan else np.mean
            return func(multilayer_grid, axis=2, dtype=dtype)

        elif strategy == "sum":
            func = np.nansum if ignore_nan else np.sum
            return func(multilayer_grid, axis=2, dtype=dtype)

        elif strategy == "top":
            # k=0 is top layer
            return multilayer_grid[:, :, 0].astype(dtype)  # type: ignore[return-value]

        elif strategy == "bottom":
            # k=nz-1 is bottom layer
            return multilayer_grid[:, :, -1].astype(dtype)  # type: ignore[return-value]

        else:
            raise ValidationError(
                f"Unknown strategy '{strategy}'. Valid options: "
                "'max', 'min', 'mean', 'sum', 'top', 'bottom', 'weighted_mean'"
            )

    elif callable(strategy):
        # Check if function is vectorized (much faster)
        try:
            # Test with a small slice
            test_slice = multilayer_grid[0, 0, :]
            result_scalar = strategy(test_slice)

            # Check if result is scalar
            if not np.isscalar(result_scalar):
                raise ValidationError(
                    f"Custom strategy function must return a scalar, got {type(result_scalar)}"
                )

            # Try vectorized approach first
            # Reshape to (nx*ny, nz) for efficient processing
            reshaped = multilayer_grid.reshape(-1, nz)

            # Check if function works on 2D array (vectorized)
            try:
                # Some numpy functions can handle 2D input
                result_flat = strategy(reshaped)
                if result_flat.shape == (nx * ny,):  # type: ignore[union-attr]
                    return result_flat.reshape(nx, ny).astype(dtype)  # type: ignore[union-attr]
            except (ValueError, TypeError):
                pass  # Fall back to `apply_along_axis`

            # Fall back to slower `apply_along_axis`
            warnings.warn(
                "Using `apply_along_axis` for custom strategy. "
                "For better performance, use vectorized numpy functions or built-in strategies.",
                UserWarning,
                stacklevel=2,
            )
            result = np.apply_along_axis(strategy, axis=2, arr=multilayer_grid)
            return result.astype(dtype)  # type: ignore[return-value]

        except Exception as exc:
            raise ValidationError(f"Custom strategy function failed: {exc}") from exc

    raise ValidationError(
        f"`strategy` must be a string or callable, got {type(strategy)}"
    )


def flatten_multilayer_grids(
    grids: typing.Dict[str, ThreeDimensionalGrid],
    strategy: typing.Union[FlattenStrategy, typing.Dict[str, FlattenStrategy]] = "max",
    weights: typing.Optional[typing.Dict[str, ThreeDimensionalGrid]] = None,
    ignore_nan: bool = True,
) -> typing.Dict[str, TwoDimensionalGrid]:
    """
    Flatten multiple 3D grids to 2D surfaces using specified strategies.

    Convenient wrapper for flattening multiple related grids (e.g., all saturations,
    all pressures) with a single call.

    :param grids: Dictionary of {name: 3D_grid} to flatten
    :param strategy: Single strategy for all grids, or dict of {name: strategy}
    :param weights: Optional dict of {name: weight_grid} for weighted averaging
    :param ignore_nan: If True, use NaN-aware operations
    :return: Dictionary of {name: 2D_grid} after flattening

    Example:
    ```python
    grids_3d = {
        'oil_saturation': so_grid,
        'water_saturation': sw_grid,
        'pressure': p_grid,
    }

    # Use same strategy for all
    grids_2d = flatten_multilayer_grids(grids_3d, strategy="max")

    # Use different strategies per grid
    strategies = {
        'oil_saturation': 'max',
        'water_saturation': 'mean',
        'pressure': 'weighted_mean',
    }
    weights_dict = {
        'pressure': thickness_grid,
    }
    grids_2d = flatten_multilayer_grids(
        grids_3d,
        strategy=strategies,
        weights=weights_dict
    )
    ```
    """
    result = {}
    for name, grid in grids.items():
        if isinstance(strategy, dict):
            grid_strategy = strategy.get(name, "max")  # type: ignore
        else:
            grid_strategy = strategy

        grid_weights = None
        if weights is not None and name in weights:
            grid_weights = weights[name]

        result[name] = flatten_multilayer_grid_to_surface(
            grid,
            strategy=grid_strategy,
            weights=grid_weights,
            ignore_nan=ignore_nan,
        )
    return result


class PadMixin(typing.Generic[NDimension]):
    """Mixin class to add padding functionality to attrs classes with numpy array fields."""

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        """Return iterable of attrs fields that can be padded."""
        raise NotImplementedError

    def pad(
        self,
        pad_width: int = 1,
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
        padded_fields_values = {}
        non_init_fields_values = {}
        for field in target_fields:
            value = getattr(self, field.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Field '{field.name}' is not a numpy array and cannot be padded"
                )

            padded_value = pad_grid(grid=value, pad_width=pad_width)
            if hook and (not exclude or field.name not in exclude):
                padded_value = hook(padded_value)

            if not field.init:
                non_init_fields_values[field.name] = padded_value
            else:
                padded_fields_values[field.name] = padded_value

        instance = attrs.evolve(self, **padded_fields_values)  # type: ignore[misc]
        for name, value in non_init_fields_values:
            object.__setattr__(instance, name, value)
        return instance

    def unpad(self, pad_width: int = 1) -> Self:
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
        unpadded_fields_values = {}
        non_init_fields_values = {}
        for field in target_fields:
            value = getattr(self, field.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Field '{field.name}' is not a numpy array and cannot be padded"
                )

            padded_value = unpad_grid(grid=value, pad_width=pad_width)
            if not field.init:
                non_init_fields_values[field.name] = padded_value
            else:
                unpadded_fields_values[field.name] = padded_value

        instance = attrs.evolve(self, **unpadded_fields_values)  # type: ignore[misc]
        for name, value in non_init_fields_values:
            object.__setattr__(instance, name, value)
        return instance

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
        return attrs.evolve(self, **hooked_fields)  # type: ignore[misc]


@attrs.frozen(slots=True)
class RelPermGrids(PadMixin[NDimension], Serializable):  # type: ignore[override]
    """
    Wrapper for n-dimensional grids representing relative permeabilities
    for different fluid phases (oil, water, gas).
    """

    oil_relative_permeability: NDimensionalGrid[NDimension]
    """Grid representing oil relative permeability."""
    water_relative_permeability: NDimensionalGrid[NDimension]
    """Grid representing water relative permeability."""
    gas_relative_permeability: NDimensionalGrid[NDimension]
    """Grid representing gas relative permeability."""

    @property
    def kro(self) -> NDimensionalGrid[NDimension]:
        return self.oil_relative_permeability

    @property
    def krw(self) -> NDimensionalGrid[NDimension]:
        return self.water_relative_permeability

    @property
    def krg(self) -> NDimensionalGrid[NDimension]:
        return self.gas_relative_permeability

    Kro = kro
    Krw = krw
    Krg = krg

    def __iter__(self) -> typing.Iterator[NDimensionalGrid[NDimension]]:
        yield self.water_relative_permeability
        yield self.oil_relative_permeability
        yield self.gas_relative_permeability

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        return attrs.fields(self.__class__)


@attrs.frozen(slots=True)
class RelativeMobilityGrids(PadMixin[NDimension], Serializable):
    """
    Wrapper for n-dimensional grids representing relative mobilities
    for different fluid phases (oil, water, gas).
    """

    oil_relative_mobility: NDimensionalGrid[NDimension]
    """Grid representing oil relative mobility."""
    water_relative_mobility: NDimensionalGrid[NDimension]
    """Grid representing water relative mobility."""
    gas_relative_mobility: NDimensionalGrid[NDimension]
    """Grid representing gas relative mobility."""

    @property
    def λo(self) -> NDimensionalGrid[NDimension]:
        return self.oil_relative_mobility

    @property
    def λw(self) -> NDimensionalGrid[NDimension]:
        return self.water_relative_mobility

    @property
    def λg(self) -> NDimensionalGrid[NDimension]:
        return self.gas_relative_mobility

    def __iter__(self) -> typing.Iterator[NDimensionalGrid[NDimension]]:
        yield self.water_relative_mobility
        yield self.oil_relative_mobility
        yield self.gas_relative_mobility

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        return attrs.fields(self.__class__)


@attrs.frozen(slots=True)
class CapillaryPressureGrids(PadMixin[NDimension], Serializable):
    """
    Wrapper for n-dimensional grids representing capillary pressures
    for different fluid phases (oil-water, oil-gas).
    """

    oil_water_capillary_pressure: NDimensionalGrid[NDimension]
    """Grid representing oil-water capillary pressure."""
    gas_oil_capillary_pressure: NDimensionalGrid[NDimension]
    """Grid representing gas-oil capillary pressure."""

    @property
    def pcow(self) -> NDimensionalGrid[NDimension]:
        return self.oil_water_capillary_pressure

    @property
    def pcgo(self) -> NDimensionalGrid[NDimension]:
        return self.gas_oil_capillary_pressure

    Pcow = pcow
    Pcgo = pcgo

    def __iter__(self) -> typing.Iterator[NDimensionalGrid[NDimension]]:
        yield self.oil_water_capillary_pressure
        yield self.gas_oil_capillary_pressure

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        return attrs.fields(self.__class__)


@attrs.frozen(slots=True)
class RateGrids(PadMixin[NDimension], Serializable):
    """
    Wrapper for n-dimensional grids representing fluid flow rates (oil, water, gas).
    """

    oil: typing.Optional[NDimensionalGrid[NDimension]] = None
    """Grid representing oil flow rates."""
    water: typing.Optional[NDimensionalGrid[NDimension]] = None
    """Grid representing water flow rates."""
    gas: typing.Optional[NDimensionalGrid[NDimension]] = None
    """Grid representing gas flow rates."""

    @property
    def total(self) -> typing.Optional[NDimensionalGrid[NDimension]]:
        """
        Returns the total fluid flow rate (oil + water + gas) at each grid cell.

        Ensure that at least one of the phase grids is defined before accessing this property.
        Also, all defined phase grids should have the same shape and unit.

        If none of the individual phase grids are defined, returns None.
        """
        total_grid = None
        if self.oil is not None:
            total_grid = self.oil.copy()
        if self.water is not None:
            if total_grid is None:
                total_grid = self.water.copy()
            else:
                total_grid += self.water
        if self.gas is not None:
            if total_grid is None:
                total_grid = self.gas.copy()
            else:
                total_grid += self.gas
        return total_grid

    def __getitem__(self, key: NDimension) -> typing.Tuple[float, float, float]:
        """
        Returns the oil, water, and gas flow rates at the specified grid cell.

        If a phase grid is not defined, its flow rate is returned as 0.0.

        :param key: The grid cell index (tuple of integers).
        :return: A tuple containing the oil, water, and gas flow rates.
        """
        oil = self.oil[key] if self.oil is not None else 0.0
        water = self.water[key] if self.water is not None else 0.0
        gas = self.gas[key] if self.gas is not None else 0.0
        return oil, water, gas

    def get_paddable_fields(self) -> typing.Iterable[typing.Any]:
        return attrs.fields(self.__class__)


@attrs.frozen(slots=True)
class _RateGridsProxy(typing.Generic[NDimension]):
    """
    Proxy to allow (controlled) item assignment on an n-dimensional rate grids.
    without exposing the grid itself
    """

    oil: NDimensionalGrid[NDimension]
    water: NDimensionalGrid[NDimension]
    gas: NDimensionalGrid[NDimension]

    def __setitem__(
        self, key: NDimension, value: typing.Tuple[float, float, float]
    ) -> None:
        """
        Sets the oil, water, and gas production rates at the specified grid cell.

        :param key: The grid cell index (tuple of integers).
        :param value: A tuple containing the oil, water, and gas production rates.
        """
        oil, water, gas = value
        self.oil[key] = oil
        self.water[key] = water
        self.gas[key] = gas
