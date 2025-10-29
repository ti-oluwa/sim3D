import typing

import numpy as np
from numpy.typing import DTypeLike

from sim3D.types import ArrayLike, NDimension, NDimensionalGrid, Orientation

__all__ = [
    "uniform_grid",
    "layered_grid",
    "build_uniform_grid",
    "build_layered_grid",
    "elevation_grid",
    "depth_grid",
    "build_elevation_grid",
    "build_depth_grid",
    "apply_structural_dip",
    "edge_pad_grid",
    "coarsen_grid",
]


def build_uniform_grid(
    grid_shape: NDimension,
    value: float = 0.0,
    dtype: DTypeLike = np.float64,
) -> NDimensionalGrid[NDimension]:
    """
    Constructs a N-Dimensional uniform grid with the specified initial value.

    :param grid_shape: Tuple of number of cells in all directions (x, y, z).
    :param value: Initial value to fill the grid with
    :param dtype: Data type of the grid elements (default: np.float64)
    :return: Numpy array representing the grid
    """
    return np.full(grid_shape, fill_value=value, dtype=dtype)


uniform_grid = build_uniform_grid  # Alias for convenience


def build_layered_grid(
    grid_shape: NDimension,
    layer_values: ArrayLike[float],
    orientation: Orientation,
    dtype: DTypeLike = np.float64,
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
        raise ValueError("At least one layer value must be provided.")

    layered_grid = build_uniform_grid(grid_shape=grid_shape, value=0.0, dtype=dtype)
    if orientation == Orientation.X:  # Layering along x-axis
        if len(layer_values) != grid_shape[0]:
            raise ValueError(
                "Number of layer values must match number of cells in x direction."
            )

        for i, layer_value in enumerate(layer_values):
            layered_grid[i, :, :] = layer_value
        return layered_grid

    elif orientation == Orientation.Y:  # Layering along y-axis
        if len(layer_values) != grid_shape[1]:
            raise ValueError(
                "Number of layer values must match number of cells in y direction."
            )

        for j, layer_value in enumerate(layer_values):
            layered_grid[:, j, :] = layer_value
        return layered_grid

    elif orientation == Orientation.Z:  # Layering along z-axis
        if len(grid_shape) != 3:
            raise ValueError(
                "Grid dimension must be N-Dimensional for z-direction layering."
            )

        if len(layer_values) != grid_shape[2]:
            raise ValueError(
                "Number of layer values must match number of cells in z direction."
            )

        for k, layer_value in enumerate(layer_values):
            layered_grid[:, :, k] = layer_value
        return layered_grid

    raise ValueError("Invalid layering direction. Must be one of 'x', 'y', or 'z'.")


layered_grid = build_layered_grid  # Alias for convenience


def _build_elevation_grid(
    thickness_grid: NDimensionalGrid[NDimension],
    direction: typing.Literal["downward", "upward"] = "downward",
) -> NDimensionalGrid[NDimension]:
    """
    Convert a cell thickness (height) grid into an absolute elevation grid (cell center z-coordinates).

    The elevation grid is generated based on the thickness of each cell, starting from the top or bottom
    of the reservoir, depending on the specified direction.

    :param thickness_grid: N-dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :param direction: Direction to generate the elevation grid.
        Can be "downward" (from top to bottom) or "upward" (from bottom to top).
        "downward" basically gives a depth grid from the top of the reservoir.
    :return: N-dimensional numpy array representing the elevation of each cell in the reservoir (ft).
    """
    if direction not in {"downward", "upward"}:
        raise ValueError("direction must be 'downward' or 'upward'")

    nx, ny, nz = thickness_grid.shape  # Now Z is LAST
    elevation_grid = np.zeros_like(thickness_grid, dtype=float)

    if direction == "downward":
        # Iterate over Z in the LAST dimension
        elevation_grid[:, :, 0] = thickness_grid[:, :, 0] / 2
        for k in range(1, nz):
            elevation_grid[:, :, k] = (
                elevation_grid[:, :, k - 1]
                + thickness_grid[:, :, k - 1] / 2
                + thickness_grid[:, :, k] / 2
            )
    else:
        # Start from bottom layer
        elevation_grid[:, :, -1] = thickness_grid[:, :, -1] / 2
        for k in range(nz - 2, -1, -1):
            elevation_grid[:, :, k] = (
                elevation_grid[:, :, k + 1]
                + thickness_grid[:, :, k + 1] / 2
                + thickness_grid[:, :, k] / 2
            )
    return typing.cast(NDimensionalGrid[NDimension], elevation_grid)


def build_elevation_grid(
    thickness_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Public wrapper for building an elevation grid from a thickness grid.

    Elevation is measured from base level upward.

    :param thickness_grid: N-dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :return: N-dimensional numpy array representing the elevation of each cell in the reservoir (ft).
    """
    return _build_elevation_grid(thickness_grid, direction="upward")


elevation_grid = build_elevation_grid  # Alias for convenience


def build_depth_grid(
    thickness_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Public wrapper for building a downward depth grid from a thickness grid.

    Depth is measured from top level downward.

    :param thickness_grid: N-dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :return: N-dimensional numpy array representing the depth of each cell in the reservoir (ft).
    """
    return _build_elevation_grid(thickness_grid, direction="downward")


depth_grid = build_depth_grid  # Alias for convenience


def apply_structural_dip(
    elevation_grid: NDimensionalGrid[NDimension],
    cell_dimension: typing.Tuple[float, float],
    elevation_direction: typing.Literal["downward", "upward"],
    dip_angle: float,
    dip_direction: typing.Literal["N", "S", "E", "W"],
) -> NDimensionalGrid[NDimension]:
    """
        Apply structural dip to a base elevation grid.

        The dip is applied by adding a linear gradient in the dip direction.
        The dip angle represents the angle of the reservoir surface from horizontal.

        ---

        ## 🧭 **Dip Direction Convention:**
        ```
        Grid Coordinate System:
        North (↑ +y)
            |
            |
    West ←─────┼─────→ East
    (-x)     |    (+x)
            |
        South (↓ -y)

        ```

        Dip Direction Examples:
        - "N": Reservoir dips toward North → Higher elevation at South
        - "S": Reservoir dips toward South → Higher elevation at North
        - "E": Reservoir dips toward East → Higher elevation at West
        - "W": Reservoir dips toward West → Higher elevation at East

        :param elevation_grid: Base flat elevation grid
        :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet
        :param elevation_direction: Whether elevation of the base grid is increasing "downward" (depth) or "upward" (elevation)
        :param dip_angle: Dip angle in degrees
        :param dip_direction: Direction of dip ("N", "S", "E", "W")
        :return: Elevation grid with structural dip applied
    """
    dipped_elevation_grid = elevation_grid.copy()
    dip_angle_radians = np.radians(dip_angle)
    cell_size_x, cell_size_y = cell_dimension
    grid_shape = elevation_grid.shape

    # Determine dip direction in grid coordinates
    # Grid convention: x = East-West, y = North-South
    if dip_direction == "E":
        # Dips toward East (positive x-direction)
        # Elevation increases toward West (negative x)
        for i in range(grid_shape[0]):
            dip_offset = i * cell_size_x * np.tan(dip_angle_radians)
            if elevation_direction == "upward":
                # Upward elevation: increase offset toward positive x
                dipped_elevation_grid[i, :, :] += dip_offset
            else:
                # Downward elevation (depth): increase depth toward positive x
                dipped_elevation_grid[i, :, :] += dip_offset

    elif dip_direction == "W":
        # Dips toward West (negative x-direction)
        # Elevation increases toward East (positive x)
        for i in range(grid_shape[0]):
            dip_offset = (
                (grid_shape[0] - 1 - i) * cell_size_x * np.tan(dip_angle_radians)
            )
            if elevation_direction == "upward":
                dipped_elevation_grid[i, :, :] += dip_offset
            else:
                dipped_elevation_grid[i, :, :] += dip_offset

    elif dip_direction == "N":
        # Dips toward North (positive y-direction)
        # Elevation increases toward South (negative y)
        for j in range(grid_shape[1]):
            dip_offset = j * cell_size_y * np.tan(dip_angle_radians)
            if elevation_direction == "upward":
                dipped_elevation_grid[:, j, :] += dip_offset
            else:
                dipped_elevation_grid[:, j, :] += dip_offset

    elif dip_direction == "S":
        # Dips toward South (negative y-direction)
        # Elevation increases toward North (positive y)
        for j in range(grid_shape[1]):
            dip_offset = (
                (grid_shape[1] - 1 - j) * cell_size_y * np.tan(dip_angle_radians)
            )
            if elevation_direction == "upward":
                dipped_elevation_grid[:, j, :] += dip_offset
            else:
                dipped_elevation_grid[:, j, :] += dip_offset

    return typing.cast(NDimensionalGrid[NDimension], dipped_elevation_grid)


def edge_pad_grid(
    grid: NDimensionalGrid[NDimension], pad_width: int = 1
) -> NDimensionalGrid[NDimension]:
    """
    Pads a N-Dimensional grid with the edge values to create a border around the grid.

    This is useful for finite difference methods where boundary conditions are applied.

    :param grid: N-Dimensional numpy array representing the grid to be padded
    :param pad_width: Width of the padding to be applied on all sides of the grid
    :return: Padded N-Dimensional numpy array
    """
    padded_grid = np.pad(grid, pad_width=pad_width, mode="edge")
    padded_grid = typing.cast(NDimensionalGrid[NDimension], padded_grid)
    return padded_grid


def coarsen_grid(
    data: NDimensionalGrid[NDimension],
    batch_size: NDimension,
    method: typing.Literal["mean", "sum", "max", "min"] = "mean",
) -> NDimensionalGrid[NDimension]:
    """
    Coarsen (downsample) a 2D or 3D grid by aggregating blocks of adjacent cells.

    Pads the grid if necessary to make dimensions divisible by batch_size, so no cells are lost.

    :param data: 2D or 3D numpy array to coarsen. Shape can be (nx, ny) or (nx, ny, nz).
    :param batch_size: Tuple of ints representing the coarsening factor along each dimension.
                       Length must match data.ndim.
                       Example: (2,2) for 2D, (2,2,2) for 3D.
    :param method: Aggregation method to use on each block. One of 'mean', 'sum', 'max', 'min'.
                   Default is 'mean'.
    :return: Coarsened numpy array.
    :raises ValueError: if batch_size length does not match data.ndim or if method is unsupported.

    :example:
    >>> data2d = np.arange(16).reshape(4,4)
    >>> coarsen_grid(data2d, batch_size=(2,2))
    array([[2.5, 4.5],
           [10.5, 12.5]])

    >>> data3d = np.arange(64).reshape(4,4,4)
    >>> coarsen_grid(data3d, batch_size=(2,2,2), method='max')
    array([[[ 5,  7],
            [13, 15]],
           [[21, 23],
            [29, 31]]])
    """
    if len(batch_size) != data.ndim:
        raise ValueError(
            f"batch_size length {len(batch_size)} must match data.ndim {data.ndim}"
        )

    pad_width = []
    for dim, b in zip(data.shape, batch_size):
        remainder = dim % b
        if remainder == 0:
            pad_width.append((0, 0))
        else:
            pad_width.append((0, b - remainder))

    # Pad with NaNs for mean, zeros for sum, -inf for max, +inf for min
    if method == "mean":
        pad_value = np.nan
    elif method == "sum":
        pad_value = 0
    elif method == "max":
        pad_value = -np.inf
    elif method == "min":
        pad_value = np.inf
    else:
        raise ValueError(f"Unsupported method '{method}'")

    data_padded = np.pad(
        data, pad_width=pad_width, mode="constant", constant_values=pad_value
    )

    # Reshape to group blocks along each dimension
    reshape_shape = []
    for dim, b in zip(data_padded.shape, batch_size):
        reshape_shape.extend([dim // b, b])
    data_reshaped = data_padded.reshape(reshape_shape)

    # Axes to aggregate over: every second axis (the 'b' axes)
    agg_axes = tuple(range(1, data_reshaped.ndim, 2))

    # Apply aggregation
    if method == "mean":
        coarsened = np.nanmean(data_reshaped, axis=agg_axes)
    elif method == "sum":
        coarsened = data_reshaped.sum(axis=agg_axes)
    elif method == "max":
        coarsened = data_reshaped.max(axis=agg_axes)
    elif method == "min":
        coarsened = data_reshaped.min(axis=agg_axes)

    return typing.cast(NDimensionalGrid[NDimension], coarsened)
