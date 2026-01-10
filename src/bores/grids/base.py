import typing

import attrs
import numba
import numpy as np
from typing_extensions import Self

from bores._precision import get_dtype
from bores.errors import ValidationError
from bores.types import (
    ArrayLike,
    NDimension,
    NDimensionalGrid,
    OneDimension,
    Orientation,
    ThreeDimensionalGrid,
    TwoDimensionalGrid,
)

__all__ = [
    "array",
    "uniform_grid",
    "layered_grid",
    "build_uniform_grid",
    "build_layered_grid",
    "elevation_grid",
    "depth_grid",
    "build_elevation_grid",
    "build_depth_grid",
    "apply_structural_dip",
    "pad_grid",
    "get_pad_mask",
    "unpad_grid",
    "coarsen_grid",
    "flatten_multilayer_grid_to_surface",
    "RateGrids",
    "RelativeMobilityGrids",
    "CapillaryPressureGrids",
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
    return np.full(grid_shape, fill_value=value, dtype=get_dtype(), order="C")  # type: ignore


uniform_grid = build_uniform_grid  # Alias for convenience


def build_layered_grid(
    grid_shape: NDimension,
    layer_values: ArrayLike[float],
    orientation: Orientation,
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
) -> NDimensionalGrid[NDimension]:
    """
    Compute elevation grid in downward direction (depth from top).

    :param thickness_grid: 3D array of cell thicknesses (ft)
    :param dtype: NumPy dtype for array allocation
    :return: 3D elevation grid (ft)
    """
    nx, ny, nz = thickness_grid.shape
    elevation_grid = np.zeros_like(thickness_grid, dtype=dtype)

    # Start from top layer
    elevation_grid[:, :, 0] = thickness_grid[:, :, 0] / 2
    for k in range(1, nz):
        elevation_grid[:, :, k] = (
            elevation_grid[:, :, k - 1]
            + thickness_grid[:, :, k - 1] / 2
            + thickness_grid[:, :, k] / 2
        )

    return elevation_grid  # type: ignore


@numba.njit(cache=True)
def _compute_elevation_upward(
    thickness_grid: NDimensionalGrid[NDimension],
    dtype: np.typing.DTypeLike,
) -> NDimensionalGrid[NDimension]:
    """
    Compute elevation grid in upward direction (elevation from bottom).

    :param thickness_grid: 3D array of cell thicknesses (ft)
    :param dtype: NumPy dtype for array allocation
    :return: 3D elevation grid (ft)
    """
    nx, ny, nz = thickness_grid.shape
    elevation_grid = np.zeros_like(thickness_grid, dtype=dtype)

    # Start from bottom layer
    elevation_grid[:, :, -1] = thickness_grid[:, :, -1] / 2
    for k in range(nz - 2, -1, -1):
        elevation_grid[:, :, k] = (
            elevation_grid[:, :, k + 1]
            + thickness_grid[:, :, k + 1] / 2
            + thickness_grid[:, :, k] / 2
        )

    return elevation_grid  # type: ignore


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
        raise ValidationError("direction must be 'downward' or 'upward'")

    dtype = get_dtype()

    if direction == "downward":
        return _compute_elevation_downward(thickness_grid, dtype=dtype)
    return _compute_elevation_upward(thickness_grid, dtype=dtype)


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

    # Group parameters for cleaner function calls
    grid_dimensions = (nx, ny)
    dip_components = (dx_component, dy_component, tan_dip_angle)

    # Apply dip using njitted helper functions
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
    padded_grid = np.pad(grid, pad_width=pad_width, mode="edge")
    return padded_grid  # type: ignore


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

    Example:
    ```python
    data2d = np.arange(16).reshape(4,4)
    coarsen_grid(data2d, batch_size=(2,2))
    # array([[2.5, 4.5],
    #         [10.5, 12.5]])

    data3d = np.arange(64).reshape(4,4,4)
    coarsen_grid(data3d, batch_size=(2,2,2), method='max')
    # array([[[ 5,  7],
    #          [13, 15]],
    #         [[21, 23],
    #          [29, 31]]])
    ```
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
        raise ValidationError(f"Unsupported method '{method}'")

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

    dtype = get_dtype()
    # Apply aggregation
    if method == "mean":
        coarsened = np.nanmean(data_reshaped, axis=agg_axes, dtype=dtype)
    elif method == "sum":
        coarsened = data_reshaped.sum(axis=agg_axes, dtype=dtype)
    elif method == "max":
        coarsened = data_reshaped.max(axis=agg_axes)
    elif method == "min":
        coarsened = data_reshaped.min(axis=agg_axes)

    return coarsened  # type: ignore


FlattenStrategy = typing.Union[
    typing.Callable[[NDimensionalGrid[OneDimension]], float],
    typing.Literal["max", "min", "mean"],
]


def flatten_multilayer_grid_to_surface(
    multilayer_grid: ThreeDimensionalGrid,
    strategy: FlattenStrategy = "max",
) -> TwoDimensionalGrid:
    """
    Vectorized flattening of a multilayer grid shaped (nx, ny, nz)
    into a 2D surface (nx, ny) by collapsing the z-axis using the
    provided strategy.

    :param multilayer_grid: 3D numpy array with shape (nx, ny, nz)
    :param strategy: Flattening strategy to apply along the z-axis.
        Can be one of the built-in strategies: "max", "min", "mean",
        or a custom callable that takes a 1D array and returns a float.
    :return: 2D numpy array with shape (nx, ny) after flattening
    """
    if multilayer_grid.ndim != 3:
        raise ValueError(
            "multilayer_grid must be a three-dimensional array with shape (nx, ny, nz)"
        )

    if strategy == "max":
        return np.max(multilayer_grid, axis=2)
    elif strategy == "min":
        return np.min(multilayer_grid, axis=2)
    elif strategy == "mean":
        return np.mean(multilayer_grid, axis=2)

    elif callable(strategy):
        # Apply function along z axis: shape → (nx, ny)
        return np.apply_along_axis(strategy, axis=2, arr=multilayer_grid)

    raise ValidationError(f"Unsupported flatten strategy: {strategy}")


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

        instance = attrs.evolve(self, **padded_fields_values)
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

        instance = attrs.evolve(self, **unpadded_fields_values)
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
        return attrs.evolve(self, **hooked_fields)


@attrs.frozen(slots=True)
class RelPermGrids(PadMixin[NDimension]):  # type: ignore[override]
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
class RelativeMobilityGrids(PadMixin[NDimension]):
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
class CapillaryPressureGrids(PadMixin[NDimension]):
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
class RateGrids(PadMixin[NDimension]):
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
