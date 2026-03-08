"""
Shared utilities for visualization modules.
"""

import typing

import numpy as np

from bores._precision import get_dtype
from bores.errors import ValidationError
from bores.models import ReservoirModel
from bores.states import ModelState
from bores.types import ThreeDimensionalGrid, ThreeDimensions


def slice_grid(
    data: ThreeDimensionalGrid,
    x_slice: typing.Optional[typing.Union[int, slice, typing.Tuple[int, int]]] = None,
    y_slice: typing.Optional[typing.Union[int, slice, typing.Tuple[int, int]]] = None,
    z_slice: typing.Optional[typing.Union[int, slice, typing.Tuple[int, int]]] = None,
) -> typing.Tuple[ThreeDimensionalGrid, typing.Tuple[slice, slice, slice]]:
    """
    Apply slicing operations to a 3D grid.

    :param data: 3D array to slice (nx, ny, nz)
    :param x_slice: X-axis slice specification (int, slice, or tuple of (start, end))
    :param y_slice: Y-axis slice specification (int, slice, or tuple of (start, end))
    :param z_slice: Z-axis slice specification (int, slice, or tuple of (start, end))
    :return: Tuple of (sliced_data, (x_slice_obj, y_slice_obj, z_slice_obj))
    :raises ValidationError: If slice specifications are invalid or result in empty grid

    Examples:
    ```python
    data = np.random.rand(10, 10, 10)
    # Extract single plane at x=5
    sliced, _ = slice_grid(data, x_slice=5)
    # Extract range from x=2 to x=8
    sliced, _ = slice_grid(data, x_slice=(2, 8))
    # Extract middle section in all dimensions
    sliced, _ = slice_grid(data, x_slice=(2, 8), y_slice=(3, 7), z_slice=(1, 9))
    ```
    """
    nx, ny, nz = data.shape

    def normalize_slice_spec(
        spec: typing.Optional[typing.Union[int, slice, typing.Tuple[int, int]]],
        dimension_size: int,
        axis_name: str,
    ) -> slice:
        """
        Convert various slice specifications to a standard slice object.

        :param spec: Slice specification (None, int, slice, or (start, end) tuple)
        :param dimension_size: Size of the dimension being sliced
        :param axis_name: Name of axis for error messages ('x', 'y', or 'z')
        :return: Normalized slice object
        :raises ValidationError: If specification is invalid
        """
        if spec is None:
            return slice(None)

        if isinstance(spec, int):
            # Convert negative indices
            if spec < 0:
                spec += dimension_size
            # Validate range
            if not (0 <= spec < dimension_size):
                raise ValidationError(
                    f"{axis_name}_slice {spec} out of range [0, {dimension_size - 1}]"
                )
            # Single index -> slice with width 1
            return slice(spec, spec + 1)

        if isinstance(spec, tuple) and len(spec) == 2:
            start, end = spec
            # Handle negative indices
            if start < 0:
                start += dimension_size
            if end < 0:
                end += dimension_size
            return slice(start, end)

        if isinstance(spec, slice):
            return spec

        raise ValidationError(
            f"Invalid '{axis_name}_slice': {spec}. "
            f"Expected int, slice, or (start, end) tuple"
        )

    # Normalize all slice specifications
    slice_x = normalize_slice_spec(x_slice, nx, "x")
    slice_y = normalize_slice_spec(y_slice, ny, "y")
    slice_z = normalize_slice_spec(z_slice, nz, "z")

    # Apply slicing
    sliced_data = data[slice_x, slice_y, slice_z]

    # Validate result
    if sliced_data.ndim != 3 or any(d < 1 for d in sliced_data.shape):
        raise ValidationError(
            f"Slice operation produced invalid shape {sliced_data.shape}. "
            f"Original shape: {data.shape}, "
            f"Slices: x={slice_x}, y={slice_y}, z={slice_z}"
        )

    return typing.cast(ThreeDimensionalGrid, sliced_data), (slice_x, slice_y, slice_z)


_missing = object()


def get_data(
    source: typing.Union[ModelState[ThreeDimensions], ReservoirModel],
    name: str,
) -> ThreeDimensionalGrid:
    """
    Extract property data from model state or reservoir model.

    Supports nested property access via dot notation (e.g., "permeability.x").

    :param source: The model or model state containing the property
    :param name: Property name as defined by `PropertyMeta.name`, supports dot notation
    :return: 3D numpy array containing the property data
    :raises ValidationError: If property name is invalid for the source type
    :raises AttributeError: If property is not found or has invalid value
    :raises TypeError: If property is not a 3-dimensional array

    Examples:
    ```python
    state = ModelState(...)
    pressure = get_data(state, "pressure")
    perm_x = get_data(state, "permeability.x")
    porosity = get_data(model, "model.porosity")
    ```
    """
    source_type = "model state"
    if isinstance(source, ReservoirModel):
        if not name.startswith("model."):
            raise ValidationError(
                f"Property {name.split('.')[-1]} not available on model. "
                f"Model properties must be prefixed with 'model.'"
            )
        name = name.removeprefix("model.")
        source_type = "reservoir model"

    # Navigate through nested attributes using dot notation
    obj = source
    for part in name.split("."):
        val = getattr(obj, part, _missing)
        if val is _missing:
            raise AttributeError(f"'{name}' not found in {source_type}")
        obj = val

    if obj is None or obj is _missing:
        raise AttributeError(f"Property '{name}' is invalid")

    if not isinstance(obj, np.ndarray):
        obj = np.array(obj, dtype=get_dtype())

    if obj.ndim != 3:
        raise TypeError(f"Property '{name}' is not a 3-D array")

    return typing.cast(ThreeDimensionalGrid, obj)
