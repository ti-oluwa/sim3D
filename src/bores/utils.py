import gzip
import lzma
from os import PathLike
from pathlib import Path
import pickle
import typing

import attrs
import numba
from numba.extending import overload
import numpy as np
from typing_extensions import Self

from bores.types import T


@numba.vectorize(cache=True)
def clip(val, min_, max_):
    return np.maximum(np.minimum(val, max_), min_)


@numba.njit(cache=True)
def clip_scalar(value: float, min_val: float, max_val: float) -> float:
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value


@typing.overload
def is_array(x: np.ndarray) -> typing.TypeGuard[np.typing.NDArray]: ...


@typing.overload
def is_array(x: typing.Any) -> typing.TypeGuard[np.typing.NDArray]: ...


@numba.njit(cache=True)
def is_array(x: typing.Any) -> bool:
    return hasattr(x, "shape") and isinstance(x.shape, tuple)


@numba.njit(cache=True)
def _apply_mask_2d(
    arr: np.typing.NDArray, mask: np.typing.NDArray, values: np.typing.NDArray
) -> None:
    """
    Apply values (scalar or array) to a 2D array where mask is True (in-place).

    :param arr: 2D array to modify
    :param mask: 2D boolean mask with same shape as arr
    :param values: scalar or 2D array of values to assign where mask is True
    """
    nx, ny = arr.shape

    for i in range(nx):  # type: ignore
        for j in range(ny):
            if mask[i, j]:
                arr[i, j] = values[i, j]


@numba.njit(cache=True)
def _apply_mask_3d(
    arr: np.typing.NDArray, mask: np.typing.NDArray, values: np.typing.NDArray
) -> None:
    """
    Apply values (scalar or array) to a 3D array where mask is True (in-place).

    :param arr: 3D array to modify
    :param mask: 3D boolean mask with same shape as arr
    :param values: scalar or 3D array of values to assign where mask is True
    """
    nx, ny, nz = arr.shape
    for i in range(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    arr[i, j, k] = values[i, j, k]


@numba.njit(cache=True)
def _apply_mask_nd(
    arr: np.typing.NDArray, mask: np.typing.NDArray, values: np.typing.NDArray
) -> None:
    """
    Apply values (scalar or array) to an N-dimensional array where mask is True (in-place).

    :param arr: N-dimensional array to modify
    :param mask: N-dimensional boolean mask with same shape as arr
    :param values: scalar or N-dimensional array of values to assign where mask is True
    """
    for idx in np.ndindex(arr.shape):
        if mask[idx]:
            arr[idx] = values[idx]


@numba.njit(cache=True)
def apply_mask(
    arr: np.typing.NDArray, mask: np.typing.NDArray, values: np.typing.NDArray
) -> None:
    """
    Dispatcher to apply scalar or array values to an array where mask is True.

    :param arr: Array to modify (2D, 3D, or N-dimensional)
    :param mask: Boolean mask with same shape as arr
    :param values: scalar or array of values to assign where mask is True
    """
    ndim = arr.ndim
    if ndim == 2:
        _apply_mask_2d(arr, mask, values)
    elif ndim == 3:
        _apply_mask_3d(arr, mask, values)
    else:
        _apply_mask_nd(arr, mask, values)


@numba.njit(cache=True)
def _get_mask_2d(arr: np.typing.NDArray, mask: np.typing.NDArray, fill_value: float):
    """
    Return a new 2D array where values are kept if mask is True, otherwise replaced with fill_value.

    :param arr: 2D input array
    :param mask: 2D boolean mask with same shape as arr
    :param fill_value: Scalar value to fill where mask is False
    :return: 2D array with masked values applied
    """
    nx, ny = arr.shape
    out = np.empty_like(arr)
    for i in range(nx):  # type: ignore
        for j in range(ny):
            if mask[i, j]:
                out[i, j] = arr[i, j]
            else:
                out[i, j] = fill_value
    return out


@numba.njit(cache=True)
def _get_mask_3d(arr: np.typing.NDArray, mask: np.typing.NDArray, fill_value: float):
    """
    Return a new 3D array where values are kept if mask is True, otherwise replaced with fill_value.

    :param arr: 3D input array
    :param mask: 3D boolean mask with same shape as arr
    :param fill_value: Scalar value to fill where mask is False
    :return: 3D array with masked values applied
    """
    nx, ny, nz = arr.shape
    out = np.empty_like(arr)
    for i in range(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    out[i, j, k] = arr[i, j, k]
                else:
                    out[i, j, k] = fill_value
    return out


@numba.njit(cache=True)
def _get_mask_nd(arr: np.typing.NDArray, mask: np.typing.NDArray, fill_value: float):
    """
    Return a new N-dimensional array where values are kept if mask is True, otherwise replaced with fill_value.

    :param arr: N-dimensional input array
    :param mask: N-dimensional boolean mask with same shape as arr
    :param fill_value: Scalar value to fill where mask is False
    :return: N-dimensional array with masked values applied
    """
    out = np.empty_like(arr)
    for idx in np.ndindex(arr.shape):
        if mask[idx]:
            out[idx] = arr[idx]
        else:
            out[idx] = fill_value
    return out


@numba.njit(cache=True)
def get_mask(
    arr: np.typing.NDArray, mask: np.typing.NDArray, fill_value: float = np.nan
):
    """
    Dispatcher to return a masked copy of an array.

    :param arr: Input array (2D, 3D, or N-dimensional)
    :param mask: Boolean mask with same shape as arr
    :param fill_value: Scalar value to fill where mask is False
    :return: Array with masked values applied
    """
    ndim = arr.ndim
    if ndim == 2:
        return _get_mask_2d(arr, mask, fill_value)
    elif ndim == 3:
        return _get_mask_3d(arr, mask, fill_value)
    return _get_mask_nd(arr, mask, fill_value)


# When used in pure-python, this called
def min_(x) -> np.floating[typing.Any]:
    if isinstance(x, float):
        return x  # type: ignore[return-value]
    return np.min(x)


def max_(x) -> np.floating[typing.Any]:
    if isinstance(x, float):
        return x  # type: ignore[return-value]
    return np.max(x)


# In numba context, these overloads are used
@overload(min_)
def min_overload(x):
    # SCALAR CASE
    if isinstance(x, numba.types.Number):

        def impl(x):
            return x

        return impl

    # ARRAY CASE
    if isinstance(x, numba.types.Array):

        def impl(x):
            return np.min(x)

        return impl


@overload(max_)
def max_overload(x):
    if isinstance(x, numba.types.Number):

        def impl(x):
            return x

        return impl

    if isinstance(x, numba.types.Array):

        def impl(x):
            return np.max(x)

        return impl


def save_as_pickle(
    obj: typing.Any,
    filepath: PathLike,
    exist_ok: bool = False,
    compression: typing.Optional[typing.Literal["gzip", "lzma"]] = "gzip",
    compression_level: int = 6,
) -> None:
    """Saves an object as a pickle file with optional compression.

    :param obj: The object to be saved.
    :param filepath: The path to the pickle file.
    :param exist_ok: If True, will overwrite existing files.
    :param compression: Compression method - "gzip" (fast, good compression),
        "lzma" (slower, better compression), or None
    :param compression_level: Compression level (1-9 for gzip, 0-9 for lzma)
    """
    filepath = Path(filepath)
    if compression == "gzip":
        target_suffix = ".pkl.gz"
    elif compression == "lzma":
        target_suffix = ".pkl.xz"
    else:
        target_suffix = ".pkl"

    if filepath.suffix.split(".")[-1] not in ["pkl", "gz", "xz"]:
        filepath = filepath.with_suffix(target_suffix)
    elif not str(filepath).endswith(target_suffix):
        filepath = Path(str(filepath).replace(".pkl", target_suffix))

    if not exist_ok and filepath.exists():
        raise FileExistsError(f"File {filepath} already exists.")

    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True, mode=0o755)

    # Use pickle protocol 4 or 5 for better performance with large objects
    pickle_protocol = pickle.HIGHEST_PROTOCOL

    if compression == "gzip":
        with gzip.open(filepath, "wb", compresslevel=compression_level) as f:
            pickle.dump(obj, f, protocol=pickle_protocol)
    elif compression == "lzma":
        with lzma.open(filepath, "wb", preset=compression_level) as f:
            pickle.dump(obj, f, protocol=pickle_protocol)
    else:
        with filepath.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle_protocol)


def load_pickle(filepath: PathLike) -> typing.Any:
    """Loads an object from a pickle file with automatic compression detection.

    :param filepath: The path to the pickle file.
    :return: The loaded object.
    """
    filepath = Path(filepath)

    # Auto-detect compression from extension
    if str(filepath).endswith(".gz"):
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)
    elif str(filepath).endswith(".xz"):
        with lzma.open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        with filepath.open("rb") as f:
            return pickle.load(f)


@attrs.frozen(slots=True)
class Lazy(typing.Generic[T]):
    """
    A lazy wrapper that defers computation until accessed.

    Example:
    ```python
    lazy = Lazy(lambda: expensive_computation())
    result = lazy.get()  # Evaluates here
    result2 = lazy.get()  # Returns cached value
    ```
    """

    _factory: typing.Union[T, typing.Callable[[], T]] = attrs.field(repr=False)
    _cached: bool = attrs.field(default=False, init=False, repr=False)
    _value: typing.Optional[T] = attrs.field(default=None, init=False, repr=False)

    @classmethod
    def of(cls, value: T) -> Self:
        """Create a `Lazy` wrapper from an already-computed value."""
        instance = cls(value)
        object.__setattr__(instance, "_cached", True)
        object.__setattr__(instance, "_value", value)
        return instance

    @classmethod
    def defer(cls, factory: typing.Callable[[], T]) -> Self:
        """Create a `Lazy` wrapper from a factory function."""
        return cls(factory)

    def get(self) -> T:
        """Get the value, computing it if necessary."""
        if not self._cached:
            if callable(self._factory):
                value = self._factory()
            else:
                value = self._factory
            object.__setattr__(self, "_value", value)
            object.__setattr__(self, "_cached", True)
        return typing.cast(T, self._value)

    def __getattr__(self, item: str) -> typing.Any:
        if item in {"_factory", "_cached", "_value"}:
            return object.__getattribute__(self, item)
        value = self.get()
        return getattr(value, item)

    def __call__(self, *args, **kwargs) -> typing.Any:
        value = self.get()
        return value(*args, **kwargs)  # type: ignore

    def __getitem__(self, item: typing.Any) -> typing.Any:
        value = self.get()
        return value[item]  # type: ignore

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        val = self.get()
        val[key] = value  # type: ignore

    def is_evaluated(self) -> bool:
        """Check if the lazy value has been evaluated."""
        return self._cached

    def __repr__(self) -> str:
        if self._cached:
            return f"Lazy({self._value!r})"
        return "Lazy(<deferred>)"


class LazyField(typing.Generic[T]):
    """`Lazy` object descriptor"""

    def __init__(self, name: typing.Optional[str] = None):
        self.name = name

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name

    def __get__(self, instance, owner) -> T:
        if instance is None:
            return self  # type: ignore[return-value]
        lazy_obj: Lazy[T] = instance.__dict__[self.name]
        return lazy_obj.get()

    def __set__(
        self, instance, value: typing.Union[T, Lazy[T], typing.Callable[[], T]]
    ) -> None:
        if isinstance(value, Lazy):
            lazy_obj = value
        elif callable(value):
            lazy_obj = Lazy.defer(value)
        else:
            lazy_obj = Lazy.of(value)
        instance.__dict__[self.name] = lazy_obj
