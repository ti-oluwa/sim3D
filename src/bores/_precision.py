from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


__all__ = [
    "get_dtype",
    "set_dtype",
    "with_precision",
    "use_64bit_precision",
    "use_32bit_precision",
    "get_floating_point_info",
]

_bores_dtype: ContextVar[np.typing.DTypeLike] = ContextVar(
    "_bores_dtype", default=np.float32
)


def get_dtype() -> np.typing.DTypeLike:
    """
    Get the current data type for used for computations in bores.

    This defines the precision used in calculations.

    :return: The current data type.
    """
    return _bores_dtype.get()


def set_dtype(dtype: np.typing.DTypeLike) -> None:
    """
    Set the default data type for bores computations.

    Useful for setting precision for current context.

    :param dtype: The data type to set as default.
    """
    _bores_dtype.set(dtype)


@contextmanager
def with_precision(dtype: np.typing.DTypeLike):
    """
    Context manager to temporarily set the data type, and hence the precision for bores computations.

    :param dtype: The data type to set within the context.
    """
    token = _bores_dtype.set(dtype)
    try:
        yield
    finally:
        _bores_dtype.reset(token)


def use_128bit_precision() -> None:
    """
    Set the default data type to float128 for bores computations.
    """
    set_dtype(np.float128)


def use_64bit_precision() -> None:
    """
    Set the default data type to float64 for bores computations.
    """
    set_dtype(np.float64)


def use_32bit_precision() -> None:
    """
    Set the default data type to float32 for bores computations.

    Default precision for bores.
    """
    set_dtype(np.float32)


def get_floating_point_info() -> np.finfo[np.floating]:
    """
    Get the floating point information for the current data type used in bores computations.

    :return: The floating point information.
    """
    dtype = get_dtype()
    return np.finfo(dtype)  # type: ignore
