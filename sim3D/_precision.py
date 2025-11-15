from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


__all__ = [
    "get_dtype",
    "set_dtype",
    "with_precision",
    "use_64bit_precision",
    "use_32bit_precision",
    "use_16bit_precision",
]

_sim3d_dtype: ContextVar[np.typing.DTypeLike] = ContextVar(
    "_sim3d_dtype", default=np.float32
)


def get_dtype() -> np.typing.DTypeLike:
    """
    Get the current data type for used for computations in sim3D.

    This defines the precision used in calculations.

    :return: The current data type.
    """
    return _sim3d_dtype.get()


def set_dtype(dtype: np.typing.DTypeLike) -> None:
    """
    Set the default data type for sim3D computations.

    Useful for setting precision for current context.

    :param dtype: The data type to set as default.
    """
    _sim3d_dtype.set(dtype)


@contextmanager
def with_precision(dtype: np.typing.DTypeLike):
    """
    Context manager to temporarily set the data type, and hence the precision for sim3D computations.

    :param dtype: The data type to set within the context.
    """
    token = _sim3d_dtype.set(dtype)
    try:
        yield
    finally:
        _sim3d_dtype.reset(token)


def use_64bit_precision() -> None:
    """
    Set the default data type to float64 for sim3D computations.
    """
    set_dtype(np.float64)


def use_32bit_precision() -> None:
    """
    Set the default data type to float32 for sim3D computations.
    """
    set_dtype(np.float32)


def use_16bit_precision() -> None:
    """
    Set the default data type to float16 for sim3D computations.
    """
    set_dtype(np.float16)
