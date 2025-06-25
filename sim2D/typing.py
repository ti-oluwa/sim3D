import typing
import numpy as np
from numpy.typing import NDArray


Tco = typing.TypeVar("Tco", covariant=True)

TwoDimensionalGrid = NDArray[np.float64]
"""2D grid type for simulation data, represented as a 2D NumPy array of floats"""
OneDimensionalGrid = NDArray[np.float64]
"""1D grid type for simulation data, represented as a 1D NumPy array of floats"""

InjectedFluidType = typing.Literal["water", "oil", "gas"]
"""Types of fluids that can be injected in the simulation"""

DiscretizationMethod = typing.Literal["implicit", "explicit", "adaptive"]
"""Discretization methods for numerical simulations"""

FluidMiscibility = typing.Literal["logarithmic", "linear", "harmonic"]
"""Miscibility models for fluid interactions in the simulation"""


class ArrayLike(typing.Generic[Tco], typing.Protocol):
    """
    Protocol for an array-like object that supports
    basic operations like length, indexing, iteration, and containment checks.
    """

    def __len__(self) -> int:
        """Returns the length of the array-like object."""
        ...

    def __getitem__(self, index: int, /) -> Tco:
        """Returns the item at the specified index."""
        ...

    def __iter__(self) -> typing.Iterator[Tco]:
        """Returns an iterator over the items in the array-like object."""
        ...

    def __contains__(self, obj: typing.Any, /) -> bool:
        """Checks if the object is in the array-like object."""
        ...
