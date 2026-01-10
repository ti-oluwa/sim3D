import enum
import typing

import attrs
import numpy as np
from scipy.sparse import csr_array, csr_matrix
from scipy.sparse.linalg import LinearOperator
from typing_extensions import TypeAlias, TypedDict

from bores.errors import ValidationError


__all__ = [
    "NDimension",
    "WellLocation",
    "ThreeDimensions",
    "TwoDimensions",
    "OneDimension",
    "ThreeDimensionalGrid",
    "TwoDimensionalGrid",
    "OneDimensionalGrid",
    "Orientation",
    "WellFluidType",
    "EvolutionScheme",
    "MiscibilityModel",
    "ArrayLike",
    "Interpolator",
    "MixingRule",
    "RelativePermeabilities",
    "CapillaryPressures",
    "FluidPhase",
    "Wettability",
    "WettabilityType",
    "Preconditioner",
    "IterativeSolver",
    "IterativeSolverFunc",
    "Range",
    "RelativeMobilityRange",
]

T = typing.TypeVar("T")
Tco = typing.TypeVar("Tco", covariant=True)

S = typing.TypeVar("S")

HookFunc = typing.Callable[[S, T], bool]
"""A function that takes two arguments of types S and T and returns a boolean value."""
ActionFunc = typing.Callable[[S, T], None]
"""A function that takes two arguments of types S and T and returns None."""

NDimension = typing.TypeVar("NDimension", bound=typing.Tuple[int, ...])
WellLocation = typing.TypeVar("WellLocation", bound=typing.Tuple[int, ...])

ThreeDimensions: TypeAlias = typing.Tuple[int, int, int]
"""3D indices"""
TwoDimensions: TypeAlias = typing.Tuple[int, int]
"""2D indices"""
OneDimension: TypeAlias = typing.Tuple[int]
"""1D index"""

Numeric = typing.Union[int, float, np.floating, np.integer]
NDimensionalGrid = np.ndarray[NDimension, np.dtype[np.floating]]
FloatOrArray = typing.Union[float, np.typing.NDArray[np.floating]]


ThreeDimensionalGrid = NDimensionalGrid[ThreeDimensions]
"""3D grid type for simulation data, represented as a 3D NumPy array of floats"""
TwoDimensionalGrid = NDimensionalGrid[TwoDimensions]
"""2D grid type for simulation data, represented as a 2D NumPy array of floats"""
OneDimensionalGrid = NDimensionalGrid[OneDimension]
"""1D grid type for simulation data, represented as a 1D NumPy array of floats"""


class Orientation(enum.Enum):
    """
    Enum representing directional orientation in a 2D/3D simulation.
    """

    X = "x"
    Y = "y"
    Z = "z"


class FluidPhase(enum.Enum):
    """Enum representing the phase of the fluid in the reservoir."""

    WATER = "water"
    GAS = "gas"
    OIL = "oil"


WellFluidType = typing.Literal["water", "oil", "gas"]
"""Types of fluids that can be injected in the simulation"""

EvolutionScheme = typing.Literal["impes", "explicit", "implicit"]
"""
Discretization methods for numerical simulations

- "impes": Implicit pressure, Explicit saturation
- "explicit": Both pressure and saturation are treated explicitly
- "implicit": Both pressure and saturation are treated implicitly
"""

MiscibilityModel = typing.Literal["immiscible", "todd_longstaff"]
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


Interpolator = typing.Callable[[float], float]


PreconditionerStr = typing.Literal["cpr", "ilu", "amg", "diagonal"]
PreconditionerFactory = typing.Callable[
    [typing.Union[csr_array, csr_matrix]], LinearOperator
]
Preconditioner = typing.Union[
    LinearOperator, PreconditionerStr, PreconditionerFactory, str
]

IterativeSolverStr = typing.Literal["gmres", "lgmres", "bicgstab", "tfqmr"]


class IterativeSolverFunc(typing.Protocol):
    """
    Protocol for an iterative solver function.
    """

    def __call__(
        self,
        A: typing.Any,
        b: typing.Any,
        x0: typing.Optional[typing.Any],
        *,
        rtol: float,
        atol: float,
        maxiter: typing.Optional[int],
        M: typing.Optional[typing.Any],
        callback: typing.Optional[typing.Callable[[np.typing.NDArray], None]],
    ) -> np.typing.NDArray: ...


IterativeSolver = typing.Union[IterativeSolverFunc, IterativeSolverStr, str]


class MixingRule(typing.Protocol):
    """
    Protocol for a mixing rule function that combines two properties
    based on their saturations.
    """

    def __call__(
        self,
        *,
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        """
        Combines two properties based on their saturations.

        :param kro_w: Property value for water phase.
        :param kro_g: Property value for gas phase.
        :param water_saturation: Saturation of the water phase.
        :param oil_saturation: Saturation of the oil phase.
        :param gas_saturation: Saturation of the gas phase.
        :return: Combined property value.
        """
        ...


class RelativePermeabilities(TypedDict):
    """Dictionary holding relative permeabilities for different phases."""

    water: FloatOrArray
    oil: FloatOrArray
    gas: FloatOrArray


class CapillaryPressures(typing.TypedDict):
    """Dictionary containing capillary pressures for different phase pairs."""

    oil_water: FloatOrArray  # Pcow = Po - Pw
    gas_oil: FloatOrArray  # Pcgo = Pg - Po


class WettabilityType(str, enum.Enum):
    """Enum representing the wettability type of the reservoir rock."""

    WATER_WET = "water_wet"
    OIL_WET = "oil_wet"
    MIXED_WET = "mixed_wet"


Wettability = WettabilityType  # Alias for backward compatibility


@typing.runtime_checkable
class RelativePermeabilityTable(typing.Protocol):
    """
    Protocol for a relative permeability table that computes
    relative permeabilities based on fluid saturations.
    """

    def __call__(self, **kwargs: typing.Any) -> RelativePermeabilities:
        """
        Computes relative permeabilities based on fluid saturations.

        :param kwargs: Additional parameters for the relative permeability function.
        :return: A dictionary containing relative permeabilities for water, oil, and gas phases.
        """
        ...


@typing.runtime_checkable
class CapillaryPressureTable(typing.Protocol):
    """
    Protocol for a capillary pressure table that computes
    capillary pressures based on fluid saturations.
    """

    def __call__(self, **kwargs: typing.Any) -> CapillaryPressures:
        """
        Computes capillary pressures based on fluid saturations.

        :param kwargs: Saturation parameters (water_saturation, oil_saturation, gas_saturation).
        :return: A dictionary containing capillary pressures for oil-water and gas-oil systems.
        """
        ...


@attrs.frozen(slots=True)
class Range:
    """
    Class representing minimum and maximum values.
    """

    min: float
    """Minimum value."""
    max: float
    """Maximum value."""

    def __attrs_post_init__(self) -> None:
        if self.min > self.max:
            raise ValidationError("Minimum value cannot be greater than maximum value.")

    def clip(self, value: T) -> T:
        """
        Clips the given value between the minimum and maximum values.

        :param value: The value to be clipped.
        :return: The clipped value.
        """
        from bores.utils import clip

        return clip(value, self.min, self.max)

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> typing.Iterator[float]:
        yield self.min
        yield self.max

    def __contains__(self, item: float) -> bool:
        return self.min <= item <= self.max

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.min
        elif index == 1:
            return self.max
        else:
            raise IndexError("Index out of range for Range. Valid indices are 0 and 1.")


class RelativeMobilityRange(TypedDict):
    """
    Dictionary holding relative mobility ranges for different phases.
    """

    oil: Range
    water: Range
    gas: Range


K_con = typing.TypeVar("K_con", contravariant=True)
V_con = typing.TypeVar("V_con", contravariant=True)


class SupportsSetItem(typing.Generic[K_con, V_con], typing.Protocol):
    """
    Protocol for objects that support item assignment.
    """

    def __setitem__(self, key: K_con, value: V_con, /) -> None:
        """Sets the item at the specified key to the given value."""
        ...


GasZFactorMethod = typing.Literal["auto", "papay", "hall-yarborough", "dak"]
