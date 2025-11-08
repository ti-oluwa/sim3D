from datetime import timedelta
import enum
import typing

import attrs
import numpy as np
from typing_extensions import TypeAlias, TypedDict

from sim3D.constants import Constants

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
    "FluidMiscibility",
    "ArrayLike",
    "Interpolator",
    "MixingRule",
    "RelativePermeabilities",
    "FluidPhase",
    "Wettability",
    "WettabilityType",
    "Options",
    "default_options",
    "RateGrids",
    "Time",
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

NDimensionalGrid = np.ndarray[NDimension, np.dtype[np.floating]]

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

EvolutionScheme = typing.Literal["impes", "expes", "adaptive"]
"""
Discretization methods for numerical simulations

- "impes": Implicit pressure, Explicit saturation
- "expes": Both pressure and saturation are treated explicitly
- "adaptive": Adaptive method for pressure and explicit for saturation.
    Adaptive method dynamically switches between explicit and implicit
    based on stability criteria to optimize performance and accuracy.
"""

FluidMiscibility = typing.Literal["logarithmic", "linear", "harmonic"]
"""Miscibility models for fluid interactions in the simulation"""

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


class MixingRule(typing.Protocol):
    """
    Protocol for a mixing rule function that combines two properties
    based on their saturations.
    """

    def __call__(
        self,
        *,
        kro_w: float,
        kro_g: float,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
    ) -> float:
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

    water: float
    oil: float
    gas: float


class WettabilityType(str, enum.Enum):
    """Enum representing the wettability type of the reservoir rock."""

    WATER_WET = "water_wet"
    OIL_WET = "oil_wet"
    MIXED_WET = "mixed_wet"


Wettability = WettabilityType  # Alias for backward compatibility


@typing.runtime_checkable
class RelativePermeabilityFunc(typing.Protocol):
    """
    Protocol for a relative permeability function that computes
    relative permeabilities based on fluid saturations.
    """

    def __call__(self, **kwargs: typing.Any) -> RelativePermeabilities:
        """
        Computes relative permeabilities based on fluid saturations.

        :param kwargs: Additional parameters for the relative permeability function.
        :return: A dictionary containing relative permeabilities for water, oil, and gas phases.
        """
        ...


@attrs.define(slots=True, frozen=True)
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
            raise ValueError("Minimum value cannot be greater than maximum value.")

    def clip(self, value: float) -> float:
        """
        Clips the given value between the minimum and maximum values.

        :param value: The value to be clipped.
        :return: The clipped value.
        """
        return max(self.min, min(self.max, value))

    def arrayclip(self, value: np.typing.NDArray) -> np.typing.NDArray:
        """
        Clips the given NumPy array between the minimum and maximum values.

        :param value: The NumPy array to be clipped.
        :return: The clipped NumPy array.
        """
        return np.clip(value, self.min, self.max)

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
            raise IndexError(
                "Index out of range for Range. Valid indices are 0 and 1."
            )


class RelativeMobilityRange(TypedDict):
    """
    Dictionary holding relative mobility ranges for different phases.
    """

    oil: Range
    water: Range
    gas: Range


@attrs.define(slots=True, frozen=True)
class Options:
    """
    Simulation run options and parameters.
    """

    total_time: float = attrs.field(default=86400.0, validator=attrs.validators.ge(1))
    """Total simulation time in seconds (default is 1 day - 86400.0)."""
    time_step_size: float = attrs.field(default=10, validator=attrs.validators.ge(1))
    """Time step for the simulation in seconds (default is 1 hour)."""
    max_time_steps: int = attrs.field(default=1000, validator=attrs.validators.ge(1))
    """Maximum number of time steps to run for in the simulation."""
    convergence_tolerance: float = attrs.field(
        default=1e-3, validator=attrs.validators.le(1e-2)
    )
    """Convergence tolerance for the simulation."""
    max_iterations_per_evolution: int = attrs.field(
        default=25, validator=attrs.validators.ge(1)
    )
    """Maximum number of iterations allowed per time step for iterative solvers."""
    output_frequency: int = attrs.field(default=10, validator=attrs.validators.ge(1))
    """Frequency of output results during the simulation."""
    scheme: EvolutionScheme = "impes"
    """Evolution scheme to use for the simulation ('impes', 'expes', 'adaptive')."""
    diffusion_number_threshold: float = attrs.field(
        default=0.24,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    """Threshold for the diffusion number to determine stability of the simulation (default is 0.24)."""
    use_pseudo_pressure: bool = False
    """Whether to use pseudo-pressure for gas wells."""
    relative_mobility_range: RelativeMobilityRange = attrs.field(
        default={
            "oil": Range(min=1e-10, max=1e6),
            "water": Range(min=1e-11, max=1e6),
            "gas": Range(min=1e-9, max=1e6),
        }
    )
    """
    Relative mobility ranges for oil, water, and gas phases.

    Each phase has a `Range` object defining its minimum and maximum relative mobility.
    Adjust minimum or maximum values to constrain phase mobilities during simulation.
    """
    capillary_pressure_stability_factor: float = attrs.field(
        default=1.0,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    """
    Factor to scale capillary flow for numerical stability. Reduce to dampen capillary effects.
    Increase to enhance capillary effects.

    Capillary gradients can become numerically dominant in fine meshes or sharp saturation fronts.
    Damping avoids overshoot/undershoot by reducing their contribution without removing them.

    Set to 0 to disable capillary effects entirely (not recommended).
    """
    apply_dip: bool = attrs.field(default=True)
    """Whether to apply reservoir dip effects in the simulation."""
    miscibility_model: MiscibilityModel = "immiscible"
    """Miscibility model: 'immiscible', 'todd_longstaff'"""
    max_cfl_number: typing.Dict[EvolutionScheme, float] = attrs.field(
        factory=lambda: {
            "impes": 10.0,
            "expes": 1.0,
            "adaptive": 5.0,
        }
    )
    """
    Maximum CFL numbers for different evolution schemes to ensure numerical stability.

    Adjust these values based on the chosen evolution scheme:
    - 'impes': Higher CFL number allowed due to implicit pressure treatment.
    - 'expes': Lower CFL number required due to explicit treatment of both pressure and saturation.
    - 'adaptive': Moderate CFL number balancing stability and performance.

    Lowering these values increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """
    constants: Constants = attrs.field(factory=Constants)
    """Physical and conversion constants used in the simulation."""


def Time(
    milliseconds: int = 0,
    seconds: int = 0,
    minutes: int = 0,
    hours: int = 0,
    days: int = 0,
    weeks: int = 0,
) -> float:
    """
    Expresses time components as total seconds.

    :param milliseconds: Number of milliseconds.
    :param seconds: Number of seconds.
    :param minutes: Number of minutes.
    :param hours: Number of hours.
    :param days: Number of days.
    :param weeks: Number of weeks.
    :return: Total time in seconds.
    """
    delta = timedelta(
        weeks=weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )
    return delta.total_seconds()


default_options = Options()

K_con = typing.TypeVar("K_con", contravariant=True)
V_con = typing.TypeVar("V_con", contravariant=True)


class SupportsSetItem(typing.Generic[K_con, V_con], typing.Protocol):
    """
    Protocol for objects that support item assignment.
    """

    def __setitem__(self, key: K_con, value: V_con, /) -> None:
        """Sets the item at the specified key to the given value."""
        ...


@attrs.define(frozen=True, slots=True)
class RateGrids(typing.Generic[NDimension]):
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


@attrs.define(frozen=True, slots=True)
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
