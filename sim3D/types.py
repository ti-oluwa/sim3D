import enum
import typing

import attrs
import numpy as np
from typing_extensions import TypeAlias, TypedDict

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
]

T = typing.TypeVar("T")
Tco = typing.TypeVar("Tco", covariant=True)

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

EvolutionScheme = typing.Literal[
    "implicit_explicit",
    "explicit_implicit",
    "fully_explicit",
    "fully_implicit",
    "adaptive_explicit",
]
"""
Discretization methods for numerical simulations

- "implicit_explicit": Implicit pressure, Explicit saturation
- "explicit_implicit": Explicit pressure, Implicit saturation
- "fully_explicit": Both pressure and saturation are treated explicitly
- "fully_implicit": Both pressure and saturation are treated implicitly
- "adaptive_explicit": Adaptive method for pressure and explicit for saturation.
    Adaptive method dynamically switches between explicit and implicit
    based on stability criteria to optimize performance and accuracy.
"""

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
class Options:
    """
    Simulation run options and parameters.
    """

    time_step_size: float = attrs.field(default=10, validator=attrs.validators.ge(1))
    """Time step for the simulation in seconds (default is 1 hour)."""
    total_time: float = attrs.field(default=86400.0, validator=attrs.validators.ge(1))
    """Total simulation time in seconds (default is 1 day - 86400.0)."""
    max_iterations: int = attrs.field(default=1000, validator=attrs.validators.ge(1))
    """Maximum number of iterations for the simulation."""
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
    evolution_scheme: EvolutionScheme = "implicit_explicit"
    """Evolution scheme to use for the simulation ('implicit_explicit', 'fully_explicit', 'fully_implicit', 'adaptive_explicit')."""
    diffusion_number_threshold: float = attrs.field(
        default=0.24,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    """Threshold for the diffusion number to determine stability of the simulation (default is 0.24)."""
    minimum_allowable_pressure: float = attrs.field(default=1.45)
    """Minimum allowable pressure in psi to prevent negative pressures."""
    maximum_allowable_pressure: float = attrs.field(
        default=5000, validator=attrs.validators.ge(14.5)
    )
    """Maximum allowable pressure in psi to prevent unrealistic pressures."""
    use_pseudo_pressure: bool = False
    """Whether to use pseudo-pressure for gas wells."""
    minimum_allowable_relative_mobility: float = attrs.field(
        default=1e-8,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    """Minimum allowable relative mobility to prevent numerical issues."""
    capillary_pressure_stability_factor: float = attrs.field(
        default=1.0,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    """
    Factor to scale capillary flow for numerical stability.
    Capillary gradients can become numerically dominant in fine meshes or sharp saturation fronts.
    Damping avoids overshoot/undershoot by reducing their contribution without removing them.
    """
