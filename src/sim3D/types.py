from datetime import timedelta
import enum
import typing

import attrs
import numpy as np
from scipy.sparse import csr_array, csr_matrix
from scipy.sparse.linalg import LinearOperator
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
    "MiscibilityModel",
    "ArrayLike",
    "Interpolator",
    "MixingRule",
    "RelativePermeabilities",
    "CapillaryPressures",
    "FluidPhase",
    "Wettability",
    "WettabilityType",
    "Options",
    "default_options",
    "Time",
    "PhaseFluxDerivatives",
    "FluxDerivativesWithRespectToSaturations",
    "MisciblePhaseFluxDerivatives",
    "MiscibleFluxDerivativesWithRespectToSaturations",
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
FloatOrArray = typing.TypeVar("FloatOrArray", float, np.typing.NDArray[np.floating])


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


PreconditionerStr = typing.Literal["ilu", "amg", "diagonal"]
PreconditionerFactory = typing.Callable[
    [typing.Union[csr_array, csr_matrix]],
    typing.Union[LinearOperator, np.typing.NDArray],
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


class CapillaryPressures(typing.TypedDict):
    """Dictionary containing capillary pressures for different phase pairs."""

    oil_water: float  # Pcow = Po - Pw
    gas_oil: float  # Pcgo = Pg - Po


@attrs.frozen(slots=True)
class PhaseFluxDerivatives:
    """
    Derivatives of a single phase flux with respect to saturations at cell and neighbour.

    For a phase flux F_phase, stores:
    - ∂F_phase/∂S_water_cell: derivative w.r.t. water saturation at the current cell
    - ∂F_phase/∂S_water_neighbour: derivative w.r.t. water saturation at the neighbour cell
    - ∂F_phase/∂S_oil_cell: derivative w.r.t. oil saturation at the current cell
    - ∂F_phase/∂S_oil_neighbour: derivative w.r.t. oil saturation at the neighbour cell
    """

    derivative_wrt_water_saturation_at_cell: float
    derivative_wrt_water_saturation_at_neighbour: float
    derivative_wrt_oil_saturation_at_cell: float
    derivative_wrt_oil_saturation_at_neighbour: float


@attrs.frozen(slots=True)
class FluxDerivativesWithRespectToSaturations:
    """
    Complete set of flux derivatives for all three phases.

    Each phase (water, oil, gas) has derivatives with respect to all saturations
    at both the current cell and its neighbour, for use in Jacobian assembly.
    """

    water_phase_flux_derivatives: PhaseFluxDerivatives
    oil_phase_flux_derivatives: PhaseFluxDerivatives
    gas_phase_flux_derivatives: PhaseFluxDerivatives


@attrs.frozen(slots=True)
class MisciblePhaseFluxDerivatives:
    """
    Derivatives of a single phase flux with respect to saturations AND solvent concentration.

    For miscible displacement (e.g., CO2-EOR), the solvent can be:
    1. Free gas phase (tracked by gas_saturation)
    2. Dissolved in oil (tracked by solvent_concentration)

    This includes all derivatives for the immiscible case plus solvent concentration terms.
    """

    derivative_wrt_water_saturation_at_cell: float
    derivative_wrt_water_saturation_at_neighbour: float
    derivative_wrt_oil_saturation_at_cell: float
    derivative_wrt_oil_saturation_at_neighbour: float
    derivative_wrt_gas_saturation_at_cell: float
    derivative_wrt_gas_saturation_at_neighbour: float
    derivative_wrt_solvent_concentration_at_cell: float
    derivative_wrt_solvent_concentration_at_neighbour: float


@attrs.frozen(slots=True)
class MiscibleFluxDerivativesWithRespectToSaturations:
    """
    Complete set of flux derivatives for miscible displacement (4 equations).

    Includes derivatives for:
    - Three phase saturations (water, oil, gas)
    - Solvent concentration in oil phase

    Each flux has derivatives w.r.t. all four variables at both cell and neighbour.
    """

    water_phase_flux_derivatives: MisciblePhaseFluxDerivatives
    oil_phase_flux_derivatives: MisciblePhaseFluxDerivatives
    gas_phase_flux_derivatives: MisciblePhaseFluxDerivatives
    solvent_mass_flux_derivatives: MisciblePhaseFluxDerivatives


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
            raise IndexError("Index out of range for Range. Valid indices are 0 and 1.")


class RelativeMobilityRange(TypedDict):
    """
    Dictionary holding relative mobility ranges for different phases.
    """

    oil: Range
    water: Range
    gas: Range


@attrs.frozen(slots=True)
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
        default=1e-6, validator=attrs.validators.le(1e-2)
    )
    """Convergence tolerance for iterative solvers (default is 1e-6)."""
    max_iterations: int = attrs.field(
        default=200,
        validator=attrs.validators.and_(
            attrs.validators.ge(1), attrs.validators.le(250)
        ),
    )
    """
    Maximum number of iterations allowed per time step for iterative solvers.
    
    Capped at 250 to prevent excessive computation time in case of non-convergence.
    If the solver does not converge within this limit, the matrix is most likely
    ill-conditioned or the problem setup needs to be reviewed. Use a stronger
    preconditioner, try another solver, or adjust simulation parameters accordingly.
    """
    output_frequency: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    """Frequency at which model states are yielded/outputted during the simulation."""
    scheme: EvolutionScheme = "impes"
    """Evolution scheme to use for the simulation ('impes', 'explicit', 'implicit')."""
    use_pseudo_pressure: bool = True
    """Whether to use pseudo-pressure for gas (when applicable)."""
    relative_mobility_range: RelativeMobilityRange = attrs.field(
        default=RelativeMobilityRange(
            oil=Range(min=1e-12, max=1e6),
            water=Range(min=1e-12, max=1e6),
            gas=Range(min=1e-12, max=1e6),
        )
    )
    """
    Relative mobility ranges for oil, water, and gas phases.

    Each phase has a `Range` object defining its minimum and maximum relative mobility.
    Adjust minimum or maximum values to constrain phase mobilities during simulation.
    """
    capillary_strength_factor: float = attrs.field(
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
    disable_capillary_effects: bool = False
    """Whether to include capillary pressure effects in the simulation."""
    disable_structural_dip: bool = attrs.field(default=False)
    """Whether to disable structural dip effects in reservoir modeling/simulation."""
    miscibility_model: MiscibilityModel = "immiscible"
    """Miscibility model: 'immiscible', 'todd_longstaff'"""
    max_cfl_number: typing.Dict[EvolutionScheme, float] = attrs.field(
        factory=lambda: {
            "impes": 10.0,
            "explicit": 1.0,
            # Fully implicit scheme can handle larger CFL numbers, but we set a conservative default
            "implicit": 20.0,
        }
    )
    """
    Maximum CFL numbers for different evolution schemes to ensure numerical stability.

    Adjust these values based on the chosen evolution scheme:
    - 'impes': Higher CFL number allowed due to implicit pressure treatment.
    - 'explicit': Lower CFL number required due to explicit treatment of both pressure and saturation.

    Lowering these values increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """
    constants: Constants = attrs.field(factory=Constants)
    """Physical and conversion constants used in the simulation."""
    warn_rates_anomalies: bool = True
    """Whether to warn about anomalous flow rates during the simulation."""
    progress_log_interval: int = attrs.field(
        default=10, validator=attrs.validators.ge(1)
    )
    """Interval (in time steps) at which to log simulation progress."""
    preconditioner: typing.Optional[Preconditioner] = "cpr"
    """Preconditioner to use for iterative solvers."""
    iterative_solver: typing.Union[
        IterativeSolver, typing.Iterable[IterativeSolver]
    ] = "lgmres"
    """Iterative solver(s) to use for solving linear systems."""


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
