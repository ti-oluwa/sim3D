"""Simulation time management with smart and adaptive time stepping."""

from collections import deque
from datetime import timedelta
import logging
import typing

import attrs
from typing_extensions import Self

from bores.errors import TimingError, ValidationError

__all__ = ["Time", "Timer", "TimerState", "StepMetricsDict"]

logger = logging.getLogger(__name__)


def Time(
    milliseconds: float = 0,
    seconds: float = 0,
    minutes: float = 0,
    hours: float = 0,
    days: float = 0,
    weeks: float = 0,
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


class StepMetricsDict(typing.TypedDict):
    """Dictionary representation of step metrics."""

    step_number: int
    step_size: float
    cfl: typing.Optional[float]
    newton_iters: typing.Optional[int]
    success: bool


class TimerState(typing.TypedDict):
    """Complete state of a timer instance for serialization."""

    # Configuration parameters
    initial_step_size: float
    max_step_size: float
    min_step_size: float
    simulation_time: float
    max_cfl_number: float
    cfl_safety_margin: float
    ramp_up_factor: typing.Optional[float]
    max_growth_per_step: float
    step_size_smoothing: float
    growth_cooldown_steps: int
    failure_memory_window: int
    metrics_history_size: int
    use_constant_step_size: bool
    # Current state
    elapsed_time: float
    step: int
    step_size: float
    next_step_size: float
    ema_step_size: float
    last_step_failed: bool
    rejection_count: int
    steps_since_last_failure: int
    # History
    recent_metrics: typing.List[StepMetricsDict]
    failed_step_sizes: typing.List[float]


@attrs.frozen(slots=True)
class StepMetrics:
    """Metrics for a single time step."""

    step_number: int
    step_size: float
    cfl: typing.Optional[float] = None
    newton_iters: typing.Optional[int] = None
    success: bool = True


@attrs.define
class Timer:
    """
    Simulation time manager for smart and adaptive time stepping.
    """

    initial_step_size: float
    """Initial time step size in seconds."""
    max_step_size: float
    """Maximum allowable time step size in seconds."""
    min_step_size: float
    """Minimum allowable time step size in seconds."""
    simulation_time: float
    """Total simulation time in seconds."""
    max_cfl_number: float = 0.9
    """Default maximum CFL number for time step adjustments."""
    ramp_up_factor: typing.Optional[float] = None
    """Factor by which to ramp up time step size on successful steps."""
    backoff_factor: float = 0.5
    """Factor by which to reduce time step size on failed steps."""
    aggressive_backoff_factor: float = 0.25
    """Factor by which to aggressively reduce time step size on failed steps."""
    max_steps: typing.Optional[int] = None
    """Maximum number of time steps to run for."""

    # Adaptive parameters
    growth_cooldown_steps: int = 5
    """Minimum successful steps required before allowing aggressive growth. Higher values lead to more conservative growth."""
    max_growth_per_step: float = 1.3
    """Maximum multiplicative growth allowed per step (e.g., 1.3 = 30% max growth). Lower values lead to much smoother growth."""
    cfl_safety_margin: float = 0.85
    """Safety factor for CFL-based adjustments (target below max CFL)."""
    step_size_smoothing: float = 0.2
    """EMA smoothing factor (0 = no smoothing, 1 = maximum smoothing). Higher values lead to smoother step size changes and higher dampening of fluctuations."""
    metrics_history_size: int = 10
    """Number of recent steps to track for performance analysis."""
    failure_memory_window: int = 5
    """Number of recent failures to remember for adaptive behavior."""

    # State variables
    elapsed_time: float = attrs.field(init=False, default=0.0)
    """Current simulation time in seconds (sum of all accepted steps)."""
    step_size: float = attrs.field(init=False, default=0.0)
    """The time step size (in seconds) that was used for the most recently accepted step."""
    next_step_size: float = attrs.field(init=False, default=0.0)
    """Time step size (in seconds) to propose for the next step."""
    ema_step_size: float = attrs.field(init=False, default=0.0)
    """Exponential moving average of step size for smoothing."""
    step: int = attrs.field(init=False, default=0)
    """Number of accepted time steps completed so far (0-indexed)."""
    last_step_failed: bool = attrs.field(init=False, default=False)
    """Whether the most recent step attempt was rejected."""
    max_rejects: int = 10
    """Maximum number of consecutive time step rejections allowed."""
    rejection_count: int = attrs.field(init=False, default=0)
    """Count of consecutive time step rejections."""
    steps_since_last_failure: int = attrs.field(init=False, default=0)
    """Number of successful steps since the last failure."""
    use_constant_step_size: bool = attrs.field(init=False, default=False)
    """Whether to use a constant time step size."""

    # Performance tracking
    recent_metrics: typing.Deque[StepMetrics] = attrs.field(init=False)
    """Recent step performance metrics."""
    failed_step_sizes: typing.Deque[float] = attrs.field(init=False)
    """Recent failed step sizes for memory."""

    def __attrs_post_init__(self) -> None:
        self.next_step_size = self.initial_step_size
        self.step_size = self.initial_step_size
        self.ema_step_size = self.initial_step_size
        self.use_constant_step_size = (
            self.initial_step_size == self.max_step_size
            and self.initial_step_size == self.min_step_size
        )
        self.recent_metrics = deque(maxlen=self.metrics_history_size)
        self.failed_step_sizes = deque(maxlen=self.failure_memory_window)

    @property
    def next_step(self) -> int:
        """Returns the next time step count."""
        return self.step + 1

    def done(self) -> bool:
        """
        Checks if the simulation has reached its end criteria.

        If True, simulation has reached it ends.
        """
        # Use small tolerance for floating-point comparison
        if self.elapsed_time >= self.simulation_time - 1e-9:
            return True

        if self.max_steps is not None and self.step >= self.max_steps:
            return True
        return False

    @property
    def time_remaining(self) -> float:
        """Calculates the remaining simulation time in seconds."""
        return max(self.simulation_time - self.elapsed_time, 0.0)

    @property
    def is_last_step(self) -> bool:
        """Determines if the latest accepted step was the last one (simulation is now complete)."""
        if self.time_remaining <= 0:
            return True

        if self.max_steps is not None and self.step >= self.max_steps:
            return True
        return False

    def _is_near_failed_size(self, dt: float, tolerance: float = 0.15) -> bool:
        """Check if proposed step size is near a recently failed size."""
        for failed_size in self.failed_step_sizes:
            if abs(dt - failed_size) / failed_size < tolerance:
                return True
        return False

    def _compute_performance_factor(self) -> float:
        """
        Analyze recent performance metrics to compute an adaptive factor.

        Returns a factor in (0, 1] where:
        - 1.0 = excellent performance, allow normal growth
        - <1.0 = concerning trends, be more conservative
        """
        if len(self.recent_metrics) < 3:
            return 1.0

        factor = 1.0

        # Check CFL trend (are we pushing limits?)
        recent_cfls = [
            m.cfl
            for m in list(self.recent_metrics)[-5:]
            if m.cfl is not None and m.success
        ]
        if len(recent_cfls) >= 3:
            # If CFL is consistently high or increasing, be conservative
            avg_cfl = sum(recent_cfls) / len(recent_cfls)
            if avg_cfl > 0.75 * self.max_cfl_number:
                factor *= 0.95

            # Check if CFL is trending upward
            if len(recent_cfls) >= 4:
                cfl_trend = recent_cfls[-1] - recent_cfls[-4]
                if cfl_trend > 0.15:
                    factor *= 0.9

        # Check Newton iteration trends (is solver struggling?)
        recent_iters = [
            m.newton_iters
            for m in list(self.recent_metrics)[-5:]
            if m.newton_iters is not None and m.success
        ]
        if len(recent_iters) >= 3:
            avg_iters = sum(recent_iters) / len(recent_iters)
            if avg_iters > 8:
                factor *= 0.85
            elif all(i > 10 for i in recent_iters[-3:]):
                factor *= 0.75  # Solver consistently struggling

        return max(factor, 0.5)  # Never be too aggressive in reduction

    def propose_step_size(self) -> float:
        """Proposes the next time step size without updating state."""
        if self.use_constant_step_size:
            return self.initial_step_size

        dt = self.next_step_size
        remaining_time = self.time_remaining

        if dt > remaining_time:
            return (
                max(remaining_time, self.min_step_size)
                if remaining_time >= self.min_step_size
                else remaining_time
            )

        logger.debug(
            f"Proposing time step of size {dt} for time step {self.next_step} "
            f"at elapsed time {self.elapsed_time}."
        )
        return dt

    def reject_step(
        self,
        step_size: float,
        aggressive: bool = False,
        failure_severity: typing.Optional[float] = None,
    ) -> float:
        """
        Registers a rejected time step proposal and computes an adjusted time step size.

        :param step_size: The step size that was rejected.
        :param aggressive: Whether to use aggressive backoff.
        :param failure_severity: Optional severity metric (0-1), where 1.0 is catastrophic.
        :return: The new/adjusted time step size in seconds.
        """
        if self.rejection_count >= self.max_rejects:
            raise TimingError(
                "Maximum number of consecutive time step rejections exceeded"
            )

        if self.use_constant_step_size:
            return self.initial_step_size

        # Store failed step size for memory
        self.failed_step_sizes.append(step_size)

        # Record metrics
        metrics = StepMetrics(
            step_number=self.next_step, step_size=step_size, success=False
        )
        self.recent_metrics.append(metrics)

        # Determine backoff factor
        if failure_severity is not None:
            if failure_severity < 0.0 or failure_severity > 1.0:
                raise ValidationError(
                    "failure_severity must be in the range [0.0, 1.0]"
                )

            # Adaptive backoff based on severity
            factor = 1.0 - (failure_severity * (1.0 - self.backoff_factor))
            factor = max(factor, self.aggressive_backoff_factor)
        else:
            factor = (
                self.aggressive_backoff_factor if aggressive else self.backoff_factor
            )

        self.next_step_size *= factor

        # Warn when hitting minimum step size
        if self.next_step_size < self.min_step_size:
            logger.warning(
                f"Step size {self.next_step_size:.6e} would be below minimum "
                f"{self.min_step_size:.6e}. Clamping to minimum."
            )
        self.next_step_size = max(self.next_step_size, self.min_step_size)

        # Check if we're stuck at minimum step size (panic mode detection)
        if self.next_step_size <= self.min_step_size * 1.01:
            min_failures = sum(
                1 for s in self.failed_step_sizes if s <= self.min_step_size * 1.1
            )
            if min_failures >= 3:
                logger.error(
                    f"Repeated failures ({min_failures}) at or near minimum step size "
                    f"({self.min_step_size:.6e}). Simulation may be unstable."
                )

        # Update EMA to reflect the reduction
        self.ema_step_size = self.next_step_size

        self.last_step_failed = True
        self.rejection_count += 1
        self.steps_since_last_failure = 0

        logger.debug(
            f"Time step of size {step_size} rejected for time step {self.next_step} "
            f"at elapsed time {self.elapsed_time}. New size: {self.next_step_size}"
        )
        return self.next_step_size

    def accept_step(
        self,
        step_size: float,
        max_cfl_encountered: typing.Optional[float] = None,
        cfl_threshold: typing.Optional[float] = None,
        newton_iterations: typing.Optional[int] = None,
    ) -> float:
        """
        Registers an accepted time step and computes the next time step size
        based on criteria from the accepted step.

        :param step_size: The time step size that was just accepted.
        :param max_cfl_encountered: The maximum CFL number encountered during the step.
        :param cfl_threshold: The CFL threshold used during the step.
        :param newton_iterations: Number of Newton iterations taken (if applicable).
        :return: The next proposed time step size.
        """
        # Use small tolerance for floating point comparison as step sizes may slightly overshoot
        if step_size > self.time_remaining + 1e-9:
            raise TimingError(
                f"Step size {step_size} exceeds remaining time {self.time_remaining}. "
                "This indicates a bug in the time stepping logic."
            )

        if self.use_constant_step_size:
            return self.initial_step_size

        # Advance time and step count
        self.elapsed_time += step_size
        self.step_size = step_size
        self.step += 1
        self.steps_since_last_failure += 1

        # Record metrics
        metrics = StepMetrics(
            step_number=self.step,
            step_size=step_size,
            cfl=max_cfl_encountered,
            newton_iters=newton_iterations,
            success=True,
        )
        self.recent_metrics.append(metrics)

        # Start with current step size as base
        dt = self.next_step_size

        # CFL-based adjustment with safety margin
        max_cfl = cfl_threshold if cfl_threshold is not None else self.max_cfl_number
        if max_cfl_encountered is not None and max_cfl_encountered > 0.0:
            target_cfl = max_cfl * self.cfl_safety_margin
            cfl_ratio = target_cfl / max_cfl_encountered

            # Cap CFL-based growth to prevent wild jumps when CFL is very low
            cfl_ratio = min(cfl_ratio, self.max_growth_per_step)

            # Be more conservative if we were close to the limit
            if max_cfl_encountered > 0.8 * max_cfl:
                dt *= min(cfl_ratio, 1.0)  # Only decrease or maintain
            else:
                dt *= cfl_ratio

        # Performance-based factor
        performance_factor = self._compute_performance_factor()
        dt *= performance_factor

        # Apply ramp-up factor (only after cooldown period)
        can_ramp_up = (
            self.ramp_up_factor is not None
            and not self.last_step_failed
            and self.steps_since_last_failure >= self.growth_cooldown_steps
        )
        if can_ramp_up:
            dt *= self.ramp_up_factor  # type: ignore

        # Newton iteration-based adjustment
        if newton_iterations is not None:
            if newton_iterations > 10:
                dt *= 0.7
            elif newton_iterations < 4 and self.steps_since_last_failure >= 3:
                dt *= 1.2

        # Limit growth rate relative to current step
        max_allowed_growth = self.step_size * self.max_growth_per_step
        dt = min(dt, max_allowed_growth)

        # Check if we're approaching a previously failed step size
        if self._is_near_failed_size(dt):
            dt *= 0.8  # Be more conservative near failure zones
            logger.debug(
                f"Step size {dt} is near a previously failed size, reducing conservatively"
            )

        # Enforce absolute bounds
        dt = min(dt, self.max_step_size)
        dt = max(dt, self.min_step_size)

        # Apply smoothing via EMA
        if self.ema_step_size == 0.0:
            self.ema_step_size = dt
        else:
            self.ema_step_size = (
                self.step_size_smoothing * self.ema_step_size
                + (1 - self.step_size_smoothing) * dt
            )

        self.next_step_size = self.ema_step_size

        # Reset rejection tracking
        self.last_step_failed = False
        self.rejection_count = 0

        logger.debug(
            f"Time step of size {step_size} accepted for time step {self.step} "
            f"at elapsed time {self.elapsed_time}. Next size: {self.next_step_size:.6f}"
        )
        return self.next_step_size

    def dump_state(self) -> TimerState:
        """
        Serialize the current timer state to a dictionary.

        Returns all the internal state needed to reconstruct this timer's
        exact state at this point in time. Useful for checkpointing, saving
        simulation progress, or debugging.

        :return: `TimerState` dictionary containing all timer state variables

        Example:
        ```python
        timer = Timer(initial_step_size=0.1, simulation_time=1000.0)
        # ... run simulation for a while ...

        # Save timer state
        timer_state = timer.dump_state()
        save_to_file(timer_state, "timer_state.json")

        # Later, restore timer
        timer_state = load_from_file("timer_state.json")
        timer = Timer.load_state(timer_state)
        ```
        """
        return {
            # Configuration parameters
            "initial_step_size": self.initial_step_size,
            "max_step_size": self.max_step_size,
            "min_step_size": self.min_step_size,
            "simulation_time": self.simulation_time,
            "max_cfl_number": self.max_cfl_number,
            "cfl_safety_margin": self.cfl_safety_margin,
            "ramp_up_factor": self.ramp_up_factor,
            "max_growth_per_step": self.max_growth_per_step,
            "step_size_smoothing": self.step_size_smoothing,
            "growth_cooldown_steps": self.growth_cooldown_steps,
            "failure_memory_window": self.failure_memory_window,
            "metrics_history_size": self.metrics_history_size,
            "use_constant_step_size": self.use_constant_step_size,
            # Current state
            "elapsed_time": self.elapsed_time,
            "step": self.step,
            "step_size": self.step_size,
            "next_step_size": self.next_step_size,
            "ema_step_size": self.ema_step_size,
            "last_step_failed": self.last_step_failed,
            "rejection_count": self.rejection_count,
            "steps_since_last_failure": self.steps_since_last_failure,
            # History
            "recent_metrics": [
                typing.cast(StepMetricsDict, attrs.asdict(m))
                for m in self.recent_metrics
            ],
            "failed_step_sizes": list(self.failed_step_sizes),
        }

    @classmethod
    def load_state(cls, state: TimerState) -> Self:
        """
        Reconstruct a timer from a previously saved timer state.

        Creates a new timer instance and restores all internal state from
        the provided dictionary. This is the inverse of `dump_state()`.

        :param state: `TimerState` dictionary containing timer state (from `dump_state()`)
        :return: A new timer instance with the restored state
        :raises `ValidationError`: If state dictionary is invalid or incomplete

        Example:
        ```python
        # Save timer state during simulation
        timer_state = timer.dump_state()

        # Later, restore and continue
        timer = Timer.load_state(timer_state)
        for state in run(model, timer, wells):
            process(state)
        ```
        """
        # Validate required keys
        required_keys = {
            "initial_step_size",
            "max_step_size",
            "min_step_size",
            "simulation_time",
            "elapsed_time",
            "step",
            "step_size",
        }
        missing_keys = required_keys - set(state.keys())
        if missing_keys:
            raise ValidationError(f"Timer state missing required keys: {missing_keys}")

        # Extract configuration parameters
        config_params = {
            "initial_step_size": state["initial_step_size"],
            "max_step_size": state["max_step_size"],
            "min_step_size": state["min_step_size"],
            "simulation_time": state["simulation_time"],
            "max_cfl_number": state.get("max_cfl_number", 1.0),
            "cfl_safety_margin": state.get("cfl_safety_margin", 0.9),
            "ramp_up_factor": state.get("ramp_up_factor"),
            "max_growth_per_step": state.get("max_growth_per_step", 1.5),
            "step_size_smoothing": state.get("step_size_smoothing", 0.7),
            "growth_cooldown_steps": state.get("growth_cooldown_steps", 5),
            "failure_memory_window": state.get("failure_memory_window", 10),
            "metrics_history_size": state.get("metrics_history_size", 20),
            "use_constant_step_size": state.get("use_constant_step_size", False),
        }
        timer = cls(**config_params)  # type: ignore

        # Restore runtime state (use object.__setattr__ since timer may be frozen)
        object.__setattr__(timer, "elapsed_time", state["elapsed_time"])
        object.__setattr__(timer, "step", state["step"])
        object.__setattr__(timer, "step_size", state["step_size"])
        object.__setattr__(
            timer, "next_step_size", state.get("next_step_size", state["step_size"])
        )
        object.__setattr__(
            timer, "ema_step_size", state.get("ema_step_size", state["step_size"])
        )
        object.__setattr__(
            timer, "last_step_failed", state.get("last_step_failed", False)
        )
        object.__setattr__(timer, "rejection_count", state.get("rejection_count", 0))
        object.__setattr__(
            timer,
            "steps_since_last_failure",
            state.get("steps_since_last_failure", 0),
        )

        # Restore history
        recent_metrics_data = state.get("recent_metrics", [])
        recent_metrics = deque(
            [StepMetrics(**m) for m in recent_metrics_data],
            maxlen=timer.metrics_history_size,
        )
        object.__setattr__(timer, "recent_metrics", recent_metrics)

        failed_step_sizes_data = state.get("failed_step_sizes", [])
        failed_step_sizes = deque(
            failed_step_sizes_data, maxlen=timer.failure_memory_window
        )
        object.__setattr__(timer, "failed_step_sizes", failed_step_sizes)

        logger.info(
            f"Timer state loaded: step {timer.step}, "
            f"elapsed time {timer.elapsed_time:.2f}s, "
            f"step size {timer.step_size:.6f}s"
        )
        return timer
