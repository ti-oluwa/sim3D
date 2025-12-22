from bores.errors import TimingError
from datetime import timedelta
import typing
import logging

import attrs

__all__ = ["Time", "Timer"]

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


@attrs.define
class Timer:
    """
    Simulation time manager for adaptive time stepping.
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
    elapsed_time: float = attrs.field(init=False, default=0.0)
    """Current simulation time in seconds (sum of all accepted steps)."""
    step_size: float = attrs.field(init=False, default=0.0)
    """The time step size (in seconds) that was used for the most recently accepted step."""
    next_step_size: float = attrs.field(init=False, default=0.0)
    """Time step size (in seconds) to propose for the next step."""
    step: int = attrs.field(init=False, default=0)
    """Number of accepted time steps completed so far (0-indexed)."""
    last_step_failed: bool = attrs.field(init=False, default=False)
    """Whether the most recent step attempt was rejected."""
    max_rejects: int = 10
    """Maximum number of consecutive time step rejections allowed."""
    rejection_count: int = attrs.field(init=False, default=0)
    """Count of consecutive time step rejections."""
    use_constant_step_size: bool = attrs.field(init=False, default=False)
    """Whether to use a constant time step size."""

    def __attrs_post_init__(self) -> None:
        self.next_step_size = self.initial_step_size
        self.step_size = self.initial_step_size
        self.use_constant_step_size = (
            self.initial_step_size == self.max_step_size
            and self.initial_step_size == self.min_step_size
        )

    @property
    def next_step(self) -> int:
        """Returns the next time step count."""
        return self.step + 1

    def done(self) -> bool:
        """
        Checks if the simulation has reached its end criteria.

        If True, simulation has reached it ends.
        """
        if self.elapsed_time >= self.simulation_time:
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
        # Check if time has been exhausted
        if self.time_remaining <= 0:
            return True

        # Check if timestep count has been exhausted
        if self.max_steps is not None and self.step >= self.max_steps:
            return True
        return False

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
            f"Proposing time step of size {dt} for time step {self.next_step} at elapsed time {self.elapsed_time}."
        )
        return dt

    def reject_step(self, step_size: float, aggressive: bool = False) -> float:
        """
        Registers a rejected time step proposal and computes an adjusted time step size.

        Reduces the current time step size by the backoff factor. If
        `aggressive` is True, uses the aggressive backoff factor instead.

        :param aggressive: Whether to use aggressive backoff.
        :return: The new/adjusted time step size in seconds.
        """
        if self.rejection_count >= self.max_rejects:
            raise TimingError(
                "Maximum number of consecutive time step rejections exceeded"
            )

        if self.use_constant_step_size:
            return self.initial_step_size

        factor = self.aggressive_backoff_factor if aggressive else self.backoff_factor
        self.next_step_size *= factor
        self.next_step_size = max(self.next_step_size, self.min_step_size)

        self.last_step_failed = True
        self.rejection_count += 1
        logger.debug(
            f"Time step of size {step_size} rejected for times step {self.next_step} at elapsed time {self.elapsed_time}."
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
        :return: The accepted time step size.
        """
        # Ensure step size doesn't exceed remaining time (would indicate a bug)
        if step_size > self.time_remaining:
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

        dt = self.next_step_size

        # Limit by CFL
        max_cfl = cfl_threshold if cfl_threshold is not None else self.max_cfl_number
        if max_cfl_encountered is not None and max_cfl_encountered > 0.0:
            dt *= max_cfl / max_cfl_encountered

        # Apply ramp-up factor
        if self.ramp_up_factor is not None and not self.last_step_failed:
            dt *= self.ramp_up_factor

        # Apply Newton iteration based adjustment
        if newton_iterations is not None:
            if newton_iterations > 10:
                dt *= 0.7
            elif newton_iterations < 4:
                dt *= 1.2

        # Enforce time step size bounds
        dt = min(dt, self.max_step_size)
        dt = max(dt, self.min_step_size)

        self.next_step_size = dt
        # Reset rejection tracking
        self.last_step_failed = False
        self.rejection_count = 0
        logger.debug(
            f"Time step of size {step_size} accepted for time step {self.next_step} at elapsed time {self.elapsed_time}."
        )
        return self.next_step_size
