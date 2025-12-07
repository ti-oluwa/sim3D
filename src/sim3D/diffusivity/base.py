import typing
import warnings

import attrs

from sim3D.types import T, ThreeDimensions


__all__ = ["EvolutionResult"]


def _warn_producer_is_injecting(
    production_rate: float,
    well_name: str,
    cell: ThreeDimensions,
    time: float,
    rate_unit: str = "ft³/day",
) -> None:
    """
    Issues a warning if a production well is found to be injecting fluid
    instead of producing it. i.e., if the production rate is positive.
    """
    warnings.warn(
        f"Warning: Production well '{well_name}' at cell {cell} has a positive rate of {production_rate:.2f} {rate_unit}, "
        f"indicating it is no longer producing fluid at {time:.3f} seconds. Production rates should be negative. Please check well configuration.",
        UserWarning,
    )


def _warn_injector_is_producing(
    injection_rate: float,
    well_name: str,
    cell: ThreeDimensions,
    time: float,
    rate_unit: str = "ft³/day",
) -> None:
    """
    Issues a warning if an injection well is found to be producing fluid
    instead of injecting it. i.e., if the injection rate is negative.
    """
    warnings.warn(
        f"Warning: Injection well '{well_name}' at cell {cell} has a negative rate of {injection_rate:.2f} {rate_unit}, "
        f"indicating it is no longer injecting fluid at {time:.3f} seconds. Injection rates should be postive. Please check well configuration.",
        UserWarning,
    )


@attrs.define(slots=True, frozen=True)
class EvolutionResult(typing.Generic[T]):
    value: T
    scheme: typing.Literal["implicit", "explicit"]
