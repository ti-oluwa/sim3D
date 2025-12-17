"""Well control mechanisms for reservoir simulation."""

import logging
import typing

import attrs
import numba

from bores.constants import c
from bores.pvt.core import compute_gas_compressibility_factor
from bores.types import FluidPhase
from bores.wells.core import (
    WellFluid,
    compute_gas_well_rate,
    compute_oil_well_rate,
    compute_required_bhp_for_gas_rate,
    compute_required_bhp_for_oil_rate,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WellControl",
    "RateClamp",
    "ProductionClamp",
    "InjectionClamp",
    "BHPControl",
    "ConstantRateControl",
    "AdaptiveBHPRateControl",
    "MultiPhaseRateControl",
]


WellFluidT_con = typing.TypeVar("WellFluidT_con", bound=WellFluid, contravariant=True)


@typing.runtime_checkable
class RateClamp(typing.Protocol):
    """
    Protocol for a flow rate clamp.

    Determines when a computed flow rate should be clamped to zero
    to prevent unphysical scenarios (e.g., production during injection).
    """

    def __call__(
        self, rate: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """
        Determine if the flow rate should be clamped to zero.

        :param rate: The computed flow rate (bbl/day or ft³/day).
        :param pressure: The reservoir pressure at the well location (psi).
        :param kwargs: Additional context for clamping decision.
        :return: The clamped flow rate if clamping condition is met, else None.
        """
        ...


@typing.runtime_checkable
class WellControl(typing.Protocol[WellFluidT_con]):
    """
    Protocol for well control implementations.

    Defines the interface for computing flow rates under different control schemes.
    """

    def __call__(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: WellFluidT_con,
        formation_volume_factor: float,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
    ) -> float:
        """
        Compute the flow rate based on the control method.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The relative mobility of the fluid phase.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected.
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :return: The flow rate in (bbl/day or ft³/day). Positive for injection, negative for production.
        """
        ...


def _should_return_zero(
    fluid: typing.Optional[WellFluid],
    phase_mobility: float,
    is_active: bool,
) -> bool:
    """Check if well should return zero flow rate."""
    return fluid is None or phase_mobility <= 0.0 or not is_active


def _setup_gas_pseudo_pressure(
    fluid: WellFluid,
    pressure: float,
    temperature: float,
    use_pseudo_pressure: bool,
) -> typing.Tuple[bool, typing.Optional[typing.Any]]:
    """
    Setup pseudo-pressure table for gas wells if needed.

    :return: Tuple of (use_pseudo_pressure, pseudo_pressure_table)
    """
    if not use_pseudo_pressure or pressure <= c.GAS_PSEUDO_PRESSURE_THRESHOLD:
        return False, None

    pseudo_pressure_table = fluid.get_pseudo_pressure_table(
        temperature=temperature, points=c.GAS_PSEUDO_PRESSURE_POINTS
    )
    return True, pseudo_pressure_table


@numba.njit(cache=True)
def _compute_avg_z_factor(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    bottom_hole_pressure: typing.Optional[float] = None,
) -> float:
    """
    Compute average gas compressibility factor.

    :param bottom_hole_pressure: If provided, uses average of reservoir and BHP.
        Otherwise uses reservoir pressure.
    """
    if bottom_hole_pressure is not None:
        avg_pressure = (pressure + bottom_hole_pressure) * 0.5
    else:
        avg_pressure = pressure
    return compute_gas_compressibility_factor(
        pressure=avg_pressure,
        temperature=temperature,
        gas_gravity=gas_gravity,
    )


def _apply_clamp(
    rate: float,
    clamp: typing.Optional["RateClamp"],
    pressure: float,
    control_name: str,
) -> typing.Optional[float]:
    """
    Apply clamping condition if provided.

    :return: Clamped rate if clamp condition is met, None if not clamped (caller should return original rate)
    """
    if clamp is not None:
        clamped_rate = clamp(rate=rate, pressure=pressure)
        if clamped_rate is not None:
            logger.debug(
                f"Clamping rate {rate:.4f} to {clamped_rate:.4f} "
                f"({control_name}, pressure={pressure:.3f} psi)"
            )
            return clamped_rate
    return None


def _compute_required_bhp(
    target_rate: float,
    fluid: WellFluid,
    well_index: float,
    pressure: float,
    temperature: float,
    phase_mobility: float,
    use_pseudo_pressure: bool,
    formation_volume_factor: float,
    fluid_compressibility: typing.Optional[float],
) -> float:
    """
    Compute required BHP to achieve target rate.

    :return: Required bottom hole pressure (psi)
    :raises ValueError: If computation is not possible (e.g., zero mobility)
    :raises ZeroDivisionError: If rate equation has numerical issues
    """
    if fluid.phase == FluidPhase.GAS:
        # Setup pseudo-pressure if needed
        use_pp, pp_table = _setup_gas_pseudo_pressure(
            fluid=fluid,
            pressure=pressure,
            temperature=temperature,
            use_pseudo_pressure=use_pseudo_pressure,
        )

        # Compute Z-factor using reservoir pressure as initial estimate
        avg_z = _compute_avg_z_factor(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=fluid.specific_gravity,
        )
        return compute_required_bhp_for_gas_rate(
            target_rate=target_rate,
            well_index=well_index,
            pressure=pressure,
            temperature=temperature,
            phase_mobility=phase_mobility,
            average_compressibility_factor=avg_z,
            use_pseudo_pressure=use_pp,
            pseudo_pressure_table=pp_table,
            formation_volume_factor=formation_volume_factor,
        )
    # For oil/water
    return compute_required_bhp_for_oil_rate(
        target_rate=target_rate,
        well_index=well_index,
        pressure=pressure,
        phase_mobility=phase_mobility,
        fluid_compressibility=fluid_compressibility,
    )


@attrs.frozen(slots=True)
class ProductionClamp:
    """Clamp condition for production wells."""

    value: float = 0.0
    """Clamp value to return when condition is met."""

    def __call__(
        self, rate: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """Clamp if rate is positive (injection during production)."""
        if rate > 0.0:
            return self.value
        return None


@attrs.frozen(slots=True)
class InjectionClamp:
    """Clamp condition for injection wells."""

    value: float = 0.0
    """Clamp value to return when condition is met."""

    def __call__(
        self, rate: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """Clamp if rate is negative (production during injection)."""
        if rate < 0.0:
            return self.value
        return None


@attrs.frozen(slots=True)
class BHPControl(typing.Generic[WellFluidT_con]):
    """
    Bottom Hole Pressure (BHP) control.

    Computes flow rate based on pressure differential between reservoir and
    wellbore using Darcy's law. This is the traditional well control method.
    """

    bottom_hole_pressure: float = attrs.field(validator=attrs.validators.gt(0))
    """Well bottom-hole flowing pressure in psi."""
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def __call__(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: WellFluidT_con,
        formation_volume_factor: float,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
    ) -> float:
        """
        Compute flow rate using BHP control (Darcy's law).

        Flow rate is proportional to (P_reservoir - P_wellbore).

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :return: Flow rate in (bbl/day or ft³/day).
        """
        # Early return checks
        if _should_return_zero(
            fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
        ):
            return 0.0

        bhp = self.bottom_hole_pressure

        # Compute rate based on fluid phase
        if fluid.phase == FluidPhase.GAS:
            # Setup pseudo-pressure if needed
            use_pp, pp_table = _setup_gas_pseudo_pressure(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
            )
            avg_z = _compute_avg_z_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=fluid.specific_gravity,
                bottom_hole_pressure=bhp,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=bhp,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pp,
                pseudo_pressure_table=pp_table,
                average_compressibility_factor=avg_z,
                formation_volume_factor=formation_volume_factor,
            )
        else:
            # For water and oil wells
            rate = compute_oil_well_rate(
                well_index=well_index,
                pressure=pressure,
                bottom_hole_pressure=bhp,
                phase_mobility=phase_mobility,
                fluid_compressibility=fluid_compressibility,
            )
        logger.info(f"BHP Control: {rate:.4f} (BHP={bhp:.4f}, Pr={pressure:.4f})")

        # Apply clamp condition if any
        clamped = _apply_clamp(
            rate=rate,
            clamp=self.clamp,
            pressure=pressure,
            control_name="BHP control",
        )
        return clamped if clamped is not None else rate

    def __str__(self) -> str:
        """String representation."""
        return f"BHP Control (BHP={self.bottom_hole_pressure:.4f} psi)"


@attrs.frozen(slots=True)
class ConstantRateControl(typing.Generic[WellFluidT_con]):
    """
    Constant rate control.

    Maintains a target flow rate regardless of reservoir pressure,
    as long as the pressure constraint is satisfied.
    """

    target_rate: float
    """Target flow rate (STB/day or SCF/day). Positive for injection, negative for production."""
    target_phase: typing.Union[str, FluidPhase] = attrs.field(converter=FluidPhase)
    """Target fluid phase for the control."""
    minimum_bottom_hole_pressure: typing.Optional[float] = None
    """
    Minimum bottom hole pressure constraint (psi).
    If specified, rate is limited to prevent BHP from dropping below this value.
    """
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def __attrs_post_init__(self) -> None:
        """Validate control parameters."""
        if self.target_rate == 0.0:
            raise ValueError("Target rate cannot be zero. Use well.shut_in() instead.")
        if (
            self.minimum_bottom_hole_pressure is not None
            and self.minimum_bottom_hole_pressure <= 0.0
        ):
            raise ValueError("Minimum bottom hole pressure must be positive.")

    def __call__(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: WellFluidT_con,
        formation_volume_factor: float,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
    ) -> float:
        """
        Return constant target rate, subject to BHP constraint.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :return: Target flow rate if the required bottom hole pressure to produce/inject
            is above or equal to the minimum bottom hole pressure constraint (if any). Otherwise returns 0.0.
        """
        if _should_return_zero(
            fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
        ):
            return 0.0

        if fluid.phase != self.target_phase:
            return 0.0

        target_rate = (
            self.target_rate * formation_volume_factor
        )  # Convert to reservoir rate
        is_production = target_rate < 0.0  # Negative rate indicates production
        min_bhp = self.minimum_bottom_hole_pressure
        # Check if achieving target rate would violate minimum bottom hole pressure constraint
        if min_bhp is not None:
            try:
                required_bhp = _compute_required_bhp(
                    target_rate=target_rate,
                    fluid=fluid,
                    well_index=well_index,
                    pressure=pressure,
                    temperature=temperature,
                    phase_mobility=phase_mobility,
                    use_pseudo_pressure=use_pseudo_pressure,
                    fluid_compressibility=fluid_compressibility,
                    formation_volume_factor=formation_volume_factor,
                )

            except (ValueError, ZeroDivisionError) as exc:
                logger.warning(
                    f"Failed to compute required BHP for target rate {target_rate:.4f}: {exc}. "
                    "Returning 0."
                )
                return 0.0
            else:
                logger.debug(
                    f"Required BHP: {required_bhp:.4f} psi, Reservoir pressure: {pressure:.4f} psi, Fluid phase: {fluid.phase}"
                )
                if is_production:
                    can_achieve_rate = required_bhp >= min_bhp
                else:
                    can_achieve_rate = required_bhp <= min_bhp

                if can_achieve_rate is False:
                    logger.debug(
                        f"Cannot achieve target rate {target_rate:.4f} "
                        f"without violating minimum bottom hole pressure {min_bhp:.3f} psi "
                        f"(required BHP: {required_bhp:.3f} psi, pressure: {pressure:.3f} psi)"
                    )
                    return 0.0

        clamped = _apply_clamp(
            rate=target_rate,
            clamp=self.clamp,
            pressure=pressure,
            control_name="constant rate control",
        )
        return clamped if clamped is not None else target_rate

    def update(
        self,
        target_rate: typing.Optional[float] = None,
        minimum_bottom_hole_pressure: typing.Optional[float] = None,
        clamp: typing.Optional[RateClamp] = None,
    ) -> "ConstantRateControl[WellFluidT_con]":
        """
        Create a new `ConstantRateControl` with updated parameters.

        :param target_rate: New target flow rate. If None, retains existing.
        :param minimum_bottom_hole_pressure: New minimum BHP. If None, retains existing.
        :param clamp: New clamp condition. If None, retains existing.
        :return: New `ConstantRateControl` instance with updated parameters.
        """
        return type(self)(
            target_rate=target_rate or self.target_rate,
            target_phase=self.target_phase,
            minimum_bottom_hole_pressure=(
                minimum_bottom_hole_pressure or self.minimum_bottom_hole_pressure
            ),
            clamp=clamp or self.clamp,
        )

    def __str__(self) -> str:
        """String representation."""
        return f"Constant Rate Control (Rate={self.target_rate:.4f})"


@attrs.frozen(slots=True)
class AdaptiveBHPRateControl(typing.Generic[WellFluidT_con]):
    """
    Adaptive control that switches between rate and BHP control.

    Operates at constant rate until minimum BHP is reached, then switches
    to BHP control. This prevents excessive pressure drawdown while maintaining
    target production/injection when feasible.
    """

    target_rate: float
    """Target flow rate (STB/day or SCF/day). Positive for injection, negative for production."""
    target_phase: typing.Union[str, FluidPhase] = attrs.field(converter=FluidPhase)
    """Target fluid phase for the control."""
    minimum_bottom_hole_pressure: float
    """
    Minimum bottom hole pressure (psi).
    Control switches from rate to BHP control when this limit is reached.
    """
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def __attrs_post_init__(self) -> None:
        """Validate control parameters."""
        if self.target_rate == 0.0:
            raise ValueError(
                "Target rate cannot be zero. Use `well.shut_in()` instead."
            )
        if self.minimum_bottom_hole_pressure <= 0.0:
            raise ValueError("Minimum bottom hole pressure must be positive.")

    def __call__(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: WellFluidT_con,
        formation_volume_factor: float,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
    ) -> float:
        """
        Compute flow rate adaptively.

        Uses rate control if achievable within BHP constraint,
        otherwise switches to BHP control at minimum_bottom_hole_pressure.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :return: Flow rate in (bbl/day or ft³/day).
        """
        # Early return checks
        if _should_return_zero(
            fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
        ):
            return 0.0

        if fluid.phase != self.target_phase:
            return 0.0

        target_rate = (
            self.target_rate * formation_volume_factor
        )  # Convert to reservoir rate
        is_production = target_rate < 0.0  # Negative rate indicates production
        min_bhp = self.minimum_bottom_hole_pressure
        # Compute required BHP to achieve target rate
        try:
            required_bhp = _compute_required_bhp(
                target_rate=target_rate,
                fluid=fluid,
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                formation_volume_factor=formation_volume_factor,
            )
        except (ValueError, ZeroDivisionError) as exc:
            logger.warning(
                f"Failed to compute required BHP for adaptive control: {exc}. "
                "Switching to BHP mode.",
                exc_info=True,
            )
        else:
            logger.debug(
                f"Required BHP: {required_bhp:.4f} psi, Reservoir pressure: {pressure:.4f} psi, Fluid phase: {fluid.phase}"
            )
            if is_production:
                can_achieve_rate = required_bhp >= min_bhp
            else:
                can_achieve_rate = required_bhp <= min_bhp

            if can_achieve_rate:
                # Can achieve target rate without violating minimum bottom hole pressure
                clamped = _apply_clamp(
                    rate=target_rate,
                    clamp=self.clamp,
                    pressure=pressure,
                    control_name="adaptive control - rate mode",
                )
                if clamped is not None:
                    return clamped

                logger.debug(
                    f"Using rate control at {target_rate:.4f} "
                    f"(required BHP: {required_bhp:.3f} psi > minimum: {min_bhp:.3f} psi)"
                )
                return target_rate

        # Target rate would violate minimum bottom hole pressure, switch to BHP control
        logger.debug(
            f"Switching to BHP control at {min_bhp:.3f} psi "
            f"(target rate not achievable within pressure constraints)"
        )

        # Compute rate at minimum bottom hole pressure using same logic as BHP control
        if fluid.phase == FluidPhase.GAS:
            use_pp, pp_table = _setup_gas_pseudo_pressure(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
            )
            avg_z = _compute_avg_z_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=fluid.specific_gravity,
                bottom_hole_pressure=min_bhp,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=min_bhp,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pp,
                pseudo_pressure_table=pp_table,
                average_compressibility_factor=avg_z,
                formation_volume_factor=formation_volume_factor,
            )
        else:
            # For water and oil wells
            rate = compute_oil_well_rate(
                well_index=well_index,
                pressure=pressure,
                bottom_hole_pressure=min_bhp,
                phase_mobility=phase_mobility,
                fluid_compressibility=fluid_compressibility,
            )

        clamped = _apply_clamp(
            rate=rate,
            clamp=self.clamp,
            pressure=pressure,
            control_name="adaptive control - BHP mode",
        )
        return clamped if clamped is not None else rate

    def update(
        self,
        target_rate: typing.Optional[float] = None,
        minimum_bottom_hole_pressure: typing.Optional[float] = None,
        clamp: typing.Optional[RateClamp] = None,
    ) -> "AdaptiveBHPRateControl[WellFluidT_con]":
        """
        Create a new `AdaptiveBHPRateControl` with updated parameters.

        :param target_rate: New target flow rate. If None, retains existing.
        :param minimum_bottom_hole_pressure: New minimum BHP. If None, retains existing.
        :param clamp: New clamp condition. If None, retains existing.
        :return: New `AdaptiveBHPRateControl` instance with updated parameters.
        """
        return type(self)(
            target_rate=target_rate or self.target_rate,
            target_phase=self.target_phase,
            minimum_bottom_hole_pressure=(
                minimum_bottom_hole_pressure or self.minimum_bottom_hole_pressure
            ),
            clamp=clamp or self.clamp,
        )

    def __str__(self) -> str:
        return f"Adaptive BHP/Rate Control (Rate={self.target_rate:.4f}, Min BHP={self.minimum_bottom_hole_pressure:.4f} psi)"


@attrs.frozen(slots=True)
class MultiPhaseRateControl:
    """
    Multi-phase rate control for wells.

    Defines separate rate controls for oil, gas, and water phases.
    """

    oil_control: WellControl
    """Oil phase well control. Ensure that this is intended for oil phase."""
    gas_control: WellControl
    """Gas phase well control. Ensure that this is intended for gas phase."""
    water_control: WellControl
    """Water phase well control. Ensure that this is intended for water phase."""

    def __call__(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: WellFluid,
        formation_volume_factor: float,
        is_active: bool = True,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
    ) -> float:
        """
        Compute flow rate based on fluid phase.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :return: Flow rate in (bbl/day or ft³/day).
        """
        if _should_return_zero(
            fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
        ):
            return 0.0

        if fluid.phase == FluidPhase.OIL:
            return self.oil_control(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
            )
        elif fluid.phase == FluidPhase.GAS:
            return self.gas_control(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
            )
        elif fluid.phase == FluidPhase.WATER:
            return self.water_control(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
            )
        else:
            raise ValueError(f"Unsupported fluid phase: {fluid.phase}")

    def update(
        self,
        oil_control: typing.Optional[WellControl] = None,
        gas_control: typing.Optional[WellControl] = None,
        water_control: typing.Optional[WellControl] = None,
    ) -> "MultiPhaseRateControl":
        """
        Create a new `MultiPhaseRateControl` with updated controls.

        :param oil_control: New oil phase control. If None, retains existing.
        :param gas_control: New gas phase control. If None, retains existing.
        :param water_control: New water phase control. If None, retains existing.
        :return: New `MultiPhaseRateControl` instance with updated controls.
        """
        return type(self)(
            oil_control=oil_control or self.oil_control,
            gas_control=gas_control or self.gas_control,
            water_control=water_control or self.water_control,
        )

    def __str__(self) -> str:
        return f"""
        Multi-Phase Rate Control (
            Oil Control: {self.oil_control!s},
            Gas Control: {self.gas_control!s},
            Water Control: {self.water_control!s}
        )"""
