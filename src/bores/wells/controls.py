"""Well control mechanisms for reservoir simulation."""

import logging
import typing

import attrs
import numba

from bores.constants import c
from bores.errors import ValidationError
from bores.pvt.core import compute_gas_compressibility_factor
from bores.pvt.tables import PVTTables
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

    Determines when a computed flow rate or BHP should be clamped
    to prevent unphysical scenarios (e.g., production during injection).
    """

    def clamp_rate(
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

    def clamp_bhp(
        self, bottom_hole_pressure: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """
        Determine if the bottom-hole pressure should be clamped.

        :param bottom_hole_pressure: The computed bottom-hole pressure (psi).
        :param pressure: The reservoir pressure at the well location (psi).
        :param kwargs: Additional context for clamping decision.
        :return: The clamped bottom-hole pressure if clamping condition is met, else None.
        """
        ...


@typing.runtime_checkable
class WellControl(typing.Protocol[WellFluidT_con]):
    """
    Protocol for well control implementations.

    Defines the interface for computing flow rates under different control schemes.
    """

    def get_flow_rate(
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
        pvt_tables: typing.Optional[PVTTables] = None,
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
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: The flow rate in (bbl/day or ft³/day). Positive for injection, negative for production.
        """
        ...

    def get_bottom_hole_pressure(
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
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """
        Compute the effective bottom-hole pressure for this control at current conditions.

        This is used for semi-implicit treatment in the pressure equation.

        For BHP control: returns the specified BHP
        For rate control: returns the BHP required to achieve target rate
        For adaptive control: returns BHP based on current operating mode

        :param pressure: Reservoir pressure at well location (psi)
        :param temperature: Reservoir temperature (°F)
        :param phase_mobility: Phase mobility (1/cP)
        :param well_index: Well index (md*ft)
        :param fluid: Fluid being produced/injected
        :param formation_volume_factor: Formation volume factor
        :param is_active: Whether well is active
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas
        :param fluid_compressibility: Fluid compressibility (1/psi)
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Effective bottom-hole pressure (psi)
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
    pvt_tables: typing.Optional[PVTTables] = None,
) -> typing.Tuple[bool, typing.Optional[typing.Any]]:
    """
    Setup pseudo-pressure table for gas wells if needed.

    :return: Tuple of (use_pseudo_pressure, pseudo_pressure_table)
    """
    if not use_pseudo_pressure or pressure <= c.GAS_PSEUDO_PRESSURE_THRESHOLD:
        return False, None

    pseudo_pressure_table = fluid.get_pseudo_pressure_table(
        temperature=temperature,
        points=c.GAS_PSEUDO_PRESSURE_POINTS,
        pvt_tables=pvt_tables,
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
    pressure: float,
    control_name: str,
    rate: typing.Optional[float] = None,
    bhp: typing.Optional[float] = None,
    clamp: typing.Optional["RateClamp"] = None,
) -> typing.Optional[float]:
    """
    Apply clamping condition if provided.

    :return: Clamped rate/bhp if clamp condition is met, None if not clamped (caller should return original rate)
    """
    if clamp is not None:
        if rate is not None:
            clamped_rate = clamp.clamp_rate(rate, pressure)
            if clamped_rate is not None:
                logger.debug(
                    f"Clamping rate {rate:.6f} to {clamped_rate:.6f} "
                    f"({control_name}, pressure={pressure:.3f} psi)"
                )
                return clamped_rate
        elif bhp is not None:
            clamped_bhp = clamp.clamp_bhp(bhp, pressure)
            if clamped_bhp is not None:
                logger.debug(
                    f"Clamping BHP {bhp:.6f} to {clamped_bhp:.6f} "
                    f"({control_name}, pressure={pressure:.3f} psi)"
                )
                return clamped_bhp
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
    pvt_tables: typing.Optional[PVTTables] = None,
) -> float:
    """
    Compute required BHP to achieve target rate.

    :return: Required bottom hole pressure (psi)
    :raises ValidationError: If computation is not possible (e.g., zero mobility)
    :raises ZeroDivisionError: If rate equation has numerical issues
    """
    if fluid.phase == FluidPhase.GAS:
        # Setup pseudo-pressure if needed
        use_pp, pp_table = _setup_gas_pseudo_pressure(
            fluid=fluid,
            pressure=pressure,
            temperature=temperature,
            use_pseudo_pressure=use_pseudo_pressure,
            pvt_tables=pvt_tables,
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


@attrs.frozen
class ProductionClamp:
    """Clamp condition for production wells."""

    value: float = 0.0
    """Clamp value to return when condition is met."""

    def clamp_rate(
        self, rate: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """Clamp if rate is positive (injection during production)."""
        if rate > 0.0:
            return self.value
        return None

    def clamp_bhp(
        self, bottom_hole_pressure: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        if bottom_hole_pressure > pressure:
            return pressure
        return None


@attrs.frozen
class InjectionClamp:
    """Clamp condition for injection wells."""

    value: float = 0.0
    """Clamp value to return when condition is met."""

    def clamp_rate(
        self, rate: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        """Clamp if rate is negative (production during injection)."""
        if rate < 0.0:
            return self.value
        return None

    def clamp_bhp(
        self, bottom_hole_pressure: float, pressure: float, **kwargs
    ) -> typing.Optional[float]:
        if bottom_hole_pressure < pressure:
            return pressure
        return None


@attrs.frozen
class BHPControl(typing.Generic[WellFluidT_con]):
    """
    Bottom Hole Pressure (BHP) control.

    Computes flow rate based on pressure differential between reservoir and
    wellbore using Darcy's law. This is the traditional well control method.
    """

    bhp: float = attrs.field(validator=attrs.validators.gt(0))
    """Well bottom-hole flowing pressure in psi."""
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def get_flow_rate(
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
        pvt_tables: typing.Optional[PVTTables] = None,
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
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Flow rate in (bbl/day or ft³/day).
        """
        # Early return checks
        if _should_return_zero(
            fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
        ):
            return 0.0

        bhp = self.bhp

        # Compute rate based on fluid phase
        if fluid.phase == FluidPhase.GAS:
            # Setup pseudo-pressure if needed
            use_pp, pp_table = _setup_gas_pseudo_pressure(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
                pvt_tables=pvt_tables,
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

        logger.info(f"BHP Control: {rate:.6f} (BHP={bhp:.6f}, Pr={pressure:.6f})")
        # Apply clamp condition if any
        clamped = _apply_clamp(
            rate=rate,
            clamp=self.clamp,
            pressure=pressure,
            control_name="BHP control",
        )
        return clamped if clamped is not None else rate

    def get_bottom_hole_pressure(
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
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """
        Return the specified bottom-hole pressure for BHP control.

        :param pressure: Reservoir pressure at well location (psi)
        :param temperature: Reservoir temperature (°F)
        :param phase_mobility: Phase mobility (1/cP)
        :param well_index: Well index (md*ft)
        :param fluid: Fluid being produced/injected
        :param formation_volume_factor: Formation volume factor
        :param is_active: Whether well is active
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas
        :param fluid_compressibility: Fluid compressibility (1/psi)
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Effective bottom-hole pressure (psi)
        """
        if _should_return_zero(
            fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
        ):
            # Return reservoir pressure (no driving force)
            return pressure

        bhp = self.bhp
        return (
            _apply_clamp(
                pressure=pressure,
                control_name="BHP control",
                clamp=self.clamp,
                bhp=bhp,
            )
            or bhp
        )

    def __str__(self) -> str:
        """String representation."""
        return f"BHP Control (BHP={self.bhp:.6f} psi)"


@attrs.frozen
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
    bhp_limit: typing.Optional[float] = None
    """
    Minimum allowable BHP for production wells, and maximum allowable BHP for injection wells.

    If specified, rate is limited to prevent BHP from dropping below this value.
    """
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def __attrs_post_init__(self) -> None:
        """Validate control parameters."""
        if self.target_rate == 0.0:
            raise ValidationError(
                "Target rate cannot be zero. Use `well.shut_in()` instead."
            )
        if self.bhp_limit is not None and self.bhp_limit <= 0.0:
            raise ValidationError("Minimum bottom hole pressure must be positive.")

    def get_flow_rate(
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
        pvt_tables: typing.Optional[PVTTables] = None,
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
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Target flow rate if the required bottom hole pressure to produce/inject
            is above or equal to the minimum bottom hole pressure constraint (if any). Otherwise returns 0.0.
        """
        if (
            _should_return_zero(
                fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
            )
            or fluid.phase != self.target_phase
        ):
            return 0.0

        target_rate = (
            self.target_rate * formation_volume_factor
        )  # Convert to reservoir rate
        is_production = target_rate < 0.0  # Negative rate indicates production
        bhp_limit = self.bhp_limit
        # Check if achieving target rate would violate minimum bottom hole pressure constraint
        if bhp_limit is not None:
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
                    pvt_tables=pvt_tables,
                )

            except (ValueError, ZeroDivisionError) as exc:
                logger.warning(
                    f"Failed to compute required BHP for target rate {target_rate:.6f}: {exc}. "
                    "Returning 0."
                )
                return 0.0
            else:
                logger.debug(
                    f"Required BHP: {required_bhp:.6f} psi, Reservoir pressure: {pressure:.6f} psi, Fluid phase: {fluid.phase}"
                )
                if is_production:
                    can_achieve_rate = required_bhp >= bhp_limit
                else:
                    can_achieve_rate = required_bhp <= bhp_limit

                if can_achieve_rate is False:
                    logger.debug(
                        f"Cannot achieve target rate {target_rate:.6f} "
                        f"without violating bottom hole pressure limit {bhp_limit:.3f} psi "
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

    def get_bottom_hole_pressure(
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
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """
        Compute BHP required to achieve target rate.

        :param pressure: Reservoir pressure at well location (psi)
        :param temperature: Reservoir temperature (°F)
        :param phase_mobility: Phase mobility (1/cP)
        :param well_index: Well index (md*ft)
        :param fluid: Fluid being produced/injected
        :param formation_volume_factor: Formation volume factor
        :param is_active: Whether well is active
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas
        :param fluid_compressibility: Fluid compressibility (1/psi)
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Effective bottom-hole pressure (psi)
        """
        if (
            _should_return_zero(
                fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
            )
            or fluid.phase != self.target_phase
        ):
            return pressure

        target_rate_reservoir = self.target_rate * formation_volume_factor

        # Compute required BHP for target rate
        try:
            required_bhp = _compute_required_bhp(
                target_rate=target_rate_reservoir,
                fluid=fluid,
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=formation_volume_factor,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )
        except (ValueError, ZeroDivisionError) as exc:
            logger.warning(
                f"Cannot compute required BHP: {exc}. Using reservoir pressure."
            )
            return (
                _apply_clamp(
                    pressure=pressure,
                    control_name="constant rate control",
                    clamp=self.clamp,
                    bhp=pressure,
                )
                or pressure
            )

        # Check BHP constraint
        bhp = required_bhp
        bhp_limit = self.bhp_limit
        if bhp_limit is not None:
            is_production = target_rate_reservoir < 0.0

            if is_production:
                # Production: BHP must be >= bhp_limit
                if required_bhp < bhp_limit:
                    logger.debug(
                        f"Required BHP {required_bhp:.4f} < min {bhp_limit:.4f}. "
                        f"Using constraint BHP."
                    )
                    bhp = bhp_limit
            else:
                # Injection: BHP must be <= max_bhp (bhp_limit is actually max here)
                if required_bhp > bhp_limit:
                    logger.debug(
                        f"Required BHP {required_bhp:.4f} > max {bhp_limit:.4f}. "
                        f"Using constraint BHP."
                    )
                    bhp = bhp_limit

        return (
            _apply_clamp(
                pressure=pressure,
                control_name="constant rate control",
                clamp=self.clamp,
                bhp=bhp,
            )
            or bhp
        )

    def update(
        self,
        target_rate: typing.Optional[float] = None,
        bhp_limit: typing.Optional[float] = None,
        clamp: typing.Optional[RateClamp] = None,
    ) -> "ConstantRateControl[WellFluidT_con]":
        """
        Create a new `ConstantRateControl` with updated parameters.

        :param target_rate: New target flow rate. If None, retains existing.
        :param bhp_limit: New minimum BHP. If None, retains existing.
        :param clamp: New clamp condition. If None, retains existing.
        :return: New `ConstantRateControl` instance with updated parameters.
        """
        return type(self)(
            target_rate=target_rate or self.target_rate,
            target_phase=self.target_phase,
            bhp_limit=(bhp_limit or self.bhp_limit),
            clamp=clamp or self.clamp,
        )

    def __str__(self) -> str:
        """String representation."""
        return f"Constant Rate Control (Rate={self.target_rate:.6f})"


@attrs.frozen
class AdaptiveBHPRateControl(typing.Generic[WellFluidT_con]):
    """
    Adaptive control that switches between rate and BHP control.

    Operates at constant rate until BHP limit is reached, then switches
    to BHP control. This prevents excessive pressure drawdown while maintaining
    target production/injection when feasible.
    """

    target_rate: float
    """Target flow rate (STB/day or SCF/day). Positive for injection, negative for production."""
    target_phase: typing.Union[str, FluidPhase] = attrs.field(converter=FluidPhase)
    """Target fluid phase for the control."""
    bhp_limit: float
    """
    Minimum allowable BHP for production wells, and maximum allowable BHP for injection wells.

    Control switches from rate to BHP control when this limit is reached.
    """
    clamp: typing.Optional[RateClamp] = None
    """Condition for clamping flow rates to zero. None means no clamping."""

    def __attrs_post_init__(self) -> None:
        """Validate control parameters."""
        if self.target_rate == 0.0:
            raise ValidationError(
                "Target rate cannot be zero. Use `well.shut_in()` instead."
            )
        if self.bhp_limit <= 0.0:
            raise ValidationError("Minimum bottom hole pressure must be positive.")

    def get_flow_rate(
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
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """
        Compute flow rate adaptively.

        Uses rate control if achievable within BHP constraint,
        otherwise switches to BHP control at bhp_limit.

        :param pressure: Reservoir pressure at the well location (psi).
        :param temperature: Reservoir temperature at the well location (°F).
        :param phase_mobility: Relative mobility of the fluid phase.
        :param well_index: Well index (md*ft).
        :param fluid: Fluid being produced or injected.
        :param is_active: Whether the well is active/open.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells.
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor (bbl/STB or ft³/SCF).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Flow rate in (bbl/day or ft³/day).
        """
        # Early return checks
        if (
            _should_return_zero(
                fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
            )
            or fluid.phase != self.target_phase
        ):
            return 0.0

        target_rate = (
            self.target_rate * formation_volume_factor
        )  # Convert to reservoir rate
        is_production = target_rate < 0.0  # Negative rate indicates production
        bhp_limit = self.bhp_limit
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
                pvt_tables=pvt_tables,
            )
        except (ValueError, ZeroDivisionError) as exc:
            logger.warning(
                f"Failed to compute required BHP for adaptive control: {exc}. "
                "Switching to BHP mode.",
                exc_info=True,
            )
        else:
            logger.debug(
                f"Required BHP: {required_bhp:.6f} psi, Reservoir pressure: {pressure:.6f} psi, Fluid phase: {fluid.phase}"
            )
            if is_production:
                can_achieve_rate = required_bhp >= bhp_limit
            else:
                can_achieve_rate = required_bhp <= bhp_limit

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
                    f"Using rate control at {target_rate:.6f} "
                    f"(required BHP: {required_bhp:.3f} psi > minimum: {bhp_limit:.3f} psi)"
                )
                return target_rate

        # Target rate would violate minimum bottom hole pressure, switch to BHP control
        logger.debug(
            f"Switching to BHP control at {bhp_limit:.3f} psi "
            f"(target rate not achievable within pressure constraints)"
        )

        # Compute rate at minimum bottom hole pressure using same logic as BHP control
        if fluid.phase == FluidPhase.GAS:
            use_pp, pp_table = _setup_gas_pseudo_pressure(
                fluid=fluid,
                pressure=pressure,
                temperature=temperature,
                use_pseudo_pressure=use_pseudo_pressure,
                pvt_tables=pvt_tables,
            )
            avg_z = _compute_avg_z_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=fluid.specific_gravity,
                bottom_hole_pressure=bhp_limit,
            )
            rate = compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=bhp_limit,
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
                bottom_hole_pressure=bhp_limit,
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

    def get_bottom_hole_pressure(
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
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """
        Compute BHP based on current operating mode.

        :param pressure: Reservoir pressure at well location (psi)
        :param temperature: Reservoir temperature (°F)
        :param phase_mobility: Phase mobility (1/cP)
        :param well_index: Well index (md*ft)
        :param fluid: Fluid being produced/injected
        :param formation_volume_factor: Formation volume factor
        :param is_active: Whether well is active
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas
        :param fluid_compressibility: Fluid compressibility (1/psi)
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Effective bottom-hole pressure (psi)
        """
        if (
            _should_return_zero(
                fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
            )
            or fluid.phase != self.target_phase
        ):
            return pressure

        target_rate_reservoir = self.target_rate * formation_volume_factor
        bhp_limit = self.bhp_limit

        # Try to compute required BHP for target rate
        try:
            required_bhp = _compute_required_bhp(
                target_rate=target_rate_reservoir,
                fluid=fluid,
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=formation_volume_factor,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Cannot achieve rate mode: {e}. Using BHP mode.")
            return (
                _apply_clamp(
                    pressure=pressure,
                    control_name="adaptive control - BHP mode",
                    clamp=self.clamp,
                    bhp=bhp_limit,
                )
                or bhp_limit
            )

        # Check if rate is achievable within BHP constraint
        is_production = target_rate_reservoir < 0.0
        if is_production:
            can_achieve = required_bhp >= bhp_limit
        else:
            can_achieve = required_bhp <= bhp_limit

        if can_achieve:
            logger.debug(f"Adaptive control: rate mode (BHP={required_bhp:.4f})")
            bhp = required_bhp
        else:
            logger.debug(f"Adaptive control: BHP mode (BHP={bhp_limit:.4f})")
            bhp = bhp_limit
        return (
            _apply_clamp(
                pressure=pressure,
                control_name="adaptive control",
                clamp=self.clamp,
                bhp=bhp,
            )
            or bhp
        )

    def update(
        self,
        target_rate: typing.Optional[float] = None,
        bhp_limit: typing.Optional[float] = None,
        clamp: typing.Optional[RateClamp] = None,
    ) -> "AdaptiveBHPRateControl[WellFluidT_con]":
        """
        Create a new `AdaptiveBHPRateControl` with updated parameters.

        :param target_rate: New target flow rate. If None, retains existing.
        :param bhp_limit: New minimum BHP. If None, retains existing.
        :param clamp: New clamp condition. If None, retains existing.
        :return: New `AdaptiveBHPRateControl` instance with updated parameters.
        """
        return type(self)(
            target_rate=target_rate or self.target_rate,
            target_phase=self.target_phase,
            bhp_limit=(bhp_limit or self.bhp_limit),
            clamp=clamp or self.clamp,
        )

    def __str__(self) -> str:
        return f"Adaptive BHP/Rate Control (Rate={self.target_rate:.6f}, Min BHP={self.bhp_limit:.6f} psi)"


@attrs.frozen
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

    def get_flow_rate(
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
        pvt_tables: typing.Optional[PVTTables] = None,
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
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: Flow rate in (bbl/day or ft³/day).
        """
        if _should_return_zero(
            fluid=fluid, phase_mobility=phase_mobility, is_active=is_active
        ):
            return 0.0

        if fluid.phase == FluidPhase.OIL:
            return self.oil_control.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )
        elif fluid.phase == FluidPhase.GAS:
            return self.gas_control.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )
        elif fluid.phase == FluidPhase.WATER:
            return self.water_control.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )
        else:
            raise ValidationError(f"Unsupported fluid phase: {fluid.phase}")

    def get_bottom_hole_pressure(
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
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """Delegate to appropriate phase control."""
        if fluid.phase == FluidPhase.OIL:
            return self.oil_control.get_bottom_hole_pressure(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )
        elif fluid.phase == FluidPhase.GAS:
            return self.gas_control.get_bottom_hole_pressure(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )
        elif fluid.phase == FluidPhase.WATER:
            return self.water_control.get_bottom_hole_pressure(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                formation_volume_factor=formation_volume_factor,
                is_active=is_active,
                use_pseudo_pressure=use_pseudo_pressure,
                fluid_compressibility=fluid_compressibility,
                pvt_tables=pvt_tables,
            )
        return pressure

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
