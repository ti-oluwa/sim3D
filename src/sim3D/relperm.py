"""Relative permeability models and mixing rules for multiphase flow simulations."""

from functools import cached_property
import typing
import numpy as np
import numba

import attrs
from scipy.interpolate import interp1d

from sim3D.utils import clip
from sim3D.types import (
    ArrayLike,
    FluidPhase,
    Interpolator,
    MixingRule,
    RelativePermeabilities,
    WettabilityType,
)


__all__ = [
    "TwoPhaseRelPermTable",
    "ThreePhaseRelPermTable",
    "min_rule",
    "stone_I_rule",
    "stone_II_rule",
    "arithmetic_mean_rule",
    "geometric_mean_rule",
    "harmonic_mean_rule",
    "saturation_weighted_interpolation_rule",
    "baker_linear_rule",
    "blunt_rule",
    "hustad_hansen_rule",
    "aziz_settari_rule",
    "eclipse_rule",
    "max_rule",
    "product_saturation_weighted_rule",
    "linear_interpolation_rule",
    "compute_corey_three_phase_relative_permeabilities",
    "BrooksCoreyThreePhaseRelPermModel",
]


"""
Comparison of common three-phase relative permeability mixing rules:

| Rule                | Conservativeness     | Complexity  | Typical Use Case                            |
|---------------------|----------------------|-------------|---------------------------------------------|
| Min                 | Very conservative    | Simple      | Lower bound, safety factor                  |
| Harmonic Mean       | Very conservative    | Simple      | Series flow, tight rocks                    |
| Geometric Mean      | Conservative         | Simple      | General purpose                             |
| Stone I             | Moderate             | Moderate    | Water-wet systems                           |
| Stone II            | Moderate             | Moderate    | Standard industry practice                  |
| Arithmetic Mean     | Optimistic           | Simple      | Upper bound estimate                        |
| Max                 | Very optimistic      | Simple      | Upper bound, sensitivity                    |
| Saturation Weighted | Moderate             | Moderate    | Varying wettability                         |
| Blunt               | Conservative         | Moderate    | Strong water-wet                            |
| Eclipse             | Moderate             | Moderate    | Commercial simulator standard               |
| Aziz-Settari        | Variable             | Moderate    | Empirical tuning                            |
"""


@numba.njit(cache=True)
def min_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Conservative rule for 3-phase oil relative permeability.
    kro = min(kro_w, kro_g)
    """
    return min(kro_w, kro_g)


@numba.njit(cache=True)
def stone_I_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Stone I rule (1970) for 3-phase oil relative permeability.
    kro = (kro_w * kro_g) / (kro_w + kro_g - kro_w * kro_g)
    """
    if kro_w <= 0.0 and kro_g <= 0.0:
        return 0.0
    return (kro_w * kro_g) / max((kro_w + kro_g - kro_w * kro_g), 1e-12)


@numba.njit(cache=True)
def stone_II_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Stone II rule (1973) for 3-phase oil relative permeability.

    kro = kro_w * (So / (So + Sg)) + kro_g * (So / (So + Sw))

    Notes:
        - Ensures smooth interpolation between the oil-water and oil-gas systems.
        - If denominators vanish (e.g., So=0), returns 0.0.
    """
    kro = 0.0
    if (oil_saturation + gas_saturation) > 0.0:
        kro += kro_w * (oil_saturation / (oil_saturation + gas_saturation))
    if (oil_saturation + water_saturation) > 0.0:
        kro += kro_g * (oil_saturation / (oil_saturation + water_saturation))
    return kro


@numba.njit(cache=True)
def arithmetic_mean_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Simple arithmetic mean of oil-water and oil-gas relative permeabilities.

    kro = (kro_w + kro_g) / 2

    Notes:
        - Simple and conservative
        - Does not account for saturation distribution
        - Tends to overestimate kro compared to other methods
    """
    return (kro_w + kro_g) / 2.0


@numba.njit(cache=True)
def geometric_mean_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Geometric mean of oil-water and oil-gas relative permeabilities.

    kro = sqrt(kro_w * kro_g)

    Notes:
        - More conservative than arithmetic mean
        - If either kro_w or kro_g is zero, result is zero
        - Smooth transition between two-phase limits
    """
    return np.sqrt(kro_w * kro_g)


@numba.njit(cache=True)
def harmonic_mean_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Harmonic mean of oil-water and oil-gas relative permeabilities.

    kro = 2 / (1/kro_w + 1/kro_g)

    Notes:
        - Most conservative of the mean rules
        - Heavily weighted by the smaller value
        - Useful for series flow paths
    """
    if kro_w <= 0.0 or kro_g <= 0.0:
        return 0.0
    return 2.0 / (1.0 / kro_w + 1.0 / kro_g)


@numba.njit(cache=True)
def saturation_weighted_interpolation_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Saturation-weighted interpolation between oil-water and oil-gas systems.

    kro = kro_w * (Sw / (Sw + Sg)) + kro_g * (Sg / (Sw + Sg))

    Notes:
        - Interpolates based on ratio of water to gas saturation
        - Reduces to kro_w when Sg=0 (oil-water system)
        - Reduces to kro_g when Sw=0 (oil-gas system)
        - Similar to Stone II but uses different saturation ratios
    """
    total_displacing_phase = water_saturation + gas_saturation
    if total_displacing_phase <= 0.0:
        return max(kro_w, kro_g)  # Pure oil

    water_weight = water_saturation / total_displacing_phase
    gas_weight = gas_saturation / total_displacing_phase

    return kro_w * water_weight + kro_g * gas_weight


@numba.njit(cache=True)
def baker_linear_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Baker's linear interpolation rule (1988).

    kro = kro_w * (1 - Sg_norm) + kro_g * (1 - Sw_norm)

    where Sg_norm and Sw_norm are normalized saturations.

    Notes:
        - Linear interpolation based on normalized saturations
        - Tends to give higher kro values than Stone methods
        - Simple to implement and understand
    """
    # Normalize saturations
    total_sat = water_saturation + oil_saturation + gas_saturation
    if total_sat <= 0.0:
        return 0.0

    sg_norm = gas_saturation / total_sat
    sw_norm = water_saturation / total_sat

    return kro_w * (1.0 - sg_norm) + kro_g * (1.0 - sw_norm)


@numba.njit(cache=True)
def blunt_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Blunt's rule for three-phase relative permeability.

    kro = kro_w * kro_g * (2 - kro_w - kro_g)

    Notes:
        - Developed for strongly water-wet systems
        - Accounts for pore-level displacement mechanisms
        - Generally gives conservative estimates
    """
    if kro_w <= 0.0 or kro_g <= 0.0:
        return 0.0

    return kro_w * kro_g * (2.0 - kro_w - kro_g)


@numba.njit(cache=True)
def hustad_hansen_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Hustad-Hansen rule (1995) for three-phase relative permeability.

    kro = (kro_w * kro_g) / max(kro_w, kro_g)

    Notes:
        - Conservative estimate
        - Ensures kro ≤ min(kro_w, kro_g)
        - Good for intermediate wettability systems
    """
    if kro_w <= 0.0 and kro_g <= 0.0:
        return 0.0

    max_kr = max(kro_w, kro_g, 1e-12)
    return (kro_w * kro_g) / max_kr


def aziz_settari_rule(a: float = 0.5, b: float = 0.5) -> MixingRule:
    """
    Aziz-Settari empirical correlation.

    kro = kro_w^a * kro_g^b

    where a and b are empirical exponents (typically a=0.5, b=0.5).

    Notes:
        - Empirical correlation from petroleum engineering textbook
        - Can be tuned with different exponents
        - Generally conservative

    :param a: Exponent for oil-water system (default 0.5).
    :param b: Exponent for oil-gas system (default 0.5).
    :return: A mixing rule function implementing the Aziz-Settari correlation.
    """

    @numba.njit(cache=True)
    def _rule(
        kro_w: float,
        kro_g: float,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
    ) -> float:
        if kro_w <= 0.0 or kro_g <= 0.0:
            return 0.0
        return (kro_w**a) * (kro_g**b)

    return _rule


@numba.njit(cache=True)
def eclipse_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    ECLIPSE simulator default three-phase rule.

    Similar to Stone II but with saturation normalization.

    kro = kro_w * f_w + kro_g * f_g

    where f_w and f_g are saturation-dependent factors.

    Notes:
        - Used in commercial ECLIPSE simulator
        - Provides smooth transition between phases
        - Handles edge cases robustly
    """
    total_mobile = oil_saturation + water_saturation + gas_saturation

    if total_mobile <= 0.0:
        return 0.0

    # Saturation factors
    if (oil_saturation + gas_saturation) > 0.0:
        f_w = oil_saturation / (oil_saturation + gas_saturation)
    else:
        f_w = 0.0

    if (oil_saturation + water_saturation) > 0.0:
        f_g = oil_saturation / (oil_saturation + water_saturation)
    else:
        f_g = 0.0

    return kro_w * f_w + kro_g * f_g


@numba.njit(cache=True)
def max_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Maximum rule - most optimistic estimate.

    kro = max(kro_w, kro_g)

    Notes:
        - Upper bound for oil relative permeability
        - Rarely used in practice (too optimistic)
        - Useful for sensitivity analysis
    """
    return max(kro_w, kro_g)


@numba.njit(cache=True)
def product_saturation_weighted_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Product of two-phase kr values weighted by oil saturation.

    kro = (kro_w * kro_g) * (So / So_max)^n

    where n is an empirical exponent (typically 0.5-2.0).

    Notes:
        - Accounts for reduction in connectivity at lower oil saturations
        - Empirical parameter n can be tuned to match experimental data
        - Conservative for low oil saturations
    """
    n = 1.0  # Empirical exponent

    if oil_saturation <= 0.0:
        return 0.0

    # Assume maximum oil saturation is 1.0 - Swi - Sgr
    # For simplicity, use total saturation to normalize
    total_sat = water_saturation + oil_saturation + gas_saturation
    if total_sat <= 0.0:
        return 0.0

    so_normalized = oil_saturation / total_sat
    return (kro_w * kro_g) * (so_normalized**n)


@numba.njit(cache=True)
def linear_interpolation_rule(
    kro_w: float,
    kro_g: float,
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
) -> float:
    """
    Simple linear interpolation based on gas-to-water ratio.

    kro = kro_g * R + kro_w * (1 - R)

    where R = Sg / (Sw + Sg)

    Notes:
        - Simple and intuitive
        - Smooth transition between two-phase systems
        - Does not account for three-phase interference effects
    """
    total_displacing = water_saturation + gas_saturation

    if total_displacing <= 0.0:
        # Pure oil - return maximum
        return max(kro_w, kro_g)

    gas_fraction = gas_saturation / total_displacing
    water_fraction = water_saturation / total_displacing
    return (kro_g * gas_fraction) + (kro_w * water_fraction)


@attrs.frozen(slots=True)
class TwoPhaseRelPermTable:
    """
    Two-phase relative permeability lookup table.

    Interpolates relative permeabilities for two fluid phases based on saturation values.

    Example:
    - Oil-Water system: oil is non-wetting, water is wetting
    - Gas-Oil system: gas is non-wetting, oil is wetting
    """

    wetting_phase: FluidPhase
    """The wetting fluid phase, e.g., 'water' (for oil-water (water-wet)) or 'oil' (for gas-oil (oil-wet))."""
    non_wetting_phase: FluidPhase
    """The non-wetting fluid phase, e.g., 'oil' (for oil-water (water-wet)) or 'gas' (for gas-oil (oil-wet))."""
    wetting_phase_saturation: ArrayLike[float]
    """The saturation values for the wetting phase, ranging from 0 to 1."""
    wetting_phase_relative_permeability: ArrayLike[float]
    """Relative permeability values for wetting phase corresponding to the saturation values."""
    non_wetting_phase_relative_permeability: ArrayLike[float]
    """Relative permeability values for non-wetting phase corresponding to the saturation values."""

    @cached_property
    def wetting_phase_interpolator(self) -> Interpolator:
        """Return the interpolator for wetting phase relative permeability."""
        return interp1d(
            self.wetting_phase_saturation,
            self.wetting_phase_relative_permeability,
            bounds_error=False,
            fill_value=(
                self.wetting_phase_relative_permeability[0],  # type: ignore
                self.wetting_phase_relative_permeability[-1],
            ),
        )

    @cached_property
    def non_wetting_phase_interpolator(self) -> Interpolator:
        """Return the interpolator for non-wetting phase relative permeability."""
        return interp1d(
            self.wetting_phase_saturation,
            self.non_wetting_phase_relative_permeability,
            bounds_error=False,
            fill_value=(
                self.non_wetting_phase_relative_permeability[0],  # type: ignore
                self.non_wetting_phase_relative_permeability[-1],
            ),
        )

    def get_interpolators(self) -> typing.Tuple[Interpolator, Interpolator]:
        """Return interpolators for both phases (wetting, non-wetting)."""
        return self.wetting_phase_interpolator, self.non_wetting_phase_interpolator


@attrs.frozen(slots=True)
class ThreePhaseRelPermTable:
    """
    Three-phase relative permeability lookup table, with mixing rules.

    Interpolates relative permeabilities for water, oil, and gas based on saturation values.

    This is the most common approach to handle three-phase relative permeabilities.
    Uses two two-phase tables (oil-water and gas-oil) and a mixing rule for oil in three-phase system.

    The values for the two-phase tables should be obtained from PVT experiments or literature.

    Supported mixing rules: `min_rule`, `stone_I_rule`, `stone_II_rule`, etc.
    Additional custom rules can be defined as needed.
    """

    oil_water_table: TwoPhaseRelPermTable
    """Relative permeability table for oil-water system (water = wetting, oil = non-wetting)."""
    gas_oil_table: TwoPhaseRelPermTable
    """Relative permeability table for gas-oil system (oil = wetting, gas = non-wetting)."""
    mixing_rule: typing.Optional[MixingRule] = None
    """
    Mixing rule function to compute oil relative permeability in three-phase system.

    The function should take the following parameters in order:
    - kro_w: Oil relative permeability from oil-water table
    - kro_g: Oil relative permeability from gas-oil table
    - Sw: Water saturation
    - So: Oil saturation
    - Sg: Gas saturation
    and return the mixed oil relative permeability.
    If None, a simple conservative rule (min(kro_w, kro_g)) is used.
    """

    def __attrs_post_init__(self) -> None:
        """Validate that the tables are set up correctly for three-phase flow."""
        if {
            self.oil_water_table.wetting_phase,
            self.oil_water_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValueError("`oil_water_table` must be between water and oil phases.")
        if {self.gas_oil_table.wetting_phase, self.gas_oil_table.non_wetting_phase} != {
            FluidPhase.OIL,
            FluidPhase.GAS,
        }:
            raise ValueError("`gas_oil_table` must be between oil and gas phases.")

        if self.oil_water_table.wetting_phase == self.gas_oil_table.non_wetting_phase:
            raise ValueError(
                "Wetting phase of `oil_water_table` cannot be the same as non-wetting phase of `gas_oil_table`."
            )
        if self.gas_oil_table.wetting_phase != FluidPhase.OIL:
            raise ValueError(
                "`gas_oil_table` wetting phase must be oil in three-phase system."
            )

    def get_relative_permeabilities(
        self, water_saturation: float, oil_saturation: float, gas_saturation: float
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for oil, water, gas.
        Uses two-phase tables + mixing rule for oil in 3-phase system.

        :param water_saturation: Water saturation (fraction).
        :param oil_saturation: Oil saturation (fraction).
        :param gas_saturation: Gas saturation (fraction).
        :return: A dictionary with relative permeabilities for water, oil, and gas.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        if not (
            0 <= water_saturation <= 1
            and 0 <= oil_saturation <= 1
            and 0 <= gas_saturation <= 1
        ):
            raise ValueError(
                "Saturations must be between 0 and 1. "
                f"Received: Sw={water_saturation}, So={oil_saturation}, Sg={gas_saturation}"
            )

        total_saturation = water_saturation + oil_saturation + gas_saturation
        if abs(total_saturation - 1.0) > 1e-6 and total_saturation > 0.0:
            # Normalize saturations if they do not sum to 1
            water_saturation /= total_saturation
            oil_saturation /= total_saturation
            gas_saturation /= total_saturation

        # For oil-water table
        # krw = wetting phase kr at wetting phase saturation
        # kro_w = non-wetting phase kr at wetting phase saturation
        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            krw = float(
                self.oil_water_table.wetting_phase_interpolator(water_saturation)
            )
            kro_w = float(
                self.oil_water_table.non_wetting_phase_interpolator(water_saturation)
            )
        else:
            # Oil is wetting phase in oil-water table
            kro_w = float(
                self.oil_water_table.wetting_phase_interpolator(oil_saturation)
            )
            krw = float(
                self.oil_water_table.non_wetting_phase_interpolator(oil_saturation)
            )

        # For gas-oil table: oil is wetting phase
        # kro_g = wetting phase kr at oil saturation
        # krg = non-wetting phase kr at oil saturation
        kro_g = float(self.gas_oil_table.wetting_phase_interpolator(oil_saturation))
        krg = float(self.gas_oil_table.non_wetting_phase_interpolator(oil_saturation))

        # Apply mixing rule for three-phase oil relative permeability
        if self.mixing_rule is not None:
            kro = self.mixing_rule(
                kro_w=kro_w,
                kro_g=kro_g,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
            )
        else:
            # Simple conservative rule if no mixing rule supplied
            kro = min(kro_w, kro_g)

        return RelativePermeabilities(water=krw, oil=kro, gas=krg)

    def __call__(
        self,
        *,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for oil, water, gas.

        :param water_saturation: Water saturation (fraction).
        :param oil_saturation: Oil saturation (fraction).
        :param gas_saturation: Gas saturation (fraction).
        :return: A dictionary with relative permeabilities for water, oil, and gas.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        return self.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )


def compute_corey_three_phase_relative_permeabilities(
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    water_exponent: float,
    oil_exponent: float,
    gas_exponent: float,
    wettability: WettabilityType = WettabilityType.WATER_WET,
    mixing_rule: MixingRule = stone_II_rule,
) -> typing.Tuple[float, float, float]:
    """
    Computes relative permeability for water, oil, and gas in a three-phase system.
    Supports water-wet and oil-wet wettability assumptions.

    Uses Corey-type models for krw, krg, and Stone I rule for kro.

    :param water_saturation: Current water saturation (fraction, between 0 and 1).
    :param oil_saturation: Current oil saturation (fraction, between 0 and 1).
    :param gas_saturation: Current gas saturation (fraction, between 0 and 1).
    :param irreducible_water_saturation: Irreducible water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation after water flood (Sorw).
    :param residual_oil_saturation_gas: Residual oil saturation after gas flood (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param water_exponent: Corey exponent for water relative permeability.
    :param oil_exponent: Corey exponent for oil relative permeability (affects Stone I blending).
    :param gas_exponent: Corey exponent for gas relative permeability.
    :param wettability: Wettability type (water-wet or oil-wet).
    :return: (water_relative_permeability, oil_relative_permeability, gas_relative_permeability)
    """
    if not (
        0 <= water_saturation <= 1
        and 0 <= oil_saturation <= 1
        and 0 <= gas_saturation <= 1
    ):
        raise ValueError(
            "Saturations must be between 0 and 1. "
            f"Received: Sw={water_saturation}, So={oil_saturation}, Sg={gas_saturation}"
        )

    total_saturation = water_saturation + oil_saturation + gas_saturation
    if abs(total_saturation - 1.0) > 1e-6 and total_saturation > 0.0:
        water_saturation /= total_saturation
        oil_saturation /= total_saturation
        gas_saturation /= total_saturation

    if wettability == WettabilityType.WATER_WET:
        # 1. Water relperm (wetting phase)
        movable_water_range = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water
        )
        if movable_water_range <= 1e-6:
            effective_water_saturation = 0.0
        else:
            effective_water_saturation = (
                water_saturation - irreducible_water_saturation
            ) / movable_water_range
            effective_water_saturation = clip(
                effective_water_saturation, 0.0, 1.0
            )
        krw = effective_water_saturation**water_exponent

        # 2. Gas relperm (nonwetting)
        movable_gas_range = 1.0 - residual_gas_saturation - residual_oil_saturation_gas
        if movable_gas_range <= 1e-6:
            effective_gas_saturation = 0.0
        else:
            effective_gas_saturation = (
                gas_saturation - residual_gas_saturation
            ) / movable_gas_range
            effective_gas_saturation = clip(effective_gas_saturation, 0.0, 1.0)
        krg = effective_gas_saturation**gas_exponent

        # 3. Oil relperm (intermediate phase) → Stone I blending
        kro = mixing_rule(
            kro_w=(1.0 - krw),
            kro_g=(1.0 - krg),
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )
        kro = kro**oil_exponent  # Apply oil Corey exponent as a curvature control

    elif wettability == WettabilityType.OIL_WET:
        # Oil is wetting, water becomes intermediate
        # 1. Oil relperm (wetting phase)
        movable_oil_range = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas
        )
        if movable_oil_range <= 1e-6:
            effective_oil_saturation = 0.0
        else:
            effective_oil_saturation = (
                oil_saturation
                - min(residual_oil_saturation_water, residual_oil_saturation_gas)
            ) / movable_oil_range
            effective_oil_saturation = clip(effective_oil_saturation, 0.0, 1.0)
        kro = effective_oil_saturation**oil_exponent

        # 2. Gas relperm (nonwetting phase)
        movable_gas_range = 1.0 - residual_gas_saturation - irreducible_water_saturation
        if movable_gas_range <= 1e-6:
            effective_gas_saturation = 0.0
        else:
            effective_gas_saturation = (
                gas_saturation - residual_gas_saturation
            ) / movable_gas_range
            effective_gas_saturation = clip(effective_gas_saturation, 0.0, 1.0)
        krg = effective_gas_saturation**gas_exponent

        # 3. Water relperm (intermediate phase, use Stone I style blending)
        krw = mixing_rule(
            kro_w=(1.0 - kro),  # treat oil as wetting
            kro_g=(1.0 - krg),  # treat gas as nonwetting
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )
        krw = krw**water_exponent

    else:
        raise ValueError(f"Wettability {wettability} not implemented.")

    # Clip all results to [0, 1]
    krw = clip(krw, 0.0, 1.0)
    kro = clip(kro, 0.0, 1.0)
    krg = clip(krg, 0.0, 1.0)
    return float(krw), float(kro), float(krg)


@attrs.frozen(slots=True)
class BrooksCoreyThreePhaseRelPermModel:
    """
    Brooks-Corey-type three-phase relative permeability model.

    Uses the Brooks-Corey model for two-phase relative permeabilities
    and Stone I mixing rule for oil in three-phase system.

    Supports water-wet and oil-wet wettability assumptions.
    """

    irreducible_water_saturation: typing.Optional[float] = None
    """(Default) Irreducible water saturation (Swc)."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """(Default) Residual oil saturation after water flood (Sorw)."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """(Default) Residual oil saturation after gas flood (Sorg)."""
    residual_gas_saturation: typing.Optional[float] = None
    """(Default) Residual gas saturation (Sgr)."""
    water_exponent: float = 2.0
    """Corey exponent for water relative permeability."""
    oil_exponent: float = 2.0
    """Corey exponent for oil relative permeability (affects Stone I blending)."""
    gas_exponent: float = 2.0
    """Corey exponent for gas relative permeability."""
    wettability: WettabilityType = WettabilityType.WATER_WET
    """Wettability type (water-wet or oil-wet)."""
    mixing_rule: MixingRule = stone_II_rule
    """
    Mixing rule function to compute oil relative permeability in three-phase system.

    The function should take the following parameters in order:
    - kro_w: Oil relative permeability from oil-water table
    - kro_g: Oil relative permeability from oil-gas table
    - Sw: Water saturation
    - So: Oil saturation
    - Sg: Gas saturation
    and return the mixed oil relative permeability.
    """

    def get_relative_permeabilities(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        :param water_saturation: Water saturation (fraction).
        :param oil_saturation: Oil saturation (fraction).
        :param gas_saturation: Gas saturation (fraction).
        :param irreducible_water_saturation: Optional override for irreducible water saturation.
        :param residual_oil_saturation_water: Optional override for residual oil saturation after water flood.
        :param residual_oil_saturation_gas: Optional override for residual oil saturation after gas flood.
        :param residual_gas_saturation: Optional override for residual gas saturation.
        :return: A dictionary with relative permeabilities for water, oil, and gas.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        Sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        Sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        Srg = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )
        Swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        if Swc is None or Sorw is None or Sorg is None or Srg is None:
            raise ValueError(
                "Residual saturations must be provided either as arguments or set in the model instance."
                f"Missing values: Swc={Swc}, Sorw={Sorw}, Sorw={Sorg}, Srg={Srg}"
            )

        krw, kro, krg = compute_corey_three_phase_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=Swc,
            residual_oil_saturation_water=Sorw,
            residual_oil_saturation_gas=Sorg,
            residual_gas_saturation=Srg,
            water_exponent=self.water_exponent,
            oil_exponent=self.oil_exponent,
            gas_exponent=self.gas_exponent,
            wettability=self.wettability,
            mixing_rule=self.mixing_rule,
        )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)

    def __call__(
        self,
        *,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        :param water_saturation: Water saturation (fraction).
        :param oil_saturation: Oil saturation (fraction).
        :param gas_saturation: Gas saturation (fraction).
        :kwarg irreducible_water_saturation: Optional override for irreducible water saturation.
        :kwarg residual_oil_saturation_water: Optional override for residual oil saturation after water flood.
        :kwarg residual_oil_saturation_gas: Optional override for residual oil saturation after gas flood.
        :kwarg residual_gas_saturation: Optional override for residual gas saturation.
        :return: A dictionary with relative permeabilities for water, oil, and gas.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        return self.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get(
                "irreducible_water_saturation", None
            ),
            residual_oil_saturation_water=kwargs.get(
                "residual_oil_saturation_water", None
            ),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas", None),
            residual_gas_saturation=kwargs.get("residual_gas_saturation", None),
        )
