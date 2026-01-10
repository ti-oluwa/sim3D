"""Relative permeability models and mixing rules for multiphase flow simulations."""

import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.types import (
    FloatOrArray,
    FluidPhase,
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
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Conservative rule for 3-phase oil relative permeability.
    kro = min(kro_w, kro_g)
    """
    return np.minimum(kro_w, kro_g)


@numba.njit(cache=True)
def stone_I_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Stone I rule (1970) for 3-phase oil relative permeability.
    kro = (kro_w * kro_g) / (kro_w + kro_g - kro_w * kro_g)
    """
    denom = np.maximum(((kro_w + kro_g) - (kro_w * kro_g)), 1e-12)
    result = (kro_w * kro_g) / denom
    # Return 0 if both kro_w and kro_g are zero
    return np.where((kro_w <= 0.0) & (kro_g <= 0.0), 0.0, result)


@numba.njit(cache=True)
def stone_II_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Stone II rule (1973) for 3-phase oil relative permeability.

    kro = kro_w * (So / (So + Sg)) + kro_g * (So / (So + Sw))

    Notes:
        - Ensures smooth interpolation between the oil-water and oil-gas systems.
        - If denominators vanish (e.g., So=0), returns 0.0.
    """
    denom_1 = oil_saturation + gas_saturation
    term_1 = np.where(denom_1 > 0.0, kro_w * (oil_saturation / denom_1), 0.0)

    denom_2 = oil_saturation + water_saturation
    term_2 = np.where(denom_2 > 0.0, kro_g * (oil_saturation / denom_2), 0.0)
    return term_1 + term_2


@numba.njit(cache=True)
def arithmetic_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
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
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
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
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Harmonic mean of oil-water and oil-gas relative permeabilities.

    kro = 2 / (1/kro_w + 1/kro_g)

    Notes:
        - Most conservative of the mean rules
        - Heavily weighted by the smaller value
        - Useful for series flow paths
    """
    result = 2.0 / ((1.0 / kro_w) + (1.0 / kro_g))
    # Return 0 if either kro_w or kro_g is zero
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)


@numba.njit(cache=True)
def saturation_weighted_interpolation_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
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

    water_weight = np.where(
        total_displacing_phase > 0.0, water_saturation / total_displacing_phase, 0.0
    )
    gas_weight = np.where(
        total_displacing_phase > 0.0, gas_saturation / total_displacing_phase, 0.0
    )

    result = (kro_w * water_weight) + (kro_g * gas_weight)
    # Return maximum of kro_w and kro_g if total_displacing_phase is zero (pure oil)
    return np.where(total_displacing_phase > 0.0, result, np.maximum(kro_w, kro_g))


@numba.njit(cache=True)
def baker_linear_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Baker's linear interpolation rule (1988).

    kro = kro_w * (1 - Sg_norm) + kro_g * (1 - Sw_norm)

    where `Sg_norm` and `Sw_norm` are normalized saturations.

    Notes:
        - Linear interpolation based on normalized saturations
        - Tends to give higher kro values than Stone methods
        - Simple to implement and understand
    """
    # Normalize saturations
    total_sat = water_saturation + oil_saturation + gas_saturation

    sg_norm = np.where(total_sat > 0.0, gas_saturation / total_sat, 0.0)
    sw_norm = np.where(total_sat > 0.0, water_saturation / total_sat, 0.0)

    result = kro_w * (1.0 - sg_norm) + kro_g * (1.0 - sw_norm)
    return np.where(total_sat > 0.0, result, 0.0)


@numba.njit(cache=True)
def blunt_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Blunt's rule for three-phase relative permeability.

    kro = kro_w * kro_g * (2 - kro_w - kro_g)

    Notes:
        - Developed for strongly water-wet systems
        - Accounts for pore-level displacement mechanisms
        - Generally gives conservative estimates
    """
    result = kro_w * kro_g * (2.0 - kro_w - kro_g)
    # Return 0 if either kro_w or kro_g is zero
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)


@numba.njit(cache=True)
def hustad_hansen_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Hustad-Hansen rule (1995) for three-phase relative permeability.

    kro = (kro_w * kro_g) / max(kro_w, kro_g)

    Notes:
        - Conservative estimate
        - Ensures kro ≤ min(kro_w, kro_g)
        - Good for intermediate wettability systems
    """
    max_kr = np.maximum(np.maximum(kro_w, kro_g), 1e-12)
    result = (kro_w * kro_g) / max_kr
    # Return 0 if both kro_w and kro_g are zero
    return np.where((kro_w <= 0.0) & (kro_g <= 0.0), 0.0, result)


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
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        result = (kro_w**a) * (kro_g**b)
        # Return 0 if either kro_w or kro_g is zero
        return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)

    return _rule


@numba.njit(cache=True)
def eclipse_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
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

    # Saturation factors - use np.where to avoid type inconsistency
    denom_w = oil_saturation + gas_saturation
    f_w = np.where(denom_w > 0.0, oil_saturation / denom_w, 0.0)

    denom_g = oil_saturation + water_saturation
    f_g = np.where(denom_g > 0.0, oil_saturation / denom_g, 0.0)

    # Return 0 if total_mobile is zero, otherwise compute kro
    result = (kro_w * f_w) + (kro_g * f_g)
    return np.where(total_mobile > 0.0, result, 0.0)


@numba.njit(cache=True)
def max_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Maximum rule - most optimistic estimate.

    kro = max(kro_w, kro_g)

    Notes:
        - Upper bound for oil relative permeability
        - Rarely used in practice (too optimistic)
        - Useful for sensitivity analysis
    """
    return np.maximum(kro_w, kro_g)


@numba.njit(cache=True)
def product_saturation_weighted_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
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

    # Assume maximum oil saturation is 1.0 - Swi - Sgr
    # For simplicity, use total saturation to normalize
    total_sat = water_saturation + oil_saturation + gas_saturation

    so_normalized = np.where(total_sat > 0.0, oil_saturation / total_sat, 0.0)
    result = (kro_w * kro_g) * (so_normalized**n)

    # Return 0 if oil_saturation or total_sat is zero
    return np.where((oil_saturation > 0.0) & (total_sat > 0.0), result, 0.0)


@numba.njit(cache=True)
def linear_interpolation_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
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

    gas_fraction = np.where(
        total_displacing > 0.0, gas_saturation / total_displacing, 0.0
    )
    water_fraction = np.where(
        total_displacing > 0.0, water_saturation / total_displacing, 0.0
    )

    result = (kro_g * gas_fraction) + (kro_w * water_fraction)
    # Pure oil - return maximum
    return np.where(total_displacing > 0.0, result, np.maximum(kro_w, kro_g))


@attrs.frozen
class TwoPhaseRelPermTable:
    """
    Two-phase relative permeability lookup table.

    Interpolates relative permeabilities for two fluid phases based on saturation values.
    Uses `np.interp` for fast vectorized interpolation.

    Supports both scalar and array inputs up to 3D.

    Example:
    - Oil-Water system: oil is non-wetting, water is wetting
    - Gas-Oil system: gas is non-wetting, oil is wetting
    """

    wetting_phase: FluidPhase
    """The wetting fluid phase, e.g., 'water' (for oil-water (water-wet)) or 'oil' (for gas-oil (oil-wet))."""
    non_wetting_phase: FluidPhase
    """The non-wetting fluid phase, e.g., 'oil' (for oil-water (water-wet)) or 'gas' (for gas-oil (oil-wet))."""
    wetting_phase_saturation: npt.NDArray[np.floating] = attrs.field(
        converter=np.asarray
    )
    """The saturation values for the wetting phase, ranging from 0 to 1."""
    wetting_phase_relative_permeability: npt.NDArray[np.floating] = attrs.field(
        converter=np.asarray
    )
    """Relative permeability values for wetting phase corresponding to the saturation values."""
    non_wetting_phase_relative_permeability: npt.NDArray[np.floating] = attrs.field(
        converter=np.asarray
    )
    """Relative permeability values for non-wetting phase corresponding to the saturation values."""

    def __attrs_post_init__(self) -> None:
        """Validate table data."""
        if len(self.wetting_phase_saturation) != len(
            self.wetting_phase_relative_permeability
        ):
            raise ValidationError(
                f"Saturation and wetting phase kr arrays must have same length. "
                f"Got {len(self.wetting_phase_saturation)} vs {len(self.wetting_phase_relative_permeability)}"
            )
        if len(self.wetting_phase_saturation) != len(
            self.non_wetting_phase_relative_permeability
        ):
            raise ValidationError(
                f"Saturation and non-wetting phase kr arrays must have same length. "
                f"Got {len(self.wetting_phase_saturation)} vs {len(self.non_wetting_phase_relative_permeability)}"
            )
        if len(self.wetting_phase_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation")

        # Ensure arrays are sorted by saturation (required for np.interp)
        if not np.all(np.diff(self.wetting_phase_saturation) >= 0):
            raise ValidationError(
                "Wetting phase saturation must be monotonically increasing"
            )

    def get_wetting_phase_relative_permeability(
        self, wetting_phase_saturation: FloatOrArray
    ) -> FloatOrArray:
        """
        Get wetting phase relative permeability at given saturation(s).

        Uses `np.interp` for fast linear interpolation. Supports both scalar
        and array inputs up to 3D.

        :param wetting_phase_saturation: Saturation of the wetting phase (scalar or array).
        :return: Relative permeability value(s) - type matches input type.
        """
        # Handle scalar and array inputs
        saturation = np.atleast_1d(wetting_phase_saturation)
        original_shape = saturation.shape
        saturation_flat = saturation.ravel()

        # Fast linear interpolation using np.interp
        kr_flat = np.interp(
            x=saturation_flat,
            xp=self.wetting_phase_saturation,  # type: ignore[arg-type]
            fp=self.wetting_phase_relative_permeability,  # type: ignore[arg-type]
            left=self.wetting_phase_relative_permeability[0],
            right=self.wetting_phase_relative_permeability[-1],
        )
        # Reshape back to original dimensions
        return kr_flat.reshape(original_shape)

    def get_non_wetting_phase_relative_permeability(
        self, wetting_phase_saturation: FloatOrArray
    ) -> FloatOrArray:
        """
        Get non-wetting phase relative permeability at given wetting phase saturation(s).

        Uses `np.interp` for fast linear interpolation. Supports both scalar
        and array inputs up to 3D.

        :param wetting_phase_saturation: Saturation of the wetting phase (scalar or array).
        :return: Relative permeability value(s) - type matches input type.
        """
        # Handle scalar and array inputs
        saturation = np.atleast_1d(wetting_phase_saturation)
        original_shape = saturation.shape
        saturation_flat = saturation.ravel()

        # Fast linear interpolation using np.interp
        kr_flat = np.interp(
            x=saturation_flat,
            xp=self.wetting_phase_saturation,  # type: ignore[arg-type]
            fp=self.non_wetting_phase_relative_permeability,  # type: ignore[arg-type]
            left=self.non_wetting_phase_relative_permeability[0],
            right=self.non_wetting_phase_relative_permeability[-1],
        )
        # Reshape back to original dimensions
        return kr_flat.reshape(original_shape)

    def get_relative_permeabilities(
        self, wetting_phase_saturation: FloatOrArray
    ) -> typing.Tuple[FloatOrArray, FloatOrArray]:
        """
        Get both wetting and non-wetting phase relative permeabilities.

        :param wetting_phase_saturation: Saturation of the wetting phase (scalar or array).
        :return: Tuple of (wetting_kr, non_wetting_kr) - types match input type.
        """
        kr_wetting = self.get_wetting_phase_relative_permeability(
            wetting_phase_saturation
        )
        kr_non_wetting = self.get_non_wetting_phase_relative_permeability(
            wetting_phase_saturation
        )
        return kr_wetting, kr_non_wetting


@attrs.frozen
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
    supports_arrays: bool = True
    """Flag indicating support for array inputs."""

    def __attrs_post_init__(self) -> None:
        """Validate that the tables are set up correctly for three-phase flow."""
        if {
            self.oil_water_table.wetting_phase,
            self.oil_water_table.non_wetting_phase,
        } != {FluidPhase.WATER, FluidPhase.OIL}:
            raise ValidationError(
                "`oil_water_table` must be between water and oil phases."
            )
        if {self.gas_oil_table.wetting_phase, self.gas_oil_table.non_wetting_phase} != {
            FluidPhase.OIL,
            FluidPhase.GAS,
        }:
            raise ValidationError("`gas_oil_table` must be between oil and gas phases.")

        if self.oil_water_table.wetting_phase == self.gas_oil_table.non_wetting_phase:
            raise ValidationError(
                "Wetting phase of `oil_water_table` cannot be the same as non-wetting phase of `gas_oil_table`."
            )
        if self.gas_oil_table.wetting_phase != FluidPhase.OIL:
            raise ValidationError(
                "`gas_oil_table` wetting phase must be oil in three-phase system."
            )

    def get_relative_permeabilities(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for oil, water, gas.
        Uses two-phase tables + mixing rule for oil in 3-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :return: A dictionary with relative permeabilities for water, oil, and gas.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        # Convert to arrays for vectorized operations
        sw = np.atleast_1d(water_saturation)
        so = np.atleast_1d(oil_saturation)
        sg = np.atleast_1d(gas_saturation)

        # Broadcast all arrays to same shape
        sw, so, sg = np.broadcast_arrays(sw, so, sg)

        # Validate saturations
        if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
            raise ValidationError("Saturations must be between 0 and 1.")

        # Normalize saturations if they do not sum to 1
        total_saturation = sw + so + sg
        needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
        if np.any(needs_norm):
            sw = np.where(needs_norm, sw / total_saturation, sw)
            so = np.where(needs_norm, so / total_saturation, so)
            sg = np.where(needs_norm, sg / total_saturation, sg)

        # For oil-water table
        # krw = wetting phase kr at wetting phase saturation
        # kro_w = non-wetting phase kr at wetting phase saturation
        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            krw = self.oil_water_table.get_wetting_phase_relative_permeability(sw)
            kro_w = self.oil_water_table.get_non_wetting_phase_relative_permeability(sw)
        else:
            # Oil is wetting phase in oil-water table
            kro_w = self.oil_water_table.get_wetting_phase_relative_permeability(so)
            krw = self.oil_water_table.get_non_wetting_phase_relative_permeability(so)

        # For gas-oil table: oil is wetting phase
        # kro_g = wetting phase kr at oil saturation
        # krg = non-wetting phase kr at oil saturation
        kro_g = self.gas_oil_table.get_wetting_phase_relative_permeability(so)
        krg = self.gas_oil_table.get_non_wetting_phase_relative_permeability(so)

        # Apply mixing rule for three-phase oil relative permeability
        if self.mixing_rule is not None:
            kro = self.mixing_rule(
                kro_w=kro_w,
                kro_g=kro_g,
                water_saturation=sw,
                oil_saturation=so,
                gas_saturation=sg,
            )
        else:
            # Simple conservative rule if no mixing rule supplied
            kro = np.minimum(kro_w, kro_g)

        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def __call__(
        self,
        *,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for oil, water, gas.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
        :return: A dictionary with relative permeabilities for water, oil, and gas.
        0 <= water_saturation, oil_saturation, gas_saturation <= 1
        """
        return self.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )


def compute_corey_three_phase_relative_permeabilities(
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    water_exponent: float,
    oil_exponent: float,
    gas_exponent: float,
    wettability: WettabilityType = WettabilityType.WATER_WET,
    mixing_rule: MixingRule = stone_II_rule,
) -> typing.Tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
    """
    Computes relative permeability for water, oil, and gas in a three-phase system.
    Supports water-wet and oil-wet wettability assumptions.

    Uses Corey-type models for krw, krg, and Stone I rule for kro.

    Supports both scalar and array inputs for saturations.

    :param water_saturation: Current water saturation (fraction, between 0 and 1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, between 0 and 1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, between 0 and 1) - scalar or array.
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
    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)

    # Broadcast all arrays to same shape
    sw, so, sg = np.broadcast_arrays(sw, so, sg)

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValueError("Saturations must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    if wettability == WettabilityType.WATER_WET:
        # 1. Water relperm (wetting phase)
        movable_water_range = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water
        )
        effective_water_saturation = np.where(
            movable_water_range <= 1e-6,
            np.zeros_like(sw),
            np.clip((sw - irreducible_water_saturation) / movable_water_range, 0.0, 1.0)
        )
        krw = effective_water_saturation**water_exponent

        # 2. Gas relperm (nonwetting)
        movable_gas_range = 1.0 - residual_gas_saturation - residual_oil_saturation_gas
        effective_gas_saturation = np.where(
            movable_gas_range <= 1e-6,
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0)
        )
        krg = effective_gas_saturation**gas_exponent

        # 3. Oil relperm (intermediate phase) → Stone I blending
        kro = mixing_rule(
            kro_w=(1.0 - krw),
            kro_g=(1.0 - krg),
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )
        kro = kro**oil_exponent  # Apply oil Corey exponent as a curvature control

    elif wettability == WettabilityType.OIL_WET:
        # Oil is wetting, water becomes intermediate
        # 1. Oil relperm (wetting phase)
        movable_oil_range = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas
        )
        min_residual = np.minimum(
            residual_oil_saturation_water, residual_oil_saturation_gas
        )
        effective_oil_saturation = np.where(
            movable_oil_range <= 1e-6,
            np.zeros_like(so),
            np.clip((so - min_residual) / movable_oil_range, 0.0, 1.0)
        )
        kro = effective_oil_saturation**oil_exponent

        # 2. Gas relperm (nonwetting phase)
        movable_gas_range = 1.0 - residual_gas_saturation - irreducible_water_saturation
        effective_gas_saturation = np.where(
            movable_gas_range <= 1e-6,
            np.zeros_like(sg),
            np.clip((sg - residual_gas_saturation) / movable_gas_range, 0.0, 1.0)
        )
        krg = effective_gas_saturation**gas_exponent

        # 3. Water relperm (intermediate phase, use Stone I style blending)
        krw = mixing_rule(
            kro_w=(1.0 - kro),  # treat oil as wetting
            kro_g=(1.0 - krg),  # treat gas as nonwetting
            water_saturation=sw,
            oil_saturation=so,
            gas_saturation=sg,
        )
        krw = krw**water_exponent

    else:
        raise ValueError(f"Wettability {wettability} not implemented.")

    # Clip all results to [0, 1]
    krw = np.clip(krw, 0.0, 1.0)
    kro = np.clip(kro, 0.0, 1.0)
    krg = np.clip(krg, 0.0, 1.0)
    return krw, kro, krg  # type: ignore[return-value]


@attrs.frozen
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
    supports_arrays: bool = True
    """Flag indicating support for array inputs."""

    def get_relative_permeabilities(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
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
        params_missing = []
        if Swc is None:
            params_missing.append("Swc")
        if Sorw is None:
            params_missing.append("Sorw")
        if Sorg is None:
            params_missing.append("Sorg")
        if Srg is None:
            params_missing.append("Srg")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as arguments or set in the model instance. "
                f"Missing: {', '.join(params_missing)}"
            )

        krw, kro, krg = compute_corey_three_phase_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=Swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=Sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=Sorg,  # type: ignore[arg-type]
            residual_gas_saturation=Srg,  # type: ignore[arg-type]
            water_exponent=self.water_exponent,
            oil_exponent=self.oil_exponent,
            gas_exponent=self.gas_exponent,
            wettability=self.wettability,
            mixing_rule=self.mixing_rule,
        )
        return RelativePermeabilities(water=krw, oil=kro, gas=krg)  # type: ignore[typeddict-item]

    def __call__(
        self,
        *,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Compute relative permeabilities for water, oil, and gas.

        Supports both scalar and array inputs for saturations.

        :param water_saturation: Water saturation (fraction) - scalar or array.
        :param oil_saturation: Oil saturation (fraction) - scalar or array.
        :param gas_saturation: Gas saturation (fraction) - scalar or array.
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
