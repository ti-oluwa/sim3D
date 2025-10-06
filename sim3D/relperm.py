from functools import cached_property
import typing

from attrs import define
from scipy.interpolate import interp1d

from sim3D.properties import clip_scalar
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
    "compute_corey_three_phase_relative_permeabilities",
    "BrooksCoreyThreePhaseRelPermModel",
]


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


@define(slots=True, frozen=True)
class TwoPhaseRelPermTable:
    """
    Two-phase relative permeability lookup table.

    Interpolates relative permeabilities for two fluid phases based on saturation values.
    """

    phase1: FluidPhase
    """The first fluid phase, e.g., 'oil' or 'water'."""
    phase2: FluidPhase
    """The second fluid phase, e.g., 'water' or 'gas'."""
    saturation: ArrayLike[float]
    """The saturation values for phase2, ranging from 0 to 1."""
    phase1_relative_permeability: ArrayLike[float]
    """Relative permeability values for phase1 corresponding to the saturation values."""
    phase2_relative_permeability: ArrayLike[float]
    """Relative permeability values for phase2 corresponding to the saturation values."""

    @cached_property
    def phase1_interpolator(self) -> Interpolator:
        """Return the interpolator for phase1 relative permeability."""
        return interp1d(
            self.saturation,
            self.phase1_relative_permeability,
            bounds_error=False,
            fill_value=(
                self.phase1_relative_permeability[0],  # type: ignore
                self.phase1_relative_permeability[-1],
            ),
        )

    @cached_property
    def phase2_interpolator(self) -> Interpolator:
        """Return the interpolator for phase2 relative permeability."""
        return interp1d(
            self.saturation,
            self.phase2_relative_permeability,
            bounds_error=False,
            fill_value=(
                self.phase2_relative_permeability[0],  # type: ignore
                self.phase2_relative_permeability[-1],
            ),
        )

    def get_interpolators(self) -> typing.Tuple[Interpolator, Interpolator]:
        """Return interpolators for both phases."""
        return self.phase1_interpolator, self.phase2_interpolator


@define(slots=True, frozen=True)
class ThreePhaseRelPermTable:
    """
    Three-phase relative permeability lookup table, with mixing rules.

    Interpolates relative permeabilities for water, oil, and gas based on saturation values.

    This is the most common approach to handle three-phase relative permeabilities.
    Uses two two-phase tables (oil-water and oil-gas) and a mixing rule for oil in three-phase system.

    The values for the two-phase tables should be gotten from PVT experiments or literature.

    Supported mixing rules: min_rule, stone_I_rule, stone_II_rule. Additional rules can be defined as needed.
    """

    oil_water_table: TwoPhaseRelPermTable
    """Relative permeability table for oil vs water."""
    oil_gas_table: TwoPhaseRelPermTable
    """Relative permeability table for oil vs gas."""
    mixing_rule: typing.Optional[MixingRule] = stone_I_rule
    """
    Mixing rule function to compute oil relative permeability in three-phase system.

    The function should take the following parameters in order:
    - kro_w: Oil relative permeability from oil-water table
    - kro_g: Oil relative permeability from oil-gas table
    - Sw: Water saturation
    - So: Oil saturation
    - Sg: Gas saturation
    and return the mixed oil relative permeability.
    If None, a simple conservative rule (min(kro_w, kro_g)) is used.
    """

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
        water_relperm_interpolator = self.oil_water_table.phase2_interpolator
        gas_relperm_interpolator = self.oil_gas_table.phase2_interpolator
        oil_relperm_to_water_interpolator = self.oil_water_table.phase1_interpolator
        oil_relperm_to_gas_interpolator = self.oil_gas_table.phase1_interpolator

        krw = float(water_relperm_interpolator(water_saturation))
        krg = float(gas_relperm_interpolator(gas_saturation))
        kro_w = float(oil_relperm_to_water_interpolator(water_saturation))
        kro_g = float(oil_relperm_to_gas_interpolator(gas_saturation))

        if self.mixing_rule is not None:
            kro = self.mixing_rule(
                kro_w=kro_w,
                kro_g=kro_g,
                water_saturation=water_saturation,
                oil_saturation=oil_saturation,
                gas_saturation=gas_saturation,
            )
        else:
            # simple conservative rule if no mixing rule supplied
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
    mixing_rule: MixingRule = stone_I_rule,
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
    water_sat = clip_scalar(water_saturation, 0.0, 1.0)
    oil_sat = clip_scalar(oil_saturation, 0.0, 1.0)
    gas_sat = clip_scalar(gas_saturation, 0.0, 1.0)

    total_saturation = water_sat + oil_sat + gas_sat
    if abs(total_saturation - 1.0) > 1e-6 and total_saturation > 0.0:
        water_sat /= total_saturation
        oil_sat /= total_saturation
        gas_sat /= total_saturation

    if wettability == WettabilityType.WATER_WET:
        # 1. Water relperm (wetting phase)
        movable_water_range = (
            1.0 - irreducible_water_saturation - residual_oil_saturation_water
        )
        if movable_water_range <= 1e-6:
            effective_water_sat = 0.0
        else:
            effective_water_sat = (
                water_sat - irreducible_water_saturation
            ) / movable_water_range
            effective_water_sat = clip_scalar(effective_water_sat, 0.0, 1.0)
        krw = effective_water_sat**water_exponent

        # 2. Gas relperm (nonwetting)
        movable_gas_range = 1.0 - residual_gas_saturation - residual_oil_saturation_gas
        if movable_gas_range <= 1e-6:
            effective_gas_sat = 0.0
        else:
            effective_gas_sat = (gas_sat - residual_gas_saturation) / movable_gas_range
            effective_gas_sat = clip_scalar(effective_gas_sat, 0.0, 1.0)
        krg = effective_gas_sat**gas_exponent

        # 3. Oil relperm (intermediate phase) → Stone I blending
        kro = mixing_rule(
            kro_w=(1.0 - krw),
            kro_g=(1.0 - krg),
            water_saturation=water_sat,
            oil_saturation=oil_sat,
            gas_saturation=gas_sat,
        )
        kro = kro**oil_exponent  # apply oil Corey exponent as a curvature control

    elif wettability == WettabilityType.OIL_WET:
        # Oil is wetting, water becomes intermediate
        # 1. Oil relperm (wetting phase)
        movable_oil_range = (
            1.0 - residual_oil_saturation_water - residual_oil_saturation_gas
        )
        if movable_oil_range <= 1e-6:
            effective_oil_sat = 0.0
        else:
            effective_oil_sat = (
                oil_sat
                - min(residual_oil_saturation_water, residual_oil_saturation_gas)
            ) / movable_oil_range
            effective_oil_sat = clip_scalar(effective_oil_sat, 0.0, 1.0)
        kro = effective_oil_sat**oil_exponent

        # 2. Gas relperm (nonwetting phase)
        movable_gas_range = 1.0 - residual_gas_saturation - irreducible_water_saturation
        if movable_gas_range <= 1e-6:
            effective_gas_sat = 0.0
        else:
            effective_gas_sat = (gas_sat - residual_gas_saturation) / movable_gas_range
            effective_gas_sat = clip_scalar(effective_gas_sat, 0.0, 1.0)
        krg = effective_gas_sat**gas_exponent

        # 3. Water relperm (intermediate phase, use Stone I style blending)
        krw = mixing_rule(
            kro_w=(1.0 - kro),  # treat oil as wetting
            kro_g=(1.0 - krg),  # treat gas as nonwetting
            water_saturation=water_sat,
            oil_saturation=oil_sat,
            gas_saturation=gas_sat,
        )
        krw = krw**water_exponent

    else:
        raise ValueError(f"Wettability {wettability} not implemented.")

    # Clip all results to [0, 1]
    krw = clip_scalar(krw, 0.0, 1.0)
    kro = clip_scalar(kro, 0.0, 1.0)
    krg = clip_scalar(krg, 0.0, 1.0)
    return float(krw), float(kro), float(krg)


@define(slots=True, frozen=True)
class BrooksCoreyThreePhaseRelPermModel:
    """
    Brooks-Corey-type three-phase relative permeability model.

    Uses the Brooks-Corey model for two-phase relative permeabilities
    and Stone I mixing rule for oil in three-phase system.

    Supports water-wet and oil-wet wettability assumptions.
    """

    irreducible_water_saturation: float
    """(Default) Irreducible water saturation (Swc)."""
    residual_oil_saturation_water: float
    """(Default) Residual oil saturation after water flood (Sorw)."""
    residual_oil_saturation_gas: float
    """(Default) Residual oil saturation after gas flood (Sorg)."""
    residual_gas_saturation: float
    """(Default) Residual gas saturation (Sgr)."""
    water_exponent: float = 2.0
    """Corey exponent for water relative permeability."""
    oil_exponent: float = 2.0
    """Corey exponent for oil relative permeability (affects Stone I blending)."""
    gas_exponent: float = 2.0
    """Corey exponent for gas relative permeability."""
    wettability: WettabilityType = WettabilityType.WATER_WET
    """Wettability type (water-wet or oil-wet)."""
    mixing_rule: MixingRule = stone_I_rule
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
        krw, kro, krg = compute_corey_three_phase_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=(
                irreducible_water_saturation
                if irreducible_water_saturation is not None
                else self.irreducible_water_saturation
            ),
            residual_oil_saturation_water=(
                residual_oil_saturation_water
                if residual_oil_saturation_water is not None
                else self.residual_oil_saturation_water
            ),
            residual_oil_saturation_gas=(
                residual_oil_saturation_gas
                if residual_oil_saturation_gas is not None
                else self.residual_oil_saturation_gas
            ),
            residual_gas_saturation=(
                residual_gas_saturation
                if residual_gas_saturation is not None
                else self.residual_gas_saturation
            ),
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
