"""Capillary pressure models and tables for multiphase flow simulations."""

from functools import cached_property
import typing

from attrs import define
import numpy as np
from scipy.interpolate import interp1d

from sim3D.properties import clip_scalar
from sim3D.types import (
    ArrayLike,
    CapillaryPressures,
    FluidPhase,
    Interpolator,
    WettabilityType,
)


__all__ = [
    "TwoPhaseCapillaryPressureTable",
    "ThreePhaseCapillaryPressureTable",
    "BrooksCoreyCapillaryPressureModel",
    "VanGenuchtenCapillaryPressureModel",
    "LeverettJCapillaryPressureModel",
    "compute_brooks_corey_capillary_pressures",
    "compute_van_genuchten_capillary_pressures",
    "compute_leverett_j_capillary_pressures",
]


@define(slots=True, frozen=True)
class TwoPhaseCapillaryPressureTable:
    """
    Two-phase capillary pressure lookup table.

    Interpolates capillary pressure for two fluid phases based on saturation values.
    """

    wetting_phase: FluidPhase
    """The first fluid phase (typically the wetting phase), e.g., 'water' or 'oil'."""
    non_wetting_phase: FluidPhase
    """The second fluid phase (typically the non-wetting phase), e.g., 'oil' or 'gas'."""
    wetting_phase_saturation: ArrayLike[float]
    """The saturation values for the wetting phase (phase1), ranging from 0 to 1."""
    capillary_pressure: ArrayLike[float]
    """Capillary pressure values (Pc = P_non-wetting - P_wetting) corresponding to saturations."""

    @cached_property
    def interpolator(self) -> Interpolator:
        """Return the interpolator for capillary pressure."""
        return interp1d(
            self.wetting_phase_saturation,
            self.capillary_pressure,
            bounds_error=False,
            fill_value=(
                self.capillary_pressure[0],  # type: ignore
                self.capillary_pressure[-1],
            ),
        )

    def get_capillary_pressure(self, wetting_phase_saturation: float) -> float:
        """
        Get capillary pressure at given wetting phase saturation.

        :param wetting_phase_saturation: Saturation of the wetting phase.
        :return: Capillary pressure value.
        """
        return float(self.interpolator(wetting_phase_saturation))

    def __call__(self, wetting_phase_saturation: float, **kwargs: typing.Any) -> float:
        """
        Get capillary pressure at given wetting phase saturation.

        :param wetting_phase_saturation: Saturation of the wetting phase.
        :return: Capillary pressure value.
        """
        return self.get_capillary_pressure(wetting_phase_saturation)


@define(slots=True, frozen=True)
class ThreePhaseCapillaryPressureTable:
    """
    Three-phase capillary pressure lookup table.

    Uses two two-phase tables (oil-water and gas-oil) to compute capillary pressures
    in a three-phase system (water, oil, gas).

    Pcow = Po - Pw (oil-water capillary pressure)
    Pcgo = Pg - Po (gas-oil capillary pressure)
    """

    oil_water_table: TwoPhaseCapillaryPressureTable
    """
    Capillary pressure table for oil-water system (wetting phase = water or oil).

    A table of Pcow against wetting phase saturation (water saturation if water is wetting phase,
    oil saturation if oil is wetting phase).
    """
    gas_oil_table: TwoPhaseCapillaryPressureTable
    """
    Capillary pressure table for gas-oil system (wetting phase = oil).

    A table of Pcgo against oil saturation.
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

    def get_capillary_pressures(
        self, water_saturation: float, oil_saturation: float, gas_saturation: float
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for three-phase system.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        # Oil-water capillary pressure (based on wetting phase saturation)
        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            pcow = self.oil_water_table(water_saturation)
        else:
            pcow = self.oil_water_table(oil_saturation)

        # Gas-oil capillary pressure (based on oil saturation - wetting phase)
        pcgo = self.gas_oil_table(oil_saturation)
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)

    def __call__(
        self,
        *,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for three-phase system.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )


def compute_brooks_corey_capillary_pressures(
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    wettability: WettabilityType,
    oil_water_entry_pressure_water_wet: float,
    oil_water_entry_pressure_oil_wet: float,
    oil_water_pore_size_distribution_index_water_wet: float,
    oil_water_pore_size_distribution_index_oil_wet: float,
    gas_oil_entry_pressure: float,
    gas_oil_pore_size_distribution_index: float,
    mixed_wet_water_fraction: float = 0.5,
) -> typing.Tuple[float, float]:
    """
    Computes capillary pressures (Pcow, Pcgo) using Brooks-Corey model.

    Pcow is defined as Po - Pw.
    Pcgo is defined as Pg - Po.

    Brooks-Corey model: Pc = Pd * (Se)^(-1/λ)
    where:
    - Pd is the displacement/entry pressure
    - Se is the effective saturation
    - λ is the pore size distribution index

    Wettability behavior:
    - WATER_WET: Pcow > 0, Pcgo > 0 (water preferentially wets rock)
    - OIL_WET:   Pcow < 0, Pcgo > 0 (oil preferentially wets rock)
    - MIXED_WET: Pcow varies with saturation (weighted combination)

    :param water_saturation: Current water saturation (fraction, 0-1).
    :param oil_saturation: Current oil saturation (fraction, 0-1).
    :param gas_saturation: Current gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param wettability: Wettability type (WATER_WET, OIL_WET, or MIXED_WET).
    :param oil_water_entry_pressure_water_wet: Entry pressure for oil-water in water-wet system (psi).
    :param oil_water_entry_pressure_oil_wet: Entry pressure for oil-water in oil-wet system (psi).
    :param oil_water_pore_size_distribution_index_water_wet: Pore size distribution index (λ) for oil-water in water-wet.
    :param oil_water_pore_size_distribution_index_oil_wet: Pore size distribution index (λ) for oil-water in oil-wet.
    :param gas_oil_entry_pressure: Entry pressure for gas-oil (psi).
    :param gas_oil_pore_size_distribution_index: Pore size distribution index (λ) for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1, default 0.5).
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
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

    # Normalize saturations if they do not sum to 1
    total_saturation = water_saturation + oil_saturation + gas_saturation
    if abs(total_saturation - 1.0) > 1e-6 and total_saturation > 0.0:
        water_saturation /= total_saturation
        oil_saturation /= total_saturation
        gas_saturation /= total_saturation

    # Effective pore spaces
    total_mobile_pore_space_water = (
        1.0
        - irreducible_water_saturation
        - residual_oil_saturation_water
        - residual_gas_saturation
    )
    total_mobile_pore_space_gas = (
        1.0
        - irreducible_water_saturation
        - residual_oil_saturation_gas
        - residual_gas_saturation
    )

    # ---------------- Pcow (Po - Pw) ----------------
    oil_water_capillary_pressure = 0.0
    if total_mobile_pore_space_water > 1e-9:
        effective_water_saturation = (
            water_saturation - irreducible_water_saturation
        ) / total_mobile_pore_space_water
        effective_water_saturation = clip_scalar(effective_water_saturation, 1e-6, 1.0)

        if effective_water_saturation < 1.0 - 1e-6:
            if wettability == WettabilityType.WATER_WET:
                # Pure water-wet: Pcow > 0
                oil_water_capillary_pressure = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                ) ** (-1.0 / oil_water_pore_size_distribution_index_water_wet)

            elif wettability == WettabilityType.OIL_WET:
                # Pure oil-wet: Pcow < 0
                oil_water_capillary_pressure = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_oil_wet)
                )

            elif wettability == WettabilityType.MIXED_WET:
                # Mixed-wet: Weighted average
                pcow_water_wet = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                ) ** (-1.0 / oil_water_pore_size_distribution_index_water_wet)

                pcow_oil_wet = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_oil_wet)
                )

                oil_water_capillary_pressure = (
                    mixed_wet_water_fraction * pcow_water_wet
                    + (1.0 - mixed_wet_water_fraction) * pcow_oil_wet
                )

    # ---------------- Pcgo (Pg - Po) ----------------
    gas_oil_capillary_pressure = 0.0
    if total_mobile_pore_space_gas > 1e-9:
        effective_gas_saturation = (
            gas_saturation - residual_gas_saturation
        ) / total_mobile_pore_space_gas
        effective_gas_saturation = clip_scalar(effective_gas_saturation, 1e-6, 1.0)

        if effective_gas_saturation < 1.0 - 1e-6:
            gas_oil_capillary_pressure = gas_oil_entry_pressure * (
                effective_gas_saturation
            ) ** (-1.0 / gas_oil_pore_size_distribution_index)

    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@define(slots=True, frozen=True)
class BrooksCoreyCapillaryPressureModel:
    """
    Brooks-Corey capillary pressure model for three-phase systems.

    Implements the Brooks-Corey model: Pc = Pd * (Se)^(-1/λ)

    Supports water-wet, oil-wet, and mixed-wet systems.
    """

    irreducible_water_saturation: typing.Optional[float] = None
    """Default irreducible water saturation (Swc). Can be overridden per call."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """Default residual oil saturation after water flood (Sorw). Can be overridden per call."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg). Can be overridden per call."""
    residual_gas_saturation: typing.Optional[float] = None
    """Default residual gas saturation (Sgr). Can be overridden per call."""
    oil_water_entry_pressure_water_wet: float = 5.0
    """Entry pressure for oil-water in water-wet system (psi)."""
    oil_water_entry_pressure_oil_wet: float = 5.0
    """Entry pressure for oil-water in oil-wet system (psi)."""
    oil_water_pore_size_distribution_index_water_wet: float = 2.0
    """Pore size distribution index (λ) for oil-water in water-wet system."""
    oil_water_pore_size_distribution_index_oil_wet: float = 2.0
    """Pore size distribution index (λ) for oil-water in oil-wet system."""
    gas_oil_entry_pressure: float = 1.0
    """Entry pressure for gas-oil (psi)."""
    gas_oil_pore_size_distribution_index: float = 2.0
    """Pore size distribution index (λ) for gas-oil."""
    wettability: WettabilityType = WettabilityType.WATER_WET
    """Wettability type (WATER_WET, OIL_WET, or MIXED_WET)."""
    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    def get_capillary_pressures(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Brooks-Corey model.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :param irreducible_water_saturation: Optional override for Swc.
        :param residual_oil_saturation_water: Optional override for Sorw.
        :param residual_oil_saturation_gas: Optional override for Sorg.
        :param residual_gas_saturation: Optional override for Sgr.
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        # Use provided values or fall back to defaults
        swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        # Ensure all required parameters are available
        if swc is None or sorw is None or sorg is None or sgr is None:
            raise ValueError(
                "Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: Swc={swc}, Sorw={sorw}, Sorg={sorg}, Sgr={sgr}"
            )

        pcow, pcgo = compute_brooks_corey_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,
            residual_oil_saturation_water=sorw,
            residual_oil_saturation_gas=sorg,
            residual_gas_saturation=sgr,
            wettability=self.wettability,
            oil_water_entry_pressure_water_wet=self.oil_water_entry_pressure_water_wet,
            oil_water_entry_pressure_oil_wet=self.oil_water_entry_pressure_oil_wet,
            oil_water_pore_size_distribution_index_water_wet=self.oil_water_pore_size_distribution_index_water_wet,
            oil_water_pore_size_distribution_index_oil_wet=self.oil_water_pore_size_distribution_index_oil_wet,
            gas_oil_entry_pressure=self.gas_oil_entry_pressure,
            gas_oil_pore_size_distribution_index=self.gas_oil_pore_size_distribution_index,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)

    def __call__(
        self,
        *,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Brooks-Corey model.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )


def compute_van_genuchten_capillary_pressures(
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    wettability: WettabilityType,
    oil_water_alpha_water_wet: float,
    oil_water_alpha_oil_wet: float,
    oil_water_n_water_wet: float,
    oil_water_n_oil_wet: float,
    gas_oil_alpha: float,
    gas_oil_n: float,
    mixed_wet_water_fraction: float = 0.5,
) -> typing.Tuple[float, float]:
    """
    Computes capillary pressures using van Genuchten model.

    van Genuchten model: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)]
    where m = 1 - 1/n

    This model is widely used in unsaturated soil mechanics and petroleum engineering.
    Provides smoother transitions near residual saturations compared to Brooks-Corey.

    :param water_saturation: Current water saturation (fraction, 0-1).
    :param oil_saturation: Current oil saturation (fraction, 0-1).
    :param gas_saturation: Current gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param wettability: Wettability type (WATER_WET, OIL_WET, or MIXED_WET).
    :param oil_water_alpha_water_wet: van Genuchten α parameter for oil-water (water-wet) [1/psi].
    :param oil_water_alpha_oil_wet: van Genuchten α parameter for oil-water (oil-wet) [1/psi].
    :param oil_water_n_water_wet: van Genuchten n parameter for oil-water (water-wet).
    :param oil_water_n_oil_wet: van Genuchten n parameter for oil-water (oil-wet).
    :param gas_oil_alpha: van Genuchten α parameter for gas-oil [1/psi].
    :param gas_oil_n: van Genuchten n parameter for gas-oil.
    :param mixed_wet_water_fraction: Fraction of pore space that is water-wet (0-1, default 0.5).
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
    """
    if oil_water_alpha_water_wet <= 0.0 or oil_water_alpha_oil_wet <= 0.0:
        raise ValueError("Oil-water alpha parameters must be positive.")
    if gas_oil_alpha <= 0.0:
        raise ValueError("Gas-oil alpha parameter must be positive.")
    if oil_water_n_water_wet <= 1.0 or oil_water_n_oil_wet <= 1.0:
        raise ValueError("Oil-water n parameters must be greater than 1.")
    if gas_oil_n <= 1.0:
        raise ValueError("Gas-oil n parameter must be greater than 1.")

    if not (
        0 <= water_saturation <= 1
        and 0 <= oil_saturation <= 1
        and 0 <= gas_saturation <= 1
    ):
        raise ValueError(
            "Saturations must be between 0 and 1. "
            f"Received: Sw={water_saturation}, So={oil_saturation}, Sg={gas_saturation}"
        )

    # Normalize saturations if they do not sum to 1
    total_saturation = water_saturation + oil_saturation + gas_saturation
    if abs(total_saturation - 1.0) > 1e-6 and total_saturation > 0.0:
        water_saturation /= total_saturation
        oil_saturation /= total_saturation
        gas_saturation /= total_saturation

    # Effective pore spaces
    total_mobile_pore_space_water = (
        1.0
        - irreducible_water_saturation
        - residual_oil_saturation_water
        - residual_gas_saturation
    )
    total_mobile_pore_space_gas = (
        1.0
        - irreducible_water_saturation
        - residual_oil_saturation_gas
        - residual_gas_saturation
    )

    # ---------------- Pcow (Po - Pw) ----------------
    oil_water_capillary_pressure = 0.0
    if total_mobile_pore_space_water > 1e-9:
        effective_water_saturation = (
            water_saturation - irreducible_water_saturation
        ) / total_mobile_pore_space_water
        effective_water_saturation = clip_scalar(
            effective_water_saturation, 1e-6, 1.0 - 1e-6
        )

        if wettability == WettabilityType.WATER_WET:
            m_ww = 1.0 - 1.0 / oil_water_n_water_wet
            term = (effective_water_saturation ** (-1.0 / m_ww) - 1.0) ** (
                1.0 / oil_water_n_water_wet
            )
            oil_water_capillary_pressure = (1.0 / oil_water_alpha_water_wet) * term

        elif wettability == WettabilityType.OIL_WET:
            m_ow = 1.0 - 1.0 / oil_water_n_oil_wet
            term = (effective_water_saturation ** (-1.0 / m_ow) - 1.0) ** (
                1.0 / oil_water_n_oil_wet
            )
            oil_water_capillary_pressure = -(1.0 / oil_water_alpha_oil_wet) * term

        elif wettability == WettabilityType.MIXED_WET:
            # Water-wet contribution
            m_ww = 1.0 - 1.0 / oil_water_n_water_wet
            term_ww = (effective_water_saturation ** (-1.0 / m_ww) - 1.0) ** (
                1.0 / oil_water_n_water_wet
            )
            pcow_water_wet = (1.0 / oil_water_alpha_water_wet) * term_ww

            # Oil-wet contribution
            m_ow = 1.0 - 1.0 / oil_water_n_oil_wet
            term_ow = (effective_water_saturation ** (-1.0 / m_ow) - 1.0) ** (
                1.0 / oil_water_n_oil_wet
            )
            pcow_oil_wet = -(1.0 / oil_water_alpha_oil_wet) * term_ow

            oil_water_capillary_pressure = (
                mixed_wet_water_fraction * pcow_water_wet
                + (1.0 - mixed_wet_water_fraction) * pcow_oil_wet
            )

    # ---------------- Pcgo (Pg - Po) ----------------
    gas_oil_capillary_pressure = 0.0
    if total_mobile_pore_space_gas > 1e-9:
        effective_gas_saturation = (
            gas_saturation - residual_gas_saturation
        ) / total_mobile_pore_space_gas
        effective_gas_saturation = clip_scalar(
            effective_gas_saturation, 1e-6, 1.0 - 1e-6
        )

        m_go = 1.0 - 1.0 / gas_oil_n
        term = (effective_gas_saturation ** (-1.0 / m_go) - 1.0) ** (1.0 / gas_oil_n)
        gas_oil_capillary_pressure = (1.0 / gas_oil_alpha) * term

    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@define(slots=True, frozen=True)
class VanGenuchtenCapillaryPressureModel:
    """
    van Genuchten capillary pressure model for three-phase systems.

    Implements: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)] where m = 1 - 1/n

    Provides smoother transitions than Brooks-Corey model.
    """

    irreducible_water_saturation: typing.Optional[float] = None
    """Default irreducible water saturation (Swc). Can be overridden per call."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """Default residual oil saturation after water flood (Sorw). Can be overridden per call."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg). Can be overridden per call."""
    residual_gas_saturation: typing.Optional[float] = None
    """Default residual gas saturation (Sgr). Can be overridden per call."""
    oil_water_alpha_water_wet: float = 0.01
    """van Genuchten α parameter for oil-water (water-wet) [1/psi]."""
    oil_water_alpha_oil_wet: float = 0.01
    """van Genuchten α parameter for oil-water (oil-wet) [1/psi]."""
    oil_water_n_water_wet: float = 2.0
    """van Genuchten n parameter for oil-water (water-wet)."""
    oil_water_n_oil_wet: float = 2.0
    """van Genuchten n parameter for oil-water (oil-wet)."""
    gas_oil_alpha: float = 0.01
    """van Genuchten α parameter for gas-oil [1/psi]."""
    gas_oil_n: float = 2.0
    """van Genuchten n parameter for gas-oil."""
    wettability: WettabilityType = WettabilityType.WATER_WET
    """Wettability type (WATER_WET, OIL_WET, or MIXED_WET)."""
    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet in mixed-wet systems (0-1)."""

    def get_capillary_pressures(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using van Genuchten model.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :param irreducible_water_saturation: Optional override for Swc.
        :param residual_oil_saturation_water: Optional override for Sorw.
        :param residual_oil_saturation_gas: Optional override for Sorg.
        :param residual_gas_saturation: Optional override for Sgr.
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        # Use provided values or fall back to defaults
        swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        # Ensure all required parameters are available
        if swc is None or sorw is None or sorg is None or sgr is None:
            raise ValueError(
                "Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: Swc={swc}, Sorw={sorw}, Sorg={sorg}, Sgr={sgr}"
            )

        pcow, pcgo = compute_van_genuchten_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,
            residual_oil_saturation_water=sorw,
            residual_oil_saturation_gas=sorg,
            residual_gas_saturation=sgr,
            wettability=self.wettability,
            oil_water_alpha_water_wet=self.oil_water_alpha_water_wet,
            oil_water_alpha_oil_wet=self.oil_water_alpha_oil_wet,
            oil_water_n_water_wet=self.oil_water_n_water_wet,
            oil_water_n_oil_wet=self.oil_water_n_oil_wet,
            gas_oil_alpha=self.gas_oil_alpha,
            gas_oil_n=self.gas_oil_n,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)

    def __call__(
        self,
        *,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using van Genuchten model.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )


def compute_leverett_j_capillary_pressures(
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation_water: float,
    residual_oil_saturation_gas: float,
    residual_gas_saturation: float,
    permeability: float,
    porosity: float,
    oil_water_interfacial_tension: float,
    gas_oil_interfacial_tension: float,
    contact_angle_oil_water: float = 0.0,
    contact_angle_gas_oil: float = 0.0,
    wettability: WettabilityType = WettabilityType.WATER_WET,
) -> typing.Tuple[float, float]:
    """
    Computes capillary pressures using Leverett J-function approach.

    The Leverett J-function is a dimensionless correlation that relates capillary pressure
    to rock properties (porosity, permeability) and fluid properties (IFT, contact angle).

    Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)

    where:
    - σ is interfacial tension
    - θ is contact angle
    - φ is porosity
    - k is permeability
    - J(Se) is the dimensionless Leverett J-function (typically fit to data)

    For this implementation, we use a simplified power-law form:
    J(Se) = a * Se^(-b)

    :param water_saturation: Current water saturation (fraction, 0-1).
    :param oil_saturation: Current oil saturation (fraction, 0-1).
    :param gas_saturation: Current gas saturation (fraction, 0-1).
    :param irreducible_water_saturation: Irreducible water saturation (Swc).
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw).
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg).
    :param residual_gas_saturation: Residual gas saturation (Sgr).
    :param permeability: Absolute permeability (mD).
    :param porosity: Porosity (fraction, 0-1).
    :param oil_water_interfacial_tension: Oil-water interfacial tension (dyne/cm).
    :param gas_oil_interfacial_tension: Gas-oil interfacial tension (dyne/cm).
    :param contact_angle_oil_water: Oil-water contact angle in degrees (0° = water-wet).
    :param contact_angle_gas_oil: Gas-oil contact angle in degrees (0° = oil-wet).
    :param wettability: Wettability type (affects sign of capillary pressure).
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
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
    if permeability < 0.0:
        raise ValueError("Permeability must be positive.")
    if not (0.0 <= porosity <= 1.0):
        raise ValueError("Porosity must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = water_saturation + oil_saturation + gas_saturation
    if abs(total_saturation - 1.0) > 1e-6 and total_saturation > 0.0:
        water_saturation /= total_saturation
        oil_saturation /= total_saturation
        gas_saturation /= total_saturation

    # Effective pore spaces
    total_mobile_pore_space_water = (
        1.0
        - irreducible_water_saturation
        - residual_oil_saturation_water
        - residual_gas_saturation
    )
    total_mobile_pore_space_gas = (
        1.0
        - irreducible_water_saturation
        - residual_oil_saturation_gas
        - residual_gas_saturation
    )

    # Leverett J-function parameters (can be tuned to match experimental data)
    j_coeff_a = 0.5  # Empirical coefficient
    j_coeff_b = 0.5  # Empirical exponent

    # Convert contact angles to radians
    theta_ow_rad = np.deg2rad(contact_angle_oil_water)
    theta_go_rad = np.deg2rad(contact_angle_gas_oil)

    # Leverett scaling factor: sqrt(φ/k)
    # k in mD, convert to consistent units (dimensionless scaling)
    if permeability <= 0 or porosity <= 0:
        return 0.0, 0.0

    leverett_factor_ow = np.sqrt(porosity / permeability)
    leverett_factor_go = np.sqrt(porosity / permeability)

    # ---------------- Pcow (Po - Pw) ----------------
    oil_water_capillary_pressure = 0.0
    if total_mobile_pore_space_water > 1e-9:
        effective_water_saturation = (
            water_saturation - irreducible_water_saturation
        ) / total_mobile_pore_space_water
        effective_water_saturation = clip_scalar(
            effective_water_saturation, 1e-6, 1.0 - 1e-6
        )

        # J-function value
        j_value_ow = j_coeff_a * (effective_water_saturation ** (-j_coeff_b))

        # Capillary pressure (converting dyne/cm to psi: 1 dyne/cm = 0.00145038 psi)
        # Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)
        pc_ow = (
            oil_water_interfacial_tension
            * 0.00145038  # Convert to psi
            * np.cos(theta_ow_rad)
            * leverett_factor_ow
            * j_value_ow
        )

        # Apply wettability sign convention
        if wettability == WettabilityType.WATER_WET:
            oil_water_capillary_pressure = pc_ow
        elif wettability == WettabilityType.OIL_WET:
            oil_water_capillary_pressure = -pc_ow
        else:  # MIXED_WET
            # For mixed wettability, interpolate between water-wet and oil-wet
            # This is a simplified approach
            oil_water_capillary_pressure = pc_ow * (2.0 * np.cos(theta_ow_rad))

    # ---------------- Pcgo (Pg - Po) ----------------
    gas_oil_capillary_pressure = 0.0
    if total_mobile_pore_space_gas > 1e-9:
        effective_gas_saturation = (
            gas_saturation - residual_gas_saturation
        ) / total_mobile_pore_space_gas
        effective_gas_saturation = clip_scalar(
            effective_gas_saturation, 1e-6, 1.0 - 1e-6
        )

        # J-function value
        j_value_go = j_coeff_a * (effective_gas_saturation ** (-j_coeff_b))

        # Capillary pressure
        gas_oil_capillary_pressure = (
            gas_oil_interfacial_tension
            * 0.00145038  # Convert to psi
            * np.cos(theta_go_rad)
            * leverett_factor_go
            * j_value_go
        )

    return float(oil_water_capillary_pressure), float(gas_oil_capillary_pressure)


@define(slots=True, frozen=True)
class LeverettJCapillaryPressureModel:
    """
    Leverett J-function capillary pressure model for three-phase systems.

    Uses dimensionless J-function correlation to relate capillary pressure
    to rock and fluid properties: Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)

    Useful when capillary pressure data needs to be scaled across different
    rock types or fluid systems.
    """

    irreducible_water_saturation: typing.Optional[float] = None
    """Default irreducible water saturation (Swc). Can be overridden per call."""
    residual_oil_saturation_water: typing.Optional[float] = None
    """Default residual oil saturation after water flood (Sorw). Can be overridden per call."""
    residual_oil_saturation_gas: typing.Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg). Can be overridden per call."""
    residual_gas_saturation: typing.Optional[float] = None
    """Default residual gas saturation (Sgr). Can be overridden per call."""
    permeability: float = 100.0
    """Absolute permeability (mD)."""
    porosity: float = 0.2
    """Porosity (fraction, 0-1)."""
    oil_water_interfacial_tension: float = 30.0
    """Oil-water interfacial tension (dyne/cm)."""
    gas_oil_interfacial_tension: float = 20.0
    """Gas-oil interfacial tension (dyne/cm)."""
    contact_angle_oil_water: float = 0.0
    """Oil-water contact angle in degrees (0° = water-wet, 180° = oil-wet)."""
    contact_angle_gas_oil: float = 0.0
    """Gas-oil contact angle in degrees (0° = oil-wet to gas)."""
    wettability: WettabilityType = WettabilityType.WATER_WET
    """Wettability type (affects sign convention)."""

    def get_capillary_pressures(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        irreducible_water_saturation: typing.Optional[float] = None,
        residual_oil_saturation_water: typing.Optional[float] = None,
        residual_oil_saturation_gas: typing.Optional[float] = None,
        residual_gas_saturation: typing.Optional[float] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Leverett J-function.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :param irreducible_water_saturation: Optional override for Swc.
        :param residual_oil_saturation_water: Optional override for Sorw.
        :param residual_oil_saturation_gas: Optional override for Sorg.
        :param residual_gas_saturation: Optional override for Sgr.
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        # Use provided values or fall back to defaults
        swc = (
            irreducible_water_saturation
            if irreducible_water_saturation is not None
            else self.irreducible_water_saturation
        )
        sorw = (
            residual_oil_saturation_water
            if residual_oil_saturation_water is not None
            else self.residual_oil_saturation_water
        )
        sorg = (
            residual_oil_saturation_gas
            if residual_oil_saturation_gas is not None
            else self.residual_oil_saturation_gas
        )
        sgr = (
            residual_gas_saturation
            if residual_gas_saturation is not None
            else self.residual_gas_saturation
        )

        # Ensure all required parameters are available
        if swc is None or sorw is None or sorg is None or sgr is None:
            raise ValueError(
                "Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: Swc={swc}, Sorw={sorw}, Sorg={sorg}, Sgr={sgr}"
            )

        pcow, pcgo = compute_leverett_j_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,
            residual_oil_saturation_water=sorw,
            residual_oil_saturation_gas=sorg,
            residual_gas_saturation=sgr,
            permeability=self.permeability,
            porosity=self.porosity,
            oil_water_interfacial_tension=self.oil_water_interfacial_tension,
            gas_oil_interfacial_tension=self.gas_oil_interfacial_tension,
            contact_angle_oil_water=self.contact_angle_oil_water,
            contact_angle_gas_oil=self.contact_angle_gas_oil,
            wettability=self.wettability,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)

    def __call__(
        self,
        *,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Leverett J-function.

        :param water_saturation: Water saturation (fraction, 0-1).
        :param oil_saturation: Oil saturation (fraction, 0-1).
        :param gas_saturation: Gas saturation (fraction, 0-1).
        :kwarg irreducible_water_saturation: Optional override for Swc.
        :kwarg residual_oil_saturation_water: Optional override for Sorw.
        :kwarg residual_oil_saturation_gas: Optional override for Sorg.
        :kwarg residual_gas_saturation: Optional override for Sgr.
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=kwargs.get("irreducible_water_saturation"),
            residual_oil_saturation_water=kwargs.get("residual_oil_saturation_water"),
            residual_oil_saturation_gas=kwargs.get("residual_oil_saturation_gas"),
            residual_gas_saturation=kwargs.get("residual_gas_saturation"),
        )
