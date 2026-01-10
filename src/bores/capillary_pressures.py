"""Capillary pressure models and tables for multiphase flow simulations."""

import typing

import attrs
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.types import CapillaryPressures, FloatOrArray, FluidPhase, WettabilityType


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


@attrs.frozen
class TwoPhaseCapillaryPressureTable:
    """
    Two-phase capillary pressure lookup table.

    Interpolates capillary pressure for two fluid phases based on saturation values.
    Uses `np.interp` for fast vectorized interpolation.

    Supports both scalar and array inputs up to 3D.
    """

    wetting_phase: FluidPhase
    """The first fluid phase (typically the wetting phase), e.g., 'water' or 'oil'."""
    non_wetting_phase: FluidPhase
    """The second fluid phase (typically the non-wetting phase), e.g., 'oil' or 'gas'."""
    wetting_phase_saturation: npt.NDArray[np.floating] = attrs.field(
        converter=np.asarray
    )
    """The saturation values for the wetting phase (phase1), ranging from 0 to 1."""
    capillary_pressure: npt.NDArray[np.floating] = attrs.field(converter=np.asarray)
    """Capillary pressure values (Pc = P_non-wetting - P_wetting) corresponding to saturations."""

    def __attrs_post_init__(self) -> None:
        """Validate table data."""
        if len(self.wetting_phase_saturation) != len(self.capillary_pressure):
            raise ValidationError(
                f"Saturation and pressure arrays must have same length. "
                f"Got {len(self.wetting_phase_saturation)} vs {len(self.capillary_pressure)}"
            )
        if len(self.wetting_phase_saturation) < 2:
            raise ValidationError("At least 2 points required for interpolation")

        # Ensure arrays are sorted by saturation (required for np.interp)
        if not np.all(np.diff(self.wetting_phase_saturation) >= 0):
            raise ValidationError(
                "Wetting phase saturation must be monotonically increasing"
            )

    def get_capillary_pressure(
        self, wetting_phase_saturation: FloatOrArray
    ) -> FloatOrArray:
        """
        Get capillary pressure at given wetting phase saturation(s).

        Uses `np.interp` for fast linear interpolation. Supports both scalar
        and array inputs up to 3D. For out-of-bounds values, uses constant
        extrapolation (returns edge values).

        :param wetting_phase_saturation: Saturation of the wetting phase (scalar or array).
        :return: Capillary pressure value(s) - type matches input type.

        Examples:
        ```python
        # Scalar input
        pc = table.get_capillary_pressure(0.5)  # Returns float

        # Array input
        sw_grid = np.array([[0.2, 0.3], [0.4, 0.5]])
        pc_grid = table.get_capillary_pressure(sw_grid)  # Returns same shape array
        ```
        """
        # Handle scalar input
        is_scalar = np.isscalar(wetting_phase_saturation)
        saturation = np.atleast_1d(wetting_phase_saturation)

        # Store original shape for multi-dimensional arrays
        original_shape = saturation.shape

        # Flatten to 1D for `np.interp` (required)
        saturation_flat = saturation.ravel(order="C")

        # Fast linear interpolation using np.interp
        # left/right specify extrapolation values for out-of-bounds
        capillary_pressure_flat = np.interp(
            x=saturation_flat,
            xp=self.wetting_phase_saturation,  # type: ignore[arg-type]
            fp=self.capillary_pressure,  # type: ignore[arg-type]
            left=self.capillary_pressure[0],  # Extrapolate low saturations
            right=self.capillary_pressure[-1],  # Extrapolate high saturations
        )

        # Reshape back to original dimensions
        capillary_pressure = capillary_pressure_flat.reshape(original_shape)
        return capillary_pressure[0] if is_scalar else capillary_pressure

    def __call__(
        self, wetting_phase_saturation: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get capillary pressure at given wetting phase saturation(s).

        :param wetting_phase_saturation: Saturation of the wetting phase (scalar or array).
        :return: Capillary pressure value(s).
        """
        return self.get_capillary_pressure(wetting_phase_saturation)


@attrs.frozen
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

    def get_capillary_pressures(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for three-phase system.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: Dictionary with oil_water and gas_oil capillary pressures (matching input type).
        """
        # Oil-water capillary pressure (based on wetting phase saturation)
        if self.oil_water_table.wetting_phase == FluidPhase.WATER:
            pcow = self.oil_water_table(water_saturation)
        else:
            pcow = self.oil_water_table(oil_saturation)

        # Gas-oil capillary pressure (based on oil saturation - wetting phase)
        pcgo = self.gas_oil_table(oil_saturation)
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def __call__(
        self,
        *,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures for three-phase system.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :return: Dictionary with oil_water and gas_oil capillary pressures.
        """
        return self.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
        )


def compute_brooks_corey_capillary_pressures(
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: FloatOrArray,
    residual_oil_saturation_water: FloatOrArray,
    residual_oil_saturation_gas: FloatOrArray,
    residual_gas_saturation: FloatOrArray,
    wettability: WettabilityType,
    oil_water_entry_pressure_water_wet: float,
    oil_water_entry_pressure_oil_wet: float,
    oil_water_pore_size_distribution_index_water_wet: float,
    oil_water_pore_size_distribution_index_oil_wet: float,
    gas_oil_entry_pressure: float,
    gas_oil_pore_size_distribution_index: float,
    mixed_wet_water_fraction: float = 0.5,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Computes capillary pressures (Pcow, Pcgo) using Brooks-Corey model.

    Supports both scalar and array inputs (up to 3D).

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

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (Swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (Sgr) - scalar or array.
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
    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    swc = np.atleast_1d(irreducible_water_saturation)
    sorw = np.atleast_1d(residual_oil_saturation_water)
    sorg = np.atleast_1d(residual_oil_saturation_gas)
    sgr = np.atleast_1d(residual_gas_saturation)

    # Broadcast all arrays to same shape
    sw, so, sg, swc, sorw, sorg, sgr = np.broadcast_arrays(
        sw, so, sg, swc, sorw, sorg, sgr
    )

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

    # Effective pore spaces
    total_mobile_pore_space_water = 1.0 - swc - sorw - sgr
    total_mobile_pore_space_gas = 1.0 - swc - sorg - sgr

    # Pcow (Po - Pw)
    oil_water_capillary_pressure = np.zeros_like(sw)

    # Mask for valid mobile pore space
    valid_water = total_mobile_pore_space_water > 1e-9

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - swc) / total_mobile_pore_space_water, 0.0
        )
        effective_water_saturation = np.clip(effective_water_saturation, 1e-6, 1.0)

        # Mask for undersaturated conditions
        undersaturated = valid_water & (effective_water_saturation < 1.0 - 1e-6)

        if np.any(undersaturated):
            if wettability == WettabilityType.WATER_WET:
                # Pure water-wet: Pcow > 0
                pcow = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_water_wet)
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, 0.0)

            elif wettability == WettabilityType.OIL_WET:
                # Pure oil-wet: Pcow < 0
                pcow = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_oil_wet)
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, 0.0)

            elif wettability == WettabilityType.MIXED_WET:
                # Mixed-wet: Weighted average
                pcow_water_wet = oil_water_entry_pressure_water_wet * (
                    effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_water_wet)
                )
                pcow_oil_wet = -(
                    oil_water_entry_pressure_oil_wet
                    * effective_water_saturation
                    ** (-1.0 / oil_water_pore_size_distribution_index_oil_wet)
                )
                pcow = (
                    mixed_wet_water_fraction * pcow_water_wet
                    + (1.0 - mixed_wet_water_fraction) * pcow_oil_wet
                )
                oil_water_capillary_pressure = np.where(undersaturated, pcow, 0.0)

    # Pcgo (Pg - Po)
    gas_oil_capillary_pressure = np.zeros_like(sg)

    valid_gas = total_mobile_pore_space_gas > 1e-9

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - sgr) / total_mobile_pore_space_gas, 0.0
        )
        effective_gas_saturation = np.clip(effective_gas_saturation, 1e-6, 1.0)

        undersaturated_gas = valid_gas & (effective_gas_saturation < 1.0 - 1e-6)

        if np.any(undersaturated_gas):
            pcgo = gas_oil_entry_pressure * (
                effective_gas_saturation
                ** (-1.0 / gas_oil_pore_size_distribution_index)
            )
            gas_oil_capillary_pressure = np.where(undersaturated_gas, pcgo, 0.0)

    # Return scalars if inputs were scalars
    is_scalar = np.isscalar(water_saturation)
    if is_scalar:
        return float(oil_water_capillary_pressure), float(gas_oil_capillary_pressure)
    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@attrs.frozen
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

    supports_arrays: bool = True
    """Flag indicating support for array inputs."""

    def get_capillary_pressures(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Brooks-Corey model.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for Swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for Sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for Sorg - scalar or array.
        :param residual_gas_saturation: Optional override for Sgr - scalar or array.
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
        params_missing = []
        if swc is None:
            params_missing.append("Swc")
        if sorw is None:
            params_missing.append("Sorw")
        if sorg is None:
            params_missing.append("Sorg")
        if sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_brooks_corey_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            wettability=self.wettability,
            oil_water_entry_pressure_water_wet=self.oil_water_entry_pressure_water_wet,
            oil_water_entry_pressure_oil_wet=self.oil_water_entry_pressure_oil_wet,
            oil_water_pore_size_distribution_index_water_wet=self.oil_water_pore_size_distribution_index_water_wet,
            oil_water_pore_size_distribution_index_oil_wet=self.oil_water_pore_size_distribution_index_oil_wet,
            gas_oil_entry_pressure=self.gas_oil_entry_pressure,
            gas_oil_pore_size_distribution_index=self.gas_oil_pore_size_distribution_index,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def __call__(
        self,
        *,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Brooks-Corey model.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
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
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: FloatOrArray,
    residual_oil_saturation_water: FloatOrArray,
    residual_oil_saturation_gas: FloatOrArray,
    residual_gas_saturation: FloatOrArray,
    wettability: WettabilityType,
    oil_water_alpha_water_wet: float,
    oil_water_alpha_oil_wet: float,
    oil_water_n_water_wet: float,
    oil_water_n_oil_wet: float,
    gas_oil_alpha: float,
    gas_oil_n: float,
    mixed_wet_water_fraction: float = 0.5,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Computes capillary pressures using van Genuchten model.

    Supports both scalar and array inputs (up to 3D).

    van Genuchten model: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)]
    where m = 1 - 1/n

    This model is widely used in unsaturated soil mechanics and petroleum engineering.
    Provides smoother transitions near residual saturations compared to Brooks-Corey.

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (Swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (Sgr) - scalar or array.
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
    # Parameter validation
    if oil_water_alpha_water_wet <= 0.0 or oil_water_alpha_oil_wet <= 0.0:
        raise ValueError("Oil-water alpha parameters must be positive.")
    if gas_oil_alpha <= 0.0:
        raise ValueError("Gas-oil alpha parameter must be positive.")
    if oil_water_n_water_wet <= 1.0 or oil_water_n_oil_wet <= 1.0:
        raise ValueError("Oil-water n parameters must be greater than 1.")
    if gas_oil_n <= 1.0:
        raise ValueError("Gas-oil n parameter must be greater than 1.")

    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    swc = np.atleast_1d(irreducible_water_saturation)
    sorw = np.atleast_1d(residual_oil_saturation_water)
    sorg = np.atleast_1d(residual_oil_saturation_gas)
    sgr = np.atleast_1d(residual_gas_saturation)

    # Broadcast all arrays to same shape
    sw, so, sg, swc, sorw, sorg, sgr = np.broadcast_arrays(
        sw, so, sg, swc, sorw, sorg, sgr
    )

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

    # Effective pore spaces
    total_mobile_pore_space_water = 1.0 - swc - sorw - sgr
    total_mobile_pore_space_gas = 1.0 - swc - sorg - sgr

    #  Pcow (Po - Pw)
    oil_water_capillary_pressure = np.zeros_like(sw)

    # Mask for valid mobile pore space
    valid_water = total_mobile_pore_space_water > 1e-9

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - swc) / total_mobile_pore_space_water, 0.0
        )
        effective_water_saturation = np.clip(
            effective_water_saturation, 1e-6, 1.0 - 1e-6
        )

        if wettability == WettabilityType.WATER_WET:
            m_ww = 1.0 - 1.0 / oil_water_n_water_wet
            term = (effective_water_saturation ** (-1.0 / m_ww) - 1.0) ** (
                1.0 / oil_water_n_water_wet
            )
            pcow = (1.0 / oil_water_alpha_water_wet) * term
            oil_water_capillary_pressure = np.where(valid_water, pcow, 0.0)

        elif wettability == WettabilityType.OIL_WET:
            m_ow = 1.0 - 1.0 / oil_water_n_oil_wet
            term = (effective_water_saturation ** (-1.0 / m_ow) - 1.0) ** (
                1.0 / oil_water_n_oil_wet
            )
            pcow = -(1.0 / oil_water_alpha_oil_wet) * term
            oil_water_capillary_pressure = np.where(valid_water, pcow, 0.0)

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

            pcow = (
                mixed_wet_water_fraction * pcow_water_wet
                + (1.0 - mixed_wet_water_fraction) * pcow_oil_wet
            )
            oil_water_capillary_pressure = np.where(valid_water, pcow, 0.0)

    #  Pcgo (Pg - Po)
    gas_oil_capillary_pressure = np.zeros_like(sg)

    valid_gas = total_mobile_pore_space_gas > 1e-9

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - sgr) / total_mobile_pore_space_gas, 0.0
        )
        effective_gas_saturation = np.clip(effective_gas_saturation, 1e-6, 1.0 - 1e-6)

        m_go = 1.0 - 1.0 / gas_oil_n
        term = (effective_gas_saturation ** (-1.0 / m_go) - 1.0) ** (1.0 / gas_oil_n)
        pcgo = (1.0 / gas_oil_alpha) * term
        gas_oil_capillary_pressure = np.where(valid_gas, pcgo, 0.0)

    # Return scalars if inputs were scalars
    is_scalar = np.isscalar(water_saturation)
    if is_scalar:
        return float(oil_water_capillary_pressure), float(gas_oil_capillary_pressure)
    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@attrs.frozen
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
    supports_arrays: bool = True
    """Flag indicating support for array inputs."""

    def get_capillary_pressures(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using van Genuchten model.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for Swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for Sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for Sorg - scalar or array.
        :param residual_gas_saturation: Optional override for Sgr - scalar or array.
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
        params_missing = []
        if swc is None:
            params_missing.append("Swc")
        if sorw is None:
            params_missing.append("Sorw")
        if sorg is None:
            params_missing.append("Sorg")
        if sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_van_genuchten_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            wettability=self.wettability,
            oil_water_alpha_water_wet=self.oil_water_alpha_water_wet,
            oil_water_alpha_oil_wet=self.oil_water_alpha_oil_wet,
            oil_water_n_water_wet=self.oil_water_n_water_wet,
            oil_water_n_oil_wet=self.oil_water_n_oil_wet,
            gas_oil_alpha=self.gas_oil_alpha,
            gas_oil_n=self.gas_oil_n,
            mixed_wet_water_fraction=self.mixed_wet_water_fraction,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def __call__(
        self,
        *,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using van Genuchten model.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
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
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: FloatOrArray,
    residual_oil_saturation_water: FloatOrArray,
    residual_oil_saturation_gas: FloatOrArray,
    residual_gas_saturation: FloatOrArray,
    permeability: FloatOrArray,
    porosity: FloatOrArray,
    oil_water_interfacial_tension: float,
    gas_oil_interfacial_tension: float,
    contact_angle_oil_water: float = 0.0,
    contact_angle_gas_oil: float = 0.0,
    wettability: WettabilityType = WettabilityType.WATER_WET,
) -> typing.Tuple[FloatOrArray, FloatOrArray]:
    """
    Computes capillary pressures using Leverett J-function approach.

    Supports both scalar and array inputs (up to 3D).

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

    :param water_saturation: Current water saturation (fraction, 0-1) - scalar or array.
    :param oil_saturation: Current oil saturation (fraction, 0-1) - scalar or array.
    :param gas_saturation: Current gas saturation (fraction, 0-1) - scalar or array.
    :param irreducible_water_saturation: Irreducible water saturation (Swc) - scalar or array.
    :param residual_oil_saturation_water: Residual oil saturation during water flooding (Sorw) - scalar or array.
    :param residual_oil_saturation_gas: Residual oil saturation during gas flooding (Sorg) - scalar or array.
    :param residual_gas_saturation: Residual gas saturation (Sgr) - scalar or array.
    :param permeability: Absolute permeability (mD) - scalar or array.
    :param porosity: Porosity (fraction, 0-1) - scalar or array.
    :param oil_water_interfacial_tension: Oil-water interfacial tension (dyne/cm).
    :param gas_oil_interfacial_tension: Gas-oil interfacial tension (dyne/cm).
    :param contact_angle_oil_water: Oil-water contact angle in degrees (0° = water-wet).
    :param contact_angle_gas_oil: Gas-oil contact angle in degrees (0° = oil-wet).
    :param wettability: Wettability type (affects sign of capillary pressure).
    :return: Tuple of (oil_water_capillary_pressure, gas_oil_capillary_pressure) in psi.
    """
    # Convert to arrays for vectorized operations
    sw = np.atleast_1d(water_saturation)
    so = np.atleast_1d(oil_saturation)
    sg = np.atleast_1d(gas_saturation)
    swc = np.atleast_1d(irreducible_water_saturation)
    sorw = np.atleast_1d(residual_oil_saturation_water)
    sorg = np.atleast_1d(residual_oil_saturation_gas)
    sgr = np.atleast_1d(residual_gas_saturation)
    perm = np.atleast_1d(permeability)
    phi = np.atleast_1d(porosity)

    # Broadcast all arrays to same shape
    sw, so, sg, swc, sorw, sorg, sgr, perm, phi = np.broadcast_arrays(
        sw, so, sg, swc, sorw, sorg, sgr, perm, phi
    )

    # Validate saturations
    if np.any((sw < 0) | (sw > 1) | (so < 0) | (so > 1) | (sg < 0) | (sg > 1)):
        raise ValueError("Saturations must be between 0 and 1.")
    if np.any(perm < 0.0):
        raise ValueError("Permeability must be positive.")
    if np.any((phi < 0.0) | (phi > 1.0)):
        raise ValueError("Porosity must be between 0 and 1.")

    # Normalize saturations if they do not sum to 1
    total_saturation = sw + so + sg
    needs_norm = (np.abs(total_saturation - 1.0) > 1e-6) & (total_saturation > 0.0)
    if np.any(needs_norm):
        sw = np.where(needs_norm, sw / total_saturation, sw)
        so = np.where(needs_norm, so / total_saturation, so)
        sg = np.where(needs_norm, sg / total_saturation, sg)

    # Effective pore spaces
    total_mobile_pore_space_water = 1.0 - swc - sorw - sgr
    total_mobile_pore_space_gas = 1.0 - swc - sorg - sgr

    # Leverett J-function parameters (can be tuned to match experimental data)
    j_coeff_a = 0.5  # Empirical coefficient
    j_coeff_b = 0.5  # Empirical exponent

    # Convert contact angles to radians
    theta_ow_rad = np.deg2rad(contact_angle_oil_water)
    theta_go_rad = np.deg2rad(contact_angle_gas_oil)

    # Leverett scaling factor: sqrt(φ/k)
    # Check for zero/invalid values
    valid_rock = (perm > 0) & (phi > 0)
    leverett_factor = np.where(valid_rock, np.sqrt(phi / perm), 0.0)

    # ---------------- Pcow (Po - Pw) ----------------
    oil_water_capillary_pressure = np.zeros_like(sw)

    valid_water = (total_mobile_pore_space_water > 1e-9) & valid_rock

    if np.any(valid_water):
        effective_water_saturation = np.where(
            valid_water, (sw - swc) / total_mobile_pore_space_water, 0.0
        )
        effective_water_saturation = np.clip(
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
            * leverett_factor
            * j_value_ow
        )

        # Apply wettability sign convention
        if wettability == WettabilityType.WATER_WET:
            oil_water_capillary_pressure = np.where(valid_water, pc_ow, 0.0)
        elif wettability == WettabilityType.OIL_WET:
            oil_water_capillary_pressure = np.where(valid_water, -pc_ow, 0.0)
        else:  # MIXED_WET
            # For mixed wettability, interpolate based on contact angle
            mixed_pc_ow = pc_ow * (2.0 * np.cos(theta_ow_rad))
            oil_water_capillary_pressure = np.where(valid_water, mixed_pc_ow, 0.0)

    # ---------------- Pcgo (Pg - Po) ----------------
    gas_oil_capillary_pressure = np.zeros_like(sg)

    valid_gas = (total_mobile_pore_space_gas > 1e-9) & valid_rock

    if np.any(valid_gas):
        effective_gas_saturation = np.where(
            valid_gas, (sg - sgr) / total_mobile_pore_space_gas, 0.0
        )
        effective_gas_saturation = np.clip(effective_gas_saturation, 1e-6, 1.0 - 1e-6)

        # J-function value
        j_value_go = j_coeff_a * (effective_gas_saturation ** (-j_coeff_b))

        # Capillary pressure
        pcgo = (
            gas_oil_interfacial_tension
            * 0.00145038  # Convert to psi
            * np.cos(theta_go_rad)
            * leverett_factor
            * j_value_go
        )
        gas_oil_capillary_pressure = np.where(valid_gas, pcgo, 0.0)

    # Return scalars if inputs were scalars
    is_scalar = np.isscalar(water_saturation)
    if is_scalar:
        return float(oil_water_capillary_pressure), float(gas_oil_capillary_pressure)
    return oil_water_capillary_pressure, gas_oil_capillary_pressure


@attrs.frozen
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
    supports_arrays: bool = True
    """Flag indicating support for array inputs."""

    def get_capillary_pressures(
        self,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        irreducible_water_saturation: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_water: typing.Optional[FloatOrArray] = None,
        residual_oil_saturation_gas: typing.Optional[FloatOrArray] = None,
        residual_gas_saturation: typing.Optional[FloatOrArray] = None,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Leverett J-function.

        Supports both scalar and array inputs.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
        :param irreducible_water_saturation: Optional override for Swc - scalar or array.
        :param residual_oil_saturation_water: Optional override for Sorw - scalar or array.
        :param residual_oil_saturation_gas: Optional override for Sorg - scalar or array.
        :param residual_gas_saturation: Optional override for Sgr - scalar or array.
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
        params_missing = []
        if swc is None:
            params_missing.append("Swc")
        if sorw is None:
            params_missing.append("Sorw")
        if sorg is None:
            params_missing.append("Sorg")
        if sgr is None:
            params_missing.append("Sgr")
        if params_missing:
            raise ValidationError(
                f"Residual saturations must be provided either as model defaults or in the call. "
                f"Missing: {', '.join(params_missing)}"
            )

        pcow, pcgo = compute_leverett_j_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=swc,  # type: ignore[arg-type]
            residual_oil_saturation_water=sorw,  # type: ignore[arg-type]
            residual_oil_saturation_gas=sorg,  # type: ignore[arg-type]
            residual_gas_saturation=sgr,  # type: ignore[arg-type]
            permeability=self.permeability,
            porosity=self.porosity,
            oil_water_interfacial_tension=self.oil_water_interfacial_tension,
            gas_oil_interfacial_tension=self.gas_oil_interfacial_tension,
            contact_angle_oil_water=self.contact_angle_oil_water,
            contact_angle_gas_oil=self.contact_angle_gas_oil,
            wettability=self.wettability,
        )
        return CapillaryPressures(oil_water=pcow, gas_oil=pcgo)  # type: ignore[typeddict-item]

    def __call__(
        self,
        *,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Compute capillary pressures using Leverett J-function.

        :param water_saturation: Water saturation (fraction, 0-1) - scalar or array.
        :param oil_saturation: Oil saturation (fraction, 0-1) - scalar or array.
        :param gas_saturation: Gas saturation (fraction, 0-1) - scalar or array.
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
