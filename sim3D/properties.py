"""Utilities for computing reservoir rock and fluid properties."""

import functools
import typing
import warnings
import numpy as np
from scipy.optimize import brentq, root_scalar
from CoolProp.CoolProp import PropsSI

from sim3D.types import NDimension, NDimensionalGrid, FluidMiscibility
from sim3D.models import CapillaryPressureParameters, WettabilityType
from sim3D.constants import (
    FT3_TO_BBL,
    OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL,
    POUNDS_PER_FT3_TO_GRAMS_PER_CM3,
    SECONDS_TO_DAYS,
    STANDARD_TEMPERATURE_IMPERIAL,
    STANDARD_PRESSURE_IMPERIAL,
    BBL_TO_FT3,
    FT3_TO_STB,
    KG_PER_M3_TO_POUNDS_PER_FT3,
    MOLECULAR_WEIGHT_AIR,
    POUNDS_PER_FT3_TO_KG_PER_M3,
    SCF_PER_POUND_MOLE,
    STANDARD_WATER_DENSITY_IMPERIAL,
    STANDARD_TEMPERATURE,
    STANDARD_PRESSURE,
    STANDARD_WATER_DENSITY,
    M3_PER_M3_TO_SCF_PER_STB,
    PSI_TO_PA,
    PA_TO_PSI,
    PA_S_TO_CENTIPOISE,
    IDEAL_GAS_CONSTANT_IMPERIAL,
    GRAMS_PER_MOLE_TO_POUNDS_PER_MOLE,
    MIN_VALID_PRESSURE,
    MAX_VALID_PRESSURE,
    MIN_VALID_TEMPERATURE,
    MAX_VALID_TEMPERATURE,
    MOLECULAR_WEIGHT_CO2,
    MOLECULAR_WEIGHT_NACL,
    MOLECULAR_WEIGHT_METHANE,
    MOLECULAR_WEIGHT_N2,
)


def validate_input_temperature(temperature: typing.Union[float, np.ndarray]) -> None:
    """
    Validates that the input temperature(s) are within valid/reservoir-like range.

    Accepts scalar or ndarray input.

    :param temperature: Temperature(s) in Kelvin (°F)
    :raises ValueError: If any temperature is outside the valid range.
    """
    temp_array = np.asarray(temperature)
    invalid = (temp_array < MIN_VALID_TEMPERATURE) | (
        temp_array > MAX_VALID_TEMPERATURE
    )

    if np.any(invalid):
        raise ValueError(
            f"Temperature(s) out of valid range [{MIN_VALID_TEMPERATURE}, {MAX_VALID_TEMPERATURE}] K: "
            f"{temp_array[invalid]}"
        )


def validate_input_pressure(pressure: typing.Union[float, np.ndarray]) -> None:
    """
    Validates that the input pressure(s) are within valid/reservoir-like range.

    Accepts scalar or ndarray input.

    :param pressure: Pressure(s) in Pascals (psi)
    :raises ValueError: If any pressure is outside the valid range.
    """
    pressure_array = np.asarray(pressure)
    invalid = (pressure_array < MIN_VALID_PRESSURE) | (
        pressure_array > MAX_VALID_PRESSURE
    )

    if np.any(invalid):
        raise ValueError(
            f"Pressure(s) out of valid range [{MIN_VALID_PRESSURE}, {MAX_VALID_PRESSURE}] Pa: "
            f"{pressure_array[invalid]}"
        )


def is_CoolProp_supported_fluid(fluid: str) -> bool:
    """
    Check if the fluid is supported by CoolProp.

    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: True if the fluid is supported, False otherwise.
    """
    return PropsSI("D", "T", 300, "P", 101325, fluid) is not None


def clip_pressure(pressure: float, fluid: str) -> float:
    """
    Clips pressure to be within CoolProp's valid pressure range for the given fluid.

    :param pressure: Pressure in Pascals (psi)
    :param fluid: CoolProp fluid name
    :return: Clipped pressure in Pascals
    """
    p_min = PropsSI("P_MIN", fluid)  # Minimum pressure allowed
    p_max = PropsSI("P_MAX", fluid)  # Maximum pressure allowed
    return min(max(pressure, p_min + 1.0), p_max - 1.0)  # Add small buffer


def clip_temperature(temperature: float, fluid: str) -> float:
    """
    Clips temperature to be within CoolProp's valid temperature range for the given fluid.

    :param temperature: Temperature in Kelvin (°F)
    :param fluid: CoolProp fluid name
    :return: Clipped temperature in Kelvin
    """
    t_min = PropsSI("T_MIN", fluid)
    t_max = PropsSI("T_MAX", fluid)
    return min(max(temperature, t_min + 0.1), t_max - 0.1)  # Add small buffer


##################################################
# GENERIC FLUID PROPERTIES COMPUTATION FUNCTIONS #
##################################################


@functools.lru_cache(maxsize=128)
def compute_fluid_density(pressure: float, temperature: float, fluid: str) -> float:
    """
    Compute fluid density from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Density in lbm/ft³
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_in_pascals = pressure * PSI_TO_PA
    density = PropsSI(
        "D",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )
    return density * KG_PER_M3_TO_POUNDS_PER_FT3


@functools.lru_cache(maxsize=128)
def compute_fluid_viscosity(pressure: float, temperature: float, fluid: str) -> float:
    """
    Compute fluid dynamic viscosity from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Viscosity in centipoise (cP)
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_in_pascals = pressure * PSI_TO_PA
    viscosity = PropsSI(
        "V",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )
    return viscosity * PA_S_TO_CENTIPOISE


@functools.lru_cache(maxsize=128)
def compute_fluid_compressibility_factor(
    pressure: float, temperature: float, fluid: str
) -> float:
    """
    Compute fluid compressibility factor Z from EOS using CoolProp.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name (must be supported by CoolProp, e.g., "CO2", "Methane")
    :return: Compressibility factor Z (dimensionless)
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_in_pascals = pressure * PSI_TO_PA
    return PropsSI(
        "Z",
        "P",
        clip_pressure(pressure_in_pascals, fluid),
        "T",
        clip_temperature(temperature_in_kelvin, fluid),
        fluid,
    )


@functools.lru_cache(maxsize=128)
def compute_fluid_compressibility(
    pressure: float,
    temperature: float,
    fluid: str,
) -> float:
    """
    Computes the isothermal compressibility of a fluid at a given pressure and temperature.

    Compressibility is defined as:

        C_f = -(1/ρ) * (dρ/dP) at constant temperature

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param fluid: str name supported by CoolProp (e.g., 'n-Octane')
    :return: Compressibility in psi⁻¹
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_in_pascals = pressure * PSI_TO_PA
    return (
        PropsSI(
            "ISOTHERMAL_COMPRESSIBILITY",
            "P",
            clip_pressure(pressure_in_pascals, fluid),
            "T",
            clip_temperature(temperature_in_kelvin, fluid),
            fluid,
        )
        / PA_TO_PSI
    )


@functools.lru_cache(maxsize=128)
def compute_gas_gravity(gas: str) -> float:
    """
    Computes the specific gravity of a gas at a given pressure and temperature.

    Gas gravity is defined as the ratio of the density of the gas to the density of air at standard conditions.

    :param gas: gas name supported by CoolProp (e.g., 'Methane')
    :return: Gas gravity (dimensionless)
    """
    density = compute_fluid_density(
        STANDARD_PRESSURE_IMPERIAL, STANDARD_TEMPERATURE_IMPERIAL, gas
    )
    return density / PropsSI(
        "D", "T", STANDARD_TEMPERATURE, "P", STANDARD_PRESSURE, "Air"
    )


####################################################
# SPECIALIZED FLUID PROPERTY COMPUTATION FUNCTIONS #
####################################################


@functools.lru_cache(maxsize=128)
def compute_gas_gravity_from_density(
    pressure: float,
    temperature: float,
    density: float,
) -> float:
    """
    Computes the gas gravity from density.

    Gas gravity for this case, is derived as the ratio of the gas density to the
    air density at the same temperature and pressure.

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param density: Density of the gas in lbm/ft³
    :return: Gas gravity (dimensionless)
    """
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)
    pressure_in_pascals = pressure * PSI_TO_PA
    air_density = compute_fluid_density(
        pressure_in_pascals, temperature_in_kelvin, fluid="Air"
    )
    return density / (air_density * KG_PER_M3_TO_POUNDS_PER_FT3)


@functools.lru_cache(maxsize=128)
def mix_fluid_property(
    fluid1_saturation: float,
    fluid1_property: float,
    fluid2_saturation: float,
    fluid2_property: float,
    miscibility: FluidMiscibility = "logarithmic",
) -> float:
    """
    Mixes two fluid properties based on their saturations using a specified mixing method.

    The mixing methods available are:
        - Logarithmic: property_eff = property1^S1 * property2^(1 - S1)
        - Harmonic: 1/property_eff = (S1/property1) + ((1 - S1)/property2)
        - Linear: property_eff = S1 * property1 + (1 - S1) * property2

    :param fluid1_saturation: Saturation of the first fluid (fraction, 0 to 1)
    :param fluid1_property: Property of the first fluid (e.g., viscosity, compressibility)
    :param fluid2_saturation: Saturation of the second fluid (fraction, 0 to 1)
    :param fluid2_property: Property of the second fluid (e.g., viscosity, compressibility)
    :param miscibility: Method for mixing properties ('logarithmic', 'harmonic', 'linear')
    :return: Effective mixed property value
    """
    # Clip saturations to avoid numerical issues
    fluid1_saturation = np.clip(fluid1_saturation, 1e-8, 1 - 1e-8)
    fluid2_saturation = np.clip(fluid2_saturation, 1e-8, 1 - 1e-8)

    if miscibility == "logarithmic":
        return (fluid1_property**fluid1_saturation) * (
            fluid2_property**fluid2_saturation
        )

    elif miscibility == "harmonic":
        # Adjust minimum properties to avoid division by zero
        fluid1_property = max(fluid1_property, 1e-8)
        fluid2_property = max(fluid2_property, 1e-8)
        return 1 / (
            (fluid1_saturation / fluid1_property)
            + (fluid2_saturation / fluid2_property)
        )

    elif miscibility == "linear":
        return (fluid1_saturation * fluid1_property) + (
            fluid2_saturation * fluid2_property
        )
    raise ValueError("Unknown mixing method.")


@functools.lru_cache(maxsize=128)
def compute_total_fluid_compressibility(
    water_saturation: float,
    oil_saturation: float,
    water_compressibility: float,
    oil_compressibility: float,
    gas_saturation: typing.Optional[float] = None,
    gas_compressibility: typing.Optional[float] = None,
) -> float:
    """
    Calculates the total fluid compressibility as a saturation-weighted average of
    individual phase compressibilities.

    :param water_saturation: Water saturation (fraction).
    :param oil_saturation: Oil saturation (fraction).
    :param water_compressibility: Compressibility of the water phase (psi⁻¹).
    :param oil_compressibility: Compressibility of the oil phase (psi⁻¹).
    :param gas_saturation: Optional gas saturation (fraction) for three-phase systems.
    :param gas_compressibility: Optional gas compressibility (psi⁻¹) for three-phase systems.
    :return: Total fluid compressibility (psi⁻¹).
    """
    total_fluid_compressibility = (water_saturation * water_compressibility) + (
        oil_saturation * oil_compressibility
    )

    if gas_saturation is not None and gas_compressibility is not None:
        total_fluid_compressibility += gas_saturation * gas_compressibility

    return total_fluid_compressibility


@functools.lru_cache(maxsize=128)
def linear_decay_factor_to_exponential_decay_constant(
    linear_decay_factor: float,
) -> float:
    """
    Converts a linear decay factor (alpha) to an equivalent
    exponential decay constant (beta) such that:

        1 - alpha * S ≈ exp(-beta * S)

    :param linear_decay_factor: Linear decay coefficient (alpha), 0 < alpha < 1
    :return: Exponential decay constant (beta)
    """
    if not (0 < linear_decay_factor < 1):
        raise ValueError("Linear decay factor must be between 0 and 1.")
    return -np.log(1 - linear_decay_factor)


@functools.lru_cache(maxsize=128)
def compute_diffusion_number(
    porosity: float,
    total_mobility: float,
    total_compressibility: float,
    time_step_size: float,
    cell_size: float,
) -> float:
    """
    Computes the diffusion number (also known as the stability criterion or CFL number)
    for a single reservoir grid block in a pressure diffusion simulation.

    The diffusion number is a dimensionless quantity that determines whether an
    explicit finite-difference method will be stable for the given set of physical
    and numerical parameters. If the diffusion number is greater than or equal to 0.25,
    an implicit method is typically recommended.

    The formula used is:

        D = [k * sum(k_r / μ)] / [(φ * C_t)) * (Δt / Δh²)]

    where:
        - k is the permeability in mD (millidarcies),
        - φ is the porosity (fraction),
        - k_r is the relative permeability (dimensionless),
        - μ is the fluid viscosity (cP),
        - C_t is the total compressibility (psi⁻¹),
        - Δt is the time step size (s),
        - Δh is the grid block size (ft) in a specific direction (∆x, ∆y, ∆z).

    :param permeability: Rock permeability in millidarcies (mD)
    :param porosity: Rock porosity as a fraction (e.g., 0.2)
    :param total_mobility: Fluid total mobility (ft²/psi.day), which is typically calculated as [k * sum(k_r / μ)]
    :param total_compressibility: Total compressibility of the system (psi⁻¹)
    :param time_step_size: Time step size (seconds)
    :param cell_size: Size of the grid block (ft)
    :return: Diffusion number (dimensionless)
    """
    time_in_days = time_step_size * SECONDS_TO_DAYS
    diffusion_number = (total_mobility / (porosity * total_compressibility)) * (
        time_in_days / cell_size**2
    )
    return diffusion_number


@functools.lru_cache(maxsize=128)
def compute_harmonic_mean(value1: float, value2: float) -> float:
    """
    Computes the harmonic mean of two values.

    :param value1: First value (e.g., transmissibility, permeability, viscosity)
    :param value2: Second value (e.g., transmissibility, permeability, viscosity)
    :return: Harmonic mean scaled by the square of the spacing
    """
    if value1 + value2 == 0:
        return 0.0
    return (2 * value1 * value2) / (value1 + value2)


def compute_harmonic_mobility(
    index1: NDimension,
    index2: NDimension,
    mobility_grid: NDimensionalGrid[NDimension],
) -> float:
    """
    Computes harmonic average mobility between two cells.

    If both mobilities are zero, returns zero to avoid division by zero.

    :param index1: Index of the first cell in the mobility grid comprising
        of N-dimensional indices (x, y, z, ...).
    :param index2: Index of the second cell in the mobility grid comprising
        of N-dimensional indices (x, y, z, ...).
    :param mobility_grid: N-dimensional grid containing mobility values.
    """
    λ1 = mobility_grid[index1]
    λ2 = mobility_grid[index2]
    λ_harmonic = compute_harmonic_mean(λ1, λ2)
    return λ_harmonic


def compute_three_phase_relative_permeabilities(
    water_saturation: float,
    oil_saturation: float,
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation: float,
    residual_gas_saturation: float,
    water_exponent: float,
    oil_exponent: float,
    gas_exponent: float,
) -> typing.Tuple[float, float, float]:
    """
    Computes relative permeability for water, oil, and gas in a three-phase system.
    This uses a simplified approach (e.g., modified Corey-type models for each phase),
    acknowledging that oil's relative permeability (oil_relative_permeability) in three phases is complex
    and often requires models like Stone's Method I or II for accuracy.

    Assumptions for this simplified model:
    - Water (wetting phase) water_relative_permeability depends primarily on Sw.
    - Gas (non-wetting phase) krg depends primarily on Sg.
    - Oil (intermediate wetting) oil_relative_permeability is approximated here.
    - krw_max=1, kro_max=1, krg_max=1 (maximum relative permeabilities are 1).

    :param water_saturation: Current water saturation (fraction, between 0 and 1).
    :param oil_saturation: Current oil saturation (fraction, between 0 and 1).
    :param gas_saturation: Current gas saturation (fraction, between 0 and 1).
    :param irreducible_water_saturation: Irreducible water saturation (fraction).
    :param residual_oil_saturation: Residual oil saturation (fraction).
    :param residual_gas_saturation: Residual gas saturation (fraction).
    :param water_exponent: Corey exponent for water. Controls the curvature of the water relative permeability curve.
    :param oil_exponent: Corey exponent for oil. Controls the curvature of the oil relative permeability curve.
    :param gas_exponent: Corey exponent for gas. Controls the curvature of the gas relative permeability curve.
    :return: A tuple (water_relative_permeability, oil_relative_permeability, gas_relative_permeability),
    """
    # Ensure saturations are within physical bounds and sum is manageable
    water_saturation = np.clip(water_saturation, 0.0, 1.0)
    gas_saturation = np.clip(gas_saturation, 0.0, 1.0)
    oil_saturation = np.clip(oil_saturation, 0.0, 1.0)

    # Effective Saturations for Corey-type Models #

    # 1. Effective Water Saturation (for water_relative_permeability)
    movable_water_range = 1.0 - irreducible_water_saturation
    if movable_water_range <= 1e-6:
        effective_water_saturation = 0.0
    else:
        effective_water_saturation = (
            water_saturation - irreducible_water_saturation
        ) / movable_water_range
        effective_water_saturation = np.clip(effective_water_saturation, 0.0, 1.0)
    water_relative_permeability = effective_water_saturation**water_exponent

    # 2. Effective Gas Saturation (for krg)
    movable_gas_range = (
        1.0 - residual_gas_saturation
    )  # Assuming gas becomes mobile above Sgr
    if movable_gas_range <= 1e-6:
        effective_gas_saturation = 0.0
    else:
        effective_gas_saturation = (
            gas_saturation - residual_gas_saturation
        ) / movable_gas_range
        effective_gas_saturation = np.clip(effective_gas_saturation, 0.0, 1.0)
    gas_relative_permeability = effective_gas_saturation**gas_exponent

    # 3. Effective Oil Saturation (for oil_relative_permeability) - Highly Simplified!
    # This is typically the most complex part in 3-phase relative permeability.
    # Stone's models (I or II) are common.
    # A very basic approximation (e.g., for demonstration):
    # Oil's mobile range considering both water and gas residuals.
    total_non_oil_pore_volume = (
        irreducible_water_saturation + residual_gas_saturation + residual_oil_saturation
    )  # This `residual_oil_saturation` is tricky, should be `min(sorw, sorg)`

    # Let's define the total movable pore space for any phase.
    total_movable_pore_space = (
        1.0 - total_non_oil_pore_volume
    )  # Assuming a generic minimum residual oil

    if total_movable_pore_space <= 1e-6:
        effective_oil_saturation = 0.0
    else:
        effective_oil_saturation = (
            oil_saturation - residual_oil_saturation
        ) / total_movable_pore_space
        effective_oil_saturation = np.clip(effective_oil_saturation, 0.0, 1.0)
    oil_relative_permeability = effective_oil_saturation**oil_exponent

    # Ensure all relative permeabilities are within [0, 1]
    water_relative_permeability = np.clip(water_relative_permeability, 0.0, 1.0)
    oil_relative_permeability = np.clip(oil_relative_permeability, 0.0, 1.0)
    gas_relative_permeability = np.clip(gas_relative_permeability, 0.0, 1.0)

    return (
        float(water_relative_permeability),
        float(oil_relative_permeability),
        float(gas_relative_permeability),
    )


def compute_three_phase_capillary_pressures(
    water_saturation: float,
    gas_saturation: float,
    irreducible_water_saturation: float,
    residual_oil_saturation: float,
    residual_gas_saturation: float,
    capillary_pressure_params: CapillaryPressureParameters,
) -> typing.Tuple[float, float]:
    """
    Computes capillary pressures (Pcow, Pcgo) for a given set of saturations
    using Brooks-Corey type models, considering rock wettability.

    Pcow is defined as Po - Pw.
    Pcgo is defined as Pg - Po.

    If water-wet: Pcow > 0, Pcgo > 0
    If oil-wet:   Pcow < 0, Pcgo > 0 (as Po < Pw)

    The capillary pressures are calculated based on the effective saturations
    relative to the mobile pore space, which is defined as:
    total_mobile_pore_space = 1.0 - irreducible_water_saturation - residual_oil_saturation - residual_gas_saturation
    This function handles both water-wet and oil-wet systems, returning the capillary pressures

    :param water_saturation: Current water saturation (fraction, between 0 and 1).
    :param gas_saturation: Current gas saturation (fraction, between 0 and 1).
    :param irreducible_water_saturation: Irreducible water saturation (fraction).
    :param residual_oil_saturation: Residual oil saturation (fraction).
    :param residual_gas_saturation: Residual gas saturation (fraction).
    :param capillary_pressure_params: Parameters defining the capillary pressure curve (e.g., Brooks-Corey).
    :return: A tuple (oil_water_capillary_pressure, gas_oil_capillary_pressure) (psi).
    """

    # Ensure saturations are within bounds before calculation
    water_saturation = np.clip(water_saturation, 0.0, 1.0)
    gas_saturation = np.clip(gas_saturation, 0.0, 1.0)
    oil_saturation = 1.0 - water_saturation - gas_saturation
    oil_saturation = np.clip(oil_saturation, 0.0, 1.0)

    # Effective saturation range - this is the total mobile pore space
    total_mobile_pore_space = (
        1.0
        - irreducible_water_saturation
        - residual_oil_saturation
        - residual_gas_saturation
    )
    if (
        total_mobile_pore_space <= 1e-9
    ):  # Avoid division by zero if all phases are immobile
        return 0.0, 0.0  # Return zero capillary pressure if no mobile pore space

    # Oil-Water Capillary Pressure: (Po - Pw)
    oil_water_capillary_pressure = 0.0

    if capillary_pressure_params.wettability == WettabilityType.WATER_WET:
        # Water-wet system: Water is wetting, Oil is non-wetting.
        # So, P_oil > P_water. Pcow = Po - Pw will be POSITIVE.

        # Effective saturation for water (wetting phase)
        effective_water_saturation_wetting = (
            water_saturation - irreducible_water_saturation
        ) / total_mobile_pore_space
        effective_water_saturation_wetting = np.clip(
            effective_water_saturation_wetting, 1e-6, 1.0
        )  # Clip to avoid division by zero or very large numbers

        if (
            effective_water_saturation_wetting > 1.0 - 1e-6
        ):  # At very high water saturation, capillary pressure approaches zero
            oil_water_capillary_pressure = 0.0
        else:
            # Brooks-Corey model: Pc = Pe * (Se)^(-1/lambda)
            oil_water_capillary_pressure = (
                capillary_pressure_params.oil_water_entry_pressure_water_wet
                * (effective_water_saturation_wetting)
                ** (
                    -1.0
                    / capillary_pressure_params.oil_water_pore_size_distribution_index_water_wet
                )
            )

    elif capillary_pressure_params.wettability == WettabilityType.OIL_WET:
        # Oil-wet system: Oil is wetting, Water is non-wetting.
        # So, P_water > P_oil. Pcow = Po - Pw will be NEGATIVE.

        # Effective saturation for water (non-wetting phase)
        # Note: In an oil-wet system, water is often the invading (non-wetting) phase
        # displacing oil (wetting phase). The Brooks-Corey formula, by default,
        # models a non-wetting phase displacing a wetting phase.
        effective_water_saturation_non_wetting = (
            water_saturation - irreducible_water_saturation
        ) / total_mobile_pore_space
        effective_water_saturation_non_wetting = np.clip(
            effective_water_saturation_non_wetting, 1e-6, 1.0
        )

        if effective_water_saturation_non_wetting > 1.0 - 1e-6:
            oil_water_capillary_pressure_magnitude = 0.0
        else:
            # This calculates the *magnitude* of (P_water - P_oil) for an oil-wet system.
            oil_water_capillary_pressure_magnitude = (
                capillary_pressure_params.oil_water_entry_pressure_oil_wet
                * (effective_water_saturation_non_wetting)
                ** (
                    -1.0
                    / capillary_pressure_params.oil_water_pore_size_distribution_index_oil_wet
                )
            )

        # Since Pcow = Po - Pw, and for oil-wet, Pw > Po, Pcow must be negative.
        oil_water_capillary_pressure = -oil_water_capillary_pressure_magnitude

    # --- Pcgo (Gas-Oil Capillary Pressure: Pg - Po) ---
    # Gas is generally non-wetting to oil, regardless of water/oil wettability.
    # So, Pg > Po. Pcgo will be POSITIVE.
    gas_oil_capillary_pressure = 0.0
    # Effective gas saturation
    effective_gas_saturation = (
        gas_saturation - residual_gas_saturation
    ) / total_mobile_pore_space
    effective_gas_saturation = np.clip(effective_gas_saturation, 1e-6, 1.0)

    if effective_gas_saturation > 1.0 - 1e-6:
        gas_oil_capillary_pressure = 0.0
    else:
        gas_oil_capillary_pressure = (
            capillary_pressure_params.gas_oil_entry_pressure
            * (effective_gas_saturation)
            ** (-1.0 / capillary_pressure_params.gas_oil_pore_size_distribution_index)
        )

    return oil_water_capillary_pressure, gas_oil_capillary_pressure


def compute_oil_specific_gravity_from_density(
    oil_density: float,
    pressure: float,
    temperature: float,
    oil_compressibility: float,
) -> float:
    """
    Converts oil density (lbm/ft³) at given reservoir conditions to specific gravity (dimensionless)
    by adjusting for pressure and temperature effects using a linearized approximation.

    The oil density is corrected to standard temperature and pressure (STP) using the following formula:

        ρ_stp ≈ ρ * exp([Co * (P_stp - P) + α * (T_stp - T)])

    where:
        - ρ_stp: Oil density at standard conditions (lbm/ft³)
        - ρ: Oil density at reservoir conditions (lbm/ft³)
        - Co: Oil compressibility (psi⁻¹)
        - α: Oil thermal expansion coefficient (1/°F)
        - T: Reservoir temperature (°F)
        - P: Reservoir pressure (psi)
        - T_stp: Standard temperature = 60 °F
        - P_stp: Standard pressure = 14.696 psi

    Specific gravity is then calculated as:

        SG = ρ_stp / ρ_water

    where ρ_water is the density of water at standard conditions (assumed 62.4 lbm/ft³).

    :param oil_density: Oil density at reservoir conditions (lbm/ft³)
    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param oil_compressibility: Oil compressibility (psi⁻¹)
    :return: Specific gravity of oil (dimensionless)
    """
    delta_p = STANDARD_PRESSURE_IMPERIAL - pressure
    delta_t = STANDARD_TEMPERATURE_IMPERIAL - temperature
    correction_factor = np.exp(
        (oil_compressibility * delta_p)
        + (OIL_THERMAL_EXPANSION_COEFFICIENT_IMPERIAL * delta_t)
    )
    correction_factor = np.clip(
        correction_factor, 0.2, 2.0
    )  # Avoid numerical issues with very small values
    oil_density_at_stp = oil_density * correction_factor
    return oil_density_at_stp / STANDARD_WATER_DENSITY_IMPERIAL


def convert_surface_rate_to_reservoir(
    surface_rate: float, formation_volume_factor: float
) -> float:
    """
    Converts a surface rate (e.g., STB/day) to reservoir conditions.

    :param surface_rate: Surface volumetric flow rate (e.g., STB/day)
    :param formation_volume_factor: Formation volume factor (FVF) for the fluid (bbl/STB)
    :return: Reservoir volumetric flow rate (e.g., bbl/day)
    """
    if surface_rate > 0:  # Injection
        return surface_rate * formation_volume_factor
    else:  # Production (rate is negative)
        return (
            surface_rate / formation_volume_factor
        )  # Production reservoir volume is larger than surface


def convert_reservoir_rate_to_surface(
    reservoir_rate: float, formation_volume_factor: float
) -> float:
    """
    Converts a reservoir rate (e.g., bbl/day) to surface conditions.

    :param reservoir_rate: Reservoir volumetric flow rate (e.g., bbl/day)
    :param formation_volume_factor: Formation volume factor (FVF) for the fluid (bbl/STB)
    :return: Surface volumetric flow rate (e.g., STB/day)
    """
    if reservoir_rate > 0:  # Injection
        return reservoir_rate / formation_volume_factor
    else:  # Production (rate is negative)
        return (
            reservoir_rate * formation_volume_factor
        )  # Production surface volume is larger than reservoir volume


def compute_oil_formation_volume_factor_standing(
    temperature: float,
    oil_specific_gravity: float,
    gas_gravity: float,
    gas_to_oil_ratio: float,
) -> float:
    """
    Computes the oil formation volume factor (Bo) in m³ oil at reservoir conditions per m³ oil at standard conditions
    using the Standing correlation.

    Formula (Standing, 1947):

        Bo = 0.972 + 0.000147 * [ (R_s * (γ_g / γ_o)^0.5) + (1.25 * T_F) ]^1.175

    Where:
        - Bo: Oil formation volume factor (bbl/STB)
        - R_s: Gas-to-oil ratio in scf/STB
        - γ_g: Gas specific gravity (air = 1.0)
        - γ_o: Oil specific gravity (water = 1.0)
        - T_F: Temperature in degrees Fahrenheit

    Limitations:
        - Valid for light oils and saturated conditions
        - Typical range: 60-300 °F, 0.5-0.95 oil SG, 20 - 2000 scf/STB

    :param temperature: Temperature (°F)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Gas-to-oil ratio in SCF/STB
    :return: Formation volume factor (Bo) in bbl/STB
    """
    if oil_specific_gravity <= 0 or gas_gravity <= 0:
        raise ValueError("Specific gravities must be positive.")
    if gas_to_oil_ratio < 0:
        raise ValueError("Gas-to-oil ratio must be non-negative.")
    if temperature < 32:
        raise ValueError("Temperature seems unphysical (<32 °F). Check units.")

    x = (gas_to_oil_ratio * (gas_gravity / oil_specific_gravity) ** 0.5) + (
        1.25 * temperature
    )
    oil_fvf = 0.972 + 0.000147 * (x**1.175)
    return oil_fvf


def _get_vazquez_beggs_oil_fvf_coefficients(
    oil_api_gravity: float,
) -> typing.Tuple[float, float, float]:
    if oil_api_gravity <= 30:
        return 4.677e-4, 1.751e-5, -1.811e-8
    return 4.670e-4, 1.100e-5, 1.337e-9


def compute_oil_formation_volume_factor_vazquez_and_beggs(
    temperature: float,
    oil_specific_gravity: float,
    gas_gravity: float,
    gas_to_oil_ratio: float,
) -> float:
    """
    Computes the oil formation volume factor (Bo) using the Vazquez and Beggs correlation.

    Formula (Vazquez and Beggs, 1980):
        Bo = 1 + (a1 * R_s) + (a2 * (T - 60) * (γ_o / γ_g)) + (a3 * (T - 60) * R_s * (γ_o / γ_g))
    Where:
        - Bo: Oil formation volume factor (bbl/STB)
        - R_s: Gas-to-oil ratio in scf/STB
        - γ_o: Oil specific gravity (dimensionless)
        - γ_g: Gas specific gravity (dimensionless)
        - T: Temperature in degrees Fahrenheit

    Limitations:
        - Valid for API from 16 - 58
        - Typical range: 100-300 °F, 0.56-1.30 oil SG, 0 - 2000 scf/STB

    :param temperature: Reservoir temperature (°F)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Gas-to-oil ratio in SCF/STB
    :return: Formation volume factor (Bo) in bbl/STB
    """
    oil_api_gravity = compute_oil_api_gravity(oil_specific_gravity)
    a1, a2, a3 = _get_vazquez_beggs_oil_fvf_coefficients(oil_api_gravity)
    oil_fvf = (
        1
        + (a1 * gas_to_oil_ratio)
        + (a2 * (temperature - 60) * (oil_specific_gravity / gas_gravity))
        + (
            a3
            * (temperature - 60)
            * gas_to_oil_ratio
            * (oil_specific_gravity / gas_gravity)
        )
    )
    return oil_fvf


def correct_oil_fvf_for_pressure(
    saturated_oil_fvf: float,
    oil_compressibility: float,
    bubble_point_pressure: float,
    current_pressure: float,
) -> float:
    """
    Applies exponential shrinkage correction to oil FVF for pressures above bubble point.

    Formula:
        B_o(P) = B_o(sat) * exp[c_o * (Pb - P)]

    :param saturated_oil_fvf: Bo at bubble point pressure (saturated conditions) (bbl/STB)
    :param oil_compressibility: Isothermal oil compressibility (psi⁻¹)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param current_pressure: Current reservoir pressure (psi)
    :return: Adjusted Bo at current pressure (bbl/STB)
    """
    if current_pressure <= bubble_point_pressure:
        return saturated_oil_fvf

    delta_p = bubble_point_pressure - current_pressure
    correction_factor = np.clip(
        np.exp(oil_compressibility * delta_p), 1e-6, 5.0
    )  # Avoid numerical issues with very small values
    return saturated_oil_fvf * correction_factor


def compute_oil_formation_volume_factor(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,
    oil_specific_gravity: float,
    gas_gravity: float,
    gas_to_oil_ratio: float,
    oil_compressibility: float,
) -> float:
    """
    Computes the oil formation volume factor (Bo) in bbl/STB of oil
    based on pressure and temperature deviations from reference conditions.

    The formula used is based on the Standing correlation for temperatures below 100°F
    and the Vazquez and Beggs correlation for temperatures above 100°F.

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :param gas_gravity: Gas specific gravity (dimensionless)
    :param gas_to_oil_ratio: Gas-to-oil ratio in SCF/STB
    :param oil_compressibility: Oil isothermal compressibility (psi⁻¹)
    :return: Oil formation volume factor (Bo) in bbl/STB
    """
    # Use standing correlation if temperature is above 100°F
    if temperature <= 100:
        oil_fvf = compute_oil_formation_volume_factor_standing(
            temperature=temperature,
            oil_specific_gravity=oil_specific_gravity,
            gas_gravity=gas_gravity,
            gas_to_oil_ratio=gas_to_oil_ratio,
        )

    else:
        oil_fvf = compute_oil_formation_volume_factor_vazquez_and_beggs(
            temperature=temperature,
            oil_specific_gravity=oil_specific_gravity,
            gas_gravity=gas_gravity,
            gas_to_oil_ratio=gas_to_oil_ratio,
        )

    return correct_oil_fvf_for_pressure(
        saturated_oil_fvf=oil_fvf,
        oil_compressibility=oil_compressibility,
        bubble_point_pressure=bubble_point_pressure,
        current_pressure=pressure,
    )


def compute_water_formation_volume_factor(
    water_density: float,
    salinity: float,
) -> float:
    """
    Computes the water formation volume factor (B_w) in bbl/STB of water
    based on pressure and temperature deviations from reference conditions.

    The formula used is:

        B_w = ρ_stp / ρ_w

    Where:
        - B_w: Water formation volume factor (bbl/STB)
        - ρ_stp: Water density at standard conditions (assumed 62.4 lbm/ft³)
        - ρ_w: Live water density at reservoir conditions (lbm/ft³)

    :param water_density: (Live) Water density at reservoir conditions (lbm/ft³)
    :param salinity: Water salinity (ppm of NaCl)
    :return: Water formation volume factor in bbl/STB
    """
    standard_water_density = compute_standard_water_density(salinity)
    if water_density <= 0:
        raise ValueError("Water density must be positive.")
    if standard_water_density <= 0:
        raise ValueError("Standard water density must be positive.")

    water_fvf = standard_water_density / water_density
    return water_fvf


def compute_gas_formation_volume_factor(
    pressure: float,
    temperature: float,
    gas_compressibility_factor: float,
) -> float:
    """
    Computes the gas formation volume factor (B_g) in ft³/SCF, using the real gas law.

    Formula (real gas EOS):

        B_g = (Z * T * P_std) / (P * T_std)

    Where:
        - B_g: Gas formation volume factor (ft³/SCF)
        - Z: Gas compressibility factor (dimensionless)
        - T: Reservoir temperature (°F)
        - P: Reservoir pressure (psi)
        - P_std: Standard pressure = 14.696 psi
        - T_std: Standard temperature = 60°F

    Assumes ideal gas law corrected with Z-factor (real gas behavior).

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param gas_compressibility_factor: Z-factor (dimensionless)
    :return: Gas formation volume factor (ft³/SCF)
    """
    if pressure <= 0 or temperature <= 0:
        raise ValueError("Pressure and temperature must be positive.")
    if gas_compressibility_factor <= 0:
        raise ValueError("Z-factor must be positive.")

    return (
        gas_compressibility_factor
        * temperature
        * STANDARD_PRESSURE_IMPERIAL
        / (pressure * STANDARD_TEMPERATURE_IMPERIAL)
    )


def compute_gas_compressibility_factor(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> float:
    """
    Computes gas compressibility factor using Papay's correlation for gas compressibility factor,
    with corrections for sour gases using the Wichert-Aziz method.

    Papay's correlation is a widely used empirical relationship to estimate the
    compressibility factor of natural gas based on its pseudo-reduced properties.

    The equation is:

        Z = 1 - ((3.52 * P_r * exp(-0.869 * T_r)) / T_r) + ((0.274 * P_r**2)/T_r**2)

    where:
    - Z is the compressibility factor (dimensionless)
    - P_r is the pseudo-reduced pressure (dimensionless)
    - T_r is the pseudo-reduced temperature (dimensionless)
    - P_r = P / P_pc
    - T_r = T / T_pc
    - P_pc is the pseudo-critical pressure (psi)
    - T_pc is the pseudo-critical temperature (°F)
    - P is the pressure (psi)
    - T is the temperature (°F)
    - P_pc and T_pc are calculated based on the gas specific gravity (gas_gravity).

    :param gas_gravity: Gas specific gravity (dimensionless)
    :param pressure: Pressure in Pascals (psi)
    :param temperature: Temperature in Kelvin (°F)
    :param h2s_mole_fraction: Mole fraction of H2S in the gas (dimensionless, default is 0.0)
    :param co2_mole_fraction: Mole fraction of CO2 in the gas (dimensionless, default is 0.0)
    :param n2_mole_fraction: Mole fraction of N2 in the gas (dimensionless, default is 0.0)
    :return: Compressibility factor Z (dimensionless)
    """
    if pressure <= 0 or temperature <= 0 or gas_gravity <= 0:
        raise ValueError(
            "Pressure, temperature, and gas specific gravity must be positive."
        )
    if not 0 <= h2s_mole_fraction <= 1:
        raise ValueError("H2S mole fraction must be between 0 and 1.")
    if not 0 <= co2_mole_fraction <= 1:
        raise ValueError("CO2 mole fraction must be between 0 and 1.")
    if not 0 <= n2_mole_fraction <= 1:
        raise ValueError("N2 mole fraction must be between 0 and 1.")

    pseudo_critical_pressure, pseudo_critical_temperature = (
        compute_gas_pseudocritical_properties(
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    )

    pseudo_reduced_pressure = pressure / pseudo_critical_pressure
    pseudo_reduced_temperature = temperature / pseudo_critical_temperature

    # Clamp pseudo-reduced values to avoid extreme/unphysical Z outputs
    pseudo_reduced_pressure = np.clip(pseudo_reduced_pressure, 0.2, 30)
    pseudo_reduced_temperature = np.clip(pseudo_reduced_temperature, 1.0, 3.0)
    # Papay's correlation for gas compressibility factor
    compressibility_factor = (
        1
        - (
            (
                3.52
                * pseudo_reduced_pressure
                * np.exp(-0.869 * pseudo_reduced_temperature)
            )
            / pseudo_reduced_temperature
        )
        + ((0.274 * pseudo_reduced_pressure**2) / pseudo_reduced_temperature**2)
    )
    return max(
        compressibility_factor, 0.1
    )  # Ensure Z-factor is not less than 0.1 as Papay's correlation may under-predict at times


def compute_oil_api_gravity(oil_specific_gravity: float) -> float:
    """
    Computes the API gravity (in degrees) from oil specific gravity.

    Formula:

        API = (141.5 / SG) - 131.5

    Where:
        - API: API gravity (degrees)
        - SG: Specific gravity of oil (dimensionless, relative to water at 60°F)

    :param oil_specific_gravity: Oil specific gravity (dimensionless)
    :return: API gravity in degrees (°API)
    """
    if oil_specific_gravity <= 0:
        raise ValueError("Oil specific gravity must be greater than zero.")

    return (141.5 / oil_specific_gravity) - 131.5


def _get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
    oil_api_gravity: float,
) -> typing.Tuple[float, float, float]:
    """
    Returns the empirical coefficients (C₁, C₂, C₃) used in the Vazquez-Beggs
    bubble point pressure correlation based on oil API gravity.

    Coefficients vary for API ≤ 30 and API > 30:

        If API ≤ 30:
            C₁ = 0.0362, C₂ = 1.0937, C₃ = 25.7240
        Else:
            C₁ = 0.0178, C₂ = 1.1870, C₃ = 23.9310

    :param oil_api_gravity: Oil API gravity (°API)
    :return: Tuple (C₁, C₂, C₃)
    """
    if oil_api_gravity <= 30.0:
        return 0.0362, 1.0937, 25.7240
    else:
        return 0.0178, 1.1870, 23.9310


def compute_oil_bubble_point_pressure(
    gas_gravity: float,
    oil_api_gravity: float,
    temperature: float,
    gas_to_oil_ratio: float,
) -> float:
    """
    Computes the bubble point pressure of oil using the Vazquez-Beggs correlation.

    The correlation is defined as:

        P_b = [ R_s / (C₁ * SG * exp((C₃ * API) / T_R)) ]^(1 / C₂)

    Where:
        - P_b: Bubble point pressure (psi)
        - R_s: Gas-to-oil ratio (GOR) in SCF/STB
        - SG: Gas specific gravity (dimensionless)
        - API: Oil gravity in degrees API
        - T_R: Temperature in Rankine (°R)
        - C₁, C₂, C₃: Empirical constants (depend on API gravity)

    Valid for:
        - Oil API: ~16-45°
        - T: ~100-300 °F (converted to Rankine)
        - GOR up to ~2000 scf/STB

    :param gas_gravity: Gas specific gravity (dimensionless)
    :param oil_api_gravity: Oil API gravity in degrees API.
    :param temperature: Temperature (°F)
    :param gas_to_oil_ratio: Gas-to-oil ratio (GOR) in SCF/STB at reservoir conditions
    :return: Bubble point pressure (psi)
    """
    if gas_gravity <= 0:
        raise ValueError("Oil specific gravity must be greater than zero.")
    if oil_api_gravity <= 0:
        raise ValueError("Oil API gravity must be greater than zero.")
    if temperature <= 32.0:
        raise ValueError("Temperature must be greater than absolute zero (32 °F).")
    if gas_to_oil_ratio < 0:
        raise ValueError("Gas-to-oil ratio must be non-negative.")

    c1, c2, c3 = _get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
        oil_api_gravity
    )

    temperature_rankine = temperature + 459.67

    pressure = (
        gas_to_oil_ratio
        / (c1 * gas_gravity * np.exp((c3 * oil_api_gravity) / temperature_rankine))
    ) ** (1 / c2)
    return pressure


def compute_water_bubble_point_pressure(
    temperature: float,
    gas_solubility_in_water: float,
    salinity: float,
    gas: str = "methane",
) -> float:
    """
    Computes the bubble point pressure where the given gas solubility in water is reached.
    Uses analytical inversion for McCain, otherwise numerical root-finding.

    :param temperature: Temperature (°F)
    :param gas_solubility_in_water: Target gas solubility in SCF/STB
    :param salinity: Salinity in ppm
    :param gas: Gas name ("co2", "methane", "n2")
    :return: Bubble point pressure (psi)
    """
    gas = gas.lower()
    if gas == "methane" and 100 <= temperature <= 400:
        # Inverted McCain
        A = 2.12 + 0.00345 * temperature - 0.0000125 * temperature**2
        B = 0.000045
        denominator = B * (1.0 - 0.000001 * salinity)
        bubble_point_pressure = max(0.0, (gas_solubility_in_water - A) / denominator)
        return bubble_point_pressure

    # Use numerical solver for Duan/Henry
    def residual(pressure: float) -> float:
        return (
            compute_gas_solubility_in_water(pressure, temperature, salinity, gas)
            - gas_solubility_in_water
        )

    bubble_point_pressure = brentq(residual, 1.45, 14, 503.8, xtol=1.45e-7)
    return typing.cast(float, bubble_point_pressure)


def compute_gas_to_oil_ratio(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,
    gas_gravity: float,
    oil_api_gravity: float,
    gor_at_bubble_point_pressure: typing.Optional[float] = None,
) -> float:
    """
    Computes the gas-to-oil ratio (GOR) using the Vazquez-Beggs correlation.

    GOR is the amount of gas dissolved in oil at a given pressure and temperature.

    Two regimes:
        - **Saturated region (P < Pb)**: GOR is pressure-dependent.
        - **Undersaturated region (P ≥ Pb)**: GOR = GORb (constant). If not given, it is computed.

    The Vazquez-Beggs formula is:

        GOR = P^C₂ * C₁ * SG * exp[(C₃ * API) / T_R]

    where:
        - GOR: Gas-oil ratio (scf/STB)
        - P: Pressure in psi
        - SG: Gas specific gravity
        - API: Oil API gravity (°API)
        - T_R: Temperature in Rankine
        - C₁, C₂, C₃: Empirical coefficients

    :param pressure: Reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param gas_gravity: Gas specific gravity (dimensionless, air = 1)
    :param oil_api_gravity: Oil API gravity in degrees API
    :param gor_at_bubble_point_pressure: GOR at the bubble point pressure SCF/STB, optional
    :return: Gas-to-oil ratio SCF/STB
    """
    temperature_in_rankine = temperature + 459.67
    c1, c2, c3 = _get_vazquez_beggs_oil_bubble_point_pressure_coefficients(
        oil_api_gravity
    )

    def _gor(pressure: float) -> float:
        return (
            (pressure**c2)
            * c1
            * gas_gravity
            * np.exp((c3 * oil_api_gravity) / temperature_in_rankine)
        )

    if pressure >= bubble_point_pressure:
        if gor_at_bubble_point_pressure is not None:
            return gor_at_bubble_point_pressure

        gor = _gor(bubble_point_pressure)
        if pressure > bubble_point_pressure:
            warnings.warn(
                f"GOR at bubble point was inferred. Current pressure ({pressure:.2f} psi) > Pb ({bubble_point_pressure:.2f} psi)."
            )
        return gor

    gor = _gor(pressure)
    return max(0.0, gor)


def compute_dead_oil_viscosity_beggs(
    temperature: float, oil_api_gravity: float
) -> float:
    """
    Calculates the dead oil viscosity (mu_od) using the Beggs and Robinson correlation.
    Viscosity is in centipoise (cP).

    mu_od = 10^A - 1

    A = 10^(3.0324 - 0.0202 * API) * T_F^(-1.163)

    where:
    - mu_od is the dead oil viscosity (cP)
    - API is the API gravity of the oil (degrees)
    - T_F is the temperature in Fahrenheit (absolute scale, i.e., T_F = T_K * 9/5 - 459.67)

    :param temperature: Reservoir temperature (°F).
    :param oil_api_gravity: API gravity of the oil (degrees).
    :return: Dead oil viscosity in (cP).
    """
    if not (5 <= oil_api_gravity <= 75):  # Some condensates may have very high API
        warnings.warn(
            f"API gravity {oil_api_gravity:.2f} is outside typical range [5, 75]. "
            f"Dead oil viscosity may be inaccurate."
        )

    if temperature < 50:
        warnings.warn(
            f"Temperature {temperature:.2f}°F is unusually low for this correlation. "
            f"Dead oil viscosity may be unreliable."
        )

    # Handle potential issues with temperature if very low
    if temperature <= 0:
        # A more robust approach might be to use a different correlation for very low temps
        # or raise an error for out-of-range inputs for this correlation.
        # For simplicity, we'll return a very high viscosity or raise an error.
        raise ValueError(
            "Temperature in Fahrenheit must be positive for Beggs & Robinson dead oil viscosity."
        )

    exponent_term_A = (3.0324 - 0.0202 * oil_api_gravity) * (temperature**-1.163)
    dead_oil_viscosity = 10 ** (exponent_term_A) - 1

    # Viscosity cannot be negative
    return max(0.0, dead_oil_viscosity)


def _calculate_beggs_robinson_saturated_viscosity_coefficients(
    gas_to_oil_ratio: float,
) -> typing.Tuple[float, float]:
    """
    Calculates the 'a' and 'b' coefficients for the Beggs and Robinson
    saturated oil viscosity correlation.

    a = 10.715 * (Rs + 100)^-0.515
    b = 5.44 * (Rs + 150)^-0.338

    :param gas_to_oil_ratio: Gas-to-oil ratio in standard cubic feet per stock tank barrel (scf/stb).
    :return: A tuple (a, b) where 'a' and 'b' are coefficients for the Beggs and Robinson correlation.
    """
    if gas_to_oil_ratio < 0:
        raise ValueError("GOR (Rs) must be non-negative.")
    elif gas_to_oil_ratio > 5000:
        warnings.warn(
            f"GOR {gas_to_oil_ratio:.2f} scf/STB is above typical range for oil systems. "
            f"Saturated viscosity correlation may be extrapolated."
        )

    a = 10.715 * (gas_to_oil_ratio + 100) ** -0.515
    b = 5.44 * (gas_to_oil_ratio + 150) ** -0.338
    return a, b


def compute_oil_viscosity(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,
    oil_specific_gravity: float,
    gas_to_oil_ratio: float,
    gor_at_bubble_point_pressure: float,
) -> float:
    """
    Computes oil viscosity (cP) using the Beggs & Robinson correlation for dead, saturated, and undersaturated oil.

    This function covers three viscosity regimes:

    - Dead oil viscosity (mu_od):
        mu_od = 10^A - 1
        A = (3.0324 - 0.0202 * API) * T_F^-1.163

    - Saturated oil viscosity (P <= Pb):
        mu_o = a * mu_od^b
        a = 10.715 * (GOR + 100)^-0.515
        b = 5.44 * (GOR + 150)^-0.338

    - Undersaturated oil viscosity (P > Pb):
        mu_o = mu_ob + 0.001 * (P - Pb) * (0.024 * mu_ob^1.6 + 0.038 * mu_ob^0.56)


    :param pressure: Current reservoir pressure (psi)
    :param temperature: Reservoir temperature (°F)
    :param bubble_point_pressure: Bubble point pressure of the oil (psi)
    :param oil_specific_gravity: Specific gravity of the oil (dimensionless)
    :param gas_to_oil_ratio: GOR at current pressure in standard SCF/STB
    :param gor_at_bubble_point_pressure: GOR at bubble point pressure in standard SCF/STB
    :return: Oil viscosity in cP
    """
    if not (32 <= temperature <= 621):
        warnings.warn(
            f"Temperature {temperature:.2f} K is outside common operating range for oil viscosity models."
        )

    if bubble_point_pressure <= 0 or pressure <= 0:
        raise ValueError(
            "Pressure and bubble point pressure must be greater than zero."
        )

    if gas_to_oil_ratio < 0 or gor_at_bubble_point_pressure < 0:
        raise ValueError("GOR values must be non-negative.")

    if oil_specific_gravity < 0.1 or oil_specific_gravity > 1.0:
        warnings.warn(
            f"Oil specific gravity {oil_specific_gravity:.3f} is outside common range [0.1, 1.0]."
        )

    oil_api_gravity = compute_oil_api_gravity(oil_specific_gravity)
    # Calculate Dead Oil Viscosity (mu_od)
    dead_oil_viscosity = compute_dead_oil_viscosity_beggs(temperature, oil_api_gravity)

    # --- Case 1: Saturated Oil Viscosity (P <= Pb) ---
    if pressure <= bubble_point_pressure:
        # Use gas_to_oil_ratio for saturated viscosity
        a, b = _calculate_beggs_robinson_saturated_viscosity_coefficients(
            gas_to_oil_ratio
        )
        saturated_oil_viscosity = a * (dead_oil_viscosity**b)
        return max(0.0, saturated_oil_viscosity)  # Ensure viscosity is not negative

    # --- Case 2: Undersaturated Oil Viscosity (P > Pb) ---
    else:
        # First, calculate viscosity at the bubble point (mu_ob)
        # This uses Rsb because at Pb, the oil is saturated with its max gas.
        a, b = _calculate_beggs_robinson_saturated_viscosity_coefficients(
            gor_at_bubble_point_pressure
        )
        oil_viscosity_at_bubble_point_pressure = a * (dead_oil_viscosity**b)

        # Now, calculate undersaturated viscosity using Vazquez and Beggs extension
        # mu_o = mu_ob + 0.001 * (P - Pb) * (0.024 * mu_ob^1.6 + 0.038 * mu_ob^0.56)
        under_saturated_oil_viscosity = (
            oil_viscosity_at_bubble_point_pressure
            + 0.001
            * (pressure - bubble_point_pressure)
            * (
                0.024 * (oil_viscosity_at_bubble_point_pressure**1.6)
                + 0.038 * (oil_viscosity_at_bubble_point_pressure**0.56)
            )
        )
        return max(
            0.0, under_saturated_oil_viscosity
        )  # Ensure viscosity is not negative


def compute_gas_molecular_weight(gas_gravity: float) -> float:
    """
    Computes the apparent molecular weight of a gas (g/mol) from its specific gravity relative to air.

    Formula:
        MW_gas = γ_gas * MW_air

    where:
    - MW_gas is the molecular weight of the gas (g/mol)
    - γ_gas is the gas specific gravity (dimensionless, air = 1.0)
    - MW_air = 28.96 (g/mol) is the molecular weight of air

    :param gas_gravity: Specific gravity of the gas relative to air (dimensionless)
    :return: Molecular weight of the gas in grams per mole (g/mol)
    """
    if gas_gravity <= 0:
        raise ValueError("Gas specific gravity must be greater than zero.")
    return gas_gravity * MOLECULAR_WEIGHT_AIR


def compute_gas_pseudocritical_properties(
    gas_gravity: float,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> typing.Tuple[float, float]:
    """
    Computes pseudocritical pressure and temperature of natural gas in psi and °F.

    The pseudocritical properties are estimated using Sutton's correlation, which is
    widely used for sweet natural gas. For sour gases, the Wichert-Aziz correction
    adjusts the values based on H₂S, CO₂, and N₂ content.

    This is used as input to pseudo-reduced property models and EOS calculations.

    Sutton's correlation (for sweet gas):
        P_pc = 756.8 - 131.0 * γ_g - 3.6 * γ_g²     [psia]
        T_pc = 169.2 + 349.5 * γ_g - 74.0 * γ_g²    [°R]

    Wichert-Aziz Correction:
        ε = 120[(X_H2S + X_N2)^0.9 - (X_H2S + X_N2)^1.6] + 15[√X_CO2 - X_CO2⁴]

        Then:
        T_pc' = T_pc - ε
        P_pc' = P_pc * T_pc' / (T_pc + X_H2S(1 - X_H2S) * ε)

    :param gas_gravity: Gas specific gravity (dimensionless, air = 1.0).
    :param h2s_mole_fraction: Mole fraction of H₂S (dimensionless).
    :param co2_mole_fraction: Mole fraction of CO₂ (dimensionless).
    :param n2_mole_fraction: Mole fraction of N₂ (dimensionless).
    :return: Tuple (P_pc in psi, T_pc in °F)
    """
    if gas_gravity <= 0:
        raise ValueError("Gas specific gravity must be greater than zero.")

    # Sutton's pseudocritical properties (psia and Rankine)
    pseudocritical_pressure = 756.8 - 131.0 * gas_gravity - 3.6 * gas_gravity**2
    pseudocritical_temperature_rankine = (
        169.2 + 349.5 * gas_gravity - 74.0 * gas_gravity**2
    )

    total_acid_gas_fraction = h2s_mole_fraction + co2_mole_fraction
    if total_acid_gas_fraction > 0.001:
        epsilon = 120.0 * (
            (h2s_mole_fraction + n2_mole_fraction) ** 0.9
            - (h2s_mole_fraction + n2_mole_fraction) ** 1.6
        ) + 15.0 * (co2_mole_fraction**0.5 - co2_mole_fraction**4)

        pseudocritical_temperature_rankine -= epsilon
        pseudocritical_pressure = (
            pseudocritical_pressure
            * pseudocritical_temperature_rankine
            / (
                pseudocritical_temperature_rankine
                + h2s_mole_fraction * (1 - h2s_mole_fraction) * epsilon
            )
        )

    pseudocritical_temperature_in_fahrenheit = (
        pseudocritical_temperature_rankine - 459.67
    )
    return pseudocritical_pressure, pseudocritical_temperature_in_fahrenheit


def compute_gas_density(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    gas_compressibility_factor: float,
) -> float:
    """
    Calculates the gas density (lbm/ft³) using the real gas equation of state.

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :param gas_gravity: Gas specific gravity (dimensionless, air=1.0).
    :param gas_compressibility_factor: Gas compressibility factor (dimensionless).
    :return: Gas density in lbm/ft³.
    """
    temperature_in_rankine = temperature + 459.67
    gas_molecular_weight_lbm_per_mole = (
        compute_gas_molecular_weight(gas_gravity) * GRAMS_PER_MOLE_TO_POUNDS_PER_MOLE
    )

    # Density in lbm/ft3
    gas_density = (pressure * gas_molecular_weight_lbm_per_mole) / (
        gas_compressibility_factor
        * IDEAL_GAS_CONSTANT_IMPERIAL
        * temperature_in_rankine
    )
    return gas_density


def compute_gas_viscosity(
    temperature: float,
    gas_density: float,
    gas_molecular_weight: float,
) -> float:
    """
    Calculates the gas viscosity (cP) using the Lee-Gonzalez-Eakin (LGE) correlation.

    This correlation estimates gas viscosity from gas density and temperature using:

        μ_g = (k * 1e-4) * exp(x * ρ_g^y)

    where:
        - μ_g is the gas viscosity in centipoise (cP)
        - k is a temperature-dependent constant
        - x and y are empirical coefficients based on gas molecular weight and temperature
        - k = [(9.4 + 0.02 * M_g) * T^1.5] / [209 + 19 * M_g + T]
        - x = 3.5 + 986 / T + 0.01 * M_g
        - y = 2.4 - 0.2 * x
        - M_g is the gas molecular weight (lbm/lbmol)
        - ρ_g is gas density in g/cm³
        - T is temperature in Rankine

    Internally computes the gas compressibility factor (Z) and real gas density.

    This correlation is valid for a wide range of temperatures and pressures,
    typically from 100 °F to 400 °F and pressures up to 10,000 psi.

    :param temperature: Temperature (°F)
    :param gas_density: Gas density (lbm/ft³)
    :param gas_molecular_weight: Gas molecular weight (g/mol)
    :return: Gas viscosity in (cP)
    """
    temperature_in_rankine = temperature + 459.67
    gas_molecular_weight_lbm_per_mole = (
        gas_molecular_weight * GRAMS_PER_MOLE_TO_POUNDS_PER_MOLE
    )
    density_in_grams_per_cm3 = gas_density * POUNDS_PER_FT3_TO_GRAMS_PER_CM3

    k = (
        (9.4 + 0.02 * gas_molecular_weight_lbm_per_mole)
        * (temperature_in_rankine**1.5)
        / (209 + 19 * gas_molecular_weight_lbm_per_mole + temperature_in_rankine)
    )

    x = (
        3.5
        + (986 / temperature_in_rankine)
        + (0.01 * gas_molecular_weight_lbm_per_mole)
    )
    y = 2.4 - (0.2 * x)

    exponent = x * (density_in_grams_per_cm3**y)
    exponent = min(700, max(-700, exponent))  # cap to prevent overflow

    gas_viscosity = (k * 1e-4) * np.exp(exponent)
    return max(0.0, gas_viscosity)


def kelvin_to_fahrenheit(temp_K: float) -> float:
    """Converts temperature from Kelvin to Fahrenheit."""
    return (temp_K - 273.15) * 9 / 5 + 32


def fahrenheit_to_kelvin(temp_F: float) -> float:
    """Converts temperature from Fahrenheit to Kelvin."""
    return (temp_F - 32) * 5 / 9 + 273.15


def compute_water_viscosity(
    temperature: float,
    salinity: float = 0.0,
    pressure: typing.Optional[float] = None,
) -> float:
    """
    Calculates the water (brine) viscosity using the McCain correlation.

    This correlation considers the effects of temperature and salinity.
    The effect of pressure is generally considered negligible for water
    viscosity in most reservoir engineering applications, especially at
    moderate pressures, and is not explicitly included here.

    The McCain correlation is given by:

    - For pure water viscosity at atmospheric pressure:

        mu_w_o = 2.414 * 10**(-5) * 10**(247.8 / (temp_K - 140))

    - For brine viscosity:

        mu_ws = mu_w_o * (1 + 0.0001 * S_ppm * (2.2 - 0.015 * T_F))

    - At a given pressure, the effect on water viscosity is typically
    negligible, but if needed, it can be included as a small correction.

        mu_ws_p = (0.9994 + (4.029e-5 * P) + (3.106e-9 * P**2)) * mu_ws

    Note:
    The McCain correlation is typically valid for temperatures between
    approximately 100°F and 400°F (37.8°C to 204.4°C) and salinities
    up to about 260,000 ppm NaCl equivalent. It may not be accurate
    for extreme conditions outside this range.

    where:
    - mu_w_o is the pure water viscosity at atmospheric pressure (cP)
    - mu_ws is the saline water (brine) viscosity (cP) at atmospheric pressure
    - S_ppm is the salinity in parts per million (ppm)
    - T_F is the temperature in Fahrenheit (absolute scale, i.e., T_F = T_K * 9/5 - 459.67)
    - P is the pressure (psi), but typically not included in McCain's correlation
    - mu_ws_p is the water viscosity at pressure P (cP)

    :param temperature: Reservoir temperature in Kelvin.
    :param salinity: Salinity of the formation water in parts per million (ppm).
        (e.g., 50000 for 50,000 ppm NaCl equivalent).
    :param pressure: Optional reservoir pressure (psi). If not provided, the effect of pressure is ignored.
    :return: Water (brine) viscosity in (cP).
    """
    # Convert temperature to Fahrenheit for the correlation

    # Validate temperature range for McCain correlation (approx. 100-400 F)
    # The correlation is less accurate outside this range, and should ideally
    # be used with lab data if available for extreme conditions.
    if not (100.0 <= temperature <= 400.0):
        warnings.warn(
            f"Warning: Temperature {temperature:.2f} °F is outside the typical "
            f"range (100-400 °F) for McCain's water viscosity correlation. "
            f"Results may be less accurate."
        )

    # 1. Calculate pure water viscosity at atmospheric pressure (mu_w_o)
    # A = 1.002
    # B = 7.915 - 1.95 * ln(T_F)
    # mu_w_o = A * T_F^B

    # There are slight variations in the constants for McCain's correlation
    # for pure water viscosity. Let's use a very common one:

    # McCain's (1990) equation for pure water at atmospheric pressure (often used in reservoir engineering)
    # mu_w_o = 2.414 * 10**(-5) * 10**(247.8 / (temp_K - 140)) # This is for Pa.s or mPa.s

    # Let's stick with the form given in most petroleum engineering texts for McCain's pure water viscosity.
    # A common form for pure water viscosity at atmospheric pressure:

    # This is a very common empirical fit for pure water viscosity vs. temperature (in cP):
    # This is often attributed to Standing or generalized fits.
    # For example, using a common polynomial fit for water viscosity as function of temperature in F
    # This is a simplified representation of water viscosity vs. temperature:

    # Let's use a more explicit set of coefficients for pure water viscosity from a reliable source.
    # A widely cited empirical equation (e.g., from Petroleum Reservoir Engineering Practice by Tarek Ahmed):
    # For pure water viscosity (cP) as function of temperature (F):
    pure_water_viscosity_at_atm_pressure = np.exp(
        1.0035 - 1.479e-02 * temperature + 1.9825e-05 * temperature**2
    )

    # Validate for extreme salinity
    if salinity < 0:
        raise ValueError("Salinity cannot be negative.")
    if salinity > 300000:  # Typical upper limit for correlation validity
        warnings.warn(
            f"Warning: Salinity {salinity} ppm is very high. "
            f"McCain correlation may be less accurate for salinities above ~260,000 ppm."
        )

    # 2. Apply correction for salinity
    # S_corr = 0.0001 * (S_ppm) * (2.2 - 0.015 * T_F)
    # mu_ws = mu_w_o * (1 + S_corr)

    # This is McCain's salinity correction:
    salinity_correction_factor = 0.0001 * salinity * (2.2 - 0.015 * temperature)
    saline_water_viscosity_at_atm_pressure = pure_water_viscosity_at_atm_pressure * (
        1.0 + salinity_correction_factor
    )

    saline_water_viscosity_at_atm_pressure = max(
        0.0, saline_water_viscosity_at_atm_pressure
    )

    # 3: Apply correction for pressure
    # Factor to correct viscosity from atmospheric pressure to reservoir pressure
    # mu_w / mu_w_o = (0.9994 + 4.029 * 10^-5 * P + 3.106 * 10^-9 * P^2)
    # Where P is in psia
    if pressure is not None:
        pressure_correction_ratio = (
            0.9994 + (4.029e-5 * pressure) + (3.106e-9 * pressure**2)
        )
        return saline_water_viscosity_at_atm_pressure * pressure_correction_ratio

    return saline_water_viscosity_at_atm_pressure


@functools.lru_cache(maxsize=128)
def _compute_oil_compressibility_correction_term(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    oil_api_gravity: float,
    bubble_point_pressure: float,
    gas_formation_volume_factor: float,
    oil_formation_volume_factor: float,
    gor_at_bubble_point_pressure: float,
) -> float:
    """
    Computes the correction term for oil compressibility below bubble point pressure.

    The correction term is give by:

        x = (Bg/Bo * dRs/dp) / 5.615

        dRs/dp = (R_s(P + ΔP) - R_s(P - ΔP)) / (2 * ΔP)

    where:
    - R_s is the solution Gas-Oil Ratio (GOR) at current pressure and temperature
      in standard cubic feet per stock tank barrel (scf/stb).
    - ΔP is a small pressure increment (e.g., 0.01 psi or 0.0001 * P).
    - Bg is the gas formation volume factor (bbl/scf).
    - Bo is the oil formation volume factor (bbl/STB).
    - 5.615 is a conversion factor to convert from scf/STB to bbl/STB.
    - x is the correction term for oil compressibility.


    :param pressure: Current reservoir pressure (psi).
    :param bubble_point_pressure: Bubble point pressure (psi).
    :param gas_formation_volume_factor: Gas formation volume factor (bbl/scf).
    :param oil_formation_volume_factor: Oil formation volume factor (bbl/STB).
    :param gor_at_bubble_point_pressure: GOR at bubble point pressure (scf/stb).
    :return: Correction term for oil compressibility.
    """
    delta_p = max(0.01, 1e-4 * pressure)
    gor_plus_delta = compute_gas_to_oil_ratio(
        pressure=pressure + delta_p,
        temperature=temperature,
        bubble_point_pressure=bubble_point_pressure,
        gas_gravity=gas_gravity,
        oil_api_gravity=oil_api_gravity,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
    )
    gor_minus_delta = compute_gas_to_oil_ratio(
        pressure=pressure - delta_p,
        temperature=temperature,
        bubble_point_pressure=bubble_point_pressure,
        gas_gravity=gas_gravity,
        oil_api_gravity=oil_api_gravity,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
    )
    dRs_dp = (gor_plus_delta - gor_minus_delta) / (2 * delta_p)

    return (gas_formation_volume_factor / oil_formation_volume_factor) * (
        dRs_dp / 5.615
    )


def compute_oil_compressibility(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,
    oil_api_gravity: float,
    gas_gravity: float,
    gor_at_bubble_point_pressure: float,
    gas_formation_volume_factor: float = 1.0,
    oil_formation_volume_factor: float = 1.0,
) -> float:
    """
    Calculates the oil compressibility (C_o) in psi⁻¹ using the Vasquez and Beggs (1980) correlation.

    - If Pressure (P) > Bubble Point Pressure (Pb): Uses the Vasquez and Beggs
      correlation for undersaturated oil compressibility.
    - If Pressure (P) <= Bubble Point Pressure (Pb): The oil is saturated.
      For the *liquid phase compressibility*, it's commonly approximated as
      the compressibility at the bubble point pressure. The dominant volume
      change below Pb is due to gas liberation, which is handled in total system
      compressibility.

    The Vasquez and Beggs correlation is given by:
    For P > Pb (Undersaturated Oil):

        C_o = (-1433 + 5 * R_s + 17.2 * T_F - 1180 * S.G + 12.61 * API) / P

    For P <= Pb (Saturated Oil):

        C_o = C_o(P) + (Bg/Bo * dRs/dp) / 5.615

    where:
    - C_o is the oil compressibility (psi⁻¹)
    - R_s is the solution Gas-Oil Ratio (GOR) at current pressure and temperature
      in standard cubic feet per stock tank barrel (scf/stb).
    - T_F is the temperature in Fahrenheit (°F).
    - S.G is the specific gravity of the solution gas (dimensionless, air=1.0).
    - API is the API gravity of the stock tank oil (degrees).
    - P is the pressure in psi (pounds per square inch).

    Vasquez and Beggs correlation is typically valid for:
    - Pressure: 100 to 5,000 psi
    - Temperature: 100 to 300 °F
    - API gravity: 16 to 58 degrees

    :param pressure: Reservoir pressure (psi).
    :param temperature: Reservoir temperature (°F).
    :param bubble_point_pressure: Bubble point pressure (psi).
    :param oil_api_gravity: API gravity of the stock tank oil.
    :param gas_gravity: Specific gravity of the solution gas (air=1).
    :param gor_at_bubble_point_pressure: Solution Gas-Oil Ratio at bubble point pressure (SCF/STB).
        This value should be obtained from a GOR correlation (e.g., Vazquez-Beggs GOR).
    :return: Oil compressibility in psi⁻¹
    """
    if (
        pressure <= 0
        or bubble_point_pressure <= 0
        or temperature <= 0
        or gas_gravity <= 0
        or oil_api_gravity <= 0
    ):
        raise ValueError(
            "All input parameters (P, Pb, T, Gas SG, API) must be positive."
        )

    def base_formula(pressure: float) -> float:
        val = (
            -1433
            + 5 * gor_at_bubble_point_pressure
            + 17.2 * temperature
            - 1180 * gas_gravity
            + 12.61 * oil_api_gravity
        ) / pressure
        return max(val, 0.0)

    if pressure > bubble_point_pressure:
        return base_formula(pressure)

    correction_term = _compute_oil_compressibility_correction_term(
        pressure=pressure,
        temperature=temperature,
        gas_gravity=gas_gravity,
        oil_api_gravity=oil_api_gravity,
        bubble_point_pressure=bubble_point_pressure,
        gas_formation_volume_factor=gas_formation_volume_factor,
        oil_formation_volume_factor=oil_formation_volume_factor,
        gor_at_bubble_point_pressure=gor_at_bubble_point_pressure,
    )
    return base_formula(pressure) + correction_term


def compute_gas_compressibility(
    pressure: float,
    temperature: float,
    gas_gravity: float,
    gas_compressibility_factor: typing.Optional[float] = None,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> float:
    """
    Calculates the isothermal gas compressibility (C_g) in psi⁻¹ using Papay's Z-factor correlation
    and its analytical derivative.

    :param pressure: Reservoir pressure (psi).
    :param temperature: Reservoir temperature (°F).
    :param gas_gravity: Specific gravity of the gas (air=1).
    :param gas_compressibility_factor: Optional pre-computed Z-factor (dimensionless).
        If provided, it will be used directly instead of (re)calculating it.
    :param h2s_mole_fraction: H2S mole fraction (0 to 1).
    :param co2_mole_fraction: CO2 mole fraction (0 to 1).
    :param n2_mole_fraction: N2 mole fraction (0 to 1).
    :return: Gas compressibility in psi⁻¹.
    """
    if pressure <= 0 or temperature <= 0 or gas_gravity <= 0:
        raise ValueError(
            "Pressure, temperature, and gas specific gravity must be positive."
        )

    # Calculate pseudocritical properties
    pseudo_critical_pressure, pseudo_critical_temperature = (
        compute_gas_pseudocritical_properties(
            gas_gravity=gas_gravity,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )
    )

    # Calculate pseudo-reduced properties at the given pressure and temperature
    pseudo_reduced_pressure = pressure / pseudo_critical_pressure
    pseudo_reduced_temperature = temperature / pseudo_critical_temperature

    # Calculate Z-factor using your provided function
    if gas_compressibility_factor is not None:
        # If a Z-factor is provided, use it directly
        Z = gas_compressibility_factor
    else:
        Z = compute_gas_compressibility_factor(
            gas_gravity=gas_gravity,
            pressure=pressure,
            temperature=temperature,
            h2s_mole_fraction=h2s_mole_fraction,
            co2_mole_fraction=co2_mole_fraction,
            n2_mole_fraction=n2_mole_fraction,
        )

    # Calculate the analytical derivative of Z with respect to P_r (pseudo-reduced pressure)
    # from Papay's correlation: dZ/dP_r = -3.52 * exp(-0.869 * T_r) / T_r + 0.548 * P_r / T_r^2

    exp_term = np.exp(-0.869 * pseudo_reduced_temperature)
    dZ_dP_r = (-3.52 * exp_term / pseudo_reduced_temperature) + (
        0.548 * pseudo_reduced_pressure / (pseudo_reduced_temperature**2)
    )

    # Calculate the gas compressibility (C_g)
    # C_g = (1/P) - (1/(Z * Ppc)) * (dZ/dPpr)

    # Ensure Ppc is not zero, which should be handled by compute_gas_pseudocritical_properties
    if pseudo_critical_pressure == 0:
        raise ValueError(
            "Pseudo-critical pressure cannot be zero for compressibility calculation."
        )

    gas_compressibility = (1 / pressure) - (
        1 / (Z * pseudo_critical_pressure)
    ) * dZ_dP_r
    return max(0.0, gas_compressibility)  # Compressibility must be non-negative


def _gas_solubility_in_water_mccain_methane(
    pressure: float,
    temperature: float,
    salinity: float = 0.0,
) -> float:
    """
    Calculates gas solubility in water (Rsw) using McCain's correlation (1990).

    This correlation is valid for typical reservoir conditions:
        - Temperature: 311 K to 478 K (100 °F to 400 °F)
        - Pressure: 0-10,000 psia
        - Salinity: 0-150,000 ppm

    The formula is:

        Rsw = A(T_F) + (B * P_psia * (1 - 1e-6 * Salinity_ppm))

    where:
        A(T_F) = 2.12 + 0.00345 * T_F - 0.0000125 * T_F²
        B = 0.000045
        T_F = temperature in degrees Fahrenheit
        P_psia = pressure in psia
        Rsw = gas solubility in scf/STB

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :param salinity: Salinity in parts per million (ppm).
    :return: Gas solubility in water in SCF/STB.
    """
    # print(pressure, temperature, salinity)
    if pressure < 0 or temperature < 0 or salinity < 0:
        raise ValueError("Pressure, temperature, and salinity must be non-negative.")

    if not (100 <= temperature <= 400):
        raise ValueError(
            "Temperature out of valid range for McCain's Rsw correlation (311 K to 478 K)."
        )

    # A(T_F) term from McCain
    A_term = 2.12 + 0.00345 * temperature - 0.0000125 * temperature**2

    # B is a constant in the validated McCain form
    B = 0.000045

    salinity_correction = 1.0 - 0.000001 * salinity
    gas_solubility = A_term + (B * pressure * salinity_correction)
    return max(0.0, gas_solubility)  # clamp to non-negative


def _gas_solubility_in_water_duan_co2(
    pressure: float, temperature: float, salinity: float = 0.0
) -> float:
    """
    Calculates CO₂ solubility in water (Rsw) using Duan's empirical correlation
    with salinity correction.

    The model is valid for:
        - Temperature: 273.15 K to 573.15 K
        - Pressure: up to 200 MPa
        - Salinity: expressed as molality derived from ppm NaCl

    The solubility is computed from the mole fraction of CO₂ in water:

        ln(x_CO2) = A + B*P + C*ln(P) + D/T + E*T + F*T² + G*P² + H/T² + I*ln(T) + J*P/T + ln(γ)

        ln(γ) = -S * (0.03 + 1.5/T + 0.0005*P)

    where:
        - x_CO2 is the mole fraction of CO₂ in water
        - T is temperature in Kelvin
        - P is pressure in MPa
        - S is the NaCl molality in mol/kg water
        - γ is the activity coefficient for CO₂ in saline water

    The mole fraction is converted to volume basis using:

        Rsw = (x_CO2 * M_CO2) / ρ_water

    where:
        - M_CO2 is molar mass of CO₂ (kg/mol)
        - ρ_water is water density (lbm/ft³), obtained from EOS

    :param pressure: Pressure (psi)
    :param temperature: Temperature (°F)
    :param salinity: Salinity in parts per million (ppm NaCl)
    :return: CO₂ solubility in SCF/STB.
    """
    if pressure <= 0 or temperature <= 0:
        raise ValueError("Pressure and temperature must be positive.")

    # Convert units
    P = pressure * PSI_TO_PA / 1e6  # Pa → MPa
    T = fahrenheit_to_kelvin(temperature)  # °F → K
    S = salinity / (
        MOLECULAR_WEIGHT_NACL * 1000
    )  # ppm → g/kg → approx mol/kg (assume ~1 mol/kg per 58.44 g/L NaCl)

    # Duan constants (for CO₂ in brine)
    A = -60.2409
    B = 93.4517
    C = 23.3585
    D = 0.023517
    E = -0.0000449
    F = 0.00000042
    G = 0.00000003
    H = -0.00000025
    I = 0.032975  # noqa
    J = -0.00035

    # ln(gamma) salinity correction
    ln_gamma = -S * (0.03 + (1.5 / T) + 0.0005 * P)

    # log mole fraction of CO₂
    ln_x = (
        A
        + B * P
        + C * np.log(P)
        + D / T
        + E * T
        + F * T**2
        + G * P**2
        + H / T**2
        + I * np.log(T)
        + J * P / T
        + ln_gamma
    )

    x_CO2 = np.exp(ln_x)  # mole fraction
    if x_CO2 < 0:
        raise ValueError("Calculated mole fraction of CO₂ is negative, check inputs.")

    try:
        water_density = (
            compute_fluid_density(pressure, temperature, "Water")
            * POUNDS_PER_FT3_TO_KG_PER_M3
        )
    except Exception:
        water_density = STANDARD_WATER_DENSITY

    co2_molar_mass_in_kg_per_mol = (
        MOLECULAR_WEIGHT_CO2 / 1000
    )  # Convert g/mol to kg/mol
    gas_solubility = (x_CO2 * co2_molar_mass_in_kg_per_mol) / water_density  # m³/m³
    return gas_solubility * M3_PER_M3_TO_SCF_PER_STB


MOLAR_MASSES = {
    "co2": MOLECULAR_WEIGHT_CO2 / 1000,  # Convert g/mol to kg/mol
    "methane": MOLECULAR_WEIGHT_METHANE / 1000,  # Convert g/mol to kg/mol
    "n2": MOLECULAR_WEIGHT_N2 / 1000,  # Convert g/mol to kg/mol
}

# Henry's constants (Sander, 2020) — ln(H) = A + B/T + C*ln(T)
# H in mol/(m³·Pa), will be inverted to Pa·m³/mol
HENRY_COEFFICIENTS = {
    "co2": (-58.0931, 90.5069, 0.027766),
    "methane": (-68.8862, 101.4956, 0.021599),
    "n2": (-71.0592, 120.1052, 0.02624),
}


def _gas_solubility_in_water_henry_law(
    pressure: float,
    temperature: float,
    salinity: float = 0.0,
    gas: str = "co2",
) -> float:
    """
    Estimates gas solubility in water using Henry's Law with Setschenow salinity correction.

    Formula:
        Rsw = (P / H(T)) * (M / ρ_water) * exp(-k_s * molality)

    Henry's constant H(T) is computed as:
        ln H = A + B / T + C * ln(T)     [Sander, 2020]

    Setschenow correction:
        molality = salinity_ppm / (58.44 * 1000)
        exp(-k_s * molality)

    where:
    - Rsw is the gas solubility in m³/m³
    - P is the pressure in Pa
    - H(T) is Henry's constant in Pa·m³/mol
    - M is the molar mass of the gas in kg/mol
    - ρ_water is the water density in kg/m³
    - k_s is the Setschenow constant for the gas (dimensionless)
    - molality is the salinity in mol/kg (converted from ppm NaCl)

    :param pressure: Pressure in (psi)
    :param temperature: Temperature in (°F)
    :param salinity: Salinity in ppm NaCl
    :param gas: One of "co2", "methane", or "n2"
    :return: Solubility in SCF/STB (standard cubic feet per stock tank barrel)
    """
    gas = gas.lower()
    if gas not in HENRY_COEFFICIENTS:
        raise ValueError(f"Unsupported gas '{gas}' for Henry's Law fallback.")

    A, B, C = HENRY_COEFFICIENTS[gas]
    M = MOLAR_MASSES[gas]
    temperature_in_kelvin = fahrenheit_to_kelvin(temperature)

    ln_H_inv = -(A + B / temperature_in_kelvin + C * np.log(temperature_in_kelvin))
    H_inv = np.exp(ln_H_inv)  # mol/(m³·Pa)
    H = 1.0 / H_inv  # Pa·m³/mol

    try:
        water_density = (
            compute_fluid_density(pressure, temperature, "Water")
            * POUNDS_PER_FT3_TO_KG_PER_M3
        )
    except Exception:
        water_density = STANDARD_WATER_DENSITY

    # Setschenow salinity correction
    # Converts salinity from ppm (mg/kg) to mol/kg using molar mass in g/mol
    molality = salinity / (MOLECULAR_WEIGHT_NACL * 1000)

    setschenow_constants = {"co2": 0.12, "methane": 0.17, "n2": 0.13}
    k_s = setschenow_constants[gas]
    salinity_factor = np.exp(-k_s * molality)

    gas_solubility = (
        (pressure / H) * (M / water_density) * salinity_factor
    )  # m³ gas / m³ water
    return gas_solubility * M3_PER_M3_TO_SCF_PER_STB


def compute_gas_solubility_in_water(
    pressure: float,
    temperature: float,
    salinity: float = 0.0,
    gas: str = "methane",
) -> float:
    """
    Computes gas solubility in water using McCain, Duan, or Henry's Law based on gas type and temperature.

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :param salinity: Salinity in parts per million (ppm).
    :param gas: Type of gas ("methane", "CO2", or "N2"). Default is "methane".
    :return: Gas solubility in water in SCF/STB (standard cubic feet per stock tank barrel).
    """
    gas = gas.lower()
    if gas == "methane" and 100.0 <= temperature <= 400.0:
        # For methane, we use McCain's correlation for gas solubility in water
        return _gas_solubility_in_water_mccain_methane(pressure, temperature, salinity)

    elif gas == "co2" and 32 <= temperature <= 572:
        # For CO2, we use Duan's correlation for higher accuracy
        return _gas_solubility_in_water_duan_co2(pressure, temperature, salinity)

    elif gas in {"methane", "co2", "n2"}:
        return _gas_solubility_in_water_henry_law(pressure, temperature, salinity, gas)

    else:
        raise NotImplementedError(f"No model for gas '{gas}'.")


def compute_gas_free_water_formation_volume_factor(
    pressure: float, temperature: float
) -> float:
    """
    Calculates the Water Formation Volume Factor (Bw) for dissolved-gas-free water
    using McCain's correlation (based on Petroleum Office function BwMcCain_GasFree).
    This function primarily accounts for thermal expansion and water compressibility,
    without the effect of dissolved gas.

    Bwd = (1.00012 + 1.25E-5 * T_F + 2.45E-7 * T_F**2) * (1.0 - 1.95E-9 * P_psia + 1.72E-13 * P_psia**2)
    Note: This is often for pure water at 60F, then corrected.
    Let's use a simpler form combining thermal and pressure effects,
    often attributed to McCain's overall behavior for pure water.

    From McCain, W.D., "The Properties of Petroleum Fluids", 3rd Ed., p. 326-327:
    V_T = -1.0001E-2 + 1.33391E-4 * T_F + 5.50654E-7 * T_F**2 # Thermal Expansion
    V_P = -1.95301E-9 * P_psia + 1.72492E-13 * P_psia**2 # Isothermal Compressibility from 14.7 psi
    Bw_gas_free = (1.0 + V_T) * (1.0 + V_P)

    :param pressure: Pressure (psi).
    :param temperature: Temperature (°F).
    :return: Dissolved-gas-free Water Formation Volume Factor (Bw_gas_free) in bbl/STB.
    """
    if pressure < 0 or temperature < 0:
        raise ValueError(
            "Pressure and temperature cannot be negative for gas-free water FVF."
        )

    thermal_expansion = (
        -0.010001 + (1.33391e-4 * temperature) + (5.50654e-7 * temperature**2)
    )
    isothermal_compressibility = -(1.95301e-9 * pressure) + (1.72492e-13 * pressure**2)
    gas_free_water_fvf = (1.0 + thermal_expansion) * (1.0 + isothermal_compressibility)
    return max(0.9, gas_free_water_fvf)  # Bw_gas_free is typically close to 1.0


def _compute_dRsw_dP_mccain(temperature: float, salinity: float) -> float:
    """
    Calculates the derivative of gas solubility in water (Rsw) with respect to pressure,
    based on McCain's correlation for Rsw.
    Returns dRsw/dP in scf/(STB*psi).
    """
    if temperature < 0 or salinity < 0:
        raise ValueError("Temperature and salinity cannot be negative for dRsw/dP.")

    derivative_pure_water = (
        0.0000164 + (0.000000134 * temperature) - (0.00000000185 * temperature**2)
    )
    salinity_correction_factor = 1.0 - 0.000001 * salinity
    # This derivative is positive (Rsw increases with P)
    return derivative_pure_water * salinity_correction_factor


def _compute_dBw_gas_free_dp_mccain(
    pressure: float,
    temperature: float,
) -> float:
    """
    Calculates the derivative of dissolved-gas-free Water Formation Volume Factor (Bw_gas_free)
    with respect to pressure, based on McCain's correlation.
    Returns dBw_gas_free/dP in res bbl/(STB*psi). This value will be negative.
    """
    if pressure < 0:
        raise ValueError("Pressure cannot be negative for dBw_gas_free/dP.")

    thermal_expansion_term = (
        1.0  # This 1.0 is part of (1 + VT)
        + -0.010001
        + (1.33391e-4 * temperature)
        + (5.50654e-7 * temperature**2)
    )
    # Derivative of V_P wrt P: d(V_P)/dP = -1.95301E-9 + 2 * 1.72492E-13 * P_psia
    isothermal_compressibility_derivative = -(1.95301e-9) + (2 * 1.72492e-13 * pressure)

    # dBw_gas_free/dP = d/dP [ (1+VT)*(1+VP) ] = (1+VT) * d(1+VP)/dP
    # This value will be negative as Bw decreases with increasing P.
    return thermal_expansion_term * isothermal_compressibility_derivative


def compute_water_compressibility(
    pressure: float,
    temperature: float,
    bubble_point_pressure: float,  # This Pwb is for the water's dissolved gas in water.
    gas_formation_volume_factor: float,  # Bg in ft3/SCF
    gas_solubility_in_water: float,  # Rsw in SCF/STB
    gas_free_water_formation_volume_factor: float,  # Bw_gas_free in bbl/STB (output of compute_gas_free_water_formation_volume_factor)
    salinity: float = 0.0,
) -> float:
    """
    Calculates the isothermal water compressibility (C_w) using McCain's correlations.
    Distinguishes between undersaturated and saturated water conditions.

    The McCain-based correlation for water compressibility is given by:

    - For Undersaturated Water (P >= Pwb):
        C_w = - (1/Bw) * (dBw_gas_free/dP)_T
        (Here, Bw is Bw_gas_free as no dissolved gas comes out of solution.)

    - For Saturated Water (P < Pwb):
        C_w = - (1/Bw_actual) * (dBw_gas_free/dP)_T  + (Bg / Bw_actual) * (dRsw/dP)_T
        (This form adds the effect of gas coming out of solution to the base liquid compressibility.
         Bw_actual = Bw_gas_free + Rsw * Bg)

    :param pressure: Reservoir pressure in (psi).
    :param temperature: Reservoir temperature in (°F).
    :param bubble_point_pressure: Water bubble point pressure in (psi).
        This is the pressure at which gas starts to come out of solution from water.

    :param gas_formation_volume_factor: Gas formation volume factor (Bg) in (ft³/SCF) at the current pressure and temperature.
    :param gas_solubility_in_water: Gas solubility in water (Rsw) in (SCF/STB) at the current pressure and temperature.
    :param gas_free_water_formation_volume_factor: Gas-free water formation volume factor (Bw_gas_free) in (bbl/STB).
        This should be computed using compute_gas_free_water_formation_volume_factor.

    :param salinity: Salinity in parts per million (ppm).
    :return: Water compressibility (C_w) in (psi⁻¹).
    """
    gas_fvf_in_bbl_per_scf = gas_formation_volume_factor * FT3_TO_BBL
    dBw_gas_free_dP = _compute_dBw_gas_free_dp_mccain(
        pressure=pressure,
        temperature=temperature,
    )
    dRsw_dP = _compute_dRsw_dP_mccain(
        temperature=temperature,
        salinity=salinity,
    )

    if pressure >= bubble_point_pressure:
        # --- Undersaturated Water (P >= Pwb) ---
        # C_w = - (1/Bw_gas_free) * (dBw_gas_free/dP)_T
        # dBw_gas_free_dP is negative, so -dBw_gas_free_dP is positive, resulting in positive Cw.
        if gas_free_water_formation_volume_factor <= 0:
            raise ValueError("Calculated Bw for undersaturated water is non-positive.")

        water_compressibility = (
            -(1.0 / gas_free_water_formation_volume_factor) * dBw_gas_free_dP
        )

    else:
        # --- Saturated Water (P < Pwb) ---
        # Bw_actual = Bw_gas_free + Rsw * Bg
        # C_w = C_w_gas_free + (Bg / Bw_actual) * dRsw_dP
        # This is the most common practical form where Cw_gas_free term is for the liquid compression
        # and the second term is for gas liberation.

        # Calculate Bw_actual (water formation volume factor at current conditions, accounting for dissolved gas)
        water_fvf_in_bbl_per_stb = gas_free_water_formation_volume_factor + (
            gas_solubility_in_water * gas_fvf_in_bbl_per_scf
        )
        if water_fvf_in_bbl_per_stb <= 0:
            raise ValueError("Calculated Bw for saturated water is non-positive.")

        # C_w_gas_free_component: Compressibility of the gas-free water itself
        c_w_gas_free_component = -(1.0 / water_fvf_in_bbl_per_stb) * dBw_gas_free_dP
        # Note: If water_fvf_in_bbl_per_stb is used in the denominator here, it's slightly different
        # from C_w_gas_free = -(1/Bw_gas_free)*dBw_gas_free_dP, but common in formulations for total system.

        # Gas liberation component: Contribution to compressibility from gas coming out of solution
        gas_liberation_component = (
            gas_fvf_in_bbl_per_scf / water_fvf_in_bbl_per_stb
        ) * dRsw_dP

        water_compressibility = c_w_gas_free_component + gas_liberation_component

    return max(0.0, water_compressibility)  # Ensure non-negative compressibility


def compute_live_oil_density(
    api_gravity: float,
    gas_gravity: float,
    gas_to_oil_ratio: float,
    formation_volume_factor: float,
) -> float:
    """
    Estimates live oil density at reservoir conditions at the current pressure
    and temperature, considering dissolved gas and oil compressibility.

    Based on:
        - Stock tank oil density from API gravity.
        - Contribution of dissolved gas mass.
        - Volume expansion/compression via FVF.

    :param pressure: Reservoir pressure (psi)
    :param bubble_point_pressure: Bubble point pressure (psi)
    :param api_gravity: Oil API gravity [°API]
    :param gas_gravity: Gas specific gravity (relative to air)
    :param oil_compressibility: Oil compressibility (psi⁻¹)
    :param gas_to_oil_ratio: Gas-to-oil ratio at current pressure (SCF/STB)
    :param formation_volume_factor: Oil formation volume factor at current pressure (bbl/STB)
    :return: Live oil density (lb/ft³) at reservoir conditions.
    """
    # Convert API to stock tank oil density (lb/ft³)
    stock_tank_oil_density_lb_per_ft3 = (
        141.5 / (api_gravity + 131.5)
    ) * STANDARD_WATER_DENSITY_IMPERIAL

    # Mass of oil per STB (lb)
    mass_stock_tank_oil = stock_tank_oil_density_lb_per_ft3 / FT3_TO_STB

    # Mass of dissolved gas per STB (lb)
    # Approx: 1 scf = gas_gravity * (molecular weight of air) / 379.49 lb
    gas_mass_per_scf = (gas_gravity * MOLECULAR_WEIGHT_AIR) / SCF_PER_POUND_MOLE
    mass_dissolved_gas = gas_to_oil_ratio * gas_mass_per_scf

    # Total mass and volume
    total_mass_lb_per_stb = mass_stock_tank_oil + mass_dissolved_gas
    # print(formation_volume_factor)
    total_volume_ft3_per_stb = formation_volume_factor * BBL_TO_FT3

    # Live oil density in lb/ft³
    live_oil_density_lb_per_ft3 = total_mass_lb_per_stb / total_volume_ft3_per_stb
    return live_oil_density_lb_per_ft3


def compute_standard_water_density(salinity: float) -> float:
    """
    Computes the density of brine at standard conditions (60 F, 14.7 psia)
    using McCain's correlation.

    Correlation (McCain, "Properties of Petroleum Fluids", 3rd Ed., Eq. 18.4):

        rho_w_std (lb/ft^3) = 62.368 + 0.438603 * S_wt% + 1.60074E-6 * S_wt%^2

    Where S_wt% is salinity in weight percent.

    :param salinity: Salinity in parts per million (ppm).
    :return: Standard water density (rho_w_std) in lb/ft³.
    """
    if salinity < 0:
        raise ValueError("Salinity cannot be negative.")

    # Convert salinity from ppm to weight percent (wt%)
    # Given that 10,000 ppm = 1 wt%
    salinity_wt_percent = salinity / 10000.0

    # McCain's correlation for standard brine density
    water_standard_density = (
        STANDARD_WATER_DENSITY_IMPERIAL
        + (0.438603 * salinity_wt_percent)
        + (1.60074e-6 * salinity_wt_percent**2)
    )
    return water_standard_density


def compute_water_density(
    gas_gravity: float = 0.0,
    salinity: float = 0.0,
    gas_solubility_in_water: float = 0.0,
    gas_free_water_formation_volume_factor: float = 0.0,
) -> float:
    """
    Calculates the live water (brine) density at reservoir conditions
    using McCain's correlations.

    The correlation is based on the mass balance:

        rho_w = (Mass of standard water + Mass of dissolved gas) / Volume of live water

        rho_w (lb/ft^3) = (rho_w_std (lb/ft^3) + Rsw (scf/STB) * gas_gravity * 0.01359) / Bw (res bbl/STB)

    where:
    - rho_w_std is the standard water density at 60 F and 14.7 psia (62.37 lb/ft^3 for pure water).
    - Rsw is the gas solubility in water at current pressure and temperature (scf/STB).
    - gas_gravity is the specific gravity of the dissolved gas (relative to air).
    - Bw is the water formation volume factor at current pressure and temperature (res bbl/STB).

    :param salinity: Salinity in parts per million (ppm). Defaults to 0.0 for pure water.
    :param gas_gravity: Specific gravity of dissolved gas (relative to air). Defaults to 0.0 if no gas.
    :param gas_solubility_in_water: Gas solubility in water (Rsw) in (SCF/STB) at current pressure and temperature.
    :param gas_free_water_formation_volume_factor: Gas-free water formation volume factor (Bw)(bbl/STB) at current pressure and temperature.
    :return: Live water density (lb/ft³) at reservoir conditions.
    """
    if salinity < 0 or gas_gravity < 0:
        raise ValueError("Salinity and gas gravity must be non-negative.")

    standard_water_density_in_lb_per_ft3 = compute_standard_water_density(salinity)

    # For density calculation using the formula, Bw in the denominator is the *actual*
    # Bw (live water FVF). For water, dissolved gas usually has a very minor effect
    # on Bw (which is typically close to 1.0). The `calculate_bw_gas_free_mccain`
    # handles pressure and temperature effects.
    # If the Rsw term is significant, it's captured in the numerator's mass.
    # So, bw_actual = bw_gas_free is a common approximation here, or if Rsw*Bg effect on volume is added.
    # For simplicity and given the formula structure, we use the bw_gas_free as the Bw in denominator.

    if gas_free_water_formation_volume_factor <= 0:
        raise ValueError(
            "Calculated water formation volume factor (Bw) is non-positive, cannot calculate density."
        )

    # Calculate Live Water Density (Imperial units first)
    # Mass of standard water per STB (volume of STB is 1 STB, density lb/ft3 * 5.615 ft3/bbl)
    standard_mass_water_in_lb_per_stb = (
        standard_water_density_in_lb_per_ft3 * BBL_TO_FT3
    )  # lb/STB

    # Mass of dissolved gas per STB
    # Note: The 0.01359 factor in the simple formula often implicitly converts scf to bbl for the gas mass contribution.
    # Let's use the explicit conversion:
    # Mass gas (lb) = Rsw (scf) * Density of gas at std cond (lb/scf)
    # Density of gas at std cond (lb/scf) = gas_gravity * (28.96 lb/lb-mol_air / 379.4 scf/lb-mol_ideal_gas) = gas_gravity * 0.0763 lb/scf
    mass_of_dissolved_gas_in_lb_per_stb = (
        gas_solubility_in_water * gas_gravity * MOLECULAR_WEIGHT_AIR
    ) / SCF_PER_POUND_MOLE  # lb_mass_gas/STB

    # Total mass of live water (and dissolved gas) per STB
    total_mass_in_lb_per_stb = (
        standard_mass_water_in_lb_per_stb + mass_of_dissolved_gas_in_lb_per_stb
    )

    # Volume of live water at reservoir conditions (ft^3 per STB)
    volume_of_live_water_in_ft3_per_stb = (
        gas_free_water_formation_volume_factor * BBL_TO_FT3
    )  # res bbl/STB * ft^3/bbl = ft^3/STB
    live_water_density_in_lb_per_ft3 = (
        total_mass_in_lb_per_stb / volume_of_live_water_in_ft3_per_stb
    )  # lb/ft^3

    # Ensure density is non-negative
    return max(0.0, live_water_density_in_lb_per_ft3)


def compute_gas_to_oil_ratio_standing(
    pressure: float,
    oil_api_gravity: float,
    gas_gravity: float,
) -> float:
    """
    Standing correlation to compute Rs (solution GOR) in scf/STB.

    This assumes the oil is at or below bubble point pressure, and temperature
    is not used (approximation based only on pressure, API gravity, and gas gravity).

    Estimated using the Standing correlation for solution gas-oil ratio (Rs):

        Rs = gas_gravity * [ (P / 18.2 + 1.4) * 10^(0.0125 * API) ]^(1 / 1.2048)

    where:
    - P is the pressure in psia.
    - T_F is the temperature in degrees Fahrenheit.
    - API is the API gravity of the oil.
    - gas_gravity is the specific gravity of the gas (relative to air).

    This correlation is typically used for light oils and may not be accurate
    for heavy oils (API < 10) or high pressures.

    :param pressure: Pressure (psi)
    :param oil_api_gravity: API gravity of the oil in degrees API.
    :param gas_gravity: Specific gravity of the gas (relative to air).
    :return: Solution gas-oil ratio (Rs) in (SCF/STB).
    """
    if oil_api_gravity < 0 or gas_gravity < 0 or pressure < 0:
        raise ValueError("All inputs must be non-negative for Rs calculation.")

    if oil_api_gravity < 10:
        raise ValueError(
            "API gravity must be greater than or equal to 10 for Standing's correlation."
        )

    gor = gas_gravity * (
        (pressure / 18.2 + 1.4) * 10 ** (0.0125 * oil_api_gravity)
    ) ** (1 / 1.2048)
    return gor


def estimate_bubble_point_pressure_standing(
    oil_api_gravity: float,
    gas_gravity: float,
    observed_gas_to_oil_ratio: float,
) -> float:
    """
    Estimate bubble point pressure (Pb) using Standing's correlation
    given observed Rs and known oil API gravity and gas gravity.

    THIS FUNCTION ESTIMATES THE BUBBLE POINT PRESSURE AND IS NOT A DIRECT
    MEASUREMENT. It is only valid for light oils (API > 10) and may not be accurate
    for heavy oils or high pressures.

    This assumes the oil is at or below bubble point pressure, and temperature
    is not used (approximation based only on pressure, API gravity, and gas gravity).

    The bubble point pressure is estimated by solving the equation for Rs
    using the Standing correlation:

        Rs = gas_gravity * [ (P / 18.2 + 1.4) * 10^(0.0125 * API) ]^(1 / 1.2048)

    where:
    - P is the pressure in psia.
    - T_F is the temperature in degrees Fahrenheit.
    - API is the API gravity of the oil.

    :param oil_api_gravity: API gravity of the oil in degrees API.
    :param gas_gravity: Specific gravity of the gas (relative to air).
    :param observed_gas_to_oil_ratio: Observed solution gas-oil ratio (Rs) in SCF/STB.
    :return: Estimated bubble point pressure (Pb) (psi).
    """

    def residual(pressure: float) -> float:
        gor = compute_gas_to_oil_ratio_standing(
            pressure=pressure,
            oil_api_gravity=oil_api_gravity,
            gas_gravity=gas_gravity,
        )
        return gor - observed_gas_to_oil_ratio

    solver = root_scalar(residual, bracket=[14.696, 10000], method="brentq")

    if not solver.converged:
        raise RuntimeError("Could not converge to a bubble point pressure.")

    bubble_point_pressure = solver.root
    return bubble_point_pressure
