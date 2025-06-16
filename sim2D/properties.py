import functools
import typing
import numpy as np
from CoolProp.CoolProp import PropsSI

from sim2D.typing import TwoDimensionalGrid, InjectionFluid, FluidMiscibility

mD_to_m2 = 9.869e-16
"""Conversion factor from millidarcies to square meters"""

DEFAULT_DISPLACED_FLUID = (
    "n-Octane"  # Heavy oil, can be changed to any fluid supported by CoolProp
)
"""Default displaced fluid for simulations, can be changed to any fluid supported by CoolProp"""


MIN_VALID_PRESSURE = 1  # 1 Pa
MAX_VALID_PRESSURE = 1e8  # 100 MPa (to be adjusted based on fluid model limits)
MIN_VALID_TEMPERATURE = 1e-3  # 0.001 K (to avoid division by zero in some properties)
MAX_VALID_TEMPERATURE = 1000  # 1000 K (to be adjusted based on fluid model limits)


def is_CoolProp_supported_fluid(fluid: str) -> bool:
    """
    Check if the fluid is supported by CoolProp.

    :param fluid: Fluid name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: True if the fluid is supported, False otherwise.
    """
    return PropsSI("D", "T", 300, "P", 101325, fluid) is not None


def clip_pressure(pressure: float) -> float:
    """
    Clip pressure to a minimum of 1e-6 Pa and a maximum of 1e9 Pa
    to avoid non-physical conditions.

    This

    :param pressure: Pressure in Pascals (Pa)
    :return: Clipped pressure value
    """
    return np.clip(pressure, MIN_VALID_PRESSURE, MAX_VALID_PRESSURE)


def clip_temperature(temperature: float) -> float:
    """
    Clip temperature to a minimum of 1e-3 K and a maximum of 1000 K
    to avoid non-physical conditions.

    :param temperature: Temperature in Kelvin (K)
    :return: Clipped temperature value
    """
    return np.clip(temperature, MIN_VALID_TEMPERATURE, MAX_VALID_TEMPERATURE)


@functools.lru_cache(maxsize=128)
def get_fluid_density(pressure: float, temperature: float, fluid: str) -> float:
    """
    Get fluid density from EOS using CoolProp.

    :param pressure: Pressure in Pascals (Pa)
    :param temperature: Temperature in Kelvin (K)
    :param fluid: Fluid name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Density in kg/m³
    """
    return PropsSI(
        "D", "P", clip_pressure(pressure), "T", clip_temperature(temperature), fluid
    )


@functools.lru_cache(maxsize=128)
def get_fluid_viscosity(pressure: float, temperature: float, fluid: str) -> float:
    """
    Get fluid dynamic viscosity from EOS using CoolProp.

    :param pressure: Pressure in Pascals (Pa)
    :param temperature: Temperature in Kelvin (K)
    :param fluid: Fluid name (must be supported by CoolProp, e.g., "CO2", "Water")
    :return: Viscosity in Pascal-seconds (Pa·s)
    """
    return PropsSI(
        "V", "P", clip_pressure(pressure), "T", clip_temperature(temperature), fluid
    )


@functools.lru_cache(maxsize=128)
def get_fluid_compressibility_factor(
    pressure: float, temperature: float, fluid: str
) -> float:
    """
    Get fluid compressibility factor Z from EOS using CoolProp.

    :param pressure: Pressure in Pascals (Pa)
    :param temperature: Temperature in Kelvin (K)
    :param fluid: Fluid name (must be supported by CoolProp, e.g., "CO2", "Methane")
    :return: Compressibility factor Z (dimensionless)
    """
    return PropsSI(
        "Z", "P", clip_pressure(pressure), "T", clip_temperature(temperature), fluid
    )


@functools.lru_cache(maxsize=128)
def get_adaptive_delta_p(pressure: float) -> float:
    """Determine a suitable delta-P for numerical differentiation using adaptive finite difference"""
    eps = np.finfo(float).eps
    return max(np.sqrt(eps) * max(pressure, 1.0), 1e-3)


@functools.lru_cache(maxsize=256)
def compute_fluid_compressibility(
    pressure: float,
    temperature: float,
    fluid: InjectionFluid,
    delta_p: typing.Optional[float] = None,
) -> float:
    """
    Computes the isothermal compressibility of a fluid at a given pressure and temperature.

    Compressibility is defined as:

        C_f = -(1/Rho) * (dRho/dP) at constant temperature

    :param pressure: Pressure in Pascals (Pa)
    :param temperature: Temperature in Kelvin (K)
    :param fluid: Fluid name supported by CoolProp (default: 'CO2')
    :param delta_p: Small pressure step for numerical derivative (Pa)
    :return: Fluid compressibility in 1/Pa
    """
    delta_p = delta_p or get_adaptive_delta_p(pressure)
    density = get_fluid_density(pressure, temperature, fluid)
    density_plus = get_fluid_density(pressure + delta_p, temperature, fluid)
    dRho_dP = (density_plus - density) / delta_p
    compressibility = -dRho_dP / density
    return compressibility


@functools.lru_cache(maxsize=256)
def compute_miscible_viscosity(
    injected_fluid_saturation: float,
    injected_fluid_viscosity: float,
    displaced_fluid_viscosity: float,
    miscibility: FluidMiscibility = "logarithmic",
) -> float:
    """
    Computes the effective viscosity of a mixture of two miscible fluids based on their saturations and viscosities.

    The effective viscosity is calculated using one of the following methods:
        - Logarithmic: μ_eff = μ_g^S_g * μ_o^(1 - S_g)
        - Harmonic: 1/μ_eff = (S_g/μ_g) + ((1 - S_g)/μ_o)
        - Linear: μ_eff = S_g * μ_g + (1 - S_g) * μ_o

    where:
        - S_g is the saturation of the injected fluid (gas)
        - μ_g is the viscosity of the injected fluid (gas)
        - μ_o is the viscosity of the displaced fluid (oil)

    :param injected_fluid_saturation: Saturation of the injected fluid (fraction, 0 to 1)
    :param injected_fluid_viscosity: Viscosity of the injected fluid (Pa·s)
    :param displaced_fluid_viscosity: Viscosity of the displaced fluid (Pa·s)
    :param miscibility: Method for mixing viscosities ('logarithmic', 'harmonic', 'linear')
    :return: Effective viscosity of the mixture (Pa·s)
    """
    # Clip injected fluid saturation to avoid numerical issues
    injected_fluid_saturation = np.clip(injected_fluid_saturation, 1e-8, 1 - 1e-8)
    displaced_fluid_saturation = 1 - injected_fluid_saturation
    if miscibility == "logarithmic":
        return (injected_fluid_viscosity**injected_fluid_saturation) * (
            displaced_fluid_viscosity**displaced_fluid_saturation
        )

    elif miscibility == "harmonic":
        # Adjust minimum viscosities to avoid division by zero
        injected_fluid_viscosity = max(injected_fluid_viscosity, 1e-8)
        displaced_fluid_viscosity = max(displaced_fluid_viscosity, 1e-8)
        return 1 / (
            (injected_fluid_saturation / injected_fluid_viscosity)
            + (displaced_fluid_saturation / displaced_fluid_viscosity)
        )

    elif miscibility == "linear":
        return (injected_fluid_saturation * injected_fluid_viscosity) + (
            displaced_fluid_saturation * displaced_fluid_viscosity
        )
    raise ValueError("Unknown viscosity mixing method.")


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
    permeability: float,
    porosity: float,
    viscosity: float,
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

        D = (k / (φ * μ * C_t)) * (Δt / Δx²)

    where:
        - k is the permeability in mD (converted internally to m²),
        - φ is the porosity (fraction),
        - μ is the fluid viscosity (Pa·s),
        - C_t is the total compressibility (1/Pa),
        - Δt is the time step size (s),
        - Δx is the grid block size (m)

    :param permeability: Rock permeability in millidarcies (mD)
    :param porosity: Rock porosity as a fraction (e.g., 0.2)
    :param viscosity: Fluid viscosity in Pascal-seconds (Pa·s)
    :param total_compressibility: Total compressibility of the system (1/Pa)
    :param time_step_size: Time step size (seconds)
    :param cell_size: Size of the grid block (meters)
    :return: Diffusion number (dimensionless)
    """
    diffusion_number = (
        (permeability * mD_to_m2) / (porosity * viscosity * total_compressibility)
    ) * (time_step_size / cell_size**2)

    return diffusion_number


@functools.lru_cache(maxsize=128)
def compute_fractional_flow(
    injected_fluid_saturation: float,
    injected_fluid_viscosity: float,
    displaced_fluid_viscosity: float,
) -> float:
    """
    Calculates the fractional flow of the injected phase using the viscosity ratio (mobility ratio)
    based on a simplified Buckley-Leverett formulation.

    This model assumes negligible capillary pressure and relative permeabilities proportional to saturations.
    It is applicable for miscible or immiscible displacement of one fluid by another (e.g., CO₂-oil, water-oil, gas-liquid).

    :param injected_fluid_saturation: Saturation of the injected phase (fraction between 0 and 1)
    :param injected_fluid_viscosity: Viscosity of the injected phase in Pa·s
    :param displaced_fluid_viscosity: Viscosity of the displaced phase in Pa·s
    :return: Fractional flow of the injected phase (array of values between 0 and 1)
    """
    mobility_ratio = injected_fluid_viscosity / displaced_fluid_viscosity
    saturation = np.clip(
        injected_fluid_saturation, 1e-8, 1 - 1e-8
    )  # avoid zero division
    return 1.0 / (1.0 + (mobility_ratio * ((1 - saturation) / saturation)))


def compute_fractional_flow_corey(
    injected_fluid_saturation: float,
    injected_fluid_viscosity: float,
    displaced_fluid_viscosity: float,
    injected_fluid_residual_saturation: float,
    displaced_fluid_residual_saturation: float,
    injected_fluid_corey_exponent: float = 2.0,
    displaced_fluid_corey_exponent: float = 2.0,
) -> float:
    """
    Calculates the fractional flow of the injected fluid using Corey-type relative permeability models.

    This function models multiphase flow (e.g., gas-oil, water-oil) by incorporating residual saturations
    and nonlinear saturation-permeability relationships using the Corey model.

    :param injected_fluid_saturation: Saturation of the injected phase (fraction between 0 and 1)
    :param injected_fluid_viscosity: Dynamic viscosity of the injected phase (Pa·s)
    :param displaced_fluid_viscosity: Dynamic viscosity of the displaced phase (Pa·s)
    :param injected_fluid_residual_saturation: Residual saturation of the injected phase (e.g., irreducible water)
    :param displaced_fluid_residual_saturation: Residual saturation of the displaced phase (e.g., residual oil)
    :param injected_fluid_corey_exponent: Corey exponent for the injected phase (controls curvature)
    :param displaced_fluid_corey_exponent: Corey exponent for the displaced phase (controls curvature)
    :return: Fractional flow of the injected phase (array of values between 0 and 1)
    """
    # Compute normalized (effective) saturation for Corey correlations
    effective_saturation = (
        injected_fluid_saturation - injected_fluid_residual_saturation
    ) / (1.0 - injected_fluid_residual_saturation - displaced_fluid_residual_saturation)
    effective_saturation = np.clip(effective_saturation, 1e-6, 1 - 1e-6)

    # Relative permeabilities using Corey correlations
    relative_perm_injected = effective_saturation**injected_fluid_corey_exponent
    relative_perm_displaced = (
        1.0 - effective_saturation
    ) ** displaced_fluid_corey_exponent

    # Phase mobilities
    mobility_injected = relative_perm_injected / injected_fluid_viscosity
    mobility_displaced = relative_perm_displaced / displaced_fluid_viscosity

    # Fractional flow calculation
    injected_fluid_fractional_flow = mobility_injected / (
        mobility_injected + mobility_displaced
    )
    return injected_fluid_fractional_flow


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


def compute_harmonic_transmissibility(
    i1: int,
    j1: int,
    i2: int,
    j2: int,
    spacing: float,
    transmissibility_grid: TwoDimensionalGrid,
) -> float:
    """
    Computes harmonic average transmissibility between two cells.

    If both transmissibilities are zero, returns zero to avoid division by zero.

    :param i1: Row index of first cell
    :param j1: Column index of first cell
    :param i2: Row index of second cell
    :param j2: Column index of second cell
    """
    T1 = transmissibility_grid[i1, j1]
    T2 = transmissibility_grid[i2, j2]
    T_harmonic = compute_harmonic_mean(T1, T2)
    return T_harmonic / spacing**2
