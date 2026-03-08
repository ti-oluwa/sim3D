"""Core well calculations and utilities."""

import logging
import typing

import attrs
import numba
import numpy as np

from bores.constants import c
from bores.correlations.arrays import (
    compute_gas_compressibility as compute_gas_compressibility_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_compressibility_factor as compute_gas_compressibility_factor_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_density as compute_gas_density_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_formation_volume_factor as compute_gas_formation_volume_factor_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_free_water_formation_volume_factor as compute_gas_free_water_formation_volume_factor_vectorized,
)
from bores.correlations.arrays import (
    compute_gas_viscosity as compute_gas_viscosity_vectorized,
)
from bores.correlations.arrays import (
    compute_water_compressibility as compute_water_compressibility_vectorized,
)
from bores.correlations.arrays import (
    compute_water_density as compute_water_density_vectorized,
)
from bores.correlations.arrays import (
    compute_water_formation_volume_factor as compute_water_formation_volume_factor_vectorized,
)
from bores.correlations.arrays import (
    compute_water_viscosity as compute_water_viscosity_vectorized,
)
from bores.correlations.core import (
    compute_gas_compressibility,
    compute_gas_compressibility_factor,
    compute_gas_density,
    compute_gas_formation_volume_factor,
    compute_gas_free_water_formation_volume_factor,
    compute_gas_viscosity,
    compute_water_compressibility,
    compute_water_density,
    compute_water_formation_volume_factor,
    compute_water_viscosity,
    fahrenheit_to_rankine,
)
from bores.errors import ComputationError, ValidationError
from bores.serialization import Serializable
from bores.tables.pseudo_pressure import (
    GasPseudoPressureTable,
    build_gas_pseudo_pressure_table,
)
from bores.tables.pvt import PVTTables
from bores.types import (
    FloatOrArray,
    FluidPhase,
    Orientation,
    ThreeDimensions,
    TwoDimensions,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InjectedFluid",
    "ProducedFluid",
    "WellFluid",
    "compute_2D_effective_drainage_radius",
    "compute_3D_effective_drainage_radius",
    "compute_gas_well_rate",
    "compute_oil_well_rate",
    "compute_required_bhp_for_gas_rate",
    "compute_required_bhp_for_oil_rate",
    "compute_well_index",
]


@numba.njit(cache=True)
def compute_well_index(
    permeability: float,
    interval_thickness: float,
    wellbore_radius: float,
    effective_drainage_radius: float,
    skin_factor: float = 0.0,
) -> float:
    """
    Compute the well index for a given well using the Peaceman equation.

    The well index is a measure of the productivity of a well, defined as the ratio of the
    well flow rate to the pressure drop across the well.

    The formula for the well index is:

    W = (k * h) / (ln(re/rw) + s)

    where:
        - W is the well index (md*ft)
        - k is the absolute permeability of the reservoir rock (mD)
        - h is the thickness of the reservoir interval (ft)
        - re is the effective drainage radius (ft)
        - rw is the wellbore radius (ft)
        - s is the skin factor (dimensionless, default is 0)

    :param permeability: Absolute permeability of the reservoir rock (mD).
    :param interval_thickness: Thickness of the reservoir interval (ft).
    :param wellbore_radius: Radius of the wellbore (ft).
    :param effective_drainage_radius: Effective drainage radius (ft).
    :param skin_factor: Skin factor for the well (dimensionless, default is 0).
    :return: The well index in (mD*ft).
    """
    return (permeability * interval_thickness) / (
        np.log(effective_drainage_radius / wellbore_radius) + skin_factor
    )


@numba.njit(cache=True)
def compute_3D_effective_drainage_radius(
    interval_thickness: ThreeDimensions,
    permeability: ThreeDimensions,
    well_orientation: Orientation,
) -> float:
    """
    Compute the effective drainage radius for a well in a 3D reservoir model
    using Peaceman's (1983) anisotropic effective drainage radius formula.

    Peaceman's formula for a well aligned along a given axis uses the two
    grid dimensions and permeabilities *perpendicular* to the wellbore.
    For a well in the Z-direction (standard vertical well)::

        r_o = 0.28 * sqrt[ (ky/kx)^0.5 * dx^2 + (kx/ky)^0.5 * dy^2 ]
                           -----------------------------------------
                                  (ky/kx)^0.25 + (kx/ky)^0.25

    The same pattern applies by cyclic permutation for X- and Y-oriented
    (horizontal) wells, substituting the two directions perpendicular to
    the wellbore axis in each case.

    For the isotropic case (kx = ky = k) this reduces to::

        r_o = 0.28 * sqrt(dx^2 + dy^2) / 2^0.5
            = 0.198 * sqrt(dx^2 + dy^2)

    which matches the standard Peaceman result for uniform square grids
    (r_o = 0.2 * dx when dx = dy).

    If either perpendicular permeability is zero (e.g. a tight layer or shale
    streak), the well has no drainage capacity in that plane and 0.0 is
    returned.  The caller should treat a zero return value as an indicator
    that no well index should be assigned to this interval.

    :param interval_thickness: Tuple of cell dimensions (dx, dy, dz) in ft.
    :param permeability: Tuple of permeabilities (kx, ky, kz) in mD.
    :param well_orientation: Wellbore axis — Orientation.X for a horizontal
        well along x, Orientation.Y for a horizontal well along y, or
        Orientation.Z for a standard vertical well.
    :return: Effective drainage (Peaceman) radius in ft, or 0.0 if either
        perpendicular permeability is zero.
    """
    dx, dy, dz = interval_thickness[0], interval_thickness[1], interval_thickness[2]
    kx, ky, kz = permeability[0], permeability[1], permeability[2]

    if well_orientation == Orientation.X:
        # Wellbore along x — perpendicular plane is y-z
        if ky <= 0.0 or kz <= 0.0:
            return 0.0
        r1, r2 = ky / kz, kz / ky
        numerator = np.sqrt(r1) * dy**2 + np.sqrt(r2) * dz**2
        denominator = r1**0.25 + r2**0.25
    elif well_orientation == Orientation.Y:
        # Wellbore along y — perpendicular plane is x-z
        if kx <= 0.0 or kz <= 0.0:
            return 0.0
        r1, r2 = kx / kz, kz / kx
        numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dz**2
        denominator = r1**0.25 + r2**0.25
    elif well_orientation == Orientation.Z:
        # Wellbore along z — perpendicular plane is x-y (standard vertical well)
        if kx <= 0.0 or ky <= 0.0:
            return 0.0
        r1, r2 = ky / kx, kx / ky
        numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dy**2
        denominator = r1**0.25 + r2**0.25
    else:
        raise ValidationError(f"Invalid well orientation {well_orientation}")

    return 0.28 * np.sqrt(numerator / denominator)


@numba.njit(cache=True)
def compute_2D_effective_drainage_radius(
    interval_thickness: TwoDimensions,
    permeability: TwoDimensions,
) -> float:
    """
    Compute the effective drainage radius for a well in a 2D reservoir model
    using Peaceman's (1983) anisotropic formula.

    In a 2D x-y model the wellbore is always perpendicular to the grid plane
    (i.e. implicitly Z-oriented), so there is a single drainage radius
    expression that uses both in-plane grid dimensions and permeabilities::

        r_o = 0.28 * sqrt[ (ky/kx)^0.5 * dx^2 + (kx/ky)^0.5 * dy^2 ]
                           -----------------------------------------
                                  (ky/kx)^0.25 + (kx/ky)^0.25

    This is identical to the Z-orientation case of
    :func:`compute_3D_effective_drainage_radius`.

    For the isotropic case (kx = ky) this reduces to::

        r_o = 0.28 * sqrt(dx^2 + dy^2) / sqrt(2)

    which gives r_o = 0.2 * dx for a uniform square grid (dx = dy),
    matching the classic Peaceman result.

    If either permeability is zero or negative (e.g. a tight or shale cell),
    the well has no drainage capacity and 0.0 is returned.  The caller should
    treat a zero return value as an indicator that no well index should be
    assigned to this interval.

    :param interval_thickness: Tuple of cell dimensions (dx, dy) in ft.
    :param permeability: Tuple of permeabilities (kx, ky) in mD.
    :return: Effective drainage (Peaceman) radius in ft, or 0.0 if either
        permeability is zero or negative.
    """
    kx, ky = permeability[0], permeability[1]

    if kx <= 0.0 or ky <= 0.0:
        return 0.0

    dx, dy = interval_thickness[0], interval_thickness[1]

    r1, r2 = ky / kx, kx / ky
    numerator = np.sqrt(r1) * dx**2 + np.sqrt(r2) * dy**2
    denominator = r1**0.25 + r2**0.25
    return 0.28 * np.sqrt(numerator / denominator)


@numba.njit(cache=True)
def compute_oil_well_rate(
    well_index: float,
    pressure: float,
    bottom_hole_pressure: float,
    phase_mobility: float,
    fluid_compressibility: typing.Optional[float] = None,
    incompressibility_threshold: float = 1e-6,
) -> float:
    """
    Compute the well rate at reservoir conditions using the Darcy well equation.

    Returns the rate at **reservoir conditions** (rb/day).  The caller is
    responsible for converting to stock-tank conditions by dividing by the
    formation volume factor (Bo, Bw, or Bg), consistent with the rest of the
    simulator.

    Because `phase_mobility = kr / mu` (no FVF term), and because Bo is
    applied *downstream*, the linear Darcy formula is the correct formulation
    for black-oil phases.  Bo already encodes fluid compressibility relative to
    surface conditions via `co = -(1/Bo) * (dBo/dP)`, so applying an
    additional exponential compressibility correction on top of a downstream Bo
    division would double-count the same physical effect.

    The slightly-compressible exponential correction is retained only for
    phases where no FVF is applied downstream (e.g. a standalone single-phase
    water or gas module that does not track Bw/Bg explicitly), and only when
    `c * dP` is large enough to produce a materially different result from
    the linear formula (> 1%) but still within the range where the exponential
    density model is physically valid (c * dP <= 0.7).

    Sign convention:
        - Negative rate indicates production (BHP < reservoir pressure).
        - Positive rate  indicates injection  (BHP > reservoir pressure).

    Formula (linear / incompressible)::

        Q = 7.08e-3 * W * M * (P_bhp - P)

    Formula (slightly compressible, exponential)::

        Q = 7.08e-3 * W * M * [exp(c * dP) - 1] / c

    where dP = P_bhp - P.

    :param well_index: Well index (mD*ft).
    :param pressure: Reservoir cell pressure at the perforation interval (psi).
    :param bottom_hole_pressure: Well bottom-hole pressure (psi).
    :param phase_mobility: Phase relative mobility kr/mu (md/cP).
        Must not include the FVF term; Bo/Bw/Bg is applied by the caller.
    :param fluid_compressibility: Fluid compressibility (psi^-1).  When
        provided and above `incompressibility_threshold`, the exponential
        correction is evaluated.  Pass None (default) to always use the
        linear formula, which is correct for black-oil phases.
    :param incompressibility_threshold: Minimum compressibility (psi^-1) below
        which the fluid is treated as incompressible and the linear formula is
        used regardless.  Default 1e-6 psi^-1.
    :return: Well rate at reservoir conditions (bbl/day).  Negative for
        production, positive for injection.
    """
    if well_index <= 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility <= 0:
        raise ValidationError("Phase mobility must be a positive value.")

    pressure_difference = bottom_hole_pressure - pressure
    is_compressible = (
        fluid_compressibility is not None
        and fluid_compressibility >= incompressibility_threshold
    )
    if is_compressible:
        argument = fluid_compressibility * pressure_difference  # type: ignore
        # Apply exponential correction only when c*dP is large enough to
        # produce a >1% deviation from the linear result, but small enough
        # that the slightly-compressible exponential model remains valid.
        # For black-oil oil/water with Bo tracked downstream this branch is
        # essentially never entered (typical c*dP << 0.01); it is provided
        # for single-phase modules that do not apply a downstream FVF.
        #
        # c*dP < 1e-4 : exp(x)-1 ~ x, numerically identical to linear.
        # c*dP > 0.7  : model has broken down at this drawdown magnitude;
        #               fall back to linear; Bo correction handles the rest.
        if 1e-4 < abs(argument) <= 0.7:
            return (
                7.08e-3
                * well_index
                * phase_mobility
                * (np.exp(argument) - 1.0)
                / fluid_compressibility
            )

    return 7.08e-3 * well_index * phase_mobility * pressure_difference


@numba.njit(cache=True)
def compute_required_bhp_for_oil_rate(
    target_rate: float,
    well_index: float,
    pressure: float,
    phase_mobility: float,
    fluid_compressibility: typing.Optional[float] = None,
    incompressibility_threshold: float = 1e-6,
) -> float:
    """
    Compute the bottom-hole pressure required to achieve a target well rate.

    This is the exact algebraic inverse of `compute_oil_well_rate`.
    The same FVF and compressibility convention applies: `target_rate` must
    be at reservoir conditions (rb/day), and `phase_mobility = kr / mu`
    with no FVF term.

    Sign convention:
        - Negative `target_rate` indicates production (returned BHP < reservoir pressure).
        - Positive `target_rate`  indicates injection  (returned BHP > reservoir pressure).

    Inverse formula (linear / incompressible)::

        P_bhp = P + Q / (7.08e-3 * W * M)

    Inverse formula (slightly compressible, exponential)::

        P_bhp = P + ln(Q * c / (7.08e-3 * W * M) + 1) / c

    The logarithmic inverse is the algebraically exact inverse of the
    exponential forward formula, not an approximation.  The same c * dP
    validity window used in `compute_oil_well_rate` is enforced here
    by checking that the `ln` argument is positive and within a physically
    meaningful range before applying it, guaranteeing that the two functions
    are exact inverses of each other within the same operating regime.

    :param target_rate: Target well rate at reservoir conditions (rb/day).
        Negative for production, positive for injection.
    :param well_index: Well index (mD*ft).
    :param pressure: Reservoir cell pressure at the perforation interval (psi).
    :param phase_mobility: Phase relative mobility kr/mu (md/cP).
        Must not include the FVF term.
    :param fluid_compressibility: Fluid compressibility (psi^-1).  When
        provided and above `incompressibility_threshold`, the logarithmic
        inverse is evaluated.  Pass None to always use the linear inverse.
    :param incompressibility_threshold: Minimum compressibility (psi^-1) below
        which the fluid is treated as incompressible.  Default 1e-6 psi^-1.
    :return: Required bottom-hole pressure (psi).
    """
    if well_index <= 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility <= 0:
        raise ValidationError("Phase mobility must be a positive value.")

    denominator = 7.08e-3 * well_index * phase_mobility

    is_compressible = (
        fluid_compressibility is not None
        and fluid_compressibility >= incompressibility_threshold
    )
    if is_compressible:
        # Exact inverse of Q = (7.08e-3 * W * M / c) * [exp(c * dP) - 1]:
        #   c * dP = ln(Q * c / (7.08e-3 * W * M) + 1)
        #   dP     = ln(argument) / c
        argument = target_rate * fluid_compressibility / denominator + 1.0  # type: ignore

        # argument must be > 0 for ln() to be defined.
        # argument <= 0 means the target rate exceeds what the
        # compressibility-driven supply can deliver at this pressure;
        # fall through to the linear formula as a conservative limit.
        #
        # argument > 1e10 implies a c*dP outside the valid range of the
        # slightly-compressible model; fall through to linear.
        if 0.0 < argument <= 1e10:
            delta_p = np.log(argument) / fluid_compressibility
            # Enforce the same c*dP <= 0.7 validity window as the forward
            # function so the two are guaranteed to be exact inverses within
            # the same operating regime.
            if abs(fluid_compressibility * delta_p) <= 0.7:
                return float(pressure + delta_p)

    # Linear / incompressible inverse: dP = Q / (7.08e-3 * W * M)
    return float(pressure + target_rate / denominator)


def compute_gas_well_rate(
    well_index: float,
    pressure: float,
    temperature: float,
    bottom_hole_pressure: float,
    phase_mobility: float,
    average_compressibility_factor: float = 1.0,
    use_pseudo_pressure: bool = True,
    pseudo_pressure_table: typing.Optional[GasPseudoPressureTable] = None,
    formation_volume_factor: typing.Optional[float] = None,
    gas_gravity: typing.Optional[float] = None,
) -> float:
    """
    Compute the gas well rate at reservoir conditions (ft³/day).

    The well equation is first evaluated at surface conditions (SCF/day) via
    the (Tsc/Psc) prefactor, then multiplied by Bg (ft³/SCF) to convert to
    reservoir conditions, consistent with the oil/water rate functions which
    return reservoir bbl/day.  The caller divides by Bg downstream to obtain
    surface SCF/day, following the same pattern used throughout the simulator.

    Two formulations are supported:

    Pseudo-pressure (recommended, valid over the full pressure range):

    ```
    Q_scf = 1.9875e-2 * (Tsc/Psc) * (W * M / T) * (m(Pbhp) - m(P))
    Q_res = Q_scf * Bg
    ```

    Pressure-squared (valid when mu*Z is approximately constant, i.e.
    low-to-moderate pressures below ~2000 psi):

    ```
    Q_scf = 1.9875e-2 * (Tsc/Psc) * (W * M / T) * (Pbhp^2 - P^2) / Z
    Q_res = Q_scf * Bg
    ```

    Since phase_mobility = kr/mu (no Bg term), Bg must be supplied via
    `formation_volume_factor` or computed internally from `gas_gravity`.

    Sign convention:
        - Negative rate indicates production (BHP < reservoir pressure).
        - Positive rate  indicates injection  (BHP > reservoir pressure).

    :param well_index: Well index (mD*ft).
    :param pressure: Reservoir pressure at the perforation interval (psi).
    :param temperature: Reservoir temperature (deg F).
    :param bottom_hole_pressure: Well bottom-hole pressure (psi).
    :param phase_mobility: Phase relative mobility kr/mu (md/cP).
        Must not include the Bg term; surface-to-reservoir conversion is
        handled here via `formation_volume_factor` or `gas_gravity`.
    :param average_compressibility_factor: Average Z-factor over the pressure
        interval.  Used only for the pressure-squared formulation.
        Default 1.0.
    :param use_pseudo_pressure: If True (default), use the pseudo-pressure
        formulation.  If False, use the pressure-squared formulation.
    :param pseudo_pressure_table: Pre-computed pseudo-pressure lookup table.
        Required when `use_pseudo_pressure=True`.
    :param formation_volume_factor: Gas formation volume factor Bg (ft³/SCF)
        at reservoir conditions.  If None, Bg is computed internally from
        `gas_gravity`, `pressure`, and `temperature`.
    :param gas_gravity: Gas specific gravity (air = 1.0).  Required when
        `formation_volume_factor` is None.
    :return: Gas well rate at reservoir conditions (ft³/day).  Negative for
        production, positive for injection.
    """
    if well_index <= 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility <= 0:
        raise ValidationError("Phase mobility must be a positive value.")

    Tsc = c.STANDARD_TEMPERATURE_RANKINE
    Psc = c.STANDARD_PRESSURE_IMPERIAL
    temperature_rankine = fahrenheit_to_rankine(temperature)

    if formation_volume_factor is not None:
        gas_fvf = formation_volume_factor
    else:
        if gas_gravity is None:
            raise ComputationError(
                "`gas_gravity` is required if `formation_volume_factor` is not provided."
            )
        z_at_reservoir = compute_gas_compressibility_factor(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=gas_gravity,
            method="dak",
        )
        # Bg = 0.02827 * Z * T / P  (ft³/SCF)
        gas_fvf = 0.02827 * z_at_reservoir * temperature_rankine / pressure

    # Common prefactor: 1.9875e-2 * (Tsc/Psc) * (W * M / T)
    # Gives Q in SCF/day; multiply by Bg at the end for ft³/day.
    prefactor = (
        1.9875e-2 * (Tsc / Psc) * (well_index * phase_mobility / temperature_rankine)
    )
    if use_pseudo_pressure:
        if pseudo_pressure_table is None:
            raise ValidationError(
                "`pseudo_pressure_table` must be provided when `use_pseudo_pressure` is True."
            )
        pseudo_pressure_difference = pseudo_pressure_table(
            bottom_hole_pressure
        ) - pseudo_pressure_table(pressure)
        well_rate_scf = prefactor * pseudo_pressure_difference
    else:
        pressure_squared_difference = bottom_hole_pressure**2 - pressure**2
        well_rate_scf = (
            prefactor * pressure_squared_difference / average_compressibility_factor
        )

    # Convert SCF/day to reservoir ft³/day
    return well_rate_scf * gas_fvf


def compute_required_bhp_for_gas_rate(
    target_rate: float,
    well_index: float,
    pressure: float,
    temperature: float,
    phase_mobility: float,
    average_compressibility_factor: float = 1.0,
    use_pseudo_pressure: bool = True,
    pseudo_pressure_table: typing.Optional[GasPseudoPressureTable] = None,
    formation_volume_factor: typing.Optional[float] = None,
) -> float:
    """
    Compute the bottom-hole pressure required to achieve a target gas rate.

    This is the exact algebraic inverse of `compute_gas_well_rate`.
    `target_rate` must be at reservoir conditions (ft³/day), consistent
    with the return value of :func:`compute_gas_well_rate`.

    Sign convention:
        - Negative `target_rate` indicates production (returned BHP < reservoir pressure).
        - Positive `target_rate`  indicates injection  (returned BHP > reservoir pressure).

    Inverse formula (pseudo-pressure):

    ```
    Q_scf   = target_rate / Bg
    m(Pbhp) = m(P) + Q_scf * T * Psc / (1.9875e-2 * Tsc * W * M)
    Pbhp    = m^-1( m(Pbhp) )  [inverse table interpolation]
    ```

    Inverse formula (pressure-squared):

    ```
    Q_scf  = target_rate / Bg
    Pbhp   = sqrt( P^2 + Q_scf * T * Psc * Z / (1.9875e-2 * Tsc * W * M) )
    ```

    :param target_rate: Target gas rate at reservoir conditions (ft³/day).
        Negative for production, positive for injection.
    :param well_index: Well index (mD*ft).
    :param pressure: Reservoir pressure at the perforation interval (psi).
    :param temperature: Reservoir temperature (deg F).
    :param phase_mobility: Phase relative mobility kr/mu (md/cP).
        Must not include the Bg term.
    :param average_compressibility_factor: Average Z-factor over the pressure
        interval.  Used only for the pressure-squared formulation.
        Default 1.0.
    :param use_pseudo_pressure: If True (default), use the pseudo-pressure
        inverse.  If False, use the pressure-squared inverse.
    :param pseudo_pressure_table: Pre-computed pseudo-pressure lookup table
        supporting inverse interpolation.  Required when
        `use_pseudo_pressure=True`.
    :param formation_volume_factor: Gas Bg (ft³/SCF) at reservoir conditions.
        Required to convert the reservoir-condition target rate back to
        surface SCF/day before inverting the well equation.
    :return: Required bottom-hole pressure (psi).
    """
    if well_index <= 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility <= 0:
        raise ValidationError("Phase mobility must be a positive value.")
    if formation_volume_factor is None:
        raise ValidationError(
            "`formation_volume_factor` (Bg) must be provided to convert the "
            "reservoir-condition target rate to surface conditions."
        )

    Tsc = c.STANDARD_TEMPERATURE_RANKINE
    Psc = c.STANDARD_PRESSURE_IMPERIAL
    temperature_rankine = fahrenheit_to_rankine(temperature)

    # Convert reservoir ft³/day back to SCF/day for inversion
    target_rate_scf = target_rate / formation_volume_factor

    # Common denominator: 1.9875e-2 * Tsc * W * M / T
    denominator = 1.9875e-2 * Tsc * well_index * phase_mobility / temperature_rankine

    if use_pseudo_pressure:
        if pseudo_pressure_table is None:
            raise ValidationError(
                "`pseudo_pressure_table` must be provided when `use_pseudo_pressure` is True."
            )
        # m(Pbhp) = m(P) + Q_scf * Psc / denominator
        reservoir_pseudo_pressure = pseudo_pressure_table(pressure)
        required_pseudo_pressure = (
            reservoir_pseudo_pressure + target_rate_scf * Psc / denominator
        )
        try:
            return float(
                pseudo_pressure_table.inverse_interpolate(
                    pseudo_pressure=required_pseudo_pressure
                )
            )
        except ValidationError as exc:
            min_mp = pseudo_pressure_table.pseudo_pressures[0]
            max_mp = pseudo_pressure_table.pseudo_pressures[-1]
            raise ComputationError(
                f"Cannot achieve target rate {target_rate:.2f} ft³/day — "
                f"required pseudo-pressure {required_pseudo_pressure:.2f} is outside "
                f"the valid table range [{min_mp:.2f}, {max_mp:.2f}]."
            ) from exc

    # Pressure-squared inverse.
    # Pbhp^2 = P^2 + Q_scf * Psc * Z / denominator
    required_bhp_squared = (
        pressure**2
        + target_rate_scf * Psc * average_compressibility_factor / denominator
    )
    if required_bhp_squared < 0:
        raise ComputationError(
            f"Cannot achieve target rate {target_rate:.2f} ft³/day — "
            "results in negative pressure squared.  "
            "The requested rate exceeds reservoir deliverability."
        )
    return float(np.sqrt(required_bhp_squared))


def _build_table_interpolator(
    pvt_tables: PVTTables, property_name: str, temperature: FloatOrArray
):
    """
    Build a 1D interpolator for a given property at given temperature(s).

    :param pvt_tables: PVT tables containing the property data.
    :param property_name: Name of the property to interpolate (e.g., "gas_compressibility_factor").
    :param temperature: Temperature(s) at which to interpolate the property.
    """

    def interpolator(pressure: FloatOrArray) -> FloatOrArray:
        # Clamp pressure to table bounds
        result = pvt_tables.pt_interpolate(
            name=property_name, pressure=pressure, temperature=temperature
        )
        # Ensure positive values
        if result is None:
            raise ComputationError(
                f"Result cannot be None ensure PVT table contains {property_name!r} interpolator. Use `table.exists({property_name!r})`"
            )
        return result

    interpolator._supports_arrays = True  # type: ignore
    interpolator.__name__ = f"{property_name}_interpolator"
    return interpolator


@attrs.frozen
class WellFluid(Serializable):
    """Base class for fluid properties in wells."""

    name: str
    """Name of the fluid. Examples: Methane, CO2, Water, Oil."""
    phase: typing.Union[FluidPhase, str] = attrs.field(converter=FluidPhase)
    """Phase of the fluid. Examples: WATER, GAS, OIL."""
    specific_gravity: float = attrs.field(validator=attrs.validators.ge(0))
    """Specific gravity of the fluid in (lbm/ft³)."""
    molecular_weight: float = attrs.field(validator=attrs.validators.ge(0))
    """Molecular weight of the fluid in (g/mol)."""

    def _build_pseudo_pressure_cache_key(
        self,
        temperature: float,
        reference_pressure: typing.Optional[float] = None,
        pressure_range: typing.Optional[typing.Tuple[float, float]] = None,
        points: typing.Optional[int] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> typing.Tuple[typing.Any, ...]:
        """
        Build a hashable cache key for pseudo-pressure table lookup.

        The key uniquely identifies a pseudo-pressure table based on all parameters
        that affect the Z-factor and viscosity functions.

        :param temperature: Temperature (°F)
        :param reference_pressure: Reference pressure (psi)
        :param pressure_range: (min, max) pressure range (psi)
        :param points: Number of points
        :param pvt_tables: Optional PVT tables for Z and μ interpolation
        :return: Hashable tuple that can be used as cache key
        """
        # PVT tables hash: if tables provided, use a hash of their configuration
        # Otherwise, use None to indicate correlation-based calculation
        if pvt_tables is not None:
            # Hash based on table metadata
            p_bounds = pvt_tables._extrapolation_bounds.get("pressure", (0.0, 0.0))
            t_bounds = pvt_tables._extrapolation_bounds.get("temperature", (0.0, 0.0))
            pvt_hash = (
                (round(p_bounds[0], 2), round(p_bounds[1], 2)),  # Pressure bounds
                (round(t_bounds[0], 2), round(t_bounds[1], 2)),  # Temperature bounds
                pvt_tables.interpolation_method,  # Interpolation method
                pvt_tables.exists("gas_compressibility_factor"),  # Z table exists?
                pvt_tables.exists("gas_viscosity"),  # μ table exists?
            )
        else:
            pvt_hash = None

        cache_key = (
            self.name,
            self.phase.value,  # type: ignore
            round(self.specific_gravity, 6),
            round(self.molecular_weight, 6),
            round(temperature, 2),
            round(reference_pressure, 2) if reference_pressure is not None else None,
            tuple(round(p, 2) for p in pressure_range)
            if pressure_range is not None
            else None,
            points,
            pvt_hash,
        )
        return cache_key

    def get_pseudo_pressure_table(
        self,
        temperature: float,
        reference_pressure: typing.Optional[float] = None,
        pressure_range: typing.Optional[typing.Tuple[float, float]] = None,
        points: typing.Optional[int] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
        use_cache: bool = True,
    ) -> GasPseudoPressureTable:
        """
        Get gas pseudo-pressure table for this fluid.

        Uses global caching to avoid recomputing tables for identical fluid properties.
        Multiple `WellFluid` instances with the same properties will share cached tables.

        :param temperature: Temperature (°F)
        :param reference_pressure: Reference pressure (psi), default 14.7
        :param pressure_range: (min, max) pressure range (psi), default (14.7, 5000)
        :param points: Number of points, default 100
        :param pvt_tables: Optional PVT tables for Z and μ interpolation
        :param use_cache: If True, use global cache. If False, always compute new table.
        :return: `GasPseudoPressureTable` instance

        Example:
        ```python
        # These two fluids will share the same cached table:
        methane1 = WellFluid(
            name="CH4-1",
            phase=FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04
        )
        methane2 = WellFluid(
            name="CH4-2",
            phase=FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04
        )

        table1 = methane1.get_pseudo_pressure_table(temperature=150)
        table2 = methane2.get_pseudo_pressure_table(temperature=150)
        # table1 is table2  # True! Same cached instance
        ```
        """
        if self.phase != FluidPhase.GAS:
            raise ValidationError(
                "Pseudo-pressure table is only applicable for gas phase."
            )

        z_factor_func = None  # type: ignore
        viscosity_func = None  # type: ignore

        if pvt_tables is not None:
            if pvt_tables.exists("gas_compressibility_factor"):
                z_factor_func = _build_table_interpolator(  # type: ignore
                    pvt_tables=pvt_tables,
                    property_name="gas_compressibility_factor",
                    temperature=temperature,
                )
            if pvt_tables.exists("gas_viscosity"):
                viscosity_func = _build_table_interpolator(  # type: ignore
                    pvt_tables=pvt_tables,
                    property_name="gas_viscosity",
                    temperature=temperature,
                )

        if z_factor_func is None:

            def z_factor_func(pressure: np.typing.NDArray) -> np.typing.NDArray:
                temperature_array = np.full_like(pressure, temperature)
                specific_gravity_array = np.full_like(pressure, self.specific_gravity)
                return compute_gas_compressibility_factor_vectorized(
                    pressure=pressure,
                    temperature=temperature_array,
                    gas_gravity=specific_gravity_array,
                )

            z_factor_func._supports_arrays = True  # type: ignore

        if viscosity_func is None:

            def viscosity_func(pressure: np.typing.NDArray) -> np.typing.NDArray:
                temperature_array = np.full_like(pressure, temperature)
                specific_gravity_array = np.full_like(pressure, self.specific_gravity)
                gas_density = compute_gas_density_vectorized(
                    pressure=pressure,
                    temperature=temperature_array,
                    gas_gravity=specific_gravity_array,
                    gas_compressibility_factor=z_factor_func(pressure),
                )
                return compute_gas_viscosity_vectorized(
                    temperature=temperature_array,
                    gas_density=gas_density,
                    gas_molecular_weight=self.molecular_weight,
                )

            viscosity_func._supports_arrays = True  # type: ignore

        cache_key = None
        if use_cache:
            cache_key = self._build_pseudo_pressure_cache_key(
                temperature=temperature,
                reference_pressure=reference_pressure,
                pressure_range=pressure_range,
                points=points,
                pvt_tables=pvt_tables,
            )

        return build_gas_pseudo_pressure_table(
            z_factor_func=z_factor_func,  # type: ignore[arg-type]
            viscosity_func=viscosity_func,  # type: ignore[arg-type]
            reference_pressure=reference_pressure,
            pressure_range=pressure_range,
            points=points,
            cache_key=cache_key,
        )


@typing.final
@attrs.frozen
class InjectedFluid(WellFluid):
    """Properties of the fluid being injected into or produced by a well."""

    salinity: typing.Optional[float] = None
    """Salinity of the fluid (if water) in (ppm NaCl)."""
    is_miscible: bool = False
    """Whether this fluid is miscible with oil (e.g., CO2, N2)"""
    todd_longstaff_omega: float = attrs.field(
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0), attrs.validators.le(1.0)
        ),
        default=0.67,
    )
    """Todd-Longstaff mixing parameter for miscible displacement (0 to 1)."""
    minimum_miscibility_pressure: typing.Optional[float] = None
    """Minimum miscibility pressure for this fluid-oil system (psi)"""
    miscibility_transition_width: float = attrs.field(  # type: ignore
        default=500.0, validator=attrs.validators.ge(0)
    )
    """Pressure range over which miscibility transitions from immiscible to miscible (psi)"""
    concentration: float = attrs.field(
        default=1.0,
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0), attrs.validators.le(1.0)
        ),
    )
    """Concentration (preferrably volume-based) of the fluid in the mixture (0 to 1). Relevant for miscible fluids."""
    density: typing.Optional[float] = None
    """Fluid density (lbm/ft³) at reservoir conditions. If provided, bypasses table/correlation-based density calculations. Useful for non-ideal gases like CO2."""
    viscosity: typing.Optional[float] = None
    """Fluid viscosity (cP) at reservoir conditions. If provided, bypasses table/correlation-based viscosity calculations. Useful for non-ideal gases like CO2."""

    def __attrs_post_init__(self) -> None:
        """Validate the fluid properties."""
        if self.phase not in (FluidPhase.GAS, FluidPhase.WATER):
            raise ValidationError("Only gases and water are supported for injection.")

        if self.is_miscible:
            if self.phase != FluidPhase.GAS:
                raise ValidationError("Only gas phase fluids can be miscible.")
            elif not self.minimum_miscibility_pressure or not self.todd_longstaff_omega:
                raise ValidationError(
                    "Miscible fluids must have both `minimum_miscibility_pressure` and `todd_longstaff_omega` defined."
                )

    def get_density(
        self, pressure: FloatOrArray, temperature: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get the density of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the density (psi).
        :param temperature: The temperature at which to evaluate the density (°F).
        :kwargs: Additional parameters for phase density calculations.
        :return: The density of the fluid (lbm/ft³).
        """
        if self.density is not None:
            if isinstance(pressure, np.ndarray):
                return np.full_like(pressure, self.density)
            return self.density

        vectorize_pressure = isinstance(pressure, np.ndarray)
        vectorize_temperature = isinstance(temperature, np.ndarray)
        use_vectorization = vectorize_pressure or vectorize_temperature
        if use_vectorization and not vectorize_pressure:
            pressure = np.full_like(temperature, pressure)
        elif use_vectorization and not vectorize_temperature:
            temperature = np.full_like(pressure, temperature)

        if self.phase == FluidPhase.WATER:
            gas_free_water_fvf = kwargs.get(
                "gas_free_water_formation_volume_factor", None
            )
            if gas_free_water_fvf is None:
                if use_vectorization:
                    gas_free_water_fvf = (
                        compute_gas_free_water_formation_volume_factor_vectorized(
                            pressure=pressure,  # type: ignore
                            temperature=temperature,  # type: ignore
                        )
                    )
                else:
                    gas_free_water_fvf = compute_gas_free_water_formation_volume_factor(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                    )
                kwargs["gas_free_water_formation_volume_factor"] = gas_free_water_fvf

            # Assume no-gas in injection water if not explicitly specified
            if "gas_solubility_in_water" not in kwargs:
                kwargs["gas_solubility_in_water"] = 0.0
            if "gas_gravity" not in kwargs:
                kwargs["gas_gravity"] = self.specific_gravity

            if use_vectorization:
                return compute_water_density_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    salinity=self.salinity or 0.0,
                    **kwargs,
                )

            return compute_water_density(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                salinity=self.salinity or 0.0,
                **kwargs,
            )

        gas_z_factor = kwargs.get("gas_compressibility_factor", None)
        if gas_z_factor is None:
            if use_vectorization:
                gas_z_factor = compute_gas_compressibility_factor_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=np.full_like(pressure, self.specific_gravity),
                    method="dak",
                )
            else:
                gas_z_factor = compute_gas_compressibility_factor(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=self.specific_gravity,
                    method="dak",
                )
        if use_vectorization:
            return compute_gas_density_vectorized(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                gas_gravity=np.full_like(pressure, self.specific_gravity),
                gas_compressibility_factor=gas_z_factor,
            )
        return compute_gas_density(
            pressure=pressure,  # type: ignore
            temperature=temperature,  # type: ignore
            gas_gravity=self.specific_gravity,
            gas_compressibility_factor=gas_z_factor,  # type: ignore
        )

    def get_viscosity(
        self, pressure: FloatOrArray, temperature: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get the viscosity of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the viscosity (psi).
        :param temperature: The temperature at which to evaluate the viscosity (°F).
        :kwargs: Additional parameters for viscosity calculations.
        :return: The viscosity of the fluid (cP).
        """
        if self.viscosity is not None:
            if isinstance(pressure, np.ndarray):
                return np.full_like(pressure, self.viscosity)
            return self.viscosity

        vectorize_pressure = isinstance(pressure, np.ndarray)
        vectorize_temperature = isinstance(temperature, np.ndarray)
        use_vectorization = vectorize_pressure or vectorize_temperature
        if use_vectorization and not vectorize_pressure:
            pressure = np.full_like(temperature, pressure)
        elif use_vectorization and not vectorize_temperature:
            temperature = np.full_like(pressure, temperature)

        if self.phase == FluidPhase.WATER:
            if use_vectorization:
                return compute_water_viscosity_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    salinity=self.salinity or 0.0,
                )
            return compute_water_viscosity(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                salinity=self.salinity or 0.0,
            )

        gas_density = kwargs.get("gas_density", None)
        if gas_density is None:
            gas_z_factor = kwargs.get("gas_compressibility_factor", None)
            if gas_z_factor is None:
                if use_vectorization:
                    gas_z_factor = compute_gas_compressibility_factor_vectorized(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                        gas_gravity=np.full_like(pressure, self.specific_gravity),
                        method="dak",
                    )
                else:
                    gas_z_factor = compute_gas_compressibility_factor(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                        gas_gravity=self.specific_gravity,
                        method="dak",
                    )

            if use_vectorization:
                gas_density = compute_gas_density_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=np.full_like(pressure, self.specific_gravity),
                    gas_compressibility_factor=gas_z_factor,
                )
            else:
                gas_density = compute_gas_density(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=self.specific_gravity,
                    gas_compressibility_factor=gas_z_factor,  # type: ignore
                )

        if use_vectorization:
            return compute_gas_viscosity_vectorized(
                temperature=temperature,  # type: ignore
                gas_density=gas_density,  # type: ignore
                gas_molecular_weight=self.molecular_weight,
            )
        return compute_gas_viscosity(
            temperature=temperature,  # type: ignore
            gas_density=gas_density,  # type: ignore
            gas_molecular_weight=self.molecular_weight,
        )

    def get_compressibility(
        self, pressure: FloatOrArray, temperature: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get the compressibility of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the compressibility (psi).
        :param temperature: The temperature at which to evaluate the compressibility (°F).
        :kwargs: Additional parameters for compressibility calculations.

        For water:
            :kwarg bubble_point_pressure: The bubble point pressure (psi).
            :kwarg gas_formation_volume_factor: The gas formation volume factor (ft³/scf).
            :kwarg gas_solubility_in_water: The gas solubility in water (scf/stb).

        For gas:
            :kwarg gas_gravity: The specific gravity of the gas (dimensionless). Optional
                Uses the fluid's specific gravity if not provided.
            :kwarg gas_compressibility_factor: The gas compressibility factor (dimensionless).

        :return: The compressibility of the fluid (psi⁻¹).
        """
        vectorize_pressure = isinstance(pressure, np.ndarray)
        vectorize_temperature = isinstance(temperature, np.ndarray)
        use_vectorization = vectorize_pressure or vectorize_temperature
        if use_vectorization and not vectorize_pressure:
            pressure = np.full_like(temperature, pressure)
        elif use_vectorization and not vectorize_temperature:
            temperature = np.full_like(pressure, temperature)

        if self.phase == FluidPhase.WATER:
            gas_free_water_fvf = kwargs.get(
                "gas_free_water_formation_volume_factor", None
            )
            if gas_free_water_fvf is None:
                if use_vectorization:
                    gas_free_water_fvf = (
                        compute_gas_free_water_formation_volume_factor_vectorized(
                            pressure=pressure,  # type: ignore
                            temperature=temperature,  # type: ignore
                        )
                    )
                else:
                    gas_free_water_fvf = compute_gas_free_water_formation_volume_factor(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                    )
                kwargs["gas_free_water_formation_volume_factor"] = gas_free_water_fvf

            if use_vectorization:
                return compute_water_compressibility_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    salinity=self.salinity or 0.0,
                    **kwargs,
                )

            return compute_water_compressibility(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                salinity=self.salinity or 0.0,
                **kwargs,
            )

        kwargs.setdefault("gas_gravity", self.specific_gravity)
        if use_vectorization:
            if "gas_gravity" not in kwargs:
                kwargs["gas_gravity"] = np.full_like(pressure, self.specific_gravity)
            return compute_gas_compressibility_vectorized(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                **kwargs,
            )
        return compute_gas_compressibility(
            pressure=pressure,  # type: ignore
            temperature=temperature,  # type: ignore
            **kwargs,
        )

    def get_formation_volume_factor(
        self, pressure: FloatOrArray, temperature: FloatOrArray, **kwargs: typing.Any
    ) -> FloatOrArray:
        """
        Get the formation volume factor of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the formation volume factor (psi).
        :param temperature: The temperature at which to evaluate the formation volume factor (°F).
        :kwargs: Additional parameters for formation volume factor calculations.
        :return: The formation volume factor of the fluid (bbl/STB or ft³/SCF).
        """
        vectorize_pressure = isinstance(pressure, np.ndarray)
        vectorize_temperature = isinstance(temperature, np.ndarray)
        use_vectorization = vectorize_pressure or vectorize_temperature
        if use_vectorization and not vectorize_pressure:
            pressure = np.full_like(temperature, pressure)
        elif use_vectorization and not vectorize_temperature:
            temperature = np.full_like(pressure, temperature)

        if self.phase == FluidPhase.WATER:
            water_density = kwargs.get("water_density", None)
            if water_density is None:
                # Not need for gas free fvf or gas fvf, since injection water
                # is typically gas free fresh water or degassed formation water
                if use_vectorization:
                    water_density = compute_water_density_vectorized(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                        salinity=self.salinity or 0.0,
                    )
                else:
                    water_density = compute_water_density(
                        pressure=pressure,  # type: ignore
                        temperature=temperature,  # type: ignore
                        salinity=self.salinity or 0.0,
                    )

            if use_vectorization:
                return compute_water_formation_volume_factor_vectorized(
                    salinity=self.salinity or 0.0,
                    water_density=water_density,  # type: ignore
                )
            return compute_water_formation_volume_factor(
                salinity=self.salinity or 0.0,
                water_density=water_density,  # type: ignore
            )

        gas_z_factor = kwargs.get("gas_compressibility_factor", None)
        if gas_z_factor is None:
            if use_vectorization:
                gas_z_factor = compute_gas_compressibility_factor_vectorized(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=np.full_like(pressure, self.specific_gravity),
                    method="dak",
                )
            else:
                gas_z_factor = compute_gas_compressibility_factor(
                    pressure=pressure,  # type: ignore
                    temperature=temperature,  # type: ignore
                    gas_gravity=self.specific_gravity,
                    method="dak",
                )

        if use_vectorization:
            return compute_gas_formation_volume_factor_vectorized(
                pressure=pressure,  # type: ignore
                temperature=temperature,  # type: ignore
                gas_compressibility_factor=gas_z_factor,  # type: ignore
            )
        return compute_gas_formation_volume_factor(
            pressure=pressure,  # type: ignore
            temperature=temperature,  # type: ignore
            gas_compressibility_factor=gas_z_factor,  # type: ignore
        )


@typing.final
@attrs.frozen
class ProducedFluid(WellFluid):
    """Properties of the fluid being produced by a well."""

    pass


WellFluidT = typing.TypeVar("WellFluidT", bound=WellFluid)


@numba.njit(cache=True, inline="always")
def _geometric_mean(values: typing.Sequence[float]) -> float:
    prod = 1.0
    n = 0
    for v in values:
        prod *= max(v, 0.0)  # ensure non-negative
        n += 1
    if n == 0:
        raise ValidationError("No permeability values provided")
    return prod ** (1.0 / n)


@numba.njit(cache=True)
def compute_effective_permeability_for_well(
    permeability: typing.Sequence[float], orientation: Orientation
) -> float:
    """
    Compute `k_eff` for Peaceman WI using geometric mean of the two permeabilities
    perpendicular to the well axis. `permeability` is (kx, ky, kz).
    orientation is one of Orientation.X/Y/Z (or a string equivalent).
    """
    if len(permeability) != 3:
        # If 2D, fall back to geometric mean of available components:
        return _geometric_mean(permeability)

    kx, ky, kz = permeability
    if orientation == Orientation.Z:  # vertical well: transverse are x,y
        return np.sqrt(max(kx, 0.0) * max(ky, 0.0))
    elif orientation == Orientation.X:  # well along x: transverse are y,z
        return np.sqrt(max(ky, 0.0) * max(kz, 0.0))
    elif orientation == Orientation.Y:  # well along y: transverse are x,z
        return np.sqrt(max(kx, 0.0) * max(kz, 0.0))
    # For Oblique/unknown orientation, use geometric mean of all three
    return _geometric_mean((kx, ky, kz))
