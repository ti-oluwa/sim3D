"""Core well calculations and utilities."""

import logging
import typing

import attrs
import numba
import numpy as np

from bores.constants import c
from bores.errors import ComputationError, ValidationError
from bores.pvt.core import (
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
from bores.types import FluidPhase, Orientation, ThreeDimensions, TwoDimensions
from bores.pvt.tables import (
    GasPseudoPressureTable,
    PVTTables,
    build_gas_pseudo_pressure_table,
)

logger = logging.getLogger(__name__)

__all__ = [
    "compute_well_index",
    "compute_3D_effective_drainage_radius",
    "compute_2D_effective_drainage_radius",
    "compute_oil_well_rate",
    "compute_gas_well_rate",
    "compute_required_bhp_for_oil_rate",
    "compute_required_bhp_for_gas_rate",
    "WellFluid",
    "InjectedFluid",
    "ProducedFluid",
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
    well_index = (permeability * interval_thickness) / (
        np.log(effective_drainage_radius / wellbore_radius) + skin_factor
    )
    return well_index


@numba.njit(cache=True)
def compute_3D_effective_drainage_radius(
    interval_thickness: ThreeDimensions,
    permeability: ThreeDimensions,
    well_orientation: Orientation,
) -> float:
    """
    Compute the effective drainage radius for a well ina 3D reservoir model using
    Peaceman's effective drainage radius formula.

    The formula for is given by:

    For x-direction:

        r_x = 0.28 * √[ (∆y² + ∆z²) / (√(k_y / k_z) + √(k_z / k_y)) ]

    For y-direction:

        r_y = 0.28 * √[ (∆x² + ∆z²) / (√(k_x / k_z) + √(k_z / k_x)) ]

    For z-direction:
        r_z = 0.28 * √[ (∆x² + ∆y²) / (√(k_x / k_y) + √(k_y / k_x)) ]

    where:
        - r_x, r_y, r_z are the effective drainage radii in the x, y, and z directions respectively.
        - ∆x, ∆y, ∆z are the thicknesses of the reservoir interval in the x, y, and z directions respectively.
        - k_x, k_y, k_z are the permeabilities of the reservoir rock in the x, y, and z directions respectively.

    :param interval_thickness: A tuple representing the thickness of the reservoir interval in the x, y, and z directions (ft).
    :param permeability: A tuple representing the permeability of the reservoir rock in the x, y, and z directions (mD).
    :param well_orientation: The orientation of the well (Orientation.X, Orientation.Y, or Orientation.Z).
    :return: The effective drainage radius in the direction of the well (ft).
    """
    if well_orientation == Orientation.X:
        delta_y, delta_z = interval_thickness[1], interval_thickness[2]
        k_y, k_z = permeability[1], permeability[2]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_y**2 + delta_z**2) / (np.sqrt(k_y / k_z) + np.sqrt(k_z / k_y))
        )
    elif well_orientation == Orientation.Y:
        delta_x, delta_z = interval_thickness[0], interval_thickness[2]
        k_x, k_z = permeability[0], permeability[2]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_x**2 + delta_z**2) / (np.sqrt(k_x / k_z) + np.sqrt(k_z / k_x))
        )
    elif well_orientation == Orientation.Z:
        delta_x, delta_y = interval_thickness[0], interval_thickness[1]
        k_x, k_y = permeability[0], permeability[1]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_x**2 + delta_y**2) / (np.sqrt(k_x / k_y) + np.sqrt(k_y / k_x))
        )
    else:
        raise ValidationError("Invalid well orientation")

    return effective_drainage_radius


@numba.njit(cache=True)
def compute_2D_effective_drainage_radius(
    interval_thickness: TwoDimensions,
    permeability: TwoDimensions,
    well_orientation: Orientation,
) -> float:
    """
    Compute the effective drainage radius for a well in a 2D reservoir model.

    The formula for is given by:

        r = 0.28 * √[ ( (∆x² * √(k_y / k_x)) + (∆y² * √(k_x / k_y)) ) / ( √(k_y / k_x) + √(k_x / k_y) ) ]

    where:
        - r_x, r_y are the effective drainage radii in the x and y directions respectively.
        - ∆x, ∆y are the thicknesses of the reservoir interval in the x and y directions respectively.
        - k_x, k_y are the permeabilities of the reservoir rock in the x and y directions respectively.

    :param interval_thickness: A tuple representing the thickness of the reservoir interval in the x and y directions (ft).
    :param permeability: A tuple representing the permeability of the reservoir rock in the x and y directions (mD).
    :param well_orientation: The orientation of the well (Orientation.X or Orientation.Y).
    :return: The effective drainage radius in the direction of the well (ft).
    """
    if well_orientation == Orientation.X:
        delta_x, delta_y = interval_thickness[0], interval_thickness[1]
        k_x, k_y = permeability[0], permeability[1]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_x**2 * np.sqrt(k_y / k_x) + delta_y**2 * np.sqrt(k_x / k_y))
            / (np.sqrt(k_y / k_x) + np.sqrt(k_x / k_y))
        )
    elif well_orientation == Orientation.Y:
        delta_x, delta_y = interval_thickness[0], interval_thickness[1]
        k_x, k_y = permeability[0], permeability[1]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_x**2 * np.sqrt(k_x / k_y) + delta_y**2 * np.sqrt(k_y / k_x))
            / (np.sqrt(k_x / k_y) + np.sqrt(k_y / k_x))
        )
    else:
        raise ValidationError("Invalid well orientation")
    return effective_drainage_radius


@numba.njit(cache=True)
def compute_oil_well_rate(
    well_index: float,
    pressure: float,
    bottom_hole_pressure: float,
    phase_mobility: float,
    fluid_compressibility: typing.Optional[float] = None,
) -> float:
    """
    Compute the well rate using the well index and pressure drop.

    This assumes radial flow to/from the wellbore.
    May be steady-state or pseudo-steady-state flow depending on the well index calculation.

    The formula for the well rate is:

        Q = 7.08e-3 * W * (P_bhp - P) * M

    or for slightly compressible fluids:

        Q = 7.08e-3 * W * M * ln(1 + c_f * (P_bhp - P)) / c_f

    where:
        - Q is the well rate (bbl/day)
        - W is the well index (mD*ft)
        - P is the reservoir pressure (psi)
        - P_bhp is the bottom-hole pressure (psi)
        - M is the phase mobility (cP⁻¹, default is 1.0) (k_r / μ) or (k_r / (μ * B)).

    Negative rate result indicates that the well is producing, while positive rates indicate injection.

    :param well_index: The well index (mD*ft).
    :param pressure: The reservoir pressure (psi).
    :param bottom_hole_pressure: The bottom-hole pressure (psi).
    :param phase_mobility: The phase relative mobility (cP⁻¹, default is 1.0) (k_r / μ) or (k_r / (μ * B)).
    :param fluid_compressibility: The fluid compressibility (1/psi). For slightly compressible fluids.
    :return: The well rate in bbl/day.
    """
    if well_index <= 0:
        raise ValidationError("Well index must be a positive value.")

    pressure_difference = bottom_hole_pressure - pressure
    if fluid_compressibility:
        well_rate = (
            7.08e-3
            * well_index
            * phase_mobility
            * np.log(1 + (fluid_compressibility * pressure_difference))
            / fluid_compressibility
        )
    else:
        well_rate = 7.08e-3 * well_index * phase_mobility * pressure_difference
    return well_rate


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
) -> float:
    """
    Compute the gas well rate using the well index and pressure drop.

    This assumes radial flow to/from the wellbore.
    May be steady-state or pseudo-steady-state flow depending on the well index calculation.

    The formula for the gas well rate is:

    For pseudo-pressure formulation:

        Q = 1.9875e-2 * (Tsc / Psc) * (W / T) * (m(P_bhp) - m(P))

    For pressure squared formulation:

        Q = 1.9875e-2 * (Tsc / Psc) * (W / T) * M * ((P_bhp² - P²) / Z)

    where:
        - Q is the gas well rate (SCF/day)
        - W is the well index (mD*ft)
        - P is the reservoir pressure (psi)
        - P_bhp is the bottom-hole pressure (psi)
        - m(P) is the pseudo-pressure at pressure P
        - T is the reservoir temperature (°F)
        - Tsc is the standard temperature (°R), typically 520 °R (60 °F)
        - Psc is the standard pressure (psi), typically 14.7 psi
        - M is the phase mobility (cP⁻¹, default is 1.0) (k_r / μ) or (k_r / (μ * B)).
        - Z_avg is the average compressibility factor in the reservoir interval.

    Negative rate result indicates that the well is producing, while positive rates indicate injection.

    :param well_index: The well index (mD*ft).
    :param pressure: The reservoir pressure (psi).
    :param temperature: The reservoir temperature (°F).
    :param bottom_hole_pressure: The bottom-hole pressure (psi).
    :param phase_mobility: The phase relative mobility (cP⁻¹, default is 1.0) (k_r / μ) or (k_r / (μ * B)).
    :param average_compressibility_factor: The average gas compressibility factor Z (default is 1.0).
    :param use_pseudo_pressure: Whether to use pseudo-pressure formulation (default is True).
    :param pseudo_pressure_table: Pre-computed pseudo-pressure table for fast lookup (required if use_pseudo_pressure is True).
    :param formation_volume_factor: Gas formation volume factor (ft³/SCF). If provided, it will be used directly instead of calculating from Z, T, and P.
    :return: The gas well rate (ft³/day).
    """
    if well_index <= 0:
        raise ValidationError("Well index must be a positive value.")

    Tsc = c.STANDARD_TEMPERATURE_RANKINE
    Psc = c.STANDARD_PRESSURE_IMPERIAL
    temperature_rankine = fahrenheit_to_rankine(temperature)

    if use_pseudo_pressure:
        if pseudo_pressure_table is None:
            raise ValidationError(
                "`pseudo_pressure_table` must be provided when use_pseudo_pressure is True."
            )

        bottom_hole_pseudo_pressure = pseudo_pressure_table(bottom_hole_pressure)
        reservoir_pseudo_pressure = pseudo_pressure_table(pressure)
        pseudo_pressure_difference = (
            bottom_hole_pseudo_pressure - reservoir_pseudo_pressure
        )
        well_rate = (
            1.9875e-2
            * (Tsc / Psc)
            * (well_index / temperature_rankine)
            * pseudo_pressure_difference
        )
    else:
        pressure_difference_squared = bottom_hole_pressure**2 - pressure**2
        well_rate = (
            1.9875e-2
            * (Tsc / Psc)
            * (well_index / temperature_rankine)
            * phase_mobility
            * (pressure_difference_squared / average_compressibility_factor)
        )

    if formation_volume_factor:
        gas_fvf = formation_volume_factor
    else:
        # Compute gas formation volume factor (ft³/SCF)
        # Bg = 0.02827 * (Z_avg * T) / P_avg
        average_pressure = 0.5 * (pressure + bottom_hole_pressure)
        gas_fvf = (
            0.02827
            * average_compressibility_factor
            * temperature_rankine
            / average_pressure
        )  # ft³/SCF
    return well_rate * gas_fvf  # ft³/day


@numba.njit(cache=True)
def compute_required_bhp_for_oil_rate(
    target_rate: float,
    well_index: float,
    pressure: float,
    phase_mobility: float,
    fluid_compressibility: typing.Optional[float] = None,
) -> float:
    """
    Compute the required bottom-hole pressure to achieve a target oil/water rate.

    This is the inverse of `compute_oil_well_rate()`.

    For incompressible fluids:
        Q = 7.08e-3 * W * M * (P_bhp - P)
        => P_bhp = P + Q / (7.08e-3 * W * M)

    For slightly compressible fluids:
        Q = 7.08e-3 * W * M * ln(1 + c_f * (P_bhp - P)) / c_f
        => P_bhp = P + (exp(Q * c_f / (7.08e-3 * W * M)) - 1) / c_f

    :param target_rate: Target well rate (bbl/day). Positive for injection, negative for production.
    :param well_index: The well index (mD*ft).
    :param pressure: The reservoir pressure (psi).
    :param phase_mobility: The phase relative mobility (cP⁻¹) (k_r / μ) or (k_r / (μ * B)).
    :param fluid_compressibility: The fluid compressibility (1/psi) for slightly compressible fluids.
    :return: Required bottom-hole pressure (psi).
    """
    if well_index <= 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility <= 0:
        raise ValidationError("Phase mobility must be a positive value.")

    conversion_factor = 7.08e-3

    if fluid_compressibility:
        # Slightly compressible: P_bhp = P + (exp(Q * c_f / (conversion * W * M)) - 1) / c_f
        exponent = (
            target_rate
            * fluid_compressibility
            / (conversion_factor * well_index * phase_mobility)
        )
        required_bhp = pressure + (np.exp(exponent) - 1.0) / fluid_compressibility
    else:
        # Incompressible: P_bhp = P + Q / (conversion * W * M)
        required_bhp = pressure + target_rate / (
            conversion_factor * well_index * phase_mobility
        )

    return float(required_bhp)


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
    Compute the required bottom-hole pressure to achieve a target gas rate.

    This is the inverse of `compute_gas_well_rate()`.

    For pseudo-pressure formulation:
        Q = 1.9875e-2 * (Tsc / Psc) * (W / T) * (m(P_bhp) - m(P))
        => m(P_bhp) = m(P) + Q * T * Psc / (1.9875e-2 * Tsc * W)
        => P_bhp = m^(-1)(m(P_bhp))  [inverse interpolation]

    For pressure squared formulation:
        Q = 1.9875e-2 * (Tsc / Psc) * (W / T) * M * (P_bhp² - P²) / Z
        => P_bhp = sqrt(P² + Q * T * Psc * Z / (1.9875e-2 * Tsc * W * M))

    :param target_rate: Target gas rate (ft³/day). Positive for injection, negative for production.
    :param well_index: The well index (mD*ft).
    :param pressure: The reservoir pressure (psi).
    :param temperature: The reservoir temperature (°F).
    :param phase_mobility: The phase relative mobility (cP⁻¹) (k_r / μ) or (k_r / (μ * B)).
    :param average_compressibility_factor: Average gas compressibility factor Z (default is 1.0).
    :param use_pseudo_pressure: Whether to use pseudo-pressure formulation (default is True).
    :param pseudo_pressure_table: Pre-computed pseudo-pressure table for inverse lookup.
    :param formation_volume_factor: Gas formation volume factor (ft³/SCF).
    :return: Required bottom-hole pressure (psi).
    """
    if well_index <= 0:
        raise ValidationError("Well index must be a positive value.")
    if phase_mobility <= 0:
        raise ValidationError("Phase mobility must be a positive value.")

    Tsc = c.STANDARD_TEMPERATURE_RANKINE
    Psc = c.STANDARD_PRESSURE_IMPERIAL
    temperature_rankine = fahrenheit_to_rankine(temperature)

    # Account for formation volume factor if provided
    if formation_volume_factor:
        # Convert target rate back to surface conditions
        target_rate_surface = target_rate / formation_volume_factor
    else:
        target_rate_surface = target_rate

    if use_pseudo_pressure:
        if pseudo_pressure_table is None:
            raise ValidationError(
                "Pseudo-pressure table is required for pseudo-pressure formulation."
            )

        # m(P_bhp) = m(P) + Q * T * Psc / (1.9875e-2 * Tsc * W)
        reservoir_pseudo_pressure = pseudo_pressure_table(pressure)
        pseudo_pressure_change = (
            target_rate_surface
            * temperature_rankine
            * Psc
            / (1.9875e-2 * Tsc * well_index)
        )
        required_pseudo_pressure = reservoir_pseudo_pressure + pseudo_pressure_change

        # Find the pressure corresponding to required_pseudo_pressure
        try:
            required_bhp = pseudo_pressure_table.inverse_interpolate(
                pseudo_pressure=required_pseudo_pressure
            )
        except ValidationError as exc:
            min_pseudo_p = pseudo_pressure_table.pseudo_pressures[0]
            max_pseudo_p = pseudo_pressure_table.pseudo_pressures[-1]
            raise ComputationError(
                f"Cannot achieve target rate {target_rate:.2f} - "
                f"required pseudo-pressure {required_pseudo_pressure:.2f} outside valid range. "
                f"Valid range: [{min_pseudo_p:.2f}, {max_pseudo_p:.2f}]"
            ) from exc
    else:
        # P_bhp² = P² + Q * T * Psc * Z / (1.9875e-2 * Tsc * W * M)
        pressure_squared_change = (
            target_rate_surface
            * temperature_rankine
            * Psc
            * average_compressibility_factor
            / (1.9875e-2 * Tsc * well_index * phase_mobility)
        )
        required_bhp_squared = pressure**2 + pressure_squared_change

        if required_bhp_squared < 0:
            raise ComputationError(
                f"Cannot achieve target rate {target_rate:.2f} - results in negative pressure squared."
            )
        required_bhp = np.sqrt(required_bhp_squared)

    return float(required_bhp)


def _build_table_interpolator(
    pvt_tables: PVTTables, property_name: str, temperature: float
):
    """Build a 1D interpolator for a given property at fixed temperature."""

    def _interpolator(pressure: float) -> float:
        # Clamp pressure to table bounds
        result = pvt_tables.pt_interpolate(
            name=property_name, pressure=pressure, temperature=temperature
        )
        # Ensure positive values
        if result is None:
            raise ComputationError(
                f"Result cannot be None ensure PVT table contains {property_name!r} interpolator. Use `table.exists({property_name!r})`"
            )
        return typing.cast(float, result)

    return _interpolator


@attrs.frozen
class WellFluid:
    """Base class for fluid properties in wells."""

    name: str
    """Name of the fluid. Examples: Methane, CO2, Water, Oil."""
    phase: FluidPhase
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

        # Build cache key
        cache_key = (
            self.name,  # Fluid identifier (e.g., "CH4", "CO2")
            self.phase.value,  # Phase enum value
            round(
                self.specific_gravity, 6
            ),  # Gas gravity (rounded to avoid float precision issues)
            round(self.molecular_weight, 6),  # Molecular weight (rounded)
            round(temperature, 2),  # Temperature (rounded to 0.01 °F precision)
            round(reference_pressure, 2)
            if reference_pressure is not None
            else None,  # Reference pressure
            tuple(round(p, 2) for p in pressure_range)
            if pressure_range is not None
            else None,  # Pressure range
            points,  # Number of points
            pvt_hash,  # PVT table configuration (or None)
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
        methane1 = WellFluid(name="CH4-1", phase=FluidPhase.GAS,
                             specific_gravity=0.65, molecular_weight=16.04)
        methane2 = WellFluid(name="CH4-2", phase=FluidPhase.GAS,
                             specific_gravity=0.65, molecular_weight=16.04)

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
                z_factor_func = _build_table_interpolator(
                    pvt_tables=pvt_tables,
                    property_name="gas_compressibility_factor",
                    temperature=temperature,
                )
            if pvt_tables.exists("gas_viscosity"):
                viscosity_func = _build_table_interpolator(
                    pvt_tables=pvt_tables,
                    property_name="gas_viscosity",
                    temperature=temperature,
                )

        if z_factor_func is None:

            def z_factor_func(pressure: float) -> float:
                return compute_gas_compressibility_factor(
                    pressure=pressure,
                    temperature=temperature,
                    gas_gravity=self.specific_gravity,
                )

        if viscosity_func is None:

            def viscosity_func(pressure: float) -> float:
                return compute_gas_viscosity(
                    temperature=temperature,
                    gas_density=compute_gas_density(
                        pressure=pressure,
                        temperature=temperature,
                        gas_gravity=self.specific_gravity,
                        gas_compressibility_factor=z_factor_func(pressure),
                    ),
                    gas_molecular_weight=self.molecular_weight,
                )

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
            z_factor_func=z_factor_func,
            viscosity_func=viscosity_func,
            reference_pressure=reference_pressure,
            pressure_range=pressure_range,
            points=points,
            cache_key=cache_key,
        )


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
        self, pressure: float, temperature: float, **kwargs: typing.Any
    ) -> float:
        """
        Get the density of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the density (psi).
        :param temperature: The temperature at which to evaluate the density (°F).
        :kwargs: Additional parameters for phase density calculations.
        :return: The density of the fluid (lbm/ft³).
        """
        if self.phase == FluidPhase.WATER:
            gas_free_water_fvf = kwargs.get(
                "gas_free_water_formation_volume_factor", None
            )
            if gas_free_water_fvf is None:
                gas_free_water_fvf = compute_gas_free_water_formation_volume_factor(
                    pressure=pressure, temperature=temperature
                )
                kwargs["gas_free_water_formation_volume_factor"] = gas_free_water_fvf

            # Assume no-gas in injection water if not explicitly specified
            if "gas_solubility_in_water" not in kwargs:
                kwargs["gas_solubility_in_water"] = 0.0
            if "gas_gravity" not in kwargs:
                kwargs["gas_gravity"] = self.specific_gravity

            return compute_water_density(
                pressure=pressure,
                temperature=temperature,
                salinity=self.salinity or 0.0,
                **kwargs,
            )

        gas_z_factor = kwargs.get("gas_compressibility_factor", None)
        if gas_z_factor is None:
            gas_z_factor = compute_gas_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=self.specific_gravity,
            )
        return compute_gas_density(
            pressure=pressure,
            temperature=temperature,
            gas_gravity=self.specific_gravity,
            gas_compressibility_factor=gas_z_factor,
        )

    def get_viscosity(
        self, pressure: float, temperature: float, **kwargs: typing.Any
    ) -> float:
        """
        Get the viscosity of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the viscosity (psi).
        :param temperature: The temperature at which to evaluate the viscosity (°F).
        :kwargs: Additional parameters for viscosity calculations.
        :return: The viscosity of the fluid (cP).
        """
        if self.phase == FluidPhase.WATER:
            return compute_water_viscosity(
                pressure=pressure,
                temperature=temperature,
                salinity=self.salinity or 0.0,
            )

        gas_density = kwargs.get("gas_density", None)
        if gas_density is None:
            gas_z_factor = kwargs.get("gas_compressibility_factor", None)
            if gas_z_factor is None:
                gas_z_factor = compute_gas_compressibility_factor(
                    pressure=pressure,
                    temperature=temperature,
                    gas_gravity=self.specific_gravity,
                )
            gas_density = compute_gas_density(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=self.specific_gravity,
                gas_compressibility_factor=gas_z_factor,
            )
        return compute_gas_viscosity(
            temperature=temperature,
            gas_density=gas_density,
            gas_molecular_weight=self.molecular_weight,
        )

    def get_compressibility(
        self, pressure: float, temperature: float, **kwargs: typing.Any
    ) -> float:
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
        if self.phase == FluidPhase.WATER:
            gas_free_water_fvf = kwargs.get(
                "gas_free_water_formation_volume_factor", None
            )
            if gas_free_water_fvf is None:
                gas_free_water_fvf = compute_gas_free_water_formation_volume_factor(
                    pressure=pressure, temperature=temperature
                )
                kwargs["gas_free_water_formation_volume_factor"] = gas_free_water_fvf

            return compute_water_compressibility(
                pressure=pressure,
                temperature=temperature,
                **kwargs,
                salinity=self.salinity or 0.0,
            )

        kwargs.setdefault("gas_gravity", self.specific_gravity)
        return compute_gas_compressibility(
            pressure=pressure, temperature=temperature, **kwargs
        )

    def get_formation_volume_factor(
        self, pressure: float, temperature: float, **kwargs: typing.Any
    ) -> float:
        """
        Get the formation volume factor of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the formation volume factor (psi).
        :param temperature: The temperature at which to evaluate the formation volume factor (°F).
        :kwargs: Additional parameters for formation volume factor calculations.
        :return: The formation volume factor of the fluid (bbl/STB or ft³/SCF).
        """
        if self.phase == FluidPhase.WATER:
            water_density = kwargs.get("water_density", None)
            if water_density is None:
                # Not need for gas free fvf or gas fvf, since injection water
                # is typically gas free fresh water or degassed formation water
                water_density = compute_water_density(
                    pressure=pressure,
                    temperature=temperature,
                    salinity=self.salinity or 0.0,
                )
            return compute_water_formation_volume_factor(
                salinity=self.salinity or 0.0,
                water_density=water_density,
            )

        gas_z_factor = kwargs.get("gas_compressibility_factor", None)
        if gas_z_factor is None:
            gas_z_factor = compute_gas_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=self.specific_gravity,
            )
        return compute_gas_formation_volume_factor(
            pressure=pressure,
            temperature=temperature,
            gas_compressibility_factor=gas_z_factor,
        )


@attrs.frozen
class ProducedFluid(WellFluid):
    """Properties of the fluid being produced by a well."""

    pass


WellFluidT = typing.TypeVar("WellFluidT", bound=WellFluid)


@numba.njit(cache=True)
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
    Compute k_eff for Peaceman WI using geometric mean of the two permeabilities
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
    # Oblique/unknown orientation: conservative fallback = geometric mean of all three
    return _geometric_mean((kx, ky, kz))
