import logging
import typing

import attrs
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, interp1d
from typing_extensions import Self

from bores.constants import c
from bores.errors import ComputationError, ValidationError
from bores.types import (
    NDimensionalGrid,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    TwoDimensionalGrid,
)

logger = logging.getLogger(__name__)

__all__ = ["compute_gas_pseudo_pressure", "GasPseudoPressureTable", "PVTTables"]


def compute_gas_pseudo_pressure(
    pressure: float,
    z_factor_func: typing.Callable[[float], float],
    viscosity_func: typing.Callable[[float], float],
    reference_pressure: float = 14.7,
) -> float:
    """
    Compute the gas pseudo-pressure using Al-Hussainy real-gas potential.

    The pseudo-pressure is defined as:
        m(P) = ∫[P_ref to P] (2*P' / (μ(P') * Z(P'))) dP'

    This formulation accounts for gas compressibility and non-Darcy effects,
    allowing the use of standard liquid-like flow equations for gas.

    Physical Interpretation:
        - m(P) transforms the nonlinear gas diffusivity equation into a linear form
        - At low pressure: m(P) ≈ P² (ideal gas limit)
        - At high pressure: deviations due to Z-factor and viscosity changes

    :param pressure: Current pressure (psi)
    :param z_factor_func: Function returning Z-factor at given pressure Z(P)
    :param viscosity_func: Function returning viscosity at given pressure μ(P) in cP
    :param reference_pressure: Reference pressure for integration (psi), typically 14.7
    :return: Pseudo-pressure m(P) in psi²/cP

    References:
        Al-Hussainy, R., Ramey, H.J., and Crawford, P.B. (1966).
        "The Flow of Real Gases Through Porous Media."
        JPT, May 1966, pp. 624-636.
    """
    if pressure <= 0:
        raise ValidationError(f"Pressure must be positive, got {pressure}")
    if reference_pressure <= 0:
        raise ValidationError(
            f"Reference pressure must be positive, got {reference_pressure}"
        )

    # If pressure equals reference, pseudo-pressure is zero by definition
    if abs(pressure - reference_pressure) < 1e-6:
        return 0.0

    # Define the integrand: 2*P / (μ*Z)
    def integrand(P: float) -> float:
        """Integrand for pseudo-pressure calculation."""
        # Add safety checks to prevent division by zero
        Z = z_factor_func(P)
        mu = viscosity_func(P)

        if Z <= 0 or mu <= 0:
            raise ValidationError(f"Invalid Z={Z} or μ={mu} at P={P}")

        return 2.0 * P / (mu * Z)

    # Perform numerical integration
    # Use higher accuracy for gas (epsabs, epsrel)
    try:
        if pressure > reference_pressure:
            result, error = quad(
                integrand,
                reference_pressure,
                pressure,
                epsabs=1e-8,  # Absolute error tolerance
                epsrel=1e-6,  # Relative error tolerance
                limit=100,  # Maximum number of subintervals
            )
            return float(result)
        else:
            # Integrate backwards and negate
            result, error = quad(
                integrand,
                pressure,
                reference_pressure,
                epsabs=1e-8,
                epsrel=1e-6,
                limit=100,
            )
            return -float(result)
    except Exception as exc:
        raise ComputationError(
            f"Failed to compute pseudo-pressure at P={pressure} psi: {exc}"
        )


class GasPseudoPressureTable:
    """
    Pre-computed gas pseudo-pressure table for fast lookup during simulation.
    """

    def __init__(
        self,
        z_factor_func: typing.Callable[[float], float],
        viscosity_func: typing.Callable[[float], float],
        pressure_range: typing.Optional[typing.Tuple[float, float]] = None,
        points: typing.Optional[int] = None,
        reference_pressure: typing.Optional[float] = None,
    ):
        """
        Build pseudo-pressure lookup table.

        :param z_factor_func: Z-factor correlation Z(P)
        :param viscosity_func: Gas viscosity correlation μ(P)
        :param pressure_range: (P_min, P_max) for table. Defaults to the (c.MIN_VALID_PRESSURE, c.MAX_VALID_PRESSURE)
            if not provided.
        :param points: Number of points in table. typically 500-2000 for good accuracy. The higher the number, the more accurate the interpolation.
            1000 points is a good balance between accuracy and memory usage.
            2000 points may be used for very high accuracy at the cost of memory.
            500 points may be used for low memory usage at the cost of accuracy.
        :param reference_pressure: Reference pressure (psi)
        """
        self.reference_pressure = typing.cast(
            float, reference_pressure or c.MIN_VALID_PRESSURE
        )
        self.z_factor_func = z_factor_func
        self.viscosity_func = viscosity_func

        # Create pressure grid (log-spaced for better resolution at low P)
        min_pressure, max_pressure = pressure_range or (
            c.MIN_VALID_PRESSURE,
            c.MAX_VALID_PRESSURE,
        )
        points = typing.cast(int, points or c.GAS_PSEUDO_PRESSURE_POINTS)
        self.pressures = np.logspace(
            np.log10(min_pressure), np.log10(max_pressure), points
        )

        # Compute pseudo-pressure at each point
        logger.debug(f"Building pseudo-pressure table with {points} points...")
        self.pseudo_pressures = np.zeros(points)
        for i, pressure in enumerate(self.pressures):
            self.pseudo_pressures[i] = compute_gas_pseudo_pressure(
                pressure=pressure,
                z_factor_func=z_factor_func,
                viscosity_func=viscosity_func,
                reference_pressure=self.reference_pressure,
            )
        logger.debug("Pseudo-pressure table computation complete.")

        # Build cubic spline interpolator's for fast lookup
        self.interpolator = interp1d(
            self.pressures,
            self.pseudo_pressures,
            kind="cubic",
            bounds_error=True,
            fill_value=(self.pseudo_pressures[0], self.pseudo_pressures[-1]),  # type: ignore
        )
        self.inverse_interpolator = interp1d(
            self.pseudo_pressures,
            self.pressures,
            kind="cubic",
            bounds_error=True,
            fill_value=(self.pressures[0], self.pressures[-1]),  # type: ignore
        )
        logger.debug(
            f"Pseudo-pressure table built: P ∈ [{min_pressure:.1f}, {max_pressure:.1f}] psi"
        )

    def __call__(self, pressure: float) -> float:
        """
        Fast lookup of pseudo-pressure via interpolation.

        :param pressure: Pressure (psi)
        :return: Pseudo-pressure m(P) (psi²/cP)
        """
        return float(self.interpolator(pressure))

    def gradient(self, pressure: float) -> float:
        """
        Compute dm/dP = 2P/(μ*Z) for use in well models.

        :param pressure: Pressure (psi)
        :return: dm/dP (psi/cP)
        """
        Z = self.z_factor_func(pressure)
        mu = self.viscosity_func(pressure)
        return 2.0 * pressure / (mu * Z)


QueryType = typing.Union[NDimensionalGrid, list, float, np.floating]
InterpolationMethod = typing.Literal["linear", "cubic", "quintic"]

__interpolation_degree_map = {"linear": 1, "cubic": 3, "quintic": 5}


@attrs.frozen
class PVTTables:
    """
    PVT (Pressure-Volume-Temperature) tables for fluid properties.

    Provides interpolation methods for various fluid properties as functions
    of pressure, temperature, and optionally other parameters like salinity.
    """

    pressures: OneDimensionalGrid
    """One-dimensional grid of pressures (psi)."""

    temperatures: OneDimensionalGrid
    """One-dimensional grid of temperatures (°F)."""

    bubble_point_pressures: typing.Optional[
        typing.Union[OneDimensionalGrid, TwoDimensionalGrid]
    ] = None
    """
    Bubble point pressures for dead oil (psi).
    
    Can be either:
    - 1D array as function of temperature: Pb(T) for single composition
    - 2D array as function of Rs and temperature: Pb(Rs, T) for varying composition
    
    If 1D: shape (n_temperatures,)
    If 2D: shape (n_solution_gor, n_temperatures)
    
    Note: For black oil models with varying Rs, provide 2D table.
    """

    interpolation_method: InterpolationMethod = "linear"
    """Interpolation method: 'linear', 'cubic', or 'quintic'."""

    validate_tables: bool = True
    """If True, perform physical consistency checks on table data."""

    warn_on_extrapolation: bool = False
    """If True, log warnings when extrapolating beyond table bounds."""

    salinities: typing.Optional[OneDimensionalGrid] = None
    """One-dimensional grid of salinities (ppm NaCl) for salinity-dependent properties."""

    oil_viscosities: typing.Optional[TwoDimensionalGrid] = None
    """Oil viscosity table (cP) - shape: (n_pressures, n_temperatures)."""

    oil_compressibilities: typing.Optional[TwoDimensionalGrid] = None
    """Oil compressibility table (psi⁻¹) - shape: (n_pressures, n_temperatures)."""

    oil_specific_gravities: typing.Optional[TwoDimensionalGrid] = None
    """Oil specific gravity table (dimensionless) - shape: (n_pressures, n_temperatures). Usually constant for a given oil."""

    oil_api_gravities: typing.Optional[TwoDimensionalGrid] = None
    """Oil API gravity table (°API) - shape: (n_pressures, n_temperatures). Usually constant for a given oil."""

    oil_densities: typing.Optional[TwoDimensionalGrid] = None
    """Oil density table (lbm/ft³) - shape: (n_pressures, n_temperatures)."""

    oil_formation_volume_factors: typing.Optional[TwoDimensionalGrid] = None
    """Oil formation volume factor table (bbl/STB) - shape: (n_pressures, n_temperatures)."""

    water_bubble_point_pressures: typing.Optional[TwoDimensionalGrid] = None
    """Water bubble point pressure table (psi) - shape: (n_pressures, n_temperatures)."""

    water_viscosities: typing.Optional[TwoDimensionalGrid] = None
    """Water viscosity table (cP) - shape: (n_pressures, n_temperatures)."""

    water_compressibilities: typing.Optional[TwoDimensionalGrid] = None
    """Water compressibility table (psi⁻¹) - shape: (n_pressures, n_temperatures)."""

    water_densities: typing.Optional[TwoDimensionalGrid] = None
    """Water density table (lbm/ft³) - shape: (n_pressures, n_temperatures)."""

    water_formation_volume_factors: typing.Optional[TwoDimensionalGrid] = None
    """Water formation volume factor table (bbl/STB) - shape: (n_pressures, n_temperatures)."""

    water_salinities: typing.Optional[TwoDimensionalGrid] = None
    """Water salinity table (ppm NaCl) - shape: (n_pressures, n_temperatures). Usually constant for a given water type."""

    gas_viscosities: typing.Optional[TwoDimensionalGrid] = None
    """Gas viscosity table (cP) - shape: (n_pressures, n_temperatures)."""

    gas_compressibilities: typing.Optional[TwoDimensionalGrid] = None
    """Gas compressibility table (psi⁻¹) - shape: (n_pressures, n_temperatures)."""

    gas_gravities: typing.Optional[TwoDimensionalGrid] = None
    """Gas gravity table (dimensionless) - shape: (n_pressures, n_temperatures). Usually constant for a given gas."""

    gas_molecular_weights: typing.Optional[TwoDimensionalGrid] = None
    """Gas molecular weight table (g/mol) - shape: (n_pressures, n_temperatures). Usually constant for a given gas."""

    gas_densities: typing.Optional[TwoDimensionalGrid] = None
    """Gas density table (lbm/ft³) - shape: (n_pressures, n_temperatures)."""

    gas_formation_volume_factors: typing.Optional[TwoDimensionalGrid] = None
    """Gas formation volume factor table (ft³/SCF) - shape: (n_pressures, n_temperatures)."""

    gas_compressibility_factors: typing.Optional[TwoDimensionalGrid] = None
    """Gas z-factor (compressibility factor) table (dimensionless) - shape: (n_pressures, n_temperatures)."""

    solution_gas_to_oil_ratios: typing.Optional[TwoDimensionalGrid] = None
    """
    Solution gas-to-oil ratio table (SCF/STB).
    
    For saturated conditions (P ≤ Pb): Rs varies with pressure
    For undersaturated conditions (P > Pb): Rs remains constant at Rsb
    
    Shape: (n_pressures, n_temperatures)
    """

    gas_solubilities_in_water: typing.Optional[ThreeDimensionalGrid] = None
    """Gas solubility in water table (SCF/STB) - shape: (n_pressures, n_temperatures, n_salinities)."""

    _interpolators: dict = attrs.field(init=False, factory=dict, repr=False)
    """Cache of pre-computed interpolators for fast lookup."""

    _extrapolation_bounds: dict = attrs.field(init=False, factory=dict, repr=False)
    """Cache of table bounds for extrapolation detection."""

    def __attrs_post_init__(self):
        """Validate dimensions and build interpolators after initialization."""
        self._validate_grids()

        if self.validate_tables:
            self._check_physical_consistency()

        # Store bounds for extrapolation detection
        object.__setattr__(
            self,
            "_extrapolation_bounds",
            {
                "pressure": (self.pressures[0], self.pressures[-1]),
                "temperature": (self.temperatures[0], self.temperatures[-1]),
            },
        )
        if self.salinities is not None:
            self._extrapolation_bounds["salinity"] = (
                self.salinities[0],
                self.salinities[-1],
            )

        # Pre-build all interpolators
        self.build_interpolators()
        logger.info(
            f"PVTTables initialized: P ∈ [{self.pressures[0]:.1f}, {self.pressures[-1]:.1f}] psi, "
            f"T ∈ [{self.temperatures[0]:.1f}, {self.temperatures[-1]:.1f}] °F, "
            f"method={self.interpolation_method}"
        )

    def _validate_grids(self):
        """Validate grid dimensions and monotonicity."""
        # Check dimensionality
        if self.pressures.ndim != 1:
            raise ValueError("`pressures` must be 1-dimensional")
        if self.temperatures.ndim != 1:
            raise ValueError("`temperatures` must be 1-dimensional")

        # Check monotonicity (critical for interpolation)
        if not np.all(np.diff(self.pressures) > 0):
            raise ValueError("`pressures` must be strictly monotonically increasing")
        if not np.all(np.diff(self.temperatures) > 0):
            raise ValueError("`temperatures` must be strictly monotonically increasing")

        # Validate bubble point pressures
        if self.bubble_point_pressures is not None:
            if self.bubble_point_pressures.ndim == 1:
                # 1D: Pb(T) for single composition
                if len(self.bubble_point_pressures) != len(self.temperatures):
                    raise ValueError(
                        f"1D `bubble_point_pressures` length ({len(self.bubble_point_pressures)}) "
                        f"must match `temperatures` length ({len(self.temperatures)})"
                    )
            elif self.bubble_point_pressures.ndim == 2:
                # 2D: Pb(Rs, T) for varying composition
                if self.solution_gas_to_oil_ratios is None:
                    raise ValueError(
                        "`solution_gas_to_oil_ratios` grid required for 2D bubble_point_pressures"
                    )

                if self.solution_gas_to_oil_ratios.ndim != 1:
                    raise ValueError(
                        "`solution_gas_to_oil_ratios` must be 1-dimensional"
                    )

                if not np.all(np.diff(self.solution_gas_to_oil_ratios) > 0):
                    raise ValueError(
                        "solution_gas_to_oil_ratios must be strictly monotonically increasing"
                    )

                n_rs, n_t = self.bubble_point_pressures.shape  # type: ignore
                if n_rs != len(self.solution_gas_to_oil_ratios):
                    raise ValueError(
                        f"`bubble_point_pressures` rows ({n_rs}) must match "
                        f"solution_gas_to_oil_ratios length ({len(self.solution_gas_to_oil_ratios)})"
                    )
                if n_t != len(self.temperatures):
                    raise ValueError(
                        f"`bubble_point_pressures` columns ({n_t}) must match "
                        f"temperatures length ({len(self.temperatures)})"
                    )
            else:
                raise ValueError("`bubble_point_pressures` must be 1D or 2D")

        if self.salinities is not None:
            if self.salinities.ndim != 1:
                raise ValueError("salinities must be 1-dimensional")
            if not np.all(np.diff(self.salinities) > 0):
                raise ValueError("salinities must be strictly monotonically increasing")

    def _check_physical_consistency(self):
        """Perform physical consistency checks on table data."""
        n_p, n_t = len(self.pressures), len(self.temperatures)

        # Check 2D table shapes
        tables_2d = {
            "oil_viscosities": self.oil_viscosities,
            "oil_densities": self.oil_densities,
            "water_viscosities": self.water_viscosities,
            "water_densities": self.water_densities,
            "gas_viscosities": self.gas_viscosities,
            "gas_densities": self.gas_densities,
        }

        for name, table in tables_2d.items():
            if table is not None:
                if table.shape != (n_p, n_t):
                    raise ValueError(
                        f"{name} shape {table.shape} must match (n_pressures={n_p}, n_temperatures={n_t})"
                    )
                # Check for NaN or inf
                if not np.all(np.isfinite(table)):
                    raise ValueError(f"{name} contains NaN or infinite values")

        # Physical bounds checks
        if self.oil_viscosities is not None:
            if np.any(self.oil_viscosities <= 0):
                raise ValueError("oil_viscosities must be positive")

        if self.gas_viscosities is not None:
            if np.any(self.gas_viscosities <= 0):
                raise ValueError("gas_viscosities must be positive")

        if self.water_viscosities is not None:
            if np.any(self.water_viscosities <= 0):
                raise ValueError("water_viscosities must be positive")

        # Density checks
        if self.oil_densities is not None:
            if np.any(self.oil_densities <= 0):
                raise ValueError("oil_densities must be positive")

        if self.gas_densities is not None:
            if np.any(self.gas_densities <= 0):
                raise ValueError("gas_densities must be positive")
            # Gas should be less dense than oil
            if self.oil_densities is not None:
                if np.any(self.gas_densities > self.oil_densities):
                    logger.warning(
                        "gas_densities exceed oil_densities at some conditions"
                    )

        # Formation volume factor checks
        if self.oil_formation_volume_factors is not None:
            if np.any(self.oil_formation_volume_factors <= 0):
                raise ValueError("oil_formation_volume_factors must be positive")

        logger.debug("Physical consistency checks passed")

    def _check_extrapolation(self, pressure: QueryType, temperature: QueryType):
        """Log warning if extrapolating beyond table bounds."""
        if not self.warn_on_extrapolation:
            return

        p_arr = np.atleast_1d(pressure)
        t_arr = np.atleast_1d(temperature)

        p_min, p_max = self._extrapolation_bounds["pressure"]
        t_min, t_max = self._extrapolation_bounds["temperature"]

        if np.any(p_arr < p_min) or np.any(p_arr > p_max):
            logger.warning(
                f"Pressure extrapolation: queried P ∈ [{p_arr.min():.1f}, {p_arr.max():.1f}] psi, "
                f"table range [{p_min:.1f}, {p_max:.1f}] psi"
            )

        if np.any(t_arr < t_min) or np.any(t_arr > t_max):
            logger.warning(
                f"Temperature extrapolation: queried T ∈ [{t_arr.min():.1f}, {t_arr.max():.1f}] °F, "
                f"table range [{t_min:.1f}, {t_max:.1f}] °F"
            )

    def build_interpolators(self):
        """Build interpolators for all provided properties for maximum performance."""
        # Build bubble point pressure interpolator (1D or 2D)
        if self.bubble_point_pressures is not None:
            if self.bubble_point_pressures.ndim == 1:
                # 1D: Pb(T)
                kind_1d = "cubic" if self.interpolation_method == "cubic" else "linear"
                self._interpolators["bubble_point_pressure"] = interp1d(
                    x=self.temperatures,
                    y=self.bubble_point_pressures,
                    kind=kind_1d,
                    fill_value="extrapolate",  # type: ignore
                    assume_sorted=True,
                )
            else:
                # 2D: Pb(Rs, T)
                k = __interpolation_degree_map[self.interpolation_method]
                self._interpolators["bubble_point_pressure"] = RectBivariateSpline(
                    x=self.solution_gas_oil_ratios,  # type: ignore
                    y=self.temperatures,
                    z=self.bubble_point_pressures,
                    kx=k,
                    ky=k,
                    s=0,
                )

        # Map interpolation method to spline degree
        k = __interpolation_degree_map[self.interpolation_method]

        # Build 2D (Pressure-Temperature) interpolators for all provided properties
        property_map = {
            "oil_viscosity": self.oil_viscosities,
            "oil_compressibility": self.oil_compressibilities,
            "oil_specific_gravity": self.oil_specific_gravities,
            "oil_api_gravity": self.oil_api_gravities,
            "oil_density": self.oil_densities,
            "oil_formation_volume_factor": self.oil_formation_volume_factors,
            "water_bubble_point_pressure": self.water_bubble_point_pressures,
            "water_viscosity": self.water_viscosities,
            "water_compressibility": self.water_compressibilities,
            "water_density": self.water_densities,
            "water_formation_volume_factor": self.water_formation_volume_factors,
            "water_salinity": self.water_salinities,
            "gas_viscosity": self.gas_viscosities,
            "gas_compressibility": self.gas_compressibilities,
            "gas_gravity": self.gas_gravities,
            "gas_molecular_weight": self.gas_molecular_weights,
            "gas_density": self.gas_densities,
            "gas_formation_volume_factor": self.gas_formation_volume_factors,
            "gas_compressibility_factor": self.gas_compressibility_factors,
            "solution_gas_to_oil_ratio": self.solution_gas_to_oil_ratios,
        }
        for name, data in property_map.items():
            if data is not None:
                # RectBivariateSpline is significantly faster than RegularGridInterpolator
                # for 2D data, especially for vectorized queries (10-50x faster)
                self._interpolators[name] = RectBivariateSpline(
                    x=self.pressures,
                    y=self.temperatures,
                    z=data,
                    kx=k,
                    ky=k,
                    s=0,  # No smoothing - Interpolate exactly through data points
                )

        # Build 3D (Pressure-Temperature-Salinity) interpolator for salinity-dependent properties
        if self.gas_solubilities_in_water is not None and self.salinities is not None:
            # For 3D, use RegularGridInterpolator (RectBivariateSpline is 2D only)
            method_3d = "linear" if self.interpolation_method == "linear" else "cubic"
            self._interpolators["gas_solubility_in_water"] = RegularGridInterpolator(
                points=(self.pressures, self.temperatures, self.salinities),
                values=self.gas_solubilities_in_water,
                method=method_3d,
                bounds_error=False,
                fill_value=None,  # type: ignore  # Enables extrapolation
            )

        logger.debug(f"Built {len(self._interpolators)} interpolators")

    def Interpolate(
        self, name: str, value: QueryType
    ) -> typing.Union[float, np.ndarray, None]:
        """Fast lookup using built 1D interpolator."""
        interp = self._interpolators.get(name)
        if interp is None:
            return None

        # interp1d handles both scalars and arrays efficiently
        result = interp(value)
        return float(result) if np.isscalar(value) else result

    def InterpolatePT(
        self, name: str, pressure: QueryType, temperature: QueryType
    ) -> typing.Union[float, np.ndarray, None]:
        """Fast lookup using built Pressure-Temperature interpolator."""
        interp = self._interpolators.get(name)
        if interp is None:
            return None

        # Convert to arrays
        p = np.atleast_1d(pressure)
        t = np.atleast_1d(temperature)

        # Ensure compatible shapes
        if p.shape != t.shape:
            if p.size == 1:
                p = np.full_like(t, p[0])
            elif t.size == 1:
                t = np.full_like(p, t[0])
            else:
                raise ValueError("pressure and temperature must have compatible shapes")

        # RectBivariateSpline.ev is optimized for vectorized evaluation
        result = interp.ev(p, t)

        # Return scalar if inputs were scalar
        return float(result) if result.size == 1 else result

    def InterpolatePTS(
        self,
        name: str,
        pressure: QueryType,
        temperature: QueryType,
        salinity: QueryType,
    ) -> typing.Union[float, np.ndarray, None]:
        """Fast lookup using built Pressure-Temperature-Salinity interpolator."""
        interp = self._interpolators.get(name)
        if interp is None:
            return None

        # Convert to arrays and broadcast
        p = np.atleast_1d(pressure)
        t = np.atleast_1d(temperature)
        s = np.atleast_1d(salinity)
        p, t, s = np.broadcast_arrays(p, t, s)

        # Stack and evaluate
        points = np.column_stack([p.ravel(), t.ravel(), s.ravel()])
        result = interp(points).reshape(p.shape)

        return float(result) if result.size == 1 else result

    def oil_bubble_point_pressure(
        self, temperature: QueryType, solution_gor: typing.Optional[QueryType] = None
    ) -> typing.Union[float, np.ndarray]:
        """
        Get bubble point pressure.

        For 1D tables: Pb(T)
        For 2D tables: Pb(Rs, T) - solution_gor must be provided

        :param temperature: Temperature(s) in °F
        :param solution_gor: Solution gas-oil ratio(s) in SCF/STB (required for 2D tables)
        :return: Bubble point pressure(s) in psi
        """
        interp = self._interpolators.get("bubble_point_pressure")
        if interp is None:
            raise ValueError("Bubble point pressure table not provided")

        if self.bubble_point_pressures.ndim == 1:  # type: ignore
            # 1D: Pb(T)
            if solution_gor is not None:
                logger.warning(
                    "`solution_gor` provided but `bubble_point_pressures` is 1D - ignoring"
                )
            result = interp(temperature)
            return float(result) if np.isscalar(temperature) else result

        # 2D: Pb(Rs, T)
        if solution_gor is None:
            raise ValueError(
                "`solution_gor` required for 2D `bubble_point_pressures` table"
            )

        rs = np.atleast_1d(solution_gor)
        t = np.atleast_1d(temperature)

        if rs.shape != t.shape:
            if rs.size == 1:
                rs = np.full_like(t, rs[0])
            elif t.size == 1:
                t = np.full_like(rs, t[0])
            else:
                raise ValueError(
                    "`solution_gor` and `temperature` must have compatible shapes"
                )

        result = interp.ev(rs, t)
        return float(result) if result.size == 1 else result

    def is_saturated(
        self,
        pressure: QueryType,
        temperature: QueryType,
        solution_gor: typing.Optional[QueryType] = None,
    ) -> typing.Union[bool, np.ndarray]:
        """
        Determine if conditions are saturated (P ≤ Pb) or undersaturated (P > Pb).

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param solution_gor: Solution gas-oil ratio(s) in SCF/STB (for 2D bubble point tables)
        :return: Boolean or array: True if saturated, False if undersaturated
        """
        pb = self.oil_bubble_point_pressure(temperature, solution_gor)

        p_arr = np.atleast_1d(pressure)
        pb_arr = np.atleast_1d(pb)

        result = p_arr <= pb_arr
        return bool(result) if result.size == 1 else result

    def oil_viscosity(
        self,
        pressure: QueryType,
        temperature: QueryType,
        solution_gor: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get oil viscosity.

        For saturated conditions (P ≤ Pb): Interpolate directly from table
        For undersaturated conditions (P > Pb): Use correlation or special handling

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param solution_gor: Solution gas-oil ratio(s) in SCF/STB (for 2D bubble point)
        :return: Oil viscosity in cP
        """
        if self.oil_viscosities is None:
            return None

        # Get bubble point pressure
        bubble_point_pressure = self.oil_bubble_point_pressure(
            temperature, solution_gor
        )

        # Convert to arrays for vectorized operations
        p = np.atleast_1d(pressure)
        t = np.atleast_1d(temperature)
        pb = np.atleast_1d(bubble_point_pressure)

        # Broadcast to compatible shapes
        p, t, pb = np.broadcast_arrays(p, t, pb)

        # Interpolate viscosity
        result = np.zeros_like(p, dtype=float)

        # Saturated conditions: P ≤ Pb - use table directly
        saturated = p <= pb
        if np.any(saturated):
            result[saturated] = self.InterpolatePT(
                "oil_viscosity", p[saturated], t[saturated]
            )  # type: ignore

        # Undersaturated conditions: P > Pb
        undersaturated = ~saturated
        if np.any(undersaturated):
            mu_b = self.InterpolatePT(
                "oil_viscosity", pb[undersaturated], t[undersaturated]
            )
            co = self.InterpolatePT(
                name="oil_compressibility",
                pressure=pb[undersaturated],
                temperature=t[undersaturated],
            )
            delta_p = p[undersaturated] - pb[undersaturated]
            # μ_o increases slightly with pressure above Pb
            # μ_o = μ_ob * exp(c_o * (P - Pb))
            result[undersaturated] = mu_b * np.exp(co * delta_p)
        return float(result) if result.size == 1 else result

    def oil_compressibility(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil compressibility at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("oil_compressibility", pressure, temperature)

    def oil_specific_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil specific gravity at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("oil_specific_gravity", pressure, temperature)

    def oil_api_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil API gravity at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("oil_api_gravity", pressure, temperature)

    def oil_density(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil density at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("oil_density", pressure, temperature)

    def oil_formation_volume_factor(
        self,
        pressure: QueryType,
        temperature: QueryType,
        solution_gor: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get oil formation volume factor.

        For saturated conditions (P ≤ Pb): Bo increases as pressure increases
        For undersaturated conditions (P > Pb): Bo decreases as pressure increases (compression)

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param solution_gor: Solution gas-oil ratio(s) in SCF/STB (for 2D bubble point)
        :return Oil formation volume factor in bbl/STB
        """
        if self.oil_formation_volume_factors is None:
            return None

        # Get bubble point pressure
        bubble_point_pressure = self.oil_bubble_point_pressure(
            temperature, solution_gor
        )

        # Convert to arrays
        p = np.atleast_1d(pressure)
        t = np.atleast_1d(temperature)
        pb = np.atleast_1d(bubble_point_pressure)
        p, t, pb = np.broadcast_arrays(p, t, pb)

        result = np.zeros_like(p, dtype=float)

        # Saturated: P ≤ Pb
        saturated = p <= pb
        if np.any(saturated):
            result[saturated] = self.InterpolatePT(
                "oil_formation_volume_factor", p[saturated], t[saturated]
            )  # type: ignore

        # Undersaturated: P > Pb
        # Bo(P) = Bob * exp(-co * (P - Pb))
        undersaturated = ~saturated
        if np.any(undersaturated):
            # Get Bo at bubble point
            bob = self.InterpolatePT(
                "oil_formation_volume_factor", pb[undersaturated], t[undersaturated]
            )  # type: ignore

            # Get compressibility if available
            if self.oil_compressibilities is not None:
                co = self.InterpolatePT(
                    "oil_compressibility", pb[undersaturated], t[undersaturated]
                )  # type: ignore
                assert co is not None
                # Apply compression: Bo = Bob * exp(-co * (P - Pb))
                result[undersaturated] = bob * np.exp(
                    -co * (p[undersaturated] - pb[undersaturated])
                )
            else:
                # Without compressibility, use Bob as approximation
                result[undersaturated] = bob
                if self.warn_on_extrapolation:
                    logger.debug(
                        "Undersaturated Bo: using bubble point value (no compressibility data)"
                    )

        return float(result) if result.size == 1 else result

    def solution_gas_to_oil_ratio(
        self,
        pressure: QueryType,
        temperature: QueryType,
        solution_gor: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get solution gas-to-oil ratio.

        For saturated conditions (P ≤ Pb): Rs varies with pressure (more gas dissolves)
        For undersaturated conditions (P > Pb): Rs = Rsb (constant, no more gas dissolves)

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param solution_gor: Solution gas-oil ratio(s) in SCF/STB (for 2D bubble point)
        :return: Solution gas-to-oil ratio in SCF/STB
        """
        if self.solution_gas_to_oil_ratios is None:
            return None

        # Get bubble point pressure
        bubble_point_pressure = self.oil_bubble_point_pressure(
            temperature, solution_gor
        )

        # Convert to arrays
        p = np.atleast_1d(pressure)
        t = np.atleast_1d(temperature)
        pb = np.atleast_1d(bubble_point_pressure)
        p, t, pb = np.broadcast_arrays(p, t, pb)

        result = np.zeros_like(p, dtype=float)

        # Saturated: P ≤ Pb - Rs varies with P
        saturated = p <= pb
        if np.any(saturated):
            result[saturated] = self.InterpolatePT(
                "solution_gas_to_oil_ratio", p[saturated], t[saturated]
            )  # type: ignore

        # Undersaturated: P > Pb - Rs = Rsb (constant)
        undersaturated = ~saturated
        if np.any(undersaturated):
            # Use Rs at bubble point
            result[undersaturated] = self.InterpolatePT(
                "solution_gas_to_oil_ratio", pb[undersaturated], t[undersaturated]
            )  # type: ignore

        return float(result) if result.size == 1 else result

    def water_bubble_point_pressure(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water bubble point pressure at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("water_bubble_point_pressure", pressure, temperature)

    def water_viscosity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water viscosity at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("water_viscosity", pressure, temperature)

    def water_compressibility(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water compressibility at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("water_compressibility", pressure, temperature)

    def water_density(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water density at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("water_density", pressure, temperature)

    def water_formation_volume_factor(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water formation volume factor at given pressure(s) and temperature(s)."""
        return self.InterpolatePT(
            "water_formation_volume_factor", pressure, temperature
        )

    def water_salinity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water salinity at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("water_salinity", pressure, temperature)

    def gas_viscosity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas viscosity at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("gas_viscosity", pressure, temperature)

    def gas_compressibility(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas compressibility at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("gas_compressibility", pressure, temperature)

    def gas_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas gravity at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("gas_gravity", pressure, temperature)

    def gas_molecular_weight(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas molecular weight at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("gas_molecular_weight", pressure, temperature)

    def gas_density(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas density at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("gas_density", pressure, temperature)

    def gas_formation_volume_factor(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas formation volume factor at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("gas_formation_volume_factor", pressure, temperature)

    def gas_compressibility_factor(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas compressibility factor at given pressure(s) and temperature(s)."""
        return self.InterpolatePT("gas_compressibility_factor", pressure, temperature)

    def gas_solubility_in_water(
        self, pressure: QueryType, temperature: QueryType, salinity: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas solubility in water at given pressure(s), temperature(s), and salinity(ies)."""
        return self.InterpolatePTS(
            "gas_solubility_in_water", pressure, temperature, salinity
        )

    def update(self, **kwargs: typing.Any) -> Self:
        """Get a new `PVTTables` instance with updated fields."""
        return attrs.evolve(self, **kwargs)


def build_pvt_tables(
    pressures: OneDimensionalGrid,
    temperatures: OneDimensionalGrid,
    bubble_point_pressures: typing.Optional[
        typing.Union[OneDimensionalGrid, TwoDimensionalGrid]
    ] = None,
    validate_tables: bool = True,
    interpolation_method: InterpolationMethod = "cubic",
    warn_on_extrapolation: bool = True,
    **kwargs: typing.Any,
) -> PVTTables:
    """
    Build PVT tables using empirical/analytical correlations and provided data.

    :param pressures: 1D array of pressures (psi)
    :param temperatures: 1D array of temperatures (°F)
    :param bubble_point_pressures: 1D or 2D array of bubble point pressures (psi)
    :param validate_tables: Whether to perform physical consistency checks
    :param interpolation_method: Interpolation method: 'linear', 'cubic', or 'quintic'
    :param warn_on_extrapolation: Whether to log warnings on extrapolation
    :param kwargs: Additional property tables as 2D arrays
    :return: `PVTTables` instance
    """
    return PVTTables(
        pressures=pressures, 
        temperatures=temperatures,
        bubble_point_pressures=bubble_point_pressures,
        validate_tables=validate_tables,
        interpolation_method=interpolation_method,
        warn_on_extrapolation=warn_on_extrapolation, 
        **kwargs
    )
