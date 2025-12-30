import logging
import typing
import warnings

import attrs
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, interp1d
from typing_extensions import Self

from bores.constants import c
from bores.errors import ComputationError, ValidationError
from bores.grids.pvt import (
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_molecular_weight_grid,
    build_gas_solubility_in_water_grid,
    build_gas_viscosity_grid,
    build_live_oil_density_grid,
    build_oil_api_gravity_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_compressibility_grid,
    build_oil_formation_volume_factor_grid,
    build_oil_viscosity_grid,
    build_solution_gas_to_oil_ratio_grid,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from bores.pvt.core import compute_gas_gravity
from bores.types import (
    NDimensionalGrid,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    TwoDimensionalGrid,
)

logger = logging.getLogger(__name__)

__all__ = [
    "compute_gas_pseudo_pressure",
    "GasPseudoPressureTable",
    "PVTTables",
    "build_pvt_tables",
]


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

    # Base grids for interpolation

    pressures: OneDimensionalGrid
    """One-dimensional grid of pressures (psi)."""

    temperatures: OneDimensionalGrid
    """One-dimensional grid of temperatures (°F)."""

    salinities: typing.Optional[OneDimensionalGrid] = None
    """One-dimensional grid of salinities (ppm NaCl) for salinity-dependent properties."""

    solution_gas_oil_ratios: typing.Optional[OneDimensionalGrid] = None
    """
    One-dimensional grid of solution gas-oil ratios (SCF/STB) for varying composition properties.

    Used along side bubble point pressures when provided as 2D tables.
    """

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

    # Table options

    interpolation_method: InterpolationMethod = "linear"
    """Interpolation method: 'linear', 'cubic', or 'quintic'."""

    validate_tables: bool = True
    """If True, perform physical consistency checks on table data."""

    warn_on_extrapolation: bool = False
    """If True, log warnings when extrapolating beyond table bounds."""

    # Oil properties tables

    oil_viscosity_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil viscosity table (cP) - shape: (n_pressures, n_temperatures)."""

    oil_compressibility_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil compressibility table (psi⁻¹) - shape: (n_pressures, n_temperatures)."""

    oil_specific_gravity_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil specific gravity table (dimensionless) - shape: (n_pressures, n_temperatures). Usually constant for a given oil."""

    oil_api_gravity_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil API gravity table (°API) - shape: (n_pressures, n_temperatures). Usually constant for a given oil."""

    oil_density_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil density table (lbm/ft³) - shape: (n_pressures, n_temperatures)."""

    oil_formation_volume_factor_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil formation volume factor table (bbl/STB) - shape: (n_pressures, n_temperatures)."""

    solution_gas_to_oil_ratio_table: typing.Optional[TwoDimensionalGrid] = None
    """
    Solution gas-to-oil ratio table (SCF/STB).
    
    For saturated conditions (P ≤ Pb): Rs varies with pressure
    For undersaturated conditions (P > Pb): Rs remains constant at Rsb
    
    Shape: (n_pressures, n_temperatures)
    """

    # Water properties tables

    water_bubble_point_pressure_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water bubble point pressure table (psi) - shape: (n_pressures, n_temperatures)."""

    water_viscosity_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water viscosity table (cP) - shape: (n_pressures, n_temperatures, n_salinities)."""

    water_compressibility_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water compressibility table (psi⁻¹) - shape: (n_pressures, n_temperatures)."""

    water_density_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water density table (lbm/ft³) - shape: (n_pressures, n_temperatures)."""

    water_formation_volume_factor_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water formation volume factor table (bbl/STB) - shape: (n_pressures, n_temperatures)."""

    # Gas properties tables

    gas_viscosity_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas viscosity table (cP) - shape: (n_pressures, n_temperatures)."""

    gas_compressibility_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas compressibility table (psi⁻¹) - shape: (n_pressures, n_temperatures)."""

    gas_gravity_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas gravity table (dimensionless) - shape: (n_pressures, n_temperatures). Usually constant for a given gas."""

    gas_molecular_weight_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas molecular weight table (g/mol) - shape: (n_pressures, n_temperatures). Usually constant for a given gas."""

    gas_density_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas density table (lbm/ft³) - shape: (n_pressures, n_temperatures)."""

    gas_formation_volume_factor_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas formation volume factor table (ft³/SCF) - shape: (n_pressures, n_temperatures)."""

    gas_compressibility_factor_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas z-factor (compressibility factor) table (dimensionless) - shape: (n_pressures, n_temperatures)."""

    gas_solubility_in_water_table: typing.Optional[ThreeDimensionalGrid] = None
    """Gas solubility in water table (SCF/STB) - shape: (n_pressures, n_temperatures, n_salinities)."""

    # Caches for performance

    _interpolators: dict = attrs.field(init=False, factory=dict, repr=False)
    """Cache of pre-computed interpolators for fast lookup."""

    _extrapolation_bounds: dict = attrs.field(init=False, factory=dict, repr=False)
    """Cache of table bounds for extrapolation detection."""

    def __attrs_post_init__(self):
        """Validate dimensions and build interpolators after initialization."""
        self.validate_grids()

        if self.validate_tables:
            self.check_physical_consistency()

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

    def validate_grids(self):
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
                if self.solution_gas_to_oil_ratio_table is None:
                    raise ValueError(
                        "`solution_gas_to_oil_ratio_table` grid required for 2D bubble_point_pressures"
                    )

                if self.solution_gas_to_oil_ratio_table.ndim != 1:
                    raise ValueError(
                        "`solution_gas_to_oil_ratio_table` must be 1-dimensional"
                    )

                if not np.all(np.diff(self.solution_gas_to_oil_ratio_table) > 0):
                    raise ValueError(
                        "solution_gas_to_oil_ratio_table must be strictly monotonically increasing"
                    )

                n_rs, n_t = self.bubble_point_pressures.shape  # type: ignore
                if n_rs != len(self.solution_gas_to_oil_ratio_table):
                    raise ValueError(
                        f"`bubble_point_pressures` rows ({n_rs}) must match "
                        f"solution_gas_to_oil_ratio_table length ({len(self.solution_gas_to_oil_ratio_table)})"
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

    def check_physical_consistency(self):
        """Perform physical consistency checks on table data."""
        n_p, n_t = len(self.pressures), len(self.temperatures)

        # Check 2D table shapes
        tables_2d = {
            "oil_viscosity_table": self.oil_viscosity_table,
            "oil_density_table": self.oil_density_table,
            "gas_viscosity_table": self.gas_viscosity_table,
            "gas_density_table": self.gas_density_table,
        }
        for name, table in tables_2d.items():
            if table is not None:
                if table.shape != (n_p, n_t):
                    raise ValueError(
                        f"`{name}` shape {table.shape} must match (n_pressures={n_p}, n_temperatures={n_t})"
                    )
                # Check for NaN or inf
                if not np.all(np.isfinite(table)):
                    raise ValueError(f"`{name}` contains NaN or infinite values")

        # Physical bounds checks
        if self.oil_viscosity_table is not None:
            if np.any(self.oil_viscosity_table <= 0):
                raise ValueError("`oil_viscosity_table` must be positive")

        if self.gas_viscosity_table is not None:
            if np.any(self.gas_viscosity_table <= 0):
                raise ValueError("`gas_viscosity_table` must be positive")

        if self.water_viscosity_table is not None:
            if np.any(self.water_viscosity_table <= 0):
                raise ValueError("`water_viscosity_table` must be positive")

        # Density checks
        if self.oil_density_table is not None:
            if np.any(self.oil_density_table <= 0):
                raise ValueError("`oil_density_table` must be positive")

        if self.gas_density_table is not None:
            if np.any(self.gas_density_table <= 0):
                raise ValueError("`gas_density_table` must be positive")
            # Gas should be less dense than oil
            if self.oil_density_table is not None:
                if np.any(self.gas_density_table > self.oil_density_table):
                    logger.warning(
                        "`gas_density_table` exceed `oil_density_table` at some conditions"
                    )

        # Formation volume factor checks
        if self.oil_formation_volume_factor_table is not None:
            if np.any(self.oil_formation_volume_factor_table <= 0):
                raise ValueError("`oil_formation_volume_factor_table` must be positive")

        logger.debug("Physical consistency checks passed")

    def _check_extrapolation(self, pressure: QueryType, temperature: QueryType):
        """Log warning if extrapolating beyond table bounds."""
        if not self.warn_on_extrapolation:
            return

        pressures = np.atleast_1d(pressure)
        temperatures = np.atleast_1d(temperature)

        p_min, p_max = self._extrapolation_bounds["pressure"]
        t_min, t_max = self._extrapolation_bounds["temperature"]

        if np.any(pressures < p_min) or np.any(pressures > p_max):
            logger.warning(
                f"Pressure extrapolation: queried P ∈ [{pressures.min():.1f}, {pressures.max():.1f}] psi, "
                f"table range [{p_min:.1f}, {p_max:.1f}] psi"
            )

        if np.any(temperatures < t_min) or np.any(temperatures > t_max):
            logger.warning(
                f"Temperature extrapolation: queried T ∈ [{temperatures.min():.1f}, {temperatures.max():.1f}] °F, "
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

        # Build 2D (Pressure-Temperature) interpolators for all provided properties
        property_map_2d = {
            "oil_viscosity": self.oil_viscosity_table,
            "oil_compressibility": self.oil_compressibility_table,
            "oil_specific_gravity": self.oil_specific_gravity_table,
            "oil_api_gravity": self.oil_api_gravity_table,
            "oil_density": self.oil_density_table,
            "oil_formation_volume_factor": self.oil_formation_volume_factor_table,
            "gas_viscosity": self.gas_viscosity_table,
            "gas_compressibility": self.gas_compressibility_table,
            "gas_gravity": self.gas_gravity_table,
            "gas_molecular_weight": self.gas_molecular_weight_table,
            "gas_density": self.gas_density_table,
            "gas_formation_volume_factor": self.gas_formation_volume_factor_table,
            "gas_compressibility_factor": self.gas_compressibility_factor_table,
            "solution_gas_to_oil_ratio": self.solution_gas_to_oil_ratio_table,
        }
        # Map interpolation method to spline degree
        k = __interpolation_degree_map[self.interpolation_method]
        for name, data in property_map_2d.items():
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
        # For 3D, use RegularGridInterpolator (RectBivariateSpline is 2D only)
        if self.salinities is None:
            return  # No salinity-dependent properties to build

        method_3d = "linear" if self.interpolation_method == "linear" else "cubic"
        property_map_3d = {
            "water_viscosity": self.water_viscosity_table,
            "water_bubble_point_pressure": self.water_bubble_point_pressure_table,
            "water_compressibility": self.water_compressibility_table,
            "water_density": self.water_density_table,
            "water_formation_volume_factor": self.water_formation_volume_factor_table,
            "gas_solubility_in_water": self.gas_solubility_in_water_table,
        }
        for name, data in property_map_3d.items():
            if data is not None:
                self._interpolators[name] = RegularGridInterpolator(
                    points=(self.pressures, self.temperatures, self.salinities),
                    values=data,
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

    def PTInterpolate(
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

    def PTSInterpolate(
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
                logger.debug(
                    "`solution_gor` provided but `bubble_point_pressures` is 1D - ignoring"
                )
            result = interp(temperature)
            return float(result) if np.isscalar(temperature) else result

        # 2D: Pb(Rs, T)
        if solution_gor is None:
            raise ValueError(
                "`solution_gor` required as PVT table uses 2D bubble point pressures: Pb(Rs, T)"
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
        if self.oil_viscosity_table is None:
            return None

        # Get bubble point pressure
        bubble_point_pressure = self.oil_bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
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
            result[saturated] = self.PTInterpolate(
                name="oil_viscosity", pressure=p[saturated], temperature=t[saturated]
            )

        # Undersaturated conditions: P > Pb
        undersaturated = ~saturated
        if np.any(undersaturated):
            mu_b = self.PTInterpolate(
                name="oil_viscosity",
                pressure=pb[undersaturated],
                temperature=t[undersaturated],
            )
            co = self.PTInterpolate(
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
        return self.PTInterpolate(
            name="oil_compressibility", pressure=pressure, temperature=temperature
        )

    def oil_specific_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil specific gravity at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="oil_specific_gravity", pressure=pressure, temperature=temperature
        )

    def oil_api_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil API gravity at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="oil_api_gravity", pressure=pressure, temperature=temperature
        )

    def oil_density(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil density at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="oil_density", pressure=pressure, temperature=temperature
        )

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
        if self.oil_formation_volume_factor_table is None:
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
            result[saturated] = self.PTInterpolate(
                name="oil_formation_volume_factor",
                pressure=p[saturated],
                temperature=t[saturated],
            )

        # Undersaturated: P > Pb
        # Bo(P) = Bob * exp(-co * (P - Pb))
        undersaturated = ~saturated
        if np.any(undersaturated):
            # Get Bo at bubble point
            bob = self.PTInterpolate(
                name="oil_formation_volume_factor",
                pressure=pb[undersaturated],
                temperature=t[undersaturated],
            )

            # Get compressibility if available
            if self.oil_compressibility_table is not None:
                co = self.PTInterpolate(
                    name="oil_compressibility",
                    pressure=pb[undersaturated],
                    temperature=t[undersaturated],
                )
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
        if self.solution_gas_to_oil_ratio_table is None:
            return None

        # Get bubble point pressure
        bubble_point_pressure = self.oil_bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
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
            result[saturated] = self.PTInterpolate(
                name="solution_gas_to_oil_ratio",
                pressure=p[saturated],
                temperature=t[saturated],
            )

        # Undersaturated: P > Pb - Rs = Rsb (constant)
        undersaturated = ~saturated
        if np.any(undersaturated):
            # Use Rs at bubble point
            result[undersaturated] = self.PTInterpolate(
                name="solution_gas_to_oil_ratio",
                pressure=pb[undersaturated],
                temperature=t[undersaturated],
            )

        return float(result) if result.size == 1 else result

    def water_bubble_point_pressure(
        self, pressure: QueryType, temperature: QueryType, salinity: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water bubble point pressure at given pressure(s), temperature(s), and salinity(ies)."""
        return self.PTSInterpolate(
            name="water_bubble_point_pressure",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
        )

    def water_viscosity(
        self, pressure: QueryType, temperature: QueryType, salinity: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water viscosity at given pressure(s), temperature(s), and salinity(ies)."""
        return self.PTSInterpolate(
            name="water_viscosity",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
        )

    def water_compressibility(
        self, pressure: QueryType, temperature: QueryType, salinity: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water compressibility at given pressure(s), temperature(s), and salinity(ies)."""
        return self.PTSInterpolate(
            name="water_compressibility",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
        )

    def water_density(
        self, pressure: QueryType, temperature: QueryType, salinity: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water density at given pressure(s), temperature(s), and salinity(ies)."""
        return self.PTSInterpolate(
            name="water_density",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
        )

    def water_formation_volume_factor(
        self, pressure: QueryType, temperature: QueryType, salinity: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water formation volume factor at given pressure(s), temperature(s), and salinity(ies)."""
        return self.PTSInterpolate(
            name="water_formation_volume_factor",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
        )

    def gas_viscosity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas viscosity at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="gas_viscosity", pressure=pressure, temperature=temperature
        )

    def gas_compressibility(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas compressibility at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="gas_compressibility", pressure=pressure, temperature=temperature
        )

    def gas_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas gravity at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="gas_gravity", pressure=pressure, temperature=temperature
        )

    def gas_molecular_weight(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas molecular weight at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="gas_molecular_weight", pressure=pressure, temperature=temperature
        )

    def gas_density(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas density at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="gas_density", pressure=pressure, temperature=temperature
        )

    def gas_formation_volume_factor(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas formation volume factor at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="gas_formation_volume_factor",
            pressure=pressure,
            temperature=temperature,
        )

    def gas_compressibility_factor(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas compressibility factor at given pressure(s) and temperature(s)."""
        return self.PTInterpolate(
            name="gas_compressibility_factor",
            pressure=pressure,
            temperature=temperature,
        )

    def gas_solubility_in_water(
        self, pressure: QueryType, temperature: QueryType, salinity: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas solubility in water at given pressure(s), temperature(s), and salinity(ies)."""
        return self.PTSInterpolate(
            name="gas_solubility_in_water",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
        )

    def update(self, **kwargs: typing.Any) -> Self:
        """Get a new `PVTTables` instance with updated fields."""
        return attrs.evolve(self, **kwargs)


def build_pvt_tables(
    pressures: OneDimensionalGrid,
    temperatures: OneDimensionalGrid,
    salinities: typing.Optional[OneDimensionalGrid] = None,
    oil_specific_gravity: typing.Optional[float] = None,
    gas_gravity: typing.Optional[float] = None,
    reservoir_gas: typing.Optional[str] = None,
    solution_gas_to_oil_ratios: typing.Optional[OneDimensionalGrid] = None,
    build_oil_properties: bool = True,
    build_water_properties: bool = True,
    build_gas_properties: bool = True,
    oil_viscosities: typing.Optional[TwoDimensionalGrid] = None,
    oil_compressibilities: typing.Optional[TwoDimensionalGrid] = None,
    oil_specific_gravities: typing.Optional[TwoDimensionalGrid] = None,
    oil_api_gravities: typing.Optional[TwoDimensionalGrid] = None,
    oil_densities: typing.Optional[TwoDimensionalGrid] = None,
    oil_formation_volume_factors: typing.Optional[TwoDimensionalGrid] = None,
    bubble_point_pressures: typing.Optional[
        typing.Union[OneDimensionalGrid, TwoDimensionalGrid]
    ] = None,
    gas_solubilities_in_water: typing.Optional[ThreeDimensionalGrid] = None,
    water_bubble_point_pressures: typing.Optional[ThreeDimensionalGrid] = None,
    water_viscosities: typing.Optional[ThreeDimensionalGrid] = None,
    water_compressibilities: typing.Optional[ThreeDimensionalGrid] = None,
    water_densities: typing.Optional[ThreeDimensionalGrid] = None,
    water_formation_volume_factors: typing.Optional[ThreeDimensionalGrid] = None,
    gas_viscosities: typing.Optional[TwoDimensionalGrid] = None,
    gas_compressibilities: typing.Optional[TwoDimensionalGrid] = None,
    gas_gravities: typing.Optional[TwoDimensionalGrid] = None,
    gas_molecular_weights: typing.Optional[TwoDimensionalGrid] = None,
    gas_densities: typing.Optional[TwoDimensionalGrid] = None,
    gas_formation_volume_factors: typing.Optional[TwoDimensionalGrid] = None,
    gas_compressibility_factors: typing.Optional[TwoDimensionalGrid] = None,
    solution_gas_to_oil_ratio_table: typing.Optional[TwoDimensionalGrid] = None,
    validate_tables: bool = True,
    interpolation_method: InterpolationMethod = "cubic",
    warn_on_extrapolation: bool = True,
) -> PVTTables:
    """
    Build comprehensive PVT tables using empirical correlations.

    This factory function generates complete 2D (P-T) and 3D (P-T-Salinity) lookup tables
    for fluid properties. It uses the same correlations as `reservoir_model()` but pre-computes
    them on a grid for fast interpolation during simulation.

    :param pressures: 1D array of pressures (psi), e.g. np.linspace(500, 5000, 50)
    :param temperatures: 1D array of temperatures (°F), e.g. np.linspace(100, 250, 30)

    :param oil_specific_gravity: Oil specific gravity (dimensionless), e.g. 0.85
    :param gas_gravity: Gas specific gravity (dimensionless), e.g. 0.65
    :param water_salinity: Water salinity (ppm NaCl), e.g. 35000. Used for single-value water properties.
    :param salinities: 1D array of salinities (ppm) for 3D gas solubility table
    :param reservoir_gas: Gas type (e.g. "CO2", "Methane"), defaults to c.RESERVOIR_GAS_NAME
    :param solution_gas_to_oil_ratios: 1D array of Rs values (SCF/STB) for 2D bubble point table

    :param build_oil_properties: If True, compute oil property tables
    :param build_water_properties: If True, compute water property tables
    :param build_gas_properties: If True, compute gas property tables
    :param build_3d_gas_solubility: If True, build 3D Rsw(P,T,S) table (requires `salinities`)

    :param oil_viscosities: Optional pre-computed oil viscosity table (n_p, n_t)
    :param oil_compressibilities: Optional pre-computed oil compressibility table (n_p, n_t)
    :param oil_specific_gravities: Optional pre-computed oil SG table (n_p, n_t) - usually constant
    :param oil_api_gravities: Optional pre-computed API gravity table (n_p, n_t) - usually constant
    :param oil_densities: Optional pre-computed oil density table (n_p, n_t)
    :param oil_formation_volume_factors: Optional pre-computed Bo table (n_p, n_t)
    :param bubble_point_pressures: Optional pre-computed Pb table - 1D (n_t) or 2D (n_rs, n_t)
    :param solution_gas_to_oil_ratio_table: Optional pre-computed Rs table (n_p, n_t)

    :param water_bubble_point_pressures: Optional pre-computed water Pb table (n_p, n_t)
    :param water_viscosities: Optional pre-computed water viscosity table (n_p, n_t)
    :param water_compressibilities: Optional pre-computed water compressibility table (n_p, n_t)
    :param water_densities: Optional pre-computed water density table (n_p, n_t)
    :param water_formation_volume_factors: Optional pre-computed Bw table (n_p, n_t)
    :param salinities: Optional pre-computed salinity table (n_p, n_t) - usually constant

    :param gas_viscosities: Optional pre-computed gas viscosity table (n_p, n_t)
    :param gas_compressibilities: Optional pre-computed gas compressibility table (n_p, n_t)
    :param gas_gravities: Optional pre-computed gas gravity table (n_p, n_t) - usually constant
    :param gas_molecular_weights: Optional pre-computed MW table (n_p, n_t) - usually constant
    :param gas_densities: Optional pre-computed gas density table (n_p, n_t)
    :param gas_formation_volume_factors: Optional pre-computed Bg table (n_p, n_t)
    :param gas_compressibility_factors: Optional pre-computed Z-factor table (n_p, n_t)
    :param gas_solubilities_in_water: Optional pre-computed 3D Rsw table (n_p, n_t, n_s)

    :param validate_tables: If True, perform physical consistency checks
    :param interpolation_method: 'linear', 'cubic', or 'quintic'
    :param warn_on_extrapolation: If True, warn when extrapolating beyond table bounds

    :return: PVTTables instance with all computed/provided properties

    :raises ValueError: If required inputs are missing for requested property tables
    :raises ValidationError: If computed tables contain non-physical values

    Example:

    ```python
    # Build comprehensive tables for black oil system
    pressures = np.linspace(500, 5000, 50)
    temperatures = np.linspace(100, 250, 30)

    tables = build_pvt_tables(
        pressures=pressures,
        temperatures=temperatures,
        oil_specific_gravity=0.85,
        gas_gravity=0.65,
        water_salinity=35000,
        reservoir_gas="Methane",
        interpolation_method="cubic",
    )

    # Query properties during simulation
    mu_o = tables.oil_viscosity(pressure=2500, temperature=180)
    bo = tables.oil_formation_volume_factor(pressure=2500, temperature=180)
    ```
    """
    # Validate pressure and temperature arrays
    if pressures.ndim != 1:
        raise ValueError("`pressures` must be 1-dimensional")
    if temperatures.ndim != 1:
        raise ValueError("`temperatures` must be 1-dimensional")
    if not np.all(np.diff(pressures) > 0):
        raise ValueError("`pressures` must be strictly increasing")
    if not np.all(np.diff(temperatures) > 0):
        raise ValueError("`temperatures` must be strictly increasing")

    n_p = len(pressures)
    n_t = len(temperatures)

    # Set defaults
    reservoir_gas = reservoir_gas or c.RESERVOIR_GAS_NAME
    reservoir_gas = typing.cast(str, reservoir_gas)

    # Check requirements for oil properties
    if build_oil_properties:
        if oil_specific_gravity is None:
            raise ValueError(
                "`oil_specific_gravity` required when `build_oil_properties=True`"
            )
        if gas_gravity is None:
            raise ValueError("`gas_gravity` required when `build_oil_properties=True`")

    # Create meshgrid for 2D tables: pressure_grid[i,j] = pressures[i]
    # This creates all (P, T) combinations for vectorized computation
    pressure_grid_2d, temperature_grid_2d = np.meshgrid(
        pressures, temperatures, indexing="ij"
    )

    # Create meshgrid for 3D tables if needed
    pressure_grid_3d, temperature_grid_3d, salinity_grid_3d = np.meshgrid(
        pressures,
        temperatures,
        np.atleast_1d(salinities) if salinities is not None else np.array([35000.0]),
        indexing="ij",
    )

    # BUILD GAS PROPERTIES (if requested)
    if build_gas_properties:
        # Gas gravity (usually constant, but allow spatial variation)
        if gas_gravities is None:
            if gas_gravity is None:
                computed_gas_gravity = compute_gas_gravity(gas=reservoir_gas)
            else:
                computed_gas_gravity = gas_gravity

            gas_gravities = np.full((n_p, n_t), computed_gas_gravity)

        # Gas molecular weight (derived from gravity)
        if gas_molecular_weights is None:
            gas_molecular_weights = build_gas_molecular_weight_grid(
                gas_gravity_grid=gas_gravities
            )

        # Gas Z-factor (compressibility factor)
        if gas_compressibility_factors is None:
            gas_compressibility_factors = build_gas_compressibility_factor_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_gravity_grid=gas_gravities,
            )

        # Gas formation volume factor
        if gas_formation_volume_factors is None:
            gas_formation_volume_factors = build_gas_formation_volume_factor_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_compressibility_factor_grid=gas_compressibility_factors,
            )

        # Gas compressibility
        if gas_compressibilities is None:
            gas_compressibilities = build_gas_compressibility_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_gravity_grid=gas_gravities,
                gas_compressibility_factor_grid=gas_compressibility_factors,
            )

        # Gas density
        if gas_densities is None:
            gas_densities = build_gas_density_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_gravity_grid=gas_gravities,
                gas_compressibility_factor_grid=gas_compressibility_factors,
            )

        # Gas viscosity
        if gas_viscosities is None:
            gas_viscosities = build_gas_viscosity_grid(
                temperature_grid=temperature_grid_2d,
                gas_density_grid=gas_densities,
                gas_molecular_weight_grid=gas_molecular_weights,
            )

    # BUILD WATER PROPERTIES (if requested)
    if build_water_properties:
        # Water viscosity
        if water_viscosities is None:
            water_viscosities = build_water_viscosity_grid(
                temperature_grid=temperature_grid_3d,
                salinity_grid=salinity_grid_3d,
                pressure_grid=pressure_grid_3d,
            )

        # Gas solubility in water (2D by default, 3D if requested)
        # Build 3D table: Rsw(P, T, S)
        if gas_solubilities_in_water is None:
            gas_solubilities_in_water = build_gas_solubility_in_water_grid(
                pressure_grid=pressure_grid_3d,
                temperature_grid=temperature_grid_3d,
                salinity_grid=salinity_grid_3d,
                gas=reservoir_gas,
            )

        # Water bubble point pressure
        if water_bubble_point_pressures is None:
            water_bubble_point_pressures = build_water_bubble_point_pressure_grid(
                temperature_grid=temperature_grid_3d,
                gas_solubility_in_water_grid=gas_solubilities_in_water,
                salinity_grid=salinity_grid_3d,
                gas=reservoir_gas,
            )

        # Gas-free water FVF (needed for compressibility)
        gas_free_water_fvf_grid = build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=pressure_grid_3d,
            temperature_grid=temperature_grid_2d,
        )

        # Water compressibility
        if water_compressibilities is None:
            water_compressibilities = build_water_compressibility_grid(
                pressure_grid=pressure_grid_3d,
                temperature_grid=temperature_grid_3d,
                bubble_point_pressure_grid=water_bubble_point_pressures,
                gas_formation_volume_factor_grid=gas_formation_volume_factors,
                gas_solubility_in_water_grid=gas_solubilities_in_water,
                gas_free_water_formation_volume_factor_grid=gas_free_water_fvf_grid,
            )

        # Water density
        if water_densities is None:
            water_densities = build_water_density_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_gravity_grid=gas_gravities
                if build_gas_properties
                else np.full((n_p, n_t), 0.65),
                salinity_grid=water_salinities,
                gas_solubility_in_water_grid=(
                    gas_solubility_in_water_2d
                    if not build_3d_gas_solubility
                    else gas_solubilities_in_water[:, :, 0]
                ),
                gas_free_water_formation_volume_factor_grid=gas_free_water_fvf_grid,
            )

        # Water formation volume factor
        if water_formation_volume_factors is None:
            water_formation_volume_factors = build_water_formation_volume_factor_grid(
                water_density_grid=water_densities,
                salinity_grid=salinities,
            )

    # BUILD OIL PROPERTIES (if requested)
    if build_oil_properties:
        # Oil specific gravity (usually constant)
        if oil_specific_gravities is None:
            oil_specific_gravities = np.full((n_p, n_t), oil_specific_gravity)

        # Oil API gravity
        if oil_api_gravities is None:
            oil_api_gravities = build_oil_api_gravity_grid(
                oil_specific_gravity_grid=oil_specific_gravities
            )

        # Determine bubble point pressure structure (1D or 2D)
        if bubble_point_pressures is None:
            if solution_gas_to_oil_ratios is not None:
                # Build 2D bubble point table: Pb(Rs, T)
                n_rs = len(solution_gas_to_oil_ratios)
                bubble_point_pressures = np.zeros((n_rs, n_t))

                for i, rs_value in enumerate(solution_gas_to_oil_ratios):
                    # For each Rs value, compute Pb at each temperature
                    rs_grid = np.full(n_t, rs_value)
                    bubble_point_pressures[i, :] = build_oil_bubble_point_pressure_grid(
                        gas_gravity_grid=np.full(n_t, gas_gravity),
                        oil_api_gravity_grid=build_oil_api_gravity_grid(
                            np.full(n_t, oil_specific_gravity)
                        ),
                        temperature_grid=temperatures,
                        solution_gas_to_oil_ratio_grid=rs_grid,
                    )
            else:
                # Build 1D bubble point table: Pb(T) at reference conditions
                # Use a reference pressure to estimate initial Rs
                reference_pressure = pressures[len(pressures) // 2]  # Middle pressure

                # Estimate Rs at reference conditions
                estimated_rs_grid = np.zeros(n_t)
                for j, temp in enumerate(temperatures):
                    # Use Standing correlation to estimate Rs
                    # This is a circular dependency, so we use an iterative approach
                    # For simplicity, assume Rs ≈ 500 SCF/STB initially
                    estimated_rs_grid[j] = 500.0

                # Compute Pb(T) using estimated Rs
                bubble_point_pressures = build_oil_bubble_point_pressure_grid(
                    gas_gravity_grid=np.full(n_t, gas_gravity),
                    oil_api_gravity_grid=build_oil_api_gravity_grid(
                        np.full(n_t, oil_specific_gravity)
                    ),
                    temperature_grid=temperatures,
                    solution_gas_to_oil_ratio_grid=estimated_rs_grid,
                )

        # Create bubble point grid for 2D computations
        # For 1D Pb(T), broadcast to (n_p, n_t)
        if bubble_point_pressures.ndim == 1:
            bubble_point_pressure_grid_2d = np.broadcast_to(
                bubble_point_pressures[np.newaxis, :], (n_p, n_t)
            ).copy()
        else:
            # For 2D Pb(Rs, T), we need to interpolate to pressure grid
            # This is complex - for now, use middle Rs value
            warnings.warn(
                "2D bubble point pressure table provided. Using middle Rs value "
                "for property calculations. Consider providing full Rs(P,T) table."
            )
            mid_rs_idx = len(solution_gas_to_oil_ratios) // 2
            bubble_point_pressure_grid_2d = np.broadcast_to(
                bubble_point_pressures[mid_rs_idx, :][np.newaxis, :], (n_p, n_t)
            ).copy()

        # Solution GOR table: Rs(P, T)
        if solution_gas_to_oil_ratio_table is None:
            solution_gas_to_oil_ratio_table = build_solution_gas_to_oil_ratio_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity),
                oil_api_gravity_grid=oil_api_gravities,
            )

        # Oil formation volume factor: Bo(P, T)
        if oil_formation_volume_factors is None:
            oil_formation_volume_factors = build_oil_formation_volume_factor_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                oil_specific_gravity_grid=oil_specific_gravities,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity),
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,
                oil_compressibility_grid=(
                    oil_compressibilities
                    if oil_compressibilities is not None
                    else np.full((n_p, n_t), 1e-5)  # Temporary value
                ),
            )

        # Oil compressibility: Co(P, T)
        if oil_compressibilities is None:
            # Compute Rs at bubble point for each T
            rs_at_bp_grid = np.zeros((n_p, n_t))
            for j in range(n_t):
                pb_at_t = bubble_point_pressure_grid_2d[0, j]
                rs_at_bp_grid[:, j] = build_solution_gas_to_oil_ratio_grid(
                    pressure_grid=np.full(n_p, pb_at_t),
                    temperature_grid=np.full(n_p, temperature_grid_2d[0, j]),
                    bubble_point_pressure_grid=np.full(n_p, pb_at_t),
                    gas_gravity_grid=np.full(n_p, gas_gravity),
                    oil_api_gravity_grid=oil_api_gravities[:, j],
                )

            oil_compressibilities = build_oil_compressibility_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                oil_api_gravity_grid=oil_api_gravities,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity),
                gor_at_bubble_point_pressure_grid=rs_at_bp_grid,
                gas_formation_volume_factor_grid=(
                    gas_formation_volume_factors
                    if build_gas_properties
                    else np.ones((n_p, n_t))
                ),
                oil_formation_volume_factor_grid=oil_formation_volume_factors,
            )

        # Oil density: ρo(P, T)
        if oil_densities is None:
            oil_densities = build_live_oil_density_grid(
                oil_api_gravity_grid=oil_api_gravities,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity),
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,
                formation_volume_factor_grid=oil_formation_volume_factors,
            )

        # Oil viscosity: μo(P, T)
        if oil_viscosities is None:
            # Compute Rs at bubble point (already computed above if Co was computed)
            if "rs_at_bp_grid" not in locals():
                rs_at_bp_grid = np.zeros((n_p, n_t))
                for j in range(n_t):
                    pb_at_t = bubble_point_pressure_grid_2d[0, j]
                    rs_at_bp_grid[:, j] = build_solution_gas_to_oil_ratio_grid(
                        pressure_grid=np.full(n_p, pb_at_t),
                        temperature_grid=np.full(n_p, temperature_grid_2d[0, j]),
                        bubble_point_pressure_grid=np.full(n_p, pb_at_t),
                        gas_gravity_grid=np.full(n_p, gas_gravity),
                        oil_api_gravity_grid=oil_api_gravities[:, j],
                    )

            oil_viscosities = build_oil_viscosity_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                oil_specific_gravity_grid=oil_specific_gravities,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,
                gor_at_bubble_point_pressure_grid=rs_at_bp_grid,
            )

    return PVTTables(
        pressures=pressures,
        temperatures=temperatures,
        salinities=salinities,
        bubble_point_pressures=bubble_point_pressures,
        oil_viscosities=oil_viscosities,
        oil_compressibilities=oil_compressibilities,
        oil_specific_gravities=oil_specific_gravities,
        oil_api_gravities=oil_api_gravities,
        oil_densities=oil_densities,
        oil_formation_volume_factors=oil_formation_volume_factors,
        solution_gas_to_oil_ratios=solution_gas_to_oil_ratio_table,
        water_bubble_point_pressures=water_bubble_point_pressures,
        water_viscosities=water_viscosities,
        water_compressibilities=water_compressibilities,
        water_densities=water_densities,
        water_formation_volume_factors=water_formation_volume_factors,
        gas_viscosities=gas_viscosities,
        gas_compressibilities=gas_compressibilities,
        gas_gravities=gas_gravities,
        gas_molecular_weights=gas_molecular_weights,
        gas_densities=gas_densities,
        gas_formation_volume_factors=gas_formation_volume_factors,
        gas_compressibility_factors=gas_compressibility_factors,
        gas_solubilities_in_water=gas_solubilities_in_water,
        interpolation_method=interpolation_method,
        validate_tables=validate_tables,
        warn_on_extrapolation=warn_on_extrapolation,
    )
