import logging
import threading
import typing
from concurrent.futures import ThreadPoolExecutor

import attrs
from cachetools import LFUCache
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, interp1d

from bores._precision import get_dtype
from bores.constants import c
from bores.errors import ValidationError
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
    build_estimated_solution_gas_to_oil_ratio_grid,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from bores.pvt.core import compute_gas_gravity, compute_oil_api_gravity
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
    "PVTTableData",
    "PVTTables",
    "build_pvt_table_data",
    "build_gas_pseudo_pressure_table",
    "clear_pseudo_pressure_table_cache",
    "get_pseudo_pressure_table_cache_info",
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
        # Clamp pressure to avoid extrapolation issues
        P_clamped = max(1.0, P)  # Don't go below 1 psi

        try:
            Z = z_factor_func(P_clamped)
            mu = viscosity_func(P_clamped)
        except Exception as exc:
            logger.warning(
                f"Failed to evaluate Z or μ at P={P_clamped} using ideal gas approximation as fallback: {exc}"
            )
            # Use ideal gas approximation as fallback
            Z = 1.0
            mu = 0.01  # Typical gas viscosity in cP

        # Protect against division by zero or negative values
        if Z <= 0 or mu <= 0 or not np.isfinite(Z) or not np.isfinite(mu):
            logger.warning(
                f"Invalid Z={Z} or μ={mu} at P={P_clamped}. Using ideal gas approximation."
            )
            Z = max(Z, 0.01)  # Minimum reasonable Z
            mu = max(mu, 0.001)  # Minimum reasonable μ (cP)

        result = 2.0 * P_clamped / (mu * Z)
        # Sanity check on integrand value
        if not np.isfinite(result) or result < 0:
            logger.warning(f"Invalid integrand {result} at P={P_clamped}")
            return 0.0

        return result

    # Perform numerical integration with adaptive strategy
    p_min = min(pressure, reference_pressure)
    p_max = max(pressure, reference_pressure)

    # Split integration into segments if range is large
    # This helps `quad()` adapt better to different pressure regimes
    if (p_max - p_min) > 1000:  # Large pressure range
        # Split into low, medium, high pressure segments
        split_points = np.logspace(start=np.log10(p_min), stop=np.log10(p_max), num=5)
        total_integral = 0.0

        for i in range(len(split_points) - 1):
            try:
                segment_result, segment_error = quad(
                    func=integrand,
                    a=split_points[i],
                    b=split_points[i + 1],
                    epsabs=1e-6,
                    epsrel=1e-4,
                    limit=200,
                )
                total_integral += segment_result
            except Exception as exc:
                logger.warning(
                    f"Integration failed for segment [{split_points[i]:.1f}, {split_points[i + 1]:.1f}]: {exc}. Using trapezoidal approximation"
                )
                # Use trapezoidal approximation for failed segment
                p_seg = np.linspace(split_points[i], split_points[i + 1], 50)
                y_seg = np.array([integrand(p) for p in p_seg])
                total_integral += np.trapezoid(y=y_seg, x=p_seg)

        result = total_integral
    else:
        # Single integration for small range
        try:
            result, error = quad(
                func=integrand,
                a=p_min,
                b=p_max,
                epsabs=1e-6,
                epsrel=1e-4,
                limit=200,
            )
        except Exception as exc:
            logger.warning(f"Integration failed: {exc}. Using trapezoidal fallback.")
            # Fallback to simple trapezoidal rule
            p_points = np.linspace(p_min, p_max, 100)
            y_points = np.array([integrand(p) for p in p_points])
            result = float(np.trapezoid(y=y_points, x=p_points))

    # Apply sign based on integration direction
    if pressure < reference_pressure:
        result = -result

    return float(result)


class GasPseudoPressureTable:
    """
    Pre-computed gas pseudo-pressure table for fast lookup during simulation.

    Uses np.interp for fast linear interpolation.
    Supports both forward (pressure → pseudo-pressure) and inverse (pseudo-pressure → pressure)
    interpolation.
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
        logger.info(f"Building pseudo-pressure table with {points} points...")

        def _compute(pressure: float) -> float:
            """Compute pseudo-pressure for a single pressure point."""
            return compute_gas_pseudo_pressure(
                pressure=pressure,
                z_factor_func=z_factor_func,
                viscosity_func=viscosity_func,
                reference_pressure=self.reference_pressure,
            )

        # Use thread pool for parallel computation (I/O bound due to scipy.integrate.quad)
        # Each integration is independent, so embarrassingly parallel
        max_workers = min(8, points // 50 + 1)  # Scale workers with problem size
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.pseudo_pressures = np.array(
                list(executor.map(_compute, self.pressures)), dtype=get_dtype()
            )

        logger.debug("Pseudo-pressure table computation complete.")
        logger.info(
            f"Pseudo-pressure table built: P ∈ [{min_pressure:.4f}, {max_pressure:.4f}] psi"
        )

    def interpolate(self, pressure: float) -> float:
        """
        Interpolate pseudo-pressure at given pressure.

        Forward interpolation: pressure → pseudo-pressure

        :param pressure: Pressure (psi)
        :return: Pseudo-pressure m(P) (psi²/cP)
        """
        return np.interp(
            x=pressure,
            xp=self.pressures,
            fp=self.pseudo_pressures,
            left=self.pseudo_pressures[0],
            right=self.pseudo_pressures[-1],
        )

    def inverse_interpolate(self, pseudo_pressure: float) -> float:
        """
        Inverse interpolate pressure at given pseudo-pressure.

        Inverse interpolation: pseudo-pressure → pressure

        :param pseudo_pressure: Pseudo-pressure m(P) (psi²/cP)
        :return: Pressure (psi)
        """
        return np.interp(
            x=pseudo_pressure,
            xp=self.pseudo_pressures,
            fp=self.pressures,
            left=self.pressures[0],
            right=self.pressures[-1],
        )

    def __call__(self, pressure: float) -> float:
        """
        Fast lookup of pseudo-pressure via interpolation.

        :param pressure: Pressure (psi)
        :return: Pseudo-pressure m(P) (psi²/cP)
        """
        return self.interpolate(pressure)

    def gradient(self, pressure: float) -> float:
        """
        Compute dm/dP = 2P/(μ*Z) for use in well models.

        :param pressure: Pressure (psi)
        :return: dm/dP (psi/cP)
        """
        if pressure <= 0:
            return 0.0  # Gradient at P=0 is 0

        Z = self.z_factor_func(pressure)
        mu = self.viscosity_func(pressure)

        # Protect against invalid values
        if Z <= 0 or mu <= 0 or not np.isfinite(Z) or not np.isfinite(mu):
            logger.warning(
                f"Invalid Z={Z} or μ={mu} at P={pressure} in gradient calculation. "
                "Using safe defaults."
            )
            Z = max(Z, 0.01) if np.isfinite(Z) else 1.0
            mu = max(mu, 0.001) if np.isfinite(mu) else 0.01

        return 2.0 * pressure / (mu * Z)


_PSEUDO_PRESSURE_TABLE_CACHE: LFUCache[typing.Hashable, GasPseudoPressureTable] = (
    LFUCache(maxsize=100)
)
"""Global cache for pseudo-pressure tables"""

_PSEUDO_PRESSURE_CACHE_LOCK = threading.Lock()
"""Thread-safe lock for pseudo-pressure table cache access"""


def build_gas_pseudo_pressure_table(
    z_factor_func: typing.Callable[[float], float],
    viscosity_func: typing.Callable[[float], float],
    reference_pressure: typing.Optional[float] = None,
    pressure_range: typing.Optional[typing.Tuple[float, float]] = None,
    points: typing.Optional[int] = None,
    cache_key: typing.Optional[typing.Hashable] = None,
) -> GasPseudoPressureTable:
    """
    Build a gas pseudo-pressure table with optional global caching.

    Creates `GasPseudoPressureTable` instances with intelligent caching
    to avoid recomputing expensive integrals for identical fluid properties.

    **Thread Safety:**
    This function is thread-safe. Cache access is protected by a lock.

    **Caching Strategy:**
    - If `cache_key` is provided and a table with that key exists, return cached table
    - If `cache_key` is provided but table doesn't exist, compute and cache it
    - If `cache_key` is None, always compute a new table (no caching)

    **Cache Key Construction:**
    The cache key should uniquely identify the table based on:
    - Gas properties (specific gravity, molecular weight)
    - Temperature
    - Pressure range and resolution
    - Whether PVT tables are used

    Example:
    ```python
    # Build cache key from fluid properties
    cache_key = (
        "CH4",  # fluid name
        0.65,   # gas gravity
        16.04,  # molecular weight (g/mol)
        150.0,  # temperature (°F)
        14.7,   # reference pressure (psi)
        (14.7, 5000),  # pressure range
        100,    # points
        None,   # pvt_tables (or hash of tables)
    )

    table = build_gas_pseudo_pressure_table(
        z_factor_func=z_func,
        viscosity_func=mu_func,
        cache_key=cache_key,
    )
    ```

    :param z_factor_func: Function to compute Z-factor at a given pressure
    :param viscosity_func: Function to compute viscosity at a given pressure
    :param reference_pressure: Reference pressure (psi), default 14.7
    :param pressure_range: (min, max) pressure range (psi), default (14.7, 5000)
    :param points: Number of pressure points, default 100
    :param interpolation_method: "linear" or "cubic"
    :param cache_key: Optional hashable key for caching. If None, no caching.
    :return: `GasPseudoPressureTable` instance

    Note:
        The global cache persists for the lifetime of the Python process.
        Use `clear_pseudo_pressure_table_cache()` to free memory if needed.
    """
    # Check cache if key provided
    if cache_key is not None:
        with _PSEUDO_PRESSURE_CACHE_LOCK:
            if cache_key in _PSEUDO_PRESSURE_TABLE_CACHE:
                logger.debug(f"Using cached pseudo-pressure table for key: {cache_key}")
                return _PSEUDO_PRESSURE_TABLE_CACHE[cache_key]

    # Build new table (outside lock to avoid blocking other threads)
    logger.debug(f"Building new pseudo-pressure table for key: {cache_key}")
    table = GasPseudoPressureTable(
        z_factor_func=z_factor_func,
        viscosity_func=viscosity_func,
        reference_pressure=reference_pressure,
        pressure_range=pressure_range,
        points=points,
    )

    # Cache if key provided
    if cache_key is not None:
        with _PSEUDO_PRESSURE_CACHE_LOCK:
            # Double-check in case another thread built it while we were working
            if cache_key not in _PSEUDO_PRESSURE_TABLE_CACHE:
                _PSEUDO_PRESSURE_TABLE_CACHE[cache_key] = table
                logger.debug(
                    f"Cached pseudo-pressure table. Cache size: {len(_PSEUDO_PRESSURE_TABLE_CACHE)}"
                )
            else:
                # Another thread cached it first, use that one
                table = _PSEUDO_PRESSURE_TABLE_CACHE[cache_key]
    return table


def clear_pseudo_pressure_table_cache() -> None:
    """Clear the global pseudo-pressure table cache to free memory."""
    global _PSEUDO_PRESSURE_TABLE_CACHE
    with _PSEUDO_PRESSURE_CACHE_LOCK:
        cache_size = len(_PSEUDO_PRESSURE_TABLE_CACHE)
        _PSEUDO_PRESSURE_TABLE_CACHE.clear()
    logger.info(f"Cleared {cache_size} cached pseudo-pressure tables")


def get_pseudo_pressure_table_cache_info() -> typing.Dict[str, typing.Any]:
    """Get information about the current cache state."""
    with _PSEUDO_PRESSURE_CACHE_LOCK:
        return {
            "cache_size": len(_PSEUDO_PRESSURE_TABLE_CACHE),
            "cached_keys": list(_PSEUDO_PRESSURE_TABLE_CACHE.keys()),
        }


QueryType = typing.Union[NDimensionalGrid, list, float, np.floating]
InterpolationMethod = typing.Literal["linear", "cubic"]
_INTERPOLATION_DEGREES = {"linear": 1, "cubic": 3}


@attrs.frozen(slots=True)
class PVTTableData:
    """
    Raw PVT table data for serialization, inspection, and interpolator building.

    Holds the complete tabulated data for oil, water, and gas properties.
    It can be serialized/saved to disk and used to build PVTTables interpolators.

    All raw table arrays are stored here. After building interpolators, these can be
    discarded to save memory (~50% reduction).
    """

    # Base grids for interpolation
    pressures: OneDimensionalGrid
    """One-dimensional grid of pressures (psi)."""

    temperatures: OneDimensionalGrid
    """One-dimensional grid of temperatures (°F)."""

    salinities: typing.Optional[OneDimensionalGrid] = None
    """One-dimensional grid of salinities (ppm NaCl) for salinity-dependent properties."""

    solution_gas_oil_ratios: typing.Optional[OneDimensionalGrid] = None
    """One-dimensional grid of solution gas-oil ratios (SCF/STB) for varying composition properties."""

    bubble_point_pressures: typing.Optional[
        typing.Union[OneDimensionalGrid, TwoDimensionalGrid]
    ] = None
    """
    Bubble point pressures for dead oil (psi). Can be 1D: Pb(T) or 2D: Pb(Rs, T)
    """

    # Oil properties tables (2D: n_pressures x n_temperatures)
    oil_viscosity_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil viscosity μo(P,T) in cP. 2D table: (n_pressures x n_temperatures)."""

    oil_compressibility_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil compressibility co(P,T) in psi⁻¹. 2D table: (n_pressures x n_temperatures)."""

    oil_specific_gravity_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil specific gravity γo(P,T) (dimensionless, water=1). 2D table: (n_pressures x n_temperatures)."""

    oil_api_gravity_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil API gravity °API(P,T). 2D table: (n_pressures x n_temperatures)."""

    oil_density_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil density ρo(P,T) in lbm/ft³. 2D table: (n_pressures x n_temperatures)."""

    oil_formation_volume_factor_table: typing.Optional[TwoDimensionalGrid] = None
    """Oil formation volume factor Bo(P,T) in bbl/STB. 2D table: (n_pressures x n_temperatures)."""

    solution_gas_to_oil_ratio_table: typing.Optional[TwoDimensionalGrid] = None
    """Solution gas-oil ratio Rs(P,T) in SCF/STB. 2D table: (n_pressures x n_temperatures)."""

    # Water properties tables (3D: n_pressures x n_temperatures x n_salinities)
    water_bubble_point_pressure_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water bubble point pressure Pb,w(P,T,S) in psi. 3D table: (n_pressures x n_temperatures x n_salinities)."""

    water_viscosity_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water viscosity μw(P,T,S) in cP. 3D table: (n_pressures x n_temperatures x n_salinities)."""

    water_compressibility_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water compressibility cw(P,T,S) in psi⁻¹. 3D table: (n_pressures x n_temperatures x n_salinities)."""

    water_density_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water density ρw(P,T,S) in lbm/ft³. 3D table: (n_pressures x n_temperatures x n_salinities)."""

    water_formation_volume_factor_table: typing.Optional[ThreeDimensionalGrid] = None
    """Water formation volume factor Bw(P,T,S) in bbl/STB. 3D table: (n_pressures x n_temperatures x n_salinities)."""

    # Gas properties tables (2D: n_pressures x n_temperatures)
    gas_viscosity_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas viscosity μg(P,T) in cP. 2D table: (n_pressures x n_temperatures)."""

    gas_compressibility_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas compressibility cg(P,T) in psi⁻¹. 2D table: (n_pressures x n_temperatures)."""

    gas_gravity_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas specific gravity γg(P,T) (dimensionless, air=1). 2D table: (n_pressures x n_temperatures)."""

    gas_molecular_weight_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas molecular weight Mg(P,T) in lbm/lb-mol. 2D table: (n_pressures x n_temperatures)."""

    gas_density_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas density ρg(P,T) in lbm/ft³. 2D table: (n_pressures x n_temperatures)."""

    gas_formation_volume_factor_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas formation volume factor Bg(P,T) in bbl/SCF. 2D table: (n_pressures x n_temperatures)."""

    gas_solubility_in_water_table: typing.Optional[ThreeDimensionalGrid] = None
    """Gas solubility in water Rsw(P,T,S) in SCF/STB. 3D table: (n_pressures x n_temperatures x n_salinities)."""

    gas_compressibility_factor_table: typing.Optional[TwoDimensionalGrid] = None
    """Gas compressibility factor (z-factor) z(P,T) (dimensionless). 2D table: (n_pressures x n_temperatures)."""

    def __attrs_post_init__(self) -> None:
        """Post-initialization hook to enforce dtype consistency on all arrays."""
        self._ensure_dtype()

    def _ensure_dtype(self) -> None:
        """
        Ensure all grids and tables use the correct global dtype from get_dtype().

        Converts all non-None array attributes to the precision specified by get_dtype().
        This prevents dtype mismatches that can cause numerical issues and memory overhead.
        """
        dtype = get_dtype()
        for field in attrs.fields(type(self)):
            value = getattr(self, field.name)

            if value is not None and isinstance(value, np.ndarray):
                if value.dtype != dtype:
                    object.__setattr__(
                        self, field.name, value.astype(dtype, copy=False)
                    )


class PVTTables:
    """
    PVT (Pressure-Volume-Temperature) property lookup using pre-built interpolators.

    Provides O(1) lookup for fluid properties using pre-computed interpolators.

    NOTE: This does not store raw table data (only interpolators), saving ~50% memory vs storing both.
    To preserve raw data for serialization/inspection, keep the `PVTTableData` object separately.

    **Extrapolation Behavior:**
    When querying outside table bounds:
    - 2D properties (P-T): Extrapolates linearly/cubically based on interpolation_method
    - 3D properties (P-T-S): Extrapolates if fill_value=None (current behavior)
    - Set warn_on_extrapolation=True to log warnings for out-of-bounds queries

    **Performance:**
    Pre-computed interpolators provide O(1) lookup after initial build.
    Uses `RectBivariateSpline` for 2D (10-50x faster than `RegularGridInterpolator`).
    """

    def __init__(
        self,
        table_data: PVTTableData,
        interpolation_method: InterpolationMethod = "linear",
        validate_tables: bool = True,
        warn_on_extrapolation: bool = False,
    ):
        """
        Initialize PVT tables from raw table data.

        Builds fast interpolators from raw table data. The initialization process mostly involves:
        1. Validates grid monotonicity and consistency
        2. Optionally validates physical consistency of property tables
        3. Builds interpolators for all non-None properties in `table_data`
        4. Discards raw table arrays (Garbage Collects) (keeping only interpolators)

        Please note that `table_data` can be discarded after initialization to free memory,
        as all necessary data is stored in the interpolators. This saves a substantial amount of memory
        (~50%) when working with large PVT tables. The garbage collector will reclaim the memory used by
        the raw arrays once there are no references to `table_data` but it is still recommended to explicitly delete.

        For Example;
        ```python
        import gc

        pvt_data = build_pvt_table_data(...)  # Build raw table data
        pvt_tables = PVTTables(pvt_data)     # Build interpolators
        del pvt_data                         # Discard raw data to free memory

        gc.collect()                        # Optionally force garbage collection
        ```

        :param table_data: `PVTTableData` containing all raw property tables and grids
        :param interpolation_method: Interpolation method for property lookup
            - 'linear': Fast, 1st-order accurate
            - 'cubic': Slower, 3rd-order accurate, smooth derivatives

        :param validate_tables: If True, perform physical consistency checks on tables
            - Checks monotonicity of viscosities, densities, etc.
            - Validates FVF > 0, compressibilities > 0
            - Raises ValidationError if inconsistencies found

        :param warn_on_extrapolation: If True, log warnings when queries exceed table bounds
            - Useful for detecting simulation conditions outside calibrated range
            - No warnings by default to avoid log spam
        """
        # Store options
        if interpolation_method not in _INTERPOLATION_DEGREES:
            raise ValidationError(
                f"Invalid interpolation_method '{interpolation_method}'. "
                f"Must be one of: {list(_INTERPOLATION_DEGREES.keys())}"
            )

        pressures = table_data.pressures
        temperatures = table_data.temperatures
        salinities = table_data.salinities

        if interpolation_method != "linear":
            # Cubic+ interpolation methods requires atleast 4 points
            if len(pressures) < 4:
                raise ValidationError(
                    f"Atleast 4 pressure points required for "
                    f"'{interpolation_method}' interpolation, got {len(pressures)}"
                )
            if len(temperatures) < 4:
                raise ValidationError(
                    f"Atleast 4 temperature points required for "
                    f"'{interpolation_method}' interpolation, got {len(temperatures)}"
                )
            if salinities is not None and len(salinities) < 4:
                raise ValidationError(
                    f"Atleast 4 salinity points required for "
                    f"'{interpolation_method}' interpolation, got {len(salinities)}"
                )

        self.interpolation_method = interpolation_method
        self.validate_tables = validate_tables
        self.warn_on_extrapolation = warn_on_extrapolation

        # Initialize caches
        self._interpolators = {}
        self._extrapolation_bounds = {}
        self.default_salinity = salinities[0] if salinities is not None else None
        self._pb_ndim = (
            table_data.bubble_point_pressures.ndim
            if table_data.bubble_point_pressures is not None
            else None
        )

        # Validate and build
        self._validate_grids(table_data)

        if self.validate_tables:
            self._check_physical_consistency(table_data)

        # Store bounds for extrapolation detection
        self._extrapolation_bounds = {
            "pressure": (pressures[0], pressures[-1]),
            "temperature": (temperatures[0], temperatures[-1]),
        }
        if salinities is not None:
            self._extrapolation_bounds["salinity"] = (salinities[0], salinities[-1])

        # Build all interpolators from table_data
        self._build_interpolators(table_data)
        logger.info(
            f"PVT tables initialized: P ∈ [{pressures[0]:.4f}, {pressures[-1]:.4f}] psi, "
            f"T ∈ [{temperatures[0]:.4f}, {temperatures[-1]:.4f}] °F, "
            f"interpolation_method={self.interpolation_method!r}"
        )

    def _validate_grids(self, table_data: PVTTableData):
        """
        Validate grid dimensions and monotonicity.

        Ensures that:
        - Pressure and temperature grids are 1D arrays
        - All grids are strictly monotonically increasing (required for interpolation)
        - Bubble point pressure dimensions match grid dimensions
        - Solution GOR array is provided when using 2D bubble point pressures

        :param table_data: `PVTTableData` containing grids to validate
        :raises `ValidationError`: If any validation check fails
        """
        pressures = table_data.pressures
        temperatures = table_data.temperatures
        salinities = table_data.salinities
        solution_gas_oil_ratios = table_data.solution_gas_oil_ratios
        bubble_point_pressures = table_data.bubble_point_pressures

        # Check dimensionality
        if pressures.ndim != 1:
            raise ValidationError("`pressures` must be 1-dimensional")
        if temperatures.ndim != 1:
            raise ValidationError("`temperatures` must be 1-dimensional")

        # Check monotonicity (critical for interpolation)
        if not np.all(np.diff(pressures) > 0):
            raise ValidationError(
                "`pressures` must be strictly monotonically increasing"
            )
        if not np.all(np.diff(temperatures) > 0):
            raise ValidationError(
                "`temperatures` must be strictly monotonically increasing"
            )

        # Validate bubble point pressures
        if bubble_point_pressures is not None:
            if bubble_point_pressures.ndim == 1:
                # 1D: Pb(T) for single composition
                if len(bubble_point_pressures) != len(temperatures):
                    raise ValidationError(
                        f"`bubble_point_pressures` length ({len(bubble_point_pressures)}) "
                        f"must match temperatures length ({len(temperatures)})"
                    )
            elif bubble_point_pressures.ndim == 2:
                # 2D: Pb(Rs, T) for varying composition
                if solution_gas_oil_ratios is None:
                    raise ValidationError(
                        "`solution_gas_oil_ratios` array required for 2D bubble_point_pressures"
                    )

                if not np.all(np.diff(solution_gas_oil_ratios) > 0):
                    raise ValidationError(
                        "`solution_gas_oil_ratios` must be strictly monotonically increasing"
                    )

                n_rs, n_t = bubble_point_pressures.shape  # type: ignore
                if n_rs != len(solution_gas_oil_ratios):
                    raise ValidationError(
                        f"`bubble_point_pressures` first dimension ({n_rs}) "
                        f"must match solution_gas_oil_ratios length ({len(solution_gas_oil_ratios)})"
                    )
                if n_t != len(temperatures):
                    raise ValidationError(
                        f"`bubble_point_pressures` second dimension ({n_t}) "
                        f"must match temperatures length ({len(temperatures)})"
                    )
            else:
                raise ValidationError("`bubble_point_pressures` must be 1D or 2D")

        if salinities is not None:
            if salinities.ndim != 1:
                raise ValidationError("salinities must be 1-dimensional")
            if not np.all(np.diff(salinities) > 0):
                raise ValidationError(
                    "salinities must be strictly monotonically increasing"
                )

    def _check_physical_consistency(self, table_data: PVTTableData):
        """
        Perform physical consistency checks on table data.

        Validates that property values are physically reasonable:
        - All viscosities must be positive
        - All densities must be positive
        - Gas density < oil density (always true for reservoir fluids)
        - All formation volume factors must be positive
        - 2D tables have shape (n_pressures, n_temperatures)
        - 3D tables have shape (n_pressures, n_temperatures, n_salinities)

        :param table_data: `PVTTableData` to validate
        :raises `ValidationError`: If any physical consistency check fails
        """
        n_p, n_t = len(table_data.pressures), len(table_data.temperatures)

        # Check 2D table shapes
        tables_2d = {
            "oil_viscosity_table": table_data.oil_viscosity_table,
            "oil_density_table": table_data.oil_density_table,
            "gas_viscosity_table": table_data.gas_viscosity_table,
            "gas_density_table": table_data.gas_density_table,
        }
        for name, table in tables_2d.items():
            if table is not None:
                if table.shape != (n_p, n_t):
                    raise ValidationError(
                        f"{name} shape {table.shape} must match (n_pressures={n_p}, n_temperatures={n_t})"
                    )

        # Physical bounds checks
        if table_data.oil_viscosity_table is not None:
            if np.any(table_data.oil_viscosity_table <= 0):
                raise ValidationError("Oil viscosity must be positive")

        if table_data.gas_viscosity_table is not None:
            if np.any(table_data.gas_viscosity_table <= 0):
                raise ValidationError("Gas viscosity must be positive")

        if table_data.water_viscosity_table is not None:
            if np.any(table_data.water_viscosity_table <= 0):
                raise ValidationError("Water viscosity must be positive")

        # Density checks
        if table_data.oil_density_table is not None:
            if np.any(table_data.oil_density_table <= 0):
                raise ValidationError("Oil density must be positive")

        if table_data.gas_density_table is not None:
            if np.any(table_data.gas_density_table <= 0):
                raise ValidationError("Gas density must be positive")
            # # Gas should be less dense than oil
            # if table_data.oil_density_table is not None:
            #     if np.any(table_data.gas_density_table >= table_data.oil_density_table):
            #         raise ValidationError("Gas density should be less than oil density")

        # Formation volume factor checks
        if table_data.oil_formation_volume_factor_table is not None:
            if np.any(table_data.oil_formation_volume_factor_table <= 0):
                raise ValidationError("Oil formation volume factor must be positive")

        logger.debug("Physical consistency checks passed")

    def _check_extrapolation(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ):
        """
        Log warning if extrapolating beyond table bounds.

        Checks if query values fall outside the valid range of the property tables.
        Only logs warnings if warn_on_extrapolation=True was set during initialization.

        :param pressure: Pressure value(s) to check
        :param temperature: Temperature value(s) to check
        :param salinity: Optional salinity value(s) to check (for 3D properties)
        """
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

        # Also check salinity bounds if provided
        if salinity is not None:
            s_min, s_max = self._extrapolation_bounds.get("salinity", (None, None))
            if s_min is not None and s_max is not None:
                salinities_arr = np.atleast_1d(salinity)
                if np.any(salinities_arr < s_min) or np.any(salinities_arr > s_max):
                    logger.warning(
                        f"Salinity extrapolation: queried S ∈ [{salinities_arr.min():.0f}, {salinities_arr.max():.0f}] ppm, "
                        f"table range [{s_min:.0f}, {s_max:.0f}] ppm"
                    )

    def _build_interpolators(self, table_data: PVTTableData):
        """
        Build fast interpolators for all provided properties.

        Creates scipy interpolator objects for efficient property lookup:
        - 1D interpolators: For Pb(T) using interp1d
        - 2D interpolators: For oil/gas properties using RectBivariateSpline (10-50x faster than RegularGridInterpolator)
        - 3D interpolators: For water properties using RegularGridInterpolator

        Only builds interpolators for non-None properties in `table_data`.
        Raw table data is not stored after interpolators are built.

        :param table_data: PVTTableData containing raw property tables
        """
        # Build bubble point pressure interpolator (1D or 2D)
        if table_data.bubble_point_pressures is not None:
            if table_data.bubble_point_pressures.ndim == 1:
                # 1D - Pb(T)
                self._interpolators["bubble_point_pressure"] = interp1d(
                    x=table_data.temperatures,
                    y=table_data.bubble_point_pressures,
                    kind=self.interpolation_method,
                    bounds_error=False,
                    fill_value="extrapolate",  # type: ignore
                )
            else:
                # 2D - Pb(Rs, T)
                k = _INTERPOLATION_DEGREES[self.interpolation_method]
                self._interpolators["bubble_point_pressure"] = RectBivariateSpline(
                    x=table_data.solution_gas_oil_ratios,
                    y=table_data.temperatures,
                    z=table_data.bubble_point_pressures,
                    kx=k,
                    ky=k,
                )

        # Build 2D (Pressure-Temperature) interpolators for all provided properties
        property_map_2d = {
            "oil_viscosity": table_data.oil_viscosity_table,
            "oil_compressibility": table_data.oil_compressibility_table,
            "oil_specific_gravity": table_data.oil_specific_gravity_table,
            "oil_api_gravity": table_data.oil_api_gravity_table,
            "oil_density": table_data.oil_density_table,
            "oil_formation_volume_factor": table_data.oil_formation_volume_factor_table,
            "gas_viscosity": table_data.gas_viscosity_table,
            "gas_compressibility": table_data.gas_compressibility_table,
            "gas_gravity": table_data.gas_gravity_table,
            "gas_molecular_weight": table_data.gas_molecular_weight_table,
            "gas_density": table_data.gas_density_table,
            "gas_formation_volume_factor": table_data.gas_formation_volume_factor_table,
            "gas_compressibility_factor": table_data.gas_compressibility_factor_table,
            "solution_gas_to_oil_ratio": table_data.solution_gas_to_oil_ratio_table,
        }
        # Map interpolation method to spline degree
        k = _INTERPOLATION_DEGREES[self.interpolation_method]
        for name, data in property_map_2d.items():
            if data is not None:
                # RectBivariateSpline is 10-50x faster than RegularGridInterpolator for 2D
                self._interpolators[name] = RectBivariateSpline(
                    x=table_data.pressures,
                    y=table_data.temperatures,
                    z=data,
                    kx=k,
                    ky=k,
                )

        # Build 3D (Pressure-Temperature-Salinity) interpolator for salinity-dependent properties
        # For 3D, use RegularGridInterpolator (RectBivariateSpline is 2D only)
        if table_data.salinities is None:
            return  # No salinity-dependent properties to build

        property_map_3d = {
            "water_viscosity": table_data.water_viscosity_table,
            "water_bubble_point_pressure": table_data.water_bubble_point_pressure_table,
            "water_compressibility": table_data.water_compressibility_table,
            "water_density": table_data.water_density_table,
            "water_formation_volume_factor": table_data.water_formation_volume_factor_table,
            "gas_solubility_in_water": table_data.gas_solubility_in_water_table,
        }
        for name, data in property_map_3d.items():
            if data is not None:
                self._interpolators[name] = RegularGridInterpolator(
                    points=(
                        table_data.pressures,
                        table_data.temperatures,
                        table_data.salinities,
                    ),
                    values=data,
                    method=self.interpolation_method,
                    bounds_error=False,
                    fill_value=None,  # type: ignore  # Extrapolate
                )

        logger.debug(f"Built {len(self._interpolators)} interpolators")

    def exists(self, name: str) -> bool:
        """Check if a specific property interpolator exists."""
        return name in self._interpolators

    def pt_interpolate(
        self, name: str, pressure: QueryType, temperature: QueryType
    ) -> typing.Union[float, np.ndarray, None]:
        """
        Fast 2D property lookup using Pressure-Temperature interpolator.

        Uses `RectBivariateSpline` for efficient vectorized evaluation. Handles scalar and array inputs.

        :param name: Property name (e.g., "oil_viscosity", "gas_density")
        :param pressure: Pressure value(s) in psi
        :param temperature: Temperature value(s) in °F
        :return: Interpolated property value(s), or None if property not available
        """
        interp = self._interpolators.get(name)
        if interp is None:
            return None

        # Fast path for scalar inputs to avoid numpy array conversion overhead
        if isinstance(pressure, (int, float)) and isinstance(temperature, (int, float)):
            if self.warn_on_extrapolation:
                self._check_extrapolation(pressure, temperature)
            return float(interp.ev(pressure, temperature))

        # Check for extrapolation if requested
        if self.warn_on_extrapolation:
            self._check_extrapolation(pressure, temperature)

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
                raise ValidationError(
                    f"Incompatible shapes: pressure {p.shape}, temperature {t.shape}"
                )

        # RectBivariateSpline.ev is optimized for vectorized evaluation
        result = interp.ev(p, t)

        # Return scalar if inputs were scalar
        return float(result) if result.size == 1 else result

    def pts_interpolate(
        self,
        name: str,
        pressure: QueryType,
        temperature: QueryType,
        salinity: QueryType,
    ) -> typing.Union[float, np.ndarray, None]:
        """
        Fast 3D property lookup using Pressure-Temperature-Salinity interpolator.

        Uses `RegularGridInterpolator` for salinity-dependent water properties.
        Handles scalar and array inputs with broadcasting.

        :param name: Property name (e.g., "water_viscosity", "gas_solubility_in_water")
        :param pressure: Pressure value(s) in psi
        :param temperature: Temperature value(s) in °F
        :param salinity: Salinity value(s) in ppm NaCl
        :return: Interpolated property value(s), or None if property not available
        """
        interp = self._interpolators.get(name)
        if interp is None:
            return None

        # Check for extrapolation if requested
        if self.warn_on_extrapolation:
            self._check_extrapolation(pressure, temperature, salinity)

        # Convert to arrays and broadcast
        p = np.atleast_1d(pressure)
        t = np.atleast_1d(temperature)
        s = np.atleast_1d(salinity)
        p, t, s = np.broadcast_arrays(p, t, s)

        # Stack and evaluate
        points = np.column_stack([p.ravel(), t.ravel(), s.ravel()])
        result = interp(points).reshape(p.shape)

        # Preserve global dtype precision
        dtype = get_dtype()
        if result.dtype != dtype:
            result = result.astype(dtype, copy=False)

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
            raise ValidationError("Bubble point pressure table not provided")

        if self._pb_ndim == 1:  # type: ignore
            # 1D: Pb(T)
            if solution_gor is not None:
                logger.debug(
                    "`solution_gor` provided but `bubble_point_pressures` is 1D - ignoring"
                )
            result = interp(temperature)
            return float(result) if np.isscalar(temperature) else result

        # 2D: Pb(Rs, T)
        if solution_gor is None:
            raise ValidationError(
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
                raise ValidationError(
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
        pb = self.oil_bubble_point_pressure(
            temperature=temperature, solution_gor=solution_gor
        )

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
        if "oil_viscosity" not in self._interpolators:
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
            result[saturated] = self.pt_interpolate(
                name="oil_viscosity", pressure=p[saturated], temperature=t[saturated]
            )

        # Undersaturated conditions: P > Pb
        undersaturated = np.invert(saturated)
        if np.any(undersaturated):
            # Get viscosity at bubble point (mu_ob)
            mu_b = self.pt_interpolate(
                name="oil_viscosity",
                pressure=pb[undersaturated],
                temperature=t[undersaturated],
            )

            if mu_b is None:
                # Fallback: use saturated viscosity as approximation
                result[undersaturated] = self.pt_interpolate(
                    name="oil_viscosity",
                    pressure=p[undersaturated],
                    temperature=t[undersaturated],
                )  # type: ignore[assignment]
            else:
                # Apply Modified Beggs & Robinson undersaturated viscosity correlation
                # (consistent with bores.pvt.arrays.compute_oil_viscosity)
                # mu_o = mu_ob * (P / Pb)^X
                # where X = 2.6 * P^1.187 * exp(-11.513 - 8.98e-5 * P)
                # Reference: Beggs & Robinson (1975), Vazquez & Beggs (1980)

                p_under = p[undersaturated]
                pb_under = pb[undersaturated]

                # Compute undersaturated exponent X
                X = 2.6 * (p_under**1.187) * np.exp(-11.513 - 8.98e-5 * p_under)
                # Apply pressure ratio correction
                result[undersaturated] = mu_b * ((p_under / pb_under) ** X)

        return float(result) if result.size == 1 else result

    def oil_compressibility(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil compressibility at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="oil_compressibility", pressure=pressure, temperature=temperature
        )

    def oil_specific_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil specific gravity at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="oil_specific_gravity", pressure=pressure, temperature=temperature
        )

    def oil_api_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil API gravity at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="oil_api_gravity", pressure=pressure, temperature=temperature
        )

    def oil_density(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get oil density at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
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
        if "oil_formation_volume_factor" not in self._interpolators:
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
            result[saturated] = self.pt_interpolate(
                name="oil_formation_volume_factor",
                pressure=p[saturated],
                temperature=t[saturated],
            )

        # Undersaturated: P > Pb
        # Bo(P) = Bob * exp(-co_avg * (P - Pb))
        # where `co_avg` is the average compressibility between Pb and P
        undersaturated = np.invert(saturated)
        if np.any(undersaturated):
            # Get Bo at bubble point
            bob = self.pt_interpolate(
                name="oil_formation_volume_factor",
                pressure=pb[undersaturated],
                temperature=t[undersaturated],
            )

            # Get compressibility if available
            if "oil_compressibility" in self._interpolators:
                # Get compressibility at bubble point
                co_pb = self.pt_interpolate(
                    name="oil_compressibility",
                    pressure=pb[undersaturated],
                    temperature=t[undersaturated],
                )
                # Get compressibility at current pressure for average calculation
                co_p = self.pt_interpolate(
                    name="oil_compressibility",
                    pressure=p[undersaturated],
                    temperature=t[undersaturated],
                )

                if co_pb is not None:
                    # Use average compressibility between Pb and P (McCain method)
                    if co_p is not None:
                        co_avg = 0.5 * (co_pb + co_p)
                    else:
                        # Fallback to bubble point compressibility if P is out of table range
                        co_avg = co_pb

                    # Apply compression: Bo decreases with pressure above Pb
                    # Bo(P) = Bob * exp(-co_avg * (P - Pb))
                    result[undersaturated] = bob * np.exp(
                        -co_avg * (p[undersaturated] - pb[undersaturated])
                    )
                else:
                    # Interpolator exists but returned None (shouldn't happen)
                    result[undersaturated] = bob
            else:
                # No compressibility table provided
                result[undersaturated] = bob
                if self.warn_on_extrapolation:
                    logger.warning(
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
        if "solution_gas_to_oil_ratio" not in self._interpolators:
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
            result[saturated] = self.pt_interpolate(
                name="solution_gas_to_oil_ratio",
                pressure=p[saturated],
                temperature=t[saturated],
            )

        # Undersaturated: P > Pb - Rs = Rsb (constant)
        undersaturated = np.invert(saturated)
        if np.any(undersaturated):
            # Use Rs at bubble point
            result[undersaturated] = self.pt_interpolate(
                name="solution_gas_to_oil_ratio",
                pressure=pb[undersaturated],
                temperature=t[undersaturated],
            )

        return float(result) if result.size == 1 else result

    def water_bubble_point_pressure(
        self, pressure: QueryType, temperature: QueryType, salinity: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get water bubble point pressure at given pressure(s), temperature(s), and salinity(ies)."""
        return self.pts_interpolate(
            name="water_bubble_point_pressure",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
        )

    def water_viscosity(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get water viscosity at given pressure(s), temperature(s), and salinity(ies).

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param salinity: Salinity(ies) in ppm NaCl. If None, uses first salinity value from table.
        :return: Water viscosity in cP
        """
        if salinity is None:
            # Use the first (or only) salinity value as default
            if self.default_salinity is None:
                raise ValidationError(
                    "Salinity required but no salinity grid provided in PVT tables"
                )
            salinity = self.default_salinity

        return self.pts_interpolate(
            name="water_viscosity",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,  # type: ignore[arg-type]
        )

    def water_compressibility(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get water compressibility at given pressure(s), temperature(s), and salinity(ies).

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param salinity: Salinity(ies) in ppm NaCl. If None, uses first salinity value from table.
        :return: Water compressibility in psi⁻¹
        """
        if salinity is None:
            if self.default_salinity is None:
                raise ValidationError(
                    "Salinity required but no salinity grid provided in PVT tables"
                )
            salinity = self.default_salinity

        return self.pts_interpolate(
            name="water_compressibility",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,  # type: ignore[arg-type]
        )

    def water_density(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get water density at given pressure(s), temperature(s), and salinity(ies).

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param salinity: Salinity(ies) in ppm NaCl. If None, uses first salinity value from table.
        :return: Water density in lbm/ft³
        """
        if salinity is None:
            if self.default_salinity is None:
                raise ValidationError(
                    "Salinity required but no salinity grid provided in PVT tables"
                )
            salinity = self.default_salinity

        return self.pts_interpolate(
            name="water_density",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,  # type: ignore[arg-type]
        )

    def water_formation_volume_factor(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get water formation volume factor at given pressure(s), temperature(s), and salinity(ies).

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param salinity: Salinity(ies) in ppm NaCl. If None, uses first salinity value from table.
        :return: Water formation volume factor in bbl/STB
        """
        if salinity is None:
            if self.default_salinity is None:
                raise ValidationError(
                    "Salinity required but no salinity grid provided in PVT tables"
                )
            salinity = self.default_salinity

        return self.pts_interpolate(
            name="water_formation_volume_factor",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,  # type: ignore[arg-type]
        )

    def gas_viscosity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas viscosity at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="gas_viscosity", pressure=pressure, temperature=temperature
        )

    def gas_compressibility(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas compressibility at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="gas_compressibility", pressure=pressure, temperature=temperature
        )

    def gas_gravity(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas gravity at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="gas_gravity", pressure=pressure, temperature=temperature
        )

    def gas_molecular_weight(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas molecular weight at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="gas_molecular_weight", pressure=pressure, temperature=temperature
        )

    def gas_density(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas density at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="gas_density", pressure=pressure, temperature=temperature
        )

    def gas_formation_volume_factor(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas formation volume factor at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="gas_formation_volume_factor",
            pressure=pressure,
            temperature=temperature,
        )

    def gas_compressibility_factor(
        self, pressure: QueryType, temperature: QueryType
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """Get gas compressibility factor at given pressure(s) and temperature(s)."""
        return self.pt_interpolate(
            name="gas_compressibility_factor",
            pressure=pressure,
            temperature=temperature,
        )

    def gas_solubility_in_water(
        self,
        pressure: QueryType,
        temperature: QueryType,
        salinity: typing.Optional[QueryType] = None,
    ) -> typing.Optional[typing.Union[float, np.ndarray]]:
        """
        Get gas solubility in water at given pressure(s), temperature(s), and salinity(ies).

        :param pressure: Pressure(s) in psi
        :param temperature: Temperature(s) in °F
        :param salinity: Salinity(ies) in ppm NaCl. If None, uses first salinity value from table.
        :return: Gas solubility in water in SCF/STB
        """
        if salinity is None:
            if self.default_salinity is None:
                raise ValidationError(
                    "Salinity required but no salinity grid provided in PVT tables"
                )
            salinity = self.default_salinity

        return self.pts_interpolate(
            name="gas_solubility_in_water",
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,  # type: ignore[arg-type]
        )


def _validate_table_shape(
    table: typing.Optional[np.ndarray],
    expected_shape: typing.Tuple[int, ...],
    name: str,
) -> None:
    """Helpers to validate pre-computed table shapes."""
    if table is not None and table.shape != expected_shape:
        raise ValidationError(
            f"`{name}` shape {table.shape} does not match expected {expected_shape}"
        )


def build_pvt_table_data(
    pressures: OneDimensionalGrid,
    temperatures: OneDimensionalGrid,
    oil_specific_gravity: float = 0.85,  # Typical light-medium crude oil
    gas_gravity: typing.Optional[float] = None,
    water_salinity: typing.Optional[float] = None,
    estimated_solution_gor: typing.Optional[float] = None,
    salinities: typing.Optional[OneDimensionalGrid] = None,
    solution_gas_to_oil_ratios: typing.Optional[OneDimensionalGrid] = None,
    bubble_point_pressures: typing.Optional[
        typing.Union[OneDimensionalGrid, TwoDimensionalGrid]
    ] = None,
    reservoir_gas: typing.Optional[str] = None,
    build_oil_properties: bool = True,
    build_water_properties: bool = True,
    build_gas_properties: bool = True,
    # Pre-computed 2D oil tables (n_pressures, n_temperatures)
    oil_viscosity_table: typing.Optional[TwoDimensionalGrid] = None,
    oil_compressibility_table: typing.Optional[TwoDimensionalGrid] = None,
    oil_specific_gravity_table: typing.Optional[TwoDimensionalGrid] = None,
    oil_api_gravity_table: typing.Optional[TwoDimensionalGrid] = None,
    oil_density_table: typing.Optional[TwoDimensionalGrid] = None,
    oil_formation_volume_factor_table: typing.Optional[TwoDimensionalGrid] = None,
    solution_gas_to_oil_ratio_table: typing.Optional[TwoDimensionalGrid] = None,
    # Pre-computed 2D gas tables (n_pressures, n_temperatures)
    gas_viscosity_table: typing.Optional[TwoDimensionalGrid] = None,
    gas_compressibility_table: typing.Optional[TwoDimensionalGrid] = None,
    gas_gravity_table: typing.Optional[TwoDimensionalGrid] = None,
    gas_molecular_weight_table: typing.Optional[TwoDimensionalGrid] = None,
    gas_density_table: typing.Optional[TwoDimensionalGrid] = None,
    gas_formation_volume_factor_table: typing.Optional[TwoDimensionalGrid] = None,
    gas_compressibility_factor_table: typing.Optional[TwoDimensionalGrid] = None,
    # Pre-computed 3D water tables (n_pressures, n_temperatures, n_salinities)
    water_bubble_point_pressure_table: typing.Optional[ThreeDimensionalGrid] = None,
    water_viscosity_table: typing.Optional[ThreeDimensionalGrid] = None,
    water_compressibility_table: typing.Optional[ThreeDimensionalGrid] = None,
    water_density_table: typing.Optional[ThreeDimensionalGrid] = None,
    water_formation_volume_factor_table: typing.Optional[ThreeDimensionalGrid] = None,
    gas_solubility_in_water_table: typing.Optional[ThreeDimensionalGrid] = None,
) -> PVTTableData:
    """
    Build comprehensive PVT table data using empirical correlations.

    This factory function generates complete 2D (P-T) and 3D (P-T-Salinity) lookup tables
    for fluid properties. It uses the same correlations as `reservoir_model()` but pre-computes
    them on a grid for fast interpolation during simulation.

    **IMPORTANT: Table Dimensions**
    - Oil properties: 2D tables with shape (n_pressures, n_temperatures)
    - Gas properties: 2D tables with shape (n_pressures, n_temperatures)
    - Water properties: 3D tables with shape (n_pressures, n_temperatures, n_salinities)
    - Bubble point: 1D shape (n_temperatures) OR 2D shape (n_solution_gor, n_temperatures)

    **Water Properties are 3D**
    Water properties depend on salinity, so all water tables are 3D (P, T, S).
    If `salinities` is not provided, a default single salinity value will be used,
    resulting in a 3D table with shape (n_p, n_t, 1).

    :param pressures: 1D array of pressures (psi), e.g. np.linspace(500, 5000, 50)
    :param temperatures: 1D array of temperatures (°F), e.g. np.linspace(100, 250, 30)

    :param oil_specific_gravity: Oil specific gravity (dimensionless), e.g. 0.85
    :param gas_gravity: Gas specific gravity (dimensionless), e.g. 0.65
    :param water_salinity: Single salinity value (ppm NaCl), e.g. 35000.
        Used if `salinities` array is not provided.
    :param estimated_solution_gor: Estimated solution gas-to-oil ratio (SCF/STB) for
        1D bubble point calculation when `solution_gas_to_oil_ratios` is not provided.
        If None, defaults to 500 SCF/STB (typical value for medium crude oils).
        This only affects the bubble point curve, not the actual Rs(P,T) table.
    :param salinities: 1D array of salinities (ppm) for 3D water property tables.
        If None, uses [water_salinity] as single-value array.
    :param solution_gas_to_oil_ratios: 1D array of Rs values (SCF/STB) for 2D bubble point table
    :param reservoir_gas: Gas type (e.g. "CO2", "Methane"), defaults to `constants.RESERVOIR_GAS_NAME`

    :param build_oil_properties: If True, compute oil property tables (2D)
    :param build_water_properties: If True, compute water property tables (3D)
    :param build_gas_properties: If True, compute gas property tables (2D)

    :param oil_viscosity_table: Optional pre-computed oil viscosity (n_p, n_t)
    :param oil_compressibility_table: Optional pre-computed oil compressibility (n_p, n_t)
    :param oil_specific_gravity_table: Optional pre-computed oil SG (n_p, n_t)
    :param oil_api_gravity_table: Optional pre-computed API gravity (n_p, n_t)
    :param oil_density_table: Optional pre-computed oil density (n_p, n_t)
    :param oil_formation_volume_factor_table: Optional pre-computed Bo (n_p, n_t)
    :param solution_gas_to_oil_ratio_table: Optional pre-computed Rs (n_p, n_t)
    :param bubble_point_pressures: Optional Pb - 1D (n_t) or 2D (n_rs, n_t)

    :param gas_viscosity_table: Optional pre-computed gas viscosity (n_p, n_t)
    :param gas_compressibility_table: Optional pre-computed gas compressibility (n_p, n_t)
    :param gas_gravity_table: Optional pre-computed gas gravity (n_p, n_t)
    :param gas_molecular_weight_table: Optional pre-computed MW (n_p, n_t)
    :param gas_density_table: Optional pre-computed gas density (n_p, n_t)
    :param gas_formation_volume_factor_table: Optional pre-computed Bg (n_p, n_t)
    :param gas_compressibility_factor_table: Optional pre-computed Z-factor (n_p, n_t)

    :param water_viscosity_table: Optional pre-computed water viscosity (n_p, n_t, n_s)
    :param water_compressibility_table: Optional pre-computed water compressibility (n_p, n_t, n_s)
    :param water_density_table: Optional pre-computed water density (n_p, n_t, n_s)
    :param water_formation_volume_factor_table: Optional pre-computed Bw (n_p, n_t, n_s)
    :param water_bubble_point_pressure_table: Optional pre-computed water Pb (n_p, n_t, n_s)
    :param gas_solubility_in_water_table: Optional pre-computed Rsw (n_p, n_t, n_s)

    :return: `PVTTableData` instance with all computed/provided properties

    Example:
    ```python
    # Build comprehensive table data for black oil system
    pressures = np.linspace(500, 5000, 50)
    temperatures = np.linspace(100, 250, 30)
    salinities = np.array([0, 35000, 70000, 100000])  # Multiple salinities

    table_data = build_pvt_table_data(
        pressures=pressures,
        temperatures=temperatures,
        salinities=salinities,  # Will create 3D water tables
        oil_specific_gravity=0.85,
        gas_gravity=0.65,
        reservoir_gas="Methane",
    )

    # Create table with interpolators for fast lookup
    tables = PVTTables(
        table_data=table_data,
        interpolation_method="cubic",
        validate_tables=True,
        warn_on_extrapolation=True,
    )

    # Query properties during simulation
    mu_o = tables.oil_viscosity(pressure=2500, temperature=180)
    mu_w = tables.water_viscosity(pressure=2500, temperature=180, salinity=35000)
    ```
    """
    if pressures.ndim != 1:
        raise ValidationError("`pressures` must be 1-dimensional")
    if temperatures.ndim != 1:
        raise ValidationError("`temperatures` must be 1-dimensional")
    if not np.all(np.diff(pressures) > 0):
        raise ValidationError("`pressures` must be strictly increasing")
    if not np.all(np.diff(temperatures) > 0):
        raise ValidationError("`temperatures` must be strictly increasing")

    # Validate solution_gas_to_oil_ratios if provided
    if solution_gas_to_oil_ratios is not None:
        if solution_gas_to_oil_ratios.ndim != 1:
            raise ValidationError("`solution_gas_to_oil_ratios` must be 1-dimensional")
        if not np.all(np.diff(solution_gas_to_oil_ratios) > 0):
            raise ValidationError(
                "`solution_gas_to_oil_ratios` must be strictly increasing"
            )

    n_p = len(pressures)
    n_t = len(temperatures)

    # Set defaults
    reservoir_gas = reservoir_gas or c.RESERVOIR_GAS_NAME
    reservoir_gas = typing.cast(str, reservoir_gas)
    water_salinity_value = (
        water_salinity if water_salinity is not None else c.DEFAULT_WATER_SALINITY_PPM
    )
    dtype = get_dtype()

    # Handle salinity array
    if salinities is None:
        # Use single salinity value
        salinities = np.array([water_salinity_value], dtype=dtype)
    else:
        if salinities.ndim != 1:
            raise ValidationError("`salinities` must be 1-dimensional")
        if not np.all(np.diff(salinities) > 0):
            raise ValidationError("`salinities` must be strictly increasing")

    n_s = len(salinities)

    # Validate pre-computed 2D table shapes (n_p, n_t)
    expected_2d = (n_p, n_t)
    _validate_table_shape(
        table=oil_viscosity_table,
        expected_shape=expected_2d,
        name="oil_viscosity_table",
    )
    _validate_table_shape(
        table=oil_compressibility_table,
        expected_shape=expected_2d,
        name="oil_compressibility_table",
    )
    _validate_table_shape(
        table=oil_specific_gravity_table,
        expected_shape=expected_2d,
        name="oil_specific_gravity_table",
    )
    _validate_table_shape(
        table=oil_api_gravity_table,
        expected_shape=expected_2d,
        name="oil_api_gravity_table",
    )
    _validate_table_shape(
        table=oil_density_table, expected_shape=expected_2d, name="oil_density_table"
    )
    _validate_table_shape(
        table=oil_formation_volume_factor_table,
        expected_shape=expected_2d,
        name="oil_formation_volume_factor_table",
    )
    _validate_table_shape(
        table=solution_gas_to_oil_ratio_table,
        expected_shape=expected_2d,
        name="solution_gas_to_oil_ratio_table",
    )
    _validate_table_shape(
        table=gas_viscosity_table,
        expected_shape=expected_2d,
        name="gas_viscosity_table",
    )
    _validate_table_shape(
        table=gas_compressibility_table,
        expected_shape=expected_2d,
        name="gas_compressibility_table",
    )
    _validate_table_shape(
        table=gas_gravity_table, expected_shape=expected_2d, name="gas_gravity_table"
    )
    _validate_table_shape(
        table=gas_molecular_weight_table,
        expected_shape=expected_2d,
        name="gas_molecular_weight_table",
    )
    _validate_table_shape(
        table=gas_density_table, expected_shape=expected_2d, name="gas_density_table"
    )
    _validate_table_shape(
        table=gas_formation_volume_factor_table,
        expected_shape=expected_2d,
        name="gas_formation_volume_factor_table",
    )
    _validate_table_shape(
        table=gas_compressibility_factor_table,
        expected_shape=expected_2d,
        name="gas_compressibility_factor_table",
    )

    # Validate pre-computed 3D table shapes (n_p, n_t, n_s)
    expected_3d = (n_p, n_t, n_s)
    _validate_table_shape(
        table=water_bubble_point_pressure_table,
        expected_shape=expected_3d,
        name="water_bubble_point_pressure_table",
    )
    _validate_table_shape(
        table=water_viscosity_table,
        expected_shape=expected_3d,
        name="water_viscosity_table",
    )
    _validate_table_shape(
        table=water_compressibility_table,
        expected_shape=expected_3d,
        name="water_compressibility_table",
    )
    _validate_table_shape(
        table=water_density_table,
        expected_shape=expected_3d,
        name="water_density_table",
    )
    _validate_table_shape(
        table=water_formation_volume_factor_table,
        expected_shape=expected_3d,
        name="water_formation_volume_factor_table",
    )
    _validate_table_shape(
        table=gas_solubility_in_water_table,
        expected_shape=expected_3d,
        name="gas_solubility_in_water_table",
    )

    # Validate bubble_point_pressures shape if provided
    if bubble_point_pressures is not None:
        if bubble_point_pressures.ndim == 1:
            if len(bubble_point_pressures) != n_t:
                raise ValidationError(
                    f"`bubble_point_pressures` 1D length {len(bubble_point_pressures)} "
                    f"does not match n_temperatures={n_t}"
                )
        elif bubble_point_pressures.ndim == 2:
            if solution_gas_to_oil_ratios is None:
                raise ValidationError(
                    "2D `bubble_point_pressures` requires `solution_gas_to_oil_ratios` to be provided"
                )
            expected_bp_shape = (len(solution_gas_to_oil_ratios), n_t)
            if bubble_point_pressures.shape != expected_bp_shape:
                raise ValidationError(
                    f"`bubble_point_pressures` shape {bubble_point_pressures.shape} "
                    f"does not match expected {expected_bp_shape}"
                )
        else:
            raise ValidationError("`bubble_point_pressures` must be 1D or 2D")

    logger.info(
        f"Building PVT tables: {n_p} pressures x {n_t} temperatures x {n_s} salinities",
    )
    # CREATE MESHGRIDS
    # 2D meshgrid for oil and gas properties: (n_p, n_t)
    pressure_grid_2d, temperature_grid_2d = np.meshgrid(
        pressures, temperatures, indexing="ij"
    )
    # 3D meshgrid for water properties: (n_p, n_t, n_s)
    pressure_grid_3d, temperature_grid_3d, salinity_grid_3d = np.meshgrid(
        pressures, temperatures, salinities, indexing="ij"
    )

    if gas_gravity is None:
        gas_gravity = compute_gas_gravity(gas=reservoir_gas)
        logger.debug(f"Computed gas gravity = {gas_gravity:.4f} for {reservoir_gas}")

    # BUILD GAS PROPERTIES (2D TABLES)
    if build_gas_properties:
        logger.debug("Building gas properties...")
        # Gas gravity (usually constant)
        if gas_gravity_table is None:
            gas_gravity_table = np.full((n_p, n_t), gas_gravity, dtype=dtype)

        # Gas molecular weight
        if gas_molecular_weight_table is None:
            gas_molecular_weight_table = build_gas_molecular_weight_grid(
                gas_gravity_grid=gas_gravity_table
            )

        # Gas Z-factor
        if gas_compressibility_factor_table is None:
            gas_compressibility_factor_table = build_gas_compressibility_factor_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_gravity_grid=gas_gravity_table,
            )

        # Gas formation volume factor
        if gas_formation_volume_factor_table is None:
            gas_formation_volume_factor_table = build_gas_formation_volume_factor_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_compressibility_factor_grid=gas_compressibility_factor_table,
            )

        # Gas compressibility
        if gas_compressibility_table is None:
            gas_compressibility_table = build_gas_compressibility_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_gravity_grid=gas_gravity_table,
                gas_compressibility_factor_grid=gas_compressibility_factor_table,
            )

        # Gas density
        if gas_density_table is None:
            gas_density_table = build_gas_density_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                gas_gravity_grid=gas_gravity_table,
                gas_compressibility_factor_grid=gas_compressibility_factor_table,
            )

        # Gas viscosity
        if gas_viscosity_table is None:
            gas_viscosity_table = build_gas_viscosity_grid(
                temperature_grid=temperature_grid_2d,
                gas_density_grid=gas_density_table,
                gas_molecular_weight_grid=gas_molecular_weight_table,
            )
        logger.debug("Gas properties built")

    # BUILD WATER PROPERTIES (3D TABLES)
    if build_water_properties:
        logger.debug("Building water properties...")
        # Water viscosity: μw(P, T, S)
        if water_viscosity_table is None:
            water_viscosity_table = build_water_viscosity_grid(
                temperature_grid=temperature_grid_3d,
                salinity_grid=salinity_grid_3d,
                pressure_grid=pressure_grid_3d,
            )

        # Gas solubility in water: Rsw(P, T, S)
        if gas_solubility_in_water_table is None:
            gas_solubility_in_water_table = build_gas_solubility_in_water_grid(
                pressure_grid=pressure_grid_3d,
                temperature_grid=temperature_grid_3d,
                salinity_grid=salinity_grid_3d,
                gas=reservoir_gas,
            )

        # Water bubble point pressure: Pb,w(P, T, S)
        if water_bubble_point_pressure_table is None:
            water_bubble_point_pressure_table = build_water_bubble_point_pressure_grid(
                temperature_grid=temperature_grid_3d,
                gas_solubility_in_water_grid=gas_solubility_in_water_table,
                salinity_grid=salinity_grid_3d,
                gas=reservoir_gas,
            )

        # Gas-free water FVF (temporary variable for compressibility and density calculations)
        gas_free_water_fvf_temp = build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=pressure_grid_3d,
            temperature_grid=temperature_grid_3d,
        )

        # Need gas FVF for water compressibility calculation
        # Broadcast 2D gas FVF to 3D if gas properties were built
        if build_gas_properties and gas_formation_volume_factor_table is not None:
            # Broadcast (n_p, n_t) to (n_p, n_t, n_s)
            gas_fvf_3d = np.broadcast_to(
                gas_formation_volume_factor_table[:, :, np.newaxis], (n_p, n_t, n_s)
            ).copy()
        else:
            # Use default gas FVF
            gas_fvf_3d = np.ones((n_p, n_t, n_s))

        # Need gas gravity for water density calculation
        if build_gas_properties and gas_gravity_table is not None:
            # Broadcast (n_p, n_t) to (n_p, n_t, n_s)
            gas_gravity_3d = np.broadcast_to(
                gas_gravity_table[:, :, np.newaxis], (n_p, n_t, n_s)
            ).copy()
        else:
            # Use default gas gravity. This will mostlikely not be reached.
            gas_gravity_3d = np.full((n_p, n_t, n_s), 0.65, dtype=dtype)

        # Water compressibility: Cw(P, T, S)
        if water_compressibility_table is None:
            water_compressibility_table = build_water_compressibility_grid(
                pressure_grid=pressure_grid_3d,
                temperature_grid=temperature_grid_3d,
                bubble_point_pressure_grid=water_bubble_point_pressure_table,
                gas_formation_volume_factor_grid=gas_fvf_3d,
                gas_solubility_in_water_grid=gas_solubility_in_water_table,
                gas_free_water_formation_volume_factor_grid=gas_free_water_fvf_temp,
            )

        # Water density: ρw(P, T, S)
        if water_density_table is None:
            water_density_table = build_water_density_grid(
                pressure_grid=pressure_grid_3d,
                temperature_grid=temperature_grid_3d,
                gas_gravity_grid=gas_gravity_3d,
                salinity_grid=salinity_grid_3d,
                gas_solubility_in_water_grid=gas_solubility_in_water_table,
                gas_free_water_formation_volume_factor_grid=gas_free_water_fvf_temp,
            )

        # Water formation volume factor: Bw(P, T, S)
        if water_formation_volume_factor_table is None:
            water_formation_volume_factor_table = (
                build_water_formation_volume_factor_grid(
                    water_density_grid=water_density_table,
                    salinity_grid=salinity_grid_3d,
                )
            )
        logger.debug("Water properties built")

    # BUILD OIL PROPERTIES (2D TABLES)
    if build_oil_properties:
        logger.debug("Building oil properties...")
        # Oil specific gravity (usually constant)
        if oil_specific_gravity_table is None:
            oil_specific_gravity_table = np.full(
                (n_p, n_t), oil_specific_gravity, dtype=dtype
            )

        # Oil API gravity
        if oil_api_gravity_table is None:
            oil_api_gravity_table = build_oil_api_gravity_grid(
                oil_specific_gravity_grid=oil_specific_gravity_table
            )

        # BUBBLE POINT PRESSURE (1D or 2D)
        if bubble_point_pressures is None:
            if solution_gas_to_oil_ratios is not None:
                # Build 2D bubble point table: Pb(Rs, T)
                n_rs = len(solution_gas_to_oil_ratios)
                bubble_point_pressures = np.zeros((n_rs, n_t), dtype=dtype)

                for i, rs_value in enumerate(solution_gas_to_oil_ratios):
                    rs_grid = np.full(n_t, rs_value, dtype=dtype)
                    bubble_point_pressures[i, :] = build_oil_bubble_point_pressure_grid(
                        gas_gravity_grid=np.full(n_t, gas_gravity, dtype=dtype),
                        oil_api_gravity_grid=build_oil_api_gravity_grid(
                            np.full(n_t, oil_specific_gravity, dtype=dtype)
                        ),
                        temperature_grid=temperatures,
                        solution_gas_to_oil_ratio_grid=rs_grid,
                    )
            else:
                # Build 1D bubble point table: Pb(T)
                # Use user-specified Rs estimate or default to 500 SCF/STB
                # The default of 500 is a moderate value typical for medium crude oils
                # and provides stable behavior for most reservoir conditions.
                if estimated_solution_gor is not None:
                    estimated_rs = estimated_solution_gor
                else:
                    estimated_rs = 500.0

                estimated_rs_grid = np.full(n_t, estimated_rs, dtype=dtype)
                oil_api = compute_oil_api_gravity(oil_specific_gravity)
                logger.debug(
                    f"Using Rs = {estimated_rs:.1f} SCF/STB for 1D bubble point (API = {oil_api:.1f}°)"
                )
                bubble_point_pressures = build_oil_bubble_point_pressure_grid(
                    gas_gravity_grid=np.full(n_t, gas_gravity, dtype=dtype),
                    oil_api_gravity_grid=build_oil_api_gravity_grid(
                        np.full(n_t, oil_specific_gravity, dtype=dtype)
                    ),
                    temperature_grid=temperatures,
                    solution_gas_to_oil_ratio_grid=estimated_rs_grid,
                )

        # Create 2D bubble point grid for property calculations
        if bubble_point_pressures.ndim == 1:
            # 1D: Broadcast Pb(T) to (n_p, n_t)
            bubble_point_pressure_grid_2d = np.broadcast_to(
                bubble_point_pressures[np.newaxis, :], (n_p, n_t)
            ).copy()
        else:
            # 2D: Pb(Rs, T) - compute Rs(P, T) iteratively if not provided
            if solution_gas_to_oil_ratio_table is None:
                # Use iterative solver to estimate Rs(P, T) without needing Pb
                logger.debug(
                    "Computing Rs(P, T) iteratively for 2D bubble point pressures..."
                )
                solution_gas_to_oil_ratio_table = (
                    build_estimated_solution_gas_to_oil_ratio_grid(
                        pressure_grid=pressure_grid_2d,
                        temperature_grid=temperature_grid_2d,
                        oil_api_gravity_grid=oil_api_gravity_table,
                        gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                    )
                )

            # Use the Rs(P,T) table to determine Pb at each (P,T)
            # For each (P, T), interpolate Pb from the 2D table Pb(Rs, T)
            bubble_point_pressure_grid_2d = np.zeros((n_p, n_t), dtype=dtype)
            pb_interp = RectBivariateSpline(
                x=solution_gas_to_oil_ratios,
                y=temperatures,
                z=bubble_point_pressures,
                kx=1,
                ky=1,
            )
            # Evaluate Pb at each Rs(P, T) value
            # Create temperature grid matching `solution_gas_to_oil_ratio_table` shape
            t_grid = np.broadcast_to(temperatures, (n_p, n_t))
            bubble_point_pressure_grid_2d = (
                pb_interp.ev(solution_gas_to_oil_ratio_table.ravel(), t_grid.ravel())
                .reshape(n_p, n_t)
                .astype(dtype)
            )

        # SOLUTION GOR TABLE: Rs(P, T)
        if solution_gas_to_oil_ratio_table is None:
            solution_gas_to_oil_ratio_table = build_solution_gas_to_oil_ratio_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                oil_api_gravity_grid=oil_api_gravity_table,
            )

        # OIL FORMATION VOLUME FACTOR & COMPRESSIBILITY
        # These have a circular dependency: oil FVF depends on oil compressibility, oil compressibility depends on oil FVF
        # So we use temporary a oil compressibility estimate for initial oil FVF, then compute actual oil compressibility,
        # then recalculate Bo with accurate oil compressibility values.
        need_bo = oil_formation_volume_factor_table is None
        need_co = oil_compressibility_table is None

        # Rs at bubble point will be computed lazily when needed for oil compressibility or viscosity
        rs_at_bp_grid: typing.Optional[TwoDimensionalGrid] = None
        gas_fvf_for_co: typing.Optional[TwoDimensionalGrid] = None

        # Compute Rs at bubble point (needed for Co calculation and viscosity)
        if need_co:
            rs_at_bp_grid = build_solution_gas_to_oil_ratio_grid(
                pressure_grid=bubble_point_pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                oil_api_gravity_grid=oil_api_gravity_table,
            )

            # Need gas FVF for compressibility calculation
            if build_gas_properties and gas_formation_volume_factor_table is not None:
                gas_fvf_for_co = gas_formation_volume_factor_table
            else:
                gas_fvf_for_co = np.ones((n_p, n_t), dtype=dtype)

        # Do initial Bo calculation with temporary Co estimate
        if need_bo:
            # Use typical oil compressibility as temporary value for initial Bo
            oil_compressibility_temp = np.full((n_p, n_t), 1e-5, dtype=dtype)
            oil_formation_volume_factor_table = build_oil_formation_volume_factor_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                oil_specific_gravity_grid=oil_specific_gravity_table,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,
                oil_compressibility_grid=oil_compressibility_temp,
            )

        # Compute actual Co using initial Bo estimate
        if need_co:
            assert rs_at_bp_grid is not None  # Guaranteed by need_co check above
            assert gas_fvf_for_co is not None
            oil_compressibility_table = build_oil_compressibility_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                oil_api_gravity_grid=oil_api_gravity_table,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                gor_at_bubble_point_pressure_grid=rs_at_bp_grid,
                gas_formation_volume_factor_grid=gas_fvf_for_co,
                oil_formation_volume_factor_grid=oil_formation_volume_factor_table,
            )

        # Recalculate Bo with accurate Co values
        if need_bo and need_co:
            logger.debug("Recalculating Bo with accurate Co values...")
            oil_formation_volume_factor_table = build_oil_formation_volume_factor_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                oil_specific_gravity_grid=oil_specific_gravity_table,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,
                oil_compressibility_grid=oil_compressibility_table,
            )

        # OIL DENSITY: ρo(P, T)
        if oil_density_table is None:
            oil_density_table = build_live_oil_density_grid(
                oil_api_gravity_grid=oil_api_gravity_table,
                gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,
                formation_volume_factor_grid=oil_formation_volume_factor_table,
            )

        # OIL VISCOSITY: μo(P, T)
        if oil_viscosity_table is None:
            if rs_at_bp_grid is None:
                # Compute Rs at bubble point for each (P, T) pair
                # When P = Pb, oil is saturated: Rs(Pb, T) = Rsb(T)
                rs_at_bp_grid = build_solution_gas_to_oil_ratio_grid(
                    pressure_grid=bubble_point_pressure_grid_2d,
                    temperature_grid=temperature_grid_2d,
                    bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                    gas_gravity_grid=np.full((n_p, n_t), gas_gravity, dtype=dtype),
                    oil_api_gravity_grid=oil_api_gravity_table,
                )

            oil_viscosity_table = build_oil_viscosity_grid(
                pressure_grid=pressure_grid_2d,
                temperature_grid=temperature_grid_2d,
                bubble_point_pressure_grid=bubble_point_pressure_grid_2d,
                oil_specific_gravity_grid=oil_specific_gravity_table,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_table,
                gor_at_bubble_point_pressure_grid=rs_at_bp_grid,
            )
        logger.debug("Oil properties built")

    table_data = PVTTableData(
        pressures=pressures,
        temperatures=temperatures,
        salinities=salinities,
        solution_gas_oil_ratios=solution_gas_to_oil_ratios,
        bubble_point_pressures=bubble_point_pressures,
        oil_viscosity_table=oil_viscosity_table,
        oil_compressibility_table=oil_compressibility_table,
        oil_specific_gravity_table=oil_specific_gravity_table,
        oil_api_gravity_table=oil_api_gravity_table,
        oil_density_table=oil_density_table,
        oil_formation_volume_factor_table=oil_formation_volume_factor_table,
        solution_gas_to_oil_ratio_table=solution_gas_to_oil_ratio_table,
        water_bubble_point_pressure_table=water_bubble_point_pressure_table,
        water_viscosity_table=water_viscosity_table,
        water_compressibility_table=water_compressibility_table,
        water_density_table=water_density_table,
        water_formation_volume_factor_table=water_formation_volume_factor_table,
        gas_solubility_in_water_table=gas_solubility_in_water_table,
        gas_viscosity_table=gas_viscosity_table,
        gas_compressibility_table=gas_compressibility_table,
        gas_gravity_table=gas_gravity_table,
        gas_molecular_weight_table=gas_molecular_weight_table,
        gas_density_table=gas_density_table,
        gas_formation_volume_factor_table=gas_formation_volume_factor_table,
        gas_compressibility_factor_table=gas_compressibility_factor_table,
    )
    return table_data
