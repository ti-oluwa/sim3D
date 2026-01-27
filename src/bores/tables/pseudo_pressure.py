from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import typing

from cachetools import LFUCache
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

from bores._precision import get_dtype
from bores.constants import c
from bores.errors import ValidationError
from bores.types import FloatOrArray


logger = logging.getLogger(__name__)

__all__ = ["GasPseudoPressureTable", "build_gas_pseudo_pressure_table"]


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
            y_points = np.array([integrand(p) for p in p_points])  # type: ignore
            result = float(np.trapezoid(y=y_points, x=p_points))

    # Apply sign based on integration direction
    if pressure < reference_pressure:
        result = -result

    return float(result)


def _supports_vectorization(
    z_factor_func: typing.Callable[[FloatOrArray], FloatOrArray],
    viscosity_func: typing.Callable[[FloatOrArray], FloatOrArray],
) -> bool:
    """
    Check if both z_factor and viscosity functions support vectorized operations.

    :param z_factor_func: Z-factor function
    :param viscosity_func: Viscosity function
    :return: True if both functions support arrays
    """
    z_supports = getattr(z_factor_func, "_supports_arrays", False)
    mu_supports = getattr(viscosity_func, "_supports_arrays", False)
    return z_supports and mu_supports


def _build_pseudo_pressure_table_vectorized(
    pressures: np.typing.NDArray,
    z_factor_func: typing.Callable[[FloatOrArray], FloatOrArray],
    viscosity_func: typing.Callable[[FloatOrArray], FloatOrArray],
    reference_pressure: float,
    dtype: typing.Optional[np.typing.DTypeLike] = None,
) -> np.typing.NDArray:
    """
    Build entire pseudo-pressure table using vectorized operations.

    This is much faster than computing each point individually because:
    - Single vectorized call to Z(P_array) and μ(P_array)
    - Vectorized integration using cumulative_trapezoid
    - No threading overhead needed

    :param pressures: Array of pressure points
    :param z_factor_func: Vectorized Z-factor function
    :param viscosity_func: Vectorized viscosity function
    :param reference_pressure: Reference pressure
    :return: Array of pseudo-pressures
    """
    # Clamp pressures
    p_clamped = np.maximum(pressures, 1.0)

    # Single vectorized call for all pressures
    Z_array = np.asarray(z_factor_func(p_clamped))
    mu_array = np.asarray(viscosity_func(p_clamped))

    # Validate shapes
    if Z_array.shape != pressures.shape or mu_array.shape != pressures.shape:
        raise ValueError(
            f"Shape mismatch: P={pressures.shape}, Z={Z_array.shape}, μ={mu_array.shape}"
        )

    # Handle invalid values
    invalid_Z = (Z_array <= 0) | ~np.isfinite(Z_array)
    invalid_mu = (mu_array <= 0) | ~np.isfinite(mu_array)

    if np.any(invalid_Z):
        logger.warning(f"Clamping {np.sum(invalid_Z)} invalid Z-factor values")
        Z_array = np.maximum(Z_array, 0.01)

    if np.any(invalid_mu):
        logger.warning(f"Clamping {np.sum(invalid_mu)} invalid viscosity values")
        mu_array = np.maximum(mu_array, 0.001)

    # Compute integrand: 2*P / (μ*Z)
    integrand_array = 2.0 * p_clamped / (mu_array * Z_array)

    # Handle invalid integrand values
    invalid = ~np.isfinite(integrand_array) | (integrand_array < 0)
    if np.any(invalid):
        logger.warning(f"Setting {np.sum(invalid)} invalid integrand values to zero")
        integrand_array = np.where(invalid, 0.0, integrand_array)

    # Cumulative integration from reference pressure
    # Find index closest to reference pressure
    ref_idx = np.searchsorted(pressures, reference_pressure)

    if ref_idx == 0:
        # Reference is at or below minimum, hence we integrate forward only
        pseudo_pressures = cumulative_trapezoid(integrand_array, pressures, initial=0.0)
    elif ref_idx >= len(pressures):
        # Reference is at or above maximum, hence we integrate backward only
        pseudo_pressures = -cumulative_trapezoid(
            integrand_array[::-1], pressures[::-1], initial=0.0
        )[::-1]
    else:
        # Reference is in the middle so we integrate both directions
        # Backward from ref to start
        backward = -cumulative_trapezoid(
            integrand_array[: ref_idx + 1][::-1],
            pressures[: ref_idx + 1][::-1],
            initial=0.0,
        )[::-1]

        # Forward from ref to end
        forward = cumulative_trapezoid(
            integrand_array[ref_idx:], pressures[ref_idx:], initial=0.0
        )
        # Then we combine
        pseudo_pressures = np.concatenate([backward[:-1], forward])

    return np.ascontiguousarray(pseudo_pressures, dtype=dtype)


def _build_pseudo_pressure_table_scalar(
    pressures: np.typing.NDArray,
    z_factor_func: typing.Callable[[float], float],
    viscosity_func: typing.Callable[[float], float],
    reference_pressure: float,
    max_workers: int,
    dtype: typing.Optional[np.typing.DTypeLike] = None,
) -> np.typing.NDArray:
    """
    Build pseudo-pressure table using threaded scalar computation.

    Uses `ThreadPoolExecutor` to parallelize individual integrations.

    :param pressures: Array of pressure points
    :param z_factor_func: Z-factor function
    :param viscosity_func: Viscosity function
    :param reference_pressure: Reference pressure
    :param max_workers: Maximum number of threads to use
    :return: Array of pseudo-pressures
    """

    def _compute(pressure: float) -> float:
        return compute_gas_pseudo_pressure(
            pressure=pressure,
            z_factor_func=z_factor_func,
            viscosity_func=viscosity_func,
            reference_pressure=reference_pressure,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pseudo_pressures = list(executor.map(_compute, pressures))

    return np.ascontiguousarray(pseudo_pressures, dtype=dtype)


def build_pseudo_pressure_table(
    pressures: np.typing.NDArray,
    z_factor_func: typing.Callable[[FloatOrArray], FloatOrArray],
    viscosity_func: typing.Callable[[FloatOrArray], FloatOrArray],
    reference_pressure: float,
    dtype: typing.Optional[np.typing.DTypeLike] = None,
) -> np.typing.NDArray:
    """
    Build pseudo-pressure table with automatic vectorization detection.

    If both z_factor_func and viscosity_func have `_supports_arrays=True`,
    uses fast vectorized computation. Otherwise uses threaded scalar computation.

    :param pressures: Array of pressure points
    :param z_factor_func: Z-factor function
    :param viscosity_func: Viscosity function
    :param reference_pressure: Reference pressure
    :return: Array of pseudo-pressures
    """
    points = len(pressures)
    if _supports_vectorization(z_factor_func, viscosity_func):
        logger.debug(
            f"Building pseudo-pressure table using vectorized computation ({points} points)..."
        )
        try:
            return _build_pseudo_pressure_table_vectorized(
                pressures=pressures,
                z_factor_func=z_factor_func,
                viscosity_func=viscosity_func,
                reference_pressure=reference_pressure,
                dtype=dtype,
            )
        except Exception as exc:
            logger.warning(
                f"Vectorized computation failed: {exc}, falling back to threaded scalar",
                exc_info=True,
            )

    logger.debug(
        f"Building pseudo-pressure table using threaded scalar computation ({points} points)..."
    )
    max_workers = min(8, points // 50 + 1)
    return _build_pseudo_pressure_table_scalar(
        pressures=pressures,
        z_factor_func=z_factor_func,  # type: ignore[arg-type]
        viscosity_func=viscosity_func,  # type: ignore[arg-type]
        reference_pressure=reference_pressure,
        max_workers=max_workers,
        dtype=dtype,
    )


class GasPseudoPressureTable:
    """
    Pre-computed gas pseudo-pressure table for fast lookup during simulation.

    Uses `np.interp` for fast linear interpolation.
    Supports both forward (pressure → pseudo-pressure) and inverse (pseudo-pressure → pressure)
    interpolation.
    """

    def __init__(
        self,
        z_factor_func: typing.Callable[[FloatOrArray], FloatOrArray],
        viscosity_func: typing.Callable[[FloatOrArray], FloatOrArray],
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
        dtype = get_dtype()
        self.pressures = np.logspace(
            np.log10(min_pressure), np.log10(max_pressure), points, dtype=dtype
        )

        # Compute pseudo-pressure at each point
        logger.info(f"Building pseudo-pressure table with {points} points...")
        self.pseudo_pressures = build_pseudo_pressure_table(
            pressures=self.pressures,
            z_factor_func=self.z_factor_func,
            viscosity_func=self.viscosity_func,
            reference_pressure=self.reference_pressure,
            dtype=dtype,
        )
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

        return 2.0 * pressure / (mu * Z)  # type: ignore[return-value]


_PSEUDO_PRESSURE_TABLE_CACHE: LFUCache[typing.Hashable, GasPseudoPressureTable] = (
    LFUCache(maxsize=100)
)
"""Global cache for pseudo-pressure tables"""

_pseudo_pressure_cache_lock = threading.Lock()
"""Thread-safe lock for pseudo-pressure table cache access"""


def build_gas_pseudo_pressure_table(
    z_factor_func: typing.Callable[[FloatOrArray], FloatOrArray],
    viscosity_func: typing.Callable[[FloatOrArray], FloatOrArray],
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
        with _pseudo_pressure_cache_lock:
            if cache_key in _PSEUDO_PRESSURE_TABLE_CACHE:
                logger.debug(f"Using cached pseudo-pressure table for key: {cache_key}")
                return _PSEUDO_PRESSURE_TABLE_CACHE[cache_key]

    # Build new table outside lock to avoid blocking other threads
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
        with _pseudo_pressure_cache_lock:
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
    with _pseudo_pressure_cache_lock:
        cache_size = len(_PSEUDO_PRESSURE_TABLE_CACHE)
        _PSEUDO_PRESSURE_TABLE_CACHE.clear()
    logger.info(f"Cleared {cache_size} cached pseudo-pressure tables")


def get_pseudo_pressure_table_cache_info() -> typing.Dict[str, typing.Any]:
    """Get information about the current cache state."""
    with _pseudo_pressure_cache_lock:
        return {
            "cache_size": len(_PSEUDO_PRESSURE_TABLE_CACHE),
            "cached_keys": list(_PSEUDO_PRESSURE_TABLE_CACHE.keys()),
        }
