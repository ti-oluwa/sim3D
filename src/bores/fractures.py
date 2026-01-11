"""
Fracture definition API for 3D reservoir models.
"""

import logging
import typing

import numba
import attrs
import numpy as np

from bores.constants import c
from bores.errors import ValidationError
from bores.models import ReservoirModel
from bores.types import ThreeDimensions
from bores._precision import get_dtype

__all__ = [
    "Fracture",
    "FractureGeometry",
    "apply_fracture",
    "apply_fractures",
    "validate_fracture",
    "FractureDefaults",
    "vertical_sealing_fault",
    "normal_fault_with_throw",
    "reverse_fault_with_throw",
    "inclined_sealing_fault",
    "damage_zone_fault",
    "conductive_fracture_network",
    "fault_with_throw_and_damage_zone",
]

logger = logging.getLogger(__name__)


@numba.njit(parallel=True)
def _mask_orientation_x(
    mask: np.ndarray,
    cell_min: int,
    cell_max: int,
    slope: float,
    intercept: float,
    z_min: int,
    z_max: int,
    y_min: int,
    y_max: int,
):
    """
    Build fracture mask for x-oriented fractures.

    :param mask: Output mask array to fill
    :param cell_min: Minimum x-index of fracture plane
    :param cell_max: Maximum x-index of fracture plane
    :param slope: Slope for inclined fractures (dz/dy)
    :param intercept: Z-intercept of fracture plane
    :param z_min: Minimum z-index extent
    :param z_max: Maximum z-index extent
    :param y_min: Minimum y-index lateral extent
    :param y_max: Maximum y-index lateral extent
    """
    nx, _, _ = mask.shape
    for i in numba.prange(cell_min, cell_max + 1):  # type: ignore[misc]
        if 0 <= i < nx:
            if abs(slope) < 1e-6:
                for j in range(y_min, y_max + 1):
                    for k in range(z_min, z_max + 1):
                        mask[i, j, k] = True
            else:
                for j in range(y_min, y_max + 1):
                    z_fracture = int(intercept + slope * j)
                    if z_min <= z_fracture <= z_max:
                        mask[i, j, z_fracture] = True


@numba.njit(parallel=True)
def _mask_orientation_y(
    mask: np.ndarray,
    cell_min: int,
    cell_max: int,
    slope: float,
    intercept: float,
    z_min: int,
    z_max: int,
    x_min: int,
    x_max: int,
):
    """
    Build fracture mask for y-oriented fractures.

    :param mask: Output mask array to fill
    :param cell_min: Minimum y-index of fracture plane
    :param cell_max: Maximum y-index of fracture plane
    :param slope: Slope for inclined fractures (dz/dx)
    :param intercept: Z-intercept of fracture plane
    :param z_min: Minimum z-index extent
    :param z_max: Maximum z-index extent
    :param x_min: Minimum x-index lateral extent
    :param x_max: Maximum x-index lateral extent
    """
    _, ny, _ = mask.shape
    for j in numba.prange(cell_min, cell_max + 1):  # type: ignore[misc]
        if 0 <= j < ny:
            if abs(slope) < 1e-6:
                for i in range(x_min, x_max + 1):
                    for k in range(z_min, z_max + 1):
                        mask[i, j, k] = True
            else:
                for i in range(x_min, x_max + 1):
                    z_fracture = int(intercept + slope * i)
                    if z_min <= z_fracture <= z_max:
                        mask[i, j, z_fracture] = True


@numba.njit(parallel=True)
def _mask_orientation_z(
    mask: np.ndarray,
    cell_min: int,
    cell_max: int,
    slope: float,
    intercept: float,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
):
    _, _, nz = mask.shape
    for k in numba.prange(cell_min, cell_max + 1):  # type: ignore[misc]
        if 0 <= k < nz:
            if abs(slope) < 1e-6:
                for i in range(x_min, x_max + 1):
                    for j in range(y_min, y_max + 1):
                        mask[i, j, k] = True
            else:
                for i in range(x_min, x_max + 1):
                    y_fracture = int(intercept + slope * i)
                    if y_min <= y_fracture <= y_max:
                        mask[i, y_fracture, k] = True


@numba.njit(parallel=True)
def _scale_permeability_x_boundary(
    perm_x: np.ndarray, mask: np.ndarray, scale: float
) -> None:
    """
    Scale x-direction permeability for cells in the fracture zone.

    This scales the permeability of all cells marked in the mask,
    which affects transmissibility for flow in the x-direction.
    """
    nx, ny, nz = perm_x.shape
    for i in numba.prange(nx):  # type: ignore[misc]
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    perm_x[i, j, k] *= scale


@numba.njit(parallel=True)
def _scale_permeability_y_boundary(
    perm_y: np.ndarray, mask: np.ndarray, scale: float
) -> None:
    """
    Scale y-direction permeability for cells in the fracture zone.

    This scales the permeability of all cells marked in the mask,
    which affects transmissibility for flow in the y-direction.
    """
    nx, ny, nz = perm_y.shape
    for i in numba.prange(nx):  # type: ignore[misc]
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    perm_y[i, j, k] *= scale


@numba.njit(parallel=True)
def _scale_permeability_z_boundary(
    perm_z: np.ndarray, mask: np.ndarray, scale: float
) -> None:
    """
    Scale z-direction permeability for cells in the fracture zone.

    This scales the permeability of all cells marked in the mask,
    which affects transmissibility for flow in the z-direction.
    """
    nx, ny, nz = perm_z.shape
    for i in numba.prange(nx):  # type: ignore[misc]
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    perm_z[i, j, k] *= scale


FloatDefault = typing.Union[float, np.floating[typing.Any]]
StatDefault = typing.Literal[
    "mean",
    "median",
    "min",
    "max",
    "zero",
    "nearest",
    "linear",
]
HookDefault = typing.Callable[[np.typing.NDArray], FloatDefault]
DefaultDef = typing.Union[FloatDefault, StatDefault, HookDefault]


def _compute_nearest_default(
    grid: np.typing.NDArray,
    valid_data: np.typing.NDArray,
    property_type: typing.Optional[str] = None,
    dtype: typing.Optional[np.typing.DTypeLike] = None,
) -> np.floating:
    """
    Compute nearest-neighbor default using boundary values.

    Extracts values from grid boundaries and uses median for robustness.
    This provides better spatial continuity than global statistics.

    :param grid: Original grid array
    :param valid_data: Pre-filtered valid data (fallback)
    :param property_type: Optional property type for filtering
    :return: Median of boundary values
    """
    # Extract boundary values
    if grid.ndim == 3:
        # Get boundary slices (first and last in each dimension)
        boundary_values = np.concatenate(
            [
                grid[0, :, :].flatten(),
                grid[-1, :, :].flatten(),
                grid[:, 0, :].flatten(),
                grid[:, -1, :].flatten(),
                grid[:, :, 0].flatten(),
                grid[:, :, -1].flatten(),
            ]
        )
    else:
        # Fallback for non-3D grids
        boundary_values = valid_data

    # Filter boundary values
    boundary_valid = boundary_values[np.isfinite(boundary_values)]
    if property_type == "saturation":
        boundary_valid = boundary_valid[(boundary_valid >= 0) & (boundary_valid <= 1)]
    elif property_type in ("permeability", "porosity", "thickness"):
        boundary_valid = boundary_valid[boundary_valid > 0]

    # Use median of boundaries (robust to outliers)
    dtype = dtype if dtype is not None else get_dtype()
    if len(boundary_valid) > 0:
        return dtype(np.median(boundary_valid))  # type: ignore
    return dtype(np.mean(valid_data))  # type: ignore


def _compute_linear_default(
    grid: np.typing.NDArray,
    valid_data: np.typing.NDArray,
    property_type: typing.Optional[str] = None,
    dtype: typing.Optional[np.typing.DTypeLike] = None,
) -> np.floating:
    """
    Compute linear interpolation default using boundary values.

    Extracts values from grid boundaries and uses mean for smooth averaging.
    This provides smooth transitions appropriate for diffusion-like properties.

    :param grid: Original grid array
    :param valid_data: Pre-filtered valid data (fallback)
    :param property_type: Optional property type for filtering
    :return: Mean of boundary values
    """
    # Extract boundary values
    if grid.ndim == 3:
        boundary_values = np.concatenate(
            [
                grid[0, :, :].flatten(),
                grid[-1, :, :].flatten(),
                grid[:, 0, :].flatten(),
                grid[:, -1, :].flatten(),
                grid[:, :, 0].flatten(),
                grid[:, :, -1].flatten(),
            ]
        )
    else:
        boundary_values = valid_data

    # Filter boundary values
    boundary_valid = boundary_values[np.isfinite(boundary_values)]
    if property_type == "saturation":
        boundary_valid = boundary_valid[(boundary_valid >= 0) & (boundary_valid <= 1)]
    elif property_type in ("permeability", "porosity", "thickness"):
        boundary_valid = boundary_valid[boundary_valid > 0]

    # Use mean of boundaries (smooth average)
    dtype = dtype if dtype is not None else get_dtype()
    if len(boundary_valid) > 0:
        return dtype(np.mean(boundary_valid))  # type: ignore
    return dtype(np.mean(valid_data))  # type: ignore


@attrs.define(slots=True, frozen=True)
class FractureDefaults:
    """
    Default value provider for reservoir properties.

    Supports multiple modes:
    1. Constant values (float)
    2. Callable that generates values based on grid statistics
    3. Statistical keywords: 'mean', 'median', 'min', 'max', 'zero'
    4. Interpolation methods: 'nearest', 'linear' (for spatial continuity)

    DEFAULT PHILOSOPHY:
    For most properties, we use statistical defaults from the existing grid
    rather than arbitrary constants. This ensures the filled regions are
    consistent with the rest of the reservoir.

    - Pressure/Temperature: Use 'median' to avoid extremes
    - Saturations: Use 'mean' to maintain reservoir averages
    - Permeability/Porosity: Use 'mean' for continuity
    - Can use 'nearest' or 'linear' for smooth spatial transitions
    """

    thickness: DefaultDef = "mean"
    porosity: DefaultDef = "mean"
    permeability: DefaultDef = "mean"
    net_to_gross: DefaultDef = "mean"
    water_saturation: DefaultDef = "mean"
    oil_saturation: DefaultDef = "mean"
    gas_saturation: DefaultDef = "mean"
    irreducible_water_saturation: DefaultDef = "mean"
    residual_oil_saturation_water: DefaultDef = "mean"
    residual_oil_saturation_gas: DefaultDef = "mean"
    residual_gas_saturation: DefaultDef = "mean"
    pressure: DefaultDef = "median"  # Use median for pressure
    temperature: DefaultDef = "median"
    oil_viscosity: DefaultDef = "mean"
    water_viscosity: DefaultDef = "mean"
    gas_viscosity: DefaultDef = "mean"
    oil_density: DefaultDef = "mean"
    water_density: DefaultDef = "mean"
    gas_density: DefaultDef = "mean"
    oil_fvf: DefaultDef = "mean"
    water_fvf: DefaultDef = "mean"
    gas_fvf: DefaultDef = "mean"
    compressibility: DefaultDef = "mean"
    bubble_point_pressure: DefaultDef = "mean"
    # Generic fallback for unspecified properties
    generic: DefaultDef = "mean"

    def get_value(
        self,
        property_name: str,
        grid: np.typing.NDArray,
        property_type: typing.Optional[str] = None,
        dtype: typing.Optional[np.typing.DTypeLike] = None,
    ) -> np.floating:
        """
        Compute value for a property based on configuration.

        :param property_name: Name of the property (e.g., 'porosity')
        :param grid: Original grid array to analyze
        :param property_type: typing.Optional type hint ('saturation', 'permeability', etc.)
        :return: Computed default value
        """
        # Try to find specific configuration
        spec = getattr(self, property_name, None)
        if spec is None:
            spec = self.generic

        dtype = dtype if dtype is not None else get_dtype()
        # Handle different specification types
        if isinstance(spec, (int, float)):
            return dtype(spec)  # type: ignore

        if callable(spec):
            return dtype(spec(grid))  # type: ignore

        if isinstance(spec, str):
            spec = typing.cast(StatDefault, spec)
            return self._compute_statistical_default(
                method=spec,
                grid=grid,
                property_type=property_type,
                dtype=dtype,
            )

        # Fallback to mean if all else fails
        return self._compute_statistical_default(
            method="mean",
            grid=grid,
            property_type=property_type,
            dtype=dtype,
        )

    def _compute_statistical_default(
        self,
        method: StatDefault,
        grid: np.typing.NDArray,
        property_type: typing.Optional[str] = None,
        dtype: typing.Optional[np.typing.DTypeLike] = None,
    ) -> np.floating:
        """
        Compute statistical defaults from grid data.

        Supports statistical methods (mean, median, min, max, zero) and
        interpolation methods (nearest, linear) for spatial continuity.

        For 'nearest' and 'linear', we extract boundary values and compute
        an average, which provides better spatial continuity than pure statistics.
        """
        # Filter out invalid values
        valid_data = grid[np.isfinite(grid)]

        # Apply property-specific constraints
        if property_type == "saturation":
            valid_data = valid_data[(valid_data >= 0) & (valid_data <= 1)]
        elif property_type in ("permeability", "porosity", "thickness"):
            valid_data = valid_data[valid_data > 0]

        if len(valid_data) == 0:
            raise ValidationError(f"No valid data available for {property_type}")

        dtype = dtype if dtype is not None else get_dtype()
        # Compute based on method
        if method == "mean":
            return dtype(np.mean(valid_data))  # type: ignore
        elif method == "median":
            return dtype(np.median(valid_data))  # type: ignore
        elif method == "min":
            return dtype(np.min(valid_data))  # type: ignore
        elif method == "max":
            return dtype(np.max(valid_data))  # type: ignore
        elif method == "zero":
            return dtype(0.0)  # type: ignore
        elif method == "nearest":
            return _compute_nearest_default(
                grid, valid_data, property_type, dtype=dtype
            )
        elif method == "linear":
            return _compute_linear_default(grid, valid_data, property_type, dtype=dtype)

        logger.warning(f"Unknown method '{method}', using mean")
        return dtype(np.mean(valid_data))  # type: ignore


@attrs.frozen(slots=True)
class FractureGeometry:
    """
    Fracture geometry specification.

    Fracture Orientation:
    - "x": Fracture plane perpendicular to x-axis (strikes in y-direction)
    - "y": Fracture plane perpendicular to y-axis (strikes in x-direction)
    - "z": Fracture plane perpendicular to z-axis (horizontal fracture/bedding slip)

    Coordinate Specification:
    The fracture is defined by specifying ranges in each dimension:

    For x-oriented fracture:
        - x_range: Location of fracture plane (can be single cell or damage zone)
        - y_range: Optional lateral extent (None = full extent)
        - z_range: Optional vertical extent (None = full extent)

    For y-oriented fracture:
        - y_range: Location of fracture plane (can be single cell or damage zone)
        - x_range: Optional lateral extent (None = full extent)
        - z_range: Optional vertical extent (None = full extent)

    For z-oriented fracture (horizontal):
        - z_range: Location of fracture plane (can be single layer or zone)
        - x_range: Optional lateral extent (None = full extent)
        - y_range: Optional lateral extent (None = full extent)

    Examples:

    1. Simple vertical fault at x=25:
    ```
       FractureGeometry(orientation="x", x_range=(25, 25))
    ```

    2. Wide damage zone from x=20 to x=30, only in upper reservoir:
    ```
       FractureGeometry(orientation="x", x_range=(20, 30), z_range=(0, 15))
    ```

    3. Horizontal bedding plane slip at z=10:
    ```
       FractureGeometry(orientation="z", z_range=(10, 10))
    ```

    4. Y-oriented fault with limited lateral extent:
    ```
       FractureGeometry(orientation="y", y_range=(15, 15), x_range=(10, 40), z_range=(5, 25))
    ```

    """

    orientation: typing.Literal["x", "y", "z"]
    """
    Fracture plane orientation:
    - "x": Perpendicular to x-axis (fracture plane in y-z plane)
    - "y": Perpendicular to y-axis (fracture plane in x-z plane)
    - "z": Perpendicular to z-axis (horizontal fracture plane in x-y plane)
    """

    x_range: typing.Optional[typing.Tuple[int, int]] = None
    """
    Range of x-indices affected by fracture (inclusive).
    - For x-oriented fractures: Location of fracture plane
    - For y/z-oriented fractures: Optional lateral extent
    - None means full x-extent of grid
    """

    y_range: typing.Optional[typing.Tuple[int, int]] = None
    """
    Range of y-indices affected by fracture (inclusive).
    - For y-oriented fractures: Location of fracture plane
    - For x/z-oriented fractures: Optional lateral extent
    - None means full y-extent of grid
    """

    z_range: typing.Optional[typing.Tuple[int, int]] = None
    """
    Range of z-indices affected by fracture (inclusive).
    - For z-oriented fractures: Location of fracture plane
    - For x/y-oriented fractures: Optional vertical extent
    - None means full z-extent of grid
    """

    displacement_range: typing.Optional[typing.Tuple[int, int]] = None
    """
    Optional explicit range of cells to displace (in fracture orientation direction).
    
    If specified, Only cells within this range are displaced, independent of
    the fracture plane location. This provides fine-grained control over which
    cells experience throw.
    
    Bahaviour:
    - None (default): Displace all cells on the positive side of the fracture plane
    - (min, max): Only cells with indices in [min, max] are displaced
    
    Examples:
    
    For x-oriented fracture at x=20:
        - displacement_range=(25, 50): Only cells with x ∈ [25, 50] are displaced
        - displacement_range=(10, 19): Only cells with x ∈ [10, 19] are displaced (reverse side)
        - Useful for: "Displace only a specific block, not everything beyond fault"
    
    For y-oriented fracture:
        - displacement_range=(10, 25): Only cells with y ∈ [10, 25] are displaced
    
    For z-oriented fracture:
        - displacement_range=(5, 15): Only cells with z ∈ [5, 15] are displaced
    
    Notable Cases:
    - If displacement_range doesn't overlap with fracture plane range, displacement
      still occurs (allows modelling of detached/blind segments)
    - Can be used to displace fracture zone + adjacent cells:
      displacement_range = (x_range[0], x_range[1] + 10)
    """

    geometric_throw: int = 0
    """
    Vertical displacement (throw) in number of cells.
    
    - Positive values: Displaced block moves DOWN (normal fault)
    - Negative values: Displaced block moves UP (reverse fault)
    - Zero (default): No vertical displacement (sealing fault only)
    
    The cells to be displaced are determined by displacement_range.
    If displacement_range is None, all cells on the positive side of the
    fracture plane are displaced.
    """

    slope: float = 0.0
    """
    Slope of inclined fracture plane (for non-vertical/non-horizontal fractures).
    
    For x-oriented fractures: z = intercept + slope * y
    For y-oriented fractures: z = intercept + slope * x
    For z-oriented fractures: y = intercept + slope * x (or x = intercept + slope * y)
    
    A slope of 0.0 creates a planar fracture (vertical for x/y, horizontal for z).
    """

    intercept: float = 0.0
    """
    Intercept of the fracture plane equation.
    Interpretation depends on orientation and slope.
    """

    def __attrs_post_init__(self) -> None:
        """Validate geometry configuration."""
        # Check that primary range is specified
        if self.orientation == "x" and self.x_range is None:
            raise ValidationError(
                "For x-oriented fractures, `x_range` must be specified"
            )
        if self.orientation == "y" and self.y_range is None:
            raise ValidationError(
                "For y-oriented fractures, `y_range` must be specified"
            )
        if self.orientation == "z" and self.z_range is None:
            raise ValidationError(
                "For z-oriented fractures, `z_range` must be specified"
            )

        # Validate ranges are properly ordered
        for range_name, range_val in [
            ("x_range", self.x_range),
            ("y_range", self.y_range),
            ("z_range", self.z_range),
            ("displacement_range", self.displacement_range),
        ]:
            if range_val is not None:
                if range_val[0] > range_val[1]:
                    raise ValidationError(
                        f"{range_name} min ({range_val[0]}) > max ({range_val[1]})"
                    )
                if range_val[0] < 0:
                    raise ValidationError(
                        f"{range_name} min ({range_val[0]}) must be >= 0"
                    )

    def get_fracture_plane_range(self) -> typing.Tuple[int, int]:
        """Get the primary fracture plane range based on orientation."""
        if self.orientation == "x":
            assert self.x_range is not None
            return self.x_range
        elif self.orientation == "y":
            assert self.y_range is not None
            return self.y_range
        else:  # z
            assert self.z_range is not None
            return self.z_range


@attrs.frozen(slots=True)
class Fracture:
    """
    Configuration for applying fractures to reservoir models.

    Defines the geometric and hydraulic properties of a fracture,
    including its orientation, hydraulic properties, and modeling approach.

    Note: A geological fault is a specific type of fracture.

    Usage Examples:

    ```python
    # Example 1: Sealing fault using permeability multiplier
    geometry = FractureGeometry(
        orientation="x",
        x_range=(25, 25),
        z_range=(0, 15),
        geometric_throw=3
    )
    fracture = Fracture(
        id="fault_1",
        geometry=geometry,
        permeability_multiplier=1e-4  # Scale existing permeabilities
    )

    # Example 2: Damage zone with absolute permeability values
    damage_geometry = FractureGeometry(
        orientation="y",
        y_range=(10, 15)  # Multi-cell damage zone
    )
    damage_fracture = Fracture(
        id="damage_zone",
        geometry=damage_geometry,
        permeability=0.01,  # Set absolute permeability (mD)
        porosity=0.05
    )
    ```
    """

    id: str
    """Unique identifier for the fracture."""

    geometry: FractureGeometry
    """
    Fracture geometry specification using unified FractureGeometry class.
    
    Defines the spatial location, orientation, extent, and displacement
    of the fracture using a consistent coordinate system.
    """

    permeability_multiplier: typing.Optional[float] = None
    """
    Multiplier for permeabilities across the fracture.

    - Values < 1.0 create sealing barriers (typical: 1e-3 to 1e-6)
    - Values > 1.0 create conductive zones (for enhanced fractures)
    - Must be > `MIN_TRANSMISSIBILITY_FACTOR` for numerical stability
    - If None and permeability is also None, defaults to 1e-3 for sealing behavior
    
    This scales the permeability values at the fracture interface, directly
    affecting transmissibility (flow capacity) across the fracture.
    
    Mutually exclusive with `permeability`. Use one or the other:
    - `permeability_multiplier`: Scale existing permeabilities
    - `permeability`: Set absolute permeability value
    """

    permeability: typing.Optional[float] = None
    """
    Permeability value for fracture zone cells (mD).
    If None, fracture zone properties are not modified.

    Mutually exclusive with `permeability_multiplier`. Use one or the other:
    - `permeability`: Set absolute permeability value
    - `permeability_multiplier`: Scale existing permeabilities
    """

    porosity: typing.Optional[float] = None
    """
    Porosity value for fracture zone cells (fraction).
    If None, fracture zone properties are not modified.
    """

    conductive: bool = False
    """
    If True, the fracture acts as a high-permeability conduit.
    This automatically sets appropriate `permeability_multiplier` if not manually specified.
    """

    mask: typing.Optional[np.ndarray] = None
    """
    Optional 3D boolean mask defining fracture geometry.

    If provided, overrides geometric fracture plane calculation.
    Must match the reservoir model grid dimensions.
    """

    preserve_grid_data: bool = False
    """
    If True, expand grid dimensions to preserve all displaced data.
    If False, use traditional displacement with data loss or defaults.
    """

    defaults: FractureDefaults = attrs.field(factory=lambda: FractureDefaults())
    """
    Custom default values for properties in displaced/expanded regions.
    Uses `FractureDefaults()` with sensible defaults.
    """

    def __attrs_post_init__(self) -> None:
        """Validate fracture configuration parameters."""
        # Mutual exclusivity check
        if self.permeability_multiplier is not None and self.permeability is not None:
            raise ValidationError(
                f"Fracture {self.id!r}: `permeability_multiplier` and `permeability` are mutually exclusive. "
                "Use `permeability_multiplier` to scale existing permeabilities, or `permeability` to set absolute values."
            )

        # Validate permeability_multiplier if set
        if self.permeability_multiplier is not None:
            if self.permeability_multiplier < c.MIN_TRANSMISSIBILITY_FACTOR:
                object.__setattr__(
                    self, "permeability_multiplier", c.MIN_TRANSMISSIBILITY_FACTOR
                )
                logger.warning(
                    f"Fracture {self.id!r}: `permeability_multiplier` clamped to {c.MIN_TRANSMISSIBILITY_FACTOR}"
                )

            if self.conductive and self.permeability_multiplier < 1.0:
                raise ValidationError(
                    f"Fracture {self.id!r}: conductive fractures must have `permeability_multiplier` >= 1.0"
                )

        if self.permeability is not None and self.permeability < 0:
            raise ValidationError(
                f"Fracture {self.id!r}: `permeability` must be non-negative"
            )

        if self.porosity is not None and not (0 <= self.porosity <= 1):
            raise ValidationError(
                f"Fracture {self.id!r}: `porosity` must be between 0 and 1"
            )


def apply_fracture(
    model: ReservoirModel[ThreeDimensions], fracture: Fracture
) -> ReservoirModel[ThreeDimensions]:
    """
    Apply a single fracture to a reservoir model.

    This function modifies transmissibilities, rock properties, and geometry
    based on the fracture configuration. The input model is not modified;
    a new model instance is returned.

    :param model: Input reservoir model to modify
    :param fracture: Fracture configuration defining geometry and properties
    :return: The reservoir model with the fracture applied
    """
    logger.debug(f"Applying fracture '{fracture.id}' to reservoir model")

    errors = validate_fracture(fracture=fracture, grid_shape=model.grid_shape)
    if errors:
        msg = ""
        for error in errors:
            msg += f"\n - {error}"
        raise ValidationError(f"Fracture {fracture.id} configuration is invalid: {msg}")

    grid_shape = model.grid_shape

    if len(grid_shape) != 3:
        raise ValidationError("Fracture application requires 3D reservoir models")

    # Generate or validate fracture mask
    if fracture.mask is not None:
        if fracture.mask.shape != grid_shape:
            raise ValidationError(
                f"Fracture {fracture.id}: mask shape {fracture.mask.shape} != grid shape {grid_shape}"
            )
        fracture_mask = fracture.mask.copy()
    else:
        fracture_mask = make_fracture_mask(
            grid_shape=grid_shape,
            fracture=fracture,
        )  # type: ignore[arg-type]

    # Apply fracture effects in proper order
    # Modify fracture zone properties (before geometric displacement)
    if fracture.permeability is not None or fracture.porosity is not None:
        model = _apply_fracture_zone_properties(
            model=model, fracture_mask=fracture_mask, fracture=fracture
        )

    # Scale permeabilities across fracture (before geometric displacement)
    if fracture.permeability_multiplier is not None:
        model = _scale_permeability(
            model=model, fracture_mask=fracture_mask, fracture=fracture
        )

    # Apply geometric displacement (throw). This may change grid dimensions
    if fracture.geometry.geometric_throw != 0:
        model = _apply_geometric_throw(model=model, fracture=fracture)

    logger.debug(f"Successfully applied fracture '{fracture.id}'")
    return model


def apply_fractures(
    model: ReservoirModel[ThreeDimensions], *fractures: Fracture
) -> ReservoirModel[ThreeDimensions]:
    """
    Apply multiple fractures to a reservoir model.

    Fractures are applied sequentially in the order provided.

    :param model: Input reservoir model
    :param fractures: Sequence of fracture configurations
    :return: The reservoir model with all fractures applied
    """
    logger.debug(f"Applying {len(fractures)} fractures to reservoir model")

    faulted_model = model
    for fracture in fractures:
        faulted_model = apply_fracture(model=faulted_model, fracture=fracture)

    logger.debug(f"Successfully applied all {len(fractures)} fractures")
    return faulted_model


def make_fracture_mask(
    grid_shape: typing.Tuple[int, int, int], fracture: Fracture
) -> np.ndarray:
    """
    Generate a 3D boolean mask defining the fracture geometry.

    For inclined fractures, the mask follows the equation:
    z = z0 + slope * (coord - coord0)

    Where:
        z = vertical index
        coord = x or y index depending on orientation
        coord0 = fracture plane location
        z0 = intercept

    :param grid_shape: Shape of the reservoir grid (nx, ny, nz)
    :param fracture: Fracture configuration
    :return: 3D boolean array marking fracture cells
    """
    nx, ny, nz = grid_shape
    mask = np.zeros((nx, ny, nz), dtype=np.bool_)

    geom = fracture.geometry

    # Determine the cell range to apply
    cell_min, cell_max = geom.get_fracture_plane_range()

    # Determine vertical extent (z_range)
    if geom.z_range is not None:
        z_min, z_max = geom.z_range
        z_min = max(0, min(z_min, nz - 1))  # Clamp to valid range
        z_max = max(0, min(z_max, nz - 1))
    else:
        z_min, z_max = 0, nz - 1  # Full vertical extent

    if geom.orientation == "x":
        # For x-oriented fractures, determine y lateral extent
        if geom.y_range is not None:
            y_min, y_max = geom.y_range
            y_min = max(0, min(y_min, ny - 1))
            y_max = max(0, min(y_max, ny - 1))
        else:
            y_min, y_max = 0, ny - 1

        _mask_orientation_x(
            mask=mask,
            cell_min=cell_min,
            cell_max=cell_max,
            slope=float(geom.slope),
            intercept=float(geom.intercept),
            z_min=z_min,
            z_max=z_max,
            y_min=y_min,
            y_max=y_max,
        )

    elif geom.orientation == "y":
        # For y-oriented fractures, determine x lateral extent
        if geom.x_range is not None:
            x_min, x_max = geom.x_range
            x_min = max(0, min(x_min, nx - 1))
            x_max = max(0, min(x_max, nx - 1))
        else:
            x_min, x_max = 0, nx - 1

        _mask_orientation_y(
            mask=mask,
            cell_min=cell_min,
            cell_max=cell_max,
            slope=float(geom.slope),
            intercept=float(geom.intercept),
            z_min=z_min,
            z_max=z_max,
            x_min=x_min,
            x_max=x_max,
        )

    elif geom.orientation == "z":
        # Horizontal fracture - fracture plane in x-y plane
        # Determine x and y extents
        if geom.x_range is not None:
            x_min, x_max = geom.x_range
            x_min = max(0, min(x_min, nx - 1))
            x_max = max(0, min(x_max, nx - 1))
        else:
            x_min, x_max = 0, nx - 1

        if geom.y_range is not None:
            y_min, y_max = geom.y_range
            y_min = max(0, min(y_min, ny - 1))
            y_max = max(0, min(y_max, ny - 1))
        else:
            y_min, y_max = 0, ny - 1

        _mask_orientation_z(
            mask=mask,
            cell_min=cell_min,
            cell_max=cell_max,
            slope=float(geom.slope),
            intercept=float(geom.intercept),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
    return mask


def _scale_permeability(
    model: ReservoirModel[ThreeDimensions],
    fracture_mask: np.ndarray,
    fracture: Fracture,
) -> ReservoirModel[ThreeDimensions]:
    """
    Scale permeabilities across fracture boundaries.

    Identifies cells at the fracture and scales their permeability values
    by the permeability_multiplier to reduce/increase flow capacity across
    the fracture interface.

    :param model: Reservoir model to modify
    :param fracture_mask: 3D boolean array marking fracture cells
    :param fracture: Fracture configuration
    :return: Modified reservoir model
    """
    logger.debug(f"Scaling transmissibilities for fracture '{fracture.id}'")

    geom = fracture.geometry
    if fracture.permeability_multiplier is None:
        return model

    # Scale connections based on fracture orientation
    if geom.orientation == "x":
        _scale_permeability_x_boundary(
            perm_x=model.rock_properties.absolute_permeability.x,
            mask=fracture_mask,
            scale=float(fracture.permeability_multiplier),
        )
    elif geom.orientation == "y":
        _scale_permeability_y_boundary(
            perm_y=model.rock_properties.absolute_permeability.y,
            mask=fracture_mask,
            scale=float(fracture.permeability_multiplier),
        )
    elif geom.orientation == "z":
        _scale_permeability_z_boundary(
            perm_z=model.rock_properties.absolute_permeability.z,
            mask=fracture_mask,
            scale=float(fracture.permeability_multiplier),
        )

    # Always scale z-direction for inclined x/y fractures
    if geom.orientation in ("x", "y"):
        _scale_permeability_z_boundary(
            perm_z=model.rock_properties.absolute_permeability.z,
            mask=fracture_mask,
            scale=float(fracture.permeability_multiplier),
        )
    return model


def _apply_fracture_zone_properties(
    model: ReservoirModel[ThreeDimensions],
    fracture_mask: np.ndarray,
    fracture: Fracture,
) -> ReservoirModel[ThreeDimensions]:
    """
    Apply fracture zone rock properties to cells within the fracture.

    :param model: Reservoir model to modify
    :param fracture_mask: 3D boolean array marking fracture cells
    :param fracture: Fracture configuration
    :return: Modified reservoir model
    """
    logger.debug(f"Applying fracture zone properties for fracture '{fracture.id}'")

    if fracture.permeability is not None:
        model.rock_properties.absolute_permeability.x[fracture_mask] = (
            fracture.permeability
        )
        model.rock_properties.absolute_permeability.y[fracture_mask] = (
            fracture.permeability
        )
        model.rock_properties.absolute_permeability.z[fracture_mask] = (
            fracture.permeability
        )

    if fracture.porosity is not None:
        model.rock_properties.porosity_grid[fracture_mask] = fracture.porosity
    return model


def _apply_geometric_throw(
    model: ReservoirModel[ThreeDimensions],
    fracture: Fracture,
) -> ReservoirModel[ThreeDimensions]:
    """
    Apply geometric displacement (throw) using improved block-based algorithm.

    Key improvements:
    - Block-based displacement (no cell-by-cell jumbling)
    - Smart property-aware default filling
    - Clean separation between upthrown and downthrown blocks

    :param model: Reservoir model to modify
    :param fracture_mask: 3D boolean array marking fracture cells
    :param fracture: Fracture configuration
    :return: Modified reservoir model with updated grid dimensions
    """
    throw = fracture.geometry.geometric_throw
    logger.debug(f"Applying geometric throw ({throw} cells)")

    if throw == 0:
        return model

    displacement_mask = make_displacement_mask(
        grid_shape=model.grid_shape, fracture=fracture
    )
    defaults = fracture.defaults

    # Create displacement context for all grids
    ctx = DisplacementContext(
        original_shape=model.grid_shape,
        throw=throw,
        displacement_mask=displacement_mask,
        preserve_data=fracture.preserve_grid_data,
        defaults=defaults,
    )

    # Apply displacement to all property grids
    logger.debug("Displacing all property grids with block-based algorithm")

    # Geometric grids
    new_thickness = ctx.displace_grid(
        grid=model.thickness_grid, property_name="thickness", property_type="thickness"
    )

    # Rock properties
    rock = model.rock_properties
    new_rock = attrs.evolve(
        rock,
        porosity_grid=ctx.displace_grid(
            grid=rock.porosity_grid, property_name="porosity", property_type="porosity"
        ),
        net_to_gross_ratio_grid=ctx.displace_grid(
            grid=rock.net_to_gross_ratio_grid, property_name="net_to_gross"
        ),
        irreducible_water_saturation_grid=ctx.displace_grid(
            grid=rock.irreducible_water_saturation_grid,
            property_name="irreducible_water_saturation",
            property_type="saturation",
        ),
        residual_oil_saturation_water_grid=ctx.displace_grid(
            grid=rock.residual_oil_saturation_water_grid,
            property_name="residual_oil_saturation_water",
            property_type="saturation",
        ),
        residual_oil_saturation_gas_grid=ctx.displace_grid(
            grid=rock.residual_oil_saturation_gas_grid,
            property_name="residual_oil_saturation_gas",
            property_type="saturation",
        ),
        residual_gas_saturation_grid=ctx.displace_grid(
            grid=rock.residual_gas_saturation_grid,
            property_name="residual_gas_saturation",
            property_type="saturation",
        ),
        absolute_permeability=attrs.evolve(
            rock.absolute_permeability,
            x=ctx.displace_grid(
                grid=rock.absolute_permeability.x,
                property_name="permeability",
                property_type="permeability",
            ),
            y=ctx.displace_grid(
                grid=rock.absolute_permeability.y,
                property_name="permeability",
                property_type="permeability",
            ),
            z=ctx.displace_grid(
                grid=rock.absolute_permeability.z,
                property_name="permeability",
                property_type="permeability",
            ),
        ),
    )

    # Fluid properties
    fluid = model.fluid_properties
    new_fluid = attrs.evolve(
        fluid,
        pressure_grid=ctx.displace_grid(
            grid=fluid.pressure_grid, property_name="pressure", property_type="pressure"
        ),
        temperature_grid=ctx.displace_grid(
            grid=fluid.temperature_grid,
            property_name="temperature",
            property_type="temperature",
        ),
        oil_bubble_point_pressure_grid=ctx.displace_grid(
            grid=fluid.oil_bubble_point_pressure_grid,
            property_name="bubble_point_pressure",
            property_type="pressure",
        ),
        oil_saturation_grid=ctx.displace_grid(
            grid=fluid.oil_saturation_grid,
            property_name="oil_saturation",
            property_type="saturation",
        ),
        water_saturation_grid=ctx.displace_grid(
            grid=fluid.water_saturation_grid,
            property_name="water_saturation",
            property_type="saturation",
        ),
        gas_saturation_grid=ctx.displace_grid(
            grid=fluid.gas_saturation_grid,
            property_name="gas_saturation",
            property_type="saturation",
        ),
        oil_viscosity_grid=ctx.displace_grid(
            grid=fluid.oil_viscosity_grid,
            property_name="oil_viscosity",
            property_type="viscosity",
        ),
        water_viscosity_grid=ctx.displace_grid(
            grid=fluid.water_viscosity_grid,
            property_name="water_viscosity",
            property_type="viscosity",
        ),
        gas_viscosity_grid=ctx.displace_grid(
            grid=fluid.gas_viscosity_grid,
            property_name="gas_viscosity",
            property_type="viscosity",
        ),
        oil_density_grid=ctx.displace_grid(
            grid=fluid.oil_density_grid,
            property_name="oil_density",
            property_type="density",
        ),
        water_density_grid=ctx.displace_grid(
            grid=fluid.water_density_grid,
            property_name="water_density",
            property_type="density",
        ),
        gas_density_grid=ctx.displace_grid(
            grid=fluid.gas_density_grid,
            property_name="gas_density",
            property_type="density",
        ),
        oil_compressibility_grid=ctx.displace_grid(
            grid=fluid.oil_compressibility_grid, property_name="compressibility"
        ),
        water_compressibility_grid=ctx.displace_grid(
            grid=fluid.water_compressibility_grid, property_name="compressibility"
        ),
        gas_compressibility_grid=ctx.displace_grid(
            grid=fluid.gas_compressibility_grid, property_name="compressibility"
        ),
        oil_formation_volume_factor_grid=ctx.displace_grid(
            grid=fluid.oil_formation_volume_factor_grid,
            property_name="oil_fvf",
            property_type="fvf",
        ),
        water_formation_volume_factor_grid=ctx.displace_grid(
            grid=fluid.water_formation_volume_factor_grid,
            property_name="water_fvf",
            property_type="fvf",
        ),
        gas_formation_volume_factor_grid=ctx.displace_grid(
            grid=fluid.gas_formation_volume_factor_grid,
            property_name="gas_fvf",
            property_type="fvf",
        ),
        # Add remaining fluid properties as needed...
        oil_specific_gravity_grid=ctx.displace_grid(
            grid=fluid.oil_specific_gravity_grid, property_name="generic"
        ),
        oil_api_gravity_grid=ctx.displace_grid(
            grid=fluid.oil_api_gravity_grid, property_name="generic"
        ),
        water_bubble_point_pressure_grid=ctx.displace_grid(
            grid=fluid.water_bubble_point_pressure_grid,
            property_name="bubble_point_pressure",
            property_type="pressure",
        ),
        gas_gravity_grid=ctx.displace_grid(
            grid=fluid.gas_gravity_grid, property_name="generic"
        ),
        gas_molecular_weight_grid=ctx.displace_grid(
            grid=fluid.gas_molecular_weight_grid, property_name="generic"
        ),
        solution_gas_to_oil_ratio_grid=ctx.displace_grid(
            grid=fluid.solution_gas_to_oil_ratio_grid, property_name="generic"
        ),
        gas_solubility_in_water_grid=ctx.displace_grid(
            grid=fluid.gas_solubility_in_water_grid, property_name="generic"
        ),
        water_salinity_grid=ctx.displace_grid(
            grid=fluid.water_salinity_grid, property_name="generic"
        ),
        oil_effective_viscosity_grid=ctx.displace_grid(
            grid=fluid.oil_effective_viscosity_grid,
            property_name="oil_viscosity",
            property_type="viscosity",
        ),
        solvent_concentration_grid=ctx.displace_grid(
            grid=fluid.solvent_concentration_grid, property_name="generic"
        ),
    )

    # Update model with new grids and shape
    new_model = model.evolve(
        grid_shape=ctx.new_shape,
        thickness_grid=new_thickness,
        rock_properties=new_rock,
        fluid_properties=new_fluid,
    )
    logger.debug(f"Grid shape: {model.grid_shape} → {ctx.new_shape}")
    return new_model


@attrs.frozen(slots=True)
class DisplacementContext:
    """
    Context object that handles block-based displacement for all grids.

    This encapsulates the displacement logic and default value computation,
    making the code cleaner and more maintainable.
    """

    original_shape: typing.Tuple[int, int, int]
    throw: int
    displacement_mask: np.typing.NDArray
    preserve_data: bool
    defaults: FractureDefaults

    @property
    def new_shape(self) -> typing.Tuple[int, int, int]:
        """Compute new grid shape after displacement."""
        nx, ny, nz = self.original_shape
        if self.preserve_data:
            return (nx, ny, nz + abs(self.throw))
        return self.original_shape

    def displace_grid(
        self,
        grid: np.ndarray,
        property_name: str,
        property_type: typing.Optional[str] = None,
    ) -> np.ndarray:
        """
        Displace a single grid using block-based algorithm.

        :param grid: Original grid to displace
        :param property_name: Name of property (for default value lookup)
        :param property_type: Type hint for default computation
        :return: Displaced grid (possibly expanded)
        """
        # Handle zero throw case - just return copy with correct shape
        if self.throw == 0:
            if self.preserve_data:
                # Even with preserve_data, no expansion needed for zero throw
                return grid.copy()
            return grid.copy()

        if self.preserve_data:
            return self._displace_with_expansion(
                grid=grid, property_name=property_name, property_type=property_type
            )
        return self._displace_without_expansion(
            grid=grid, property_name=property_name, property_type=property_type
        )

    def _displace_with_expansion(
        self,
        grid: np.typing.NDArray,
        property_name: str,
        property_type: typing.Optional[str],
    ) -> np.typing.NDArray:
        """
        Block-based displacement with grid expansion (preserves all data).

        Algorithm:
        1. Create expanded grid with smart defaults
        2. Identify upthrown and downthrown blocks
        3. Copy upthrown block to new position (stays in place)
        4. Copy downthrown block to displaced position
        5. Fill gaps with property-aware defaults
        """
        nx, ny, nz = self.original_shape
        abs_throw = abs(self.throw)
        new_nz = nz + abs_throw

        # Get smart default value for this property, preserving grid dtype
        default_value = self.defaults.get_value(
            property_name=property_name,
            grid=grid,
            property_type=property_type,
            dtype=grid.dtype,
        )

        # Create expanded grid filled with defaults
        new_grid = np.full((nx, ny, new_nz), fill_value=default_value, dtype=grid.dtype)

        # Identify upthrown and downthrown blocks
        upthrown_mask = np.invert(self.displacement_mask)  # Not displaced
        downthrown_mask = self.displacement_mask  # Displaced

        if self.throw > 0:
            # Normal fault: downthrown block moves DOWN
            # Layout: [upthrown at top | gap filled with defaults | downthrown at bottom]

            # Copy upthrown block to top of new grid (unchanged position)
            for k in range(nz):
                new_grid[:, :, k][upthrown_mask[:, :, k]] = grid[:, :, k][
                    upthrown_mask[:, :, k]
                ]

            # Copy downthrown block to displaced position (down by throw)
            for k in range(nz):
                target_k = k + self.throw
                if target_k < new_nz:
                    new_grid[:, :, target_k][downthrown_mask[:, :, k]] = grid[:, :, k][
                        downthrown_mask[:, :, k]
                    ]

            # Gap at top of downthrown block (k to k+throw) already filled with defaults

        else:
            # Reverse fault: downthrown block moves UP
            # Layout: [downthrown at top | upthrown at bottom | gap at very bottom]

            # Copy upthrown block to bottom portion (shifted down by abs_throw)
            for k in range(nz):
                target_k = k + abs_throw
                if target_k < new_nz:
                    new_grid[:, :, target_k][upthrown_mask[:, :, k]] = grid[:, :, k][
                        upthrown_mask[:, :, k]
                    ]

            # Copy downthrown block to top (displaced up, so k -> k position in new grid)
            for k in range(nz):
                if k < new_nz:
                    new_grid[:, :, k][downthrown_mask[:, :, k]] = grid[:, :, k][
                        downthrown_mask[:, :, k]
                    ]

            # Gap at bottom already filled with defaults
        return new_grid

    def _displace_without_expansion(
        self,
        grid: np.typing.NDArray,
        property_name: str,
        property_type: typing.Optional[str],
    ) -> np.typing.NDArray:
        """
        Block-based displacement without grid expansion (data loss at boundaries).

        Algorithm:
        1. Create new grid same size as original
        2. Copy upthrown block (stays in place)
        3. Copy downthrown block to displaced position (may go out of bounds)
        4. Fill exposed regions with smart defaults
        """
        nx, ny, nz = self.original_shape
        # Get smart default value, preserving grid dtype
        default_value = self.defaults.get_value(
            property_name=property_name,
            grid=grid,
            property_type=property_type,
            dtype=grid.dtype,
        )
        # Create new grid filled with defaults
        new_grid = np.full((nx, ny, nz), fill_value=default_value, dtype=grid.dtype)

        # Identify blocks
        upthrown_mask = np.invert(self.displacement_mask)
        downthrown_mask = self.displacement_mask

        if self.throw > 0:
            # Normal fault: downthrown moves down
            # Copy upthrown block (unchanged)
            for k in range(nz):
                new_grid[:, :, k][upthrown_mask[:, :, k]] = grid[:, :, k][
                    upthrown_mask[:, :, k]
                ]

            # Copy downthrown block (displaced down, may lose bottom data)
            for k in range(nz):
                target_k = k + self.throw
                if target_k < nz:  # Only copy if target is in bounds
                    new_grid[:, :, target_k][downthrown_mask[:, :, k]] = grid[:, :, k][
                        downthrown_mask[:, :, k]
                    ]

            # Top of downthrown block exposed and already filled with defaults

        else:
            # Reverse fault: downthrown moves up
            abs_throw = abs(self.throw)

            # Copy upthrown block (unchanged)
            for k in range(nz):
                new_grid[:, :, k][upthrown_mask[:, :, k]] = grid[:, :, k][
                    upthrown_mask[:, :, k]
                ]

            # Copy downthrown block (displaced up, may lose top data)
            for k in range(nz):
                target_k = k - abs_throw
                if target_k >= 0:  # Only copy if target is in bounds
                    new_grid[:, :, target_k][downthrown_mask[:, :, k]] = grid[:, :, k][
                        downthrown_mask[:, :, k]
                    ]

            # Bottom of downthrown block exposed - already filled with defaults

        return new_grid


def vertical_sealing_fault(
    fault_id: str,
    orientation: typing.Literal["x", "y", "z"],
    index: int,
    permeability_multiplier: float = 1e-4,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
    defaults: typing.Optional[FractureDefaults] = None,
) -> Fracture:
    """
    Create a sealing barrier at a grid index with optional extent control.

    Creates a planar barrier that reduces fluid flow. Despite the name "vertical_sealing_fault",
    this function supports all orientations including horizontal barriers (z-orientation).

    Flexibility:
    - Control lateral extent with `x_range`/`y_range`
    - Control vertical extent with `z_range`
    - Can create stratigraphically-bounded faults
    - Can create faults that don't span entire grid
    - Supports horizontal sealing layers (z-orientation)

    :param fault_id: Unique identifier for the fault
    :param orientation: Fault orientation ('x', 'y' for vertical, 'z' for horizontal)
    :param index: Grid index where the fault/barrier is located
    :param permeability_multiplier: Flow reduction factor (default: 1e-4 = 99.99% sealing)
    :param x_range: Optional lateral extent in x
    :param y_range: Optional lateral extent in y
    :param z_range: Optional vertical extent (for x/y orientation) or layer index (for z orientation)
    :param defaults: Optional custom defaults for properties
    :return: Configured `Fracture` object

    Examples:
    ```python
    # Simple vertical fault through entire grid
    vertical_sealing_fault(fault_id="f1", orientation="x", index=25)

    # Shallow fault only in top 10 layers
    vertical_sealing_fault(
        fault_id="f1",
        orientation="x",
        index=25,
        z_range=(0, 10)
    )

    # Horizontal sealing layer (e.g., shale barrier)
    vertical_sealing_fault(
        fault_id="shale_barrier",
        orientation="z",
        index=15,
        x_range=(0, 50),
        y_range=(0, 50)
    )
    ```
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=(index, index),
            y_range=y_range,
            z_range=z_range,
        )
    elif orientation == "y":
        geometry = FractureGeometry(
            orientation="y",
            y_range=(index, index),
            x_range=x_range,
            z_range=z_range,
        )
    else:  # z-orientation
        geometry = FractureGeometry(
            orientation="z",
            z_range=(index, index),
            x_range=x_range,
            y_range=y_range,
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        permeability_multiplier=permeability_multiplier,
        defaults=defaults or FractureDefaults(),
    )


def normal_fault_with_throw(
    fault_id: str,
    orientation: typing.Literal["x", "y"],
    index: int,
    throw_cells: int,
    permeability_multiplier: float = 1e-4,
    preserve_data: bool = True,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
    displacement_range: typing.Optional[typing.Tuple[int, int]] = None,
    defaults: typing.Optional[FractureDefaults] = None,
) -> Fracture:
    """
    Create a normal fault with vertical displacement (throw).

    Normal faults occur when the hanging wall (downthrown block) moves down
    relative to the footwall. This creates juxtaposition of different reservoir
    layers across the fault. Use positive `throw_cells`.

    Displacement Behaviour:
    - By default, only the fault plane itself (at the specified index) is displaced
    - Use `displacement_range` to explicitly control which cells move
    - This allows for localized faulting or displacing entire blocks

    Flexibility:
    - Control which cells are displaced with `displacement_range`
    - Limit fault extent with x_range/y_range/z_range
    - Create segmented faults or relay ramps

    :param fault_id: Unique identifier for the fault
    :param orientation: Fault orientation ('x' or 'y')
    :param index: Grid index where the fault is located
    :param throw_cells: Number of cells to displace vertically (positive for normal fault)
    :param permeability_multiplier: Flow reduction factor (default: 1e-4)
    :param preserve_data: If True, expand grid to preserve all data (recommended)
    :param x_range: Optional lateral extent in x (for y-oriented faults)
    :param y_range: Optional lateral extent in y (for x-oriented faults)
    :param z_range: Optional vertical extent
    :param displacement_range: Optional explicit control over which cells are displaced.
        If None, only the fault plane at 'index' is displaced.
    :param defaults: Optional custom defaults for expanded regions
    :return: Configured `Fracture` object

    Examples:
    ```python
    # Simple fault - only displaces the fault plane at x=25
    normal_fault_with_throw(fault_id="f1", orientation="x", index=25, throw_cells=3)

    # Fault affecting entire block from x=30 onward
    normal_fault_with_throw(
        fault_id="f1",
        orientation="x",
        index=25,
        throw_cells=3,
        displacement_range=(30, 50)
    )

    # Shallow fault with limited throw
    normal_fault_with_throw(
        fault_id="f1",
        orientation="x",
        index=25,
        throw_cells=2,
        z_range=(0, 10)
    )
    ```
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=(index, index),
            y_range=y_range,
            z_range=z_range,
            displacement_range=displacement_range,
            geometric_throw=throw_cells,
        )
    else:
        geometry = FractureGeometry(
            orientation="y",
            y_range=(index, index),
            x_range=x_range,
            z_range=z_range,
            displacement_range=displacement_range,
            geometric_throw=throw_cells,
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        permeability_multiplier=permeability_multiplier,
        preserve_grid_data=preserve_data,
        defaults=defaults or FractureDefaults(),
    )


def reverse_fault_with_throw(
    fault_id: str,
    orientation: typing.Literal["x", "y"],
    index: int,
    throw_cells: int,
    permeability_multiplier: float = 1e-4,
    preserve_data: bool = True,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
    displacement_range: typing.Optional[typing.Tuple[int, int]] = None,
    defaults: typing.Optional[FractureDefaults] = None,
) -> Fracture:
    """
    Create a reverse fault with vertical displacement (throw).

    Reverse faults occur when the hanging wall moves up relative to the footwall,
    typically due to compressional forces. Use negative throw_cells.

    Displacement Behaviour:
    - By default, only the fault plane itself (at the specified index) is displaced
    - Use `displacement_range` to explicitly control which cells move

    :param fault_id: Unique identifier for the fault
    :param orientation: Fault orientation ('x' or 'y')
    :param index: Grid index where the fault is located
    :param throw_cells: Number of cells to displace vertically (use negative for reverse)
    :param permeability_multiplier: Flow reduction factor (default: 1e-4)
    :param preserve_data: If True, expand grid to preserve all data (recommended)
    :param x_range: Optional lateral extent in x (for y-oriented faults)
    :param y_range: Optional lateral extent in y (for x-oriented faults)
    :param z_range: Optional vertical extent
    :param displacement_range: Optional explicit control over which cells are displaced.
        If None, only the fault plane at 'index' is displaced.
    :param defaults: Optional custom defaults for expanded regions
    :return: Configured `Fracture` object
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=(index, index),
            y_range=y_range,
            z_range=z_range,
            displacement_range=displacement_range,
            geometric_throw=-abs(throw_cells),  # Ensure negative
        )
    else:
        geometry = FractureGeometry(
            orientation="y",
            y_range=(index, index),
            x_range=x_range,
            z_range=z_range,
            displacement_range=displacement_range,
            geometric_throw=-abs(throw_cells),  # Ensure negative
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        permeability_multiplier=permeability_multiplier,
        preserve_grid_data=preserve_data,
        defaults=defaults or FractureDefaults(),
    )


def inclined_sealing_fault(
    fault_id: str,
    orientation: typing.Literal["x", "y"],
    index: int,
    slope: float,
    intercept: float,
    permeability_multiplier: float = 1e-4,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
    defaults: typing.Optional[FractureDefaults] = None,
) -> Fracture:
    """
    Create an inclined (non-vertical) sealing fault.

    The fault plane follows:
    ```
    z = intercept + slope * (coordinate)
    ```
    where coordinate is x-position for y-oriented faults, or y-position for x-oriented faults.

    :param fault_id: Unique identifier for the fault
    :param orientation: Fault orientation ('x' or 'y')
    :param index: Grid index where the fault intersects
    :param slope: Slope of the fault plane (dz/dx or dz/dy)
    :param intercept: Z-intercept of the fault plane
    :param permeability_multiplier: Flow reduction factor (default: 1e-4)
    :param x_range: Optional lateral extent in x (for y-oriented faults)
    :param y_range: Optional lateral extent in y (for x-oriented faults)
    :param z_range: Optional vertical extent to clip the inclined plane
    :param defaults: Optional custom defaults for properties
    :return: Configured `Fracture` object
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=(index, index),
            y_range=y_range,
            z_range=z_range,
            slope=slope,
            intercept=intercept,
        )
    else:
        geometry = FractureGeometry(
            orientation="y",
            y_range=(index, index),
            x_range=x_range,
            z_range=z_range,
            slope=slope,
            intercept=intercept,
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        permeability_multiplier=permeability_multiplier,
        defaults=defaults or FractureDefaults(),
    )


def damage_zone_fault(
    fault_id: str,
    orientation: typing.Literal["x", "y", "z"],
    cell_range: typing.Tuple[int, int],
    permeability_multiplier: float = 1e-3,
    zone_permeability: typing.Optional[float] = None,
    zone_porosity: typing.Optional[float] = None,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
    defaults: typing.Optional[FractureDefaults] = None,
) -> Fracture:
    """
    Create a fault/barrier with a damage zone (multiple cells wide).

    Fault damage zones are regions of reduced permeability and altered rock
    properties surrounding the main fault plane. This is more realistic than
    single-cell faults for large displacement faults. Also useful for modeling
    horizontal low-permeability layers (z-orientation).

    :param fault_id: Unique identifier for the fault
    :param orientation: Fault orientation ('x', 'y' for vertical, 'z' for horizontal)
    :param cell_range: Range of cells defining the damage zone (inclusive)
    :param permeability_multiplier: Flow reduction across the zone (default: 1e-3)
    :param zone_permeability: Permeability within damage zone (mD), if different
    :param zone_porosity: Porosity within damage zone (fraction), if different
    :param x_range: Optional lateral extent in x
    :param y_range: Optional lateral extent in y
    :param z_range: Optional vertical extent (for x/y) or limits for z-orientation
    :param defaults: Optional custom defaults for properties
    :return: Configured `Fracture` object

    Examples:
    ```python
    # Wide vertical damage zone from x=20 to x=30
    damage_zone_fault(fault_id="f1", orientation="x", cell_range=(20, 30))

    # Damage zone only in middle layers
    damage_zone_fault(
        fault_id="f1",
        orientation="x",
        cell_range=(20, 25),
        z_range=(10, 20)
    )

    # Horizontal low-permeability layer (e.g., shale layer spanning z=10 to z=12)
    damage_zone_fault(
        fault_id="shale",
        orientation="z",
        cell_range=(10, 12),
        zone_permeability=0.01
    )
    ```
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x", x_range=cell_range, y_range=y_range, z_range=z_range
        )
    elif orientation == "y":
        geometry = FractureGeometry(
            orientation="y", y_range=cell_range, x_range=x_range, z_range=z_range
        )
    else:  # z-orientation
        geometry = FractureGeometry(
            orientation="z", z_range=cell_range, x_range=x_range, y_range=y_range
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        permeability_multiplier=permeability_multiplier,
        permeability=zone_permeability,
        porosity=zone_porosity,
        defaults=defaults or FractureDefaults(),
    )


def conductive_fracture_network(
    fracture_id: str,
    orientation: typing.Literal["x", "y", "z"],
    cell_range: typing.Tuple[int, int],
    fracture_permeability: float,
    fracture_porosity: float = 0.01,
    permeability_multiplier: float = 10.0,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
    defaults: typing.Optional[FractureDefaults] = None,
) -> Fracture:
    """
    Create a highly conductive fracture network (opposite of sealing fault).

    Used for modeling natural fracture systems, hydraulically-fractured zones,
    highly permeable conduits, or high-permeability horizontal layers that enhance flow.

    :param fracture_id: Unique identifier for the fracture
    :param orientation: Fracture orientation ('x', 'y' for vertical, 'z' for horizontal)
    :param cell_range: Range of cells defining the fracture network
    :param fracture_permeability: High permeability within fracture (mD)
    :param fracture_porosity: Fracture porosity (typically low, default: 0.01)
    :param permeability_multiplier: Flow enhancement factor (default: 10.0)
    :param x_range: Optional lateral extent in x
    :param y_range: Optional lateral extent in y
    :param z_range: Optional vertical extent or limits for z-orientation
    :param defaults: Optional custom defaults for properties
    :return: Configured `Fracture` object

    Examples:
    ```python
    # Conductive vertical fracture in pay zone only
    conductive_fracture_network(
        fault_id="frac1",
        orientation="x",
        cell_range=(25, 27),
        fracture_permeability=1000,
        z_range=(10, 20)
    )

    # Horizontal high-permeability layer (e.g., karst zone)
    conductive_fracture_network(
        fault_id="karst",
        orientation="z",
        cell_range=(15, 17),
        fracture_permeability=5000
    )
    ```
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=cell_range,
            y_range=y_range,
            z_range=z_range,
        )
    elif orientation == "y":
        geometry = FractureGeometry(
            orientation="y",
            y_range=cell_range,
            x_range=x_range,
            z_range=z_range,
        )
    else:  # z-orientation
        geometry = FractureGeometry(
            orientation="z",
            z_range=cell_range,
            x_range=x_range,
            y_range=y_range,
        )

    return Fracture(
        id=fracture_id,
        geometry=geometry,
        permeability=fracture_permeability,
        porosity=fracture_porosity,
        permeability_multiplier=permeability_multiplier,
        conductive=True,
        defaults=defaults or FractureDefaults(),
    )


def fault_with_throw_and_damage_zone(
    fault_id: str,
    orientation: typing.Literal["x", "y"],
    cell_range: typing.Tuple[int, int],
    throw_cells: int,
    permeability_multiplier: float = 1e-4,
    zone_permeability: typing.Optional[float] = None,
    zone_porosity: typing.Optional[float] = None,
    preserve_data: bool = True,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
    displacement_range: typing.Optional[typing.Tuple[int, int]] = None,
    defaults: typing.Optional[FractureDefaults] = None,
) -> Fracture:
    """
    Create a fault with both geometric throw and a damage zone.

    This combines vertical displacement with altered rock properties across
    multiple cells, representing realistic large-displacement faults.

    Displacement Behaviour:
    - By default, only the damage zone cells (cell_range) are displaced
    - Use displacement_range to control which cells move (e.g., displace entire blocks)

    :param fault_id: Unique identifier for the fault
    :param orientation: Fault orientation ('x' or 'y')
    :param cell_range: Range of cells defining the damage zone
    :param throw_cells: Number of cells to displace vertically
    :param permeability_multiplier: Flow reduction factor (default: 1e-4)
    :param zone_permeability: Permeability within damage zone (mD)
    :param zone_porosity: Porosity within damage zone (fraction)
    :param preserve_data: If True, expand grid to preserve all data
    :param x_range: Optional lateral extent in x (for y-oriented faults)
    :param y_range: Optional lateral extent in y (for x-oriented faults)
    :param z_range: Optional vertical extent
    :param displacement_range: Optional explicit control over which cells are displaced.
        If None, only the damage zone (cell_range) is displaced.
    :param defaults: Optional custom defaults for expanded regions
    :return: Configured `Fracture` object

    Examples:
    ```python
    # Fault with damage zone - by default only displaces the damage zone itself
    fault_with_throw_and_damage_zone(
        fault_id="f1",
        orientation="x",
        cell_range=(20, 25),
        throw_cells=3
    )

    # Fault with damage zone, but displacing entire right side
    fault_with_throw_and_damage_zone(
        fault_id="f1",
        orientation="x",
        cell_range=(20, 25),
        throw_cells=3,
        displacement_range=(26, 50)
    )

    # Segmented fault in specific layers
    fault_with_throw_and_damage_zone(
        fault_id="f1",
        orientation="x",
        cell_range=(20, 25),
        throw_cells=2,
        z_range=(10, 20)
    )
    ```
    """
    if orientation == "x":
        geometry = FractureGeometry(
            orientation="x",
            x_range=cell_range,
            y_range=y_range,
            z_range=z_range,
            displacement_range=displacement_range,
            geometric_throw=throw_cells,
        )
    else:
        geometry = FractureGeometry(
            orientation="y",
            y_range=cell_range,
            x_range=x_range,
            z_range=z_range,
            displacement_range=displacement_range,
            geometric_throw=throw_cells,
        )

    return Fracture(
        id=fault_id,
        geometry=geometry,
        permeability_multiplier=permeability_multiplier,
        permeability=zone_permeability,
        porosity=zone_porosity,
        preserve_grid_data=preserve_data,
        defaults=defaults or FractureDefaults(),
    )


def validate_fracture(
    fracture: Fracture, grid_shape: typing.Tuple[int, ...]
) -> typing.List[str]:
    """
    Validate fracture configuration against grid dimensions.

    :param fracture: Fracture configuration to validate
    :param grid_shape: Shape of the reservoir grid (nx, ny, nz)
    :return: List of validation error messages (empty if valid)
    """
    errors = []

    if len(grid_shape) != 3:
        errors.append("Grid must be 3D")
        return errors

    nx, ny, nz = grid_shape
    geom = fracture.geometry

    # Validate primary range based on orientation
    if geom.orientation == "x":
        if geom.x_range is not None:
            x_min, x_max = geom.x_range
            if x_min > x_max:
                errors.append(
                    f"Fracture {fracture.id!r}: x_range min > max ({x_min} > {x_max})"
                )
            if not (0 <= x_min < nx and 0 <= x_max < nx):
                errors.append(
                    f"Fracture {fracture.id!r}: x_range ({x_min}, {x_max}) out of bounds [0, {nx - 1}]"
                )

    elif geom.orientation == "y":
        if geom.y_range is not None:
            y_min, y_max = geom.y_range
            if y_min > y_max:
                errors.append(
                    f"Fracture {fracture.id!r}: y_range min > max ({y_min} > {y_max})"
                )
            if not (0 <= y_min < ny and 0 <= y_max < ny):
                errors.append(
                    f"Fracture {fracture.id!r}: y_range ({y_min}, {y_max}) out of bounds [0, {ny - 1}]"
                )

    elif geom.orientation == "z":
        if geom.z_range is not None:
            z_min, z_max = geom.z_range
            if z_min > z_max:
                errors.append(
                    f"Fracture {fracture.id!r}: z_range min > max ({z_min} > {z_max})"
                )
            if not (0 <= z_min < nz and 0 <= z_max < nz):
                errors.append(
                    f"Fracture {fracture.id!r}: z_range ({z_min}, {z_max}) out of bounds [0, {nz - 1}]"
                )

    # Validate z_range if specified (for x/y oriented fractures)
    if geom.orientation in ("x", "y") and geom.z_range is not None:
        z_min, z_max = geom.z_range
        if z_min > z_max:
            errors.append(
                f"Fracture {fracture.id!r}: z_range min > max ({z_min} > {z_max})"
            )
        if not (0 <= z_min < nz and 0 <= z_max < nz):
            errors.append(
                f"Fracture {fracture.id!r}: z_range ({z_min}, {z_max}) out of bounds [0, {nz - 1}]"
            )

    # Validate intercept if applicable
    if geom.intercept != 0.0 and not (0 <= geom.intercept < nz):
        errors.append(
            f"Fracture {fracture.id!r}: intercept {geom.intercept} out of z-range [0, {nz - 1}]"
        )

    # Validate geometric throw
    if geom.orientation == "z" and geom.geometric_throw != 0:
        errors.append(
            f"Fracture {fracture.id!r}: geometric_throw is not supported for z-oriented fractures"
        )
    if abs(geom.geometric_throw) >= nz:
        errors.append(
            f"Fracture {fracture.id!r}: geometric_throw {geom.geometric_throw} too large for nz={nz}"
        )

    # Validate displacement_range if specified
    if geom.displacement_range is not None:
        disp_min, disp_max = geom.displacement_range
        if disp_min > disp_max:
            errors.append(
                f"Fracture {fracture.id!r}: displacement_range min > max ({disp_min} > {disp_max})"
            )
        if geom.orientation == "x" and not (0 <= disp_min < nx and 0 <= disp_max < nx):
            errors.append(
                f"Fracture {fracture.id!r}: displacement_range ({disp_min}, {disp_max}) out of x-bounds [0, {nx - 1}]"
            )
        elif geom.orientation == "y" and not (
            0 <= disp_min < ny and 0 <= disp_max < ny
        ):
            errors.append(
                f"Fracture {fracture.id!r}: displacement_range ({disp_min}, {disp_max}) out of y-bounds [0, {ny - 1}]"
            )
        elif geom.orientation == "z" and not (
            0 <= disp_min < nz and 0 <= disp_max < nz
        ):
            errors.append(
                f"Fracture {fracture.id!r}: displacement_range ({disp_min}, {disp_max}) out of z-bounds [0, {nz - 1}]"
            )

    # Validate mask shape if provided
    if fracture.mask is not None and fracture.mask.shape != grid_shape:
        errors.append(
            f"Fracture {fracture.id!r}: mask shape {fracture.mask.shape} != grid shape {grid_shape}"
        )

    return errors


def make_displacement_mask(
    grid_shape: typing.Tuple[int, int, int], fracture: Fracture
) -> np.ndarray:
    """
    Create a boolean mask indicating which cells should be displaced by the fracture.

    This defines the "downthrown block", i.e., cells that will move vertically.

    DISPLACEMENT LOGIC:

    If displacement_range is specified in geometry:
        - Only cells within that explicit range are displaced
        - Provides fine-grained control over which cells move

    If displacement_range is None (default):
        - Only the fracture zone itself is displaced
        - For x-oriented: cells within x_range are displaced
        - For y-oriented: cells within y_range are displaced
        - For z-oriented: cells within z_range are displaced
        - This creates localized faulting rather than displacing entire half-grids

    :param grid_shape: Shape of the reservoir grid (nx, ny, nz)
    :param fracture: Fracture configuration
    :return: 3D boolean array marking cells to be displaced
    """
    nx, ny, nz = grid_shape
    displacement_mask = np.zeros((nx, ny, nz), dtype=bool)

    geom = fracture.geometry

    # Determine the displacement range
    if geom.displacement_range is not None:
        # Explicit displacement_range provided
        disp_min, disp_max = geom.displacement_range
    else:
        # Default: displace only the fracture zone itself
        fault_plane_range = geom.get_fracture_plane_range()
        disp_min, disp_max = fault_plane_range

    # Apply displacement based on orientation
    if geom.orientation == "x":
        # Displace cells in x-range [disp_min, disp_max]
        if disp_min < nx and disp_max >= 0:
            disp_min = max(0, disp_min)
            disp_max = min(nx - 1, disp_max)
            displacement_mask[disp_min : disp_max + 1, :, :] = True

    elif geom.orientation == "y":
        # Displace cells in y-range [disp_min, disp_max]
        if disp_min < ny and disp_max >= 0:
            disp_min = max(0, disp_min)
            disp_max = min(ny - 1, disp_max)
            displacement_mask[:, disp_min : disp_max + 1, :] = True

    elif geom.orientation == "z":
        # Displace cells in z-range [disp_min, disp_max]
        if disp_min < nz and disp_max >= 0:
            disp_min = max(0, disp_min)
            disp_max = min(nz - 1, disp_max)
            displacement_mask[:, :, disp_min : disp_max + 1] = True

    return displacement_mask
