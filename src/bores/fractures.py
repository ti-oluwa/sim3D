"""
### Fracture and Fault Modeling for 3D Reservoir Simulation

This module provides a comprehensive API for modeling fractures, faults, and their effects
on fluid flow in 3D reservoir simulations. Faults can act as barriers, conduits, or both,
depending on their properties and the reservoir conditions.

### Core Concepts

**Faults vs Fractures:**
- **Faults**: Geological discontinuities that offset rock layers, typically sealing or
  partially sealing to fluid flow. Modeled using transmissibility multipliers.
- **Fractures**: Enhanced permeability pathways (natural or induced) that facilitate
  fluid flow.
- **Damage Zones**: Regions of altered rock properties adjacent to major faults.

**Industry-Standard Approach:**
This module uses transmissibility multipliers (permeability_multiplier) to model fault
effects, following commercial simulator practices (Eclipse, CMG, tNavigator). This approach
modifies cross-fault flow without changing grid geometry, making it compatible with wells,
boundary conditions, and user-defined property distributions.

### Quick Start Examples

**1. Simple Sealing Fault (Vertical Barrier)**

Create a fault that reduces cross-fault flow by 99.99% across the entire reservoir:

```python
from bores.fractures import vertical_sealing_fault, apply_fracture

# Vertical fault at x=25, spanning entire grid
fault = vertical_sealing_fault(
    fault_id="main_fault",
    orientation="x",
    index=25,
    permeability_multiplier=1e-4  # 99.99% sealing
)

# Apply to reservoir model
model = apply_fracture(model=reservoir_model, fracture=fault)
```

**2. Stratigraphically-Bounded Fault (Limited Vertical Extent)**

Model a fault that only affects specific reservoir layers:

```python
# Fault in shallow layers only (top 15 cells)
shallow_fault = vertical_sealing_fault(
    fault_id="shallow_fault",
    orientation="y",
    index=40,
    z_range=(0, 15),  # Only in upper layers
    permeability_multiplier=5e-5
)
```

**3. Partially Sealing Fault (Calibrated from History Match)**

After history matching, you find a fault transmissibility of 0.01:

```python
# Fault allows 1% of matrix flow across it
calibrated_fault = vertical_sealing_fault(
    fault_id="fault_f3",
    orientation="x",
    index=50,
    permeability_multiplier=0.01,  # 1% transmissibility
    y_range=(0, 60),  # Limited lateral extent
    z_range=(10, 35)  # Offsetting layers 10-35
)
```

**4. Inclined Fault (Non-Vertical Plane)**

Model a dipping fault plane:

```python
from bores.fractures import inclined_sealing_fault

# Fault dipping at 60° (slope ≈ 1.73 = tan(60°))
# Intersects at y=30, dips eastward
dipping_fault = inclined_sealing_fault(
    fault_id="dipping_fault",
    orientation="y",  # Strikes in y-direction
    index=30,  # Intersects at y=30
    slope=1.73,  # dz/dx (eastward dip)
    intercept=5.0,  # Fault at z=5 when x=0
    permeability_multiplier=1e-4
)
```

**5. Fault with Damage Zone (Altered Rock Properties)**

Major faults often have damage zones with reduced permeability and porosity:

```python
from bores.fractures import damage_zone_fault

# Fault with 5-cell-wide damage zone
fault_with_damage = damage_zone_fault(
    fault_id="thrust_fault",
    orientation="x",
    cell_range=(48, 52),  # 5 cells wide (x=48 to x=52)
    permeability_multiplier=1e-5,  # Very low cross-fault flow
    zone_permeability=10.0,  # Damaged rock: 10 mD (vs 100 mD matrix)
    zone_porosity=0.12,  # Reduced porosity (vs 0.20 matrix)
    z_range=(15, 45)  # Only in producing interval
)
```

**6. Conductive Fracture Network (Enhanced Flow Paths)**

Model natural or hydraulic fractures that enhance permeability:

```python
from bores.fractures import conductive_fracture_network

# Vertical fracture swarm enhancing flow
fracture_swarm = conductive_fracture_network(
    fracture_id="fracture_corridor",
    orientation="y",
    cell_range=(15, 18),  # 3-cell-wide corridor
    fracture_permeability=5000.0,  # High perm: 5 Darcy
    fracture_porosity=0.01,  # Low storage (fractures)
    permeability_multiplier=10.0,  # 10x enhanced cross-corridor flow
    z_range=(20, 40)
)
```

**7. Multiple Faults (Compartmentalized Reservoir)**

Apply several faults to create isolated compartments:

```python
from bores.fractures import apply_fractures

# Define fault system
faults = [
    vertical_sealing_fault("fault_1", "x", 25, permeability_multiplier=1e-4),
    vertical_sealing_fault("fault_2", "y", 35, permeability_multiplier=5e-4),
    vertical_sealing_fault("fault_3", "x", 60, permeability_multiplier=2e-4),
]

# Apply all faults at once
model = apply_fractures(reservoir_model, *faults)
```

**8. Horizontal Sealing Layer (Stratigraphic Barrier)**

Model an impermeable shale layer:

```python
# Horizontal shale barrier at depth layer z=12
shale_barrier = vertical_sealing_fault(
    fault_id="shale_layer",
    orientation="z",  # Horizontal orientation
    index=12,  # At layer 12
    permeability_multiplier=1e-6,  # Nearly impermeable
    x_range=(0, 80),  # Lateral extent
    y_range=(0, 60)
)
```

### Best Practices

**Choosing Permeability Multipliers:**
- **Sealing faults**: 1e-6 to 1e-3 (highly sealing to moderate barrier)
- **Partially sealing**: 1e-2 to 0.1 (allow some cross-flow)
- **Open faults/fractures**: 1.0 to 100.0 (no barrier to enhanced flow)
- **Calibrate from**: history matching, well test interference, or tracer studies

**Fault Extent Control:**
- Use `x_range`, `y_range`, `z_range` to limit fault extent
- Segmented faults are common in nature (relay ramps, fault tip zones)
- Shallow faults may not penetrate deep layers

**Damage Zones:**
- Typical width: 1-10% of fault displacement
- Permeability reduction: 10-100x relative to matrix
- Porosity reduction: 20-40% of matrix values

**Performance Tips:**
- Fault masks are computed once during model setup
- Minimal runtime overhead after fracture application
- Multiple faults are efficiently handled via vectorized operations

### Validation

Before simulation, validate fault configurations:

```python
from bores.fractures import validate_fracture

errors = validate_fracture(fracture=fault, grid_shape=(100, 80, 50))
if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

### API Reference

**Factory Functions** (recommended entry points):
- `vertical_sealing_fault()` - Planar sealing barriers (most common)
- `inclined_sealing_fault()` - Dipping/inclined fault planes
- `damage_zone_fault()` - Faults with altered rock properties
- `conductive_fracture_network()` - Enhanced permeability zones

**Core Classes:**
- `Fracture` - Main fault/fracture configuration object
- `FractureGeometry` - Geometric definition (orientation, extent, inclination)

**Application Functions:**
- `apply_fracture()` - Apply single fault to reservoir model
- `apply_fractures()` - Apply multiple faults efficiently
- `validate_fracture()` - Check fault configuration validity

### Technical Details

**Transmissibility Modification:**
Faults modify inter-block transmissibility across fault planes:

    T_fault = T_matrix x permeability_multiplier

For a sealing fault with multiplier=1e-4, cross-fault flow is reduced by 99.99%.

**Mask-Based Implementation:**
Each fault creates a 3D boolean mask defining affected cells. Multiple faults combine
via logical operations. Numba JIT compilation ensures high performance.

**Compatibility:**
- Preserves grid geometry (no cell displacement)
- Compatible with wells at any location
- Works with all boundary condition types
- Supports all evolution schemes (IMPES, explicit, implicit)

### See Also
- `bores.models.ReservoirModel` - Reservoir model class
- `bores.boundary_conditions` - Domain boundary handling
- `bores.wells` - Well modeling

### Examples Directory
See `scenarios/` for complete workflow examples including faulted reservoir simulations.
"""

import logging
import typing

import attrs
import numba  # type: ignore[import-untyped]
import numpy as np

from bores.constants import c
from bores.errors import ValidationError
from bores.models import ReservoirModel
from bores.serialization import Serializable
from bores.stores import StoreSerializable
from bores.types import ThreeDimensions

__all__ = [
    "Fracture",
    "FractureGeometry",
    "apply_fracture",
    "apply_fractures",
    "validate_fracture",
    "vertical_sealing_fault",
    "inclined_sealing_fault",
    "damage_zone_fault",
    "conductive_fracture_network",
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


@attrs.frozen
class FractureGeometry(Serializable):
    """
    Fracture geometry specification.

    Fracture Orientation:
    - `"x"`: Fracture plane perpendicular to x-axis (strikes in y-direction)
    - `"y"`: Fracture plane perpendicular to y-axis (strikes in x-direction)
    - `"z"`: Fracture plane perpendicular to z-axis (horizontal fracture/bedding slip)

    Coordinate Specification:
    The fracture is defined by specifying ranges in each dimension:

    For x-oriented fracture:
        - `x_range`: Location of fracture plane (can be single cell or damage zone)
        - `y_range`: Optional lateral extent (None = full extent)
        - `z_range`: Optional vertical extent (None = full extent)

    For y-oriented fracture:
        - `y_range`: Location of fracture plane (can be single cell or damage zone)
        - `x_range`: Optional lateral extent (None = full extent)
        - `z_range`: Optional vertical extent (None = full extent)

    For z-oriented fracture (horizontal):
        - `z_range`: Location of fracture plane (can be single layer or zone)
        - `x_range`: Optional lateral extent (None = full extent)
        - `y_range`: Optional lateral extent (None = full extent)

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


@attrs.frozen
class Fracture(StoreSerializable):
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
    - Must be > `MINIMUM_TRANSMISSIBILITY_FACTOR` for numerical stability
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
            if self.permeability_multiplier < c.MINIMUM_TRANSMISSIBILITY_FACTOR:
                object.__setattr__(
                    self, "permeability_multiplier", c.MINIMUM_TRANSMISSIBILITY_FACTOR
                )
                logger.warning(
                    f"Fracture {self.id!r}: `permeability_multiplier` clamped to {c.MINIMUM_TRANSMISSIBILITY_FACTOR}"
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
    based on the fracture configuration. The input model is not modified inplace.

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

    # Apply fracture effects
    # Modify fracture zone properties
    if fracture.permeability is not None or fracture.porosity is not None:
        model = _apply_fracture_zone_properties(
            model=model, fracture_mask=fracture_mask, fracture=fracture
        )

    # Scale permeabilities across fracture
    if fracture.permeability_multiplier is not None:
        model = _scale_permeability(
            model=model, fracture_mask=fracture_mask, fracture=fracture
        )

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


def vertical_sealing_fault(
    fault_id: str,
    orientation: typing.Literal["x", "y", "z"],
    index: int,
    permeability_multiplier: float = 1e-4,
    x_range: typing.Optional[typing.Tuple[int, int]] = None,
    y_range: typing.Optional[typing.Tuple[int, int]] = None,
    z_range: typing.Optional[typing.Tuple[int, int]] = None,
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

    # Validate mask shape if provided
    if fracture.mask is not None and fracture.mask.shape != grid_shape:
        errors.append(
            f"Fracture {fracture.id!r}: mask shape {fracture.mask.shape} != grid shape {grid_shape}"
        )

    return errors
