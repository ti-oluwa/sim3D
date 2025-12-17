"""
Fracture definition system for 3D reservoir models.
"""

import copy
import itertools
import logging
import typing

import attrs
import numpy as np

from bores.constants import c
from bores.models import ReservoirModel
from bores.types import NDimension, ThreeDimensions

__all__ = [
    "Fracture",
    "apply_fracture",
    "apply_fractures",
    "validate_fracture",
    "FractureDefaults",
]

logger = logging.getLogger(__name__)


FloatDefault = typing.Union[float, np.floating[typing.Any]]
StatDefault = typing.Literal[
    "mean", "median", "min", "max", "zero", "nearest", "linear"
]
HookDefault = typing.Callable[[np.typing.NDArray], FloatDefault]
DefaultDef = typing.Union[FloatDefault, StatDefault, HookDefault]


def _compute_nearest_default(
    grid: np.typing.NDArray,
    valid_data: np.typing.NDArray,
    property_type: typing.Optional[str] = None,
) -> float:
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
    if len(boundary_valid) > 0:
        return float(np.median(boundary_valid))
    return float(np.mean(valid_data))


def _compute_linear_default(
    grid: np.typing.NDArray,
    valid_data: np.typing.NDArray,
    property_type: typing.Optional[str] = None,
) -> float:
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
    if len(boundary_valid) > 0:
        return float(np.mean(boundary_valid))
    return float(np.mean(valid_data))


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

    # Geometric properties
    thickness: DefaultDef = "mean"

    # Rock properties
    porosity: DefaultDef = "mean"
    permeability: DefaultDef = "mean"
    net_to_gross: DefaultDef = "mean"

    # Saturation properties
    water_saturation: DefaultDef = "mean"
    oil_saturation: DefaultDef = "mean"
    gas_saturation: DefaultDef = "mean"
    irreducible_water_saturation: DefaultDef = "mean"
    residual_oil_saturation_water: DefaultDef = "mean"
    residual_oil_saturation_gas: DefaultDef = "mean"
    residual_gas_saturation: DefaultDef = "mean"

    # Pressure and temperature
    pressure: DefaultDef = "median"  # Use median for pressure
    temperature: DefaultDef = "median"

    # Fluid properties
    oil_viscosity: DefaultDef = "mean"
    water_viscosity: DefaultDef = "mean"
    gas_viscosity: DefaultDef = "mean"

    oil_density: DefaultDef = "mean"
    water_density: DefaultDef = "mean"
    gas_density: DefaultDef = "mean"

    # Formation volume factors
    oil_fvf: DefaultDef = "mean"
    water_fvf: DefaultDef = "mean"
    gas_fvf: DefaultDef = "mean"

    # Other properties
    compressibility: DefaultDef = "mean"
    bubble_point_pressure: DefaultDef = "mean"

    # Generic fallback for unspecified properties
    generic: DefaultDef = "mean"

    def get_value(
        self,
        property_name: str,
        grid: np.typing.NDArray,
        property_type: typing.Optional[str] = None,
    ) -> float:
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

        # Handle different specification types
        if isinstance(spec, (int, float)):
            return float(spec)

        if callable(spec):
            return float(spec(grid))  # type: ignore[call-arg]

        if isinstance(spec, str):
            spec = typing.cast(StatDefault, spec)
            return self._compute_statistical_default(
                method=spec, grid=grid, property_type=property_type
            )

        # Fallback to mean if all else fails
        return self._compute_statistical_default(
            method="mean", grid=grid, property_type=property_type
        )

    def _compute_statistical_default(
        self,
        method: StatDefault,
        grid: np.typing.NDArray,
        property_type: typing.Optional[str] = None,
    ) -> float:
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
            raise ValueError(f"No valid data available for {property_type}")

        # Compute based on method
        if method == "mean":
            return float(np.mean(valid_data))
        elif method == "median":
            return float(np.median(valid_data))
        elif method == "min":
            return float(np.min(valid_data))
        elif method == "max":
            return float(np.max(valid_data))
        elif method == "zero":
            return 0.0
        elif method == "nearest":
            return _compute_nearest_default(grid, valid_data, property_type)
        elif method == "linear":
            return _compute_linear_default(grid, valid_data, property_type)

        logger.warning(f"Unknown method '{method}', using mean")
        return float(np.mean(valid_data))


@attrs.define(slots=True, frozen=True)
class Fracture:
    """
    Configuration for applying fractures to reservoir models.

    Defines the geometric and hydraulic properties of a fracture,
    including its orientation, hydraulic properties, and modeling approach.

    Note: A geological fault is a specific type of fracture.
    """

    id: str
    """Unique identifier for the fracture."""

    slope: float = 0.0
    """
    Slope of the fracture plane: z = z0 + slope*(x - x0) for x-oriented fractures,
    or z = z0 + slope*(y - y0) for y-oriented fractures.
    A slope of 0.0 creates a vertical fracture.
    """

    intercept: float = 0.0
    """
    Z-intercept (z0) of the fracture plane at the reference position.
    For vertical fractures, this represents the base z-level of the fracture.
    """

    orientation: typing.Literal["x", "y"] = "x"
    """
    Orientation of the fracture plane:
    - "x": fracture plane cuts through x-direction (perpendicular to x-axis)
    - "y": fracture plane cuts through y-direction (perpendicular to y-axis)
    """

    fracture_index: int = 0
    """
    Nominal grid index where the fracture intersects:
    - For x-oriented fractures: x-index of the fracture plane
    - For y-oriented fractures: y-index of the fracture plane
    """

    transmissibility_scale: float = 1e-3
    """
    Scaling factor for transmissibilities across the fracture.
    - Values < 1.0 create sealing barriers (typical: 1e-3 to 1e-6)
    - Values > 1.0 create conductive zones (for fractured fractures)
    - Must be > MIN_TRANSMISSIBILITY_FACTOR for numerical stability
    """

    fracture_permeability: typing.Optional[float] = None
    """
    Permeability value for fracture zone cells (mD).
    If None, fracture zone properties are not modified.
    """

    fracture_porosity: typing.Optional[float] = None
    """
    Porosity value for fracture zone cells (fraction).
    If None, fracture zone properties are not modified.
    """

    geometric_throw_cells: int = 0
    """
    Number of cells to displace the downthrown block in z-direction.
    Positive values create normal fractures (faults), negative values create reverse fractures (faults).
    """

    conductive: bool = False
    """
    If True, the fracture acts as a high-permeability conduit.
    This automatically sets appropriate transmissibility_scale if not manually specified.
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
    If None, uses `FractureDefaults()` with sensible defaults.
    """

    def __attrs_post_init__(self) -> None:
        """Validate fracture configuration parameters."""
        if self.transmissibility_scale < c.MIN_TRANSMISSIBILITY_FACTOR:
            object.__setattr__(
                self, "transmissibility_scale", c.MIN_TRANSMISSIBILITY_FACTOR
            )
            logger.warning(
                f"Fracture {self.id}: transmissibility_scale clamped to {c.MIN_TRANSMISSIBILITY_FACTOR}"
            )

        if self.conductive and self.transmissibility_scale < 1.0:
            raise ValueError(
                f"Fracture {self.id}: conductive fractures must have transmissibility_scale >= 1.0"
            )

        if self.fracture_permeability is not None and self.fracture_permeability < 0:
            raise ValueError(
                f"Fracture {self.id}: fracture_permeability must be non-negative"
            )

        if self.fracture_porosity is not None and not (
            0 <= self.fracture_porosity <= 1
        ):
            raise ValueError(
                f"Fracture {self.id}: fracture_porosity must be between 0 and 1"
            )


def apply_fracture(
    model: ReservoirModel[NDimension], fracture: Fracture
) -> ReservoirModel[NDimension]:
    """
    Apply a single fracture to a reservoir model.

    This function modifies transmissibilities, rock properties, and geometry
    based on the fracture configuration. The input model is not modified;
    a new model instance is returned.

    :param model: Input reservoir model to modify
    :param fracture: Fracture configuration defining geometry and properties
    :return: New reservoir model with fracture applied
    """
    logger.debug(f"Applying fracture '{fracture.id}' to reservoir model")

    errors = validate_fracture(fracture, model.grid_shape)
    if errors:
        error = ""
        for e in errors:
            error += f"\n - {e}"
        raise ValueError(f"Fracture {fracture.id} configuration is invalid: {error}")

    new_model = copy.deepcopy(model)
    grid_shape = new_model.grid_shape

    if len(grid_shape) != 3:
        raise ValueError("Fracture application requires 3D reservoir models")

    new_model = typing.cast(ReservoirModel[ThreeDimensions], new_model)

    # Generate or validate fracture mask
    if fracture.mask is not None:
        if fracture.mask.shape != grid_shape:
            raise ValueError(
                f"Fracture {fracture.id}: mask shape {fracture.mask.shape} != grid shape {grid_shape}"
            )
        fracture_mask = fracture.mask.copy()
    else:
        fracture_mask = _make_fracture_mask(grid_shape, fracture)  # type: ignore[arg-type]

    # Apply fracture effects in proper order
    # Modify fracture zone properties (before geometric displacement)
    if (
        fracture.fracture_permeability is not None
        or fracture.fracture_porosity is not None
    ):
        new_model = _apply_fracture_zone_properties(new_model, fracture_mask, fracture)

    # Scale transmissibilities across fracture (before geometric displacement)
    new_model = _scale_transmissibility(new_model, fracture_mask, fracture)

    # Apply geometric displacement (throw). This may change grid dimensions
    if fracture.geometric_throw_cells != 0:
        new_model = _apply_geometric_throw(new_model, fracture_mask, fracture)

    logger.debug(f"Successfully applied fracture '{fracture.id}'")
    return typing.cast(ReservoirModel[NDimension], new_model)


def apply_fractures(
    model: ReservoirModel[NDimension], *fractures: Fracture
) -> ReservoirModel[NDimension]:
    """
    Apply multiple fractures to a reservoir model.

    Fractures are applied sequentially in the order provided.

    :param model: Input reservoir model
    :param fractures: Sequence of fracture configurations
    :return: New reservoir model with all fractures applied
    """
    logger.debug(f"Applying {len(fractures)} fractures to reservoir model")

    faulted_model = model
    for fracture in fractures:
        faulted_model = apply_fracture(faulted_model, fracture)

    logger.debug(f"Successfully applied all {len(fractures)} fractures")
    return faulted_model


def _make_fracture_mask(
    grid_shape: typing.Tuple[int, int, int], fracture: Fracture
) -> np.ndarray:
    """
    Generate a 3D boolean mask defining the fracture geometry.

    For inclined fractures, the mask follows the equation:
    z = z0 + slope * (coord - coord0)

    :param grid_shape: Shape of the reservoir grid (nx, ny, nz)
    :param fracture: Fracture configuration
    :return: 3D boolean array marking fracture cells
    """
    nx, ny, nz = grid_shape
    mask = np.zeros((nx, ny, nz), dtype=bool)

    if fracture.orientation == "x":
        # Fracture cuts through x-direction
        if abs(fracture.slope) < 1e-6:
            # Vertical fracture
            if 0 <= fracture.fracture_index < nx:
                mask[fracture.fracture_index, :, :] = True
        else:
            # Inclined fracture
            for j in range(ny):
                fracture_z = fracture.intercept + fracture.slope * (
                    j - fracture.fracture_index
                )
                fracture_z = np.clip(fracture_z, 0, nz - 1)
                z_low = max(0, int(fracture_z - 0.5))
                z_high = min(nz - 1, int(fracture_z + 0.5))
                mask[fracture.fracture_index, j, z_low : z_high + 1] = True

    elif fracture.orientation == "y":
        # Fracture cuts through y-direction
        if abs(fracture.slope) < 1e-6:
            # Vertical fracture
            if 0 <= fracture.fracture_index < ny:
                mask[:, fracture.fracture_index, :] = True
        else:
            # Inclined fracture
            for i in range(nx):
                fracture_z = fracture.intercept + fracture.slope * (
                    i - fracture.fracture_index
                )
                fracture_z = np.clip(fracture_z, 0, nz - 1)
                z_low = max(0, int(fracture_z - 0.5))
                z_high = min(nz - 1, int(fracture_z + 0.5))
                mask[i, fracture.fracture_index, z_low : z_high + 1] = True

    return mask


def _scale_transmissibility(
    model: ReservoirModel[ThreeDimensions],
    fracture_mask: np.ndarray,
    fracture: Fracture,
) -> ReservoirModel[ThreeDimensions]:
    """
    Scale transmissibilities across fracture boundaries.

    Identifies connections that cross the fracture and scales permeability
    to reduce/increase transmissibility.

    :param model: Reservoir model to modify
    :param fracture_mask: 3D boolean array marking fracture cells
    :param fracture: Fracture configuration
    :return: Modified reservoir model
    """
    logger.debug(f"Scaling transmissibilities for fracture '{fracture.id}'")

    nx, ny, nz = model.grid_shape

    # Scale connections based on fracture orientation
    if fracture.orientation == "x":
        # Scale x-direction connections
        for i, j, k in itertools.product(range(nx - 1), range(ny), range(nz)):
            if fracture_mask[i, j, k] or fracture_mask[i + 1, j, k]:
                if fracture_mask[i, j, k]:
                    perm = model.rock_properties.absolute_permeability.x[i, j, k]
                    model.rock_properties.absolute_permeability.x[i, j, k] = (
                        perm * fracture.transmissibility_scale
                    )
                if fracture_mask[i + 1, j, k]:
                    perm = model.rock_properties.absolute_permeability.x[i + 1, j, k]
                    model.rock_properties.absolute_permeability.x[i + 1, j, k] = (
                        perm * fracture.transmissibility_scale
                    )

    elif fracture.orientation == "y":
        # Scale y-direction connections
        for i, j, k in itertools.product(range(nx), range(ny - 1), range(nz)):
            if fracture_mask[i, j, k] or fracture_mask[i, j + 1, k]:
                if fracture_mask[i, j, k]:
                    perm = model.rock_properties.absolute_permeability.y[i, j, k]
                    model.rock_properties.absolute_permeability.y[i, j, k] = (
                        perm * fracture.transmissibility_scale
                    )
                if fracture_mask[i, j + 1, k]:
                    perm = model.rock_properties.absolute_permeability.y[i, j + 1, k]
                    model.rock_properties.absolute_permeability.y[i, j + 1, k] = (
                        perm * fracture.transmissibility_scale
                    )

    # Always scale z-direction for inclined fractures
    for i, j, k in itertools.product(range(nx), range(ny), range(nz - 1)):
        if fracture_mask[i, j, k] or fracture_mask[i, j, k + 1]:
            if fracture_mask[i, j, k]:
                perm = model.rock_properties.absolute_permeability.z[i, j, k]
                model.rock_properties.absolute_permeability.z[i, j, k] = (
                    perm * fracture.transmissibility_scale
                )
            if fracture_mask[i, j, k + 1]:
                perm = model.rock_properties.absolute_permeability.z[i, j, k + 1]
                model.rock_properties.absolute_permeability.z[i, j, k + 1] = (
                    perm * fracture.transmissibility_scale
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

    if fracture.fracture_permeability is not None:
        model.rock_properties.absolute_permeability.x[fracture_mask] = (
            fracture.fracture_permeability
        )
        model.rock_properties.absolute_permeability.y[fracture_mask] = (
            fracture.fracture_permeability
        )
        model.rock_properties.absolute_permeability.z[fracture_mask] = (
            fracture.fracture_permeability
        )

    if fracture.fracture_porosity is not None:
        model.rock_properties.porosity_grid[fracture_mask] = fracture.fracture_porosity
    return model


def _apply_geometric_throw(
    model: ReservoirModel[ThreeDimensions],
    fracture_mask: np.typing.NDArray,
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
    logger.debug(f"Applying geometric throw ({fracture.geometric_throw_cells} cells)")

    if fracture.geometric_throw_cells == 0:
        return model

    throw = fracture.geometric_throw_cells
    displacement_mask = _create_displacement_mask(model.grid_shape, fracture)
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
    new_model = attrs.evolve(
        model,
        grid_shape=ctx.new_shape,
        thickness_grid=new_thickness,
        rock_properties=new_rock,
        fluid_properties=new_fluid,
    )
    logger.debug(f"Grid shape: {model.grid_shape} → {ctx.new_shape}")
    return new_model


@attrs.define(slots=True, frozen=True)
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
        if self.preserve_data:
            return self._displace_with_expansion(grid, property_name, property_type)
        return self._displace_without_expansion(grid, property_name, property_type)

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

        # Get smart default value for this property
        default_value = self.defaults.get_value(
            property_name=property_name,
            grid=grid,
            property_type=property_type,
        )

        # Create expanded grid filled with defaults
        new_grid = np.full((nx, ny, new_nz), fill_value=default_value, dtype=grid.dtype)

        # Identify upthrown and downthrown blocks
        upthrown_mask = ~self.displacement_mask  # Not displaced
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
        # Get smart default value
        default_value = self.defaults.get_value(
            property_name=property_name,
            grid=grid,
            property_type=property_type,
        )
        # Create new grid filled with defaults
        new_grid = np.full((nx, ny, nz), fill_value=default_value, dtype=grid.dtype)

        # Identify blocks
        upthrown_mask = ~self.displacement_mask
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

            # Top of downthrown block exposed - already filled with defaults

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

    # Check fracture index bounds
    if fracture.orientation == "x" and not (0 <= fracture.fracture_index < nx):
        errors.append(
            f"Fracture {fracture.id}: fracture_index {fracture.fracture_index} out of x-range [0, {nx - 1}]"
        )
    elif fracture.orientation == "y" and not (0 <= fracture.fracture_index < ny):
        errors.append(
            f"Fracture {fracture.id}: fracture_index {fracture.fracture_index} out of y-range [0, {ny - 1}]"
        )

    # Check intercept bounds
    if not (0 <= fracture.intercept < nz):
        errors.append(
            f"Fracture {fracture.id}: intercept {fracture.intercept} out of z-range [0, {nz - 1}]"
        )

    # Check geometric throw bounds
    if abs(fracture.geometric_throw_cells) >= nz:
        errors.append(
            f"Fracture {fracture.id}: geometric_throw_cells {fracture.geometric_throw_cells} too large for nz={nz}"
        )

    # Check mask shape if provided
    if fracture.mask is not None and fracture.mask.shape != grid_shape:
        errors.append(
            f"Fracture {fracture.id}: mask shape {fracture.mask.shape} != grid shape {grid_shape}"
        )

    return errors


def _create_displacement_mask(
    grid_shape: typing.Tuple[int, int, int], fracture: Fracture
) -> np.ndarray:
    """
    Create a boolean mask indicating which cells should be displaced by the fracture.

    This defines the "downthrown block" - cells that will move vertically.

    :param grid_shape: Shape of the reservoir grid (nx, ny, nz)
    :param fracture: Fracture configuration
    :return: 3D boolean array marking cells to be displaced
    """
    nx, ny, nz = grid_shape
    displacement_mask = np.zeros((nx, ny, nz), dtype=bool)

    # Determine which side is downthrown based on fracture orientation
    if fracture.orientation == "x":
        # Cells with x > fracture_index are downthrown
        if fracture.fracture_index < nx - 1:
            displacement_mask[fracture.fracture_index + 1 :, :, :] = True

    elif fracture.orientation == "y":
        # Cells with y > fracture_index are downthrown
        if fracture.fracture_index < ny - 1:
            displacement_mask[:, fracture.fracture_index + 1 :, :] = True

    return displacement_mask
