"""
Fault-application system for 3D reservoir models.
"""

import copy
import itertools
import logging
import typing
from typing import List, Optional, Tuple

import attrs
import numpy as np

from sim3D.statics import ReservoirModel
from sim3D.types import NDimension, ThreeDimensions

__all__ = ["Fault", "apply_fault", "apply_faults"]

logger = logging.getLogger(__name__)


MIN_TRANSMISSIBILITY_FACTOR = 1e-12
"""Minimum transmissibility scaling factor for numerical stability."""


@attrs.define(slots=True, frozen=True)
class Fault:
    """
    Configuration for applying faults to reservoir models.

    Defines the geometric and hydraulic properties of a fault,
    including its orientation, hydraulic properties, and modeling approach.
    """

    id: str
    """Unique identifier for the fault."""

    slope: float = 0.0
    """
    Slope of the fault plane: z = z0 + slope*(x - x0) for x-oriented faults,
    or z = z0 + slope*(y - y0) for y-oriented faults.
    A slope of 0.0 creates a vertical fault.
    """

    intercept: float = 0.0
    """
    Z-intercept (z0) of the fault plane at the reference position.
    For vertical faults, this represents the base z-level of the fault.
    """

    orientation: typing.Literal["x", "y"] = "x"
    """
    Orientation of the fault plane:
    - "x": fault plane cuts through x-direction (perpendicular to x-axis)
    - "y": fault plane cuts through y-direction (perpendicular to y-axis)
    """

    fault_index: int = 0
    """
    Nominal grid index where the fault intersects:
    - For x-oriented faults: x-index of the fault plane
    - For y-oriented faults: y-index of the fault plane
    """

    transmissibility_scale: float = 1e-3
    """
    Scaling factor for transmissibilities across the fault.
    - Values < 1.0 create sealing barriers (typical: 1e-3 to 1e-6)
    - Values > 1.0 create conductive zones (for fractured faults)
    - Must be > MIN_TRANSMISSIBILITY_FACTOR for numerical stability
    """

    fault_permeability: Optional[float] = None
    """
    Permeability value for fault zone cells (mD).
    If None, fault zone properties are not modified.
    """

    fault_porosity: Optional[float] = None
    """
    Porosity value for fault zone cells (fraction).
    If None, fault zone properties are not modified.
    """

    geometric_throw_cells: int = 0
    """
    Number of cells to displace the downthrown block in z-direction.
    Positive values create normal faults, negative values create reverse faults.
    """

    conductive: bool = False
    """
    If True, the fault acts as a high-permeability conduit.
    This automatically sets appropriate transmissibility_scale if not manually specified.
    """

    mask: Optional[np.typing.NDArray] = None
    """
    Optional 3D boolean mask defining fault geometry.
    If provided, overrides geometric fault plane calculation.
    Must match the reservoir model grid dimensions.
    """

    preserve_grid_data: bool = False
    """
    If True, expand grid dimensions to preserve all displaced data (Option 2).
    If False, use traditional displacement with data loss (Option 1, default).
    
    Example with 5-layer grid and 3-cell downthrow:
    
    preserve_grid_data=False (Option 1 - Default):

    ```markdown
    ┌─────────────────┬─────────────────┐
    │ BEFORE (5x5x5)  │ AFTER (5x5x5)   │
    │ z=4: [A A A A A]│ z=4: [A A ■ ■ ■]│ ← Fill with defaults
    │ z=3: [B B B B B]│ z=3: [B B ■ ■ ■]│ 
    │ z=2: [C C C C C]│ z=2: [C C ■ ■ ■]│
    │ z=1: [D D D D D]│ z=1: [D D A A A]│ ← A moved down
    │ z=0: [E E E E E]│ z=0: [E E B B B]│ ← C,D,E LOST! 💀
    └─────────────────┴─────────────────┘
    ```

    preserve_grid_data=True (Option 2 - Grid Expansion):

    ```markdown
    ┌─────────────────┬─────────────────┐
    │ BEFORE (5x5x5)  │ AFTER (5x5x8)   │
    │ z=4: [A A A A A]│ z=7: [A A ■ ■ ■]│ ← Fill exposed top
    │ z=3: [B B B B B]│ z=6: [B B ■ ■ ■]│
    │ z=2: [C C C C C]│ z=5: [C C ■ ■ ■]│
    │ z=1: [D D D D D]│ z=4: [D D A A A]│ ← A moved down 3
    │ z=0: [E E E E E]│ z=3: [E E B B B]│ ← B moved down 3
    │                 │ z=2: [■ ■ C C C]│ ← C moved down 3
    │                 │ z=1: [■ ■ D D D]│ ← D moved down 3
    │                 │ z=0: [■ ■ E E E]│ ← E moved down 3
    └─────────────────┴─────────────────┘
    ```

    Grid dimensions: 5x5x5 → 5x5x8 (expanded by throw amount)
    ALL data preserved ✓
    """

    def __attrs_post_init__(self) -> None:
        """Validate fault configuration parameters."""
        if self.transmissibility_scale < MIN_TRANSMISSIBILITY_FACTOR:
            object.__setattr__(
                self, "transmissibility_scale", MIN_TRANSMISSIBILITY_FACTOR
            )
            logger.warning(
                f"Fault {self.id}: transmissibility_scale clamped to {MIN_TRANSMISSIBILITY_FACTOR} "
                f"for numerical stability"
            )

        if self.conductive and self.transmissibility_scale < 1.0:
            # For conductive faults, increase transmissibility
            object.__setattr__(
                self, "transmissibility_scale", max(10.0, self.transmissibility_scale)
            )
            logger.info(
                f"Fault {self.id}: Set as conductive with transmissibility_scale = {self.transmissibility_scale}"
            )

        if self.fault_permeability is not None and self.fault_permeability < 0:
            raise ValueError(
                f"Fault {self.id}: fault_permeability must be non-negative"
            )

        if self.fault_porosity is not None and not (0 <= self.fault_porosity <= 1):
            raise ValueError(f"Fault {self.id}: fault_porosity must be between 0 and 1")


def apply_fault(
    model: ReservoirModel[NDimension], fault: Fault
) -> ReservoirModel[NDimension]:
    """
    Apply a single fault to a reservoir model.

    This function modifies transmissibilities, rock properties, and geometry
    based on the fault configuration. The input model is not modified;
    a new model instance is returned.

    :param model: Input reservoir model to modify
    :param fault: Fault configuration defining geometry and properties
    :return: New reservoir model with fault applied
    """
    logger.info(f"Applying fault '{fault.id}' to reservoir model")
    errors = validate_fault(fault, model.grid_shape)
    if errors:
        for error in errors:
            logger.error(error)
        raise ValueError(
            f"Fault {fault.id} configuration is invalid; see log for details"
        )

    new_model = copy.deepcopy(model)
    grid_shape = new_model.grid_shape
    if len(grid_shape) != 3:
        raise ValueError("Fault application requires 3D reservoir models")

    new_model = typing.cast(ReservoirModel[ThreeDimensions], new_model)
    # Generate or validate fault mask
    if fault.mask is not None:
        if fault.mask.shape != grid_shape:
            raise ValueError(
                f"Fault {fault.id}: mask shape {fault.mask.shape} "
                f"does not match grid shape {grid_shape}"
            )
        fault_mask = fault.mask.copy()
    else:
        fault_mask = _make_fault_mask(grid_shape, fault)

    # Apply fault effects in proper order to handle grid dimension changes
    # 1. Modify fault zone properties (before geometric displacement)
    if fault.fault_permeability is not None or fault.fault_porosity is not None:
        new_model = _apply_fault_zone_properties(new_model, fault_mask, fault)

    # 2. Scale transmissibilities across fault (before geometric displacement)
    new_model = _scale_transmissibility(new_model, fault_mask, fault)

    # 3. Apply geometric displacement (throw) to all grids if specified
    # This may change grid dimensions, so it should be done last
    if fault.geometric_throw_cells != 0:
        new_model = _apply_geometric_throw(new_model, fault_mask, fault)

    logger.info(f"Successfully applied fault '{fault.id}'")
    return typing.cast(ReservoirModel[NDimension], new_model)


def apply_faults(
    model: ReservoirModel[NDimension], *configs: Fault
) -> ReservoirModel[NDimension]:
    """
    Apply multiple faults to a reservoir model.

    Faults are applied sequentially in the order provided. Each fault
    is applied to the result of the previous fault application.

    :param model: Input reservoir model
    :param configs: Sequence of fault configurations
    :return: New reservoir model with all faults applied
    """
    logger.info(f"Applying {len(configs)} faults to reservoir model")

    faulted_model = model
    for fault in configs:
        faulted_model = apply_fault(faulted_model, fault)

    logger.info(f"Successfully applied all {len(configs)} faults")
    return faulted_model


def _make_fault_mask(grid_shape: Tuple[int, int, int], fault: Fault) -> np.ndarray:
    """
    Generate a 3D boolean mask defining the fault geometry.

    For inclined faults, the mask follows the equation:
    z = z0 + slope * (coord - coord0)
    where coord is x or y depending on orientation.

    :param grid_shape: Shape of the reservoir grid (nx, ny, nz)
    :param fault: Fault configuration
    :return: 3D boolean array marking fault cells
    """
    nx, ny, nz = grid_shape

    # Create coordinate grids
    x_coords = np.arange(nx)
    y_coords = np.arange(ny)
    z_coords = np.arange(nz)

    if fault.orientation == "x":
        # Fault cuts through x-direction
        Y, Z = np.meshgrid(y_coords, z_coords, indexing="ij")

        # Create mask: cells at or near the fault plane
        mask = np.zeros((nx, ny, nz), dtype=bool)

        # For vertical fault (slope = 0), mark cells at fault_index
        if abs(fault.slope) < 1e-6:
            if 0 <= fault.fault_index < nx:
                mask[fault.fault_index, :, :] = True
        else:
            # For inclined fault, mark cells intersected by the plane
            for i in range(nx):
                for j in range(ny):
                    fault_z_at_j = fault.intercept + fault.slope * (
                        j - fault.fault_index
                    )
                    fault_z_at_j = max(0, min(nz - 1, fault_z_at_j))

                    # Mark cells within 0.5 cells of the fault plane
                    z_low = max(0, int(fault_z_at_j - 0.5))
                    z_high = min(nz - 1, int(fault_z_at_j + 0.5))

                    if i == fault.fault_index:
                        for k in range(z_low, z_high + 1):
                            mask[i, j, k] = True

    elif fault.orientation == "y":
        # Fault cuts through y-direction
        X, Z = np.meshgrid(x_coords, z_coords, indexing="ij")

        mask = np.zeros((nx, ny, nz), dtype=bool)

        # For vertical fault (slope = 0), mark cells at fault_index
        if abs(fault.slope) < 1e-6:
            if 0 <= fault.fault_index < ny:
                mask[:, fault.fault_index, :] = True
        else:
            # For inclined fault, mark cells intersected by the plane
            for i in range(nx):
                for j in range(ny):
                    fault_z_at_i = fault.intercept + fault.slope * (
                        i - fault.fault_index
                    )
                    fault_z_at_i = max(0, min(nz - 1, fault_z_at_i))

                    # Mark cells within 0.5 cells of the fault plane
                    z_low = max(0, int(fault_z_at_i - 0.5))
                    z_high = min(nz - 1, int(fault_z_at_i + 0.5))

                    if j == fault.fault_index:
                        for k in range(z_low, z_high + 1):
                            mask[i, j, k] = True

    else:
        raise ValueError(f"Invalid fault orientation: {fault.orientation}")
    return mask


def _scale_transmissibility(
    model: ReservoirModel[ThreeDimensions],
    fault_mask: np.typing.NDArray,
    fault: Fault,
) -> ReservoirModel[ThreeDimensions]:
    """
    Scale transmissibilities across fault boundaries.

    This function identifies transmissibility connections that cross
    the fault and scales them by the transmissibility_scale factor.

    :param model: Reservoir model to modify
    :param fault_mask: 3D boolean array marking fault cells
    :param fault: Fault configuration
    :return: Modified reservoir model
    """
    logger.debug(f"Scaling transmissibilities for fault '{fault.id}'")

    # We'll modify permeability which affects transmissibility
    # Since we don't have direct access to transmissibility arrays,
    # as they are computed from permeability and geometry.
    nx, ny, nz = model.grid_shape

    # Identify fault boundaries and scale connections
    if fault.orientation == "x":
        # Scale x-direction transmissibilities
        for i, j, k in itertools.product(range(nx - 1), range(ny), range(nz)):
            # Check if connection crosses fault
            cell1_is_fault = fault_mask[i, j, k]
            cell2_is_fault = fault_mask[i + 1, j, k]

            if cell1_is_fault or cell2_is_fault:
                # This connection crosses or touches the fault
                # In a real implementation, scale transmissibility_x[i, j, k]

                # For now, we'll reduce permeability at fault interface
                if cell1_is_fault:
                    current_perm = model.rock_properties.absolute_permeability.x[
                        i, j, k
                    ]
                    new_perm = current_perm * fault.transmissibility_scale
                    model.rock_properties.absolute_permeability.x[i, j, k] = new_perm

                if cell2_is_fault:
                    current_perm = model.rock_properties.absolute_permeability.x[
                        i + 1, j, k
                    ]
                    new_perm = current_perm * fault.transmissibility_scale
                    model.rock_properties.absolute_permeability.x[i + 1, j, k] = (
                        new_perm
                    )

    elif fault.orientation == "y":
        # Scale y-direction transmissibilities
        for i, j, k in itertools.product(range(nx), range(ny - 1), range(nz)):
            # Check if connection crosses fault
            cell1_is_fault = fault_mask[i, j, k]
            cell2_is_fault = fault_mask[i, j + 1, k]

            if cell1_is_fault or cell2_is_fault:
                # This connection crosses or touches the fault
                if cell1_is_fault:
                    current_perm = model.rock_properties.absolute_permeability.y[
                        i, j, k
                    ]
                    new_perm = current_perm * fault.transmissibility_scale
                    model.rock_properties.absolute_permeability.y[i, j, k] = new_perm

                if cell2_is_fault:
                    current_perm = model.rock_properties.absolute_permeability.y[
                        i, j + 1, k
                    ]
                    new_perm = current_perm * fault.transmissibility_scale
                    model.rock_properties.absolute_permeability.y[i, j + 1, k] = (
                        new_perm
                    )

    # Always consider z-direction scaling for inclined faults
    for i, j, k in itertools.product(range(nx), range(ny), range(nz - 1)):
        cell1_is_fault = fault_mask[i, j, k]
        cell2_is_fault = fault_mask[i, j, k + 1]

        if cell1_is_fault or cell2_is_fault:
            # Scale z-direction connection
            if cell1_is_fault:
                current_perm = model.rock_properties.absolute_permeability.z[i, j, k]
                new_perm = current_perm * fault.transmissibility_scale
                model.rock_properties.absolute_permeability.z[i, j, k] = new_perm

            if cell2_is_fault:
                current_perm = model.rock_properties.absolute_permeability.z[
                    i, j, k + 1
                ]
                new_perm = current_perm * fault.transmissibility_scale
                model.rock_properties.absolute_permeability.z[i, j, k + 1] = new_perm

    return model


def _apply_fault_zone_properties(
    model: ReservoirModel[ThreeDimensions],
    fault_mask: np.typing.NDArray,
    fault: Fault,
) -> ReservoirModel[ThreeDimensions]:
    """
    Apply fault zone rock properties to cells within the fault.

    This function modifies permeability and/or porosity values
    for cells identified by the fault mask.

    :param model: Reservoir model to modify
    :param fault_mask: 3D boolean array marking fault cells
    :param fault: Fault configuration
    :return: Modified reservoir model
    """
    logger.debug(f"Applying fault zone properties for fault '{fault.id}'")

    if fault.fault_permeability is not None:
        # Apply fault permeability to x, y, and z directions
        model.rock_properties.absolute_permeability.x[fault_mask] = (
            fault.fault_permeability
        )
        if model.rock_properties.absolute_permeability.y.size > 0:
            model.rock_properties.absolute_permeability.y[fault_mask] = (
                fault.fault_permeability
            )

        if model.rock_properties.absolute_permeability.z.size > 0:
            model.rock_properties.absolute_permeability.z[fault_mask] = (
                fault.fault_permeability
            )

        logger.debug(f"Set fault zone permeability to {fault.fault_permeability} mD")

    if fault.fault_porosity is not None:
        model.rock_properties.porosity_grid[fault_mask] = fault.fault_porosity
        logger.debug(f"Set fault zone porosity to {fault.fault_porosity}")

    return model


def _apply_geometric_throw(
    model: ReservoirModel[ThreeDimensions],
    fault_mask: np.typing.NDArray,
    fault: Fault,
) -> ReservoirModel[ThreeDimensions]:
    """
    Apply geometric displacement (throw) to all property grids consistently.

    This function shifts all property grids in the "downthrown" block by the specified
    number of cells in the z-direction, modifying the grids in-place.

    :param model: Reservoir model to modify
    :param fault_mask: 3D boolean array marking fault cells
    :param fault: Fault configuration
    :return: Modified reservoir model
    """
    logger.debug(
        f"Applying geometric throw of {fault.geometric_throw_cells} cells for fault '{fault.id}'"
    )

    if fault.geometric_throw_cells == 0:
        return model

    throw = fault.geometric_throw_cells
    displacement_mask = _create_displacement_mask(model.grid_shape, fault)

    # Apply displacement to all property grids consistently
    logger.debug(
        f"Applying {throw}-cell displacement to all property grids (preserve_data={fault.preserve_grid_data})"
    )

    # Apply to geometric grids
    displaced_thickness_grid = _displace_grid(
        grid=model.thickness_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )

    # Apply to rock properties
    rock_props = model.rock_properties
    displaced_porosity_grid = _displace_grid(
        grid=rock_props.porosity_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_net_to_gross_ratio_grid = _displace_grid(
        grid=rock_props.net_to_gross_ratio_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_irreducible_water_saturation_grid = _displace_grid(
        grid=rock_props.irreducible_water_saturation_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_residual_oil_saturation_water_grid = _displace_grid(
        grid=rock_props.residual_oil_saturation_water_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_residual_oil_saturation_gas_grid = _displace_grid(
        grid=rock_props.residual_oil_saturation_gas_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_residual_gas_saturation_grid = _displace_grid(
        grid=rock_props.residual_gas_saturation_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )

    # Apply to permeability grids
    displaced_permeability_x = _displace_grid(
        grid=rock_props.absolute_permeability.x,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_permeability_y = _displace_grid(
        grid=rock_props.absolute_permeability.y,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_permeability_z = _displace_grid(
        grid=rock_props.absolute_permeability.z,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )

    # Update rock properties with displaced grids
    new_absolute_permeability = attrs.evolve(
        rock_props.absolute_permeability,
        x=displaced_permeability_x,
        y=displaced_permeability_y
        if displaced_permeability_y is not None
        else rock_props.absolute_permeability.y,
        z=displaced_permeability_z
        if displaced_permeability_z is not None
        else rock_props.absolute_permeability.z,
    )
    new_rock_properties = attrs.evolve(
        rock_props,
        porosity_grid=displaced_porosity_grid,
        net_to_gross_ratio_grid=displaced_net_to_gross_ratio_grid,
        irreducible_water_saturation_grid=displaced_irreducible_water_saturation_grid,
        residual_oil_saturation_water_grid=displaced_residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=displaced_residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=displaced_residual_gas_saturation_grid,
        absolute_permeability=new_absolute_permeability,
    )

    # Apply to fluid properties
    fluid_props = model.fluid_properties
    displaced_pressure_grid = _displace_grid(
        grid=fluid_props.pressure_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_temperature_grid = _displace_grid(
        grid=fluid_props.temperature_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_oil_bubble_point_pressure_grid = _displace_grid(
        grid=fluid_props.oil_bubble_point_pressure_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_oil_saturation_grid = _displace_grid(
        grid=fluid_props.oil_saturation_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_oil_viscosity_grid = _displace_grid(
        grid=fluid_props.oil_viscosity_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_oil_compressibility_grid = _displace_grid(
        grid=fluid_props.oil_compressibility_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_oil_specific_gravity_grid = _displace_grid(
        grid=fluid_props.oil_specific_gravity_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_oil_api_gravity_grid = _displace_grid(
        grid=fluid_props.oil_api_gravity_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_oil_density_grid = _displace_grid(
        grid=fluid_props.oil_density_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_water_bubble_point_pressure_grid = _displace_grid(
        grid=fluid_props.water_bubble_point_pressure_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_water_saturation_grid = _displace_grid(
        grid=fluid_props.water_saturation_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_water_viscosity_grid = _displace_grid(
        grid=fluid_props.water_viscosity_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_water_compressibility_grid = _displace_grid(
        grid=fluid_props.water_compressibility_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_water_density_grid = _displace_grid(
        grid=fluid_props.water_density_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_gas_saturation_grid = _displace_grid(
        grid=fluid_props.gas_saturation_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_gas_viscosity_grid = _displace_grid(
        grid=fluid_props.gas_viscosity_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_gas_compressibility_grid = _displace_grid(
        grid=fluid_props.gas_compressibility_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_gas_gravity_grid = _displace_grid(
        grid=fluid_props.gas_gravity_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_gas_molecular_weight_grid = _displace_grid(
        grid=fluid_props.gas_molecular_weight_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_gas_density_grid = _displace_grid(
        grid=fluid_props.gas_density_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_solution_gas_to_oil_ratio_grid = _displace_grid(
        grid=fluid_props.solution_gas_to_oil_ratio_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_gas_solubility_in_water_grid = _displace_grid(
        grid=fluid_props.gas_solubility_in_water_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_oil_formation_volume_factor_grid = _displace_grid(
        grid=fluid_props.oil_formation_volume_factor_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_gas_formation_volume_factor_grid = _displace_grid(
        grid=fluid_props.gas_formation_volume_factor_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_water_formation_volume_factor_grid = _displace_grid(
        grid=fluid_props.water_formation_volume_factor_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )
    displaced_water_salinity_grid = _displace_grid(
        grid=fluid_props.water_salinity_grid,
        displacement_mask=displacement_mask,
        throw=throw,
        preserve_grid_data=fault.preserve_grid_data,
    )

    # Update fluid properties with all displaced grids
    new_fluid_properties = attrs.evolve(
        fluid_props,
        pressure_grid=displaced_pressure_grid,
        temperature_grid=displaced_temperature_grid,
        oil_bubble_point_pressure_grid=displaced_oil_bubble_point_pressure_grid,
        oil_saturation_grid=displaced_oil_saturation_grid,
        oil_viscosity_grid=displaced_oil_viscosity_grid,
        oil_compressibility_grid=displaced_oil_compressibility_grid,
        oil_specific_gravity_grid=displaced_oil_specific_gravity_grid,
        oil_api_gravity_grid=displaced_oil_api_gravity_grid,
        oil_density_grid=displaced_oil_density_grid,
        water_bubble_point_pressure_grid=displaced_water_bubble_point_pressure_grid,
        water_saturation_grid=displaced_water_saturation_grid,
        water_viscosity_grid=displaced_water_viscosity_grid,
        water_compressibility_grid=displaced_water_compressibility_grid,
        water_density_grid=displaced_water_density_grid,
        gas_saturation_grid=displaced_gas_saturation_grid,
        gas_viscosity_grid=displaced_gas_viscosity_grid,
        gas_compressibility_grid=displaced_gas_compressibility_grid,
        gas_gravity_grid=displaced_gas_gravity_grid,
        gas_molecular_weight_grid=displaced_gas_molecular_weight_grid,
        gas_density_grid=displaced_gas_density_grid,
        solution_gas_to_oil_ratio_grid=displaced_solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=displaced_gas_solubility_in_water_grid,
        oil_formation_volume_factor_grid=displaced_oil_formation_volume_factor_grid,
        gas_formation_volume_factor_grid=displaced_gas_formation_volume_factor_grid,
        water_formation_volume_factor_grid=displaced_water_formation_volume_factor_grid,
        water_salinity_grid=displaced_water_salinity_grid,
    )

    # Update the model with new thickness grid, rock and fluid properties
    # Also update grid_shape to match the new dimensions if grid was expanded
    new_grid_shape = displaced_thickness_grid.shape
    model = attrs.evolve(
        model,
        grid_shape=new_grid_shape,
        thickness_grid=displaced_thickness_grid,
        rock_properties=new_rock_properties,
        fluid_properties=new_fluid_properties,
    )
    logger.debug(f"Successfully applied geometric displacement for fault '{fault.id}'")
    return model


def validate_fault(fault: Fault, grid_shape: Tuple[int, ...]) -> List[str]:
    """
    Validate fault configuration against grid dimensions.

    :param fault: Fault configuration to validate
    :param grid_shape: Shape of the reservoir grid (nx, ny, nz)
    :return: List of validation error messages (empty if valid)
    """
    errors = []
    nx, ny, nz = grid_shape

    # Check fault index bounds
    if fault.orientation == "x" and not (0 <= fault.fault_index < nx):
        errors.append(
            f"Fault {fault.id}: fault_index {fault.fault_index} out of x-range [0, {nx - 1}]"
        )
    elif fault.orientation == "y" and not (0 <= fault.fault_index < ny):
        errors.append(
            f"Fault {fault.id}: fault_index {fault.fault_index} out of y-range [0, {ny - 1}]"
        )

    # Check intercept bounds
    if not (0 <= fault.intercept < nz):
        errors.append(
            f"Fault {fault.id}: intercept {fault.intercept} out of z-range [0, {nz - 1}]"
        )

    # Check geometric throw bounds
    if abs(fault.geometric_throw_cells) >= nz:
        errors.append(
            f"Fault {fault.id}: geometric_throw_cells {fault.geometric_throw_cells} too large for nz={nz}"
        )

    # Check mask shape if provided
    if fault.mask is not None and fault.mask.shape != grid_shape:
        errors.append(
            f"Fault {fault.id}: mask shape {fault.mask.shape} != grid shape {grid_shape}"
        )
    return errors


def _create_displacement_mask(
    grid_shape: Tuple[int, int, int], fault: Fault
) -> np.ndarray:
    """
    Create a boolean mask indicating which cells should be displaced by the fault.

    Determines displacement pattern based on fault orientation and throw direction.

    :param grid_shape: Shape of the reservoir grid (nx, ny, nz)
    :param fault: Fault configuration
    :return: 3D boolean array marking cells to be displaced
    """
    nx, ny, nz = grid_shape
    displacement_mask = np.zeros((nx, ny, nz), dtype=bool)

    # Determine which side is downthrown based on fault orientation
    if fault.orientation == "x":
        # Cells with x > fault_index are downthrown
        if fault.fault_index < nx - 1:
            displacement_mask[fault.fault_index + 1 :, :, :] = True
    elif fault.orientation == "y":
        # Cells with y > fault_index are downthrown
        if fault.fault_index < ny - 1:
            displacement_mask[:, fault.fault_index + 1 :, :] = True
    return displacement_mask


def _displace_grid(
    grid: np.typing.NDArray,
    displacement_mask: np.typing.NDArray,
    throw: int,
    preserve_grid_data: bool = False,
) -> np.typing.NDArray:
    """
    Apply vertical displacement to a single grid array.

    This function modifies the grid in-place by shifting values according to
    the displacement pattern. Can either use traditional displacement (with data loss)
    or grid expansion (preserving all data).

    :param grid: 3D numpy array to displace
    :param displacement_mask: Boolean mask indicating cells to displace
    :param throw: Number of cells to displace (positive = down, negative = up)
    :param preserve_grid_data: If True, expand grid to preserve all data (Option 2).
                               If False, use traditional displacement with data loss (Option 1).
    :return: Displaced grid array (possibly expanded)
    """
    if preserve_grid_data:
        return _displace_grid_with_expansion(grid, displacement_mask, throw)
    return _displace_grid_without_expansion(grid, displacement_mask, throw)


def _displace_grid_without_expansion(
    grid: np.typing.NDArray, displacement_mask: np.typing.NDArray, throw: int
) -> np.typing.NDArray:
    """
    Traditional displacement method with fixed grid size and potential data loss.

    This is the original implementation (Option 1).
    """
    nx, ny, nz = grid.shape
    new_grid = grid.copy()
    default_value = _get_grid_default_value(grid)

    if throw > 0:
        # Normal fault: downthrown block moves down
        for i, j in itertools.product(range(nx), range(ny)):
            if displacement_mask[i, j, :].any():
                # Process only cells that can fit after displacement
                for k in range(nz - throw):
                    if displacement_mask[i, j, k]:
                        if k + throw < nz:
                            new_grid[i, j, k + throw] = grid[i, j, k]
                            # Fill vacated space with default
                            new_grid[i, j, k] = default_value
    else:
        # Reverse fault: downthrown block moves up
        abs_throw = abs(throw)
        for i, j in itertools.product(range(nx), range(ny)):
            if displacement_mask[i, j, :].any():
                for k in range(abs_throw, nz):
                    if displacement_mask[i, j, k]:
                        if k - abs_throw >= 0:
                            new_grid[i, j, k - abs_throw] = grid[i, j, k]
                            # Fill vacated space with default
                            new_grid[i, j, k] = default_value

    return new_grid


def _displace_grid_with_expansion(
    grid: np.typing.NDArray, displacement_mask: np.typing.NDArray, throw: int
) -> np.typing.NDArray:
    """
    Displacement with grid expansion to preserve all data (Option 2).

    This method expands the grid dimensions to accommodate displaced cells,
    ensuring no geological data is lost during fault displacement.

    :param grid: Original 3D grid array
    :param displacement_mask: Boolean mask indicating cells to displace
    :param throw: Number of cells to displace (positive = down, negative = up)
    :return: Expanded grid with all data preserved
    """
    nx, ny, nz = grid.shape
    abs_throw = abs(throw)
    default_value = _get_grid_default_value(grid)

    # Calculate new grid dimensions
    new_nz = nz + abs_throw

    # Create expanded grid filled with default values
    expanded_grid = np.full((nx, ny, new_nz), default_value, dtype=grid.dtype)

    if throw > 0:
        # Normal fault: downthrown block moves down
        # Structure: [original_upthrown | displaced_downthrown | defaults_at_bottom]

        # First, copy all original data to the top portion
        expanded_grid[:, :, :nz] = grid
        # Then apply displacement to downthrown cells
        for i, j in itertools.product(range(nx), range(ny)):
            if displacement_mask[i, j, :].any():
                # Displace each cell in the downthrown block
                for k in range(nz):
                    if displacement_mask[i, j, k]:
                        # Move cell down by throw amount
                        target_k = k + throw
                        expanded_grid[i, j, target_k] = grid[i, j, k]

                        # Fill the vacated space with default (newly exposed area)
                        expanded_grid[i, j, k] = default_value

    else:
        # Reverse fault: downthrown block moves up
        # Structure: [defaults_at_top | displaced_downthrown | original_upthrown]

        # First, copy all original data to the bottom portion
        expanded_grid[:, :, abs_throw:] = grid
        # Fill the top portion with defaults (newly exposed area)
        expanded_grid[:, :, :abs_throw] = default_value
        # Then apply upward displacement to downthrown cells
        for i, j in itertools.product(range(nx), range(ny)):
            if displacement_mask[i, j, :].any():
                # Process from bottom to top to avoid overwriting
                for k in range(nz - 1, -1, -1):
                    if displacement_mask[i, j, k]:
                        # Move cell up by abs_throw amount
                        target_k = k  # Position in expanded grid (already shifted)
                        source_k = (
                            k + abs_throw
                        )  # Position of original data in expanded grid
                        # Move the cell up
                        expanded_grid[i, j, target_k] = expanded_grid[i, j, source_k]
                        # Fill the vacated space at bottom with default
                        expanded_grid[i, j, source_k] = default_value

    return expanded_grid


def _get_grid_default_value(grid: np.typing.NDArray) -> float:
    """
    Determine an appropriate default value for filling gaps in displaced grids.

    :param grid: Grid array to analyze
    :return: Appropriate default value based on grid statistics
    """
    # Use mean value for most properties, with some bounds checking
    non_zero_values = grid[grid > 0]
    if len(non_zero_values) > 0:
        default_value = float(np.mean(non_zero_values))
        # For properties that should be small/positive, use a conservative minimum
        return max(default_value, 1e-6)
    # If no positive values, return a small positive number
    return 1e-6
