import typing
import warnings

import numpy as np

from bores.boundary_conditions import BoundaryConditions, GridBoundaryCondition
from bores.constants import c
from bores.errors import ValidationError
from bores.fractures import Fracture, apply_fractures
from bores.grids.base import build_uniform_grid
from bores.grids.pvt import (
    build_fluid_viscosity_grid,
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_molecular_weight_grid,
    build_gas_solubility_in_water_grid,
    build_live_oil_density_grid,
    build_oil_api_gravity_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_formation_volume_factor_grid,
    build_solution_gas_to_oil_ratio_grid,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockFluidProperties,
    RockPermeability,
    RockProperties,
    SaturationHistory,
)
from bores.pvt.arrays import compute_gas_to_oil_ratio_standing
from bores.pvt.core import (
    compute_gas_gravity,
    validate_input_pressure,
    validate_input_temperature,
)
from bores.pvt.tables import PVTTables
from bores.types import (
    CapillaryPressureTable,
    NDimension,
    NDimensionalGrid,
    RelativePermeabilityTable,
    WellLocation,
)
from bores.wells import (
    InjectedFluid,
    InjectionWell,
    ProducedFluid,
    ProductionWell,
    WellControl,
    Wells,
)

__all__ = [
    "reservoir_model",
    "injection_well",
    "production_well",
    "wells_",
    "validate_saturation_grids",
]


def validate_saturation_grids(
    oil_saturation_grid: NDimensionalGrid[NDimension],
    water_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    porosity_grid: NDimensionalGrid[NDimension],
    normalize: bool = True,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Validate and optionally normalize saturation grids for reservoir modeling.

    This function performs comprehensive validation on saturation grids, considering
    only active cells (cells with finite and non-zero porosity). It checks:
    - Saturation sum equals 1.0 (with optional normalization)
    - Saturation values are within valid ranges [0, 1)
    - Residual saturations are physically valid
    - Residual oil saturation consistency (warning if sor_gas > sor_water)

    :param oil_saturation_grid: Oil saturation grid (fraction).
    :param water_saturation_grid: Water saturation grid (fraction).
    :param gas_saturation_grid: Gas saturation grid (fraction).
    :param residual_oil_saturation_water_grid: Residual oil saturation during water flooding grid (fraction).
    :param residual_oil_saturation_gas_grid: Residual oil saturation during gas flooding grid (fraction).
    :param residual_gas_saturation_grid: Residual gas saturation grid (fraction).
    :param connate_water_saturation_grid: Connate water saturation grid (fraction).
    :param irreducible_water_saturation_grid: Irreducible water saturation grid (fraction).
    :param porosity_grid: Porosity grid (fraction) used to identify active cells.
    :param normalize: If True, normalizes saturations to sum to 1.0 in active cells (default: True).
    :return: Tuple of (oil_saturation_grid, water_saturation_grid, gas_saturation_grid),
             potentially normalized if normalize=True.
    :raises ValueError: If saturation constraints are violated in active cells.
    :raises UserWarning: If residual oil saturation for gas is greater than for water.
    """
    # Identify active cells (finite and non-zero porosity)
    active_cells = np.isfinite(porosity_grid) & (porosity_grid > 0)

    # Check if saturations sum to 1.0 in active cells
    total_saturation = oil_saturation_grid + water_saturation_grid + gas_saturation_grid
    if not np.all(np.isclose(total_saturation[active_cells], 1.0)):
        if normalize:
            warnings.warn(
                "The sum of oil, water, and gas saturations does not equal 1 everywhere in active cells. "
                "Adjusting saturations to ensure they sum to 1.",
                UserWarning,
            )
            # Avoid division by zero
            total_saturation_safe = np.where(total_saturation == 0, 1, total_saturation)
            oil_saturation_grid = oil_saturation_grid / total_saturation_safe  # type: ignore
            water_saturation_grid = water_saturation_grid / total_saturation_safe  # type: ignore
            gas_saturation_grid = gas_saturation_grid / total_saturation_safe  # type: ignore
        else:
            raise ValidationError(
                "The sum of oil, water, and gas saturations does not equal 1 everywhere in active cells."
            )

    # Validate saturation ranges in active cells
    if np.any(
        (oil_saturation_grid[active_cells] < 0)
        | (oil_saturation_grid[active_cells] >= 1)
    ):
        raise ValidationError(
            "Oil saturation grid values must be in the range [0, 1) in active cells."
        )

    if np.any(
        (water_saturation_grid[active_cells] < 0)
        | (water_saturation_grid[active_cells] >= 1)
    ):
        raise ValidationError(
            "Water saturation grid values must be in the range [0, 1) in active cells."
        )

    if np.any(
        (gas_saturation_grid[active_cells] < 0)
        | (gas_saturation_grid[active_cells] >= 1)
    ):
        raise ValidationError(
            "Gas saturation grid values must be in the range [0, 1) in active cells."
        )

    # Check if residual oil saturation for gas is greater than for water
    if np.any(
        residual_oil_saturation_gas_grid[active_cells]
        > residual_oil_saturation_water_grid[active_cells]
    ):
        warnings.warn(
            "Residual oil saturation during gas flooding (Sor_g) is greater than residual oil "
            "saturation during water flooding (Sor_w) in some active cells. This may indicate "
            "that gas is less efficient at displacing oil than water, which is unusual. "
            "Please verify your residual saturation values.",
            UserWarning,
        )

    return oil_saturation_grid, water_saturation_grid, gas_saturation_grid


def reservoir_model(
    grid_shape: NDimension,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[NDimension],
    pressure_grid: NDimensionalGrid[NDimension],
    rock_compressibility: float,
    absolute_permeability: RockPermeability[NDimension],
    porosity_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    water_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    oil_saturation_grid: NDimensionalGrid[NDimension],
    oil_viscosity_grid: NDimensionalGrid[NDimension],
    oil_compressibility_grid: NDimensionalGrid[NDimension],
    oil_bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension],
    connate_water_saturation_grid: NDimensionalGrid[NDimension],
    relative_permeability_table: RelativePermeabilityTable,
    capillary_pressure_table: CapillaryPressureTable,
    oil_specific_gravity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_gravity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_viscosity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_compressibility_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    gas_molecular_weight_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_density_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_viscosity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_density_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_bubble_point_pressure_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    solution_gas_to_oil_ratio_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    gas_solubility_in_water_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    oil_formation_volume_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    gas_formation_volume_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    water_formation_volume_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    net_to_gross_ratio_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_salinity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    solvent_concentration_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    oil_effective_viscosity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    oil_effective_density_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    boundary_conditions: typing.Optional[BoundaryConditions] = None,
    saturation_history: typing.Optional[SaturationHistory[NDimension]] = None,
    fractures: typing.Optional[typing.Iterable[Fracture]] = None,
    dip_angle: float = 0.0,
    dip_azimuth: float = 0.0,
    reservoir_gas: typing.Optional[str] = None,
    pvt_tables: typing.Optional[PVTTables] = None,
) -> ReservoirModel[NDimension]:
    """
    Constructs a N-Dimensional reservoir model with given rock and fluid properties.

    Note:

    - If both `oil_bubble_point_pressure_grid` and `solution_gas_to_oil_ratio_grid` are omitted,
      the function attempts to estimate these based on pressure, temperature, and oil properties,
      but this may lead to less accurate results.
    - Saturation grids will be adjusted internally to ensure saturation sums to 1.
    - All fluid property grids are estimated from pressure and temperature if not provided.
    - Provide consistent and physically realistic input grids to ensure stable simulations.

    :param grid_shape: Number of grid cells (rows, columns).
    :param cell_dimension: Physical size of each cell (dx, dy) in feets.
    :param thickness_grid: Reservoir cells' thickness grid (ft).
    :param pressure_grid: Reservoir cells' pressure grid (psi).
    :param rock_compressibility: Rock compressibility in psi⁻¹.
    :param absolute_permeability_grid: Absolute permeability grid (mD).
    :param porosity_grid: Porosity grid (fraction).
    :param temperature_grid: Reservoir temperature grid (°F).
    :param oil_saturation_grid: Oil saturation grid (fraction).
    :param oil_viscosity_grid: Oil viscosity grid (cP).
    :param oil_compressibility_grid: Oil compressibility grid (psi⁻¹).
    :param oil_specific_gravity_grid: Oil specific gravity grid (dimensionless).
    :param oil_bubble_point_pressure_grid: Oil bubble point pressure grid (psi).
    :param gas_gravity_grid: Gas gravity grid (dimensionless), optional.
    :param residual_oil_saturation_water_grid: Residual oil saturation during water flooding grid (fraction).
    :param residual_oil_saturation_gas_grid: Residual oil saturation during gas flooding grid (fraction).
    :param gas_saturation_grid: Gas saturation grid (fraction), optional.
    :param gas_viscosity_grid: Gas viscosity grid (cP), optional.
    :param gas_compressibility_grid: Gas compressibility grid (psi⁻¹), optional.
    :param gas_compressibility_factor_grid: Gas compressibility factor grid (dimensionless), optional.
    :param gas_density_grid: Gas density grid (lb/ft³), optional.
    :param residual_gas_saturation_grid: Residual gas saturation grid (fraction), optional.
    :param water_saturation_grid: Water saturation grid (fraction), optional.
    :param water_viscosity_grid: Water viscosity grid (cP), optional.
    :param water_compressibility_grid: Water compressibility grid (psi⁻¹), optional.
    :param water_density_grid: Water density grid (lb/ft³), optional.
    :param water_bubble_point_pressure_grid: Water bubble point pressure grid (psi), optional.
    :param connate_water_saturation_grid: Connate water saturation grid (fraction).
    :param irreducible_water_saturation_grid: Irreducible water saturation grid (fraction).
    :param solution_gas_to_oil_ratio_grid: Solution gas to oil ratio grid (scf/bbl), optional.
    :param gas_solubility_in_water_grid: Gas solubility in water grid (scf/bbl), optional.
    :param oil_formation_volume_factor_grid: Oil formation volume factor grid (bbl/scf), optional.
    :param gas_formation_volume_factor_grid: Gas formation volume factor grid (bbl/scf), optional.
    :param water_formation_volume_factor_grid: Water formation volume factor grid (bbl/scf), optional.
    :param net_to_gross_ratio_grid: Net-to-gross ratio grid (fraction), optional.
    :param water_salinity_grid: Water salinity grid (ppm), optional.
    :param solvent_concentration_grid: Solvent concentration in oil phase (0=pure oil, 1=pure solvent), optional.
    :param oil_effective_viscosity_grid: Effective oil-solvent mixture viscosity using miscible model (e.g Todd Longstaff) (cP), optional.
    :param boundary_conditions: Boundary conditions for the model, optional. Defaults to no-flow conditions.
    :param fractures: Iterable of fractures to be applied to the reservoir model, optional.
    :param reservoir_gas: Name of the reservoir gas, defaults to `RESERVOIR_GAS_NAME`. Can also be the name of the gas injected into the reservoir.
    :param pvt_tables: PVT tables for fluid properties, optional.
    :return: The constructed N-Dimensional reservoir model with fluid and rock properties.
    """
    if not 1 <= len(grid_shape) <= 3:
        raise ValidationError(
            "`grid_shape` must be a tuple of one to three integers (rows, columns, [depth])."
        )
    if len(cell_dimension) < 2:
        raise ValidationError(
            "`cell_dimension` must be a tuple of two floats (cell width, cell height)."
        )

    validate_input_pressure(pressure_grid)
    validate_input_temperature(temperature_grid)

    reservoir_gas = typing.cast(str, reservoir_gas or c.RESERVOIR_GAS_NAME)

    if water_salinity_grid is None:
        water_salinity_grid = build_uniform_grid(
            grid_shape=grid_shape,
            value=c.DEFAULT_WATER_SALINITY_PPM,
        )

    if net_to_gross_ratio_grid is None:
        # Assume uniform net-to-gross ratio of 1.0 (fully net)
        net_to_gross_ratio_grid = build_uniform_grid(
            grid_shape=grid_shape,
            value=1.0,
        )

    # Validate and normalize saturation grids
    oil_saturation_grid, water_saturation_grid, gas_saturation_grid = (
        validate_saturation_grids(
            oil_saturation_grid=oil_saturation_grid,
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            porosity_grid=porosity_grid,
            normalize=True,
        )
    )

    # Viscosity Grids
    if gas_viscosity_grid is None:
        if pvt_tables is not None:
            gas_viscosity_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.gas_viscosity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_viscosity_grid is None:
            gas_viscosity_grid = build_fluid_viscosity_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                fluid=reservoir_gas,
            )

    if gas_gravity_grid is None:
        if pvt_tables is not None:
            gas_gravity_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.gas_gravity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_gravity_grid is None:
            gas_gravity = compute_gas_gravity(gas=reservoir_gas)
            gas_gravity_grid = build_uniform_grid(
                grid_shape=grid_shape,
                value=gas_gravity,
            )

    if water_viscosity_grid is None:
        if pvt_tables is not None:
            water_viscosity_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.water_viscosity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_viscosity_grid is None:
            water_viscosity_grid = build_water_viscosity_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                salinity_grid=water_salinity_grid,
            )

    if gas_compressibility_factor_grid is None:
        if pvt_tables is not None:
            gas_compressibility_factor_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.gas_compressibility_factor(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_compressibility_factor_grid is None:
            gas_compressibility_factor_grid = build_gas_compressibility_factor_grid(
                gas_gravity_grid=gas_gravity_grid,
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
            )

    # Compressibility Grids
    if gas_compressibility_grid is None:
        if pvt_tables is not None:
            gas_compressibility_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.gas_compressibility(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_compressibility_grid is None:
            gas_compressibility_grid = build_gas_compressibility_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                gas_gravity_grid=gas_gravity_grid,
                gas_compressibility_factor_grid=gas_compressibility_factor_grid,
            )

    if gas_molecular_weight_grid is None:
        if pvt_tables is not None:
            gas_molecular_weight_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.gas_molecular_weight(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_molecular_weight_grid is None:
            gas_molecular_weight_grid = build_gas_molecular_weight_grid(
                gas_gravity_grid=gas_gravity_grid  # type: ignore
            )

    # Density Grids
    if gas_density_grid is None:
        if pvt_tables is not None:
            gas_density_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.gas_density(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_density_grid is None:
            gas_density_grid = build_gas_density_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                gas_gravity_grid=gas_gravity_grid,  # type: ignore
                gas_compressibility_factor_grid=gas_compressibility_factor_grid,  # type: ignore
            )

    # Do not use PVT tables for oil api gravity and live oil density as these are
    # direct functions of oil specific gravity
    if oil_specific_gravity_grid is None:
        if pvt_tables is None:
            raise ValidationError(
                "`oil_specific_gravity_grid` must be provided if `pvt_tables` is not used."
            )
        else:
            oil_specific_gravity_grid = typing.cast(
                NDimensionalGrid[NDimension],
                pvt_tables.oil_specific_gravity(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )

    oil_api_gravity_grid = build_oil_api_gravity_grid(oil_specific_gravity_grid)
    if (
        solution_gas_to_oil_ratio_grid is None
        and oil_bubble_point_pressure_grid is not None
    ):
        # User provided Pb but not Rs, use correlations to compute Rs from Pb
        # Note: We use correlations here even if `pvt_tables` is provided because
        # the user explicitly provided Pb and we need to honor that value
        solution_gas_to_oil_ratio_grid = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            gas_gravity_grid=gas_gravity_grid,
            oil_api_gravity_grid=oil_api_gravity_grid,
        )
    elif (
        solution_gas_to_oil_ratio_grid is not None
        and oil_bubble_point_pressure_grid is None
    ):
        # When Rs is provided, try to get Pb from `pvt_tables` first, then fall back to correlations
        if pvt_tables is not None:
            oil_bubble_point_pressure_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.oil_bubble_point_pressure(
                    temperature=temperature_grid,
                    solution_gor=solution_gas_to_oil_ratio_grid,
                ),
            )
        if oil_bubble_point_pressure_grid is None:
            oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
                gas_gravity_grid=gas_gravity_grid,
                oil_api_gravity_grid=oil_api_gravity_grid,
                temperature_grid=temperature_grid,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
            )
    elif (
        solution_gas_to_oil_ratio_grid is None
        and oil_bubble_point_pressure_grid is None
    ):
        # Neither was provided so we estimate both
        warnings.warn(
            "Both `oil_bubble_point_pressure_grid` and `solution_gas_to_oil_ratio_grid` are not provided. "
            "Attempting to estimate the bubble point pressure and GOR. If estimation fails, "
            "please provide at least one of them. Note, estimating the bubble point pressure "
            "and GOR may not yield accurate results.",
            UserWarning,
        )

        # Try to use `pvt_tables` first to compute Rs and Pb, fall back to correlations
        if pvt_tables is not None:
            solution_gas_to_oil_ratio_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.solution_gas_to_oil_ratio(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    solution_gor=None,  # Will use 1D table if available
                ),
            )
        if solution_gas_to_oil_ratio_grid is None:
            # Use traditional correlations for estimation
            # Try to estimate the GOR and then calculate the bubble point pressure from that
            # As either GOR or bubble point is needed to build the model
            estimated_gor_grid = compute_gas_to_oil_ratio_standing(
                pressure=pressure_grid,
                oil_api_gravity=oil_api_gravity_grid,
                gas_gravity=gas_gravity_grid,
            )
            solution_gas_to_oil_ratio_grid = estimated_gor_grid

        # Now get Pb from Rs, trying `pvt_tables` first
        if pvt_tables is not None:
            oil_bubble_point_pressure_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.oil_bubble_point_pressure(
                    temperature=temperature_grid,
                    solution_gor=solution_gas_to_oil_ratio_grid,
                ),
            )
        if oil_bubble_point_pressure_grid is None:
            oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
                temperature_grid=temperature_grid,
                gas_gravity_grid=gas_gravity_grid,
                oil_api_gravity_grid=oil_api_gravity_grid,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
            )

    solution_gas_to_oil_ratio_grid = typing.cast(
        NDimensionalGrid[NDimension], solution_gas_to_oil_ratio_grid
    )

    if gas_solubility_in_water_grid is None:
        if pvt_tables is not None:
            gas_solubility_in_water_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.gas_solubility_in_water(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if gas_solubility_in_water_grid is None:
            gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                salinity_grid=water_salinity_grid,
                gas=reservoir_gas,
            )

    # Formation Volume Factor Grids
    if oil_formation_volume_factor_grid is None:
        if pvt_tables is not None:
            oil_formation_volume_factor_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.oil_formation_volume_factor(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    solution_gor=solution_gas_to_oil_ratio_grid,
                ),
            )
        if oil_formation_volume_factor_grid is None:
            oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
                oil_specific_gravity_grid=oil_specific_gravity_grid,
                gas_gravity_grid=gas_gravity_grid,
                solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
                oil_compressibility_grid=oil_compressibility_grid,
            )

    if gas_formation_volume_factor_grid is None:
        if pvt_tables is not None:
            gas_formation_volume_factor_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.gas_formation_volume_factor(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                ),
            )
        if gas_formation_volume_factor_grid is None:
            gas_formation_volume_factor_grid = build_gas_formation_volume_factor_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                gas_compressibility_factor_grid=gas_compressibility_factor_grid,  # type: ignore
            )

    gas_free_water_formation_volume_factor_grid = (
        build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=pressure_grid, temperature_grid=temperature_grid
        )
    )
    if water_density_grid is None:
        if pvt_tables is not None:
            water_density_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.water_density(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_density_grid is None:
            water_density_grid = build_water_density_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                gas_gravity_grid=gas_gravity_grid,  # type: ignore
                salinity_grid=water_salinity_grid,
                gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore
                gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
            )

    if water_formation_volume_factor_grid is None:
        if pvt_tables is not None:
            water_formation_volume_factor_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.water_formation_volume_factor(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_formation_volume_factor_grid is None:
            water_formation_volume_factor_grid = (
                build_water_formation_volume_factor_grid(
                    water_density_grid=water_density_grid,  # type: ignore
                    salinity_grid=water_salinity_grid,
                )
            )

    if water_bubble_point_pressure_grid is None:
        if pvt_tables is not None:
            water_bubble_point_pressure_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.water_bubble_point_pressure(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_bubble_point_pressure_grid is None:
            water_bubble_point_pressure_grid = build_water_bubble_point_pressure_grid(
                temperature_grid=temperature_grid,
                gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore
                salinity_grid=water_salinity_grid,
                gas=reservoir_gas,
            )

    if water_compressibility_grid is None:
        if pvt_tables is not None:
            water_compressibility_grid = typing.cast(
                typing.Optional[NDimensionalGrid[NDimension]],
                pvt_tables.water_compressibility(
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                    salinity=water_salinity_grid,
                ),
            )
        if water_compressibility_grid is None:
            water_compressibility_grid = build_water_compressibility_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                bubble_point_pressure_grid=water_bubble_point_pressure_grid,  # type: ignore
                gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,  # type: ignore
                gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore
                gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
            )

    oil_density_grid = None
    if pvt_tables is not None:
        oil_density_grid = typing.cast(
            typing.Optional[NDimensionalGrid[NDimension]],
            pvt_tables.oil_density(
                pressure=pressure_grid, temperature=temperature_grid
            ),
        )

    if oil_density_grid is None:
        oil_density_grid = build_live_oil_density_grid(
            oil_api_gravity_grid=oil_api_gravity_grid,
            gas_gravity_grid=gas_gravity_grid,  # type: ignore
            solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
            formation_volume_factor_grid=oil_formation_volume_factor_grid,  # type: ignore
        )

    solvent_concentration_grid = solvent_concentration_grid or build_uniform_grid(
        grid_shape=grid_shape, value=0.0
    )
    oil_effective_viscosity_grid = (
        oil_effective_viscosity_grid or oil_viscosity_grid.copy()
    )
    oil_effective_density_grid = oil_effective_density_grid or oil_density_grid.copy()
    fluid_properties = FluidProperties(
        pressure_grid=pressure_grid,
        temperature_grid=temperature_grid,
        oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
        oil_saturation_grid=oil_saturation_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        oil_specific_gravity_grid=oil_specific_gravity_grid,
        oil_api_gravity_grid=oil_api_gravity_grid,
        oil_density_grid=oil_density_grid,
        gas_saturation_grid=gas_saturation_grid,
        gas_viscosity_grid=gas_viscosity_grid,  # type: ignore
        gas_compressibility_grid=gas_compressibility_grid,  # type: ignore
        gas_gravity_grid=gas_gravity_grid,  # type: ignore
        gas_molecular_weight_grid=gas_molecular_weight_grid,  # type: ignore
        gas_density_grid=gas_density_grid,  # type: ignore
        water_saturation_grid=water_saturation_grid,
        water_viscosity_grid=water_viscosity_grid,  # type: ignore
        water_compressibility_grid=water_compressibility_grid,  # type: ignore
        water_density_grid=water_density_grid,  # type: ignore
        water_bubble_point_pressure_grid=water_bubble_point_pressure_grid,  # type: ignore
        solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,  # type: ignore
        oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,  # type: ignore
        gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,  # type: ignore
        water_formation_volume_factor_grid=water_formation_volume_factor_grid,  # type: ignore
        water_salinity_grid=water_salinity_grid,
        solvent_concentration_grid=solvent_concentration_grid,
        oil_effective_viscosity_grid=oil_effective_viscosity_grid,
        oil_effective_density_grid=oil_effective_density_grid,
        reservoir_gas=reservoir_gas,
    )
    rock_properties = RockProperties(
        compressibility=rock_compressibility,
        absolute_permeability=absolute_permeability,
        net_to_gross_ratio_grid=net_to_gross_ratio_grid,
        porosity_grid=porosity_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        connate_water_saturation_grid=connate_water_saturation_grid,
    )
    rock_fluid_properties = RockFluidProperties(
        relative_permeability_table=relative_permeability_table,
        capillary_pressure_table=capillary_pressure_table,
    )

    if saturation_history is None:
        # Just store the initial saturations as the max saturations
        saturation_history = SaturationHistory.from_initial_saturations(
            water_saturation_grid=water_saturation_grid.copy(),
            gas_saturation_grid=gas_saturation_grid.copy(),
        )
    if boundary_conditions is None:
        boundary_conditions = BoundaryConditions(
            conditions={
                "pressure": GridBoundaryCondition(),
                "oil_saturation": GridBoundaryCondition(),
                "gas_saturation": GridBoundaryCondition(),
                "water_saturation": GridBoundaryCondition(),
            }
        )

    model = ReservoirModel(
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        rock_fluid_properties=rock_fluid_properties,
        saturation_history=saturation_history,
        boundary_conditions=boundary_conditions,
        dip_angle=dip_angle,
        dip_azimuth=dip_azimuth,
    )
    if fractures is not None and len(model.grid_shape) == 3:
        return apply_fractures(model, *fractures)  # type: ignore[return-value]
    return model


def injection_well(
    well_name: str,
    perforating_intervals: typing.Sequence[typing.Tuple[WellLocation, WellLocation]],
    radius: float,
    control: WellControl,
    injected_fluid: InjectedFluid,
    **kwargs: typing.Any,
) -> InjectionWell[WellLocation]:
    """
    Constructs an injection well with the given parameters.

    :param well_name: Name or identifier for the well
    :param perforating_intervals: Sequence of tuples representing the start and end locations of each interval in the grid
    :param radius: Radius of the well (ft)
    :param injected_fluid: The fluid being injected into the well, represented as a `WellFluid` instance.
    :param control: Control parameters for the well, represented as a `WellControl` instance.
    :param kwargs: Additional keyword arguments to be passed to the `InjectionWell` constructor
    :return: `InjectionWell` instance
    """
    return InjectionWell(
        name=well_name,
        perforating_intervals=perforating_intervals,
        radius=radius,
        control=control,
        injected_fluid=injected_fluid,
        **kwargs,
    )


def production_well(
    well_name: str,
    perforating_intervals: typing.Sequence[typing.Tuple[WellLocation, WellLocation]],
    radius: float,
    control: WellControl,
    produced_fluids: typing.Sequence[ProducedFluid],
    skin_factor: float = 0.0,
    **kwargs: typing.Any,
) -> ProductionWell[WellLocation]:
    """
    Constructs a production well with the given parameters.

    :param well_name: Name or identifier for the well
    :param perforating_intervals: Sequence of tuples representing the start and end locations of each interval in the grid
    :param radius: Radius of the well (ft)
    :param produced_fluids: List of fluids being produced by the well, represented as a sequence of `WellFluid` instances.
    :param skin_factor: Skin factor for the well, default is 0.0
    :param control: Control parameters for the well, represented as a `WellControl` instance.
    :param kwargs: Additional keyword arguments to be passed to the `ProductionWell` constructor
    :return: `ProductionWell` instance
    """
    if not produced_fluids:
        raise ValueError("Produced fluids list must not be empty.")
    return ProductionWell(
        name=well_name,
        perforating_intervals=perforating_intervals,
        radius=radius,
        skin_factor=skin_factor,
        control=control,
        produced_fluids=produced_fluids,
        **kwargs,
    )


def wells_(
    injectors: typing.Optional[typing.Sequence[InjectionWell[WellLocation]]] = None,
    producers: typing.Optional[typing.Sequence[ProductionWell[WellLocation]]] = None,
    **kwargs: typing.Any,
) -> Wells[WellLocation]:
    """
    Constructs a ``Wells`` instance containing both injection and production wells.

    :param injectors: Sequence of injection wells
    :param producers: Sequence of production wells
    :param kwargs: Additional keyword arguments to be passed to the `Wells` constructor
    :return: ``Wells`` instance
    """
    return Wells(
        injection_wells=injectors or [],
        production_wells=producers or [],
        **kwargs,
    )
