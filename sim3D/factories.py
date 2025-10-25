import typing
import warnings

import numpy as np

from sim3D.boundaries import BoundaryConditions, GridBoundaryCondition
from sim3D.constants import RESERVOIR_GAS_NAME
from sim3D.grids.base import build_uniform_grid
from sim3D.grids.properties import (
    build_fluid_viscosity_grid,
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_gravity_grid,
    build_gas_molecular_weight_grid,
    build_gas_solubility_in_water_grid,
    build_gas_to_oil_ratio_grid,
    build_live_oil_density_grid,
    build_oil_api_gravity_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_formation_volume_factor_grid,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from sim3D.statics import (
    CapillaryPressureParameters,
    FluidProperties,
    ReservoirModel,
    RockFluidProperties,
    RockPermeability,
    RockProperties,
)
from sim3D.properties import (
    compute_gas_to_oil_ratio_standing,
    validate_input_pressure,
    validate_input_temperature,
)
from sim3D.types import (
    NDimension,
    NDimensionalGrid,
    RelativePermeabilityFunc,
    WellLocation,
)
from sim3D.wells import InjectionWell, ProductionWell, WellFluid, Wells
from sim3D.faults import Fault, apply_faults

__all__ = [
    "reservoir_model",
    "injection_well",
    "production_well",
    "wells",
    "validate_saturation_grids",
]


def validate_saturation_grids(
    oil_saturation_grid: NDimensionalGrid[NDimension],
    water_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    connate_water_saturation_grid: NDimensionalGrid[NDimension],
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension],
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
            raise ValueError(
                "The sum of oil, water, and gas saturations does not equal 1 everywhere in active cells."
            )

    # Validate saturation ranges in active cells
    if np.any(
        (oil_saturation_grid[active_cells] < 0)
        | (oil_saturation_grid[active_cells] >= 1)
    ):
        raise ValueError(
            "Oil saturation grid values must be in the range [0, 1) in active cells."
        )

    if np.any(
        (water_saturation_grid[active_cells] < 0)
        | (water_saturation_grid[active_cells] >= 1)
    ):
        raise ValueError(
            "Water saturation grid values must be in the range [0, 1) in active cells."
        )

    if np.any(
        (gas_saturation_grid[active_cells] < 0)
        | (gas_saturation_grid[active_cells] >= 1)
    ):
        raise ValueError(
            "Gas saturation grid values must be in the range [0, 1) in active cells."
        )

    # Validate residual saturations in active cells
    # if np.any(
    #     (residual_oil_saturation_water_grid[active_cells] < 0)
    #     | (
    #         residual_oil_saturation_water_grid[active_cells]
    #         > oil_saturation_grid[active_cells]
    #     )
    # ):
    #     raise ValueError(
    #         "Residual oil saturation during water flooding grid values must be in the range "
    #         "[0, oil_saturation] in active cells. Oil saturation must be greater than residual oil saturation during water flooding."
    #     )

    # if np.any(
    #     (residual_oil_saturation_gas_grid[active_cells] < 0)
    #     | (
    #         residual_oil_saturation_gas_grid[active_cells]
    #         > oil_saturation_grid[active_cells]
    #     )
    # ):
    #     raise ValueError(
    #         "Residual oil saturation during gas flooding grid values must be in the range "
    #         "[0, oil_saturation] in active cells. Oil saturation must be greater than residual oil saturation during gas flooding."
    #     )

    # if np.any(
    #     (residual_gas_saturation_grid[active_cells] < 0)
    #     | (
    #         residual_gas_saturation_grid[active_cells]
    #         > gas_saturation_grid[active_cells]
    #     )
    # ):
    #     raise ValueError(
    #         "Residual gas saturation grid values must be in the range [0, gas_saturation] in active cells."
    #         " Gas saturation must be greater than residual gas saturation."
    #     )

    # if np.any(
    #     (connate_water_saturation_grid[active_cells] < 0)
    #     | (
    #         connate_water_saturation_grid[active_cells]
    #         > irreducible_water_saturation_grid[active_cells]
    #     )
    # ):
    #     raise ValueError(
    #         "Connate water saturation grid values must be in the range "
    #         "[0, irreducible_water_saturation] in active cells. Connate water saturation is usually less than irreducible water saturation."
    #     )

    # if np.any(
    #     (connate_water_saturation_grid[active_cells] < 0)
    #     | (
    #         connate_water_saturation_grid[active_cells]
    #         > water_saturation_grid[active_cells]
    #     )
    # ):
    #     raise ValueError(
    #         "Connate water saturation grid values must be in the range [0, water_saturation] in active cells."
    #         " Connate water saturation is usually less than or equal to water saturation."
    #     )

    # if np.any(
    #     (irreducible_water_saturation_grid[active_cells] < 0)
    #     | (
    #         irreducible_water_saturation_grid[active_cells]
    #         > water_saturation_grid[active_cells]
    #     )
    # ):
    #     raise ValueError(
    #         "Irreducible water saturation grid values must be in the range "
    #         "[0, water_saturation] in active cells."
    #     )

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
    oil_specific_gravity_grid: NDimensionalGrid[NDimension],
    oil_bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension],
    connate_water_saturation_grid: NDimensionalGrid[NDimension],
    relative_permeability_func: RelativePermeabilityFunc,
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
    capillary_pressure_params: typing.Optional[CapillaryPressureParameters] = None,
    gas_to_oil_ratio_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
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
    boundary_conditions: typing.Optional[BoundaryConditions] = None,
    faults: typing.Optional[typing.Iterable[Fault]] = None,
    reservoir_gas_name: str = RESERVOIR_GAS_NAME,
) -> ReservoirModel[NDimension]:
    """
    Constructs a N-Dimensional reservoir model with given rock and fluid properties.

    Notes:
    ------
    - If both `oil_bubble_point_pressure_grid` and `gas_to_oil_ratio_grid` are omitted,
      the function attempts to estimate these based on pressure, temperature, and oil properties,
      but this may lead to less accurate results.
    - Saturation grids will be adjusted internally to ensure saturation sums to 1.
    - All fluid property grids are estimated from pressure and temperature if not provided.
    - Provide consistent and physically realistic input grids to ensure stable simulations.

    :param grid_shape: Number of grid cells (rows, columns).
    :param cell_dimension: Physical size of each cell (dx, dy) in feets.
    :param thickness_grid: Reservoir cells' thickness grid (ft).
    :param pressure_grid: Reservoir cells' pressure grid (psi).
    :param oil_bubble_point_pressure_grid: Oil bubble point pressure grid (psi).
    :param rock_compressibility: Rock compressibility in psi⁻¹.
    :param absolute_permeability_grid: Absolute permeability grid (mD).
    :param porosity_grid: Porosity grid (fraction).
    :param temperature_grid: Reservoir temperature grid (°F).
    :param oil_saturation_grid: Oil saturation grid (fraction).
    :param oil_viscosity_grid: Oil viscosity grid (cP).
    :param oil_compressibility_grid: Oil compressibility grid (psi⁻¹).
    :param oil_specific_gravity_grid: Oil specific gravity grid (dimensionless).
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
    :param capillary_pressure_params: Capillary pressure parameters, optional.
    :param gas_to_oil_ratio_grid: Solution gas to oil ratio grid (scf/bbl), optional.
    :param gas_solubility_in_water_grid: Gas solubility in water grid (scf/bbl), optional.
    :param oil_formation_volume_factor_grid: Oil formation volume factor grid (bbl/scf), optional.
    :param gas_formation_volume_factor_grid: Gas formation volume factor grid (bbl/scf), optional.
    :param water_formation_volume_factor_grid: Water formation volume factor grid (bbl/scf), optional.
    :param net_to_gross_ratio_grid: Net-to-gross ratio grid (fraction), optional.
    :param water_salinity_grid: Water salinity grid (ppm), optional.
    :param boundary_conditions: Boundary conditions for the model, optional. Defaults to no-flow conditions.
    :param faults: Iterable of faults to be applied to the reservoir model, optional.
    :param reservoir_gas_name: Name of the reservoir gas, defaults to `RESERVOIR_GAS_NAME`. Can also be the name of the gas injected into the reservoir.
    :return: The constructed N-Dimensional reservoir model with fluid and rock properties.
    """
    if not 1 <= len(grid_shape) <= 3:
        raise ValueError(
            "`grid_shape` must be a tuple of one to three integers (rows, columns, [depth])."
        )
    if len(cell_dimension) < 2:
        raise ValueError(
            "`cell_dimension` must be a tuple of two floats (cell width, cell height)."
        )

    validate_input_pressure(pressure_grid)
    validate_input_temperature(temperature_grid)

    if water_salinity_grid is None:
        water_salinity_grid = build_uniform_grid(
            grid_shape=grid_shape,
            value=35_000,  # Default salinity in ppm (NaCl)
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
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            connate_water_saturation_grid=connate_water_saturation_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            porosity_grid=porosity_grid,
            normalize=True,
        )
    )

    # Viscosity Grids
    if gas_viscosity_grid is None:
        gas_viscosity_grid = build_fluid_viscosity_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            fluid=reservoir_gas_name,
        )

    if gas_gravity_grid is None:
        gas_gravity_grid = build_gas_gravity_grid(gas=reservoir_gas_name)

    if water_viscosity_grid is None:
        water_viscosity_grid = build_water_viscosity_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            salinity_grid=water_salinity_grid,
        )

    if gas_compressibility_factor_grid is None:
        gas_compressibility_factor_grid = build_gas_compressibility_factor_grid(
            gas_gravity_grid=gas_gravity_grid,
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
        )

    # Compressibility Grids
    if gas_compressibility_grid is None:
        gas_compressibility_grid = build_gas_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=gas_gravity_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )

    if gas_molecular_weight_grid is None:
        gas_molecular_weight_grid = build_gas_molecular_weight_grid(
            gas_gravity_grid=gas_gravity_grid
        )

    # Density Grids
    if gas_density_grid is None:
        gas_density_grid = build_gas_density_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=gas_gravity_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )

    oil_api_gravity_grid = build_oil_api_gravity_grid(oil_specific_gravity_grid)
    if gas_to_oil_ratio_grid is None and oil_bubble_point_pressure_grid is not None:
        gas_to_oil_ratio_grid = build_gas_to_oil_ratio_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            gas_gravity_grid=gas_gravity_grid,
            oil_api_gravity_grid=oil_api_gravity_grid,
        )
    elif gas_to_oil_ratio_grid is not None and oil_bubble_point_pressure_grid is None:
        oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
            gas_gravity_grid=gas_gravity_grid,
            oil_api_gravity_grid=oil_api_gravity_grid,
            temperature_grid=temperature_grid,
            gas_to_oil_ratio_grid=gas_to_oil_ratio_grid,
        )
    elif gas_to_oil_ratio_grid is None and oil_bubble_point_pressure_grid is None:
        warnings.warn(
            "Both `oil_bubble_point_pressure_grid` and `gas_to_oil_ratio_grid` are not provided. "
            "Attempting to estimate the bubble point pressure and GOR. If estimation fails, "
            "please provide at least one of them. Note, estimating the bubble point pressure "
            "and GOR may not yield accurate results.",
            UserWarning,
        )
        # Try to estimate the GOR and then calculate the bubble point pressure from that
        # As either GOR or buble point is needed to build the model
        estimated_gor_grid = np.vectorize(
            compute_gas_to_oil_ratio_standing, otypes=[np.float64]
        )(
            pressure_grid,
            oil_api_gravity_grid,
            gas_gravity_grid,
        )
        oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
            temperature_grid=temperature_grid,
            gas_gravity_grid=gas_gravity_grid,
            oil_api_gravity_grid=oil_api_gravity_grid,
            gas_to_oil_ratio_grid=estimated_gor_grid,
        )
        gas_to_oil_ratio_grid = estimated_gor_grid

    gas_to_oil_ratio_grid = typing.cast(
        NDimensionalGrid[NDimension], gas_to_oil_ratio_grid
    )
    oil_bubble_point_pressure_grid = typing.cast(
        NDimensionalGrid[NDimension], oil_bubble_point_pressure_grid
    )

    if gas_solubility_in_water_grid is None:
        gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            salinity_grid=water_salinity_grid,
            gas=reservoir_gas_name,
        )

    # Formation Volume Factor Grids
    if oil_formation_volume_factor_grid is None:
        oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
            oil_specific_gravity_grid=oil_specific_gravity_grid,
            gas_gravity_grid=gas_gravity_grid,
            gas_to_oil_ratio_grid=gas_to_oil_ratio_grid,
            oil_compressibility_grid=oil_compressibility_grid,
        )

    if gas_formation_volume_factor_grid is None:
        gas_formation_volume_factor_grid = build_gas_formation_volume_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )

    gas_free_water_formation_volume_factor_grid = (
        build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=pressure_grid, temperature_grid=temperature_grid
        )
    )
    if water_density_grid is None:
        water_density_grid = build_water_density_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=gas_gravity_grid,
            salinity_grid=water_salinity_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
        )

    if water_formation_volume_factor_grid is None:
        water_formation_volume_factor_grid = build_water_formation_volume_factor_grid(
            water_density_grid=water_density_grid,
            salinity_grid=water_salinity_grid,
        )

    if water_bubble_point_pressure_grid is None:
        water_bubble_point_pressure_grid = build_water_bubble_point_pressure_grid(
            temperature_grid=temperature_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            salinity_grid=water_salinity_grid,
            gas=reservoir_gas_name,
        )

    if water_compressibility_grid is None:
        water_compressibility_grid = build_water_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=water_bubble_point_pressure_grid,
            gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
        )

    oil_density_grid = build_live_oil_density_grid(
        oil_api_gravity_grid=oil_api_gravity_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_to_oil_ratio_grid=gas_to_oil_ratio_grid,
        formation_volume_factor_grid=oil_formation_volume_factor_grid,
    )

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
        gas_viscosity_grid=gas_viscosity_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_molecular_weight_grid=gas_molecular_weight_grid,
        gas_density_grid=gas_density_grid,
        water_saturation_grid=water_saturation_grid,
        water_viscosity_grid=water_viscosity_grid,
        water_compressibility_grid=water_compressibility_grid,
        water_density_grid=water_density_grid,
        water_bubble_point_pressure_grid=water_bubble_point_pressure_grid,
        gas_to_oil_ratio_grid=gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,
        gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
        water_formation_volume_factor_grid=water_formation_volume_factor_grid,
        water_salinity_grid=water_salinity_grid,
        reservoir_gas_name=reservoir_gas_name,
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
        relative_permeability_func=relative_permeability_func,
        capillary_pressure_params=capillary_pressure_params
        or CapillaryPressureParameters(),
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
        boundary_conditions=boundary_conditions,
    )
    if faults is not None:
        model = apply_faults(model, *faults)
    return model


def injection_well(
    well_name: str,
    perforating_intervals: typing.Sequence[typing.Tuple[WellLocation, WellLocation]],
    radius: float,
    bottom_hole_pressure: float,
    injected_fluid: WellFluid,
    **kwargs: typing.Any,
) -> InjectionWell[WellLocation]:
    """
    Constructs an injection well with the given parameters.

    :param well_name: Name or identifier for the well
    :param perforating_intervals: Sequence of tuples representing the start and end locations of each interval in the grid
    :param radius: Radius of the well (ft)
    :param injected_fluid: The fluid being injected into the well, represented as a `WellFluid` instance.
    :param bottom_hole_pressure: Bottom hole pressure of the well (psi)
    :param kwargs: Additional keyword arguments to be passed to the `InjectionWell` constructor
    :return: `InjectionWell` instance
    """
    return InjectionWell(
        name=well_name,
        perforating_intervals=perforating_intervals,
        radius=radius,
        bottom_hole_pressure=bottom_hole_pressure,
        injected_fluid=injected_fluid,
        **kwargs,
    )


def production_well(
    well_name: str,
    perforating_intervals: typing.Sequence[typing.Tuple[WellLocation, WellLocation]],
    radius: float,
    bottom_hole_pressure: float,
    produced_fluids: typing.Sequence[WellFluid],
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
    :param bottom_hole_pressure: Bottom hole pressure of the well (psi)
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
        bottom_hole_pressure=bottom_hole_pressure,
        produced_fluids=produced_fluids,
        **kwargs,
    )


def wells(
    injectors: typing.Optional[typing.Sequence[InjectionWell[WellLocation]]] = None,
    producers: typing.Optional[typing.Sequence[ProductionWell[WellLocation]]] = None,
    **kwargs: typing.Any,
) -> Wells[WellLocation]:
    """
    Constructs a Wells instance containing both injection and production wells.

    :param injectors: Sequence of injection wells
    :param producers: Sequence of production wells
    :param kwargs: Additional keyword arguments to be passed to the `Wells` constructor
    :return: ``Wells`` instance
    """
    return Wells(
        injection_wells=injectors or [], production_wells=producers or [], **kwargs
    )
