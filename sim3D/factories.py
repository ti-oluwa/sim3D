import typing
import warnings
import numpy as np

from sim3D.constants import RESERVOIR_GAS_NAME
from sim3D.types import NDimension, NDimensionalGrid, WellLocation
from sim3D.grids import (
    build_gas_compressibility_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_fluid_viscosity_grid,
    build_uniform_grid,
    build_oil_formation_volume_factor_grid,
    build_gas_formation_volume_factor_grid,
    build_water_formation_volume_factor_grid,
    build_gas_solubility_in_water_grid,
    build_oil_bubble_point_pressure_grid,
    build_water_bubble_point_pressure_grid,
    build_gas_to_oil_ratio_grid,
    build_oil_api_gravity_grid,
    build_water_viscosity_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_gas_gravity_grid,
    build_gas_molecular_weight_grid,
    build_live_oil_density_grid,
)
from sim3D.boundary_conditions import BoundaryConditions, GridBoundaryCondition
from sim3D.models import (
    ReservoirModel,
    FluidProperties,
    RockPermeability,
    RockProperties,
    RelativePermeabilityParameters,
    CapillaryPressureParameters,
)
from sim3D.wells import (
    InjectedFluid,
    ProducedFluid,
    InjectionWell,
    ProductionWell,
    Wells,
)
from sim3D.properties import (
    validate_input_temperature,
    validate_input_pressure,
    compute_gas_to_oil_ratio_standing,
)

__all__ = [
    "build_reservoir_model",
    "build_injection_well",
    "build_production_well",
    "build_wells",
]


def build_reservoir_model(
    grid_dimension: NDimension,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[NDimension],
    pressure_grid: NDimensionalGrid[NDimension],
    rock_compressibility: float,
    absolute_permeability: RockPermeability[NDimension],
    porosity_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    oil_saturation_grid: NDimensionalGrid[NDimension],
    oil_viscosity_grid: NDimensionalGrid[NDimension],
    oil_compressibility_grid: NDimensionalGrid[NDimension],
    oil_specific_gravity_grid: NDimensionalGrid[NDimension],
    oil_bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    residual_oil_saturation_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_saturation_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_viscosity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_compressibility_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    gas_molecular_weight_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_density_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    residual_gas_saturation_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_saturation_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_viscosity_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_density_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    water_bubble_point_pressure_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    irreducible_water_saturation_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    relative_permeability_params: typing.Optional[
        RelativePermeabilityParameters
    ] = None,
    capillary_pressure_params: typing.Optional[CapillaryPressureParameters] = None,
    gas_to_oil_ratio_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
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

    :param grid_dimension: Number of grid cells (rows, columns).
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
    :param residual_oil_saturation_grid: Residual oil saturation grid (fraction), optional.
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
    :param irreducible_water_saturation_grid: Irreducible water saturation grid (fraction), optional.
    :param relative_permeability_params: Relative permeability parameters, optional.
    :param capillary_pressure_params: Capillary pressure parameters, optional.
    :param gas_to_oil_ratio_grid: Gas to oil ratio grid (scf/bbl), optional.
    :param oil_formation_volume_factor_grid: Oil formation volume factor grid (bbl/scf), optional.
    :param gas_formation_volume_factor_grid: Gas formation volume factor grid (bbl/scf), optional.
    :param water_formation_volume_factor_grid: Water formation volume factor grid (bbl/scf), optional.
    :param net_to_gross_ratio_grid: Net-to-gross ratio grid (fraction), optional.
    :param water_salinity_grid: Water salinity grid (ppm), optional.
    :param boundary_conditions: Boundary conditions for the model, optional. Defaults to no-flow conditions.
    :param reservoir_gas_name: Name of the reservoir gas, defaults to `RESERVOIR_GAS_NAME`.
    :return: The constructed N-Dimensional reservoir model with fluid and rock properties.
    """
    if not 1 <= len(grid_dimension) <= 3:
        raise ValueError(
            "`grid_dimension` must be a tuple of one to three integers (rows, columns, [depth])."
        )
    if len(cell_dimension) < 2:
        raise ValueError(
            "`cell_dimension` must be a tuple of two floats (cell width, cell height)."
        )

    validate_input_pressure(pressure_grid)
    validate_input_temperature(temperature_grid)

    # Generics
    if water_salinity_grid is None:
        water_salinity_grid = build_uniform_grid(
            grid_dimension=grid_dimension,
            value=35_000,  # Default salinity in ppm (NaCl)
        )

    if net_to_gross_ratio_grid is None:
        # assuming a uniform net-to-gross ratio of 1.0 (fully net)
        net_to_gross_ratio_grid = build_uniform_grid(
            grid_dimension=grid_dimension,
            value=1.0,
        )

    # Saturation grids
    if residual_oil_saturation_grid is None:
        residual_oil_saturation_grid = typing.cast(
            NDimensionalGrid[NDimension], oil_saturation_grid * 0.2
        )

    if residual_gas_saturation_grid is None:
        residual_gas_saturation_grid = build_uniform_grid(
            grid_dimension=grid_dimension,
            value=0.0,
        )

    if gas_saturation_grid is None:
        # assuming no gas in the model by default
        gas_saturation_grid = build_uniform_grid(
            grid_dimension=grid_dimension,
            value=0.0,
        )
    if water_saturation_grid is None:
        water_saturation_grid = typing.cast(
            NDimensionalGrid[NDimension],
            1.0 - (oil_saturation_grid + gas_saturation_grid),
        )

    if irreducible_water_saturation_grid is None:
        irreducible_water_saturation_grid = typing.cast(
            NDimensionalGrid[NDimension], water_saturation_grid * 0.2
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
    gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
        pressure_grid=pressure_grid,
        temperature_grid=temperature_grid,
        salinity_grid=water_salinity_grid,
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
        oil_api_gravity_grid=   oil_api_gravity_grid,
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
        oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,
        gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
        water_formation_volume_factor_grid=water_formation_volume_factor_grid,
        water_salinity_grid=water_salinity_grid,
    )
    rock_properties = RockProperties(
        compressibility=rock_compressibility,
        absolute_permeability=absolute_permeability,
        net_to_gross_ratio_grid=net_to_gross_ratio_grid,
        porosity_grid=porosity_grid,
        residual_oil_saturation_grid=residual_oil_saturation_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        relative_permeability_params=relative_permeability_params
        or RelativePermeabilityParameters(),
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
    return ReservoirModel(
        grid_dimension=grid_dimension,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        boundary_conditions=boundary_conditions,
    )


def build_injection_well(
    well_name: str,
    perforating_interval: typing.Tuple[WellLocation, WellLocation],
    radius: float,
    bottom_hole_pressure: float,
    injected_fluid: InjectedFluid,
) -> InjectionWell[WellLocation]:
    """
    Constructs an injection well with the given parameters.

    :param well_name: Name or identifier for the well
    :param perforating_interval: Tuple representing the start and end locations of the well in the grid
    :param radius: Radius of the well (ft)
    :param injected_fluid: The fluid being injected into the well, represented as a `InjectedFluid` instance.
    :return: `InjectionWell` instance
    """
    return InjectionWell(
        name=well_name,
        perforating_interval=perforating_interval,
        radius=radius,
        bottom_hole_pressure=bottom_hole_pressure,
        injected_fluid=injected_fluid,
    )


def build_production_well(
    well_name: str,
    perforating_interval: typing.Tuple[WellLocation, WellLocation],
    radius: float,
    bottom_hole_pressure: float,
    produced_fluids: typing.Sequence[ProducedFluid],
    skin_factor: float = 0.0,
) -> ProductionWell[WellLocation]:
    """
    Constructs a production well with the given parameters.

    :param well_name: Name or identifier for the well
    :param perforating_interval: Tuple representing the start and end locations of the well in the grid
    :param radius: Radius of the well (ft)
    :param produced_fluids: List of fluids being produced by the well, represented as a sequence of `ProducedFluid` instances.
    :param skin_factor: Skin factor for the well, default is 0.0
    :return: `ProductionWell` instance
    """
    if not produced_fluids:
        raise ValueError("Produced fluids list must not be empty.")
    return ProductionWell(
        name=well_name,
        perforating_interval=perforating_interval,
        radius=radius,
        skin_factor=skin_factor,
        bottom_hole_pressure=bottom_hole_pressure,
        produced_fluids=produced_fluids,
    )


def build_wells(
    injection_wells: typing.Optional[typing.Sequence[InjectionWell[WellLocation]]] = None,
    production_wells: typing.Optional[typing.Sequence[ProductionWell[WellLocation]]] = None,
) -> Wells[WellLocation]:
    """
    Constructs a Wells instance containing both injection and production wells.

    :param injection_wells: Sequence of injection wells
    :param production_wells: Sequence of production wells
    :return: ``Wells`` instance
    """
    return Wells(
        injection_wells=injection_wells or [],
        production_wells=production_wells or [],
    )
