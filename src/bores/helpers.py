import logging
import typing

import attrs
import numpy as np

from bores._precision import get_dtype
from bores.boundaries import BoundaryConditions, BoundaryMetadata, default_bc
from bores.grids.base import (
    CapillaryPressureGrids,
    RelPermGrids,
    RelativeMobilityGrids,
    build_uniform_grid,
)
from bores.grids.pvt import (
    build_gas_compressibility_factor_grid,
    build_gas_compressibility_grid,
    build_gas_density_grid,
    build_gas_formation_volume_factor_grid,
    build_gas_free_water_formation_volume_factor_grid,
    build_gas_solubility_in_water_grid,
    build_gas_viscosity_grid,
    build_live_oil_density_grid,
    build_oil_bubble_point_pressure_grid,
    build_oil_compressibility_grid,
    build_oil_effective_density_grid,
    build_oil_effective_viscosity_grid,
    build_oil_formation_volume_factor_grid,
    build_oil_viscosity_grid,
    build_solution_gas_to_oil_ratio_grid,
    build_three_phase_capillary_pressure_grids,
    build_three_phase_relative_mobilities_grids,
    build_three_phase_relative_permeabilities_grids,
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from bores.models import FluidProperties, RockFluidProperties, RockProperties
from bores.types import (
    MiscibilityModel,
    NDimension,
    NDimensionalGrid,
    RelativeMobilityRange,
    ThreeDimensions,
)
from bores.wells import Wells

logger = logging.getLogger(__name__)


def _mirror_neighbour(
    grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """Mirrors the neighbour cells for boundary padding."""
    default_bc.apply(grid)
    return grid


def apply_boundary_conditions(
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    cell_dimension: typing.Tuple[float, float],
    grid_shape: typing.Tuple[int, int, int],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    time: float,
) -> typing.Tuple[FluidProperties, RockProperties]:
    """
    Applies boundary conditions to the fluid property grids.

    :param fluid_properties: The padded fluid properties.
    :param rock_properties: The padded rock properties.
    :param boundary_conditions: The boundary conditions to apply.
    :param cell_dimension: The dimensions of each grid cell.
    :param grid_shape: The shape of the simulation grid.
    :param thickness_grid: The (unpadded) thickness grid of the reservoir.
    :param time: The current simulation time.
    """
    boundary_conditions["pressure"].apply(
        fluid_properties.pressure_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="pressure",
        ),
    )
    boundary_conditions["oil_saturation"].apply(
        fluid_properties.oil_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="oil_saturation",
        ),
    )
    boundary_conditions["water_saturation"].apply(
        fluid_properties.water_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="water_saturation",
        ),
    )
    boundary_conditions["gas_saturation"].apply(
        fluid_properties.gas_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="gas_saturation",
        ),
    )
    boundary_conditions["temperature"].apply(
        fluid_properties.temperature_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            grid_shape=grid_shape,
            property_name="temperature",
        ),
    )
    # Clamp saturations to [0, 1] after applying BCs
    dtype = get_dtype()
    fluid_properties.oil_saturation_grid.clip(
        min=0.0, max=1.0, out=fluid_properties.oil_saturation_grid, dtype=dtype
    )
    fluid_properties.water_saturation_grid.clip(
        min=0.0, max=1.0, out=fluid_properties.water_saturation_grid, dtype=dtype
    )
    fluid_properties.gas_saturation_grid.clip(
        min=0.0, max=1.0, out=fluid_properties.gas_saturation_grid, dtype=dtype
    )
    excluded_fluid_properties = (
        "pressure_grid",
        "oil_saturation_grid",
        "water_saturation_grid",
        "gas_saturation_grid",
        "temperature_grid",
    )
    fluid_properties = fluid_properties.apply_hook(
        hook=_mirror_neighbour, exclude=excluded_fluid_properties
    )
    rock_properties = rock_properties.apply_hook(hook=_mirror_neighbour)
    return fluid_properties, rock_properties


def build_rock_fluid_properties_grids(
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    disable_capillary_effects: bool = False,
    capillary_strength_factor: float = 1.0,
    relative_mobility_range: typing.Optional[RelativeMobilityRange] = None,
) -> typing.Tuple[
    RelPermGrids[ThreeDimensions],
    RelativeMobilityGrids[ThreeDimensions],
    CapillaryPressureGrids[ThreeDimensions],
]:
    """
    Builds the rock-fluid properties grids required for simulation.

    :param fluid_properties: `FluidProperties` object containing fluid property grids.
    :param rock_properties: `RockProperties` object containing rock property grids.
    :param rock_fluid_properties: `RockFluidProperties` object containing rock-fluid property tables.
    :param disable_capillary_effects: If True, capillary effects are disabled (zero capillary pressures).
    :param capillary_strength_factor: Factor to scale capillary pressure grids.
    :param relative_mobility_range: Optional clamping range for relative mobility grids.
    :return: A tuple containing:
        - RelPermGrids: Relative permeability grids for oil, water, and gas.
        - RelativeMobilityGrids: Relative mobility grids for oil, water, and gas.
        - CapillaryPressureGrids: Capillary pressure grids for oil-water and gas-oil.
    """
    # Collect and clamp saturation grids
    water_saturation_grid = fluid_properties.water_saturation_grid
    oil_saturation_grid = fluid_properties.oil_saturation_grid
    gas_saturation_grid = fluid_properties.gas_saturation_grid
    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_water_grid = (
        rock_properties.residual_oil_saturation_water_grid
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    water_viscosity_grid = fluid_properties.water_viscosity_grid
    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid
    relperm_table = rock_fluid_properties.relative_permeability_table
    capillary_pressure_table = rock_fluid_properties.capillary_pressure_table
    krw_grid, kro_grid, krg_grid = build_three_phase_relative_permeabilities_grids(
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        relative_permeability_table=relperm_table,
    )
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_three_phase_relative_mobilities_grids(
        oil_relative_permeability_grid=kro_grid,
        water_relative_permeability_grid=krw_grid,
        gas_relative_permeability_grid=krg_grid,
        water_viscosity_grid=water_viscosity_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
    )

    if relative_mobility_range is not None:
        # Clamp relative mobility grids to avoid numerical issues
        water_relative_mobility_grid = relative_mobility_range["water"].arrayclip(
            water_relative_mobility_grid
        )
        oil_relative_mobility_grid = relative_mobility_range["oil"].arrayclip(
            oil_relative_mobility_grid
        )
        gas_relative_mobility_grid = relative_mobility_range["gas"].arrayclip(
            gas_relative_mobility_grid
        )

    if disable_capillary_effects:
        logger.debug("Capillary effects disabled; using zero capillary pressure grids")
        oil_water_capillary_pressure_grid = build_uniform_grid(
            grid_shape=water_saturation_grid.shape, value=0.0
        )
        gas_oil_capillary_pressure_grid = build_uniform_grid(
            grid_shape=water_saturation_grid.shape, value=0.0
        )
    else:
        (
            oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid,
        ) = build_three_phase_capillary_pressure_grids(
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            capillary_pressure_table=capillary_pressure_table,
        )
        if capillary_strength_factor != 1.0:
            logger.debug(
                f"Scaling capillary pressure grids by factor {capillary_strength_factor}"
            )
            oil_water_capillary_pressure_grid = typing.cast(
                NDimensionalGrid[ThreeDimensions],
                oil_water_capillary_pressure_grid * capillary_strength_factor,
            )
            gas_oil_capillary_pressure_grid = typing.cast(
                NDimensionalGrid[ThreeDimensions],
                gas_oil_capillary_pressure_grid * capillary_strength_factor,
            )

    padded_relperm_grids = RelPermGrids(
        oil_relative_permeability=kro_grid,
        water_relative_permeability=krw_grid,
        gas_relative_permeability=krg_grid,
    )
    padded_relative_mobility_grids = RelativeMobilityGrids(
        water_relative_mobility=water_relative_mobility_grid,
        oil_relative_mobility=oil_relative_mobility_grid,
        gas_relative_mobility=gas_relative_mobility_grid,
    )
    padded_relative_mobility_grids = typing.cast(
        RelativeMobilityGrids[ThreeDimensions], padded_relative_mobility_grids
    )
    padded_capillary_pressure_grids = CapillaryPressureGrids[ThreeDimensions](
        oil_water_capillary_pressure=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure=gas_oil_capillary_pressure_grid,
    )
    return (
        padded_relperm_grids,
        padded_relative_mobility_grids,
        padded_capillary_pressure_grids,
    )


def update_phase_densities(
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
) -> FluidProperties[ThreeDimensions]:
    """
    Updates only the phase density grids based on current pressure, temperature, and composition.

    This is a lightweight version of update_pvt_properties that only recalculates densities,
    making it suitable for use in perturbation calculations where only densities are needed.

    Updates:
    - Gas density (ρg)
    - Water density (ρw)
    - Oil density (ρo)
    - Oil effective density (ρo_eff) - includes miscibility effects if applicable

    :param fluid_properties: Current fluid property grids
    :param wells: Current wells configuration
    :param miscibility_model: The miscibility model used in the simulation
    :return: FluidProperties with updated density grids only
    """
    # GAS DENSITY
    # Need z-factor to compute gas density
    gas_compressibility_factor_grid = build_gas_compressibility_factor_grid(
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
    )
    new_gas_density_grid = build_gas_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )

    # WATER DENSITY
    # Need gas solubility and gas-free water FVF for water density
    gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        gas=fluid_properties.reservoir_gas,
    )
    gas_free_water_formation_volume_factor_grid = (
        build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=fluid_properties.pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
        )
    )
    new_water_density_grid = build_water_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )

    # OIL DENSITY
    new_oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        temperature_grid=fluid_properties.temperature_grid,
        solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
    )
    gor_at_bubble_point_pressure_grid = build_solution_gas_to_oil_ratio_grid(
        pressure_grid=new_oil_bubble_point_pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
    )
    new_solution_gas_to_oil_ratio_grid = build_solution_gas_to_oil_ratio_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
    )
    new_oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        oil_compressibility_grid=fluid_properties.oil_compressibility_grid,
    )
    new_oil_density_grid = build_live_oil_density_grid(
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
    )
    new_oil_effective_density_grid = new_oil_density_grid

    # Handle miscible injections if applicable
    if miscibility_model != "immiscible":
        for well in wells.injection_wells:
            injected_fluid = well.injected_fluid
            if injected_fluid and injected_fluid.is_miscible:
                injected_fluid_viscosity_grid = np.vectorize(
                    injected_fluid.get_viscosity, otypes=[get_dtype()]
                )(
                    pressure=fluid_properties.pressure_grid,
                    temperature=fluid_properties.temperature_grid,
                )
                # Update effective oil density using Todd-Longstaff model
                new_oil_effective_density_grid = build_oil_effective_density_grid(
                    oil_density_grid=new_oil_density_grid,
                    solvent_density_grid=injected_fluid_viscosity_grid,
                    oil_viscosity_grid=fluid_properties.oil_effective_viscosity_grid,
                    solvent_viscosity_grid=injected_fluid_viscosity_grid,
                    solvent_concentration_grid=fluid_properties.solvent_concentration_grid,
                    base_omega=injected_fluid.todd_longstaff_omega,
                    pressure_grid=fluid_properties.pressure_grid,
                    minimum_miscibility_pressure=injected_fluid.minimum_miscibility_pressure,
                    transition_width=injected_fluid.miscibility_transition_width,
                )

    updated_fluid_properties = attrs.evolve(
        fluid_properties,
        oil_density_grid=new_oil_density_grid,
        oil_effective_density_grid=new_oil_effective_density_grid,
        water_density_grid=new_water_density_grid,
        gas_density_grid=new_gas_density_grid,
    )
    return updated_fluid_properties


def update_pvt_properties(
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
) -> FluidProperties[ThreeDimensions]:
    """
    Updates PVT fluid properties across the simulation grid using the current pressure and temperature values.
    This function recalculates the fluid PVT properties in a physically consistent sequence:

    ```markdown
    ┌────────────┐
    │  PRESSURE  │
    └────┬───────┘
         ▼
    ┌─────────────┐
    │  TEMPERATURE│
    └────┬───────┘
         ▼
    ┌──────────────┐
    │ GAS PROPERTIES│
    └────┬─────────┘
         ▼
    ┌──────────────────┐
    │ WATER PROPERTIES │
    └────┬─────────────┘
         ▼
    ┌────────────────────────────────────────────────────────────┐
    │ OIL PROPERTIES                                             │
    │  • Compute oil specific gravity and API gravity            │
    │  • Recalculate bubble point pressure (Pb)                  │
    │  • If pressure < Pb: recompute GOR (Rs) using Vazquez-Beggs│
    │  • Compute FVF using pressure, Rs, Pb                      │
    │  • Then compute oil compressibility, density, viscosity    │
    └────────────────────────────────────────────────────────────┘

    # GAS PROPERTIES
    - Computes gas gravity from density.
    - Uses gas gravity to derive molecular weight.
    - Computes gas z-factor (compressibility factor) from pressure, temperature, and gas gravity.
    - Updates:
        - Formation volume factor (Bg)
        - Compressibility (Cg)
        - Density (ρg)
        - Viscosity (μg)

    # WATER PROPERTIES
    - Computes gas solubility in water (Rs_w) based on salinity, pressure, and temperature.
    - Determines water bubble point pressure from solubility and salinity.
    - Computes gas-free water FVF (for use in density calculation).
    - Updates:
        - Water compressibility (Cw) considering dissolved gas
        - Water FVF (Bw)
        - Water density (ρw)
        - Water viscosity (μw)

    # OIL PROPERTIES
    - Computes oil specific gravity and API gravity from base density.
    - Recalculates bubble point pressure (Pb) using API, temperature, and gas gravity.
    - Determines GOR:
        • If current pressure < Pb: compute GOR using Vazquez-Beggs correlation.
        • If current pressure ≥ Pb: GOR = GOR at Pb (Rs = Rs_b).
    - Computes:
        - Oil formation volume factor (Bo) using pressure, Pb, GOR, and gravity.
        - Oil compressibility (Co) using updated GOR and Pb.
        - Oil density (ρo) using GOR and Bo.
        - Oil viscosity (μo) using Rs, Pb, and API.
    ```

    :param fluid_properties: Current fluid property grids (pressure, temperature, salinity, densities, etc.)
    :param wells: Current wells configuration (used for fluid type information).
    :param miscibility_model: The miscibility model used in the simulation.
    :return: Updated FluidProperties object with recalculated gas, water, and oil properties.
    """
    # GAS PROPERTIES
    gas_compressibility_factor_grid = build_gas_compressibility_factor_grid(
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
    )
    new_gas_formation_volume_factor_grid = build_gas_formation_volume_factor_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )
    new_gas_compressibility_grid = build_gas_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )
    new_gas_density_grid = build_gas_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gas_compressibility_factor_grid=gas_compressibility_factor_grid,
    )
    new_gas_viscosity_grid = build_gas_viscosity_grid(
        temperature_grid=fluid_properties.temperature_grid,
        gas_density_grid=new_gas_density_grid,
        gas_molecular_weight_grid=fluid_properties.gas_molecular_weight_grid,
    )

    # WATER PROPERTIES
    gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        gas=fluid_properties.reservoir_gas,
    )
    new_water_bubble_point_pressure_grid = build_water_bubble_point_pressure_grid(
        temperature_grid=fluid_properties.temperature_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
    )
    gas_free_water_formation_volume_factor_grid = (
        build_gas_free_water_formation_volume_factor_grid(
            pressure_grid=fluid_properties.pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
        )
    )
    new_water_compressibility_grid = build_water_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
        gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )
    new_water_density_grid = build_water_density_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
    )
    new_water_formation_volume_factor_grid = build_water_formation_volume_factor_grid(
        water_density_grid=new_water_density_grid,  # Use new density here
        salinity_grid=fluid_properties.water_salinity_grid,
    )
    new_water_viscosity_grid = build_water_viscosity_grid(
        temperature_grid=fluid_properties.temperature_grid,
        salinity_grid=fluid_properties.water_salinity_grid,
        pressure_grid=fluid_properties.pressure_grid,
    )

    # OIL PROPERTIES (tricky due to bubble point)
    # Make sure to always compute the oil bubble point pressure grid
    # before the gas to oil ratio grid, as the latter depends on the former.

    # Step 1: Compute NEW bubble point using CURRENT Rs
    new_oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        temperature_grid=fluid_properties.temperature_grid,
        solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
    )

    # Step 2: Compute Rs at NEW bubble point
    gor_at_bubble_point_pressure_grid = build_solution_gas_to_oil_ratio_grid(
        pressure_grid=new_oil_bubble_point_pressure_grid,  # New bubble point here
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,  # Use same NEW bubble point here
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
    )
    # Step 3: Compute NEW Rs at current pressure
    new_solution_gas_to_oil_ratio_grid = build_solution_gas_to_oil_ratio_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,  # New bubble point here
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,  # GOR at new bubble point here
    )
    # Step 4: Compute oil FVF (may use lagged compressibility - acceptable)
    # Oil FVF does not depend necessarily on the new compressibility grid,
    # so we can use the old one (compressibility changes are small, hence, it can be lagged).
    # FVF is a function of pressure and phase behavior. Only when pressure changes,
    # does FVF need to be recalculated.
    new_oil_formation_volume_factor_grid = build_oil_formation_volume_factor_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        oil_compressibility_grid=fluid_properties.oil_compressibility_grid,
    )
    # Step 5: Compute oil compressibility
    new_oil_compressibility_grid = build_oil_compressibility_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
        gas_formation_volume_factor_grid=new_gas_formation_volume_factor_grid,
        oil_formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
    )
    # Step 6: Compute oil density and viscosity
    new_oil_density_grid = build_live_oil_density_grid(
        oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        gas_gravity_grid=fluid_properties.gas_gravity_grid,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
    )
    new_oil_effective_density_grid = new_oil_density_grid
    new_oil_viscosity_grid = build_oil_viscosity_grid(
        pressure_grid=fluid_properties.pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
    )
    new_oil_effective_viscosity_grid = new_oil_viscosity_grid
    # If there are miscible injections, update the effective oil viscosity and density grids
    if miscibility_model != "immiscible":
        for well in wells.injection_wells:
            injected_fluid = well.injected_fluid
            if injected_fluid and injected_fluid.is_miscible:
                injected_fluid_viscosity_grid = np.vectorize(
                    injected_fluid.get_viscosity, otypes=[get_dtype()]
                )(
                    pressure=fluid_properties.pressure_grid,
                    temperature=fluid_properties.temperature_grid,
                )
                # Update effective oil viscosity grid using Todd-Longstaff model
                new_oil_effective_viscosity_grid = build_oil_effective_viscosity_grid(
                    oil_viscosity_grid=new_oil_effective_viscosity_grid,
                    solvent_viscosity_grid=injected_fluid_viscosity_grid,
                    solvent_concentration_grid=fluid_properties.solvent_concentration_grid,
                    base_omega=injected_fluid.todd_longstaff_omega,
                    pressure_grid=fluid_properties.pressure_grid,
                    minimum_miscibility_pressure=injected_fluid.minimum_miscibility_pressure,
                    transition_width=injected_fluid.miscibility_transition_width,
                )
                new_oil_effective_density_grid = build_oil_effective_density_grid(
                    oil_density_grid=new_oil_density_grid,
                    solvent_density_grid=injected_fluid_viscosity_grid,
                    oil_viscosity_grid=new_oil_effective_viscosity_grid,
                    solvent_viscosity_grid=injected_fluid_viscosity_grid,
                    solvent_concentration_grid=fluid_properties.solvent_concentration_grid,
                    base_omega=injected_fluid.todd_longstaff_omega,
                    pressure_grid=fluid_properties.pressure_grid,
                    minimum_miscibility_pressure=injected_fluid.minimum_miscibility_pressure,
                    transition_width=injected_fluid.miscibility_transition_width,
                )

    # Finally, update the fluid properties with all the new grids
    updated_fluid_properties = attrs.evolve(
        fluid_properties,
        solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        oil_bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
        water_bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
        oil_viscosity_grid=new_oil_viscosity_grid,
        oil_effective_viscosity_grid=new_oil_effective_viscosity_grid,
        water_viscosity_grid=new_water_viscosity_grid,
        gas_viscosity_grid=new_gas_viscosity_grid,
        oil_formation_volume_factor_grid=new_oil_formation_volume_factor_grid,
        water_formation_volume_factor_grid=new_water_formation_volume_factor_grid,
        gas_formation_volume_factor_grid=new_gas_formation_volume_factor_grid,
        oil_compressibility_grid=new_oil_compressibility_grid,
        water_compressibility_grid=new_water_compressibility_grid,
        gas_compressibility_grid=new_gas_compressibility_grid,
        oil_density_grid=new_oil_density_grid,
        oil_effective_density_grid=new_oil_effective_density_grid,
        water_density_grid=new_water_density_grid,
        gas_density_grid=new_gas_density_grid,
    )
    return updated_fluid_properties
