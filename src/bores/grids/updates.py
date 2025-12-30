from bores.pvt.tables import PVTTables
import attrs
import typing
import numpy as np

from bores._precision import get_dtype
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
    build_water_bubble_point_pressure_grid,
    build_water_compressibility_grid,
    build_water_density_grid,
    build_water_formation_volume_factor_grid,
    build_water_viscosity_grid,
)
from bores.models import FluidProperties, SaturationHistory, RockProperties
from bores.types import MiscibilityModel, NDimensionalGrid, ThreeDimensions
from bores.wells import Wells
from bores.grids.rock_fluid import build_effective_residual_saturation_grids


def update_pvt_grids(
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    pvt_tables: typing.Optional[PVTTables] = None,
) -> FluidProperties[ThreeDimensions]:
    """
    Updates PVT fluid properties grids using the current pressure and temperature values.
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
    :param
    :return: Updated FluidProperties object with recalculated gas, water, and oil properties.
    """
    pressure_grid = fluid_properties.pressure_grid
    temperature_grid = fluid_properties.temperature_grid
    if pvt_tables is None:
        # GAS PROPERTIES
        gas_compressibility_factor_grid = build_gas_compressibility_factor_grid(
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
        )
        new_gas_formation_volume_factor_grid = build_gas_formation_volume_factor_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )
        new_gas_compressibility_grid = build_gas_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )
        new_gas_density_grid = build_gas_density_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            gas_compressibility_factor_grid=gas_compressibility_factor_grid,
        )
        new_gas_viscosity_grid = build_gas_viscosity_grid(
            temperature_grid=temperature_grid,
            gas_density_grid=new_gas_density_grid,
            gas_molecular_weight_grid=fluid_properties.gas_molecular_weight_grid,
        )

        # WATER PROPERTIES
        gas_solubility_in_water_grid = build_gas_solubility_in_water_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            salinity_grid=fluid_properties.water_salinity_grid,
            gas=fluid_properties.reservoir_gas,
        )
        new_water_bubble_point_pressure_grid = build_water_bubble_point_pressure_grid(
            temperature_grid=temperature_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            salinity_grid=fluid_properties.water_salinity_grid,
        )
        gas_free_water_formation_volume_factor_grid = (
            build_gas_free_water_formation_volume_factor_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
            )
        )
        new_water_compressibility_grid = build_water_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_water_bubble_point_pressure_grid,
            gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
        )
        new_water_density_grid = build_water_density_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            salinity_grid=fluid_properties.water_salinity_grid,
            gas_solubility_in_water_grid=gas_solubility_in_water_grid,
            gas_free_water_formation_volume_factor_grid=gas_free_water_formation_volume_factor_grid,
        )
        new_water_formation_volume_factor_grid = (
            build_water_formation_volume_factor_grid(
                water_density_grid=new_water_density_grid,  # Use new density here
                salinity_grid=fluid_properties.water_salinity_grid,
            )
        )
        new_water_viscosity_grid = build_water_viscosity_grid(
            temperature_grid=temperature_grid,
            salinity_grid=fluid_properties.water_salinity_grid,
            pressure_grid=pressure_grid,
        )
    else:
        # GAS PROPERTIES
        gas_compressibility_factor_grid = pvt_tables.gas_compressibility_factor(
            pressure=pressure_grid, temperature=temperature_grid
        )
        new_gas_formation_volume_factor_grid = pvt_tables.gas_formation_volume_factor(
            pressure=pressure_grid, temperature=temperature_grid
        )
        new_gas_compressibility_grid = pvt_tables.gas_compressibility(
            pressure=pressure_grid, temperature=temperature_grid
        )
        new_gas_density_grid = pvt_tables.gas_density(
            pressure=pressure_grid, temperature=temperature_grid
        )
        new_gas_viscosity_grid = pvt_tables.gas_viscosity(
            pressure=pressure_grid, temperature=temperature_grid
        )

        # WATER PROPERTIES
        water_salinity_grid = fluid_properties.water_salinity_grid
        gas_solubility_in_water_grid = pvt_tables.gas_solubility_in_water(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )
        new_water_bubble_point_pressure_grid = pvt_tables.water_bubble_point_pressure(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )
        gas_free_water_formation_volume_factor_grid = (
            build_gas_free_water_formation_volume_factor_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
            )
        )
        new_water_compressibility_grid = pvt_tables.water_compressibility(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )
        new_water_density_grid = pvt_tables.water_density(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )
        new_water_formation_volume_factor_grid = (
            pvt_tables.water_formation_volume_factor(
                pressure=pressure_grid,
                temperature=temperature_grid,
                salinity=water_salinity_grid,
            )
        )
        new_water_viscosity_grid = pvt_tables.water_viscosity(
            pressure=pressure_grid,
            temperature=temperature_grid,
            salinity=water_salinity_grid,
        )

    # OIL PROPERTIES (tricky due to bubble point)
    # Make sure to always compute the oil bubble point pressure grid
    # before the gas to oil ratio grid, as the latter depends on the former.

    if pvt_tables is None:
        # Step 1: Compute NEW bubble point using CURRENT Rs
        new_oil_bubble_point_pressure_grid = build_oil_bubble_point_pressure_grid(
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
            temperature_grid=temperature_grid,
            solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
        )

        # Step 2: Compute Rs at NEW bubble point
        gor_at_bubble_point_pressure_grid = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=new_oil_bubble_point_pressure_grid,  # New bubble point here
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,  # Use same NEW bubble point here
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
        )
        # Step 3: Compute NEW Rs at current pressure
        new_solution_gas_to_oil_ratio_grid = build_solution_gas_to_oil_ratio_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
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
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
            oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
            oil_compressibility_grid=fluid_properties.oil_compressibility_grid,
        )
        # Step 5: Compute oil compressibility
        new_oil_compressibility_grid = build_oil_compressibility_grid(
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
            oil_api_gravity_grid=fluid_properties.oil_api_gravity_grid,
            gas_gravity_grid=fluid_properties.gas_gravity_grid,
            gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
            gas_formation_volume_factor_grid=new_gas_formation_volume_factor_grid,  # type: ignore
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
            pressure_grid=pressure_grid,
            temperature_grid=temperature_grid,
            bubble_point_pressure_grid=new_oil_bubble_point_pressure_grid,
            oil_specific_gravity_grid=fluid_properties.oil_specific_gravity_grid,
            solution_gas_to_oil_ratio_grid=new_solution_gas_to_oil_ratio_grid,
            gor_at_bubble_point_pressure_grid=gor_at_bubble_point_pressure_grid,
        )
    else:
        # Step 1: Compute NEW bubble point using CURRENT Rs
        new_oil_bubble_point_pressure_grid = pvt_tables.oil_bubble_point_pressure(
            temperature=temperature_grid,
            solution_gor=fluid_properties.solution_gas_to_oil_ratio_grid,
        )

        # Step 2: Compute NEW Rs at current pressure
        new_solution_gas_to_oil_ratio_grid = pvt_tables.solution_gas_to_oil_ratio(
            pressure=pressure_grid,
            temperature=temperature_grid,
            solution_gor=fluid_properties.solution_gas_to_oil_ratio_grid,
        )
        # Step 4: Compute oil FVF (may use lagged compressibility - acceptable)
        # Oil FVF does not depend necessarily on the new compressibility grid,
        # so we can use the old one (compressibility changes are small, hence, it can be lagged).
        # FVF is a function of pressure and phase behavior. Only when pressure changes,
        # does FVF need to be recalculated.
        new_oil_formation_volume_factor_grid = pvt_tables.oil_formation_volume_factor(
            pressure=pressure_grid,
            temperature=temperature_grid,
            solution_gor=new_solution_gas_to_oil_ratio_grid,
        )
        # Step 5: Compute oil compressibility
        new_oil_compressibility_grid = pvt_tables.oil_compressibility(
            pressure=pressure_grid,
            temperature=temperature_grid,
        )
        # Step 6: Compute oil density and viscosity
        new_oil_density_grid = pvt_tables.oil_density(
            pressure=pressure_grid,
            temperature=temperature_grid,
        )
        new_oil_effective_density_grid = new_oil_density_grid
        new_oil_viscosity_grid = pvt_tables.oil_viscosity(
            pressure=pressure_grid,
            temperature=temperature_grid,
            solution_gor=new_solution_gas_to_oil_ratio_grid,
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
                    pressure=pressure_grid,
                    temperature=temperature_grid,
                )
                # Update effective oil viscosity grid using Todd-Longstaff model
                new_oil_effective_viscosity_grid = build_oil_effective_viscosity_grid(
                    oil_viscosity_grid=new_oil_effective_viscosity_grid,  # type: ignore
                    solvent_viscosity_grid=injected_fluid_viscosity_grid,
                    solvent_concentration_grid=fluid_properties.solvent_concentration_grid,
                    base_omega=injected_fluid.todd_longstaff_omega,
                    pressure_grid=pressure_grid,
                    minimum_miscibility_pressure=injected_fluid.minimum_miscibility_pressure,
                    transition_width=injected_fluid.miscibility_transition_width,
                )
                new_oil_effective_density_grid = build_oil_effective_density_grid(
                    oil_density_grid=new_oil_density_grid,  # type: ignore
                    solvent_density_grid=injected_fluid_viscosity_grid,
                    oil_viscosity_grid=new_oil_effective_viscosity_grid,
                    solvent_viscosity_grid=injected_fluid_viscosity_grid,
                    solvent_concentration_grid=fluid_properties.solvent_concentration_grid,
                    base_omega=injected_fluid.todd_longstaff_omega,
                    pressure_grid=pressure_grid,
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


def update_residual_saturation_grids(
    rock_properties: RockProperties[ThreeDimensions],
    saturation_history: SaturationHistory[ThreeDimensions],
    water_saturation_grid: NDimensionalGrid[ThreeDimensions],
    gas_saturation_grid: NDimensionalGrid[ThreeDimensions],
    residual_oil_drainage_ratio_water_flood: float = 0.6,  # Sorw_drainage = 0.6 × Sorw_imbibition
    residual_oil_drainage_ratio_gas_flood: float = 0.6,  # Sorg_drainage = 0.6 × Sorg_imbibition
    residual_gas_drainage_ratio: float = 0.5,  # Sgr_drainage = 0.5 × Sgr_imbibition
    tolerance: float = 1e-6,
) -> typing.Tuple[RockProperties[ThreeDimensions], SaturationHistory[ThreeDimensions]]:
    """
    Updates the effective residual saturation grids based on current displacement regimes
    (drainage or imbibition) determined from saturation history and current saturations.

    :param rock_properties: Current rock properties including residual saturations
    :param saturation_history: Current saturation history including max saturations and imbibition flags
    :param water_saturation_grid: Current water saturation grid
    :param gas_saturation_grid: Current gas saturation grid
    :param residual_oil_drainage_ratio_water_flood: Ratio to compute oil drainage residual from imbibition value.
    :param residual_gas_drainage_ratio: Ratio to compute gas drainage residual from imbibition value.
    :param residual_oil_drainage_ratio_gas_flood: Ratio to compute oil drainage residual from gas flooding imbibition value.
    :param tolerance: Tolerance to determine significant saturation changes.
    :return: Tuple of updated `RockProperties` and `SaturationHistory` with new effective residual saturations
    """
    residual_oil_saturation_water_grid = (
        rock_properties.residual_oil_saturation_water_grid
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    max_water_saturation_grid = saturation_history.max_water_saturation_grid
    max_gas_saturation_grid = saturation_history.max_gas_saturation_grid
    water_imbibition_flag_grid = saturation_history.water_imbibition_flag_grid
    gas_imbibition_flag_grid = saturation_history.gas_imbibition_flag_grid

    (
        new_max_water_saturation_grid,
        new_max_gas_saturation_grid,
        effective_residual_oil_saturation_water_grid,
        effective_residual_oil_saturation_gas_grid,
        effective_residual_gas_saturation_grid,
        new_water_imbibition_flag_grid,
        new_gas_imbibition_flag_grid,
    ) = build_effective_residual_saturation_grids(
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        max_water_saturation_grid=max_water_saturation_grid,
        max_gas_saturation_grid=max_gas_saturation_grid,
        water_imbibition_flag_grid=water_imbibition_flag_grid,
        gas_imbibition_flag_grid=gas_imbibition_flag_grid,
        residual_oil_drainage_ratio_water_flood=residual_oil_drainage_ratio_water_flood,
        residual_oil_drainage_ratio_gas_flood=residual_oil_drainage_ratio_gas_flood,
        residual_gas_drainage_ratio=residual_gas_drainage_ratio,
        tolerance=tolerance,
    )

    updated_rock_properties = attrs.evolve(
        rock_properties,
        residual_oil_saturation_water_grid=effective_residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=effective_residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=effective_residual_gas_saturation_grid,
    )
    updated_saturation_history = attrs.evolve(
        saturation_history,
        max_water_saturation_grid=new_max_water_saturation_grid,
        max_gas_saturation_grid=new_max_gas_saturation_grid,
        water_imbibition_flag_grid=new_water_imbibition_flag_grid,
        gas_imbibition_flag_grid=new_gas_imbibition_flag_grid,
    )
    return updated_rock_properties, updated_saturation_history
