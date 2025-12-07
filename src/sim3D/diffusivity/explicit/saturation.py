import itertools
import logging
import typing

import numpy as np

from sim3D.constants import c
from sim3D.diffusivity.base import (
    EvolutionResult,
    _warn_injector_is_producing,
    _warn_producer_is_injecting,
)
from sim3D.models import FluidProperties, RockFluidProperties, RockProperties
from sim3D.types import (
    CapillaryPressureGrids,
    FluidPhase,
    Options,
    RelativeMobilityGrids,
    SupportsSetItem,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from sim3D.wells import Wells

__all__ = ["evolve_saturation_explicitly", "evolve_miscible_saturation_explicitly"]

logger = logging.getLogger(__name__)


"""
Explicit finite difference formulation for saturation transport in a 3D reservoir
(immiscible three-phase flow: oil, water, and gas with slightly compressible fluids):

The governing equation for saturation evolution is the conservation of mass with advection:

    ∂S/∂t * (φ * V_cell) = -∇ · (f_x * v ) * V_cell + q_x * V_cell

Where:
    ∂S/∂t * φ * V_cell = Accumulation term (change in phase saturation) (ft³/day)
    ∇ · (f_x * v ) * V_cell = Advection term (Darcy velocity * fractional flow) (ft³/day)
    q_x * V_cell = Source/sink term for the phase (injection/production) (ft³/day)

Assuming constant cell volume, the equation simplifies to:

        ∂S/∂t * φ = -∇ · (f_x * v ) + q_x

where:
    S = phase saturation (fraction)
    φ = porosity (fraction)
    V_cell = cell bulk volume = Δx * Δy * Δz (ft³)
    f_x = phase fractional flow function (depends on S_x)
    v = Darcy velocity vector [v_x, v_y, v_z] (ft/day)
    q_x = source/sink term per unit volume (1/day)

Discretization:

Time: Forward Euler
    ∂S/∂t ≈ (Sⁿ⁺¹_ijk - Sⁿ_ijk) / Δt

Space: First-order upwind scheme:

    ∇ · (f_x * v ) ≈ [(F_x_east + F_x_west)/Δx + (F_y_north + F_y_south)/Δy + (F_z_top + F_z_bottom)/Δz]

    Sⁿ⁺¹_ijk = Sⁿ_ijk + Δt / (φ * V_cell) * [
        (F_x_east + F_x_west) + (F_y_north + F_y_south) + (F_z_top + F_z_bottom) + q_x_ijk * V_cell
    ]

    F_dir = phase volumetric flux at face in direction `dir` (ft³/day)

Volumetric phase flux at face F_dir is computed as:
    F_dir = f_x(S_upwind) * v_dir * A_face (ft³/day)
    f_x = phase fractional flow = [k_r(S_upwind) / μ] / λ_total
    v_dir = Darcy velocity component in direction `dir` (ft/day)
    v_dir = λ_total * ∂∅/∂dir
    ∂∅/∂dir = (∅neighbour - ∅current) / ΔL_dir

    where; 
    λ_total = Σ [k_r(S_upwind) / μ] for all phases
    A_face = face area perpendicular to flow direction (ft²)
    ∅ = phase potential (pressure + gravity effects + capillary effects)


Upwind saturation S_upwind is selected based on the sign of v_dir:
    - If v_dir < 0 → S_upwind = Sⁿ_current (flow from current cell)
    - If v_dir > 0 → S_upwind = Sⁿ_neighbour (flow from neighbour into current cell)

Velocity Components:
    v_x = λ_total * ∂p/∂x
    v_y = λ_total * ∂p/∂y
    v_z = λ_total * ∂p/∂z

Note: This is taking the convention that flux from cell to neighbour is negative.
and flux from neighbour to cell is positive.

Where:
    λ_total = Σ [k_r(S_upwind) / μ] for all phases
    f_x = phase fractional flow = [k_r(S_upwind) / μ] / λ_total
    k_r = relative permeability of the phase(s)
    ∂p = Pressure/Potential difference in a specific direction

Variables:
    Sⁿ_ijk = saturation at cell (i,j,k) at time step n
    Sⁿ⁺¹_ijk = updated saturation
    φ = porosity
    Δx, Δy, Δz = cell dimensions (ft)
    A_x = Δy * Δz (face area for x-direction flow)
    A_y = Δx * Δz (face area for y-direction flow)
    A_z = Δx * Δy (face area for z-direction flow)
    q_x_ijk = phase source/sink rate per unit volume (1/day)
    F_x, F_y, F_z = phase volumetric fluxes (ft³/day)

Assumptions:
- Darcy flow
- No dispersion or diffusion (purely advective)
- Saturation-dependent fractional flow model (Corey, Brooks-Corey, etc.)
- Time step satisfies CFL condition

Stability (CFL) condition:

    max(|v_x|/Δx + |v_y|/Δy + |v_z|/Δz) * Δt / φ ≤ 1

Notes:
- Pressure field must be computed before solving saturation.
- Upwind saturation is selected based on local flow direction.
- A single saturation equation must be solved per phase (water, oil, gas).
"""


def evolve_saturation_explicitly(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    options: Options,
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
) -> EvolutionResult[
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]
]:
    """
    Computes the new/updated saturation distribution for water, oil, and gas
    across the reservoir grid using an explicit upwind finite difference method.

    This function simulates three-phase immiscible flow, considering pressure
    gradients (including capillary pressure effects) and relative permeabilities.

    :param cell_dimension: Tuple representing the dimensions of each grid cell (cell_size_x, cell_size_y) in feet (ft).
    :param thickness_grid: N-Dimensional numpy array representing the height of each cell in the grid (ft).
    :param elevation_grid: N-Dimensional numpy array representing the elevation of each cell in the grid (ft).
    :param time_step: Current time step index (starting from 0).
    :param time_step_size: Time step duration in seconds for the simulation.
    :param rock_properties: `RockProperties` object containing rock physical properties.
    :param fluid_properties: `FluidProperties` object containing fluid physical properties,
        including current pressure and saturation grids.
    :param rock_fluid_properties: `RockFluidProperties` object containing properties
        that depend on both rock and fluid characteristics.

    :param wells: ``Wells`` object containing information about injection and production wells.
    :param options: Simulation options and parameters.
    :param injection_grid: Object supporting setitem to set cell injection rates for each phase in ft³/day.
    :param production_grid: Object supporting setitem to set cell production rates for each phase in ft³/day.
    :return: A tuple of N-Dimensional numpy arrays representing the updated saturation distributions
        for water, oil, and gas, respectively.
        (updated_water_saturation_grid, updated_oil_saturation_grid, updated_gas_saturation_grid)
    """
    # Extract properties from provided objects for clarity and convenience
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    current_oil_pressure_grid = (
        fluid_properties.pressure_grid
    )  # This is P_oil or Pⁿ_{i,j}
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Determine grid dimensions and cell sizes
    cell_count_x, cell_count_y, cell_count_z = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    # Compute mobility grids for x, y, z directions
    # λ_x = 0.001127 * k_abs * (kr / mu) (mD/cP to ft²/psi.day)
    water_mobility_grid_x = (
        absolute_permeability.x
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_x = (
        absolute_permeability.x
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_x = (
        absolute_permeability.x
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    water_mobility_grid_y = (
        absolute_permeability.y
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_y = (
        absolute_permeability.y
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_y = (
        absolute_permeability.y
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    water_mobility_grid_z = (
        absolute_permeability.z
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_z = (
        absolute_permeability.z
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_z = (
        absolute_permeability.z
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    # Create new grids for updated saturations (time 'n+1')
    updated_water_saturation_grid = current_water_saturation_grid.copy()
    updated_oil_saturation_grid = current_oil_saturation_grid.copy()
    updated_gas_saturation_grid = current_gas_saturation_grid.copy()

    temperature_grid = fluid_properties.temperature_grid
    # Iterate over each interior cell to compute saturation evolution
    # # Assume boundary cells are added via padding for boundary conditions application purposes
    # Thus, we iterate from 1 to N-1 in each dimension
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_temperature = temperature_grid[i, j, k]
        cell_thickness = thickness_grid[i, j, k]
        cell_total_volume = cell_size_x * cell_size_y * cell_thickness
        # Current cell properties
        cell_porosity = porosity_grid[i, j, k]
        # Cell pore volume = φ * V_cell
        cell_pore_volume = cell_total_volume * cell_porosity
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]

        flux_configurations = {
            "x": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_x,
                    "oil_mobility_grid": oil_mobility_grid_x,
                    "gas_mobility_grid": gas_mobility_grid_x,
                },
                "neighbours": [
                    (i + 1, j, k),
                    (i - 1, j, k),
                ],  # East and West neighbours
                "flow_area": cell_size_y * cell_thickness,  # A_x = Δy * Δz
                "flow_length": cell_size_x,  # Δx
            },
            "y": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_y,
                    "oil_mobility_grid": oil_mobility_grid_y,
                    "gas_mobility_grid": gas_mobility_grid_y,
                },
                "neighbours": [
                    (i, j - 1, k),
                    (i, j + 1, k),
                ],  # North and South neighbours
                "flow_area": cell_size_x * cell_thickness,  # A_y = Δx * Δz
                "flow_length": cell_size_y,  # Δy
            },
            "z": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_z,
                    "oil_mobility_grid": oil_mobility_grid_z,
                    "gas_mobility_grid": gas_mobility_grid_z,
                },
                "neighbours": [
                    (i, j, k - 1),
                    (i, j, k + 1),
                ],  # Top and Bottom neighbours
                "flow_area": cell_size_x * cell_size_y,  # A_z = Δx * Δy
                "flow_length": cell_thickness,  # Δz
            },
        }

        net_water_flux = 0.0
        net_oil_flux = 0.0
        net_gas_flux = 0.0
        for _, config in flux_configurations.items():
            flow_area = typing.cast(float, config["flow_area"])
            flow_length = typing.cast(float, config["flow_length"])
            mobility_grids = typing.cast(
                typing.Dict[str, ThreeDimensionalGrid], config["mobility_grids"]
            )
            for neighbour in config["neighbours"]:  # type: ignore
                # Compute fluxes from neighbour
                (
                    water_flux,
                    oil_flux,
                    gas_flux,
                ) = compute_phase_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=neighbour,
                    flow_area=flow_area,
                    flow_length=flow_length,
                    oil_pressure_grid=current_oil_pressure_grid,
                    **mobility_grids,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                )
                # Update the net fluxes
                net_water_flux += water_flux
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux

        # Compute Source/Sink Term (WellParameters) - q * V (ft³/day)
        injection_well, production_well = wells[i, j, k]
        cell_water_injection_rate = 0.0
        cell_water_production_rate = 0.0
        cell_oil_injection_rate = 0.0
        cell_oil_production_rate = 0.0
        cell_gas_injection_rate = 0.0
        cell_gas_production_rate = 0.0
        permeability = (
            absolute_permeability.x[i, j, k],
            absolute_permeability.y[i, j, k],
            absolute_permeability.z[i, j, k],
        )
        oil_pressure = current_oil_pressure_grid[i, j, k]
        if (
            injection_well is not None
            and injection_well.is_open
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            injected_phase = injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                phase_mobility = gas_relative_mobility_grid[i, j, k]
                compressibility_kwargs = {}
            else:
                phase_mobility = water_relative_mobility_grid[i, j, k]
                compressibility_kwargs = {
                    "bubble_point_pressure": fluid_properties.oil_bubble_point_pressure_grid[
                        i, j, k
                    ],
                    "gas_formation_volume_factor": fluid_properties.gas_formation_volume_factor_grid[
                        i, j, k
                    ],
                    "gas_solubility_in_water": fluid_properties.gas_solubility_in_water_grid[
                        i, j, k
                    ],
                }
            fluid_compressibility = injected_fluid.get_compressibility(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                **compressibility_kwargs,
            )
            fluid_formation_volume_factor = injected_fluid.get_formation_volume_factor(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
            )

            use_pseudo_pressure = (
                options.use_pseudo_pressure and injected_phase == FluidPhase.GAS
            )
            well_index = injection_well.get_well_index(
                interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                permeability=permeability,
                skin_factor=injection_well.skin_factor,
            )
            # The rate returned here is in bbls/day for oil and water, and ft³/day for gas
            # Since phase relative mobility does not include formation volume factor
            cell_injection_rate = injection_well.get_flow_rate(
                pressure=oil_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_mobility=phase_mobility,
                fluid=injected_fluid,
                fluid_compressibility=fluid_compressibility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=fluid_formation_volume_factor,
            )
            if cell_injection_rate < 0.0 and options.warn_rates_anomalies:
                _warn_injector_is_producing(
                    injection_rate=cell_injection_rate,
                    well_name=injection_well.name,
                    cell=(i, j, k),
                    time=time_step * time_step_size,
                    rate_unit="ft³/day"
                    if injected_phase == FluidPhase.GAS
                    else "bbls/day",
                )

            if injected_phase == FluidPhase.GAS:
                cell_gas_injection_rate = cell_injection_rate
                # Record gas injection rate for the cell
                if injection_grid is not None:
                    injection_grid[i, j, k] = (0.0, 0.0, cell_gas_injection_rate)

            else:
                cell_water_injection_rate = cell_injection_rate * c.BBL_TO_FT3
                # Record water injection rate for the cell
                if injection_grid is not None:
                    injection_grid[i, j, k] = (0.0, cell_water_injection_rate, 0.0)

        if production_well is not None and production_well.is_open:
            water_formation_volume_factor_grid = (
                fluid_properties.water_formation_volume_factor_grid
            )
            oil_formation_volume_factor_grid = (
                fluid_properties.oil_formation_volume_factor_grid
            )
            gas_formation_volume_factor_grid = (
                fluid_properties.gas_formation_volume_factor_grid
            )
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = gas_relative_mobility_grid[i, j, k]
                    fluid_compressibility = gas_compressibility_grid[i, j, k]
                    fluid_formation_volume_factor = gas_formation_volume_factor_grid[
                        i, j, k
                    ]
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = water_relative_mobility_grid[i, j, k]
                    fluid_compressibility = water_compressibility_grid[i, j, k]
                    fluid_formation_volume_factor = water_formation_volume_factor_grid[
                        i, j, k
                    ]
                else:
                    phase_mobility = oil_relative_mobility_grid[i, j, k]
                    fluid_compressibility = oil_compressibility_grid[i, j, k]
                    fluid_formation_volume_factor = oil_formation_volume_factor_grid[
                        i, j, k
                    ]

                use_pseudo_pressure = (
                    options.use_pseudo_pressure and produced_phase == FluidPhase.GAS
                )
                well_index = production_well.get_well_index(
                    interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                    permeability=permeability,
                    skin_factor=production_well.skin_factor,
                )
                # The rate returned here is in bbls/day for oil and water, and ft³/day for gas
                # Since phase relative mobility does not include formation volume factor
                production_rate = production_well.get_flow_rate(
                    pressure=oil_pressure,
                    temperature=cell_temperature,
                    well_index=well_index,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                    fluid_compressibility=fluid_compressibility,
                    use_pseudo_pressure=use_pseudo_pressure,
                    formation_volume_factor=fluid_formation_volume_factor,
                )
                if production_rate > 0.0 and options.warn_rates_anomalies:
                    _warn_producer_is_injecting(
                        production_rate=production_rate,
                        well_name=production_well.name,
                        cell=(i, j, k),
                        time=time_step * time_step_size,
                        rate_unit="ft³/day"
                        if produced_phase == FluidPhase.GAS
                        else "bbls/day",
                    )

                if produced_fluid.phase == FluidPhase.GAS:
                    cell_gas_production_rate += production_rate
                elif produced_fluid.phase == FluidPhase.WATER:
                    cell_water_production_rate += production_rate * c.BBL_TO_FT3
                else:
                    cell_oil_production_rate += production_rate * c.BBL_TO_FT3

            # Record total production rate for the cell (all phases)
            if production_grid is not None:
                production_grid[i, j, k] = (
                    cell_oil_production_rate,
                    cell_water_production_rate,
                    cell_gas_production_rate,
                )

        # Compute the net volumetric rate for each phase. Just add injection and production rates (since production rates are negative)
        net_water_flow_rate = cell_water_injection_rate + cell_water_production_rate
        net_oil_flow_rate = cell_oil_injection_rate + cell_oil_production_rate
        net_gas_flow_rate = cell_gas_injection_rate + cell_gas_production_rate

        # Calculate total throughput (fluid moving through the cell)
        # Advective fluxes (already signed: positive = inflow, negative = outflow)
        water_inflow_advection = max(0.0, net_water_flux)
        oil_inflow_advection = max(0.0, net_oil_flux)
        gas_inflow_advection = max(0.0, net_gas_flux)

        water_outflow_advection = abs(min(0.0, net_water_flux))
        oil_outflow_advection = abs(min(0.0, net_oil_flux))
        gas_outflow_advection = abs(min(0.0, net_gas_flux))

        # Well flows (production is negative, injection is positive)
        water_inflow_well = max(0.0, net_water_flow_rate)
        oil_inflow_well = max(0.0, net_oil_flow_rate)
        gas_inflow_well = max(0.0, net_gas_flow_rate)

        water_outflow_well = abs(min(0.0, net_water_flow_rate))
        oil_outflow_well = abs(min(0.0, net_oil_flow_rate))
        gas_outflow_well = abs(min(0.0, net_gas_flow_rate))

        # Total throughput
        total_inflow = (
            water_inflow_advection
            + oil_inflow_advection
            + gas_inflow_advection
            + water_inflow_well
            + oil_inflow_well
            + gas_inflow_well
        )
        total_outflow = (
            water_outflow_advection
            + oil_outflow_advection
            + gas_outflow_advection
            + water_outflow_well
            + oil_outflow_well
            + gas_outflow_well
        )
        total_throughput = total_inflow + total_outflow
        # CFL check
        cfl_number = (total_throughput * time_step_in_days) / cell_pore_volume
        max_cfl_number = options.max_cfl_number.get(options.scheme, 1.0)
        if cfl_number > max_cfl_number:
            raise RuntimeError(
                f"CFL condition violated at cell ({i}, {j}, {k}) at timestep {time_step}: "
                f"CFL number {cfl_number:.4f} exceeds limit {max_cfl_number:.4f}. "
                f"Inflow = {total_inflow:.2f} ft³/day, Outflow = {total_outflow:.2f} ft³/day, "
                f"Pore volume = {cell_pore_volume:.2f} ft³. "
                f"Consider reducing time step size from {time_step_size} seconds."
            )

        # Calculate saturation changes for each phase
        # dS = Δt / (φ * V_cell) * [
        #     ([F_x_east - F_x_west] * Δy * Δz / Δx) + ([F_y_north - F_y_south] * Δx * Δz / Δy) + ([F_z_up - F_z_down] * Δx * Δy / Δz)
        #     + (q_x_ij * V)
        # ]
        # The change in saturation is (Net_Flux + Net_Well_Rate) * dt / Pore_Volume
        water_saturation_change = (
            (net_water_flux + net_water_flow_rate)
            * time_step_in_days
            / cell_pore_volume
        )
        oil_saturation_change = (
            (net_oil_flux + net_oil_flow_rate) * time_step_in_days / cell_pore_volume
        )
        gas_saturation_change = (
            (net_gas_flux + net_gas_flow_rate) * time_step_in_days / cell_pore_volume
        )
        # Update phase saturations
        updated_water_saturation_grid[i, j, k] += water_saturation_change
        updated_oil_saturation_grid[i, j, k] += oil_saturation_change
        updated_gas_saturation_grid[i, j, k] += gas_saturation_change

    # Apply saturation constraints and normalization across all cells
    # Clean up any remaining minor negative values caused by floating point errors
    updated_water_saturation_grid[updated_water_saturation_grid < 0.0] = 0.0
    updated_oil_saturation_grid[updated_oil_saturation_grid < 0.0] = 0.0
    updated_gas_saturation_grid[updated_gas_saturation_grid < 0.0] = 0.0

    total_saturation_grid = (
        updated_water_saturation_grid
        + updated_oil_saturation_grid
        + updated_gas_saturation_grid
    )
    # Only normalize cells where there is saturation present (total > `SATURATION_EPSILON` to avoid division by zero edge cases)
    mask = total_saturation_grid > c.SATURATION_EPSILON
    if np.any(mask):
        updated_water_saturation_grid[mask] /= total_saturation_grid[mask]
        updated_oil_saturation_grid[mask] /= total_saturation_grid[mask]
        updated_gas_saturation_grid[mask] /= total_saturation_grid[mask]
    return EvolutionResult(
        (
            updated_water_saturation_grid,
            updated_oil_saturation_grid,
            updated_gas_saturation_grid,
        ),
        scheme="explicit",
    )


def compute_phase_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    flow_area: float,
    flow_length: float,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_mobility_grid: ThreeDimensionalGrid,
    oil_mobility_grid: ThreeDimensionalGrid,
    gas_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    water_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    gas_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    elevation_grid: typing.Optional[ThreeDimensionalGrid] = None,
) -> typing.Tuple[float, float, float]:
    # Current cell pressures (P_oil is direct, P_water and P_gas derived)
    cell_oil_pressure = oil_pressure_grid[cell_indices]
    cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[cell_indices]

    # For the neighbour
    neighbour_oil_pressure = oil_pressure_grid[neighbour_indices]
    neighbour_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[
        neighbour_indices
    ]
    neighbour_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[
        neighbour_indices
    ]

    # Compute pressure differences
    oil_pressure_difference = neighbour_oil_pressure - cell_oil_pressure
    oil_water_capillary_pressure_difference = (
        neighbour_oil_water_capillary_pressure - cell_oil_water_capillary_pressure
    )
    water_pressure_difference = (
        oil_pressure_difference - oil_water_capillary_pressure_difference
    )
    gas_oil_capillary_pressure_difference = (
        neighbour_gas_oil_capillary_pressure - cell_gas_oil_capillary_pressure
    )
    gas_pressure_difference = (
        oil_pressure_difference + gas_oil_capillary_pressure_difference
    )

    if elevation_grid is not None:
        # Calculate the elevation difference between the neighbour and current cell
        elevation_delta = (
            elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
        )
    else:
        elevation_delta = 0.0

    # Determine the upwind densities and solubilities based on pressure difference
    # If pressure difference is positive (P_neighbour - P_current > 0), we use the neighbour's density
    if water_density_grid is not None:
        upwind_water_density = (
            water_density_grid[neighbour_indices]
            if water_pressure_difference > 0.0
            else water_density_grid[cell_indices]
        )
    else:
        upwind_water_density = 0.0

    if oil_density_grid is not None:
        upwind_oil_density = (
            oil_density_grid[neighbour_indices]
            if oil_pressure_difference > 0.0
            else oil_density_grid[cell_indices]
        )
    else:
        upwind_oil_density = 0.0

    if gas_density_grid is not None:
        upwind_gas_density = (
            gas_density_grid[neighbour_indices]
            if gas_pressure_difference > 0.0
            else gas_density_grid[cell_indices]
        )
    else:
        upwind_gas_density = 0.0

    # Computing the Darcy velocities (ft/day) for the three phases
    # v_x = λ_x * ∆P / Δx
    # For water: v_w = λ_w * [(P_oil - P_cow) + (upwind_ρ_water * g * Δz)] / ΔL
    water_gravity_potential = (
        upwind_water_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total water phase potential
    water_phase_potential = water_pressure_difference + water_gravity_potential

    # For oil: v_o = λ_o * [(P_oil) + (upwind_ρ_oil * g * Δz)] / ΔL
    oil_gravity_potential = (
        upwind_oil_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total oil phase potential
    oil_phase_potential = oil_pressure_difference + oil_gravity_potential

    # For gas: v_g = λ_g * ∆P / ΔL
    # v_g = λ_g * [(P_oil + P_go) - (P_cog + P_gas) + (upwind_ρ_gas * g * Δz)] / ΔL
    gas_gravity_potential = (
        upwind_gas_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total gas phase potential
    gas_phase_potential = gas_pressure_difference + gas_gravity_potential

    upwind_water_mobility = (
        water_mobility_grid[neighbour_indices]
        if water_phase_potential > 0.0  # Flow from neighbour to cell
        else water_mobility_grid[cell_indices]
    )

    upwind_oil_mobility = (
        oil_mobility_grid[neighbour_indices]
        if oil_phase_potential > 0.0
        else oil_mobility_grid[cell_indices]
    )

    upwind_gas_mobility = (
        gas_mobility_grid[neighbour_indices]
        if gas_phase_potential > 0.0
        else gas_mobility_grid[cell_indices]
    )

    water_velocity = upwind_water_mobility * water_phase_potential / flow_length
    oil_velocity = upwind_oil_mobility * oil_phase_potential / flow_length
    gas_velocity = upwind_gas_mobility * gas_phase_potential / flow_length

    # Compute volumetric fluxes at the face for each phase
    # F_x = v_x * A
    # For water: F_w = v_w * A
    water_volumetric_flux_at_face = water_velocity * flow_area
    # For oil: F_o = v_o * A
    oil_volumetric_flux_at_face = oil_velocity * flow_area
    # For gas: F_g = v_g * A
    gas_volumetric_flux_at_face = gas_velocity * flow_area
    return (
        water_volumetric_flux_at_face,
        oil_volumetric_flux_at_face,
        gas_volumetric_flux_at_face,
    )


def evolve_miscible_saturation_explicitly(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    options: Options,
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ] = None,
) -> EvolutionResult[
    typing.Tuple[
        ThreeDimensionalGrid,  # water_saturation
        ThreeDimensionalGrid,  # oil_saturation
        ThreeDimensionalGrid,  # gas_saturation
        ThreeDimensionalGrid,  # solvent_concentration
    ]
]:
    """
    Evolve saturations with Todd-Longstaff miscible displacement.

    Solvent (e.g., CO2) can exist as:
    1. Free gas phase (tracked by gas_saturation)
    2. Dissolved in oil (tracked by solvent_concentration in oil)

    Returns: (water_sat, oil_sat, gas_sat, solvent_conc_in_oil)
    """
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid

    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid
    current_solvent_concentration_grid = fluid_properties.solvent_concentration_grid

    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    cell_count_x, cell_count_y, cell_count_z = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    # Compute mobility grids for x, y, z directions
    water_mobility_grid_x = (
        absolute_permeability.x
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_x = (
        absolute_permeability.x
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_x = (
        absolute_permeability.x
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    water_mobility_grid_y = (
        absolute_permeability.y
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_y = (
        absolute_permeability.y
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_y = (
        absolute_permeability.y
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    water_mobility_grid_z = (
        absolute_permeability.z
        * water_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_z = (
        absolute_permeability.z
        * oil_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_z = (
        absolute_permeability.z
        * gas_relative_mobility_grid
        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    updated_water_saturation_grid = current_water_saturation_grid.copy()
    updated_oil_saturation_grid = current_oil_saturation_grid.copy()
    updated_gas_saturation_grid = current_gas_saturation_grid.copy()
    updated_solvent_concentration_grid = current_solvent_concentration_grid.copy()

    # Iterate over internal cells only
    # Assume boundary cells are added via padding for boundary conditions application purposes
    # Thus, we iterate from 1 to N-1 in each dimension
    temperature_grid = fluid_properties.temperature_grid
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_thickness = thickness_grid[i, j, k]
        cell_volume = cell_size_x * cell_size_y * cell_thickness
        cell_porosity = porosity_grid[i, j, k]
        cell_pore_volume = cell_volume * cell_porosity
        cell_oil_saturation = current_oil_saturation_grid[i, j, k]
        cell_solvent_concentration = current_solvent_concentration_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]
        cell_temperature = temperature_grid[i, j, k]

        flux_configurations = {
            "x": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_x,
                    "oil_mobility_grid": oil_mobility_grid_x,
                    "gas_mobility_grid": gas_mobility_grid_x,
                },
                "neighbours": [
                    (i + 1, j, k),
                    (i - 1, j, k),
                ],  # East and West neighbours
                "flow_area": cell_size_y * cell_thickness,
                "flow_length": cell_size_x,
            },
            "y": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_y,
                    "oil_mobility_grid": oil_mobility_grid_y,
                    "gas_mobility_grid": gas_mobility_grid_y,
                },
                "neighbours": [
                    (i, j - 1, k),
                    (i, j + 1, k),
                ],  # North and South neighbours
                "flow_area": cell_size_x * cell_thickness,
                "flow_length": cell_size_y,
            },
            "z": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_z,
                    "oil_mobility_grid": oil_mobility_grid_z,
                    "gas_mobility_grid": gas_mobility_grid_z,
                },
                "neighbours": [
                    (i, j, k - 1),
                    (i, j, k + 1),
                ],  # Top and Bottom neighbours
                "flow_area": cell_size_x * cell_size_y,
                "flow_length": cell_thickness,
            },
        }

        # Accumulate fluxes from all directions
        net_water_flux = 0.0
        net_oil_flux = 0.0
        net_gas_flux = 0.0
        net_solvent_flux = 0.0
        for config in flux_configurations.values():
            flow_area = typing.cast(float, config["flow_area"])
            flow_length = typing.cast(float, config["flow_length"])
            mobility_grids = typing.cast(
                typing.Dict[str, ThreeDimensionalGrid], config["mobility_grids"]
            )

            for neighbour in config["neighbours"]:  # type: ignore
                # Compute fluxes from neighbour
                (
                    water_flux,
                    oil_flux,
                    gas_flux,
                    solvent_flux,
                ) = compute_miscible_phase_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=neighbour,
                    flow_area=flow_area,
                    flow_length=flow_length,
                    oil_pressure_grid=current_oil_pressure_grid,
                    **mobility_grids,
                    solvent_concentration_grid=current_solvent_concentration_grid,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                )
                net_water_flux += water_flux
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux
                net_solvent_flux += solvent_flux

        # Well contributions
        injection_well, production_well = wells[i, j, k]
        cell_water_injection_rate = 0.0
        cell_water_production_rate = 0.0
        cell_oil_injection_rate = 0.0
        cell_oil_production_rate = 0.0
        cell_gas_injection_rate = 0.0
        cell_gas_production_rate = 0.0
        cell_solvent_injection_concentration = 0.0
        permeability = (
            absolute_permeability.x[i, j, k],
            absolute_permeability.y[i, j, k],
            absolute_permeability.z[i, j, k],
        )
        if (
            injection_well is not None
            and injection_well.is_open
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            # If there is an injection well, add its flow rate to the cell
            injected_phase = injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                phase_mobility = gas_relative_mobility_grid[i, j, k]
                compressibility_kwargs = {}
            else:
                phase_mobility = water_relative_mobility_grid[i, j, k]
                compressibility_kwargs = {
                    "bubble_point_pressure": fluid_properties.oil_bubble_point_pressure_grid[
                        i, j, k
                    ],
                    "gas_formation_volume_factor": fluid_properties.gas_formation_volume_factor_grid[
                        i, j, k
                    ],
                    "gas_solubility_in_water": fluid_properties.gas_solubility_in_water_grid[
                        i, j, k
                    ],
                }
            fluid_compressibility = injected_fluid.get_compressibility(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                **compressibility_kwargs,
            )
            fluid_formation_volume_factor = injected_fluid.get_formation_volume_factor(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
            )

            use_pseudo_pressure = (
                options.use_pseudo_pressure and injected_phase == FluidPhase.GAS
            )
            well_index = injection_well.get_well_index(
                interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                permeability=permeability,
                skin_factor=injection_well.skin_factor,
            )
            cell_injection_rate = injection_well.get_flow_rate(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_mobility=phase_mobility,
                fluid=injected_fluid,
                fluid_compressibility=fluid_compressibility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=fluid_formation_volume_factor,
            )
            if cell_injection_rate < 0.0 and options.warn_rates_anomalies:
                _warn_injector_is_producing(
                    injection_rate=cell_injection_rate,
                    well_name=injection_well.name,
                    cell=(i, j, k),
                    time=time_step * time_step_size,
                    rate_unit="ft³/day"
                    if injected_phase == FluidPhase.GAS
                    else "bbls/day",
                )

            # Handle miscible solvent injection
            if injected_phase == FluidPhase.GAS and injected_fluid.is_miscible:
                # Miscible solvent injection (e.g., CO2)
                cell_gas_injection_rate = cell_injection_rate  # ft³/day
                # This will be mixed with existing oil in the mass balance
                cell_solvent_injection_concentration += injected_fluid.concentration
                if injection_grid is not None:
                    injection_grid[i, j, k] = (0.0, 0.0, cell_gas_injection_rate)

            elif injected_phase == FluidPhase.GAS:
                # Non-miscible gas injection
                cell_gas_injection_rate = cell_injection_rate
                if injection_grid is not None:
                    injection_grid[i, j, k] = (0.0, 0.0, cell_gas_injection_rate)

            else:  # WATER INJECTION
                cell_water_injection_rate = cell_injection_rate * c.BBL_TO_FT3
                if injection_grid is not None:
                    injection_grid[i, j, k] = (0.0, cell_water_injection_rate, 0.0)

        if production_well is not None and production_well.is_open:
            water_formation_volume_factor_grid = (
                fluid_properties.water_formation_volume_factor_grid
            )
            oil_formation_volume_factor_grid = (
                fluid_properties.oil_formation_volume_factor_grid
            )
            gas_formation_volume_factor_grid = (
                fluid_properties.gas_formation_volume_factor_grid
            )
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase

                if produced_phase == FluidPhase.GAS:
                    phase_mobility = gas_relative_mobility_grid[i, j, k]
                    fluid_compressibility = gas_compressibility_grid[i, j, k]
                    fluid_formation_volume_factor = gas_formation_volume_factor_grid[
                        i, j, k
                    ]
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = water_relative_mobility_grid[i, j, k]
                    fluid_compressibility = water_compressibility_grid[i, j, k]
                    fluid_formation_volume_factor = water_formation_volume_factor_grid[
                        i, j, k
                    ]
                else:  # OIL
                    phase_mobility = oil_relative_mobility_grid[i, j, k]
                    fluid_compressibility = oil_compressibility_grid[i, j, k]
                    fluid_formation_volume_factor = oil_formation_volume_factor_grid[
                        i, j, k
                    ]

                use_pseudo_pressure = (
                    options.use_pseudo_pressure and produced_phase == FluidPhase.GAS
                )
                well_index = production_well.get_well_index(
                    interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                    permeability=permeability,
                    skin_factor=production_well.skin_factor,
                )
                production_rate = production_well.get_flow_rate(
                    pressure=cell_oil_pressure,
                    temperature=cell_temperature,
                    well_index=well_index,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                    fluid_compressibility=fluid_compressibility,
                    use_pseudo_pressure=use_pseudo_pressure,
                    formation_volume_factor=fluid_formation_volume_factor,
                )

                if production_rate > 0.0 and options.warn_rates_anomalies:
                    _warn_producer_is_injecting(
                        production_rate=production_rate,
                        well_name=production_well.name,
                        cell=(i, j, k),
                        time=time_step * time_step_size,
                        rate_unit="ft³/day"
                        if produced_phase == FluidPhase.GAS
                        else "bbls/day",
                    )

                if produced_phase == FluidPhase.GAS:
                    cell_gas_production_rate += production_rate
                elif produced_phase == FluidPhase.WATER:
                    cell_water_production_rate += production_rate * c.BBL_TO_FT3
                else:  # OIL
                    cell_oil_production_rate += production_rate * c.BBL_TO_FT3

            if production_grid is not None:
                production_grid[i, j, k] = (
                    cell_oil_production_rate,
                    cell_water_production_rate,
                    cell_gas_production_rate,
                )

        # Net well flow rates
        net_water_flow_rate = cell_water_injection_rate + cell_water_production_rate
        net_oil_flow_rate = cell_oil_injection_rate + cell_oil_production_rate
        net_gas_flow_rate = cell_gas_injection_rate + cell_gas_production_rate

        # Calculate total throughput (fluid moving through the cell)
        # Advective fluxes (already signed: positive = inflow, negative = outflow)
        water_inflow_advection = max(0.0, net_water_flux)
        oil_inflow_advection = max(0.0, net_oil_flux)
        gas_inflow_advection = max(0.0, net_gas_flux)

        water_outflow_advection = abs(min(0.0, net_water_flux))
        oil_outflow_advection = abs(min(0.0, net_oil_flux))
        gas_outflow_advection = abs(min(0.0, net_gas_flux))

        # Well flows (production is negative, injection is positive)
        water_inflow_well = max(0.0, net_water_flow_rate)
        oil_inflow_well = max(0.0, net_oil_flow_rate)
        gas_inflow_well = max(0.0, net_gas_flow_rate)

        water_outflow_well = abs(min(0.0, net_water_flow_rate))
        oil_outflow_well = abs(min(0.0, net_oil_flow_rate))
        gas_outflow_well = abs(min(0.0, net_gas_flow_rate))

        # Total throughput
        total_inflow = (
            water_inflow_advection
            + oil_inflow_advection
            + gas_inflow_advection
            + water_inflow_well
            + oil_inflow_well
            + gas_inflow_well
        )
        total_outflow = (
            water_outflow_advection
            + oil_outflow_advection
            + gas_outflow_advection
            + water_outflow_well
            + oil_outflow_well
            + gas_outflow_well
        )
        total_throughput = total_inflow + total_outflow
        # CFL check
        cfl_number = (total_throughput * time_step_in_days) / cell_pore_volume
        max_cfl_number = options.max_cfl_number.get(options.scheme, 1.0)
        if cfl_number > max_cfl_number:
            raise RuntimeError(
                f"CFL condition violated at cell ({i}, {j}, {k}) at timestep {time_step}: "
                f"CFL number {cfl_number:.4f} exceeds limit {max_cfl_number:.4f}. "
                f"Inflow = {total_inflow:.2f} ft³/day, Outflow = {total_outflow:.2f} ft³/day, "
                f"Pore volume = {cell_pore_volume:.2f} ft³. "
                f"Consider reducing time step size from {time_step_size} seconds."
            )

        # Total flow rates (advection + wells)
        total_water_flow = net_water_flux + net_water_flow_rate
        total_oil_flow = net_oil_flux + net_oil_flow_rate
        total_gas_flow = net_gas_flux + net_gas_flow_rate

        # Update saturations
        water_saturation_change = (
            total_water_flow * time_step_in_days
        ) / cell_pore_volume
        oil_saturation_change = (total_oil_flow * time_step_in_days) / cell_pore_volume
        gas_saturation_change = (total_gas_flow * time_step_in_days) / cell_pore_volume

        updated_water_saturation_grid[i, j, k] += water_saturation_change
        updated_oil_saturation_grid[i, j, k] += oil_saturation_change
        updated_gas_saturation_grid[i, j, k] += gas_saturation_change

        # Update solvent concentration in oil phase
        # Mass balance: (C_old * V_oil_old) + (C_in * V_in) = (C_new * V_oil_new)
        new_oil_saturation = updated_oil_saturation_grid[i, j, k]
        if new_oil_saturation > 1e-9:  # Avoid division by zero
            # Current solvent mass in oil
            old_solvent_mass = (
                cell_solvent_concentration * cell_oil_saturation * cell_pore_volume
            )
            # Solvent mass flux from advection (already computed)
            advected_solvent_mass = net_solvent_flux * time_step_in_days

            # Solvent mass from injection (if miscible)
            injected_solvent_mass = 0.0
            if (
                cell_gas_injection_rate > 0.0
                and cell_solvent_injection_concentration > 0.0
            ):
                # Miscible solvent dissolves into oil immediately
                # Assumption: All injected solvent mixes with oil
                injected_solvent_mass = (
                    cell_solvent_injection_concentration
                    * cell_gas_injection_rate
                    * time_step_in_days
                )

            # Total solvent mass in oil
            new_solvent_mass = (
                old_solvent_mass + advected_solvent_mass + injected_solvent_mass
            )
            # New oil volume
            new_oil_volume = new_oil_saturation * cell_pore_volume
            # New concentration
            new_concentration = new_solvent_mass / new_oil_volume
            # Clamp to [0, 1]
            updated_solvent_concentration_grid[i, j, k] = np.clip(
                new_concentration, 0.0, 1.0
            )
        else:
            # No oil in cell, concentration is undefined (set to 0)
            updated_solvent_concentration_grid[i, j, k] = 0.0

    # Apply saturation constraints and normalization across all cells
    # Clean up any remaining minor negative values caused by floating point errors
    updated_water_saturation_grid[updated_water_saturation_grid < 0.0] = 0.0
    updated_oil_saturation_grid[updated_oil_saturation_grid < 0.0] = 0.0
    updated_gas_saturation_grid[updated_gas_saturation_grid < 0.0] = 0.0
    total_saturation_grid = (
        updated_water_saturation_grid
        + updated_oil_saturation_grid
        + updated_gas_saturation_grid
    )
    # Only normalize cells where there is saturation present (total > SATURATION_EPSILON to avoid division by zero edge cases)
    mask = total_saturation_grid > c.SATURATION_EPSILON
    if np.any(mask):
        updated_water_saturation_grid[mask] /= total_saturation_grid[mask]
        updated_oil_saturation_grid[mask] /= total_saturation_grid[mask]
        updated_gas_saturation_grid[mask] /= total_saturation_grid[mask]
    return EvolutionResult(
        (
            updated_water_saturation_grid,
            updated_oil_saturation_grid,
            updated_gas_saturation_grid,
            updated_solvent_concentration_grid,
        ),
        scheme="explicit",
    )


def compute_miscible_phase_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    flow_area: float,
    flow_length: float,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_mobility_grid: ThreeDimensionalGrid,
    oil_mobility_grid: ThreeDimensionalGrid,
    gas_mobility_grid: ThreeDimensionalGrid,
    solvent_concentration_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    water_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    gas_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    elevation_grid: typing.Optional[ThreeDimensionalGrid] = None,
) -> typing.Tuple[float, float, float, float]:  # water, oil, gas, solvent_in_oil
    """
    Compute phase fluxes including solvent concentration transport.

    Returns: (water_flux, oil_flux, gas_flux, solvent_mass_flux_in_oil)

    The solvent_mass_flux_in_oil is the mass flux of dissolved solvent
    moving with the oil phase (ft³/day * concentration).
    """
    cell_oil_pressure = oil_pressure_grid[cell_indices]
    cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[cell_indices]
    cell_solvent_concentration = solvent_concentration_grid[cell_indices]

    neighbour_oil_pressure = oil_pressure_grid[neighbour_indices]
    neighbour_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[
        neighbour_indices
    ]
    neighbour_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[
        neighbour_indices
    ]
    neighbour_solvent_concentration = solvent_concentration_grid[neighbour_indices]

    # Pressure differences
    oil_pressure_difference = neighbour_oil_pressure - cell_oil_pressure
    oil_water_capillary_pressure_difference = (
        neighbour_oil_water_capillary_pressure - cell_oil_water_capillary_pressure
    )
    water_pressure_difference = (
        oil_pressure_difference - oil_water_capillary_pressure_difference
    )
    gas_oil_capillary_pressure_difference = (
        neighbour_gas_oil_capillary_pressure - cell_gas_oil_capillary_pressure
    )
    gas_pressure_difference = (
        oil_pressure_difference + gas_oil_capillary_pressure_difference
    )

    # Elevation effects
    if elevation_grid is not None:
        elevation_delta = (
            elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
        )
    else:
        elevation_delta = 0.0

    # Upwind densities
    if water_density_grid is not None:
        upwind_water_density = (
            water_density_grid[neighbour_indices]
            if water_pressure_difference > 0.0
            else water_density_grid[cell_indices]
        )
    else:
        upwind_water_density = 0.0

    if oil_density_grid is not None:
        upwind_oil_density = (
            oil_density_grid[neighbour_indices]
            if oil_pressure_difference > 0.0
            else oil_density_grid[cell_indices]
        )
    else:
        upwind_oil_density = 0.0

    if gas_density_grid is not None:
        upwind_gas_density = (
            gas_density_grid[neighbour_indices]
            if gas_pressure_difference > 0.0
            else gas_density_grid[cell_indices]
        )
    else:
        upwind_gas_density = 0.0

    # Darcy velocities with gravity
    water_gravity_potential = (
        upwind_water_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    water_phase_potential = water_pressure_difference + water_gravity_potential

    oil_gravity_potential = (
        upwind_oil_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    oil_phase_potential = oil_pressure_difference + oil_gravity_potential

    gas_gravity_potential = (
        upwind_gas_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    gas_phase_potential = gas_pressure_difference + gas_gravity_potential

    upwind_water_mobility = (
        water_mobility_grid[neighbour_indices]
        if water_phase_potential > 0.0  # Flow from neighbour to cell
        else water_mobility_grid[cell_indices]
    )

    upwind_oil_mobility = (
        oil_mobility_grid[neighbour_indices]
        if oil_phase_potential > 0.0
        else oil_mobility_grid[cell_indices]
    )

    upwind_gas_mobility = (
        gas_mobility_grid[neighbour_indices]
        if gas_phase_potential > 0.0
        else gas_mobility_grid[cell_indices]
    )

    water_velocity = upwind_water_mobility * water_phase_potential / flow_length
    oil_velocity = upwind_oil_mobility * oil_phase_potential / flow_length
    gas_velocity = upwind_gas_mobility * gas_phase_potential / flow_length

    # Upwind solvent concentration (moves with oil)
    upwinded_solvent_concentration = (
        neighbour_solvent_concentration
        if oil_velocity > 0
        else cell_solvent_concentration
    )

    # Volumetric fluxes (ft³/day)
    water_volumetric_flux = water_velocity * flow_area
    oil_volumetric_flux = oil_velocity * flow_area
    gas_volumetric_flux = gas_velocity * flow_area

    # Solvent mass flux in oil phase
    # The solvent concentration travels with the oil phase
    solvent_mass_flux_in_oil = oil_volumetric_flux * upwinded_solvent_concentration
    return (
        water_volumetric_flux,
        oil_volumetric_flux,
        gas_volumetric_flux,
        solvent_mass_flux_in_oil,
    )
