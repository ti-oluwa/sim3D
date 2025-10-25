import itertools
import typing

import numpy as np

from sim3D.boundaries import BoundaryConditions, BoundaryMetadata
from sim3D.constants import (
    ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2,
    BBL_TO_FT3,
    DAYS_PER_SECOND,
    MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY,
)
from sim3D.diffusivity.base import (
    EvolutionResult,
    _warn_injector_is_producing,
    _warn_producer_is_injecting,
)
from sim3D.grids.base import edge_pad_grid
from sim3D.grids.properties import (
    build_three_phase_capillary_pressure_grids,
    build_three_phase_relative_mobilities_grids,
    build_total_fluid_compressibility_grid,
)
from sim3D.properties import compute_harmonic_mobility
from sim3D.statics import FluidProperties, RockFluidProperties, RockProperties
from sim3D.types import (
    FluidPhase,
    Options,
    RelativePermeabilityFunc,
    SupportsSetItem,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from sim3D.wells import Wells

__all__ = ["evolve_saturation_explicitly"]

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

    ∇ · (f_x * v ) ≈ [(F_x_east - F_x_west)/Δx + (F_y_north - F_y_south)/Δy + (F_z_top - F_z_bottom)/Δz]

    Sⁿ⁺¹_ijk = Sⁿ_ijk + Δt / (φ * V_cell) * [
        (F_x_east - F_x_west) + (F_y_north - F_y_south) + (F_z_top - F_z_bottom) + q_x_ijk * V_cell
    ]

Volumetric phase flux at face F_dir is computed as:
    F_dir = f_x(S_upwind) * v_dir * A_face (ft³/day)

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


def _compute_explicit_saturation_phase_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    flow_area: float,
    flow_length: float,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_mobility_grid: ThreeDimensionalGrid,
    oil_mobility_grid: ThreeDimensionalGrid,
    gas_mobility_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    oil_viscosity_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    irreducible_water_saturation_grid: ThreeDimensionalGrid,
    residual_oil_saturation_water_grid: ThreeDimensionalGrid,
    residual_oil_saturation_gas_grid: ThreeDimensionalGrid,
    residual_gas_saturation_grid: ThreeDimensionalGrid,
    relative_permeability_func: RelativePermeabilityFunc,
    oil_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    water_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    gas_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    elevation_grid: typing.Optional[ThreeDimensionalGrid] = None,
) -> typing.Tuple[float, float, float]:
    # Current cell pressures (P_oil is direct, P_water and P_gas derived)
    cell_oil_pressure = oil_pressure_grid[cell_indices]
    cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[cell_indices]

    # Current cell saturations
    cell_water_saturation = water_saturation_grid[cell_indices]
    cell_oil_saturation = oil_saturation_grid[cell_indices]
    cell_gas_saturation = gas_saturation_grid[cell_indices]
    cell_oil_viscosity = oil_viscosity_grid[cell_indices]
    cell_water_viscosity = water_viscosity_grid[cell_indices]
    cell_gas_viscosity = gas_viscosity_grid[cell_indices]

    # For the neighbour
    neighbour_oil_pressure = oil_pressure_grid[neighbour_indices]
    neighbour_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[
        neighbour_indices
    ]
    neighbour_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[
        neighbour_indices
    ]
    neighbour_water_saturation = water_saturation_grid[neighbour_indices]
    neighbour_oil_saturation = oil_saturation_grid[neighbour_indices]
    neighbour_gas_saturation = gas_saturation_grid[neighbour_indices]
    neighbour_oil_viscosity = oil_viscosity_grid[neighbour_indices]
    neighbour_water_viscosity = water_viscosity_grid[neighbour_indices]
    neighbour_gas_viscosity = gas_viscosity_grid[neighbour_indices]

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

    # Compute harmonic mobility of the phases from the neighbour (direction of flow)
    water_harmonic_mobility = compute_harmonic_mobility(
        cell_indices,
        neighbour_indices,
        mobility_grid=water_mobility_grid,
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        cell_indices,
        neighbour_indices,
        mobility_grid=oil_mobility_grid,
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        cell_indices,
        neighbour_indices,
        mobility_grid=gas_mobility_grid,
    )

    # Computing the Darcy velocities (ft/day) for the three phases
    # v_x = λ_x * ∆P / Δx
    # For water: v_w = λ_w * [(P_oil - P_cow) + (upwind_ρ_water * g * Δz)] / ΔL
    water_gravity_potential = (
        upwind_water_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total water phase potential
    water_phase_potential = water_pressure_difference + water_gravity_potential
    water_velocity = water_harmonic_mobility * water_phase_potential / flow_length

    # For oil: v_o = λ_o * [(P_oil) + (upwind_ρ_oil * g * Δz)] / ΔL
    oil_gravity_potential = (
        upwind_oil_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total oil phase potential
    oil_phase_potential = oil_pressure_difference + oil_gravity_potential
    oil_velocity = oil_harmonic_mobility * oil_phase_potential / flow_length

    # For gas: v_g = λ_g * ∆P / ΔL
    # v_g = λ_g * [(P_oil + P_go) - (P_cog + P_gas) + (upwind_ρ_gas * g * Δz)] / ΔL
    gas_gravity_potential = (
        upwind_gas_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total gas phase potential
    gas_phase_potential = gas_pressure_difference + gas_gravity_potential
    gas_velocity = gas_harmonic_mobility * gas_phase_potential / flow_length

    # Select upwind saturations in the direction of flow
    upwinded_water_saturation = (
        neighbour_water_saturation if water_velocity > 0 else cell_water_saturation
    )
    upwinded_oil_saturation = (
        neighbour_oil_saturation if oil_velocity > 0 else cell_oil_saturation
    )
    upwinded_gas_saturation = (
        neighbour_gas_saturation if gas_velocity > 0 else cell_gas_saturation
    )

    # Select upwind viscosities in the direction of flow
    upwinded_water_viscosity = (
        neighbour_water_viscosity if water_velocity > 0 else cell_water_viscosity
    )
    upwinded_oil_viscosity = (
        neighbour_oil_viscosity if oil_velocity > 0 else cell_oil_viscosity
    )
    upwinded_gas_viscosity = (
        neighbour_gas_viscosity if gas_velocity > 0 else cell_gas_viscosity
    )

    # Compute the total fractional flow
    upwinded_relative_permeabilities = relative_permeability_func(
        water_saturation=upwinded_water_saturation,
        oil_saturation=upwinded_oil_saturation,
        gas_saturation=upwinded_gas_saturation,
        irreducible_water_saturation=irreducible_water_saturation_grid[cell_indices],
        residual_oil_saturation_water=residual_oil_saturation_water_grid[cell_indices],
        residual_oil_saturation_gas=residual_oil_saturation_gas_grid[cell_indices],
        residual_gas_saturation=residual_gas_saturation_grid[cell_indices],
    )
    upwinded_water_relative_permeability = upwinded_relative_permeabilities["water"]
    upwinded_oil_relative_permeability = upwinded_relative_permeabilities["oil"]
    upwinded_gas_relative_permeability = upwinded_relative_permeabilities["gas"]

    water_upwinded_mobility = (
        upwinded_water_relative_permeability / upwinded_water_viscosity
    )
    oil_upwinded_mobility = upwinded_oil_relative_permeability / upwinded_oil_viscosity
    gas_upwinded_mobility = upwinded_gas_relative_permeability / upwinded_gas_viscosity

    # f_phase = λ_phase(S_upwind) / (λ_w(S_upwind) + λ_o(S_upwind) + λ_g(S_upwind))
    total_upwinded_mobility = (
        water_upwinded_mobility + oil_upwinded_mobility + gas_upwinded_mobility
    )
    total_upwinded_mobility = np.maximum(
        total_upwinded_mobility, 1e-18
    )  # Avoid division by zero
    # For water: f_w = λ_w / (λ_w + λ_o + λ_g)
    water_fractional_flow = water_upwinded_mobility / total_upwinded_mobility
    # For oil: f_o = λ_o / (λ_w + λ_o + λ_g)
    oil_fractional_flow = oil_upwinded_mobility / total_upwinded_mobility
    # For gas: f_g = λ_g / (λ_w + λ_o + λ_g)
    gas_fractional_flow = gas_upwinded_mobility / total_upwinded_mobility

    # Compute volumetric fluxes at the face for each phase
    # F_x = f_x * v_x * A
    # For water: F_w = f_w * v_w * A
    water_volumetric_flux_at_face = water_fractional_flow * water_velocity * flow_area
    # For oil: F_o = f_o * v_o * A
    oil_volumetric_flux_at_face = oil_fractional_flow * oil_velocity * flow_area
    # For gas: F_g = f_g * v_g * A
    gas_volumetric_flux_at_face = gas_fractional_flow * gas_velocity * flow_area

    # Compute the component fluxes in the direction of flow - from the neighbour to the current cell
    # These are the fluxes that will be used to update the current cell's saturation
    # For water: F_w_from_neighbour = F_w
    water_volumetric_flux_from_neighbour = water_volumetric_flux_at_face
    # For oil: F_o_from_neighbour = F_o
    oil_volumetric_flux_from_neighbour = oil_volumetric_flux_at_face
    # For gas: F_g_from_neighbour = F_g
    gas_volumetric_flux_from_neighbour = gas_volumetric_flux_at_face
    return (
        water_volumetric_flux_from_neighbour,
        oil_volumetric_flux_from_neighbour,
        gas_volumetric_flux_from_neighbour,
    )


def evolve_saturation_explicitly(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
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
    :param boundary_conditions: Boundary conditions for pressure and saturation grids.
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
    time_step_in_days = time_step_size * DAYS_PER_SECOND
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid
    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_water_grid = (
        rock_properties.residual_oil_saturation_water_grid
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_func = rock_fluid_properties.relative_permeability_func
    capillary_pressure_params = (
        rock_fluid_properties.capillary_pressure_params
    )  # Contains wettability type

    current_oil_pressure_grid = (
        fluid_properties.pressure_grid
    )  # This is P_oil or Pⁿ_{i,j}
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_viscosity_grid = fluid_properties.water_viscosity_grid
    oil_viscosity_grid = fluid_properties.oil_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Determine grid dimensions and cell sizes
    cell_count_x, cell_count_y, cell_count_z = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Compute total fluid system compressibility for each cell
    total_fluid_compressibility_grid = build_total_fluid_compressibility_grid(
        oil_saturation_grid=current_oil_saturation_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        water_saturation_grid=current_water_saturation_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_saturation_grid=current_gas_saturation_grid,
        gas_compressibility_grid=gas_compressibility_grid,
    )
    # Total compressibility (psi⁻¹) = (fluid compressibility * porosity) + rock compressibility
    total_compressibility_grid = (
        total_fluid_compressibility_grid * porosity_grid
    ) + rock_compressibility

    # Ensure total compressibility is never zero or negative (for numerical stability)
    total_compressibility_grid = np.maximum(total_compressibility_grid, 1e-18)

    # Pad all necessary grids for boundary conditions and neighbour access
    padded_oil_pressure_grid = edge_pad_grid(current_oil_pressure_grid)
    padded_oil_saturation_grid = edge_pad_grid(current_oil_saturation_grid)
    padded_water_saturation_grid = edge_pad_grid(current_water_saturation_grid)
    padded_gas_saturation_grid = edge_pad_grid(current_gas_saturation_grid)
    padded_irreducible_water_saturation_grid = edge_pad_grid(
        irreducible_water_saturation_grid
    )
    padded_residual_oil_saturation_water_grid = edge_pad_grid(
        residual_oil_saturation_water_grid
    )
    padded_residual_oil_saturation_gas_grid = edge_pad_grid(
        residual_oil_saturation_gas_grid
    )
    padded_residual_gas_saturation_grid = edge_pad_grid(residual_gas_saturation_grid)
    padded_water_viscosity_grid = edge_pad_grid(water_viscosity_grid)
    padded_oil_viscosity_grid = edge_pad_grid(oil_viscosity_grid)
    padded_gas_viscosity_grid = edge_pad_grid(gas_viscosity_grid)
    padded_oil_density_grid = edge_pad_grid(oil_density_grid)
    padded_water_density_grid = edge_pad_grid(water_density_grid)
    padded_gas_density_grid = edge_pad_grid(gas_density_grid)
    padded_elevation_grid = edge_pad_grid(elevation_grid)

    # Apply boundary conditions to relevant padded grids
    boundary_conditions["pressure"].apply(
        padded_oil_pressure_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time_step_size * time_step,
            grid_shape=current_oil_pressure_grid.shape,
            property_name="pressure",
        ),
    )
    boundary_conditions["oil_saturation"].apply(
        padded_oil_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time_step_size * time_step,
            grid_shape=current_oil_saturation_grid.shape,
            property_name="oil_saturation",
        ),
    )
    boundary_conditions["water_saturation"].apply(
        padded_water_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time_step_size * time_step,
            grid_shape=current_water_saturation_grid.shape,
            property_name="water_saturation",
        ),
    )
    boundary_conditions["gas_saturation"].apply(
        padded_gas_saturation_grid,
        metadata=BoundaryMetadata(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time_step_size * time_step,
            grid_shape=current_gas_saturation_grid.shape,
            property_name="gas_saturation",
        ),
    )

    # Compute phase mobilities (kr / mu) for each cell
    # `build_three_phase_relative_mobilities_grids` should handle `k_abs * kr / mu` for each phase.
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_three_phase_relative_mobilities_grids(
        water_saturation_grid=current_water_saturation_grid,
        oil_saturation_grid=current_oil_saturation_grid,
        gas_saturation_grid=current_gas_saturation_grid,
        water_viscosity_grid=water_viscosity_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        relative_permeability_func=relative_permeability_func,
    )

    # Clamp relative mobility grids to avoid numerical issues
    # This ensures mobilities are never zero or negative (for numerical stability)
    water_relative_mobility_grid = options.relative_mobility_range["water"].arrayclip(
        water_relative_mobility_grid
    )
    oil_relative_mobility_grid = options.relative_mobility_range["oil"].arrayclip(
        oil_relative_mobility_grid
    )
    gas_relative_mobility_grid = options.relative_mobility_range["gas"].arrayclip(
        gas_relative_mobility_grid
    )

    # Compute mobility grids for x, y, z directions
    # λ_x = 0.001127 * k_abs * (kr / mu) (mD/cP to ft²/psi.day)
    water_mobility_grid_x = (
        absolute_permeability.x
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_x = (
        absolute_permeability.x
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_x = (
        absolute_permeability.x
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    water_mobility_grid_y = (
        absolute_permeability.y
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_y = (
        absolute_permeability.y
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_y = (
        absolute_permeability.y
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    water_mobility_grid_z = (
        absolute_permeability.z
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    oil_mobility_grid_z = (
        absolute_permeability.z
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )
    gas_mobility_grid_z = (
        absolute_permeability.z
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    )

    # Pad mobility grids for neighbour access
    padded_water_mobility_grid_x = edge_pad_grid(water_mobility_grid_x)
    padded_oil_mobility_grid_x = edge_pad_grid(oil_mobility_grid_x)
    padded_gas_mobility_grid_x = edge_pad_grid(gas_mobility_grid_x)
    padded_water_mobility_grid_y = edge_pad_grid(water_mobility_grid_y)
    padded_oil_mobility_grid_y = edge_pad_grid(oil_mobility_grid_y)
    padded_gas_mobility_grid_y = edge_pad_grid(gas_mobility_grid_y)
    padded_water_mobility_grid_z = edge_pad_grid(water_mobility_grid_z)
    padded_oil_mobility_grid_z = edge_pad_grid(oil_mobility_grid_z)
    padded_gas_mobility_grid_z = edge_pad_grid(gas_mobility_grid_z)

    # Compute Capillary Pressures Grids (local to each cell, based on current saturations)
    # P_cow = P_oil - P_water (can be negative for oil-wet systems)
    # P_cgo = P_gas - P_oil (generally positive)
    padded_oil_water_capillary_pressure_grid, padded_gas_oil_capillary_pressure_grid = (
        build_three_phase_capillary_pressure_grids(
            water_saturation_grid=padded_water_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    # Create new grids for updated saturations (time 'n+1')
    updated_water_saturation_grid = current_water_saturation_grid.copy()
    updated_oil_saturation_grid = current_oil_saturation_grid.copy()
    updated_gas_saturation_grid = current_gas_saturation_grid.copy()

    # Iterate over each interior cell to compute saturation evolution
    for i, j, k in itertools.product(
        range(cell_count_x), range(cell_count_y), range(cell_count_z)
    ):
        ip, jp, kp = i + 1, j + 1, k + 1  # Indices of cell in padded grid
        cell_temperature = fluid_properties.temperature_grid[i, j, k]
        cell_thickness = thickness_grid[i, j, k]
        cell_total_volume = cell_size_x * cell_size_y * cell_thickness
        # Current cell properties
        cell_porosity = porosity_grid[i, j, k]
        # Cell pore volume = φ * V_cell
        cell_pore_volume = cell_total_volume * cell_porosity
        cell_oil_saturation = current_oil_saturation_grid[i, j, k]
        cell_water_saturation = current_water_saturation_grid[i, j, k]
        cell_gas_saturation = current_gas_saturation_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]

        flux_configurations = {
            "x": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_x,
                    "oil_mobility_grid": padded_oil_mobility_grid_x,
                    "gas_mobility_grid": padded_gas_mobility_grid_x,
                },
                "positive_neighbour": (ip + 1, jp, kp),
                "negative_neighbour": (ip - 1, jp, kp),
                "flow_area": cell_size_y * cell_thickness,  # A_x = Δy * Δz
                "flow_length": cell_size_x,  # Δx
            },
            "y": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_y,
                    "oil_mobility_grid": padded_oil_mobility_grid_y,
                    "gas_mobility_grid": padded_gas_mobility_grid_y,
                },
                "positive_neighbour": (ip, jp + 1, kp),
                "negative_neighbour": (ip, jp - 1, kp),
                "flow_area": cell_size_x * cell_thickness,  # A_y = Δx * Δz
                "flow_length": cell_size_y,  # Δy
            },
            "z": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_z,
                    "oil_mobility_grid": padded_oil_mobility_grid_z,
                    "gas_mobility_grid": padded_gas_mobility_grid_z,
                },
                "positive_neighbour": (ip, jp, kp + 1),
                "negative_neighbour": (ip, jp, kp - 1),
                "flow_area": cell_size_x * cell_size_y,  # A_z = Δx * Δy
                "flow_length": cell_thickness,  # Δz
            },
        }

        outgoing_water_flux = 0.0
        outgoing_oil_flux = 0.0
        outgoing_gas_flux = 0.0
        net_water_flux_from_neighbours = 0.0
        net_oil_flux_from_neighbours = 0.0
        net_gas_flux_from_neighbours = 0.0
        for direction, config in flux_configurations.items():
            positive_neighbour_indices = config["positive_neighbour"]
            negative_neighbour_indices = config["negative_neighbour"]
            flow_area = config["flow_area"]
            flow_length = config["flow_length"]

            # Compute fluxes from the positive neighbour
            (
                positive_water_flux,
                positive_oil_flux,
                positive_gas_flux,
            ) = _compute_explicit_saturation_phase_fluxes_from_neighbour(
                cell_indices=(i, j, k),
                neighbour_indices=positive_neighbour_indices,
                flow_area=flow_area,
                flow_length=flow_length,
                oil_pressure_grid=padded_oil_pressure_grid,
                **config["mobility_grids"],
                water_saturation_grid=padded_water_saturation_grid,
                oil_saturation_grid=padded_oil_saturation_grid,
                gas_saturation_grid=padded_gas_saturation_grid,
                oil_viscosity_grid=padded_oil_viscosity_grid,
                water_viscosity_grid=padded_water_viscosity_grid,
                gas_viscosity_grid=padded_gas_viscosity_grid,
                oil_water_capillary_pressure_grid=padded_oil_water_capillary_pressure_grid,
                gas_oil_capillary_pressure_grid=padded_gas_oil_capillary_pressure_grid,
                irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
                residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
                residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
                residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
                relative_permeability_func=relative_permeability_func,
                oil_density_grid=padded_oil_density_grid,
                water_density_grid=padded_water_density_grid,
                gas_density_grid=padded_gas_density_grid,
                elevation_grid=padded_elevation_grid,
            )

            # Compute fluxes from the negative neighbour
            (
                negative_water_flux,
                negative_oil_flux,
                negative_gas_flux,
            ) = _compute_explicit_saturation_phase_fluxes_from_neighbour(
                cell_indices=(i, j, k),
                neighbour_indices=negative_neighbour_indices,
                flow_area=flow_area,
                flow_length=flow_length,
                oil_pressure_grid=padded_oil_pressure_grid,
                **config["mobility_grids"],
                water_saturation_grid=padded_water_saturation_grid,
                oil_saturation_grid=padded_oil_saturation_grid,
                gas_saturation_grid=padded_gas_saturation_grid,
                oil_viscosity_grid=padded_oil_viscosity_grid,
                water_viscosity_grid=padded_water_viscosity_grid,
                gas_viscosity_grid=padded_gas_viscosity_grid,
                oil_water_capillary_pressure_grid=padded_oil_water_capillary_pressure_grid,
                gas_oil_capillary_pressure_grid=padded_gas_oil_capillary_pressure_grid,
                irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
                residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
                residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
                residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
                relative_permeability_func=relative_permeability_func,
                oil_density_grid=padded_oil_density_grid,
                water_density_grid=padded_water_density_grid,
                gas_density_grid=padded_gas_density_grid,
                elevation_grid=padded_elevation_grid,
            )
            # Accumulate net fluxes from both neighbours
            # Net fluxes are the sum of positive and negative fluxes, as negative fluxes
            # are already negative (i.e., flow from the neighbour to the current cell).
            net_water_flux_from_neighbour = positive_water_flux + negative_water_flux
            net_oil_flux_from_neighbour = positive_oil_flux + negative_oil_flux
            net_gas_flux_from_neighbour = positive_gas_flux + negative_gas_flux

            # Update the net fluxes from all neighbours
            net_water_flux_from_neighbours += net_water_flux_from_neighbour
            net_oil_flux_from_neighbours += net_oil_flux_from_neighbour
            net_gas_flux_from_neighbours += net_gas_flux_from_neighbour

            outgoing_water_flux += sum(
                min(f, 0.0) for f in (positive_water_flux, negative_water_flux)
            )
            outgoing_oil_flux += sum(
                min(f, 0.0) for f in (positive_oil_flux, negative_oil_flux)
            )
            outgoing_gas_flux += sum(
                min(f, 0.0) for f in (positive_gas_flux, negative_gas_flux)
            )

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
            # If there is an injection well, add its flow rate to the cell
            injected_phase = injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                phase_mobility = gas_relative_mobility_grid[i, j, k]
                fluid_compressibility = gas_compressibility_grid[i, j, k]
            elif injected_phase == FluidPhase.WATER:
                phase_mobility = water_relative_mobility_grid[i, j, k]
                fluid_compressibility = water_compressibility_grid[i, j, k]
            else:
                phase_mobility = oil_relative_mobility_grid[i, j, k]
                fluid_compressibility = oil_compressibility_grid[i, j, k]

            # Use pseudo pressure if the phase is GAS, if it is set in the options, and the pressure is high
            use_pseudo_pressure = (
                injected_phase == FluidPhase.GAS
                and options.use_pseudo_pressure
                and cell_oil_pressure >= 1500.0
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
            )
            if cell_injection_rate < 0.0:
                if injection_well.auto_clamp:
                    cell_injection_rate = 0.0
                else:
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

            elif injected_phase == FluidPhase.WATER:
                cell_water_injection_rate = cell_injection_rate * BBL_TO_FT3
                # Record water injection rate for the cell
                if injection_grid is not None:
                    injection_grid[i, j, k] = (0.0, cell_water_injection_rate, 0.0)

            else:
                cell_oil_injection_rate = cell_injection_rate * BBL_TO_FT3
                # Record oil injection rate for the cell
                if injection_grid is not None:
                    injection_grid[i, j, k] = (cell_oil_injection_rate, 0.0, 0.0)

        if production_well is not None and production_well.is_open:
            # If there is a production well, subtract its flow rate from the cell
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = gas_relative_mobility_grid[i, j, k]
                    fluid_compressibility = gas_compressibility_grid[i, j, k]
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = water_relative_mobility_grid[i, j, k]
                    fluid_compressibility = water_compressibility_grid[i, j, k]
                else:
                    phase_mobility = oil_relative_mobility_grid[i, j, k]
                    fluid_compressibility = oil_compressibility_grid[i, j, k]

                # Use pseudo pressure if the phase is GAS, if it is set in the options, and the pressure is high
                use_pseudo_pressure = (
                    produced_phase == FluidPhase.GAS
                    and options.use_pseudo_pressure
                    and cell_oil_pressure >= 1500.0
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
                )
                if production_rate > 0.0:
                    if production_well.auto_clamp:
                        production_rate = 0.0
                    else:
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
                    outgoing_gas_flux += production_rate
                elif produced_fluid.phase == FluidPhase.WATER:
                    cell_water_production_rate += production_rate * BBL_TO_FT3
                    outgoing_water_flux += production_rate * BBL_TO_FT3
                else:
                    cell_oil_production_rate += production_rate * BBL_TO_FT3
                    outgoing_oil_flux += production_rate * BBL_TO_FT3

            # Record total production rate for the cell (all phases)
            if production_grid is not None:
                production_grid[i, j, k] = (
                    cell_oil_production_rate,
                    cell_water_production_rate,
                    cell_gas_production_rate,
                )

        # CFL stability check: Outgoing volume flux should not exceed pore volume in the time step
        # Outflux * Δt <= φ * V_cell
        total_outgoing_flux = (
            abs(outgoing_water_flux) + abs(outgoing_oil_flux) + abs(outgoing_gas_flux)
        )
        stability_limit = (total_outgoing_flux * time_step_in_days) / (cell_pore_volume)
        if stability_limit > 1.0:
            raise RuntimeError(
                f"CFL condition violated at cell ({i}, {j}, {k}): "
                f"max normalized outgoing flux {total_outgoing_flux:.4f} exceeds cell pore volume {cell_pore_volume:.4f} "
                f"in time step {time_step} - {time_step_size} seconds. "
                f"Consider reducing the time step size for stability of the explicit saturation scheme."
            )

        # Compute the net volumetric rate for each phase. Just add injection and production rates (since production rates are negative)
        net_water_flow_rate_into_cell = (
            cell_water_injection_rate + cell_water_production_rate
        )
        net_oil_flow_rate_into_cell = cell_oil_injection_rate + cell_oil_production_rate
        net_gas_flow_rate_into_cell = cell_gas_injection_rate + cell_gas_production_rate

        # Calculate saturation changes for each phase
        # dS = Δt / (φ * V_cell) * [
        #     ([F_x_east - F_x_west] * Δy * Δz / Δx) + ([F_y_north - F_y_south] * Δx * Δz / Δy) + ([F_z_up - F_z_down] * Δx * Δy / Δz)
        #     + (q_x_ij * V)
        # ]
        # The change in saturation is (Net_Flux + Net_Well_Rate) * dt / Pore_Volume
        water_saturation_change = (
            (net_water_flux_from_neighbours + net_water_flow_rate_into_cell)
            * time_step_in_days
            / cell_pore_volume
        )
        oil_saturation_change = (
            (net_oil_flux_from_neighbours + net_oil_flow_rate_into_cell)
            * time_step_in_days
            / cell_pore_volume
        )
        gas_saturation_change = (
            (net_gas_flux_from_neighbours + net_gas_flow_rate_into_cell)
            * time_step_in_days
            / cell_pore_volume
        )

        # Update phase saturations
        updated_water_saturation_grid[i, j, k] = (
            cell_water_saturation + water_saturation_change
        )
        updated_oil_saturation_grid[i, j, k] = (
            cell_oil_saturation + oil_saturation_change
        )
        updated_gas_saturation_grid[i, j, k] = (
            cell_gas_saturation + gas_saturation_change
        )

    # Apply saturation constraints and normalization across all cells
    # This loop runs *after* all cells have been updated in the previous loop.
    for i, j, k in itertools.product(
        range(cell_count_x), range(cell_count_y), range(cell_count_z)
    ):
        # 1. Clip saturations to ensure they remain physically meaningful [max(Residual, 1.0e-6), 1.0 - 1.0e-6]
        # This ensures no saturation is exactly 0.0 or 1.0, which can cause numerical issues and saturation does not
        # go below residual values.
        min_water_saturation = max(irreducible_water_saturation_grid[i, j, k], 1e-6)
        min_oil_saturation = max(
            residual_oil_saturation_water_grid[i, j, k],
            residual_oil_saturation_gas_grid[i, j, k],
            1e-6,
        )
        min_gas_saturation = max(residual_gas_saturation_grid[i, j, k], 1e-6)
        updated_water_saturation_grid[i, j, k] = np.clip(
            updated_water_saturation_grid[i, j, k], min_water_saturation, 1.0 - 1e-6
        )
        updated_oil_saturation_grid[i, j, k] = np.clip(
            updated_oil_saturation_grid[i, j, k], min_oil_saturation, 1.0 - 1e-6
        )
        updated_gas_saturation_grid[i, j, k] = np.clip(
            updated_gas_saturation_grid[i, j, k], min_gas_saturation, 1.0 - 1e-6
        )

        # 2. Normalize saturations to ensure their sum is 1.0
        total_saturation = (
            updated_water_saturation_grid[i, j, k]
            + updated_oil_saturation_grid[i, j, k]
            + updated_gas_saturation_grid[i, j, k]
        )
        updated_water_saturation_grid[i, j, k] /= total_saturation
        updated_oil_saturation_grid[i, j, k] /= total_saturation
        updated_gas_saturation_grid[i, j, k] /= total_saturation

    return EvolutionResult(
        (
            updated_water_saturation_grid,
            updated_oil_saturation_grid,
            updated_gas_saturation_grid,
        ),
        scheme="explicit",
    )
