import functools
import itertools
import typing

import attrs
import numba
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from sim3D.boundary_conditions import BoundaryConditions
from sim3D.constants import (
    ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2,
    BBL_TO_FT3,
    DAYS_PER_SECOND,
    MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY,
)
from sim3D.grids import (
    build_three_phase_capillary_pressure_grids,
    build_three_phase_relative_mobilities_grids,
    build_total_fluid_compressibility_grid,
    edge_pad_grid,
)
from sim3D.models import FluidProperties, RockProperties
from sim3D.properties import (
    compute_diffusion_number,
    compute_harmonic_mobility,
    compute_three_phase_relative_permeabilities,
)
from sim3D.types import EvolutionScheme, T, ThreeDimensionalGrid, ThreeDimensions
from sim3D.wells import FluidPhase, Wells


###########################################################################
# MISCIBLE TWO-PHASE FLOW SOLVERS USING VISCOSITY MIXING (PROXY)
# -------------------------------------------------------------------------
# The Viscosity Mixing model is a simplified model that blends viscosities based on saturation
# and pressure to mimic miscibility effects.
#
# This model is suitable for simulating miscible two-phase flow in reservoirs
# where the injected fluid (e.g., CO₂) dissolves into the displaced fluid (e.g., oil).
#
# The model uses Darcy's law and an First-Order upwind finite difference method to compute
# the evolution of fluid saturation and reservoir pressure over time across a N-Dimensional reservoir grid.
#
# Here, we iterate over interior grid cells only (excluding boundary cells at edges).
# This exclusion is necessary because the (upwind) finite difference method
# used here relies on neighbouring cells (i±1, j or i, j±1) to compute
# pressure gradients and saturation fluxes. Accessing neighbours for boundary
# cells (i=0 or i=cell_count_x-1, etc.) would lead to out-of-bounds indexing errors.
#
# Additionally, in real-world reservoir simulations, boundary cells often have
# special physical conditions—such as fixed pressures, no-flow boundaries, or
# injection/production wells—which are typically handled using separate logic.
# Including them in this loop without appropriate treatment could result in
# non-physical results or numerical instability.
#
# Therefore, we safely exclude the outermost cells and perform updates only
# on interior cells where neighbour access is valid and consistent.
# ###########################################################################


@attrs.define(slots=True, frozen=True)
class EvolutionResult(typing.Generic[T]):
    value: T
    scheme: EvolutionScheme


"""
Explicit finite difference formulation for pressure diffusion in a N-Dimensional reservoir
(slightly compressible fluid):

The governing equation is the N-Dimensional linear-flow diffusivity equation:

    ∂p/∂t * (ρ·φ·c_t) * V = ∇ · (λ·∇p) * A + q * V

where:
    - ∂p/∂t * (ρ·φ·c_t) * V is the accumulation term (mass storage)
    - ∇ · (λ·∇p) * A is the diffusion term
    - q * V is the source/sink term (injection/production)

Assumptions:
    - Constant porosity (φ), total compressibility (c_t), and fluid density (ρ)
    - No convection or reaction terms
    - Cartesian grid (structured)
    - Fluid is slightly compressible; mobility is scalar

Simplified equation:

    ∂p/∂t * (φ·c_t) * V = ∇ · (λ·∇p) * A + q * V

Where:
    - V = cell volume = Δx * Δy * Δz
    - A = directional area across cell face
    - λ = mobility = k / μ
    - ∇p = pressure gradient
    - q = source/sink rate per unit volume

The diffusion term is expanded as:

    ∇ · (λ·∇p) = ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y) + ∂/∂z (λ·∂p/∂z)

Explicit Discretization (Forward Euler in time, central difference in space):

    ∂p/∂t ≈ (pⁿ⁺¹_ijk - pⁿ_ijk) / Δt

    ∂/∂x (λ·∂p/∂x) ≈ (λ_{i+½,j,k}(pⁿ_{i+1,j,k} - pⁿ_{i,j,k}) - λ_{i-½,j,k}(pⁿ_{i,j,k} - pⁿ_{i-1,j,k})) / Δx²
    ∂/∂y (λ·∂p/∂y) ≈ (λ_{i,j+½,k}(pⁿ_{i,j+1,k} - pⁿ_{i,j,k}) - λ_{i,j-½,k}(pⁿ_{i,j,k} - pⁿ_{i,j-1,k})) / Δy²
    ∂/∂z (λ·∂p/∂z) ≈ (λ_{i,j,k+½}(pⁿ_{i,j,k+1} - pⁿ_{i,j,k}) - λ_{i,j,k-½}(pⁿ_{i,j,k} - pⁿ_{i,j,k-1})) / Δz²

Final explicit update formula:

    pⁿ⁺¹_ijk = pⁿ_ijk + (Δt / (φ·c_t·V)) * [
        A_x * (λ_{i+½,j,k}(pⁿ_{i+1,j,k} - pⁿ_{i,j,k}) - λ_{i-½,j,k}(pⁿ_{i,j,k} - pⁿ_{i-1,j,k})) / Δx² +
        A_y * (λ_{i,j+½,k}(pⁿ_{i,j+1,k} - pⁿ_{i,j,k}) - λ_{i,j-½,k}(pⁿ_{i,j,k} - pⁿ_{i,j-1,k})) / Δy² +
        A_z * (λ_{i,j,k+½}(pⁿ_{i,j,k+1} - pⁿ_{i,j,k}) - λ_{i,j,k-½}(pⁿ_{i,j,k} - pⁿ_{i,j,k-1})) / Δz² +
        q_{i,j,k} * V
    ]

Where:
    - Δt is time step (s)
    - Δx, Δy, Δz are cell dimensions (ft)
    - A_x = Δy * Δz; A_y = Δx * Δz; A_z = Δx * Δy (ft²)
    - V = Δx * Δy * Δz (ft³)
    - λ_{i±½,...} = harmonic average of mobility between adjacent cells
    - q_{i,j,k} = injection/production rate per unit volume (ft³/s/ft³)

Stability Condition:
    Explicit scheme is conditionally stable. The N-Dimensional diffusion CFL criterion requires:

        D_x = λ·Δt / (φ·c_t·Δx²) < 1/6
        D_y = λ·Δt / (φ·c_t·Δy²) < 1/6
        D_z = λ·Δt / (φ·c_t·Δz²) < 1/6

Notes:
    - Harmonic averaging ensures continuity of flux across interfaces.
    - Volume-normalized source/sink terms only affect the cell where the well is located.
    - Can be easily adapted for anisotropic permeability by using λ_x, λ_y, λ_z separately.
"""


@numba.njit(cache=True)
def _compute_pseudo_volumetric_flux_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
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
) -> float:
    """
    Computes the total "pseudo" volumetric flux from a neighbour cell into the current cell
    based on the pressure differences, mobilities, and capillary pressures.

    The pseudo flux comes about due to the fact that we are are not using the pressure gradient, ∆P/∆x, or ∆P/∆y, or ∆P/∆z,
    but rather the pressure difference between the cells, ∆P.

    This will later be normalized by the cell geometric factor (A/∆x) to obtain the actual volumetric flow rate.

    This function calculates the pseudo volumetric flux for each phase (water, oil, gas)
    from the neighbour cell into the current cell using the harmonic mobility approach.
    The total volumetric flux is the sum of the individual phase fluxes, converted to ft³/day.
    The formula used is:

    q_total = (λ_w * Water Phase Potential Difference)
              + (λ_o * Oil Phase Potential Difference)
              + (λ_g * Gas Phase Potential Difference)

    Where:
        - λ_w, λ_o, λ_g are the harmonic mobilities of water, oil, and gas respectively.
        - Water Phase Potential Difference = [(P_neighbour - P_current) - (P_cow_neighbour - P_cow_current)]
                                            + (upwind_water_density * gravity * height_difference / 144.0)
        - Oil Phase Potential Difference = (P_neighbour - P_current) + (upwind_oil_density * gravity * height_difference / 144.0)
        - Gas Phase Potential Difference = [(P_neighbour + P_cgo_neighbour) - (P_current + P_cgo_current)]
                                            + (upwind_gas_density * gravity * height_difference / 144.0)
        - Upwinded densities are determined based on the pressure difference:
        - If the pressure difference is positive (P_neighbour - P_current > 0), we use the neighbour's density i.e.,
            Neighbour is upstream in the flow direction.
        - If the pressure difference is negative (P_neighbour - P_current < 0), we use the current cell's density.

    :param cell_indices: Indices of the current cell (i, j, k).
    :param neighbour_indices: Indices of the neighbour cell (i±1, j, k) or (i, j±1, k) or (i, j, k±1).
    :param oil_pressure_grid: N-Dimensional numpy array representing the oil phase pressure grid (psi).
    :param water_mobility_grid: N-Dimensional numpy array representing the water phase mobility grid (ft²/psi/day).
    :param oil_mobility_grid: N-Dimensional numpy array representing the oil phase mobility grid (ft²/psi/day).
    :param gas_mobility_grid: N-Dimensional numpy array representing the gas phase mobility grid (ft²/psi/day).
    :param oil_water_capillary_pressure_grid: N-Dimensional numpy array representing the oil-water capillary pressure grid (psi).
    :param gas_oil_capillary_pressure_grid: N-Dimensional numpy array representing the gas-oil capillary pressure grid (psi).

    The following parameters are optional and are only necessary when computing flux in the z-direction:

    :param oil_density_grid: N-Dimensional numpy array representing the oil phase density grid (lb/ft³).
    :param water_density_grid: N-Dimensional numpy array representing the water phase density grid (lb/ft³).
    :param gas_density_grid: N-Dimensional numpy array representing the gas phase density grid (lb/ft³).
    :param elevation_grid: N-Dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :return: Total pseudo volumetric flux from neighbour to current cell (ft²/day).
    """
    cell_oil_pressure = oil_pressure_grid[cell_indices]
    cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[cell_indices]
    neighbour_oil_pressure = oil_pressure_grid[neighbour_indices]
    neighbour_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[
        neighbour_indices
    ]
    neighbour_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[
        neighbour_indices
    ]

    # Calculate pressure differences relative to current cell (Neighbour - Current)
    # These represent the gradients driving flow from current to neighbour, or vice versa
    oil_pressure_difference = neighbour_oil_pressure - cell_oil_pressure
    oil_water_capillary_pressure_difference = (
        neighbour_oil_water_capillary_pressure - cell_oil_water_capillary_pressure
    )
    water_pressure_difference = (
        oil_pressure_difference - oil_water_capillary_pressure_difference
    )
    # Gas pressure difference is calculated as:
    gas_pressure_difference = (
        neighbour_oil_pressure + neighbour_gas_oil_capillary_pressure
    ) - (cell_oil_pressure + cell_gas_oil_capillary_pressure)

    if elevation_grid is not None:
        # Calculate the elevation difference between the neighbour and current cell
        elevation_delta = (
            elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
        )
    else:
        elevation_delta = 0.0

    # Determine the upwind densities based on pressure difference
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

    # Calculate harmonic mobilities for each phase across the face (in the direction of flow)
    water_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices,
        index2=neighbour_indices,
        mobility_grid=water_mobility_grid,
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices,
        index2=neighbour_indices,
        mobility_grid=oil_mobility_grid,
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices,
        index2=neighbour_indices,
        mobility_grid=gas_mobility_grid,
    )

    # Calculate volumetric flux for each phase from neighbour INTO the current cell
    # Flux_in = λ * 'Phase Potential Difference'
    # P_water_neighbour - P_water_current = (P_oil_neighbour - P_oil_current) - (neighbour_cell_oil_water_capillary_pressure - cell_oil_water_capillary_pressure)
    # P_gas_neighbour - P_gas_current = (P_oil_neighbour - P_oil_current) + (neighbour_cell_gas_oil_capillary_pressure - cell_gas_oil_capillary_pressure)

    # NOTE: Phase potential differences is the same as the pressure difference

    # For Oil and Water:
    # q = λ * (∆P + Gravity Potential) (ft³/day)
    # Calculate the water gravity potential (hydrostatic/gravity head)
    water_gravity_potential = (
        upwind_water_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total water phase potential
    water_phase_potential = water_pressure_difference + water_gravity_potential
    # Calculate the volumetric flux of water from neighbour to current cell
    water_pseudo_volumetric_flux_from_neighbour = (
        water_harmonic_mobility * water_phase_potential
    )

    # For Oil:
    # Calculate the oil gravity potential (gravity head)
    oil_gravity_potential = (
        upwind_oil_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total oil phase potential
    oil_phase_potential = oil_pressure_difference + oil_gravity_potential
    # Calculate the volumetric flux of oil from neighbour to current cell
    oil_pseudo_volumetric_flux_from_neighbour = (
        oil_harmonic_mobility * oil_phase_potential
    )

    # For Gas:
    # q = λ * ∆P (ft²/day)
    # Calculate the gas gravity potential (gravity head)
    gas_gravity_potential = (
        upwind_gas_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total gas phase potential
    gas_phase_potential = gas_pressure_difference + gas_gravity_potential
    # Calculate the volumetric flux of gas from neighbour to current cell
    gas_pseudo_volumetric_flux_from_neighbour = (
        gas_harmonic_mobility * gas_phase_potential
    )

    # Add these incoming fluxes to the net total for the cell, q (ft²/day)
    total_pseudo_volumetric_flux_from_neighbour = (
        water_pseudo_volumetric_flux_from_neighbour
        + oil_pseudo_volumetric_flux_from_neighbour
        + gas_pseudo_volumetric_flux_from_neighbour
    )
    return total_pseudo_volumetric_flux_from_neighbour


def compute_explicit_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_size: float,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
) -> EvolutionResult[ThreeDimensionalGrid]:
    """
    Computes the pressure evolution (specifically, oil phase pressure P_oil) in the reservoir grid
    for one time step using an explicit finite difference method. This function incorporates
    three-phase flow (water, oil, gas), phase-dependent mobility (derived from relative permeabilities
    and individual phase viscosities), and explicit capillary pressure gradients (both oil-water
    and gas-oil) as contributions to the pressure change. Wettability (water-wet or oil-wet)
    is accounted for in the capillary pressure calculations.

    The governing equation for pressure evolution (derived from total fluid conservation) is:
        (φ·c_t) · ∂P_oil/∂t = ∇ · [ (λ_o + λ_w + λ_g) · ∇P_oil ]
                                  - ∇ · [ λ_w · ∇P_cow ]
                                  + ∇ · [ λ_g · ∇P_cgo ]
                                  + q_total

    Where:
        P_oil     = Oil phase pressure (psi) - the primary unknown being solved for.
        φ         = porosity (fraction).
        c_t = total compressibility of the system (psi⁻¹), including rock compressibility
                    and saturation-weighted fluid compressibilities.
        λ_o, λ_w, λ_g = mobilities of oil, water, and gas phases respectively (m²/(Pa·s)).
                        Calculated as (absolute_permeability * relative_permeability) / viscosity.
        P_cow     = Capillary pressure between oil and water (Po - Pw), function of S_w and wettability.
        P_cgo     = Capillary pressure between gas and oil (Pg - Po), function of S_g.
        q_total   = total source/sink term (injection + production) (m³/s).

    Capillary pressure gradient terms are treated explicitly (using saturations from the previous time step)
    and contribute to the right-hand side as source terms.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet (ft).
    :param thickness_grid: N-Dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :param elevation_grid: N-Dimensional numpy array representing the elevation of each cell in the reservoir (ft).
    :param time_step_size: Time step size (s) for each iteration.
    :param boundary_conditions: Boundary conditions for pressure and saturation grids.
    :param rock_properties: `RockProperties` object containing rock physical properties including
        absolute permeability, porosity, residual saturations, and relative/capillary
        pressure parameters (which include wettability information).

    :param fluid_properties: `FluidProperties` object containing fluid physical properties like
        pressure, temperature, saturations, viscosities, and compressibilities for
        water, oil, and gas.

    :param wells: `Wells` object containing well parameters for injection and production wells
    :return: A N-Dimensional numpy array representing the updated oil phase pressure field (psi).
    """
    # Extract properties from provided objects for clarity and convenience
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid
    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_grid = rock_properties.residual_oil_saturation_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_params = rock_properties.relative_permeability_params
    capillary_pressure_params = (
        rock_properties.capillary_pressure_params
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

    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid
    )
    oil_formation_volume_factor_grid = fluid_properties.oil_formation_volume_factor_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid

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
    padded_water_saturation_grid = edge_pad_grid(current_water_saturation_grid)
    padded_gas_saturation_grid = edge_pad_grid(current_gas_saturation_grid)
    padded_irreducible_water_saturation_grid = edge_pad_grid(
        irreducible_water_saturation_grid
    )
    padded_residual_oil_saturation_grid = edge_pad_grid(residual_oil_saturation_grid)
    padded_residual_gas_saturation_grid = edge_pad_grid(residual_gas_saturation_grid)
    padded_oil_density_grid = edge_pad_grid(oil_density_grid)
    padded_water_density_grid = edge_pad_grid(water_density_grid)
    padded_gas_density_grid = edge_pad_grid(gas_density_grid)
    padded_elevation_grid = edge_pad_grid(elevation_grid)

    # Apply boundary conditions to relevant padded grids
    boundary_conditions["pressure"].apply(padded_oil_pressure_grid)
    boundary_conditions["water_saturation"].apply(padded_water_saturation_grid)
    boundary_conditions["gas_saturation"].apply(padded_gas_saturation_grid)

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
        residual_oil_saturation_grid=residual_oil_saturation_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        relative_permeability_params=relative_permeability_params,
    )

    # Compute mobility grids for x, y, z directions
    # λ_x = k_abs * (kr / mu) / B_x. Where B_x is the formation volume factor of the phase
    water_mobility_grid_x = (
        absolute_permeability.x
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_x = (
        absolute_permeability.x
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_x = (
        absolute_permeability.x
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    water_mobility_grid_y = (
        absolute_permeability.y
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_y = (
        absolute_permeability.y
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_y = (
        absolute_permeability.y
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    water_mobility_grid_z = (
        absolute_permeability.z
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_z = (
        absolute_permeability.z
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_z = (
        absolute_permeability.z
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    # Clamp mobility grids to avoid numerical issues
    # This ensures mobilities are never zero or negative (for numerical stability)
    water_mobility_grid_x = np.maximum(water_mobility_grid_x, 1e-12)
    oil_mobility_grid_x = np.maximum(oil_mobility_grid_x, 1e-12)
    gas_mobility_grid_x = np.maximum(gas_mobility_grid_x, 1e-12)
    water_mobility_grid_y = np.maximum(water_mobility_grid_y, 1e-12)
    oil_mobility_grid_y = np.maximum(oil_mobility_grid_y, 1e-12)
    gas_mobility_grid_y = np.maximum(gas_mobility_grid_y, 1e-12)
    water_mobility_grid_z = np.maximum(water_mobility_grid_z, 1e-12)
    oil_mobility_grid_z = np.maximum(oil_mobility_grid_z, 1e-12)
    gas_mobility_grid_z = np.maximum(gas_mobility_grid_z, 1e-12)

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
            residual_oil_saturation_grid=padded_residual_oil_saturation_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    # Initialize a new grid to store the updated pressures for the current time step
    updated_oil_pressure_grid = current_oil_pressure_grid.copy()

    # --- ITERATE OVER INTERIOR CELLS FOR EXPLICIT PRESSURE UPDATE ---
    for i, j, k in itertools.product(
        range(cell_count_x), range(cell_count_y), range(cell_count_z)
    ):
        ip, jp, kp = (i + 1, j + 1, k + 1)  # Padded indices for current cell (i, j, k)
        cell_thickness = thickness_grid[i, j, k]  # Same as `cell_size_z`
        cell_volume = cell_size_x * cell_size_y * cell_thickness
        cell_porosity = porosity_grid[i, j, k]
        cell_total_compressibility = total_compressibility_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]
        cell_temperature = fluid_properties.temperature_grid[i, j, k]

        flux_configurations = {
            "x": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_x,
                    "oil_mobility_grid": padded_oil_mobility_grid_x,
                    "gas_mobility_grid": padded_gas_mobility_grid_x,
                },
                "positive_neighbour": (ip + 1, jp, kp),
                "negative_neighbour": (ip - 1, jp, kp),
                "geometric_factor": (cell_size_y * cell_thickness / (cell_size_x**2)),
            },
            "y": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_y,
                    "oil_mobility_grid": padded_oil_mobility_grid_y,
                    "gas_mobility_grid": padded_gas_mobility_grid_y,
                },
                "positive_neighbour": (ip, jp + 1, kp),
                "negative_neighbour": (ip, jp - 1, kp),
                "geometric_factor": (cell_size_x * cell_thickness / (cell_size_y**2)),
            },
            "z": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_z,
                    "oil_mobility_grid": padded_oil_mobility_grid_z,
                    "gas_mobility_grid": padded_gas_mobility_grid_z,
                },
                "positive_neighbour": (ip, jp, kp + 1),
                "negative_neighbour": (ip, jp, kp - 1),
                "geometric_factor": (cell_size_x * cell_size_y / (cell_thickness**2)),
            },
        }

        # Compute the net volumetric flow rate into the current cell (i, j, k)
        # A_x * (λ_{i+1/2,j,k}(pⁿ_{i+1,j,k} - pⁿ_{i,j,k}) - λ_{i-1/2,j,k}(pⁿ_{i,j,k} - pⁿ_{i-1,j,k})) / Δx² +
        # A_y * (λ_{i,j+1/2,k}(pⁿ_{i,j+1,k} - pⁿ_{i,j,k}) - λ_{i,j-1/2,k}(pⁿ_{i,j,k} - pⁿ_{i,j-1,k})) / Δy² +
        # A_z * (λ_{i,j,k+1/2}(pⁿ_{i,j,k+1} - pⁿ_{i,j,k}) - λ_{i,j,k-1/2}(pⁿ_{i,j,k} - pⁿ_{i,j,k-1})) / Δz²
        net_volumetric_flow_rate_into_cell = 0.0
        for direction, configuration in flux_configurations.items():
            mobility_grids = configuration["mobility_grids"]
            positive_neighbour_indices = configuration["positive_neighbour"]
            negative_neighbour_indices = configuration["negative_neighbour"]

            # Include the density and elevation grids only for z-direction
            # For gravity segregation to be included/modelled in the flux computation
            if direction == "z":
                flux_oil_density_grid = padded_oil_density_grid
                flux_water_density_grid = padded_water_density_grid
                flux_gas_density_grid = padded_gas_density_grid
                flux_elevation_grid = padded_elevation_grid
            else:
                flux_oil_density_grid = None
                flux_water_density_grid = None
                flux_gas_density_grid = None
                flux_elevation_grid = None

            # Compute the total flux from positive and negative neighbours
            positive_pseudo_flux = _compute_pseudo_volumetric_flux_from_neighbour(
                cell_indices=(ip, jp, kp),
                neighbour_indices=positive_neighbour_indices,
                oil_pressure_grid=padded_oil_pressure_grid,
                **mobility_grids,
                oil_water_capillary_pressure_grid=padded_oil_water_capillary_pressure_grid,
                gas_oil_capillary_pressure_grid=padded_gas_oil_capillary_pressure_grid,
                oil_density_grid=flux_oil_density_grid,
                water_density_grid=flux_water_density_grid,
                gas_density_grid=flux_gas_density_grid,
                elevation_grid=flux_elevation_grid,
            )
            # This already incorporates the outer negative sign in the function
            # as it does (P_negative_neighbour - P_cell) instead of (P_cell - P_negative_neighbour)
            negative_pseudo_flux = _compute_pseudo_volumetric_flux_from_neighbour(
                cell_indices=(ip, jp, kp),
                neighbour_indices=negative_neighbour_indices,
                oil_pressure_grid=padded_oil_pressure_grid,
                **mobility_grids,
                oil_water_capillary_pressure_grid=padded_oil_water_capillary_pressure_grid,
                gas_oil_capillary_pressure_grid=padded_gas_oil_capillary_pressure_grid,
                oil_density_grid=flux_oil_density_grid,
                water_density_grid=flux_water_density_grid,
                gas_density_grid=flux_gas_density_grid,
                elevation_grid=flux_elevation_grid,
            )
            # So we just add directly here instead of subtracting
            net_pseudo_flux_in_direction = positive_pseudo_flux + negative_pseudo_flux
            net_flow_rate_in_direction = (
                net_pseudo_flux_in_direction * configuration["geometric_factor"]
            )
            net_volumetric_flow_rate_into_cell += net_flow_rate_in_direction

        # Add Source/Sink Term (WellParameters) - q * V (ft³/day)
        injection_well, production_well = wells[i, j, k]
        cell_injection_rate = 0.0
        cell_production_rate = 0.0
        permeability = (
            absolute_permeability.x[i, j, k],
            absolute_permeability.y[i, j, k],
            absolute_permeability.z[i, j, k],
        )
        if (
            injection_well is not None
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            # If there is an injection well, add its flow rate to the cell
            injected_phase = injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                phase_mobility = gas_mobility_grid_z[i, j, k]
            elif injected_phase == FluidPhase.WATER:
                phase_mobility = water_mobility_grid_z[i, j, k]
            else:
                phase_mobility = oil_mobility_grid_z[i, j, k]

            well_index = injection_well.get_well_index(
                interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                permeability=permeability,
            )
            cell_injection_rate = injection_well.get_flow_rate(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_mobility=phase_mobility,
                fluid=injected_fluid,
            )  # STB/day or SCF/day
            if injected_phase == FluidPhase.GAS:
                # Get the volumetric flow rate in ft³/day
                gas_fvf = (
                    injected_fluid.formation_volume_factor
                    or fluid_properties.gas_formation_volume_factor_grid[i, j, k]
                )
                cell_injection_rate *= gas_fvf  # ft³/SCF
            elif injected_phase == FluidPhase.WATER:
                # For water, convert bbl/day to ft³/day
                water_fvf = (
                    injected_fluid.formation_volume_factor
                    or fluid_properties.water_formation_volume_factor_grid[i, j, k]
                )  # bbl/STB
                cell_injection_rate *= (
                    water_fvf * BBL_TO_FT3
                )  # Convert bbl/day to ft³/day
            else:
                # For oil and water, convert bbl/day to ft³/day
                cell_injection_rate *= (
                    fluid_properties.oil_formation_volume_factor_grid[
                        i, j, k
                    ]  # bbl/STB
                    * BBL_TO_FT3  # Convert bbl/day to ft³/day
                )

        if production_well is not None:
            # If there is a production well, subtract its flow rate from the cell
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = gas_mobility_grid_y[i, j, k]
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = water_mobility_grid_y[i, j, k]
                else:
                    phase_mobility = oil_mobility_grid_y[i, j, k]

                well_index = production_well.get_well_index(
                    interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                    permeability=permeability,
                )
                production_rate = production_well.get_flow_rate(
                    pressure=cell_oil_pressure,
                    temperature=cell_temperature,
                    well_index=well_index,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                )  # STB/day or SCF/day

                if produced_fluid.phase == FluidPhase.GAS:
                    # Get the volumetric flow rate in ft³/day
                    gas_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.gas_formation_volume_factor_grid[i, j, k]
                    )  # ft³/SCF
                    production_rate *= gas_fvf
                elif produced_fluid.phase == FluidPhase.WATER:
                    # For water, convert bbl/day to ft³/day
                    water_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.water_formation_volume_factor_grid[i, j, k]
                    )  # bbl/STB
                    production_rate *= (
                        water_fvf * BBL_TO_FT3
                    )  # Convert bbl/day to ft³/day
                else:
                    # For oil, convert bbl/day to ft³/day
                    oil_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.oil_formation_volume_factor_grid[i, j, k]
                    )  # bbl/STB
                    production_rate *= (
                        oil_fvf * BBL_TO_FT3
                    )  # Convert bbl/day to ft³/day

                cell_production_rate += production_rate

        # Calculate the net well flow rate into the cell. Just add injection and production rates (since production rates are negative)
        # q_{i,j,k} * V = (q_{i,j,k}_injection - q_{i,j,k}_production)
        net_well_flow_rate_into_cell = cell_injection_rate + cell_production_rate

        # Add the well flow rate to the net volumetric flow rate into the cell
        # Total flow rate into the cell (ft³/day)
        # Total flow rate = Net Volumetric Flow Rate + Net Well Flow Rate
        # [ A_x * (λ_{i+1/2,j,k}(pⁿ_{i+1,j,k} - pⁿ_{i,j,k}) - λ_{i-1/2,j,k}(pⁿ_{i,j,k} - pⁿ_{i-1,j,k})) / Δx² +
        # A_y * (λ_{i,j+1/2,k}(pⁿ_{i,j+1,k} - pⁿ_{i,j,k}) - λ_{i,j-1/2,k}(pⁿ_{i,j,k} - pⁿ_{i,j-1,k})) / Δy² +
        # A_z * (λ_{i,j,k+1/2}(pⁿ_{i,j,k+1} - pⁿ_{i,j,k}) - λ_{i,j,k-1/2}(pⁿ_{i,j,k} - pⁿ_{i,j,k-1})) / Δz² ] +
        # (q_{i,j,k} * V)
        total_flow_rate_into_cell = (
            net_volumetric_flow_rate_into_cell + net_well_flow_rate_into_cell
        )

        # Full Explicit Pressure Update Equation (P_oil) ---
        # The accumulation term is (φ * C_t * cell_volume) * dP_oil/dt
        # dP_oil/dt = [Net_Volumetric_Flow_Rate_Into_Cell] / (φ * C_t * cell_volume)
        # dP_{i,j} = (Δt / (φ·c_t·V)) * [
        #     A_x * (λ_{i+1/2,j,k}(pⁿ_{i+1,j,k} - pⁿ_{i,j,k}) - λ_{i-1/2,j,k}(pⁿ_{i,j,k} - pⁿ_{i-1,j,k})) / Δx² +
        #     A_y * (λ_{i,j+1/2,k}(pⁿ_{i,j+1,k} - pⁿ_{i,j,k}) - λ_{i,j-1/2,k}(pⁿ_{i,j,k} - pⁿ_{i,j-1,k})) / Δy² +
        #     A_z * (λ_{i,j,k+1/2}(pⁿ_{i,j,k+1} - pⁿ_{i,j,k}) - λ_{i,j,k-1/2}(pⁿ_{i,j,k} - pⁿ_{i,j,k-1})) / Δz² +
        #     q_{i,j,k} * V
        # ]
        time_step_size_in_days = time_step_size * DAYS_PER_SECOND
        change_in_pressure = (
            time_step_size_in_days
            / (cell_porosity * cell_total_compressibility * cell_volume)
        ) * total_flow_rate_into_cell

        # Apply the update to the pressure grid
        # P_oil^(n+1) = P_oil^n + dP_oil
        updated_oil_pressure_grid[i, j, k] += change_in_pressure
    return EvolutionResult(updated_oil_pressure_grid, scheme="explicit")


"""
Implicit finite difference formulation for pressure diffusion in a 3D reservoir
(slightly compressible fluid):

The governing equation for pressure evolution is the linear-flow diffusivity equation:

    ∂p/∂t * (φ·c_t) * V = ∇·(λ·∇p) * A + q * V

Where:
    ∂p/∂t * (φ·c_t) * V — Accumulation term
    ∇·(λ·∇p) * A — Diffusion term (Darcy's law)
    q * V — Source/sink term

Assumptions:
    - Constant porosity (φ), compressibility (c_t), and density (ρ)
    - No reaction or advection terms (pressure-only evolution)
    - Capillary effects optional, appear in source term via pressure corrections

Diffusion term expanded in 3D:

    ∇·(λ·∇p) = ∂/∂x(λ·∂p/∂x) + ∂/∂y(λ·∂p/∂y) + ∂/∂z(λ·∂p/∂z)

Discretization:

Using Backward Euler in time:

    ∂p/∂t ≈ (pⁿ⁺¹_ijk - pⁿ_ijk) / Δt

Using central differences in space:

    ∂/∂x(λ·∂p/∂x) ≈ [λ_{i+½,j,k}·(pⁿ⁺¹_{i+1,j,k} - pⁿ⁺¹_{i,j,k}) - λ_{i⁻½,j,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i⁻1,j,k})] / Δx²
    ∂/∂y(λ·∂p/∂y) ≈ [λ_{i,j+½,k}·(pⁿ⁺¹_{i,j+1,k} - pⁿ⁺¹_{i,j,k}) - λ_{i,j⁻½,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j⁻1,k})] / Δy²
    ∂/∂z(λ·∂p/∂z) ≈ [λ_{i,j,k+½}·(pⁿ⁺¹_{i,j,k+1} - pⁿ⁺¹_{i,j,k}) - λ_{i,j,k⁻½}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j,k⁻1})] / Δz²

Putting it all together:

    (pⁿ⁺¹_ijk - pⁿ_ijk) * (φ·c_t·V) / Δt =
        A/Δx² · [λ_{i+½,j,k}·(pⁿ⁺¹_{i+1,j,k} - pⁿ⁺¹_{i,j,k}) - λ_{i⁻½,j,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i⁻1,j,k})] +
        A/Δy² · [λ_{i,j+½,k}·(pⁿ⁺¹_{i,j+1,k} - pⁿ⁺¹_{i,j,k}) - λ_{i,j⁻½,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j⁻1,k})] +
        A/Δz² · [λ_{i,j,k+½}·(pⁿ⁺¹_{i,j,k+1} - pⁿ⁺¹_{i,j,k}) - λ_{i,j,k⁻½}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j,k⁻1})] +
        qⁿ⁺¹_ijk * V

Matrix form:

Let:
    Tx⁺ = λ_{i+½,j,k}·A / Δx²
    Tx⁻ = λ_{i⁻½,j,k}·A / Δx²
    Ty⁺ = λ_{i,j+½,k}·A / Δy²
    Ty⁻ = λ_{i,j⁻½,k}·A / Δy²
    Tz⁺ = λ_{i,j,k+½}·A / Δz²
    Tz⁻ = λ_{i,j,k⁻½}·A / Δz²
    β   = φ·c_t·V / Δt

Then:

    A_{ijk,ijk}     = β + Tx⁺ + Tx⁻ + Ty⁺ + Ty⁻ + Tz⁺ + Tz⁻
    A_{ijk,i+1jk}   = -Tx⁺
    A_{ijk,i-1jk}   = -Tx⁻
    A_{ijk,ij+1k}   = -Ty⁺
    A_{ijk,ij-1k}   = -Ty⁻
    A_{ijk,ijk+1}   = -Tz⁺
    A_{ijk,ijk-1}   = -Tz⁻

RHS vector: (Contains terms that actually drive flow)

    b_{ijk} = (β * pⁿ_{ijk}) + (q_{ijk} * V) + Total Capillary Driven Flow + Gravity Driven Flow/Segregation

Capillary pressure driven flow term (if multiphase):

    total_capillary_flow = sum of directional contributions:
        For each direction:
            [(λ_w * ∇P_cow) + (λ_g * ∇P_cgo)] * A / (Δx², Δy², Δz²)

Gravity driven segregation (only in effect in the z-direction):

    total_gravity_flow = (
            [λ_w * (upwind_ρ_w * g * ∆z) / 144] 
            + [λ_g * (upwind_ρ_g * g * ∆z) / 144] 
            + [λ_o * (upwind_ρ_o * g * ∆z) / 144]
    ) * A / Δz²

    Where;
    g is the gravitational acceleration (32.174 ft/s²),
    upwind_ρ_w, upwind_ρ_g, upwind_ρ_o are the densities of water, gas, and oil

This results in a 7-point stencil sparse matrix (in 3D) for solving A·pⁿ⁺¹ = b.

Notes:
    - Harmonic averaging is used for λ at cell interfaces
    - Capillary pressure is optional but included via ∇P_cow and ∇P_cgo terms
    - Units: ft (length), s (time), psi (pressure), cP (viscosity), mD (permeability)
    - The system is solved each time step to advance pressure implicitly

Stability:
    - Fully implicit scheme is unconditionally stable for linear pressure diffusion
"""


@numba.njit(cache=True)
def to_1D_index(
    i: int,
    j: int,
    k: int,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> int:
    """
    Converts 3D grid indices (i, j, k) to a 1D index for the sparse matrix.
    The indexing is done in z-fastest order: (i, j, k) -> i * (Ny * Nz) + j * Nz + k

    :param i: Index along x-axis
    :param j: Index along y-axis
    :param k: Index along z-axis
    :param cell_count_x: Total number of cells in x-direction
    :param cell_count_y: Total number of cells in y-direction
    :param cell_count_z: Total number of cells in z-direction
    :return: Flattened 1D index
    """
    if not (0 <= i < cell_count_x and 0 <= j < cell_count_y and 0 <= k < cell_count_z):
        return -1  # Out of bounds
    return i * (cell_count_y * cell_count_z) + j * cell_count_z + k


@numba.njit(cache=True)
def _compute_pseudo_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
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
    """
    Computes and returns a tuple of the total harmonic mobility of the phases, the capillary pseudo flux,
    and the gravity pseudo flux from the neighbour to the current cell.

    Pseudo flux comes about from the fact that the fluxes returned are not actual fluxes with units of ft/day,
    but rather pseudo fluxes with units of ft²/day, which are then used to compute the actual fluxes and subsequently,
    the flow rates in the implicit pressure evolution scheme.

    :param cell_indices: Indices of the current cell (i, j, k)
    :param neighbour_indices: Indices of the neighbouring cell (i±1, j, k) or (i, j±1, k) or (i, j, k±1)
    :param oil_pressure_grid: 3D grid of oil pressures (psi)
    :param water_mobility_grid: 3D grid of water mobilities (ft²/psi.day)
    :param oil_mobility_grid: 3D grid of oil mobilities (ft²/psi.day)
    :param gas_mobility_grid: 3D grid of gas mobilities (ft²/psi.day)
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi)
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi)
    :param oil_density_grid: 3D grid of oil densities (lb/ft³), optional
    :param water_density_grid: 3D grid of water densities (lb/ft³), optional
    :param gas_density_grid: 3D grid of gas densities (lb/ft³), optional
    :param elevation_grid: 3D grid of elevations (ft), optional
    :return: A tuple containing:
        - Total harmonic mobility (ft²/psi.day)
        - Total capillary pseudo flux (ft²/day)
        - Total gravity pseudo flux (ft²/day)
    """
    cell_oil_pressure = oil_pressure_grid[cell_indices]
    cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[cell_indices]
    neighbour_oil_pressure = oil_pressure_grid[neighbour_indices]
    neighbour_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[
        neighbour_indices
    ]
    neighbour_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[
        neighbour_indices
    ]

    # Calculate pressure differences relative to current cell (Neighbour - Current)
    # These represent the gradients driving flow from neighbour to current cell, or vice versa
    oil_pressure_difference = neighbour_oil_pressure - cell_oil_pressure
    oil_water_capillary_pressure_difference = (
        neighbour_oil_water_capillary_pressure - cell_oil_water_capillary_pressure
    )
    gas_oil_capillary_pressure_difference = (
        neighbour_gas_oil_capillary_pressure - cell_gas_oil_capillary_pressure
    )
    water_pressure_difference = (
        oil_pressure_difference - oil_water_capillary_pressure_difference
    )
    # Gas pressure difference is calculated as:
    gas_pressure_difference = (
        neighbour_oil_pressure + neighbour_gas_oil_capillary_pressure
    ) - (cell_oil_pressure + cell_gas_oil_capillary_pressure)

    if elevation_grid is not None:
        # Calculate the elevation difference between the neighbour and current cell
        elevation_delta = (
            elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
        )
    else:
        elevation_delta = 0.0

    # Determine the upwind densities based on pressure difference
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

    # Calculate harmonic mobilities for each phase across the face
    water_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices,
        index2=neighbour_indices,
        mobility_grid=water_mobility_grid,
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices,
        index2=neighbour_indices,
        mobility_grid=oil_mobility_grid,
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices,
        index2=neighbour_indices,
        mobility_grid=gas_mobility_grid,
    )
    total_harmonic_mobility = (
        water_harmonic_mobility + oil_harmonic_mobility + gas_harmonic_mobility
    )

    # λ_w * (P_cow_{n+1} - P_cow_{n}) (ft²/psi.day * psi = ft²/day)
    water_capillary_pseudo_flux = (
        water_harmonic_mobility * oil_water_capillary_pressure_difference
    )
    # λ_g * (P_cgo_{n+1} - P_cgo_{n}) (ft²/psi.day * psi = ft²/day)
    gas_capillary_pseudo_flux = (
        gas_harmonic_mobility * gas_oil_capillary_pressure_difference
    )
    # Total capillary flux from the neighbour (ft²/day)
    total_capillary_pseudo_flux = (
        water_capillary_pseudo_flux + gas_capillary_pseudo_flux
    )

    # Calculate the phase gravity potentials (hydrostatic/gravity head)
    water_gravity_potential = (
        upwind_water_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    oil_gravity_potential = (
        upwind_oil_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    gas_gravity_potential = (
        upwind_gas_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Total gravity pseudo flux (ft²/day)
    water_gravity_pseudo_flux = water_harmonic_mobility * water_gravity_potential
    oil_gravity_pseudo_flux = oil_harmonic_mobility * oil_gravity_potential
    gas_gravity_pseudo_flux = gas_harmonic_mobility * gas_gravity_potential
    total_gravity_pseudo_flux = (
        water_gravity_pseudo_flux + oil_gravity_pseudo_flux + gas_gravity_pseudo_flux
    )
    return (
        total_harmonic_mobility,
        total_capillary_pseudo_flux,
        total_gravity_pseudo_flux,
    )


def compute_implicit_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_size: float,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    capillary_pressure_stability_factor: float = 1.0,
) -> EvolutionResult[ThreeDimensionalGrid]:
    """
    Solves the fully implicit finite-difference pressure equation for a slightly compressible,
    three-phase flow system in a 3D reservoir.

    Constructs and solves the linear system A·Pⁿ⁺¹ = b using backward Euler time integration.
    The coefficient matrix A includes accumulation and transmissibility terms. The right-hand side
    vector b includes contributions from accumulation, well source/sink terms, and capillary pressure gradients.

    :param cell_dimension: Cell dimensions (Δx, Δy) used to compute transmissibility and volume.
    :param thickness_grid: Grid of cell heights (Δz) defining vertical thickness per cell.
    :param elevation_grid: Grid of cell elevations (ft) defining the top of each cell.
    :param time_step_size: Time step size in seconds for the implicit update.
    :param boundary_conditions: Dictionary of callable boundary conditions applied to pressure and saturations.
    :param rock_properties: Rock data including porosity, permeability, compressibility, saturation endpoints,
        and relative permeability/capillary pressure parameters.
    :param fluid_properties: Current pressure and saturation fields along with viscosity, compressibility,
        and formation volume factors for each phase.
    :param wells: Well configuration and parameters for injectors and producers including phase, location,
        orientation, radius, skin factor, and BHP.
    :param capillary_pressure_stability_factor: Factor to scale capillary flow for numerical stability.
        Capillary gradients can become numerically dominant in fine meshes or sharp saturation fronts.
        Damping avoids overshoot/undershoot by reducing their contribution without removing them.

    :return: Updated 3D oil pressure grid at the new time level (Pⁿ⁺¹).
    """
    # Extract properties from provided objects for clarity and convenience
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid
    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_grid = rock_properties.residual_oil_saturation_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_params = rock_properties.relative_permeability_params
    capillary_pressure_params = (
        rock_properties.capillary_pressure_params
    )  # Contains wettability type

    current_oil_pressure_grid = (
        fluid_properties.pressure_grid
    )  # This is P_oil or Pⁿ_{i,j,k}
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_viscosity_grid = fluid_properties.water_viscosity_grid
    oil_viscosity_grid = fluid_properties.oil_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid
    )
    oil_formation_volume_factor_grid = fluid_properties.oil_formation_volume_factor_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid

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
    padded_water_saturation_grid = edge_pad_grid(current_water_saturation_grid)
    padded_gas_saturation_grid = edge_pad_grid(current_gas_saturation_grid)
    padded_irreducible_water_saturation_grid = edge_pad_grid(
        irreducible_water_saturation_grid
    )
    padded_residual_oil_saturation_grid = edge_pad_grid(residual_oil_saturation_grid)
    padded_residual_gas_saturation_grid = edge_pad_grid(residual_gas_saturation_grid)
    padded_oil_density_grid = edge_pad_grid(oil_density_grid)
    padded_water_density_grid = edge_pad_grid(water_density_grid)
    padded_gas_density_grid = edge_pad_grid(gas_density_grid)
    padded_elevation_grid = edge_pad_grid(elevation_grid)

    # Apply boundary conditions to relevant padded grids
    boundary_conditions["pressure"].apply(padded_oil_pressure_grid)
    boundary_conditions["water_saturation"].apply(padded_water_saturation_grid)
    boundary_conditions["gas_saturation"].apply(padded_gas_saturation_grid)

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
        residual_oil_saturation_grid=residual_oil_saturation_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        relative_permeability_params=relative_permeability_params,
    )

    # Compute mobility grids for x, y, z directions
    # λ_x = k_abs * (kr / mu) / B_x. Where B_x is the formation volume factor of the phase
    water_mobility_grid_x = (
        absolute_permeability.x
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_x = (
        absolute_permeability.x
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_x = (
        absolute_permeability.x
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    water_mobility_grid_y = (
        absolute_permeability.y
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_y = (
        absolute_permeability.y
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_y = (
        absolute_permeability.y
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    water_mobility_grid_z = (
        absolute_permeability.z
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_z = (
        absolute_permeability.z
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_z = (
        absolute_permeability.z
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    # Clamp mobility grids to avoid numerical issues
    # This ensures mobilities are never zero or negative (for numerical stability)
    water_mobility_grid_x = np.maximum(water_mobility_grid_x, 1e-12)
    oil_mobility_grid_x = np.maximum(oil_mobility_grid_x, 1e-12)
    gas_mobility_grid_x = np.maximum(gas_mobility_grid_x, 1e-12)
    water_mobility_grid_y = np.maximum(water_mobility_grid_y, 1e-12)
    oil_mobility_grid_y = np.maximum(oil_mobility_grid_y, 1e-12)
    gas_mobility_grid_y = np.maximum(gas_mobility_grid_y, 1e-12)
    water_mobility_grid_z = np.maximum(water_mobility_grid_z, 1e-12)
    oil_mobility_grid_z = np.maximum(oil_mobility_grid_z, 1e-12)
    gas_mobility_grid_z = np.maximum(gas_mobility_grid_z, 1e-12)

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
            residual_oil_saturation_grid=padded_residual_oil_saturation_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    # Initialize sparse coefficient matrix (A * P_o_new = b)
    total_cell_count = cell_count_x * cell_count_y * cell_count_z
    A = lil_matrix((total_cell_count, total_cell_count), dtype=np.float32)
    # Initialize RHS source vector
    b = np.zeros(total_cell_count)
    _to_1D_index = functools.partial(
        to_1D_index,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )
    # Iterate over interior cells to populate the linear system
    for i, j, k in itertools.product(
        range(cell_count_x), range(cell_count_y), range(cell_count_z)
    ):
        ip, jp, kp = (i + 1, j + 1, k + 1)  # Padded indices for current cell (i, j, k)
        cell_1D_index = _to_1D_index(i, j, k)
        cell_thickness = thickness_grid[i, j, k]
        cell_volume = cell_size_x * cell_size_y * cell_thickness
        cell_porosity = porosity_grid[i, j, k]
        cell_total_compressibility = total_compressibility_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]
        cell_temperature = fluid_properties.temperature_grid[i, j, k]
        time_step_size_in_days = time_step_size * DAYS_PER_SECOND

        # Accumulation term coefficient for the diagonal of A
        # β = φ·c_t·V / Δt
        accumulation_coefficient = (
            cell_porosity * cell_total_compressibility * cell_volume
        ) / time_step_size_in_days

        # Diagonal entry for the current cell (i, j, k)
        # A[i, j, k] = β + ∑(transimissibility_terms)
        A[cell_1D_index, cell_1D_index] = accumulation_coefficient

        flux_configurations = {
            "x": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_x,
                    "oil_mobility_grid": padded_oil_mobility_grid_x,
                    "gas_mobility_grid": padded_gas_mobility_grid_x,
                },
                "positive_neighbour": (ip + 1, jp, kp),
                "negative_neighbour": (ip - 1, jp, kp),
                "geometric_factor": (
                    cell_size_y * cell_thickness / (cell_size_x**2)
                ),  # A_x / Δx²
            },
            "y": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_y,
                    "oil_mobility_grid": padded_oil_mobility_grid_y,
                    "gas_mobility_grid": padded_gas_mobility_grid_y,
                },
                "positive_neighbour": (ip, jp + 1, kp),
                "negative_neighbour": (ip, jp - 1, kp),
                "geometric_factor": (
                    cell_size_x * cell_thickness / (cell_size_y**2)
                ),  # A_y / Δy²
            },
            "z": {
                "mobility_grids": {
                    "water_mobility_grid": padded_water_mobility_grid_z,
                    "oil_mobility_grid": padded_oil_mobility_grid_z,
                    "gas_mobility_grid": padded_gas_mobility_grid_z,
                },
                "positive_neighbour": (ip, jp, kp + 1),
                "negative_neighbour": (ip, jp, kp - 1),
                "geometric_factor": (
                    cell_size_x * cell_size_y / (cell_thickness**2)
                ),  # A_z / Δz²
            },
        }

        total_capillary_flow = 0.0
        total_gravity_flow = 0.0
        for direction, configuration in flux_configurations.items():
            mobility_grids = configuration["mobility_grids"]
            positive_neighbour_indices = configuration["positive_neighbour"]
            negative_neighbour_indices = configuration["negative_neighbour"]

            # Include the density and elevation grids only for z-direction
            # For gravity segregation to be included/modelled in the flux computation
            if direction == "z":
                flux_oil_density_grid = padded_oil_density_grid
                flux_water_density_grid = padded_water_density_grid
                flux_gas_density_grid = padded_gas_density_grid
                flux_elevation_grid = padded_elevation_grid
            else:
                flux_oil_density_grid = None
                flux_water_density_grid = None
                flux_gas_density_grid = None
                flux_elevation_grid = None

            (
                positive_harmonic_mobility,
                positive_capillary_flux,
                positive_gravity_flux,
            ) = _compute_pseudo_fluxes_from_neighbour(
                cell_indices=(ip, jp, kp),
                neighbour_indices=positive_neighbour_indices,
                oil_pressure_grid=padded_oil_pressure_grid,
                **mobility_grids,
                oil_water_capillary_pressure_grid=padded_oil_water_capillary_pressure_grid,
                gas_oil_capillary_pressure_grid=padded_gas_oil_capillary_pressure_grid,
                oil_density_grid=flux_oil_density_grid,
                water_density_grid=flux_water_density_grid,
                gas_density_grid=flux_gas_density_grid,
                elevation_grid=flux_elevation_grid,
            )
            (
                negative_harmonic_mobility,
                negative_capillary_flux,
                negative_gravity_flux,
            ) = _compute_pseudo_fluxes_from_neighbour(
                cell_indices=(ip, jp, kp),
                neighbour_indices=negative_neighbour_indices,
                oil_pressure_grid=padded_oil_pressure_grid,
                **mobility_grids,
                oil_water_capillary_pressure_grid=padded_oil_water_capillary_pressure_grid,
                gas_oil_capillary_pressure_grid=padded_gas_oil_capillary_pressure_grid,
                oil_density_grid=flux_oil_density_grid,
                water_density_grid=flux_water_density_grid,
                gas_density_grid=flux_gas_density_grid,
                elevation_grid=flux_elevation_grid,
            )

            positive_transmissibility = (
                positive_harmonic_mobility * configuration["geometric_factor"]
            )
            negative_transmissibility = (
                negative_harmonic_mobility * configuration["geometric_factor"]
            )

            # Add the transmissibility term to the diagonal entry
            A[cell_1D_index, cell_1D_index] += (
                positive_transmissibility + negative_transmissibility
            )
            # Add the off-diagonal entries for the positive and negative neighbours
            positive_neighbour_1D_index = _to_1D_index(*positive_neighbour_indices)
            negative_neighbour_1D_index = _to_1D_index(*negative_neighbour_indices)
            if positive_neighbour_1D_index != -1:
                A[
                    cell_1D_index, positive_neighbour_1D_index
                ] = -positive_transmissibility
            if negative_neighbour_1D_index != -1:
                A[
                    cell_1D_index, negative_neighbour_1D_index
                ] = -negative_transmissibility

            # Calculate the net capillary flow (rate)
            # The negative sign is already accounted for in the negative capillary flux term
            # contributions based on the implementation of function used to calculate
            # it. Hence, we can directly sum the positive and negative contributions.
            net_capillary_flux = positive_capillary_flux + negative_capillary_flux
            net_capillary_flow = (
                net_capillary_flux
                * configuration["geometric_factor"]
                * capillary_pressure_stability_factor
            )
            # Add the net capillary flow to the total capillary flow
            total_capillary_flow += net_capillary_flow

            # Calculate the net gravity flow (rate)
            net_gravity_flux = positive_gravity_flux + negative_gravity_flux
            net_gravity_flow = net_gravity_flux * configuration["geometric_factor"]
            # Add the net gravity flow to the total gravity flow
            total_gravity_flow += net_gravity_flow

        # Compute Source/Sink Term (WellParameters) - q * V (ft³/day)
        injection_well, production_well = wells[i, j, k]
        cell_injection_rate = 0.0
        cell_production_rate = 0.0
        permeability = (
            absolute_permeability.x[i, j, k],
            absolute_permeability.y[i, j, k],
            absolute_permeability.z[i, j, k],
        )
        if (
            injection_well is not None
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            # If there is an injection well, add its flow rate to the cell
            injected_phase = injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                phase_mobility = gas_mobility_grid_z[i, j, k]
            elif injected_phase == FluidPhase.WATER:
                phase_mobility = water_mobility_grid_z[i, j, k]
            else:
                phase_mobility = oil_mobility_grid_z[i, j, k]

            well_index = injection_well.get_well_index(
                interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                permeability=permeability,
            )
            cell_injection_rate = injection_well.get_flow_rate(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_mobility=phase_mobility,
                fluid=injected_fluid,
            )  # STB/day or SCF/day
            if injected_phase == FluidPhase.GAS:
                # Get the volumetric flow rate in ft³/day
                gas_fvf = (
                    injected_fluid.formation_volume_factor
                    or fluid_properties.gas_formation_volume_factor_grid[i, j, k]
                )
                cell_injection_rate *= gas_fvf  # ft³/SCF
            elif injected_phase == FluidPhase.WATER:
                # For water, convert bbl/day to ft³/day
                water_fvf = (
                    injected_fluid.formation_volume_factor
                    or fluid_properties.water_formation_volume_factor_grid[i, j, k]
                )  # bbl/STB
                cell_injection_rate *= (
                    water_fvf * BBL_TO_FT3
                )  # Convert bbl/day to ft³/day
            else:
                # For oil and water, convert bbl/day to ft³/day
                cell_injection_rate *= (
                    fluid_properties.oil_formation_volume_factor_grid[
                        i, j, k
                    ]  # bbl/STB
                    * BBL_TO_FT3  # Convert bbl/day to ft³/day
                )

        if production_well is not None:
            # If there is a production well, subtract its flow rate from the cell
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = gas_mobility_grid_y[i, j, k]
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = water_mobility_grid_y[i, j, k]
                else:
                    phase_mobility = oil_mobility_grid_y[i, j, k]

                well_index = production_well.get_well_index(
                    interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                    permeability=permeability,
                )
                production_rate = production_well.get_flow_rate(
                    pressure=cell_oil_pressure,
                    temperature=cell_temperature,
                    well_index=well_index,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                )  # STB/day or SCF/day

                if produced_fluid.phase == FluidPhase.GAS:
                    # Get the volumetric flow rate in ft³/day
                    gas_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.gas_formation_volume_factor_grid[i, j, k]
                    )  # ft³/SCF
                    production_rate *= gas_fvf
                elif produced_fluid.phase == FluidPhase.WATER:
                    # For water, convert bbl/day to ft³/day
                    water_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.water_formation_volume_factor_grid[i, j, k]
                    )  # bbl/STB
                    production_rate *= (
                        water_fvf * BBL_TO_FT3
                    )  # Convert bbl/day to ft³/day
                else:
                    # For oil, convert bbl/day to ft³/day
                    oil_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.oil_formation_volume_factor_grid[i, j, k]
                    )  # bbl/STB
                    production_rate *= (
                        oil_fvf * BBL_TO_FT3
                    )  # Convert bbl/day to ft³/day

                cell_production_rate += production_rate

        # Calculate the net well flow rate into the cell. Just add injection and production rates (since production rates are negative)
        # q_{i,j,k} * V = (q_{i,j,k}_injection - q_{i,j,k}_production)
        net_well_flow_rate_into_cell = cell_injection_rate + cell_production_rate

        # Compute the right-hand side source term vector b
        # b[i, j, k] = (β * P_{ijk}) + (q * V) + (capillary driven flow) + (gravity driven flow/segregation)
        b[cell_1D_index] = (
            (accumulation_coefficient * cell_oil_pressure)
            + net_well_flow_rate_into_cell
            + total_capillary_flow
            + total_gravity_flow
        )

    # Solve the linear system A * P_oilⁿ⁺¹ = b
    new_1D_pressure_grid = spsolve(A.tocsr(), b)
    # Reshape the 1D solution back to a 3 Dimensional grid
    new_pressure_grid = new_1D_pressure_grid.reshape(
        (cell_count_x, cell_count_y, cell_count_z)
    )
    new_pressure_grid = typing.cast(ThreeDimensionalGrid, new_pressure_grid)
    return EvolutionResult(new_pressure_grid, scheme="implicit")


def compute_adaptive_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_size: float,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    diffusion_number_threshold: float = 0.24,  # Slightly below 0.25 for safety
    capillary_pressure_stability_factor: float = 1.0,
) -> EvolutionResult[ThreeDimensionalGrid]:
    """
    Computes the pressure distribution in the reservoir grid for a single time step,
    adaptively choosing between explicit and implicit methods based on the maximum
    diffusion number in the grid for a three-phase flow system.

    This function now uses RockProperties and FluidProperties to derive the necessary
    physical parameters for the three-phase flow.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet (ft)
    :param thickness_grid: N-Dimensional numpy array representing the thickness of each cell in the grid (ft).
    :param elevation_grid: N-Dimensional numpy array representing the elevation of each cell in the grid (ft).
    :param time_step_size: Time step size (s) used in pressure update and stability criterion.
    :param boundary_conditions: Boundary conditions for pressure, saturation, etc, grids.
    :param rock_properties: RockProperties object containing rock physical properties.
    :param fluid_properties: FluidProperties object containing fluid physical properties.
    :param wells: Wells object containing information about injection and production wells.
    :param diffusion_number_threshold: The maximum allowed diffusion number for explicit stability.
        If any cell exceeds this, the implicit solver is used.

    :param capillary_pressure_stability_factor: Factor to scale capillary pressure terms for numerical stability.
        Capillary gradients can become numerically dominant in fine meshes or sharp saturation fronts.
        Damping avoids overshoot/undershoot by reducing their contribution without removing them.
        Used only if the implicit scheme is chosen.

    :return: A N-Dimensional numpy array representing the updated pressure distribution (psi)
        after solving the chosen system for the current time step.
    """
    # Extract properties from provided objects for clarity and convenience
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_grid = rock_properties.residual_oil_saturation_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_params = rock_properties.relative_permeability_params

    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_viscosity_grid = fluid_properties.water_viscosity_grid
    oil_viscosity_grid = fluid_properties.oil_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid
    )
    oil_formation_volume_factor_grid = fluid_properties.oil_formation_volume_factor_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid

    # Determine grid dimensions and cell sizes
    cell_size_x, cell_size_y = cell_dimension
    min_cell_thickness = typing.cast(float, np.min(thickness_grid))

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
        residual_oil_saturation_grid=residual_oil_saturation_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        relative_permeability_params=relative_permeability_params,
    )

    # Compute mobility grids for x, y, z directions
    # λ_x = k_abs * (kr / mu) / B_x. Where B_x is the formation volume factor of the phase
    water_mobility_grid_x = (
        absolute_permeability.x
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_x = (
        absolute_permeability.x
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_x = (
        absolute_permeability.x
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    water_mobility_grid_y = (
        absolute_permeability.y
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_y = (
        absolute_permeability.y
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_y = (
        absolute_permeability.y
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    water_mobility_grid_z = (
        absolute_permeability.z
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_z = (
        absolute_permeability.z
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_z = (
        absolute_permeability.z
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    # Calculate total mobility for diffusion number calculation
    total_mobility_grid_x = (
        water_mobility_grid_x + oil_mobility_grid_x + gas_mobility_grid_x
    )
    total_mobility_grid_y = (
        water_mobility_grid_y + oil_mobility_grid_y + gas_mobility_grid_y
    )
    total_mobility_grid_z = (
        water_mobility_grid_z + oil_mobility_grid_z + gas_mobility_grid_z
    )
    # Ensure total mobility is never zero or negative (for numerical stability)
    total_mobility_grid_x = np.maximum(total_mobility_grid_x, 1e-12)
    total_mobility_grid_y = np.maximum(total_mobility_grid_y, 1e-12)
    total_mobility_grid_z = np.maximum(total_mobility_grid_z, 1e-12)

    min_cell_size = min(cell_size_x, cell_size_y, min_cell_thickness)

    diffusion_number_grid_x = np.vectorize(
        compute_diffusion_number,
        excluded=["time_step_size", "cell_size"],
    )(
        porosity=porosity_grid,
        total_mobility=total_mobility_grid_x,
        total_compressibility=total_compressibility_grid,
        time_step_size=time_step_size,
        cell_size=min_cell_size,
    )
    diffusion_number_grid_y = np.vectorize(
        compute_diffusion_number,
        excluded=["time_step_size", "cell_size"],
    )(
        porosity=porosity_grid,
        total_mobility=total_mobility_grid_y,
        total_compressibility=total_compressibility_grid,
        time_step_size=time_step_size,
        cell_size=min_cell_size,
    )
    diffusion_number_grid_z = np.vectorize(
        compute_diffusion_number,
        excluded=["time_step_size", "cell_size"],
    )(
        porosity=porosity_grid,
        total_mobility=total_mobility_grid_z,
        total_compressibility=total_compressibility_grid,
        time_step_size=time_step_size,
        cell_size=min_cell_size,
    )

    # Determine max diffusion number
    max_diffusion_number_x = np.nanmax(diffusion_number_grid_x)
    max_diffusion_number_y = np.nanmax(diffusion_number_grid_y)
    max_diffusion_number_z = np.nanmax(diffusion_number_grid_z)
    max_diffusion_number = (
        max_diffusion_number_x + max_diffusion_number_y + max_diffusion_number_z
    )

    # Choose solver based on criterion
    if max_diffusion_number > diffusion_number_threshold:
        return compute_implicit_pressure_evolution(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            wells=wells,
            capillary_pressure_stability_factor=capillary_pressure_stability_factor,
        )
    return compute_explicit_pressure_evolution(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step_size=time_step_size,
        boundary_conditions=boundary_conditions,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        wells=wells,
    )


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

    Sⁿ⁺¹_ijk = Sⁿ_ijk - Δt / (φ * V_cell) * [
        (F_x_east - F_x_west) + (F_y_north - F_y_south) + (F_z_top - F_z_bottom) + q_x_ijk * V_cell
    ]

Volumetric phase flux at face F_dir is computed as:
    F_dir = f_x(S_upwind) * v_dir * A_face (ft³/day)

Upwind saturation S_upwind is selected based on the sign of v_dir:
    - If v_dir > 0 → S_upwind = Sⁿ_current (flow from current cell)
    - If v_dir < 0 → S_upwind = Sⁿ_neighbour (flow from neighbour into current cell)

Velocity Components:
    v_x = -λ_total * ∂p/∂x
    v_y = -λ_total * ∂p/∂y
    v_z = -λ_total * ∂p/∂z

Where:
    λ_total = Σ (k_r / μ) for all phases
    f_x = phase fractional flow = (k_r / μ) / λ_total
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


@numba.njit(cache=True)
def _compute_phase_fluxes_from_neighbour(
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
    residual_oil_saturation_grid: ThreeDimensionalGrid,
    residual_gas_saturation_grid: ThreeDimensionalGrid,
    water_exponent: float = 2.0,
    oil_exponent: float = 2.0,
    gas_exponent: float = 2.0,
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
    gas_pressure_difference = (
        neighbour_oil_pressure + neighbour_gas_oil_capillary_pressure
    ) - (cell_oil_pressure + cell_gas_oil_capillary_pressure)

    if elevation_grid is not None:
        # Calculate the elevation difference between the neighbour and current cell
        elevation_delta = (
            elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
        )
    else:
        elevation_delta = 0.0

    # Determine the upwind densities based on pressure difference
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
    # v_x = -λ_x * ∆P / Δx
    # For water: v_w = -λ_w * [(P_oil - P_cow) + (upwind_ρ_water * g * Δz)] / ΔL
    water_gravity_potential = (
        upwind_water_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total water phase potential
    water_phase_potential = water_pressure_difference + water_gravity_potential
    water_velocity = -water_harmonic_mobility * water_phase_potential / flow_length

    # For oil: v_o = -λ_o * [(P_oil) + (upwind_ρ_oil * g * Δz)] / ΔL
    oil_gravity_potential = (
        upwind_oil_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total oil phase potential
    oil_phase_potential = oil_pressure_difference + oil_gravity_potential
    oil_velocity = -oil_harmonic_mobility * oil_phase_potential / flow_length

    # For gas: v_g = -λ_g * ∆P / ΔL
    # v_g = -λ_g * [(P_oil + P_go) - (P_cog + P_gas) + (upwind_ρ_gas * g * Δz)] / ΔL
    gas_gravity_potential = (
        upwind_gas_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    # Calculate the total gas phase potential
    gas_phase_potential = gas_pressure_difference + gas_gravity_potential
    gas_velocity = -gas_harmonic_mobility * gas_phase_potential / flow_length

    # Select upwind saturations in the direction of flow
    upwinded_water_saturation = (
        cell_water_saturation if water_velocity >= 0 else neighbour_water_saturation
    )
    upwinded_oil_saturation = (
        cell_oil_saturation if oil_velocity >= 0 else neighbour_oil_saturation
    )
    upwinded_gas_saturation = (
        cell_gas_saturation if gas_velocity >= 0 else neighbour_gas_saturation
    )

    # Select upwind viscosities in the direction of flow
    upwinded_water_viscosity = (
        cell_water_viscosity if water_velocity >= 0 else neighbour_water_viscosity
    )
    upwinded_oil_viscosity = (
        cell_oil_viscosity if oil_velocity >= 0 else neighbour_oil_viscosity
    )
    upwinded_gas_viscosity = (
        cell_gas_viscosity if gas_velocity >= 0 else neighbour_gas_viscosity
    )

    # Compute the total fractional flow
    (
        upwinded_water_relative_permeability,
        upwinded_oil_relative_permeability,
        upwinded_gas_relative_permeability,
    ) = compute_three_phase_relative_permeabilities(
        water_saturation=upwinded_water_saturation,
        oil_saturation=upwinded_oil_saturation,
        gas_saturation=upwinded_gas_saturation,
        irreducible_water_saturation=irreducible_water_saturation_grid[cell_indices],
        residual_oil_saturation=residual_oil_saturation_grid[cell_indices],
        residual_gas_saturation=residual_gas_saturation_grid[cell_indices],
        water_exponent=water_exponent,
        oil_exponent=oil_exponent,
        gas_exponent=gas_exponent,
    )
    water_upwinded_mobility = (
        upwinded_water_relative_permeability / upwinded_water_viscosity
    )
    oil_upwinded_mobility = upwinded_oil_relative_permeability / upwinded_oil_viscosity
    gas_upwinded_mobility = upwinded_gas_relative_permeability / upwinded_gas_viscosity

    # f_phase = λ_phase / (λ_w + λ_o + λ_g)
    total_upwinded_mobility = (
        water_upwinded_mobility + oil_upwinded_mobility + gas_upwinded_mobility
    )
    total_upwinded_mobility = np.maximum(
        total_upwinded_mobility, 1e-12
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
    # For water: F_w_from_neighbour = F_w * S_upwind
    water_volumetric_flux_from_neighbour = (
        water_volumetric_flux_at_face * upwinded_water_saturation
    )
    oil_volumetric_flux_from_neighbour = (
        oil_volumetric_flux_at_face * upwinded_oil_saturation
    )
    gas_volumetric_flux_from_neighbour = (
        gas_volumetric_flux_at_face * upwinded_gas_saturation
    )
    return (
        water_volumetric_flux_from_neighbour,
        oil_volumetric_flux_from_neighbour,
        gas_volumetric_flux_from_neighbour,
    )


def compute_saturation_evolution(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_size: float,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
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
    :param time_step_size: Time step duration in seconds for the simulation.
    :param boundary_conditions: Boundary conditions for pressure and saturation grids.
    :param rock_properties: `RockProperties` object containing rock physical properties.
    :param fluid_properties: `FluidProperties` object containing fluid physical properties,
        including current pressure and saturation grids.

    :param wells: ``Wells`` object containing information about injection and production wells.
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
    residual_oil_saturation_grid = rock_properties.residual_oil_saturation_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_params = rock_properties.relative_permeability_params
    capillary_pressure_params = (
        rock_properties.capillary_pressure_params
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

    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid
    )
    oil_formation_volume_factor_grid = fluid_properties.oil_formation_volume_factor_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid

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
    padded_oil_pressure_grid = edge_pad_grid(current_oil_pressure_grid.copy())
    padded_oil_saturation_grid = edge_pad_grid(current_oil_saturation_grid.copy())
    padded_water_saturation_grid = edge_pad_grid(current_water_saturation_grid.copy())
    padded_gas_saturation_grid = edge_pad_grid(current_gas_saturation_grid.copy())
    padded_irreducible_water_saturation_grid = edge_pad_grid(
        irreducible_water_saturation_grid.copy()
    )
    padded_residual_oil_saturation_grid = edge_pad_grid(
        residual_oil_saturation_grid.copy()
    )
    padded_residual_gas_saturation_grid = edge_pad_grid(
        residual_gas_saturation_grid.copy()
    )
    padded_water_viscosity_grid = edge_pad_grid(water_viscosity_grid.copy())
    padded_oil_viscosity_grid = edge_pad_grid(oil_viscosity_grid.copy())
    padded_gas_viscosity_grid = edge_pad_grid(gas_viscosity_grid.copy())
    padded_oil_density_grid = edge_pad_grid(oil_density_grid)
    padded_water_density_grid = edge_pad_grid(water_density_grid)
    padded_gas_density_grid = edge_pad_grid(gas_density_grid)
    padded_elevation_grid = edge_pad_grid(elevation_grid)

    # Apply boundary conditions to relevant padded grids
    boundary_conditions["pressure"].apply(padded_oil_pressure_grid)
    boundary_conditions["oil_saturation"].apply(padded_oil_saturation_grid)
    boundary_conditions["water_saturation"].apply(padded_water_saturation_grid)
    boundary_conditions["gas_saturation"].apply(padded_gas_saturation_grid)

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
        residual_oil_saturation_grid=residual_oil_saturation_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        relative_permeability_params=relative_permeability_params,
    )

    # Compute mobility grids for x, y, z directions
    # λ_x = k_abs * (kr / mu) / B_x. Where B_x is the formation volume factor of the phase
    water_mobility_grid_x = (
        absolute_permeability.x
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_x = (
        absolute_permeability.x
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_x = (
        absolute_permeability.x
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    water_mobility_grid_y = (
        absolute_permeability.y
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_y = (
        absolute_permeability.y
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_y = (
        absolute_permeability.y
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    water_mobility_grid_z = (
        absolute_permeability.z
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid_z = (
        absolute_permeability.z
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid_z = (
        absolute_permeability.z
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    # Clamp mobility grids to avoid numerical issues
    # This ensures mobilities are never zero or negative (for numerical stability)
    water_mobility_grid_x = np.maximum(water_mobility_grid_x, 1e-12)
    oil_mobility_grid_x = np.maximum(oil_mobility_grid_x, 1e-12)
    gas_mobility_grid_x = np.maximum(gas_mobility_grid_x, 1e-12)
    water_mobility_grid_y = np.maximum(water_mobility_grid_y, 1e-12)
    oil_mobility_grid_y = np.maximum(oil_mobility_grid_y, 1e-12)
    gas_mobility_grid_y = np.maximum(gas_mobility_grid_y, 1e-12)
    water_mobility_grid_z = np.maximum(water_mobility_grid_z, 1e-12)
    oil_mobility_grid_z = np.maximum(oil_mobility_grid_z, 1e-12)
    gas_mobility_grid_z = np.maximum(gas_mobility_grid_z, 1e-12)

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
            residual_oil_saturation_grid=padded_residual_oil_saturation_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    # Create new grids for updated saturations (time 'n+1')
    updated_water_saturation_grid = current_water_saturation_grid.copy()
    updated_oil_saturation_grid = current_oil_saturation_grid.copy()
    updated_gas_saturation_grid = current_gas_saturation_grid.copy()

    water_exponent = relative_permeability_params.water_exponent
    oil_exponent = relative_permeability_params.oil_exponent
    gas_exponent = relative_permeability_params.gas_exponent

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

        net_water_flux_from_neighbours = 0.0
        net_oil_flux_from_neighbours = 0.0
        net_gas_flux_from_neighbours = 0.0
        for direction, config in flux_configurations.items():
            positive_neighbour_indices = config["positive_neighbour"]
            negative_neighbour_indices = config["negative_neighbour"]
            flow_area = config["flow_area"]
            flow_length = config["flow_length"]

            # Include the density and elevation grids only for z-direction
            # For gravity segregation to be included/modelled in the flux computation
            if direction == "z":
                flux_oil_density_grid = padded_oil_density_grid
                flux_water_density_grid = padded_water_density_grid
                flux_gas_density_grid = padded_gas_density_grid
                flux_elevation_grid = padded_elevation_grid
            else:
                flux_oil_density_grid = None
                flux_water_density_grid = None
                flux_gas_density_grid = None
                flux_elevation_grid = None

            # Compute fluxes from the positive neighbour
            (
                positive_water_flux,
                positive_oil_flux,
                positive_gas_flux,
            ) = _compute_phase_fluxes_from_neighbour(
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
                residual_oil_saturation_grid=padded_residual_oil_saturation_grid,
                residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
                water_exponent=water_exponent,
                oil_exponent=oil_exponent,
                gas_exponent=gas_exponent,
                oil_density_grid=flux_oil_density_grid,
                water_density_grid=flux_water_density_grid,
                gas_density_grid=flux_gas_density_grid,
                elevation_grid=flux_elevation_grid,
            )

            # Compute fluxes from the negative neighbour
            (
                negative_water_flux,
                negative_oil_flux,
                negative_gas_flux,
            ) = _compute_phase_fluxes_from_neighbour(
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
                residual_oil_saturation_grid=padded_residual_oil_saturation_grid,
                residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
                water_exponent=water_exponent,
                oil_exponent=oil_exponent,
                gas_exponent=gas_exponent,
                oil_density_grid=flux_oil_density_grid,
                water_density_grid=flux_water_density_grid,
                gas_density_grid=flux_gas_density_grid,
                elevation_grid=flux_elevation_grid,
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

        # CFL stability check: |flux| / V * Δt / φ ≤ 1
        max_flux_rate_per_volume = max(
            net_water_flux_from_neighbours / cell_total_volume,
            net_oil_flux_from_neighbours / cell_total_volume,
            net_gas_flux_from_neighbours / cell_total_volume,
        )
        if max_flux_rate_per_volume * time_step_in_days / cell_porosity > 1.0:
            raise RuntimeError(
                f"CFL condition violated at cell ({i}, {j}, {k}): "
                f"max normalized flux {max_flux_rate_per_volume:.4f} exceeds stability limit."
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
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            # If there is an injection well, add its flow rate to the cell
            injected_phase = injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                phase_mobility = gas_mobility_grid_z[i, j, k]
            elif injected_phase == FluidPhase.WATER:
                phase_mobility = water_mobility_grid_z[i, j, k]
            else:
                phase_mobility = oil_mobility_grid_z[i, j, k]

            well_index = injection_well.get_well_index(
                interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                permeability=permeability,
            )
            cell_injection_rate = injection_well.get_flow_rate(
                pressure=oil_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_mobility=phase_mobility,
                fluid=injected_fluid,
            )  # STB/day or SCF/day
            if injected_phase == FluidPhase.GAS:
                # Get the volumetric flow rate in ft³/day
                gas_fvf = (
                    injected_fluid.formation_volume_factor
                    or fluid_properties.gas_formation_volume_factor_grid[i, j, k]
                )  # ft³/SCF
                cell_injection_rate *= gas_fvf
                cell_gas_injection_rate = cell_injection_rate
            elif injected_phase == FluidPhase.WATER:
                # For water, convert bbl/day to ft³/day
                water_fvf = (
                    injected_fluid.formation_volume_factor
                    or fluid_properties.water_formation_volume_factor_grid[i, j, k]
                )  # bbl/STB
                cell_injection_rate *= water_fvf
                cell_water_injection_rate = (
                    cell_injection_rate * BBL_TO_FT3
                )  # Convert bbl/day to ft³/day
            else:
                # For oil and water, convert bbl/day to ft³/day
                oil_fvf = (
                    injected_fluid.formation_volume_factor
                    or fluid_properties.oil_formation_volume_factor_grid[i, j, k]
                )  # bbl/STB
                cell_injection_rate *= oil_fvf
                cell_oil_injection_rate = (
                    cell_injection_rate * BBL_TO_FT3
                )  # Convert bbl/day to ft³/day

        if production_well is not None:
            # If there is a production well, subtract its flow rate from the cell
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = gas_mobility_grid_y[i, j, k]
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = water_mobility_grid_y[i, j, k]
                else:
                    phase_mobility = oil_mobility_grid_y[i, j, k]

                well_index = production_well.get_well_index(
                    interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                    permeability=permeability,
                )
                production_rate = production_well.get_flow_rate(
                    pressure=oil_pressure,
                    temperature=cell_temperature,
                    well_index=well_index,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                )  # STB/day or SCF/day

                if produced_fluid.phase == FluidPhase.GAS:
                    # Get the volumetric flow rate in ft³/day
                    gas_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.gas_formation_volume_factor_grid[i, j, k]
                    )  # ft³/SCF
                    production_rate *= gas_fvf
                    cell_gas_production_rate += production_rate
                elif produced_fluid.phase == FluidPhase.WATER:
                    # For water, convert bbl/day to ft³/day
                    water_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.water_formation_volume_factor_grid[i, j, k]
                    )  # bbl/STB
                    production_rate *= water_fvf
                    cell_water_production_rate += (
                        production_rate * BBL_TO_FT3
                    )  # Convert bbl/day to ft³/day
                else:
                    # For oil, convert bbl/day to ft³/day
                    oil_fvf = (
                        produced_fluid.formation_volume_factor
                        or fluid_properties.oil_formation_volume_factor_grid[i, j, k]
                    )  # bbl/STB
                    production_rate *= oil_fvf
                    cell_oil_production_rate += (
                        production_rate * BBL_TO_FT3
                    )  # Convert bbl/day to ft³/day

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
        if (
            cell_pore_volume > 1e-18
        ):  # Avoid division by zero for cells with no significant pore volume
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
        else:
            # If no pore volume, saturations remain unchanged (or set to 0 if preferred for initial state)
            water_saturation_change = 0.0
            oil_saturation_change = 0.0
            gas_saturation_change = 0.0

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
        # 1. Clip saturations to ensure they remain physically meaningful [0.0, 1.0]
        updated_water_saturation_grid[i, j, k] = np.clip(
            updated_water_saturation_grid[i, j, k], 0.0, 1.0
        )
        updated_oil_saturation_grid[i, j, k] = np.clip(
            updated_oil_saturation_grid[i, j, k], 0.0, 1.0
        )
        updated_gas_saturation_grid[i, j, k] = np.clip(
            updated_gas_saturation_grid[i, j, k], 0.0, 1.0
        )

        # 2. Normalize saturations to ensure their sum is 1.0
        total_saturation = (
            updated_water_saturation_grid[i, j, k]
            + updated_oil_saturation_grid[i, j, k]
            + updated_gas_saturation_grid[i, j, k]
        )

        # Avoid division by zero if total_saturation is extremely small (e.g., in a void)
        if total_saturation > 1e-12:
            updated_water_saturation_grid[i, j, k] /= total_saturation
            updated_oil_saturation_grid[i, j, k] /= total_saturation
            updated_gas_saturation_grid[i, j, k] /= total_saturation
        else:
            # If total saturation is practically zero, set all to zero to maintain consistency.
            # This might happen in cells with extremely low porosity or where all fluids have been depleted.
            updated_water_saturation_grid[i, j, k] = 0.0
            updated_oil_saturation_grid[i, j, k] = 0.0
            updated_gas_saturation_grid[i, j, k] = 0.0

    return EvolutionResult(
        (
            updated_water_saturation_grid,
            updated_oil_saturation_grid,
            updated_gas_saturation_grid,
        ),
        scheme="explicit",
    )
