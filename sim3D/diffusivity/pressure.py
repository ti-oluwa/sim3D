import functools
import itertools
import typing

import numpy as np
from scipy.sparse import lil_matrix
import pyamg
from scipy.sparse.linalg import gmres

from sim3D.constants import c
from sim3D.diffusivity.base import (
    EvolutionResult,
    _warn_injector_is_producing,
    _warn_producer_is_injecting,
)
from sim3D.grids.properties import (
    build_three_phase_capillary_pressure_grids,
    build_three_phase_relative_mobilities_grids,
    build_total_fluid_compressibility_grid,
)
from sim3D.properties import compute_diffusion_number, compute_harmonic_mobility
from sim3D.models import FluidProperties, RockFluidProperties, RockProperties
from sim3D.types import FluidPhase, Options, ThreeDimensionalGrid, ThreeDimensions
from sim3D.wells import Wells, compute_effective_permeability_for_well

__all__ = [
    "evolve_pressure_explicitly",
    "evolve_pressure_implicitly",
    "evolve_pressure_adaptively",
]

"""
Explicit finite difference formulation for pressure diffusion in a N-Dimensional reservoir
(slightly compressible fluid):

The governing equation is the N-Dimensional linear-flow diffusivity equation:

    ∂p/∂t * (ρ·φ·c_t) * V = ∇ · (λ·∇p) * V + q * V

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

    ∂p/∂t * (φ·c_t) * V = ∇ · (λ·∇p) * V + q * V

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
        A_x * (λ_{i+½,j,k}(pⁿ_{i+1,j,k} - pⁿ_{i,j,k}) - λ_{i-½,j,k}(pⁿ_{i,j,k} - pⁿ_{i-1,j,k})) / Δx +
        A_y * (λ_{i,j+½,k}(pⁿ_{i,j+1,k} - pⁿ_{i,j,k}) - λ_{i,j-½,k}(pⁿ_{i,j,k} - pⁿ_{i,j-1,k})) / Δy +
        A_z * (λ_{i,j,k+½}(pⁿ_{i,j,k+1} - pⁿ_{i,j,k}) - λ_{i,j,k-½}(pⁿ_{i,j,k} - pⁿ_{i,j,k-1})) / Δz +
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


def _compute_explicit_pressure_pseudo_flux_from_neighbour(
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
        upwind_water_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
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
        upwind_oil_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
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
        upwind_gas_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
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


def evolve_pressure_explicitly(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    wells: Wells[ThreeDimensions],
    options: Options,
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
    :param time_step: Current time step number (starting from 0).
    :param time_step_size: Time step size (s) for each iteration.
    :param rock_properties: `RockProperties` object containing rock physical properties including
        absolute permeability, porosity, residual saturations.

    :param fluid_properties: `FluidProperties` object containing fluid physical properties like
        pressure, temperature, saturations, viscosities, and compressibilities for
        water, oil, and gas.

    :param rock_fluid_properties: `RockFluidProperties` containing relative/capillary
        pressure parameters (which include wettability information).

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

    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    water_viscosity_grid = fluid_properties.water_viscosity_grid
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
    total_compressibility_grid = np.maximum(total_compressibility_grid, 1e-24)

    # Compute phase mobilities (kr / mu) for each cell
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

    # Compute Capillary Pressures Grids (local to each cell, based on current saturations)
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        build_three_phase_capillary_pressure_grids(
            water_saturation_grid=current_water_saturation_grid,
            gas_saturation_grid=current_gas_saturation_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    # Initialize a new grid to store the updated pressures for the current time step
    updated_oil_pressure_grid = current_oil_pressure_grid.copy()
    # Iterate over internal cells only (excluding boundary cells)
    # Assume boundary cells are added via padding for boundary conditions application purposes
    # Thus, we iterate from 1 to N-1 in each dimension
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_thickness = thickness_grid[i, j, k]  # Same as `cell_size_z`
        cell_volume = cell_size_x * cell_size_y * cell_thickness
        cell_porosity = porosity_grid[i, j, k]
        cell_total_compressibility = total_compressibility_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]
        cell_temperature = fluid_properties.temperature_grid[i, j, k]

        flux_configurations = {
            "x": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_x,
                    "oil_mobility_grid": oil_mobility_grid_x,
                    "gas_mobility_grid": gas_mobility_grid_x,
                },
                "positive_neighbour": (i + 1, j, k),
                "negative_neighbour": (i - 1, j, k),
                "geometric_factor": (cell_size_y * cell_thickness / cell_size_x),
            },
            "y": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_y,
                    "oil_mobility_grid": oil_mobility_grid_y,
                    "gas_mobility_grid": gas_mobility_grid_y,
                },
                "positive_neighbour": (i, j + 1, k),
                "negative_neighbour": (i, j - 1, k),
                "geometric_factor": (cell_size_x * cell_thickness / cell_size_y),
            },
            "z": {
                "mobility_grids": {
                    "water_mobility_grid": water_mobility_grid_z,
                    "oil_mobility_grid": oil_mobility_grid_z,
                    "gas_mobility_grid": gas_mobility_grid_z,
                },
                "positive_neighbour": (i, j, k + 1),
                "negative_neighbour": (i, j, k - 1),
                "geometric_factor": (cell_size_x * cell_size_y / cell_thickness),
            },
        }

        # Compute the net volumetric flow rate into the current cell (i, j, k)
        # A_x * (λ_{i+1/2,j,k}(pⁿ_{i+1,j,k} - pⁿ_{i,j,k}) - λ_{i-1/2,j,k}(pⁿ_{i,j,k} - pⁿ_{i-1,j,k})) / Δx +
        # A_y * (λ_{i,j+1/2,k}(pⁿ_{i,j+1,k} - pⁿ_{i,j,k}) - λ_{i,j-1/2,k}(pⁿ_{i,j,k} - pⁿ_{i,j-1,k})) / Δy +
        # A_z * (λ_{i,j,k+1/2}(pⁿ_{i,j,k+1} - pⁿ_{i,j,k}) - λ_{i,j,k-1/2}(pⁿ_{i,j,k} - pⁿ_{i,j,k-1})) / Δz
        net_volumetric_flow_rate_into_cell = 0.0
        for _, configuration in flux_configurations.items():
            mobility_grids = configuration["mobility_grids"]
            positive_neighbour_indices = configuration["positive_neighbour"]
            negative_neighbour_indices = configuration["negative_neighbour"]
            geometric_factor = configuration["geometric_factor"]

            # Compute the total flux from positive and negative neighbours
            positive_pseudo_flux = (
                _compute_explicit_pressure_pseudo_flux_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=positive_neighbour_indices,
                    oil_pressure_grid=current_oil_pressure_grid,
                    **mobility_grids,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                )
            )
            # This already incorporates the outer negative sign in the function
            # as it does (P_negative_neighbour - P_cell) instead of (P_cell - P_negative_neighbour)
            negative_pseudo_flux = (
                _compute_explicit_pressure_pseudo_flux_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=negative_neighbour_indices,
                    oil_pressure_grid=current_oil_pressure_grid,
                    **mobility_grids,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                )
            )
            # So we just add directly here instead of subtracting
            net_pseudo_flux_into_cell = positive_pseudo_flux + negative_pseudo_flux
            net_flux_into_cell = (
                net_pseudo_flux_into_cell * geometric_factor
            )  # (ft³/day)
            net_volumetric_flow_rate_into_cell += net_flux_into_cell

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
                pressure=cell_oil_pressure,
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
                        time=time_step * time_step_size,
                        cell=(i, j, k),
                        rate_unit="ft³/day"
                        if injected_phase == FluidPhase.GAS
                        else "bbls/day",
                    )

            if injected_phase != FluidPhase.GAS:
                cell_injection_rate *= c.BBL_TO_FT3  # Convert bbls/day to ft³/day

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
                    pressure=cell_oil_pressure,
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
                            time=time_step * time_step_size,
                            cell=(i, j, k),
                            rate_unit="ft³/day"
                            if produced_phase == FluidPhase.GAS
                            else "bbls/day",
                        )

                if produced_fluid.phase != FluidPhase.GAS:
                    production_rate *= c.BBL_TO_FT3  # Convert bbls/day to ft³/day

                cell_production_rate += production_rate

        # Calculate the net well flow rate into the cell. Just add injection and production rates (since production rates are negative)
        # q_{i,j,k} * V = (q_{i,j,k}_injection - q_{i,j,k}_production)
        net_well_flow_rate_into_cell = cell_injection_rate + cell_production_rate

        # Add the well flow rate to the net volumetric flow rate into the cell
        # Total flow rate into the cell (ft³/day)
        # Total flow rate = Net Volumetric Flow Rate + Net Well Flow Rate
        total_flow_rate_into_cell = (
            net_volumetric_flow_rate_into_cell + net_well_flow_rate_into_cell
        )

        # Full Explicit Pressure Update Equation (P_oil)
        # The accumulation term is (φ * C_t * cell_volume) * dP_oil/dt
        # dP_oil/dt = [Net_Volumetric_Flow_Rate_Into_Cell] / (φ * C_t * cell_volume)
        # dP_{i,j} = (Δt / (φ·c_t·V)) * [
        #     A_x * (λ_{i+1/2,j,k}(pⁿ_{i+1,j,k} - pⁿ_{i,j,k}) - λ_{i-1/2,j,k}(pⁿ_{i,j,k} - pⁿ_{i-1,j,k})) / Δx +
        #     A_y * (λ_{i,j+1/2,k}(pⁿ_{i,j+1,k} - pⁿ_{i,j,k}) - λ_{i,j-1/2,k}(pⁿ_{i,j,k} - pⁿ_{i,j-1,k})) / Δy +
        #     A_z * (λ_{i,j,k+1/2}(pⁿ_{i,j,k+1} - pⁿ_{i,j,k}) - λ_{i,j,k-1/2}(pⁿ_{i,j,k} - pⁿ_{i,j,k-1})) / Δz +
        #     q_{i,j,k} * V
        # ]
        time_step_size_in_days = time_step_size * c.DAYS_PER_SECOND
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
        A/Δx · [λ_{i+½,j,k}·(pⁿ⁺¹_{i+1,j,k} - pⁿ⁺¹_{i,j,k}) - λ_{i⁻½,j,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i⁻1,j,k})] +
        A/Δy · [λ_{i,j+½,k}·(pⁿ⁺¹_{i,j+1,k} - pⁿ⁺¹_{i,j,k}) - λ_{i,j⁻½,k}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j⁻1,k})] +
        A/Δz · [λ_{i,j,k+½}·(pⁿ⁺¹_{i,j,k+1} - pⁿ⁺¹_{i,j,k}) - λ_{i,j,k⁻½}·(pⁿ⁺¹_{i,j,k} - pⁿ⁺¹_{i,j,k⁻1})] +
        qⁿ⁺¹_ijk * V

Matrix form:

Let:
    Tx⁺ = λ_{i+½,j,k}·A / Δx
    Tx⁻ = λ_{i⁻½,j,k}·A / Δx
    Ty⁺ = λ_{i,j+½,k}·A / Δy
    Ty⁻ = λ_{i,j⁻½,k}·A / Δy
    Tz⁺ = λ_{i,j,k+½}·A / Δz
    Tz⁻ = λ_{i,j,k⁻½}·A / Δz
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
            [(λ_w * ∇P_cow) + (λ_g * ∇P_cgo)] * A / (Δx, Δy, Δz)

Gravity driven segregation (only in effect in the z-direction):

    total_gravity_flow = (
            [λ_w * (upwind_ρ_w * g * ∆z) / 144] 
            + [λ_g * (upwind_ρ_g * g * ∆z) / 144] 
            + [λ_o * (upwind_ρ_o * g * ∆z) / 144]
    ) * A / Δz

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


def to_1D_index_interior_only(
    i: int,
    j: int,
    k: int,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> int:
    """
    Converts 3D grid indices to 1D index for interior cells only.
    Padding cells (i=0, i=Nx-1, etc.) return -1.
    Interior cells are mapped to [0, (Nx-2)*(Ny-2)*(Nz-2))
    """
    if not (
        0 < i < cell_count_x - 1
        and 0 < j < cell_count_y - 1
        and 0 < k < cell_count_z - 1
    ):
        return -1  # Padding cell

    # Adjust indices to 0-based for interior grid
    i_interior = i - 1
    j_interior = j - 1
    k_interior = k - 1

    # Interior dimensions
    ny_interior = cell_count_y - 2
    nz_interior = cell_count_z - 2
    return (
        i_interior * (ny_interior * nz_interior) + j_interior * nz_interior + k_interior
    )


def _compute_implicit_pressure_pseudo_fluxes_from_neighbour(
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
        oil_pressure_difference + gas_oil_capillary_pressure_difference
    )

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
            if water_pressure_difference > 0
            else water_density_grid[cell_indices]
        )
    else:
        upwind_water_density = 0.0

    if oil_density_grid is not None:
        upwind_oil_density = (
            oil_density_grid[neighbour_indices]
            if oil_pressure_difference > 0
            else oil_density_grid[cell_indices]
        )
    else:
        upwind_oil_density = 0.0

    if gas_density_grid is not None:
        upwind_gas_density = (
            gas_density_grid[neighbour_indices]
            if gas_pressure_difference > 0
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
    if total_harmonic_mobility <= 0.0:
        # No flow can occur if there is no mobility
        return 0.0, 0.0, 0.0

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
        upwind_water_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    oil_gravity_potential = (
        upwind_oil_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    gas_gravity_potential = (
        upwind_gas_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
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


def evolve_pressure_implicitly(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    wells: Wells[ThreeDimensions],
    options: Options,
) -> EvolutionResult[ThreeDimensionalGrid]:
    """
    Solves the fully implicit finite-difference pressure equation for a slightly compressible,
    three-phase flow system in a 3D reservoir.

    Constructs and solves the linear system A·Pⁿ⁺¹ = b using backward Euler time integration.
    The coefficient matrix A includes accumulation and transmissibility terms. The right-hand side
    vector b includes contributions from accumulation, well source/sink terms, and capillary pressure gradients.
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
    residual_oil_saturation_water_grid = (
        rock_properties.residual_oil_saturation_water_grid
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_func = rock_fluid_properties.relative_permeability_func
    capillary_pressure_params = rock_fluid_properties.capillary_pressure_params

    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    water_viscosity_grid = fluid_properties.water_viscosity_grid
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
    total_compressibility_grid = (
        total_fluid_compressibility_grid * porosity_grid
    ) + rock_compressibility
    total_compressibility_grid = np.maximum(total_compressibility_grid, 1e-24)

    # Compute phase mobilities
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

    # Compute Capillary Pressures Grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        build_three_phase_capillary_pressure_grids(
            water_saturation_grid=current_water_saturation_grid,
            gas_saturation_grid=current_gas_saturation_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    # Initialize sparse coefficient matrix and RHS vector
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    A = lil_matrix((interior_cell_count, interior_cell_count), dtype=np.float64)
    b = np.zeros(interior_cell_count)

    _to_1D_index = functools.partial(
        to_1D_index_interior_only,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )
    time_step_size_in_days = time_step_size * c.DAYS_PER_SECOND

    # FIRST PASS: Initialize accumulation terms for all cells
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_1D_index = _to_1D_index(i, j, k)
        cell_thickness = thickness_grid[i, j, k]
        cell_volume = cell_size_x * cell_size_y * cell_thickness
        cell_porosity = porosity_grid[i, j, k]
        cell_total_compressibility = total_compressibility_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]

        # Accumulation term coefficient
        accumulation_coefficient = (
            cell_porosity * cell_total_compressibility * cell_volume
        ) / time_step_size_in_days

        # Initialize matrix diagonal and RHS with accumulation
        A[cell_1D_index, cell_1D_index] = accumulation_coefficient
        b[cell_1D_index] = accumulation_coefficient * cell_oil_pressure

    # SECOND PASS: Add face transmissibilities and fluxes
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_1D_index = _to_1D_index(i, j, k)
        cell_thickness = thickness_grid[i, j, k]

        # Process inter-cell fluxes - iterate only positive offsets
        neighbor_offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        for di, dj, dk in neighbor_offsets:
            neighbour_indices = (i + di, j + dj, k + dk)
            neighbour_1D = _to_1D_index(*neighbour_indices)
            if neighbour_1D == -1:
                continue

            # Choose geometric factor and mobility grids for this offset
            if (di, dj, dk) == (1, 0, 0):
                geometric_factor = cell_size_y * cell_thickness / cell_size_x
                mobility_grids = {
                    "water_mobility_grid": water_mobility_grid_x,
                    "oil_mobility_grid": oil_mobility_grid_x,
                    "gas_mobility_grid": gas_mobility_grid_x,
                }
            elif (di, dj, dk) == (0, 1, 0):
                geometric_factor = cell_size_x * cell_thickness / cell_size_y
                mobility_grids = {
                    "water_mobility_grid": water_mobility_grid_y,
                    "oil_mobility_grid": oil_mobility_grid_y,
                    "gas_mobility_grid": gas_mobility_grid_y,
                }
            else:  # (0, 0, 1)
                geometric_factor = cell_size_x * cell_size_y / cell_thickness
                mobility_grids = {
                    "water_mobility_grid": water_mobility_grid_z,
                    "oil_mobility_grid": oil_mobility_grid_z,
                    "gas_mobility_grid": gas_mobility_grid_z,
                }

            # Compute pseudo flux from neighbour to cell
            harmonic_mobility, cap_flux, grav_flux = (
                _compute_implicit_pressure_pseudo_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=neighbour_indices,
                    oil_pressure_grid=current_oil_pressure_grid,
                    **mobility_grids,  # type: ignore
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                )
            )

            # Face transmissibility
            T_face = harmonic_mobility * geometric_factor

            # Add transmissibility to both cells' diagonals
            A[cell_1D_index, cell_1D_index] += T_face
            A[neighbour_1D, neighbour_1D] += T_face

            # Set off-diagonals (each face processed once, so use =)
            A[cell_1D_index, neighbour_1D] = -T_face
            A[neighbour_1D, cell_1D_index] = -T_face

            # Add capillary/gravity contributions to RHS
            cap_face = (
                cap_flux
                * geometric_factor
                * options.capillary_pressure_stability_factor
            )
            grav_face = grav_flux * geometric_factor

            b[cell_1D_index] += cap_face + grav_face
            b[neighbour_1D] += -(cap_face + grav_face)

    # THIRD PASS: Add well terms
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_1D_index = _to_1D_index(i, j, k)
        cell_thickness = thickness_grid[i, j, k]

        injection_well, production_well = wells[i, j, k]
        well_coefficient = 0.0
        well_source_term = 0.0

        permeability = (
            absolute_permeability.x[i, j, k],
            absolute_permeability.y[i, j, k],
            absolute_permeability.z[i, j, k],
        )

        # Process injection wells
        if (
            injection_well is not None
            and injection_well.is_open
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            injected_phase = injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                phase_relative_mobility = gas_relative_mobility_grid[i, j, k]
            else:
                phase_relative_mobility = water_relative_mobility_grid[i, j, k]

            avg_permeability = compute_effective_permeability_for_well(
                permeability=permeability,
                orientation=injection_well.orientation,
            )
            phase_absolute_mobility = (
                avg_permeability
                * phase_relative_mobility
                * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
            )
            well_index = injection_well.get_well_index(
                interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                permeability=permeability,
                skin_factor=injection_well.skin_factor,
            )
            wi_lambda = well_index * phase_absolute_mobility
            well_coefficient += wi_lambda
            well_source_term += wi_lambda * injection_well.bottom_hole_pressure

        # Process production wells
        if production_well is not None and production_well.is_open:
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase
                if produced_phase == FluidPhase.GAS:
                    phase_relative_mobility = gas_relative_mobility_grid[i, j, k]
                elif produced_phase == FluidPhase.WATER:
                    phase_relative_mobility = water_relative_mobility_grid[i, j, k]
                else:
                    phase_relative_mobility = oil_relative_mobility_grid[i, j, k]

                avg_permeability = compute_effective_permeability_for_well(
                    permeability=permeability,
                    orientation=production_well.orientation,
                )
                phase_absolute_mobility = (
                    avg_permeability
                    * phase_relative_mobility
                    * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
                )
                well_index = production_well.get_well_index(
                    interval_thickness=(cell_size_x, cell_size_y, cell_thickness),
                    permeability=permeability,
                    skin_factor=production_well.skin_factor,
                )
                wi_lambda = well_index * phase_absolute_mobility
                well_coefficient += wi_lambda
                well_source_term += wi_lambda * production_well.bottom_hole_pressure

        # Add well contributions
        A[cell_1D_index, cell_1D_index] += well_coefficient
        b[cell_1D_index] += well_source_term

    # Solve the linear system
    A_csr = A.tocsr()
    ml = pyamg.ruge_stuben_solver(A_csr)
    M = ml.aspreconditioner(cycle="V")
    new_1D_pressure_grid, info = gmres(
        A_csr, b, M=M, rtol=1e-6, atol=1e-12, restart=50, maxiter=500
    )
    if info != 0:
        raise RuntimeError(f"GMRES did not converge, info={info}")

    # Initialize with current pressure (preserves boundary values)
    new_pressure_grid = current_oil_pressure_grid.copy()

    # Fill interior cells with solution
    idx = 0
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        new_pressure_grid[i, j, k] = new_1D_pressure_grid[idx]
        idx += 1

    new_pressure_grid = typing.cast(ThreeDimensionalGrid, new_pressure_grid)
    return EvolutionResult(new_pressure_grid, scheme="implicit")


v_compute_diffusion_number = np.vectorize(
    compute_diffusion_number,
    excluded=["time_step_size", "cell_size"],
    otypes=[np.float64],
)


def evolve_pressure_adaptively(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_fluid_properties: RockFluidProperties,
    wells: Wells[ThreeDimensions],
    options: Options,
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
    :param time_step: Current time step number (for logging/debugging purposes).
    :param time_step_size: Time step size (s) used in pressure update and stability criterion.
    :param rock_properties: RockProperties object containing rock physical properties.
    :param fluid_properties: FluidProperties object containing fluid physical properties.
    :param rock_fluid_properties: RockFluidProperties object containing relative permeability
        and capillary pressure functions and parameters.
    :param wells: Wells object containing information about injection and production wells.
    :param diffusion_number_threshold: The maximum allowed diffusion number for explicit stability.
        If any cell exceeds this, the implicit solver is used.

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
    residual_oil_saturation_water_grid = (
        rock_properties.residual_oil_saturation_water_grid
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_func = rock_fluid_properties.relative_permeability_func

    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    water_viscosity_grid = fluid_properties.water_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

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
    total_compressibility_grid = np.maximum(total_compressibility_grid, 1e-24)

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
    # λ_x = k_abs * (kr / mu) * 0.001127 (mD/cP to ft²/psi.day)
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

    min_cell_size = min(cell_size_x, cell_size_y, min_cell_thickness)
    diffusion_number_grid_x = v_compute_diffusion_number(
        porosity=porosity_grid,
        total_mobility=total_mobility_grid_x,
        total_compressibility=total_compressibility_grid,
        time_step_size=time_step_size,
        cell_size=min_cell_size,
    )
    diffusion_number_grid_y = v_compute_diffusion_number(
        porosity=porosity_grid,
        total_mobility=total_mobility_grid_y,
        total_compressibility=total_compressibility_grid,
        time_step_size=time_step_size,
        cell_size=min_cell_size,
    )
    diffusion_number_grid_z = v_compute_diffusion_number(
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
    if max_diffusion_number > options.diffusion_number_threshold:
        return evolve_pressure_implicitly(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            rock_fluid_properties=rock_fluid_properties,
            wells=wells,
            options=options,
        )
    return evolve_pressure_explicitly(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        rock_fluid_properties=rock_fluid_properties,
        wells=wells,
        options=options,
    )
