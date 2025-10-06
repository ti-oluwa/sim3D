import functools
import itertools
import typing
import warnings

import attrs
import numba
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, gmres

from sim3D.boundaries import BoundaryConditions, BoundaryMetadata
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
from sim3D.static import FluidProperties, RockFluidProperties, RockProperties
from sim3D.properties import compute_diffusion_number, compute_harmonic_mobility
from sim3D.types import (
    FluidPhase,
    Options,
    RelativePermeabilityFunc,
    T,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from sim3D.wells import Wells


def _warn_producer_is_injecting(
    production_rate: float,
    well_name: str,
    cell: ThreeDimensions,
    time: float,
    rate_unit: str = "ft³/day",
) -> None:
    """
    Issues a warning if a production well is found to be injecting fluid
    instead of producing it. i.e., if the production rate is positive.
    """
    warnings.warn(
        f"Warning: Production well '{well_name}' at cell {cell} has a positive rate of {production_rate:.2f} {rate_unit}, "
        f"indicating it is no longer producing fluid at {time:.3f} seconds. Production rates should be negative. Please check well configuration.",
        UserWarning,
    )


def _warn_injector_is_producing(
    injection_rate: float,
    well_name: str,
    cell: ThreeDimensions,
    time: float,
    rate_unit: str = "ft³/day",
) -> None:
    """
    Issues a warning if an injection well is found to be producing fluid
    instead of injecting it. i.e., if the injection rate is negative.
    """
    warnings.warn(
        f"Warning: Injection well '{well_name}' at cell {cell} has a negative rate of {injection_rate:.2f} {rate_unit}, "
        f"indicating it is no longer injecting fluid at {time:.3f} seconds. Injection rates should be postive. Please check well configuration.",
        UserWarning,
    )


@attrs.define(slots=True, frozen=True)
class EvolutionResult(typing.Generic[T]):
    value: T
    scheme: typing.Literal["implicit", "explicit"]


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


def evolve_pressure_explicitly(
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
    :param boundary_conditions: Boundary conditions for pressure and saturation grids.
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
    water_relative_mobility_grid = np.maximum(
        water_relative_mobility_grid, options.minimum_allowable_relative_mobility
    )
    oil_relative_mobility_grid = np.maximum(
        oil_relative_mobility_grid, options.minimum_allowable_relative_mobility
    )
    gas_relative_mobility_grid = np.maximum(
        gas_relative_mobility_grid, options.minimum_allowable_relative_mobility
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
            positive_pseudo_flux = _compute_explicit_pressure_pseudo_flux_from_neighbour(
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
            negative_pseudo_flux = _compute_explicit_pressure_pseudo_flux_from_neighbour(
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
                cell_injection_rate *= BBL_TO_FT3  # Convert bbls/day to ft³/day

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
                    production_rate *= BBL_TO_FT3  # Convert bbls/day to ft³/day

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


def evolve_pressure_implicitly(
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
    :param time_step: Current time step number (for logging/debugging purposes).
    :param time_step_size: Time step size in seconds for the implicit update.
    :param boundary_conditions: Dictionary of callable boundary conditions applied to pressure and saturations.
    :param rock_properties: Rock data including porosity, permeability, compressibility, saturation endpoints,
        and relative permeability/capillary pressure parameters.
    :param fluid_properties: Current pressure and saturation fields along with viscosity, compressibility,
        and formation volume factors for each phase.
    :param rock_fluid_properties: Relative permeability and capillary pressure functions and parameters.
        This includes wettability information and other relevant PVT properties.
    :param wells: Well configuration and parameters for injectors and producers including phase, location,
        orientation, radius, skin factor, and BHP.

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
    padded_residual_oil_saturation_water_grid = edge_pad_grid(
        residual_oil_saturation_water_grid
    )
    padded_residual_oil_saturation_gas_grid = edge_pad_grid(
        residual_oil_saturation_gas_grid
    )
    padded_residual_gas_saturation_grid = edge_pad_grid(residual_gas_saturation_grid)
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
    # `build_three_phase_relative_mobilities_grids` should handle `kr / mu` for each phase.
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
    water_relative_mobility_grid = np.maximum(
        water_relative_mobility_grid, options.minimum_allowable_relative_mobility
    )
    oil_relative_mobility_grid = np.maximum(
        oil_relative_mobility_grid, options.minimum_allowable_relative_mobility
    )
    gas_relative_mobility_grid = np.maximum(
        gas_relative_mobility_grid, options.minimum_allowable_relative_mobility
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
            ) = _compute_implicit_pressure_pseudo_fluxes_from_neighbour(
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
            ) = _compute_implicit_pressure_pseudo_fluxes_from_neighbour(
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
                * options.capillary_pressure_stability_factor
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
                        cell=(i, j, k),
                        time=time_step * time_step_size,
                        rate_unit="ft³/day"
                        if injected_phase == FluidPhase.GAS
                        else "bbls/day",
                    )

            if injected_phase != FluidPhase.GAS:
                cell_injection_rate *= BBL_TO_FT3

        if production_well is not None and production_well.is_open:
            # If there is a production well, subtract its flow rate from the cell
            for produced_fluid in production_well.produced_fluids:
                produced_phase = produced_fluid.phase
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = oil_relative_mobility_grid[i, j, k]
                    fluid_compressibility = oil_compressibility_grid[i, j, k]
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
                            cell=(i, j, k),
                            time=time_step * time_step_size,
                            rate_unit="ft³/day"
                            if produced_phase == FluidPhase.GAS
                            else "bbls/day",
                        )

                if produced_fluid.phase != FluidPhase.GAS:
                    production_rate *= BBL_TO_FT3

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


def evolve_pressure_adaptively(
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
    :param boundary_conditions: Boundary conditions for pressure, saturation, etc, grids.
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

    water_viscosity_grid = fluid_properties.water_viscosity_grid
    oil_viscosity_grid = fluid_properties.oil_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Convert water and oil formation volume factors from bbl/STB to ft³/STB, since reservoir
    # Volumes are treated in ft³ and flow rates are in ft³/day.
    # From bbl/STB to ft³/STB: multiply by 5.615 (1 bbl = 5.615 ft³)
    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid * BBL_TO_FT3
    )
    oil_formation_volume_factor_grid = (
        fluid_properties.oil_formation_volume_factor_grid * BBL_TO_FT3
    )
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
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        relative_permeability_func=relative_permeability_func,
    )

    # Clamp relative mobility grids to avoid numerical issues
    # This ensures mobilities are never zero or negative (for numerical stability)
    water_relative_mobility_grid = np.maximum(
        water_relative_mobility_grid, options.minimum_allowable_relative_mobility
    )
    oil_relative_mobility_grid = np.maximum(
        oil_relative_mobility_grid, options.minimum_allowable_relative_mobility
    )
    gas_relative_mobility_grid = np.maximum(
        gas_relative_mobility_grid, options.minimum_allowable_relative_mobility
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
    if max_diffusion_number > options.diffusion_number_threshold:
        return evolve_pressure_implicitly(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
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
        boundary_conditions=boundary_conditions,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        rock_fluid_properties=rock_fluid_properties,
        wells=wells,
        options=options,
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
        neighbour_water_saturation if water_velocity >= 0 else cell_water_saturation
    )
    upwinded_oil_saturation = (
        neighbour_oil_saturation if oil_velocity >= 0 else cell_oil_saturation
    )
    upwinded_gas_saturation = (
        neighbour_gas_saturation if gas_velocity >= 0 else cell_gas_saturation
    )

    # Select upwind viscosities in the direction of flow
    upwinded_water_viscosity = (
        neighbour_water_viscosity if water_velocity >= 0 else cell_water_viscosity
    )
    upwinded_oil_viscosity = (
        neighbour_oil_viscosity if oil_velocity >= 0 else cell_oil_viscosity
    )
    upwinded_gas_viscosity = (
        neighbour_gas_viscosity if gas_velocity >= 0 else cell_gas_viscosity
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

    # Convert water and oil formation volume factors from bbl/STB to ft³/STB, since reservoir
    # Volumes are treated in ft³ and flow rates are in ft³/day.
    # From bbl/STB to ft³/STB: multiply by 5.615 (1 bbl = 5.615 ft³)
    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid * BBL_TO_FT3
    )
    oil_formation_volume_factor_grid = (
        fluid_properties.oil_formation_volume_factor_grid * BBL_TO_FT3
    )
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid
    gas_to_oil_ratio_grid = fluid_properties.gas_to_oil_ratio_grid
    gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid
    # Convert GOR and gas solubility to reservoir conditions if necessary
    # Rso = Rso_sc * B_o / B_g # From SCF/STB to ft³/ft³
    # Rsw = Rsw_sc * B_w / B_g # From SCF/STB to ft³/ft³
    gas_to_oil_ratio_grid = gas_to_oil_ratio_grid * (
        oil_formation_volume_factor_grid / gas_formation_volume_factor_grid
    )
    gas_solubility_in_water_grid = gas_solubility_in_water_grid * (
        water_formation_volume_factor_grid / gas_formation_volume_factor_grid
    )
    gas_to_oil_ratio_grid = typing.cast(ThreeDimensionalGrid, gas_to_oil_ratio_grid)
    gas_solubility_in_water_grid = typing.cast(
        ThreeDimensionalGrid, gas_solubility_in_water_grid
    )

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
    water_relative_mobility_grid = np.maximum(
        water_relative_mobility_grid, options.minimum_allowable_relative_mobility
    )
    oil_relative_mobility_grid = np.maximum(
        oil_relative_mobility_grid, options.minimum_allowable_relative_mobility
    )
    gas_relative_mobility_grid = np.maximum(
        gas_relative_mobility_grid, options.minimum_allowable_relative_mobility
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
                oil_density_grid=flux_oil_density_grid,
                water_density_grid=flux_water_density_grid,
                gas_density_grid=flux_gas_density_grid,
                elevation_grid=flux_elevation_grid,
            )
            # Accumulate net fluxes from both neighbours
            # Net fluxes are the sum of positive and negative fluxes, as negative fluxes
            # are already negative (i.e., flow from the neighbour to the current cell).
            # print(
            #     f"Cell ({i}, {j}, {k}) - Direction {direction} - "
            #     f"Postive water flux: {positive_water_flux:.4f} ft³/day, "
            #     f"Negative water flux: {negative_water_flux:.4f} ft³/day; "
            #     f"Positive oil flux: {positive_oil_flux:.4f} ft³/day, "
            #     f"Negative oil flux: {negative_oil_flux:.4f} ft³/day; "
            #     f"Positive gas flux: {positive_gas_flux:.4f} ft³/day, "
            #     f"Negative gas flux: {negative_gas_flux:.4f} ft³/day"
            # )
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

        # print(
        #     f"Cell ({i}, {j}, {k}) - Net Fluxes (ft³/day): "
        #     f"Water: {net_water_flux_from_neighbours:.4f}, "
        #     f"Oil: {net_oil_flux_from_neighbours:.4f}, "
        #     f"Gas: {net_gas_flux_from_neighbours:.4f}"
        # )

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
                print("Cell gas injection rate (ft³/day):", cell_injection_rate)
                cell_gas_injection_rate = cell_injection_rate
            elif injected_phase == FluidPhase.WATER:
                cell_water_injection_rate = cell_injection_rate * BBL_TO_FT3
                print("Cell water injection rate (ft³/day):", cell_water_injection_rate)
            else:
                cell_oil_injection_rate = cell_injection_rate * BBL_TO_FT3

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
                    print("Cell gas production rate (ft³/day):", production_rate)
                elif produced_fluid.phase == FluidPhase.WATER:
                    cell_water_production_rate += production_rate * BBL_TO_FT3
                    outgoing_water_flux += production_rate * BBL_TO_FT3
                    print("Cell water production rate (ft³/day):", production_rate)
                else:
                    cell_oil_production_rate += production_rate * BBL_TO_FT3
                    outgoing_oil_flux += production_rate * BBL_TO_FT3
                    print("Cell oil production rate (ft³/day):", production_rate)

        # CFL stability check: Outgoing volume flux should not exceed pore volume in the time step
        # Outflux * Δt <= φ * V_cell
        # print(
        #     f"Cell ({i}, {j}, {k}) - Outgoing Fluxes (ft³/day): "
        #     f"Water: {outgoing_water_flux:.4f}, "
        #     f"Oil: {outgoing_oil_flux:.4f}, "
        #     f"Gas: {outgoing_gas_flux:.4f}, "
        #     f"Time step in days: {time_step_in_days:.4f}, "
        #     f"Cell pore volume (ft³): {cell_pore_volume:.4f}"
        # )
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
        # print(
        #     f"Cell ({i}, {j}, {k}) - Saturation Changes: "
        #     f"Water: {water_saturation_change:.6f}, "
        #     f"Oil: {oil_saturation_change:.6f}, "
        #     f"Gas: {gas_saturation_change:.6f}"
        # )

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


"""
Fully implicit finite difference formulation for saturation transport in a 3D reservoir
(immiscible three-phase flow: oil, water, and gas with slightly compressible fluids):

The governing equation for saturation evolution is the conservation of mass with advection:

    φ * ∂S_α/∂t = -∇ · F_α + q_α

Where:
    φ * ∂S_α/∂t = Accumulation term (change in phase saturation per unit volume) (1/day)
    ∇ · F_α = Flux divergence term (volumetric flux per unit volume) (1/day)
    q_α = Source/sink term for phase α per unit pore volume (1/day)
    α ∈ {water, oil, gas}

The volumetric flux for each phase is given by:
    F_α = f_α * λ_α * ∇Φ_α

where:
    f_α = fractional flow of phase α = λ_α / λ_total
    λ_α = phase mobility = k_abs * k_r,α / μ_α (ft²/psi·day)
    λ_total = Σ_β λ_β (total mobility across all phases)
    ∇Φ_α = phase potential gradient = ∇P_α + ρ_α * g * ∇z (psi/ft)
    P_α = phase pressure (psi)
    k_abs = absolute permeability (mD)
    k_r,α = relative permeability of phase α (fraction)
    μ_α = viscosity of phase α (cP)
    ρ_α = density of phase α (lb/ft³)
    g = gravitational acceleration (ft/s²)
    z = elevation (ft)

Phase pressures are related through capillary pressures:
    P_water = P_oil - P_cow (oil-water capillary pressure)
    P_gas = P_oil + P_cgo (gas-oil capillary pressure)

Implicit Discretization:

Time: Backward Euler (fully implicit)
    ∂S_α/∂t ≈ (S^{n+1}_α,ijk - S^n_α,ijk) / Δt

Space: Central differences with upwind fractional flow
    ∇ · F_α ≈ (1/V_pore) * [
        (F_α,x_i+1/2 - F_α,x_i-1/2) * A_x / Δx +
        (F_α,y_j+1/2 - F_α,y_j-1/2) * A_y / Δy +
        (F_α,z_k+1/2 - F_α,z_k-1/2) * A_z / Δz
    ]

Discretized equation (implicit form):
    φ * (S^{n+1}_α,ijk - S^n_α,ijk) / Δt = -∇ · F^{n+1}_α + q^{n+1}_α

Rearranging:
    φ * S^{n+1}_α,ijk / Δt + ∇ · F^{n+1}_α = φ * S^n_α,ijk / Δt + q^{n+1}_α

This forms a nonlinear system because:
    - F^{n+1}_α depends on S^{n+1}_α through f_α and k_r,α
    - Capillary pressures P_cow, P_cgo depend on S^{n+1}_α

Newton-Raphson Linearization:

For iteration m → m+1, we linearize around current guess S^{(m)}:
    R_α(S^{(m+1)}) ≈ R_α(S^{(m)}) + J_α · δS

where:
    R_α = residual vector for phase α
    J_α = Jacobian matrix = ∂R_α/∂S
    δS = S^{(m+1)} - S^{(m)} = correction to saturations

Residual for each cell (i,j,k) and phase α:
    R_α,ijk = φ * (S^{(m)}_α,ijk - S^n_α,ijk) / Δt + (∇ · F^{(m)}_α)_ijk - q^{n+1}_α,ijk

Jacobian entries (simplified, assuming flux derivatives w.r.t. saturations):
    J[α,ijk][β,lmn] = ∂R_α,ijk / ∂S_β,lmn

Diagonal terms (accumulation + flux derivatives w.r.t. own cell):
    J[α,ijk][α,ijk] = φ / Δt + Σ_dir (∂F_α,dir / ∂S_α,ijk)

Off-diagonal terms (flux coupling to neighbor cells):
    J[α,ijk][α,neighbor] = (∂F_α,dir / ∂S_α,neighbor)

Cross-phase coupling (via fractional flow):
    J[α,ijk][β,ijk] = (∂F_α / ∂S_β) for α ≠ β

The linear system to solve at each Newton iteration:
    J · δS = -R

Update:
    S^{(m+1)} = S^{(m)} + δS

Flux Calculation (at face i+1/2):

Transmissibility coefficient:
    T_α,i+1/2 = (A_x / Δx) * λ̄_α,i+1/2

where λ̄_α,i+1/2 is the harmonic mean of phase mobilities:
    λ̄_α,i+1/2 = 2 * λ_α,i * λ_α,i+1 / (λ_α,i + λ_α,i+1)

Phase potential difference:
    ΔΦ_α,i+1/2 = Φ_α,i+1 - Φ_α,i
                = (P_α,i+1 - P_α,i) + (ρ_α * g / 144) * (z_i+1 - z_i)

Upwind fractional flow:
    f_α,i+1/2 = f_α(S_i)     if ΔΦ_α > 0 (flow from i → i+1)
              = f_α(S_i+1)   if ΔΦ_α ≤ 0 (flow from i+1 → i)

Phase flux:
    F_α,i+1/2 = T_α,i+1/2 * f_α,i+1/2 * ΔΦ_α,i+1/2 (ft³/day)

Variables:
    S^n_α,ijk = saturation of phase α at cell (i,j,k) at time step n (fraction)
    S^{n+1}_α,ijk = updated saturation at time step n+1 (fraction)
    φ = porosity (fraction)
    Δt = time step size (days)
    Δx, Δy, Δz = cell dimensions (ft)
    V_cell = Δx * Δy * Δz = cell bulk volume (ft³)
    V_pore = φ * V_cell = cell pore volume (ft³)
    A_x = Δy * Δz (face area for x-direction flow) (ft²)
    A_y = Δx * Δz (face area for y-direction flow) (ft²)
    A_z = Δx * Δy (face area for z-direction flow) (ft²)
    q_α,ijk = phase source/sink rate per unit pore volume (1/day)
    F_α = phase volumetric flux (ft³/day)

Constraint:
    S_water + S_oil + S_gas = 1

This constraint is enforced by:
    1. Normalizing saturations after each Newton iteration
    2. Clipping saturations to physical bounds [S_residual, 1-ε]

Advantages of Implicit Method:

1. Unconditional Stability: No CFL restriction on Δt
   - Can use much larger time steps than explicit method
   - Particularly beneficial for problems with large mobility contrasts

2. Better Handling of Stiff Problems:
   - Sharp saturation fronts
   - High mobility ratios (μ_water/μ_oil or μ_gas/μ_oil >> 1)
   - Strong capillary effects

3. Improved Accuracy for Large Δt:
   - Captures nonlinear coupling between phases
   - Properly accounts for saturation-dependent properties at new time level

Computational Cost:

- Each time step requires multiple Newton iterations (typically 3-10)
- Each Newton iteration requires solving a large sparse linear system (3N × 3N)
- More expensive per time step than explicit, but allows much larger Δt
- Overall can be faster for stiff problems due to larger allowable time steps

Convergence Criteria:

Newton iteration converges when:
    ||R|| < ε_residual  (residual norm small enough)
    AND
    ||δS|| < ε_solution (saturation correction small enough)

Typical values: ε_residual = ε_solution = 10^{-6}

Notes:
- Pressure field P_oil must be known/computed before solving saturation
- In IMPES (Implicit Pressure, Explicit Saturation), pressure is implicit, saturation is explicit
- In fully implicit, both pressure and saturation are solved simultaneously
- This implementation uses sequential implicit: pressure implicit, then saturation implicit
- Upwinding is critical for maintaining solution monotonicity
- Jacobian can be simplified by neglecting some derivative terms for efficiency
- Well terms q_α depend on current saturations through phase mobilities
"""


def _compute_implicit_saturation_phase_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    geometric_factor: float,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_mobility_grid: ThreeDimensionalGrid,
    oil_mobility_grid: ThreeDimensionalGrid,
    gas_mobility_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    oil_viscosity_grid: ThreeDimensionalGrid,
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
) -> typing.Tuple[float, float, float, float, float, float]:
    """
    Computes volumetric fluxes and transmissibility coefficients for each phase from
    the neighbour to the current cell for implicit saturation evolution.

    This function calculates the complete phase fluxes using:
    F_α = T_α * f_α * ∇Φ_α * (A/L)

    where:
    - T_α is the phase harmonic transmissibility (ft²/psi.day)
    - f_α is the upwinded fractional flow (dimensionless)
    - ∇Φ_α is the phase potential gradient (psi)
    - A/L is the geometric factor (ft)

    The fluxes are computed with proper upwinding based on phase potentials, and
    transmissibility coefficients are returned for Jacobian assembly.

    :param cell_indices: Indices of the current cell (i, j, k)
    :param neighbour_indices: Indices of the neighbouring cell
    :param geometric_factor: Geometric factor (A/L) for this direction (ft)
    :param oil_pressure_grid: 3D grid of oil pressures (psi)
    :param water_saturation_grid: 3D grid of water saturations (fraction)
    :param oil_saturation_grid: 3D grid of oil saturations (fraction)
    :param gas_saturation_grid: 3D grid of gas saturations (fraction)
    :param water_mobility_grid: 3D grid of water mobilities (ft²/psi.day)
    :param oil_mobility_grid: 3D grid of oil mobilities (ft²/psi.day)
    :param gas_mobility_grid: 3D grid of gas mobilities (ft²/psi.day)
    :param water_viscosity_grid: 3D grid of water viscosities (cP)
    :param oil_viscosity_grid: 3D grid of oil viscosities (cP)
    :param gas_viscosity_grid: 3D grid of gas viscosities (cP)
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi)
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi)
    :param irreducible_water_saturation_grid: 3D grid of irreducible water saturations
    :param residual_oil_saturation_water_grid: 3D grid of residual oil saturations (water flooding)
    :param residual_oil_saturation_gas_grid: 3D grid of residual oil saturations (gas flooding)
    :param residual_gas_saturation_grid: 3D grid of residual gas saturations
    :param relative_permeability_func: Function to compute relative permeabilities
    :param oil_density_grid: 3D grid of oil densities (lb/ft³), optional
    :param water_density_grid: 3D grid of water densities (lb/ft³), optional
    :param gas_density_grid: 3D grid of gas densities (lb/ft³), optional
    :param elevation_grid: 3D grid of elevations (ft), optional
    :return: Tuple containing:
        - Water volumetric flux from neighbour (ft³/day)
        - Oil volumetric flux from neighbour (ft³/day)
        - Gas volumetric flux from neighbour (ft³/day)
        - Water transmissibility coefficient for Jacobian (ft²/psi.day)
        - Oil transmissibility coefficient for Jacobian (ft²/psi.day)
        - Gas transmissibility coefficient for Jacobian (ft²/psi.day)
    """
    # Get cell and neighbour properties
    cell_oil_pressure = oil_pressure_grid[cell_indices]
    cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[cell_indices]
    cell_water_saturation = water_saturation_grid[cell_indices]
    cell_oil_saturation = oil_saturation_grid[cell_indices]
    cell_gas_saturation = gas_saturation_grid[cell_indices]
    cell_water_viscosity = water_viscosity_grid[cell_indices]
    cell_oil_viscosity = oil_viscosity_grid[cell_indices]
    cell_gas_viscosity = gas_viscosity_grid[cell_indices]

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
    neighbour_water_viscosity = water_viscosity_grid[neighbour_indices]
    neighbour_oil_viscosity = oil_viscosity_grid[neighbour_indices]
    neighbour_gas_viscosity = gas_viscosity_grid[neighbour_indices]

    # Calculate pressure differences
    oil_pressure_difference = neighbour_oil_pressure - cell_oil_pressure
    oil_water_capillary_pressure_difference = (
        neighbour_oil_water_capillary_pressure - cell_oil_water_capillary_pressure
    )
    # Water pressure difference: P_w = P_o - P_cow
    water_pressure_difference = (
        oil_pressure_difference - oil_water_capillary_pressure_difference
    )

    # Gas pressure difference: P_g = P_o + P_cgo
    gas_pressure_difference = (
        neighbour_oil_pressure + neighbour_gas_oil_capillary_pressure
    ) - (cell_oil_pressure + cell_gas_oil_capillary_pressure)

    # Calculate gravity potentials if elevation grid is provided
    if elevation_grid is not None:
        elevation_delta = (
            elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
        )
    else:
        elevation_delta = 0.0

    # Determine upwind densities based on pressure differences
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

    # Calculate gravity potentials (ρ * g * Δz / 144 to convert to psi)
    water_gravity_potential = (
        upwind_water_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    oil_gravity_potential = (
        upwind_oil_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    gas_gravity_potential = (
        upwind_gas_density * ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0

    # Total phase potential differences
    water_phase_potential = water_pressure_difference + water_gravity_potential
    oil_phase_potential = oil_pressure_difference + oil_gravity_potential
    gas_phase_potential = gas_pressure_difference + gas_gravity_potential

    # Calculate harmonic mobilities (transmissibility coefficients before geometric factor)
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

    # Determine upwind direction for saturations based on total potential
    # Use the potential of each phase to determine flow direction
    upwind_water_saturation = (
        neighbour_water_saturation
        if water_phase_potential > 0.0
        else cell_water_saturation
    )
    upwind_oil_saturation = (
        neighbour_oil_saturation if oil_phase_potential > 0.0 else cell_oil_saturation
    )
    upwind_gas_saturation = (
        neighbour_gas_saturation if gas_phase_potential > 0.0 else cell_gas_saturation
    )

    upwind_water_viscosity = (
        neighbour_water_viscosity
        if water_phase_potential > 0.0
        else cell_water_viscosity
    )
    upwind_oil_viscosity = (
        neighbour_oil_viscosity if oil_phase_potential > 0.0 else cell_oil_viscosity
    )
    upwind_gas_viscosity = (
        neighbour_gas_viscosity if gas_phase_potential > 0.0 else cell_gas_viscosity
    )

    # Compute upwind relative permeabilities
    upwind_relative_permeabilities = relative_permeability_func(
        water_saturation=upwind_water_saturation,
        oil_saturation=upwind_oil_saturation,
        gas_saturation=upwind_gas_saturation,
        irreducible_water_saturation=irreducible_water_saturation_grid[cell_indices],
        residual_oil_saturation_water=residual_oil_saturation_water_grid[cell_indices],
        residual_oil_saturation_gas=residual_oil_saturation_gas_grid[cell_indices],
        residual_gas_saturation=residual_gas_saturation_grid[cell_indices],
    )

    upwind_water_relative_permeability = upwind_relative_permeabilities["water"]
    upwind_oil_relative_permeability = upwind_relative_permeabilities["oil"]
    upwind_gas_relative_permeability = upwind_relative_permeabilities["gas"]

    # Calculate upwind mobilities (kr/μ)
    water_upwind_mobility = upwind_water_relative_permeability / upwind_water_viscosity
    oil_upwind_mobility = upwind_oil_relative_permeability / upwind_oil_viscosity
    gas_upwind_mobility = upwind_gas_relative_permeability / upwind_gas_viscosity

    # Total upwind mobility
    total_upwind_mobility = (
        water_upwind_mobility + oil_upwind_mobility + gas_upwind_mobility
    )
    total_upwind_mobility = max(total_upwind_mobility, 1e-12)  # Avoid division by zero

    # Fractional flows (upwinded)
    water_fractional_flow = water_upwind_mobility / total_upwind_mobility
    oil_fractional_flow = oil_upwind_mobility / total_upwind_mobility
    gas_fractional_flow = gas_upwind_mobility / total_upwind_mobility

    # Calculate volumetric fluxes (ft³/day)
    # F_α = T_α * f_α * ∇Φ_α * (A/L)
    water_volumetric_flux = (
        water_harmonic_mobility
        * water_fractional_flow
        * water_phase_potential
        * geometric_factor
    )
    oil_volumetric_flux = (
        oil_harmonic_mobility
        * oil_fractional_flow
        * oil_phase_potential
        * geometric_factor
    )
    gas_volumetric_flux = (
        gas_harmonic_mobility
        * gas_fractional_flow
        * gas_phase_potential
        * geometric_factor
    )

    # Transmissibility coefficients for Jacobian assembly
    # These are T_α * (A/L) where T_α is the harmonic transmissibility
    water_transmissibility_coefficient = water_harmonic_mobility * geometric_factor
    oil_transmissibility_coefficient = oil_harmonic_mobility * geometric_factor
    gas_transmissibility_coefficient = gas_harmonic_mobility * geometric_factor

    return (
        water_volumetric_flux,
        oil_volumetric_flux,
        gas_volumetric_flux,
        water_transmissibility_coefficient,
        oil_transmissibility_coefficient,
        gas_transmissibility_coefficient,
    )


def evolve_saturation_implicitly(
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
) -> EvolutionResult[
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]
]:
    """
    Computes the new/updated saturation distribution for water, oil, and gas
    across the reservoir grid using a fully implicit finite difference method
    with Newton-Raphson iteration.

    This solver constructs and solves a nonlinear system for saturation evolution:
    φ * ∂S_α/∂t = -∇·F_α + q_α

    where F_α is the volumetric flux of phase α and q_α is the source/sink term.

    The system is linearized using Newton-Raphson iteration:
    J * δS = -R

    where J is the Jacobian matrix, R is the residual vector, and δS is the
    correction to saturations at the current iteration.

    Key features:
    - Fully implicit time integration (unconditionally stable for large time steps)
    - Upwind differencing for phase mobilities based on phase potentials
    - Harmonic averaging of transmissibilities
    - Proper handling of capillary pressure gradients
    - Gravity segregation effects (in z-direction)
    - Well injection/production with phase mobility weighting
    - Saturation constraint: S_w + S_o + S_g = 1
    - Newton-Raphson iteration for nonlinear relative permeability coupling

    :param cell_dimension: Cell dimensions (Δx, Δy) in feet
    :param thickness_grid: Grid of cell heights (Δz) in feet
    :param elevation_grid: Grid of cell elevations (ft)
    :param time_step: Current time step number
    :param time_step_size: Time step size in seconds
    :param boundary_conditions: Boundary conditions for pressure and saturations
    :param rock_properties: Rock properties including porosity, permeability, saturations
    :param fluid_properties: Fluid properties including pressure, saturations, viscosities
    :param rock_fluid_properties: Relative permeability and capillary pressure functions
    :param wells: Well configuration and parameters
    :param options: Simulation options
    :return: Updated water, oil, and gas saturation grids
    :raises RuntimeError: If Newton iteration fails to converge
    """
    # Extract properties from provided objects
    time_step_in_days = time_step_size * DAYS_PER_SECOND
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
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

    water_viscosity_grid = fluid_properties.water_viscosity_grid
    oil_viscosity_grid = fluid_properties.oil_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Determine grid dimensions
    cell_count_x, cell_count_y, cell_count_z = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Initialize solution vectors for Newton iteration
    # Start with current saturations as initial guess
    new_water_saturation_grid = current_water_saturation_grid.copy()
    new_oil_saturation_grid = current_oil_saturation_grid.copy()
    new_gas_saturation_grid = current_gas_saturation_grid.copy()

    # Initialize sparse coefficient matrix and RHS vector
    total_cell_count = cell_count_x * cell_count_y * cell_count_z
    _to_1D_index = functools.partial(
        to_1D_index,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )
    max_newton_iterations = options.max_iterations_per_evolution

    # Newton-Raphson iteration loop
    for newton_iter in range(max_newton_iterations):
        # Pad grids for boundary conditions and neighbour access
        padded_oil_pressure_grid = edge_pad_grid(current_oil_pressure_grid)
        padded_water_saturation_grid = edge_pad_grid(new_water_saturation_grid)
        padded_oil_saturation_grid = edge_pad_grid(new_oil_saturation_grid)
        padded_gas_saturation_grid = edge_pad_grid(new_gas_saturation_grid)
        padded_irreducible_water_saturation_grid = edge_pad_grid(
            irreducible_water_saturation_grid
        )
        padded_residual_oil_saturation_water_grid = edge_pad_grid(
            residual_oil_saturation_water_grid
        )
        padded_residual_oil_saturation_gas_grid = edge_pad_grid(
            residual_oil_saturation_gas_grid
        )
        padded_residual_gas_saturation_grid = edge_pad_grid(
            residual_gas_saturation_grid
        )
        padded_water_viscosity_grid = edge_pad_grid(water_viscosity_grid)
        padded_oil_viscosity_grid = edge_pad_grid(oil_viscosity_grid)
        padded_gas_viscosity_grid = edge_pad_grid(gas_viscosity_grid)
        padded_oil_density_grid = edge_pad_grid(oil_density_grid)
        padded_water_density_grid = edge_pad_grid(water_density_grid)
        padded_gas_density_grid = edge_pad_grid(gas_density_grid)
        padded_elevation_grid = edge_pad_grid(elevation_grid)

        # Apply boundary conditions
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
        boundary_conditions["water_saturation"].apply(
            padded_water_saturation_grid,
            metadata=BoundaryMetadata(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                time=time_step_size * time_step,
                grid_shape=new_water_saturation_grid.shape,
                property_name="water_saturation",
            ),
        )
        boundary_conditions["oil_saturation"].apply(
            padded_oil_saturation_grid,
            metadata=BoundaryMetadata(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                time=time_step_size * time_step,
                grid_shape=new_oil_saturation_grid.shape,
                property_name="oil_saturation",
            ),
        )
        boundary_conditions["gas_saturation"].apply(
            padded_gas_saturation_grid,
            metadata=BoundaryMetadata(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                time=time_step_size * time_step,
                grid_shape=new_gas_saturation_grid.shape,
                property_name="gas_saturation",
            ),
        )

        # Compute phase mobilities
        (
            water_relative_mobility_grid,
            oil_relative_mobility_grid,
            gas_relative_mobility_grid,
        ) = build_three_phase_relative_mobilities_grids(
            water_saturation_grid=new_water_saturation_grid,
            oil_saturation_grid=new_oil_saturation_grid,
            gas_saturation_grid=new_gas_saturation_grid,
            water_viscosity_grid=water_viscosity_grid,
            oil_viscosity_grid=oil_viscosity_grid,
            gas_viscosity_grid=gas_viscosity_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            relative_permeability_func=relative_permeability_func,
        )

        # Clamp mobilities
        water_relative_mobility_grid = np.maximum(
            water_relative_mobility_grid, options.minimum_allowable_relative_mobility
        )
        oil_relative_mobility_grid = np.maximum(
            oil_relative_mobility_grid, options.minimum_allowable_relative_mobility
        )
        gas_relative_mobility_grid = np.maximum(
            gas_relative_mobility_grid, options.minimum_allowable_relative_mobility
        )

        # Compute mobility grids for x, y, z directions
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

        # Pad mobility grids
        padded_water_mobility_grid_x = edge_pad_grid(water_mobility_grid_x)
        padded_oil_mobility_grid_x = edge_pad_grid(oil_mobility_grid_x)
        padded_gas_mobility_grid_x = edge_pad_grid(gas_mobility_grid_x)
        padded_water_mobility_grid_y = edge_pad_grid(water_mobility_grid_y)
        padded_oil_mobility_grid_y = edge_pad_grid(oil_mobility_grid_y)
        padded_gas_mobility_grid_y = edge_pad_grid(gas_mobility_grid_y)
        padded_water_mobility_grid_z = edge_pad_grid(water_mobility_grid_z)
        padded_oil_mobility_grid_z = edge_pad_grid(oil_mobility_grid_z)
        padded_gas_mobility_grid_z = edge_pad_grid(gas_mobility_grid_z)

        # Compute capillary pressures
        (
            padded_oil_water_capillary_pressure_grid,
            padded_gas_oil_capillary_pressure_grid,
        ) = build_three_phase_capillary_pressure_grids(
            water_saturation_grid=padded_water_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )

        # Initialize Jacobian and residual for this Newton iteration
        # We solve for 3 unknowns per cell: S_w, S_o, S_g
        # But we enforce constraint S_w + S_o + S_g = 1 implicitly
        # So we actually solve for only 2 independent saturations (e.g., S_w and S_g)
        # and compute S_o = 1 - S_w - S_g
        # For simplicity, we'll solve the full 3-equation system with constraint
        J = lil_matrix((3 * total_cell_count, 3 * total_cell_count), dtype=np.float32)
        R = np.zeros(3 * total_cell_count)

        # Iterate over cells to build Jacobian and residual
        for i, j, k in itertools.product(
            range(cell_count_x), range(cell_count_y), range(cell_count_z)
        ):
            ip, jp, kp = i + 1, j + 1, k + 1  # Padded indices
            cell_1D_index = _to_1D_index(i, j, k)

            # Cell geometry
            cell_thickness = thickness_grid[i, j, k]
            cell_volume = cell_size_x * cell_size_y * cell_thickness
            cell_porosity = porosity_grid[i, j, k]
            cell_pore_volume = cell_volume * cell_porosity
            cell_temperature = fluid_properties.temperature_grid[i, j, k]
            cell_oil_pressure = current_oil_pressure_grid[i, j, k]

            # Current and old saturations
            cell_water_saturation_new = new_water_saturation_grid[i, j, k]
            cell_oil_saturation_new = new_oil_saturation_grid[i, j, k]
            cell_gas_saturation_new = new_gas_saturation_grid[i, j, k]
            cell_water_saturation_old = current_water_saturation_grid[i, j, k]
            cell_oil_saturation_old = current_oil_saturation_grid[i, j, k]
            cell_gas_saturation_old = current_gas_saturation_grid[i, j, k]

            # Accumulation terms: φ * (S^{n+1} - S^n) / Δt
            water_accumulation = (
                cell_porosity
                * (cell_water_saturation_new - cell_water_saturation_old)
                / time_step_in_days
            )
            oil_accumulation = (
                cell_porosity
                * (cell_oil_saturation_new - cell_oil_saturation_old)
                / time_step_in_days
            )
            gas_accumulation = (
                cell_porosity
                * (cell_gas_saturation_new - cell_gas_saturation_old)
                / time_step_in_days
            )

            # Define flux computation configurations
            flux_configurations = {
                "x": {
                    "mobility_grids": {
                        "water_mobility_grid": padded_water_mobility_grid_x,
                        "oil_mobility_grid": padded_oil_mobility_grid_x,
                        "gas_mobility_grid": padded_gas_mobility_grid_x,
                    },
                    "positive_neighbour": (ip + 1, jp, kp),
                    "negative_neighbour": (ip - 1, jp, kp),
                    "geometric_factor": cell_size_y * cell_thickness / cell_size_x,
                },
                "y": {
                    "mobility_grids": {
                        "water_mobility_grid": padded_water_mobility_grid_y,
                        "oil_mobility_grid": padded_oil_mobility_grid_y,
                        "gas_mobility_grid": padded_gas_mobility_grid_y,
                    },
                    "positive_neighbour": (ip, jp + 1, kp),
                    "negative_neighbour": (ip, jp - 1, kp),
                    "geometric_factor": cell_size_x * cell_thickness / cell_size_y,
                },
                "z": {
                    "mobility_grids": {
                        "water_mobility_grid": padded_water_mobility_grid_z,
                        "oil_mobility_grid": padded_oil_mobility_grid_z,
                        "gas_mobility_grid": padded_gas_mobility_grid_z,
                    },
                    "positive_neighbour": (ip, jp, kp + 1),
                    "negative_neighbour": (ip, jp, kp - 1),
                    "geometric_factor": cell_size_x * cell_size_y / cell_thickness,
                },
            }

            # Initialize flux divergence terms
            total_water_flux_divergence = 0.0
            total_oil_flux_divergence = 0.0
            total_gas_flux_divergence = 0.0

            # Process fluxes from all directions
            for direction, configuration in flux_configurations.items():
                mobility_grids = configuration["mobility_grids"]
                positive_neighbour_indices = configuration["positive_neighbour"]
                negative_neighbour_indices = configuration["negative_neighbour"]
                geometric_factor = configuration["geometric_factor"]

                # Include density and elevation grids only for z-direction
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

                # Compute fluxes from positive neighbour
                (
                    pos_water_flux,
                    pos_oil_flux,
                    pos_gas_flux,
                    pos_water_transmissibility,
                    pos_oil_transmissibility,
                    pos_gas_transmissibility,
                ) = _compute_implicit_saturation_phase_fluxes_from_neighbour(
                    cell_indices=(ip, jp, kp),
                    neighbour_indices=positive_neighbour_indices,
                    geometric_factor=geometric_factor,
                    oil_pressure_grid=padded_oil_pressure_grid,
                    water_saturation_grid=padded_water_saturation_grid,
                    oil_saturation_grid=padded_oil_saturation_grid,
                    gas_saturation_grid=padded_gas_saturation_grid,
                    **mobility_grids,
                    water_viscosity_grid=padded_water_viscosity_grid,
                    oil_viscosity_grid=padded_oil_viscosity_grid,
                    gas_viscosity_grid=padded_gas_viscosity_grid,
                    oil_water_capillary_pressure_grid=padded_oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=padded_gas_oil_capillary_pressure_grid,
                    irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
                    residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
                    residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
                    residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
                    relative_permeability_func=relative_permeability_func,
                    oil_density_grid=flux_oil_density_grid,
                    water_density_grid=flux_water_density_grid,
                    gas_density_grid=flux_gas_density_grid,
                    elevation_grid=flux_elevation_grid,
                )

                # Compute fluxes from negative neighbour
                (
                    neg_water_flux,
                    neg_oil_flux,
                    neg_gas_flux,
                    neg_water_transmissibility,
                    neg_oil_transmissibility,
                    neg_gas_transmissibility,
                ) = _compute_implicit_saturation_phase_fluxes_from_neighbour(
                    cell_indices=(ip, jp, kp),
                    neighbour_indices=negative_neighbour_indices,
                    geometric_factor=geometric_factor,
                    oil_pressure_grid=padded_oil_pressure_grid,
                    water_saturation_grid=padded_water_saturation_grid,
                    oil_saturation_grid=padded_oil_saturation_grid,
                    gas_saturation_grid=padded_gas_saturation_grid,
                    **mobility_grids,
                    water_viscosity_grid=padded_water_viscosity_grid,
                    oil_viscosity_grid=padded_oil_viscosity_grid,
                    gas_viscosity_grid=padded_gas_viscosity_grid,
                    oil_water_capillary_pressure_grid=padded_oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=padded_gas_oil_capillary_pressure_grid,
                    irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
                    residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
                    residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
                    residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
                    relative_permeability_func=relative_permeability_func,
                    oil_density_grid=flux_oil_density_grid,
                    water_density_grid=flux_water_density_grid,
                    gas_density_grid=flux_gas_density_grid,
                    elevation_grid=flux_elevation_grid,
                )

                # Net fluxes into cell (positive means inflow)
                net_water_flux = pos_water_flux + neg_water_flux
                net_oil_flux = pos_oil_flux + neg_oil_flux
                net_gas_flux = pos_gas_flux + neg_gas_flux

                # Accumulate flux divergences (convert from ft³/day to 1/day by dividing by pore volume)
                total_water_flux_divergence += net_water_flux / cell_pore_volume
                total_oil_flux_divergence += net_oil_flux / cell_pore_volume
                total_gas_flux_divergence += net_gas_flux / cell_pore_volume

                # Add Jacobian contributions for flux terms
                # The Jacobian entries represent ∂F/∂S for the linearization
                # Simplified approach: use transmissibility coefficients as approximation
                pos_neighbour_1D_index = _to_1D_index(*positive_neighbour_indices)
                neg_neighbour_1D_index = _to_1D_index(*negative_neighbour_indices)

                # Jacobian factor for converting fluxes to saturation changes
                flux_jacobian_factor = 1.0 / cell_pore_volume

                # Water equation Jacobian contributions
                if pos_neighbour_1D_index != -1:
                    J[cell_1D_index, cell_1D_index] += (
                        pos_water_transmissibility * flux_jacobian_factor
                    )
                if neg_neighbour_1D_index != -1:
                    J[cell_1D_index, cell_1D_index] += (
                        neg_water_transmissibility * flux_jacobian_factor
                    )

                # Oil equation Jacobian contributions
                if pos_neighbour_1D_index != -1:
                    J[
                        cell_1D_index + total_cell_count,
                        cell_1D_index + total_cell_count,
                    ] += pos_oil_transmissibility * flux_jacobian_factor
                if neg_neighbour_1D_index != -1:
                    J[
                        cell_1D_index + total_cell_count,
                        cell_1D_index + total_cell_count,
                    ] += neg_oil_transmissibility * flux_jacobian_factor

                # Gas equation Jacobian contributions
                if pos_neighbour_1D_index != -1:
                    J[
                        cell_1D_index + 2 * total_cell_count,
                        cell_1D_index + 2 * total_cell_count,
                    ] += pos_gas_transmissibility * flux_jacobian_factor
                if neg_neighbour_1D_index != -1:
                    J[
                        cell_1D_index + 2 * total_cell_count,
                        cell_1D_index + 2 * total_cell_count,
                    ] += neg_gas_transmissibility * flux_jacobian_factor

            # Handle well source/sink terms
            injection_well, production_well = wells[i, j, k]
            cell_water_source_rate = 0.0
            cell_oil_source_rate = 0.0
            cell_gas_source_rate = 0.0
            permeability = (
                absolute_permeability.x[i, j, k],
                absolute_permeability.y[i, j, k],
                absolute_permeability.z[i, j, k],
            )

            # Injection well handling
            if (
                injection_well is not None
                and injection_well.is_open
                and (injected_fluid := injection_well.injected_fluid) is not None
            ):
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
                            cell=(i, j, k),
                            time=time_step * time_step_size,
                            rate_unit="ft³/day"
                            if injected_phase == FluidPhase.GAS
                            else "bbls/day",
                        )

                if injected_phase == FluidPhase.GAS:
                    cell_gas_source_rate = cell_injection_rate
                elif injected_phase == FluidPhase.WATER:
                    cell_water_source_rate = cell_injection_rate * BBL_TO_FT3
                else:
                    cell_oil_source_rate = cell_injection_rate * BBL_TO_FT3

            # Production well handling
            if production_well is not None and production_well.is_open:
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
                                cell=(i, j, k),
                                time=time_step * time_step_size,
                                rate_unit="ft³/day"
                                if produced_phase == FluidPhase.GAS
                                else "bbls/day",
                            )

                    if produced_fluid.phase == FluidPhase.GAS:
                        cell_gas_source_rate += production_rate
                    elif produced_fluid.phase == FluidPhase.WATER:
                        cell_water_source_rate += production_rate * BBL_TO_FT3
                    else:
                        cell_oil_source_rate += production_rate * BBL_TO_FT3

            # Convert volumetric rates to saturation rates
            water_source_term = cell_water_source_rate / cell_pore_volume
            oil_source_term = cell_oil_source_rate / cell_pore_volume
            gas_source_term = cell_gas_source_rate / cell_pore_volume

            # Assemble residual vector
            # R = ∂S/∂t - ∇·F - q
            R[cell_1D_index] = (
                water_accumulation - total_water_flux_divergence - water_source_term
            )
            R[cell_1D_index + total_cell_count] = (
                oil_accumulation - total_oil_flux_divergence - oil_source_term
            )
            R[cell_1D_index + 2 * total_cell_count] = (
                gas_accumulation - total_gas_flux_divergence - gas_source_term
            )

            # Add accumulation Jacobian contributions (time derivative terms)
            J[cell_1D_index, cell_1D_index] += cell_porosity / time_step_in_days
            J[cell_1D_index + total_cell_count, cell_1D_index + total_cell_count] += (
                cell_porosity / time_step_in_days
            )
            J[
                cell_1D_index + 2 * total_cell_count,
                cell_1D_index + 2 * total_cell_count,
            ] += cell_porosity / time_step_in_days

        # Solve the linear system J * δS = -R
        J_csr = J.tocsr()
        try:
            saturation_delta = spsolve(J_csr, -R)
        except Exception:
            # Fallback to iterative solver if direct solver fails
            saturation_delta, solve_info = gmres(J_csr, -R, rtol=1e-6)
            if solve_info != 0:
                raise RuntimeError(
                    f"Linear solver failed to converge at Newton iteration {newton_iter + 1}"
                )

        # Update saturations
        water_saturation_delta = saturation_delta[:total_cell_count].reshape(
            cell_count_x, cell_count_y, cell_count_z
        )
        oil_saturation_delta = saturation_delta[
            total_cell_count : 2 * total_cell_count
        ].reshape(cell_count_x, cell_count_y, cell_count_z)
        gas_saturation_delta = saturation_delta[2 * total_cell_count :].reshape(
            cell_count_x, cell_count_y, cell_count_z
        )

        new_water_saturation_grid += water_saturation_delta
        new_oil_saturation_grid += oil_saturation_delta
        new_gas_saturation_grid += gas_saturation_delta

        # Enforce saturation constraints
        for i, j, k in itertools.product(
            range(cell_count_x), range(cell_count_y), range(cell_count_z)
        ):
            min_water_sat = max(irreducible_water_saturation_grid[i, j, k], 1e-6)
            min_oil_sat = max(
                residual_oil_saturation_water_grid[i, j, k],
                residual_oil_saturation_gas_grid[i, j, k],
                1e-6,
            )
            min_gas_sat = max(residual_gas_saturation_grid[i, j, k], 1e-6)

            new_water_saturation_grid[i, j, k] = np.clip(
                new_water_saturation_grid[i, j, k], min_water_sat, 1.0 - 1e-6
            )
            new_oil_saturation_grid[i, j, k] = np.clip(
                new_oil_saturation_grid[i, j, k], min_oil_sat, 1.0 - 1e-6
            )
            new_gas_saturation_grid[i, j, k] = np.clip(
                new_gas_saturation_grid[i, j, k], min_gas_sat, 1.0 - 1e-6
            )

            # Normalize saturations
            total_sat = (
                new_water_saturation_grid[i, j, k]
                + new_oil_saturation_grid[i, j, k]
                + new_gas_saturation_grid[i, j, k]
            )
            if total_sat > 1e-12:
                new_water_saturation_grid[i, j, k] /= total_sat
                new_oil_saturation_grid[i, j, k] /= total_sat
                new_gas_saturation_grid[i, j, k] /= total_sat

        # Check for convergence
        residual_norm = np.linalg.norm(R)
        solution_change_norm = np.linalg.norm(saturation_delta)  # type: ignore
        newton_tolerance = options.convergence_tolerance

        if residual_norm < newton_tolerance and solution_change_norm < newton_tolerance:
            print(
                f"Implicit saturation solver converged in {newton_iter + 1} iterations"
            )
            print(f"  Residual norm: {residual_norm:.2e}")
            print(f"  Solution change norm: {solution_change_norm:.2e}")
            break

        if newton_iter == max_newton_iterations - 1:
            raise RuntimeError(
                f"Implicit saturation solver failed to converge after "
                f"{max_newton_iterations} iterations. "
                f"Final residual norm: {residual_norm:.2e}, "
                f"Final solution change norm: {solution_change_norm:.2e}"
            )

    return EvolutionResult(
        (
            new_water_saturation_grid,
            new_oil_saturation_grid,
            new_gas_saturation_grid,
        ),
        scheme="implicit",
    )
