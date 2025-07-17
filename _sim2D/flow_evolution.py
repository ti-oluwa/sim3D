import typing
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import itertools

from _sim2D.types import TwoDimensionalGrid
from _sim2D.constants import (
    BBL_TO_FT3,
    SECONDS_TO_DAYS,
    MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY,
)
from _sim2D.properties import (
    compute_harmonic_mobility,
    compute_diffusion_number,
    compute_three_phase_relative_permeabilities,
)
from _sim2D.grids import (
    build_2D_total_fluid_compressibility_grid,
    edge_pad_grid,
    build_2D_three_phase_relative_mobilities_grids,
    build_2D_three_phase_capillary_pressure_grids,
)
from _sim2D.models import RockProperties, FluidProperties
from _sim2D.wells import Wells, FluidPhase
from _sim2D.boundary_conditions import BoundaryConditions


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
# the evolution of fluid saturation and reservoir pressure over time across a 2D reservoir grid.
#
# Here, we iterate over interior grid cells only (excluding boundary cells at edges).
# This exclusion is necessary because the (upwind) finite difference method
# used here relies on neighboring cells (i±1, j or i, j±1) to compute
# pressure gradients and saturation fluxes. Accessing neighbors for boundary
# cells (i=0 or i=cell_count_x-1, etc.) would lead to out-of-bounds indexing errors.
#
# Additionally, in real-world reservoir simulations, boundary cells often have
# special physical conditions—such as fixed pressures, no-flow boundaries, or
# injection/production wells—which are typically handled using separate logic.
# Including them in this loop without appropriate treatment could result in
# non-physical results or numerical instability.
#
# Therefore, we safely exclude the outermost cells and perform updates only
# on interior cells where neighbor access is valid and consistent.
# ###########################################################################


"""
Explicit finite difference formulation for pressure diffusion in a 2D reservoir
(slightly compressible fluid):

The governing equation for pressure evolution is the linear-flow diffusivity equation:

    (∂p/∂t * (ρ·φ·c_t) * A∆x) + (K·ρ * A∆x) = (∇ρ ⋅ (vf) * A∆x) + (∇ρ · (λ·∇p) * A) + (q * A∆x)

    Note: compressibility (c_t) is added due to the fact that the fluid is compressible
    as pressure changes affect pore volume (φ), and fluid density (ρ) and thus the mass storage term.

where:
    (∂p/∂t * (ρ·φ·c_t) - Mass storage term
    K·ρ * A∆x - Reaction term
    ∇ρ ⋅ (vf) * A∆x - Convection term (advective transport)
    ∇ρ · (λ·∇p) * A - Diffusion term
    q * A∆x - Source/sink term

Given basic assumptions:

- Constant porosity (φ), total compressibility (c_t) and density (ρ) across the grid
- No reaction term
- No convection (advection) term - Advection term models fluid transport, not pressure evolution/propagation

The equation simplifies to:

    ∂p/∂t * (φ·c_t) * A∆x = (∇ · (λ·∇p) * A) + (q * A∆x)

where:
    ∂p/∂t * (φ·c_t) * A∆x = Accumulation term (ft³/s)
    ∇ · (λ·∇p) * A = Diffusion term (ft³/s)
    q * A∆x = Source/sink term (ft³/s) (Injection/production rates normalized by area)

To break it down further, we can express the diffusion term in terms of pressure gradients:

    ∇ · (λ·∇p) = ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)

For a 2D reservoir, ∇ · (λ·∇p) this expands to:

    ∇ · (λ·∇p) = ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)

Where:
    λ = mobility = k / μ
    ∇p = gradient of pressure (∂p/∂x, ∂p/∂y)
    (λ·∇p) = q = transmissibility (mobility) times pressure gradient (from Darcy's law) (ft³/s)
    (λ·∂p/∂x) = x-direction diffusion term
    (λ·∂p/∂y) = y-direction diffusion term

So ∇ · (λ·∇p) becomes the sum of the differential operators in the x and y directions:

    ∇ · (λ·∇p) = ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)

So the full equation becomes:

    ∂p/∂t * (φ·c_t) * A∆x = [ (∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)) * A ] + (q * A∆x) --- eqn (α)

Moving terms around gives us the explicit update formula:

    ∂p/∂t = [1 / ((φ·c_t) * A∆x)] * [ (∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)) * A ] + (q * A∆x)

Where:
    A     = cell cross-sectional area (ft²) = Δy * Δz (assuming unit depth for 2D, so Δz = 1 ft)
    ∆x    = cell size in x direction (ft)
    p     = pressure (psi)
    φ     = porosity (fraction)
    c_t   = total compressibility (rock + fluid) (psi⁻¹)
    λ     = mobility = k / μ (mD/cP) (convert to ft²·psi⁻¹·s⁻¹·lbm⁻¹ (0.001127) for consistency)
    k     = permeability (mD)
    μ     = viscosity (cP)
    ∂p/∂x = spatial derivative of pressure in x direction (x pressure gradient) (psi/ft)
    ∂p/∂y = spatial derivative of pressure in y direction (y pressure gradient) (psi/ft)
    q     = source/sink term (normalized injection/production) per unit volume (ft³/s * 1/ft³)
    q * A∆x = source/sink term (ft³/s) (normalized by volume)

Explicit Discretization (Forward Euler in time, central difference in space):

Discretizing in time (constant space) using Forward Euler , we get:

    ∂p/∂t = (pⁿ⁺¹_ij - pⁿ_ij) / Δt

Discretizing in space (constant time) using central differences, we get:

    ∂/∂x (λ·∂p/∂x) = (λ_{i+1/2,j} · (pⁿ_{i+1,j} - pⁿ_{i,j}) - λ_{i-1/2,j} · (pⁿ_{i,j} - pⁿ_{i-1,j})) / Δx²
    ∂/∂y (λ·∂p/∂y) = (λ_{i,j+1/2} · (pⁿ_{i,j+1} - pⁿ_{i,j}) - λ_{i,j-1/2} · (pⁿ_{i,j} - pⁿ_{i,j-1})) / Δy²

Combining these gives us the explicit update formula for pressure at cell (i,j) at time step n+1:

    pⁿ⁺¹_ij = pⁿ_ij + (Δt / [(φ·c_t) * A∆x]) * [
        [ (A_{i,j} / Δx²) * (λ_{i+1/2,j} · (pⁿ_{i+1,j} - pⁿ_{i,j}) - λ_{i-1/2,j} · (pⁿ_{i,j} - pⁿ_{i-1,j})) ]+
        [ (A_{i,j} / Δy²) * (λ_{i,j+1/2} · (pⁿ_{i,j+1} - pⁿ_{i,j}) - λ_{i,j-1/2} · (pⁿ_{i,j} - pⁿ_{i,j-1})) ]+
        (q_{i,j} * A_{i,j} * Δx)
    ]

Where:
    Δt = time step size (s)
    Δx = cell size in x direction (ft)
    Δy = cell size in y direction (ft)
    pⁿ_ij = pressure at cell (i,j) at time step n (psi)
    pⁿ⁺¹_ij = pressure at cell (i,j) at time step n+1 (psi)
    λ_{i+1/2,j} = harmonic average mobility between cells (i,j) and (i+1,j)
    λ_{i-1/2,j} = harmonic average mobility between cells (i,j) and (i-1,j)
    λ_{i,j+1/2} = harmonic average mobility between cells (i,j) and (i,j+1)
    λ_{i,j-1/2} = harmonic average mobility between cells (i,j) and (i,j-1)
    q_{i,j} = source/sink term at cell (i,j) per unit cell volume (ft³/s * 1/ft³)
    q_{i,j} * A_{i,j} * Δx = source/sink term at cell (i,j) (ft³/s)
    A_{i,j} = cross-sectional area of cell (i,j) (ft²) = Δy * Δz (assuming unit depth for 2D, so Δz = 1 ft)
    

Stability Condition:
    CFL-like (Courant-Friedrichs-Lewy) condition.
    Stable if dimensionless diffusion number D = λ·Δt / (φ·c_t·Δx²) < 0.25 in both x and y directions.

Notes:
    - Harmonic averaging is typically used to compute λ at interfaces (e.g., λ_{i+1/2,j}).
    - Δt must be small enough to ensure numerical stability.
    - This method is conditionally stable but simple to implement.
"""


def compute_explicit_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    height_grid: TwoDimensionalGrid,
    time_step_size: float,
    boundary_conditions: BoundaryConditions,
    rock_properties: RockProperties,
    fluid_properties: FluidProperties,
    wells: Wells,
) -> TwoDimensionalGrid:
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
    :param height_grid: 2D numpy array representing the height of each cell in the reservoir (ft).
    :param time_step_size: Time step size (s) for each iteration.
    :param boundary_conditions: Boundary conditions for pressure and saturation grids.
    :param rock_properties: `RockProperties` object containing rock physical properties including
        absolute permeability, porosity, residual saturations, and relative/capillary
        pressure parameters (which include wettability information).

    :param fluid_properties: `FluidProperties` object containing fluid physical properties like
        pressure, temperature, saturations, viscosities, and compressibilities for
        water, oil, and gas.

    :param wells: `Wells` object containing well parameters for injection and production wells
    :return: A 2D numpy array representing the updated oil phase pressure field (psi).
    """
    # Extract properties from provided objects for clarity and convenience
    absolute_permeability_grid = rock_properties.absolute_permeability_grid
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
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
    cell_count_x, cell_count_y = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Compute total fluid system compressibility for each cell
    total_fluid_compressibility_grid = build_2D_total_fluid_compressibility_grid(
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

    # Pad all necessary grids for boundary conditions and neighbor access
    padded_oil_pressure_grid = edge_pad_grid(current_oil_pressure_grid.copy())
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

    # Apply boundary conditions to relevant padded grids
    boundary_conditions["pressure"].apply(padded_oil_pressure_grid)
    boundary_conditions["water_saturation"].apply(padded_water_saturation_grid)
    boundary_conditions["gas_saturation"].apply(padded_gas_saturation_grid)

    # Compute phase mobilities (kr / mu) for each cell
    # `build_2D_three_phase_relative_mobilities_grids` should handle `k_abs * kr / mu` for each phase.
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_2D_three_phase_relative_mobilities_grids(
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

    # Compute mobility grids for harmonic averaging at interfaces
    # λ_x = k_abs * (kr / mu) / B_x. Where B_x is the formation volume factor of the phase
    water_mobility_grid = (
        absolute_permeability_grid
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid = (
        absolute_permeability_grid
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid = (
        absolute_permeability_grid
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    # Pad transmissibility grids for neighbor access
    padded_water_mobility_grid = edge_pad_grid(water_mobility_grid)
    padded_oil_mobility_grid = edge_pad_grid(oil_mobility_grid)
    padded_gas_mobility_grid = edge_pad_grid(gas_mobility_grid)

    # Compute Capillary Pressures Grids (local to each cell, based on current saturations)
    # P_cow = P_oil - P_water (can be negative for oil-wet systems)
    # P_cgo = P_gas - P_oil (generally positive)
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        build_2D_three_phase_capillary_pressure_grids(
            water_saturation_grid=padded_water_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
            residual_oil_saturation_grid=padded_residual_oil_saturation_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    # Pad capillary pressure grids for gradient calculations at interfaces
    padded_oil_water_capillary_pressure_grid = edge_pad_grid(
        oil_water_capillary_pressure_grid
    )
    padded_gas_oil_capillary_pressure_grid = edge_pad_grid(
        gas_oil_capillary_pressure_grid
    )

    # Initialize a new grid to store the updated pressures for the current time step
    updated_oil_pressure_grid = current_oil_pressure_grid.copy()

    # --- ITERATE OVER INTERIOR CELLS FOR EXPLICIT PRESSURE UPDATE ---
    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        current_cell_padded_index_x, current_cell_padded_index_y = (
            i + 1,
            j + 1,
        )  # Padded indices for current cell (i,j)

        current_cell_depth = height_grid[i, j]
        current_cell_oil_pressure = padded_oil_pressure_grid[
            current_cell_padded_index_x, current_cell_padded_index_y
        ]
        current_cell_porosity = porosity_grid[i, j]
        current_cell_total_compressibility = total_compressibility_grid[i, j]

        # Get current cell's capillary pressures (from padded grid for consistency with neighbors)
        current_cell_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                current_cell_padded_index_x, current_cell_padded_index_y
            ]
        )
        current_cell_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                current_cell_padded_index_x, current_cell_padded_index_y
            ]
        )

        # For the volumetric flux into the current cell in the x direction,
        # [ (A_{i,j} / Δx²) * (λ_{i+1/2,j} · (pⁿ_{i+1,j} - pⁿ_{i,j}) - λ_{i-1/2,j} · (pⁿ_{i,j} - pⁿ_{i-1,j})) ]
        # Where A_{i,j} = cell_size_y * cell_depth (assuming unit depth for 2D)

        # Calculate Fluxes from East Neighbor (i+1,j) - λ_{i+1/2,j} · (pⁿ_{i+1,j} - pⁿ_{i,j})
        east_neighbor_padded_index_x, east_neighbor_padded_index_y = (
            current_cell_padded_index_x + 1,
            current_cell_padded_index_y,
        )

        # Get values from East neighbor
        east_neighbor_oil_pressure = padded_oil_pressure_grid[
            east_neighbor_padded_index_x, east_neighbor_padded_index_y
        ]
        east_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                east_neighbor_padded_index_x, east_neighbor_padded_index_y
            ]
        )
        east_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                east_neighbor_padded_index_x, east_neighbor_padded_index_y
            ]
        )

        # Calculate pressure differences relative to current cell (Neighbor - Current)
        # These represent the gradients driving flow from current to neighbor, or vice versa
        oil_pressure_difference_east = (
            east_neighbor_oil_pressure - current_cell_oil_pressure
        )
        average_pressure_east = (
            current_cell_oil_pressure + east_neighbor_oil_pressure
        ) / 2.0
        oil_water_capillary_pressure_difference_east = (
            east_neighbor_oil_water_capillary_pressure
            - current_cell_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_east = (
            east_neighbor_gas_oil_capillary_pressure
            - current_cell_gas_oil_capillary_pressure
        )

        # Calculate harmonic mobilities for each phase across the East face
        water_harmonic_mobility_east = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=east_neighbor_padded_index_x,
            j2=east_neighbor_padded_index_y,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_east = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=east_neighbor_padded_index_x,
            j2=east_neighbor_padded_index_y,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_east = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=east_neighbor_padded_index_x,
            j2=east_neighbor_padded_index_y,
            mobility_grid=padded_gas_mobility_grid,
        )

        # Calculate volumetric flux for each phase from East neighbor INTO the current cell
        # Flux_in = λ * (P_neighbor - P_current)
        # P_water_neighbor - P_water_current = (P_oil_neighbor - P_oil_current) - (neighbour_cell_oil_water_capillary_pressurebor - current_cell_oil_water_capillary_pressureent)
        # P_gas_neighbor - P_gas_current = (P_oil_neighbor - P_oil_current) + (P_cgo_neighbor - current_cell_gas_oil_capillary_pressureent)

        # For Oil and Water:
        # q = λ * (A / L) * ∆P (bbl/day)
        # A = face area (dy * unit_depth)
        water_pressure_difference_east = (
            oil_pressure_difference_east - oil_water_capillary_pressure_difference_east
        )
        water_volumetric_flux_from_east = (
            water_harmonic_mobility_east * water_pressure_difference_east
        )
        oil_volumetric_flux_from_east = (
            oil_harmonic_mobility_east * oil_pressure_difference_east
        )
        # For gas:
        # q = 2P_avg * λ * ∆P
        # Here, P_avg is the average pressure between the two cells
        # We used this to linearize our equation as it is easier to follow
        # than the pressure squared difference approximation.
        gas_pressure_difference_east = (
            oil_pressure_difference_east + gas_oil_capillary_pressure_difference_east
        )
        gas_volumetric_flux_from_east = (
            (2 * average_pressure_east)
            * gas_harmonic_mobility_east
            * gas_pressure_difference_east
        )

        # Add these incoming fluxes to the net total for the cell, q (ft³/day)
        total_volumetric_flux_from_east = (
            (water_volumetric_flux_from_east * BBL_TO_FT3)
            + (oil_volumetric_flux_from_east * BBL_TO_FT3)
            + gas_volumetric_flux_from_east  # This is in ft³/day
        )

        # Calculate Fluxes from West Neighbor (i-1,j) - λ_{i-1/2,j} · (pⁿ_{i,j} - pⁿ_{i-1,j})
        west_neighbor_padded_index_x, west_neighbor_padded_index_y = (
            current_cell_padded_index_x - 1,
            current_cell_padded_index_y,
        )

        west_neighbor_oil_pressure = padded_oil_pressure_grid[
            west_neighbor_padded_index_x, west_neighbor_padded_index_y
        ]
        west_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                west_neighbor_padded_index_x, west_neighbor_padded_index_y
            ]
        )
        west_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                west_neighbor_padded_index_x, west_neighbor_padded_index_y
            ]
        )

        oil_pressure_difference_west = (
            current_cell_oil_pressure - west_neighbor_oil_pressure
        )
        average_pressure_west = (
            current_cell_oil_pressure + west_neighbor_oil_pressure
        ) / 2.0
        oil_water_capillary_pressure_difference_west = (
            current_cell_oil_water_capillary_pressure
            - west_neighbor_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_west = (
            current_cell_gas_oil_capillary_pressure
            - west_neighbor_gas_oil_capillary_pressure
        )

        water_harmonic_mobility_west = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=west_neighbor_padded_index_x,
            j2=west_neighbor_padded_index_y,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_west = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=west_neighbor_padded_index_x,
            j2=west_neighbor_padded_index_y,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_west = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=west_neighbor_padded_index_x,
            j2=west_neighbor_padded_index_y,
            mobility_grid=padded_gas_mobility_grid,
        )

        water_pressure_difference_west = (
            oil_pressure_difference_west - oil_water_capillary_pressure_difference_west
        )
        water_volumetric_flux_from_west = (
            water_harmonic_mobility_west * water_pressure_difference_west
        )
        oil_volumetric_flux_from_west = (
            oil_harmonic_mobility_west * oil_pressure_difference_west
        )
        gas_pressure_difference_west = (
            oil_pressure_difference_west + gas_oil_capillary_pressure_difference_west
        )
        gas_volumetric_flux_from_west = (
            (2 * average_pressure_west)
            * gas_harmonic_mobility_west
            * gas_pressure_difference_west
        )

        total_volumetric_flux_from_west = (
            (water_volumetric_flux_from_west * BBL_TO_FT3)
            + (oil_volumetric_flux_from_west * BBL_TO_FT3)
            + gas_volumetric_flux_from_west
        )

        net_volumetric_flux_into_cell_in_x_direction = (
            total_volumetric_flux_from_east - total_volumetric_flux_from_west
        )
        net_volumetric_flow_rate_into_cell_in_x_direction = (
            net_volumetric_flux_into_cell_in_x_direction
            * ((cell_size_y * current_cell_depth) / cell_size_x**2)  # A_{i,j} / Δx²
            # Multiply by face area (dy * cell_depth) / Δx²
        )

        # For the volumetric flux into the current cell in the y direction,
        # [ (A_{i,j} / Δy²) * (λ_{i,j+1/2} · (pⁿ_{i,j+1} - pⁿ_{i,j}) - λ_{i,j-1/2} · (pⁿ_{i,j} - pⁿ_{i,j-1})) ]
        # Where A_{i,j} = cell_size_x * cell_depth (assuming unit depth for 2D)

        # Calculate Fluxes from North Neighbor (i,j+1) - λ_{i,j+1/2} · (pⁿ_{i,j+1} - pⁿ_{i,j})
        north_neighbor_padded_index_x, north_neighbor_padded_index_y = (
            current_cell_padded_index_x,
            current_cell_padded_index_y + 1,
        )

        north_neighbor_oil_pressure = padded_oil_pressure_grid[
            north_neighbor_padded_index_x, north_neighbor_padded_index_y
        ]
        north_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                north_neighbor_padded_index_x, north_neighbor_padded_index_y
            ]
        )
        north_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                north_neighbor_padded_index_x, north_neighbor_padded_index_y
            ]
        )

        oil_pressure_difference_north = (
            north_neighbor_oil_pressure - current_cell_oil_pressure
        )
        average_pressure_north = (
            current_cell_oil_pressure + north_neighbor_oil_pressure
        ) / 2.0
        oil_water_capillary_pressure_difference_north = (
            north_neighbor_oil_water_capillary_pressure
            - current_cell_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_north = (
            north_neighbor_gas_oil_capillary_pressure
            - current_cell_gas_oil_capillary_pressure
        )

        water_harmonic_mobility_north = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=north_neighbor_padded_index_x,
            j2=north_neighbor_padded_index_y,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_north = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=north_neighbor_padded_index_x,
            j2=north_neighbor_padded_index_y,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_north = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=north_neighbor_padded_index_x,
            j2=north_neighbor_padded_index_y,
            mobility_grid=padded_gas_mobility_grid,
        )

        water_pressure_difference_north = (
            oil_pressure_difference_north
            - oil_water_capillary_pressure_difference_north
        )
        water_volumetric_flux_from_north = (
            water_harmonic_mobility_north * water_pressure_difference_north
        )
        oil_volumetric_flux_from_north = (
            oil_harmonic_mobility_north * oil_pressure_difference_north
        )
        gas_pressure_difference_north = (
            oil_pressure_difference_north + gas_oil_capillary_pressure_difference_north
        )
        gas_volumetric_flux_from_north = (
            (2 * average_pressure_north)
            * gas_harmonic_mobility_north
            * gas_pressure_difference_north
        )

        total_volumetric_flux_from_north = (
            (water_volumetric_flux_from_north * BBL_TO_FT3)
            + (oil_volumetric_flux_from_north * BBL_TO_FT3)
            + gas_volumetric_flux_from_north
        )

        # Calculate Fluxes from South Neighbor (i,j-1) - λ_{i,j-1/2} · (pⁿ_{i,j} - pⁿ_{i,j-1})
        south_neighbor_padded_index_x, south_neighbor_padded_index_y = (
            current_cell_padded_index_x,
            current_cell_padded_index_y - 1,
        )

        south_neighbor_oil_pressure = padded_oil_pressure_grid[
            south_neighbor_padded_index_x, south_neighbor_padded_index_y
        ]
        south_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                south_neighbor_padded_index_x, south_neighbor_padded_index_y
            ]
        )
        south_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                south_neighbor_padded_index_x, south_neighbor_padded_index_y
            ]
        )

        oil_pressure_difference_south = (
            current_cell_oil_pressure - south_neighbor_oil_pressure
        )
        average_pressure_south = (
            current_cell_oil_pressure + south_neighbor_oil_pressure
        ) / 2.0
        oil_water_capillary_pressure_difference_south = (
            current_cell_oil_water_capillary_pressure
            - south_neighbor_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_south = (
            current_cell_gas_oil_capillary_pressure
            - south_neighbor_gas_oil_capillary_pressure
        )

        water_harmonic_mobility_south = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=south_neighbor_padded_index_x,
            j2=south_neighbor_padded_index_y,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_south = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=south_neighbor_padded_index_x,
            j2=south_neighbor_padded_index_y,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_south = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=south_neighbor_padded_index_x,
            j2=south_neighbor_padded_index_y,
            mobility_grid=padded_gas_mobility_grid,
        )
        water_pressure_difference_south = (
            oil_pressure_difference_south
            - oil_water_capillary_pressure_difference_south
        )
        water_volumetric_flux_from_south = (
            water_harmonic_mobility_south * water_pressure_difference_south
        )
        oil_volumetric_flux_from_south = (
            oil_harmonic_mobility_south * oil_pressure_difference_south
        )
        gas_pressure_difference_south = (
            oil_pressure_difference_south + gas_oil_capillary_pressure_difference_south
        )
        gas_volumetric_flux_from_south = (
            (2 * average_pressure_south)
            * gas_harmonic_mobility_south
            * gas_pressure_difference_south
        )

        total_volumetric_flux_from_south = (
            (water_volumetric_flux_from_south * BBL_TO_FT3)
            + (oil_volumetric_flux_from_south * BBL_TO_FT3)
            + gas_volumetric_flux_from_south
        )

        net_volumetric_flux_into_cell_in_y_direction = (
            total_volumetric_flux_from_north - total_volumetric_flux_from_south
        )

        net_volumetric_flow_rate_into_cell_in_y_direction = (
            net_volumetric_flux_into_cell_in_y_direction
            * ((cell_size_x * current_cell_depth) / cell_size_y**2)  # A_{i,j} / Δy²
            # Multiply by face area (dx * current_cell_depth) / Δy²
        )

        # Combine the net volumetric flow rates from both directions
        # [ (A_{i,j} / Δx²) * (λ_{i+1/2,j} · (pⁿ_{i+1,j} - pⁿ_{i,j}) - λ_{i-1/2,j} · (pⁿ_{i,j} - pⁿ_{i-1,j})) ]+
        # [ (A_{i,j} / Δy²) * (λ_{i,j+1/2} · (pⁿ_{i,j+1} - pⁿ_{i,j}) - λ_{i,j-1/2} · (pⁿ_{i,j} - pⁿ_{i,j-1})) ]
        net_volumetric_flow_rate_into_cell = (  # (ft³/day)
            net_volumetric_flow_rate_into_cell_in_x_direction
            + net_volumetric_flow_rate_into_cell_in_y_direction
        )
        # Add Source/Sink Term (WellParameters) - q * A∆x (ft³/day)
        injection_well, production_well = wells[i, j]
        cell_injection_rate = cell_production_rate = 0.0

        if injection_well is not None:
            # If there is an injection well, add its flow rate to the cell
            cell_injection_rate = (
                injection_well.injected_fluid.volumetric_flow_rate
            )  # STB/day or SCF/day
            injected_phase = injection_well.injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                # Get the volumetric flow rate in ft³/day
                cell_injection_rate *= (
                    fluid_properties.gas_formation_volume_factor_grid[i, j]
                )  # ft³/SCF
            elif injected_phase == FluidPhase.WATER:
                # For water, convert bbl/day to ft³/day
                cell_injection_rate *= (
                    fluid_properties.water_formation_volume_factor_grid[i, j]  # bbl/STB
                    * BBL_TO_FT3  # Convert bbl/day to ft³/day
                )
            else:
                # For oil and water, convert bbl/day to ft³/day
                cell_injection_rate *= (
                    fluid_properties.oil_formation_volume_factor_grid[i, j]  # bbl/STB
                    * BBL_TO_FT3  # Convert bbl/day to ft³/day
                )

        if production_well is not None:
            # If there is a production well, subtract its flow rate from the cell
            for produced_fluid in production_well.produced_fluids:
                production_rate = (
                    produced_fluid.volumetric_flow_rate
                )  # STB/day or SCF/day
                if produced_fluid.phase == FluidPhase.GAS:
                    # Get the volumetric flow rate in ft³/day
                    production_rate *= (
                        fluid_properties.gas_formation_volume_factor_grid[i, j]
                    )
                elif produced_fluid.phase == FluidPhase.WATER:
                    # For water, convert bbl/day to ft³/day
                    production_rate *= (
                        fluid_properties.water_formation_volume_factor_grid[
                            i, j
                        ]  # bbl/STB
                        * BBL_TO_FT3  # Convert bbl/day to ft³/day
                    )
                else:
                    # For oil, convert bbl/day to ft³/day
                    production_rate *= (
                        fluid_properties.oil_formation_volume_factor_grid[
                            i, j
                        ]  # bbl/STB
                        * BBL_TO_FT3  # Convert bbl/day to ft³/day
                    )
                cell_production_rate += production_rate

        # Calculate the net well flow rate into the cell
        # q_{i,j} * A_{i,j} * Δx (ft³/day) = # (q_{i,j}_injection - q_{i,j}_production)
        net_well_flow_rate_into_cell = cell_injection_rate - cell_production_rate

        # Add the well flow rate to the net volumetric flow rate into the cell
        # Total flow rate into the cell (ft³/day)
        # [
        #     [ (A_{i,j} / Δx²) * (λ_{i+1/2,j} · (pⁿ_{i+1,j} - pⁿ_{i,j}) - λ_{i-1/2,j} · (pⁿ_{i,j} - pⁿ_{i-1,j})) ]+
        #     [ (A_{i,j} / Δy²) * (λ_{i,j+1/2} · (pⁿ_{i,j+1} - pⁿ_{i,j}) - λ_{i,j-1/2} · (pⁿ_{i,j} - pⁿ_{i,j-1})) ]+
        #     (q_{i,j} * A_{i,j} * Δx)
        # ]
        total_flow_rate_into_cell = (
            net_volumetric_flow_rate_into_cell + net_well_flow_rate_into_cell
        )

        # Full Explicit Pressure Update Equation (P_oil) ---
        # The accumulation term is (φ * C_t * cell_volume) * dP_oil/dt
        # dP_oil/dt = [Net_Volumetric_Flow_Rate_Into_Cell] / (φ * C_t * cell_volume)
        # dP_{i,j} = (Δt / [(φ·c_t) * A∆x]) * [
        #     [ (A_{i,j} / Δx²) * (λ_{i+1/2,j} · (pⁿ_{i+1,j} - pⁿ_{i,j}) - λ_{i-1/2,j} · (pⁿ_{i,j} - pⁿ_{i-1,j})) ]+
        #     [ (A_{i,j} / Δy²) * (λ_{i,j+1/2} · (pⁿ_{i,j+1} - pⁿ_{i,j}) - λ_{i,j-1/2} · (pⁿ_{i,j} - pⁿ_{i,j-1})) ]+
        #     (q_{i,j} * A_{i,j} * Δx)
        # ]

        time_step_size_in_days = time_step_size * SECONDS_TO_DAYS
        change_in_pressure = (
            time_step_size_in_days
            / (
                current_cell_porosity
                * current_cell_total_compressibility
                * (
                    cell_size_x * cell_size_y * current_cell_depth
                )  # Volume of the cell in ft³
            )
        ) * total_flow_rate_into_cell

        # Apply the update to the pressure grid
        # P_oil^(n+1) = P_oil^n + dP_oil
        updated_oil_pressure_grid[i, j] += change_in_pressure
    return updated_oil_pressure_grid


"""
Implicit finite difference formulation for pressure diffusion in a 2D reservoir
(slightly compressible fluid):

The governing equation for pressure evolution is the linear-flow diffusivity equation:

    (∂p/∂t * (ρ·φ·c_t) * A∆x) + (K·ρ * A∆x) = (∇ρ ⋅ (vf) * A∆x) + (∇ρ · (λ·∇p) * A) + (q * A∆x)

    Note: compressibility (c_t) is added due to the fact that the fluid is compressible
    as pressure changes affect pore volume (φ), and fluid density (ρ) and thus the mass storage term.

where:
    (∂p/∂t * (ρ·φ·c_t) - Mass storage term
    K·ρ * A∆x - Reaction term
    ∇ρ ⋅ (vf) * A∆x - Convection term (advective transport)
    ∇ρ · (λ·∇p) * A - Diffusion term
    q * A∆x - Source/sink term

Given basic assumptions:

- Constant porosity (φ), total compressibility (c_t) and density (ρ) across the grid
- No reaction term
- No convection (advection) term - Advection term models fluid transport, not pressure evolution/propagation

The equation simplifies to:

    ∂p/∂t * (φ·c_t) * A∆x = (∇ · (λ·∇p) * A) + (q * A∆x)

where:
    ∂p/∂t * (φ·c_t) * A∆x = Accumulation term (ft³/s)
    ∇ · (λ·∇p) * A = Diffusion term (ft³/s)
    q * A∆x = Source/sink term (ft³/s) (Injection/production rates normalized by area)

To break it down further, we can express the diffusion term in terms of pressure gradients:

    ∇ · (λ·∇p) = ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)

For a 2D reservoir, ∇ · (λ·∇p) this expands to:

    ∇ · (λ·∇p) = ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)

Where:
    λ = mobility = k / μ
    ∇p = gradient of pressure (∂p/∂x, ∂p/∂y)
    (λ·∇p) = q = transmissibility (mobility) times pressure gradient (from Darcy's law) (ft³/s)
    (λ·∂p/∂x) = x-direction diffusion term
    (λ·∂p/∂y) = y-direction diffusion term

So ∇ · (λ·∇p) becomes the sum of the differential operators in the x and y directions:

    ∇ · (λ·∇p) = ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)

So the full equation becomes:

    ∂p/∂t * (φ·c_t) * A∆x = [ (∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)) * A ] + (q * A∆x) --- eqn (α)

Moving terms around gives us the explicit update formula:

    ∂p/∂t = [1 / ((φ·c_t) * A∆x)] * [ (∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y)) * A ] + (q * A∆x)

Where:
    A     = cell cross-sectional area (ft²) = Δy * Δz (assuming unit depth for 2D, so Δz = 1 ft)
    ∆x    = cell size in x direction (ft)
    p     = pressure (psi)
    φ     = porosity (fraction)
    c_t   = total compressibility (rock + fluid) (psi⁻¹)
    λ     = mobility = k / μ (mD/cP) (convert to ft²·psi⁻¹·s⁻¹·lbm⁻¹ (0.001127) for consistency)
    k     = permeability (mD)
    μ     = viscosity (cP)
    ∂p/∂x = spatial derivative of pressure in x direction (x pressure gradient) (psi/ft)
    ∂p/∂y = spatial derivative of pressure in y direction (y pressure gradient) (psi/ft)
    q     = source/sink term (normalized injection/production) per unit volume (ft³/s * 1/ft³)
    q * A∆x = source/sink term (ft³/s) (normalized by volume)

Fully Implicit Discretization (Backward Euler in time, central difference in space):

Discretizing in time (constant space) using Backward Euler , we get:

    ∂p/∂t = (pⁿ⁺¹_ij - pⁿ_ij) / Δt

Discretizing in space (constant time) using central differences, we get:

    ∂/∂x (λ·∂p/∂x) = (λ_{i+1/2,j} · (pⁿ⁺¹_{i+1,j} - pⁿ⁺¹_{i,j}) - λ_{i-1/2,j} · (pⁿ⁺¹_{i,j} - pⁿ⁺¹_{i-1,j})) / Δx²
    ∂/∂y (λ·∂p/∂y) = (λ_{i,j+1/2} · (pⁿ⁺¹_{i,j+1} - pⁿ⁺¹_{i,j}) - λ_{i,j-1/2} · (pⁿ⁺¹_{i,j} - pⁿ⁺¹_{i,j-1})) / Δy²

Combining these gives us the implicit update formula for pressure at cell (i,j) at time step n+1:
    
    pⁿ⁺¹_ij = pⁿ_ij + (Δt / [(φ·c_t) * A∆x]) * [
        [ (A_{i,j} / Δx²) * (λ_{i+1/2,j} · (pⁿ⁺¹_{i+1,j} - pⁿ⁺¹_{i,j}) - λ_{i-1/2,j} · (pⁿ⁺¹_{i,j} - pⁿ⁺¹_{i-1,j})) ] +
        [ (A_{i,j} / Δy²) * (λ_{i,j+1/2} · (pⁿ⁺¹_{i,j+1} - pⁿ⁺¹_{i,j}) - λ_{i,j-1/2} · (pⁿ⁺¹_{i,j} - pⁿ⁺¹_{i,j-1})) ] +
        (q_{i,j} * A_{i,j} * Δx)
    ]

Where:
    Δt = time step size (s)
    Δx = cell size in x direction (m)
    Δy = cell size in y direction (m)
    pⁿ_ij = pressure at cell (i,j) at time step n (psi)
    pⁿ⁺¹_ij = pressure at cell (i,j) at time step n+1 (psi)
    λ_{i+1/2,j} = harmonic average transmissibility between cells (i,j) and (i+1,j)
    λ_{i-1/2,j} = harmonic average transmissibility between cells (i,j) and (i-1,j)
    λ_{i,j+1/2} = harmonic average transmissibility between cells (i,j) and (i,j+1) 
    λ_{i,j-1/2} = harmonic average transmissibility between cells (i,j) and (i,j-1)
    q_{i,j} = source/sink term at cell (i,j) (m³/s)

Discretized as a linear system:
    A · pⁿ⁺¹ = b

Full discretized form for a grid cell (i,j):

    (pⁿ⁺¹_ij - pⁿ_ij) * [(φ·c_t * A∆x) / Δt] =
        [ (A_{i,j} / Δx²) * (λ_{i+1/2,j} · (pⁿ⁺¹_{i+1,j} - pⁿ⁺¹_{i,j}) - λ_{i-1/2,j} · (pⁿ⁺¹_{i,j} - pⁿ⁺¹_{i-1,j})) ] +
        [ (A_{i,j} / Δy²) * (λ_{i,j+1/2} · (pⁿ⁺¹_{i,j+1} - pⁿ⁺¹_{i,j}) - λ_{i,j-1/2} · (pⁿ⁺¹_{i,j} - pⁿ⁺¹_{i,j-1})) ] +
        (q_{i,j} * A_{i,j} * Δx)

Definitions:
    φ               = porosity (fraction)
    c_t             = total compressibility (psi⁻¹)
    λ               = mobility (k / μ) at cell interfaces (mD/cP or field units)
    pⁿ_ij           = pressure at time step n at cell (i,j)
    pⁿ⁺¹_ij         = pressure at time step n+1 (unknown)
    A_{i,j}         = cross-sectional area of current cell (ft²)
    Δx, Δy          = grid spacing in x and y directions (ft)
    Δt              = time step (s)
    q_{i,j}         = source/sink rate per unit volume (ft³/s/ft³)
    λ_{i±1/2,j}, λ_{i,j±1/2} = harmonic average mobility at interfaces

Matrix system construction:

Let:
    Tx⁺ = λ_{i+1/2,j} * A_{i,j} / Δx²
    Tx⁻ = λ_{i-1/2,j} * A_{i,j} / Δx²
    Ty⁺ = λ_{i,j+1/2} * A_{i,j} / Δy²
    Ty⁻ = λ_{i,j-1/2} * A_{i,j} / Δy²
    β   = φ·c_t·A·Δx / Δt

Matrix entries for cell (i,j):

    A_{ij,ij}     = β + Tx⁺ + Tx⁻ + Ty⁺ + Ty⁻     (main diagonal)
    A_{ij,i+1j}   = -Tx⁺                         (east neighbor)
    A_{ij,i-1j}   = -Tx⁻                         (west neighbor)
    A_{ij,ij+1}   = -Ty⁺                         (north neighbor)
    A_{ij,ij-1}   = -Ty⁻                         (south neighbor)

Right-hand side vector b:

    b_{ij} = (β * pⁿ_{ij}) + (q_{ij}^{n+1} * A_{i,j} * Δx) + total_capillary_pressure_term

Where:
    total_capillary_pressure_term = contributions from capillary pressure gradients for each direction:
    = ∑[ [ (λ_w · ∇P_cow) + (λ_g · ∇P_cgo) ] * A / Δx ]

    For the west neighbor, it is:
    = [ (λ_w * (P_cow_{i,j} - P_cow_{i-1,j})) + (λ_g * (P_cgo_{i,j} - P_cgo_{i-1,j})) ] * A_{i,j} / Δx

    For the east neighbor, it is:
    = [ (λ_w * (P_cow_{i+1,j} - P_cow_{i,j})) + (λ_g * (P_cgo_{i+1,j} - P_cgo_{i,j})) ] * A_{i,j} / Δx

    For the north neighbor, it is:
    = [ (λ_w * (P_cow_{i,j+1} - P_cow_{i,j})) + (λ_g * (P_cgo_{i,j+1} - P_cgo_{i,j})) ] * A_{i,j} / Δy

    For the south neighbor, it is:
    = [ (λ_w * (P_cow_{i,j} - P_cow_{i,j-1})) + (λ_g * (P_cgo_{i,j} - P_cgo_{i,j-1})) ] * A_{i,j} / Δy

Total capillary pressure term for cell (i,j):

    total_capillary_pressure_term = (
        capillary_pressure_term_east - capillary_pressure_term_west 
        + capillary_pressure_term_north - capillary_pressure_term_south
    )

Notes:
    - This forms a sparse matrix with a 5-point stencil for each interior cell.
    - The system A·p = b is solved at each time step to update pressure.
    - Harmonic averaging is used at cell interfaces for mobility.
    - Units must be consistent (e.g., ft, psi, s).
    - The source term is applied in reservoir volume units (not surface volumes).
    
Stability Condition:
    Unconditionally stable for implicit methods, but requires careful selection of Δt

Where:
    A encodes mobility (harmonic average of λ), geometry, and boundary conditions.
    pⁿ⁺¹ is the pressure vector at the next time step.
    b contains contributions from previous pressures and sources/sinks.
"""


def compute_implicit_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    height_grid: TwoDimensionalGrid,
    time_step_size: float,
    boundary_conditions: BoundaryConditions,
    rock_properties: RockProperties,
    fluid_properties: FluidProperties,
    wells: Wells,
) -> TwoDimensionalGrid:
    """
    Computes the pressure evolution (specifically, oil phase pressure P_oil) in the reservoir grid
    for one time step using an IMPES (Implicit Pressure, Explicit Saturation) finite difference method.
    This function implements the implicit version of the explicit three-phase flow equation provided previously.

    The governing equation for pressure evolution (derived from total fluid conservation) is:
    φ·c_t_total · ∂P_oil/∂t = ∇ · [ (λ_o + λ_w + λ_g) · ∇P_oil ]
                            - ∇ · [ λ_w · ∇P_cow ]
                            + ∇ · [ λ_g · ∇P_cgo ]
                            + q_total

    In the implicit formulation, we solve for P_oil at the new time step (n+1):

    A * P_oil^(n+1) = B

    Where:
    - Left-hand side (A matrix) contains terms related to P_oil^(n+1) (accumulation and total pressure-driven flux).
    - Right-hand side (B vector) contains accumulation terms from P_oil^n, and explicit flux terms
      from capillary pressure gradients (P_cow, P_cgo are evaluated at time step n, using old saturations).
    - Mobilities (λ) are also evaluated at time step n.

    This formulation keeps the system linear for P_oil, making it solvable with direct methods like `spsolve`.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet (ft).
    :param height_grid: A 2D grid representing the depth of each cell in the reservoir (ft).
    :param time_step_size: Time step size (s) for the implicit scheme.
    :param boundary_conditions: Boundary conditions for pressure and saturation grids.
    :param rock_properties: ``RockProperties`` object containing rock physical properties including
        absolute permeability, porosity, residual saturations, and relative/capillary
        pressure parameters (which include wettability information).

    :param fluid_properties: `FluidProperties` object containing fluid physical properties like
        pressure, temperature, saturations, viscosities, and compressibilities for
        water, oil, and gas.

    :param wells: `Wells` object containing information about injection and production wells,
    :return: A 2D numpy array representing the updated oil phase pressure field (psi).
    """
    # Extract properties from provided objects for clarity and convenience
    absolute_permeability_grid = rock_properties.absolute_permeability_grid
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_grid = rock_properties.residual_oil_saturation_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_params = rock_properties.relative_permeability_params
    capillary_pressure_params = rock_properties.capillary_pressure_params

    current_oil_pressure_grid = (
        fluid_properties.pressure_grid
    )  # This is P_oil at time n
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
    cell_count_x, cell_count_y = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Compute total fluid system compressibility for each cell (evaluated at time n)
    total_fluid_compressibility_grid = build_2D_total_fluid_compressibility_grid(
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

    total_compressibility_grid = np.maximum(total_compressibility_grid, 1e-18)

    padded_oil_pressure_grid = edge_pad_grid(current_oil_pressure_grid.copy())
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

    # Apply boundary conditions to relevant padded grids
    # Pressure BCs will be applied directly to the matrix/RHS during assembly for new pressure.
    # Saturation BCs are applied here as they are used to compute mobilities and capillary pressures explicitly.
    boundary_conditions["pressure"].apply(padded_oil_pressure_grid)
    boundary_conditions["water_saturation"].apply(padded_water_saturation_grid)
    boundary_conditions["gas_saturation"].apply(padded_gas_saturation_grid)

    # Compute phase mobilities (kr / mu) for each cell based on old saturations
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_2D_three_phase_relative_mobilities_grids(
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

    # Compute mobility grids for harmonic averaging at interfaces
    # λ_x = k_abs * (kr / mu) / B_x. Where B_x is the formation volume factor of the phase
    water_mobility_grid = (
        absolute_permeability_grid
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid = (
        absolute_permeability_grid
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid = (
        absolute_permeability_grid
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    # Pad mobility grids for neighbor access
    padded_water_mobility_grid = edge_pad_grid(water_mobility_grid)
    padded_oil_mobility_grid = edge_pad_grid(oil_mobility_grid)
    padded_gas_mobility_grid = edge_pad_grid(gas_mobility_grid)

    # Compute Capillary Pressures Grids (local to each cell, based on old saturations)
    # These are also padded
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        build_2D_three_phase_capillary_pressure_grids(
            water_saturation_grid=padded_water_saturation_grid,  # Using padded for consistency
            gas_saturation_grid=padded_gas_saturation_grid,  # Using padded for consistency
            irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
            residual_oil_saturation_grid=padded_residual_oil_saturation_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    # Pad capillary pressure grids for gradient calculations at interfaces
    padded_oil_water_capillary_pressure_grid = edge_pad_grid(
        oil_water_capillary_pressure_grid
    )
    padded_gas_oil_capillary_pressure_grid = edge_pad_grid(
        gas_oil_capillary_pressure_grid
    )

    # Initialize sparse coefficient matrix (A * P_o_new = b)
    total_cell_count = cell_count_x * cell_count_y
    A = lil_matrix((total_cell_count, total_cell_count), dtype=np.float64)
    # Initialize RHS source/accumulation vector
    b = np.zeros(total_cell_count)

    def to_1d_index(i: int, j: int) -> int:
        """Converts 2D grid indices (i, j) to a 1D index for the sparse matrix."""
        if not (0 <= i < cell_count_x and 0 <= j < cell_count_y):
            # This should not happen for internal loops, but good for robust checks.
            return -1  # Indicate out of bounds for special handling or error
        return i * cell_count_y + j

    # Iterate over interior cells to populate the linear system
    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        current_cell_padded_index_x, current_cell_padded_index_y = i + 1, j + 1
        current_cell_1d_index = to_1d_index(i, j)
        current_cell_depth = height_grid[i, j]
        current_cell_volume = cell_size_x * cell_size_y * current_cell_depth
        current_cell_porosity = porosity_grid[i, j]
        current_cell_total_compressibility = total_compressibility_grid[i, j]
        current_cell_oil_pressure = current_oil_pressure_grid[i, j]
        time_step_size_in_days = time_step_size * SECONDS_TO_DAYS
        current_cell_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                current_cell_padded_index_x, current_cell_padded_index_y
            ]
        )
        current_cell_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                current_cell_padded_index_x, current_cell_padded_index_y
            ]
        )

        # Accumulation term coefficient for the diagonal of A
        # β = φ·c_t·A·Δx / Δt
        accumulation_coefficient = (
            current_cell_porosity
            * current_cell_total_compressibility
            * current_cell_volume
        ) / time_step_size_in_days

        # For the east neighbor (i+1, j) - Tx⁺ = λ_{i+1/2,j} * A / Δx²
        east_neighbor_padded_index_x, east_neighbor_padded_index_y = (
            current_cell_padded_index_x + 1,
            current_cell_padded_index_y,
        )
        # Get values from East neighbor
        east_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                east_neighbor_padded_index_x, east_neighbor_padded_index_y
            ]
        )
        east_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                east_neighbor_padded_index_x, east_neighbor_padded_index_y
            ]
        )

        # Calculate pressure differences relative to current cell (Neighbor - Current)
        # These represent the gradients driving flow from current to neighbor, or vice versa
        oil_water_capillary_pressure_difference_east = (
            east_neighbor_oil_water_capillary_pressure
            - current_cell_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_east = (
            east_neighbor_gas_oil_capillary_pressure
            - current_cell_gas_oil_capillary_pressure
        )

        # Calculate harmonic mobilities for each phase across the East face
        water_harmonic_mobility_east = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=east_neighbor_padded_index_x,
            j2=east_neighbor_padded_index_y,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_east = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=east_neighbor_padded_index_x,
            j2=east_neighbor_padded_index_y,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_east = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=east_neighbor_padded_index_x,
            j2=east_neighbor_padded_index_y,
            mobility_grid=padded_gas_mobility_grid,
        )
        # λ_{i+1/2,j}
        total_harmonic_mobility_east = (
            water_harmonic_mobility_east
            + oil_harmonic_mobility_east
            + gas_harmonic_mobility_east
        )
        # Tx⁺ = λ_{i+1/2,j} * A / Δx²
        transmissibility_east = (
            total_harmonic_mobility_east
            * (cell_count_y * current_cell_depth)
            / cell_size_x**2
        )
        # λ_w * (P_cow_{i+1,j} - P_cow_{i,j}) (ft²/psi.day * psi = ft²/day)
        water_capillary_pressure_term_east = (
            water_harmonic_mobility_east * oil_water_capillary_pressure_difference_east
        )
        # λ_g * (P_cgo_{i+1,j} - P_cgo_{i,j}) (ft²/psi.day * psi = ft²/day)
        gas_capillary_pressure_term_east = (
            gas_harmonic_mobility_east * gas_oil_capillary_pressure_difference_east
        )
        # Total capillary pressure term for the east neighbor (ft²/day * ft = ft³/day)
        capillary_pressure_term_east = (
            (water_capillary_pressure_term_east + gas_capillary_pressure_term_east)
            * (cell_size_y * current_cell_depth)
            / cell_size_x**2
        )  # Area of the face in ft²

        # For the West neighbor (i-1, j) - Tx⁻ = λ_{i-1/2,j} * A / Δx²
        west_neighbor_padded_index_x, west_neighbor_padded_index_y = (
            current_cell_padded_index_x - 1,
            current_cell_padded_index_y,
        )

        west_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                west_neighbor_padded_index_x, west_neighbor_padded_index_y
            ]
        )
        west_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                west_neighbor_padded_index_x, west_neighbor_padded_index_y
            ]
        )

        oil_water_capillary_pressure_difference_west = (
            current_cell_oil_water_capillary_pressure
            - west_neighbor_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_west = (
            current_cell_gas_oil_capillary_pressure
            - west_neighbor_gas_oil_capillary_pressure
        )

        water_harmonic_mobility_west = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=west_neighbor_padded_index_x,
            j2=west_neighbor_padded_index_y,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_west = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=west_neighbor_padded_index_x,
            j2=west_neighbor_padded_index_y,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_west = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=west_neighbor_padded_index_x,
            j2=west_neighbor_padded_index_y,
            mobility_grid=padded_gas_mobility_grid,
        )

        # λ_{i-1/2,j}
        total_harmonic_mobility_west = (
            water_harmonic_mobility_west
            + oil_harmonic_mobility_west
            + gas_harmonic_mobility_west
        )
        # Tx⁻ = λ_{i-1/2,j} * A / Δx²
        transmissibility_west = (
            total_harmonic_mobility_west
            * (cell_count_y * current_cell_depth)
            / cell_size_x**2
        )
        # λ_w * (P_cow_{i,j} - P_cow_{i-1,j})
        water_capillary_pressure_term_west = (
            water_harmonic_mobility_west * oil_water_capillary_pressure_difference_west
        )
        # λ_g * (P_cgo_{i,j} - P_cgo_{i-1,j})
        gas_capillary_pressure_term_west = (
            gas_harmonic_mobility_west * gas_oil_capillary_pressure_difference_west
        )
        # Total capillary pressure term for the west neighbor
        capillary_pressure_term_west = (
            (water_capillary_pressure_term_west + gas_capillary_pressure_term_west)
            * (cell_size_y * current_cell_depth)
            / cell_size_x**2
        )  # Area of the face in ft²

        # For the North neighbor (i, j+1) - Ty⁺ = λ_{i,j+1/2} * A / Δy²
        north_neighbor_padded_index_x, north_neighbor_padded_index_y = (
            current_cell_padded_index_x,
            current_cell_padded_index_y + 1,
        )
        north_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                north_neighbor_padded_index_x, north_neighbor_padded_index_y
            ]
        )
        north_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                north_neighbor_padded_index_x, north_neighbor_padded_index_y
            ]
        )

        oil_water_capillary_pressure_difference_north = (
            north_neighbor_oil_water_capillary_pressure
            - current_cell_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_north = (
            north_neighbor_gas_oil_capillary_pressure
            - current_cell_gas_oil_capillary_pressure
        )

        water_harmonic_mobility_north = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=north_neighbor_padded_index_x,
            j2=north_neighbor_padded_index_y,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_north = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=north_neighbor_padded_index_x,
            j2=north_neighbor_padded_index_y,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_north = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=north_neighbor_padded_index_x,
            j2=north_neighbor_padded_index_y,
            mobility_grid=padded_gas_mobility_grid,
        )
        # λ_{i,j+1/2}
        total_harmonic_mobility_north = (
            water_harmonic_mobility_north
            + oil_harmonic_mobility_north
            + gas_harmonic_mobility_north
        )
        # Ty⁺ = λ_{i,j+1/2} * A / Δy²
        transmissibility_north = (
            total_harmonic_mobility_north
            * (cell_count_x * current_cell_depth)
            / cell_size_y**2
        )
        # λ_w * (P_cow_{i,j+1} - P_cow_{i,j})
        water_capillary_pressure_term_north = (
            water_harmonic_mobility_north
            * oil_water_capillary_pressure_difference_north
        )
        # λ_g * (P_cgo_{i,j+1} - P_cgo_{i,j})
        gas_capillary_pressure_term_north = (
            gas_harmonic_mobility_north * gas_oil_capillary_pressure_difference_north
        )
        # Total capillary pressure term for the north neighbor
        capillary_pressure_term_north = (
            (water_capillary_pressure_term_north + gas_capillary_pressure_term_north)
            * (cell_size_x * current_cell_depth)
            / cell_size_y**2
        )  # Area of the face in ft²

        # For the South neighbor (i, j-1) - Ty⁻ = λ_{i,j-1/2} * A / Δy²
        south_neighbor_padded_index_x, south_neighbor_padded_index_y = (
            current_cell_padded_index_x,
            current_cell_padded_index_y - 1,
        )
        south_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                south_neighbor_padded_index_x, south_neighbor_padded_index_y
            ]
        )
        south_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[
                south_neighbor_padded_index_x, south_neighbor_padded_index_y
            ]
        )

        oil_water_capillary_pressure_difference_south = (
            current_cell_oil_water_capillary_pressure
            - south_neighbor_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_south = (
            current_cell_gas_oil_capillary_pressure
            - south_neighbor_gas_oil_capillary_pressure
        )
        water_harmonic_mobility_south = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=south_neighbor_padded_index_x,
            j2=south_neighbor_padded_index_y,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_south = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=south_neighbor_padded_index_x,
            j2=south_neighbor_padded_index_y,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_south = compute_harmonic_mobility(
            i1=current_cell_padded_index_x,
            j1=current_cell_padded_index_y,
            i2=south_neighbor_padded_index_x,
            j2=south_neighbor_padded_index_y,
            mobility_grid=padded_gas_mobility_grid,
        )
        # λ_{i,j-1/2}
        total_harmonic_mobility_south = (
            water_harmonic_mobility_south
            + oil_harmonic_mobility_south
            + gas_harmonic_mobility_south
        )
        # Ty⁻ = λ_{i,j-1/2} * A / Δy²
        transmissibility_south = (
            total_harmonic_mobility_south
            * (cell_count_x * current_cell_depth)
            / cell_size_y**2
        )
        # λ_w * (P_cow_{i,j} - P_cow_{i,j-1})
        water_capillary_pressure_term_south = (
            water_harmonic_mobility_south
            * oil_water_capillary_pressure_difference_south
        )
        # λ_g * (P_cgo_{i,j} - P_cgo_{i,j-1})
        gas_capillary_pressure_term_south = (
            gas_harmonic_mobility_south * gas_oil_capillary_pressure_difference_south
        )
        # Total capillary pressure term for the south neighbor
        capillary_pressure_term_south = (
            (water_capillary_pressure_term_south + gas_capillary_pressure_term_south)
            * (cell_size_x * current_cell_depth)
            / cell_size_y**2
        )

        # Populate the diagonal of A matrix for current cell
        # A[current_cell_1d_index, current_cell_1d_index] = β + Tx⁺ + Tx⁻ + Ty⁺ + Ty⁻
        A[current_cell_1d_index, current_cell_1d_index] = (
            accumulation_coefficient
            + transmissibility_east
            + transmissibility_west
            + transmissibility_north
            + transmissibility_south
        )
        # For the A matrix, we need to account for the transmissibilities
        A[
            current_cell_1d_index, to_1d_index(i + 1, j)
        ] = -transmissibility_east  # East neighbor
        A[
            current_cell_1d_index, to_1d_index(i - 1, j)
        ] = -transmissibility_west  # West neighbor
        A[
            current_cell_1d_index, to_1d_index(i, j + 1)
        ] = -transmissibility_north  # North neighbor
        A[
            current_cell_1d_index, to_1d_index(i, j - 1)
        ] = -transmissibility_south  # South neighbor

        # For the right-hand side vector b: b = (β * P_oilⁿ) + (q_{i,j} * A∆x) + capillary terms
        # Add Source/Sink Term (WellParameters) - q * A∆x (ft³/day)
        injection_well, production_well = wells[i, j]
        cell_injection_rate = cell_production_rate = 0.0

        if injection_well is not None:
            # If there is an injection well, add its flow rate to the cell
            cell_injection_rate = (
                injection_well.injected_fluid.volumetric_flow_rate
            )  # STB/day or SCF/day
            injected_phase = injection_well.injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                # Get the volumetric flow rate in ft³/day
                cell_injection_rate *= (
                    fluid_properties.gas_formation_volume_factor_grid[i, j]
                )  # ft³/SCF
            elif injected_phase == FluidPhase.WATER:
                # For water, convert bbl/day to ft³/day
                cell_injection_rate *= (
                    fluid_properties.water_formation_volume_factor_grid[i, j]  # bbl/STB
                    * BBL_TO_FT3  # Convert bbl/day to ft³/day
                )
            else:
                # For oil and water, convert bbl/day to ft³/day
                cell_injection_rate *= (
                    fluid_properties.oil_formation_volume_factor_grid[i, j]  # bbl/STB
                    * BBL_TO_FT3  # Convert bbl/day to ft³/day
                )

        if production_well is not None:
            # If there is a production well, subtract its flow rate from the cell
            for produced_fluid in production_well.produced_fluids:
                production_rate = (
                    produced_fluid.volumetric_flow_rate
                )  # STB/day or SCF/day
                if produced_fluid.phase == FluidPhase.GAS:
                    # Get the volumetric flow rate in ft³/day
                    production_rate *= (
                        fluid_properties.gas_formation_volume_factor_grid[i, j]
                    )
                elif produced_fluid.phase == FluidPhase.WATER:
                    # For water, convert bbl/day to ft³/day
                    production_rate *= (
                        fluid_properties.water_formation_volume_factor_grid[
                            i, j
                        ]  # bbl/STB
                        * BBL_TO_FT3  # Convert bbl/day to ft³/day
                    )
                else:
                    # For oil, convert bbl/day to ft³/day
                    production_rate *= (
                        fluid_properties.oil_formation_volume_factor_grid[
                            i, j
                        ]  # bbl/STB
                        * BBL_TO_FT3  # Convert bbl/day to ft³/day
                    )
                cell_production_rate += production_rate

        # Calculate the net well flow rate into the cell
        # q_{i,j} * A_{i,j} * Δx (ft³/day) = # (q_{i,j}_injection - q_{i,j}_production)
        net_well_flow_rate_into_cell = cell_injection_rate - cell_production_rate

        # b = (β * P_oilⁿ) + (q_{i,j} * A∆x) + total_capillary_pressure_term
        # For the total_capillary_pressure_term, we sum contributions from all faces
        total_capillary_pressure_term = (
            capillary_pressure_term_east
            - capillary_pressure_term_west
            + capillary_pressure_term_north
            - capillary_pressure_term_south
        )
        b[current_cell_1d_index] = (
            (accumulation_coefficient * current_cell_oil_pressure)
            + net_well_flow_rate_into_cell
            + total_capillary_pressure_term
        )

    # Solve the linear system A * P_oilⁿ⁺¹ = b
    new_pressure_grid_1d = spsolve(A.tocsr(), b)
    # Reshape the 1D solution back to a 2D grid
    new_pressure_grid = new_pressure_grid_1d.reshape((cell_count_x, cell_count_y))
    return typing.cast(TwoDimensionalGrid, new_pressure_grid)


def compute_adaptive_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    height_grid: TwoDimensionalGrid,
    time_step_size: float,
    boundary_conditions: BoundaryConditions,
    rock_properties: RockProperties,
    fluid_properties: FluidProperties,
    wells: Wells,
    diffusion_number_threshold: float = 0.24,  # Slightly below 0.25 for safety
) -> TwoDimensionalGrid:
    """
    Computes the pressure distribution in the reservoir grid for a single time step,
    adaptively choosing between explicit and implicit methods based on the maximum
    diffusion number in the grid for a three-phase flow system.

    This function now uses RockProperties and FluidProperties to derive the necessary
    physical parameters for the three-phase flow.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet (ft)
    :param height_grid: 2D numpy array representing the height of each cell in the grid (ft).
    :param time_step_size: Time step size (s) for the implicit scheme
    :param boundary_conditions: Boundary conditions for pressure, saturation, etc, grids.
    :param rock_properties: RockProperties object containing rock physical properties.
    :param fluid_properties: FluidProperties object containing fluid physical properties.
    :param wells: Wells object containing information about injection and production wells.
    :param diffusion_number_threshold: The maximum allowed diffusion number for explicit stability.
        If any cell exceeds this, the implicit solver is used.

    :return: A 2D numpy array representing the updated pressure distribution (psi)
        after solving the chosen system for the current time step.
    """
    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid

    absolute_permeability_grid = rock_properties.absolute_permeability_grid
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility

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

    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_grid = rock_properties.residual_oil_saturation_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_params = rock_properties.relative_permeability_params

    cell_count_x, cell_count_y = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Compute total system compressibility for each cell (φ * cf_total + cr)
    total_fluid_compressibility_grid = build_2D_total_fluid_compressibility_grid(
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
    total_compressibility_grid = np.maximum(total_compressibility_grid, 1e-18)

    # Compute relative phase mobilities (kr / mu) (psi⁻¹) for each cell
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_2D_three_phase_relative_mobilities_grids(
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

    # Compute mobility grids for harmonic averaging at interfaces
    # λ_x = k_abs * (kr / mu) / B_x. Where B_x is the formation volume factor of the phase
    water_mobility_grid = (
        absolute_permeability_grid
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid = (
        absolute_permeability_grid
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid = (
        absolute_permeability_grid
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    # Calculate total mobility for diffusion number calculation
    total_mobility_grid = water_mobility_grid + oil_mobility_grid + gas_mobility_grid
    total_mobility_grid[total_mobility_grid == 0] = 1e-12

    # Determine max diffusion number
    max_diffusion_number = 0.0
    min_cell_size = min(cell_size_x, cell_size_y)

    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        cell_porosity = porosity_grid[i, j]
        cell_total_compressibility = total_compressibility_grid[i, j]
        cell_total_mobility = total_mobility_grid[i, j]

        diffusion_number = compute_diffusion_number(
            permeability=absolute_permeability_grid[i, j],
            porosity=cell_porosity,
            mobility=cell_total_mobility,
            total_compressibility=cell_total_compressibility,
            time_step_size=time_step_size,
            cell_size=min_cell_size,
        )
        if diffusion_number > max_diffusion_number:
            max_diffusion_number = diffusion_number

    # Choose solver based on criterion
    if max_diffusion_number > diffusion_number_threshold:
        updated_pressure_grid = compute_implicit_pressure_evolution(
            cell_dimension=cell_dimension,
            height_grid=height_grid,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            wells=wells,
        )
    else:
        updated_pressure_grid = compute_explicit_pressure_evolution(
            cell_dimension=cell_dimension,
            height_grid=height_grid,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            wells=wells,
        )

    return updated_pressure_grid


"""
Explicit finite difference formulation for saturation transport in a 2D reservoir
(immiscible three-phase flow: oil and water, slightly compressible fluids):

The governing equation for saturation evolution is the conservation of mass with advection:

    ∂S/∂t * (φ * AΔx) = (-∇ · (f_x * v)  * AΔx) + (q_x * AΔx)

Where:
    ∂S/∂t * φ * AΔx = Accumulation term (change in phase saturation) (ft³/day)
    ∇ · (f_x * v) * AΔx = Advection term (Darcy velocity multiplied by fractional flow) (ft/day * 1/ft * ft³ = ft³/day)
    q_x * AΔx = Source/sink term for the phase (injection/production) (ft³/day)

If the cell bulk volume AΔx is constant, we can simplify the equation to:

    ∂S/∂t * φ = -∇ · (f_x * v) + q_x

where:
    S = phase saturation (fraction)
    φ = porosity (fraction)
    A = cell face area (ft²)
    Δx = grid spacing in x (ft)
    f_x = phase fractional flow function (depends on S_x)
    v = Darcy velocity vector [v_x, v_y] (ft/day)
    q_x = phase injection/production term (per unit volume) (1/day)

Assumptions:
- Water/oil/gas fractional flow model (Brooks-Corey, Corey, or user-defined f_x(S_x))
- Darcy velocity is known from pressure solution: v = -λ_t * ∇p
- φ, v, and A are constant per cell
- No diffusion or dispersion considered (purely advective)

Explicit update formula:

Discretizing time using Forward Euler:
    ∂S/∂t = (Sⁿ⁺¹_ij - Sⁿ_ij) / Δt

Discretizing space using 1st-order upwind scheme (based on flow direction):

    ∇ · (f_x * v) ≈ [ (f_x * v)_x⁺ - (f_x * v)_x⁻ ] / Δx + [ (f_x * v)_y⁺ - (f_x * v)_y⁻ ] / Δy

Then:

    Sⁿ⁺¹_ij = Sⁿ_ij - Δt / (φ * A∆x) * [ 
        [(f_x_east * v_x_east - f_x_west * v_x_west) * A∆x / Δx] + [(f_y_north * v_y_north - f_y_south * v_y_south) * A∆y / Δy]
        + (q_x_ij * A∆x) 
    ]

Replacing f_x * v_x * A with F_x and similarly for y components, we get:

    Sⁿ⁺¹_ij = Sⁿ_ij - Δt / (φ * A∆x) * [(F_x_east - F_x_west) + (F_y_north - F_y_south) + (q_x_ij * A∆x)]

The ∆x / Δx and ∆y / Δy terms resolve to 1.

Where:
- F_x_east = f_w(S_upwind) * v_x_east * A (ft³/day) (use upstream S based on v_x sign)
- F_x_west = f_w(S_upwind) * v_x_west * A (ft³/day)
- F_y_north = f_w(S_upwind) * v_y_north * A (ft³/day)
- F_y_south = f_w(S_upwind) * v_y_south * A (ft³/day)
- v_x = -λ_x * (∂p/∂x) (ft/day)
- v_y = -λ_y * (∂p/∂y) (ft/day)

For selecting upwind values of S_x , S_upwind is determined based on the flow direction:
    S_upwind = Sⁿ_ij if (v_x_east > 0), flow is from current cell to east neighbor
    S_upwind = Sⁿ⁺¹_ij if (v_x_east < 0), flow is from east neighbor to current cell
    S_upwind = Sⁿ_ij if (v_y_west > 0), flow is from west neighbor to current cell
    S_upwind = Sⁿ⁺¹_ij if (v_y_west < 0), flow is from current cell to west neighbor
    S_upwind = Sⁿ_ij if (v_y_north > 0), flow is from current cell to north neighbor
    S_upwind = Sⁿ⁺¹_ij if (v_y_north < 0), flow is from north neighbor to current cell
    S_upwind = Sⁿ_ij if (v_y_south > 0), flow is from south neighbor to current cell
    S_upwind = Sⁿ⁺¹_ij if (v_y_south < 0), flow is from current cell to south neighbor

Variables:
    Sⁿ_ij = phase saturation at cell (i,j) at time step n (fraction)
    Sⁿ⁺¹_ij = updated phase saturation (fraction)
    φ = porosity (fraction)
    A = cell face area = Δz * Δy or Δz * Δx (ft²), assuming Δz = 1 ft for 2D
    Δx = grid spacing in x (ft)
    Δy = grid spacing in y (ft)
    q_x_ij = phase injection/production term (fractional volume per time)
    f_x = phase upwinded fractional flow (depends on S_x)
    v = Darcy velocity vector [v_x, v_y] (ft/day)
    f_x * v = phase volumetric flux per unit area (ft/day) or phase velocity (ft/day)
    F_x = f_x * v_x * A - phase volumetric flux in x direction (ft³/day)
    F_y = f_x * v_y * A - phase volumetric flux in y direction (ft³/day)
    S_upwind = phase saturation at upstream cell based on flow direction
    v_x_east, v_x_west = x-component of Darcy velocity at east/west neighbors (ft/day)
    v_y_north, v_y_south = y-component of Darcy velocity at north/south neighbors (ft/day)

Stability Condition:
    CFL constraint for explicit advection:

    [ max( |v_x| / Δx + |v_y| / Δy ) * Δt / φ ] <= 1

Notes:
- Use upwind values of S_x based on flow direction.
- Requires pressure solution beforehand to compute velocities.
- No cross-term or capillary term included (extendable if needed).
"""


def compute_saturation_evolution(
    cell_dimension: typing.Tuple[float, float],
    height_grid: TwoDimensionalGrid,
    time_step_size: float,
    boundary_conditions: BoundaryConditions,
    rock_properties: RockProperties,
    fluid_properties: FluidProperties,
    wells: Wells,
) -> typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid, TwoDimensionalGrid]:
    """
    Computes the new/updated saturation distribution for water, oil, and gas
    across the reservoir grid using an explicit upwind finite difference method.

    This function simulates three-phase immiscible flow, considering pressure
    gradients (including capillary pressure effects) and relative permeabilities.

    :param cell_dimension: Tuple representing the dimensions of each grid cell (cell_size_x, cell_size_y) in feet (ft).
    :param height_grid: 2D numpy array representing the height of each cell in the grid (ft).
    :param time_step_size: Time step duration in seconds for the simulation.
    :param boundary_conditions: Boundary conditions for pressure and saturation grids.
    :param rock_properties: `RockProperties` object containing rock physical properties.
    :param fluid_properties: `FluidProperties` object containing fluid physical properties,
        including current pressure and saturation grids.

    :param wells: `Wells` object containing information about injection and production wells.
    :return: A tuple of 2D numpy arrays representing the updated saturation distributions
        for water, oil, and gas, respectively.
        (updated_water_saturation_grid, updated_oil_saturation_grid, updated_gas_saturation_grid)
    """
    cell_count_x, cell_count_y = fluid_properties.pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension
    time_step_size_in_days = time_step_size * SECONDS_TO_DAYS

    # --- Extract current state and properties ---
    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    absolute_permeability_grid = rock_properties.absolute_permeability_grid
    porosity_grid = rock_properties.porosity_grid
    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_grid = rock_properties.residual_oil_saturation_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid
    relative_permeability_params = rock_properties.relative_permeability_params
    capillary_pressure_params = rock_properties.capillary_pressure_params

    water_formation_volume_factor_grid = (
        fluid_properties.water_formation_volume_factor_grid
    )
    oil_formation_volume_factor_grid = fluid_properties.oil_formation_volume_factor_grid
    gas_formation_volume_factor_grid = fluid_properties.gas_formation_volume_factor_grid

    water_viscosity_grid = fluid_properties.water_viscosity_grid
    oil_viscosity_grid = fluid_properties.oil_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    # Padding grids to handle boundary conditions and neighbor access
    padded_pressure_grid = edge_pad_grid(current_oil_pressure_grid)
    padded_water_saturation_grid = edge_pad_grid(current_water_saturation_grid)
    padded_oil_saturation_grid = edge_pad_grid(current_oil_saturation_grid)
    padded_gas_saturation_grid = edge_pad_grid(current_gas_saturation_grid)

    # Apply boundary conditions to padded grids
    boundary_conditions["pressure"].apply(padded_pressure_grid)
    boundary_conditions["water_saturation"].apply(padded_water_saturation_grid)
    boundary_conditions["oil_saturation"].apply(padded_oil_saturation_grid)
    boundary_conditions["gas_saturation"].apply(padded_gas_saturation_grid)

    # Compute phase mobilities (kr / mu) for each cell based on old saturations
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_2D_three_phase_relative_mobilities_grids(
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

    # Compute mobility grids for harmonic averaging at interfaces
    # λ_x = k_abs * (kr / mu) / B_x. Where B_x is the formation volume factor of the phase
    water_mobility_grid = (
        absolute_permeability_grid
        * water_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / water_formation_volume_factor_grid
    oil_mobility_grid = (
        absolute_permeability_grid
        * oil_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / oil_formation_volume_factor_grid
    gas_mobility_grid = (
        absolute_permeability_grid
        * gas_relative_mobility_grid
        * MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
    ) / gas_formation_volume_factor_grid

    # Compute total transmissibility for each cell
    total_mobility_grid = water_mobility_grid + oil_mobility_grid + gas_mobility_grid
    total_mobility_grid[total_mobility_grid == 0] = 1e-12

    padded_water_mobility_grid = edge_pad_grid(water_mobility_grid)
    padded_oil_mobility_grid = edge_pad_grid(oil_mobility_grid)
    padded_gas_mobility_grid = edge_pad_grid(gas_mobility_grid)

    # Capillary Pressures for each cell
    (oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid) = (
        build_2D_three_phase_capillary_pressure_grids(
            water_saturation_grid=current_water_saturation_grid,
            gas_saturation_grid=current_gas_saturation_grid,
            irreducible_water_saturation_grid=irreducible_water_saturation_grid,
            residual_oil_saturation_grid=residual_oil_saturation_grid,
            residual_gas_saturation_grid=residual_gas_saturation_grid,
            capillary_pressure_params=capillary_pressure_params,
        )
    )

    padded_oil_water_capillary_pressure_grid = edge_pad_grid(
        oil_water_capillary_pressure_grid
    )
    padded_gas_oil_capillary_pressure_grid = edge_pad_grid(
        gas_oil_capillary_pressure_grid
    )

    # Create new grids for updated saturations (time 'n+1')
    updated_water_saturation_grid = current_water_saturation_grid.copy()
    updated_oil_saturation_grid = current_oil_saturation_grid.copy()
    updated_gas_saturation_grid = current_gas_saturation_grid.copy()

    # Define Corey's exponents
    water_corey_exponent = relative_permeability_params.water_exponent
    oil_corey_exponent = relative_permeability_params.oil_exponent
    gas_corey_exponent = relative_permeability_params.gas_exponent

    # Iterate over each interior cell to compute saturation evolution
    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        ip, jp = i + 1, j + 1  # Indices of cell in padded grid

        current_cell_depth = height_grid[i, j]
        cell_total_volume = cell_size_x * cell_size_y * current_cell_depth
        # Current cell properties
        cell_porosity = porosity_grid[i, j]
        # Cell pore volume = φ * A∆x
        cell_pore_volume = cell_total_volume * cell_porosity

        # Current cell phase pressures (P_oil is direct, P_water and P_gas derived)
        current_cell_oil_pressure = current_oil_pressure_grid[i, j]
        current_cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[
            i, j
        ]
        current_cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[i, j]

        # Current cell saturations
        current_cell_water_saturation = current_water_saturation_grid[i, j]
        current_cell_oil_saturation = current_oil_saturation_grid[i, j]
        current_cell_gas_saturation = current_gas_saturation_grid[i, j]

        # For the east neighbour
        east_neighbor_ip, east_neighbor_jp = ip + 1, jp
        east_neighbor_oil_pressure = padded_pressure_grid[
            east_neighbor_ip, east_neighbor_jp
        ]
        east_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[east_neighbor_ip, east_neighbor_jp]
        )
        east_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[east_neighbor_ip, east_neighbor_jp]
        )
        east_neighbor_water_saturation = padded_water_saturation_grid[
            east_neighbor_ip, east_neighbor_jp
        ]
        east_neighbor_oil_saturation = padded_oil_saturation_grid[
            east_neighbor_ip, east_neighbor_jp
        ]
        east_neighbor_gas_saturation = padded_gas_saturation_grid[
            east_neighbor_ip, east_neighbor_jp
        ]

        # Compute pressure differences for capillary pressure terms
        oil_pressure_difference_east = (
            east_neighbor_oil_pressure - current_cell_oil_pressure
        )
        average_pressure_east = (
            current_cell_oil_pressure + east_neighbor_oil_pressure
        ) / 2.0
        oil_water_capillary_pressure_difference_east = (
            east_neighbor_oil_water_capillary_pressure
            - current_cell_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_east = (
            east_neighbor_gas_oil_capillary_pressure
            - current_cell_gas_oil_capillary_pressure
        )

        # Compute harmonic mobility for the east neighbor
        water_harmonic_mobility_east = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=east_neighbor_ip,
            j2=east_neighbor_jp,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_east = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=east_neighbor_ip,
            j2=east_neighbor_jp,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_east = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=east_neighbor_ip,
            j2=east_neighbor_jp,
            mobility_grid=padded_gas_mobility_grid,
        )

        # Computing the Darcy velocities (ft/day) for the three phases
        # v_x = -λ_x * ∆P / Δx
        # For water: v_w = -λ_w * (P_oil - P_cow) / Δx
        water_pressure_difference_east = (
            oil_pressure_difference_east - oil_water_capillary_pressure_difference_east
        )
        water_velocity_east = (
            -water_harmonic_mobility_east * water_pressure_difference_east / cell_size_x
        )

        # For oil: v_o = -λ_o * (P_oil) / Δx
        oil_velocity_east = (
            -oil_harmonic_mobility_east * oil_pressure_difference_east / cell_size_x
        )

        # For gas: v_g = -λ_g * 2 * P_avg * (P_oil - Pcgo) / Δx
        gas_pressure_difference_east = (
            oil_pressure_difference_east - gas_oil_capillary_pressure_difference_east
        )
        gas_velocity_east = (
            -gas_harmonic_mobility_east
            * 2
            * average_pressure_east
            * gas_pressure_difference_east
            / cell_size_y
        )

        # Select upwind saturations
        upwinded_water_saturation_east = (
            current_cell_water_saturation
            if water_velocity_east >= 0
            else east_neighbor_water_saturation
        )
        upwinded_oil_saturation_east = (
            current_cell_oil_saturation
            if oil_velocity_east >= 0
            else east_neighbor_oil_saturation
        )
        upwinded_gas_saturation_east = (
            current_cell_gas_saturation
            if gas_velocity_east >= 0
            else east_neighbor_gas_saturation
        )

        # Compute the total fractional flow
        (
            water_upwinded_mobility_east,
            oil_upwinded_mobility_east,
            gas_upwinded_mobility_east,
        ) = compute_three_phase_relative_permeabilities(
            water_saturation=upwinded_water_saturation_east,
            oil_saturation=upwinded_oil_saturation_east,
            gas_saturation=upwinded_gas_saturation_east,
            irreducible_water_saturation=irreducible_water_saturation_grid[i, j],
            residual_oil_saturation=residual_oil_saturation_grid[i, j],
            residual_gas_saturation=residual_gas_saturation_grid[i, j],
            water_exponent=water_corey_exponent,
            oil_exponent=oil_corey_exponent,
            gas_exponent=gas_corey_exponent,
        )
        # f_x = λ_x / (λ_w + λ_o + λ_g)
        total_upwinded_mobility_east = (
            water_upwinded_mobility_east
            + oil_upwinded_mobility_east
            + gas_upwinded_mobility_east
        )
        total_upwinded_mobility_east = np.maximum(
            total_upwinded_mobility_east, 1e-12
        )  # Avoid division by zero
        # For water: f_w = λ_w / (λ_w + λ_o + λ_g)
        water_fractional_flow_east = (
            water_upwinded_mobility_east / total_upwinded_mobility_east
        )
        # For oil: f_o = λ_o / (λ_w + λ_o + λ_g)
        oil_fractional_flow_east = (
            oil_upwinded_mobility_east / total_upwinded_mobility_east
        )
        # For gas: f_g = λ_g / (λ_w + λ_o + λ_g)
        gas_fractional_flow_east = (
            gas_upwinded_mobility_east / total_upwinded_mobility_east
        )

        # Compute volumetric fluxes from the east neighbor for each phase
        # F_x_east = f_w * v_x * A
        east_face_area = cell_size_y * current_cell_depth  # Area of the face (ft²)

        # For water: F_w_east = f_w * v_w * A
        water_volumetric_flux_at_east_face = (
            water_fractional_flow_east * water_velocity_east * east_face_area
        )
        # For oil: F_o_east = f_o * v_o * A
        oil_volumetric_flux_at_east_face = (
            oil_fractional_flow_east * oil_velocity_east * east_face_area
        )
        # For gas: F_g_east = f_w * v_g * A
        gas_volumetric_flux_at_east_face = (
            gas_fractional_flow_east * gas_velocity_east * east_face_area
        )

        # Compute the component fluxes for the east face
        water_volumetric_flux_from_east = (
            water_volumetric_flux_at_east_face * upwinded_water_saturation_east
        )
        oil_volumetric_flux_from_east = (
            oil_volumetric_flux_at_east_face * upwinded_oil_saturation_east
        )
        gas_volumetric_flux_from_east = (
            gas_volumetric_flux_at_east_face * upwinded_gas_saturation_east
        )

        # For the west neighbour
        west_neighbor_ip, west_neighbor_jp = ip - 1, jp
        west_neighbor_oil_pressure = padded_pressure_grid[
            west_neighbor_ip, west_neighbor_jp
        ]
        west_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[west_neighbor_ip, west_neighbor_jp]
        )
        west_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[west_neighbor_ip, west_neighbor_jp]
        )
        west_neighbor_water_saturation = padded_water_saturation_grid[
            west_neighbor_ip, west_neighbor_jp
        ]
        west_neighbor_oil_saturation = padded_oil_saturation_grid[
            west_neighbor_ip, west_neighbor_jp
        ]
        west_neighbor_gas_saturation = padded_gas_saturation_grid[
            west_neighbor_ip, west_neighbor_jp
        ]

        # Compute pressure differences for capillary pressure terms
        oil_pressure_difference_west = (
            west_neighbor_oil_pressure - current_cell_oil_pressure
        )
        average_pressure_west = (
            current_cell_oil_pressure + west_neighbor_oil_pressure
        ) / 2.0
        oil_water_capillary_pressure_difference_west = (
            west_neighbor_oil_water_capillary_pressure
            - current_cell_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_west = (
            west_neighbor_gas_oil_capillary_pressure
            - current_cell_gas_oil_capillary_pressure
        )

        # Compute harmonic mobility for the west neighbor
        water_harmonic_mobility_west = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=west_neighbor_ip,
            j2=west_neighbor_jp,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_west = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=west_neighbor_ip,
            j2=west_neighbor_jp,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_west = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=west_neighbor_ip,
            j2=west_neighbor_jp,
            mobility_grid=padded_gas_mobility_grid,
        )
        # Computing the Darcy velocities (ft/day) for the three phases
        # v_x = -λ_x * ∆P / Δx
        # For water: v_w = -λ_w * (P_oil - P_cow) / Δx
        water_pressure_difference_west = (
            oil_pressure_difference_west - oil_water_capillary_pressure_difference_west
        )
        water_velocity_west = (
            -water_harmonic_mobility_west * water_pressure_difference_west / cell_size_x
        )
        # For oil: v_o = -λ_o * (P_oil) / Δx
        oil_velocity_west = (
            -oil_harmonic_mobility_west * oil_pressure_difference_west / cell_size_x
        )
        # For gas: v_g = -λ_g * 2 * P_avg * (P_oil - Pcgo) / Δx
        gas_pressure_difference_west = (
            oil_pressure_difference_west - gas_oil_capillary_pressure_difference_west
        )
        gas_velocity_west = (
            -gas_harmonic_mobility_west
            * 2
            * average_pressure_west
            * gas_pressure_difference_west
            / cell_size_y
        )

        # Select upwind saturations
        upwinded_water_saturation_west = (
            west_neighbor_water_saturation
            if water_velocity_west >= 0
            else current_cell_water_saturation
        )
        upwinded_oil_saturation_west = (
            west_neighbor_oil_saturation
            if oil_velocity_west >= 0
            else current_cell_oil_saturation
        )
        upwinded_gas_saturation_west = (
            west_neighbor_gas_saturation
            if gas_velocity_west >= 0
            else current_cell_gas_saturation
        )

        # Compute the total fractional flow
        (
            water_upwinded_mobility_west,
            oil_upwinded_mobility_west,
            gas_upwinded_mobility_west,
        ) = compute_three_phase_relative_permeabilities(
            water_saturation=upwinded_water_saturation_west,
            oil_saturation=upwinded_oil_saturation_west,
            gas_saturation=upwinded_gas_saturation_west,
            irreducible_water_saturation=irreducible_water_saturation_grid[i, j],
            residual_oil_saturation=residual_oil_saturation_grid[i, j],
            residual_gas_saturation=residual_gas_saturation_grid[i, j],
            water_exponent=water_corey_exponent,
            oil_exponent=oil_corey_exponent,
            gas_exponent=gas_corey_exponent,
        )
        # f_x = λ_x / (λ_w + λ_o + λ_g)
        total_upwinded_mobility_west = (
            water_upwinded_mobility_west
            + oil_upwinded_mobility_west
            + gas_upwinded_mobility_west
        )
        total_upwinded_mobility_west = np.maximum(
            total_upwinded_mobility_west, 1e-12
        )  # Avoid division by zero
        # For water: f_w = λ_w / (λ_w + λ_o + λ_g)
        water_fractional_flow_west = (
            water_upwinded_mobility_west / total_upwinded_mobility_west
        )
        # For oil: f_o = λ_o / (λ_w + λ_o + λ_g)
        oil_fractional_flow_west = (
            oil_upwinded_mobility_west / total_upwinded_mobility_west
        )
        # For gas: f_g = λ_g / (λ_w + λ_o + λ_g)
        gas_fractional_flow_west = (
            gas_upwinded_mobility_west / total_upwinded_mobility_west
        )

        # Compute volumetric fluxes from the west neighbor for each phase
        # F_x_west = f_w * v_x * A
        west_face_area = cell_size_y * current_cell_depth  # Area of the face (ft²)

        # For water: F_w_west = f_w * v_w * A
        water_volumetric_flux_at_west_face = (
            water_fractional_flow_west * water_velocity_west * west_face_area
        )
        # For oil: F_o_west = f_o * v_o * A
        oil_volumetric_flux_at_west_face = (
            oil_fractional_flow_west * oil_velocity_west * west_face_area
        )
        # For gas: F_g_west = f_w * v_g * A
        gas_volumetric_flux_at_west_face = (
            gas_fractional_flow_west * gas_velocity_west * west_face_area
        )

        # Compute the component fluxes for the west face
        water_volumetric_flux_from_west = (
            water_volumetric_flux_at_west_face * upwinded_water_saturation_west
        )
        oil_volumetric_flux_from_west = (
            oil_volumetric_flux_at_west_face * upwinded_oil_saturation_west
        )
        gas_volumetric_flux_from_west = (
            gas_volumetric_flux_at_west_face * upwinded_gas_saturation_west
        )

        # For the north neighbour
        north_neighbor_ip, north_neighbor_jp = ip, jp + 1
        north_neighbor_oil_pressure = padded_pressure_grid[
            north_neighbor_ip, north_neighbor_jp
        ]
        north_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                north_neighbor_ip, north_neighbor_jp
            ]
        )
        north_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[north_neighbor_ip, north_neighbor_jp]
        )
        north_neighbor_water_saturation = padded_water_saturation_grid[
            north_neighbor_ip, north_neighbor_jp
        ]
        north_neighbor_oil_saturation = padded_oil_saturation_grid[
            north_neighbor_ip, north_neighbor_jp
        ]
        north_neighbor_gas_saturation = padded_gas_saturation_grid[
            north_neighbor_ip, north_neighbor_jp
        ]
        # Compute pressure differences for capillary pressure terms
        oil_pressure_difference_north = (
            north_neighbor_oil_pressure - current_cell_oil_pressure
        )
        average_pressure_north = (
            current_cell_oil_pressure + north_neighbor_oil_pressure
        ) / 2.0
        oil_water_capillary_pressure_difference_north = (
            north_neighbor_oil_water_capillary_pressure
            - current_cell_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_north = (
            north_neighbor_gas_oil_capillary_pressure
            - current_cell_gas_oil_capillary_pressure
        )

        # Compute harmonic mobility for the north neighbor
        water_harmonic_mobility_north = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=north_neighbor_ip,
            j2=north_neighbor_jp,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_north = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=north_neighbor_ip,
            j2=north_neighbor_jp,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_north = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=north_neighbor_ip,
            j2=north_neighbor_jp,
            mobility_grid=padded_gas_mobility_grid,
        )
        # Computing the Darcy velocities (ft/day) for the three phases
        # v_y = -λ_y * ∆P / Δy
        # For water: v_w = -λ_w * (P_oil - P_cow) / Δy
        water_pressure_difference_north = (
            oil_pressure_difference_north
            - oil_water_capillary_pressure_difference_north
        )
        water_velocity_north = (
            -water_harmonic_mobility_north
            * water_pressure_difference_north
            / cell_size_y
        )
        # For oil: v_o = -λ_o * (P_oil) / Δy
        oil_velocity_north = (
            -oil_harmonic_mobility_north * oil_pressure_difference_north / cell_size_y
        )
        # For gas: v_g = -λ_g * 2 * P_avg * (P_oil - Pcgo) / Δy
        gas_pressure_difference_north = (
            oil_pressure_difference_north - gas_oil_capillary_pressure_difference_north
        )
        gas_velocity_north = (
            -gas_harmonic_mobility_north
            * 2
            * average_pressure_north
            * gas_pressure_difference_north
            / cell_size_y
        )

        # Select upwind saturations
        upwinded_water_saturation_north = (
            current_cell_water_saturation
            if water_velocity_north >= 0
            else north_neighbor_water_saturation
        )
        upwinded_oil_saturation_north = (
            current_cell_oil_saturation
            if oil_velocity_north >= 0
            else north_neighbor_oil_saturation
        )
        upwinded_gas_saturation_north = (
            current_cell_gas_saturation
            if gas_velocity_north >= 0
            else north_neighbor_gas_saturation
        )

        # Compute the total fractional flow
        (
            water_upwinded_mobility_north,
            oil_upwinded_mobility_north,
            gas_upwinded_mobility_north,
        ) = compute_three_phase_relative_permeabilities(
            water_saturation=upwinded_water_saturation_north,
            oil_saturation=upwinded_oil_saturation_north,
            gas_saturation=upwinded_gas_saturation_north,
            irreducible_water_saturation=irreducible_water_saturation_grid[i, j],
            residual_oil_saturation=residual_oil_saturation_grid[i, j],
            residual_gas_saturation=residual_gas_saturation_grid[i, j],
            water_exponent=water_corey_exponent,
            oil_exponent=oil_corey_exponent,
            gas_exponent=gas_corey_exponent,
        )
        # f_x = λ_x / (λ_w + λ_o + λ_g)
        total_upwinded_mobility_north = (
            water_upwinded_mobility_north
            + oil_upwinded_mobility_north
            + gas_upwinded_mobility_north
        )
        total_upwinded_mobility_north = np.maximum(
            total_upwinded_mobility_north, 1e-12
        )  # Avoid division by zero
        # For water: f_w = λ_w / (λ_w + λ_o + λ_g)
        water_fractional_flow_north = (
            water_upwinded_mobility_north / total_upwinded_mobility_north
        )
        # For oil: f_o = λ_o / (λ_w + λ_o + λ_g)
        oil_fractional_flow_north = (
            oil_upwinded_mobility_north / total_upwinded_mobility_north
        )
        # For gas: f_g = λ_g / (λ_w + λ_o + λ_g)
        gas_fractional_flow_north = (
            gas_upwinded_mobility_north / total_upwinded_mobility_north
        )

        # Compute volumetric fluxes from the north neighbor for each phase
        # F_y_north = f_w * v_y * A
        north_face_area = cell_size_x * current_cell_depth
        # For water: F_w_north = f_w * v_w * A
        water_volumetric_flux_at_north_face = (
            water_fractional_flow_north * water_velocity_north * north_face_area
        )
        # For oil: F_o_north = f_o * v_o * A
        oil_volumetric_flux_at_north_face = (
            oil_fractional_flow_north * oil_velocity_north * north_face_area
        )
        # For gas: F_g_north = f_w * v_g * A
        gas_volumetric_flux_at_north_face = (
            gas_fractional_flow_north * gas_velocity_north * north_face_area
        )

        # Compute the component fluxes for the north face
        water_volumetric_flux_from_north = (
            water_volumetric_flux_at_north_face * upwinded_water_saturation_north
        )
        oil_volumetric_flux_from_north = (
            oil_volumetric_flux_at_north_face * upwinded_oil_saturation_north
        )
        gas_volumetric_flux_from_north = (
            gas_volumetric_flux_at_north_face * upwinded_gas_saturation_north
        )

        # For the south neighbour
        south_neighbor_ip, south_neighbor_jp = ip, jp - 1
        south_neighbor_oil_pressure = padded_pressure_grid[
            south_neighbor_ip, south_neighbor_jp
        ]
        south_neighbor_oil_water_capillary_pressure = (
            padded_oil_water_capillary_pressure_grid[
                south_neighbor_ip, south_neighbor_jp
            ]
        )
        south_neighbor_gas_oil_capillary_pressure = (
            padded_gas_oil_capillary_pressure_grid[south_neighbor_ip, south_neighbor_jp]
        )
        south_neighbor_water_saturation = padded_water_saturation_grid[
            south_neighbor_ip, south_neighbor_jp
        ]
        south_neighbor_oil_saturation = padded_oil_saturation_grid[
            south_neighbor_ip, south_neighbor_jp
        ]
        south_neighbor_gas_saturation = padded_gas_saturation_grid[
            south_neighbor_ip, south_neighbor_jp
        ]
        # Compute pressure differences for capillary pressure terms
        oil_pressure_difference_south = (
            south_neighbor_oil_pressure - current_cell_oil_pressure
        )
        average_pressure_south = (
            current_cell_oil_pressure + south_neighbor_oil_pressure
        ) / 2.0
        oil_water_capillary_pressure_difference_south = (
            south_neighbor_oil_water_capillary_pressure
            - current_cell_oil_water_capillary_pressure
        )
        gas_oil_capillary_pressure_difference_south = (
            south_neighbor_gas_oil_capillary_pressure
            - current_cell_gas_oil_capillary_pressure
        )
        # Compute harmonic mobility for the south neighbor
        water_harmonic_mobility_south = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=south_neighbor_ip,
            j2=south_neighbor_jp,
            mobility_grid=padded_water_mobility_grid,
        )
        oil_harmonic_mobility_south = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=south_neighbor_ip,
            j2=south_neighbor_jp,
            mobility_grid=padded_oil_mobility_grid,
        )
        gas_harmonic_mobility_south = compute_harmonic_mobility(
            i1=ip,
            j1=jp,
            i2=south_neighbor_ip,
            j2=south_neighbor_jp,
            mobility_grid=padded_gas_mobility_grid,
        )
        # Computing the Darcy velocities (ft/day) for the three phases
        # v_y = -λ_y * ∆P / Δy
        # For water: v_w = -λ_w * (P_oil - P_cow) / Δy
        water_pressure_difference_south = (
            oil_pressure_difference_south
            - oil_water_capillary_pressure_difference_south
        )
        water_velocity_south = (
            -water_harmonic_mobility_south
            * water_pressure_difference_south
            / cell_size_y
        )
        # For oil: v_o = -λ_o * (P_oil) / Δy
        oil_velocity_south = (
            -oil_harmonic_mobility_south * oil_pressure_difference_south / cell_size_y
        )
        # For gas: v_g = -λ_g * 2 * P_avg * (P_oil - Pcgo) / Δy
        gas_pressure_difference_south = (
            oil_pressure_difference_south - gas_oil_capillary_pressure_difference_south
        )
        gas_velocity_south = (
            -gas_harmonic_mobility_south
            * 2
            * average_pressure_south
            * gas_pressure_difference_south
            / cell_size_y
        )

        # Select upwind saturations
        upwinded_water_saturation_south = (
            south_neighbor_water_saturation
            if water_velocity_south >= 0
            else current_cell_water_saturation
        )
        upwinded_oil_saturation_south = (
            south_neighbor_oil_saturation
            if oil_velocity_south >= 0
            else current_cell_oil_saturation
        )
        upwinded_gas_saturation_south = (
            south_neighbor_gas_saturation
            if gas_velocity_south >= 0
            else current_cell_gas_saturation
        )

        # Compute the total fractional flow
        (
            water_upwinded_mobility_south,
            oil_upwinded_mobility_south,
            gas_upwinded_mobility_south,
        ) = compute_three_phase_relative_permeabilities(
            water_saturation=upwinded_water_saturation_south,
            oil_saturation=upwinded_oil_saturation_south,
            gas_saturation=upwinded_gas_saturation_south,
            irreducible_water_saturation=irreducible_water_saturation_grid[i, j],
            residual_oil_saturation=residual_oil_saturation_grid[i, j],
            residual_gas_saturation=residual_gas_saturation_grid[i, j],
            water_exponent=water_corey_exponent,
            oil_exponent=oil_corey_exponent,
            gas_exponent=gas_corey_exponent,
        )
        # f_x = λ_x / (λ_w + λ_o + λ_g)
        total_upwinded_mobility_south = (
            water_upwinded_mobility_south
            + oil_upwinded_mobility_south
            + gas_upwinded_mobility_south
        )
        total_upwinded_mobility_south = np.maximum(
            total_upwinded_mobility_south, 1e-12
        )  # Avoid division by zero
        # For water: f_w = λ_w / (λ_w + λ_o + λ_g)
        water_fractional_flow_south = (
            water_upwinded_mobility_south / total_upwinded_mobility_south
        )
        # For oil: f_o = λ_o / (λ_w + λ_o + λ_g)
        oil_fractional_flow_south = (
            oil_upwinded_mobility_south / total_upwinded_mobility_south
        )
        # For gas: f_g = λ_g / (λ_w + λ_o + λ_g)
        gas_fractional_flow_south = (
            gas_upwinded_mobility_south / total_upwinded_mobility_south
        )

        # Compute volumetric fluxes from the south neighbor for each phase
        # F_y_south = f_w * v_y * A
        south_face_area = cell_size_x * current_cell_depth
        # For water: F_w_south = f_w * v_w * A
        water_volumetric_flux_at_south_face = (
            water_fractional_flow_south * water_velocity_south * south_face_area
        )
        # For oil: F_o_south = f_o * v_o * A
        oil_volumetric_flux_at_south_face = (
            oil_fractional_flow_south * oil_velocity_south * south_face_area
        )
        # For gas: F_g_south = f_w * v_g * A
        gas_volumetric_flux_at_south_face = (
            gas_fractional_flow_south * gas_velocity_south * south_face_area
        )

        # Compute the component fluxes for the south face
        water_volumetric_flux_from_south = (
            water_volumetric_flux_at_south_face * upwinded_water_saturation_south
        )
        oil_volumetric_flux_from_south = (
            oil_volumetric_flux_at_south_face * upwinded_oil_saturation_south
        )
        gas_volumetric_flux_from_south = (
            gas_volumetric_flux_at_south_face * upwinded_gas_saturation_south
        )

        # Compute net volumetric fluxes from neighbours for each phase for the current cell
        net_water_flux_from_neighbors = (
            water_volumetric_flux_from_east - water_volumetric_flux_from_west
        ) + (water_volumetric_flux_from_north - water_volumetric_flux_from_south)
        net_oil_flux_from_neighbors = (
            oil_volumetric_flux_from_east - oil_volumetric_flux_from_west
        ) + (oil_volumetric_flux_from_north - oil_volumetric_flux_from_south)
        net_gas_flux_from_neighbors = (
            gas_volumetric_flux_from_east - gas_volumetric_flux_from_west
        ) + (gas_volumetric_flux_from_north - gas_volumetric_flux_from_south)

        # Max face flux for water (absolute values from all directions)
        max_water_flux = max(
            abs(water_volumetric_flux_from_east),
            abs(water_volumetric_flux_from_west),
            abs(water_volumetric_flux_from_north),
            abs(water_volumetric_flux_from_south),
        )
        # CFL condition for explicit saturation update
        # CFL condition: (Δt * max_flux) / (φ * A∆x) <= 1
        if (time_step_size_in_days * max_water_flux) / (
            cell_porosity * cell_pore_volume
        ) > 1.0:
            raise RuntimeError(f"CFL violation in cell ({i},{j}) for water phase")

        # Compute Source/Sink Term for each phase (WellParameters) - q * A∆x (ft³/day)
        injection_well, production_well = wells[i, j]
        cell_water_injection_rate = cell_water_production_rate = 0.0
        cell_oil_injection_rate = cell_oil_production_rate = 0.0
        cell_gas_injection_rate = cell_gas_production_rate = 0.0

        if injection_well is not None:
            # If there is an injection well, add its flow rate to the cell
            cell_injection_rate = (
                injection_well.injected_fluid.volumetric_flow_rate
            )  # STB/day or SCF/day
            injected_phase = injection_well.injected_fluid.phase
            if injected_phase == FluidPhase.GAS:
                # Get the volumetric flow rate in ft³/day
                cell_injection_rate *= (
                    fluid_properties.gas_formation_volume_factor_grid[i, j]
                )  # ft³/SCF
                cell_gas_injection_rate = cell_injection_rate
            elif injected_phase == FluidPhase.WATER:
                # For water, convert bbl/day to ft³/day
                cell_injection_rate *= (
                    fluid_properties.water_formation_volume_factor_grid[i, j]  # bbl/STB
                    * BBL_TO_FT3  # Convert bbl/day to ft³/day
                )
                cell_water_injection_rate = cell_injection_rate
            else:
                # For oil and water, convert bbl/day to ft³/day
                cell_injection_rate *= (
                    fluid_properties.oil_formation_volume_factor_grid[i, j]  # bbl/STB
                    * BBL_TO_FT3  # Convert bbl/day to ft³/day
                )
                cell_oil_injection_rate = cell_injection_rate

        if production_well is not None:
            # If there is a production well, subtract its flow rate from the cell
            for produced_fluid in production_well.produced_fluids:
                production_rate = (
                    produced_fluid.volumetric_flow_rate
                )  # STB/day or SCF/day
                if produced_fluid.phase == FluidPhase.GAS:
                    # Get the volumetric flow rate in ft³/day
                    production_rate *= (
                        fluid_properties.gas_formation_volume_factor_grid[i, j]
                    )
                    cell_gas_production_rate += production_rate
                elif produced_fluid.phase == FluidPhase.WATER:
                    # For water, convert bbl/day to ft³/day
                    production_rate *= (
                        fluid_properties.water_formation_volume_factor_grid[
                            i, j
                        ]  # bbl/STB
                        * BBL_TO_FT3  # Convert bbl/day to ft³/day
                    )
                    cell_water_production_rate += production_rate
                else:
                    # For oil, convert bbl/day to ft³/day
                    production_rate *= (
                        fluid_properties.oil_formation_volume_factor_grid[
                            i, j
                        ]  # bbl/STB
                        * BBL_TO_FT3  # Convert bbl/day to ft³/day
                    )
                    cell_oil_production_rate += production_rate

        # Calculate the net well flow rate into the cell
        # q_{i,j} * A_{i,j} * Δx (ft³/day) = # (q_{i,j}_injection - q_{i,j}_production)
        net_water_flow_rate_into_cell = (
            cell_water_injection_rate - cell_water_production_rate
        )
        net_oil_flow_rate_into_cell = cell_oil_injection_rate - cell_oil_production_rate
        net_gas_flow_rate_into_cell = cell_gas_injection_rate - cell_gas_production_rate

        # Calculate saturation deltas for each phase
        # dS = Δt / (φ * A∆x) * [
        #     ([F_x_east - F_x_west] * Δx / Δx) + ([F_y_north - F_y_south] * Δx / Δy)
        #     + (q_x_ij * A∆x)
        # ]
        if (
            cell_pore_volume > 1e-12
        ):  # Avoid division by zero for cells with no significant pore volume
            # The change in saturation is (Net_Flux + Net_Well_Rate) * dt / Pore_Volume
            water_saturation_delta = (
                (net_water_flux_from_neighbors + net_water_flow_rate_into_cell)
                * time_step_size_in_days
                / cell_pore_volume
            )
            oil_saturation_delta = (
                (net_oil_flux_from_neighbors + net_oil_flow_rate_into_cell)
                * time_step_size_in_days
                / cell_pore_volume
            )
            gas_saturation_delta = (
                (net_gas_flux_from_neighbors + net_gas_flow_rate_into_cell)
                * time_step_size_in_days
                / cell_pore_volume
            )
        else:
            # If no pore volume, saturations remain unchanged (or set to 0 if preferred for initial state)
            water_saturation_delta = 0.0
            oil_saturation_delta = 0.0
            gas_saturation_delta = 0.0

        # Update phase saturations
        updated_water_saturation_grid[i, j] = (
            current_cell_water_saturation + water_saturation_delta
        )
        updated_oil_saturation_grid[i, j] = (
            current_cell_oil_saturation + oil_saturation_delta
        )
        updated_gas_saturation_grid[i, j] = (
            current_cell_gas_saturation + gas_saturation_delta
        )

    # Apply Saturation Constraints and Normalization across all cells
    # This loop runs *after* all cells have been updated in the previous loop.
    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        # 1. Clip saturations to ensure they remain physically meaningful [0.0, 1.0]
        updated_water_saturation_grid[i, j] = np.clip(
            updated_water_saturation_grid[i, j], 0.0, 1.0
        )
        updated_oil_saturation_grid[i, j] = np.clip(
            updated_oil_saturation_grid[i, j], 0.0, 1.0
        )
        updated_gas_saturation_grid[i, j] = np.clip(
            updated_gas_saturation_grid[i, j], 0.0, 1.0
        )

        # 2. Normalize saturations to ensure their sum is 1.0
        total_saturation = (
            updated_water_saturation_grid[i, j]
            + updated_oil_saturation_grid[i, j]
            + updated_gas_saturation_grid[i, j]
        )

        # Avoid division by zero if total_saturation is extremely small (e.g., in a void)
        if total_saturation > 1e-9:
            updated_water_saturation_grid[i, j] /= total_saturation
            updated_oil_saturation_grid[i, j] /= total_saturation
            updated_gas_saturation_grid[i, j] /= total_saturation
        else:
            # If total saturation is practically zero, set all to zero to maintain consistency.
            # This might happen in cells with extremely low porosity or where all fluids have been depleted.
            updated_water_saturation_grid[i, j] = 0.0
            updated_oil_saturation_grid[i, j] = 0.0
            updated_gas_saturation_grid[i, j] = 0.0

    return (
        updated_water_saturation_grid,
        updated_oil_saturation_grid,
        updated_gas_saturation_grid,
    )
