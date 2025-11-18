import functools
import itertools
import typing

import numpy as np
import pyamg
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import gmres

from sim3D._precision import get_dtype, get_floating_point_info
from sim3D.constants import c
from sim3D.diffusivity.base import (
    EvolutionResult,
    _warn_injector_is_producing,
    _warn_producer_is_injecting,
)
from sim3D.grids.pvt import build_total_fluid_compressibility_grid
from sim3D.models import FluidProperties, RockFluidProperties, RockProperties
from sim3D.pvt import compute_harmonic_mean, compute_harmonic_mobility
from sim3D.types import (
    CapillaryPressureGrids,
    FluidPhase,
    Options,
    RelativeMobilityGrids,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from sim3D.wells import Wells

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
    cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[cell_indices]
    neighbour_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[
        neighbour_indices
    ]
    neighbour_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[
        neighbour_indices
    ]

    # Calculate pressure differences relative to current cell (Neighbour - Current)
    # These represent the gradients driving flow from neighbour to current cell, or vice versa
    oil_water_capillary_pressure_difference = (
        neighbour_oil_water_capillary_pressure - cell_oil_water_capillary_pressure
    )
    gas_oil_capillary_pressure_difference = (
        neighbour_gas_oil_capillary_pressure - cell_gas_oil_capillary_pressure
    )

    if elevation_grid is not None:
        # Calculate the elevation difference between the neighbour and current cell
        elevation_delta = (
            elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
        )
    else:
        elevation_delta = 0.0

    # Determine the harmonic densities for each phase across the face
    if water_density_grid is not None:
        harmonic_water_density = compute_harmonic_mean(
            water_density_grid[neighbour_indices], water_density_grid[cell_indices]
        )
    else:
        harmonic_water_density = 0.0

    if oil_density_grid is not None:
        harmonic_oil_density = compute_harmonic_mean(
            oil_density_grid[neighbour_indices], oil_density_grid[cell_indices]
        )
    else:
        harmonic_oil_density = 0.0

    if gas_density_grid is not None:
        harmonic_gas_density = compute_harmonic_mean(
            gas_density_grid[neighbour_indices], gas_density_grid[cell_indices]
        )
    else:
        harmonic_gas_density = 0.0

    # Calculate harmonic mobilities for each phase across the face
    water_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices, index2=neighbour_indices, mobility_grid=water_mobility_grid
    )
    oil_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices, index2=neighbour_indices, mobility_grid=oil_mobility_grid
    )
    gas_harmonic_mobility = compute_harmonic_mobility(
        index1=cell_indices, index2=neighbour_indices, mobility_grid=gas_mobility_grid
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
        harmonic_water_density
        * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2
        * elevation_delta
    ) / 144.0
    oil_gravity_potential = (
        harmonic_oil_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    gas_gravity_potential = (
        harmonic_gas_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
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
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    options: Options,
) -> EvolutionResult[ThreeDimensionalGrid]:
    """
    Solves the fully implicit finite-difference pressure equation for a slightly compressible,
    three-phase flow system in a 3D reservoir.

    This version treats well terms explicitly by computing flow rates directly (like the explicit method)
    rather than linearizing them. This approach:
    1. Handles pseudo-pressure for gas wells correctly
    2. Simplifies unit conversions
    3. Maintains stability through implicit treatment of inter-cell fluxes
    4. Uses lagged well rates (computed at beginning of time step)

    The well treatment is explicit in time but doesn't compromise overall stability because:
    - Wells are localized (affect only specific cells)
    - Inter-cell fluxes (which dominate) remain implicit
    - Time step restrictions are primarily governed by saturation changes, not well rates
    """
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

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

    # Initialize sparse coefficient matrix and RHS vector
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    dtype = get_dtype()
    A = lil_matrix((interior_cell_count, interior_cell_count), dtype=dtype)
    b = np.zeros(interior_cell_count, dtype=dtype)

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
            cap_face = cap_flux * geometric_factor
            grav_face = grav_flux * geometric_factor

            rhs_face_term = cap_face + grav_face
            b[cell_1D_index] += rhs_face_term
            b[neighbour_1D] += -rhs_face_term

    # THIRD PASS: Compute well flow rates and add to RHS
    # This is done explicitly (using current pressure) similar to explicit method
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_1D_index = _to_1D_index(i, j, k)
        cell_thickness = thickness_grid[i, j, k]
        cell_temperature = fluid_properties.temperature_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]

        injection_well, production_well = wells[i, j, k]
        # Net well flow rate into the cell (ft³/day)
        # Positive = injection, Negative = production
        net_well_flow_rate = 0.0

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
            injected_phase = injected_fluid.phase

            # Get phase mobility
            if injected_phase == FluidPhase.GAS:
                phase_mobility = gas_relative_mobility_grid[i, j, k]
                compressibility_kwargs = {}
            else:  # Water injection
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

            # Get fluid properties
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

            # Compute injection rate (bbls/day for liquids, ft³/day for gas)
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

            # Check for backflow (negative injection)
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

            # Convert to ft³/day if not already
            if injected_phase != FluidPhase.GAS:
                cell_injection_rate *= c.BBL_TO_FT3

            net_well_flow_rate += cell_injection_rate

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

                # Get phase-specific properties
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
                else:  # Oil
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
                # Compute production rate (bbls/day for liquids, ft³/day for gas)
                # Note: Production rates are negative by convention
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

                # Check for backflow (positive production = injection)
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

                # Convert to ft³/day if not already
                if produced_phase != FluidPhase.GAS:
                    production_rate *= c.BBL_TO_FT3

                net_well_flow_rate += production_rate  # Production rates are negative

        # Add well flow rate contribution to RHS
        # The well flow rate is in ft³/day (reservoir conditions)
        # We add it directly to the RHS without any coefficient in the matrix
        # This makes the well treatment explicit in time
        #
        # The contribution to the pressure equation RHS is simply the volumetric flow rate
        # Units: ft³/day
        b[cell_1D_index] += net_well_flow_rate

    # Solve the linear system
    A_csr = A.tocsr()
    ml = pyamg.ruge_stuben_solver(A_csr)
    M = ml.aspreconditioner(cycle="V")

    b_norm = np.linalg.norm(b)
    epsilon = get_floating_point_info().eps
    rtol = epsilon * 50
    atol = max(1e-8, 20 * epsilon * b_norm)
    new_1D_pressure_grid, info = gmres(
        A=A_csr,
        b=b,
        M=M,
        rtol=rtol,  # type: ignore
        atol=atol,  # type: ignore
        restart=50,
        maxiter=options.max_iterations,
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
