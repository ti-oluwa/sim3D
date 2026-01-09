import itertools
import typing

import attrs
import numba
import numpy as np

from bores._precision import get_dtype
from bores.config import Config
from bores.constants import c
from bores.diffusivity.base import (
    EvolutionResult,
    _warn_injection_rate_is_negative,
    _warn_production_rate_is_positive,
    compute_mobility_grids,
)
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.pvt import build_total_fluid_compressibility_grid
from bores.models import FluidProperties, RockProperties
from bores.pvt.core import compute_harmonic_mean
from bores.types import FluidPhase, ThreeDimensionalGrid, ThreeDimensions
from bores.wells import Wells

__all__ = ["evolve_pressure_explicitly"]


@attrs.frozen
class ExplicitPressureSolution:
    pressure_grid: ThreeDimensionalGrid
    max_cfl_encountered: float
    cfl_threshold: float


def evolve_pressure_explicitly(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
) -> EvolutionResult[ExplicitPressureSolution, None]:
    """
    Computes the pressure evolution (specifically, oil phase pressure P_oil) in the reservoir grid
    for one time step using an explicit finite volume method.

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
    :param wells: `Wells` object containing well parameters for injection and production wells
    :return: A N-Dimensional numpy array representing the updated oil phase pressure field (psi).
    """
    time_step_size_in_days = time_step_size * c.DAYS_PER_SECOND
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    # This is P_oil or Pⁿ_{i,j}
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
    # Total compressibility (psi⁻¹) = fluid compressibility + rock compressibility
    total_compressibility_grid = total_fluid_compressibility_grid + rock_compressibility
    # Clamp the compressibility within range
    total_compressibility_grid = config.total_compressibility_range.clip(
        total_compressibility_grid
    )

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    # Compute mobility grids for x, y, z directions
    mobility_grids = compute_mobility_grids(
        absolute_permeability_x=absolute_permeability.x,
        absolute_permeability_y=absolute_permeability.y,
        absolute_permeability_z=absolute_permeability.z,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        md_per_cp_to_ft2_per_psi_per_day=c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY,
    )

    (
        (water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x),
        (water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y),
        (water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z),
    ) = mobility_grids

    dtype = get_dtype()

    # Compute CFL number for this time step
    pressure_cfl = compute_pressure_cfl_number(
        time_step_size_in_days=time_step_size_in_days,
        porosity_grid=porosity_grid,
        total_compressibility_grid=total_compressibility_grid,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        water_mobility_grid_x=water_mobility_grid_x,
        oil_mobility_grid_x=oil_mobility_grid_x,
        gas_mobility_grid_x=gas_mobility_grid_x,
        water_mobility_grid_y=water_mobility_grid_y,
        oil_mobility_grid_y=oil_mobility_grid_y,
        gas_mobility_grid_y=gas_mobility_grid_y,
        water_mobility_grid_z=water_mobility_grid_z,
        oil_mobility_grid_z=oil_mobility_grid_z,
        gas_mobility_grid_z=gas_mobility_grid_z,
    )
    max_pressure_cfl = config.explicit_pressure_cfl_threshold
    if pressure_cfl > max_pressure_cfl:
        return EvolutionResult(
            success=False,
            scheme="explicit",
            value=ExplicitPressureSolution(
                pressure_grid=current_oil_pressure_grid.astype(dtype, copy=False),
                max_cfl_encountered=pressure_cfl,
                cfl_threshold=max_pressure_cfl,
            ),
            message=f"Pressure evolution failed with CFL={pressure_cfl:.4f}.",
        )

    # Compute net flux contributions from neighbors
    net_flux_grid = compute_net_flux_contributions(
        current_oil_pressure_grid=current_oil_pressure_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        water_mobility_grid_x=water_mobility_grid_x,
        oil_mobility_grid_x=oil_mobility_grid_x,
        gas_mobility_grid_x=gas_mobility_grid_x,
        water_mobility_grid_y=water_mobility_grid_y,
        oil_mobility_grid_y=oil_mobility_grid_y,
        gas_mobility_grid_y=gas_mobility_grid_y,
        water_mobility_grid_z=water_mobility_grid_z,
        oil_mobility_grid_z=oil_mobility_grid_z,
        gas_mobility_grid_z=gas_mobility_grid_z,
        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
        oil_density_grid=oil_density_grid,
        water_density_grid=water_density_grid,
        gas_density_grid=gas_density_grid,
        elevation_grid=elevation_grid,
        acceleration_due_to_gravity_ft_per_s2=c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2,
        dtype=dtype,
    )

    # Compute well rate contributions
    well_rate_grid = compute_well_rate_grid(
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        wells=wells,
        current_oil_pressure_grid=current_oil_pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        absolute_permeability=absolute_permeability,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        water_compressibility_grid=water_compressibility_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        fluid_properties=fluid_properties,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step=time_step,
        time_step_size=time_step_size,
        config=config,
        dtype=dtype,
    )

    # Apply pressure updates
    updated_oil_pressure_grid = apply_pressure_updates(
        current_oil_pressure_grid=current_oil_pressure_grid,
        net_flux_grid=net_flux_grid,
        well_rate_grid=well_rate_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        porosity_grid=porosity_grid,
        total_compressibility_grid=total_compressibility_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step_size_in_days=time_step_size_in_days,
    )
    return EvolutionResult(
        success=True,
        scheme="explicit",
        value=ExplicitPressureSolution(
            pressure_grid=updated_oil_pressure_grid.astype(dtype, copy=False),
            max_cfl_encountered=pressure_cfl,
            cfl_threshold=max_pressure_cfl,
        ),
        message=f"Pressure evolution from time step {time_step} successful with CFL={pressure_cfl:.4f}.",
    )


@numba.njit(cache=True)
def compute_pressure_cfl_number(
    time_step_size_in_days: float,
    porosity_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
) -> float:
    """
    Compute the maximum CFL number across all cells for pressure evolution.

    CFL = Δt * (Σ Transmissibility) / (φ * c_t * V)

    For stability, CFL should be ≤ 1.0 (or a safety factor like 0.5)

    :param time_step_size_in_days: Current time step size (days)
    :param porosity_grid: Porosity grid (fraction)
    :param total_compressibility_grid: Total compressibility (1/psi)
    :param thickness_grid: Cell thickness (ft)
    :param cell_size_x: Cell size in x (ft)
    :param cell_size_y: Cell size in y (ft)
    :param water_mobility_grid_x: Water mobility in x-direction (ft²/psi·day)
    :param oil_mobility_grid_x: Oil mobility in x-direction (ft²/psi·day)
    :param gas_mobility_grid_x: Gas mobility in x-direction (ft²/psi·day)
    :param water_mobility_grid_y: Water mobility in y-direction (ft²/psi·day)
    :param oil_mobility_grid_y: Oil mobility in y-direction (ft²/psi·day)
    :param gas_mobility_grid_y: Gas mobility in y-direction (ft²/psi·day)
    :param water_mobility_grid_z: Water mobility in z-direction (ft²/psi·day)
    :param oil_mobility_grid_z: Oil mobility in z-direction (ft²/psi·day)
    :param gas_mobility_grid_z: Gas mobility in z-direction (ft²/psi·day)
    :return: Maximum CFL number across all cells
    """
    cell_count_x, cell_count_y, cell_count_z = porosity_grid.shape
    max_cfl = 0.0

    for i in range(1, cell_count_x - 1):
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_thickness = thickness_grid[i, j, k]
                cell_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_compressibility = total_compressibility_grid[i, j, k]

                # Compute total transmissibility to all neighbors
                total_transmissibility = 0.0

                # X-direction transmissibilities (2 neighbors: i+1 and i-1)
                geometric_factor_x = cell_size_y * cell_thickness / cell_size_x
                total_mobility_x = (
                    water_mobility_grid_x[i, j, k]
                    + oil_mobility_grid_x[i, j, k]
                    + gas_mobility_grid_x[i, j, k]
                )
                total_transmissibility += 2.0 * total_mobility_x * geometric_factor_x

                # Y-direction transmissibilities (2 neighbors: j+1 and j-1)
                geometric_factor_y = cell_size_x * cell_thickness / cell_size_y
                total_mobility_y = (
                    water_mobility_grid_y[i, j, k]
                    + oil_mobility_grid_y[i, j, k]
                    + gas_mobility_grid_y[i, j, k]
                )
                total_transmissibility += 2.0 * total_mobility_y * geometric_factor_y

                # Z-direction transmissibilities (2 neighbors: k+1 and k-1)
                geometric_factor_z = cell_size_x * cell_size_y / cell_thickness
                total_mobility_z = (
                    water_mobility_grid_z[i, j, k]
                    + oil_mobility_grid_z[i, j, k]
                    + gas_mobility_grid_z[i, j, k]
                )
                total_transmissibility += 2.0 * total_mobility_z * geometric_factor_z

                # Compute CFL number for this cell
                if cell_compressibility > 0.0 and cell_porosity > 0.0:
                    cell_cfl = (time_step_size_in_days * total_transmissibility) / (
                        cell_porosity * cell_compressibility * cell_volume
                    )

                    max_cfl = max(max_cfl, cell_cfl)
    return max_cfl


@numba.njit(cache=True)
def compute_pseudo_flux_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_mobility_grid: ThreeDimensionalGrid,
    oil_mobility_grid: ThreeDimensionalGrid,
    gas_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    acceleration_due_to_gravity_ft_per_s2: float,
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

    :param cell_indices: Indices of the current cell (i, j, k).
    :param neighbour_indices: Indices of the neighbour cell (i±1, j, k) or (i, j±1, k) or (i, j, k±1).
    :param oil_pressure_grid: N-Dimensional numpy array representing the oil phase pressure grid (psi).
    :param water_mobility_grid: N-Dimensional numpy array representing the water phase mobility grid (ft²/psi/day).
    :param oil_mobility_grid: N-Dimensional numpy array representing the oil phase mobility grid (ft²/psi/day).
    :param gas_mobility_grid: N-Dimensional numpy array representing the gas phase mobility grid (ft²/psi/day).
    :param oil_water_capillary_pressure_grid: N-Dimensional numpy array representing the oil-water capillary pressure grid (psi).
    :param gas_oil_capillary_pressure_grid: N-Dimensional numpy array representing the gas-oil capillary pressure grid (psi).
    :param oil_density_grid: N-Dimensional numpy array representing the oil phase density grid (lb/ft³).
    :param water_density_grid: N-Dimensional numpy array representing the water phase density grid (lb/ft³).
    :param gas_density_grid: N-Dimensional numpy array representing the gas phase density grid (lb/ft³).
    :param elevation_grid: N-Dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :return: Total pseudo volumetric flux from neighbour to current cell (ft²/day).
    """
    # Calculate pressure differences relative to current cell (Neighbour - Current)
    # These represent the gradients driving flow from Neighbour to currrent cell, or vice versa
    oil_pressure_difference = (
        oil_pressure_grid[neighbour_indices] - oil_pressure_grid[cell_indices]
    )
    oil_water_capillary_pressure_difference = (
        oil_water_capillary_pressure_grid[neighbour_indices]
        - oil_water_capillary_pressure_grid[cell_indices]
    )
    water_pressure_difference = (
        oil_pressure_difference - oil_water_capillary_pressure_difference
    )
    # Gas pressure difference is calculated as:
    gas_oil_capillary_pressure_difference = (
        gas_oil_capillary_pressure_grid[neighbour_indices]
        - gas_oil_capillary_pressure_grid[cell_indices]
    )
    gas_pressure_difference = (
        oil_pressure_difference + gas_oil_capillary_pressure_difference
    )

    # Calculate the elevation difference between the neighbour and current cell
    elevation_difference = (
        elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
    )
    # Determine the harmonic densities for each phase across the face
    harmonic_water_density = compute_harmonic_mean(
        water_density_grid[neighbour_indices], water_density_grid[cell_indices]
    )
    harmonic_oil_density = compute_harmonic_mean(
        oil_density_grid[neighbour_indices], oil_density_grid[cell_indices]
    )
    harmonic_gas_density = compute_harmonic_mean(
        gas_density_grid[neighbour_indices], gas_density_grid[cell_indices]
    )

    # Calculate harmonic mobilities for each phase across the face (in the direction of flow)
    water_harmonic_mobility = compute_harmonic_mean(
        water_mobility_grid[neighbour_indices], water_mobility_grid[cell_indices]
    )
    oil_harmonic_mobility = compute_harmonic_mean(
        oil_mobility_grid[neighbour_indices], oil_mobility_grid[cell_indices]
    )
    gas_harmonic_mobility = compute_harmonic_mean(
        gas_mobility_grid[neighbour_indices], gas_mobility_grid[cell_indices]
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
        harmonic_water_density
        * acceleration_due_to_gravity_ft_per_s2
        * elevation_difference
    ) / 144.0
    # Calculate the total water phase potential
    water_potential_difference = water_pressure_difference + water_gravity_potential
    # Calculate the volumetric flux of water from neighbour to current cell
    water_pseudo_volumetric_flux = water_harmonic_mobility * water_potential_difference

    # For Oil:
    # Calculate the oil gravity potential (gravity head)
    oil_gravity_potential = (
        harmonic_oil_density
        * acceleration_due_to_gravity_ft_per_s2
        * elevation_difference
    ) / 144.0
    # Calculate the total oil phase potential
    oil_potential_difference = oil_pressure_difference + oil_gravity_potential
    # Calculate the volumetric flux of oil from neighbour to current cell
    oil_pseudo_volumetric_flux = oil_harmonic_mobility * oil_potential_difference

    # For Gas:
    # q = λ * ∆P (ft²/day)
    # Calculate the gas gravity potential (gravity head)
    gas_gravity_potential = (
        harmonic_gas_density
        * acceleration_due_to_gravity_ft_per_s2
        * elevation_difference
    ) / 144.0
    # Calculate the total gas phase potential
    gas_potential_difference = gas_pressure_difference + gas_gravity_potential
    # Calculate the volumetric flux of gas from neighbour to current cell
    gas_pseudo_volumetric_flux = gas_harmonic_mobility * gas_potential_difference

    # Add these incoming fluxes to the net total for the cell, q (ft²/day)
    total_pseudo_volumetric_flux = (
        water_pseudo_volumetric_flux
        + oil_pseudo_volumetric_flux
        + gas_pseudo_volumetric_flux
    )
    return total_pseudo_volumetric_flux


@numba.njit(parallel=True, cache=True)
def compute_net_flux_contributions(
    current_oil_pressure_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    acceleration_due_to_gravity_ft_per_s2: float,
    dtype: np.typing.DTypeLike,
) -> ThreeDimensionalGrid:
    """
    Compute net volumetric flux into each interior cell from all 6 neighbors (excluding wells).

    This function implements the flux computation portion of the explicit pressure evolution.
    For each interior cell, it computes fluxes from neighbors in all three directions (x, y, z),
    considering capillary pressure effects and gravity.

    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param water_mobility_grid_x: Water mobility grid in x-direction (ft²/psi·day)
    :param oil_mobility_grid_x: Oil mobility grid in x-direction (ft²/psi·day)
    :param gas_mobility_grid_x: Gas mobility grid in x-direction (ft²/psi·day)
    :param water_mobility_grid_y: Water mobility grid in y-direction (ft²/psi·day)
    :param oil_mobility_grid_y: Oil mobility grid in y-direction (ft²/psi·day)
    :param gas_mobility_grid_y: Gas mobility grid in y-direction (ft²/psi·day)
    :param water_mobility_grid_z: Water mobility grid in z-direction (ft²/psi·day)
    :param oil_mobility_grid_z: Oil mobility grid in z-direction (ft²/psi·day)
    :param gas_mobility_grid_z: Gas mobility grid in z-direction (ft²/psi·day)
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure (psi)
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure (psi)
    :param oil_density_grid: Oil density grid (lb/ft³)
    :param water_density_grid: Water density grid (lb/ft³)
    :param gas_density_grid: Gas density grid (lb/ft³)
    :param elevation_grid: Elevation grid (ft)
    :param acceleration_due_to_gravity_ft_per_s2: Gravitational acceleration (ft/s²)
    :param dtype: NumPy dtype for array allocation (np.float32 or np.float64)
    :return: 3D grid of net volumetric fluxes (ft³/day), positive = flow into cell
    """
    flux_grid = np.zeros((cell_count_x, cell_count_y, cell_count_z), dtype=dtype)

    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_thickness = thickness_grid[i, j, k]
                net_flux = 0.0

                # X-direction neighbors (i+1 and i-1)
                geometric_factor_x = cell_size_y * cell_thickness / cell_size_x
                for ni in (i + 1, i - 1):
                    flux = compute_pseudo_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(ni, j, k),
                        oil_pressure_grid=current_oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_x,
                        oil_mobility_grid=oil_mobility_grid_x,
                        gas_mobility_grid=gas_mobility_grid_x,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                    )
                    net_flux += flux * geometric_factor_x

                # Y-direction neighbors (j+1 and j-1)
                geometric_factor_y = cell_size_x * cell_thickness / cell_size_y
                for nj in (j + 1, j - 1):
                    flux = compute_pseudo_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, nj, k),
                        oil_pressure_grid=current_oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_y,
                        oil_mobility_grid=oil_mobility_grid_y,
                        gas_mobility_grid=gas_mobility_grid_y,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                    )
                    net_flux += flux * geometric_factor_y

                # Z-direction neighbors (k+1 and k-1)
                geometric_factor_z = cell_size_x * cell_size_y / cell_thickness
                for nk in (k + 1, k - 1):
                    flux = compute_pseudo_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, nk),
                        oil_pressure_grid=current_oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_z,
                        oil_mobility_grid=oil_mobility_grid_z,
                        gas_mobility_grid=gas_mobility_grid_z,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        acceleration_due_to_gravity_ft_per_s2=acceleration_due_to_gravity_ft_per_s2,
                    )
                    net_flux += flux * geometric_factor_z

                flux_grid[i, j, k] = net_flux

    return flux_grid


def compute_well_rate_grid(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    wells: Wells[ThreeDimensions],
    current_oil_pressure_grid: ThreeDimensionalGrid,
    temperature_grid: ThreeDimensionalGrid,
    absolute_permeability: typing.Any,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    water_compressibility_grid: ThreeDimensionalGrid,
    oil_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    fluid_properties: FluidProperties[ThreeDimensions],
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step: int,
    time_step_size: float,
    config: Config,
    dtype: np.typing.DTypeLike,
) -> ThreeDimensionalGrid:
    """
    Compute well rates for all cells (injection + production).

    This function computes the net well flow rate for each cell by evaluating injection
    and production well contributions. It handles phase-specific calculations, pseudo-pressure
    for gas wells, and backflow warnings. This function is NOT jitted because it requires
    method calls on well and fluid objects.

    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param wells: Wells grid containing injection and production wells
    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param temperature_grid: Temperature grid (°F or °R)
    :param absolute_permeability: Absolute permeability in x, y, z directions (mD)
    :param water_relative_mobility_grid: Water relative mobility (1/cP)
    :param oil_relative_mobility_grid: Oil relative mobility (1/cP)
    :param gas_relative_mobility_grid: Gas relative mobility (1/cP)
    :param water_compressibility_grid: Water compressibility grid (1/psi)
    :param oil_compressibility_grid: Oil compressibility grid (1/psi)
    :param gas_compressibility_grid: Gas compressibility grid (1/psi)
    :param fluid_properties: Fluid properties container
    :param thickness_grid: Cell thickness grid (ft)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step: Current time step number
    :param time_step_size: Time step size (seconds)
    :param config: Simulation config
    :param dtype: NumPy dtype for array allocation (np.float32 or np.float64)
    :return: 3D grid of net well flow rates (ft³/day), positive = injection, negative = production
    """
    well_rate_grid = np.zeros((cell_count_x, cell_count_y, cell_count_z), dtype=dtype)

    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_thickness = thickness_grid[i, j, k]
        cell_temperature = temperature_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]

        injection_well, production_well = wells[i, j, k]
        net_well_flow_rate = 0.0

        permeability = (
            absolute_permeability.x[i, j, k],
            absolute_permeability.y[i, j, k],
            absolute_permeability.z[i, j, k],
        )
        interval_thickness = (cell_size_x, cell_size_y, cell_thickness)

        # Handle injection wells
        if (
            injection_well is not None
            and injection_well.is_open
            and (injected_fluid := injection_well.injected_fluid) is not None
        ):
            injected_phase = injected_fluid.phase
            phase_fvf = injected_fluid.get_formation_volume_factor(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
            )
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
                    "gas_formation_volume_factor": phase_fvf,
                    "gas_solubility_in_water": fluid_properties.gas_solubility_in_water_grid[
                        i, j, k
                    ],
                }

            # Get fluid properties
            phase_compressibility = injected_fluid.get_compressibility(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                **compressibility_kwargs,
            )

            use_pseudo_pressure = (
                config.use_pseudo_pressure and injected_phase == FluidPhase.GAS
            )
            well_index = injection_well.get_well_index(
                interval_thickness=interval_thickness,
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
                fluid_compressibility=phase_compressibility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=phase_fvf,
                pvt_tables=config.pvt_tables,
            )

            # Check for backflow (negative injection)
            if cell_injection_rate < 0.0 and config.warn_well_anomalies:
                _warn_injection_rate_is_negative(
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

        # Handle production wells
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
                    phase_compressibility = gas_compressibility_grid[i, j, k]
                    phase_fvf = gas_formation_volume_factor_grid[i, j, k]
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = water_relative_mobility_grid[i, j, k]
                    phase_compressibility = water_compressibility_grid[i, j, k]
                    phase_fvf = water_formation_volume_factor_grid[i, j, k]
                else:  # Oil
                    phase_mobility = oil_relative_mobility_grid[i, j, k]
                    phase_compressibility = oil_compressibility_grid[i, j, k]
                    phase_fvf = oil_formation_volume_factor_grid[i, j, k]

                use_pseudo_pressure = (
                    config.use_pseudo_pressure and produced_phase == FluidPhase.GAS
                )
                well_index = production_well.get_well_index(
                    interval_thickness=interval_thickness,
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
                    fluid_compressibility=phase_compressibility,
                    use_pseudo_pressure=use_pseudo_pressure,
                    formation_volume_factor=phase_fvf,
                    pvt_tables=config.pvt_tables,
                )

                # Check for backflow (positive production = injection)
                if production_rate > 0.0 and config.warn_well_anomalies:
                    _warn_production_rate_is_positive(
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

        well_rate_grid[i, j, k] = net_well_flow_rate

    return well_rate_grid


@numba.njit(parallel=True, cache=True)
def apply_pressure_updates(
    current_oil_pressure_grid: ThreeDimensionalGrid,
    net_flux_grid: ThreeDimensionalGrid,
    well_rate_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_size_in_days: float,
) -> ThreeDimensionalGrid:
    """
    Apply pressure updates to all interior cells using pre-computed flux and well contributions.

    This function combines the net flux contributions (from neighbors) and well contributions
    to compute and apply pressure changes to each interior cell. The computation is parallelized
    across all interior cells using prange.

    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param net_flux_grid: Pre-computed net flux contributions (ft³/day)
    :param well_rate_grid: Pre-computed well rate contributions (ft³/day)
    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param porosity_grid: Cell porosity grid (fraction)
    :param total_compressibility_grid: Total compressibility grid (1/psi)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step_size_in_days: Time step size (days)
    :return: Updated oil pressure grid (psi)
    """
    updated_grid = current_oil_pressure_grid.copy()

    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                # Cell properties
                cell_thickness = thickness_grid[i, j, k]
                cell_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_total_compressibility = total_compressibility_grid[i, j, k]

                # Total flow rate = flux from neighbors + well contribution
                net_volumetric_flow = net_flux_grid[i, j, k]
                net_well_flow = well_rate_grid[i, j, k]
                total_flow_rate = net_volumetric_flow + net_well_flow

                # Calculate pressure change
                # dP = (Δt / (φ * c_t * V)) * Q_total
                change_in_pressure = (
                    time_step_size_in_days
                    / (cell_porosity * cell_total_compressibility * cell_volume)
                ) * total_flow_rate

                # Apply the update
                updated_grid[i, j, k] += change_in_pressure

    return updated_grid


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
    Δt ≤ min( (φ·c_t·V) / (2 * λ_max * (A_x/Δx + A_y/Δy + A_z/Δz)) )

Notes:
    - Harmonic averaging ensures continuity of flux across interfaces.
    - Volume-normalized source/sink terms only affect the cell where the well is located.
"""
