import itertools
import typing

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, gmres
import pyamg
import logging

from sim3D._precision import get_dtype, get_floating_point_info
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
    FluxDerivativesWithRespectToSaturations,
    MiscibleFluxDerivativesWithRespectToSaturations,
    MisciblePhaseFluxDerivatives,
    Options,
    PhaseFluxDerivatives,
    RelativeMobilityGrids,
    RelativePermeabilityTable,
    SupportsSetItem,
    ThreeDimensionalGrid,
    ThreeDimensions,
    Time,
)
from sim3D.wells import Wells

__all__ = [
    "evolve_saturation_explicitly",
    "evolve_miscible_saturation_explicitly",
    "evolve_saturation_implicitly",
    "evolve_miscible_saturation_implicitly",
]

logger = logging.getLogger(__name__)


def _clamp_tolerances(
    dtype: np.typing.DTypeLike,
    user_rtol: float,
    user_atol: float,
    b_norm: float,
) -> typing.Tuple[float, float]:
    """
    Clamp solver tolerances based on machine precision to prevent numerical instability.

    For float32:
    - atol >= 1e-5 (practical lower bound for stable GMRES)
    - rtol >= 3e-4 (sqrt(eps) for float32)

    For float64:
    - atol >= 1e-12
    - rtol >= 1e-8 (sqrt(eps) for float64)

    :param dtype: Data type of arrays (float32 or float64)
    :param user_rtol: User-specified relative tolerance
    :param user_atol: User-specified absolute tolerance
    :param b_norm: Norm of RHS vector for adaptive clamping
    :return: Tuple of (clamped_rtol, clamped_atol)
    """
    fp_info = get_floating_point_info()
    eps = fp_info.eps

    # Practical bounds based on dtype
    if dtype == np.float32:
        min_atol = 1e-5  # Practical lower bound for float32 GMRES
        min_rtol = 3e-4  # sqrt(eps) for float32
    else:  # float64
        min_atol = 1e-12
        min_rtol = 1e-8  # sqrt(eps) for float64

    # Clamp relative tolerance
    rtol_clamped = max(user_rtol, min_rtol)

    # Clamp absolute tolerance with adaptive component
    # atol should scale with problem size (b_norm)
    adaptive_atol = max(min_atol, 10 * eps * b_norm)
    atol_clamped = max(user_atol, adaptive_atol)
    atol_clamped = typing.cast(float, atol_clamped)
    return rtol_clamped, atol_clamped


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


def _compute_explicit_saturation_phase_fluxes_from_neighbour(
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

    # Iterate over each interior cell to compute saturation evolution
    # # Assume boundary cells are added via padding for boundary conditions application purposes
    # Thus, we iterate from 1 to N-1 in each dimension
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
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
                ) = _compute_explicit_saturation_phase_fluxes_from_neighbour(
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

    # Clean up any remaining minor negative values caused by floating point errors
    updated_water_saturation_grid[updated_water_saturation_grid < 0.0] = 0.0
    updated_oil_saturation_grid[updated_oil_saturation_grid < 0.0] = 0.0
    updated_gas_saturation_grid[updated_gas_saturation_grid < 0.0] = 0.0
    return EvolutionResult(
        (
            updated_water_saturation_grid,
            updated_oil_saturation_grid,
            updated_gas_saturation_grid,
        ),
        scheme="explicit",
    )


def _compute_explicit_miscible_phase_fluxes_from_neighbour(
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
    for i, j, k in itertools.product(
        range(1, cell_count_x - 1),
        range(1, cell_count_y - 1),
        range(1, cell_count_z - 1),
    ):
        cell_thickness = thickness_grid[i, j, k]
        cell_volume = cell_size_x * cell_size_y * cell_thickness
        cell_porosity = porosity_grid[i, j, k]
        cell_pore_volume = cell_volume * cell_porosity
        cell_water_saturation = current_water_saturation_grid[i, j, k]
        cell_gas_saturation = current_gas_saturation_grid[i, j, k]
        cell_oil_saturation = current_oil_saturation_grid[i, j, k]
        cell_solvent_concentration = current_solvent_concentration_grid[i, j, k]
        cell_oil_pressure = current_oil_pressure_grid[i, j, k]
        cell_temperature = fluid_properties.temperature_grid[i, j, k]

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
                ) = _compute_explicit_miscible_phase_fluxes_from_neighbour(
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

        updated_water_saturation_grid[i, j, k] = (
            cell_water_saturation + water_saturation_change
        )
        updated_oil_saturation_grid[i, j, k] = (
            cell_oil_saturation + oil_saturation_change
        )
        updated_gas_saturation_grid[i, j, k] = (
            cell_gas_saturation + gas_saturation_change
        )

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

    # Clean up any remaining minor negative values caused by floating point errors
    updated_water_saturation_grid[updated_water_saturation_grid < 0.0] = 0.0
    updated_oil_saturation_grid[updated_oil_saturation_grid < 0.0] = 0.0
    updated_gas_saturation_grid[updated_gas_saturation_grid < 0.0] = 0.0
    return EvolutionResult(
        (
            updated_water_saturation_grid,
            updated_oil_saturation_grid,
            updated_gas_saturation_grid,
            updated_solvent_concentration_grid,
        ),
        scheme="explicit",
    )


"""
Implicit finite difference formulation for saturation transport in a 3D reservoir
(immiscible three-phase flow: oil, water, and gas with slightly compressible fluids):

The governing equation for saturation evolution is the conservation of mass with advection:

    ∂S/∂t * (φ * V_cell) = -∇ · (f_x * v) * V_cell + q_x * V_cell

Implicit Discretization:

Time: Backward Euler (implicit)
    ∂S/∂t ≈ (Sⁿ⁺¹_ijk - Sⁿ_ijk) / Δt

Space: First-order upwind scheme evaluated at new time level n+1:

    Sⁿ⁺¹_ijk = Sⁿ_ijk + Δt / (φ * V_cell) * [
        (F_x_east^(n+1) + F_x_west^(n+1)) + (F_y_north^(n+1) + F_y_south^(n+1)) + 
        (F_z_top^(n+1) + F_z_bottom^(n+1)) + q_x_ijk^(n+1) * V_cell
    ]

Key difference from explicit: All fluxes F_dir^(n+1) and fractional flows f_x^(n+1) 
are evaluated using saturations at the NEW time level (n+1), making this a coupled 
nonlinear system that requires iterative solution (e.g., Newton-Raphson).

For the implicit scheme:
- Upwind saturations S_upwind^(n+1) depend on new saturations
- Relative permeabilities k_r(S^(n+1)) depend on new saturations
- Fractional flows f_x(S^(n+1)) depend on new saturations
- This creates a system of nonlinear equations requiring iterative solution

Residual Form for Newton-Raphson:

For each phase x and cell (i,j,k), define residual:

    R_x_ijk = Sⁿ⁺¹_x_ijk - Sⁿ_x_ijk - Δt / (φ * V_cell) * [
        Σ(F_dir^(n+1)) + q_x_ijk^(n+1) * V_cell
    ]

At convergence: R_x_ijk = 0 for all cells and phases

Jacobian computation requires derivatives ∂R_x_ijk/∂S_y_lmn for all phases x,y 
and cells (i,j,k), (l,m,n) coupled by flow.

Stability:
- Unconditionally stable (no CFL restriction)
- Allows larger time steps than explicit
- Requires iterative solution per time step
"""


def _adaptive_step_size(s: float) -> float:
    """
    Compute adaptive step size for numerical derivatives.
    Uses sqrt(machine_epsilon) * max(1, |s|) as a robust choice.
    """
    eps = np.finfo(np.float64).eps
    return max(1e-8, np.sqrt(eps) * max(1.0, abs(float(s))))


def _compute_phase_fluxes_two_saturation(
    cell_water_saturation: float,
    cell_oil_saturation: float,
    neighbour_water_saturation: float,
    neighbour_oil_saturation: float,
    # Residual saturations
    cell_connate_water: float,
    cell_residual_oil_water: float,
    cell_residual_oil_gas: float,
    cell_residual_gas: float,
    neighbour_connate_water: float,
    neighbour_residual_oil_water: float,
    neighbour_residual_oil_gas: float,
    neighbour_residual_gas: float,
    # Viscosities
    cell_water_viscosity: float,
    cell_oil_viscosity: float,
    cell_gas_viscosity: float,
    neighbour_water_viscosity: float,
    neighbour_oil_viscosity: float,
    neighbour_gas_viscosity: float,
    # Phase potentials (already computed with gravity)
    water_phase_potential: float,
    oil_phase_potential: float,
    gas_phase_potential: float,
    # Geometric factors
    flow_area: float,
    flow_length: float,
    absolute_permeability_multiplier: float,
    # Relative permeability table
    relative_permeability_table: typing.Callable,
    # Upwind control
    use_locked_upwind: bool = False,
    locked_water_upwind_sign: float = 0.0,
    locked_oil_upwind_sign: float = 0.0,
    locked_gas_upwind_sign: float = 0.0,
) -> typing.Tuple[float, float, float]:
    """
    Compute phase fluxes given water and oil saturations.
    Gas saturation is computed as: S_g = 1 - S_w - S_o

    This function enforces the saturation constraint automatically.

    Args:
        use_locked_upwind: If True, use pre-computed upwind signs (for derivative evaluation)
        locked_*_upwind_sign: Pre-computed upwind direction signs

    Returns:
        (water_flux, oil_flux, gas_flux) in ft³/day
    """
    # Enforce physical bounds and compute gas saturations
    cell_water_saturation = float(np.clip(cell_water_saturation, 0.0, 1.0))
    cell_oil_saturation = float(np.clip(cell_oil_saturation, 0.0, 1.0))

    # Ensure S_w + S_o <= 1
    if cell_water_saturation + cell_oil_saturation > 1.0:
        total = cell_water_saturation + cell_oil_saturation
        cell_water_saturation /= total
        cell_oil_saturation /= total

    cell_gas_saturation = float(
        max(0.0, 1.0 - cell_water_saturation - cell_oil_saturation)
    )

    # Same for neighbour
    neighbour_water_saturation = float(np.clip(neighbour_water_saturation, 0.0, 1.0))
    neighbour_oil_saturation = float(np.clip(neighbour_oil_saturation, 0.0, 1.0))

    if neighbour_water_saturation + neighbour_oil_saturation > 1.0:
        total = neighbour_water_saturation + neighbour_oil_saturation
        neighbour_water_saturation /= total
        neighbour_oil_saturation /= total

    neighbour_gas_saturation = float(
        max(0.0, 1.0 - neighbour_water_saturation - neighbour_oil_saturation)
    )

    # Compute relative permeabilities
    rel_perms_cell = relative_permeability_table(
        water_saturation=cell_water_saturation,
        oil_saturation=cell_oil_saturation,
        gas_saturation=cell_gas_saturation,
        connate_water_saturation=cell_connate_water,
        residual_oil_saturation_water=cell_residual_oil_water,
        residual_oil_saturation_gas=cell_residual_oil_gas,
        residual_gas_saturation=cell_residual_gas,
    )

    rel_perms_neighbour = relative_permeability_table(
        water_saturation=neighbour_water_saturation,
        oil_saturation=neighbour_oil_saturation,
        gas_saturation=neighbour_gas_saturation,
        connate_water_saturation=neighbour_connate_water,
        residual_oil_saturation_water=neighbour_residual_oil_water,
        residual_oil_saturation_gas=neighbour_residual_oil_gas,
        residual_gas_saturation=neighbour_residual_gas,
    )

    # Compute mobilities: λ = k * kr / μ
    water_mobility_cell = (
        rel_perms_cell["water"]
        / cell_water_viscosity
        * absolute_permeability_multiplier
        if cell_water_viscosity > 0
        else 0.0
    )
    oil_mobility_cell = (
        rel_perms_cell["oil"] / cell_oil_viscosity * absolute_permeability_multiplier
        if cell_oil_viscosity > 0
        else 0.0
    )
    gas_mobility_cell = (
        rel_perms_cell["gas"] / cell_gas_viscosity * absolute_permeability_multiplier
        if cell_gas_viscosity > 0
        else 0.0
    )

    water_mobility_neighbour = (
        rel_perms_neighbour["water"]
        / neighbour_water_viscosity
        * absolute_permeability_multiplier
        if neighbour_water_viscosity > 0
        else 0.0
    )
    oil_mobility_neighbour = (
        rel_perms_neighbour["oil"]
        / neighbour_oil_viscosity
        * absolute_permeability_multiplier
        if neighbour_oil_viscosity > 0
        else 0.0
    )
    gas_mobility_neighbour = (
        rel_perms_neighbour["gas"]
        / neighbour_gas_viscosity
        * absolute_permeability_multiplier
        if neighbour_gas_viscosity > 0
        else 0.0
    )

    # Upwind mobility selection
    if use_locked_upwind:
        # Use frozen upwind decision (for derivative evaluation)
        water_mobility_upwind = (
            water_mobility_neighbour
            if locked_water_upwind_sign > 0.0
            else water_mobility_cell
        )
        oil_mobility_upwind = (
            oil_mobility_neighbour
            if locked_oil_upwind_sign > 0.0
            else oil_mobility_cell
        )
        gas_mobility_upwind = (
            gas_mobility_neighbour
            if locked_gas_upwind_sign > 0.0
            else gas_mobility_cell
        )
    else:
        # Recompute upwind decision based on potential
        water_mobility_upwind = (
            water_mobility_neighbour
            if water_phase_potential > 0.0
            else water_mobility_cell
        )
        oil_mobility_upwind = (
            oil_mobility_neighbour if oil_phase_potential > 0.0 else oil_mobility_cell
        )
        gas_mobility_upwind = (
            gas_mobility_neighbour if gas_phase_potential > 0.0 else gas_mobility_cell
        )

    # Compute Darcy velocities
    water_velocity = float(water_mobility_upwind * water_phase_potential / flow_length)
    oil_velocity = float(oil_mobility_upwind * oil_phase_potential / flow_length)
    gas_velocity = float(gas_mobility_upwind * gas_phase_potential / flow_length)

    # Compute volumetric fluxes
    water_flux = float(water_velocity * flow_area)
    oil_flux = float(oil_velocity * flow_area)
    gas_flux = float(gas_velocity * flow_area)

    return water_flux, oil_flux, gas_flux


def _compute_implicit_fluxes_and_derivatives_two_saturation(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    flow_area: float,
    flow_length: float,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    oil_viscosity_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    irreducible_water_saturation_grid: ThreeDimensionalGrid,
    residual_oil_saturation_water_grid: ThreeDimensionalGrid,
    residual_oil_saturation_gas_grid: ThreeDimensionalGrid,
    residual_gas_saturation_grid: ThreeDimensionalGrid,
    relative_permeability_table: typing.Callable,
    oil_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    water_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    gas_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    elevation_grid: typing.Optional[ThreeDimensionalGrid] = None,
    absolute_permeability_multiplier: float = 1.0,
) -> typing.Tuple[
    typing.Tuple[float, float, float],  # Fluxes
    FluxDerivativesWithRespectToSaturations,  # Derivatives
]:
    """
    Compute phase fluxes and derivatives using two-saturation formulation.
    Only S_w and S_o are independent; S_g = 1 - S_w - S_o.
    """
    # Get pressures
    cell_oil_pressure = oil_pressure_grid[cell_indices]
    cell_oil_water_pc = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_pc = gas_oil_capillary_pressure_grid[cell_indices]

    neighbour_oil_pressure = oil_pressure_grid[neighbour_indices]
    neighbour_oil_water_pc = oil_water_capillary_pressure_grid[neighbour_indices]
    neighbour_gas_oil_pc = gas_oil_capillary_pressure_grid[neighbour_indices]

    # Compute pressure differences
    oil_pressure_diff = neighbour_oil_pressure - cell_oil_pressure
    oil_water_pc_diff = neighbour_oil_water_pc - cell_oil_water_pc
    water_pressure_diff = oil_pressure_diff - oil_water_pc_diff
    gas_oil_pc_diff = neighbour_gas_oil_pc - cell_gas_oil_pc
    gas_pressure_diff = oil_pressure_diff + gas_oil_pc_diff

    # Gravity terms
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
            if water_pressure_diff > 0.0
            else water_density_grid[cell_indices]
        )
    else:
        upwind_water_density = 0.0

    if oil_density_grid is not None:
        upwind_oil_density = (
            oil_density_grid[neighbour_indices]
            if oil_pressure_diff > 0.0
            else oil_density_grid[cell_indices]
        )
    else:
        upwind_oil_density = 0.0

    if gas_density_grid is not None:
        upwind_gas_density = (
            gas_density_grid[neighbour_indices]
            if gas_pressure_diff > 0.0
            else gas_density_grid[cell_indices]
        )
    else:
        upwind_gas_density = 0.0

    # Phase potentials with gravity
    water_gravity_potential = (
        upwind_water_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    water_phase_potential = water_pressure_diff + water_gravity_potential

    oil_gravity_potential = (
        upwind_oil_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    oil_phase_potential = oil_pressure_diff + oil_gravity_potential

    gas_gravity_potential = (
        upwind_gas_density * c.ACCELERATION_DUE_TO_GRAVITY_FT_PER_S2 * elevation_delta
    ) / 144.0
    gas_phase_potential = gas_pressure_diff + gas_gravity_potential

    # Get current saturations (only need S_w and S_o)
    cell_water_sat = float(water_saturation_grid[cell_indices])
    cell_oil_sat = float(oil_saturation_grid[cell_indices])

    neighbour_water_sat = float(water_saturation_grid[neighbour_indices])
    neighbour_oil_sat = float(oil_saturation_grid[neighbour_indices])

    # Get residual saturations
    cell_connate_water = float(irreducible_water_saturation_grid[cell_indices])
    cell_residual_oil_water = float(residual_oil_saturation_water_grid[cell_indices])
    cell_residual_oil_gas = float(residual_oil_saturation_gas_grid[cell_indices])
    cell_residual_gas = float(residual_gas_saturation_grid[cell_indices])

    neighbour_connate_water = float(
        irreducible_water_saturation_grid[neighbour_indices]
    )
    neighbour_residual_oil_water = float(
        residual_oil_saturation_water_grid[neighbour_indices]
    )
    neighbour_residual_oil_gas = float(
        residual_oil_saturation_gas_grid[neighbour_indices]
    )
    neighbour_residual_gas = float(residual_gas_saturation_grid[neighbour_indices])

    # Get viscosities
    cell_water_visc = float(water_viscosity_grid[cell_indices])
    cell_oil_visc = float(oil_viscosity_grid[cell_indices])
    cell_gas_visc = float(gas_viscosity_grid[cell_indices])

    neighbour_water_visc = float(water_viscosity_grid[neighbour_indices])
    neighbour_oil_visc = float(oil_viscosity_grid[neighbour_indices])
    neighbour_gas_visc = float(gas_viscosity_grid[neighbour_indices])

    # Freeze upwind directions for derivative computation
    water_upwind_sign = np.sign(water_phase_potential)
    oil_upwind_sign = np.sign(oil_phase_potential)
    gas_upwind_sign = np.sign(gas_phase_potential)

    # Compute derivatives using central differences with locked upwind
    # ∂F/∂S_w_cell
    h_w_cell = _adaptive_step_size(cell_water_sat)

    fw_plus = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat + h_w_cell,
        cell_oil_saturation=cell_oil_sat,
        neighbour_water_saturation=neighbour_water_sat,
        neighbour_oil_saturation=neighbour_oil_sat,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    fw_minus = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat - h_w_cell,
        cell_oil_saturation=cell_oil_sat,
        neighbour_water_saturation=neighbour_water_sat,
        neighbour_oil_saturation=neighbour_oil_sat,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    dwater_dw_cell = float((fw_plus[0] - fw_minus[0]) / (2.0 * h_w_cell))
    doil_dw_cell = float((fw_plus[1] - fw_minus[1]) / (2.0 * h_w_cell))
    dgas_dw_cell = float((fw_plus[2] - fw_minus[2]) / (2.0 * h_w_cell))

    # ∂F/∂S_o_cell
    h_o_cell = _adaptive_step_size(cell_oil_sat)

    fo_plus = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat,
        cell_oil_saturation=cell_oil_sat + h_o_cell,
        neighbour_water_saturation=neighbour_water_sat,
        neighbour_oil_saturation=neighbour_oil_sat,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    fo_minus = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat,
        cell_oil_saturation=cell_oil_sat - h_o_cell,
        neighbour_water_saturation=neighbour_water_sat,
        neighbour_oil_saturation=neighbour_oil_sat,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    dwater_do_cell = float((fo_plus[0] - fo_minus[0]) / (2.0 * h_o_cell))
    doil_do_cell = float((fo_plus[1] - fo_minus[1]) / (2.0 * h_o_cell))
    dgas_do_cell = float((fo_plus[2] - fo_minus[2]) / (2.0 * h_o_cell))

    # Compute baseline flux (no locked upwind)
    water_flux, oil_flux, gas_flux = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat,
        cell_oil_saturation=cell_oil_sat,
        neighbour_water_saturation=neighbour_water_sat,
        neighbour_oil_saturation=neighbour_oil_sat,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    # ∂F/∂S_w_neighbour
    h_w_nbr = _adaptive_step_size(neighbour_water_sat)

    fnw_plus = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat,
        cell_oil_saturation=cell_oil_sat,
        neighbour_water_saturation=neighbour_water_sat + h_w_nbr,
        neighbour_oil_saturation=neighbour_oil_sat,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    fnw_minus = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat,
        cell_oil_saturation=cell_oil_sat,
        neighbour_water_saturation=neighbour_water_sat - h_w_nbr,
        neighbour_oil_saturation=neighbour_oil_sat,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    dwater_dw_nbr = float((fnw_plus[0] - fnw_minus[0]) / (2.0 * h_w_nbr))
    doil_dw_nbr = float((fnw_plus[1] - fnw_minus[1]) / (2.0 * h_w_nbr))
    dgas_dw_nbr = float((fnw_plus[2] - fnw_minus[2]) / (2.0 * h_w_nbr))

    # ∂F/∂S_o_neighbour
    h_o_nbr = _adaptive_step_size(neighbour_oil_sat)

    fno_plus = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat,
        cell_oil_saturation=cell_oil_sat,
        neighbour_water_saturation=neighbour_water_sat,
        neighbour_oil_saturation=neighbour_oil_sat + h_o_nbr,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    fno_minus = _compute_phase_fluxes_two_saturation(
        cell_water_saturation=cell_water_sat,
        cell_oil_saturation=cell_oil_sat,
        neighbour_water_saturation=neighbour_water_sat,
        neighbour_oil_saturation=neighbour_oil_sat - h_o_nbr,
        cell_connate_water=cell_connate_water,
        cell_residual_oil_water=cell_residual_oil_water,
        cell_residual_oil_gas=cell_residual_oil_gas,
        cell_residual_gas=cell_residual_gas,
        neighbour_connate_water=neighbour_connate_water,
        neighbour_residual_oil_water=neighbour_residual_oil_water,
        neighbour_residual_oil_gas=neighbour_residual_oil_gas,
        neighbour_residual_gas=neighbour_residual_gas,
        cell_water_viscosity=cell_water_visc,
        cell_oil_viscosity=cell_oil_visc,
        cell_gas_viscosity=cell_gas_visc,
        neighbour_water_viscosity=neighbour_water_visc,
        neighbour_oil_viscosity=neighbour_oil_visc,
        neighbour_gas_viscosity=neighbour_gas_visc,
        water_phase_potential=water_phase_potential,
        oil_phase_potential=oil_phase_potential,
        gas_phase_potential=gas_phase_potential,
        flow_area=flow_area,
        flow_length=flow_length,
        absolute_permeability_multiplier=absolute_permeability_multiplier,
        relative_permeability_table=relative_permeability_table,
        use_locked_upwind=True,
        locked_water_upwind_sign=water_upwind_sign,
        locked_oil_upwind_sign=oil_upwind_sign,
        locked_gas_upwind_sign=gas_upwind_sign,
    )

    dwater_do_nbr = float((fno_plus[0] - fno_minus[0]) / (2.0 * h_o_nbr))
    doil_do_nbr = float((fno_plus[1] - fno_minus[1]) / (2.0 * h_o_nbr))
    dgas_do_nbr = float((fno_plus[2] - fno_minus[2]) / (2.0 * h_o_nbr))

    # Construct derivative structure
    flux_derivatives = FluxDerivativesWithRespectToSaturations(
        water_phase_flux_derivatives=PhaseFluxDerivatives(
            derivative_wrt_water_saturation_at_cell=dwater_dw_cell,
            derivative_wrt_water_saturation_at_neighbour=dwater_dw_nbr,
            derivative_wrt_oil_saturation_at_cell=dwater_do_cell,
            derivative_wrt_oil_saturation_at_neighbour=dwater_do_nbr,
        ),
        oil_phase_flux_derivatives=PhaseFluxDerivatives(
            derivative_wrt_water_saturation_at_cell=doil_dw_cell,
            derivative_wrt_water_saturation_at_neighbour=doil_dw_nbr,
            derivative_wrt_oil_saturation_at_cell=doil_do_cell,
            derivative_wrt_oil_saturation_at_neighbour=doil_do_nbr,
        ),
        gas_phase_flux_derivatives=PhaseFluxDerivatives(
            derivative_wrt_water_saturation_at_cell=dgas_dw_cell,
            derivative_wrt_water_saturation_at_neighbour=dgas_dw_nbr,
            derivative_wrt_oil_saturation_at_cell=dgas_do_cell,
            derivative_wrt_oil_saturation_at_neighbour=dgas_do_nbr,
        ),
    )
    return (water_flux, oil_flux, gas_flux), flux_derivatives


def _clamp_tolerances(
    dtype: np.typing.DTypeLike,
    user_rtol: float,
    user_atol: float,
    b_norm: float,
) -> typing.Tuple[float, float]:
    """Clamp tolerances to avoid underflow for given dtype."""
    eps = np.finfo(dtype).eps
    min_rtol = max(1e-6, 10.0 * eps)
    min_atol = max(1e-10, 10.0 * eps * b_norm) if b_norm > 0 else 1e-10

    rtol_clamped = max(min_rtol, user_rtol)
    atol_clamped = max(min_atol, user_atol)
    return rtol_clamped, atol_clamped  # type: ignore


def evolve_saturation_implicitly(
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
    use_explicit_warmup: bool = True,
    warmup_steps: int = 20,
    max_warmup_time_step_size: float = Time(hours=24),
) -> EvolutionResult[
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]
]:
    """
    Implicit saturation solver with optional explicit warmup.

    Uses two-saturation formulation (S_w, S_o independent; S_g = 1 - S_w - S_o).

    Args:
        use_explicit_warmup: If True, run explicit steps first for better initial guess
        warmup_steps: Number of explicit sub-steps to take (with reduced timestep)

    Returns:
        EvolutionResult containing updated (S_w, S_o, S_g) grids
    """
    logger.info(f"Starting implicit saturation solve for timestep {time_step}")

    # Phase 1: Optional explicit warmup
    if use_explicit_warmup and warmup_steps > 0:
        logger.info(f"Running {warmup_steps} explicit warmup steps...")
        # Take small explicit steps
        warmup_dt = max(time_step_size / warmup_steps, max_warmup_time_step_size)

        for warmup_step in range(warmup_steps):
            result = evolve_saturation_explicitly(
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                elevation_grid=elevation_grid,
                time_step=time_step,
                time_step_size=warmup_dt,
                rock_properties=rock_properties,
                fluid_properties=fluid_properties,
                rock_fluid_properties=rock_fluid_properties,
                relative_mobility_grids=relative_mobility_grids,
                capillary_pressure_grids=capillary_pressure_grids,
                wells=wells,
                options=options,
                injection_grid=injection_grid,
                production_grid=production_grid,
            )

            # Update fluid properties for next sub-step (copy into existing arrays)
            fluid_properties.water_saturation_grid[:] = result.value[0]
            fluid_properties.oil_saturation_grid[:] = result.value[1]
            fluid_properties.gas_saturation_grid[:] = result.value[2]

            logger.debug(f"  Warmup step {warmup_step + 1}/{warmup_steps} complete")

        logger.info("Explicit warmup complete, starting implicit solve...")

    # Phase 2: Implicit solve with two-saturation formulation
    return _evolve_saturation_implicitly_two_saturation(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        rock_fluid_properties=rock_fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        wells=wells,
        options=options,
        injection_grid=injection_grid,
        production_grid=production_grid,
    )


def _evolve_saturation_implicitly_two_saturation(
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
    Core implicit solver using two-saturation formulation.
    Solves for S_w and S_o only; computes S_g = 1 - S_w - S_o.
    """
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid
    oil_density_grid = fluid_properties.oil_effective_density_grid
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
    relative_permeability_table = rock_fluid_properties.relative_permeability_table

    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid

    water_viscosity_grid = fluid_properties.water_viscosity_grid
    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    max_iterations = options.max_iterations
    tolerance = options.convergence_tolerance

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

    # Initialize solution with current saturations (only S_w and S_o)
    updated_water_saturation_grid = current_water_saturation_grid.copy().astype(
        np.float64
    )
    updated_oil_saturation_grid = current_oil_saturation_grid.copy().astype(np.float64)

    # Count interior cells
    n_interior_cells = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    n_equations = 2 * n_interior_cells  # Only 2 equations per cell (water and oil)

    logger.info(f"Solving system: {n_interior_cells} cells, {n_equations} equations")

    def get_equation_index(i: int, j: int, k: int, phase: int) -> int:
        """Map (i,j,k,phase) to linear equation index. phase: 0=water, 1=oil"""
        interior_i = i - 1
        interior_j = j - 1
        interior_k = k - 1
        cell_offset = (
            interior_i * (cell_count_y - 2) * (cell_count_z - 2)
            + interior_j * (cell_count_z - 2)
            + interior_k
        )
        return 2 * cell_offset + phase

    # Newton-Raphson iteration
    initial_residual = 0.0

    for iteration in range(max_iterations):
        # Recompute relative mobilities with current iterate
        # (This is necessary because kr = kr(S))
        water_relative_mobility_grid = np.zeros_like(
            updated_water_saturation_grid, dtype=np.float64
        )
        oil_relative_mobility_grid = np.zeros_like(
            updated_oil_saturation_grid, dtype=np.float64
        )
        gas_relative_mobility_grid = np.zeros_like(
            updated_water_saturation_grid, dtype=np.float64
        )

        for i, j, k in itertools.product(
            range(cell_count_x),
            range(cell_count_y),
            range(cell_count_z),
        ):
            # Compute gas saturation
            gas_sat = max(
                0.0,
                1.0
                - updated_water_saturation_grid[i, j, k]
                - updated_oil_saturation_grid[i, j, k],
            )

            rel_perms = relative_permeability_table(
                water_saturation=updated_water_saturation_grid[i, j, k],
                oil_saturation=updated_oil_saturation_grid[i, j, k],
                gas_saturation=gas_sat,
                connate_water_saturation=irreducible_water_saturation_grid[i, j, k],
                residual_oil_saturation_water=residual_oil_saturation_water_grid[
                    i, j, k
                ],
                residual_oil_saturation_gas=residual_oil_saturation_gas_grid[i, j, k],
                residual_gas_saturation=residual_gas_saturation_grid[i, j, k],
            )

            water_relative_mobility_grid[i, j, k] = (
                rel_perms["water"] / water_viscosity_grid[i, j, k]
                if water_viscosity_grid[i, j, k] > 0
                else 0.0
            )
            oil_relative_mobility_grid[i, j, k] = (
                rel_perms["oil"] / oil_viscosity_grid[i, j, k]
                if oil_viscosity_grid[i, j, k] > 0
                else 0.0
            )
            gas_relative_mobility_grid[i, j, k] = (
                rel_perms["gas"] / gas_viscosity_grid[i, j, k]
                if gas_viscosity_grid[i, j, k] > 0
                else 0.0
            )

        # Build residual vector and Jacobian
        residual = np.zeros(n_equations, dtype=np.float64)
        jacobian_dict = {}  # (row, col) -> value

        def add_jacobian_entry(row_idx: int, col_idx: int, value: float):
            """Add entry to Jacobian dictionary."""
            key = (row_idx, col_idx)
            jacobian_dict[key] = jacobian_dict.get(key, 0.0) + float(value)

        # Iterate over interior cells
        for i, j, k in itertools.product(
            range(1, cell_count_x - 1),
            range(1, cell_count_y - 1),
            range(1, cell_count_z - 1),
        ):
            cell_temperature = fluid_properties.temperature_grid[i, j, k]
            cell_thickness = thickness_grid[i, j, k]
            cell_total_volume = cell_size_x * cell_size_y * cell_thickness
            cell_porosity = porosity_grid[i, j, k]
            cell_pore_volume = cell_total_volume * cell_porosity

            cell_oil_saturation = current_oil_saturation_grid[i, j, k]
            cell_water_saturation = current_water_saturation_grid[i, j, k]
            cell_oil_pressure = current_oil_pressure_grid[i, j, k]

            # Get equation indices
            water_eq_idx = get_equation_index(i, j, k, 0)
            oil_eq_idx = get_equation_index(i, j, k, 1)

            # Configure flux computation for each direction
            flux_configurations = {
                "x": {
                    "neighbours": [(i + 1, j, k), (i - 1, j, k)],
                    "flow_area": cell_size_y * cell_thickness,
                    "flow_length": cell_size_x,
                    "abs_perm_multiplier": (
                        absolute_permeability.x[i, j, k]
                        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
                    ),
                },
                "y": {
                    "neighbours": [(i, j - 1, k), (i, j + 1, k)],
                    "flow_area": cell_size_x * cell_thickness,
                    "flow_length": cell_size_y,
                    "abs_perm_multiplier": (
                        absolute_permeability.y[i, j, k]
                        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
                    ),
                },
                "z": {
                    "neighbours": [(i, j, k - 1), (i, j, k + 1)],
                    "flow_area": cell_size_x * cell_size_y,
                    "flow_length": cell_thickness,
                    "abs_perm_multiplier": (
                        absolute_permeability.z[i, j, k]
                        * c.MILLIDARCIES_PER_CENTIPOISE_TO_FT2_PER_PSI_PER_DAY
                    ),
                },
            }

            net_water_flux = 0.0
            net_oil_flux = 0.0
            net_gas_flux = 0.0

            # Compute fluxes from all neighbours
            for direction, config in flux_configurations.items():
                flow_area = config["flow_area"]
                flow_length = config["flow_length"]
                abs_perm_mult = config["abs_perm_multiplier"]

                for neighbour in config["neighbours"]:  # type: ignore[arg-type]
                    # Compute implicit fluxes and derivatives
                    (
                        (water_flux, oil_flux, gas_flux),
                        flux_derivatives,
                    ) = _compute_implicit_fluxes_and_derivatives_two_saturation(
                        cell_indices=(i, j, k),
                        neighbour_indices=neighbour,
                        flow_area=flow_area,  # type: ignore[arg-type]
                        flow_length=flow_length,  # type: ignore[arg-type]
                        oil_pressure_grid=current_oil_pressure_grid,
                        water_saturation_grid=updated_water_saturation_grid,
                        oil_saturation_grid=updated_oil_saturation_grid,
                        oil_viscosity_grid=oil_viscosity_grid,
                        water_viscosity_grid=water_viscosity_grid,
                        gas_viscosity_grid=gas_viscosity_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
                        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
                        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
                        residual_gas_saturation_grid=residual_gas_saturation_grid,
                        relative_permeability_table=relative_permeability_table,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        absolute_permeability_multiplier=abs_perm_mult,  # type: ignore[arg-type]
                    )

                    net_water_flux += water_flux
                    net_oil_flux += oil_flux
                    net_gas_flux += gas_flux

                    # Time discretization factor for Jacobian
                    time_factor = -time_step_in_days / cell_pore_volume

                    neighbour_i, neighbour_j, neighbour_k = neighbour

                    water_derivs = flux_derivatives.water_phase_flux_derivatives
                    oil_derivs = flux_derivatives.oil_phase_flux_derivatives

                    # Add derivatives w.r.t. current cell saturations
                    add_jacobian_entry(
                        water_eq_idx,
                        water_eq_idx,
                        time_factor
                        * water_derivs.derivative_wrt_water_saturation_at_cell,
                    )
                    add_jacobian_entry(
                        water_eq_idx,
                        oil_eq_idx,
                        time_factor
                        * water_derivs.derivative_wrt_oil_saturation_at_cell,
                    )

                    add_jacobian_entry(
                        oil_eq_idx,
                        water_eq_idx,
                        time_factor
                        * oil_derivs.derivative_wrt_water_saturation_at_cell,
                    )
                    add_jacobian_entry(
                        oil_eq_idx,
                        oil_eq_idx,
                        time_factor * oil_derivs.derivative_wrt_oil_saturation_at_cell,
                    )

                    # Add derivatives w.r.t. neighbour saturations (if neighbour is interior)
                    if (
                        1 <= neighbour_i < cell_count_x - 1
                        and 1 <= neighbour_j < cell_count_y - 1
                        and 1 <= neighbour_k < cell_count_z - 1
                    ):
                        water_eq_idx_nbr = get_equation_index(
                            neighbour_i, neighbour_j, neighbour_k, 0
                        )
                        oil_eq_idx_nbr = get_equation_index(
                            neighbour_i, neighbour_j, neighbour_k, 1
                        )

                        add_jacobian_entry(
                            water_eq_idx,
                            water_eq_idx_nbr,
                            time_factor
                            * water_derivs.derivative_wrt_water_saturation_at_neighbour,
                        )
                        add_jacobian_entry(
                            water_eq_idx,
                            oil_eq_idx_nbr,
                            time_factor
                            * water_derivs.derivative_wrt_oil_saturation_at_neighbour,
                        )

                        add_jacobian_entry(
                            oil_eq_idx,
                            water_eq_idx_nbr,
                            time_factor
                            * oil_derivs.derivative_wrt_water_saturation_at_neighbour,
                        )
                        add_jacobian_entry(
                            oil_eq_idx,
                            oil_eq_idx_nbr,
                            time_factor
                            * oil_derivs.derivative_wrt_oil_saturation_at_neighbour,
                        )

            # Compute well contributions
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

            # Injection wells
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
                fluid_formation_volume_factor = (
                    injected_fluid.get_formation_volume_factor(
                        pressure=cell_oil_pressure,
                        temperature=cell_temperature,
                    )
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

                if cell_injection_rate < 0.0:
                    if injection_well.auto_clamp:
                        cell_injection_rate = 0.0
                    else:
                        logger.warning(
                            f"Injector {injection_well.name} at ({i},{j},{k}) is producing: {cell_injection_rate:.2e}"
                        )

                if injected_phase == FluidPhase.GAS:
                    cell_gas_injection_rate = cell_injection_rate
                    if injection_grid is not None:
                        injection_grid[i, j, k] = (0.0, 0.0, cell_gas_injection_rate)
                else:
                    cell_water_injection_rate = cell_injection_rate * c.BBL_TO_FT3
                    if injection_grid is not None:
                        injection_grid[i, j, k] = (0.0, cell_water_injection_rate, 0.0)

            # Production wells
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
                        fluid_formation_volume_factor = (
                            gas_formation_volume_factor_grid[i, j, k]
                        )
                    elif produced_phase == FluidPhase.WATER:
                        phase_mobility = water_relative_mobility_grid[i, j, k]
                        fluid_compressibility = water_compressibility_grid[i, j, k]
                        fluid_formation_volume_factor = (
                            water_formation_volume_factor_grid[i, j, k]
                        )
                    else:
                        phase_mobility = oil_relative_mobility_grid[i, j, k]
                        fluid_compressibility = oil_compressibility_grid[i, j, k]
                        fluid_formation_volume_factor = (
                            oil_formation_volume_factor_grid[i, j, k]
                        )

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

                    if production_rate > 0.0:
                        if production_well.auto_clamp:
                            production_rate = 0.0
                        else:
                            logger.warning(
                                f"Producer {production_well.name} at ({i},{j},{k}) is injecting: {production_rate:.2e}"
                            )

                    if produced_fluid.phase == FluidPhase.GAS:
                        cell_gas_production_rate += production_rate
                    elif produced_fluid.phase == FluidPhase.WATER:
                        cell_water_production_rate += production_rate * c.BBL_TO_FT3
                    else:
                        cell_oil_production_rate += production_rate * c.BBL_TO_FT3

                if production_grid is not None:
                    production_grid[i, j, k] = (
                        cell_oil_production_rate,
                        cell_water_production_rate,
                        cell_gas_production_rate,
                    )

            net_water_flow_rate = cell_water_injection_rate + cell_water_production_rate
            net_oil_flow_rate = cell_oil_injection_rate + cell_oil_production_rate
            net_gas_flow_rate = cell_gas_injection_rate + cell_gas_production_rate

            # Compute residuals
            # R_w = S_w^(n+1) - S_w^n - Δt/(φ*V) * [F_w + Q_w]
            # R_o = S_o^(n+1) - S_o^n - Δt/(φ*V) * [F_o + Q_o]
            water_residual = (
                updated_water_saturation_grid[i, j, k]
                - cell_water_saturation
                - (net_water_flux + net_water_flow_rate)
                * time_step_in_days
                / cell_pore_volume
            )
            oil_residual = (
                updated_oil_saturation_grid[i, j, k]
                - cell_oil_saturation
                - (net_oil_flux + net_oil_flow_rate)
                * time_step_in_days
                / cell_pore_volume
            )

            residual[water_eq_idx] = water_residual
            residual[oil_eq_idx] = oil_residual

            # Add accumulation term to diagonal: dR/dS = 1.0
            add_jacobian_entry(water_eq_idx, water_eq_idx, 1.0)
            add_jacobian_entry(oil_eq_idx, oil_eq_idx, 1.0)

        # Convert Jacobian dictionary to sparse matrix
        jacobian_rows = []
        jacobian_cols = []
        jacobian_data = []

        for (row, col), value in jacobian_dict.items():
            jacobian_rows.append(row)
            jacobian_cols.append(col)
            jacobian_data.append(float(value))

        jacobian = csr_matrix(
            (jacobian_data, (jacobian_rows, jacobian_cols)),
            shape=(n_equations, n_equations),
            dtype=np.float64,
        )

        # Check Jacobian properties on first iteration
        if iteration == 0:
            nnz = jacobian.nnz
            density = nnz / (n_equations * n_equations)
            logger.info(
                f"  Jacobian: {n_equations}x{n_equations}, nnz={nnz}, density={density:.2e}"
            )

            row_sums = np.abs(jacobian).sum(axis=1).A1  # type: ignore[attr-defined]
            zero_rows = np.where(row_sums < 1e-15)[0]
            if len(zero_rows) > 0:
                logger.warning(f"  WARNING: {len(zero_rows)} zero rows in Jacobian!")

            diag_vals = jacobian.diagonal()
            logger.info(
                f"  Diagonal range: [{np.min(diag_vals):.2e}, {np.max(diag_vals):.2e}]"
            )
            zero_diag = np.sum(np.abs(diag_vals) < 1e-15)
            if zero_diag > 0:
                logger.warning(f"  WARNING: {zero_diag} zero diagonal entries!")

        # Check convergence
        residual_norm = np.linalg.norm(residual)
        max_residual = np.max(np.abs(residual))

        if iteration == 0:
            initial_residual = float(residual_norm)
            logger.info(
                f"  Iteration {iteration}: ||R|| = {residual_norm:.2e}, max|R| = {max_residual:.2e}"
            )
        else:
            reduction = (
                residual_norm / initial_residual if initial_residual > 0 else 0.0
            )
            logger.info(
                f"  Iteration {iteration}: ||R|| = {residual_norm:.2e}, max|R| = {max_residual:.2e}, "
                f"reduction = {reduction:.2e}"
            )

        # Adaptive tolerance
        effective_tolerance = max(tolerance, initial_residual * 1e-3)
        print("Effective tolerance: ", effective_tolerance)
        print("Residual Norm: ", residual_norm)

        if residual_norm < effective_tolerance:
            logger.info(
                f"Converged at iteration {iteration}: ||R|| = {residual_norm:.2e} < {effective_tolerance:.2e}"
            )
            break

        # Check for stalling
        if iteration > 10:
            recent_progress = abs(residual_norm - initial_residual) / max(
                initial_residual, 1e-10
            )
            if recent_progress < 1e-4 and iteration > 50:
                logger.warning(
                    f"Newton iteration stalling after {iteration} iterations"
                )
                logger.warning(
                    f"  Residual: {residual_norm:.2e}, progress: {recent_progress:.2e}"
                )
                break

        if iteration == max_iterations - 1:
            logger.error(f"Failed to converge after {max_iterations} iterations")
            logger.error(
                f"  Final ||R|| = {residual_norm:.2e}, tolerance = {effective_tolerance:.2e}"
            )
            raise RuntimeError(
                f"Newton-Raphson failed to converge. Final residual: {residual_norm:.2e}. "
                f"Consider reducing time step or using explicit warmup."
            )

        # Solve linear system: J * ΔS = -R
        try:
            jacobian_csr = jacobian.tocsr()
            rhs = -residual
            rhs_norm = float(np.linalg.norm(rhs))

            # Clamp tolerances for numerical stability
            arr_dtype = np.float64
            rtol_clamped, atol_clamped = _clamp_tolerances(
                dtype=arr_dtype,  # type: ignore[arg-type]
                user_rtol=1e-3,
                user_atol=1e-5,
                b_norm=rhs_norm,
            )

            logger.debug(f"  Linear solve: ||b|| = {rhs_norm:.2e}")
            logger.debug(
                f"  Tolerances: rtol={rtol_clamped:.2e}, atol={atol_clamped:.2e}"
            )

            # Build AMG preconditioner
            try:
                ml = pyamg.ruge_stuben_solver(jacobian_csr)
                M = ml.aspreconditioner(cycle="V")
                logger.debug("  AMG preconditioner built")
            except Exception as e:
                logger.warning(f"  AMG failed: {e}, using no preconditioner")
                M = None

            # GMRES solve
            delta_saturation, gmres_info = gmres(
                A=jacobian_csr,
                b=rhs,
                M=M,
                rtol=rtol_clamped,
                atol=atol_clamped,
                restart=50,
                maxiter=max_iterations,
            )

            if gmres_info > 0:
                logger.warning(f"  GMRES did not converge: {gmres_info} iterations")
            elif gmres_info < 0:
                raise RuntimeError(f"GMRES illegal input or breakdown: {gmres_info}")
            else:
                logger.debug("  GMRES converged")

        except Exception as e:
            logger.error(f"Linear solver failed at iteration {iteration}: {e}")
            raise RuntimeError(f"Linear solver failed: {e}")

        # Check for valid solution
        if not np.isfinite(delta_saturation).all():
            logger.error(
                f"Linear solver produced non-finite values at iteration {iteration}"
            )
            raise RuntimeError("Linear solver produced non-finite values")

        delta_norm = np.linalg.norm(delta_saturation)
        max_change = np.max(np.abs(delta_saturation))

        if iteration % 10 == 0 or iteration < 3:
            logger.info(f"  ||ΔS|| = {delta_norm:.2e}, max|ΔS| = {max_change:.2e}")

        if delta_norm < 1e-10:
            logger.warning(f"Newton step essentially zero: ||ΔS|| = {delta_norm:.2e}")
            logger.warning("  Possible singular Jacobian or fixed point")
            break

        # Backtracking line search
        damping_factor = 1.0
        min_damping = 0.01
        backtrack_factor = 0.5
        max_backtrack_iterations = 5

        best_damping = damping_factor

        for backtrack_iter in range(max_backtrack_iterations):
            # Try this damping factor
            test_water = updated_water_saturation_grid.copy()
            test_oil = updated_oil_saturation_grid.copy()

            # Apply damped update
            for i, j, k in itertools.product(
                range(1, cell_count_x - 1),
                range(1, cell_count_y - 1),
                range(1, cell_count_z - 1),
            ):
                water_eq_idx = get_equation_index(i, j, k, 0)
                oil_eq_idx = get_equation_index(i, j, k, 1)

                test_water[i, j, k] += damping_factor * delta_saturation[water_eq_idx]
                test_oil[i, j, k] += damping_factor * delta_saturation[oil_eq_idx]

                # Clamp to [0, 1]
                test_water[i, j, k] = np.clip(test_water[i, j, k], 0.0, 1.0)
                test_oil[i, j, k] = np.clip(test_oil[i, j, k], 0.0, 1.0)

                # Enforce S_w + S_o <= 1
                if test_water[i, j, k] + test_oil[i, j, k] > 1.0:
                    total = test_water[i, j, k] + test_oil[i, j, k]
                    test_water[i, j, k] /= total
                    test_oil[i, j, k] /= total

            # Check validity
            saturation_valid = (
                np.all(test_water >= 0.0)
                and np.all(test_water <= 1.0)
                and np.all(test_oil >= 0.0)
                and np.all(test_oil <= 1.0)
                and np.all(test_water + test_oil <= 1.0 + 1e-6)
            )

            if saturation_valid:
                best_damping = damping_factor
                if backtrack_iter > 0:
                    logger.debug(
                        f"  Backtracking accepted damping={damping_factor:.3f} after {backtrack_iter} steps"
                    )
                break

            # Reduce damping
            damping_factor *= backtrack_factor
            if damping_factor < min_damping:
                damping_factor = min_damping
                logger.warning(
                    f"  Backtracking reached minimum damping={min_damping:.3f}"
                )
                break

        damping_factor = best_damping

        if iteration % 10 == 0 or damping_factor < 1.0:
            logger.info(f"  Applying damping factor: {damping_factor:.3f}")

        # Track saturation statistics
        if iteration % 10 == 0:
            water_range = (
                np.min(updated_water_saturation_grid[1:-1, 1:-1, 1:-1]),
                np.max(updated_water_saturation_grid[1:-1, 1:-1, 1:-1]),
            )
            oil_range = (
                np.min(updated_oil_saturation_grid[1:-1, 1:-1, 1:-1]),
                np.max(updated_oil_saturation_grid[1:-1, 1:-1, 1:-1]),
            )
            logger.info(
                f"  Saturation ranges: S_w=[{water_range[0]:.3f}, {water_range[1]:.3f}], "
                f"S_o=[{oil_range[0]:.3f}, {oil_range[1]:.3f}]"
            )

        # Update saturations with damped Newton step
        for i, j, k in itertools.product(
            range(1, cell_count_x - 1),
            range(1, cell_count_y - 1),
            range(1, cell_count_z - 1),
        ):
            water_eq_idx = get_equation_index(i, j, k, 0)
            oil_eq_idx = get_equation_index(i, j, k, 1)

            updated_water_saturation_grid[i, j, k] += (
                damping_factor * delta_saturation[water_eq_idx]
            )
            updated_oil_saturation_grid[i, j, k] += (
                damping_factor * delta_saturation[oil_eq_idx]
            )

        # Apply saturation constraints
        updated_water_saturation_grid = np.clip(updated_water_saturation_grid, 0.0, 1.0)
        updated_oil_saturation_grid = np.clip(updated_oil_saturation_grid, 0.0, 1.0)

        # Enforce S_w + S_o <= 1
        total_wo = updated_water_saturation_grid + updated_oil_saturation_grid
        mask = total_wo > 1.0
        if np.any(mask):
            updated_water_saturation_grid[mask] /= total_wo[mask]
            updated_oil_saturation_grid[mask] /= total_wo[mask]

    # Final cleanup and compute gas saturation
    updated_water_saturation_grid = np.clip(updated_water_saturation_grid, 0.0, 1.0)
    updated_oil_saturation_grid = np.clip(updated_oil_saturation_grid, 0.0, 1.0)

    # Compute gas saturation: S_g = 1 - S_w - S_o
    updated_gas_saturation_grid = np.maximum(
        0.0, 1.0 - updated_water_saturation_grid - updated_oil_saturation_grid
    )

    # Final normalization to ensure S_w + S_o + S_g = 1
    total_saturation_grid = (
        updated_water_saturation_grid
        + updated_oil_saturation_grid
        + updated_gas_saturation_grid
    )
    mask = total_saturation_grid > c.SATURATION_EPSILON
    if np.any(mask):
        updated_water_saturation_grid[mask] /= total_saturation_grid[mask]
        updated_oil_saturation_grid[mask] /= total_saturation_grid[mask]
        updated_gas_saturation_grid[mask] /= total_saturation_grid[mask]

    logger.info("Implicit saturation solve complete")

    return EvolutionResult(
        (
            updated_water_saturation_grid,
            updated_oil_saturation_grid,
            updated_gas_saturation_grid,
        ),
        scheme="implicit",
    )


def _compute_implicit_miscible_saturation_phase_fluxes_and_derivatives(
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
    solvent_concentration_grid: ThreeDimensionalGrid,
    oil_viscosity_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    irreducible_water_saturation_grid: ThreeDimensionalGrid,
    residual_oil_saturation_water_grid: ThreeDimensionalGrid,
    residual_oil_saturation_gas_grid: ThreeDimensionalGrid,
    residual_gas_saturation_grid: ThreeDimensionalGrid,
    relative_permeability_table: RelativePermeabilityTable,
    oil_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    water_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    gas_density_grid: typing.Optional[ThreeDimensionalGrid] = None,
    elevation_grid: typing.Optional[ThreeDimensionalGrid] = None,
) -> typing.Tuple[
    typing.Tuple[float, float, float, float],  # Fluxes: (water, oil, gas, solvent)
    MiscibleFluxDerivativesWithRespectToSaturations,  # Flux derivatives for Jacobian
]:
    """
    Compute miscible phase fluxes and their derivatives with respect to saturations
    and solvent concentration for implicit formulation.

    The mobility grids passed to this function contain mobilities computed from
    the current saturation iterate. However, to compute derivatives ∂F/∂S and ∂F/∂C,
    we need to recompute mobilities with perturbed saturations/concentration using
    the relative permeability table. This is why relative_permeability_table is
    required even though mobility grids are provided.

    Returns:
        fluxes: (water_flux, oil_flux, gas_flux, solvent_mass_flux_in_oil)
        derivatives: MiscibleFluxDerivativesWithRespectToSaturations containing ∂F/∂S and ∂F/∂C terms
    """
    # First, compute the base fluxes using the same logic as explicit
    cell_oil_pressure = oil_pressure_grid[cell_indices]
    cell_oil_water_capillary_pressure = oil_water_capillary_pressure_grid[cell_indices]
    cell_gas_oil_capillary_pressure = gas_oil_capillary_pressure_grid[cell_indices]

    # Neighbour cell pressures
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
        elevation_delta = (
            elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
        )
    else:
        elevation_delta = 0.0

    # Determine upwind densities
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

    # Compute phase potentials with gravity
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

    # Determine upwind mobilities (implicit: use current iterate saturations)
    upwind_water_mobility = (
        water_mobility_grid[neighbour_indices]
        if water_phase_potential > 0.0
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

    # Compute velocities
    water_velocity = upwind_water_mobility * water_phase_potential / flow_length
    oil_velocity = upwind_oil_mobility * oil_phase_potential / flow_length
    gas_velocity = upwind_gas_mobility * gas_phase_potential / flow_length

    # Compute volumetric fluxes
    water_volumetric_flux = water_velocity * flow_area
    oil_volumetric_flux = oil_velocity * flow_area
    gas_volumetric_flux = gas_velocity * flow_area

    # Upwind solvent concentration (moves with oil)
    cell_solvent_concentration = solvent_concentration_grid[cell_indices]
    neighbour_solvent_concentration = solvent_concentration_grid[neighbour_indices]

    upwinded_solvent_concentration = (
        neighbour_solvent_concentration
        if oil_velocity > 0
        else cell_solvent_concentration
    )

    # Solvent mass flux in oil phase
    solvent_mass_flux_in_oil = oil_volumetric_flux * upwinded_solvent_concentration

    # Now compute derivatives using numerical differentiation
    # We need ∂F/∂S_w, ∂F/∂S_o, ∂F/∂S_g, ∂F/∂C for each of cell and neighbour

    perturbation = 1e-7  # Small perturbation for numerical derivatives

    # Get current saturations
    cell_water_saturation = water_saturation_grid[cell_indices]
    cell_oil_saturation = oil_saturation_grid[cell_indices]
    cell_gas_saturation = gas_saturation_grid[cell_indices]

    neighbour_water_saturation = water_saturation_grid[neighbour_indices]
    neighbour_oil_saturation = oil_saturation_grid[neighbour_indices]
    neighbour_gas_saturation = gas_saturation_grid[neighbour_indices]

    # Get residual saturations
    cell_connate_water_saturation = irreducible_water_saturation_grid[cell_indices]
    cell_residual_oil_saturation_water = residual_oil_saturation_water_grid[
        cell_indices
    ]
    cell_residual_oil_saturation_gas = residual_oil_saturation_gas_grid[cell_indices]
    cell_residual_gas_saturation = residual_gas_saturation_grid[cell_indices]

    neighbour_connate_water_saturation = irreducible_water_saturation_grid[
        neighbour_indices
    ]
    neighbour_residual_oil_saturation_water = residual_oil_saturation_water_grid[
        neighbour_indices
    ]
    neighbour_residual_oil_saturation_gas = residual_oil_saturation_gas_grid[
        neighbour_indices
    ]
    neighbour_residual_gas_saturation = residual_gas_saturation_grid[neighbour_indices]

    # Get viscosities
    cell_water_viscosity = water_viscosity_grid[cell_indices]
    cell_oil_viscosity = oil_viscosity_grid[cell_indices]
    cell_gas_viscosity = gas_viscosity_grid[cell_indices]

    neighbour_water_viscosity = water_viscosity_grid[neighbour_indices]
    neighbour_oil_viscosity = oil_viscosity_grid[neighbour_indices]
    neighbour_gas_viscosity = gas_viscosity_grid[neighbour_indices]

    # Initialize derivative accumulators for all phases
    water_derivative_wrt_water_saturation_cell = 0.0
    water_derivative_wrt_water_saturation_neighbour = 0.0
    water_derivative_wrt_oil_saturation_cell = 0.0
    water_derivative_wrt_oil_saturation_neighbour = 0.0
    water_derivative_wrt_gas_saturation_cell = 0.0
    water_derivative_wrt_gas_saturation_neighbour = 0.0
    water_derivative_wrt_solvent_concentration_cell = 0.0
    water_derivative_wrt_solvent_concentration_neighbour = 0.0

    oil_derivative_wrt_water_saturation_cell = 0.0
    oil_derivative_wrt_water_saturation_neighbour = 0.0
    oil_derivative_wrt_oil_saturation_cell = 0.0
    oil_derivative_wrt_oil_saturation_neighbour = 0.0
    oil_derivative_wrt_gas_saturation_cell = 0.0
    oil_derivative_wrt_gas_saturation_neighbour = 0.0
    oil_derivative_wrt_solvent_concentration_cell = 0.0
    oil_derivative_wrt_solvent_concentration_neighbour = 0.0

    gas_derivative_wrt_water_saturation_cell = 0.0
    gas_derivative_wrt_water_saturation_neighbour = 0.0
    gas_derivative_wrt_oil_saturation_cell = 0.0
    gas_derivative_wrt_oil_saturation_neighbour = 0.0
    gas_derivative_wrt_gas_saturation_cell = 0.0
    gas_derivative_wrt_gas_saturation_neighbour = 0.0
    gas_derivative_wrt_solvent_concentration_cell = 0.0
    gas_derivative_wrt_solvent_concentration_neighbour = 0.0

    solvent_derivative_wrt_water_saturation_cell = 0.0
    solvent_derivative_wrt_water_saturation_neighbour = 0.0
    solvent_derivative_wrt_oil_saturation_cell = 0.0
    solvent_derivative_wrt_oil_saturation_neighbour = 0.0
    solvent_derivative_wrt_gas_saturation_cell = 0.0
    solvent_derivative_wrt_gas_saturation_neighbour = 0.0
    solvent_derivative_wrt_solvent_concentration_cell = 0.0
    solvent_derivative_wrt_solvent_concentration_neighbour = 0.0

    # Helper function to compute flux with perturbed saturations/concentration
    def _compute_flux_with_saturations(
        water_saturation_at_cell: float,
        oil_saturation_at_cell: float,
        gas_saturation_at_cell: float,
        solvent_concentration_at_cell: float,
        water_saturation_at_neighbour: float,
        oil_saturation_at_neighbour: float,
        gas_saturation_at_neighbour: float,
        solvent_concentration_at_neighbour: float,
    ) -> typing.Tuple[float, float, float, float]:
        """Compute phase fluxes given saturations and solvent concentration at cell and neighbour."""
        # Clamp saturations to valid range [0, 1]
        water_saturation_at_cell = max(0.0, min(1.0, water_saturation_at_cell))
        oil_saturation_at_cell = max(0.0, min(1.0, oil_saturation_at_cell))
        gas_saturation_at_cell = max(0.0, min(1.0, gas_saturation_at_cell))
        solvent_concentration_at_cell = max(
            0.0, min(1.0, solvent_concentration_at_cell)
        )
        water_saturation_at_neighbour = max(
            0.0, min(1.0, water_saturation_at_neighbour)
        )
        oil_saturation_at_neighbour = max(0.0, min(1.0, oil_saturation_at_neighbour))
        gas_saturation_at_neighbour = max(0.0, min(1.0, gas_saturation_at_neighbour))
        solvent_concentration_at_neighbour = max(
            0.0, min(1.0, solvent_concentration_at_neighbour)
        )

        # Compute relative permeabilities using the table
        relative_permeabilities_at_cell = relative_permeability_table(
            water_saturation=water_saturation_at_cell,
            oil_saturation=oil_saturation_at_cell,
            gas_saturation=gas_saturation_at_cell,
            connate_water_saturation=cell_connate_water_saturation,
            residual_oil_saturation_water=cell_residual_oil_saturation_water,
            residual_oil_saturation_gas=cell_residual_oil_saturation_gas,
            residual_gas_saturation=cell_residual_gas_saturation,
        )
        relative_permeabilities_at_neighbour = relative_permeability_table(
            water_saturation=water_saturation_at_neighbour,
            oil_saturation=oil_saturation_at_neighbour,
            gas_saturation=gas_saturation_at_neighbour,
            connate_water_saturation=neighbour_connate_water_saturation,
            residual_oil_saturation_water=neighbour_residual_oil_saturation_water,
            residual_oil_saturation_gas=neighbour_residual_oil_saturation_gas,
            residual_gas_saturation=neighbour_residual_gas_saturation,
        )

        # Compute mobilities (λ = kr / μ, absorbing absolute perm into mobility)
        # Note: Absolute permeability is assumed constant and cancels in derivatives
        water_mobility_at_cell = (
            relative_permeabilities_at_cell["water"] / cell_water_viscosity
            if cell_water_viscosity > 0
            else 0.0
        )
        oil_mobility_at_cell = (
            relative_permeabilities_at_cell["oil"] / cell_oil_viscosity
            if cell_oil_viscosity > 0
            else 0.0
        )
        gas_mobility_at_cell = (
            relative_permeabilities_at_cell["gas"] / cell_gas_viscosity
            if cell_gas_viscosity > 0
            else 0.0
        )

        water_mobility_at_neighbour = (
            relative_permeabilities_at_neighbour["water"] / neighbour_water_viscosity
            if neighbour_water_viscosity > 0
            else 0.0
        )
        oil_mobility_at_neighbour = (
            relative_permeabilities_at_neighbour["oil"] / neighbour_oil_viscosity
            if neighbour_oil_viscosity > 0
            else 0.0
        )
        gas_mobility_at_neighbour = (
            relative_permeabilities_at_neighbour["gas"] / neighbour_gas_viscosity
            if neighbour_gas_viscosity > 0
            else 0.0
        )

        # Upwind mobilities based on phase potentials
        water_mobility_upwind = (
            water_mobility_at_cell
            if water_phase_potential > 0.0
            else water_mobility_at_neighbour
        )
        oil_mobility_upwind = (
            oil_mobility_at_cell
            if oil_phase_potential > 0.0
            else oil_mobility_at_neighbour
        )
        gas_mobility_upwind = (
            gas_mobility_at_cell
            if gas_phase_potential > 0.0
            else gas_mobility_at_neighbour
        )

        # Compute velocities
        water_velocity = water_mobility_upwind * water_phase_potential / flow_length
        oil_velocity = oil_mobility_upwind * oil_phase_potential / flow_length
        gas_velocity = gas_mobility_upwind * gas_phase_potential / flow_length

        # Compute fluxes
        water_flux = water_velocity * flow_area
        oil_flux = oil_velocity * flow_area
        gas_flux = gas_velocity * flow_area

        # Upwind solvent concentration (moves with oil)
        upwinded_conc = (
            solvent_concentration_at_neighbour
            if oil_velocity > 0
            else solvent_concentration_at_cell
        )
        solvent_flux = oil_flux * upwinded_conc
        return water_flux, oil_flux, gas_flux, solvent_flux

    # Compute derivatives w.r.t. cell saturations using numerical differentiation
    # ∂F/∂S_w_cell
    (
        water_flux_perturbed,
        oil_flux_perturbed,
        gas_flux_perturbed,
        solvent_flux_perturbed,
    ) = _compute_flux_with_saturations(
        cell_water_saturation + perturbation,
        cell_oil_saturation,
        cell_gas_saturation,
        cell_solvent_concentration,
        neighbour_water_saturation,
        neighbour_oil_saturation,
        neighbour_gas_saturation,
        neighbour_solvent_concentration,
    )
    water_derivative_wrt_water_saturation_cell = (
        water_flux_perturbed - water_volumetric_flux
    ) / perturbation
    oil_derivative_wrt_water_saturation_cell = (
        oil_flux_perturbed - oil_volumetric_flux
    ) / perturbation
    gas_derivative_wrt_water_saturation_cell = (
        gas_flux_perturbed - gas_volumetric_flux
    ) / perturbation
    solvent_derivative_wrt_water_saturation_cell = (
        solvent_flux_perturbed - solvent_mass_flux_in_oil
    ) / perturbation

    # ∂F/∂S_o_cell
    (
        water_flux_perturbed,
        oil_flux_perturbed,
        gas_flux_perturbed,
        solvent_flux_perturbed,
    ) = _compute_flux_with_saturations(
        cell_water_saturation,
        cell_oil_saturation + perturbation,
        cell_gas_saturation,
        cell_solvent_concentration,
        neighbour_water_saturation,
        neighbour_oil_saturation,
        neighbour_gas_saturation,
        neighbour_solvent_concentration,
    )
    water_derivative_wrt_oil_saturation_cell = (
        water_flux_perturbed - water_volumetric_flux
    ) / perturbation
    oil_derivative_wrt_oil_saturation_cell = (
        oil_flux_perturbed - oil_volumetric_flux
    ) / perturbation
    gas_derivative_wrt_oil_saturation_cell = (
        gas_flux_perturbed - gas_volumetric_flux
    ) / perturbation
    solvent_derivative_wrt_oil_saturation_cell = (
        solvent_flux_perturbed - solvent_mass_flux_in_oil
    ) / perturbation

    # ∂F/∂S_g_cell
    (
        water_flux_perturbed,
        oil_flux_perturbed,
        gas_flux_perturbed,
        solvent_flux_perturbed,
    ) = _compute_flux_with_saturations(
        cell_water_saturation,
        cell_oil_saturation,
        cell_gas_saturation + perturbation,
        cell_solvent_concentration,
        neighbour_water_saturation,
        neighbour_oil_saturation,
        neighbour_gas_saturation,
        neighbour_solvent_concentration,
    )
    water_derivative_wrt_gas_saturation_cell = (
        water_flux_perturbed - water_volumetric_flux
    ) / perturbation
    oil_derivative_wrt_gas_saturation_cell = (
        oil_flux_perturbed - oil_volumetric_flux
    ) / perturbation
    gas_derivative_wrt_gas_saturation_cell = (
        gas_flux_perturbed - gas_volumetric_flux
    ) / perturbation
    solvent_derivative_wrt_gas_saturation_cell = (
        solvent_flux_perturbed - solvent_mass_flux_in_oil
    ) / perturbation

    # ∂F/∂C_cell
    (
        water_flux_perturbed,
        oil_flux_perturbed,
        gas_flux_perturbed,
        solvent_flux_perturbed,
    ) = _compute_flux_with_saturations(
        cell_water_saturation,
        cell_oil_saturation,
        cell_gas_saturation,
        cell_solvent_concentration + perturbation,
        neighbour_water_saturation,
        neighbour_oil_saturation,
        neighbour_gas_saturation,
        neighbour_solvent_concentration,
    )
    water_derivative_wrt_solvent_concentration_cell = (
        water_flux_perturbed - water_volumetric_flux
    ) / perturbation
    oil_derivative_wrt_solvent_concentration_cell = (
        oil_flux_perturbed - oil_volumetric_flux
    ) / perturbation
    gas_derivative_wrt_solvent_concentration_cell = (
        gas_flux_perturbed - gas_volumetric_flux
    ) / perturbation
    solvent_derivative_wrt_solvent_concentration_cell = (
        solvent_flux_perturbed - solvent_mass_flux_in_oil
    ) / perturbation

    # Compute derivatives w.r.t. neighbour saturations
    # ∂F/∂S_w_neighbour
    (
        water_flux_perturbed,
        oil_flux_perturbed,
        gas_flux_perturbed,
        solvent_flux_perturbed,
    ) = _compute_flux_with_saturations(
        cell_water_saturation,
        cell_oil_saturation,
        cell_gas_saturation,
        cell_solvent_concentration,
        neighbour_water_saturation + perturbation,
        neighbour_oil_saturation,
        neighbour_gas_saturation,
        neighbour_solvent_concentration,
    )
    water_derivative_wrt_water_saturation_neighbour = (
        water_flux_perturbed - water_volumetric_flux
    ) / perturbation
    oil_derivative_wrt_water_saturation_neighbour = (
        oil_flux_perturbed - oil_volumetric_flux
    ) / perturbation
    gas_derivative_wrt_water_saturation_neighbour = (
        gas_flux_perturbed - gas_volumetric_flux
    ) / perturbation
    solvent_derivative_wrt_water_saturation_neighbour = (
        solvent_flux_perturbed - solvent_mass_flux_in_oil
    ) / perturbation

    # ∂F/∂S_o_neighbour
    (
        water_flux_perturbed,
        oil_flux_perturbed,
        gas_flux_perturbed,
        solvent_flux_perturbed,
    ) = _compute_flux_with_saturations(
        cell_water_saturation,
        cell_oil_saturation,
        cell_gas_saturation,
        cell_solvent_concentration,
        neighbour_water_saturation,
        neighbour_oil_saturation + perturbation,
        neighbour_gas_saturation,
        neighbour_solvent_concentration,
    )
    water_derivative_wrt_oil_saturation_neighbour = (
        water_flux_perturbed - water_volumetric_flux
    ) / perturbation
    oil_derivative_wrt_oil_saturation_neighbour = (
        oil_flux_perturbed - oil_volumetric_flux
    ) / perturbation
    gas_derivative_wrt_oil_saturation_neighbour = (
        gas_flux_perturbed - gas_volumetric_flux
    ) / perturbation
    solvent_derivative_wrt_oil_saturation_neighbour = (
        solvent_flux_perturbed - solvent_mass_flux_in_oil
    ) / perturbation

    # ∂F/∂S_g_neighbour
    (
        water_flux_perturbed,
        oil_flux_perturbed,
        gas_flux_perturbed,
        solvent_flux_perturbed,
    ) = _compute_flux_with_saturations(
        cell_water_saturation,
        cell_oil_saturation,
        cell_gas_saturation,
        cell_solvent_concentration,
        neighbour_water_saturation,
        neighbour_oil_saturation,
        neighbour_gas_saturation + perturbation,
        neighbour_solvent_concentration,
    )
    water_derivative_wrt_gas_saturation_neighbour = (
        water_flux_perturbed - water_volumetric_flux
    ) / perturbation
    oil_derivative_wrt_gas_saturation_neighbour = (
        oil_flux_perturbed - oil_volumetric_flux
    ) / perturbation
    gas_derivative_wrt_gas_saturation_neighbour = (
        gas_flux_perturbed - gas_volumetric_flux
    ) / perturbation
    solvent_derivative_wrt_gas_saturation_neighbour = (
        solvent_flux_perturbed - solvent_mass_flux_in_oil
    ) / perturbation

    # ∂F/∂C_neighbour
    (
        water_flux_perturbed,
        oil_flux_perturbed,
        gas_flux_perturbed,
        solvent_flux_perturbed,
    ) = _compute_flux_with_saturations(
        cell_water_saturation,
        cell_oil_saturation,
        cell_gas_saturation,
        cell_solvent_concentration,
        neighbour_water_saturation,
        neighbour_oil_saturation,
        neighbour_gas_saturation,
        neighbour_solvent_concentration + perturbation,
    )
    water_derivative_wrt_solvent_concentration_neighbour = (
        water_flux_perturbed - water_volumetric_flux
    ) / perturbation
    oil_derivative_wrt_solvent_concentration_neighbour = (
        oil_flux_perturbed - oil_volumetric_flux
    ) / perturbation
    gas_derivative_wrt_solvent_concentration_neighbour = (
        gas_flux_perturbed - gas_volumetric_flux
    ) / perturbation
    solvent_derivative_wrt_solvent_concentration_neighbour = (
        solvent_flux_perturbed - solvent_mass_flux_in_oil
    ) / perturbation

    # Construct typed derivative structure
    flux_derivatives = MiscibleFluxDerivativesWithRespectToSaturations(
        water_phase_flux_derivatives=MisciblePhaseFluxDerivatives(
            derivative_wrt_water_saturation_at_cell=water_derivative_wrt_water_saturation_cell,
            derivative_wrt_water_saturation_at_neighbour=water_derivative_wrt_water_saturation_neighbour,
            derivative_wrt_oil_saturation_at_cell=water_derivative_wrt_oil_saturation_cell,
            derivative_wrt_oil_saturation_at_neighbour=water_derivative_wrt_oil_saturation_neighbour,
            derivative_wrt_gas_saturation_at_cell=water_derivative_wrt_gas_saturation_cell,
            derivative_wrt_gas_saturation_at_neighbour=water_derivative_wrt_gas_saturation_neighbour,
            derivative_wrt_solvent_concentration_at_cell=water_derivative_wrt_solvent_concentration_cell,
            derivative_wrt_solvent_concentration_at_neighbour=water_derivative_wrt_solvent_concentration_neighbour,
        ),
        oil_phase_flux_derivatives=MisciblePhaseFluxDerivatives(
            derivative_wrt_water_saturation_at_cell=oil_derivative_wrt_water_saturation_cell,
            derivative_wrt_water_saturation_at_neighbour=oil_derivative_wrt_water_saturation_neighbour,
            derivative_wrt_oil_saturation_at_cell=oil_derivative_wrt_oil_saturation_cell,
            derivative_wrt_oil_saturation_at_neighbour=oil_derivative_wrt_oil_saturation_neighbour,
            derivative_wrt_gas_saturation_at_cell=oil_derivative_wrt_gas_saturation_cell,
            derivative_wrt_gas_saturation_at_neighbour=oil_derivative_wrt_gas_saturation_neighbour,
            derivative_wrt_solvent_concentration_at_cell=oil_derivative_wrt_solvent_concentration_cell,
            derivative_wrt_solvent_concentration_at_neighbour=oil_derivative_wrt_solvent_concentration_neighbour,
        ),
        gas_phase_flux_derivatives=MisciblePhaseFluxDerivatives(
            derivative_wrt_water_saturation_at_cell=gas_derivative_wrt_water_saturation_cell,
            derivative_wrt_water_saturation_at_neighbour=gas_derivative_wrt_water_saturation_neighbour,
            derivative_wrt_oil_saturation_at_cell=gas_derivative_wrt_oil_saturation_cell,
            derivative_wrt_oil_saturation_at_neighbour=gas_derivative_wrt_oil_saturation_neighbour,
            derivative_wrt_gas_saturation_at_cell=gas_derivative_wrt_gas_saturation_cell,
            derivative_wrt_gas_saturation_at_neighbour=gas_derivative_wrt_gas_saturation_neighbour,
            derivative_wrt_solvent_concentration_at_cell=gas_derivative_wrt_solvent_concentration_cell,
            derivative_wrt_solvent_concentration_at_neighbour=gas_derivative_wrt_solvent_concentration_neighbour,
        ),
        solvent_mass_flux_derivatives=MisciblePhaseFluxDerivatives(
            derivative_wrt_water_saturation_at_cell=solvent_derivative_wrt_water_saturation_cell,
            derivative_wrt_water_saturation_at_neighbour=solvent_derivative_wrt_water_saturation_neighbour,
            derivative_wrt_oil_saturation_at_cell=solvent_derivative_wrt_oil_saturation_cell,
            derivative_wrt_oil_saturation_at_neighbour=solvent_derivative_wrt_oil_saturation_neighbour,
            derivative_wrt_gas_saturation_at_cell=solvent_derivative_wrt_gas_saturation_cell,
            derivative_wrt_gas_saturation_at_neighbour=solvent_derivative_wrt_gas_saturation_neighbour,
            derivative_wrt_solvent_concentration_at_cell=solvent_derivative_wrt_solvent_concentration_cell,
            derivative_wrt_solvent_concentration_at_neighbour=solvent_derivative_wrt_solvent_concentration_neighbour,
        ),
    )

    return (
        water_volumetric_flux,
        oil_volumetric_flux,
        gas_volumetric_flux,
        solvent_mass_flux_in_oil,
    ), flux_derivatives


def evolve_miscible_saturation_implicitly(
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
    Evolve saturations with Todd-Longstaff miscible displacement using
    implicit backward Euler time discretization with Newton-Raphson.

    Solvent (e.g., CO2) can exist as:
    1. Free gas phase (tracked by gas_saturation)
    2. Dissolved in oil (tracked by solvent_concentration in oil)

    The implicit formulation solves:
        R_S = S^(n+1) - S^n - Δt/(φ*V) * [F^(n+1) + q^(n+1)]
        R_C = C^(n+1) - C^n - Δt/(φ*V*S_o^(n+1)) * [F_solvent^(n+1) + q_solvent^(n+1)]

    where all fluxes and mobilities are evaluated at the new time level n+1.

    :param options: Simulation options and parameters (includes max_iterations and convergence_tolerance).

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

    oil_viscosity_grid = fluid_properties.oil_effective_viscosity_grid
    water_viscosity_grid = fluid_properties.water_viscosity_grid
    gas_viscosity_grid = fluid_properties.gas_viscosity_grid

    oil_density_grid = fluid_properties.oil_effective_density_grid
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
    relative_permeability_table = rock_fluid_properties.relative_permeability_table

    # Extract solver parameters from options
    max_iterations = options.max_iterations
    tolerance = options.convergence_tolerance

    # Grid dimensions
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

    # Initialize solution with current saturations
    updated_water_saturation_grid = current_water_saturation_grid.copy()
    updated_oil_saturation_grid = current_oil_saturation_grid.copy()
    updated_gas_saturation_grid = current_gas_saturation_grid.copy()
    updated_solvent_concentration_grid = current_solvent_concentration_grid.copy()

    # Count interior cells
    n_interior_cells = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    n_equations = 4 * n_interior_cells  # 4 unknowns: S_w, S_o, S_g, C_solvent

    # Map from cell indices to equation indices
    def get_equation_index(i: int, j: int, k: int, variable: int) -> int:
        """
        Map (i,j,k,variable) to linear equation index.
        variable: 0=water, 1=oil, 2=gas, 3=solvent_concentration
        """
        interior_i = i - 1
        interior_j = j - 1
        interior_k = k - 1
        cell_offset = (
            interior_i * (cell_count_y - 2) * (cell_count_z - 2)
            + interior_j * (cell_count_z - 2)
            + interior_k
        )
        return 4 * cell_offset + variable

    # Newton-Raphson iteration
    for iteration in range(max_iterations):
        # Recompute mobility grids with current iterate
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

        # Build residual vector and Jacobian
        residual = np.zeros(n_equations)
        jacobian_rows = []
        jacobian_cols = []
        jacobian_data = []

        # Iterate over internal cells
        for i, j, k in itertools.product(
            range(1, cell_count_x - 1),
            range(1, cell_count_y - 1),
            range(1, cell_count_z - 1),
        ):
            cell_thickness = thickness_grid[i, j, k]
            cell_volume = cell_size_x * cell_size_y * cell_thickness
            cell_porosity = porosity_grid[i, j, k]
            cell_pore_volume = cell_volume * cell_porosity
            cell_water_saturation = current_water_saturation_grid[i, j, k]
            cell_oil_saturation = current_oil_saturation_grid[i, j, k]
            cell_gas_saturation = current_gas_saturation_grid[i, j, k]
            cell_solvent_concentration = current_solvent_concentration_grid[i, j, k]
            cell_oil_pressure = current_oil_pressure_grid[i, j, k]
            cell_temperature = fluid_properties.temperature_grid[i, j, k]

            # Get equation indices
            water_eq_idx = get_equation_index(i, j, k, 0)
            oil_eq_idx = get_equation_index(i, j, k, 1)
            gas_eq_idx = get_equation_index(i, j, k, 2)
            solvent_eq_idx = get_equation_index(i, j, k, 3)

            flux_configurations = {
                "x": {
                    "mobility_grids": {
                        "water_mobility_grid": water_mobility_grid_x,
                        "oil_mobility_grid": oil_mobility_grid_x,
                        "gas_mobility_grid": gas_mobility_grid_x,
                    },
                    "neighbours": [(i + 1, j, k), (i - 1, j, k)],
                    "flow_area": cell_size_y * cell_thickness,
                    "flow_length": cell_size_x,
                },
                "y": {
                    "mobility_grids": {
                        "water_mobility_grid": water_mobility_grid_y,
                        "oil_mobility_grid": oil_mobility_grid_y,
                        "gas_mobility_grid": gas_mobility_grid_y,
                    },
                    "neighbours": [(i, j - 1, k), (i, j + 1, k)],
                    "flow_area": cell_size_x * cell_thickness,
                    "flow_length": cell_size_y,
                },
                "z": {
                    "mobility_grids": {
                        "water_mobility_grid": water_mobility_grid_z,
                        "oil_mobility_grid": oil_mobility_grid_z,
                        "gas_mobility_grid": gas_mobility_grid_z,
                    },
                    "neighbours": [(i, j, k - 1), (i, j, k + 1)],
                    "flow_area": cell_size_x * cell_size_y,
                    "flow_length": cell_thickness,
                },
            }

            # Accumulate fluxes
            net_water_flux = 0.0
            net_oil_flux = 0.0
            net_gas_flux = 0.0
            net_solvent_flux = 0.0

            # Compute miscible fluxes from neighbours (implicit)
            for config in flux_configurations.values():
                flow_area = typing.cast(float, config["flow_area"])
                flow_length = typing.cast(float, config["flow_length"])
                mobility_grids = typing.cast(
                    typing.Dict[str, ThreeDimensionalGrid], config["mobility_grids"]
                )

                for neighbour in config["neighbours"]:  # type: ignore
                    # Compute implicit miscible fluxes and derivatives
                    (
                        (
                            water_flux,
                            oil_flux,
                            gas_flux,
                            solvent_flux,
                        ),
                        flux_derivatives,
                    ) = _compute_implicit_miscible_saturation_phase_fluxes_and_derivatives(
                        cell_indices=(i, j, k),
                        neighbour_indices=neighbour,
                        flow_area=flow_area,
                        flow_length=flow_length,
                        oil_pressure_grid=current_oil_pressure_grid,
                        **mobility_grids,
                        water_saturation_grid=updated_water_saturation_grid,
                        oil_saturation_grid=updated_oil_saturation_grid,
                        gas_saturation_grid=updated_gas_saturation_grid,
                        solvent_concentration_grid=updated_solvent_concentration_grid,
                        oil_viscosity_grid=oil_viscosity_grid,
                        water_viscosity_grid=water_viscosity_grid,
                        gas_viscosity_grid=gas_viscosity_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
                        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
                        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
                        residual_gas_saturation_grid=residual_gas_saturation_grid,
                        relative_permeability_table=relative_permeability_table,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                    )

                    net_water_flux += water_flux
                    net_oil_flux += oil_flux
                    net_gas_flux += gas_flux
                    net_solvent_flux += solvent_flux

                    # Add Jacobian entries for flux derivatives
                    # Derivatives link this cell to its neighbour
                    # Get neighbour equation indices
                    neighbour_i, neighbour_j, neighbour_k = typing.cast(
                        ThreeDimensions, neighbour
                    )
                    water_equation_index_at_neighbour = get_equation_index(
                        neighbour_i, neighbour_j, neighbour_k, 0
                    )
                    oil_equation_index_at_neighbour = get_equation_index(
                        neighbour_i, neighbour_j, neighbour_k, 1
                    )
                    gas_equation_index_at_neighbour = get_equation_index(
                        neighbour_i, neighbour_j, neighbour_k, 2
                    )
                    solvent_equation_index_at_neighbour = get_equation_index(
                        neighbour_i, neighbour_j, neighbour_k, 3
                    )

                    time_discretization_factor = -time_step_in_days / cell_pore_volume

                    # Water phase equation derivatives
                    water_equation_index = get_equation_index(i, j, k, 0)
                    water_derivs = flux_derivatives.water_phase_flux_derivatives

                    if (
                        abs(water_derivs.derivative_wrt_water_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(water_equation_index)
                        jacobian_cols.append(water_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * water_derivs.derivative_wrt_water_saturation_at_neighbour
                        )

                    if (
                        abs(water_derivs.derivative_wrt_oil_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(water_equation_index)
                        jacobian_cols.append(oil_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * water_derivs.derivative_wrt_oil_saturation_at_neighbour
                        )

                    if (
                        abs(water_derivs.derivative_wrt_gas_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(water_equation_index)
                        jacobian_cols.append(gas_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * water_derivs.derivative_wrt_gas_saturation_at_neighbour
                        )

                    if (
                        abs(
                            water_derivs.derivative_wrt_solvent_concentration_at_neighbour
                        )
                        > 1e-12
                    ):
                        jacobian_rows.append(water_equation_index)
                        jacobian_cols.append(solvent_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * water_derivs.derivative_wrt_solvent_concentration_at_neighbour
                        )

                    if (
                        abs(water_derivs.derivative_wrt_water_saturation_at_cell)
                        > 1e-12
                    ):
                        jacobian_rows.append(water_equation_index)
                        jacobian_cols.append(water_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * water_derivs.derivative_wrt_water_saturation_at_cell
                        )

                    if abs(water_derivs.derivative_wrt_oil_saturation_at_cell) > 1e-12:
                        jacobian_rows.append(water_equation_index)
                        jacobian_cols.append(oil_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * water_derivs.derivative_wrt_oil_saturation_at_cell
                        )

                    if abs(water_derivs.derivative_wrt_gas_saturation_at_cell) > 1e-12:
                        jacobian_rows.append(water_equation_index)
                        jacobian_cols.append(gas_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * water_derivs.derivative_wrt_gas_saturation_at_cell
                        )

                    if (
                        abs(water_derivs.derivative_wrt_solvent_concentration_at_cell)
                        > 1e-12
                    ):
                        jacobian_rows.append(water_equation_index)
                        jacobian_cols.append(solvent_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * water_derivs.derivative_wrt_solvent_concentration_at_cell
                        )

                    # Oil phase equation derivatives
                    oil_equation_index = get_equation_index(i, j, k, 1)
                    oil_derivs = flux_derivatives.oil_phase_flux_derivatives

                    if (
                        abs(oil_derivs.derivative_wrt_water_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(oil_equation_index)
                        jacobian_cols.append(water_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * oil_derivs.derivative_wrt_water_saturation_at_neighbour
                        )

                    if (
                        abs(oil_derivs.derivative_wrt_oil_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(oil_equation_index)
                        jacobian_cols.append(oil_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * oil_derivs.derivative_wrt_oil_saturation_at_neighbour
                        )

                    if (
                        abs(oil_derivs.derivative_wrt_gas_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(oil_equation_index)
                        jacobian_cols.append(gas_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * oil_derivs.derivative_wrt_gas_saturation_at_neighbour
                        )

                    if (
                        abs(
                            oil_derivs.derivative_wrt_solvent_concentration_at_neighbour
                        )
                        > 1e-12
                    ):
                        jacobian_rows.append(oil_equation_index)
                        jacobian_cols.append(solvent_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * oil_derivs.derivative_wrt_solvent_concentration_at_neighbour
                        )

                    if abs(oil_derivs.derivative_wrt_water_saturation_at_cell) > 1e-12:
                        jacobian_rows.append(oil_equation_index)
                        jacobian_cols.append(water_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * oil_derivs.derivative_wrt_water_saturation_at_cell
                        )

                    if abs(oil_derivs.derivative_wrt_oil_saturation_at_cell) > 1e-12:
                        jacobian_rows.append(oil_equation_index)
                        jacobian_cols.append(oil_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * oil_derivs.derivative_wrt_oil_saturation_at_cell
                        )

                    if abs(oil_derivs.derivative_wrt_gas_saturation_at_cell) > 1e-12:
                        jacobian_rows.append(oil_equation_index)
                        jacobian_cols.append(gas_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * oil_derivs.derivative_wrt_gas_saturation_at_cell
                        )

                    if (
                        abs(oil_derivs.derivative_wrt_solvent_concentration_at_cell)
                        > 1e-12
                    ):
                        jacobian_rows.append(oil_equation_index)
                        jacobian_cols.append(solvent_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * oil_derivs.derivative_wrt_solvent_concentration_at_cell
                        )

                    # Gas phase equation derivatives
                    gas_equation_index = get_equation_index(i, j, k, 2)
                    gas_derivs = flux_derivatives.gas_phase_flux_derivatives

                    if (
                        abs(gas_derivs.derivative_wrt_water_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(gas_equation_index)
                        jacobian_cols.append(water_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * gas_derivs.derivative_wrt_water_saturation_at_neighbour
                        )

                    if (
                        abs(gas_derivs.derivative_wrt_oil_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(gas_equation_index)
                        jacobian_cols.append(oil_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * gas_derivs.derivative_wrt_oil_saturation_at_neighbour
                        )

                    if (
                        abs(gas_derivs.derivative_wrt_gas_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(gas_equation_index)
                        jacobian_cols.append(gas_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * gas_derivs.derivative_wrt_gas_saturation_at_neighbour
                        )

                    if (
                        abs(
                            gas_derivs.derivative_wrt_solvent_concentration_at_neighbour
                        )
                        > 1e-12
                    ):
                        jacobian_rows.append(gas_equation_index)
                        jacobian_cols.append(solvent_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * gas_derivs.derivative_wrt_solvent_concentration_at_neighbour
                        )

                    if abs(gas_derivs.derivative_wrt_water_saturation_at_cell) > 1e-12:
                        jacobian_rows.append(gas_equation_index)
                        jacobian_cols.append(water_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * gas_derivs.derivative_wrt_water_saturation_at_cell
                        )

                    if abs(gas_derivs.derivative_wrt_oil_saturation_at_cell) > 1e-12:
                        jacobian_rows.append(gas_equation_index)
                        jacobian_cols.append(oil_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * gas_derivs.derivative_wrt_oil_saturation_at_cell
                        )

                    if abs(gas_derivs.derivative_wrt_gas_saturation_at_cell) > 1e-12:
                        jacobian_rows.append(gas_equation_index)
                        jacobian_cols.append(gas_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * gas_derivs.derivative_wrt_gas_saturation_at_cell
                        )

                    if (
                        abs(gas_derivs.derivative_wrt_solvent_concentration_at_cell)
                        > 1e-12
                    ):
                        jacobian_rows.append(gas_equation_index)
                        jacobian_cols.append(solvent_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * gas_derivs.derivative_wrt_solvent_concentration_at_cell
                        )

                    # Solvent concentration equation derivatives
                    solvent_equation_index = get_equation_index(i, j, k, 3)
                    solvent_derivs = flux_derivatives.solvent_mass_flux_derivatives

                    if (
                        abs(solvent_derivs.derivative_wrt_water_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(solvent_equation_index)
                        jacobian_cols.append(water_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * solvent_derivs.derivative_wrt_water_saturation_at_neighbour
                        )

                    if (
                        abs(solvent_derivs.derivative_wrt_oil_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(solvent_equation_index)
                        jacobian_cols.append(oil_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * solvent_derivs.derivative_wrt_oil_saturation_at_neighbour
                        )

                    if (
                        abs(solvent_derivs.derivative_wrt_gas_saturation_at_neighbour)
                        > 1e-12
                    ):
                        jacobian_rows.append(solvent_equation_index)
                        jacobian_cols.append(gas_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * solvent_derivs.derivative_wrt_gas_saturation_at_neighbour
                        )

                    if (
                        abs(
                            solvent_derivs.derivative_wrt_solvent_concentration_at_neighbour
                        )
                        > 1e-12
                    ):
                        jacobian_rows.append(solvent_equation_index)
                        jacobian_cols.append(solvent_equation_index_at_neighbour)
                        jacobian_data.append(
                            time_discretization_factor
                            * solvent_derivs.derivative_wrt_solvent_concentration_at_neighbour
                        )

                    if (
                        abs(solvent_derivs.derivative_wrt_water_saturation_at_cell)
                        > 1e-12
                    ):
                        jacobian_rows.append(solvent_equation_index)
                        jacobian_cols.append(water_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * solvent_derivs.derivative_wrt_water_saturation_at_cell
                        )

                    if (
                        abs(solvent_derivs.derivative_wrt_oil_saturation_at_cell)
                        > 1e-12
                    ):
                        jacobian_rows.append(solvent_equation_index)
                        jacobian_cols.append(oil_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * solvent_derivs.derivative_wrt_oil_saturation_at_cell
                        )

                    if (
                        abs(solvent_derivs.derivative_wrt_gas_saturation_at_cell)
                        > 1e-12
                    ):
                        jacobian_rows.append(solvent_equation_index)
                        jacobian_cols.append(gas_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * solvent_derivs.derivative_wrt_gas_saturation_at_cell
                        )

                    if (
                        abs(solvent_derivs.derivative_wrt_solvent_concentration_at_cell)
                        > 1e-12
                    ):
                        jacobian_rows.append(solvent_equation_index)
                        jacobian_cols.append(solvent_eq_idx)
                        jacobian_data.append(
                            time_discretization_factor
                            * solvent_derivs.derivative_wrt_solvent_concentration_at_cell
                        )

            # Compute well contributions
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

            # Well rate calculations
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
                fluid_formation_volume_factor = (
                    injected_fluid.get_formation_volume_factor(
                        pressure=cell_oil_pressure,
                        temperature=cell_temperature,
                    )
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

                # Handle miscible solvent injection
                if injected_phase == FluidPhase.GAS and injected_fluid.is_miscible:
                    # Miscible solvent injection (e.g., CO2)
                    cell_gas_injection_rate = cell_injection_rate  # ft³/day
                    # Track solvent concentration for mixing with oil
                    cell_solvent_injection_concentration = injected_fluid.concentration
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
                water_compressibility_grid = fluid_properties.water_compressibility_grid
                oil_compressibility_grid = fluid_properties.oil_compressibility_grid
                gas_compressibility_grid = fluid_properties.gas_compressibility_grid

                for produced_fluid in production_well.produced_fluids:
                    produced_phase = produced_fluid.phase
                    if produced_phase == FluidPhase.GAS:
                        phase_mobility = gas_relative_mobility_grid[i, j, k]
                        fluid_compressibility = gas_compressibility_grid[i, j, k]
                        fluid_formation_volume_factor = (
                            gas_formation_volume_factor_grid[i, j, k]
                        )
                    elif produced_phase == FluidPhase.WATER:
                        phase_mobility = water_relative_mobility_grid[i, j, k]
                        fluid_compressibility = water_compressibility_grid[i, j, k]
                        fluid_formation_volume_factor = (
                            water_formation_volume_factor_grid[i, j, k]
                        )
                    else:
                        phase_mobility = oil_relative_mobility_grid[i, j, k]
                        fluid_compressibility = oil_compressibility_grid[i, j, k]
                        fluid_formation_volume_factor = (
                            oil_formation_volume_factor_grid[i, j, k]
                        )

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
                    elif produced_fluid.phase == FluidPhase.WATER:
                        cell_water_production_rate += production_rate * c.BBL_TO_FT3
                    else:
                        cell_oil_production_rate += production_rate * c.BBL_TO_FT3

                if production_grid is not None:
                    production_grid[i, j, k] = (
                        cell_oil_production_rate,
                        cell_water_production_rate,
                        cell_gas_production_rate,
                    )

            net_water_flow_rate = cell_water_injection_rate + cell_water_production_rate
            net_oil_flow_rate = cell_oil_injection_rate + cell_oil_production_rate
            net_gas_flow_rate = cell_gas_injection_rate + cell_gas_production_rate

            # Total flows
            total_water_flow = net_water_flux + net_water_flow_rate
            total_oil_flow = net_oil_flux + net_oil_flow_rate
            total_gas_flow = net_gas_flux + net_gas_flow_rate

            # Compute residuals for saturation equations
            water_residual = (
                updated_water_saturation_grid[i, j, k]
                - cell_water_saturation
                - (total_water_flow * time_step_in_days) / cell_pore_volume
            )
            oil_residual = (
                updated_oil_saturation_grid[i, j, k]
                - cell_oil_saturation
                - (total_oil_flow * time_step_in_days) / cell_pore_volume
            )
            gas_residual = (
                updated_gas_saturation_grid[i, j, k]
                - cell_gas_saturation
                - (total_gas_flow * time_step_in_days) / cell_pore_volume
            )

            # Compute residual for solvent concentration equation
            new_oil_saturation = updated_oil_saturation_grid[i, j, k]
            if new_oil_saturation > 1e-9:
                old_solvent_mass = (
                    cell_solvent_concentration * cell_oil_saturation * cell_pore_volume
                )
                advected_solvent_mass = net_solvent_flux * time_step_in_days
                injected_solvent_mass = 0.0
                if (
                    cell_gas_injection_rate > 0.0
                    and cell_solvent_injection_concentration > 0.0
                ):
                    injected_solvent_mass = (
                        cell_solvent_injection_concentration
                        * cell_gas_injection_rate
                        * time_step_in_days
                    )

                new_solvent_mass = (
                    old_solvent_mass + advected_solvent_mass + injected_solvent_mass
                )
                new_oil_volume = new_oil_saturation * cell_pore_volume
                predicted_concentration = new_solvent_mass / new_oil_volume

                solvent_residual = (
                    updated_solvent_concentration_grid[i, j, k]
                    - predicted_concentration
                )
            else:
                solvent_residual = updated_solvent_concentration_grid[i, j, k]

            # Store residuals
            residual[water_eq_idx] = water_residual
            residual[oil_eq_idx] = oil_residual
            residual[gas_eq_idx] = gas_residual
            residual[solvent_eq_idx] = solvent_residual

            # Add diagonal Jacobian entries for identity term (∂S/∂S = 1)
            jacobian_rows.extend([water_eq_idx, oil_eq_idx, gas_eq_idx, solvent_eq_idx])
            jacobian_cols.extend([water_eq_idx, oil_eq_idx, gas_eq_idx, solvent_eq_idx])
            jacobian_data.extend([1.0, 1.0, 1.0, 1.0])

        # Assemble sparse Jacobian
        jacobian = csr_matrix(
            (jacobian_data, (jacobian_rows, jacobian_cols)),
            shape=(n_equations, n_equations),
        )

        # Check convergence
        residual_norm = np.linalg.norm(residual)
        if residual_norm < tolerance:
            print(
                f"Miscible implicit solver converged at iteration {iteration} "
                f"with residual norm {residual_norm:.2e}"
            )
            break

        if iteration == max_iterations - 1:
            raise RuntimeError(
                f"Miscible implicit solver failed to converge after {max_iterations} iterations. "
                f"Final residual norm: {residual_norm:.2e}, tolerance: {tolerance:.2e}"
            )

        # Solve linear system
        try:
            delta_solution = spsolve(jacobian, -residual)
        except Exception as exc:
            raise RuntimeError(
                f"Linear solver failed at iteration {iteration}: {str(exc)}"
            )

        # Update solution
        for i, j, k in itertools.product(
            range(1, cell_count_x - 1),
            range(1, cell_count_y - 1),
            range(1, cell_count_z - 1),
        ):
            water_eq_idx = get_equation_index(i, j, k, 0)
            oil_eq_idx = get_equation_index(i, j, k, 1)
            gas_eq_idx = get_equation_index(i, j, k, 2)
            solvent_eq_idx = get_equation_index(i, j, k, 3)

            updated_water_saturation_grid[i, j, k] += delta_solution[water_eq_idx]
            updated_oil_saturation_grid[i, j, k] += delta_solution[oil_eq_idx]
            updated_gas_saturation_grid[i, j, k] += delta_solution[gas_eq_idx]
            updated_solvent_concentration_grid[i, j, k] += delta_solution[
                solvent_eq_idx
            ]

        # Apply constraints
        updated_water_saturation_grid = np.clip(updated_water_saturation_grid, 0.0, 1.0)
        updated_oil_saturation_grid = np.clip(updated_oil_saturation_grid, 0.0, 1.0)
        updated_gas_saturation_grid = np.clip(updated_gas_saturation_grid, 0.0, 1.0)
        updated_solvent_concentration_grid = np.clip(
            updated_solvent_concentration_grid, 0.0, 1.0
        )

        # Normalize saturations
        total_saturation_grid = (
            updated_water_saturation_grid
            + updated_oil_saturation_grid
            + updated_gas_saturation_grid
        )
        mask = total_saturation_grid > c.SATURATION_EPSILON
        if np.any(mask):
            updated_water_saturation_grid[mask] /= total_saturation_grid[mask]
            updated_oil_saturation_grid[mask] /= total_saturation_grid[mask]
            updated_gas_saturation_grid[mask] /= total_saturation_grid[mask]

    # Final cleanup
    updated_water_saturation_grid[updated_water_saturation_grid < 0.0] = 0.0
    updated_oil_saturation_grid[updated_oil_saturation_grid < 0.0] = 0.0
    updated_gas_saturation_grid[updated_gas_saturation_grid < 0.0] = 0.0
    return EvolutionResult(
        (
            updated_water_saturation_grid,
            updated_oil_saturation_grid,
            updated_gas_saturation_grid,
            updated_solvent_concentration_grid,
        ),
        scheme="implicit",
    )
