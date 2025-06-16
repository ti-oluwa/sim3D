import typing
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import itertools

from sim2D.typing import InjectionFluid, FluidMiscibility, TwoDimensionalGrid
from sim2D.properties import (
    mD_to_m2,
    DEFAULT_DISPLACED_FLUID,
    linear_decay_factor_to_exponential_decay_constant,
    compute_miscible_viscosity,
    compute_fluid_compressibility,
    compute_harmonic_transmissibility,
    compute_harmonic_mean,
    compute_diffusion_number,
)
from sim2D.grids import (
    build_2D_fluid_viscosity_grid,
    build_2D_injection_grid,
    build_2D_production_grid,
    build_2D_uniform_grid,
    edge_pad_grid,
    build_miscible_viscosity_grid,
    build_effective_mobility_grid,
    build_pressure_and_saturation_dependent_viscosity_grid,
)
from sim2D.boundary_conditions import BoundaryConditions


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

Governing Equation:
    ∂p/∂t = (1 / (φ·c_t)) · [ ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y) + q ]

Where:
    p     = pressure (Pa)
    φ     = porosity (fraction)
    c_t   = total compressibility (rock + fluid) (1/Pa)
    λ     = mobility = k / μ (m²·s/kg)
    k     = permeability (m²)
    μ     = viscosity (Pa·s)
    ∂p/∂x = spatial derivative of pressure in x direction (Pa/m)
    ∂p/∂y = spatial derivative of pressure in y direction (Pa/m)
    q     = source/sink term (normalized injection/production) (Pa/s)

Explicit Discretization (Forward Euler in time, central difference in space):
    pⁿ⁺¹_ij = pⁿ_ij + Δt / (φ·c_t) · [
        (1 / Δx²) · (λ_{i+1/2,j} · (pⁿ_{i+1,j} - pⁿ_{i,j}) - λ_{i-1/2,j} · (pⁿ_{i,j} - pⁿ_{i-1,j})) +
        (1 / Δy²) · (λ_{i,j+1/2} · (pⁿ_{i,j+1} - pⁿ_{i,j}) - λ_{i,j-1/2} · (pⁿ_{i,j} - pⁿ_{i,j-1})) +
        q_{i,j}
    ]

Where:
    Δt = time step size (s)
    Δx = cell size in x direction (m)
    Δy = cell size in y direction (m)
    pⁿ_ij = pressure at cell (i,j) at time step n (Pa)
    pⁿ⁺¹_ij = pressure at cell (i,j) at time step n+1 (Pa)
    λ_{i+1/2,j} = harmonic average transmissibility between cells (i,j) and (i+1,j)
    λ_{i-1/2,j} = harmonic average transmissibility between cells (i,j) and (i-1,j)
    λ_{i,j+1/2} = harmonic average transmissibility between cells (i,j) and (i,j+1)
    λ_{i,j-1/2} = harmonic average transmissibility between cells (i,j) and (i,j-1)
    q_{i,j} = source/sink term at cell (i,j) (m³/s)

Stability Condition:
    CFL-like (Courant-Friedrichs-Lewy) condition.
    Stable if dimensionless diffusion number D = λ·Δt / (φ·c_t·Δx²) < 0.25 in both x and y directions.

Notes:
    - Harmonic averaging is typically used to compute λ at interfaces (e.g., λ_{i+1/2,j}).
    - Δt must be small enough to ensure numerical stability.
    - This method is conditionally stable but simple to implement.
"""


# Let 'T' mean transmissibility, and 'P' mean pressure


def compute_explicit_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    time_step_size: float,
    boundary_conditions: BoundaryConditions,
    rock_compressibility: float,
    pressure_grid: TwoDimensionalGrid,
    permeability_grid: TwoDimensionalGrid,
    porosity_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    displaced_fluid_viscosity_grid: TwoDimensionalGrid,
    production_grid: typing.Optional[TwoDimensionalGrid] = None,
    injection_grid: typing.Optional[TwoDimensionalGrid] = None,
    injected_fluid: typing.Optional[InjectionFluid] = None,
    injected_fluid_saturation_grid: typing.Optional[TwoDimensionalGrid] = None,
    injected_fluid_viscosity_grid: typing.Optional[TwoDimensionalGrid] = None,
) -> TwoDimensionalGrid:
    """
    Computes the pressure evolution in the reservoir grid for one time step
    using an explicit finite difference method with saturation-dependent mobility, and
    viscosity mixing to model miscible two-phase flow (e.g., CO₂ dissolving in oil).

    This method is most suitable for cases where the dimensionless diffusion number is
    less than 0.25 (to ensure stability of the explicit scheme). It updates the pressure
    field over each time step by discretizing the pressure diffusion equation in both
    spatial dimensions (x and y), and incorporates source and sink terms due to injection
    and production wells.

    This method assumes that the reservoir is slightly compressible, meaning that
    the fluid density changes slightly with pressure, and that the fluid viscosity
    is a function of pressure and saturation. Also, it assumes that the injected fluid
    is miscible with the displaced fluid (e.g., CO₂ dissolving in oil), and that the
    viscosity of the displaced fluid changes with pressure and saturation, which is
    modeled using a viscosity mixing approach.

    The pressure diffusion equation for the reservoir is given by:

    ∂p/∂t = (1 / (φ·c_t)) · [ ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y) + q ]

    where:
        p     = pressure (Pa)
        φ     = porosity (fraction)
        c_t   = total compressibility (rock + fluid) (1/Pa)
        λ     = effective mobility = k / μ (m²·s/kg)
        k     = permeability (m²)
        μ     = viscosity (Pa·s)
        ∂p/∂x = spatial derivative of pressure in x direction (Pa/m)
        ∂p/∂y = spatial derivative of pressure in y direction (Pa/m)
        q     = source/sink term (normalized injection/production) (Pa/s)

    Mobility is computed as a function of injected phase saturation and fluid viscosities.

    Transmissibility between cells is computed using harmonic averaging of effective
    mobility and permeability.

    Assumes Cartesian grid and uniform grid spacing in both x and y directions.
    Although it can handle non-uniform grids, it is primarily designed for uniform grids.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in meters (m)
    :param time_step_size: Time step size (s) for each iteration
    :param boundary_conditions: Boundary conditions for pressure, saturation, etc, grids.
        Defaults to no-flow boundary for all sides of the grid

    :param rock_compressibility: Reservoir rock compressibility (1/Pa)
    :param pressure_grid: 2D array of initial pressure values (Pa) of the reservoir grid cells
    :param permeability_grid: 2D array of permeability values (mD) of the reservoir grid cells
    :param porosity_grid: 2D array of porosity values (fraction) of the reservoir grid cells
    :param temperature_grid: 2D array of temperature values (K) of the reservoir grid cells
    :param displaced_fluid_viscosity_grid: 2D array of displaced/reservoir fluid (e.g, gas) viscosity values (Pa·s) per grid cell
    :param production_grid: 2D array of production rates per grid cell (m³/s)
    :param injection_grid: 2D array of injection rates per grid cell (m³/s)
    :param injected_fluid: Type of injected fluid (e.g., 'CO2', 'N2')
    :param injected_fluid_saturation_grid: 2D array of injected fluid (e.g, gas) saturation values (fraction) per grid cell
    :param injected_fluid_viscosity_grid: 2D array of injected fluid (e.g, gas) viscosity values (Pa·s) per grid cell

    :return: A 2D numpy array representing the updated pressure field (Pa)
        after the specified number of time steps.
    """
    cell_count_x, cell_count_y = pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    if injection_grid is None:
        # Build an empty injection grid if not provided.
        injection_grid = build_2D_injection_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            injectors_positions=[],
            injection_rates=[],
        )
        injected_fluid_viscosity_grid = None

    else:
        if injected_fluid is None:
            raise ValueError("`injected_fluid` must be provided for injection grid.")

        if injected_fluid_viscosity_grid is None:
            injected_fluid_viscosity_grid = build_2D_fluid_viscosity_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                fluid=injected_fluid,
            )

    if production_grid is None:
        # Build an empty production grid if not provided.
        production_grid = build_2D_production_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            producers_positions=[],
            production_rates=[],
        )

    if injected_fluid_saturation_grid is None:
        injected_fluid_saturation_grid = build_2D_uniform_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            value=0.0,
            dtype=np.float64,
        )

    padded_pressure_grid = edge_pad_grid(pressure_grid.copy())
    padded_injected_fluid_saturation_grid = edge_pad_grid(
        injected_fluid_saturation_grid.copy()
    )
    pressure_boundary_condition = boundary_conditions["pressure"]
    saturation_boundary_condition = boundary_conditions["saturation"]
    pressure_boundary_condition.apply(padded_grid=padded_pressure_grid)
    saturation_boundary_condition.apply(
        padded_grid=padded_injected_fluid_saturation_grid
    )
    padded_permeability_grid = edge_pad_grid(permeability_grid.copy())
    # Compute λ effective mobility (per cell) in m²/(Pa·s)
    mobility_grid = build_effective_mobility_grid(
        injected_fluid_saturation_grid=injected_fluid_saturation_grid,
        injected_fluid_viscosity_grid=injected_fluid_viscosity_grid,
        displaced_fluid_viscosity_grid=displaced_fluid_viscosity_grid,
    )
    padded_mobility_grid = edge_pad_grid(mobility_grid)
    # Compute T = k * λ (pseudo-transmissibility per cell, not including area or Δx)
    padded_transmissibility_grid = (
        padded_permeability_grid * mD_to_m2 * padded_mobility_grid
    )
    # New grid for the updates
    updated_pressure_grid = pressure_grid.copy()

    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        ip, jp = i + 1, j + 1  # Indices of cell in padded grid
        cell_pressure = padded_pressure_grid[ip, jp]
        cell_porosity = porosity_grid[i, j]
        cell_temperature = temperature_grid[i, j]

        Tx_plus = compute_harmonic_transmissibility(
            i1=ip,
            j1=jp,
            i2=ip + 1,
            j2=jp,
            spacing=cell_size_x,
            transmissibility_grid=padded_transmissibility_grid,
        )
        Px_plus = padded_pressure_grid[ip + 1, jp]

        Tx_minus = compute_harmonic_transmissibility(
            i1=ip,
            j1=jp,
            i2=ip - 1,
            j2=jp,
            spacing=cell_size_x,
            transmissibility_grid=padded_transmissibility_grid,
        )
        Px_minus = padded_pressure_grid[ip - 1, jp]

        Ty_plus = compute_harmonic_transmissibility(
            i1=ip,
            j1=jp,
            i2=ip,
            j2=jp + 1,
            spacing=cell_size_y,
            transmissibility_grid=padded_transmissibility_grid,
        )
        Py_plus = padded_pressure_grid[ip, jp + 1]

        Ty_minus = compute_harmonic_transmissibility(
            i1=ip,
            j1=jp,
            i2=ip,
            j2=jp - 1,
            spacing=cell_size_y,
            transmissibility_grid=padded_transmissibility_grid,
        )
        Py_minus = padded_pressure_grid[ip, jp - 1]

        # Compute λ·(∂¹p/∂x¹) term
        pressure_driven_flow_rate_x = (Tx_plus * (Px_plus - cell_pressure)) + (
            Tx_minus * (cell_pressure - Px_minus)
        )
        # Compute λ·(∂¹p/∂y¹) term
        pressure_driven_flow_rate_y = (Ty_plus * (Py_plus - cell_pressure)) + (
            Ty_minus * (cell_pressure - Py_minus)
        )
        # Compute q (net rate) term (injection + production; where production is negative)
        cell_source_sink_net_flow_rate = injection_grid[i, j] + production_grid[i, j]

        if injected_fluid is not None:
            fluid_compressibility = compute_fluid_compressibility(
                pressure=cell_pressure,
                temperature=cell_temperature,
                fluid=injected_fluid,
            )
            # Total compressibility is the sum of fluid and rock compressibilities
            # c_t = c_f + c_r
            total_compressibility = fluid_compressibility + rock_compressibility
        else:
            total_compressibility = rock_compressibility

        # Compute net pressure-driven flow rate
        # net_pressure_driven_flow_rate = ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y) + q
        net_pressure_driven_flow_rate = (
            pressure_driven_flow_rate_x  # λ·(∂¹p/∂x¹) or ∂/∂x (λ·∂p/∂x) term
            + pressure_driven_flow_rate_y  # λ·(∂¹p/∂y¹) or ∂/∂y (λ·∂p/∂y) term
            + cell_source_sink_net_flow_rate  # q term
        )

        # Update pressure using the explicit finite difference formula
        # dP/dt = (1 / (φ·c_t)) · [ ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y) + q ]
        dP_dt = net_pressure_driven_flow_rate / (cell_porosity * total_compressibility)
        dP = dP_dt * time_step_size  # dP = (dP/dt) * Δt
        updated_pressure_grid[i, j] += dP

    return updated_pressure_grid


"""
Implicit finite difference formulation for pressure diffusion in a 2D reservoir (slightly compressible fluid):

Governing Equation:
    ∂p/∂t = (1 / (φ·c_t)) · [ ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y) + q ]

Where:
    p     = pressure (Pa)
    φ     = porosity (fraction)
    c_t   = total compressibility (rock + fluid) (1/Pa)
    λ     = mobility = k / μ (m²·s/kg)
    k     = permeability (m²)
    μ     = viscosity (Pa·s)
    ∂p/∂x = spatial derivative of pressure in x direction (Pa/m)
    ∂p/∂y = spatial derivative of pressure in y direction (Pa/m)
    q     = source/sink term (normalized injection/production) (Pa/s)

Fully Implicit Discretization (Backward Euler in time, central difference in space):

    (pⁿ⁺¹_ij - pⁿ_ij) / Δt = (1 / (φ·c_t)) · [
        (1 / Δx²) · (λ_{i+1/2,j} · (pⁿ⁺¹_{i+1,j} - pⁿ⁺¹_{i,j}) - λ_{i-1/2,j} · (pⁿ⁺¹_{i,j} - pⁿ⁺¹_{i-1,j})) +
        (1 / Δy²) · (λ_{i,j+1/2} · (pⁿ⁺¹_{i,j+1} - pⁿ⁺¹_{i,j}) - λ_{i,j-1/2} · (pⁿ⁺¹_{i,j} - pⁿ⁺¹_{i,j-1})) +
        q_{i,j}
    ]

Where:
    Δt = time step size (s)
    Δx = cell size in x direction (m)
    Δy = cell size in y direction (m)
    pⁿ_ij = pressure at cell (i,j) at time step n (Pa)
    pⁿ⁺¹_ij = pressure at cell (i,j) at time step n+1 (Pa)
    λ_{i+1/2,j} = harmonic average transmissibility between cells (i,j) and (i+1,j)
    λ_{i-1/2,j} = harmonic average transmissibility between cells (i,j) and (i-1,j)
    λ_{i,j+1/2} = harmonic average transmissibility between cells (i,j) and (i,j+1) 
    λ_{i,j-1/2} = harmonic average transmissibility between cells (i,j) and (i,j-1)
    q_{i,j} = source/sink term at cell (i,j) (m³/s)

Discretized as a linear system:
    A · pⁿ⁺¹ = b

Stability Condition:
    Unconditionally stable for implicit methods, but requires careful selection of Δt

Where:
    A encodes transmissibility (harmonic average of λ), geometry, and boundary conditions.
    pⁿ⁺¹ is the pressure vector at the next time step.
    b contains contributions from previous pressures and sources/sinks.
"""


def compute_implicit_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    time_step_size: float,
    boundary_conditions: BoundaryConditions,
    rock_compressibility: float,
    pressure_grid: TwoDimensionalGrid,
    permeability_grid: TwoDimensionalGrid,
    porosity_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    displaced_fluid_viscosity_grid: TwoDimensionalGrid,
    production_grid: typing.Optional[TwoDimensionalGrid] = None,
    injection_grid: typing.Optional[TwoDimensionalGrid] = None,
    injected_fluid: typing.Optional[InjectionFluid] = None,
    injected_fluid_saturation_grid: typing.Optional[TwoDimensionalGrid] = None,
    injected_fluid_viscosity_grid: typing.Optional[TwoDimensionalGrid] = None,
) -> TwoDimensionalGrid:
    """
    Computes the pressure evolution in the reservoir grid for a single time step using an
    implicit finite difference method with saturation-dependent mobility for miscible systems.

    This method is appropriate when stability is a concern (i.e., for large time steps or stiff systems),
    since it avoids the restrictive stability criterion associated with explicit schemes. It computes a
    linear system of equations derived from the discretized pressure diffusion equation, including the
    effects of spatially varying rock and fluid properties, injection/production sources, and the
    compressibility of the reservoir.

    The pressure diffusion equation for the reservoir is given by:

    ∂p/∂t = (1 / (φ·c_t)) · [ ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y) + q ]

    where:
        p     = pressure (Pa)
        φ     = porosity (fraction)
        c_t   = total compressibility (rock + fluid) (1/Pa)
        λ     = effective mobility = k / μ (m²·s/kg)
        k     = permeability (m²)
        μ     = viscosity (Pa·s)
        ∂p/∂x = spatial derivative of pressure in x direction (Pa/m)
        ∂p/∂y = spatial derivative of pressure in y direction (Pa/m)
        q     = source/sink term (normalized injection/production) (Pa/s)

    This method assumes that the reservoir is slightly compressible, meaning that
    the fluid density changes slightly with pressure, and that the fluid viscosity
    is a function of pressure and saturation. Also, it assumes that the injected fluid
    is miscible with the displaced fluid (e.g., CO₂ dissolving in oil), and that the
    viscosity of the displaced fluid changes with pressure and saturation, which is
    modeled using a viscosity mixing approach.

    Effective mobility is computed as a function of injected phase saturation and fluid viscosities.

    The finite-difference stencil incorporates harmonic averaging of permeability and mobility
    between adjacent cells to compute intercell transmissibility, and constructs a sparse
    coefficient matrix for the full reservoir grid. The resulting system is computed at each time step.

    Assumes a regular Cartesian grid with uniform spacing in both x and y directions.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in meters (m)
    :param time_step_size: Time step size (s) for the implicit scheme
    :param boundary_conditions: Boundary conditions for pressure, saturation, etc, grids.
        Defaults to no-flow boundary for all sides of the grid

    :param rock_compressibility: Reservoir rock compressibility (1/Pa)
    :param pressure_grid: 2D array of initial pressure values (Pa) of the reservoir grid cells
    :param permeability_grid: 2D array of permeability values (mD) of the reservoir grid cells
    :param porosity_grid: 2D array of porosity values (fraction) of the reservoir grid cells
    :param temperature_grid: 2D array of temperature values (K) of the reservoir grid cells
    :param displaced_fluid_viscosity_grid: 2D array of displaced/reservoir fluid (e.g, gas) viscosity values (Pa·s) per grid cell
    :param production_grid: 2D array of production rates per grid cell (m³/s)
    :param injection_grid: 2D array of injection rates per grid cell (m³/s)
    :param injected_fluid: Type of injected fluid (e.g., 'CO2', 'N2')
    :param injected_fluid_saturation_grid: 2D array of injected phase (e.g, gas) saturation values (fraction) per grid cell
    :param injected_fluid_viscosity_grid: 2D array of injected fluid (e.g, gas) viscosity values (Pa·s) per grid cell
    :return: A 2D numpy array representing the updated pressure distribution (Pa)
        after solving the implicit system for the current time step.
    """
    cell_size_x, cell_size_y = cell_dimension
    cell_count_x, cell_count_y = pressure_grid.shape
    # Total number of grid cells
    N = cell_count_x * cell_count_y

    # Sparse coefficient matrix
    A = lil_matrix((N, N))
    # RHS source/accumulation vector
    b = np.zeros(N)

    def to_1d_index(i: int, j: int) -> int:
        """
        Convert 2D grid indices (i, j) to a 1D index for the sparse matrix.

        :param i: Row index
        :param j: Column index
        :return: 1D index corresponding to (i, j)
        """
        if not (0 <= i < cell_count_x and 0 <= j < cell_count_y):
            raise IndexError(
                f"Indices ({i}, {j}) are out of bounds for grid size ({cell_count_x}, {cell_count_y})."
            )

        return i * cell_count_y + j

    if injection_grid is None:
        # Build an empty injection grid if not provided.
        injection_grid = build_2D_injection_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            injectors_positions=[],
            injection_rates=[],
        )
        injected_fluid_viscosity_grid = None

    else:
        if injected_fluid is None:
            raise ValueError("`injected_fluid` must be provided for injection grid.")

        if injected_fluid_viscosity_grid is None:
            injected_fluid_viscosity_grid = build_2D_fluid_viscosity_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                fluid=injected_fluid,
            )

    if production_grid is None:
        # Build an empty production grid if not provided.
        production_grid = build_2D_production_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            producers_positions=[],
            production_rates=[],
        )

    if injected_fluid_saturation_grid is None:
        injected_fluid_saturation_grid = build_2D_uniform_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            value=0.0,
            dtype=np.float64,
        )

    padded_pressure_grid = edge_pad_grid(pressure_grid.copy())
    padded_injected_fluid_saturation_grid = edge_pad_grid(
        injected_fluid_saturation_grid.copy()
    )
    padded_permeability_grid = edge_pad_grid(permeability_grid.copy())
    pressure_boundary_condition = boundary_conditions["pressure"]
    pressure_boundary_condition.apply(padded_grid=padded_pressure_grid)
    saturation_boundary_condition = boundary_conditions["saturation"]
    saturation_boundary_condition.apply(
        padded_grid=padded_injected_fluid_saturation_grid
    )
    # Compute λ effective mobility (per cell) in m²/(Pa·s)
    mobility_grid = build_effective_mobility_grid(
        injected_fluid_saturation_grid=injected_fluid_saturation_grid,
        injected_fluid_viscosity_grid=injected_fluid_viscosity_grid,
        displaced_fluid_viscosity_grid=displaced_fluid_viscosity_grid,
    )
    padded_mobility_grid = edge_pad_grid(mobility_grid)
    # Compute T = k * λ (pseudo-transmissibility per cell, not including area or Δx)
    padded_transmissibility_grid = (
        padded_permeability_grid * mD_to_m2 * padded_mobility_grid
    )

    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        ip, jp = i + 1, j + 1  # Indices of cell in padded grid
        cell_porosity = porosity_grid[i, j]
        cell_pressure = padded_pressure_grid[ip, jp]
        cell_temperature = temperature_grid[i, j]

        if injected_fluid:
            fluid_compressibility = compute_fluid_compressibility(
                pressure=cell_pressure,
                temperature=cell_temperature,
                fluid=injected_fluid,
            )
            # Total compressibility is the sum of fluid and rock compressibilities
            # c_t = c_f + c_r
            total_compressibility = fluid_compressibility + rock_compressibility
        else:
            total_compressibility = rock_compressibility

        # Accumulation term for the implicit scheme
        # (φ * c_t * (dx * dy) )/ dt where:
        # φ = porosity, c_t = total compressibility, dx = cell_size_x, dy = cell_size_y, (dx * dy) = area of the cell,
        # dt = time_step_size
        # This term represents the change in pressure over time due to fluid flow and compressibility
        acccumulation = (
            cell_porosity * total_compressibility * (cell_size_x * cell_size_y)
        ) / time_step_size
        Pi_1d = to_1d_index(i, j)
        A[Pi_1d, Pi_1d] = acccumulation
        # Set the diagonal entry for the current cell in the sparse matrix
        cell_source_sink_net_flow_rate = injection_grid[i, j] + production_grid[i, j]
        b[Pi_1d] = acccumulation * cell_pressure + cell_source_sink_net_flow_rate

        # Compute transmissibility between this cell and its neighbors
        # di, dj are the offsets to the neighboring cell
        # spacing is the distance between the cells in the respective direction
        for di, dj, spacing in [
            (-1, 0, cell_size_x),
            (1, 0, cell_size_x),
            (0, -1, cell_size_y),
            (0, 1, cell_size_y),
        ]:
            # Compute the neighbor indices
            # ni, nj are the indices of the neighboring cell
            ni, nj = i + di, j + dj

            # Compute harmonic transmissibility between the current cell (i, j)
            # and the neighbor cell (ni, nj)
            T_ij = compute_harmonic_transmissibility(
                i1=i,
                j1=j,
                i2=ni,
                j2=nj,
                spacing=spacing,
                transmissibility_grid=padded_transmissibility_grid,
            ) * (cell_size_x * cell_size_y)

            # Compute the 1D index for the neighbor cell
            Pni_1d = to_1d_index(ni, nj)

            # Set the off-diagonal entries in the sparse matrix
            A[Pi_1d, Pi_1d] += T_ij  # type: ignore
            A[Pi_1d, Pni_1d] -= T_ij  # type: ignore

    # Solve linear system
    updated_pressure_grid = spsolve(A.tocsr(), b).reshape((cell_count_x, cell_count_y))
    return typing.cast(TwoDimensionalGrid, updated_pressure_grid)


def compute_adaptive_pressure_evolution(
    cell_dimension: typing.Tuple[float, float],
    time_step_size: float,
    boundary_conditions: BoundaryConditions,
    rock_compressibility: float,
    pressure_grid: TwoDimensionalGrid,
    permeability_grid: TwoDimensionalGrid,
    porosity_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    displaced_fluid_viscosity_grid: TwoDimensionalGrid,
    production_grid: typing.Optional[TwoDimensionalGrid] = None,
    injection_grid: typing.Optional[TwoDimensionalGrid] = None,
    injected_fluid: typing.Optional[InjectionFluid] = None,
    injected_fluid_saturation_grid: typing.Optional[TwoDimensionalGrid] = None,
    injected_fluid_viscosity_grid: typing.Optional[TwoDimensionalGrid] = None,
    diffusion_number_threshold: float = 0.24,  # Slightly below 0.25 for safety
) -> TwoDimensionalGrid:
    """
    Computes the pressure distribution in the reservoir grid for a single time step,
    adaptively choosing between explicit and implicit methods based on the maximum
    diffusion number in the grid.

    The pressure diffusion equation for the reservoir is given by:

    ∂p/∂t = (1 / (φ·c_t)) · [ ∂/∂x (λ·∂p/∂x) + ∂/∂y (λ·∂p/∂y) + q ]

    where:
        p     = pressure (Pa)
        φ     = porosity (fraction)
        c_t   = total compressibility (rock + fluid) (1/Pa)
        λ     = effective mobility = k / μ (m²·s/kg)
        k     = permeability (m²)
        μ     = viscosity (Pa·s)
        ∂p/∂x = spatial derivative of pressure in x direction (Pa/m)
        ∂p/∂y = spatial derivative of pressure in y direction (Pa/m)
        q     = source/sink term (normalized injection/production) (Pa/s)

    This method assumes that the reservoir is slightly compressible, meaning that
    the fluid density changes slightly with pressure, and that the fluid viscosity
    is a function of pressure and saturation. Also, it assumes that the injected fluid
    is miscible with the displaced fluid (e.g., CO₂ dissolving in oil), and that the
    viscosity of the displaced fluid changes with pressure and saturation, which is
    modeled using a viscosity mixing approach.

    Effective mobility is computed as a function of injected phase saturation and fluid viscosities.

    The finite-difference stencil incorporates harmonic averaging of permeability and mobility
    between adjacent cells to compute intercell transmissibility, and constructs a sparse
    coefficient matrix for the full reservoir grid. The resulting system is computed at each time step.

    Assumes a regular Cartesian grid with uniform spacing in both x and y directions.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in meters (m)
    :param time_step_size: Time step size (s) for the implicit scheme
    :param boundary_conditions: Boundary conditions for pressure, saturation, etc, grids.
        Defaults to no-flow boundary for all sides of the grid

    :param rock_compressibility: Reservoir rock compressibility (1/Pa)
    :param pressure_grid: 2D array of initial pressure values (Pa) of the reservoir grid cells
    :param permeability_grid: 2D array of permeability values (mD) of the reservoir grid cells
    :param porosity_grid: 2D array of porosity values (fraction) of the reservoir grid cells
    :param temperature_grid: 2D array of temperature values (K) of the reservoir grid cells
    :param displaced_fluid_viscosity_grid: 2D array of displaced/reservoir fluid (e.g, gas) viscosity values (Pa·s) per grid cell
    :param production_grid: 2D array of production rates per grid cell (m³/s)
    :param injection_grid: 2D array of injection rates per grid cell (m³/s)
    :param injected_fluid: Type of injected fluid (e.g., 'CO2', 'N2')
    :param injected_fluid_saturation_grid: 2D array of injected phase (e.g, gas) saturation values (fraction) per grid cell
    :param injected_fluid_viscosity_grid: 2D array of injected fluid (e.g, gas) viscosity values (Pa·s) per grid cell
    :param diffusion_number_threshold: The maximum allowed diffusion number for explicit stability.
        If any cell exceeds this, the implicit solver is used.

    :return: A 2D numpy array representing the updated pressure distribution (Pa)
        after solving the implicit system for the current time step.
    """
    cell_count_x, cell_count_y = pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Prepare viscosity grids for THIS time step's calculation (based on current P, S)
    # This step is crucial and should be done ONCE per time step before
    # deciding on the solver.
    if injection_grid is None:
        injected_fluid_viscosity_grid = None
    else:
        if injected_fluid is None:
            raise ValueError("`injected_fluid` must be provided for injection grid.")

        if injected_fluid_viscosity_grid is None:
            injected_fluid_viscosity_grid = build_2D_fluid_viscosity_grid(
                pressure_grid=pressure_grid,  # Use current pressure for viscosity
                temperature_grid=temperature_grid,
                fluid=injected_fluid,
            )

    if injected_fluid_saturation_grid is None:
        injected_fluid_saturation_grid = build_2D_uniform_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            value=0.0,
            dtype=np.float64,
        )

    # Determine max diffusion number not considering miscibility yet
    max_diffusion_number = 0.0
    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        cell_pressure = pressure_grid[i, j]
        cell_porosity = porosity_grid[i, j]
        cell_permeability = permeability_grid[i, j]
        cell_temperature = temperature_grid[i, j]
        displaced_fluid_viscosity = displaced_fluid_viscosity_grid[i, j]

        if injected_fluid:
            fluid_compressibility = compute_fluid_compressibility(
                pressure=cell_pressure,
                temperature=cell_temperature,
                fluid=injected_fluid,
            )
            total_compressibility = fluid_compressibility + rock_compressibility
        else:
            total_compressibility = rock_compressibility

        # Need to consider the smaller cell dimension for the worst-case diffusion number
        min_cell_size = min(cell_size_x, cell_size_y)

        # Compute diffusion number for this cell
        diffusion_number = compute_diffusion_number(
            permeability=cell_permeability,
            porosity=cell_porosity,
            viscosity=displaced_fluid_viscosity,
            total_compressibility=total_compressibility,
            time_step_size=time_step_size,
            cell_size=min_cell_size,
        )
        if diffusion_number > max_diffusion_number:
            max_diffusion_number = diffusion_number

    # Choose solver based on criterion
    if max_diffusion_number > diffusion_number_threshold:
        updated_pressure_grid = compute_implicit_pressure_evolution(
            cell_dimension=cell_dimension,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_compressibility=rock_compressibility,
            pressure_grid=pressure_grid,
            permeability_grid=permeability_grid,
            porosity_grid=porosity_grid,
            temperature_grid=temperature_grid,
            displaced_fluid_viscosity_grid=displaced_fluid_viscosity_grid,
            production_grid=production_grid,
            injection_grid=injection_grid,
            injected_fluid_saturation_grid=injected_fluid_saturation_grid,
            injected_fluid=injected_fluid,
            injected_fluid_viscosity_grid=injected_fluid_viscosity_grid,
        )
    else:
        updated_pressure_grid = compute_explicit_pressure_evolution(
            cell_dimension=cell_dimension,
            time_step_size=time_step_size,
            boundary_conditions=boundary_conditions,
            rock_compressibility=rock_compressibility,
            pressure_grid=pressure_grid,
            permeability_grid=permeability_grid,
            porosity_grid=porosity_grid,
            temperature_grid=temperature_grid,
            displaced_fluid_viscosity_grid=displaced_fluid_viscosity_grid,
            production_grid=production_grid,
            injection_grid=injection_grid,
            injected_fluid_saturation_grid=injected_fluid_saturation_grid,
            injected_fluid=injected_fluid,
            injected_fluid_viscosity_grid=injected_fluid_viscosity_grid,
        )

    return updated_pressure_grid


def compute_cell_displaced_fluid_effective_viscosity(
    i: int,
    j: int,
    injected_fluid_saturation_grid: TwoDimensionalGrid,
    injected_fluid_viscosity_grid: typing.Optional[TwoDimensionalGrid],
    displaced_fluid_viscosity_grid: TwoDimensionalGrid,
    viscosity_mixing_method: str = "logarithmic",
) -> float:
    if injected_fluid_viscosity_grid is not None:
        return compute_miscible_viscosity(
            injected_fluid_saturation=injected_fluid_saturation_grid[i, j],
            injected_fluid_viscosity=injected_fluid_viscosity_grid[i, j],
            displaced_fluid_viscosity=displaced_fluid_viscosity_grid[i, j],
            miscibility=viscosity_mixing_method,
        )
    return displaced_fluid_viscosity_grid[i, j]


def compute_saturation_evolution(
    cell_dimension: tuple[float, float],
    time_step_size: float,
    boundary_conditions: BoundaryConditions,
    pressure_grid: TwoDimensionalGrid,
    permeability_grid: TwoDimensionalGrid,
    porosity_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    displaced_fluid_viscosity_grid: TwoDimensionalGrid,
    production_grid: typing.Optional[TwoDimensionalGrid] = None,
    injection_grid: typing.Optional[TwoDimensionalGrid] = None,
    injected_fluid: typing.Optional[InjectionFluid] = None,
    injected_fluid_saturation_grid: typing.Optional[TwoDimensionalGrid] = None,
    injected_fluid_viscosity_grid: typing.Optional[TwoDimensionalGrid] = None,
) -> TwoDimensionalGrid:
    """
    Computes the new/updated saturation distribution of the injected fluid across the reservoir grid
    using an upwind finite difference method and standard viscosity-ratio-based fractional flow function.

    We assume **miscible two-phase displacement** with known or unknown fluid viscosities, and compute
    directional fluxes using Darcy's law and the upwind approximation.

    We also assume no interfacial tension effects, capillary pressure, or gravity effects,
    and that the injected fluid is miscible with the displaced fluid (e.g., CO₂ dissolving in oil).
    This method is suitable for simulating miscible two-phase flow in reservoirs
    where the injected fluid (e.g., CO₂) dissolves into the displaced fluid (e.g., oil).

    :param cell_dimension: Tuple representing the dimensions of each grid cell (cell_size_x, cell_size_y) in meters.
    :param time_step_size: Time step duration in seconds for the simulation.
    :param boundary_conditions: Boundary conditions for pressure, saturation, etc, grids.
        Defaults to no-flow boundary for all sides of the grid

    :param initial_reservoir_pressure: Initial (average) pressure of the reservoir (Pa) used for viscosity calculations.
    :param pressure_grid: 2D array of pressure values (Pa) representing the pressure distribution
        of the reservoir across the grid cells.

    :param permeability_grid: 2D array of permeability values (mD) representing the permeability distribution of the reservoir rock
        across the grid cells. Values are converted to m² for calculations.

    :param porosity_grid: 2D array of reservoir porosity values (fraction) representing the porosity distribution of the reservoir rock
        across the grid cells.

    :param temperature_grid: 2D array of temperature values (K) representing the temperature distribution
        of the reservoir across the grid cells.

    :param displaced_fluid_viscosity_grid: 2D array of displaced/reservoir fluid (e.g, gas) viscosity values (Pa·s) per grid cell
    :param production_grid: 2D array of production rates (m³/s) per grid cell. If not provided, an empty grid is created.
    :param injection_grid: 2D array of injection rates (m³/s) per grid cell. If not provided, an empty grid is created.
    :param injected_fluid: Type of injected fluid (e.g., "CO2", "water"). Used to determine viscosity.
    :param injected_fluid_saturation_grid: 2D array of injected fluid saturation (fraction between 0 and 1) representing
        the saturation distribution of the injected fluid (e.g., CO₂, water) across the grid cells.
    :param injected_fluid_viscosity_grid: 2D array of injected fluid (e.g, gas) viscosity values (Pa·s) per grid cell

    :return: 2D array of updated injected fluid saturation representing the new saturation distribution
        of the injected fluid across the grid cells after applying the upwind finite difference method.
    """
    cell_count_x, cell_count_y = pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    if injected_fluid_saturation_grid is None:
        injected_fluid_saturation_grid = build_2D_uniform_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            value=0.0,
            dtype=np.float64,
        )

    if injection_grid is None:
        # Build an empty injection grid if not provided.
        injection_grid = build_2D_injection_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            injectors_positions=[],
            injection_rates=[],
        )
        padded_injected_fluid_viscosity_grid = injected_fluid_viscosity_grid = None

    else:
        if injected_fluid is None:
            raise ValueError("`injected_fluid` must be provided for injection grid.")

        if injected_fluid_viscosity_grid is None:
            injected_fluid_viscosity_grid = build_2D_fluid_viscosity_grid(
                pressure_grid=pressure_grid,
                temperature_grid=temperature_grid,
                fluid=injected_fluid,
            )
        padded_injected_fluid_viscosity_grid = edge_pad_grid(
            injected_fluid_viscosity_grid
        )

    if production_grid is None:
        # Build an empty production grid if not provided.
        production_grid = build_2D_production_grid(
            grid_dimension=(cell_count_x, cell_count_y),
            producers_positions=[],
            production_rates=[],
        )

    padded_pressure_grid = edge_pad_grid(pressure_grid)
    padded_injected_fluid_saturation_grid = edge_pad_grid(
        injected_fluid_saturation_grid
    )
    padded_displaced_fluid_viscosity_grid = edge_pad_grid(
        displaced_fluid_viscosity_grid
    )
    pressure_boundary_condition = boundary_conditions["pressure"]
    pressure_boundary_condition.apply(
        padded_grid=padded_pressure_grid,
    )
    saturation_boundary_condition = boundary_conditions["saturation"]
    saturation_boundary_condition.apply(
        padded_grid=padded_injected_fluid_saturation_grid,
    )

    updated_saturation_grid = injected_fluid_saturation_grid.copy()
    iterator = itertools.product(range(cell_count_x), range(cell_count_y))
    for i, j in iterator:
        ip, jp = i + 1, j + 1  # Indices of cell in padded grid
        cell_permeability = permeability_grid[i, j] * mD_to_m2
        cell_porosity = porosity_grid[i, j]
        cell_pressure = pressure_grid[i, j]
        injected_fluid_saturation = injected_fluid_saturation_grid[i, j]
        displaced_fluid_viscosity = compute_cell_displaced_fluid_effective_viscosity(
            i=i,
            j=j,
            injected_fluid_saturation_grid=injected_fluid_saturation_grid,
            injected_fluid_viscosity_grid=injected_fluid_viscosity_grid,
            displaced_fluid_viscosity_grid=displaced_fluid_viscosity_grid,
        )

        # --- Compute Volumetric Flux of INJECTED FLUID from Neighbouring Cells ---
        # Compute volumetric fluxes using Darcy's law and upwind approximation
        # For each cell, we compute the volumetric flux in the
        # - East [i + 1, j] (right)
        # - West [i - 1, j] (left)
        # - North [i, j + 1] (up)
        # - South [i, j - 1] (down)
        # We use the upwind approximation to determine the direction of flow based on the pressure gradient.
        # The calculated fluxes represent the flow of the injected fluid component.

        #######################
        # Computing East flux #
        #######################
        injected_fluid_saturation_in_eastwards_cell = (
            padded_injected_fluid_saturation_grid[ip + 1, jp]
        )
        pressure_in_eastwards_cell = padded_pressure_grid[ip + 1, jp]
        displaced_fluid_viscosity_in_eastwards_cell = (
            compute_cell_displaced_fluid_effective_viscosity(
                i=ip + 1,
                j=jp,
                injected_fluid_saturation_grid=padded_injected_fluid_saturation_grid,
                injected_fluid_viscosity_grid=padded_injected_fluid_viscosity_grid,
                displaced_fluid_viscosity_grid=padded_displaced_fluid_viscosity_grid,
            )
        )
        # Compute the harmonic mean of the effective viscosities at the east face
        # This is used to compute the velocity across the east face
        displaced_fluid_viscosity_at_east_face = compute_harmonic_mean(
            value1=displaced_fluid_viscosity,
            value2=displaced_fluid_viscosity_in_eastwards_cell,
        )
        # Compute the velocity across the east face using Darcy's law
        # v = -k / mu * (dP/dx)
        velocity_across_east_face = (
            -cell_permeability / displaced_fluid_viscosity_at_east_face
        ) * ((pressure_in_eastwards_cell - cell_pressure) / cell_size_x)

        # If the velocity is positive, flow is towards the east face (out of the cell),
        # otherwise it is towards the west face (in to the cell).
        upwinded_saturation_across_east_face = (
            injected_fluid_saturation
            if velocity_across_east_face >= 0
            else injected_fluid_saturation_in_eastwards_cell
        )
        # Compute the volumetric flux through the east face
        # This is the product of the velocity, upwinded saturation, and cell size in y direction
        volumetric_flux_through_east_face = (
            velocity_across_east_face
            * upwinded_saturation_across_east_face
            * cell_size_y
        )

        #######################
        # Computing West flux #
        #######################
        injected_fluid_saturation_in_westwards_cell = (
            padded_injected_fluid_saturation_grid[ip - 1, jp]
        )
        pressure_in_westwards_cell = padded_pressure_grid[ip - 1, jp]
        displaced_fluid_viscosity_in_westwards_cell = (
            compute_cell_displaced_fluid_effective_viscosity(
                i=ip - 1,
                j=jp,
                injected_fluid_saturation_grid=padded_injected_fluid_saturation_grid,
                injected_fluid_viscosity_grid=padded_injected_fluid_viscosity_grid,
                displaced_fluid_viscosity_grid=padded_displaced_fluid_viscosity_grid,
            )
        )
        # Compute the harmonic mean of the effective viscosities at the west face
        displaced_fluid_viscosity_at_west_face = compute_harmonic_mean(
            value1=displaced_fluid_viscosity,
            value2=displaced_fluid_viscosity_in_westwards_cell,
        )

        # Compute the velocity across the west face using Darcy's law
        # v = -k / mu * (dP/dx)
        velocity_across_west_face = (
            -cell_permeability / displaced_fluid_viscosity_at_west_face
        ) * ((cell_pressure - pressure_in_westwards_cell) / cell_size_x)
        # If the velocity is positive, flow is towards the west face (out of the cell),
        # otherwise it is towards the east face (in to the cell).
        upwinded_saturation_across_west_face = (
            injected_fluid_saturation_in_westwards_cell
            if velocity_across_west_face > 0
            else injected_fluid_saturation
        )
        # Compute the volumetric flux through the west face
        volumetric_flux_through_west_face = (
            velocity_across_west_face
            * upwinded_saturation_across_west_face
            * cell_size_y
        )

        ########################
        # Computing North flux #
        ########################

        injected_fluid_saturation_in_northwards_cell = (
            padded_injected_fluid_saturation_grid[ip, jp + 1]
        )
        pressure_in_northwards_cell = padded_pressure_grid[ip, jp + 1]
        displaced_fluid_viscosity_in_northwards_cell = (
            compute_cell_displaced_fluid_effective_viscosity(
                i=ip,
                j=jp + 1,
                injected_fluid_saturation_grid=padded_injected_fluid_saturation_grid,
                injected_fluid_viscosity_grid=padded_injected_fluid_viscosity_grid,
                displaced_fluid_viscosity_grid=padded_displaced_fluid_viscosity_grid,
            )
        )
        # Compute the harmonic mean of the effective viscosities at the north face
        displaced_fluid_viscosity_at_north_face = compute_harmonic_mean(
            value1=displaced_fluid_viscosity,
            value2=displaced_fluid_viscosity_in_northwards_cell,
        )

        # Compute the velocity across the north face using Darcy's law
        # v = -k / mu * (dP/dy)
        velocity_across_north_face = (
            -cell_permeability / displaced_fluid_viscosity_at_north_face
        ) * ((pressure_in_northwards_cell - cell_pressure) / cell_size_y)
        # If the velocity is positive, flow is towards the north face (out of the cell),
        # otherwise it is towards the south face (in to the cell).
        upwinded_saturation_across_north_face = (
            injected_fluid_saturation
            if velocity_across_north_face >= 0
            else injected_fluid_saturation_in_northwards_cell
        )
        # Compute the volumetric flux through the north face
        volumetric_flux_through_north_face = (
            velocity_across_north_face
            * upwinded_saturation_across_north_face
            * cell_size_x
        )

        ########################
        # Computing South flux #
        ########################
        injected_fluid_saturation_in_southwards_cell = (
            padded_injected_fluid_saturation_grid[ip, jp - 1]
        )
        pressure_in_southwards_cell = padded_pressure_grid[ip, jp - 1]
        displaced_fluid_viscosity_in_southwards_cell = (
            compute_cell_displaced_fluid_effective_viscosity(
                i=ip,
                j=jp - 1,
                injected_fluid_saturation_grid=padded_injected_fluid_saturation_grid,
                injected_fluid_viscosity_grid=padded_injected_fluid_viscosity_grid,
                displaced_fluid_viscosity_grid=padded_displaced_fluid_viscosity_grid,
            )
        )
        # Compute the harmonic mean of the effective viscosities at the south face
        displaced_fluid_viscosity_at_south_face = compute_harmonic_mean(
            value1=displaced_fluid_viscosity,
            value2=displaced_fluid_viscosity_in_southwards_cell,
        )

        # Compute the velocity across the south face using Darcy's law
        # v = -k / mu * (dP/dy)
        velocity_across_south_face = (
            -cell_permeability / displaced_fluid_viscosity_at_south_face
        ) * ((cell_pressure - pressure_in_southwards_cell) / cell_size_y)
        # If the velocity is positive, flow is towards the south face (out of the cell),
        # otherwise it is towards the north face (in to the cell).
        upwinded_saturation_across_south_face = (
            injected_fluid_saturation_in_southwards_cell
            if velocity_across_south_face > 0
            else injected_fluid_saturation
        )
        # Compute the volumetric flux through the south face
        volumetric_flux_through_south_face = (
            velocity_across_south_face
            * upwinded_saturation_across_south_face
            * cell_size_x
        )

        # ---- Compute inwards and outwards volumetric fluxes from/to neighboring cells ----
        # Inwards fluxes are the sum of fluxes into the cell from the west and south faces
        net_inwards_volumetric_flux_from_neighbour_cells = (
            volumetric_flux_through_west_face + volumetric_flux_through_south_face
        )
        # Outwards fluxes are the sum of fluxes out of the cell to the east and north faces
        net_outwards_volumetric_flux_to_neighbour_cells = (
            volumetric_flux_through_east_face + volumetric_flux_through_north_face
        )
        # Net volumetric flux from neighbouring cells (positive is inflow, negative is outflow)
        net_volumetric_flux_from_neighbour_cells = (
            net_inwards_volumetric_flux_from_neighbour_cells
            - net_outwards_volumetric_flux_to_neighbour_cells
        )

        # Get well rates (source/sink terms)
        injection_rate = injection_grid[i, j]
        production_rate = production_grid[i, j]

        # Production removes injected fluid proportional to its saturation in the cell
        injected_fluid_produced = production_rate * injected_fluid_saturation

        # Combine inter-cell fluxes with well source/sink terms
        total_net_injected_fluid_flux = (
            net_volumetric_flux_from_neighbour_cells
            + injection_rate
            - injected_fluid_produced
        )

        # Compute the pore volume of the cell (% of the cell volume that can be saturated)
        cell_pore_volume = cell_size_x * cell_size_y * cell_porosity
        # If pore volume is greater than zero, that means the cell has pore space that can be saturated
        # with the injected fluid, we can update the saturation.
        if cell_pore_volume > 0:
            # Compute the change in saturation due to the net volumetric flux
            saturation_delta = (
                total_net_injected_fluid_flux / cell_pore_volume
            ) * time_step_size
            updated_saturation_grid[i, j] += saturation_delta

        # Ensure saturation remains within [0, 1]
        updated_saturation_grid[i, j] = np.clip(updated_saturation_grid[i, j], 0.0, 1.0)

    return updated_saturation_grid
