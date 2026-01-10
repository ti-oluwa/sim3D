import functools
import logging
import typing
import warnings

import attrs
import numba
import numpy as np
import pyamg
from scipy.sparse import csr_array, csr_matrix, diags, isspmatrix_csr
from scipy.sparse.linalg import (
    LinearOperator,
    bicgstab,
    gmres,
    lgmres,
    spilu,
    spsolve,
    tfqmr,
)

from bores.errors import SolverError, PreconditionerError
from bores.types import (
    IterativeSolver,
    IterativeSolverFunc,
    Preconditioner,
    T,
    PreconditionerFactory,
    ThreeDimensions,
    ThreeDimensionalGrid,
)

logger = logging.getLogger(__name__)


__all__ = [
    "EvolutionResult",
    "build_cpr_preconditioner",
    "build_ilu_preconditioner",
    "build_diagonal_preconditioner",
    "build_amg_preconditioner",
    "solve_linear_system",
    "to_1D_index_interior_only",
    "from_1D_index_interior_only",
]


def _warn_production_rate_is_positive(
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
        f"Warning: Production well '{well_name}' at cell {cell} has a positive rate of {production_rate:.4f} {rate_unit}, "
        f"indicating it is no longer producing fluid at {time:.3f} seconds. Production rates should be negative. Please check well configuration.",
        UserWarning,
    )


def _warn_injection_rate_is_negative(
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
        f"Warning: Injection well '{well_name}' at cell {cell} has a negative rate of {injection_rate:.4f} {rate_unit}, "
        f"indicating it is no longer injecting fluid at {time:.3f} seconds. Injection rates should be postive. Please check well configuration.",
        UserWarning,
    )


def _warn_production_pressure_is_high(
    bhp: float,
    well_name: str,
    cell: ThreeDimensions,
    cell_pressure: float,
    time: float,
):
    warnings.warn(
        f"Warning: Production well '{well_name}' at cell {cell} has a high BHP of {bhp:.4f}psi, cell pressure is {cell_pressure:.4f}psi, "
        f"indicating it is no longer producing fluid at {time:.3f} seconds. Production pressure should be lower than reservoir pressure. Please check well configuration.",
        UserWarning,
    )


def _warn_injection_pressure_is_low(
    bhp: float,
    well_name: str,
    cell: ThreeDimensions,
    cell_pressure: float,
    time: float,
):
    warnings.warn(
        f"Warning: Injection well '{well_name}' at cell {cell} has a low BHP of {bhp:.4f}psi, cell pressure is {cell_pressure:.4f}psi, "
        f"indicating it is no longer injecting fluid at {time:.3f} seconds. Injection pressure should be higher than reservoir pressure. Please check well configuration.",
        UserWarning,
    )


M = typing.TypeVar("M")


@attrs.frozen
class EvolutionResult(typing.Generic[T, M]):
    """
    Result of a single evolution step in the simulation.
    """

    value: T
    """The result value if successful, otherwise None."""
    scheme: typing.Literal["implicit", "explicit"]
    """The numerical scheme used for the evolution step."""
    success: bool = True
    """Indicates if the evolution step was successful."""
    message: typing.Optional[str] = None
    """A message providing additional information about the result."""
    metadata: typing.Optional[M] = None
    """Optional metadata related to the evolution step."""


@numba.njit(inline="always", cache=True)
def to_1D_index_interior_only(
    i: int,
    j: int,
    k: int,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> int:
    """
    Convert 3D interior cell indices to 1D array index.

    For a grid with dimensions (Nx, Ny, Nz), interior cells are
    indexed from (1, 1, 1) to (Nx-2, Ny-2, Nz-2).

    The 1D index starts at 0 for cell (1, 1, 1).

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
        i_interior * (ny_interior * nz_interior)
        + (j_interior * nz_interior)
        + k_interior
    )


@numba.njit(cache=True)
def from_1D_index_interior_only(
    idx: int,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> typing.Tuple[int, int, int]:
    """
    Convert 1D interior cell index back to 3D grid indices.

    This is the inverse of to_1D_index_interior_only.

    For a grid with dimensions (Nx, Ny, Nz), interior cells are
    indexed from (1, 1, 1) to (Nx-2, Ny-2, Nz-2).

    The 1D index starts at 0 for cell (1, 1, 1).

    :param idx: 1D array index (0 to interior_cell_count - 1)
    :param cell_count_x: Total number of cells in x-direction (including boundaries)
    :param cell_count_y: Total number of cells in y-direction (including boundaries)
    :param cell_count_z: Total number of cells in z-direction (including boundaries)
    :return: Tuple of (i, j, k) indices in the full grid (interior cells only)
    """
    # Compute interior dimensions
    interior_Ny = cell_count_y - 2
    interior_Nz = cell_count_z - 2

    # Reverse the row-major ordering
    # idx = i_interior * (interior_Ny * interior_Nz) + j_interior * interior_Nz + k_interior
    i_interior = idx // (interior_Ny * interior_Nz)
    remainder = idx % (interior_Ny * interior_Nz)
    j_interior = remainder // interior_Nz
    k_interior = remainder % interior_Nz

    # Convert back to full grid coordinates (add 1 to shift from interior to full grid)
    i = i_interior + 1
    j = j_interior + 1
    k = k_interior + 1

    return i, j, k


@numba.njit(cache=True)
def compute_mobility_grids(
    absolute_permeability_x: ThreeDimensionalGrid,
    absolute_permeability_y: ThreeDimensionalGrid,
    absolute_permeability_z: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
]:
    """
    Compute mobility grids for all three phases in all three directions (x, y, z).

    Mobility = (absolute permeability * relative mobility) * conversion factor

    :param absolute_permeability_x: Absolute permeability in x-direction (mD)
    :param absolute_permeability_y: Absolute permeability in y-direction (mD)
    :param absolute_permeability_z: Absolute permeability in z-direction (mD)
    :param water_relative_mobility_grid: Water relative mobility (1/cP)
    :param oil_relative_mobility_grid: Oil relative mobility (1/cP)
    :param gas_relative_mobility_grid: Gas relative mobility (1/cP)
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion constant
    :return: Tuple of 3 direction tuples, each containing (water, oil, gas) mobility grids:
        (x_mobilities, y_mobilities, z_mobilities) where each is (water, oil, gas)
        All with units (ft²/psi·day)
    """
    # X-direction mobilities
    water_mobility_grid_x = (
        absolute_permeability_x
        * water_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_mobility_grid_x = (
        absolute_permeability_x
        * oil_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_mobility_grid_x = (
        absolute_permeability_x
        * gas_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # Y-direction mobilities
    water_mobility_grid_y = (
        absolute_permeability_y
        * water_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_mobility_grid_y = (
        absolute_permeability_y
        * oil_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_mobility_grid_y = (
        absolute_permeability_y
        * gas_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # Z-direction mobilities
    water_mobility_grid_z = (
        absolute_permeability_z
        * water_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_mobility_grid_z = (
        absolute_permeability_z
        * oil_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_mobility_grid_z = (
        absolute_permeability_z
        * gas_relative_mobility_grid
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # Group by direction: (water, oil, gas) for each direction
    x_mobilities = (water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x)
    y_mobilities = (water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y)
    z_mobilities = (water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z)
    return (x_mobilities, y_mobilities, z_mobilities)


def build_amg_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix], **kwargs: typing.Any
) -> LinearOperator:
    """
    Creates an Algebraic Multigrid (AMG) preconditioner using PyAMG

    :param A_csr: The coefficient matrix in CSR format.
    :return: A SciPy `LinearOperator` that represents the AMG preconditioner.
    """
    ml_solver = pyamg.smoothed_aggregation_solver(A_csr, **kwargs)
    M_amg = ml_solver.aspreconditioner(cycle="V")  # V-cycle is standard
    return M_amg


def build_diagonal_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix],
) -> LinearOperator:
    """
    Creates a diagonal preconditioner from the coefficient matrix.

    :param A_csr: The coefficient matrix in CSR format.
    :return: A SciPy `LinearOperator` that represents the diagonal preconditioner.
    """
    diag_elements = A_csr.diagonal()
    # Avoid division by zero by replacing zeros with a small number
    diag_elements = np.where(np.abs(diag_elements) < 1e-10, 1.0, diag_elements)
    M_diag = diags(1.0 / diag_elements, format="csr")
    return LinearOperator(shape=A_csr.shape, matvec=M_diag.dot)  # type: ignore[arg-type]


def build_ilu_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix], **kwargs: typing.Any
) -> LinearOperator:
    """
    Creates an Incomplete LU (ILU) preconditioner using `spilu`.

    :param A_csr: The coefficient matrix in CSR format. It will be
        converted to CSC for efficiency with `spilu`.
    :return: A SciPy `LinearOperator` that solves the preconditioned system.
    """
    # spilu works most efficiently with the CSC matrix format.
    A_csc = A_csr.tocsc()

    # Compute the Incomplete LU factorization
    # You can tune 'drop_tol' and 'fill_factor' for better performance/accuracy trade-offs
    # spilu returns a SuperLU object which has a .solve() method
    kwargs.setdefault("drop_tol", 1e-4)  # Drop tolerance (controls sparsity/accuracy)
    kwargs.setdefault("fill_factor", 10)  # Memory allocation factor
    ilu_factor = spilu(A_csc, **kwargs)
    # Create a `LinearOperator` that uses the .solve() method as the preconditioning step
    M = LinearOperator(shape=A_csc.shape, matvec=ilu_factor.solve)  # type: ignore[arg-type]
    return M


_CPR_AMG_KWARGS = {
    "max_coarse": 500,
    "presmoother": ("gauss_seidel", {"sweep": "symmetric", "iterations": 1}),
    "postsmoother": ("gauss_seidel", {"sweep": "symmetric", "iterations": 1}),
}
"""Default AMG parameters for CPR preconditioner."""
_CPR_ILU_KWARGS = {
    "drop_tol": 1e-4,
    "fill_factor": 10,
}
"""Default ILU parameters for CPR preconditioner."""


def build_cpr_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix],
    *,
    n_variables_per_cell: int = 3,
    pressure_variable_index: int = 0,
    amg_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ilu_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
) -> LinearOperator:
    """
    Creates a Constrained Pressure Residual (CPR) preconditioner for a fully-implicit
    multiphase reservoir Jacobian.

    CPR is a two-stage preconditioner:

        Stage 1 (global): AMG solve of the pressure-pressure sub-block A_pp.
        Stage 2 (local): ILU solve on the full system to smooth high-frequency errors.

    This preconditioner significantly improves convergence for 3D multiphase
    reservoir simulations where the Jacobian is strongly coupled but the pressure
    equation dominates the long-range behavior.

    :param A_csr: The full Jacobian matrix in CSR format.
    :param n_variables_per_cell: Number of unknowns per grid cell
        (e.g., pressure, saturation_o, saturation_g).
    :param pressure_variable_index: Index of the pressure unknown inside each cell's
        variable block. Example: if block = [pressure, So, Sg], then index = 0.
    :param amg_kwargs: Keyword arguments for `pyamg.smoothed_aggregation_solver`.
    :param ilu_kwargs: Keyword arguments for `scipy.sparse.linalg.spilu`.
    :return: A SciPy `LinearOperator` implementing the CPR preconditioner M such that
        x = M @ r approximately solves J^{-1} r.
    :raises ValueError: If no pressure DOFs are found.
    :raises RuntimeError: If AMG or ILU construction fails.

    Notes
    -----
    CPR workflow:

        1. Restrict residual to pressure DOFs:      r_p = R * r
        2. Solve pressure block:                    z_p = A_pp^{-1} * r_p   (AMG)
        3. Prolongate correction to full space:     z = P * z_p
        4. Compute remaining residual:              w = r - A * z
        5. Smooth locally:                          y = ILU^{-1} * w
        6. Final CPR output:                        x = z + y
    """
    if not isspmatrix_csr(A_csr):
        A_csr = csr_matrix(A_csr)

    amg_kwargs = amg_kwargs or dict(_CPR_AMG_KWARGS)
    ilu_kwargs = ilu_kwargs or dict(_CPR_ILU_KWARGS)
    number_of_equations = A_csr.shape[0]  # type: ignore
    # Identify the pressure DOFs: [p_i, So_i, Sg_i, ...]
    pressure_dof_indices = np.arange(
        pressure_variable_index,
        number_of_equations,
        n_variables_per_cell,
        dtype=np.int64,
    )
    if pressure_dof_indices.size == 0:
        raise ValueError(
            "No pressure DOFs found. Check `n_variables_per_cell` or `pressure_variable_index`."
        )

    # Extract the pressure-pressure block (A_pp)
    A_pp = A_csr[pressure_dof_indices, :][:, pressure_dof_indices].tocsr()

    # Build AMG preconditioner for A_pp
    try:
        M_amg = build_amg_preconditioner(A_pp, **amg_kwargs)
    except Exception as exc:
        raise PreconditionerError(
            f"AMG construction for pressure block failed: {exc}"
        ) from exc

    # Build ILU preconditioner for the full matrix
    try:
        M_ilu = build_ilu_preconditioner(A_csr, **ilu_kwargs)
    except Exception as exc:
        raise PreconditionerError(f"ILU factorization failed for CPR: {exc}") from exc

    # Restriction and prolongation operators (implicit)
    def restrict_to_pressure(vec_full: np.typing.NDArray) -> np.typing.NDArray:
        return vec_full[pressure_dof_indices]

    def prolongate_to_full(vec_pressure: np.typing.NDArray) -> np.typing.NDArray:
        out = np.zeros(number_of_equations, dtype=vec_pressure.dtype)
        out[pressure_dof_indices] = vec_pressure
        return out

    def matvec(residual: np.typing.NDArray) -> np.typing.NDArray:
        """CPR preconditioner application: x = M^{-1} r"""
        # Stage 1: pressure solve
        r_p = restrict_to_pressure(residual)
        try:
            z_p = M_amg.dot(r_p)
        except TypeError:
            z_p = M_amg(r_p)

        z = prolongate_to_full(z_p)  # type: ignore[arg-type]

        # Stage 2: ILU correction
        w = residual - A_csr.dot(z)
        y = M_ilu.dot(w)
        return z + y

    return LinearOperator(shape=A_csr.shape, matvec=matvec)  # type: ignore[arg-type]


__preconditioner_factories = {
    "cpr": build_cpr_preconditioner,
    "amg": build_amg_preconditioner,
    "ilu": build_ilu_preconditioner,
    "diagonal": build_diagonal_preconditioner,
}

_lgmres = functools.partial(lgmres, inner_m=50, outer_k=5)
__iterative_solvers = {
    "lgmres": _lgmres,
    "bicgstab": bicgstab,
    "tfqmr": tfqmr,
    "gmres": gmres,
}


def _get_preconditioner(
    A_csr: typing.Union[csr_array, csr_matrix],
    preconditioner: typing.Optional[Preconditioner],
) -> typing.Optional[LinearOperator]:
    if isinstance(preconditioner, (type(None), LinearOperator)):
        return preconditioner
    elif isinstance(preconditioner, str):
        if preconditioner in __preconditioner_factories:
            preconditioner_factory = __preconditioner_factories[preconditioner]
            M = preconditioner_factory(A_csr)
            return M
        else:
            raise ValueError(f"Unknown preconditioner type: {preconditioner!r}")
    elif callable(preconditioner):
        preconditioner_factory = typing.cast(PreconditionerFactory, preconditioner)
        M = preconditioner_factory(A_csr)
        return M
    return preconditioner  # type: ignore[return-value]


def _get_solver_funcs(
    solver: typing.Union[IterativeSolver, typing.Iterable[IterativeSolver]],
) -> typing.List[IterativeSolverFunc]:
    if isinstance(solver, str):
        if solver in __iterative_solvers:
            solver_func = __iterative_solvers[solver]
            if isinstance(solver_func, (list, tuple)):
                return list(solver_func)  # type: ignore[return-value]
            return [solver_func]
        raise ValueError(f"Unknown solver type: {solver!r}")
    elif isinstance(solver, (list, tuple, set)):
        solver_funcs = []
        for s in solver:
            if isinstance(s, str) and s in __iterative_solvers:
                solver_funcs.append(__iterative_solvers[s])
            elif callable(s):
                solver_funcs.append(s)
            else:
                raise ValueError(f"Unknown solver type in sequence: {s!r}")
        return solver_funcs
    raise TypeError("solver must be a string or a sequence of strings.")


def solve_linear_system(
    A_csr: typing.Union[csr_array, csr_matrix],
    b: np.typing.NDArray,
    max_iterations: int,
    rtol: typing.Optional[float] = None,
    atol: typing.Optional[float] = None,
    solver: typing.Union[
        IterativeSolver, typing.Iterable[IterativeSolver]
    ] = "bicgstab",
    preconditioner: typing.Optional[Preconditioner] = "ilu",
    fallback_to_direct: bool = False,
) -> typing.Tuple[np.typing.NDArray, typing.Optional[LinearOperator]]:
    """
    Solves the linear system A·x = b using an iterative solver with a fallback strategy.

    The function first attempts to solve the system using the BiCGSTAB method, which is
    generally efficient for large, sparse, non-symmetric systems. If BiCGSTAB fails to
    converge within the specified number of iterations, the function falls back to the
    LGMRES method, which is more robust but typically slower.

    Preconditioning is applied using a diagonal preconditioner derived from the diagonal
    elements of matrix A to improve convergence.

    :param A: Coefficient matrix in CSR format.
    :param b: Right-hand side vector.
    :param max_iterations: Maximum number of iterations for each solver.
    :param solver: Iterative solver or sequence of solvers to use ("bicgstab", "gmres", "lgmres", "tfqmr"), or custom callable(s).
        If a sequence is provided, solvers will be tried in order until one converges.
    :param preconditioner: Type of preconditioner to use ("ilu", "amg", "diagonal"), or None.
        Can also be a preconditioner factory function, that takes A and returns a preconditioner.
        If None, no preconditioning is applied.
    :param rtol: Relative tolerance for convergence (optional).
    :param atol: Absolute tolerance for convergence (optional).
    :param fallback_to_direct: Whether to fall back to a direct solver if all iterative solvers fail.
        Not suitable for large or production use cases due to performance and memory constraints.
    :return: A tuple (x, M) where x is the solution vector and M is the preconditioner used,
    :raises RuntimeError: If both solvers fail to converge.
    """
    try:
        M = _get_preconditioner(A_csr, preconditioner)
    except Exception as exc:
        raise PreconditionerError(f"Error building preconditioner: {exc}") from exc

    b_norm = np.linalg.norm(b)
    # Too strict
    # epsilon = get_floating_point_info().eps
    # rtol = rtol if rtol is not None else float(epsilon * 50)
    # atol = atol if atol is not None else float(max(1e-8, 20 * epsilon * b_norm))
    rtol = rtol if rtol is not None else 1e-6
    atol = atol if atol is not None else float(max(1e-8, 1e-6 * b_norm))
    solver_funcs = _get_solver_funcs(solver)

    for solver_func in solver_funcs:
        x, info = solver_func(
            A=A_csr,
            b=b,
            x0=None,
            M=M,
            rtol=rtol,
            atol=atol,
            maxiter=max_iterations,
            callback=None,
        )
        if info == 0:
            return np.ascontiguousarray(x), M
        else:
            logger.warning(
                f"Solver {solver_func!r} failed to converge within {max_iterations} iterations. Info: {info}"
            )

    if not fallback_to_direct:
        raise SolverError(
            f"All iterative solvers failed to converge within {max_iterations} iterations."
        )

    logger.info("Falling back to direct solver (spsolve).")
    try:
        x = spsolve(A_csr, b)
    except Exception as exc:
        logger.error(f"Direct solver failed: {exc}")
        raise SolverError(
            "All iterative solvers and direct solver failed to solve the system."
        ) from exc

    return np.ascontiguousarray(x), None  # type: ignore[return-value]


"""
Guidelines for Selecting Iterative Solvers and Preconditioners
==============================================================

This guide explains how to choose the most appropriate combination of
iterative solver(s) and preconditioner(s) for large 3D, three-phase,
fully-implicit or IMPES reservoir simulations. The goal is to balance:

    • Convergence robustness  
    • Runtime speed  
    • Memory usage  
    • Ability to customize components and keyword arguments  

The solver stack in this module is intentionally flexible: users can select
built-in solvers and preconditioners, provide custom factory functions, or
override default keyword arguments. The guidance below describes how each
choice affects performance.

---------------------------------------------------------------------------
1. Recommended Default Combination (Best Overall for Reservoir Simulation)
---------------------------------------------------------------------------

    solver="lgmres"
    preconditioner="cpr"

Use this if you do not have a strict memory constraint.

Why this is optimal:
    • CPR isolates the pressure block and solves it with AMG, capturing
      long-range pressure coupling very efficiently.
    • ILU smoothing removes high-frequency saturation/phase-coupling errors.
    • LGMRES handles the mildly non-symmetric structure of the Jacobian
      better than GMRES and can recover from stagnation.
    • Convergence is fastest on most three-phase, highly-coupled systems.

Memory cost:
    • Moderate to high (AMG hierarchy + ILU factors).
    • Recommended for full-scale 3D implicit simulations.

---------------------------------------------------------------------------
2. When Memory is Limited (Prefer Low-Memory Configuration)
---------------------------------------------------------------------------

Use:

    solver="bicgstab"
    preconditioner="ilu"

Why:
    • BiCGSTAB has low memory overhead compared to GMRES/LGMRES
      (no Krylov basis storage).
    • ILU requires significantly less memory than AMG or CPR.
    • Convergence is generally good for mildly stiff Jacobians.

Tradeoffs:
    • More sensitive to non-symmetric or strongly coupled Jacobians.
    • May require reducing timestep if convergence deteriorates.

Memory footprint:
    • Low to moderate.

---------------------------------------------------------------------------
3. Extremely Memory-Constrained Environments (Minimal Footprint)
---------------------------------------------------------------------------

Use:

    solver="tfqmr"
    preconditioner="diagonal"

Why:
    • TFQMR has near-minimal memory usage.
    • Diagonal preconditioning is essentially free to build.

Important:
    • Convergence will be slow.
    • Only recommended for academic testing or very small models.
    • Not recommended for fully implicit multiphase simulations with strong
      coupling, gravity, or capillarity.

Memory:
    • Very low.

---------------------------------------------------------------------------
4. High-Accuracy, Difficult Jacobians (Most Robust Configuration)
---------------------------------------------------------------------------

Use:

    solver=["gmres", "lgmres"]
    preconditioner="cpr"

Why:
    • GMRES handles highly non-symmetric systems extremely well.
    • LGMRES is a fallback when GMRES stagnates.
    • CPR gives the best robustness for stiff systems (tight permeability
      contrast, high viscosity ratio, strong gravity).

Tradeoffs:
    • Highest memory usage of all combinations.
    • GMRES basis growth increases memory as iterations increase.

---------------------------------------------------------------------------
5. Using Custom Factories or Override Keyword Arguments
---------------------------------------------------------------------------

All built-in preconditioners (`ilu`, `amg`, `cpr`, `diagonal`) accept user-supplied
factory functions or dictionaries of keyword arguments.

Examples:

    # Custom ILU with tighter fill
    solve_linear_system(
        A_csr, b,
        solver="bicgstab",
        preconditioner=lambda A: build_ilu_preconditioner(
            A, drop_tol=1e-6, fill_factor=20
        ),
    )

    # Adjust AMG coarsening parameters
    solve_linear_system(
        A_csr, b,
        solver="lgmres",
        preconditioner=lambda A: build_amg_preconditioner(
            A, max_coarse=300, presmoother=("gauss_seidel", 2)
        ),
    )

    # Full custom CPR with overridden AMG+ILU kwargs
    solve_linear_system(
        A_csr, b,
        solver="lgmres",
        preconditioner=lambda A: build_cpr_preconditioner(
            A,
            amg_kwargs={"max_coarse": 300, "presmoother": ("gauss_seidel", 2)},
            ilu_kwargs={"drop_tol": 1e-5, "fill_factor": 8},
        )
    )

General rules for custom tuning:

    • AMG:
        - Increase max_coarse to reduce memory but increase iteration count.
        - Increase presmoother/postsmoother sweeps for stiffer systems.
        - Use smoothed aggregation (default) for multiphase problems.

    • ILU:
        - Lower drop_tol → more accurate, more memory, faster convergence.
        - Higher fill_factor → larger ILU factors, higher memory.

    • CPR:
        - If CPR fails, reduce ILU strength or give AMG more smoothing.
        - If CPR is slow, the pressure block or transmissibility assembly may be stiff.

---------------------------------------------------------------------------
6. Selecting a Sequence of Solvers
---------------------------------------------------------------------------

The `solver=` argument also accepts a list:

    solver=["bicgstab", "lgmres"]

The solver is tried in order. If one fails to converge, the next is attempted.

Typical robust sequence:

    solver=["lgmres", "bicgstab"]

Typical low-memory sequence:

    solver=["bicgstab", "tfqmr"]

---------------------------------------------------------------------------
7. Direct Solver Fallback
---------------------------------------------------------------------------

Enable only for debugging or tiny models:

    fallback_to_direct=True

This will use `spsolve` only if all iterative solvers fail.

Never enable on large 3D grids (memory explosion).

---------------------------------------------------------------------------
8. Summary Table: Best → Least Favorable
---------------------------------------------------------------------------

    Highest convergence speed:
        CPR + LGMRES  >  CPR + GMRES  >  ILU + BiCGSTAB > ILU + TFQMR

    Lowest memory:
        Diagonal + TFQMR  <  ILU + BiCGSTAB  < CPR + LGMRES

    Most robust for real reservoir engineering:
        CPR + LGMRES

    Best overall default for users:
        solver="lgmres", preconditioner="cpr"

---------------------------------------------------------------------------

This guide is meant to give intuitive decision rules for users modifying
solver strategies, tuning preconditioner strength, or supplying their
own factory functions to meet memory, robustness, or speed requirements.
"""
