"""
Advanced linear solvers and preconditioners for implicit saturation systems.

This module provides production-grade linear solvers optimized for the sparse,
non-symmetric systems arising from implicit reservoir simulation:

1. ILU Preconditioner: Incomplete LU factorization for general sparse matrices
2. Block-Jacobi Preconditioner: Specialized for multi-saturation formulations
3. ORTHOMIN Solver: Orthogonal minimization for non-symmetric systems
4. Flexible GMRES: Wrapper with automatic fallback and adaptive tolerance

Mathematical Background:
-----------------------
For the linear system Ax = b arising from Newton linearization:

    J(S^k) * ΔS = -R(S^k)

where:
    J = Jacobian matrix (sparse, non-symmetric)
    R = Residual vector
    ΔS = Update to saturation vector
    S^k = Current Newton iterate

The system is typically:
- Large (10^4 to 10^6 unknowns)
- Sparse (O(10) non-zeros per row)
- Non-symmetric (due to upwinding and capillary pressure)
- Ill-conditioned (saturation changes span multiple orders of magnitude)

Preconditioning M approximates A^{-1} such that:
    M^{-1} A x = M^{-1} b
has better conditioning.

"""

import logging
import typing

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, lgmres, spilu, spsolve

__all__ = [
    "ILUPreconditioner",
    "BlockJacobiPreconditioner",
    "orthomin_solver",
    "solve_linear_system",
]

logger = logging.getLogger(__name__)


class ILUPreconditioner:
    """
    Incomplete LU (ILU) preconditioner for sparse matrices.

    ILU factorization computes approximate L and U factors:
        A ≈ L * U

    where L is lower triangular and U is upper triangular, subject to sparsity
    constraints (controlled by drop_tol and fill_factor).

    Usage:
        precond = ILUPreconditioner(jacobian, drop_tol=1e-4, fill_factor=10)
        M = precond.get_linear_operator()
        x, info = gmres(jacobian, rhs, M=M)

    :param drop_tol: Drop tolerance for ILU. Smaller values retain more fill-in (more accurate,
        more memory). Typical range: [1e-5, 1e-3]
    :param fill_factor: Maximum fill-in factor. Larger values allow more fill-in (more accurate,
        more memory). Typical range: [10, 50]

    Performance Notes:
    ------------------
    - ILU setup cost: O(nnz * fill_factor^2) time, O(nnz * fill_factor) memory
    - Application cost: O(nnz * fill_factor) per solve
    - Effective for general sparse matrices with moderate condition number
    - May fail for extremely ill-conditioned systems
    """

    def __init__(
        self,
        matrix: csr_matrix,
        drop_tol: float = 1e-4,
        fill_factor: int = 10,
    ):
        """
        Construct ILU preconditioner.

        :param matrix: Sparse matrix in CSR format
        :param drop_tol: Drop tolerance for small entries
        :param fill_factor: Maximum additional non-zeros per row
        """
        self.matrix = matrix
        self.drop_tol = drop_tol
        self.fill_factor = fill_factor
        self.ilu = None
        self._build_preconditioner()

    def _build_preconditioner(self):
        """Compute ILU factorization."""
        try:
            # Convert to CSC for better column access during factorization
            matrix_csc = self.matrix.tocsc()

            # Compute ILU with specified parameters
            self.ilu = spilu(
                matrix_csc,
                drop_tol=self.drop_tol,
                fill_factor=self.fill_factor,
                drop_rule="basic",  # Drop small entries
                permc_spec="NATURAL",  # No column permutation
            )

            nnz_factor = self.ilu.nnz / self.matrix.nnz if self.matrix.nnz > 0 else 0.0
            logger.debug(
                f"ILU preconditioner built: "
                f"fill_factor={nnz_factor:.2f}, "
                f"L+U nnz={self.ilu.nnz}, "
                f"A nnz={self.matrix.nnz}"
            )

        except RuntimeError as e:
            logger.warning(
                f"ILU factorization failed: {e}. Using identity preconditioner."
            )
            self.ilu = None

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply preconditioner: solve M * x = rhs, where M ≈ A.

        :param rhs: Right-hand side vector
        :return: Solution to preconditioned system
        """
        if self.ilu is None:
            # Fallback to identity (no preconditioning)
            return rhs.copy()

        return self.ilu.solve(rhs)

    def get_linear_operator(self) -> LinearOperator:
        """
        Return LinearOperator interface for scipy iterative solvers.

        :return: LinearOperator wrapping the preconditioner
        """
        n = self.matrix.shape[0]  # type: ignore[union-attr]

        def matvec_function(v: np.ndarray) -> np.ndarray:
            return self.solve(v)

        return LinearOperator(
            shape=(n, n),
            matvec=matvec_function,  # type: ignore[call-arg]
            dtype=self.matrix.dtype,  # type: ignore[union-attr]
        )


class BlockJacobiPreconditioner:
    """
    Block-Jacobi preconditioner for multi-saturation systems.

    For systems with n_phases saturations per cell, the Jacobian has block structure:

        J = | J_11  J_12  ...  J_1n |
            | J_21  J_22  ...  J_2n |
            |  :     :    ...   :   |
            | J_n1  J_n2  ...  J_nn |

    where J_ij is n_cells × n_cells.

    Block-Jacobi preconditioner extracts diagonal blocks (per cell):

        M^{-1} = | (J_cell1)^{-1}     0           ...    0         |
                 |     0          (J_cell2)^{-1}  ...    0         |
                 |     :               :          ...    :         |
                 |     0               0          ... (J_cellN)^{-1}|

    Each block J_cell_i is n_phases × n_phases (small, dense, easy to invert).

    Advantages:
    -----------
    - Perfectly parallel (each block independent)
    - Captures saturation coupling within each cell
    - Cheap to construct and apply
    - Robust for moderately coupled systems

    Limitations:
    ------------
    - Ignores spatial coupling between cells
    - Less effective for strongly convection-dominated flow
    - May require many iterations for large grids

    Usage:
        precond = BlockJacobiPreconditioner(
            jacobian, n_phases=2, n_cells=1000
        )
        M = precond.get_linear_operator()
        x, info = gmres(jacobian, rhs, M=M)
    """

    def __init__(
        self,
        matrix: csr_matrix,
        n_phases: int,
        n_cells: int,
    ):
        """
        Construct Block-Jacobi preconditioner.

        :param matrix: Sparse Jacobian in CSR format
        :param n_phases: Number of saturation equations per cell (2 or 3)
        :param n_cells: Number of cells in grid
        """
        self.matrix = matrix
        self.n_phases = n_phases
        self.n_cells = n_cells
        self.block_size = n_phases
        self.n_blocks = n_cells
        self.blocks_inv = []

        self._extract_diagonal_blocks()

    def _extract_diagonal_blocks(self):
        """Extract and invert diagonal blocks."""
        logger.debug(
            f"Building Block-Jacobi preconditioner: "
            f"{self.n_blocks} blocks, size {self.block_size}x{self.block_size}"
        )

        # Extract diagonal blocks
        for block_idx in range(self.n_blocks):
            start_row = block_idx * self.block_size
            end_row = start_row + self.block_size

            # Extract block as dense matrix
            block = self.matrix[start_row:end_row, start_row:end_row].toarray()

            # Invert block (small matrix, use direct inversion)
            try:
                block_inv = np.linalg.inv(block)
                self.blocks_inv.append(block_inv)
            except np.linalg.LinAlgError:
                # Singular block, use pseudo-inverse or identity
                logger.warning(f"Block {block_idx} is singular, using pseudo-inverse")
                try:
                    block_inv = np.linalg.pinv(block)
                    self.blocks_inv.append(block_inv)
                except Exception:
                    # Ultimate fallback: identity
                    self.blocks_inv.append(np.eye(self.block_size))

        logger.debug("Block-Jacobi preconditioner built successfully")

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply block-Jacobi preconditioner.

        :param rhs: Right-hand side vector
        :return: Solution to M * x = rhs
        """
        solution = np.zeros_like(rhs, dtype=np.float64)

        for block_idx, block_inv in enumerate(self.blocks_inv):
            start = block_idx * self.block_size
            end = start + self.block_size

            # Apply inverse block: x_block = (J_block)^{-1} * rhs_block
            solution[start:end] = block_inv @ rhs[start:end]

        return solution

    def get_linear_operator(self) -> LinearOperator:
        """
        Return LinearOperator interface for scipy iterative solvers.

        :return: LinearOperator wrapping the preconditioner
        """
        n = self.matrix.shape[0]  # type: ignore[union-attr]

        def matvec_function(v: np.ndarray) -> np.ndarray:
            return self.solve(v)

        return LinearOperator(
            shape=(n, n),
            matvec=matvec_function,  # type: ignore[call-arg]
            dtype=self.matrix.dtype,  # type: ignore[union-attr]
        )


def orthomin_solver(
    A: csr_matrix,
    b: np.ndarray,
    x0: typing.Optional[np.ndarray] = None,
    M: typing.Optional[LinearOperator] = None,
    tol: float = 1e-6,
    maxiter: int = 100,
) -> typing.Tuple[np.ndarray, int, float]:
    """
    ORTHOMIN(1) iterative solver for non-symmetric linear systems.

    ORTHOMIN (Orthogonal Minimization) is a Krylov subspace method particularly
    effective for non-symmetric systems arising from reservoir simulation.

    Algorithm:
    ----------
    Given A * x = b, with preconditioner M ≈ A^{-1}:

    1. r_0 = b - A * x_0
    2. z_0 = M * r_0
    3. p_0 = z_0
    4. For k = 0, 1, 2, ...
        a. Ap_k = A * p_k
        b. α_k = (r_k^T * z_k) / (Ap_k^T * Ap_k)
        c. x_{k+1} = x_k + α_k * p_k
        d. r_{k+1} = r_k - α_k * Ap_k
        e. Check convergence: ||r_{k+1}|| / ||b|| < tol
        f. z_{k+1} = M * r_{k+1}
        g. β_k = (r_{k+1}^T * z_{k+1}) / (r_k^T * z_k)
        h. p_{k+1} = z_{k+1} + β_k * p_k

    Advantages over GMRES:
    ----------------------
    - More robust for highly non-symmetric systems
    - Lower memory footprint (stores only 3 vectors vs. m vectors for GMRES(m))
    - Often converges faster for reservoir simulation problems
    - Better suited for systems with strong upwinding

    :param A: Sparse coefficient matrix (CSR format)
    :param b: Right-hand side vector
    :param x0: Initial guess (default: zero vector)
    :param M: Preconditioner as LinearOperator (default: identity)
    :param tol: Relative residual tolerance
    :param maxiter: Maximum iterations
    :return: Tuple of (solution vector, iterations performed, final relative residual)

    References:
    -----------
    - Vinsome, P.K.W. (1976). "ORTHOMIN, an Iterative Method for Solving
      Sparse Sets of Simultaneous Linear Equations". SPE 5729.
    - Behie, A., & Vinsome, P.K.W. (1982). "Block Iterative Methods for
      Fully Implicit Reservoir Simulation". SPE Journal, 22(5), 658-668.
    """
    n = A.shape[0]  # type: ignore[union-attr]

    # Initial guess
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = x0.copy()

    # Initial residual
    r = b - A @ x
    b_norm = np.linalg.norm(b)

    # Handle zero RHS
    if b_norm < 1e-14:
        logger.warning("RHS vector is numerically zero")
        return x, 0, 0.0

    # Apply preconditioner
    if M is not None:
        z = M @ r
    else:
        z = r.copy()

    # Search direction
    p = np.asarray(z).copy()  # type: ignore[arg-type]

    # Check initial residual
    residual_norm = np.linalg.norm(r) / b_norm
    if residual_norm < tol:
        logger.debug(f"ORTHOMIN: Already converged, ||r||/||b|| = {residual_norm:.2e}")
        return x, 0, float(residual_norm)  # type: ignore[return-value]

    logger.debug(f"ORTHOMIN: Starting with ||r||/||b|| = {residual_norm:.2e}")

    for iteration in range(maxiter):
        # Apply matrix to search direction
        Ap = A @ p

        # Compute step length (minimize residual in direction p)
        rz_dot = float(np.dot(r, z))  # type: ignore[arg-type]
        ApAp_dot = float(np.dot(Ap, Ap))

        if ApAp_dot < 1e-14:
            logger.warning(f"ORTHOMIN: Breakdown at iteration {iteration}, ||Ap|| ≈ 0")
            return x, iteration, float(residual_norm)  # type: ignore[return-value]

        alpha = rz_dot / ApAp_dot

        # Update solution
        x = x + alpha * p

        # Update residual
        r = r - alpha * Ap

        # Check convergence
        residual_norm = np.linalg.norm(r) / b_norm
        if residual_norm < tol:
            logger.debug(
                f"ORTHOMIN: Converged in {iteration + 1} iterations, "
                f"||r||/||b|| = {residual_norm:.2e}"
            )
            return x, iteration + 1, float(residual_norm)  # type: ignore[return-value]

        # Apply preconditioner to new residual
        if M is not None:
            z_new = M @ r
        else:
            z_new = r.copy()

        # Compute new search direction (β ensures orthogonality)
        rz_new_dot = float(np.dot(r, z_new))  # type: ignore[arg-type]
        beta = rz_new_dot / rz_dot if abs(rz_dot) > 1e-14 else 0.0

        p = np.asarray(z_new).copy() + beta * p  # type: ignore[arg-type]
        z = z_new

        # Logging
        if (iteration + 1) % 10 == 0:
            logger.debug(
                f"  ORTHOMIN iter {iteration + 1}: ||r||/||b|| = {residual_norm:.2e}"
            )

    # Max iterations reached
    logger.warning(
        f"ORTHOMIN: Max iterations ({maxiter}) reached, "
        f"||r||/||b|| = {residual_norm:.2e}"
    )
    return x, maxiter, float(residual_norm)  # type: ignore[return-value]


def solve_linear_system(
    jacobian: csr_matrix,
    rhs: np.ndarray,
    x0: typing.Optional[np.ndarray] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    maxiter: int = 100,
    use_ilu: bool = True,
    use_block_jacobi: bool = False,
    n_phases: typing.Optional[int] = None,
    n_cells: typing.Optional[int] = None,
    prefer_orthomin: bool = False,
) -> typing.Optional[np.ndarray]:
    """
    Flexible linear system solver with automatic fallback hierarchy.

    Attempts multiple solving strategies in order of sophistication:

    Strategy Hierarchy:
    -------------------
    1. ORTHOMIN with ILU preconditioner (if prefer_orthomin=True)
    2. LGMRES with ILU preconditioner (improved GMRES with restart)
    3. BiCGSTAB with ILU preconditioner (stabilized bi-conjugate gradient)
    4. GMRES with Block-Jacobi preconditioner (if use_block_jacobi=True)
    5. Unpreconditioned BiCGSTAB
    6. Direct sparse solver (last resort)

    This automatic fallback ensures robustness while attempting to use the
    most efficient method first.

    :param jacobian: Sparse Jacobian matrix (CSR format)
    :param rhs: Right-hand side vector
    :param x0: Initial guess (default: zero)
    :param rtol: Relative tolerance ||r|| / ||b||
    :param atol: Absolute tolerance ||r||
    :param maxiter: Maximum iterations for iterative solvers
    :param use_ilu: Attempt ILU preconditioning
    :param use_block_jacobi: Attempt Block-Jacobi preconditioning
    :param n_phases: Number of phases (required for Block-Jacobi)
    :param n_cells: Number of cells (required for Block-Jacobi)
    :param prefer_orthomin: Try ORTHOMIN before GMRES variants
    :return: Solution vector or None if all methods fail

    Convergence Criteria:
    ---------------------
    Converged if: ||r|| < max(rtol * ||b||, atol)

    Performance Notes:
    ------------------
    - ILU preconditioning: 2-10x speedup for moderate systems
    - ORTHOMIN: Best for highly non-symmetric systems (upwinding dominant)
    - LGMRES: Good general-purpose solver with flexible preconditioning
    - BiCGSTAB: Fast for mildly non-symmetric systems
    - Direct solver: O(n^3) complexity, only use as last resort

    Example:
        >>> J = build_jacobian(...)
        >>> rhs = -compute_residual(...)
        >>> delta_s = solve_linear_system(
        ...     J, rhs, rtol=1e-6, use_ilu=True, prefer_orthomin=True
        ... )
        >>> if delta_s is not None:
        ...     print("Converged")
        >>> else:
        ...     print("Failed to converge")
    """
    n = jacobian.shape[0]  # type: ignore[union-attr]
    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)

    rhs_norm = np.linalg.norm(rhs)
    convergence_tol = max(rtol * rhs_norm, atol)

    logger.debug(
        f"Solving linear system: n={n}, ||b||={rhs_norm:.2e}, tol={convergence_tol:.2e}"
    )

    # Strategy 1: ORTHOMIN with ILU (if preferred)
    if prefer_orthomin and use_ilu:
        try:
            logger.debug("Attempting ORTHOMIN with ILU preconditioner...")
            precond = ILUPreconditioner(jacobian, drop_tol=1e-4, fill_factor=20)
            M = precond.get_linear_operator()

            solution, iterations, residual = orthomin_solver(
                A=jacobian,
                b=rhs,
                x0=x0,
                M=M,
                tol=rtol,
                maxiter=maxiter,
            )

            if residual < rtol:
                logger.info(
                    f"ORTHOMIN+ILU converged in {iterations} iterations, "
                    f"||r||/||b||={residual:.2e}"
                )
                return solution
            else:
                logger.debug(
                    f"ORTHOMIN+ILU did not converge: "
                    f"||r||/||b||={residual:.2e} > {rtol:.2e}"
                )

        except Exception as e:
            logger.debug(f"ORTHOMIN+ILU failed: {e}")

    # Strategy 2: LGMRES with ILU
    if use_ilu:
        try:
            logger.debug("Attempting LGMRES with ILU preconditioner...")
            precond = ILUPreconditioner(jacobian, drop_tol=1e-4, fill_factor=20)
            M = precond.get_linear_operator()

            solution, info = lgmres(
                A=jacobian,
                b=rhs,
                x0=x0,
                M=M,
                atol=atol,
                maxiter=maxiter,
                inner_m=30,  # Inner iterations before restart
                outer_k=3,  # Number of vectors to carry between restarts
            )

            residual_norm = np.linalg.norm(jacobian @ solution - rhs)

            if info == 0:
                logger.info(f"LGMRES+ILU converged, ||r||={residual_norm:.2e}")
                return solution
            else:
                logger.debug(
                    f"LGMRES+ILU did not converge (info={info}), "
                    f"||r||={residual_norm:.2e}"
                )

        except Exception as e:
            logger.debug(f"LGMRES+ILU failed: {e}")

    # Strategy 3: BiCGSTAB with ILU
    if use_ilu:
        try:
            logger.debug("Attempting BiCGSTAB with ILU preconditioner...")
            precond = ILUPreconditioner(jacobian, drop_tol=1e-4, fill_factor=15)
            M = precond.get_linear_operator()

            solution, info = bicgstab(
                A=jacobian,
                b=rhs,
                x0=x0,
                M=M,
                atol=atol,
                maxiter=maxiter,
            )

            residual_norm = np.linalg.norm(jacobian @ solution - rhs)

            if info == 0:
                logger.info(f"BiCGSTAB+ILU converged, ||r||={residual_norm:.2e}")
                return solution
            else:
                logger.debug(
                    f"BiCGSTAB+ILU did not converge (info={info}), "
                    f"||r||={residual_norm:.2e}"
                )

        except Exception as e:
            logger.debug(f"BiCGSTAB+ILU failed: {e}")

    # Strategy 4: GMRES with Block-Jacobi
    if use_block_jacobi and n_phases is not None and n_cells is not None:
        try:
            logger.debug("Attempting GMRES with Block-Jacobi preconditioner...")
            precond = BlockJacobiPreconditioner(
                jacobian, n_phases=n_phases, n_cells=n_cells
            )
            M = precond.get_linear_operator()

            solution, info = gmres(
                A=jacobian,
                b=rhs,
                x0=x0,
                M=M,
                atol=atol,
                maxiter=maxiter,
                restart=30,
            )

            residual_norm = np.linalg.norm(jacobian @ solution - rhs)

            if info == 0:
                logger.info(f"GMRES+BlockJacobi converged, ||r||={residual_norm:.2e}")
                return solution
            else:
                logger.debug(
                    f"GMRES+BlockJacobi did not converge (info={info}), "
                    f"||r||={residual_norm:.2e}"
                )

        except Exception as e:
            logger.debug(f"GMRES+BlockJacobi failed: {e}")

    # Strategy 5: Unpreconditioned BiCGSTAB
    try:
        logger.debug("Attempting unpreconditioned BiCGSTAB...")
        solution, info = bicgstab(
            A=jacobian,
            b=rhs,
            x0=x0,
            atol=atol,
            maxiter=maxiter * 2,  # Allow more iterations without preconditioner
        )

        residual_norm = np.linalg.norm(jacobian @ solution - rhs)

        if info == 0:
            logger.info(
                f"BiCGSTAB (unpreconditioned) converged, ||r||={residual_norm:.2e}"
            )
            return solution
        else:
            logger.debug(
                f"BiCGSTAB did not converge (info={info}), ||r||={residual_norm:.2e}"
            )

    except Exception as e:
        logger.debug(f"BiCGSTAB failed: {e}")

    # Strategy 6: Direct solver (last resort)
    logger.warning("All iterative methods failed, using direct solver...")
    try:
        solution = spsolve(jacobian, rhs)
        residual_norm = np.linalg.norm(jacobian @ solution - rhs)
        logger.info(f"Direct solver succeeded, ||r||={residual_norm:.2e}")
        return np.asarray(solution)  # type: ignore[return-value]

    except Exception as e:
        logger.error(f"Direct solver failed: {e}")
        return None
