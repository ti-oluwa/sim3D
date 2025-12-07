# Fully Implicit Solver — Detailed Debugging and Fix Instructions

This document contains step-by-step, machine-actionable debugging instructions and code patches to fix the non-convergence of the fully implicit Newton solver in your codebase. All names use your existing, fully descriptive naming convention (no abbreviations): `pressure_grid`, `oil_saturation_grid`, `gas_saturation_grid`, `water_saturation_grid`, `relative_mobility_grids`, `capillary_pressure_grids`, `compute_residuals_for_cell`, `assemble_jacobian_matrix`, `apply_newton_update_with_clamping`, and so on.

Target audience: an AI coding agent that will apply changes directly to the repository. Each step is explicit, includes minimal code patches, and explains why the change is required.

---

## Contents

1. Objective and summary of the root causes
2. Fixes (ordered by priority) with code patches
3. Changes required in `compute_flux_divergence_for_cell`
4. Changes required in `assemble_jacobian_matrix` (finite-difference Jacobian)
5. Local recomputation pattern for dependent grids during finite-difference
6. Finite-difference Jacobian validation script (machine-run)
7. Adaptive perturbation rules and numerical tolerances
8. Diagnostics and logging additions
9. Recommended solver runtime improvements (line search, damping)
10. Prioritized checklist for implementation and verification

---

## 1. Objective and summary of the root causes

The fully implicit solver fails to converge because the numerical Jacobian assembled with finite differences is inconsistent with how residuals are evaluated. Main problems:

* When perturbing oil or gas saturations for finite-difference columns, the implied change to water saturation (`water_saturation_grid = 1 - oil_saturation_grid - gas_saturation_grid`) is not applied. This produces wrong finite-difference columns.
* When perturbing pressure or saturations, dependent precomputed grids (`relative_mobility_grids`, `capillary_pressure_grids`, PVT-derived density/viscosity grids, solution gas-oil ratio grid) are not recomputed for the perturbed state, causing mismatched residual evaluations.
* Dissolved gas transported with oil flux uses only the local solution gas-oil ratio at the sender cell; transport must use an interface-average solution gas-oil ratio.
* Small mistakes in FD perturbation magnitudes and units (solution GOR units) amplify mismatch and create large Jacobian errors.

Fix these items first in the exact order given below.

---

## 2. Fixes (ordered by priority) with code patches

### Fix 1 — Conserve mass when perturbing saturations for finite-difference Jacobian

**Why:** Finite-difference approximations must respect the algebraic constraint `Sw = 1 − So − Sg`. When `So` or `Sg` is perturbed, `Sw` must be updated accordingly. Without this, columns of the Jacobian are incorrect.

**Where to change:** `assemble_jacobian_matrix`, in every place that perturbs `oil_saturation_grid` or `gas_saturation_grid`.

**Patch (drop-in helper to add near top of file):**

```python
def make_saturation_perturbation(
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    i: int,
    j: int,
    k: int,
    delta_oil_saturation: float = 0.0,
    delta_gas_saturation: float = 0.0,
) -> tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Return new saturation grids after perturbing a single cell (i,j,k).

    Perturbation conserves mass by applying the negative sum to water saturation
    and performs local clamping and renormalization of the three saturations.
    """
    oil_saturation_perturbed = oil_saturation_grid.copy()
    gas_saturation_perturbed = gas_saturation_grid.copy()
    water_saturation_perturbed = water_saturation_grid.copy()

    oil_saturation_perturbed[i, j, k] += delta_oil_saturation
    gas_saturation_perturbed[i, j, k] += delta_gas_saturation
    water_saturation_perturbed[i, j, k] -= (delta_oil_saturation + delta_gas_saturation)

    # clamp and renormalize the perturbed cell to avoid numerical issues
    eps = 1e-12
    so = np.clip(oil_saturation_perturbed[i, j, k], eps, 1.0 - eps)
    sg = np.clip(gas_saturation_perturbed[i, j, k], eps, 1.0 - eps)
    sw = np.clip(water_saturation_perturbed[i, j, k], eps, 1.0 - eps)
    total = so + sg + sw
    so /= total; sg /= total; sw = 1.0 - so - sg

    oil_saturation_perturbed[i, j, k] = so
    gas_saturation_perturbed[i, j, k] = sg
    water_saturation_perturbed[i, j, k] = sw

    return oil_saturation_perturbed, gas_saturation_perturbed, water_saturation_perturbed
```

**How to apply:** Replace any direct `oil_saturation_grid.copy()` then `oil_saturation_grid[i,j,k] += epsilon` pattern with a call to `make_saturation_perturbation(..., delta_oil_saturation=eps)` and pass the returned three-grid tuple into `compute_residuals_for_cell`.

### Fix 2 — Pass the updated water saturation grid to `compute_residuals_for_cell` whenever you perturb So or Sg

**Why:** You currently perturb oil or gas only and continue to pass the original `water_saturation_grid` into `compute_residuals_for_cell`. Residual evaluation uses `water_saturation_grid` explicitly; mismatch produces incorrect finite differences.

**Patch (example replacement inside diagonal So perturb block):**

```python
# old: created oil_saturation_perturbed only and then passed original water_saturation_grid
# new: use make_saturation_perturbation
so_p, sg_p, sw_p = make_saturation_perturbation(
    oil_saturation_grid, gas_saturation_grid, water_saturation_grid, i, j, k, delta_oil_saturation=epsilon_saturation
)
perturbed_residuals = compute_residuals_for_cell(
    i=i, j=j, k=k, ...,
    pressure_grid=pressure_grid,
    oil_saturation_grid=so_p,
    gas_saturation_grid=sg_p,
    water_saturation_grid=sw_p,
    ...
)
```

Apply analogous changes for gas saturation perturbations and for neighbor perturbations.

### Fix 3 — Use interface-average solution gas-oil ratio when computing dissolved gas transport

**Why:** Dissolved gas transported with oil across an interface should use an interface-averaged solution gas-oil ratio. Using only the sender cell's solution gas-oil ratio is inconsistent and leads to physical mistakes and Jacobian errors.

**Where to change:** `compute_flux_divergence_for_cell` where `gas_flux_total` is computed.

**Patch:**

```python
# inside the neighbor loop in compute_flux_divergence_for_cell
solution_gor_interface = 0.5 * (
    solution_gor_grid[i, j, k] + solution_gor_grid[ni, nj, nk]
)
# ensure solution_gor_interface is in reservoir gas-volume-per-reservoir-oil-volume (ft^3 gas / ft^3 oil)
# If solution_gor is in standard units (scf/STB) convert before use using formation-volume-factor grid, check constants.py to see what conversion constants are needed

gas_flux_total = gas_flux_free + (oil_flux * solution_gor_interface)
```

If you use a mobility-weighted or harmonic averaging for other interface properties, replace `0.5*()` with the same weighting function for `solution_gor_interface` for consistent transport.

### Fix 4 — Ensure `compute_harmonic_mobility` returns a face (interface) mobility that uses both cell values

**Why:** The flux discretization assumes interface mobility. Finite-difference perturbations also assume an interface quantity. If `compute_harmonic_mobility(index1, index2, mobility_grid)` returns a per-cell mobility rather than an interface value, the Jacobian will be inconsistent.

**Action item:** Inspect `compute_harmonic_mobility` implementation. It must compute the harmonic mean between the two cell mobilities and include transmissibility-distance factors consistently. If it does not (I believe it does though), implement this interface harmonic mean:

```python
def compute_harmonic_mobility(index1, index2, mobility_grid):
    m1 = mobility_grid[index1]
    m2 = mobility_grid[index2]
    if m1 <= 0 or m2 <= 0:
        return 0.0
    return 2.0 / (1.0 / m1 + 1.0 / m2)
```

Use the same geometry/distance convention used to compute the `geometric_factor` in `compute_flux_divergence_for_cell`.

### Fix 5 — Adaptive perturbation magnitudes for finite differences

**Why:** Absolute perturbation values can be too small relative to machine precision or too large relative to variable scale (saturations near 0 or 1). Use relative/adaptive perturbations.

**Recommendation (function to add):**

```python
def compute_perturbation_magnitude_for_saturation(value: float) -> float:
    return max(1e-8, 1e-6 * max(1.0, abs(value)))

def compute_perturbation_magnitude_for_pressure(value: float) -> float:
    return max(1e-6, 1e-6 * max(1.0, abs(value)))
```

Replace constant `epsilon_saturation` and `epsilon_pressure` with these functions when building finite differences.

### Fix 6 — Units check and conversion for solution gas-oil ratio

**Why:** The code uses `BBL_TO_FT3` to convert liquid rates to ft³/day and then multiplies oil flux by `solution_gor`. Verify `solution_gor` is expressed in the same volumetric basis.

**Action:** If `solution_gor_grid` is `scf/STB` (standard ft³ gas per stock-tank bbl), convert before multiplying by oil flux in reservoir ft³/day:

Ensure `compute_flux_divergence_for_cell` multiplies quantities with consistent units.

### Fix 7 — Recompute dependent precomputed grids for perturbed states used by finite differences

**Why:** `compute_residuals_for_cell` relies on `relative_mobility_grids`, `capillary_pressure_grids`, and PVT grids that depend on `pressure_grid` and saturations. When you compute finite-difference residuals at a perturbed state, you must use mobility and capillary-pressure values consistent with that perturbed state.

**Approach A (preferred for FD):** Recompute only the local dependent values that change when perturbing a single cell and its immediate neighbors. This is cheap and accurate.

**Pattern to implement (helper function skeleton):**

```python
def recompute_local_rock_fluid_properties(
    relative_mobility_grids: RelativeMobilityGrids,
    capillary_pressure_grids: CapillaryPressureGrids,
    i: int, j: int, k: int,
    perturbed_pressure_grid: ThreeDimensionalGrid,
    perturbed_oil_saturation_grid: ThreeDimensionalGrid,
    perturbed_gas_saturation_grid: ThreeDimensionalGrid,
    perturbed_water_saturation_grid: ThreeDimensionalGrid,
    rock_fluid_models: RockFluidProperties,
    fluid_models: FluidProperties,
) -> tuple[RelativeMobilityGrids, CapillaryPressureGrids]:
    """
    Recompute only the mobility and capillary pressure entries that depend on cell (i,j,k)
    and its six neighbors. Return new small copies or modified copies suitable for passing
    into compute_residuals_for_cell for FD.
    """
    # Implementation details depend on how your rock-fluid model computes kr and pc;
    # the function must update mobility grids at indices (i,j,k) and all neighbor faces
    # that reference the perturbed saturations.
    # Return modified copies of the required grids.
    raise NotImplementedError
```

**How to use:** After creating the perturbed `pressure_grid`, `oil_saturation_grid`, `gas_saturation_grid`, and `water_saturation_grid` for a single FD perturbation, call `recompute_local_rock_fluid_properties` and pass the returned mobility and capillary-pressure grids into `compute_residuals_for_cell`.

### Fix 8 — Ensure `apply_newton_update_with_clamping` returns a consistent `water_saturation_grid` and that the next Jacobian build uses the consistent grid

**Why:** The evolution loop recomputes `water_saturation_grid = 1 - oil_saturation_grid - gas_saturation_grid` after the Newton update. Confirm the Jacobian assembly uses the same `water_saturation_grid` values when computing FD checks and solving the next Newton iteration.

**Patch:** After `apply_newton_update_with_clamping` returns, set `water_saturation_grid = 1.0 - oil_saturation_grid - gas_saturation_grid` and clamp/renormalize immediately. Do this before recomputing `relative_mobility_grids` and `capillary_pressure_grids`.

```python
water_saturation_grid = 1.0 - oil_saturation_grid - gas_saturation_grid
# clamp and renormalize whole grid in-place if small violations exist
eps = 1e-12
np.clip(water_saturation_grid, eps, 1.0 - eps, out=water_saturation_grid)
# optional renormalize per cell if sum != 1 due to rounding
```

## 3. Changes required in `compute_flux_divergence_for_cell`

Make these targeted changes:

1. Add `solution_gor_grid` argument if it is not already passed. Use interface-average as shown in Fix 3.
2. Convert `solution_gor_interface` to reservoir volumetric units if your `solution_gor_grid` is in standard units. Use `oil_formation_volume_factor_grid` or `oil_formation_volume_factor_grid` at interface to convert.
3. Ensure `compute_harmonic_mobility` is used and that it returns an interface mobility — update or replace as necessary (see Fix 4).
4. Ensure the sign convention and geometric factor are exactly the same as used by your Jacobian FD routine.

Add explicit assertion checks and debug logging (only when `options.debug_mode` is true) to print interface values for one face per iteration for a few iterations.

## 4. Changes required in `assemble_jacobian_matrix` (finite-difference Jacobian)

Modify the finite-difference code to:

* Use `compute_perturbation_magnitude_for_pressure` and `compute_perturbation_magnitude_for_saturation` when forming `delta_p` and `epsilon_saturation`.
* When perturbing saturations call `make_saturation_perturbation` and recompute local `relative_mobility_grids` and `capillary_pressure_grids` for the perturbed state before calling `compute_residuals_for_cell`.
* When perturbing neighbor cells do the same (perturb neighbor saturations and recompute local dependent grids that affect the residual at the central cell).
* Store finite-difference derivatives into the Jacobian with identical ordering and scaling as used by the residual vector (if residuals are scaled by `1/cell_volume` the Jacobian FD must use the same residual definition).

**Concrete replacement example for perturbing neighbor oil saturation (inside neighbor loop):**

```python
# BEFORE: changed only oil saturation at neighbor then passed original sw grid
# AFTER: use make_saturation_perturbation and recompute local rock-fluid properties
so_pert, sg_pert, sw_pert = make_saturation_perturbation(
    oil_saturation_grid,
    gas_saturation_grid,
    water_saturation_grid,
    ni, nj, nk,
    delta_oil_saturation=epsilon_saturation_neighbor,
)
# Recompute local rel_mob_neighbor and cap_pc_neighbor only for indices affected
rel_mob_for_fd, cap_pc_for_fd = recompute_local_rock_fluid_properties(
    relative_mobility_grids,
    capillary_pressure_grids,
    ni, nj, nk,
    pressure_grid,
    so_pert, sg_pert, sw_pert,
    rock_fluid_properties,
    fluid_properties,
)
perturbed_residuals_neighbor = compute_residuals_for_cell(
    i=i, j=j, k=k,
    cell_dimension=cell_dimension,
    thickness_grid=thickness_grid,
    elevation_grid=elevation_grid,
    time_step_size=time_step_size,
    rock_properties=rock_properties,
    fluid_properties=fluid_properties,
    old_fluid_properties=old_fluid_properties,
    rock_fluid_properties=rock_fluid_properties,
    relative_mobility_grids=rel_mob_for_fd,
    capillary_pressure_grids=cap_pc_for_fd,
    pressure_grid=pressure_grid,
    oil_saturation_grid=so_pert,
    gas_saturation_grid=sg_pert,
    water_saturation_grid=sw_pert,
    wells=wells,
    options=options,
)
# then compute FD column as (perturbed - base)/epsilon
```

## 5. Local recomputation pattern for dependent grids during finite-difference

Implement `recompute_local_rock_fluid_properties` such that:

* It accepts copies of `relative_mobility_grids` and `capillary_pressure_grids` (or creates shallow copies) and updates only the entries that depend on the perturbed cell and its neighbors (6 faces).
* If your relative-permeability model requires neighboring saturations for slope or smoothing, update those neighbor cells as well.
* The recomputation should call the same functions your regular grid builder uses for a single-cell or local update, to guarantee consistency.

This is a local, deterministic routine. Example pseudo-logic:

1. Identify indices to update: `[(i,j,k)] + [(i+1,j,k), (i-1,j,k), (i,j+1,k), ...]`.
2. For each index in that list, recompute relative permeability and mobility entries using the perturbed saturations and pressure.
3. For interface mobilities (used by fluxes) recompute harmonic means using updated per-cell mobilities.

Return the updated mobility grids and capillary grids.

## 6. Finite-difference Jacobian validation script (machine-run)

Add a unit-test-style script that runs automatically in debug mode to validate a small set of Jacobian columns. Save it in repository under `tests/test_jacobian_columns.py` and call it from CI when `options.debug_mode` is enabled.

**Validation script (complete):**

```python
# tests/test_jacobian_columns.py
import numpy as np
from scipy.sparse import csr_matrix

def build_residual_vector_full(
    pressure_grid, oil_saturation_grid, gas_saturation_grid, water_saturation_grid,
    relative_mobility_grids, capillary_pressure_grids, ...
):
    # Build full residual vector using compute_residuals_for_cell for every interior cell
    # Follow the exact ordering used by your main solver (to_1D_index_interior_only)
    # Return vector R as numpy array
    pass

def test_jacobian_column_agreement(
    pressure_grid, oil_saturation_grid, gas_saturation_grid, water_saturation_grid,
    relative_mobility_grids, capillary_pressure_grids, jacobian_matrix, indices_to_test
):
    base_R = build_residual_vector_full(
        pressure_grid, oil_saturation_grid, gas_saturation_grid, water_saturation_grid,
        relative_mobility_grids, capillary_pressure_grids
    )

    jacobian_csr = jacobian_matrix.tocsr()

    for (ci, cj, ck, phase_index) in indices_to_test:
        idx = to_1D_index_interior_only(ci, cj, ck, *pressure_grid.shape)
        col_index = 3 * idx + phase_index

        # create perturbed grids
        p_grid_p = pressure_grid.copy()
        so_p = oil_saturation_grid.copy()
        sg_p = gas_saturation_grid.copy()
        sw_p = water_saturation_grid.copy()

        if phase_index == 0:
            dp = compute_perturbation_magnitude_for_pressure(p_grid_p[ci, cj, ck])
            p_grid_p[ci, cj, ck] += dp
        elif phase_index == 1:
            ds = compute_perturbation_magnitude_for_saturation(so_p[ci, cj, ck])
            so_p, sg_p, sw_p = make_saturation_perturbation(so_p, sg_p, sw_p, ci, cj, ck, delta_oil_saturation=ds)
        else:
            ds = compute_perturbation_magnitude_for_saturation(sg_p[ci, cj, ck])
            so_p, sg_p, sw_p = make_saturation_perturbation(so_p, sg_p, sw_p, ci, cj, ck, delta_gas_saturation=ds)

        # recompute local dependent grids
        rel_mob_p, cap_pc_p = recompute_local_rock_fluid_properties(
            relative_mobility_grids, capillary_pressure_grids,
            ci, cj, ck, p_grid_p, so_p, sg_p, sw_p, rock_fluid_properties, fluid_properties
        )

        R_pert = build_residual_vector_full(p_grid_p, so_p, sg_p, sw_p, rel_mob_p, cap_pc_p)

        fd_col = (R_pert - base_R) / (dp if phase_index == 0 else ds)
        analytic_col = jacobian_csr[:, col_index].toarray().ravel()

        rel_error = np.linalg.norm(analytic_col - fd_col) / (np.linalg.norm(fd_col) + 1e-16)

        print(f"Test column index {col_index}: relative error {rel_error:.3e}")
        assert rel_error < 1e-3, f"Jacobian column {col_index} mismatch: {rel_error:.3e}"
```

**How to use:** call `test_jacobian_column_agreement` for several representative cells (center cell, cell near injection well, cell near production well, cell near boundary). Fix analytic FD until test passes.

## 7. Adaptive perturbation rules and numerical tolerances

* Use `compute_perturbation_magnitude_for_saturation` and `compute_perturbation_magnitude_for_pressure` functions described in Fix 5.
* Use machine epsilon-based relative tolerances when solving the linear system. Keep `rtol` and `atol` scaled by the `rhs` norm as you currently do, but ensure they are not overly loose: `rtol = max(1e-8, epsilon * 50)` is fine; `atol = max(1e-10, 1e-8 * rhs_norm)`.
* When computing FD relative error thresholds, use `1e-3` as the pass threshold for the most important columns; a lower threshold (1e-4) is better if numerical perturbations are well-controlled.

## 8. Diagnostics and logging additions

Add logging (only when `options.debug_mode` is True):

* After mapping interior indices, assert `len(np.unique(mapped_indices)) == total_unknowns` and log `total_unknowns` and index min/max.
* After computing the residual vector in the first Newton iteration log:

  * `initial_residual_norm`, `max_abs_residual`, `mean_abs_residual` (already present). Also log the top 5 cells with largest absolute residuals and their (i,j,k) coordinates and residual vector components.
* After assembling the Jacobian, compute `row_norms = np.sqrt((jacobian_csr.power(2)).sum(axis=1))` or use `np.linalg.norm` on dense rows for small tests; log min and max row norms and the number of zero rows.
* For FD validation run print the per-column relative error and fail early if any exceed `1e-2`.

Example diagnostic snippet to add immediately after Jacobian assembly:

```python
jac_csr = jacobian.tocsr()
row_norms = np.sqrt(jac_csr.multiply(jac_csr).sum(axis=1)).A1
logger.debug("Jacobian row norms: min=%e, max=%e, zeros=%d", row_norms.min(), row_norms.max(), (row_norms==0).sum())
```

## 9. Recommended solver runtime improvements (line search, damping)

If FD/analytic Jacobian agreement is achieved but Newton still diverges for large time steps, implement a backtracking line search or stronger damping in `apply_newton_update_with_clamping`.

**Minimal line search pattern (insert before accepting Newton update):**

```python
alpha = 1.0
orig_pressure, orig_so, orig_sg = pressure_grid.copy(), oil_saturation_grid.copy(), gas_saturation_grid.copy()
base_res_norm = np.linalg.norm(current_residual_vector)
for _ in range(10):
    trial_updates = alpha * update_vector
    trial_state = apply_updates_and_return_state(orig_pressure, orig_so, orig_sg, trial_updates)
    # ensure water saturation is consistent and dependent grids recomputed
    trial_water = 1.0 - trial_state.oil_saturation_grid - trial_state.gas_saturation_grid
    rel_mob_trial, cap_pc_trial = recompute_local_rock_fluid_properties(..., trial_state)
    trial_res = build_residual_vector_full(trial_state.pressure_grid, trial_state.oil_saturation_grid, trial_state.gas_saturation_grid, trial_water, rel_mob_trial, cap_pc_trial)
    trial_norm = np.linalg.norm(trial_res)
    if trial_norm < base_res_norm:
        accept trial_state
        break
    alpha *= 0.5
```

If backtracking accepts only very small alpha values repeatedly, reduce the time step and/or implement pseudo-transient continuation.

## 10. Prioritized checklist for implementation and verification

1. Implement `make_saturation_perturbation` and replace all saturation FD perturbations. Verify `assemble_jacobian_matrix` uses the helper.
2. Implement `compute_perturbation_magnitude_for_saturation` and `compute_perturbation_magnitude_for_pressure` and use them.
3. Implement `recompute_local_rock_fluid_properties` to update the mobility and capillary-pressure entries affected by a single-cell perturbation.
4. Update all FD calls in `assemble_jacobian_matrix` to pass the perturbed `water_saturation_grid` and recomputed dependent grids.
5. Update `compute_flux_divergence_for_cell` to use interface-averaged `solution_gor_interface` and ensure units are consistent (convert `solution_gor_grid` to reservoir volume basis if needed).
6. Add the Jacobian validation script and run tests on representative cells. Fix analytic FD entries until relative error < `1e-3` for tested columns.
7. Add diagnostic logs for Jacobian row norms and worst residual cells.
8. Implement line-search/backtracking if Newton still fails for large time steps. Combine with adaptive time stepping (reduce time step when Newton fails repeatedly).
