# Performance

## Overview

Reservoir simulation performance is dominated by two costs: assembling the coefficient matrices from fluid properties and transmissibilities, and solving the resulting linear systems. Everything else (well calculations, saturation updates, I/O) is typically a small fraction of total runtime. Understanding where time is spent lets you focus optimization effort where it matters and avoid changes that have negligible impact.

BORES is designed to be fast for its class of simulator. The core numerical routines use Numba JIT compilation for near-C performance, the linear algebra relies on SciPy's sparse solvers backed by optimized LAPACK/BLAS libraries, and the PVT correlations operate on entire grids in a single vectorized call. Despite this, simulation of large models over many timesteps can still take significant time, and there are several configuration choices that materially affect performance.

This page covers the main performance levers available to you: precision settings, grid sizing, preconditioner caching, timestep optimization, and memory management. Each section explains the trade-off involved and provides concrete recommendations.

---

## Precision Settings

BORES defaults to 32-bit (single precision) floating-point arithmetic. This is controlled by `bores.use_32bit_precision()`, which is called automatically when the package is imported. You can switch to 64-bit with `bores.use_64bit_precision()`.

### 32-bit vs 64-bit Performance

Single precision is faster than double precision for two reasons: it uses half the memory (meaning more data fits in CPU cache) and modern CPUs can process twice as many 32-bit operations per clock cycle as 64-bit operations. In practice, 32-bit mode is typically 30 to 50% faster than 64-bit for the same model.

| Aspect | 32-bit (float32) | 64-bit (float64) |
| --- | --- | --- |
| Memory per cell | ~50% less | Baseline |
| Computation speed | 30-50% faster | Baseline |
| Accuracy | ~7 significant digits | ~15 significant digits |
| Solver convergence | May need more iterations | Fewer iterations |
| Numerical stability | Adequate for most problems | Required for ill-conditioned problems |

### When to Use Each

**Use 32-bit (default) when:**

- Grid has fewer than 500,000 cells
- Permeability contrasts are moderate (within 3 to 4 orders of magnitude)
- Solver converges comfortably within the iteration limit
- You want faster development iteration

**Use 64-bit when:**

- Solver convergence is marginal (iteration count near the maximum)
- Extreme permeability contrasts (> 4 orders of magnitude)
- Very tight convergence tolerances (< 1e-8)
- Capillary pressure gradients are important for accuracy
- You see oscillations or non-physical results in 32-bit

```python
import bores
import numpy as np

# Switch to 64-bit for a challenging problem
bores.use_64bit_precision()

model = bores.reservoir_model(...)

# Or use temporarily with a context manager
with bores.with_precision(np.float64):
    model = bores.reservoir_model(...)
    # 64-bit inside this block
# Back to 32-bit outside
```

---

## Grid Size Impact

The number of cells in your grid is the primary driver of both memory usage and computation time. The relationship is roughly linear for memory and slightly super-linear for computation (because solver convergence can degrade for larger problems).

| Grid Cells | Typical Memory | Relative Speed |
| --- | --- | --- |
| 1,000 | ~10 MB | 1x (baseline, very fast) |
| 10,000 | ~50 MB | ~5x slower |
| 100,000 | ~500 MB | ~30-50x slower |
| 1,000,000 | ~5 GB | ~300-500x slower |

These numbers are approximate and depend on the number of phases, the evolution scheme, and the solver configuration. Fully implicit schemes use more memory per cell than IMPES because they store the full Jacobian matrix.

### Practical Recommendations

1. **Start with the coarsest grid that captures your physics.** A 10x10x5 model runs in seconds and lets you verify that the model setup is correct before scaling up.

2. **Use `coarsen_grid` for quick sensitivity runs.** Coarsen a fine model by 2x in each direction for parameter sweeps, then run the final cases at full resolution.

3. **Profile before optimizing.** If a simulation is slow, check whether the bottleneck is the solver (increase preconditioner strength or cache it), the property calculations (use PVT tables instead of correlations), or the I/O (reduce save frequency).

---

## Preconditioner Caching

Building a preconditioner is expensive. ILU factorization has a cost comparable to 5 to 10 solver iterations, and AMG setup can cost as much as 50 to 100 iterations. Since the coefficient matrix changes modestly from one timestep to the next (only the values change, not the sparsity structure), reusing a previously built preconditioner for several timesteps can save substantial time.

The `CachedPreconditionerFactory` wraps any preconditioner factory and adds caching logic:

```python
import bores

# Cache the ILU preconditioner, rebuild every 10 steps or when the matrix
# changes by more than 30%
cached = bores.CachedPreconditionerFactory(
    factory="ilu",
    update_frequency=10,
    recompute_threshold=0.3,
)
cached.register(override=True)
```

### Tuning Cache Parameters

**`update_frequency`**: How many timesteps between forced rebuilds. Higher values save more time but risk using a stale preconditioner that increases solver iteration counts. Start with 10 and adjust based on whether solver iterations increase over time.

**`recompute_threshold`**: Relative change in matrix values that triggers a rebuild, regardless of `update_frequency`. A threshold of 0.3 means the preconditioner is rebuilt when the matrix values change by more than 30% from the last build. Lower values keep the preconditioner fresh at higher cost.

For stable simulations (constant rate production, steady boundary conditions), you can increase `update_frequency` to 20 or even 50. For transient simulations (well shutins, rate changes, front arrivals), keep it at 5 to 10.

!!! tip "AMG Caching"

    AMG benefits the most from caching because its setup cost is the highest. If you use AMG preconditioning, always wrap it with `CachedPreconditionerFactory`. The setup cost for AMG is typically 5 to 10x higher than ILU, so even caching for 5 steps can cut total solver time in half.

---

## Timestep Optimization

The total number of timesteps directly affects runtime. Fewer, larger timesteps are faster, but accuracy and stability impose upper bounds. The adaptive timer in BORES handles this automatically, but you can influence its behavior:

1. **Set a generous `max_step_size`.** If you know your problem has no short-timescale transients, allow larger maximum timesteps (30 to 90 days for multi-year simulations).

2. **Use `ramp_up_factor`.** Setting this to 1.1 or 1.2 lets the timer grow the timestep faster after a period of stability, reaching the maximum step size sooner.

3. **Avoid unnecessary restarts.** Each time the simulation reduces the timestep (due to CFL violation or solver failure), it takes several steps to ramp back up. Reducing the frequency of these events by using a slightly smaller `max_step_size` can actually reduce total runtime compared to an aggressive setting that causes frequent backoffs.

4. **Consider fully implicit for long simulations.** If your simulation runs for 20+ years with 1-day timesteps under IMPES, switching to fully implicit with 30 to 90 day timesteps can reduce the total step count by 10 to 30x, more than offsetting the higher cost per step.

---

## PVT Tables vs Correlations

PVT property calculations are called at every timestep for every cell. Correlation functions (like Standing's Bo or the Dranchuk-Abou-Kassem Z-factor) involve floating-point arithmetic that, while fast, adds up over millions of evaluations. PVT tables replace these calculations with interpolation lookups that are typically 2 to 5x faster.

```python
import bores

# Build PVT tables at model construction time
pvt_tables = bores.PVTTables(
    oil_fvf=bores.build_pvt_table_data(pressures, bo_values),
    gas_fvf=bores.build_pvt_table_data(pressures, bg_values),
    # ... other properties
)

config = bores.Config(
    pvt_tables=pvt_tables,
    # ...
)
```

The accuracy trade-off depends on how many pressure points you include in the table. With 50 to 100 pressure points spanning the expected range, linear interpolation reproduces correlation values to within 0.1% for most properties. Cubic interpolation is available for even higher accuracy.

For small models, the speedup from PVT tables is negligible. For models with more than 50,000 cells running for hundreds of timesteps, it can reduce total runtime by 10 to 20%.

---

## Memory Management

For large models, memory can become a constraint. Here are strategies to reduce memory usage:

1. **Use 32-bit precision.** This halves the memory for all grid arrays. For a million-cell model, this can save 2 to 3 GB.

2. **Reduce save frequency.** If you are streaming results to a `StateStream`, saving every timestep can consume significant memory and disk space. Save every 5th or 10th step for long simulations unless you need fine temporal resolution.

3. **Use HDF5 or Zarr storage.** These backends compress data efficiently. HDF5 with gzip compression typically achieves 3 to 5x compression on reservoir simulation data. Zarr supports chunked storage that can reduce peak memory usage.

4. **Avoid holding all states in memory.** If you are processing results programmatically, iterate over the simulation generator and process each step individually rather than collecting all steps into a list.

```python
# Good: process each step as it comes
for step in bores.run(model=model, config=config):
    update_metrics(step)

# Avoid: collecting all steps into memory
all_steps = list(bores.run(model=model, config=config))  # High memory
```

---

## Numba Compilation Overhead

Many BORES functions use Numba JIT compilation. The first call to a Numba-compiled function triggers compilation, which takes 1 to 5 seconds per function. This is a one-time cost that is cached to disk, so subsequent runs in the same environment are fast.

If you notice slow startup times, this is likely Numba compilation. It happens once per Python environment and is cached in the `__pycache__` directories. Deleting these caches (for example, after upgrading BORES) will trigger recompilation on the next run.

For benchmarking, always exclude the first run (which includes compilation) from your timing measurements, or run a warm-up step first.

---

## Quick Reference

| Optimization | Impact | Effort | When to Use |
| --- | --- | --- | --- |
| 32-bit precision | 30-50% faster, 50% less memory | One line of code | Default, always |
| Preconditioner caching | 20-40% faster solver | 3-5 lines of code | Medium and large models |
| PVT tables | 10-20% faster | Moderate setup | Large models, many timesteps |
| Coarser grid | Proportional to cell count | Model redesign | Screening, sensitivity |
| Fully implicit scheme | Fewer total timesteps | Configuration change | Long simulations, large timesteps |
| HDF5/Zarr compression | Reduces disk and I/O | One line of code | All saved simulations |
