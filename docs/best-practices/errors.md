# Error Handling

## Overview

BORES uses a structured exception hierarchy to communicate what went wrong and where. Every BORES-specific error inherits from `BORESError`, so you can catch all framework errors with a single except clause when you need broad error handling, or catch specific subclasses when you want targeted recovery logic. Understanding what each error type means and what typically causes it saves significant debugging time.

This page covers the error hierarchy, common error scenarios you are likely to encounter, and practical strategies for diagnosing and fixing each one. The errors are organized roughly in the order you are most likely to encounter them: validation errors during model setup, solver and simulation errors during execution, and serialization errors during save/load operations.

All BORES errors include descriptive messages that tell you what failed and often suggest corrective action. When you encounter an error, read the full message carefully before diving into debugging. In many cases, the message itself contains the fix.

---

## Error Hierarchy

```
Exception
├── BORESError                    # Base for all BORES errors
│   ├── ValidationError           # Invalid inputs (also inherits ValueError)
│   ├── ComputationError          # Numerical computation failure
│   ├── SolverError               # Linear solver non-convergence
│   ├── PreconditionerError       # Preconditioner build/apply failure
│   ├── SimulationError           # Simulation execution failure
│   │   └── TimingError           # Timer configuration or state error
│   ├── StorageError              # Storage backend failure
│   ├── StreamError               # Data streaming failure
│   └── SerializableError         # Serialization protocol error
│       ├── SerializationError    # Error during serialization (to dict)
│       └── DeserializationError  # Error during deserialization (from dict)
└── StopSimulation                # Graceful simulation halt (not an error)
```

Note that `StopSimulation` inherits directly from `Exception`, not from `BORESError`. This is intentional: it is a control flow mechanism, not an error condition. You raise it to stop a simulation early (for example, when a recovery target is reached), and the simulation loop handles it gracefully.

---

## Validation Errors

`ValidationError` is raised during model construction when input parameters are outside valid ranges or violate physical constraints. Because it also inherits from Python's built-in `ValueError`, you can catch it with either `except bores.ValidationError` or `except ValueError`.

### Common Causes

**Negative or zero physical properties.** Porosity, permeability, pressure, temperature, and viscosity must all be positive. Saturation values must be between 0 and 1.

```python
# This will raise ValidationError
model = bores.reservoir_model(porosity=-0.1, ...)
```

**Saturations do not sum to 1.** The oil, water, and gas saturations in every cell must sum to 1.0. If you build saturation grids manually, small floating-point errors can cause this check to fail. Use `build_saturation_grids` to avoid this issue, as it enforces the constraint automatically.

**Grid dimension mismatches.** All property grids (porosity, permeability, saturation, etc.) must have the same shape as the grid dimensions specified in `reservoir_model()`. If you pass a (10, 10, 3) grid for porosity but a (10, 10, 5) grid for permeability, you will get a `ValidationError`.

**Well perforations outside the grid.** Well perforation intervals must reference valid cell coordinates within the grid dimensions. A perforation at cell (15, 5, 2) in a 10x10x3 grid will raise a `ValidationError`.

### How to Fix

Read the error message. It tells you which parameter failed and what the valid range is. Double-check your input data against the expected types and ranges described in the [API Reference](../api-reference/full-api.md).

---

## Solver Errors

`SolverError` is raised when the linear solver fails to converge within the maximum iteration count. This is one of the most common errors during simulation execution and usually indicates a problem with the linear system rather than a bug in the code.

### Common Causes

**Ill-conditioned matrix.** Extreme contrasts in permeability (e.g., 0.001 mD next to 10,000 mD), very thin cells, or large aspect ratios create matrices that are difficult for iterative solvers. The condition number of the matrix determines how many iterations the solver needs, and badly conditioned matrices may require more iterations than the maximum.

**Timestep too large.** For implicit schemes, very large timesteps make the matrix harder to solve. The adaptive timer usually prevents this, but if you override with a fixed large timestep, solver failures are likely.

**Weak preconditioner.** If you are using `"diagonal"` preconditioning or no preconditioning (`None`), the solver may need many more iterations than the maximum. Switch to `"ilu"` or `"amg"`.

### How to Fix

1. **Increase `max_iterations`** in `Config` (from 250 to 500 or 1000). This is the simplest fix and often works.

2. **Use a stronger preconditioner.** Switch from `"ilu"` to `"amg"`, or add a solver fallback chain (`pressure_solver=["bicgstab", "lgmres"]`).

3. **Switch to 64-bit precision.** Ill-conditioned matrices are more sensitive to floating-point precision. Call `bores.use_64bit_precision()` before building the model.

4. **Reduce the timestep.** Set a smaller `max_step_size` in the timer to keep matrices better conditioned.

5. **Check the model.** Extreme property contrasts may indicate a model setup error (e.g., a permeability of 0 in a cell that should be part of the flow domain).

See [Solver Selection](solver-selection.md) for detailed guidance.

---

## Simulation Errors

`SimulationError` is the base class for errors that occur during simulation execution. It covers situations where the simulation reaches a physically impossible state or cannot continue for some reason other than solver failure.

### Common Causes

**Negative pressures.** If the pressure in any cell drops below zero, the simulation raises a `SimulationError`. This usually means wells are producing at a rate that exceeds the reservoir's deliverability. Check your well controls and make sure production rates are achievable at the given reservoir pressure.

**Timestep at minimum.** If the adaptive timer reduces the timestep to `min_step_size` and the step still fails, a `SimulationError` is raised because the simulation cannot make progress. This typically indicates a fundamental model problem (unphysical properties, extreme boundary conditions, or a well operating far outside its capacity).

**Saturation constraint violations.** If saturations go negative or exceed 1 after the explicit saturation update, this indicates that the CFL condition was violated. The timer should catch this and reduce the timestep, but if `min_step_size` is too large to maintain stability, the simulation will fail.

### How to Fix

1. **Lower `min_step_size`** in the timer to give the adaptive controller more room to reduce the timestep.

2. **Check well controls.** Make sure production rates are reasonable for the reservoir pressure and permeability. Use `BHPControl` with a minimum BHP to prevent wells from drawing pressure below a physical limit.

3. **Check boundary conditions.** Constant-pressure boundaries at very high or very low pressures can drive extreme flow rates.

4. **Verify fluid properties.** Non-physical PVT values (e.g., formation volume factor less than 1.0 for live oil, or negative viscosity) will cause computation errors that propagate to simulation failure.

---

## Timing Errors

`TimingError` is a subclass of `SimulationError` raised when the timer configuration is invalid or the timer reaches an inconsistent state.

### Common Causes

**Invalid timer parameters.** Setting `min_step_size` greater than `max_step_size`, or `initial_step_size` outside the min/max range, will raise a `TimingError`.

**Simulation time is zero or negative.** The `simulation_time` parameter must be a positive number representing the total simulation duration in seconds.

### How to Fix

Check that your timer parameters are consistent: `min_step_size <= initial_step_size <= max_step_size` and `simulation_time > 0`. Use the `Time()` helper to convert from human-readable units:

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=1),
    max_step_size=bores.Time(days=30),
    min_step_size=bores.Time(seconds=1),
    simulation_time=bores.Time(years=5),
)
```

---

## Preconditioner Errors

`PreconditionerError` is raised when a preconditioner cannot be constructed or applied. This is less common than solver errors but can occur with certain matrix structures.

### Common Causes

**Zero diagonal elements.** ILU preconditioning requires non-zero diagonal elements. If a cell has zero transmissibility in all directions (completely isolated), the diagonal element is zero and ILU fails.

**AMG setup failure.** AMG can fail if the matrix structure is degenerate or if the pyamg package encounters a numerical issue.

### How to Fix

1. **Check for isolated cells.** Make sure every cell in the grid has non-zero transmissibility to at least one neighbor.

2. **Fall back to diagonal preconditioning** as a diagnostic: if the simulation works with `"diagonal"` but fails with `"ilu"`, the matrix has structural issues that need investigation.

3. **Use a different preconditioner.** If `"ilu"` fails, try `"block_jacobi"` or `"amg"`.

---

## Serialization Errors

`SerializationError` and `DeserializationError` occur during save and load operations.

### Common Causes

**Custom objects not registered.** If you create custom well controls, solvers, or preconditioners using the decorator registration system, you must register them before attempting to deserialize a file that contains them. The deserializer looks up types by name, and if the name is not in the registry, it raises a `DeserializationError`.

**File format mismatch.** Loading an HDF5 file with `JSONStore` or vice versa raises a `StorageError`.

**Corrupted or incomplete files.** If a simulation was interrupted during a save operation, the output file may be incomplete.

### How to Fix

1. **Register custom types before loading.** Import the module that contains your custom decorators (`@well_control`, `@solver_func`, etc.) before calling `load()`.

2. **Use the correct store type.** Match the store to the file extension: `.h5` for `HDF5Store`, `.zarr` for `ZarrStore`, `.json` for `JSONStore`.

3. **Check file integrity.** If a file was corrupted, re-run the simulation from the last good checkpoint.

---

## Using `StopSimulation`

`StopSimulation` is not an error. It is a control-flow exception you can raise to halt a simulation early. The simulation loop catches it and stops cleanly, preserving all results up to that point.

```python
import bores

config = bores.Config(timer=timer, wells=wells)

for step in bores.run(model=model, config=config):
    recovery = compute_recovery_factor(step)
    if recovery >= 0.30:
        raise bores.StopSimulation("Target recovery of 30% reached")
```

This is useful for economic limit studies, parameter sensitivity runs where you only need results up to a certain point, or interactive workflows where a user wants to stop and inspect intermediate results.

---

## General Debugging Strategy

When a simulation fails, follow this sequence:

1. **Read the error message.** It usually tells you what went wrong and where.

2. **Check the last successful timestep.** If you are using `StateStream`, the stored data up to the failure point can reveal trends (declining pressure, extreme saturations, oscillating rates) that explain the failure.

3. **Simplify the model.** Remove wells, boundary conditions, or heterogeneity to isolate the cause. If the simplified model works, add complexity back one piece at a time.

4. **Try 64-bit precision.** Many convergence and accuracy issues are caused by insufficient floating-point precision. This is a quick test that rules out (or confirms) precision as the cause.

5. **Reduce the timestep.** Set a smaller `max_step_size` and `initial_step_size`. If the simulation works with tiny timesteps, the problem is a CFL or convergence issue that the adaptive controller cannot handle at larger steps.

6. **Check units.** BORES uses oil-field units throughout: pressure in psi, temperature in degrees Fahrenheit, density in lbm/ft³, viscosity in cP, length in feet, time in seconds (internally). Mixing SI and oil-field values is a common source of non-physical results.
