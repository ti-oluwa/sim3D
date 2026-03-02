# Evolution Schemes

## Overview

An evolution scheme determines how the pressure and saturation equations are discretized and solved at each time step. The choice of scheme affects numerical stability, accuracy, computational cost, and the maximum time step size you can use. BORES supports three evolution schemes, each offering a different trade-off between these factors.

In black-oil reservoir simulation, you are solving two coupled systems of equations: a pressure equation (derived from mass conservation and Darcy's law) and saturation transport equations (one for each mobile phase). These equations are coupled because pressure depends on fluid saturations (through compressibility and density) and saturations depend on pressure (through flow velocities). The evolution scheme defines how this coupling is handled at each time step.

The scheme is set through the `scheme` parameter in the `Config`:

```python
import bores

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",  # "impes", "explicit", or "implicit"
)
```

---

## IMPES (Recommended Default)

IMPES stands for **IM**plicit **P**ressure, **E**xplicit **S**aturation. It is the most widely used scheme in black-oil simulation and the recommended default in BORES.

In IMPES, the pressure equation is solved implicitly (using a linear system solver) while the saturation equations are updated explicitly (using the pressure solution from the current step). This gives you the stability benefits of implicit pressure solving while keeping the saturation update simple and fast.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)
```

The implicit pressure step assembles a sparse linear system $A \cdot p = b$ where $A$ is the transmissibility matrix and $b$ contains source terms (wells, boundary conditions) and accumulation terms. This system is solved using an iterative solver (BiCGSTAB by default) with a preconditioner (ILU by default). Because pressure is solved implicitly, there is no CFL stability limit on the pressure equation, which allows larger time steps.

The explicit saturation step uses the pressure solution to compute phase velocities and then advances the saturations forward in time using a first-order upwind scheme. This step is fast (no linear system to solve) but is subject to a CFL stability condition that limits the maximum time step. If the time step is too large, the explicit saturation update can produce unphysical oscillations or negative saturations.

IMPES is the best balance for most problems. It handles pressure diffusion (which is fast and long-range) implicitly for stability, while treating saturation transport (which is local and advective) explicitly for efficiency.

!!! tip "When to Use IMPES"

    IMPES is appropriate for the vast majority of black-oil simulations: primary depletion, waterflooding, gas injection, and miscible flooding. It is the default in BORES and in most commercial simulators. Only switch to another scheme if you encounter specific numerical issues that IMPES cannot handle.

---

## Explicit

The fully explicit scheme treats both pressure and saturation explicitly. Both equations are advanced forward in time using the values from the previous time step, with no linear systems to solve.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="explicit",
)
```

The advantage of the explicit scheme is simplicity and low cost per time step. There are no sparse matrix assemblies, no linear system solves, and no preconditioners. Each step is essentially a series of element-wise array operations.

The disadvantage is that the scheme is conditionally stable. Both the pressure and saturation CFL conditions must be satisfied, which often requires very small time steps. The pressure CFL condition is particularly restrictive because pressure diffuses rapidly across the grid. In practice, the explicit scheme often requires time steps 10 to 100 times smaller than IMPES to remain stable.

The CFL thresholds can be tuned in the `Config`:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="explicit",
    pressure_cfl_threshold=0.9,     # Max pressure CFL number
    saturation_cfl_threshold=0.6,   # Max saturation CFL number
)
```

Lowering these thresholds increases stability at the cost of requiring even smaller time steps. Raising them improves performance but risks numerical instability.

!!! warning "Explicit Stability"

    The fully explicit scheme is useful for debugging, for very small models where the cost per step is negligible, or for educational purposes where you want to observe the CFL condition in action. For production simulations, IMPES is almost always a better choice because it allows much larger time steps while maintaining stability.

---

## Implicit

The fully implicit scheme treats both pressure and saturation implicitly. Both equations are assembled into linear systems and solved using iterative solvers with preconditioners.

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="implicit",
)
```

The fully implicit scheme is unconditionally stable, meaning there is no CFL limit on the time step size. You can take very large time steps without worrying about numerical oscillations or instabilities. This makes the implicit scheme attractive for problems where the explicit saturation CFL condition is very restrictive, such as fine-grid simulations, high-permeability contrasts, or simulations with strong capillary pressure effects.

The cost of this stability is higher computational effort per time step. The implicit scheme must solve a larger linear system that couples pressure and saturation, and it typically requires Newton iterations to handle the nonlinearity. Each Newton iteration involves assembling and solving a Jacobian system, which is significantly more expensive than the simple explicit saturation update in IMPES.

The implicit scheme is particularly useful when:

- Fine grids cause the IMPES saturation CFL to require impractically small time steps
- Strong capillary pressure creates fast local saturation changes
- High permeability contrasts (>1000:1) cause stability issues with IMPES
- You want to take very large time steps and are willing to pay more per step

You can control convergence behavior with the following `Config` parameters:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="implicit",
    pressure_convergence_tolerance=1e-6,    # Tighter for pressure
    saturation_convergence_tolerance=1e-4,  # Relaxed for saturation
    max_iterations=250,                      # Max solver iterations per step
)
```

!!! info "Implicit vs IMPES Cost"

    A rough guideline: the implicit scheme costs 3 to 10 times more per time step than IMPES, but can use time steps 5 to 50 times larger. Whether implicit is faster overall depends on the specific problem. For most field-scale models with moderate grid resolution, IMPES wins. For fine-grid studies or problems with severe CFL restrictions, implicit can be faster.

---

## Choosing a Scheme

| Feature | IMPES | Explicit | Implicit |
|---|---|---|---|
| Pressure solve | Implicit | Explicit | Implicit |
| Saturation solve | Explicit | Explicit | Implicit |
| Stability | Conditionally stable (saturation CFL) | Conditionally stable (both CFL) | Unconditionally stable |
| Cost per step | Moderate | Low | High |
| Max time step | Moderate | Small | Large |
| Best for | Most problems | Debugging, small models | Fine grids, strong capillary |
| Default | Yes | No | No |

For almost all reservoir simulation work, start with IMPES. If you encounter stability issues (oscillating saturations, frequent timestep rejections, negative saturations), try reducing the time step first. If that does not help, switch to the fully implicit scheme. The explicit scheme is primarily useful for educational purposes and very small, fast-running models.

---

## Convergence Controls

The `Config` provides several parameters that control how the solver behaves within each time step, regardless of scheme:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",

    # Solver convergence
    pressure_convergence_tolerance=1e-6,
    saturation_convergence_tolerance=1e-4,
    max_iterations=250,

    # Saturation change limits (trigger timestep rejection if exceeded)
    max_oil_saturation_change=0.5,
    max_water_saturation_change=0.4,
    max_gas_saturation_change=0.85,
    max_pressure_change=100.0,         # psi per step

    # CFL thresholds (explicit and IMPES saturation)
    saturation_cfl_threshold=0.6,
    pressure_cfl_threshold=0.9,

    # Output control
    output_frequency=1,                # Yield state every N steps
    log_interval=5,                    # Log progress every N steps
)
```

The `pressure_convergence_tolerance` controls when the iterative solver considers the pressure solution converged. A tighter tolerance (smaller number) gives more accurate pressure but requires more iterations. The default of `1e-6` is appropriate for most cases.

The `saturation_convergence_tolerance` plays the same role for the implicit saturation solver. It can be more relaxed than the pressure tolerance because the saturation transport equation is typically better conditioned.

The `max_iterations` parameter caps how many iterations the solver attempts before giving up. If the solver hits this limit, the time step is rejected and retried with a smaller step size. The default of 250 is generous; well-conditioned problems typically converge in 20 to 50 iterations.

The saturation and pressure change limits (`max_oil_saturation_change`, `max_pressure_change`, etc.) are safety valves. If any cell's saturation or pressure changes by more than these limits in a single step, the step is rejected and retried with a smaller time step. This prevents large, potentially unphysical jumps in the solution. You can tighten these limits for more conservative behavior or relax them for faster simulations.
