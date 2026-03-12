# Configuration

## Overview

The `Config` class holds every parameter that controls how a BORES simulation runs. It specifies the timer (time stepping behavior), the rock-fluid tables (relative permeability and capillary pressure models), the wells, the numerical scheme, the solvers, the preconditioners, the convergence tolerances, and all the physical constraints that keep the simulation stable and accurate.

`Config` is a frozen (immutable) attrs class. Once you create a `Config`, its fields cannot be changed in place. To modify a configuration, you create a new one using the `copy()` or `with_updates()` methods. This immutability prevents accidental modification of simulation parameters during a run and makes configurations safe to pass between functions without defensive copying.

Every simulation in BORES requires a `Config`. You pass it (along with a reservoir model) to `bores.run()` to start the simulation. The `Config` is the single point of control for all numerical behavior. If two simulations use different schemes, solvers, or convergence criteria, those differences are captured entirely in their respective `Config` objects.

---

## Creating a Config

At minimum, a `Config` requires a `Timer` and a `RockFluidTables`:

```python
import bores

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyThreePhaseRelPermModel(
        irreducible_water_saturation=0.25,
        residual_oil_saturation_water=0.30,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.05,
        water_exponent=2.5,
        oil_exponent=2.0,
        gas_exponent=2.0,
    ),
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(
        irreducible_water_saturation=0.25,
        residual_oil_saturation_water=0.30,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.05,
    ),
)

config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=1),
        max_step_size=bores.Time(days=10),
        min_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(years=3),
    ),
    rock_fluid_tables=rock_fluid_tables,
)
```

This creates a valid configuration with all defaults. The IMPES scheme, BiCGSTAB solver with ILU preconditioning, and standard convergence tolerances are used automatically.

### Adding Wells

Most simulations include wells. Pass them through the `wells` parameter:

```python
wells = bores.wells_(injectors=[injector], producers=[producer])

config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=1),
        max_step_size=bores.Time(days=10),
        min_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(years=3),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
)
```

### Adding Well Schedules

For simulations with time-varying well controls, use `well_schedules`:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    well_schedules=schedules,
)
```

When `well_schedules` is provided, the simulator automatically switches well controls at the scheduled times. See the [Well Scheduling](wells/schedules.md) page for details.

### Adding Boundary Conditions

Boundary conditions (aquifer support, constant pressure boundaries, etc.) are specified through the `boundary_conditions` parameter:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    boundary_conditions=boundary_conditions,
)
```

---

## All Config Parameters

### Required Parameters

| Parameter | Type | Description |
|---|---|---|
| `timer` | `Timer` | Time stepping manager (initial/max/min step sizes, simulation time) |
| `rock_fluid_tables` | `RockFluidTables` | Relative permeability and capillary pressure models |

### Optional Model Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `wells` | `Wells` | `None` | Well configuration (injectors and producers) |
| `well_schedules` | `WellSchedules` | `None` | Dynamic well control schedules |
| `boundary_conditions` | `BoundaryConditions` | `None` | Boundary conditions (aquifers, constant pressure, etc.) |
| `pvt_tables` | `PVTTables` | `None` | Tabulated PVT properties (alternative to correlations) |
| `constants` | `Constants` | Default | Physical and conversion constants |

### Numerical Scheme

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scheme` | `str` | `"impes"` | Evolution scheme: `"impes"`, `"explicit"`, or `"implicit"` |
| `use_pseudo_pressure` | `bool` | `True` | Use pseudo-pressure formulation for gas |

See [Schemes](simulation/schemes.md) for detailed information on each evolution scheme.

### Solver Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pressure_solver` | `str` or list | `"bicgstab"` | Solver(s) for the pressure equation |
| `saturation_solver` | `str` or list | `"bicgstab"` | Solver(s) for the saturation equation |
| `pressure_preconditioner` | `str` or `None` | `"ilu"` | Preconditioner for pressure solvers |
| `saturation_preconditioner` | `str` or `None` | `"ilu"` | Preconditioner for saturation solvers |
| `pressure_convergence_tolerance` | `float` | `1e-6` | Relative convergence tolerance for pressure |
| `saturation_convergence_tolerance` | `float` | `1e-4` | Relative convergence tolerance for saturation |
| `max_iterations` | `int` | `250` | Maximum solver iterations per step (capped at 500) |
| `task_pool` | `ThreadPoolExecutor` | `None` | Task pool for assembling solver matrices in parrallel |

See [Solvers](simulation/solvers.md) and [Preconditioners](simulation/preconditioners.md) for details.

### Time Step Controls

| Parameter | Type | Default | Description |
|---|---|---|---|
| `saturation_cfl_threshold` | `float` | `0.7` | Maximum saturation CFL number |
| `pressure_cfl_threshold` | `float` | `0.9` | Maximum pressure CFL number |
| `max_oil_saturation_change` | `float` | `0.6` | Maximum oil saturation change per step |
| `max_water_saturation_change` | `float` | `0.6` | Maximum water saturation change per step |
| `max_gas_saturation_change` | `float` | `0.5` | Maximum gas saturation change per step |
| `max_pressure_change` | `float` | `500.0` | Maximum pressure change per step (psi) |

!!! tip "Gas Saturation Change Limits"

    The default `max_gas_saturation_change` of 0.1 is intentionally lenient. Gas saturation can change rapidly during solution gas liberation or gas injection, and tightening this limit forces very small timesteps that slow the simulation significantly without meaningful accuracy gains. Only lower this value when you specifically need fine resolution of gas saturation evolution, such as detailed gas coning studies or near-critical fluid behavior. For most simulations, leave it at the default or increase it further.

See [Time Step Control](simulation/timestep-control.md) for guidance on adjusting these.

### Physical Controls

| Parameter | Type | Default | Description |
|---|---|---|---|
| `capillary_strength_factor` | `float` | `1.0` | Scale factor for capillary effects (0 to 1) |
| `disable_capillary_effects` | `bool` | `False` | Completely disable capillary pressure |
| `disable_structural_dip` | `bool` | `False` | Disable gravity/structural dip effects |
| `miscibility_model` | `str` | `"immiscible"` | Miscibility model: `"immiscible"` or `"todd_longstaff"` |
| `freeze_saturation_pressure` | `bool` | `False` | Keep bubble point pressure constant |

### Fluid Mobility

| Parameter | Type | Default | Description |
|---|---|---|---|
| `relative_mobility_range` | `RelativeMobilityRange` | See below | Min/max relative mobility per phase |
| `total_compressibility_range` | `Range` | `(1e-24, 1e-2)` | Min/max total compressibility |
| `phase_appearance_tolerance` | `float` | `1e-6` | Saturation below which a phase is absent |

The default relative mobility ranges are:

- Oil: $10^{-12}$ to $10^{6}$
- Water: $10^{-12}$ to $10^{6}$
- Gas: $10^{-12}$ to $10^{6}$

These ranges prevent division by zero and numerical overflow in mobility calculations. You rarely need to change them.

### Hysteresis

| Parameter | Type | Default | Description |
|---|---|---|---|
| `residual_oil_drainage_ratio_water_flood` | `float` | `0.6` | Oil drainage residual ratio (waterflood) |
| `residual_oil_drainage_ratio_gas_flood` | `float` | `0.6` | Oil drainage residual ratio (gas flood) |
| `residual_gas_drainage_ratio` | `float` | `0.5` | Gas drainage residual ratio |

### Output and Logging

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_frequency` | `int` | `1` | Yield a state every N steps |
| `log_interval` | `int` | `5` | Log progress every N steps |
| `warn_well_anomalies` | `bool` | `True` | Warn about anomalous well flow rates |

---

## Modifying a Config

Since `Config` is immutable, you cannot modify fields directly. Use `copy()` or `with_updates()` to create modified versions:

### `copy()`

```python
# Create a new config with a different scheme
implicit_config = config.copy(scheme="implicit")

# Multiple changes at once
tuned_config = config.copy(
    scheme="implicit",
    pressure_solver="gmres",
    pressure_preconditioner="amg",
    max_iterations=400,
)
```

### `with_updates()`

`with_updates()` works the same way as `copy()` but validates that all provided keys are valid `Config` attributes:

```python
# This works
updated = config.with_updates(scheme="implicit")

# This raises AttributeError because "schemee" is not a valid field
updated = config.with_updates(schemee="implicit")  # AttributeError
```

Use `with_updates()` when you want protection against typos in parameter names. Use `copy()` when you prefer the shorter name and are confident in the parameter names.

---

## Freeze Saturation Pressure

The `freeze_saturation_pressure` flag controls whether the oil bubble point pressure (Pb) is recomputed at each time step or held constant at its initial value.

```python
# Keep Pb constant (standard black-oil assumption)
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    freeze_saturation_pressure=True,
)
```

When `freeze_saturation_pressure=True`, the following properties are computed using the initial bubble point pressure rather than a dynamically updated value:

- Bubble point pressure (Pb) itself
- Solution gas-oil ratio (Rs)
- Oil formation volume factor (Bo)
- Oil compressibility (Co)
- Oil viscosity (indirectly through Rs)
- Oil density (indirectly through Rs and Bo)

This is appropriate for natural depletion and waterflooding where oil composition remains constant. Set it to `False` (the default) for miscible injection or any process where dissolved gas content changes significantly during the simulation.

---

## Capillary Strength Factor

The `capillary_strength_factor` scales capillary pressure effects without changing the capillary pressure model itself. It ranges from 0.0 (no capillary effects) to 1.0 (full capillary effects).

```python
# Reduce capillary effects by 50% for numerical stability
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    capillary_strength_factor=0.5,
)
```

Capillary gradients can become numerically dominant in fine meshes or at sharp saturation fronts, causing oscillations or overshoot. Reducing the capillary strength factor damps these effects without removing them entirely. This is a common technique for improving convergence in difficult models while preserving the qualitative influence of capillary pressure on fluid distribution.

Setting `disable_capillary_effects=True` is equivalent to `capillary_strength_factor=0.0` but is more explicit in intent.

---

## Example Configurations

### Simple Depletion Study

```python
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=2),
        max_step_size=bores.Time(days=30),
        min_step_size=bores.Time(days=1),
        simulation_time=bores.Time(years=10),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    freeze_saturation_pressure=True,
)
```

### High-Resolution Waterflood

```python
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=0.5),
        max_step_size=bores.Time(days=5),
        min_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(years=5),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pressure_solver="gmres",
    pressure_preconditioner="amg",
    max_water_saturation_change=0.15,
    max_pressure_change=50.0,
)
```

### Miscible Gas Injection

```python
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(hours=6),
        max_step_size=bores.Time(days=3),
        min_step_size=bores.Time(minutes=30),
        simulation_time=bores.Time(years=3),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    miscibility_model="todd_longstaff",
    freeze_saturation_pressure=False,
    max_pressure_change=75.0,
)
```
