# BORES

**3D 3-Phase Black-Oil Reservoir Modelling and Simulation Framework**

> ⚠️ **Important Disclaimer**: BORES is designed for **educational, research, and prototyping purposes**. It is **not production-grade software** and should not be used for making critical business decisions, regulatory compliance, or field development planning. Results should be validated against established commercial simulators before any real-world application. Use at your own discretion.

> 📚 **Documentation Notice**: Full API documentation is coming soon. In the meantime, this README provides an intuitive introduction through practical examples. For detailed API information, refer to the source code and docstrings.

BORES is a reservoir engineering framework designed for 3D black-oil modelling and simulation of three-phase (oil, water, gas) flow in porous media built with Python. It provides a clean and modular API for building reservoir models, defining wells, defining fractures and faults, running simulations, and analyzing results. BORES APIs are also easily extensible for custom models and workflows, if you know what you are doing.

BORES started as a final year project for my Bachelor's degree in Petroleum Engineering at the Federal University of Petroleum Resources, Effurun, Nigeria. Why write this when there are other commercial and open-source reservoir simulators like Eclipse, CMG, MRST, OpenPorousMedia, etc.? Well, Existing libraries are either closed-source, written in low-level languages (C/C++, Fortran), or have complex APIs with poor documentation that make prototyping and experimentation difficult. BORES aims to fill this gap by providing a simple, Pythonic interface for reservoir simulation that is easy to understand and extend. Simply put, this make thing more accessible to petroleum engineers and researchers who may not be expert programmers as Python is such a simple language to learn and use, and is widely adopted in the scientific computing community.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Static Grid/Model Building](#static-gridmodel-building)
- [PVT Correlations vs Tables](#pvt-correlations-vs-tables)
- [Simulation Configs & Timers](#simulation-configs--timers)
- [Precision Control](#precision-control)
- [Model States, Streams & Stores](#model-states-streams--stores)
- [Wells](#wells)
- [Fractures](#fractures)
- [Capillary Pressure & Relative Permeability](#capillary-pressure--relative-permeability)
- [Constants](#constants)
- [Boundary Conditions](#boundary-conditions)
- [Errors](#errors)
- [Solvers](#solvers)
- [Visualization](#visualization)
- [Performance & Optimization](#performance--optimization)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Contributing & Support](#contributing--support)

---

## Installation

```bash
uv add bores-framework

pip install bores-framework

# or for development
pip install -e .
```

**Dependencies**:

- NumPy
- SciPy
- CoolProp
- Numba
- attrs
- h5py
- zarr
- plotly

---

## Quick Start

See the [Complete Example](#example-complete-simulation-workflow-on-a-heterogeneous-reservoir-model) at the end of this README for a full working example. The `reservoir_model` factory requires many parameters including thickness grids, saturation endpoint grids, rock compressibility, and relative permeability/capillary pressure models. The complete example shows all the steps involved in building a sample heterogeneous reservoir model from scratch.

---

## Static Grid/Model Building

BORES provides several utilities for building 3D grids with varying properties.

### Basic Grid Construction

```python
import bores
import numpy as np

# Define grid dimensions
grid_shape = (30, 20, 6)  # nx, ny, nz
cell_dimension = (100.0, 100.0)  # dx, dy in feet (dz comes from thickness_grid)
```

### Layered Grids

Use `layered_grid` to define properties that vary by layer. **Note**: Requires `orientation` parameter:

```python
# Porosity varying by layer (z-direction)
porosity_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=[0.22, 0.20, 0.18, 0.15, 0.12, 0.10],  # One per layer
    orientation=bores.Orientation.Z,  # Layer along z-axis
)  # Shape: (30, 20, 6)

# Permeability varying by layer
permeability_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=[150.0, 120.0, 100.0, 80.0, 50.0, 30.0],  # mD
    orientation=bores.Orientation.Z,
)

# Layer thickness (required for depth calculations)
thickness_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=[30.0, 25.0, 20.0, 25.0, 30.0, 20.0],  # ft per layer
    orientation=bores.Orientation.Z,
)
```

### Uniform Grids

For constant properties across the entire grid:

```python
temperature_grid = bores.uniform_grid(
    grid_shape=grid_shape, value=180.0  # °F
)
```

### Depth Grids

Generate grids representing cell-center depths from a thickness grid:

```python
# Calculate depth at cell centers from thickness
depth_grid = bores.depth_grid(thickness_grid)  # Takes `thickness_grid` only
```

### Structural Dip

Apply structural dip to your depth grid using azimuth convention:

```python
# Apply 3° dip toward East (azimuth 90°)
dipped_depth_grid = bores.apply_structural_dip(
    elevation_grid=depth_grid,
    cell_dimension=cell_dimension,  # (dx, dy) tuple
    elevation_direction="downward",  # or "upward" for elevation convention
    dip_angle=3.0,  # degrees (0-90)
    dip_azimuth=90.0,  # degrees (0-360, clockwise from North)
)
```

**Azimuth Convention:**

- 0° = North (+y direction)
- 90° = East (+x direction)  
- 180° = South (-y direction)
- 270° = West (-x direction)

### Saturation Distribution with Fluid Contacts

Build realistic saturation distributions with oil-water and gas-oil contacts. **Note**: Requires residual saturation grids, not scalars:

```python
# First create residual saturation grids
connate_water_saturation_grid = bores.uniform_grid(grid_shape, value=0.15)
residual_oil_saturation_water_grid = bores.uniform_grid(grid_shape, value=0.25)
residual_oil_saturation_gas_grid = bores.uniform_grid(grid_shape, value=0.15)
residual_gas_saturation_grid = bores.uniform_grid(grid_shape, value=0.05)

# Build saturation grids with transition zones
water_saturation, oil_saturation, gas_saturation = bores.build_saturation_grids(
    depth_grid=dipped_depth_grid,
    gas_oil_contact=60.0,  # Depth below top of reservoir (ft)
    oil_water_contact=150.0,  # Depth below top of reservoir (ft)
    connate_water_saturation_grid=connate_water_saturation_grid,
    residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
    residual_gas_saturation_grid=residual_gas_saturation_grid,
    porosity_grid=porosity_grid,
    use_transition_zones=True,  # Enable smooth transitions
    gas_oil_transition_thickness=8.0,  # ft
    oil_water_transition_thickness=12.0,  # ft
    transition_curvature_exponent=1.2,
)
```

### Building the Reservoir Model

The `reservoir_model` factory requires many parameters. Here's the structure (see [Complete Example](#example-complete-simulation-workflow-on-a-heterogeneous-reservoir-model) for a full working example):

```python
# First, create permeability structure for anisotropic permeability
absolute_permeability = bores.RockPermeability(
    x=x_permeability_grid,  # mD
    y=y_permeability_grid,  # typically 0.8x of x-direction
    z=z_permeability_grid,  # typically 0.1x of x-direction (vertical)
)

# Create relative permeability and capillary pressure models
relative_permeability_table = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    wettability=bores.WettabilityType.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,  # psi
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,  # psi
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.WettabilityType.WATER_WET,
)

# Build the reservoir model
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,  # (dx, dy) tuple
    thickness_grid=thickness_grid,
    pressure_grid=pressure_grid,
    rock_compressibility=4.5e-6,  # 1/psi
    absolute_permeability=absolute_permeability,
    porosity_grid=porosity_grid,
    temperature_grid=temperature_grid,
    oil_saturation_grid=oil_saturation,
    water_saturation_grid=water_saturation,
    gas_saturation_grid=gas_saturation,
    oil_viscosity_grid=oil_viscosity_grid,  # cP
    oil_compressibility_grid=oil_compressibility_grid,  # 1/psi
    oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,  # psia
    residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
    residual_gas_saturation_grid=residual_gas_saturation_grid,
    irreducible_water_saturation_grid=irreducible_water_saturation_grid,
    connate_water_saturation_grid=connate_water_saturation_grid,
    relative_permeability_table=relative_permeability_table,
    capillary_pressure_table=capillary_pressure_table,
    # Optional parameters:
    oil_specific_gravity_grid=oil_specific_gravity_grid,
    gas_gravity_grid=gas_gravity_grid,
    net_to_gross_ratio_grid=net_to_gross_grid,
    boundary_conditions=boundary_conditions,
    dip_angle=dip_angle,
    dip_azimuth=dip_azimuth,
    reservoir_gas="methane", # Assumed that reservoir gas is methane. Can be any gas supported by CoolProp
    pvt_tables=pvt_tables,
)
```

---

## PVT Correlations vs Tables

BORES supports two approaches for PVT (Pressure-Volume-Temperature) property calculations:

1. **Correlations**: Direct calculation using empirical correlations (e.g., Standing, Beggs & Robinson)
2. **Tables**: Pre-computed lookup tables for faster interpolation during simulation

### Why Use PVT Tables?

PVT tables offer several advantages:

- **Performance**: Interpolation is faster than evaluating complex correlations at each cell/timestep
- **Flexibility**: Can incorporate lab PVT data directly
- **Consistency**: Ensures thermodynamic consistency through pre-computation
- **Pseudo-pressure support**: Used in pre-computed gas pseudo-pressure tables for efficient gas well calculations

### Building PVT Tables

```python
import bores

# Define the pressure, temperature, and salinity ranges
pvt_table_data = bores.build_pvt_table_data(
    pressures=bores.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]),  # psia
    temperatures=bores.array([120, 140, 160, 180, 200, 220]),  # °F
    salinities=bores.array([30000, 32000, 33500, 35000]),  # ppm
    oil_specific_gravity=0.845,  # API ~36
    gas_gravity=0.65,  # relative to air
    reservoir_gas="methane",  # or "co2", "n2", etc. (Gas names supported by CoolProp)
)
```

This builds tables for:

- **Oil**: Bubble point pressure, formation volume factor (Bo), viscosity (μo), compressibility (co), solution gas-oil ratio (Rs)
- **Water**: Formation volume factor (Bw), viscosity (μw), compressibility (cw)
- **Gas**: Z-factor, formation volume factor (Bg), viscosity (μg), compressibility (cg), density

### Creating the `PVTTables` Object

```python
pvt_tables = bores.PVTTables(
    table_data=pvt_table_data,
    interpolation_method="linear",  # or "cubic" for smoother but slower interpolation
)
```

### Using PVT Tables in Config

```python
config = bores.Config(
    scheme="impes",
    pvt_tables=pvt_tables,
    use_pseudo_pressure=True,  # Enable gas pseudo-pressure for gas wells (Enabled by default)
)
```

### Querying PVT Properties

The `PVTTables` object provides methods for querying properties:

```python
# Single point query
bo = pvt_tables.oil_formation_volume_factor(
    pressure=2500.0,  # psia
    temperature=180.0,  # °F
    solution_gor=500.0,  # scf/STB
)

# Grid-based query (for simulation)
viscosity_grid = pvt_tables.oil_viscosity(
    pressure=pressure_grid,
    temperature=temperature_grid,
    solution_gor=rs_grid,
)
```

---

## Simulation Configs & Timers

### `Timer` Configuration

The `Timer` manages adaptive time-stepping with CFL-based control:

```python
timer = bores.Timer(
    # Step size bounds
    initial_step_size=bores.Time(hours=4.5),
    max_step_size=bores.Time(days=5),
    min_step_size=bores.Time(minutes=10),
    
    # Simulation duration
    simulation_time=bores.Time(days=bores.c.DAYS_PER_YEAR * 5),  # 5 years
    
    # CFL control
    max_cfl_number=0.9,  # Maximum CFL for stability
    
    # Adaptive stepping
    ramp_up_factor=1.2,      # Grow step by 20% on success
    backoff_factor=0.5,       # Halve step on failure
    aggressive_backoff_factor=0.25,  # Quarter step on repeated failure
    max_rejects=20,           # Max rejections before error
)
```

### The `Time` Helper

`bores.Time()` converts human-readable time to seconds:

```python
bores.Time(hours=4.5)                    # 16200.0 seconds
bores.Time(days=30)                      # 2592000.0 seconds
bores.Time(days=365, hours=12)           # 31579200.0 seconds
bores.Time(weeks=2, days=3, hours=6)     # Mix units freely
```

### Simulation Config

The `Config` object controls simulation behavior:

```python
config = bores.Config(
    # Evolution scheme
    scheme="impes",  # IMPES (Implicit Pressure, Explicit Saturation)
    # or "explicit" for fully explicit scheme
    
    # Output control
    output_frequency=1,  # Yield state every N steps
    log_interval=5,      # Log progress every N steps
    
    # Physics options
    miscibility_model="immiscible",  # or "todd_longstaff" for miscible flooding
    use_pseudo_pressure=True,        # Use pseudo-pressure for gas
    disable_capillary_effects=False, # Include capillary pressure
    
    # Numerical options
    capillary_strength_factor=1.0,   # Scale capillary effects (0-1)
    convergence_tolerance=1e-6,      # Solver tolerance
    max_iterations=250,              # Max solver iterations
    
    # Solver selection
    iterative_solver="bicgstab",     # BiCGSTAB, LGMRES, etc.
    preconditioner="ilu",            # ILU preconditioning, can be AMG, Diagonal, CPR, or None
    
    # PVT tables (if using)
    pvt_tables=pvt_tables,
)
```

### Supported Schemes

| Scheme       | Description                               | Stability         | Speed            |
| ------------ | ----------------------------------------- | ----------------- | ---------------- |
| `"impes"`    | Implicit Pressure, Explicit Saturation    | Moderate          | Fast             |
| `"explicit"` | Fully explicit in pressure and saturation | Requires small Δt | Fastest per step |

> **Note**: Fully implicit scheme (`"implicit"`) is planned but not yet implemented. The option may be added in future releases. I'll gladly accept contributions toward this feature too.

---

## Precision Control

BORES supports both 32-bit and 64-bit floating-point precision (mainly for memory and speed optimization). By default, BORES uses 32-bit (`np.float32`):

```python
import bores

# Enable 32-bit precision (default is already 32-bit)
bores.use_32bit_precision()

# Arrays created via `bores.array()` will use the configured precision
pressure = bores.array([1000, 2000, 3000])  # float32 array
```

### Why Use 32-bit Precision?

- **Memory**: Half the memory footprint for large grids
- **Speed**: Faster computation, especially with NumPy/Numba
- **GPU-ready**: Better compatibility with GPU acceleration

### When to Use 64-bit (Default)

- **Accuracy**: For simulations requiring high numerical precision
- **Stability**: When dealing with very large pressure differentials or long simulation times

```python
# Check current precision
dtype = bores.get_dtype()  # Returns np.float32 or np.float64
```

---

## Model States, Streams & Stores

### Model States

Each time step produces a `ModelState` containing:

```python
state = states[100]  # Get state at step 100

state.step          # Time step index
state.time          # Simulation time (seconds)
state.step_size     # Time step size (seconds)
state.model         # ReservoirModel with updated properties
state.wells         # Wells configuration at this state
state.injection     # Injection rate grids
state.production    # Production rate grids
state.relative_permeabilities  # krw, kro, krg grids
state.relative_mobilities      # Phase mobility grids
state.capillary_pressures      # Pcow, Pcgo grids
state.timer_state            # Dumped `Timer` state at this step
```

### Accessing Model Properties

```python
model = state.model
fluid_props = model.fluid_properties

# Pressure and saturations
pressure = fluid_props.pressure_grid
oil_saturation = fluid_props.oil_saturation_grid
water_saturation = fluid_props.water_saturation_grid
gas_saturation = fluid_props.gas_saturation_grid

# Derived properties
viscosity = fluid_props.oil_effective_viscosity_grid
density = fluid_props.oil_effective_density_grid
```

### State Streams

For long simulations, use `StateStream` to process states incrementally and persist to storage. It provides substantial performance improvements by reducing memory usage and I/O overhead. It supports checkpointing and batch persistence.

A checkpoint is a saved state of the simulation that allows resuming from that point in case of interruptions from errors or system failures.

```python
from pathlib import Path

# Create a store
store = bores.ZarrStore(
    store=Path("./results/simulation.zarr"),
    metadata_dir=Path("./results/metadata/"),
)

# Run with streaming
stream = bores.StateStream(
    bores.run(model=model, timer=timer, wells=wells, config=config),
    store=store,
    async_io=True, # Prevent state persistence I/O operations from block stream
    batch_size=50,  # Persist every 50 states
    checkpoint_interval=200,  # Checkpoint every 200 steps
    checkpoint_dir=Path("./results/checkpoints/"),
)

# Process states as they come
with stream:
    for state in stream:
        # Process each state (optional)
        print(f"Step {state.step}: P_avg = {state.model.fluid_properties.pressure_grid.mean():.1f}")
    
    # Or consume all at once
    # stream.consume()

    # Or collect states from specific steps
    # selected_states = stream.collect(1, 5, 10, 20)

    # Or collect states with a predicate. Can be computationally expensive.
    # high_pressure_states = stream.collect(key=lambda s: s.model.fluid_properties.pressure_grid.mean() > 3000)
```

### State Stores

BORES provides multiple storage backends:

```python
# Zarr (recommended for large simulations)
zarr_store = bores.ZarrStore(
    store=Path("./results/simulation.zarr"),
    metadata_dir=Path("./results/metadata/"),
    compression_level=3,  # Zlib compression level (0-9)
)

# HDF5
hdf5_store = bores.HDF5Store(
    filepath=Path("./results/simulation.h5"),
    metadata_dir=Path("./results/metadata/"),
)

# Pickle (for small simulations or debugging)
pickle_store = bores.PickleStore(
    filepath=Path("./results/pickle/")
)

# NPZ (NumPy compressed)
npz_store = bores.NPZStore(
    filepath=Path("./results/npz/"),
    metadata_dir=Path("./results/metadata/"),
)
```

### Loading States

```python
# Load all states
states = store.load(validate=False, lazy=False)

# Lazy loading (loads metadata only, grids on access)
states = store.load(validate=False, lazy=True)

# Get the last state for continuation
# Note that `load` returns a generator and we do not use list(states)
# to avoid loading all states into memory.
last_state = None
while True:
    try:
        last_state = next(states)
    except StopIteration:
        break
model = last_state.model
```

---

## Wells

### Production Wells

```python
# Define production clamp (limits). Prevents positive rates as production rates are taken as negative.
# Hence production clamp prevents production wells from injecting fluids.
clamp = bores.ProductionClamp()

# Multi-phase rate control
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-100,      # STB/day (negative = production)
        target_phase="oil",
        bhp_limit=1200,        # Minimum BHP (psia)
        clamp=clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-500,      # Mscf/day
        target_phase="gas",
        bhp_limit=1200,
        clamp=clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-10,       # STB/day
        target_phase="water",
        bhp_limit=1200,
        clamp=clamp,
    ),
)

# Create producer
producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((14, 10, 3), (14, 10, 4))],  # Grid cells (i, j, k)
    radius=0.3542,  # ft (8.5" wellbore)
    control=control,
    # We produce oil, gas, and water
    produced_fluids=(
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.845,
            molecular_weight=180.0,
        ),
        bores.ProducedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.05,
            molecular_weight=18.015,
        ),
    ),
    skin_factor=2.5,
    is_active=True,
)
```

### Injection Wells

```python
# Injection clamp. Prevents negative rates as injection rates are positive.
# Hence injection clamp prevents injection wells from producing fluids.
injection_clamp = bores.InjectionClamp()

# Rate control for injector
control = bores.AdaptiveBHPRateControl(
    target_rate=1_000_000,  # SCF/day (positive = injection)
    target_phase="gas",
    bhp_limit=3500,         # Maximum BHP (psia)
    clamp=injection_clamp,
)

# Create CO2 injector
gas_injector = bores.injection_well(
    well_name="GI-1",
    perforating_intervals=[((16, 3, 1), (16, 3, 3))],
    radius=0.3542,
    control=control,
    injected_fluid=bores.InjectedFluid(
        name="CO2",
        phase=bores.FluidPhase.GAS,
        specific_gravity=0.65,
        molecular_weight=44.0,
        minimum_miscibility_pressure=3250.0,  # For miscible flooding
        todd_longstaff_omega=0.67,
        is_miscible=True,
    ),
    skin_factor=2.0,
    is_active=True,
)

# Duplicate wells with different locations
gas_injector_2 = gas_injector.duplicate(
    name="GI-2",
    perforating_intervals=[((16, 16, 1), (16, 16, 3))],
)
```

### Well Events & Scheduling

Schedule changes to wells during simulation:

```python
# Start producer inactive, activate after 100 days
producer = bores.production_well(
    well_name="P-1",
    ...,
    is_active=False,  # Initially shut-in
)

# Schedule activation
producer.schedule_event(
    bores.WellEvent(
        hook=bores.well_time_hook(time=bores.Time(days=100)),
        action=bores.well_update_action(is_active=True),
    )
)
```

### Combining Wells

```python
# Create wells collection
wells = bores.wells_(
    injectors=[gas_injector, gas_injector_2],
    producers=[producer],
)

# Run simulation with wells
states = bores.run(model=model, timer=timer, wells=wells, config=config)
```

---

## Fractures

BORES supports various fracture types for modeling faults and fracture networks:

### Vertical Sealing Fault

```python
# Simple vertical fault through entire grid
fault = bores.vertical_sealing_fault(
    fault_id="F-1",
    orientation="x",  # Perpendicular to x-axis
    index=10,         # Cell index where fault is located
    permeability_multiplier=1e-4,  # 99.99% sealing
)

# Fault with limited extent
fault_limited = bores.vertical_sealing_fault(
    fault_id="F-2",
    orientation="y",
    index=15,
    y_range=(5, 25),   # Lateral extent
    z_range=(0, 8),    # Vertical extent (shallow fault)
)

model = bores.apply_fracture(model, fault)
```

### Normal Fault with Throw

```python
# Normal fault (hanging wall moves down)
fault = bores.normal_fault_with_throw(
    fault_id="NF-1",
    orientation="x",
    index=20,
    throw_cells=2,      # Number of cells of displacement
    permeability_multiplier=1e-4,  # Sealing factor
    preserve_data=True,  # Preserve displaced grid data
)
```

### Reverse Fault with Throw

```python
# Reverse fault (hanging wall moves up)
fault = bores.reverse_fault_with_throw(
    fault_id="RF-1",
    orientation="y",
    index=25,
    throw_cells=3,
    permeability_multiplier=1e-4,
)
```

### Inclined Sealing Fault

```python
fault = bores.inclined_sealing_fault(
    fault_id="IF-1",
    orientation="x",
    index=15,
    slope=1.0,  # Rise over run
    permeability_multiplier=1e-4,
)
```

### Damage Zone Fault

```python
# Fault with damage zone (enhanced permeability around fault)
fault = bores.damage_zone_fault(
    fault_id="DZ-1",
    orientation="x",
    cell_range=(15, 18),
    permeability_multiplier=1e-4,   # Fault core (sealing)
)
```

### Conductive Fracture Network

```python
# High-permeability fracture corridor
fracture_network = bores.conductive_fracture_network(
    network_id="CFN-1",
    orientation="x",
    cell_range=(10, 12),  # Multiple fracture planes
    permeability_multiplier=15.0,  # High conductivity
)
```

### Applying Fractures

```python
# Apply single fracture
model = bores.apply_fracture(model, fault)

# Apply multiple fractures (using *args)
model = bores.apply_fractures(model, fault1, fault2, fracture_network)
```

---

## Capillary Pressure & Relative Permeability

BORES provides both **analytical models** and **tables** for relative permeability and capillary pressure. For three-phase flow, two-phase data must be combined into three-phase tables using appropriate mixing rules.

### Key Concepts

**Two-Phase vs Three-Phase**: Laboratory data typically measures two-phase systems (oil-water or gas-oil). For three-phase simulation, these must be combined using mixing rules that account for phase interactions.

**Wetting Phase**: The phase that preferentially adheres to rock surfaces:

- **Water-wet**: Water preferentially wets rock (most sandstones)
- **Oil-wet**: Oil preferentially wets rock (some carbonates)
- **Mixed-wet**: Both water-wet and oil-wet regions exist

**Saturation Endpoints**:

- `Swc` (irreducible water saturation): Minimum water saturation
- `Sorw` (residual oil to water): Oil remaining after water flood
- `Sorg` (residual oil to gas): Oil remaining after gas flood  
- `Sgr` (residual gas): Minimum gas saturation

---

### Relative Permeability Models

#### Brooks-Corey Three-Phase Model

The most common model using Corey-type power-law functions:

```python
relperm_model = bores.BrooksCoreyThreePhaseRelPermModel(
    # Saturation endpoints
    irreducible_water_saturation=0.15,   # Swc
    residual_oil_saturation_water=0.25,  # Sor to water flood
    residual_oil_saturation_gas=0.15,    # Sor to gas flood
    residual_gas_saturation=0.045,       # Sgr
    
    # Corey exponents (typically 1.5-4.0)
    water_exponent=2.0,  # Higher = more convex curve
    oil_exponent=2.0,
    gas_exponent=2.0,
    
    # Wettability
    wettability=bores.WettabilityType.WATER_WET,  # WATER_WET, OIL_WET
    
    # Three-phase oil mixing rule
    mixing_rule=bores.stone_II_rule,
)

# Use in reservoir model
model = bores.reservoir_model(
    # ... other params ...
    relative_permeability_table=relperm_model,
)
```

---

### Three-Phase Oil Relative Permeability Mixing Rules

The challenge in three-phase flow is computing oil relative permeability when both water and gas are present. BORES provides multiple mixing rules:

```python
# Conservative rules (lower kro estimates)
bores.min_rule              # kro = min(kro_w, kro_g) - most conservative
bores.harmonic_mean_rule    # 2/(1/kro_w + 1/kro_g) - series flow
bores.geometric_mean_rule   # sqrt(kro_w × kro_g)
bores.hustad_hansen_rule    # (kro_w × kro_g) / max(kro_w, kro_g)
bores.blunt_rule            # For strongly water-wet systems

# Industry standard rules
bores.stone_I_rule          # Stone I (1970) - water-wet systems
bores.stone_II_rule         # Stone II (1973) - most widely used (default)
bores.eclipse_rule          # ECLIPSE simulator default

# Other rules
bores.arithmetic_mean_rule  # (kro_w + kro_g) / 2 - optimistic
bores.baker_linear_rule     # Baker's linear interpolation (1988)
bores.saturation_weighted_interpolation_rule  # Weighted by Sw, Sg
bores.linear_interpolation_rule  # Simple linear interpolation
bores.max_rule              # max(kro_w, kro_g) - most optimistic

# Parameterized rule
bores.aziz_settari_rule(a=0.5, b=0.5)  # kro = kro_w^a × kro_g^b
```

**Comparison Table**:

| Rule | Conservativeness | Best For |
|------|------------------|----------|
| `min_rule` | Very conservative | Lower bound, safety analysis |
| `harmonic_mean_rule` | Very conservative | Tight rocks, series flow |
| `geometric_mean_rule` | Conservative | General purpose |
| `stone_I_rule` | Moderate | Water-wet sandstones |
| `stone_II_rule` | Moderate | Industry standard |
| `eclipse_rule` | Moderate | Matching commercial simulators |
| `arithmetic_mean_rule` | Optimistic | Upper bound estimate |
| `max_rule` | Very optimistic | Sensitivity analysis |

---

### Relative Permeability Tables

For lab-measured or history-matched data, use tabular input. **Two-phase tables must be combined into a three-phase table.**

#### Two-Phase Tables

```python
# Oil-Water system (water = wetting phase in water-wet rock)
oil_water_relperm = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=bores.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]),
    wetting_phase_relative_permeability=bores.array([0.0, 0.02, 0.06, 0.12, 0.22, 0.35, 0.45]),  # krw
    non_wetting_phase_relative_permeability=bores.array([1.0, 0.7, 0.45, 0.25, 0.10, 0.02, 0.0]),  # kro
)

# Gas-Oil system (oil = wetting phase)
gas_oil_relperm = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=bores.array([0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95]),  # Oil saturation
    wetting_phase_relative_permeability=bores.array([0.0, 0.05, 0.15, 0.35, 0.55, 0.80, 1.0]),  # kro
    non_wetting_phase_relative_permeability=bores.array([0.9, 0.65, 0.40, 0.20, 0.08, 0.01, 0.0]),  # krg
)
```

#### Constructing Three-Phase Table from Two-Phase Data

**Two-phase tables cannot be used directly** — they must be combined with a mixing rule:

```python
# Construct three-phase table from two-phase data
three_phase_relperm = bores.ThreePhaseRelPermTable(
    oil_water_table=oil_water_relperm,  # Water-oil relative permeabilities
    gas_oil_table=gas_oil_relperm,      # Gas-oil relative permeabilities
    mixing_rule=bores.stone_II_rule,    # How to compute kro in 3-phase
)

# Use in reservoir model
model = bores.reservoir_model(
    # ... other params ...
    relative_permeability_table=three_phase_relperm,
)
```

#### Querying Relative Permeabilities

```python
# Query at specific saturations
result = three_phase_relperm.get_relative_permeabilities(
    water_saturation=0.3,
    oil_saturation=0.5,
    gas_saturation=0.2,
)

print(result["water"])  # krw
print(result["oil"])    # kro (computed via mixing rule)
print(result["gas"])    # krg
```

---

### Capillary Pressure Models

#### Brooks-Corey Capillary Pressure Model

The standard model for petroleum applications:

```python
capillary_model = bores.BrooksCoreyCapillaryPressureModel(
    # Saturation endpoints
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    
    # Oil-water system parameters (water-wet)
    oil_water_entry_pressure_water_wet=5.0,  # Entry/threshold pressure (psi)
    oil_water_pore_size_distribution_index_water_wet=2.0,  # λ (higher = narrower pore size dist)
    
    # Oil-water system parameters (oil-wet, used for mixed-wet)
    oil_water_entry_pressure_oil_wet=8.0,
    oil_water_pore_size_distribution_index_oil_wet=1.5,
    
    # Gas-oil system parameters
    gas_oil_entry_pressure=2.0,
    gas_oil_pore_size_distribution_index=2.5,
    
    # Wettability
    wettability=bores.WettabilityType.WATER_WET,  # WATER_WET, OIL_WET, MIXED_WET
    mixed_wet_water_fraction=0.6,  # Fraction water-wet (for MIXED_WET only)
)
```

**Wettability effects:**

- **WATER_WET**: Pcow > 0 (oil pressure > water pressure)
- **OIL_WET**: Pcow < 0 (water pressure > oil pressure)
- **MIXED_WET**: Pcow varies with saturation (weighted combination)

#### Van Genuchten Capillary Pressure Model

Alternative model with smoother transitions near endpoints:

```python
capillary_model = bores.VanGenuchtenCapillaryPressureModel(
    # Saturation endpoints
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    
    # Oil-water parameters (α in 1/psi, n > 1)
    oil_water_alpha_water_wet=0.1,   # Higher α = lower entry pressure
    oil_water_n_water_wet=2.5,       # Higher n = sharper transition
    oil_water_alpha_oil_wet=0.08,
    oil_water_n_oil_wet=2.0,
    
    # Gas-oil parameters
    gas_oil_alpha=0.15,
    gas_oil_n=2.2,
    
    # Wettability
    wettability=bores.WettabilityType.WATER_WET,
    mixed_wet_water_fraction=0.5,
)
```

#### Leverett J-Function Model

For scaling capillary pressure across rock types using the dimensionless J-function:

```python
capillary_model = bores.LeverettJCapillaryPressureModel(
    # Saturation endpoints
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    
    # Rock properties
    permeability=100.0,  # mD
    porosity=0.2,
    
    # Interfacial tensions (dynes/cm)
    oil_water_interfacial_tension=30.0,
    gas_oil_interfacial_tension=20.0,
    
    # Contact angles (degrees)
    contact_angle_oil_water=0.0,   # 0° = water-wet, 180° = oil-wet
    contact_angle_gas_oil=0.0,
    
    wettability=bores.WettabilityType.WATER_WET,
)
```

---

### Capillary Pressure Tables

Similar to relative permeability, two-phase tables must be combined.

#### Two-Phase Capillary Pressure Tables

```python
# Oil-water capillary pressure (Pcow = Po - Pw)
oil_water_pc = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=np.array([0.15, 0.25, 0.35, 0.50, 0.65, 0.75]),
    capillary_pressure=np.array([50.0, 15.0, 6.0, 2.0, 0.5, 0.0]),  # psi
)

# Gas-oil capillary pressure (Pcgo = Pg - Po)
gas_oil_pc = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=np.array([0.15, 0.30, 0.50, 0.70, 0.85, 0.95]),  # Oil saturation
    capillary_pressure=np.array([30.0, 12.0, 5.0, 1.5, 0.3, 0.0]),  # psi
)
```

#### Constructing Three-Phase Capillary Pressure Table

```python
# Combine into three-phase table
three_phase_pc = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=oil_water_pc,
    gas_oil_table=gas_oil_pc,
)

# Use in reservoir model
model = bores.reservoir_model(
    # ... other params ...
    capillary_pressure_table=three_phase_pc,
)
```

#### Querying Capillary Pressures

```python
result = three_phase_pc.get_capillary_pressures(
    water_saturation=0.3,
    oil_saturation=0.5,
    gas_saturation=0.2,
)

print(result["oil_water"])  # Pcow = Po - Pw (psi)
print(result["gas_oil"])    # Pcgo = Pg - Po (psi)
```

---

### Wettability Types

```python
# Using WettabilityType enum
bores.WettabilityType.WATER_WET   # Water preferentially wets rock
bores.WettabilityType.OIL_WET     # Oil preferentially wets rock
bores.WettabilityType.MIXED_WET   # Both water-wet and oil-wet regions

# Alias for convenience
bores.Wettability.WATER_WET  # Same as WettabilityType.WATER_WET
```

---

### Complete Example: Lab Data to Simulation

```python
import numpy as np
import bores

# 1. Define two-phase relative permeability from lab SCAL data
oil_water_kr = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=bores.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]),
    wetting_phase_relative_permeability=bores.array([0.0, 0.01, 0.04, 0.10, 0.20, 0.32, 0.38]),
    non_wetting_phase_relative_permeability=bores.array([1.0, 0.65, 0.40, 0.20, 0.08, 0.01, 0.0]),
)
gas_oil_kr = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=bores.array([0.25, 0.35, 0.50, 0.65, 0.80, 0.95]),
    wetting_phase_relative_permeability=bores.array([0.0, 0.08, 0.25, 0.50, 0.78, 1.0]),
    non_wetting_phase_relative_permeability=bores.array([0.85, 0.55, 0.28, 0.10, 0.02, 0.0]),
)

# 2. Combine into three-phase table with Stone II mixing
three_phase_kr = bores.ThreePhaseRelPermTable(
    oil_water_table=oil_water_kr,
    gas_oil_table=gas_oil_kr,
    mixing_rule=bores.stone_II_rule,
)

# 3. Define capillary pressure from lab data
oil_water_pc = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=bores.array([0.20, 0.30, 0.45, 0.60, 0.75]),
    capillary_pressure=bores.array([35.0, 12.0, 4.0, 1.0, 0.0]),
)
gas_oil_pc = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=bores.array([0.25, 0.40, 0.60, 0.80, 0.95]),
    capillary_pressure=bores.array([20.0, 8.0, 3.0, 0.8, 0.0]),
)
three_phase_pc = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=oil_water_pc,
    gas_oil_table=gas_oil_pc,
)

# 4. Use in reservoir model
model = bores.reservoir_model(
    # ... grid, properties, etc. ...
    relative_permeability_table=three_phase_kr,
    capillary_pressure_table=three_phase_pc,
)
```

---

## Constants

BORES also provides and uses physical constants and conversion factors with the context of a simulation run.

```python
import bores

# Access via the constants object
c = bores.c  # or bores.Constants()

# Time conversions
c.DAYS_PER_YEAR        # 365.25
c.SECONDS_PER_DAY      # 86400.0

# Standard conditions
c.STANDARD_PRESSURE_IMPERIAL    # 14.696 psia
c.STANDARD_TEMPERATURE_IMPERIAL # 60.0 °F

# Fluid properties
c.STANDARD_WATER_DENSITY_IMPERIAL  # 62.37 lb/ft³
c.IDEAL_GAS_CONSTANT_IMPERIAL            # 10.73 (psia·ft³)/(lbmol·°R)

# Conversion factors
c.PSI_TO_PA            # 6894.76
c.BBL_TO_FT3     # 5.6146

# And many others...
```

### Customizing Constants

```python
# Create custom constants context
my_constants = bores.Constants()
my_constants["STANDARD_TEMPERATURE_IMPERIAL"] = bores.Constant(
    70.0,  # Different standard temperature
    unit="°F",
    description="Custom standard temperature",
)
# Or just set constant directly
my_constants.STANDARD_PRESSURE_IMPERIAL = 70.0  # Different standard pressure

# Use in specific calculations
with bores.ConstantsContext(my_constants):
    # Calculations here use custom constants
    pass

# Or use constant object directly
with my_constants():
    # Calculations here use custom constants
    pass
```

---

## Boundary Conditions

### Boundary Directions

```python
from bores.boundary_conditions import BoundaryDirection

BoundaryDirection.X_MINUS  # Left face
BoundaryDirection.X_PLUS   # Right face
BoundaryDirection.Y_MINUS  # Front face
BoundaryDirection.Y_PLUS   # Back face
BoundaryDirection.Z_MINUS  # Bottom face
BoundaryDirection.Z_PLUS   # Top face
```

### Boundary Types

```python
# No-flow (default for reservoir boundaries)
no_flow = bores.NoFlowBoundary()

# Constant pressure/saturation (aquifer support)
constant = bores.ConstantBoundary(value=3000.0)  # psia

# Dirichlet (fixed value)
dirichlet = bores.DirichletBoundary(value=2800.0)

# Neumann (fixed flux)
neumann = bores.NeumannBoundary(flux=100.0)  # e.g bbl/day/ft²

# Linear gradient
gradient = bores.LinearGradientBoundary(
    gradient=0.433,      # psi/ft
    reference_value=3000.0,
    reference_position=0,
)
```

### Applying Boundaries

Boundary conditions are organized by property (pressure, saturation, etc.) using a conditions dictionary:

```python
# Create boundary conditions using `GridBoundaryCondition` for each property
boundary_conditions = bores.BoundaryConditions(
    conditions={
        # Pressure boundary conditions
        "pressure": bores.GridBoundaryCondition(
            left=bores.NoFlowBoundary(),
            right=bores.NoFlowBoundary(),
            front=bores.NoFlowBoundary(),
            back=bores.NoFlowBoundary(),
            bottom=bores.ConstantBoundary(constant=4600),  # Aquifer pressure
            top=bores.NoFlowBoundary(),
        ),
        # Oil saturation boundary conditions
        "oil_saturation": bores.GridBoundaryCondition(),  # Default: NoFlow all sides
        # Gas saturation boundary conditions
        "gas_saturation": bores.GridBoundaryCondition(),
        # Water saturation boundary conditions
        "water_saturation": bores.GridBoundaryCondition(),
    }
)

# Apply to model
model = bores.reservoir_model(
    ...,
    boundary_conditions=boundary_conditions,
)
```

### GridBoundaryCondition Parameters

The `GridBoundaryCondition` accepts boundary conditions for each face:

- `left` / `right`: X-direction faces
- `front` / `back`: Y-direction faces  
- `bottom` / `top`: Z-direction faces

---

## Errors

BORES defines specific exceptions for different error types:

```python
from bores.errors import (
    BORESError,           # Base class for all BORES errors
    ValidationError,      # Invalid input data
    SolverError,          # Solver convergence failure
    ComputationError,     # Numerical computation errors
    SimulationError,      # General simulation errors
    TimingError,          # Time stepping issues
    StorageError,         # State persistence errors
    StreamError,          # Streaming operation errors
    PreconditionerError,  # Preconditioner setup/application errors
    StopSimulation,       # Graceful simulation termination signal
)
```

### Error Handling

Below is an example of handling solver errors during simulation:

```python
import bores
from bores.errors import SolverError, StopSimulation

config = bores.Config(
    scheme="impes",
    convergence_tolerance=1e-6,
    max_iterations=250,
)

for _ in range(3):  # Retry up to 3 times
    try:
        for state in bores.run(model=model, timer=timer, wells=wells, config=config):
            ... # Process each state
        
    except SolverError as e:
        print(f"Solver failed to converge: {e}")
        # Try with relaxed settings
        config = bores.Config(
            scheme="impes",
            convergence_tolerance=1e-4,  # Relax tolerance
            max_iterations=500,
        )
    except StopSimulation:
        print("Simulation terminated gracefully")
        break
```

---

## Solvers

### Supported Solvers

BORES currently supports **IMPES** and **Explicit** schemes:

| Solver   | Pressure | Saturation | Use Case                         |
| -------- | -------- | ---------- | -------------------------------- |
| IMPES    | Implicit | Explicit   | General purpose, good stability  |
| Explicit | Explicit | Explicit   | Fast per step, requires small Δt |

### IMPES Configuration

```python
config = bores.Config(
    scheme="impes",
    
    # Iterative solver options
    iterative_solver="bicgstab",  # BiCGSTAB (recommended)
    # or: "lgmres", "gmres", "tfqmr"
    
    preconditioner="ilu",  # ILU (recommended)
    # or: "diagonal", "ilu", "amg", "cpr", None
    
    # Convergence settings
    convergence_tolerance=1e-6,
    max_iterations=250,
    
    # CFL control (internal)
    impes_cfl_threshold=0.9,
)
```

### Explicit Configuration

```python
config = bores.Config(
    scheme="explicit",
    
    # Separate CFL limits for pressure and saturation
    explicit_pressure_cfl_threshold=0.9,
    explicit_saturation_cfl_threshold=0.6,
)
```

> **Note**: Fully implicit scheme with Newton-Raphson iteration is planned for a future release.

---

## Visualization

BORES provides visualization utilities for time-series data and 3D reservoir visualization.

### Configure Image Output

```python
bores.image_config(scale=3)  # Higher DPI for publication-quality
```

### Time-Series Plots

```python
import numpy as np

# Prepare data: list of (time_step, value) tuples
pressure_history = [
    (state.step, state.model.fluid_properties.pressure_grid.mean()) 
    for state in states
]

# Create series plot
fig = bores.make_series_plot(
    data={"Avg. Reservoir Pressure": np.array(pressure_history)},
    title="Pressure Decline",
    x_label="Time Step",
    y_label="Pressure (psia)",
    marker_sizes=6,
    show_markers=True,
    width=720,
    height=460,
)
fig.show()
```

### Multi-Series Plots

```python
# Multiple data series
saturation_data = {
    "Oil Saturation": np.array(oil_sat_history),
    "Water Saturation": np.array(water_sat_history),
    "Gas Saturation": np.array(gas_sat_history),
}

fig = bores.make_series_plot(
    data=saturation_data,
    title="Saturation History",
    x_label="Time Step",
    y_label="Saturation (fraction)",
    line_colors=["brown", "blue", "red"],  # Custom colors
)
fig.show()
```

### Merging Plots

```python
# Create individual plots
oil_plot = bores.make_series_plot(data={"Oil": oil_data}, title="Oil Production")
gas_plot = bores.make_series_plot(data={"Gas": gas_data}, title="Gas Production")

# Merge into subplot grid
combined = bores.merge_plots(
    oil_plot,
    gas_plot,
    cols=2,
    title="Production Analysis",
)
combined.show()
```

### 3D Visualization

```python
from bores.visualization import plotly3d

# Create a DataVisualizer
viz = plotly3d.DataVisualizer()

# Volume rendering of pressure
fig = viz.make_plot(
    source=states[-1],  # ModelState from simulation
    property="pressure",  # Property name from registry
    plot_type="volume",  # "volume", "isosurface", "scatter_3d", "cell_blocks"
    title="Pressure Distribution",
    width=720,
    height=460,
    opacity=0.67,
    z_scale=3,  # Exaggerate vertical scale
)
fig.show()

# With wells overlay and labels
labels = plotly3d.Labels()
wells = states[0].wells
injector_locations, producer_locations = wells.locations
injector_names, producer_names = wells.names
labels.add_well_labels(
    [*injector_locations, *producer_locations],
    [*injector_names, *producer_names]
)

fig = viz.make_plot(
    source=states[-1],
    property="oil-saturation",
    plot_type="scatter_3d",
    show_wells=True,
    show_surface_marker=True,
    show_perforations=True,
    labels=labels,
    aspect_mode="data",
    marker_size=4,
)
fig.show()

# Slice visualization (view specific layers)
fig = viz.make_plot(
    source=states[-1],
    property="temperature",
    x_slice=(0, 25),  # First 25 cells in X
    y_slice=(0, 25),  # First 25 cells in Y
    z_slice=5,        # Single layer at Z=5
)
fig.show()
```

### Model Analysis

Use `ModelAnalyst` for common analysis operations:

> Check the `bores.analyses` module for more details or check `scenerios/*_analysis.py` for real usage examples.

```python
store = bores.ZarrStore(
    store=Path("/results/simulation.zarr"), 
    metadata_dir=Path("/results/metadata")
)
# Create stream in store replay mode (no need for lazy loading since grid will mostly be used immediately)
stream = bores.StateStream(
    store=store, 
    lazy_load=False, 
    auto_replay=True
)
# Collect only the initial state and states at every 10th step
states = stream.collect(key=lambda s: s.step == 0 or s.step % 10 == 0) # this returns a generator
analyst = bores.ModelAnalyst(states)

# Sweep efficiency over time
sweep_history = analyst.sweep_efficiency_history(
    interval=1,
    from_step=1,
    displacing_phase="water",
)

# Production rates
oil_production = analyst.oil_production_history(
    interval=1,
    cumulative=False,
    from_step=1,
)

# Cumulative production
cumulative_oil = analyst.oil_production_history(
    interval=1,
    cumulative=True,
    from_step=1,
)

# Injection history
gas_injection = analyst.gas_injection_history(
    interval=1,
    cumulative=True,
    from_step=1,
)

# Instantaneous rates with water cut and GOR
rates = analyst.instantaneous_rates_history(
    interval=1,
    from_step=1,
    rate_type="production",
)
for step, result in rates:
    print(f"Step {step}: WOR = {result.water_cut:.3f}, GOR = {result.gas_oil_ratio:.1f}")
```

---

## Performance & Optimization

BORES has been optimized for reasonable performance within Python's constraints, but users should understand the computational trade-offs involved in reservoir simulation.

### Computational Complexity

Reservoir simulation is inherently computationally intensive. Several factors affect simulation time:

| Factor                 | Impact on Performance                                                      |
| ---------------------- | -------------------------------------------------------------------------- |
| **Grid Size**          | Scales roughly as O(n³) for 3D grids. A 100×100×50 grid has 500,000 cells. |
| **Number of Wells**    | Each well adds coupling terms and flow calculations per time step.         |
| **Fractures & Faults** | Increase matrix complexity and may reduce sparsity.                        |
| **Time Step Size**     | Smaller steps = more iterations; larger steps may hit CFL limits.          |
| **Solver Iterations**  | Complex pressure/saturation distributions require more iterations.         |
| **PVT Table Lookups**  | 3D interpolation (P-T-S) is slower than 2D (P-T).                          |

### CFL Condition & Time Stepping

The Courant-Friedrichs-Lewy (CFL) condition limits the maximum stable time step for explicit schemes:

```math
CFL = (velocity × Δt) / Δx ≤ CFL_max
```

**Implications:**

- High flow velocities (near wells, fractures) force smaller time steps
- Finer grids require smaller time steps for the same CFL number
- The adaptive timer will automatically reduce step size when CFL limits are exceeded

**If stability is not critical** (e.g., for quick prototyping):

```python
# Relax CFL constraints (use with caution!)
config = bores.Config(
    impes_cfl_threshold=1.2,           # Default is 0.9
    explicit_saturation_cfl_threshold=0.8,  # Default is 0.6
    explicit_pressure_cfl_threshold=1.0,    # Default is 0.9
)
```

> ⚠️ **Warning**: Relaxing CFL limits may cause numerical instability, oscillations, or non-physical results. Always validate results when using relaxed settings.

### Memory Considerations

Large grids consume significant memory especially when running analysis with multiple property grids (pressure, saturations, mobilities, etc.):

| Grid Size   | Cells | ~Memory per Property Grid |
| ----------- | ----- | ------------------------- |
| 50×50×20    | 50K   | ~400 KB (float64)         |
| 100×100×50  | 500K  | ~4 MB (float64)           |
| 200×200×100 | 4M    | ~32 MB (float64)          |

With 20+ property grids, state history storage, and solver matrices, memory can grow quickly.

**Tips for reducing memory:**

- Use 32-bit precision: `bores.use_32bit_precision()` (halves grid memory usage)
- Use `StateStream` with a `StateStore` to persist states to disk instead of holding in memory. Most stores also support lazy loading, so use that when analyzing results.
- Increase `output_frequency` in `Config` to store fewer states
- Use coarser grids during prototyping, then refine

### Performance Tips

1. **Use 32-bit precision** for large grids (significant speedup with NumPy/Numba)
2. **Start with coarse grids** and refine after validating the model setup
3. **Use `bicgstab` solver** with `ilu` preconditioner for most cases. Although using a `diagonal` or no preconditioner may be faster, it may require more iterations.
4. **Batch state storage** with `StateStream` to avoid memory buildup
5. **Avoid keeping states in memory** unless necessary for analysis. When using `bores.run`, iterate through states directly or use a store. Never do list(bores.run(...)) for large simulations.
6. **Use `StateStream` for post-simulation analysis** — Model analysis with `ModelAnalyst` can be memory-intensive since all collected states need to be loaded into memory. Use `StateStream.collect()` with a predicate to filter only the timesteps you need for analysis (e.g., every 10th step, specific time intervals, or final state only). This prevents loading hundreds of states when only a subset is needed.
7. **Profile your simulation** to identify bottlenecks:

   ```python
   import cProfile
   cProfile.run('list(bores.run(model, timer, wells, config))')
   ```

   ```python
   import memory_profiler

   @memory_profiler.profile
   def run_simulation():
       # Setup model, timer, wells, config and run simulation
       ...
    
   run_simulation()
   ```

### Language & Hardware Limitations

BORES is written in Python with NumPy, SciPy, and Numba for numerical operations. While optimizations have been made:

- **Python's GIL** limits true parallelism for CPU-bound operations
- **Numba JIT compilation** helps but has startup overhead (only on first run)
- **Memory bandwidth** often limits performance more than CPU speed
- **Single-threaded by design** — commercial simulators use MPI/GPU parallelism

For large-scale production simulations requiring maximum performance, consider commercial simulators (ECLIPSE, CMG, tNavigator) or compiled frameworks (OPM Flow, MRST).

---

## Troubleshooting & FAQ

### Frequently Asked Questions

**Q: My simulation is running very slowly. What can I do?**

A: Several factors affect speed:

1. Reduce grid resolution during prototyping
2. Enable 32-bit precision: `bores.use_32bit_precision()`
3. Increase `min_step_size` in Timer if stability permits
4. Use `output_frequency > 1` to reduce state storage overhead
5. Check if wells have very high rates causing CFL violations

**Q: The solver is not converging. What should I check?**

A: Non-convergence usually indicates numerical issues:

1. Check for extreme property values (very high/low permeability, porosity near 0 or 1)
2. Ensure saturation endpoints are physical (Swc + Sor < 1)
3. Try a stronger preconditioner (`"amg"` instead of `"ilu"`)
4. Reduce initial time step size
5. Check well BHP/control limits aren't causing numerical instability

**Q: I'm getting negative saturations or saturations > 1.**

A: This indicates numerical instability, although BORES enforces saturation bounds:

1. Reduce time step size (lower `max_step_size`, `initial_step_size`)
2. Tighten CFL thresholds in `Config`
3. Check relative permeability endpoints for consistency
4. Ensure capillary pressure curves are monotonic

**Q: How do I handle a gas cap or aquifer?**

A: Use boundary conditions:

```python
# Bottom aquifer support
boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(
            bottom=bores.ConstantBoundary(constant=4500),  # Aquifer pressure
        ),
    }
)
```

For gas caps, set appropriate initial saturations using `build_saturation_grids` with `gas_oil_contact`.

**Q: Can BORES handle compositional simulation?**

A: No, BORES is specifically designed for black-oil (3-phase) simulation. Compositional simulation with equation-of-state (EOS) calculations is not supported.

**Q: How do I restart a simulation from a saved state?**

A: Load the final state from storage and use it to rebuild the model:

```python
# Load states
store = bores.HDF5Store(filepath=Path("results/simulation.h5"), metadata_dir=Path("results/metadata"))
stream = bores.StateStream(store=store, validate=False, auto_replay=True, lazy_load=False)
last_state = stream.last()

# Continue simulation with last recorded timer state
timer = bores.Timer.load_state(last_state.timer_state)
```

**Q: Why are my well rates different from what I specified?**

A: Wells operate under constraints:

1. BHP limits may cap achievable rates
2. Reservoir deliverability may be insufficient
3. Skin factor reduces productivity
4. Check `warn_well_anomalies=True` in Config for warnings

**Q: The simulation crashes with "out of memory" error.**

A: Large simulations can exhaust RAM (especially with many stored states). Solutions:

1. Use `StateStream` with disk storage instead of holding all states in memory
2. Enable 32-bit precision
3. Reduce grid resolution
4. Increase `output_frequency` to store fewer states

---

## Contributing & Support

### Getting Help

- **Questions & Discussions**: Open a [GitHub Discussion](https://github.com/ti-oluwa/bores/discussions) for questions, ideas, or general conversation
- **Bug Reports**: File an [Issue](https://github.com/ti-oluwa/bores/issues) with a minimal reproducible example
- **Feature Requests**: Open an Issue describing the use case and proposed feature

### Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest`)
5. Submit a Pull Request

**Areas where contributions are especially welcome:**

- Additions, corrections or improvements to PVT correlations, relperm models, capillary pressure models
- New well models/controls or enhancements to existing well controls
- Improvements to grid construction utilities
- Enhancements to visualization utilities
- Performance optimizations
- Documentation improvements
- Bug fixes and test coverage

### API Reference

For detailed API information beyond this README:

1. **Source Code**: The source code contains comprehensive docstrings for all public APIs. Browse the `src/bores/` directory for detailed documentation on each module.

2. **Module Structure**:
   - `bores/factories.py` — Main factory functions (`reservoir_model`, `production_well`, etc.)
   - `bores/grids/` — Grid construction utilities
   - `bores/wells/` — Well models and controls
   - `bores/pvt/` — PVT correlations and tables
   - `bores/relperm.py` — Relative permeability models
   - `bores/capillary_pressures.py` — Capillary pressure models
   - `bores/fractures.py` — Fracture and fault models
   - `bores/states.py` — State management and storage
   - `bores/visualization/` — Plotting utilities
   - `bores/analyses.py` — Post-simulation analysis tools

3. **Examples**: The `scenarios/` directory contains complete working examples demonstrating various simulation workflows (primary depletion, CO₂ injection, CH₄ injection, etc.)

> 📚 **Coming Soon**: Full API documentation with detailed usage guides, tutorials, and best practices.

---

## Example: Complete Simulation Workflow on a Heterogeneous Reservoir Model

This complete example demonstrates building a heterogeneous reservoir model with wells and running a simulation. It follows the actual API patterns used in the scenario files.

```python
import typing
import logging
from pathlib import Path
import numpy as np
import bores

logging.basicConfig(level=logging.INFO)

# 1. Enable 32-bit precision for better performance
bores.use_32bit_precision()

# 2. Define grid dimensions
cell_dimension = (100.0, 100.0)  # 100ft x 100ft cells (dx, dy)
grid_shape = typing.cast(
    bores.ThreeDimensions,
    (20, 20, 10),  # 20x20 cells, 10 layers
)
dip_angle = 2.0
dip_azimuth = 90.0  # Dipping toward East

# 3. Build thickness grid (variable layer thickness)
thickness_values = bores.array([30.0, 20.0, 25.0, 30.0, 25.0, 30.0, 20.0, 25.0, 30.0, 25.0])  # ft
thickness_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=thickness_values,
    orientation=bores.Orientation.Z,
)

# 4. Build pressure grid (hydrostatic gradient)
reservoir_top_depth = 8000.0  # ft
pressure_gradient = 0.38  # psi/ft
layer_depths = reservoir_top_depth + np.cumsum(np.concatenate([[0], thickness_values[:-1]]))
layer_pressures = 14.7 + (layer_depths * pressure_gradient)

pressure_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=layer_pressures,
    orientation=bores.Orientation.Z,
)

# Bubble point pressure (undersaturated oil)
oil_bubble_point_pressure_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=layer_pressures - 400.0,  # 400 psi below formation pressure
    orientation=bores.Orientation.Z,
)

# 5. Build saturation endpoint grids
residual_oil_saturation_water_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.25)
residual_oil_saturation_gas_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.15)
irreducible_water_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.15)
connate_water_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.12)
residual_gas_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.045)

# 6. Build porosity grid (compaction trend)
porosity_values = bores.array([0.04, 0.07, 0.09, 0.1, 0.08, 0.12, 0.14, 0.16, 0.11, 0.08])
porosity_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=porosity_values,
    orientation=bores.Orientation.Z,
)

# 7. Build depth grid with structural dip
goc_depth = 8060.0  # Gas-oil contact
owc_depth = 8220.0  # Oil-water contact

depth_grid = bores.depth_grid(thickness_grid)
depth_grid = bores.apply_structural_dip(
    elevation_grid=depth_grid,
    elevation_direction="downward",
    cell_dimension=cell_dimension,
    dip_angle=dip_angle,
    dip_azimuth=dip_azimuth,
)

# 8. Build saturation grids with fluid contacts
water_saturation_grid, oil_saturation_grid, gas_saturation_grid = bores.build_saturation_grids(
    depth_grid=depth_grid,
    gas_oil_contact=goc_depth - reservoir_top_depth,
    oil_water_contact=owc_depth - reservoir_top_depth,
    connate_water_saturation_grid=connate_water_saturation_grid,
    residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
    residual_gas_saturation_grid=residual_gas_saturation_grid,
    porosity_grid=porosity_grid,
    use_transition_zones=True,
    oil_water_transition_thickness=12.0,
    gas_oil_transition_thickness=8.0,
    transition_curvature_exponent=1.2,
)

# 9. Build oil viscosity grid (increases with depth)
oil_viscosity_values = np.linspace(1.2, 2.5, grid_shape[2])  # cP
oil_viscosity_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=oil_viscosity_values,
    orientation=bores.Orientation.Z,
)

# Oil compressibility and specific gravity
oil_compressibility_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.2e-5)  # 1/psi
oil_specific_gravity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.845)  # ~36 API

# 10. Build permeability structure (anisotropic)
x_perm_values = bores.array([12, 25, 40, 18, 55, 70, 90, 35, 48, 22])  # mD
x_permeability_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=x_perm_values,
    orientation=bores.Orientation.Z,
)
y_permeability_grid = typing.cast(bores.ThreeDimensionalGrid, x_permeability_grid * 0.8)
z_permeability_grid = typing.cast(bores.ThreeDimensionalGrid, x_permeability_grid * 0.1)

absolute_permeability = bores.RockPermeability(
    x=x_permeability_grid,
    y=y_permeability_grid,
    z=z_permeability_grid,
)

# 11. Create relative permeability model
relative_permeability_table = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.045,
    wettability=bores.WettabilityType.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

# 12. Create capillary pressure model
capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.WettabilityType.WATER_WET,
)

# 13. Build temperature grid (geothermal gradient)
surface_temp = 60.0  # °F
temp_gradient = 0.015  # °F/ft
layer_temps = surface_temp + (layer_depths * temp_gradient)
temperature_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=layer_temps,
    orientation=bores.Orientation.Z,
)

# Rock compressibility
rock_compressibility = 4.5e-6  # 1/psi

# Net-to-gross ratio
net_to_gross_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=[0.42, 0.55, 0.68, 0.35, 0.60, 0.72, 0.80, 0.50, 0.63, 0.47],
    orientation=bores.Orientation.Z,
)

# Gas gravity
gas_gravity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.65)

# 14. Create boundary conditions
boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(
            bottom=bores.ConstantBoundary(constant=4600),  # Aquifer support
        ),
    }
)

# 15. Build PVT tables
pvt_table_data = bores.build_pvt_table_data(
    pressures=bores.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]),
    temperatures=bores.array([120, 140, 160, 180, 200, 220]),
    salinities=bores.array([30000, 32000, 36000, 40000]),
    oil_specific_gravity=0.845,
    gas_gravity=0.65,
    reservoir_gas="methane",
)
pvt_tables = bores.PVTTables(table_data=pvt_table_data, interpolation_method="linear")

# 16. Build the reservoir model
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness_grid,
    pressure_grid=pressure_grid,
    oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
    absolute_permeability=absolute_permeability,
    porosity_grid=porosity_grid,
    temperature_grid=temperature_grid,
    rock_compressibility=rock_compressibility,
    oil_saturation_grid=oil_saturation_grid,
    water_saturation_grid=water_saturation_grid,
    gas_saturation_grid=gas_saturation_grid,
    oil_viscosity_grid=oil_viscosity_grid,
    oil_specific_gravity_grid=oil_specific_gravity_grid,
    oil_compressibility_grid=oil_compressibility_grid,
    gas_gravity_grid=gas_gravity_grid,
    residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
    irreducible_water_saturation_grid=irreducible_water_saturation_grid,
    connate_water_saturation_grid=connate_water_saturation_grid,
    residual_gas_saturation_grid=residual_gas_saturation_grid,
    net_to_gross_ratio_grid=net_to_gross_grid,
    boundary_conditions=boundary_conditions,
    relative_permeability_table=relative_permeability_table,
    capillary_pressure_table=capillary_pressure_table,
    reservoir_gas="methane",
    dip_angle=dip_angle,
    dip_azimuth=dip_azimuth,
    pvt_tables=pvt_tables,
)

# 17. (Optional) Add a sealing fault
fault = bores.vertical_sealing_fault(
    fault_id="F-1",
    orientation="x",
    index=10,
    permeability_multiplier=1e-4,
    z_range=(0, 8),  # Shallow fault
)
model = bores.apply_fracture(model, fault)

# 18. Create production well
production_clamp = bores.ProductionClamp()
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-150,
        target_phase="oil",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-500,
        target_phase="gas",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-10,
        target_phase="water",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((10, 10, 4), (10, 10, 6))],
    radius=0.3542,  # 8.5" wellbore
    control=control,
    produced_fluids=(
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.845,
            molecular_weight=180.0,
        ),
        bores.ProducedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.05,
            molecular_weight=18.015,
        ),
    ),
    skin_factor=2.5,
    is_active=True,
)

wells = bores.wells_(injectors=None, producers=[producer])

# 19. Configure timer and simulation
timer = bores.Timer(
    initial_step_size=bores.Time(hours=4.5),
    max_step_size=bores.Time(days=5.0),
    min_step_size=bores.Time(hours=1.0),
    simulation_time=bores.Time(days=365),  # 1 year
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
)

config = bores.Config(
    scheme="impes",
    output_frequency=1,
    miscibility_model="immiscible",
    use_pseudo_pressure=True,
    max_iterations=500,
    iterative_solver="bicgstab",
    preconditioner="ilu",
    pvt_tables=pvt_tables,
)

# 20. Run simulation with storage
store = bores.ZarrStore(
    store=Path.cwd() / "results/simulation.zarr",
    metadata_dir=Path.cwd() / "results/metadata",
)

stream = bores.StateStream(
    bores.run(model=model, timer=timer, wells=wells, config=config),
    store=store,
    batch_size=50,
)
with stream: # Use context manager to ensure proper stream setup and teardown
    for state in stream.collect(key=lambda s: s.step % 10 == 0):
        avg_pressure = state.model.fluid_properties.pressure_grid.mean()
        print(f"Step {state.step}: Avg pressure = {avg_pressure:.1f} psia")

# 21. Analyze results (reuse stream with `auto_replay` for memory-efficient analysis)
stream = bores.StateStream(store=store, lazy_load=False, auto_replay=True)
# Collect only states at every 5th step to reduce memory footprint
states = list(stream.collect(key=lambda s: s.step == 0 or s.step % 5 == 0))
analyst = bores.ModelAnalyst(states)

# Cumulative oil production
oil_cum = list(analyst.oil_production_history(interval=1, cumulative=True, from_step=1))
print(f"Total oil produced: {oil_cum[-1][1]:.0f} STB")

# Plot production history (collect states again for a separate analysis)
pressure_history = [(s.step, s.model.fluid_properties.pressure_grid.mean()) for s in states]
fig = bores.make_series_plot(
    data={"Avg. Reservoir Pressure": np.array(pressure_history)},
    title="Pressure Decline",
    x_label="Time Step",
    y_label="Pressure (psia)",
)
fig.show()
```

---

## License

See [LICENSE](LICENSE) for details.

---

*Built for students, researchers, reservoir engineers and simulation enthusiasts.*
