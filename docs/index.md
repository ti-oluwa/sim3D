<p align="center">
  <img src="images/logo.svg" alt="BORES Logo" width="200">
</p>


# BORES - Black-Oil REservoir Simulation Framework

BORES is a 3D three-phase black-oil reservoir modelling and simulation framework written in Python. It provides a Pythonic API for constructing reservoir models, defining wells and boundary conditions, running multiphase flow simulations, and analyzing results. BORES targets petroleum engineers, researchers, and students who need a transparent, scriptable alternative to closed-source commercial simulators.

!!! warning "Research and Educational Use"

    BORES is currently in **alpha** (v0.1.0). It is designed for educational and research purposes. Do not use it for production field development planning or regulatory submissions without independent verification. Results should always be validated against analytical solutions or established commercial simulators before drawing engineering conclusions.

---

## Quick Example

The following example sets up and runs a small 3D waterflood simulation from scratch. It defines a 10x10x3 grid, places one injector and one producer, and runs for 365 days using the default IMPES scheme.

```python
import bores
import logging
import numpy as np

# Set log level
logging.basicConfig(level=logging.INFO)

# Set precision (32-bit is the default)
bores.use_32bit_precision()

# Grid dimensions: 10x10x3 cells, each 100 ft x 100 ft, 20 ft thick
grid_shape = (10, 10, 3)
cell_dimension = (100.0, 100.0)

# Build property grids
thickness = bores.build_uniform_grid(grid_shape, value=20.0)  # ft
pressure = bores.build_uniform_grid(grid_shape, value=3000.0)  # psi
porosity = bores.build_uniform_grid(grid_shape, value=0.20)  # fraction
temperature = bores.build_uniform_grid(grid_shape, value=180.0)  # deg F
oil_viscosity = bores.build_uniform_grid(grid_shape, value=1.5)  # cP
bubble_point = bores.build_uniform_grid(grid_shape, value=2500.0)  # psi

# Residual and irreducible saturations
Sorw = bores.build_uniform_grid(grid_shape, value=0.20)
Sorg = bores.build_uniform_grid(grid_shape, value=0.15)
Sgr = bores.build_uniform_grid(grid_shape, value=0.05)
Swir = bores.build_uniform_grid(grid_shape, value=0.20)
Swc = bores.build_uniform_grid(grid_shape, value=0.20)

# Build depth grid and compute initial saturations from fluid contacts
depth = bores.build_depth_grid(thickness, datum=5000.0)  # Top at 5000 ft
Sw, So, Sg = bores.build_saturation_grids(
    depth_grid=depth,
    gas_oil_contact=4999.0,  # Above reservoir (no gas cap)
    oil_water_contact=5100.0,  # Below reservoir (all oil zone)
    connate_water_saturation_grid=Swc,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    porosity_grid=porosity,
)

# Isotropic permeability: 100 mD
perm_grid = bores.build_uniform_grid(grid_shape, value=100.0)
permeability = bores.RockPermeability(x=perm_grid, y=perm_grid, z=perm_grid)

oil_sg_grid = bores.build_uniform_grid(grid_shape, value=0.85)

# Build the reservoir model
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness,
    pressure_grid=pressure,
    rock_compressibility=3e-6,
    absolute_permeability=permeability,
    porosity_grid=porosity,
    temperature_grid=temperature,
    water_saturation_grid=Sw,
    gas_saturation_grid=Sg,
    oil_saturation_grid=So,
    oil_viscosity_grid=oil_viscosity,
    oil_bubble_point_pressure_grid=bubble_point,
    residual_oil_saturation_water_grid=Sorw,
    residual_oil_saturation_gas_grid=Sorg,
    residual_gas_saturation_grid=Sgr,
    irreducible_water_saturation_grid=Swir,
    connate_water_saturation_grid=Swc,
    oil_specific_gravity_grid=oil_sg_grid,
    datum_depth=5000.0,
)

# Define wells
injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((0, 0, 0), (0, 0, 0))],
    radius=0.25,
    control=bores.ConstantRateControl(
        target_rate=500.0,
        bhp_limit=5000.0,
        clamp=bores.InjectionClamp(),
    ),
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.0,
        molecular_weight=18.015,
    ),
)
producer = bores.production_well(
    well_name="PROD-1",
    perforating_intervals=[((9, 9, 2), (9, 9, 2))],
    radius=0.25,
    control=bores.PrimaryPhaseRateControl(
        primary_phase=bores.FluidPhase.OIL,
        primary_control=bores.AdaptiveBHPRateControl(
            target_rate=-500.0,
            target_phase="oil",
            bhp_limit=1000.0,
            clamp=bores.ProductionClamp(),
        ),
        secondary_clamp=bores.ProductionClamp(),
    ),
    produced_fluids=[
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.85,
            molecular_weight=200.0,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.0,
            molecular_weight=18.015,
        ),
    ],
)
wells = bores.wells_(injectors=[injector], producers=[producer])

# Rock-fluid tables (Brooks-Corey relative permeability + capillary pressure)
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=bores.BrooksCoreyThreePhaseRelPermModel(
        water_exponent=2.0,
        oil_exponent=2.0,
        gas_exponent=2.0,
    ),
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(),
)

# Simulation configuration
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(days=1),
        max_step_size=bores.Time(days=10),
        min_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(days=365),
    ),
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)

# Run the simulation and collect states
states = list(bores.run(model, config))
final = states[-1]
print(f"Completed {final.step} steps in {final.time_in_days:.1f} days")
print(
    f"Final avg pressure: {final.model.fluid_properties.pressure_grid.mean():.1f} psi"
)
```

<p align="center">
  <img src="images/quick-example.gif" width="800" alt="Quick Example"/>
</p>

---

## Key Capabilities

BORES covers the core elements of black-oil reservoir simulation within a single Python package.

**Three-phase flow simulation** - Solves coupled pressure and saturation equations for oil, water, and gas phases using the IMPES (Implicit Pressure, Explicit Saturation) scheme. Explicit and fully implicit schemes are also available.

**PVT correlations and tables** - Ships with industry-standard correlations (Standing, Vasquez-Beggs, Lee-Gonzalez, and others) for computing fluid properties from pressure and temperature. You can also supply your own PVT tables for direct lookup.

**Well modelling** - Supports injection and production wells with multiple control modes: constant rate, BHP control, and adaptive BHP-rate control. Wells can have multiple perforating intervals, skin factors, and scheduled control changes.

**Flexible boundary conditions** - Includes constant pressure, no-flow, flux boundaries, periodic boundaries, and Carter-Tracy aquifer models. Boundary conditions can be combined on different faces of the reservoir grid.

**Post-simulation analysis** - Built-in analysis tools for computing recovery factors, production profiles, front tracking, and mobility ratios. Plotly-based visualization produces 1D series, 2D maps, and interactive 3D volume renders.

---

## Who Is BORES For?

BORES is built for people who want to understand and experiment with reservoir simulation at the code level. If you are a petroleum engineering student working through textbook problems, BORES lets you set up models programmatically and inspect every intermediate calculation. You can trace how pressure propagates through a grid, watch saturation fronts develop, and compare numerical results against analytical solutions.

Researchers working on new recovery methods, relative permeability models, or solver algorithms will find value in BORES as a testbed. The codebase uses immutable data models, making it straightforward to run parameter sweeps or compare different configurations without worrying about accidental state mutation. The factory-function design keeps model construction explicit and auditable.

Practicing engineers who want a quick scripting tool for screening studies or generating initial estimates may also find BORES useful, provided they validate results against trusted tools. BORES is not a replacement for commercial simulators like Eclipse, CMG, or tNavigator, but it serves well as a complementary learning and prototyping tool.

---

## Getting Started

<div class="grid cards" markdown>

- **Installation**

    ---

    Install BORES with `uv` or `pip` and verify your setup.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

- **Quickstart**

    ---

    Build and run your first reservoir simulation in under 5 minutes.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

- **Core Concepts**

    ---

    Understand the simulation pipeline, data model design, and conventions.

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

</div>
