# Boundary Conditions

## Overview

Boundary conditions define how the edges of your simulation grid interact with the outside world. By default, all grid boundaries are no-flow (closed), meaning no fluid enters or leaves through any face of the grid. This is appropriate for many reservoir simulations where the reservoir is bounded by impermeable rock, but many real reservoirs have open boundaries that communicate with aquifers, adjacent formations, or surface conditions.

BORES provides several boundary condition types that model different physical situations. You can set different conditions on each face of the grid (left, right, front, back, top, bottom), and you can combine multiple boundary types to model complex reservoir geometries. Each boundary condition specifies the physical behavior at one or more grid faces.

The boundary condition system follows the standard classification from partial differential equations: Dirichlet (fixed value), Neumann (fixed flux), and Robin (mixed) conditions. In reservoir simulation terms, these correspond to constant pressure boundaries, fixed injection/production rate boundaries, and boundaries with pressure-dependent flow, respectively.

---

## Boundary Condition Types

### No-Flow (Default)

No-flow boundaries prevent any fluid from crossing the boundary face. This is the default for all grid faces and represents impermeable rock surrounding the reservoir.

```python
from bores.boundary_conditions import NoFlowBoundary

no_flow = NoFlowBoundary()
```

You rarely need to create `NoFlowBoundary` objects explicitly because they are applied automatically to all faces that do not have another condition assigned.

### Constant Pressure (Dirichlet)

A constant pressure boundary maintains a fixed pressure on the boundary face. Fluid flows into or out of the reservoir to maintain this pressure. This models an infinite aquifer or a large connected volume that acts as a pressure source or sink.

```python
from bores.boundary_conditions import ConstantBoundary

# 2500 psi on the left face
constant_p = ConstantBoundary(value=2500.0)
```

Constant pressure boundaries provide infinite pressure support. If the reservoir pressure drops below the boundary pressure, water flows in. If it rises above, fluid flows out. This is the simplest aquifer model but is unrealistic for finite aquifers that lose pressure support over time.

### Fixed Flux (Neumann)

A flux boundary specifies a fixed volumetric flow rate across the boundary face. Positive values indicate inflow (injection), negative values indicate outflow (production).

```python
from bores.boundary_conditions import FluxBoundary

# Inject 100 RB/day through the bottom face
influx = FluxBoundary(value=100.0)

# Produce 50 RB/day from the right face
outflux = FluxBoundary(value=-50.0)
```

!!! info "Sign Convention"

    Throughout BORES, positive flow means injection (into the reservoir) and negative flow means production (out of the reservoir). This applies to wells, boundary conditions, and all flow terms.

### Linear Gradient

A linear gradient boundary applies a pressure that varies linearly across the boundary face:

```python
from bores.boundary_conditions import LinearGradientBoundary

# Pressure varies from 2500 to 2800 psi across the face
gradient = LinearGradientBoundary(
    base_value=2500.0,
    gradient=0.5,  # psi per ft
)
```

This is useful for modeling tilted aquifer contacts or regional pressure gradients.

### Time-Dependent Boundary

A time-dependent boundary changes its value according to a function of time:

```python
from bores.boundary_conditions import TimeDependentBoundary

# Pressure that declines over time
def declining_pressure(t):
    """t is time in seconds."""
    return 2500.0 - 0.1 * (t / 86400.0)  # 0.1 psi/day decline

time_bc = TimeDependentBoundary(function=declining_pressure)
```

### Periodic Boundary

Periodic boundaries connect opposite faces of the grid, so fluid leaving one face enters the opposite face. This creates a repeating tile pattern and is useful for simulating a small section of a larger, repeating well pattern.

```python
from bores.boundary_conditions import PeriodicBoundary

periodic = PeriodicBoundary()
```

!!! warning "Periodic Boundary Pairing"

    Periodic boundaries must be applied to both opposite faces simultaneously. If the left face is periodic, the right face must also be periodic. BORES validates this during configuration and raises a `ValidationError` if periodic boundaries are not properly paired.

### Robin (Mixed) Boundary

A Robin boundary combines Dirichlet and Neumann conditions, specifying a relationship between the boundary value and the flux. This is useful for modeling semi-permeable barriers.

```python
from bores.boundary_conditions import RobinBoundary

robin = RobinBoundary(alpha=1.0, beta=0.5, gamma=2500.0)
```

The Robin condition is: $\alpha \cdot u + \beta \cdot \frac{\partial u}{\partial n} = \gamma$, where $u$ is the variable (pressure), $n$ is the outward normal, and $\alpha$, $\beta$, $\gamma$ are parameters.

---

## Custom Boundary Functions

For complex boundary behavior that is not captured by the built-in types, you can define custom boundary functions and register them for serialization:

```python
from bores.boundary_conditions import boundary_function, SpatialBoundary

@boundary_function
def hydrostatic_pressure(x, y, depth):
    """Hydrostatic pressure gradient at the boundary."""
    return 14.696 + 0.433 * depth

spatial_bc = SpatialBoundary(function=hydrostatic_pressure)
```

The `@boundary_function` decorator registers the function by name, enabling it to be serialized and loaded from disk. Without registration, custom boundary functions cannot be saved as part of a simulation configuration.

---

## Applying Boundary Conditions

`BoundaryConditions` is a defaultdict that maps property names (strings) to `GridBoundaryCondition` objects. Each `GridBoundaryCondition` assigns a boundary type to each face of the grid. The property name tells BORES which equation the boundary applies to (for example, `"pressure"` for the pressure equation, `"temperature"` for heat transfer).

```python
from bores.boundary_conditions import (
    GridBoundaryCondition,
    BoundaryConditions,
    ConstantBoundary,
    NoFlowBoundary,
    FluxBoundary,
)

boundary_conditions = BoundaryConditions(
    conditions={
        "pressure": GridBoundaryCondition(
            left=ConstantBoundary(constant=2500.0),
            right=NoFlowBoundary(),
            front=NoFlowBoundary(),
            back=NoFlowBoundary(),
            top=NoFlowBoundary(),
            bottom=ConstantBoundary(constant=2600.0),
        ),
    },
)

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    boundary_conditions=boundary_conditions,
)
```

The six faces are `left` (x=0), `right` (x=max), `front` (y=0), `back` (y=max), `top` (z=0), and `bottom` (z=max). Any face without an explicit condition defaults to no-flow. Any property name not present in the dictionary also defaults to all no-flow through the defaultdict factory.

### Multiple Property Boundaries

You can define separate boundary conditions for different properties. For example, pressure and temperature may have different boundary behavior:

```python
boundary_conditions = BoundaryConditions(
    conditions={
        "pressure": GridBoundaryCondition(
            left=ConstantBoundary(constant=2500.0),
            right=NoFlowBoundary(),
        ),
        "temperature": GridBoundaryCondition(
            left=ConstantBoundary(constant=180.0),
            right=NoFlowBoundary(),
        ),
    },
)
```

### Default Factory

You can provide a custom default factory that creates the `GridBoundaryCondition` used for any property not explicitly listed:

```python
boundary_conditions = BoundaryConditions(
    conditions={
        "pressure": GridBoundaryCondition(
            left=ConstantBoundary(constant=2500.0),
        ),
    },
    factory=lambda: GridBoundaryCondition(
        left=NoFlowBoundary(),
        right=NoFlowBoundary(),
        front=NoFlowBoundary(),
        back=NoFlowBoundary(),
    ),
)
```

### Saturation Normalization

After applying boundary conditions at each time step, BORES normalizes saturations to ensure $S_o + S_w + S_g = 1.0$ in every cell. Boundary conditions can inject or remove specific phases, which may temporarily violate the saturation sum constraint. The normalization step restores physical consistency using safe division to handle near-zero total saturations.

---

## Carter-Tracy Aquifer

For a realistic finite aquifer model, see the dedicated [Aquifers](aquifers.md) page, which covers the Carter-Tracy semi-analytical aquifer model in detail.
