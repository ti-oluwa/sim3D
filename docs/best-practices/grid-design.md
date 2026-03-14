# Grid Design

## Overview

The computational grid is the foundation of every reservoir simulation. It defines how finely you resolve spatial variations in pressure, saturation, and fluid properties, and it directly controls both the accuracy of your results and the computational cost of running the simulation. Choosing the right grid is one of the most important decisions you will make when building a model, and getting it wrong can lead to anything from artificially smeared saturation fronts to prohibitively long run times.

Grid design in reservoir simulation is fundamentally a trade-off between resolution and performance. Finer grids capture more physical detail but require more memory, more computation per timestep, and smaller timesteps to maintain numerical stability. Coarser grids run faster but can miss important features like thin barriers, localized coning, or sharp displacement fronts. The art of grid design is finding the resolution that captures the physics you care about without paying for detail you do not need.

BORES uses structured Cartesian grids with uniform or layered cell dimensions. While this is simpler than unstructured or corner-point grids used in some commercial simulators, it covers the vast majority of simulation workflows and makes it straightforward to reason about cell sizes, aspect ratios, and refinement strategies. The grid construction functions in BORES (`build_uniform_grid`, `build_layered_grid`, `build_depth_grid`, `build_saturation_grids`) are designed to make it easy to experiment with different resolutions without rewriting your model setup.

---

## Cell Size Selection

The most fundamental grid design decision is how large to make your cells. The right cell size depends on what physical processes you are trying to resolve and how quickly you need results.

### Areal Dimensions (dx, dy)

For areal (horizontal) cell sizes, the key consideration is the scale of the features you need to resolve. Wells, faults, and high-permeability channels all create localized flow patterns that require smaller cells to capture accurately. Far from these features, the flow field is smoother and coarser cells are adequate.

As a starting point, consider these ranges for different applications:

| Application | Typical Cell Size (ft) | Notes |
| --- | --- | --- |
| Near-wellbore studies | 10 - 50 | Captures coning, cusping, and radial flow |
| Pattern floods (quarter five-spot) | 50 - 200 | Resolves sweep patterns and breakthrough |
| Full-field sector models | 100 - 500 | Balance between detail and field-scale coverage |
| Screening studies | 500 - 1000 | Quick results for sensitivity analysis |

The Peaceman well model used in BORES (and all finite-difference simulators) assumes that the well is small relative to the grid cell. If your cells are smaller than roughly 5 times the wellbore radius, the well model assumptions start to break down. In practice, this is rarely a concern because wellbore radii are typically 0.25 to 0.5 ft, making the minimum practical cell size around 2 to 5 ft.

### Vertical Dimension (dz)

Vertical resolution is often more critical than horizontal resolution, especially in reservoirs with gravity-driven flow. Gravity segregation, gas coning, and water coning all require adequate vertical resolution to capture correctly. As a general rule, you need more vertical layers than you might initially expect.

For reservoirs with significant gravity effects (gas caps, bottom water, or gravity-stable displacement), aim for at least 10 to 20 layers in the pay zone. If you are modeling gas or water coning into a well, you may need 30 or more layers near the well perforation interval. For simple depletion studies without strong gravity effects, 3 to 5 layers may be sufficient.

The `build_layered_grid` function lets you assign different thickness values to each layer, so you can use thin layers near fluid contacts and thicker layers elsewhere:

```python
import bores

# 10 layers: thin near contacts, thick in the middle
layer_thicknesses = bores.build_layered_grid(
    grid_shape=(20, 20, 10),
    values=[5.0, 5.0, 10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 5.0, 5.0],
)
```

This creates a 10-layer grid with 5 ft layers at the top and bottom (near the gas-oil and oil-water contacts) and 20 ft layers in the middle of the oil column.

---

## Aspect Ratio

The aspect ratio of a grid cell is the ratio of its largest dimension to its smallest dimension. Extreme aspect ratios (very flat or very tall cells) can cause numerical problems because flow calculations become ill-conditioned when cells are much longer in one direction than another.

For explicit and IMPES simulations, keep the aspect ratio below 10:1. For fully implicit simulations, you can push to 20:1 or higher, but the linear solver will work harder and may need a stronger preconditioner.

Common problematic scenarios and how to handle them:

**Very thin reservoir, wide cells.** If your pay zone is 10 ft thick but your cells are 500 ft wide, the vertical aspect ratio is 50:1. This is too extreme. Either increase the number of vertical layers (splitting the 10 ft into 5 layers of 2 ft each gives a 250:1 ratio, which is worse) or reduce the horizontal cell size. A better approach for thin reservoirs is to use 50 to 100 ft horizontal cells with 2 to 5 ft vertical cells, giving a 20:1 to 50:1 ratio.

**Tall cells near wells.** If you refine vertically near a well perforation, make sure the horizontal cells are not too large relative to the refined vertical cells. A 2 ft vertical cell in a 500 ft horizontal cell creates a 250:1 ratio that will cause stability issues.

The general rule is to keep all three dimensions within an order of magnitude of each other when possible, and never let the ratio exceed about 20:1 without testing that your solver and timestep handle it well.

---

## Grid Sizing for Different Problems

Different simulation scenarios call for different grid design strategies. Here are recommendations for common workflows.

### Primary Depletion

For simple pressure depletion without water or gas injection, coarse grids are usually adequate. The pressure field varies smoothly across the reservoir, and saturation changes (if any) are driven by solution gas liberation, which is a bulk process rather than a sharp-front displacement. A 10x10x5 to 30x30x10 grid is often sufficient for engineering purposes.

### Waterflood

Waterfloods develop sharp saturation fronts that can span only a few cells. If your cells are too large, the front will appear smeared across many cells, which underestimates sweep efficiency and delays predicted breakthrough. For quarter five-spot patterns, a 20x20 to 50x50 areal grid captures the displacement front reasonably well. Use at least 5 vertical layers to capture gravity segregation.

!!! tip "Grid Orientation Effects"

    In a Cartesian grid, flow along the grid axes (x and y) and flow along the diagonals are treated differently due to the stencil geometry. This can cause artificial preferential flow along one direction in a five-spot pattern. Using a finer grid reduces this effect. If grid orientation is a concern, try rotating your well pattern 45 degrees relative to the grid and compare results.

### Gas Injection and Miscible Flooding

Gas injection simulations are more demanding than waterfloods because gas is much more mobile than oil, leading to sharper displacement fronts and stronger gravity segregation. Miscible floods add concentration transport that requires adequate resolution to avoid excessive numerical diffusion. Use at least 20x20x10 grids for gas injection, and consider 30x30x20 or finer for miscible flooding studies where mixing zone resolution matters.

### Coning Studies

Water or gas coning near wells requires fine vertical resolution around the perforation interval. A common approach is to use a 1D or 2D radial-equivalent model with 30 to 50 vertical layers concentrated around the perforation depth. In 3D Cartesian grids, you can approximate this by using locally refined vertical layers near the well.

---

## Coarsening and Refinement

BORES provides the `coarsen_grid` function to reduce grid resolution by averaging cell values, and `coarsen_permeability_grids` for permeability (which uses harmonic averaging to preserve flow behavior). These are useful for testing whether your grid is fine enough: run your simulation at your chosen resolution, coarsen by a factor of 2, run again, and compare. If the results are nearly identical, your original resolution is adequate. If they differ significantly, you need a finer grid.

```python
import bores

# Original fine grid: shape (40, 40, 20)
pressure_fine = bores.build_uniform_grid((40, 40, 20), 3000.0)

# Coarsen by 2x in each direction: (40,40,20) -> (20,20,10)
pressure_coarse = bores.coarsen_grid(pressure_fine, batch_size=(2, 2, 2), method="mean")

# For permeability, use direction-appropriate averaging
# coarsen_permeability_grids takes all directional grids and applies
# harmonic mean in the flow direction, arithmetic mean perpendicular
kx_coarse, ky_coarse, kz_coarse = bores.coarsen_permeability_grids(
    kx_fine, ky_fine, kz_fine, batch_size=(2, 2, 2)
)

# 2D case (no kz)
kx_coarse_2d, ky_coarse_2d = bores.coarsen_permeability_grids(
    kx_fine_2d, ky_fine_2d, batch_size=(2, 2)
)
```

!!! warning "Permeability Coarsening"

    Never coarsen permeability using arithmetic averaging (`bores.coarsen_grid` with `method="mean"`). Arithmetic averaging of permeability overpredicts flow through heterogeneous media. Always use `coarsen_permeability_grids`, which applies harmonic averaging in the flow direction and arithmetic averaging perpendicular to it. This is the physically correct approach for series-flow configurations.

---

## Practical Guidelines Summary

Here is a quick reference for grid design decisions:

1. **Start coarse, refine as needed.** Begin with a 10x10x5 grid to get your model running, then increase resolution and check that results converge. This is faster than starting fine and trying to figure out why the simulation is slow.

2. **Prioritize vertical resolution.** In most reservoir problems, vertical resolution matters more than horizontal resolution because of gravity effects and permeability contrasts between layers.

3. **Keep aspect ratios reasonable.** Stay below 10:1 for IMPES, below 20:1 for implicit. Test if you need to go higher.

4. **Match cell size to the physics.** Sharp fronts (waterfloods, gas injection) need finer grids than smooth processes (primary depletion).

5. **Use layered grids for heterogeneity.** The `build_layered_grid` function lets you specify per-layer values, which is more efficient than uniform fine grids for vertically heterogeneous reservoirs.

6. **Validate with grid refinement.** Always run a coarsened version and compare. If results differ by more than 5 to 10%, refine further.

7. **Remember the CFL constraint.** Finer grids require smaller timesteps in explicit and IMPES schemes. A grid that is 2x finer in each direction is 8x more cells and typically requires 2x smaller timesteps, for a total of 16x more computation. See the [Timestep Selection](timestep-selection.md) page for details.
