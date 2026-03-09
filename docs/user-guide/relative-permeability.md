# Relative Permeability

## What is Relative Permeability?

When a single fluid flows through porous rock, the flow rate is governed by Darcy's law and depends on the absolute permeability $k$ of the rock. But real reservoirs contain multiple fluids - oil, water, and gas - competing for the same pore space. Each fluid does not have access to the full permeability of the rock. Instead, each phase flows through a fraction of the pore network, and the effective permeability available to each phase is lower than the absolute permeability. The ratio of a phase's effective permeability to the absolute permeability is called relative permeability.

Relative permeability $k_{r\alpha}$ is a dimensionless number between 0 and 1 that describes how easily phase $\alpha$ flows relative to the rock's full permeability. It depends primarily on the saturation of each phase present in the pore space. When a phase's saturation is below its residual value, that phase is immobile and its relative permeability is zero. As the saturation increases, the phase gains access to more connected pathways through the pore network and its relative permeability increases. At very high saturation, the phase occupies most of the pore space and its relative permeability approaches its maximum (endpoint) value, which is typically less than 1.0 because even at high saturation, some pore throats remain occupied by the other phases.

The mathematical form of relative permeability connects saturation to flow through the multiphase extension of Darcy's law:

$$q_\alpha = -\frac{k \cdot k_{r\alpha}(S_\alpha)}{\mu_\alpha} \nabla p_\alpha$$

where $q_\alpha$ is the volumetric flux of phase $\alpha$, $k$ is the absolute permeability, $k_{r\alpha}$ is the relative permeability of phase $\alpha$, $\mu_\alpha$ is the viscosity, and $\nabla p_\alpha$ is the pressure gradient. The ratio $k_{r\alpha} / \mu_\alpha$ is called the phase mobility $\lambda_\alpha$, and it is this quantity that controls how fast each phase moves relative to the others. The mobility ratio between displacing and displaced fluids is the single most important number governing sweep efficiency in any displacement process.

Relative permeability curves are among the most uncertain inputs to a reservoir simulation. They are measured in the laboratory using core flood experiments (steady-state or unsteady-state methods), but these measurements are expensive, time-consuming, and sensitive to experimental conditions. In practice, engineers often use analytical correlations (like Brooks-Corey) calibrated to a few measured endpoints, rather than full tabular curves. BORES supports both approaches.

---

## Saturation Endpoints

Before configuring any relative permeability model, you need to understand the four critical saturation endpoints that define the boundaries of the mobile saturation range for each phase. These endpoints have a direct physical meaning tied to the pore-scale distribution of fluids.

**Irreducible water saturation** ($S_{wc}$ or $S_{wi}$) is the minimum water saturation that can exist in the reservoir. This water is trapped in the smallest pores and in thin films coating the rock grains (in water-wet rock). It cannot be displaced by oil or gas injection. Typical values range from 0.15 to 0.35, with lower values in well-sorted, coarse-grained rocks and higher values in fine-grained or shaly formations.

**Residual oil saturation to waterflood** ($S_{or,w}$) is the oil left behind after water displaces oil. This oil is trapped as isolated ganglia in the pore network, held in place by capillary forces. Typical values range from 0.20 to 0.35 for water-wet sandstones. The residual oil saturation determines the maximum oil recovery achievable by waterflooding: $\text{RF}_{max} = 1 - S_{wc} - S_{or,w}$ as a fraction of original oil in place (assuming complete sweep).

**Residual oil saturation to gas flood** ($S_{or,g}$) is the oil left behind after gas displaces oil. Gas typically achieves lower residual oil than water because of its lower viscosity and different pore-scale displacement mechanisms. Typical values range from 0.10 to 0.25.

**Residual gas saturation** ($S_{gr}$) is the gas trapped after being displaced by liquid (oil or water). When rising oil or advancing water contacts a gas zone, some gas becomes trapped as disconnected bubbles. Typical values range from 0.03 to 0.10.

!!! info "Why Residual Oil Differs by Displacing Fluid"

    Water and gas trap oil through different mechanisms. Water (in a water-wet rock) tends to imbibe into smaller pores first, bypassing and snapping off oil in larger pores. Gas, being less viscous, creates more uniform displacement fronts at the pore scale. This is why $S_{or,g}$ is typically lower than $S_{or,w}$.

---

## The Brooks-Corey Model

The primary relative permeability model in BORES is `BrooksCoreyThreePhaseRelPermModel`. This is a power-law (Corey-type) model that computes two-phase relative permeability curves for each phase pair and then combines them into three-phase curves using a mixing rule.

The Brooks-Corey model expresses relative permeability as a power function of normalized saturation. For water in a water-oil system:

$$k_{rw} = k_{rw}^{max} \left(\frac{S_w - S_{wc}}{1 - S_{wc} - S_{or,w}}\right)^{n_w}$$

For oil in a water-oil system:

$$k_{ro,w} = k_{ro}^{max} \left(\frac{1 - S_w - S_{or,w}}{1 - S_{wc} - S_{or,w}}\right)^{n_o}$$

where $n_w$ and $n_o$ are the Corey exponents for water and oil respectively, and $k_{rw}^{max}$ and $k_{ro}^{max}$ are the endpoint relative permeabilities (both default to 1.0 in BORES).

### Basic Usage

```python
import bores

relperm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
)
```

The model computes relative permeability at every grid cell from the current saturation state. You do not need to supply saturation endpoints separately to the model if you already specified them in the `reservoir_model()` call. When the endpoints in the model are set to `None` (the default), BORES uses the endpoint grids from the reservoir model. When you provide explicit values to the rel perm model, those values override the grid-level endpoints as defaults.

### Parameter Reference

| Parameter | Default | Description |
| --- | --- | --- |
| `irreducible_water_saturation` | `None` | Connate water saturation $S_{wc}$. If `None`, uses grid values. |
| `residual_oil_saturation_water` | `None` | Residual oil to waterflood $S_{or,w}$. If `None`, uses grid values. |
| `residual_oil_saturation_gas` | `None` | Residual oil to gas flood $S_{or,g}$. If `None`, uses grid values. |
| `residual_gas_saturation` | `None` | Trapped gas saturation $S_{gr}$. If `None`, uses grid values. |
| `water_exponent` | `2.0` | Corey exponent for water phase $n_w$ |
| `oil_exponent` | `2.0` | Corey exponent for oil phase $n_o$ |
| `gas_exponent` | `2.0` | Corey exponent for gas phase $n_g$ |
| `wettability` | `WATER_WET` | Rock wettability (`WATER_WET` or `OIL_WET`) |
| `mixing_rule` | `eclipse_rule` | Three-phase oil relative permeability mixing rule |

---

## Corey Exponents

The Corey exponents ($n_w$, $n_o$, $n_g$) control the curvature of the relative permeability curves. They are the primary tuning parameters when matching laboratory data or calibrating simulation models to historical production data. Understanding what they control physically helps you choose appropriate values.

**Low exponents (1.0 - 1.5)** produce nearly linear curves. This means relative permeability increases rapidly with saturation, giving high mobility at moderate saturations. Low exponents are typical of fractured or vuggy carbonates where flow channels are large and well-connected.

**Medium exponents (2.0 - 3.0)** produce moderately curved relationships. An exponent of 2.0 is the most common default and is appropriate for many consolidated sandstones. This is a reasonable starting point when no laboratory data is available.

**High exponents (3.0 - 6.0)** produce strongly concave curves where relative permeability stays low until saturation is quite high. These are typical of fine-grained rocks, mixed-wet systems, or situations where pore-scale heterogeneity creates tortuous flow paths.

The practical effect on simulation results is significant. Higher water exponents delay water breakthrough because water relative permeability remains low until high water saturation develops. Higher oil exponents cause oil production to decline more gradually. The ratio of exponents affects the fractional flow curve and therefore the shape and speed of displacement fronts.

!!! tip "Choosing Exponents Without Lab Data"

    When you lack laboratory relative permeability measurements, start with exponents of 2.0 for all phases. If your waterflood history match shows water breaking through too early, increase the water exponent. If oil production declines too fast after breakthrough, increase the oil exponent. These adjustments change the fractional flow behavior without affecting the endpoint saturations.

---

## Wettability

Wettability describes which fluid preferentially coats the rock surface and occupies the smaller pores. It fundamentally changes the shape of relative permeability curves and has a major impact on recovery efficiency.

In a **water-wet** rock (the default in BORES), water preferentially occupies the smaller pores and forms a continuous film along grain surfaces. Oil sits in the larger pores. During waterflooding, water advances through the smaller pores first (imbibition), efficiently displacing oil from the pore network. Water-wet rocks typically have:

- Lower $S_{or,w}$ (better waterflood recovery)
- Higher $k_{rw}$ at residual oil (water flows easily in its preferred small pores)
- Crossover point at $S_w > 0.5$ (water curve crosses oil curve at higher water saturation)

In an **oil-wet** rock, oil preferentially coats the grain surfaces and occupies the smaller pores. Water sits in the larger, less well-connected pores. During waterflooding, water tends to channel through larger pores, bypassing oil in smaller ones. Oil-wet rocks typically have:

- Higher $S_{or,w}$ (worse waterflood recovery)
- Lower $k_{rw}$ at residual oil
- Crossover point at $S_w < 0.5$

```python
import bores

# Water-wet system (default)
relperm_ww = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.30,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.5,
    oil_exponent=2.0,
    gas_exponent=2.0,
    wettability=bores.Wettability.WATER_WET,
)

# Oil-wet system
relperm_ow = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.35,
    residual_oil_saturation_gas=0.20,
    residual_gas_saturation=0.05,
    water_exponent=3.0,
    oil_exponent=1.5,
    gas_exponent=2.0,
    wettability=bores.Wettability.OIL_WET,
)
```

Most conventional sandstone reservoirs are water-wet or mixed-wet. Carbonate reservoirs are more commonly oil-wet or mixed-wet. If you are unsure, start with water-wet, which is the industry default.

### Mixed-Wet Systems

Many real reservoirs exhibit mixed wettability, where some pore surfaces are water-wet and others are oil-wet. This occurs naturally when crude oil contacts the rock surface over geological time: the larger pores that were originally oil-filled become oil-wet, while smaller pores that retained water films remain water-wet. Mixed-wet rock typically produces distinctive relative permeability curves that fall between the pure water-wet and oil-wet end members, often with relatively high mobility for both phases at intermediate saturations.

The `BrooksCoreyThreePhaseRelPermModel` in BORES currently supports `WATER_WET` and `OIL_WET` wettability settings. It does not have a direct `MIXED_WET` mode for relative permeability. However, you can approximate mixed-wet relative permeability behavior through careful selection of Corey exponents. Mixed-wet systems typically have characteristics that lie between the two end members:

- **Lower water exponents** (1.5 to 2.5) compared to strongly water-wet systems, because oil-wet pores provide easier pathways for water
- **Lower oil exponents** (1.5 to 2.0) compared to strongly oil-wet systems, because water-wet pores keep some oil mobile
- **Crossover point near $S_w = 0.5$**, between the high crossover of water-wet and low crossover of oil-wet

```python
import bores

# Approximate mixed-wet behavior using tuned exponents
relperm_mixed = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.20,
    residual_oil_saturation_water=0.28,
    residual_oil_saturation_gas=0.18,
    residual_gas_saturation=0.05,
    water_exponent=2.0,     # Lower than typical water-wet (2.5-3.0)
    oil_exponent=1.8,       # Lower than typical oil-wet (2.0-2.5)
    gas_exponent=2.0,
    wettability=bores.Wettability.WATER_WET,
)
```

If you have laboratory SCAL data from mixed-wet core plugs, the best approach is to use a `ThreePhaseRelPermTable` with the measured data directly. Tabular data can reproduce the exact curve shapes measured in the lab, including the subtle features characteristic of mixed-wet rock (gradual crossover, relatively high endpoint permeabilities for both phases, and S-shaped curve segments) that analytical models cannot easily capture.

!!! tip "Mixed-Wet Capillary Pressure"

    While the relative permeability model approximates mixed-wet behavior through exponent tuning, the capillary pressure models in BORES do support a direct `MIXED_WET` wettability mode with a `mixed_wet_water_fraction` parameter. See [Capillary Pressure](capillary-pressure.md) for details on configuring mixed-wet capillary pressure curves, which can produce the characteristic sign change at intermediate saturations.

---

## Three-Phase Mixing Rules

In a three-phase system (oil, water, and gas present simultaneously), BORES needs a way to compute the oil relative permeability $k_{ro}$ from the two-phase curves $k_{ro,w}(S_w)$ and $k_{ro,g}(S_g)$. This is done through a mixing rule that interpolates between the two-phase oil curves.

BORES provides several mixing rules, selectable by name or by passing the function directly. The choice of mixing rule can significantly affect results in regions where all three phases are mobile (near wells, at gas-oil-water contacts, and during WAG injection).

### Available Mixing Rules

| Rule | String Name | Description |
| --- | --- | --- |
| Eclipse rule | `"eclipse_rule"` | Industry standard default. Conservative, well-tested. |
| Stone I | `"stone_I_rule"` | Stone's first model. Good for water-wet systems. |
| Stone II | `"stone_II_rule"` | Stone's second model. More conservative than Stone I. |
| Baker linear | `"baker_linear_rule"` | Linear saturation-weighted interpolation. Simple and stable. |
| Saturation weighted | `"saturation_weighted_interpolation_rule"` | Weighted by normalized saturations. |
| Blunt rule | `"blunt_rule"` | Conservative, designed for strongly water-wet rocks. |
| Harmonic mean | `"harmonic_mean_rule"` | Very conservative. Good for tight rocks, series flow. |
| Geometric mean | `"geometric_mean_rule"` | Moderately conservative. General purpose. |
| Arithmetic mean | `"arithmetic_mean_rule"` | Optimistic upper bound estimate. |
| Min rule | `"min_rule"` | Most conservative. Lower bound on oil mobility. |
| Max rule | `"max_rule"` | Most optimistic. Upper bound on oil mobility. |
| Aziz-Settari | `"aziz_settari_rule"` | Empirical, tunable for specific reservoirs. |

```python
import bores

# Using the default Eclipse rule
relperm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.30,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule="eclipse_rule",
)

# Using Stone I for a water-wet carbonate
relperm_stone = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.30,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.5,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule="stone_I_rule",
)
```

The Eclipse rule is the default and recommended starting point. It is the same approach used by the Schlumberger Eclipse commercial simulator and has been validated against decades of field data. Stone I and Stone II are classical alternatives from the petroleum engineering literature and are appropriate when you need to match specific laboratory three-phase relative permeability data.

!!! warning "Mixing Rule Sensitivity"

    The choice of mixing rule primarily matters in regions where all three phases are simultaneously mobile. In a pure waterflood (no free gas) or pure gas injection (connate water only), the mixing rule has no effect because one of the two-phase curves is not active. However, in WAG injection, gas cap expansion into the oil zone, or near the gas-oil-water contact, the mixing rule can change oil recovery predictions by 5-15%.

---

## Tabular Relative Permeability

While the Brooks-Corey model is convenient because it generates smooth curves from a small number of parameters, many engineers prefer to use tabular data from laboratory core flood experiments. BORES provides `TwoPhaseRelPermTable` and `ThreePhaseRelPermTable` for this purpose. These classes store measured saturation and relative permeability values as arrays and interpolate between them using fast linear interpolation (`np.interp`).

Tabular relative permeability is the industry-standard approach when you have Special Core Analysis (SCAL) data from laboratory steady-state or unsteady-state experiments. In these experiments, two fluids are injected simultaneously through a core plug at different fractional flow rates, and the pressure drop and produced fluid volumes are used to back-calculate relative permeability at each saturation. The result is a table of saturation values paired with the corresponding relative permeability for each phase. Using these measured data directly avoids the approximation inherent in fitting an analytical model.

### `TwoPhaseRelPermTable`

A `TwoPhaseRelPermTable` stores one set of saturation values and the corresponding relative permeabilities for both phases. You specify which fluid is the wetting phase and which is the non-wetting phase, along with the saturation and permeability arrays.

```python
import bores
import numpy as np

# Oil-water relative permeability from lab data
ow_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=np.array([0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]),
    wetting_phase_relative_permeability=np.array([0.0, 0.01, 0.03, 0.10, 0.22, 0.40, 0.65, 0.80]),
    non_wetting_phase_relative_permeability=np.array([1.0, 0.85, 0.68, 0.40, 0.20, 0.08, 0.01, 0.0]),
)
```

The `wetting_phase_saturation` array must be monotonically increasing and contain at least two points. BORES uses `np.interp` for interpolation, which means values outside the table range are clamped to the endpoint values (constant extrapolation). This is physically reasonable because relative permeability should be zero at or below residual saturation and at its maximum at or above the maximum saturation.

You can query the table for relative permeability values at any saturation, including full 3D grid arrays:

```python
# Query at a single saturation
krw_at_05 = ow_table.get_wetting_phase_relative_permeability(0.5)
kro_at_05 = ow_table.get_non_wetting_phase_relative_permeability(0.5)

# Query with a grid array
Sw_grid = np.random.uniform(0.2, 0.75, size=(20, 20, 5))
krw_grid = ow_table.get_wetting_phase_relative_permeability(Sw_grid)
kro_grid = ow_table.get_non_wetting_phase_relative_permeability(Sw_grid)
```

### `ThreePhaseRelPermTable`

For three-phase simulation, you combine two `TwoPhaseRelPermTable` objects (one for oil-water and one for gas-oil) into a `ThreePhaseRelPermTable`. This table uses the same mixing rules as the Brooks-Corey model to compute three-phase oil relative permeability from the two two-phase curves.

```python
import bores
import numpy as np

# Oil-water table (water is wetting phase)
ow_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]),
    wetting_phase_relative_permeability=np.array([0.0, 0.02, 0.08, 0.20, 0.38, 0.62, 0.80]),
    non_wetting_phase_relative_permeability=np.array([1.0, 0.70, 0.42, 0.20, 0.07, 0.01, 0.0]),
)

# Gas-oil table (oil is wetting phase)
go_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=np.array([0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.85]),
    wetting_phase_relative_permeability=np.array([0.0, 0.02, 0.08, 0.25, 0.50, 0.82, 1.0]),
    non_wetting_phase_relative_permeability=np.array([0.90, 0.65, 0.40, 0.18, 0.05, 0.005, 0.0]),
)

# Combine into three-phase table with Eclipse mixing rule
three_phase = bores.ThreePhaseRelPermTable(
    oil_water_table=ow_table,
    gas_oil_table=go_table,
    mixing_rule=bores.eclipse_rule,
)
```

The `ThreePhaseRelPermTable` validates that the phase assignments are consistent: the oil-water table must involve water and oil, the gas-oil table must involve oil and gas, and the gas-oil table must have oil as the wetting phase. If these constraints are violated, BORES raises a `ValidationError` with a clear message explaining the issue.

The mixing rule parameter accepts the same functions as the Brooks-Corey model. If you set `mixing_rule=None`, BORES defaults to a conservative `min(kro_w, kro_g)` rule. The available mixing rules are the same ones listed in the [Three-Phase Mixing Rules](#three-phase-mixing-rules) section above.

!!! tip "When to Use Tables vs. Correlations"

    Use tabular relative permeability when you have laboratory SCAL data, when you need to match specific curve shapes that power-law models cannot reproduce (such as S-shaped curves or curves with inflection points), or when your curves come from pore-network modeling or digital rock analysis. Use the Brooks-Corey model when you lack lab data, when you want quick sensitivity studies by varying exponents, or when you need a smooth, well-behaved function for numerical stability.

---

## Direct Usage (Outside Simulation)

You do not need to run a full simulation to evaluate relative permeability models. Both analytical and tabular models can be called directly with saturation values, which is useful for plotting curves, debugging, validating against lab data, or building custom workflows. Every model supports both scalar and grid-array inputs.

### Calling the Brooks-Corey Model Directly

The `BrooksCoreyThreePhaseRelPermModel` can be called directly with `get_relative_permeabilities()` or using the `__call__` interface. Both accept water, oil, and gas saturations and return a dictionary with `"water"`, `"oil"`, and `"gas"` keys containing the computed relative permeabilities.

```python
import bores
import numpy as np

relperm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.5,
    oil_exponent=2.0,
    gas_exponent=2.0,
)

# Scalar evaluation
result = relperm.get_relative_permeabilities(
    water_saturation=0.4,
    oil_saturation=0.55,
    gas_saturation=0.05,
)
print(f"krw = {result['water']:.4f}")
print(f"kro = {result['oil']:.4f}")
print(f"krg = {result['gas']:.4f}")

# Using __call__ (same result)
result = relperm(
    water_saturation=0.4,
    oil_saturation=0.55,
    gas_saturation=0.05,
)

# Grid array evaluation (3D)
Sw = bores.build_uniform_grid((20, 20, 5), value=0.4)
So = bores.build_uniform_grid((20, 20, 5), value=0.55)
Sg = bores.build_uniform_grid((20, 20, 5), value=0.05)

result = relperm(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)
krw_grid = result["water"]  # Shape: (20, 20, 5)
kro_grid = result["oil"]    # Shape: (20, 20, 5)
krg_grid = result["gas"]    # Shape: (20, 20, 5)
```

The returned dictionary uses string keys `"water"`, `"oil"`, and `"gas"`. Each value is either a float (for scalar inputs) or a NumPy array matching the shape of the input saturations (for array inputs). This makes it easy to sweep over saturations for plotting or to evaluate the model on a full simulation grid.

### Calling the `ThreePhaseRelPermTable` Directly

The tabular model works the same way. Use `get_relative_permeabilities()` or `__call__` with the three saturation values:

```python
# Using the three_phase table from earlier
result = three_phase.get_relative_permeabilities(
    water_saturation=0.35,
    oil_saturation=0.55,
    gas_saturation=0.10,
)
print(f"krw = {result['water']:.4f}")
print(f"kro = {result['oil']:.4f}")
print(f"krg = {result['gas']:.4f}")

# Array evaluation works identically
result = three_phase(
    water_saturation=Sw,
    oil_saturation=So,
    gas_saturation=Sg,
)
```

### Calling a `TwoPhaseRelPermTable` Directly

For the two-phase table, you query by wetting phase saturation using the dedicated methods:

```python
# Get individual phase relative permeabilities
krw = ow_table.get_wetting_phase_relative_permeability(0.45)
kro = ow_table.get_non_wetting_phase_relative_permeability(0.45)

# Get both at once
krw, kro = ow_table.get_relative_permeabilities(0.45)

# Using __call__ (returns wetting phase kr only)
krw = ow_table(wetting_phase_saturation=0.45)
```

This direct evaluation capability is valuable for generating relative permeability curves for reports, comparing analytical and tabular models side by side, and verifying that your model parameters produce physically reasonable curves before committing to a full simulation run.

---

## Integrating with `RockFluidTables`

Relative permeability is passed to the simulation through the `RockFluidTables` object, which also holds the capillary pressure model. You then pass `RockFluidTables` to the `Config`. This works the same way whether you use the Brooks-Corey analytical model or a tabular approach:

```python
import bores

relperm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.5,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule="eclipse_rule",
)

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=relperm,
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(),
)

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    scheme="impes",
)
```

You can also pass a `ThreePhaseRelPermTable` instead of the Brooks-Corey model:

```python
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=three_phase,  # ThreePhaseRelPermTable from lab data
    capillary_pressure_table=bores.BrooksCoreyCapillaryPressureModel(),
)
```

Both analytical models and tabular data are `RockFluidTables`-compatible and serializable. When you save a `Config` to disk, the relative permeability model (whether Brooks-Corey or tabular) and all its parameters or data are preserved and can be reloaded exactly.

---

## Visualizing Relative Permeability Curves

Understanding what your relative permeability curves look like before running a simulation is critical for quality assurance. The best way to do this is to create the actual model you plan to use and call it directly across a saturation sweep. This ensures the plotted curves are exactly what the simulator will use.

### Water-Oil Curves

```python
import bores
import numpy as np

# Create the model
relperm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.5,
    oil_exponent=2.0,
    gas_exponent=2.0,
)

# Sweep water saturation across the mobile range (no free gas)
Sw_values = np.linspace(0.25, 0.75, 50)
krw_values = np.zeros_like(Sw_values)
kro_values = np.zeros_like(Sw_values)

for i, sw in enumerate(Sw_values):
    so = 1.0 - sw  # No free gas: So = 1 - Sw
    result = relperm.get_relative_permeabilities(
        water_saturation=sw, oil_saturation=so, gas_saturation=0.0,
    )
    krw_values[i] = result["water"]
    kro_values[i] = result["oil"]

# Plot with BORES visualization
fig = bores.make_series_plot(
    data={
        "krw": np.column_stack([Sw_values, krw_values]),
        "kro": np.column_stack([Sw_values, kro_values]),
    },
    title="Brooks-Corey Water-Oil Relative Permeability",
    x_label="Water Saturation (fraction)",
    y_label="Relative Permeability",
)
fig.show()
# Output: [PLACEHOLDER: Insert relperm_water_oil_curves.png]
```

This visualization lets you verify that the curves match your expectations: endpoint values are correct, curvature is reasonable, and the crossover point is where you expect it. If the curves look wrong, adjust the exponents and endpoints before running the simulation.

### Gas-Oil Curves

You can also plot the gas-oil relative permeability curves by sweeping gas saturation while keeping water at connate:

```python
# Sweep gas saturation (water at connate)
Sg_values = np.linspace(0.0, 0.55, 50)
krg_values = np.zeros_like(Sg_values)
kro_g_values = np.zeros_like(Sg_values)

for i, sg in enumerate(Sg_values):
    so = 1.0 - 0.25 - sg  # Sw = Swc = 0.25
    result = relperm.get_relative_permeabilities(
        water_saturation=0.25, oil_saturation=so, gas_saturation=sg,
    )
    krg_values[i] = result["gas"]
    kro_g_values[i] = result["oil"]

fig = bores.make_series_plot(
    data={
        "krg": np.column_stack([Sg_values, krg_values]),
        "kro": np.column_stack([Sg_values, kro_g_values]),
    },
    title="Brooks-Corey Gas-Oil Relative Permeability",
    x_label="Gas Saturation (fraction)",
    y_label="Relative Permeability",
)
fig.show()
# Output: [PLACEHOLDER: Insert relperm_gas_oil_curves.png]
```

### Comparing Tabular and Analytical Curves

If you have both a tabular model from lab data and an analytical model, you can plot them together to assess how well the analytical fit matches the measurements:

```python
# Analytical model
relperm_bc = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.20,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.5,
    oil_exponent=2.0,
)

# Tabular model from lab data
ow_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]),
    wetting_phase_relative_permeability=np.array([0.0, 0.02, 0.08, 0.20, 0.38, 0.62, 0.80]),
    non_wetting_phase_relative_permeability=np.array([1.0, 0.70, 0.42, 0.20, 0.07, 0.01, 0.0]),
)

# Evaluate both across the same saturation range
Sw_range = np.linspace(0.20, 0.75, 50)
krw_bc = np.zeros_like(Sw_range)
kro_bc = np.zeros_like(Sw_range)

for i, sw in enumerate(Sw_range):
    result = relperm_bc.get_relative_permeabilities(
        water_saturation=sw, oil_saturation=1.0 - sw, gas_saturation=0.0,
    )
    krw_bc[i] = result["water"]
    kro_bc[i] = result["oil"]

krw_tab = ow_table.get_wetting_phase_relative_permeability(Sw_range)
kro_tab = ow_table.get_non_wetting_phase_relative_permeability(Sw_range)

fig = bores.make_series_plot(
    data={
        "krw (Brooks-Corey)": np.column_stack([Sw_range, krw_bc]),
        "kro (Brooks-Corey)": np.column_stack([Sw_range, kro_bc]),
        "krw (Lab Data)": np.column_stack([Sw_range, krw_tab]),
        "kro (Lab Data)": np.column_stack([Sw_range, kro_tab]),
    },
    title="Analytical vs. Tabular Relative Permeability",
    x_label="Water Saturation (fraction)",
    y_label="Relative Permeability",
)
fig.show()
# Output: [PLACEHOLDER: Insert relperm_comparison.png]
```

This kind of comparison is essential for calibration. If the analytical and tabular curves diverge significantly, you should either adjust the Corey exponents to improve the fit or use the tabular data directly for the simulation.

---

## Tips for Selecting Parameters

Choosing relative permeability parameters is one of the most important decisions in reservoir simulation. Here are practical guidelines based on field experience:

**Start with the endpoints.** Irreducible water saturation and residual oil saturation have a much larger impact on recovery predictions than the exponents. Get these right first from core analysis or analogous field data.

**Use exponents of 2.0 as a baseline.** Adjust up or down based on history matching or laboratory data. Higher exponents mean more non-linear behavior and generally more pessimistic displacement efficiency.

**Match the fractional flow curve, not just the relative permeability.** The fractional flow function $f_w = \lambda_w / (\lambda_w + \lambda_o)$ controls the displacement front shape and water cut evolution. You can have very different relative permeability curves that produce similar fractional flow behavior if the mobility ratio is similar.

**Be consistent with wettability.** If you set oil-wet wettability, your endpoint saturations should also reflect oil-wet behavior (higher $S_{or,w}$, lower crossover point). Mixing water-wet endpoints with oil-wet curve shapes produces physically inconsistent models.

!!! example "Quick Reference: Typical Parameter Sets"

    === "Water-Wet Sandstone"

        ```python
        relperm = bores.BrooksCoreyThreePhaseRelPermModel(
            irreducible_water_saturation=0.25,
            residual_oil_saturation_water=0.25,
            residual_oil_saturation_gas=0.15,
            residual_gas_saturation=0.05,
            water_exponent=2.5,
            oil_exponent=2.0,
            gas_exponent=2.0,
            wettability=bores.Wettability.WATER_WET,
        )
        ```

    === "Oil-Wet Carbonate"

        ```python
        relperm = bores.BrooksCoreyThreePhaseRelPermModel(
            irreducible_water_saturation=0.15,
            residual_oil_saturation_water=0.35,
            residual_oil_saturation_gas=0.20,
            residual_gas_saturation=0.05,
            water_exponent=3.5,
            oil_exponent=1.5,
            gas_exponent=2.5,
            wettability=bores.Wettability.OIL_WET,
        )
        ```

    === "Unconsolidated Sand"

        ```python
        relperm = bores.BrooksCoreyThreePhaseRelPermModel(
            irreducible_water_saturation=0.30,
            residual_oil_saturation_water=0.20,
            residual_oil_saturation_gas=0.12,
            residual_gas_saturation=0.03,
            water_exponent=1.5,
            oil_exponent=1.5,
            gas_exponent=1.5,
        )
        ```
