# Fluid Properties

## Overview

Fluid properties describe how oil, gas, and water behave at reservoir conditions. In petroleum engineering, these are called PVT (Pressure-Volume-Temperature) properties because they depend primarily on pressure and temperature. BORES computes most PVT properties automatically from a few key inputs using industry-standard correlations.

You do not need to supply every fluid property manually. When you call `bores.reservoir_model()`, the factory estimates formation volume factors, solution gas-oil ratio, gas compressibility factor, densities, and other derived properties from the inputs you provide: pressure, temperature, oil specific gravity (or API gravity), and gas specific gravity. This design lets you build a complete model from commonly available field data without needing a full PVT laboratory report.

If you do have laboratory PVT data or equation-of-state results, you can provide them as PVT tables instead of relying on correlations. See [PVT Tables](advanced/pvt-tables.md) for that workflow.

---

## Oil Properties

Oil properties are the most complex because they depend on whether gas is dissolved in the oil (saturated) or all gas is in solution (undersaturated).

### Key Oil Inputs

| Parameter | Units | Typical Range | Description |
|---|---|---|---|
| `oil_specific_gravity_grid` | dimensionless | 0.75 - 0.95 | Oil specific gravity relative to water |
| `oil_viscosity_grid` | cP | 0.5 - 100+ | Dead oil viscosity at reservoir temperature |
| `oil_bubble_point_pressure_grid` | psi | 500 - 5000 | Pressure below which gas liberates from oil |

```python
import bores

grid_shape = (20, 20, 5)

oil_sg = bores.build_uniform_grid(grid_shape, value=0.85)        # ~35 API
oil_visc = bores.build_uniform_grid(grid_shape, value=1.5)       # cP
bubble_pt = bores.build_uniform_grid(grid_shape, value=2500.0)   # psi
```

The oil specific gravity of 0.85 corresponds to about 35 degrees API, which is a light crude oil. API gravity and specific gravity are related by:

$$\text{API} = \frac{141.5}{\gamma_o} - 131.5$$

### Properties Computed by the Factory

From your oil inputs plus pressure and temperature, `bores.reservoir_model()` computes:

- **Solution gas-oil ratio** ($R_s$): Volume of gas dissolved per volume of oil at surface conditions (SCF/STB). Uses the Standing correlation.
- **Oil formation volume factor** ($B_o$): Ratio of oil volume at reservoir conditions to surface conditions (RB/STB). Accounts for dissolved gas expansion. Uses Standing or Vasquez-Beggs.
- **Oil compressibility** ($c_o$): Isothermal compressibility of oil (psi$^{-1}$).
- **Oil density**: Computed from specific gravity, $B_o$, and $R_s$.
- **Live oil viscosity**: Adjusted from dead oil viscosity for dissolved gas using the Beggs-Robinson correlation.

---

## Gas Properties

Gas behavior is described by the real gas law, where the compressibility factor $Z$ accounts for deviations from ideal gas behavior.

### Key Gas Inputs

| Parameter | Units | Typical Range | Description |
|---|---|---|---|
| `gas_specific_gravity` | dimensionless | 0.55 - 1.5 | Gas gravity relative to air (methane ~ 0.55) |

```python
model = bores.reservoir_model(
    # ... other parameters ...
    gas_specific_gravity=0.65,  # Light hydrocarbon gas
)
```

### Properties Computed by the Factory

- **Gas compressibility factor** ($Z$): Hall-Yarborough or Dranchuk-Abou-Kazeem correlation.
- **Gas formation volume factor** ($B_g$): ft³/SCF. Computed from $Z$, $p$, and $T$.
- **Gas viscosity** ($\mu_g$): Lee-Gonzalez-Eakin correlation.
- **Gas density**: Computed from molecular weight, $Z$, $p$, and $T$.

!!! info "Gas Gravity Values"

    Common gas specific gravities:

    - Methane (CH4): 0.553
    - Natural gas (typical): 0.60 - 0.75
    - CO2: 1.52
    - Nitrogen (N2): 0.967

---

## Water Properties

Water is typically the simplest phase. BORES computes water properties from the water specific gravity and reservoir conditions.

```python
model = bores.reservoir_model(
    # ... other parameters ...
    water_specific_gravity=1.02,  # Slightly saline formation water
)
```

BORES computes water formation volume factor ($B_w$), water compressibility ($c_w$), water viscosity ($\mu_w$), and water density from the specific gravity and reservoir conditions.

!!! tip "Default Water Properties"

    If you do not specify `water_specific_gravity`, BORES defaults to 1.0 (fresh water). For saline formation water, typical values range from 1.01 to 1.15 depending on salinity.

---

## PVT Correlations

BORES uses the following industry-standard correlations:

| Property | Correlation | Reference |
|---|---|---|
| Bubble point pressure | Standing (1947) | Standing, M.B. |
| Solution GOR ($R_s$) | Standing (1947) | Standing, M.B. |
| Oil FVF ($B_o$) | Standing / Vasquez-Beggs | Vasquez and Beggs (1980) |
| Oil compressibility | Vasquez-Beggs | Vasquez and Beggs (1980) |
| Dead oil viscosity | Beggs-Robinson | Beggs and Robinson (1975) |
| Live oil viscosity | Beggs-Robinson | Beggs and Robinson (1975) |
| Gas Z-factor | Hall-Yarborough/Dranchuk-Abou-Kazeem | Hall and Yarborough (1973) |
| Gas viscosity | Lee-Gonzalez-Eakin | Lee, Gonzalez, Eakin (1966) |
| Water properties | McCain | McCain (1990) |

These correlations are implemented in `bores.correlations` as both scalar functions (in `core.py`) and vectorized array functions (in `arrays.py`) for grid-wide evaluation.

---

## How `reservoir_model()` Works

The `bores.reservoir_model()` factory follows this sequence:

1. **Validate inputs**: Check grid shapes match, saturations sum to 1, values are in physical ranges.
2. **Estimate missing oil properties**: From oil specific gravity, viscosity, and bubble point, compute $R_s$, $B_o$, $c_o$, density, and live oil viscosity using PVT correlations.
3. **Estimate gas properties**: From gas specific gravity and reservoir conditions, compute $Z$, $B_g$, $\mu_g$, and gas density.
4. **Estimate water properties**: From water specific gravity, compute $B_w$, $c_w$, $\mu_w$, and water density.
5. **Build the grid**: Construct the internal grid data structure with transmissibilities and cell volumes.
6. **Return the model**: An immutable `ReservoirModel` object.

You can override any computed property by providing it explicitly. For example, if you have lab-measured gas viscosity, pass `gas_viscosity_grid` and BORES will use your values instead of the Lee-Gonzalez correlation.

---

## Accessing Fluid Properties

After building the model, you can inspect all computed fluid properties:

```python
import bores

model = bores.reservoir_model(...)

# Oil properties
Rs = model.fluid_properties.solution_gas_oil_ratio_grid    # SCF/STB
Bo = model.fluid_properties.oil_formation_volume_factor_grid  # RB/STB
oil_density = model.fluid_properties.oil_density_grid       # lbm/ft³

# Gas properties
Bg = model.fluid_properties.gas_formation_volume_factor_grid  # RB/SCF
z_factor = model.fluid_properties.gas_compressibility_factor_grid

# Pressure and saturation
pressure = model.fluid_properties.pressure_grid
So = model.fluid_properties.oil_saturation_grid
Sw = model.fluid_properties.water_saturation_grid
Sg = model.fluid_properties.gas_saturation_grid
```

All properties are NumPy arrays with the same shape as your grid.

!!! warning "Correlation Limitations"

    PVT correlations are empirical fits to experimental data and have limited accuracy. They work well for conventional light-to-medium oils (25-45 API) with hydrocarbon gases. For heavy oils (below 20 API), CO2-rich systems, or near-critical fluids, consider using PVT tables from equation-of-state calculations or laboratory measurements. See [PVT Tables](advanced/pvt-tables.md) for details.
