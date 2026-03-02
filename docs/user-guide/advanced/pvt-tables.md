# PVT Tables

## Overview

By default, BORES computes fluid properties (viscosity, density, formation volume factor, compressibility, and solution gas-oil ratio) at each time step using empirical correlations. These correlations are fast and cover a wide range of conditions, but they are general-purpose approximations. When you have laboratory PVT data from your specific reservoir fluids, you can provide that data as tabulated properties and BORES will interpolate directly from your measurements instead of using correlations.

PVT tables offer several advantages over correlations. They capture the specific behavior of your reservoir fluids, including any non-ideal interactions between components that correlations miss. They eliminate the uncertainty inherent in choosing among different correlation families (Standing vs. Vasquez-Beggs vs. Glaso for bubble point, for example). And they allow you to incorporate laboratory measurements, equation-of-state results, or data from specialized PVT analysis software directly into your simulation.

The PVT table system in BORES consists of two classes: `PVTTableData` stores the raw tabulated data, and `PVTTables` builds fast interpolators from that data for runtime lookups. Once the interpolators are built, the raw data can be discarded to save memory.

---

## Building PVT Table Data

The simplest way to create PVT tables is using `build_pvt_table_data()`, which generates tables from correlations over a specified pressure and temperature range. This gives you a baseline that you can then replace with laboratory data for specific properties.

```python
import bores
import numpy as np
from bores.tables.pvt import build_pvt_table_data, PVTTables

bores.use_32bit_precision()

# Define pressure and temperature grids
pressures = np.linspace(500, 5000, 50)      # 500 to 5000 psi
temperatures = np.linspace(100, 250, 30)    # 100 to 250 F

# Build table data from correlations
pvt_data = build_pvt_table_data(
    pressures=pressures,
    temperatures=temperatures,
    oil_specific_gravity=0.85,
)
```

The `build_pvt_table_data()` function computes all oil, water, and gas properties over the specified pressure-temperature grid using BORES's internal correlations. The result is a `PVTTableData` object containing 2D arrays (pressure x temperature) for each property.

### Specifying Fluid Properties

You can customize the fluid characterization by providing additional parameters:

```python
pvt_data = build_pvt_table_data(
    pressures=pressures,
    temperatures=temperatures,
    oil_specific_gravity=0.87,
    gas_gravity=0.65,                  # Gas specific gravity (air = 1.0)
    water_salinity=50000.0,            # ppm NaCl
    estimated_solution_gor=500.0,      # SCF/STB estimate
)
```

The gas gravity affects gas density, viscosity, and z-factor calculations. The water salinity affects water density, viscosity, and formation volume factor. The estimated solution GOR provides a starting point for bubble point pressure estimation.

### Providing Laboratory Data

If you have laboratory measurements for specific properties, pass them directly as pre-computed tables:

```python
# Lab-measured oil viscosity (n_pressures x n_temperatures)
lab_oil_viscosity = np.loadtxt("lab_viscosity.csv")

pvt_data = build_pvt_table_data(
    pressures=pressures,
    temperatures=temperatures,
    oil_specific_gravity=0.87,
    oil_viscosity_table=lab_oil_viscosity,        # Use lab data
    oil_density_table=lab_oil_density,             # Use lab data
    # Other properties still computed from correlations
)
```

Any property table you provide directly overrides the corresponding correlation-computed table. Properties you do not provide are computed from correlations as usual. This lets you mix laboratory data (for the properties you have measured) with correlation estimates (for the rest).

### Salinity-Dependent Water Properties

Water properties depend on salinity in addition to pressure and temperature. If your reservoir has varying formation water salinity, provide a salinity grid:

```python
salinities = np.array([10000, 50000, 100000, 200000])  # ppm NaCl

pvt_data = build_pvt_table_data(
    pressures=pressures,
    temperatures=temperatures,
    oil_specific_gravity=0.87,
    salinities=salinities,
)
```

With a salinity grid, water property tables become 3D arrays (pressure x temperature x salinity), enabling interpolation across all three dimensions.

---

## Creating PVT Tables (Interpolators)

Once you have a `PVTTableData` object, wrap it in `PVTTables` to build the interpolators:

```python
pvt_tables = PVTTables(
    data=pvt_data,
    interpolation_method="linear",
    validate=True,
    warn_on_extrapolation=False,
)
```

The `PVTTables` constructor validates the data (grid monotonicity, physical consistency) and builds fast interpolators using `scipy.interpolate.RectBivariateSpline` for 2D properties and `scipy.interpolate.RegularGridInterpolator` for 3D properties. The interpolators provide O(1) lookup performance.

### Interpolation Methods

| Method | Speed | Accuracy | Minimum Points |
|---|---|---|---|
| `"linear"` | Fast | 1st order | 2 per dimension |
| `"cubic"` | Slower | 3rd order, smooth derivatives | 4 per dimension |

Linear interpolation is recommended for most cases. Use cubic interpolation when you need smooth derivatives (for example, in Newton iterations where property derivatives affect convergence) and have enough data points (at least 4 per dimension).

### Memory Management

After building the interpolators, the raw table data is no longer needed. You can discard it to save approximately 50% of the memory used by the PVT system:

```python
import gc

pvt_data = build_pvt_table_data(...)
pvt_tables = PVTTables(pvt_data)

del pvt_data
gc.collect()  # Optional: force garbage collection
```

### Physical Consistency Validation

When `validate=True` (the default), `PVTTables` checks that:

- All viscosities are positive
- All densities are positive
- All formation volume factors are positive
- Gas density is less than oil density at all conditions
- 2D tables have shape (n_pressures, n_temperatures)
- 3D tables have shape (n_pressures, n_temperatures, n_salinities)

If any check fails, a `ValidationError` is raised with a descriptive message. Set `validate=False` to skip these checks for faster initialization when you are confident in your data quality.

### Property Clamping

By default, interpolated values are clamped to physically reasonable ranges to prevent unphysical extrapolation artifacts:

| Property | Minimum | Maximum |
|---|---|---|
| Oil viscosity | 1e-6 cP | 10,000 cP |
| Oil compressibility | 0 | 0.1 psi-1 |
| Oil density | 1.0 lb/ft³ | 80.0 lb/ft³ |
| Oil FVF | 0.5 bbl/STB | 5.0 bbl/STB |
| Solution GOR | 0 SCF/STB | 5,000 SCF/STB |
| Gas viscosity | 1e-6 cP | 100 cP |
| Gas z-factor | 0.1 | 3.0 |
| Water viscosity | 1e-6 cP | 10.0 cP |
| Water FVF | 0.9 bbl/STB | 2.0 bbl/STB |

You can provide custom clamp ranges:

```python
custom_clamps = {
    "oil_viscosity": (0.1, 500.0),
    "oil_density": (30.0, 60.0),
}

pvt_tables = PVTTables(
    data=pvt_data,
    clamps=custom_clamps,
)
```

---

## Using PVT Tables in a Simulation

Pass the `PVTTables` object to the `Config`:

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pvt_tables=pvt_tables,
)

states = list(bores.run(model, config))
```

When `pvt_tables` is provided, BORES uses the tabulated data for all fluid property lookups instead of correlations. This affects pressure equation coefficients, mobility calculations, well rate conversions, and all other computations that depend on fluid properties.

---

## Extrapolation Behavior

When the simulation encounters conditions outside the table bounds (pressure or temperature beyond the defined range), the interpolators extrapolate using the same method (linear or cubic) used for interpolation. This can produce physically unreasonable values if the extrapolation extends far beyond the data.

To detect extrapolation, set `warn_on_extrapolation=True`:

```python
pvt_tables = PVTTables(
    data=pvt_data,
    warn_on_extrapolation=True,
)
```

This logs a warning each time a query falls outside the table bounds, helping you identify whether your pressure and temperature ranges are sufficient for the simulation conditions.

!!! tip "Choosing Table Ranges"

    Build your PVT tables with pressure and temperature ranges that extend slightly beyond the expected simulation conditions. If your initial pressure is 3,000 psi and you expect it to decline to 500 psi, define your pressure grid from 400 to 3,500 psi. This margin prevents extrapolation during normal operation while keeping the table compact.
