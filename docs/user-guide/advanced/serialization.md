# Serialization and Storage

## Overview

BORES provides a comprehensive serialization system for saving and loading simulation data. You can save reservoir models, configurations, simulation states, and analysis results to disk in multiple formats, then load them later to continue a simulation, reproduce results, or share data with collaborators.

The serialization system has two layers. The base layer (`Serializable`) handles conversion between Python objects and dictionaries. The storage layer (`StoreSerializable`) adds the ability to write those dictionaries to various file backends. Most BORES objects inherit from both, so they support both in-memory round-tripping and file-based persistence.

---

## Storage Backends

BORES supports four storage backends, each with different trade-offs:

| Backend | Format | Append | Best For |
|---|---|---|---|
| `ZarrStore` | Zarr (chunked arrays) | Yes | Large simulations, streaming, array-heavy data |
| `HDF5Store` | HDF5 | Yes | Interoperability, scientific data exchange |
| `JSONStore` | JSON | No | Small configs, human-readable, debugging |
| `YAMLStore` | YAML | No | Configuration files, human-readable |

### ZarrStore

Zarr is the recommended backend for simulation state data. It stores arrays in a chunked, compressed format that is fast to write and read, supports appending without rewriting existing data, and handles large datasets efficiently. Apart from saving in local directories, `ZarrStore` should support any `BaseStore` type support by zarr.

```python
from bores.stores import ZarrStore

store = ZarrStore("simulation.zarr")

# Save states
store.dump(states)

# Load states
from bores.states import ModelState
loaded_states = list(store.load(ModelState))
```

### HDF5Store

HDF5 is widely supported in the scientific computing ecosystem. Use it when you need to share data with other tools (MATLAB, Paraview, commercial simulators) or when your organization has standardized on HDF5.

```python
from bores.stores import HDF5Store

store = HDF5Store("simulation.h5")
store.dump(states)
loaded_states = list(store.load(ModelState))
```

### JSONStore

JSON is human-readable and useful for saving small objects like configurations. It does not handle large numpy arrays efficiently, so avoid using it for state data.

```python
from bores.stores import JSONStore

store = JSONStore("config.json")
store.dump([config])
loaded_config = next(store.load(type(config)))
```

### YAMLStore

YAML is similar to JSON but more human-friendly. Use it for configuration files that you want to edit by hand.

```python
from bores.stores import YAMLStore

store = YAMLStore("config.yaml")
store.dump([config])
```

### Creating Stores with `new_store`

The `new_store` factory function creates the appropriate store based on the file extension:

```python
from bores.stores import new_store

store = new_store("output.zarr")   # ZarrStore
store = new_store("output.h5")    # HDF5Store
store = new_store("output.json")  # JSONStore
store = new_store("output.yaml")  # YAMLStore
```

---

## Store Operations

### Dumping Data

The `dump()` method writes an iterable of serializable objects to the store, overwriting any existing content:

```python
store.dump(states)
```

You can optionally provide a validator (called on each item before writing) and a metadata function (for lightweight filtering without deserializing):

```python
store.dump(
    states,
    validator=lambda s: validate_state(s, dtype="global"),
    meta=lambda s: {"step": s.step, "time": s.time_in_days},
)
```

### Loading Data

The `load()` method returns a generator that yields deserialized objects:

```python
from bores.states import ModelState

# Load all states
for state in store.load(ModelState):
    process(state)

# Load specific indices
for state in store.load(ModelState, indices=[0, 50, 99]):
    analyze(state)

# Load with a predicate on metadata
for state in store.load(ModelState, predicate=lambda e: e.idx < 10):
    plot_early(state)
```

The `indices` parameter loads only the items at the specified positions (zero-based). The `predicate` parameter filters based on `EntryMeta` objects, which contain the index and any metadata stored during `dump()`. Both options avoid deserializing items that you do not need.

### Appending Data

Stores that support appending (`ZarrStore`, `HDF5Store`) can add items without rewriting existing content:

```python
if store.supports_append:
    entry_meta = store.append(new_state)
    print(f"Appended at index {entry_meta.idx}")
```

### Inspecting Entries

The `entries()` method returns metadata for all stored items without deserializing any array data:

```python
entries = store.entries()
print(f"Store contains {len(entries)} items")
for entry in entries:
    print(f"  Index {entry.idx}: {entry.group_name}")
```

---

## Serializable Protocol

All BORES objects that support serialization inherit from the `Serializable` base class. This class provides `dump()` and `load()` methods for dictionary round-tripping:

```python
# Serialize to dictionary
data = config.dump()

# Deserialize from dictionary
restored_config = type(config).load(data)
```

### Custom Serialization

If you create custom classes that need to be serialized, you can implement the `__dump__` and `__load__` methods:

```python
from bores.serialization import Serializable

class MyData(Serializable):
    def __init__(self, values):
        self.values = values

    def __dump__(self, recurse=True):
        return {"values": self.values.tolist()}

    @classmethod
    def __load__(cls, data):
        import numpy as np
        return cls(values=np.array(data["values"]))
```

### Custom Storage Backends

You can register custom storage backends using the `@data_store` decorator:

```python
from bores.stores import data_store, DataStore

@data_store("parquet")
class ParquetStore(DataStore):
    def dump(self, data, **kwargs):
        ...

    def load(self, typ, **kwargs):
        ...

    def entries(self):
        ...

    def flush(self):
        ...
```

---

## File-Based Persistence

Most BORES objects inherit from `StoreSerializable`, which provides convenient methods for saving and loading directly to and from files. The file extension determines which storage backend is used automatically.

### `to_file` and `from_file`

The `to_file()` method saves any serializable object to a file. The `from_file()` class method loads it back. The file extension determines the storage format: `.zarr` uses ZarrStore, `.h5` uses HDF5Store, `.json` uses JSONStore, and `.yaml` uses YAMLStore.

```python
import bores
from bores.models import ReservoirModel

# Save a model to HDF5
model.to_file("my_model.h5")

# Load it back
loaded_model = ReservoirModel.from_file("my_model.h5")
```

The `save` method is an alias for `to_file`, so you can use whichever reads better in your code:

```python
model.save("my_model.h5")  # Same as model.to_file("my_model.h5")
```

### `to_store` and `from_store`

When you already have a store object (for example, because you want to reuse a connection or configure specific options), use `to_store()` and `from_store()` directly:

```python
from bores.stores import ZarrStore

store = ZarrStore("models.zarr")

# Save to an existing store
model.to_store(store)

# Load from an existing store
loaded_model = ReservoirModel.from_store(store)
```

The `from_store()` method returns the first item in the store (or `None` if the store is empty). If the store contains multiple items, only the first is returned.

### Saving and Loading Configurations

Configurations are best saved as YAML or JSON for human readability. You can edit the YAML file by hand and reload it:

```python
# Save config to YAML
config.to_file("simulation_config.yaml")

# Edit the YAML file by hand, then reload
loaded_config = bores.Config.from_file("simulation_config.yaml")
```

### Saving and Loading Models

Reservoir models contain large arrays (porosity, permeability, saturations), so HDF5 or Zarr are the best formats:

```python
# Save model to HDF5
model.to_file("reservoir_model.h5")

# Load model
from bores.models import ReservoirModel
loaded_model = ReservoirModel.from_file("reservoir_model.h5")
```

### Timer State

The timer supports its own state serialization for checkpointing mid-simulation:

```python
# Save timer state
timer_state = timer.dump_state()

# Restore timer
restored_timer = bores.Timer.load_state(timer_state)
```

---

## Run Objects

The `Run` class bundles a reservoir model and a simulation configuration into a single, serializable unit. You can think of a `Run` as a simulation definition: it contains everything needed to reproduce a simulation. Because `Run` inherits from `StoreSerializable`, you can save and load entire simulation setups with a single call.

### Creating a Run

```python
import bores

run = bores.Run(
    model=model,
    config=config,
    name="Base Case Waterflood",
    description="5-spot pattern, 1000 STB/day injection",
    tags=("waterflood", "base-case", "field-A"),
)
```

The `name`, `description`, and `tags` fields are optional metadata for organizing your simulation library. The `created_at` field is set automatically to the current UTC timestamp.

### Executing a Run

A `Run` is callable and iterable. Calling it returns a generator of `ModelState` objects, exactly like `bores.run(model, config)`. Iterating over it does the same thing.

```python
# As a callable
for state in run():
    process(state)

# As an iterable
for state in run:
    process(state)

# With bores.run() (accepts Run objects directly)
for state in bores.run(run):
    process(state)

# Override config at execution time
new_config = config.with_updates(output_frequency=5)
for state in bores.run(run, config=new_config):
    process(state)
```

### Saving and Loading Runs

Because `Run` is a `StoreSerializable`, you can save and load complete simulation definitions:

```python
# Save run to HDF5
run.to_file("base_case.h5")

# Load run later
loaded_run = bores.Run.from_file("base_case.h5")
print(f"Run: {loaded_run.name}")
print(f"Created: {loaded_run.created_at}")

# Execute the loaded run
for state in loaded_run:
    process(state)
```

### Loading from Separate Files

When your model and config are stored in separate files, use the `from_files()` class method:

```python
run = bores.Run.from_files(
    model_path="reservoir_model.h5",
    config_path="simulation_config.yaml",
    pvt_data_path="pvt_tables.h5",  # Optional
)
```

The `pvt_data_path` parameter is optional. When provided, the PVT tables are loaded and attached to the config automatically. This is useful when PVT data comes from a separate laboratory analysis file.

---

## Complete Workflows

### Run, Save States, and Analyze Later

The most common workflow is to run a simulation with streaming persistence, then analyze the saved results in a separate session:

```python
import bores
from bores.stores import ZarrStore
from bores.streams import StateStream

# Session 1: Run and save
store = ZarrStore("simulation.zarr")
with StateStream(
    states=bores.run(model, config),
    store=store,
    checkpoint_store=ZarrStore("checkpoints.zarr"),
    checkpoint_interval=50,
) as stream:
    stream.consume()

# Session 2: Analyze saved results (no re-simulation needed)
from bores.states import ModelState
from bores.analyses import ModelAnalyst

store = ZarrStore("simulation.zarr")
analyst = ModelAnalyst(store.load(ModelState))
print(f"Recovery factor: {analyst.oil_recovery_factor:.2%}")
```

### Save and Reproduce a Simulation

Package a simulation as a `Run` object so anyone can reproduce it:

```python
# Save the simulation definition
run = bores.Run(
    model=model,
    config=config,
    name="SPE1 Benchmark",
    tags=("benchmark", "spe1"),
)
run.to_file("spe1_run.h5")

# Later (or on another machine): reproduce exactly
loaded_run = bores.Run.from_file("spe1_run.h5")
states = list(bores.run(loaded_run))
```

### Load Just the Final State

When you only need the end result from a previously saved simulation:

```python
store = ZarrStore("simulation.zarr")
max_idx = store.max_index()
final_state = next(store.load(ModelState, indices=[max_idx]))
print(f"Final pressure: {final_state.model.fluid_properties.pressure_grid.mean():.1f} psi")
```
