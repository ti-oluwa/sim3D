"""State storage classes for reservoir simulation states."""

from abc import ABC, abstractmethod
import logging
from os import PathLike
from pathlib import Path
import typing
import sys
from numcodecs import Blosc

import attrs
import h5py
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from zarr.storage import StoreLike

from bores._precision import get_dtype
from bores.errors import StorageError, ValidationError
from bores.grids.base import (
    CapillaryPressureGrids,
    RateGrids,
    RelPermGrids,
    RelativeMobilityGrids,
)
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockPermeability,
    RockProperties,
    SaturationHistory,
)
from bores.states.base import ModelState, validate_state
from bores.timing import StepMetricsDict, TimerState
from bores.utils import Lazy, load_pickle, save_as_pickle

logger = logging.getLogger(__name__)


__all__ = [
    "new_store",
    "state_store",
    "ZarrStore",
    "PickleStore",
    "HDF5Store",
    "NPZStore",
]

IS_PYTHON_310_OR_LOWER = sys.version_info < (3, 11)


class StateStore(ABC):
    """Abstract base class for state storage classes."""

    @abstractmethod
    def load(self, **kwargs: typing.Any) -> typing.Iterable[ModelState]: ...

    @abstractmethod
    def dump(
        self,
        states: typing.Iterable[ModelState],
        **kwargs: typing.Any,
    ) -> None: ...


StoreT = typing.TypeVar("StoreT", bound=StateStore)

_STATE_STORES: typing.Dict[str, typing.Type[StateStore]] = {}


@typing.overload
def state_store(
    name: str,
    store_cls: typing.Type[StoreT],
) -> None:
    """Register a state store class with a given name."""
    ...


@typing.overload
def state_store(
    name: str,
) -> typing.Callable[[typing.Type[StoreT]], typing.Type[StoreT]]:
    """Register a state store class with a given name."""
    ...


def state_store(
    name: str,
    store_cls: typing.Optional[typing.Type[StoreT]] = None,
) -> typing.Union[None, typing.Callable[[typing.Type[StoreT]], typing.Type[StoreT]]]:
    """
    Register a state store class with a given name.

    :param
    """

    def _decorator(store_cls: typing.Type[StoreT]) -> typing.Type[StoreT]:
        _STATE_STORES[name] = store_cls
        return store_cls

    if store_cls is not None:
        _STATE_STORES[name] = store_cls
        return
    return _decorator


def _validate_filepath(
    filepath: PathLike,
    expected_extension: typing.Optional[str] = None,
    is_directory: bool = False,
) -> Path:
    """
    Validate and normalize a filepath for state storage.

    :param filepath: Path to validate
    :param expected_extension: Expected file extension (e.g., '.pkl', '.h5', '.npz')
        If None, no extension validation is performed
    :param is_directory: If True, validates that the path is suitable for a directory
        (no extension or matches expected extension for directory-based stores)
    :return: Validated Path object
    :raises StorageError: If filepath is invalid or has wrong extension
    """
    path = Path(filepath)

    # Check for empty path
    if not str(path).strip():
        raise StorageError("Filepath cannot be empty")

    # Check for invalid characters
    if "\x00" in str(path):
        raise StorageError("Filepath contains null characters")

    # For directory-based stores
    if is_directory:
        # If an extension is expected, ensure it matches (e.g., '.zarr')
        if expected_extension and path.suffix:
            if expected_extension not in path.suffixes:
                raise StorageError(
                    f"Directory-based store expected extension '{expected_extension}', "
                    f"got '{''.join(path.suffixes)}'. Use '{path.with_suffix(expected_extension)}' instead."
                )
        # Warn if the path looks like a file (has an unexpected extension)
        elif path.suffix and ".zarr" not in path.suffixes:
            logger.warning(
                f"Path '{path}' has extension '{path.suffix}' but will be treated as a directory. "
                f"Consider using a name without extension or '.zarr' for clarity."
            )
        return path

    # For file-based stores
    if expected_extension:
        if not path.suffix:
            # Auto-add extension if missing
            path = path.with_suffix(expected_extension)
            logger.debug(f"Added extension: {path}")
        elif expected_extension not in path.suffixes:
            raise StorageError(
                f"Expected file extension '{expected_extension}', got '{''.join(path.suffixes)}'. "
                f"Use '{path.with_suffix(expected_extension)}' instead."
            )

    return path


class StateMetadata(typing.TypedDict):
    """Time-invariant metadata stored once per simulation."""

    rock_fluid_properties: typing.Any
    boundary_conditions: typing.Any
    wells: typing.Any


@state_store("pickle")
class PickleStore(StateStore):
    """
    Pickle-based storage.

    Python-native, simple, easy to use store.
    """

    def __init__(
        self,
        filepath: PathLike,
        compression: typing.Optional[typing.Literal["gzip", "lzma"]] = "gzip",
        compression_level: int = 5,
    ):
        """
        Initialize the store

        :param filepath: Path to the pickle file
        :param compression: Compression method - "gzip" (fast, good compression),
            "lzma" (slower, better compression), or None
        :param compression_level: Compression level (1-9 for gzip, 0-9 for lzma)
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        self.filepath = _validate_filepath(filepath, expected_extension=".pkl")
        self.compression = compression
        self.compression_level = compression_level

    def dump(
        self,
        states: typing.Iterable[ModelState],
        exist_ok: bool = True,
        validate: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump states using pickle with compression.

        :param states: Iterable of `ModelState` instances to dump
        :param exist_ok: If True, will overwrite existing files safely
        """
        if validate:
            state_list = [validate_state(state) for state in states]
        else:
            state_list = list(states)

        save_as_pickle(
            state_list,
            self.filepath,
            exist_ok=exist_ok,
            compression=self.compression,  # type: ignore
            compression_level=self.compression_level,
        )

    def load(
        self,
        validate: bool = True,
        dtype: typing.Optional[np.typing.DTypeLike] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[ModelState, None, None]:
        """
        Load states from pickle file.

        :param validate: If True, validate each state after loading
        :param dtype: Optional dtype to coerce loaded arrays to. If None, uses global dtype.
            Only applied when validate=True.
        :return: Generator yielding ModelState instances
        """
        if validate and dtype is None:
            dtype = get_dtype()

        states = load_pickle(self.filepath)
        if isinstance(states, dict):
            for state in states.values():
                if validate:
                    yield validate_state(state, dtype=dtype)
                else:
                    yield state
        else:
            for state in states:
                if validate:
                    yield validate_state(state, dtype=dtype)
                else:
                    yield state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath}, compression={self.compression})"


@state_store("zarr")
class ZarrStore(StateStore):
    """
    Zarr-based storage with hybrid array/pickle approach.

    Fast, efficient compression with lazy loading.
    Best for large 3D numpy arrays.
    Best lazy loading support among available formats.
    """

    def __init__(
        self,
        store: StoreLike,
        metadata_dir: PathLike,
        compressor: typing.Literal["zstd", "lz4", "blosclz"] = "zstd",
        compression_level: int = 3,
        chunks: typing.Optional[typing.Tuple[int, ...]] = None,
    ):
        """
        Initialize the store

        :param store: Zarr store (file path, directory, or `Store` object)
        :param metadata_dir: Directory path to store pickled metadata
        :param compressor: Compression algorithm - 'zstd', 'lz4', 'blosclz'. blosc with zstd is fastest for scientific data
        :param compression_level: Compression level (1-9)
        :param chunks: Chunk size for the Zarr arrays
        :raises StorageError: If filepath is invalid or has incompatible extension
        """
        self.store = store
        self.chunks = chunks
        self.metadata_dir = _validate_filepath(metadata_dir, is_directory=True)
        if IS_PYTHON_310_OR_LOWER:
            self.compressor = Blosc(
                cname=compressor,
                clevel=compression_level,
                shuffle=Blosc.BITSHUFFLE,
            )
        else:
            self.compressor = BloscCodec(
                cname=compressor,
                clevel=compression_level,
                shuffle=BloscShuffle.bitshuffle,
            )

    def _ensure_metadata_dir(self) -> Path:
        """Ensure metadata directory exists for pickled objects."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        return self.metadata_dir

    def _dump_metadata(self, state: ModelState) -> None:
        """
        Store state metadata.

        :param state: `ModelState` containing the metadata
        """
        metadata_dir = self._ensure_metadata_dir()

        rf_path = metadata_dir / "rock_fluid_properties.pkl"
        bc_path = metadata_dir / "boundary_conditions.pkl"
        wells_path = metadata_dir / "wells.pkl"

        save_as_pickle(
            obj=state.model.rock_fluid_properties,
            filepath=rf_path,
            exist_ok=True,
            compression="gzip",
        )
        save_as_pickle(
            obj=state.model.boundary_conditions,
            filepath=bc_path,
            exist_ok=True,
            compression="gzip",
        )
        save_as_pickle(
            obj=state.wells, filepath=wells_path, exist_ok=True, compression="gzip"
        )
        logger.debug(f"Saved state metadata for step {state.step}")

    def _load_metadata(self, lazy: bool = False) -> StateMetadata:
        """
        Load state metadata.

        :param lazy: If True, returns Lazy objects that defer loading until accessed
        :return: StateMetadata dict with metadata
        """
        metadata_dir = self.metadata_dir
        if not metadata_dir.exists():
            raise StorageError(
                f"Metadata directory not found: {metadata_dir}. "
                "This Zarr store may be incomplete or corrupted."
            )

        rf_path = metadata_dir / "rock_fluid_properties.pkl.gz"
        bc_path = metadata_dir / "boundary_conditions.pkl.gz"
        wells_path = metadata_dir / "wells.pkl.gz"

        if not rf_path.exists() or not bc_path.exists() or not wells_path.exists():
            raise StorageError(
                f"Required metadata files not found in {metadata_dir}. "
                "Expected: rock_fluid_properties.pkl, boundary_conditions.pkl, wells.pkl"
            )

        if lazy:
            # Return lazy-loading factories
            rock_fluid_properties = Lazy.defer(lambda: load_pickle(filepath=rf_path))
            boundary_conditions = Lazy.defer(lambda: load_pickle(filepath=bc_path))
            wells = Lazy.defer(lambda: load_pickle(filepath=wells_path))
            logger.debug("Created lazy loaders for state metadata")
        else:
            # Load immediately
            rock_fluid_properties = load_pickle(filepath=rf_path)
            boundary_conditions = load_pickle(filepath=bc_path)
            wells = load_pickle(filepath=wells_path)
            logger.debug("Loaded state metadata")

        return StateMetadata(
            rock_fluid_properties=rock_fluid_properties,
            boundary_conditions=boundary_conditions,
            wells=wells,
        )

    def _get_chunks(
        self, shape: typing.Tuple[int, ...]
    ) -> typing.Optional[typing.Tuple[int, ...]]:
        """
        Determine optimal chunk size if not provided.

        :param shape: Shape of the array to chunk
        :return: Optimal chunk shape or None for auto-chunking
        """
        if self.chunks:
            return self.chunks

        # As a rule of thumb, use chunks of ~1-10 MB for good performance
        # For 3D grids, use smaller chunks for better I/O
        if len(shape) == 3:
            return (
                min(20, shape[0]),
                min(20, shape[1]),
                min(20, shape[2]),
            )
        elif len(shape) == 2:
            return (
                min(100, shape[0]),
                min(100, shape[1]),
            )
        return None  # Auto-chunking

    def _create_dataset(
        self, group: zarr.Group, name: str, data: np.typing.NDArray
    ) -> zarr.Array:
        """
        Helper to create compressed dataset.

        :param group: Zarr group to create dataset in
        :param name: Name of the dataset
        :param data: NumPy array data to store
        :return: Created Zarr array
        """
        return group.create_array(
            name=name,
            data=data,
            chunks=self._get_chunks(shape=data.shape) or "auto",
            compressors=[self.compressor],
            overwrite=True,
        )

    def dump(
        self,
        states: typing.Iterable[ModelState],
        exist_ok: bool = True,
        validate: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump states to Zarr format with compression.

        Storage Structure:
        - metadata/: Pickled RockFluidProperties and BoundaryConditions (once)
        - step_NNNNNN/: Zarr groups containing all array data per timestep

        :param states: Iterable of `ModelState` instances to dump
        :param exist_ok: If True, will append to existing storage or create new
        :param validate: If True, validate each state before dumping
        """
        # Use 'a' (append) mode to reuse existing store, or 'w-' to fail if exists
        # This allows streaming to work properly without overwriting on each flush
        mode = "a" if exist_ok else "w-"
        root = zarr.open_group(store=self.store, mode=mode, zarr_version=3)  # type: ignore

        num_steps = 0
        dumped_metadata = False
        for state in states:
            if validate:
                state = validate_state(state=state)

            # Save pickled metadata once on first step
            if not dumped_metadata:
                self._dump_metadata(state=state)
                dumped_metadata = True

            step_name = f"step_{state.step:06d}"
            step_group = root.require_group(step_name)

            # Store metadata as attributes
            step_group.attrs["step"] = state.step
            step_group.attrs["step_size"] = state.step_size
            step_group.attrs["time"] = state.time
            step_group.attrs["grid_shape"] = state.model.grid_shape

            # Store timer state if present
            if state.timer_state is not None:
                timer_group = step_group.require_group("timer_state")
                self._dump_timer_state(group=timer_group, timer_state=state.timer_state)

            self._dump_model(group=step_group.require_group("model"), model=state.model)
            self._dump_rates(
                group=step_group.require_group("injection"), rates=state.injection
            )
            self._dump_rates(
                group=step_group.require_group("production"), rates=state.production
            )
            self._dump_relperm(
                group=step_group.require_group("relative_permeabilities"),
                relperm=state.relative_permeabilities,
            )
            self._dump_relative_mobilities(
                group=step_group.require_group("relative_mobilities"),
                relative_mobilities=state.relative_mobilities,
            )
            self._dump_capillary_pressures(
                group=step_group.require_group("capillary_pressures"),
                capillary_pressures=state.capillary_pressures,
            )
            logger.debug(f"Stored state at step {state.step}")
            num_steps = state.step + 1

        # Store global metadata
        root.attrs["version"] = 3
        root.attrs["num_steps"] = num_steps
        logger.debug(f"Completed dump of {num_steps} states to {self.store}")

    def _dump_model(self, group: zarr.Group, model: ReservoirModel):
        """
        Dump model properties.

        :param group: Zarr group to dump into
        :param model: ReservoirModel instance to dump
        """
        # Fluid properties
        fluid_group = group.require_group(name="fluid_properties")
        fluid_properties = model.fluid_properties

        for field in attrs.fields(fluid_properties.__class__):
            value = getattr(fluid_properties, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=fluid_group, name=field.name, data=value)
            else:
                fluid_group.attrs[field.name] = value

        # Rock properties
        rock_properties_group = group.require_group(name="rock_properties")
        rock_properties = model.rock_properties

        rock_properties_group.attrs["compressibility"] = rock_properties.compressibility

        for field in attrs.fields(rock_properties.__class__):
            if field.name == "compressibility":
                continue

            value = getattr(rock_properties, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(
                    group=rock_properties_group, name=field.name, data=value
                )
            elif field.name == "absolute_permeability":
                perm_group = rock_properties_group.require_group(
                    name="absolute_permeability"
                )
                self._create_dataset(group=perm_group, name="x", data=value.x)
                self._create_dataset(group=perm_group, name="y", data=value.y)
                self._create_dataset(group=perm_group, name="z", data=value.z)

        # Other model attributes
        self._create_dataset(
            group=group, name="thickness_grid", data=model.thickness_grid
        )
        group.attrs["dip_angle"] = model.dip_angle
        group.attrs["dip_azimuth"] = model.dip_azimuth
        group.attrs["cell_dimension"] = model.cell_dimension
        group.attrs["grid_shape"] = model.grid_shape

        # Saturation history
        saturation_history_group = group.require_group(name="saturation_history")
        saturation_history = model.saturation_history
        self._create_dataset(
            group=saturation_history_group,
            name="max_water_saturation_grid",
            data=saturation_history.max_water_saturation_grid,
        )
        self._create_dataset(
            group=saturation_history_group,
            name="max_gas_saturation_grid",
            data=saturation_history.max_gas_saturation_grid,
        )
        self._create_dataset(
            group=saturation_history_group,
            name="water_imbibition_flag_grid",
            data=saturation_history.water_imbibition_flag_grid,
        )
        self._create_dataset(
            group=saturation_history_group,
            name="gas_imbibition_flag_grid",
            data=saturation_history.gas_imbibition_flag_grid,
        )

    def _dump_rates(self, group: zarr.Group, rates: RateGrids):
        """
        Dump injection/production rates.

        :param group: Zarr group to dump into
        :param rates: RateGrids instance to dump
        """
        for field in attrs.fields(rates.__class__):
            value = getattr(rates, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=field.name, data=value)

    def _dump_relperm(self, group: zarr.Group, relperm: RelPermGrids):
        """
        Dump relative permeabilities.

        :param group: Zarr group to dump into
        :param relperm: RelPermGrids instance to dump
        """
        for field in attrs.fields(relperm.__class__):
            value = getattr(relperm, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=field.name, data=value)

    def _dump_relative_mobilities(
        self, group: zarr.Group, relative_mobilities: RelativeMobilityGrids
    ):
        """
        Dump relative mobilities.

        :param group: Zarr group to dump into
        :param relative_mobilities: `RelativeMobilityGrids` instance to dump
        """
        for field in attrs.fields(relative_mobilities.__class__):
            value = getattr(relative_mobilities, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=field.name, data=value)

    def _dump_capillary_pressures(
        self, group: zarr.Group, capillary_pressures: CapillaryPressureGrids
    ):
        """
        Dump capillary pressures.

        :param group: Zarr group to dump into
        :param capillary_pressures: CapillaryPressureGrids instance to dump
        """
        for field in attrs.fields(capillary_pressures.__class__):
            value = getattr(capillary_pressures, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=field.name, data=value)

    def _dump_timer_state(self, group: zarr.Group, timer_state: TimerState):
        """
        Dump timer state to Zarr group.

        :param group: Zarr group to dump into
        :param timer_state: TimerState dict to dump
        """
        # Store all scalar/simple fields as attributes
        for key, value in timer_state.items():
            if key == "recent_metrics":
                # Store list of StepMetricsDict
                metrics_list = typing.cast(
                    typing.List[typing.Dict[str, typing.Any]], value
                )
                metrics_group = group.require_group("recent_metrics")
                for idx, metric in enumerate(metrics_list):
                    metric_group = metrics_group.require_group(f"metric_{idx}")
                    for metric_key, metric_value in metric.items():
                        if metric_value is not None:
                            # Convert to native Python types for JSON compatibility
                            if isinstance(metric_value, (np.integer, np.floating)):
                                metric_value = metric_value.item()
                            elif isinstance(metric_value, np.bool_):
                                metric_value = bool(metric_value)
                            metric_group.attrs[metric_key] = metric_value
            elif key == "failed_step_sizes":
                # Store as array
                failed_sizes = typing.cast(typing.List[float], value)
                if failed_sizes:
                    self._create_dataset(
                        group=group,
                        name="failed_step_sizes",
                        data=np.array(failed_sizes),
                    )
            else:
                # Store scalar values as attributes (skip None values)
                if value is not None:
                    # Convert to native Python types for JSON compatibility
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    # Cast to ensure JSON-compatible type
                    group.attrs[key] = typing.cast(
                        typing.Union[str, int, float, bool], value
                    )

    def load(
        self,
        lazy: bool = True,
        validate: bool = True,
        dtype: typing.Optional[np.typing.DTypeLike] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[ModelState, None, None]:
        """
        Load states from Zarr format.

        :param lazy: If True, loads arrays only when accessed (memory efficient).
                     If False, loads all data immediately.
        :param validate: If True, validate each state after loading
        :param dtype: Optional dtype to coerce loaded arrays to. If None, uses global dtype.
                     Only applied when validate=True.
        :return: Generator yielding `ModelState` instances
        """
        root = zarr.open_group(store=self.store, mode="r", zarr_version=3)

        # Load metadata once (time-invariant)
        metadata = self._load_metadata(lazy=lazy)

        if validate and dtype is None:
            dtype = get_dtype()

        for key in sorted(root.keys()):
            step_group = root[key]

            # Load metadata
            step = typing.cast(int, step_group.attrs["step"])
            step_size = typing.cast(float, step_group.attrs["step_size"])
            time = typing.cast(float, step_group.attrs["time"])

            # Load timer state if present
            timer_state: typing.Optional[TimerState] = None
            if "timer_state" in step_group:  # type: ignore
                timer_group = typing.cast(zarr.Group, step_group["timer_state"])  # type: ignore
                timer_state = self._load_timer_state(group=timer_group)

            if lazy:
                model = Lazy.defer(
                    lambda sg=step_group: self._load_model(
                        group=typing.cast(zarr.Group, sg["model"]),  # type: ignore[index]
                        metadata=metadata,
                    )
                )
                injection = Lazy.defer(
                    lambda sg=step_group: self._load_rates(
                        group=typing.cast(zarr.Group, sg["injection"])  # type: ignore[index]
                    )
                )
                production = Lazy.defer(
                    lambda sg=step_group: self._load_rates(
                        group=typing.cast(zarr.Group, sg["production"])  # type: ignore[index]
                    )
                )
                relperm = Lazy.defer(
                    lambda sg=step_group: self._load_relperm(
                        group=typing.cast(zarr.Group, sg["relative_permeabilities"])  # type: ignore[index]
                    )
                )
                relative_mobilities = Lazy.defer(
                    lambda sg=step_group: self._load_relative_mobilities(
                        group=typing.cast(zarr.Group, sg["relative_mobilities"])  # type: ignore[index]
                    )
                )
                capillary_pressures = Lazy.defer(
                    lambda sg=step_group: self._load_capillary_pressures(
                        group=typing.cast(zarr.Group, sg["capillary_pressures"])  # type: ignore[index]
                    )
                )
            else:
                model = self._load_model(
                    group=typing.cast(zarr.Group, step_group["model"]),  # type: ignore[index]
                    metadata=metadata,
                )
                injection = self._load_rates(
                    group=typing.cast(zarr.Group, step_group["injection"])  # type: ignore[index]
                )
                production = self._load_rates(
                    group=typing.cast(zarr.Group, step_group["production"])  # type: ignore[index]
                )
                relperm = self._load_relperm(
                    group=typing.cast(zarr.Group, step_group["relative_permeabilities"])  # type: ignore[index]
                )
                relative_mobilities = self._load_relative_mobilities(
                    group=typing.cast(zarr.Group, step_group["relative_mobilities"])  # type: ignore[index]
                )
                capillary_pressures = self._load_capillary_pressures(
                    group=typing.cast(zarr.Group, step_group["capillary_pressures"])  # type: ignore[index]
                )

            state = ModelState(
                step=step,
                step_size=step_size,
                time=time,
                model=model,
                wells=metadata["wells"],
                injection=injection,
                production=production,
                relative_permeabilities=relperm,
                relative_mobilities=relative_mobilities,
                capillary_pressures=capillary_pressures,
                timer_state=timer_state,
            )

            if validate:
                state = validate_state(state=state, dtype=dtype)
            yield state

    def _load_model(self, group: zarr.Group, metadata: StateMetadata) -> ReservoirModel:
        """
        Load model from Zarr group.

        :param group: Zarr group containing model data
        :param metadata: Pre-loaded StateMetadata containing time-invariant data
        :return: Reconstructed `ReservoirModel` instance.
        """
        # Load fluid properties
        fluid_group = group["fluid_properties"]  # type: ignore
        fluid_arrays = {}
        for key in fluid_group.array_keys():  # type: ignore
            array = fluid_group[key]  # type: ignore
            fluid_arrays[key] = array[:]  # type: ignore

        # Add any scalar attributes
        for attr_name in fluid_group.attrs.keys():
            fluid_arrays[attr_name] = fluid_group.attrs[attr_name]

        fluid_properties = FluidProperties(**fluid_arrays)

        # Load rock properties
        rock_properties_group = group["rock_properties"]  # type: ignore
        rock_data = {}
        rock_data["compressibility"] = rock_properties_group.attrs["compressibility"]

        for key in rock_properties_group.array_keys():  # type: ignore
            array = rock_properties_group[key]  # type: ignore
            rock_data[key] = array[:]  # type: ignore

        # Load absolute permeability
        perm_group = rock_properties_group["absolute_permeability"]  # type: ignore
        x = perm_group["x"][:]  # type: ignore
        y = perm_group["y"][:]  # type: ignore
        z = perm_group["z"][:]  # type: ignore
        rock_data["absolute_permeability"] = RockPermeability(x=x, y=y, z=z)  # type: ignore

        rock_properties = RockProperties(**rock_data)

        # Load saturation history
        saturation_history_group = group["saturation_history"]  # type: ignore
        saturation_history = SaturationHistory(
            max_water_saturation_grid=saturation_history_group[  # type: ignore
                "max_water_saturation_grid"
            ][:],
            max_gas_saturation_grid=saturation_history_group["max_gas_saturation_grid"][  # type: ignore
                :
            ],
            water_imbibition_flag_grid=saturation_history_group[  # type: ignore
                "water_imbibition_flag_grid"
            ][:],
            gas_imbibition_flag_grid=saturation_history_group[  # type: ignore
                "gas_imbibition_flag_grid"
            ][:],
        )

        # Load other model attributes
        thickness_grid = group["thickness_grid"][:]  # type: ignore
        grid_shape = tuple(group.attrs["grid_shape"])  # type: ignore
        cell_dimension = tuple(group.attrs["cell_dimension"])  # type: ignore
        dip_angle = typing.cast(float, group.attrs["dip_angle"])
        dip_azimuth = typing.cast(float, group.attrs["dip_azimuth"])

        return ReservoirModel(
            grid_shape=grid_shape,  # type: ignore
            cell_dimension=cell_dimension,  # type: ignore
            thickness_grid=thickness_grid,  # type: ignore
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            rock_fluid_properties=metadata["rock_fluid_properties"],
            saturation_history=saturation_history,
            boundary_conditions=metadata["boundary_conditions"],
            dip_angle=dip_angle,
            dip_azimuth=dip_azimuth,
        )

    def _load_rates(self, group: zarr.Group) -> RateGrids:
        """
        Load rates from Zarr group.

        :param group: Zarr group containing rate data
        :return: Reconstructed `RateGrids` instance.
        """
        data = {}
        for key in group.array_keys():  # type: ignore
            array = group[key]  # type: ignore
            data[key] = array[:]  # type: ignore

        return RateGrids(**data)

    def _load_relperm(self, group: zarr.Group) -> RelPermGrids:
        """
        Load relative permeabilities from Zarr group.

        :param group: Zarr group containing relative permeability data
        :return: Reconstructed `RelPermGrids` instance.
        """
        data = {}
        for key in group.array_keys():  # type: ignore
            array = group[key]  # type: ignore
            data[key] = array[:]  # type: ignore

        return RelPermGrids(**data)

    def _load_relative_mobilities(self, group: zarr.Group) -> RelativeMobilityGrids:
        """
        Load relative mobilities from Zarr group.

        :param group: Zarr group containing mobility data
        :return: Reconstructed `RelativeMobilityGrids` instance.
        """
        data = {}
        for key in group.array_keys():  # type: ignore
            array = group[key]  # type: ignore
            data[key] = array[:]  # type: ignore

        return RelativeMobilityGrids(**data)

    def _load_capillary_pressures(self, group: zarr.Group) -> CapillaryPressureGrids:
        """
        Load capillary pressures from Zarr group.

        :param group: Zarr group containing capillary pressure data
        :return: Reconstructed `CapillaryPressureGrids` instance.
        """
        data = {}
        for key in group.array_keys():  # type: ignore
            array = group[key]  # type: ignore
            data[key] = array[:]  # type: ignore
        return CapillaryPressureGrids(**data)

    def _load_timer_state(self, group: zarr.Group) -> TimerState:
        """
        Load timer state from Zarr group.

        :param group: Zarr group containing timer state
        :return: TimerState dict
        """
        timer_state: typing.Dict[str, typing.Any] = {}

        # Load scalar attributes
        for key in group.attrs.keys():
            timer_state[key] = group.attrs[key]

        # Load recent_metrics if present
        if "recent_metrics" in group.group_keys():
            metrics_group = typing.cast(zarr.Group, group["recent_metrics"])  # type: ignore
            recent_metrics: typing.List[StepMetricsDict] = []
            for metric_key in sorted(metrics_group.group_keys()):  # type: ignore
                metric_group = typing.cast(zarr.Group, metrics_group[metric_key])  # type: ignore
                metric = StepMetricsDict(
                    step_number=typing.cast(int, metric_group.attrs["step_number"]),
                    step_size=typing.cast(float, metric_group.attrs["step_size"]),
                    cfl=typing.cast(float, metric_group.attrs["cfl"])
                    if metric_group.attrs.get("cfl") is not None
                    else None,
                    newton_iters=typing.cast(int, metric_group.attrs["newton_iters"])
                    if metric_group.attrs.get("newton_iters") is not None
                    else None,
                    success=typing.cast(bool, metric_group.attrs["success"]),
                )
                recent_metrics.append(metric)
            timer_state["recent_metrics"] = recent_metrics
        else:
            timer_state["recent_metrics"] = []

        # Load failed_step_sizes if present
        if "failed_step_sizes" in group.array_keys():
            failed_array = group["failed_step_sizes"]  # type: ignore
            timer_state["failed_step_sizes"] = failed_array[:].tolist()  # type: ignore
        else:
            timer_state["failed_step_sizes"] = []

        return typing.cast(TimerState, timer_state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(store={self.store}, compressor={self.compressor.cname})"


@state_store("hdf5")
class HDF5Store(StateStore):
    """
    HDF5-based storage with hybrid array/pickle approach.

    Industry standard, good compression, wide tool support.
    Slightly slower than Zarr for many small writes.
    """

    def __init__(
        self,
        filepath: PathLike,
        metadata_dir: typing.Optional[PathLike] = None,
        compression: typing.Literal["gzip", "lzf", "szip"] = "gzip",
        compression_opts: int = 3,
    ):
        """
        Initialize the store

        :param filepath: Path to the HDF5 file
        :param metadata_dir: Directory path to store pickled metadata.
        :param compression: Compression algorithm - 'gzip', 'lzf', or 'szip'
        :param compression_opts: Compression level (1-9 for gzip)
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        self.filepath = _validate_filepath(filepath, expected_extension=".h5")
        self.compression = compression
        self.compression_opts = compression_opts  # 1-9 for gzip
        self.metadata_dir = Path(
            metadata_dir or self.filepath.parent / f"{self.filepath.stem}_metadata"
        )

    def _ensure_metadata_dir(self) -> Path:
        """Ensure metadata directory exists for pickled objects."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        return self.metadata_dir

    def _dump_metadata(self, state: ModelState) -> None:
        """
        Store state metadata.

        These are time-invariant and contain callables, so we store them once.

        :param state: `ModelState` containing the metadata
        """
        metadata_dir = self._ensure_metadata_dir()

        rf_path = metadata_dir / "rock_fluid_properties.pkl"
        bc_path = metadata_dir / "boundary_conditions.pkl"
        wells_path = metadata_dir / "wells.pkl"

        save_as_pickle(
            obj=state.model.rock_fluid_properties,
            filepath=rf_path,
            exist_ok=True,
            compression="gzip",
        )
        save_as_pickle(
            obj=state.model.boundary_conditions,
            filepath=bc_path,
            exist_ok=True,
            compression="gzip",
        )
        save_as_pickle(
            obj=state.wells, filepath=wells_path, exist_ok=True, compression="gzip"
        )
        logger.debug(f"Saved state metadata for step {state.step}")

    def _load_metadata(self, lazy: bool = False) -> StateMetadata:
        """
        Load state metadata.

        :param lazy: If True, returns Lazy objects that defer loading until accessed
        :return: StateMetadata dict with metadata
        """
        metadata_dir = self.metadata_dir
        if not metadata_dir.exists():
            raise StorageError(
                f"Metadata directory not found: {metadata_dir}. "
                "This HDF5 store may be incomplete or corrupted."
            )

        rf_path = metadata_dir / "rock_fluid_properties.pkl.gz"
        bc_path = metadata_dir / "boundary_conditions.pkl.gz"
        wells_path = metadata_dir / "wells.pkl.gz"

        if not rf_path.exists() or not bc_path.exists() or not wells_path.exists():
            raise StorageError(
                f"Required metadata files not found in {metadata_dir}. "
                "Expected: rock_fluid_properties.pkl, boundary_conditions.pkl, wells.pkl"
            )

        if lazy:
            # Return lazy-loading factories
            rock_fluid_properties = Lazy.defer(lambda: load_pickle(filepath=rf_path))
            boundary_conditions = Lazy.defer(lambda: load_pickle(filepath=bc_path))
            wells = Lazy.defer(lambda: load_pickle(filepath=wells_path))
            logger.debug("Created lazy loaders for state metadata")
        else:
            # Load immediately
            rock_fluid_properties = load_pickle(filepath=rf_path)
            boundary_conditions = load_pickle(filepath=bc_path)
            wells = load_pickle(filepath=wells_path)
            logger.debug("Loaded state metadata")

        return StateMetadata(
            rock_fluid_properties=rock_fluid_properties,
            boundary_conditions=boundary_conditions,
            wells=wells,
        )

    def _create_dataset(self, group: h5py.Group, name: str, data: np.ndarray):
        """
        Create a compressed dataset. Deletes the dataset if it already exists to recreate a new one.

        :param group: HDF5 group to create dataset in
        :param name: Name of the dataset
        :param data: NumPy array data to store
        :return: Created HDF5 dataset
        """
        if group.get(name) is not None:
            del group[name]
        return group.create_dataset(
            name=name,
            data=data,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )

    def dump(
        self,
        states: typing.Iterable[ModelState],
        exist_ok: bool = True,
        validate: bool = False,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump states to HDF5 format with compression.

        :param states: Iterable of ModelState instances to dump
        :param exist_ok: If True, will append to existing storage or create new
        :param validate: If True, validate each state before dumping
        """
        mode = "a" if exist_ok else "w-"

        with h5py.File(name=str(self.filepath), mode=mode) as f:
            num_steps = 0
            dumped_metadata = False
            for state in states:
                if validate:
                    state = validate_state(state=state)

                # Save pickled metadata once on first step
                if not dumped_metadata:
                    self._dump_metadata(state=state)
                    dumped_metadata = True

                step_name = f"step_{state.step:06d}"
                step_group = f.require_group(name=step_name)

                # Store metadata as attributes
                step_group.attrs["step"] = state.step
                step_group.attrs["step_size"] = state.step_size
                step_group.attrs["time"] = state.time
                step_group.attrs["grid_shape"] = state.model.grid_shape

                # Store timer state if present
                if state.timer_state is not None:
                    timer_group = step_group.require_group("timer_state")
                    self._dump_timer_state(
                        group=timer_group, timer_state=state.timer_state
                    )

                self._dump_model(
                    group=step_group.require_group(name="model"), model=state.model
                )
                self._dump_rates(
                    group=step_group.require_group(name="injection"),
                    rates=state.injection,
                )
                self._dump_rates(
                    group=step_group.require_group(name="production"),
                    rates=state.production,
                )
                self._dump_relperm(
                    group=step_group.require_group(name="relative_permeabilities"),
                    relperm=state.relative_permeabilities,
                )
                self._dump_relative_mobilities(
                    group=step_group.require_group(name="relative_mobilities"),
                    relative_mobilities=state.relative_mobilities,
                )
                self._dump_capillary_pressures(
                    group=step_group.require_group(name="capillary_pressures"),
                    capillary_pressures=state.capillary_pressures,
                )
                logger.debug(f"Stored state at step {state.step}")
                num_steps = state.step + 1

            # Store global metadata
            f.attrs["num_steps"] = num_steps
            logger.debug(f"Completed dump of {num_steps} states to {self.filepath}")

    def _dump_model(self, group: h5py.Group, model: ReservoirModel):
        """
        Dump model properties.

        :param group: HDF5 group to dump into
        :param model: ReservoirModel instance to dump
        """
        fluid_group = group.require_group(name="fluid_properties")
        fluid = model.fluid_properties

        for field in attrs.fields(fluid.__class__):
            value = getattr(fluid, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=fluid_group, name=field.name, data=value)
            else:
                fluid_group.attrs[field.name] = value

        # Rock properties
        rock_group = group.require_group(name="rock_properties")
        rock = model.rock_properties

        rock_group.attrs["compressibility"] = rock.compressibility

        for field in attrs.fields(rock.__class__):
            if field.name == "compressibility":
                continue

            value = getattr(rock, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=rock_group, name=field.name, data=value)
            elif field.name == "absolute_permeability":
                perm_group = rock_group.require_group(name="absolute_permeability")
                self._create_dataset(group=perm_group, name="x", data=value.x)
                self._create_dataset(group=perm_group, name="y", data=value.y)
                self._create_dataset(group=perm_group, name="z", data=value.z)

        # Other model attributes
        self._create_dataset(
            group=group, name="thickness_grid", data=model.thickness_grid
        )
        group.attrs["dip_angle"] = model.dip_angle
        group.attrs["dip_azimuth"] = model.dip_azimuth
        group.attrs["cell_dimension"] = model.cell_dimension
        group.attrs["grid_shape"] = model.grid_shape

        # Saturation history
        saturation_history_group = group.require_group(name="saturation_history")
        sat_hist = model.saturation_history
        self._create_dataset(
            group=saturation_history_group,
            name="max_water_saturation_grid",
            data=sat_hist.max_water_saturation_grid,
        )
        self._create_dataset(
            group=saturation_history_group,
            name="max_gas_saturation_grid",
            data=sat_hist.max_gas_saturation_grid,
        )
        self._create_dataset(
            group=saturation_history_group,
            name="water_imbibition_flag_grid",
            data=sat_hist.water_imbibition_flag_grid,
        )
        self._create_dataset(
            group=saturation_history_group,
            name="gas_imbibition_flag_grid",
            data=sat_hist.gas_imbibition_flag_grid,
        )

    def _dump_rates(self, group: h5py.Group, rates):
        """
        Dump injection/production rates.

        :param group: HDF5 group to dump into
        :param rates: RateGrids instance to dump
        """
        for field in attrs.fields(rates.__class__):
            value = getattr(rates, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=field.name, data=value)

    def _dump_relperm(self, group: h5py.Group, relperm):
        """
        Dump relative permeabilities.

        :param group: HDF5 group to dump into
        :param relperm: RelPermGrids instance to dump
        """
        for field in attrs.fields(relperm.__class__):
            value = getattr(relperm, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=field.name, data=value)

    def _dump_relative_mobilities(
        self, group: h5py.Group, relative_mobilities: RelativeMobilityGrids
    ):
        """
        Dump relative mobilities.

        :param group: HDF5 group to dump into
        :param relative_mobilities: `RelativeMobilityGrids` instance to dump
        """
        for field in attrs.fields(relative_mobilities.__class__):
            value = getattr(relative_mobilities, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=field.name, data=value)

    def _dump_capillary_pressures(
        self, group: h5py.Group, capillary_pressures: CapillaryPressureGrids
    ):
        """
        Dump capillary pressures.

        :param group: HDF5 group to dump into
        :param capillary_pressures: CapillaryPressureGrids instance to dump
        """
        for field in attrs.fields(capillary_pressures.__class__):
            value = getattr(capillary_pressures, field.name)
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=field.name, data=value)

    def _dump_timer_state(self, group: h5py.Group, timer_state: TimerState):
        """
        Dump timer state to HDF5 group.

        :param group: HDF5 group to dump into
        :param timer_state: TimerState dict to dump
        """
        # Store all scalar/simple fields as attributes
        for key, value in timer_state.items():
            if key == "recent_metrics":
                # Store list of StepMetricsDict
                metrics_list = typing.cast(
                    typing.List[typing.Dict[str, typing.Any]], value
                )
                metrics_group = group.require_group("recent_metrics")
                for idx, metric in enumerate(metrics_list):
                    metric_group = metrics_group.require_group(f"metric_{idx}")
                    for metric_key, metric_value in metric.items():
                        if metric_value is not None:
                            # Convert to native Python types for compatibility
                            if isinstance(metric_value, (np.integer, np.floating)):
                                metric_value = metric_value.item()
                            elif isinstance(metric_value, np.bool_):
                                metric_value = bool(metric_value)
                            metric_group.attrs[metric_key] = metric_value
            elif key == "failed_step_sizes":
                # Store as dataset
                failed_sizes = typing.cast(typing.List[float], value)
                if failed_sizes:
                    self._create_dataset(
                        group=group,
                        name="failed_step_sizes",
                        data=np.array(failed_sizes),
                    )
            else:
                # Store scalar values as attributes (skip None values)
                if value is not None:
                    # Convert to native Python types for compatibility
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    group.attrs[key] = value

    def load(
        self,
        lazy: bool = False,
        validate: bool = True,
        dtype: typing.Optional[np.typing.DTypeLike] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[ModelState, None, None]:
        """
        Load states from HDF5 format.

        :param lazy: If True, defer loading of state properties until access.
            Uses batch lazy loading: on first field access for a state, all fields
            are loaded in a single file open/close cycle and cached.
            Note: For best performance, use `lazy=False` (default) when you need all fields.
            Lazy loading is most beneficial when loading many states but only accessing
            a few of them.
        :param validate: If True, validate each state after loading
        :param dtype: Optional dtype to coerce loaded arrays to. If None, uses global dtype.
                     Only applied when validate=True.
        :return: Generator yielding ModelState instances
        """
        if validate and dtype is None:
            dtype = get_dtype()

        metadata = self._load_metadata(lazy=lazy)
        filepath = str(self.filepath)  # Capture for lazy closures

        with h5py.File(name=filepath, mode="r") as f:
            for key in sorted(f.keys()):
                step_group = f[key]  # type: ignore
                step_key = key  # Capture for lazy closures

                step = int(step_group.attrs["step"])  # type: ignore
                step_size = float(step_group.attrs["step_size"])  # type: ignore
                time = float(step_group.attrs["time"])  # type: ignore

                # Load timer state if present (always load eagerly as it's small)
                timer_state: typing.Optional[TimerState] = None
                if "timer_state" in step_group:  # type: ignore
                    timer_group = typing.cast(h5py.Group, step_group["timer_state"])  # type: ignore
                    timer_state = self._load_timer_state(group=timer_group)

                if lazy:
                    # Batch lazy loading: load all fields in one file open when any is accessed
                    # Use a shared cache dict that gets populated on first access
                    lazy_cache: typing.Dict[str, typing.Any] = {}

                    def load_all_fields(
                        fp: str,
                        sk: str,
                        cache: typing.Dict[str, typing.Any],
                        meta: StateMetadata,
                    ) -> None:
                        """Load all fields for this state in one file open."""
                        if cache:  # Already loaded
                            return
                        with h5py.File(name=fp, mode="r") as lazy_f:
                            sg = lazy_f[sk]
                            cache["model"] = self._load_model(
                                group=typing.cast(h5py.Group, sg["model"]),  # type: ignore[index]
                                metadata=meta,
                            )
                            cache["injection"] = self._load_rates(
                                group=typing.cast(h5py.Group, sg["injection"])  # type: ignore[index]
                            )
                            cache["production"] = self._load_rates(
                                group=typing.cast(h5py.Group, sg["production"])  # type: ignore[index]
                            )
                            cache["relperm"] = self._load_relperm(
                                group=typing.cast(
                                    h5py.Group,
                                    sg["relative_permeabilities"],  # type: ignore[index]
                                )
                            )
                            cache["relative_mobilities"] = (
                                self._load_relative_mobilities(
                                    group=typing.cast(
                                        h5py.Group,
                                        sg["relative_mobilities"],  # type: ignore[index]
                                    )
                                )
                            )
                            cache["capillary_pressures"] = (
                                self._load_capillary_pressures(
                                    group=typing.cast(
                                        h5py.Group,
                                        sg["capillary_pressures"],  # type: ignore[index]
                                    )
                                )
                            )

                    # Create lazy wrappers that share the same cache
                    model = Lazy.defer(
                        lambda fp=filepath, sk=step_key, c=lazy_cache, m=metadata: (
                            load_all_fields(fp, sk, c, m),
                            c["model"],
                        )[1]
                    )
                    injection = Lazy.defer(
                        lambda fp=filepath, sk=step_key, c=lazy_cache, m=metadata: (
                            load_all_fields(fp, sk, c, m),
                            c["injection"],
                        )[1]
                    )
                    production = Lazy.defer(
                        lambda fp=filepath, sk=step_key, c=lazy_cache, m=metadata: (
                            load_all_fields(fp, sk, c, m),
                            c["production"],
                        )[1]
                    )
                    relperm = Lazy.defer(
                        lambda fp=filepath, sk=step_key, c=lazy_cache, m=metadata: (
                            load_all_fields(fp, sk, c, m),
                            c["relperm"],
                        )[1]
                    )
                    relative_mobilities = Lazy.defer(
                        lambda fp=filepath, sk=step_key, c=lazy_cache, m=metadata: (
                            load_all_fields(fp, sk, c, m),
                            c["relative_mobilities"],
                        )[1]
                    )
                    capillary_pressures = Lazy.defer(
                        lambda fp=filepath, sk=step_key, c=lazy_cache, m=metadata: (
                            load_all_fields(fp, sk, c, m),
                            c["capillary_pressures"],
                        )[1]
                    )
                else:
                    model = self._load_model(
                        group=typing.cast(h5py.Group, step_group["model"]),  # type: ignore[index]
                        metadata=metadata,
                    )
                    injection = self._load_rates(
                        group=typing.cast(h5py.Group, step_group["injection"])  # type: ignore[index]
                    )
                    production = self._load_rates(
                        group=typing.cast(h5py.Group, step_group["production"])  # type: ignore[index]
                    )
                    relperm = self._load_relperm(
                        group=typing.cast(
                            h5py.Group,
                            step_group["relative_permeabilities"],  # type: ignore[index]
                        )
                    )
                    relative_mobilities = self._load_relative_mobilities(
                        group=typing.cast(h5py.Group, step_group["relative_mobilities"])  # type: ignore[index]
                    )
                    capillary_pressures = self._load_capillary_pressures(
                        group=typing.cast(h5py.Group, step_group["capillary_pressures"])  # type: ignore[index]
                    )

                state = ModelState(
                    step=step,
                    step_size=step_size,
                    time=time,
                    model=model,
                    wells=metadata["wells"],
                    injection=injection,
                    production=production,
                    relative_permeabilities=relperm,
                    relative_mobilities=relative_mobilities,
                    capillary_pressures=capillary_pressures,
                    timer_state=timer_state,
                )

                if validate:
                    state = validate_state(state=state, dtype=dtype)
                yield state

    def _load_model(self, group: h5py.Group, metadata: StateMetadata) -> ReservoirModel:
        """
        Load model from HDF5 group.

        :param group: HDF5 group containing model data
        :param metadata: Pre-loaded StateMetadata containing time-invariant data
        :return: Reconstructed ReservoirModel instance
        """
        fluid_group = group["fluid_properties"]  # type: ignore
        fluid_arrays = {}
        for key in fluid_group.keys():  # type: ignore
            if key in fluid_group:
                dataset = fluid_group[key]  # type: ignore
                if hasattr(dataset, "shape"):
                    fluid_arrays[key] = dataset[:]  # type: ignore

        for attr_name in fluid_group.attrs.keys():
            fluid_arrays[attr_name] = fluid_group.attrs[attr_name]

        fluid_properties = FluidProperties(**fluid_arrays)

        rock_group = group["rock_properties"]  # type: ignore
        rock_data = {}
        rock_data["compressibility"] = rock_group.attrs["compressibility"]

        for key in rock_group.keys():  # type: ignore
            if key != "absolute_permeability":
                dataset = rock_group[key]  # type: ignore
                if hasattr(dataset, "shape"):
                    rock_data[key] = dataset[:]  # type: ignore

        perm_group = rock_group["absolute_permeability"]  # type: ignore
        x = perm_group["x"][:]  # type: ignore
        y = perm_group["y"][:]  # type: ignore
        z = perm_group["z"][:]  # type: ignore
        rock_data["absolute_permeability"] = RockPermeability(x=x, y=y, z=z)  # type: ignore

        rock_properties = RockProperties(**rock_data)

        saturation_history_group = group["saturation_history"]  # type: ignore
        saturation_history = SaturationHistory(
            max_water_saturation_grid=saturation_history_group[  # type: ignore[index]
                "max_water_saturation_grid"
            ][:],
            max_gas_saturation_grid=saturation_history_group["max_gas_saturation_grid"][  # type: ignore[index]
                :
            ][:],
            water_imbibition_flag_grid=saturation_history_group[  # type: ignore[index]
                "water_imbibition_flag_grid"
            ][:],
            gas_imbibition_flag_grid=saturation_history_group[  # type: ignore[index]
                "gas_imbibition_flag_grid"
            ][:],
        )

        thickness_grid = group["thickness_grid"][:]  # type: ignore
        grid_shape = tuple(group.attrs["grid_shape"])  # type: ignore
        cell_dimension = tuple(group.attrs["cell_dimension"])  # type: ignore
        dip_angle = typing.cast(float, group.attrs["dip_angle"])
        dip_azimuth = typing.cast(float, group.attrs["dip_azimuth"])

        return ReservoirModel(
            grid_shape=grid_shape,  # type: ignore
            cell_dimension=cell_dimension,  # type: ignore
            thickness_grid=thickness_grid,  # type: ignore
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            rock_fluid_properties=metadata["rock_fluid_properties"],
            saturation_history=saturation_history,
            boundary_conditions=metadata["boundary_conditions"],
            dip_angle=dip_angle,
            dip_azimuth=dip_azimuth,
        )

    def _load_rates(self, group: h5py.Group) -> RateGrids:
        """
        Load rates from HDF5 group.

        :param group: HDF5 group containing rate data
        :return: Reconstructed RateGrids instance
        """
        data = {}
        for key in group.keys():
            dataset = group[key]  # type: ignore
            if hasattr(dataset, "shape"):
                data[key] = dataset[:]  # type: ignore

        return RateGrids(**data)

    def _load_relperm(self, group: h5py.Group) -> RelPermGrids:
        """
        Load relative permeabilities from HDF5 group.

        :param group: HDF5 group containing relative permeability data
        :return: Reconstructed RelPermGrids instance
        """
        data = {}
        for key in group.keys():
            dataset = group[key]  # type: ignore
            if hasattr(dataset, "shape"):
                data[key] = dataset[:]  # type: ignore

        return RelPermGrids(**data)

    def _load_relative_mobilities(self, group: h5py.Group) -> RelativeMobilityGrids:
        """
        Load relative mobilities from HDF5 group.

        :param group: HDF5 group containing mobility data
        :return: Reconstructed RelativeMobilityGrids instance
        """
        data = {}
        for key in group.keys():
            dataset = group[key]  # type: ignore
            if hasattr(dataset, "shape"):
                data[key] = dataset[:]  # type: ignore

        return RelativeMobilityGrids(**data)

    def _load_capillary_pressures(self, group: h5py.Group) -> CapillaryPressureGrids:
        """
        Load capillary pressures from HDF5 group.

        :param group: HDF5 group containing capillary pressure data
        :return: Reconstructed CapillaryPressureGrids instance
        """
        data = {}
        for key in group.keys():
            dataset = group[key]  # type: ignore
            if hasattr(dataset, "shape"):
                data[key] = dataset[:]  # type: ignore

        return CapillaryPressureGrids(**data)

    def _load_timer_state(self, group: h5py.Group) -> TimerState:
        """
        Load timer state from HDF5 group.

        :param group: HDF5 group containing timer state
        :return: TimerState dict
        """
        timer_state: typing.Dict[str, typing.Any] = {}

        # Load scalar attributes
        for key in group.attrs.keys():
            timer_state[key] = group.attrs[key]

        # Load `recent_metrics` if present
        if "recent_metrics" in group:
            metrics_group = group["recent_metrics"]  # type: ignore
            recent_metrics: typing.List[StepMetricsDict] = []
            for metric_key in sorted(metrics_group.keys()):  # type: ignore
                metric_group = metrics_group[metric_key]  # type: ignore
                metric = StepMetricsDict(
                    step_number=typing.cast(int, metric_group.attrs["step_number"]),
                    step_size=typing.cast(float, metric_group.attrs["step_size"]),
                    cfl=typing.cast(float, metric_group.attrs["cfl"])
                    if metric_group.attrs.get("cfl") is not None
                    else None,
                    newton_iters=typing.cast(int, metric_group.attrs["newton_iters"])
                    if metric_group.attrs.get("newton_iters") is not None
                    else None,
                    success=typing.cast(bool, metric_group.attrs["success"]),
                )
                recent_metrics.append(metric)
            timer_state["recent_metrics"] = recent_metrics
        else:
            timer_state["recent_metrics"] = []

        # Load `failed_step_sizes` if present
        if "failed_step_sizes" in group:
            failed_dataset = group["failed_step_sizes"]  # type: ignore
            timer_state["failed_step_sizes"] = failed_dataset[:].tolist()  # type: ignore
        else:
            timer_state["failed_step_sizes"] = []

        return typing.cast(TimerState, timer_state)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(filepath={self.filepath}, "
            f"compression={self.compression}, compression_opts={self.compression_opts})"
        )


@state_store("npz")
class NPZStore(StateStore):
    """
    NumPy NPZ-based storage with hybrid array/pickle approach.

    Good compression, simple API, but loads everything into memory.
    Best for smaller datasets or when dependencies are limited.
    """

    def __init__(
        self, filepath: PathLike, metadata_dir: typing.Optional[PathLike] = None
    ):
        """
        Initialize the store

        :param filepath: Path to the NPZ file
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        self.filepath = _validate_filepath(filepath, expected_extension=".npz")
        self.metadata_dir = Path(
            metadata_dir or self.filepath.parent / f"{self.filepath.stem}_metadata"
        )

    def _ensure_metadata_dir(self) -> Path:
        """Ensure metadata directory exists for pickled objects."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        return self.metadata_dir

    def _dump_metadata(self, state: ModelState) -> None:
        """
        Store state metadata.

        :param state: `ModelState` containing the metadata
        """
        metadata_dir = self._ensure_metadata_dir()

        rf_path = metadata_dir / "rock_fluid_properties.pkl"
        bc_path = metadata_dir / "boundary_conditions.pkl"
        wells_path = metadata_dir / "wells.pkl"

        save_as_pickle(
            obj=state.model.rock_fluid_properties,
            filepath=rf_path,
            exist_ok=True,
            compression="gzip",
        )
        save_as_pickle(
            obj=state.model.boundary_conditions,
            filepath=bc_path,
            exist_ok=True,
            compression="gzip",
        )
        save_as_pickle(
            obj=state.wells, filepath=wells_path, exist_ok=True, compression="gzip"
        )
        logger.debug(f"Saved state metadata for step {state.step}")

    def _load_metadata(self, lazy: bool = False) -> StateMetadata:
        """
        Load state metadata.

        :param lazy: If True, returns Lazy objects that defer loading until accessed
        :return: StateMetadata dict with metadata
        """
        metadata_dir = self.metadata_dir
        if not metadata_dir.exists():
            raise StorageError(
                f"Metadata directory not found: {metadata_dir}. "
                "This NPZ store may be incomplete or corrupted."
            )

        rf_path = metadata_dir / "rock_fluid_properties.pkl.gz"
        bc_path = metadata_dir / "boundary_conditions.pkl.gz"
        wells_path = metadata_dir / "wells.pkl.gz"

        if not rf_path.exists() or not bc_path.exists() or not wells_path.exists():
            raise StorageError(
                f"Required metadata files not found in {metadata_dir}. "
                "Expected: rock_fluid_properties.pkl, boundary_conditions.pkl, wells.pkl"
            )

        if lazy:
            # Return lazy-loading factories
            rock_fluid_properties = Lazy.defer(lambda: load_pickle(filepath=rf_path))
            boundary_conditions = Lazy.defer(lambda: load_pickle(filepath=bc_path))
            wells = Lazy.defer(lambda: load_pickle(filepath=wells_path))
            logger.debug("Created lazy loaders for state metadata")
        else:
            # Load immediately
            rock_fluid_properties = load_pickle(filepath=rf_path)
            boundary_conditions = load_pickle(filepath=bc_path)
            wells = load_pickle(filepath=wells_path)
            logger.debug("Loaded state metadata")

        return StateMetadata(
            rock_fluid_properties=rock_fluid_properties,
            boundary_conditions=boundary_conditions,
            wells=wells,
        )

    def _dump_timer_state(
        self, save_dict: dict, prefix: str, timer_state: TimerState
    ) -> None:
        """
        Store timer state in NPZ save dictionary.

        :param save_dict: Dictionary to add timer state arrays to
        :param prefix: Prefix for array keys
        :param timer_state: Timer state to store
        """
        # Store all scalar/simple fields
        for key, value in timer_state.items():
            if key == "recent_metrics":
                # Store list of StepMetricsDict as flattened arrays
                metrics_list = typing.cast(typing.List[StepMetricsDict], value)
                save_dict[f"{prefix}timer_recent_metrics_count"] = np.array(
                    [len(metrics_list)]
                )
                if metrics_list:
                    step_numbers = [m["step_number"] for m in metrics_list]
                    step_sizes = [m["step_size"] for m in metrics_list]
                    cfls = [
                        m["cfl"] if m["cfl"] is not None else -1.0 for m in metrics_list
                    ]
                    newton_iters = [
                        m["newton_iters"] if m["newton_iters"] is not None else -1
                        for m in metrics_list
                    ]
                    successes = [m["success"] for m in metrics_list]

                    save_dict[f"{prefix}timer_recent_step_numbers"] = np.array(
                        step_numbers
                    )
                    save_dict[f"{prefix}timer_recent_step_sizes"] = np.array(step_sizes)
                    save_dict[f"{prefix}timer_recent_cfls"] = np.array(cfls)
                    save_dict[f"{prefix}timer_recent_newton_iters"] = np.array(
                        newton_iters
                    )
                    save_dict[f"{prefix}timer_recent_successes"] = np.array(successes)
            elif key == "failed_step_sizes":
                # Store as array (empty array if no failures)
                failed_sizes = typing.cast(typing.List[float], value)
                save_dict[f"{prefix}timer_failed_step_sizes"] = np.array(
                    failed_sizes if failed_sizes else []
                )
            else:
                # Store scalar values
                if value is not None:
                    save_dict[f"{prefix}timer_{key}"] = np.array([value])

    def _dump_model(self, save_dict: dict, prefix: str, model: ReservoirModel) -> None:
        """
        Store model properties in NPZ save dictionary.

        :param save_dict: Dictionary to add model arrays to
        :param prefix: Prefix for array keys
        :param model: ReservoirModel instance to store
        """
        # Store model metadata
        save_dict[f"{prefix}grid_shape"] = np.array(model.grid_shape)
        save_dict[f"{prefix}cell_dimension"] = np.array(model.cell_dimension)
        save_dict[f"{prefix}dip_angle"] = np.array([model.dip_angle])
        save_dict[f"{prefix}dip_azimuth"] = np.array([model.dip_azimuth])
        save_dict[f"{prefix}thickness_grid"] = model.thickness_grid

        # Store fluid properties
        fluid = model.fluid_properties
        for field in attrs.fields(fluid.__class__):
            value = getattr(fluid, field.name)
            if isinstance(value, np.ndarray):
                save_dict[f"{prefix}fluid_{field.name}"] = value

        # Store rock properties
        rock = model.rock_properties
        save_dict[f"{prefix}rock_compressibility"] = np.array([rock.compressibility])
        save_dict[f"{prefix}rock_perm_x"] = rock.absolute_permeability.x
        save_dict[f"{prefix}rock_perm_y"] = rock.absolute_permeability.y
        save_dict[f"{prefix}rock_perm_z"] = rock.absolute_permeability.z

        for field in attrs.fields(rock.__class__):
            if field.name not in ("compressibility", "absolute_permeability"):
                value = getattr(rock, field.name)
                if isinstance(value, np.ndarray):
                    save_dict[f"{prefix}rock_{field.name}"] = value

        # Store saturation history
        sat_hist = model.saturation_history
        save_dict[f"{prefix}sat_max_water"] = sat_hist.max_water_saturation_grid
        save_dict[f"{prefix}sat_max_gas"] = sat_hist.max_gas_saturation_grid
        save_dict[f"{prefix}sat_water_imb"] = sat_hist.water_imbibition_flag_grid
        save_dict[f"{prefix}sat_gas_imb"] = sat_hist.gas_imbibition_flag_grid

    def _dump_rates(self, save_dict: dict, prefix: str, rates: RateGrids) -> None:
        """
        Store injection/production rates in NPZ save dictionary.

        :param save_dict: Dictionary to add rate arrays to
        :param prefix: Prefix for array keys (e.g., 'step_000001_injection_')
        :param rates: RateGrids instance to store
        """
        for field in attrs.fields(rates.__class__):
            value = getattr(rates, field.name)
            if isinstance(value, np.ndarray):
                save_dict[f"{prefix}{field.name}"] = value

    def _dump_relperm(
        self, save_dict: dict, prefix: str, relperm: RelPermGrids
    ) -> None:
        """
        Store relative permeabilities in NPZ save dictionary.

        :param save_dict: Dictionary to add relperm arrays to
        :param prefix: Prefix for array keys
        :param relperm: RelPermGrids instance to store
        """
        for field in attrs.fields(relperm.__class__):
            value = getattr(relperm, field.name)
            if isinstance(value, np.ndarray):
                save_dict[f"{prefix}{field.name}"] = value

    def _dump_relative_mobilities(
        self, save_dict: dict, prefix: str, relative_mobilities: RelativeMobilityGrids
    ) -> None:
        """
        Store relative mobilities in NPZ save dictionary.

        :param save_dict: Dictionary to add mobility arrays to
        :param prefix: Prefix for array keys
        :param relative_mobilities: RelativeMobilityGrids instance to store
        """
        for field in attrs.fields(relative_mobilities.__class__):
            value = getattr(relative_mobilities, field.name)
            if isinstance(value, np.ndarray):
                save_dict[f"{prefix}{field.name}"] = value

    def _dump_capillary_pressures(
        self, save_dict: dict, prefix: str, capillary_pressures: CapillaryPressureGrids
    ) -> None:
        """
        Store capillary pressures in NPZ save dictionary.

        :param save_dict: Dictionary to add capillary pressure arrays to
        :param prefix: Prefix for array keys
        :param capillary_pressures: CapillaryPressureGrids instance to store
        """
        for field in attrs.fields(capillary_pressures.__class__):
            value = getattr(capillary_pressures, field.name)
            if isinstance(value, np.ndarray):
                save_dict[f"{prefix}{field.name}"] = value

    def dump(
        self,
        states: typing.Iterable[ModelState],
        exist_ok: bool = True,
        validate: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump states to compressed NPZ format.

        :param states: Iterable of ModelState instances to dump
        :param exist_ok: If True, will append to existing storage or create new
        :param validate: If True, validate each state before dumping
        """
        if not exist_ok and self.filepath.exists():
            raise StorageError(f"File {self.filepath} already exists")

        if validate:
            states_list = [validate_state(state=state) for state in states]
        else:
            states_list = list(states)

        if not states_list:
            logger.warning("No states to dump")
            return

        # Save pickled metadata once
        self._dump_metadata(state=states_list[0])

        # Load existing data if appending to existing file
        save_dict = {}
        if exist_ok and self.filepath.exists():
            try:
                existing_data = np.load(file=str(self.filepath))
                save_dict = {key: existing_data[key] for key in existing_data.files}
                logger.debug(
                    f"Loaded {len(existing_data.files)} existing arrays from {self.filepath}"
                )
            except Exception as exc:
                logger.warning(
                    f"Could not load existing NPZ file: {exc}. Creating new file."
                )
                save_dict = {}

        for state in states_list:
            prefix = f"step_{state.step:06d}_"

            # Store step metadata
            save_dict[f"{prefix}step"] = np.array([state.step])
            save_dict[f"{prefix}step_size"] = np.array([state.step_size])
            save_dict[f"{prefix}time"] = np.array([state.time])

            # Store timer state if present
            if state.timer_state is not None:
                self._dump_timer_state(
                    save_dict=save_dict, prefix=prefix, timer_state=state.timer_state
                )

            self._dump_model(save_dict=save_dict, prefix=prefix, model=state.model)
            self._dump_rates(
                save_dict=save_dict,
                prefix=f"{prefix}injection_",
                rates=state.injection,
            )
            self._dump_rates(
                save_dict=save_dict,
                prefix=f"{prefix}production_",
                rates=state.production,
            )
            self._dump_relperm(
                save_dict=save_dict,
                prefix=f"{prefix}relperm_",
                relperm=state.relative_permeabilities,
            )
            self._dump_relative_mobilities(
                save_dict=save_dict,
                prefix=f"{prefix}mobility_",
                relative_mobilities=state.relative_mobilities,
            )
            self._dump_capillary_pressures(
                save_dict=save_dict,
                prefix=f"{prefix}capillary_",
                capillary_pressures=state.capillary_pressures,
            )
            logger.debug(f"Prepared state at step {state.step}")

        np.savez_compressed(file=str(self.filepath), **save_dict)
        logger.debug(f"Completed dump of {len(states_list)} states to {self.filepath}")

    def load(
        self,
        lazy: bool = False,
        validate: bool = True,
        dtype: typing.Optional[np.typing.DTypeLike] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[ModelState, None, None]:
        """
        Load states from NPZ format.

        :param lazy: If True, defer loading of state properties until access.
            Wrap factory functions in Lazy.defer() for deferred loading.
            Note: The NPZ file stays open as long as any lazy-loaded state is in memory.
        :param validate: If True, validate each state after loading
        :param dtype: Optional dtype to coerce loaded arrays to. If None, uses global dtype.
                     Only applied when validate=True.
        :return: Generator yielding ModelState instances
        """
        # Determine dtype for validation
        if validate and dtype is None:
            dtype = get_dtype()

        metadata = self._load_metadata(lazy=lazy)

        data = np.load(file=str(self.filepath))

        steps = set()
        for key in data.keys():
            if key.startswith("step_"):
                step_str = key.split("_")[1]
                steps.add(int(step_str))

        for step_num in sorted(steps):
            prefix = f"step_{step_num:06d}_"

            step = int(data[f"{prefix}step"][0])
            step_size = float(data[f"{prefix}step_size"][0])
            time = float(data[f"{prefix}time"][0])

            # Load timer state if present
            timer_state: typing.Optional[TimerState] = None
            if (
                f"{prefix}timer_initial_step_size" in data
                or f"{prefix}timer_elapsed_time" in data
            ):
                timer_state = self._load_timer_state(data=data, prefix=prefix)

            if lazy:
                model = Lazy.defer(
                    lambda pfx=prefix: self._load_model(
                        data=data, prefix=pfx, metadata=metadata
                    )
                )
                injection = Lazy.defer(
                    lambda pfx=prefix: self._load_rates(
                        data=data, prefix=f"{pfx}injection_"
                    )
                )
                production = Lazy.defer(
                    lambda pfx=prefix: self._load_rates(
                        data=data, prefix=f"{pfx}production_"
                    )
                )
                relperm = Lazy.defer(
                    lambda pfx=prefix: self._load_relperm(
                        data=data, prefix=f"{pfx}relperm_"
                    )
                )
                relative_mobilities = Lazy.defer(
                    lambda pfx=prefix: self._load_relative_mobilities(
                        data=data, prefix=f"{pfx}mobility_"
                    )
                )
                capillary_pressures = Lazy.defer(
                    lambda pfx=prefix: self._load_capillary_pressures(
                        data=data, prefix=f"{pfx}capillary_"
                    )
                )
            else:
                model = self._load_model(data=data, prefix=prefix, metadata=metadata)
                injection = self._load_rates(data=data, prefix=f"{prefix}injection_")
                production = self._load_rates(data=data, prefix=f"{prefix}production_")
                relperm = self._load_relperm(data=data, prefix=f"{prefix}relperm_")
                relative_mobilities = self._load_relative_mobilities(
                    data=data, prefix=f"{prefix}mobility_"
                )
                capillary_pressures = self._load_capillary_pressures(
                    data=data, prefix=f"{prefix}capillary_"
                )

            state = ModelState(
                step=step,
                step_size=step_size,
                time=time,
                model=model,
                wells=metadata["wells"],
                injection=injection,
                production=production,
                relative_permeabilities=relperm,
                relative_mobilities=relative_mobilities,
                capillary_pressures=capillary_pressures,
                timer_state=timer_state,
            )

            if validate:
                state = validate_state(state=state, dtype=dtype)
            yield state

    def _load_model(
        self, data: typing.Any, prefix: str, metadata: StateMetadata
    ) -> ReservoirModel:
        """
        Load model from NPZ data.

        :param data: NPZ file data object
        :param prefix: Prefix for array keys
        :param metadata: State metadata containing rock_fluid_properties, etc.
        :return: Reconstructed ReservoirModel
        """
        grid_shape = tuple(data[f"{prefix}grid_shape"])  # type: ignore
        cell_dimension = tuple(data[f"{prefix}cell_dimension"])  # type: ignore
        dip_angle = float(data[f"{prefix}dip_angle"][0])
        dip_azimuth = float(data[f"{prefix}dip_azimuth"][0])
        thickness_grid = data[f"{prefix}thickness_grid"]

        # Load fluid properties
        fluid_arrays = {}
        for key in data.keys():
            if key.startswith(f"{prefix}fluid_"):
                field_name = key.replace(f"{prefix}fluid_", "")
                fluid_arrays[field_name] = data[key]
        fluid_properties = FluidProperties(**fluid_arrays)

        # Load rock properties
        rock_compressibility = float(data[f"{prefix}rock_compressibility"][0])
        perm_x = data[f"{prefix}rock_perm_x"]
        perm_y = data[f"{prefix}rock_perm_y"]
        perm_z = data[f"{prefix}rock_perm_z"]
        absolute_permeability = RockPermeability(x=perm_x, y=perm_y, z=perm_z)  # type: ignore

        rock_data = {
            "compressibility": rock_compressibility,
            "absolute_permeability": absolute_permeability,
        }
        for key in data.keys():
            if (
                key.startswith(f"{prefix}rock_")
                and not key.startswith(f"{prefix}rock_perm")
                and not key.endswith("compressibility")
            ):
                field_name = key.replace(f"{prefix}rock_", "")
                rock_data[field_name] = data[key]
        rock_properties = RockProperties(**rock_data)  # type: ignore

        # Load saturation history
        saturation_history = SaturationHistory(
            max_water_saturation_grid=data[f"{prefix}sat_max_water"],  # type: ignore
            max_gas_saturation_grid=data[f"{prefix}sat_max_gas"],  # type: ignore
            water_imbibition_flag_grid=data[f"{prefix}sat_water_imb"],  # type: ignore
            gas_imbibition_flag_grid=data[f"{prefix}sat_gas_imb"],  # type: ignore
        )

        return ReservoirModel(
            grid_shape=grid_shape,  # type: ignore
            cell_dimension=cell_dimension,  # type: ignore
            thickness_grid=thickness_grid,  # type: ignore
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            rock_fluid_properties=metadata["rock_fluid_properties"],
            saturation_history=saturation_history,
            boundary_conditions=metadata["boundary_conditions"],
            dip_angle=dip_angle,
            dip_azimuth=dip_azimuth,
        )

    def _load_rates(self, data: typing.Any, prefix: str) -> RateGrids:
        """
        Load rates from NPZ data.

        :param data: NPZ file data object
        :param prefix: Prefix for array keys
        :return: Reconstructed RateGrids
        """
        rate_data = {}
        for key in data.keys():
            if key.startswith(prefix):
                field_name = key.replace(prefix, "")
                rate_data[field_name] = data[key]
        return RateGrids(**rate_data)

    def _load_relperm(self, data: typing.Any, prefix: str) -> RelPermGrids:
        """
        Load relative permeabilities from NPZ data.

        :param data: NPZ file data object
        :param prefix: Prefix for array keys
        :return: Reconstructed RelPermGrids
        """
        relperm_data = {}
        for key in data.keys():
            if key.startswith(prefix):
                field_name = key.replace(prefix, "")
                relperm_data[field_name] = data[key]
        return RelPermGrids(**relperm_data)

    def _load_relative_mobilities(
        self, data: typing.Any, prefix: str
    ) -> RelativeMobilityGrids:
        """
        Load relative mobilities from NPZ data.

        :param data: NPZ file data object
        :param prefix: Prefix for array keys
        :return: Reconstructed RelativeMobilityGrids
        """
        mobility_data = {}
        for key in data.keys():
            if key.startswith(prefix):
                field_name = key.replace(prefix, "")
                mobility_data[field_name] = data[key]
        return RelativeMobilityGrids(**mobility_data)

    def _load_capillary_pressures(
        self, data: typing.Any, prefix: str
    ) -> CapillaryPressureGrids:
        """
        Load capillary pressures from NPZ data.

        :param data: NPZ file data object
        :param prefix: Prefix for array keys
        :return: Reconstructed CapillaryPressureGrids
        """
        capillary_data = {}
        for key in data.keys():
            if key.startswith(prefix):
                field_name = key.replace(prefix, "")
                capillary_data[field_name] = data[key]
        return CapillaryPressureGrids(**capillary_data)

    def _load_timer_state(self, data: typing.Any, prefix: str) -> TimerState:
        """
        Load timer state from NPZ data.

        :param data: NPZ file data object
        :param prefix: Prefix for array keys
        :return: Reconstructed timer state
        """
        timer_state: typing.Dict[str, typing.Any] = {}

        # Load all scalar attributes
        for key in data.keys():
            if (
                key.startswith(f"{prefix}timer_")
                and not key.startswith(f"{prefix}timer_recent_")
                and not key.startswith(f"{prefix}timer_failed_")
            ):
                field_name = key.replace(f"{prefix}timer_", "")
                value = data[key][0]
                # Convert numpy types to Python types
                if isinstance(value, np.integer):
                    timer_state[field_name] = int(value)
                elif isinstance(value, np.floating):
                    timer_state[field_name] = float(value)
                elif isinstance(value, np.bool_):
                    timer_state[field_name] = bool(value)
                else:
                    timer_state[field_name] = value

        # Reconstruct recent_metrics
        metrics_count_key = f"{prefix}timer_recent_metrics_count"
        if metrics_count_key in data:
            metrics_count = int(data[metrics_count_key][0])
            if metrics_count > 0:
                step_numbers = data[f"{prefix}timer_recent_step_numbers"]
                step_sizes = data[f"{prefix}timer_recent_step_sizes"]
                cfls = data[f"{prefix}timer_recent_cfls"]
                newton_iters = data[f"{prefix}timer_recent_newton_iters"]
                successes = data[f"{prefix}timer_recent_successes"]

                recent_metrics = []
                for i in range(metrics_count):
                    metric = StepMetricsDict(
                        step_number=int(step_numbers[i]),
                        step_size=float(step_sizes[i]),
                        cfl=float(cfls[i]) if cfls[i] >= 0 else None,
                        newton_iters=int(newton_iters[i])
                        if newton_iters[i] >= 0
                        else None,
                        success=bool(successes[i]),
                    )
                    recent_metrics.append(metric)
                timer_state["recent_metrics"] = recent_metrics
            else:
                timer_state["recent_metrics"] = []
        else:
            timer_state["recent_metrics"] = []

        # Load failed_step_sizes
        failed_key = f"{prefix}timer_failed_step_sizes"
        if failed_key in data:
            failed_step_sizes_array = data[failed_key]
            timer_state["failed_step_sizes"] = (
                failed_step_sizes_array.tolist()
                if len(failed_step_sizes_array) > 0
                else []
            )
        else:
            timer_state["failed_step_sizes"] = []

        return typing.cast(TimerState, timer_state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath})"


def new_store(
    backend: typing.Union[
        str, typing.Literal["zarr", "hdf5", "npz", "pickle"]
    ] = "zarr",
    *args: typing.Any,
    **kwargs: typing.Any,
) -> StateStore:
    """
    Create a new state storage.

    :param backend: Storage backend to use ('zarr', 'hdf5', 'npz', 'pickle', etc.)
    :param args: Additional positional arguments for the store constructor
    :param kwargs: Additional keyword arguments for the store constructor
    :return: An instance of the selected `StateStore` backend

    Example:
    ```python
    store = new_store('simulation.zarr', backend='zarr')
    store.dump(states)
    loaded = list(store.load())
    ```
    """
    if backend not in _STATE_STORES:
        raise ValidationError(
            f"Unknown backend: {backend}. Choose from {list(_STATE_STORES.keys())}"
        )

    store_class = _STATE_STORES[backend]
    return store_class(*args, **kwargs)
