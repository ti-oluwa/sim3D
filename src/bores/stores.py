"""State storage backends."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import functools
import logging
from os import PathLike
from pathlib import Path
import sys
import typing
import uuid

import h5py
from numcodecs import Blosc
import numpy as np
import orjson
from typing_extensions import Self
from typing_extensions import ParamSpec
import yaml
import zarr
from zarr.storage import StoreLike

from bores.errors import StorageError, ValidationError
from bores.serialization import Serializable, SerializableT
from bores.utils import Lazy, load_pickle, save_as_pickle


__all__ = [
    "new_store",
    "storage_backend",
    "ZarrStore",
    "PickleStore",
    "HDF5Store",
    "JSONStore",
    "YAMLStore",
]

IS_PYTHON_310_OR_LOWER = sys.version_info < (3, 11)

logger = logging.getLogger(__name__)


class DataStore(typing.Generic[SerializableT], ABC):
    """Abstract base class for data storage classes."""

    supports_append: bool = False
    """Indicates if the store supports appending data."""

    @abstractmethod
    def load(
        self, typ: typing.Type[SerializableT], *args, **kwargs
    ) -> typing.Iterable[SerializableT]: ...

    @abstractmethod
    def dump(self, data: typing.Iterable[SerializableT], *args, **kwargs) -> None: ...


StoreT = typing.TypeVar("StoreT", bound=DataStore)
DataValidator = typing.Callable[[SerializableT], SerializableT]

_STORAGE_BACKENDS: typing.Dict[str, typing.Type[DataStore]] = {}


@typing.overload
def storage_backend(
    *names: str,
    store_cls: typing.Type[StoreT],
) -> typing.Type[StoreT]:
    """Register a data store class with a given name."""
    ...


@typing.overload
def storage_backend(
    *names: str,
) -> typing.Callable[[typing.Type[StoreT]], typing.Type[StoreT]]:
    """Register a data store class with a given name."""
    ...


def storage_backend(
    *names: str,
    store_cls: typing.Optional[typing.Type[StoreT]] = None,
) -> typing.Union[
    typing.Type[StoreT], typing.Callable[[typing.Type[StoreT]], typing.Type[StoreT]]
]:
    """
    Data store registration decorator.

    Register a data store class with a given name.

    :param name: Name of the data store backend
    :param store_cls: Data store class to register
    :return: Decorator function if store_cls is None, else None
    """

    def _decorator(store_cls: typing.Type[StoreT]) -> typing.Type[StoreT]:
        for name in names:
            _STORAGE_BACKENDS[name] = store_cls
        return store_cls

    if store_cls is not None:
        return _decorator(store_cls)
    return _decorator


def _validate_filepath(
    filepath: typing.Union[PathLike, str],
    expected_extension: typing.Optional[str] = None,
    is_directory: bool = False,
    create_if_not_exists: bool = False,
) -> Path:
    """
    Validate and normalize a filepath for state storage.

    :param filepath: Path to validate
    :param expected_extension: Expected file extension (e.g., '.pkl', '.h5', '.npz')
        If None, no extension validation is performed
    :param is_directory: If True, validates that the path is suitable for a directory
        (no extension or matches expected extension for directory-based stores)
    :param create_if_not_exists: If True, creates the file/directory if it does not exist
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

    if create_if_not_exists:
        # Ensure parent directory exists
        is_file = path.suffix != ""
        directory = path.parent if is_file else path
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created parent directory: {directory}")
            except Exception as exc:
                raise StorageError(
                    f"Failed to create parent directory '{directory}': {exc}"
                ) from exc

        if is_file and not path.exists():
            try:
                path.touch(exist_ok=True)
                logger.debug(f"Created file: {path}")
            except Exception as exc:
                raise StorageError(f"Failed to create file '{path}': {exc}") from exc
    return path


P = ParamSpec("P")
R = typing.TypeVar("R")


def _raise_storage_error(func: typing.Callable[P, R]) -> typing.Callable[P, R]:
    """
    Wraps a function to raise StorageError on exceptions.

    :param func: Function to wrap
    """

    @functools.wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            raise StorageError(exc) from exc

    return _wrapper


# Sentinel for None values to avoid object dtype issues
_NONE_SENTINEL = "__NONE_SENTINEL__"


def _is_none_sentinel(value: typing.Any) -> bool:
    """Check if value is the None sentinel."""
    return isinstance(value, str) and value == _NONE_SENTINEL


def _sequence_to_ndarray(value: Sequence, path: str) -> np.typing.NDArray:
    """
    Convert a (possibly nested) sequence into a NumPy array
    that is safe for HDF5/Zarr storage.

    Raises if the sequence would produce dtype=object.
    """
    if not value:
        # Empty to int8 empty array (safe default)
        return np.empty((0,), dtype=np.int8)

    # Check for mappings (should be stored as groups)
    if any(isinstance(v, Mapping) for v in value):
        raise TypeError(
            f"Sequence of mappings must be stored as groups, not datasets: {path}"
        )

    # Nested sequences
    if isinstance(value[0], Sequence) and not isinstance(value[0], (str, bytes)):
        arrays = [_sequence_to_ndarray(v, path) for v in value]
        try:
            arr = np.stack(arrays)
        except ValueError as exc:
            raise TypeError(f"Inconsistent nested sequence shapes at {path}") from exc
        return arr

    # Flat bool (check before numeric since bool is subclass of int)
    if all(isinstance(v, (bool, np.bool_)) for v in value):
        return np.asarray(value, dtype=bool)

    # Flat integer (pure int, no floats)
    if all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        for v in value
    ):
        return np.asarray(value)  # Preserves int32/int64 based on values

    # Flat numeric (mixed int/float or pure float)
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
        return np.asarray(value)  # Preserves dtype (float32/float64 based on input)

    # Flat strings (including None sentinels)
    if all(isinstance(v, str) for v in value):
        # For HDF5
        if hasattr(h5py, "string_dtype"):
            dtype = h5py.string_dtype(encoding="utf-8")
            return np.asarray(value, dtype=dtype)
        # For Zarr (doesn't have string_dtype)
        return np.asarray(value, dtype="U")

    raise TypeError(
        f"Unsupported or mixed sequence contents at {path}: "
        f"{set(type(v).__name__ for v in value)}"
    )


def _normalize_for_storage(value: typing.Any) -> typing.Any:
    """Normalize Python values for storage (replace None with sentinel)."""
    if value is None:
        return _NONE_SENTINEL
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, np.ndarray)
    ):
        return [_normalize_for_storage(v) for v in value]
    elif isinstance(value, Mapping):
        return {k: _normalize_for_storage(v) for k, v in value.items()}
    return value


def _denormalize_from_storage(value: typing.Any) -> typing.Any:
    """Denormalize values from storage (replace sentinel with None)."""
    if _is_none_sentinel(value):
        return None
    elif isinstance(value, list):
        return [_denormalize_from_storage(v) for v in value]
    elif isinstance(value, dict):
        return {k: _denormalize_from_storage(v) for k, v in value.items()}
    return value


def _normalize_loaded_value(value: typing.Any) -> typing.Any:
    """Normalize values loaded from HDF5/Zarr datasets."""
    if isinstance(value, np.ndarray):
        # String arrays
        if value.dtype.kind in ("U", "S", "O"):
            result = value.astype(str).tolist()
            return _denormalize_from_storage(result)
        # Numeric arrays
        return value
    return _denormalize_from_storage(value)


def _normalize_loaded_mapping_sequence(value: typing.Any) -> typing.Any:
    """
    Detect if a loaded mapping represents a sequence and convert it back.

    Groups with numeric string keys "0", "1", "2", ... are sequences.
    """
    if (
        isinstance(value, dict)
        and value
        and all(isinstance(k, str) and k.isdigit() for k in value.keys())
    ):
        # Reconstruct list in order
        max_idx = max(int(k) for k in value.keys())
        result = []
        for i in range(max_idx + 1):
            key = str(i)
            if key in value:
                result.append(value[key])
            else:
                # Missing index - shouldn't happen but handle gracefully
                result.append(None)
        return result
    return value


@storage_backend("pickle", "pkl")
class PickleStore(DataStore[SerializableT]):
    """
    Pickle-based storage.

    Python-native, simple, easy to use store. Does not support appending.
    """

    supports_append: bool = False

    def __init__(
        self,
        filepath: typing.Union[PathLike, str],
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
        self.filepath = _validate_filepath(
            filepath, expected_extension=".pkl", create_if_not_exists=True
        )
        self.compression = compression
        self.compression_level = compression_level

    @_raise_storage_error
    def dump(  # type: ignore[override]
        self,
        data: typing.Iterable[SerializableT],
        exist_ok: bool = True,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump states using pickle with compression.

        :param states: Iterable of serializable instances to dump
        :param exist_ok: If True, will overwrite existing files safely
        :param validator: Optional callable to validate/transform each item before dumping
        """
        if validator:
            data_list = [validator(item).dump(recurse=True) for item in data]
        else:
            data_list = [item.dump(recurse=True) for item in data]

        save_as_pickle(
            data_list,
            self.filepath,
            exist_ok=exist_ok,
            compression=self.compression,  # type: ignore
            compression_level=self.compression_level,
        )
        return

    @_raise_storage_error
    def load(  # type: ignore[override]
        self,
        typ: typing.Type[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load states from pickle file.

        :param typ: Type of the serializable objects to load
        :param validator: Optional callable to validate/transform each item after loading
        :return: Generator yielding instances of the specified type
        """
        data = load_pickle(self.filepath)
        if isinstance(data, dict):
            for item in data.values():
                obj = typ.load(item)
                if validator is not None:
                    yield validator(obj)
                else:
                    yield obj
        else:
            for item in data:
                obj = typ.load(item)
                if validator is not None:
                    yield validator(obj)
                else:
                    yield obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath}, compression={self.compression})"


@storage_backend("zarr")
class ZarrStore(DataStore[SerializableT]):
    """
    Zarr-based storage.

    Fast, efficient compression with lazy loading.
    Best for large 3D numpy arrays.
    Best lazy loading support among available formats.
    """

    supports_append: bool = True

    def __init__(
        self,
        store: typing.Union[StoreLike, PathLike, str],
        compressor: typing.Literal["zstd", "lz4", "blosclz"] = "zstd",
        compression_level: int = 3,
        chunks: typing.Optional[typing.Tuple[int, ...]] = None,
        group_name_gen: typing.Optional[
            typing.Callable[[int, SerializableT], str]
        ] = None,
    ):
        """
        Initialize the store

        :param store: Zarr store (file path, directory, or `Store` object)
        :param compressor: Compression algorithm - 'zstd', 'lz4', 'blosclz'. blosc with zstd is fastest for scientific data
        :param compression_level: Compression level (1-9)
        :param chunks: Chunk size for the Zarr arrays
        :param group_name_gen: Optional callable to generate group names based on index and item
        :raises StorageError: If filepath is invalid or has incompatible extension
        """
        self.store = (
            _validate_filepath(store, is_directory=True, create_if_not_exists=True)
            if isinstance(store, (str, PathLike))
            else store
        )
        self.chunks = chunks
        if IS_PYTHON_310_OR_LOWER:
            self.compressor = Blosc(
                cname=compressor,
                clevel=compression_level,
                shuffle=Blosc.BITSHUFFLE,
            )
        else:
            from zarr.codecs import BloscCodec, BloscShuffle  # type: ignore[import]

            self.compressor = BloscCodec(
                cname=compressor,
                clevel=compression_level,
                shuffle=BloscShuffle.bitshuffle,
            )
        self.group_name_gen = group_name_gen or self._default_group_name_gen

    @staticmethod
    def _default_group_name_gen(idx: int, item: SerializableT) -> str:
        """
        Default group name generator based on step number.

        :param step: Step number
        :param state: ModelState instance
        :return: Group name string
        """
        return f"item_{uuid.uuid4().hex[:12]}"

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
        chunks = self._get_chunks(shape=data.shape)
        return group.create_dataset(
            name=name,
            data=data,
            chunks=chunks,
            compressor=self.compressor,
            overwrite=True,
        )

    def _write_data(
        self, group: zarr.Group, data: typing.Mapping[str, typing.Any]
    ) -> None:
        """
        Write data to a Zarr group with nested structure support.

        :param group: Zarr group to write data to
        :param data: Dictionary of data to write
        """
        for key, value in data.items():
            # Normalize values
            value = _normalize_for_storage(value)

            # Mapping to subgroup
            if isinstance(value, Mapping):
                sub_group = group.require_group(name=key)
                self._write_data(group=sub_group, data=value)

            # NumPy array to dataset
            elif isinstance(value, np.ndarray):
                if value.dtype == object:
                    raise TypeError(
                        f"Zarr cannot store object-dtype arrays: {group.name}/{key}"
                    )
                self._create_dataset(group=group, name=key, data=value)

            # Python scalars to attributes
            elif isinstance(value, (np.integer, np.floating)):
                group.attrs[key] = value.item()
            elif isinstance(value, np.bool_):
                group.attrs[key] = bool(value)

            # Sequence to dataset or subgroup (depending on contents)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                # Empty sequence
                if not value:
                    self._create_dataset(
                        group=group, name=key, data=np.empty((0,), dtype=np.int8)
                    )
                    continue

                # Sequence of mappings to subgroup with indexed keys
                if isinstance(value[0], Mapping):
                    seq_group = group.require_group(key)
                    for i, item in enumerate(value):
                        item_group = seq_group.require_group(str(i))
                        self._write_data(group=item_group, data=item)
                    continue

                # Other sequences to dataset
                array = _sequence_to_ndarray(
                    value=value,
                    path=f"{group.name}/{key}",
                )
                self._create_dataset(group=group, name=key, data=array)

            # Simple scalars to attributes
            else:
                group.attrs[key] = value

    @_raise_storage_error
    def dump(  # type: ignore[override]
        self,
        data: typing.Iterable[SerializableT],
        exist_ok: bool = True,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump data to Zarr store with compression.

        :param data: Iterable of serializable instances to dump
        :param exist_ok: If True, will append to existing storage or create new
        :param validator: Optional callable to validate/transform each item before dumping
        """
        # Use 'a' (append) mode to reuse existing store, or 'w-' to fail if exists
        # This allows streaming to work properly without overwriting on each flush
        mode = "a" if exist_ok else "w-"
        root = zarr.open_group(
            store=self.store,  # type: ignore
            mode=mode,
            zarr_version=2,
        )

        count = 0
        for idx, item in enumerate(data):
            if validator is not None:
                item = validator(item)

            group_name = self.group_name_gen(idx, item)
            item_group = root.require_group(group_name)

            dump = item.dump(recurse=True)
            self._write_data(group=item_group, data=dump)
            logger.debug(f"Wrote item {idx} to group '{group_name}' in store")
            count += 1

        # Store global metadata
        root.attrs["version"] = 2
        root.attrs["count"] = count
        logger.debug(f"Completed dump of {count} items to {self.store}")

    def _load_data(
        self, group: zarr.Group, lazy: bool = False
    ) -> typing.Dict[str, typing.Any]:
        """
        Load data from a Zarr group with nested structure support.

        :param group: Zarr group to load data from
        :param lazy: If True, returns Lazy objects that defer loading until accessed
        :return: Dictionary of loaded data
        """
        data: typing.Dict[str, typing.Any] = {}

        # Load datasets
        for key in group.array_keys():  # type: ignore
            array = group[key]  # type: ignore
            if lazy:
                data[key] = Lazy.defer(  # type: ignore
                    lambda arr=array: _normalize_loaded_value(arr[:])
                )
            else:
                data[key] = _normalize_loaded_value(array[:])  # type: ignore

        # Load subgroups
        for key in group.group_keys():  # type: ignore
            sub_group = typing.cast(zarr.Group, group[key])  # type: ignore
            loaded = self._load_data(group=sub_group, lazy=lazy)
            # Check if subgroup represents a sequence
            data[key] = _normalize_loaded_mapping_sequence(loaded)

        # Load attributes
        for attr_name in group.attrs.keys():
            data[attr_name] = _denormalize_from_storage(group.attrs[attr_name])

        return data

    @_raise_storage_error
    def load(  # type: ignore[override]
        self,
        typ: typing.Type[SerializableT],
        lazy: bool = True,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load data instances from Zarr format.
        """
        root = zarr.open_group(store=self.store, mode="r", zarr_version=2)

        for key in sorted(root.group_keys()):
            item_group = typing.cast(zarr.Group, root[key])
            logger.debug(f"Loading item from group '{key}'")
            dump = self._load_data(group=item_group, lazy=lazy)
            dump.pop("count", None)  # Remove any stored count metadata
            dump.pop("version", None)  # Remove any stored version metadata
            obj = typ.load(dump)
            if validator is not None:
                yield validator(obj)
            else:
                yield obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(store={self.store}, compressor={self.compressor.cname})"


@storage_backend("hdf5", "h5")
class HDF5Store(DataStore[SerializableT]):
    """
    HDF5-based storage.

    Industry standard, good compression, wide tool support.
    Slightly slower than Zarr for many small writes.
    """

    supports_append: bool = True

    def __init__(
        self,
        filepath: typing.Union[PathLike, str],
        compression: typing.Literal["gzip", "lzf", "szip"] = "gzip",
        compression_opts: int = 3,
        group_name_gen: typing.Optional[
            typing.Callable[[int, SerializableT], str]
        ] = None,
    ):
        """
        Initialize the store

        :param filepath: Path to the HDF5 file
        :param compression: Compression algorithm - 'gzip', 'lzf', or 'szip'
        :param compression_opts: Compression level (1-9 for gzip)
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        self.filepath = _validate_filepath(
            filepath, expected_extension=".h5", create_if_not_exists=True
        )
        self.compression = compression
        self.compression_opts = compression_opts  # 1-9 for gzip
        self.group_name_gen = group_name_gen or self._default_group_name_gen

    @staticmethod
    def _default_group_name_gen(idx: int, item: SerializableT) -> str:
        """
        Default group name generator based on step number.

        :param step: Step number
        :param state: ModelState instance
        :return: Group name string
        """
        return f"item_{uuid.uuid4().hex[:12]}"

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

    def _write_data(self, group: h5py.Group, data: Mapping[str, typing.Any]) -> None:
        """
        Write data to an HDF5 group with nested structure support.

        Rules:
        - Mappings to subgroups (recursive)
        - Scalars to attributes
        - Sequences/arrays to datasets (never attributes)
        - None values to sentinel string
        """
        for key, value in data.items():
            # Normalize None values
            value = _normalize_for_storage(value)

            # Mapping to subgroup
            if isinstance(value, Mapping):
                sub_group = group.require_group(name=key)
                self._write_data(group=sub_group, data=value)

            # NumPy array to dataset
            elif isinstance(value, np.ndarray):
                if value.dtype == object:
                    raise TypeError(
                        f"HDF5 cannot store object-dtype arrays: {group.name}/{key}"
                    )
                self._create_dataset(group=group, name=key, data=value)

            # Python scalars to attributes
            elif isinstance(value, (np.integer, np.floating)):
                group.attrs[key] = value.item()
            elif isinstance(value, np.bool_):
                group.attrs[key] = bool(value)

            # Sequence to dataset or subgroup (depending on contents)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                # Empty sequence
                if not value:
                    self._create_dataset(
                        group=group, name=key, data=np.empty((0,), dtype=np.int8)
                    )
                    continue

                # Sequence of mappings to subgroup with indexed keys
                if isinstance(value[0], Mapping):
                    seq_group = group.require_group(key)
                    for i, item in enumerate(value):
                        item_group = seq_group.require_group(str(i))
                        self._write_data(group=item_group, data=item)
                    continue

                # Other sequences to dataset
                array = _sequence_to_ndarray(
                    value=value,
                    path=f"{group.name}/{key}",
                )
                self._create_dataset(group=group, name=key, data=array)

            # Simple scalars to attributes
            else:
                group.attrs[key] = value

    @_raise_storage_error
    def dump(  # type: ignore[override]
        self,
        data: typing.Iterable[SerializableT],
        exist_ok: bool = True,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump data to HDF5 store with compression.

        :param data: Iterable of serializable instances to dump
        :param exist_ok: If True, will append to existing storage or create new
        :param validator: Optional callable to validate/transform each item before dumping
        :raises StorageError: If unable to write to file
        """
        mode = "a" if exist_ok else "w-"

        with h5py.File(name=str(self.filepath), mode=mode) as f:
            count = 0
            for item in data:
                if validator is not None:
                    item = validator(item)

                group_name = self.group_name_gen(count, item)
                item_group = f.require_group(group_name)

                dump = item.dump(recurse=True)
                self._write_data(group=item_group, data=dump)
                logger.debug(f"Wrote item {count} to group '{group_name}' in store")
                count += 1

            # Store global metadata
            f.attrs["count"] = count
            logger.debug(f"Completed dump of {count} states to {self.filepath}")

    def _load_data(self, group: h5py.Group) -> typing.Dict[str, typing.Any]:
        """
        Load data from an HDF5 group with nested structure support.

        :param group: HDF5 group to load data from
        :return: Dictionary of loaded data
        """
        data: typing.Dict[str, typing.Any] = {}

        # Load datasets
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = _normalize_loaded_value(item[:])  # type: ignore
            elif isinstance(item, h5py.Group):
                loaded = self._load_data(group=item)
                # Check if subgroup represents a sequence
                data[key] = _normalize_loaded_mapping_sequence(loaded)

        # Load attributes
        for attr_name in group.attrs.keys():
            data[attr_name] = _denormalize_from_storage(group.attrs[attr_name])

        return data

    @_raise_storage_error
    def load(  # type: ignore[override]
        self,
        typ: typing.Type[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load data instances from HDF5 format.

        :param typ: Type of the serializable objects to load
        :param validator: Optional callable to validate/transform each item after loading
        :return: Generator yielding instances of the specified type
        """
        filepath = str(self.filepath)  # Capture for lazy closures

        with h5py.File(name=filepath, mode="r") as f:
            for key in sorted(f.keys()):
                item_group = typing.cast(h5py.Group, f[key])

                logger.debug(f"Loading item from group '{key}'")
                dump = self._load_data(group=item_group)
                dump.pop("count", None)  # Remove any stored count metadata
                obj = typ.load(dump)
                if validator is not None:
                    yield validator(obj)
                else:
                    yield obj

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(filepath={self.filepath}, "
            f"compression={self.compression}, compression_opts={self.compression_opts})"
        )


@storage_backend("json")
class JSONStore(DataStore[SerializableT]):
    supports_append: bool = False

    def __init__(
        self,
        filepath: typing.Union[PathLike, str],
    ):
        """
        Initialize the store

        :param filepath: Path to the JSON file
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        self.filepath = _validate_filepath(
            filepath, expected_extension=".json", create_if_not_exists=True
        )

    @_raise_storage_error
    def dump(  # type: ignore[override]
        self,
        data: typing.Iterable[SerializableT],
        exist_ok: bool = True,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump states using JSON.

        :param states: Iterable of serializable instances to dump
        :param exist_ok: If True, will overwrite existing files safely
        :param validator: Optional callable to validate/transform each item before dumping
        """
        data_list = []
        for item in data:
            if validator:
                item = validator(item)
            data_list.append(item.dump(recurse=True))

        with open(self.filepath, "wb") as f:
            f.write(
                orjson.dumps(
                    data_list, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
                )
            )
        return

    @_raise_storage_error
    def load(  # type: ignore[override]
        self,
        typ: typing.Type[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load states from JSON file.

        :param typ: Type of the serializable objects to load
        :param validator: Optional callable to validate/transform each item after loading
        :return: Generator yielding instances of the specified type
        """
        with open(self.filepath, "rb") as f:
            data = orjson.loads(f.read())

        for item in data:
            obj = typ.load(item)
            if validator is not None:
                yield validator(obj)
            else:
                yield obj


@storage_backend("yaml", "yml")
class YAMLStore(DataStore[SerializableT]):
    """
    YAML-based storage.

    Human-readable format, good for configs, small datasets, and debugging.
    """

    supports_append: bool = False

    def __init__(
        self,
        filepath: typing.Union[PathLike, str],
    ):
        """
        Initialize the store

        :param filepath: Path to the YAML file
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        self.filepath = _validate_filepath(
            filepath, expected_extension=".yaml", create_if_not_exists=True
        )

    @_raise_storage_error
    def dump(  # type: ignore[override]
        self,
        data: typing.Iterable[SerializableT],
        exist_ok: bool = True,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        Dump states using YAML.

        :param states: Iterable of serializable instances to dump
        :param exist_ok: If True, will overwrite existing files safely
        :param validator: Optional callable to validate/transform each item before dumping
        """
        data_list = []
        for item in data:
            if validator:
                item = validator(item)
            data_list.append(item.dump(recurse=True))

        with open(self.filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_list, f, sort_keys=False)
        return

    @_raise_storage_error
    def load(  # type: ignore[override]
        self,
        typ: typing.Type[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load states from YAML file.

        :param typ: Type of the serializable objects to load
        :param validator: Optional callable to validate/transform each item after loading
        :return: Generator yielding instances of the specified type
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for item in data:
            obj = typ.load(item)
            if validator is not None:
                yield validator(obj)
            else:
                yield obj


def new_store(
    backend: typing.Union[
        str, typing.Literal["zarr", "hdf5", "json", "pickle", "yaml"]
    ] = "zarr",
    *args: typing.Any,
    **kwargs: typing.Any,
) -> DataStore:
    """
    Create a new state storage.

    :param backend: Storage backend to use ('zarr', 'hdf5', 'npz', 'pickle', etc.)
    :param args: Additional positional arguments for the store constructor
    :param kwargs: Additional keyword arguments for the store constructor
    :return: An instance of the selected `DataStore` backend

    Example:
    ```python
    store = new_store('simulation.zarr', backend='zarr')
    store.dump(states)
    loaded = list(store.load())
    ```
    """
    if backend not in _STORAGE_BACKENDS:
        raise ValidationError(
            f"Unknown backend: {backend}. Choose from {list(_STORAGE_BACKENDS.keys())}"
        )

    store_class = _STORAGE_BACKENDS[backend]
    return store_class(*args, **kwargs)


class StoreSerializable(Serializable):
    """Serializable mixin with built-in store/file support."""

    __abstract_serializable__ = True

    @classmethod
    def from_store(
        cls, store: DataStore[Self], **load_kwargs: typing.Any
    ) -> typing.Optional[Self]:
        """
        Load a `Config` instance from a `DataStore`.

        :param store: `DataStore` to load the `Config` from.
        :return: Loaded `Config` instance.
        """
        return next(iter(store.load(cls, **load_kwargs)), None)

    def to_store(self, store: DataStore[Self], **dump_kwargs: typing.Any) -> None:
        """
        Dump the `Config` instance to a `DataStore`.

        :param store: `DataStore` to dump the Config to.
        """
        store.dump([self], **dump_kwargs)

    @classmethod
    def from_file(
        cls, filepath: typing.Union[str, PathLike], **load_kwargs: typing.Any
    ) -> typing.Optional[Self]:
        """
        Load a `Config` instance from a file.

        :param filepath: Path to the file to load the `Config` from.
        :return: Loaded `Config` instance.
        """
        path = Path(filepath)
        ext = path.suffix.lower().lstrip(".")
        store = new_store(ext, path)
        return cls.from_store(store, **load_kwargs)

    def to_file(
        self, filepath: typing.Union[str, PathLike], **dump_kwargs: typing.Any
    ) -> None:
        """
        Dump the `Config` instance to a file.

        :param filepath: Path to the file to dump the `Config` to.
        """
        path = Path(filepath)
        ext = path.suffix.lower().lstrip(".")
        store = new_store(ext, path)
        self.to_store(store, **dump_kwargs)
