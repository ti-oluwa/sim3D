"""State storage backends for reservoir simulation states."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
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
import zarr
from zarr.storage import StoreLike

from bores.errors import StorageError, ValidationError
from bores.serialization import SerializableT
from bores.utils import Lazy, load_pickle, save_as_pickle

__all__ = [
    "new_store",
    "store",
    "ZarrStore",
    "PickleStore",
    "HDF5Store",
    "JSONStore",
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

_DATA_STORE_TYPES: typing.Dict[str, typing.Type[DataStore]] = {}


@typing.overload
def store(
    name: str,
    store_cls: typing.Type[StoreT],
) -> None:
    """Register a data store class with a given name."""
    ...


@typing.overload
def store(
    name: str,
) -> typing.Callable[[typing.Type[StoreT]], typing.Type[StoreT]]:
    """Register a data store class with a given name."""
    ...


def store(
    name: str,
    store_cls: typing.Optional[typing.Type[StoreT]] = None,
) -> typing.Union[None, typing.Callable[[typing.Type[StoreT]], typing.Type[StoreT]]]:
    """
    Data store registration decorator.

    Register a data store class with a given name.

    :param name: Name of the data store backend
    :param store_cls: Data store class to register
    :return: Decorator function if store_cls is None, else None
    """

    def _decorator(store_cls: typing.Type[StoreT]) -> typing.Type[StoreT]:
        _DATA_STORE_TYPES[name] = store_cls
        return store_cls

    if store_cls is not None:
        _DATA_STORE_TYPES[name] = store_cls
        return
    return _decorator


def _validate_filepath(
    filepath: typing.Union[PathLike, str],
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


@store("pickle")
class PickleStore(DataStore[SerializableT]):
    """
    Pickle-based storage.

    Python-native, simple, easy to use store.
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
        self.filepath = _validate_filepath(filepath, expected_extension=".pkl")
        self.compression = compression
        self.compression_level = compression_level

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


@store("zarr")
class ZarrStore(DataStore[SerializableT]):
    """
    Zarr-based storage with hybrid array/pickle approach.

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
        self.store = store
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
        Write data to a Zarr group.

        :param group: Zarr group to write data to
        :param data: Dictionary of data arrays to write
        """
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=key, data=value)
            elif isinstance(value, Mapping):
                sub_group = group.require_group(name=key)
                self._write_data(group=sub_group, data=value)
            elif isinstance(value, (np.integer, np.floating)):
                group.attrs[key] = value.item()
            elif isinstance(value, np.bool_):
                group.attrs[key] = bool(value)
            else:
                group.attrs[key] = value

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
        Load data from a Zarr group.

        :param group: Zarr group to load data from
        :param lazy: If True, returns Lazy objects that defer loading until accessed
        :return: Dictionary of loaded data
        """
        data: typing.Dict[str, typing.Any] = {}
        for key in group.array_keys():  # type: ignore
            array = group[key]  # type: ignore
            if lazy:
                data[key] = Lazy.defer(lambda arr=array: arr[:])  # type: ignore
            else:
                data[key] = array[:]  # type: ignore

        for key in group.group_keys():  # type: ignore
            sub_group = typing.cast(zarr.Group, group[key])  # type: ignore
            data[key] = self._load_data(group=sub_group, lazy=lazy)

        for attr_name in group.attrs.keys():
            data[attr_name] = group.attrs[attr_name]

        return data

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


@store("hdf5")
class HDF5Store(DataStore[SerializableT]):
    """
    HDF5-based storage with hybrid array/pickle approach.

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
        self.filepath = _validate_filepath(filepath, expected_extension=".h5")
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

    def _write_data(
        self, group: h5py.Group, data: typing.Mapping[str, typing.Any]
    ) -> None:
        """
        Write data to an HDF5 group.

        :param group: HDF5 group to write data to
        :param data: Dictionary of data arrays to write
        """
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                self._create_dataset(group=group, name=key, data=value)
            elif isinstance(value, Mapping):
                sub_group = group.require_group(name=key)
                self._write_data(group=sub_group, data=value)
            elif isinstance(value, (np.integer, np.floating)):
                group.attrs[key] = value.item()
            elif isinstance(value, np.bool_):
                group.attrs[key] = bool(value)
            else:
                group.attrs[key] = value

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
        Load data from an HDF5 group.

        :param group: HDF5 group to load data from
        :return: Dictionary of loaded data
        """
        data: typing.Dict[str, typing.Any] = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data[key] = item[:]  # type: ignore
            elif isinstance(item, h5py.Group):
                data[key] = self._load_data(group=item)
        for attr_name in group.attrs.keys():
            data[attr_name] = group.attrs[attr_name]
        return data

    def load(  # type: ignore[override]
        self,
        typ: typing.Type[SerializableT],
        lazy: bool = False,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        **kwargs: typing.Any,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load data instances from HDF5 format.

        :param typ: Type of the serializable objects to load
        :param lazy: If True, returns Lazy objects that defer loading until accessed
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


@store("json")
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
        self.filepath = _validate_filepath(filepath, expected_extension=".json")

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


def new_store(
    backend: typing.Union[
        str, typing.Literal["zarr", "hdf5", "json", "pickle"]
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
    if backend not in _DATA_STORE_TYPES:
        raise ValidationError(
            f"Unknown backend: {backend}. Choose from {list(_DATA_STORE_TYPES.keys())}"
        )

    store_class = _DATA_STORE_TYPES[backend]
    return store_class(*args, **kwargs)
