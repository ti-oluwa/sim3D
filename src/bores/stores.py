"""State storage backends."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import functools
import logging
from os import PathLike
from pathlib import Path
import sys
import typing

import h5py  # type: ignore[import-untyped]
from numcodecs import Blosc
import numpy as np
import orjson
from typing_extensions import Self
from typing_extensions import ParamSpec
import yaml
import zarr  # type: ignore[import-untyped]
from zarr.storage import StoreLike  # type: ignore[import-untyped]

from bores.errors import StorageError, ValidationError
from bores.serialization import Serializable, SerializableT

__all__ = [
    "new_store",
    "storage_backend",
    "ZarrStore",
    "HDF5Store",
    "JSONStore",
    "YAMLStore",
]

IS_PYTHON_310_OR_LOWER = sys.version_info < (3, 11)

logger = logging.getLogger(__name__)


DataValidator = typing.Callable[[SerializableT], SerializableT]


class EntryMeta(typing.NamedTuple):
    """
    Lightweight record describing one persisted item.

    Stored alongside each entry so the store can answer index-based and
    predicate-based queries without deserialising any payload data.
    """

    idx: int
    """Zero-based position in insertion order."""
    group_name: str
    """Internal storage key (opaque to callers)."""
    meta: typing.Dict[str, str] = {}
    """JSON serializable metadata dictionary"""


class DataStore(typing.Generic[SerializableT], ABC):
    """
    Abstract base class for all storage backends.

    Every backend maintains a compact metadata index (``list[EntryMeta]``) so
    callers can inspect stored entries and jump directly to specific ones without
    a full scan.  Group naming is internal and fixed — callers never supply it.
    All writes overwrite existing content.

    **Interface**
    `dump(data)`
        Persist an iterable of `Serializable` items.  Always overwrites.

    `load(typ)`
        Load every item.  Returns a generator.

    `load(typ, indices=[0, 3, 7])`
        Load only the items at the given positional indices.

    `load(typ, predicate=lambda e: e.idx < 10)`
        Load only items whose `EntryMeta` satisfies *predicate*.

    `entries()`
        Return the full `list[EntryMeta]` without deserialising any payload.
        Use this for `count()`, `max_index()`, membership checks, etc.
    """

    supports_append: bool = False

    @abstractmethod
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        """
        Persist *data*, always overwriting any existing content in the store.

        Every item in *data* is written in iteration order.  If the backing
        file or directory already exists it is truncated first, so calling
        `dump` twice is equivalent to calling it once with the second dataset.
        Use `append` when you need to add items to an existing store without
        discarding what is already there.

        :param data: Iterable of `Serializable` instances to persist.
        :param validator: Optional callable applied to each item before it is
            written.  Receives the item and must return a (possibly transformed)
            item of the same type.  Raise to abort persistence of that item.
        :param meta: Optional callable that receives each item and returns a
            plain `dict` of JSON-serialisable values (str, int, float, bool).
            The returned dict is stored alongside the entry and surfaced on
            `EntryMeta.meta`, making it available for zero-deserialisation
            filtering in `load` and `entries`.
        """
        ...

    @abstractmethod
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        """
        Load and yield items from the store in insertion order.

        Filtering is applied before any array data is deserialised, so entries
        that do not match have no I/O cost beyond reading their metadata.
        When both `indices` and `predicate` are supplied, `indices` takes
        priority and `predicate` is ignored.

        :param typ: The `Serializable` subclass to deserialise each entry into.
        :param indices: If given, load only the entries at these zero-based
            insertion-order positions.  Out-of-range indices raise `IndexError`.
        :param predicate: If given (and `indices` is `None`), yield only entries
            for which `predicate(entry_meta)` returns `True`.  The predicate
            receives an `EntryMeta` instance and may inspect `entry_meta.meta`
            to filter on stored metadata without touching array data.
        :param validator: Optional callable applied to each deserialised item
            before it is yielded.  Receives the item and must return a
            (possibly transformed) item of the same type.
        :return: Generator yielding deserialised items matching the filter.
        """
        ...

    def append(
        self,
        item: SerializableT,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        """
        Append a single item to the store without rewriting existing entries.

        The item is assigned the next available insertion-order index.  Unlike
        `dump`, existing entries are never touched.  Backends that do not
        support append-style writes (e.g. `JSONStore`, `YAMLStore`) raise
        `NotImplementedError` at call time rather than at construction time;
        check `supports_append` before calling if in doubt.

        :param item: The `Serializable` instance to persist.
        :param validator: Optional callable applied to *item* before it is
            written.  Receives the item and must return a (possibly transformed)
            item of the same type.  Raise to abort the write.
        :param meta: Optional callable that receives *item* and returns a plain
            `dict` of JSON-serialisable values stored on `EntryMeta.meta`.
            Use this to record lightweight metadata (e.g. `{"step": state.step}`)
            that can later be used to filter entries via `load(predicate=...)`
            or `entries()` without deserialising array data.
        :return: The `EntryMeta` record created for the appended item, including
            its assigned index, group name, and any stored metadata.
        :raises NotImplementedError: If the backend does not support appending.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__!r} does not implement `append(...)`"
        )

    @abstractmethod
    def entries(self) -> typing.List[EntryMeta]:
        """
        Return metadata for every stored item in insertion order.

        This method must not deserialise any payload data.  Implementations
        should read only group names, file keys, and lightweight attributes —
        never array datasets.  The returned list can therefore be used for
        cheap introspection (counts, step lookups, predicate filtering) without
        triggering any significant I/O.

        :return: List of `EntryMeta` instances in insertion order.
        """
        ...

    def count(self) -> int:
        """
        Return the number of items currently stored.

        Delegates to `entries()` and returns its length.  No payload data is
        deserialised.

        :return: Total number of stored entries.
        """
        return len(self.entries())

    def max_index(self) -> typing.Optional[int]:
        """
        Return the highest insertion-order index in the store, or `None` if empty.

        Useful for targeting the last written entry via
        `store.load(typ, indices=[store.max_index()])` without replaying
        the entire store.  No payload data is deserialised.

        :return: The highest `EntryMeta.index` value, or `None` if the store
            contains no entries.
        """
        metas = self.entries()
        return max(e.idx for e in metas) if metas else None


StoreT = typing.TypeVar("StoreT", bound=DataStore)

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


def reraise_storage_error(func: typing.Callable[P, R]) -> typing.Callable[P, R]:
    """
    Wraps a function to raise `StorageError` on exceptions.

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


def _get_group_name(index: int) -> str:
    """
    Returns a group name using fixed naming scheme: `entry_{index:010d}`.

    Zero-padded to 10 digits so lexicographic order == insertion order,
    meaning `sorted(keys)` always gives the correct traversal order.
    """
    return f"entry_{index:010d}"


def _get_index_from_group_name(name: str) -> typing.Optional[int]:
    """Parse group name of form `entry_NNNNNNNNNN` to integer index, or `None` if not our format."""
    if name.startswith("entry_") and len(name) == 16:
        try:
            return int(name[6:])
        except ValueError:
            return None
    return None


@storage_backend("zarr")
class ZarrStore(DataStore[SerializableT]):
    """
    Zarr-based storage.

    Fast, efficient compression with lazy loading.
    Best for large 3D numpy arrays.
    Best lazy loading support among available formats.

    Layout:
    ```mermaid
    <root.zarr>/
        entry_0000000000/      ← one group per item
            <field>            ← zarr array  (numpy arrays)
            <nested>/          ← zarr subgroup (mappings / sequences of mappings)
                                    attrs hold scalars, strings, None sentinels
        entry_0000000001/
        ...
        (root attrs: count)
    ```
    """

    supports_append: bool = True

    def __init__(
        self,
        store: typing.Union[StoreLike, PathLike, str],
        compressor: typing.Literal["zstd", "lz4", "blosclz"] = "zstd",
        compression_level: int = 3,
        chunks: typing.Optional[typing.Tuple[int, ...]] = None,
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

    def _load_data(self, group: zarr.Group) -> typing.Dict[str, typing.Any]:
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
            data[key] = _normalize_loaded_value(array[:])  # type: ignore

        # Load subgroups
        for key in group.group_keys():  # type: ignore
            sub_group = typing.cast(zarr.Group, group[key])  # type: ignore
            loaded = self._load_data(group=sub_group)
            # Check if subgroup represents a sequence
            data[key] = _normalize_loaded_mapping_sequence(loaded)

        # Load attributes
        for attr_name in group.attrs.keys():
            data[attr_name] = _denormalize_from_storage(group.attrs[attr_name])

        return data

    def _open_root(self, mode: str) -> zarr.Group:
        return zarr.open_group(store=self.store, mode=mode, zarr_version=2)  # type: ignore[arg-type]

    def _write_entry(
        self,
        root: zarr.Group,
        index: int,
        item: SerializableT,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        group_name = _get_group_name(index)
        item_group = root.require_group(group_name)
        self._write_data(item_group, item.dump(recurse=True))

        item_group.attrs["_meta"] = meta(item) if meta is not None else {}
        item_group.attrs["_index"] = index
        item_group.attrs["_group_name"] = group_name
        return EntryMeta(idx=index, group_name=group_name)

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        root = self._open_root("w")  # always overwrite
        count = 0
        for index, item in enumerate(data):
            if validator is not None:
                item = validator(item)

            self._write_entry(root, index, item, meta)
            logger.debug(f"{self.__class__.__name__}: wrote entry {index}")
            count += 1
        root.attrs["count"] = count
        logger.debug(f"{self.__class__.__name__}: dump complete, {count} entries")

    @reraise_storage_error
    def append(
        self,
        item: SerializableT,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        root = self._open_root("a")
        current_count = int(root.attrs.get("count", 0))
        if validator is not None:
            item = validator(item)
        entry = self._write_entry(root, current_count, item, meta)
        root.attrs["count"] = current_count + 1
        logger.debug(f"{self.__class__.__name__}: appended entry {entry.idx}")
        return entry

    @reraise_storage_error
    def entries(self) -> typing.List[EntryMeta]:
        try:
            root = self._open_root("r")
        except Exception:
            return []

        metas = []
        for name in sorted(root.group_keys()):  # type: ignore[attr-defined]
            idx = _get_index_from_group_name(name)
            if idx is not None:
                group = root[name]
                metas.append(
                    EntryMeta(
                        idx=idx,
                        group_name=name,
                        meta=dict(group.attrs.get("_meta", {})),
                    )
                )
        return metas

    @reraise_storage_error
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        root = self._open_root("r")

        if indices is not None:
            index_set = set(indices)
            for name in sorted(root.group_keys()):  # type: ignore[attr-defined]
                idx = _get_index_from_group_name(name)
                if idx is not None and idx in index_set:
                    item_group = typing.cast(zarr.Group, root[name])  # type: ignore[index]
                    logger.debug(f"{self.__class__.__name__}: loading entry {idx}")
                    raw = self._load_data(item_group)
                    raw.pop("_index", None)
                    raw.pop("_group_name", None)
                    raw.pop("count", None)
                    raw.pop("version", None)
                    obj = typ.load(raw)
                    yield validator(obj) if validator is not None else obj
        else:
            for name in sorted(root.group_keys()):  # type: ignore[attr-defined]
                idx = _get_index_from_group_name(name)
                if idx is not None:
                    group = root[name]
                    entry_meta = EntryMeta(
                        idx=idx,
                        group_name=name,
                        meta=dict(group.attrs.get("_meta", {})),
                    )

                    if predicate is None or predicate(entry_meta):
                        item_group = typing.cast(
                            zarr.Group, root[entry_meta.group_name]
                        )  # type: ignore[index]
                        logger.debug(
                            f"{self.__class__.__name__}: loading entry {entry_meta.idx}"
                        )
                        raw = self._load_data(item_group)
                        raw.pop("_index", None)
                        raw.pop("_group_name", None)
                        raw.pop("count", None)
                        raw.pop("version", None)
                        obj = typ.load(raw)
                        yield validator(obj) if validator is not None else obj

    def __repr__(self) -> str:
        cname = getattr(self.compressor, "cname", str(self.compressor))
        return f"{self.__class__.__name__}(store={self.store!r}, compressor={cname!r})"


@storage_backend("hdf5", "h5")
class HDF5Store(DataStore[SerializableT]):
    """
    HDF5-based storage.

    Industry standard, good compression, wide tool support.
    Slightly slower than Zarr for many small writes.

    Layout:
    ```mermaid
    <file.h5>
        /entry_0000000000      ← one group per item
            <field>            ← dataset  (numpy arrays)
            <nested>/          ← subgroup (mappings / sequences of mappings)
                                    attrs hold scalars, strings, None sentinels
        /entry_0000000001
        ...
        (file attrs: count)
    ```
    """

    supports_append: bool = True

    def __init__(
        self,
        filepath: typing.Union[PathLike, str],
        compression: typing.Literal["gzip", "lzf", "szip"] = "gzip",
        compression_opts: int = 3,
        chunks: typing.Optional[typing.Tuple[int, ...]] = None,
    ):
        """
        Initialize the store

        :param filepath: Path to the HDF5 file
        :param compression: Compression algorithm - 'gzip', 'lzf', or 'szip'
        :param compression_opts: Compression level (1-9 for gzip)
        :param chunks: Custom chunk shape for datasets. If None, uses optimized defaults based on array dimensions.
        :raises StorageError: If filepath is invalid or has wrong extension
        """
        self.filepath = _validate_filepath(
            filepath, expected_extension=".h5", create_if_not_exists=True
        )
        self.compression = compression
        self.compression_opts = compression_opts  # 1-9 for gzip
        self.chunks = chunks

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

    def _create_dataset(self, group: h5py.Group, name: str, data: np.ndarray):
        """
        Create a compressed dataset with optimized chunking. Deletes the dataset if it already exists to recreate a new one.

        :param group: HDF5 group to create dataset in
        :param name: Name of the dataset
        :param data: NumPy array data to store
        :return: Created HDF5 dataset
        """
        if group.get(name) is not None:
            del group[name]

        chunks = self._get_chunks(shape=data.shape)
        return group.create_dataset(
            name=name,
            data=data,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=chunks if chunks is not None else True,
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

    def _write_entry(
        self,
        f: h5py.File,
        index: int,
        item: SerializableT,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        group_name = _get_group_name(index)
        item_group = f.require_group(group_name)
        self._write_data(item_group, item.dump(recurse=True))

        item_group.attrs["_meta"] = orjson.dumps(meta(item) if meta is not None else {})
        item_group.attrs["_index"] = index
        item_group.attrs["_group_name"] = group_name
        return EntryMeta(idx=index, group_name=group_name)

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        with h5py.File(str(self.filepath), "w") as f:  # always truncate
            count = 0
            for index, item in enumerate(data):
                if validator is not None:
                    item = validator(item)

                self._write_entry(f, index, item, meta)
                logger.debug(f"{self.__class__.__name__}: wrote entry {index}")
                count += 1
            f.attrs["count"] = count
        logger.debug(
            f"{self.__class__.__name__}: dump complete, {count} entries → {self.filepath}"
        )

    @reraise_storage_error
    def append(
        self,
        item: SerializableT,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> EntryMeta:
        mode = "a" if self.filepath.exists() else "w"
        with h5py.File(str(self.filepath), mode) as f:
            current_count = int(f.attrs.get("count", 0))
            if validator is not None:
                item = validator(item)

            entry = self._write_entry(f, current_count, item, meta)
            f.attrs["count"] = current_count + 1
        logger.debug(f"{self.__class__.__name__}: appended entry {entry.idx}")
        return entry

    @reraise_storage_error
    def entries(self) -> typing.List[EntryMeta]:
        if not self.filepath.exists():
            return []
        metas = []
        with h5py.File(str(self.filepath), "r") as f:
            for name in sorted(f.keys()):
                idx = _get_index_from_group_name(name)
                if idx is not None:
                    group = f[name]
                    metas.append(
                        EntryMeta(
                            idx=idx,
                            group_name=name,
                            meta=orjson.loads(group.attrs.get("_meta", "{}")),
                        )
                    )
        return metas

    @reraise_storage_error
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        with h5py.File(str(self.filepath), "r") as f:
            if indices is not None:
                index_set = set(indices)
                for name in sorted(f.keys()):
                    idx = _get_index_from_group_name(name)
                    if idx is not None and idx in index_set:
                        item_group = typing.cast(h5py.Group, f[name])
                        logger.debug(f"{self.__class__.__name__}: loading entry {idx}")
                        raw = self._load_data(item_group)
                        raw.pop("_index", None)
                        raw.pop("_group_name", None)
                        raw.pop("count", None)
                        obj = typ.load(raw)
                        yield validator(obj) if validator is not None else obj
            else:
                for name in sorted(f.keys()):
                    idx = _get_index_from_group_name(name)
                    if idx is not None:
                        group = f[name]
                        entry_meta = EntryMeta(
                            idx=idx,
                            group_name=name,
                            meta=orjson.loads(group.attrs.get("_meta", "{}")),
                        )

                        if predicate is None or predicate(entry_meta):
                            item_group = typing.cast(
                                h5py.Group, f[entry_meta.group_name]
                            )
                            logger.debug(
                                f"{self.__class__.__name__}: loading entry {entry_meta.idx}"
                            )
                            raw = self._load_data(item_group)
                            raw.pop("_index", None)
                            raw.pop("_group_name", None)
                            raw.pop("count", None)
                            obj = typ.load(raw)
                            yield validator(obj) if validator is not None else obj

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"filepath={self.filepath!r}, "
            f"compression={self.compression!r}, "
            f"compression_opts={self.compression_opts})"
        )


@storage_backend("json")
class JSONStore(DataStore[SerializableT]):
    """JSON-based storage.  Human-readable, no compression.  Good for configs."""

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

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        items = []
        for index, item in enumerate(data):
            if validator is not None:
                item = validator(item)
            items.append(
                {
                    "_index": index,
                    "_group_name": _get_group_name(index),
                    "_meta": meta(item) if meta is not None else {},
                    "data": item.dump(recurse=True),
                }
            )

        with open(self.filepath, "wb") as f:
            f.write(
                orjson.dumps(
                    items, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
                )
            )

    @reraise_storage_error
    def entries(self) -> typing.List[EntryMeta]:
        if not self.filepath.exists():
            return []

        with open(self.filepath, "rb") as f:
            items = orjson.loads(f.read())
        return [
            EntryMeta(
                idx=e["_index"],
                group_name=e["_group_name"],
                meta=e["_meta"],
            )
            for e in items
        ]

    @reraise_storage_error
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        with open(self.filepath, "rb") as f:
            items = orjson.loads(f.read())

        if indices is not None:
            index_set = set(indices)
            items = [e for e in items if e["_index"] in index_set]
        elif predicate is not None:
            items = [
                e
                for e in items
                if predicate(
                    EntryMeta(
                        idx=e["_index"],
                        group_name=e["_group_name"],
                        meta=e["_meta"],
                    )
                )
            ]

        for entry in items:
            obj = typ.load(entry["data"])
            yield validator(obj) if validator is not None else obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath!r})"


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

    @reraise_storage_error
    def dump(
        self,
        data: typing.Iterable[SerializableT],
        validator: typing.Optional[DataValidator[SerializableT]] = None,
        meta: typing.Optional[
            typing.Callable[[SerializableT], typing.Dict[str, typing.Any]]
        ] = None,
    ) -> None:
        items = []
        for index, item in enumerate(data):
            if validator is not None:
                item = validator(item)
            items.append(
                {
                    "_index": index,
                    "_group_name": _get_group_name(index),
                    "_meta": meta(item) if meta is not None else {},
                    "data": item.dump(recurse=True),
                }
            )
        with open(self.filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(items, f, sort_keys=False)

    @reraise_storage_error
    def entries(self) -> typing.List[EntryMeta]:
        if not self.filepath.exists():
            return []

        with open(self.filepath, "r", encoding="utf-8") as f:
            items = yaml.safe_load(f) or []
        return [
            EntryMeta(
                idx=e["_index"],
                group_name=e["_group_name"],
                meta=e["_meta"],
            )
            for e in items
        ]

    @reraise_storage_error
    def load(
        self,
        typ: typing.Type[SerializableT],
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        validator: typing.Optional[DataValidator[SerializableT]] = None,
    ) -> typing.Generator[SerializableT, None, None]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            items = yaml.safe_load(f) or []

        if indices is not None:
            index_set = set(indices)
            items = [e for e in items if e["_index"] in index_set]
        elif predicate is not None:
            items = [
                e
                for e in items
                if predicate(
                    EntryMeta(
                        idx=e["_index"],
                        group_name=e["_group_name"],
                        meta=e["_meta"],
                    )
                )
            ]

        for entry in items:
            obj = typ.load(entry["data"])
            yield validator(obj) if validator is not None else obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath!r})"


def new_store(
    backend: typing.Union[str, typing.Literal["zarr", "hdf5", "json", "yaml"]] = "zarr",
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
    store = new_store('zarr', 'simulation.zarr')
    store.dump(states)
    loaded = list(store.load())
    ```
    """
    if backend not in _STORAGE_BACKENDS:
        raise ValidationError(
            f"Unknown backend: {backend}. Available backends are: {list(_STORAGE_BACKENDS.keys())}"
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
        Load a `Serializable` instance from a `DataStore`.

        :param store: `DataStore` to load the `Serializable` from.
        :return: Loaded `Serializable` instance.
        """
        return next(iter(store.load(cls, **load_kwargs)), None)

    def to_store(self, store: DataStore[Self], **dump_kwargs: typing.Any) -> None:
        """
        Dump the `Serializable` instance to a `DataStore`.

        :param store: `DataStore` to dump the Serializable to.
        """
        store.dump([self], **dump_kwargs)

    @classmethod
    def from_file(
        cls, filepath: typing.Union[str, PathLike], **load_kwargs: typing.Any
    ) -> typing.Optional[Self]:
        """
        Load a `Serializable` instance from a file.

        :param filepath: Path to the file to load the `Serializable` from.
        :return: Loaded `Serializable` instance.
        """
        path = Path(filepath)
        ext = path.suffix.lower().lstrip(".")
        store = new_store(ext, path)
        return cls.from_store(store, **load_kwargs)

    def to_file(
        self, filepath: typing.Union[str, PathLike], **dump_kwargs: typing.Any
    ) -> None:
        """
        Dump the `Serializable` instance to a file.

        :param filepath: Path to the file to dump the `Serializable` to.
        """
        path = Path(filepath)
        ext = path.suffix.lower().lstrip(".")
        store = new_store(ext, path)
        self.to_store(store, **dump_kwargs)
