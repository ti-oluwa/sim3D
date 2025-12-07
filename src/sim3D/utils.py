import gzip
import lzma
from os import PathLike
from pathlib import Path
import pickle
import typing


def save_as_pickle(
    obj: typing.Any,
    filepath: PathLike,
    exist_ok: bool = False,
    compression: typing.Optional[typing.Literal["gzip", "lzma"]] = "gzip",
    compression_level: int = 6,
) -> None:
    """Saves an object as a pickle file with optional compression.

    :param obj: The object to be saved.
    :param filepath: The path to the pickle file.
    :param exist_ok: If True, will overwrite existing files.
    :param compression: Compression method - "gzip" (fast, good compression),
        "lzma" (slower, better compression), or None
    :param compression_level: Compression level (1-9 for gzip, 0-9 for lzma)
    """
    filepath = Path(filepath)
    if compression == "gzip":
        target_suffix = ".pkl.gz"
    elif compression == "lzma":
        target_suffix = ".pkl.xz"
    else:
        target_suffix = ".pkl"

    if filepath.suffix.split(".")[-1] not in ["pkl", "gz", "xz"]:
        filepath = filepath.with_suffix(target_suffix)
    elif not str(filepath).endswith(target_suffix):
        filepath = Path(str(filepath).replace(".pkl", target_suffix))

    if not exist_ok and filepath.exists():
        raise FileExistsError(f"File {filepath} already exists.")

    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True, mode=0o755)

    # Use pickle protocol 4 or 5 for better performance with large objects
    pickle_protocol = pickle.HIGHEST_PROTOCOL

    if compression == "gzip":
        with gzip.open(filepath, "wb", compresslevel=compression_level) as f:
            pickle.dump(obj, f, protocol=pickle_protocol)
    elif compression == "lzma":
        with lzma.open(filepath, "wb", preset=compression_level) as f:
            pickle.dump(obj, f, protocol=pickle_protocol)
    else:
        with filepath.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle_protocol)


def load_from_pickle(filepath: PathLike) -> typing.Any:
    """Loads an object from a pickle file with automatic compression detection.

    :param filepath: The path to the pickle file.
    :return: The loaded object.
    """
    filepath = Path(filepath)

    # Auto-detect compression from extension
    if str(filepath).endswith(".gz"):
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)
    elif str(filepath).endswith(".xz"):
        with lzma.open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        with filepath.open("rb") as f:
            return pickle.load(f)
