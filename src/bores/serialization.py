from collections.abc import Mapping, Sequence
from enum import Enum
import functools
import threading
import typing

import attrs
import cattrs
from typing_extensions import Self

from bores.errors import DeserializationError, SerializationError, ValidationError


__all__ = ["Serializable", "dump", "load"]

converter = cattrs.Converter()


def fallback_unstructure(value):
    return value


def fallback_structure(value, typ):
    return value


def _is_generic_alias(typ: typing.Any) -> bool:
    """Check if a type is a generic alias (e.g., List[int], Dict[str, float])"""
    return hasattr(typ, "__origin__") and typ.__origin__ is not None


@functools.lru_cache(maxsize=512)
def _get_origin_class(typ: typing.Any) -> typing.Optional[type]:
    """
    Extract the origin class from a generic type.

    Examples:
        RockPermeability[NDimension] -> RockPermeability
        List[int] -> list
        int -> int
    """
    if _is_generic_alias(typ):
        origin = typing.get_origin(typ)
        # For custom generic classes, get_origin returns the base class
        return origin
    elif isinstance(typ, type):
        return typ
    return None


@functools.lru_cache(maxsize=512)
def _is_serializable_type(typ: typing.Any) -> bool:
    """
    Check if a type (including generics) is a Serializable subclass.
    """
    origin = _get_origin_class(typ)
    if origin is None:
        return False

    try:
        return isinstance(origin, type) and issubclass(origin, Serializable)
    except TypeError:
        return False


def _is_optional_type(typ: typing.Any) -> bool:
    """Check if a type is Optional[T] (i.e., Union[T, None])."""
    if not _is_generic_alias(typ):
        return False

    origin = typing.get_origin(typ)
    if origin is not typing.Union:
        return False

    args = typing.get_args(typ)
    return type(None) in args


converter.register_unstructure_hook_func(
    check_func=lambda t: _is_generic_alias(t) or not attrs.has(t),
    func=fallback_unstructure,
)
converter.register_structure_hook_func(
    check_func=lambda t: _is_generic_alias(t) or not attrs.has(t),
    func=fallback_structure,
)


def _dump_value(value: typing.Any, recurse: bool):
    """Dump a value using cattrs, handling nested `Serializable` objects."""
    if isinstance(value, Serializable):
        return value.__dump__(recurse)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {k: _dump_value(v, recurse) for k, v in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_dump_value(v, recurse) for v in value]

    return converter.unstructure(value)


def _load_value(value: typing.Any, typ: typing.Type[typing.Any]):
    """Load a value using cattrs, handling nested `Serializable` objects."""
    if value is None:
        return None

    if _is_serializable_type(typ):
        return _get_origin_class(typ).__load__(value)  # type: ignore[attr-type]

    if isinstance(typ, type) and issubclass(typ, Enum):
        return typ(value)

    if _is_generic_alias(typ):
        origin = typing.get_origin(typ)
        args = typing.get_args(typ)

        if origin in (list, tuple, Sequence):
            return [_load_value(v, args[0]) for v in value]

        if origin in (dict, Mapping):
            return {k: _load_value(v, args[1]) for k, v in value.items()}

    return converter.structure(value, typ)


class SerializableMeta(type):
    """Metaclass for `Serializable` classes"""

    def __init__(
        cls,
        name: str,
        bases: typing.Tuple,
        namespace: typing.Dict[str, typing.Any],
        fields: typing.Optional[typing.Mapping[str, typing.Type]] = None,
        dump_exclude: typing.Optional[typing.Iterable[str]] = None,
        load_exclude: typing.Optional[typing.Iterable[str]] = None,
        serializers: typing.Optional[
            typing.Mapping[
                typing.Union[str, typing.Type],
                typing.Callable[[typing.Any, bool], typing.Any],
            ]
        ] = None,
        deserializers: typing.Optional[
            typing.Mapping[
                typing.Union[str, typing.Type],
                typing.Callable[[typing.Any], typing.Any],
            ]
        ] = None,
    ):
        super().__init__(name, bases, namespace)
        if (
            "__abstract_serializable__" in namespace
            and namespace["__abstract_serializable__"]
        ):
            return

        parent_serializers = {}
        parent_deserializers = {}
        parent_fields = {}
        for cl in cls.__mro__[:-1][::-1]:  # Exclude `object`
            if serializable_fields := getattr(cl, "__serializable_fields__", None):
                parent_fields.update(serializable_fields)

            if serializable_serializers := getattr(
                cl, "__serializable_serializers__", None
            ):
                parent_serializers.update(serializable_serializers)

            if serializable_deserializers := getattr(
                cl, "__serializable_deserializers__", None
            ):
                parent_deserializers.update(serializable_deserializers)

        cls_fields = fields or namespace.get("__annotations__", None)
        all_fields = {**parent_fields, **(cls_fields or {})}

        auto_serializers = cls._discover_type_serializers(all_fields)
        auto_deserializers = cls._discover_type_deserializers(all_fields)
        # Build final serializers/deserializers with proper precedence:
        # - Auto-discovered (lowest priority)
        # - Parent class (medium priority)
        # - Explicit on this class (highest priority)
        all_serializers = {
            **auto_serializers,
            **parent_serializers,
            **(serializers or {}),
        }
        all_deserializers = {
            **auto_deserializers,
            **parent_deserializers,
            **(deserializers or {}),
        }

        if not all_fields:
            raise ValidationError(
                "Serializable classes must have fields defined. If the class is an abstract base class, set `__abstract_serializable__` to True"
            )

        if not isinstance(all_fields, Mapping):
            raise ValidationError("`fields` must be a mapping of field names to types.")

        if "__dump__" not in namespace or getattr(
            namespace["__dump__"], "__is_placeholder__", False
        ):
            dumper = cls._build_default_dumper(
                fields=all_fields,
                exclude=dump_exclude,
                serializers=all_serializers,
            )
            cls.__dump__ = dumper

        if "__load__" not in namespace or getattr(
            namespace["__load__"], "__is_placeholder__", False
        ):
            loader = cls._build_default_loader(
                fields=all_fields,
                exclude=load_exclude,
                deserializers=all_deserializers,
            )
            cls.__load__ = loader

        cls.__serializable_fields__ = all_fields
        cls.__serializable_serializers__ = all_serializers
        cls.__serializable_deserializers__ = all_deserializers
        if "__abstract_serializable__" not in namespace:
            cls.__abstract_serializable__ = False

    @staticmethod
    def _discover_type_serializers(
        fields: typing.Mapping[str, typing.Type],
    ) -> typing.Dict[str, typing.Callable]:
        """
        Auto-discover serializers for fields based on their types.

        Looks up the global type serializer registry to find matching
        serializers for field types. Walks the MRO to support inheritance.
        """
        discovered = {}
        with _type_serializers_lock:
            for field_name, field_type in fields.items():
                # Get the origin class for generic types
                origin = _get_origin_class(field_type)
                if origin is None:
                    continue

                # Check if origin is actually a class (not a typing special form)
                if not isinstance(origin, type):
                    # For typing special forms like list, dict, try direct lookup
                    if origin in _TYPE_SERIALIZERS:
                        serializer = _TYPE_SERIALIZERS[origin]
                        discovered[field_name] = serializer
                    continue

                # Check if there's a registered serializer for this type
                # Walk the MRO to find base class serializers
                for base in origin.__mro__:
                    if base in _TYPE_SERIALIZERS:
                        serializer = _TYPE_SERIALIZERS[base]
                        discovered[field_name] = serializer
                        break

        return discovered

    @staticmethod
    def _discover_type_deserializers(
        fields: typing.Mapping[str, typing.Type],
    ) -> typing.Dict[str, typing.Callable]:
        """
        Auto-discover deserializers for fields based on their types.

        Looks up the global type deserializer registry to find matching
        deserializers for field types. Walks the MRO to support inheritance.
        """
        discovered = {}
        with _type_deserializers_lock:
            for field_name, field_type in fields.items():
                origin = _get_origin_class(field_type)
                if origin is None:
                    continue

                # Check if origin is actually a class (not a typing special form)
                if not isinstance(origin, type):
                    # For typing special forms like list, dict, try direct lookup
                    if origin in _TYPE_DESERIALIZERS:
                        deserializer = _TYPE_DESERIALIZERS[origin]
                        discovered[field_name] = deserializer
                    continue

                # Walk MRO to find deserializer
                for base in origin.__mro__:
                    if base in _TYPE_DESERIALIZERS:
                        deserializer = _TYPE_DESERIALIZERS[base]
                        discovered[field_name] = deserializer
                        break

        return discovered

    @staticmethod
    def _build_default_dumper(
        fields: typing.Mapping[str, typing.Type],
        exclude: typing.Optional[typing.Iterable[str]] = None,
        serializers: typing.Optional[
            typing.Mapping[
                typing.Union[str, typing.Type],
                typing.Callable[[typing.Any, bool], typing.Any],
            ]
        ] = None,
    ) -> typing.Callable:
        """
        Build a dumper function for the class.

        :param fields: Mapping of field names to types.
        :param exclude: Optional iterable of field names to exclude from dumping.
        :param serializers: Optional mapping of field names or types to custom serializer callables.
        :return: A dumper function.
        """

        def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
            result = {}
            for field, typ in fields.items():
                if exclude and field in exclude:
                    continue

                value = getattr(self, field)

                # Custom serializer by field name (highest priority)
                if serializers and (field in serializers):
                    serializer = serializers[field]
                    try:
                        result[field] = serializer(value, recurse)
                    except Exception as exc:
                        raise SerializationError(
                            f"Failed to serialize field '{field}' using custom serializer"
                        ) from exc
                    continue

                # Custom serializer by type (handle generics)
                if serializers:
                    origin = _get_origin_class(typ)
                    if origin and origin in serializers:
                        serializer = serializers[origin]
                        try:
                            result[field] = serializer(value, recurse)
                        except Exception as exc:
                            raise SerializationError(
                                f"Failed to serialize field '{field}' of type {origin} using custom serializer"
                            ) from exc
                        continue
                    elif typ in serializers:
                        serializer = serializers[typ]
                        try:
                            result[field] = serializer(value, recurse)
                        except Exception as exc:
                            raise SerializationError(
                                f"Failed to serialize field '{field}' of type {typ} using custom serializer"
                            ) from exc
                        continue

                # Nested `Serializable`
                if isinstance(value, Serializable):
                    try:
                        result[field] = value.__dump__(recurse)
                    except Exception as exc:
                        raise SerializationError(
                            f"Failed to serialize nested `Serializable` field '{field}'"
                        ) from exc
                else:
                    # Default cattrs unstructure
                    try:
                        result[field] = _dump_value(value, recurse)
                    except Exception as exc:
                        raise SerializationError(
                            f"Failed to unstructure field '{field}' of type {typ}"
                        ) from exc

            return result

        return __dump__

    @staticmethod
    def _build_default_loader(
        fields: typing.Mapping[str, typing.Type],
        exclude: typing.Optional[typing.Iterable[str]] = None,
        deserializers: typing.Optional[
            typing.Mapping[typing.Union[str, typing.Type], typing.Callable]
        ] = None,
    ) -> typing.Callable:
        """
        Build a loader function for the class.

        :param fields: Mapping of field names to types.
        :param exclude: Optional iterable of field names to exclude from loading.
        :param deserializers: Optional mapping of field names or types to custom deserializer callables.
        :return: A loader function.
        """

        @classmethod
        def __load__(cls, data: typing.Mapping[str, typing.Any]):
            init_kwargs = {}
            for field, typ in fields.items():
                if exclude and field in exclude:
                    continue

                # Field must exist in data (let __init__ handle defaults)
                if field not in data:
                    continue

                value = data[field]

                # Handle None for Optional types
                if value is None and _is_optional_type(typ):
                    init_kwargs[field] = None
                    continue

                # Custom deserializer by field name (highest priority)
                if deserializers and (field in deserializers):
                    deserializer = deserializers[field]
                    try:
                        init_kwargs[field] = deserializer(value)
                    except Exception as exc:
                        raise DeserializationError(
                            f"Failed to deserialize field '{field}' using custom deserializer"
                        ) from exc
                    continue

                # Custom deserializer by type (handle generics)
                if deserializers:
                    origin = _get_origin_class(typ)
                    if origin and origin in deserializers:
                        deserializer = deserializers[origin]
                        try:
                            init_kwargs[field] = deserializer(value)
                        except Exception as exc:
                            raise DeserializationError(
                                f"Failed to deserialize field '{field}' of type {origin} using custom deserializer"
                            ) from exc
                        continue
                    elif typ in deserializers:
                        deserializer = deserializers[typ]
                        try:
                            init_kwargs[field] = deserializer(value)
                        except Exception as exc:
                            raise DeserializationError(
                                f"Failed to deserialize field '{field}' of type {typ} using custom deserializer"
                            ) from exc
                        continue

                # Check if it's a `Serializable` (including generics)
                if _is_serializable_type(typ):
                    origin_cls = _get_origin_class(typ)
                    try:
                        init_kwargs[field] = origin_cls.__load__(value)  # type: ignore[attr-type]
                    except Exception as exc:
                        raise DeserializationError(
                            f"Failed to deserialize nested `Serializable` field '{field}' of type {typ}"
                        ) from exc
                else:
                    # Default cattrs structure
                    try:
                        init_kwargs[field] = _load_value(value, typ)
                    except Exception as exc:
                        raise DeserializationError(
                            f"Failed to structure field '{field}' of type {typ}"
                        ) from exc

            return cls(**init_kwargs)

        return __load__


class Serializable(metaclass=SerializableMeta):
    """Base class for serializable objects."""

    __abstract_serializable__ = True

    def __init_subclass__(cls, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init_subclass__()

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Dump the object to a dictionary."""
        raise NotImplementedError

    @classmethod
    def __load__(cls: typing.Type[Self], data: typing.Mapping[str, typing.Any]) -> Self:
        """Load an object from a mapping."""
        raise NotImplementedError

    __dump__.__is_placeholder__ = True  # type: ignore
    __load__.__is_placeholder__ = True  # type: ignore

    def dump(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return self.__dump__(recurse)

    @classmethod
    def load(cls, data: typing.Dict[str, typing.Any]) -> Self:
        return cls.__load__(data)


# Register `Serializable` with cattrs converter
def structure_serializable(
    data: typing.Mapping[str, typing.Any], cls: typing.Type[Serializable]
) -> Serializable:
    return cls.__load__(data)


def unstructure_serializable(obj: Serializable) -> typing.Mapping[str, typing.Any]:
    return obj.__dump__()


converter.register_structure_hook(Serializable, structure_serializable)
converter.register_unstructure_hook(Serializable, unstructure_serializable)


SerializableT = typing.TypeVar("SerializableT", bound=Serializable)
_SerializableT = typing.TypeVar("_SerializableT", bound=Serializable)


def dump(
    self, o: Serializable, /, recurse: bool = True
) -> typing.Dict[str, typing.Any]:
    """Dump a `Serializable` object to a dictionary."""
    return o.__dump__(recurse=recurse)


def load(
    cls: typing.Type[SerializableT], data: typing.Mapping[str, typing.Any]
) -> SerializableT:
    """Load a `Serializable` object from a dictionary."""
    return cls.__load__(data)


_TYPE_SERIALIZERS: typing.Dict[
    typing.Type[typing.Any],
    typing.Callable[[typing.Any, bool], typing.Dict[str, typing.Any]],
] = {}
_TYPE_DESERIALIZERS: typing.Dict[
    typing.Type[typing.Any],
    typing.Callable[[typing.Mapping[str, typing.Any]], typing.Any],
] = {}
_type_serializers_lock = threading.Lock()
_type_deserializers_lock = threading.Lock()


def make_serializable_type_registrar(
    base_cls: typing.Type[SerializableT],
    registry: typing.Dict[str, typing.Type[SerializableT]],
    key_attr: str = "__type__",
    lock: typing.Optional[threading.Lock] = None,
    key_factory: typing.Optional[
        typing.Callable[[typing.Type[SerializableT]], str]
    ] = None,
    allow_override: bool = False,
    auto_register_serializer: bool = True,
    auto_register_deserializer: bool = True,
) -> typing.Callable[[typing.Type[_SerializableT]], typing.Type[_SerializableT]]:
    """
    Decorator factory to create a registrar for `Serializable` subclasses.

    Creates a decorator to register `Serializable` subclasses in a registry.

    :param base_cls: The base class for the `Serializable` subclasses.
    :param registry: The registry to store the subclasses.
    :param key_attr: The attribute to use as the registry key.
    :param lock: An optional lock for thread safety.
    :param key_factory: An optional factory function to generate the registry key.
    :param allow_override: Whether to allow overriding existing registrations.
    :return: A decorator to register `Serializable` subclasses.
    """
    if not key_attr:
        raise ValueError("`key_attr` must be a non-empty string.")

    lock = lock or threading.Lock()

    def registrar(cls: typing.Type[_SerializableT]) -> typing.Type[_SerializableT]:
        """Decorator to register a `Serializable` subclass."""
        if not issubclass(cls, base_cls):
            raise ValidationError(
                f"Class {cls.__name__} is not a subclass of {base_cls.__name__}"
            )

        if getattr(cls, "__abstract_serializable__", False):
            return cls  # type: ignore[return-value]

        key = getattr(cls, key_attr, None)
        if key is None:
            key = key_factory(cls) if key_factory else cls.__name__.lower()
            setattr(cls, key_attr, key)

        if (
            not allow_override
            and key in registry
            and not issubclass(cls, registry[key])
        ):
            raise ValidationError(
                f"Class {cls.__name__} is already registered under key '{key}'. Rename the class or set a unique '{key_attr}'."
            )
        with lock:
            registry[key] = cls

        return cls  # type: ignore[return-value]

    if auto_register_serializer:
        serializer = make_registry_serializer(
            base_cls=base_cls,
            registry=registry,
            key_attr=key_attr,
        )
        register_type_serializer(
            typ=base_cls,
            serializer=serializer,
        )

    if auto_register_deserializer:
        deserializer = make_registry_deserializer(
            base_cls=base_cls,
            registry=registry,
        )
        register_type_deserializer(
            typ=base_cls,
            deserializer=deserializer,
        )

    registrar.__name__ = f"register_{base_cls.__name__.lower()}_type"
    return registrar


def make_registry_serializer(
    base_cls: typing.Type[SerializableT],
    registry: typing.Dict[str, typing.Type[SerializableT]],
    key_attr: str = "__type__",
) -> typing.Callable[[SerializableT, bool], typing.Dict[str, typing.Any]]:
    """
    Create a serializer function for a registry of `Serializable` subclasses.

    :param registry: The registry of `Serializable` subclasses.
    :param key_attr: The attribute used as the registry key.
    :return: A serializer function.
    """

    def serializer(
        obj: SerializableT, recurse: bool = True
    ) -> typing.Dict[str, typing.Any]:
        key = getattr(obj, key_attr, None)
        if not key or key not in registry:
            raise ValidationError(
                f"Unsupported {base_cls.__name__} type: {type(obj)!r}"
            )

        dump = {key: obj.dump(recurse)}
        return dump

    serializer.__name__ = f"serialize_{base_cls.__name__.lower()}"
    return serializer


def make_registry_deserializer(
    base_cls: typing.Type[SerializableT],
    registry: typing.Dict[str, typing.Type[SerializableT]],
) -> typing.Callable[[typing.Mapping[str, typing.Any]], SerializableT]:
    """
    Create a deserializer function for a registry of `Serializable` subclasses.

    :param registry: The registry of `Serializable` subclasses.
    :param key_attr: The attribute used as the registry key.
    :return: A deserializer function.
    """

    def deserializer(data: typing.Mapping[str, typing.Any]) -> SerializableT:
        if not isinstance(data, Mapping) or len(data) != 1:
            raise DeserializationError("Invalid data format for deserialization.")

        key, value = next(iter(data.items()))
        if key not in registry:
            raise DeserializationError(
                f"Unsupported {base_cls.__name__} type key: {key!r}"
            )

        cls = registry[key]
        return cls.load(value)

    deserializer.__name__ = f"deserialize_{base_cls.__name__.lower()}"
    return deserializer


def register_type_serializer(
    typ: typing.Type[SerializableT],
    serializer: typing.Callable[[SerializableT, bool], typing.Dict[str, typing.Any]],
) -> None:
    """Register a global type serializer for a specific type."""
    with _type_serializers_lock:
        _TYPE_SERIALIZERS[typ] = serializer


def register_type_deserializer(
    typ: typing.Type[SerializableT],
    deserializer: typing.Callable[[typing.Mapping[str, typing.Any]], SerializableT],
) -> None:
    """Register a global type deserializer for a specific type."""
    with _type_deserializers_lock:
        _TYPE_DESERIALIZERS[typ] = deserializer
