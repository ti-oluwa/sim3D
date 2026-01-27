from collections.abc import Mapping, Sequence
from enum import Enum
import sys
import threading
import typing
import warnings

import attrs
import cattrs
from typing_extensions import Self

from bores.errors import DeserializationError, SerializationError, ValidationError


__all__ = ["Serializable", "dump", "load", "converter"]


_TYPE_SERIALIZERS: typing.Dict[
    typing.Type[typing.Any],
    typing.Callable[[typing.Any, bool], typing.Dict[str, typing.Any]],
] = {}
"""Registry of type serializers."""
_TYPE_DESERIALIZERS: typing.Dict[
    typing.Type[typing.Any],
    typing.Callable[[typing.Mapping[str, typing.Any]], typing.Any],
] = {}
"""Registry of type deserializers."""
_type_serializers_lock = threading.Lock()
_type_deserializers_lock = threading.Lock()


converter = cattrs.Converter()


def fallback_unstructure(value):
    return value


def fallback_structure(value, typ):
    return value


def _is_generic_alias(typ: typing.Any) -> bool:
    """Check if a type is a generic alias (e.g., List[int], Dict[str, float])"""
    return hasattr(typ, "__origin__") and typ.__origin__ is not None


converter.register_unstructure_hook_func(
    check_func=lambda t: _is_generic_alias(t) or not attrs.has(t),
    func=fallback_unstructure,
)
converter.register_structure_hook_func(
    check_func=lambda t: _is_generic_alias(t) or not attrs.has(t),
    func=fallback_structure,
)


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


def _is_typed_dict_type(typ: typing.Any) -> bool:
    return (
        isinstance(typ, type)
        and issubclass(typ, dict)
        and hasattr(typ, "__annotations__")
        and hasattr(typ, "__total__")
    )


def _is_namedtuple_type(typ: typing.Any) -> bool:
    return (
        isinstance(typ, type)
        and issubclass(typ, tuple)
        and hasattr(typ, "_fields")
        and isinstance(typ._fields, tuple)  # type: ignore[attr-defined]
    )


def _unwrap_type(typ: typing.Any) -> typing.List[typing.Type[typing.Any]]:
    """
    Recursively unwrap a type to get all constituent non-None types.

    Examples:
        Optional[Foo] -> [Foo]
        Union[Foo, Bar, None] -> [Foo, Bar]
        List[Optional[Foo]] -> [List[Foo], Foo]
        Dict[str, Optional[Foo]] -> [Dict[str, Foo], Foo]
        Optional[List[Foo]] -> [List[Foo], Foo]
        Union[List[Foo], Dict[str, Bar]] -> [List[Foo], Dict[str, Bar], Foo, Bar]
    """
    result = []

    if typ is type(None):
        return []

    origin = typing.get_origin(typ)
    args = typing.get_args(typ)

    # Handle Union types (including Optional)
    if origin is typing.Union:
        for arg in args:
            if arg is not type(None):
                result.extend(_unwrap_type(arg))
        return result

    # For generic types, add the container first
    if origin is not None and args:
        result.append(typ)

        # Only unwrap nested custom types, not primitives
        for arg in args:
            arg_origin = _get_origin_class(arg)
            # Skip primitives like str, int, float
            if arg_origin and not _is_primitive_type(arg_origin):
                result.extend(_unwrap_type(arg))

        return result

    # Base case: non-generic type
    result.append(typ)
    return result


def _is_primitive_type(typ: type) -> bool:
    """Check if a type is a primitive built-in type."""
    return typ in (str, int, float, bool, bytes, type(None))


def _get_primary_types(typ: typing.Any) -> typing.List[typing.Type[typing.Any]]:
    """
    Get the primary types to check for serializers/deserializers.

    This extracts the outermost meaningful types after unwrapping Unions/Optional,
    plus any deeply nested types that might have custom serializers.

    Examples:
        Optional[Foo] -> [Foo]
        List[Foo] -> [List[Foo], Foo]
        Dict[str, Foo] -> [Dict[str, Foo], Foo]
        Optional[List[Foo]] -> [List[Foo], Foo]
        Union[Foo, Bar] -> [Foo, Bar]
    """
    all_types = _unwrap_type(typ)

    # Remove duplicates while preserving order
    seen = set()
    unique_types = []
    for t in all_types:
        # Create a hashable representation
        type_id = id(t)
        if type_id not in seen:
            seen.add(type_id)
            unique_types.append(t)

    return unique_types


def _discover_type_serializers(
    fields: typing.Mapping[str, typing.Type[typing.Any]],
) -> typing.Dict[str, typing.Callable[[typing.Any, bool], typing.Any]]:
    """
    Auto-discover serializers for fields based on their types.

    Unwraps Optional, Union, and other generic containers to find
    all types that might need custom serializers.
    """
    discovered = {}

    with _type_serializers_lock:
        for field_name, field_type in fields.items():
            # Get all types to check (unwrapping Optional/Union and extracting generics)
            types_to_check = _get_primary_types(field_type)

            # Check each type in order of specificity (most specific first)
            for typ in types_to_check:
                origin = _get_origin_class(typ)
                if origin is None:
                    continue

                # Handle typing special forms (list, dict, etc.)
                if not isinstance(origin, type):
                    if origin in _TYPE_SERIALIZERS:
                        discovered[typ] = _TYPE_SERIALIZERS[origin]
                        break
                    continue

                # Walk MRO for class-based types
                for base in origin.__mro__:
                    if base in _TYPE_SERIALIZERS:
                        discovered[typ] = _TYPE_SERIALIZERS[base]
                        break
                else:
                    # Continue to next type if no serializer found
                    continue

                # Break outer loop if we found a serializer
                break

    return discovered


def _discover_type_deserializers(
    fields: typing.Mapping[str, typing.Type[typing.Any]],
) -> typing.Dict[str, typing.Callable[[typing.Any], typing.Any]]:
    """
    Auto-discover deserializers for fields based on their types.

    Unwraps Optional, Union, and other generic containers to find
    all types that might need custom deserializers.
    """
    discovered = {}

    with _type_deserializers_lock:
        for field_name, field_type in fields.items():
            # Get all types to check (unwrapping Optional/Union and extracting generics)
            types_to_check = _get_primary_types(field_type)

            # Check each type in order of specificity (most specific first)
            for typ in types_to_check:
                origin = _get_origin_class(typ)
                if origin is None:
                    continue

                # Handle typing special forms (list, dict, etc.)
                if not isinstance(origin, type):
                    if origin in _TYPE_DESERIALIZERS:
                        discovered[typ] = _TYPE_DESERIALIZERS[origin]
                        break
                    continue

                # Walk MRO to find deserializer
                for base in origin.__mro__:
                    if base in _TYPE_DESERIALIZERS:
                        discovered[typ] = _TYPE_DESERIALIZERS[base]
                        break
                else:
                    # Continue to next type if no deserializer found
                    continue

                # Break outer loop if we found a deserializer
                break

    return discovered


def _dump_value(
    value: typing.Any,
    recurse: bool,
    serializers: typing.Optional[
        typing.Mapping[
            typing.Union[str, typing.Type],
            typing.Callable[[typing.Any, bool], typing.Any],
        ]
    ] = None,
    typ: typing.Optional[typing.Type[typing.Any]] = None,
):
    """Dump a value using cattrs, handling nested `Serializable` objects."""
    typ = typ or type(value)
    if serializers and typ in serializers:
        return serializers[typ](value, True)

    if value is None:
        return None

    if isinstance(value, Serializable):
        return value.dump(recurse)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {k: _dump_value(v, recurse, serializers) for k, v in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_dump_value(v, recurse, serializers) for v in value]

    return converter.unstructure(value)


def _load_value(
    value: typing.Any,
    typ: typing.Type[typing.Any],
    deserializers: typing.Optional[
        typing.Mapping[
            typing.Union[str, typing.Type],
            typing.Callable[[typing.Any], typing.Any],
        ]
    ] = None,
) -> typing.Any:
    """Load a value using cattrs, handling nested `Serializable` objects."""
    if deserializers and typ in deserializers:
        return deserializers[typ](value)

    if value is None:
        return None

    if _is_serializable_type(typ):
        return _get_origin_class(typ).load(value)  # type: ignore[attr-type]

    if isinstance(typ, type) and issubclass(typ, Enum):
        return typ(value)

    if _is_optional_type(typ):
        args = typing.get_args(typ)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _load_value(value, non_none_args[0], deserializers)
        else:
            for arg in non_none_args:
                try:
                    return _load_value(value, arg, deserializers)
                except Exception:
                    continue
            raise DeserializationError(
                f"Value {value!r} does not match any type in {typ}"
            )

    if _is_generic_alias(typ):
        origin = typing.get_origin(typ)
        args = typing.get_args(typ)

        if origin in (list, tuple, Sequence) or (
            origin and isinstance(origin, type) and issubclass(origin, Sequence)
        ):
            return [_load_value(v, args[0], deserializers) for v in value]

        if origin in (dict, Mapping) or (
            origin and isinstance(origin, type) and issubclass(origin, Mapping)
        ):
            return {k: _load_value(v, args[1], deserializers) for k, v in value.items()}

    if _is_typed_dict_type(typ):
        annotations = typing.get_type_hints(typ, include_extras=False)
        return typ(
            {k: _load_value(v, annotations[k], deserializers) for k, v in value.items()}
        )

    if _is_namedtuple_type(typ):
        annotations = typing.get_type_hints(typ, include_extras=False)
        return typ(
            **{
                k: _load_value(v, annotations[k], deserializers)
                for k, v in value.items()
            }
        )

    return converter.structure(value, typ)


def _build_dumper(
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
                # Try exact type match first
                if typ in serializers:
                    serializer = serializers[typ]
                    try:
                        result[field] = serializer(value, recurse)
                    except Exception as exc:
                        raise SerializationError(
                            f"Failed to serialize field '{field}' of type {typ} using custom serializer"
                        ) from exc
                    continue

                # Then try origin class for generics
                origin = _get_origin_class(typ)
                if origin and origin != typ and origin in serializers:
                    serializer = serializers[origin]
                    try:
                        result[field] = serializer(value, recurse)
                    except Exception as exc:
                        raise SerializationError(
                            f"Failed to serialize field '{field}' of type {origin} using custom serializer"
                        ) from exc
                    continue

            # Nested `Serializable`
            if isinstance(value, Serializable):
                try:
                    result[field] = value.dump(recurse)
                except Exception as exc:
                    raise SerializationError(
                        f"Failed to serialize nested `Serializable` field '{field}'"
                    ) from exc
            else:
                # Default cattrs unstructure
                try:
                    result[field] = _dump_value(value, recurse, serializers, typ)
                except Exception as exc:
                    raise SerializationError(
                        f"Failed to unstructure field '{field}' of type {typ}"
                    ) from exc

        return result

    return __dump__


def _build_loader(
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
                # Check exact type match first
                if typ in deserializers:
                    deserializer = deserializers[typ]
                    try:
                        init_kwargs[field] = deserializer(value)
                    except Exception as exc:
                        raise DeserializationError(
                            f"Failed to deserialize field '{field}' of type {typ} using custom deserializer"
                        ) from exc
                    continue

                # Then check origin class for generics
                origin = _get_origin_class(typ)
                if origin and origin != typ and origin in deserializers:
                    deserializer = deserializers[origin]
                    try:
                        init_kwargs[field] = deserializer(value)
                    except Exception as exc:
                        raise DeserializationError(
                            f"Failed to deserialize field '{field}' of type {origin} using custom deserializer"
                        ) from exc
                    continue

            # Check if it's a `Serializable` (including generics)
            if _is_serializable_type(typ):
                origin_cls = _get_origin_class(typ)
                try:
                    init_kwargs[field] = origin_cls.load(value)  # type: ignore[attr-defined]
                except Exception as exc:
                    raise DeserializationError(
                        f"Failed to deserialize nested `Serializable` field '{field}' of type {typ}"
                    ) from exc
            else:
                # Default cattrs structure
                try:
                    init_kwargs[field] = _load_value(value, typ, deserializers)
                except Exception as exc:
                    raise DeserializationError(
                        f"Failed to structure field '{field}' of type {typ}"
                    ) from exc

        return cls(**init_kwargs)

    return __load__


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
        parent_serializers = {}
        parent_deserializers = {}
        parent_fields = {}
        for cl in reversed(cls.__mro__[1:-1]):  # Exclude `object`
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

        try:
            module = sys.modules.get(cls.__module__)
            annotations = typing.get_type_hints(
                cls,
                globalns=vars(module),
                localns=dict(vars(cls)),
                include_extras=False,
            )
        except NameError as exc:
            annotations = namespace.get("__annotations__", {})
            warnings.warn(
                f"Could not resolve type hints for {cls.__name__}: {exc}. "
                f"Using raw annotations which may not work with forward references.",
                RuntimeWarning,
            )

        cls_fields = fields or annotations
        all_fields = {**parent_fields, **cls_fields}
        # Clean fields: remove any with value `None` or starting with dunder
        all_fields = {
            k: v
            for k, v in all_fields.items()
            if v is not None and not k.startswith("__")
        }

        type_serializers = _discover_type_serializers(all_fields)
        type_deserializers = _discover_type_deserializers(all_fields)
        # Build final serializers/deserializers with proper precedence
        # - Auto-discovered (lowest priority)
        # - Parent class (medium priority)
        # - Explicit on this class (highest priority)
        all_serializers = {
            **type_serializers,
            **parent_serializers,
            **(serializers or {}),
        }
        all_deserializers = {
            **type_deserializers,
            **parent_deserializers,
            **(deserializers or {}),
        }

        is_abstract_cls = (
            "__abstract_serializable__" in namespace
            and namespace["__abstract_serializable__"]
        )
        if is_abstract_cls is False and not all_fields:
            raise ValidationError(
                "Serializable classes must have fields defined. If the class is an abstract base class, "
                "set `__abstract_serializable__` to True"
            )

        if all_fields:
            if "__dump__" not in namespace or getattr(
                namespace["__dump__"], "_is_placeholder", False
            ):
                cls.__dump__ = _build_dumper(
                    fields=all_fields,
                    exclude=dump_exclude,
                    serializers=all_serializers,
                )

            if "__load__" not in namespace or getattr(
                namespace["__load__"], "_is_placeholder", False
            ):
                cls.__load__ = _build_loader(
                    fields=all_fields,
                    exclude=load_exclude,
                    deserializers=all_deserializers,
                )

        cls.__serializable_fields__ = all_fields
        cls.__serializable_serializers__ = all_serializers
        cls.__serializable_deserializers__ = all_deserializers
        if "__abstract_serializable__" not in namespace:
            cls.__abstract_serializable__ = False


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

    __dump__._is_placeholder = True  # type: ignore
    __load__._is_placeholder = True  # type: ignore

    def dump(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        try:
            return self.__dump__(recurse)
        except Exception as exc:
            raise SerializationError("Failed to dump serializable object") from exc

    @classmethod
    def load(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        try:
            return cls.__load__(data)
        except Exception as exc:
            raise DeserializationError(
                f"Failed to load serializable object of type {cls.__name__!r}"
            ) from exc


# Register `Serializable` with cattrs converter
def structure_serializable(
    data: typing.Mapping[str, typing.Any], cls: typing.Type[Serializable]
) -> Serializable:
    return cls.load(data)


def unstructure_serializable(obj: Serializable) -> typing.Mapping[str, typing.Any]:
    return obj.dump(recurse=True)


converter.register_structure_hook(Serializable, structure_serializable)
converter.register_unstructure_hook(Serializable, unstructure_serializable)


SerializableT = typing.TypeVar("SerializableT", bound=Serializable)
_SerializableT = typing.TypeVar("_SerializableT", bound=Serializable)


def dump(o: Serializable, /, recurse: bool = True) -> typing.Dict[str, typing.Any]:
    """Dump a `Serializable` object to a dictionary."""
    return o.__dump__(recurse)


def load(
    cls: typing.Type[SerializableT], data: typing.Mapping[str, typing.Any]
) -> SerializableT:
    """Load a `Serializable` object from a dictionary."""
    return cls.__load__(data)


def make_serializable_type_registrar(
    base_cls: typing.Type[SerializableT],
    registry: typing.Dict[str, typing.Type[SerializableT]],
    key_attr: str = "__type__",
    lock: typing.Optional[threading.Lock] = None,
    key_factory: typing.Optional[
        typing.Callable[[typing.Type[SerializableT]], str]
    ] = None,
    override: bool = False,
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
    :param override: Whether to allow overriding existing registrations.
    :return: A decorator to register `Serializable` subclasses.
    """
    if not key_attr:
        raise ValidationError("`key_attr` must be a non-empty string.")

    lock = lock or threading.Lock()

    def registrar(cls: typing.Type[_SerializableT]) -> typing.Type[_SerializableT]:
        """Decorator to register a `Serializable` subclass."""
        if not issubclass(cls, base_cls):
            raise ValidationError(
                f"Class {cls.__name__} is not a subclass of {base_cls.__name__}"
            )

        key = getattr(cls, key_attr, None)
        if not key:
            key = key_factory(cls) if key_factory else cls.__name__.lower()
            setattr(cls, key_attr, key)

        if not override and key in registry and not issubclass(cls, registry[key]):
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
            raise DeserializationError(f"Unsupported {base_cls.__name__} type: {key!r}")

        cls = registry[key]
        return cls.load(value)

    deserializer.__name__ = f"deserialize_{base_cls.__name__.lower()}"
    return deserializer


T = typing.TypeVar("T")


def register_type_serializer(
    typ: typing.Type[T],
    serializer: typing.Callable[[T, bool], typing.Dict[str, typing.Any]],
) -> None:
    """Register a global type serializer for a specific type."""
    with _type_serializers_lock:
        _TYPE_SERIALIZERS[typ] = serializer


def register_type_deserializer(
    typ: typing.Type[T],
    deserializer: typing.Callable[[typing.Mapping[str, typing.Any]], T],
) -> None:
    """Register a global type deserializer for a specific type."""
    with _type_deserializers_lock:
        _TYPE_DESERIALIZERS[typ] = deserializer
