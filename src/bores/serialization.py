from collections.abc import Mapping
import typing

import attrs
import cattrs
from typing_extensions import Self


__all__ = ["Serializable", "dump", "load"]


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


class SerializableMeta(type):
    """Metaclass for `Serializable` classes"""

    def __new__(
        mcs,
        name: str,
        bases: typing.Tuple,
        namespace: typing.Dict[str, typing.Any],
        fields: typing.Optional[typing.Mapping[str, typing.Type]] = None,
        dump_exclude: typing.Optional[typing.Iterable[str]] = None,
        load_exclude: typing.Optional[typing.Iterable[str]] = None,
        serializers: typing.Optional[
            typing.Mapping[typing.Union[str, typing.Type], typing.Callable]
        ] = None,
        deserializers: typing.Optional[
            typing.Mapping[typing.Union[str, typing.Type], typing.Callable]
        ] = None,
    ):
        """
        Create a new `Serializable` class.

        :param fields: An optional mapping of field names to their types to include.
            If not provided, uses annotations from the class to determine fields.
        :param exclude: An optional iterable of field names to exclude.
        :param serializers: An optional mapping of field names/types to custom serializer functions.
        :param deserializers: An optional mapping of field names/types to custom deserializer functions.
        :return: A new `Serializable` class.
        """
        if (
            "__abstract_serializable__" in namespace
            and namespace["__abstract_serializable__"]
        ):
            return super().__new__(mcs, name, bases, namespace)

        parent_serializers = {}
        parent_deserializers = {}
        parent_fields = {}
        for cls in bases[::-1]:
            if serializable_fields := getattr(cls, "__serializable_fields__", None):
                parent_fields.update(serializable_fields)

            if serializable_serializers := getattr(
                cls, "__serializable_serializers__", None
            ):
                parent_serializers.update(serializable_serializers)

            if serializable_deserializers := getattr(
                cls, "__serializable_deserializers__", None
            ):
                parent_deserializers.update(serializable_deserializers)

        cls_fields = fields or namespace.get(
            "__annotations__", None
        )  # assume __annotations__ is present
        all_fields = {**parent_fields, **(cls_fields or {})}
        all_serializers = {**parent_serializers, **(serializers or {})}
        all_deserializers = {**parent_deserializers, **(deserializers or {})}

        if not all_fields:
            raise ValueError(
                "Serializable classes must have fields defined. If the class is an abstract base class, set `__abstract_serializable__` to True"
            )

        if not isinstance(all_fields, Mapping):
            raise TypeError("`fields` must be a mapping of field names to types.")

        if "__dump__" not in namespace or namespace["__dump__"].__is_placeholder__:
            namespace["__dump__"] = mcs._build_default_dumper(
                fields=all_fields,
                exclude=dump_exclude,
                serializers=serializers,
            )
            namespace["__dump__"].__is_placeholder__ = False  # type: ignore
        if "__load__" not in namespace or namespace["__load__"].__is_placeholder__:
            namespace["__load__"] = mcs._build_default_loader(
                fields=all_fields,
                exclude=load_exclude,
                deserializers=deserializers,
            )
            namespace["__load__"].__is_placeholder__ = False  # type: ignore

        namespace["__serializable_fields__"] = all_fields
        namespace["__serializable_serializers__"] = all_serializers
        namespace["__serializable_deserializers__"] = all_deserializers
        namespace.setdefault("__abstract_serializable__", False)
        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    @staticmethod
    def _build_default_dumper(
        fields: typing.Mapping[str, typing.Type],
        exclude: typing.Optional[typing.Iterable[str]] = None,
        serializers: typing.Optional[
            typing.Mapping[typing.Union[str, typing.Type], typing.Callable]
        ] = None,
    ) -> typing.Callable:
        """
        Build a dumper function for the class.

        :param fields: Mapping of field names to their types.
        :return: A dumper function.
        """

        def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
            result = {}
            for field, typ in fields.items():
                if exclude and field in exclude:
                    continue
                value = getattr(self, field)

                if serializers and (field in serializers):
                    serializer = serializers[field]
                    result[field] = serializer(value)
                elif serializers and (typ in serializers):
                    serializer = serializers[typ]
                    result[field] = serializer(value)
                elif isinstance(value, Serializable):
                    result[field] = value.__dump__(recurse)
                else:
                    result[field] = converter.unstructure(value, typ)
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

        :param fields: Mapping of field names to their types.
        :param exclude: An optional iterable of field names to exclude.
        :param deserializers: An optional mapping of field names/types to custom deserializer functions.
        :return: A loader function.
        """

        @classmethod
        def __load__(cls, data: typing.Mapping[str, typing.Any]):
            init_kwargs = {}
            for field, typ in fields.items():
                if exclude and field in exclude:
                    continue

                value = data[field]
                if deserializers and (field in deserializers):
                    deserializer = deserializers[field]
                    init_kwargs[field] = deserializer(value)
                elif deserializers and (typ in deserializers):
                    deserializer = deserializers[typ]
                    init_kwargs[field] = deserializer(value)
                elif isinstance(typ, type) and issubclass(typ, Serializable):
                    init_kwargs[field] = typ.__load__(value)
                else:
                    init_kwargs[field] = converter.structure(value, typ)

            return cls(**init_kwargs)

        return __load__


class Serializable(metaclass=SerializableMeta):
    """
    Base class for serializable objects.

    Examples:
    ```python

    # Attrs uses annotations-based approach.
    # So no need to explicitly pass fields.
    @attrs.define
    class Point(Serializable):
        x: float
        y: float
        z: float

    # For regular classes, pass fields explicitly.
    class Point(Serializable, fields={"x": float, "y": float, "z": float}):
        def __init__(self, x: float, y: float, z: float):
            self.x = x
            self.y = y
            self.z = z

    # Using custom serializers/deserializers
    def serialize_vector(vec: List[float]) -> str:
        return ",".join(map(str, vec))

    def deserialize_vector(data: str) -> List[float]:
        return list(map(float, data.split(",")))

    class Vector(
        Serializable,
        fields={"components": List[float]},
        serializers={"components": serialize_vector},
        deserializers={"components": deserialize_vector}
    ):
        def __init__(self, components: List[float]):
            self.components = components

    # Using exclude to skip fields
    # Since password is not needed during loading, we exclude it.
    # But we would still get it during dumping.
    @attrs.define
    class User(Serializable, load_exclude={"password"}):
        username: str
        password: str = attrs.field(init=False)
    ```
    """

    __abstract_serializable__ = True

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """
        Dump the object to a dictionary.

        :param recurse: Whether to recursively dump nested `Serializable` objects.
        :return: A dictionary representation of the object.
        """
        raise NotImplementedError

    @classmethod
    def __load__(cls: typing.Type[Self], data: typing.Mapping[str, typing.Any]) -> Self:
        """
        Load an object from a dictionary.

        :param data: The dictionary representation of the object.
        :return: An instance of the object.
        """
        raise NotImplementedError

    __dump__.__is_placeholder__ = True  # type: ignore
    __load__.__is_placeholder__ = True  # type: ignore

    def dump(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return self.__dump__(recurse)

    @classmethod
    def load(cls, data: typing.Dict[str, typing.Any]) -> Self:
        return cls.__load__(data)


SerializableT = typing.TypeVar("SerializableT", bound=Serializable)


def dump(self, o: Serializable, recurse: bool = True) -> typing.Dict[str, typing.Any]:
    """
    Dump a `Serializable` object to a dictionary.

    :param o: The `Serializable` object to dump.
    :param recurse: Whether to recursively dump nested `Serializable` objects.
    :return: A dictionary representation of the object.
    """
    return o.__dump__(recurse=recurse)


def load(
    cls: typing.Type[SerializableT], data: typing.Mapping[str, typing.Any]
) -> SerializableT:
    """
    Load a `Serializable` object from a dictionary.

    :param cls: The class of the `Serializable` object to load.
    :param data: The dictionary representation of the object.
    :return: An instance of the `Serializable` object.
    """
    return cls.__load__(data)
