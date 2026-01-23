import threading
import typing

import attrs
from typing_extensions import Self

from bores.errors import DeserializationError, SerializationError, ValidationError
from bores.serialization import Serializable
from bores.states import ModelState
from bores.stores import StoreSerializable
from bores.types import ActionFunc, HookFunc
from bores.types import Coordinates
from bores.wells.base import InjectionWell, ProductionWell, Well, Wells
from bores.wells.controls import WellControl
from bores.wells.core import InjectedFluid, ProducedFluid, WellFluid


__all__ = [
    "WellEvent",
    "WellSchedule",
    "WellSchedules",
    "well_time_hook",
    "well_hooks",
    "well_update_action",
    "well_actions",
    "well_hook",
    "well_action",
    "list_well_hooks",
    "list_well_actions",
    "get_well_hook",
    "get_well_action",
    "serialize_well_hook",
    "serialize_well_action",
]


_HOOKS: typing.Dict[str, HookFunc] = {}
_ACTIONS: typing.Dict[str, ActionFunc] = {}
_hook_lock = threading.Lock()
_action_lock = threading.Lock()


@typing.overload
def well_hook(func: HookFunc) -> HookFunc: ...


@typing.overload
def well_hook(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[[HookFunc], HookFunc]: ...


@typing.overload
def well_hook(
    func: HookFunc,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> HookFunc: ...


def well_hook(
    func: typing.Optional[HookFunc] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[HookFunc, typing.Callable[[HookFunc], HookFunc]]:
    """
    Register a well hook function for serialization.


    A well hook is a callable that takes a well and the model state as arguments
    and returns a boolean indicating whether to apply a scheduled event.

    :param func: The hook function to register.
    :param name: The name to register the hook under. If None, the function's
        __name__ attribute is used.
    :param override: If True, override any existing hook with the same name.
    :return: The registered hook function or a decorator to register the function.
    """

    def decorator(func: HookFunc) -> HookFunc:
        hook_name = name or getattr(func, "__name__", None)
        if not hook_name:
            raise ValidationError(
                "Hook function must have a `__name__` attribute or a name must be provided."
            )

        with _hook_lock:
            if hook_name in _HOOKS and not override:
                raise ValidationError(
                    f"Hook '{hook_name}' already registered. Use `override=True` or provide a different name."
                )

            _HOOKS[hook_name] = func
        return func

    if func is not None:
        return decorator(func)
    return decorator


@typing.overload
def well_action(func: ActionFunc) -> ActionFunc: ...


@typing.overload
def well_action(
    func: None = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Callable[[ActionFunc], ActionFunc]: ...


@typing.overload
def well_action(
    func: ActionFunc,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> ActionFunc: ...


def well_action(
    func: typing.Optional[ActionFunc] = None,
    name: typing.Optional[str] = None,
    override: bool = False,
) -> typing.Union[ActionFunc, typing.Callable[[ActionFunc], ActionFunc]]:
    """
    Register a well action function for serialization.

    A well action is a callable that takes a well and the model state as arguments
    and performs the event action.

    :param func: The action function to register.
    :param name: The name to register the action under. If None, the function's
        __name__ attribute is used.
    :param override: If True, override any existing action with the same name.
    :return: The registered action function or a decorator to register the function.
    """

    def decorator(func: ActionFunc) -> ActionFunc:
        action_name = name or getattr(func, "__name__", None)
        if not action_name:
            raise ValidationError(
                "Action function must have a `__name__` attribute or a name must be provided."
            )
        with _action_lock:
            if action_name in _ACTIONS and not override:
                raise ValidationError(
                    f"Action '{action_name}' already registered. Use `override=True` or provide a different name."
                )

            _ACTIONS[action_name] = func
        return func

    if func is not None:
        return decorator(func)
    return decorator


def list_well_hooks() -> typing.List[str]:
    """List all registered well hooks."""
    with _hook_lock:
        return list(_HOOKS.keys())


def list_well_actions() -> typing.List[str]:
    """List all registered well actions."""
    with _action_lock:
        return list(_ACTIONS.keys())


def get_well_hook(name: str) -> HookFunc:
    """
    Get a registered well hook by name.

    :param name: The name of the registered hook.
    :return: The hook function associated with the given name.
    :raises ValidationError: If the hook is not registered.
    """
    with _hook_lock:
        if name not in _HOOKS:
            raise ValidationError(
                f"Hook '{name}' not registered. Use `@well_hook` to register it."
            )
        return _HOOKS[name]


def get_well_action(name: str) -> ActionFunc:
    """
    Get a registered well action by name.

    :param name: The name of the registered action.
    :return: The action function associated with the given name.
    :raises ValidationError: If the action is not registered.
    """
    with _action_lock:
        if name not in _ACTIONS:
            raise ValidationError(
                f"Action '{name}' not registered. Use `@well_action` to register it."
            )
        return _ACTIONS[name]


def serialize_well_hook(
    hook: HookFunc, recurse: bool = True
) -> typing.Dict[str, typing.Any]:
    """Serialize a hook function."""
    # Check for registered hooks
    with _hook_lock:
        for name, registered_hook in _HOOKS.items():
            if hook is registered_hook:
                return {"type": "registered", "name": name}

    # Check for built-in hook types
    if isinstance(hook, WellTimeHook):
        return {
            "type": "time_hook",
            "data": hook.dump(recurse),
        }

    if isinstance(hook, WellHooks):
        return {
            "type": "composite_hooks",
            "data": hook.dump(recurse),
        }

    raise SerializationError(
        f"Cannot serialize hook {hook}. Please register it with `@well_hook`."
    )


def deserialize_well_hook(data: typing.Mapping[str, typing.Any]) -> HookFunc:
    """Deserialize a hook function."""
    if "type" not in data:
        raise DeserializationError("Invalid data for hook deserialization")

    hook_type = data["type"]

    if hook_type == "registered":
        with _hook_lock:
            if data["name"] not in _HOOKS:
                raise DeserializationError(f"Hook '{data['name']}' not registered")
            return _HOOKS[data["name"]]

    elif hook_type == "time_hook":
        if "data" not in data:
            raise DeserializationError("Invalid data for time hook deserialization")
        return WellTimeHook.load(data["data"])

    elif hook_type == "composite_hooks":
        if "data" not in data:
            raise DeserializationError(
                "Invalid data for composite hooks deserialization"
            )
        return WellHooks.load(data["data"])

    raise DeserializationError(
        f"Unknown hook type: {hook_type}. Please register it with `@well_hook`."
    )


def serialize_well_action(
    action: ActionFunc, recurse: bool = True
) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """Serialize an action function."""
    # Check for registered actions
    with _action_lock:
        for name, registered_action in _ACTIONS.items():
            if action is registered_action:
                return {"type": "registered", "name": name}

    # Check for built-in action types
    if isinstance(action, WellUpdateAction):
        return {
            "type": "update_action",
            "data": action.dump(recurse),
        }

    if isinstance(action, WellActions):
        return {
            "type": "composite_actions",
            "data": action.dump(recurse),
        }

    raise SerializationError(
        f"Cannot serialize action {action}. Please register it with `@well_action`."
    )


def deserialize_well_action(data: typing.Mapping[str, typing.Any]) -> ActionFunc:
    """Deserialize an action function."""
    if "type" not in data:
        raise DeserializationError("Invalid data for action deserialization")

    action_type = data["type"]

    if action_type == "registered":
        with _action_lock:
            if data["name"] not in _ACTIONS:
                raise DeserializationError(f"Action '{data['name']}' not registered")
            return _ACTIONS[data["name"]]

    elif action_type == "update_action":
        if "data" not in data:
            raise DeserializationError("Invalid data for update action deserialization")
        return WellUpdateAction.load(data["data"])

    elif action_type == "composite_actions":
        if "data" not in data:
            raise DeserializationError(
                "Invalid data for composite actions deserialization"
            )
        return WellActions.load(data["data"])

    raise DeserializationError(
        f"Unknown action type: {action_type}. Please register it with `@well_action`."
    )


@attrs.define(slots=True, hash=True)
class WellEvent(typing.Generic[Coordinates], Serializable):
    """
    Represents a scheduled event for a well at a specific time step.

    This event can include changes to the well's bottom-hole pressure, skin factor,
    and whether the well is active or not.
    The event is applied to the well at the specified time step.
    """

    hook: HookFunc[Well[Coordinates, WellFluid], typing.Any]
    """A callable hook that takes the well and model state as arguments and returns a boolean indicating whether to apply the event."""
    action: ActionFunc[Well[Coordinates, WellFluid], typing.Any]
    """A callable action that takes the well and model state as arguments and performs the event action."""

    def apply(
        self, well: Well[Coordinates, WellFluid], state: ModelState[Coordinates]
    ) -> None:
        """
        Apply this schedule to a well.

        :param well: The well to which this schedule will be applied.
        :param state: The current model state in the simulation.
        """
        self.action(well, state)
        return None

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize the well event."""
        return {
            "hook": serialize_well_hook(self.hook),
            "action": serialize_well_action(self.action),
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> "WellEvent":
        """Deserialize the well event."""
        if "hook" not in data or "action" not in data:
            raise DeserializationError("Invalid data for well event deserialization")
        return cls(
            hook=deserialize_well_hook(data["hook"]),
            action=deserialize_well_action(data["action"]),
        )


@attrs.frozen(slots=True)
class WellSchedule(typing.Generic[Coordinates], Serializable):
    """
    Represents a schedule of events for a well over time.
    """

    events: typing.Dict[str, WellEvent[Coordinates]] = attrs.field(
        factory=dict, init=False
    )
    """A dictionary mapping time steps to scheduled events."""
    _events_hashes: typing.Dict[int, str] = attrs.field(factory=dict, init=False)
    """A set to track hashes of added events for uniqueness."""

    def add(self, id_: str, event: WellEvent[Coordinates]) -> None:
        """
        Adds an event to the schedule.

        :param event: The event to add.
        :param id: The unique identifier for the event.
        """
        self.events[id_] = event
        self._events_hashes[hash(event)] = id_

    def remove(self, o: typing.Union[WellEvent[Coordinates], str], /) -> None:
        """
        Removes an event from the schedule if it exists.

        :param o: The event or its identifier to remove.
        """
        if isinstance(o, WellEvent):
            # If o is a WellEvent, find its ID and remove it
            event_hash = hash(o)
            if event_hash in self._events_hashes:
                del self.events[self._events_hashes[event_hash]]
                del self._events_hashes[event_hash]
        else:
            # If o is an ID, remove the event with that ID
            if o in self.events:
                event_hash = hash(self.events[o])
                del self._events_hashes[event_hash]
                del self.events[o]

    def clear(self) -> None:
        """Clears all events from the schedule."""
        self.events.clear()
        self._events_hashes.clear()

    def get(self, id_: str) -> typing.Optional[WellEvent[Coordinates]]:
        """
        Retrieves an event from the schedule by its identifier.

        :param id: The unique identifier for the event.
        :return: The event if found, None otherwise.
        """
        return self.events.get(id_, None)

    def __getitem__(self, id_: str) -> WellEvent[Coordinates]:
        """
        Retrieves an event from the schedule by its identifier.

        :param id: The unique identifier for the event.
        :return: The event associated with the given identifier.
        :raises KeyError: If the identifier is not found in the schedule.
        """
        try:
            return self.events[id_]
        except KeyError:
            raise KeyError(f"Event with id {id_} not found")

    def __setitem__(self, id_: str, event: WellEvent[Coordinates]) -> None:
        """
        Sets an event in the schedule with the given identifier.

        :param id: The unique identifier for the event.
        :param event: The event to set.
        """
        self.events[id_] = event
        self._events_hashes[hash(event)] = id_

    def __contains__(self, o: typing.Union[WellEvent, str], /) -> bool:
        """
        Checks if an event or its identifier is in the schedule.

        :param o: The event or its identifier to check.
        :return: True if the event or its identifier is in the schedule, False otherwise.
        """
        if isinstance(o, WellEvent):
            return hash(o) in self._events_hashes
        return o in self.events

    def __len__(self) -> int:
        """Returns the number of events in the schedule."""
        return len(self.events)

    def __iter__(self) -> typing.Iterator[str]:
        """Returns an iterator over the event identifiers in the schedule."""
        return iter(self.events)

    def apply(
        self,
        well: Well[Coordinates, WellFluid],
        state: ModelState[Coordinates],
        *ids: str,
    ) -> None:
        """
        Applies the schedule to a well.

        :param well: The well to which the schedule will be applied.
        :param state: The current model state in the simulation.
        :param ids: The unique identifiers of the events to apply.
        """
        if not ids:
            for event in self.events.values():
                if event.hook(well, state):
                    event.apply(well, state)

        else:
            for id_ in ids:
                event = self.events.get(id_, None)
                if event is None:
                    continue
                if event.hook(well, state):
                    event.apply(well, state)
        return None

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "events": {id_: event.dump(recurse) for id_, event in self.events.items()}
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        if "events" not in data:
            raise DeserializationError("Invalid data for well schedule deserialization")

        events = {
            id_: WellEvent.load(event_data)
            for id_, event_data in data["events"].items()
        }
        schedule = cls()
        for id_, event in events.items():
            schedule.add(id_, event)
        return schedule


@attrs.define(slots=True)
class WellSchedules(typing.Generic[Coordinates], StoreSerializable):
    """
    Represents a collection of well schedules for multiple wells.
    """

    schedules: typing.Dict[str, WellSchedule[Coordinates]] = attrs.field(
        factory=dict, init=False
    )
    """A dictionary mapping well names to their schedules."""

    def add(self, well_name: str, schedule: WellSchedule[Coordinates]) -> None:
        """
        Adds a schedule for a given well.

        :param well_name: The name of the well.
        :param schedule: The schedule for the well.
        """
        self.schedules[well_name] = schedule

    def get(self, well_name: str) -> typing.Optional[WellSchedule[Coordinates]]:
        """
        Retrieves the schedule for a given well by its name.

        :param well_name: The name of the well.
        :return: The schedule for the well if found, None otherwise.
        """
        return self.schedules.get(well_name, None)

    def remove(self, well_name: str) -> None:
        """
        Removes the schedule for a given well by its name.

        :param well_name: The name of the well.
        """
        if well_name in self.schedules:
            del self.schedules[well_name]

    def clear(self) -> None:
        """Clears all well schedules."""
        self.schedules.clear()

    def __len__(self) -> int:
        """Returns the number of well schedules."""
        return len(self.schedules)

    def __iter__(self) -> typing.Iterator[str]:
        """Returns an iterator over the well names in the schedules."""
        return iter(self.schedules)

    def __contains__(self, well_name: str) -> bool:
        """
        Checks if a schedule for a given well name exists.

        :param well_name: The name of the well.
        :return: True if the schedule exists, False otherwise.
        """
        return well_name in self.schedules

    def __setitem__(self, well_name: str, schedule: WellSchedule[Coordinates]) -> None:
        """
        Sets the schedule for a given well name.

        :param well_name: The name of the well.
        :param schedule: The schedule for the well.
        """
        self.schedules[well_name] = schedule

    def __getitem__(self, well_name: str) -> WellSchedule[Coordinates]:
        """
        Retrieves the schedule for a given well name.

        :param well_name: The name of the well.
        :return: The schedule for the well.
        :raises KeyError: If the well name is not found in the schedules.
        """
        try:
            return self.schedules[well_name]
        except KeyError:
            raise KeyError(f"Schedule for well {well_name} not found")

    def apply(self, wells: Wells[Coordinates], state: ModelState[Coordinates]) -> None:
        """
        Applies all well schedules to their respective wells.

        :param wells: The collection of wells to which the schedules will be applied.
        :param state: The current model state in the simulation.
        """
        for well_name, schedule in self.schedules.items():
            well_list = wells.get_by_name(well_name)
            if not well_list:
                continue
            for well in well_list:
                schedule.apply(well, state)  # type: ignore
        return None

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            "schedules": {
                well_name: schedule.dump(recurse)
                for well_name, schedule in self.schedules.items()
            }
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        if "schedules" not in data:
            raise DeserializationError(
                "Invalid data for well schedules deserialization"
            )

        schedules = {
            well_name: WellSchedule.load(schedule_data)
            for well_name, schedule_data in data["schedules"].items()
        }
        well_schedules = cls()
        for well_name, schedule in schedules.items():
            well_schedules.add(well_name, schedule)
        return well_schedules


class WellTimeHook(
    Serializable,
    fields={"time_step": typing.Optional[int], "time": typing.Optional[float]},
):
    def __init__(
        self,
        time_step: typing.Optional[int] = None,
        time: typing.Optional[float] = None,
    ):
        """
        Initializes the well_time_hook with either a specific time step or time.

        :param time_step: The specific time step at which to trigger the event.
        :param time: The specific simulation time at which to trigger the event.
        """
        if not (time_step or time):
            raise ValidationError("Either time_step or time must be provided.")
        self.time_step = time_step
        self.time = time

    def __call__(self, well: Well, state: ModelState[Coordinates]) -> bool:
        """
        The hook function that checks if the event should be applied based on the model state.

        :param well: The well to which this hook is applied.
        :param state: The current model state in the simulation.
        :return: A boolean indicating whether to apply the event.
        """
        if self.time_step is not None and state.step == self.time_step:
            return True
        if self.time is not None and state.time == self.time:
            return True
        return False


class WellUpdateAction(
    Serializable,
    fields={
        "control": typing.Optional[WellControl],
        "skin_factor": typing.Optional[float],
        "is_active": typing.Optional[bool],
        "injected_fluid": typing.Optional[InjectedFluid],
        "produced_fluids": typing.Optional[typing.Sequence[ProducedFluid]],
    },
):
    def __init__(
        self,
        control: typing.Optional[WellControl] = None,
        skin_factor: typing.Optional[float] = None,
        is_active: typing.Optional[bool] = None,
        injected_fluid: typing.Optional[InjectedFluid] = None,
        produced_fluids: typing.Optional[typing.Sequence[ProducedFluid]] = None,
    ):
        """
        Initializes the WellUpdateAction with properties to update.

        :param control: New control strategy for the well.
        :param skin_factor: New skin factor for the well.
        :param is_active: New active status for the well (True for open, False for shut in).
        :param injected_fluid: New fluid properties for injection wells.
        :param produced_fluids: New fluid properties for production wells.
        """
        valid = any(
            param is not None
            for param in [
                control,
                skin_factor,
                is_active,
                injected_fluid,
                produced_fluids,
            ]
        )
        if not valid:
            raise ValidationError("At least one property must be provided to update.")

        self.control = control
        self.skin_factor = skin_factor
        self.is_active = is_active
        self.injected_fluid = injected_fluid
        self.produced_fluids = produced_fluids

    def __call__(self, well: Well, state: ModelState[Coordinates]) -> None:
        """
        The action function that modifies well configuration.

        :param well: The well to which this action is applied.
        :param state: The current model state in the simulation.
        """
        if self.control is not None:
            well.control = self.control
        if self.skin_factor is not None:
            well.skin_factor = self.skin_factor
        if self.is_active is True:
            well.open()
        elif self.is_active is False:
            well.shut_in()

        if self.injected_fluid is not None and isinstance(well, InjectionWell):
            well.injected_fluid = self.injected_fluid
        if self.produced_fluids is not None and isinstance(well, ProductionWell):
            well.produced_fluids = self.produced_fluids
        return


class WellHooks(Serializable):
    """Composite hook that chains multiple hook functions."""

    __abstract_serializable__ = True

    def __init__(self, *hooks: HookFunc[Well, typing.Any], on_any: bool = False):
        """
        Initializes the WellHooks with a sequence of hook functions.

        :param hooks: A sequence of hook functions to be chained.
        :param on_any: If True, the composite hook returns True if any of the hooks return True.
                       If False, it returns True only if all hooks return True.
        """
        self.hooks = hooks
        self.on_any = on_any

    def __call__(self, well: Well, state: ModelState[Coordinates]) -> bool:
        """
        Calls the composite hook function.

        :param well: The well to which this hook is applied.
        :param state: The current model state in the simulation.
        :return: A boolean indicating whether to apply the event.
        """
        results = (hook(well, state) for hook in self.hooks)
        return any(results) if self.on_any else all(results)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize the composite hook."""
        return {
            "hooks": [serialize_well_hook(hook, recurse) for hook in self.hooks],
            "on_any": self.on_any,
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Deserialize the composite hook."""
        if "hooks" not in data or "on_any" not in data:
            raise DeserializationError("Invalid data for well hooks deserialization")

        hooks = [deserialize_well_hook(hook_data) for hook_data in data["hooks"]]
        return cls(*hooks, on_any=data["on_any"])


class WellActions(Serializable):
    """Composite action that chains multiple action functions."""

    __abstract_serializable__ = True

    def __init__(self, *actions: ActionFunc[Well, typing.Any]):
        """
        Initializes the WellActions with a sequence of action functions.

        :param actions: A sequence of action functions to be chained.
        """
        if not actions:
            raise ValidationError("At least one action must be provided to chain.")
        self.actions = actions

    def __call__(self, well: Well, state: ModelState[Coordinates]) -> None:
        """
        Executes all chained action functions in sequence.

        :param well: The well to which the actions are applied.
        :param state: The current model state in the simulation.
        """
        for action in self.actions:
            action(well, state)

    def __dump__(self, recurse: bool = True) -> typing.Dict[str, typing.Any]:
        """Serialize the composite action."""
        return {
            "actions": [
                serialize_well_action(action, recurse) for action in self.actions
            ]
        }

    @classmethod
    def __load__(cls, data: typing.Mapping[str, typing.Any]) -> Self:
        """Deserialize the composite action."""
        if "actions" not in data:
            raise DeserializationError("Invalid data for well actions deserialization")

        actions = [
            deserialize_well_action(action_data) for action_data in data["actions"]
        ]
        return cls(*actions)


def well_time_hook(
    time_step: typing.Optional[int] = None,
    time: typing.Optional[float] = None,
) -> HookFunc[Well, typing.Any]:
    """
    Returns a hook function that triggers at a specific time step or time.

    :param time_step: The specific time step at which to trigger the event.
    :param time: The specific simulation time at which to trigger the event.
    :return: A hook function that takes a well and model state as arguments and returns a boolean indicating whether to apply the event.
    """
    return WellTimeHook(time_step=time_step, time=time)


def well_update_action(
    control: typing.Optional[WellControl] = None,
    skin_factor: typing.Optional[float] = None,
    is_active: typing.Optional[bool] = None,
    injected_fluid: typing.Optional[InjectedFluid] = None,
    produced_fluids: typing.Optional[typing.Sequence[ProducedFluid]] = None,
) -> ActionFunc[Well, typing.Any]:
    """
    Returns an action function that modifies well configuration.

    :param control: New control strategy for the well.
    :param skin_factor: New skin factor for the well.
    :param is_active: New active status for the well (True for open, False for shut in).
    :param injected_fluid: New fluid properties for injection wells.
    :param produced_fluids: New fluid properties for production wells.
    :return: An action function that takes a well and model state as arguments and performs the property updates.
    """
    return WellUpdateAction(
        control=control,
        skin_factor=skin_factor,
        is_active=is_active,
        injected_fluid=injected_fluid,
        produced_fluids=produced_fluids,
    )


def well_hooks(
    *hooks: HookFunc[Well, typing.Any],
    on_any: bool = False,
) -> HookFunc[Well, typing.Any]:
    """
    Returns a composite hook function that chains multiple hooks.

    :param hooks: A sequence of hook functions to be chained.
    :param on_any: If True, the composite hook returns True if any of the hooks return True.
                   If False, it returns True only if all hooks return True.
    :return: A composite hook function that takes a well and model state as arguments and returns a boolean indicating whether to apply the event.
    """
    return WellHooks(*hooks, on_any=on_any)


def well_actions(
    *actions: ActionFunc[Well, typing.Any],
) -> ActionFunc[Well, typing.Any]:
    """
    Returns a composite action function that chains multiple actions.

    :param actions: A sequence of action functions to be chained.
    :return: A composite action function that takes a well and model state as arguments and performs all actions in sequence.
    """
    return WellActions(*actions)
