import typing

import attrs

from bores.errors import ValidationError
from bores.states import ModelState
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
]


@attrs.define(slots=True, hash=True)
class WellEvent(typing.Generic[Coordinates]):
    """
    Represents a scheduled event for a well at a specific time step.

    This event can include changes to the well's bottom-hole pressure, skin factor,
    and whether the well is active or not.
    The event is applied to the well at the specified time step.
    """

    hook: typing.Optional[HookFunc[Well[Coordinates, WellFluid], typing.Any]] = None
    """A callable hook that takes the well and model state as arguments and returns a boolean indicating whether to apply the event."""
    action: typing.Optional[ActionFunc[Well[Coordinates, WellFluid], typing.Any]] = None
    """A callable action that takes the well and model state as arguments and performs the event action."""

    def apply(
        self, well: Well[Coordinates, WellFluid], state: ModelState[Coordinates]
    ) -> None:
        """
        Apply this schedule to a well.

        :param well: The well to which this schedule will be applied.
        :param state: The current model state in the simulation.
        """
        if self.action is not None:
            self.action(well, state)
        return None


@attrs.frozen(slots=True)
class WellSchedule(typing.Generic[Coordinates]):
    """
    Represents a schedule of events for a well over time.
    """

    events: typing.Dict[typing.Hashable, WellEvent[Coordinates]] = attrs.field(
        factory=dict, init=False
    )
    """A dictionary mapping time steps to scheduled events."""
    _events_hashes: typing.Dict[int, typing.Hashable] = attrs.field(
        factory=dict, init=False
    )
    """A set to track hashes of added events for uniqueness."""

    def add(self, id: typing.Hashable, event: WellEvent[Coordinates]) -> None:
        """
        Adds an event to the schedule.

        :param event: The event to add.
        :param id: The unique identifier for the event.
        """
        self.events[id] = event
        self._events_hashes[hash(event)] = id

    def remove(
        self, o: typing.Union[WellEvent[Coordinates], typing.Hashable], /
    ) -> None:
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

    def get(self, id: typing.Hashable) -> typing.Optional[WellEvent[Coordinates]]:
        """
        Retrieves an event from the schedule by its identifier.

        :param id: The unique identifier for the event.
        :return: The event if found, None otherwise.
        """
        return self.events.get(id, None)

    def __getitem__(self, id: typing.Hashable) -> WellEvent[Coordinates]:
        """
        Retrieves an event from the schedule by its identifier.

        :param id: The unique identifier for the event.
        :return: The event associated with the given identifier.
        :raises KeyError: If the identifier is not found in the schedule.
        """
        try:
            return self.events[id]
        except KeyError:
            raise KeyError(f"Event with id {id} not found")

    def __setitem__(self, id: typing.Hashable, event: WellEvent[Coordinates]) -> None:
        """
        Sets an event in the schedule with the given identifier.

        :param id: The unique identifier for the event.
        :param event: The event to set.
        """
        self.events[id] = event
        self._events_hashes[hash(event)] = id

    def __contains__(self, o: typing.Union[WellEvent, typing.Hashable], /) -> bool:
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

    def __iter__(self) -> typing.Iterator[typing.Hashable]:
        """Returns an iterator over the event identifiers in the schedule."""
        return iter(self.events)

    def apply(
        self,
        well: Well[Coordinates, WellFluid],
        state: ModelState[Coordinates],
        *ids: typing.Hashable,
    ) -> None:
        """
        Applies the schedule to a well.

        :param well: The well to which the schedule will be applied.
        :param state: The current model state in the simulation.
        :param ids: The unique identifiers of the events to apply.
        """
        if not ids:
            for event in self.events.values():
                if not event.hook or event.hook(well, state):
                    event.apply(well, state)

        else:
            for id in ids:
                event = self.events.get(id, None)
                if event is None:
                    continue
                if not event.hook or event.hook(well, state):
                    event.apply(well, state)
        return None


@attrs.define(slots=True)
class WellSchedules(typing.Generic[Coordinates]):
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


class WellTimeHook:
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


class WellHooks:
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


class WellUpdateAction:
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


class WellActions:
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


def well_actions(
    *actions: ActionFunc[Well, typing.Any],
) -> ActionFunc[Well, typing.Any]:
    """
    Returns a composite action function that chains multiple actions.

    :param actions: A sequence of action functions to be chained.
    :return: A composite action function that takes a well and model state as arguments and performs all actions in sequence.
    """
    return WellActions(*actions)
