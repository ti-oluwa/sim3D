"""Well implementations and base classes."""

import itertools
import logging
import typing

import attrs
import numpy as np
from typing_extensions import Self

from bores.errors import ValidationError
from bores.types import (
    ActionFunc,
    HookFunc,
    Orientation,
    ThreeDimensions,
    TwoDimensions,
    WellLocation,
)
from bores.wells.controls import WellControl
from bores.wells.core import (
    InjectedFluid,
    ProducedFluid,
    WellFluidT,
    compute_2D_effective_drainage_radius,
    compute_3D_effective_drainage_radius,
    compute_effective_permeability_for_well,
    compute_well_index,
)
from bores.pvt.tables import PVTTables


logger = logging.getLogger(__name__)

__all__ = [
    "InjectionWell",
    "ProductionWell",
    "Wells",
    "WellEvent",
    "well_time_hook",
    "well_hooks",
    "well_update_action",
    "well_actions",
    "_expand_intervals",
]


@attrs.define(slots=True, hash=True)
class Well(typing.Generic[WellLocation, WellFluidT]):
    """Models a well in the reservoir model."""

    name: str
    """Name of the well."""
    perforating_intervals: typing.Sequence[typing.Tuple[WellLocation, WellLocation]]
    """Perforating intervals of the well. Each interval is a tuple of (start_location, end_location)."""
    radius: float
    """Radius of the wellbore (ft)."""
    control: WellControl[WellFluidT]
    """Control strategy for the well (e.g., rate control, pressure control)."""
    skin_factor: float = 0.0
    """Skin factor for the well, affecting flow performance."""
    orientation: Orientation = attrs.field(init=False, default=Orientation.Z)
    """Orientation of the well, indicating its dominant direction in the reservoir grid."""
    is_active: bool = True
    """Indicates whether the well is active or not. Set to False if the well is shut in or inactive."""
    schedule: typing.Set["WellEvent[Self]"] = attrs.field(factory=set)  # type: ignore
    """Schedule of events for the well, mapping time steps to scheduled events."""

    def __attrs_post_init__(self) -> None:
        """Ensure the well has a valid orientation."""
        self.orientation = self.get_orientation()

    @property
    def is_shut_in(self) -> bool:
        """Check if the well is shut in."""
        return not self.is_active

    @property
    def is_open(self) -> bool:
        """Check if the well is open."""
        return self.is_active

    def get_orientation(self) -> Orientation:
        """
        Determine the dominant orientation of a straight well (even if slanted)
        by estimating which axis the well is most aligned with.
        Uses the first perforating interval to determine orientation.

        :returns: The dominant orientation of the well
        """
        if not self.perforating_intervals:
            return Orientation.Z  # Default to Z if no intervals

        start, end = self.perforating_intervals[0]

        # Convert to numpy arrays and pad to 3D if needed
        start = np.array(start + (0,) * (3 - len(start)))
        end = np.array(end + (0,) * (3 - len(end)))

        # Compute the direction vector
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm == 0:
            return Orientation.Z  # Default to Z if start and end are the same

        # Normalize and take absolute value
        unit_vector = np.abs(direction / norm)
        axis = np.argmax(unit_vector)
        return Orientation(("x", "y", "z")[axis])

    def schedule_event(self, event: "WellEvent[Self]", /) -> None:
        """
        Add a new `WellEvent` to the well schedule.

        :param event: The event to be scheduled for the well.
            If the event has no hook, it will always be applied after each time step.
        """
        self.schedule.add(event)

    def schedule_events(self, *events: "WellEvent[Self]") -> None:
        """
        Add multiple `WellEvent`s to the well schedule.

        :param events: An iterable of events to be scheduled for the well.
            If an event has no hook, it will always be applied after each time step.
        """
        for event in events:
            self.schedule.add(event)

    def evolve(self, model_state: typing.Any) -> None:
        """
        Evolve the well for the next time step.

        This method updates the state of the well based on its schedule.

        :param model_state: The current model state in the simulation.
        """
        for event in self.schedule:
            if not event.hook or event.hook(self, model_state):
                event.apply(self, model_state)

    def check_location(self, grid_dimensions: typing.Tuple[int, ...]) -> None:
        """
        Check if the well's perforating intervals are within the grid dimensions.

        :param grid_dimensions: The dimensions of the reservoir grid (i, j, k).
        :raises ValidationError: If any of the well's perforating intervals are out of bounds.
        """
        for interval_idx, (start, end) in enumerate(self.perforating_intervals):
            if not all(0 <= coord < dim for coord, dim in zip(start, grid_dimensions)):
                raise ValidationError(
                    f"Start location {start} for interval {interval_idx} of well {self.name!r} is out of bounds."
                )
            if not all(0 <= coord < dim for coord, dim in zip(end, grid_dimensions)):
                raise ValidationError(
                    f"End location {end} for interval {interval_idx} of well {self.name!r} is out of bounds."
                )

    def get_effective_drainage_radius(
        self,
        interval_thickness: typing.Tuple[float, ...],
        permeability: typing.Tuple[float, ...],
    ) -> float:
        """
        Compute the effective drainage radius for the well based on its orientation.

        :param interval_thickness: A tuple representing the thickness of the reservoir interval in each direction (ft).
        :param permeability: A tuple representing the permeability of the reservoir rock in each direction (mD).
        :return: The effective drainage radius in the direction of the well (ft).
        """
        dimensions = len(interval_thickness)
        if dimensions < 2 or dimensions > 3:
            raise ValidationError("2D/3D locations are required")

        if dimensions == 2:
            if len(permeability) != 2:
                raise ValidationError(
                    "Permeability must be a 2D tuple for 2D locations"
                )
            interval_thickness = typing.cast(TwoDimensions, interval_thickness)
            permeability = typing.cast(TwoDimensions, permeability)
            return compute_2D_effective_drainage_radius(
                interval_thickness=interval_thickness,
                permeability=permeability,
                well_orientation=self.orientation,
            )

        if len(permeability) != 3:
            raise ValidationError("Permeability must be a 3D tuple for 3D locations")
        interval_thickness = typing.cast(ThreeDimensions, interval_thickness)
        permeability = typing.cast(ThreeDimensions, permeability)
        return compute_3D_effective_drainage_radius(
            interval_thickness=interval_thickness,
            permeability=permeability,
            well_orientation=self.orientation,
        )

    def get_well_index(
        self,
        interval_thickness: typing.Tuple[float, ...],
        permeability: typing.Tuple[float, ...],
        skin_factor: typing.Optional[float] = None,
    ) -> float:
        """
        Compute the well index for the well using the Peaceman equation.

        :param interval_thickness: A tuple representing the thickness of the reservoir interval in each direction (ft).
        :param permeability: A tuple representing the permeability of the reservoir rock in each direction (mD).
        :return: The well index (md*ft).
        """
        dimensions = len(interval_thickness)
        if dimensions < 2 or dimensions > 3:
            raise ValidationError("2D/3D locations are required")

        orientation = self.orientation
        effective_drainage_radius = self.get_effective_drainage_radius(
            interval_thickness=interval_thickness,
            permeability=permeability,
        )
        skin_factor = skin_factor if skin_factor is not None else self.skin_factor
        radius = self.radius
        effective_permeability = compute_effective_permeability_for_well(
            permeability=permeability, orientation=orientation
        )

        if orientation == Orientation.X:
            directional_thickness = interval_thickness[0]
        elif orientation == Orientation.Y:
            directional_thickness = interval_thickness[1]
        elif dimensions == 3 and orientation == Orientation.Z:
            directional_thickness = interval_thickness[2]
        else:  # dimensions == 2 and orientation == Orientation.Z:
            raise ValidationError("Z-oriented wells are not supported in 2D models")

        return compute_well_index(
            permeability=effective_permeability,
            interval_thickness=directional_thickness,
            wellbore_radius=radius,
            effective_drainage_radius=effective_drainage_radius,
            skin_factor=skin_factor,
        )

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: WellFluidT,
        formation_volume_factor: float,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """
        Compute the flow rate for the well using the configured control strategy.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The relative mobility of the fluid phase being produced or injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: The flow rate in (bbl/day or ft³/day).
        """
        return self.control.get_flow_rate(
            pressure=pressure,
            temperature=temperature,
            phase_mobility=phase_mobility,
            well_index=well_index,
            fluid=fluid,
            is_active=self.is_open,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            formation_volume_factor=formation_volume_factor,
            pvt_tables=pvt_tables,
        )

    def get_bottom_hole_pressure(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: WellFluidT,
        formation_volume_factor: float,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """
        Compute the bottom-hole pressure for the well using the configured control strategy.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The relative mobility of the fluid phase being produced or injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹).
        :param formation_volume_factor: Formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: The bottom-hole pressure (psi).
        """
        return self.control.get_bottom_hole_pressure(
            pressure=pressure,
            temperature=temperature,
            phase_mobility=phase_mobility,
            well_index=well_index,
            fluid=fluid,
            is_active=self.is_open,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            formation_volume_factor=formation_volume_factor,
            pvt_tables=pvt_tables,
        )

    def shut_in(self) -> None:
        """Shut in the well."""
        self.is_active = False

    def open(self) -> None:
        """Open the well."""
        self.is_active = True

    def duplicate(self: Self, *, name: typing.Optional[str] = None, **kwargs) -> Self:
        """
        Create a duplicate of the well with an optional new name.

        :param name: The name for the duplicated well. If None, uses the original well's name.
        :kwargs: Additional properties to override in the duplicated well.
        :return: A new instance of the well with the same properties.
        """
        return attrs.evolve(self, name=name or self.name, **kwargs)


WellT = typing.TypeVar("WellT", bound=Well)


@attrs.define(slots=True, hash=True)
class WellEvent(typing.Generic[WellT]):
    """
    Represents a scheduled event for a well at a specific time step.

    This event can include changes to the well's bottom-hole pressure, skin factor,
    and whether the well is active or not.
    The event is applied to the well at the specified time step.
    """

    hook: typing.Optional[HookFunc[WellT, typing.Any]] = None
    """A callable hook that takes the well and model state as arguments and returns a boolean indicating whether to apply the event."""
    action: typing.Optional[ActionFunc[WellT, typing.Any]] = None
    """A callable action that takes the well and model state as arguments and performs the event action."""

    def apply(self, well: WellT, model_state: typing.Any) -> WellT:
        """
        Apply this schedule to a well.

        :param well: The well to which this schedule will be applied.
        :param model_state: The current model state in the simulation.
        """
        if self.action is not None:
            self.action(well, model_state)
        return well


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

    def __call__(self, well: Well, model_state: typing.Any) -> bool:
        """
        The hook function that checks if the event should be applied based on the model state.

        :param well: The well to which this hook is applied.
        :param model_state: The current model state in the simulation.
        :return: A boolean indicating whether to apply the event.
        """
        if self.time_step is not None and model_state.step == self.time_step:
            return True
        if self.time is not None and model_state.time == self.time:
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

    def __call__(self, well: Well, model_state: typing.Any) -> bool:
        """
        Calls the composite hook function.

        :param well: The well to which this hook is applied.
        :param model_state: The current model state in the simulation.
        :return: A boolean indicating whether to apply the event.
        """
        results = (hook(well, model_state) for hook in self.hooks)
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

    def __call__(self, well: Well, model_state: typing.Any) -> None:
        """
        The action function that modifies well configuration.

        :param well: The well to which this action is applied.
        :param model_state: The current model state in the simulation.
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

    def __call__(self, well: Well, model_state: typing.Any) -> None:
        """
        Executes all chained action functions in sequence.

        :param well: The well to which the actions are applied.
        :param model_state: The current model state in the simulation.
        """
        for action in self.actions:
            action(well, model_state)


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


@attrs.define(slots=True, hash=True)
class InjectionWell(Well[WellLocation, InjectedFluid]):
    """
    Models an injection well in the reservoir model.

    This well injects fluids into the reservoir.
    """

    injected_fluid: typing.Optional[InjectedFluid] = None
    """Properties of the fluid being injected into the well."""

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: InjectedFluid,
        formation_volume_factor: float,
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        pvt_tables: typing.Optional[PVTTables] = None,
    ) -> float:
        """
        Compute the flow rate for the injection well using the configured control strategy.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The relative mobility of the fluid phase being injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being injected into the well. If None, uses the well's injected_fluid property.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹). For slightly compressible fluids, this can be used to adjust the flow rate calculation.
        :param formation_volume_factor: The formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :param pvt_tables: `PVTTables` object for fluid property lookups
        :return: The flow rate (bbl/day or ft³/day)
        """
        return super().get_flow_rate(
            pressure=pressure,
            temperature=temperature,
            phase_mobility=phase_mobility,
            well_index=well_index,
            fluid=fluid,
            use_pseudo_pressure=use_pseudo_pressure,
            fluid_compressibility=fluid_compressibility,
            formation_volume_factor=formation_volume_factor,
            pvt_tables=pvt_tables,
        )


@attrs.define(slots=True, hash=True)
class ProductionWell(Well[WellLocation, ProducedFluid]):
    """
    Models a production well in the reservoir model.

    This well produces fluids from the reservoir.
    """

    produced_fluids: typing.Sequence[ProducedFluid] = attrs.field(factory=list)
    """List of fluids produced by the well. This can include multiple phases (e.g., oil, gas, water)."""


InjectionWellT = typing.TypeVar("InjectionWellT", bound=InjectionWell)
ProductionWellT = typing.TypeVar("ProductionWellT", bound=ProductionWell)


def _expand_interval(
    interval: typing.Tuple[WellLocation, WellLocation], orientation: Orientation
) -> typing.List[WellLocation]:
    """Expand a well perforating interval into a list of grid locations."""
    start, end = interval
    dimensions = len(start)
    if dimensions < 2:
        raise ValidationError("2D/3D locations are required")

    # Normalize start and end to ensure ranges are valid regardless of order
    start = tuple(min(s, e) for s, e in zip(start, end))
    end = tuple(max(s, e) for s, e in zip(start, end))

    if dimensions == 2:
        start = start + (0,)
        end = end + (0,)
        dimensions = 3  # Pad to 3D for uniform logic

    # Create iterator for the correct orientation
    if orientation == Orientation.X:
        locations = list(
            itertools.product(
                range(start[0], end[0] + 1),
                [start[1]],
                [start[2]],
            )
        )
    elif orientation == Orientation.Y:
        locations = list(
            itertools.product(
                [start[0]],
                range(start[1], end[1] + 1),
                [start[2]],
            )
        )
    elif orientation == Orientation.Z:
        locations = list(
            itertools.product(
                [start[0]],
                [start[1]],
                range(start[2], end[2] + 1),
            )
        )
    else:
        raise ValidationError("Invalid well orientation")

    return typing.cast(typing.List[WellLocation], locations)


def _expand_intervals(
    intervals: typing.Sequence[typing.Tuple[WellLocation, WellLocation]],
    orientation: Orientation,
) -> typing.List[WellLocation]:
    """Expand multiple well perforating intervals into a list of grid locations."""
    locations = []
    for interval in intervals:
        locations.extend(_expand_interval(interval=interval, orientation=orientation))
    return locations


def _prepare_wells_map(
    wells: typing.Sequence[WellT],
) -> typing.Dict[typing.Tuple[int, ...], WellT]:
    """Prepare the wells map for quick access."""
    wells_map = {
        loc: well
        for well in wells
        for loc in _expand_intervals(
            intervals=well.perforating_intervals,
            orientation=well.orientation,
        )
    }
    return wells_map


@attrs.frozen(slots=True)
class WellsProxy(typing.Generic[WellLocation, WellT]):
    """A proxy class for quick access to wells by their location."""

    wells: typing.Sequence[WellT]
    """A map of well perforating intervals to the well objects."""

    wells_map: typing.Dict[WellLocation, WellT] = attrs.field(init=False)
    """A map to store wells by their location for quick access."""
    check_interval_overlap: bool = True
    """
    Whether to check for overlapping perforating intervals between wells.

    You can disable this check if you are certain there are no overlapping wells or
    you want to allow overlapping wells (e.g in multi-layered reservoirs or multi-lateral wells).
    """

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "wells_map", _prepare_wells_map(self.wells))
        if not self.check_interval_overlap:
            return
        # Check for overlapping wells
        expected_location_count = sum(
            len(_expand_intervals(well.perforating_intervals, well.orientation))
            for well in self.wells
        )
        actual_location_count = len(self.wells_map)
        if expected_location_count != actual_location_count:
            raise ValidationError(
                f"Overlapping wells found at some locations. Expected {expected_location_count} unique locations, but got {actual_location_count}."
            )

    def __getitem__(self, location: WellLocation) -> typing.Optional[WellT]:
        """Get a well by its location."""
        return self.wells_map.get(location, None)

    def __setitem__(self, location: WellLocation, well: WellT) -> None:
        """Set a well at a specific location."""
        self.wells_map[location] = well


@attrs.define(slots=True)
class Wells(typing.Generic[WellLocation]):
    """
    Models a collection of injection and production wells in the reservoir model.

    This includes both production and injection wells.
    """

    injection_wells: typing.Sequence[InjectionWell[WellLocation]] = attrs.field(
        factory=list
    )
    """List of injection wells in the reservoir."""
    production_wells: typing.Sequence[ProductionWell[WellLocation]] = attrs.field(
        factory=list
    )
    """List of production wells in the reservoir."""
    injectors: WellsProxy[WellLocation, InjectionWell[WellLocation]] = attrs.field(
        init=False
    )
    """
    Proxy for injection wells.

    This allows quick access to injection wells by their location.
    """
    producers: WellsProxy[WellLocation, ProductionWell[WellLocation]] = attrs.field(
        init=False
    )
    """
    Proxy for production wells.

    This allows quick access to production wells by their location.
    """
    check_interval_overlap: bool = True
    """
    Whether to check for overlapping perforating intervals between injection wells and/or production wells.
    
    You can disable this check if you are certain there are no overlapping wells or
    you want to allow overlapping wells (e.g in multi-layered reservoirs or multi-lateral wells).
    """

    def __attrs_post_init__(self) -> None:
        self.injectors = WellsProxy(
            wells=self.injection_wells,
            check_interval_overlap=self.check_interval_overlap,
        )
        self.producers = WellsProxy(
            wells=self.production_wells,
            check_interval_overlap=self.check_interval_overlap,
        )

        if self.check_interval_overlap:
            # Check for overlapping wells. Injection and production wells should not overlap.
            overlapping_locations = set(self.injectors.wells_map).intersection(
                self.producers.wells_map
            )
            if overlapping_locations:
                raise ValidationError(
                    f"Overlapping wells found at locations: {overlapping_locations}"
                )

    def __getitem__(
        self, location: WellLocation
    ) -> typing.Tuple[
        typing.Optional[InjectionWell[WellLocation]],
        typing.Optional[ProductionWell[WellLocation]],
    ]:
        """
        Get a well by its grid coordinates.

        :param location: The (i, j) coordinates of the well in the reservoir grid.
        :return: Well or None: The well at the specified location, or None if not found.
        """
        return self.injectors[location], self.producers[location]

    @property
    def locations(
        self,
    ) -> typing.Tuple[typing.List[WellLocation], typing.List[WellLocation]]:
        """
        Get the starting locations of all wells in the reservoir.

        :return: A tuple of (injection_well_locations, production_well_locations).
        This returns a tuple containing two lists:
            - A list of locations for injection wells (starting location of first interval).
            - A list of locations for production wells (starting location of first interval).
        """
        injection_well_heads = []
        production_well_heads = []
        for well in self.injection_wells:
            if well.perforating_intervals:
                injection_well_heads.append(well.perforating_intervals[0][0])

        for well in self.production_wells:
            if well.perforating_intervals:
                production_well_heads.append(well.perforating_intervals[0][0])
        return injection_well_heads, production_well_heads

    @property
    def names(self) -> typing.Tuple[typing.List[str], typing.List[str]]:
        """
        Get all well names in the reservoir.

        :return: A tuple of (injection_well_names, production_well_names).
        This returns a tuple containing two lists:
            - A list of names for injection wells.
            - A list of names for production wells.
        """
        return (
            [well.name for well in self.injection_wells],
            [well.name for well in self.production_wells],
        )

    def evolve(self, model_state) -> None:
        """
        Evolve all wells in the reservoir model for the next time step.

        This method updates the state of each well based on its schedule.

        :param model_state: The current model state in the simulation.
        """
        for well in itertools.chain(self.injection_wells, self.production_wells):
            well.evolve(model_state)

    def check_location(self, grid_shape: typing.Tuple[int, ...]) -> None:
        """
        Check if all wells' perforating intervals are within the grid dimensions.

        :param grid_shape: The shape of the reservoir grid (nx, ny, nz).
        :raises ValidationError: If any well's perforating interval is out of bounds.
        """
        for well in itertools.chain(self.injection_wells, self.production_wells):
            well.check_location(grid_shape)

    def exists(self) -> bool:
        """
        Check if there are any wells in the reservoir model.

        :return: True if there are injection or production wells, False otherwise.
        """
        return bool(self.injection_wells or self.production_wells)
