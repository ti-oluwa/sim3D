import functools
import itertools
import logging
import math
import typing

import attrs
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from typing_extensions import Self

from sim3D.constants import c
from sim3D.pvt import (
    compute_gas_compressibility,
    compute_gas_compressibility_factor,
    compute_gas_density,
    compute_gas_formation_volume_factor,
    compute_gas_free_water_formation_volume_factor,
    compute_gas_viscosity,
    compute_water_compressibility,
    compute_water_density,
    compute_water_formation_volume_factor,
    compute_water_viscosity,
    fahrenheit_to_rankine,
)
from sim3D.types import (
    ActionFunc,
    FluidPhase,
    HookFunc,
    Orientation,
    ThreeDimensions,
    TwoDimensions,
    WellLocation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WellFluid",
    "InjectedFluid",
    "ProducedFluid",
    "InjectionWell",
    "ProductionWell",
    "Wells",
    "WellEvent",
    "well_time_hook",
    "well_hooks",
    "well_update_action",
    "well_actions",
    "compute_well_index",
    "compute_3D_effective_drainage_radius",
    "compute_2D_effective_drainage_radius",
    "compute_oil_well_rate",
    "compute_gas_well_rate",
    "_expand_intervals",
]


@attrs.define(slots=True, frozen=True)
class WellFluid:
    """Base class for fluid properties in wells."""

    name: str
    """Name of the fluid. Examples: Methane, CO2, Water, Oil."""
    phase: FluidPhase
    """Phase of the fluid. Examples: WATER, GAS, OIL."""
    specific_gravity: float = attrs.field(validator=attrs.validators.ge(0))
    """Specific gravity of the fluid in (lbm/ft³)."""
    molecular_weight: float = attrs.field(validator=attrs.validators.ge(0))
    """Molecular weight of the fluid in (g/mol)."""

    @functools.cache
    def get_pseudo_pressure_table(
        self,
        temperature: float = 60.0,
        reference_pressure: float = 14.7,
        pressure_range: typing.Tuple[float, float] = (14.7, 147000.0),
        points: int = 1000,
    ) -> "GasPseudoPressureTable":
        """
        Gas pseudo-pressure table for this fluid.

        :param temperature: The temperature at which to evaluate the pseudo-pressure table (°F).
        :param reference_pressure: The reference pressure for the pseudo-pressure table (psi).
        :param pressure_range: The pressure range for the pseudo-pressure table (psi).
        :param points: The number of points in the pseudo-pressure table.
        :return: A `GasPseudoPressureTable` instance for the fluid.
        """
        if self.phase != FluidPhase.GAS:
            raise ValueError("Pseudo-pressure table is only applicable for gas phase.")

        @functools.lru_cache(maxsize=1024)
        def z_factor_func(pressure: float) -> float:
            return compute_gas_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=self.specific_gravity,
            )

        def viscosity_func(pressure: float) -> float:
            return compute_gas_viscosity(
                temperature=temperature,
                gas_density=compute_gas_density(
                    pressure=pressure,
                    temperature=temperature,
                    gas_gravity=self.specific_gravity,
                    gas_compressibility_factor=z_factor_func(pressure),
                ),
                gas_molecular_weight=self.molecular_weight,
            )

        return GasPseudoPressureTable(
            z_factor_func=z_factor_func,
            viscosity_func=viscosity_func,
            reference_pressure=reference_pressure,
            pressure_range=pressure_range,
            points=points,
        )


@attrs.define(slots=True, frozen=True)
class InjectedFluid(WellFluid):
    """Properties of the fluid being injected into or produced by a well."""

    salinity: typing.Optional[float] = None
    """Salinity of the fluid (if water) in (ppm NaCl)."""
    is_miscible: bool = False
    """Whether this fluid is miscible with oil (e.g., CO2, N2)"""
    todd_longstaff_omega: float = attrs.field(
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0), attrs.validators.le(1.0)
        ),
        default=0.67,
    )
    """Todd-Longstaff mixing parameter for miscible displacement (0 to 1)."""
    minimum_miscibility_pressure: typing.Optional[float] = None
    """Minimum miscibility pressure for this fluid-oil system (psi)"""
    miscibility_transition_width: float = attrs.field(
        default=500.0, validator=attrs.validators.ge(0)
    )
    """Pressure range over which miscibility transitions from immiscible to miscible (psi)"""
    concentration: float = attrs.field(
        default=1.0,
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0), attrs.validators.le(1.0)
        ),
    )
    """Concentration of the fluid in the mixture (0 to 1). Relevant for miscible fluids."""

    def __attrs_post_init__(self) -> None:
        """Validate the fluid properties."""
        if self.phase not in (FluidPhase.GAS, FluidPhase.WATER):
            raise ValueError("Only gases and water are supported for injection.")

        if self.is_miscible:
            if self.phase != FluidPhase.GAS:
                raise ValueError("Only gas phase fluids can be miscible.")
            elif not self.minimum_miscibility_pressure or not self.todd_longstaff_omega:
                raise ValueError(
                    "Miscible fluids must have both `minimum_miscibility_pressure` and `todd_longstaff_omega` defined."
                )

    def get_viscosity(
        self, pressure: float, temperature: float, **kwargs: typing.Any
    ) -> float:
        """
        Get the viscosity of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the viscosity (psi).
        :param temperature: The temperature at which to evaluate the viscosity (°F).
        :kwargs: Additional parameters for viscosity calculations.
        :return: The viscosity of the fluid (cP).
        """
        if self.phase == FluidPhase.WATER:
            return compute_water_viscosity(
                pressure=pressure,
                temperature=temperature,
                salinity=self.salinity or 0.0,
            )

        gas_density = kwargs.get("gas_density", None)
        if gas_density is None:
            gas_z_factor = kwargs.get("gas_compressibility_factor", None)
            if gas_z_factor is None:
                gas_z_factor = compute_gas_compressibility_factor(
                    pressure=pressure,
                    temperature=temperature,
                    gas_gravity=self.specific_gravity,
                )
            gas_density = compute_gas_density(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=self.specific_gravity,
                gas_compressibility_factor=gas_z_factor,
            )
        return compute_gas_viscosity(
            temperature=temperature,
            gas_density=gas_density,
            gas_molecular_weight=self.molecular_weight,
        )

    def get_compressibility(
        self, pressure: float, temperature: float, **kwargs: typing.Any
    ) -> float:
        """
        Get the compressibility of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the compressibility (psi).
        :param temperature: The temperature at which to evaluate the compressibility (°F).
        :kwargs: Additional parameters for compressibility calculations.

        For water:
            :kwarg bubble_point_pressure: The bubble point pressure (psi).
            :kwarg gas_formation_volume_factor: The gas formation volume factor (ft³/scf).
            :kwarg gas_solubility_in_water: The gas solubility in water (scf/stb).

        For gas:
            :kwarg gas_gravity: The specific gravity of the gas (dimensionless). Optional
                Uses the fluid's specific gravity if not provided.
            :kwarg gas_compressibility_factor: The gas compressibility factor (dimensionless).

        :return: The compressibility of the fluid (psi⁻¹).
        """
        if self.phase == FluidPhase.WATER:
            gas_free_water_fvf = kwargs.get(
                "gas_free_water_formation_volume_factor", None
            )
            if gas_free_water_fvf is None:
                gas_free_water_fvf = compute_gas_free_water_formation_volume_factor(
                    pressure=pressure, temperature=temperature
                )
                kwargs["gas_free_water_formation_volume_factor"] = gas_free_water_fvf

            return compute_water_compressibility(
                pressure=pressure,
                temperature=temperature,
                **kwargs,
                salinity=self.salinity or 0.0,
            )

        kwargs.setdefault("gas_gravity", self.specific_gravity)
        return compute_gas_compressibility(
            pressure=pressure, temperature=temperature, **kwargs
        )

    def get_formation_volume_factor(
        self, pressure: float, temperature: float, **kwargs: typing.Any
    ) -> float:
        """
        Get the formation volume factor of the fluid at given pressure and temperature.

        :param pressure: The pressure at which to evaluate the formation volume factor (psi).
        :param temperature: The temperature at which to evaluate the formation volume factor (°F).
        :kwargs: Additional parameters for formation volume factor calculations.
        :return: The formation volume factor of the fluid (bbl/STB or ft³/SCF).
        """
        if self.phase == FluidPhase.WATER:
            water_density = kwargs.get("water_density", None)
            if water_density is None:
                # Not need for gas free fvf or gas fvf, since injection water
                # is typically gas free fresh water or degassed formation water
                water_density = compute_water_density(
                    pressure=pressure,
                    temperature=temperature,
                    salinity=self.salinity or 0.0,
                )
            return compute_water_formation_volume_factor(
                salinity=self.salinity or 0.0,
                water_density=water_density,
            )

        gas_z_factor = kwargs.get("gas_compressibility_factor", None)
        if gas_z_factor is None:
            gas_z_factor = compute_gas_compressibility_factor(
                pressure=pressure,
                temperature=temperature,
                gas_gravity=self.specific_gravity,
            )
        return compute_gas_formation_volume_factor(
            pressure=pressure,
            temperature=temperature,
            gas_compressibility_factor=gas_z_factor,
        )


@attrs.define(slots=True, frozen=True)
class ProducedFluid(WellFluid):
    """Properties of the fluid being produced by a well."""

    pass


WellFluidT = typing.TypeVar("WellFluidT", bound=WellFluid)


def _geometric_mean(values: typing.Sequence[float]) -> float:
    prod = 1.0
    n = 0
    for v in values:
        prod *= max(v, 0.0)  # ensure non-negative
        n += 1
    if n == 0:
        raise ValueError("No permeability values provided")
    return prod ** (1.0 / n)


def compute_effective_permeability_for_well(
    permeability: typing.Sequence[float], orientation: Orientation
) -> float:
    """
    Compute k_eff for Peaceman WI using geometric mean of the two permeabilities
    perpendicular to the well axis. `permeability` is (kx, ky, kz).
    orientation is one of Orientation.X/Y/Z (or a string equivalent).
    """
    if len(permeability) != 3:
        # If 2D, fall back to geometric mean of available components:
        return _geometric_mean(permeability)

    kx, ky, kz = permeability
    if orientation == Orientation.Z:  # vertical well: transverse are x,y
        return math.sqrt(max(kx, 0.0) * max(ky, 0.0))
    elif orientation == Orientation.X:  # well along x: transverse are y,z
        return math.sqrt(max(ky, 0.0) * max(kz, 0.0))
    elif orientation == Orientation.Y:  # well along y: transverse are x,z
        return math.sqrt(max(kx, 0.0) * max(kz, 0.0))
    # Oblique/unknown orientation: conservative fallback = geometric mean of all three
    return _geometric_mean((kx, ky, kz))


@attrs.define(slots=True, hash=True)
class Well(typing.Generic[WellLocation, WellFluidT]):
    """Models a well in the reservoir model."""

    name: str
    """Name of the well."""
    perforating_intervals: typing.Sequence[typing.Tuple[WellLocation, WellLocation]]
    """Perforating intervals of the well. Each interval is a tuple of (start_location, end_location)."""
    radius: float
    """Radius of the wellbore (ft)."""
    bottom_hole_pressure: float
    """Well bottom-hole flowing pressure in psi"""
    skin_factor: float = 0.0
    """Skin factor for the well, affecting flow performance."""
    orientation: Orientation = attrs.field(init=False, default=Orientation.Z)
    """Orientation of the well, indicating its dominant direction in the reservoir grid."""
    is_active: bool = True
    """Indicates whether the well is active or not. Set to False if the well is shut in or inactive."""
    schedule: typing.Set["WellEvent[Self]"] = attrs.field(factory=set)  # type: ignore
    """Schedule of events for the well, mapping time steps to scheduled events."""
    auto_clamp: bool = True
    """
    Whether to automatically clamp the well rate to zero if the flow direction is reversed.

    If True, Production wells with positive flow rates and Injection wells with negative flow rates will be automatically clamped.
    If False, wells will remain active regardless of flow direction.

    It is advised to keep this option enabled to prevent unphysical scenarios in the simulation.
    """

    def __attrs_post_init__(self) -> None:
        """Ensure the well has a valid orientation."""
        if abs(self.bottom_hole_pressure) != self.bottom_hole_pressure:
            raise ValueError(
                "Well bottom-hole flowing pressure must be a positive value."
            )
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
        :raises ValueError: If any of the well's perforating intervals are out of bounds.
        """
        for interval_idx, (start, end) in enumerate(self.perforating_intervals):
            if not all(0 <= coord < dim for coord, dim in zip(start, grid_dimensions)):
                raise ValueError(
                    f"Start location {start} for interval {interval_idx} of well {self.name!r} is out of bounds."
                )
            if not all(0 <= coord < dim for coord, dim in zip(end, grid_dimensions)):
                raise ValueError(
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
            raise ValueError("2D/3D locations are required")

        if dimensions == 2:
            if len(permeability) != 2:
                raise ValueError("Permeability must be a 2D tuple for 2D locations")
            interval_thickness = typing.cast(TwoDimensions, interval_thickness)
            permeability = typing.cast(TwoDimensions, permeability)
            return compute_2D_effective_drainage_radius(
                interval_thickness=interval_thickness,
                permeability=permeability,
                well_orientation=self.orientation,
            )

        if len(permeability) != 3:
            raise ValueError("Permeability must be a 3D tuple for 3D locations")
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
            raise ValueError("2D/3D locations are required")

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
            raise ValueError("Z-oriented wells are not supported in 2D models")

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
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        formation_volume_factor: typing.Optional[float] = None,
    ) -> float:
        """
        Compute the flow rate for the well using Darcy's law.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The relative mobility of the fluid phase being produced or injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected. If None, uses the well's injected_fluid.
        :param formation_volume_factor: The formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹). For slightly compressible fluids, this can be used to adjust the flow rate calculation.
        :param formation_volume_factor: Formation volume factor of the fluid (bbl/STB or ft³/SCF).
        :return: The flow rate in (bbl/day or ft³/day).
        """
        # If no fluid is injected, return 0 flow rate
        if fluid is None or phase_mobility <= 0.0 or not self.is_open:
            return 0.0

        if fluid.phase == FluidPhase.GAS:
            pseudo_pressure_table = None
            # Only use pseudo-pressure for gas wells above threshold
            if use_pseudo_pressure and pressure > c.GAS_PSEUDO_PRESSURE_THRESHOLD:
                pseudo_pressure_table = fluid.get_pseudo_pressure_table(
                    temperature=temperature, points=c.GAS_PSEUDO_PRESSURE_POINTS
                )
            else:
                use_pseudo_pressure = False

            avg_pressure = (pressure + self.bottom_hole_pressure) * 0.5
            avg_compressibility_factor = compute_gas_compressibility_factor(
                pressure=avg_pressure,
                temperature=temperature,
                gas_gravity=fluid.specific_gravity,
            )
            return compute_gas_well_rate(
                well_index=well_index,
                pressure=pressure,
                temperature=temperature,
                bottom_hole_pressure=self.bottom_hole_pressure,
                phase_mobility=phase_mobility,
                use_pseudo_pressure=bool(use_pseudo_pressure),
                pseudo_pressure_table=pseudo_pressure_table,
                average_compressibility_factor=avg_compressibility_factor,
                formation_volume_factor=formation_volume_factor,
            )

        # For water and oil wells
        return compute_oil_well_rate(
            well_index=well_index,
            pressure=pressure,
            bottom_hole_pressure=self.bottom_hole_pressure,
            phase_mobility=phase_mobility,
            fluid_compressibility=fluid_compressibility,
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

    def __reduce_ex__(
        self, protocol: typing.SupportsIndex
    ) -> typing.Tuple[typing.Any, ...]:
        """
        Custom pickle reduction that excludes the schedule attribute.

        Returns a tuple of (callable, args) where callable(*args) reconstructs the object.
        The schedule is replaced with an empty set to avoid pickling non-serializable hooks.
        Only init=True fields are included in the reconstruction.

        :param protocol: The pickle protocol version.
        :return: A tuple for pickle reconstruction.
        """
        # Get all init field values except schedule
        field_values = {}
        for field in attrs.fields(type(self)):
            # Skip non-init fields (like orientation) and schedule
            if not field.init:
                continue

            if field.name == "schedule":
                # Replace schedule with empty set
                field_values[field.name] = set()
            else:
                field_values[field.name] = getattr(self, field.name)

        # Return constructor and kwargs for reconstruction
        return (_reconstruct_well, (type(self), field_values))


def _reconstruct_well(
    cls: typing.Type, field_values: typing.Dict[str, typing.Any]
) -> typing.Any:
    """
    Helper function to reconstruct a Well instance from pickled data.

    :param cls: The Well class to instantiate.
    :param field_values: Dictionary of field names to values.
    :return: A reconstructed Well instance.
    """
    return cls(**field_values)


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


def well_time_hook(
    time_step: typing.Optional[int] = None, time: typing.Optional[float] = None
):
    """
    Returns a hook function that triggers based on the simulation time step or time.

    :param time_step: The specific time step at which to trigger the event.
    :param time: The specific simulation time at which to trigger the event.
    :return: A hook function that takes a well and model state as arguments and returns a boolean indicating whether to apply the event.
    """

    if not (time_step or time):
        raise ValueError("Either time_step or time must be provided.")

    def hook(well: Well, model_state: typing.Any) -> bool:
        if time_step is not None and model_state.time_step == time_step:
            return True
        if time is not None and model_state.time == time:
            return True
        return False

    return hook


def well_hooks(
    *hooks: HookFunc[Well, typing.Any], on_any: bool = False
) -> HookFunc[Well, typing.Any]:
    """
    Composes hook functions to be executed in sequence.

    :param hooks: A sequence of hook functions to be chained.
    :param on_any: If True, the composite hook returns True if any of the hooks return True.
                   If False, it returns True only if all hooks return True.
    :return: A composite hook function that takes a well and model state as arguments and returns a boolean indicating whether to apply the event.
    """

    if not hooks:
        raise ValueError("At least one hook must be provided to chain.")

    def hook(well: Well, model_state: typing.Any) -> bool:
        results = (h(well, model_state) for h in hooks)
        return any(results) if on_any else all(results)

    return hook


def well_update_action(
    bottom_hole_pressure: typing.Optional[float] = None,
    skin_factor: typing.Optional[float] = None,
    is_active: typing.Optional[bool] = None,
    injected_fluid: typing.Optional[InjectedFluid] = None,
    produced_fluids: typing.Optional[typing.Sequence[ProducedFluid]] = None,
) -> ActionFunc[Well, typing.Any]:
    """
    Returns an action function that modifies well configuration.

    :param bottom_hole_pressure: New bottom-hole pressure for the well (psi).
    :param skin_factor: New skin factor for the well.
    :param is_active: New active status for the well (True for open, False for shut in).
    :param injected_fluid: New fluid properties for injection wells.
    :param produced_fluids: New fluid properties for production wells.
    :return: An action function that takes a well and model state as arguments and performs the property updates.
    """
    valid = any(
        param is not None
        for param in [
            bottom_hole_pressure,
            skin_factor,
            is_active,
            injected_fluid,
            produced_fluids,
        ]
    )
    if not valid:
        raise ValueError("At least one property must be provided to update.")

    def action(well: Well, model_state: typing.Any) -> None:
        if bottom_hole_pressure is not None:
            if abs(bottom_hole_pressure) != bottom_hole_pressure:
                raise ValueError(
                    "Well bottom-hole flowing pressure must be a positive value."
                )
            well.bottom_hole_pressure = bottom_hole_pressure
        if skin_factor is not None:
            well.skin_factor = skin_factor
        if is_active is True:
            well.open()
        elif is_active is False:
            well.shut_in()

        if injected_fluid is not None and isinstance(well, InjectionWell):
            well.injected_fluid = injected_fluid
        if produced_fluids is not None and isinstance(well, ProductionWell):
            well.produced_fluids = produced_fluids
        return

    return action


def well_actions(
    *actions: ActionFunc[Well, typing.Any],
) -> ActionFunc[Well, typing.Any]:
    """
    Composes action functions to be executed in sequence.

    :param actions: A sequence of action functions to be chained.
    :return: A composite action function that takes a well and model state as arguments and performs all the actions in sequence.
    """

    if not actions:
        raise ValueError("At least one action must be provided to chain.")

    def action(well: Well, model_state: typing.Any) -> None:
        for act in actions:
            act(well, model_state)

    return action


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
        use_pseudo_pressure: bool = False,
        fluid_compressibility: typing.Optional[float] = None,
        formation_volume_factor: typing.Optional[float] = None,
    ) -> float:
        """
        Compute the flow rate for the injection well using Darcy's law.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The relative mobility of the fluid phase being injected.
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being injected into the well. If None, uses the well's injected_fluid property.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param fluid_compressibility: Compressibility of the fluid (psi⁻¹). For slightly compressible fluids, this can be used to adjust the flow rate calculation.
        :param formation_volume_factor: The formation volume factor of the fluid (bbl/STB or ft³/SCF).
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
        raise ValueError("2D/3D locations are required")

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
        raise ValueError("Invalid well orientation")

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


@attrs.define(slots=True, frozen=True)
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
            raise ValueError(
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

        if not self.check_interval_overlap:
            return

        # Check for overlapping wells. Injection and production wells should not overlap.
        overlapping_locations = set(self.injectors.wells_map).intersection(
            self.producers.wells_map
        )
        if overlapping_locations:
            raise ValueError(
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

    def check_location(self, grid_dimensions: typing.Tuple[int, ...]) -> None:
        """
        Check if all wells' perforating intervals are within the grid dimensions.

        :param grid_dimensions: The dimensions of the reservoir grid (i, j, k).
        :raises ValueError: If any well's perforating interval is out of bounds.
        """
        for well in itertools.chain(self.injection_wells, self.production_wells):
            well.check_location(grid_dimensions)

    def has_wells(self) -> bool:
        """
        Check if there are any wells in the reservoir model.

        :return: True if there are injection or production wells, False otherwise.
        """
        return bool(self.injection_wells or self.production_wells)

    def __reduce_ex__(
        self, protocol: typing.SupportsIndex
    ) -> typing.Tuple[typing.Any, ...]:
        """
        Custom pickle reduction that clears schedules from all wells.

        Returns a tuple of (callable, args) where callable(*args) reconstructs the object.
        All well schedules are cleared to avoid pickling non-serializable hooks.
        Only fields with init=True are included.

        :param protocol: The pickle protocol version.
        :return: A tuple for pickle reconstruction.
        """
        # Clear schedules from all wells
        injection_wells_cleared = [
            attrs.evolve(well, schedule=set()) for well in self.injection_wells
        ]
        production_wells_cleared = [
            attrs.evolve(well, schedule=set()) for well in self.production_wells
        ]

        # Get only init=True field values
        field_values = {}
        for field in attrs.fields(type(self)):
            # Skip non-init fields (injectors, producers)
            if not field.init:
                continue

            if field.name == "injection_wells":
                field_values[field.name] = injection_wells_cleared
            elif field.name == "production_wells":
                field_values[field.name] = production_wells_cleared
            else:
                field_values[field.name] = getattr(self, field.name)

        # Return constructor and kwargs for reconstruction
        return (_reconstruct_wells, (type(self), field_values))


def _reconstruct_wells(
    cls: typing.Type, field_values: typing.Dict[str, typing.Any]
) -> typing.Any:
    """
    Helper function to reconstruct a Wells instance from pickled data.

    :param cls: The Wells class to instantiate.
    :param field_values: Dictionary of field names to values.
    :return: A reconstructed Wells instance.
    """
    return cls(**field_values)


def compute_well_index(
    permeability: float,
    interval_thickness: float,
    wellbore_radius: float,
    effective_drainage_radius: float,
    skin_factor: float = 0.0,
) -> float:
    """
    Compute the well index for a given well using the Peaceman equation.

    The well index is a measure of the productivity of a well, defined as the ratio of the
    well flow rate to the pressure drop across the well.

    The formula for the well index is:
    W = (k * h) / (ln(re/rw) + s)

    where:
        - W is the well index (md*ft)
        - k is the absolute permeability of the reservoir rock (mD)
        - h is the thickness of the reservoir interval (ft)
        - re is the effective drainage radius (ft)
        - rw is the wellbore radius (ft)
        - s is the skin factor (dimensionless, default is 0)

    :param permeability: Absolute permeability of the reservoir rock (mD).
    :param interval_thickness: Thickness of the reservoir interval (ft).
    :param wellbore_radius: Radius of the wellbore (ft).
    :param effective_drainage_radius: Effective drainage radius (ft).
    :param skin_factor: Skin factor for the well (dimensionless, default is 0).
    :return: The well index in (mD*ft).
    """
    well_index = (permeability * interval_thickness) / (
        np.log(effective_drainage_radius / wellbore_radius) + skin_factor
    )
    return well_index


def compute_3D_effective_drainage_radius(
    interval_thickness: ThreeDimensions,
    permeability: ThreeDimensions,
    well_orientation: Orientation,
) -> float:
    """
    Compute the effective drainage radius for a well ina 3D reservoir model using
    Peaceman's effective drainage radius formula.

    The formula for is given by:

    For x-direction:

        r_x = 0.28 * √[ (∆y² + ∆z²) / (√(k_y / k_z) + √(k_z / k_y)) ]

    For y-direction:

        r_y = 0.28 * √[ (∆x² + ∆z²) / (√(k_x / k_z) + √(k_z / k_x)) ]

    For z-direction:
        r_z = 0.28 * √[ (∆x² + ∆y²) / (√(k_x / k_y) + √(k_y / k_x)) ]

    where:
        - r_x, r_y, r_z are the effective drainage radii in the x, y, and z directions respectively.
        - ∆x, ∆y, ∆z are the thicknesses of the reservoir interval in the x, y, and z directions respectively.
        - k_x, k_y, k_z are the permeabilities of the reservoir rock in the x, y, and z directions respectively.

    :param interval_thickness: A tuple representing the thickness of the reservoir interval in the x, y, and z directions (ft).
    :param permeability: A tuple representing the permeability of the reservoir rock in the x, y, and z directions (mD).
    :param well_orientation: The orientation of the well (Orientation.X, Orientation.Y, or Orientation.Z).
    :return: The effective drainage radius in the direction of the well (ft).
    """
    if well_orientation == Orientation.X:
        delta_y, delta_z = interval_thickness[1], interval_thickness[2]
        k_y, k_z = permeability[1], permeability[2]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_y**2 + delta_z**2) / (np.sqrt(k_y / k_z) + np.sqrt(k_z / k_y))
        )
    elif well_orientation == Orientation.Y:
        delta_x, delta_z = interval_thickness[0], interval_thickness[2]
        k_x, k_z = permeability[0], permeability[2]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_x**2 + delta_z**2) / (np.sqrt(k_x / k_z) + np.sqrt(k_z / k_x))
        )
    elif well_orientation == Orientation.Z:
        delta_x, delta_y = interval_thickness[0], interval_thickness[1]
        k_x, k_y = permeability[0], permeability[1]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_x**2 + delta_y**2) / (np.sqrt(k_x / k_y) + np.sqrt(k_y / k_x))
        )
    else:
        raise ValueError("Invalid well orientation")

    return effective_drainage_radius


def compute_2D_effective_drainage_radius(
    interval_thickness: TwoDimensions,
    permeability: TwoDimensions,
    well_orientation: Orientation,
) -> float:
    """
    Compute the effective drainage radius for a well in a 2D reservoir model.

    The formula for is given by:

        r = 0.28 * √[ ( (∆x² * √(k_y / k_x)) + (∆y² * √(k_x / k_y)) ) / ( √(k_y / k_x) + √(k_x / k_y) ) ]

    where:
        - r_x, r_y are the effective drainage radii in the x and y directions respectively.
        - ∆x, ∆y are the thicknesses of the reservoir interval in the x and y directions respectively.
        - k_x, k_y are the permeabilities of the reservoir rock in the x and y directions respectively.

    :param interval_thickness: A tuple representing the thickness of the reservoir interval in the x and y directions (ft).
    :param permeability: A tuple representing the permeability of the reservoir rock in the x and y directions (mD).
    :param well_orientation: The orientation of the well (Orientation.X or Orientation.Y).
    :return: The effective drainage radius in the direction of the well (ft).
    """
    if well_orientation == Orientation.X:
        delta_x, delta_y = interval_thickness[0], interval_thickness[1]
        k_x, k_y = permeability[0], permeability[1]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_x**2 * np.sqrt(k_y / k_x) + delta_y**2 * np.sqrt(k_x / k_y))
            / (np.sqrt(k_y / k_x) + np.sqrt(k_x / k_y))
        )
    elif well_orientation == Orientation.Y:
        delta_x, delta_y = interval_thickness[0], interval_thickness[1]
        k_x, k_y = permeability[0], permeability[1]
        effective_drainage_radius = 0.28 * np.sqrt(
            (delta_x**2 * np.sqrt(k_x / k_y) + delta_y**2 * np.sqrt(k_y / k_x))
            / (np.sqrt(k_x / k_y) + np.sqrt(k_y / k_x))
        )
    else:
        raise ValueError("Invalid well orientation")
    return effective_drainage_radius


def compute_gas_pseudo_pressure(
    pressure: float,
    z_factor_func: typing.Callable[[float], float],
    viscosity_func: typing.Callable[[float], float],
    reference_pressure: float = 14.7,
) -> float:
    """
    Compute the gas pseudo-pressure using Al-Hussainy real-gas potential.

    The pseudo-pressure is defined as:
        m(P) = ∫[P_ref to P] (2*P' / (μ(P') * Z(P'))) dP'

    This formulation accounts for gas compressibility and non-Darcy effects,
    allowing the use of standard liquid-like flow equations for gas.

    Physical Interpretation:
        - m(P) transforms the nonlinear gas diffusivity equation into a linear form
        - At low pressure: m(P) ≈ P² (ideal gas limit)
        - At high pressure: deviations due to Z-factor and viscosity changes

    :param pressure: Current pressure (psi)
    :param z_factor_func: Function returning Z-factor at given pressure Z(P)
    :param viscosity_func: Function returning viscosity at given pressure μ(P) in cP
    :param reference_pressure: Reference pressure for integration (psi), typically 14.7
    :return: Pseudo-pressure m(P) in psi²/cP

    References:
        Al-Hussainy, R., Ramey, H.J., and Crawford, P.B. (1966).
        "The Flow of Real Gases Through Porous Media."
        JPT, May 1966, pp. 624-636.
    """
    if pressure <= 0:
        raise ValueError(f"Pressure must be positive, got {pressure}")
    if reference_pressure <= 0:
        raise ValueError(
            f"Reference pressure must be positive, got {reference_pressure}"
        )

    # If pressure equals reference, pseudo-pressure is zero by definition
    if abs(pressure - reference_pressure) < 1e-6:
        return 0.0

    # Define the integrand: 2*P / (μ*Z)
    def integrand(P: float) -> float:
        """Integrand for pseudo-pressure calculation."""
        # Add safety checks to prevent division by zero
        Z = z_factor_func(P)
        mu = viscosity_func(P)

        if Z <= 0 or mu <= 0:
            raise ValueError(f"Invalid Z={Z} or μ={mu} at P={P}")

        return 2.0 * P / (mu * Z)

    # Perform numerical integration
    # Use higher accuracy for gas (epsabs, epsrel)
    try:
        if pressure > reference_pressure:
            result, error = quad(
                integrand,
                reference_pressure,
                pressure,
                epsabs=1e-8,  # Absolute error tolerance
                epsrel=1e-6,  # Relative error tolerance
                limit=100,  # Maximum number of subintervals
            )
            return float(result)
        else:
            # Integrate backwards and negate
            result, error = quad(
                integrand,
                pressure,
                reference_pressure,
                epsabs=1e-8,
                epsrel=1e-6,
                limit=100,
            )
            return -float(result)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compute pseudo-pressure at P={pressure} psi: {exc}"
        )


class GasPseudoPressureTable:
    """
    Pre-computed gas pseudo-pressure table for fast lookup during simulation.
    """

    def __init__(
        self,
        z_factor_func: typing.Callable[[float], float],
        viscosity_func: typing.Callable[[float], float],
        pressure_range: typing.Tuple[float, float] = (14.7, 14700.0),
        points: int = 1000,
        reference_pressure: float = 14.7,
    ):
        """
        Build pseudo-pressure lookup table.

        :param z_factor_func: Z-factor correlation Z(P)
        :param viscosity_func: Gas viscosity correlation μ(P)
        :param pressure_range: (P_min, P_max) for table
        :param points: Number of points in table
        :param reference_pressure: Reference pressure (psi)
        """
        self.reference_pressure = reference_pressure
        self.z_factor_func = z_factor_func
        self.viscosity_func = viscosity_func

        # Create pressure grid (log-spaced for better resolution at low P)
        min_pressure, max_pressure = pressure_range
        self.pressures = np.logspace(
            np.log10(min_pressure), np.log10(max_pressure), points
        )

        # Compute pseudo-pressure at each point
        logger.debug(f"Building pseudo-pressure table with {points} points...")
        self.pseudo_pressures = np.zeros(points)
        for i, pressure in enumerate(self.pressures):
            self.pseudo_pressures[i] = compute_gas_pseudo_pressure(
                pressure=pressure,
                z_factor_func=z_factor_func,
                viscosity_func=viscosity_func,
                reference_pressure=reference_pressure,
            )

        # Build cubic spline interpolator for fast lookup
        self.interpolator = interp1d(
            self.pressures,
            self.pseudo_pressures,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",  # type: ignore  # Extrapolate outside table range
        )
        logger.debug(
            f"Pseudo-pressure table built: P ∈ [{min_pressure:.1f}, {max_pressure:.1f}] psi"
        )

    def __call__(self, pressure: float) -> float:
        """
        Fast lookup of pseudo-pressure via interpolation.

        :param pressure: Pressure (psi)
        :return: Pseudo-pressure m(P) (psi²/cP)
        """
        return float(self.interpolator(pressure))

    def gradient(self, pressure: float) -> float:
        """
        Compute dm/dP = 2P/(μ*Z) for use in well models.

        :param pressure: Pressure (psi)
        :return: dm/dP (psi/cP)
        """
        Z = self.z_factor_func(pressure)
        mu = self.viscosity_func(pressure)
        return 2.0 * pressure / (mu * Z)


def compute_oil_well_rate(
    well_index: float,
    pressure: float,
    bottom_hole_pressure: float,
    phase_mobility: float,
    fluid_compressibility: typing.Optional[float] = None,
) -> float:
    """
    Compute the well rate using the well index and pressure drop.

    This assumes radial flow to/from the wellbore.
    May be steady-state or pseudo-steady-state flow depending on the well index calculation.

    The formula for the well rate is:

        Q = 7.08e-3 * W * (P_bhp - P) * M

    or for slightly compressible fluids:

        Q = 7.08e-3 * W * M * ln(1 + c_f * (P_bhp - P)) / c_f

    where:
        - Q is the well rate (bbl/day)
        - W is the well index (mD*ft)
        - P is the reservoir pressure (psi)
        - P_bhp is the bottom-hole pressure (psi)
        - M is the phase mobility (cP⁻¹, default is 1.0) (k_r / μ) or (k_r / (μ * B)).

    Negative rate result indicates that the well is producing, while positive rates indicate injection.

    :param well_index: The well index (mD*ft).
    :param pressure: The reservoir pressure (psi).
    :param bottom_hole_pressure: The bottom-hole pressure (psi).
    :param phase_mobility: The phase relative mobility (cP⁻¹, default is 1.0) (k_r / μ) or (k_r / (μ * B)).
    :param fluid_compressibility: The fluid compressibility (1/psi). For slightly compressible fluids.
    :return: The well rate in bbl/day.
    """
    if well_index <= 0:
        raise ValueError("Well index must be a positive value.")

    pressure_difference = bottom_hole_pressure - pressure
    if fluid_compressibility:
        well_rate = (
            7.08e-3
            * well_index
            * phase_mobility
            * np.log(1 + (fluid_compressibility * pressure_difference))
            / fluid_compressibility
        )
    else:
        well_rate = 7.08e-3 * well_index * phase_mobility * pressure_difference
    return well_rate


def compute_gas_well_rate(
    well_index: float,
    pressure: float,
    temperature: float,
    bottom_hole_pressure: float,
    phase_mobility: float,
    average_compressibility_factor: float = 1.0,
    use_pseudo_pressure: bool = True,
    pseudo_pressure_table: typing.Optional[GasPseudoPressureTable] = None,
    formation_volume_factor: typing.Optional[float] = None,
) -> float:
    """
    Compute the gas well rate using the well index and pressure drop.

    This assumes radial flow to/from the wellbore.
    May be steady-state or pseudo-steady-state flow depending on the well index calculation.

    The formula for the gas well rate is:

    For pseudo-pressure formulation:

        Q = 1.9875e-2 * (Tsc / Psc) * (W / T) * (m(P) - m(P_bhp))

    For pressure squared formulation:

        Q = 1.9875e-2 * (Tsc / Psc) * (W / T) * M * ((P² - P_bhp²) / Z)

    where:
        - Q is the gas well rate (SCF/day)
        - W is the well index (mD*ft)
        - P is the reservoir pressure (psi)
        - P_bhp is the bottom-hole pressure (psi)
        - m(P) is the pseudo-pressure at pressure P
        - T is the reservoir temperature (°F)
        - Tsc is the standard temperature (°R), typically 520 °R (60 °F)
        - Psc is the standard pressure (psi), typically 14.7 psi
        - M is the phase mobility (cP⁻¹, default is 1.0) (k_r / μ) or (k_r / (μ * B)).
        - Z_avg is the average compressibility factor in the reservoir interval.

    Negative rate result indicates that the well is producing, while positive rates indicate injection.

    :param well_index: The well index (mD*ft).
    :param pressure: The reservoir pressure (psi).
    :param temperature: The reservoir temperature (°F).
    :param bottom_hole_pressure: The bottom-hole pressure (psi).
    :param phase_mobility: The phase relative mobility (cP⁻¹, default is 1.0) (k_r / μ) or (k_r / (μ * B)).
    :param average_compressibility_factor: The average gas compressibility factor Z (default is 1.0).
    :param use_pseudo_pressure: Whether to use pseudo-pressure formulation (default is True).
    :param pseudo_pressure_table: Pre-computed pseudo-pressure table for fast lookup (required if use_pseudo_pressure is True).
    :param formation_volume_factor: Gas formation volume factor (ft³/SCF). If provided, it will be used directly instead of calculating from Z, T, and P.
    :return: The gas well rate (ft³/day).
    """
    if well_index <= 0:
        raise ValueError("Well index must be a positive value.")

    Tsc = c.STANDARD_TEMPERATURE_RANKINE
    Psc = c.STANDARD_PRESSURE_IMPERIAL
    temperature_rankine = fahrenheit_to_rankine(temperature)

    if use_pseudo_pressure:
        if pseudo_pressure_table is None:
            raise ValueError(
                "`pseudo_pressure_table` must be provided when use_pseudo_pressure is True."
            )

        bottom_hole_pseudo_pressure = pseudo_pressure_table(bottom_hole_pressure)
        reservoir_pseudo_pressure = pseudo_pressure_table(pressure)
        pseudo_pressure_difference = (
            bottom_hole_pseudo_pressure - reservoir_pseudo_pressure
        )
        well_rate = (
            1.9875e-2
            * (Tsc / Psc)
            * (well_index / temperature_rankine)
            * pseudo_pressure_difference
        )
    else:
        pressure_difference_squared = bottom_hole_pressure**2 - pressure**2
        well_rate = (
            1.9875e-2
            * (Tsc / Psc)
            * (well_index / temperature_rankine)
            * phase_mobility
            * (pressure_difference_squared / average_compressibility_factor)
        )

    if formation_volume_factor is not None:
        gas_fvf = formation_volume_factor
    else:
        # Compute gas formation volume factor (ft³/SCF)
        # Bg = 0.02827 * (Z_avg * T) / P_avg
        average_pressure = 0.5 * (pressure + bottom_hole_pressure)
        gas_fvf = (
            0.02827
            * average_compressibility_factor
            * temperature_rankine
            / average_pressure
        )  # ft³/SCF
    return well_rate * gas_fvf  # ft³/day
