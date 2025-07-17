import copy
import itertools
import typing
import enum
from attrs import define, field
import numpy as np

from sim3D.types import Orientation, WellLocation


__all__ = [
    "FluidPhase",
    "InjectedFluid",
    "ProducedFluid",
    "InjectionWell",
    "ProductionWell",
    "Wells",
    "ScheduledEvent",
    "InjectionWellScheduledEvent",
    "ProductionWellScheduledEvent",
    "compute_well_index",
    "compute_3D_effective_drainage_radius",
    "compute_2D_effective_drainage_radius",
    "compute_well_rate",
]


class FluidPhase(enum.Enum):
    """Enum representing the phase of the fluid in the reservoir."""

    WATER = "water"
    GAS = "gas"
    OIL = "oil"


@define(slots=True, frozen=True)
class _WellFluid:
    """Properties of the fluid being injected into or produced by a well."""

    name: str
    """Name of the injected fluid. Examples: Methane, CO2, Water, Oil."""
    phase: FluidPhase
    """Phase of the injected fluid. Examples: WATER, GAS, OIL."""


@define(slots=True, frozen=True)
class InjectedFluid(_WellFluid):
    """
    Models a fluid being injected into the reservoir.
    """

    density: float
    """Density of the injected fluid in (lbm/ft³). This is used to calculate the wellbore storage and skin effects."""
    viscosity: float
    """Viscosity of the injected fluid in (cP). This is used to calculate the wellbore storage and skin effects."""
    compressibility: float
    """Compressibility of the injected fluid in (psi⁻¹). This is used to calculate the wellbore storage and skin effects."""
    formation_volume_factor: float
    """Formation volume factor of the injected fluid in (bbl/STB) or (ft³/SCF), depending on the phase of the fluid injected."""
    salinity: float = 0.0
    """Salinity of the injected fluid in (ppm NaCl). This is used to calculate the wellbore storage and skin effects."""


@define(slots=True, frozen=True)
class ProducedFluid(_WellFluid):
    """
    Models a fluid being produced from the reservoir.
    """

    density: typing.Optional[float] = None
    """Density of the produced fluid in (lbm/ft³). This is used to calculate the wellbore storage and skin effects."""
    viscosity: typing.Optional[float] = None
    """Viscosity of the produced fluid in (cP). This is used to calculate the wellbore storage and skin effects."""
    compressibility: typing.Optional[float] = None
    """Compressibility of the produced fluid in (psi⁻¹). This is used to calculate the wellbore storage and skin effects."""
    formation_volume_factor: typing.Optional[float] = None
    """Formation volume factor of the produced fluid in (bbl/STB) or (ft³/SCF), depending on the phase of the fluid produced."""
    salinity: typing.Optional[float] = None
    """Salinity of the produced fluid in (ppm NaCl). This is used to calculate the wellbore storage and skin effects."""


@define(slots=True)
class _Well(typing.Generic[WellLocation]):
    """Models a well in the reservoir model."""

    name: str
    """Name of the well."""
    perforating_interval: typing.Tuple[WellLocation, WellLocation]
    """Perforating interval of the well."""
    radius: float
    """Radius of the wellbore (ft)."""
    bottom_hole_pressure: float
    """Well bottom-hole flowing pressure in psi"""
    skin_factor: float = 0.0
    """Skin factor for the well, affecting flow performance."""
    orientation: Orientation = field(init=False, default=Orientation.Z)
    """Orientation of the well, indicating its dominant direction in the reservoir grid."""
    is_active: bool = True
    """Indicates whether the well is active or not. Set to False if the well is shut in or inactive."""
    schedule: typing.MutableMapping[int, "ScheduledEvent"] = field(factory=dict)

    def __attrs_post_init__(self) -> None:
        """Ensure the well has a valid orientation."""
        if abs(self.bottom_hole_pressure) != self.bottom_hole_pressure:
            raise ValueError(
                "Well bottom-hole flowing pressure must be a positive value."
            )
        self.orientation = self.get_orientation()

    def get_orientation(self) -> Orientation:
        """
        Determine the dominant orientation of a straight well (even if slanted)
        by estimating which axis the well is most aligned with.

        :returns: The dominant orientation of the well
        """
        start, end = self.perforating_interval

        # Convert to numpy arrays and pad to 3D if needed
        start = np.array(start + (0,) * (3 - len(start)))
        end = np.array(end + (0,) * (3 - len(end)))

        # Compute the direction vector
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Start and end points are the same.")

        # Normalize and take absolute value
        unit_vector = np.abs(direction / norm)
        axis = np.argmax(unit_vector)
        return Orientation(("x", "y", "z")[axis])

    def update_schedule(self, event: "ScheduledEvent") -> None:
        """
        Update the well schedule

        :param event:
        """
        self.schedule[event.time_step] = event

    def evolve(self, time_step: int) -> None:
        """
        Evolve the well state to the next time step.

        :param time_step: The current time step in the simulation.
        """
        if time_step in self.schedule:
            event = self.schedule[time_step]
            event.apply(self)
        return None


WellT = typing.TypeVar("WellT", bound=_Well)


@define(slots=True)
class ScheduledEvent(typing.Generic[WellT]):
    """
    Represents a scheduled event for a well at a specific time step.

    This event can include changes to the well's bottom-hole pressure, skin factor,
    and whether the well is active or not.
    The event is applied to the well at the specified time step.
    """

    time_step: int
    """The time step at which this schedule is applicable."""
    bottom_hole_pressure: typing.Optional[float] = None
    """Bottom-hole pressure for the well at this time step."""
    skin_factor: typing.Optional[float] = None
    """Skin factor for the well at this time step."""
    is_active: typing.Optional[bool] = None
    """Indicates whether the well is active at this time step."""

    def apply(self, well: WellT) -> WellT:
        """
        Apply this schedule to a well.

        :param well: The well to which this schedule will be applied.
        """
        if self.bottom_hole_pressure is not None:
            well.bottom_hole_pressure = self.bottom_hole_pressure
        if self.skin_factor is not None:
            well.skin_factor = self.skin_factor
        if self.is_active is not None:
            well.is_active = self.is_active
        return well


@define(slots=True)
class InjectionWell(_Well[WellLocation]):
    """
    Models an injection well in the reservoir model.

    This well injects fluids into the reservoir.
    """

    injected_fluid: typing.Optional[InjectedFluid] = None
    """Properties of the fluid being injected into the well."""


@define(slots=True)
class ProductionWell(_Well[WellLocation]):
    """
    Models a production well in the reservoir model.

    This well produces fluids from the reservoir.
    """

    produced_fluids: typing.Sequence[ProducedFluid] = field(factory=list)
    """List of fluids produced by the well. This can include multiple phases (e.g., oil, gas, water)."""


InjectionWellT = typing.TypeVar("InjectionWellT", bound=InjectionWell)
ProductionWellT = typing.TypeVar("ProductionWellT", bound=ProductionWell)


@define(slots=True)
class InjectionWellScheduledEvent(ScheduledEvent[InjectionWellT]):
    """
    Scheduled event for an injection well.

    This includes the well's perforating interval, bottom-hole pressure, skin factor, and injected fluid.
    """

    injected_fluid: typing.Optional[InjectedFluid] = field(
        default=None, metadata={"description": "Injected fluid properties at this time step."}
    )
    """Injected fluid properties at this time step."""

    def __attrs_post_init__(self) -> None:
        """Ensure the injected fluid is properly initialized."""
        self.injected_fluid = (
            copy.deepcopy(self.injected_fluid) if self.injected_fluid else None
        )

    def apply(self, well: InjectionWellT) -> InjectionWellT:
        """Apply this schedule to an injection well."""
        well = super().apply(well)
        if self.injected_fluid is not None:
            well.injected_fluid = self.injected_fluid
        return well


@define(slots=True)
class ProductionWellScheduledEvent(ScheduledEvent[ProductionWellT]):
    """
    Scheduled event for a production well.

    This includes the well's perforating interval, bottom-hole pressure, skin factor, and produced fluids.
    """

    produced_fluids: typing.Optional[typing.Sequence[ProducedFluid]] = field(
        default=None, metadata={"description": "List of produced fluids at this time step."}
    )
    """Produced fluids properties at this time step."""

    def __attrs_post_init__(self) -> None:
        """Ensure the produced fluids are properly initialized."""
        self.produced_fluids = (
            copy.deepcopy(self.produced_fluids) if self.produced_fluids else None
        )

    def apply(self, well: ProductionWellT) -> ProductionWellT:
        """Apply this schedule to a production well."""
        well = super().apply(well)
        if self.produced_fluids is not None:
            well.produced_fluids = self.produced_fluids
        return well


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


def _prepare_wells_map(
    wells: typing.Sequence[WellT],
) -> typing.Dict[typing.Tuple[int, ...], WellT]:
    """Prepare the wells map for quick access."""
    wells_map = {
        loc: well
        for well in wells
        for loc in _expand_interval(
            interval=well.perforating_interval,
            orientation=well.orientation,
        )
    }
    return wells_map


@define(slots=True, frozen=True)
class WellsProxy(typing.Generic[WellLocation, WellT]):
    """A proxy class for quick access to wells by their location."""

    wells: typing.Sequence[WellT]
    """A map of well perforating intervals to the well objects."""

    wells_map: typing.Dict[WellLocation, WellT] = field(init=False)
    """A map to store wells by their location for quick access."""

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "wells_map", _prepare_wells_map(self.wells))

    def __getitem__(self, location: WellLocation) -> typing.Optional[WellT]:
        """Get a well by its location."""
        return self.wells_map.get(location, None)

    def __setitem__(self, location: WellLocation, well: WellT) -> None:
        """Set a well at a specific location."""
        self.wells_map[location] = well


@define(slots=True)
class Wells(typing.Generic[WellLocation]):
    """
    Models a collection of injection and production wells in the reservoir model.

    This includes both production and injection wells.
    """

    injection_wells: typing.Sequence[InjectionWell[WellLocation]] = field(factory=list)
    """List of injection wells in the reservoir."""
    production_wells: typing.Sequence[ProductionWell[WellLocation]] = field(
        factory=list
    )
    """List of production wells in the reservoir."""
    injectors: WellsProxy[WellLocation, InjectionWell[WellLocation]] = field(init=False)
    """
    Proxy for injection wells.

    This allows quick access to injection wells by their location.
    """
    producers: WellsProxy[WellLocation, ProductionWell[WellLocation]] = field(
        init=False
    )
    """
    Proxy for production wells.

    This allows quick access to production wells by their location.
    """

    def __attrs_post_init__(self) -> None:
        self.injectors = WellsProxy(self.injection_wells)
        self.producers = WellsProxy(self.production_wells)

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

    def evolve(self, time_step: int) -> None:
        """
        Evolve all wells in the reservoir model to the next time step.

        This method updates the state of each well based on its schedule.
        :param time_step: The current time step in the simulation.
        """
        for well in itertools.chain(self.injection_wells, self.production_wells):
            well.evolve(time_step)


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
    :return: The well index (md*ft).
    """
    well_index = (permeability * interval_thickness) / (
        np.log(effective_drainage_radius / wellbore_radius) + skin_factor
    )
    return well_index


def compute_3D_effective_drainage_radius(
    interval_thickness: typing.Tuple[float, float, float],
    permeability: typing.Tuple[float, float, float],
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
    interval_thickness: typing.Tuple[float, float],
    permeability: typing.Tuple[float, float],
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


def compute_well_rate(
    well_index: float,
    pressure: float,
    bottom_hole_pressure: float,
    phase_mobility: float = 1.0,
    use_pressure_squared: bool = False,
) -> float:
    """
    Compute the well rate using the well index and pressure drop.

    The formula for the well rate is:

        Q = 6.33e-3 * W * (P - P_bhp) * M

    Or for gas wells:
        Q = 6.33e-3 * W * (P² - P_bhp²) * M

    where:
        - Q is the well rate (STB/day) or (SCF/day)
        - W is the well index (STB/day/psi) or (SCF/day/psi)
        - P is the reservoir pressure (psi)
        - P_bhp is the bottom-hole pressure (psi)
        - M is the phase mobility (dimensionless, default is 1.0) (k_r / μ) (psi⁻¹)

    :param well_index: The well index (STB/day/psi) or (SCF/day/psi).
    :param pressure: The reservoir pressure (psi).
    :param bottom_hole_pressure: The bottom-hole pressure (psi).
    :param phase_mobility: The phase mobility (dimensionless, default is 1.0) (k_r / μ) (psi⁻¹).
    :param use_pressure_squared: If True, use the squared pressure difference for gas wells.
        This is typically used for gas wells to account for the non-linear relationship.
    :return: The well rate (STB/day) or (SCF/day).
    """
    if well_index <= 0:
        raise ValueError("Well index must be a positive value.")
    if pressure <= bottom_hole_pressure:
        raise ValueError(
            "Reservoir pressure must be greater than bottom-hole pressure."
        )

    if use_pressure_squared:
        pressure_difference = pressure**2 - bottom_hole_pressure**2
    else:
        pressure_difference = pressure - bottom_hole_pressure
    well_rate = 6.33e-3 * well_index * pressure_difference * phase_mobility
    return well_rate
