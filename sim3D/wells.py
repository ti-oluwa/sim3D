import copy
import itertools
import typing
import enum
from attrs import define, field
import numba
import numpy as np
from scipy.integrate import quad

from sim3D.types import Orientation, WellLocation
from sim3D.properties import (
    compute_gas_density,
    compute_gas_viscosity,
    compute_gas_compressibility_factor,
)


__all__ = [
    "FluidPhase",
    "WellFluid",
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
class WellFluid:
    """Properties of the fluid being injected into or produced by a well."""

    name: str
    """Name of the fluid. Examples: Methane, CO2, Water, Oil."""
    phase: FluidPhase
    """Phase of the fluid. Examples: WATER, GAS, OIL."""
    specific_gravity: float
    """Specific gravity of the fluid in (lbm/ft³)."""
    molecular_weight: float
    """Molecular weight of the fluid in (g/mol)."""
    compressibility: typing.Optional[float] = None
    """Compressibility of the fluid in (psi⁻¹)."""
    formation_volume_factor: typing.Optional[float] = None
    """Formation volume factor of the fluid in (bbl/STB) or (ft³/SCF), depending on the phase of the fluid."""
    salinity: typing.Optional[float] = None
    """Salinity of the fluid in (ppm NaCl)."""


@define(slots=True)
class Well(typing.Generic[WellLocation]):
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

    def check_location(self, grid_dimensions: typing.Tuple[int, ...]) -> None:
        """
        Check if the well's perforating interval is within the grid dimensions.

        :param grid_dimensions: The dimensions of the reservoir grid (i, j, k).
        :raises ValueError: If the well's perforating interval is out of bounds.
        """
        start, end = self.perforating_interval
        if not all(0 <= coord < dim for coord, dim in zip(start, grid_dimensions)):
            raise ValueError(
                f"Start location {start} for well {self.name!r} is out of bounds."
            )
        if not all(0 <= coord < dim for coord, dim in zip(end, grid_dimensions)):
            raise ValueError(
                f"End location {end} for well {self.name!r} is out of bounds."
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
            interval_thickness = typing.cast(
                typing.Tuple[float, float], interval_thickness
            )
            return compute_2D_effective_drainage_radius(
                interval_thickness=interval_thickness,
                permeability=permeability,
                well_orientation=self.orientation,
            )

        if len(permeability) != 3:
            raise ValueError("Permeability must be a 3D tuple for 3D locations")
        interval_thickness = typing.cast(
            typing.Tuple[float, float, float], interval_thickness
        )
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
        if orientation == Orientation.X:
            directional_permeability = permeability[0]
            directional_thickness = interval_thickness[0]

        elif orientation == Orientation.Y:
            directional_permeability = permeability[1]
            directional_thickness = interval_thickness[1]

        elif dimensions == 3 and orientation == Orientation.Z:
            directional_permeability = permeability[2]
            directional_thickness = interval_thickness[2]

        else:  # dimensions == 2 and orientation == Orientation.Z:
            raise ValueError("Z-oriented wells are not supported in 2D models")
        return compute_well_index(
            permeability=directional_permeability,
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
        fluid: typing.Optional[WellFluid] = None,
        use_pseudo_pressure: typing.Optional[bool] = None,
        z_factor_func: typing.Optional[typing.Callable[[float], float]] = None,
        viscosity_func: typing.Optional[typing.Callable[[float], float]] = None,
    ) -> float:
        """
        Compute the flow rate for the well using Darcy's law.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The mobility of the fluid phase being produced or injected (ft²/(psi·day)).
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being produced or injected. If None, uses the well's injected_fluid.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param z_factor_func: Function to compute the gas compressibility factor (dimensionless) at a given pressure. Required if use_pseudo_pressure is True.
        :param viscosity_func: Function to compute the gas viscosity (cP) at a given pressure. Required if use_pseudo_pressure is True.
        :return: The flow rate (STB/day or SCF/day).
        """
        # If no fluid is injected, return 0 flow rate
        if fluid is None:
            return 0.0
        use_pseudo_pressure = (
            use_pseudo_pressure
            if use_pseudo_pressure is not None
            else fluid.phase == FluidPhase.GAS
        )

        if use_pseudo_pressure and z_factor_func is None and viscosity_func is None:
            specific_gravity = fluid.specific_gravity

            def _z_factor_func(pressure: float) -> float:
                nonlocal specific_gravity, temperature

                return compute_gas_compressibility_factor(
                    pressure=pressure,
                    temperature=temperature,
                    gas_gravity=specific_gravity,
                )

            def _viscosity_func(pressure: float) -> float:
                nonlocal specific_gravity, temperature

                z_factor = compute_gas_compressibility_factor(
                    pressure=pressure,
                    temperature=temperature,
                    gas_gravity=specific_gravity,
                )
                density = compute_gas_density(
                    pressure=pressure,
                    temperature=temperature,
                    gas_gravity=specific_gravity,
                    gas_compressibility_factor=z_factor,
                )
                return compute_gas_viscosity(
                    temperature=temperature,
                    gas_density=density,
                    gas_molecular_weight=fluid.molecular_weight,
                )

            z_factor_func = _z_factor_func
            viscosity_func = _viscosity_func

        return compute_well_rate(
            well_index=well_index,
            pressure=pressure,
            bottom_hole_pressure=self.bottom_hole_pressure,
            phase_mobility=phase_mobility,
            use_pseudo_pressure=use_pseudo_pressure,
            z_factor_func=z_factor_func,
            viscosity_func=viscosity_func,
        )


WellT = typing.TypeVar("WellT", bound=Well)


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
class InjectionWell(Well[WellLocation]):
    """
    Models an injection well in the reservoir model.

    This well injects fluids into the reservoir.
    """

    injected_fluid: typing.Optional[WellFluid] = None
    """Properties of the fluid being injected into the well."""

    def get_flow_rate(
        self,
        pressure: float,
        temperature: float,
        phase_mobility: float,
        well_index: float,
        fluid: typing.Optional[WellFluid] = None,
        use_pseudo_pressure: typing.Optional[bool] = None,
        z_factor_func: typing.Optional[typing.Callable[[float], float]] = None,
        viscosity_func: typing.Optional[typing.Callable[[float], float]] = None,
    ) -> float:
        """
        Compute the flow rate for the injection well using Darcy's law.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobility: The mobility of the fluid phase being produced or injected (ft²/(psi·day)).
        :param well_index: The well index (md*ft).
        :param fluid: The fluid being injected into the well. If None, uses the well's injected_fluid property.
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param z_factor_func: Function to compute the gas compressibility factor (dimensionless) at a given pressure. Required if use_pseudo_pressure is True.
        :param viscosity_func: Function to compute the gas viscosity (cP) at a given pressure. Required if use_pseudo_pressure is True.
        :return: The flow rate (STB/day or SCF/day).
        """
        return super().get_flow_rate(
            pressure=pressure,
            temperature=temperature,
            phase_mobility=phase_mobility,
            well_index=well_index,
            fluid=fluid or self.injected_fluid,
            use_pseudo_pressure=use_pseudo_pressure,
            z_factor_func=z_factor_func,
            viscosity_func=viscosity_func,
        )


@define(slots=True)
class ProductionWell(Well[WellLocation]):
    """
    Models a production well in the reservoir model.

    This well produces fluids from the reservoir.
    """

    produced_fluids: typing.Sequence[WellFluid] = field(factory=list)
    """List of fluids produced by the well. This can include multiple phases (e.g., oil, gas, water)."""

    def get_flow_rates(
        self,
        pressure: float,
        temperature: float,
        phase_mobilities: typing.Sequence[float],
        well_index: float,
        use_pseudo_pressure: typing.Optional[bool] = None,
        z_factor_func: typing.Optional[typing.Callable[[float], float]] = None,
        viscosity_func: typing.Optional[typing.Callable[[float], float]] = None,
    ) -> typing.Iterator[float]:
        """
        Compute the flow rates for the produced fluids using Darcy's law.

        :param pressure: The reservoir pressure at the well location (psi).
        :param temperature: The reservoir temperature at the well location (°F).
        :param phase_mobilities: The mobilities of the fluid phases being produced (ft²/(psi·day)).
            This should be a sequence with the same length as produced_fluids.
        :param well_index: The well index (md*ft).
        :param use_pseudo_pressure: Whether to use pseudo-pressure for gas wells (default is False).
        :param z_factor_func: Function to compute the gas compressibility factor (dimensionless) at a given pressure. Required if use_pseudo_pressure is True.
        :param viscosity_func: Function to compute the gas viscosity (cP) at a given pressure. Required if use_pseudo_pressure is True.
        :return: A list of flow rates (STB/day or SCF/day) for each produced fluid.
        """
        return (
            self.get_flow_rate(
                pressure=pressure,
                temperature=temperature,
                phase_mobility=phase_mobility,
                well_index=well_index,
                fluid=fluid,
                use_pseudo_pressure=use_pseudo_pressure,
                z_factor_func=z_factor_func,
                viscosity_func=viscosity_func,
            )
            for fluid, phase_mobility in zip(self.produced_fluids, phase_mobilities)
        )


InjectionWellT = typing.TypeVar("InjectionWellT", bound=InjectionWell)
ProductionWellT = typing.TypeVar("ProductionWellT", bound=ProductionWell)


@define(slots=True)
class InjectionWellScheduledEvent(ScheduledEvent[InjectionWellT]):
    """
    Scheduled event for an injection well.

    This includes the well's perforating interval, bottom-hole pressure, skin factor, and injected fluid.
    """

    injected_fluid: typing.Optional[WellFluid] = field(
        default=None,
        metadata={"description": "Injected fluid properties at this time step."},
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

    produced_fluids: typing.Optional[typing.Sequence[WellFluid]] = field(
        default=None,
        metadata={"description": "List of produced fluids at this time step."},
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
            len(_expand_interval(well.perforating_interval, well.orientation))
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
        Get all well locations in the reservoir.

        :return: A tuple of (injection_well_locations, production_well_locations).
        This returns a tuple containing two lists:
            - A list of locations for injection wells.
            - A list of locations for production wells.
        """
        return (
            list(self.injectors.wells_map.keys()),
            list(self.producers.wells_map.keys()),
        )

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

    def evolve(self, time_step: int) -> None:
        """
        Evolve all wells in the reservoir model to the next time step.

        This method updates the state of each well based on its schedule.
        :param time_step: The current time step in the simulation.
        """
        for well in itertools.chain(self.injection_wells, self.production_wells):
            well.evolve(time_step)

    def check_location(self, grid_dimensions: typing.Tuple[int, ...]) -> None:
        """
        Check if all wells' perforating intervals are within the grid dimensions.

        :param grid_dimensions: The dimensions of the reservoir grid (i, j, k).
        :raises ValueError: If any well's perforating interval is out of bounds.
        """
        for well in itertools.chain(self.injection_wells, self.production_wells):
            well.check_location(grid_dimensions)


@numba.njit(cache=True)
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


@numba.njit(cache=True)
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


def compute_gas_pseudo_pressure(
    pressure: float,
    z_factor_func: typing.Callable[[float], float],
    viscosity_func: typing.Callable[[float], float],
) -> float:
    """
    Compute the gas pseudo-pressure.

    The formula for the gas pseudo-pressure is:

        m(p) = ∫(2 * p / z(p)) dp

    where:
        - m(p) is the gas pseudo-pressure (psi²)
        - p is the pressure (psi)
        - z(p) is the gas compressibility factor (dimensionless)

    :param pressure: The pressure (psi).
    :param compressibility_factor: The gas compressibility factor (dimensionless) at the given pressure.
    :return: The gas pseudo-pressure (psi²).
    """
    if pressure <= 0:
        return 0.0

    def integrand(pressure: float) -> float:
        return 2 * pressure / (viscosity_func(pressure) * z_factor_func(pressure))

    result, _ = quad(integrand, 0, pressure)
    return typing.cast(float, result)


def compute_well_rate(
    well_index: float,
    pressure: float,
    bottom_hole_pressure: float,
    phase_mobility: float = 1.0,
    use_pseudo_pressure: bool = False,
    z_factor_func: typing.Optional[typing.Callable[[float], float]] = None,
    viscosity_func: typing.Optional[typing.Callable[[float], float]] = None,
) -> float:
    """
    Compute the well rate using the well index and pressure drop.

    The formula for the well rate is:

        Q = 6.33e-3 * W * (P_bhp - P) * M

    Or for gas wells:
        Q = 6.33e-3 * W * (m(P_bhp) - m(P)) * M

    where:
        - Q is the well rate (STB/day) or (SCF/day)
        - W is the well index (STB/day/psi) or (SCF/day/psi)
        - P is the reservoir pressure (psi)
        - P_bhp is the bottom-hole pressure (psi)
        - M is the phase mobility (dimensionless, default is 1.0) (k_r / μ) (psi⁻¹)

    Negative rate result indicates that the well is producing, while positive rates indicate injection.

    :param well_index: The well index (STB/day/psi) or (SCF/day/psi).
    :param pressure: The reservoir pressure (psi).
    :param bottom_hole_pressure: The bottom-hole pressure (psi).
    :param phase_mobility: The phase mobility (dimensionless, default is 1.0) (k_r / μ) (psi⁻¹).
    :param use_pseudo_pressure: If True, use the real gas pseudo-pressure difference instead of simple pressure difference.
        This is typically used for gas wells to account for the non-linear relationship.
    :param z_factor_func: A callable function that returns the gas compressibility factor (z) at a given pressure.
        Only required if `use_pseudo_pressure` is True.
    :param viscosity_func: A callable function that returns the gas viscosity (μ) at a given pressure.
        Only required if `use_pseudo_pressure` is True.
    :return: The well rate (STB/day) or (SCF/day).
    """
    if well_index <= 0:
        raise ValueError("Well index must be a positive value.")

    if use_pseudo_pressure:
        if z_factor_func is None or viscosity_func is None:
            raise ValueError(
                "z_factor_func and viscosity_func must be provided when use_pseudo_pressure is True."
            )

        bottom_hole_pseudo_pressure = compute_gas_pseudo_pressure(
            bottom_hole_pressure, z_factor_func, viscosity_func
        )
        reservoir_pseudo_pressure = compute_gas_pseudo_pressure(
            pressure, z_factor_func, viscosity_func
        )
        pressure_difference = bottom_hole_pseudo_pressure - reservoir_pseudo_pressure
    else:
        pressure_difference = bottom_hole_pressure - pressure
    well_rate = 6.33e-3 * well_index * pressure_difference * phase_mobility
    return well_rate
