import typing
import enum
from dataclasses import dataclass, field
import functools


__all__ = [
    "FluidPhase",
    "InjectedFluid",
    "ProducedFluid",
    "InjectionWell",
    "ProductionWell",
    "Wells",
]


class FluidPhase(enum.Enum):
    """Enum representing the phase of the fluid in the reservoir."""

    WATER = "water"
    GAS = "gas"
    OIL = "oil"


@dataclass(slots=True, frozen=True)
class _WellFluid:
    """Properties of the fluid being injected into or produced by a well."""

    name: str
    """Name of the injected fluid. Examples: Methane, CO2, Water, Oil."""
    phase: FluidPhase
    """Phase of the injected fluid. Examples: WATER, GAS, OIL."""
    volumetric_flow_rate: float
    """Volumetric flow rate of the well in (STB/day) or (SCF/day), depending of the phase of the fluid injected."""

    def __post_init__(self) -> None:
        """Ensure the produced fluids list is not empty."""
        if abs(self.volumetric_flow_rate) != self.volumetric_flow_rate:
            raise ValueError("Fluid flow rate must be a positive value.")


@dataclass(slots=True, frozen=True)
class InjectedFluid(_WellFluid):
    """
    Represents a fluid being injected into the reservoir.
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


@dataclass(slots=True, frozen=True)
class ProducedFluid(_WellFluid):
    """
    Represents a fluid being produced from the reservoir.
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


@dataclass(slots=True, frozen=True)
class _Well:
    """Represents a well in the reservoir model."""

    name: typing.Optional[str]
    """Name of the well."""
    location: typing.Tuple[int, int]
    """Location of the well in the reservoir grid (x, y) coordinates."""
    radius: float
    """Radius of the well (ft). This is used to calculate the wellbore storage and skin effects."""


@dataclass(slots=True, frozen=True)
class InjectionWell(_Well):
    """
    Represents an injection well in the reservoir model.

    This well injects fluids into the reservoir.
    """

    injected_fluid: InjectedFluid
    """Properties of the fluid being injected into the well."""


@dataclass(slots=True, frozen=True)
class ProductionWell(_Well):
    """
    Represents a production well in the reservoir model.

    This well produces fluids from the reservoir.
    """

    skin_factor: float = 0.0
    """Skin factor for the well, affecting flow performance."""
    produced_fluids: typing.Sequence[ProducedFluid] = field(default_factory=list)
    """List of fluids produced by the well. This can include multiple phases (e.g., oil, gas, water)."""


WellT = typing.TypeVar("WellT", bound=_Well)


@dataclass(slots=True, frozen=True)
class WellsProxy(typing.Generic[WellT]):
    """A proxy class for quick access to wells by their location."""

    wells_map: typing.Dict[typing.Tuple[int, int], WellT]
    """A map to store wells by their location for quick access."""

    def __getitem__(self, location: typing.Tuple[int, int]) -> typing.Optional[WellT]:
        """Get a well by its location."""
        return self.wells_map.get(location, None)

    def __setitem__(self, location: typing.Tuple[int, int], well: WellT) -> None:
        """Set a well at a specific location."""
        self.wells_map[location] = well


@dataclass(frozen=True)
class Wells:
    """
    Represents a collection of wells in the reservoir model.

    This includes both production and injection wells.
    """

    injection_wells: typing.List[InjectionWell] = field(default_factory=list)
    """List of injection wells in the reservoir."""
    production_wells: typing.List[ProductionWell] = field(default_factory=list)
    """List of production wells in the reservoir."""
    _injectors_map: typing.Dict[typing.Tuple[int, int], InjectionWell] = field(
        init=False, repr=False
    )
    """Map of injection well locations to their instances for quick access."""
    _producers_map: typing.Dict[typing.Tuple[int, int], ProductionWell] = field(
        init=False, repr=False
    )
    """Map of production well locations to their instances for quick access."""

    def __post_init__(self) -> None:
        """Initialize the well maps for quick access."""
        object.__setattr__(
            self,
            "_injectors_map",
            {well.location: well for well in self.injection_wells},
        )
        object.__setattr__(
            self,
            "_producers_map",
            {well.location: well for well in self.production_wells},
        )

    @functools.cached_property
    def injectors(self) -> WellsProxy[InjectionWell]:
        """
        Get a proxy for injection wells.

        This allows quick access to injection wells by their location.
        """
        return WellsProxy(self._injectors_map)

    @functools.cached_property
    def producers(self) -> WellsProxy[ProductionWell]:
        """
        Get a proxy for production wells.

        This allows quick access to production wells by their location.
        """
        return WellsProxy(self._producers_map)

    def __getitem__(
        self, location: typing.Tuple[int, int]
    ) -> typing.Tuple[typing.Optional[InjectionWell], typing.Optional[ProductionWell]]:
        """
        Get a well by its grid coordinates.

        :param location: The (i, j) coordinates of the well in the reservoir grid.
        :return: Well or None: The well at the specified location, or None if not found.
        """
        return self.injectors[location], self.producers[location]
