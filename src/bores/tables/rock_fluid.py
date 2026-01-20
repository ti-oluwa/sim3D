import typing

import attrs

from bores.types import CapillaryPressures, RelativePermeabilities


__all__ = ["RelativePermeabilityTable", "CapillaryPressureTable", "RockFluidTables"]


@typing.runtime_checkable
class RelativePermeabilityTable(typing.Protocol):
    """
    Protocol for a relative permeability table that computes
    relative permeabilities based on fluid saturations.
    """

    def __call__(self, *args, **kwargs: typing.Any) -> RelativePermeabilities:
        """
        Computes relative permeabilities based on fluid saturations.

        :param kwargs: Additional parameters for the relative permeability function.
        :return: A dictionary containing relative permeabilities for water, oil, and gas phases.
        """
        ...


@typing.runtime_checkable
class CapillaryPressureTable(typing.Protocol):
    """
    Protocol for a capillary pressure table that computes
    capillary pressures based on fluid saturations.
    """

    def __call__(self, *args, **kwargs: typing.Any) -> CapillaryPressures:
        """
        Computes capillary pressures based on fluid saturations.

        :param kwargs: Saturation parameters (water_saturation, oil_saturation, gas_saturation).
        :return: A dictionary containing capillary pressures for oil-water and gas-oil systems.
        """
        ...


@attrs.frozen()
class RockFluidTables:
    """
    Tables defining rock-fluid interactions in the reservoir.
    """

    relative_permeability_table: RelativePermeabilityTable
    """Callable that evaluates the relative permeability curves based on fluid saturations."""
    capillary_pressure_table: CapillaryPressureTable
    """Callable that evaluates the capillary pressure curves based on fluid saturations."""
