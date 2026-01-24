import attrs

from bores.capillary_pressures import CapillaryPressureTable
from bores.relperm import RelativePermeabilityTable
from bores.serialization import Serializable


__all__ = ["RockFluidTables"]


@attrs.frozen
class RockFluidTables(Serializable):
    """
    Tables defining rock-fluid interactions in the reservoir.
    """

    relative_permeability_table: RelativePermeabilityTable
    """Callable that evaluates the relative permeability curves based on fluid saturations."""
    capillary_pressure_table: CapillaryPressureTable
    """Callable that evaluates the capillary pressure curves based on fluid saturations."""
