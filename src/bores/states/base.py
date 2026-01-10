"""Model state management."""

import functools
import logging
import typing

import attrs
import numpy as np

from bores.errors import ValidationError
from bores.grids.base import (
    CapillaryPressureGrids,
    RateGrids,
    RelPermGrids,
    RelativeMobilityGrids,
)
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockPermeability,
    RockProperties,
    SaturationHistory,
)
from bores.timing import TimerState
from bores.types import NDimension, T
from bores.utils import Lazy, LazyField
from bores.wells.base import Wells

logger = logging.getLogger(__name__)


__all__ = ["ModelState", "validate_state"]

_Lazy = typing.Union[Lazy[T], T, typing.Callable[[], T]]


class ModelState(typing.Generic[NDimension]):
    """
    The state of the reservoir model at a specific time step during a simulation.
    """

    model = LazyField[ReservoirModel[NDimension]]()
    wells = LazyField[Wells[NDimension]]()
    injection = LazyField[RateGrids[NDimension]]()
    production = LazyField[RateGrids[NDimension]]()
    relative_permeabilities = LazyField[RelPermGrids[NDimension]]()
    relative_mobilities = LazyField[RelativeMobilityGrids[NDimension]]()
    capillary_pressures = LazyField[CapillaryPressureGrids[NDimension]]()

    def __init__(
        self,
        step: int,
        step_size: float,
        time: float,
        model: _Lazy[ReservoirModel[NDimension]],
        wells: _Lazy[Wells[NDimension]],
        injection: _Lazy[RateGrids[NDimension]],
        production: _Lazy[RateGrids[NDimension]],
        relative_permeabilities: _Lazy[RelPermGrids[NDimension]],
        relative_mobilities: _Lazy[RelativeMobilityGrids[NDimension]],
        capillary_pressures: _Lazy[CapillaryPressureGrids[NDimension]],
        timer_state: typing.Optional[TimerState] = None,
    ) -> None:
        """
        Initialize the model state.

        :param step: The time step index
        :param step_size: The time step size in seconds
        :param time: The simulation time in seconds
        :param model: The reservoir model at this state
        :param wells: The wells configuration at this state
        :param injection: Fluids injection rates at this state in ft³/day
        :param production: Fluids production rates at this state in ft³/day
        :param relative_permeabilities: Relative permeabilities at this state
        :param relative_mobilities: Relative mobilities at this state
        :param capillary_pressures: Capillary pressures at this state
        :param timer_state: Optional timer state at this model state
        """
        self.step = step
        self.step_size = step_size
        self.time = time
        self.model = typing.cast(ReservoirModel[NDimension], model)
        self.wells = typing.cast(Wells[NDimension], wells)
        self.injection = typing.cast(RateGrids[NDimension], injection)
        self.production = typing.cast(RateGrids[NDimension], production)
        self.relative_permeabilities = typing.cast(
            RelPermGrids[NDimension], relative_permeabilities
        )
        self.relative_mobilities = typing.cast(
            RelativeMobilityGrids[NDimension], relative_mobilities
        )
        self.capillary_pressures = typing.cast(
            CapillaryPressureGrids[NDimension], capillary_pressures
        )
        self.timer_state = timer_state

    def asdict(self) -> typing.Dict[str, typing.Any]:
        """
        Get a dictionary representation of the model state.
        """
        return {
            "step": self.step,
            "step_size": self.step_size,
            "time": self.time,
            "model": self.model,
            "wells": self.wells,
            "injection": self.injection,
            "production": self.production,
            "relative_permeabilities": self.relative_permeabilities,
            "relative_mobilities": self.relative_mobilities,
            "capillary_pressures": self.capillary_pressures,
            "timer_state": self.timer_state,
        }

    @functools.cache
    def wells_exists(self) -> bool:
        """Check if there are any wells in this state."""
        return self.wells.exists()


def _validate_and_coerce_array(
    model_shape: tuple[int, ...],
    grid: np.ndarray,
    field_name: str,
    dtype_target: typing.Optional[np.typing.DTypeLike],
) -> np.ndarray:
    if grid.shape != model_shape:
        raise ValidationError(
            f"{field_name} has shape {grid.shape}, expected {model_shape}."
        )
    if dtype_target is not None and np.issubdtype(grid.dtype, np.floating):
        return grid.astype(dtype_target, copy=False)
    return grid


def validate_state(
    state: ModelState[NDimension], dtype: typing.Optional[np.typing.DTypeLike] = None
) -> ModelState[NDimension]:
    """
    Validate state grids have matching shapes and optionally coerce to specified dtype.

    :param state: `ModelState` to validate
    :param dtype: Optional dtype to coerce all array fields to. If None, no coercion is performed.
    :return: Validated (and optionally coerced) `ModelState`
    """
    # Check that all grids have matching shapes
    model = state.model
    model_shape = model.grid_shape
    fluid_properties = model.fluid_properties
    rock_properties = model.rock_properties
    injection = state.injection
    production = state.production
    relative_mobilities = state.relative_mobilities
    relative_permeabilities = state.relative_permeabilities
    capillary_pressures = state.capillary_pressures
    thickness_grid = model.thickness_grid
    if thickness_grid.shape != model_shape:
        raise ValidationError(
            f"Thickness grid has shape {thickness_grid.shape}, expected {model_shape}."
        )

    # Validate and coerce fluid properties
    if dtype is not None:
        fluid_dict = {}
        for field in attrs.fields(fluid_properties.__class__):
            value = getattr(fluid_properties, field.name)
            if isinstance(value, np.ndarray):
                fluid_dict[field.name] = _validate_and_coerce_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Fluid property grid {field.name}",
                    dtype_target=dtype,
                )
            else:
                fluid_dict[field.name] = value
        fluid_properties = FluidProperties(**fluid_dict)
    else:
        for field in attrs.fields(fluid_properties.__class__):
            grid = getattr(fluid_properties, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Fluid property grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce rock properties
    if dtype is not None:
        rock_dict = {}
        for field in attrs.fields(rock_properties.__class__):
            value = getattr(rock_properties, field.name)
            if isinstance(value, np.ndarray):
                rock_dict[field.name] = _validate_and_coerce_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Rock property grid {field.name}",
                    dtype_target=dtype,
                )
            elif field.name == "absolute_permeability":
                perm = value
                rock_dict[field.name] = RockPermeability(
                    x=_validate_and_coerce_array(
                        model_shape=model_shape,
                        grid=perm.x,
                        field_name="Rock permeability x",
                        dtype_target=dtype,
                    ),
                    y=_validate_and_coerce_array(
                        model_shape=model_shape,
                        grid=perm.y,
                        field_name="Rock permeability y",
                        dtype_target=dtype,
                    ),
                    z=_validate_and_coerce_array(
                        model_shape=model_shape,
                        grid=perm.z,
                        field_name="Rock permeability z",
                        dtype_target=dtype,
                    ),
                )
            else:
                rock_dict[field.name] = value
        rock_properties = RockProperties(**rock_dict)
    else:
        for field in attrs.fields(rock_properties.__class__):
            grid = getattr(rock_properties, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Rock property grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce injection
    if dtype is not None:
        injection_dict = {}
        for field in attrs.fields(injection.__class__):
            value = getattr(injection, field.name)
            if isinstance(value, np.ndarray):
                injection_dict[field.name] = _validate_and_coerce_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Injection rate grid {field.name}",
                    dtype_target=dtype,
                )
            else:
                injection_dict[field.name] = value
        injection = RateGrids(**injection_dict)
    else:
        for field in attrs.fields(injection.__class__):
            grid = getattr(injection, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Injection rate grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce production
    if dtype is not None:
        production_dict = {}
        for field in attrs.fields(production.__class__):
            value = getattr(production, field.name)
            if isinstance(value, np.ndarray):
                production_dict[field.name] = _validate_and_coerce_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Production rate grid {field.name}",
                    dtype_target=dtype,
                )
            else:
                production_dict[field.name] = value
        production = RateGrids(**production_dict)
    else:
        for field in attrs.fields(production.__class__):
            grid = getattr(production, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Production rate grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce relative mobilities
    if dtype is not None:
        mobility_dict = {}
        for field in attrs.fields(relative_mobilities.__class__):
            value = getattr(relative_mobilities, field.name)
            if isinstance(value, np.ndarray):
                mobility_dict[field.name] = _validate_and_coerce_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Relative mobility grid {field.name}",
                    dtype_target=dtype,
                )
            else:
                mobility_dict[field.name] = value
        relative_mobilities = RelativeMobilityGrids(**mobility_dict)
    else:
        for field in attrs.fields(relative_mobilities.__class__):
            grid = getattr(relative_mobilities, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Relative mobility grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce relative permeabilities
    if dtype is not None:
        relperm_dict = {}
        for field in attrs.fields(relative_permeabilities.__class__):
            value = getattr(relative_permeabilities, field.name)
            if isinstance(value, np.ndarray):
                relperm_dict[field.name] = _validate_and_coerce_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Relative permeability grid {field.name}",
                    dtype_target=dtype,
                )
            else:
                relperm_dict[field.name] = value
        relative_permeabilities = RelPermGrids(**relperm_dict)
    else:
        for field in attrs.fields(relative_permeabilities.__class__):
            grid = getattr(relative_permeabilities, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Relative permeability grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce capillary pressures
    if dtype is not None:
        capillary_dict = {}
        for field in attrs.fields(capillary_pressures.__class__):
            value = getattr(capillary_pressures, field.name)
            if isinstance(value, np.ndarray):
                capillary_dict[field.name] = _validate_and_coerce_array(
                    model_shape=model_shape,
                    grid=value,
                    field_name=f"Capillary pressure grid {field.name}",
                    dtype_target=dtype,
                )
            else:
                capillary_dict[field.name] = value
        capillary_pressures = CapillaryPressureGrids(**capillary_dict)
    else:
        for field in attrs.fields(capillary_pressures.__class__):
            grid = getattr(capillary_pressures, field.name)
            if not isinstance(grid, np.ndarray):
                continue
            if grid.shape != model_shape:
                raise ValidationError(
                    f"Capillary pressure grid {field.name} has shape {grid.shape}, "
                    f"expected {model_shape}."
                )

    # Validate and coerce thickness grid and saturation history
    if dtype is not None:
        thickness_grid = _validate_and_coerce_array(
            model_shape=model_shape,
            grid=thickness_grid,
            field_name="Thickness grid",
            dtype_target=dtype,
        )
        sat_hist = model.saturation_history
        saturation_history = SaturationHistory(
            max_water_saturation_grid=_validate_and_coerce_array(
                model_shape=model_shape,
                grid=sat_hist.max_water_saturation_grid,
                field_name="Max water saturation grid",
                dtype_target=dtype,
            ),
            max_gas_saturation_grid=_validate_and_coerce_array(
                model_shape=model_shape,
                grid=sat_hist.max_gas_saturation_grid,
                field_name="Max gas saturation grid",
                dtype_target=dtype,
            ),
            water_imbibition_flag_grid=_validate_and_coerce_array(
                model_shape=model_shape,
                grid=sat_hist.water_imbibition_flag_grid,
                field_name="Water imbibition flag grid",
                dtype_target=dtype,
            ),
            gas_imbibition_flag_grid=_validate_and_coerce_array(
                model_shape=model_shape,
                grid=sat_hist.gas_imbibition_flag_grid,
                field_name="Gas imbibition flag grid",
                dtype_target=dtype,
            ),
        )
        # Reconstruct model and model state with coerced data
        model = ReservoirModel(
            grid_shape=model.grid_shape,
            cell_dimension=model.cell_dimension,
            thickness_grid=thickness_grid,
            fluid_properties=fluid_properties,  # type: ignore
            rock_properties=rock_properties,  # type: ignore
            rock_fluid_properties=model.rock_fluid_properties,
            saturation_history=saturation_history,
            boundary_conditions=model.boundary_conditions,  # type: ignore
            dip_angle=model.dip_angle,
            dip_azimuth=model.dip_azimuth,
        )
        state = ModelState(  # type: ignore
            step=state.step,
            step_size=state.step_size,
            time=state.time,
            model=model,
            wells=state.wells,  # type: ignore
            injection=injection,  # type: ignore
            production=production,  # type: ignore
            relative_permeabilities=relative_permeabilities,  # type: ignore
            relative_mobilities=relative_mobilities,  # type: ignore
            capillary_pressures=capillary_pressures,  # type: ignore
            timer_state=state.timer_state,  # type: ignore
        )
    return state
