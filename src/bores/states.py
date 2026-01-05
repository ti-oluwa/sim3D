import functools
import logging
from os import PathLike
import typing

import attrs
import numpy as np

from bores.config import Config
from bores.grids.base import (
    CapillaryPressureGrids,
    RateGrids,
    RelPermGrids,
    RelativeMobilityGrids,
)
from bores.models import ReservoirModel
from bores.types import NDimension, NDimensionalGrid
from bores.utils import load_from_pickle, save_as_pickle
from bores.wells import Wells

logger = logging.getLogger(__name__)


__all__ = ["ModelState", "dump_states", "load_states"]


@attrs.frozen(slots=True)
class ModelState(typing.Generic[NDimension]):
    """
    The state of the reservoir model at a specific time step during a simulation.
    """

    step: int
    """The time step index of the model state."""
    step_size: float
    """The time step size in seconds."""
    model: ReservoirModel[NDimension]
    """The reservoir model at this state."""
    wells: Wells[NDimension]
    """The wells configuration at this state."""
    config: Config
    """Simulation configuration used for this state."""
    injection: RateGrids[NDimension]
    """Fluids injection rates at this state in ft³/day."""
    production: RateGrids[NDimension]
    """Fluids production rates at this state in ft³/day."""
    relative_permeabilities: RelPermGrids[NDimension]
    """Relative permeabilities at this state."""
    relative_mobilities: RelativeMobilityGrids[NDimension]
    """Relative mobilities at this state."""
    capillary_pressures: CapillaryPressureGrids[NDimension]
    """Capillary pressures at this state."""

    @property
    def time(self) -> float:
        """
        Returns the total simulation time at this state.
        """
        return self.step * self.step_size

    @functools.cached_property
    def depth_grid(self) -> NDimensionalGrid[NDimension]:
        """
        Returns the depth grid of the reservoir model at this state.
        """
        return self.model.get_depth_grid(
            apply_dip=not self.config.disable_structural_dip
        )

    @functools.cached_property
    def elevation_grid(self) -> NDimensionalGrid[NDimension]:
        """
        Returns the elevation grid of the reservoir model at this state.
        """
        return self.model.get_elevation_grid(
            apply_dip=not self.config.disable_structural_dip
        )

    def dump(
        self,
        filepath: PathLike,
        exist_ok: bool = True,
        compression: typing.Optional[typing.Literal["gzip", "lzma"]] = "gzip",
        compression_level: int = 6,
    ) -> None:
        """
        Dumps the model state to a pickle file.

        :param filepath: The path to the pickle file or a file-like object.
        :param exist_ok: If True, will overwrite existing files.
        :param compression: Compression method - "gzip" (fast, good compression),
            "lzma" (slower, better compression), or None
        :param compression_level: Compression level (1-9 for gzip, 0-9 for lzma)
        """
        save_as_pickle(
            self,
            filepath,
            exist_ok=exist_ok,
            compression=compression,
            compression_level=compression_level,
        )

    @classmethod
    def load(cls, filepath: PathLike) -> "ModelState[NDimension]":
        """
        Loads a model state from a pickle file.

        :param filepath: The path to the pickle file or a file-like object.
        :return: The loaded ModelState instance.
        """
        return load_from_pickle(filepath)


def _validate_dumped_state(state: ModelState) -> ModelState:
    if not isinstance(state, ModelState):
        raise TypeError(
            f"Expected ModelState instance, got {type(state).__name__} instead."
        )
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
        raise ValueError(
            f"Thickness grid has shape {thickness_grid.shape}, expected {model_shape}."
        )

    for field in attrs.fields(fluid_properties.__class__):
        grid = getattr(fluid_properties, field.name)
        if not isinstance(grid, np.ndarray):
            continue
        if grid.shape != model_shape:
            raise ValueError(
                f"Fluid property grid {field.name} has shape {grid.shape}, "
                f"expected {model_shape}."
            )
    for field in attrs.fields(rock_properties.__class__):
        grid = getattr(rock_properties, field.name)
        if not isinstance(grid, np.ndarray):
            continue
        if grid.shape != model_shape:
            raise ValueError(
                f"Rock property grid {field.name} has shape {grid.shape}, "
                f"expected {model_shape}."
            )
    for field in attrs.fields(injection.__class__):
        grid = getattr(injection, field.name)
        if not isinstance(grid, np.ndarray):
            continue
        if grid.shape != model_shape:
            raise ValueError(
                f"Injection rate grid {field.name} has shape {grid.shape}, "
                f"expected {model_shape}."
            )
    for field in attrs.fields(production.__class__):
        grid = getattr(production, field.name)
        if not isinstance(grid, np.ndarray):
            continue
        if grid.shape != model_shape:
            raise ValueError(
                f"Production rate grid {field.name} has shape {grid.shape}, "
                f"expected {model_shape}."
            )
    for field in attrs.fields(relative_mobilities.__class__):
        grid = getattr(relative_mobilities, field.name)
        if not isinstance(grid, np.ndarray):
            continue
        if grid.shape != model_shape:
            raise ValueError(
                f"Relative mobility grid {field.name} has shape {grid.shape}, "
                f"expected {model_shape}."
            )
    for field in attrs.fields(relative_permeabilities.__class__):
        grid = getattr(relative_permeabilities, field.name)
        if not isinstance(grid, np.ndarray):
            continue
        if grid.shape != model_shape:
            raise ValueError(
                f"Relative permeability grid {field.name} has shape {grid.shape}, "
                f"expected {model_shape}."
            )
    for field in attrs.fields(capillary_pressures.__class__):
        grid = getattr(capillary_pressures, field.name)
        if not isinstance(grid, np.ndarray):
            continue
        if grid.shape != model_shape:
            raise ValueError(
                f"Capillary pressure grid {field.name} has shape {grid.shape}, "
                f"expected {model_shape}."
            )
    return state


def dump_states(
    states: typing.Iterable[ModelState],
    filepath: PathLike,
    exist_ok: bool = True,
    compression: typing.Optional[typing.Literal["gzip", "lzma"]] = "gzip",
    compression_level: int = 6,
) -> None:
    """
    Dumps multiple model states to pickle files in a specified directory.

    :param states: An iterable of `ModelState` instances to be dumped.
    :param filepath: The path to the pickle file or a file-like object.
    :param exist_ok: If True, will overwrite existing files.
    :param compression: Compression method - "gzip" (fast, good compression),
        "lzma" (slower, better compression), or None
    :param compression_level: Compression level (1-9 for gzip, 0-9 for lzma)
    """
    save_as_pickle(
        [_validate_dumped_state(state) for state in states],
        filepath,
        exist_ok=exist_ok,
        compression=compression,
        compression_level=compression_level,
    )


def _load_states(
    states: typing.Iterable[ModelState[NDimension]], as_tuple: bool = False
) -> typing.Generator[
    typing.Union[typing.Tuple[int, ModelState[NDimension]], ModelState[NDimension]],
    None,
    None,
]:
    for state in states:
        # Older pickles may have inconsistent grid shapes due to omitting unpadding
        # when saving. Check and unpad if necessary.
        grid_shape = state.model.grid_shape
        relative_mobilities = state.relative_mobilities
        capillary_pressures = state.capillary_pressures
        relative_permeabilities = state.relative_permeabilities
        relative_mobilities_shape = relative_mobilities.oil_relative_mobility.shape
        capillary_pressures_shape = (
            capillary_pressures.oil_water_capillary_pressure.shape
        )
        relative_permeabilities_shape = relative_permeabilities.kro.shape
        if relative_mobilities_shape != grid_shape:
            logger.warning(
                f"State at time step {state.step} has inconsistent relative mobility grid shapes. "
                "Recomputing relative mobilities."
            )
            relative_mobilities = relative_mobilities.unpad(pad_width=1)

        if capillary_pressures_shape != grid_shape:
            logger.warning(
                f"State at time step {state.step} has inconsistent capillary pressure grid shapes. "
                "Recomputing capillary pressures."
            )
            capillary_pressures = capillary_pressures.unpad(pad_width=1)
        if relative_permeabilities_shape != grid_shape:
            logger.warning(
                f"State at time step {state.step} has inconsistent relative permeability grid shapes. "
                "Recomputing relative permeabilities."
            )
            relative_permeabilities = relative_permeabilities.unpad(pad_width=1)

        state = attrs.evolve(
            state,
            relative_mobilities=relative_mobilities,
            capillary_pressures=capillary_pressures,
            relative_permeabilities=relative_permeabilities,
        )
        if as_tuple:
            yield state.step, state
        else:
            yield state


@typing.overload
def load_states(
    filepath: PathLike, as_tuple: typing.Literal[True]
) -> typing.Generator[typing.Tuple[int, ModelState], None, None]:
    """Loads multiple model states from a pickle file."""
    ...


@typing.overload
def load_states(
    filepath: PathLike, as_tuple: typing.Literal[False]
) -> typing.Generator[ModelState, None, None]:
    """Loads multiple model states from a pickle file."""
    ...


@typing.overload
def load_states(filepath: PathLike) -> typing.Generator[ModelState, None, None]:
    """Loads multiple model states from a pickle file."""
    ...


def load_states(
    filepath: PathLike, as_tuple: bool = False
) -> typing.Generator[
    typing.Union[typing.Tuple[int, ModelState], ModelState], None, None
]:
    """
    Loads multiple model states from pickle files in a specified directory.

    :param filepath: The path to the pickle file or a file-like object.
    :param as_tuple: If True, yields (step, ModelState) tuples.
    :return: A dictionary mapping time step indices to ModelState instances.
    """
    states = load_from_pickle(filepath)
    if isinstance(states, dict):  # For older pickles saved as dicts
        return _load_states(states.values(), as_tuple=as_tuple)
    return _load_states(states, as_tuple=as_tuple)



class StateStore(typing.Protocol):

    def load(self, *args, **kwargs) -> typing.Iterable[ModelState]:
        ...

    def dump(self, states: typing.Iterable[ModelState], *args, **kwargs) -> None:
        ...
