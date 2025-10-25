import functools
import typing
import logging

import attrs
import numpy as np
from scipy.optimize import curve_fit

from sim3D.constants import ACRE_FT_TO_FT3, FT2_TO_ACRES, FT3_TO_BBL
from sim3D.grids import uniform_grid
from sim3D.properties import compute_hydrocarbon_in_place
from sim3D.statics import ReservoirModel
from sim3D.types import NDimension, RateGrids
from sim3D.wells import Wells, _expand_intervals

logger = logging.getLogger(__name__)


__all__ = ["ModelState", "ModelAnalyst"]


@attrs.define(frozen=True, slots=True)
class ReservoirVolumetrics:
    """Reservoir volumetrics analysis results."""

    oil_in_place: float
    """Total oil in place in stock tank barrels (STB)."""
    gas_in_place: float
    """Total gas in place in standard cubic feet (scf)."""
    water_in_place: float
    """Total water in place in stock tank barrels (STB)."""
    pore_volume: float
    """Total pore volume in cubic feet (ft³)."""
    hydrocarbon_pore_volume: float
    """Hydrocarbon pore volume in cubic feet (ft³)."""


@attrs.define(frozen=True, slots=True)
class InstantaneousRates:
    """Instantaneous production/injection rates."""

    oil_rate: float
    """Oil production/injection rate in stock tank barrels per day (STB/day)."""
    gas_rate: float
    """Gas production/injection rate in standard cubic feet per day (scf/day)."""
    water_rate: float
    """Water production/injection rate in stock tank barrels per day (STB/day)."""
    total_liquid_rate: float
    """Total liquid (oil + water) rate in stock tank barrels per day (STB/day)."""
    gas_oil_ratio: float
    """Gas-oil ratio in standard cubic feet per stock tank barrel (scf/STB)."""
    water_cut_fraction: float
    """Water cut as a fraction (0 to 1) of total liquid production."""


@attrs.define(frozen=True, slots=True)
class CumulativeProduction:
    """Cumulative production analysis results."""

    cumulative_oil: float
    """Cumulative oil produced in stock tank barrels (STB)."""
    cumulative_gas: float
    """Cumulative gas produced in standard cubic feet (scf)."""
    cumulative_water: float
    """Cumulative water produced in stock tank barrels (STB)."""
    oil_recovery_factor: float
    """Oil recovery factor as a fraction (0 to 1) of initial oil in place."""
    gas_recovery_factor: float
    """Gas recovery factor as a fraction (0 to 1) of initial gas in place."""


@attrs.define(frozen=True, slots=True)
class MaterialBalanceAnalysis:
    """Material balance analysis results."""

    pressure: float
    """Current reservoir pressure in pounds per square inch absolute (psia)."""
    oil_expansion_factor: float
    """Oil expansion factor relative to initial conditions."""
    solution_gas_drive_index: float
    """Solution gas drive index as a fraction of total production mechanism."""
    gas_cap_drive_index: float
    """Gas cap drive index as a fraction of total production mechanism."""
    water_drive_index: float
    """Water drive index as a fraction of total production mechanism."""
    compaction_drive_index: float
    """Compaction drive index as a fraction of total production mechanism."""
    aquifer_influx: float
    """Estimated aquifer influx in stock tank barrels (STB)."""


@attrs.define(frozen=True, slots=True)
class ProductivityAnalysis:
    """Well productivity analysis results."""

    productivity_index: float
    """Productivity index in stock tank barrels per day per psi (STB/day/psi) or (SCF/day/psi)."""
    inflow_performance_relationship: float
    """Inflow performance relationship flow rate in stock tank barrels per day (STB/day)."""
    skin_factor: float
    """Dimensionless skin factor indicating wellbore damage or stimulation."""
    flow_efficiency: float
    """Flow efficiency as a fraction (0 to 1) accounting for skin effects."""
    ipr_method: typing.Optional[
        typing.Literal["vogel", "linear", "fetkovich", "jones"]
    ] = None
    """IPR correlation method used for the analysis."""


@attrs.define(frozen=True, slots=True)
class SweepEfficiencyAnalysis:
    """Sweep efficiency analysis results."""

    volumetric_sweep_efficiency: float
    """Volumetric sweep efficiency as a fraction (0 to 1) of reservoir contacted."""
    displacement_efficiency: float
    """Displacement efficiency as a fraction (0 to 1) in contacted zones."""
    recovery_efficiency: float
    """Overall recovery efficiency as a fraction (0 to 1) combining sweep and displacement."""
    contacted_oil: float
    """Oil in contacted reservoir zones in stock tank barrels (STB)."""
    uncontacted_oil: float
    """Oil in uncontacted reservoir zones in stock tank barrels (STB)."""


@attrs.define(frozen=True, slots=True)
class DeclineCurveResult:
    """Decline curve analysis results."""

    decline_type: typing.Literal["exponential", "hyperbolic", "harmonic"]
    """Type of decline curve analysis performed."""
    initial_rate: float
    """Initial production rate in stock tank barrels per day (STB/day) for oil/water or standard cubic feet per day (scf/day) for gas."""
    decline_rate_per_year: float
    """Decline rate per year as a fraction."""
    b_factor: float
    """Hyperbolic decline exponent (0 for exponential, 1 for harmonic)."""
    r_squared: float
    """Coefficient of determination (R²) indicating goodness of fit."""
    phase: typing.Literal["oil", "gas", "water"] = "oil"
    """Phase analyzed ('oil', 'gas', 'water')."""
    error: typing.Optional[str] = None
    """Error message if analysis could not be completed."""
    time_steps: typing.Optional[typing.List[int]] = None
    """Time steps used in the analysis."""
    actual_rates: typing.Optional[typing.List[float]] = None
    """Actual production rates in STB/day or scf/day depending on phase."""
    predicted_rates: typing.Optional[typing.List[float]] = None
    """Predicted production rates from decline curve in STB/day or scf/day depending on phase."""


@attrs.define(frozen=True, slots=True)
class ModelState(typing.Generic[NDimension]):
    """
    The state of the reservoir model at a specific time step during a simulation.
    """

    time_step: int
    """The time step index of the model state."""
    time_step_size: float
    """The time step size in seconds."""
    model: ReservoirModel[NDimension]
    """The reservoir model at this state."""
    wells: Wells[NDimension]
    """The wells configuration at this state."""
    injection: typing.Optional[RateGrids[NDimension]] = None
    """Fluids injection rates at this state in ft³/day."""
    production: typing.Optional[RateGrids[NDimension]] = None
    """Fluids production rates at this state in ft³/day."""

    @property
    def time(self) -> float:
        """
        Returns the total simulation time at this state.
        """
        return self.time_step * self.time_step_size


hcip_vectorized = np.vectorize(
    compute_hydrocarbon_in_place, excluded=["hydrocarbon_type"], cache=True
)


class ModelAnalyst(typing.Generic[NDimension]):
    """
    Analysis tools for evaluating reservoir performance over a series of model states.
    """

    def __init__(self, states: typing.Iterable[ModelState[NDimension]]) -> None:
        """
        Initializes the model analyst with a series of model states.

        :param states: An iterable of `ModelState` instances representing the simulation states.
        """
        self._states = sorted(states, key=lambda s: s.time_step)
        self._max_time_step = self._states[-1].time_step
        self._state_count = len(self._states)
        if self._max_time_step != (self._state_count - 1):
            logger.debug(
                "Model states have non-sequential time steps. Max time step: %d, State count: %d",
                self._max_time_step,
                self._state_count,
            )

    def get_state(self, time_step: int) -> ModelState[NDimension]:
        """
        Retrieves the model state for a specific time step.

        :param time_step: The time step index to retrieve the state for.
        :return: The ModelState corresponding to the specified time step.
        """
        return self._states[time_step]

    @property
    def stock_tank_oil_initially_in_place(self) -> float:
        """The stock tank oil initially in place (STOIIP) at the start of the simulation in stock tank barrels (STB)."""
        return self.oil_in_place(0)

    stoiip = stock_tank_oil_initially_in_place
    """The stock tank oil initially in place (STOIIP) in stock tank barrels (STB)."""

    @property
    def stock_tank_gas_initially_in_place(self) -> float:
        """The stock tank gas initially in place (STGIIP) at the start of the simulation in standard cubic feet (scf)."""
        return self.gas_in_place(0)

    stgiip = stock_tank_gas_initially_in_place
    """The stock tank gas initially in place (STGIIP) in standard cubic feet (scf)."""

    @property
    def stock_tank_water_initially_in_place(self) -> float:
        """The stock tank water initially in place at the start of the simulation in stock tank barrels (STB)."""
        return self.water_in_place(0)

    @property
    def cumulative_oil_produced(self) -> float:
        """The cumulative oil produced in stock tank barrels (STB) from the start of the simulation to the current time step."""
        return self.net_oil_produced(0, -1)

    No = cumulative_oil_produced
    """Cumulative oil produced in stock tank barrels (STB)."""

    @property
    def cumulative_gas_produced(self) -> float:
        """Return the cumulative gas produced in standard cubic feet (SCF)."""
        return self.net_gas_produced(0, -1)

    Ng = cumulative_gas_produced
    """Cumulative gas produced in standard cubic feet (scf)."""

    @property
    def cumulative_water_produced(self) -> float:
        """Return the cumulative water produced in stock tank barrels (STB)."""
        return self.net_water_produced(0, -1)

    Nw = cumulative_water_produced
    """Cumulative water produced in stock tank barrels (STB)."""

    @property
    def oil_recovery_factor(self) -> float:
        """
        The recovery factor based on initial oil in place and cumulative oil produced
        over the entire simulation period.

        :return: The recovery factor as a fraction (0 to 1)
        """
        if self.stock_tank_oil_initially_in_place == 0:
            return 0.0
        return self.cumulative_oil_produced / self.stock_tank_oil_initially_in_place

    @property
    def gas_recovery_factor(self) -> float:
        """
        The recovery factor based on initial gas in place and cumulative gas produced
        over the entire simulation period.

        :return: The recovery factor as a fraction (0 to 1)
        """
        if self.stock_tank_gas_initially_in_place == 0:
            return 0.0
        return self.cumulative_gas_produced / self.stock_tank_gas_initially_in_place

    @functools.cache
    def _get_cell_area_in_acres(self, x_dim: float, y_dim: float) -> float:
        """
        Computes the area of a grid cell in acres.

        :param x_dim: The dimension of the cell in the x-direction (ft).
        :param y_dim: The dimension of the cell in the y-direction (ft).
        :return: The area of the cell in acres.
        """
        cell_area_in_ft2 = x_dim * y_dim
        return cell_area_in_ft2 * FT2_TO_ACRES

    @functools.cache
    def oil_in_place(self, time_step: int = -1) -> float:
        """
        Computes the total oil in place at a specific time step.

        :param time_step: The time step index to compute oil in place for.
        :return: The total oil in place in STB
        """
        state = self.get_state(time_step)
        model = state.model
        cell_area_in_acres = self._get_cell_area_in_acres(*model.cell_dimension[:2])
        cell_area_grid = uniform_grid(
            grid_shape=model.grid_shape,
            value=cell_area_in_acres,
            dtype=np.float64,
        )
        stoiip_grid = hcip_vectorized(
            area=cell_area_grid,
            thickness=model.thickness_grid,
            porosity=model.rock_properties.porosity_grid,
            phase_saturation=model.fluid_properties.oil_saturation_grid,
            formation_volume_factor=model.fluid_properties.oil_formation_volume_factor_grid,
            net_to_gross_ratio=model.rock_properties.net_to_gross_ratio_grid,
            hydrocarbon_type="oil",
        )
        return np.nansum(stoiip_grid)

    @functools.cache
    def gas_in_place(self, time_step: int = -1) -> float:
        """
        Computes the total free gas in place at a specific time step.

        :param time_step: The time step index to compute gas in place for.
        :return: The total free gas in place in SCF
        """
        state = self.get_state(time_step)
        model = state.model
        cell_area_in_acres = self._get_cell_area_in_acres(*model.cell_dimension[:2])
        cell_area_grid = uniform_grid(
            grid_shape=model.grid_shape,
            value=cell_area_in_acres,
            dtype=np.float64,
        )
        stgiip_grid = hcip_vectorized(
            area=cell_area_grid,
            thickness=model.thickness_grid,
            porosity=model.rock_properties.porosity_grid,
            phase_saturation=model.fluid_properties.gas_saturation_grid,
            formation_volume_factor=model.fluid_properties.gas_formation_volume_factor_grid,
            net_to_gross_ratio=model.rock_properties.net_to_gross_ratio_grid,
            hydrocarbon_type="gas",
        )
        return np.nansum(stgiip_grid)

    @functools.cache
    def water_in_place(self, time_step: int = -1) -> float:
        """
        Computes the total water in place at a specific time step.

        :param time_step: The time step index to compute water in place for.
        :return: The total water in place in STB
        """
        state = self.get_state(time_step)
        model = state.model
        cell_area_in_acres = self._get_cell_area_in_acres(*model.cell_dimension[:2])
        cell_area_grid = uniform_grid(
            grid_shape=model.grid_shape,
            value=cell_area_in_acres,
            dtype=np.float64,
        )
        water_in_place_grid = hcip_vectorized(
            area=cell_area_grid,
            thickness=model.thickness_grid,
            porosity=model.rock_properties.porosity_grid,
            phase_saturation=model.fluid_properties.water_saturation_grid,
            formation_volume_factor=model.fluid_properties.water_formation_volume_factor_grid,
            net_to_gross_ratio=model.rock_properties.net_to_gross_ratio_grid,
            hydrocarbon_type="oil",  # Use "oil" since there's no "water" hydrocarbon_type and they use equivalent calculation
        )
        return np.nansum(water_in_place_grid)

    def oil_in_place_history(
        self, from_time_step: int = 0, to_time_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the oil in place history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: A generator yielding tuples of time step and oil in place in (STB).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.oil_in_place(t))

    def gas_in_place_history(
        self, from_time_step: int = 0, to_time_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Computes the free gas in place history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: A generator yielding tuples of time step and gas in place in (SCF).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.gas_in_place(t))

    def water_in_place_history(
        self, from_time_step: int = 0, to_time_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Computes the water in place history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: A generator yielding tuples of time step and water in place in (STB).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.water_in_place(t))

    @functools.cache
    def net_oil_produced(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative net oil produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - injection rates are present, they contribute negatively to production.
        - `from_time_step` equals `to_time_step`, the production at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative production is returned.

        Negative production values indicate net injection.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative oil produced in STB
        """
        total_production = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Production is in ft³/day, convert to STB using FVF
            oil_production = (
                state.production.oil if state.production is not None else None
            )
            oil_injection = state.injection.oil if state.injection is not None else None
            time_step_in_days = (
                state.time_step_size / 86400.0
            )  # Convert seconds to days
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid

            time_step_production = 0.0
            if oil_production is not None:
                oil_production_stb_day = oil_production * FT3_TO_BBL / oil_fvf_grid
                time_step_production += np.nansum(
                    oil_production_stb_day * time_step_in_days
                )

            if oil_injection is not None:
                oil_injection_stb_day = oil_injection * FT3_TO_BBL / oil_fvf_grid
                time_step_production -= np.nansum(
                    oil_injection_stb_day * time_step_in_days
                )

            total_production += time_step_production
            print(f"Time Step {t}: Total Oil Production STB = {time_step_production}")
        return float(total_production)

    @functools.cache
    def net_gas_produced(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative net gas produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - injection rates are present, they contribute negatively to production.
        - `from_time_step` equals `to_time_step`, the production at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative production is returned.

        Negative production values indicate net injection.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative gas produced in scf
        """
        total_production = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Production is in ft³/day, convert to scf using FVF
            gas_production = (
                state.production.gas if state.production is not None else None
            )
            gas_injection = state.injection.gas if state.injection is not None else None
            time_step_in_days = (
                state.time_step_size / 86400.0
            )  # Convert seconds to days
            gas_fvf_grid = state.model.fluid_properties.gas_formation_volume_factor_grid

            time_step_production = 0.0
            if gas_production is not None:
                gas_production_scf_day = gas_production / gas_fvf_grid
                time_step_production += np.nansum(
                    gas_production_scf_day * time_step_in_days
                )

            if gas_injection is not None:
                gas_injection_scf_day = gas_injection / gas_fvf_grid
                time_step_production -= np.nansum(
                    gas_injection_scf_day * time_step_in_days
                )

            total_production += time_step_production
            print(f"Time Step {t}: Total Gas Production SCF = {time_step_production}")

        return float(total_production)

    @functools.cache
    def net_water_produced(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative net water produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - injection rates are present, they contribute negatively to production.
        - `from_time_step` equals `to_time_step`, the production at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative production is returned.

        Negative production values indicate net injection.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative water produced in STB
        """
        total_production = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Production is in ft³/day, convert to STB using FVF
            water_production = (
                state.production.water if state.production is not None else None
            )
            water_injection = (
                state.injection.water if state.injection is not None else None
            )
            time_step_in_days = (
                state.time_step_size / 86400.0
            )  # Convert seconds to days
            water_fvf_grid = (
                state.model.fluid_properties.water_formation_volume_factor_grid
            )

            time_step_production = 0.0
            if water_production is not None:
                water_production_stb_day = (
                    water_production * FT3_TO_BBL / water_fvf_grid
                )
                time_step_production += np.nansum(
                    water_production_stb_day * time_step_in_days
                )

            if water_injection is not None:
                water_injection_stb_day = water_injection * FT3_TO_BBL / water_fvf_grid
                time_step_production -= np.nansum(
                    water_injection_stb_day * time_step_in_days
                )

            total_production += time_step_production
            print(f"Time Step {t}: Total Water Production STB = {time_step_production}")
        return float(total_production)

    def net_oil_injected(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative net oil injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - production rates are present, they contribute negatively to injection.
        - `from_time_step` equals `to_time_step`, the injection at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative injection is returned.

        Negative injection values indicate net production.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative oil injected in STB
        """
        # Negate net oil production to get net injection
        oil_injected = -self.net_oil_produced(from_time_step, to_time_step)
        return oil_injected

    def net_gas_injected(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative net gas injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - production rates are present, they contribute negatively to injection.
        - `from_time_step` equals `to_time_step`, the injection at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative injection is returned.

        Negative injection values indicate net production.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative gas injected in scf
        """
        # Negate net gas production to get net injection
        gas_injected = -self.net_gas_produced(from_time_step, to_time_step)
        return gas_injected

    def net_water_injected(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative net water injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - production rates are present, they contribute negatively to injection.
        - `from_time_step` equals `to_time_step`, the injection at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative injection is returned.

        Negative injection values indicate net production.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative water injected in STB
        """
        # Negate net water production to get net injection
        water_injected = -self.net_water_produced(from_time_step, to_time_step)
        return water_injected

    def oil_production_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the oil production history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative production from start. If False, returns production at each time step.
        :return: A generator yielding tuples of time step and oil produced (cumulative or exclusive).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            if cumulative:
                # Cumulative production from start of simulation to time step t
                yield (t, self.net_oil_produced(0, t))
            else:
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.net_oil_produced(t, t))

    def gas_production_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the gas production history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative production from start. If False, returns production at each time step.
        :return: A generator yielding tuples of time step and gas produced (cumulative or exclusive).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            if cumulative:
                # Cumulative production from start of simulation
                yield (t, self.net_gas_produced(0, t))
            else:
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.net_gas_produced(t, t))

    def water_production_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the water production history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative production from start. If False, returns production at each time step.
        :return: A generator yielding tuples of time step and water produced (cumulative or exclusive).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            if cumulative:
                # Cumulative production from start of simulation
                yield (t, self.net_water_produced(0, t))
            else:
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.net_water_produced(t, t))

    def oil_injection_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the oil injection history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative injection from start. If False, returns injection at each time step.
        :return: A generator yielding tuples of time step and oil injected (cumulative or exclusive).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            if cumulative:
                # Cumulative injection from start of simulation
                yield (t, self.net_oil_injected(0, t))
            else:
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.net_oil_injected(t, t))

    def gas_injection_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the gas injection history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative injection from start. If False, returns injection at each time step.
        :return: A generator yielding tuples of time step and gas injected (cumulative or exclusive).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            if cumulative:
                # Cumulative injection from start of simulation
                yield (t, self.net_gas_injected(0, t))
            else:
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.net_gas_injected(t, t))

    def water_injection_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the water injection history between two time steps.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :param cumulative: If True (default), returns cumulative injection from start. If False, returns injection at each time step.
        :return: A generator yielding tuples of time step and water injected (cumulative or exclusive).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            if cumulative:
                # Cumulative injection from start of simulation
                yield (t, self.net_water_injected(0, t))
            else:
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.net_water_injected(t, t))

    def reservoir_volumetrics_analysis(
        self, time_step: int = -1
    ) -> ReservoirVolumetrics:
        """
        Comprehensive reservoir volumetrics analysis at a specific time step.

        :param time_step: The time step index to analyze volumetrics for.
        :return: `ReservoirVolumetrics` containing detailed volume analysis.
        """
        state = self.get_state(time_step)
        model = state.model

        cell_area_in_acres = self._get_cell_area_in_acres(*model.cell_dimension[:2])
        cell_area_grid = uniform_grid(
            grid_shape=model.grid_shape,
            value=cell_area_in_acres,
            dtype=np.float64,
        )
        pore_volume_grid = (
            cell_area_grid
            * model.thickness_grid
            * model.rock_properties.porosity_grid
            * model.rock_properties.net_to_gross_ratio_grid
            * ACRE_FT_TO_FT3  # Convert acre-ft to ft³
        )
        total_pore_volume = np.nansum(pore_volume_grid)

        hydrocarbon_saturation_grid = (
            model.fluid_properties.oil_saturation_grid
            + model.fluid_properties.gas_saturation_grid
        )
        hydrocarbon_pore_volume = np.nansum(
            pore_volume_grid * hydrocarbon_saturation_grid
        )
        return ReservoirVolumetrics(
            oil_in_place=self.oil_in_place(time_step),
            gas_in_place=self.gas_in_place(time_step),
            water_in_place=self.water_in_place(time_step),
            pore_volume=total_pore_volume,
            hydrocarbon_pore_volume=hydrocarbon_pore_volume,
        )

    def instantaneous_production_rates(self, time_step: int = -1) -> InstantaneousRates:
        """
        Calculates instantaneous production rates at a specific time step.

        :param time_step: The time step index to calculate rates for.
        :return: `InstantaneousRates` containing detailed rate analysis.
        """
        state = self.get_state(time_step)
        oil_rate = 0.0
        gas_rate = 0.0
        water_rate = 0.0

        # Sum production rates from all grid cells
        if state.production is not None:
            if state.production.oil is not None:
                # Convert from ft³/day to STB/day using oil FVF
                oil_fvf_grid = (
                    state.model.fluid_properties.oil_formation_volume_factor_grid
                )
                oil_rate = np.nansum(state.production.oil * FT3_TO_BBL / oil_fvf_grid)

            if state.production.gas is not None:
                # Convert from ft³/day to scf/day using gas FVF
                gas_fvf_grid = (
                    state.model.fluid_properties.gas_formation_volume_factor_grid
                )
                gas_rate = np.nansum(state.production.gas / gas_fvf_grid)

            if state.production.water is not None:
                # Convert from ft³/day to STB/day using water FVF
                water_fvf_grid = (
                    state.model.fluid_properties.water_formation_volume_factor_grid
                )
                water_rate = np.nansum(
                    state.production.water * FT3_TO_BBL / water_fvf_grid
                )

        total_liquid_rate = oil_rate + water_rate
        gas_oil_ratio = gas_rate / oil_rate if oil_rate > 0 else 0.0
        water_cut = water_rate / total_liquid_rate if total_liquid_rate > 0 else 0.0
        return InstantaneousRates(
            oil_rate=oil_rate,
            gas_rate=gas_rate,
            water_rate=water_rate,
            total_liquid_rate=total_liquid_rate,
            gas_oil_ratio=gas_oil_ratio,
            water_cut_fraction=water_cut,
        )

    def instantaneous_injection_rates(self, time_step: int = -1) -> InstantaneousRates:
        """
        Calculates instantaneous injection rates at a specific time step.

        :param time_step: The time step index to calculate rates for.
        :return: `InstantaneousRates` containing detailed injection rate analysis.
        """
        state = self.get_state(time_step)
        oil_rate = 0.0
        gas_rate = 0.0
        water_rate = 0.0

        # Sum injection rates from all grid cells
        if state.injection is not None:
            if state.injection.oil is not None:
                # Convert from ft³/day to STB/day using oil FVF
                oil_fvf_grid = (
                    state.model.fluid_properties.oil_formation_volume_factor_grid
                )
                oil_rate = np.nansum(state.injection.oil * FT3_TO_BBL / oil_fvf_grid)

            if state.injection.gas is not None:
                # Convert from ft³/day to scf/day using gas FVF
                gas_fvf_grid = (
                    state.model.fluid_properties.gas_formation_volume_factor_grid
                )
                gas_rate = np.nansum(state.injection.gas / gas_fvf_grid)

            if state.injection.water is not None:
                # Convert from ft³/day to STB/day using water FVF
                water_fvf_grid = (
                    state.model.fluid_properties.water_formation_volume_factor_grid
                )
                water_rate = np.nansum(
                    state.injection.water * FT3_TO_BBL / water_fvf_grid
                )

        total_liquid_rate = oil_rate + water_rate
        gas_oil_ratio = gas_rate / oil_rate if oil_rate > 0 else 0.0
        water_cut = water_rate / total_liquid_rate if total_liquid_rate > 0 else 0.0
        return InstantaneousRates(
            oil_rate=oil_rate,
            gas_rate=gas_rate,
            water_rate=water_rate,
            total_liquid_rate=total_liquid_rate,
            gas_oil_ratio=gas_oil_ratio,
            water_cut_fraction=water_cut,
        )

    def cumulative_production_analysis(
        self, time_step: int = -1
    ) -> CumulativeProduction:
        """
        Comprehensive cumulative production analysis at a specific time step.

        :param time_step: The time step index to analyze cumulative production for.
        :return: `CumulativeProduction` containing detailed cumulative analysis.
        """
        cumulative_oil = self.net_oil_produced(0, time_step)
        cumulative_gas = self.net_gas_produced(0, time_step)
        cumulative_water = self.net_water_produced(0, time_step)
        return CumulativeProduction(
            cumulative_oil=cumulative_oil,
            cumulative_gas=cumulative_gas,
            cumulative_water=cumulative_water,
            oil_recovery_factor=self.oil_recovery_factor,
            gas_recovery_factor=self.gas_recovery_factor,
        )

    def material_balance_analysis(self, time_step: int = -1) -> MaterialBalanceAnalysis:
        """
        Material balance analysis for reservoir drive mechanism identification.

        Uses the generalized material balance equation to quantify drive mechanisms:
        - Solution gas drive (oil expansion + gas coming out of solution)
        - Gas cap drive (free gas expansion)
        - Natural water drive (aquifer influx)
        - Rock and fluid compressibility drive
        - Combined drive indices

        The generalized material balance equation is:

        N * (Boi - Bo) + N * (Rsi - Rs) * Bg + G * (Bg - Bgi) + W * (Bw - Bwi) = V * ct * ΔP + We

        Where:
        - N = Initial oil in place (STB)
        - G = Initial gas in place (scf)
        - W = Initial water in place (STB)
        - Boi, Bo = Initial and current oil formation volume factors (bbl/STB)
        - Rsi, Rs = Initial and current solution gas-oil ratios (scf/STB)
        - Bg, Bgi = Current and initial gas formation volume factors (bbl/scf)
        - Bw, Bwi = Current and initial water formation volume factors (bbl/STB)
        - V = Pore volume (ft³)
        - ct = Total compressibility (1/psi)
        - ΔP = Pressure decline (psi)
        - We = Cumulative water influx (STB)
        - All volumes are in stock tank barrels (STB) unless otherwise noted.

        :param time_step: The time step index to analyze material balance for.
        :return: `MaterialBalanceAnalysis` containing drive mechanism analysis.
        """
        state = self.get_state(time_step)
        initial_state = self.get_state(0)

        # Current reservoir conditions
        current_pressure = np.nanmean(state.model.fluid_properties.pressure_grid)
        initial_pressure = np.nanmean(
            initial_state.model.fluid_properties.pressure_grid
        )
        pressure_decline = initial_pressure - current_pressure
        # Formation volume factors
        current_oil_fvf = np.nanmean(
            state.model.fluid_properties.oil_formation_volume_factor_grid
        )
        initial_oil_fvf = np.nanmean(
            initial_state.model.fluid_properties.oil_formation_volume_factor_grid
        )
        current_gas_fvf = np.nanmean(
            state.model.fluid_properties.gas_formation_volume_factor_grid
        )
        current_water_fvf = np.nanmean(
            state.model.fluid_properties.water_formation_volume_factor_grid
        )
        # Gas-oil ratio evolution
        current_gor = np.nanmean(state.model.fluid_properties.gas_to_oil_ratio_grid)
        initial_gor = np.nanmean(
            initial_state.model.fluid_properties.gas_to_oil_ratio_grid
        )
        # Saturation changes
        current_oil_sat = np.nanmean(state.model.fluid_properties.oil_saturation_grid)
        current_gas_sat = np.nanmean(state.model.fluid_properties.gas_saturation_grid)
        initial_gas_sat = np.nanmean(
            initial_state.model.fluid_properties.gas_saturation_grid
        )
        current_water_sat = np.nanmean(
            state.model.fluid_properties.water_saturation_grid
        )
        initial_water_sat = np.nanmean(
            initial_state.model.fluid_properties.water_saturation_grid
        )
        # Compressibilities
        rock_compressibility = state.model.rock_properties.compressibility
        oil_compressibility = np.nanmean(
            state.model.fluid_properties.oil_compressibility_grid
        )
        water_compressibility = np.nanmean(
            state.model.fluid_properties.water_compressibility_grid
        )
        gas_compressibility = np.nanmean(
            state.model.fluid_properties.gas_compressibility_grid
        )
        # Cumulative production
        cumulative_oil = self.net_oil_produced(0, time_step)
        cumulative_water = self.net_water_produced(0, time_step)

        # Initial volumes in place
        initial_oil = self.oil_in_place(0)
        initial_gas = self.gas_in_place(0)
        initial_water = self.water_in_place(0)

        # Calculate total compressibility (rock + fluid)
        total_compressibility = (
            rock_compressibility
            + current_oil_sat * oil_compressibility
            + current_water_sat * water_compressibility
            + current_gas_sat * gas_compressibility
        )

        # DRIVE MECHANISM CALCULATIONS
        # Solution Gas Drive (oil expansion + liberated gas)
        # ΔVo = N * (Bo - Boi) + N * (Rsi - Rs) * Bg
        oil_expansion_factor = current_oil_fvf / initial_oil_fvf
        oil_expansion_drive = (
            cumulative_oil * (current_oil_fvf - initial_oil_fvf) / initial_oil
            if initial_oil > 0
            else 0.0
        )

        # Gas liberation from oil
        gas_liberation_factor = (initial_gor - current_gor) * current_gas_fvf
        gas_liberation_drive = (
            cumulative_oil * gas_liberation_factor / initial_oil
            if initial_oil > 0
            else 0.0
        )
        solution_gas_drive = oil_expansion_drive + gas_liberation_drive

        # Gas Cap Drive (free gas expansion)
        # Estimated from gas saturation increase beyond solution gas effects
        gas_saturation_increase = current_gas_sat - initial_gas_sat
        gas_cap_expansion = gas_saturation_increase * current_gas_fvf
        gas_cap_drive = (
            gas_cap_expansion * (initial_gas / initial_oil) if initial_oil > 0 else 0.0
        )

        # Water Drive (aquifer influx + water injection)
        # Calculate net water influx considering production and saturation changes
        water_saturation_change = current_water_sat - initial_water_sat
        water_influx_from_saturation = water_saturation_change * current_water_fvf
        # Natural aquifer influx estimation
        current_water = self.water_in_place(time_step)
        aquifer_influx = max(0.0, current_water - initial_water + cumulative_water)
        water_drive = (
            (aquifer_influx + water_influx_from_saturation) / initial_oil
            if initial_oil > 0
            else 0.0
        )

        # Rock and Fluid Compressibility Drive
        # ΔVc = V * ct * ΔP
        pore_volume = (
            np.nansum(
                state.model.thickness_grid
                * state.model.rock_properties.porosity_grid
                * state.model.rock_properties.net_to_gross_ratio_grid
            )
            * self._get_cell_area_in_acres(*state.model.cell_dimension[:2])
            * ACRE_FT_TO_FT3
        )
        compressibility_expansion = (
            pore_volume * total_compressibility * pressure_decline * FT3_TO_BBL
        )
        compaction_drive = (
            compressibility_expansion / initial_oil if initial_oil > 0 else 0.0
        )

        # Normalize drive contributions to get drive indices
        total_drive = (
            solution_gas_drive + gas_cap_drive + water_drive + compaction_drive
        )
        if total_drive > 0:
            solution_gas_drive_index = solution_gas_drive / total_drive
            gas_cap_drive_index = gas_cap_drive / total_drive
            water_drive_index = water_drive / total_drive
            compaction_drive_index = compaction_drive / total_drive
        else:
            solution_gas_drive_index = 0.0
            gas_cap_drive_index = 0.0
            water_drive_index = 0.0
            compaction_drive_index = 0.0

        # Ensure drive indices sum to 1.0
        total_indices = (
            solution_gas_drive_index
            + gas_cap_drive_index
            + water_drive_index
            + compaction_drive_index
        )
        if total_indices > 0:
            solution_gas_drive_index /= total_indices
            gas_cap_drive_index /= total_indices
            water_drive_index /= total_indices
            compaction_drive_index /= total_indices

        return MaterialBalanceAnalysis(
            pressure=float(current_pressure),
            oil_expansion_factor=float(oil_expansion_factor),
            solution_gas_drive_index=float(solution_gas_drive_index),
            gas_cap_drive_index=float(gas_cap_drive_index),
            water_drive_index=float(water_drive_index),
            compaction_drive_index=float(compaction_drive_index),
            aquifer_influx=float(aquifer_influx),
        )

    def sweep_efficiency_analysis(self, time_step: int = -1) -> SweepEfficiencyAnalysis:
        """
        Sweep efficiency analysis to evaluate reservoir contact and displacement.

        :param time_step: The time step index to analyze sweep efficiency for.
        :return: `SweepEfficiencyAnalysis` containing sweep efficiency metrics.
        """
        state = self.get_state(time_step)
        initial_state = self.get_state(0)

        # Calculate volumetric sweep efficiency
        initial_oil_saturation = (
            initial_state.model.fluid_properties.oil_saturation_grid
        )
        current_oil_saturation = state.model.fluid_properties.oil_saturation_grid

        # Cells that have been contacted have a decreased oil saturation
        contacted_cells = current_oil_saturation < initial_oil_saturation
        total_cells = initial_oil_saturation.size
        contacted_fraction = np.sum(contacted_cells) / total_cells

        # Calculate displacement efficiency in contacted zones
        initial_oil_contacted = np.sum(initial_oil_saturation[contacted_cells])
        current_oil_contacted = np.sum(current_oil_saturation[contacted_cells])
        displacement_efficiency = float(
            (initial_oil_contacted - current_oil_contacted) / initial_oil_contacted
            if initial_oil_contacted > 0
            else 0.0
        )

        # Overall recovery efficiency
        recovery_efficiency = contacted_fraction * displacement_efficiency

        # Calculate contacted and uncontacted oil
        cell_area_in_acres = self._get_cell_area_in_acres(
            *state.model.cell_dimension[:2]
        )
        cell_area_grid = uniform_grid(
            grid_shape=state.model.grid_shape,
            value=cell_area_in_acres,
            dtype=np.float64,
        )
        oil_volume_grid = (
            cell_area_grid
            * state.model.thickness_grid
            * state.model.rock_properties.porosity_grid
            * state.model.rock_properties.net_to_gross_ratio_grid  # Include net-to-gross ratio
            * initial_oil_saturation
            / initial_state.model.fluid_properties.oil_formation_volume_factor_grid
            * ACRE_FT_TO_FT3
            / FT3_TO_BBL  # Convert to STB
        )
        contacted_oil = np.sum(oil_volume_grid[contacted_cells])
        uncontacted_oil = np.sum(oil_volume_grid[~contacted_cells])
        return SweepEfficiencyAnalysis(
            volumetric_sweep_efficiency=contacted_fraction,
            displacement_efficiency=displacement_efficiency,
            recovery_efficiency=recovery_efficiency,
            contacted_oil=contacted_oil,
            uncontacted_oil=uncontacted_oil,
        )

    def productivity_analysis(
        self,
        time_step: int = -1,
        ipr_method: typing.Literal["vogel", "linear", "fetkovich", "jones"] = "vogel",
        phase: typing.Literal["oil", "gas"] = "oil",
    ) -> ProductivityAnalysis:
        """
        Well productivity analysis using actual well model data with multiple IPR methods.

        :param time_step: The time step index to analyze productivity for.
        :param ipr_method: IPR correlation method ('vogel', 'linear', 'fetkovich', 'jones').
        :return: `ProductivityAnalysis` containing productivity metrics based on actual well model data.
        """
        state = self.get_state(time_step)
        if state.production is None:
            return ProductivityAnalysis(
                productivity_index=0.0,
                inflow_performance_relationship=0.0,
                skin_factor=0.0,
                flow_efficiency=1.0,
                ipr_method=ipr_method,
            )

        rates = self.instantaneous_production_rates(time_step)
        production_wells = state.wells.production_wells
        if not production_wells:
            return ProductivityAnalysis(
                productivity_index=0.0,
                inflow_performance_relationship=0.0,
                skin_factor=0.0,
                flow_efficiency=1.0,
                ipr_method=ipr_method,
            )

        # Calculate weighted averages based on actual well properties
        total_productivity_index = 0.0
        total_ipr_flow_rate = 0.0
        total_skin_factor = 0.0
        total_flow_efficiency = 0.0
        active_wells = 0

        for well in production_wells:
            if not well.is_open:
                continue

            active_wells += 1
            bottom_hole_pressure = well.bottom_hole_pressure
            actual_skin_factor = well.skin_factor
            # Well-level accumulators
            well_productivity_index = 0.0
            well_ipr_flow_rate = 0.0
            well_reservoir_pressure = 0.0

            cell_locations = _expand_intervals(
                well.perforating_intervals, orientation=well.orientation
            )
            for cell_location in cell_locations:
                i, j, k = cell_location
                cell_pressure = float(
                    state.model.fluid_properties.pressure_grid[i, j, k]
                )
                cell_pressure_drawdown = cell_pressure - bottom_hole_pressure
                if cell_pressure_drawdown < 0:
                    continue

                if phase == "oil":
                    if state.production.oil is None:
                        continue
                    oil_fvf = float(
                        state.model.fluid_properties.oil_formation_volume_factor_grid[
                            i, j, k
                        ]
                    )
                    cell_flow_rate = (
                        state.production.oil[i, j, k] * FT3_TO_BBL / oil_fvf
                    )  # stb/day
                else:
                    if state.production.gas is None:
                        continue
                    gas_fvf = float(
                        state.model.fluid_properties.gas_formation_volume_factor_grid[
                            i, j, k
                        ]
                    )
                    cell_flow_rate = state.production.gas[i, j, k] / gas_fvf  # scf/day

                # PI = q / (Pr - Pwf)
                cell_productivity_index = cell_flow_rate / cell_pressure_drawdown
                cell_ipr_flow_rate = self._calculate_ipr_flow_rate(
                    ipr_method=ipr_method,
                    reservoir_pressure=float(cell_pressure),
                    bottom_hole_pressure=float(bottom_hole_pressure),
                    current_rate=float(
                        rates.oil_rate / len(production_wells) / len(cell_locations)
                    ),
                    productivity_index=float(cell_productivity_index),
                    state=state,
                    cell_location=(i, j, k),
                )

                # Accumulate cell values for this well
                well_productivity_index += cell_productivity_index
                well_ipr_flow_rate += cell_ipr_flow_rate
                well_reservoir_pressure += cell_pressure
                well_productivity_index += cell_productivity_index
                well_ipr_flow_rate += cell_ipr_flow_rate
                well_reservoir_pressure += cell_pressure

            # Flow efficiency using actual skin factor
            well_flow_efficiency = (
                1.0 / (1.0 + actual_skin_factor) if actual_skin_factor > -1 else 1.0
            )
            total_productivity_index += well_productivity_index
            total_ipr_flow_rate += well_ipr_flow_rate
            total_skin_factor += actual_skin_factor
            total_flow_efficiency += well_flow_efficiency

        # Calculate averages across active wells
        if active_wells > 0:
            avg_productivity_index = total_productivity_index
            avg_ipr_flow_rate = total_ipr_flow_rate
            avg_skin_factor = total_skin_factor / active_wells
            avg_flow_efficiency = total_flow_efficiency / active_wells
        else:
            avg_productivity_index = 0.0
            avg_ipr_flow_rate = 0.0
            avg_skin_factor = 0.0
            avg_flow_efficiency = 1.0

        return ProductivityAnalysis(
            productivity_index=float(avg_productivity_index),
            inflow_performance_relationship=float(avg_ipr_flow_rate),
            skin_factor=avg_skin_factor,
            flow_efficiency=avg_flow_efficiency,
            ipr_method=ipr_method,
        )

    def _calculate_fetkovich_n_exponent(
        self,
        state: ModelState[NDimension],
        reservoir_pressure: float,
        bottom_hole_pressure: float,
        cell_location: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> float:
        """
        Calculate the n exponent for Fetkovich IPR based on reservoir conditions.

        The n exponent in the Fetkovich equation q = C * (Pr² - Pwf²)^n varies
        based on flow regime and fluid properties:
        - n = 1.0: Laminar flow (high gas saturation)
        - n = 0.5: Turbulent flow (liquid-dominated)
        - n = 0.5-1.0: Transitional flow regimes

        :param state: Current model state containing fluid properties
        :param reservoir_pressure: Reservoir pressure in psia
        :param bottom_hole_pressure: Bottom hole pressure in psia
        :param cell_location: Optional specific cell location (i, j, k) for cell-specific properties
        :return: Calculated n exponent (0.5 to 1.0)
        """
        if cell_location is not None:
            # Use cell-specific properties
            i, j, k = cell_location
            gas_saturation = float(
                state.model.fluid_properties.gas_saturation_grid[i, j, k]
            )
            oil_saturation = float(
                state.model.fluid_properties.oil_saturation_grid[i, j, k]
            )
            gor = float(state.model.fluid_properties.gas_to_oil_ratio_grid[i, j, k])
        else:
            # Use reservoir averages as fallback
            gas_saturation = np.nanmean(
                state.model.fluid_properties.gas_saturation_grid
            )
            oil_saturation = np.nanmean(
                state.model.fluid_properties.oil_saturation_grid
            )
            gor = np.nanmean(state.model.fluid_properties.gas_to_oil_ratio_grid)

        if gas_saturation > 0.8:
            # High gas saturation - laminar gas flow dominates
            n_exponent = 1.0  # Linear relationship for laminar flow
        elif gas_saturation > 0.5:
            # Moderate gas saturation - transitional flow
            # n varies between 0.5-1.0 based on gas fraction
            gas_fraction = gas_saturation / (gas_saturation + oil_saturation)
            n_exponent = 0.5 + 0.5 * gas_fraction
        elif gor > 1000:
            # High GOR reservoir - significant solution gas drive
            # Use pressure-dependent n based on non-Darcy effects
            pressure_ratio = reservoir_pressure / (
                reservoir_pressure + bottom_hole_pressure
            )
            n_exponent = 0.5 + 0.3 * pressure_ratio
        else:
            # Low gas content - closer to liquid flow
            n_exponent = 0.5  # Square root relationship for turbulent flow

        # Clamp n_exponent to physically reasonable bounds
        n_exponent = max(0.5, min(1.0, n_exponent))
        return float(n_exponent)

    def _calculate_ipr_flow_rate(
        self,
        ipr_method: typing.Literal["vogel", "linear", "fetkovich", "jones"],
        reservoir_pressure: float,
        bottom_hole_pressure: float,
        current_rate: float,
        productivity_index: float,
        state: ModelState[NDimension],
        cell_location: typing.Optional[typing.Tuple[int, int, int]] = None,
    ) -> float:
        """
        Calculate IPR flow rate using various correlations.

        Available IPR Methods:

        - "linear": q = PI x (Pr - Pwf) - Best for single-phase oil above bubble point
        - "vogel": Vogel's correlation - Best for solution gas drive reservoirs below bubble point
        - "fetkovich": q = C x (Pr² - Pwf²)^n - Best for gas wells and gas condensate
        - "jones": Combined linear/Vogel - Best for multi-phase flow with changing properties

        :param ipr_method: IPR correlation method
        :param reservoir_pressure: Reservoir pressure in psia
        :param bottom_hole_pressure: Bottom hole pressure in psia
        :param current_rate: Current production rate in STB/day
        :param productivity_index: Well productivity index in STB/day/psi
        :param state: Current model state
        :param cell_location: Optional specific cell location (i, j, k) for cell-specific properties
        :return: IPR flow rate in STB/day
        """
        if reservoir_pressure <= 0 or bottom_hole_pressure < 0:
            return 0.0

        if ipr_method not in {"vogel", "linear", "fetkovich", "jones"}:
            ipr_method = "vogel"

        normalized_pressure = bottom_hole_pressure / reservoir_pressure

        if ipr_method == "linear":
            # Linear IPR: q = PI * (Pr - Pwf)
            # Most accurate for single-phase oil above bubble point
            return productivity_index * (reservoir_pressure - bottom_hole_pressure)

        elif ipr_method == "vogel":
            # Vogel's IPR for solution gas drive reservoirs
            # q/qmax = 1 - 0.2*(Pwf/Pr) - 0.8*(Pwf/Pr)²
            if normalized_pressure >= 1.0:
                return 0.0

            vogel_factor = (
                1.0 - (0.2 * normalized_pressure) - (0.8 * normalized_pressure**2)
            )

            # Estimate qmax from current conditions
            if current_rate > 0 and vogel_factor > 0:
                qmax_estimate = current_rate / vogel_factor
            else:
                # Fallback estimation: assume current condition is at 80% depletion
                qmax_estimate = productivity_index * reservoir_pressure / 0.8

            return qmax_estimate * vogel_factor

        elif ipr_method == "fetkovich":
            # Fetkovich IPR for gas condensate wells and high-velocity gas flow
            # q = C * (Pr² - Pwf²)^n, typically n = 0.5 to 1.0

            n_exponent = self._calculate_fetkovich_n_exponent(
                state, reservoir_pressure, bottom_hole_pressure, cell_location
            )
            pressure_squared_diff = reservoir_pressure**2 - bottom_hole_pressure**2

            if pressure_squared_diff <= 0:
                return 0.0

            # Estimate C coefficient from current conditions
            if current_rate > 0:
                current_pressure_diff_squared = (
                    reservoir_pressure**2
                    - (reservoir_pressure - current_rate / productivity_index) ** 2
                )
                if current_pressure_diff_squared > 0:
                    c_coefficient = current_rate / (
                        current_pressure_diff_squared**n_exponent
                    )
                else:
                    c_coefficient = productivity_index / (2 * reservoir_pressure)
            else:
                # Fallback coefficient based on linear approximation
                c_coefficient = productivity_index / (2 * reservoir_pressure)

            return float(c_coefficient * (pressure_squared_diff**n_exponent))

        # ipr_method == "jones"
        # Jones, Blount, and Glaze IPR for multi-phase flow
        # Combines linear flow above bubble point with Vogel below bubble point
        if cell_location is not None:
            # Use cell-specific bubble point pressure
            bubble_point_pressure = float(
                state.model.fluid_properties.oil_bubble_point_pressure_grid[
                    cell_location
                ]
            )
        else:
            # Use reservoir average bubble point pressure
            bubble_point_pressure = np.nanmean(
                state.model.fluid_properties.oil_bubble_point_pressure_grid
            )

        if bottom_hole_pressure >= bubble_point_pressure:
            # Above bubble point - linear IPR (single-phase oil)
            return productivity_index * (reservoir_pressure - bottom_hole_pressure)

        # Below bubble point - combination approach
        # Linear portion from reservoir pressure to bubble point
        linear_rate = productivity_index * (reservoir_pressure - bubble_point_pressure)

        # Vogel portion from bubble point to bottom hole pressure
        if bubble_point_pressure > 0:
            pb_normalized = bottom_hole_pressure / bubble_point_pressure
            pb_normalized = max(0.0, min(1.0, pb_normalized))  # Clamp to valid range

            # Modified Vogel equation for the portion below bubble point
            vogel_portion = (
                productivity_index
                * bubble_point_pressure
                * (1.0 - 0.2 * pb_normalized - 0.8 * pb_normalized**2)
            )
        else:
            vogel_portion = 0.0

        return float(linear_rate + vogel_portion)

    def decline_curve_analysis(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        decline_type: typing.Literal[
            "exponential", "hyperbolic", "harmonic"
        ] = "exponential",
        phase: typing.Literal["oil", "gas", "water"] = "oil",
    ) -> DeclineCurveResult:
        """
        Decline curve analysis for production forecasting.

        Performs decline curve analysis using exponential, hyperbolic, or harmonic models:
        - Exponential: q = qi * exp(-Di * t)
        - Hyperbolic: q = qi / (1 + b * Di * t)^(1/b)
        - Harmonic: q = qi / (1 + Di * t) [special case of hyperbolic with b=1]

        :param from_time_step: Starting time step for analysis.
        :param to_time_step: Ending time step for analysis.
        :param decline_type: Type of decline curve ('exponential', 'hyperbolic', 'harmonic').
        :param phase: Phase to analyze ('oil', 'gas', 'water').
        :return: DeclineCurveResult containing decline curve parameters and forecasts.
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        # Collect production rate data over specified time range
        time_steps_list = []
        production_rates_list = []

        for time_step in range(from_time_step, to_time_step + 1):
            instantaneous_rates = self.instantaneous_production_rates(time_step)
            time_steps_list.append(time_step)

            if phase == "oil":
                production_rates_list.append(instantaneous_rates.oil_rate)
            elif phase == "gas":
                production_rates_list.append(instantaneous_rates.gas_rate)
            elif phase == "water":
                production_rates_list.append(instantaneous_rates.water_rate)
            else:
                raise ValueError("Invalid phase specified for decline curve analysis.")

        time_steps_array = np.array(time_steps_list)
        production_rates_array = np.array(production_rates_list)

        # Filter out zero and negative rates for meaningful decline analysis
        positive_production_mask = production_rates_array > 0
        if np.sum(positive_production_mask) < 2:
            return DeclineCurveResult(
                decline_type=decline_type,
                initial_rate=0.0,
                decline_rate_per_year=0.0,
                b_factor=0.0,
                r_squared=0.0,
                phase=phase,
                error=f"Insufficient positive {phase} rate data for analysis",
                time_steps=None,
                actual_rates=None,
                predicted_rates=None,
            )

        filtered_time_steps = time_steps_array[positive_production_mask]
        filtered_production_rates = production_rates_array[positive_production_mask]

        if decline_type == "exponential":
            # Exponential decline: q = qi * exp(-Di*t)
            # Use linear regression on ln(q) vs t to find parameters
            log_production_rates = np.log(filtered_production_rates)
            linear_regression_coefficients = np.polyfit(
                filtered_time_steps, log_production_rates, 1
            )

            exponential_decline_rate_per_timestep = -linear_regression_coefficients[0]
            log_initial_rate_intercept = linear_regression_coefficients[1]
            exponential_initial_production_rate = np.exp(log_initial_rate_intercept)

            # Convert decline rate to annual basis (assuming time steps are days)
            exponential_decline_rate_per_year = (
                exponential_decline_rate_per_timestep * 365.25
            )

            # Calculate coefficient of determination (R²) for goodness of fit
            predicted_exponential_rates = exponential_initial_production_rate * np.exp(
                -exponential_decline_rate_per_timestep * filtered_time_steps
            )
            sum_squared_residuals = np.sum(
                (filtered_production_rates - predicted_exponential_rates) ** 2
            )
            total_sum_squares = np.sum(
                (filtered_production_rates - np.mean(filtered_production_rates)) ** 2
            )
            exponential_r_squared = (
                1 - (sum_squared_residuals / total_sum_squares)
                if total_sum_squares > 0
                else 0.0
            )

            return DeclineCurveResult(
                decline_type="exponential",
                initial_rate=exponential_initial_production_rate,
                decline_rate_per_year=exponential_decline_rate_per_year,
                b_factor=0.0,
                r_squared=exponential_r_squared,
                phase=phase,
                error=None,
                time_steps=filtered_time_steps.tolist(),
                actual_rates=filtered_production_rates.tolist(),
                predicted_rates=predicted_exponential_rates.tolist(),
            )

        elif decline_type == "harmonic":
            # Harmonic decline: q = qi / (1 + Di*t) [special case of hyperbolic with b=1]
            # Use linear regression on 1/q vs t to find parameters
            reciprocal_production_rates = 1.0 / filtered_production_rates
            harmonic_regression_coefficients = np.polyfit(
                filtered_time_steps, reciprocal_production_rates, 1
            )

            harmonic_decline_rate_per_timestep = harmonic_regression_coefficients[0]
            reciprocal_initial_rate_intercept = harmonic_regression_coefficients[1]
            harmonic_initial_production_rate = 1.0 / reciprocal_initial_rate_intercept

            # Convert decline rate to annual basis
            harmonic_decline_rate_per_year = harmonic_decline_rate_per_timestep * 365.25

            # Calculate predicted rates and R²
            predicted_harmonic_rates = harmonic_initial_production_rate / (
                1.0 + harmonic_decline_rate_per_timestep * filtered_time_steps
            )
            harmonic_sum_squared_residuals = np.sum(
                (filtered_production_rates - predicted_harmonic_rates) ** 2
            )
            harmonic_total_sum_squares = np.sum(
                (filtered_production_rates - np.mean(filtered_production_rates)) ** 2
            )
            harmonic_r_squared = (
                1 - (harmonic_sum_squared_residuals / harmonic_total_sum_squares)
                if harmonic_total_sum_squares > 0
                else 0.0
            )

            return DeclineCurveResult(
                decline_type="harmonic",
                initial_rate=harmonic_initial_production_rate,
                decline_rate_per_year=harmonic_decline_rate_per_year,
                b_factor=1.0,  # Harmonic decline has b=1
                r_squared=harmonic_r_squared,
                phase=phase,
                error=None,
                time_steps=filtered_time_steps.tolist(),
                actual_rates=filtered_production_rates.tolist(),
                predicted_rates=predicted_harmonic_rates.tolist(),
            )

        elif decline_type == "hyperbolic":
            # Hyperbolic decline: q = qi / (1 + b*Di*t)^(1/b)
            # This requires non-linear regression using scipy curve_fit
            def hyperbolic_decline_function(
                time_array, initial_rate_param, decline_rate_param, b_factor_param
            ):
                """Hyperbolic decline curve function for curve fitting."""
                return initial_rate_param / (
                    1 + b_factor_param * decline_rate_param * time_array
                ) ** (1 / b_factor_param)

            # Initial parameter estimates
            estimated_initial_rate = filtered_production_rates[0]
            estimated_decline_rate = 0.001  # Small initial estimate
            estimated_b_factor = 0.5  # Typical hyperbolic exponent

            # Perform non-linear curve fitting
            optimized_parameters, parameter_covariance = curve_fit(
                hyperbolic_decline_function,
                filtered_time_steps,
                filtered_production_rates,
                p0=[estimated_initial_rate, estimated_decline_rate, estimated_b_factor],
                bounds=([0, 0, 0.1], [np.inf, 1.0, 2.0]),  # Reasonable parameter bounds
                maxfev=1000,
            )

            (
                hyperbolic_initial_rate,
                hyperbolic_decline_rate_per_timestep,
                hyperbolic_b_factor,
            ) = optimized_parameters

            # Convert decline rate to annual basis
            hyperbolic_decline_rate_per_year = (
                hyperbolic_decline_rate_per_timestep * 365.25
            )

            # Calculate predicted rates and R²
            predicted_hyperbolic_rates = hyperbolic_decline_function(
                filtered_time_steps,
                hyperbolic_initial_rate,
                hyperbolic_decline_rate_per_timestep,
                hyperbolic_b_factor,
            )
            hyperbolic_sum_squared_residuals = np.sum(
                (filtered_production_rates - predicted_hyperbolic_rates) ** 2
            )
            hyperbolic_total_sum_squares = np.sum(
                (filtered_production_rates - np.mean(filtered_production_rates)) ** 2
            )
            hyperbolic_r_squared = (
                1 - (hyperbolic_sum_squared_residuals / hyperbolic_total_sum_squares)
                if hyperbolic_total_sum_squares > 0
                else 0.0
            )

            return DeclineCurveResult(
                decline_type="hyperbolic",
                initial_rate=hyperbolic_initial_rate,
                decline_rate_per_year=hyperbolic_decline_rate_per_year,
                b_factor=hyperbolic_b_factor,
                r_squared=hyperbolic_r_squared,
                phase=phase,
                error=None,
                time_steps=filtered_time_steps.tolist(),
                actual_rates=filtered_production_rates.tolist(),
                predicted_rates=predicted_hyperbolic_rates.tolist(),
            )

        # This should never be reached given the type hints, but included for completeness
        return DeclineCurveResult(
            decline_type=decline_type,
            initial_rate=filtered_production_rates[0]
            if len(filtered_production_rates) > 0
            else 0.0,
            decline_rate_per_year=0.0,
            b_factor=0.0,
            r_squared=0.0,
            phase=phase,
            error=f"Unknown decline type: {decline_type}",
            time_steps=None,
            actual_rates=None,
            predicted_rates=None,
        )

    def reservoir_volumetrics_history(
        self, from_time_step: int = 0, to_time_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, ReservoirVolumetrics], None, None]:
        """
        Generator for reservoir volumetrics history over time.

        :param from_time_step: Starting time step index.
        :param to_time_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (time_step, `ReservoirVolumetrics`) tuples.
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.reservoir_volumetrics_analysis(t))

    def instantaneous_rates_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        rate_type: typing.Literal["production", "injection"] = "production",
    ) -> typing.Generator[typing.Tuple[int, InstantaneousRates], None, None]:
        """
        Generator for instantaneous rates history over time.

        :param from_time_step: Starting time step index.
        :param to_time_step: Ending time step index.
        :param interval: Time step interval.
        :param rate_type: Type of rates ('production' or 'injection').
        :return: Generator yielding (time_step, `InstantaneousRates`) tuples.
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        rate_method = (
            self.instantaneous_production_rates
            if rate_type == "production"
            else self.instantaneous_injection_rates
        )

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, rate_method(t))

    def cumulative_production_history(
        self, from_time_step: int = 0, to_time_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, CumulativeProduction], None, None]:
        """
        Generator for cumulative production history over time.

        :param from_time_step: Starting time step index.
        :param to_time_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (time_step, `CumulativeProduction`) tuples.
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.cumulative_production_analysis(t))

    def material_balance_history(
        self, from_time_step: int = 0, to_time_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, MaterialBalanceAnalysis], None, None]:
        """
        Generator for material balance analysis history over time.

        :param from_time_step: Starting time step index.
        :param to_time_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (time_step, `MaterialBalanceAnalysis`) tuples.
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.material_balance_analysis(t))

    def sweep_efficiency_history(
        self, from_time_step: int = 0, to_time_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, SweepEfficiencyAnalysis], None, None]:
        """
        Generator for sweep efficiency analysis history over time.

        :param from_time_step: Starting time step index.
        :param to_time_step: Ending time step index.
        :param interval: Time step interval.
        :return: Generator yielding (time_step, `SweepEfficiencyAnalysis`) tuples.
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.sweep_efficiency_analysis(t))

    def productivity_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        ipr_method: typing.Literal["vogel", "linear", "fetkovich", "jones"] = "vogel",
        phase: typing.Literal["oil", "gas"] = "oil",
    ) -> typing.Generator[typing.Tuple[int, ProductivityAnalysis], None, None]:
        """
        Generator for productivity analysis history over time with selectable IPR method.

        :param from_time_step: Starting time step index.
        :param to_time_step: Ending time step index.
        :param interval: Time step interval.
        :param ipr_method: IPR correlation method ('vogel', 'linear', 'fetkovich', 'jones').
        :return: Generator yielding (time_step, `ProductivityAnalysis`) tuples.
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.productivity_analysis(t, ipr_method=ipr_method, phase=phase))

    def recommend_ipr_method(
        self, time_step: int = -1
    ) -> typing.Literal["vogel", "linear", "fetkovich", "jones"]:
        """
        Recommend the most appropriate IPR method based on reservoir conditions.

        This method analyzes the current reservoir state and suggests the most
        suitable IPR correlation based on fluid properties and well conditions.

        :param time_step: The time step to analyze for IPR method recommendation
        :return: Recommended IPR method
        """
        state = self.get_state(time_step)
        oil_saturation = np.nanmean(state.model.fluid_properties.oil_saturation_grid)
        gas_saturation = np.nanmean(state.model.fluid_properties.gas_saturation_grid)
        reservoir_pressure = np.nanmean(state.model.fluid_properties.pressure_grid)

        estimated_bubble_point = np.nanmean(
            state.model.fluid_properties.oil_bubble_point_pressure_grid
        )

        # Check if this is primarily a gas reservoir
        if gas_saturation >= 0.6:
            return "fetkovich"  # Best for gas wells

        # Check if we're above bubble point (single-phase oil)
        elif reservoir_pressure > estimated_bubble_point and oil_saturation > 0.7:
            return "linear"  # Best for undersaturated oil

        # Check if we have significant multi-phase flow
        elif oil_saturation > 0.3 and gas_saturation > 0.2:
            return "jones"  # Best for complex multi-phase systems

        # Default to Vogel for solution gas drive reservoirs
        return "vogel"  # Best for two-phase oil/gas systems

    def compare_ipr_methods(
        self, time_step: int = -1
    ) -> typing.Dict[str, ProductivityAnalysis]:
        """
        Compare all available IPR methods for the same reservoir conditions.

        This method runs productivity analysis using all four IPR methods and
        returns the results for comparison. Useful for sensitivity analysis
        and method validation.

        :param time_step: The time step to analyze
        :return: Mapping of IPR method names to their analysis results
        """
        methods: typing.List[str] = ["vogel", "linear", "fetkovich", "jones"]
        results = {}
        for method in methods:
            results[method] = self.productivity_analysis(
                time_step=time_step,
                ipr_method=method,  # type: ignore
            )
        return results
