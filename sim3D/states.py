import functools
import typing
import logging

import attrs
import numpy as np
from scipy.optimize import curve_fit

from sim3D.constants import c
from sim3D.grids import uniform_grid
from sim3D.properties import compute_hydrocarbon_in_place
from sim3D.models import ReservoirModel
from sim3D.types import (
    NDimension,
    RateGrids,
    RelPermGrids,
    RelativeMobilityGrids,
    CapillaryPressureGrids,
)
from sim3D.wells import Wells, _expand_intervals

logger = logging.getLogger(__name__)


__all__ = ["ModelState", "ProductionAnalyst"]


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
        return self.time_step * self.time_step_size


@attrs.define(frozen=True, slots=True)
class ReservoirVolumetrics:
    """Reservoir volumetrics analysis results."""

    oil_in_place: float
    """Total oil in place in stock tank barrels (STB)."""
    gas_in_place: float
    """Total gas in place in standard cubic feet (SCF)."""
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
    """Gas production/injection rate in standard cubic feet per day (SCF/day)."""
    water_rate: float
    """Water production/injection rate in stock tank barrels per day (STB/day)."""
    total_liquid_rate: float
    """Total liquid (oil + water) rate in stock tank barrels per day (STB/day)."""
    gas_oil_ratio: float
    """Gas-oil ratio in standard cubic feet per stock tank barrel (SCF/STB)."""
    water_cut_fraction: float
    """Water cut as a fraction (0 to 1) of total liquid production."""


@attrs.define(frozen=True, slots=True)
class CumulativeProduction:
    """Cumulative production analysis results."""

    cumulative_oil: float
    """Cumulative oil produced in stock tank barrels (STB)."""
    cumulative_free_gas: float
    """Cumulative free gas produced in standard cubic feet (SCF)."""
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
    """Initial production rate in stock tank barrels per day (STB/day) for oil/water or standard cubic feet per day (SCF/day) for gas."""
    decline_rate_per_timestep: float
    """Decline rate per time step as a fraction."""
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
    """Actual production rates in STB/day or SCF/day depending on phase."""
    predicted_rates: typing.Optional[typing.List[float]] = None
    """Predicted production rates from decline curve in STB/day or SCF/day depending on phase."""


hcip_vectorized = np.vectorize(
    compute_hydrocarbon_in_place, excluded=["hydrocarbon_type"], cache=True
)


class ProductionAnalyst(typing.Generic[NDimension]):
    """
    Production analysis for evaluating reservoir performance over a series of model states.
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
        """The stock tank gas initially in place (STGIIP) at the start of the simulation in standard cubic feet (SCF)."""
        return self.gas_in_place(0)

    stgiip = stock_tank_gas_initially_in_place
    """The stock tank gas initially in place (STGIIP) in standard cubic feet (SCF)."""

    @property
    def stock_tank_water_initially_in_place(self) -> float:
        """The stock tank water initially in place at the start of the simulation in stock tank barrels (STB)."""
        return self.water_in_place(0)

    @property
    def cumulative_oil_produced(self) -> float:
        """The cumulative oil produced in stock tank barrels (STB) from the start of the simulation to the current time step."""
        return self.oil_produced(0, -1)

    No = cumulative_oil_produced
    """Cumulative oil produced in stock tank barrels (STB)."""

    @property
    def cumulative_free_gas_produced(self) -> float:
        """Return the cumulative gas produced in standard cubic feet (SCF)."""
        return self.free_gas_produced(0, -1)

    Ng = cumulative_free_gas_produced
    """Cumulative gas produced in standard cubic feet (SCF)."""

    @property
    def cumulative_water_produced(self) -> float:
        """Return the cumulative water produced in stock tank barrels (STB)."""
        return self.water_produced(0, -1)

    Nw = cumulative_water_produced
    """Cumulative water produced in stock tank barrels (STB)."""

    @property
    def oil_recovery_factor(self) -> float:
        """
        The oil recovery factor based on initial oil in place and cumulative oil produced
        over the entire simulation period.

        Oil Recovery Factor (RF) = Cumulative Oil Produced / Stock Tank Oil Initially in Place

        This is the fundamental measure of reservoir performance, representing the fraction
        of original oil in place that has been recovered. The recovery factor depends on:

        Primary Recovery Mechanisms:
            - Solution Gas Drive: 5-30% RF (gas expansion as pressure drops)
            - Gas Cap Drive: 20-40% RF (gas cap expansion displaces oil)
            - Natural Water Drive: 35-75% RF (aquifer influx maintains pressure)
            - Gravity Drainage: 40-80% RF (oil drains to bottom under gravity)
            - Combination Drive: 25-50% RF (multiple mechanisms)

        Secondary Recovery (Pressure Maintenance):
            - Water Flooding: 30-60% RF (injected water displaces oil)
            - Gas Injection: 40-70% RF (gas maintains pressure, displaces oil)

        Enhanced Oil Recovery (EOR):
            - Chemical Flooding: 50-70% RF (surfactants reduce interfacial tension)
            - Thermal Recovery: 50-80% RF (steam/combustion reduces viscosity)
            - CO2 Flooding: 40-60% RF (miscible displacement)

        Factors Affecting Recovery:
            - Reservoir properties: porosity, permeability, heterogeneity
            - Fluid properties: viscosity, density, API gravity
            - Drive mechanism: water drive > gas cap > solution gas
            - Well spacing and placement
            - Operating conditions: production rates, pressures
            - Sweep efficiency: areal, vertical, and displacement
            - Residual oil saturation

        Units:
            Numerator: Cumulative oil produced in STB (stock tank barrels)
            Denominator: Initial oil in place in STB (stock tank barrels)
            Result: Dimensionless fraction (0.0 to 1.0, or 0% to 100%)

        Calculation Details:
            STOIIP = Σ [Area x h x φ x (1-Swi) x NTG / Boi]
            Np = Σ [q_oil x Δt / Bo]
            RF = Np / STOIIP

        Where:
            Area = Grid cell area (acres)
            h = Net pay thickness (ft)
            φ = Porosity (fraction)
            Swi = Initial water saturation (fraction)
            NTG = Net-to-gross ratio (fraction)
            Boi = Initial oil formation volume factor (bbl/STB)
            q_oil = Oil production rate (STB/day)
            Δt = Time step (days)
            Bo = Oil formation volume factor (bbl/STB)

        :return: The oil recovery factor as a fraction (0 to 1)

        Example:
        ```python
        analyst = Analyst(states)
        rf = analyst.oil_recovery_factor
        print(f"Oil Recovery Factor: {rf:.2%}")
        # Oil Recovery Factor: 32.50%

        stoiip = analyst.stoiip
        cumulative = analyst.cumulative_oil_produced
        remaining = stoiip - cumulative
        print(f"Recovered: {cumulative:,.0f} STB ({rf:.1%})")
        print(f"Remaining: {remaining:,.0f} STB ({1-rf:.1%})")
        # Recovered: 1,250,000 STB (32.5%)
        # Remaining: 2,596,154 STB (67.5%)
        ```

        Notes:
            - Recovery factor increases monotonically over production life
            - Ultimate recovery factor (URF) is RF at economic abandonment
            - Low RF (<20%) suggests poor drive mechanism or early production stage
            - High RF (>50%) indicates good drive mechanism or enhanced recovery
            - For mature fields, RF typically plateaus unless EOR is applied
            - Compare RF to material balance drive indices to understand mechanism
            - Use sweep efficiency analysis to identify uncontacted oil
        """
        if self.stock_tank_oil_initially_in_place == 0:
            return 0.0
        return self.cumulative_oil_produced / self.stock_tank_oil_initially_in_place

    @property
    def free_gas_recovery_factor(self) -> float:
        """
        The recovery factor for free gas only, based on initial free gas in place
        and cumulative free gas produced over the entire simulation period.

        This recovery factor specifically tracks depletion of the free gas phase
        (gas cap or gas reservoir) and does not include solution gas.

        Free Gas Recovery Factor = Free Gas Produced / Initial Free Gas in Place

        Typical Values:
            - Gas reservoirs: 60-90% (high recovery due to expansion drive)
            - Gas caps: 40-70% (depends on pressure maintenance)
            - Dry gas fields: 70-95% (highest recovery)
            - Wet gas/condensate: 50-80% (liquid dropout reduces recovery)

        Use this metric for:
            - Gas cap drive reservoirs
            - Pure gas reservoirs
            - Tracking gas cap depletion separately from solution gas
            - Comparing with initial free gas estimates

        :return: The free gas recovery factor as a fraction (0 to 1)
        """
        if self.stock_tank_gas_initially_in_place == 0:
            return 0.0
        return (
            self.cumulative_free_gas_produced / self.stock_tank_gas_initially_in_place
        )

    @property
    def total_gas_recovery_factor(self) -> float:
        """
        The total gas recovery factor based on all initial gas (free + solution)
        and all cumulative gas produced over the entire simulation period.

        This comprehensive recovery factor accounts for:
            1. Free gas produced from gas cap or gas phase
            2. Solution gas produced (dissolved gas liberated from oil production)

        Total Gas Recovery Factor = (Free Gas + Solution Gas Produced) /
                                    (Initial Free Gas + Initial Solution Gas)

        Calculation:
            Initial Total Gas = GIIP_free + (STOIIP x Rs_initial)
            Produced Total Gas = G_produced_free + (N_produced x Rs_avg)

        Where:
            GIIP_free = Initial free gas in place (scf)
            STOIIP = Stock tank oil initially in place (STB)
            Rs_initial = Initial solution gas-oil ratio (scf/STB)
            G_produced_free = Cumulative free gas produced (scf)
            N_produced = Cumulative oil produced (STB)
            Rs_avg = Average solution GOR during production (scf/STB)

        Typical Values:
            - Solution gas drive oil reservoirs: 15-30% total gas recovery
            - Gas cap drive oil reservoirs: 30-50% total gas recovery
            - Pure gas reservoirs: 60-90% total gas recovery
            - Combination drive: 25-45% total gas recovery

        Use this metric for:
            - Oil reservoirs with significant solution gas
            - Combined gas cap and solution gas drive
            - Total gas resource recovery assessment
            - Gas sales and revenue calculations
            - Material balance validation

        Note:
            For pure gas reservoirs with minimal oil, this equals free_gas_recovery_factor.
            For oil reservoirs, this provides a complete picture of gas recovery from all sources.

        :return: The total gas recovery factor as a fraction (0 to 1)
        """
        # Get initial free gas in place
        initial_free_gas = self.stock_tank_gas_initially_in_place  # scf

        # Get initial solution gas in oil
        initial_state = self.get_state(0)
        avg_initial_gor = np.nanmean(
            initial_state.model.fluid_properties.solution_gas_to_oil_ratio_grid
        )  # scf/STB
        initial_oil = self.stock_tank_oil_initially_in_place  # STB
        initial_solution_gas = initial_oil * avg_initial_gor  # scf

        # Total initial gas
        total_initial_gas = initial_free_gas + initial_solution_gas

        if total_initial_gas == 0:
            return 0.0

        # Get cumulative free gas produced
        cumulative_free_gas = self.cumulative_free_gas_produced  # scf (free gas only)

        # Get cumulative solution gas produced
        cumulative_oil = self.cumulative_oil_produced  # STB
        cumulative_solution_gas = cumulative_oil * avg_initial_gor  # scf

        # Total gas produced
        total_gas_produced = cumulative_free_gas + cumulative_solution_gas
        return float(total_gas_produced / total_initial_gas)

    @property
    def gas_recovery_factor(self) -> float:
        """
        The recovery factor based on initial free gas in place and cumulative free gas produced
        over the entire simulation period.

        Note: This only considers free gas. For total gas recovery including solution gas,
        use total_gas_recovery_factor property.

        :return: The free gas recovery factor as a fraction (0 to 1)
        """
        return self.free_gas_recovery_factor

    @functools.cache
    def _get_cell_area_in_acres(self, x_dim: float, y_dim: float) -> float:
        """
        Computes the area of a grid cell in acres.

        :param x_dim: The dimension of the cell in the x-direction (ft).
        :param y_dim: The dimension of the cell in the y-direction (ft).
        :return: The area of the cell in acres.
        """
        cell_area_in_ft2 = x_dim * y_dim
        return cell_area_in_ft2 * c.FT2_TO_ACRES

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
    def oil_produced(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative oil produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - `from_time_step` equals `to_time_step`, the production at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative production is returned.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative oil produced in STB
        """
        total_production = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Production is in ft³/day, convert to STB using FVF
            oil_production = state.production.oil
            time_step_in_days = state.time_step_size / c.SECONDS_PER_DAY
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid

            time_step_production = 0.0
            if oil_production is not None:
                oil_production_stb_day = oil_production * c.FT3_TO_BBL / oil_fvf_grid
                time_step_production += np.nansum(
                    oil_production_stb_day * time_step_in_days
                )

            total_production += time_step_production
        return float(total_production)

    @functools.cache
    def free_gas_produced(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative free gas produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - `from_time_step` equals `to_time_step`, the production at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative production is returned.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative gas produced in SCF
        """
        total_production = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Production is in ft³/day, convert to SCF using FVF
            gas_production = state.production.gas
            time_step_in_days = state.time_step_size / c.SECONDS_PER_DAY
            gas_fvf_grid = state.model.fluid_properties.gas_formation_volume_factor_grid

            time_step_production = 0.0
            if gas_production is not None:
                gas_production_SCF_day = gas_production / gas_fvf_grid
                time_step_production += np.nansum(
                    gas_production_SCF_day * time_step_in_days
                )

            total_production += time_step_production
        return float(total_production)

    @functools.cache
    def water_produced(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative water produced between two time steps.

        If:
        - production rates are present, they contribute positively to production.
        - `from_time_step` equals `to_time_step`, the production at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative production is returned.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative water produced in STB
        """
        total_production = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Production is in ft³/day, convert to STB using FVF
            water_production = state.production.water
            time_step_in_days = state.time_step_size / c.SECONDS_PER_DAY
            water_fvf_grid = (
                state.model.fluid_properties.water_formation_volume_factor_grid
            )

            time_step_production = 0.0
            if water_production is not None:
                water_production_stb_day = (
                    water_production * c.FT3_TO_BBL / water_fvf_grid
                )
                time_step_production += np.nansum(
                    water_production_stb_day * time_step_in_days
                )

            total_production += time_step_production
        return float(total_production)

    def oil_injected(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative oil injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - `from_time_step` equals `to_time_step`, the injection at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative injection is returned.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative oil injected in STB
        """
        total_injection = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Injection is in ft³/day, convert to STB using FVF
            oil_injection = state.injection.oil
            time_step_in_days = state.time_step_size / c.SECONDS_PER_DAY
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid
            time_step_injection = 0.0
            if oil_injection is not None:
                oil_injection_stb_day = oil_injection * c.FT3_TO_BBL / oil_fvf_grid
                time_step_injection += np.nansum(
                    oil_injection_stb_day * time_step_in_days
                )
            total_injection += time_step_injection
        return float(total_injection)

    def gas_injected(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative gas injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - `from_time_step` equals `to_time_step`, the injection at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative injection is returned.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative gas injected in SCF
        """
        total_injection = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Injection is in ft³/day, convert to SCF using FVF
            gas_injection = state.injection.gas
            time_step_in_days = state.time_step_size / c.SECONDS_PER_DAY
            gas_fvf_grid = state.model.fluid_properties.gas_formation_volume_factor_grid
            time_step_injection = 0.0
            if gas_injection is not None:
                gas_injection_SCF_day = gas_injection / gas_fvf_grid
                time_step_injection += np.nansum(
                    gas_injection_SCF_day * time_step_in_days
                )
            total_injection += time_step_injection
        return float(total_injection)

    def water_injected(self, from_time_step: int, to_time_step: int) -> float:
        """
        Computes the cumulative water injected between two time steps.

        If:
        - injection rates are present, they contribute positively to injection.
        - `from_time_step` equals `to_time_step`, the injection at that time step is returned.
        - `from_time_step` is 0 and `to_time_step` is -1, the total cumulative injection is returned.

        :param from_time_step: The starting time step index (inclusive).
        :param to_time_step: The ending time step index (inclusive).
        :return: The cumulative water injected in STB
        """
        total_injection = 0.0
        for t in range(from_time_step, to_time_step + 1):
            state = self.get_state(t)
            # Injection is in ft³/day, convert to STB using FVF
            water_injection = state.injection.water
            time_step_in_days = state.time_step_size / c.SECONDS_PER_DAY
            water_fvf_grid = (
                state.model.fluid_properties.water_formation_volume_factor_grid
            )
            time_step_injection = 0.0
            if water_injection is not None:
                water_injection_stb_day = (
                    water_injection * c.FT3_TO_BBL / water_fvf_grid
                )
                time_step_injection += np.nansum(
                    water_injection_stb_day * time_step_in_days
                )
            total_injection += time_step_injection
        return float(total_injection)

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
                yield (t, self.oil_produced(0, t))
            else:
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.oil_produced(t, t))

    def free_gas_production_history(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        interval: int = 1,
        cumulative: bool = False,
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Get the free gas production history between two time steps.

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
                yield (t, self.free_gas_produced(0, t))
            else:
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.free_gas_produced(t, t))

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
                yield (t, self.water_produced(0, t))
            else:
                # Calculate production at time step t (exclusive)
                # Use time step t for both from and to to get production at that step
                yield (t, self.water_produced(t, t))

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
                yield (t, self.oil_injected(0, t))
            else:
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.oil_injected(t, t))

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
                yield (t, self.gas_injected(0, t))
            else:
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.gas_injected(t, t))

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
                yield (t, self.water_injected(0, t))
            else:
                # Calculate injection at time step t (exclusive)
                # Use time step t for both from and to to get injection at that step
                yield (t, self.water_injected(t, t))

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
            * c.ACRE_FT_TO_FT3  # Convert acre-ft to ft³
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
        if (oil_production := state.production.oil) is not None:
            # Convert from ft³/day to STB/day using oil FVF
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid
            oil_rate = np.nansum(oil_production * c.FT3_TO_BBL / oil_fvf_grid)

        if (gas_production := state.production.gas) is not None:
            # Convert from ft³/day to SCF/day using gas FVF
            gas_fvf_grid = state.model.fluid_properties.gas_formation_volume_factor_grid
            gas_rate = np.nansum(gas_production / gas_fvf_grid)

        if (water_production := state.production.water) is not None:
            # Convert from ft³/day to STB/day using water FVF
            water_fvf_grid = (
                state.model.fluid_properties.water_formation_volume_factor_grid
            )
            water_rate = np.nansum(water_production * c.FT3_TO_BBL / water_fvf_grid)

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
        if (oil_injection := state.injection.oil) is not None:
            # Convert from ft³/day to STB/day using oil FVF
            oil_fvf_grid = state.model.fluid_properties.oil_formation_volume_factor_grid
            oil_rate = np.nansum(oil_injection * c.FT3_TO_BBL / oil_fvf_grid)

        if (gas_injection := state.injection.gas) is not None:
            # Convert from ft³/day to SCF/day using gas FVF
            gas_fvf_grid = state.model.fluid_properties.gas_formation_volume_factor_grid
            gas_rate = np.nansum(gas_injection / gas_fvf_grid)

        if (water_injection := state.injection.water) is not None:
            # Convert from ft³/day to STB/day using water FVF
            water_fvf_grid = (
                state.model.fluid_properties.water_formation_volume_factor_grid
            )
            water_rate = np.nansum(water_injection * c.FT3_TO_BBL / water_fvf_grid)

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
        cumulative_oil = self.oil_produced(0, time_step)
        cumulative_free_gas = self.free_gas_produced(0, time_step)
        cumulative_water = self.water_produced(0, time_step)
        return CumulativeProduction(
            cumulative_oil=cumulative_oil,
            cumulative_free_gas=cumulative_free_gas,
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
        - G = Initial gas in place (SCF)
        - W = Initial water in place (STB)
        - Boi, Bo = Initial and current oil formation volume factors (bbl/STB)
        - Rsi, Rs = Initial and current solution gas-oil ratios (SCF/STB)
        - Bg, Bgi = Current and initial gas formation volume factors (bbl/SCF)
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
        current_gor = np.nanmean(
            state.model.fluid_properties.solution_gas_to_oil_ratio_grid
        )
        initial_gor = np.nanmean(
            initial_state.model.fluid_properties.solution_gas_to_oil_ratio_grid
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
        cumulative_oil = self.oil_produced(0, time_step)
        cumulative_water = self.water_produced(0, time_step)

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
            * c.ACRE_FT_TO_FT3
        )
        compressibility_expansion = (
            pore_volume * total_compressibility * pressure_decline * c.FT3_TO_BBL
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
            * c.ACRE_FT_TO_FT3
            / c.FT3_TO_BBL  # Convert to STB
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
                        state.production.oil[i, j, k] * c.FT3_TO_BBL / oil_fvf
                    )  # stb/day
                else:
                    if state.production.gas is None:
                        continue
                    gas_fvf = float(
                        state.model.fluid_properties.gas_formation_volume_factor_grid[
                            i, j, k
                        ]
                    )
                    cell_flow_rate = state.production.gas[i, j, k] / gas_fvf  # SCF/day

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
            gor = float(
                state.model.fluid_properties.solution_gas_to_oil_ratio_grid[i, j, k]
            )
        else:
            # Use reservoir averages as fallback
            gas_saturation = np.nanmean(
                state.model.fluid_properties.gas_saturation_grid
            )
            oil_saturation = np.nanmean(
                state.model.fluid_properties.oil_saturation_grid
            )
            gor = np.nanmean(
                state.model.fluid_properties.solution_gas_to_oil_ratio_grid
            )

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

    def voidage_replacement_ratio(self, time_step: int = -1) -> float:
        """
        Calculates the voidage replacement ratio (VRR) at a specific time step.

        The VRR measures the ratio of injected reservoir volumes to produced reservoir volumes:
        VRR = (Injected volumes) / (Produced volumes)

        VRR = (Wi*Bwi + Ggi*Bginj) / (Np*Bo + Wp*Bwp + Bg*(GOR - Rs)*Np)

        Where:
        - Wi = Water injected (m3 or STB)
        - Bwi = Injected water formation volume factor (m3/m3 or bbl/STB)
        - Ggi = Injected gas volume (m3 or SCF)
        - Bginj = Injected gas formation volume factor at reservoir pressure (m3/m3 or bbl/SCF)
        - Np = Oil produced (m3 or STB)
        - Bo = Oil formation volume factor (m3/m3 or bbl/STB)
        - Wp = Water produced (m3 or STB)
        - Bwp = Produced water formation volume factor (m3/m3 or bbl/STB)
        - Bg = Gas formation volume factor (m3/m3 or bbl/SCF)
        - GOR = Produced gas oil ratio (m3/m3 or SCF/STB)
        - Rs = Solution gas oil ratio (m3/m3 or SCF/STB)

        A VRR > 1.0 indicates pressure maintenance through injection.
        A VRR = 1.0 indicates a balanced reservoir.
        A VRR < 1.0 indicates reservoir pressure is declining.

        :param time_step: The time step index to calculate VRR for.
        :return: The voidage replacement ratio (dimensionless fraction).
        """
        state = self.get_state(time_step)

        # Get cumulative injection volumes
        cumulative_water_injected = self.water_injected(0, time_step)  # STB
        cumulative_gas_injected = self.gas_injected(0, time_step)  # SCF (free gas only)

        # Get cumulative production volumes
        cumulative_oil_produced = self.oil_produced(0, time_step)  # STB
        cumulative_water_produced = self.water_produced(0, time_step)  # STB
        cumulative_free_gas_produced = self.free_gas_produced(
            0, time_step
        )  # SCF (free gas only)

        # Get average formation volume factors at current reservoir conditions
        avg_oil_fvf = np.nanmean(
            state.model.fluid_properties.oil_formation_volume_factor_grid
        )  # bbl/STB
        avg_gas_fvf = np.nanmean(
            state.model.fluid_properties.gas_formation_volume_factor_grid
        )  # bbl/SCF
        avg_water_fvf_produced = np.nanmean(
            state.model.fluid_properties.water_formation_volume_factor_grid
        )  # bbl/STB
        avg_water_fvf_injected = avg_water_fvf_produced  # Typically same as produced

        # Get gas formation volume factor for injected gas at reservoir pressure
        avg_gas_fvf_injected = avg_gas_fvf  # Injected gas FVF at reservoir conditions

        # Get average solution GOR (Rs)
        avg_solution_gor = np.nanmean(
            state.model.fluid_properties.solution_gas_to_oil_ratio_grid
        )  # SCF/STB (solution GOR)

        # Calculate injected reservoir volumes (numerator)
        injected_water_reservoir_volume = (
            cumulative_water_injected * avg_water_fvf_injected
        )  # bbl
        injected_gas_reservoir_volume = (
            cumulative_gas_injected * avg_gas_fvf_injected
        )  # bbl
        total_injected_volume = (
            injected_water_reservoir_volume + injected_gas_reservoir_volume
        )

        # Calculate produced reservoir volumes (denominator)
        produced_oil_reservoir_volume = cumulative_oil_produced * avg_oil_fvf  # bbl
        produced_water_reservoir_volume = (
            cumulative_water_produced * avg_water_fvf_produced
        )  # bbl

        # Free gas produced (already free gas only from free_gas_produced)
        # Plus solution gas that came out of the oil
        solution_free_gas_produced = cumulative_oil_produced * avg_solution_gor  # SCF
        total_free_gas_produced = (
            cumulative_free_gas_produced + solution_free_gas_produced
        )  # SCF

        # Calculate produced GOR for the equation
        produced_gor = (
            total_free_gas_produced / cumulative_oil_produced
            if cumulative_oil_produced > 0
            else 0.0
        )  # SCF/STB

        # Free gas component in produced volumes: Bg * (GOR - Rs) * Np
        free_gas_component = (
            produced_gor - avg_solution_gor
        ) * cumulative_oil_produced  # SCF
        produced_gas_reservoir_volume = free_gas_component * avg_gas_fvf  # bbl

        total_produced_volume = (
            produced_oil_reservoir_volume
            + produced_water_reservoir_volume
            + produced_gas_reservoir_volume
        )

        # Calculate VRR
        if total_produced_volume <= 0:
            return 0.0

        voidage_replacement_ratio = total_injected_volume / total_produced_volume
        return float(voidage_replacement_ratio)

    def recommend_decline_model(
        self,
        from_time_step: int = 0,
        to_time_step: int = -1,
        phase: typing.Literal["oil", "gas", "water"] = "oil",
    ) -> typing.Tuple[str, typing.Dict[str, DeclineCurveResult]]:
        """
        Automatically recommend the best decline curve model based on statistical fit and physical constraints.

        This method fits all three decline curve models (exponential, harmonic, and hyperbolic) to the
        historical production data and recommends the most appropriate model based on multiple criteria:

        1. Statistical Fit Quality (R² coefficient of determination)
        2. Physical Reasonableness of Parameters
        3. Standard Industry Guidelines

        Model Selection Criteria:

        Exponential Decline - Best For:
        - Wells in boundary-dominated flow (late-time production)
        - Wells with relatively constant pressure drawdown
        - Single-phase flow above bubble point
        - Conservative reserves estimates
        - Short production history where flow regime is uncertain

        Characteristics:
        - Constant percentage decline rate
        - Most conservative EUR estimates
        - Simple to extrapolate and understand
        - Commonly required for SEC reserves reporting

        Equation: q(t) = qi x exp(-Di x t)

        Harmonic Decline - Best For:
        - Wells with linear flow geometry (fractured wells, channel reservoirs)
        - Wells experiencing stable water influx
        - Gas wells in steady-state flow
        - Certain infinite-acting reservoirs

        Characteristics:
        - Decline rate decreases inversely with time
        - Can overestimate EUR if misapplied
        - Special case of hyperbolic with b = 1

        Equation: q(t) = qi / (1 + Di x t)

        Hyperbolic Decline - Best For:
        - Wells transitioning between flow regimes
        - Multi-phase flow systems
        - Wells with changing flowing conditions
        - Modern unconventional wells (shale, tight formations)
        - Wells with sufficient production history (>1-2 years)

        Characteristics:
        - Most flexible model with b-factor controlling curvature
        - 0 < b < 1: Most common for conventional oil wells
        - b ≈ 0.5: Typical for many hydraulically fractured wells
        - Requires more data for reliable parameter estimation

        Equation: q(t) = qi / (1 + b x Di x t)^(1/b)

        Parameter Validation Rules:

        Decline Rate (Di):
        - Valid range: 0.0 to 2.0 per year (0% to 200% annual decline)
        - Typical ranges:
          * Conventional oil wells: 5-25% per year (0.05-0.25)
          * Tight oil/shale: 30-80% first year (0.3-0.8)
          * Gas wells: 10-40% per year (0.1-0.4)
          * Mature fields: <10% per year (<0.1)
        - Models with Di outside valid range are rejected

        Hyperbolic Exponent (b):
        - Valid range: 0.0 to 2.0 (dimensionless)
        - Physical interpretation:
          * b = 0: Exponential decline
          * 0 < b < 0.3: Slightly curved, approaching exponential
          * 0.3 < b < 0.7: Moderate curvature, common for fractured wells
          * 0.7 < b < 1.0: Strong curvature, approaching harmonic
          * b = 1.0: Harmonic decline
          * b > 1.0: Very slow decline, often unrealistic for long-term forecast
        - SEC guidelines: b > 1.0 often requires justification
        - Models with b outside 0.0-2.0 range are rejected

        R² (Coefficient of Determination):
        - Measures goodness of fit: R² = 1 - (SS_residual / SS_total)
        - R² ranges from 0.0 to 1.0:
          * R² > 0.95: Excellent fit
          * 0.90 < R² < 0.95: Good fit
          * 0.80 < R² < 0.90: Acceptable fit
          * R² < 0.80: Poor fit, model may not be appropriate
        - Higher R² is preferred, but must also pass physical constraints

        Selection Algorithm:
        1. Fit all three models to historical data
        2. Eliminate models with physically unreasonable parameters
        3. Among remaining models, select the one with highest R²
        4. If all models fail validation, return "exponential" as conservative default

        Minimum Data Requirements:
        - Exponential: Minimum 6-12 months (2-4 data points)
        - Harmonic: Minimum 12-18 months (4-6 data points)
        - Hyperbolic: Minimum 18-24 months (6-8 data points) for reliable b-factor

        :param from_time_step: Starting time step index for analysis (inclusive).
            Should include stable production period, avoiding
            initial transient flow or well cleanup period.
        :param to_time_step: Ending time step index for analysis (inclusive).
            Use -1 for most recent time step. More data generally
            improves fit quality, but ensure data represents similar
            operating conditions.
        :param phase: Phase to analyze - "oil", "gas", or "water".
            Different phases may have different optimal decline models.
            Gas typically has steeper decline than oil.
        :return: Tuple containing:
            1. str: Recommended model name ("exponential", "harmonic", or "hyperbolic")
            2. dict: Dictionary with all three model results for comparison
                Keys: "exponential", "harmonic", "hyperbolic"
                Values: DeclineCurveResult objects with parameters and fit statistics

        Example:
        ```python
        # Get recommendation and all model results
        recommended, all_models = analyst.recommend_decline_model(
            from_time_step=0,
            to_time_step=-1,
            phase="oil"
        )
        print(f"Recommended model: {recommended}")
        print(f"R² = {all_models[recommended].r_squared:.4f}")
        print(f"Di = {all_models[recommended].decline_rate_per_timestep:.3f} per year")
        if recommended == "hyperbolic":
            print(f"b = {all_models[recommended].b_factor:.3f}")

        # Compare EUR estimates from all models
        for model_name, result in all_models.items():
            if not result.error:
                eur = analyst.estimate_economic_ultimate_recovery(result, forecast_time_steps=30)
                print(f"{model_name}: EUR = {eur:,.0f} STB")

        # Use recommended model for forecasting
        best_model = all_models[recommended]
        forecast = analyst.forecast_production(best_model, time_steps=365*10)
        ```

        Notes:
            - Always review all three model fits visually before accepting recommendation
            - High R² does not guarantee accurate long-term forecasts
            - Consider using multiple models for P10/P50/P90 reserve scenarios:
              * P90 (conservative): Exponential or steepest decline
              * P50 (best estimate): Recommended model
              * P10 (optimistic): Harmonic or shallowest decline
            - For regulatory reporting (SEC), exponential is often required regardless of fit
            - Hyperbolic decline should transition to exponential for long-term forecasts
              (typically after 3-5 years or when bxDixt > 0.5)
            - Be cautious with harmonic decline (b=1.0) as it may overestimate reserves
            - If well has undergone stimulation, workover, or facility changes during
              history period, consider analyzing pre- and post-change periods separately
            - Water production typically follows different decline patterns than oil/gas
            - For unconventional wells, hyperbolic is usually most appropriate for
              first 2-5 years, then transition to exponential

        Industry Best Practices:
            - SPE: Use hyperbolic to exponential transition for proved reserves
            - SEC: Often requires exponential for proved undeveloped reserves
            - Operators: Use model that best matches analog well performance
            - Always perform sensitivity analysis on decline parameters
            - Update decline curves annually as new production data becomes available
        """
        models = ["exponential", "hyperbolic", "harmonic"]
        results: typing.Dict[str, DeclineCurveResult] = {}

        for model in models:
            result = self.decline_curve_analysis(
                from_time_step=from_time_step,
                to_time_step=to_time_step,
                decline_type=model,  # type: ignore
                phase=phase,
            )
            results[model] = result

        # Get time step size to convert decline rate bounds
        state = self.get_state(from_time_step)
        time_step_size_seconds = state.time_step_size
        timesteps_per_year = c.SECONDS_PER_YEAR / time_step_size_seconds

        # Convert typical decline rate bounds from per-year to per-timestep
        max_decline_per_year = 2.0  # 0-200% per year
        max_decline_per_timestep = max_decline_per_year / timesteps_per_year

        # Recommend based on R² and physical reasonableness
        best_model = "exponential"
        best_r_squared = 0.0

        for model, result in results.items():
            if result.error:
                continue

            # Check for physically reasonable parameters (adjusted for timestep units)
            if (
                result.decline_rate_per_timestep < 0
                or result.decline_rate_per_timestep > max_decline_per_timestep
            ):
                continue

            if model == "hyperbolic" and (result.b_factor < 0 or result.b_factor > 2.0):
                continue

            if result.r_squared > best_r_squared:
                best_r_squared = result.r_squared
                best_model = model

        return best_model, results

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
        Fits a decline curve model to production data over a specified time range.

        Performs decline curve analysis using exponential, hyperbolic, or harmonic models:
        - Exponential: q = qi * exp(-Di * t)
        - Hyperbolic: q = qi / (1 + b * Di * t)^(1/b)
        - Harmonic: q = qi / (1 + Di * t) [special case of hyperbolic with b=1]

        NOTE: All decline rates are stored per time step (not per year).

        :param from_time_step: Starting time step for analysis.
        :param to_time_step: Ending time step for analysis.
        :param decline_type: Type of decline curve ('exponential', 'hyperbolic', 'harmonic').
        :param phase: Phase to analyze ('oil', 'gas', 'water').
        :return: `DeclineCurveResult` containing fitted decline curve parameters and forecasts.
        """
        if decline_type not in {"exponential", "hyperbolic", "harmonic"}:
            raise ValueError("Invalid decline type specified for analysis.")

        if phase not in {"oil", "gas", "water"}:
            raise ValueError("Invalid phase specified for decline curve analysis.")

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
            else:  # phase == "water"
                production_rates_list.append(instantaneous_rates.water_rate)

        time_steps_array = np.array(time_steps_list)
        production_rates_array = np.array(production_rates_list)

        # Filter out zero and negative rates for meaningful decline analysis
        positive_production_mask = production_rates_array > 0
        if np.sum(positive_production_mask) < 2:
            return DeclineCurveResult(
                decline_type=decline_type,
                initial_rate=0.0,
                decline_rate_per_timestep=0.0,  # Will be 0 for error cases
                b_factor=0.0,
                r_squared=0.0,
                phase=phase,
                error=f"Insufficient positive {phase} rate data for analysis",
                time_steps=None,
                actual_rates=None,
                predicted_rates=None,
            )

        filtered_time_steps = time_steps_array[positive_production_mask]
        positive_production_rates = production_rates_array[positive_production_mask]

        if decline_type == "exponential":
            # Exponential decline: q = qi * exp(-Di*t)
            # Use linear regression on ln(q) vs t to find parameters
            log_production_rates = np.log(positive_production_rates)
            linear_regression_coefficients = np.polyfit(
                filtered_time_steps, log_production_rates, 1
            )

            exponential_decline_rate_per_timestep = -linear_regression_coefficients[0]
            log_initial_rate_intercept = linear_regression_coefficients[1]
            exponential_initial_production_rate = np.exp(log_initial_rate_intercept)

            # Calculate coefficient of determination (R²) for goodness of fit
            predicted_exponential_rates = exponential_initial_production_rate * np.exp(
                -exponential_decline_rate_per_timestep * filtered_time_steps
            )
            sum_squared_residuals = np.sum(
                (positive_production_rates - predicted_exponential_rates) ** 2
            )
            total_sum_squares = np.sum(
                (positive_production_rates - np.mean(positive_production_rates)) ** 2
            )
            exponential_r_squared = (
                1 - (sum_squared_residuals / total_sum_squares)
                if total_sum_squares > 0
                else 0.0
            )
            return DeclineCurveResult(
                decline_type="exponential",
                initial_rate=exponential_initial_production_rate,
                decline_rate_per_timestep=exponential_decline_rate_per_timestep,  # Now per timestep
                b_factor=0.0,
                r_squared=exponential_r_squared,
                phase=phase,
                error=None,
                time_steps=filtered_time_steps.tolist(),
                actual_rates=positive_production_rates.tolist(),
                predicted_rates=predicted_exponential_rates.tolist(),
            )

        if decline_type == "harmonic":
            # Harmonic decline: q = qi / (1 + Di*t) [special case of hyperbolic with b=1]
            # Use linear regression on 1/q vs t to find parameters
            reciprocal_production_rates = 1.0 / positive_production_rates
            harmonic_regression_coefficients = np.polyfit(
                filtered_time_steps, reciprocal_production_rates, 1
            )

            harmonic_decline_rate_per_timestep = harmonic_regression_coefficients[0]
            reciprocal_initial_rate_intercept = harmonic_regression_coefficients[1]
            harmonic_initial_production_rate = 1.0 / reciprocal_initial_rate_intercept

            # Calculate predicted rates and R²
            predicted_harmonic_rates = harmonic_initial_production_rate / (
                1.0 + harmonic_decline_rate_per_timestep * filtered_time_steps
            )
            harmonic_sum_squared_residuals = np.sum(
                (positive_production_rates - predicted_harmonic_rates) ** 2
            )
            harmonic_total_sum_squares = np.sum(
                (positive_production_rates - np.mean(positive_production_rates)) ** 2
            )
            harmonic_r_squared = (
                1 - (harmonic_sum_squared_residuals / harmonic_total_sum_squares)
                if harmonic_total_sum_squares > 0
                else 0.0
            )
            return DeclineCurveResult(
                decline_type="harmonic",
                initial_rate=harmonic_initial_production_rate,
                decline_rate_per_timestep=harmonic_decline_rate_per_timestep,  # Now per timestep
                b_factor=1.0,  # Harmonic decline has b=1
                r_squared=harmonic_r_squared,
                phase=phase,
                error=None,
                time_steps=filtered_time_steps.tolist(),
                actual_rates=positive_production_rates.tolist(),
                predicted_rates=predicted_harmonic_rates.tolist(),
            )

        # decline_type == "hyperbolic":
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
        estimated_initial_rate = positive_production_rates[0]
        estimated_decline_rate = 0.001  # Small initial estimate per timestep
        estimated_b_factor = 0.5  # Typical hyperbolic exponent

        # Get time step size to set appropriate bounds for decline rate
        # Typical decline rates: 0.05-0.8 per year for oil wells
        # Convert to per-timestep by dividing by number of timesteps per year
        state = self.get_state(from_time_step)
        time_step_size_seconds = state.time_step_size
        timesteps_per_year = c.SECONDS_PER_YEAR / time_step_size_seconds

        # Upper bound for decline rate: 2.0 per year → 2.0/timesteps_per_year per timestep
        max_decline_per_timestep = 2.0 / timesteps_per_year

        # Perform non-linear curve fitting
        optimized_parameters, parameter_covariance = curve_fit(
            hyperbolic_decline_function,
            filtered_time_steps,
            positive_production_rates,
            p0=[estimated_initial_rate, estimated_decline_rate, estimated_b_factor],
            bounds=([0, 0, 0.1], [np.inf, max_decline_per_timestep, 2.0]),
            maxfev=1000,
        )

        (
            hyperbolic_initial_rate,
            hyperbolic_decline_rate_per_timestep,
            hyperbolic_b_factor,
        ) = optimized_parameters

        # Calculate predicted rates and R²
        predicted_hyperbolic_rates = hyperbolic_decline_function(
            filtered_time_steps,
            hyperbolic_initial_rate,
            hyperbolic_decline_rate_per_timestep,
            hyperbolic_b_factor,
        )
        hyperbolic_sum_squared_residuals = np.sum(
            (positive_production_rates - predicted_hyperbolic_rates) ** 2
        )
        hyperbolic_total_sum_squares = np.sum(
            (positive_production_rates - np.mean(positive_production_rates)) ** 2
        )
        hyperbolic_r_squared = (
            1 - (hyperbolic_sum_squared_residuals / hyperbolic_total_sum_squares)
            if hyperbolic_total_sum_squares > 0
            else 0.0
        )
        return DeclineCurveResult(
            decline_type="hyperbolic",
            initial_rate=hyperbolic_initial_rate,
            decline_rate_per_timestep=hyperbolic_decline_rate_per_timestep,  # Now per timestep
            b_factor=hyperbolic_b_factor,
            r_squared=hyperbolic_r_squared,
            phase=phase,
            error=None,
            time_steps=filtered_time_steps.tolist(),
            actual_rates=positive_production_rates.tolist(),
            predicted_rates=predicted_hyperbolic_rates.tolist(),
        )

    def forecast_production(
        self,
        decline_result: DeclineCurveResult,
        time_steps: int,
        economic_limit: typing.Optional[float] = None,
    ) -> typing.List[typing.Tuple[int, float]]:
        """
        Forecast future production rates based on fitted decline curve parameters.

        This method extrapolates production rates into the future using the decline curve
        model and parameters obtained from historical data analysis. The forecast continues
        until either the specified number of time steps is reached or the production rate
        falls below the economic limit.

        Forecast Equations by Decline Type:

        1. Exponential Decline:
        q(t) = qi x exp(-Di x t)

        2. Harmonic Decline:
        q(t) = qi / (1 + Di x t)

        3. Hyperbolic Decline:
        q(t) = qi / (1 + b x Di x t)^(1/b)

        Where:
        - q(t) = production rate at time step t (STB/day for oil/water, scf/day for gas)
        - qi = initial production rate at start of forecast
        - Di = decline rate per time step (NOT per year)
        - b = hyperbolic exponent (0 < b ≤ 2)
        - t = number of time steps since start of forecast

        NOTE: Production rates are in "per day" units (STB/day, scf/day) regardless of
        time step size. The decline rate is per timestep, but rates remain in per-day units.

        :param decline_result: `DeclineCurveResult` object containing fitted decline parameters
            from `decline_curve_analysis()` method. Must have valid parameters
            (no error flag set).
        :param time_steps: Number of time steps to forecast into the future.
        :param economic_limit: Optional minimum economic production rate in per-day units
            (STB/day for oil/water, scf/day for gas). Forecasting stops when predicted
            rate falls below this value. Default is None (no economic limit applied).
        :return: List of tuples containing (time_step, forecasted_rate). Rates are in
            per-day units (STB/day or scf/day). Time steps are absolute values continuing
            from the last historical time step. Returns empty list if decline_result
            contains errors.
        """
        if decline_result.error:
            return []

        last_time_step = (
            decline_result.time_steps[-1] if decline_result.time_steps else 0
        )
        forecast = []
        decline_rate_per_timestep = decline_result.decline_rate_per_timestep

        for t in range(1, time_steps + 1):
            future_time = last_time_step + t
            time_since_start = t  # Time steps since start of forecast

            if decline_result.decline_type == "exponential":
                rate = decline_result.initial_rate * np.exp(
                    -decline_rate_per_timestep * time_since_start
                )
            elif decline_result.decline_type == "harmonic":
                rate = decline_result.initial_rate / (
                    1 + decline_rate_per_timestep * time_since_start
                )
            elif decline_result.decline_type == "hyperbolic":
                rate = decline_result.initial_rate / (
                    1
                    + decline_result.b_factor
                    * decline_rate_per_timestep
                    * time_since_start
                ) ** (1 / decline_result.b_factor)
            else:
                rate = 0.0

            # Economic limit is already in per-day units (same as rate), so direct comparison works
            if economic_limit and rate < economic_limit:
                break

            forecast.append((future_time, float(rate)))
        return forecast

    def estimate_economic_ultimate_recovery(
        self,
        decline_result: DeclineCurveResult,
        forecast_time_steps: int,
        economic_limit: float = 0.0,
    ) -> float:
        """
        Calculate Estimated Ultimate Recovery (EUR) using decline curve analysis.

        EUR represents the total cumulative production expected from a well or reservoir
        over its economic life. This method uses analytical integration of decline curve
        equations to calculate cumulative production.

        Cumulative Production Equations by Decline Type:

        1. Exponential Decline:
        Np = (qi - qf) / Di

        2. Harmonic Decline:
        Np = (qi / Di) x ln(qi / qf)

        3. Hyperbolic Decline:
        For b ≠ 1:
        Np = (qi^b / (Di x (1-b))) x [qi^(1-b) - qf^(1-b)]

        For b = 1 (harmonic case):
        Np = (qi / Di) x ln(qi / qf)

        Where:
        - Np = cumulative production (STB for oil/water, scf for gas)
        - qi = initial production rate (STB/day or scf/day)
        - qf = final production rate (STB/day or scf/day, at forecast end or economic limit)
        - Di = decline rate per time step
        - b = hyperbolic exponent (0 < b ≤ 2)

        NOTE: The returned EUR is in stock tank barrels (STB) for oil/water or standard
        cubic feet (scf) for gas. The calculation accounts for the time step size by
        using the integrated decline equations which naturally handle the time step units.

        :param decline_result: `DeclineCurveResult` object from `decline_curve_analysis()` containing
            the fitted decline parameters (qi, Di, b) and decline type.
        :param forecast_time_steps: Number of time steps to forecast for EUR calculation.
        :param economic_limit: Minimum economic production rate in per-day units
            (STB/day for oil/water, scf/day for gas). Production below this
            rate is assumed to be uneconomic. Default is 0.0 (no limit).
        :return: Estimated Ultimate Recovery (EUR) in stock tank barrels (STB) for oil/water
            or standard cubic feet (scf) for gas. Returns 0.0 if decline_result contains
            errors or if parameters are invalid.
        """
        if decline_result.error:
            return 0.0

        qi = decline_result.initial_rate  # STB/day or scf/day
        di = decline_result.decline_rate_per_timestep  # per timestep
        b = decline_result.b_factor

        if di <= 0:
            return 0.0

        # Calculate final rate after forecast_time_steps (in STB/day or scf/day)
        if decline_result.decline_type == "exponential":
            q_final = qi * np.exp(-di * forecast_time_steps)
        elif decline_result.decline_type == "harmonic":
            q_final = qi / (1 + di * forecast_time_steps)
        elif decline_result.decline_type == "hyperbolic":
            q_final = qi / (1 + b * di * forecast_time_steps) ** (1 / b)
        else:
            return 0.0

        # Apply economic limit (both in per-day units)
        q_final = max(q_final, economic_limit)

        # Calculate time to reach economic limit if applicable
        if economic_limit > 0 and q_final == economic_limit:
            if decline_result.decline_type == "exponential":
                time_to_limit = (
                    -np.log(economic_limit / qi) / di if qi > economic_limit else 0
                )
            elif decline_result.decline_type == "harmonic":
                time_to_limit = (
                    (qi - economic_limit) / (di * economic_limit)
                    if economic_limit > 0
                    else 0
                )
            elif decline_result.decline_type == "hyperbolic":
                if b > 0:
                    time_to_limit = ((qi / economic_limit) ** b - 1) / (b * di)
                else:
                    time_to_limit = 0
            else:
                time_to_limit = 0

            # Use shorter of forecast_time_steps or time_to_limit
            effective_time_steps = min(forecast_time_steps, int(time_to_limit))

            # Recalculate q_final at effective time
            if decline_result.decline_type == "exponential":
                q_final = qi * np.exp(-di * effective_time_steps)
            elif decline_result.decline_type == "harmonic":
                q_final = qi / (1 + di * effective_time_steps)
            elif decline_result.decline_type == "hyperbolic":
                q_final = qi / (1 + b * di * effective_time_steps) ** (1 / b)

        # Calculate cumulative production analytically
        # NOTE: These formulas give results in units of [rate × time]
        # Since rate is in per-day units and time is in timesteps, we get [per-day × timesteps]
        # This naturally gives us the correct volume units (STB or scf) as long as
        # the decline rate Di is per timestep and qi, qf are per day

        if decline_result.decline_type == "exponential":
            # Q = (qi - q_final) / Di
            # Units: [(STB/day) / (1/timestep)] = [STB/day × timestep]
            # If timestep = 1 day, this gives STB ✓
            cumulative = (qi - q_final) / di

        elif decline_result.decline_type == "harmonic":
            # Q = (qi / Di) * ln(qi / q_final)
            if q_final > 0:
                cumulative = (qi / di) * np.log(qi / q_final)
            else:
                cumulative = 0.0

        elif decline_result.decline_type == "hyperbolic":
            # Q = qi^b / (Di*(1-b)) * [qi^(1-b) - q_final^(1-b)]
            if abs(b - 1.0) < 0.001:  # Close to harmonic
                if q_final > 0:
                    cumulative = (qi / di) * np.log(qi / q_final)
                else:
                    cumulative = 0.0
            else:
                cumulative = (
                    (qi**b) / (di * (1 - b)) * (qi ** (1 - b) - q_final ** (1 - b))
                )
        else:
            cumulative = 0.0

        return float(cumulative)

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

    def voidage_replacement_ratio_history(
        self, from_time_step: int = 0, to_time_step: int = -1, interval: int = 1
    ) -> typing.Generator[typing.Tuple[int, float], None, None]:
        """
        Generator for voidage replacement ratio (VRR) history over time.

        The VRR tracks how well injection is maintaining reservoir pressure by
        comparing injected volumes to produced volumes on a reservoir volume basis.

        :param from_time_step: Starting time step index (inclusive).
        :param to_time_step: Ending time step index (inclusive).
        :param interval: Time step interval for sampling.
        :return: Generator yielding tuples of (time_step, voidage_replacement_ratio).
        """
        if to_time_step < 0:
            to_time_step = self._state_count + to_time_step

        for t in range(from_time_step, to_time_step + 1, interval):
            yield (t, self.voidage_replacement_ratio(t))
