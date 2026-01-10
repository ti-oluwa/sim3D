import typing

import attrs

from bores.constants import Constants
from bores.types import (
    EvolutionScheme,
    IterativeSolver,
    MiscibilityModel,
    Preconditioner,
    Range,
    RelativeMobilityRange,
)
from bores.pvt.tables import PVTTables

__all__ = ["Config", "DampingController"]


@attrs.define
class DampingController:
    """
    Adaptive controller for Newton damping factor used in Fully Implicit scheme.
    """

    initial_damping: float = 1.0
    """Initial damping factor value."""
    min_damping: float = 0.01
    """Minimum allowable damping factor."""
    max_damping: float = 1.0
    """Maximum allowable damping factor."""
    increase_factor: float = 1.2
    """Factor to increase damping on successful steps."""
    decrease_factor: float = 0.5
    """Factor to decrease damping on unsuccessful/stagnant steps."""

    damping: float = attrs.field(init=False, default=1.0)
    """Current damping factor."""

    def __attrs_post_init__(self):
        self.damping = self.initial_damping

    def reset(self):
        """Resets the damping factor to its initial value."""
        self.damping = self.initial_damping

    def decrease(self):
        """Decreases the damping factor."""
        self.damping = max(self.damping * self.decrease_factor, self.min_damping)
        return self.damping

    def increase(self):
        """Increases the damping factor."""
        self.damping = min(self.damping * self.increase_factor, self.max_damping)
        return self.damping

    def set(self, value: float):
        """Sets the damping factor to a specific value within bounds."""
        self.damping = min(max(value, self.min_damping), self.max_damping)
        return self.damping

    def get(self):
        """Gets the current applicable damping factor."""
        return self.damping


@attrs.frozen
class Config:
    """Simulation run configuration and parameters."""

    convergence_tolerance: float = attrs.field(
        default=1e-6, validator=attrs.validators.le(1e-2)
    )
    """Convergence tolerance for iterative solvers (default is 1e-6)."""
    max_iterations: int = attrs.field(
        default=250,
        validator=attrs.validators.and_(
            attrs.validators.ge(1), attrs.validators.le(500)
        ),
    )
    """
    Maximum number of iterations allowed per time step for iterative solvers.
    
    Capped at 500 to prevent excessive computation time in case of non-convergence.
    If the solver does not converge within this limit, the matrix is most likely
    ill-conditioned or the problem setup needs to be reviewed. Use a stronger
    preconditioner, try another solver, or adjust simulation parameters accordingly.
    """
    output_frequency: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    """Frequency at which model states are yielded/outputted during the simulation."""
    scheme: EvolutionScheme = "impes"
    """Evolution scheme to use for the simulation ('impes', 'explicit', 'implicit')."""
    use_pseudo_pressure: bool = True
    """Whether to use pseudo-pressure for gas (when applicable)."""
    relative_mobility_range: RelativeMobilityRange = attrs.field(
        default=RelativeMobilityRange(
            oil=Range(min=1e-9, max=1e6),
            water=Range(min=1e-9, max=1e6),
            gas=Range(min=1e-12, max=1e6),
        )
    )
    """
    Relative mobility ranges for oil, water, and gas phases.

    Each phase has a `Range` object defining its minimum and maximum relative mobility.
    Adjust minimum or maximum values to constrain phase mobilities during simulation.

    Minimum values should not be exactly zero for the best numerical stability.
    """
    total_compressibility_range: Range = attrs.field(default=Range(min=1e-24, max=1e-2))
    """Range to constrain total compressibility for the simulation. This is usually necessary for numerical stability."""
    capillary_strength_factor: float = attrs.field(
        default=1.0,
        validator=attrs.validators.and_(attrs.validators.ge(0), attrs.validators.le(1)),
    )
    """
    Factor to scale capillary flow for numerical stability. Reduce to dampen capillary effects.
    Increase to enhance capillary effects.

    Capillary gradients can become numerically dominant in fine meshes or sharp saturation fronts.
    Damping avoids overshoot/undershoot by reducing their contribution without removing them.

    Set to 0 to disable capillary effects entirely (not recommended).
    """
    disable_capillary_effects: bool = False
    """Whether to include capillary pressure effects in the simulation."""
    disable_structural_dip: bool = attrs.field(default=False)
    """Whether to disable structural dip effects in reservoir modeling/simulation."""
    miscibility_model: MiscibilityModel = "immiscible"
    """Miscibility model: 'immiscible', 'todd_longstaff'"""
    impes_cfl_threshold: float = 0.9
    """Maximum allowable CFL number for the 'impes' evolution scheme to ensure numerical stability.

    Typically kept below 1.0 to prevent instability in explicit pressure updates.

    Lowering this value increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """
    explicit_saturation_cfl_threshold: float = 0.6
    """
    Maximum allowable saturation CFL number for the 'explicit' evolution scheme to ensure numerical stability.

    Typically kept below 1.0 to prevent instability in explicit saturation updates.

    Lowering this value increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """
    explicit_pressure_cfl_threshold: float = 0.9
    """
    Maximum allowable pressure CFL number for the 'explicit' evolution scheme to ensure numerical stability.

    Typically kept below 1.0 to prevent instability in explicit pressure updates.
    
    Lowering this value increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """
    constants: Constants = attrs.field(factory=Constants)
    """Physical and conversion constants used in the simulation."""
    warn_well_anomalies: bool = True
    """Whether to warn about anomalous flow rates during the simulation."""
    log_interval: int = attrs.field(default=5, validator=attrs.validators.ge(0))
    """Interval (in time steps) at which to log simulation progress."""
    preconditioner: typing.Optional[Preconditioner] = "ilu"
    """Preconditioner to use for iterative solvers."""
    iterative_solver: typing.Union[
        IterativeSolver, typing.Iterable[IterativeSolver]
    ] = "bicgstab"
    """Iterative solver(s) to use for solving linear systems."""
    phase_appearance_tolerance: float = attrs.field(
        default=1e-6, validator=attrs.validators.ge(0)
    )
    """
    Tolerance for determining phase appearance/disappearance based on saturation levels.

    Used to avoid numerical issues when a phase's saturation approaches zero. This helps
    maintain stability in relative permeability and mobility calculations by treating phases
    with saturations below this threshold as absent from the system.
    """
    residual_oil_drainage_ratio_water_flood: float = 0.6
    """Ratio to compute oil drainage residual from imbibition value during water flooding."""
    residual_oil_drainage_ratio_gas_flood: float = 0.6
    """Ratio to compute oil drainage residual from imbibition value during gas flooding."""
    residual_gas_drainage_ratio: float = 0.5
    """Ratio to compute gas drainage residual from imbibition value."""
    damping_controller_factory: typing.Callable[[], DampingController] = (
        DampingController
    )
    """Factory for creating the adaptive Newton damping factor controller in Fully Implicit scheme."""
    max_jacobian_reuses: int = 3
    """
    Maximum number of time steps to reuse the cached Jacobian matrix in the Fully Implicit solver.

    The fully implcit solver uses a quasi-Newton approach where the Jacobian matrix is reused
    for multiple newton iterations to save computational cost, when certian criteria are met.
    This parameter sets the maximum number of time steps the Jacobian can be reused before
    it is recomputed.
    """
    pvt_tables: typing.Optional[PVTTables] = None
    """PVT tables for fluid property lookups during the simulation."""
