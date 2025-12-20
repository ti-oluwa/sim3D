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

__all__ = ["Config"]


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
    """
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
    explicit_saturation_cfl_threshold: float = 0.5
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
    warn_rates_anomalies: bool = True
    """Whether to warn about anomalous flow rates during the simulation."""
    log_interval: int = attrs.field(default=3, validator=attrs.validators.ge(1))
    """Interval (in time steps) at which to log simulation progress."""
    preconditioner: typing.Optional[Preconditioner] = "ilu"
    """Preconditioner to use for iterative solvers."""
    iterative_solver: typing.Union[
        IterativeSolver, typing.Iterable[IterativeSolver]
    ] = "bicgstab"
    """Iterative solver(s) to use for solving linear systems."""
