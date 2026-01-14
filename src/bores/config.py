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
    saturation_cfl_threshold: float = 0.6
    """
    Maximum allowable saturation CFL number for the 'explicit' evolution scheme to ensure numerical stability.

    Typically kept below 1.0 to prevent instability in explicit saturation updates.

    Lowering this value increases stability but may require smaller time steps.
    Raising them can improve performance but risks instability. Use with caution and monitor simulation behavior.
    """
    pressure_cfl_threshold: float = 0.9
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
    log_interval: int = attrs.field(default=5, validator=attrs.validators.ge(0))  # type: ignore
    """Interval (in time steps) at which to log simulation progress."""
    preconditioner: typing.Optional[Preconditioner] = "ilu"
    """Preconditioner to use for iterative solvers."""
    iterative_solver: typing.Union[
        IterativeSolver, typing.Iterable[IterativeSolver]
    ] = "bicgstab"
    """Iterative solver(s) to use for solving linear systems."""
    phase_appearance_tolerance: float = attrs.field(  # type: ignore
        default=1e-6,
        validator=attrs.validators.ge(0),
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

    max_oil_saturation_change: float = 0.5
    """
    Maximum allowable oil saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """
    max_water_saturation_change: float = attrs.field(  # type: ignore
        default=0.4, validator=attrs.validators.ge(0)
    )
    """
    Maximum allowable water saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """
    max_gas_saturation_change: float = attrs.field(  # type: ignore
        default=0.7, validator=attrs.validators.ge(0)
    )
    """
    Maximum allowable gas saturation change (absolute, fractional) per time step.

    Controls time step size by limiting saturation variations to prevent numerical
    instabilities and maintain material balance accuracy. When exceeded, the time
    step is reduced or rejected.
    """
    max_pressure_change: float = attrs.field(  # type: ignore
        default=100.0, validator=attrs.validators.ge(0)
    )
    """
    Maximum allowable pressure change (in psi) per time step.

    Controls time step size by limiting pressure variations to maintain numerical stability
    and physical accuracy. When exceeded, the time step is reduced or rejected.

    Default: 100 psi (~7 bar). This is suitable for most field-scale simulations with typical
    reservoir pressures of 1000-5000 psi.

    Adjust based on simulation characteristics:

    **Tighten to 50-75 psi when:**
    - Simulating high-rate wells with large near-wellbore pressure gradients
    - Reservoir pressure is low (<1000 psi) to maintain relative accuracy
    - Using highly compressible fluids (gas reservoirs)
    - Fine-grid simulations (<10m cells) where local variations are significant
    - Observing pressure oscillations or convergence issues

    **Relax to 150-300 psi when:**
    - Depletion studies with slow, uniform pressure decline
    - Field-scale models (>100m cells) where averaging reduces local variations
    - Reservoir pressure is high (>5000 psi) making relative changes small
    - Simulation is stable and material balance errors are acceptable
    - Computational efficiency is critical and accuracy requirements are relaxed

    **Guidelines by reservoir pressure:**
    - Low pressure (<1000 psi): 25-50 psi (2.5-5% relative change)
    - Moderate pressure (1000-3000 psi): 50-100 psi (2-5% relative change)
    - High pressure (3000-6000 psi): 100-200 psi (2-4% relative change)
    - Very high pressure (>6000 psi): 150-300 psi (2-5% relative change)

    Note: Larger changes can cause density/viscosity jumps and well control issues.
    """

    pvt_tables: typing.Optional[PVTTables] = None
    """PVT tables for fluid property lookups during the simulation."""
