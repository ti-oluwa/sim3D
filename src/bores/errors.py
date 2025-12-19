class BORESError(Exception):
    """Base class for all BORES-related errors."""

    pass


class ValidationError(BORESError, ValueError):
    """Raised when input data fails validation checks."""

    pass


class PreconditionerError(BORESError):
    """Raised when there is an error related to preconditioners."""

    pass


class SolverError(BORESError):
    """Raised when a solver fails to converge within the specified iterations."""

    pass


class ComputationError(BORESError):
    """Raised when there is an error during numerical computations."""

    pass


class SimulationError(BORESError):
    """Base class for simulation-related errors."""

    pass


class TimingError(SimulationError):
    """Raised when there is an error related to simulation timing."""

    pass
