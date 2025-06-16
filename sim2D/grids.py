import typing
import numpy as np
from numpy.typing import DTypeLike, NDArray

from sim2D.typing import FluidMiscibility, TwoDimensionalGrid, ArrayLike, InjectionFluid
from sim2D.properties import (
    get_fluid_viscosity,
    compute_fluid_compressibility,
    compute_miscible_viscosity,
)


def build_2D_uniform_grid(
    grid_dimension: typing.Tuple[int, int],
    value: float = 0.0,
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Constructs a 2D uniform grid with the specified initial value.

    :param grid_dimension: Tuple of number of cells in x and y directions (cell_count_x, cell_count_y)
    :param value: Initial value to fill the grid with
    :param dtype: Data type of the grid elements (default: np.float64)
    :return: 2D numpy array representing the grid
    """
    cell_count_x, cell_count_y = grid_dimension
    return np.full((cell_count_x, cell_count_y), fill_value=value, dtype=dtype)


def build_2D_layered_grid(
    grid_dimension: typing.Tuple[int, int],
    layer_values: ArrayLike[float],
    layering_direction: typing.Literal["horizontal", "vertical"] = "vertical",
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Constructs a 2D layered grid with specified layer values.

    :param grid_dimension: Tuple of number of cells in x and y directions (cell_count_x, cell_count_y)
    :param layering_direction: Direction or axis along which layers are defined ('horizontal' or 'vertical')
    :param layer_values: Values for each layer (must match number of layers).
        The number of values should match the number of cells in that direction.
        If the grid dimension is (50, 30) and layering_direction is 'horizontal',
        then values should have exactly 50 values.
        If layering_direction is 'vertical', then values should have exactly 30 values.

    :return: 2D numpy array representing the grid
    """
    cell_count_x, cell_count_y = grid_dimension
    if len(layer_values) < 1:
        raise ValueError("At least one layer value must be provided.")
    if layering_direction not in ["horizontal", "vertical"]:
        raise ValueError("Layering direction must be 'horizontal' or 'vertical'.")

    if layering_direction == "horizontal":
        if len(layer_values) != cell_count_x:
            raise ValueError(
                "Number of layer values must match number of cells in x direction."
            )

        grid = np.zeros((cell_count_x, cell_count_y), dtype=dtype)
        for i, layer_value in enumerate(layer_values):
            grid[i, :] = layer_value

    else:  # vertical layering
        if len(layer_values) != cell_count_y:
            raise ValueError(
                "Number of layer values must match number of cells in y direction."
            )

        grid = np.zeros((cell_count_x, cell_count_y), dtype=dtype)
        for j, layer_value in enumerate(layer_values):
            grid[:, j] = layer_value

    return grid


def build_2D_injection_grid(
    grid_dimension: typing.Tuple[int, int],
    injectors_positions: ArrayLike[typing.Tuple[int, int]],
    injection_rates: ArrayLike[float],
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Constructs a 2D injection grid with specified injector positions and rates.

    NOTE:

    Injector positions must be within the bounds of the grid dimensions but should not
    include boundary cells, as they would not be accounted for in the simulation, since
    the models use finite difference methods that do not include boundary cells.

    :param grid_dimension: Tuple of number of cells in x and y directions (cell_count_x, cell_count_y)
    :param injectors_positions: Sequence of tuples representing (x, y) positions of injectors
    :param injection_rates: Sequence containing rates of injection in m³/s for each injection position
    :param dtype: Data type of the grid elements (default: np.float64)
    :return: 2D numpy array representing the injection grid
    """
    if len(injectors_positions) != len(injection_rates):
        raise ValueError("Number of positions must match number of provided rates.")

    injection_grid = build_2D_uniform_grid(
        grid_dimension=grid_dimension, value=0.0, dtype=dtype
    )
    # If no positions are provided, return a zeros grid
    if not injectors_positions:
        return injection_grid

    cell_count_x, cell_count_y = grid_dimension

    for index, position in enumerate(injectors_positions):
        if not (0 <= position[0] < cell_count_x and 0 <= position[1] < cell_count_y):
            raise ValueError(
                f"Position {position} is out of bounds. It must be within the grid dimensions {grid_dimension}."
            )
        injection_grid[position[0], position[1]] = injection_rates[index]

    return injection_grid


def build_2D_production_grid(
    grid_dimension: typing.Tuple[int, int],
    producers_positions: ArrayLike[typing.Tuple[int, int]],
    production_rates: ArrayLike[float],
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Constructs a 2D production grid with specified producer positions and rates.

    NOTE:

    Injector positions must be within the bounds of the grid dimensions but should not
    include boundary cells, as they would not be accounted for in the simulation, since
    the models use finite difference methods that do not include boundary cells.

    :param grid_dimension: Tuple of number of cells in x and y directions (cell_count_x, cell_count_y)
    :param producers_positions: Sequence of tuples representing (x, y) positions of producers
    :param production_rate: Sequence containing rates of production in m³/s for each production position
    :param dtype: Data type of the grid elements (default: np.float64)
    :return: 2D numpy array representing the production grid
    """
    return build_2D_injection_grid(
        grid_dimension=grid_dimension,
        injectors_positions=producers_positions,
        injection_rates=-1 * np.array(production_rates, dtype=dtype),
        dtype=dtype,
    )


def build_2D_fluid_viscosity_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    fluid: typing.Union[InjectionFluid, str],
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Builds a 2D grid of fluid viscosities based on pressure and temperature grids.

    :param pressure_grid: 2D array of pressure values (Pa) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (K) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute viscosity (e.g., "CO2", "water") supported by `CoolProp`.
    :return: 2D array of fluid viscosities (Pa·s) corresponding to each grid cell.
    """
    return np.vectorize(get_fluid_viscosity, otypes=[dtype])(
        pressure_grid, temperature_grid, fluid
    )


def build_2D_fluid_compressibility_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    fluid: InjectionFluid = "CO2",
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Builds a 2D grid of fluid compressibilities based on pressure and temperature grids.

    :param pressure_grid: 2D array of pressure values (Pa) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (K) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute compressibility (e.g., "CO2", "water") supported by `CoolProp`.
    :return: 2D array of fluid compressibilities (1/Pa) corresponding to each grid cell.
    """
    return np.vectorize(compute_fluid_compressibility, otypes=[dtype])(
        pressure_grid, temperature_grid, fluid
    )


def edge_pad_grid(grid: NDArray, pad_width: int = 1) -> NDArray:
    """
    Pads a 2D grid with the edge values to create a border around the grid.

    This is useful for finite difference methods where boundary conditions are applied.

    :param grid: 2D numpy array representing the grid to be padded
    :param pad_width: Width of the padding to be applied on all sides of the grid
    :return: Padded 2D numpy array
    """
    return np.pad(grid, pad_width=pad_width, mode="edge")


def build_miscible_viscosity_grid(
    injected_fluid_saturation_grid: TwoDimensionalGrid,
    injected_fluid_viscosity_grid: typing.Optional[TwoDimensionalGrid],
    displaced_fluid_viscosity_grid: TwoDimensionalGrid,
    fluid_misciblity: FluidMiscibility = "logarithmic",
) -> TwoDimensionalGrid:
    """
    Builds a 2D grid of miscible viscosities based on the saturation of the injected fluid
    and the viscosities of the injected and displaced fluids.

    The miscible viscosity is computed using the saturation of the injected fluid and its viscosity,
    along with the viscosity of the displaced fluid. The method used for computation can be specified
    through the `fluid_misciblity` parameter.
    If the injected fluid viscosity is not provided, the displaced fluid viscosity is returned unchanged.

    :param injected_fluid_saturation_grid: 2D array of saturation values of the injected fluid (fraction)
    :param injected_fluid_viscosity_grid: 2D array of viscosity values of the injected fluid (Pa·s)
    :param displaced_fluid_viscosity_grid: 2D array of viscosity values of the displaced fluid (Pa·s)
    :param fluid_misciblity: Method to compute miscible viscosity, can be "logarithmic" or "linear"
    :return: 2D array of miscible viscosities (Pa·s) corresponding to each grid cell
    """
    if injected_fluid_viscosity_grid is not None:
        return np.vectorize(
            compute_miscible_viscosity, otypes=[displaced_fluid_viscosity_grid.dtype]
        )(
            injected_fluid_saturation_grid,
            injected_fluid_viscosity_grid,
            displaced_fluid_viscosity_grid,
            fluid_misciblity,
        )
    return displaced_fluid_viscosity_grid.copy()


def build_effective_mobility_grid(
    injected_fluid_saturation_grid: TwoDimensionalGrid,
    injected_fluid_viscosity_grid: typing.Optional[TwoDimensionalGrid],
    displaced_fluid_viscosity_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D array of effective mobilities based on the saturation of the injected fluid
    and the viscosities of the injected and displaced fluids.

    This assumes that the fluid saturation is directly proportional to relative permeability

    The effective mobility is defined as:
        λ_eff = S_g / μ_g + (1 - S_g) / μ_o

    where:
        - S_g is the saturation of the injected fluid (gas)
        - μ_g is the viscosity of the injected fluid (gas)
        - μ_o is the viscosity of the displaced fluid (oil)

    :param injected_fluid_saturation_grid: 2D array of injected fluid saturation values (fraction)
    :param injected_fluid_viscosity_grid: Optional 2D array of injected fluid viscosity values (Pa·s)
        If None, the effective mobility is computed as if the injected fluid has zero viscosity.
        
    :param displaced_fluid_viscosity_grid:  2D array of displaced fluid viscosity vaues (Pa·s)
    :return: 2D array of effective mobility values
    """
    if injected_fluid_viscosity_grid is None:
        return (1 - injected_fluid_saturation_grid) / displaced_fluid_viscosity_grid

    return (injected_fluid_saturation_grid / injected_fluid_viscosity_grid) + (
        (1 - injected_fluid_saturation_grid) / displaced_fluid_viscosity_grid
    )


def build_pressure_dependent_viscosity_grid(
    pressure_grid: TwoDimensionalGrid,
    reference_pressure: float,
    reference_viscosity: float,
    pressure_decay_constant: float = 1e-8,
) -> TwoDimensionalGrid:
    """
    Computes a 2D array of pressure-dependent displaced fluid (e.g oil) viscosities due to Injected fluid (e.g, CO₂) dissolution
    using an exponential decay model.

    Useful when simulating how the viscosity of the displaced fluid changes with pressure only (i.e, no miscibility effects).
    If miscibility effects are to be considered, the saturation of the injected fluid should be taken into account.
    use `build_pressure_and_saturation_dependent_viscosity_grid` instead.

    The model is defined as:

        μ(P) = μ_ref * exp(-k_p * (P - P_ref))

    where:
        - μ(P) is the viscosity at pressure, P
        - μ_ref is the viscosity at reference pressure, P_ref
        - k_p is a decay constant that controls how fast viscosity changes with pressure
        - P_ref is the reference pressure where viscosity is known
        - P is the current pressure

    :param pressure_grid: 2D array of current reservoir pressures (Pa)
    :param reference_pressure: Initial reservoir pressure (Pa) where viscosity is known.
    :param reference_viscosity: Displaced fluid viscosity at reference pressure (Pa·s).
        Usually, this is the viscosity of the displaced fluid at initial conditions.

    :param pressure_decay_constant: Empirical constant controlling how fast viscosity changes with pressure, exponentially
    :return: 2D array of viscosity values (Pa·s) for the displaced fluid
    """
    delta_p = pressure_grid - reference_pressure
    pressure_decay = np.exp(-pressure_decay_constant * delta_p)
    viscosity_grid = reference_viscosity * pressure_decay
    return np.clip(viscosity_grid, 1e-8, reference_viscosity)  # enforce lower bound


def build_pressure_and_saturation_dependent_viscosity_grid(
    pressure_grid: TwoDimensionalGrid,
    injected_fluid_saturation_grid: TwoDimensionalGrid,
    reference_pressure: float,
    reference_viscosity: float,
    pressure_decay_constant: float = 1e-8,
    saturation_decay_constant: float = 1e-3,
) -> TwoDimensionalGrid:
    """
    Computes a 2D array of pressure and saturation-dependent displaced fluid (e.g oil) viscosities
    due to Injected fluid (e.g, CO₂) dissolution using an exponential decay model.

    Useful when simulating how the viscosity of the displaced fluid changes with both pressure and saturation
    of the injected fluid. Especially relevant in miscible displacement scenarios, where the dissolution of the injected fluid
    into the displaced fluid affects its viscosity.

    If only pressure effects are to be considered, use `build_pressure_dependent_viscosity_grid` instead.

    The model is defined as:

        μ(P, S_i) = μ_ref * exp(-k_p * (P - P_ref)) * (1 + k_s * S_i)

    where:
        - μ(P, S_i) is the viscosity at pressure, P and saturation, S_i of the injected fluid
        - μ_ref is the viscosity at reference pressure, P_ref
        - k_p is a decay constant that controls how fast viscosity changes with pressure
        - k_s is a decay constant that controls how fast viscosity changes with saturation or miscibility
        - S_i is the saturation of the injected fluid (fraction)
        - P is the current pressure
        - P_ref is the reference pressure where viscosity is known

    :param pressure_grid: 2D array of current reservoir pressures (Pa)
    :param injected_fluid_saturation_grid: 2D array of saturation values of the injected fluid (fraction)
    :param reference_pressure: Initial reservoir pressure (Pa) where viscosity is known.
    :param reference_viscosity: Displaced fluid viscosity at reference pressure (Pa·s).
        Usually, this is the viscosity of the displaced fluid at initial conditions.

    :param pressure_decay_constant: Empirical constant controlling how fast viscosity changes with pressure, exponentially
    :param saturation_decay_constant: Empirical constant controlling how fast viscosity changes with saturation or miscibility, exponentially
    :return: 2D array of viscosity values (Pa·s) for the displaced fluid
    """
    delta_p = pressure_grid - reference_pressure
    pressure_decay = np.exp(-pressure_decay_constant * delta_p)
    saturation_decay = 1 + (saturation_decay_constant * injected_fluid_saturation_grid)
    viscosity_grid = (reference_viscosity * pressure_decay) * saturation_decay
    return np.clip(viscosity_grid, 1e-8, reference_viscosity)  # enforce lower bound
