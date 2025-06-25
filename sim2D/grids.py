"""Utils for building 2D cartesian grids for reservoir simulation."""

import typing
import itertools
import numpy as np
from numpy.typing import DTypeLike, NDArray

from sim2D.typing import TwoDimensionalGrid, ArrayLike
from sim2D.properties import (
    compute_fluid_viscosity,
    compute_fluid_density,
    compute_fluid_compressibility,
    compute_gas_molecular_weight,
    mix_fluid_property,
    compute_total_fluid_compressibility,
    compute_three_phase_relative_permeabilities,
    compute_three_phase_capillary_pressures,
    compute_oil_api_gravity,
    compute_oil_specific_gravity_from_density,
    compute_oil_formation_volume_factor,
    compute_water_formation_volume_factor,
    compute_gas_formation_volume_factor,
    compute_gas_compressibility_factor,
    compute_oil_bubble_point_pressure,
    compute_gas_to_oil_ratio,
    compute_gas_solubility_in_water,
    compute_water_bubble_point_pressure,
    compute_oil_viscosity,
    compute_gas_viscosity,
    compute_water_viscosity,
    compute_oil_compressibility,
    compute_gas_free_water_formation_volume_factor,
    compute_water_compressibility,
    compute_gas_compressibility,
    compute_live_oil_density,
    compute_gas_density,
    compute_water_density,
    compute_gas_gravity,
    compute_gas_gravity_from_density,
)
from sim2D.models import CapillaryPressureParameters, RelativePermeabilityParameters


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
    :param injection_rates: Sequence containing rates of injection for each injection position
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
    :param production_rate: Sequence containing rates of production for each production position
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
    fluid: typing.Union[str, str],
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Builds a 2D grid of fluid viscosities based on pressure and temperature grids.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute viscosity (e.g., "CO2", "water") supported by `CoolProp`.
    :return: 2D array of fluid viscosities (cP) corresponding to each grid cell.
    """
    return np.vectorize(compute_fluid_viscosity, otypes=[dtype])(
        pressure_grid, temperature_grid, fluid
    )


def build_2D_fluid_compressibility_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    fluid: str,
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Builds a 2D grid of fluid compressibilities based on pressure and temperature grids.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute compressibility (e.g., "CO2", "water") supported by `CoolProp`.
    :return: 2D array of fluid compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    return np.vectorize(compute_fluid_compressibility, otypes=[dtype])(
        pressure_grid, temperature_grid, fluid
    )


def build_2D_fluid_density_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    fluid: str,
    dtype: DTypeLike = np.float64,
) -> TwoDimensionalGrid:
    """
    Builds a 2D grid of fluid densities based on pressure and temperature grids.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute density (e.g., "CO2", "water") supported by `CoolProp`.
    :return: 2D array of fluid densities (lbm/ft³) corresponding to each grid cell.
    """
    return np.vectorize(compute_fluid_density, otypes=[dtype])(
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


def build_2D_gas_gravity_grid(gas: str) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas gravity based gas type.

    The gas gravity is computed using the pressure and temperature conditions.

    :param gas: Type of gas (e.g., "methane", "co2", "n2") for which to compute gas gravity.
    :return: 2D array of gas gravity values (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_gravity, otypes=[np.float64])(gas)


def build_2D_gas_gravity_from_density_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    density_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    omputes a 2D grid of gas gravity based on pressure, temperature, and gas den.

    The gas gravity is computed using the pressure and temperature conditions.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param density_grid: 2D array of gas density values (lbm/ft³) corresponding to each grid cell.
    :return: 2D array of gas gravity values (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_gravity_from_density, otypes=[np.float64])(
        pressure_grid, temperature_grid, density_grid
    )


def build_2D_mixed_fluid_property_grid(
    fluid1_saturation_grid: TwoDimensionalGrid,
    fluid1_property_grid: TwoDimensionalGrid,
    fluid2_saturation_grid: TwoDimensionalGrid,
    fluid2_property_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Builds a 2D grid of mixed fluid properties based on the saturation of two fluids
    and their respective properties.

    The mixed property is computed as a weighted average based on the saturations of the two fluids.

    :param fluid1_saturation_grid: 2D array of saturation values of the first fluid (fraction)
    :param fluid1_property_grid: 2D array of property values of the first fluid (e.g., viscosity, compressibility)
    :param fluid2_saturation_grid: 2D array of saturation values of the second fluid (fraction)
    :param fluid2_property_grid: 2D array of property values of the second fluid (e.g., viscosity, compressibility)
    :return: 2D array of mixed fluid properties corresponding to each grid cell
    """
    return np.vectorize(mix_fluid_property, otypes=[fluid1_property_grid.dtype])(
        fluid1_saturation_grid,
        fluid1_property_grid,
        fluid2_saturation_grid,
        fluid2_property_grid,
    )


def build_2D_total_fluid_compressibility_grid(
    oil_saturation_grid: TwoDimensionalGrid,
    oil_compressibility_grid: TwoDimensionalGrid,
    water_saturation_grid: TwoDimensionalGrid,
    water_compressibility_grid: TwoDimensionalGrid,
    gas_saturation_grid: typing.Optional[TwoDimensionalGrid] = None,
    gas_compressibility_grid: typing.Optional[TwoDimensionalGrid] = None,
) -> TwoDimensionalGrid:
    """
    Computes a 2D array of total fluid compressibility based on the saturation of water, oil,
    and optionally the gas, along with their respective compressibilities.

    The total fluid compressibility is defined as:
        C_total = (S_w * C_w) + (S_o * C_o) +( S_g * C_g)

    where:
        - C_total is the total fluid compressibility (psi⁻¹)
        - S_w is the saturation of water (fraction)
        - C_w is the compressibility of water (psi⁻¹)
        - S_o is the saturation of oil (fraction)
        - C_o is the compressibility of oil (psi⁻¹)
        - S_g is the saturation of the gas (fraction)
        - C_g is the compressibility of the gas (psi⁻¹)

    :param water_saturation_grid: 2D array of water saturation values (fraction)
    :param oil_saturation_grid: 2D array of oil saturation values (fraction)
    :param water_compressibility_grid: 2D array of water compressibility values (psi⁻¹)
    :param oil_compressibility_grid: 2D array of oil compressibility values (psi⁻¹)
    :param gas_saturation_grid: Optional 2D array of gas saturation values (fraction)
    :param gas_compressibility_grid: Optional 2D array of gas compressibility values (psi⁻¹)

    :return: 2D array of total fluid compressibility values (psi⁻¹) corresponding to each grid cell
    """
    return np.vectorize(
        compute_total_fluid_compressibility, otypes=[water_saturation_grid.dtype]
    )(
        water_saturation_grid,
        oil_saturation_grid,
        water_compressibility_grid,
        oil_compressibility_grid,
        gas_saturation_grid,
        gas_compressibility_grid,
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

    :param pressure_grid: 2D array of current reservoir pressures (psi)
    :param reference_pressure: Initial reservoir pressure (psi) where viscosity is known.
    :param reference_viscosity: Displaced fluid viscosity at reference pressure (cP).
        Usually, this is the viscosity of the displaced fluid at initial conditions.

    :param pressure_decay_constant: Empirical constant controlling how fast viscosity changes with pressure, exponentially
    :return: 2D array of viscosity values (cP) for the displaced fluid
    """
    delta_p = pressure_grid - reference_pressure
    pressure_decay = np.exp(-pressure_decay_constant * delta_p)
    viscosity_grid = reference_viscosity * pressure_decay
    return np.clip(viscosity_grid, 1e-5, reference_viscosity)  # enforce lower bound


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

    :param pressure_grid: 2D array of current reservoir pressures (psi)
    :param injected_fluid_saturation_grid: 2D array of saturation values of the injected fluid (fraction)
    :param reference_pressure: Initial reservoir pressure (psi) where viscosity is known.
    :param reference_viscosity: Displaced fluid viscosity at reference pressure (cP).
        Usually, this is the viscosity of the displaced fluid at initial conditions.

    :param pressure_decay_constant: Empirical constant controlling how fast viscosity changes with pressure, exponentially
    :param saturation_decay_constant: Empirical constant controlling how fast viscosity changes with saturation or miscibility, exponentially
    :return: 2D array of viscosity values (cP) for the displaced fluid
    """
    delta_p = pressure_grid - reference_pressure
    pressure_decay = np.exp(-pressure_decay_constant * delta_p)
    saturation_decay = 1 + (saturation_decay_constant * injected_fluid_saturation_grid)
    viscosity_grid = (reference_viscosity * pressure_decay) * saturation_decay
    return np.clip(viscosity_grid, 1e-5, reference_viscosity)  # enforce lower bound


def build_2D_three_phase_capillary_pressure_grids(
    water_saturation_grid: TwoDimensionalGrid,
    gas_saturation_grid: TwoDimensionalGrid,
    irreducible_water_saturation_grid: TwoDimensionalGrid,
    residual_oil_saturation_grid: TwoDimensionalGrid,
    residual_gas_saturation_grid: TwoDimensionalGrid,
    capillary_pressure_params: CapillaryPressureParameters,
) -> typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid]:
    """
    Computes the capillary pressure grids for water, oil, and gas three-phase system.

    This function calculates the capillary pressures based on Corey model with residual saturations.

    :param water_saturation_grid: 2D array of water saturation values (fraction).
    :param gas_saturation_grid: 2D array of gas saturation values (fraction).
    :param irreducible_water_saturation_grid: 2D array of irreducible water saturation values (fraction).
    :param residual_oil_saturation_grid: 2D array of residual oil saturation values (fraction).
    :param residual_gas_saturation_grid: 2D array of residual gas saturation values (fraction).
    :param capillary_pressure_params: `CapillaryPressureParameters` object containing parameters for capillary pressure calculations.
    :return: Tuple of (oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid)
        where each grid is a 2D numpy array of capillary pressures (psi).
    """
    cell_count_x, cell_count_y = water_saturation_grid.shape
    oil_water_capillary_pressure_grid = build_2D_uniform_grid(
        grid_dimension=(cell_count_x, cell_count_y), value=0.0, dtype=np.float64
    )
    gas_oil_capillary_pressure_grid = build_2D_uniform_grid(
        grid_dimension=(cell_count_x, cell_count_y), value=0.0, dtype=np.float64
    )

    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        # Get current saturations for the cell
        water_saturation = water_saturation_grid[i, j]
        gas_saturation = gas_saturation_grid[i, j]

        # Get cell-specific rock properties
        irreducible_water_saturation = irreducible_water_saturation_grid[i, j]
        residual_oil_saturation = residual_oil_saturation_grid[i, j]
        residual_gas_saturation = residual_gas_saturation_grid[i, j]

        # Compute three-phase capillary pressures
        (
            oil_water_capillary_pressure_grid[i, j],
            gas_oil_capillary_pressure_grid[i, j],
        ) = compute_three_phase_capillary_pressures(
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation=residual_oil_saturation,
            residual_gas_saturation=residual_gas_saturation,
            capillary_pressure_params=capillary_pressure_params,
        )

    return (
        oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid,
    )


def build_2D_three_phase_relative_mobilities_grids(
    water_saturation_grid: TwoDimensionalGrid,
    oil_saturation_grid: TwoDimensionalGrid,
    gas_saturation_grid: TwoDimensionalGrid,
    water_viscosity_grid: TwoDimensionalGrid,
    oil_viscosity_grid: TwoDimensionalGrid,
    gas_viscosity_grid: TwoDimensionalGrid,
    irreducible_water_saturation_grid: TwoDimensionalGrid,
    residual_oil_saturation_grid: TwoDimensionalGrid,
    residual_gas_saturation_grid: TwoDimensionalGrid,
    relative_permeability_params: "RelativePermeabilityParameters",
) -> typing.Tuple[TwoDimensionalGrid, TwoDimensionalGrid, TwoDimensionalGrid]:
    """
    Computes the relative mobility grids for water, oil, and gas phases for a three-phase system.

    This function calculates the relative mobilities based on relative permeabilities
    (using Corey model with residual saturations), and phase viscosities.

    Relative mobility = Relative Permeability / Viscosity

    :param water_saturation_grid: 2D array of water saturation values (fraction).
    :param oil_saturation_grid: 2D array of oil saturation values (fraction).
    :param gas_saturation_grid: 2D array of gas saturation values (fraction).
    :param water_viscosity_grid: 2D array of water viscosity values (cP).
    :param oil_viscosity_grid: 2D array of oil viscosity values (cP).
    :param gas_viscosity_grid: 2D array of gas viscosity values (cP).
    :param irreducible_water_saturation_grid: 2D array of irreducible water saturation values (fraction).
    :param residual_oil_saturation_grid: 2D array of residual oil saturation values (fraction).
    :param residual_gas_saturation_grid: 2D array of residual gas saturation values (fraction).
    :param relative_permeability_params: `RelativePermeabilityParameters` object containing Corey exponents for water, oil, and gas.
    :return: Tuple of (water_relative_mobility_grid, oil_relative_mobility_grid, gas_relative_mobility_grid) in 1/(cP).
    """
    cell_count_x, cell_count_y = water_saturation_grid.shape
    water_relative_mobility_grid = build_2D_uniform_grid(
        grid_dimension=(cell_count_x, cell_count_y), value=0.0, dtype=np.float64
    )
    oil_relative_mobility_grid = build_2D_uniform_grid(
        grid_dimension=(cell_count_x, cell_count_y), value=0.0, dtype=np.float64
    )
    gas_relative_mobility_grid = build_2D_uniform_grid(
        grid_dimension=(cell_count_x, cell_count_y), value=0.0, dtype=np.float64
    )

    for i, j in itertools.product(range(cell_count_x), range(cell_count_y)):
        # Get current saturations for the cell
        water_saturation = water_saturation_grid[i, j]
        oil_saturation = oil_saturation_grid[i, j]
        gas_saturation = gas_saturation_grid[i, j]

        # Get viscosities for the cell
        # Ensure viscosities are not zero to avoid division by zero
        water_viscosity = np.maximum(water_viscosity_grid[i, j], 1e-7)
        oil_viscosity = np.maximum(oil_viscosity_grid[i, j], 1e-7)
        gas_viscosity = np.maximum(gas_viscosity_grid[i, j], 1e-7)

        # Get cell-specific rock properties
        irreducible_water_saturation = irreducible_water_saturation_grid[i, j]
        residual_oil_saturation = residual_oil_saturation_grid[i, j]
        residual_gas_saturation = residual_gas_saturation_grid[i, j]

        # Compute three-phase relative permeabilities
        (
            water_relative_permeability,
            oil_relative_permeability,
            gas_relative_permeability,
        ) = compute_three_phase_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation=residual_oil_saturation,
            residual_gas_saturation=residual_gas_saturation,
            water_exponent=relative_permeability_params.water_exponent,
            oil_exponent=relative_permeability_params.oil_exponent,
            gas_exponent=relative_permeability_params.gas_exponent,
        )

        # Compute phase mobilities
        water_relative_mobility_grid[i, j] = (
            water_relative_permeability / water_viscosity
        )
        oil_relative_mobility_grid[i, j] = oil_relative_permeability / oil_viscosity
        gas_relative_mobility_grid[i, j] = gas_relative_permeability / gas_viscosity

    # Ensure no NaN or Inf values in the mobility grids
    water_relative_mobility_grid[
        np.isnan(water_relative_mobility_grid) | np.isinf(water_relative_mobility_grid)
    ] = 0.0
    oil_relative_mobility_grid[
        np.isnan(oil_relative_mobility_grid) | np.isinf(oil_relative_mobility_grid)
    ] = 0.0
    gas_relative_mobility_grid[
        np.isnan(gas_relative_mobility_grid) | np.isinf(gas_relative_mobility_grid)
    ] = 0.0

    return (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    )


def build_2D_oil_api_gravity_grid(
    oil_specific_gravity_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of oil API gravity based on oil density.

    The API gravity is computed using the formula:
    API Gravity = (141.5 / Specific Gravity) - 131.5

    :param oil_specific_gravity_grid: 2D array of oil specific gravity values (dimensionless).
    :return: 2D array of oil API gravity values (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_api_gravity, otypes=[np.float64])(
        oil_specific_gravity_grid
    )


def build_2D_oil_specific_gravity_grid(
    oil_density_grid: TwoDimensionalGrid,
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    oil_compressibility_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of oil specific gravity based on pressure, temperature, and oil density grids.

    The specific gravity is computed as the ratio of the oil density to the standard density of water (1000 kg/m³).

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param oil_density_grid: 2D array of oil density values (lbm/ft³).
    :param oil_compressibility_grid: 2D array of oil compressibility values (psi⁻¹).
    :return: 2D array of oil specific gravity (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_specific_gravity_from_density, otypes=[np.float64])(
        oil_density_grid, pressure_grid, temperature_grid, oil_compressibility_grid
    )


def build_2D_gas_compressibility_factor_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    gas_gravity_grid: TwoDimensionalGrid,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas compressibility factors based on gas gravity, pressure, and temperature grids.

    The compressibility factor is computed using the gas gravity and the pressure and temperature conditions.

    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param h2s_mole_fraction: Mole fraction of H₂S in the gas mixture (default: 0.0).
    :param co2_mole_fraction: Mole fraction of CO₂ in the gas mixture (default: 0.0).
    :param n2_mole_fraction: Mole fraction of N₂ in the gas mixture (default: 0.0).
    :return: 2D array of gas compressibility factors (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_compressibility_factor, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        gas_gravity_grid,
        h2s_mole_fraction,
        co2_mole_fraction,
        n2_mole_fraction,
    )


def build_2D_oil_formation_volume_factor_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    bubble_point_pressure_grid: TwoDimensionalGrid,
    oil_specific_gravity_grid: TwoDimensionalGrid,
    gas_gravity_grid: TwoDimensionalGrid,
    gas_to_oil_ratio_grid: TwoDimensionalGrid,
    oil_compressibility_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of oil formation volume factors based on pressure and temperature grids.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: 2D array of oil bubble point pressures (psi) corresponding to each grid cell.
    :param oil_specific_gravity_grid: 2D array of oil specific gravity values (dimensionless),
    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_to_oil_ratio_grid: 2D array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param oil_compressibility_grid: 2D array of oil compressibility values (psi⁻¹) representing the compressibility of oil.
    :return: 2D array of oil formation volume factors (bbl/STB) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_formation_volume_factor, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        bubble_point_pressure_grid,
        oil_specific_gravity_grid,
        gas_gravity_grid,
        gas_to_oil_ratio_grid,
        oil_compressibility_grid,
    )


def build_2D_water_formation_volume_factor_grid(
    water_density_grid: TwoDimensionalGrid,
    salinity_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of water formation volume factors.

    :param water_density_grid: 2D array of water density values (lbm/ft³) representing the density of water at reservoir conditions.
    :param salinity_grid: 2D array of water salinity values (ppm of NaCl) representing the salinity of water at reservoir conditions.
    :return: 2D array of water formation volume factors (bbl/STB) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_formation_volume_factor, otypes=[np.float64])(
        water_density_grid, salinity_grid
    )


def build_2D_gas_formation_volume_factor_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    gas_compressibility_factor_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas formation volume factors based on pressure and temperature grids.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_compressibility_factor_grid: 2D array of gas compressibility factor values (dimensionless) representing the compressibility of gas.
    :return: 2D array of gas formation volume factors (ft³/SCF) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_formation_volume_factor, otypes=[np.float64])(
        pressure_grid, temperature_grid, gas_compressibility_factor_grid
    )


def build_2D_gas_to_oil_ratio_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    bubble_point_pressure_grid: TwoDimensionalGrid,
    gas_gravity_grid: TwoDimensionalGrid,
    oil_api_gravity_grid: TwoDimensionalGrid,
    gor_at_bubble_point_pressure_grid: typing.Optional[TwoDimensionalGrid] = None,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of solution gas-to-oil ratios based on pressure, temperature, and bubble point pressure grids.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.

    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: 2D array of bubble point pressures (psi) corresponding to each grid cell.
    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param oil_api_gravity_grid: 2D array of oil API gravity values (dimensionless) representing the density of oil relative to water.
    :param gor_at_bubble_point_pressure_grid: Optional 2D array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: 2D array of solution gas-to-oil ratios (SCF/STB) corresponding to each grid cell.
    """
    return np.vectorize(
        compute_gas_to_oil_ratio,
        otypes=[np.float64],
    )(
        pressure_grid,
        temperature_grid,
        bubble_point_pressure_grid,
        gas_gravity_grid,
        oil_api_gravity_grid,
        gor_at_bubble_point_pressure_grid,
    )


def build_2D_oil_bubble_point_pressure_grid(
    gas_gravity_grid: TwoDimensionalGrid,
    oil_api_gravity_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    gas_to_oil_ratio_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of oil bubble point pressures based on oil specific gravity, gas gravity, and temperature grids.

    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param oil_api_gravity_grid: 2D array of API gravity values.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gor_at_bubble_point_pressure_grid: 2D array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: 2D array of oil bubble point pressures (psi) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_bubble_point_pressure, otypes=[np.float64])(
        gas_gravity_grid,
        oil_api_gravity_grid,
        temperature_grid,
        gas_to_oil_ratio_grid,
    )


def build_2D_gas_solubility_in_water_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    salinity_grid: TwoDimensionalGrid,
    gas: str = "methane",
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas solubility in water based on pressure, temperature, and salinity grids.

    The solubility is computed using the Henry's law constant for the specific gas in water.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param salinity_grid: 2D array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas: Type of gas dissolved in water (default: "methane"). Can be CO₂, N₂, etc.
    :return: 2D array of gas solubility in water (SCF/STB) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_solubility_in_water, otypes=[np.float64])(
        pressure_grid, temperature_grid, salinity_grid, gas=gas
    )


def build_2D_water_bubble_point_pressure_grid(
    temperature_grid: TwoDimensionalGrid,
    gas_solubility_in_water_grid: TwoDimensionalGrid,
    salinity_grid: TwoDimensionalGrid,
    gas: str = "methane",
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of water bubble point pressures based on temperature, gas solubility in water, and salinity grids.

    The bubble point pressure is computed using the gas solubility in water and the salinity of the water.

    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_solubility_in_water_grid: 2D array of gas solubility in water (SCF/STB) at bubble point pressure corresponding to each grid cell.
    :param salinity_grid: 2D array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas: Type of gas dissolved in water (default: "methane"). Can be CO₂, N₂, etc.
    :return: 2D array of water bubble point pressures (psi) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_bubble_point_pressure, otypes=[np.float64])(
        temperature_grid,
        gas_solubility_in_water_grid,
        salinity_grid,
        gas=gas,
    )


def build_2D_oil_viscosity_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    bubble_point_pressure_grid: TwoDimensionalGrid,
    oil_specific_gravity_grid: TwoDimensionalGrid,
    gas_to_oil_ratio_grid: TwoDimensionalGrid,
    gor_at_bubble_point_pressure_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of oil viscosities based on pressure, temperature, bubble point pressure, and gas-to-oil ratio grids.

    The viscosity is computed using the oil specific gravity, gas gravity, and the gas-to-oil ratio.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: 2D array of bubble point pressures (psi) corresponding to each grid cell.
    :param oil_specific_gravity_grid: 2D array of oil specific gravity values (dimensionless).
    :param gas_to_oil_ratio_grid: 2D array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param gor_at_bubble_point_pressure_grid: Optional 2D array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: 2D array of oil viscosities (cP) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_viscosity, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        bubble_point_pressure_grid,
        oil_specific_gravity_grid,
        gas_to_oil_ratio_grid,
        gor_at_bubble_point_pressure_grid,
    )


def build_2D_gas_molecular_weight_grid(
    gas_gravity_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas molecular weights based on gas gravity.

    The molecular weight is computed using the formula:
    Molecular Weight = Gas Gravity * 28.9644 g/mol (molecular weight of air)

    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :return: 2D array of gas molecular weights (g/mol) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_molecular_weight, otypes=[np.float64])(
        gas_gravity_grid
    )


def build_2D_gas_viscosity_grid(
    temperature_grid: TwoDimensionalGrid,
    gas_density_grid: TwoDimensionalGrid,
    gas_molecular_weight_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas viscosities.


    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_density_grid: 2D array of gas density values (lbm/ft³) representing the density of gas in each grid cell.
    :param gas_molecular_weight_grid: 2D array of gas molecular weight values (g/mol) representing the molecular weight of gas in each grid cell.
    :return: 2D array of gas viscosities (cP) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_viscosity, otypes=[np.float64])(
        temperature_grid,
        gas_density_grid,
        gas_molecular_weight_grid,
    )


def build_2D_water_viscosity_grid(
    temperature_grid: TwoDimensionalGrid,
    salinity_grid: TwoDimensionalGrid,
    pressure_grid: typing.Optional[TwoDimensionalGrid] = None,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of water viscosities based on pressure, temperature, and salinity grids.

    The viscosity is computed using the pressure, temperature, and salinity conditions.

    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param salinity_grid: 2D array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param pressure_grid: Optional 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :return: 2D array of water viscosities (cP) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_viscosity, otypes=[np.float64])(
        temperature_grid, salinity_grid, pressure_grid
    )


def build_2D_oil_compressibility_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    bubble_point_pressure_grid: TwoDimensionalGrid,
    oil_api_gravity_grid: TwoDimensionalGrid,
    gas_gravity_grid: TwoDimensionalGrid,
    gas_to_oil_ratio_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of oil compressibilities based on pressure, temperature, bubble point pressure, and gas-to-oil ratio grids.

    The compressibility is computed using the oil API gravity, gas gravity, and the gas-to-oil ratio.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: 2D array of bubble point pressures (psi) corresponding to each grid cell.
    :param oil_api_gravity_grid: 2D array of oil API gravity values (dimensionless).
    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_to_oil_ratio_grid: 2D array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :return: 2D array of oil compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_compressibility, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        bubble_point_pressure_grid,
        oil_api_gravity_grid,
        gas_gravity_grid,
        gas_to_oil_ratio_grid,
    )


def build_2D_gas_compressibility_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    gas_gravity_grid: TwoDimensionalGrid,
    gas_compressibility_factor_grid: typing.Optional[TwoDimensionalGrid] = None,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas compressibilities based on pressure, temperature, and gas gravity grids.

    The compressibility is computed using the gas gravity and the pressure and temperature conditions.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_compressibility_factor_grid: Optional 2D array of precalculated gas compressibility factors (dimensionless).
        If provided, it will be used directly; otherwise, it will be computed.
    :param h2s_mole_fraction: Mole fraction of H₂S in the gas mixture (default: 0.0).
    :param co2_mole_fraction: Mole fraction of CO₂ in the gas mixture (default: 0.0).
    :param n2_mole_fraction: Mole fraction of N₂ in the gas mixture (default: 0.0).
    :return: 2D array of gas compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_compressibility, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        gas_gravity_grid,
        gas_compressibility_factor_grid,
        h2s_mole_fraction,
        co2_mole_fraction,
        n2_mole_fraction,
    )


def build_2D_gas_free_water_formation_volume_factor_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas free water formation volume factors based on pressure and temperature grids.

    The gas free water formation volume factor is computed using the pressure and temperature conditions.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :return: 2D array of gas free water formation volume factors (bbl/STB) corresponding to each grid cell.
    """
    return np.vectorize(
        compute_gas_free_water_formation_volume_factor, otypes=[np.float64]
    )(pressure_grid, temperature_grid)


def build_2D_water_compressibility_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    bubble_point_pressure_grid: TwoDimensionalGrid,
    gas_formation_volume_factor_grid: TwoDimensionalGrid,
    gas_solubility_in_water_grid: TwoDimensionalGrid,
    gas_free_water_formation_volume_factor_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of water compressibilities based on pressure, temperature, and bubble point pressure grids.

    The compressibility is computed using the gas formation volume factor, gas solubility in water, and gas free water formation volume factor.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: 2D array of bubble point pressures (psi) corresponding to each grid cell.
    :param gas_formation_volume_factor_grid: 2D array of gas formation volume factors (ft³/SCF) corresponding to each grid cell.
    :param gas_solubility_in_water_grid: 2D array of gas solubility in water (SCF/STB) corresponding to each grid cell.
    :param gas_free_water_formation_volume_factor_grid: 2D array of gas free water formation volume factors (bbl/STB).
    :return: 2D array of water compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_compressibility, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        bubble_point_pressure_grid,
        gas_formation_volume_factor_grid,
        gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid,
    )


def build_2D_live_oil_density_grid(
    oil_api_gravity_grid: TwoDimensionalGrid,
    gas_gravity_grid: TwoDimensionalGrid,
    gas_to_oil_ratio_grid: TwoDimensionalGrid,
    formation_volume_factor_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of live oil densities based on pressure, bubble point pressure, and gas-to-oil ratio grids.

    The density is computed using the oil API gravity, gas gravity, and the gas-to-oil ratio.

    :param oil_api_gravity_grid: 2D array of oil API gravity values (dimensionless).
    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_to_oil_ratio_grid: 2D array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param formation_volume_factor_grid: 2D array of formation volume factors (bbl/STB).
    :return: 2D array of oil densities (lbm/ft³) corresponding to each grid cell.
    """
    return np.vectorize(compute_live_oil_density, otypes=[np.float64])(
        oil_api_gravity_grid,
        gas_gravity_grid,
        gas_to_oil_ratio_grid,
        formation_volume_factor_grid,
    )


def build_2D_gas_density_grid(
    pressure_grid: TwoDimensionalGrid,
    temperature_grid: TwoDimensionalGrid,
    gas_gravity_grid: TwoDimensionalGrid,
    gas_compressibility_factor_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of gas densities based on pressure, temperature, and gas gravity grids.

    The density is computed using the gas gravity and the pressure and temperature conditions.

    :param pressure_grid: 2D array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: 2D array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_compressibility_factor_grid: 2D array of gas compressibility factor values (dimensionless)
        representing the compressibility of gas.
    :return: 2D array of gas densities (lbm/ft³) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_density, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        gas_gravity_grid,
        gas_compressibility_factor_grid,
    )


def build_2D_water_density_grid(
    gas_gravity_grid: TwoDimensionalGrid,
    salinity_grid: TwoDimensionalGrid,
    gas_solubility_in_water_grid: TwoDimensionalGrid,
    gas_free_water_formation_volume_factor_grid: TwoDimensionalGrid,
) -> TwoDimensionalGrid:
    """
    Computes a 2D grid of water densities based on gas gravity, salinity, and gas solubility in water grids.

    The density is computed using the gas gravity, salinity, and gas solubility in water.

    :param gas_gravity_grid: 2D array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param salinity_grid: 2D array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas_solubility_in_water_grid: 2D array of gas solubility in water (SCF/STB).
    :param gas_free_water_formation_volume_factor_grid: 2D array of gas free water formation volume factors (bbl/STB).
    :return: 2D array of water densities (lbm/ft³) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_density, otypes=[np.float64])(
        gas_gravity_grid,
        salinity_grid,
        gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid,
    )
