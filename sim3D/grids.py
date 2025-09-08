"""Utils for building N-Dimensional cartesian grids for reservoir simulation."""

import typing
import itertools
import numba
import numpy as np
from numpy.typing import DTypeLike

from sim3D.types import Orientation, NDimensionalGrid, NDimension, ArrayLike
from sim3D.properties import (
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
from sim3D.models import CapillaryPressureParameters, RelativePermeabilityParameters


@numba.njit(cache=True)
def build_uniform_grid(
    grid_dimension: NDimension,
    value: float = 0.0,
    dtype: DTypeLike = np.float64,
) -> NDimensionalGrid[NDimension]:
    """
    Constructs a N-Dimensional uniform grid with the specified initial value.

    :param grid_dimension: Tuple of number of cells in all directions (x, y, z).
    :param value: Initial value to fill the grid with
    :param dtype: Data type of the grid elements (default: np.float64)
    :return: Numpy array representing the grid
    """
    return np.full(grid_dimension, fill_value=value, dtype=dtype)


@numba.njit(cache=True)
def build_layered_grid(
    grid_dimension: NDimension,
    layer_values: ArrayLike[float],
    orientation: Orientation,
    dtype: DTypeLike = np.float64,
) -> NDimensionalGrid[NDimension]:
    """
    Constructs a N-Dimensional layered grid with specified layer values.

    :param grid_dimension: Tuple of number of cells in x, y, and z directions (cell_count_x, cell_count_y, cell_count_z)
    :param orientation: Direction or axis along which layers are defined ('x', 'y', or 'z')
    :param layer_values: Values for each layer (must match number of layers).
        The number of values should match the number of cells in that direction.
        If the grid NDimension is (50, 30, 10) and orientation is 'horizontal',
        then values should have exactly 50 values.
        If orientation is 'vertical', then values should have exactly 30 values.

    :return: N-Dimensional numpy array representing the grid
    """
    if len(layer_values) < 1:
        raise ValueError("At least one layer value must be provided.")

    layered_grid = build_uniform_grid(
        grid_dimension=grid_dimension, value=0.0, dtype=dtype
    )
    if orientation == Orientation.X:  # Layering along x-axis
        if len(layer_values) != grid_dimension[0]:
            raise ValueError(
                "Number of layer values must match number of cells in x direction."
            )

        for i, layer_value in enumerate(layer_values):
            layered_grid[i, :, :] = layer_value
        return layered_grid

    elif orientation == Orientation.Y:  # Layering along y-axis
        if len(layer_values) != grid_dimension[1]:
            raise ValueError(
                "Number of layer values must match number of cells in y direction."
            )

        for j, layer_value in enumerate(layer_values):
            layered_grid[:, j, :] = layer_value
        return layered_grid

    elif orientation == Orientation.Z:  # Layering along z-axis
        if len(grid_dimension) != 3:
            raise ValueError(
                "Grid dimension must be N-Dimensional for z-direction layering."
            )

        if len(layer_values) != grid_dimension[2]:
            raise ValueError(
                "Number of layer values must match number of cells in z direction."
            )

        for k, layer_value in enumerate(layer_values):
            layered_grid[:, :, k] = layer_value
        return layered_grid

    raise ValueError("Invalid layering direction. Must be one of 'x', 'y', or 'z'.")


def build_fluid_viscosity_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    fluid: typing.Union[str, str],
    dtype: DTypeLike = np.float64,
) -> NDimensionalGrid[NDimension]:
    """
    Builds a N-Dimensional grid of fluid viscosities.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute viscosity (e.g., "CO2", "water") supported by `CoolProp`.
    :return: N-Dimensional array of fluid viscosities (cP) corresponding to each grid cell.
    """
    return np.vectorize(compute_fluid_viscosity, otypes=[dtype], excluded=["fluid"])(
        pressure_grid, temperature_grid, fluid=fluid
    )


def build_fluid_compressibility_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    fluid: str,
    dtype: DTypeLike = np.float64,
) -> NDimensionalGrid[NDimension]:
    """
    Builds a N-Dimensional grid of fluid compressibilities.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute compressibility (e.g., "CO2", "water") supported by `CoolProp`.
    :return: N-Dimensional array of fluid compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    return np.vectorize(
        compute_fluid_compressibility, otypes=[dtype], excluded=["fluid"]
    )(pressure_grid, temperature_grid, fluid=fluid)


def build_fluid_density_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    fluid: str,
    dtype: DTypeLike = np.float64,
) -> NDimensionalGrid[NDimension]:
    """
    Builds a N-Dimensional grid of fluid densities.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param fluid: Type of fluid for which to compute density (e.g., "CO2", "water") supported by `CoolProp`.
    :return: N-Dimensional array of fluid densities (lbm/ft³) corresponding to each grid cell.
    """
    return np.vectorize(compute_fluid_density, otypes=[dtype], excluded=["fluid"])(
        pressure_grid, temperature_grid, fluid=fluid
    )


def edge_pad_grid(
    grid: NDimensionalGrid[NDimension], pad_width: int = 1
) -> NDimensionalGrid[NDimension]:
    """
    Pads a N-Dimensional grid with the edge values to create a border around the grid.

    This is useful for finite difference methods where boundary conditions are applied.

    :param grid: N-Dimensional numpy array representing the grid to be padded
    :param pad_width: Width of the padding to be applied on all sides of the grid
    :return: Padded N-Dimensional numpy array
    """
    padded_grid = np.pad(grid, pad_width=pad_width, mode="edge")
    padded_grid = typing.cast(NDimensionalGrid[NDimension], padded_grid)
    return padded_grid


def coarsen_grid(
    data: NDimensionalGrid[NDimension],
    batch_size: NDimension,
    method: typing.Literal["mean", "sum", "max", "min"] = "mean",
) -> NDimensionalGrid[NDimension]:
    """
    Coarsen (downsample) a 2D or 3D grid by aggregating blocks of adjacent cells.

    Pads the grid if necessary to make dimensions divisible by batch_size, so no cells are lost.

    :param data: 2D or 3D numpy array to coarsen. Shape can be (nx, ny) or (nx, ny, nz).
    :param batch_size: Tuple of ints representing the coarsening factor along each dimension.
                       Length must match data.ndim.
                       Example: (2,2) for 2D, (2,2,2) for 3D.
    :param method: Aggregation method to use on each block. One of 'mean', 'sum', 'max', 'min'.
                   Default is 'mean'.
    :return: Coarsened numpy array.
    :raises ValueError: if batch_size length does not match data.ndim or if method is unsupported.

    :example:
    >>> data2d = np.arange(16).reshape(4,4)
    >>> coarsen_grid(data2d, batch_size=(2,2))
    array([[2.5, 4.5],
           [10.5, 12.5]])

    >>> data3d = np.arange(64).reshape(4,4,4)
    >>> coarsen_grid(data3d, batch_size=(2,2,2), method='max')
    array([[[ 5,  7],
            [13, 15]],
           [[21, 23],
            [29, 31]]])
    """
    if len(batch_size) != data.ndim:
        raise ValueError(
            f"batch_size length {len(batch_size)} must match data.ndim {data.ndim}"
        )

    pad_width = []
    for dim, b in zip(data.shape, batch_size):
        remainder = dim % b
        if remainder == 0:
            pad_width.append((0, 0))
        else:
            pad_width.append((0, b - remainder))

    # Pad with NaNs for mean, zeros for sum, -inf for max, +inf for min
    if method == "mean":
        pad_value = np.nan
    elif method == "sum":
        pad_value = 0
    elif method == "max":
        pad_value = -np.inf
    elif method == "min":
        pad_value = np.inf
    else:
        raise ValueError(f"Unsupported method '{method}'")

    data_padded = np.pad(
        data, pad_width=pad_width, mode="constant", constant_values=pad_value
    )

    # Reshape to group blocks along each dimension
    reshape_shape = []
    for dim, b in zip(data_padded.shape, batch_size):
        reshape_shape.extend([dim // b, b])
    data_reshaped = data_padded.reshape(reshape_shape)

    # Axes to aggregate over: every second axis (the 'b' axes)
    agg_axes = tuple(range(1, data_reshaped.ndim, 2))

    # Apply aggregation
    if method == "mean":
        coarsened = np.nanmean(data_reshaped, axis=agg_axes)
    elif method == "sum":
        coarsened = data_reshaped.sum(axis=agg_axes)
    elif method == "max":
        coarsened = data_reshaped.max(axis=agg_axes)
    elif method == "min":
        coarsened = data_reshaped.min(axis=agg_axes)

    return typing.cast(NDimensionalGrid[NDimension], coarsened)


def build_gas_gravity_grid(gas: str) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas gravity based gas type.

    The gas gravity is computed using the pressure and temperature conditions.

    :param gas: Type of gas (e.g., "methane", "co2", "n2") for which to compute gas gravity.
    :return: N-Dimensional array of gas gravity values (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_gravity, otypes=[np.float64])(gas)


def build_gas_gravity_from_density_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    density_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    omputes a N-Dimensional grid of gas gravity based on pressure, temperature, and gas den.

    The gas gravity is computed using the pressure and temperature conditions.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param density_grid: N-Dimensional array of gas density values (lbm/ft³) corresponding to each grid cell.
    :return: N-Dimensional array of gas gravity values (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_gravity_from_density, otypes=[np.float64])(
        pressure_grid, temperature_grid, density_grid
    )


def build_mixed_fluid_property_grid(
    fluid1_saturation_grid: NDimensionalGrid[NDimension],
    fluid1_property_grid: NDimensionalGrid[NDimension],
    fluid2_saturation_grid: NDimensionalGrid[NDimension],
    fluid2_property_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Builds a N-Dimensional grid of mixed fluid properties based on the saturation of two fluids
    and their respective properties.

    The mixed property is computed as a weighted average based on the saturations of the two fluids.

    :param fluid1_saturation_grid: N-Dimensional array of saturation values of the first fluid (fraction)
    :param fluid1_property_grid: N-Dimensional array of property values of the first fluid (e.g., viscosity, compressibility)
    :param fluid2_saturation_grid: N-Dimensional array of saturation values of the second fluid (fraction)
    :param fluid2_property_grid: N-Dimensional array of property values of the second fluid (e.g., viscosity, compressibility)
    :return: N-Dimensional array of mixed fluid properties corresponding to each grid cell
    """
    return np.vectorize(mix_fluid_property, otypes=[fluid1_property_grid.dtype])(
        fluid1_saturation_grid,
        fluid1_property_grid,
        fluid2_saturation_grid,
        fluid2_property_grid,
    )


def build_total_fluid_compressibility_grid(
    oil_saturation_grid: NDimensionalGrid[NDimension],
    oil_compressibility_grid: NDimensionalGrid[NDimension],
    water_saturation_grid: NDimensionalGrid[NDimension],
    water_compressibility_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
    gas_compressibility_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional array of total fluid compressibilities.

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

    :param water_saturation_grid: N-Dimensional array of water saturation values (fraction)
    :param oil_saturation_grid: N-Dimensional array of oil saturation values (fraction)
    :param water_compressibility_grid: N-Dimensional array of water compressibility values (psi⁻¹)
    :param oil_compressibility_grid: N-Dimensional array of oil compressibility values (psi⁻¹)
    :param gas_saturation_grid: Optional N-Dimensional array of gas saturation values (fraction)
    :param gas_compressibility_grid: Optional N-Dimensional array of gas compressibility values (psi⁻¹)

    :return: N-Dimensional array of total fluid compressibility values (psi⁻¹) corresponding to each grid cell
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
    pressure_grid: NDimensionalGrid[NDimension],
    reference_pressure: float,
    reference_viscosity: float,
    pressure_decay_constant: float = 1e-8,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional array of pressure-dependent displaced fluid (e.g oil) viscosities due to Injected fluid (e.g, CO₂) dissolution
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

    :param pressure_grid: N-Dimensional array of current reservoir pressures (psi)
    :param reference_pressure: Initial reservoir pressure (psi) where viscosity is known.
    :param reference_viscosity: Displaced fluid viscosity at reference pressure (cP).
        Usually, this is the viscosity of the displaced fluid at initial conditions.

    :param pressure_decay_constant: Empirical constant controlling how fast viscosity changes with pressure, exponentially
    :return: N-Dimensional array of viscosity values (cP) for the displaced fluid
    """
    delta_p = pressure_grid - reference_pressure
    pressure_decay = np.exp(-pressure_decay_constant * delta_p)
    viscosity_grid = reference_viscosity * pressure_decay
    return typing.cast(
        NDimensionalGrid[NDimension],
        np.clip(
            viscosity_grid, 1e-5, reference_viscosity, dtype=np.float32
        ),  # enforce lower bound
    )


def build_pressure_and_saturation_dependent_viscosity_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    injected_fluid_saturation_grid: NDimensionalGrid[NDimension],
    reference_pressure: float,
    reference_viscosity: float,
    pressure_decay_constant: float = 1e-8,
    saturation_decay_constant: float = 1e-3,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional array of pressure and saturation-dependent displaced fluid (e.g oil) viscosities
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

    :param pressure_grid: N-Dimensional array of current reservoir pressures (psi)
    :param injected_fluid_saturation_grid: N-Dimensional array of saturation values of the injected fluid (fraction)
    :param reference_pressure: Initial reservoir pressure (psi) where viscosity is known.
    :param reference_viscosity: Displaced fluid viscosity at reference pressure (cP).
        Usually, this is the viscosity of the displaced fluid at initial conditions.

    :param pressure_decay_constant: Empirical constant controlling how fast viscosity changes with pressure, exponentially
    :param saturation_decay_constant: Empirical constant controlling how fast viscosity changes with saturation or miscibility, exponentially
    :return: N-Dimensional array of viscosity values (cP) for the displaced fluid
    """
    delta_p = pressure_grid - reference_pressure
    pressure_decay = np.exp(-pressure_decay_constant * delta_p)
    saturation_decay = 1 + (saturation_decay_constant * injected_fluid_saturation_grid)
    viscosity_grid = (reference_viscosity * pressure_decay) * saturation_decay
    return typing.cast(
        NDimensionalGrid[NDimension],
        np.clip(viscosity_grid, 1e-5, reference_viscosity),  # enforce lower bound
    )


def build_three_phase_capillary_pressure_grids(
    water_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    capillary_pressure_params: CapillaryPressureParameters,
) -> typing.Tuple[NDimensionalGrid[NDimension], NDimensionalGrid[NDimension]]:
    """
    Computes the capillary pressure grids for water, oil, and gas three-phase system.

    This function calculates the capillary pressures based on Corey model with residual saturations.

    :param water_saturation_grid: N-Dimensional array of water saturation values (fraction).
    :param gas_saturation_grid: N-Dimensional array of gas saturation values (fraction).
    :param irreducible_water_saturation_grid: N-Dimensional array of irreducible water saturation values (fraction).
    :param residual_oil_saturation_grid: N-Dimensional array of residual oil saturation values (fraction).
    :param residual_gas_saturation_grid: N-Dimensional array of residual gas saturation values (fraction).
    :param capillary_pressure_params: `CapillaryPressureParameters` object containing parameters for capillary pressure calculations.
    :return: Tuple of (oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid)
        where each grid is a N-Dimensional numpy array of capillary pressures (psi).
    """
    oil_water_capillary_pressure_grid = build_uniform_grid(
        grid_dimension=water_saturation_grid.shape,
        value=0.0,
        dtype=np.float32,
    )
    gas_oil_capillary_pressure_grid = build_uniform_grid(
        grid_dimension=water_saturation_grid.shape,
        value=0.0,
        dtype=np.float32,
    )

    for indices in itertools.product(*map(range, water_saturation_grid.shape)):
        # Get current saturations for the cell
        water_saturation = water_saturation_grid[indices]
        gas_saturation = gas_saturation_grid[indices]

        # Get cell-specific rock properties
        irreducible_water_saturation = irreducible_water_saturation_grid[indices]
        residual_oil_saturation = residual_oil_saturation_grid[indices]
        residual_gas_saturation = residual_gas_saturation_grid[indices]

        # Compute three-phase capillary pressures
        (
            oil_water_capillary_pressure_grid[indices],
            gas_oil_capillary_pressure_grid[indices],
        ) = compute_three_phase_capillary_pressures(
            water_saturation=water_saturation,
            gas_saturation=gas_saturation,
            irreducible_water_saturation=irreducible_water_saturation,
            residual_oil_saturation=residual_oil_saturation,
            residual_gas_saturation=residual_gas_saturation,
            wettability=capillary_pressure_params.wettability,
            oil_water_entry_pressure_oil_wet=capillary_pressure_params.oil_water_entry_pressure_oil_wet,
            oil_water_entry_pressure_water_wet=capillary_pressure_params.oil_water_entry_pressure_water_wet,
            gas_oil_entry_pressure=capillary_pressure_params.gas_oil_entry_pressure,
            oil_water_pore_size_distribution_index_oil_wet=capillary_pressure_params.oil_water_pore_size_distribution_index_oil_wet,
            oil_water_pore_size_distribution_index_water_wet=capillary_pressure_params.oil_water_pore_size_distribution_index_water_wet,
            gas_oil_pore_size_distribution_index=capillary_pressure_params.gas_oil_pore_size_distribution_index,
        )

    return (
        oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid,
    )


def build_three_phase_relative_mobilities_grids(
    water_saturation_grid: NDimensionalGrid[NDimension],
    oil_saturation_grid: NDimensionalGrid[NDimension],
    gas_saturation_grid: NDimensionalGrid[NDimension],
    water_viscosity_grid: NDimensionalGrid[NDimension],
    oil_viscosity_grid: NDimensionalGrid[NDimension],
    gas_viscosity_grid: NDimensionalGrid[NDimension],
    irreducible_water_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    relative_permeability_params: RelativePermeabilityParameters,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Computes the relative mobility grids for water, oil, and gas phases for a three-phase system.

    This function calculates the relative mobilities based on relative permeabilities
    (using Corey model with residual saturations), and phase viscosities.

    Relative mobility = Relative Permeability / Viscosity

    :param water_saturation_grid: N-Dimensional array of water saturation values (fraction).
    :param oil_saturation_grid: N-Dimensional array of oil saturation values (fraction).
    :param gas_saturation_grid: N-Dimensional array of gas saturation values (fraction).
    :param water_viscosity_grid: N-Dimensional array of water viscosity values (cP).
    :param oil_viscosity_grid: N-Dimensional array of oil viscosity values (cP).
    :param gas_viscosity_grid: N-Dimensional array of gas viscosity values (cP).
    :param irreducible_water_saturation_grid: N-Dimensional array of irreducible water saturation values (fraction).
    :param residual_oil_saturation_grid: N-Dimensional array of residual oil saturation values (fraction).
    :param residual_gas_saturation_grid: N-Dimensional array of residual gas saturation values (fraction).
    :param relative_permeability_params: `RelativePermeabilityParameters` object containing Corey exponents for water, oil, and gas.
    :return: Tuple of (water_relative_mobility_grid, oil_relative_mobility_grid, gas_relative_mobility_grid) in 1/(cP).
    """
    water_relative_mobility_grid = build_uniform_grid(
        grid_dimension=water_saturation_grid.shape,
        value=0.0,
        dtype=np.float32,
    )
    oil_relative_mobility_grid = build_uniform_grid(
        grid_dimension=water_saturation_grid.shape,
        value=0.0,
        dtype=np.float32,
    )
    gas_relative_mobility_grid = build_uniform_grid(
        grid_dimension=water_saturation_grid.shape,
        value=0.0,
        dtype=np.float32,
    )

    for indices in itertools.product(*map(range, water_saturation_grid.shape)):
        # Get current saturations for the cell
        water_saturation = water_saturation_grid[indices]
        oil_saturation = oil_saturation_grid[indices]
        gas_saturation = gas_saturation_grid[indices]

        # Get viscosities for the cell
        # Ensure viscosities are not zero to avoid division by zero
        water_viscosity = np.maximum(water_viscosity_grid[indices], 1e-7)
        oil_viscosity = np.maximum(oil_viscosity_grid[indices], 1e-7)
        gas_viscosity = np.maximum(gas_viscosity_grid[indices], 1e-7)

        # Get cell-specific rock properties
        irreducible_water_saturation = irreducible_water_saturation_grid[indices]
        residual_oil_saturation = residual_oil_saturation_grid[indices]
        residual_gas_saturation = residual_gas_saturation_grid[indices]

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
        water_relative_mobility_grid[indices] = (
            water_relative_permeability / water_viscosity
        )
        oil_relative_mobility_grid[indices] = oil_relative_permeability / oil_viscosity
        gas_relative_mobility_grid[indices] = gas_relative_permeability / gas_viscosity

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


def build_oil_api_gravity_grid(
    oil_specific_gravity_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil API gravities based on oil density.

    The API gravity is computed using the formula:
    API Gravity = (141.5 / Specific Gravity) - 131.5

    :param oil_specific_gravity_grid: N-Dimensional array of oil specific gravity values (dimensionless).
    :return: N-Dimensional array of oil API gravity values (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_api_gravity, otypes=[np.float64])(
        oil_specific_gravity_grid
    )


def build_oil_specific_gravity_grid(
    oil_density_grid: NDimensionalGrid[NDimension],
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    oil_compressibility_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil specific gravities.

    The specific gravity is computed as the ratio of the oil density to the standard density of water (1000 kg/m³).

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param oil_density_grid: N-Dimensional array of oil density values (lbm/ft³).
    :param oil_compressibility_grid: N-Dimensional array of oil compressibility values (psi⁻¹).
    :return: N-Dimensional array of oil specific gravity (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_specific_gravity_from_density, otypes=[np.float64])(
        oil_density_grid, pressure_grid, temperature_grid, oil_compressibility_grid
    )


def build_gas_compressibility_factor_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas compressibility factors.

    The compressibility factor is computed using the gas gravity and the pressure and temperature conditions.

    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param h2s_mole_fraction: Mole fraction of H₂S in the gas mixture (default: 0.0).
    :param co2_mole_fraction: Mole fraction of CO₂ in the gas mixture (default: 0.0).
    :param n2_mole_fraction: Mole fraction of N₂ in the gas mixture (default: 0.0).
    :return: N-Dimensional array of gas compressibility factors (dimensionless) corresponding to each grid cell.
    """
    return np.vectorize(
        compute_gas_compressibility_factor,
        otypes=[np.float64],
        excluded=["h2s_mole_fraction", "co2_mole_fraction", "n2_mole_fraction"],
    )(
        pressure_grid,
        temperature_grid,
        gas_gravity_grid,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )


def build_oil_formation_volume_factor_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    oil_specific_gravity_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    gas_to_oil_ratio_grid: NDimensionalGrid[NDimension],
    oil_compressibility_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil formation volume factors.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of oil bubble point pressures (psi) corresponding to each grid cell.
    :param oil_specific_gravity_grid: N-Dimensional array of oil specific gravity values (dimensionless),
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_to_oil_ratio_grid: N-Dimensional array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param oil_compressibility_grid: N-Dimensional array of oil compressibility values (psi⁻¹) representing the compressibility of oil.
    :return: N-Dimensional array of oil formation volume factors (bbl/STB) corresponding to each grid cell.
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


def build_water_formation_volume_factor_grid(
    water_density_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water formation volume factors.

    :param water_density_grid: N-Dimensional array of water density values (lbm/ft³) representing the density of water at reservoir conditions.
    :param salinity_grid: N-Dimensional array of water salinity values (ppm of NaCl) representing the salinity of water at reservoir conditions.
    :return: N-Dimensional array of water formation volume factors (bbl/STB) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_formation_volume_factor, otypes=[np.float64])(
        water_density_grid, salinity_grid
    )


def build_gas_formation_volume_factor_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_compressibility_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas formation volume factors.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_compressibility_factor_grid: N-Dimensional array of gas compressibility factor values (dimensionless) representing the compressibility of gas.
    :return: N-Dimensional array of gas formation volume factors (ft³/SCF) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_formation_volume_factor, otypes=[np.float64])(
        pressure_grid, temperature_grid, gas_compressibility_factor_grid
    )


def build_gas_to_oil_ratio_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    gor_at_bubble_point_pressure_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of solution gas-to-oil ratios.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.

    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of bubble point pressures (psi) corresponding to each grid cell.
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param oil_api_gravity_grid: N-Dimensional array of oil API gravity values (dimensionless) representing the density of oil relative to water.
    :param gor_at_bubble_point_pressure_grid: Optional N-Dimensional array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: N-Dimensional array of solution gas-to-oil ratios (SCF/STB) corresponding to each grid cell.
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


def build_oil_bubble_point_pressure_grid(
    gas_gravity_grid: NDimensionalGrid[NDimension],
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_to_oil_ratio_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil bubble point pressures.

    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param oil_api_gravity_grid: N-Dimensional array of API gravity values.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gor_at_bubble_point_pressure_grid: N-Dimensional array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: N-Dimensional array of oil bubble point pressures (psi) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_bubble_point_pressure, otypes=[np.float64])(
        gas_gravity_grid,
        oil_api_gravity_grid,
        temperature_grid,
        gas_to_oil_ratio_grid,
    )


def build_gas_solubility_in_water_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
    gas: str = "methane",
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas solubilities.

    The solubility is computed using the Henry's law constant for the specific gas in water.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param salinity_grid: N-Dimensional array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas: Type of gas dissolved in water (default: "methane"). Can be CO₂, N₂, etc.
    :return: N-Dimensional array of gas solubility in water (SCF/STB) corresponding to each grid cell.
    """
    return np.vectorize(
        compute_gas_solubility_in_water, otypes=[np.float64], excluded=["gas"]
    )(pressure_grid, temperature_grid, salinity_grid, gas=gas)


def build_water_bubble_point_pressure_grid(
    temperature_grid: NDimensionalGrid[NDimension],
    gas_solubility_in_water_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
    gas: str = "methane",
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water bubble point pressures.

    The bubble point pressure is computed using the gas solubility in water and the salinity of the water.

    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_solubility_in_water_grid: N-Dimensional array of gas solubility in water (SCF/STB) at bubble point pressure corresponding to each grid cell.
    :param salinity_grid: N-Dimensional array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas: Type of gas dissolved in water (default: "methane"). Can be CO₂, N₂, etc.
    :return: N-Dimensional array of water bubble point pressures (psi) corresponding to each grid cell.
    """
    return np.vectorize(
        compute_water_bubble_point_pressure, otypes=[np.float64], excluded=["gas"]
    )(
        temperature_grid,
        gas_solubility_in_water_grid,
        salinity_grid,
        gas=gas,
    )


def build_oil_viscosity_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    oil_specific_gravity_grid: NDimensionalGrid[NDimension],
    gas_to_oil_ratio_grid: NDimensionalGrid[NDimension],
    gor_at_bubble_point_pressure_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil viscosities.

    The viscosity is computed using the oil specific gravity, gas gravity, and the gas-to-oil ratio.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of bubble point pressures (psi) corresponding to each grid cell.
    :param oil_specific_gravity_grid: N-Dimensional array of oil specific gravity values (dimensionless).
    :param gas_to_oil_ratio_grid: N-Dimensional array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param gor_at_bubble_point_pressure_grid: Optional N-Dimensional array of gas-to-oil ratios at bubble point pressure (SCF/STB).
    :return: N-Dimensional array of oil viscosities (cP) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_viscosity, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        bubble_point_pressure_grid,
        oil_specific_gravity_grid,
        gas_to_oil_ratio_grid,
        gor_at_bubble_point_pressure_grid,
    )


def build_gas_molecular_weight_grid(
    gas_gravity_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas molecular weights.

    The molecular weight is computed using the formula:
    Molecular Weight = Gas Gravity * 28.9644 g/mol (molecular weight of air)

    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :return: N-Dimensional array of gas molecular weights (g/mol) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_molecular_weight, otypes=[np.float64])(
        gas_gravity_grid
    )


def build_gas_viscosity_grid(
    temperature_grid: NDimensionalGrid[NDimension],
    gas_density_grid: NDimensionalGrid[NDimension],
    gas_molecular_weight_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas viscosities.


    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_density_grid: N-Dimensional array of gas density values (lbm/ft³) representing the density of gas in each grid cell.
    :param gas_molecular_weight_grid: N-Dimensional array of gas molecular weight values (g/mol) representing the molecular weight of gas in each grid cell.
    :return: N-Dimensional array of gas viscosities (cP) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_viscosity, otypes=[np.float64])(
        temperature_grid,
        gas_density_grid,
        gas_molecular_weight_grid,
    )


def build_water_viscosity_grid(
    temperature_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
    pressure_grid: typing.Optional[NDimensionalGrid[NDimension]] = None,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water/brine viscosities.

    The viscosity is computed using the pressure, temperature, and salinity conditions.

    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param salinity_grid: N-Dimensional array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param pressure_grid: Optional N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :return: N-Dimensional array of water/brine viscosities (cP) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_viscosity, otypes=[np.float64])(
        temperature_grid, salinity_grid, pressure_grid
    )


def build_oil_compressibility_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    gor_at_bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    gas_formation_volume_factor_grid: NDimensionalGrid[NDimension],
    oil_formation_volume_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of oil compressibilities.

    The compressibility is computed using the oil API gravity, gas gravity, and the gas-to-oil ratio.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of bubble point pressures (psi) corresponding to each grid cell.
    :param oil_api_gravity_grid: N-Dimensional array of oil API gravity values (dimensionless).
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_to_oil_ratio_grid: N-Dimensional array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :return: N-Dimensional array of oil compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    return np.vectorize(compute_oil_compressibility, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        bubble_point_pressure_grid,
        oil_api_gravity_grid,
        gas_gravity_grid,
        gor_at_bubble_point_pressure_grid,
        gas_formation_volume_factor_grid,
        oil_formation_volume_factor_grid,
    )


def build_gas_compressibility_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    gas_compressibility_factor_grid: typing.Optional[
        NDimensionalGrid[NDimension]
    ] = None,
    h2s_mole_fraction: float = 0.0,
    co2_mole_fraction: float = 0.0,
    n2_mole_fraction: float = 0.0,
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas compressibilities.

    The compressibility is computed using the gas gravity and the pressure and temperature conditions.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_compressibility_factor_grid: Optional N-Dimensional array of precalculated gas compressibility factors (dimensionless).
        If provided, it will be used directly; otherwise, it will be computed.
    :param h2s_mole_fraction: Mole fraction of H₂S in the gas mixture (default: 0.0).
    :param co2_mole_fraction: Mole fraction of CO₂ in the gas mixture (default: 0.0).
    :param n2_mole_fraction: Mole fraction of N₂ in the gas mixture (default: 0.0).
    :return: N-Dimensional array of gas compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    return np.vectorize(
        compute_gas_compressibility,
        otypes=[np.float64],
        excluded=["h2s_mole_fraction", "co2_mole_fraction", "n2_mole_fraction"],
    )(
        pressure_grid,
        temperature_grid,
        gas_gravity_grid,
        gas_compressibility_factor_grid,
        h2s_mole_fraction=h2s_mole_fraction,
        co2_mole_fraction=co2_mole_fraction,
        n2_mole_fraction=n2_mole_fraction,
    )


def build_gas_free_water_formation_volume_factor_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas free water formation volume factors.

    The gas free water formation volume factor is computed using the pressure and temperature conditions.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :return: N-Dimensional array of gas free water formation volume factors (bbl/STB) corresponding to each grid cell.
    """
    return np.vectorize(
        compute_gas_free_water_formation_volume_factor, otypes=[np.float64]
    )(pressure_grid, temperature_grid)


def build_water_compressibility_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    bubble_point_pressure_grid: NDimensionalGrid[NDimension],
    gas_formation_volume_factor_grid: NDimensionalGrid[NDimension],
    gas_solubility_in_water_grid: NDimensionalGrid[NDimension],
    gas_free_water_formation_volume_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water compressibilities.

    The compressibility is computed using the gas formation volume factor, gas solubility in water, and gas free water formation volume factor.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param bubble_point_pressure_grid: N-Dimensional array of water bubble point pressures (psi) corresponding to each grid cell.
    :param gas_formation_volume_factor_grid: N-Dimensional array of gas formation volume factors (ft³/SCF) corresponding to each grid cell.
    :param gas_solubility_in_water_grid: N-Dimensional array of gas solubility in water (SCF/STB) corresponding to each grid cell.
    :param gas_free_water_formation_volume_factor_grid: N-Dimensional array of gas free water formation volume factors (bbl/STB).
    :return: N-Dimensional array of water compressibilities (psi⁻¹) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_compressibility, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        bubble_point_pressure_grid,
        gas_formation_volume_factor_grid,
        gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid,
    )


def build_live_oil_density_grid(
    oil_api_gravity_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    gas_to_oil_ratio_grid: NDimensionalGrid[NDimension],
    formation_volume_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of live oil densities.

    The density is computed using the oil API gravity, gas gravity, and the gas-to-oil ratio.

    :param oil_api_gravity_grid: N-Dimensional array of oil API gravity values (dimensionless).
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_to_oil_ratio_grid: N-Dimensional array of gas-to-oil ratio values (SCF/STB) representing the ratio of gas to oil in the reservoir.
    :param formation_volume_factor_grid: N-Dimensional array of formation volume factors (bbl/STB).
    :return: N-Dimensional array of oil densities (lbm/ft³) corresponding to each grid cell.
    """
    return np.vectorize(compute_live_oil_density, otypes=[np.float64])(
        oil_api_gravity_grid,
        gas_gravity_grid,
        gas_to_oil_ratio_grid,
        formation_volume_factor_grid,
    )


def build_gas_density_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    gas_compressibility_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of gas densities.

    The density is computed using the gas gravity and the pressure and temperature conditions.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param gas_compressibility_factor_grid: N-Dimensional array of gas compressibility factor values (dimensionless)
        representing the compressibility of gas.
    :return: N-Dimensional array of gas densities (lbm/ft³) corresponding to each grid cell.
    """
    return np.vectorize(compute_gas_density, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        gas_gravity_grid,
        gas_compressibility_factor_grid,
    )


def build_water_density_grid(
    pressure_grid: NDimensionalGrid[NDimension],
    temperature_grid: NDimensionalGrid[NDimension],
    gas_gravity_grid: NDimensionalGrid[NDimension],
    salinity_grid: NDimensionalGrid[NDimension],
    gas_solubility_in_water_grid: NDimensionalGrid[NDimension],
    gas_free_water_formation_volume_factor_grid: NDimensionalGrid[NDimension],
) -> NDimensionalGrid[NDimension]:
    """
    Computes a N-Dimensional grid of water/brine densities.

    The density is computed using the gas gravity, salinity, and gas solubility in water.

    :param pressure_grid: N-Dimensional array of pressure values (psi) representing the pressure distribution
        of the reservoir across the grid cells.
    :param temperature_grid: N-Dimensional array of temperature values (°F) representing the temperature distribution
        of the reservoir across the grid cells.
    :param gas_gravity_grid: N-Dimensional array of gas gravity values (dimensionless) representing the density of gas relative to air.
    :param salinity_grid: N-Dimensional array of salinity values (ppm) representing the salinity of water in each grid cell.
    :param gas_solubility_in_water_grid: N-Dimensional array of gas solubility in water (SCF/STB).
    :param gas_free_water_formation_volume_factor_grid: N-Dimensional array of gas free water formation volume factors (bbl/STB).
    :return: N-Dimensional array of water/brine densities (lbm/ft³) corresponding to each grid cell.
    """
    return np.vectorize(compute_water_density, otypes=[np.float64])(
        pressure_grid,
        temperature_grid,
        gas_gravity_grid,
        salinity_grid,
        gas_solubility_in_water_grid,
        gas_free_water_formation_volume_factor_grid,
    )
