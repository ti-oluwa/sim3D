import typing
import warnings

import numpy as np

from bores._precision import get_dtype
from bores.errors import ValidationError
from bores.types import NDimension, NDimensionalGrid


__all__ = ["build_saturation_grids"]


def build_saturation_grids(
    depth_grid: NDimensionalGrid[NDimension],
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_water_grid: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas_grid: NDimensionalGrid[NDimension],
    residual_gas_saturation_grid: NDimensionalGrid[NDimension],
    porosity_grid: NDimensionalGrid[NDimension],
    use_transition_zones: bool = False,
    gas_oil_transition_thickness: float = 5.0,
    oil_water_transition_thickness: float = 5.0,
    transition_curvature_exponent: float = 2.0,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Build 3-phase (Sw, So, Sg) saturation grids from fluid contact depths.

    Supports capillary-pressure-like transitions where saturation gradients
    follow a power-law profile, producing more realistic blending than
    linear interpolation.

    **Physical Basis:**
    This implementation accounts for the fact that residual oil saturation depends
    on the displacing fluid:
    - Gas cap: Uses Sor_gas (residual oil when gas displaces oil)
    - Water zone: Uses Sor_water (residual oil when water displaces oil)
    - Oil zone: Contains mobile oil with connate water and residual gas

    **Zones (sharp):**
      - Above GOC → gas cap (gas + residual oil to gas + connate water)
      - Between GOC and OWC → oil zone (oil + connate water + residual gas)
      - Below OWC → water zone (water + residual oil to water + no gas)

    **Zones (with capillary transition):**
      - Gas-oil and oil-water interfaces vary smoothly based on
        `transition_curvature_exponent`.

      - Small exponents (< 1) → more abrupt change.
      - Larger exponents (> 1) → smoother, curved transition.

    Example

    ```python
    # Typical values for a water-wet sandstone reservoir
    thickness = np.ones((10, 5, 5)) * 10.0  # 10 ft per layer
    swc = np.full_like(thickness, 0.25)     # 25% connate water
    sor_w = np.full_like(thickness, 0.30)   # 30% residual oil to water
    sor_g = np.full_like(thickness, 0.15)   # 15% residual oil to gas
    sgr = np.full_like(thickness, 0.05)     # 5% residual gas
    porosity = np.full_like(thickness, 0.22)
    sw, so, sg = build_saturation_grids(
        thickness, goc=1500, owc=1700,
        swc, sor_w, sor_g, sgr, porosity
    )
    ```

    :param depth_grid: Depth grid for each cell (ft).
    :param gas_oil_contact: Depth separating gas and oil (ft).
    :param oil_water_contact: Depth separating oil and water (ft).
    :param connate_water_saturation_grid: Connate water saturation (Swc) - immobile water.
    :param residual_oil_saturation_water_grid: Residual oil saturation when displaced by water (Sor_w).
        Typically 0.20-0.35 for water-wet systems.
    :param residual_oil_saturation_gas_grid: Residual oil saturation when displaced by gas (Sor_g).
        Typically 0.10-0.25, usually lower than Sor_w.
    :param residual_gas_saturation_grid: Residual gas saturation (Sgr) - immobile gas.
    :param porosity_grid: Porosity array used to mask inactive cells.
    :param use_transition_zones: Enable smooth blending (default: False).
    :param gas_oil_transition_thickness: Thickness of gas-oil transition zone (ft).
    :param oil_water_transition_thickness: Thickness of oil-water transition zone (ft).
    :param transition_curvature_exponent: Controls nonlinearity of saturation gradient.
    :return: Tuple of (water_saturation, oil_saturation, gas_saturation) grids.
    :raises ValidationError: If contact depths are invalid or transitions overlap.
    """
    # Input validation
    _validate_inputs(
        depth_grid=depth_grid,
        gas_oil_contact=gas_oil_contact,
        oil_water_contact=oil_water_contact,
        connate_water_saturation=connate_water_saturation_grid,
        residual_oil_saturation_water=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
        residual_gas_saturation=residual_gas_saturation_grid,
        porosity=porosity_grid,
        use_transitions=use_transition_zones,
        gas_oil_transition_thickness=gas_oil_transition_thickness,
        oil_water_transition_thickness=oil_water_transition_thickness,
        transition_curvature_exponent=transition_curvature_exponent,
    )

    # Initialize saturation arrays
    dtype = get_dtype()
    water_saturation = np.zeros_like(depth_grid, dtype=dtype)
    oil_saturation = np.zeros_like(depth_grid, dtype=dtype)
    gas_saturation = np.zeros_like(depth_grid, dtype=dtype)
    # Cast to NDimensionalGrid to please type checker
    water_saturation = typing.cast(NDimensionalGrid[NDimension], water_saturation)
    oil_saturation = typing.cast(NDimensionalGrid[NDimension], oil_saturation)
    gas_saturation = typing.cast(NDimensionalGrid[NDimension], gas_saturation)

    # Identify active cells based on porosity
    # If porosity is NaN or <= 0, cell is inactive
    active_mask = np.isfinite(porosity_grid) & (porosity_grid > 0)
    if not use_transition_zones:
        water_saturation, oil_saturation, gas_saturation = _build_sharp_contacts(
            depth_grid=depth_grid,
            gas_oil_contact=gas_oil_contact,
            oil_water_contact=oil_water_contact,
            connate_water_saturation=connate_water_saturation_grid,
            residual_oil_saturation_water=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
            residual_gas_saturation=residual_gas_saturation_grid,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            active=active_mask,
        )
    else:
        water_saturation, oil_saturation, gas_saturation = _build_transition_zones(
            depth_grid=depth_grid,
            gas_oil_contact=gas_oil_contact,
            oil_water_contact=oil_water_contact,
            connate_water_saturation=connate_water_saturation_grid,
            residual_oil_saturation_water=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
            residual_gas_saturation=residual_gas_saturation_grid,
            gas_oil_transition_thickness=gas_oil_transition_thickness,
            oil_water_transition_thickness=oil_water_transition_thickness,
            transition_curvature_exponent=transition_curvature_exponent,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            active=active_mask,
        )

    # Normalize saturations to sum to 1.0 in active cells
    water_saturation, oil_saturation, gas_saturation = _normalize_saturations(
        water_saturation=water_saturation,
        oil_saturation=oil_saturation,
        gas_saturation=gas_saturation,
        active=active_mask,
    )
    return water_saturation, oil_saturation, gas_saturation


def _validate_inputs(
    depth_grid: np.typing.NDArray,
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation: np.typing.NDArray,
    residual_oil_saturation_water: np.typing.NDArray,
    residual_oil_saturation_gas: np.typing.NDArray,
    residual_gas_saturation: np.typing.NDArray,
    porosity: np.typing.NDArray,
    use_transitions: bool,
    gas_oil_transition_thickness: float,
    oil_water_transition_thickness: float,
    transition_curvature_exponent: float,
) -> None:
    """
    Validate all input parameters for saturation grid building.

    Performs comprehensive validation including:
    - Contact depth ordering (GOC must be above OWC)
    - Array shape consistency across all grids
    - Saturation value ranges (must be between 0 and 1)
    - Physical constraints (e.g., Swc + Sor_w <= 1.0, Swc + Sor_g <= 1.0)
    - Transition zone parameters (positive thicknesses, non-overlapping zones)
    - Oil column thickness warnings
    - Sor_w >= Sor_g check (water displacement typically leaves more residual oil)

    :param depth_grid: Depth grid array (ft).
    :param gas_oil_contact: Gas-oil contact depth (ft).
    :param oil_water_contact: Oil-water contact depth (ft).
    :param connate_water_saturation: Connate water saturation grid (fraction).
    :param residual_oil_saturation_water: Residual oil saturation to water displacement (fraction).
    :param residual_oil_saturation_gas: Residual oil saturation to gas displacement (fraction).
    :param residual_gas_saturation: Residual gas saturation grid (fraction).
    :param porosity: Porosity grid (fraction).
    :param use_transitions: Whether transition zones are enabled.
    :param gas_oil_transition_thickness: Gas-oil transition zone thickness (ft).
    :param oil_water_transition_thickness: Oil-water transition zone thickness (ft).
    :param transition_curvature_exponent: Transition curvature exponent (dimensionless).
    :raises ValidationError: If any validation check fails.
    :raises UserWarning: If oil column is very thin (< 1.0 ft) or Sor_g > Sor_w.
    """

    # Check contact depth ordering
    if gas_oil_contact >= oil_water_contact:
        raise ValidationError(
            f"Gas-oil contact ({gas_oil_contact}) must be above oil-water contact ({oil_water_contact}). "
            "Depths increase downward."
        )

    # Check array shapes match
    if not (
        depth_grid.shape
        == connate_water_saturation.shape
        == residual_oil_saturation_water.shape
        == residual_oil_saturation_gas.shape
        == residual_gas_saturation.shape
        == porosity.shape
    ):
        raise ValidationError("All grid arrays must have the same shape.")

    # Check saturation ranges
    if np.any((connate_water_saturation < 0) | (connate_water_saturation > 1)):
        raise ValidationError("Connate water saturation must be in [0, 1].")
    if np.any(
        (residual_oil_saturation_water < 0) | (residual_oil_saturation_water > 1)
    ):
        raise ValidationError("Residual oil saturation (water) must be in [0, 1].")
    if np.any((residual_oil_saturation_gas < 0) | (residual_oil_saturation_gas > 1)):
        raise ValidationError("Residual oil saturation (gas) must be in [0, 1].")
    if np.any((residual_gas_saturation < 0) | (residual_gas_saturation > 1)):
        raise ValidationError("Residual gas saturation must be in [0, 1].")

    # Check for physically impossible saturation combinations
    active = np.isfinite(porosity) & (porosity > 0)
    if np.any(
        (connate_water_saturation[active] + residual_oil_saturation_gas[active]) > 1.0
    ):
        raise ValidationError(
            "Swc + Sor_gas exceeds 1.0 in some cells (gas zone constraint)."
        )
    if np.any(
        (connate_water_saturation[active] + residual_gas_saturation[active]) > 1.0
    ):
        raise ValidationError(
            "Swc + Sgr exceeds 1.0 in some cells (oil zone constraint)."
        )
    if np.any(
        (residual_oil_saturation_water[active] + residual_gas_saturation[active]) > 1.0
    ):
        raise ValidationError(
            "Sor_water + Sgr exceeds 1.0 in some cells (not physical in any zone)."
        )

    # Warn if Sor_gas > Sor_water (unusual but not impossible)
    if np.any(
        residual_oil_saturation_gas[active] > residual_oil_saturation_water[active]
    ):
        warnings.warn(
            "Sor_gas > Sor_water in some cells. Typically, water displacement leaves "
            "more residual oil than gas displacement due to wettability effects.",
            UserWarning,
        )

    # Check transition zone parameters
    if use_transitions:
        if gas_oil_transition_thickness <= 0 or oil_water_transition_thickness <= 0:
            raise ValidationError("Transition thicknesses must be positive.")

        if transition_curvature_exponent <= 0:
            raise ValidationError("Transition curvature exponent must be positive.")

        # Check for overlapping transitions
        gas_oil_contact_bottom = gas_oil_contact + gas_oil_transition_thickness / 2
        oil_water_contact_top = oil_water_contact - oil_water_transition_thickness / 2

        if gas_oil_contact_bottom >= oil_water_contact_top:
            raise ValidationError(
                f"Transition zones overlap: GOC transition ends at {gas_oil_contact_bottom:.2f}, "
                f"but OWC transition starts at {oil_water_contact_top:.2f}. "
                "Reduce transition thicknesses or increase contact separation."
            )

        # Warn if oil column is thin
        oil_column_thickness = oil_water_contact_top - gas_oil_contact_bottom
        if oil_column_thickness < 1.0:
            warnings.warn(
                f"Oil column between transitions is very thin ({oil_column_thickness:.2f} units). "
                "Consider reducing transition thicknesses.",
                UserWarning,
            )


def _build_sharp_contacts(
    depth_grid: NDimensionalGrid[NDimension],
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation: NDimensionalGrid[NDimension],
    residual_oil_saturation_water: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas: NDimensionalGrid[NDimension],
    residual_gas_saturation: NDimensionalGrid[NDimension],
    water_saturation: NDimensionalGrid[NDimension],
    oil_saturation: NDimensionalGrid[NDimension],
    gas_saturation: NDimensionalGrid[NDimension],
    active: np.typing.NDArray,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Build saturations with sharp (discontinuous) fluid contacts.

    Creates three distinct zones with abrupt saturation changes at the contact depths.
    Uses different residual oil saturations depending on the displacing fluid.

    **Gas Zone (depth < GOC):**
        Gas has displaced oil, leaving residual oil trapped by gas:
        - Gas saturation: Sg = 1.0 - Sor_gas - Swc (mobile gas)
        - Oil saturation: So = Sor_gas (residual oil to gas displacement)
        - Water saturation: Sw = Swc (connate water)

    **Oil Zone (GOC <= depth < OWC):**
        Original oil accumulation with connate water:
        - Oil saturation: So = 1.0 - Swc - Sgr (mobile oil)
        - Water saturation: Sw = Swc (connate water)
        - Gas saturation: Sg = Sgr (residual gas, if any)

    **Water Zone (depth >= OWC):**
        Water has displaced oil, leaving residual oil trapped by water:
        - Water saturation: Sw = 1.0 - Sor_water (mobile water)
        - Oil saturation: So = Sor_water (residual oil to water displacement)
        - Gas saturation: Sg = 0.0 (no gas)

    :param depth_grid: Depth grid for each cell (ft).
    :param gas_oil_contact: Gas-oil contact depth (ft).
    :param oil_water_contact: Oil-water contact depth (ft).
    :param connate_water_saturation: Connate water saturation grid (fraction).
    :param residual_oil_saturation_water: Residual oil saturation to water (fraction).
    :param residual_oil_saturation_gas: Residual oil saturation to gas (fraction).
    :param residual_gas_saturation: Residual gas saturation grid (fraction).
    :param water_saturation: Water saturation grid to be populated (fraction).
    :param oil_saturation: Oil saturation grid to be populated (fraction).
    :param gas_saturation: Gas saturation grid to be populated (fraction).
    :param active: Boolean mask indicating active cells.
    :return: Tuple of (water_saturation, oil_saturation, gas_saturation) grids.
    """
    gas_zone = (depth_grid < gas_oil_contact) & active
    oil_zone = (
        (depth_grid >= gas_oil_contact) & (depth_grid < oil_water_contact) & active
    )
    water_zone = (depth_grid >= oil_water_contact) & active

    # Gas cap - uses Sor_gas (gas has displaced oil)
    gas_saturation[gas_zone] = (
        1.0 - residual_oil_saturation_gas[gas_zone] - connate_water_saturation[gas_zone]
    )
    oil_saturation[gas_zone] = residual_oil_saturation_gas[gas_zone]
    water_saturation[gas_zone] = connate_water_saturation[gas_zone]

    # Oil zone - original oil accumulation
    oil_saturation[oil_zone] = (
        1.0 - connate_water_saturation[oil_zone] - residual_gas_saturation[oil_zone]
    )
    water_saturation[oil_zone] = connate_water_saturation[oil_zone]
    gas_saturation[oil_zone] = residual_gas_saturation[oil_zone]

    # Water zone - uses Sor_water (water has displaced oil)
    water_saturation[water_zone] = 1.0 - residual_oil_saturation_water[water_zone]
    oil_saturation[water_zone] = residual_oil_saturation_water[water_zone]
    gas_saturation[water_zone] = 0.0
    return water_saturation, oil_saturation, gas_saturation


def _build_transition_zones(
    depth_grid: NDimensionalGrid[NDimension],
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation: NDimensionalGrid[NDimension],
    residual_oil_saturation_water: NDimensionalGrid[NDimension],
    residual_oil_saturation_gas: NDimensionalGrid[NDimension],
    residual_gas_saturation: NDimensionalGrid[NDimension],
    gas_oil_transition_thickness: float,
    oil_water_transition_thickness: float,
    transition_curvature_exponent: float,
    water_saturation: NDimensionalGrid[NDimension],
    oil_saturation: NDimensionalGrid[NDimension],
    gas_saturation: NDimensionalGrid[NDimension],
    active: np.typing.NDArray,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Build saturations with smooth capillary transition zones at fluid contacts.

    Creates five distinct zones with gradual saturation changes in transition regions.
    Uses appropriate residual oil saturations based on the displacing fluid:
    - Gas cap uses Sor_gas
    - Water zone uses Sor_water
    - Transition zones blend between appropriate end-members

    **Gas Cap Zone (depth < GOC - gas_oil_transition_thickness/2):**
        Pure gas zone where gas has displaced oil:
        - Sg = 1.0 - Sor_gas - Swc, So = Sor_gas, Sw = Swc

    **Gas-Oil Transition Zone (GOC - gas_oil_transition_thickness/2 <= depth <= GOC + gas_oil_transition_thickness/2):**
        Smooth transition from gas-dominated to oil-dominated:
        - Uses power-law weighting: w = ((depth - goc_top) / gas_oil_transition_thickness)^exponent
        - Gas saturation decreases from (1 - Sor_gas - Swc) to Sgr
        - Oil saturation increases from Sor_gas to (1 - Swc - Sgr)
        - Water saturation remains at Swc

    **Oil Zone (GOC + gas_oil_transition_thickness/2 < depth < OWC - oil_water_transition_thickness/2):**
        Pure oil zone with original accumulation:
        - So = 1.0 - Swc - Sgr, Sw = Swc, Sg = Sgr

    **Oil-Water Transition Zone (OWC - oil_water_transition_thickness/2 <= depth <= OWC + oil_water_transition_thickness/2):**
        Smooth transition from oil-dominated to water-dominated:
        - Uses power-law weighting: w = ((depth - owc_top) / oil_water_transition_thickness)^exponent
        - Water saturation increases from Swc to (1 - Sor_water)
        - Oil saturation decreases from (1 - Swc - Sgr) to Sor_water
        - Gas saturation decreases from Sgr to 0

    **Water Zone (depth > OWC + oil_water_transition_thickness/2):**
        Pure water zone where water has displaced oil:
        - Sw = 1.0 - Sor_water, So = Sor_water, Sg = 0.0

    The transition curvature exponent controls the shape of the saturation profile:
    - exponent < 1: More abrupt change (approaching sharp contact)
    - exponent = 1: Linear interpolation
    - exponent > 1: Smoother, S-curved transition (more realistic capillary pressure effect)

    :param depth_grid: Depth grid for each cell (ft).
    :param gas_oil_contact: Gas-oil contact depth (ft).
    :param oil_water_contact: Oil-water contact depth (ft).
    :param connate_water_saturation: Connate water saturation grid (fraction).
    :param residual_oil_saturation_water: Residual oil to water displacement (fraction).
    :param residual_oil_saturation_gas: Residual oil to gas displacement (fraction).
    :param residual_gas_saturation: Residual gas saturation grid (fraction).
    :param gas_oil_transition_thickness: Gas-oil transition zone thickness (ft).
    :param oil_water_transition_thickness: Oil-water transition zone thickness (ft).
    :param transition_curvature_exponent: Power-law exponent controlling transition curvature (dimensionless).
    :param water_saturation: Water saturation grid to be populated (fraction).
    :param oil_saturation: Oil saturation grid to be populated (fraction).
    :param gas_saturation: Gas saturation grid to be populated (fraction).
    :param active: Boolean mask indicating active cells.
    :return: Tuple of (water_saturation, oil_saturation, gas_saturation) grids.
    """
    # Define transition boundaries
    gas_oil_contact_top = gas_oil_contact - gas_oil_transition_thickness / 2
    gas_oil_contact_bottom = gas_oil_contact + gas_oil_transition_thickness / 2
    oil_water_contact_top = oil_water_contact - oil_water_transition_thickness / 2
    oil_water_contact_bottom = oil_water_contact + oil_water_transition_thickness / 2

    # Gas cap (above GOC transition) - uses Sor_gas
    gas_cap = (depth_grid < gas_oil_contact_top) & active
    gas_saturation[gas_cap] = (
        1.0 - residual_oil_saturation_gas[gas_cap] - connate_water_saturation[gas_cap]
    )
    oil_saturation[gas_cap] = residual_oil_saturation_gas[gas_cap]
    water_saturation[gas_cap] = connate_water_saturation[gas_cap]

    # Gas–oil transition: blend from (gas cap with Sor_gas) to (oil zone with Sgr)
    gas_oil_zone = (
        (depth_grid >= gas_oil_contact_top)
        & (depth_grid <= gas_oil_contact_bottom)
        & active
    )
    if np.any(gas_oil_zone):
        frac = (
            depth_grid[gas_oil_zone] - gas_oil_contact_top
        ) / gas_oil_transition_thickness
        weight = np.power(frac, transition_curvature_exponent)

        # Gas: from (1 - Sor_gas - Swc) to Sgr
        gas_saturation[gas_oil_zone] = (
            1.0
            - residual_oil_saturation_gas[gas_oil_zone]
            - connate_water_saturation[gas_oil_zone]
        ) * (1 - weight) + residual_gas_saturation[gas_oil_zone] * weight

        # Oil: from Sor_gas to (1 - Swc - Sgr)
        oil_saturation[gas_oil_zone] = (
            residual_oil_saturation_gas[gas_oil_zone] * (1 - weight)
            + (
                1.0
                - connate_water_saturation[gas_oil_zone]
                - residual_gas_saturation[gas_oil_zone]
            )
            * weight
        )

        # Water: remains at Swc throughout transition
        water_saturation[gas_oil_zone] = connate_water_saturation[gas_oil_zone]

    # Oil zone (between transitions) - original oil accumulation
    oil_zone = (
        (depth_grid > gas_oil_contact_bottom)
        & (depth_grid < oil_water_contact_top)
        & active
    )
    oil_saturation[oil_zone] = (
        1.0 - connate_water_saturation[oil_zone] - residual_gas_saturation[oil_zone]
    )
    water_saturation[oil_zone] = connate_water_saturation[oil_zone]
    gas_saturation[oil_zone] = residual_gas_saturation[oil_zone]

    # Oil–water transition: blend from (oil zone) to (water zone with Sor_water)
    oil_water_zone = (
        (depth_grid >= oil_water_contact_top)
        & (depth_grid <= oil_water_contact_bottom)
        & active
    )
    if np.any(oil_water_zone):
        frac = (
            depth_grid[oil_water_zone] - oil_water_contact_top
        ) / oil_water_transition_thickness
        weight = np.power(frac, transition_curvature_exponent)

        # Water: from Swc to (1 - Sor_water)
        water_saturation[oil_water_zone] = (
            connate_water_saturation[oil_water_zone] * (1 - weight)
            + (1.0 - residual_oil_saturation_water[oil_water_zone]) * weight
        )

        # Oil: from (1 - Swc - Sgr) to Sor_water
        oil_saturation[oil_water_zone] = (
            1.0
            - connate_water_saturation[oil_water_zone]
            - residual_gas_saturation[oil_water_zone]
        ) * (1 - weight) + residual_oil_saturation_water[oil_water_zone] * weight

        # Gas: from Sgr to 0
        gas_saturation[oil_water_zone] = residual_gas_saturation[oil_water_zone] * (
            1 - weight
        )

    # Water zone (below OWC transition) - uses Sor_water
    water_zone = (depth_grid > oil_water_contact_bottom) & active
    water_saturation[water_zone] = 1.0 - residual_oil_saturation_water[water_zone]
    oil_saturation[water_zone] = residual_oil_saturation_water[water_zone]
    gas_saturation[water_zone] = 0.0
    return water_saturation, oil_saturation, gas_saturation


def _normalize_saturations(
    water_saturation: NDimensionalGrid[NDimension],
    oil_saturation: NDimensionalGrid[NDimension],
    gas_saturation: NDimensionalGrid[NDimension],
    active: np.typing.NDArray,
) -> typing.Tuple[
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
    NDimensionalGrid[NDimension],
]:
    """
    Normalize saturations to ensure they sum to 1.0 in all active cells.

    This function ensures mass balance by forcing the saturation constraint:
        Sw + So + Sg = 1.0

    The normalization process:
    1. Computes total saturation: total = Sw + So + Sg
    2. Identifies valid cells: active cells with total > 0
    3. Normalizes each phase: S_phase = S_phase / total
    4. Zeros out saturations in inactive cells

    Inactive cells (porosity <= 0 or NaN) are set to zero saturation for all phases.

    :param water_saturation: Water saturation grid (fraction).
    :param oil_saturation: Oil saturation grid (fraction).
    :param gas_saturation: Gas saturation grid (fraction).
    :param active: Boolean mask indicating active cells.
    :return: Tuple of normalized (water_saturation, oil_saturation, gas_saturation) grids.
    :note: All saturations in active cells are guaranteed to sum to exactly 1.0 after normalization.
    """
    total = water_saturation + oil_saturation + gas_saturation
    valid = active & (total > 0)

    water_saturation[valid] /= total[valid]
    oil_saturation[valid] /= total[valid]
    gas_saturation[valid] /= total[valid]

    # Ensure inactive cells are zero
    water_saturation[~active] = 0.0
    oil_saturation[~active] = 0.0
    gas_saturation[~active] = 0.0
    return water_saturation, oil_saturation, gas_saturation
