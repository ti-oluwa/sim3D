import typing
from functools import partial
from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation

from sim2D.typing import (
    ArrayLike,
    DiscretizationMethod,
    FluidMiscibility,
    InjectionFluid,
    TwoDimensionalGrid,
)
from sim2D.models import TwoDimensionalReservoirModel
from sim2D.properties import is_CoolProp_supported_fluid
from sim2D.grids import (
    build_2D_uniform_grid,
    build_2D_injection_grid,
    build_2D_production_grid,
    build_2D_fluid_viscosity_grid,
    build_miscible_viscosity_grid,
)
from sim2D.boundary_conditions import BoundaryConditions
from sim2D.flow_evolution import (
    compute_adaptive_pressure_evolution,
    compute_explicit_pressure_evolution,
    compute_implicit_pressure_evolution,
    compute_saturation_evolution,
)


@dataclass(frozen=True, slots=True)
class ModelTimeState:
    """
    Represents the state of the reservoir model at a specific time step during a simulation.
    """

    time: float
    """The time in seconds taken to reach this state."""
    injected_fluid: typing.Optional[InjectionFluid]
    """The type of fluid injected into the reservoir."""
    pressure_grid: TwoDimensionalGrid
    """2D numpy array representing the pressure distribution in the reservoir (Pa)."""
    injected_fluid_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the saturation distribution of the injected fluid in the reservoir (fraction)."""
    displaced_fluid_saturation_grid: TwoDimensionalGrid
    """2D numpy array representing the displaced (reservoir) fluid saturation distribution in the reservoir (Pa)."""
    displaced_fluid_viscosity_grid: TwoDimensionalGrid
    """2D numpy array representing the displaced (reservoir) fluid viscosity distribution in the reservoir (Pa.s)."""


class TwoDimensionalModelSimulator:
    """A simulator for two-dimensional reservoir models."""

    def __init__(
        self,
        model: TwoDimensionalReservoirModel,
        pressure_decay_constant: float = 1e-8,
        saturation_mixing_factor: float = 0.5,
        diffusion_number_threshold: float = 0.24,
        fluid_miscibility: typing.Optional[FluidMiscibility] = None,
    ) -> None:
        """
        Initializes the simulator with a reservoir model and simulation parameters.

        :param model: The two-dimensional reservoir model to simulate.
        :param pressure_decay_constant: Constant used in pressure decay calculations.
        :param saturation_mixing_factor: Factor used to mix fluid saturations during simulation.
        :param diffusion_number_threshold: Threshold for diffusion number to determine the discretization method.
        :param fluid_miscibility: Optional fluid miscibility properties for the simulation.
            If not provided, defaults to None, indicating no miscibility effects.
        """
        self.model = model
        self.fluid_miscibility: typing.Optional[FluidMiscibility] = fluid_miscibility
        self.pressure_decay_constant = pressure_decay_constant
        self.saturation_mixing_factor = saturation_mixing_factor
        self.diffusion_number_threshold = diffusion_number_threshold

    def run_simulation(
        self,
        num_of_time_steps: int,
        time_step_size: float,
        boundary_conditions: BoundaryConditions,
        producers_positions: typing.Optional[ArrayLike[typing.Tuple[int, int]]] = None,
        production_rates: typing.Optional[ArrayLike[float]] = None,
        injectors_positions: typing.Optional[ArrayLike[typing.Tuple[int, int]]] = None,
        injection_rates: typing.Optional[ArrayLike[float]] = None,
        injected_fluid: typing.Optional[InjectionFluid] = None,
        discretization_method: DiscretizationMethod = "adaptive",
        capture_interval: int = 1,
    ) -> typing.List[ModelTimeState]:
        if injected_fluid is not None and not is_CoolProp_supported_fluid(
            injected_fluid
        ):
            raise ValueError(
                f"Injected fluid '{injected_fluid}' is not supported. Provide a valid fluid name supported by CoolProp."
            )

        model = self.model
        cell_dimension = model.cell_dimension
        reservoir_pressure_grid = model.fluid_properties.pressure_grid.copy()
        reservoir_fluid_saturation_grid = (
            model.fluid_properties.fluid_saturation_grid.copy()
        )
        permeability_grid = model.rock_properties.permeability_grid.copy()
        porosity_grid = model.rock_properties.porosity_grid.copy()
        temperature_grid = model.temperature_grid.copy()
        rock_compressibility = model.rock_properties.compressibility
        reservoir_fluid_viscosity_grid = model.fluid_properties.fluid_viscosity_grid
        unit_saturation_grid = build_2D_uniform_grid(model.grid_dimension, value=1.0)
        injected_fluid_saturation_grid = (
            unit_saturation_grid - reservoir_fluid_saturation_grid
        )  # Assume saturation of injected fluid is '1 - reservoir_fluid_saturation'
        fluid_miscibility = self.fluid_miscibility

        # Build grids for injection and production
        if injectors_positions:
            injection_grid = build_2D_injection_grid(
                model.grid_dimension,
                injectors_positions=injectors_positions,
                injection_rates=injection_rates or [],
            )
        else:
            # No injection
            injection_grid = build_2D_uniform_grid(model.grid_dimension, value=0.0)

        if producers_positions:
            production_grid = build_2D_production_grid(
                model.grid_dimension,
                producers_positions=producers_positions,
                production_rates=production_rates or [],
            )
        else:
            # No production
            production_grid = build_2D_uniform_grid(model.grid_dimension, value=0.0)

        boundary_conditions = boundary_conditions or BoundaryConditions()
        if (method := discretization_method.lower()) == "adaptive":
            compute_pressure_evolution = partial(
                compute_adaptive_pressure_evolution,
                diffusion_number_threshold=self.diffusion_number_threshold,
            )
        elif method == "implicit":
            compute_pressure_evolution = compute_implicit_pressure_evolution
        else:
            compute_pressure_evolution = compute_explicit_pressure_evolution

        current_pressure_grid = reservoir_pressure_grid
        current_injected_fluid_saturation_grid = injected_fluid_saturation_grid
        if injected_fluid is not None:
            current_injected_fluid_viscosity_grid = build_2D_fluid_viscosity_grid(
                pressure_grid=current_pressure_grid,
                temperature_grid=temperature_grid,
                fluid=injected_fluid,
            )
        else:
            current_injected_fluid_viscosity_grid = None

        if fluid_miscibility is not None:
            current_displaced_fluid_viscosity = build_miscible_viscosity_grid(
                injected_fluid_saturation_grid=current_injected_fluid_saturation_grid,
                injected_fluid_viscosity_grid=current_injected_fluid_viscosity_grid,
                displaced_fluid_viscosity_grid=reservoir_fluid_viscosity_grid,
                fluid_misciblity=fluid_miscibility,
            )
        else:
            current_displaced_fluid_viscosity = reservoir_fluid_viscosity_grid

        model_time_states = [
            # At starting time (0)
            ModelTimeState(
                time=0.0,
                injected_fluid=injected_fluid,
                pressure_grid=reservoir_pressure_grid.copy(),
                injected_fluid_saturation_grid=injected_fluid_saturation_grid.copy(),
                displaced_fluid_saturation_grid=reservoir_fluid_saturation_grid.copy(),
                displaced_fluid_viscosity_grid=current_displaced_fluid_viscosity.copy(),
            )
        ]
        for time_step in range(1, num_of_time_steps + 1):
            # Pressure evolution
            updated_pressure_grid = compute_pressure_evolution(
                cell_dimension=cell_dimension,
                time_step_size=time_step_size,
                boundary_conditions=boundary_conditions,
                rock_compressibility=rock_compressibility,
                pressure_grid=current_pressure_grid,
                permeability_grid=permeability_grid,
                porosity_grid=porosity_grid,
                temperature_grid=temperature_grid,
                displaced_fluid_viscosity_grid=current_displaced_fluid_viscosity,
                production_grid=production_grid,
                injection_grid=injection_grid,
                injected_fluid=injected_fluid,
                injected_fluid_saturation_grid=current_injected_fluid_saturation_grid,
                injected_fluid_viscosity_grid=current_injected_fluid_viscosity_grid,
            )

            # Injected fluid viscosity evolution due to Pressure evolution (If there's injection)
            if injected_fluid is not None:
                updated_injected_fluid_viscosity_grid = build_2D_fluid_viscosity_grid(
                    pressure_grid=updated_pressure_grid,
                    temperature_grid=temperature_grid,
                    fluid=injected_fluid,
                )
            else:
                updated_injected_fluid_viscosity_grid = None

            # Displaced fluid (effective) viscosity evolution due to Injected
            # fluid viscosity evolution (If fluid miscibility occurs)
            if fluid_miscibility is not None:
                updated_displaced_fluid_viscosity = build_miscible_viscosity_grid(
                    injected_fluid_saturation_grid=current_injected_fluid_saturation_grid,
                    injected_fluid_viscosity_grid=updated_injected_fluid_viscosity_grid,
                    displaced_fluid_viscosity_grid=current_displaced_fluid_viscosity,
                    fluid_misciblity=fluid_miscibility,
                )
            else:
                updated_displaced_fluid_viscosity = current_displaced_fluid_viscosity

            # Saturation evolution
            updated_injected_fluid_saturation_grid = compute_saturation_evolution(
                cell_dimension=cell_dimension,
                time_step_size=time_step_size,
                boundary_conditions=boundary_conditions,
                pressure_grid=updated_pressure_grid,
                permeability_grid=permeability_grid,
                porosity_grid=porosity_grid,
                temperature_grid=temperature_grid,
                displaced_fluid_viscosity_grid=updated_displaced_fluid_viscosity,
                production_grid=production_grid,
                injection_grid=injection_grid,
                injected_fluid=injected_fluid,
                injected_fluid_saturation_grid=current_injected_fluid_saturation_grid,
                injected_fluid_viscosity_grid=updated_injected_fluid_viscosity_grid,
            )

            # Displaced fluid (effective) viscosity evolution due to Injected
            # fluid saturation evolution (If fluid miscibility occurs)
            if fluid_miscibility is not None:
                updated_displaced_fluid_viscosity = build_miscible_viscosity_grid(
                    injected_fluid_saturation_grid=updated_injected_fluid_saturation_grid,
                    injected_fluid_viscosity_grid=updated_injected_fluid_viscosity_grid,
                    displaced_fluid_viscosity_grid=updated_displaced_fluid_viscosity,
                    fluid_misciblity=fluid_miscibility,
                )
            else:
                updated_displaced_fluid_viscosity = updated_displaced_fluid_viscosity

            # Capture the model state at specified intervals and at the last time step
            if (time_step % capture_interval == 0) or (time_step == num_of_time_steps):
                model_time_state = ModelTimeState(
                    time=time_step * time_step_size,
                    injected_fluid=injected_fluid,
                    pressure_grid=updated_pressure_grid.copy(),
                    injected_fluid_saturation_grid=updated_injected_fluid_saturation_grid.copy(),
                    displaced_fluid_saturation_grid=(
                        unit_saturation_grid - updated_injected_fluid_saturation_grid
                    ),
                    displaced_fluid_viscosity_grid=updated_displaced_fluid_viscosity.copy(),
                )
                model_time_states.append(model_time_state)

            current_pressure_grid = updated_pressure_grid
            current_injected_fluid_saturation_grid = (
                updated_injected_fluid_saturation_grid
            )
            current_injected_fluid_viscosity_grid = (
                updated_injected_fluid_viscosity_grid
            )
            current_displaced_fluid_viscosity = updated_displaced_fluid_viscosity

        return model_time_states


def plot_model_time_state(
    model_time_state: ModelTimeState,
) -> None:
    """
    Creates a 2x2 grid of plots showing the reservoir pressure, injected fluid saturation,
    displaced fluid saturation, and displaced fluid viscosity distributions for a
    simulated reservoir model time state.

    :param model_time_state: Simulated reservoir model time state containing the grids.
    """
    pressure_grid = model_time_state.pressure_grid
    injected_saturation_grid = model_time_state.injected_fluid_saturation_grid
    displaced_saturation_grid = model_time_state.displaced_fluid_saturation_grid
    displaced_viscosity_grid = model_time_state.displaced_fluid_viscosity_grid

    total_time_in_hrs = model_time_state.time / 3600
    injected_fluid = model_time_state.injected_fluid

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2x2 array of axes for easier iteration

    # Pressure Plot
    pcm1 = axes[0].pcolormesh(pressure_grid.T, cmap="viridis", shading="auto")
    axes[0].set_title("Reservoir Pressure Distribution")
    axes[0].set_xlabel("X cell index")
    axes[0].set_ylabel("Y cell index")
    axes[0].set_aspect("equal")
    fig.colorbar(pcm1, ax=axes[0], label="Pressure (Pa)")

    # Injected Fluid Saturation Plot
    pcm2 = axes[1].pcolormesh(
        injected_saturation_grid.T,
        cmap="plasma",
        shading="auto",
        norm=Normalize(vmin=0, vmax=1),
    )
    axes[1].set_title(
        f"Injected Fluid ({injected_fluid or '--'}) Saturation Distribution".strip()
    )
    axes[1].set_xlabel("X cell index")
    axes[1].set_ylabel("Y cell index")
    axes[1].set_aspect("equal")
    fig.colorbar(pcm2, ax=axes[1], label="Saturation")

    # Displaced Fluid Saturation Plot
    pcm3 = axes[2].pcolormesh(
        displaced_saturation_grid.T,
        cmap="cividis",  # Using a different colormap for distinction
        shading="auto",
        norm=Normalize(vmin=0, vmax=1),
    )
    axes[2].set_title("Reservoir (Displaced) Fluid Saturation Distribution")
    axes[2].set_xlabel("X cell index")
    axes[2].set_ylabel("Y cell index")
    axes[2].set_aspect("equal")
    fig.colorbar(pcm3, ax=axes[2], label="Saturation")

    # Displaced Fluid Viscosity Plot
    # It's good practice to set vmin/vmax for viscosity if you have a known range
    # Otherwise, matplotlib will auto-scale each time, which can make comparison difficult.
    # For now, let's keep it auto-scaled or infer from data.
    # We'll calculate global min/max if possible for the animation.

    # Calculate a suitable range for viscosity if you want fixed colorbar in plot_model_time_state
    # If not, it will auto-scale, which is fine for single plots.
    viscosity_min = displaced_viscosity_grid.min()
    viscosity_max = displaced_viscosity_grid.max()

    pcm4 = axes[3].pcolormesh(
        displaced_viscosity_grid.T,
        cmap="magma",  # Another distinct colormap
        shading="auto",
        norm=Normalize(
            vmin=viscosity_min, vmax=viscosity_max
        ),  # Optional: set fixed scale
    )
    axes[3].set_title("Reservoir (Displaced) Fluid Viscosity Distribution")
    axes[3].set_xlabel("X cell index")
    axes[3].set_ylabel("Y cell index")
    axes[3].set_aspect("equal")
    fig.colorbar(pcm4, ax=axes[3], label="Viscosity (Pa.s)")

    fig.suptitle(
        f"Reservoir Simulation at {total_time_in_hrs:.2f} hr(s)",
        fontsize=16,
    )
    plt.show()


def animate_model_states(
    model_states: typing.Sequence[ModelTimeState],
    interval_ms: int = 100,
    save_path: typing.Optional[str] = None,
) -> None:
    """
    Animates the reservoir pressure, injected fluid saturation,
    displaced fluid saturation, and displaced fluid viscosity distributions
    over multiple time states.

    :param model_states: A list of ModelTimeState objects,
        each representing a snapshot of the reservoir at a time step.
    :param interval_ms: Delay between frames in milliseconds.
    :param save_path: Optional file path to save the animation (e.g., "simulation.gif", "simulation.mp4").
        Requires ffmpeg or imagemagick for MP4/GIF respectively.
    """
    if not model_states:
        print("No model states provided for animation.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()  # Flatten for easy indexing

    initial_state = model_states[0]

    # Determine global min/max for pressure and viscosity across all states for consistent colorbars
    all_pressures = np.array([s.pressure_grid for s in model_states])
    all_viscosities = np.array(
        [s.displaced_fluid_viscosity_grid for s in model_states]
    )  # Note: Typo `viscosiy`

    min_pressure, max_pressure = all_pressures.min(), all_pressures.max()
    min_viscosity, max_viscosity = all_viscosities.min(), all_viscosities.max()

    # Pressure Plot
    pcm1 = axes[0].pcolormesh(
        initial_state.pressure_grid.T,
        cmap="viridis",
        shading="auto",
        norm=Normalize(vmin=min_pressure, vmax=max_pressure),  # Fixed color scale
    )
    axes[0].set_title("Reservoir Pressure Distribution")
    axes[0].set_xlabel("X cell index")
    axes[0].set_ylabel("Y cell index")
    axes[0].set_aspect("equal")
    fig.colorbar(pcm1, ax=axes[0], label="Pressure (Pa)")

    # Injected Fluid Saturation Plot
    pcm2 = axes[1].pcolormesh(
        initial_state.injected_fluid_saturation_grid.T,
        cmap="plasma",
        shading="auto",
        norm=Normalize(vmin=0, vmax=1),
    )
    axes[1].set_title(
        f"Injected Fluid ({initial_state.injected_fluid or '--'}) Saturation Distribution".strip()
    )
    axes[1].set_xlabel("X cell index")
    axes[1].set_ylabel("Y cell index")
    axes[1].set_aspect("equal")
    fig.colorbar(pcm2, ax=axes[1], label="Saturation")

    # Displaced Fluid Saturation Plot
    pcm3 = axes[2].pcolormesh(
        initial_state.displaced_fluid_saturation_grid.T,
        cmap="cividis",
        shading="auto",
        norm=Normalize(vmin=0, vmax=1),
    )
    axes[2].set_title("Reservoir (Displaced) Fluid Saturation Distribution")
    axes[2].set_xlabel("X cell index")
    axes[2].set_ylabel("Y cell index")
    axes[2].set_aspect("equal")
    fig.colorbar(pcm3, ax=axes[2], label="Saturation")

    # Displaced Fluid Viscosity Plot
    pcm4 = axes[3].pcolormesh(
        initial_state.displaced_fluid_viscosity_grid.T,
        cmap="magma",
        shading="auto",
        norm=Normalize(vmin=min_viscosity, vmax=max_viscosity),  # Fixed color scale
    )
    axes[3].set_title("Reservoir (Displaced) Fluid Viscosity Distribution")
    axes[3].set_xlabel("X cell index")
    axes[3].set_ylabel("Y cell index")
    axes[3].set_aspect("equal")
    fig.colorbar(pcm4, ax=axes[3], label="Viscosity (Pa.s)")

    # Overall title
    current_time_hrs = initial_state.time / 3600
    fig_suptitle = fig.suptitle(
        f"Reservoir Simulation at {current_time_hrs:.2f} hr(s)",
        fontsize=16,
    )

    def update(frame_index: int):
        """
        Updates the plot data for each frame of the animation.
        """
        state = model_states[frame_index]

        # Update data for all four plots
        pcm1.set_array(state.pressure_grid.T.ravel())
        pcm2.set_array(state.injected_fluid_saturation_grid.T.ravel())
        pcm3.set_array(state.displaced_fluid_saturation_grid.T.ravel())
        pcm4.set_array(state.displaced_fluid_viscosity_grid.T.ravel())

        # Update titles
        current_time_hrs = state.time / 3600
        fig_suptitle.set_text(f"Reservoir Simulation at {current_time_hrs:.2f} hr(s)")
        axes[1].set_title(
            f"Injected Fluid ({state.injected_fluid or '--'}) Saturation Distribution".strip()
        )

        # Return all artists that were modified
        return (
            pcm1,
            pcm2,
            pcm3,
            pcm4,
            fig_suptitle,
            axes[1].title,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(model_states),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        Writer = (
            animation.writers["ffmpeg"]
            if save_path.endswith(".mp4")
            else animation.writers["pillow"]
        )
        writer = Writer(
            fps=1000 // interval_ms, metadata=dict(artist="Sim2D"), bitrate=1800
        )
        anim.save(save_path, writer=writer, dpi=100)
        print("Animation saved.")

    plt.show()
    plt.close(fig)
