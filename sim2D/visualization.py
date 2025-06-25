import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation

from sim2D.simulation import ModelTimeState



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
    displaced_saturation_grid = model_time_state.oil_saturation_grid
    displaced_viscosity_grid = model_time_state.oil_viscosity_grid

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
    fig.colorbar(pcm1, ax=axes[0], label="Pressure (psi)")

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
        [s.oil_viscosity_grid for s in model_states]
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
    fig.colorbar(pcm1, ax=axes[0], label="Pressure (psi)")

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
        initial_state.oil_saturation_grid.T,
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
        initial_state.oil_viscosity_grid.T,
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
        pcm3.set_array(state.oil_saturation_grid.T.ravel())
        pcm4.set_array(state.oil_viscosity_grid.T.ravel())

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
