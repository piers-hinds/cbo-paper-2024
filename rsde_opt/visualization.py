import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch


def animate_particles(logger, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlim(logger.path[:, :, 0].min(), logger.path[:, :, 0].max())
        ax.set_ylim(logger.path[:, :, 1].min(), logger.path[:, :, 1].max())
    else:
        fig = ax.figure

    # Particle scatter plot
    scatter = ax.scatter([], [], animated=True, s=1, color='blue', label='Particles')

    # Consensus scatter plot, different color and labeled
    consensus_scatter = ax.scatter([], [], animated=True, s=20, color='red', label='Consensus')

    def init():
        ax.legend(loc='upper left')
        return scatter, consensus_scatter,

    def update(frame):
        x_data = logger.path[frame, :, 0].numpy()
        y_data = logger.path[frame, :, 1].numpy()
        scatter.set_offsets(np.column_stack([x_data, y_data]))

        # Update consensus position
        consensus_x = logger.consensus_path[frame, 0].numpy()
        consensus_y = logger.consensus_path[frame, 1].numpy()
        consensus_scatter.set_offsets(np.array([[consensus_x, consensus_y]]))

        return scatter, consensus_scatter,

    frames = logger.path.shape[0]
    interval = 2000 / frames
    animation = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval)
    return animation


def animate_particles_with_progress(logger, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlim(logger.path[:, :, 0].min(), logger.path[:, :, 0].max())
        ax.set_ylim(logger.path[:, :, 1].min(), logger.path[:, :, 1].max())
    else:
        fig = ax.figure

    # Particle scatter plot
    scatter = ax.scatter([], [], animated=True, s=1, color='blue', label='Particles')

    consensus_scatter = ax.scatter([], [], animated=True, s=20, color='red', label='Consensus')

    progress_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    progress_bar = progress_ax.barh(0, 0, color='green')
    progress_ax.set_xlim(0, 1)
    progress_ax.set_xticks([])
    progress_ax.set_yticks([])

    def init():
        ax.legend(loc='upper left')
        return scatter, consensus_scatter, progress_bar.patches[0]

    def update(frame):
        # Update particle positions
        x_data = logger.path[frame, :, 0].numpy()
        y_data = logger.path[frame, :, 1].numpy()
        scatter.set_offsets(np.column_stack([x_data, y_data]))

        # Update consensus position
        consensus_x = logger.consensus_path[frame, 0].numpy()
        consensus_y = logger.consensus_path[frame, 1].numpy()
        consensus_scatter.set_offsets(np.array([[consensus_x, consensus_y]]))

        # Update progress bar
        normalized_time = frame / logger.path.shape[0]
        progress_bar.patches[0].set_width(normalized_time)

        return scatter, consensus_scatter, progress_bar.patches[0]

    frames = logger.path.shape[0]
    interval = 2000 / frames  # Duration of the animation
    animation = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval, repeat=False)

    def on_finish(*_):
        plt.pause(0)

    fig.canvas.mpl_connect("close_event", on_finish)

    return animation


def create_ax(objective, feasible_region, x_lim, y_lim):
    x = np.linspace(x_lim[0], x_lim[1], 400)
    y = np.linspace(y_lim[0], y_lim[1], 400)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.stack((x_grid.flatten(), y_grid.flatten()), axis=-1)

    z_feasible = feasible_region(x_grid, y_grid)

    points_tensor = torch.tensor(points, dtype=torch.float32)
    z_objective = objective(points_tensor).numpy().reshape(x_grid.shape)

    fig, ax = plt.subplots()

    ax.contour(x_grid, y_grid, z_feasible, levels=[0, 0.001], cmap='Blues', alpha=1)

    z_objective_masked = np.where(z_feasible <= 0, z_objective, np.nan)
    ax.contourf(x_grid, y_grid, z_objective_masked, levels=20, cmap='viridis', alpha=0.5)
    return ax
