import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
    plt.show()
    return animation
