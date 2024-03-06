import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class ParticleSystemLogger:
    def __init__(self, particle_system):
        self.path = torch.empty([0, particle_system.num_particles, particle_system.dim])
        self.consensus_path = torch.empty([0, particle_system.dim])
        self.objective = particle_system.objective

    def log_state(self, state, consensus):
        self.path = torch.cat([self.path, state.unsqueeze(0)])
        self.consensus_path = torch.cat([self.consensus_path, consensus.unsqueeze(0)])

    def animate_particles(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(self.path[:, :, 0].min(), self.path[:, :, 0].max())
            ax.set_ylim(self.path[:, :, 1].min(), self.path[:, :, 1].max())
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
            x_data = self.path[frame, :, 0].numpy()
            y_data = self.path[frame, :, 1].numpy()
            scatter.set_offsets(np.column_stack([x_data, y_data]))

            # Update consensus position
            consensus_x = self.consensus_path[frame, 0].numpy()
            consensus_y = self.consensus_path[frame, 1].numpy()
            consensus_scatter.set_offsets(np.array([[consensus_x, consensus_y]]))

            return scatter, consensus_scatter,

        frames = self.path.shape[0]
        interval = 2000 / frames
        ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval)
        plt.show()
        return ani
