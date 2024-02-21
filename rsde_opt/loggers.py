import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ParticleSystemLogger:
    def __init__(self, particle_system):
        self.path = torch.empty([0, particle_system.num_particles, particle_system.dim])
        self.consensus_path = torch.empty([0, particle_system.dim])
        self.objective = particle_system.objective

    def log_state(self, state, consensus):
        self.path = torch.cat([self.path, state.unsqueeze(0)])
        self.consensus_path = torch.cat([self.consensus_path, consensus.unsqueeze(0)])

    def animate_particles(self):
        fig, ax = plt.subplots()
        scatter = ax.scatter([], [], animated=True, s=1)

        def init():
            ax.set_xlim(self.path[:, :, 0].min(), self.path[:, :, 0].max())
            ax.set_ylim(self.path[:, :, 1].min(), self.path[:, :, 1].max())
            return scatter,

        def update(frame):
            x_data = self.path[frame, :, 0].numpy()
            y_data = self.path[frame, :, 1].numpy()
            scatter.set_offsets(np.column_stack([x_data, y_data]))
            return scatter,
        frames = self.path.shape[0]
        interval = 2000 / frames
        ani = FuncAnimation(fig, update, frames=frames,
                            init_func=init, blit=True, interval=interval)

        plt.show()