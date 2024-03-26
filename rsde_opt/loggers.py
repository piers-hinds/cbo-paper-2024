import torch


class ParticleSystemLogger:
    def __init__(self, particle_system):
        self.path = torch.empty([0, particle_system.num_particles, particle_system.dim])
        self.consensus_path = torch.empty([0, particle_system.dim])
        self.objective = particle_system.objective

    def log_state(self, state, consensus):
        self.path = torch.cat([self.path, state.unsqueeze(0)])
        self.consensus_path = torch.cat([self.consensus_path, consensus.unsqueeze(0)])
