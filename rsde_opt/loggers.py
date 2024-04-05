import torch


class ParticleSystemLogger:
    def __init__(self, particle_system):
        self.path = torch.empty([0, particle_system.num_particles, particle_system.dim])
        self.consensus_path = torch.empty([0, particle_system.dim])
        self.objective = particle_system.objective

    def log_state(self, state, consensus):
        self.path = torch.cat([self.path, state.unsqueeze(0)])
        self.consensus_path = torch.cat([self.consensus_path, consensus.unsqueeze(0)])


class ExperimentLogger:
    def __init__(self, num_experiments):
        self.num_experiments = num_experiments
        self.consensus_paths = [torch.empty(0) for _ in range(num_experiments)]
        self.objective_values = [torch.empty(0) for _ in range(num_experiments)]

    def log_consensus(self, experiment_index, consensus):
        self.consensus_paths[experiment_index] = torch.cat([self.consensus_paths[experiment_index],
                                                            consensus.unsqueeze(0)])

    def log_objective(self, experiment_index, objective_value):
        self.objective_values[experiment_index] = torch.cat([self.objective_values[experiment_index],
                                                             objective_value])
