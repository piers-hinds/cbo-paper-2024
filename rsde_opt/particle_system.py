import torch
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ParticleSystem:
    objective: Callable
    initial_state: Callable
    alpha: float
    beta: Callable
    sigma: Callable
    dim: int
    num_particles: int
    step_size: float
    h: torch.tensor = field(init=False)
    state: torch.Tensor = field(init=False)
    device: str = 'cpu'

    def __post_init__(self):
        self.state = self.initial_state(self.num_particles).to(self.device)
        self.t = torch.tensor(0., device=self.device)
        self.h = torch.tensor(self.step_size, device=self.device)

    def consensus(self, x=None):
        if x is None:
            x = self.state
        objective_values = self.objective(x)
        weights = torch.nn.functional.softmax(- self.alpha * objective_values, dim=0)
        return torch.matmul(weights, x)

    @abstractmethod
    def step(self, normals):
        pass

    def reset(self):
        self.__post_init__()


class SimpleProjectionParticleSystem(ParticleSystem):
    def __init__(self, projection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals):
        beta, sigma = self.beta(self.t), self.sigma(self.t)
        x_bar = self.consensus()
        self.state += -beta * (self.state - x_bar) * self.h + sigma * (self.state - x_bar) * normals * self.h.sqrt()
        self.projection(self.state)
        self.t += self.h
        return self.state, x_bar


class ProjectionParticleSystem(ParticleSystem):
    def __init__(self, projection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals):
        beta, sigma = self.beta(self.t), self.sigma(self.t)
        # Consensus
        objective_values = self.objective(self.state)
        weights = torch.nn.functional.softmax(- self.alpha * objective_values, dim=0)
        x_bar = torch.matmul(weights, self.state)

        # Objective function at consensus
        objective_consensus = self.objective(x_bar.unsqueeze(0)).item()

        # Kill drift if the particle is somewhere better than the consensus
        drift = torch.where((objective_values < objective_consensus).unsqueeze(-1),
                            - beta * (self.state - x_bar),
                            torch.zeros_like(self.state))

        # Projected Euler step
        self.state += drift * self.h + sigma * (self.state - x_bar) * normals * self.h.sqrt()
        self.projection(self.state)
        self.t += self.h

        return self.state, x_bar


class SimplePenaltyParticleSystem(ParticleSystem):
    def __init__(self, projection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals):
        beta, sigma = self.beta(self.t), self.sigma(self.t)
        x_bar = self.consensus()
        current_state = self.state.clone()
        self.projection(self.state)
        self.state += -beta * (current_state - x_bar) * self.h + sigma * (current_state - x_bar) * normals * self.h.sqrt()
        self.t += self.h
        return self.state, x_bar


class UnconstrainedParticleSystem(ParticleSystem):
    def __init__(self, penalty_function, penalty_parameter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty_function = penalty_function
        self.penalty_parameter = penalty_parameter
        self._objective = self.objective
        self.objective = lambda x: self._objective(x) + self.penalty_parameter * self.penalty_function(x)

    def step(self, normals):
        beta, sigma = self.beta(self.t), self.sigma(self.t)
        x_bar = self.consensus()
        self.state += -beta * (self.state - x_bar) * self.h + sigma * (self.state - x_bar) * normals * self.h.sqrt()
        self.t += self.h
        return self.state, x_bar
