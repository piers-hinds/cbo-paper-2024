import torch
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ParticleSystem:
    objective: Callable
    initial_state: Callable
    alpha: float
    beta: float
    sigma: float
    dim: int
    num_particles: int
    step_size: float
    h: torch.tensor = field(init=False)
    state: torch.Tensor = field(init=False)
    _objective: Callable = field(init=False)
    device: str = 'cpu'

    def __post_init__(self):
        self._objective = self.objective
        self.state = self.initial_state(self.num_particles).to(self.device)
        self.t = torch.tensor(0., device=self.device)
        self.h = torch.tensor(self.step_size, device=self.device)

    def consensus(self, x=None):
        if x is None:
            x = self.state
        objective_values = self._objective(x)
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
        x_bar = self.consensus()
        self.state = self.state - self.beta * (self.state - x_bar) * self.h + self.sigma * (
                    self.state - x_bar) * normals * self.h.sqrt()
        self.projection(self.state)
        self.t += self.h
        return self.state, x_bar


class ProjectionParticleSystem(ParticleSystem):
    def __init__(self, projection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals):
        # Consensus
        objective_values = self._objective(self.state)
        weights = torch.nn.functional.softmax(- self.alpha * objective_values, dim=0)
        x_bar = torch.matmul(weights, self.state)

        # Objective function at consensus
        objective_consensus = self.objective(x_bar.unsqueeze(0)).item()

        # Kill drift if the particle is somewhere better than the consensus
        drift = torch.where((objective_values < objective_consensus).unsqueeze(-1),
                            - self.beta * (self.state - x_bar),
                            torch.zeros_like(self.state))

        # Projected Euler step
        self.state = self.state + drift * self.h + self.sigma * (self.state - x_bar) * normals * self.h.sqrt()
        self.projection(self.state)
        self.t += self.h

        return self.state, x_bar


class SimplePenaltyParticleSystem(ParticleSystem):
    def __init__(self, penalty_function, penalty_parameter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty_function = penalty_function
        self.penalty_parameter = penalty_parameter
        self._objective = lambda x: self.objective(x) + self.penalty_parameter * self.penalty_function(x)

    def step(self, normals):
        x_bar = self.consensus()
        self.state = self.state - self.beta * (self.state - x_bar) * self.h + self.sigma * (
                self.state - x_bar) * normals * self.h.sqrt()
        self.t += self.h
        return self.state, x_bar
