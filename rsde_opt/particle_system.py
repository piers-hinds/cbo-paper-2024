import torch
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ParticleSystem:
    objective: Callable
    projection: Callable
    alpha: float
    beta: float
    sigma: float
    dim: int
    num_particles: int
    h: torch.tensor
    state: torch.Tensor = field(init=False)
    device: str = 'cpu'

    def __post_init__(self):
        self.state = torch.empty([self.num_particles, self.dim], device=self.device)
        self.t = torch.tensor(0., device=self.device)
        self.h = torch.tensor(self.h, device=self.device)

    def consensus(self, x=None):
        if x is None:
            x = self.state
        objective_values = self.objective(x)
        weights = torch.exp(- self.alpha * objective_values)
        return torch.matmul(weights, x) / weights.sum()

    @abstractmethod
    def step(self, normals):
        pass


class ProjectionParticleSystem(ParticleSystem):
    def step(self, normals):
        x_bar = self.consensus()
        self.state = self.state - self.beta * (self.state - x_bar) * self.h + self.sigma * (
                    self.state - x_bar) * normals * self.h.sqrt()
        self.t += self.h
        return self.state, x_bar

