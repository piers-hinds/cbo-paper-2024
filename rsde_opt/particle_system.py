import torch
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ParticleSystem:
    """
    Base class for particle systems.

    Attributes:
        objective (Callable): Objective function to be minimized.
        initial_state (Callable): Function to generate the initial state of particles.
        alpha (float): Weighting parameter for consensus calculation.
        beta (Callable[[torch.Tensor], torch.Tensor]): Function of time controlling attraction strength.
        sigma (Callable[[torch.Tensor], torch.Tensor]): Function of time controlling diffusion intensity.
        dim (int): Dimensionality of the particles.
        num_particles (int): Number of particles in the system.
        step_size (float): Step size for the update rule.
        h (torch.Tensor): Time step tensor (initialized post-creation).
        state (torch.Tensor): Current state of the particles (initialized post-creation).
        device (str): Device to perform calculations on (default: 'cpu').
    """
    objective: Callable[[torch.Tensor], torch.Tensor]
    initial_state: Callable[[int], torch.Tensor]
    alpha: float
    beta: Callable[[torch.Tensor], torch.Tensor]
    sigma: Callable[[torch.Tensor], torch.Tensor]
    dim: int
    num_particles: int
    step_size: float
    h: torch.Tensor = field(init=False)
    state: torch.Tensor = field(init=False)
    device: str = 'cpu'

    def __post_init__(self):
        self.state = self.initial_state(self.num_particles).to(self.device)
        self.t = torch.tensor(0., device=self.device)
        self.h = torch.tensor(self.step_size, device=self.device)

    def consensus(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the weighted consensus point based on particle positions.

        Args:
            x (torch.Tensor): Optional tensor of particle positions. If None, uses current state.

        Returns:
            torch.Tensor: Consensus point.
        """
        if x is None:
            x = self.state
        objective_values = self.objective(x)
        weights = torch.nn.functional.softmax(-self.alpha * objective_values, dim=0)
        return torch.matmul(weights, x)

    @abstractmethod
    def step(self, normals: torch.Tensor) -> torch.Tensor:
        """
        Perform one update step for the particle system.

        Args:
            normals (torch.Tensor): Sample of standard normal random variables for diffusion.

        Returns:
            torch.Tensor: Updated state and consensus.
        """
        pass

    def reset(self):
        """Reset the particle system to its initial state."""
        self.__post_init__()


class SimpleProjectionParticleSystem(ParticleSystem):
    """
    Particle system with projection to feasible regions.
    """
    def __init__(self, projection: Callable[[torch.Tensor], None], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals: torch.Tensor) -> torch.Tensor:
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)
        x_bar = self.consensus()
        self.state += -beta * (self.state - x_bar) * self.h + sigma * (self.state - x_bar) * normals * self.h.sqrt()
        self.projection(self.state)
        self.t += self.h
        return self.state, x_bar


class ProjectionParticleSystem(ParticleSystem):
    """
    Particle system with projection to feasible regions and dynamics where drift to consensus is killed if particle is
    in a better location.
    """
    def __init__(self, projection: Callable[[torch.Tensor], None], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals: torch.Tensor) -> torch.Tensor:
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)
        x_bar = self.consensus()

        objective_values = self.objective(self.state)
        objective_consensus = self.objective(x_bar.unsqueeze(0)).item()

        drift = torch.where((objective_values < objective_consensus).unsqueeze(-1),
                            -beta * (self.state - x_bar),
                            torch.zeros_like(self.state))

        self.state += drift * self.h + sigma * (self.state - x_bar) * normals * self.h.sqrt()
        self.projection(self.state)
        self.t += self.h
        return self.state, x_bar


class SimplePenaltyParticleSystem(ParticleSystem):
    """
    Particle system with penalty-based constraints.
    """
    def __init__(self, projection: Callable[[torch.Tensor], None], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals: torch.Tensor) -> torch.Tensor:
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)
        x_bar = self.consensus()
        current_state = self.state.clone()
        self.projection(self.state)
        self.state += -beta * (current_state - x_bar) * self.h + sigma * (current_state - x_bar) * normals * self.h.sqrt()
        self.t += self.h
        return self.state, x_bar


class UnconstrainedParticleSystem(ParticleSystem):
    """
    Particle system without explicit constraints. Objective function is modified to encode the constraints.
    """
    def __init__(self, penalty_function: Callable[[torch.Tensor], torch.Tensor],
                 penalty_parameter: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty_function = penalty_function
        self.penalty_parameter = penalty_parameter
        self._objective = self.objective
        self.objective = lambda x: self._objective(x) + self.penalty_parameter * self.penalty_function(x)

    def step(self, normals: torch.Tensor) -> torch.Tensor:
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)
        x_bar = self.consensus()
        self.state += -beta * (self.state - x_bar) * self.h + sigma * (self.state - x_bar) * normals * self.h.sqrt()
        self.t += self.h
        return self.state, x_bar


class RepellingParticleSystem(ParticleSystem):
    """
    Particle system with repelling dynamics and projection.
    """
    def __init__(self, projection: Callable[[torch.Tensor], None], lambda_func: Callable[[torch.Tensor], torch.Tensor], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = projection
        self.lambda_func = lambda_func

    def step(self, normals: torch.Tensor) -> torch.Tensor:
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)
        lambd = self.lambda_func(self.t)

        x_bar = self.consensus()

        attraction = -beta * (self.state - x_bar)

        pairwise_diff = self.state.unsqueeze(1) - self.state.unsqueeze(0)
        distances = torch.norm(pairwise_diff, dim=-1, keepdim=True).clamp(min=1e-8)
        repulsion = lambd * torch.sum(
            torch.exp(-0.5 * distances**2) * pairwise_diff / distances, dim=1
        )

        diffusion = sigma * (self.state - x_bar)

        self.state += (attraction + repulsion) * self.h + diffusion * normals * self.h.sqrt()
        self.projection(self.state)
        self.t += self.h
        return self.state, x_bar
