import torch
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Tuple
from .experiment import SuccessCriterion


@dataclass
class VectorizedParticleSystem:
    """
    A vectorized implementation of a particle system for parallel optimization experiments.

    Attributes:
        objective (Callable): Objective function to be minimized.
        initial_state (Callable): Function to generate the initial state of particles.
                                  Takes the total number of particles as input.
        alpha (float): Weighting parameter for consensus calculation.
        beta (Callable[[torch.Tensor], torch.Tensor]): Function of time controlling attraction strength.
        sigma (Callable[[torch.Tensor], torch.Tensor]): Function of time controlling diffusion intensity.
        dim (int): Dimensionality of the optimization problem.
        num_particles (int): Number of particles in each experiment.
        step_size (float): Step size for the particle system updates.
        num_experiments (int): Number of parallel experiments to run.
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
    num_experiments: int
    h: torch.Tensor = field(init=False)
    state: torch.Tensor = field(init=False)
    device: str = 'cpu'

    def __post_init__(self):
        """
        Initialize the particle system, setting up the state tensor and time variables.
        """
        self.state = self.initial_state(self.num_experiments * self.num_particles).to(self.device)
        self.state = self.state.view(self.num_experiments, self.num_particles, self.dim)
        self.t = torch.tensor(0., device=self.device)
        self.h = torch.tensor(self.step_size, device=self.device)

    def consensus(self) -> torch.Tensor:
        """
        Compute the consensus points for all experiments.

        Returns:
            torch.Tensor: A tensor of shape (num_experiments, dim) representing
                          the consensus points for each experiment.
        """
        objective_values = self.objective(self.state.view(-1, self.dim)).view(self.num_experiments, self.num_particles)
        weights = torch.nn.functional.softmax(-self.alpha * objective_values, dim=1)
        weights = weights.unsqueeze(-1)
        return (weights * self.state).sum(dim=1)

    @abstractmethod
    def step(self, normals: torch.Tensor) -> torch.Tensor:
        """
        Perform one update step for all experiments.

        Args:
            normals: Random noise tensor with shape (num_experiments, num_particles, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated state tensor and consensus tensor.
        """
        pass

    def reset(self):
        """
        Reset the particle system to its initial state for all experiments.
        """
        self.__post_init__()

    def run_system(self, num_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Runs the particle system for a specified number of steps and computes the final consensus
            and the corresponding objective value.

            Args:
                num_steps (int): The number of iterations to update the particle system.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    - final_consensus (torch.Tensor): The consensus values at the end of the simulation.
                    - final_objective (torch.Tensor): The objective values corresponding to the final consensus values.
            """
        for _ in range(num_steps):
            normals = torch.randn(self.num_experiments, self.num_particles, self.dim, device=self.device)
            self.step(normals)

        final_consensus = self.consensus()
        final_objective = self.objective(final_consensus)
        return final_consensus, final_objective

    def run_experiments(self,
                        num_steps: int,
                        success_criterion: SuccessCriterion
                        ) -> Tuple[float, float]:
        """
        Run the particle system experiments in parallel and calculate the success rate.

        Args:
            num_steps: The number of iterations of the numerical scheme.
            success_criterion: A SuccessCriterion object to determine if an experiment is successful.


        Returns:
            Tuple[float, float]: The success rate and standard error.
        """
        final_consensus, final_objective = self.run_system(num_steps)

        # Apply success criterion individually to each experiment - maybe change this eventually
        success_mask = torch.tensor([
            success_criterion.check(final_consensus[i], final_objective[i])
            for i in range(self.num_experiments)
        ], device=self.device)

        success_rate = success_mask.double().mean().item()
        se = (success_rate * (1 - success_rate) / self.num_experiments) ** 0.5

        return success_rate, se


class VecProjectionParticleSystem(VectorizedParticleSystem):
    """
    A vectorized implementation of the simple projection particle system.

    Attributes:
        projection (Callable): Function to enforce constraints (e.g. projection to feasible region).
    """

    def __init__(self, projection: Callable[[torch.Tensor], None], *args, **kwargs):
        """
        Initialize the vectorized projection particle system.

        Args:
            projection: Callable to enforce constraints.
            *args, **kwargs: Additional arguments passed to the base VectorizedParticleSystem class.
        """
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)

        x_bar = self.consensus().unsqueeze(1)  # Shape (num_experiments, 1, dim)

        self.state += -beta * (self.state - x_bar) * self.h + sigma * (self.state - x_bar) * normals * self.h.sqrt()
        self.state = self.projection(self.state.view(-1, self.dim)).view(self.num_experiments, self.num_particles,
                                                                         self.dim)

        self.t += self.h
        return self.state, x_bar.squeeze(1)


class VecPenaltyParticleSystem(VectorizedParticleSystem):
    """
    A vectorized implementation of the penalty particle system.

    Attributes:
        projection (Callable): Function to enforce constraints (e.g. projection to feasible region).
    """

    def __init__(self, projection: Callable[[torch.Tensor], None], *args, **kwargs):
        """
        Initialize the vectorized projection particle system.

        Args:
            projection: Callable to enforce constraints.
            *args, **kwargs: Additional arguments passed to the base VectorizedParticleSystem class.
        """
        super().__init__(*args, **kwargs)
        self.projection = projection

    def step(self, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)

        x_bar = self.consensus().unsqueeze(1)  # Shape (num_experiments, 1, dim)
        current_state = self.state.clone()
        self.state = self.projection(self.state.view(-1, self.dim)).view(self.num_experiments, self.num_particles,
                                                                         self.dim)
        self.state += -beta * (current_state - x_bar) * self.h + sigma * (
                current_state - x_bar) * normals * self.h.sqrt()

        self.t += self.h
        return self.state, x_bar.squeeze(1)


class VecRepellingParticleSystem(VectorizedParticleSystem):
    """
    A vectorized implementation of the repelling particle system.

    Attributes:
        projection (Callable): Function to enforce constraints (e.g., projection to feasible region).
        lambda_func (Callable): Function of time controlling the repulsion strength.
    """

    def __init__(self, projection: Callable[[torch.Tensor], None], lambda_func: Callable[[torch.Tensor], torch.Tensor],
                 *args, **kwargs):
        """
        Initialize the vectorized repelling particle system.

        Args:
            projection: Callable to enforce constraints.
            lambda_func: Function of time controlling repulsion strength.
            *args, **kwargs: Additional arguments passed to the base VectorizedParticleSystem class.
        """
        super().__init__(*args, **kwargs)
        self.projection = projection
        self.lambda_func = lambda_func

    def step(self, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)
        lambd = self.lambda_func(self.t)

        x_bar = self.consensus().unsqueeze(1)  # Shape (num_experiments, 1, dim)

        pairwise_diff = self.state.unsqueeze(2) - self.state.unsqueeze(
            1)
        distances = torch.norm(pairwise_diff, dim=-1, keepdim=True).clamp(min=1e-8)

        repulsion_forces = torch.exp(-0.5 * distances ** 2) * pairwise_diff / distances
        repulsion_sum = repulsion_forces.sum(dim=2)
        attraction = -beta * (self.state - x_bar)
        diffusion = sigma * (self.state - x_bar) * normals
        self.state += (attraction + lambd * repulsion_sum) * self.h + diffusion * self.h.sqrt()

        self.state = self.projection(self.state.view(-1, self.dim)).view(self.num_experiments, self.num_particles,
                                                                         self.dim)

        self.t += self.h
        return self.state, x_bar.squeeze(1)
