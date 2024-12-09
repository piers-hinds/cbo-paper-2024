from .particle_system import ParticleSystem
import torch
from collections import defaultdict
from typing import Tuple, Dict, Callable


class RepellingParticleSystemWithLSH(ParticleSystem):
    def __init__(self, projection: Callable[[torch.Tensor], None], lambda_func: Callable[[torch.Tensor], torch.Tensor],
                 cutoff: float, grid_size: float, *args, **kwargs):
        """
        RepellingParticleSystem with locality-sensitive hashing (LSH).

        Args:
            projection: Function to enforce constraints (e.g., projection to feasible region).
            lambda_func: A callable function of time t for the repelling parameter lambda(t).
            cutoff: Cutoff distance for repelling interactions.
            grid_size: Size of the grid cells for spatial hashing.
            *args, **kwargs: Other arguments passed to the base ParticleSystem class.
        """
        super().__init__(*args, **kwargs)
        self.projection = projection
        self.lambda_func = lambda_func
        self.cutoff = cutoff
        self.grid_size = grid_size

    def _hash_particles(self) -> Dict[Tuple[int, ...], list]:
        """
        Hash particles into a spatial grid based on their positions.

        Returns:
            dict: A dictionary mapping grid cell indices to lists of particle indices.
        """
        grid = defaultdict(list)
        grid_indices = torch.floor(self.state / self.grid_size).int()
        for idx, grid_index in enumerate(grid_indices):
            grid[tuple(grid_index.tolist())].append(idx)
        return grid

    def _get_neighbor_bins(self, bin_index: Tuple[int, ...]) -> list:
        """
        Get the neighboring bins of a given grid cell index.

        Args:
            bin_index (Tuple[int, ...]): The index of the grid cell.

        Returns:
            list: A list of neighboring bin indices.
        """
        # Generate offsets as the Cartesian product of [-1, 0, 1]
        offsets = torch.cartesian_prod(*[torch.tensor([-1, 0, 1]) for _ in bin_index])

        # Add offsets to the current bin index
        neighbors = [tuple(torch.tensor(bin_index) + offset) for offset in offsets]
        return neighbors

    def _compute_repulsion(self, grid: dict) -> torch.Tensor:
        """
        Compute repelling forces using LSH.

        Args:
            grid (dict): The grid mapping bins to particle indices.

        Returns:
            torch.Tensor: The computed repelling forces for each particle.
        """
        repelling_forces = torch.zeros_like(self.state)  # Initialize forces

        for bin_index, particle_indices in grid.items():
            # Get all particles in the current bin and its neighbors
            neighbor_bins = self._get_neighbor_bins(bin_index)
            candidates = []
            for neighbor_bin in neighbor_bins:
                candidates.extend(grid.get(neighbor_bin, []))

            # Ensure unique candidates
            candidates = list(set(candidates))

            if len(candidates) > 1:  # Only compute if there's more than 1 particle
                candidate_states = self.state[candidates]

                # Compute pairwise differences and distances
                pairwise_diff = candidate_states.unsqueeze(1) - candidate_states.unsqueeze(0)
                distances = torch.norm(pairwise_diff, dim=-1).clamp(min=1e-8)

                # Mask pairwise interactions based on cutoff
                mask = distances < self.cutoff + 100
                pairwise_diff = pairwise_diff * mask.unsqueeze(-1)  # Apply mask to differences

                # Compute repulsion forces
                repulsion = torch.exp(-0.5 * distances ** 2).unsqueeze(-1) * pairwise_diff / distances.unsqueeze(-1)
                repulsion_sum = torch.sum(repulsion, dim=1)

                # Accumulate forces for each particle
                for idx, force in zip(candidates, repulsion_sum):
                    repelling_forces[idx] += force

        return repelling_forces

    def step(self, normals: torch.Tensor) -> torch.Tensor:
        """
        Perform one step of the particle dynamics with LSH-based repulsion.

        Args:
            normals: Random noise (Gaussian increments) for diffusion.

        Returns:
            torch.Tensor: Updated state and consensus.
        """
        beta = self.beta(self.t)
        sigma = self.sigma(self.t)
        lambd = self.lambda_func(self.t)

        x_bar = self.consensus()

        # Attraction term
        attraction = -beta * (self.state - x_bar)

        # Repulsion term using LSH
        grid = self._hash_particles()
        repulsion = lambd * self._compute_repulsion(grid)

        # Diffusion term
        diffusion = sigma * (self.state - x_bar)

        # Update state
        self.state += (attraction + repulsion) * self.h + diffusion * normals * self.h.sqrt()
        self.projection(self.state)
        self.t += self.h
        return self.state, x_bar
