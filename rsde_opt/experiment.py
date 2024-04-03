from tqdm.auto import tqdm
import torch
from .particle_system import ParticleSystem


def run_experiment(system: ParticleSystem,
                   num_steps: int,
                   true_optimum: torch.tensor,
                   epsilon: float,
                   num_runs: int) -> float:
    """
    Run the particle system experiment multiple times and calculate the success rate.

    Args:
        system: The particle system to run the experiment on.
        num_steps: The number of iterations of the numerical scheme.
        true_optimum: The true optimal objective value.
        epsilon: The tolerance for considering a run successful.
        num_runs: The number of times to run the experiment.

    Returns:
        The success rate of the experiment.
    """
    success_count = 0

    pbar = tqdm(range(num_runs))
    pbar.set_description("N={}, ".format(system.num_particles) +
                         "alpha={:.1f}, ".format(system.alpha) +
                         "beta={:.1f}, ".format(system.beta) +
                         "sigma={:.1f} ".format(system.sigma)
                         )

    for i in pbar:
        for _ in range(num_steps):
            normals = torch.randn_like(system.state, device=system.device)
            state, _ = system.step(normals)

        if torch.linalg.norm(system.consensus() - true_optimum) < epsilon:
            success_count += 1

        system.reset()

        pbar.set_postfix({'Success rate': success_count / (i+1)})
    return success_count / num_runs
