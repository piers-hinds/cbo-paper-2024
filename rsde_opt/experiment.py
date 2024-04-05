from tqdm.auto import tqdm
import torch
from .particle_system import ParticleSystem
from .loggers import ExperimentLogger


def run_experiment(system: ParticleSystem,
                   num_steps: int,
                   true_optimum: torch.tensor,
                   epsilon: float,
                   num_runs: int,
                   logger: ExperimentLogger = None) -> float:
    """
    Run the particle system experiment multiple times and calculate the success rate.

    Args:
        system: The particle system to run the experiment on.
        num_steps: The number of iterations of the numerical scheme.
        true_optimum: The true optimal objective value.
        epsilon: The tolerance for considering a run successful.
        num_runs: The number of times to run the experiment.
        logger: An option logger to record the particles consensus

    Returns:
        The success rate of the experiment.
    """
    success_count = 0

    progress_bar = tqdm(range(num_runs))
    progress_bar.set_description("N={}, ".format(system.num_particles) +
                                 "alpha={:.1f}, ".format(system.alpha) +
                                 "beta={:.1f}, ".format(system.beta) +
                                 "sigma={:.1f} ".format(system.sigma)
                                 )

    for i in progress_bar:
        for _ in range(num_steps):
            normals = torch.randn_like(system.state, device=system.device)
            state, consensus = system.step(normals)

            if logger is not None:
                logger.log_consensus(experiment_index=i, consensus=consensus)
                objective_value = system.objective(consensus.unsqueeze(0))
                logger.log_objective(experiment_index=i, objective_value=objective_value)

        if torch.linalg.norm(system.consensus() - true_optimum) < epsilon:
            success_count += 1

        system.reset()

        progress_bar.set_postfix({'Success rate': success_count / (i+1)})
    return success_count / num_runs
