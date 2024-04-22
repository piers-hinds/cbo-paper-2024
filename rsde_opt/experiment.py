from tqdm.auto import tqdm
import torch
from .particle_system import ParticleSystem
from .loggers import ExperimentLogger


class SuccessCriterion:
    def __init__(self,
                 true_optimum: torch.Tensor,
                 epsilon: float,
                 optimum_type: str):
        """
        Args:
            true_optimum: The true optimum of the optimization problem. Can be the coordinates of the location or the
                function value.
            epsilon: The tolerance level.
            optimum_type: One of 'objective_value' or 'x_value' depending on whether the true optimum given is the
                objective function value or the location (x value).
        """
        optimum_types = ['objective_value', 'x_value']
        if optimum_type not in optimum_types:
            raise ValueError("Invalid optimum_type value. Expected one of: %s" % optimum_types)
        self.true_optimum = true_optimum
        self.epsilon = epsilon
        self.optimum_type = optimum_type

    def check(self, consensus: torch.Tensor, objective: torch.Tensor):
        if self.optimum_type == 'objective_value':
            return objective - self.true_optimum < self.epsilon

        if self.optimum_type == 'x_value':
            assert consensus.shape == self.true_optimum.shape, \
                "Success check failed: shape of true optimum does not match consensus shape"
            return torch.linalg.norm(consensus - self.true_optimum) < self.epsilon


def run_experiment(system: ParticleSystem,
                   num_steps: int,
                   success_criterion: SuccessCriterion,
                   num_runs: int,
                   logger: ExperimentLogger = None,
                   progress_bar: bool = True) -> float:
    """
    Run the particle system experiment multiple times and calculate the success rate.

    Args:
        system: The particle system to run the experiment on.
        num_steps: The number of iterations of the numerical scheme.
        success_criterion: A SuccessCriterion object to determine if an experiment is successful.
        num_runs: The number of times to run the experiment.
        logger: An option logger to record the particles consensus
        progress_bar: If True, a progress bar is displayed

    Returns:
        The success rate of the experiment.
    """
    success_count = 0

    progress_bar = tqdm(range(num_runs), disable=not progress_bar)
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

        final_consensus = system.consensus().cpu()
        final_objective = system.objective(final_consensus.unsqueeze(0)).cpu()

        if success_criterion.check(final_consensus, final_objective):
            success_count += 1

        system.reset()

        progress_bar.set_postfix({'Success rate': success_count / (i+1)})
    return success_count / num_runs
