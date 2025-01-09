from rsde_opt import *
from functools import partial
import pandas as pd
from tqdm.auto import tqdm


def run_experiments():
    # Parameters
    num_experiments = 1000
    true_optimum = torch.tensor([2.0, 2.0])
    epsilon = 0.1
    radius = 3
    alpha = 10000
    beta = lambda x: torch.tensor(1.0)
    sigma = lambda x: torch.tensor(4.0)
    dim = 2
    device = "cpu"

    # Sweep over num_steps and num_particles
    num_steps_list = [5, 10, 20, 50, 100]
    num_particles_list = [10, 20, 50, 100]

    sc = SuccessCriterion(true_optimum, epsilon, "x_value")

    penalty_results = []
    projection_results = []

    total_combinations = len(num_steps_list) * len(num_particles_list) * 2
    with tqdm(total=total_combinations, desc="Running Experiments") as pbar:
        for num_steps in num_steps_list:
            for num_particles in num_particles_list:
                step_size = 1.0 / num_steps

                penalty_system = VecPenaltyParticleSystem(
                    objective=ackley_function,
                    num_experiments=num_experiments,
                    projection=partial(project_unit_ball, r=radius),
                    initial_state=partial(random_uniform_ball, d=dim, r=radius),
                    alpha=alpha,
                    beta=beta,
                    sigma=sigma,
                    dim=dim,
                    num_particles=num_particles,
                    step_size=step_size,
                    device=device,
                )

                success_rate, standard_error = penalty_system.run_experiments(
                    num_steps=num_steps, success_criterion=sc
                )
                penalty_results.append(
                    {
                        "num_steps": num_steps,
                        "num_particles": num_particles,
                        "success_rate": success_rate,
                        "standard_error": standard_error,
                    }
                )
                pbar.update(1)

                projection_system = VecProjectionParticleSystem(
                    objective=ackley_function,
                    num_experiments=num_experiments,
                    projection=partial(project_unit_ball, r=radius),
                    initial_state=partial(random_uniform_ball, d=dim, r=radius),
                    alpha=alpha,
                    beta=beta,
                    sigma=sigma,
                    dim=dim,
                    num_particles=num_particles,
                    step_size=step_size,
                    device=device,
                )

                success_rate, standard_error = projection_system.run_experiments(
                    num_steps=num_steps, success_criterion=sc
                )
                projection_results.append(
                    {
                        "num_steps": num_steps,
                        "num_particles": num_particles,
                        "success_rate": success_rate,
                        "standard_error": standard_error,
                    }
                )
                pbar.update(1)

    penalty_df = pd.DataFrame(penalty_results)
    projection_df = pd.DataFrame(projection_results)

    penalty_df.to_csv("ackley_penalty_results.csv", index=False)
    projection_df.to_csv("ackley_projection_results.csv", index=False)


if __name__ == "__main__":
    torch.manual_seed(1)
    run_experiments()
