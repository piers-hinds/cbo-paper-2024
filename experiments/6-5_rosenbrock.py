from rsde_opt import *
from functools import partial
import pandas as pd
from tqdm.auto import tqdm


def run_experiments():
    # Parameters
    num_experiments = 1000
    true_optimum = torch.tensor([1.0, 1.0])
    epsilon = 0.1
    radius = torch.sqrt(torch.tensor(2.0))
    alpha = 10000
    beta = lambda x: torch.tensor(1.0)
    sigma = lambda x: torch.tensor(4.0)
    dim = 2
    device = "cpu"

    num_steps_list = [5, 10, 20, 50, 100]
    num_particles_list = [10, 20, 50, 100]

    sc = SuccessCriterion(true_optimum, epsilon, "x_value")

    projection_results = []
    repelling_results = []

    total_combinations = len(num_steps_list) * len(num_particles_list) * 2
    with tqdm(total=total_combinations, desc="Running Experiments") as pbar:
        for num_steps in num_steps_list:
            for num_particles in num_particles_list:
                step_size = 1.0 / 20

                projection_system = VecProjectionParticleSystem(
                    objective=rosenbrock_function,
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
                    num_steps=num_steps,
                    success_criterion=sc
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

                repelling_system = VecRepellingParticleSystem(
                    objective=rosenbrock_function,
                    num_experiments=num_experiments,
                    projection=partial(project_unit_ball, r=radius),
                    initial_state=partial(random_uniform_ball, d=dim, r=radius),
                    alpha=alpha,
                    beta=beta,
                    sigma=sigma,
                    lambda_func=lambda x: 3 * torch.exp(-5 * x),
                    dim=dim,
                    num_particles=num_particles,
                    step_size=step_size,
                    device=device,
                )

                success_rate, standard_error = repelling_system.run_experiments(
                    num_steps=num_steps,
                    success_criterion=sc
                )
                repelling_results.append(
                    {
                        "num_steps": num_steps,
                        "num_particles": num_particles,
                        "success_rate": success_rate,
                        "standard_error": standard_error,
                    }
                )
                pbar.update(1)

    projection_df = pd.DataFrame(projection_results)
    repelling_df = pd.DataFrame(repelling_results)

    projection_df.to_csv("rosenbrock_projection_time_results.csv", index=False)
    repelling_df.to_csv("rosenbrock_repelling_time_results.csv", index=False)


if __name__ == "__main__":
    torch.manual_seed(1)
    run_experiments()
