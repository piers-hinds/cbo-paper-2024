from rsde_opt import *
from functools import partial
import pandas as pd

if __name__ == "__main__":
    torch.manual_seed(1)
    dimensions = [5, 20, 100, 500]
    num_particles_list = [10, 20, 50, 100]

    radius = 5
    epsilon = 0.1
    num_experiments = 1000
    step_size = 1 / 500
    set_num_steps = [50, 200, 500, 1000]

    for num_steps in set_num_steps:
        results = []

        for dim in dimensions:
            true_optimum = torch.zeros(size=(dim,))
            sc = SuccessCriterion(true_optimum, epsilon, 'x_value')
            for num_particles in num_particles_list:
                system = VecProjectionParticleSystem(
                    objective=rastrigin_function,
                    projection=partial(project_unit_ball, r=radius),
                    initial_state=partial(random_uniform_ball, d=dim, r=radius),
                    alpha=10000,
                    beta=lambda t: 1 * (0 + 10 * t),
                    sigma=lambda t: 10 * torch.exp(-np.log(10) * t),
                    dim=dim,
                    num_particles=num_particles,
                    num_experiments=num_experiments,
                    step_size=step_size,
                )
                success_rate, _ = system.run_experiments(num_steps, sc)
                row = {"d": dim, "N": num_particles, "success_rate": success_rate}
                print(num_steps, ': ', row)
                results.append(row)

        df_results = pd.DataFrame(results)
        df_results.to_csv(f"rastrigin_dimension_results_{num_steps}.csv", index=False)
