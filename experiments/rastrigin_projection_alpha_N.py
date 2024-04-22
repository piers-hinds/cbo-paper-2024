import csv
import itertools
from rsde_opt import *
from functools import partial


if __name__ == '__main__':
    radius = 5
    dim = 1000
    true_optimum = torch.tensor([0, 0])
    epsilon = 0.1
    sc = SuccessCriterion(true_optimum,
                          epsilon,
                          optimum_type='x_value')
    num_runs = 100
    num_steps = 500

    alphas = [1, 10, 100, 1000]
    Ns = [5, 10, 50, 100]

    system = SimpleProjectionParticleSystem(objective=rastrigin_function,
                                            projection=partial(project_unit_ball, r=radius),
                                            initial_state=partial(random_uniform_ball, d=dim, r=radius),
                                            alpha=10000,
                                            beta=1,
                                            sigma=4,
                                            dim=dim,
                                            num_particles=100,
                                            step_size=0.01)

    results = []
    for alpha, N in itertools.product(alphas, Ns):
        torch.manual_seed(0)
        system.reset()
        system.alpha = alpha
        system.num_particles = N
        success_rate = run_experiment(system,
                                      num_steps,
                                      sc,
                                      num_runs)
        results.append([alpha, N, success_rate])

    with open('alpha_n_rastigin_100.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Alpha', 'N', 'Success Rate'])
        writer.writerows(results)

    best_params = max(results, key=lambda x: x[2])
    print(f"\nBest hyperparameters:")
    print(f"alpha: {best_params[0]:.2f}", f"N: {best_params[1]:.2f}", f"Success rate: {best_params[2]:.2f}")