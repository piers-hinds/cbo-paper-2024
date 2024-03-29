from rsde_opt import *
from functools import partial


if __name__ == '__main__':
    radius = 5
    true_optimum = torch.tensor([0., 0.])
    epsilon = 0.1
    num_runs = 100
    num_steps = 500

    system = ProjectionParticleSystem(objective=rastrigin_function,
                                      projection=partial(project_unit_ball, r=radius),
                                      initial_state=partial(random_uniform_ball, d=2, r=radius),
                                      alpha=5,
                                      beta=0.5,
                                      sigma=0.5,
                                      dim=2,
                                      num_particles=100,
                                      step_size=0.01)

    success_rate = run_experiment(system,
                                  num_steps,
                                  true_optimum,
                                  epsilon,
                                  num_runs)

    print(f"Success rate: {success_rate:.2f}")
