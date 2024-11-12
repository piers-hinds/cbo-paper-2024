import csv
import itertools
from rsde_opt import *
from functools import partial


def grad_heart_level_set(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    t = torch.atan2(x1, x2)

    cos_t, sin_t = torch.cos(t), torch.sin(t)
    cos_2t, sin_2t = torch.cos(2 * t), torch.sin(2 * t)
    cos_3t, sin_3t = torch.cos(3 * t), torch.sin(3 * t)
    cos_4t, sin_4t = torch.cos(4 * t), torch.sin(4 * t)

    dt_dx1 = x2 / (x1 ** 2 + x2 ** 2)
    dt_dx2 = -x1 / (x1 ** 2 + x2 ** 2)

    dcos_terms_dt = -2 * sin_t + sin_2t + 0.75 * sin_3t + 0.5 * sin_4t
    dsin_term_dt = 2 * cos_t

    dcos_terms_dx1 = dcos_terms_dt * dt_dx1
    dcos_terms_dx2 = dcos_terms_dt * dt_dx2
    dsin_term_dx1 = dsin_term_dt * dt_dx1
    dsin_term_dx2 = dsin_term_dt * dt_dx2

    df_dx1 = 2 * x1 - 2 * (2 * cos_t - 0.5 * cos_2t - 0.25 * cos_3t - 0.125 * cos_4t) * dcos_terms_dx1 - 2 * (2 * sin_t) * dsin_term_dx1
    df_dx2 = 2 * x2 - 2 * (2 * cos_t - 0.5 * cos_2t - 0.25 * cos_3t - 0.125 * cos_4t) * dcos_terms_dx2 - 2 * (2 * sin_t) * dsin_term_dx2

    return torch.stack([df_dx1, df_dx2], dim=1)


def vheart_level_set(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    t = torch.atan2(x1, x2)

    cos_terms = 2 * torch.cos(t) - 0.5 * torch.cos(2 * t) - 0.25 * torch.cos(3 * t) - 0.125 * torch.cos(4 * t)
    sin_term = 2 * torch.sin(t)

    return x1 ** 2 + x2 ** 2 - cos_terms ** 2 - sin_term ** 2


def heart_level_set(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[0], x[1]
    t = torch.atan2(x1, x2)

    cos_terms = 2 * torch.cos(t) - 0.5 * torch.cos(2 * t) - 0.25 * torch.cos(3 * t) - 0.125 * torch.cos(4 * t)
    sin_term = 2 * torch.sin(t)

    return x1 ** 2 + x2 ** 2 - cos_terms ** 2 - sin_term ** 2


grad_level_set = torch.vmap(torch.func.grad(heart_level_set))


def normal_heart_constraint(x):
    grads = - grad_heart_level_set(x)
    return grads / torch.linalg.vector_norm(grads, dim=1, keepdim=True)


def newton_root(f, x0):
    x = torch.tensor(x0, requires_grad=True)
    for i in range(100):
        fx = f(x)
        fx.backward()
        x_new = x - fx / x.grad
        if torch.abs(x_new - x) < 1e-6:
            break
        x.grad.zero_()
        x = x_new.detach().requires_grad_(True)
    return x.detach()


def heart_projection(x: torch.Tensor) -> torch.Tensor:
    mask = vheart_level_set(x) > 0
    inward_normals = normal_heart_constraint(x[mask])
    x_masked = x[mask]

    for i, (point, normal) in enumerate(zip(x_masked, inward_normals)):
        r_opt = newton_root(lambda r: heart_level_set(point + r * normal), 0.1)
        x_masked[i] = point + r_opt * normal

    x[mask] = x_masked
    return x


def townsend_init(n):
    samples = random_uniform_ball(n, 2, 2.5)
    heart_projection(samples)
    return samples


if __name__ == '__main__':
    dim = 2
    true_optimum = torch.tensor([2.0052938, 1.194451])
    epsilon = 0.1
    sc = SuccessCriterion(true_optimum,
                          epsilon,
                          optimum_type='x_value')
    num_runs = 100
    num_steps = 500

    alphas = [1, 10, 100, 1000]
    Ns = [5, 10, 50, 100]

    system = SimpleProjectionParticleSystem(objective=townsend_function,
                                            projection=heart_projection,
                                            initial_state=townsend_init,
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

    with open('alpha_n_townsend.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Alpha', 'N', 'Success Rate'])
        writer.writerows(results)

    best_params = max(results, key=lambda x: x[2])
    print(f"\nBest hyperparameters:")
    print(f"alpha: {best_params[0]:.2f}", f"N: {best_params[1]:.2f}", f"Success rate: {best_params[2]:.2f}")