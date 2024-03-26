from rsde_opt import *
from functools import partial


def unit_circle_feasible_region(x, y, r=1):
    return x**2 + y**2 - r**2


def rastrigin_function(x):
    """
    Rastrigin function.

    Parameters:
    x (torch.Tensor): Input tensor of shape (n, d) where n is the number of points and d is the dimension.

    Returns:
    torch.Tensor: Output tensor of shape (n,) containing the Rastrigin function values for each input point.
    """
    A = 10
    n = x.shape[1]

    term1 = A * n
    term2 = torch.sum(x ** 2 - A * torch.cos(2 * np.pi * x), dim=1)

    return term1 + term2


if __name__ == '__main__':
    radius = 5
    system = ProjectionParticleSystem(rastrigin_function,
                                      partial(project_unit_ball, r=radius),
                                      20,
                                      0.5,
                                      0.5,
                                      2,
                                      1000,
                                      0.001,
                                      'cpu')

    logger = ParticleSystemLogger(system)
    torch.manual_seed(20)
    system.state = uniform_ball(system.num_particles, 2) * radius

    ax = create_ax(system.objective,
                   partial(unit_circle_feasible_region, r=radius),
                   (-(radius + 0.1), radius + 0.1),
                   (-(radius + 0.1), radius + 0.1)
                   )

    for i in range(2000):
        normals = torch.randn_like(system.state)
        state, consensus = system.step(normals)
        logger.log_state(state, consensus)

    animation = animate_particles(logger, ax)
    plt.show()
