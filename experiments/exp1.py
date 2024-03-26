from rsde_opt import *
from functools import partial


def project_unit_ball(x, r=1):
    """
    Projects the positions of N particles in R^d onto the closed unit ball.

    Parameters:
    x (torch.Tensor): A tensor of shape (N, d) representing the positions of N particles in R^d.

    Returns:
    torch.Tensor: A tensor of shape (N, d) where each d-dimensional row vector has been
                  either left unchanged (if it was inside the unit ball) or normalized
                  to lie on the boundary of the unit ball.
    """
    norms = torch.norm(x, p=2, dim=1)
    x[norms > r] *= (r / norms[norms > r].unsqueeze(1))


def unit_circle_feasible_region(x, y, r=1):
    return x**2 + y**2 - r**2





if __name__ == '__main__':
    r = 2
    system = ProjectionParticleSystem(lambda x: torch.sqrt( (x[:, 0]-0.2) ** 2 + (x[:, 1]-0.2) ** 2),
                                      partial(project_unit_ball, r=r),
                                      5,
                                      0.5,
                                      0.2,
                                      2,
                                      100,
                                      0.01,
                                      'cpu')

    logger = ParticleSystemLogger(system)
    system.state = uniform_disk(system.num_particles) * 2

    system.consensus()

    for i in range(500):
        norms = torch.randn_like(system.state)
        state, consensus = system.step(norms)
        logger.log_state(state, consensus)

    ax = create_ax(system.objective, partial(unit_circle_feasible_region, r=r))
    animate_particles(logger, ax)

