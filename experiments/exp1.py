from rsde_opt import *


def project_unit_ball(x):
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
    x[norms > 1] /= norms[norms > 1].unsqueeze(1)


if __name__ == '__main__':

    system = ProjectionParticleSystem(lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
                                      project_unit_ball,
                                      1,
                                      1,
                                      0.6,
                                      2,
                                      1000,
                                      0.01,
                                      'cpu')

    logger = ParticleSystemLogger(system)
    system.state = uniform_disk(system.num_particles)
    print(system.state)

    print(system.state)

    system.consensus()

    for i in range(500):
        norms = torch.randn_like(system.state)
        state, consensus = system.step(norms)
        logger.log_state(state, consensus)

    logger.animate_particles()

