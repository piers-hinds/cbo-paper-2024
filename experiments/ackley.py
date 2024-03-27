import torch
from rsde_opt import *
from functools import partial
from matplotlib.animation import FuncAnimation, PillowWriter


def unit_circle_feasible_region(x, y, r=1):
    return x**2 + y**2 - r**2


if __name__ == '__main__':
    radius = 3
    system = ProjectionParticleSystem(ackley_function,
                                      partial(project_unit_ball, r=radius),
                                      10,
                                      0.5,
                                      0.5,
                                      2,
                                      100,
                                      0.001,
                                      'cpu')

    logger = ParticleSystemLogger(system)
    torch.manual_seed(2)
    system.state = random_uniform_ball(system.num_particles, 2) * radius

    for i in range(2000):
        normals = torch.randn_like(system.state)
        state, consensus = system.step(normals)
        logger.log_state(state, consensus)

    ax = create_ax(system.objective,
                   partial(unit_circle_feasible_region, r=radius),
                   (-(radius + 0.1), radius + 0.1),
                   (-(radius + 0.1), radius + 0.1)
                   )
    animation = animate_particles(logger, ax)
    plt.show()
    # animation.save('C:/Users/pmxph7/OneDrive - The University of Nottingham/PhD/ccbo/plots/ackley.gif',
    #          writer=PillowWriter(fps=30))
