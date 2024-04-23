from rsde_opt import *
from functools import partial


def unit_circle_feasible_region(x, y, r=1):
    return x**2 + y**2 - r**2


if __name__ == '__main__':
    radius = 3
    system = SimpleProjectionParticleSystem(objective=ackley_function,
                                            projection=partial(project_unit_ball, r=radius),
                                            initial_state=partial(random_uniform_ball, d=2, r=radius),
                                            alpha=10,
                                            beta=1,
                                            sigma=3,
                                            dim=2,
                                            num_particles=100,
                                            step_size=0.002,
                                            device='cpu')

    logger = ParticleSystemLogger(system)
    torch.manual_seed(3)

    for i in range(200):
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
