from rsde_opt import *

if __name__ == '__main__':
    system = ProjectionParticleSystem(lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
                                      lambda x: x,
                                      1,
                                      1,
                                      0.6,
                                      2,
                                      500,
                                      0.001,
                                      'cpu')

    logger = ParticleSystemLogger(system)
    system.state = torch.rand(system.state.shape) * 100 - 50
    print(system.state)

    print(system.state)

    system.consensus()

    for i in range(2500):
        norms = torch.randn_like(system.state)
        state, consensus = system.step(norms)
        logger.log_state(state, consensus)

    logger.animate_particles()