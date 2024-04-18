import pytest
import torch
from rsde_opt.particle_system import ProjectionParticleSystem, SimpleProjectionParticleSystem


@pytest.fixture
def example_proj_particle_system():
    def unit_clamp(x):
        x.clamp(-1, 1)

    eps = ProjectionParticleSystem(objective=lambda x: torch.sum(x ** 2, dim=1),
                                   projection=unit_clamp,
                                   initial_state=lambda n: torch.zeros(n, 2),
                                   alpha=1,
                                   beta=1,
                                   sigma=4,
                                   dim=2,
                                   num_particles=5,
                                   step_size=0.01)

    return eps


@pytest.fixture
def example_simple_proj_particle_system():
    def unit_clamp(x):
        x.clamp(-1, 1)

    eps = SimpleProjectionParticleSystem(objective=lambda x: torch.sum(x ** 2, dim=1),
                                         projection=unit_clamp,
                                         initial_state=lambda n: torch.zeros(n, 2),
                                         alpha=1,
                                         beta=1,
                                         sigma=4,
                                         dim=2,
                                         num_particles=5,
                                         step_size=0.01)

    return eps


def test_particle_system_init(example_proj_particle_system):
    assert example_proj_particle_system.state.shape == (example_proj_particle_system.num_particles,
                                                        example_proj_particle_system.dim)


def test_particle_system_consensus(example_proj_particle_system):
    consensus = example_proj_particle_system.consensus()
    assert consensus.shape == (2,)


def test_particle_system_consensus_external(example_proj_particle_system):
    state = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0]])
    consensus = example_proj_particle_system.consensus(state)
    assert consensus.shape == (2,)


def test_particle_system_reset(example_proj_particle_system):
    initial_state = example_proj_particle_system.state.clone()
    example_proj_particle_system.step(torch.ones([example_proj_particle_system.num_particles,
                                                  example_proj_particle_system.dim]))
    example_proj_particle_system.reset()
    assert torch.allclose(example_proj_particle_system.state, initial_state)


def test_simple_projection_particle_system_step(example_proj_particle_system):
    normals = torch.ones([5, 2])
    new_state, consensus = example_proj_particle_system.step(normals)
    assert new_state.shape == (5, 2)


def test_projection_particle_system_step(example_simple_proj_particle_system):
    normals = torch.randn(5, 2)
    new_state, consensus = example_simple_proj_particle_system.step(normals)
    assert new_state.shape == (5, 2)
