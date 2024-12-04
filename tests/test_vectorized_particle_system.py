import pytest
import torch
from rsde_opt.vectorized_particle_system import VecProjectionParticleSystem
from rsde_opt.projections import project_unit_ball
from rsde_opt.benchmarks import ackley_function
from rsde_opt.helpers import random_uniform_ball
from rsde_opt.experiment import SuccessCriterion


@pytest.fixture
def example_vec_proj_particle_system():
    eps = VecProjectionParticleSystem(
        objective=ackley_function,
        projection=lambda x: project_unit_ball(x, r=3),
        initial_state=lambda n: random_uniform_ball(n, d=2, r=3),
        alpha=10.0,
        beta=lambda t: 0.1,
        sigma=lambda t: 0.05,
        dim=2,
        num_particles=5,
        step_size=0.01,
        num_experiments=3,
        device='cpu'
    )
    return eps


def test_vec_particle_system_init(example_vec_proj_particle_system):
    assert example_vec_proj_particle_system.state.shape == (
        example_vec_proj_particle_system.num_experiments,
        example_vec_proj_particle_system.num_particles,
        example_vec_proj_particle_system.dim
    )


def test_vec_particle_system_consensus(example_vec_proj_particle_system):
    consensus = example_vec_proj_particle_system.consensus()
    assert consensus.shape == (example_vec_proj_particle_system.num_experiments,
                                example_vec_proj_particle_system.dim)


def test_vec_particle_system_reset(example_vec_proj_particle_system):
    initial_state = example_vec_proj_particle_system.state.clone()
    example_vec_proj_particle_system.step(
        torch.ones([example_vec_proj_particle_system.num_experiments,
                    example_vec_proj_particle_system.num_particles,
                    example_vec_proj_particle_system.dim])
    )
    example_vec_proj_particle_system.reset()
    assert not torch.allclose(initial_state, example_vec_proj_particle_system.state)


def test_vec_projection_particle_system_step(example_vec_proj_particle_system):
    normals = torch.randn(
        example_vec_proj_particle_system.num_experiments,
        example_vec_proj_particle_system.num_particles,
        example_vec_proj_particle_system.dim
    )
    new_state, consensus = example_vec_proj_particle_system.step(normals)
    assert new_state.shape == example_vec_proj_particle_system.state.shape
    assert consensus.shape == (example_vec_proj_particle_system.num_experiments,
                                example_vec_proj_particle_system.dim)


@pytest.fixture
def success_criterion():
    return SuccessCriterion(true_optimum=torch.tensor([2.0, 2.0]),
                            epsilon=0.1,
                            optimum_type='x_value')


def test_vec_particle_system_run_experiments(example_vec_proj_particle_system, success_criterion):
    success_rate, se = example_vec_proj_particle_system.run_experiments(
        num_steps=50,
        success_criterion=success_criterion
    )
    assert 0.0 <= success_rate <= 1.0
    assert se >= 0.0
