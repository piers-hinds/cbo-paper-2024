import torch
import pytest
from rsde_opt import VecRepellingParticleSystem


def dummy_objective(x):
    return torch.sum(x ** 2, dim=1)


def dummy_initial_state(num_particles):
    return torch.rand(num_particles, 2)


def dummy_projection(x):
    return x.clamp(-1, 1)


def dummy_beta(t):
    return torch.tensor(0.1)


def dummy_sigma(t):
    return torch.tensor(0.05)


def dummy_lambda(t):
    return torch.tensor(0.2)


# Fixture to create the particle system
@pytest.fixture
def vec_repelling_system():
    return VecRepellingParticleSystem(
        objective=dummy_objective,
        initial_state=dummy_initial_state,
        alpha=1.0,
        beta=dummy_beta,
        sigma=dummy_sigma,
        lambda_func=dummy_lambda,
        projection=dummy_projection,
        dim=2,
        num_particles=10,
        step_size=0.1,
        num_experiments=5,
        device="cpu"
    )


# Fixture for normals tensor
@pytest.fixture
def normals(vec_repelling_system):
    return torch.randn(vec_repelling_system.num_experiments, vec_repelling_system.num_particles,
                       vec_repelling_system.dim)


# Test the output shapes of the step method
def test_step_output_shapes(vec_repelling_system, normals):
    state, x_bar = vec_repelling_system.step(normals)
    assert state.shape == (
        vec_repelling_system.num_experiments, vec_repelling_system.num_particles, vec_repelling_system.dim)
    assert x_bar.shape == (vec_repelling_system.num_experiments, vec_repelling_system.dim)


# Test projection application
def test_projection(vec_repelling_system, normals):
    state, _ = vec_repelling_system.step(normals)
    assert (state >= -1).all() and (state <= 1).all(), "Particles are not correctly projected."


# Test dynamics (attraction, repulsion, diffusion)
def test_dynamics(vec_repelling_system, normals):
    initial_state = vec_repelling_system.state.clone()
    state, _ = vec_repelling_system.step(normals)
    assert not torch.allclose(initial_state, state), "Particles are not moving as expected."


# Test multiple steps for stability
def test_multiple_steps(vec_repelling_system):
    normals = torch.randn(vec_repelling_system.num_experiments, vec_repelling_system.num_particles,
                          vec_repelling_system.dim)
    for _ in range(10):
        state, x_bar = vec_repelling_system.step(normals)

    assert state.shape == (
        vec_repelling_system.num_experiments, vec_repelling_system.num_particles, vec_repelling_system.dim)
    assert x_bar.shape == (vec_repelling_system.num_experiments, vec_repelling_system.dim)


# Test consensus computation
def test_consensus(vec_repelling_system, normals):
    _, x_bar = vec_repelling_system.step(normals)
    assert (x_bar >= -1).all() and (x_bar <= 1).all(), "Consensus point is out of bounds."
