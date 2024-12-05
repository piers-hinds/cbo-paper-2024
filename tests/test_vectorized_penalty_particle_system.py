import pytest
import torch
from rsde_opt.vectorized_particle_system import VecPenaltyParticleSystem
from rsde_opt.projections import project_unit_ball
from rsde_opt.helpers import random_uniform_ball
from rsde_opt.experiment import SuccessCriterion


@pytest.fixture
def example_vec_penalty_particle_system():
    """Fixture for creating an instance of VecPenaltyParticleSystem."""
    eps = VecPenaltyParticleSystem(
        objective=lambda x: torch.sum(x ** 2, dim=1),
        projection=lambda x: project_unit_ball(x, r=2),
        initial_state=lambda n: random_uniform_ball(n, d=3, r=2),
        alpha=5.0,
        beta=lambda t: 0.1,
        sigma=lambda t: 0.05,
        dim=3,
        num_particles=10,
        step_size=0.01,
        num_experiments=4,
        device='cpu'
    )
    return eps


def test_vec_penalty_particle_system_init(example_vec_penalty_particle_system):
    """Test initialization of VecPenaltyParticleSystem."""
    assert example_vec_penalty_particle_system.state.shape == (
        example_vec_penalty_particle_system.num_experiments,
        example_vec_penalty_particle_system.num_particles,
        example_vec_penalty_particle_system.dim
    ), "State shape does not match expected dimensions."


def test_vec_penalty_particle_system_consensus(example_vec_penalty_particle_system):
    """Test consensus computation."""
    consensus = example_vec_penalty_particle_system.consensus()
    assert consensus.shape == (
        example_vec_penalty_particle_system.num_experiments,
        example_vec_penalty_particle_system.dim
    ), "Consensus shape does not match expected dimensions."


def test_vec_penalty_particle_system_projection(example_vec_penalty_particle_system):
    """Test that projection is correctly applied during step."""
    normals = torch.randn(
        example_vec_penalty_particle_system.num_experiments,
        example_vec_penalty_particle_system.num_particles,
        example_vec_penalty_particle_system.dim
    )
    _, _ = example_vec_penalty_particle_system.step(normals)

    # Check that all states lie within the projected region (radius 2)
    norms = torch.norm(example_vec_penalty_particle_system.state.view(-1, example_vec_penalty_particle_system.dim),
                       dim=1)
    assert torch.all(norms <= 2.0), "Projection failed to enforce constraints."


def test_vec_penalty_particle_system_step(example_vec_penalty_particle_system):
    """Test the step method for proper updates."""
    normals = torch.randn(
        example_vec_penalty_particle_system.num_experiments,
        example_vec_penalty_particle_system.num_particles,
        example_vec_penalty_particle_system.dim
    )
    new_state, consensus = example_vec_penalty_particle_system.step(normals)

    assert new_state.shape == (
        example_vec_penalty_particle_system.num_experiments,
        example_vec_penalty_particle_system.num_particles,
        example_vec_penalty_particle_system.dim
    ), "New state shape does not match expected dimensions."

    assert consensus.shape == (
        example_vec_penalty_particle_system.num_experiments,
        example_vec_penalty_particle_system.dim
    ), "Consensus shape does not match expected dimensions."


@pytest.fixture
def success_criterion():
    """Fixture for creating a success criterion."""
    return SuccessCriterion(true_optimum=torch.tensor([0.0, 0.0, 0.0]), epsilon=0.1, optimum_type='x_value')


def test_vec_penalty_particle_system_run_experiments(example_vec_penalty_particle_system, success_criterion):
    """Test running experiments with VecPenaltyParticleSystem."""
    success_rate, se = example_vec_penalty_particle_system.run_experiments(
        num_steps=50,
        success_criterion=success_criterion
    )

    assert 0.0 <= success_rate <= 1.0, "Success rate should be between 0 and 1."
    assert se >= 0.0, "Standard error should be non-negative."


def test_vec_penalty_particle_system_reset(example_vec_penalty_particle_system):
    """Test reset functionality of VecPenaltyParticleSystem."""
    initial_state = example_vec_penalty_particle_system.state.clone()
    normals = torch.randn(
        example_vec_penalty_particle_system.num_experiments,
        example_vec_penalty_particle_system.num_particles,
        example_vec_penalty_particle_system.dim
    )
    example_vec_penalty_particle_system.step(normals)
    example_vec_penalty_particle_system.reset()

    assert not torch.allclose(initial_state, example_vec_penalty_particle_system.state), \
        "State should be reinitialized after reset."
