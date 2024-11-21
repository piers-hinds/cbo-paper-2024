from rsde_opt.experiment import SuccessCriterion, run_experiment
from rsde_opt.particle_system import SimpleProjectionParticleSystem
from rsde_opt.loggers import ExperimentLogger
import torch
import pytest


@pytest.fixture
def example_simple_proj_particle_system():
    def unit_clamp(x):
        x.clamp(-1, 1)

    eps = SimpleProjectionParticleSystem(objective=lambda x: torch.sum(x ** 2, dim=1),
                                         projection=unit_clamp,
                                         initial_state=lambda n: torch.zeros(n, 2),
                                         alpha=1,
                                         beta=lambda x: 1,
                                         sigma=lambda x: 4,
                                         dim=2,
                                         num_particles=5,
                                         step_size=0.01)

    return eps


def test_success_criterion_obj_init():
    sc = SuccessCriterion(torch.tensor(0.),
                          0.1,
                          'objective_value')


def test_success_criterion_x_init():
    sc = SuccessCriterion(torch.zeros(2),
                          0.1,
                          'x_value')


def test_success_criterion_obj_check():
    sc = SuccessCriterion(torch.tensor(0.),
                          0.1,
                          'objective_value')
    assert sc.check(torch.zeros(2), torch.tensor(0.01))


def test_success_criterion_x_check():
    sc = SuccessCriterion(torch.zeros(2),
                          0.1,
                          'x_value')
    assert sc.check(torch.zeros(2) + 0.001, torch.tensor(0.01))


def test_success_criterion_x_check_false():
    sc = SuccessCriterion(torch.zeros(2),
                          0.1,
                          'x_value')
    assert not sc.check(torch.zeros(2) + 10, torch.tensor(0.01))


def test_run_experiment(example_simple_proj_particle_system):
    sc = SuccessCriterion(torch.tensor(0.),
                          0.1,
                          'objective_value')
    out = run_experiment(example_simple_proj_particle_system,
                         10,
                         sc,
                         5,
                         progress_bar=False)
    assert 0 <= out <= 1


def test_run_experiment_logger(example_simple_proj_particle_system):
    sc = SuccessCriterion(torch.tensor(0.),
                          0.1,
                          'objective_value')
    n = 5
    logger = ExperimentLogger(n)
    out = run_experiment(example_simple_proj_particle_system,
                         10,
                         sc,
                         n,
                         logger=logger,
                         progress_bar=False)
    assert 0 <= out <= 1
