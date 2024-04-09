from rsde_opt.helpers import random_uniform_ball, project_unit_ball
import torch


def test_random_uniform_ball_shape():
    n = 20
    d = 4
    r = 1
    torch.manual_seed(0)
    samples = random_uniform_ball(n, d, r)
    assert samples.shape == (n, d)


def test_random_uniform_ball_radius():
    n = 100
    d = 4
    r = 5
    torch.manual_seed(0)
    samples = random_uniform_ball(n, d, r)
    norms = samples.norm(dim=1, p=2)
    assert torch.all(norms <= r)


def test_project_unit_ball_shape():
    n = 100
    d = 10
    x = torch.zeros([n, d])
    project_unit_ball(x)
    assert x.shape == (n, d)


def test_project_unit_ball_unit_radius():
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [0.1, 0.1, 0.2],
                      [0., 0., 0.],
                      [0., 0., 1000.]])
    project_unit_ball(x)
    assert torch.all(x.norm(dim=1, p=2) <= 1)


def test_project_unit_ball_radius():
    r = 3
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [0.1, 0.1, 0.2],
                      [0., 0., 0.],
                      [0., 0., 1000.]])
    project_unit_ball(x, r)
    assert torch.all(x.norm(dim=1, p=2) <= r + 1e-6)
