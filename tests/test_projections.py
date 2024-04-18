from rsde_opt.projections import project_unit_ball
import torch


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