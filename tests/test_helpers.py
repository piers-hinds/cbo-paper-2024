from rsde_opt.helpers import random_uniform_ball
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
