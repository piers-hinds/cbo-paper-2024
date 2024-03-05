import numpy as np
import torch


def uniform_disk(n: int):
    theta = 2 * np.pi * torch.rand(n)
    r = torch.rand(n).sqrt()
    return torch.stack([r * theta.cos(), r * theta.sin()], dim=-1)
