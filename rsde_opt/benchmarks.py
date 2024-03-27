import torch
import numpy as np


def rastrigin_function(x):
    """
    Rastrigin function.

    Parameters:
    x (torch.Tensor): Input tensor of shape (n, d) where n is the number of points and d is the dimension.

    Returns:
    torch.Tensor: Output tensor of shape (n,) containing the Rastrigin function values for each input point.
    """
    d = x.shape[1]
    return 10 * d + torch.sum(x ** 2 - 10 * torch.cos(2 * np.pi * x), dim=1)


def ackley_function(x):
    """
    Ackley function, translated so the global minimum is at (2, 2)

    Parameters:
    x (torch.Tensor): Input tensor of shape (n, 2), where n is the number of points.
                      The tensor should contain 2-dimensional points.

    Returns:
    torch.Tensor: Output tensor of shape (n,) containing the Ackley function values
                  for each input point.
    """
    x_part, y_part = x[:, 0]-2, x[:, 1]-2
    a = 20
    b = 0.2
    c = 2 * torch.pi
    term1 = -a * torch.exp(-b * torch.sqrt(0.5 * (x_part ** 2 + y_part ** 2)))
    term2 = -torch.exp(0.5 * (torch.cos(c * x_part) + torch.cos(c * y_part)))
    ackley_value = term1 + term2 + a + torch.exp(torch.tensor(1.0))
    return ackley_value
