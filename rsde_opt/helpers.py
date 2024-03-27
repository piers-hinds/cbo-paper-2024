import torch


def random_uniform_ball(n: int, d: int):
    """
    Samples n points uniformly distributed within a d-dimensional unit ball.

    Parameters:
    n (int): Number of points to generate.
    d (int): Dimension of the ball.

    Returns:
    torch.Tensor: Tensor of shape (n, d) containing the generated points.
    """
    normal_samples = torch.randn(n, d)
    radii = torch.pow(torch.rand(n), 1 / d)
    uniform_sphere_samples = normal_samples / torch.norm(normal_samples, dim=1, keepdim=True)
    points = uniform_sphere_samples * radii.unsqueeze(1)
    return points


def project_unit_ball(x, r=1):
    """
    Projects the positions of N particles in R^d onto the closed unit ball.

    Parameters:
    x (torch.Tensor): A tensor of shape (N, d) representing the positions of N particles in R^d.

    Returns:
    torch.Tensor: A tensor of shape (N, d) where each d-dimensional row vector has been
                  either left unchanged (if it was inside the unit ball) or normalized
                  to lie on the boundary of the unit ball.
    """
    norms = torch.norm(x, p=2, dim=1)
    x[norms > r] *= (r / norms[norms > r].unsqueeze(1))