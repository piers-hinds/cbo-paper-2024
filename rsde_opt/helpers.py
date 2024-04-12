import torch
from .projections import project_heart_constraint


def random_uniform_ball(n: int,
                        d: int,
                        r: float = 1) -> torch.Tensor:
    """
    Samples n points uniformly distributed within a d-dimensional unit ball.

    Args:
    n: Number of points to generate.
    d: Dimension of the ball.
    r: Radius of the ball.

    Returns:
    torch.Tensor: Tensor of shape (n, d) containing the generated points.
    """
    normal_samples = torch.randn(n, d)
    radii = torch.pow(torch.rand(n), 1 / d)
    uniform_sphere_samples = normal_samples / torch.norm(normal_samples, dim=1, keepdim=True)
    points = uniform_sphere_samples * radii.unsqueeze(1)
    return points * r


def sample_heart_initial_state(n: int) -> torch.Tensor:
    """
    Samples n points from a standard Gaussian distribution and projects onto the heart shaped region
    Args:
        n:

    Returns:
    torch.Tensor: Tensor of shape (n, 2) containing the generated points.
    """
    x = torch.randn([n, 2])
    return project_heart_constraint(x)

