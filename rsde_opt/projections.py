import torch


def project_unit_ball(x: torch.Tensor,
                      r: float = 1) -> torch.Tensor:
    """
    Projects from R^d onto the closed ball of radius r.

    Args:
    x: A tensor of shape (N, d) representing the positions of N particles in R^d.
    r: Radius of unit ball to project onto.

    Returns:
    torch.Tensor: A tensor of shape (N, d) where each d-dimensional row vector has been
                  either left unchanged (if it was inside the unit ball) or normalized
                  to lie on the boundary of the unit ball.
    """
    norms = torch.norm(x, p=2, dim=1)
    x[norms > r] *= (r / norms[norms > r].unsqueeze(1))
    return x


def project_heart_constraint(x: torch.Tensor) -> torch.Tensor:
    """
    Projects the from R^d onto the heart-shaped region defined in
    https://www.chebfun.org/examples/opt/ConstrainedOptimization.html

    Args:
        x: A tensor of shape (N, d) representing the positions of N particles in R^d.

    Returns:
    torch.Tensor: A tensor of shape (N, d)
    """
    x_part = x[:, 0]
    y_part = x[:, 1]

    t = torch.atan2(x_part, y_part)

    cos_terms = 2 * torch.cos(t) - 0.5 * torch.cos(2 * t) - 0.25 * torch.cos(3 * t) - 0.125 * torch.cos(4 * t)
    sin_term = 2 * torch.sin(t)

    r_boundary_sq = cos_terms ** 2 + sin_term ** 2
    r_sq = x_part ** 2 + y_part ** 2

    mask = r_sq >= r_boundary_sq

    x[mask] *= torch.sqrt(r_boundary_sq[mask] / r_sq[mask]).unsqueeze(1)

    return x
