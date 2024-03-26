import torch
from rsde_opt import *
from functools import partial
from matplotlib.animation import FuncAnimation, PillowWriter


def ackley_function(x):
    x_part, y_part = x[:, 0]-2, x[:, 1]-2

    # First term
    a = 20
    b = 0.2
    c = 2 * torch.pi
    term1 = -a * torch.exp(-b * torch.sqrt(0.5 * (x_part ** 2 + y_part ** 2)))

    # Second term
    term2 = -torch.exp(0.5 * (torch.cos(c * x_part) + torch.cos(c * y_part)))

    # Final value (e is Euler's number, approx. 2.71828)
    ackley_value = term1 + term2 + a + torch.exp(torch.tensor(1.0))

    return ackley_value


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


def unit_circle_feasible_region(x, y, r=1):
    return x**2 + y**2 - r**2


def create_ax(objective, feasible_region):
    x = np.linspace(-3.1, 3.1, 400)
    y = np.linspace(-3.1, 3.1, 400)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.stack((x_grid.flatten(), y_grid.flatten()), axis=-1)

    # Evaluate the feasible region function
    z_feasible = feasible_region(x_grid, y_grid)

    # Evaluate the objective function (ensure points are a tensor for the example objective function)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    z_objective = objective(points_tensor).numpy().reshape(x_grid.shape)

    # Now create the plot with contours
    fig, ax = plt.subplots()

    # Contour plot for the feasible region
    # Adjust levels and colors as needed
    ax.contour(x_grid, y_grid, z_feasible, levels=[0, 0.001], cmap='Blues', alpha=1)

    # Contour plot for the objective function within the feasible region
    # Masking values outside the feasible region (optional)
    z_objective_masked = np.where(z_feasible <= 0, z_objective, np.nan)
    ax.contourf(x_grid, y_grid, z_objective_masked, levels=20, cmap='viridis', alpha=0.5)
    return ax


if __name__ == '__main__':
    r = 3
    system = ProjectionParticleSystem(ackley_function,
                                      partial(project_unit_ball, r=r),
                                      10,
                                      0.5,
                                      0.5,
                                      2,
                                      100,
                                      0.001,
                                      'cpu')

    logger = ParticleSystemLogger(system)
    torch.manual_seed(2)
    system.state = uniform_disk(system.num_particles) * 3

    system.consensus()

    for i in range(2000):
        norms = torch.randn_like(system.state)
        state, consensus = system.step(norms)
        logger.log_state(state, consensus)

    ax = create_ax(system.objective, partial(unit_circle_feasible_region, r=r))

    ani = animate_particles(logger, ax)
    # ani.save('C:/Users/pmxph7/OneDrive - The University of Nottingham/PhD/ccbo/plots/ackley.gif',
    #          writer=PillowWriter(fps=30))