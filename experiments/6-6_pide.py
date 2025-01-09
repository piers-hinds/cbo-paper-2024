from rsde_opt import *
import pandas as pd
import math

STANDARD_NORMAL = torch.distributions.Normal(0.0, 1.0)


def bs_call(spot, strike, expiry, r, sigma):
    """ Computes the true value of a European call option under Black-Scholes assumptions
    """
    d1 = (torch.log(spot / strike) + (r + sigma ** 2 / 2) * expiry) / (sigma * torch.sqrt(expiry))
    d2 = d1 - sigma * torch.sqrt(expiry)
    return spot * STANDARD_NORMAL.cdf(d1) - strike * torch.exp(-r * expiry) * STANDARD_NORMAL.cdf(d2)


def merton_call(spot, t, r, sigma, alpha, gamma, rate):
    """Computes the true value of a European call option under the Merton jump-diffusion model
    """
    strike = torch.tensor(1.0)
    expiry = 3 - t

    beta = torch.exp(alpha + 0.5 * gamma * gamma) - 1
    partial_sum = 0
    for k in range(20):
        r_k = r - rate * beta + (k * torch.log(beta + 1)) / expiry
        sigma_k = torch.sqrt(sigma ** 2 + (k * gamma ** 2) / expiry)
        k_fact = torch.tensor(math.factorial(k))
        term = (torch.exp(-(beta + 1) * rate * expiry) * ((beta + 1) * rate * expiry) ** k / k_fact) * bs_call(
            spot, strike, expiry, r_k, sigma_k)
        partial_sum += term
        if (term.abs() < 1e-14).all():
            break
    return partial_sum


def vec_forward_map(theta, points):
    r = torch.tensor(0.00)
    sigma = theta[:, 0:1]
    alpha = theta[:, 1:2]
    gamma = theta[:, 2:3]
    rate = torch.tensor(1.0)
    t, x = points[:, 0], points[:, 1]

    t = t.unsqueeze(0)
    x = x.unsqueeze(0)
    return merton_call(x, t, r, sigma, alpha, gamma, rate)


if __name__ == "__main__":
    torch.manual_seed(1)
    data = pd.read_csv('merton.csv')
    time_space_points = torch.tensor(data.iloc[:, :-1].to_numpy(), dtype=torch.float32)
    observations = torch.tensor(data.iloc[:, -1].to_numpy(), dtype=torch.float32).view(1, -1)
    time_space_points = time_space_points
    observations = observations

    true_theta = torch.tensor([[0.1, -0.2, 0.3]])
    success_criterion = SuccessCriterion(true_theta.squeeze(0),
                                         0.01,
                                         'x_value')


    def objective(theta):
        preds = vec_forward_map(theta, time_space_points)
        expanded_observations = observations.expand(preds.shape[0], -1)
        mse = torch.sum((preds - expanded_observations) ** 2, dim=1)
        reg = 1e-5 * torch.linalg.norm(theta, dim=1)
        return mse + reg


    def projection(theta):
        theta[..., 0] = torch.clamp(theta[..., 0], min=1e-8, max=1)
        theta[..., 1] = torch.clamp(theta[..., 1], min=-1, max=1)
        theta[..., 2] = torch.clamp(theta[..., 2], min=1e-8, max=1)
        return theta


    def sample_initial_state(n):
        sigma = torch.rand(size=(n, 1))
        alpha = -torch.rand(size=(n, 1)) * 2 - 1
        gamma = torch.rand(size=(n, 1))
        return torch.cat([sigma, alpha, gamma], dim=1)


    num_experiments = 100
    T = 10
    num_steps = 100
    step_size = T / num_steps

    alpha = 10 ** 14
    num_particles = 400

    system = VecProjectionParticleSystem(objective=objective,
                                         num_experiments=num_experiments,
                                         projection=projection,
                                         initial_state=sample_initial_state,
                                         alpha=alpha,
                                         beta=lambda t: 10 * t,
                                         sigma=lambda t: 10 * torch.exp(-np.log(10) * t),
                                         dim=3,
                                         num_particles=num_particles,
                                         step_size=step_size,
                                         device='cpu')
    success_rate, _ = system.run_experiments(num_steps, success_criterion)
    print(success_rate)


    def plot_histograms(final_consensus):
        for i in range(3):
            plt.figure()
            plt.hist(final_consensus[:, i].cpu().numpy(), bins=30, alpha=0.7)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()

    plot_histograms(system.consensus())
