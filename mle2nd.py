from pathlib import Path
from scipy.special import expit
from scipy.optimize import minimize
import numpy as np
from hawkes_simulation import hawkes_sim
import matplotlib.pyplot as plt
T = 200

events = hawkes_sim(
    0.5,
    0.8,
    1.2,
    T,
    seed = 7
)

def unpack_params(params):
    mu = np.exp(params[0])
    ratio = 0.999 * expit(params[1])
    beta = np.exp(params[2])
    alpha = beta * ratio
    return mu, alpha, beta


def negative_log_likelihood(params, events, T):
    mu, alpha, beta = unpack_params(params)
    log_sum = 0
    excitation = 0
    for i,j in enumerate(events):
        if i > 0:
            delta_time = j - events[i-1]
            excitation = (
                np.exp(-beta * delta_time)
                *
                (excitation + 1)
            )
        lambda_i = mu + alpha * excitation
        if lambda_i <= 0 or not np.isfinite(lambda_i):
            return np.inf
        log_sum += np.log(lambda_i)

    integral = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - events)))

    return -log_sum + integral

def intensity(mu, alpha, beta, events, T):
    t_grid = np.linspace(0, T,  3000)
    time_difference = (
            t_grid[:, None] - events[None, :]
    )

    simulated_intensity = mu + alpha * np.sum(
        np.exp(-beta * time_difference) * (time_difference > 0),
        axis=1,
    )

    return simulated_intensity


if __name__ == '__main__':
    mu_initial = 0.25
    branching_initial = 0.5
    beta_initial = 1

    theta = np.log([mu_initial, branching_initial / (1 - branching_initial), beta_initial ])

    result = minimize(
        negative_log_likelihood,
        theta,
        args = (events, T),
        method = "L-BFGS-B",
        bounds = [(-20,20),(-20,20),(-20,20)]
    )
    test_set = hawkes_sim(
        0.5,
        0.8,
        1.2,
        200,
        seed = 2
    )
    nll = negative_log_likelihood(result.x, test_set, T)
    print(nll / len(events))
    true_params = (0.5, 0.8, 1.2)
    fig_width, fig_height = plt.rcParams["figure.figsize"]
    plt.figure(figsize=(fig_width * 1.5, fig_height))
    plt.plot(
        intensity(*unpack_params(result.x), test_set, T)
    )
    plt.plot(
        intensity(*true_params, test_set, T),
        linestyle = "dashed"
    )
    output_path = Path(__file__).resolve().with_name("evaluation.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
