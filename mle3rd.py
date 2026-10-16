import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from hawkes_simulation import hawkes_sim

T = 200

def unpack(params):
    mu = np.exp(params[0])
    ratio = 0.999 * expit(params[1])
    beta = np.exp(params[2])
    alpha = ratio * beta
    return mu, alpha, beta

def negative_log_likelihood(params, events, T):
    mu, alpha, beta = unpack(params)
    log_sum = 0
    excitation = 0
    for i, j in enumerate(events):
        if i > 0:
            delta_time = j - events[i - 1]
            excitation = np.exp(-beta * delta_time) * (excitation + 1)
        lambda_i = mu + alpha * excitation
        if lambda_i <= 0 or not np.isfinite(lambda_i):
            return np.inf
        log_sum += np.log(lambda_i)

    integral = mu * T + (alpha / beta) * np.sum(
        1 - np.exp(-beta * (T - events))
    )
    return -log_sum + integral

def intensity(mu, alpha, beta, events, T, t_grid):
    time_difference = t_grid[:, None] - events[None, :]
    simulated_intensity = mu + alpha * np.sum(
        np.exp(-beta * time_difference) * (time_difference > 0),
        axis = 1
    )
    return simulated_intensity

def multi_loss(params, event_list, T):
    nll = 0
    for events in event_list:
        nll += negative_log_likelihood(params, events, T)
    return nll

if __name__ == "__main__":
    event_list = [hawkes_sim(0.5, 0.8, 1.2, T, seed) for seed in range(3,80)]
    test_events = hawkes_sim(0.5, 0.8, 1.2, 200, 2)
    mu_init = 0.25
    branching_init = 0.5
    beta_init = 1
    theta = np.log([mu_init, branching_init / (1 - branching_init), beta_init])
    result = minimize(
        multi_loss,
        theta,
        args = (event_list, T),
        method = "L-BFGS-B"
    )
    mu_hat, alpha_hat, beta_hat = unpack(result.x)
    print(mu_hat, alpha_hat, beta_hat)
    loss = negative_log_likelihood(result.x, test_events, T) / len(test_events)
    print(loss)