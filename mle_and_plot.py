from scipy.optimize import minimize
from scipy.special import expit
import numpy as np
from hawkes_simulation import hawkes_sim
import matplotlib.pyplot as plt

# Independent controls for fitting and visualization.
n_train_events = 100
total_horizon = 200

mu_true = 0.5
alpha_true = 0.8
beta_true = 1.2
events = hawkes_sim(
    mu=mu_true,
    alpha=alpha_true,
    beta=beta_true,
    horizon=total_horizon,
    seed=7,
)
events = np.asarray(events)
events = np.sort(events)

if not 1 <= n_train_events <= len(events):
    raise ValueError(
        f"n_train_events must be between 1 and {len(events)}, "
        f"got {n_train_events}."
    )

# MLE only sees these events. Because the split is specified by event count,
# its observation window ends at the final training event.
train_events = events[:n_train_events]
train_horizon = train_events[-1]
observed_rate = len(train_events) / train_horizon

def unpack_params(theta):
    mu = np.exp(theta[0])
    beta = np.exp(theta[1])
    ratio = 0.999 * expit(theta[2])
    alpha = ratio * beta
    return mu, alpha, beta

def negative_log_likelihood(theta, events, T):
    mu, alpha, beta = unpack_params(theta)
    log_intensity_sum = 0
    excitation = 0
    for i, j in enumerate(events):
        if i > 0 :
            delta_t = j - events[i-1]
            excitation = (
                np.exp(-beta * delta_t)
                *
                (excitation + 1.0)
            )
        lambda_i = mu + alpha * excitation
        if lambda_i <= 0 or not np.isfinite(lambda_i):
            return np.inf
    
        log_intensity_sum += np.log(lambda_i)

    integral = mu * T + (alpha / beta) * np.sum(1- np.exp(-beta * (T - events)))
    return -(log_intensity_sum - integral)


mu_initial = max(0.5 * observed_rate, 1e-3)
beta_initial = 1.0
branching_initial = 0.5
theta_initial = np.log([mu_initial, beta_initial, branching_initial / (1 - branching_initial)])

result = minimize(
    negative_log_likelihood,
    theta_initial,
    args=(train_events, train_horizon),
    method="L-BFGS-B",
)
mu_hat, alpha_hat, beta_hat = unpack_params(result.x)

result_full = minimize(
    negative_log_likelihood,
    theta_initial,
    args=(events, total_horizon),
    method="L-BFGS-B",
)
mu_full, alpha_full, beta_full = unpack_params(result_full.x)
print(f"Training events: {len(train_events)}")
print(f"Training horizon: {train_horizon:.4f}")
print(f"Plot horizon: {total_horizon:.4f}")
print("Estimated parameters:")
print(f"mu: {mu_hat}")
print(f"alpha: {alpha_hat}")
print(f"beta: {beta_hat}")
print()

print("Estimated parameters (full data):")
print(f"mu: {mu_full}")
print(f"alpha: {alpha_full}")
print(f"beta: {beta_full}")
print()

print("True parameters:")
print(f"mu: {mu_true}")
print(f"alpha: {alpha_true}")
print(f"beta: {beta_true}")
t_grid = np.linspace(0, total_horizon, 3000)

time_difference = (
    t_grid[:, None] - events[None, :]
)

simulated_intensity = mu_true + alpha_true * np.sum(
    np.exp(-beta_true * time_difference) * (time_difference > 0),
    axis=1,
)

simulation_fig, simulation_ax = plt.subplots(figsize=(10, 4))

simulation_ax.plot(
    t_grid,
    simulated_intensity,
    label="Simulated intensity",
    color="tab:blue",
)

simulation_ax.vlines(
    events,
    0,
    0.1,
    color="tab:red",
    linewidth=1,
    label="Events",
)

simulation_ax.axhline(
    mu_true,
    linestyle="--",
    color="gray",
    label=r"True $\mu$",
)

simulation_ax.set_xlim(0, total_horizon)
simulation_ax.set_ylim(bottom=0)
simulation_ax.set_xticks(np.linspace(0, total_horizon, 9))
simulation_ax.set_xlabel("Time")
simulation_ax.set_ylabel("Intensity")
simulation_ax.set_title("Simulated Hawkes Process")
simulation_ax.legend()
simulation_fig.tight_layout()
simulation_fig.savefig("hawkes_simulated_intensity.png", dpi=300)
plt.close(simulation_fig)

estimated_intensity = mu_hat + alpha_hat * np.sum(
    np.exp(-beta_hat * time_difference) * (time_difference > 0),
    axis=1,
)

full_data_intensity = mu_full + alpha_full * np.sum(
    np.exp(-beta_full * time_difference) * (time_difference > 0),
    axis=1,
)

full_data_fig, full_data_ax = plt.subplots(figsize=(10, 4))

full_data_ax.plot(
    t_grid,
    full_data_intensity,
    label="Full-data estimated intensity",
    color="tab:blue",
)

full_data_ax.vlines(
    events,
    0,
    0.1,
    color="tab:red",
    linewidth=1,
    label="Training events (all data)",
)

full_data_ax.axhline(
    mu_full,
    linestyle="--",
    color="gray",
    label=r"Full-data estimated $\mu$",
)

full_data_ax.set_xlim(0, total_horizon)
full_data_ax.set_ylim(bottom=0)
full_data_ax.set_xticks(np.linspace(0, total_horizon, 9))
full_data_ax.set_xlabel("Time")
full_data_ax.set_ylabel("Intensity")
full_data_ax.set_title("Fitted Hawkes Process Using Full Data")
full_data_ax.legend()
full_data_fig.tight_layout()
full_data_fig.savefig("hawkes_full_data_estimated_intensity.png", dpi=300)
plt.close(full_data_fig)


fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    t_grid,
    estimated_intensity,
    label="Estimated intensity",
    color="tab:blue"
)

ax.vlines(
    train_events,
    0,
    0.1,
    color="tab:red",
    linewidth=1,
    label="Training events",
)

held_out_events = events[n_train_events:]
ax.vlines(
    held_out_events,
    0,
    0.1,
    color="tab:orange",
    linewidth=1,
    label="Held-out events",
)

ax.axvline(
    train_horizon,
    linestyle=":",
    color="black",
    linewidth=1,
    label="Training cutoff",
)

ax.axhline(
    mu_hat,
    linestyle="--",
    color="gray",
    label=r"Estimated $\mu$"
)

ax.set_xlim(0, total_horizon)
ax.set_ylim(bottom=0)
ax.set_xticks(
    np.linspace(0, total_horizon, 9)
)
ax.set_xlabel("Time")
ax.set_ylabel("Intensity")
ax.legend()
fig.tight_layout()
fig.savefig("hawkes_estimated_intensity.png", dpi=300)
plt.close(fig)
