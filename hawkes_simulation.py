import numpy as np

def hawkes_sim(mu, alpha, beta, horizon, seed = 7):
    rng = np.random.default_rng(seed)

    time = 0.0
    events = []
    excitation = 0.0

    while True:
        # Calculate the upper-bound on the intensity
        upper_bound = mu + excitation
        if upper_bound <= 0:
            break

        # Generate candidate event time and waiting time
        candidate_time = time + rng.exponential(1 / upper_bound)

        # If the candidate time exceeds the horizon, we stop
        if candidate_time > horizon:
            break

        # Decay the excitation up to the candidate time.
        excitation *= np.exp(-beta * (candidate_time - time))
        candidate_intensity = mu + excitation

        # Decide whether to accept the candidate event by thinning
        if rng.uniform(0, 1) <= candidate_intensity / upper_bound:
            events.append(candidate_time)
            excitation += alpha  # Increase excitation due to the new event
            time = candidate_time
        else:
            time = candidate_time

    return np.asarray(events)

if __name__ == "__main__":
    events = hawkes_sim(
        mu=0.4,
        alpha=0.8,
        beta=1.5,
        horizon=20.0,
        seed=7,
    )

    print("Number of events:", len(events))
    print("First ten events:", events[:10])

        