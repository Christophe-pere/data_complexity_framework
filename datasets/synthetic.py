import numpy as np

def generate_parity_dataset(n_bits=6, n_samples=1000, noise=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    X = rng.integers(0, 2, size=(n_samples, n_bits))
    y = np.mod(np.sum(X, axis=1), 2)
    flips = rng.random(n_samples) < noise
    y[flips] = 1 - y[flips]
    return X, y