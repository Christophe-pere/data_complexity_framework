from collections import Counter
from math import log2
import numpy as np

def dataset_entropy(X):
    """Average entropy per feature."""
    n_samples, n_features = X.shape
    entropies = []
    for j in range(n_features):
        counts = Counter(X[:, j])
        probs = [c / n_samples for c in counts.values()]
        entropies.append(-sum(p * log2(p) for p in probs))
    return np.mean(entropies)

