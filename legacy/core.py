from metrics.entropy import dataset_entropy
from metrics.interaction import interaction_info
from metrics.compression import compression_complexity

def compute_complexity_card(X, y):
    S = dataset_entropy(X)
    Icorr = interaction_info(X, y)
    K = compression_complexity(X, y)
    C_data = S + Icorr + K
    return {
        "Entropy": S,
        "Interaction": Icorr,
        "Compression": K,
        "C_data": C_data
    }