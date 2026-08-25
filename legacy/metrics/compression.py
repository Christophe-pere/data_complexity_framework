import gzip
import numpy as np

def compression_complexity(X, y):
    """Compression ratio proxy for dataset complexity."""
    raw = np.hstack([X, y[:, None]]).astype(np.int8).tobytes()
    compressed = gzip.compress(raw)
    return len(compressed) / len(raw)