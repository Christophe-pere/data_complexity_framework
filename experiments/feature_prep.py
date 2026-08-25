"""Feature preparation for Phase B: PCA-reduce / zero-pad / scale per encoding library §1.3.

Encodings expect feature vectors of a specific length (n_qubits for angle/ZZ/reup;
2^n_qubits for amplitude). Raw datasets have arbitrary feature dimension d, so we
must reduce or pad before encoding. The PCA basis is fit on the train split and
applied identically to the test split — this is what makes Phase B's reduction
step deterministic per (dataset, encoding) pair.

The choice of "angular range" matters for angle / ZZ / re-uploading encodings:
features are interpreted as rotation angles, so out-of-range values alias. We
standardize per feature (μ=0, σ=1) then scale by π so ~95% of inputs land in
[-2π, 2π] — a reasonable working range.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qmlc.encodings import AmplitudeEncoding, EncodingBase


@dataclass
class FeaturePrep:
    """Fitted transform: standardize → (PCA reduce | zero-pad) → output_dim features."""

    output_dim: int            # target dimension (n_qubits for angle/ZZ/reup; 2^n for amplitude)
    scaler: StandardScaler
    pca: Optional[PCA]         # None if input_dim ≤ output_dim
    input_dim: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        elif X.shape[1] < self.output_dim:
            pad = np.zeros((X.shape[0], self.output_dim - X.shape[1]), dtype=np.float64)
            X = np.hstack([X, pad])
        return X


def fit_feature_prep(X_train: np.ndarray, encoding: EncodingBase) -> FeaturePrep:
    """Fit a transform from raw features to the encoding's expected dimension."""
    X_train = np.asarray(X_train, dtype=np.float64)
    input_dim = X_train.shape[1]
    if isinstance(encoding, AmplitudeEncoding):
        output_dim = 2 ** encoding.n_qubits
    else:
        output_dim = encoding.n_qubits

    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)

    pca: Optional[PCA] = None
    if input_dim > output_dim:
        pca = PCA(n_components=output_dim, random_state=0).fit(Xs)

    return FeaturePrep(output_dim=output_dim, scaler=scaler, pca=pca, input_dim=input_dim)
