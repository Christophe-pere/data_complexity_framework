"""Variational quantum binary classifier.

Forward pass:
    f(x; θ) = ⟨ψ(x; θ)| Z₀ ⊗ I |ψ(x; θ)⟩         ∈ [-1, 1]
    p(x; θ) = (f + 1) / 2                          ∈ [0, 1]

with ψ produced by either:
    - Standard:      ψ = ansatz.apply(encoding.encode(x), θ)
    - Re-uploading:  ψ = apply_reuploading_circuit(x, n_qubits, n_reuploads, ansatz, θ)
                     (auto-detected from the encoding being a `DataReuploading` instance)

Loss: binary cross-entropy on labels y ∈ {0, 1}.

Gradient: parameter-shift rule (exact for Ry/Rz gates with ±1/2 generator
eigenvalues), applied per parameter per batch sample. Cost per gradient step:
2 · n_params · batch_size forward passes.

This is the baseline trainer used in Phase B to produce ε_train and ε_test
for the quantum branch of the (e_obs, gap_obs) record. Convergence is not
optimized — Phase B sweeps will likely upgrade to Adam / mini-batch shuffling
once the harness is validated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .ansatze import (
    AnsatzBase,
    _expectation_z0,
    _expectation_z0_batch,
    apply_reuploading_circuit,
    apply_reuploading_circuit_batch,
)
from .encodings import DataReuploading, EncodingBase


_LOSS_EPS = 1e-7


def encode_states(encoding: EncodingBase, X: np.ndarray, feature_scale: float = math.pi) -> np.ndarray:
    """Encode a (pre-unscaled) feature matrix into the stack of embedded states
    ρ_φ(x_i) = |ψ_enc(x_i)⟩, shape (n, 2**n_qubits) complex.

    This is the encoding-state cache mandated by the Phase B compute plan: compute
    once per (dataset, encoding, seed) and reuse across every ansatz × the whole
    SGD inner loop. Only valid for the standard (non-re-uploading) path, where the
    embedded state is independent of the trainable parameters θ. Features are
    scaled by `feature_scale` to match `QuantumBinaryClassifier._scale`.
    """
    Xs = np.asarray(X, dtype=np.float64) * feature_scale
    return np.stack([encoding.encode(x) for x in Xs], axis=0)


@dataclass
class TrainingHistory:
    epoch: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)


class QuantumBinaryClassifier:
    """Encoding + ansatz + ⟨Z₀⟩ measurement, trained by parameter-shift SGD."""

    def __init__(
        self,
        encoding: EncodingBase,
        ansatz: AnsatzBase,
        feature_scale: float = math.pi,
    ):
        self.encoding = encoding
        self.ansatz = ansatz
        self.feature_scale = feature_scale
        self._is_reupload = isinstance(encoding, DataReuploading)
        if self._is_reupload:
            self._n_params = encoding.n_reuploads * ansatz.n_params
        else:
            self._n_params = ansatz.n_params
        self.params: Optional[np.ndarray] = None

    @property
    def n_params(self) -> int:
        return self._n_params

    @property
    def n_qubits(self) -> int:
        return self.encoding.n_qubits

    # ------------------------------------------------------------------ forward

    def _forward_one(self, x: np.ndarray, theta: np.ndarray) -> float:
        """⟨Z₀⟩ for one sample at parameters θ. x is pre-scaled."""
        if self._is_reupload:
            psi = apply_reuploading_circuit(x, self.n_qubits, self.encoding.n_reuploads, self.ansatz, theta)
        else:
            psi = self.encoding.encode(x)
            psi = self.ansatz.apply(psi, theta)
        return _expectation_z0(psi, self.n_qubits)

    def _forward_enc(self, psi_enc: np.ndarray, theta: np.ndarray) -> float:
        """⟨Z₀⟩ for one pre-encoded state |ψ_enc⟩ at θ (standard path only)."""
        return _expectation_z0(self.ansatz.apply(psi_enc, theta), self.n_qubits)

    def _forward_enc_batch(self, enc_stack: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """⟨Z₀⟩ over a batch of pre-encoded states (standard path). Propagates the
        whole batch in one set of tensordot ops via ansatz.apply_batch."""
        psi = self.ansatz.apply_batch(enc_stack, theta)
        return _expectation_z0_batch(psi, self.n_qubits)

    def _forward_reup_batch(self, Xs_batch: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """⟨Z₀⟩ over a batch for the re-uploading path (pre-scaled Xs_batch)."""
        psi = apply_reuploading_circuit_batch(
            Xs_batch, self.n_qubits, self.encoding.n_reuploads, self.ansatz, theta
        )
        return _expectation_z0_batch(psi, self.n_qubits)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Rescale features to the encoding's natural angular range."""
        return np.asarray(X, dtype=np.float64) * self.feature_scale

    # ------------------------------------------------------------------ predict

    def predict_proba(
        self,
        X: np.ndarray,
        params: Optional[np.ndarray] = None,
        enc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        params = self.params if params is None else params
        if params is None:
            raise RuntimeError("classifier not initialized — call fit() first")
        if enc is not None and not self._is_reupload:
            f = self._forward_enc_batch(enc, params)
        elif self._is_reupload:
            f = self._forward_reup_batch(self._scale(X), params)
        else:
            Xs = self._scale(X)
            f = np.array([self._forward_one(x, params) for x in Xs])
        return np.clip((f + 1.0) / 2.0, 0.0, 1.0)

    def predict(
        self, X: np.ndarray, params: Optional[np.ndarray] = None, enc: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return (self.predict_proba(X, params=params, enc=enc) >= 0.5).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray, enc: Optional[np.ndarray] = None) -> float:
        """Classification accuracy."""
        return float(np.mean(self.predict(X, enc=enc) == np.asarray(y, dtype=np.int64)))

    def loss(
        self, X: np.ndarray, y: np.ndarray, params: Optional[np.ndarray] = None,
        enc: Optional[np.ndarray] = None,
    ) -> float:
        p = self.predict_proba(X, params=params, enc=enc)
        p = np.clip(p, _LOSS_EPS, 1.0 - _LOSS_EPS)
        y = np.asarray(y, dtype=np.float64)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    # ------------------------------------------------------------------ gradient

    def _batch_gradient(
        self, Xs_batch: np.ndarray, y_batch: np.ndarray, enc_batch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Parameter-shift gradient of mean BCE loss over a batch.

        If `enc_batch` (pre-encoded states) is given and the path is standard, the
        forward passes reuse the cached embedding instead of re-encoding — the
        embedding is θ-independent, so this is numerically identical to the
        re-encoding path. Otherwise `Xs_batch` (pre-scaled features) is used.
        """
        use_enc = enc_batch is not None and not self._is_reupload
        if use_enc:
            fwd = lambda theta: self._forward_enc_batch(enc_batch, theta)
        elif self._is_reupload:
            fwd = lambda theta: self._forward_reup_batch(Xs_batch, theta)
        else:
            fwd = lambda theta: np.array([self._forward_one(x, theta) for x in Xs_batch])
        n_p = self._n_params
        grad = np.zeros(n_p)
        shift = math.pi / 2.0
        # Precompute predictions at θ for the chain-rule factor
        f_center = fwd(self.params)
        p_center = np.clip((f_center + 1.0) / 2.0, _LOSS_EPS, 1.0 - _LOSS_EPS)
        # dL/dp · dp/df = -(y/p - (1-y)/(1-p)) · 1/2
        dL_df = 0.5 * (-(y_batch / p_center) + ((1.0 - y_batch) / (1.0 - p_center)))
        for k in range(n_p):
            theta_plus = self.params.copy(); theta_plus[k] += shift
            theta_minus = self.params.copy(); theta_minus[k] -= shift
            df_dtheta_k = 0.5 * (fwd(theta_plus) - fwd(theta_minus))
            grad[k] = float(np.mean(dL_df * df_dtheta_k))
        return grad

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.1,
        seed: int = 0,
        verbose: bool = False,
        record_every: int = 1,
        enc: Optional[np.ndarray] = None,
    ) -> "QuantumBinaryClassifier":
        """Mini-batch SGD with parameter-shift gradients on BCE loss.

        `enc` is an optional pre-computed encoding-state cache for `X` (see
        `encode_states`). When supplied (standard path), every forward pass in
        training and the verbose-mode loss/accuracy reuses it instead of
        re-encoding — the mandated per-(dataset, encoding, seed) reuse across
        ansätze. For the standard path without `enc`, the cache is built once
        here so it is still reused across all epochs / batches / param-shifts.
        """
        rng = np.random.default_rng(seed)
        self.params = self.ansatz.random_params(seed=seed)
        if self._is_reupload:
            # Tile the ansatz init across the n_reuploads layers
            base = self.ansatz.random_params(seed=seed)
            self.params = np.tile(base, self.encoding.n_reuploads).copy()
            # Re-randomize beyond the first copy so layers aren't degenerate
            for ell in range(1, self.encoding.n_reuploads):
                start = ell * self.ansatz.n_params
                self.params[start:start + self.ansatz.n_params] = self.ansatz.random_params(seed=seed + ell)

        Xs = self._scale(X)
        y = np.asarray(y, dtype=np.float64)
        n = len(Xs)

        # Build the encoding-state cache once (standard path) if not supplied.
        if enc is None and not self._is_reupload:
            enc = np.stack([self.encoding.encode(x) for x in Xs], axis=0)

        history = TrainingHistory()
        for epoch in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                enc_batch = None if enc is None else enc[idx]
                grad = self._batch_gradient(Xs[idx], y[idx], enc_batch=enc_batch)
                self.params -= lr * grad
            if verbose and (epoch % record_every == 0 or epoch == epochs - 1):
                lo = self.loss(X, y, enc=enc)
                ac = self.score(X, y, enc=enc)
                history.epoch.append(epoch)
                history.loss.append(lo)
                history.accuracy.append(ac)
                print(f"  epoch {epoch:3d}: loss={lo:.4f}  acc={ac:.4f}")
        self.history = history
        return self
