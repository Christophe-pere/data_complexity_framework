"""Classical shadows (random-Pauli) for the Mode-N honest classical baseline.

The quantum-native (Class B) classical competitor must NOT see the amplitude vector — that is
free O(2^n) tomography and manufactures a quantum edge the same way QRAM asymmetry does on the
classical side (framework_v2 §2.5 / H6 rationale). Instead it gets `T` single-copy randomized
Pauli measurements per state — classical shadows [Huang–Kueng–Preskill, Nat. Phys. 2020] — and
learns from k-local Pauli-expectation features estimated from them (shadow-based ML,
[Huang et al., Science 2022]). The measurement budget `T` is the matched resource.

Design decisions (locked 2026-07-06, `docs/class_b_quantum_native_spec.md` §2):
  * random single-qubit Pauli basis per qubit per snapshot (X/Y/Z uniform),
  * features = estimated 1- and 2-local Pauli expectations (k ≤ 2),
  * T grid {1e2, 1e3, 1e4}, noiseless state processing (shot noise deferred, limitation #5).

State convention matches the rest of the codebase: flat statevector of length 2**n, qubit 0 is
the most-significant index (row-major `reshape([2]*n)`), consistent with `qmlc.ansatze`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Single-qubit rotations U such that measuring Z on U|ψ⟩ ≡ measuring the Pauli on |ψ⟩:
#   X:  H            (H X H = Z)
#   Y:  H · S†       (rotates Y-eigenbasis onto computational basis)
#   Z:  I
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_ROT = {0: _H, 1: _H @ _SDG, 2: np.eye(2, dtype=complex)}  # 0=X, 1=Y, 2=Z
PAULI_LABELS = {0: "X", 1: "Y", 2: "Z"}


def _apply_1q(psi: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    t = psi.reshape([2] * n)
    t = np.tensordot(U, t, axes=([1], [q]))
    t = np.moveaxis(t, 0, q)
    return t.reshape(2 ** n)


def sample_shadows(psi: np.ndarray, T: int, n: int, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Draw `T` random-Pauli shadow snapshots of pure state `psi`.

    Returns
    -------
    bases   : (T, n) uint8   — measured Pauli per qubit (0=X,1=Y,2=Z)
    signs   : (T, n) int8     — ±1 measurement outcome (eigenvalue) per qubit
    """
    bases = rng.integers(0, 3, size=(T, n)).astype(np.uint8)
    signs = np.empty((T, n), dtype=np.int8)
    # Group snapshots by basis pattern so each distinct rotation is applied once.
    order = np.lexsort(bases.T)
    b_sorted = bases[order]
    # boundaries of equal-basis runs
    cut = np.concatenate(([0], 1 + np.where(np.any(np.diff(b_sorted, axis=0), axis=1))[0], [T]))
    dims = [2] * n
    out_signs = np.empty((T, n), dtype=np.int8)
    for a, b in zip(cut[:-1], cut[1:]):
        pattern = b_sorted[a]
        rot = psi
        for q in range(n):
            rot = _apply_1q(rot, _ROT[int(pattern[q])], q, n)
        probs = np.abs(rot) ** 2
        probs = probs / probs.sum()
        idx = rng.choice(2 ** n, size=(b - a), p=probs)          # sampled bitstrings
        bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int8)  # (m, n) qubit bits
        out_signs[a:b] = 1 - 2 * bits                            # 0→+1, 1→−1
    # unsort
    inv = np.empty(T, dtype=np.int64); inv[order] = np.arange(T)
    signs = out_signs[inv]
    return bases, signs


def enumerate_local_paulis(n: int, k_max: int = 2) -> list[tuple[tuple[int, int], ...]]:
    """All non-identity Pauli terms of weight 1..k_max as tuples of (qubit, pauli)."""
    terms: list[tuple[tuple[int, int], ...]] = []
    for w in range(1, k_max + 1):
        for support in itertools.combinations(range(n), w):
            for paulis in itertools.product(range(3), repeat=w):
                terms.append(tuple(zip(support, paulis)))
    return terms


def estimate_paulis(bases: np.ndarray, signs: np.ndarray,
                    terms: Sequence[tuple[tuple[int, int], ...]]) -> np.ndarray:
    """Single-copy shadow estimator of ⟨P⟩ for each Pauli term.

    For a term with support S: a snapshot contributes 3^|S| · ∏_{j∈S} sign_j iff every qubit in S
    was measured in that term's Pauli basis, else 0. Estimate = mean over snapshots.
    """
    T = bases.shape[0]
    out = np.empty(len(terms), dtype=np.float64)
    for i, term in enumerate(terms):
        qs = np.array([q for q, _ in term])
        ps = np.array([p for _, p in term])
        match = np.all(bases[:, qs] == ps[None, :], axis=1)       # (T,) basis matches on support
        prod = np.prod(signs[:, qs], axis=1)                      # (T,) ±1 product on support
        contrib = np.where(match, (3 ** len(term)) * prod, 0.0)
        out[i] = contrib.mean()
    return out


@dataclass
class ShadowFeatureConfig:
    T: int = 1000
    k_max: int = 2
    seed: int = 0


def shadow_feature_matrix(states: np.ndarray, n: int, cfg: ShadowFeatureConfig,
                          terms: Optional[Sequence] = None) -> tuple[np.ndarray, list]:
    """Build the classical ML feature matrix from T-shot shadows of each state.

    Parameters
    ----------
    states : (n_samples, 2**n) complex, one pure state per row.

    Returns
    -------
    features : (n_samples, n_terms) estimated k-local Pauli expectations.
    terms    : the Pauli-term list (column meaning), for provenance.
    """
    if terms is None:
        terms = enumerate_local_paulis(n, cfg.k_max)
    rng = np.random.default_rng(cfg.seed)
    feats = np.empty((len(states), len(terms)), dtype=np.float64)
    for i, psi in enumerate(states):
        b, s = sample_shadows(np.asarray(psi, dtype=complex), cfg.T, n,
                              np.random.default_rng(rng.integers(1 << 32)))
        feats[i] = estimate_paulis(b, s, terms)
    return feats, list(terms)


# ---------------------------------------------------------------------------
def _self_test() -> None:
    """Shadow estimates converge to exact Pauli expectations as T grows."""
    rng = np.random.default_rng(0)
    n = 3
    psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    psi /= np.linalg.norm(psi)

    # exact single-qubit expectations via Pauli matrices
    P = {0: np.array([[0, 1], [1, 0]], complex), 1: np.array([[0, -1j], [1j, 0]], complex),
         2: np.array([[1, 0], [0, -1]], complex)}

    def exact(term):
        d = dict(term)
        full = np.array([[1]], complex)
        for q in range(n):
            full = np.kron(full, P[d[q]] if q in d else np.eye(2, dtype=complex))
        return np.real(psi.conj() @ full @ psi)

    terms = enumerate_local_paulis(n, 2)
    ex = np.array([exact(t) for t in terms])
    for T in (2000, 50000):
        b, s = sample_shadows(psi, T, n, np.random.default_rng(1))
        est = estimate_paulis(b, s, terms)
        err = np.abs(est - ex).mean()
        print(f"  T={T:6d}: mean|est-exact| over {len(terms)} Paulis = {err:.4f}")
    assert err < 0.05, "shadow estimator not converging"
    print("  shadows self-test OK")


if __name__ == "__main__":
    _self_test()
