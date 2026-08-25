"""Budget-matched edge ê_budget — H5 primary target (framework_v2 §2.5).

Operationalization (fidelity-discounted edge, chosen 2026-06-28; pre-OSF):
Phase B simulates *noiselessly*, but Eq. 2.9 prices an encoding/ansatz by the
two-qubit-gate count it would execute on NISQ hardware, with a fidelity penalty
(1-ε₂)^{D·W}. We carry that same fidelity into the edge: a circuit with n₂q
two-qubit gates retains signal fraction F = (1-ε₂)^{n₂q} under the frozen
reference noise model, so its *effective* (hardware) accuracy shrinks toward
chance (½) by F. The budget-matched edge is then the effective quantum accuracy
minus the classical accuracy:

    n₂q       = D_φ·W_φ  +  (n_ru · D_a·W_a)        # enc + (repeated) ansatz
    F         = (1 - ε₂)^{n₂q}                       # ε₂ = 1e-3 (frozen)
    acc_q     = 1 - ε_test_q ,   acc_c = 1 - ε_test_c
    ê_budget  = [½ + F·(acc_q - ½)] - acc_c
              = (ε_test_c - ½) - F·(ε_test_q - ½)

Properties (sanity-checked in tests):
    F = 1 (zero two-qubit cost) ⇒ ê_budget = acc_q - acc_c = ê_raw.
    F → 0 (deep/expensive)      ⇒ ê_budget = ½ - acc_c (quantum signal erased).
    amplitude (n₂q≈256) is penalized ~23× vs angle-linear (n₂q≈14), per §2.5.

This needs no FLOP↔gate↔wall-clock constant and no extra Phase B data — only the
already-defined `encoding.depth()/gates_per_layer()` and `ansatz.depth()/
gates_per_layer()` (the n₂q in Eq. 2.9's exponent) and the frozen ε₂. Re-uploading
repeats the ansatz once per re-upload, so its two-qubit gates are counted n_ru
times; the re-uploading *encoding* depth already includes every re-upload's
encoding layer (reup-2 → 28, reup-4 → 56).
"""
from __future__ import annotations

from .ansatze import AnsatzBase
from .encodings import DataReuploading, EncodingBase

EPS_2_REFERENCE: float = 1e-3  # frozen reference two-qubit gate error (encoding_library §1.2)


def two_qubit_gate_count(encoding: EncodingBase, ansatz: AnsatzBase) -> int:
    """Total two-qubit gates n₂q of the executed circuit (encoding + ansatz).

    Standard path: D_φ·W_φ + D_a·W_a. Re-uploading: the ansatz is interleaved and
    applied once per re-upload, so its gates count n_reuploads times; the
    re-uploading encoding's depth already spans all re-upload encoding layers.
    """
    enc_n2q = encoding.depth() * encoding.gates_per_layer()
    ans_n2q = ansatz.depth() * ansatz.gates_per_layer()
    if isinstance(encoding, DataReuploading):
        return int(enc_n2q + encoding.n_reuploads * ans_n2q)
    return int(enc_n2q + ans_n2q)


def circuit_fidelity(n_2q: int, eps_2: float = EPS_2_REFERENCE) -> float:
    """Signal-survival fidelity F = (1 - ε₂)^{n₂q} under the reference noise model."""
    return float((1.0 - eps_2) ** n_2q)


def budget_matched_edge(
    epsilon_test_q: float,
    epsilon_test_c: float,
    encoding: EncodingBase,
    ansatz: AnsatzBase,
    eps_2: float = EPS_2_REFERENCE,
) -> float:
    """ê_budget for one (D, φ, ansatz, seed) cell — fidelity-discounted edge.

    `epsilon_test_q/c` are test *error rates* (1 - accuracy). Returns the
    budget-matched edge (positive ⇒ quantum beats classical at matched cost)."""
    n_2q = two_qubit_gate_count(encoding, ansatz)
    F = circuit_fidelity(n_2q, eps_2)
    # ê_budget = (ε_test_c - ½) - F·(ε_test_q - ½)
    return float((epsilon_test_c - 0.5) - F * (epsilon_test_q - 0.5))
