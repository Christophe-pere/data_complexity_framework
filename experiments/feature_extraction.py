"""v(D) and q(D, φ) computation for Phase B records.

v(D) uses the implemented `qmlc.classical` metrics (9 components — same set used
in pre-screening). The four formerly-missing components (second_order_mi,
higher_order_mi via MINE, kernel_effective_rank via median-heuristic RBF, and
mdl_autoencoder) were implemented 2026-06-01. The only remaining frozen v(D)
component not computed here is v_κ (Ollivier-Ricci curvature), which is defined
for graph/hypergraph data only and is dropped for point-cloud datasets.

q(D, φ) is computed on a subsample of embedded states {ρ_φ(x_i)}:
    - entanglement_entropy_mean, _var: bipartite (half-cut) entropy via partial trace
    - embedded_kernel_effective_rank: effective rank of the fidelity Gram matrix
      K_ij = |⟨ψ_i|ψ_j⟩|²

The other q components (expressibility KL, locality / lightcone width, embedded
PH) are not yet implemented; they are flagged in the output as `null` for
downstream code to handle.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from qmlc.classical import (
    CompressionRatio,
    EntropyKL,
    HigherOrderMI,
    IntrinsicDimPCA,
    IntrinsicDimTwoNN,
    KernelEffectiveRankRBF,
    MDLAutoencoder,
    PairwiseMutualInformation,
    PersistentHomology,
)
from qmlc.quantum import _von_neumann, partial_trace


# v(D) components matching the registry's `screening.implemented_components`.
V_COMPONENTS: tuple[str, ...] = (
    "differential_entropy_knn_kl",
    "intrinsic_dim_pca",
    "intrinsic_dim_twonn",
    "compression_ratio_gzip",
    "topological_complexity_rips",
    "second_order_mi",
    "higher_order_mi",
    "kernel_effective_rank",
    "mdl_autoencoder",
)


def compute_v(X: np.ndarray, ph_subsample: int = 500, ph_max_dim: int = 2) -> dict[str, dict]:
    """Apply implemented v(D) metrics. Returns metric_name → {value, estimator, elapsed_s}.

    PH defaults are looser than screening (max_dim=2, n=min(500, |X|), default
    edge fraction) to align with the Phase B v_top0/v_top1/v_top2 target.
    """
    out: dict[str, dict] = {}
    metrics = [
        ("differential_entropy_knn_kl", EntropyKL(k=4), False),
        ("intrinsic_dim_pca", IntrinsicDimPCA(), False),
        ("intrinsic_dim_twonn", IntrinsicDimTwoNN(), False),
        ("compression_ratio_gzip", CompressionRatio(codec="gzip"), False),
        ("topological_complexity_rips", PersistentHomology(max_dim=ph_max_dim), True),
        ("second_order_mi", PairwiseMutualInformation(), False),
        ("higher_order_mi", HigherOrderMI(), False),
        ("kernel_effective_rank", KernelEffectiveRankRBF(), False),
        ("mdl_autoencoder", MDLAutoencoder(), False),
    ]
    rng = np.random.default_rng(42)
    for label, metric, subsample in metrics:
        X_in = X
        if subsample and X.shape[0] > ph_subsample:
            idx = rng.choice(X.shape[0], ph_subsample, replace=False)
            X_in = X[idx]
        try:
            r = metric.compute(X_in)
            out[label] = {
                "value": float(r.value),
                "uncertainty": float(r.uncertainty) if r.uncertainty is not None else None,
                "estimator": r.estimator,
                "elapsed_s": float(r.elapsed_s),
            }
            if label == "topological_complexity_rips":
                # Surface per-dim Betti for downstream v_top0/v_top1/v_top2 split
                betti = r.metadata.get("betti", {})
                out[label]["betti"] = {str(k): int(v) for k, v in betti.items()}
        except Exception as e:
            out[label] = {"value": None, "error": f"{type(e).__name__}: {e}"}
    return out


# q components implemented as of Phase A scaffolding.
Q_COMPONENTS_IMPLEMENTED: tuple[str, ...] = (
    "entanglement_entropy_mean",
    "entanglement_entropy_var",
    "embedded_kernel_effective_rank",
)
Q_COMPONENTS_NOT_IMPLEMENTED: tuple[str, ...] = (
    "expressibility_kl",
    "locality_lightcone_width",
    "embedded_persistent_homology",
)


def compute_q(states: np.ndarray, n_qubits: int, half_cut: Optional[list[int]] = None) -> dict[str, Optional[float]]:
    """Apply implemented q(D, φ) metrics on a stack of embedded states.

    Parameters
    ----------
    states : shape (n_samples, 2**n_qubits)
    n_qubits : int
    half_cut : qubit indices defining subsystem A for bipartite entropy
        (defaults to first n//2 qubits).

    Returns dict with all six q-component keys; not-implemented ones are None.
    """
    states = np.asarray(states, dtype=complex)
    if half_cut is None:
        half_cut = list(range(n_qubits // 2))

    # 1. Bipartite entanglement entropy (mean + var across samples)
    ents = []
    for psi in states:
        rho = np.outer(psi, psi.conj())
        rho_A = partial_trace(rho, half_cut, n=n_qubits)
        ents.append(_von_neumann(rho_A))
    ents = np.array(ents)

    # 2. Embedded kernel effective rank — fidelity kernel |⟨ψ_i|ψ_j⟩|²
    inner = states.conj() @ states.T            # (n, n)
    K = np.abs(inner) ** 2                       # PSD
    eig_K = np.linalg.eigvalsh(K).clip(0.0)
    num = float(eig_K.sum() ** 2)
    den = float(np.maximum((eig_K ** 2).sum(), 1e-30))
    eff_rank = num / den

    return {
        "entanglement_entropy_mean": float(ents.mean()),
        "entanglement_entropy_var": float(ents.var()),
        "embedded_kernel_effective_rank": float(eff_rank),
        "expressibility_kl": None,
        "locality_lightcone_width": None,
        "embedded_persistent_homology": None,
    }
