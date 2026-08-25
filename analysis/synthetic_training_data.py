"""Synthetic training data for predictor harness validation.

Phase B will populate (D, φ, ansatz, seed) → (v, q, e_obs, gap_obs, ε_emp)
tuples from real experiments. Until then, this module generates plausible
synthetic data with a known planted relationship so the harness can be
validated end-to-end.

Planted relationships (deterministic + Gaussian noise):
    e_true(v, q)   = α·tanh(w_e^T (v ⊕ q)) + bias_e
    gap_true(v, q) = β_planted · (1 - σ(w_a^T (v ⊕ q)))        # the gap (ε_gen − ε_emp)
    ε_emp_true     = small constant + small dataset-specific deviation

with sigmoid(w_a^T x) playing the role of the planted alignment a*. A
well-fit predictor should recover (i) reasonable R² on edge / gap, and
(ii) a β estimate close to β_planted.

`v` is read from the dataset registry's screening block (5 implemented
components per dataset, scaled to [0, 1] via min-max normalization across
the 13 Class A datasets); `q` is generated synthetically per (dataset,
encoding) pair so that different encodings yield different q vectors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from qmlc.predictor import TrainingPoint


# Components extracted from registry.json screening (in fixed order).
_REG_COMPONENTS = (
    "differential_entropy_knn_kl",
    "intrinsic_dim_pca",
    "intrinsic_dim_twonn",
    "compression_ratio_gzip",
    "topological_complexity_rips",
)


def _load_v_from_registry(registry_path: Path) -> dict[str, np.ndarray]:
    """Read v(D) for each registered dataset from the screening block.

    Returns dataset_id -> raw component vector (5,). The caller normalizes.
    """
    with open(registry_path) as f:
        reg = json.load(f)
    v_raw: dict[str, np.ndarray] = {}
    for d in reg["datasets"]:
        if not d["screening"]["passed"]:
            continue
        comps = d["screening"]["components"]
        # Skip quantum-native (Class B) datasets — their screening block carries
        # quantum-native components (entanglement, …), not the classical v(D) set.
        if any(c not in comps for c in _REG_COMPONENTS):
            continue
        vec = np.array([comps[c]["value"] for c in _REG_COMPONENTS], dtype=np.float64)
        v_raw[d["name"]] = vec
    return v_raw


def _rank_normalize(stack: np.ndarray) -> np.ndarray:
    """Per-column rank normalize to [0, 1]. Mirrors framework_v2 §2.1 Eq. 2.1."""
    n = stack.shape[0]
    out = np.empty_like(stack, dtype=np.float64)
    for j in range(stack.shape[1]):
        order = np.argsort(stack[:, j])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1) / n
        out[:, j] = ranks
    return out


@dataclass(frozen=True)
class SyntheticConfig:
    encodings: tuple[str, ...] = (
        "angle-none", "angle-linear", "angle-ring",
        "zz-r1-linear", "zz-r2-full", "amplitude", "reup-2", "reup-4",
    )
    ansatze: tuple[str, ...] = ("hea-L1", "hea-L3", "alt-L1", "alt-L3", "equiv-L1", "equiv-L3")
    n_seeds: int = 3
    q_dim: int = 6                  # framework_v2 §2.2 specifies M = 6
    beta_planted: float = 0.4
    edge_amplitude: float = 0.6
    edge_bias: float = 0.0
    eps_emp_base: float = 0.08
    e_obs_noise: float = 0.05
    gap_obs_noise: float = 0.03
    eps_emp_noise: float = 0.02
    seed: int = 0


def make_synthetic_training_set(
    registry_path: Path | str = "datasets/registry.json",
    config: Optional[SyntheticConfig] = None,
) -> tuple[list[TrainingPoint], dict[str, float]]:
    """Build a synthetic Phase-B-shaped training set.

    Returns (points, ground_truth) where ground_truth records the planted weights
    so harness tests can verify recovery.
    """
    cfg = config if config is not None else SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    # 1. v(D): rank-normalized from the real registry
    v_raw = _load_v_from_registry(Path(registry_path))
    if not v_raw:
        raise RuntimeError(f"no passed datasets in {registry_path}")
    dataset_ids = sorted(v_raw.keys())
    v_stack = np.stack([v_raw[d] for d in dataset_ids], axis=0)
    v_norm = _rank_normalize(v_stack)            # (n_datasets, 5)
    v_by_id = {d: v_norm[i] for i, d in enumerate(dataset_ids)}

    K = v_norm.shape[1]
    M = cfg.q_dim
    D = K + M

    # 2. q(D, φ): per-encoding offset + per-dataset modulation
    enc_offsets = rng.uniform(-1.0, 1.0, size=(len(cfg.encodings), M))
    enc_to_offset = {e: enc_offsets[i] for i, e in enumerate(cfg.encodings)}

    def q_for(dataset_id: str, encoding_id: str, seed: int) -> np.ndarray:
        # deterministic-ish per (dataset, encoding) with small per-seed jitter
        base = enc_to_offset[encoding_id] + 0.3 * np.sin(2.0 * v_by_id[dataset_id][:M] if K >= M else
                                                         np.tile(v_by_id[dataset_id], (M // K + 1))[:M])
        jitter = np.random.default_rng(hash((dataset_id, encoding_id, seed)) & 0xFFFFFFFF).normal(0, 0.05, size=M)
        return base + jitter

    # 3. Planted linear weights
    w_e = rng.normal(0, 1.0, size=D)
    w_a = rng.normal(0, 1.0, size=D)

    def edge_true(x: np.ndarray) -> float:
        return float(cfg.edge_amplitude * np.tanh(w_e @ x) + cfg.edge_bias)

    def alignment_true(x: np.ndarray) -> float:
        z = w_a @ x
        return float(1.0 / (1.0 + np.exp(-z)))

    # 4. Generate (D, φ, ansatz, seed) tuples
    points: list[TrainingPoint] = []
    for did in dataset_ids:
        v = v_by_id[did]
        for enc in cfg.encodings:
            for ansz in cfg.ansatze:
                for s in range(cfg.n_seeds):
                    q = q_for(did, enc, s)
                    x = np.concatenate([v, q])
                    e_clean = edge_true(x)
                    a_clean = alignment_true(x)
                    eps_emp = cfg.eps_emp_base + rng.normal(0, cfg.eps_emp_noise)
                    gap_clean = cfg.beta_planted * (1.0 - a_clean)
                    e_obs = e_clean + rng.normal(0, cfg.e_obs_noise)
                    # gap_obs is the generalization GAP (ε_gen − ε_emp), matching real Phase B
                    # (sweep stores eps_test_q − eps_train_q). Do NOT add ε_emp here — that was
                    # the pre-2026-07-10 convention mismatch that made solve_beta work on
                    # synthetic but collapse β→0 on real data. gap = β(1 − a) + noise.
                    gap_obs = gap_clean + rng.normal(0, cfg.gap_obs_noise)
                    points.append(TrainingPoint(
                        dataset_id=did,
                        encoding_id=enc,
                        ansatz_id=ansz,
                        seed=s,
                        v=v.copy(),
                        q=q,
                        e_obs=e_obs,
                        gap_obs=gap_obs,
                        epsilon_emp=eps_emp,
                    ))

    ground_truth = {
        "n_datasets": len(dataset_ids),
        "n_encodings": len(cfg.encodings),
        "n_ansatze": len(cfg.ansatze),
        "n_seeds": cfg.n_seeds,
        "n_points": len(points),
        "K": K,
        "M": M,
        "beta_planted": cfg.beta_planted,
        "edge_amplitude": cfg.edge_amplitude,
        "edge_bias": cfg.edge_bias,
    }
    return points, ground_truth


if __name__ == "__main__":
    pts, gt = make_synthetic_training_set()
    print(f"Generated {gt['n_points']} synthetic points "
          f"({gt['n_datasets']} datasets × {gt['n_encodings']} encodings "
          f"× {gt['n_ansatze']} ansatze × {gt['n_seeds']} seeds)")
    print(f"  K (intrinsic) = {gt['K']}, M (interaction) = {gt['M']}, total dim = {gt['K'] + gt['M']}")
    print(f"  β planted     = {gt['beta_planted']}")
    print(f"  e_obs range   = [{min(p.e_obs for p in pts):+.3f}, {max(p.e_obs for p in pts):+.3f}]")
    print(f"  gap_obs range = [{min(p.gap_obs for p in pts):+.3f}, {max(p.gap_obs for p in pts):+.3f}]")
