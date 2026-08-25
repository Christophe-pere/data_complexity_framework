"""Phase C training-data loader — real (D, φ, ansatz, seed) → TrainingPoint.

Drop-in replacement for `analysis.synthetic_training_data.make_synthetic_training_set`:
reads the Phase B results JSONL (`experiments/results/phase_b.jsonl`) produced by
`experiments/sweep_phase_b.py` and emits the `TrainingPoint` list the predictor
harness consumes (`qmlc.predictor.stack_training_points` → `qmlc.cv.run_lodocv`).

Feature layout (real, not fabricated):
  v(D)  — all 9 implemented components, in `feature_extraction.V_COMPONENTS` order (K=9).
  q(D,φ)— the 3 *implemented* components (entanglement mean/var, embedded kernel
          effective rank); the 3 not-yet-implemented q components are dropped, so
          M=3. (framework_v2 specifies M=6; the loader exposes `q_components` so M
          grows automatically once the remaining estimators land.)

Edge target:
  The frozen primary target is ê_budget (framework_v2 §2.5), now implemented as
  the fidelity-discounted edge in `qmlc.budget` (chosen 2026-06-28): the noiseless
  quantum signal is shrunk toward chance by F = (1-ε₂)^{n₂q}, the survival fidelity
  of its two-qubit-gate count under the frozen reference noise model. New Phase B
  records store `e_budget`; this loader reads it for `edge_metric="budget"` (and
  back-fills via `compute_e_budget` if an older record has None). The default
  remains `edge_metric="raw"` (ê_raw, a §2.5 *diagnostic*, with a warning) — for
  **confirmatory** P5/H5 pass `edge_metric="budget"`.

v normalization: `normalize_v` controls a *global* rank-normalization (framework_v2
§2.1 Eq. 2.1) done here at load time. For **confirmatory** LODO-CV this is leaky
(the held-out dataset informs the rank transform), so the trainer loads with
`normalize_v=False` (raw v) and `qmlc.cv.run_lodocv(rank_normalize_k=K)` rank-
normalizes the v block **per fold**, fit on outer-train rows only. Global
normalization remains available for quick exploratory inspection.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from qmlc.predictor import TrainingPoint

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.ansatze import all_phase_b_ansatze  # noqa: E402
from qmlc.budget import budget_matched_edge  # noqa: E402
from qmlc.encodings import all_phase_b_configurations  # noqa: E402
from experiments.feature_extraction import (  # noqa: E402
    Q_COMPONENTS_IMPLEMENTED,
    V_COMPONENTS,
)
from experiments.results import iter_records  # noqa: E402
from analysis.synthetic_training_data import _rank_normalize  # noqa: E402


# Cache encoding/ansatz maps per n_qubits for on-the-fly ê_budget back-fill.
_ENC_MAPS: dict[int, dict] = {}
_ANS_MAPS: dict[int, dict] = {}


def compute_e_budget(record) -> float:
    """Fidelity-discounted budget-matched edge ê_budget for a record (framework_v2
    §2.5; see `qmlc.budget`). Reconstructs the encoding/ansatz objects from the
    record's ids to recover the two-qubit-gate count. Used to back-fill records
    whose stored `e_budget` is None."""
    nq = record.n_qubits
    enc_map = _ENC_MAPS.setdefault(nq, all_phase_b_configurations(nq))
    ans_map = _ANS_MAPS.setdefault(nq, all_phase_b_ansatze(nq))
    return budget_matched_edge(
        record.epsilon_test_q, record.epsilon_test_c,
        enc_map[record.encoding_id], ans_map[record.ansatz_id],
    )


@dataclass(frozen=True)
class LoaderConfig:
    v_components: tuple[str, ...] = V_COMPONENTS                 # K = 9
    q_components: tuple[str, ...] = Q_COMPONENTS_IMPLEMENTED      # M = 3 (implemented)
    edge_metric: str = "raw"                                     # "raw" | "budget" | "pareto"
    normalize_v: bool = True                                     # rank-normalize per component
    min_seeds_per_dataset: int = 1                               # drop datasets with fewer


def _v_value(vdict: dict, comp: str) -> float:
    entry = vdict.get(comp)
    if entry is None or entry.get("value") is None:
        return np.nan
    return float(entry["value"])


def load_phase_b_training_set(
    jsonl_path: Path | str = "experiments/results/phase_b.jsonl",
    config: Optional[LoaderConfig] = None,
) -> tuple[list[TrainingPoint], dict]:
    """Load Phase B records into (points, meta). `meta` mirrors the synthetic
    loader's `ground_truth` dict shape (no planted weights — real data)."""
    cfg = config if config is not None else LoaderConfig()
    path = Path(jsonl_path)
    records = list(iter_records(path))
    if not records:
        raise RuntimeError(f"no records in {path} — run experiments/sweep_phase_b.py first")

    if cfg.edge_metric == "raw":
        warnings.warn(
            "edge_metric='raw' (ê_raw) is a §2.5 diagnostic, not the frozen primary "
            "target ê_budget. Confirmatory selector validity (P5) requires ê_budget "
            "once budget calibration lands.", stacklevel=2,
        )

    # --- raw extraction ---
    raw_v: list[np.ndarray] = []
    rows: list[dict] = []
    for r in records:
        v_vec = np.array([_v_value(r.v, c) for c in cfg.v_components], dtype=np.float64)
        q_vec = np.array([r.q.get(c) for c in cfg.q_components], dtype=np.float64)
        if np.isnan(q_vec).any():
            raise ValueError(
                f"record {r.key()} has unimplemented/None q component in "
                f"{cfg.q_components}; restrict q_components to implemented ones"
            )
        if cfg.edge_metric == "raw":
            e_obs = float(r.e_raw)
        elif cfg.edge_metric == "budget":
            # Prefer the stored ê_budget; back-fill from costs if an older record
            # predates the budget calibration (e_budget=None).
            e_obs = float(r.e_budget) if r.e_budget is not None else compute_e_budget(r)
        else:
            raise ValueError(f"unknown edge_metric {cfg.edge_metric!r}")
        raw_v.append(v_vec)
        rows.append({
            "dataset_id": r.dataset_id, "encoding_id": r.encoding_id,
            "ansatz_id": r.ansatz_id, "seed": int(r.seed), "q": q_vec,
            "e_obs": e_obs, "gap_obs": float(r.gap_obs),
            "epsilon_emp": float(r.epsilon_emp),
        })

    v_stack = np.stack(raw_v, axis=0)
    # Impute any NaN v component with the column median before normalizing.
    for j in range(v_stack.shape[1]):
        col = v_stack[:, j]
        if np.isnan(col).any():
            med = np.nanmedian(col)
            warnings.warn(f"v component {cfg.v_components[j]!r} has NaNs; imputing median {med:.4g}",
                          stacklevel=2)
            col[np.isnan(col)] = med
    v_feat = _rank_normalize(v_stack) if cfg.normalize_v else v_stack

    # --- drop datasets below the seed-count threshold ---
    from collections import Counter
    seeds_per_ds = Counter()
    seen_pairs = set()
    for row in rows:
        key = (row["dataset_id"], row["seed"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            seeds_per_ds[row["dataset_id"]] += 1
    keep_ds = {d for d, c in seeds_per_ds.items() if c >= cfg.min_seeds_per_dataset}

    points: list[TrainingPoint] = []
    for i, row in enumerate(rows):
        if row["dataset_id"] not in keep_ds:
            continue
        points.append(TrainingPoint(
            dataset_id=row["dataset_id"], encoding_id=row["encoding_id"],
            ansatz_id=row["ansatz_id"], seed=row["seed"],
            v=v_feat[i].copy(), q=row["q"],
            e_obs=row["e_obs"], gap_obs=row["gap_obs"], epsilon_emp=row["epsilon_emp"],
        ))

    dataset_ids = sorted({p.dataset_id for p in points})
    meta = {
        "source": str(path),
        "n_points": len(points),
        "n_datasets": len(dataset_ids),
        "datasets": dataset_ids,
        "n_encodings": len({p.encoding_id for p in points}),
        "n_ansatze": len({p.ansatz_id for p in points}),
        "K": len(cfg.v_components),
        "M": len(cfg.q_components),
        "edge_metric": cfg.edge_metric,
        "normalize_v": cfg.normalize_v,
        "beta_planted": None,  # real data — no ground-truth β
    }
    return points, meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a Phase B training set")
    ap.add_argument("--jsonl", default="experiments/results/phase_b.jsonl")
    ap.add_argument("--edge-metric", default="raw")
    args = ap.parse_args()
    pts, meta = load_phase_b_training_set(args.jsonl, LoaderConfig(edge_metric=args.edge_metric))
    print(f"Loaded {meta['n_points']} points across {meta['n_datasets']} datasets "
          f"({meta['n_encodings']} enc × {meta['n_ansatze']} ans)")
    print(f"  feature dim = K({meta['K']}) + M({meta['M']}) = {meta['K'] + meta['M']}")
    print(f"  edge_metric = {meta['edge_metric']}")
    if pts:
        print(f"  e_obs   range = [{min(p.e_obs for p in pts):+.3f}, {max(p.e_obs for p in pts):+.3f}]")
        print(f"  gap_obs range = [{min(p.gap_obs for p in pts):+.3f}, {max(p.gap_obs for p in pts):+.3f}]")
