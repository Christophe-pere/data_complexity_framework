"""Best-of-standard-classical baseline set for the Phase B edge (OSF amendment 2026-07-06).

Motivation: Phase B as-run used a single classical baseline (`LogisticRegression`), but
framework_v2 §2.5 defines the edge against the **best classical model**. The LR-only edge is
an artifact of a linear baseline — the first-analysis baseline-sensitivity check
(`analysis/phase_b_first_analysis.md`) showed the entire apparent quantum advantage (parity
family, 17% cell win-rate) collapses to ~1% and goes significantly negative on all 27 datasets
once a nonlinear classical baseline is allowed.

This module defines the frozen standard classical benchmark — one model per major family,
fixed sensible defaults (deliberately undertuned so a quantum loss is robust) — recomputes the
classical error per (dataset, seed), and augments the frozen Phase B records with the
best-of-set edge. The quantum errors are byte-identical to the sweep (deterministic), so no
quantum re-fit is needed; only the classical side is recomputed.

Outputs:
  experiments/results/classical_baselines.json    per-(dataset, seed) per-model test/train error
  experiments/results/phase_b_best_classical.jsonl same schema as phase_b.jsonl, with
                                                   epsilon_test_c / e_raw / e_budget replaced by
                                                   the best-of-set values (epsilon_test_c_lr etc.
                                                   preserved under new keys is NOT done — the
                                                   original phase_b.jsonl remains the LR record).

Usage:
  python -m analysis.classical_baselines            # compute + augment
  python -m analysis.classical_baselines --refit    # force classical re-fit (ignore cache)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qmlc.datasets import all_phase_b_specs
from qmlc.ansatze import all_phase_b_ansatze
from qmlc.encodings import all_phase_b_configurations
from qmlc.budget import budget_matched_edge

REPO = Path(__file__).resolve().parent.parent
PHASE_B = REPO / "experiments/results/phase_b.jsonl"
BASE_JSON = REPO / "experiments/results/classical_baselines.json"
OUT_JSONL = REPO / "experiments/results/phase_b_best_classical.jsonl"
SEEDS = (0, 1, 2, 3, 4)

# --- FROZEN standard classical benchmark set (OSF amendment 2026-07-06) ---
# One model per major family; fixed defaults. `best` = min test error over these.
def _model_set(seed: int) -> dict:
    return {
        "lr":  LogisticRegression(C=1.0, max_iter=2000, random_state=seed),          # linear (as-run)
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)), # instance
        "svm": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)),  # kernel
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64),
                             max_iter=800, alpha=1e-3, random_state=seed)),           # neural
        "gbm": HistGradientBoostingClassifier(random_state=seed),                     # boosted trees
        "rf":  RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1), # bagged trees
    }

MODEL_KEYS = ("lr", "knn", "svm", "mlp", "gbm", "rf")


def _binarize(y: np.ndarray) -> np.ndarray:
    yu = np.unique(y)
    return y if len(yu) == 2 else (y > np.median(y)).astype(np.int64)


def compute_baselines(refit: bool = False) -> dict:
    """Per (dataset, seed): test+train error of each model + the best (min test error)."""
    if BASE_JSON.exists() and not refit:
        return json.loads(BASE_JSON.read_text())
    specs = {s.name: s for s in all_phase_b_specs()}
    names = sorted(n for n, s in specs.items() if s.klass != "B")
    out: dict = {}
    t0 = time.time()
    for name in names:
        ds = specs[name].load()
        if ds.X is None or ds.y is None:
            continue
        y = _binarize(ds.y)
        out[name] = {}
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                ds.X, y, test_size=0.2, stratify=y, random_state=seed)
            rec = {}
            for mk, m in _model_set(seed).items():
                m.fit(Xtr, ytr)
                rec[mk] = {"test": float(1.0 - m.score(Xte, yte)),
                           "train": float(1.0 - m.score(Xtr, ytr))}
            best_mk = min(MODEL_KEYS, key=lambda k: rec[k]["test"])
            rec["best_model"] = best_mk
            rec["test"] = rec[best_mk]["test"]
            rec["train"] = rec[best_mk]["train"]
            out[name][str(seed)] = rec
        print(f"  {name:24s} done ({time.time()-t0:.0f}s)")
    BASE_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {BASE_JSON}")
    return out


def augment_records(baselines: dict) -> int:
    """Rewrite phase_b.jsonl with epsilon_test_c / e_raw / e_budget set to the best-of-set
    classical baseline. Quantum fields untouched. Returns records written."""
    enc_maps: dict[int, dict] = {}
    ans_maps: dict[int, dict] = {}
    n_written = 0
    lr_repro_err = 0.0
    with open(PHASE_B) as fin, open(OUT_JSONL, "w") as fout:
        for line in fin:
            r = json.loads(line)
            d, s = r["dataset_id"], str(r["seed"])
            b = baselines[d][s]
            eps_c_best = b["test"]
            # validate LR reproduction against the stored (LR) edge
            lr_repro_err = max(lr_repro_err, abs((b["lr"]["test"] - r["epsilon_test_q"]) - r["e_raw"]))
            nq = r["n_qubits"]
            if nq not in enc_maps:
                enc_maps[nq] = all_phase_b_configurations(nq)
                ans_maps[nq] = all_phase_b_ansatze(nq)
            enc = enc_maps[nq][r["encoding_id"]]
            ans = ans_maps[nq][r["ansatz_id"]]
            eq = r["epsilon_test_q"]
            r["epsilon_train_c"] = b["train"]
            r["epsilon_test_c"] = eps_c_best
            r["e_raw"] = float(eps_c_best - eq)
            r["e_budget"] = budget_matched_edge(eq, eps_c_best, enc, ans)
            # Only existing keys are mutated (no new keys) so iter_records' PhaseBRecord(**data)
            # loads this file unchanged. Per-model detail lives in classical_baselines.json.
            fout.write(json.dumps(r) + "\n")
            n_written += 1
    print(f"wrote {OUT_JSONL}: {n_written} records")
    print(f"LR-edge reproduction max error vs stored e_raw: {lr_repro_err:.2e}")
    return n_written


def main() -> int:
    p = argparse.ArgumentParser(description="Best-of-standard-classical baseline augmentation")
    p.add_argument("--refit", action="store_true", help="force classical re-fit (ignore cache)")
    args = p.parse_args()
    baselines = compute_baselines(refit=args.refit)
    augment_records(baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
