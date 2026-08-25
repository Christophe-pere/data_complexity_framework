"""Aggregate Phase C per-fold JSONs → confirmatory selector report + H5 read.

Reads every `<out-dir>/<holdout>.json` written by run_phase_c_fold, computes the LODOcv summary
(mean/median edge R², ECE, gap R², β), the pooled leave-one-dataset-out R² (all held-out points
concatenated), and a fold-level bootstrap 95% CI on the mean edge R² — then states the H5 read
(selector validity threshold R² ≥ 0.4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--folds-dir", default="analysis/phase_c_folds")
    p.add_argument("--jsonl", default="experiments/results/phase_b_best_classical.jsonl")
    p.add_argument("--output", default="analysis/predictor_confirmatory_best_classical.json")
    p.add_argument("--r2-threshold", type=float, default=0.4)  # H5 decision rule
    args = p.parse_args()

    fold_files = sorted(Path(args.folds_dir).glob("*.json"))
    folds = [json.loads(f.read_text()) for f in fold_files]
    n_datasets = len({json.loads(l)["dataset_id"] for l in open(args.jsonl)})
    print(f"aggregating {len(folds)}/{n_datasets} folds from {args.folds_dir}")
    if len(folds) < n_datasets:
        missing = set(range(n_datasets))  # informational only
        print(f"  ⚠ INCOMPLETE: {n_datasets - len(folds)} folds not yet done — partial report")

    def col(k):
        return np.array([f[k] for f in folds], dtype=float)

    edge_r2_gp, edge_r2_gbm = col("edge_r2_gp"), col("edge_r2_gbm")

    # Pooled LODO R²: concatenate every held-out dataset's per-point predictions vs truth.
    e_obs = np.concatenate([np.array(f["e_obs_test"]) for f in folds])
    e_hat_gp = np.concatenate([np.array(f["e_hat_gp"]) for f in folds])
    e_hat_gbm = np.concatenate([np.array(f["e_hat_gbm"]) for f in folds])
    pooled_r2_gp = float(r2_score(e_obs, e_hat_gp))
    pooled_r2_gbm = float(r2_score(e_obs, e_hat_gbm))

    # Fold-level bootstrap 95% CI on mean edge R² (GP head — the H5 target).
    rng = np.random.default_rng(0)
    boot = edge_r2_gp[rng.integers(0, len(edge_r2_gp), size=(20000, len(edge_r2_gp)))].mean(1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    summary = {
        "n_folds": len(folds), "n_datasets_expected": n_datasets,
        "complete": len(folds) == n_datasets,
        "edge_r2_gp_mean": float(edge_r2_gp.mean()), "edge_r2_gp_median": float(np.median(edge_r2_gp)),
        "edge_r2_gp_ci95": ci,
        "edge_r2_gbm_mean": float(edge_r2_gbm.mean()), "edge_r2_gbm_median": float(np.median(edge_r2_gbm)),
        "edge_pooled_r2_gp": pooled_r2_gp, "edge_pooled_r2_gbm": pooled_r2_gbm,
        "edge_ece_gp_mean": float(col("edge_ece_gp").mean()),
        "gap_r2_gp_mean": float(col("gap_r2_gp").mean()), "gap_r2_gbm_mean": float(col("gap_r2_gbm").mean()),
        "beta_gp_mean": float(col("beta_gp").mean()), "beta_gbm_mean": float(col("beta_gbm").mean()),
    }
    # H5 read (primary: mean edge R² of the GP head vs threshold, with fold-bootstrap CI).
    h5_pass = summary["edge_r2_gp_mean"] >= args.r2_threshold and ci[0] >= args.r2_threshold
    summary["H5_threshold"] = args.r2_threshold
    summary["H5_pass"] = bool(h5_pass)

    out = {"summary": summary, "folds": folds}
    Path(args.output).write_text(json.dumps(out, indent=2))

    print("=" * 60)
    print(f"  edge R² GP   mean={summary['edge_r2_gp_mean']:+.3f}  median={summary['edge_r2_gp_median']:+.3f}"
          f"  95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}]")
    print(f"  edge R² GBM  mean={summary['edge_r2_gbm_mean']:+.3f}  median={summary['edge_r2_gbm_median']:+.3f}")
    print(f"  pooled LODO R²  GP={pooled_r2_gp:+.3f}  GBM={pooled_r2_gbm:+.3f}")
    print(f"  ECE_GP={summary['edge_ece_gp_mean']:.3f}  β_GP={summary['beta_gp_mean']:+.3f}")
    print(f"  H5 (R²≥{args.r2_threshold}): {'PASS' if h5_pass else 'FAIL'}"
          + ("" if summary["complete"] else "   [PARTIAL — not confirmatory until all folds done]"))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
