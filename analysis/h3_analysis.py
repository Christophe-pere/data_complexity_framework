#!/usr/bin/env python3
"""H3 — expressibility-data alignment. Pre-registered test.

Decision rule (OSF, framework_v2 §2.6):
  PRIMARY   beta > 0 in Eq. 2.7, with 95% CI excluding zero, the CI constructed
            by bootstrap over leave-one-dataset-out folds.
  SECONDARY Spearman rho between predicted a(D,U) and observed -gap_obs on
            held-out folds satisfies rho >= 0.3.

The primary criterion is computed here from the per-fold beta estimates.

The secondary criterion is NOT computable from the persisted fold outputs: the
fold JSONs store per-point edge predictions (`e_hat_*`, `e_obs_test`) but not
per-point alignment scores a(D,U) or gap predictions, so the rho cannot be
recovered without re-running with extra instrumentation. We report the
per-fold gap R^2 distribution in its place and say so explicitly rather than
substituting a different statistic and calling it the registered one.

Run against the 2026-08-19 re-run (corrected gap/beta head), NOT the
2026-07-10 folds whose beta=0 / gap R^2 = -300 were a convention bug.

Usage:
  python -m analysis.h3_analysis --folds-dir analysis/phase_c_folds_rerun_20260819
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--folds-dir", default="analysis/phase_c_folds_rerun_20260819")
    p.add_argument("--out", default="analysis/h3_report.json")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    files = sorted(glob.glob(f"{args.folds_dir}/*.json"))
    if not files:
        print(f"no fold JSONs in {args.folds_dir}", file=sys.stderr)
        return 1

    holdouts, beta_gp, beta_gbm, gap_gp, gap_gbm = [], [], [], [], []
    for f in files:
        d = json.load(open(f))
        holdouts.append(d["held_out_dataset"])
        beta_gp.append(d["beta_gp"]); beta_gbm.append(d["beta_gbm"])
        gap_gp.append(d["gap_r2_gp"]); gap_gbm.append(d["gap_r2_gbm"])
    beta_gp = np.array(beta_gp, float); beta_gbm = np.array(beta_gbm, float)
    gap_gp = np.array(gap_gp, float); gap_gbm = np.array(gap_gbm, float)
    n = len(beta_gp)
    print(f"[H3] {n} folds from {args.folds_dir}")

    rng = np.random.default_rng(args.seed)

    def boot_ci(x: np.ndarray) -> tuple[float, float, float]:
        idx = rng.integers(0, len(x), size=(args.boot, len(x)))
        means = x[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return float(x.mean()), float(lo), float(hi)

    b_mean, b_lo, b_hi = boot_ci(beta_gp)
    g_mean, g_lo, g_hi = boot_ci(beta_gbm)

    print(f"\n  beta (GP)   mean {b_mean:+.4f}   95% CI ({b_lo:+.4f}, {b_hi:+.4f})")
    print(f"  beta (GBM)  mean {g_mean:+.4f}   95% CI ({g_lo:+.4f}, {g_hi:+.4f})")
    print(f"  per-fold beta range (GP): [{beta_gp.min():+.4f}, {beta_gp.max():+.4f}]"
          f"   folds with beta>0: {int((beta_gp > 0).sum())}/{n}")

    excludes_zero = bool(b_lo > 0.0)
    h3_primary = bool(b_mean > 0 and excludes_zero)
    print(f"\n[H3 PRIMARY] beta > 0 and 95% CI excludes zero -> {h3_primary}")

    print(f"\n  gap R^2 (GP)  mean {gap_gp.mean():+.4f}  median {np.median(gap_gp):+.4f}"
          f"  range [{gap_gp.min():+.4f}, {gap_gp.max():+.4f}]")
    print(f"  gap R^2 (GBM) mean {gap_gbm.mean():+.4f}  median {np.median(gap_gbm):+.4f}")
    print("\n[H3 SECONDARY] Spearman rho criterion NOT computable from persisted "
          "fold outputs (no per-point a(D,U) / gap predictions stored).")

    worst = np.argsort(gap_gp)[:3]
    print("\n  weakest gap folds:", ", ".join(
        f"{holdouts[i]} ({gap_gp[i]:+.3f})" for i in worst))

    rep = {
        "folds_dir": args.folds_dir, "n_folds": n, "holdouts": holdouts,
        "beta_gp": {"mean": b_mean, "ci95": [b_lo, b_hi],
                    "per_fold": beta_gp.tolist(),
                    "n_positive": int((beta_gp > 0).sum())},
        "beta_gbm": {"mean": g_mean, "ci95": [g_lo, g_hi]},
        "gap_r2_gp": {"mean": float(gap_gp.mean()),
                      "median": float(np.median(gap_gp)),
                      "min": float(gap_gp.min()), "max": float(gap_gp.max())},
        "gap_r2_gbm": {"mean": float(gap_gbm.mean()),
                       "median": float(np.median(gap_gbm))},
        "H3_primary_pass": h3_primary,
        "H3_secondary": "not computable from persisted fold outputs; "
                        "requires per-point alignment/gap predictions",
        "note": "run on the 2026-08-19 corrected-gap-head re-run; the 2026-07-10 "
                "beta=0 / gap R2=-300 values were a convention bug, now fixed",
    }
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
