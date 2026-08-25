#!/usr/bin/env python3
"""Is Eq. 2.4's functional form right? Model comparison on the n-scaling arm.

Eq. 2.4 assumes log Var[g] is *linear in the product n*d*. The n-scaling arm
(n in {4,6,8,10,12,14}) makes that assumption testable for the first time: at
fixed n=8 it is unfalsifiable, because n*d is then just a rescaling of d.

This is an EXPLORATORY analysis in the pre-registration's sense (it is not one
of H1-H6) and must be labelled as such in Part II. It does not change any
pre-registered decision rule; it explains why H1 fails.

Candidate forms for log Var[g]:
    (a) 1 + n*d      -- Eq. 2.4 as written
    (b) 1 + n + d    -- width and depth enter separately
    (c) 1 + n        -- width only
    (d) 1 + log n    -- power-law decay in width
    (e) 1 + n + n^2  -- curvature in width

Reference: McClean et al. give Var[g] ~ 2^-n for Haar-random circuits, i.e. a
coefficient of -ln 2 = -0.693 on n.

Usage:  python -m analysis.h1_functional_form [--datasets ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUBSET = [
    "synth-parity-3", "synth-torus", "synth-prng", "synth-concentric-rings",
    "real-iris", "real-breast-cancer", "real-higgs",
    "adv-easy-looking-hard", "adv-hard-looking-easy",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="experiments/results/h1_gradients.jsonl")
    p.add_argument("--datasets", nargs="*", default=SUBSET)
    p.add_argument("--out", default="analysis/h1_functional_form.json")
    p.add_argument("--boot", type=int, default=2000)
    args = p.parse_args()

    keep = set(args.datasets)
    y, n, d, ds = [], [], [], []
    with open(args.jsonl) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["dataset_id"] not in keep or r.get("log_grad_var") is None:
                continue
            y.append(r["log_grad_var"]); n.append(r["n_qubits"])
            d.append(r["depth"]); ds.append(r["dataset_id"])
    y = np.array(y); n = np.array(n, float); d = np.array(d, float); ds = np.array(ds)
    one = np.ones((len(y), 1))
    print(f"[form] {len(y)} records | n levels {sorted(set(n.tolist()))} "
          f"| depths {sorted(set(d.tolist()))} | {len(set(ds))} datasets")

    models = {
        "(a) Eq2.4: 1 + n*d":   np.hstack([one, (n * d)[:, None]]),
        "(b) 1 + n + d":        np.hstack([one, n[:, None], d[:, None]]),
        "(c) 1 + n":            np.hstack([one, n[:, None]]),
        "(d) 1 + log n":        np.hstack([one, np.log(n)[:, None]]),
        "(e) 1 + n + n^2":      np.hstack([one, n[:, None], (n ** 2)[:, None]]),
    }

    out = {}
    print(f"\n{'model':<22}{'R^2':>9}{'AIC':>12}   coefficients")
    for name, X in models.items():
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ beta
        rss = float(((y - pred) ** 2).sum())
        r2 = 1.0 - rss / float(((y - y.mean()) ** 2).sum())
        a = len(y) * np.log(rss / len(y)) + 2 * X.shape[1]
        out[name] = {"r2": r2, "aic": a, "coef": [float(b) for b in beta]}
        print(f"{name:<22}{r2:+9.4f}{a:12.1f}   {np.round(beta, 4)}")

    best = min(out, key=lambda k: out[k]["aic"])
    d_aic = out["(a) Eq2.4: 1 + n*d"]["aic"] - out[best]["aic"]
    print(f"\nbest by AIC: {best}   (better than Eq. 2.4 form by dAIC = {d_aic:.1f})")

    # depth contribution, isolated: (b) vs (c)
    d_depth = out["(c) 1 + n"]["aic"] - out["(b) 1 + n + d"]["aic"]
    print(f"depth contribution, AIC(c) - AIC(b) = {d_depth:+.1f} "
          f"(positive favours keeping d)")

    # linear-in-n slope with cluster-bootstrap CI, against the McClean rate
    X = models["(c) 1 + n"]
    uniq = np.unique(ds)
    rng = np.random.default_rng(0)
    boots = np.empty(args.boot)
    for b in range(args.boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(ds == k) for k in pick])
        boots[b] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    slope = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
    mcclean = -float(np.log(2))
    print(f"\nslope on n = {slope:+.4f}  95% CI ({lo:+.4f}, {hi:+.4f})")
    print(f"McClean Haar rate = {mcclean:+.4f}  -> "
          f"{'EXCLUDED' if (hi < mcclean or lo > mcclean) else 'not excluded'} by the CI")

    # per-n means, for the paper's table
    per_n = {}
    for nn in sorted(set(n.tolist())):
        for dd in sorted(set(d.tolist())):
            m = (n == nn) & (d == dd)
            if m.sum():
                per_n[f"n={int(nn)},L={int(dd)}"] = {
                    "mean_log_var": float(y[m].mean()), "n_cells": int(m.sum())}

    Path(args.out).write_text(json.dumps({
        "models": out, "best_by_aic": best, "delta_aic_vs_eq24": d_aic,
        "depth_aic_contribution": d_depth,
        "slope_on_n": {"estimate": slope, "ci95": [float(lo), float(hi)],
                       "mcclean_rate": mcclean,
                       "mcclean_excluded": bool(hi < mcclean or lo > mcclean)},
        "per_n_means": per_n,
        "label": "EXPLORATORY — not a pre-registered hypothesis",
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
