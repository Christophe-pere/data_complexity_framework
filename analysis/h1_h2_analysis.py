#!/usr/bin/env python3
"""H1 (barren-plateau scaling) and H2 (topology contribution) — pre-registered tests.

H1 decision rule: R^2 >= 0.7 in the cross-validated linear regression of
log Var[g] on (n, d, n*d*v), AND at least one kappa_i with 95% CI excluding 0.

H2 decision rule: the model including the topological (and/or curvature)
components of v achieves AIC lower than the model excluding them by >= 10.

Model, Eq. 2.4:   log Var[g] ~ log A - alpha*n*d - n*d * kappa^T v(D)

Caveat, stated in the paper: every Phase B cell uses n = 8, so `n` is collinear
with the intercept and n*d collapses to a rescaling of d (two levels, L in {1,3}).
alpha is therefore weakly identified; the kappa_i remain identifiable because
n*d*v varies across datasets. This script reports the fit as specified and flags
the limitation rather than silently dropping the term.

v is rank-normalized per framework_v2 §2.1, fit on the training folds only
(leakage-clean: a held-out dataset never informs its own scaling).

Usage:  python -m analysis.h1_h2_analysis [--jsonl ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

V_ORDER = [
    "differential_entropy_knn_kl", "intrinsic_dim_pca", "intrinsic_dim_twonn",
    "compression_ratio_gzip", "topological_complexity_rips", "second_order_mi",
    "higher_order_mi", "kernel_effective_rank", "mdl_autoencoder",
]
TOPO = ["topological_complexity_rips"]  # no explicit curvature component in the frozen v


def load(jsonl: str, datasets=None, n_qubits=None, balanced=False, topo_override=None):
    """Load records, optionally restricted to a dataset subset and/or n values.

    balanced=True keeps only (dataset, seed) groups present at *every* requested n,
    so the n-scaling fit sees a balanced design rather than a suite that is deep at
    one width and shallow elsewhere."""
    ds, y, nd, V, nq, seeds = [], [], [], [], [], []
    with open(jsonl) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("log_grad_var") is None or not np.isfinite(r["log_grad_var"]):
                continue
            if datasets and r["dataset_id"] not in datasets:
                continue
            if n_qubits and r["n_qubits"] not in n_qubits:
                continue
            ds.append(r["dataset_id"]); y.append(r["log_grad_var"])
            nd.append(r["nd"]); nq.append(r["n_qubits"]); seeds.append(r["seed"])
            row = [r["v"][k]["value"] for k in V_ORDER]
            if topo_override is not None:
                j = V_ORDER.index("topological_complexity_rips")
                if r["dataset_id"] not in topo_override:
                    raise KeyError(f"no topology override for {r['dataset_id']}")
                row[j] = topo_override[r["dataset_id"]]
            V.append(row)
    ds = np.array(ds); y = np.array(y); nd = np.array(nd)
    V = np.array(V, dtype=float); nq = np.array(nq); seeds = np.array(seeds)

    if balanced and n_qubits:
        want = set(n_qubits)
        keep_groups = {
            (d, s) for d in np.unique(ds) for s in np.unique(seeds)
            if want.issubset(set(nq[(ds == d) & (seeds == s)].tolist()))
        }
        mask = np.array([(d, s) in keep_groups for d, s in zip(ds, seeds)])
        ds, y, nd, V, nq = ds[mask], y[mask], nd[mask], V[mask], nq[mask]
    return ds, y, nd, V, nq


def rank_normalize(train: np.ndarray, apply_to: np.ndarray) -> np.ndarray:
    """Map each column onto [0,1] by the training ECDF."""
    out = np.empty_like(apply_to)
    for j in range(train.shape[1]):
        col = np.sort(train[:, j])
        out[:, j] = np.searchsorted(col, apply_to[:, j], side="right") / max(len(col), 1)
    return out


def design(nd: np.ndarray, Vn: np.ndarray, cols: list[int]) -> np.ndarray:
    """[1, n*d, n*d*v_cols] — intercept = log A, then -alpha, then -kappa."""
    blocks = [np.ones((len(nd), 1)), nd[:, None]]
    if cols:
        blocks.append(nd[:, None] * Vn[:, cols])
    return np.hstack(blocks)


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def aic(y: np.ndarray, yhat: np.ndarray, k: int) -> float:
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    return n * np.log(max(rss, 1e-300) / n) + 2 * k


def lodo_r2(ds, y, nd, V, cols) -> tuple[float, np.ndarray]:
    """Pooled leave-one-dataset-out R^2, and the OOF predictions."""
    pred = np.empty_like(y)
    for hold in np.unique(ds):
        te = ds == hold
        tr = ~te
        Vn_tr = rank_normalize(V[tr], V[tr])
        Vn_te = rank_normalize(V[tr], V[te])
        beta = ols(design(nd[tr], Vn_tr, cols), y[tr])
        pred[te] = design(nd[te], Vn_te, cols) @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot, pred


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="experiments/results/h1_gradients.jsonl")
    p.add_argument("--out", default="analysis/h1_h2_report.json")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--topo-override", default=None,
                   help="JSON {dataset_id: value} replacing topological_complexity_rips "
                        "(corrected estimator; leaves the results JSONL untouched)")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--n-qubits", nargs="*", type=int, default=None)
    p.add_argument("--balanced", action="store_true",
                   help="keep only (dataset,seed) groups present at every requested n")
    args = p.parse_args()

    _ov = None
    if args.topo_override:
        _ov = json.load(open(args.topo_override))
        print(f"[topology] override loaded for {len(_ov)} datasets from {args.topo_override}")
    ds, y, nd, V, nq = load(args.jsonl, args.datasets, args.n_qubits, args.balanced,
                            topo_override=_ov)
    if len(y) == 0:
        print("no records after filtering", file=sys.stderr); return 1
    print(f"[h1] {len(y)} records, {len(np.unique(ds))} datasets, "
          f"n levels = {sorted(set(nq.tolist()))}, "
          f"n*d levels = {sorted(set(nd.tolist()))}")

    all_cols = list(range(len(V_ORDER)))
    r2, _ = lodo_r2(ds, y, nd, V, all_cols)
    print(f"[H1] pooled LODO R^2 = {r2:+.4f}   (threshold 0.7)")

    # in-sample fit for kappa + bootstrap CIs (cluster bootstrap over datasets)
    Vn = rank_normalize(V, V)
    X = design(nd, Vn, all_cols)
    beta = ols(X, y)
    names = ["logA", "nd(-alpha)"] + [f"kappa:{k}" for k in V_ORDER]

    uniq = np.unique(ds)
    rng = np.random.default_rng(0)
    boots = np.empty((args.boot, X.shape[1]))
    for b in range(args.boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(ds == d) for d in pick])
        boots[b] = ols(X[idx], y[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    print("\n  coefficient            estimate      95% CI (cluster bootstrap)")
    kappa_excl = []
    for i, nm in enumerate(names):
        star = "  *" if (lo[i] > 0 or hi[i] < 0) else ""
        print(f"  {nm:<22}{beta[i]:+9.4f}   ({lo[i]:+.4f}, {hi[i]:+.4f}){star}")
        if nm.startswith("kappa:") and (lo[i] > 0 or hi[i] < 0):
            kappa_excl.append(nm)

    h1_pass = bool(r2 >= 0.7 and kappa_excl)
    print(f"\n[H1] R^2 >= 0.7: {r2 >= 0.7} | kappa CI excluding 0: "
          f"{len(kappa_excl)} -> H1_pass = {h1_pass}")

    # ---- H2: AIC with vs without the topological component -------------------
    topo_idx = [V_ORDER.index(t) for t in TOPO]
    red_cols = [c for c in all_cols if c not in topo_idx]
    Xf, Xr = design(nd, Vn, all_cols), design(nd, Vn, red_cols)
    aic_f = aic(y, Xf @ ols(Xf, y), Xf.shape[1])
    aic_r = aic(y, Xr @ ols(Xr, y), Xr.shape[1])
    d_aic = aic_r - aic_f
    h2_pass = bool(d_aic >= 10.0)
    print(f"\n[H2] AIC full = {aic_f:.1f} | AIC without topology = {aic_r:.1f}"
          f" | dAIC = {d_aic:+.1f} (threshold >= 10) -> H2_pass = {h2_pass}")
    r2_red, _ = lodo_r2(ds, y, nd, V, red_cols)
    print(f"[H2] LODO R^2 without topology = {r2_red:+.4f}")

    rep = {
        "n_records": int(len(y)), "n_datasets": int(len(np.unique(ds))),
        "n_levels": sorted(set(nq.tolist())),
        "nd_levels": sorted(set(nd.tolist())),
        "H1": {"pooled_lodo_r2": r2, "threshold": 0.7,
               "coefficients": {nm: {"estimate": float(beta[i]),
                                     "ci95": [float(lo[i]), float(hi[i])]}
                                for i, nm in enumerate(names)},
               "kappa_ci_excluding_zero": kappa_excl, "H1_pass": h1_pass},
        "H2": {"aic_full": aic_f, "aic_reduced": aic_r, "delta_aic": d_aic,
               "threshold": 10.0, "lodo_r2_reduced": r2_red, "H2_pass": h2_pass,
               "note": "no explicit curvature component in the frozen v; "
                       "topology = topological_complexity_rips"},
        "caveat_fixed_n": "all cells n=8; alpha weakly identified from two depth levels",
    }
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
