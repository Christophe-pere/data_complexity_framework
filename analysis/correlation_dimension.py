#!/usr/bin/env python3
"""Correlation fractal dimension D2 of every Phase B dataset.

Motivation. Appel (arXiv:2609.00475) shows angle-encoded quantum kernels collapse once the
feature map is WIDER than the data's intrinsic dimension, with the correlation dimension D2
giving the threshold a priori. Every Phase B cell encodes on n=8 qubits, and several suite
datasets are low-dimensional manifolds in slightly higher ambient space (synth-torus is a
2-torus in R^3). If the mechanism transfers, a_q pinned at chance would be a PREDICTED
consequence of encoding width rather than a statement about data complexity.

This script only measures D2. It does not test the transfer -- Appel measures fidelity
kernels and Phase B used variational circuits, and the shared mechanism (exponential
concentration) is an assumption until checked.

Method: Grassberger-Procaccia. C(r) = fraction of point pairs closer than r; D2 is the slope
of log C(r) vs log r in the scaling region. The scaling region is chosen automatically as the
widest window of log-spaced radii whose local slopes are most nearly constant, which avoids
the two standard failure modes -- saturation at large r (C -> 1) and depletion at small r
(too few pairs).
"""
from __future__ import annotations
import sys, json, warnings
import numpy as np
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from qmlc.datasets import all_phase_b_specs, class_b_specs

def correlation_dimension(X, n_r=28, max_pts=1200, seed=0):
    """(D2, r2_of_fit, n_points_used). Returns (nan, nan, m) if no clean scaling region."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    if X.shape[0] > max_pts:                       # O(m^2) pair distances
        X = X[rng.choice(X.shape[0], max_pts, replace=False)]
    m = X.shape[0]
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)       # standardise: D2 is scale-free, this
                                                   # only stabilises the radius grid
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(m, 1)
    dist = np.sort(d[iu])
    dist = dist[dist > 0]
    if dist.size < 100:
        return float("nan"), float("nan"), m
    # Grassberger-Procaccia assumes a CONTINUOUS distance distribution. On binary/discrete
    # data (parity families, spin glasses on {0,1}^d) the pairwise distances take only a few
    # values, there is no scaling region, and the fit returns a slope near 0 that is an
    # artefact rather than a dimension. Detect and refuse rather than report a number.
    if np.unique(np.round(dist, 6)).size < 20:
        return float("nan"), float("nan"), m
    lo, hi = np.percentile(dist, 1), np.percentile(dist, 90)
    rs = np.logspace(np.log10(lo), np.log10(hi), n_r)
    C = np.array([(dist < r).mean() for r in rs])
    ok = C > 0
    logr, logC = np.log(rs[ok]), np.log(C[ok])
    if logr.size < 8:
        return float("nan"), float("nan"), m
    # widest window with the most nearly constant local slope
    best = (None, np.inf)
    for w in range(logr.size, 5, -1):
        for i in range(0, logr.size - w + 1):
            sl = np.diff(logC[i:i+w]) / np.diff(logr[i:i+w])
            score = sl.std() / (abs(sl.mean()) + 1e-12) - 0.02 * w   # prefer wide + flat
            if score < best[1]:
                best = ((i, i + w), score)
    (i, j), _ = best
    A = np.vstack([logr[i:j], np.ones(j - i)]).T
    coef, res, *_ = np.linalg.lstsq(A, logC[i:j], rcond=None)
    pred = A @ coef
    ss_res = ((logC[i:j] - pred) ** 2).sum()
    ss_tot = ((logC[i:j] - logC[i:j].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(r2), m

if __name__ == "__main__":
    out = {}
    print("%-26s %6s %8s %8s %6s %8s" % ("dataset", "E", "D2", "ceil(D2)", "R2", "n_pts"))
    print("-" * 70)
    class_b = {s.name for s in class_b_specs()}
    for spec in all_phase_b_specs():
        if spec.name in class_b:
            print("  %-26s  class B state ensemble - skipped without loading" % spec.name, flush=True)
            continue
        print("  loading %-24s" % spec.name, end="", flush=True)
        try:
            ds = spec.load()
        except Exception as e:
            print("%-26s  load failed: %s" % (spec.name, str(e)[:30])); continue
        if ds.X is None:
            print("%-26s  %s" % (spec.name, "state ensemble (class B) - skipped")); continue
        D2, r2, m = correlation_dimension(ds.X)
        E = ds.X.shape[1]
        out[spec.name] = {"E": E, "D2": D2, "ceil_D2": (int(np.ceil(D2)) if np.isfinite(D2) else None),
                                "fit_r2": r2, "n_points": m}
        print("%-26s %6d %8.2f %8s %6.3f %8d"
              % (spec.name, E, D2, (int(np.ceil(D2)) if np.isfinite(D2) else "--"), r2, m))
    json.dump(out, open("analysis/correlation_dimension.json", "w"), indent=1)
    print("\nwrote analysis/correlation_dimension.json")
