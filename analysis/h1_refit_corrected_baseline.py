#!/usr/bin/env python3
"""Does data complexity predict gradient variance over a CORRECT leading term?

Part II's H1 fitted a data-complexity correction kappa^T v on top of Eq. (bp), whose leading
term is the product n*d. The n-scaling arm (Sec. results:nscaling) then showed that leading
term is mis-specified: log n beats n*d by dAIC 1783 and depth contributes nothing. The
correction was therefore fitted on a wrong baseline, and the paper says so -- but never refits.

This is that refit. The question is narrow and answerable from data already on disk: over the
corrected leading term, does kappa^T v buy anything?

Restricted to the n-scaling arm (datasets measured at more than one width), because log n
carries no variance on the n=8-only datasets.
"""
from __future__ import annotations
import json, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = [json.loads(l) for l in open(ROOT / "experiments/results/h1_gradients.jsonl") if l.strip()]

widths = collections.defaultdict(set)
for r in R:
    widths[r["dataset_id"]].add(r["n_qubits"])
multi = {d for d, w in widths.items() if len(w) > 1}
R = [r for r in R if r["dataset_id"] in multi]
COMP = sorted(R[0]["v"].keys())
print("n-scaling arm: %d records, %d datasets, widths %s"
      % (len(R), len(multi), sorted({r["n_qubits"] for r in R})))
print("complexity components (%d): %s\n" % (len(COMP), ", ".join(c[:22] for c in COMP)))

y  = np.array([r["log_grad_var"] for r in R], float)
n  = np.array([r["n_qubits"] for r in R], float)
d  = np.array([r["depth"] for r in R], float)
V  = np.array([[r["v"][c]["value"] for c in COMP] for r in R], float)
V  = (V - V.mean(0)) / (V.std(0) + 1e-12)          # standardise so kappa are comparable
ds = np.array([r["dataset_id"] for r in R])

def fit(X, y):
    X = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid); k = X.shape[1]; m = len(y)
    aic = m * np.log(rss / m) + 2 * k
    r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
    return aic, r2, k

def lodo(Xf):
    """Leave-one-dataset-out R^2 -- Part II's standard for a predictive claim."""
    preds = np.empty_like(y)
    for held in multi:
        tr, te = ds != held, ds == held
        X = Xf(); Xtr = np.column_stack([np.ones(tr.sum()), X[tr]]) if X.size else np.ones((tr.sum(),1))
        Xte = np.column_stack([np.ones(te.sum()), X[te]]) if X.size else np.ones((te.sum(),1))
        b, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        preds[te] = Xte @ b
    return 1 - float(((y - preds) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())

MODELS = [
    ("1 + n*d                (Eq. bp leading term)", lambda: (n * d)[:, None]),
    ("1 + n*d + kappa^T v    (H1 as fitted)",        lambda: np.column_stack([n * d, V])),
    ("1 + log n              (corrected baseline)",  lambda: np.log(n)[:, None]),
    ("1 + log n + d",                                lambda: np.column_stack([np.log(n), d])),
    ("1 + log n + kappa^T v  (THE REFIT)",           lambda: np.column_stack([np.log(n), V])),
]
print("%-46s %8s %8s %8s %10s" % ("model", "k", "R2", "AIC", "LODO R2"))
print("-" * 86)
out = {}
for name, f in MODELS:
    a, r2, k = fit(f(), y)
    l = lodo(f)
    out[name.split("(")[0].strip()] = {"aic": a, "r2": r2, "k": k, "lodo_r2": l}
    print("%-46s %8d %8.4f %8.1f %10.4f" % (name, k, r2, a, l))

base = out["1 + log n"]["aic"]; refit = out["1 + log n + kappa^T v"]["aic"]
print("\n  dAIC(corrected baseline -> + data complexity) = %+.1f" % (base - refit))
print("  positive means the complexity vector HELPS; Part II's threshold for a real")
print("  contribution elsewhere is dAIC >= 10 (Burnham & Anderson).")
print("  LODO R2: %.4f -> %.4f" % (out["1 + log n"]["lodo_r2"], out["1 + log n + kappa^T v"]["lodo_r2"]))
json.dump(out, open(ROOT / "analysis/h1_refit.json", "w"), indent=1)
print("\nwrote analysis/h1_refit.json")
