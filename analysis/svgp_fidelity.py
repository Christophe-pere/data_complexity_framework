"""Fidelity + scaling check: SVGPMultiHead vs exact GPMultiHead.

(1) Fidelity: on a tractable subsample where the exact GP is feasible, fit both on the
    SAME (X, y_edge, y_align, γ) and compare predictions. SVGP should recover the exact GP.
(2) Scaling: fit SVGP on a full-Phase-B LODO train fold (~7,800 rows) and time it.
"""
import time, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from analysis.phase_b_training_data import load_phase_b_training_set, LoaderConfig
from qmlc.predictor import GPMultiHead, _logit
from qmlc.predictor_svgp import SVGPMultiHead

pts, gt = load_phase_b_training_set(
    "experiments/results/phase_b_best_classical.jsonl", LoaderConfig(edge_metric="budget"))
X = np.array([p.features() for p in pts])
e = np.array([p.e_obs for p in pts])
align = _logit(np.clip(np.array([p.epsilon_emp for p in pts]), 1e-3, 1 - 1e-3))
ds = np.array([p.dataset_id for p in pts])
rng = np.random.default_rng(0)

# ---------- (1) fidelity on a tractable subsample ----------
print("=" * 64)
print("(1) FIDELITY — exact GP vs SVGP on identical inputs (subsample)")
sub = rng.choice(len(X), size=1200, replace=False)
tr, te = sub[:800], sub[800:]
Xtr, Xte = X[tr], X[te]

t0 = time.perf_counter()
gp = GPMultiHead(n_iter=200)
gp._fit_two_targets(Xtr, e[tr], align[tr], gamma=1.0)
e_exact, _, _, _ = gp._predict_two_targets(Xte)
t_exact = time.perf_counter() - t0

t0 = time.perf_counter()
sv = SVGPMultiHead(n_iter=600, num_inducing=256, num_latents=2)
sv._fit_two_targets(Xtr, e[tr], align[tr], gamma=1.0)
e_svgp, _, _, _ = sv._predict_two_targets(Xte)
t_svgp = time.perf_counter() - t0

print(f"  exact GP fit: {t_exact:5.1f}s   SVGP fit: {t_svgp:5.1f}s   (n_train=800)")
print(f"  Pearson(exact, svgp) on held-out ê     = {pearsonr(e_exact, e_svgp)[0]:+.4f}")
print(f"  RMSE(exact − svgp)                      = {np.sqrt(np.mean((e_exact-e_svgp)**2)):.4f}")
print(f"  R²(exact | truth)={r2_score(e[te], e_exact):+.3f}   R²(svgp | truth)={r2_score(e[te], e_svgp):+.3f}")

# ---------- (2) scaling on a full LODO fold ----------
print("=" * 64)
print("(2) SCALING — SVGP on a full Phase-B LODO train fold (~7,800 rows)")
ho = "real-iris"
tr_full = ds != ho
te_full = ds == ho
t0 = time.perf_counter()
sv2 = SVGPMultiHead(n_iter=600, num_inducing=256, num_latents=2)
sv2._fit_two_targets(X[tr_full], e[tr_full], align[tr_full], gamma=1.0)
e_pred, _, _, _ = sv2._predict_two_targets(X[te_full])
print(f"  holdout={ho}  n_train={tr_full.sum()}  SVGP fit+predict: {time.perf_counter()-t0:.1f}s")
print(f"  held-out edge: pred mean={e_pred.mean():+.3f} actual mean={e[te_full].mean():+.3f}")

# full-suite LODO with SVGP (all 27 folds) — the confirmatory-shape run, timed
print("=" * 64)
print("(3) FULL LODO with SVGP (27 folds, no inner γ-CV, γ=1) — timing + R²")
t0 = time.perf_counter(); preds = np.full(len(e), np.nan)
for d in np.unique(ds):
    tr_i = ds != d
    m = SVGPMultiHead(n_iter=400, num_inducing=256, num_latents=2)
    m._fit_two_targets(X[tr_i], e[tr_i], align[tr_i], gamma=1.0)
    preds[ds == d], _, _, _ = m._predict_two_targets(X[ds == d])
print(f"  27-fold SVGP LODO: {time.perf_counter()-t0:.0f}s total   pooled R²={r2_score(e, preds):+.3f}")
