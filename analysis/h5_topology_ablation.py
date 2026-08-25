"""Is H5's failure an artifact of the broken topology slot?

Direct test: vary ONLY component 5 and see whether the verdict moves.
  full-production  : 9 components, the defective estimator (as published)
  corrected-l2     : 9 components, corrected threshold-free summary
  corrected-count  : 9 components, corrected thresholded count
  dropped          : 8 components, topology removed entirely
  random-noise     : 9 components, topology replaced by U(0,1) noise
  shuffled         : 9 components, topology values permuted across datasets

If every row lands in the same place, the H5 falsification does not rest on
that slot and no topology estimator could have rescued it.
"""
import sys, json, argparse, warnings, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from analysis.phase_b_training_data import load_phase_b_training_set, LoaderConfig
from experiments.feature_extraction import V_COMPONENTS
from sklearn.ensemble import HistGradientBoostingRegressor as GBM

pts,_ = load_phase_b_training_set("experiments/results/phase_b_best_classical.jsonl",
                                  LoaderConfig(edge_metric="budget", normalize_v=False))
K = len(V_COMPONENTS); j = list(V_COMPONENTS).index("topological_complexity_rips")
X0 = np.array([np.concatenate([p.v, p.q]) for p in pts], float)
y  = np.array([p.e_obs for p in pts], float)
ds = np.array([p.dataset_id for p in pts])
uds = np.unique(ds)

def ranknorm(tr, ap, k):
    o = ap.copy()
    for c in range(k):
        col = np.sort(tr[:, c]); o[:, c] = np.searchsorted(col, ap[:, c], side="right")/max(len(col),1)
    return o

def lodo(X, k):
    pred = np.empty_like(y)
    for h in uds:
        te = ds == h; tr = ~te
        m = GBM(max_iter=300, learning_rate=0.05).fit(ranknorm(X[tr], X[tr], k), y[tr])
        pred[te] = m.predict(ranknorm(X[tr], X[te], k))
    return 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()

rng = np.random.default_rng(0)
cases = {}
cases["full-production"] = (X0, K)
for nm, f in [("corrected-l2", "analysis/topology_corrected_values.json"), ("corrected-count", "analysis/topology_corrected_count.json")]:
    ov = json.load(open(f)); ov = ov.get("values", ov); Xc = X0.copy(); Xc[:, j] = [ov[d] for d in ds]; cases[nm] = (Xc, K)
cases["dropped (8 comps)"] = (np.delete(X0, j, axis=1), K-1)
Xn = X0.copy(); Xn[:, j] = rng.random(len(X0)); cases["random-noise"] = (Xn, K)
perm = {d: v for d, v in zip(uds, rng.permutation([X0[ds==d][0, j] for d in uds]))}
Xs = X0.copy(); Xs[:, j] = [perm[d] for d in ds]; cases["shuffled"] = (Xs, K)

print(f"{'topology slot':22}{'pooled LODO R^2':>18}   verdict (threshold +0.40)")
print("-"*70)
for nm, (X, k) in cases.items():
    r2 = lodo(X, k)
    print(f"{nm:22}{r2:+18.4f}   {'FAIL' if r2 < 0.40 else 'pass'}", flush=True)
