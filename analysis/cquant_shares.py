#!/usr/bin/env python3
"""Variance decomposition of Part I Eq. 10 (C_quant) on the quantum-native track.

Reads analysis/cquant_part2.json (produced by cquant_part2.py) and reports, for the
six components: raw range, share of the composite variance under equal weights, the
same share after per-component rank normalisation (Eq. ranknorm of the paper, the
repair applied to Eq. 7), how many of the ensemble pairs each component separates,
and its range across the parity probes (the one family with a complexity knob w).
EXPLORATORY: Eq. 10 is not among the pre-specified hypotheses H1-H6.
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["S_ent", "I_multi", "rank_eff", "N_magic", "F_qfi", "C_top_q"]
rows = json.load(open(ROOT / "analysis/cquant_part2.json"))
names = sorted(rows)
M = np.array([[rows[d][k] for k in ORDER] for d in names], float)


def shares(X: np.ndarray) -> np.ndarray:
    v = X.var(0)
    return 100 * v / v.sum()


def ranknorm(X: np.ndarray) -> np.ndarray:
    # v_i(D) = |{D' : f_i(D') <= f_i(D)}| / |B|  (ties share the upper rank)
    return np.array([[np.mean(X[:, j] <= X[i, j]) for j in range(X.shape[1])]
                     for i in range(X.shape[0])])


def separated_pairs(x: np.ndarray, rtol: float = 1e-6) -> int:
    scale = max(np.abs(x).max(), 1e-12)
    return sum(abs(a - b) > rtol * scale for a, b in itertools.combinations(x, 2))


raw, rk = shares(M), shares(ranknorm(M))
parity = [i for i, d in enumerate(names) if d.startswith("qn-parity")]
npairs = len(names) * (len(names) - 1) // 2
out = {"n_ensembles": len(names), "n_pairs": npairs, "components": {}}
print(f"{len(names)} ensembles ({npairs} pairs); parity probes: {[names[i] for i in parity]}")
print(f"{'component':10}{'min':>10}{'max':>10}{'raw %Var':>11}{'rank %Var':>11}{'pairs sep':>11}{'parity range':>14}")
for j, k in enumerate(ORDER):
    pr = float(M[parity, j].max() - M[parity, j].min())
    sp = separated_pairs(M[:, j])
    out["components"][k] = dict(min=float(M[:, j].min()), max=float(M[:, j].max()),
                                raw_share=float(raw[j]), rank_share=float(rk[j]),
                                pairs_separated=int(sp), parity_range=pr)
    print(f"{k:10}{M[:, j].min():10.4f}{M[:, j].max():10.4f}{raw[j]:11.2f}{rk[j]:11.2f}{sp:11d}{pr:14.4g}")
json.dump(out, open(ROOT / "analysis/cquant_shares.json", "w"), indent=1)
