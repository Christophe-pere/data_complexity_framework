#!/usr/bin/env python3
"""Evaluate Part I's composite C_data (Eq. 7) on the 27 classical-track datasets.

Part II SS2.1 replaces Eq. 7 with a learned predictor on v(D), arguing the
composite is (a) dimensionally incoherent and (b) unidentifiable. Both are
asserted there. This script measures them, so the paper can show the failure
instead of only claiming it.

Pillar -> component mapping (Part I SS4):
  S(D)      entropy          differential_entropy_knn_kl
  I_corr(D) correlation      second_order_mi + higher_order_mi
  K(D)      compressibility  compression_ratio_gzip  (mdl_autoencoder reported separately)
  C_top(D)  topology         topological_complexity_rips  (and its corrected form)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC  = ROOT/"experiments/results/phase_b_best_classical.jsonl"
CORR = ROOT/"analysis/topology_corrected_values.json"

vals = {}
for line in open(SRC):
    d = json.loads(line)
    if d["dataset_id"] in vals: continue
    vals[d["dataset_id"]] = {k: v["value"] for k, v in d["v"].items()}
corr = json.load(open(CORR))["values"]

ds = sorted(vals)
def col(k): return np.array([vals[d][k] for d in ds], float)

S     = col("differential_entropy_knn_kl")
Icorr = col("second_order_mi") + col("higher_order_mi")
K     = col("compression_ratio_gzip")
Ctop  = col("topological_complexity_rips")
Ctop_c= np.array([corr[d] for d in ds], float)

print(f"{'pillar':16}{'min':>14}{'max':>14}{'range (log10)':>16}{'units':>28}")
print("-"*90)
for nm,v,u in [("S (entropy)",S,"nats"),("I_corr",Icorr,"nats per order"),
               ("K (compression)",K,"dimensionless [0,1]"),
               ("C_top (as used)",Ctop,"LENGTH on data manifold"),
               ("C_top (corrected)",Ctop_c,"dimensionless")]:
    nz = np.abs(v[v!=0])
    span = np.log10(nz.max()/nz.min()) if len(nz) and nz.min()>0 else float('nan')
    print(f"{nm:16}{v.min():14.4g}{v.max():14.4g}{span:16.1f}{u:>28}")

def decompose(top, label):
    P = np.column_stack([S, Icorr, K, top])
    names = ["S", "I_corr", "K", "C_top"]
    C = P.sum(1)                                   # Part I Eq. 7, equal weights
    tot = P.var(0).sum()
    print(f"\n=== {label} ===")
    print(f"C_data (equal weights, unnormalised): min={C.min():.4g}  max={C.max():.4g}")
    print(f"{'pillar':10}{'variance':>16}{'share of total':>18}")
    for n, vv in zip(names, P.var(0)):
        print(f"{n:10}{vv:16.4g}{100*vv/tot:17.2f}%")
    # rank correlation of the composite with each pillar
    from scipy.stats import spearmanr
    print(f"{'pillar':10}{'spearman(C_data, pillar)':>28}")
    for n, j in zip(names, range(4)):
        print(f"{n:10}{spearmanr(C, P[:,j]).statistic:28.3f}")
    return C

C_old = decompose(Ctop,   "Part I Eq. 7 with C_top AS DEPLOYED (defective)")
C_new = decompose(Ctop_c, "Part I Eq. 7 with C_top CORRECTED")

print("\n=== does the correction change the composite's ranking? ===")
from scipy.stats import spearmanr
print(f"  spearman(C_data_old, C_data_new) = {spearmanr(C_old, C_new).statistic:.4f}")
out = {d: {"S":float(S[i]),"I_corr":float(Icorr[i]),"K":float(K[i]),
           "C_top":float(Ctop[i]),"C_top_corrected":float(Ctop_c[i]),
           "C_data_equalweights":float(C_old[i]),
           "C_data_equalweights_corrected":float(C_new[i])} for i,d in enumerate(ds)}
json.dump(out, open(ROOT/"analysis/part1_composite_values.json","w"), indent=1)
print(f"\nper-dataset values -> analysis/part1_composite_values.json")
