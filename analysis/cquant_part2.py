#!/usr/bin/env python3
"""Part I Eq. 10 ($C_quant$) evaluated on the quantum-native track — all six components.

Every component value is parameter-free: no threshold, no tuned constant, no
external parametrisation. Components 5 and 6 were under-specified in Part I and
are made well-posed here (see docs/class_b_quantum_native_spec.md):
  5  F   -> max over collective-spin directions of F_Q, / n^2   (bounded [1/n, 1])
  6  Ctq -> sum_{k>=1} (l_k / (pi/2))^2 under Fubini-Study      (H0 excluded)

Components 1, 2 and 4 are read from cached results (registry screening for the
frozen Class B suite, phase_b_classB.jsonl for the probes) rather than recomputed:
exact stabilizer magic is 4^n Pauli expectations per state and the values are
deterministic.
"""
from __future__ import annotations
import importlib, json, sys
from pathlib import Path
import numpy as np
from ripser import ripser

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qmlc.quantum import EnsembleKernelEffectiveRank, EnsembleQFI          # noqa: E402
from qmlc.datasets import class_b_probe_specs                              # noqa: E402

FS_MAX = np.pi / 2

def c_top_q(S, maxdim=2, sub=250, seed=0):
    """Component 6: threshold-free l2 of FS-normalised persistence, H0 excluded."""
    S = np.asarray(S)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    if len(S) > sub:
        S = S[np.random.default_rng(seed).permutation(len(S))[:sub]]
    dm = np.arccos(np.clip(np.abs(S.conj() @ S.T), 0.0, 1.0))
    dgms = ripser(dm, distance_matrix=True, maxdim=maxdim, thresh=FS_MAX)['dgms']
    tot = 0.0
    for k in range(1, maxdim + 1):
        d = np.asarray(dgms[k], float)
        if not len(d):
            continue
        dd = d[:, 1].copy(); dd[~np.isfinite(dd)] = FS_MAX
        L = np.clip(dd - d[:, 0], 0, None) / FS_MAX
        tot += float((L ** 2).sum())
    return tot

CACHED = {"S_ent": "entanglement_entropy_mean", "I_multi": "total_correlation_mean",
          "N_magic": "stabilizer_renyi2_mean"}
PROBEK = {"S_ent": "v_ent_mean", "I_multi": "v_mi_mean", "N_magic": "v_mag_mean"}
ORDER = ["S_ent", "I_multi", "rank_eff", "N_magic", "F_qfi", "C_top_q"]

probe_cache = {}
pf = ROOT / "experiments/results/phase_b_classB.jsonl"
if pf.exists():
    for line in open(pf):
        r = json.loads(line); probe_cache.setdefault(r["dataset_id"], r["v_native"])

def states_of(mod, fn, args):
    return getattr(importlib.import_module(mod), fn)(**args).states

rank_m, qfi_m = EnsembleKernelEffectiveRank(), EnsembleQFI()
rows = {}

def do(name, S, cached):
    rec = dict(cached)
    rec["rank_eff"] = float(rank_m.compute(S).value)
    rec["F_qfi"]    = float(qfi_m.compute(S).value)
    rec["C_top_q"]  = float(c_top_q(S))
    rows[name] = rec
    print("  " + f"{name:22}" + "".join(f"{rec[k]:10.4f}" for k in ORDER), flush=True)

print(f"  {'ensemble':22}" + "".join(f"{k:>10}" for k in ORDER))
reg = json.load(open(ROOT / "datasets/registry.json"))
for e in reg["datasets"]:
    if e["klass"] != "B":
        continue
    comp = e.get("screening", {}).get("components", {})
    cached = {lbl: float(comp.get(key, {}).get("value", np.nan)) for lbl, key in CACHED.items()}
    do(e["name"], states_of(e["loader"]["module"], e["loader"]["function"], e["loader"]["args"]), cached)
for spec in class_b_probe_specs():
    pv = probe_cache.get(spec.name, {})
    cached = {lbl: float(pv[key]) if key in pv else float("nan") for lbl, key in PROBEK.items()}
    do(spec.name, states_of(spec.module, spec.function, spec.args), cached)

json.dump(rows, open(ROOT / "analysis/cquant_part2.json", "w"), indent=1)
names = sorted(rows)
M = np.array([[rows[d][k] for k in ORDER] for d in names])
ok = np.all(np.isfinite(M), axis=0)
print(f"\n{'component':12}{'min':>11}{'max':>11}{'variance':>13}{'share of Var':>15}")
tot = M[:, ok].var(0).sum()
for i, k in enumerate(ORDER):
    if not ok[i]:
        print(f"{k:12}{'--':>11}{'--':>11}{'(missing)':>13}"); continue
    print(f"{k:12}{M[:,i].min():11.4f}{M[:,i].max():11.4f}{M[:,i].var():13.5g}{100*M[:,i].var()/tot:14.2f}%")
print(f"\n{len(names)} ensembles, all at n=8 qubits and N=256 states.")
