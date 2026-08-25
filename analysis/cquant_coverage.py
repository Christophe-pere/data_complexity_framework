#!/usr/bin/env python3
"""Coverage of Part I Eq. 10 ($C_quant$) on the quantum-native track.

Reports which of Eq. 10's six components are computable, evaluates those that
are on the frozen Class B suite plus the exploratory parity probes, and records
the natural scale each carries. Part II does NOT assemble the composite; this
is a coverage statement and a Part III specification input.
"""
from __future__ import annotations
import importlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qmlc.quantum import (EnsembleEntanglement, EnsembleTotalCorrelation,      # noqa: E402
                          EnsembleStabilizerMagic, EnsembleKernelEffectiveRank)
from qmlc.datasets import class_b_probe_specs                                   # noqa: E402

ORDER = ["S_ent", "I_multi", "rank_eff", "N_magic"]

# Only rank_eff is recomputed here: the other three are already stored in the
# registry screening (frozen Class B) and in phase_b_classB.jsonl (probes), and
# recomputing exact stabilizer magic (4^q Pauli expectations per state) would
# duplicate hours of work for no new information.
METRICS = [("rank_eff", EnsembleKernelEffectiveRank())]

REGISTRY_KEYS = {"S_ent": "entanglement_entropy_mean",
                 "I_multi": "total_correlation_mean",
                 "N_magic": "stabilizer_renyi2_mean"}
PROBE_KEYS = {"S_ent": "v_ent_mean", "I_multi": "v_mi_mean", "N_magic": "v_mag_mean"}

def states_of(mod, fn, args):
    obj = getattr(importlib.import_module(mod), fn)(**args)
    return obj.states

PROBE = {}
_pf = ROOT/"experiments/results/phase_b_classB.jsonl"
if _pf.exists():
    for line in open(_pf):
        r = json.loads(line); PROBE.setdefault(r["dataset_id"], r["v_native"])

rows = {}
reg = json.load(open(ROOT/"datasets/registry.json"))
for e in reg["datasets"]:
    if e["klass"] != "B": continue
    S = states_of(e["loader"]["module"], e["loader"]["function"], e["loader"]["args"])
    rows[e["name"]] = {k: float(m.compute(S).value) for k, m in METRICS}
    comp = e.get("screening", {}).get("components", {})
    for lbl, key in REGISTRY_KEYS.items():
        v = comp.get(key, {}).get("value")
        rows[e["name"]][lbl] = float(v) if v is not None else float("nan")
    print(f"  {e['name']:22} " + "  ".join(f"{k}={rows[e['name']][k]:9.3f}" for k in ORDER), flush=True)
for spec in class_b_probe_specs():
    S = states_of(spec.module, spec.function, spec.args)
    rows[spec.name] = {k: float(m.compute(S).value) for k, m in METRICS}
    pv = PROBE.get(spec.name, {})
    for lbl, key in PROBE_KEYS.items():
        rows[spec.name][lbl] = float(pv[key]) if key in pv else float("nan")
    print(f"  {spec.name:22} " + "  ".join(f"{k}={rows[spec.name][k]:9.3f}" for k in ORDER), flush=True)

json.dump(rows, open(ROOT/"analysis/cquant_coverage.json","w"), indent=1)
names = sorted(rows)
M = np.array([[rows[d][k] for k in ORDER] for d in names])
print(f"\n{'component':12}{'min':>10}{'max':>10}{'variance':>13}{'share of Var':>15}")
tot = M.var(0).sum()
for i,k in enumerate(ORDER):
    print(f"{k:12}{M[:,i].min():10.3f}{M[:,i].max():10.3f}{M[:,i].var():13.4g}{100*M[:,i].var()/tot:14.2f}%")
print(f"\n{len(names)} ensembles, all at n=8 qubits and N=256 states.")
