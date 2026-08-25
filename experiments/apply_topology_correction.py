#!/usr/bin/env python3
"""Rebuild the results files with the corrected topological component.

The frozen `topological_complexity_rips` estimator was found defective on
2026-08-24 (see `docs/deviations_log.md`): H0 included at full weight, no scale
normalisation, an over-truncated filtration, and `collapse_edges` deleting the
essential classes the truncation had manufactured. It returns P1 = 0.000 for a
perfect circle.

This script recomputes that component with `NormalisedPersistentHomology` and
writes NEW results files, leaving the originals untouched so the published
numbers stay reproducible.

**Choice of summary.** The default is `l2` (sum of squared diameter-normalised
persistences). It was selected on *validation* grounds, before any hypothesis
was re-run with it: it is threshold-free (no arbitrary significance cut to
tune), continuous, and it passes the known-shape battery. The thresholded
`count` summary also passes validation but requires a cut theta, and the H2
verdict is sensitive to that cut -- which is precisely why we do not use a
tuned quantity as the primary. See the deviations log for all variants.

Usage:
  python -m experiments.apply_topology_correction \
      --in  experiments/results/phase_b_best_classical.jsonl \
      --out experiments/results/phase_b_best_classical_topofix.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qmlc.classical import NormalisedPersistentHomology  # noqa: E402

COMPONENT = "topological_complexity_rips"


def corrected_values(measure: str, theta: float, subsample: int, seed: int) -> dict[str, float]:
    """Recompute the topological component for every classical-track dataset."""
    reg = json.load(open(ROOT / "datasets" / "registry.json"))
    metric = NormalisedPersistentHomology(
        max_dim=2, measure=measure, theta=theta, subsample=subsample, seed=seed
    )
    out: dict[str, float] = {}
    for e in reg["datasets"]:
        if e["klass"] == "B":          # quantum-native parallel track
            continue
        mod = importlib.import_module(e["loader"]["module"])
        X = np.asarray(getattr(mod, e["loader"]["function"])(**e["loader"]["args"]).X, float)
        t0 = time.time()
        res = metric.compute(X)
        out[e["name"]] = float(res.value)
        print(f"  {e['name']:28} {res.value:10.5f}   ({time.time()-t0:5.1f}s)", flush=True)
    return out


def rewrite(src: Path, dst: Path, vals: dict[str, float], meta: dict) -> tuple[int, int]:
    n = miss = 0
    with open(src) as fh, open(dst, "w") as out:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ds = r["dataset_id"]
            if ds not in vals:
                miss += 1
                continue
            entry = r["v"].get(COMPONENT)
            if entry is None:
                r["v"][COMPONENT] = {"value": vals[ds]}
            else:
                entry["value_original"] = entry.get("value")
                entry["value"] = vals[ds]
                entry["estimator"] = "ripser_diameter_normalised"
                entry["correction"] = meta
            n += 1
            out.write(json.dumps(r) + "\n")
    return n, miss


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", nargs="+", required=True)
    p.add_argument("--out", dest="dst", nargs="+", required=True)
    p.add_argument("--measure", default="l2", choices=("l2", "total", "max", "count", "sigsum"))
    p.add_argument("--theta", type=float, default=0.10)
    p.add_argument("--subsample", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--values-out", default="analysis/topology_corrected_values.json")
    a = p.parse_args()
    if len(a.src) != len(a.dst):
        raise SystemExit("--in and --out must have the same number of paths")

    meta = {"measure": a.measure, "theta": a.theta, "subsample": a.subsample,
            "seed": a.seed, "date": "2026-08-24",
            "reason": "frozen estimator defective; see docs/deviations_log.md 2026-08-24"}

    print(f"[topofix] recomputing {COMPONENT} (measure={a.measure})")
    vals = corrected_values(a.measure, a.theta, a.subsample, a.seed)
    vp = ROOT / a.values_out
    vp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"values": vals, "meta": meta}, open(vp, "w"), indent=1)
    print(f"[topofix] {len(vals)} datasets -> {vp}")

    for s, d in zip(a.src, a.dst):
        n, miss = rewrite(ROOT / s, ROOT / d, vals, meta)
        print(f"[topofix] {s} -> {d}: {n} records rewritten, {miss} skipped (no corrected value)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
