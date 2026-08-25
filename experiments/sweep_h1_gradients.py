#!/usr/bin/env python3
"""Gradient-variance sweep for H1 (barren-plateau scaling) and H2 (topology/curvature AIC).

Why this exists
---------------
`qmlc/ansatze.py::gradient_variance()` was built at Phase A but never called: the Phase B
sweep records accuracies and edges, not gradient statistics, so H1 and H2 have no data.
This collects it.

Eq. 2.4 (framework_v2 §2.3) is

    log Var[g(theta)] ~ log A - alpha * n * d - n * d * kappa^T v(D)

so each record carries `grad_var` together with the regressors `n` (qubits), `d` (ansatz
depth) and the 9-component `v(D)`. H1 regresses log Var[g] on (n, d, n*d*v); H2 compares
AIC with and without the topological / curvature components of v.

Design notes
------------
* **v is reused, not recomputed.** Phase B already stored v(D) per (dataset, seed) in
  `phase_b.jsonl`. Recomputing would cost ~26 s/dataset for the MINE higher-order-MI
  estimator alone and risks a v that differs from the one H5 was fit on. We load it, so
  the H1/H2 regressors are bit-identical to the Phase B vectors.
* **Re-uploading encodings are excluded.** `gradient_variance` takes a fixed encoded state
  stack; in a re-uploading circuit the data re-enters at every layer, so there is no fixed
  rho_phi(x) to hand it. That leaves the 8 non-re-uploading encodings. Logged as a
  deviation; it removes 2 of 10 encodings from H1/H2 only, not from H5.
* **n is a free argument.** Phase B ran at n_qubits=8 throughout, so a fixed-n sweep cannot
  identify the alpha*n*d term separately from the depth contrast (see the H1 caveat in the
  Part II draft). `--n-qubits` accepts several values so an n-scaling arm can be added.
* Resumable: dedup key is (dataset, encoding, ansatz, seed, n_qubits); existing rows in the
  output JSONL are skipped, so a crash or a later n-arm just appends.

Usage
-----
    python -m experiments.sweep_h1_gradients --workers 6
    python -m experiments.sweep_h1_gradients --n-qubits 4 6 8 --datasets synth-parity-3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.ansatze import all_phase_b_ansatze  # noqa: E402
from qmlc.datasets import all_phase_b_specs  # noqa: E402
from qmlc.encodings import DataReuploading, all_phase_b_configurations  # noqa: E402
from qmlc.qclassifier import encode_states  # noqa: E402

from experiments.feature_prep import fit_feature_prep  # noqa: E402
from experiments.sweep_phase_b import _binarize  # noqa: E402

PHASE_B_JSONL = "experiments/results/phase_b.jsonl"
DEFAULT_OUTPUT = "experiments/results/h1_gradients.jsonl"


def load_v_table(jsonl: str) -> dict[tuple[str, int], dict]:
    """(dataset_id, seed) -> v dict, taken from the Phase B records."""
    table: dict[tuple[str, int], dict] = {}
    with open(jsonl) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            table.setdefault((r["dataset_id"], int(r["seed"])), r["v"])
    return table


def load_done(path: str) -> set[tuple]:
    done: set[tuple] = set()
    p = Path(path)
    if not p.exists():
        return done
    with p.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["dataset_id"], r["encoding_id"], r["ansatz_id"],
                      int(r["seed"]), int(r["n_qubits"])))
    return done


_W: dict = {}


def _init(n_qubits_list: list[int]) -> None:
    _W["specs"] = {s.name: s for s in all_phase_b_specs()}
    _W["enc"] = {n: all_phase_b_configurations(n) for n in n_qubits_list}
    _W["ans"] = {n: all_phase_b_ansatze(n) for n in n_qubits_list}


def _cell(payload: tuple) -> list[dict]:
    """One (dataset, seed, n_qubits) group -> records for its pending (encoding, ansatz)."""
    ds_name, seed, n_qubits, pending, v, n_states, n_samples, test_frac = payload
    spec = _W["specs"][ds_name]
    encodings = _W["enc"][n_qubits]
    ansatze = _W["ans"][n_qubits]

    ds = spec.load()
    if ds.X is None or ds.y is None:
        return []
    y_all = _binarize(ds.y)
    X_train, _, _, _ = train_test_split(
        ds.X, y_all, test_size=test_frac, stratify=y_all, random_state=seed
    )

    out: list[dict] = []
    want_enc = sorted({e for e, _ in pending})
    for enc_name in want_enc:
        encoding = encodings[enc_name]
        if isinstance(encoding, DataReuploading):
            continue  # no fixed encoded state; see module docstring
        prep = fit_feature_prep(X_train, encoding)
        Xp = prep.transform(X_train)
        k = min(n_states, len(Xp))
        idx = np.random.default_rng(seed).choice(len(Xp), size=k, replace=False)
        states = encode_states(encoding, Xp[idx])

        for ans_name in sorted(a for e, a in pending if e == enc_name):
            ansatz = ansatze[ans_name]
            t0 = time.perf_counter()
            gv = ansatz.gradient_variance(states, n_samples=n_samples, seed=seed)
            d = int(ansatz.depth())
            out.append({
                "dataset_id": ds_name, "encoding_id": enc_name, "ansatz_id": ans_name,
                "seed": int(seed), "n_qubits": int(n_qubits), "depth": d,
                "n_params": int(ansatz.n_params),
                "grad_var": float(gv),
                "log_grad_var": float(np.log(gv)) if gv > 0 else None,
                "nd": float(n_qubits * d),
                "v": v,
                "n_states": int(k), "n_samples": int(n_samples),
                "wall_time_s": float(time.perf_counter() - t0),
            })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="H1/H2 gradient-variance sweep")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--phase-b-jsonl", default=PHASE_B_JSONL)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--n-qubits", nargs="*", type=int, default=[8])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-states", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=200,
                   help="parameter-shift draws per cell")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    v_table = load_v_table(args.phase_b_jsonl)
    if not v_table:
        print(f"error: no v vectors in {args.phase_b_jsonl}", file=sys.stderr)
        return 1
    done = load_done(args.output)

    enc_names = [e for e, cfg in all_phase_b_configurations(args.n_qubits[0]).items()
                 if not isinstance(cfg, DataReuploading)]
    ans_names = sorted(all_phase_b_ansatze(args.n_qubits[0]).keys())
    ds_names = sorted({d for d, _ in v_table})
    if args.datasets:
        ds_names = [d for d in ds_names if d in args.datasets]

    groups = []
    n_cells = 0
    for n_q in args.n_qubits:
        for ds in ds_names:
            for seed in args.seeds:
                v = v_table.get((ds, seed))
                if v is None:
                    continue
                pending = [(e, a) for e in enc_names for a in ans_names
                           if (ds, e, a, seed, n_q) not in done]
                if not pending:
                    continue
                groups.append((ds, seed, n_q, pending, v,
                               args.n_states, args.n_samples, args.test_frac))
                n_cells += len(pending)

    print(f"[h1] {len(groups)} groups, {n_cells} pending cells "
          f"(done: {len(done)}) → {args.output}")
    print(f"[h1] encodings={len(enc_names)} (re-uploading excluded) "
          f"ansatze={len(ans_names)} n_qubits={args.n_qubits} seeds={args.seeds}")
    if args.dry_run:
        return 0
    if args.limit:
        groups = groups[: args.limit]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.perf_counter()
    with open(args.output, "a") as sink:
        if args.workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers, initializer=_init,
                          initargs=(args.n_qubits,)) as pool:
                for recs in pool.imap_unordered(_cell, groups):
                    for r in recs:
                        sink.write(json.dumps(r) + "\n")
                    sink.flush()
                    written += len(recs)
                    print(f"[h1] {written}/{n_cells} cells "
                          f"({time.perf_counter() - t0:.0f}s)", flush=True)
        else:
            _init(args.n_qubits)
            for g in groups:
                recs = _cell(g)
                for r in recs:
                    sink.write(json.dumps(r) + "\n")
                sink.flush()
                written += len(recs)
                print(f"[h1] {written}/{n_cells} cells "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"[h1] done: {written} records in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
