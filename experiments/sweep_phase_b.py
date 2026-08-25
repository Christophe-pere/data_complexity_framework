"""Phase B sweep driver — bulk orchestration over (dataset × encoding × ansatz × seed).

This wraps the single-run logic of `run_phase_b.run_one` in a resumable sweep with
the mandated encoding-state caching (CLAUDE.md §Computational scope):

    per (dataset, seed):           load + split + y-binarize + v(D) + classical baseline
    per (dataset, encoding, seed): feature prep + q(D, φ) + ρ_φ(x_i) encoding cache
    per (dataset, encoding, ansatz, seed): train quantum classifier reusing the cache

The encoding cache ρ_φ(x_i) is computed once per (dataset, encoding, seed) and reused
across all ansätze; `v(D)`, the classical baseline, and the train/test split are reused
across all (encoding, ansatz). Every value reused is a deterministic function of its
inputs, so each emitted record is byte-identical to what `run_one` would produce
standalone (verified against the smoke-test record).

Resumability: every record is appended to the JSONL immediately, and the existing
(D, φ, ansatz, seed) keys are loaded up front and skipped. Killing and restarting the
process resumes where it left off. Expensive per-group work (split/v/baseline, prep/
cache) is skipped entirely when all ansätze under that group are already done.

Class B (quantum-native, X=None) is the parallel track (framework_v2 §8 / CLAUDE.md
limitation #7) and is skipped here with a logged note; it needs its own runner.

Usage:
    python -m experiments.sweep_phase_b                       # full pending sweep, all A/C/D
    python -m experiments.sweep_phase_b --datasets synth-2sphere real-sonar
    python -m experiments.sweep_phase_b --seeds 0 1 2 3 4 --limit 200
    python -m experiments.sweep_phase_b --dry-run             # report the pending matrix only
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.ansatze import all_phase_b_ansatze  # noqa: E402
from qmlc.datasets import all_phase_b_specs  # noqa: E402
from qmlc.budget import budget_matched_edge  # noqa: E402
from qmlc.encodings import DataReuploading, all_phase_b_configurations  # noqa: E402
from qmlc.qclassifier import QuantumBinaryClassifier, encode_states  # noqa: E402

from experiments.feature_extraction import compute_q, compute_v  # noqa: E402
from experiments.feature_prep import fit_feature_prep  # noqa: E402
from experiments.results import (  # noqa: E402
    PhaseBRecord,
    append_record,
    load_existing_keys,
    now_utc,
)

DEFAULT_OUTPUT = "experiments/results/phase_b.jsonl"
PARALLEL_TRACK_KLASS = {"B"}  # quantum-native, handled by a separate runner


def _binarize(y: np.ndarray, verbose: bool = False) -> np.ndarray:
    yu = np.unique(y)
    if len(yu) == 2:
        return y
    if verbose:
        print(f"  note: {len(yu)}-class target; binarizing y > median(y)")
    return (y > np.median(y)).astype(np.int64)


def compute_group(
    spec,
    seed: int,
    encodings: dict,
    ansatze: dict,
    pending_cells: Sequence[tuple[str, str]],
    *,
    n_qubits: int = 8,
    test_frac: float = 0.2,
    q_n_states: int = 200,
    q_epochs: int = 8,
    q_batch_size: int = 32,
    q_lr: float = 0.1,
    classical_C: float = 1.0,
    sink_path: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Compute + write the records for the pending (encoding, ansatz) cells under one
    (dataset, seed); return the count written. The per-(dataset, seed) work (load,
    split, v(D), classical baseline) and the per-(dataset, encoding, seed) work
    (prep, q(D, φ), ρ_φ cache) are each done once and reused across cells.

    Each record is appended to `sink_path` **as soon as it is computed** (not after
    the whole group), so progress is visible cell-by-cell and a crash loses only the
    in-flight cell. Encodings are ordered standard-before-reup so the cheap cached
    cells flush before the slow re-uploading ones."""
    dataset_name = spec.name
    want = set(pending_cells)
    if not want:
        return 0

    ds = spec.load()
    if ds.X is None:
        if verbose:
            print(f"  [skip] {dataset_name}: quantum-native (X=None) — parallel track")
        return 0
    if ds.y is None:
        if verbose:
            print(f"  [skip] {dataset_name}: unlabeled")
        return 0
    y_all = _binarize(ds.y, verbose=verbose)
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, y_all, test_size=test_frac, stratify=y_all, random_state=seed
    )
    v = compute_v(X_train)
    classical = LogisticRegression(C=classical_C, max_iter=2000, random_state=seed)
    classical.fit(X_train, y_train)
    eps_train_c = float(1.0 - classical.score(X_train, y_train))
    eps_test_c = float(1.0 - classical.score(X_test, y_test))

    config = {
        "test_frac": test_frac, "q_n_states": q_n_states, "q_epochs": q_epochs,
        "q_batch_size": q_batch_size, "q_lr": q_lr,
        "classical": "sklearn.LogisticRegression", "classical_C": classical_C,
    }

    n_written = 0
    # Cheap encodings first (smallest two-qubit-gate count → fastest cell / cache),
    # re-uploading last — so the first record flushes ASAP (angle-none before amplitude).
    pending_encs = sorted(
        {e for e, _ in want},
        key=lambda e: (isinstance(encodings[e], DataReuploading),
                       encodings[e].depth() * encodings[e].gates_per_layer(), e),
    )
    for encoding_name in pending_encs:
        encoding = encodings[encoding_name]
        is_reupload = isinstance(encoding, DataReuploading)

        prep = fit_feature_prep(X_train, encoding)
        X_train_p = prep.transform(X_train)
        X_test_p = prep.transform(X_test)

        n_states = min(q_n_states, len(X_train_p))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train_p), size=n_states, replace=False)
        psi_stack = np.stack([encoding.encode(X_train_p[i]) for i in idx], axis=0)
        q = compute_q(psi_stack, n_qubits=n_qubits)

        enc_train = None if is_reupload else encode_states(encoding, X_train_p)
        enc_test = None if is_reupload else encode_states(encoding, X_test_p)

        for ansatz_name in sorted(a for e, a in want if e == encoding_name):
            t0 = time.perf_counter()
            ansatz = ansatze[ansatz_name]
            qclf = QuantumBinaryClassifier(encoding=encoding, ansatz=ansatz)
            qclf.fit(
                X_train_p, y_train, epochs=q_epochs, batch_size=q_batch_size,
                lr=q_lr, seed=seed, verbose=verbose, enc=enc_train,
            )
            eps_train_q = float(1.0 - qclf.score(X_train_p, y_train, enc=enc_train))
            eps_test_q = float(1.0 - qclf.score(X_test_p, y_test, enc=enc_test))
            e_budget = budget_matched_edge(eps_test_q, eps_test_c, encoding, ansatz)
            record = PhaseBRecord(
                dataset_id=dataset_name, encoding_id=encoding_name, ansatz_id=ansatz_name,
                seed=int(seed), n_qubits=n_qubits, v=v, q=q,
                epsilon_train_q=eps_train_q, epsilon_test_q=eps_test_q,
                epsilon_train_c=eps_train_c, epsilon_test_c=eps_test_c,
                e_raw=float(eps_test_c - eps_test_q), e_budget=e_budget,
                gap_obs=float(eps_test_q - eps_train_q), epsilon_emp=eps_train_q,
                wall_time_s={"quantum_fit": float(time.perf_counter() - t0)},
                timestamp=now_utc(), config=config,
            )
            if sink_path is not None:
                append_record(sink_path, record)
            n_written += 1
            if verbose:
                print(f"  [{dataset_name} × {encoding_name} × {ansatz_name} s{seed}] "
                      f"e_raw={record.e_raw:+.3f} e_budget={e_budget:+.3f} "
                      f"({record.wall_time_s['quantum_fit']:.1f}s)")
    return n_written


# --- multiprocessing worker (top-level + per-worker init so maps build once) ---
_WORKER: dict = {}


def _init_worker(n_qubits: int) -> None:
    _WORKER["specs"] = {s.name: s for s in all_phase_b_specs()}
    _WORKER["enc"] = all_phase_b_configurations(n_qubits)
    _WORKER["ans"] = all_phase_b_ansatze(n_qubits)
    _WORKER["n_qubits"] = n_qubits


def _group_worker(payload: tuple) -> tuple[str, int, int]:
    """Worker: compute a (dataset, seed) group, streaming each record to the shared
    output (flock-serialized append). Returns (dataset, seed, n_written)."""
    dataset_name, seed, pending_cells, output, run_kwargs = payload
    n = compute_group(
        _WORKER["specs"][dataset_name], seed, _WORKER["enc"], _WORKER["ans"],
        pending_cells, n_qubits=_WORKER["n_qubits"], sink_path=output,
        verbose=False, **run_kwargs,
    )
    return dataset_name, int(seed), n


def sweep(
    datasets: Optional[Sequence[str]] = None,
    encodings: Optional[Sequence[str]] = None,
    ansatze: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    output: str = DEFAULT_OUTPUT,
    n_qubits: int = 8,
    limit: Optional[int] = None,
    workers: int = 1,
    dry_run: bool = False,
    verbose: bool = True,
    **run_kwargs,
) -> int:
    """Run (or report) the pending Phase B sweep. Returns records appended.

    Parallelism: the work unit is a (dataset, seed) group so the per-group cache
    (split, v(D), classical baseline, ρ_φ) is shared within a worker. Workers
    compute records and return them; the parent process is the single JSONL
    writer, so appends never race. Resumable: existing keys are skipped, and each
    record is flushed as soon as its group returns."""
    specs = {s.name: s for s in all_phase_b_specs()}
    enc_map = all_phase_b_configurations(n_qubits)
    ans_map = all_phase_b_ansatze(n_qubits)

    ds_names = list(datasets) if datasets else [
        n for n, s in specs.items() if s.klass not in PARALLEL_TRACK_KLASS
    ]
    enc_names = list(encodings) if encodings else sorted(enc_map)
    ans_names = list(ansatze) if ansatze else sorted(ans_map)

    for n in ds_names:
        if n not in specs:
            raise ValueError(f"unknown dataset {n!r}; choose from {sorted(specs)}")
    for e in enc_names:
        if e not in enc_map:
            raise ValueError(f"unknown encoding {e!r}; choose from {sorted(enc_map)}")
    for a in ans_names:
        if a not in ans_map:
            raise ValueError(f"unknown ansatz {a!r}; choose from {sorted(ans_map)}")

    seen = load_existing_keys(output)
    full = list(itertools.product(ds_names, enc_names, ans_names, seeds))
    pending = [k for k in full if k not in seen]
    print(f"Sweep matrix: {len(ds_names)} datasets × {len(enc_names)} enc × "
          f"{len(ans_names)} ans × {len(seeds)} seeds = {len(full)} cells")
    print(f"  already done: {len(full) - len(pending)}   pending: {len(pending)}"
          + (f"   (this run capped at {limit})" if limit else ""))

    # Group pending cells by (dataset, seed); skip the quantum-native parallel track.
    groups: list[tuple[str, int, list[tuple[str, str]]]] = []
    n_cells = 0
    for dataset_name in ds_names:
        if specs[dataset_name].klass in PARALLEL_TRACK_KLASS:
            continue
        for seed in seeds:
            cells = [(e, a) for e in enc_names for a in ans_names
                     if (dataset_name, e, a, seed) not in seen]
            if limit is not None:
                cells = cells[: max(0, limit - n_cells)]
            if cells:
                groups.append((dataset_name, int(seed), cells))
                n_cells += len(cells)
        if limit is not None and n_cells >= limit:
            break
    print(f"  dispatching {len(groups)} (dataset, seed) groups = {n_cells} cells "
          f"on {workers} worker(s)")
    if dry_run:
        return 0

    appended = 0
    t_start = time.perf_counter()

    def _progress(n: int, tag: str) -> None:
        nonlocal appended
        appended += n
        if n and verbose:
            rate = appended / (time.perf_counter() - t_start)
            print(f"[{tag}] +{n} (total {appended}/{n_cells}, {rate:.2f} rec/s)")

    if workers <= 1:
        # Serial: write straight to the main file, cell-by-cell.
        for dataset_name, seed, cells in groups:
            n = compute_group(
                specs[dataset_name], seed, enc_map, ans_map, cells,
                n_qubits=n_qubits, sink_path=output, verbose=verbose, **run_kwargs,
            )
            _progress(n, f"{dataset_name} s{seed}")
    else:
        # Parallel: workers stream to the shared output (flock-serialized append).
        import multiprocessing as mp
        payloads = [(d, s, cells, output, run_kwargs) for d, s, cells in groups]
        ctx = mp.get_context("spawn")
        # maxtasksperchild recycles each worker after a bounded number of (dataset,
        # seed) groups, releasing memory that gudhi/numpy do not return to the OS
        # (the persistent-homology Rips complex in compute_v is the main hog). This
        # caps per-worker RSS to ~one group's peak instead of letting it accumulate
        # over the whole run — the fix for the 32 GB machine swap-thrash collapse.
        with ctx.Pool(
            workers, initializer=_init_worker, initargs=(n_qubits,),
            maxtasksperchild=2,
        ) as pool:
            for d, s, n in pool.imap_unordered(_group_worker, payloads):
                _progress(n, f"{d} s{s}")

    print(f"\nDone: appended {appended} records to {output}")
    return appended


def main() -> int:
    p = argparse.ArgumentParser(description="Phase B sweep driver (resumable, cached)")
    p.add_argument("--datasets", nargs="*", default=None, help="default: all A/C/D in registry")
    p.add_argument("--encodings", nargs="*", default=None, help="default: all 10")
    p.add_argument("--ansatze", nargs="*", default=None, help="default: all 6")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--q-n-states", type=int, default=200)
    p.add_argument("--limit", type=int, default=None, help="max records to append this invocation")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes (1 = serial)")
    p.add_argument("--dry-run", action="store_true", help="report pending matrix, run nothing")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    sweep(
        datasets=args.datasets, encodings=args.encodings, ansatze=args.ansatze,
        seeds=args.seeds, output=args.output, n_qubits=args.n_qubits, limit=args.limit,
        workers=args.workers, dry_run=args.dry_run, verbose=not args.quiet,
        test_frac=0.2, q_n_states=args.q_n_states, q_epochs=args.epochs,
        q_batch_size=args.batch_size, q_lr=args.lr, classical_C=1.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
