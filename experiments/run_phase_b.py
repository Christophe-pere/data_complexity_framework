"""Phase B experiment runner — one (dataset, encoding, ansatz, seed) per invocation.

Stages:
    1. Load dataset from registry by name.
    2. Stratified deterministic train/test split (80/20).
    3. Fit FeaturePrep (StandardScaler + PCA/zero-pad to encoding's input dim).
    4. Compute v(D) on the training split (5 implemented components).
    5. Encode a subsample of training points → embedded states; compute q(D, φ).
    6. Train classical baseline (sklearn LogisticRegression) → ε_train_c, ε_test_c.
    7. Train quantum classifier (encoding + ansatz, parameter-shift SGD) →
       ε_train_q, ε_test_q.
    8. Build PhaseBRecord and append to JSONL (dedup by (D, φ, ansatz, seed)).

This is the "full skeleton + smoke test" scaffolding deliverable. Bulk-sweep
orchestration (Hydra configs, parallel workers, encoding-state caching across
ansatz pairs) is layered on top later and does not require changes to this
file.

Usage:
    source ../../research-dev/bin/activate
    python -m experiments.run_phase_b \\
        --dataset synth-2sphere --encoding angle-linear --ansatz hea-L1 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.ansatze import all_phase_b_ansatze  # noqa: E402
from qmlc.budget import budget_matched_edge  # noqa: E402
from qmlc.datasets import all_phase_b_specs  # noqa: E402
from qmlc.encodings import all_phase_b_configurations  # noqa: E402
from qmlc.qclassifier import QuantumBinaryClassifier  # noqa: E402

from experiments.feature_extraction import compute_q, compute_v  # noqa: E402
from experiments.feature_prep import fit_feature_prep  # noqa: E402
from experiments.results import (  # noqa: E402
    PhaseBRecord,
    append_record,
    load_existing_keys,
    now_utc,
)


def _resolve_spec(name: str, n_qubits: int):
    """Resolve dataset, encoding, ansatz names to instances. Raises on miss."""
    specs = {s.name: s for s in all_phase_b_specs()}
    enc_map = all_phase_b_configurations(n_qubits)
    ans_map = all_phase_b_ansatze(n_qubits)
    return specs, enc_map, ans_map


def run_one(
    dataset_name: str,
    encoding_name: str,
    ansatz_name: str,
    seed: int = 0,
    n_qubits: int = 8,
    test_frac: float = 0.2,
    q_n_states: int = 200,
    q_epochs: int = 8,
    q_batch_size: int = 32,
    q_lr: float = 0.1,
    classical_C: float = 1.0,
    verbose: bool = False,
) -> PhaseBRecord:
    """Run one (D, φ, ansatz, seed) experiment and return the record."""
    specs, enc_map, ans_map = _resolve_spec(dataset_name, n_qubits)
    if dataset_name not in specs:
        raise ValueError(f"unknown dataset {dataset_name!r}; choose from {sorted(specs)}")
    if encoding_name not in enc_map:
        raise ValueError(f"unknown encoding {encoding_name!r}; choose from {sorted(enc_map)}")
    if ansatz_name not in ans_map:
        raise ValueError(f"unknown ansatz {ansatz_name!r}; choose from {sorted(ans_map)}")

    spec = specs[dataset_name]
    encoding = enc_map[encoding_name]
    ansatz = ans_map[ansatz_name]

    wall: dict[str, float] = {}
    t0 = time.perf_counter()

    # 1. Load dataset
    ds = spec.load()
    if ds.y is None:
        raise ValueError(f"dataset {dataset_name} has no labels; need supervised data")
    y_unique = np.unique(ds.y)
    if len(y_unique) != 2:
        if verbose:
            print(f"  note: multi-class dataset ({len(y_unique)} classes); binarizing y > median(y).")
        y_all = (ds.y > np.median(ds.y)).astype(np.int64)
    else:
        y_all = ds.y
    wall["load"] = time.perf_counter() - t0

    # 2. Split
    t1 = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, y_all, test_size=test_frac, stratify=y_all, random_state=seed
    )
    wall["split"] = time.perf_counter() - t1

    # 3. Feature preparation
    t1 = time.perf_counter()
    prep = fit_feature_prep(X_train, encoding)
    X_train_p = prep.transform(X_train)
    X_test_p = prep.transform(X_test)
    wall["feature_prep"] = time.perf_counter() - t1

    # 4. v(D) on training split
    t1 = time.perf_counter()
    v = compute_v(X_train)
    wall["compute_v"] = time.perf_counter() - t1

    # 5. q(D, φ) on a subsample of embedded states
    t1 = time.perf_counter()
    n_states = min(q_n_states, len(X_train_p))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train_p), size=n_states, replace=False)
    # Encode (re-uploading is encoded without trainable layers — encoding-only path)
    psi_stack = np.stack([encoding.encode(X_train_p[i]) for i in idx], axis=0)
    q = compute_q(psi_stack, n_qubits=n_qubits)
    wall["compute_q"] = time.perf_counter() - t1

    # 6. Classical baseline
    t1 = time.perf_counter()
    classical = LogisticRegression(C=classical_C, max_iter=2000, random_state=seed)
    classical.fit(X_train, y_train)
    eps_train_c = 1.0 - classical.score(X_train, y_train)
    eps_test_c = 1.0 - classical.score(X_test, y_test)
    wall["classical_fit"] = time.perf_counter() - t1

    # 7. Quantum classifier
    t1 = time.perf_counter()
    qclf = QuantumBinaryClassifier(encoding=encoding, ansatz=ansatz)
    qclf.fit(
        X_train_p, y_train,
        epochs=q_epochs, batch_size=q_batch_size, lr=q_lr,
        seed=seed, verbose=verbose,
    )
    eps_train_q = 1.0 - qclf.score(X_train_p, y_train)
    eps_test_q = 1.0 - qclf.score(X_test_p, y_test)
    wall["quantum_fit"] = time.perf_counter() - t1

    # 8. Edges & gap
    e_raw = float(eps_test_c - eps_test_q)
    e_budget = budget_matched_edge(eps_test_q, eps_test_c, encoding, ansatz)
    gap_obs = float(eps_test_q - eps_train_q)
    epsilon_emp = float(eps_train_q)

    wall["total"] = time.perf_counter() - t0

    record = PhaseBRecord(
        dataset_id=dataset_name,
        encoding_id=encoding_name,
        ansatz_id=ansatz_name,
        seed=int(seed),
        n_qubits=n_qubits,
        v=v,
        q=q,
        epsilon_train_q=float(eps_train_q),
        epsilon_test_q=float(eps_test_q),
        epsilon_train_c=float(eps_train_c),
        epsilon_test_c=float(eps_test_c),
        e_raw=e_raw,
        e_budget=e_budget,            # fidelity-discounted ê_budget (qmlc.budget, §2.5)
        gap_obs=gap_obs,
        epsilon_emp=epsilon_emp,
        wall_time_s={k: float(v_) for k, v_ in wall.items()},
        timestamp=now_utc(),
        config={
            "test_frac": test_frac,
            "q_n_states": q_n_states,
            "q_epochs": q_epochs,
            "q_batch_size": q_batch_size,
            "q_lr": q_lr,
            "classical": "sklearn.LogisticRegression",
            "classical_C": classical_C,
        },
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B runner — one (D, φ, ansatz, seed) per invocation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoding", required=True)
    parser.add_argument("--ansatz", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--q-n-states", type=int, default=200, help="state subsample for q(D, φ)")
    parser.add_argument("--output", default="experiments/results/phase_b.jsonl")
    parser.add_argument("--force", action="store_true", help="re-run even if (D, φ, ansatz, seed) is already in the JSONL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    key = (args.dataset, args.encoding, args.ansatz, args.seed)
    if not args.force and key in load_existing_keys(args.output):
        print(f"Skipping {key} — already present in {args.output} (use --force to override)")
        return 0

    print(f"Running Phase B: {args.dataset} × {args.encoding} × {args.ansatz} (seed={args.seed})")
    record = run_one(
        dataset_name=args.dataset,
        encoding_name=args.encoding,
        ansatz_name=args.ansatz,
        seed=args.seed,
        n_qubits=args.n_qubits,
        q_epochs=args.epochs,
        q_batch_size=args.batch_size,
        q_lr=args.lr,
        q_n_states=args.q_n_states,
        verbose=args.verbose,
    )

    append_record(args.output, record)
    print()
    print(f"  ε_train_q = {record.epsilon_train_q:.4f}   ε_test_q = {record.epsilon_test_q:.4f}")
    print(f"  ε_train_c = {record.epsilon_train_c:.4f}   ε_test_c = {record.epsilon_test_c:.4f}")
    print(f"  e_raw     = {record.e_raw:+.4f}   (positive ⇒ Q beats C)")
    print(f"  gap_obs   = {record.gap_obs:+.4f}   ε_emp = {record.epsilon_emp:.4f}")
    print(f"  q[ent_S]  = {record.q['entanglement_entropy_mean']:.4f} (±{np.sqrt(record.q['entanglement_entropy_var']):.4f})")
    print(f"  q[eff_rk] = {record.q['embedded_kernel_effective_rank']:.2f}")
    print(f"  wall times (s): {record.wall_time_s}")
    print(f"\nAppended record to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
