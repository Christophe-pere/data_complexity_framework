"""Class B (quantum-native / Mode-N) runner — PROTOTYPE.

Per (dataset_B, ansatz, T, seed):
  * Quantum model: variational ansatz applied DIRECTLY to the input state |ψ_i⟩ (no encoding —
    the state is the input; reuses QuantumBinaryClassifier's `enc=` path), measured on Z₀ → a_q.
  * Honest classical baseline: T-shot classical shadows of each state → k≤2 Pauli features →
    best-of-standard-classical (same 6-model set as Amendment 1) → a_c. Matched resource = T.
  * Edge (Mode N): ê_N = a_c − a_q, with C_φ = 0 (framework_v2 §2.5) ⇒ ê_budget = ê_raw = ê_N.
  * v_N(D): quantum-native complexity (entanglement mean/var here; v_mi/v_mag/v_bond are TODO).

Separate from the classical-data sweep (limitation #7). Output: experiments/results/phase_b_classB.jsonl.
Exploratory / Part III-leaning — see docs/class_b_quantum_native_spec.md.

Usage:
  python -m experiments.sweep_phase_b_classB --datasets qn-tfim --ansatze hea-L3 --T 1000 --limit 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.ansatze import all_phase_b_ansatze  # noqa: E402
from qmlc.datasets import (  # noqa: E402
    all_phase_b_specs, class_b_probe_specs, class_b_qspin_specs,
)
from qmlc.encodings import AngleEncoding  # noqa: E402 (placeholder; unused on enc path)
from qmlc.qclassifier import QuantumBinaryClassifier  # noqa: E402
from qmlc.quantum import (  # noqa: E402
    EnsembleEntanglement, EnsembleTotalCorrelation,
    EnsembleStabilizerMagic, EnsembleMPSBond,
)
from qmlc.shadows import ShadowFeatureConfig, shadow_feature_matrix  # noqa: E402
from analysis.classical_baselines import _model_set, MODEL_KEYS  # noqa: E402

OUT = "experiments/results/phase_b_classB.jsonl"


# Full quantum-native complexity vector v_N (qmlc/quantum.py ensemble metrics):
#   v_ent  — mean half-cut entanglement entropy  (nats)
#   v_mi   — mean multipartite total correlation (nats)
#   v_mag  — mean stabilizer Rényi-2 magic       (bits)
#   v_bond — mean optimal MPS bond dim @ fixed ε  (dimensionless)
_VN_METRICS = {
    "v_ent": EnsembleEntanglement(), "v_mi": EnsembleTotalCorrelation(),
    "v_mag": EnsembleStabilizerMagic(), "v_bond": EnsembleMPSBond(),
}


def quantum_native_v(states: np.ndarray, n: int) -> dict:
    """Full v_N: ensemble means (+ var/uncertainty) of the four quantum-native components
    (docs/class_b_quantum_native_spec.md §3; framework_v2 §3 parallel track)."""
    out: dict = {}
    for name, metric in _VN_METRICS.items():
        r = metric.compute(states)
        out[f"{name}_mean"] = float(r.value)
        out[f"{name}_var"] = float(r.metadata.get("var", 0.0))
    return out


def best_classical_error(Xtr, ytr, Xte, yte, seed) -> tuple[float, str]:
    errs = {}
    for mk, m in _model_set(seed).items():
        m.fit(Xtr, ytr)
        errs[mk] = float(1.0 - m.score(Xte, yte))
    best = min(MODEL_KEYS, key=lambda k: errs[k])
    return errs[best], best, errs


def _parity_expectation(states: np.ndarray, w: int, n: int) -> np.ndarray:
    """⟨∏_{j<w} Z_j⟩ for each state (batched): Σ |amp|² · (−1)^(parity of first w bits)."""
    prob = (np.abs(states) ** 2).reshape((-1,) + (2,) * n)
    signs = np.ones([2] * n)
    for q in range(w):
        shp = [1] * n; shp[q] = 2
        signs = signs * np.array([1.0, -1.0]).reshape(shp)
    return (prob * signs).reshape(states.shape[0], -1).sum(axis=1)


# --- trainable measurement-selection head (learns the Z-parity support, not told w) ---
def _gated_rx_gate(theta: float) -> np.ndarray:
    """4×4 gate on (control=j, target=0): identity on |0⟩_j, Rx(θ) on |1⟩_j (q_j high bit).
    θ=π ⇒ CNOT(j→0) (up to phase, irrelevant for ⟨Z₀⟩); θ=0 ⇒ identity ⇒ qubit j excluded."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    G = np.eye(4, dtype=complex)
    G[2:, 2:] = np.array([[c, -1j * s], [-1j * s, c]])
    return G


def _apply_2q(T: np.ndarray, G: np.ndarray, qa: int, qb: int, n: int) -> np.ndarray:
    """Apply 4×4 gate G (qa high bit, qb low bit) to a batched state tensor (B,)+(2,)*n."""
    B = T.shape[0]
    axes = [1 + qa, 1 + qb]
    T2 = np.moveaxis(T, axes, [1, 2])
    shp = T2.shape
    T2 = np.einsum("ij,bjk->bik", G, T2.reshape(B, 4, -1)).reshape(shp)
    return np.moveaxis(T2, [1, 2], axes)


def _msel_forward(states: np.ndarray, theta: np.ndarray, n: int) -> np.ndarray:
    """⟨Z₀⟩ after gated-CNOT ladder folding qubits 1..n-1 (angle θ_{j-1}) onto qubit 0."""
    T = states.reshape((-1,) + (2,) * n).astype(complex)
    for j in range(1, n):
        T = _apply_2q(T, _gated_rx_gate(theta[j - 1]), j, 0, n)
    psi = T.reshape(states.shape[0], -1)
    return _parity_expectation(psi, 1, n)          # ⟨Z₀⟩ of the folded state


def measurement_selection_error(st_tr, y_tr, st_te, y_te, n, seed, epochs=200, lr=0.3) -> tuple:
    """Train the parity-support angles θ on BCE (finite-difference GD over n-1 angles), then
    report test error. The model must DISCOVER which qubits carry the label — it is not told w."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.3, 0.6, size=n - 1)      # small init, symmetry-broken
    y = np.asarray(y_tr, float)

    def bce(th):
        z0 = _msel_forward(st_tr, th, n)
        p = np.clip((1 - z0) / 2, 1e-6, 1 - 1e-6)  # P(label=1)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    eps = 1e-3
    for _ in range(epochs):
        base = bce(theta)
        grad = np.empty_like(theta)
        for i in range(len(theta)):
            th2 = theta.copy(); th2[i] += eps
            grad[i] = (bce(th2) - base) / eps
        theta -= lr * grad
        theta = np.clip(theta, 0.0, np.pi)
    z0_te = _msel_forward(st_te, theta, n)
    pred = (z0_te < 0).astype(np.int64)
    return float(np.mean(pred != y_te)), theta


def run_cell(spec, ansatz_name, ansatz, T, seed, n, quantum_model="variational-z0",
             verbose=False) -> dict:
    ds = spec.load()
    states, y = ds.states, np.asarray(ds.y)
    idx = np.arange(len(states))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    st_tr, st_te = states[tr], states[te]
    y_tr, y_te = y[tr], y[te]

    # --- quantum model on states ---
    if quantum_model == "parity-readout":
        # Structure-aware Mode-N learner: measure the global parity observable P_w directly
        # (a physically O(1) quantum measurement the shadow-limited classical learner cannot
        # afford). w is taken from the dataset construction — this is the positive-control /
        # oracle-readout upper bound on the quantum side, establishing that a baseline-robust
        # edge EXISTS given the right observable, not that a generic ansatz finds it.
        w = int(spec.load().metadata.get("weight", n))
        pred_te = (_parity_expectation(st_te, w, n) < 0).astype(np.int64)  # ⟨P_w⟩<0 ⇒ label 1
        a_q = float(np.mean(pred_te != y_te))
    elif quantum_model == "measurement-selection":
        # Trainable middle ground: learns which qubits' Z folds into the readout (support S),
        # NOT told w. Tests whether the edge is *learnable* rather than oracle-given.
        a_q, theta = measurement_selection_error(st_tr, y_tr, st_te, y_te, n, seed)
    else:
        # Generic variational circuit + Z₀ readout (enc= path; placeholder encoding, dummy X).
        enc = AngleEncoding(n_qubits=n)
        qclf = QuantumBinaryClassifier(encoding=enc, ansatz=ansatz)
        dummyX_tr = np.zeros((len(st_tr), n))  # unused on enc path (indexing/length only)
        dummyX_te = np.zeros((len(st_te), n))
        qclf.fit(dummyX_tr, y_tr, epochs=8, batch_size=32, lr=0.1, seed=seed,
                 verbose=False, enc=st_tr)
        a_q = 1.0 - qclf.score(dummyX_te, y_te, enc=st_te)

    # --- honest classical baseline: shadows @ budget T → best-of-standard-classical ---
    cfg = ShadowFeatureConfig(T=T, k_max=2, seed=seed)
    feats, terms = shadow_feature_matrix(states, n, cfg)
    a_c, best_model, errs = best_classical_error(feats[tr], y_tr, feats[te], y_te, seed)

    v_n = quantum_native_v(st_tr, n)
    rec = {
        "dataset_id": spec.name, "encoding_id": "none-native", "ansatz_id": ansatz_name,
        "quantum_model": quantum_model,
        "data_access_mode": "N", "T": int(T), "seed": int(seed), "n_qubits": n,
        "C_phi": 0.0, "n_shadow_features": len(terms),
        "epsilon_test_q": float(a_q), "epsilon_test_c": float(a_c),
        "classical_best_model": best_model, "classical_errors": errs,
        "e_raw": float(a_c - a_q), "e_budget": float(a_c - a_q),  # C_φ=0 ⇒ equal
        "v_native": v_n, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if verbose:
        print(f"  [{spec.name} × {ansatz_name} × T={T} s{seed}] a_q={a_q:.3f} "
              f"a_c={a_c:.3f}({best_model}) ê_N={rec['e_raw']:+.3f}")
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description="Class B quantum-native runner (prototype)")
    p.add_argument("--datasets", nargs="*", default=None, help="default: all klass B")
    p.add_argument("--probe-weights", nargs="*", type=int, default=None,
                   help="run the EXPLORATORY shadow-hard parity probe at these weights "
                        "(e.g. 1 2 4 6 8) instead of the registry Class B datasets")
    p.add_argument("--qspin", nargs="*", default=None,
                   help="run PennyLane qspin phase-classification datasets (e.g. Ising Heisenberg)")
    p.add_argument("--ansatze", nargs="*", default=["hea-L3"])
    p.add_argument("--quantum-model", default="variational-z0",
                   choices=("variational-z0", "parity-readout", "measurement-selection"),
                   help="quantum learner: generic variational circuit + Z0 readout; the "
                        "structure-aware global-parity measurement (oracle upper bound); or the "
                        "trainable measurement-selection head that must discover the parity support")
    p.add_argument("--T", nargs="*", type=int, default=[1000])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=OUT)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.probe_weights:
        specs = {s.name: s for s in class_b_probe_specs(weights=args.probe_weights,
                                                        n_qubits=args.n_qubits)}
    elif args.qspin:
        specs = {s.name: s for s in class_b_qspin_specs(systems=args.qspin)}
    else:
        specs = {s.name: s for s in all_phase_b_specs() if s.klass == "B"}
    ans_map = all_phase_b_ansatze(args.n_qubits)
    ds_names = args.datasets or sorted(specs)
    written = 0
    with open(args.output, "a") as fout:
        for dn in ds_names:
            spec = specs[dn]
            nq = spec.load().n_qubits or args.n_qubits
            for an in args.ansatze:
                for T in args.T:
                    for seed in args.seeds:
                        rec = run_cell(spec, an, ans_map[an], T, seed, nq,
                                       quantum_model=args.quantum_model, verbose=args.verbose)
                        fout.write(json.dumps(rec) + "\n"); fout.flush()
                        written += 1
                        if args.limit and written >= args.limit:
                            print(f"limit {args.limit} reached; wrote {written}")
                            return 0
    print(f"wrote {written} Class B records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
