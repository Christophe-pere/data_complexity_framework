"""Dataset registry — hashing, pre-screening, and frozen registry generation.

Pre-screening verifies that v(D) components are computable on each dataset.
A dataset passes if every implemented v_i metric returns a finite value within
`max_retries` attempts. Per CLAUDE.md ("Pre-screening rejection threshold:
any v_i that fails to converge after three retries"), failing datasets are still
written to the registry with `screening.passed = false` so the user can inspect
and decide; they are not silently dropped.

Currently implemented v(D) components (in qmlc/classical.py):
  - differential_entropy_knn_kl  (v_S)
  - intrinsic_dim_pca            (v_d_pca)
  - intrinsic_dim_twonn          (v_d_twonn)
  - compression_ratio_gzip       (v_compress; placeholder for v_K)
  - topological_complexity_rips  (v_top, with per-dim Betti in metadata)

Not yet implemented (flagged in registry.json under `not_implemented`):
  - second_order_mi              (v_corr2)
  - higher_order_mi              (v_corr3+)
  - kernel_effective_rank        (v_kspec)
  - mdl_autoencoder              (v_K — current proxy is gzip)

These will be added in follow-up sessions; their absence does not block
Phase B if the registered v_i suffice to characterize the dataset.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .base import MetricResult
from .classical import (
    CompressionRatio,
    EntropyKL,
    HigherOrderMI,
    IntrinsicDimPCA,
    IntrinsicDimTwoNN,
    KernelEffectiveRankRBF,
    MDLAutoencoder,
    PairwiseMutualInformation,
    PersistentHomology,
)
from .datasets import LoadedDataset, LoaderSpec, all_phase_b_specs
from .quantum import (
    EnsembleEntanglement,
    EnsembleMPSBond,
    EnsembleStabilizerMagic,
    EnsembleTotalCorrelation,
)


# Components flagged as not yet implemented in qmlc/. Documented in registry.json
# so downstream code knows these v_i values will be missing for now. All four
# of the previously-missing components (v_corr2, v_corr3+, v_kspec, v_K) were
# implemented 2026-06-01; the list is now empty.
NOT_IMPLEMENTED_COMPONENTS: list[str] = []

# Quantum-native v(D) components (Class B parallel track, OSF §5.2). Computed on
# the state ensemble rather than on classical (X, y).
QN_IMPLEMENTED_COMPONENTS = [
    "entanglement_entropy_mean",   # v_ent
    "total_correlation_mean",      # v_mi
    "stabilizer_renyi2_mean",      # v_mag
    "mps_bond_dim_mean",           # v_bond
]

# Persistent homology is the screening bottleneck on n=2000 in d≤8. We subsample
# the point cloud and tighten the Rips radius for the screening only; final
# v_top values for Phase B will be computed at the full sample / default params.
PH_SCREENING_SUBSAMPLE = 300
PH_SCREENING_MAX_DIM = 1            # b_0, b_1 only — b_2 explodes in d≥6
PH_SCREENING_MAX_EDGE_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_dataset(ds: LoadedDataset) -> str:
    """SHA-256 over the byte representation of (X, y).

    Uses ascontiguousarray + tobytes for canonical ordering. Different dtypes
    produce different hashes, so loaders must be dtype-stable across runs.
    """
    h = hashlib.sha256()
    if ds.X is not None:
        X = np.ascontiguousarray(ds.X)
        h.update(X.dtype.str.encode("ascii"))
        h.update(str(X.shape).encode("ascii"))
        h.update(X.tobytes())
    if ds.states is not None:
        S = np.ascontiguousarray(ds.states)
        h.update(b"|states|")
        h.update(S.dtype.str.encode("ascii"))
        h.update(str(S.shape).encode("ascii"))
        h.update(S.tobytes())
    if ds.y is not None:
        y = np.ascontiguousarray(ds.y)
        h.update(b"|y|")
        h.update(y.dtype.str.encode("ascii"))
        h.update(str(y.shape).encode("ascii"))
        h.update(y.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pre-screening
# ---------------------------------------------------------------------------

@dataclass
class ComponentResult:
    """Per-metric outcome on a single dataset."""

    value: Optional[float]
    uncertainty: Optional[float]
    elapsed_s: float
    estimator: str
    metadata_summary: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None  # str(exception) if computation failed


@dataclass
class ScreeningResult:
    passed: bool
    components: dict[str, ComponentResult]
    failures: list[str]            # names of metrics that failed all retries
    not_implemented: list[str]     # v_i components not yet in qmlc/


def _summarize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Strip large fields (barcodes, raw arrays) from metric metadata for JSON output."""
    summary = {}
    for k, v in meta.items():
        if k == "barcodes":
            # Barcodes are a list of (birth, death) tuples per dim — keep counts only
            summary["n_barcodes_per_dim"] = {dim: len(bars) for dim, bars in v.items()}
        elif isinstance(v, dict):
            # Convert nested dicts (per-dim Betti, persistence) to JSON-safe form
            summary[k] = {str(kk): vv for kk, vv in v.items()}
        elif isinstance(v, (int, float, str, bool, type(None))):
            summary[k] = v
        elif isinstance(v, np.ndarray):
            summary[k] = v.tolist() if v.size < 100 else f"<ndarray shape={v.shape}>"
        else:
            summary[k] = str(v)
    return summary


def _try_compute(metric, X: np.ndarray, max_retries: int) -> tuple[Optional[MetricResult], Optional[str]]:
    """Run metric.compute(X) with retries. Returns (result, error_or_None).

    A retry is triggered when a non-finite value is produced or an exception is
    raised. Estimator parameters are not increased between retries (the Phase A
    metric library does not expose a uniform "increase budget" hook); for those
    metrics that fail under the default budget on a given dataset, the failure
    is recorded and surfaced rather than silently retried with no change.
    """
    last_err: Optional[str] = None
    for _retry in range(max_retries):
        try:
            result = metric.compute(X)
            if result.value is None or not np.isfinite(result.value):
                last_err = f"non-finite value: {result.value}"
                continue
            return result, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err


def pre_screen_quantum(ds: LoadedDataset, max_retries: int = 3) -> ScreeningResult:
    """Quantum-native screening (Class B): the 4 parallel v-components on `states`."""
    metrics = [
        ("entanglement_entropy_mean", EnsembleEntanglement()),
        ("total_correlation_mean", EnsembleTotalCorrelation()),
        ("stabilizer_renyi2_mean", EnsembleStabilizerMagic()),
        ("mps_bond_dim_mean", EnsembleMPSBond()),
    ]
    components: dict[str, ComponentResult] = {}
    failures: list[str] = []
    for label, metric in metrics:
        result, err = _try_compute(metric, ds.states, max_retries)
        if result is None:
            failures.append(label)
            components[label] = ComponentResult(None, None, 0.0, metric.estimator, {}, err)
        else:
            components[label] = ComponentResult(
                value=float(result.value),
                uncertainty=float(result.uncertainty) if result.uncertainty is not None else None,
                elapsed_s=float(result.elapsed_s),
                estimator=result.estimator,
                metadata_summary=_summarize_metadata(result.metadata),
                error=None,
            )
    return ScreeningResult(
        passed=(len(failures) == 0),
        components=components,
        failures=failures,
        not_implemented=[],
    )


def pre_screen(ds: LoadedDataset, max_retries: int = 3, ph_subsample: int = PH_SCREENING_SUBSAMPLE) -> ScreeningResult:
    """Run v(D) component metrics on the dataset, return per-component outcomes.

    Quantum-native datasets (Class B) branch to `pre_screen_quantum`. For
    classical datasets, persistent homology is run on a subsample (`ph_subsample`
    points) for screening speed; the final v_top values used in Phase B will be
    computed on the full dataset.
    """
    if ds.klass == "B" or ds.states is not None:
        return pre_screen_quantum(ds, max_retries=max_retries)

    metrics: list[tuple[str, Any, bool]] = [
        # (label, metric_instance, subsample_for_PH?)
        ("differential_entropy_knn_kl", EntropyKL(k=4), False),
        ("intrinsic_dim_pca", IntrinsicDimPCA(), False),
        ("intrinsic_dim_twonn", IntrinsicDimTwoNN(), False),
        ("compression_ratio_gzip", CompressionRatio(codec="gzip"), False),
        (
            "topological_complexity_rips",
            PersistentHomology(
                max_dim=PH_SCREENING_MAX_DIM,
                max_edge_fraction=PH_SCREENING_MAX_EDGE_FRACTION,
            ),
            True,
        ),
        ("second_order_mi", PairwiseMutualInformation(), False),
        ("higher_order_mi", HigherOrderMI(), False),
        ("kernel_effective_rank", KernelEffectiveRankRBF(), False),
        ("mdl_autoencoder", MDLAutoencoder(), False),
    ]

    components: dict[str, ComponentResult] = {}
    failures: list[str] = []

    for label, metric, do_subsample in metrics:
        X_in = ds.X
        if do_subsample and X_in.shape[0] > ph_subsample:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_in.shape[0], ph_subsample, replace=False)
            X_in = X_in[idx]

        result, err = _try_compute(metric, X_in, max_retries)
        if result is None:
            failures.append(label)
            components[label] = ComponentResult(
                value=None,
                uncertainty=None,
                elapsed_s=0.0,
                estimator=metric.estimator,
                metadata_summary={},
                error=err,
            )
        else:
            components[label] = ComponentResult(
                value=float(result.value),
                uncertainty=float(result.uncertainty) if result.uncertainty is not None else None,
                elapsed_s=float(result.elapsed_s),
                estimator=result.estimator,
                metadata_summary=_summarize_metadata(result.metadata),
                error=None,
            )

    return ScreeningResult(
        passed=(len(failures) == 0),
        components=components,
        failures=failures,
        not_implemented=list(NOT_IMPLEMENTED_COMPONENTS),
    )


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------

@dataclass
class RegistryEntry:
    name: str
    klass: str
    description: str
    n: int
    d: int
    sha256: str
    loader: dict[str, Any]
    screening: ScreeningResult


def build_entry(spec: LoaderSpec, max_retries: int = 3, verbose: bool = True) -> RegistryEntry:
    """Load + hash + pre-screen a single dataset spec."""
    if verbose:
        print(f"  [{spec.name}] loading…", flush=True)
    ds = spec.load()
    sha = hash_dataset(ds)
    if verbose:
        print(f"  [{spec.name}] hashed ({sha[:12]}…); pre-screening…", flush=True)
    screening = pre_screen(ds, max_retries=max_retries)
    if verbose:
        status = "PASS" if screening.passed else f"FAIL ({', '.join(screening.failures)})"
        print(f"  [{spec.name}] {status}", flush=True)
    if ds.X is not None:
        n, d = int(ds.X.shape[0]), int(ds.X.shape[1])
    else:  # quantum-native: n samples, d = qubit count (Hilbert dim = 2^d)
        n, d = int(ds.states.shape[0]), int(ds.n_qubits)
    return RegistryEntry(
        name=spec.name,
        klass=spec.klass,
        description=ds.description,
        n=n,
        d=d,
        sha256=sha,
        loader={
            "module": spec.module,
            "function": spec.function,
            "args": dict(spec.args),
        },
        screening=screening,
    )


def build_registry(
    specs: Optional[list[LoaderSpec]] = None,
    max_retries: int = 3,
    verbose: bool = True,
) -> list[RegistryEntry]:
    """Build registry entries for the given specs (default: all Phase B specs)."""
    if specs is None:
        specs = all_phase_b_specs()
    if not specs:
        return []
    if verbose:
        print(f"Building registry: {len(specs)} dataset specs", flush=True)
    return [build_entry(s, max_retries=max_retries, verbose=verbose) for s in specs]


def write_registry(
    entries: list[RegistryEntry],
    path: str,
    version: str = "0.1.0-class-a",
) -> dict[str, Any]:
    """Serialize registry entries to JSON. Returns the payload dict for inspection."""
    payload = {
        "version": version,
        "frozen_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "noise_model": {"eps_2": 1e-3, "eps_1": 1e-4},
        "phase_b_qubit_count": 8,
        "screening": {
            "implemented_components": [
                "differential_entropy_knn_kl",
                "intrinsic_dim_pca",
                "intrinsic_dim_twonn",
                "compression_ratio_gzip",
                "topological_complexity_rips",
                "second_order_mi",
                "higher_order_mi",
                "kernel_effective_rank",
                "mdl_autoencoder",
            ],
            "quantum_native_components": list(QN_IMPLEMENTED_COMPONENTS),
            "not_implemented_components": list(NOT_IMPLEMENTED_COMPONENTS),
            "ph_screening_subsample": PH_SCREENING_SUBSAMPLE,
            "ph_screening_max_dim": PH_SCREENING_MAX_DIM,
            "ph_screening_max_edge_fraction": PH_SCREENING_MAX_EDGE_FRACTION,
            "ph_screening_note": (
                "Pre-screening uses tightened Rips parameters for tractability; "
                "final Phase B v_top0/v_top1/v_top2 are computed at full n with default params."
            ),
        },
        "n_datasets": len(entries),
        "n_passed": sum(1 for e in entries if e.screening.passed),
        "datasets": [asdict(e) for e in entries],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
