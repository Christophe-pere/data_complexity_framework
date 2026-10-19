"""Dataset library — Phase B benchmark suite (Classes A/B/C/D).

Class A loaders are deterministic functions of `seed` — calling the same loader
twice with identical args produces byte-identical (X, y). This is what makes
SHA-256 hashing reproducible.

All four classes are populated (32 datasets, within the OSF §4.1 28–34 range):
A (synthetic, 13), B (quantum-native, 5), C (real, 10), D (adversarial, 4).
Class C combines an offline sklearn tier with a version-pinned OpenML network
tier (`real_openml`); the version pin + seeded subsample keep the SHA-256 over
(X, y) reproducible despite the network fetch (OpenML caches locally).

Per OSF pre-reg: each dataset is sized at n ≤ 2000 samples (reduced from the
original 5000) and qubit count is fixed at n_qubits = 8 for Phase B. Feature
dimension d may differ from n_qubits — handled by encoding library §1.3
(PCA reduction or zero-padding).

Per phase_a_specification.md §2.2, Class A targets 12–15 datasets. This module
provides 13: parity-k × 4, manifolds × 4, spin-glass × 2, GMM × 1, PRNG × 1,
concentric rings × 1.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from math import pi
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Dataset payload
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedDataset:
    """A loaded dataset ready for hashing, pre-screening, or training.

    Classical datasets (Classes A, C, D): `X` is shape (n, d) float64; `y` is
    shape (n,) int64 or None for unsupervised.

    Quantum-native datasets (Class B): `states` is a complex array of shape
    (n, 2**n_qubits) holding one pure state per sample, `n_qubits` is set, and
    `X` may be None (the data *is* the ensemble of states). The quantum-native
    v-components (v_ent, v_mi, v_mag, v_bond) are computed on `states`; the
    classical v(D) pipeline is skipped (OSF §5.2, parallel track).

    `metadata` carries loader-specific extras (couplings, sweep point, etc.).
    """

    name: str
    klass: str  # "A", "B", "C", "D"
    description: str
    X: Optional[np.ndarray]
    y: Optional[np.ndarray]
    states: Optional[np.ndarray] = None       # complex (n, 2**n_qubits), Class B
    n_qubits: Optional[int] = None            # set when `states` is present
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.X is None and self.states is None:
            raise ValueError("dataset must carry either X or states")
        if self.X is not None and self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {self.X.shape}")
        n = None if self.X is None else self.X.shape[0]
        if self.states is not None:
            if self.states.ndim != 2:
                raise ValueError(f"states must be 2D (n, 2^q), got {self.states.shape}")
            if self.n_qubits is None or self.states.shape[1] != 2 ** self.n_qubits:
                raise ValueError("states width must equal 2**n_qubits; set n_qubits")
            n = self.states.shape[0] if n is None else n
            if self.states.shape[0] != n:
                raise ValueError(f"X/states length mismatch: {n} vs {self.states.shape[0]}")
        if self.y is not None and n is not None and self.y.shape[0] != n:
            raise ValueError(f"y length mismatch: {self.y.shape[0]} vs {n}")


# ---------------------------------------------------------------------------
# Loader specification (for the registry)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoaderSpec:
    """A reproducible reference to a dataset loader.

    The registry stores LoaderSpecs; `load()` reconstitutes the LoadedDataset.
    For Phase B, the spec is the contract: any future code that wants the same
    dataset must call this exact (module.function, args) pair.
    """

    name: str
    klass: str
    module: str
    function: str
    args: dict[str, Any]

    def load(self) -> LoadedDataset:
        mod = importlib.import_module(self.module)
        fn = getattr(mod, self.function)
        ds = fn(**self.args)
        # Override the dataset's name with the spec's name, since one loader
        # function may be reused across multiple specs (e.g. parity_k for k=2,3,4,5)
        return LoadedDataset(
            name=self.name,
            klass=self.klass,
            description=ds.description,
            X=ds.X,
            y=ds.y,
            states=ds.states,
            n_qubits=ds.n_qubits,
            metadata=ds.metadata,
        )


# ---------------------------------------------------------------------------
# Class A: synthetic dataset loaders
# ---------------------------------------------------------------------------

def parity_k(k: int, n: int = 2000, d: int = 8, noise: float = 0.01, seed: int = 0) -> LoadedDataset:
    """Parity-k task: y = product of first k features.

    Features are bits ∈ {-1, +1} with small Gaussian noise (default 0.01) to
    break ties in NN-based estimators. The first k features are predictive;
    the remaining d-k are pure noise.

    Used to test how QML scales with feature-order parity (k=2 is linear,
    k≥3 is genuinely high-order correlation).
    """
    if k > d:
        raise ValueError(f"k={k} exceeds d={d}")
    rng = np.random.default_rng(seed)
    bits = rng.choice([-1.0, 1.0], size=(n, d))
    X = bits + noise * rng.standard_normal((n, d))
    y = (np.prod(bits[:, :k], axis=1) > 0).astype(np.int64)
    return LoadedDataset(
        name=f"parity-{k}",
        klass="A",
        description=f"parity-{k} task on {d} bits + Gaussian noise (σ={noise}); first {k} features predictive",
        X=X,
        y=y,
        metadata={"k": k, "noise": noise, "task": "binary_classification"},
    )


def torus(n: int = 2000, R: float = 2.0, r: float = 0.7, seed: int = 0) -> LoadedDataset:
    """2-torus T² embedded in R³. Betti (Z): (1, 2, 1). Hemisphere label."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * pi, n)
    phi = rng.uniform(0, 2 * pi, n)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y_ = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    X = np.column_stack([x, y_, z])
    y = (z > 0).astype(np.int64)
    return LoadedDataset(
        name="torus",
        klass="A",
        description=f"2-torus T² in R³ (R={R}, r={r}); label = upper-vs-lower half",
        X=X,
        y=y,
        metadata={"R": R, "r": r, "task": "binary_classification", "manifold": "T2"},
    )


def klein_bottle(n: int = 2000, seed: int = 0) -> LoadedDataset:
    """Klein bottle K² in R⁴ via Lawson immersion (smooth, no self-intersection).

    Parameters: x = cos(u)cos(v), y = sin(u)cos(v), z = cos(2u)sin(v), w = sin(2u)sin(v).
    Non-orientable 2-manifold; Betti over Z/2: (1, 2, 1).
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * pi, n)
    v = rng.uniform(0, 2 * pi, n)
    X = np.column_stack([
        np.cos(u) * np.cos(v),
        np.sin(u) * np.cos(v),
        np.cos(2 * u) * np.sin(v),
        np.sin(2 * u) * np.sin(v),
    ])
    y = (np.cos(v) > 0).astype(np.int64)
    return LoadedDataset(
        name="klein-bottle",
        klass="A",
        description="Klein bottle K² in R⁴ (Lawson embedding); non-orientable 2-manifold",
        X=X,
        y=y,
        metadata={"task": "binary_classification", "manifold": "K2"},
    )


def two_sphere(n: int = 2000, seed: int = 0) -> LoadedDataset:
    """2-sphere S² in R³. Betti: (1, 0, 1). Hemisphere label."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = (X[:, 2] > 0).astype(np.int64)
    return LoadedDataset(
        name="two-sphere",
        klass="A",
        description="uniform points on S² ⊂ R³; label = upper-vs-lower hemisphere",
        X=X,
        y=y,
        metadata={"task": "binary_classification", "manifold": "S2"},
    )


def three_sphere_pc(n: int = 2000, seed: int = 0) -> LoadedDataset:
    """3-sphere S³ in R⁴ as a point cloud. Betti: (1, 0, 0, 1)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = (X[:, 3] > 0).astype(np.int64)
    return LoadedDataset(
        name="three-sphere-pc",
        klass="A",
        description="uniform points on S³ ⊂ R⁴; label = sign of last coordinate",
        X=X,
        y=y,
        metadata={"task": "binary_classification", "manifold": "S3"},
    )


def sk_spin_glass(n: int = 2000, n_spins: int = 8, seed: int = 0) -> LoadedDataset:
    """Sherrington-Kirkpatrick spin-glass: random configurations + energy-binned label.

    Couplings J_ij ~ N(0, 1/n_spins), all-to-all. Configurations are uniform
    samples (not Boltzmann); label is sign of (E - median E) — captures the
    glassy energy landscape's high-order correlations.
    """
    rng = np.random.default_rng(seed)
    J = rng.normal(0.0, 1.0 / np.sqrt(n_spins), size=(n_spins, n_spins))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    X = rng.choice([-1.0, 1.0], size=(n, n_spins))
    # Add small noise to break ties for NN estimators
    X_noisy = X + 0.01 * rng.standard_normal(X.shape)
    E = -0.5 * np.einsum("ij,bi,bj->b", J, X, X)
    y = (E < np.median(E)).astype(np.int64)
    return LoadedDataset(
        name="sk-spin-glass",
        klass="A",
        description=f"Sherrington-Kirkpatrick on {n_spins} spins; label = below-median energy",
        X=X_noisy,
        y=y,
        metadata={
            "n_spins": n_spins,
            "task": "binary_classification",
            "model": "SK",
            "couplings_seed": seed,
        },
    )


def ea_spin_glass(n: int = 2000, n_spins: int = 8, seed: int = 0) -> LoadedDataset:
    """Edwards-Anderson spin-glass on 1D chain (nearest-neighbor couplings)."""
    rng = np.random.default_rng(seed)
    J = rng.normal(0.0, 1.0, size=n_spins - 1)
    X = rng.choice([-1.0, 1.0], size=(n, n_spins))
    X_noisy = X + 0.01 * rng.standard_normal(X.shape)
    E = -np.sum(J * X[:, :-1] * X[:, 1:], axis=1)
    y = (E < np.median(E)).astype(np.int64)
    return LoadedDataset(
        name="ea-spin-glass",
        klass="A",
        description=f"Edwards-Anderson 1D chain on {n_spins} spins; label = below-median energy",
        X=X_noisy,
        y=y,
        metadata={
            "n_spins": n_spins,
            "task": "binary_classification",
            "model": "EA",
            "couplings_seed": seed,
        },
    )


def anisotropic_gmm(
    n: int = 2000,
    d: int = 8,
    k_components: int = 4,
    aniso_strength: float = 5.0,
    seed: int = 0,
) -> LoadedDataset:
    """Anisotropic Gaussian mixture: k components with stretched covariances.

    Each component has a random direction with high variance and the rest with
    low variance, producing varied effective intrinsic dimension across the
    mixture.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, size=(k_components, d))
    X_parts = []
    y_parts = []
    per_component = n // k_components
    for k in range(k_components):
        # Random rotation
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        # Anisotropic diagonal: one direction stretched, rest compressed
        scales = np.ones(d) * 0.3
        scales[0] = aniso_strength
        Z = rng.standard_normal((per_component, d)) * scales
        X_k = Z @ Q.T + centers[k]
        X_parts.append(X_k)
        y_parts.append(np.full(per_component, k, dtype=np.int64))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    # Shuffle
    perm = rng.permutation(len(X))
    return LoadedDataset(
        name="anisotropic-gmm",
        klass="A",
        description=f"anisotropic GMM with {k_components} components in R^{d}; aniso strength = {aniso_strength}",
        X=X[perm],
        y=y[perm],
        metadata={"k_components": k_components, "aniso_strength": aniso_strength, "task": "classification"},
    )


def prng_features(n: int = 2000, d: int = 8, seed: int = 0) -> LoadedDataset:
    """Pseudorandom features with random binary labels.

    Negative-control / incompressibility test: no learnable structure. v_K should
    saturate near 1.0 (gzip ratio ≈ 1), v_S should be near max for the support,
    and any classifier should achieve ≈ 50% accuracy.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    y = rng.choice([0, 1], size=n).astype(np.int64)
    return LoadedDataset(
        name="prng",
        klass="A",
        description="pseudorandom features + random binary labels (negative control)",
        X=X,
        y=y,
        metadata={"task": "binary_classification", "control": "incompressible"},
    )


def concentric_rings(
    n: int = 2000,
    n_rings: int = 3,
    noise: float = 0.05,
    d_ambient: int = 8,
    seed: int = 0,
) -> LoadedDataset:
    """Concentric rings in R²; padded/rotated into R^{d_ambient}.

    High topology (b_1 = n_rings), low entropy. Tests whether topological
    components of v(D) contribute beyond entropy/correlation.
    """
    rng = np.random.default_rng(seed)
    radii = np.linspace(1.0, 1.0 + 0.5 * n_rings, n_rings)
    ring_idx = rng.integers(0, n_rings, size=n)
    angles = rng.uniform(0, 2 * pi, n)
    r = radii[ring_idx] + noise * rng.standard_normal(n)
    X2 = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    # Pad to d_ambient and apply a random orthogonal embedding
    if d_ambient > 2:
        X = np.zeros((n, d_ambient))
        X[:, :2] = X2
        # Random rotation in R^{d_ambient}
        Q, _ = np.linalg.qr(rng.standard_normal((d_ambient, d_ambient)))
        X = X @ Q.T
    else:
        X = X2
    return LoadedDataset(
        name="concentric-rings",
        klass="A",
        description=f"{n_rings} concentric rings in R² embedded in R^{d_ambient}; high topology, low entropy",
        X=X,
        y=ring_idx.astype(np.int64),
        metadata={"n_rings": n_rings, "noise": noise, "d_ambient": d_ambient, "task": "classification"},
    )


# ---------------------------------------------------------------------------
# Phase B Class A registry
# ---------------------------------------------------------------------------

def class_a_specs() -> list[LoaderSpec]:
    """Phase B Class A (synthetic): 13 datasets covering corners of v(D) space."""
    base = "qmlc.datasets"
    return [
        # Parity tasks — high-order correlation (k=2 linear, k≥3 nonlinear).
        # Distinct seeds across k so X varies (else parity-{2..5} collapse to one
        # point in v-space, since the bit distribution is k-independent).
        LoaderSpec("synth-parity-2", "A", base, "parity_k", {"k": 2, "n": 2000, "d": 8, "seed": 2}),
        LoaderSpec("synth-parity-3", "A", base, "parity_k", {"k": 3, "n": 2000, "d": 8, "seed": 3}),
        LoaderSpec("synth-parity-4", "A", base, "parity_k", {"k": 4, "n": 2000, "d": 8, "seed": 4}),
        LoaderSpec("synth-parity-5", "A", base, "parity_k", {"k": 5, "n": 2000, "d": 8, "seed": 5}),
        # Manifolds — controlled topology
        LoaderSpec("synth-torus", "A", base, "torus", {"n": 2000, "seed": 0}),
        LoaderSpec("synth-klein-bottle", "A", base, "klein_bottle", {"n": 2000, "seed": 0}),
        LoaderSpec("synth-2sphere", "A", base, "two_sphere", {"n": 2000, "seed": 0}),
        LoaderSpec("synth-3sphere-pc", "A", base, "three_sphere_pc", {"n": 2000, "seed": 0}),
        # Spin-glass — glassy energy landscapes, high-order correlation
        LoaderSpec("synth-sk-spin-glass", "A", base, "sk_spin_glass", {"n": 2000, "n_spins": 8, "seed": 0}),
        LoaderSpec("synth-ea-spin-glass", "A", base, "ea_spin_glass", {"n": 2000, "n_spins": 8, "seed": 0}),
        # Anisotropic mixture — varied intrinsic dim
        LoaderSpec("synth-anisotropic-gmm", "A", base, "anisotropic_gmm",
                   {"n": 2000, "d": 8, "k_components": 4, "seed": 0}),
        # Pseudorandom — negative control
        LoaderSpec("synth-prng", "A", base, "prng_features", {"n": 2000, "d": 8, "seed": 0}),
        # Rings — high topology, low entropy
        LoaderSpec("synth-concentric-rings", "A", base, "concentric_rings",
                   {"n": 2000, "n_rings": 3, "d_ambient": 8, "seed": 0}),
    ]


# ---------------------------------------------------------------------------
# Class C: real, cross-domain dataset loaders
# ---------------------------------------------------------------------------

def _stratified_subsample(
    X: np.ndarray, y: np.ndarray, n_max: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced subsample to at most n_max rows (deterministic, seeded)."""
    if X.shape[0] <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    # Proportional allocation, at least 1 per class.
    alloc = np.maximum(1, np.floor(n_max * counts / counts.sum()).astype(int))
    idx_keep: list[int] = []
    for c, a in zip(classes, alloc):
        idx_c = np.where(y == c)[0]
        idx_keep.extend(rng.choice(idx_c, min(a, idx_c.size), replace=False).tolist())
    idx_keep = np.array(sorted(idx_keep))
    return X[idx_keep], y[idx_keep]


def real_sklearn(
    dataset: str,
    classes: Optional[list[int]] = None,
    n_max: int = 2000,
    seed: int = 0,
) -> LoadedDataset:
    """Real tabular/image datasets bundled with scikit-learn (offline, no network).

    `dataset` ∈ {"iris", "wine", "breast_cancer", "digits"}. Multiclass sources
    are reduced to a balanced binary task by selecting two classes (`classes`,
    remapped so classes[1] → 1); `breast_cancer` is natively binary. The binary
    reduction affects only the learning task / edge metric, not v(D) (computed on
    X). Subsampled class-balanced to ≤ n_max rows per OSF §4.2.
    """
    from sklearn.datasets import (
        load_breast_cancer,
        load_digits,
        load_iris,
        load_wine,
    )

    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    if dataset not in loaders:
        raise ValueError(f"unknown sklearn dataset {dataset!r}")
    bunch = loaders[dataset]()
    X = np.ascontiguousarray(bunch.data, dtype=np.float64)
    y = np.ascontiguousarray(bunch.target, dtype=np.int64)

    if classes is not None:
        if len(classes) != 2:
            raise ValueError("classes must select exactly two labels")
        mask = np.isin(y, classes)
        X, y = X[mask], y[mask]
        y = (y == classes[1]).astype(np.int64)
    else:
        uniq = np.unique(y)
        if uniq.size != 2:
            raise ValueError(f"{dataset} is not binary; pass `classes`")
        y = (y == uniq[1]).astype(np.int64)

    X, y = _stratified_subsample(X, y, n_max, seed)
    pos = int(y.sum())
    return LoadedDataset(
        name=f"real-{dataset}",
        klass="C",
        description=(
            f"sklearn-bundled {dataset}"
            + (f", classes {classes} (→ binary)" if classes else " (binary)")
            + f"; n={X.shape[0]}, d={X.shape[1]}, pos={pos}"
        ),
        X=X,
        y=y,
        metadata={
            "source": "sklearn",
            "dataset": dataset,
            "classes": classes,
            "task": "binary_classification",
            "domain": "image" if dataset == "digits" else "tabular",
        },
    )


def real_openml(
    name: str,
    version: int = 1,
    classes: Optional[list[Any]] = None,
    n_max: int = 2000,
    seed: int = 0,
    domain: str = "tabular",
) -> LoadedDataset:
    """Real cross-domain datasets from OpenML (network tier of Class C, OSF §4.1).

    Fetches by (name, version) — the version pin makes the source immutable, so
    the post-processing (binary reduction + seeded stratified subsample) is
    deterministic and the SHA-256 over (X, y) is reproducible across machines.
    OpenML caches the raw download under ~/scikit_learn_data.

    All chosen sources have numeric features; `X` is coerced to float64 and rows
    with any NaN are dropped (the Higgs subset carries missing values). Multiclass
    sources are reduced to a balanced binary task via `classes` (classes[1] → 1);
    natively-binary sources map sorted-unique labels → {0, 1}. As with
    `real_sklearn`, the binary reduction affects only the task / edge metric, not
    v(D) (computed on X). Subsampled class-balanced to ≤ n_max rows (OSF §4.2).
    """
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name, version=version, as_frame=False, parser="liac-arff")
    X = np.asarray(bunch.data, dtype=np.float64)
    y_raw = np.asarray(bunch.target)

    # Drop rows with missing feature values (Higgs).
    finite = np.isfinite(X).all(axis=1)
    X, y_raw = X[finite], y_raw[finite]

    if classes is not None:
        if len(classes) != 2:
            raise ValueError("classes must select exactly two labels")
        # Compare as strings — OpenML targets are arff strings even when numeric.
        y_str = y_raw.astype(str)
        c0, c1 = str(classes[0]), str(classes[1])
        mask = np.isin(y_str, [c0, c1])
        X, y_str = X[mask], y_str[mask]
        y = (y_str == c1).astype(np.int64)
    else:
        uniq = np.unique(y_raw.astype(str))
        if uniq.size != 2:
            raise ValueError(f"{name} is not binary (labels {uniq[:6]}); pass `classes`")
        y = (y_raw.astype(str) == uniq[1]).astype(np.int64)

    X, y = _stratified_subsample(X, y, n_max, seed)
    pos = int(y.sum())
    return LoadedDataset(
        name=f"real-{name}",
        klass="C",
        description=(
            f"OpenML {name} v{version}"
            + (f", classes {classes} (→ binary)" if classes else " (binary)")
            + f"; n={X.shape[0]}, d={X.shape[1]}, pos={pos}"
        ),
        X=X,
        y=y,
        metadata={
            "source": "openml",
            "dataset": name,
            "version": version,
            "classes": classes,
            "task": "binary_classification",
            "domain": domain,
        },
    )


# ---------------------------------------------------------------------------
# Class D: adversarial / negative-control loaders (OSF §4.1)
# ---------------------------------------------------------------------------

def adv_easy_looking_hard(
    n: int = 2000, d: int = 8, sep: float = 4.0, flip_frac: float = 0.3, seed: int = 0
) -> LoadedDataset:
    """"Easy-looking hard": linearly separable raw features, adversarial labels.

    Two well-separated Gaussian blobs along axis 0 — the marginal feature
    distribution looks cleanly linearly separable. But the highest-confidence
    `flip_frac` of points (those farthest from the margin) have their labels
    flipped, so the apparent linear boundary is maximally wrong and a linear
    model underperforms its visual promise. v(D) sees simple geometry; the task
    is hard.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    center = np.zeros(d)
    center[0] = sep
    X = np.vstack([
        rng.standard_normal((half, d)) - center,
        rng.standard_normal((n - half, d)) + center,
    ])
    y = np.array([0] * half + [1] * (n - half), dtype=np.int64)
    # Flip the most-confidently-separable points (largest |projection on axis 0|).
    order = np.argsort(-np.abs(X[:, 0]))
    n_flip = int(flip_frac * n)
    y[order[:n_flip]] ^= 1
    return LoadedDataset(
        name="adv-easy-looking-hard",
        klass="D",
        description=(
            f"linearly-separable blobs (sep={sep}) with adversarial label noise "
            f"(flip_frac={flip_frac}); easy geometry, hard task"
        ),
        X=X,
        y=y,
        metadata={"sep": sep, "flip_frac": flip_frac, "task": "binary_classification",
                  "adversarial_kind": "easy_looking_hard"},
    )


def adv_hard_looking_easy(
    n: int = 2000, n_rings: int = 4, d_ambient: int = 8, noise: float = 0.03, seed: int = 0
) -> LoadedDataset:
    """"Hard-looking easy": rich topology, trivially-separable label.

    Points on `n_rings` concentric circles (high persistent-homology signal),
    embedded in R^{d_ambient} via a random rotation. The label is a single
    linear cut (sign of the first plane coordinate) — trivially learnable
    despite the busy topology. v(D) sees high v_top; the task is linear.
    """
    rng = np.random.default_rng(seed)
    per = n // n_rings
    pts, lab = [], []
    for i in range(n_rings):
        m = per if i < n_rings - 1 else n - per * (n_rings - 1)
        theta = rng.uniform(0, 2 * pi, m)
        radius = 1.0 + i
        x0 = radius * np.cos(theta) + noise * rng.standard_normal(m)
        x1 = radius * np.sin(theta) + noise * rng.standard_normal(m)
        pts.append(np.column_stack([x0, x1]))
        lab.append((x0 > 0).astype(np.int64))
    P = np.vstack(pts)
    y = np.concatenate(lab)
    # Embed R^2 → R^{d_ambient} via a random orthonormal map.
    A = rng.standard_normal((d_ambient, 2))
    Q, _ = np.linalg.qr(A)
    X = P @ Q.T
    return LoadedDataset(
        name="adv-hard-looking-easy",
        klass="D",
        description=(
            f"{n_rings} concentric rings in R^{d_ambient} (high topology); "
            f"label = linear half-plane cut (trivially learnable)"
        ),
        X=X,
        y=y,
        metadata={"n_rings": n_rings, "task": "binary_classification",
                  "adversarial_kind": "hard_looking_easy"},
    )


def adv_encoded_random(source: str = "breast_cancer", n_max: int = 2000, seed: int = 0) -> LoadedDataset:
    """"Encoded random": a real dataset with its structure destroyed.

    Loads a real source dataset and permutes each feature column independently,
    breaking all feature–feature and feature–label dependence while preserving
    every marginal distribution. Negative control: should be unlearnable above
    chance, with v(D) reflecting only the (preserved) marginals.
    """
    base = real_sklearn(source, n_max=n_max, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = base.X.copy()
    for j in range(X.shape[1]):
        X[:, j] = X[rng.permutation(X.shape[0]), j]
    return LoadedDataset(
        name="adv-encoded-random",
        klass="D",
        description=(
            f"{source} with each feature column independently shuffled "
            f"(joint structure destroyed, marginals preserved); negative control"
        ),
        X=np.ascontiguousarray(X),
        y=base.y,
        metadata={"source": source, "task": "binary_classification",
                  "adversarial_kind": "encoded_random"},
    )


def adv_quantum_trivial(n: int = 2000, d_ambient: int = 8, seed: int = 0) -> LoadedDataset:
    """"Quantum-trivial": data effectively from a single qubit, embedded high.

    Samples Bloch vectors uniformly on S² (pure 1-qubit states), embeds the
    3-D Bloch coordinates into R^{d_ambient} via a random orthonormal map, and
    labels by hemisphere (sign of z = a single-qubit Z measurement). The data
    has only 1-qubit's worth of structure, so any quantum embedding into more
    qubits is wasted — quantum advantage should be absent.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0, 2 * pi, n)
    r = np.sqrt(np.maximum(1.0 - z ** 2, 0.0))
    bloch = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])  # on S²
    A = rng.standard_normal((d_ambient, 3))
    Q, _ = np.linalg.qr(A)
    X = bloch @ Q.T
    y = (z > 0).astype(np.int64)
    return LoadedDataset(
        name="adv-quantum-trivial",
        klass="D",
        description=(
            f"1-qubit Bloch-sphere states embedded in R^{d_ambient}; "
            f"label = hemisphere (single-qubit Z); quantum-trivial"
        ),
        X=X,
        y=y,
        metadata={"task": "binary_classification", "adversarial_kind": "quantum_trivial"},
    )


# ---------------------------------------------------------------------------
# Class B: quantum-native dataset loaders (OSF §4.1; states-ensemble schema)
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_PX = np.array([[0, 1], [1, 0]], dtype=complex)
_PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PZ = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _op_on(P: np.ndarray, q: int, n: int) -> np.ndarray:
    """Embed single-qubit operator P on qubit q (big-endian), identity elsewhere."""
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, P if i == q else _I2)
    return out


def _two_op(P: np.ndarray, Q: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
    """Embed P⊗Q on qubits (a, b), identity elsewhere."""
    out = np.array([[1.0 + 0j]])
    for i in range(n):
        out = np.kron(out, P if i == a else (Q if i == b else _I2))
    return out


def _ground_state(H: np.ndarray) -> np.ndarray:
    """Lowest-eigenvalue eigenvector, phase-canonicalized for reproducible hashing."""
    w, V = np.linalg.eigh(H)
    psi = V[:, 0]
    k = int(np.argmax(np.abs(psi)))
    return psi / (psi[k] / abs(psi[k]))


def _canonical_phase(psi: np.ndarray) -> np.ndarray:
    k = int(np.argmax(np.abs(psi)))
    return psi / (psi[k] / abs(psi[k]))


def _apply_1q(psi: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    t = psi.reshape([2] * n)
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
    return t.reshape(-1)


def _apply_cz(psi: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
    t = psi.reshape([2] * n).copy()
    sl = [slice(None)] * n
    sl[a] = 1
    sl[b] = 1
    t[tuple(sl)] *= -1.0
    return t.reshape(-1)


def _three_op(A, B, C, i, j, k, n):
    """A_i B_j C_k as a dense operator on n qubits."""
    ops = [np.eye(2)] * n
    ops[i], ops[j], ops[k] = A, B, C
    out = ops[0]
    for m in ops[1:]:
        out = np.kron(out, m)
    return out


def _pauli_word(letters, n):
    """Dense operator for a full-length Pauli word given as n letters."""
    M = {"I": np.eye(2), "X": _PX, "Y": _PY, "Z": _PZ}
    out = M[letters[0]]
    for L in letters[1:]:
        out = np.kron(out, M[L])
    return out


def tfim_ground_states(n_qubits: int = 8, n_states: int = 256, J: float = 1.0,
                       h_lo: float = 0.2, h_hi: float = 1.8, seed: int = 0) -> LoadedDataset:
    """TFIM ground states swept across the quantum phase transition (OSF §4.1).

    H(h) = -J Σ Z_i Z_{i+1} - h Σ X_i (open chain). Critical point h_c = J. The
    sweep of h ∈ [h_lo, h_hi] yields the state ensemble; label = paramagnetic
    (h > J) vs ferromagnetic (h < J). `seed` is unused (deterministic) but kept
    for signature uniformity.
    """
    n = n_qubits
    Hzz = sum(_two_op(_PZ, _PZ, i, i + 1, n) for i in range(n - 1))
    Hx = sum(_op_on(_PX, i, n) for i in range(n))
    hs = np.linspace(h_lo, h_hi, n_states)
    states = np.array([_ground_state(-J * Hzz - h * Hx) for h in hs])
    y = (hs > J).astype(np.int64)
    return LoadedDataset(
        name="qn-tfim", klass="B",
        description=f"TFIM ground states (n={n}, J={J}) swept h∈[{h_lo},{h_hi}]; label=para vs ferro",
        X=None, y=y, states=states, n_qubits=n,
        metadata={"model": "TFIM", "J": J, "h_range": [h_lo, h_hi], "h_c": J,
                  "task": "phase_classification"},
    )


def cluster_spt_ground_states(n_qubits: int = 8, n_states: int = 256, h: float = 0.2,
                             J: float = 1.0, seed: int = 0) -> LoadedDataset:
    """Cluster-SPT ground states, labelled by a NON-LOCAL string order parameter.

    H(h) = -J sum_{i=1}^{n-2} Z_{i-1} X_i Z_{i+1} - h sum_i X_i   (open chain),
    the cluster-Ising model. For h < J this is the symmetry-protected topological
    (cluster) phase; the transition is at h/J = 1. We work at h well inside the SPT
    phase.

    **Why this family matters, and how it differs from every other probe here.**
    All other probes build states by Haar-sampling inside an eigenspace -- the states
    are constructed. These are genuine ground states of a physical Hamiltonian,
    obtained by exact diagonalization. Only the *labelling* uses the structure.

    The open-chain cluster model has n-2 stabilizers on n qubits, so its ground space
    is 4-fold degenerate: the protected edge modes of the SPT phase. Which ground
    state you occupy is encoded in the **string order parameter**

        S = Z_0 X_1 X_2 ... X_{n-2} Z_{n-1},

    a weight-n observable and the textbook diagnostic of cluster-SPT order. The label
    is its eigenvalue. This is shadow-hard *because of the physics* -- SPT order is
    non-local by definition -- rather than because we planted a high-weight Pauli.

    Note the label is deliberately NOT "which phase". Phase would be shadow-easy:
    <X_i> alone separates the paramagnet from the SPT phase, so a weight-1 observable
    would solve it (the same reason the TFIM phase task is trivially classical here).
    """
    import numpy as _np
    n = n_qubits
    Hc = sum(_three_op(_PZ, _PX, _PZ, i - 1, i, i + 1, n) for i in range(1, n - 1))
    Hx = sum(_op_on(_PX, i, n) for i in range(n))
    H = -J * Hc - h * Hx

    # ground space (4-fold degenerate deep in the SPT phase)
    w_, V = _np.linalg.eigh(H if isinstance(H, _np.ndarray) else H.toarray())
    k = 4
    G = V[:, :k]                                   # (2**n, k) orthonormal ground space

    # LABEL = the Z2 symmetry sector, not the string order parameter.
    #
    # The string operator (a product of stabilizers) is +1 on the ENTIRE ground space,
    # so it labels nothing -- verified: its eigenvalues there are all ~+0.96. What does
    # split the 4-fold ground space is the Z2 symmetry that PROTECTS the SPT phase,
    # generated by X on the even sublattice. Its eigenvalues on the ground space are
    # exactly +-1.
    #
    # This is shadow-hard for the right reason. The symmetry generator has weight
    # ceil(n/2) > k = 2. The weight-1 edge logicals (Z_0, Z_{n-1}) also split the
    # space, but they ANTICOMMUTE with the symmetry, so <Z_0> = 0 within each sector
    # -- verified numerically. No low-weight observable reveals the label.
    sym_letters = ["X" if q % 2 == 0 else "I" for q in range(n)]
    S = _pauli_word(tuple(sym_letters), n)
    string_letters = sym_letters

    # split the ground space into the two symmetry sectors
    Sg = G.conj().T @ (S @ G)
    ev, U = _np.linalg.eigh((Sg + Sg.conj().T) / 2.0)
    plus = G @ U[:, ev > 0]
    minus = G @ U[:, ev < 0]

    rng = _np.random.default_rng(seed)
    half = n_states // 2
    def sample(basis, m):
        if basis.shape[1] == 0:
            return _np.zeros((0, 2 ** n), complex)
        c = rng.normal(size=(m, basis.shape[1])) + 1j * rng.normal(size=(m, basis.shape[1]))
        V_ = c @ basis.T
        return V_ / _np.linalg.norm(V_, axis=1, keepdims=True)

    Vp, Vm = sample(plus, half), sample(minus, n_states - half)
    states = _np.vstack([Vp, Vm])
    y = _np.r_[_np.zeros(len(Vp), int), _np.ones(len(Vm), int)]
    idx = rng.permutation(len(states))
    return LoadedDataset(
        name="qn-cluster-spt", klass="B",
        description=(f"cluster-SPT ground states (n={n}, h={h}, J={J}); label = string "
                     f"Z2 symmetry sector (weight-{(n + 1) // 2} observable)"),
        X=None, y=y[idx].astype(_np.int64), states=states[idx], n_qubits=n,
        metadata={"model": "cluster-Ising", "J": J, "h": h, "h_c": J,
                  "task": "z2_symmetry_sector",
                  "paulis": string_letters,
                  "support": [q for q, L in enumerate(string_letters) if L != "I"],
                  "physical_states": True, "exploratory": True},
    )


def xxz_ground_states(n_qubits: int = 8, n_states: int = 256,
                      d_lo: float = 0.2, d_hi: float = 1.8, seed: int = 0) -> LoadedDataset:
    """XXZ ground states swept across the Δ=1 transition (OSF §4.1).

    H(Δ) = Σ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1}) (open chain). Label =
    gapped antiferro (Δ > 1) vs gapless XY (Δ < 1).
    """
    n = n_qubits
    Hxy = sum(_two_op(_PX, _PX, i, i + 1, n) + _two_op(_PY, _PY, i, i + 1, n) for i in range(n - 1))
    Hzz = sum(_two_op(_PZ, _PZ, i, i + 1, n) for i in range(n - 1))
    ds = np.linspace(d_lo, d_hi, n_states)
    states = np.array([_ground_state(Hxy + d * Hzz) for d in ds])
    y = (ds > 1.0).astype(np.int64)
    return LoadedDataset(
        name="qn-xxz", klass="B",
        description=f"XXZ ground states (n={n}) swept Δ∈[{d_lo},{d_hi}]; label=antiferro vs XY",
        X=None, y=y, states=states, n_qubits=n,
        metadata={"model": "XXZ", "delta_range": [d_lo, d_hi], "delta_c": 1.0,
                  "task": "phase_classification"},
    )


def haar_states(n_qubits: int = 8, n_states: int = 256, seed: int = 0) -> LoadedDataset:
    """Haar-random pure states — negative control (no learnable structure).

    Labels are random (independent of the states); a classifier should not beat
    chance. The OSF Haar control for the quantum-native track.
    """
    n = n_qubits
    rng = np.random.default_rng(seed)
    d = 2 ** n
    states = []
    for _ in range(n_states):
        z = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        states.append(_canonical_phase(z / np.linalg.norm(z)))
    y = rng.integers(0, 2, n_states).astype(np.int64)
    return LoadedDataset(
        name="qn-haar", klass="B",
        description=f"Haar-random pure states (n={n}); random labels; negative control",
        X=None, y=y, states=np.array(states), n_qubits=n,
        metadata={"model": "Haar", "task": "negative_control"},
    )


def stabilizer_magic_states(n_qubits: int = 8, n_states: int = 256, n_t_max: int = 6,
                            seed: int = 0) -> LoadedDataset:
    """Cluster (stabilizer) states with controlled non-Clifford perturbations.

    Base = a 1-D cluster state (|+⟩^n + nearest-neighbour CZ — Clifford, magic 0,
    entangled). Each sample then receives k RZ(π/4) (T) gates on random qubits,
    k ∈ [0, n_t_max]; magic grows with k. Label = high-magic (k > n_t_max/2) vs
    low-magic. Tests whether the framework's v_mag tracks a magic-defined task.
    """
    n = n_qubits
    rng = np.random.default_rng(seed)
    Tgate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    # Cluster-state base, built once.
    base = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)  # |+>^n = H^n|0>
    for i in range(n - 1):
        base = _apply_cz(base, i, i + 1, n)
    states, labels = [], []
    for _ in range(n_states):
        k = int(rng.integers(0, n_t_max + 1))
        psi = base.copy()
        for q in rng.choice(n, size=k, replace=False) if k > 0 else []:
            psi = _apply_1q(psi, Tgate, int(q), n)
        states.append(_canonical_phase(psi))
        labels.append(1 if k > n_t_max / 2 else 0)
    return LoadedDataset(
        name="qn-stabilizer-magic", klass="B",
        description=f"cluster states (n={n}) + k≤{n_t_max} T-gate perturbations; label=high vs low magic",
        X=None, y=np.array(labels, dtype=np.int64), states=np.array(states), n_qubits=n,
        metadata={"model": "cluster+T", "n_t_max": n_t_max, "task": "magic_classification"},
    )


def _mps_to_statevector(tensors: list[np.ndarray]) -> np.ndarray:
    """Contract a left-to-right MPS (each tensor (χ_l, 2, χ_r), boundary χ=1)
    into a dense statevector of length 2**n."""
    psi = tensors[0]  # (1, 2, χ)
    for A in tensors[1:]:
        # psi: (1, 2^k, χ_l) ; A: (χ_l, 2, χ_r) → (1, 2^{k+1}, χ_r)
        psi = np.einsum("lpc,cqr->lpqr", psi, A)
        psi = psi.reshape(1, -1, A.shape[2])
    return psi.reshape(-1)


def random_mps_states(n_qubits: int = 8, n_states: int = 256, chi_max: int = 8,
                      chi_lo: int = 1, seed: int = 0) -> LoadedDataset:
    """Random matrix-product states with controlled bond dimension (OSF §4.1).

    Each sample is a random MPS whose maximum bond dimension χ is drawn uniformly
    in [chi_lo, chi_max]; tensor entries are complex Gaussian and the contracted
    statevector is normalised. χ caps the Schmidt rank across every cut, so the
    ensemble sweeps entanglement/compressibility between the near-product regime
    (χ=1, like TFIM deep in a phase) and near-Haar (χ→2^{n/2}). Label = high-χ
    (χ > (chi_lo+chi_max)/2) vs low-χ. This is the quantum-native probe for v_bond
    (MPS bond dimension, OSF §5.2): it tests whether the framework's compressibility
    component tracks a bond-dimension-defined task, filling the gap between the
    low-entanglement (TFIM/XXZ) and maximal (Haar) Class B datasets.
    """
    n = n_qubits
    rng = np.random.default_rng(seed)
    states, labels = [], []
    chi_mid = (chi_lo + chi_max) / 2.0
    for _ in range(n_states):
        chi = int(rng.integers(chi_lo, chi_max + 1))
        tensors = []
        left = 1
        for i in range(n):
            # Bond is capped by χ and by the exact Schmidt bound 2^min(i+1, n-i-1).
            right = 1 if i == n - 1 else min(chi, 2 ** (i + 1), 2 ** (n - i - 1))
            A = (rng.standard_normal((left, 2, right))
                 + 1j * rng.standard_normal((left, 2, right)))
            tensors.append(A)
            left = right
        psi = _mps_to_statevector(tensors)
        states.append(_canonical_phase(psi / np.linalg.norm(psi)))
        labels.append(1 if chi > chi_mid else 0)
    return LoadedDataset(
        name="qn-random-mps", klass="B",
        description=f"random MPS (n={n}) with bond dim χ∈[{chi_lo},{chi_max}]; label=high vs low χ",
        X=None, y=np.array(labels, dtype=np.int64), states=np.array(states), n_qubits=n,
        metadata={"model": "random-MPS", "chi_range": [chi_lo, chi_max],
                  "task": "bond_dimension_classification"},
    )


def parity_probe_states(n_qubits: int = 8, weight: int = 4, n_states: int = 256,
                        seed: int = 0) -> LoadedDataset:
    """Shadow-hard positive-control probe (docs/class_b_quantum_native_spec.md §5.2). EXPLORATORY.

    The label is a weight-`w` global parity: `y = ½(1 − ⟨P_w⟩)` where `P_w = ∏_{j<w} Z_j`.
    Each sample is a Haar-random state *within* the (−1)^y eigenspace of `P_w` — i.e. random
    complex amplitudes over the 2^{n−1} bitstrings whose first-`w`-qubit parity equals `y`, zero
    elsewhere, normalized. By Haar symmetry within the eigenspace, the expectation of every Pauli
    other than `P_w` is 0 for *both* labels, so **no Pauli of weight < w carries any label signal**.

    Consequence — the quantum-native analogue of the parity-k family:
      * A classical learner on k-local shadow features (k ≤ 2) can read the label only if w ≤ 2;
        for w ≥ 3 it is at chance (P_w is invisible below weight w, and estimating P_w itself from
        random-Pauli shadows costs ~3^w shots — shadow-hard).
      * A quantum circuit maps P_w (a Z-string) to a single-qubit readout via a depth-~w CNOT
        ladder, so a sufficiently expressive ansatz learns it from O(1) measurements.

    `weight` is the ground-truth complexity knob; the edge should turn on as `w` crosses the
    classical feature locality k. This is the baseline-robust positive control the classical-data
    suite lacks. Balanced labels; phase-canonicalized for reproducible hashing.
    """
    n = n_qubits
    w = int(weight)
    if not 1 <= w <= n:
        raise ValueError(f"weight must be in [1, {n}], got {w}")
    rng = np.random.default_rng(seed)
    idx = np.arange(2 ** n)
    # first-w-qubit parity of each computational basis index (qubit q bit = (idx>>(n-1-q))&1)
    parity = np.zeros(2 ** n, dtype=np.int64)
    for q in range(w):
        parity ^= (idx >> (n - 1 - q)) & 1

    states, labels = [], []
    for s in range(n_states):
        b = s % 2  # balanced
        mask = parity == b
        amp = np.zeros(2 ** n, dtype=complex)
        z = rng.standard_normal(int(mask.sum())) + 1j * rng.standard_normal(int(mask.sum()))
        amp[mask] = z
        amp /= np.linalg.norm(amp)
        states.append(_canonical_phase(amp))
        labels.append(b)
    return LoadedDataset(
        name=f"qn-parity-w{w}", klass="B",
        description=(f"shadow-hard parity probe (n={n}, weight={w}): Haar-random states in the "
                     f"±1 eigenspace of P_w=∏_{{j<{w}}}Z_j; label=parity sign; k<{w} features null"),
        X=None, y=np.array(labels, dtype=np.int64), states=np.array(states), n_qubits=n,
        metadata={"model": "parity-probe", "weight": w, "P_w": "Z-string on qubits 0..w-1",
                  "task": "shadow_hard_parity", "knob": "weight"},
    )


def qspin_states(system: str = "Ising", periodicity: str = "open", lattice: str = "chain",
                 layout: str = "1x8", seed: int = 0) -> LoadedDataset:
    """PennyLane `qspin` ground states across a phase diagram → phase-classification (Class B).

    Loads the ground-state ensemble of a spin model swept across its coupling; the label is the
    quantum phase, taken as `order_param > median` (balanced, and the order parameter is the
    physical phase discriminant). EXPLORATORY / Part III C1 — not in the frozen 32.
    `layout='1x8'` ⇒ 8 qubits, matching Phase B. Downloads on first use (PennyLane data cache).
    """
    import pennylane as qml
    (d,) = qml.data.load("qspin", sysname=system, periodicity=periodicity,
                         lattice=lattice, layout=layout)
    states = np.asarray(d.ground_states, dtype=complex)          # (n_states, 2**q)
    op = np.real(np.asarray(d.order_params)).ravel()[: len(states)]
    y = (op > np.median(op)).astype(np.int64)
    states = np.array([_canonical_phase(s / np.linalg.norm(s)) for s in states])
    n = int(round(np.log2(states.shape[1])))
    return LoadedDataset(
        name=f"qn-qspin-{system.lower()}", klass="B",
        description=(f"PennyLane qspin {system} ({periodicity} {lattice} {layout}) ground states; "
                     f"label = phase (order_param>median); n={len(states)}"),
        X=None, y=y, states=states, n_qubits=n,
        metadata={"model": system, "source": "pennylane.qspin", "layout": layout,
                  "task": "phase_classification", "num_phases": int(np.asarray(d.num_phases))},
    )


def class_b_qspin_specs(systems: Sequence[str] = ("Ising", "Heisenberg"),
                        layout: str = "1x8") -> list[LoaderSpec]:
    """EXPLORATORY PennyLane qspin phase-classification specs (Part III C1; not in frozen 32)."""
    base = "qmlc.datasets"
    return [LoaderSpec(f"qn-qspin-{s.lower()}", "B", base, "qspin_states",
                       {"system": s, "layout": layout, "seed": 0}) for s in systems]


def structure_probe_states(family: str = "z_parity", n_qubits: int = 8, weight: int = 4,
                          n_states: int = 256, seed: int = 0) -> LoadedDataset:
    """Structure-search battery probe as a Class B dataset. EXPLORATORY.

    Wraps `qmlc.structure_search.make_probe` so the battery families can be driven
    through the existing Class B sweep (gradient models, shadow baselines).
    `metadata` carries the ground-truth structure so the *designed* readout can be
    built without the learner ever seeing it.

    Families: `z_parity` (Pauli-Z string), `pauli_stab` (arbitrary Pauli string --
    tests basis dependence), `double_parity` (label = AND of two disjoint parities,
    which no single Pauli expresses), `null_control` (no structure; everything must fail).
    """
    from qmlc.structure_search import make_probe
    pr = make_probe(family, n_qubits, weight, n_states, seed)
    return LoadedDataset(
        name=f"sb-{family}-w{weight}", klass="B",
        description=(f"structure-search battery probe: {family}, weight {weight}, "
                     f"{n_qubits} qubits (EXPLORATORY)"),
        X=None, y=np.asarray(pr.y, dtype=np.int64), states=pr.states, n_qubits=n_qubits,
        metadata={"family": family, "weight": weight, "support": list(pr.support),
                  "paulis": list(pr.paulis), "parts": [list(x) for x in pr.parts],
                  "n_states": n_states, "seed": seed, "exploratory": True},
    )


def structure_probe_specs(families: Sequence[str] = ("z_parity", "pauli_stab",
                                                     "double_parity", "null_control"),
                          weights: Sequence[int] = (2, 4, 6),
                          n_qubits: int = 8, n_states: int = 256) -> list[LoaderSpec]:
    """EXPLORATORY specs for the structure-search battery (not in the frozen registry)."""
    base = "qmlc.datasets"
    return [
        LoaderSpec(f"sb-{fam}-w{w}", "B", base, "structure_probe_states",
                   {"family": fam, "n_qubits": n_qubits, "weight": w, "n_states": n_states,
                    "seed": 0})
        for fam in families for w in weights
    ]


def class_b_probe_specs(weights: Sequence[int] = (1, 2, 4, 6, 8),
                        n_qubits: int = 8, n_states: int = 256) -> list[LoaderSpec]:
    """EXPLORATORY shadow-hard probe specs (not in the frozen registry / Part II suite).

    One dataset per parity weight `w` — the complexity knob. Part III / positive-control use.
    """
    base = "qmlc.datasets"
    return [
        LoaderSpec(f"qn-parity-w{w}", "B", base, "parity_probe_states",
                   {"n_qubits": n_qubits, "weight": w, "n_states": n_states, "seed": 0})
        for w in weights
    ]


def class_b_specs() -> list[LoaderSpec]:
    """Phase B Class B (quantum-native). OSF §4.1 target: 5–6 datasets.

    Four families implemented (TFIM, XXZ, Haar control, stabilizer+magic) under
    the states-ensemble schema; the quantum-native v-components v_ent/v_mi/v_mag/
    v_bond are computed on `states` (qmlc.quantum). n_qubits=8 to match Phase B.
    """
    base = "qmlc.datasets"
    return [
        LoaderSpec("qn-tfim", "B", base, "tfim_ground_states",
                   {"n_qubits": 8, "n_states": 256, "seed": 0}),
        LoaderSpec("qn-xxz", "B", base, "xxz_ground_states",
                   {"n_qubits": 8, "n_states": 256, "seed": 0}),
        LoaderSpec("qn-haar", "B", base, "haar_states",
                   {"n_qubits": 8, "n_states": 256, "seed": 0}),
        LoaderSpec("qn-stabilizer-magic", "B", base, "stabilizer_magic_states",
                   {"n_qubits": 8, "n_states": 256, "n_t_max": 6, "seed": 0}),
        # Random MPS bond-dimension sweep — quantum-native v_bond probe, bridges
        # the low-entanglement (TFIM/XXZ) and maximal (Haar) ends of the ensemble.
        LoaderSpec("qn-random-mps", "B", base, "random_mps_states",
                   {"n_qubits": 8, "n_states": 256, "chi_max": 8, "chi_lo": 1, "seed": 0}),
    ]


def class_c_specs() -> list[LoaderSpec]:
    """Phase B Class C (real, cross-domain). OSF §4.1 target: 12–14 datasets.

    Offline tier (sklearn-bundled): UCI Iris/Wine/Breast-Cancer + digits (image).
    Network tier (UCI Sonar/Ionosphere, MNIST/Fashion-MNIST, HEP, bio, finance)
    is added separately once an openml/download path is wired.
    """
    base = "qmlc.datasets"
    return [
        # UCI tabular — Iris (1 vs 2: the non-linearly-separable pair), Wine (0 vs 1),
        # Breast Cancer (native binary).
        LoaderSpec("real-iris", "C", base, "real_sklearn",
                   {"dataset": "iris", "classes": [1, 2], "seed": 0}),
        LoaderSpec("real-wine", "C", base, "real_sklearn",
                   {"dataset": "wine", "classes": [0, 1], "seed": 0}),
        LoaderSpec("real-breast-cancer", "C", base, "real_sklearn",
                   {"dataset": "breast_cancer", "seed": 0}),
        # Image — digits, classic confusable pair 3 vs 5 (offline MNIST analog).
        LoaderSpec("real-digits-3v5", "C", base, "real_sklearn",
                   {"dataset": "digits", "classes": [3, 5], "seed": 0}),
        # Network tier (OpenML, version-pinned) — cross-domain coverage:
        # physics/sonar, radar, medical, HEP, finance/credit, image.
        LoaderSpec("real-sonar", "C", base, "real_openml",
                   {"name": "sonar", "version": 1, "domain": "physics", "seed": 0}),
        LoaderSpec("real-ionosphere", "C", base, "real_openml",
                   {"name": "ionosphere", "version": 1, "domain": "radar", "seed": 0}),
        LoaderSpec("real-diabetes", "C", base, "real_openml",
                   {"name": "diabetes", "version": 1, "domain": "medical", "seed": 0}),
        LoaderSpec("real-higgs", "C", base, "real_openml",
                   {"name": "higgs", "version": 1, "domain": "hep", "seed": 0}),
        LoaderSpec("real-credit-default", "C", base, "real_openml",
                   {"name": "default-of-credit-card-clients", "version": 1,
                    "domain": "finance", "seed": 0}),
        # Image — Fashion-MNIST confusable pair (pullover=2 vs shirt=6), 784-dim.
        LoaderSpec("real-fashion-2v6", "C", base, "real_openml",
                   {"name": "Fashion-MNIST", "version": 1, "classes": [2, 6],
                    "domain": "image", "seed": 0}),
    ]


def class_d_specs() -> list[LoaderSpec]:
    """Phase B Class D (adversarial / edge cases). OSF §4.1 target: 3–4 datasets.

    All four constructions are synthetic and built offline from the existing
    (X, y) schema + 9 v(D) components.
    """
    base = "qmlc.datasets"
    return [
        LoaderSpec("adv-easy-looking-hard", "D", base, "adv_easy_looking_hard",
                   {"n": 2000, "d": 8, "seed": 0}),
        LoaderSpec("adv-hard-looking-easy", "D", base, "adv_hard_looking_easy",
                   {"n": 2000, "n_rings": 4, "d_ambient": 8, "seed": 0}),
        LoaderSpec("adv-encoded-random", "D", base, "adv_encoded_random",
                   {"source": "breast_cancer", "seed": 0}),
        LoaderSpec("adv-quantum-trivial", "D", base, "adv_quantum_trivial",
                   {"n": 2000, "d_ambient": 8, "seed": 0}),
    ]


def all_phase_b_specs() -> list[LoaderSpec]:
    """All 32 Phase B datasets across the four classes (A:13, B:5, C:10, D:4)."""
    return [*class_a_specs(), *class_b_specs(), *class_c_specs(), *class_d_specs()]
