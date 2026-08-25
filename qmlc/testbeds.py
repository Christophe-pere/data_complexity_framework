"""Testbeds with analytical ground truth.

Each `make_*` function returns a `Testbed` carrying:
  - data        (numpy array, or callable for samplers)
  - ground_truth (dict mapping metric name -> expected value)
  - description (str)

Use these to validate metric implementations and to benchmark proxy estimators
against exact references on cases where the true value is known.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class Testbed:
    name: str
    data: Any
    ground_truth: dict[str, float]
    description: str = ""


# ---------------------------------------------------------------------------
# Classical testbeds
# ---------------------------------------------------------------------------

def make_gaussian(n: int = 2000, d: int = 2, seed: int = 0) -> Testbed:
    """N(0, I_d) — analytical entropy = (d/2)(1 + log(2π))."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    h = 0.5 * d * (1.0 + log(2.0 * pi))
    return Testbed(
        name=f"gaussian_d{d}_n{n}",
        data=X,
        ground_truth={"differential_entropy": h, "intrinsic_dim": float(d)},
        description=f"isotropic gaussian N(0,I) in R^{d}",
    )


def make_uniform_box(n: int = 2000, d: int = 2, seed: int = 0) -> Testbed:
    """U([0,1]^d) — analytical entropy = 0."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, d))
    return Testbed(
        name=f"uniform_d{d}_n{n}",
        data=X,
        ground_truth={"differential_entropy": 0.0, "intrinsic_dim": float(d)},
        description=f"uniform on [0,1]^{d}",
    )


def make_sphere(n: int = 1000, d_intrinsic: int = 2, d_ambient: int = 5, seed: int = 0) -> Testbed:
    """Points uniformly on S^{d_intrinsic} embedded in R^{d_ambient}."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_intrinsic + 1))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if d_ambient > d_intrinsic + 1:
        X = np.hstack([X, np.zeros((n, d_ambient - (d_intrinsic + 1)))])
    # Standard Betti for S^d: β_0 = β_d = 1, others zero
    betti = {k: 0 for k in range(max(d_intrinsic + 1, 3))}
    betti[0] = 1
    betti[d_intrinsic] = 1
    return Testbed(
        name=f"sphere_S{d_intrinsic}_amb{d_ambient}_n{n}",
        data=X,
        ground_truth={
            "intrinsic_dim": float(d_intrinsic),
            "betti": betti,
        },
        description=f"S^{d_intrinsic} embedded in R^{d_ambient}",
    )


def make_torus(n: int = 1500, R: float = 2.0, r: float = 0.7, seed: int = 0) -> Testbed:
    """2-torus T² in R³. Betti: (1, 2, 1)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * pi, n)
    phi = rng.uniform(0, 2 * pi, n)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    X = np.column_stack([x, y, z])
    return Testbed(
        name=f"torus_n{n}",
        data=X,
        ground_truth={
            "intrinsic_dim": 2.0,
            "betti": {0: 1, 1: 2, 2: 1},
        },
        description="2-torus T² embedded in R³",
    )


def make_correlated_gaussian_pair(n: int = 4000, rho: float = 0.7, seed: int = 0) -> Testbed:
    """Bivariate Gaussian with correlation ρ. MI = -½ ln(1-ρ²)."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    X = rng.multivariate_normal(np.zeros(2), cov, size=n)
    mi = -0.5 * log(1.0 - rho ** 2)
    return Testbed(
        name=f"corr_gauss_pair_rho{rho}_n{n}",
        data=X,
        ground_truth={"second_order_mi": mi},
        description=f"bivariate gaussian, corr={rho}",
    )


def make_equicorrelated_gaussian(n: int = 4000, k: int = 3, rho: float = 0.6, seed: int = 0) -> Testbed:
    """k-variate Gaussian, unit variance, equicorrelation ρ off-diagonal.

    Total correlation TC = ½ ln(∏ diag Σ / det Σ) with det Σ =
    (1-ρ)^(k-1)·(1+(k-1)ρ) and unit diagonal, so
    TC = -½[(k-1)ln(1-ρ) + ln(1+(k-1)ρ)].
    """
    rng = np.random.default_rng(seed)
    cov = np.full((k, k), rho)
    np.fill_diagonal(cov, 1.0)
    X = rng.multivariate_normal(np.zeros(k), cov, size=n)
    det = (1.0 - rho) ** (k - 1) * (1.0 + (k - 1) * rho)
    tc = -0.5 * log(det)
    return Testbed(
        name=f"equicorr_gauss_k{k}_rho{rho}_n{n}",
        data=X,
        ground_truth={"higher_order_mi": tc},
        description=f"{k}-variate equicorrelated gaussian, rho={rho}",
    )


def make_pseudorandom_bytes(n_bytes: int = 4096, seed: int = 0) -> Testbed:
    """High-entropy uniform bytes — gzip ratio should be near 1."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes()
    return Testbed(
        name=f"prng_bytes_{n_bytes}",
        data=data,
        ground_truth={"compression_ratio": 1.0},
        description=f"{n_bytes} pseudorandom bytes",
    )


def make_redundant_string(n_bytes: int = 4096, period: int = 8) -> Testbed:
    """Periodic byte sequence — gzip ratio near 0."""
    pattern = bytes(range(period))
    data = (pattern * (n_bytes // period + 1))[:n_bytes]
    return Testbed(
        name=f"periodic_p{period}_n{n_bytes}",
        data=data,
        ground_truth={"compression_ratio": 0.0},
        description=f"period-{period} byte sequence of length {n_bytes}",
    )


# ---------------------------------------------------------------------------
# Quantum state testbeds
# ---------------------------------------------------------------------------

def make_product_state(n_qubits: int = 4) -> Testbed:
    """|0...0> — half-cut entanglement entropy = 0."""
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[0] = 1.0
    return Testbed(
        name=f"product_state_n{n_qubits}",
        data=psi,
        ground_truth={"entanglement_entropy": 0.0},
        description=f"|0>^{n_qubits} product state",
    )


def make_ghz_state(n_qubits: int = 4) -> Testbed:
    """GHZ state — half-cut entropy = log(2)."""
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2.0)
    psi[-1] = 1.0 / np.sqrt(2.0)
    return Testbed(
        name=f"ghz_state_n{n_qubits}",
        data=psi,
        ground_truth={"entanglement_entropy": float(np.log(2.0))},
        description=f"GHZ on {n_qubits} qubits",
    )


def make_haar_random_state(n_qubits: int = 4, seed: int = 0) -> Testbed:
    """Haar-random pure state — half-cut entropy concentrates near
    log(d_A) - d_A/(2 d_B) (Page's value), with d_A = 2^(n/2)."""
    rng = np.random.default_rng(seed)
    D = 2 ** n_qubits
    z = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    psi = z / np.linalg.norm(z)
    nA = n_qubits // 2
    nB = n_qubits - nA
    dA = 2 ** nA
    dB = 2 ** nB
    # Page's formula (asymptotic): S ≈ log(d_A) - d_A/(2 d_B) for d_A <= d_B
    page = float(np.log(dA) - dA / (2.0 * dB)) if dA <= dB else float(np.log(dB) - dB / (2.0 * dA))
    return Testbed(
        name=f"haar_state_n{n_qubits}",
        data=psi,
        ground_truth={"entanglement_entropy": page},
        description=f"Haar-random pure state on {n_qubits} qubits (Page entropy)",
    )


def make_haar_state_sampler(n_qubits: int = 4, seed: int = 0):
    """Sampler returning Haar-random states. For expressibility benchmarks,
    KL to Haar should be ≈ 0."""
    rng = np.random.default_rng(seed)
    D = 2 ** n_qubits

    def sampler():
        z = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        return z / np.linalg.norm(z)

    return Testbed(
        name=f"haar_sampler_n{n_qubits}",
        data=sampler,
        ground_truth={"expressibility_kl": 0.0},
        description=f"Haar-random sampler on {n_qubits} qubits",
    )


def make_constant_state_sampler(n_qubits: int = 4) -> Testbed:
    """Constant |0…0> sampler. Maximally non-expressive: all overlaps are 1.
    KL to Haar diverges (we only check it's large)."""
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[0] = 1.0
    return Testbed(
        name=f"constant_sampler_n{n_qubits}",
        data=lambda: psi,
        ground_truth={"expressibility_kl": float("inf")},
        description=f"constant |0>^{n_qubits} sampler",
    )


# ---------------------------------------------------------------------------
# Kernel testbeds
# ---------------------------------------------------------------------------

def make_identity_kernel(n: int = 50) -> Testbed:
    """Identity Gram matrix — effective rank = n."""
    K = np.eye(n)
    return Testbed(
        name=f"identity_kernel_n{n}",
        data=K,
        ground_truth={"kernel_effective_rank": float(n)},
        description=f"I_{n} Gram matrix",
    )


def make_rank_one_kernel(n: int = 50, seed: int = 0) -> Testbed:
    """v vᵀ — effective rank = 1."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    K = np.outer(v, v)
    return Testbed(
        name=f"rank1_kernel_n{n}",
        data=K,
        ground_truth={"kernel_effective_rank": 1.0},
        description="rank-1 PSD matrix vvᵀ",
    )
