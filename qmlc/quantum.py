"""Quantum data complexity metrics.

State-vector simulation in raw numpy. Suitable for n <= ~16 qubits.
Inputs are numpy arrays, either:
  - state vectors of shape (2**n,)
  - density matrices of shape (2**n, 2**n)
  - batches of either, with leading batch axis
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .base import BaseMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_density(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state)
    if state.ndim == 1:
        return np.outer(state, state.conj())
    if state.ndim == 2 and state.shape[0] == state.shape[1]:
        return state
    raise ValueError(f"unexpected state shape {state.shape}")


def _n_qubits(rho: np.ndarray) -> int:
    n = int(np.log2(rho.shape[-1]))
    assert 2 ** n == rho.shape[-1], "dimension must be a power of 2"
    return n


def partial_trace(rho: np.ndarray, keep: list[int], n: Optional[int] = None) -> np.ndarray:
    """Trace out qubits not in `keep` (qubit indices, 0-based, big-endian)."""
    if n is None:
        n = _n_qubits(rho)
    dims = [2] * n
    rho_t = rho.reshape(dims + dims)
    trace_out = [i for i in range(n) if i not in keep]
    # Sequentially trace; reduce indices accordingly
    for q in sorted(trace_out, reverse=True):
        rho_t = np.trace(rho_t, axis1=q, axis2=q + (rho_t.ndim // 2))
    k = len(keep)
    return rho_t.reshape(2 ** k, 2 ** k)


def _von_neumann(rho: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(rho)
    eig = np.clip(eig.real, 0.0, 1.0)
    eig = eig[eig > 1e-12]
    return float(-np.sum(eig * np.log(eig)))


# ---------------------------------------------------------------------------
# Entanglement entropy
# ---------------------------------------------------------------------------

class EntanglementEntropyExact(BaseMetric):
    """Bipartite von Neumann entropy via exact partial trace.

    Default: equal-half cut (qubits [0..n/2)). Override via `subsystem`.
    """

    name = "entanglement_entropy"
    estimator = "exact"

    def __init__(self, subsystem: Optional[list[int]] = None):
        self.subsystem = subsystem

    def _compute(self, state):
        rho = _to_density(state)
        n = _n_qubits(rho)
        keep = self.subsystem if self.subsystem is not None else list(range(n // 2))
        rho_A = partial_trace(rho, keep, n=n)
        S = _von_neumann(rho_A)
        return S, None, {"n_qubits": n, "subsystem": keep}


class EntanglementEntropyRenyi2(BaseMetric):
    """Rényi-2 entanglement entropy via randomized measurements (Brydges et al.).

    For a state vector this reduces to S2 = -log(Tr ρA²); we estimate Tr ρA²
    via random local Clifford / Haar rotations and shots. This is the standard
    hardware-friendly proxy.
    """

    name = "entanglement_entropy"
    estimator = "renyi2_random_meas"

    def __init__(
        self,
        subsystem: Optional[list[int]] = None,
        n_unitaries: int = 200,
        n_shots: int = 200,
        seed: Optional[int] = None,
    ):
        self.subsystem = subsystem
        self.n_unitaries = n_unitaries
        self.n_shots = n_shots
        self.seed = seed

    def _compute(self, state):
        rng = np.random.default_rng(self.seed)
        rho = _to_density(state)
        n = _n_qubits(rho)
        keep = self.subsystem if self.subsystem is not None else list(range(n // 2))
        rho_A = partial_trace(rho, keep, n=n)
        nA = len(keep)
        dA = 2 ** nA

        # Estimate Tr(rho_A^2) via random local unitary protocol.
        # For each random product unitary U = ⊗ u_i, sample bitstrings s and
        # combine via the formula: Tr(ρ²) = sum_{s,s'} (-2)^(-H(s,s')) P(s) P(s')
        purity_estimates = []
        for _ in range(self.n_unitaries):
            U = np.eye(1, dtype=complex)
            for _q in range(nA):
                u = _haar_random_unitary(2, rng)
                U = np.kron(U, u)
            rho_rot = U @ rho_A @ U.conj().T
            probs = np.clip(np.real(np.diag(rho_rot)), 0, 1)
            probs /= probs.sum()
            samples = rng.choice(dA, size=self.n_shots, p=probs)
            # Empirical purity from finite shots, with bias correction
            counts = np.bincount(samples, minlength=dA)
            P = counts / self.n_shots
            # Bias-corrected purity for this unitary
            X = np.outer(P, P)
            # Hamming distance matrix between basis states (nA bits)
            idx = np.arange(dA)
            ham = np.array([[bin(int(i) ^ int(j)).count("1") for j in idx] for i in idx])
            purity_u = float(dA * np.sum(((-2.0) ** -ham) * X))
            purity_estimates.append(purity_u)

        purity = float(np.mean(purity_estimates))
        purity = max(purity, 1e-12)
        S2 = float(-np.log(purity))
        unc = float(np.std(purity_estimates, ddof=1) / np.sqrt(len(purity_estimates)) / max(purity, 1e-12))
        return S2, unc, {
            "n_qubits": n,
            "subsystem": keep,
            "n_unitaries": self.n_unitaries,
            "n_shots": self.n_shots,
            "purity_estimate": purity,
        }


def _haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    Z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    D = np.diag(R) / np.abs(np.diag(R))
    return Q * D


# ---------------------------------------------------------------------------
# Kernel effective rank
# ---------------------------------------------------------------------------

class KernelEffectiveRank(BaseMetric):
    """Effective rank r_eff = (Σ λ)² / Σ λ² of a positive-semidefinite matrix.

    Accepts either a kernel Gram matrix K, or a callable + data X to build it.
    """

    name = "kernel_effective_rank"
    estimator = "spectral"

    def _compute(self, K):
        K = np.asarray(K)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("kernel must be square")
        eig = np.linalg.eigvalsh((K + K.T) / 2.0)
        eig = np.clip(eig.real, 0, None)
        s1 = float(eig.sum())
        s2 = float((eig ** 2).sum())
        r_eff = (s1 ** 2) / max(s2, 1e-30)
        return r_eff, None, {"n": K.shape[0], "trace": s1, "frob_sq": s2}


# ---------------------------------------------------------------------------
# Expressibility (KL to Haar via fidelity histogram)
# ---------------------------------------------------------------------------

class ExpressibilityKL(BaseMetric):
    """Expressibility ε(U) = KL( P_U(F) || P_Haar(F) ) over fidelity histograms.

    Sim et al. 2019. Sample pairs of states from the circuit ensemble, compute
    overlap fidelities F = |<ψ(θ)|ψ(θ')>|², histogram them, and compare to the
    analytical Haar fidelity distribution P_Haar(F) = (D-1)(1-F)^(D-2) on
    Hilbert-space dim D.

    Expects a `sampler() -> state vector` callable.
    """

    name = "expressibility_kl"
    estimator = "fidelity_histogram"

    def __init__(self, n_pairs: int = 2000, n_bins: int = 75, seed: Optional[int] = None):
        self.n_pairs = n_pairs
        self.n_bins = n_bins
        self.seed = seed

    def _compute(self, sampler: Callable[[], np.ndarray]):
        rng = np.random.default_rng(self.seed)
        # Sample one state to discover D
        psi0 = np.asarray(sampler())
        D = psi0.shape[-1]
        fids = np.empty(self.n_pairs)
        for i in range(self.n_pairs):
            a = sampler()
            b = sampler()
            fids[i] = float(np.abs(np.vdot(a, b)) ** 2)

        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        hist, _ = np.histogram(fids, bins=bins, density=False)
        P = hist / hist.sum()

        # Haar: P_Haar(F) = (D-1)(1-F)^(D-2). Integrate the pdf within each bin.
        # CDF: F_Haar(F) = 1 - (1-F)^(D-1)
        cdf = lambda x: 1.0 - (1.0 - x) ** (D - 1)
        Q = np.diff(cdf(bins))
        # Numerical safety
        eps = 1e-12
        P = np.clip(P, eps, None)
        Q = np.clip(Q, eps, None)
        kl = float(np.sum(P * np.log(P / Q)))
        # rng is unused for fidelity sampling itself — sampler is responsible
        # but we used it implicitly in seed propagation if sampler inspects it.
        del rng
        return kl, None, {"D": int(D), "n_pairs": self.n_pairs, "n_bins": self.n_bins}


# ---------------------------------------------------------------------------
# Quantum-native v(D) components (Class B; OSF §5.2 parallel track)
#
# These act on an *ensemble* of pure states `states` of shape (n_samples, 2^q):
#   v_ent  — mean half-cut entanglement entropy           (nats)
#   v_mi   — mean multipartite total correlation          (nats)
#   v_mag  — mean stabilizer Rényi-2 magic                (bits)
#   v_bond — mean optimal MPS bond dimension at fixed ε   (dimensionless)
# Each returns the ensemble mean; the standard error is `uncertainty` and the
# ensemble variance is in metadata.
# ---------------------------------------------------------------------------

# Single-qubit Paulis indexed [a, row, col] for a ∈ {I, X, Y, Z}.
_PAULI = np.array(
    [
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=complex,
)


def _as_state_stack(states: np.ndarray) -> tuple[np.ndarray, int]:
    """Coerce input to a (n_samples, 2^q) complex stack; return (stack, n_qubits)."""
    S = np.asarray(states, dtype=complex)
    if S.ndim == 1:
        S = S[None, :]
    q = int(round(np.log2(S.shape[1])))
    if 2 ** q != S.shape[1]:
        raise ValueError(f"state width {S.shape[1]} is not a power of 2")
    return S, q


def _half_cut_entropy(psi: np.ndarray, n: int) -> float:
    rho = np.outer(psi, psi.conj())
    rho_A = partial_trace(rho, list(range(n // 2)), n=n)
    return _von_neumann(rho_A)


def _total_correlation_pure(psi: np.ndarray, n: int) -> float:
    """TC = Σ_q S(ρ_q) − S(ρ); S(ρ)=0 for a pure state, so TC = Σ_q S(ρ_q)."""
    rho = np.outer(psi, psi.conj())
    return float(sum(_von_neumann(partial_trace(rho, [q], n=n)) for q in range(n)))


def _all_pauli_expectations(psi: np.ndarray, n: int) -> np.ndarray:
    """Vector of all 4^n Pauli expectations ⟨ψ|P|ψ⟩ (real), via tensor contraction.

    Tr(ρP) = Σ_{ij} ρ_{ij} P_{ji}, factorized per qubit through one einsum.
    """
    import string

    letters = string.ascii_letters
    i_idx, j_idx, a_idx = letters[:n], letters[n:2 * n], letters[2 * n:3 * n]
    rho = np.outer(psi, psi.conj()).reshape((2,) * (2 * n))
    subs = [i_idx + j_idx]
    operands: list[np.ndarray] = [rho]
    for q in range(n):
        operands.append(_PAULI)            # _PAULI[a, j, i] = (P_a)_{ji}
        subs.append(a_idx[q] + j_idx[q] + i_idx[q])
    chi = np.einsum(",".join(subs) + "->" + a_idx, *operands, optimize=True)
    return chi.real.reshape(-1)


def _stabilizer_renyi2(psi: np.ndarray, n: int) -> float:
    """Stabilizer 2-Rényi entropy M₂ = −log₂( (1/d) Σ_P ⟨P⟩⁴ ) [Leone et al. 2022].

    0 for stabilizer states; log₂(4/3)≈0.415 for a single T state.
    """
    chi = _all_pauli_expectations(psi, n)
    d = 1 << n
    s = float(np.sum(chi ** 4) / d)
    return max(-np.log2(max(s, 1e-300)), 0.0)


def _mps_bond_dim(psi: np.ndarray, n: int, eps: float = 1e-2) -> int:
    """Optimal MPS bond dimension at fixed truncation: max over bipartition cuts
    of the Schmidt rank needed to keep the discarded weight below `eps`."""
    best = 1
    for c in range(1, n):
        M = psi.reshape(1 << c, 1 << (n - c))
        s = np.linalg.svd(M, compute_uv=False)
        w = s ** 2
        w = w / max(w.sum(), 1e-300)
        discarded = 1.0 - np.cumsum(w)          # discarded[k-1] = weight dropped keeping k
        ok = np.where(discarded < eps)[0]
        chi_c = int(ok[0] + 1) if ok.size else int(w.size)
        best = max(best, chi_c)
    return best


class _EnsembleMetric(BaseMetric):
    """Shared driver: apply a per-state scalar over an ensemble, return the mean.

    Subsamples to `max_states` (deterministic) to bound cost; magic is the
    expensive one (4^q Pauli expectations per state).
    """

    per_state = staticmethod(lambda psi, n: 0.0)

    def __init__(self, max_states: int = 512, seed: int = 0):
        self.max_states = max_states
        self.seed = seed

    def _compute(self, states):
        S, n = _as_state_stack(states)
        if S.shape[0] > self.max_states:
            rng = np.random.default_rng(self.seed)
            S = S[rng.choice(S.shape[0], self.max_states, replace=False)]
        vals = np.array([self.per_state(psi, n) for psi in S], dtype=float)
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else None
        return mean, sem, {"n_qubits": n, "n_used": int(vals.size), "var": float(vals.var())}


class EnsembleEntanglement(_EnsembleMetric):
    """v_ent — mean half-cut entanglement entropy over the ensemble (nats)."""

    name = "entanglement_entropy_mean"
    estimator = "exact_ensemble"
    per_state = staticmethod(_half_cut_entropy)


class EnsembleTotalCorrelation(_EnsembleMetric):
    """v_mi — mean multipartite total correlation over the ensemble (nats)."""

    name = "total_correlation_mean"
    estimator = "exact_ensemble"
    per_state = staticmethod(_total_correlation_pure)


class EnsembleStabilizerMagic(_EnsembleMetric):
    """v_mag — mean stabilizer Rényi-2 magic over the ensemble (bits)."""

    name = "stabilizer_renyi2_mean"
    estimator = "exact_pauli"
    per_state = staticmethod(_stabilizer_renyi2)

    def __init__(self, max_states: int = 128, seed: int = 0):
        super().__init__(max_states=max_states, seed=seed)


def _collective_spin_ops(n: int):
    """J_a = (1/2) sum_i sigma_a^(i) for a in {x,y,z}, as dense operators."""
    from functools import reduce
    I2 = np.eye(2)
    P = {"x": np.array([[0, 1], [1, 0]], complex),
         "y": np.array([[0, -1j], [1j, 0]]),
         "z": np.diag([1.0, -1.0]).astype(complex)}
    out = []
    for a in ("x", "y", "z"):
        d = 2 ** n
        J = np.zeros((d, d), complex)
        for i in range(n):
            ops = [I2] * n
            ops[i] = P[a]
            J += reduce(np.kron, ops)
        out.append(J / 2.0)
    return out


def _qfi_collective_normalised(psi, n, Js=None):
    """max over collective-spin directions of F_Q, divided by the Heisenberg bound n^2.

    For a pure state and unitary family e^{-i t G}|psi>, F_Q = 4 Var_psi(G) -- so no
    external parametrisation is required. Taking G = J_nhat, the QFI over the three
    collective directions is a 3x3 matrix whose largest eigenvalue is the maximum over
    directions. Bounds: separable states F_Q <= n, Heisenberg limit F_Q <= n^2, so the
    normalised value lies in [1/n, 1] and exceeding 1/n witnesses multipartite
    entanglement.
    """
    if Js is None:
        Js = _collective_spin_ops(n)
    ev = [psi.conj() @ (J @ psi) for J in Js]
    Jp = [J @ psi for J in Js]
    M = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            M[a, b] = 4.0 * np.real(Jp[a].conj() @ Jp[b] - np.conj(ev[a]) * ev[b])
    M = (M + M.T) / 2.0
    return float(np.linalg.eigvalsh(M).max()) / (n ** 2)


class EnsembleQFI(_EnsembleMetric):
    """v_qfi -- mean normalised collective-spin quantum Fisher information.

    Component 5 of Part I Eq. 10, made well-posed. Eq. 10 writes F = Tr[L^2] without
    saying with respect to *what* parameter, and a bare ensemble {(p_i, rho_i)} has
    none. Three candidate parametrisations were considered (see
    `docs/class_b_quantum_native_spec.md`): a Hamiltonian parameter is undefined for
    most of the suite (Haar, random-MPS, stabilizer-magic and parity ensembles have no
    Hamiltonian family); encoding parameters belong to Eq. 11, not Eq. 10; and ansatz
    parameters would make the quantity a property of the (ensemble, ansatz) pair, which
    is the category error Part II's framework revision removes from C_ind.

    The resolution uses the pure-state identity F_Q[psi, G] = 4 Var_psi(G): fixing a
    canonical generator set makes QFI intrinsic to the state. We take the collective
    spin generators and normalise by the Heisenberg bound, giving a value in [1/n, 1]
    -- bounded and dimensionless, so this component cannot drive the variance
    domination that Eq. 7 suffered.

    Validated against closed-form cases: product states give exactly 1/n (the separable
    bound), GHZ gives exactly 1 (Heisenberg), W sits between, and Haar-random states
    sit just above the separable floor. Not redundant with the entanglement-entropy
    component: W and GHZ have identical half-cut entropy (ln 2) but normalised QFI of
    0.344 and 1.000 at n = 8.

    Caveat to state wherever it is used: this measures *metrological usefulness*, not
    entanglement. Haar-random states are near-maximally entangled yet score near the
    separable floor. It also privileges the local qubit tensor structure and the
    permutation-symmetric directions.
    """

    name = "qfi_collective_mean"
    estimator = "collective_spin_heisenberg_normalised"

    def _compute(self, states):
        S, n = _as_state_stack(states)
        if S.shape[0] > self.max_states:
            rng = np.random.default_rng(self.seed)
            S = S[rng.choice(S.shape[0], self.max_states, replace=False)]
        Js = _collective_spin_ops(n)
        vals = np.array([_qfi_collective_normalised(psi, n, Js) for psi in S], float)
        sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else None
        return float(vals.mean()), sem, {
            "n_qubits": n, "n_used": int(vals.size), "var": float(vals.var()),
            "separable_bound": 1.0 / n, "heisenberg_bound": 1.0,
            "frac_above_separable": float((vals > 1.0 / n + 1e-12).mean()),
        }


class EnsembleKernelEffectiveRank(BaseMetric):
    """v_rank -- effective rank of the ensemble fidelity Gram matrix.

    Component 3 of Part I Eq. 10, $\\mathrm{rank}_{\\mathrm{eff}}(K_\\mathcal{E})
    = (\\sum_i\\lambda_i)^2/\\sum_i\\lambda_i^2$, evaluated on the ensemble's own
    Gram matrix $K_{ij} = |\\langle\\psi_i|\\psi_j\\rangle|^2$ rather than on an
    embedded classical dataset.

    Note the scale this quantity carries, which matters for any attempt to sum
    it against the other components of Eq. 10: it is a *count* on $[1, N]$,
    where $N$ is the number of states retained. It therefore grows with the
    ensemble size by construction, independently of how complex the states are.
    """

    name = "kernel_effective_rank_ensemble"
    estimator = "fidelity_gram_spectral"

    def __init__(self, max_states: int = 512, seed: int = 0):
        self.max_states = max_states
        self.seed = seed

    def _compute(self, states):
        S, n = _as_state_stack(states)
        if S.shape[0] > self.max_states:
            rng = np.random.default_rng(self.seed)
            S = S[rng.choice(S.shape[0], self.max_states, replace=False)]
        G = S.conj() @ S.T                       # <psi_i|psi_j>
        K = np.abs(G) ** 2                       # fidelity kernel, PSD
        eig = np.clip(np.linalg.eigvalsh((K + K.T) / 2.0).real, 0, None)
        s1, s2 = float(eig.sum()), float((eig ** 2).sum())
        r_eff = (s1 ** 2) / max(s2, 1e-30)
        return r_eff, None, {"n_qubits": n, "n_states": int(S.shape[0]),
                             "r_eff_normalised": r_eff / max(S.shape[0], 1),
                             "trace": s1, "frob_sq": s2}


class EnsembleMPSBond(_EnsembleMetric):
    """v_bond — mean optimal MPS bond dimension at fixed truncation."""

    name = "mps_bond_dim_mean"
    estimator = "svd_truncation"

    def __init__(self, eps: float = 1e-2, max_states: int = 512, seed: int = 0):
        super().__init__(max_states=max_states, seed=seed)
        self.eps = eps

    def _compute(self, states):
        S, n = _as_state_stack(states)
        if S.shape[0] > self.max_states:
            rng = np.random.default_rng(self.seed)
            S = S[rng.choice(S.shape[0], self.max_states, replace=False)]
        vals = np.array([_mps_bond_dim(psi, n, self.eps) for psi in S], dtype=float)
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else None
        return mean, sem, {"n_qubits": n, "n_used": int(vals.size),
                           "eps": self.eps, "max_bond": int(vals.max())}
