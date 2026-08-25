"""Ansatz library — parameterized variational circuits for Part II experiments.

Three families, each with shallow (L=1) and deep (L=3) Phase B configurations:
  - HardwareEfficient (hea-L1, hea-L3): Ry/Rz per qubit + linear CNOT chain
  - AlternatingLayered (alt-L1, alt-L3): Ry/Rz per qubit + alternating even/odd CNOT pairs
  - Equivariant (equiv-L1, equiv-L3): tied Ry/Rz (same param all qubits) + ring CNOT (Z_n-symmetric)

Cost interface mirrors encodings.py: depth() * gates_per_layer() via Eq. 2.9.
Budget-matched comparison uses C_total = C_encoding + C_ansatz.

gradient_variance() estimates Var[∂E/∂θ_k] via parameter-shift rule (exact for Ry/Rz).
Observable: Z ⊗ I^{n-1} (Pauli Z on the first qubit).

Re-uploading integration: apply_reuploading_circuit() handles the interleaving of
angle-linear encoding layers with ansatz layers as required by DataReuploading configs.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Gate primitives
# ---------------------------------------------------------------------------

def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    e = complex(math.cos(theta / 2.0), math.sin(theta / 2.0))
    return np.array([[e.conjugate(), 0], [0, e]], dtype=complex)


def _apply_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    state = np.tensordot(gate, state, axes=([1], [qubit]))
    state = np.moveaxis(state, 0, qubit)
    return state.reshape(2 ** n)


def _apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n).copy()
    idx: list = [slice(None)] * n
    idx[control] = 1
    sub = state[tuple(idx)]
    sub = np.flip(sub, axis=target if target < control else target - 1)
    state[tuple(idx)] = sub
    return state.reshape(2 ** n)


# ---------------------------------------------------------------------------
# Expectation value helper
# ---------------------------------------------------------------------------

def _expectation_z0(state: np.ndarray, n: int) -> float:
    """<state|Z_0 ⊗ I^{n-1}|state>."""
    s = state.reshape(2, -1)
    return float(np.sum(np.abs(s[0]) ** 2) - np.sum(np.abs(s[1]) ** 2))


def _expectation_z0_batch(states: np.ndarray, n: int) -> np.ndarray:
    """⟨Z_0⟩ for a batch of flat statevectors. states: (B, 2**n) → (B,)."""
    s = states.reshape(states.shape[0], 2, -1)
    return np.sum(np.abs(s[:, 0, :]) ** 2, axis=1) - np.sum(np.abs(s[:, 1, :]) ** 2, axis=1)


# Tensor-form gate helpers: operate on a state whose LAST n axes are the qubit
# axes, with any number of leading batch axes (0 for a single state, 1 for a
# batch). This lets one apply() propagate a whole batch in single tensordot ops.

def _apply_gate_t(state_t: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    qaxis = state_t.ndim - n + qubit
    out = np.tensordot(gate, state_t, axes=([1], [qaxis]))  # gate-output axis → 0
    return np.moveaxis(out, 0, qaxis)


def _apply_cnot_t(state_t: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    nd = state_t.ndim
    off = nd - n
    c, t = off + control, off + target
    idx: list = [slice(None)] * nd
    idx[c] = 1
    state_t = state_t.copy()
    sub = state_t[tuple(idx)]
    sub = np.flip(sub, axis=t if t < c else t - 1)
    state_t[tuple(idx)] = sub
    return state_t


def _ry_batch(thetas: np.ndarray) -> np.ndarray:
    """Stack of Ry(θ_b) gates, shape (B, 2, 2), for per-sample rotation angles."""
    c = np.cos(thetas / 2.0)
    s = np.sin(thetas / 2.0)
    g = np.zeros((len(thetas), 2, 2), dtype=complex)
    g[:, 0, 0] = c; g[:, 0, 1] = -s
    g[:, 1, 0] = s; g[:, 1, 1] = c
    return g


def _apply_gate_batched(state_t: np.ndarray, gates: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a *per-sample* single-qubit gate to a batch. state_t: (B,) + (2,)*n;
    gates: (B, 2, 2). Used for data-dependent re-uploading rotations Ry(x_b[qubit])."""
    B = state_t.shape[0]
    qaxis = 1 + qubit
    moved = np.moveaxis(state_t, qaxis, -1)          # (B, ..., 2) — qubit axis last
    shp = moved.shape
    st = moved.reshape(B, -1, 2)                     # (B, M, 2)
    out = np.einsum("bij,bmj->bmi", gates, st).reshape(shp)
    return np.moveaxis(out, -1, qaxis)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AnsatzBase(ABC):
    """Abstract base for variational ansätze.

    Subclasses implement apply(), n_params, depth(), gates_per_layer().
    gradient_variance() is shared and uses the parameter-shift rule.
    """

    name: str = "ansatz"
    n_qubits: int
    n_layers: int

    @abstractmethod
    def _ops(self, theta: np.ndarray) -> list[tuple]:
        """Gate op-list for this circuit at parameters θ. Each op is either
        ('g', qubit, gate2x2) for a single-qubit gate or ('cx', control, target)
        for a CNOT. Single source of truth shared by apply() and apply_batch()."""
        ...

    def apply(self, state: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Apply the circuit to one state. Returns evolved state of shape (2**n,)."""
        n = self.n_qubits
        st = np.asarray(state, dtype=complex).reshape((2,) * n)
        for op in self._ops(np.asarray(theta, dtype=float)):
            st = _apply_gate_t(st, op[2], op[1], n) if op[0] == "g" \
                else _apply_cnot_t(st, op[1], op[2], n)
        return st.reshape(2 ** n)

    def apply_batch(self, states: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Apply the circuit to a batch of states. states: (B, 2**n) → (B, 2**n).
        Numerically identical to looping apply() over the batch."""
        n = self.n_qubits
        B = states.shape[0]
        st = np.asarray(states, dtype=complex).reshape((B,) + (2,) * n)
        for op in self._ops(np.asarray(theta, dtype=float)):
            st = _apply_gate_t(st, op[2], op[1], n) if op[0] == "g" \
                else _apply_cnot_t(st, op[1], op[2], n)
        return st.reshape(B, 2 ** n)

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of trainable parameters."""
        ...

    @abstractmethod
    def depth(self) -> int:
        """Two-qubit-gate-equivalent ansatz depth D_ansatz."""
        ...

    @abstractmethod
    def gates_per_layer(self) -> int:
        """Average two-qubit gates per trainable layer W_ansatz."""
        ...

    def cost(self, eps_2: float = 1e-3) -> float:
        """Ansatz cost per Eq. 2.9 of framework_v2.md.

        C_ansatz = D_ansatz * W_ansatz * (1 - eps_2)^(-D_ansatz * W_ansatz)

        Add to encoding cost for budget-matched comparison.
        """
        D, W = self.depth(), self.gates_per_layer()
        if W == 0:
            return D * 0.1
        n_2q = D * W
        return float(n_2q * (1.0 - eps_2) ** (-n_2q))

    def random_params(self, seed: int | None = None) -> np.ndarray:
        """Uniform random parameters in [0, 2π]."""
        return np.random.default_rng(seed).uniform(0.0, 2 * math.pi, self.n_params)

    def gradient_variance(
        self,
        encoded_states: np.ndarray,
        n_samples: int = 200,
        seed: int | None = None,
        param_index: int | None = None,
    ) -> float:
        """Estimate Var_θ[∂E/∂θ_k] via parameter-shift rule.

        Observable E = <psi(θ)|Z_0 ⊗ I^{n-1}|psi(θ)>. For each sample: draw a
        random encoded state, draw a random parameter initialization θ ~ Uniform[0,2π],
        pick a parameter index k (random unless param_index is given), then apply the
        two-point parameter-shift estimator. Variance over n_samples draws is returned.

        Parameter-shift is exact for Ry and Rz gates (generator eigenvalues ±1/2,
        shift = π/2).

        Parameters
        ----------
        encoded_states : shape (N, 2**n_qubits)
            Pre-encoded state vectors.
        n_samples : int
            Number of (state, θ, k) draws.
        seed : int or None
        param_index : int or None
            Fix k across all samples; if None, k is drawn uniformly per sample.

        Returns
        -------
        float
            Sample variance of the gradient estimator.
        """
        rng = np.random.default_rng(seed)
        N, dim = encoded_states.shape
        n = self.n_qubits
        if dim != 2 ** n:
            raise ValueError(f"state dim {dim} != 2**{n} = {2**n}")

        shift = math.pi / 2.0
        grads = np.empty(n_samples)
        for s in range(n_samples):
            state = encoded_states[int(rng.integers(0, N))]
            theta = rng.uniform(0.0, 2 * math.pi, self.n_params)
            k = param_index if param_index is not None else int(rng.integers(0, self.n_params))
            e_k = np.zeros(self.n_params)
            e_k[k] = shift
            g = (
                _expectation_z0(self.apply(state, theta + e_k), n)
                - _expectation_z0(self.apply(state, theta - e_k), n)
            ) / 2.0
            grads[s] = g
        return float(np.var(grads))

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"n_qubits={self.n_qubits} n_layers={self.n_layers} "
            f"n_params={self.n_params}>"
        )


# ---------------------------------------------------------------------------
# Hardware-efficient ansatz (HEA)
# ---------------------------------------------------------------------------

class HardwareEfficientAnsatz(AnsatzBase):
    """Hardware-efficient ansatz.

    Each layer: Ry(θ_i) on each qubit, Rz(φ_i) on each qubit, linear CNOT chain.
    n_params = 2 * n_qubits * n_layers.
    """

    name = "hea"

    def __init__(self, n_qubits: int, n_layers: int = 1):
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.name = f"hea-L{n_layers}"

    @property
    def n_params(self) -> int:
        return 2 * self.n_qubits * self.n_layers

    def _ops(self, theta: np.ndarray) -> list[tuple]:
        if len(theta) != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {len(theta)}")
        n = self.n_qubits
        ops: list[tuple] = []
        idx = 0
        for _ in range(self.n_layers):
            for i in range(n):
                ops.append(("g", i, _ry(theta[idx]))); idx += 1
            for i in range(n):
                ops.append(("g", i, _rz(theta[idx]))); idx += 1
            for i in range(n - 1):
                ops.append(("cx", i, i + 1))
        return ops

    def depth(self) -> int:
        return self.n_layers

    def gates_per_layer(self) -> int:
        return self.n_qubits - 1


# ---------------------------------------------------------------------------
# Alternating-layered ansatz (ALA)
# ---------------------------------------------------------------------------

class AlternatingLayeredAnsatz(AnsatzBase):
    """Alternating-layered ansatz.

    Layers alternate between even (0-1, 2-3, ...) and odd (1-2, 3-4, ...) CNOT pairs.
    n_params = 2 * n_qubits * n_layers.
    """

    name = "alt"

    def __init__(self, n_qubits: int, n_layers: int = 1):
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2 for alternating pattern")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.name = f"alt-L{n_layers}"

    @property
    def n_params(self) -> int:
        return 2 * self.n_qubits * self.n_layers

    def _ops(self, theta: np.ndarray) -> list[tuple]:
        if len(theta) != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {len(theta)}")
        n = self.n_qubits
        ops: list[tuple] = []
        idx = 0
        for layer in range(self.n_layers):
            for i in range(n):
                ops.append(("g", i, _ry(theta[idx]))); idx += 1
            for i in range(n):
                ops.append(("g", i, _rz(theta[idx]))); idx += 1
            start = 0 if layer % 2 == 0 else 1
            for i in range(start, n - 1, 2):
                ops.append(("cx", i, i + 1))
        return ops

    def depth(self) -> int:
        return self.n_layers

    def gates_per_layer(self) -> int:
        return self.n_qubits // 2


# ---------------------------------------------------------------------------
# Equivariant ansatz (Z_n cyclic symmetry)
# ---------------------------------------------------------------------------

class EquivariantAnsatz(AnsatzBase):
    """Cyclic-equivariant ansatz.

    Tied single-qubit rotations (same Ry angle, same Rz angle applied to all qubits)
    + ring CNOT entanglement. Equivariant under cyclic permutations of qubits (Z_n group).
    n_params = 2 * n_layers (one Ry, one Rz angle per layer, shared across all qubits).
    """

    name = "equiv"

    def __init__(self, n_qubits: int, n_layers: int = 1):
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.name = f"equiv-L{n_layers}"

    @property
    def n_params(self) -> int:
        return 2 * self.n_layers

    def _ops(self, theta: np.ndarray) -> list[tuple]:
        if len(theta) != self.n_params:
            raise ValueError(f"expected {self.n_params} params, got {len(theta)}")
        n = self.n_qubits
        ops: list[tuple] = []
        idx = 0
        for _ in range(self.n_layers):
            ry_gate = _ry(theta[idx]); idx += 1
            rz_gate = _rz(theta[idx]); idx += 1
            for i in range(n):
                ops.append(("g", i, ry_gate))
            for i in range(n):
                ops.append(("g", i, rz_gate))
            for i in range(n):
                ops.append(("cx", i, (i + 1) % n))
        return ops

    def depth(self) -> int:
        return self.n_layers

    def gates_per_layer(self) -> int:
        return self.n_qubits  # ring: n CNOTs


# ---------------------------------------------------------------------------
# Re-uploading integration
# ---------------------------------------------------------------------------

def _apply_angle_linear_to_state(
    state: np.ndarray, x: np.ndarray, n: int
) -> np.ndarray:
    """Apply one angle-linear encoding layer to an existing state (not from |0>).

    Used by apply_reuploading_circuit to interleave encoding and ansatz layers.
    Mirrors the gate sequence of AngleEncoding(entanglement="linear").encode() but
    applied in-place rather than starting from |0>.
    """
    for i in range(n):
        state = _apply_gate(state, _ry(x[i]), i, n)
    for i in range(n - 1):
        state = _apply_cnot(state, i, i + 1, n)
    return state


def apply_reuploading_circuit(
    x: np.ndarray,
    n_qubits: int,
    n_reuploads: int,
    ansatz: AnsatzBase,
    theta: np.ndarray,
) -> np.ndarray:
    """Full re-uploading circuit with interleaved ansatz layers.

    Circuit structure: [angle-linear(x) → ansatz(θ_ℓ)] × n_reuploads, starting from |0>^n.

    This realizes the Pérez-Salinas re-uploading scheme with independent trainable layers
    between each re-upload. Used for reup-2 and reup-4 configurations in Phase B.

    Parameters
    ----------
    x : shape (n_qubits,)
        Feature vector (already PCA-reduced or padded to n_qubits if needed).
    n_qubits : int
    n_reuploads : int
        Matches the DataReuploading encoding's n_reuploads.
    ansatz : AnsatzBase
        Trainable layers applied after each re-upload. depth-1 hardware-efficient
        is the Phase B default (per encoding_library.md §2.4).
    theta : shape (n_reuploads * ansatz.n_params,)
        Concatenated parameters for each re-upload layer (independent per layer).

    Returns
    -------
    np.ndarray
        Final state vector, shape (2**n_qubits,).
    """
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    expected_params = n_reuploads * ansatz.n_params
    if len(theta) != expected_params:
        raise ValueError(
            f"expected {expected_params} params ({n_reuploads} × {ansatz.n_params}), "
            f"got {len(theta)}"
        )
    n = n_qubits
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    for ell in range(n_reuploads):
        state = _apply_angle_linear_to_state(state, x, n)
        theta_ell = theta[ell * ansatz.n_params : (ell + 1) * ansatz.n_params]
        state = ansatz.apply(state, theta_ell)
    return state


def apply_reuploading_circuit_batch(
    X: np.ndarray,
    n_qubits: int,
    n_reuploads: int,
    ansatz: AnsatzBase,
    theta: np.ndarray,
) -> np.ndarray:
    """Batched re-uploading circuit — propagates B samples at once.

    Numerically identical to looping `apply_reuploading_circuit` over the rows of
    `X` (verified). The data-dependent encoding rotations Ry(x_b[i]) differ per
    sample, so they are applied as per-sample gate stacks (`_apply_gate_batched`);
    the CNOT chain and the ansatz layers are sample-independent and reuse the
    batched fixed-gate path (`_apply_cnot_t`, `ansatz.apply_batch`).

    X: (B, n_qubits) pre-scaled features. Returns (B, 2**n_qubits).
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)
    expected_params = n_reuploads * ansatz.n_params
    if len(theta) != expected_params:
        raise ValueError(
            f"expected {expected_params} params ({n_reuploads} × {ansatz.n_params}), got {len(theta)}"
        )
    n = n_qubits
    B = X.shape[0]
    st = np.zeros((B,) + (2,) * n, dtype=complex)
    st[(slice(None),) + (0,) * n] = 1.0           # |0>^n for every sample
    for ell in range(n_reuploads):
        for i in range(n):
            st = _apply_gate_batched(st, _ry_batch(X[:, i]), i, n)
        for i in range(n - 1):
            st = _apply_cnot_t(st, i, i + 1, n)
        theta_ell = theta[ell * ansatz.n_params : (ell + 1) * ansatz.n_params]
        st = ansatz.apply_batch(st.reshape(B, 2 ** n), theta_ell).reshape((B,) + (2,) * n)
    return st.reshape(B, 2 ** n)


# ---------------------------------------------------------------------------
# Phase B configuration registry
# ---------------------------------------------------------------------------

def all_phase_b_ansatze(n_qubits: int) -> dict[str, AnsatzBase]:
    """Return all 6 Phase B ansatz configurations keyed by name."""
    return {
        "hea-L1":   HardwareEfficientAnsatz(n_qubits, n_layers=1),
        "hea-L3":   HardwareEfficientAnsatz(n_qubits, n_layers=3),
        "alt-L1":   AlternatingLayeredAnsatz(n_qubits, n_layers=1),
        "alt-L3":   AlternatingLayeredAnsatz(n_qubits, n_layers=3),
        "equiv-L1": EquivariantAnsatz(n_qubits, n_layers=1),
        "equiv-L3": EquivariantAnsatz(n_qubits, n_layers=3),
    }
