"""Encoding library — quantum feature maps for Part II experiments.

Each encoding consumes a numpy feature vector and returns a numpy state vector.
All encodings share the cost-computation interface from EncodingBase, with
per-family overrides for depth() and gates_per_layer().

Reference noise model:
    eps_2 = 1e-3 (two-qubit gate error)
    eps_1 = 1e-4 (single-qubit gate error)

State-vector simulation in raw numpy. Suitable for n <= 14 qubits exact.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class EncodingBase(ABC):
    """Abstract base for encodings. Subclasses implement encode(), depth(),
    gates_per_layer().
    """

    name: str = "encoding"
    n_qubits: int

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Map a feature vector x to a state vector |psi(x)> of shape (2**n,)."""
        ...

    @abstractmethod
    def depth(self) -> int:
        """Two-qubit-gate-equivalent encoding depth D_phi."""
        ...

    @abstractmethod
    def gates_per_layer(self) -> int:
        """Average two-qubit gates per layer W_phi.

        Returns 0 for purely single-qubit encodings (handled in cost()).
        """
        ...

    def cost(self, eps_2: float = 1e-3, single_qubit_constant: float = 0.1) -> float:
        """Encoding cost per Eq. 2.9 of framework_v2.md.

        C_phi = D_phi * W_phi * (1 - eps_2)^(-D_phi * W_phi)

        For single-qubit-only encodings (W_phi = 0), returns D_phi * single_qubit_constant
        as documented in §1.4 of encoding_library.md.
        """
        D = self.depth()
        W = self.gates_per_layer()
        if W == 0:
            return D * single_qubit_constant
        n_2q = D * W
        return float(D * W * (1.0 - eps_2) ** (-n_2q))

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r} n_qubits={self.n_qubits}>"


# ---------------------------------------------------------------------------
# Single-qubit gate primitives
# ---------------------------------------------------------------------------

def _ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _h() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a 2x2 gate to qubit `qubit` of an n-qubit state."""
    # Reshape: (2,)*n indexing, with axis `qubit` holding the gate target.
    state = state.reshape([2] * n)
    state = np.tensordot(gate, state, axes=([1], [qubit]))
    # tensordot puts the contracted axis first; move it back to its position
    state = np.moveaxis(state, 0, qubit)
    return state.reshape(2 ** n)


def _apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    """Apply CNOT(control, target) to an n-qubit state."""
    state = state.reshape([2] * n).copy()
    # Slice on control = 1, then flip the target bit.
    idx = [slice(None)] * n
    idx[control] = 1
    sub = state[tuple(idx)]
    sub = np.flip(sub, axis=target if target < control else target - 1)
    state[tuple(idx)] = sub
    return state.reshape(2 ** n)


def _apply_zz(state: np.ndarray, theta: float, q1: int, q2: int, n: int) -> np.ndarray:
    """Apply exp(-i theta Z_{q1} Z_{q2} / 2) to an n-qubit state.

    Diagonal in computational basis: phase is +theta/2 if Z_q1 Z_q2 = +1, else -theta/2.
    Equivalently: phase = (theta/2) * (-1)^{b_q1 + b_q2}.
    """
    state = state.reshape([2] * n).copy()
    # Build a phase tensor of shape (2,)*n
    bits = np.indices((2,) * n)
    parity = (bits[q1] + bits[q2]) % 2  # 0 if equal, 1 if different
    # Z_q1 Z_q2 eigenvalue: +1 if parity = 0, -1 if parity = 1
    sign = 1 - 2 * parity  # +1 or -1
    phase = np.exp(-1j * theta * sign / 2.0)
    state = state * phase
    return state.reshape(2 ** n)


# ---------------------------------------------------------------------------
# Angle encoding
# ---------------------------------------------------------------------------

class AngleEncoding(EncodingBase):
    """Angle encoding with optional entangling layer.

    Patterns:
      - "none": just R_y rotations, no entanglement
      - "linear": R_y rotations followed by linear CNOT chain
      - "ring":   R_y rotations followed by ring CNOT pattern
    """

    name = "angle"

    def __init__(self, n_qubits: int, entanglement: str = "linear"):
        if entanglement not in {"none", "linear", "ring"}:
            raise ValueError(f"unknown entanglement pattern {entanglement!r}")
        self.n_qubits = n_qubits
        self.entanglement = entanglement
        self.name = f"angle-{entanglement}"

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_qubits:
            raise ValueError(
                f"feature vector dim {x.shape[0]} != n_qubits {self.n_qubits}"
            )
        # Initialize |0>^n
        state = np.zeros(2 ** self.n_qubits, dtype=complex)
        state[0] = 1.0
        # Apply R_y(x_i) on qubit i
        for i in range(self.n_qubits):
            state = _apply_single_qubit_gate(state, _ry(x[i]), i, self.n_qubits)
        # Apply entangling layer
        if self.entanglement == "linear":
            for i in range(self.n_qubits - 1):
                state = _apply_cnot(state, i, i + 1, self.n_qubits)
        elif self.entanglement == "ring":
            for i in range(self.n_qubits):
                state = _apply_cnot(state, i, (i + 1) % self.n_qubits, self.n_qubits)
        return state

    def depth(self) -> int:
        return 1 if self.entanglement == "none" else 2

    def gates_per_layer(self) -> int:
        if self.entanglement == "none":
            return 0
        if self.entanglement == "linear":
            return self.n_qubits - 1
        if self.entanglement == "ring":
            return self.n_qubits
        return 0


# ---------------------------------------------------------------------------
# ZZ feature map
# ---------------------------------------------------------------------------

class ZZFeatureMap(EncodingBase):
    """Order-2 ZZ-feature map (Havlíček 2019).

    U_Phi(x) = exp(i sum_i x_i Z_i + i sum_{i<j in P} (pi-x_i)(pi-x_j) Z_i Z_j)

    pattern P: {linear, full}
    repetitions: r
    """

    name = "zz"

    def __init__(self, n_qubits: int, pattern: str = "linear", repetitions: int = 1):
        if pattern not in {"linear", "full"}:
            raise ValueError(f"unknown pattern {pattern!r}")
        if repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        self.n_qubits = n_qubits
        self.pattern = pattern
        self.repetitions = repetitions
        self.name = f"zz-r{repetitions}-{pattern}"
        self._pairs = self._compute_pairs()

    def _compute_pairs(self):
        if self.pattern == "linear":
            return [(i, i + 1) for i in range(self.n_qubits - 1)]
        # full
        return [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_qubits:
            raise ValueError(
                f"feature vector dim {x.shape[0]} != n_qubits {self.n_qubits}"
            )
        n = self.n_qubits
        # Initialize |0>^n, then H^n
        state = np.zeros(2 ** n, dtype=complex)
        state[0] = 1.0
        H = _h()
        for i in range(n):
            state = _apply_single_qubit_gate(state, H, i, n)
        # r repetitions of U_Phi(x)
        for _r in range(self.repetitions):
            # Single-qubit Z rotations: phase exp(-i x_i Z_i / 2 * 2) = exp(-i x_i Z_i)
            # In computational basis: diag(exp(-i x_i), exp(+i x_i))
            for i in range(n):
                state = self._apply_z_rotation(state, 2.0 * x[i], i, n)
            # ZZ couplings
            for (q1, q2) in self._pairs:
                theta = 2.0 * (math.pi - x[q1]) * (math.pi - x[q2])
                state = _apply_zz(state, theta, q1, q2, n)
            # Final layer of Hadamards (between repetitions, and final)
            if _r < self.repetitions - 1:
                for i in range(n):
                    state = _apply_single_qubit_gate(state, H, i, n)
        return state

    @staticmethod
    def _apply_z_rotation(state, theta, qubit, n):
        # exp(-i theta Z / 2) in computational basis: diag(exp(-i theta/2), exp(+i theta/2))
        state = state.reshape([2] * n).copy()
        # Multiply each branch by phase
        idx0 = [slice(None)] * n
        idx0[qubit] = 0
        idx1 = [slice(None)] * n
        idx1[qubit] = 1
        state[tuple(idx0)] *= np.exp(-1j * theta / 2.0)
        state[tuple(idx1)] *= np.exp(+1j * theta / 2.0)
        return state.reshape(2 ** n)

    def depth(self) -> int:
        # 1 layer for single-qubit Z rotations + |pairs| layers for ZZ
        # plus one more for final H pad (already counted as single-qubit)
        # For cost accounting: the depth of two-qubit operations is len(pairs)*r
        # plus the rest is single-qubit.
        return self.repetitions * len(self._pairs) + 1

    def gates_per_layer(self) -> int:
        # Each ZZ layer is a single two-qubit operation in our sequential decomposition
        return 1


# ---------------------------------------------------------------------------
# Amplitude encoding
# ---------------------------------------------------------------------------

class AmplitudeEncoding(EncodingBase):
    """Amplitude encoding: |psi> = x / ||x||, padded/truncated to dim 2**n.

    State preparation cost is O(2^n) two-qubit gates (Möttönen decomposition).
    """

    name = "amplitude"

    def __init__(self, n_qubits: int, padding: str = "zero"):
        if padding not in {"zero"}:
            raise ValueError(f"unknown padding {padding!r}")
        self.n_qubits = n_qubits
        self.padding = padding

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        D = 2 ** self.n_qubits
        if x.shape[0] > D:
            raise ValueError(f"feature dim {x.shape[0]} > 2^n = {D}; reduce first")
        if x.shape[0] < D:
            pad = np.zeros(D - x.shape[0], dtype=float)
            x = np.concatenate([x, pad])
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            # Degenerate: return |0...0>
            state = np.zeros(D, dtype=complex)
            state[0] = 1.0
            return state
        return (x / norm).astype(complex)

    def depth(self) -> int:
        # Möttönen: ~2^(n+1) two-qubit gates in worst-case sequential decomposition
        return 2 ** self.n_qubits

    def gates_per_layer(self) -> int:
        return 1  # sequential


# ---------------------------------------------------------------------------
# Data re-uploading
# ---------------------------------------------------------------------------

class DataReuploading(EncodingBase):
    """Data re-uploading with angle-linear base encoding.

    The encoding is L_ru applications of angle-linear encoding, with trainable
    layers between them. For cost accounting, only the encoding layers are
    counted here; trainable-layer cost is part of the ansatz budget.
    """

    name = "reup"

    def __init__(self, n_qubits: int, n_reuploads: int = 2):
        if n_reuploads < 1:
            raise ValueError("n_reuploads must be >= 1")
        self.n_qubits = n_qubits
        self.n_reuploads = n_reuploads
        self.name = f"reup-{n_reuploads}"
        self._base = AngleEncoding(n_qubits, entanglement="linear")

    def encode(self, x: np.ndarray) -> np.ndarray:
        # For the encoding-only path (no trainable layers), just apply the base
        # encoding repeatedly. With trainable layers, this would interleave
        # ansatz layers — handled by the ansatz module, not here.
        state = self._base.encode(x)
        for _ in range(self.n_reuploads - 1):
            # Apply base encoding "again" to the existing state (gates re-applied)
            x_arr = np.asarray(x, dtype=float)
            # Same gate sequence applied to current state
            for i in range(self.n_qubits):
                state = _apply_single_qubit_gate(state, _ry(x_arr[i]), i, self.n_qubits)
            for i in range(self.n_qubits - 1):
                state = _apply_cnot(state, i, i + 1, self.n_qubits)
        return state

    def depth(self) -> int:
        return self.n_reuploads * self._base.depth()

    def gates_per_layer(self) -> int:
        return self._base.gates_per_layer()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def all_phase_b_configurations(n_qubits: int) -> dict[str, EncodingBase]:
    """Return all 10 Phase B encoding configurations keyed by name."""
    return {
        "angle-none": AngleEncoding(n_qubits, entanglement="none"),
        "angle-linear": AngleEncoding(n_qubits, entanglement="linear"),
        "angle-ring": AngleEncoding(n_qubits, entanglement="ring"),
        "zz-r1-linear": ZZFeatureMap(n_qubits, pattern="linear", repetitions=1),
        "zz-r1-full": ZZFeatureMap(n_qubits, pattern="full", repetitions=1),
        "zz-r2-linear": ZZFeatureMap(n_qubits, pattern="linear", repetitions=2),
        "zz-r2-full": ZZFeatureMap(n_qubits, pattern="full", repetitions=2),
        "amplitude": AmplitudeEncoding(n_qubits),
        "reup-2": DataReuploading(n_qubits, n_reuploads=2),
        "reup-4": DataReuploading(n_qubits, n_reuploads=4),
    }
