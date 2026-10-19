"""Structure-search battery: families, search algorithms, and budgeted estimators.

Implements the design in `docs/structure_search_battery_prespec.md`. The object
under test is a hidden structure inside a state ensemble, and the question is
which access model and which search procedure can recover it.

Background: on the shadow-hard parity probe, gradient-trained models fail while
an unsupervised reduced-state-purity search recovers the hidden support exactly.
Gradients see nothing because <P_S> = 0 for every incomplete subset -- there is
no partial credit. Purity does not have that problem: a subset CONTAINING the
support inherits the parity constraint, so its reduced state has half the rank
and roughly twice the Haar purity, and that excess is visible at every stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np

I2 = np.eye(2)
PAULI = {"I": I2,
         "X": np.array([[0, 1], [1, 0]], complex),
         "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1.0, -1.0]).astype(complex)}


# --------------------------------------------------------------------------
# Structure families
# --------------------------------------------------------------------------

@dataclass
class Probe:
    states: np.ndarray          # (n_states, 2**n) complex, rows normalised
    y: np.ndarray               # (n_states,) binary labels
    n: int
    support: tuple[int, ...]    # ground-truth qubits carrying the structure
    paulis: tuple[str, ...]     # ground-truth Pauli letters on `support`
    family: str
    parts: tuple[tuple[int, ...], ...] = ()   # for composite families: the ACTUAL
    # partition of `support` into separately-measured observables. Storing only the
    # sorted union loses this: splitting sorted(A+B) in half does NOT recover (A,B).


def _pauli_string_op(n: int, support, letters) -> np.ndarray:
    """Dense operator for the Pauli string given by (support, letters)."""
    ops = [I2] * n
    for q, L in zip(support, letters):
        ops[q] = PAULI[L]
    out = ops[0]
    for m in ops[1:]:
        out = np.kron(out, m)
    return out


def _apply_pauli_string(V: np.ndarray, n: int, support, letters) -> np.ndarray:
    """P @ V.T for a Pauli string, matrix-free. O(n_states * 2^n), no dense operator.

    A Pauli string is a signed permutation: (P v)[j] = phase(j) * v[j ^ XM], with
    XM the qubits carrying X or Y. Writing ZM/YM for the Z/Y masks,

        phase(j) = (-i)^{nY} * (-1)^{popcount(j & (ZM | YM))}

    Kron order matches _pauli_string_op: qubit 0 is the MOST significant bit.
    Used only for large d, where the dense (eye + P)/2 projector would need
    ~10 GB at n=14. The dense path below is left untouched so every published
    Part III number (n <= 12) is produced by exactly the code that produced it.
    """
    XM = ZM = YM = 0
    nY = 0
    for q, L in zip(support, letters):
        bit = 1 << (n - 1 - q)
        if L == "X": XM |= bit
        elif L == "Z": ZM |= bit
        elif L == "Y":
            XM |= bit; YM |= bit; nY += 1
    j = np.arange(V.shape[1], dtype=np.int64)
    x = (j & (ZM | YM)).copy()                 # parity of the masked bits
    for sh in (32, 16, 8, 4, 2, 1):
        x ^= x >> sh
    sign = np.where(x & 1, -1.0, 1.0)
    phase = ((-1j) ** nY) * sign
    return V[:, j ^ XM] * phase


def _haar_in_eigenspace(P: np.ndarray, sign: int, n_states: int, rng, _MF=None) -> np.ndarray:
    """Haar-random states inside the `sign` eigenspace of the Pauli operator P."""
    d = P.shape[0] if _MF is None else 2 ** _MF[0]
    V = rng.normal(size=(n_states, d)) + 1j * rng.normal(size=(n_states, d))
    if _MF is None:
        proj = (np.eye(d) + sign * P) / 2.0
        V = V @ proj.T
    else:
        n, support, letters = _MF
        V = (V + sign * _apply_pauli_string(V, n, support, letters)) / 2.0
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V


def make_probe(family: str, n: int, w: int, n_states: int = 128, seed: int = 0) -> Probe:
    """Build one probe ensemble. `w` is the weight of the hidden structure."""
    rng = np.random.default_rng(seed)
    d = 2 ** n

    if family == "null_control":
        V = rng.normal(size=(n_states, d)) + 1j * rng.normal(size=(n_states, d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        return Probe(V, rng.integers(0, 2, n_states), n, (), (), family)

    if family == "z_parity":
        support = tuple(rng.permutation(n)[:w]); letters = ("Z",) * w
    elif family == "pauli_stab":
        support = tuple(rng.permutation(n)[:w])
        letters = tuple(rng.choice(["X", "Y", "Z"], size=w))
    elif family == "double_parity":
        # Genuinely COMPOSITE structure. Two disjoint parities P_A, P_B (commuting),
        # label = AND(s_A == +1, s_B == +1). No single Pauli yields this: P_A gives
        # s_A, P_B gives s_B, and their product P_A P_B = P_{A u B} only separates
        # {++,--} from {+-,-+}. The learner must locate BOTH supports.
        h = max(1, w // 2)
        perm = rng.permutation(n)
        A = tuple(sorted(perm[:h])); B = tuple(sorted(perm[h: 2 * h]))
        PA = _pauli_string_op(n, A, ("Z",) * h)
        PB = _pauli_string_op(n, B, ("Z",) * h)
        per = max(1, n_states // 4)
        blocks, labels = [], []
        for sA in (+1, -1):
            for sB in (+1, -1):
                d_ = PA.shape[0]
                proj = ((np.eye(d_) + sA * PA) / 2.0) @ ((np.eye(d_) + sB * PB) / 2.0)
                V = rng.normal(size=(per, d_)) + 1j * rng.normal(size=(per, d_))
                V = V @ proj.T
                nrm = np.linalg.norm(V, axis=1, keepdims=True)
                keep = (nrm[:, 0] > 1e-9)
                V = V[keep] / nrm[keep]
                blocks.append(V)
                labels.append(np.full(len(V), int(sA > 0 and sB > 0)))
        V = np.vstack(blocks); y = np.concatenate(labels)
        idx = rng.permutation(len(V))
        return Probe(V[idx], y[idx], n, tuple(sorted(A + B)), ("Z",) * (2 * h), family,
                     parts=(A, B))
    elif family == "cluster_spt":
        # PHYSICAL STATES, not constructed. Genuine ground states of the cluster-Ising
        # Hamiltonian obtained by exact diagonalization, labelled by the string order
        # parameter sector. Every other family Haar-samples inside an eigenspace; this
        # one does not. Local import to avoid a circular dependency with qmlc.datasets.
        from qmlc.datasets import cluster_spt_ground_states
        d = cluster_spt_ground_states(n_qubits=n, n_states=n_states, seed=seed)
        L = d.metadata["paulis"]
        sup = tuple(d.metadata["support"])
        return Probe(d.states, np.asarray(d.y), n, sup,
                     tuple(L[q] for q in sup), family)
    elif family == "spt_string":
        # PHYSICALLY MOTIVATED, not invented for this study. The 1D cluster state is
        # the +1 eigenstate of the stabilizers K_i = Z_{i-1} X_i Z_{i+1}; products of
        # consecutive K_i give the STRING ORDER PARAMETER of the cluster-SPT phase,
        #     S(i,j) = Z_i X_{i+1} X_{i+2} ... X_{j-1} Z_j,
        # a textbook diagnostic of symmetry-protected topological order. The label is
        # its eigenvalue. Mathematically this is still a Pauli string, so it does not
        # test a new mechanism -- it tests whether the effect survives when the
        # observable is one physics already cared about, rather than one we planted.
        start = int(rng.integers(0, max(1, n - w)))
        support = tuple(range(start, start + w))
        letters = ("Z",) + ("X",) * (w - 2) + ("Z",) if w >= 2 else ("Z",)
    elif family == "ghz_phase":
        # NON-LOCAL BY CONSTRUCTION, and a deliberate falsification test.
        # |psi_phi> = (|0...0> + e^{i phi}|1...1>)/sqrt(2). Every proper subset has an
        # IDENTICAL reduced density matrix regardless of phi -- the information lives
        # only in the weight-n observable X^{(x)n}. Prediction, recorded before running:
        # the purity search MUST FAIL here, because purity of any proper subset is
        # phi-independent by construction. If it "succeeds", the method is detecting
        # an artifact and the battery's positive results are void.
        d_ = 2 ** n
        V = np.zeros((n_states, d_), complex)
        y = rng.integers(0, 2, n_states)
        for i, lab in enumerate(y):
            phi = (0.0 if lab == 0 else np.pi) + rng.normal(0.0, 0.25)
            V[i, 0] = 1.0 / np.sqrt(2.0)
            V[i, -1] = np.exp(1j * phi) / np.sqrt(2.0)
        return Probe(V, y.astype(int), n, tuple(range(n)), ("X",) * n, family)
    else:
        raise ValueError(f"unknown family {family!r}")

    # Above DENSE_MAX_N the dense operator is prohibitive (~10 GB at n=14), so the
    # projector is applied matrix-free. Below it, the original dense path runs
    # unchanged, so every published Part III number is byte-reproducible.
    DENSE_MAX_N = 12
    mf = None if n <= DENSE_MAX_N else (n, support, letters)
    P = _pauli_string_op(n, support, letters) if mf is None else None
    half = n_states // 2
    V = np.vstack([_haar_in_eigenspace(P, +1, half, rng, mf),
                   _haar_in_eigenspace(P, -1, n_states - half, rng, mf)])
    y = np.r_[np.zeros(half, int), np.ones(n_states - half, int)]
    idx = rng.permutation(n_states)
    return Probe(V[idx], y[idx], n, support, letters, family)


# --------------------------------------------------------------------------
# Purity, and its budgeted estimators
# --------------------------------------------------------------------------

def reduced_purity(psi: np.ndarray, n: int, S) -> float:
    if len(S) == 0:
        return 1.0
    keep = list(S); rest = [q for q in range(n) if q not in S]
    T = psi.reshape([2] * n).transpose(keep + rest).reshape(2 ** len(keep), -1)
    # Tr[rho^2] = ||G||_F^2 for the Gram G on EITHER side, since Tr[(TT*)^2] =
    # Tr[(T*T)^2]. Build it on the smaller side. Forming rho = TT* and then
    # rho @ rho is O(2^(3k)) and, at k = n (where elimination starts), needs a
    # 2^n x 2^n matrix -- 4.3 GB at n=14, which is why the sweep could not reach it.
    G = T @ T.conj().T if T.shape[0] <= T.shape[1] else T.conj().T @ T
    return float(np.real(np.vdot(G, G)))


def haar_purity(n: int, k: int) -> float:
    """Exact Page value of Tr[rho_S^2] for a Haar-random pure state."""
    return (2 ** k + 2 ** (n - k)) / (2 ** n + 1)


def ensemble_purity(states: np.ndarray, n: int, S) -> float:
    if len(S) == 0:
        return 1.0
    return float(np.mean([reduced_purity(p, n, S) for p in states]))


def ensemble_purity_fast(states: np.ndarray, n: int, S) -> float:
    """Batched mean reduced purity -- identical value to ensemble_purity, no Python loop.

    ensemble_purity calls reduced_purity once per state, so a contrast sweep costs
    C(n,w) * n_states interpreter-level calls (330k per seed at n=13, w=6, 192 states)
    and the overhead dominates. This does the whole ensemble in one reshape and one
    batched matmul. The original is left untouched so every published Part III number
    is produced by exactly the code that produced it.
    """
    if len(S) == 0:
        return 1.0
    keep = list(S); rest = [q for q in range(n) if q not in S]
    m = states.shape[0]
    T = states.reshape([m] + [2] * n).transpose([0] + [k + 1 for k in keep + rest])
    T = T.reshape(m, 2 ** len(keep), -1)
    # Gram on the smaller side, then Tr[rho^2] = ||G||_F^2
    if T.shape[1] <= T.shape[2]:
        G = T @ np.conjugate(np.transpose(T, (0, 2, 1)))
    else:
        G = np.conjugate(np.transpose(T, (0, 2, 1))) @ T
    return float(np.mean(np.real(np.sum(G * np.conjugate(G), axis=(1, 2)))))

def budgeted_excess(P_exact: float, n: int, k: int, M: int, mode: str, rng) -> float:
    """Purity excess over the Haar baseline, under a declared access model.

    two-copy   : SWAP test, P(+1) = (1 + Tr rho^2)/2, M Bernoulli draws. O(1/eps^2),
                 independent of |S|.
    single-copy: classical-shadow estimate, variance ~ 2^|S| / M.
    """
    if mode == "two-copy":
        p = float(np.clip((1.0 + P_exact) / 2.0, 0.0, 1.0))
        Ph = 2.0 * rng.binomial(M, p) / M - 1.0
    elif mode == "single-copy":
        Ph = P_exact + rng.normal(0.0, np.sqrt((2.0 ** k) / max(M, 1)))
    elif mode == "exact":
        Ph = P_exact
    else:
        raise ValueError(mode)
    return Ph / haar_purity(n, k)


# --------------------------------------------------------------------------
# Search algorithms  (axis D)
# --------------------------------------------------------------------------

def _scorer(states, n, M, mode, rng, cache):
    def score(S):
        S = tuple(sorted(S))
        if S not in cache:
            cache[S] = ensemble_purity(states, n, S)
        return budgeted_excess(cache[S], n, len(S), M, mode, rng)
    return score


def search_exhaustive(states, n, w, score, rng):
    return max(combinations(range(n), w), key=score)


def search_random_subset(states, n, w, score, rng, trials=None):
    """Control: blind guessing, given the same evaluation budget as greedy_forward.

    NOTE: at small n this control is degenerate -- n^2 = 64 draws covers most of
    C(8,4) = 70, so it matches exhaustive search and carries no information. It
    only becomes a meaningful control once C(n,w) >> n^2, i.e. n >= 10.
    """
    trials = trials or (n * n)
    best, bs = None, -np.inf
    for _ in range(trials):
        S = tuple(sorted(rng.permutation(n)[:w]))
        s = score(S)
        if s > bs:
            best, bs = S, s
    return best


def search_greedy_forward(states, n, w, score, rng):
    """Grow the subset one qubit at a time by best marginal excess."""
    S: list[int] = []
    for _ in range(w):
        cand = [(score(tuple(sorted(S + [q]))), q) for q in range(n) if q not in S]
        S.append(max(cand)[1])
    return tuple(sorted(S))


def search_elimination(states, n, w, score, rng):
    """Peel qubits whose removal preserves the purity excess, down to size w."""
    S = list(range(n))
    while len(S) > w:
        cand = [(score(tuple(x for x in S if x != q)), q) for q in S]
        S = [x for x in S if x != max(cand)[1]]
    return tuple(sorted(S))


def search_local(states, n, w, score, rng, iters=60):
    """Swap one qubit at a time from a random start; keep improving moves."""
    S = list(rng.permutation(n)[:w]); best = score(tuple(sorted(S)))
    for _ in range(iters):
        out = [q for q in range(n) if q not in S]
        if not out:
            break
        i, q = rng.integers(len(S)), out[rng.integers(len(out))]
        T = S.copy(); T[i] = q
        s = score(tuple(sorted(T)))
        if s > best:
            S, best = T, s
    return tuple(sorted(S))


SEARCHES = {"exhaustive": search_exhaustive, "greedy_forward": search_greedy_forward,
            "elimination": search_elimination, "local_search": search_local,
            "random_subset": search_random_subset}


def recover(probe: Probe, w: int, algo: str, M: int, mode: str, seed: int = 0):
    """Run one search. Returns (found_subset, correct?) -- complement counts as correct."""
    rng = np.random.default_rng(seed)
    score = _scorer(probe.states, probe.n, M, mode, rng, {})
    found = SEARCHES[algo](probe.states, probe.n, w, score, rng)
    truth = set(probe.support)
    comp = set(range(probe.n)) - truth
    return found, (set(found) == truth or set(found) == comp)
