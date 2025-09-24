import pennylane as qml
#import numpy as np
from pennylane import numpy as np
from numpy.linalg import eigvalsh
import math

'''def quantum_kernel_matrix(X, circuit, n_qubits):
    """Compute kernel matrix from state overlaps <phi(x)|phi(x')>."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def inner_product(x1, x2):
        # Encode first state
        for i in range(n_qubits):
            qml.RX(x1[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i + 1])

        # Overlap with second state
        qml.adjoint(circuit)(x2)
        return qml.probs(wires=range(n_qubits))  # full vector

    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            probs = inner_product(X[i], X[j])
            val = probs[0]  # probability of |0...0>
            K[i, j] = K[j, i] = val
    return K'''


'''def effective_dimension(evals, lam):
    return np.sum(evals / (evals + lam))'''


def induced_quantum_complexity(
    X,
    embedding_fn,
    n_qubits,
    lam=1e-6,
    qfi_sample_count=10,
    entropy_bipartitions=None,
    weights=None,
    dev_name="default.qubit",
    verbose=False,
    return_logs=False,
):
    """
    Compute induced quantum complexity:
      - kernel eigenvalues + effective dimension
      - QFI spread (averaged over up to qfi_sample_count points)
      - entanglement entropies (averaged over same points)

    embedding_fn: callable embedding_fn(x, n_qubits)
    weights: dict to combine components into a single C_ind
    verbose: print warnings/info
    return_logs: if True, returns (components, logs) instead of components only
    """

    logs = {"qfi_failures": [], "entropy_failures": []}

    # ---- Kernel part ----
    K = quantum_kernel_matrix(X, embedding_fn, n_qubits, dev_name=dev_name)
    evals = np.real(eigvalsh(K))
    d_eff = effective_dimension(evals, lam)

    # ---- Choose sample indices ----
    n = len(X)
    rng = np.random.default_rng(123)
    samples = np.unique(rng.integers(0, n, size=min(qfi_sample_count, n)))

    # ---- QFI proxies ----
    qfi_traces, qfi_conds = [], []
    for idx in samples:
        r = qfi_metric_tensor_spread(X[idx], embedding_fn, n_qubits, dev_name=dev_name)
        trace_val = r.get("metric_trace", 0.0)
        cond_val = r.get("metric_cond", float("inf"))
        if trace_val == 0.0 and not np.isfinite(cond_val):
            logs["qfi_failures"].append(int(idx))
            if verbose:
                print(f"[WARN] QFI computation failed for sample {idx} → using defaults.")
        qfi_traces.append(trace_val)
        qfi_conds.append(cond_val)

    if len(qfi_traces) > 0:
        qfi_trace_mean = float(np.mean(qfi_traces))
        finite_conds = [c for c in qfi_conds if np.isfinite(c)]
        qfi_cond_mean = float(np.mean(finite_conds)) if len(finite_conds) > 0 else float("inf")
    else:
        qfi_trace_mean = 0.0
        qfi_cond_mean = float("inf")
        logs["qfi_failures"].append("all")

    # ---- Entanglement proxies ----
    entropies = []
    for idx in samples:
        try:
            ep = entanglement_proxies(
                X[idx],
                embedding_fn,
                n_qubits,
                bipartitions=entropy_bipartitions,
                dev_name=dev_name,
            )
            ent_vals = list(ep["entropies"].values())
            entropies.append(np.mean(ent_vals))
        except Exception:
            entropies.append(0.0)
            logs["entropy_failures"].append(int(idx))
            if verbose:
                print(f"[WARN] Entanglement computation failed for sample {idx} → using 0.0.")

    entropy_mean = float(np.mean(entropies)) if len(entropies) > 0 else 0.0

    # ---- Assemble components ----
    components = {
        "Kernel_eigenvalues": evals,
        "Effective_dimension": float(d_eff),
        "QFI_trace_mean": qfi_trace_mean,
        "QFI_cond_mean": qfi_cond_mean,
        "Entanglement_entropy_mean": entropy_mean,
    }

    # ---- Combine into single scalar C_ind ----
    if weights is None:
        weights = {"kernel": 1.0, "qfi": 1.0, "entropy": 1.0}

    kernel_score = float(np.log1p(d_eff))  # effective dimension proxy
    qfi_score = qfi_trace_mean
    entropy_score = entropy_mean

    C_ind = (
        weights["kernel"] * kernel_score
        + weights["qfi"] * qfi_score
        + weights["entropy"] * entropy_score
    )

    components["C_ind"] = float(C_ind)
    components["weights"] = weights

    if verbose:
        print("[INFO] Induced complexity computed.")
        for k, v in components.items():
            if k == "Kernel_eigenvalues":
                print(f"  {k}: {v[:5]} ...")
            else:
                print(f"  {k}: {v}")

    return (components, logs) if return_logs else components


# ----- Kernel matrix (uses embedding function above) -----
def quantum_kernel_matrix(X, embedding_fn, n_qubits, dev_name="default.qubit"):
    """
    Compute kernel K_ij = |<0| U(x_i)^\dagger U(x_j) |0>|^2
    embedding_fn: callable that applies gates when called inside a QNode: embedding_fn(x, n_qubits, reps)
    """
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev)
    def overlap_prob(x1, x2):
        # prepare |phi(x1)>
        embedding_fn(x1, n_qubits)
        # apply adjoint of embedding to encode x2 in reverse -> overlap with |0>
        qml.adjoint(embedding_fn)(x2, n_qubits)
        # return probabilities; probability of all-zeros is index 0
        return qml.probs(wires=range(n_qubits))

    n = len(X)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            probs = overlap_prob(X[i], X[j])
            val = np.real(probs[0])  # probability of |0...0>
            K[i, j] = K[j, i] = val
    return K


# ----- Effective dimension -----
def effective_dimension(evals, lam):
    return np.sum(evals / (evals + lam))


# ----- QFI spread (using metric tensor) -----
def qfi_metric_tensor_spread(params, embedding_fn, n_qubits, dev_name="default.qubit"):
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def _tmp_qnode(x):
        embedding_fn(x, n_qubits)
        return qml.state()

    metric_fn = qml.metric_tensor(_tmp_qnode)

    try:
        G = metric_fn(params)
    except Exception:
        # if it completely fails
        return {"metric_trace": 0.0, "metric_eigs": np.array([0.0]), "metric_cond": float("inf")}

    G = np.array(G, dtype=float)

    # handle cases based on dimensions / emptiness
    if G.size == 0:
        trace = 0.0
        eigs = np.array([0.0])
    elif G.ndim == 0:  # scalar
        trace = float(G)
        eigs = np.array([G])
    elif G.ndim == 1:  # vector
        trace = float(np.sum(G))
        eigs = G
    else:  # matrix
        trace = float(np.trace(G))
        eigs = np.linalg.eigvalsh(G)

    # regularize eigenvalues
    eigs = np.maximum(eigs, 1e-12)
    cond = float(eigs.max() / eigs.min()) if eigs.min() > 0 else float("inf")

    return {"metric_trace": trace, "metric_eigs": eigs, "metric_cond": cond}



# ----- Entanglement entropy (single bipartition) -----
def bipartition_entanglement_entropy(statevector, sysA_wires):
    """
    Compute von Neumann entropy S(rho_A) for subsystem A defined by sysA_wires.
    statevector is a 1D complex numpy array of length 2^n.
    sysA_wires: list/tuple of wire indices for subsystem A
    """
    # total number of qubits
    n = int(math.log2(len(statevector)))
    sysA = list(sysA_wires)
    sysB = [i for i in range(n) if i not in sysA]

    # reshape into tensor of shape (2,2,...,2)
    psi = np.reshape(statevector, [2] * n)

    # move axes so that sysA axes come first, sysB after
    axes = sysA + sysB
    psi_perm = np.transpose(psi, axes=axes)

    dimA = 2 ** len(sysA)
    dimB = 2 ** len(sysB)
    # reshape into (dimA, dimB)
    psi_matrix = np.reshape(psi_perm, (dimA, dimB))

    # reduced density matrix rho_A = psi_matrix @ psi_matrix^\dagger
    rho_A = np.dot(psi_matrix, np.conjugate(psi_matrix).T)

    # eigenvalues of rho_A
    eigs = np.linalg.eigvalsh(rho_A)
    # clip tiny negative numerical artifacts
    eigs = np.clip(eigs, 0.0, 1.0)
    # compute von Neumann entropy
    nonzero = eigs[eigs > 1e-12]
    S = -np.sum(nonzero * np.log2(nonzero))
    return S, eigs


# ----- Convenience: compute a QNode state and measure entropies for several bipartitions -----
def entanglement_proxies(params, embedding_fn, n_qubits, bipartitions=None, dev_name="default.qubit"):
    """
    Run embedding on 'params' to get statevector, then compute entropies for given bipartitions.
    bipartitions: list of lists, each inner list is wires in subsystem A.
    Returns dictionary with entropies and eigenvalues.
    """
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev)
    def _state_qnode(x):
        embedding_fn(x, n_qubits)
        return qml.state()

    state = _state_qnode(params)
    out = {}
    if bipartitions is None:
        # default: split half-half
        mid = n_qubits // 2
        bipartitions = [list(range(mid)), list(range(mid, n_qubits))]

    entropies = {}
    eigs_dict = {}
    for idx, A in enumerate(bipartitions):
        S, eigs = bipartition_entanglement_entropy(state, A)
        entropies[f"A{idx}"] = float(S)
        eigs_dict[f"A{idx}_eigs"] = eigs
    out["entropies"] = entropies
    out["eigs"] = eigs_dict
    return out

