# experiment.py
from core import compute_complexity_card
from classical import baselines
from quantum.complexity import induced_quantum_complexity


def run_classical_experiment(X, y, model="svm_rbf", params=None, lam=1e-6, random_state=42):
    """
    Run a classical baseline and return results in unified schema.
    """
    if params is None:
        params = {}

    if model == "logistic":
        res = baselines.logistic_baseline(X, y, random_state=random_state, **params)
    elif model == "svm_rbf":
        res = baselines.svm_rbf_baseline(X, y, random_state=random_state, **params)
    elif model == "kernel_ridge":
        res = baselines.kernel_ridge_baseline(X, y, lam=lam, random_state=random_state, **params)
    elif model == "rff":
        res = baselines.rff_baseline(X, y, lam=lam, random_state=random_state, **params)
    elif model == "nn":
        res = baselines.nn_baseline(X, y, **params)
    elif model == "transformer":
        res = baselines.transformer_baseline(X, y, **params)
    elif model == "mps":
        res = baselines.mps_baseline(X, y, **params)
    else:
        raise ValueError(f"Unknown classical model: {model}")

    out = {
        "model": model,
        "type": "classical",
        "accuracy": res.get("accuracy", None),
        "effective_dimension": res.get("effective_dimension", None),
        "details": res,
    }
    return out


def run_quantum_experiment(X_angles, y, embedding_fn, n_qubits, lam=1e-6,
                           qfi_sample_count=10, entropy_bipartitions=None,
                           weights=None, verbose=False, return_logs=True):
    """
    Run a quantum complexity experiment with unified schema.
    X_angles: features encoded as angles (shape [n_samples, n_qubits])
    """
    qres, logs = induced_quantum_complexity(
        X_angles,
        embedding_fn=embedding_fn,
        n_qubits=n_qubits,
        lam=lam,
        qfi_sample_count=qfi_sample_count,
        entropy_bipartitions=entropy_bipartitions,
        weights=weights,
        verbose=verbose,
        return_logs=True,
    )

    out = {
        "model": embedding_fn.__name__,
        "type": "quantum",
        "accuracy": None,  # (can add variational training later)
        "effective_dimension": qres["Effective_dimension"],
        "complexity_induced": qres,
        "logs": logs,
    }
    return out


def attach_complexity_card(X, y, result_dict):
    """
    Compute classical data complexity and attach to result dict.
    """
    card = compute_complexity_card(X, y)
    result_dict["complexity_card"] = card
    return result_dict