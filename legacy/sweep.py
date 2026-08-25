# sweep.py
import pandas as pd
from experiment import run_classical_experiment, run_quantum_experiment, attach_complexity_card
from datasets.synthetic import generate_parity_dataset
from quantum.embeddings import zz_embedding


def run_sweep(
    datasets,
    classical_models=None,
    quantum_models=None,
    n_qubits=3,
    lam=1e-6,
    save_path=None,
    verbose=False,
):
    """
    Run experiments across multiple datasets and models.

    datasets: list of dicts, each with keys
              {"name": str, "generator": callable, "params": dict}
    classical_models: list of model names (["logistic", "svm_rbf", "kernel_ridge", "rff"])
    quantum_models: list of embedding functions (e.g. [zz_embedding])
    n_qubits: number of qubits for quantum runs
    save_path: optional CSV path to save results
    """

    if classical_models is None:
        classical_models = ["logistic", "svm_rbf", "kernel_ridge", "rff"]
    if quantum_models is None:
        quantum_models = [zz_embedding]

    results = []

    for dset in datasets:
        name = dset["name"]
        X, y = dset["generator"](**dset["params"])
        if verbose:
            print(f"\n=== Dataset: {name} ===")

        # ---- Classical models ----
        for model in classical_models:
            try:
                res = run_classical_experiment(X, y, model=model)
                res = attach_complexity_card(X, y, res)
                res["dataset"] = name
                results.append(res)
                if verbose:
                    print(f"  [OK] Classical {model}")
            except Exception as e:
                if verbose:
                    print(f"  [FAIL] Classical {model}: {e}")

        # ---- Quantum models ----
        X_angles = (X * 3.1415).astype(float)
        for qmodel in quantum_models:
            try:
                n_q = n_qubits if n_qubits is not None else X.shape[1]
                res = run_quantum_experiment(
                    X_angles,
                    y,
                    embedding_fn=qmodel,
                    n_qubits=n_q,
                    lam=lam,
                    verbose=verbose,
                )
                res = attach_complexity_card(X, y, res)
                res["dataset"] = name
                results.append(res)
                if verbose:
                    print(f"  [OK] Quantum {qmodel.__name__}")
            except Exception as e:
                if verbose:
                    print(f"  [FAIL] Quantum {qmodel.__name__}: {e}")

    # ---- Collect into DataFrame ----
    df = pd.DataFrame(results)

    if save_path:
        df.to_csv(save_path, index=False)
        if verbose:
            print(f"\nSaved results to {save_path}")

    return df