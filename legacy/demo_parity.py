from datasets.synthetic import generate_parity_dataset
from quantum.embeddings import zz_embedding
from plotting import plot_quantum_advantage_vs_complexity
from experiment import run_classical_experiment, run_quantum_experiment, attach_complexity_card
from sweep import run_sweep
from plotting import (
    plot_accuracy_vs_effective_dimension, 
    plot_complexity_vs_induced, 
    plot_dataset_comparison,
    plot_accuracy_heatmap,
    plot_complexity_heatmap,
    plot_joint_heatmaps
)

if __name__ == "__main__":
    # parity dataset
    '''X, y = generate_parity_dataset(n_bits=6, n_samples=500, noise=0.05)
    
    # ---- Classical baseline ----
    cls_res = run_classical_experiment(X, y, model="svm_rbf", params={"gamma": 0.5})
    cls_res = attach_complexity_card(X, y, cls_res)
    print("Classical Result:", cls_res)

    # ---- Quantum embedding ----
    # encode binary data as angles
    X_angles = (X * 3.1415).astype(float)
    q_res = run_quantum_experiment(X_angles, y, embedding_fn=zz_embedding, n_qubits=6, verbose=True)
    q_res = attach_complexity_card(X, y, q_res)
    print("Quantum Result:", q_res)'''

    datasets = [
        {"name": "parity_4", "generator": generate_parity_dataset, "params": {"n_bits": 4, "n_samples": 200, "noise": 0.05}},
        {"name": "parity_6", "generator": generate_parity_dataset, "params": {"n_bits": 6, "n_samples": 500, "noise": 0.1}},
    ]

    df = run_sweep(datasets, quantum_models=[zz_embedding], n_qubits=None, verbose=True, save_path="results.csv")
    print(df.head())

    #plot_accuracy_vs_effective_dimension(df)
    #plot_complexity_vs_induced(df)
    #plot_dataset_comparison(df, dataset_name="parity_4")
    #plot_accuracy_heatmap(df)
    #plot_complexity_heatmap(df)
    #plot_joint_heatmaps(df)
    #plot_quantum_advantage_vs_complexity(df)

    X, y = generate_parity_dataset(n_bits=6, n_samples=500, noise=0.05)
    res = run_classical_experiment(X, y, model="nn", params={"hidden_dims": [64, 64]})
    print(res)