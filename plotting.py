# plotting.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_accuracy_vs_effective_dimension(df, save_path=None):
    """
    Scatter plot: accuracy vs effective dimension (classical & quantum).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for t, marker, color in [("classical", "o", "blue"), ("quantum", "s", "green")]:
        subdf = df[df["type"] == t]
        ax.scatter(
            subdf["effective_dimension"],
            subdf["accuracy"],
            marker=marker,
            label=t,
            alpha=0.7,
            color=color,
        )

    ax.set_xlabel("Effective Dimension")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Effective Dimension")
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_complexity_vs_induced(df, save_path=None):
    """
    Scatter plot: dataset classical complexity (C_data) vs induced quantum complexity (C_ind).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for t, marker, color in [("classical", "o", "red"), ("quantum", "s", "green")]:
        subdf = df[df["type"] == t]
        x = [c["C_data"] for c in subdf["complexity_card"]]
        if t == "quantum":
            y = [c["C_ind"] for c in subdf["complexity_induced"]]
        else:
            y = subdf["effective_dimension"]  # classical proxy
        ax.scatter(x, y, marker=marker, label=t, alpha=0.7, color=color)

    ax.set_xlabel("Classical Data Complexity (C_data)")
    ax.set_ylabel("Induced Complexity (C_ind / d_eff)")
    ax.set_title("Complexity Landscape")
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_dataset_comparison(df, dataset_name, save_path=None):
    """
    Bar plot: accuracy of different models for one dataset.
    """
    subdf = df[df["dataset"] == dataset_name]

    labels = []
    accs = []
    colors = []
    for _, row in subdf.iterrows():
        labels.append(f"{row['type']}-{row['model']}")
        accs.append(row["accuracy"] if row["accuracy"] is not None else 0.0)
        colors.append("blue" if row["type"] == "classical" else "green")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, accs, color=colors)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Model Comparison on {dataset_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_accuracy_heatmap(df, save_path=None):
    """
    Heatmap of accuracy (datasets x models).
    Works for both classical & quantum models.
    """

    # pivot table: rows = dataset, cols = model (type-model), values = accuracy
    df_heat = df.copy()
    df_heat["model_label"] = df_heat["type"] + "-" + df_heat["model"]
    pivot = pd.pivot_table(
        df_heat,
        values="accuracy",
        index="dataset",
        columns="model_label",
        aggfunc=np.mean,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(pivot.values, cmap="viridis", vmin=0, vmax=1)

    # set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="left")
    ax.set_yticklabels(pivot.index)

    ax.set_title("Accuracy Heatmap (Datasets x Models)", pad=20)
    fig.colorbar(cax, ax=ax, label="Accuracy")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_complexity_heatmap(df, save_path=None):
    """
    Heatmap of dataset complexity vs model-induced complexity.
    - Rows = datasets
    - Cols = models
    - Values = induced complexity (C_ind for quantum, effective_dimension for classical)
    """

    df_cplx = []

    for _, row in df.iterrows():
        label = row["type"] + "-" + row["model"]

        if row["type"] == "quantum":
            val = row["complexity_induced"]["C_ind"]
        else:
            val = row["effective_dimension"] if row["effective_dimension"] is not None else np.nan

        df_cplx.append({"dataset": row["dataset"], "model_label": label, "complexity": val})

    pivot = pd.pivot_table(
        pd.DataFrame(df_cplx),
        values="complexity",
        index="dataset",
        columns="model_label",
        aggfunc=np.mean,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(pivot.values, cmap="plasma")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="left")
    ax.set_yticklabels(pivot.index)

    ax.set_title("Complexity Heatmap (Datasets x Models)", pad=20)
    fig.colorbar(cax, ax=ax, label="Induced Complexity (C_ind / d_eff)")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_joint_heatmaps(df, save_path=None):
    """
    Side-by-side heatmaps:
      - Left: Accuracy (datasets x models)
      - Right: Complexity (datasets x models)
    """

    # --- Accuracy pivot ---
    df_acc = df.copy()
    df_acc["model_label"] = df_acc["type"] + "-" + df_acc["model"]
    pivot_acc = pd.pivot_table(
        df_acc,
        values="accuracy",
        index="dataset",
        columns="model_label",
        aggfunc=np.mean,
    )

    # --- Complexity pivot ---
    df_cplx = []
    for _, row in df.iterrows():
        label = row["type"] + "-" + row["model"]
        if row["type"] == "quantum":
            val = row["complexity_induced"]["C_ind"]
        else:
            val = row["effective_dimension"] if row["effective_dimension"] is not None else np.nan
        df_cplx.append({"dataset": row["dataset"], "model_label": label, "complexity": val})

    pivot_cplx = pd.pivot_table(
        pd.DataFrame(df_cplx),
        values="complexity",
        index="dataset",
        columns="model_label",
        aggfunc=np.mean,
    )

    # --- Plot side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy heatmap
    im1 = axes[0].matshow(pivot_acc.values, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(pivot_acc.columns)))
    axes[0].set_yticks(range(len(pivot_acc.index)))
    axes[0].set_xticklabels(pivot_acc.columns, rotation=45, ha="left")
    axes[0].set_yticklabels(pivot_acc.index)
    axes[0].set_title("Accuracy Heatmap")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Accuracy")

    # Complexity heatmap
    im2 = axes[1].matshow(pivot_cplx.values, cmap="plasma")
    axes[1].set_xticks(range(len(pivot_cplx.columns)))
    axes[1].set_yticks(range(len(pivot_cplx.index)))
    axes[1].set_xticklabels(pivot_cplx.columns, rotation=45, ha="left")
    axes[1].set_yticklabels(pivot_cplx.index)
    axes[1].set_title("Complexity Heatmap")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="C_ind / d_eff")

    plt.suptitle("Joint Heatmaps: Accuracy vs Complexity", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_quantum_advantage_vs_complexity(df, save_path=None):
    """
    Scatter plot:
      x-axis = dataset complexity (C_data)
      y-axis = (Quantum Accuracy - Best Classical Accuracy)

    Shows where quantum models start to win over classical baselines.
    """

    # Collect results
    advantage_points = []
    for dataset in df["dataset"].unique():
        subdf = df[df["dataset"] == dataset]

        # dataset complexity (all models share the same C_data)
        c_data = subdf.iloc[0]["complexity_card"]["C_data"]

        # best classical accuracy
        classical_acc = subdf[subdf["type"] == "classical"]["accuracy"].dropna()
        best_classical = classical_acc.max() if len(classical_acc) > 0 else 0.0

        # quantum accuracies
        quantum_accs = subdf[subdf["type"] == "quantum"]["accuracy"]

        for acc in quantum_accs:
            if acc is not None:
                advantage = acc - best_classical
                advantage_points.append((c_data, advantage, dataset))

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    if advantage_points:
        xs, ys, labels = zip(*advantage_points)
        sc = ax.scatter(xs, ys, c=ys, cmap="coolwarm", s=80, edgecolors="k")
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y + 0.01, label, fontsize=8, ha="center")
    else:
        ax.text(0.5, 0.5, "No quantum accuracy values found", ha="center")

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Dataset Complexity (C_data)")
    ax.set_ylabel("Quantum Advantage (Δ Accuracy)")
    ax.set_title("Quantum Advantage vs Dataset Complexity")
    fig.colorbar(sc, ax=ax, label="Δ Accuracy")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()