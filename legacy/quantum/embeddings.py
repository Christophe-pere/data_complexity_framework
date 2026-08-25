import pennylane as qml
import numpy as np

def zz_feature_map(n_qubits, reps=2):
    """Simple ZZFeatureMap-style embedding."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        # data re-uploading style
        for r in range(reps):
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def zz_embedding(x, n_qubits, reps=1):
    """
    ZZ-style embedding robust to feature <-> qubit mismatch.
    If len(x) < n_qubits, pads with zeros.
    If len(x) > n_qubits, truncates.
    """
    x = np.array(x)
    if len(x) < n_qubits:
        # pad with zeros
        x = np.pad(x, (0, n_qubits - len(x)), mode="constant")
    elif len(x) > n_qubits:
        # truncate extra features
        x = x[:n_qubits]

    for r in range(reps):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i + 1])

    # Optionally you could add global rotations or more entangling layers