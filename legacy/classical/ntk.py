# classical/ntk.py
import torch
import numpy as np
from numpy.linalg import eigh

def _flatten_gradients(param_grads):
    """Flatten a list of gradient tensors into a single 1D numpy vector."""
    vecs = []
    for g in param_grads:
        if g is None:
            continue
        vecs.append(g.detach().cpu().reshape(-1))
    if len(vecs) == 0:
        return np.zeros(0, dtype=float)
    return np.concatenate([v.numpy() for v in vecs])

def empirical_ntk_trace(model, X, device=None, max_samples=None, batch_size=32, verbose=False):
    """
    Compute empirical NTK (trace over outputs) using per-sample parameter gradients.
    Returns: (NTK_matrix (n x n), eigenvalues (numpy array))
    Notes:
      - model: PyTorch model that returns logits (shape [batch, n_classes])
      - X: numpy array or tensor of inputs (n_samples, input_dim)
      - max_samples: if not None, randomly subsample up to this many points
      - This computes grad of sum(logits) per sample -> a gradient vector g_i
        and NTK_{ij} = g_i dot g_j.
    """
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if isinstance(X, np.ndarray):
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X_t = X.to(device)

    n = X_t.shape[0]
    if max_samples is not None and max_samples < n:
        idx = np.random.default_rng(123).choice(n, size=max_samples, replace=False)
        X_t = X_t[idx]
        n = X_t.shape[0]

    # collect flattened gradients per sample
    grads = []
    # disable grad for parameters except during backward
    for p in model.parameters():
        p.requires_grad_(True)

    for i in range(0, n, batch_size):
        xb = X_t[i : i + batch_size]
        xb = xb.clone().detach().requires_grad_(True)
        logits = model(xb)  # shape (b, C)

        # We'll compute gradient of sum across outputs per sample:
        # for each sample in batch compute scalar s = sum(logits[sample, :])
        # grad_s = grad_theta(s)
        for b in range(logits.shape[0]):
            model.zero_grad(set_to_none=True)
            s = logits[b].sum()   # scalar
            s.backward(retain_graph=True)

            # collect parameter grads
            param_grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
            flat = _flatten_gradients(param_grads)  # numpy vector
            grads.append(flat)

        # free graph
        del logits

    if len(grads) == 0:
        raise RuntimeError("No gradients collected; check model / X inputs.")

    # Stack into matrix G of shape (n_samples, n_params)
    G = np.stack(grads, axis=0)  # shape (n, P)
    # Compute NTK = G @ G^T
    NTK = G @ G.T
    # symmetrize numerically
    NTK = 0.5 * (NTK + NTK.T)

    # eigenvalues (ascending)
    evals = eigh(NTK, eigvals_only=True)
    # flip to descending for usual convention
    evals = evals[::-1]
    return NTK, evals

def effective_dimension_from_eigs(eigs, lam=1e-6):
    eigs = np.array(eigs, dtype=float)
    return float(np.sum(eigs / (eigs + lam)))