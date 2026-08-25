# classical/baselines.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy.linalg import eigvalsh
from sklearn.linear_model import RidgeClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from classical.ntk import empirical_ntk_trace, effective_dimension_from_eigs

# -------- Utility: effective dimension for kernels --------
def effective_dimension(evals, lam):
    evals = np.array(evals, dtype=float)
    return np.sum(evals / (evals + lam))


# -------- Baselines --------
def logistic_baseline(X, y, test_size=0.3, random_state=42):
    """
    Train logistic regression and return test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}


def svm_rbf_baseline(X, y, gamma="scale", C=1.0, test_size=0.3, random_state=42):
    """
    Train SVM with RBF kernel and return test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = SVC(kernel="rbf", gamma=gamma, C=C)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}


def kernel_ridge_baseline(X, y, gamma=1.0, alpha=1e-3, lam=1e-6, test_size=0.3, random_state=42):
    """
    Train Kernel Ridge with RBF kernel, return test accuracy and effective dimension.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit Kernel Ridge
    clf = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    clf.fit(X_train, y_train)
    y_pred = np.sign(clf.predict(X_test))  # outputs real values → threshold
    y_pred = (y_pred > 0).astype(int)

    acc = accuracy_score(y_test, y_pred)

    # Effective dimension (like in QML kernel benchmarks)
    K_train = rbf_kernel(X_train, X_train, gamma=gamma)
    evals = eigvalsh(K_train)
    d_eff = effective_dimension(evals, lam)

    return {"accuracy": acc, "effective_dimension": float(d_eff)}

def rff_baseline(
    X,
    y,
    gamma=1.0,
    D=500,
    alpha=1e-3,
    lam=1e-6,
    test_size=0.3,
    random_state=42,
):
    """
    Random Fourier Features baseline (approximates RBF kernel).
    Returns test accuracy and approximate effective dimension.

    Parameters:
      gamma : RBF kernel width parameter
      D     : number of random Fourier features
      alpha : ridge regularization
      lam   : regularization for effective dimension
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]

    # Draw random projection matrix and bias
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(n_features, D))
    b = rng.uniform(0, 2 * np.pi, size=D)

    def rff_transform(X):
        Z = np.sqrt(2 / D) * np.cos(X @ W + b)
        return Z

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    Z_train, Z_test = rff_transform(X_train), rff_transform(X_test)

    # Ridge classifier in feature space
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    acc = accuracy_score(y_test, y_pred)

    # Approximate effective dimension via Gram matrix in feature space
    G = Z_train @ Z_train.T
    evals = np.linalg.eigvalsh(G)
    d_eff = np.sum(evals / (evals + lam))

    return {"accuracy": acc, "effective_dimension": float(d_eff), "D": D}

# add NN 
# add transformers 
# add TN 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64], output_dim=2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def nn_baseline(X, y, hidden_dims=[64, 64], epochs=20, lr=1e-3, batch_size=32, test_size=0.3, random_state=42):
    """
    Simple feedforward neural network baseline.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X.shape[1], hidden_dims, output_dim=len(np.unique(y))).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    # training loop
    for epoch in range(epochs):
        idx = torch.randperm(X_train_t.size(0))
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i + batch_size]
            xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    # choose subset size for NTK (tradeoff)
    ntk_samples = min(200, X_train.shape[0])  # or pass as param
    # compute on training inputs (or whole X)
    try:
        NTK, evals = empirical_ntk_trace(model, X_train, device=device,
                                         max_samples=ntk_max_samples,
                                         batch_size=ntk_batch_size,
                                         verbose=verbose)
        d_eff = effective_dimension_from_eigs(evals, lam=lam)
    except Exception as e:
        if verbose: print("[WARN] NTK computation failed:", e)
        d_eff, evals = None, None

    return {"accuracy": acc, "effective_dimension": d_eff, "ntk_eigs": evals}

class TinyTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=32, num_heads=4, num_layers=2, output_dim=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, features)
        # reshape to (seq_len, batch, model_dim)
        x = self.embedding(x).unsqueeze(0)
        h = self.encoder(x)
        h = h.mean(dim=0)  # pool
        return self.classifier(h)


def transformer_baseline(X, y, model_dim=32, heads=4, layers=2,
                         epochs=20, lr=1e-3, batch_size=32, test_size=0.3, random_state=42):
    """
    Tiny transformer baseline for tabular data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer(X.shape[1], model_dim, heads, layers, output_dim=len(np.unique(y))).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    for epoch in range(epochs):
        idx = torch.randperm(X_train_t.size(0))
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i + batch_size]
            xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)

    try:
        NTK, evals = empirical_ntk_trace(model, X_train, device=device,
                                         max_samples=ntk_max_samples,
                                         batch_size=ntk_batch_size,
                                         verbose=verbose)
        d_eff = effective_dimension_from_eigs(evals, lam=lam)
    except Exception as e:
        if verbose: print("[WARN] NTK computation failed:", e)
        d_eff, evals = None, None

    return {"accuracy": acc, "effective_dimension": d_eff, "ntk_eigs": evals}


class MPSClassifier(nn.Module):
    def __init__(self, input_dim, bond_dim=10, output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tensors = nn.ParameterList([
            nn.Parameter(torch.randn(bond_dim, 2, bond_dim)) for _ in range(input_dim)
        ])
        self.readout = nn.Linear(bond_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, features) with binary features
        batch_size = x.shape[0]
        state = torch.eye(self.tensors[0].shape[0]).unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)

        for i in range(self.input_dim):
            site = torch.stack([1 - x[:, i], x[:, i]], dim=-1)  # map feature to [0,1] basis
            T = self.tensors[i]  # (bond, 2, bond)
            contracted = torch.einsum("bij,bj,kjk->bik", state, site, T)
            state = contracted

        pooled = state.mean(dim=(1, 2))
        return self.readout(pooled)


def mps_baseline(X, y, bond_dim=10, epochs=20, lr=1e-3, batch_size=32, test_size=0.3, random_state=42):
    """
    Matrix Product State (MPS) classifier baseline.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPSClassifier(X.shape[1], bond_dim=bond_dim, output_dim=len(np.unique(y))).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    for epoch in range(epochs):
        idx = torch.randperm(X_train_t.size(0))
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i + batch_size]
            xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)

    try:
        NTK, evals = empirical_ntk_trace(model, X_train, device=device,
                                         max_samples=ntk_max_samples,
                                         batch_size=ntk_batch_size,
                                         verbose=verbose)
        d_eff = effective_dimension_from_eigs(evals, lam=lam)
    except Exception as e:
        if verbose: print("[WARN] NTK computation failed:", e)
        d_eff, evals = None, None

    return {"accuracy": acc, "effective_dimension": d_eff, "ntk_eigs": evals}