"""Classical data complexity metrics.

For each quantity we provide either:
  - one well-defined estimator, or
  - paired exact/proxy estimators for benchmarking.

All metrics consume a numpy array of shape (n, d) and return a MetricResult.
"""
from __future__ import annotations

import gzip
import lzma
import zlib
from math import lgamma, log, pi
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import digamma
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors

from .base import BaseMetric


# ---------------------------------------------------------------------------
# Differential entropy
# ---------------------------------------------------------------------------

class EntropyKDE(BaseMetric):
    """Differential entropy via KDE plug-in.

    Reasonable for d <= ~6 with enough samples; degrades in higher dims.
    Use as the 'reference' estimator on tractable cases.
    """

    name = "differential_entropy"
    estimator = "kde"

    def __init__(self, bandwidth: str | float = "scott"):
        self.bandwidth = bandwidth

    def _compute(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        kde = gaussian_kde(X.T, bw_method=self.bandwidth)
        log_p = kde.logpdf(X.T)
        h = float(-np.mean(log_p))
        # Bootstrap-free MC stderr from the log-density samples
        unc = float(np.std(log_p, ddof=1) / np.sqrt(len(X)))
        return h, unc, {"bandwidth": self.bandwidth, "d": X.shape[1], "n": X.shape[0]}


class EntropyKL(BaseMetric):
    """Kozachenko-Leonenko k-NN differential entropy estimator.

    Scales to higher d than KDE; the canonical 'proxy' here.
    """

    name = "differential_entropy"
    estimator = "knn_kl"

    def __init__(self, k: int = 4):
        self.k = k

    def _compute(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n, d = X.shape
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        dist, _ = nbrs.kneighbors(X)
        rho = dist[:, self.k]  # k-th NN distance (excluding self)
        # Volume of unit ball in d-dim Euclidean space
        log_vol = (d / 2) * log(pi) - lgamma(d / 2 + 1)
        h = float(
            -digamma(self.k) + digamma(n) + log_vol + d * np.mean(np.log(rho + 1e-12))
        )
        return h, None, {"k": self.k, "d": d, "n": n}


# ---------------------------------------------------------------------------
# Intrinsic dimension
# ---------------------------------------------------------------------------

class IntrinsicDimPCA(BaseMetric):
    """Effective dim from covariance spectrum (Eq. 1 in Part I).

    d_eff = (Σ λᵢ)² / Σ λᵢ². Globally linear; biased high on curved manifolds.
    """

    name = "intrinsic_dim"
    estimator = "pca_effective"

    def _compute(self, X):
        X = np.asarray(X, dtype=float)
        Xc = X - X.mean(0, keepdims=True)
        # Use SVD on the centered matrix; eigvals of cov are sigmas^2 / (n-1)
        s = np.linalg.svd(Xc, compute_uv=False)
        lam = (s ** 2) / max(len(X) - 1, 1)
        d_eff = float((lam.sum() ** 2) / np.maximum((lam ** 2).sum(), 1e-30))
        return d_eff, None, {"n_components_kept": int(np.sum(lam > 1e-12))}


class IntrinsicDimTwoNN(BaseMetric):
    """Facco et al. (2017) TwoNN estimator.

    Uses ratios of 1st and 2nd nearest-neighbor distances. Robust on curved
    manifolds where PCA fails.
    """

    name = "intrinsic_dim"
    estimator = "twonn"

    def __init__(self, frac_keep: float = 0.9):
        self.frac_keep = frac_keep

    def _compute(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        dist, _ = nbrs.kneighbors(X)
        r1, r2 = dist[:, 1], dist[:, 2]
        # Drop degenerate samples (r1 == 0 or r2 == 0)
        ok = (r1 > 0) & (r2 > r1)
        mu = r2[ok] / r1[ok]
        mu_sorted = np.sort(mu)
        keep = int(self.frac_keep * len(mu_sorted))
        mu_sorted = mu_sorted[:keep]
        # F(mu_i) ~ i / N. Slope of log(mu) vs -log(1 - F) gives d.
        F = np.arange(1, len(mu_sorted) + 1) / (len(mu_sorted) + 1)
        x = np.log(mu_sorted)
        y = -np.log(1.0 - F)
        # Linear regression through origin
        d = float(np.sum(x * y) / np.sum(x * x))
        return d, None, {"n_used": int(ok.sum()), "frac_keep": self.frac_keep}


# ---------------------------------------------------------------------------
# Compressibility (Kolmogorov complexity proxy)
# ---------------------------------------------------------------------------

class CompressionRatio(BaseMetric):
    """Lossless compression ratio C(D) = compressed / raw.

    Lower => more redundant (lower Kolmogorov complexity).
    """

    name = "compression_ratio"
    estimator = "gzip"

    def __init__(self, codec: str = "gzip"):
        self.codec = codec

    def _compute(self, data):
        if isinstance(data, np.ndarray):
            raw = data.tobytes()
        elif isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raise TypeError(f"Unsupported type for compression: {type(data)}")

        if self.codec == "gzip":
            comp = gzip.compress(raw, compresslevel=9)
        elif self.codec == "zlib":
            comp = zlib.compress(raw, level=9)
        elif self.codec == "lzma":
            comp = lzma.compress(raw, preset=9)
        else:
            raise ValueError(f"unknown codec {self.codec!r}")

        ratio = len(comp) / max(len(raw), 1)
        return ratio, None, {"codec": self.codec, "raw_bytes": len(raw), "comp_bytes": len(comp)}


# ---------------------------------------------------------------------------
# Persistent homology / topological complexity
# ---------------------------------------------------------------------------

class PersistentHomology(BaseMetric):
    """Vietoris-Rips persistent homology summary.

    Returns total persistence weighted by dimension (Eq. 3-style Ctop).
    Metadata carries per-dimension Betti numbers (long-lived features) and
    raw barcodes for downstream use.
    """

    name = "topological_complexity"
    estimator = "vietoris_rips"

    def __init__(
        self,
        max_dim: int = 1,
        max_edge_length: Optional[float] = None,
        max_edge_fraction: float = 0.5,
        weights: Optional[tuple[float, ...]] = None,
        gap_min_lifetime_fraction: float = 0.02,
    ):
        self.max_dim = max_dim
        self.max_edge_length = max_edge_length
        self.max_edge_fraction = max_edge_fraction
        self.weights = weights
        self.gap_min_lifetime_fraction = gap_min_lifetime_fraction

    @staticmethod
    def _count_essential(lifetimes: list[float], min_lifetime: float) -> int:
        """Count features whose lifetime stands out via the largest-ratio gap."""
        L = sorted([l for l in lifetimes if l > min_lifetime], reverse=True)
        if len(L) == 0:
            return 0
        if len(L) == 1:
            return 1
        ratios = [L[i] / max(L[i + 1], 1e-12) for i in range(len(L) - 1)]
        cut = int(np.argmax(ratios))
        # Require the gap to be meaningful (>= 2x), else just return all
        if ratios[cut] < 2.0:
            return len(L)
        return cut + 1

    def _compute(self, X):
        import gudhi  # local import; optional dep

        X = np.asarray(X, dtype=float)
        if self.max_edge_length is None:
            from scipy.spatial.distance import pdist
            diam = float(np.max(pdist(X))) if len(X) > 1 else 1.0
            self.max_edge_length = self.max_edge_fraction * diam

        rips = gudhi.RipsComplex(points=X, max_edge_length=self.max_edge_length)
        # Build only the 1-skeleton, then collapse edges before expanding to the
        # dimension needed for max_dim-homology. Edge collapse (Boissonnat–Pareek)
        # is exactly persistence-preserving for flag complexes, so the diagram is
        # numerically identical to expanding the full (max_dim+1)-skeleton directly
        # (verified in all dims on testbeds) — but on dense point clouds it caps the
        # simplex count by ~1000×. Without it, a low-dim dataset at edge_fraction 0.5
        # yields ~85% edge density → ~1e9 tetrahedra → ~120 GB in a single call, the
        # cause of the Phase B OOM collapse. Implementation change only, same quantity.
        st = rips.create_simplex_tree(max_dimension=1)
        st.collapse_edges(nb_iterations=2)
        st.expansion(self.max_dim + 1)
        st.compute_persistence()

        per_dim_persistence: dict[int, float] = {k: 0.0 for k in range(self.max_dim + 1)}
        per_dim_betti: dict[int, int] = {}
        per_dim_lifetimes: dict[int, list[float]] = {k: [] for k in range(self.max_dim + 1)}
        barcodes: dict[int, list[tuple[float, float]]] = {k: [] for k in range(self.max_dim + 1)}
        for dim in range(self.max_dim + 1):
            intervals = st.persistence_intervals_in_dimension(dim)
            for birth, death in intervals:
                d_eff = death if np.isfinite(death) else self.max_edge_length
                life = max(d_eff - birth, 0.0)
                per_dim_persistence[dim] += life
                per_dim_lifetimes[dim].append(life)
                barcodes[dim].append((float(birth), float(d_eff)))
            min_life = self.gap_min_lifetime_fraction * self.max_edge_length
            per_dim_betti[dim] = self._count_essential(per_dim_lifetimes[dim], min_life)

        weights = self.weights or tuple(1.0 for _ in range(self.max_dim + 1))
        ctop = float(sum(weights[k] * per_dim_persistence[k] for k in range(self.max_dim + 1)))

        return (
            ctop,
            None,
            {
                "betti": per_dim_betti,
                "persistence_by_dim": per_dim_persistence,
                "barcodes": barcodes,
                "max_edge_length": self.max_edge_length,
            },
        )


# ---------------------------------------------------------------------------
# Persistent homology, corrected estimator (2026-08-24)
# ---------------------------------------------------------------------------

class NormalisedPersistentHomology(BaseMetric):
    """Scale-normalised Vietoris-Rips topological complexity.

    Replacement for `PersistentHomology` as the v(D) topology component. The
    original is *kept unchanged* so the Phase B/C numbers stay reproducible;
    this class is the corrected quantity. Four defects motivated it (see
    `docs/deviations_log.md` 2026-08-24):

    1. **H0 was included with equal weight.** H0 total persistence is a
       dispersion statistic (n bars, each ~ an inter-point spacing), and it
       carried 50.7%-99.6% of the composite (median ~90%). Excluded here:
       topology means dims >= 1.
    2. **No scale normalisation.** Persistence was reported in absolute length
       units, so the component tracked ambient scale. Over the 27 classical-track
       datasets, r(log diam, log Ctop) = +0.921, R^2 = 0.85. Bars are now divided
       by the point cloud's diameter, making the measure scale-invariant.
    3. **The filtration was truncated at 0.5 x diam**, below where features die
       (a circle's H1 class dies at sqrt(3) r = 0.866 x diam), so real classes
       never died and were recorded as essential.
    4. **`collapse_edges` deletes essential classes on a truncated filtration.**
       Edge collapse is persistence-preserving for a *complete* flag filtration,
       not a truncated one. Demonstrated: a clean circle at edge_fraction 0.5
       gives n(H1) = 0 with collapse and n(H1) = 1 (P1 = 0.825) without; at
       fraction 0.9 the two agree exactly. Defects 3 and 4 compound — truncation
       manufactures essential classes and collapse then deletes exactly those.
       Consequence: the deployed estimator returned P1 = 0.000 for a perfect
       circle. It did not merely dilute topology, it removed it.

    Also note the deployed configuration was never the validated one:
    `qmlc/validate.py` used `max_dim=1, weights=(0.0, 1.0)` (H0 explicitly
    zeroed) while `experiments/feature_extraction.py` used `max_dim=2` with
    default weights `(1, 1, 1)`. Validated != deployed.

    Backend is `ripser`, which computes the full untruncated Rips persistence
    directly, so defects 3 and 4 cannot recur by construction.

    Measures (all over dims >= 1, all normalised by the diameter):

    - ``l2``     : sum of squared normalised persistences. Threshold-free,
                   continuous, downweights sampling noise quadratically. Default.
    - ``total``  : plain sum. **Fails validation** - it rewards bar *count*, so a
                   3D Gaussian blob outscores a clean circle. Retained only so the
                   failure is reproducible.
    - ``max``    : longest single normalised bar.
    - ``count``  : number of bars longer than `theta` (a Betti-like count).
    - ``sigsum`` : total persistence of bars longer than `theta`.

    **Known limitation.** Diameter normalisation is a *global* scale, so widely
    separated clusters inflate the denominator and multiplicity is lost: three
    circles 6 units apart score *below* one circle. Local-scale normalisation
    (median k-NN distance) fixes that but breaks sample-size invariance, which
    matters more when comparing datasets of different n. The trade is deliberate.
    """

    name = "topological_complexity_normalised"
    estimator = "ripser_diameter_normalised"

    def __init__(
        self,
        max_dim: int = 2,
        measure: str = "l2",
        theta: float = 0.10,
        subsample: int = 250,
        seed: int = 0,
        mahalanobis: bool = False,
    ):
        if measure not in {"l2", "total", "max", "count", "sigsum"}:
            raise ValueError(f"unknown measure {measure!r}")
        self.max_dim = max_dim
        self.measure = measure
        self.theta = theta
        self.subsample = subsample
        self.seed = seed
        self.mahalanobis = mahalanobis

    @staticmethod
    def _whiten(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Mahalanobis (whitening) transform: affine-invariant, isotropises a blob."""
        X = np.asarray(X, dtype=float)
        X = X - X.mean(axis=0)
        C = np.atleast_2d(np.cov(X, rowvar=False))
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        return X @ V @ np.diag(w ** -0.5)

    def _compute(self, X):
        from ripser import ripser  # local import; optional dep
        from scipy.spatial.distance import pdist, squareform

        X = np.asarray(X, dtype=float)
        if self.mahalanobis:
            X = self._whiten(X)
        n_full = len(X)
        if n_full > self.subsample:
            idx = np.random.default_rng(self.seed).permutation(n_full)[: self.subsample]
            X = X[idx]

        dm = squareform(pdist(X))
        diam = float(dm.max()) if len(X) > 1 else 1.0
        dgms = ripser(dm, distance_matrix=True, maxdim=self.max_dim)["dgms"]

        per_dim: dict[int, np.ndarray] = {}
        for k in range(self.max_dim + 1):
            d = np.asarray(dgms[k], dtype=float)
            if len(d) == 0:
                per_dim[k] = np.zeros(0)
                continue
            birth, death = d[:, 0].copy(), d[:, 1].copy()
            death[~np.isfinite(death)] = diam  # essential bars die at the diameter
            per_dim[k] = np.clip(death - birth, 0.0, None) / max(diam, 1e-300)

        L = np.concatenate([per_dim[k] for k in range(1, self.max_dim + 1)])
        L = L[L > 0]

        if self.measure == "l2":
            ctop = float((L ** 2).sum())
        elif self.measure == "total":
            ctop = float(L.sum())
        elif self.measure == "max":
            ctop = float(L.max()) if len(L) else 0.0
        elif self.measure == "count":
            ctop = float((L > self.theta).sum())
        else:  # sigsum
            ctop = float(L[L > self.theta].sum())

        return (
            ctop,
            None,
            {
                "measure": self.measure,
                "diameter": diam,
                "n_subsampled": len(X),
                "n_full": n_full,
                "normalised_persistence_by_dim": {
                    k: float(per_dim[k].sum()) for k in range(self.max_dim + 1)
                },
                "n_bars_above_theta": {
                    k: int((per_dim[k] > self.theta).sum()) for k in range(1, self.max_dim + 1)
                },
                "theta": self.theta,
                "mahalanobis": self.mahalanobis,
            },
        )


# ---------------------------------------------------------------------------
# Correlation order — pairwise mutual information (v_corr2)
# ---------------------------------------------------------------------------

class PairwiseMutualInformation(BaseMetric):
    """Mean pairwise mutual information over feature pairs (v_corr2).

    Uses sklearn's Kraskov k-NN MI estimator (`mutual_info_regression`) on each
    ordered-into-pair (X_i ; X_j). The summary scalar is the mean over all
    distinct feature pairs; `uncertainty` is the standard error of that mean
    across pairs. MI is non-negative — the estimator's small negative excursions
    are clipped at 0.

    Ground truth (bivariate Gaussian, correlation ρ): I = -½ ln(1-ρ²).
    """

    name = "second_order_mi"
    estimator = "sklearn_knn"

    def __init__(self, k: int = 3, max_pairs: int = 200, seed: int = 0):
        self.k = k
        self.max_pairs = max_pairs
        self.seed = seed

    def _compute(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n, d = X.shape
        if d < 2:
            return 0.0, 0.0, {"d": d, "n_pairs": 0, "note": "needs >=2 features"}

        rng = np.random.default_rng(self.seed)
        all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        if len(all_pairs) > self.max_pairs:
            idx = rng.choice(len(all_pairs), self.max_pairs, replace=False)
            pairs = [all_pairs[t] for t in idx]
        else:
            pairs = all_pairs

        mis = np.empty(len(pairs))
        for t, (i, j) in enumerate(pairs):
            mi = mutual_info_regression(
                X[:, [i]], X[:, j], n_neighbors=self.k, random_state=self.seed
            )[0]
            mis[t] = max(float(mi), 0.0)

        mean_mi = float(mis.mean())
        unc = float(mis.std(ddof=1) / np.sqrt(len(mis))) if len(mis) > 1 else None
        return mean_mi, unc, {
            "d": d,
            "n_pairs": len(pairs),
            "max_mi": float(mis.max()),
            "k": self.k,
        }


# ---------------------------------------------------------------------------
# Correlation order — higher-order MI via MINE (v_corr3+)
# ---------------------------------------------------------------------------

class HigherOrderMI(BaseMetric):
    """Higher-order total correlation via MINE (v_corr3+).

    For each order k in `orders`, draw `n_subsets` random k-subsets of features
    and estimate the multivariate total correlation
        TC(X_S) = D_KL( p(x_S) || ∏_j p(x_j) )
    with the Mutual Information Neural Estimator [Belghazi et al. 2018]: the
    Donsker–Varadhan lower bound is maximized over a small MLP critic T, with
    "joint" samples the rows of X[:, S] and "marginal" samples obtained by
    permuting each column of the subset independently.

    The reported value is the mean TC estimate across all (order, subset)
    trainings; `uncertainty` is the standard error across them. Because DV is a
    lower bound, the estimate under-reports the true TC — fine for a rank-
    normalized complexity component, and the bias is consistent across datasets.

    Ground truth (multivariate Gaussian Σ): TC = ½ ln( ∏ diag(Σ) / det Σ ).
    """

    name = "higher_order_mi"
    estimator = "mine"

    def __init__(
        self,
        orders: tuple[int, ...] = (3, 4, 5),
        n_subsets: int = 3,
        hidden: int = 64,
        epochs: int = 300,
        lr: float = 1e-3,
        batch: int = 256,
        seed: int = 0,
    ):
        self.orders = orders
        self.n_subsets = n_subsets
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed

    def _mine_tc(self, Z: np.ndarray, seed: int) -> float:
        """MINE DV estimate of total correlation for a (n, k) block Z."""
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        n, k = Z.shape
        # Standardize so the critic sees comparable scales.
        Z = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-8)
        Zt = torch.as_tensor(Z, dtype=torch.float32)

        critic = nn.Sequential(
            nn.Linear(k, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.Adam(critic.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(seed)

        bs = min(self.batch, n)
        ema = None  # EMA of the marginal partition for a low-bias gradient
        for _ in range(self.epochs):
            ji = torch.randint(0, n, (bs,), generator=gen)
            joint = Zt[ji]
            # Independent product: permute each column separately.
            marg = torch.stack(
                [Zt[torch.randperm(n, generator=gen)[:bs], c] for c in range(k)],
                dim=1,
            )
            t_joint = critic(joint)
            t_marg = critic(marg)
            exp_marg = torch.exp(t_marg)
            mean_exp = exp_marg.mean()
            ema = mean_exp.detach() if ema is None else 0.99 * ema + 0.01 * mean_exp.detach()
            # Biased-corrected DV loss (Belghazi §3.2): grad uses EMA denominator.
            loss = -(t_joint.mean() - (1.0 / (ema + 1e-8)) * mean_exp)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            t_joint = critic(Zt)
            marg = torch.stack(
                [Zt[torch.randperm(n, generator=gen), c] for c in range(k)], dim=1
            )
            t_marg = critic(marg)
            dv = t_joint.mean() - torch.logsumexp(t_marg, dim=0) + np.log(n)
        return max(float(dv), 0.0)

    def _compute(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        estimates: list[float] = []
        per_order: dict[int, list[float]] = {}
        for k in self.orders:
            if d < k:
                continue
            per_order[k] = []
            for s in range(self.n_subsets):
                subset = rng.choice(d, k, replace=False)
                tc = self._mine_tc(X[:, subset], seed=self.seed + 1000 * k + s)
                estimates.append(tc)
                per_order[k].append(tc)

        if not estimates:
            return 0.0, 0.0, {"d": d, "note": f"d<{min(self.orders)}; no order computable"}

        arr = np.array(estimates)
        mean_tc = float(arr.mean())
        unc = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else None
        return mean_tc, unc, {
            "d": d,
            "orders_used": sorted(per_order),
            "n_estimates": len(estimates),
            "mean_by_order": {k: float(np.mean(v)) for k, v in per_order.items()},
        }


# ---------------------------------------------------------------------------
# Compressibility — MDL via fixed autoencoder family (v_K)
# ---------------------------------------------------------------------------

class MDLAutoencoder(BaseMetric):
    """Minimum-description-length code length via a fixed autoencoder family (v_K).

    Replaces the gzip ratio as the K(D) proxy. A fixed-architecture undercomplete
    autoencoder (d → 2d → ⌈d/2⌉ → 2d → d, ReLU, fixed epochs/lr) is trained on
    standardized data; the two-part description length per sample is

        L = L_latent + L_residual
          = b · bits_per_latent + ½ Σ_j log₂(2πe · σ²_res,j)

    with b the bottleneck width and σ²_res,j the per-dimension residual variance
    (Gaussian code length for the reconstruction error). A model term
    (params · 16 bits / n) is added, amortized over the sample. Lower L ⇒ more
    compressible ⇒ lower algorithmic complexity. Reported value is bits/sample.
    """

    name = "mdl_autoencoder"
    estimator = "autoencoder_mdl"

    def __init__(
        self,
        bits_per_latent: float = 16.0,
        bits_per_param: float = 16.0,
        hidden_mult: int = 2,
        epochs: int = 400,
        lr: float = 1e-3,
        batch: int = 256,
        seed: int = 0,
    ):
        self.bits_per_latent = bits_per_latent
        self.bits_per_param = bits_per_param
        self.hidden_mult = hidden_mult
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed

    def _compute(self, X):
        import torch
        import torch.nn as nn

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n, d = X.shape
        torch.manual_seed(self.seed)

        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True) + 1e-8
        Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32)

        h = max(self.hidden_mult * d, 2)
        b = max(d // 2, 1)
        encoder = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, b))
        decoder = nn.Sequential(nn.Linear(b, h), nn.ReLU(), nn.Linear(h, d))
        ae = nn.Sequential(encoder, decoder)
        opt = torch.optim.Adam(ae.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)

        bs = min(self.batch, n)
        for _ in range(self.epochs):
            bi = torch.randint(0, n, (bs,), generator=gen)
            recon = ae(Xt[bi])
            loss = ((recon - Xt[bi]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            recon = ae(Xt)
            resid = (Xt - recon).cpu().numpy()
        # Per-dimension residual variance (standardized space).
        var_res = np.maximum(resid.var(axis=0), 1e-12)
        # Gaussian differential-entropy code length, bits per sample (sum over dims).
        l_residual = float(0.5 * np.sum(np.log2(2.0 * pi * np.e * var_res)))
        l_latent = float(b * self.bits_per_latent)
        n_params = sum(p.numel() for p in ae.parameters())
        l_model = float(n_params * self.bits_per_param / n)
        l_total = l_latent + l_residual + l_model

        return l_total, None, {
            "d": d,
            "n": n,
            "bottleneck": int(b),
            "l_latent_bits": l_latent,
            "l_residual_bits": l_residual,
            "l_model_bits_per_sample": l_model,
            "mean_residual_var": float(var_res.mean()),
        }


# ---------------------------------------------------------------------------
# Correlation order — kernel effective rank under reference RBF (v_kspec)
# ---------------------------------------------------------------------------

class KernelEffectiveRankRBF(BaseMetric):
    """Effective rank of the RBF Gram matrix at median-heuristic bandwidth (v_kspec).

    Builds K_ij = exp(-‖x_i - x_j‖² / (2σ²)) with σ the median pairwise Euclidean
    distance (median heuristic), then reports an effective rank of the spectrum.

    Two definitions are supported via `method`:
      - ``"entropy"`` (default, Roy & Vetterli 2007): r_eff = exp(H(p)) with
        p_i = λ_i / Σλ the normalized spectrum and H the Shannon entropy. This
        is the canonical "effective rank"; under the median-heuristic RBF it
        spreads ~6× more across datasets than the participation ratio (CV 0.22
        vs 0.04 on the Class A suite), cleanly separating low-dim manifolds
        (~3.4) from high-entropy data (~6.4).
      - ``"participation"``: r_eff = (Σλ)² / Σλ² — matches the convention in
        `qmlc.quantum.KernelEffectiveRank` and the embedded-kernel q-component.
        Kept for cross-checks against the q-side rank.

    Both definitions lie in [1, n]; the median heuristic makes the metric
    scale-invariant by construction (it measures structure, not scale).
    Bounded by subsampling to `max_n` points (the eigendecomposition is O(n³)).
    """

    name = "kernel_effective_rank"
    estimator = "rbf_median"

    def __init__(self, max_n: int = 1000, method: str = "entropy", seed: int = 0):
        if method not in ("entropy", "participation"):
            raise ValueError(f"unknown method {method!r}")
        self.max_n = max_n
        self.method = method
        self.seed = seed

    def _compute(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n = X.shape[0]
        if n > self.max_n:
            rng = np.random.default_rng(self.seed)
            X = X[rng.choice(n, self.max_n, replace=False)]
            n = self.max_n

        dists = pdist(X)  # condensed pairwise Euclidean distances
        sigma = float(np.median(dists))
        if sigma <= 0:
            # Degenerate (duplicate points dominate): rank collapses to 1.
            return 1.0, None, {"n": n, "sigma": sigma, "method": self.method,
                               "note": "median distance 0"}

        sq = squareform(dists) ** 2
        K = np.exp(-sq / (2.0 * sigma ** 2))
        eig = np.linalg.eigvalsh((K + K.T) / 2.0)
        eig = np.clip(eig.real, 0, None)
        s1 = float(eig.sum())

        if self.method == "participation":
            s2 = float((eig ** 2).sum())
            r_eff = (s1 ** 2) / max(s2, 1e-30)
        else:  # entropy (Roy-Vetterli)
            p = eig[eig > 1e-12] / max(s1, 1e-30)
            r_eff = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 1.0
        return r_eff, None, {"n": n, "sigma": sigma, "method": self.method, "trace": s1}
