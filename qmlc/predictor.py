"""Predictor harness — the selector S(v, q) → (ê, â) with calibrated uncertainty.

Implements the GP + GBM multi-head architecture of framework_v2.md §2.4:

    ê(D, φ)   = S_edge(Φ(v(D), q(D, φ)))                   Eq. 2.5
    a(D, U)   = σ(S_align(Φ(v(D), q(D, φ))))               Eq. 2.6
    E[ε_gen]  ≈ ε_emp + β(1 - a(D, U))                     Eq. 2.7

trained jointly via the Eq. 2.8 multi-task loss with γ multi-task weighting
(selected via inner CV in `qmlc.cv`).

Two instantiations:

GP (`GPMultiHead`):
    * gpytorch ExactGP with MultitaskKernel(ARD-RBF, rank-1 coregionalization).
    * Shared input lengthscales across heads → strict LMC, matches Eq. 2.5–2.6.
    * MultitaskGaussianLikelihood with per-task noise; γ encoded as the inverse
      ratio of the two task noises (γ ↑ → align noise ↓ → align weighted more).
    * Marginal-likelihood training; predictive variance gives sigma_e / sigma_a.

GBM (`GBMMultiHead`):
    * Two independent LightGBM regressors over the same features (v ⊕ q).
    * Edge head: quantile regression at three quantiles → median + central
      interval (used as a uniformity-of-percentiles proxy for ECE).
    * Align head: regression on the raw alignment logit; no native variance.
    * γ realized as sample-weight ratio between heads (approximation; trees are
      not shared, so the "shared representation" is the input feature space).
    * Departure from strict shared-splits is documented in
      `docs/deviations_log.md` (created lazily on first fit if missing).

Common framework around both:
    * `TrainingPoint`: per-(dataset, encoding, ansatz, seed) observation carrying
      v(D), q(D, φ), and the three observed scalars (e_obs, gap_obs, ε_emp).
    * `solve_beta`: 1D least-squares closed form for the β coefficient of Eq. 2.7
      once a_hat is known.
    * `joint_fit_loop`: alternates {fit predictor on (X, e_obs, a_target_logit)} ↔
      {recompute β}; converges in O(5) iterations.
    * `compute_ece`: expected calibration error for Gaussian predictive intervals
      (binned-percentile flavor; H5 target ≤ 0.10).
"""
from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingPoint:
    """One (dataset, encoding, ansatz, seed) observation feeding the predictor.

    For the edge head, only (v, q, e_obs) are used. For the alignment head,
    (gap_obs, epsilon_emp) supply the indirect target — see `joint_fit_loop`.
    """

    dataset_id: str
    encoding_id: str
    ansatz_id: str
    seed: int
    v: np.ndarray            # shape (K,) intrinsic data-complexity vector
    q: np.ndarray            # shape (M,) interaction vector
    e_obs: float             # observed edge (budget-matched, primary target)
    gap_obs: float           # ε_test - ε_train of the quantum model
    epsilon_emp: float       # ε_train of the quantum model

    def features(self) -> np.ndarray:
        return np.concatenate([self.v, self.q]).astype(np.float64)


@dataclass
class PredictionResult:
    """Per-sample predictor output."""

    e_hat: np.ndarray              # (n,) predicted edge
    a_hat: np.ndarray              # (n,) predicted alignment ∈ (0, 1)
    sigma_e: np.ndarray            # (n,) edge predictive std (NaN if unavailable)
    sigma_a: np.ndarray            # (n,) alignment predictive std (NaN if unavailable)
    beta: float                    # learned coefficient in Eq. 2.7

    def predicted_gap(self, epsilon_emp: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicted generalization gap ĝ = β(1 − â)  (Eq. 2.7: ε_gen = ε_emp + gap).

        Returns the *gap* (ε_gen − ε_emp), matching the stored `gap_obs`. To recover the
        absolute generalization error, add ε_emp. `epsilon_emp` is accepted for backward
        compatibility but not used (the gap is ε_emp-relative by construction)."""
        return self.beta * (1.0 - self.a_hat)


def stack_training_points(points: Sequence[TrainingPoint]) -> dict[str, np.ndarray]:
    """Vectorize a list of TrainingPoint into numpy arrays."""
    X = np.stack([p.features() for p in points], axis=0)
    return {
        "X": X,
        "e_obs": np.asarray([p.e_obs for p in points], dtype=np.float64),
        "gap_obs": np.asarray([p.gap_obs for p in points], dtype=np.float64),
        "epsilon_emp": np.asarray([p.epsilon_emp for p in points], dtype=np.float64),
        "dataset_id": np.asarray([p.dataset_id for p in points]),
        "encoding_id": np.asarray([p.encoding_id for p in points]),
        "ansatz_id": np.asarray([p.ansatz_id for p in points]),
    }


# ---------------------------------------------------------------------------
# β solver and alignment-logit helpers
# ---------------------------------------------------------------------------

_LOGIT_EPS = 1e-3  # clamp for alignment target to avoid logit blow-up


def solve_beta(gap_obs: np.ndarray, epsilon_emp: np.ndarray, a_hat: np.ndarray) -> float:
    """Closed-form least-squares for β in the gap model gap = β(1 − a).

    β = argmin_β Σ_i (gap_obs_i - β (1 - a_hat_i))²
      = Σ (gap_obs_i · (1 - a_hat_i)) / Σ (1 - a_hat_i)²

    `gap_obs` is already the generalization gap (ε_gen − ε_emp), so ε_emp does NOT enter
    here — subtracting it (the pre-2026-07-10 bug) made the residual ≈ −ε_emp < 0 and forced
    β→0 under the β ≥ 0 constraint. `epsilon_emp` is kept for API compatibility but unused.
    Enforces β ≥ 0 (framework: β = exp(b)).
    """
    residual = np.asarray(gap_obs, dtype=np.float64)
    factor = 1.0 - np.asarray(a_hat, dtype=np.float64)
    den = float(np.dot(factor, factor))
    if den < 1e-12:
        return 0.0
    return max(float(np.dot(residual, factor) / den), 0.0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def alignment_target_logit(gap_obs: np.ndarray, epsilon_emp: np.ndarray, beta: float) -> np.ndarray:
    """Derive a logit-of-alignment training target from the observed gap + current β.

    From gap = β(1 − a) inverted: a = 1 − gap_obs / β. (`gap_obs` is already the gap, so ε_emp
    does not enter — `epsilon_emp` kept for API compatibility, unused.) Clipped before logit.
    """
    if beta <= 0:
        return np.zeros_like(gap_obs)
    a_target = 1.0 - np.asarray(gap_obs, dtype=np.float64) / beta
    return _logit(a_target)


# ---------------------------------------------------------------------------
# Predictor ABC
# ---------------------------------------------------------------------------

class PredictorBase(ABC):
    """Multi-head predictor: edge ê + alignment â, with optional uncertainty.

    Joint fitting is delegated to `joint_fit_loop` (module-level), which
    alternates between fitting the predictor's internal model and updating β.
    Subclasses implement `_fit_two_targets` (raw fit on stacked targets) and
    `_predict_two_targets` (returns means + stds for both heads).
    """

    name: str = "predictor"

    def __init__(self):
        self._beta: float = 1.0  # placeholder until joint fit converges
        self._fitted: bool = False

    @property
    def beta(self) -> float:
        return self._beta

    @abstractmethod
    def _fit_two_targets(
        self,
        X: np.ndarray,
        y_edge: np.ndarray,
        y_align_logit: np.ndarray,
        gamma: float,
    ) -> None:
        ...

    @abstractmethod
    def _predict_two_targets(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (e_hat, raw_align_logit_hat, sigma_e, sigma_align_logit)."""
        ...

    def fit(
        self,
        X: np.ndarray,
        e_obs: np.ndarray,
        gap_obs: np.ndarray,
        epsilon_emp: np.ndarray,
        gamma: float = 1.0,
        n_iters: int = 4,
        beta_init: Optional[float] = None,
    ) -> "PredictorBase":
        """Joint EM-style fit: alternate (fit predictor) ↔ (update β)."""
        joint_fit_loop(self, X, e_obs, gap_obs, epsilon_emp, gamma=gamma, n_iters=n_iters, beta_init=beta_init)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> PredictionResult:
        if not self._fitted:
            raise RuntimeError("predictor not fit yet")
        e_hat, raw_logit, sigma_e, sigma_logit = self._predict_two_targets(X)
        a_hat = _sigmoid(raw_logit)
        # σ_a via delta method: a = σ(z), da/dz = a(1-a)
        sigma_a = sigma_logit * a_hat * (1.0 - a_hat)
        return PredictionResult(
            e_hat=e_hat,
            a_hat=a_hat,
            sigma_e=sigma_e,
            sigma_a=sigma_a,
            beta=self._beta,
        )


def joint_fit_loop(
    predictor: PredictorBase,
    X: np.ndarray,
    e_obs: np.ndarray,
    gap_obs: np.ndarray,
    epsilon_emp: np.ndarray,
    gamma: float,
    n_iters: int = 4,
    beta_init: Optional[float] = None,
) -> None:
    """Alternate predictor-fit and β-update.

    On iteration t:
      1. Compute a_target_logit_t = logit(clip(1 - (gap_obs - ε_emp) / β_{t-1}))
      2. Refit predictor on (X) → (e_obs, a_target_logit_t)
      3. Predict a_hat on training X, update β_t via solve_beta
    """
    if beta_init is None:
        # Anchor a ≈ 0.5 at the typical (RMS) gap: gap ≈ β·0.5 ⇒ β ≈ 2·std(gap_obs).
        # (std, not |mean|, since gap_obs is centered near 0 — |mean| would collapse β_init.)
        beta = max(2.0 * float(np.std(np.asarray(gap_obs, dtype=np.float64))), 1e-2)
    else:
        beta = max(beta_init, 1e-2)
    predictor._beta = beta

    for _ in range(n_iters):
        target_logit = alignment_target_logit(gap_obs, epsilon_emp, beta)
        predictor._fit_two_targets(X, e_obs, target_logit, gamma=gamma)
        _, raw_logit_hat, _, _ = predictor._predict_two_targets(X)
        a_hat = _sigmoid(raw_logit_hat)
        beta = solve_beta(gap_obs, epsilon_emp, a_hat)
        if beta < 1e-6:
            beta = 1e-6
        predictor._beta = beta


# ---------------------------------------------------------------------------
# GP multi-head (gpytorch LMC)
# ---------------------------------------------------------------------------

class GPMultiHead(PredictorBase):
    """Multi-output GP with rank-1 LMC kernel (gpytorch).

    Shared ARD-RBF lengthscales across the two heads → faithful to Eq. 2.5/2.6's
    "shared representation Φ" requirement. Per-task noise carries γ.

    Parameters
    ----------
    n_iter : int
        Adam steps for marginal-likelihood maximization.
    lr : float
        Adam learning rate.
    device : str
        "cpu" or "cuda". Defaults to "cpu" — Phase B sizes are O(10²) datasets.
    """

    name = "gp_lmc"

    def __init__(self, n_iter: int = 200, lr: float = 0.05, device: str = "cpu", verbose: bool = False):
        super().__init__()
        self.n_iter = n_iter
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self._model = None
        self._likelihood = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

    def _fit_two_targets(self, X, y_edge, y_align_logit, gamma):
        import torch
        import gpytorch

        X = np.asarray(X, dtype=np.float64)
        Y = np.column_stack([np.asarray(y_edge, dtype=np.float64),
                             np.asarray(y_align_logit, dtype=np.float64)])

        # Standardize features and targets for numerical stability
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        self._y_mean = Y.mean(axis=0)
        self._y_std = Y.std(axis=0) + 1e-8
        Xn = (X - self._x_mean) / self._x_std
        Yn = (Y - self._y_mean) / self._y_std

        train_x = torch.tensor(Xn, dtype=torch.float64, device=self.device)
        train_y = torch.tensor(Yn, dtype=torch.float64, device=self.device)

        ard_dim = Xn.shape[1]

        class _LMCModel(gpytorch.models.ExactGP):
            def __init__(self, tx, ty, likelihood):
                super().__init__(tx, ty, likelihood)
                self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ConstantMean(), num_tasks=2,
                )
                self.covar_module = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=ard_dim),
                    num_tasks=2,
                    rank=1,
                )

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=2, has_global_noise=False
        ).to(dtype=torch.float64, device=self.device)
        # γ encoded as inverse task-noise ratio:
        #   noise_edge = 1, noise_align = 1/γ   (γ ↑ ⇒ align fit more aggressively)
        with torch.no_grad():
            likelihood.task_noises = torch.tensor([1.0, 1.0 / max(gamma, 1e-3)], dtype=torch.float64)

        model = _LMCModel(train_x, train_y, likelihood).to(dtype=torch.float64, device=self.device)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.n_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if self.verbose and (i % max(self.n_iter // 5, 1) == 0):
                print(f"  GP iter {i}: -mll = {loss.item():.4f}")

        model.eval()
        likelihood.eval()
        self._model = model
        self._likelihood = likelihood

    def _predict_two_targets(self, X):
        import torch
        import gpytorch

        if self._model is None:
            raise RuntimeError("GPMultiHead not fitted")

        Xn = (np.asarray(X, dtype=np.float64) - self._x_mean) / self._x_std
        test_x = torch.tensor(Xn, dtype=torch.float64, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            pred = self._likelihood(self._model(test_x))
            mean = pred.mean.detach().cpu().numpy()                  # (n, 2)
            var = pred.variance.detach().cpu().numpy()               # (n, 2)
        # Destandardize: μ' = μ * σ_y + m_y; std' = std * σ_y
        mean_de = mean * self._y_std + self._y_mean
        std_de = np.sqrt(np.maximum(var, 0.0)) * self._y_std
        e_hat = mean_de[:, 0]
        raw_logit_hat = mean_de[:, 1]
        sigma_e = std_de[:, 0]
        sigma_logit = std_de[:, 1]
        return e_hat, raw_logit_hat, sigma_e, sigma_logit


# ---------------------------------------------------------------------------
# GBM multi-head (lightgbm)
# ---------------------------------------------------------------------------

class GBMMultiHead(PredictorBase):
    """Two independent LightGBM regressors over the shared (v ⊕ q) features.

    Edge head: three-quantile regression (5%, 50%, 95%) → median + central
    interval. The 90% CI half-width / 1.645 is used as a Gaussian-equivalent
    sigma_e for compatibility with ECE.

    Align head: regression on the raw alignment logit; sigma is reported as NaN
    (GBM has no native variance; quantile heads would be possible but the
    alignment target is noisier and the rank from the GP suffices for ECE).

    γ is realized as the sample-weight ratio between heads — strict shared-
    splits across heads is not supported by stock LightGBM, so the two heads
    have independent tree ensembles. Documented in `docs/deviations_log.md`.
    """

    name = "gbm"

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 5,
        random_state: int = 0,
        verbose: bool = False,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.verbose = verbose
        self._edge_models: dict[str, Any] = {}  # quantile -> LGBMRegressor
        self._align_model = None

    def _lgbm_common_kwargs(self) -> dict[str, Any]:
        return dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1 if not self.verbose else 1,
        )

    def _fit_two_targets(self, X, y_edge, y_align_logit, gamma):
        import lightgbm as lgb

        X = np.asarray(X, dtype=np.float64)
        y_edge = np.asarray(y_edge, dtype=np.float64)
        y_align_logit = np.asarray(y_align_logit, dtype=np.float64)

        # Edge head: three quantiles for uncertainty
        self._edge_models = {}
        for alpha in (0.05, 0.5, 0.95):
            m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **self._lgbm_common_kwargs())
            m.fit(X, y_edge)
            self._edge_models[alpha] = m

        # Align head: simple regression on the logit, with gamma as sample weight
        # (sample weight scales the loss; gamma > 1 ⇒ align head trained harder)
        weights = np.full(len(X), float(gamma), dtype=np.float64)
        self._align_model = lgb.LGBMRegressor(objective="regression", **self._lgbm_common_kwargs())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._align_model.fit(X, y_align_logit, sample_weight=weights)

    def _predict_two_targets(self, X):
        if not self._edge_models or self._align_model is None:
            raise RuntimeError("GBMMultiHead not fitted")
        X = np.asarray(X, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q05 = self._edge_models[0.05].predict(X)
            q50 = self._edge_models[0.5].predict(X)
            q95 = self._edge_models[0.95].predict(X)
            raw_logit_hat = self._align_model.predict(X)
        # Approximate Gaussian-equivalent sigma: half-90%-CI / 1.6449
        sigma_e = np.maximum(q95 - q05, 0.0) / (2.0 * 1.6449)
        e_hat = q50
        sigma_logit = np.full(len(X), np.nan)
        return e_hat, raw_logit_hat, sigma_e, sigma_logit


# ---------------------------------------------------------------------------
# Ensemble — GP + GBM in parallel, report both
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePrediction:
    gp: PredictionResult
    gbm: PredictionResult

    @property
    def agreement_e(self) -> np.ndarray:
        """Per-sample disagreement |ê_GP - ê_GBM|. Large values flag divergence."""
        return np.abs(self.gp.e_hat - self.gbm.e_hat)


class PredictorEnsemble:
    """Run GP and GBM in parallel; report both predictions and their disagreement.

    Per framework_v2.md §2.1: "Agreement between the two is evidence of robustness;
    disagreement is itself a finding."
    """

    name = "gp+gbm"

    def __init__(self, gp: Optional[GPMultiHead] = None, gbm: Optional[GBMMultiHead] = None):
        self.gp = gp if gp is not None else GPMultiHead()
        self.gbm = gbm if gbm is not None else GBMMultiHead()

    def fit(
        self,
        X: np.ndarray,
        e_obs: np.ndarray,
        gap_obs: np.ndarray,
        epsilon_emp: np.ndarray,
        gamma: float = 1.0,
        n_iters: int = 4,
    ) -> "PredictorEnsemble":
        self.gp.fit(X, e_obs, gap_obs, epsilon_emp, gamma=gamma, n_iters=n_iters)
        self.gbm.fit(X, e_obs, gap_obs, epsilon_emp, gamma=gamma, n_iters=n_iters)
        return self

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        return EnsemblePrediction(gp=self.gp.predict(X), gbm=self.gbm.predict(X))


# ---------------------------------------------------------------------------
# Calibration: expected calibration error
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error for Gaussian predictive intervals.

    For each prediction with mean y_pred and std sigma > 0, compute the
    standard-normal percentile of (y_true - y_pred) / sigma. A perfectly
    calibrated predictor produces these percentiles uniformly on [0, 1].
    ECE is the L1 deviation from uniformity over `n_bins` equal-width bins,
    normalized so the worst case is 1.0.

    H5 pre-registers ECE ≤ 0.10 as a calibration acceptance threshold.
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    valid = (sigma > 1e-9) & np.isfinite(sigma) & np.isfinite(y_pred) & np.isfinite(y_true)
    if int(np.sum(valid)) < n_bins:
        return float("nan")

    z = (y_true[valid] - y_pred[valid]) / sigma[valid]
    percentiles = norm.cdf(z)
    counts, _ = np.histogram(percentiles, bins=np.linspace(0.0, 1.0, n_bins + 1))
    expected = float(len(percentiles)) / n_bins
    return float(np.sum(np.abs(counts - expected)) / (2.0 * len(percentiles)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot
