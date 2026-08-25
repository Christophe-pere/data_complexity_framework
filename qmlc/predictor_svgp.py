"""PROTOTYPE — linear-scaling variational GP head for the Phase C predictor.

The frozen `GPMultiHead` (qmlc/predictor.py) is an exact gpytorch GP: O(n³) per fit.
It was validated at the 156-row smoke scale; full Phase B is 8,100 rows, and the
confirmatory LODO+inner-γ-CV wraps ~27×26×5×3 ≈ 10⁴ exact-GP fits on ~7,500 rows each
— computationally out of reach.

`SVGPMultiHead` is a stochastic variational GP (SVGP) with M inducing points and an
`LMCVariationalStrategy` over the two heads — the variational analogue of the frozen
rank-1 LMC MultitaskKernel. Cost is O(N·M² + M³), linear in N. It mirrors GPMultiHead's
interface (`_fit_two_targets` / `_predict_two_targets`, same X/Y standardization, same
γ→task-noise-ratio encoding) so it is drop-in for `joint_fit_loop` / `run_lodocv`.

This is a PROTOTYPE pending a fidelity check against the exact GP (see
`analysis/svgp_fidelity.py`). Whether adopting it needs an OSF amendment depends on that
check: if SVGP predictions match the exact GP within tolerance it "computes the same
quantity" (flexible per CLAUDE.md); if not, it is a modeling change → amendment.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from qmlc.predictor import PredictorBase

# torch ignores OMP_NUM_THREADS for its intra-op pool; pin it here so that N parallel Phase C
# fold-processes don't each spawn a multi-thread pool and oversubscribe the cores (the cause of
# the 2026-07-07 run stalling at ~60% CPU with 3 threads/proc). Guarded: interop must be set
# before any parallel work and raises if already initialized.
try:
    import torch as _torch
    _torch.set_num_threads(1)
    try:
        _torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
except Exception:
    pass


class SVGPMultiHead(PredictorBase):
    """Variational multi-output GP (LMC, M inducing points). Drop-in for GPMultiHead.

    Parameters
    ----------
    n_iter : Adam iterations over the ELBO.
    num_inducing : M inducing points (learned locations), initialized from a random
        subset of the standardized training rows.
    num_latents : latent GPs mixed into the 2 task outputs (LMC rank). 2 gives the
        exact model's rank-1-coregionalization-plus-task-noise flexibility.
    lr : Adam step.
    """

    def __init__(self, n_iter: int = 400, num_inducing: int = 256, num_latents: int = 2,
                 lr: float = 0.05, device: str = "cpu", verbose: bool = False):
        super().__init__()
        self.n_iter = n_iter
        self.num_inducing = num_inducing
        self.num_latents = num_latents
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self._model = None
        self._likelihood = None
        self._x_mean = self._x_std = self._y_mean = self._y_std = None

    def _build(self, ard_dim: int, Xn: np.ndarray):
        import torch
        import gpytorch

        L, M = self.num_latents, min(self.num_inducing, Xn.shape[0])
        rng = np.random.default_rng(0)
        idx = rng.choice(Xn.shape[0], size=M, replace=False)
        # inducing points: (num_latents, M, d) — each latent gets its own (shared init)
        ind = torch.tensor(np.repeat(Xn[idx][None], L, axis=0), dtype=torch.float64)

        class _LMCSVGP(gpytorch.models.ApproximateGP):
            def __init__(self):
                vdist = gpytorch.variational.CholeskyVariationalDistribution(
                    M, batch_shape=torch.Size([L]))
                vstrat = gpytorch.variational.LMCVariationalStrategy(
                    gpytorch.variational.VariationalStrategy(
                        self, ind, vdist, learn_inducing_locations=True),
                    num_tasks=2, num_latents=L, latent_dim=-1)
                super().__init__(vstrat)
                self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([L]))
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=ard_dim, batch_shape=torch.Size([L])),
                    batch_shape=torch.Size([L]))

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x))

        model = _LMCSVGP().to(dtype=torch.float64, device=self.device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=2, has_global_noise=False).to(dtype=torch.float64, device=self.device)
        return model, likelihood

    def _fit_two_targets(self, X, y_edge, y_align_logit, gamma):
        import torch
        import gpytorch

        X = np.asarray(X, dtype=np.float64)
        Y = np.column_stack([np.asarray(y_edge, np.float64), np.asarray(y_align_logit, np.float64)])
        self._x_mean, self._x_std = X.mean(0), X.std(0) + 1e-8
        self._y_mean, self._y_std = Y.mean(0), Y.std(0) + 1e-8
        Xn = (X - self._x_mean) / self._x_std
        Yn = (Y - self._y_mean) / self._y_std
        train_x = torch.tensor(Xn, dtype=torch.float64, device=self.device)
        train_y = torch.tensor(Yn, dtype=torch.float64, device=self.device)

        model, likelihood = self._build(Xn.shape[1], Xn)
        # γ → fixed task-noise ratio [1, 1/γ], mirroring the exact GPMultiHead.
        with torch.no_grad():
            likelihood.task_noises = torch.tensor([1.0, 1.0 / max(gamma, 1e-3)], dtype=torch.float64)

        model.train(); likelihood.train()
        # Optimize model (variational + kernel + inducing) only → task noise stays fixed.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
        for i in range(self.n_iter):
            optimizer.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            if self.verbose and (i % max(self.n_iter // 5, 1) == 0):
                print(f"  SVGP iter {i}: -elbo = {loss.item():.4f}")

        model.eval(); likelihood.eval()
        self._model, self._likelihood = model, likelihood

    def _predict_two_targets(self, X):
        import torch
        import gpytorch
        if self._model is None:
            raise RuntimeError("SVGPMultiHead not fitted")
        Xn = (np.asarray(X, np.float64) - self._x_mean) / self._x_std
        test_x = torch.tensor(Xn, dtype=torch.float64, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            pred = self._likelihood(self._model(test_x))
            mean = pred.mean.detach().cpu().numpy()
            var = pred.variance.detach().cpu().numpy()
        mean_de = mean * self._y_std + self._y_mean
        std_de = np.sqrt(np.maximum(var, 0.0)) * self._y_std
        return mean_de[:, 0], mean_de[:, 1], std_de[:, 0], std_de[:, 1]
