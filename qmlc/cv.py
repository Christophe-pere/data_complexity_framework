"""Cross-validation helpers — LODOcv outer + inner γ selection.

Frozen per CLAUDE.md:
  * Outer: leave-one-dataset-out (LODOcv) — each fold holds out all observations
    for one dataset_id; predict on those, train on the rest.
  * Inner: leave-one-additional-out — within an outer training fold, hold out
    one *additional* dataset to score candidate γ values, repeat across all
    additional choices, average.

Why dataset-level rather than observation-level: the per-(dataset, encoding,
ansatz, seed) observations are correlated within a dataset (shared v(D), shared
underlying task), so observation-level CV would overestimate generalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from .predictor import (
    PredictorBase,
    PredictorEnsemble,
    PredictionResult,
    EnsemblePrediction,
    r2_score,
    compute_ece,
)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def leave_one_dataset_out(dataset_ids: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_id, train_idx, test_idx) for each unique dataset."""
    dataset_ids = np.asarray(dataset_ids)
    unique = list(np.unique(dataset_ids))
    for ho in unique:
        test_mask = dataset_ids == ho
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield str(ho), train_idx, test_idx


def leave_one_additional_out(
    dataset_ids: np.ndarray,
    train_idx: np.ndarray,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Within `train_idx`, yield each "leave-one-additional" sub-split.

    Holds out one of the dataset_ids present in train_idx; returns
    (held_out_id, sub_train_idx, sub_test_idx) where indices are into the
    *full* sample axis (so the caller can index X[idx] directly).
    """
    sub_dataset_ids = dataset_ids[train_idx]
    unique = list(np.unique(sub_dataset_ids))
    for ho in unique:
        sub_test_mask = sub_dataset_ids == ho
        sub_train_idx = train_idx[~sub_test_mask]
        sub_test_idx = train_idx[sub_test_mask]
        yield str(ho), sub_train_idx, sub_test_idx


# ---------------------------------------------------------------------------
# γ selection via inner CV
# ---------------------------------------------------------------------------

DEFAULT_GAMMA_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass
class GammaSelection:
    best_gamma: float
    grid: tuple[float, ...]
    scores: dict[float, float]      # γ -> mean inner-fold composite score
    n_folds: int


def select_gamma(
    factory: Callable[[], PredictorBase],
    X: np.ndarray,
    e_obs: np.ndarray,
    gap_obs: np.ndarray,
    epsilon_emp: np.ndarray,
    dataset_ids: np.ndarray,
    train_idx: np.ndarray,
    gamma_grid: Sequence[float] = DEFAULT_GAMMA_GRID,
    n_iters: int = 3,
    verbose: bool = False,
) -> GammaSelection:
    """Inner CV: pick γ by leave-one-additional-out within `train_idx`.

    Composite score = R²(edge) + R²(gap_predicted vs gap_obs); higher is better.
    The composite balances both heads of Eq. 2.8 without re-introducing γ in
    the score itself.
    """
    grid = tuple(float(g) for g in gamma_grid)
    n_folds = 0
    sums = {g: 0.0 for g in grid}

    for sub_id, sub_train, sub_test in leave_one_additional_out(dataset_ids, train_idx):
        if len(sub_test) == 0 or len(sub_train) == 0:
            continue
        n_folds += 1
        for g in grid:
            pred = factory()
            try:
                pred.fit(
                    X[sub_train],
                    e_obs[sub_train],
                    gap_obs[sub_train],
                    epsilon_emp[sub_train],
                    gamma=g,
                    n_iters=n_iters,
                )
                result = pred.predict(X[sub_test])
                edge_r2 = r2_score(e_obs[sub_test], result.e_hat)
                gap_pred = result.predicted_gap(epsilon_emp[sub_test])
                gap_r2 = r2_score(gap_obs[sub_test], gap_pred)
                composite = (edge_r2 if np.isfinite(edge_r2) else -1.0) + \
                            (gap_r2 if np.isfinite(gap_r2) else -1.0)
            except Exception as e:
                if verbose:
                    print(f"  [γ={g}] fit failed on inner fold {sub_id}: {e}")
                composite = -10.0
            sums[g] += composite

    scores = {g: (sums[g] / max(n_folds, 1)) for g in grid}
    best_gamma = max(scores.items(), key=lambda kv: kv[1])[0]
    if verbose:
        print(f"  γ selection: scores={scores}; best={best_gamma}")
    return GammaSelection(best_gamma=best_gamma, grid=grid, scores=scores, n_folds=n_folds)


# ---------------------------------------------------------------------------
# Outer LODOcv
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    held_out_dataset: str
    n_train: int
    n_test: int
    gamma_selection: Optional[GammaSelection]
    edge_r2_gp: float
    edge_r2_gbm: float
    edge_ece_gp: float
    gap_r2_gp: float
    gap_r2_gbm: float
    e_hat_gp: np.ndarray
    e_hat_gbm: np.ndarray
    a_hat_gp: np.ndarray
    a_hat_gbm: np.ndarray
    sigma_e_gp: np.ndarray
    sigma_e_gbm: np.ndarray
    e_obs_test: np.ndarray
    gap_obs_test: np.ndarray
    eps_emp_test: np.ndarray
    beta_gp: float
    beta_gbm: float


@dataclass
class LODOcvReport:
    folds: list[FoldResult]

    def summary(self) -> dict[str, float]:
        edge_r2_gp = np.array([f.edge_r2_gp for f in self.folds])
        edge_r2_gbm = np.array([f.edge_r2_gbm for f in self.folds])
        ece_gp = np.array([f.edge_ece_gp for f in self.folds])
        gap_r2_gp = np.array([f.gap_r2_gp for f in self.folds])
        gap_r2_gbm = np.array([f.gap_r2_gbm for f in self.folds])
        return {
            "n_folds": len(self.folds),
            "edge_r2_gp_mean": float(np.nanmean(edge_r2_gp)),
            "edge_r2_gp_median": float(np.nanmedian(edge_r2_gp)),
            "edge_r2_gbm_mean": float(np.nanmean(edge_r2_gbm)),
            "edge_r2_gbm_median": float(np.nanmedian(edge_r2_gbm)),
            "edge_ece_gp_mean": float(np.nanmean(ece_gp)),
            "gap_r2_gp_mean": float(np.nanmean(gap_r2_gp)),
            "gap_r2_gbm_mean": float(np.nanmean(gap_r2_gbm)),
        }


def _fit_rank_normalizer(train_cols: np.ndarray) -> list[np.ndarray]:
    """Fit a per-column rank (empirical-CDF) transform on training rows only:
    store the sorted training values that define each column's CDF support."""
    return [np.sort(train_cols[:, j]) for j in range(train_cols.shape[1])]


def _apply_rank_normalizer(cols: np.ndarray, sorted_train: list[np.ndarray]) -> np.ndarray:
    """Map each value to (#train ≤ value)/n_train ∈ (0,1] via the training ECDF.
    Held-out values below/above the training range clamp to 0 / 1 — no held-out
    information enters the transform. Matches `_rank_normalize` semantics (rank/n)
    but fit on the training fold, so it is leakage-clean under LODO-CV."""
    out = np.empty_like(cols, dtype=np.float64)
    for j, s in enumerate(sorted_train):
        n = max(len(s), 1)
        out[:, j] = np.searchsorted(s, cols[:, j], side="right") / n
    return out


def run_lodocv(
    X: np.ndarray,
    e_obs: np.ndarray,
    gap_obs: np.ndarray,
    epsilon_emp: np.ndarray,
    dataset_ids: np.ndarray,
    gamma_grid: Sequence[float] = DEFAULT_GAMMA_GRID,
    select_gamma_per_fold: bool = True,
    n_iters: int = 3,
    gp_n_iter: int = 150,
    gbm_n_estimators: int = 200,
    rank_normalize_k: int = 0,
    gp_ctor: Optional[Callable[[int], "object"]] = None,
    verbose: bool = True,
) -> LODOcvReport:
    """Run leave-one-dataset-out CV with inner γ selection.

    For each outer fold:
      0. (if rank_normalize_k>0) rank-normalize the first `rank_normalize_k`
         feature columns (the v(D) block) using a transform fit on the outer
         *training* rows only, applied to all rows — leakage-clean per-fold
         normalization (framework_v2 §2.1). The remaining columns (q) pass through.
      1. Pick γ via leave-one-additional-out (or use γ=1.0 if disabled).
      2. Refit GP and GBM on the full outer training set with chosen γ.
      3. Predict on the held-out dataset; record metrics.
    """
    from .predictor import GPMultiHead

    if gp_ctor is None:
        gp_ctor = lambda n: GPMultiHead(n_iter=n)  # noqa: E731
    dataset_ids = np.asarray(dataset_ids)
    folds: list[FoldResult] = []
    for held_out, train_idx, test_idx in leave_one_dataset_out(dataset_ids):
        folds.append(run_single_fold(
            held_out, train_idx, test_idx, X, e_obs, gap_obs, epsilon_emp, dataset_ids,
            gamma_grid=gamma_grid, select_gamma_per_fold=select_gamma_per_fold, n_iters=n_iters,
            gp_n_iter=gp_n_iter, gbm_n_estimators=gbm_n_estimators,
            rank_normalize_k=rank_normalize_k, gp_ctor=gp_ctor, verbose=verbose,
        ))
    return LODOcvReport(folds=folds)


def run_single_fold(
    held_out: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    e_obs: np.ndarray,
    gap_obs: np.ndarray,
    epsilon_emp: np.ndarray,
    dataset_ids: np.ndarray,
    gamma_grid: Sequence[float] = DEFAULT_GAMMA_GRID,
    select_gamma_per_fold: bool = True,
    n_iters: int = 3,
    gp_n_iter: int = 150,
    gbm_n_estimators: int = 200,
    rank_normalize_k: int = 0,
    gp_ctor: Optional[Callable[[int], "object"]] = None,
    inner_gp_ctor: Optional[Callable[[int], "object"]] = None,
    verbose: bool = True,
) -> FoldResult:
    """One outer LODO fold: per-fold rank-norm, inner γ selection, outer GP+GBM fit, predict
    held-out. Independent of other folds ⇒ the parallel/resumable Phase C unit (run_phase_c_fold).

    `inner_gp_ctor` (optional) is the GP used inside γ-selection only; it may be a cheaper/lower-
    fidelity GP than `gp_ctor` since inner CV only *ranks* γ (the outer confirmatory fit uses
    `gp_ctor`). Defaults to `gp_ctor`."""
    from .predictor import GPMultiHead, GBMMultiHead

    if gp_ctor is None:
        gp_ctor = lambda n: GPMultiHead(n_iter=n)  # noqa: E731
    if inner_gp_ctor is None:
        inner_gp_ctor = gp_ctor
    dataset_ids = np.asarray(dataset_ids)
    if verbose:
        print(f"[LODOcv] holdout = {held_out}  (n_train={len(train_idx)}, n_test={len(test_idx)})")

    # 0. Per-fold rank-normalization of the v block (fit on outer-train only).
    if rank_normalize_k and rank_normalize_k > 0:
        k = int(rank_normalize_k)
        sorted_train = _fit_rank_normalizer(X[train_idx][:, :k])
        X_fold = X.copy()
        X_fold[:, :k] = _apply_rank_normalizer(X[:, :k], sorted_train)
    else:
        X_fold = X

    # 1. Inner γ selection — uses GP only (GBM is fast; sharing γ across both is OK)
    if select_gamma_per_fold:
        gsel = select_gamma(
            lambda: inner_gp_ctor(gp_n_iter // 2),
            X_fold, e_obs, gap_obs, epsilon_emp, dataset_ids,
            train_idx=train_idx, gamma_grid=gamma_grid, n_iters=n_iters, verbose=verbose,
        )
        gamma = gsel.best_gamma
    else:
        gsel = None
        gamma = 1.0

    # 2. Outer fit
    gp = gp_ctor(gp_n_iter)
    gbm = GBMMultiHead(n_estimators=gbm_n_estimators)
    gp.fit(X_fold[train_idx], e_obs[train_idx], gap_obs[train_idx], epsilon_emp[train_idx],
           gamma=gamma, n_iters=n_iters)
    gbm.fit(X_fold[train_idx], e_obs[train_idx], gap_obs[train_idx], epsilon_emp[train_idx],
            gamma=gamma, n_iters=n_iters)

    # 3. Predict held-out
    gp_pred = gp.predict(X_fold[test_idx])
    gbm_pred = gbm.predict(X_fold[test_idx])
    edge_r2_gp = r2_score(e_obs[test_idx], gp_pred.e_hat)
    edge_r2_gbm = r2_score(e_obs[test_idx], gbm_pred.e_hat)
    edge_ece_gp = compute_ece(e_obs[test_idx], gp_pred.e_hat, gp_pred.sigma_e)
    gap_r2_gp = r2_score(gap_obs[test_idx], gp_pred.predicted_gap(epsilon_emp[test_idx]))
    gap_r2_gbm = r2_score(gap_obs[test_idx], gbm_pred.predicted_gap(epsilon_emp[test_idx]))

    if verbose:
        print(f"  edge R² (GP/GBM) = {edge_r2_gp:.3f} / {edge_r2_gbm:.3f}; "
              f"ECE_GP = {edge_ece_gp:.3f}; β_GP = {gp.beta:.3f}")

    return FoldResult(
        held_out_dataset=held_out, n_train=len(train_idx), n_test=len(test_idx),
        gamma_selection=gsel, edge_r2_gp=edge_r2_gp, edge_r2_gbm=edge_r2_gbm,
        edge_ece_gp=edge_ece_gp, gap_r2_gp=gap_r2_gp, gap_r2_gbm=gap_r2_gbm,
        e_hat_gp=gp_pred.e_hat, e_hat_gbm=gbm_pred.e_hat, a_hat_gp=gp_pred.a_hat,
        a_hat_gbm=gbm_pred.a_hat, sigma_e_gp=gp_pred.sigma_e, sigma_e_gbm=gbm_pred.sigma_e,
        e_obs_test=e_obs[test_idx], gap_obs_test=gap_obs[test_idx],
        eps_emp_test=epsilon_emp[test_idx], beta_gp=gp.beta, beta_gbm=gbm.beta,
    )
