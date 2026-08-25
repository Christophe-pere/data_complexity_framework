"""Train and evaluate the GP+GBM selector via leave-one-dataset-out CV.

Until Phase B produces real (e_obs, gap_obs) data, this drives the harness
on synthetic training data — verifies the pipeline runs end-to-end and that
the planted β is recovered within tolerance. Replace `make_synthetic_training_set`
with a Phase B output loader once that data exists.

Usage:
    source ../../research-dev/bin/activate
    python -m analysis.train_predictor                # synthetic data, full LODOcv
    python -m analysis.train_predictor --quick        # smaller config for smoke test
    python -m analysis.train_predictor --no-gamma-cv  # skip inner γ CV (γ = 1.0)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make project root importable when run as `python analysis/train_predictor.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.cv import DEFAULT_GAMMA_GRID, run_lodocv  # noqa: E402
from qmlc.predictor import stack_training_points  # noqa: E402
from analysis.synthetic_training_data import (  # noqa: E402
    SyntheticConfig,
    make_synthetic_training_set,
)
from analysis.phase_b_training_data import (  # noqa: E402
    LoaderConfig,
    load_phase_b_training_set,
)


def _serialize_report(report, ground_truth) -> dict:
    """Convert LODOcvReport to a JSON-serializable summary dict."""
    summary = report.summary()
    fold_summaries = []
    for f in report.folds:
        gsel = None
        if f.gamma_selection is not None:
            gsel = {
                "best_gamma": f.gamma_selection.best_gamma,
                "scores": {str(g): float(s) for g, s in f.gamma_selection.scores.items()},
                "n_folds": f.gamma_selection.n_folds,
            }
        fold_summaries.append({
            "held_out_dataset": f.held_out_dataset,
            "n_train": f.n_train,
            "n_test": f.n_test,
            "edge_r2_gp": f.edge_r2_gp,
            "edge_r2_gbm": f.edge_r2_gbm,
            "edge_ece_gp": f.edge_ece_gp,
            "gap_r2_gp": f.gap_r2_gp,
            "gap_r2_gbm": f.gap_r2_gbm,
            "beta_gp": f.beta_gp,
            "beta_gbm": f.beta_gbm,
            "gamma_selection": gsel,
        })
    return {
        "ground_truth": ground_truth,
        "summary": summary,
        "folds": fold_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train predictor (GP + GBM) via LODOcv")
    parser.add_argument("--quick", action="store_true",
                        help="smaller config: fewer encodings, ansatze, seeds, GP iters")
    parser.add_argument("--no-gamma-cv", action="store_true",
                        help="skip inner γ CV; use γ = 1.0 directly")
    parser.add_argument("--output", default=None,
                        help="JSON output path (default: print summary only)")
    parser.add_argument("--registry", default="datasets/registry.json",
                        help="path to dataset registry JSON")
    parser.add_argument("--data", choices=("synthetic", "phase_b"), default="synthetic",
                        help="training-data source (synthetic harness vs real Phase B JSONL)")
    parser.add_argument("--jsonl", default="experiments/results/phase_b.jsonl",
                        help="Phase B results JSONL (used when --data phase_b)")
    parser.add_argument("--edge-metric", default="raw", choices=("raw", "budget", "pareto"),
                        help="edge target; ê_budget is the frozen primary once calibrated")
    parser.add_argument("--gp", default="exact", choices=("exact", "svgp"),
                        help="GP backend: exact gpytorch (O(n³), smoke-scale) or svgp "
                             "(linear-scaling variational LMC GP, for full Phase B)")
    parser.add_argument("--inducing", type=int, default=256,
                        help="SVGP inducing points (only used with --gp svgp)")
    parser.add_argument("--num-latents", type=int, default=2,
                        help="SVGP LMC latent GPs (only used with --gp svgp)")
    args = parser.parse_args()

    # SVGP needs more optimizer steps than exact GP to converge its variational params.
    if args.gp == "svgp":
        gp_n_iter = 200 if args.quick else 600
        from qmlc.predictor_svgp import SVGPMultiHead
        gp_ctor = (lambda n: SVGPMultiHead(n_iter=n, num_inducing=args.inducing,
                                           num_latents=args.num_latents))
    else:
        gp_n_iter = 80 if args.quick else 200
        gp_ctor = None  # run_lodocv defaults to exact GPMultiHead

    if args.data == "phase_b":
        print(f"Loading Phase B training data from {args.jsonl} (edge_metric={args.edge_metric})…")
        # Emit RAW v; the v block is rank-normalized per-fold inside LODOcv
        # (leakage-clean), so the held-out dataset never informs its own scaling.
        points, ground_truth = load_phase_b_training_set(
            args.jsonl, LoaderConfig(edge_metric=args.edge_metric, normalize_v=False),
        )
        rank_normalize_k = ground_truth["K"]
        print(f"  {ground_truth['n_points']} points across {ground_truth['n_datasets']} datasets "
              f"({ground_truth['n_encodings']} enc × {ground_truth['n_ansatze']} ans); "
              f"feature dim = {ground_truth['K'] + ground_truth['M']}; "
              f"per-fold rank-norm of {rank_normalize_k} v-cols")
    else:
        if args.quick:
            cfg = SyntheticConfig(
                encodings=("angle-linear", "zz-r1-linear", "amplitude"),
                ansatze=("hea-L1", "alt-L1"),
                n_seeds=2,
                seed=0,
            )
        else:
            cfg = SyntheticConfig(seed=0)
        print(f"Generating synthetic training data (config: "
              f"{len(cfg.encodings)} encodings × {len(cfg.ansatze)} ansatze × {cfg.n_seeds} seeds)…")
        points, ground_truth = make_synthetic_training_set(args.registry, cfg)
        # Synthetic v is already globally rank-normalized (harness validation only,
        # not a confirmatory analysis) → no per-fold re-normalization.
        rank_normalize_k = 0
        print(f"  {ground_truth['n_points']} points across {ground_truth['n_datasets']} datasets; "
              f"feature dim = {ground_truth['K'] + ground_truth['M']}")
        print(f"  planted β = {ground_truth['beta_planted']}")

    stacked = stack_training_points(points)
    X = stacked["X"]
    e_obs = stacked["e_obs"]
    gap_obs = stacked["gap_obs"]
    eps_emp = stacked["epsilon_emp"]
    dataset_ids = stacked["dataset_id"]

    print()
    print(f"Running LODOcv across {len(np.unique(dataset_ids))} datasets… "
          f"(GP backend = {args.gp}"
          + (f", {args.inducing} inducing, {args.num_latents} latents" if args.gp == "svgp" else "")
          + f", γ-CV = {not args.no_gamma_cv}, gp_n_iter = {gp_n_iter})")
    report = run_lodocv(
        X=X,
        e_obs=e_obs,
        gap_obs=gap_obs,
        epsilon_emp=eps_emp,
        dataset_ids=dataset_ids,
        gamma_grid=DEFAULT_GAMMA_GRID,
        select_gamma_per_fold=not args.no_gamma_cv,
        n_iters=3,
        gp_n_iter=gp_n_iter,
        gbm_n_estimators=200,
        rank_normalize_k=rank_normalize_k,
        gp_ctor=gp_ctor,
        verbose=True,
    )

    summary = report.summary()
    beta_gp_mean = float(np.nanmean([f.beta_gp for f in report.folds]))
    beta_gbm_mean = float(np.nanmean([f.beta_gbm for f in report.folds]))
    print()
    print("=" * 60)
    print("LODOcv summary")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30s} = {v:+.4f}")
        else:
            print(f"  {k:30s} = {v}")
    planted = ground_truth.get("beta_planted")
    planted_str = f"  (planted = {planted:+.4f})" if planted is not None else "  (real data — no planted β)"
    print(f"  {'beta_gp_mean':30s} = {beta_gp_mean:+.4f}{planted_str}")
    print(f"  {'beta_gbm_mean':30s} = {beta_gbm_mean:+.4f}")

    if args.output:
        out = _serialize_report(report, ground_truth)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
