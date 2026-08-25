"""Phase C — one LODO fold as an independent, resumable process.

The confirmatory selector run is 27 outer folds × inner γ-CV (leave-one-additional-out × 5 γ
× 3 EM SVGP fits) — days of compute. The outer folds are independent, so we run each as its own
process writing `<out-dir>/<holdout>.json`. The orchestrator (`experiments/run_phase_c.sh`) skips
folds whose JSON already exists (crash-safe / resumable) and dispatches them with bounded
parallelism; `experiments/aggregate_phase_c.py` builds the final report.

Usage:
  python -m experiments.run_phase_c_fold --holdout real-iris \
    --jsonl experiments/results/phase_b_best_classical.jsonl --edge-metric budget \
    --gp svgp --inducing 256 --num-latents 2 --out-dir analysis/phase_c_folds
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmlc.cv import DEFAULT_GAMMA_GRID, run_single_fold  # noqa: E402
from qmlc.predictor import stack_training_points  # noqa: E402
from analysis.phase_b_training_data import LoaderConfig, load_phase_b_training_set  # noqa: E402


def _serialize_fold(f) -> dict:
    gsel = None
    if f.gamma_selection is not None:
        gsel = {"best_gamma": f.gamma_selection.best_gamma,
                "scores": {str(g): float(s) for g, s in f.gamma_selection.scores.items()},
                "n_folds": f.gamma_selection.n_folds}
    return {
        "held_out_dataset": f.held_out_dataset, "n_train": f.n_train, "n_test": f.n_test,
        "edge_r2_gp": f.edge_r2_gp, "edge_r2_gbm": f.edge_r2_gbm, "edge_ece_gp": f.edge_ece_gp,
        "gap_r2_gp": f.gap_r2_gp, "gap_r2_gbm": f.gap_r2_gbm,
        "beta_gp": f.beta_gp, "beta_gbm": f.beta_gbm, "gamma_selection": gsel,
        # per-point predictions (held-out dataset) for downstream analysis / CIs
        "e_hat_gp": f.e_hat_gp.tolist(), "e_hat_gbm": f.e_hat_gbm.tolist(),
        "sigma_e_gp": f.sigma_e_gp.tolist(), "e_obs_test": f.e_obs_test.tolist(),
        # Alignment head + observed gap. REQUIRED by H3's registered SECONDARY
        # criterion (Spearman rho between predicted a(D,U) and observed -gap_obs
        # on held-out folds, threshold 0.3). These were computed by run_single_fold
        # but not persisted before 2026-08-20, which made that criterion
        # un-evaluable on the 2026-07-10 and 2026-08-19 runs. Serialization only —
        # no estimator, decision rule or numerical output is changed.
        "a_hat_gp": f.a_hat_gp.tolist(), "a_hat_gbm": f.a_hat_gbm.tolist(),
        "gap_obs_test": f.gap_obs_test.tolist(),
        "eps_emp_test": f.eps_emp_test.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Phase C single LODO fold")
    p.add_argument("--holdout", required=True)
    p.add_argument("--jsonl", default="experiments/results/phase_b_best_classical.jsonl")
    p.add_argument("--edge-metric", default="budget", choices=("raw", "budget", "pareto"))
    p.add_argument("--gp", default="svgp", choices=("exact", "svgp"))
    p.add_argument("--inducing", type=int, default=256)
    p.add_argument("--inner-inducing", type=int, default=64,
                   help="cheaper SVGP inducing points for the γ-ranking inner CV (outer fit uses "
                        "--inducing at full fidelity)")
    p.add_argument("--num-latents", type=int, default=2)
    p.add_argument("--no-gamma-cv", action="store_true")
    p.add_argument("--out-dir", default="analysis/phase_c_folds")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.holdout}.json"
    if out_path.exists():
        print(f"[fold] {args.holdout}: exists, skipping"); return 0

    points, gt = load_phase_b_training_set(
        args.jsonl, LoaderConfig(edge_metric=args.edge_metric, normalize_v=False))
    rank_normalize_k = gt["K"]
    st = stack_training_points(points)
    X, e_obs, gap_obs = st["X"], st["e_obs"], st["gap_obs"]
    eps_emp, dataset_ids = st["epsilon_emp"], np.asarray(st["dataset_id"])
    if args.holdout not in set(dataset_ids):
        raise ValueError(f"holdout {args.holdout!r} not in data ({sorted(set(dataset_ids))})")

    test_idx = np.where(dataset_ids == args.holdout)[0]
    train_idx = np.where(dataset_ids != args.holdout)[0]

    inner_gp_ctor = None
    if args.gp == "svgp":
        from qmlc.predictor_svgp import SVGPMultiHead
        gp_n_iter = 600
        gp_ctor = lambda n: SVGPMultiHead(n_iter=n, num_inducing=args.inducing,
                                          num_latents=args.num_latents)
        # γ-ranking only ⇒ cheaper GP (fewer inducing pts, capped iters) for the inner CV.
        inner_gp_ctor = lambda n: SVGPMultiHead(n_iter=min(n, 150),
                                                num_inducing=args.inner_inducing,
                                                num_latents=args.num_latents)
    else:
        gp_n_iter = 200
        gp_ctor = None

    print(f"[fold] {args.holdout}: n_train={len(train_idx)} n_test={len(test_idx)} "
          f"gp={args.gp} γ-CV={not args.no_gamma_cv}")
    t0 = time.perf_counter()
    fold = run_single_fold(
        args.holdout, train_idx, test_idx, X, e_obs, gap_obs, eps_emp, dataset_ids,
        gamma_grid=DEFAULT_GAMMA_GRID, select_gamma_per_fold=not args.no_gamma_cv,
        n_iters=3, gp_n_iter=gp_n_iter, gbm_n_estimators=200,
        rank_normalize_k=rank_normalize_k, gp_ctor=gp_ctor, inner_gp_ctor=inner_gp_ctor,
        verbose=True,
    )
    rec = _serialize_fold(fold)
    rec["wall_time_s"] = time.perf_counter() - t0
    rec["config"] = {"gp": args.gp, "inducing": args.inducing, "num_latents": args.num_latents,
                     "edge_metric": args.edge_metric, "gamma_cv": not args.no_gamma_cv,
                     "jsonl": args.jsonl}
    # atomic write
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2)); tmp.replace(out_path)
    print(f"[fold] {args.holdout}: done in {rec['wall_time_s']:.0f}s → {out_path} "
          f"(edge R² GP={fold.edge_r2_gp:+.3f} GBM={fold.edge_r2_gbm:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
