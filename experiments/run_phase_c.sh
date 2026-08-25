#!/usr/bin/env bash
# Phase C — confirmatory selector run (H5/H6) on the amended pipeline.
#   edge = ê_budget vs best-of-standard-classical  (phase_b_best_classical.jsonl)
#   GP   = SVGP (linear-scaling LMC variational GP; recovers exact GP, Pearson 0.984)
#   full LODO (27 folds) + inner leave-one-additional-out γ-CV (5-grid) + 3-step EM
#
# Parallel + resumable: each outer fold is an independent process writing
# analysis/phase_c_folds/<holdout>.json; folds already present are skipped, so a crash/restart
# resumes. ~9h/fold single-thread ⇒ ~10 days serial, ~1.5–2 days at WORKERS=6.
# OSF Amendment 1 is filed, so this run is confirmatory.
#
# Usage:
#   nohup bash experiments/run_phase_c.sh > experiments/results/phase_c.log 2>&1 &
#   WORKERS=6 bash experiments/run_phase_c.sh
set -u
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

PY="${PY:-/Users/christophe/research-dev/bin/python}"
JSONL="${JSONL:-experiments/results/phase_b_best_classical.jsonl}"
FOLDS_DIR="${FOLDS_DIR:-analysis/phase_c_folds}"
OUT="${OUT:-analysis/predictor_confirmatory_best_classical.json}"
WORKERS="${WORKERS:-6}"
INDUCING="${INDUCING:-256}"
mkdir -p "$FOLDS_DIR"

echo "[phase-c] $(date) start — WORKERS=$WORKERS INDUCING=$INDUCING JSONL=$JSONL"

# Enumerate the held-out datasets (classical track = the 27 in the best-classical JSONL).
HOLDOUTS=$("$PY" -c "import json,sys; print('\n'.join(sorted({json.loads(l)['dataset_id'] for l in open('$JSONL')})))")
NTOTAL=$(printf '%s\n' "$HOLDOUTS" | grep -c .)
echo "[phase-c] $NTOTAL folds; $(ls "$FOLDS_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ') already done"

# Dispatch folds with bounded parallelism. run_phase_c_fold self-skips completed folds.
printf '%s\n' "$HOLDOUTS" | PYTHONPATH=. xargs -P "$WORKERS" -I{} \
  "$PY" -u -m experiments.run_phase_c_fold \
    --holdout {} --jsonl "$JSONL" --edge-metric budget \
    --gp svgp --inducing "$INDUCING" --num-latents 2 --out-dir "$FOLDS_DIR"

echo "[phase-c] $(date) all folds dispatched; aggregating"
PYTHONPATH=. "$PY" -m experiments.aggregate_phase_c \
  --folds-dir "$FOLDS_DIR" --jsonl "$JSONL" --output "$OUT"
echo "[phase-c] $(date) done — report at $OUT"
