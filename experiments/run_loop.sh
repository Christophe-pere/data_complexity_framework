#!/usr/bin/env bash
# Phase B background loop — drive the resumable sweep to completion.
#
# The sweep itself processes the entire pending matrix in one invocation and is
# crash-safe (every record is appended immediately; existing keys are skipped on
# restart). This loop is the supervisor: it re-invokes the sweep until a run
# appends zero new records (i.e. nothing left to do), retrying after any crash.
#
# Output is plain JSONL at $OUT, extracted after the run for Phase C analysis.
#
# Usage:
#   nohup bash experiments/run_loop.sh > experiments/results/loop.log 2>&1 &
#   WORKERS=6 OUT=experiments/results/phase_b.jsonl bash experiments/run_loop.sh
#
# Env knobs: PY (interpreter), OUT (jsonl path), WORKERS (procs), EPOCHS,
#            BATCH, LR, SEEDS, SLEEP (retry backoff seconds).
set -u

cd "$(dirname "$0")/.."

# Pin per-process BLAS/OpenMP to a single thread. With many worker processes the
# kernel-effective-rank eigendecomposition and the numpy quantum-circuit math
# otherwise spawn one BLAS thread *per core per worker*, massively oversubscribing
# the CPU and thrashing. One thread per worker keeps workers × 1 ≤ ncpu.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${PY:-/Users/christophe/research-dev/bin/python}"
OUT="${OUT:-experiments/results/phase_b.jsonl}"
WORKERS="${WORKERS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-32}"
LR="${LR:-0.1}"
SLEEP="${SLEEP:-10}"
SEEDS="${SEEDS:-0 1 2 3 4}"

echo "[loop] $(date) start — OUT=$OUT WORKERS=$WORKERS EPOCHS=$EPOCHS SEEDS='$SEEDS'"

iter=0
while true; do
  iter=$((iter + 1))
  echo "[loop] $(date) iteration $iter — invoking sweep"
  # Capture output so we can read back how many records were appended.
  log=$("$PY" -m experiments.sweep_phase_b \
          --output "$OUT" --workers "$WORKERS" \
          --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
          --seeds $SEEDS 2>&1)
  status=$?
  echo "$log"

  if [ $status -ne 0 ]; then
    echo "[loop] $(date) sweep exited $status — retrying in ${SLEEP}s"
    sleep "$SLEEP"
    continue
  fi

  appended=$(printf '%s\n' "$log" | sed -n 's/^Done: appended \([0-9][0-9]*\) records.*/\1/p' | tail -1)
  appended="${appended:-0}"
  echo "[loop] $(date) iteration $iter appended $appended records"
  if [ "$appended" = "0" ]; then
    echo "[loop] $(date) nothing pending — sweep complete. $(wc -l < "$OUT") total records in $OUT"
    break
  fi
done
