#!/bin/zsh
# Does quantum test error a_q unpin at circuit widths other than n=8?
#
# Part II's width arm (h1_gradients.jsonl, n in {4,6,8,10,12,14}) measured GRADIENT
# VARIANCE only. a_q was never measured off n=8, so the degeneracy that drove the H5
# null -- a_q = 0.481 +- 0.069 with sd(a_q) = 0.020 across datasets, making the edge
# e ~ a_c - 1/2 -- has never been tested for width dependence. This measures it.
#
# Two deliberate choices:
#
# 1. PER-WIDTH OUTPUT FILES. load_existing_keys() dedups on
#    (dataset_id, encoding_id, ansatz_id, seed) with NO n_qubits, so writing every
#    width into one JSONL would mark each cell "already done" and silently append
#    nothing -- the same silent no-op the locality sweep hit. Separate files sidestep
#    it, and keep the frozen pre-specified phase_b.jsonl untouched.
#
# 2. THE SAME 9-DATASET SUBSET as the gradient arm, so the two arms are comparable.
#
# Exploratory, in the specification's sense: this is not a pre-specified hypothesis
# and must not be reported as a confirmatory test.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
WAIT_PID=${1:-0}
if [[ $WAIT_PID -ne 0 ]]; then
  echo "[aq] waiting for pid $WAIT_PID (n=14 depth chain) to finish..."
  while kill -0 $WAIT_PID 2>/dev/null; do sleep 120; done
  echo "[aq] pid $WAIT_PID done at $(date '+%F %T')"
fi
DS=(adv-easy-looking-hard adv-hard-looking-easy real-breast-cancer real-higgs real-iris
    synth-concentric-rings synth-parity-3 synth-prng synth-torus)
for N in 4 6 10 12 14; do
  echo "[aq] === n=$N === $(date '+%F %T')"
  python3 -u experiments/sweep_phase_b.py \
    --datasets $DS --seeds 0 1 2 3 4 --n-qubits $N --workers 8 \
    --output experiments/results/widthsweep/phase_b_n${N}.jsonl
done
echo "[aq] all widths done $(date '+%F %T')"
