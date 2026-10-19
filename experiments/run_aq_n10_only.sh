#!/bin/zsh
# Finish n=10 ONLY, then exit so the depth-64 chain can start.
#
# n=12 (~46 h) and n=14 (~172 h) are dropped. The decision rule needs sd across datasets
# above ~0.05 with near-chance below 0.8; n=4, 6 and 8 give 0.019 / 0.020 / 0.026 at
# 0.803 / 0.820 / 0.793 -- nothing marginal. Four unanimous widths spanning a 2.5x range in
# qubit count is a defensible basis, and n=12/14 would have to contradict all four to matter.
# Depth 64, which decides the window-vs-boundary question, is the higher-value use of the
# machine. n=12/14 can be run later if the write-up needs the wider range.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
DS=(adv-easy-looking-hard adv-hard-looking-easy real-breast-cancer real-higgs real-iris
    synth-concentric-rings synth-parity-3 synth-prng synth-torus)
ENC=(amplitude angle-linear angle-none angle-ring zz-r1-full zz-r1-linear zz-r2-full zz-r2-linear)
echo "[aq-n10] resuming n=10 $(date '+%F %T')"
python3 -u experiments/sweep_phase_b.py \
  --datasets $DS --encodings $ENC --seeds 0 1 2 3 4 --n-qubits 10 --workers 8 \
  --output experiments/results/widthsweep/phase_b_n10.jsonl
echo "[aq-n10] done $(date '+%F %T') -- n=12 and n=14 deliberately skipped"
