#!/bin/zsh
# Continuation of the a_q width sweep for n = 10, 12, 14.
#
# RE-UPLOADING ENCODINGS EXCLUDED. At n=10 the cells reup-2/reup-4 do not complete:
# every one of the 8 workers stalled on its group's 50th cell (the first reup-2 x alt-L3),
# 8 non-reup encodings having finished in ~12 s/cell, and nothing was written for 2h43m
# at ~8% CPU per worker. n=4 and n=6 completed all 10 encodings without incident, so the
# pathology appears at n>=10 only. It is recorded as an open anomaly, not diagnosed.
#
# The exclusion is PRAGMATIC (they do not finish), not principled -- reup encodings do
# produce a test error and do belong in the population. To keep the comparison honest the
# n=8 baseline is recomputed on the SAME 8 encodings; excluding them there moves sd across
# datasets from 0.026 to 0.019, i.e. the conclusion is unchanged either way, so this is
# not a selection that favours the result.
#
# Resumes: the dedup key (dataset, encoding, ansatz, seed) is correct WITHIN one width
# file, so the 392 cells already written at n=10 are kept and skipped.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
DS=(adv-easy-looking-hard adv-hard-looking-easy real-breast-cancer real-higgs real-iris
    synth-concentric-rings synth-parity-3 synth-prng synth-torus)
ENC=(amplitude angle-linear angle-none angle-ring
     zz-r1-full zz-r1-linear zz-r2-full zz-r2-linear)
for N in 10 12 14; do
  echo "[aq2] === n=$N (8 encodings, reup excluded) === $(date '+%F %T')"
  python3 -u experiments/sweep_phase_b.py \
    --datasets $DS --encodings $ENC --seeds 0 1 2 3 4 --n-qubits $N --workers 8 \
    --output experiments/results/widthsweep/phase_b_n${N}.jsonl
done
echo "[aq2] all widths done $(date '+%F %T')"
