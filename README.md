# Data Complexity Framework

Code for **[Does Data Complexity Predict Quantum Advantage? A framework, a pre-specified test, and an attribution of the measured edge](https://arxiv.org/abs/2509.16410)** (Christophe Pere, arXiv:2509.16410v2, 2026).

Version 1 of the paper (2025) proposed *data complexity* — the structural, statistical, algorithmic and topological richness of a dataset — as the axis that predicts when quantum machine learning beats classical ML. Version 2 corrects that framework explicitly and then tests its central predictive claim on a pre-specified protocol. **The claim fails**, and the paper's value is in why. Version 1 remains archived and citable on arXiv.

This repository contains the source code only: the metric library, the experiment runners, the analysis scripts and the frozen dataset registry. Results, specifications and the manuscript are not tracked here (see [Data](#data)).

## Headline results

The confirmatory study covers 27 datasets × 10 encodings × 6 ansätze × 5 seeds (8,100 runs) on 8-qubit circuits, simulated exactly and noiselessly.

| Hypothesis | Outcome |
|---|---|
| **H5** (primary): data complexity predicts the budget-matched quantum edge | **Falsified.** Pooled leave-one-dataset-out *R²* = −0.32 against a pre-specified threshold of 0.4. |
| **H1**: data-modulated barren-plateau law | **Fails.** Pooled LODO *R²* = +0.026 against 0.70; 2 of 9 complexity components detectable. |
| **H2**: topological contribution | **Withdrawn.** The pre-specified estimator was defective. Re-runs with the corrected estimator are inconclusive (App. B). |
| **H3**: expressibility–data alignment | Passes its literal criterion (positive slope, CI excluding zero), but the alignment head has gap *R²* = −0.013. |
| **H4**: per-feature encoding optimality | Untested (downstream of H5). |
| **H6**: dequantization-stratified validity | Not evaluable: the suite contains no QRAM-mode cells. |

What the null teaches:

1. **The classical baseline makes the edge (Sec. 7.2).** Against logistic regression alone, the selector reaches *R²* = +0.13 and the quantum model wins 17% of cells. Against the best of six standard classical models (LR, kNN, RBF-SVC, MLP, HistGBM, RF), *R²* falls to −0.94 and the win rate to 1.1%. Any single non-linear baseline already pushes the win rate below 6%.
2. **Only a residual is a quantum claim (Props. 1–2).** Any reported gap splits into a preprocessing-attributable term and a residual, and an encoding cannot add label information, *I*(𝓔(*X*); *Y*) ≤ *I*(*X*; *Y*). Classical preprocessing leaves little non-linear headroom on real datasets (Sec. 7.3).
3. **The regime is pinned (Secs. 7.5, 7.8).** The quantum model sits at or near chance in 81.5% of cells, so the measured edge is driven by the classical term (ρ = +0.913). The encoding axis carries 0.5% of the variance, and widths *n* = 4–10 do not move it.
4. **An edge can exist and still be unreachable (Sec. 7.12).** A designed weight-*w* parity control yields a +0.42 edge for *w* ≥ 4 under a structure-aware measurement. A hardware-efficient ansatz and a trainable measurement-selection head both stay at chance. A selector trained on gradient-trained outcomes therefore misses exactly the advantages worth having.

Exploratory: gradient variance decays as a power law, Var[*g*] ∼ *n*⁻²·¹⁵, independent of depth and slower than the Haar-random rate (Sec. 7.6).

## Instrument audit: read this before reusing a metric

Two of the v1 complexity components turned out to be scale-and-dimension statistics in disguise:

- **Topology.** The v1 topological component had four compounding defects (App. B) and returns zero persistence for a clean circle. It is kept unchanged as `PersistentHomology` so the pre-specified numbers stay reproducible. `NormalisedPersistentHomology` is the corrected, shape-validated replacement.
- **Differential entropy.** It is extensive in dimension and scale-dependent.

As a result, the v1 composite is effectively a one-component quantity, and which component dominates depends on an estimator detail. Per-component rank normalisation is what makes the components commensurable (Sec. 4). The quantum composite collapses the same way, onto the kernel effective rank.

The general lesson: validate metrics **as called**. In v1 the validated configuration was never the deployed one. `python -m qmlc.validate` checks every component, classical and quantum, against closed-form testbeds using the arguments production uses.

## Layout

```
qmlc/          metric library: classical and quantum complexity components,
               encodings, ansätze, predictor, classical shadows, budget model,
               structure-search battery, validation against closed forms
experiments/   Phase B/C runners, feature extraction, class-B (quantum-native)
               arm, circuit-width sweeps
analysis/      hypothesis tests, baseline ladder, attribution and headroom,
               ablations, table and figure generation
scripts/       dataset registry builder and taxonomy annotation
datasets/      registry.json: frozen registry v0.5.0-class-abcd-32
               (32 datasets, SHA-256 pinned)
legacy/        the 2025 v1-era prototype, superseded, kept for provenance
```

## Reproducing

```bash
python -m qmlc.validate                       # metric library vs closed-form testbeds
bash experiments/run_loop.sh                  # Phase B sweep (resumable)
bash experiments/run_phase_c.sh               # confirmatory Phase C (per fold, resumable)
python -m analysis.h1_h2_analysis             # H1 / H2
python analysis/h5_topology_ablation.py       # H5 sensitivity to the topology component
```

| Paper section | Script |
|---|---|
| Sec. 4, quantum composite collapse | `analysis/cquant_part2.py`, `analysis/cquant_shares.py` |
| Sec. 7.2, baseline ladder | `analysis/baseline_ladder.py` |
| Sec. 7.3, reducer cost and headroom | `analysis/make_part2_encoder_tables.py` |
| Sec. 7.7, refit on the corrected baseline | `analysis/h1_refit_corrected_baseline.py` |
| Sec. 7.8, width dependence of *a_q*; correlation-dimension check | `experiments/run_aq_width_sweep.sh`, `analysis/aq_width_dependence.py`, `analysis/correlation_dimension.py` |
| Sec. 7.12, designed quantum-signature control | `experiments/sweep_phase_b_classB.py`, `qmlc/structure_search.py` |

Analysis scripts read results from `experiments/results/` and some write LaTeX tables to `paper2/tables/`. Neither directory is tracked, so create them locally before running.

**Environment.** Python 3.12 with `numpy` 1.26.4, `scipy` 1.13.1, `scikit-learn` 1.6.1, `torch` 2.7.1, `gpytorch` 1.15.2, `lightgbm` 4.6.0, `ripser` 0.6.15 and `gudhi` (for the pre-specified topological component). Quantum simulation is exact state-vector in `numpy`. **No quantum hardware was used.**

## Data

- **Datasets.** The registry was frozen on 2026-06-14. Every dataset is generated deterministically by `qmlc.datasets` and checked against the SHA-256 recorded in `datasets/registry.json`. `python -m scripts.build_registry` rebuilds and re-screens the registry.
- **Not tracked here:** raw Phase B/C results, logs, fold outputs, specification documents, the deviations log and the manuscript. The paper is on [arXiv](https://arxiv.org/abs/2509.16410).

## Provenance, stated precisely

The hypotheses, decision rules, thresholds and cross-validation scheme were fixed in a written, dated specification before any confirmatory analysis was run. One revision, the classical-baseline definition, was made and logged before the confirmatory run. **No public pre-registration was filed.** The internal record (dated specification entries, deviations log, content hashes) is the evidence for that ordering. It is not part of this code-only repository, and this repository's commit history does not itself evidence it.

## Citation

```bibtex
@article{pere2026datacomplexity,
  title   = {Does Data Complexity Predict Quantum Advantage? A framework, a pre-specified test, and an attribution of the measured edge},
  author  = {Pere, Christophe},
  journal = {arXiv preprint arXiv:2509.16410},
  year    = {2026},
  note    = {Version 2; supersedes ``Data Complexity: a threshold between Classical and Quantum Machine Learning'' (v1, 2025)}
}
```

## License

See [`LICENSE`](LICENSE).
