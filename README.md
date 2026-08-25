# Data Complexity Framework

Implementation, specifications and analysis for the **Data Complexity** series on when — and whether — data structure predicts a quantum machine-learning advantage.

- **Part I** (2025), *Data Complexity: a threshold between Classical and Quantum Machine Learning*, [arXiv:2509.16410](https://arxiv.org/abs/2509.16410) — proposes data complexity as the missing axis, with a four-pillar composite measure.
- **Part II** (in preparation), *what a pre-specified falsification teaches about measuring quantum advantage* — the empirical test. **The central hypothesis is falsified**, and this repository holds everything needed to check that.

**This repository holds code, specifications and analysis only.** The manuscript is maintained separately and will be linked here once it is on arXiv.

## Headline results

| | Outcome |
|---|---|
| **H5** (primary): complexity ranks the budget-matched quantum edge | **Falsified.** Pooled leave-one-dataset-out *R²* = −0.32 against a pre-specified threshold of 0.4, fold-bootstrap CI entirely below it. |
| **H1**: barren-plateau scaling law | Fails. Pooled LODO *R²* = +0.026 against 0.70. Gradient variance decays as a **power law** in circuit width (≈ *n*⁻²·²), not exponentially. |
| **H2**: topological contribution | **Inconclusive — the pre-specified test is withdrawn.** The estimator was measuring dispersion and ambient scale, not topology. See below. |
| **H3**: expressibility–data alignment | Weakest result in the battery; the secondary criterion was lost to a serialisation defect. |
| **H4**: per-feature encoding optimality | Untested; the precondition is stated in the paper. |
| **H6**: dequantization-stratified validity | Descriptive by construction — the suite contains no Mode-Q pairs. |

Three findings survive the null, and are the work's actual contribution:

1. **The classical baseline, not the quantum model, determines measured advantage in the low-signal regime.** Against logistic regression alone the selector predicts the edge (*R²* = +0.13, 17% quantum win rate); against the best of six standard classical models the same pipeline gives *R²* = −0.94 and a 1% win rate.
2. **The advantage that survives a fair baseline is unreachable by gradient training.** A shadow-hard parity probe yields an edge of +0.42, switching on exactly as the label weight crosses the shadow locality — yet a generic variational circuit sits in a barren plateau, and a trainable measurement-selection head fails beyond parity weight two.
3. **The revised barren-plateau law is misspecified in the regime experiments can reach.** Ansatz depth contributes nothing detectable, and the fitted decay rate excludes the Haar-random value.

## Instrument audit — read this before reusing any metric

Two of Part I's four complexity pillars turned out to be **scale-and-dimension statistics in disguise**:

- The topological component was defective in four compounding ways: *H*₀ dispersion dominating the value, no scale normalisation, an over-truncated filtration, and an edge-collapse step deleting the essential classes truncation had manufactured. **It returns zero persistence for a clean circle.** It is retained unchanged as `PersistentHomology` so the pre-specified numbers stay reproducible; `NormalisedPersistentHomology` is the corrected, shape-validated replacement.
- Differential entropy is extensive in dimension and scale-dependent: over the 27-dataset suite *S* ≈ *d*·log(diam) with *R²* = 0.999.

Consequently Part I's composite *C*_data (Eq. 7) is a **one-pillar quantity wearing a sum**, and which pillar it collapses to depends on an estimator detail: 100% of its variance sits on topology with the deployed estimator, and 100% on entropy once that estimator is repaired. Per-component rank normalisation is what makes the four pillars commensurable — variance shares move from 100/0/0/0 to 25.00/25.00/24.99/25.00.

The root cause is worth generalising: **the validated configuration was never the deployed one.** The validation suite exercised the estimator with different arguments than production used. Validate metrics *as called*.

Full audit in [`docs/deviations_log.md`](docs/deviations_log.md).

## Layout

```
qmlc/            metric library — classical and quantum complexity components,
                 encodings, ansaetze, predictor, classical shadows, budget model
experiments/     Phase B/C runners, feature extraction, Class B quantum-native arm
analysis/        hypothesis tests, figures, tables, instrument audits, ablations
docs/            framework_v2, encoding and ansatz libraries, deviations log,
                 Class B specification, Part III design notes
datasets/        frozen registry v0.5.0-class-abcd-32 (32 datasets, SHA-256 pinned)
legacy/          the 2025 Part I-era prototype, superseded, kept for provenance
```

## Reproducing

```bash
python -m qmlc.validate                      # metric library against closed-form testbeds
bash experiments/run_loop.sh                 # Phase B sweep (resumable)
bash experiments/run_phase_c.sh              # confirmatory Phase C (per-fold, resumable)
python -m analysis.h1_h2_analysis            # H1 / H2
python analysis/h5_topology_ablation.py      # H5 sensitivity to the topology component
```

**Environment.** Python 3.12 with `numpy` 1.26.4, `scipy` 1.13.1, `scikit-learn` 1.6.1, `torch` 2.7.1, `gpytorch` 1.15.2, `lightgbm` 4.6.0, `ripser` 0.6.15, `gudhi`. Quantum simulation is exact state-vector in numpy — **no quantum hardware was used**.

## Data availability

The dataset registry was frozen on 2026-06-14; every dataset carries its own SHA-256 and is regenerated deterministically from `datasets/registry.json`.

Raw Phase B/C result files (~108 MB) are **not tracked here** and are released separately at publication; `experiments/results/phase_b_classB.jsonl` (the quantum-native arm) is small enough to include. Files with a `_topofix` suffix carry the corrected topological component, retaining the original value in each record as `value_original`. The manuscript is likewise kept out of this repository by design.

## Provenance, stated precisely

The hypotheses, decision rules, thresholds and cross-validation scheme were fixed in a written, dated specification before any confirmatory analysis was run, and one revision (the classical-baseline definition) was made and logged before the confirmatory run. **No public pre-registration was filed** — the practice is not yet standard in this field. What is externally checkable is the internal record: dated specification entries, the deviations log, and content hashes. This repository's commit history begins at publication and does not itself evidence that ordering. This is the weaker, in-house form and should be read as such.

## Citation

```bibtex
@article{pere2025datacomplexity,
  title   = {Data Complexity: a threshold between Classical and Quantum Machine Learning},
  author  = {Pere, Christophe},
  journal = {arXiv preprint arXiv:2509.16410},
  year    = {2025}
}
```

## License

See [`LICENSE`](LICENSE).
