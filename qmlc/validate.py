"""Validation runner — pairs every metric with relevant testbeds.

Outputs a table of (testbed, metric, estimator, value, ground truth, |err|, time).
Use this to certify both the metric implementation (matches ground truth) and
the proxy/exact trade-off (cost vs accuracy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BaseMetric, MetricResult
from .classical import (
    CompressionRatio,
    EntropyKDE,
    EntropyKL,
    HigherOrderMI,
    IntrinsicDimPCA,
    IntrinsicDimTwoNN,
    KernelEffectiveRankRBF,
    MDLAutoencoder,
    PairwiseMutualInformation,
    PersistentHomology,
)
from .quantum import (
    EntanglementEntropyExact,
    EntanglementEntropyRenyi2,
    ExpressibilityKL,
    KernelEffectiveRank,
)
from . import testbeds as tb


@dataclass
class ValidationRow:
    testbed: str
    metric: str
    estimator: str
    value: float
    uncertainty: float | None
    ground_truth: float | None
    abs_error: float | None
    rel_error: float | None
    elapsed_ms: float


def _row(testbed_name: str, gt: float | None, res: MetricResult) -> ValidationRow:
    abs_err = None if gt is None else abs(res.value - gt)
    rel_err = None
    if gt is not None and abs(gt) > 1e-9:
        rel_err = abs_err / abs(gt)
    return ValidationRow(
        testbed=testbed_name,
        metric=res.name,
        estimator=res.estimator,
        value=res.value,
        uncertainty=res.uncertainty,
        ground_truth=gt,
        abs_error=abs_err,
        rel_error=rel_err,
        elapsed_ms=res.elapsed_s * 1000.0,
    )


def run_classical() -> list[ValidationRow]:
    rows: list[ValidationRow] = []

    # ---- entropy
    for d in (1, 2, 4):
        bed = tb.make_gaussian(n=2000, d=d, seed=0)
        gt = bed.ground_truth["differential_entropy"]
        for metric in (EntropyKDE(), EntropyKL(k=4)):
            res = metric.compute(bed.data, ground_truth=gt)
            rows.append(_row(bed.name, gt, res))

    bed = tb.make_uniform_box(n=2000, d=2, seed=0)
    gt = bed.ground_truth["differential_entropy"]
    for metric in (EntropyKDE(), EntropyKL(k=4)):
        res = metric.compute(bed.data, ground_truth=gt)
        rows.append(_row(bed.name, gt, res))

    # ---- intrinsic dim
    for d_int in (1, 2, 3):
        bed = tb.make_sphere(n=1500, d_intrinsic=d_int, d_ambient=8, seed=0)
        gt = bed.ground_truth["intrinsic_dim"]
        for metric in (IntrinsicDimPCA(), IntrinsicDimTwoNN()):
            res = metric.compute(bed.data, ground_truth=gt)
            rows.append(_row(bed.name, gt, res))

    # ---- compression
    for bed in (
        tb.make_pseudorandom_bytes(n_bytes=8192, seed=0),
        tb.make_redundant_string(n_bytes=8192, period=8),
    ):
        gt = bed.ground_truth["compression_ratio"]
        for codec in ("gzip", "zlib", "lzma"):
            res = CompressionRatio(codec=codec).compute(bed.data, ground_truth=gt)
            rows.append(_row(bed.name, gt, res))

    # ---- pairwise mutual information (v_corr2): bivariate-gaussian closed form
    for rho in (0.3, 0.7):
        bed = tb.make_correlated_gaussian_pair(n=4000, rho=rho, seed=0)
        gt = bed.ground_truth["second_order_mi"]
        res = PairwiseMutualInformation(k=3, seed=0).compute(bed.data, ground_truth=gt)
        rows.append(_row(bed.name, gt, res))

    # ---- higher-order MI via MINE (v_corr3+): equicorrelated-gaussian total corr.
    # DV is a lower bound, so under-reporting is expected; we report value vs truth.
    for k in (3, 4):
        bed = tb.make_equicorrelated_gaussian(n=4000, k=k, rho=0.6, seed=0)
        gt = bed.ground_truth["higher_order_mi"]
        res = HigherOrderMI(orders=(k,), n_subsets=1, epochs=300, seed=0).compute(
            bed.data, ground_truth=gt
        )
        rows.append(_row(bed.name, gt, res))

    # ---- MDL autoencoder (v_K): relative ordering. A full-rank gaussian blob in
    # R^6 is less compressible than the same points confined to a 1-D line in R^6.
    rng = np.random.default_rng(0)
    full = rng.standard_normal((1500, 6))
    t = rng.standard_normal((1500, 1))
    line = t * rng.standard_normal((1, 6)) + 0.01 * rng.standard_normal((1500, 6))
    mdl = MDLAutoencoder(epochs=400, seed=0)
    l_full = mdl.compute(full).value
    l_line = mdl.compute(line).value
    for name, X, l in (("mdl_full_d6", full, l_full), ("mdl_line_d6", line, l_line)):
        rows.append(
            ValidationRow("mdl_" + name, "mdl_autoencoder", "autoencoder_mdl",
                          l, None, None, None, None, 0.0)
        )
    mdl_order_ok = float(l_full > l_line)
    rows.append(
        ValidationRow("mdl_ranking_test", "mdl_relative_order", "autoencoder_mdl",
                      mdl_order_ok, None, 1.0, 1.0 - mdl_order_ok, 1.0 - mdl_order_ok, 0.0)
    )

    # ---- RBF kernel effective rank (v_kspec): the median heuristic is
    # scale-invariant, so the discriminating axis is *structure*, not scale.
    # Collinear (1-D) points need fewer RBF directions than a full-rank gaussian;
    # exactly-identical points hit the degenerate path and collapse to rank 1.
    identical = np.ones((200, 4))
    t = rng.standard_normal((200, 1))
    line = t * rng.standard_normal((1, 4))                  # 1-D subspace in R^4
    full = rng.standard_normal((200, 4))                     # full-rank gaussian
    ker = KernelEffectiveRankRBF(seed=0)
    r_id = ker.compute(identical).value
    r_line = ker.compute(line).value
    r_full = ker.compute(full).value
    for name, r in (("rbf_identical_n200", r_id), ("rbf_line_n200", r_line),
                    ("rbf_full_n200", r_full)):
        rows.append(
            ValidationRow(name, "kernel_effective_rank", "rbf_median",
                          r, None, None, None, None, 0.0)
        )
    rbf_order_ok = float(r_id <= 1.0 + 1e-6 and r_line < r_full)
    rows.append(
        ValidationRow("rbf_ranking_test", "kspec_relative_order", "rbf_median",
                      rbf_order_ok, None, 1.0, 1.0 - rbf_order_ok, 1.0 - rbf_order_ok, 0.0)
    )

    # ---- persistent homology: validate relative ranking of total persistence
    # Ground-truth ordering: torus > 2-sphere > convex blob ≈ uniform box
    # (torus has both connected components, loops, and a void; sphere has just
    # void; blob/box have only one component, no holes.)
    rng = np.random.default_rng(0)
    blob = rng.standard_normal((250, 3)) * 0.5
    box = rng.uniform(-1, 1, size=(250, 3))
    beds_topo = [
        ("torus_n250", tb.make_torus(n=250, seed=0).data),
        ("sphere_n250", tb.make_sphere(n=250, d_intrinsic=2, d_ambient=3, seed=0).data),
        ("gauss_blob_n250", blob),
        ("uniform_box_n250", box),
    ]
    ph = PersistentHomology(max_dim=1, max_edge_fraction=0.45, weights=(0.0, 1.0))
    ctop_values: dict[str, float] = {}
    for name, X in beds_topo:
        res = ph.compute(X)
        ctop_values[name] = res.value
        rows.append(
            ValidationRow(
                testbed=name,
                metric="topological_complexity",
                estimator="vietoris_rips",
                value=res.value,
                uncertainty=None,
                ground_truth=None,
                abs_error=None,
                rel_error=None,
                elapsed_ms=res.elapsed_s * 1000.0,
            )
        )
    # The validation: torus is the only one with H_1 signal at max_dim=1.
    # Other shapes are statistically similar (only noise loops). So torus
    # should be rank #1 by a clear margin.
    actual_order = sorted(ctop_values, key=ctop_values.get, reverse=True)
    torus_top = actual_order[0] == "torus_n250"
    margin = ctop_values["torus_n250"] / max(
        max(ctop_values[k] for k in ctop_values if k != "torus_n250"), 1e-9
    )
    rank_correct = float(torus_top and margin > 1.5)
    rows.append(
        ValidationRow(
            testbed="topo_ranking_test",
            metric="topology_relative_order",
            estimator="vietoris_rips",
            value=rank_correct,
            uncertainty=None,
            ground_truth=1.0,
            abs_error=1.0 - rank_correct,
            rel_error=1.0 - rank_correct,
            elapsed_ms=0.0,
        )
    )
    return rows


def run_quantum() -> list[ValidationRow]:
    rows: list[ValidationRow] = []

    # ---- entanglement entropy
    for n_qubits in (4, 6):
        beds = [
            tb.make_product_state(n_qubits),
            tb.make_ghz_state(n_qubits),
            tb.make_haar_random_state(n_qubits, seed=1),
        ]
        for bed in beds:
            gt = bed.ground_truth["entanglement_entropy"]
            res_exact = EntanglementEntropyExact().compute(bed.data, ground_truth=gt)
            rows.append(_row(bed.name, gt, res_exact))
            res_proxy = EntanglementEntropyRenyi2(
                n_unitaries=300, n_shots=500, seed=42
            ).compute(bed.data, ground_truth=gt)
            rows.append(_row(bed.name, gt, res_proxy))

    # ---- kernel effective rank
    for bed in (tb.make_identity_kernel(50), tb.make_rank_one_kernel(50, seed=0)):
        gt = bed.ground_truth["kernel_effective_rank"]
        res = KernelEffectiveRank().compute(bed.data, ground_truth=gt)
        rows.append(_row(bed.name, gt, res))

    # ---- expressibility
    for bed in (
        tb.make_haar_state_sampler(n_qubits=3, seed=0),
        tb.make_constant_state_sampler(n_qubits=3),
    ):
        gt = bed.ground_truth["expressibility_kl"]
        if not np.isfinite(gt):
            gt = None  # don't report abs error for infinite reference
        res = ExpressibilityKL(n_pairs=1500, n_bins=60, seed=0).compute(bed.data, ground_truth=gt)
        rows.append(_row(bed.name, gt, res))

    return rows


def format_table(rows: list[ValidationRow]) -> str:
    header = (
        f"{'testbed':38} {'metric':24} {'estimator':22} "
        f"{'value':>12} {'truth':>12} {'|err|':>10} {'rel':>8} {'ms':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        v = f"{r.value:.5g}"
        gt = f"{r.ground_truth:.5g}" if r.ground_truth is not None else "—"
        ae = f"{r.abs_error:.3g}" if r.abs_error is not None else "—"
        re = f"{r.rel_error:.3g}" if r.rel_error is not None else "—"
        lines.append(
            f"{r.testbed[:38]:38} {r.metric[:24]:24} {r.estimator[:22]:22} "
            f"{v:>12} {gt:>12} {ae:>10} {re:>8} {r.elapsed_ms:>8.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    print("=== classical ===")
    rows_c = run_classical()
    print(format_table(rows_c))
    print()
    print("=== quantum ===")
    rows_q = run_quantum()
    print(format_table(rows_q))

    # Summary diagnostics
    print("\n=== exact-vs-proxy summary ===")
    by_pair: dict[tuple[str, str], dict[str, ValidationRow]] = {}
    for r in rows_c + rows_q:
        key = (r.testbed, r.metric)
        by_pair.setdefault(key, {})[r.estimator] = r
    for (testbed, metric), variants in by_pair.items():
        if len(variants) >= 2:
            estimators = list(variants.keys())
            base = variants[estimators[0]]
            others = [variants[e] for e in estimators[1:]]
            print(f"{testbed:36} {metric:24} ", end="")
            for v in [base] + others:
                err = f"{v.abs_error:.3g}" if v.abs_error is not None else "—"
                print(f"  {v.estimator}: err={err}, t={v.elapsed_ms:.1f}ms", end="")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
