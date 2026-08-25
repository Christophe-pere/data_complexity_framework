#!/usr/bin/env python3
"""Generate the complexity-vector tables for Part II from committed Phase B records.

Emits two LaTeX fragments into paper/tables/:
  tab_v_ranknorm.tex  — rank-normalised v(D) (Eq. 2.2), the quantity the selector consumes
  tab_v_raw.tex       — raw estimator outputs, for reproducibility and scale comparison

Regenerate with:  python analysis/make_paper_tables.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "paper", "tables")
SRC  = os.path.join(HERE, "..", "experiments", "results", "phase_b_best_classical.jsonl")
os.makedirs(OUT, exist_ok=True)

# component order + short column headers (legend printed in the caption)
COMPS = [
    ("differential_entropy_knn_kl", r"$H$"),
    ("intrinsic_dim_pca",           r"$d_{\mathrm{pca}}$"),
    ("intrinsic_dim_twonn",         r"$d_{\mathrm{2nn}}$"),
    ("compression_ratio_gzip",      r"$K$"),
    ("topological_complexity_rips", r"$T$"),
    ("topological_complexity_corrected", r"$T^{\star}$"),
    ("second_order_mi",             r"$I_2$"),
    ("higher_order_mi",             r"$I_h$"),
    ("kernel_effective_rank",       r"$R$"),
    ("mdl_autoencoder",             r"$M$"),
]

def load():
    vals = {}
    for line in open(SRC):
        d = json.loads(line)
        if d["dataset_id"] in vals:
            continue
        vals[d["dataset_id"]] = {k: v["value"] for k, v in d["v"].items()}
    # Corrected topological component (Appendix: estimator audit). The frozen
    # `topological_complexity_rips` is retained so the pre-specified tests stay
    # reproducible; T* is the scale-normalised, H0-excluded replacement actually
    # validated against known Betti numbers.
    corr = json.load(open(CORR))
    for ds in vals:
        vals[ds]["topological_complexity_corrected"] = corr.get(ds)
    return vals

def rank_normalise(vals):
    """v_i(D) = |{D' : f_i(D') <= f_i(D)}| / |B|   (Eq. 2.2)"""
    out = {ds: {} for ds in vals}
    N = len(vals)
    for key, _ in COMPS:
        col = {ds: vals[ds][key] for ds in vals}
        for ds, x in col.items():
            out[ds][key] = sum(1 for y in col.values() if y <= x) / N
    return out

def fmt_raw(x):
    if x is None: return "--"
    ax = abs(x)
    if ax >= 1e4 or (0 < ax < 1e-2): return f"{x:.1e}".replace("e+0", "e").replace("e-0", "e-")
    if ax >= 100: return f"{x:.0f}"
    return f"{x:.2f}"

def emit(vals, path, ranknorm, label, caption):
    order = sorted(vals)
    cols = "l" + "r" * len(COMPS)
    L = [r"\begin{table*}[t]", r"\centering", f"\\caption{{{caption}}}",
         f"\\label{{{label}}}", r"\small", r"\begin{tabular}{" + cols + "}", r"\toprule",
         "Dataset & " + " & ".join(h for _, h in COMPS) + r" \\", r"\midrule"]
    klass = None
    for ds in order:
        k = ds.split("-")[0]
        if klass is not None and k != klass:
            L.append(r"\addlinespace")
        klass = k
        row = [ds.replace("_", r"\_")]
        for key, _ in COMPS:
            x = vals[ds][key]
            row.append(f"{x:.2f}" if ranknorm else fmt_raw(x))
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(OUT, path), "w").write("\n".join(L) + "\n")

LEGEND = (r"$H$ differential entropy (kNN--KL); "
          r"$d_{\mathrm{pca}}$, $d_{\mathrm{2nn}}$ intrinsic dimension (PCA-effective, TwoNN); "
          r"$K$ compression ratio (gzip); $T$ topological complexity as pre-specified (Vietoris--Rips, defective --- see Appendix~\ref{app:topofix}); $T^{\star}$ its corrected replacement (untruncated Rips, $H_0$ excluded, diameter-normalised $\ell_2$); "
          r"$I_2$, $I_h$ second- and higher-order mutual information (kNN, MINE); "
          r"$R$ kernel effective rank (RBF, median heuristic); "
          r"$M$ MDL (autoencoder).")

CORR = os.environ.get("QMLC_TOPO_CORRECTED", "/tmp/topo_ov_L2.json")

if __name__ == "__main__":
    raw = load()
    rn  = rank_normalise(raw)
    emit(rn, "tab_v_ranknorm.tex", True, "tab:v_ranknorm",
         r"The intrinsic complexity vector $\vv(D)$ after rank normalisation "
         r"(Eq.~\eqref{eq:ranknorm}), for the $27$ classical-track datasets --- the "
         r"representation the selector consumes. Each column is the empirical CDF of that "
         r"component over the frozen suite, so entries are comparable across columns by "
         r"construction. " + LEGEND)
    emit(raw, "tab_v_raw.tex", False, "tab:v_raw",
         r"Raw estimator outputs for the nine implemented components of $\vv(D)$ (plus the corrected topology column $T^{\star}$), before rank "
         r"normalisation. Reported for reproducibility, and to make concrete the dimensional "
         r"incoherence motivating Sec.~\ref{sec:framework:predictor}: the columns differ by up "
         r"to seven orders of magnitude in range and do not share units. " + LEGEND)
    print(f"tab_v_ranknorm.tex  ({len(rn)} datasets x {len(COMPS)} components)")
    print(f"tab_v_raw.tex       ({len(raw)} datasets x {len(COMPS)} components)")
