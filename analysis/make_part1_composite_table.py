#!/usr/bin/env python3
"""Appendix table: Part I's four pillars and composite C_data, per dataset."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(OUT, exist_ok=True)
V = json.load(open(os.path.join(HERE, "part1_composite_values.json")))

def f(x):
    ax = abs(x)
    if ax >= 1e4 or (0 < ax < 1e-2): return f"{x:.1e}".replace("e+0","e").replace("e-0","e-")
    if ax >= 100: return f"{x:.0f}"
    return f"{x:.2f}"

L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Part~I's four complexity pillars and the composite $C_{\text{data}}$ "
     r"(Eq.~\eqref{eq:p17}, equal weights) for each classical-track dataset. "
     r"$C_{\text{top}}$ is the topological component as deployed for the pre-specified tests; "
     r"$C_{\text{top}}^{\star}$ and $C_{\text{data}}^{\star}$ use the corrected estimator of "
     r"Appendix~\ref{app:topofix}. Provided so the Part~I quantities can be reproduced and compared; "
     r"Part~II does not use the composite (Sec.~\ref{sec:framework:predictor}, Table~\ref{tab:p17}). "
     r"\emph{Exploratory.}}",
     r"\label{tab:p17_values}", r"\small", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
     r"Dataset & $S$ & $I_{\text{corr}}$ & $K$ & $C_{\text{top}}$ & $C_{\text{top}}^{\star}$ "
     r"& $C_{\text{data}}$ & $C_{\text{data}}^{\star}$ \\", r"\midrule"]
klass = None
for ds in sorted(V):
    k = ds.split("-")[0]
    if klass is not None and k != klass: L.append(r"\addlinespace")
    klass = k
    d = V[ds]
    L.append(" & ".join([ds.replace("_", r"\_")] + [f(d[c]) for c in
        ("S","I_corr","K","C_top","C_top_corrected","C_data_equalweights","C_data_equalweights_corrected")]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(OUT, "tab_p17_values.tex"), "w").write("\n".join(L)+"\n")
print(f"tab_p17_values.tex ({len(V)} datasets)")
