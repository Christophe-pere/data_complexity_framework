#!/usr/bin/env python3
"""Diagnostic figure for the topology-component correction (2026-08-24).

Panel (a) uses the values H2 actually consumed (experiments/results/h1_gradients.jsonl,
feature_extraction path, max_dim=2), NOT the registry screening values, which were
computed with max_dim=1 on ~100 points and are a different configuration.

Four panels:
  (a) production Ctop vs point-cloud diameter — the scale artifact (R^2 = 0.85)
  (b) corrected Ctop vs diameter — artifact removed
  (c) known-shape validation: topological shapes vs trivial controls
  (d) per-dataset corrected ranking, coloured by dataset class

Usage: python -m analysis.make_topology_diagnostic_figure
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

R = json.load(open("/tmp/topo_recomputed.json"))
COL = {"A": "#2266cc", "C": "#cc7722", "D": "#aa2255"}
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))

# (a) production: the scale artifact
d = np.array([r["diam"] for r in R]); o = np.array([r["old"] for r in R], float)
m = np.isfinite(o) & (o > 0)
for k in COL:
    s = np.array([r["klass"] == k for r in R]) & m
    ax[0,0].scatter(d[s], o[s], c=COL[k], label=f"class {k}", s=42, alpha=.85, edgecolors="none")
x, y = np.log(d[m]), np.log(o[m]); b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax[0,0].plot(np.exp(xs), np.exp(np.polyval(b, xs)), "k--", lw=1.3)
ax[0,0].set(xscale="log", yscale="log", xlabel="point-cloud diameter",
            ylabel=r"$C_{\rm top}$ (as deployed for H2)",
            title=f"(a) deployed estimator tracks scale\n$R^2$(log diam) = {np.corrcoef(x,y)[0,1]**2:.2f}")
ax[0,0].legend(frameon=False, fontsize=9)

# (b) corrected: artifact removed
c = np.array([r["L2"] for r in R])
for k in COL:
    s = np.array([r["klass"] == k for r in R])
    ax[0,1].scatter(d[s], c[s], c=COL[k], s=42, alpha=.85, edgecolors="none")
mm = c > 0
r2 = np.corrcoef(np.log(d[mm]), c[mm])[0,1] ** 2
ax[0,1].set_yscale("symlog", linthresh=1e-3)
ax[0,1].set(xscale="log", xlabel="point-cloud diameter",
            ylabel=r"$C_{\rm top}$ (corrected, $\ell_2$)",
            title=f"(b) corrected estimator is scale-free\n$R^2$(log diam) = {r2:.2f}")

# (c) known-shape validation
shapes = ["circle\n"+r"$\beta_1{=}1$", "torus\n"+r"$\beta_1{=}2,\beta_2{=}1$",
          "sphere\n"+r"$\beta_2{=}1$", "Gaussian\nblob", "uniform\nbox"]
prod  = [0.916, 2.595, 2.839, 1.799, 2.602]     # 'total' = the production form
corr  = [0.628, 0.238, 0.418, 0.048, 0.107]     # 'l2'    = the corrected form
xi = np.arange(5); w = 0.38
cols = ["#2266cc"]*3 + ["#999999"]*2
ax[1,0].bar(xi-w/2, prod, w, color=cols, alpha=.45, edgecolor="k", lw=.6, label="production (total)")
ax[1,0].bar(xi+w/2, corr, w, color=cols, edgecolor="k", lw=.6, label=r"corrected ($\ell_2$)")
ax[1,0].axvline(2.5, color="k", lw=.8, ls=":")
ax[1,0].set_xticks(xi); ax[1,0].set_xticklabels(shapes, fontsize=8)
ax[1,0].set(ylabel=r"$C_{\rm top}$", title="(c) known shapes: blue = has topology, grey = trivial\nproduction ranks the blob above the circle")
ax[1,0].legend(frameon=False, fontsize=8)

# (d) corrected per-dataset ranking
order = sorted(range(len(R)), key=lambda i: R[i]["L2"])
names = [R[i]["name"] for i in order]; vals = [R[i]["L2"] for i in order]
cs = [COL[R[i]["klass"]] for i in order]
ax[1,1].barh(np.arange(len(R)), vals, color=cs, height=.72)
ax[1,1].set_yticks(np.arange(len(R))); ax[1,1].set_yticklabels(names, fontsize=6.5)
ax[1,1].set_xscale("symlog", linthresh=1e-3)
ax[1,1].set(xlabel=r"$C_{\rm top}$ (corrected, $\ell_2$)",
            title="(d) corrected ranking\nparity / spin-glass (hypercube geometry) on top")

fig.tight_layout()
out = ROOT / "paper" / "figs" / "topology_diagnostic.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=155, bbox_inches="tight")
print("wrote", out)
