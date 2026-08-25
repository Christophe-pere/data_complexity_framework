#!/usr/bin/env python3
"""Generate Part II figures from the confirmatory analysis artifacts.

Reads only committed JSON in analysis/; writes PDF into paper/figs/.
No re-running of experiments. Regenerate with:  python analysis/make_paper_figures.py
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4, "axes.axisbelow": True,
})
def j(name): return json.load(open(os.path.join(HERE, name)))

# ---------------------------------------------------------------- Fig 1: H5 null
def fig_h5():
    d = j("predictor_confirmatory_best_classical.json")
    folds = d["folds"] if isinstance(d["folds"], list) else list(d["folds"].values())
    obs, pred = [], []
    for f in folds:
        o, p = f.get("e_obs_test"), f.get("e_hat_gp")
        if o is None or p is None: continue
        o = np.atleast_1d(np.asarray(o, float)); p = np.atleast_1d(np.asarray(p, float))
        m = min(len(o), len(p)); obs.append(o[:m]); pred.append(p[:m])
    obs = np.concatenate(obs); pred = np.concatenate(pred)
    ok = np.isfinite(obs) & np.isfinite(pred); obs, pred = obs[ok], pred[ok]
    r2 = d["summary"]["edge_pooled_r2_gp"]

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.scatter(pred, obs, s=5, alpha=0.25, edgecolors="none", color="#1f77b4")
    lo = float(min(pred.min(), obs.min())); hi = float(max(pred.max(), obs.max()))
    pad = 0.05 * (hi - lo)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k--", lw=0.8, label="perfect prediction")
    ax.axhline(obs.mean(), color="#d62728", lw=0.9, ls=":", label="mean of observed edge")
    ax.set_xlabel(r"predicted edge $\hat{e}$ (held-out)")
    ax.set_ylabel(r"observed edge $e_{\mathrm{obs}}$")
    ax.set_title(f"H5: pooled LODO $R^2$ = {r2:.2f}  (threshold $\\geq 0.4$)")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(os.path.join(OUT, "fig_h5_null.pdf")); plt.close(fig)
    return len(obs)

# ------------------------------------------------------- Fig 2: width scaling (H1)
def fig_width():
    d = j("h1_functional_form.json")
    pnm = d["per_n_means"]
    series = {}
    for k, v in pnm.items():
        npart, lpart = k.split(",")
        n = int(npart.split("=")[1]); L = int(lpart.split("=")[1])
        series.setdefault(L, []).append((n, v["mean_log_var"]))
    sl = d["slope_on_n"]; est = sl["estimate"]; lo, hi = sl["ci95"]; haar = sl["mcclean_rate"]

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    marks = {1: "o", 3: "s", 6: "^"}
    for L in sorted(series):
        pts = sorted(series[L]); ns = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        ax.plot(ns, ys, marker=marks.get(L, "o"), ms=3.5, lw=1.0, label=f"$L={L}$")
    allpts = sorted({p[0] for v in series.values() for p in v})
    ns = np.array(allpts, float)
    ref = series[sorted(series)[0]]; ref = sorted(ref)
    y0 = ref[0][1]; n0 = ref[0][0]
    ax.plot(ns, y0 + haar * (ns - n0), "k--", lw=0.9,
            label=r"Haar rate $-\ln 2$")
    ax.plot(ns, y0 + est * (ns - n0), color="#d62728", lw=0.9, ls="-.",
            label=f"fitted slope {est:.3f}")
    ax.fill_between(ns, y0 + lo*(ns-n0), y0 + hi*(ns-n0), color="#d62728", alpha=0.12, lw=0)
    ax.set_xlabel("qubits $n$"); ax.set_ylabel(r"mean $\log \mathrm{Var}[g]$")
    ax.set_title("Width scaling: depth-independent, slower than Haar")
    ax.legend(loc="lower left", frameon=False, ncol=1)
    fig.savefig(os.path.join(OUT, "fig_width_scaling.pdf")); plt.close(fig)
    return est, (lo, hi), haar

# ------------------------------------------------- Fig 3: baseline sweep (Lesson 1)
def _winrates(path):
    """Quantum win rate from a Phase B sweep: parity family and full suite."""
    par_n=par_w=all_n=all_w=0
    for line in open(os.path.join(HERE,"..",path)):
        d=json.loads(line); eb=d.get("e_budget")
        if eb is None: continue
        all_n+=1; all_w += eb>0
        if d["dataset_id"].startswith("synth-parity"):
            par_n+=1; par_w += eb>0
    return 100*par_w/par_n, 100*all_w/all_n

def fig_baseline():
    # Phase B sweeps: single logistic-regression baseline vs best-of-six.
    # The August work re-fit the predictor on phase_b_best_classical.jsonl;
    # the raw sweeps themselves are unchanged, so these are current.
    par_lr, all_lr = _winrates("experiments/results/phase_b.jsonl")
    par_bc, all_bc = _winrates("experiments/results/phase_b_best_classical.jsonl")
    # selector metrics are reported in Sec. baseline (predictor over the same sweeps)
    r2  = [ 0.13, -0.94]
    rho = [ 0.43,  0.07]
    labels = ["single\nlogistic regression", "best of six\nclassical models"]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(2); c = ["#7f7f7f", "#1f77b4"]

    a = axes[0]
    b1 = a.bar(x - 0.19, r2,  0.36, color=c, label=r"selector LODO $R^2$")
    b2 = a.bar(x + 0.19, rho, 0.36, color=c, alpha=0.55, hatch="//",
               label=r"per-dataset Spearman $\rho$")
    a.axhline(0, color="k", lw=0.7)
    a.set_xticks(x); a.set_xticklabels(labels)
    a.set_ylabel("predictability of the edge"); a.set_title("The selector's signal")
    for bars in (b1, b2):
        for r in bars:
            h=r.get_height()
            a.annotate(f"{h:+.2f}", (r.get_x()+r.get_width()/2, h), ha="center",
                       va="bottom" if h>=0 else "top", fontsize=6.5,
                       xytext=(0, 2 if h>=0 else -2), textcoords="offset points")
    a.legend(frameon=False, loc="lower left")

    a = axes[1]
    w = 0.36
    bp = a.bar(x - w/2, [par_lr, par_bc], w, color=c, label="parity family")
    ba = a.bar(x + w/2, [all_lr, all_bc], w, color=c, alpha=0.5, hatch="//",
               label="full suite")
    a.set_xticks(x); a.set_xticklabels(labels)
    a.set_ylabel("quantum win rate (%)"); a.set_title("The advantage itself")
    for bars in (bp, ba):
        for r in bars:
            h=r.get_height()
            a.annotate(f"{h:.0f}%", (r.get_x()+r.get_width()/2, h), ha="center",
                       va="bottom", fontsize=6.5, xytext=(0,2), textcoords="offset points")
    a.legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(OUT, "fig_baseline_sweep.pdf")); plt.close(fig)
    return (par_lr, par_bc, all_lr, all_bc)


# ---------------------------------------------------- Fig 4: class B edge vs w
def fig_classb():
    """Mode-N edge against parity weight, for the three readout models."""
    import collections, statistics as st
    path = os.path.join(HERE, "..", "experiments", "results", "phase_b_classB.jsonl")
    g = collections.defaultdict(list)
    for line in open(path):
        d = json.loads(line)
        w = int(d["dataset_id"].rsplit("w", 1)[1])
        g[(d["quantum_model"], w)].append(d["e_budget"])

    style = {
        "parity-readout":        ("oracle parity readout",      "o", "-",  "#1f77b4"),
        "measurement-selection": ("trainable measurement head", "s", "--", "#d62728"),
        "variational-z0":        ("generic HEA, $Z_0$ readout", "^", ":",  "#7f7f7f"),
    }
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ws = sorted({w for (_, w) in g})
    for model, (lab, mk, ls, col) in style.items():
        xs = [w for w in ws if (model, w) in g]
        ys = [st.mean(g[(model, w)]) for w in xs]
        es = [st.pstdev(g[(model, w)]) for w in xs]
        ax.errorbar(xs, ys, yerr=es, marker=mk, ls=ls, color=col, ms=4,
                    lw=1.1, capsize=2, elinewidth=0.7, label=lab)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvspan(0.5, 2.5, color="#cccccc", alpha=0.35, lw=0)
    ax.annotate("shadow locality\n$k=2$", xy=(1.5, ax.get_ylim()[1]),
                xytext=(1.5, 0.30), ha="center", fontsize=6.5, color="#444444")
    ax.set_xticks(ws); ax.set_xlabel("parity weight $w$")
    ax.set_ylabel(r"Mode-N edge $\hat{e}_N$")
    ax.set_title("The edge exists, and gradients cannot reach it")
    ax.legend(loc="center left", frameon=False)
    fig.savefig(os.path.join(OUT, "fig_classb_edge.pdf")); plt.close(fig)
    return {m: [round(st.mean(g[(m, w)]), 3) for w in ws if (m, w) in g] for m in style}

if __name__ == "__main__":
    npts = fig_h5();               print(f"fig_h5_null.pdf         ({npts} held-out points)")
    est, ci, haar = fig_width();   print(f"fig_width_scaling.pdf   (slope {est:.3f}, CI {ci[0]:.3f}..{ci[1]:.3f}, Haar {haar:.3f})")
    pl,pb,al,ab = fig_baseline(); print(f"fig_baseline_sweep.pdf   (parity {pl:.0f}%->{pb:.0f}%, suite {al:.0f}%->{ab:.0f}%)")
    cb = fig_classb();             print(f"fig_classb_edge.pdf      {cb}")
