"""Figures of experiment 004 (matplotlib imported lazily; ImportError is handled by the caller)."""
from __future__ import annotations

import numpy as np

from . import depthlib as DL


def _axis(rows, prim):
    """Combined depth axis: encoder 0..K then predictor 1..L (spec 3.5)."""
    enc = sorted({r["depth"] for r in rows if r["module"] == "encoder"})
    pre = sorted({r["depth"] for r in rows if r["module"] == "predictor" and r["depth"] >= 1})
    return [("encoder", d) for d in enc] + [("predictor", d) for d in pre]


def make(rows, cfg, cfg2, outdir):
    import os
    os.environ["MPLBACKEND"] = "Agg"      # unconditional: a kernel-inherited backend (module://matplotlib_inline...) is invalid outside the kernel; figures are only saved to files
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prim = cfg["depth"]["primary"]
    ax_ = _axis(rows, prim)
    if not ax_:
        return "skipped(no rows)"
    levels = list(cfg2["data"]["levels"])
    # heatmap: pooled-variance R2 per level (level rows)
    M = np.full((len(levels), len(ax_)), np.nan)
    for j, (mod, d) in enumerate(ax_):
        for i, l in enumerate(levels):
            v = [r["r2"] for r in rows if (r["module"], r["depth"], r["probe"], r["fit"], r["control"], r["timepoint"], r["level"]) ==
                 (mod, d, prim["probe"], prim["fit"], prim["control"], prim["timepoint"], l)]
            if v:
                M[i, j] = np.nanmean(v)
    fig, ax = plt.subplots(figsize=(1 + 0.45 * len(ax_), 4))
    im = ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels)
    ax.set_xticks(range(len(ax_)))
    ax.set_xticklabels(["%s%d" % ("E" if m == "encoder" else "P", d) for m, d in ax_], rotation=90, fontsize=7)
    fig.colorbar(im, label="R2 (level rows: 1-MSE/Var_all)")
    ax.set_title("%s / %s / %s" % (prim["probe"], prim["timepoint"], prim["fit"]))
    fig.tight_layout()
    fig.savefig(outdir / "fig_heatmap.png", dpi=120)
    plt.close(fig)
    # curves
    fig, ax = plt.subplots(figsize=(1 + 0.45 * len(ax_), 4))
    xs = np.arange(len(ax_))
    for s, col in (("train", "C0"), ("interp", "C1"), ("extrap", "C3")):
        for ctl, ls in ((prim["control"], "-"), ("random_init", ":"), ("label_shuffle", "--")):
            ys, sd = [], []
            for mod, d in ax_:
                v = np.array([r["r2"] for r in rows if (r["module"], r["depth"], r["probe"], r["fit"], r["control"], r["timepoint"], r["level"], r["split"]) ==
                              (mod, d, prim["probe"], prim["fit"], ctl, prim["timepoint"], "ALL", s)], float)
                v = v[np.isfinite(v)]
                ys.append(v.mean() if len(v) else np.nan)
                sd.append(v.std() if len(v) else np.nan)
            ys, sd = np.array(ys), np.array(sd)
            if np.isfinite(ys).any():
                ax.plot(xs, ys, ls, color=col, label="%s %s" % (s, ctl))
                if ctl == prim["control"]:
                    ax.fill_between(xs, ys - sd, ys + sd, color=col, alpha=0.15)
    for b, ls in (("pixel", "-."), ("mask_area_bbox", "-.")):
        v = [r["r2"] for r in rows if r["module"] == "baseline" and r["control"] == b and r["probe"] == (prim["probe"] if b == "pixel" else "linear") and
             r["timepoint"] == prim["timepoint"] and r["level"] == "ALL" and r["split"] == "all"]
        if v:
            ax.axhline(float(np.nanmean(v)), color="gray", ls=ls, lw=0.8, label="baseline " + b)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%s%d" % ("E" if m == "encoder" else "P", d) for m, d in ax_], rotation=90, fontsize=7)
    ax.set_ylabel("R2")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "fig_curves.png", dpi=120)
    plt.close(fig)
    return ["fig_heatmap.png", "fig_curves.png"]
