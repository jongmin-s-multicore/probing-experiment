"""Figures from results.json (no recomputation, no interpretation text). PNGs go to experiments/006_.../analysis/. Needs matplotlib."""
from __future__ import annotations

from pathlib import Path

import numpy as np

COL = {"multi_60_ind_160": "#1f77b4", "single_ind": "#d62728"}
LAB = {"multi_60_ind_160": "(b) multi_60_ind_160", "single_ind": "(a) single_ind"}


def _plt():
    import os
    os.environ["MPLBACKEND"] = "Agg"      # unconditional: a kernel-inherited backend (module://matplotlib_inline...) is invalid in the venv subprocess; figures are only saved to files
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _band(ax, x, agg, color, label, ls="-"):
    m = np.array(agg["mean"], float)
    ax.plot(x, m, ls, color=color, label=label)
    ps = np.array(agg["per_seed"], float)
    for row in ps:
        ax.plot(x, row, ".", color=color, alpha=0.35, ms=4)


def make_all(res, outdir, only_synthetic_label=None):
    plt = _plt()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    made, failed = [], []
    tag = "" if not only_synthetic_label else only_synthetic_label
    for name, fn in (("figA_slope_vs_step", fig_a_slope), ("figA_level_means", fig_a_levels), ("figB_baselines_r2", fig_b), ("figC_retrained_vs_transfer", fig_c),
                     ("figD_properties_r2", fig_d), ("figE_step12_seed_points", fig_e), ("figF_teacher_forcing_vs_open_loop", fig_f)):
        try:
            fig = fn(plt, res, tag)
            if fig is None:
                continue
            p = out / (name + ".png")
            fig.savefig(p, dpi=130, bbox_inches="tight")
            plt.close(fig)
            made.append(str(p))
        except Exception as e:                    # noqa: BLE001  (reported, not hidden)
            failed.append("%s: %s: %s" % (name, type(e).__name__, e))
    return made, failed


def _steps(res):
    return res["A"]["steps"] if res.get("A") else list(range(1, 13))


def fig_a_slope(plt, res, tag):
    A = res.get("A")
    if not A:
        return None
    src = "stored" if A["sources"].get("stored", {}).get("status") == "ok" else None
    if src is None:
        return None
    s = A["sources"][src]["seed_summary"]
    x = A["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, q in zip(axes, ("slope_episode|unseen", "intercept_episode|unseen")):
        for cond, d in s.items():
            _band(ax, x, d["rollout"][q], COL[cond], LAB[cond] + " rollout")
        e = next(iter(s.values()))["encoder"][q]
        ax.plot(x, e["mean"], "k--", label="encoder tokens (same probe)")
        ax.set_xlabel("rollout step k")
        ax.set_ylabel(q.replace("|", " "))
        ax.legend(fontsize=7)
    fig.suptitle("A: episode-level fit pred ~ label, unseen levels; dots = training seeds %s" % tag, fontsize=9)
    return fig


def fig_a_levels(plt, res, tag):
    A = res.get("A")
    if not A or A["sources"].get("stored", {}).get("status") != "ok":
        return None
    s = A["sources"]["stored"]["seed_summary"]
    lv = np.array(A["level_true_log_area"])
    steps = [1, 6, 12]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, k in zip(axes, steps):
        i = A["steps"].index(k)
        ax.plot(lv, lv, "k:", lw=1, label="y = label")
        for cond, d in s.items():
            m = np.array(d["rollout"]["per_level_mean_pred"]["mean"])[i]
            ax.plot(lv, m, "o-", color=COL[cond], label=LAB[cond])
        ax.set_title("step %d" % k, fontsize=9)
        ax.set_xlabel("label (log nominal area ratio)")
    axes[0].set_ylabel("mean probe prediction per level")
    axes[0].legend(fontsize=7)
    fig.suptitle("A: level means of the transfer-probe prediction %s" % tag, fontsize=9)
    return fig


def fig_b(plt, res, tag):
    Bq = res.get("B")
    if not Bq:
        return None
    x = Bq["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, role in zip(axes, ("unseen", "trained")):
        ax.plot(x, Bq["ceiling_encoder"]["r2"][role], "k--", label="encoder tokens (ceiling)")
        ax.plot(x, Bq["persistence"]["r2"][role], "k-", label="persistence (frame t copied)")
        for cond, d in Bq["seed_summary"].items():
            _band(ax, x, d["trained_rollout"]["r2|" + role], COL[cond], LAB[cond] + " rollout")
            for v, ls in zip(d["untrained"], (":", "-.")):
                ax.plot(x, d["untrained"][v]["r2|" + role]["mean"], ls, color=COL[cond], alpha=0.7, label=LAB[cond] + " untrained " + v)
        ax.set_title("role: " + role, fontsize=9)
        ax.set_xlabel("rollout step k")
    axes[0].set_ylabel("R2 (transfer probe)")
    axes[0].legend(fontsize=6)
    fig.suptitle("B: baselines %s" % tag, fontsize=9)
    return fig


def fig_c(plt, res, tag):
    C = res.get("C")
    if not C:
        return None
    x = C["steps"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, role in zip(axes, ("unseen", "trained")):
        ax.plot(x, C["encoder_retrained"]["r2"][role], "k--", label="encoder, retrained at frame k")
        for cond, d in C["seed_summary"].items():
            _band(ax, x, d["retrained|r2|" + role], COL[cond], LAB[cond] + " retrained")
            _band(ax, x, d["transfer|r2|" + role], COL[cond], LAB[cond] + " transfer", ls=":")
        ax.set_title("role: " + role, fontsize=9)
        ax.set_xlabel("rollout step k")
    axes[0].set_ylabel("R2")
    axes[0].legend(fontsize=6)
    fig.suptitle("C: retrained vs transfer probe %s" % tag, fontsize=9)
    return fig


def fig_d(plt, res, tag):
    D = res.get("D")
    if not D:
        return None
    props = D["properties"]
    n = len(props)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), squeeze=False)
    x = D["steps"]
    for i, p in enumerate(props):
        ax = axes[i // cols][i % cols]
        sets = D["per_property"][p]["sets"]
        ax.plot(x, [sets["encoder|%d" % k]["r2|all"] for k in x], "k--", label="encoder")
        ax.plot(x, [sets["persist|%d" % k]["r2|all"] for k in x], "k-", lw=0.8, label="persistence")
        for cond, d in D["seed_summary"].items():
            _band(ax, x, d["%s|r2|all" % p], COL[cond], LAB[cond])
        ax.set_title(p, fontsize=8)
        ax.set_xlabel("step", fontsize=7)
    axes[0][0].legend(fontsize=5)
    fig.suptitle("D: R2 (all levels) of the encoder-trained probe %s" % tag, fontsize=9)
    fig.tight_layout()
    return fig


def fig_e(plt, res, tag):
    E = res.get("E")
    if not E:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (q, ttl) in zip(axes, (("r2|rollout|unseen", "R2 rollout step 12"), ("drop|unseen", "R2 drop step 12"), ("mae|rollout|unseen", "MAE rollout step 12"))):
        for j, (cond, d) in enumerate(E["seed_summary"].items()):
            a = d[q]
            ps = np.array(a["per_seed"])[:, -1]
            ax.plot([j] * len(ps), ps, "o", color=COL[cond], alpha=0.6)
            ax.errorbar([j + 0.15], [a["mean"][-1]], yerr=[[a["mean"][-1] - a["ci_lo"][-1]], [a["ci_hi"][-1] - a["mean"][-1]]], fmt="s", color=COL[cond], capsize=4)
        ax.set_xticks(range(len(E["seed_summary"])))
        ax.set_xticklabels([LAB[c] for c in E["seed_summary"]], fontsize=7)
        ax.set_title(ttl + " (unseen); bar = mean, t CI df=2", fontsize=8)
    fig.suptitle("E: training-seed values %s" % tag, fontsize=9)
    return fig


def fig_f(plt, res, tag):
    F = res.get("F")
    if not F:
        return None
    x = F["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for j, probe in enumerate(("transfer", "retrained")):
        for ax, qn, ttl in ((axes[0][j], "r2|unseen", "R2 (unseen)"), (axes[1][j], "slope_episode|unseen", "episode slope (unseen)")):
            ax.plot(x, F["encoder_reference"][probe]["r2"]["unseen"] if qn.startswith("r2") else F["encoder_reference"][probe]["q"][qn], "k--", lw=1, label="encoder (ceiling)")
            for cond, d in F["seed_summary"].items():
                _band(ax, x, d["%s|open_loop|%s" % (probe, qn)], COL[cond], LAB[cond] + " open loop", ls="-")
                _band(ax, x, d["%s|teacher_forced|%s" % (probe, qn)], COL[cond], LAB[cond] + " teacher forced", ls=":")
            ax.set_title("%s probe: %s" % (probe, ttl), fontsize=9)
            ax.set_xlabel("step k")
    axes[0][0].legend(fontsize=6)
    fig.suptitle("F: teacher-forced vs open-loop latents %s" % tag, fontsize=9)
    return fig
