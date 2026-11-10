"""Figures for 003 stats review. Run after stats_review.py. Reads ../results.json, decoder_recon_by_epoch.json, per_run_slopes.csv."""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(H / "per_run_slopes.csv")))
fig, ax = plt.subplots(figsize=(8, 4.2))
order = ["baseline_ind(001)", "single_a026", "single_a248", "mix_adjacent", "mix_wide"]
for i, d in enumerate(order):
    for j, r in enumerate([r for r in rows if r["dist"] == d]):
        x = i + (j - 1) * 0.15; col = "tab:red" if r["collapsed_decoder"] == "True" else "tab:blue"
        ax.errorbar(x, float(r["slope"]), yerr=[[float(r["slope"]) - float(r["ci_lo"])], [float(r["ci_hi"]) - float(r["slope"])]], fmt="o" if r["resumed"] != "True" else "s", color=col, capsize=3)
ax.axhline(0.2 + 0.3101748, ls=":", c="gray"); ax.text(4.45, 0.52, "max baseline + 0.2", fontsize=7, ha="right")
ax.set_xticks(range(5)); ax.set_xticklabels(order, fontsize=8); ax.set_ylabel("slope ol@k12 (log-log)")
ax.set_title("per-run slope, 95% within-run bootstrap CI (red = decoder collapse, square = resumed)", fontsize=8); fig.tight_layout(); fig.savefig(H / "fig_slopes_by_run.png", dpi=130)
D = json.load(open(H / "decoder_recon_by_epoch.json"))["runs"]
fig, ax = plt.subplots(figsize=(7, 4))
for n, x in D.items():
    ep = sorted(x["by_epoch"], key=int); y = [x["by_epoch"][e]["val_img_mse_reconstructed"] for e in ep]
    bad = n in ("exp003_mix_adjacent_s0", "exp003_single_a026_s1")
    ax.plot([int(e) for e in ep], y, c="tab:red" if bad else "0.6", lw=2 if bad else 1, label=n if bad else None)
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("val_img_mse_reconstructed (operational log)"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(H / "fig_recon_by_epoch.png", dpi=130)
R = json.load(open(H / ".." / "results.json"))["runs"]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, d in zip(axs, ["single_a026", "mix_adjacent"]):
    for s in range(3):
        r = R["exp003_%s_s%d" % (d, s)]; g = r["groups_in_order"]
        ax.plot([r["by_size"][k]["area_real_mean"] for k in g], [r["by_size"][k]["area_ol_mean"] for k in g], "o-", label="s%d" % s)
    ax.plot([400, 5600], [400, 5600], "k:", lw=0.8); ax.set_xscale("symlog", linthresh=100); ax.set_yscale("symlog", linthresh=100); ax.set_title(d, fontsize=9); ax.set_xlabel("mean real area k12"); ax.set_ylabel("mean ol area k12"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(H / "fig_ol_vs_real_means.png", dpi=130)
