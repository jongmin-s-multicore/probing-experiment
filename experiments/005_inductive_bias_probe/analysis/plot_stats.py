"""Figure: D-IB raw by model (all pairs) and H-f gap per model. Reads analysis/score_table.csv, recomputed_ib.csv."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
A = "/home/jaekyungkim/mylab/experiments/005_inductive_bias_probe/analysis/"
d = pd.read_csv(A + "recomputed_ib.csv"); d = d[d.group == "all"]
mods = ["ours", "random", "area", "oracle", "oracle_linear", "noninf"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, (m, t) in enumerate([("D-IB", "D-IB raw (all pairs)"), ("D-IB_area=", "D-IB_area= raw"), ("D-IB_area≠", "D-IB_area!= raw")]):
    for j, k in enumerate(mods):
        v = d[(d.metric == m) & (d.model == k)].raw.values
        ax[i].scatter([j] * len(v), v, s=14); ax[i].errorbar(j, v.mean(), v.std(ddof=1), color="k", capsize=3)
    ax[i].set_xticks(range(len(mods))); ax[i].set_xticklabels(mods, rotation=30); ax[i].set_title(t)
plt.tight_layout(); plt.savefig(A + "fig_dib_raw_by_model.png", dpi=110)
