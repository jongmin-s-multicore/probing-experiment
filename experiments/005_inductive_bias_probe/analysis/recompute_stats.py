"""Independent recomputation for 005 (stats review). Run: conda lab07 python -I analysis/recompute_stats.py
Reads results/tasks.npz, results/predictions/*.npz, results/ib_scores.csv ... ; writes analysis/*.csv, analysis/summary.json"""
import numpy as np, json, csv, itertools, os, sys
R = "/home/jaekyungkim/mylab/experiments/005_inductive_bias_probe/"
A = R + "analysis/"
T = np.load(R + "results/tasks.npz", allow_pickle=True)
lvl, q, role = T["lvl"], T["q"], T["role"]
is_ind, is_ood, is_ex = T["is_ind"], T["is_ood"], T["is_extrap"]
n = len(lvl)
SEEDS = range(5)
def P(l, m, s): return np.load(R + f"results/predictions/{l}__{m}__seed{s}.npz")
# pair masks (upper triangle, independent implementation)
iu = np.triu_indices(n, 1)
I, J = iu
same_l = lvl[I] == lvl[J]; same_q = q[I] == q[J]
grp = {"all": np.ones(len(I), bool), "ind": is_ind[I] & is_ind[J], "ood_any": is_ood[I] | is_ood[J], "ood_both": is_ood[I] & is_ood[J]}
spec = {"R-IB": (same_l, True), "D-IB": (~same_l, False), "D-IB_area=": (~same_l & same_q, False), "D-IB_area≠": (~same_l & ~same_q, False), "R-IB_fold": (same_l & ~same_q, True)}
def raw(Y, metric, g):
    m = spec[metric][0] & grp[g]
    eq = (Y[:, I[m]] == Y[:, J[m]]).mean()  # mean over tasks and pairs
    return eq if spec[metric][1] else 1 - eq
def noninf(T_, seed): return (np.random.RandomState(6000 + seed).rand(T_, n) < 0.5).astype(np.float32)
rows = []
for s in SEEDS:
    Ys = {"ours": P("pred_out6", "ours", s)["A"], "random": P("pred_out6", "random", s)["A"], "area": P("base", "area", s)["A"], "oracle": P("base", "oracle_mlp", s)["A"], "oracle_linear": P("base", "oracle_linear", s)["A"]}
    Ys = {k: (v > 0.5).astype(np.int8) for k, v in Ys.items()}
    Tn = Ys["ours"].shape[0]
    Ys["noninf"] = (noninf(Tn, s) > .5).astype(np.int8); Ys["const"] = np.zeros((Tn, n), np.int8)
    for g in grp:
        for m in spec:
            rr = {k: raw(v, m, g) for k, v in Ys.items()}
            for mod in rr:
                rows.append(dict(model=mod, metric=m, group=g, seed=s, raw=rr[mod], r_noninf=rr["noninf"], r_const=rr["const"], r_orc=rr["oracle"], r_orcl=rr["oracle_linear"]))
import pandas as pd
df = pd.DataFrame(rows)
def sc(r, ref): return (r.raw - r[ref]) / (r.r_orc - r[ref])
typ = lambda m: "R" if m.startswith("R") else "D"
df["score_iid"] = (df.raw - df.r_noninf) / (df.r_orc - df.r_noninf)
df["score_const"] = (df.raw - df.r_const) / (df.r_orc - df.r_const)
df["score_orclin_const"] = (df.raw - df.r_const) / (df.r_orcl - df.r_const)
df["score_primary"] = np.where(df.metric.map(typ) == "R", df.score_iid, df.score_const)
df["score_alt"] = np.where(df.metric.map(typ) == "R", df.score_const, df.score_iid)
df.to_csv(A + "recomputed_ib.csv", index=False)
# compare with ib_scores.csv
ref = pd.read_csv(R + "results/ib_scores.csv").rename(columns={"pair_group": "group", "raw": "raw_ref", "score": "score_ref"})
mg = df.merge(ref[["model", "metric", "group", "seed", "raw_ref", "score_ref", "score_alt"]], on=["model", "metric", "group", "seed"], suffixes=("", "_ref"), how="inner")
print("rows merged", len(mg), "max|raw diff|", (mg.raw - mg.raw_ref).abs().max(), "max|score diff|", (mg.score_primary - mg.score_ref).abs().max(), "alt diff", (mg.score_alt - mg.score_alt_ref).abs().max())
mg.to_csv(A + "recompute_vs_csv.csv", index=False)
# raw table ('all', mean over seeds)
def ms(x): return x.mean(), x.std(ddof=1)
out = {}
print("\nRAW & scores (group, metric, model): mean sd over 5 seeds")
tab = []
for g in ["all", "ood_any", "ind", "ood_both"]:
    for m in spec:
        for mod in ["ours", "random", "area", "oracle", "oracle_linear", "noninf", "const"]:
            d = df[(df.group == g) & (df.metric == m) & (df.model == mod)].sort_values("seed")
            tab.append(dict(group=g, metric=m, model=mod, raw_mean=d.raw.mean(), raw_sd=d.raw.std(ddof=1), score_mean=d.score_primary.mean(), score_sd=d.score_primary.std(ddof=1), alt_mean=d.score_alt.mean(), alt_sd=d.score_alt.std(ddof=1),
                            orclin_const_mean=d.score_orclin_const.mean()))
tb = pd.DataFrame(tab); tb.to_csv(A + "score_table.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(tb[tb.group.isin(["all", "ood_any"])].round(4).to_string())
# denominators
print("\nDenominators (all, mean over seeds): oracle_raw, noninf_raw, const_raw")
for g in ["all", "ood_any"]:
    for m in spec:
        d = df[(df.group == g) & (df.metric == m) & (df.model == "ours")]
        print(g, m, "orc %.4f noninf %.4f const %.4f  den_iid %.4f den_const %.4f" % (d.r_orc.mean(), d.r_noninf.mean(), d.r_const.mean(), (d.r_orc - d.r_noninf).mean(), (d.r_orc - d.r_const).mean()))
# H-f recompute with seed-paired diff and sensitivity
def hf(col="score_primary", g="all"):
    a = df[(df.model == "ours") & (df.metric == "D-IB_area=") & (df.group == g)].sort_values("seed")[col].values
    b = df[(df.model == "ours") & (df.metric == "D-IB_area≠") & (df.group == g)].sort_values("seed")[col].values
    return a, b
print("\nH-f")
res = {}
for g in grp:
    for col in ["score_primary", "score_alt", "score_iid", "score_const", "score_orclin_const"]:
        a, b = hf(col, g)
        d = a - b
        sdm = a.std(ddof=1), b.std(ddof=1)
        tol = 2 * max(sdm)
        from scipy import stats
        tw = stats.ttest_rel(a, b); wl = stats.ttest_ind(a, b, equal_var=False)
        se = d.std(ddof=1) / np.sqrt(5); ci = (d.mean() - stats.t.ppf(.975, 4) * se, d.mean() + stats.t.ppf(.975, 4) * se)
        res[(g, col)] = dict(eq=a.mean(), neq=b.mean(), diff=abs(a.mean() - b.mean()), tol=tol, ratio=abs(a.mean() - b.mean()) / tol, paired_t_p=tw.pvalue, welch_p=wl.pvalue, d_mean=d.mean(), d_ci_lo=ci[0], d_ci_hi=ci[1], dz=d.mean() / d.std(ddof=1), n_pos=int((d < 0).sum()))
hfdf = pd.DataFrame(res).T; hfdf.to_csv(A + "hf_sensitivity.csv"); print(hfdf.round(4).to_string())
# H-f at multiple sd_mult
a, b = hf()
print("diff", abs(a.mean() - b.mean()), "per-seed a", a.round(3), "b", b.round(3))
for k in [1, 1.5, 1.92, 2, 2.5, 3]:
    print("sd_mult", k, "tol", k * max(a.std(ddof=1), b.std(ddof=1)), "supported" if abs(a.mean() - b.mean()) <= k * max(a.std(ddof=1), b.std(ddof=1)) else "refuted")
print("sd_mult threshold for flip:", abs(a.mean() - b.mean()) / max(a.std(ddof=1), b.std(ddof=1)))
# raw-level counts and pair counts
for m in ["D-IB", "D-IB_area=", "D-IB_area≠", "R-IB", "R-IB_fold"]:
    print("n pairs", m, {g: int((spec[m][0] & grp[g]).sum()) for g in grp})
# H-e hypothetical: ours vs random, ours vs area
print("\nH-e hypothetical (ood_any)")
for m in ["R-IB", "D-IB"]:
    for col in ["score_primary", "score_alt"]:
        o, r_, a_ = [df[(df.model == k) & (df.metric == m) & (df.group == "ood_any")].sort_values("seed")[col].values for k in ["ours", "random", "area"]]
        print(m, col, "ours %.4f(%.4f) random %.4f(%.4f) area %.4f(%.4f)  |o-r| %.4f  2*maxsd %.4f ratio %.2f  o-a %.4f" % (o.mean(), o.std(ddof=1), r_.mean(), r_.std(ddof=1), a_.mean(), a_.std(ddof=1), abs(o.mean() - r_.mean()), 2 * max(o.std(ddof=1), r_.std(ddof=1)), abs(o.mean() - r_.mean()) / (2 * max(o.std(ddof=1), r_.std(ddof=1))), o.mean() - a_.mean()))
# score > 1 diagnostics: per seed ours/area/oracle raw all-group D-IB
print("\nD-IB all, per seed raw: ")
for mod in ["ours", "random", "area", "oracle", "oracle_linear", "noninf", "const"]:
    d = df[(df.model == mod) & (df.metric == "D-IB") & (df.group == "all")].sort_values("seed")
    print(mod, d.raw.round(4).values)
# task-level: fraction of tasks, bootstrap over tasks for ours vs oracle D-IB (all)
rng = np.random.default_rng(0)
m_ = spec["D-IB"][0] & grp["all"]
def per_task_raw(Y, mask): return 1 - (Y[:, I[mask]] == Y[:, J[mask]]).mean(1)
for mod, key in [("ours", ("pred_out6", "ours")), ("area", ("base", "area")), ("oracle", ("base", "oracle_mlp"))]:
    pt = np.array([per_task_raw((P(key[0], key[1], s)["A"] > .5).astype(np.int8), m_) for s in SEEDS])
    print(mod, "per-task D-IB raw: mean over seeds-tasks", pt.mean().round(4), "sd across tasks (seed-avg)", pt.mean(0).std(ddof=1).round(4), "sd across seeds of task-mean", pt.mean(1).std(ddof=1).round(4))
# bootstrap over tasks (100) of ours score
ptO = np.array([per_task_raw((P("pred_out6", "ours", s)["A"] > .5).astype(np.int8), m_) for s in SEEDS]).mean(0)
ptR = np.array([per_task_raw((P("base", "oracle_mlp", s)["A"] > .5).astype(np.int8), m_) for s in SEEDS]).mean(0)
bs = []
for _ in range(2000):
    ix = rng.integers(0, 100, 100); bs.append(ptO[ix].mean() / ptR[ix].mean())
print("task-bootstrap ratio ours/oracle raw D-IB (const ref): mean %.4f 95%% [%.4f %.4f]" % (np.mean(bs), *np.percentile(bs, [2.5, 97.5])))
# 5-B calibration recompute
from numpy import triu
def dist(Y): 
    Y = np.asarray(Y, np.float64); a = (Y ** 2).mean(0); G = Y.T @ Y / Y.shape[0]; return np.maximum(a[:, None] + a[None] - 2 * G, 0)
cal = []
for s in SEEDS:
    Dm = {k: dist(P(*v, s)["B"]) for k, v in {"ours": ("pred_out6", "ours"), "random": ("pred_out6", "random"), "area": ("base", "area"), "oracle": ("base", "oracle_mlp"), "oracle_linear": ("base", "oracle_linear")}.items()}
    Dm["truth"] = dist(T["g_eval"][:100])
    norm = np.triu(Dm["oracle"], 1).max()
    for g, sel in {"all": np.triu(np.ones((n, n), bool), 1), "ind": np.triu(is_ind[:, None] & is_ind[None], 1), "extrap_any": np.triu(is_ex[:, None] | is_ex[None], 1)}.items():
        x = Dm["oracle"][sel]
        edges = np.linspace(0, x.max(), 21); idx = np.clip(np.digitize(x, edges[1:-1]), 0, 19)
        for mod in Dm:
            y = Dm[mod][sel]
            devs = [abs(y[idx == b].mean() - x[idx == b].mean()) / norm for b in range(20) if (idx == b).any()]
            # pair-weighted dev and dev vs 'truth' x-axis scale
            pw = np.abs(y - x).mean() / norm
            cal.append(dict(model=mod, group=g, seed=s, dev=np.mean(devs), dev_pairweighted=pw, norm=norm, n_pairs=int(sel.sum()), mean_oracle_dist=x.mean() / norm, mean_model_dist=y.mean() / norm))
cd = pd.DataFrame(cal); cd.to_csv(A + "recomputed_calibration_dev.csv", index=False)
cdev = pd.read_csv(R + "results/calibration_dev.csv"); print(cdev.head(3)); print(cdev.columns.tolist())
g_ = cd.groupby(["group", "model"])[["dev", "dev_pairweighted", "mean_oracle_dist", "mean_model_dist"]].agg(["mean", "std"]); print(g_.round(4).to_string())
c2 = cdev[cdev.layer == "pred_out6"] if "layer" in cdev else cdev
mm = cd.merge(c2.rename(columns={"value": "value_ref"}), on=["model", "group", "seed"]); print("calib max|diff| vs csv", (mm.dev - mm.value_ref).abs().max())
# 5-D
law = []
role_ = role
s_eval = T["s_eval"]; kind = T["kind"]
sl = {"interp": role_ == "interp", "extrap": role_ == "extrap", "early": kind == 0, "late": kind == 1}
def ols(x, y):
    A_ = np.stack([x, np.ones_like(x)], 1); c = np.linalg.lstsq(A_, y, rcond=None)[0]; res = y - A_ @ c; return c[0], c[1], 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
for s in SEEDS:
    for mod, key in {"ours": ("pred_out6", "ours"), "random": ("pred_out6", "random"), "area": ("base", "area"), "oracle": ("base", "oracle_mlp"), "oracle_linear": ("base", "oracle_linear")}.items():
        D = P(*key, s)["D"]
        for k, mk in sl.items():
            a, b, r2 = ols(s_eval[mk], D[0][mk]); law.append(dict(model=mod, slice=k, seed=s, slope=a, intercept=b, r2=r2, n=int(mk.sum())))
        for j, nm in enumerate(["labels_a060+ind", "labels_ind+a160", "labels_a060+a160"]):
            a, b, r2 = ols(s_eval, D[j + 1]); law.append(dict(model=mod, slice=nm, seed=s, slope=a, intercept=b, r2=r2, n=n))
ld = pd.DataFrame(law); ld.to_csv(A + "recomputed_law.csv", index=False)
lt = ld.groupby(["model", "slice"]).agg(slope_mean=("slope", "mean"), slope_sd=("slope", lambda x: x.std(ddof=1)), r2=("r2", "mean"), icpt=("intercept", "mean")).round(4)
print(lt.to_string()); lt.to_csv(A + "law_table.csv")
lr = pd.read_csv(R + "results/law_fits.csv"); lr = lr[lr.layer == "pred_out6"]
mm = ld.merge(lr, on=["model", "slice", "seed"], suffixes=("", "_ref")); print("law max|slope diff|", (mm.slope - mm.slope_ref).abs().max(), len(mm))
# range of s_eval and training range
print("s_eval range", s_eval.min(), s_eval.max(), "s_range", T["s_range"], "n labels per D", T["ids_d"].shape)
print("lvl s by role:", {r: (s_eval[role_ == r].min().round(3), s_eval[role_ == r].max().round(3), int((role_ == r).sum())) for r in np.unique(role_)})
print("ids_d level counts:", [np.bincount(T["lvl_flat"][ids], minlength=9).tolist() for ids in T["ids_d"]])
# slope tests vs 0.8 and 1 (t over seeds)
from scipy import stats
for k in sorted(ld.slice.unique()):
    v = ld[(ld.model == "ours") & (ld.slice == k)].slope.values
    print(k, "ours slope %.3f sd %.3f; t vs 0.8: p=%.4f ; vs 1.0 p=%.4g; 95%%CI(t) [%.3f %.3f]" % (v.mean(), v.std(ddof=1), stats.ttest_1samp(v, .8).pvalue, stats.ttest_1samp(v, 1).pvalue, *stats.t.interval(.95, 4, v.mean(), v.std(ddof=1) / np.sqrt(5))))
