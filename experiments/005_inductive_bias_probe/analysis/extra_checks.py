"""Extra stats checks for 005: label accuracy of each model, per-model H-f gap, task-bootstrap CI for H-f gap and for ours-vs-random D-IB. Run with lab05 python."""
import numpy as np
R = "/home/jaekyungkim/mylab/experiments/005_inductive_bias_probe/"
T = np.load(R + "results/tasks.npz", allow_pickle=True)
lvl, q = T["lvl"], T["q"]; n = len(lvl); L = T["labels_a"]  # (100,9)
I, J = np.triu_indices(n, 1)
sl = lvl[I] == lvl[J]; sq = q[I] == q[J]
mE, mN = (~sl & sq), (~sl & ~sq)
def P(l, m, s): return (np.load(R + f"results/predictions/{l}__{m}__seed{s}.npz")["A"] > .5).astype(np.int8)
models = {"ours": ("pred_out6", "ours"), "random": ("pred_out6", "random"), "area": ("base", "area"), "oracle_mlp": ("base", "oracle_mlp"), "oracle_linear": ("base", "oracle_linear")}
Y = {k: np.stack([P(*v, s) for s in range(5)]) for k, v in models.items()}   # (5,100,n)
truth = L[:, lvl]  # (100,n) true binary label per input
print("label accuracy on eval inputs (mean over seeds/tasks):")
for k, y in Y.items():
    acc = (y == truth[None]).mean(); accd = {r: (y[:, :, T['role'] == r] == truth[None][:, :, T['role'] == r]).mean().round(4) for r in ["trained", "interp", "extrap"]}
    print(" ", k, round(acc, 4), accd, "frac predicted 1: %.3f" % y.mean())
print("chance for random labels: 0.5; true-label D-IB raw (perfect state): %.4f" % (1 - (truth[:, I[~sl]] == truth[:, J[~sl]]).mean()))
print("true-label D-IB area= / area≠ raw:", 1 - (truth[:, I[mE]] == truth[:, J[mE]]).mean(), 1 - (truth[:, I[mN]] == truth[:, J[mN]]).mean())
# per-task raw for each model: (5,100) -> seed-avg (100)
def pt(y, m): return 1 - (y[:, :, I[m]] == y[:, :, J[m]]).mean(2)   # (5,100)
PT = {k: {"E": pt(y, mE).mean(0), "N": pt(y, mN).mean(0)} for k, y in Y.items()}
rng = np.random.default_rng(0)
def boot(f, B=4000):
    v = [f(rng.integers(0, 100, 100)) for _ in range(B)]; return np.mean(v), np.percentile(v, [2.5, 97.5])
print("\nH-f gap (score_eq - score_neq, const-ref, oracle=MLP) per model, with task-bootstrap 95% CI (resample 100 tasks; seeds averaged):")
for k in PT:
    o = PT["oracle_mlp"]
    f = lambda ix, k=k: PT[k]["E"][ix].mean() / o["E"][ix].mean() - PT[k]["N"][ix].mean() / o["N"][ix].mean()
    pt_est = f(np.arange(100)); m, ci = boot(f)
    print(" %-14s gap point %.4f  boot mean %.4f  95%% CI [%.4f, %.4f]" % (k, pt_est, m, *ci))
print("\nD-IB(all) score diff ours - random (const ref, oracle MLP), task bootstrap:")
def dall(k): return 1 - (Y[k][:, :, I[~sl]] == Y[k][:, :, J[~sl]]).mean(2)
Do = {k: dall(k).mean(0) for k in Y}
for a, b in [("ours", "random"), ("ours", "area"), ("random", "area")]:
    f = lambda ix: (Do[a][ix].mean() - Do[b][ix].mean()) / Do["oracle_mlp"][ix].mean()
    m, ci = boot(f); print(" %s-%s point %.4f 95%% CI [%.4f, %.4f]" % (a, b, f(np.arange(100)), *ci))
# R-IB ood_any ours vs random: task bootstrap
ood = np.isin(T["role"], ["interp", "extrap"]); mo = sl & (ood[I] | ood[J])
Ro = {k: (Y[k][:, :, I[mo]] == Y[k][:, :, J[mo]]).mean(2).mean(0) for k in Y}
for a, b in [("ours", "random"), ("ours", "area")]:
    f = lambda ix: (Ro[a][ix].mean() - Ro[b][ix].mean()) / 0.5   # const-ref R den ~0.5 (iid ref)
    m, ci = boot(f); print(" R-IB ood_any score diff %s-%s point %.4f CI [%.4f, %.4f]" % (a, b, f(np.arange(100)), *ci))
# seed-level independence note: variance decomposition of ours D-IB per-task (seed vs task)
d = dall("ours"); print("\nours D-IB(all) per-task raw: var across tasks %.4f ; mean within-task var across seeds %.5f" % (d.mean(0).var(ddof=1), d.var(0, ddof=1).mean()))
# prediction diversity: within-level agreement
print("Fraction of tasks where ours constant over all inputs:", float((Y["ours"].min(2) == Y["ours"].max(2)).mean()))
