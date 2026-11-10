"""Stats review of 004 (reviewer: stats). Run: ~/miniconda3/envs/lab07/bin/python recompute_stats.py   (numpy only, no pandas)
Reads ../results/depth_r2.csv, ../results/summary.json, ../results/next_z_mse.csv. Writes stats_out.txt next to itself."""
import csv, json, collections, sys, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "results")
rows = list(csv.DictReader(open(f"{R}/depth_r2.csv")))
for r in rows:
    r["r2"] = float(r["r2"]) if r["r2"] != "" else np.nan; r["depth"] = int(r["depth"])
S = json.load(open(f"{R}/summary.json"))
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def cur(module, probe, fit, control, tp, level="ALL"):
    c = {}
    for r in rows:
        if (r["module"], r["probe"], r["fit"], r["control"], r["timepoint"], r["level"]) == (module, probe, fit, control, tp, level):
            c.setdefault(r["split"], {}).setdefault(r["depth"], []).append((r["r2"], r["train_seed"], r["seed"]))
    return c
def arr(c, s, d): return np.array([x[0] for x in c[s][d]])
def first_drop(ds, vs, thr):
    m = None
    for d, v in zip(ds, vs):
        if m is not None and (m - v) >= thr - 1e-12: return d, m - v
        m = v if m is None else max(m, v)
    return None
def maxdrop(ds, vs):
    m = vs[0]; best = 0
    for v in vs:
        m = max(m, v); best = max(best, m - v)
    return best
prim = ("linear", "new", "real", "late")
P("== 1. primary curve recompute vs summary.json")
for mod in ("encoder", "predictor"):
    c = cur(mod, *prim)
    for s in ("train", "interp", "extrap"):
        ds = sorted(c[s]); vs = [arr(c, s, d).mean() for d in ds]
        sds = [arr(c, s, d).std(ddof=1) for d in ds]
        P(mod, s, "n/depth", len(c[s][ds[0]]), "mean", [round(v, 4) for v in vs], "sd", [round(v, 4) for v in sds],
          "maxdrop_from_runmax", round(maxdrop(ds, vs), 4), "first_drop@0.2", first_drop(ds, vs, 0.2))
# gap recompute
P("== 2. ind_ood_gap recompute vs summary.json")
mism = 0
for mod in ("encoder", "predictor"):
    c = cur(mod, *prim)
    for s in ("interp", "extrap"):
        for p in S["ind_ood_gap"][mod][s]["per_depth"]:
            d = p["depth"]; a = arr(c, "train", d); b = arr(c, s, d)
            tol = 2 * max(a.std(ddof=1), b.std(ddof=1)); gap = a.mean() - b.mean()
            ok = abs(gap - p["gap"]) < 1e-9 and abs(tol - p["tol"]) < 1e-9 and (abs(gap) <= tol) == p["inside_seed_sd"]
            mism += (not ok)
            P(mod, s, d, "gap", round(gap, 4), "tol", f"{tol:.3g}", "sd_train", f"{a.std(ddof=1):.3g}", "sd_ood", f"{b.std(ddof=1):.3g}", "unique_vals_train", len(set(np.round(a, 12))), "unique_ood", len(set(np.round(b, 12))), "match", ok)
P("mismatches", mism)
P("== 3. tol~0 diagnosis: encoder linear/new/real per-seed raw r2 at depth 0 and 2, split train")
c = cur("encoder", *prim)
for d in (0, 2, 4, 10):
    P("enc d", d, "train", [x[0] for x in c["train"][d]], "interp", [x[0] for x in c["interp"][d]], "extrap", [x[0] for x in c["extrap"][d]])
P("== 3b. sd over probe seeds, by (module, probe, control, tp): fraction of cells with sd<1e-9 (split rows, level ALL)")
for mod in ("encoder", "predictor"):
    for probe in ("linear", "attentive"):
        cc = cur(mod, probe, "new", "real", "late")
        z = tot = 0
        for s in cc:
            for d in cc[s]:
                tot += 1; z += arr(cc, s, d).std(ddof=1) < 1e-9
        P(mod, probe, "zero-sd cells", z, "/", tot)
P("   predictor: sd decomposition (train-seed vs probe-seed) primary, split extrap")
c = cur("predictor", *prim)
for s in ("train", "interp", "extrap"):
    for d in sorted(c[s]):
        g = collections.defaultdict(list)
        for v, ts, sd in c[s][d]: g[ts].append(v)
        means = [np.mean(v) for v in g.values()]; within = np.mean([np.std(v, ddof=1) for v in g.values()])
        P("pred", s, d, "per-train-seed mean", [round(m, 4) for m in means], "between-trainseed sd", round(np.std(means, ddof=1), 4), "mean within-trainseed(probe) sd", round(within, 5))
# 4 baselines & controls
P("== 4. controls vs real, primary linear late, level ALL, per split (mean R2 across seeds)")
def meanline(mod, probe, fit, ctrl, tp, s):
    c = cur(mod, probe, fit, ctrl, tp)
    if s not in c: return None
    return {d: arr(c, s, d).mean() for d in sorted(c[s])}
for mod in ("encoder", "predictor"):
    for s in ("train", "interp", "extrap"):
        for ctrl in ("real", "random_init", "label_shuffle"):
            m = meanline(mod, "linear", "new", ctrl, "late", s)
            P(mod, s, ctrl, {d: round(v, 3) for d, v in m.items()})
for probe in ("linear", "attentive"):
    for ctrl in ("pixel", "mask_area_bbox"):
        if probe == "attentive" and ctrl == "mask_area_bbox": continue
        c = cur("baseline", probe, "new", ctrl, "late")
        P("baseline", probe, ctrl, {s: {d: [round(arr(c, s, d).mean(), 3), round(arr(c, s, d).std(ddof=1), 3)] for d in c[s]} for s in c})
P("   baseline early:")
for probe, ctrl in (("linear", "pixel"), ("linear", "mask_area_bbox"), ("attentive", "pixel")):
    c = cur("baseline", probe, "new", ctrl, "early")
    P("baseline early", probe, ctrl, {s: {d: round(arr(c, s, d).mean(), 3) for d in c[s]} for s in c})
# effect size vs mask per depth extrap/interp
P("== 5. real vs mask_area_bbox (extrap, interp, train): Cohen d (pooled) and diff; mask baseline has depth key")
mk = cur("baseline", "linear", "new", "mask_area_bbox", "late")
for s in ("train", "interp", "extrap"):
    mv = np.concatenate([arr(mk, s, d) for d in mk[s]]); P("mask baseline", s, "depths", sorted(mk[s]), "mean", round(mv.mean(), 4), "sd", round(mv.std(ddof=1), 4), "n", len(mv))
    for mod in ("encoder", "predictor"):
        c = cur(mod, *prim)
        for d in sorted(c[s]):
            a = arr(c, s, d); sp = np.sqrt((a.var(ddof=1) + mv.var(ddof=1)) / 2)
            P("  ", mod, s, d, "real-mask", round(a.mean() - mv.mean(), 4), "d~", round((a.mean() - mv.mean()) / sp, 2) if sp > 0 else "inf", "n", len(a))
# 6 attentive vs linear
P("== 6. attentive vs linear primary-style curves")
for mod in ("encoder", "predictor"):
    for probe in ("linear", "attentive"):
        c = cur(mod, probe, "new", "real", "late")
        for s in ("train", "interp", "extrap"):
            ds = sorted(c[s]); vs = [arr(c, s, d).mean() for d in ds]
            P(mod, probe, s, [round(v, 3) for v in vs], "maxdrop", round(maxdrop(ds, vs), 4))
# 7 early vs late
P("== 7. early vs late (linear/new/real)")
for mod in ("encoder", "predictor"):
    for tp in ("early", "late"):
        c = cur(mod, "linear", "new", "real", tp)
        for s in ("train", "interp", "extrap"):
            ds = sorted(c[s]); vs = [arr(c, s, d).mean() for d in ds]
            P(mod, tp, s, [round(v, 3) for v in vs], "maxdrop", round(maxdrop(ds, vs), 4))
# 8 exp1_probe predictor (ii)
P("== 8. predictor (ii) exp1_probe, linear and attentive, by timepoint, splits")
for probe in ("linear", "attentive"):
    for tp in ("early", "late", "rollout_t+1", "rollout_t+4", "rollout_t+8"):
        c = cur("predictor", probe, "exp1_probe", "real", tp)
        for s in ("train", "interp", "extrap"):
            ds = sorted(c[s]); vs = [arr(c, s, d).mean() for d in ds]
            P("(ii)", probe, tp, s, [round(v, 3) for v in vs], "maxdrop", round(maxdrop(ds, vs), 3), "first_drop@0.2", first_drop(ds, vs, 0.2))
P("   reference encoder_tokens")
for probe in ("linear", "attentive"):
    for tp in ("late", "rollout_t+8"):
        c = cur("reference", probe, "exp1_probe", "encoder_tokens", tp)
        P("ref", probe, tp, {s: {d: round(arr(c, s, d).mean(), 3) for d in sorted(c[s])} for s in ("train", "interp", "extrap") if s in c})
# 9 next_z mse
P("== 9. (iii) next_z_mse")
m = list(csv.DictReader(open(f"{R}/next_z_mse.csv")))
print(sorted(set((r["control"], r["timepoint"], r["split"]) for r in m))[:40])
g = collections.defaultdict(list)
for r in m:
    if r["split"] in ("train", "interp", "extrap") and r["level"] in ("ALL",) : g[(r["control"], r["timepoint"], r["split"], int(r["depth"]))].append(float(r["mse"]))
for ctrl in sorted(set(k[0] for k in g)):
    for tp in sorted(set(k[1] for k in g)):
        for s in ("train", "extrap"):
            ds = sorted(k[3] for k in g if k[:3] == (ctrl, tp, s))
            if ds: P("(iii)", ctrl, tp, s, [(d, round(np.mean(g[(ctrl, tp, s, d)]), 4)) for d in ds], "n", len(g[(ctrl, tp, s, ds[0])]))
# 10 threshold sensitivity
P("== 10. threshold sensitivity (SENSITIVITY ONLY, not a verdict change): first_drop on primary and on controls' means, by split")
def sens(label, c):
    for thr in (0.05, 0.1, 0.15, 0.2, 0.3):
        res = {}
        for s in ("train", "interp", "extrap"):
            ds = sorted(c[s]); vs = [arr(c, s, d).mean() for d in ds]; fd = first_drop(ds, vs, thr)
            res[s] = None if fd is None else (fd[0], round(fd[1], 3))
        P(label, "thr", thr, res)
for mod in ("encoder", "predictor"):
    sens(mod + " primary", cur(mod, *prim))
    sens(mod + " attentive late", cur(mod, "attentive", "new", "real", "late"))
    sens(mod + " linear early", cur(mod, "linear", "new", "real", "early"))
    sens(mod + " (ii) linear late exp1", cur(mod, "linear", "exp1_probe", "real", "late")) if mod == "predictor" else None
# per-level min across the levels (maybe a level-level drop that the aggregated hides)
P("== 11. per-level primary curves (level != ALL) max drop from running max, mean over seeds")
for mod in ("encoder", "predictor"):
    for lv in ("a026", "a040", "a060", "a080", "ind", "a125", "a160", "a200", "a248"):
        c = cur(mod, *prim, level=lv)
        if not c: continue
        for s in c:
            ds = sorted(c[s]); vs = [arr(c, s, d).mean() for d in ds]
            P(mod, lv, s, [round(v, 3) for v in vs], "maxdrop", round(maxdrop(ds, vs), 3))
# seed-level drop (per-seed max drop) for primary: how many seeds individually exceed 0.2
P("== 12. per-(train_seed,probe_seed) max drop primary (do individual seeds exceed 0.2?)")
for mod in ("encoder", "predictor"):
    c = cur(mod, *prim)
    for s in ("train", "interp", "extrap"):
        per = collections.defaultdict(dict)
        for d in c[s]:
            for v, ts, sd in c[s][d]: per[(ts, sd)][d] = v
        md = [maxdrop(sorted(p), [p[d] for d in sorted(p)]) for p in per.values()]
        P(mod, s, "n series", len(md), "max drop min/mean/max", round(min(md), 3), round(np.mean(md), 3), round(max(md), 3), "n>=0.2", sum(x >= 0.2 for x in md))
# 13 bootstrap CI of key endpoint differences
P("== 13. extrap: depth gain/loss relative to depth 0 (linear late), bootstrap 95% CI over train-seed x probe-seed (predictor) / probe seeds (encoder)")
rng = np.random.default_rng(0)
for mod in ("encoder", "predictor"):
    c = cur(mod, *prim)
    for s in ("interp", "extrap"):
        ds = sorted(c[s]); last = ds[-1]; best = max(ds, key=lambda d: arr(c, s, d).mean())
        for d in (best, last):
            a = arr(c, s, d); b = arr(c, s, ds[0])
            diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
            P(mod, s, "depth", d, "minus depth", ds[0], "mean", round(a.mean() - b.mean(), 4), "CI", np.round(np.percentile(diffs, [2.5, 97.5]), 4))
open(f"{H}/stats_out.txt", "w").write("\n".join(out))
