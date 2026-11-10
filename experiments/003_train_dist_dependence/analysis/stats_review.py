"""Stats review of 003 (analyst-stats). Run: ~/miniconda3/envs/lab05/bin/python stats_review.py
Reads ../results.json, ../config.yaml values via results.json['config'], decoder_recon_by_epoch.json, ../../001_*/results.json.
Writes tables (csv/json) and figures into this folder. Does not modify experiment code."""
import json, itertools, math, os
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.load(open(HERE.parent / "results.json"))
B = json.load(open(HERE.parent.parent / "001_size_collapse_diagnosis" / "results.json"))
D = json.load(open(HERE / "decoder_recon_by_epoch.json"))
cfg = R["config"]; crit = cfg["criteria"]; runs = R["runs"]
AR = {s["name"]: s["area_ratio"] for s in cfg["data"]["datasets"]["cloth_sim"]["sizes"]}
GAIN = crit["h2_slope_gain_min"]; NREQ = crit["seeds_required_same_direction"]
base = {n: B["runs"][n]["diagnose"]["size_curve"]["slope_model_vs_real_group_means"]["ol@k12"] for n in crit["h2_baseline"]["runs"]}
bs = np.array([v["slope"] for v in base.values()])
out = {}

# ---------- helpers (copied logic from src/exp003/metrics.py, verdict.py) ----------
def boot_slope(xg, yg, n=500, seed=0):
    rng = np.random.RandomState(seed); gs = list(xg)
    def sl(idx=None):
        mx, my = [], []
        for g in gs:
            x, y = xg[g], yg[g]; ii = np.arange(len(x)) if idx is None else rng.randint(0, len(x), len(x))
            mx.append(np.mean(np.log(np.maximum(x[ii], 1)))); my.append(np.mean(np.log(np.maximum(y[ii], 1))))
        return float(np.polyfit(mx, my, 1)[0])
    pt = sl(); d = np.array([sl(True) for _ in range(n)])
    return pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def find_c(order, ratio_mean):
    la = [math.log(AR[g]) for g in order]
    y = [None if (r is None or r <= 0) else math.log(r) for r in ratio_mean]
    if any(v is None for v in y): return None, "invalid"
    cr = []
    for i in range(len(order) - 1):
        a, b = y[i], y[i + 1]
        if (a < 0 < b) or (a > 0 > b): cr.append(la[i] + (0 - a) * (la[i + 1] - la[i]) / (b - a))
    if len(cr) == 1: return cr[0], "ok"
    if len(cr) == 0: return None, "no_crossing_all_above" if all(v > 0 for v in y) else "no_crossing_all_below"
    return None, "multiple"

def h1_seed(run, train_group, ratio_ci=None, ratio_mean=None):
    """returns (status, supports_h1, supports_center, ci_met). ratio_ci: {g:(lo,hi)}, ratio_mean: {g:mean}"""
    order = run["groups_in_order"]; a_tr = AR[train_group]; flags = {}
    for g in order:
        if g == train_group: continue
        lo, hi = ratio_ci[g]; toward_smaller = a_tr < AR[g]
        flags[g] = (hi < 1.0) if toward_smaller else (lo > 1.0)
    ci_met = all(flags.values())
    lc, st = find_c(order, [ratio_mean[g] for g in order])
    if lc is None: return "not_evaluable:" + st, None, None, ci_met
    dtr, dind = abs(lc - math.log(a_tr)), abs(lc)
    return "evaluated", (dtr < dind) and ci_met, dind < dtr, ci_met

def pattern(seedres):
    ev = [s for s in seedres if s[0] == "evaluated"]
    if len(ev) < NREQ or len(seedres) < NREQ: return "not_evaluable"
    if all(s[1] for s in ev): return "h1"
    if all(s[2] for s in ev): return "center"
    return "mixed"

def stored_ratio(r): return ({g: (r["by_size"][g]["ratio_ol_over_real"]["lo"], r["by_size"][g]["ratio_ol_over_real"]["hi"]) for g in r["groups_in_order"]},
                             {g: r["by_size"][g]["ratio_ol_over_real"]["mean"] for g in r["groups_in_order"]})

# ---------- 1. recompute verdict ----------
rec = {}
for dist, tg in crit["h1_single_models"].items():
    res = []
    for s in cfg["train"]["seeds"]:
        r = runs["exp003_%s_s%d" % (dist, s)]; ci, mn = stored_ratio(r)
        res.append(h1_seed(r, tg, ci, mn))
    rec[dist] = {"seeds": [list(x) for x in res], "pattern": pattern(res)}
h1_status = "pass" if all(v["pattern"] == "h1" for v in rec.values()) else ("fail" if all(v["pattern"] == "center" for v in rec.values()) else "not_evaluable")
stored = R["verdict"]["H-H1"]
rec_ok = {d: rec[d]["pattern"] == stored["patterns"][d] for d in rec}
def h2_status(slopes_by_dist_seed_list, thr=GAIN):
    sl = [v for v in slopes_by_dist_seed_list if v is not None]
    if len(sl) < NREQ: return "not_evaluable"
    dv = np.array([a - b for a in sl for b in bs])
    if (dv >= thr).all(): return "pass"
    if (np.abs(dv) < thr).all(): return "fail"
    return "not_evaluable"
h2 = {}
for d in crit["h2_distributions"]:
    sl = [runs["exp003_%s_s%d" % (d, s)]["slope_ol"]["all_sizes"]["slope"] for s in cfg["train"]["seeds"]]
    h2[d] = h2_status(sl)
h2_all = "pass" if set(h2.values()) == {"pass"} else ("fail" if set(h2.values()) == {"fail"} else "not_evaluable")
out["verdict_recompute"] = {"H-H1": {"patterns": {d: v["pattern"] for d, v in rec.items()}, "status": h1_status, "stored_patterns": stored["patterns"], "stored_status": stored["status"]},
                            "H-H2": {"per_dist": h2, "status": h2_all, "stored": {d: R["verdict"]["H-H2"]["distributions"][d]["status"] for d in h2}, "stored_status": R["verdict"]["H-H2"]["status"]},
                            "baseline_slopes": {k: v["slope"] for k, v in base.items()}}
print("RECOMPUTE", json.dumps(out["verdict_recompute"], indent=1))

# recompute slopes/CIs from per-episode areas
chk = []
for n, r in runs.items():
    xs = {g: np.array(r["by_size"][g]["area_real_per_episode"]) for g in r["groups_in_order"]}
    ys = {g: np.array(r["by_size"][g]["area_ol_per_episode"]) for g in r["groups_in_order"]}
    pt, lo, hi = boot_slope(xs, ys)
    s = r["slope_ol"]["all_sizes"]; chk.append((n, pt, s["slope"], lo, s["lo"], hi, s["hi"]))
out["slope_recompute_max_abs_diff"] = {"slope": max(abs(c[1] - c[2]) for c in chk), "lo": max(abs(c[3] - c[4]) for c in chk), "hi": max(abs(c[5] - c[6]) for c in chk)}
print("slope recompute diff", out["slope_recompute_max_abs_diff"])

# ---------- 2. sensitivity ----------
sens = {}
# H-H2 mix_adjacent: s0 replaced by any x; or excluded
s1, s2 = [runs["exp003_mix_adjacent_s%d" % i]["slope_ol"]["all_sizes"]["slope"] for i in (1, 2)]
grid = np.linspace(-0.5, 1.5, 401)
st_grid = {x: h2_status([x, s1, s2]) for x in grid}
sens["mix_adjacent_s0_any_value_status_set"] = sorted(set(st_grid.values()))
sens["mix_adjacent_exclude_s0_status"] = h2_status([s1, s2])
d12 = np.array([a - b for a in (s1, s2) for b in bs])
sens["mix_adjacent_pairs_s1_s2"] = {"diffs": d12.tolist(), "n_ge_gain": int((d12 >= GAIN).sum()), "n_abs_lt_gain": int((np.abs(d12) < GAIN).sum()), "n": int(len(d12))}
sens["mix_adjacent_min_thr_for_pass_s1s2"] = float(d12.min())
# H-H2 overall under every mix_adjacent_s0 value: overall status
sens["H2_overall_set_over_s0_grid"] = sorted({("pass" if (h2["mix_wide"] == "pass" and v == "pass") else ("fail" if (h2["mix_wide"] == "fail" and v == "fail") else "not_evaluable")) for v in st_grid.values()})
mw = [runs["exp003_mix_wide_s%d" % i]["slope_ol"]["all_sizes"]["slope"] for i in range(3)]
dmw = np.array([a - b for a in mw for b in bs]); sens["mix_wide_pair_diffs"] = {"min": float(dmw.min()), "max": float(dmw.max()), "margin_over_gain": float(dmw.min() - GAIN)}
# how far would the weakest baseline need to rise / weakest mix_wide fall to flip
sens["mix_wide_flip"] = {"baseline_max_slope_to_flip_given_min_mix": float(min(mw) - GAIN), "observed_baseline_max": float(bs.max()),
                         "min_mix_slope_to_flip_given_max_baseline": float(bs.max() + GAIN), "observed_min_mix": float(min(mw))}
# H-H1: a026 s1 (failed): all possible outcomes; a026 s2 borderline
a026 = rec["single_a026"]["seeds"]
outcomes = {}
for name, cand in {"h1": ("evaluated", True, False, True), "center": ("evaluated", False, True, False), "neither(c closer train, CI fail)": ("evaluated", False, False, False), "null": ("not_evaluable:x", None, None, False)}.items():
    sr = [tuple(a026[0]), cand, tuple(a026[2])]; outcomes[name] = pattern(sr)
sens["a026_s1_possible_outcomes_pattern"] = outcomes
sens["a026_s1_excluded_pattern"] = pattern([tuple(a026[0]), tuple(a026[2])])
# a248 treated as c>=train when all ratios >1 (rule variant) -> pattern of a248
def a248_variant(r):
    ci, mn = stored_ratio(r); st, sh1, sc, cim = h1_seed(r, "a248", ci, mn)
    if st.startswith("not_evaluable:no_crossing_all_above"): return ("evaluated", cim, False, cim)
    return (st, sh1, sc, cim)
a248v = [a248_variant(runs["exp003_single_a248_s%d" % s]) for s in range(3)]
sens["a248_variant_all_above_means_c_ge_train"] = {"seeds": [list(x) for x in a248v], "pattern": pattern(a248v)}
sens["H1_overall_with_variant_and_any_a026_s1"] = sorted({("pass" if (pattern([tuple(a026[0]), c, tuple(a026[2])]) == "h1" and pattern(a248v) == "h1") else "not_evaluable") for c in [("evaluated", True, False, True), ("evaluated", False, True, False), ("evaluated", False, False, False)]})
# estimator sensitivity: mean of per-episode ratios (pre-registered) vs ratio of means vs median
rows = []
est = {}
for n, r in runs.items():
    est[n] = {}
    for kind in ("mean_of_ratios", "ratio_of_means", "median_ratio"):
        mn, ci = {}, {}
        for g in r["groups_in_order"]:
            x = np.array(r["by_size"][g]["area_real_per_episode"], float); y = np.array(r["by_size"][g]["area_ol_per_episode"], float)
            rng = np.random.RandomState(0); idx = rng.randint(0, len(x), (2000, len(x)))
            f = {"mean_of_ratios": lambda a, b: (b / np.maximum(a, 1)).mean(-1), "ratio_of_means": lambda a, b: b.mean(-1) / np.maximum(a.mean(-1), 1), "median_ratio": lambda a, b: np.median(b / np.maximum(a, 1), -1)}[kind]
            pt = float(f(x[None], y[None])[0]); bt = f(x[idx], y[idx]); mn[g] = pt; ci[g] = (float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5)))
        dist = r["distribution"]; tg = crit["h1_single_models"].get(dist)
        if tg: est[n][kind] = h1_seed(r, tg, ci, mn)[:3] + (find_c(r["groups_in_order"], [mn[g] for g in r["groups_in_order"]])[0],)
        est[n][kind + "_ratios"] = [round(mn[g], 3) for g in r["groups_in_order"]]
        if kind == "mean_of_ratios" and n == "exp003_single_a026_s2": sens["a026_s2_a060_ci_mean_of_ratios"] = ci["a060"]
        if kind == "ratio_of_means" and n == "exp003_single_a026_s2": sens["a026_s2_a060_ci_ratio_of_means"] = ci["a060"]
        if kind == "median_ratio" and n == "exp003_single_a026_s2": sens["a026_s2_a060_ci_median"] = ci["a060"]
        if kind == "mean_of_ratios" and n == "exp003_single_a026_s2":
            x = np.array(runs[n]["by_size"]["a060"]["area_real_per_episode"], float); y = np.array(runs[n]["by_size"]["a060"]["area_ol_per_episode"], float)
            q = y / np.maximum(x, 1); sens["a026_s2_a060_ratio_quantiles"] = np.percentile(q, [0, 25, 50, 75, 90, 99, 100]).round(3).tolist(); sens["a026_s2_a060_real_min"] = float(x.min())
            sens["a026_s2_a060_n_ratio_gt1"] = int((q > 1).sum()); sens["a026_s2_a060_top3_ratio_share_of_sum"] = float(np.sort(q)[-3:].sum() / q.sum())
# bootstrap seed sensitivity of a060 CI for a026 s2
r = runs["exp003_single_a026_s2"]; x = np.array(r["by_size"]["a060"]["area_real_per_episode"], float); y = np.array(r["by_size"]["a060"]["area_ol_per_episode"], float); q = y / np.maximum(x, 1)
his = []
for sd in range(20):
    rng = np.random.RandomState(sd); his.append(float(np.percentile(q[rng.randint(0, 100, (2000, 100))].mean(1), 97.5)))
sens["a026_s2_a060_hi_over_20_bootstrap_seeds"] = {"min": min(his), "max": max(his), "frac_hi_ge_1": float(np.mean(np.array(his) >= 1))}
sens["estimator_variants_h1"] = {n: {k: v for k, v in e.items() if not k.endswith("_ratios")} for n, e in est.items() if e and "mean_of_ratios" in e}
sens["estimator_ratios"] = est
out["sensitivity"] = sens
print(json.dumps({k: v for k, v in sens.items() if k not in ("estimator_ratios", "estimator_variants_h1")}, indent=1))
print("estimator variants H1 (status, supH1, supCenter, c)")
for n, e in sens["estimator_variants_h1"].items():
    for k, v in e.items(): print(n, k, v)

# ---------- 3. per-seed table without collapsed runs + effect size ----------
collapsed = ["exp003_mix_adjacent_s0", "exp003_single_a026_s1"]
def tinv975(df): return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}[df]
rows = []
for d in ["single_a026", "single_a248", "mix_adjacent", "mix_wide"]:
    for s in range(3):
        n = "exp003_%s_s%d" % (d, s); r = runs[n]; sl = r["slope_ol"]["all_sizes"]
        rows.append({"run": n, "dist": d, "seed": s, "collapsed_decoder": n in collapsed, "resumed": bool(r["train_env"] and r["train_env"].get("resumed")), "slope": sl["slope"], "ci_lo": sl["lo"], "ci_hi": sl["hi"],
                     "ci_halfwidth": (sl["hi"] - sl["lo"]) / 2, "c": r["center"]["c"], "ol_mean_min": min(r["by_size"][g]["area_ol_mean"] for g in r["groups_in_order"]), "ol_mean_max": max(r["by_size"][g]["area_ol_mean"] for g in r["groups_in_order"])})
for n, v in base.items(): rows.append({"run": n, "dist": "baseline_ind(001)", "seed": int(n[-1]), "collapsed_decoder": False, "resumed": False, "slope": v["slope"], "ci_lo": v["lo"], "ci_hi": v["hi"], "ci_halfwidth": (v["hi"] - v["lo"]) / 2, "c": None})
import csv
with open(HERE / "per_run_slopes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + [])
    w.writeheader(); [w.writerow({k: r.get(k) for k in rows[0]}) for r in rows]
def summ(v):
    v = np.array(v); n = len(v); m = v.mean(); sd = v.std(ddof=1) if n > 1 else float("nan")
    ci = (m - tinv975(n - 1) * sd / math.sqrt(n), m + tinv975(n - 1) * sd / math.sqrt(n)) if n > 1 else (None, None)
    return {"n": n, "mean": float(m), "sd": float(sd), "t_ci95": ci}
def hedges(a, b):
    a, b = np.array(a), np.array(b); na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)); d = (a.mean() - b.mean()) / sp
    return float(d), float(d * (1 - 3 / (4 * (na + nb) - 9)))
def welch(a, b):
    a, b = np.array(a), np.array(b); va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)); return float(a.mean() - b.mean()), se, float(df)
def perm_p(a, b):
    z = list(a) + list(b); na = len(a); obs = np.mean(a) - np.mean(b); cnt = tot = 0
    for idx in itertools.combinations(range(len(z)), na):
        ga = [z[i] for i in idx]; gb = [z[i] for i in range(len(z)) if i not in idx]; t = np.mean(ga) - np.mean(gb); tot += 1; cnt += abs(t) >= abs(obs) - 1e-12
    return cnt / tot, tot
tab = {}
tcrit_cache = {2: 4.303, 3: 3.182, 4: 2.776}
for d in ["single_a026", "single_a248", "mix_adjacent", "mix_wide"]:
    v = [r["slope"] for r in rows if r["dist"] == d and not r["collapsed_decoder"]]
    t = summ(v); t["values"] = v
    t["diff_vs_baseline"] = float(np.mean(v) - bs.mean())
    if len(v) >= 2:
        dd, se, df = welch(v, bs); t["welch_diff"] = dd; t["welch_se"] = se; t["welch_df"] = df
        from math import sqrt
        # t quantile for fractional df via simple lookup/interp over scipy-less table
        tq = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}; lo_df, hi_df = int(math.floor(df)), int(math.ceil(df)); lo_df = max(1, min(4, lo_df)); hi_df = max(1, min(4, hi_df))
        q = tq[lo_df] + (tq[hi_df] - tq[lo_df]) * (df - math.floor(df)) if hi_df != lo_df else tq[lo_df]
        t["welch_ci95"] = (dd - q * se, dd + q * se); t["hedges_g_vs_baseline"] = hedges(v, bs)[1]; t["cohens_d"] = hedges(v, bs)[0]
        p, ntot = perm_p(v, list(bs)); t["exact_perm_p_two_sided"] = p; t["n_perms"] = ntot
    tab[d] = t
tab["baseline(001)"] = summ(bs) | {"values": bs.tolist()}
out["slope_table"] = tab
# mean within-run bootstrap SE vs between-seed SD
for d in ["mix_wide", "single_a248", "mix_adjacent"]:
    hw = [r["ci_halfwidth"] for r in rows if r["dist"] == d and not r["collapsed_decoder"]]
    tab[d]["mean_within_run_boot_halfwidth"] = float(np.mean(hw))
tab["baseline(001)"]["mean_within_run_boot_halfwidth"] = float(np.mean([r["ci_halfwidth"] for r in rows if r["dist"].startswith("baseline")]))
print(json.dumps(tab, indent=1, default=str))

# ---------- a026 non-collapsed: flat prediction or artifact ----------
fl = {}
for n in ["exp003_single_a026_s0", "exp003_single_a026_s2", "exp003_single_a026_s1"]:
    r = runs[n]; g = r["groups_in_order"]
    ol = np.array([r["by_size"][x]["area_ol_mean"] for x in g]); re_ = np.array([r["by_size"][x]["area_real_mean"] for x in g])
    ratio_of_means = (ol / np.maximum(re_, 1)).round(3).tolist()
    # slope with ratio-free regression: ols of mean ol area on real area (linear), and spearman
    from itertools import combinations
    rk = lambda a: np.argsort(np.argsort(a)).astype(float)
    sp = float(np.corrcoef(rk(ol), rk(re_))[0, 1])
    # min ol area per episode anywhere (floor binding?)
    mins = min(min(r["by_size"][x]["area_ol_per_episode"]) for x in g); n0 = sum(int(v == 0) for x in g for v in r["by_size"][x]["area_ol_per_episode"]); n_below1 = sum(int(v < 1) for x in g for v in r["by_size"][x]["area_ol_per_episode"])
    fl[n] = {"ol_mean_by_size": ol.round(1).tolist(), "ol_over_trainsize_real(487.65)": (ol / 487.65).round(2).tolist(), "spearman_olmean_vs_realmean": sp, "min_episode_ol_area": int(mins), "n_episodes_ol_zero": n0, "n_episodes": 900,
             "slope_ci": [r["slope_ol"]["all_sizes"]["lo"], r["slope_ol"]["all_sizes"]["hi"]], "ratio_of_means": ratio_of_means}
# slope s0 vs s2 difference with bootstrap
def boot_diff(n1, n2, nb=2000, seed=0):
    rng = np.random.RandomState(seed); res = []
    def rs(r):
        g = r["groups_in_order"]; xs = {k: np.array(r["by_size"][k]["area_real_per_episode"]) for k in g}; ys = {k: np.array(r["by_size"][k]["area_ol_per_episode"]) for k in g}; return xs, ys
    (x1, y1), (x2, y2) = rs(runs[n1]), rs(runs[n2])
    def sl(xs, ys, rng):
        mx, my = [], []
        for k in xs:
            ii = rng.randint(0, 100, 100); mx.append(np.log(np.maximum(xs[k][ii], 1)).mean()); my.append(np.log(np.maximum(ys[k][ii], 1)).mean())
        return np.polyfit(mx, my, 1)[0]
    for _ in range(nb): res.append(sl(x1, y1, rng) - sl(x2, y2, rng))
    return float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))
fl["s2_minus_s0_slope_boot_ci"] = boot_diff("exp003_single_a026_s2", "exp003_single_a026_s0")
out["a026_flatness"] = fl
print(json.dumps(fl, indent=1))

# ---------- 4. 18 run reconstruction table ----------
dr = D["runs"]; rr = []
for n, x in dr.items():
    ep = sorted(x["by_epoch"], key=int); m = np.array([x["by_epoch"][e]["val_img_mse_reconstructed"] for e in ep]); vl = np.array([x["by_epoch"][e]["val_loss"] for e in ep]); tl = np.array([x["by_epoch"][e]["train_loss"] for e in ep])
    zp = np.array([x["by_epoch"][e]["val_z_visual_err_pred"] for e in ep]); zr = np.array([x["by_epoch"][e]["val_z_visual_err_rollout"] for e in ep])
    jump = m[1:] / m[:-1]
    vj = vl[1:] / vl[:-1]
    rr.append({"run": n, "exp": n[3:6], "dist": n.rsplit("_s", 1)[0][7:], "n_epochs": len(ep), "recon_final": m[-1], "recon_max_jump_ratio": float(jump.max()), "jump_epoch": int(ep[1 + int(jump.argmax())]), "recon_min": m.min(), "recon_final_over_min": m[-1] / m.min(),
               "val_loss_final": vl[-1], "train_loss_final": tl[-1], "val_loss_max_jump_ratio": float(vj.max()), "val_z_pred_final": zp[-1], "val_z_rollout_final": zr[-1], "collapsed": bool(m[-1] > 5 * m.min() and jump.max() > 5)})
with open(HERE / "recon_18runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); [w.writerow(r) for r in rr]
nc = sum(r["collapsed"] for r in rr)
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return c - h, c + h
def cp(k, n):  # exact Clopper-Pearson via binomial cdf bisection
    from math import comb
    def cdf(kk, p): return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))
    lo = 0.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(60):
            mid = (a + b) / 2
            if 1 - cdf(k - 1, mid) > 0.025: b = mid
            else: a = mid
        lo = b
    a, b = 0.0, 1.0
    for _ in range(60):
        mid = (a + b) / 2
        if cdf(k, mid) > 0.025: a = mid
        else: b = mid
    return lo, a
out["collapse_freq"] = {"k": nc, "n": len(rr), "wilson95": wilson(nc, len(rr)), "clopper_pearson95": cp(nc, len(rr)),
                        "k_in_003": nc, "n_003": 12, "wilson95_003": wilson(nc, 12), "note": "operational log values (epoch_logs.jsonl), not results.json. exp001 runs (3) not in the 18: only last-epoch train metrics exist; 18 = 003 (12) + 002 (6)"}
# detectability via total loss: leave-one-out z of final val_loss vs same-dist peers, and pooled-across-dist threshold
det = []
for r in rr:
    if r["exp"] != "003": continue
    peers = [p["val_loss_final"] for p in rr if p["dist"] == r["dist"] and p["run"] != r["run"]]
    det.append({"run": r["run"], "val_loss_final": r["val_loss_final"], "peer_mean": float(np.mean(peers)), "peer_sd": float(np.std(peers, ddof=1)), "rel_excess": float(r["val_loss_final"] / np.mean(peers) - 1), "z_vs_peers": float((r["val_loss_final"] - np.mean(peers)) / np.std(peers, ddof=1)), "collapsed": r["collapsed"]})
out["valloss_detect_same_dist"] = det
allv = np.array([r["val_loss_final"] for r in rr]); isc = np.array([r["collapsed"] for r in rr])
# pooled across distributions: rank / AUC of val_loss for collapsed vs rest
auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in allv[isc] for b in allv[~isc]]))
out["valloss_pooled"] = {"auc_collapsed_higher_val_loss": auc, "collapsed_vals": allv[isc].tolist(), "noncollapsed_range": [float(allv[~isc].min()), float(allv[~isc].max())], "n_noncollapsed_above_a026_s1": int((allv[~isc] > allv[isc].min()).sum())}
# epoch-wise relative jump in val_loss at collapse epoch vs normal
for n in ["exp003_mix_adjacent_s0", "exp003_single_a026_s1"]:
    x = dr[n]["by_epoch"]; ds = n.rsplit("_s", 1)[0]
    peers = [p for p in dr if p.rsplit("_s", 1)[0] == ds and p != n]
    out.setdefault("collapse_epochwise", {})[n] = {e: {"val_loss": x[e]["val_loss"], "peer_val_loss_mean": float(np.mean([dr[p]["by_epoch"][e]["val_loss"] for p in peers])), "val_z_pred": x[e]["val_z_visual_err_pred"], "peer_z_pred": float(np.mean([dr[p]["by_epoch"][e]["val_z_visual_err_pred"] for p in peers])),
                                                 "recon": x[e]["val_img_mse_reconstructed"], "peer_recon": float(np.mean([dr[p]["by_epoch"][e]["val_img_mse_reconstructed"] for p in peers]))} for e in ["1", "2", "3", "4", "5", "15"]}
# normal runs: max epoch-to-epoch recon jump
nm = [r["recon_max_jump_ratio"] for r in rr if not r["collapsed"]]
out["normal_runs_recon_jump_ratio"] = {"max": max(nm), "median": float(np.median(nm))}
out["collapse_jump_ratios"] = {r["run"]: r["recon_max_jump_ratio"] for r in rr if r["collapsed"]}
print(json.dumps({k: out[k] for k in ["collapse_freq", "valloss_detect_same_dist", "valloss_pooled", "collapse_epochwise", "normal_runs_recon_jump_ratio", "collapse_jump_ratios"]}, indent=1, default=str))

# ---------- 5. resume / baseline comparability ----------
cmp_ = {}
for d in ["single_a026", "single_a248"]:
    cmp_[d] = {"slopes": {s: runs["exp003_%s_s%d" % (d, s)]["slope_ol"]["all_sizes"]["slope"] for s in range(3)}}
real003 = np.array([runs["exp003_mix_wide_s0"]["by_size"][g]["area_real_mean"] for g in runs["exp003_mix_wide_s0"]["groups_in_order"]])
bgs = B["runs"]["exp001_cs_baseline_single_s0"]["diagnose"]["size_curve"]["by_group"]; real001 = np.array([bgs[g]["k12"]["real"] for g in bgs])
cmp_["real_k12_003_vs_001_rel_diff"] = ((real003 - real001) / real001).round(4).tolist()
cmp_["baseline_train_loss_final_001"] = [B["runs"][n]["train"]["last_epoch_log"]["train_loss"] for n in base]
cmp_["mix_wide_train_loss_final_003"] = [dr["exp003_mix_wide_s%d" % i]["by_epoch"]["15"]["train_loss"] for i in range(3)]
cmp_["baseline_recon_final_001"] = [B["runs"][n]["train"]["last_epoch_log"]["train_img_mse_reconstructed"] for n in base]
cmp_["train_env_mix_wide"] = [runs["exp003_mix_wide_s%d" % i]["train_env"] for i in range(3)]
out["comparability"] = cmp_
print(json.dumps(cmp_, indent=1, default=str))

json.dump(out, open(HERE / "stats_review_output.json", "w"), indent=1, default=str)

# ---------- figures ----------
