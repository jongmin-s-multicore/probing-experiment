"""Independent recomputation for the stats review of 002. stdlib only. Reads ../results.json. Prints everything used in notes/reviews/002_stats.md."""
import json, math, itertools, os
R = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results.json")))
DT = {r["id"]: r for r in R["decision_table"]}
mean = lambda x: sum(x) / len(x)
def sd(x):
    m = mean(x); return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
def tcdf(t, df):  # numeric integration of Student t pdf
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    a, n = 0.0, 20000; h = abs(t) / n
    s = f(0) + f(abs(t)) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))
    area = s * h / 3
    return 0.5 + math.copysign(area, t)
def welch(a, b):
    ma, mb, va, vb = mean(a), mean(b), sd(a) ** 2, sd(b) ** 2
    se = math.sqrt(va / len(a) + vb / len(b)); t = (ma - mb) / se
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    return t, df, 2 * (1 - tcdf(abs(t), df)), se
def holm(p, m=None):
    m = len(p); o = sorted(range(m), key=lambda i: p[i]); adj = [0] * m; run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj
T975_2 = 4.302652729911275  # t_{0.975, df=2}

print("=== Holm recompute")
for fam, ids in {"H_a": ["H_a:interp:linear_pooled", "H_a:interp:attentive", "H_a:extrap:linear_pooled", "H_a:extrap:attentive"],
                 "H_b": ["H_b:last_block", "H_b:last_rollout_step"],
                 "H_c": ["H_c:steering", "H_c:subspace_removal"],
                 "cond": ["cond_a_vs_b:last_block", "cond_a_vs_b:last_rollout_step"]}.items():
    p = [1.0 if DT[i]["p_value"] is None else DT[i]["p_value"] for i in ids]
    a = holm(p)
    for i, pp, aa in zip(ids, p, a):
        print(fam, i, "p=%.6f" % pp, "holm_recomputed=%.6f" % aa, "stored=%s" % DT[i]["holm_adjusted_p"], "match=%s" % (abs(aa - DT[i]["holm_adjusted_p"]) < 1e-12))
print("H_c steering if m=1 (not_evaluable dropped):", DT["H_c:steering"]["p_value"])

print("\n=== H_a recompute of status")
for i in ["H_a:interp:linear_pooled", "H_a:interp:attentive", "H_a:extrap:linear_pooled", "H_a:extrap:attentive"]:
    r = DT[i]["result"]
    ok = r["delta_vs_mask_baseline"] >= 0.1 and r["delta_vs_mask_baseline_ci95"][0] > 0 and r["delta_vs_random_init_ci95"][0] > 0
    pm = mean(r["r2_real_per_probe_seed"])
    print(i, "dmask=%.4f CI=%s  drand=%.4f CI=%s  crit=%s  stored=%s  mean(per_probe_seed)=%.6f vs point %.6f" % (
        r["delta_vs_mask_baseline"], [round(x, 4) for x in r["delta_vs_mask_baseline_ci95"]], r["delta_vs_random_init"], [round(x, 4) for x in r["delta_vs_random_init_ci95"]],
        ok, DT[i]["status"], pm, r["r2_real"]),
        " probe-seed SD real=%.4f rand=%.4f mask=%.4f" % (sd(r["r2_real_per_probe_seed"]), sd(r["r2_random_init_per_probe_seed"]), sd(r["r2_mask_baseline_per_probe_seed"])),
        " margin-0.1 gap: CI_lo-0.1=%.4f" % (r["delta_vs_mask_baseline_ci95"][0] - 0.1))
    ci = r["delta_vs_mask_baseline_ci95"]; print("   CI half-width=%.4f  point in CI=%s ; mean-of-delta check delta=real-mask: %.6f" % ((ci[1]-ci[0])/2, ci[0] <= r["delta_vs_mask_baseline"] <= ci[1], r["r2_real"]-r["r2_mask_baseline"]))
    # bootstrap p vs CI: p<=0.025 iff lower>0 roughly
print("p-values bootstrap floor 1/1001 =", 1/1001, " rank floor 1/101 =", 1/101)

print("\n=== Q1 late-window table (point, CI) incl. MAE, Pearson, pair-order")
c = R["q1"]["cells"]
for role in ["interp", "extrap", "unseen", "trained"]:
    for kind in ["linear_pooled", "attentive"]:
        for rep in ["dino", "rand", "shuffle"]:
            k = "log_nominal_area_ratio|%s|%s|all|late" % (kind, rep)
            if k in c:
                e = c[k][role]
                print(role, kind, rep, "R2 %.4f %s | MAE %.4f %s | r %.4f" % (e["r2"]["point"], [round(x, 3) if x is not None else None for x in e["r2"]["ci95_episode_bootstrap"]], e["mae"]["point"], [round(x, 4) for x in e["mae"]["ci95_episode_bootstrap"]], e["pearson"]["point"]))
    k = "log_nominal_area_ratio|mask_ridge|mask|all|late"; e = c[k][role]
    print(role, "mask", "R2 %.4f | MAE %.4f | r %s" % (e["r2"]["point"], e["mae"]["point"], e["pearson"]["point"]))
print("--- early window (report only)")
for role in ["interp", "extrap"]:
    for kind in ["linear_pooled", "attentive"]:
        for rep in ["dino", "rand", "shuffle"]:
            k = "log_nominal_area_ratio|%s|%s|all|early" % (kind, rep); e = c[k][role]
            print(role, kind, rep, "R2 %.4f MAE %.4f" % (e["r2"]["point"], e["mae"]["point"]))
    e = c["log_nominal_area_ratio|mask_ridge|mask|all|early"][role]; print(role, "mask R2 %.4f MAE %.4f" % (e["r2"]["point"], e["mae"]["point"]))
print("--- target SD implied: SST scale. levels log area:")
ar = {"a026": .259, "a040": .399, "a080": .801, "a125": 1.249, "a200": 2.0, "a248": 2.479}
for name, lv in {"interp": ["a080", "a125"], "extrap": ["a026", "a040", "a200", "a248"]}.items():
    y = [math.log(ar[l]) for l in lv]; print(name, "SD of log area (equal weights, 100 eps/level) = %.4f" % math.sqrt(sum((v - mean(y)) ** 2 for v in y) / len(y)))

print("\n=== H_b recompute")
for i in ["H_b:last_block", "H_b:last_rollout_step"]:
    r = DT[i]["result"]; d = [v["drop"] for v in r["per_training_seed"].values()]
    q2 = [v["r2_q2"] for v in r["per_training_seed"].values()]
    print(i, "drops", [round(x, 4) for x in d], "mean %.5f (stored %.5f) SD %.5f  t-CI95 over 3 seeds [%.4f, %.4f]  min/max vs 0.2: %.4f/%.4f" % (
        mean(d), r["drop_mean_over_seeds"], sd(d), mean(d) - T975_2 * sd(d) / math.sqrt(3), mean(d) + T975_2 * sd(d) / math.sqrt(3), min(d), max(d)),
        " episode-boot CI", [round(x, 4) for x in r["drop_ci95_episode_bootstrap"]], " q2 R2 SD over seeds %.4f" % sd(q2))
    t = (mean(d) - 0.2) / (sd(d) / math.sqrt(3)); print("   one-sample t vs 0.2 across seeds: t=%.2f df=2 two-sided p=%.4f one-sided p=%.4f" % (t, 2*(1-tcdf(abs(t), 2)), 1-tcdf(abs(t), 2)))

print("\n=== cond (a) vs (b)")
ra = DT["cond_a_vs_b:last_rollout_step"]["result"]
a = [v["drop"] for v in ra["per_training_seed_a"].values()]; b = [v["drop"] for v in ra["per_training_seed_b"].values()]
t, df, p, se = welch(a, b); print("rollout drops a", [round(x, 4) for x in a], "b", [round(x, 4) for x in b], "diff %.4f (stored %.4f) welch t=%.2f df=%.2f p=%.5f se=%.4f" % (mean(a)-mean(b), ra["drop_a_minus_drop_b"], t, df, p, se))
dfw = df
# t quantile via bisection
def tq(q, df):
    lo, hi = 0, 50
    for _ in range(80):
        mid = (lo + hi) / 2
        if tcdf(mid, df) < q: lo = mid
        else: hi = mid
    return lo
q = tq(0.975, df); print("   Welch CI95 for diff: [%.3f, %.3f] (t_crit %.3f) vs episode-boot CI %s" % (mean(a)-mean(b)-q*se, mean(a)-mean(b)+q*se, q, [round(x,3) for x in ra["ci95_episode_bootstrap"]]))
print("   complete separation: min(a)=%.3f > max(b)=%.3f ; exact permutation two-sided p (3 vs 3) = %.3f" % (min(a), max(b), 2/20))
rb = DT["cond_a_vs_b:last_block"]["result"]
a = [v["change_vs_layer0"] for v in rb["per_training_seed_a"].values()]; b = [v["change_vs_layer0"] for v in rb["per_training_seed_b"].values()]
t, df, p, se = welch(a, b); q = tq(0.975, df)
print("last_block change_vs_layer0 a", [round(x, 5) for x in a], "b", [round(x, 5) for x in b], "diff %.5f (stored %.5f) welch t=%.2f df=%.2f p=%.4f ; Welch CI [%.4f, %.4f] vs episode-boot %s" % (mean(a)-mean(b), rb["change_vs_layer0_a_minus_b"], t, df, p, mean(a)-mean(b)-q*se, mean(a)-mean(b)+q*se, [round(x,4) for x in rb["ci95_episode_bootstrap"]]))
print("   SD a %.5f, SD b %.5f; seed ranges overlap: a[%.4f,%.4f] b[%.4f,%.4f]" % (sd(a), sd(b), min(a), max(a), min(b), max(b)))
print("   unadjusted_two_sided_p stored", rb["unadjusted_two_sided_p"])

print("\n=== q2 layerwise (b): R2 trajectories 'unseen' per seed, SD over seeds by layer")
lw = R["q2_layerwise"]["conditions"]["multi_60_ind_160"]["per_seed"]
for L in range(7):
    v = [lw[s][str(L)]["unseen"]["r2"] for s in "012"]; ch = [lw[s][str(L)]["unseen"]["change_vs_layer0"] for s in "012"]
    print("layer", L, "R2 unseen", [round(x, 4) for x in v], "SD %.4f" % sd(v), " change", [round(x, 4) for x in ch])
la = R["q2_layerwise"]["conditions"]["single_ind"]["per_seed"]
for L in range(7):
    ch = [la[s][str(L)]["unseen"]["change_vs_layer0"] for s in "012"]
    print("(a) layer", L, "unseen change", [round(x, 4) for x in ch], "mean %.4f" % mean(ch), " trained", [round(la[s][str(L)]["trained"]["change_vs_layer0"], 4) for s in "012"])

print("\n=== H_c")
for i in ["H_c:steering", "H_c:subspace_removal"]:
    cells = DT[i]["result"]["cells"]
    for k, v in cells.items():
        print(i, k, "T=%.4f p95rand=%.4f ratio=%.1f spearman=%.4f p_rank=%.4f" % (v["T_size_direction"], v["random_percentile_value"], v["T_size_direction"]/v["random_percentile_value"], v["spearman"], v["p_rank"]))
    ts = [v["T_size_direction"] for v in cells.values()]; print("  T mean %.3f SD %.3f ; CV across 6 cells %.2f" % (mean(ts), sd(ts), sd(ts)/mean(ts)))
