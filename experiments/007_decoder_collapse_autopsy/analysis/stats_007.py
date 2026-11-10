"""007 stats review. Reads ../results.json and 003 analysis files; writes tables/figs into this dir.
run: ~/miniconda3/envs/lab05/bin/python stats_007.py"""
import json, math, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "../results.json")))
CK = R["ckpts"]; BD = R["B_delta"]
A3 = os.path.join(HERE, "../../003_train_dist_dependence/analysis")
REC = json.load(open(os.path.join(A3, "decoder_recon_by_epoch.json")))["runs"]
OUT = {}
def w(name, rows, header):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)

COL = ["exp003_mix_adjacent_s0", "exp003_single_a026_s1"]
CTL = ["exp003_mix_adjacent_s1", "exp003_mix_adjacent_s2", "exp003_single_a026_s2"]
def layers(run, ep, key="common"):
    return {l["name"]: l for l in CK[f"{run}/model_{ep}"]["D"][key]["layers"]}
names = [l["name"] for l in CK[f"{COL[0]}/model_3"]["D"]["common"]["layers"]]

# ---------- 1. criterion 2 recompute + threshold sensitivity ----------
def ctl_mean_std(ep, name): return np.mean([layers(c, ep)[name]["across_input_std"] for c in CTL])
sens = []
thr_list = [1/3, 1/10, 1/100, 1/1000, 1e-4, 1e-5]
for run, ep in [(COL[0], 3), (COL[1], 3), (COL[1], 4), (COL[0], 2), (COL[1], 2)]:
    L = layers(run, ep); cep = 3 if ep >= 3 else 2
    ratios = [L[n]["across_input_std"] / ctl_mean_std(cep, n) for n in names]
    row = [f"{run}/model_{ep}"]
    for t in thr_list:
        lost = [n for n, r in zip(names, ratios) if r < t]
        row.append(f"{lost[0] if lost else 'none'}({len(lost)})")
    sens.append(row)
w("sens_criterion2.csv", sens, ["ckpt"] + [f"<{t:g}" for t in thr_list])
# std normalised by rms (scale-free) too
sens2 = []
for run, ep in [(COL[0], 3), (COL[1], 3)]:
    L = layers(run, ep)
    for n in names:
        cr = np.mean([layers(c, 3)[n]["across_input_std_over_rms"] for c in CTL])
        sens2.append([run, n, L[n]["across_input_std_over_rms"], cr, L[n]["across_input_std_over_rms"] / cr])
w("layer_std_over_rms.csv", sens2, ["run", "layer", "std_over_rms", "ctl_mean", "ratio"])

# ---------- 2. per-layer table vs control distribution (epoch 3) ----------
rows = []; zs = {}
for n in names:
    typ = layers(COL[0], 3)[n]["type"]
    cs = np.array([layers(c, 3)[n]["across_input_std"] for c in CTL])
    cr = np.array([layers(c, 3)[n]["rms"] for c in CTL])
    row = [n, typ, *np.round(cs, 5), *np.round(cr, 4)]
    for r in COL:
        L = layers(r, 3)[n]
        row += [L["across_input_std"], L["rms"], L["across_input_std"] / cs.mean(), L["rms"] / cr.mean(),
                int(L["rms"] > cr.max() or L["rms"] < cr.min()),
                L.get("frac_input_le0", "")]
    cf = [layers(c, 3)[n].get("frac_input_le0", "") for c in CTL]
    row += cf
    rows.append(row)
hdr = ["layer", "type", "ctl_std_1", "ctl_std_2", "ctl_std_3", "ctl_rms_1", "ctl_rms_2", "ctl_rms_3"]
for r in ["mixS0", "singleS1"]:
    hdr += [f"{r}_std", f"{r}_rms", f"{r}_std_ratio", f"{r}_rms_ratio", f"{r}_outside_ctl_rms_range", f"{r}_frac_le0"]
hdr += ["ctl_le0_1", "ctl_le0_2", "ctl_le0_3"]
w("layer_table_ep3.csv", rows, hdr)
# how many layers outside control range (n=3 controls: chance 2/4 = 50% per layer)
nout = {r: sum(1 for row in rows if row[hdr.index(f"{ 'mixS0' if r==COL[0] else 'singleS1'}_outside_ctl_rms_range")]) for r in COL}
OUT["layers_total"] = len(names); OUT["layers_rms_outside_ctl_range"] = nout
# rms ratio at ep2 (is scale blow-up already present before collapse?)
sc = []
for n in ["upsample_b.blocks.0", "upsample_b.blocks.4", "upsample_b.blocks.6", "dec.blocks.0", "dec.blocks.8"]:
    r2 = [layers(r, 2)[n]["rms"] for r in COL]; c2 = np.mean([layers(c, 2)[n]["rms"] for c in CTL])
    r3 = [layers(r, 3)[n]["rms"] for r in COL]; c3 = np.mean([layers(c, 3)[n]["rms"] for c in CTL])
    sc.append([n, *np.round(np.array(r2) / c2, 3), *np.round(np.array(r3) / c3, 2)])
w("rms_ratio_ep2_ep3.csv", sc, ["layer", "mixS0_ep2/ctl", "singleS1_ep2/ctl", "mixS0_ep3/ctl", "singleS1_ep3/ctl"])
OUT["rms_ratio"] = sc

# ---------- criterion 3 sensitivity ----------
c3 = []
for run, ep in [(COL[0], 3), (COL[1], 3), (COL[1], 4)]:
    L = layers(run, ep)["upsample_b.blocks.7"]
    n_ch = 192; alive_ch = round((1 - L["dead_channel_frac"]) * n_ch)
    n_el = 54 * 192 * 28 * 28  # shape asserted below
    shp = L["shape"]; n_el = 54 * int(np.prod(shp))
    c3.append([f"{run}/model_{ep}", shp, L["frac_input_le0"], alive_ch, round((1 - L["frac_input_le0"]) * n_el), n_el,
               L["across_input_std"], L["rms"]])
w("criterion3_counts.csv", c3, ["ckpt", "shape", "frac_le0", "alive_channels_of_192", "n_positive_elements", "n_elements", "std", "rms"])
OUT["crit3"] = c3

# ---------- criterion 4: null from controls (leave-one-out) ----------
pn = list(BD[f"{COL[0]}/2->3"]["decoder"].keys())
RC = {k: np.array([BD[k]["decoder"][p]["rel_change"] for p in pn]) for k in BD}
ctlk = ["exp003_mix_adjacent_s1/2->3", "exp003_mix_adjacent_s2/2->3", "exp003_single_a026_s2/2->3"]
rows = []
for k in ctlk:
    others = np.max([RC[o] for o in ctlk if o != k], axis=0)
    rat = RC[k] / others
    rows.append([k, "LOO control", rat.max(), pn[int(rat.argmax())], int((rat > 1).sum()), int((rat > 2).sum())])
for k in ["exp003_mix_adjacent_s0/2->3", "exp003_single_a026_s1/2->3"]:
    rat = RC[k] / np.max([RC[o] for o in ctlk], axis=0)
    rows.append([k, "collapsed vs 3 ctl max", rat.max(), pn[int(rat.argmax())], int((rat > 1).sum()), int((rat > 2).sum())])
    sd = {"exp003_mix_adjacent_s0/2->3": ctlk[:2], "exp003_single_a026_s1/2->3": ctlk[2:]}[k]
    rat = RC[k] / np.max([RC[o] for o in sd], axis=0)
    rows.append([k, "collapsed vs same-dist ctl max", rat.max(), pn[int(rat.argmax())], int((rat > 1).sum()), int((rat > 2).sum())])
w("crit4_null.csv", rows, ["run", "comparison", "max_ratio", "param", "n_params_ratio_gt1", "n_params_ratio_gt2"])
OUT["crit4"] = rows; OUT["n_decoder_params"] = len(pn)
# sensitivity of "10x" threshold: ratio vs threshold
k = "exp003_mix_adjacent_s0/2->3"; rat = RC[k] / np.max([RC[o] for o in ctlk], axis=0)
OUT["crit4_ratio_quantiles_mix"] = {q: float(np.quantile(rat, q)) for q in [0.1, .5, .9, 1.0]}
k2 = "exp003_single_a026_s1/2->3"; rat2 = RC[k2] / np.max([RC[o] for o in ctlk], axis=0)
OUT["crit4_ratio_quantiles_single"] = {q: float(np.quantile(rat2, q)) for q in [0.1, .5, .9, 1.0]}
OUT["crit4_param_ratio_by_name"] = {p: [float(a), float(b)] for p, a, b in zip(pn, rat, rat2) if max(a, b) > 2}
# 3-control-max exceedance expectation under exchangeability = 1/4
OUT["exceed_expected_by_chance"] = len(pn) / 4

# ---------- 3. Adam constraints ----------
b1, b2 = 0.9, 0.999
dec = lambda r, e: CK[f"{r}/model_{e}"]["C_optimizer"]["decoder_optimizer"]["per_param"]
adam = []
for r in COL:
    for p in ["upsample_b.blocks.6.weight", "upsample_b.blocks.0.weight", "upsample_b.blocks.8.weight", "upsample_b.blocks.6.bias"]:
        v3 = dec(r, 3)[p]["exp_avg_sq"]["mean"]; v2 = dec(r, 2)[p]["exp_avg_sq"]["mean"]
        vc3 = np.mean([dec(c, 3)[p]["exp_avg_sq"]["mean"] for c in CTL]); vc2 = np.mean([dec(c, 2)[p]["exp_avg_sq"]["mean"] for c in CTL])
        adam.append([r, p, v2, vc2, v2 / vc2, v3, vc3, v3 / vc3, v3 / v2])
w("adam_v_ratio.csv", adam, ["run", "param", "v_ep2", "ctl_v_ep2", "ratio_ep2", "v_ep3", "ctl_v_ep3", "ratio_ep3", "v3_over_v2"])
OUT["adam_v"] = adam
# all-param ratio table at ep3
allr = []
for r in COL:
    for p in dec(r, 3):
        v3 = dec(r, 3)[p]["exp_avg_sq"]["mean"]; vc = np.mean([dec(c, 3)[p]["exp_avg_sq"]["mean"] for c in CTL])
        allr.append([r, p, v3 / vc])
w("adam_v_ratio_allparams.csv", allr, ["run", "param", "ratio_ep3"])
# single_a026_s1: v4/v3 under zero-gradient hypothesis 0.999^4125
OUT["decay_4125"] = b2 ** 4125
OUT["v4_over_v3_single"] = {p: dec(COL[1], 4)[p]["exp_avg_sq"]["mean"] / dec(COL[1], 3)[p]["exp_avg_sq"]["mean"] for p in
                            ["upsample_b.blocks.6.weight", "upsample_b.blocks.0.weight", "upsample_b.blocks.8.weight", "dec.blocks.8.bias"]}
# implied v at death vs n steps since death, ONLY valid if upstream grad == 0 since death (s0)
p = "upsample_b.blocks.6.weight"
v_now = dec(COL[0], 3)[p]["exp_avg_sq"]["mean"]; v_ctl = np.mean([dec(c, 3)[p]["exp_avg_sq"]["mean"] for c in CTL])
tab = []
for n in [1, 100, 855, 1000, 1732, 2500, 4125]:
    d = b2 ** n; vd = v_now / d
    g1 = math.sqrt(vd / (1 - b2))                      # one-step spike rms-per-element gradient
    g100 = math.sqrt(vd / (1 - b2 ** 100))             # sustained 100-step rms
    tab.append([n, d, vd, vd / v_ctl, g1, g1 / math.sqrt(v_ctl), g100, g100 / math.sqrt(v_ctl)])
w("adam_burst_inversion.csv", tab, ["n_steps_since_zero_grad", "0.999^n", "v_at_death_equiv", "v_at_death/ctl_v",
                                    "single_step_grad_rms", "x_ctl_rms_grad", "100step_sustained_rms", "x_ctl_rms_grad"])
OUT["adam_tab"] = tab; OUT["v_now"] = v_now; OUT["v_ctl"] = v_ctl; OUT["ctl_rms_grad"] = math.sqrt(v_ctl)
# exp_avg floor argument: float32 denormal unit
u = np.float32(1.4012984643e-45)
OUT["denorm_floor"] = {}
for k in [1, 2, 3, 4, 5, 6]:
    x = np.float32(k) * u
    OUT["denorm_floor"][k] = int(round(float(np.float32(x * np.float32(0.9)) / float(u))))
# steps for m to decay from m0 to floor (5 units ~ 7e-45)
OUT["m_decay_steps"] = {f"{m0:g}": math.log(m0 / 7.0e-45) / -math.log(b1) for m0 in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}
# which params at floor in s0 ep3
fl = [q for q, v in dec(COL[0], 3).items() if v["exp_avg"]["absmax"] < 1e-44]
OUT["s0_params_at_floor"] = [len(fl), [q for q in dec(COL[0], 3) if q not in fl]]
OUT["s0_floor_absmax_in_units"] = {q: dec(COL[0], 3)[q]["exp_avg"]["absmax"] / 1.4012984643e-45 for q in fl[:3]}
# gradient-liveness evidence in the single run: exp_avg absmax ratio
OUT["single_m_absmax"] = {q: dec(COL[1], 3)[q]["exp_avg"]["absmax"] for q in ["upsample_b.blocks.6.weight", "upsample_b.blocks.0.weight"]}

# Outside-checkpoint timing constraint: epoch mean train_loss (includes decoder recon loss, weight ~1; see val_loss diff)
def be(r, e, k): return REC[r]["by_epoch"][str(e)][k]
tl = []
base = {COL[0]: ["exp003_mix_adjacent_s1", "exp003_mix_adjacent_s2"], COL[1]: ["exp003_single_a026_s0", "exp003_single_a026_s2"]}
for r in COL:
    for e in [2, 3, 4, 5]:
        b = np.mean([be(c, e, "train_loss") for c in base[r]]); bv = np.mean([be(c, e, "val_loss") for c in base[r]])
        tl.append([r, e, be(r, e, "train_loss"), b, be(r, e, "train_loss") - b, be(r, e, "val_loss"), bv, be(r, e, "val_loss") - bv,
                   be(r, e, "val_z_visual_err_pred"), np.mean([be(c, e, "val_z_visual_err_pred") for c in base[r]])])
w("epoch_loss_gap.csv", tl, ["run", "epoch", "train_loss", "base_train_loss", "d_train", "val_loss", "base_val_loss", "d_val", "zerr", "base_zerr"])
OUT["epoch_gap"] = tl
est = {}
for r in COL:
    d3 = be(r, 3, "train_loss") - np.mean([be(c, 3, "train_loss") for c in base[r]])
    d4 = be(r, 4, "train_loss") - np.mean([be(c, 4, "train_loss") for c in base[r]])
    dv3 = be(r, 3, "val_loss") - np.mean([be(c, 3, "val_loss") for c in base[r]])
    # decoder-only steady gap from train_img_mse at epoch 5 (first-batch) minus control
    dec_gap = be(r, 5, "train_img_mse_reconstructed") - np.mean([be(c, 5, "train_img_mse_reconstructed") for c in base[r]])
    # also pred-side gap: val_z at ep3 (seed noise for z part)
    seedsd = np.std([be(c, 3, "train_loss") for c in base[r]], ddof=1)
    est[r] = dict(d_train3=d3, d_train4=d4, d_val3=dv3, dec_gap_firstbatch_ep5=dec_gap, seed_sd_train3_two_ctl=seedsd,
                  f_if_all_decoder=d3 / dec_gap, f_if_predictor_gap_equals_ep4_resid=(d3 - (d4 - dec_gap)) / dec_gap)
OUT["timing_est"] = est

# ---------- 4. predictor ----------
pr = []
for k in CK:
    E = CK[k]["E"]["common"]
    pr.append([k, CK[k]["group"], E["out_across_input_std"], E["in_across_input_std"], E["ratio_out_over_in"], E["out_rms"]])
w("predictor_E.csv", pr, ["ckpt", "group", "out_std", "in_std", "out/in", "out_rms"])
OUT["pred_E"] = pr
csvr = {r["run"]: r for r in csv.DictReader(open(os.path.join(A3, "recon_18runs.csv")))}
byd = {}
for run, r in csvr.items():
    byd.setdefault(r["dist"], []).append((run, float(r["val_z_pred_final"]), float(r["val_z_rollout_final"]), r["collapsed"] == "True", float(r["val_loss_final"])))
zt = []
for d, L in byd.items():
    for run, z, zr, c, vl in L:
        zt.append([d, run, z, zr, c])
w("zpred_by_dist.csv", zt, ["dist", "run", "val_z_pred_final", "val_z_rollout_final", "collapsed"])
OUT["zpred_dist"] = {d: [(r[0].split("exp003_")[-1], round(r[1], 4), r[3]) for r in L] for d, L in byd.items()}
# within-dist relative deviation, collapsed vs mean of others
rel = {}
for d, L in byd.items():
    for run, z, zr, c, vl in L:
        oth = [x[1] for x in L if x[0] != run]
        othr = [x[2] for x in L if x[0] != run]
        rel[run] = (z / np.mean(oth) - 1, zr / np.mean(othr) - 1, c)
OUT["z_rel_dev"] = {k: v for k, v in rel.items()}
allrel = np.array([v[0] for k, v in rel.items() if not v[2]])
OUT["z_rel_dev_noncollapsed_sd"] = float(allrel.std(ddof=1)); OUT["z_rel_dev_noncollapsed_abs_max"] = float(np.abs(allrel).max())
OUT["z_rel_dev_noncollapsed_n"] = int(len(allrel))
# per-dist relative sd of the final z err among noncollapsed
OUT["z_cv_by_dist"] = {d: float(np.std([x[1] for x in L if not x[3]], ddof=1) / np.mean([x[1] for x in L if not x[3]])) for d, L in byd.items() if sum(1 for x in L if not x[3]) >= 2}
# epoch-3 z err compare
zc = {r: [be(r, e, "val_z_visual_err_pred") for e in (3, 4, 8, 15)] for r in COL + CTL + ["exp003_single_a026_s0"]}
for r in zc: pass
OUT["zerr_epochs_3_4_8_15"] = zc if all(len(v) == 4 for v in zc.values()) else None

# ---------- 5. forecast ----------
from math import comb
k, n = 2, 18
def cp(k, n, a=0.05):
    from scipy import stats
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi
try:
    lo, hi = cp(k, n)
except Exception as e:
    # no scipy: bisection
    def cdf(p, kk, nn): return sum(comb(nn, i) * p**i * (1 - p)**(nn - i) for i in range(kk + 1))
    def bis(f, a, b):
        for _ in range(100):
            m = (a + b) / 2
            if f(m) > 0: a = m
            else: b = m
        return (a + b) / 2
    hi = bis(lambda p: cdf(p, k, n) - 0.025, 0, 1); lo = bis(lambda p: (1 - cdf(p, k - 1, n)) - 0.025, 1, 0) if False else None
    # lower: P(X>=k | p)=0.025
    a, b = 0.0, 1.0
    for _ in range(100):
        m = (a + b) / 2
        if 1 - cdf(m, k - 1, n) < 0.025: a = m
        else: b = m
    lo = (a + b) / 2
z = 1.96; ph = k / n
wil = ((ph + z*z/(2*n) - z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n))) / (1+z*z/n), (ph + z*z/(2*n) + z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n))) / (1+z*z/n))
OUT["ci"] = dict(p=ph, cp95=(lo, hi), wilson95=wil)
fc = []
for N in [3, 6, 9, 12, 18]:
    row = [N]
    for p in [ph, lo, hi, 3/20]:
        row.append(N * p)
    row.append(1 - (1 - ph) ** N)
    row.append(1 - (1 - lo) ** N); row.append(1 - (1 - hi) ** N)
    # beta-binomial posterior predictive with Beta(1,1): P(0 collapse)
    a, b = 1 + k, 1 + n - k
    p0 = 1.0
    for i in range(N): p0 *= (b + i) / (a + b + i)
    row.append(1 - p0)
    fc.append(row)
w("collapse_forecast.csv", fc, ["N_runs", "E[n]_p=2/18", "E[n]_p_lo", "E[n]_p_hi", "E[n]_Jeffreys-ish 3/20", "P(>=1)_pt", "P(>=1)_lo", "P(>=1)_hi", "P(>=1)_betabinom"])
OUT["forecast"] = fc
# P(0 collapse) with no collapse in 3 seeds etc. And dist-level heterogeneity
OUT["by_dist_collapsed"] = {d: f"{sum(1 for x in L if x[3])}/{len(L)}" for d, L in byd.items()}

# ---------- figures ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(len(names))
for r, c in zip(COL, ["tab:red", "tab:orange"]):
    ax[0].semilogy(x, [max(layers(r, 3)[n]["rms"], 1e-6) for n in names], c=c, label=r + " ep3")
for cn in CTL:
    ax[0].semilogy(x, [layers(cn, 3)[n]["rms"] for n in names], c="gray", alpha=.6, lw=1, label="ctl ep3" if cn == CTL[0] else None)
ax[0].axvline(names.index("upsample_b.blocks.7"), ls=":", c="k"); ax[0].set_title("layer output rms (epoch 3)"); ax[0].set_xlabel("layer index"); ax[0].legend(fontsize=7)
for r, c in zip(COL, ["tab:red", "tab:orange"]):
    ax[1].semilogy(x, [max(layers(r, 3)[n]["across_input_std"], 1e-9) / ctl_mean_std(3, n) for n in names], c=c, label=r)
for t in (1/100,): ax[1].axhline(t, ls="--", c="k")
ax[1].axhline(1, ls=":", c="gray"); ax[1].set_title("across-input std / ctl mean (epoch 3)"); ax[1].set_xlabel("layer index"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_layer_profile.png"), dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ns = np.arange(0, 4126); ax[0].semilogy(ns, b2 ** ns, label="v: 0.999^n"); ax[0].semilogy(ns, b1 ** ns, label="m: 0.9^n")
ax[0].set_ylim(1e-6, 1.2); ax[0].legend(); ax[0].set_xlabel("steps since gradient became 0"); ax[0].set_title("Adam decay")
t = np.array(tab); ax[1].loglog(t[:, 0], t[:, 5], "o-", label="1-step spike (x ctl rms grad)"); ax[1].loglog(t[:, 0], t[:, 7], "s-", label="100-step sustained")
ax[1].set_xlabel("n steps since death"); ax[1].set_title("implied gradient size, blocks.6.weight, mixS0"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_adam.png"), dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(5, 4)); Ns = np.arange(1, 19)
for p, lab in [(ph, "p=2/18"), (lo, "lo"), (hi, "hi")]: ax.plot(Ns, 1 - (1 - p) ** Ns, label=lab)
ax.set_xlabel("N runs"); ax.set_ylabel("P(>=1 collapse)"); ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_forecast.png"), dpi=130)
json.dump(OUT, open(os.path.join(HERE, "stats_007_out.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print(json.dumps(OUT, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))[:12000])

# ---------- 3b. per-element movement vs single-spike Adam bound ----------
lr = 3e-4
bound = lr * (1 - b1) / math.sqrt(1 - b2) / (1 - b1 / math.sqrt(b2))   # total path length of one dominating spike, per element
nmap = {p: CK[f"{COL[0]}/model_3"]["B_weights"]["decoder"][p]["n"] for p in pn}
mv = []
for k in BD:
    for p in pn:
        d = BD[k]["decoder"][p]["delta_norm"]
        mv.append([k, p, nmap[p], d / math.sqrt(nmap[p])])
w("per_element_rms_delta.csv", mv, ["pair", "param", "n", "rms_delta_per_element"])
mvd = {k: {p: d for kk, p, nn, d in mv if kk == k} for k in BD}
tabm = []
for k in BD:
    vals = np.array(list(mvd[k].values()))
    tabm.append([k, vals.max(), pn[int(vals.argmax())], int((vals > bound).sum()), float(np.median(vals))])
w("per_element_delta_summary.csv", tabm, ["pair", "max_rms_delta", "argmax", "n_params_gt_single_spike_bound", "median"])
OUT["spike_bound"] = bound; OUT["per_elem_summary"] = tabm
print("BOUND", bound); [print(r) for r in tabm]
json.dump(OUT, open(os.path.join(HERE, "stats_007_out.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))

# ---------- extras: weights vs bias ratios, predictor LOO, timing with val-gap ----------
ex = {}
for k in ["exp003_mix_adjacent_s0/2->3", "exp003_single_a026_s1/2->3"]:
    rat = RC[k] / np.max([RC[o] for o in ctlk], axis=0)
    isb = np.array([p.endswith("bias") for p in pn]); isu = np.array([p.startswith("upsample") for p in pn])
    ex[k] = {"median_ratio_weights": float(np.median(rat[~isb])), "median_ratio_bias": float(np.median(rat[isb])),
             "median_ratio_upsample": float(np.median(rat[isu])), "median_ratio_dec": float(np.median(rat[~isu])),
             "max_ratio_weights": float(rat[~isb].max()), "argmax_weights": pn[int(rat[~isb].argmax())],
             "rel_change_upsample_b.blocks.6.weight": RC[k][pn.index("upsample_b.blocks.6.weight")],
             "ctl_rel_change_blocks.6.weight": [float(RC[o][pn.index("upsample_b.blocks.6.weight")]) for o in ctlk]}
OUT["wb"] = ex
# predictor LOO
def prc(k): return np.array([BD[k]["predictor"][p]["rel_change"] for p in BD[k]["predictor"]])
pl = []
for k in ctlk:
    pl.append(float((prc(k) / np.max([prc(o) for o in ctlk if o != k], axis=0)).max()))
OUT["pred_LOO_max_ratio"] = pl
OUT["pred_collapsed_max_ratio"] = [float((prc(k) / np.max([prc(o) for o in ctlk], axis=0)).max()) for k in ["exp003_mix_adjacent_s0/2->3", "exp003_single_a026_s1/2->3"]]
# timing: use mean val gap at epochs 4,5 as steady decoder gap
t2 = {}
for r in COL:
    g = np.mean([be(r, e, "val_loss") - np.mean([be(c, e, "val_loss") for c in base[r]]) for e in (4, 5)])
    d3 = be(r, 3, "train_loss") - np.mean([be(c, 3, "train_loss") for c in base[r]])
    t2[r] = dict(steady_val_gap_ep45=g, d_train3=d3, f=d3 / g, onset_step=8250 + (1 - d3 / g) * 4125, steps_before_end=d3 / g * 4125)
OUT["timing_valgap"] = t2
# what adam says at those n
for r, d in t2.items():
    n = d["steps_before_end"]; vd = v_now / b2 ** n if r == COL[0] else None
    d["adam_equiv_single_spike_x_ctl"] = (math.sqrt(vd / (1 - b2)) / math.sqrt(v_ctl)) if vd else None
    d["adam_equiv_100step_x_ctl"] = (math.sqrt(vd / (1 - b2 ** 100)) / math.sqrt(v_ctl)) if vd else None
json.dump(OUT, open(os.path.join(HERE, "stats_007_out.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print(json.dumps({k: OUT[k] for k in ["wb", "pred_LOO_max_ratio", "pred_collapsed_max_ratio", "timing_valgap"]}, indent=1, default=float))
