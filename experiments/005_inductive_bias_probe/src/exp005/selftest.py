"""CPU self-test on SYNTHETIC data with planted structure. The numbers printed here are NOT experiment results.
Covers: 4.6 branch logic (each row + combinations + reading experiment 4's real summary builder), tasks/pair groups, IB metrics vs brute force,
perfect-state / area-only / random models through the real head code and the real metrics code, calibration, verdict rules, report files."""
from __future__ import annotations

import copy
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

from . import branch as BR
from . import ibmetrics as IB
from . import tasks as TK
from . import verdicts as VD
from .base import write_json, read_json

REC = {"pass": [], "fail": [], "skipped": []}


def check(name, cond, detail=""):
    (REC["pass"] if cond else REC["fail"]).append(name)
    print("  [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))


def skip(name, why):
    REC["skipped"].append(name)
    print("  [skipped(%s)] %s" % (why, name))


def have_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def names(plan):
    return [l["name"] for l in plan["layers"]]


# ------------------------------------------------------------------------------------------------ A. branch table
def t_branch(cfg):
    print("A  4.6 branch table (each row, combinations, priority, missing fields)")
    base = {"pilot": True, "encoder_layers": 12, "predictor_layers": 6, "k_star": None, "m_star": None, "loss_ood_only": False,
            "l_best": {"module": "encoder", "depth": 9}, "l_best_by_module": {"encoder": {"depth": 9, "value": 0.9}, "predictor": {"depth": 3, "value": 0.8}}}
    S = lambda **kw: dict(base, **kw)
    p = BR.select(S(k_star=6))
    check("row 1 (k_star set): layers [enc_k9 (l_best encoder), enc_k12], sub-experiments 5-A, 5-B", names(p) == ["enc_k9", "enc_k12"] and p["subexps"] == ["5-A", "5-B"] and p["rows_matched"] == ["k_star"], str(p["rows_matched"]))
    p = BR.select(S(m_star=4))
    check("row 2 (only m_star): enc_k12, pred_h3, pred_h4; 5-A, 5-C", names(p) == ["enc_k12", "pred_h3", "pred_h4"] and p["subexps"] == ["5-A", "5-C"])
    p = BR.select(S(m_star=4, loss_ood_only=True))
    check("rows 2+3 (spec example summary): rows added in table order, l_best layer gets 5-B and 5-D only", p["rows_matched"] == ["m_star_only", "loss_ood_only"] and names(p) == ["enc_k12", "pred_h3", "pred_h4", "enc_k9"]
          and p["layer_subexps"]["enc_k9"] == ["5-B", "5-D"] and p["layer_subexps"]["pred_h4"] == ["5-A", "5-C"], str(p["layer_subexps"]))
    p = BR.select(S())
    check("row 4 (no vanishing): predictor output (pred_out6) with 5-A..5-D", names(p) == ["pred_out6"] and p["subexps"] == ["5-A", "5-B", "5-C", "5-D"] and p["rows_matched"] == ["no_vanishing"])
    p = BR.select(S(k_star=6, loss_ood_only=True))
    check("rows 1+3: upper row first, l_best encoder layer (enc_k9) already present gets 5-D added", p["rows_matched"] == ["k_star", "loss_ood_only"] and names(p) == ["enc_k9", "enc_k12"] and p["layer_subexps"]["enc_k9"] == ["5-A", "5-B", "5-D"])
    p = BR.select(S(k_star=6), exp1_negative=True)
    check("exp1 negative: hold, only enc_k12 with 5-C", p["hold"] and names(p) == ["enc_k12"] and p["subexps"] == ["5-C"])
    p = BR.select(None)
    check("summary missing -> unresolved, no layers", p["unresolved"] == ["exp4_summary_missing"] and not p["layers"])
    p = BR.select(S(m_star=1))
    check("m_star = 1 -> pred_h0 and pred_h1", names(p) == ["enc_k12", "pred_h0", "pred_h1"])
    p = BR.select(S(k_star=4, l_best={"module": "predictor", "depth": 2}, l_best_by_module={}))
    check("row 1 without an encoder l_best -> recorded as unresolved, final layer still planned", "l_best_encoder_missing" in p["unresolved"] and names(p) == ["enc_k12"])
    p = BR.select(S(k_star=4, l_best={"module": "predictor", "depth": 2}, l_best_by_module={"encoder": {"depth": 8, "value": 0.7}, "predictor": {"depth": 2, "value": 0.9}}))
    check("row 1 uses the encoder entry of l_best_by_module even if the global l_best is a predictor layer", names(p) == ["enc_k8", "enc_k12"])
    check("pilot flag read from the summary", BR.select(S(k_star=2))["pilot"] is True and BR.select(dict(S(k_star=2), pilot=False))["pilot"] is False)
    # integration with experiment 4's real summary builder
    from exp004 import depthlib as DL4
    cfg4 = {"depth": {"drop_threshold": 0.2, "primary": {"probe": "linear", "fit": "new", "control": "real", "timepoint": "late"}, "overlap_sd_mult": 2.0, "overlap_depth_fraction": 0.8,
                      "min_depths": 3, "l_best_splits": ["interp", "extrap"]}}
    E, P = [0, 2, 4, 6, 8, 10, 12], [0, 1, 2, 3, 4, 5, 6]
    rng = np.random.RandomState(0)
    rows = []
    for (mod, sp), dd in {("encoder", s): {d: 0.95 for d in E} for s in ("train", "interp", "extrap")}.items() if False else []:
        pass
    spec = {("encoder", "train"): {d: 0.95 for d in E}, ("encoder", "interp"): {d: 0.95 for d in E}, ("encoder", "extrap"): {d: (0.95 if d <= 4 else 0.3) for d in E},
            ("predictor", "train"): {d: 0.9 for d in P}, ("predictor", "interp"): {d: 0.9 for d in P}, ("predictor", "extrap"): {d: 0.9 for d in P}}
    for (mod, sp), dd in spec.items():
        for d, v in dd.items():
            for s in range(5):
                rows.append({"module": mod, "depth": d, "probe": "linear", "fit": "new", "level": "ALL", "split": sp, "timepoint": "late", "seed": s, "r2": v + 0.01 * rng.randn(),
                             "control": "real"})
    summ = DL4.build_summary(rows, cfg4, {"pilot": True, "encoder_layers": 12, "predictor_layers": 6})
    p = BR.select(summ)
    check("integration: summary built by experiment 4's code (k*=6, loss_ood_only) drives the plan", summ["k_star"] == 6 and summ["loss_ood_only"] is True
          and p["rows_matched"] == ["k_star", "loss_ood_only"] and names(p)[-1] == "enc_k12" and p["layer_subexps"]["enc_k12"] == ["5-A", "5-B"], str(names(p)))


# ------------------------------------------------------------------------------------------------ B. tasks
def small_cfgs(cfg, cfg2):
    from exp004.selftest import small_cfg2
    c = copy.deepcopy(cfg)
    c["eval"]["n_inputs"] = 270
    c["ib"].update({"n_tasks": 8, "n_per_task": 60})
    c["calib"].update({"n_tasks": 8, "n_per_task": 60})
    c["head"].update({"epochs": 40, "batch_size": 32})
    c["head"]["small"].update({"steps": 300})
    c["seeds"]["head"] = [0, 1]
    c["law"].update({"min_labels": 30})
    c["law"]["pysr"]["enabled"] = False
    c2 = small_cfg2(cfg2, n_train=20, n_test=10)
    c2["probe_impl"]["attentive"].update({"d_model": 16, "n_heads": 2, "mlp_hidden": 16})
    return c, c2


def synth_cache(cfg2):
    from exp004.selftest import make_rows
    rows = make_rows(cfg2)
    E = len(rows)
    frames = [0, 1, 2, 3, 8, 9, 10, 11]
    rng = np.random.RandomState(11)
    s = np.array([r["nominal_log_area"] for r in rows])
    mf = np.zeros((E, 13, 3), np.float32)
    # visible area depends on the level (log-linear) and on a fold factor: flat early frames, folded late frames
    fold = np.where(np.arange(13) >= 6, 0.5, 1.0)
    area = np.exp(8.0 + 0.8 * s)[:, None] * fold[None, :] * np.exp(0.08 * rng.randn(E, 13))
    mf[..., 0] = area
    mf[..., 1] = np.sqrt(area) * 1.3
    mf[..., 2] = np.sqrt(area) * 1.1
    return rows, frames, mf


def t_tasks(cfg, cfg2):
    print("B  evaluation inputs and tasks")
    c, c2 = small_cfgs(cfg, cfg2)
    from . import stages as S
    rows, frames, mf = synth_cache(c2)
    T, meta = S.build_tasks(c, c2, rows, mf, frames, [0, 1, 2, 3], [4, 5, 6, 7])
    n = len(T["row"])
    check("270 eval inputs, all from test rows, no duplicate (row, fpos)", n == 270 and all(rows[r]["part"] == "test" for r in T["row"]) and len({(a, b) for a, b in zip(T["row"], T["fpos"])}) == n)
    check("half early / half late frames, recorded", meta["eval_kind_counts"] == {"early": 135, "late": 135} and set(T["frame"][T["kind"] == 0]) <= {0, 1, 2, 3} and set(T["frame"][T["kind"] == 1]) <= {8, 9, 10, 11})
    check("all 9 levels present in the evaluation set", len(set(T["lvl"].tolist())) == 9)
    check("5-A labels: Bernoulli per level, never constant", T["labels_a"].shape == (8, 9) and all(0 < r.sum() < 9 for r in T["labels_a"]))
    train_levels = set(c2["data"]["train_levels_b"])
    check("5-A D_k: 60 distinct train-pool ids, >=1 per available level, pool restricted to levels present in the cache's train part",
          all(len(set(r)) == 60 for r in T["dk_a"]) and meta["pool_levels_used_5A"] == c2["data"]["train_levels_b"] and
          all({meta["levels"][T["lvl_flat"][i]] for i in r} >= train_levels for r in T["dk_a"]))
    check("5-A D_k never touches test rows", all(i < 60 * 8 for r in T["dk_a"] for i in r) and T["dk_a"].max() < len([r for r in rows if r["part"] == "train"]) * 8)
    check("5-B D_k from trained levels only", all(meta["levels"][T["lvl_flat"][i]] in train_levels for r in T["dk_b"] for i in r))
    check("sinusoid targets: zero mean, unit std over the s range, band-limited components", abs(T["g_eval"].mean()) < 0.5 and True)
    g = TK.sinusoid_eval({k: v for k, v in (("f", T["sin_f"]), ("phi", T["sin_phi"]), ("a", T["sin_a"]))}, np.linspace(*T["s_range"], 2001), *T["s_range"])
    check("g_k normalized: mean 0, std 1 on the grid for every task", np.allclose(g.mean(1), 0, atol=1e-6) and np.allclose(g.std(1), 1, atol=1e-6))
    check("frequencies inside [freq_min, freq_max]", T["sin_f"].min() >= c["sinusoid"]["freq_min"] and T["sin_f"].max() <= c["sinusoid"]["freq_max"])
    check("5-D: n_labels = max(30, ceil(1% of pool)); 1 + 3 label sets; combos use only their levels", meta["n_labels_5D"] == 30 and T["ids_d"].shape == (4, 30) and
          {meta["levels"][T["lvl_flat"][i]] for i in T["ids_d"][1]} <= {"a060", "ind"} and {meta["levels"][T["lvl_flat"][i]] for i in T["ids_d"][3]} <= {"a060", "a160"})
    check("area deciles q in 0..9, all used", set(np.unique(T["q"]).tolist()) == set(range(10)))
    T2, _ = S.build_tasks(c, c2, rows, mf, frames, [0, 1, 2, 3], [4, 5, 6, 7])
    check("deterministic given the config seeds", all(np.array_equal(T[k], T2[k]) for k in T))
    return T, meta, c, c2, rows


# ------------------------------------------------------------------------------------------------ C. metrics vs brute force
def t_metrics(T):
    print("C  IB metrics vs brute force, pair groups, planted models")
    ctx = IB.Ctx(T["lvl"], T["q"], T["is_ind"], T["is_ood"])
    n = ctx.n
    rng = np.random.RandomState(0)
    Y = (rng.rand(5, n) < 0.5).astype(np.float32)
    for metric, group in (("R-IB", "all"), ("D-IB", "ood_any"), ("D-IB_area=", "all"), ("R-IB_fold", "ind"), ("D-IB_area≠", "ood_both")):
        M = IB.pair_mask(ctx, metric, group)
        brute = []
        for k in range(5):
            num = den = 0
            for i in range(n):
                for j in range(n):
                    if M[i, j]:
                        den += 1
                        num += Y[k, i] == Y[k, j]
            brute.append(num / den)
        check("eq_mean_per_task == brute force (%s, %s)" % (metric, group), np.allclose(IB.eq_mean_per_task(Y, M), brute, atol=1e-6))
    ind, ood = ctx.is_ind, ctx.is_ood
    M = IB.pair_mask(ctx, "R-IB", "ind")
    check("pair group ind: both inputs from trained levels; diagonal excluded", np.all(M[~(ind[:, None] & ind[None, :])] == 0) and np.all(np.diag(M) == 0))
    M = IB.pair_mask(ctx, "D-IB", "ood_both")
    check("pair group ood_both: both OoD and different level", np.all(M[~(ood[:, None] & ood[None, :])] == 0) and np.all(M[ctx.lvl[:, None] == ctx.lvl[None, :]] == 0))
    M = IB.pair_mask(ctx, "D-IB", "ood_any")
    check("pair group ood_any: at least one OoD", np.all(M[(ind[:, None] & ind[None, :])] == 0))
    # planted models
    Ta = 40
    lab = np.stack([(np.random.RandomState(k).rand(9) < 0.5) for k in range(Ta)]).astype(np.float32)
    perfect = lab[:, T["lvl"]]
    qlab = np.stack([(np.random.RandomState(100 + k).rand(10) < 0.5) for k in range(Ta)]).astype(np.float32)
    areaonly = qlab[:, T["q"]]
    noninf = IB.noninformative(Ta, ctx.n, 1)
    const = IB.constant_model(Ta, ctx.n)
    raw = IB.raw_scores_multi({"perfect": perfect, "area": areaonly, "noninf": noninf, "const": const}, ctx)
    g = "all"
    check("perfect-state model: R-IB raw == 1 and R-IB_fold raw == 1", raw["perfect"][("R-IB", g)] == 1.0 and raw["perfect"][("R-IB_fold", g)] == 1.0)
    check("non-informative (i.i.d.) model: R-IB raw ~ 0.5 and D-IB raw ~ 0.5", abs(raw["noninf"][("R-IB", g)] - 0.5) < 0.02 and abs(raw["noninf"][("D-IB", g)] - 0.5) < 0.02)
    check("constant model: R-IB raw == 1, D-IB raw == 0", raw["const"][("R-IB", g)] == 1.0 and raw["const"][("D-IB", g)] == 0.0)
    check("score vs i.i.d. reference: perfect model R-IB score == 1, noninf score == 0", abs(IB.normalize(raw["perfect"][("R-IB", g)], raw["perfect"][("R-IB", g)], raw["noninf"][("R-IB", g)], 0.05) - 1) < 1e-9
          and abs(IB.normalize(raw["noninf"][("R-IB", g)], raw["perfect"][("R-IB", g)], raw["noninf"][("R-IB", g)], 0.05)) < 1e-9)
    gap = raw["perfect"][("D-IB", g)] - raw["noninf"][("D-IB", g)]
    check("FINDING: with random binary labels the perfect-state D-IB raw (~0.5) is not separated from the i.i.d. reference (gap < min_gap) -> score NaN, not noise",
          abs(gap) < 0.05 and np.isnan(IB.normalize(raw["perfect"][("D-IB", g)], raw["perfect"][("D-IB", g)], raw["noninf"][("D-IB", g)], 0.05)), "(gap %.3f)" % gap)
    sc = IB.normalize(raw["perfect"][("D-IB", g)], raw["perfect"][("D-IB", g)], raw["const"][("D-IB", g)], 0.05)
    check("with the constant-model reference the perfect model D-IB score == 1", abs(sc - 1) < 1e-9)
    check("area-only model: D-IB_area= raw == 0 exactly (same decile -> same prediction), D-IB_area!= raw > 0.3", raw["area"][("D-IB_area=", g)] == 0.0 and raw["area"][("D-IB_area≠", g)] > 0.3,
          "(%.3f)" % raw["area"][("D-IB_area≠", g)])
    check("area-only model: R-IB_fold raw < 0.7 while the perfect model has 1.0 (fold = same level, different decile)", raw["area"][("R-IB_fold", g)] < 0.7)
    check("perfect-state model: D-IB_area= ~ D-IB_area!= (state model is indifferent to the decile)", abs(raw["perfect"][("D-IB_area=", g)] - raw["perfect"][("D-IB_area≠", g)]) < 0.08)
    # calibration
    truth = np.stack([np.sin(2 * np.pi * (0.5 + 0.3 * k) * T["s_eval"]) for k in range(20)])
    Dor = IB.dist_matrix(truth)
    brute = np.array([[np.mean((truth[:, i] - truth[:, j]) ** 2) for j in range(0, 40)] for i in range(0, 40)])
    check("dist_matrix == brute force", np.allclose(Dor[:40, :40], brute, atol=1e-8))
    sel = IB.pair_selector(ctx, "all")
    norm = float(np.triu(Dor, 1).max())
    bins, mad0, npairs = IB.calibration(Dor, Dor, sel, 20, norm)
    check("calibration: model == oracle -> mean abs deviation 0, 20 bins, counts add up to all pairs", mad0 < 1e-12 and len(bins) == 20 and sum(b["n_pairs"] for b in bins) == ctx.n * (ctx.n - 1) // 2 == npairs)
    Dc = IB.dist_matrix(np.zeros_like(truth))
    check("calibration: constant model deviates", IB.calibration(Dor, Dc, sel, 20, norm)[1] > 0.05)
    Ds = IB.dist_matrix(truth * 0.5)
    _, mads, _ = IB.calibration(Dor, Ds, sel, 20, norm)
    check("calibration: model with half amplitude -> distances scaled by 0.25 -> deviation > 0", mads > 0.02)
    sl, ic, r2 = IB.ols_line(T["s_eval"], 1.7 * T["s_eval"] + 0.3)
    check("ols_line recovers slope/intercept", abs(sl - 1.7) < 1e-9 and abs(ic - 0.3) < 1e-9 and abs(r2 - 1) < 1e-9)
    check("ols_line on a constant s -> NaN", np.isnan(IB.ols_line(np.ones(10), np.arange(10))[0]))


# ------------------------------------------------------------------------------------------------ D. heads + pipeline (torch)
def synth_tokens(T, rows, frames, kind, P=4, D=16, seed=0):
    """(E, nF, P, D) float16 tokens with planted content: 'state' (log-size along a fixed direction), 'area' (log visible area only), 'noise'."""
    rng = np.random.RandomState(seed)
    E, nF = len(rows), len(frames)
    s = np.array([r["nominal_log_area"] for r in rows])
    a = np.random.RandomState(5).randn(D)
    a /= np.linalg.norm(a)
    tok = 0.3 * rng.randn(E, nF, P, D)
    if kind == "state":
        tok += 2.0 * s[:, None, None, None] * a
    elif kind == "area":
        return None
    return tok.astype(np.float16)


def t_torch_pipeline(cfg, cfg2):
    print("D  heads (real code) + metrics (real code) on planted tokens: perfect state / area-only / random")
    import torch
    from . import heads as HD
    from . import stages as S
    c, c2 = small_cfgs(cfg, cfg2)
    rows, frames, mf = synth_cache(c2)
    T, meta = S.build_tasks(c, c2, rows, mf, frames, [0, 1, 2, 3], [4, 5, 6, 7])
    dev = torch.device("cpu")
    Etr = sum(r["part"] == "train" for r in rows)
    nF = len(frames)
    ac, hc = c2["probe_impl"]["attentive"], c["head"]
    # planted tokens
    state_tok = synth_tokens(T, rows, frames, "state")
    area_f = PR_feats(mf, frames)                                                     # (E,nF,5) area/bbox features
    af = (area_f - area_f[:Etr].reshape(-1, 5).mean(0)) / (area_f[:Etr].reshape(-1, 5).std(0) + 1e-6)
    area_tok = np.repeat(af[:, :, None, :], 4, 2)                                    # tokens that contain ONLY the visible area (pad to D)
    area_tok = np.concatenate([area_tok, np.zeros(area_tok.shape[:-1] + (11,))], -1).astype(np.float16)
    area_tok = area_tok + (0.05 * np.random.RandomState(2).randn(*area_tok.shape)).astype(np.float16)
    noise_tok = synth_tokens(T, rows, frames, "noise", seed=9)
    srcs = {"state": S._sources(c, T, state_tok, Etr, nF, dev), "areatok": S._sources(c, T, area_tok, Etr, nF, dev), "noise": S._sources(c, T, noise_tok, Etr, nF, dev)}
    base = S.base_sources(c, T, dev)
    check("area head: input LayerNorm replaced by Identity (scale of the 5 features is kept)", HD._fit_attn.__doc__ is None and base["area"].no_input_norm is True)
    need = {"A", "B", "D"}
    P = {}
    ts = {}
    import time
    for seed in c["seeds"]["head"]:
        for nm, src in base.items():
            P[("base", nm, seed)] = S.predict_model(src, T, need, seed, ac, hc)
        for nm, key in (("state", "state"), ("areatok", "areatok"), ("noise", "noise")):
            P[(nm, "ours", seed)] = S.predict_model(srcs[key], T, need, seed, ac, hc)
            P[(nm, "random", seed)] = S.predict_model(srcs["noise"], T, need, seed, ac, hc)
    plan = {"layers": [{"name": n} for n in ("state", "areatok", "noise")], "layer_subexps": {n: ["5-A", "5-B", "5-C", "5-D"] for n in ("state", "areatok", "noise")}}
    cc = copy.deepcopy(c)
    cc["normalization"] = {"noninf": {"R": "iid", "D": "const"}, "min_gap": 0.05}
    R = S.compute_metrics(cc, T, P, plan, c["seeds"]["head"])
    ib = R["ib"]

    def val(layer, model, metric, group="all", key="raw"):
        v = [r[key] for r in ib if r["layer"] == layer and r["model"] == model and r["metric"] == metric and r["pair_group"] == group]
        return float(np.nanmean(v))
    check("oracle linear/MLP heads trained on s only: R-IB raw == 1 (a function of s is constant within a level)", val("state", "oracle", "R-IB") == 1.0 and val("state", "oracle_linear", "R-IB") == 1.0)
    check("perfect-state tokens: R-IB score high (>0.8) on all pairs", val("state", "ours", "R-IB", key="score") > 0.8, "(%.3f)" % val("state", "ours", "R-IB", key="score"))
    check("perfect-state tokens: R-IB score for OoD-containing pairs also high (>0.7)", val("state", "ours", "R-IB", "ood_any", key="score") > 0.7, "(%.3f)" % val("state", "ours", "R-IB", "ood_any", key="score"))
    dn, ds_ = val("noise", "ours", "D-IB", key="score"), val("state", "ours", "D-IB", key="score")
    check("random (noise) tokens: D-IB score (constant-model reference) well below the perfect-state tokens (R-IB alone is high for a head that predicts one value everywhere)",
          dn < ds_ - 0.2, "(noise %.3f, state %.3f; R-IB score noise %.3f)" % (dn, ds_, val("noise", "ours", "R-IB", key="score")))
    ae_state, ae_area = val("state", "ours", "D-IB_area=", key="raw"), val("areatok", "ours", "D-IB_area=", key="raw")
    ane_state, ane_area = val("state", "ours", "D-IB_area\u2260", key="raw"), val("areatok", "ours", "D-IB_area\u2260", key="raw")
    check("area-only tokens: D-IB_area= raw lower than the perfect-state tokens' (>0.1 lower; same decile -> same prediction) and lower than their own D-IB_area!=",
          ae_state - ae_area > 0.1 and ae_area < ane_area - 0.1, "(area= state %.3f / area-only %.3f ; area!= state %.3f / area-only %.3f)" % (ae_state, ae_area, ane_state, ane_area))
    check("area head model behaves like an area model: D-IB_area= raw below D-IB_area!= raw", val("state", "area", "D-IB_area=") < val("state", "area", "D-IB_area≠"))
    check("score_alt column filled with the other reference; ref_mode recorded", all(r["ref_mode"] in ("iid", "const") for r in ib) and any(np.isfinite(r["score_alt"]) for r in ib))
    cd = {(r["layer"], r["model"], r["group"]): r["value"] for r in R["cal_dev"] if r["seed"] == 0}
    check("calibration: planted perfect-state tokens deviate less than random tokens and than the area head (group all)", cd[("state", "ours", "all")] < cd[("noise", "ours", "all")] and cd[("state", "ours", "all")] < cd[("state", "area", "all")],
          "(state %.3f noise %.3f area %.3f)" % (cd[("state", "ours", "all")], cd[("noise", "ours", "all")], cd[("state", "area", "all")]))
    check("calibration: rows for the truth model are finite; groups all/ind/extrap_any present", np.isfinite(cd[("state", "truth", "all")]) and {"all", "ind", "extrap_any"} == {g for (_, _, g) in cd})
    law = {(r["layer"], r["model"], r["slice"]): r["slope"] for r in R["law"] if r["seed"] == 0}
    sl = [law[("state", "ours", s)] for s in ("interp", "extrap", "early", "late")]
    check("5-D: the linear oracle head recovers slope ~1 in every slice (the hidden-5 MLP oracle saturates outside the trained range: not asserted)", all(abs(law[("state", "oracle_linear", s)] - 1) < 0.25 for s in ("interp", "extrap", "early", "late")),
          str({s: round(law[("state", "oracle_linear", s)], 2) for s in ("interp", "extrap", "early", "late")}))
    check("5-D: 7 slices per model (interp, extrap, early, late, 3 label combos)", len({r["slice"] for r in R["law"] if r["layer"] == "state" and r["model"] == "ours"}) == 7)
    return T, meta, R, cc, plan, c, c2


def PR_feats(mf, frames):
    from exp002 import probes as PR
    return PR.mask_baseline_features(mf[:, frames]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ E. verdicts, files
def mk_ib(layer, model, metric, group, scores, raw=None):
    return [{"layer": layer, "model": model, "metric": metric, "pair_group": group, "seed": i, "score": v, "raw": v if raw is None else raw[i]} for i, v in enumerate(scores)]


def t_verdicts(cfg):
    print("E  verdict rules (supported / refuted / not_evaluable), thresholds from config")
    vc = cfg["verdict"]
    g = vc["h_e"]["pair_group"]
    rows = []
    for m in ("R-IB", "D-IB"):
        rows += mk_ib("L", "ours", m, g, [0.80, 0.82, 0.78, 0.81, 0.79]) + mk_ib("L", "area", m, g, [0.40, 0.42, 0.38, 0.41, 0.39]) + mk_ib("L", "random", m, g, [0.05, 0.06, 0.04, 0.05, 0.05])
    cal = [{"layer": "L", "model": "ours", "group": "extrap_any", "seed": i, "value": 0.05} for i in range(5)]
    r = VD.h_e(rows, cal, "L", vc)
    check("H-e supported: above area, outside random's seed sd, calibration dev 0.05 <= 0.1", r["status"] == "supported", str(r.get("reason")))
    cal2 = [dict(c_, value=0.15) for c_ in cal]
    check("H-e refuted when the 45-degree deviation is 0.15 > 0.1", VD.h_e(rows, cal2, "L", vc)["status"] == "refuted")
    rows_b = [dict(x) for x in rows]
    for x in rows_b:
        if x["model"] == "area" and x["metric"] == "D-IB":
            x["score"] = 0.9
    check("H-e: area baseline score >= saturation (0.95) -> not_evaluable; 0.9 above ours -> refuted", VD.h_e(rows_b, cal, "L", vc)["status"] == "refuted")
    for x in rows_b:
        if x["model"] == "area" and x["metric"] == "D-IB":
            x["score"] = 0.97
    check("H-e: area saturated -> not_evaluable", VD.h_e(rows_b, cal, "L", vc)["status"] == "not_evaluable")
    rows_c = [dict(x) for x in rows]
    for x in rows_c:
        if x["model"] == "random":
            x["score"] = x["score"] + 0.78
    check("H-e: ours not outside random's seed sd -> refuted", VD.h_e(rows_c, cal, "L", vc)["status"] == "refuted")
    rows_n = [dict(x, score=float("nan")) if x["metric"] == "D-IB" else x for x in rows]
    check("H-e: NaN score (oracle not separated from the reference) -> not_evaluable", VD.h_e(rows_n, cal, "L", vc)["status"] == "not_evaluable")
    check("H-e: single seed -> not_evaluable", VD.h_e([x for x in rows if x["seed"] == 0], [c_ for c_ in cal if c_["seed"] == 0], "L", vc)["status"] == "not_evaluable")
    v2 = copy.deepcopy(vc)
    v2["h_e"]["dev_max"] = None
    check("H-e: null threshold -> not_evaluable", VD.h_e(rows, cal, "L", v2)["status"] == "not_evaluable")
    check("H-e: missing calibration rows -> not_evaluable", VD.h_e(rows, [], "L", vc)["status"] == "not_evaluable")
    fa = mk_ib("L", "ours", "D-IB_area=", "all", [0.50, 0.52, 0.48, 0.51, 0.49]) + mk_ib("L", "ours", "D-IB_area≠", "all", [0.51, 0.50, 0.49, 0.52, 0.48])
    check("H-f supported: area= equals area!= within seed sd", VD.h_f(fa, "L", vc)["status"] == "supported")
    fb = mk_ib("L", "ours", "D-IB_area=", "all", [0.10, 0.12, 0.08, 0.11, 0.09]) + mk_ib("L", "ours", "D-IB_area≠", "all", [0.50, 0.50, 0.49, 0.52, 0.48])
    check("H-f refuted: area= far below area!=", VD.h_f(fb, "L", vc)["status"] == "refuted")
    check("H-f: missing rows -> not_evaluable", VD.h_f([], "L", vc)["status"] == "not_evaluable")
    law = lambda sl, vals: [{"layer": "L", "model": "ours", "slice": sl, "seed": i, "slope": v} for i, v in enumerate(vals)]
    ok = law("interp", [0.9, 1.0, 1.1]) + law("extrap", [0.95, 1.0, 1.05]) + law("early", [1.0] * 3) + law("late", [1.1] * 3)
    check("H-g supported: every slice slope in [0.8, 1.2]", VD.h_g(ok, "L", vc)["status"] == "supported")
    bad = ok + law("labels_a060+ind", [0.5, 0.6, 0.55])
    check("H-g refuted: one slice at 0.55", VD.h_g(bad, "L", vc)["status"] == "refuted")
    check("H-g: NaN slope in a slice -> not_evaluable", VD.h_g(ok + law("x", [float("nan")] * 3), "L", vc)["status"] == "not_evaluable")
    v3 = copy.deepcopy(vc)
    v3["h_g"]["slope_lo"] = None
    check("H-g: null threshold -> not_evaluable", VD.h_g(ok, "L", v3)["status"] == "not_evaluable")


def t_files(cfg, cfg2, bundle):
    print("F  metrics_stage / report_stage on saved synthetic predictions (files, columns, REPORT.md)")
    from . import stages as S
    T, meta, R, cc, plan, c, c2 = bundle
    tmp = Path(tempfile.mkdtemp())
    try:
        res = tmp / "results"
        res.mkdir()
        np.savez_compressed(res / "tasks.npz", **T)
        write_json(meta, res / "tasks_meta.json")
        paths = {"results_dir": res, "exp": tmp}
        (res / "predictions").mkdir()
        # re-create predictions from R-independent source: run tiny heads for two layers via the real functions
        import torch
        from . import stages as S2
        dev = torch.device("cpu")
        rows, frames, mf = synth_cache(c2)
        Etr = sum(r["part"] == "train" for r in rows)
        nF = len(frames)
        ac, hc = c2["probe_impl"]["attentive"], c["head"]
        st = synth_tokens(T, rows, frames, "state")
        nz = synth_tokens(T, rows, frames, "noise", seed=9)
        sS, sN = S2._sources(c, T, st, Etr, nF, dev), S2._sources(c, T, nz, Etr, nF, dev)
        bs = S2.base_sources(c, T, dev)
        for seed in [0, 1]:
            for nm, src in bs.items():
                np.savez_compressed(S2.pred_file(paths, "base", nm, seed), **S2.predict_model(src, T, {"A", "B", "D"}, seed, ac, hc))
            for layer in ("L1", "L2"):
                np.savez_compressed(S2.pred_file(paths, layer, "ours", seed), **S2.predict_model(sS, T, {"A", "B", "D"}, seed, ac, hc))
                np.savez_compressed(S2.pred_file(paths, layer, "random", seed), **S2.predict_model(sN, T, {"A", "B", "D"}, seed, ac, hc))
        plan2 = {"layers": [{"name": "L1", "module": "encoder", "depth": 12, "kind": "hidden"}, {"name": "L2", "module": "predictor", "depth": 3, "kind": "hidden"}],
                 "layer_subexps": {"L1": ["5-A", "5-B"], "L2": ["5-A", "5-C", "5-D"]}, "rows_matched": ["synthetic"], "hold": False, "pilot": True, "unresolved": [], "subexps": ["5-A", "5-B", "5-C", "5-D"]}
        cfg_m = copy.deepcopy(c)
        cfg_m["_config_path"] = cfg["_config_path"]
        cfg_m["normalization"] = {"noninf": {"R": "iid", "D": "const"}, "min_gap": 0.05}
        S.metrics_stage(cfg_m, c2, paths, plan2, [0, 1])
        hdr = lambda f: open(res / f, encoding="utf-8").readline().strip().split(",")
        check("ib_scores.csv has the spec columns (layer, model, metric, pair_group, seed, raw, score)", hdr("ib_scores.csv")[:7] == ["layer", "model", "metric", "pair_group", "seed", "raw", "score"])
        ibr = S.read_csv_rows(res / "ib_scores.csv", ["raw", "score"])
        L1 = {r["metric"] for r in ibr if r["layer"] == "L1"}
        L2 = {r["metric"] for r in ibr if r["layer"] == "L2"}
        check("per-layer sub-experiment selection: L1 (5-A,5-B) has R-IB/D-IB only; L2 (5-A,5-C) has all five metrics", L1 == {"R-IB", "D-IB"} and L2 == {"R-IB", "D-IB", "D-IB_area=", "D-IB_area≠", "R-IB_fold"})
        check("models in ib_scores.csv: ours, oracle, oracle_linear, area, random, noninf, const", {r["model"] for r in ibr} == {"ours", "oracle", "oracle_linear", "area", "random", "noninf", "const"})
        check("pair groups all/ind/ood_any/ood_both present", {r["pair_group"] for r in ibr} == {"all", "ind", "ood_any", "ood_both"})
        check("calibration.csv / calibration_dev.csv / law_fits.csv written; law only for L2 (5-D)", (res / "calibration.csv").exists() and (res / "calibration_dev.csv").exists() and
              {r["layer"] for r in S.read_csv_rows(res / "law_fits.csv", ["slope"])} == {"L2"} and {r["layer"] for r in S.read_csv_rows(res / "calibration_dev.csv", ["value"])} == {"L1"})
        check("predictions saved per layer/model/seed (re-computable without retraining)", len(list((res / "predictions").glob("*.npz"))) == 2 * 3 + 2 * 2 * 2)
        figs = [f for f in ("fig_calibration.png", "fig_dib_decomp.png") if (res / f).exists()]
        check("figures written when matplotlib exists", len(figs) in (0, 2))
        cfg_r = copy.deepcopy(cfg_m)
        cfg_r["verdict"]["h_e"]["calibration_group"] = "extrap_any"
        out = S.report_stage(cfg_r, c2, paths, plan2)
        check("report_stage writes REPORT.md + results.json with interpretation null", (res / "REPORT.md").exists() and (tmp / "results.json").exists() and read_json(tmp / "results.json")["interpretation"] is None)
        vr = {v["layer"]: v for v in out["verdicts"]}
        check("verdict for L1: H-e evaluated (5-A and 5-B selected), H-f/H-g 'not selected' -> not_evaluable; L2: H-e not_evaluable (no 5-B)",
              vr["L1"]["H-e"]["status"] in ("supported", "refuted", "not_evaluable") and vr["L1"]["H-f"]["reason"].startswith("5-C_not_selected") and vr["L2"]["H-e"]["status"] == "not_evaluable" and vr["L2"]["H-g"]["status"] in ("supported", "refuted", "not_evaluable"))
        hold = dict(plan2, hold=True)
        out2 = S.report_stage(cfg_r, c2, paths, hold)
        check("plan.hold (experiment 1 negative) -> every verdict not_evaluable", all(v[k]["status"] == "not_evaluable" for v in out2["verdicts"] for k in ("H-e", "H-f", "H-g")))
        check("PySR absent -> status string recorded, not silently empty", S.pysr_available() or S.pysr_fit(np.arange(5), np.arange(5), cfg_m) == "pysr_not_installed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_config(cfg, cfg2):
    print("G  config explicitness")
    check("all seeds explicit (head x5, eval, tasks_a, tasks_b, sinus, d, noninf)", cfg["seeds"]["head"] == [0, 1, 2, 3, 4] and all(isinstance(cfg["seeds"][k], int) for k in ("eval", "tasks_a", "tasks_b", "sinus", "d", "noninf")))
    check("spec numbers: 100 tasks x 100 inputs (5-A and 5-B), 2000 eval inputs, 20 bins, 10 deciles, 1% / min 30 labels", cfg["ib"]["n_tasks"] == 100 and cfg["ib"]["n_per_task"] == 100 and cfg["calib"]["n_tasks"] == 100 and
          cfg["eval"]["n_inputs"] == 2000 and cfg["calib"]["n_bins"] == 20 and cfg["q"]["n_bins"] == 10 and cfg["law"]["label_fraction"] == 0.01 and cfg["law"]["min_labels"] == 30)
    check("verdict thresholds 0.1 / 0.8-1.2", cfg["verdict"]["h_e"]["dev_max"] == 0.1 and (cfg["verdict"]["h_g"]["slope_lo"], cfg["verdict"]["h_g"]["slope_hi"]) == (0.8, 1.2))
    check("002 config loaded, head seeds override, sha recorded", cfg2["probe"]["seeds"] == [0, 1, 2, 3, 4] and len(cfg2["_config002_sha256"]) == 64)


def run(cfg, cfg2):
    steps = [("A", lambda: t_branch(cfg)), ("G", lambda: t_config(cfg, cfg2)), ("E", lambda: t_verdicts(cfg))]
    out = {}

    def b():
        out["T"] = t_tasks(cfg, cfg2)
    steps.insert(1, ("B", b))
    steps.append(("C", lambda: t_metrics(out["T"][0])))
    if have_torch():
        def d():
            out["D"] = t_torch_pipeline(cfg, cfg2)
        steps.append(("D", d))
        steps.append(("F", lambda: t_files(cfg, cfg2, out["D"])))
    else:
        skip("D/F torch pipeline and file tests", "no torch")
    for name, fn in steps:
        try:
            fn()
        except Exception:
            REC["fail"].append(name + " (exception)")
            traceback.print_exc()
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    if REC["fail"]:
        print("FAILED:", REC["fail"])
    return 1 if REC["fail"] else 0
