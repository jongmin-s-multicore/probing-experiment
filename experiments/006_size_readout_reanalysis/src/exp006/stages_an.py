"""Analysis stages A-E, reproduction check. CPU only. Numbers only: no thresholds, no pass/fail, no p values, no interpretation text."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import base as B
from . import lib as L
from .base import log
from .ctx import Ctx, load_pooled, load_extract, save_part, jsonable_agg
from .stages_pre import n_enc, n_per, n_roll, n_unt, D_NAMES


def _bootstrap(ctx):
    from exp002 import stats as ST
    return ST.Bootstrap(ctx.lvl_idx, int(ctx.cfg2["probe_impl"]["n_bootstrap"]), int(ctx.cfg["seeds"]["bootstrap"]))


def series(ctx, preds_by_step, boot=None, within=False):
    """preds_by_step: list over steps of (P, Ete) probe-seed predictions. Returns {r2:{role:[..]}, mae:{role:[..]}, (CI)}; labels = Yp[test, t+k]."""
    from exp002 import stats as ST
    out = {"r2": {}, "mae": {}}
    ci = boot is not None
    if ci:
        out["r2_ci95_episode_bootstrap"], out["mae_ci95_episode_bootstrap"] = {}, {}
    if within:
        out["r2_within_level|all"] = []
    for role, m in ctx.role_masks().items():
        for kn in ("r2", "mae"):
            out[kn][role] = []
            if ci and role in ("unseen", "trained"):
                out[kn + "_ci95_episode_bootstrap"][role] = []
    for k, H in zip(ctx.steps, preds_by_step):
        S = L.cells_from_pred(ctx.label_te(k), H)
        for role, m in ctx.role_masks().items():
            for kn in ("r2", "mae"):
                if ci and role in ("unseen", "trained"):
                    b = boot.stat(S, m, kn)
                    out[kn][role].append(b["point"])
                    out[kn + "_ci95_episode_bootstrap"][role].append(ST.ci95(b["boot"]))
                else:
                    out[kn][role].append(L.metric_point(S, m, kn))
        if within:
            out["r2_within_level|all"].append(L.r2_within_levels(S, ctx.lvl_idx, len(ctx.levels)))
    return out


def _stack_quantities(per_step):
    """list over steps of dict(name -> scalar | (9,) array) -> dict(name -> array (steps, ...))."""
    names = per_step[0].keys()
    return {n: np.stack([np.asarray(d[n], float) for d in per_step]) for n in names}


def a_for_preds(ctx, preds):
    """preds {step: (P,n)} -> dict quantity -> (steps, ...) ; predictions are averaged over probe seeds first (spread recorded)."""
    per = []
    spread = 0.0
    for k in ctx.steps:
        H = np.asarray(preds[k], float)
        spread = max(spread, float(np.abs(H - H.mean(0, keepdims=True)).max()))
        per.append(L.a_stats(H.mean(0), ctx.label_te(k), ctx.lvl_idx, len(ctx.levels), ctx.role_masks()))
    return _stack_quantities(per), spread


# ------------------------------------------------------------------------------------------------ A
def stage_a(cfg, cfg2, paths, ctx=None, sources=None, transfer_pred=None):
    ctx = ctx or Ctx(cfg, cfg2, paths)
    sources = sources or list(cfg["analysis"]["a_sources"])
    out = {"definition": {
        "per_level": "mean over the 100 test episodes of a level of the probe prediction (log area ratio) at rollout step k; true = log nominal area ratio of the level; bias = mean_pred - true",
        "slope_episode|role": "OLS slope of the episode-level prediction on the label over the episodes of the role (9 levels for 'all', 6 for 'unseen', 3 for 'trained')",
        "slope_levelmeans|role": "OLS slope of the level means on the level labels",
        "probe_seed_handling": "predictions are averaged over the probe seeds before the summary; max abs spread over seeds is reported",
        "kinds": ["encoder (ceiling: same probe on encoder tokens of frame t+k)", "rollout (same probe on the predicted latent of step k)"]},
        "levels": ctx.levels, "level_true_log_area": [float(np.log(cfg2["data_impl"]["area_ratio"][l])) for l in ctx.levels], "steps": ctx.steps, "sources": {}}
    for src in sources:
        res, npz = {}, {}
        ok = True
        for cond in cfg["runs"]["conditions"]:
            res[cond] = {"runs": {}}
            for seed in cfg["runs"]["train_seeds"]:
                run = B.run_name(cfg, cond, seed)
                if src == "stored":
                    z = ctx.results002_cells(run)
                    preds = {kind: {k: ctx.stored_pred(z, run, kind, k) for k in ctx.steps} for kind in ("encoder", "rollout")}
                else:
                    if transfer_pred is None:
                        p = Path(paths["cache"]) / "transfer_pred.npz"
                        if not p.exists():
                            ok = False
                            break
                        transfer_pred = np.load(p)
                    preds = {"encoder": {k: transfer_pred[n_enc(k)] for k in ctx.steps}, "rollout": {k: transfer_pred[n_roll(run, k)] for k in ctx.steps}}
                res[cond]["runs"][str(seed)] = {}
                for kind, pk in preds.items():
                    q, spread = a_for_preds(ctx, pk)
                    res[cond]["runs"][str(seed)][kind] = {"q": q, "probe_seed_spread_max": spread}
                    for k in ctx.steps:
                        npz["%s|%s|step%d" % (run, kind, k)] = np.asarray(pk[k], float).mean(0).astype(np.float32)
            if not ok:
                break
        if not ok:
            out["sources"][src] = {"status": "skipped: %s missing (run transfer after extract)" % "transfer_pred.npz"}
            continue
        summ = {}
        for cond in cfg["runs"]["conditions"]:
            summ[cond] = {}
            for kind in ("encoder", "rollout"):
                stack = {n: np.stack([res[cond]["runs"][str(s)][kind]["q"][n] for s in cfg["runs"]["train_seeds"]]) for n in res[cond]["runs"][str(cfg["runs"]["train_seeds"][0])][kind]["q"]}
                summ[cond][kind] = {n: jsonable_agg(L.seed_agg(v)) for n, v in stack.items()}
        a, b = cfg["runs"]["conditions"][1], cfg["runs"]["conditions"][0]          # (a) single_ind, (b) multi_60_ind_160
        diffs = {}
        for kind in ("rollout",):
            for n in ("slope_episode|unseen", "slope_levelmeans|unseen", "intercept_episode|unseen", "mae|unseen", "per_level_bias", "per_level_mean_pred"):
                va = np.stack([res[a]["runs"][str(s)][kind]["q"][n] for s in cfg["runs"]["train_seeds"]])
                vb = np.stack([res[b]["runs"][str(s)][kind]["q"][n] for s in cfg["runs"]["train_seeds"]])
                diffs["%s|%s" % (kind, n)] = {kk: np.asarray(vv).tolist() if isinstance(vv, np.ndarray) else vv for kk, vv in L.diff_agg(va, vb).items()}
        out["sources"][src] = {"status": "ok", "per_run": {c: {s: {kind: {"q": {n: np.asarray(v).tolist() for n, v in d["q"].items()}, "probe_seed_spread_max": d["probe_seed_spread_max"]}
                                                          for kind, d in rr.items()} for s, rr in res[c]["runs"].items()} for c in res},
                               "seed_summary": summ, "a_minus_b_rollout": {"a": a, "b": b, "values": diffs}}
        B.probe_output_dir(paths)
        B.write_npz(paths, Path(paths["results"]) / ("A_per_episode_%s.npz" % src), lvl_idx=ctx.lvl_idx, nominal=ctx.nominal, **npz)
    save_part(ctx, "analyze_a", out)
    return out


# ------------------------------------------------------------------------------------------------ B
def _agg_over_seeds(cfg, per_seed_series, key, role):
    arr = np.array([per_seed_series[s][key][role] for s in sorted(per_seed_series)], float)
    return jsonable_agg(L.seed_agg(arr))


def stage_b(cfg, cfg2, paths, ctx=None, transfer_pred=None):
    ctx = ctx or Ctx(cfg, cfg2, paths)
    if transfer_pred is None:
        transfer_pred = np.load(Path(paths["cache"]) / "transfer_pred.npz")
    boot = _bootstrap(ctx)
    get = lambda name: transfer_pred[name]
    out = {"definition": {
        "r2": "exp002 definition: 1 - SSE/SST on the pooled episodes of the role, SST about the role mean of the label (probe-seed mean of per-seed values)",
        "persistence": "encoder tokens of the context frame t (t = num_hist - 1 = 0) copied to every step; the Q1 probe applied to the copy; label of step k = label of frame t+k",
        "ceiling": "same probe on the encoder tokens of frame t+k",
        "untrained": "predictor (and action/proprio encoders in variant pred_and_action_proprio) freshly initialised, see README section 3; DINOv2 input encoder is the original frozen one",
        "ci": "episode-cluster bootstrap 95% (levels fixed), evaluation episodes only; excludes training-seed and untrained-seed variation"},
        "steps": ctx.steps, "t": ctx.t,
        "first_frame_equals_last_context_frame": bool(ctx.t == 0),
        "note_first_frame": "num_hist = 1, so the last context frame is frame 0 = the first frame; both persistence definitions are the same set of predictions"}
    out["ceiling_encoder"] = series(ctx, [get(n_enc(k)) for k in ctx.steps], boot)
    out["persistence"] = series(ctx, [get(n_per(k)) for k in ctx.steps], boot)
    out["persistence_minus_ceiling"] = {kn: {r: (np.array(out["persistence"][kn][r]) - np.array(out["ceiling_encoder"][kn][r])).tolist() for r in out["persistence"][kn]} for kn in ("r2", "mae")}
    out["runs"] = {}
    variants = list(cfg["extract"]["untrained"]["variants"])
    useeds = [int(s) for s in cfg["seeds"]["untrained_predictor"]]
    for cond, seed, run in B.all_runs(cfg):
        d = {"trained_rollout": series(ctx, [get(n_roll(run, k)) for k in ctx.steps], boot), "untrained": {}}
        for v in variants:
            d["untrained"][v] = {}
            for us in useeds:
                d["untrained"][v][str(us)] = series(ctx, [get(n_unt(run, v, us, k)) for k in ctx.steps], boot)
        out["runs"][run] = {"condition": cond, "train_seed": seed, **d}
    # seed-level summary: per condition, mean over untrained seeds first (per training seed), then t-based CI over the training seeds
    summ = {}
    for cond in cfg["runs"]["conditions"]:
        seeds = [s for c, s, _ in B.all_runs(cfg) if c == cond]
        summ[cond] = {"trained_rollout": {}, "untrained": {v: {} for v in variants}}
        for kn in ("r2", "mae"):
            for role in ctx.role_masks():
                summ[cond]["trained_rollout"]["%s|%s" % (kn, role)] = jsonable_agg(L.seed_agg(np.array([out["runs"][B.run_name(cfg, cond, s)]["trained_rollout"][kn][role] for s in seeds])))
                for v in variants:
                    per_seed = np.array([np.mean([out["runs"][B.run_name(cfg, cond, s)]["untrained"][v][str(us)][kn][role] for us in useeds], axis=0) for s in seeds])
                    summ[cond]["untrained"][v]["%s|%s" % (kn, role)] = jsonable_agg(L.seed_agg(per_seed))
                    allv = np.array([[out["runs"][B.run_name(cfg, cond, s)]["untrained"][v][str(us)][kn][role] for us in useeds] for s in seeds])
                    summ[cond]["untrained"][v]["%s|%s|min_over_untrained_seeds_and_runs" % (kn, role)] = allv.min((0, 1)).tolist()
                    summ[cond]["untrained"][v]["%s|%s|max_over_untrained_seeds_and_runs" % (kn, role)] = allv.max((0, 1)).tolist()
    out["seed_summary"] = summ
    save_part(ctx, "analyze_b", out)
    return out


# ------------------------------------------------------------------------------------------------ C
def retrain_fit_apply(ctx, Xtr, ytr, Xte, seeds):
    """Probe retrained on TRAINING-episode latents only. Alpha (and the GroupKFold folds) are chosen inside the training rows by exp002 directions.ridge_fit.
    Xte enters only the final prediction call. Returns (H (P,n_te), alphas)."""
    from exp002 import probes as PR
    eptr = np.arange(len(ytr))
    H, alphas = [], []
    for s in seeds:
        preds, m = PR.linear_fit_predict(np.asarray(Xtr, np.float32), np.asarray(ytr, float), eptr, [np.asarray(Xte, np.float32)], s, ctx.cfg2)
        H.append(preds[0])
        alphas.append(m["alpha"])
    return np.stack(H), alphas


def stage_c(cfg, cfg2, paths, ctx=None, pooled=None, extracts=None, transfer_pred=None):
    ctx = ctx or Ctx(cfg, cfg2, paths)
    pooled = pooled if pooled is not None else load_pooled(paths)
    if transfer_pred is None:
        transfer_pred = np.load(Path(paths["cache"]) / "transfer_pred.npz")
    boot = _bootstrap(ctx)
    seeds = [int(s) for s in cfg["seeds"]["retrain_probe"]]
    out = {"definition": {
        "retrained": "ridge probe (exp002 probes.linear_fit_predict) fitted on the patch-mean latent of rollout step k of the probe-training rows (a060, ind, a160: 900 rows, one sample per row) with the label of frame t+k; alpha/folds chosen inside those rows only; evaluated on the test rows' latent of the same step",
        "transfer": "the Q1 probe fitted on encoder tokens only (002 output-space measurement) applied to the same latent",
        "encoder_retrained": "same retraining on the encoder tokens of frame t+k (single-frame Q1 ceiling, cross-checked with 002 q1 cells f<k>)",
        "limitation_a": "In condition (a) single_ind the predictor was trained on ind episodes only; its rollouts of the a060/a160 training rows (used here to train the probe) are rollouts of sizes it never saw. Condition (b) trained on all three probe-training sizes",
        "leak_rule": "no test row is passed to the fit; alpha and folds are chosen by GroupKFold inside the training rows"},
        "steps": ctx.steps, "probe_seeds": seeds, "runs": {}}
    # encoder retrained ceiling (shared by all runs)
    enc_preds, enc_alpha = [], []
    for k in ctx.steps:
        H, al = retrain_fit_apply(ctx, pooled[:ctx.Etr, ctx.t + k], ctx.Yp[:ctx.Etr, ctx.t + k], pooled[ctx.Etr:, ctx.t + k], seeds)
        enc_preds.append(H)
        enc_alpha.append(al)
    out["encoder_retrained"] = series(ctx, enc_preds, boot)
    out["encoder_retrained"]["alpha_per_probe_seed"] = enc_alpha
    out["encoder_retrained"]["slope_episode|unseen"] = [L.a_stats(H.mean(0), ctx.label_te(k), ctx.lvl_idx, len(ctx.levels), ctx.role_masks())["slope_episode|unseen"] for k, H in zip(ctx.steps, enc_preds)]
    q1 = np.load(Path(paths["results002"]) / "q1" / "cells.npz") if (Path(paths["results002"]) / "q1" / "cells.npz").exists() else None
    if q1 is not None:
        cc = {}
        for role in ("unseen", "trained"):
            m = ctx.rmasks[role]
            cc[role] = [{"step": k, "mine": L.metric_point(L.cells_from_pred(ctx.label_te(k), H), m, "r2"),
                         "q1_cell_f%d" % k: L.metric_point(q1["%s|linear_pooled|dino|f%d|frame%d" % (ctx.prim, k, k)], m, "r2")} for k, H in zip(ctx.steps, enc_preds)]
        out["encoder_retrained_vs_002_q1_single_frame_cells"] = cc
    for cond, seed, run in B.all_runs(cfg):
        ex = (extracts or {}).get(run) or load_extract(paths, run, with_untrained=False)
        roll = ex["roll"]
        preds, alphas = [], []
        for k in ctx.steps:
            H, al = retrain_fit_apply(ctx, roll[:ctx.Etr, k - 1], ctx.Yp[:ctx.Etr, ctx.t + k], roll[ctx.Etr:, k - 1], seeds)
            preds.append(H)
            alphas.append(al)
        r = series(ctx, preds, boot)
        r["alpha_per_probe_seed"] = alphas
        r["slope_episode|unseen"] = [L.a_stats(H.mean(0), ctx.label_te(k), ctx.lvl_idx, len(ctx.levels), ctx.role_masks())["slope_episode|unseen"] for k, H in zip(ctx.steps, preds)]
        r["per_level_mean_pred"] = [L.a_stats(H.mean(0), ctx.label_te(k), ctx.lvl_idx, len(ctx.levels), ctx.role_masks())["per_level_mean_pred"].tolist() for k, H in zip(ctx.steps, preds)]
        tr = series(ctx, [transfer_pred[n_roll(run, k)] for k in ctx.steps], boot)
        out["runs"][run] = {"condition": cond, "train_seed": seed, "retrained": r, "transfer": tr}
    summ = {}
    for cond in cfg["runs"]["conditions"]:
        seeds_c = [s for c, s, _ in B.all_runs(cfg) if c == cond]
        summ[cond] = {}
        for which in ("retrained", "transfer"):
            for kn in ("r2", "mae"):
                for role in ctx.role_masks():
                    arr = np.array([out["runs"][B.run_name(cfg, cond, s)][which][kn][role] for s in seeds_c])
                    summ[cond]["%s|%s|%s" % (which, kn, role)] = jsonable_agg(L.seed_agg(arr))
    out["seed_summary"] = summ
    save_part(ctx, "analyze_c", out)
    return out


# ------------------------------------------------------------------------------------------------ D
def stage_d(cfg, cfg2, paths, ctx=None, pooled=None, labels=None, extracts=None):
    from exp002 import directions as D
    from exp002 import stats as ST
    ctx = ctx or Ctx(cfg, cfg2, paths)
    pooled = pooled if pooled is not None else load_pooled(paths)
    if labels is None:
        lz = np.load(Path(paths["cache"]) / "d_labels.npz")
        labels, names = lz["labels"], [str(x) for x in lz["names"]]
    else:
        names = D_NAMES
    dp = cfg["d_properties"]
    use = []
    for i, nme in enumerate(names):
        if nme.startswith("mask_centroid") and not dp["mask_centroid"]:
            continue
        if nme.startswith("mask_bbox") and not dp["mask_bbox_center"]:
            continue
        if nme.startswith("cum_action") and not dp["cumulative_action"]:
            continue
        use.append(i)
    seeds = ctx.probe_seeds
    exts = extracts or {run: load_extract(paths, run, with_untrained=False) for _, _, run in B.all_runs(cfg)}
    sets = {}
    for k in ctx.steps:
        sets[("encoder", k)] = pooled[ctx.Etr:, ctx.t + k]
        sets[("persist", k)] = pooled[ctx.Etr:, ctx.t]
        for run, ex in exts.items():
            sets[(run, k)] = ex["roll"][ctx.Etr:, k - 1]
    Xtr = pooled[:ctx.Etr, ctx.frames_tr].reshape(ctx.Etr * len(ctx.frames_tr), -1)
    eptr = np.repeat(np.arange(ctx.Etr), len(ctx.frames_tr))
    out = {"definition": {
        "probe": "Q1 linear probe path (exp002 probes.linear_fit_predict): fitted on encoder tokens (patch mean) of the probe-training rows, frames 0..11, label of the same frame; applied unchanged to encoder tokens of frame k, persistence (frame t copied) and the rollout latent of step k",
        "labels": "mask_centroid / mask_bbox_center: cloth mask (002 HSV rule) of the cached 224x224 frame, coordinates / 224, frames with an empty mask are excluded from fit and evaluation; cum_action_*: sum of the raw actions with index < 5*frame for gripper 1 (dims dx1 dy1 dz1) and gripper 2 (dx2 dy2 dz2) = commanded displacement since frame 0 (NOT a measured gripper position; initial position not in the data fields read)",
        "r2": "role pooled R2 (1 - SSE/SST); r2_within_level|all: R2 after removing level means of label and prediction (002 stats.r2_within_from_level_sums)"},
        "steps": ctx.steps, "properties": [names[i] for i in use], "excluded_by_config": [names[i] for i in range(len(names)) if i not in use], "per_property": {}}
    for i in use:
        nme = names[i]
        ytr_all = labels[:ctx.Etr, ctx.frames_tr, i].reshape(-1)
        ok = np.isfinite(ytr_all)
        fits = [D.ridge_fit(Xtr[ok], ytr_all[ok].astype(float), eptr[ok], ctx.cfg2["probe_impl"]["ridge_alphas"], ctx.cfg2["probe_impl"]["inner_folds"], s) for s in seeds]
        res = {"n_train_samples_used": int(ok.sum()), "n_train_samples_excluded": int((~ok).sum()), "alpha_per_probe_seed": [m["alpha"] for m in fits], "sets": {}}
        for key, X in sets.items():
            k = key[1]
            y = labels[ctx.Etr:, ctx.t + k, i].astype(float)
            valid = np.isfinite(y)
            Hs = np.stack([D.ridge_predict(m, np.asarray(X, np.float32)) for m in fits])
            S = np.stack([ST.episode_stats(y[valid], h[valid], np.where(valid)[0], ctx.Ete) for h in Hs])
            d = {"n_valid": int(valid.sum())}
            for role, rm in ctx.role_masks().items():
                d["r2|" + role] = L.metric_point(S, rm, "r2")
                d["mae|" + role] = L.metric_point(S, rm, "mae")
            d["r2_within_level|all"] = L.r2_within_levels(S, ctx.lvl_idx, len(ctx.levels))
            res["sets"]["%s|%d" % (key[0], k)] = d
        out["per_property"][nme] = res
    summ = {}
    for cond in cfg["runs"]["conditions"]:
        seeds_c = [s for c, s, _ in B.all_runs(cfg) if c == cond]
        summ[cond] = {}
        for nme in out["properties"]:
            for metric in ("r2|unseen", "r2|trained", "r2|all", "r2_within_level|all", "mae|all"):
                arr = np.array([[out["per_property"][nme]["sets"]["%s|%d" % (B.run_name(cfg, cond, s), k)][metric] for k in ctx.steps] for s in seeds_c], float)
                summ[cond]["%s|%s" % (nme, metric)] = jsonable_agg(L.seed_agg(arr))
    out["seed_summary"] = summ
    save_part(ctx, "analyze_d", out)
    return out


# ------------------------------------------------------------------------------------------------ E
def stage_e(cfg, cfg2, paths, ctx=None):
    from exp002 import stats as ST
    ctx = ctx or Ctx(cfg, cfg2, paths)
    r002, r002_info = ctx.results002_json()
    grid = r002["exploratory"]["q2_layer_time_grid_r2_point_estimates"]
    rows = {r["id"]: r for r in r002["decision_table"]}
    out = {"definition": {
        "sd": "sample SD over training seeds (ddof = 1)", "ci95": "mean +- t_{0.975, n-1} * sd / sqrt(n) with n = 3 training seeds, df = 2, t = %.6f. Wide by construction; seeds are the unit, episodes are pooled inside each value" % L.t_crit(2),
        "drop": "R2(encoder tokens, step k) - R2(rollout, step k), same probe, role-pooled (002 H_b drop uses role 'unseen' at the last step)",
        "a_minus_b": "welch: diff of means with Welch-Satterthwaite t interval (no pairing); paired_by_seed_index: differences of seed s of (a) and seed s of (b), t interval df = n-1. Pairing by seed index is a convention only",
        "order_accuracy": "exp002 stats.pair_order_accuracy: fraction of cross-level episode pairs whose predicted order equals the true level order; mean over probe seeds; point estimate"},
        "results002_json": r002_info, "steps": ctx.steps, "per_run": {}, "consistency_with_results002_json": {}}
    quant = {}
    for cond, seed, run in B.all_runs(cfg):
        z = ctx.results002_cells(run)
        d = {}
        for role in ("unseen", "trained", "all"):
            m = ctx.rmasks[role]
            for kind in ("encoder", "rollout"):
                S_list = [ctx.stored_cell(z, run, kind, k) for k in ctx.steps]
                d["r2|%s|%s" % (kind, role)] = [L.metric_point(S, m, "r2") for S in S_list]
                d["mae|%s|%s" % (kind, role)] = [L.metric_point(S, m, "mae") for S in S_list]
                d["order_accuracy|%s|%s" % (kind, role)] = [float(np.mean([ST.pair_order_accuracy(p[m], ctx.nominal[m]) for p in ctx.stored_pred(z, run, kind, k)])) for k in ctx.steps]
            d["drop|%s" % role] = (np.array(d["r2|encoder|%s" % role]) - np.array(d["r2|rollout|%s" % role])).tolist()
        quant[run] = d
        out["per_run"][run] = {"condition": cond, "train_seed": seed, **d}
        g = grid.get("%s/seed%d" % (cond, seed), {})
        key12 = ctx.out_key("rollout", 12)
        if key12 in g:
            out["consistency_with_results002_json"][run] = {
                "rollout_step12_r2_unseen: drive_cells_minus_results_json": d["r2|rollout|unseen"][-1] - g[key12]["unseen"],
                "rollout_step12_r2_trained: drive_cells_minus_results_json": d["r2|rollout|trained"][-1] - g[key12]["trained"]}
    ht = rows.get("H_b:last_rollout_step", {}).get("result", {})
    ca = rows.get("cond_a_vs_b:last_rollout_step", {}).get("result", {})
    for tag, res in (("multi_60_ind_160", ht.get("per_training_seed", {})), ("single_ind", ca.get("per_training_seed_a", {}))):
        for s, v in res.items():
            run = B.run_name(cfg, tag, int(s))
            if run in quant:
                out["consistency_with_results002_json"].setdefault(run, {}).update({
                    "decision_table_r2_q2": v["r2_q2"], "decision_table_drop": v["drop"], "decision_table_r2_q1_ceiling": v["r2_q1_ceiling"],
                    "drive_drop_unseen_step12_minus_decision_table_drop": quant[run]["drop|unseen"][-1] - v["drop"]})
    summ = {}
    a, b = cfg["runs"]["conditions"][1], cfg["runs"]["conditions"][0]
    names = list(next(iter(quant.values())).keys())
    seeds_all = [int(s) for s in cfg["runs"]["train_seeds"]]
    for cond in cfg["runs"]["conditions"]:
        summ[cond] = {n: jsonable_agg(L.seed_agg(np.array([quant[B.run_name(cfg, cond, s)][n] for s in seeds_all]))) for n in names}
    out["seed_summary"] = summ
    out["a_minus_b"] = {"a": a, "b": b, "values": {n: {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in
                                                       L.diff_agg(np.array([quant[B.run_name(cfg, a, s)][n] for s in seeds_all]), np.array([quant[B.run_name(cfg, b, s)][n] for s in seeds_all])).items()}
                                                   for n in names}}
    save_part(ctx, "analyze_e", out)
    return out


# ------------------------------------------------------------------------------------------------ reproduction of 002 numbers
def stage_repro(cfg, cfg2, paths, ctx=None, transfer_pred=None):
    """Recompute 002's output-space R2 (linear, steps 1-12, roles unseen/trained) with the code path of this experiment and report the differences to (i) 002
    results.json exploratory.q2_layer_time_grid_r2_point_estimates and (ii) the values recomputed from the 002 Drive cells. NO tolerance is applied."""
    ctx = ctx or Ctx(cfg, cfg2, paths)
    if transfer_pred is None:
        transfer_pred = np.load(Path(paths["cache"]) / "transfer_pred.npz")
    r002, info = ctx.results002_json()
    grid = r002["exploratory"]["q2_layer_time_grid_r2_point_estimates"]
    roles = list(cfg["analysis"]["repro_reference_cells"])
    rows, flat = [], []
    pred_diffs = []
    z_by_run = {}
    for cond, seed, run in B.all_runs(cfg):
        if run not in z_by_run:
            z_by_run[run] = ctx.results002_cells(run)
        z = z_by_run[run]
        g = grid["%s/seed%d" % (cond, seed)]
        for kind in ("encoder", "rollout"):
            for k in ctx.steps:
                name = n_enc(k) if kind == "encoder" else n_roll(run, k)
                Hmine = np.asarray(transfer_pred[name], float)
                Hst = ctx.stored_pred(z, run, kind, k)
                S_mine = L.cells_from_pred(ctx.label_te(k), Hmine)
                pd_ = np.abs(Hmine - Hst)
                pred_diffs.append(float(pd_.max()))
                for role in roles:
                    m = ctx.rmasks[role]
                    mine = L.metric_point(S_mine, m, "r2")
                    from_cells = L.metric_point(ctx.stored_cell(z, run, kind, k), m, "r2")
                    from_json = g[ctx.out_key(kind, k)][role]
                    rows.append({"run": run, "kind": kind, "step": k, "role": role, "r2_this_experiment": mine, "r2_from_002_drive_cells": from_cells, "r2_002_results_json": from_json,
                                 "diff_vs_results_json": mine - from_json, "diff_vs_drive_cells": mine - from_cells, "max_abs_pred_diff_per_episode": float(pd_.max()),
                                 "mean_abs_pred_diff_per_episode": float(pd_.mean())})
                    if kind == "rollout":
                        flat.append(abs(mine - from_json))
    roll_rows = [r for r in rows if r["kind"] == "rollout"]
    enc_rows = [r for r in rows if r["kind"] == "encoder"]
    summ = {
        "rollout_vs_results_json": {"max_abs_r2_diff": float(max(abs(r["diff_vs_results_json"]) for r in roll_rows)), "mean_abs_r2_diff": float(np.mean([abs(r["diff_vs_results_json"]) for r in roll_rows])),
                                    "n": len(roll_rows), "max_abs_prediction_diff": float(max(r["max_abs_pred_diff_per_episode"] for r in roll_rows))},
        "encoder_vs_results_json": {"max_abs_r2_diff": float(max(abs(r["diff_vs_results_json"]) for r in enc_rows)), "mean_abs_r2_diff": float(np.mean([abs(r["diff_vs_results_json"]) for r in enc_rows])),
                                    "n": len(enc_rows), "max_abs_prediction_diff": float(max(r["max_abs_pred_diff_per_episode"] for r in enc_rows))},
        "drive_cells_vs_results_json_max_abs_r2_diff": float(max(abs(r["r2_from_002_drive_cells"] - r["r2_002_results_json"]) for r in rows)),
    }
    worst = sorted(roll_rows, key=lambda r: -abs(r["diff_vs_results_json"]))[:5]
    msg = ("REPRODUCTION OF 002 (no tolerance applied; the reader judges): rollout R2 max |diff| = %.3g, mean |diff| = %.3g over %d (run, step, role) values; encoder-probe max |diff| = %.3g. "
           "If these differences are large compared with the effects looked at in B-D, do not trust the analyses that use re-extracted latents (B untrained, C, D)." % (
               summ["rollout_vs_results_json"]["max_abs_r2_diff"], summ["rollout_vs_results_json"]["mean_abs_r2_diff"], len(roll_rows), summ["encoder_vs_results_json"]["max_abs_r2_diff"]))
    print("=" * 100 + "\n" + msg + "\n" + "=" * 100, flush=True)
    out = {"results002_json": info, "summary": summ, "banner": msg, "worst5_rollout_by_abs_diff": worst, "rows": rows, "roles": roles, "steps": ctx.steps,
           "note": "diff = this experiment - reference. Rollout latents are recomputed on this machine's GPU (batch composition, cuDNN kernels may differ from 002's run)."}
    save_part(ctx, "repro", out)
    return out
