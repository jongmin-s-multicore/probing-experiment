"""Analysis F: teacher forcing versus open loop. CPU only. Numbers only: no thresholds, no pass/fail, no p values, no interpretation text.

For each of the 6 runs and each step k = 1..12 the same two readouts are applied to two latents of the SAME checkpoint and the SAME rows:
  open_loop      the rollout latent of step k (the predictor's own outputs fed back as context; stored by `extract`)
  teacher_forced the one-step prediction of frame t+k from the REAL encoder latent of the previous frame (stored by `extract` as teacher_forced_pooled.npy)
Readouts: `transfer` = the Q1 probe fitted once on encoder tokens (transfer_pred.npz; sets `rollout|..` and `teacher|..`), `retrained` = ridge fitted on the latent of that
very step of the probe-training rows (as in C; alpha and folds are chosen inside the training rows only). Reported per step: R2, MAE (roles unseen / trained / all, episode
bootstrap CI for unseen and trained), level-wise mean predictions, bias, episode-level OLS slope and intercept. Seed level (training seeds, n = 3, df = 2) mean / SD / t CI as in E."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import base as B
from . import lib as L
from .base import log
from .ctx import Ctx, load_pooled, load_extract, save_part, jsonable_agg
from .stages_pre import n_enc, n_roll, n_tf
from .stages_an import _bootstrap, series, a_for_preds, retrain_fit_apply

Q_KEEP = ["per_level_mean_pred", "per_level_bias", "per_level_mae"]       # (steps, 9)


def _q_names(roles):
    names = list(Q_KEEP)
    for role in roles:
        names += ["slope_episode|" + role, "intercept_episode|" + role, "slope_levelmeans|" + role, "intercept_levelmeans|" + role, "bias_mean|" + role]
    return names


def _bundle(ctx, preds_by_step, boot):
    """preds_by_step: list over steps of (P, n_te). -> {series (r2, mae, CI), q (level stats, lists), probe_seed_spread_max}."""
    s = series(ctx, preds_by_step, boot)
    q, spread = a_for_preds(ctx, {k: H for k, H in zip(ctx.steps, preds_by_step)})
    keep = _q_names(ctx.role_masks().keys()) + ["per_level_true"]
    s["q"] = {n: np.asarray(q[n]).tolist() for n in keep if n in q}
    s["probe_seed_spread_max"] = spread
    s["_pred_mean"] = [np.asarray(H, float).mean(0) for H in preds_by_step]            # removed before saving
    return s


def _vec(b, name):
    """quantity name -> array over steps (or (steps, 9)). b is a bundle ('r2', 'mae', 'q') or the tf_minus_ol dict (same layout)."""
    if name.startswith(("r2|", "mae|")):
        kn, role = name.split("|")
        return np.array(b[kn][role], float)
    return np.array(b["q"][name], float)


def stage_f(cfg, cfg2, paths, ctx=None, pooled=None, extracts=None, transfer_pred=None, boot_override=None):
    ctx = ctx or Ctx(cfg, cfg2, paths)
    pooled = pooled if pooled is not None else load_pooled(paths)
    if transfer_pred is None:
        transfer_pred = np.load(Path(paths["cache"]) / "transfer_pred.npz")
    boot = boot_override or _bootstrap(ctx)
    seeds = [int(s) for s in cfg["seeds"]["retrain_probe"]]
    roles = list(ctx.role_masks().keys())
    out = {"definition": {
        "open_loop": "rollout latent of step k (autoregressive; the predictor's own earlier outputs are its context). Same array as in B and C",
        "teacher_forced": "one-step prediction of frame t+k from the real encoder latent of frame t+k-1 (num_hist = 1: a single context frame; its action part is the real action embedding, the same one the open-loop rollout writes back each step; its proprio part is the real encoded proprio instead of the predicted one). Same checkpoint, same rows, same pooling (fp16 cast, float32 patch mean)",
        "transfer": "the Q1 probe fitted once on encoder tokens (002 output-space measurement), applied unchanged to the latent",
        "retrained": "ridge fitted (exp002 probes.linear_fit_predict) on the latent of that step of the probe-training rows (a060, ind, a160: 900 rows) with the label of frame t+k; alpha and GroupKFold folds chosen inside those rows only; evaluated on the test rows' latent of the same step and the same kind (teacher-forced latents train and test the teacher-forced probe)",
        "labels": "label of step k = log nominal area ratio of frame t+k (same for both latents)",
        "tf_minus_ol": "teacher_forced minus open_loop of the same quantity, same run, same step",
        "seed_level": "n = 3 training seeds per condition, sample SD (ddof 1), CI = mean +- t_{0.975, 2} * sd / sqrt(3). Episode bootstrap CI (unseen, trained) covers evaluation episodes only",
        "no_thresholds": "no thresholds, pass/fail or p values"},
        "steps": ctx.steps, "probe_seeds": seeds, "roles": roles, "levels": ctx.levels, "runs": {}}
    # encoder references (shared by all runs)
    enc_tr = _bundle(ctx, [transfer_pred[n_enc(k)] for k in ctx.steps], boot)
    enc_rt_preds, enc_alpha = [], []
    for k in ctx.steps:
        H, al = retrain_fit_apply(ctx, pooled[:ctx.Etr, ctx.t + k], ctx.Yp[:ctx.Etr, ctx.t + k], pooled[ctx.Etr:, ctx.t + k], seeds)
        enc_rt_preds.append(H)
        enc_alpha.append(al)
    enc_rt = _bundle(ctx, enc_rt_preds, boot)
    enc_rt["alpha_per_probe_seed"] = enc_alpha
    for b in (enc_tr, enc_rt):
        b.pop("_pred_mean")
    out["encoder_reference"] = {"transfer": enc_tr, "retrained": enc_rt, "note": "encoder tokens of frame t+k (ceiling), same two readouts"}
    store = {}
    steps_available = None
    for cond, seed, run in B.all_runs(cfg):
        ex = (extracts or {}).get(run) or load_extract(paths, run, with_untrained=False)
        if ex.get("tf") is None:
            raise FileNotFoundError("extract/%s/teacher_forced_pooled.npy missing: rerun `extract` (--fresh) with the code that includes analysis F" % run)
        meta_tf = (ex.get("meta") or {}).get("teacher_forced") or {}
        steps_available = meta_tf.get("steps_available", steps_available)
        lat = {"open_loop": ex["roll"], "teacher_forced": ex["tf"]}
        res = {"condition": cond, "train_seed": seed, "latent_identity_step1_max_abs_diff_all_rows": float(np.abs(ex["tf"][:, 0] - ex["roll"][:, 0]).max()),
               "extract_meta_teacher_forced": meta_tf}
        for probe in ("transfer", "retrained"):
            res[probe] = {}
            for kind, arr in lat.items():
                if probe == "transfer":
                    preds = [transfer_pred[(n_roll if kind == "open_loop" else n_tf)(run, k)] for k in ctx.steps]
                    alphas = None
                else:
                    preds, alphas = [], []
                    for k in ctx.steps:
                        H, al = retrain_fit_apply(ctx, arr[:ctx.Etr, k - 1], ctx.Yp[:ctx.Etr, ctx.t + k], arr[ctx.Etr:, k - 1], seeds)
                        preds.append(H)
                        alphas.append(al)
                b = _bundle(ctx, preds, boot)
                if alphas is not None:
                    b["alpha_per_probe_seed"] = alphas
                res[probe][kind] = b
                log("analysis F: %s %s %s done" % (run, probe, kind))
            # tf - ol of every reported series (R2, MAE, level stats)
            a, o = res[probe]["teacher_forced"], res[probe]["open_loop"]
            d = {}
            for kn in ("r2", "mae"):
                d[kn] = {r: (np.array(a[kn][r], float) - np.array(o[kn][r], float)).tolist() for r in roles}
            d["q"] = {n: (np.array(a["q"][n], float) - np.array(o["q"][n], float)).tolist() for n in _q_names(roles)}
            res[probe]["tf_minus_ol"] = d
        # per-episode mean-over-probe-seed predictions for recomputation
        for probe in ("transfer", "retrained"):
            for kind in lat:
                store["%s|%s|%s" % (run, probe, kind)] = np.stack(res[probe][kind]["_pred_mean"]).astype(np.float32)        # (steps, n_te)
        for probe in ("transfer", "retrained"):
            for kind in lat:
                res[probe][kind].pop("_pred_mean")
        out["runs"][run] = res
    out["teacher_forcing_steps_available"] = steps_available
    # seed level summaries (per condition) and (a) - (b)
    quants = ["%s|%s" % (kn, r) for kn in ("r2", "mae") for r in roles] + _q_names(roles)
    summ = {}

    def per_seed(cond, probe, kind, qn):
        ts = [s for c, s, _ in B.all_runs(cfg) if c == cond]
        return np.array([_vec(out["runs"][B.run_name(cfg, cond, s)][probe][kind], qn) for s in ts])

    for cond in cfg["runs"]["conditions"]:
        summ[cond] = {}
        for probe in ("transfer", "retrained"):
            for kind in ("teacher_forced", "open_loop", "tf_minus_ol"):
                for qn in quants:
                    summ[cond]["%s|%s|%s" % (probe, kind, qn)] = jsonable_agg(L.seed_agg(per_seed(cond, probe, kind, qn)))
    out["seed_summary"] = summ
    a, b = cfg["runs"]["conditions"][1], cfg["runs"]["conditions"][0]
    diffs = {}
    for probe in ("transfer", "retrained"):
        for kind in ("teacher_forced", "open_loop", "tf_minus_ol"):
            for qn in ["r2|unseen", "r2|trained", "mae|unseen", "slope_episode|unseen", "intercept_episode|unseen"]:
                diffs["%s|%s|%s" % (probe, kind, qn)] = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                                         for k, v in L.diff_agg(per_seed(a, probe, kind, qn), per_seed(b, probe, kind, qn)).items()}
    out["a_minus_b"] = {"a": a, "b": b, "values": diffs}
    B.probe_output_dir(paths)
    B.write_npz(paths, Path(paths["results"]) / "F_per_episode_mean_pred.npz", lvl_idx=ctx.lvl_idx, nominal=ctx.nominal, **store)
    save_part(ctx, "analyze_f", out)
    return out
