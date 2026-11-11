"""Stages of experiment 013: check | gate | fit | measure | readout | aggregate | selftest.
gate     (CPU) stage 0: per-level ridge regressors on REAL encoder latents (Q1 cache), ratio table, passed flag -> <results>/stage0.json
fit      (CPU) erasers on encoder latents of the Q1 cache: scalar LEACE per train seed (009 fit_eraser, fit episodes = 009 rule), level means per level -> <cache>/fit/
measure  (GPU) per (condition, train seed) unit: one-step teacher-forced scores of the trained predictor and of the random-init predictor for 43 erasure configs on the
               test episodes of all 9 levels -> <results>/parts/unit_<cond>_s<ts>.{npz,json}      (refuses when the gate is not passed: exit 5, no attempt)
readout  (GPU) erasure check: 009-style linear size readout R2 at predictor blocks 0..L of the trained predictor with the input erasure applied -> parts/readout_*.json
aggregate(CPU) Delta_use with episode bootstrap, verdicts, stage 2, results.json
Resume: a finished unit (json written last) is skipped."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import base as B  # noqa: F401  (must come first: puts 002/004/009 src on sys.path)
from exp002 import cache as CA
from exp004 import deep as DP
from exp004 import stages as S4
from exp009 import curves as CU
from exp009 import erase as ER
from exp009 import stages as S9
from . import core as K
from . import lib as LB
from .base import log, read_json, write_json

BATCH_EPISODES = 8
METHODS = {"leace_scalar": "leace", "level_mean_match": "lmm"}


# ------------------------------------------------------------------------------------------------ data helpers
def rows_info(cfg2, C):
    rows = C["rows"]
    levels = list(cfg2["data"]["levels"])
    Etr = sum(r["part"] == "train" for r in rows)
    assert all(r["part"] == "train" for r in rows[:Etr]) and all(r["part"] == "test" for r in rows[Etr:])
    lvl = np.array([levels.index(r["level"]) for r in rows])
    return rows, Etr, levels, lvl


def grouped_actions(C):
    a = np.asarray(C["actions_raw"], np.float32)
    return a[:, :60].reshape(a.shape[0], 12, -1)


def patch_mean_cache(paths, C, fresh=False):
    """(E,13,dz) float32 patch-mean of the cached encoder tokens (DINO). Stored in <cache>/fit/pm.npy."""
    f = paths["cache"] / "fit" / "pm.npy"
    if f.exists() and not fresh:
        pm = np.load(f)
        if pm.shape[0] == len(C["rows"]):
            return pm
    T = C["tokens_dino"]
    out = np.zeros((T.shape[0], T.shape[1], T.shape[-1]), np.float32)
    t0 = time.time()
    for s in range(0, T.shape[0], 32):
        out[s:s + 32] = np.asarray(T[s:s + 32], np.float32).mean(2)
        if (s // 32) % 10 == 0:
            log("patch means %d/%d (%.0fs)" % (min(s + 32, T.shape[0]), T.shape[0], time.time() - t0))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, out)
    return out


def gate_path(paths):
    return paths["results_dir"] / "stage0.json"


# ------------------------------------------------------------------------------------------------ stage 0
def gate_stage(cfg, cfg2, paths, dry_run=False):
    s0 = cfg["stage0"]
    log("gate plan: levels %s, ridge alphas %s (002 probe_impl.ridge_alphas), inner folds %s (002 probe_impl.inner_folds), threshold %s in >= %s of 6 directed pairs" % (
        s0["levels"], cfg2["probe_impl"]["ridge_alphas"], cfg2["probe_impl"]["inner_folds"], s0["ratio_threshold"], s0["min_pairs"]))
    if dry_run:
        return None
    C = CA.load(B.B4.as002(paths), "")
    pm = patch_mean_cache(paths, C)
    rows = C["rows"]
    res = LB.stage0_gate(pm, grouped_actions(C), rows, list(s0["levels"]), [float(a) for a in cfg2["probe_impl"]["ridge_alphas"]], int(cfg2["probe_impl"]["inner_folds"]),
                         float(s0["ratio_threshold"]), int(s0["min_pairs"]))
    res.update({"stage": "stage0_gate", "gate_config_sha256": B.gate_config_sha(cfg), "cache_encoder_fingerprint": C["meta"]["encoder_fingerprint"],
                "inputs": "patch-mean z_t (384) + grouped raw action a_t (30) -> patch-mean (z_{t+1} - z_t) (384); frames 0..11; per-level ridge on the train-split episodes of the level, alpha by episode GroupKFold; errors on the test episodes (100 per level)",
                "created": time.strftime("%F %T"), "results_dir": str(paths["results_dir"])})
    write_json(res, gate_path(paths))
    log("gate: %d / %d pairs with ratio >= %s -> passed=%s" % (res["n_pairs_pass"], res["n_pairs"], s0["ratio_threshold"], res["passed"]))
    for p in res["pairs"]:
        log("  regressor %-5s test %-5s ratio %.3f %s" % (p["regressor_level"], p["test_level"], p["ratio_other_over_same"], "pass" if p["passes"] else ""))
    return res


def gate_status(cfg, paths):
    """(ok, message, record or None). Stage 1 (measure, readout) needs a passed gate computed with the current gate definition."""
    p = gate_path(paths)
    if not p.exists():
        return False, "stage-0 gate result missing (%s): run `python run.py gate` first" % p, None
    g = read_json(p)
    if g.get("gate_config_sha256") != B.gate_config_sha(cfg):
        return False, "stage-0 gate result %s is stale (gate definition in config.yaml changed): run `python run.py gate` again" % p, g
    if not g.get("passed"):
        return False, "stage-0 gate NOT PASSED (%s of %s pairs with ratio >= %s, needed %s): stage 1 is not run" % (g.get("n_pairs_pass"), g.get("n_pairs"), g.get("ratio_threshold"), g.get("min_pairs")), g
    return True, "gate passed (%s of %s pairs)" % (g.get("n_pairs_pass"), g.get("n_pairs")), g


# ------------------------------------------------------------------------------------------------ erasers on encoder latents
def fit_dir(paths):
    return paths["cache"] / "fit"


def compute_fit(cfg, cfg2, rows, Etr, lvl, pm, token_fn, train_seeds, P, dz):
    """Pure computation of all eraser fits (no I/O). token_fn(sorted_row_ids) -> (n, >=12, P, dz) cached encoder tokens.
    Returns dict(leace {ts: eraser dict}, fit_ep {ts: indices}, M, D, counts, fit_mask, eval_mask, ids {level: row ids}, levels)."""
    levels = list(cfg2["data"]["levels"])
    y_ep = np.array([r["nominal_log_area"] for r in rows])
    out = {"leace": {}, "fit_ep": {}, "levels": levels}
    for ts in train_seeds:
        fit_ep = LB.fit_episodes(cfg, ts, Etr)
        assert fit_ep.max() < Etr, "LEACE fit episodes must be train rows"
        out["leace"][ts] = LB.fit_leace(pm, y_ep, fit_ep, float(cfg["erasure"]["ridge_rel"]))
        out["fit_ep"][ts] = fit_ep
    rows_test = rows[Etr:]
    train_levels = list(cfg["data"]["judge_levels"])
    fmask, emask = LB.ood_halves(cfg, rows_test, levels, train_levels)
    ids = {}
    for l in levels:
        li = levels.index(l)
        if l in train_levels:
            ids[l] = np.where((lvl == li) & (np.arange(len(rows)) < Etr))[0]
        else:
            ids[l] = Etr + np.where(fmask & (lvl[Etr:] == li))[0]
    ood_ids = np.concatenate([ids[l] for l in levels if l not in train_levels]) - Etr
    assert not emask[ood_ids].any(), "level-mean fit episodes overlap the evaluation half"
    assert all((ids[l] < Etr).all() for l in train_levels)
    Mn, cnt = LB.level_means(token_fn, ids, levels, P, dz)
    out.update({"M": Mn, "D": LB.level_mean_shifts(Mn), "counts": cnt, "fit_mask": fmask, "eval_mask": emask, "ids": ids})
    return out


def fit_stage(cfg, cfg2, paths, train_seeds, fresh=False, dry_run=False):
    log("fit plan: scalar LEACE per train seed %s (009 fit_eraser, ridge_rel %s, fit_fraction %s, fit seed %s + train seed, on the Q1-cache TRAIN rows, 12 input frames); level means of all 9 levels "
        "(train levels: all train-split episodes; OoD levels: fit half of the 100 test episodes, seed %s)" % (
            train_seeds, cfg["erasure"]["ridge_rel"], cfg["erasure"]["fit_fraction"], cfg["erasure"]["fit_seed"], cfg["erasure"]["fit_seed"]))
    if dry_run:
        return None
    C = CA.load(B.B4.as002(paths), "")
    rows, Etr, levels, lvl = rows_info(cfg2, C)
    pm = patch_mean_cache(paths, C, fresh)
    fd = fit_dir(paths)
    fd.mkdir(parents=True, exist_ok=True)
    F = compute_fit(cfg, cfg2, rows, Etr, lvl, pm, lambda i: C["tokens_dino"][np.sort(i)], train_seeds, K.M.P_TOK, K.M.D_VIS)
    summ = {"leace": {}, "level_mean_match": {}}
    for ts in train_seeds:
        f = fd / ("leace_s%d.npz" % ts)
        if fresh or not f.exists():
            ER.save_eraser(F["leace"][ts], f)
        E = F["leace"][ts]
        summ["leace"][str(ts)] = {"n_fit_episodes": int(len(F["fit_ep"][ts])), "fit_episode_rows": F["fit_ep"][ts].tolist(), "corr_fit_readout_vs_label": E["corr_fit"], "c_norm": float(np.linalg.norm(E["c"]))}
        log("LEACE train seed %d: %d fit episodes (train rows), corr(a.x, y) on fit data %.3f" % (ts, len(F["fit_ep"][ts]), E["corr_fit"]))
    if fresh or not (fd / "level_means.npz").exists():
        np.savez_compressed(fd / "level_means.npz", M=F["M"], D=F["D"], counts=np.array(F["counts"]), fit_mask=F["fit_mask"], eval_mask=F["eval_mask"])
    train_levels = list(cfg["data"]["judge_levels"])
    summ["level_mean_match"] = {"levels": levels, "n_fit_episodes_by_level": dict(zip(levels, F["counts"])), "overall_mean": "equal-weighted mean of the 9 level means (per patch position)",
                                "ood_fit_half_test_episode_index": {l: np.where(F["fit_mask"] & (lvl[Etr:] == levels.index(l)))[0].tolist() for l in levels if l not in train_levels},
                                "ood_eval_half_test_episode_index": {l: np.where(F["eval_mask"] & (lvl[Etr:] == levels.index(l)))[0].tolist() for l in levels if l not in train_levels},
                                "mean_shift_norm_by_level": {l: float(np.linalg.norm(F["D"][i], axis=-1).mean()) for i, l in enumerate(levels)}}
    summ["encoder_fingerprint_cache"] = C["meta"]["encoder_fingerprint"]
    write_json(summ, paths["results_dir"] / "fit.json")
    log("fit done: level-mean fit episodes per level %s" % dict(zip(levels, F["counts"])))
    return summ


def load_erasers(paths, ts, cfg, cfg2, dev):
    """(Eraser torch object, fit_mask, eval_mask, leace dict)."""
    fd = fit_dir(paths)
    leace = ER.load_eraser(fd / ("leace_s%d.npz" % ts))
    z = np.load(fd / "level_means.npz")
    R = LB.random_dirs(int(cfg["erasure"]["random_directions"]), K.M.D_VIS, int(cfg["erasure"]["random_dir_seed"]))
    S = LB.random_signs(int(cfg["erasure"]["random_directions"]), len(cfg2["data"]["levels"]), K.M.P_TOK, int(cfg["erasure"]["random_dir_seed"]))
    return K.Eraser(leace, z["D"], R, S, dev), z["fit_mask"], z["eval_mask"], leace


# ------------------------------------------------------------------------------------------------ preflight
def run_problems(paths, conds, train_seeds):
    bad = []
    out_dir = paths["train_base"] / "outputs"
    for cond in conds:
        for ts in train_seeds:
            rd = B.run_dir(paths, cond, ts)
            if not rd.is_dir():
                near = sorted(p.name for p in out_dir.iterdir() if ("s%d" % ts) in p.name)[:6] if out_dir.is_dir() else []
                bad.append("%s seed %d: run directory %s missing (expected name %s under %s; directories there with s%d: %s)" % (cond, ts, rd, rd.name, out_dir, ts, near))
                continue
            for rel in ("DONE", "hydra.yaml", "train_manifest.json", "checkpoints/model_latest.pth"):
                if not (rd / rel).exists():
                    bad.append("%s seed %d: %s missing" % (cond, ts, rd / rel))
    return bad


def input_preflight(paths, conds, train_seeds, need_fit=True):
    bad = run_problems(paths, conds, train_seeds)
    qd = CA.cache_dir(B.B4.as002(paths), "") / "DONE"
    if not qd.exists():
        bad.append("002 Q1 cache DONE missing (%s)" % qd)
    bad += S9.metadata_missing([("%s seed %d" % (c, ts), B.run_dir(paths, c, ts) / "train_manifest.json") for c in conds for ts in train_seeds])
    if need_fit:
        fd = fit_dir(paths)
        for ts in train_seeds:
            if not (fd / ("leace_s%d.npz" % ts)).exists():
                bad.append("eraser fit missing: %s (run `python run.py fit`)" % (fd / ("leace_s%d.npz" % ts)))
        if not (fd / "level_means.npz").exists():
            bad.append("eraser fit missing: %s (run `python run.py fit`)" % (fd / "level_means.npz"))
    return bad


# ------------------------------------------------------------------------------------------------ units
def unit_name(cond, ts):
    return "%s_s%d" % (cond, ts)


def part_json(paths, kind, cond, ts):
    return paths["results_dir"] / "parts" / ("%s_%s.json" % (kind, unit_name(cond, ts)))


def part_npz(paths, cond, ts):
    return paths["results_dir"] / "parts" / ("unit_%s.npz" % unit_name(cond, ts))


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def select_test_episodes(rows_test, per_level):
    """Indices (into the test episodes) evaluated: all, or the first `per_level` (by rank_in_level) of every level (pilot)."""
    if per_level is None:
        return np.arange(len(rows_test))
    return np.array([j for j, r in enumerate(rows_test) if int(r["rank_in_level"]) < int(per_level)])


def load_model_for(paths, cfg2, cond, ts, dev, C):
    rdir = B.run_dir(paths, cond, ts)
    model, tcfg, ep = K.M.load_world_model(paths["acwm_src"], rdir, dev)
    fp = K.M.encoder_fingerprint(model.encoder)
    if fp != C["meta"]["encoder_fingerprint"]:
        raise RuntimeError("encoder of %s differs from the Q1 cache encoder: the erasers fitted on cached encoder latents are not valid for this run" % rdir)
    assert int(tcfg.num_hist) == 1, "013 assumes num_hist == 1"
    mean, std, _ = K.M.action_stats_for_run(cfg2, rdir, tcfg)
    return model, tcfg, ep, fp, mean, std, rdir


def compute_unit(cfg, cfg2, C, model, rnd, mean, std, er, dev, ep_sel, bs, label=""):
    """Scores of all configs for the trained and the random-init predictor on the test episodes ep_sel (indices into the test episodes).
    Returns (arrays dict for the npz, first-batch max |encoder - cache| ), no I/O."""
    import torch
    rows, Etr, levels, lvl = rows_info(cfg2, C)
    fns = {"trained": lambda z: model.predict(z), "random_init": lambda z: rnd.predict(z)}
    keys = K.config_keys(int(cfg["erasure"]["random_directions"]))
    acc = {k: {key: [] for key in keys} for k in fns}
    dh = {k: [] for k in fns}
    dr = []
    first_diff = None
    t0 = time.time()
    for s in range(0, len(ep_sel), bs):
        sel = ep_sel[s:s + bs]
        ids = Etr + sel
        x = K.M.frames_to_float(np.asarray(C["frames_u8"][ids])).to(dev)
        acts = K.M.acts_from_raw(torch.as_tensor(np.asarray(C["actions_raw"][ids])), mean, std).to(dev)
        lv = torch.as_tensor(lvl[ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            z_all = K.M.encode_batch(model, x, acts)
            if first_diff is None:
                first_diff = float((z_all[..., :K.M.D_VIS].float().cpu() - torch.as_tensor(np.asarray(C["tokens_dino"][ids], np.float32))).abs().max())
                log("%s first batch: max |encode_batch visual - cached tokens_dino| = %.3g (cache is fp16)" % (label, first_diff))
            o, d, r = K.eval_batch(z_all, lv, fns, er, keys)
        for k in fns:
            for key in keys:
                acc[k][key].append(o[k][key])
            dh[k].append(d[k])
        dr.append(r)
        if (s // bs) % 20 == 0:
            log("%s: %d/%d episodes (%.0fs)" % (label, min(s + bs, len(ep_sel)), len(ep_sel), time.time() - t0))
    arrs = {"metrics_%s" % k: np.stack([np.concatenate(acc[k][key], 0) for key in keys], 0) for k in fns}
    arrs.update({"dhat_%s" % k: np.concatenate(dh[k], 0).astype(np.float32) for k in fns})
    arrs["dreal"] = np.concatenate(dr, 0).astype(np.float32)
    return arrs, first_diff


def measure_unit(cfg, cfg2, paths, cond, ts, C, dev, ep_sel, bs, fresh=False, gate_overridden=False, per_level=None):
    jp = part_json(paths, "unit", cond, ts)
    if jp.exists() and not fresh:
        log("unit %s done, skip" % unit_name(cond, ts))
        return None
    t0 = time.time()
    rows, Etr, levels, lvl = rows_info(cfg2, C)
    model, tcfg, ep, fp, mean, std, rdir = load_model_for(paths, cfg2, cond, ts, dev, C)
    er, fmask, emask, leace = load_erasers(paths, ts, cfg, cfg2, dev)
    rnd = DP.random_init_predictor(model, int(cfg["predictor"]["random_init_seed"]) + ts)
    keys = K.config_keys(int(cfg["erasure"]["random_directions"]))
    log("unit %s: run %s epoch %d, %d test episodes, %d configs x 2 predictors" % (unit_name(cond, ts), rdir.name, ep, len(ep_sel), len(keys)))
    arrs, first_diff = compute_unit(cfg, cfg2, C, model, rnd, mean, std, er, dev, ep_sel, bs, label="unit " + unit_name(cond, ts))
    pp = part_npz(paths, cond, ts)
    pp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pp, **arrs)
    meta = unit_meta(cfg, rows, Etr, levels, lvl, fmask, emask, ep_sel, keys)
    meta.update({"condition": cond, "train_seed": ts, "run": rdir.name, "epoch": ep, "checkpoint": cfg["base"]["checkpoint"], "checkpoint_sha256": sha256_file(rdir / "checkpoints" / "model_latest.pth"),
                 "random_init_seed": int(cfg["predictor"]["random_init_seed"]) + ts, "encoder_fingerprint": fp, "cache_encoder_fingerprint": C["meta"]["encoder_fingerprint"],
                 "encoder_matches_cache": fp == C["meta"]["encoder_fingerprint"], "first_batch_max_abs_diff_encoder_vs_cache": first_diff, "batch_episodes": bs,
                 "leace_fit_corr": leace["corr_fit"], "gate_overridden": bool(gate_overridden), "episodes_per_level": per_level, "seconds": round(time.time() - t0, 1),
                 "finished": time.strftime("%F %T"), "device": str(dev)})
    write_json(meta, jp)                                         # written last = done marker
    log("unit %s done in %.0fs" % (unit_name(cond, ts), time.time() - t0))
    return meta


def unit_meta(cfg, rows, Etr, levels, lvl, fmask, emask, ep_sel, keys):
    return {"keys": keys, "metrics": K.METRICS, "kinds": ["trained", "random_init"], "test_episode_index": ep_sel.tolist(), "level_idx": lvl[Etr + ep_sel].tolist(), "levels": levels,
            "roles": [rows[Etr + j]["role"] for j in ep_sel], "ood_eval_half_mask": emask[ep_sel].tolist(), "ood_fit_half_mask": fmask[ep_sel].tolist(), "n_test_episodes_total": len(rows) - Etr}


def plan_text(conds, seeds, per_level, paths, what):
    done = [(c, t) for c in conds for t in seeds if part_json(paths, what, c, t).exists()]
    return "%s plan: conditions %s, train seeds %s, units %d (done %d), episodes per level %s" % (what, conds, seeds, len(conds) * len(seeds), len(done), per_level if per_level else "all (100)")


def measure_stage(cfg, cfg2, paths, conds, seeds, per_level, bs, fresh=False, dry_run=False, gate_overridden=False):
    log(plan_text(conds, seeds, per_level, paths, "unit") + "; configs per unit %d (none + 2 methods x (size + %d random)), predictors: trained + random-init" % (
        len(K.config_keys(int(cfg["erasure"]["random_directions"]))), int(cfg["erasure"]["random_directions"])))
    if dry_run:
        return None
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(B.B4.as002(paths), "")
    _, Etr, _, _ = rows_info(cfg2, C)
    ep_sel = select_test_episodes(C["rows"][Etr:], per_level)
    for cond in conds:
        for ts in seeds:
            measure_unit(cfg, cfg2, paths, cond, ts, C, dev, ep_sel, bs, fresh, gate_overridden, per_level)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return True


# ------------------------------------------------------------------------------------------------ readout (erasure check)
def readout_method(cfg2, c9, model, C, mean, std, dev, bs, er, m, lvl, tmp, ts, label=""):
    """Erasure check for one method m in none | leace_scalar | level_mean_match: 009-style linear size readout R2 at blocks 0..L (mean over probe seeds), per split."""
    frames, pos_windows = S4.frames_and_windows(cfg2)
    rows, Etr, levels, _ = rows_info(cfg2, C)
    _, _, te, Yp = S9.labels_positions(cfg2, C, frames)
    L = K.M.n_blocks(model)
    primary = c9["depth"]["primary"]
    kinds, pseeds = cfg2["probe"]["kinds"], cfg2["probe"]["seeds"]
    key = None if m == "none" else ("leace:size" if m == "leace_scalar" else "lmm:size")
    t0 = time.time()
    K.extract_erased(model, C, 1, tmp, mean, std, dev, frames, K.M.D_VIS, list(range(L + 1)), er, key, lvl, bs, label=label)
    r = S9.probe_rows(cfg2, kinds, pseeds, tmp, range(L + 1), Yp, Etr, te, levels, pos_windows, {"train_seed": ts, "m_int": "NA", "alpha": 0.0})
    shutil.rmtree(tmp, ignore_errors=True)
    return {"r2_by_split_depth": {sp: {str(d): v for d, v in CU.seed_curve(r, ts, sp, primary).items()} for sp in ("train", "interp", "extrap")},
            "primary_probe": dict(primary), "probe_seeds": list(pseeds), "n_blocks": L, "seconds": round(time.time() - t0, 1)}


def readout_unit(cfg, c9, cfg2, paths, cond, ts, C, dev, bs, fresh=False):
    jp = part_json(paths, "readout", cond, ts)
    done = read_json(jp) if (jp.exists() and not fresh) else {"condition": cond, "train_seed": ts, "methods": {}}
    todo = [m for m in ("none", "leace_scalar", "level_mean_match") if m not in done["methods"]]
    if not todo:
        log("readout %s done, skip" % unit_name(cond, ts))
        return None
    rows, Etr, levels, lvl = rows_info(cfg2, C)
    model, tcfg, ep, fp, mean, std, rdir = load_model_for(paths, cfg2, cond, ts, dev, C)
    er, fmask, emask, _ = load_erasers(paths, ts, cfg, cfg2, dev)
    tmp = paths["cache"] / ("tmp_readout_%s" % unit_name(cond, ts))
    for m in todo:
        done["methods"][m] = readout_method(cfg2, c9, model, C, mean, std, dev, bs, er, m, lvl, tmp, ts, label="readout %s %s" % (unit_name(cond, ts), m))
        write_json(done, jp)
        log("readout %s %s done (%.0fs)" % (unit_name(cond, ts), m, done["methods"][m]["seconds"]))
    done.update({"run": rdir.name, "epoch": ep, "encoder_matches_cache": fp == C["meta"]["encoder_fingerprint"],
                 "note": "linear size readout (ridge, 009 probe) of the z part of the last-context-frame tokens at block m; erasure applied to the predictor input z part before the positional embedding; "
                         "level_mean_match test-episode OoD levels: the fit-half episodes are included among the probe's test episodes (CHANGES.md item 9)"})
    write_json(done, jp)
    return done


def readout_stage(cfg, c9, cfg2, paths, conds, seeds, bs, fresh=False, dry_run=False):
    log(plan_text(conds, seeds, None, paths, "readout") + "; methods none / leace_scalar / level_mean_match, depths 0..L (L from the model), probe kinds %s seeds %s" % (cfg2["probe"]["kinds"], cfg2["probe"]["seeds"]))
    if dry_run:
        return None
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(B.B4.as002(paths), "")
    for cond in conds:
        for ts in seeds:
            readout_unit(cfg, c9, cfg2, paths, cond, ts, C, dev, bs, fresh)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return True


# ------------------------------------------------------------------------------------------------ check
def check(cfg, c9, cfg2, paths, conds, seeds):
    print("cache dir (fit, tmp features):", paths["cache"], "exists=%s" % paths["cache"].exists())
    res = B.consistency_checks(cfg, c9, cfg2)
    for n, ok, d in res:
        print("  [%s] %s %s" % ("ok" if ok else "FAIL", n, "" if ok else d))
    bad = input_preflight(paths, conds, seeds, need_fit=False)
    print("inputs (checkpoints of %s x seeds %s, Q1 cache, metadata.pt): %s" % (conds, seeds, "all present" if not bad else "MISSING %d" % len(bad)))
    for b in bad[:20]:
        print("  !! " + b)
    ok, msg, _ = gate_status(cfg, paths)
    print("stage-0 gate:", msg)
    fd = fit_dir(paths)
    print("erasers fitted:", all((fd / ("leace_s%d.npz" % t)).exists() for t in seeds) and (fd / "level_means.npz").exists())
    return 0 if (not bad and all(ok_ for _, ok_, _ in res)) else 1


# ------------------------------------------------------------------------------------------------ aggregate
def pilot_reasons(cfg, results_dir, conds_over, seeds_over, per_level, gate_overridden, metas=()):
    default_dir = B.default_results_dir().resolve()
    r = []
    if cfg["pilot"]:
        r.append("config.yaml pilot flag is true")
    if Path(results_dir).resolve() != default_dir:
        r.append("results dir %s is not the experiment results dir %s (pilot / separate run)" % (Path(results_dir).resolve(), default_dir))
    if conds_over:
        r.append("conditions overridden on the command line")
    if seeds_over:
        r.append("train seeds overridden on the command line")
    if per_level is not None:
        r.append("only the first %s test episodes per level were evaluated (--episodes-per-level)" % per_level)
    if gate_overridden or any(m.get("gate_overridden") for m in metas):
        r.append("stage-0 gate not passed and stage 1 was run with --override-gate")
    if per_level is None and any(m.get("episodes_per_level") is not None for m in metas):
        r.append("some units evaluated only the first N test episodes per level (--episodes-per-level)")
    return r


def sets_for(meta, method):
    """Evaluation sets (name -> bool mask over the measured episodes). train = test episodes of the training levels (the judged set). interp / extrap: leace on all measured
    episodes of the role, level_mean_match on the held-out half only (fit half excluded); leace_*_heldhalf: leace restricted to the same held-out half."""
    roles = np.array(meta["roles"])
    ev = np.array(meta["ood_eval_half_mask"], bool)
    out = {"train": roles == "trained"}
    for r in ("interp", "extrap"):
        m = roles == r
        out[r] = m & ev if method == "level_mean_match" else m
        if method == "leace_scalar":
            out[r + "_heldhalf"] = m & ev
    return out


def aggregate(cfg, c9, cfg2, paths, conds, seeds, conds_over, seeds_over, per_level, gate_overridden, dry_run=False):
    log("aggregate: conditions %s seeds %s" % (conds, seeds))
    if dry_run:
        return None
    succ = cfg["success"]
    Kr = int(cfg["erasure"]["random_directions"])
    levels = list(cfg2["data"]["levels"])
    ok_gate, gmsg, grec = gate_status(cfg, paths)
    units, missing, ro = {}, [], {}
    for c in conds:
        for ts in seeds:
            jp = part_json(paths, "unit", c, ts)
            if jp.exists() and part_npz(paths, c, ts).exists():
                units[(c, ts)] = (read_json(jp), dict(np.load(part_npz(paths, c, ts))))
            else:
                missing.append("unit_" + unit_name(c, ts))
            rp = part_json(paths, "readout", c, ts)
            if rp.exists():
                ro[(c, ts)] = read_json(rp)
    boot_cache = {}
    nb, bseed, ci = int(succ["bootstrap"]["n"]), int(succ["bootstrap"]["seed"]), float(succ["ci"])

    def stats_for(meta, Mx, method, kind, sname, sel):
        keys = meta["keys"]
        ix = {k: i for i, k in enumerate(keys)}
        pre = METHODS[method]
        A = Mx["metrics_" + kind]
        g = np.array(meta["level_idx"])[sel]
        bk = (tuple(g.tolist()))
        if bk not in boot_cache:
            boot_cache[bk] = LB.boot_indices(g, nb, bseed)
        col = lambda key, m: A[ix[key]][sel][:, K.MI[m]]
        st = {}
        for mname, mcol in (("motion", "motion"), ("total", "total")):
            sl = {"none": col("none", mcol), "size": col(pre + ":size", mcol), "rand": np.stack([col("%s:rand:%d" % (pre, k), mcol) for k in range(Kr)], 0), "persist": col("none", "persist")}
            st[mname] = LB.use_stats(sl, boot_cache[bk], ci)
        n_ = {"mean_" + m: {"none": float(col("none", m).mean()), "size": float(col(pre + ":size", m).mean()), "rand_mean": float(np.mean([col("%s:rand:%d" % (pre, k), m).mean() for k in range(Kr)]))} for m in K.METRICS}
        st["abs_means_over_episodes"] = n_
        sz = col(pre + ":size", "disp_norm_mean").mean()
        rn = np.mean([col("%s:rand:%d" % (pre, k), "disp_norm_mean").mean() for k in range(Kr)])
        st["norm_matching"] = {"size_mean_token_displacement_norm": float(sz), "random_mean_token_displacement_norm": float(rn), "ratio_random_over_size": float(rn / sz) if sz > 0 else None,
                               "max_abs_diff_per_episode": float(max(np.abs(col("%s:rand:%d" % (pre, k), "disp_norm_mean") - col(pre + ":size", "disp_norm_mean")).max() for k in range(Kr)))}
        return st

    stage1 = {"conditions": {}}
    for c in conds:
        cres = {"seeds": {}}
        for ts in seeds:
            if (c, ts) not in units:
                continue
            meta, Mx = units[(c, ts)]
            sres = {"meta": {k: meta[k] for k in ("run", "epoch", "checkpoint_sha256", "random_init_seed", "encoder_matches_cache", "first_batch_max_abs_diff_encoder_vs_cache", "n_episodes_evaluated") if k in meta}}
            sres["meta"]["n_episodes_evaluated"] = len(meta["test_episode_index"])
            for kind in meta["kinds"]:
                kres = {}
                for method in cfg["erasure"]["methods"]:
                    kres[method] = {sn: stats_for(meta, Mx, method, kind, sn, sel) for sn, sel in sets_for(meta, method).items() if sel.sum() > 0}
                sres[kind] = kres
            lvl = np.array(meta["level_idx"])
            sres["stage2_level_mean_delta_z"] = {kind: LB.level_mean_comparison(Mx["dhat_" + kind].astype(np.float64), Mx["dreal"].astype(np.float64), lvl, len(levels)) for kind in meta["kinds"]}
            cres["seeds"][str(ts)] = sres
        stage1["conditions"][c] = cres
    # verdicts (trained predictor, train-level set, motion error normalised by persistence)
    verdicts = {}
    b_name, a_name = "multi_60_ind_160", "single_ind"
    for method in cfg["erasure"]["methods"]:
        get = lambda cond: {ts: (stage1["conditions"].get(cond, {}).get("seeds", {}).get(str(ts), {}).get("trained", {}).get(method, {}).get("train", {}).get("motion")) for ts in seeds}
        verdicts[method] = LB.judge_method(get(b_name), get(a_name), succ, seeds) if (b_name in conds and a_name in conds) else {"status": LB.NOT_EVAL, "reason": "both conditions are needed"}
        verdicts[method]["judged"] = "trained predictor, test episodes of the training levels (a060, ind, a160), motion error / persistence error, Delta_use = size increase - mean random increase"
    ref = {}
    for c in conds:
        ref[c] = {str(ts): {m: stage1["conditions"][c]["seeds"].get(str(ts), {}).get("random_init", {}).get(m, {}).get("train", {}).get("motion", {}).get("delta_use")
                            for m in cfg["erasure"]["methods"]} for ts in seeds}
    readout = {}
    for (c, ts), d in ro.items():
        readout.setdefault(c, {})[str(ts)] = d["methods"]
    from exp002.common import env_info
    reasons = pilot_reasons(cfg, paths["results_dir"], conds_over, seeds_over, per_level, gate_overridden, [m for m, _ in units.values()])
    res = {"experiment": cfg["name"], "pilot": bool(reasons), "pilot_reasons": reasons, "pilot_config_flag": bool(cfg["pilot"]),
           "conditions": conds, "train_seeds": seeds, "episodes_per_level_override": per_level,
           "gate": {"ok": ok_gate, "message": gmsg, "overridden": bool(gate_overridden), "record": grec},
           "fit": read_json(paths["results_dir"] / "fit.json") if (paths["results_dir"] / "fit.json").exists() else None,
           "erasers_shared_across_conditions": {"encoder_matches_cache_all_units": all(m["encoder_matches_cache"] for m, _ in units.values()) if units else None,
                                                "encoder_fingerprints": sorted({m["encoder_fingerprint"] for m, _ in units.values()}),
                                                "note": "frozen DINOv2 encoder: erasers are fitted once on cached encoder latents; every checkpoint's encoder fingerprint is compared with the cache's at run time"},
           "metric_definitions": K.__doc__.split("Definitions")[1].split("Without erasure")[0].strip(), "success": dict(succ),
           "stage1": stage1, "verdict": verdicts, "random_init_delta_use_reference": ref, "erasure_check_readout_r2": readout,
           "missing_units": missing, "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(),
           "environment": env_info(), "code_version": B.code_version(paths["exp"]), "interpretation": None}
    write_json(res, paths["results_dir"] / "results.json")
    if Path(paths["results_dir"]).resolve() == B.default_results_dir().resolve():
        write_json(res, paths["exp"] / "results.json")
    log("aggregate done (pilot=%s): verdicts %s" % (res["pilot"], {m: v["status"] for m, v in verdicts.items()}))
    return res
