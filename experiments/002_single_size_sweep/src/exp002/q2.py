"""Q2 (H-b): is size information preserved in the predictor? GPU extraction + probes, one trained run at a time.
For each block output j (0 = predictor input stream ... L = output of the last block) the residual stream of the LAST context frame is read
(teacher-forced windows at frames frames.early_flat U frames.late_deformed) and probed with the same probes as Q1. Rollout: from the last
context frame t = num_hist-1, autoregressive rollout; predicted latents of frames t+1..t+K are probed (probe trained on the same step).
Reference lines come from q1 (Q1 ceiling = same probe on encoder tokens, matched frames; layer 0 = baseline) and are compared in aggregate.py.
Layer x time grids are EXPLORATORY; only (last block, last rollout step) enters the registered H-b test.
OUTPUT-SPACE measurement (확정 기록 6 4a, main for rollout): the Q1 probe (trained on ENCODER tokens only, regime 'all') is applied unchanged to the predicted
z of rollout steps t+1..; reference = same probe on the encoder tokens of the same frames and episodes, and on the trained-size test episodes (role 'trained').
No probe is ever trained on predictor outputs. Layer-wise probes (retrained on activations) are (b)'s main result; for (a) they are auxiliary (README limitation).
Diagnostics 4c (diag.py): activation shift of a060/a160 vs ind per layer, noise floor from ind halves, next-frame prediction error ratio.
Temporary feature files live in $EXP002_CACHE/q2_tmp_<run> and are deleted at the end (config q2_impl.keep_tmp)."""
from __future__ import annotations

import shutil
import time

import numpy as np

from .common import log, write_json, read_json, require, run_dir, run_name as mk_run_name
from . import cache as CA
from . import evalgrid as EG
from . import model_io as M
from . import stats as ST
from . import diag as DG

LAYERWISE_LIMITATION = ("Layer-wise probes are trained on predictor activations of the 3 sizes a060/ind/a160. In condition (a) (single_ind) the label is constant during "
                        "predictor training, so the probe reads activations of sizes the predictor never saw. For (a) only the change relative to layer 0 is reported, "
                        "no absolute comparison with (b).")


def plan(cfg, paths, cond, seed):
    H = require(cfg, "train.num_hist")
    W = EG.windows(cfg)
    nF = len(W["union"])
    E = cfg["data"]["train_episodes_per_level"] * len(cfg["data"]["train_levels_b"]) + cfg["data"]["eval_episodes_per_level"] * len(cfg["data"]["levels"])
    d = 404 if cfg["q2_impl"]["feature_part"] == "full" else 384
    L = 6
    gb = E * nF * 196 * d * 2 * (L + 1) / 1e9 + E * cfg["q2_impl"]["rollout_horizon"] * 196 * 384 * 2 / 1e9
    return {"run": mk_run_name(cond, seed), "num_hist": H, "frames": W["union"], "tmp_disk_gb_assuming_6_blocks": round(gb, 1), "feature_part": cfg["q2_impl"]["feature_part"]}


def extract(cfg, paths, cond, seed, tmp):
    import torch
    name = mk_run_name(cond, seed)
    H = require(cfg, "train.num_hist")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(paths)
    rdir = run_dir(paths, name)
    model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
    fp = M.encoder_fingerprint(model.encoder)
    if fp != C["meta"]["encoder_fingerprint"]:
        raise RuntimeError("encoder of %s differs from the Q1 cache encoder (train_encoder=true?) - Q2 would not be comparable to Q1" % name)
    assert int(tcfg.num_hist) == int(H), "checkpoint num_hist %s != config train.num_hist %s" % (tcfg.num_hist, H)
    mean, std, norm = M.action_stats_for_run(cfg, rdir, tcfg)
    L = M.n_blocks(model)
    log("q2 %s: epoch %d, %d predictor blocks, num_hist %d, normalize_action %s" % (name, ep, L, H, norm))
    W = EG.windows(cfg)
    frames = W["union"]
    assert min(frames) >= H - 1, "frames %s need context of %d frames" % (frames, H)
    t = H - 1
    horizon = min(int(cfg["q2_impl"]["rollout_horizon"]), 12 - t)
    rows = C["rows"]
    E = len(rows)
    dpart = 404 if cfg["q2_impl"]["feature_part"] == "full" else 384
    tmp.mkdir(parents=True, exist_ok=True)
    feats = [np.lib.format.open_memmap(tmp / ("feat_%d.npy" % j), mode="w+", dtype=np.float16, shape=(E, len(frames), 196, dpart)) for j in range(L + 1)]
    roll = np.lib.format.open_memmap(tmp / "roll.npy", mode="w+", dtype=np.float16, shape=(E, horizon, 196, 384))
    next_err = np.zeros((E, len(frames)))
    hooks = M.StreamHooks(model).install()
    bs = int(cfg["q2_impl"]["batch_episodes"])
    t0 = time.time()
    for s in range(0, E, bs):
        sl = slice(s, min(E, s + bs))
        x = M.frames_to_float(np.asarray(C["frames_u8"][sl])).to(dev)
        raw = torch.as_tensor(np.asarray(C["actions_raw"][sl]))
        acts = M.acts_from_raw(raw, mean, std).to(dev)
        with torch.no_grad():
            z_all = M.encode_batch(model, x, acts)
            zw = M.make_windows(z_all, H, frames)
            hooks.reset()
            hooks.capture_layers = set(range(L + 1))
            o_tf = model.predict(zw)
            n = x.shape[0]
            pred_next = o_tf.reshape(n, len(frames), H, 196, -1)[:, :, -1, :, :384].float()
            tgt_next = z_all[:, [f + 1 for f in frames], :, :384].float()
            next_err[sl] = ((pred_next - tgt_next) ** 2).mean((-1, -2)).cpu().numpy()
            for j in range(L + 1):
                tok = M.last_frame_tokens(hooks.store[j], H, dpart).reshape(n, len(frames), 196, dpart)
                feats[j][sl] = tok.half().cpu().numpy()
            hooks.reset()
            roll[sl] = M.rollout(model, z_all, acts, t, horizon, H).half().cpu().numpy()
        log("q2 extract %d/%d (%.0fs)" % (min(E, s + bs), E, time.time() - t0))
    hooks.remove()
    for m in feats + [roll]:
        m.flush()
    np.save(tmp / "next_err.npy", next_err)
    return {"run": name, "epoch": ep, "n_blocks": L, "num_hist": H, "t": t, "horizon": horizon, "frames": frames, "feature_part": cfg["q2_impl"]["feature_part"],
            "encoder_fingerprint": fp, "normalize_action": norm}


def output_space(cfg, tokens, Yp, Etr, R_te, t, horizon):
    """Pure function (testable on CPU). tokens (E,13,196,384) encoder tokens (train rows first), Yp (E,13) primary labels, R_te (Ete,horizon,196,384)
    predicted z of the TEST episodes only (training rows of the rollout are not an input, so they cannot be fitted on)."""
    prim = cfg["q1"]["primary_target"]
    frames_tr = EG.windows(cfg)[cfg["q2_impl"]["output_probe_train_regime"]]
    cells, pms, digs = {}, {}, {}
    for kind in cfg["probe"]["kinds"]:
        sets = {}
        for k in range(1, horizon + 1):
            sets["encoder|step%d" % k] = (tokens[Etr:, t + k], Yp[Etr:, t + k])
            sets["rollout|step%d" % k] = (R_te[:, k - 1], Yp[Etr:, t + k])
        c, p, d = EG.fit_apply_cells(cfg, kind, tokens[:Etr], Yp[:Etr], frames_tr, sets, cfg["probe"]["seeds"])
        for n in sets:
            cells["%s|%s|OUT|%s" % (prim, kind, n)] = c[n]
            pms["%s|%s|OUT|%s" % (prim, kind, n)] = p[n]
        digs[kind] = d
    return cells, pms, digs


def diagnostics(cfg, tmp, te_rows, Etr, L):
    """4c: per-episode activation moments of ind / a060 / a160 test episodes per layer + next-frame error -> diag.evaluate."""
    dq, dc = cfg["q2_diag"], cfg["diag_4c"]
    levels = [dq["ref_level"]] + list(dq["levels"])
    idx = {l: [Etr + i for i, r in enumerate(te_rows) if r["level"] == l] for l in levels}
    allidx = [i for l in levels for i in idx[l]]
    pos = {i: n for n, i in enumerate(allidx)}
    layer_stats = {}
    for j in range(L + 1):
        F = np.load(tmp / ("feat_%d.npy" % j), mmap_mode="r")
        layer_stats[j] = [DG.episode_stats(F[i]) for i in allidx]
        del F
    ne = np.load(tmp / "next_err.npy")
    nerr = {l: ne[idx[l]].mean(1) for l in levels}
    li = {l: [pos[i] for i in idx[l]] for l in levels}
    res = DG.evaluate(layer_stats, li[dq["ref_level"]], {l: li[l] for l in dq["levels"]}, nerr, dc["noise_floor_bootstrap"], dc["bootstrap_seed"], dc)
    res["status"] = "decided_by_rule"
    res["note"] = "ind = test episodes of level ind; noise floor uses ind only. next_frame_error = teacher-forced MSE of the predicted last-frame visual latent vs the encoder latent of the next frame"
    return res


def run(cfg, paths, cond, seed, dry_run=False):
    name = mk_run_name(cond, seed)
    if dry_run:
        log("q2 plan: %s" % plan(cfg, paths, cond, seed))
        return None
    tmp = paths["cache"] / ("q2_tmp_" + name)
    outdir = paths["results_dir"] / name
    meta = extract(cfg, paths, cond, seed, tmp)
    C = CA.load(paths)
    rows = C["rows"]
    tr_rows, te_rows = EG.split_rows(rows)
    Etr, Ete = len(tr_rows), len(te_rows)
    levels = list(cfg["data"]["levels"])
    boot = ST.Bootstrap(EG.level_index(te_rows, levels), cfg["probe_impl"]["n_bootstrap"], cfg["probe_impl"]["bootstrap_seed"])
    rmasks = EG.role_masks(te_rows)
    nominal = np.array([r["nominal_log_area"] for r in te_rows])
    frames, t, horizon, L = meta["frames"], meta["t"], meta["horizon"], meta["n_blocks"]
    W = EG.windows(cfg)
    pos = {w: [frames.index(f) for f in W[w]] for w in ("early", "late", "union") if set(W[w]) <= set(frames)}
    mf = np.asarray(C["mask_feat"])
    seeds = cfg["probe"]["seeds"]
    cells, pms, summary = {}, {}, {}
    t0 = time.time()

    def add(key, S, pm):
        cells[key], pms[key] = S, pm
        summary[key] = EG.summarize_cell(S, boot, rmasks, pm, nominal)
        log("q2 cell %s (%.0fs)" % (key, time.time() - t0))

    prim = cfg["q1"]["primary_target"]
    Yp = EG.make_labels(cfg, rows, mf, prim)
    Ya = EG.make_labels(cfg, rows, mf, cfg["q1"]["secondary_target"])
    for j in range(L + 1):
        F = np.load(tmp / ("feat_%d.npy" % j), mmap_mode="r")
        for kind in cfg["probe"]["kinds"]:
            Y = Yp[:, frames]
            c, p = EG.fit_predict_cells(cfg, kind, F[:Etr], Y[:Etr], F[Etr:], Y[Etr:], list(range(len(frames))), pos, seeds)
            for w in c:
                add("%s|%s|layer%d|%s" % (prim, kind, j, w), c[w], p[w])
        # exploratory: frame-area target, linear only, label at the INPUT frame f (primary alignment) and at the TARGET frame f+1
        for align, off in (("input_frame", 0), ("target_frame", 1)):
            Y = Ya[:, [f + off for f in frames]]
            c, p = EG.fit_predict_cells(cfg, "linear_pooled", F[:Etr], Y[:Etr], F[Etr:], Y[Etr:], list(range(len(frames))), {"union": pos["union"]}, seeds)
            add("%s|linear_pooled|layer%d|union|align=%s" % (cfg["q1"]["secondary_target"], j, align), c["union"], p["union"])
        del F
    R = np.load(tmp / "roll.npy", mmap_mode="r")
    out_sum = {}
    if cfg["q2_impl"]["output_space_probe"]:
        oc, op, dig = output_space(cfg, C["tokens_dino"], Yp, Etr, R[Etr:], t, horizon)
        for k_, S_ in oc.items():
            cells[k_], pms[k_] = S_, op[k_]
            summary[k_] = EG.summarize_cell(S_, boot, rmasks, op[k_], nominal)
        lv_te = np.array([r["level"] for r in te_rows])
        out_sum = {"probe_digest_per_kind_seed": dig, "read_by_level": {
            k_: {l: {"read_log_size_mean": float(op[k_][:, lv_te == l].mean()), "true_log_size": float(nominal[lv_te == l][0])} for l in levels}
            for k_ in op}, "note": "digest = sha1 of the probe's rounded predictions on the encoder tokens of step 1; the probe is refitted per run from the same encoder tokens/labels/seeds"}
        log("q2 output-space probe done (%.0fs)" % (time.time() - t0))
    diag_res = diagnostics(cfg, tmp, te_rows, Etr, L)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "q2_cells.npz", **cells, **{"predmean|" + k: v for k, v in pms.items()})
    meta["rollout_frame_of_step"] = {str(k): t + k for k in range(1, horizon + 1)}
    write_json({"meta": meta, "probe_seeds": seeds, "note": "numbers only; layer x time grid is exploratory; registered points are last block (layer-wise probe) and last rollout step (output-space probe)",
                "layerwise_limitation": LAYERWISE_LIMITATION, "output_space": out_sum, "diagnostics_4c": diag_res,
                "cells": summary}, outdir / "q2_summary.json")
    if not cfg["q2_impl"]["keep_tmp"]:
        shutil.rmtree(tmp, ignore_errors=True)
    log("q2 done: %s" % name)
    return outdir
