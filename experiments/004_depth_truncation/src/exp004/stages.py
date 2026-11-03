"""Stage implementations of experiment 004 (cache / probe / predictor / aggregate / check). Heavy parts need GPU + the 002 cache + a trained
002 checkpoint; none of this was run on the authoring machine (no GPU, data or checkpoints) except through toy models in selftest."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from exp002 import cache as CA
from exp002 import evalgrid as EG
from exp002 import model_io as M
from exp002 import probes as PR
from . import depthlib as DL
from .base import (as002, log, read_json, write_json, run_dir002, cget, ConfigNull)
from . import deep as DP

PNAME = {"linear_pooled": "linear", "attentive": "attentive"}
PARTS = "parts"


# ------------------------------------------------------------------------------------------------ small helpers
class Part:
    """Rows of one stage written to results/parts/<name>.csv after every unit (resume: units already present are skipped)."""

    def __init__(self, paths, name, fresh):
        self.path = paths["results_dir"] / PARTS / (name + ".csv")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = DL.read_rows(self.path) if (self.path.exists() and not fresh) else []

    def done(self, **kv):
        return any(all(str(r.get(k)) == str(v) for k, v in kv.items()) for r in self.rows)

    def add(self, rows):
        self.rows.extend(rows)
        DL.write_rows(self.rows, self.path)


def update_meta(paths, **kv):
    p = paths["results_dir"] / PARTS / "meta.json"
    d = read_json(p) if p.exists() else {}
    d.update(kv)
    write_json(d, p)


def frames_and_windows(cfg2):
    W = EG.windows(cfg2)
    fr = list(W["union"])
    return fr, {"early": [fr.index(f) for f in W["early"]], "late": [fr.index(f) for f in W["late"]]}


def labels(cfg2, C):
    rows = C["rows"]
    tr, te = EG.split_rows(rows)
    assert all(r["part"] == "train" for r in rows[:len(tr)]) and all(r["part"] == "test" for r in rows[len(tr):])
    Y = EG.make_labels(cfg2, rows, np.asarray(C["mask_feat"]), cfg2["q1"]["primary_target"])
    return rows, tr, te, Y


def pixel_tokens(u8, grid=14, q=4):
    """(n,H,W,3) uint8 -> (n, grid*grid, q*q*3) float16 in [0,1]: each grid cell is block-averaged down to q x q pixels (raw-pixel baseline input)."""
    n, H, W, _ = u8.shape
    ph, pw = H // grid, W // grid
    q = min(q, ph)
    assert H == ph * grid and W == pw * grid and ph % q == 0 and pw % q == 0, "frame size %dx%d not compatible with grid %d / q %d" % (H, W, grid, q)
    a = np.asarray(u8, np.float32) / 255.0
    a = a.reshape(n, grid, q, ph // q, grid, q, pw // q, 3).mean((3, 6))            # (n,g,q,g,q,3)
    return a.transpose(0, 1, 3, 2, 4, 5).reshape(n, grid * grid, q * q * 3).astype(np.float16)


def enc_dir(paths, suffix):
    return paths["cache"] / ("enc" + suffix)


def disk_estimate(cfg, E, nF):
    k = len(cfg["encoder"]["k_grid"])
    one = E * nF * 196 * 384 * 2 / 1e9
    reps = 1 + (1 if cfg["encoder"]["random_init_control"] else 0)
    return {"tokens_per_k_gb": round(one, 2), "tokens_total_gb": round(one * k * reps, 1), "pixel_gb": round(E * nF * 196 * 48 * 2 / 1e9, 2)}


# ------------------------------------------------------------------------------------------------ 4-A cache
def enc_cache(cfg, cfg2, paths, suffix="", overwrite=False, dry_run=False):
    p2 = as002(paths)
    ks = [int(k) for k in cfg["encoder"]["k_grid"]]
    frames, _ = frames_and_windows(cfg2)
    cd = enc_dir(paths, suffix)
    try:
        C = CA.load(p2, suffix)
        E = len(C["rows"])
    except FileNotFoundError as e:
        if not dry_run:
            raise
        log("DRY-RUN: 002 Q1 cache not found (%s); using config sizes for the estimate" % e)
        C, E = None, cfg2["data"]["train_episodes_per_level"] * len(cfg2["data"]["train_levels_b"]) + cfg2["data"]["eval_episodes_per_level"] * len(cfg2["data"]["levels"])
    log("enc_cache plan: k_grid=%s frames=%s episodes=%d random_init_control=%s -> %s ; disk %s" % (ks, frames, E, cfg["encoder"]["random_init_control"], cd, disk_estimate(cfg, E, len(frames))))
    if dry_run:
        return None
    if (cd / "DONE").exists() and not overwrite:
        log("enc_cache: %s/DONE exists, skip" % cd)
        return cd
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(cfg["encoder"]["source_train_seed"])
    rdir = run_dir002(paths, cfg, seed)
    if not (rdir / "DONE").exists():
        raise FileNotFoundError("002 run %s has no DONE (train condition (b) first): the frozen encoder is taken from it" % rdir)
    model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
    fp = M.encoder_fingerprint(model.encoder)
    if fp != C["meta"]["encoder_fingerprint"]:
        raise RuntimeError("encoder of %s differs from the Q1 cache encoder" % rdir)
    base = model.encoder.base_model
    L = len(DP.flat_blocks(base))
    log("encoder: %d blocks, emb_dim %s, patch_size %s" % (L, getattr(model.encoder, "emb_dim", "?"), getattr(model.encoder, "patch_size", "?")))
    if max(ks) > L:
        raise ValueError("k_grid contains %d but the encoder has %d blocks" % (max(ks), L))
    rnd = None
    if cfg["encoder"]["random_init_control"]:
        rc = cfg2["probe_impl"]["random_init"]
        torch.manual_seed(int(rc["seed"]))
        rnd = torch.hub.load(rc["hub_repo"], rc["hub_name"], pretrained=False).to(dev).eval()
    cd.mkdir(parents=True, exist_ok=True)
    nF = len(frames)
    mm = {}
    for k in ks:
        mm["k%d" % k] = np.lib.format.open_memmap(cd / ("tokens_k%d.npy" % k), mode="w+", dtype=np.float16, shape=(E, nF, M.P_TOK, M.D_VIS))
        if rnd is not None:
            mm["r%d" % k] = np.lib.format.open_memmap(cd / ("rand_k%d.npy" % k), mode="w+", dtype=np.float16, shape=(E, nF, M.P_TOK, M.D_VIS))
    pix = np.lib.format.open_memmap(cd / "pixel.npy", mode="w+", dtype=np.float16, shape=(E, nF, M.P_TOK, 48))
    bs = int(cfg["encoder"]["batch_episodes"])
    t0 = time.time()
    for s in range(0, E, bs):
        sl = slice(s, min(E, s + bs))
        u8 = np.asarray(C["frames_u8"][sl])[:, frames]                       # (b,nF,224,224,3)
        b = u8.shape[0]
        x = M.frames_to_float(u8).to(dev)
        xr = model.encoder_transform(x.reshape(-1, *x.shape[2:]))
        with torch.no_grad():
            st = DP.tokens_by_k(base, xr, ks)
            sr = DP.tokens_by_k(rnd, xr, ks) if rnd is not None else {}
            for k in ks:
                mm["k%d" % k][sl] = st[k].reshape(b, nF, M.P_TOK, M.D_VIS).half().cpu().numpy()
                if k in sr:
                    mm["r%d" % k][sl] = sr[k].reshape(b, nF, M.P_TOK, M.D_VIS).half().cpu().numpy()
            if s == 0:
                chk = []
                ref = np.asarray(C["tokens_dino"][sl])[:, frames].astype(np.float32)
                if L in ks:
                    d = float(np.abs(mm["k%d" % L][sl].astype(np.float32) - ref).max() / (ref.std() + 1e-8))
                    chk.append(("k=%d vs 002 tokens_dino" % L, d))
                    fo = base.get_intermediate_layers(xr, n=[L - 1], norm=True)[0].reshape(b, nF, M.P_TOK, M.D_VIS).float().cpu().numpy() if hasattr(base, "get_intermediate_layers") else None
                    if fo is not None:
                        chk.append(("k=%d vs get_intermediate_layers(norm=True)" % L, float(np.abs(mm["k%d" % L][sl].astype(np.float32) - fo).max() / (ref.std() + 1e-8))))
                if rnd is not None and L in ks:
                    rr = np.asarray(C["tokens_rand"][sl])[:, frames].astype(np.float32)
                    chk.append(("random-init k=%d vs 002 tokens_rand" % L, float(np.abs(mm["r%d" % L][sl].astype(np.float32) - rr).max() / (rr.std() + 1e-8))))
                for name, d in chk:
                    log("consistency check %s: max|diff|/std = %.4g" % (name, d))
                    if d > 0.05:
                        raise RuntimeError("truncated encoder tokens differ from the reference (%s: %.4g): wrong block loop / normalization" % (name, d))
        pix[sl] = pixel_tokens(u8.reshape(-1, *u8.shape[2:]), grid=14, q=4).reshape(b, nF, M.P_TOK, 48)
        log("enc_cache %d/%d (%.0fs)" % (sl.stop, E, time.time() - t0))
    for a in list(mm.values()) + [pix]:
        a.flush()
    write_json({"source_run": str(rdir), "epoch": ep, "encoder_fingerprint": fp, "k_grid": ks, "frames": frames, "n_episodes": E, "encoder_blocks": L,
                "random_init_control": rnd is not None, "suffix": suffix, "elapsed_s": time.time() - t0}, cd / "meta.json")
    (cd / "DONE").write_text("ok\n")
    return cd


# ------------------------------------------------------------------------------------------------ 4-A probes
def probe_hidden(cfg2, kinds, seeds, F, Ytr_pos, Yte_pos, Etr, pos_windows, te_rows, levels, base, shuffle_base=None, ctrl_name="real"):
    """Fit every kind on F[:Etr] (all cached frames) and return rows for windows early/late. F: (E,nF,P,d) array/memmap."""
    nF = F.shape[1]
    rows = []
    for kind in kinds:
        c, p = EG.fit_predict_cells(cfg2, kind, F[:Etr], Ytr_pos, F[Etr:], Yte_pos, list(range(nF)), pos_windows, seeds, shuffle_base=shuffle_base)
        for w in c:
            b = dict(base)
            b.update({"probe": PNAME[kind], "timepoint": w, "control": ctrl_name})
            rows += DL.cell_rows(c[w], p[w], te_rows, levels, b, seeds)
    return rows


def enc_probe(cfg, cfg2, paths, suffix="", fresh=False, dry_run=False):
    kinds, seeds = cfg2["probe"]["kinds"], cfg2["probe"]["seeds"]
    ks = [int(k) for k in cfg["encoder"]["k_grid"]]
    ctrls = ["real"] + (["random_init"] if cfg["encoder"]["random_init_control"] else []) + (["label_shuffle"] if cfg["encoder"]["label_shuffle_control"] else [])
    n_fit = len(ks) * len(ctrls) * len(kinds) * len(seeds) + len(kinds) * len(seeds) + len(seeds)
    log("enc_probe plan: k=%s controls=%s kinds=%s probe seeds=%s ; probe fits (linear+attentive) = %d (+ pixel baseline, mask baseline included)" % (ks, ctrls, kinds, seeds, n_fit))
    if dry_run:
        return None
    p2 = as002(paths)
    C = CA.load(p2, suffix)
    cd = enc_dir(paths, suffix)
    if not (cd / "DONE").exists():
        raise FileNotFoundError("%s/DONE missing - run `python run.py enc_cache` first" % cd)
    emeta = read_json(cd / "meta.json")
    frames = emeta["frames"]
    _, pos_windows = frames_and_windows(cfg2)
    rows, tr, te, Y = labels(cfg2, C)
    Etr = len(tr)
    levels = list(cfg2["data"]["levels"])
    Yp = Y[:, frames]
    Ytr, Yte = Yp[:Etr], Yp[Etr:]
    part = Part(paths, "encoder", fresh)
    shuf = int(cfg2["probe_impl"]["shuffle_seed"])
    t0 = time.time()
    for k in ks:
        for ctrl in ctrls:
            if part.done(module="encoder", depth=k, control=ctrl):
                continue
            F = np.load(cd / (("rand_k%d.npy" if ctrl == "random_init" else "tokens_k%d.npy") % k), mmap_mode="r")
            base = {"module": "encoder", "depth": k, "fit": "new", "train_seed": "NA"}
            part.add(probe_hidden(cfg2, kinds, seeds, F, Ytr, Yte, Etr, pos_windows, te, levels, base, shuffle_base=(shuf if ctrl == "label_shuffle" else None), ctrl_name=ctrl))
            log("enc_probe k=%d control=%s done (%.0fs)" % (k, ctrl, time.time() - t0))
    if not part.done(module="baseline", control="pixel"):
        F = np.load(cd / "pixel.npy", mmap_mode="r")
        part.add(probe_hidden(cfg2, kinds, seeds, F, Ytr, Yte, Etr, pos_windows, te, levels, {"module": "baseline", "depth": -1, "fit": "new", "train_seed": "NA"}, ctrl_name="pixel"))
    if not part.done(module="baseline", control="mask_area_bbox"):
        mf = PR.mask_baseline_features(np.asarray(C["mask_feat"]))[:, frames]
        c, p = EG.fit_predict_cells(cfg2, "ridge_feat", mf[:Etr], Ytr, mf[Etr:], Yte, list(range(len(frames))), pos_windows, seeds)
        out = []
        for w in c:
            out += DL.cell_rows(c[w], p[w], te, levels, {"module": "baseline", "depth": -1, "probe": "linear", "fit": "new", "timepoint": w, "control": "mask_area_bbox", "train_seed": "NA"}, seeds)
        part.add(out)
    update_meta(paths, encoder_layers=emeta["encoder_blocks"], partial=bool(suffix) or bool(emeta.get("suffix")))
    log("enc_probe done: %d rows -> %s" % (len(part.rows), part.path))
    return part.path


# ------------------------------------------------------------------------------------------------ 4-B predictor
def pred_stage(cfg, cfg2, paths, seeds_train, controls, suffix="", fresh=False, dry_run=False, keep_tmp=False):
    pc = cfg["predictor"]
    H = cget(cfg2, "train.num_hist")
    if H is None:
        raise ConfigNull("002 config train.num_hist is null")
    kinds, pseeds = cfg2["probe"]["kinds"], cfg2["probe"]["seeds"]
    frames, pos_windows = frames_and_windows(cfg2)
    steps = list(pc["rollout_steps"])
    E = cfg2["data"]["train_episodes_per_level"] * len(cfg2["data"]["train_levels_b"]) + cfg2["data"]["eval_episodes_per_level"] * len(cfg2["data"]["levels"])
    dpart = 384 if pc["feature_part"] == "z_only" else 404
    L_guess = 6
    gb = E * len(frames) * 196 * dpart * 2 * (L_guess + 1) / 1e9 + (E - cfg2["data"]["train_episodes_per_level"] * 3) * (len(frames) + len(steps)) * 196 * 384 * 2 * (L_guess + 1) / 1e9
    log("pred plan: 002 runs (condition %s) seeds=%s controls=%s num_hist=%s feature_part=%s rollout_steps=%s ; tmp disk about %.0f GB per run (assumes 6 blocks, 'real' control) ; "
        "hidden probe fits per run = (blocks+1) x kinds x probe seeds x (1 + label_shuffle) ; exp1-probe fits = kinds x probe seeds" % (
            cfg["base"]["condition"], seeds_train, controls, H, pc["feature_part"], steps, gb))
    if dry_run:
        return None
    import torch
    p2 = as002(paths)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(p2, suffix)
    rows, tr, te, Y = labels(cfg2, C)
    Etr = len(tr)
    levels = list(cfg2["data"]["levels"])
    Yp = Y
    shuf = int(cfg2["probe_impl"]["shuffle_seed"])
    for ts in seeds_train:
        rdir = run_dir002(paths, cfg, ts)
        if not (rdir / "DONE").exists():
            raise FileNotFoundError("002 run %s has no DONE (condition (b) checkpoint missing) - train it first" % rdir)
        model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
        if M.encoder_fingerprint(model.encoder) != C["meta"]["encoder_fingerprint"]:
            raise RuntimeError("encoder of %s differs from the Q1 cache encoder" % rdir)
        assert int(tcfg.num_hist) == int(H), "checkpoint num_hist %s != 002 config train.num_hist %s" % (tcfg.num_hist, H)
        mean, std, norm = M.action_stats_for_run(cfg2, rdir, tcfg)
        L = M.n_blocks(model)
        log("pred seed %d: epoch %d, %d predictor blocks (read from the model), num_hist %d" % (ts, ep, L, H))
        for ctrl in controls:
            name = "predictor_s%d_%s" % (ts, ctrl)
            part = Part(paths, name, fresh)
            if part.rows and not fresh:
                log("pred %s: part file exists (%d rows) - skipped (use --fresh to redo)" % (name, len(part.rows)))
                continue
            tmp = paths["cache"] / ("pred_tmp_%s_s%d_%s" % (cfg["base"]["condition"], ts, ctrl))
            mdl = model if ctrl == "real" else DP.random_init_predictor(model, int(pc["random_init_seed"]) + ts)
            meta = DP.extract(cfg, cfg2, C, mdl, H, tmp, mean, std, dev, frames, Etr, steps, dpart, with_outputs=(ctrl == "real"),
                              batch_episodes=int(pc["batch_episodes"]), label="pred s%d %s" % (ts, ctrl))
            out = pred_probe_rows(cfg, cfg2, C, tmp, meta, Yp, Etr, te, levels, ts, ctrl, pos_windows, shuf, paths)
            part.add(out["rows"])
            if out["mse_rows"]:
                DL.write_rows(out["mse_rows"], paths["results_dir"] / PARTS / (name + "_mse.csv"), fields=MSE_FIELDS)
            update_meta(paths, predictor_layers=L, partial=bool(suffix))
            if not (keep_tmp or pc.get("keep_tmp")):
                shutil.rmtree(tmp, ignore_errors=True)
    return True


MSE_FIELDS = ["module", "depth", "control", "train_seed", "level", "split", "timepoint", "mse", "n_episodes"]


def pred_probe_rows(cfg, cfg2, C, tmp, meta, Yp, Etr, te, levels, ts, ctrl, pos_windows, shuf, paths):
    """Probes on the extracted hidden states (i), exp1-probe applied to early-exit outputs (ii), next-z MSE (iii). Pure given the files in tmp."""
    tmp = Path(tmp)
    L, frames, steps = meta["n_blocks"], meta["frames"], meta["keep_steps"]
    kinds, seeds = cfg2["probe"]["kinds"], cfg2["probe"]["seeds"]
    t = meta["t"]
    Yf = Yp[:, frames]
    rows = []
    for m in range(L + 1):                                                         # (i) hidden probe, fit = new
        F = np.load(tmp / ("feat_%d.npy" % m), mmap_mode="r")
        base = {"module": "predictor", "depth": m, "fit": "new", "train_seed": ts}
        rows += probe_hidden(cfg2, kinds, seeds, F, Yf[:Etr], Yf[Etr:], Etr, pos_windows, te, levels, base, ctrl_name=ctrl)
        if ctrl == "real" and cfg["predictor"]["label_shuffle_control"]:
            rows += probe_hidden(cfg2, kinds, seeds, F, Yf[:Etr], Yf[Etr:], Etr, pos_windows, te, levels, base, shuffle_base=shuf, ctrl_name="label_shuffle")
        del F
    mse_rows = []
    if ctrl == "real" and meta["with_outputs"]:
        tokens = C["tokens_dino"]
        frames_tr = EG.windows(cfg2)[cfg2["q2_impl"]["output_probe_train_regime"]]
        sets, ref_sets = {}, {}
        outs = [np.load(tmp / ("out_%d.npy" % m), mmap_mode="r") for m in range(L + 1)]
        rolls = [np.load(tmp / ("roll_%d.npy" % m), mmap_mode="r") for m in range(L + 1)]
        for fi, f in enumerate(frames):
            ref_sets["ref|f%d" % f] = (tokens[Etr:, f + 1], Yp[Etr:, f + 1])
            for m in range(L + 1):
                sets["m%d|f%d" % (m, f)] = (outs[m][:, fi], Yp[Etr:, f + 1])
        for ki, k in enumerate(steps):
            ref_sets["ref|r%d" % k] = (tokens[Etr:, t + k], Yp[Etr:, t + k])
            for m in range(L + 1):
                sets["m%d|r%d" % (m, k)] = (rolls[m][:, ki], Yp[Etr:, t + k])
        allsets = dict(ref_sets)
        allsets.update(sets)
        for kind in kinds:
            c, p, _ = EG.fit_apply_cells(cfg2, kind, tokens[:Etr], Yp[:Etr], frames_tr, allsets, seeds)
            def agg(prefix, fl):
                S = sum(c["%s|f%d" % (prefix, f)] for f in fl)
                pm = np.mean([p["%s|f%d" % (prefix, f)] for f in fl], axis=0)
                return S, pm
            fw = {w: [frames[i] for i in pos_windows[w]] for w in pos_windows}
            targets = [("reference", -1, "encoder_tokens", "ref")] + [("predictor", m, "real", "m%d" % m) for m in range(L + 1)]
            for module, depth, ctl, pref in targets:
                base = {"module": module, "depth": depth, "probe": PNAME[kind], "fit": "exp1_probe", "control": ctl, "train_seed": ts if module == "predictor" else "NA"}
                for w, fl in fw.items():
                    S, pm = agg(pref, fl)
                    b = dict(base); b["timepoint"] = w
                    rows += DL.cell_rows(S, pm, te, levels, b, seeds)
                for k in steps:
                    b = dict(base); b["timepoint"] = "rollout_t+%d" % k
                    rows += DL.cell_rows(c["%s|r%d" % (pref, k)], p["%s|r%d" % (pref, k)], te, levels, b, seeds)
        mse_rows = mse_to_rows(np.load(tmp / "mse_tf.npy"), np.load(tmp / "mse_roll.npy"), te, levels, frames, pos_windows, steps, ts)
    return {"rows": rows, "mse_rows": mse_rows}


def mse_to_rows(mse_tf, mse_roll, te, levels, frames, pos_windows, steps, ts):
    """(Ete,nF,L+1), (Ete,nsteps,L+1) per-episode next-z MSE -> rows per level and split."""
    lv = np.array([r["level"] for r in te])
    sp = np.array([DL.ROLE2SPLIT[r["role"]] for r in te])
    groups = [(l, sp[lv == l][0], lv == l) for l in levels if (lv == l).any()] + [("ALL", s, sp == s) for s in DL.SPLITS if (sp == s).any()] + [("ALL", "all", np.ones(len(lv), bool))]
    L1 = mse_tf.shape[2]
    out = []
    series = {w: mse_tf[:, pos_windows[w]].mean(1) for w in pos_windows}
    for ki, k in enumerate(steps):
        series["rollout_t+%d" % k] = mse_roll[:, ki]
    for w, arr in series.items():
        for name, split, mk in groups:
            for m in range(L1):
                out.append({"module": "predictor", "depth": m, "control": "real", "train_seed": ts, "level": name, "split": split, "timepoint": w,
                            "mse": float(arr[mk, m].mean()), "n_episodes": int(mk.sum())})
    return out


# ------------------------------------------------------------------------------------------------ aggregate
def aggregate(cfg, cfg2, paths, dry_run=False):
    parts = sorted((paths["results_dir"] / PARTS).glob("*.csv")) if (paths["results_dir"] / PARTS).exists() else []
    parts_r2 = [p for p in parts if not p.name.endswith("_mse.csv")]
    log("aggregate: %d part files (%s)" % (len(parts_r2), [p.name for p in parts_r2]))
    if dry_run:
        return None
    if not parts_r2:
        raise FileNotFoundError("no results/parts/*.csv - run enc_probe / pred first")
    rows = []
    for p in parts_r2:
        rows += DL.read_rows(p)
    out = paths["results_dir"]
    DL.write_rows(rows, out / "depth_r2.csv")
    mrows = []
    for p in parts:
        if p.name.endswith("_mse.csv"):
            import csv
            with open(p) as f:
                mrows += list(csv.DictReader(f))
    if mrows:
        DL.write_rows(mrows, out / "next_z_mse.csv", fields=MSE_FIELDS)
    meta = read_json(paths["results_dir"] / PARTS / "meta.json") if (paths["results_dir"] / PARTS / "meta.json").exists() else {}
    pilot = bool(cfg["pilot"]) or bool(meta.get("partial"))
    summ = DL.build_summary(rows, cfg, {"pilot": pilot, "encoder_layers": meta.get("encoder_layers"), "predictor_layers": meta.get("predictor_layers")})
    write_json(summ, out / "summary.json")
    figs = "skipped"
    try:
        from . import figs as FG
        figs = FG.make(rows, cfg, cfg2, out)
    except ImportError as e:
        figs = "skipped(no matplotlib: %s)" % e
    from exp002.common import env_info
    write_json({"experiment": cfg["name"], "pilot": pilot, "config_sha256": __import__("hashlib").sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(),
                "config002_sha256": cfg2.get("_config002_sha256"), "n_rows_depth_r2": len(rows), "files": ["depth_r2.csv", "summary.json", "next_z_mse.csv"], "figures": figs,
                "summary": summ, "environment": env_info(), "interpretation": None}, paths["exp"] / "results.json")
    log("aggregate done: %d rows ; summary k_star=%s m_star=%s verdict_H_d=%s pilot=%s" % (len(rows), summ["k_star"], summ["m_star"], summ["verdict_H_d"]["status"], pilot))
    return summ


# ------------------------------------------------------------------------------------------------ check
def check(cfg, cfg2, paths):
    from exp002.common import env_info
    info = env_info()
    print("python executable :", info["python_executable"])
    print("CONDA_DEFAULT_ENV :", info["CONDA_DEFAULT_ENV"])
    print("packages          :", json.dumps(info["packages"]))
    print("cuda_available    :", info.get("cuda_available"), info.get("gpu"), info.get("gpu_mem_gb"))
    for k in ("work", "acwm_src", "train_base", "cache002", "cache"):
        print("  %-10s %s exists=%s" % (k, paths[k], paths[k].exists()))
    for suffix in ("", "_smoke"):
        print("002 Q1 cache%-7s DONE=%s" % (suffix, (CA.cache_dir(as002(paths), suffix) / "DONE").exists()))
    for s in cfg["predictor"]["train_seeds"]:
        rd = run_dir002(paths, cfg, s)
        print("002 run %s DONE=%s" % (rd.name, (rd / "DONE").exists()))
    print("config002 sha256  :", cfg2.get("_config002_sha256"))
    print("train.num_hist (002) =", cget(cfg2, "train.num_hist"), "; frameskip =", cget(cfg2, "data_impl.frameskip"))
    return 0
