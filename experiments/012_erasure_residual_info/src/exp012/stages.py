"""Stages of experiment 012: check | measure | aggregate | selftest. Units: base (train seed) = strength-0 measurements at every needed depth (shared by all blocks);
unit (train seed, block m_int) = strength-1.0 measurements at depth m_int and m_int+1 (none for m_int = last block). Same 21 units as 009 for m_int [1..6] x 3 seeds.
Resume: a unit whose part file exists (base: and covers the needed depths) is skipped. Heavy parts need GPU + 002 cache + 002 condition-(b) checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from exp002 import cache as CA
from exp002 import evalgrid as EG
from exp002 import model_io as M
from exp004 import stages as S4
from exp009 import erase as ER
from exp009 import stages as S9
from . import base as B
from . import measure as MS
from . import verdict as VD
from .base import log, read_json, write_json

L_ASSUMED = 6          # only used for planning / quick skip checks before the model is loaded (009 README: 6 predictor blocks); the real value is read from the model


# ------------------------------------------------------------------------------------------------ units and part files
def resolve_m_int(cfg, cli_value=None):
    """(list, overridden_flag, source). CLI --m-int overrides config intervention.m_int; recorded in results.json."""
    cfgv = [int(x) for x in cfg["intervention"]["m_int"]]
    val, src = cfgv, "config"
    if cli_value:
        val, src = [int(x) for x in str(cli_value).split(",") if x.strip()], "cli"
    val = sorted(set(val))
    if not val:
        raise ValueError("empty m_int list")
    return val, (val != sorted(set(cfgv))), src


def needed_depths(m_ints, L):
    """Baseline (strength 0) depths needed: m_int and m_int+1 (<= L) of every requested block."""
    return sorted({d for m in m_ints for d in (m, m + 1) if 1 <= d <= L})


def part_path(paths, kind, ts, m=None):
    d = paths["results_dir"] / "parts"
    return d / (("base_s%d.json" % ts) if kind == "base" else ("unit_m%d_s%d.json" % (m, ts)))


def units(m_ints, train_seeds):
    return [("base", ts, None) for ts in train_seeds] + [("unit", ts, m) for ts in train_seeds for m in m_ints]


def unit_done(paths, u, m_ints, L=L_ASSUMED):
    kind, ts, m = u
    p = part_path(paths, kind, ts, m)
    if not p.exists():
        return False
    if kind == "unit":
        return True
    d = read_json(p).get("depths", {})
    return all(str(x) in d for x in needed_depths(m_ints, L))


def plan_text(cfg, m_ints, train_seeds, paths):
    us = units(m_ints, train_seeds)
    done = [u for u in us if unit_done(paths, u, m_ints)]
    return ("measure plan: m_int %s, strengths [0 (shared baseline), %g], train seeds %s, method %s ; units total %d (baseline %d + erased %d), already done %d -> to run %d ; "
            "base depths needed %s ; measures %s ; probe seeds %s" % (m_ints, float(cfg["success"]["strength"]), train_seeds, cfg["intervention"]["method"], len(us), len(train_seeds),
                                                                     len(us) - len(train_seeds), len(done), len(us) - len(done), needed_depths(m_ints, L_ASSUMED), MS.MEASURES, cfg["measure"]["probe_seeds"]))


# ------------------------------------------------------------------------------------------------ preflight (no GPU, never consumes an attempt)
def checkpoint_problems(c4, paths, train_seeds):
    """The 002 run directory structure the code will read: <DINO_TRAIN>/outputs/exp002_multi_60_ind_160_s<seed>/{DONE, hydra.yaml, train_manifest.json, checkpoints/model_latest.pth}.
    (006 failed 3 times because the run name lacked the exp002_ prefix of the real Drive folder.)"""
    bad = []
    out_dir = paths["train_base"] / "outputs"
    for ts in train_seeds:
        rd = B.B4.run_dir002(paths, c4, ts)
        if not rd.is_dir():
            near = sorted(p.name for p in out_dir.iterdir() if ("s%d" % ts) in p.name)[:6] if out_dir.is_dir() else []
            bad.append("train seed %d: run directory %s missing (expected name %s under %s; directories there with s%d: %s)" % (ts, rd, rd.name, out_dir, ts, near))
            continue
        for rel in ("DONE", "hydra.yaml", "train_manifest.json", "checkpoints/model_latest.pth"):
            if not (rd / rel).exists():
                bad.append("train seed %d: %s missing" % (ts, rd / rel))
    return bad


def input_preflight(c4, paths, train_seeds):
    bad = checkpoint_problems(c4, paths, train_seeds)
    qd = CA.cache_dir(B.B4.as002(paths), "") / "DONE"
    if not qd.exists():
        bad.append("002 Q1 cache DONE missing (%s)" % qd)
    bad += S9.metadata_preflight(c4, paths, train_seeds)
    return bad


def check(cfg, c4, cfg2, paths, train_seeds, c9):
    print("cache dir (tmp features, directions):", paths["cache"], "exists=%s" % paths["cache"].exists())
    res = B.consistency_checks(cfg, c9, cfg2)
    for n, ok, d in res:
        print("  [%s] %s %s" % ("ok" if ok else "FAIL", n, "" if ok else d))
    print("009 results.json (for the reproduction check (d)):", paths["results009"], "exists=%s" % paths["results009"].exists())
    bad = input_preflight(c4, paths, train_seeds)
    print("inputs (checkpoints, Q1 cache, metadata.pt): %s" % ("all present" if not bad else "MISSING %d" % len(bad)))
    for b in bad[:20]:
        print("  !! " + b)
    return 0 if (not bad and all(ok for _, ok, _ in res)) else 1


# ------------------------------------------------------------------------------------------------ extraction
def extract12(model, C, H, tmp, mean, std, dev, frames, dz, da, depths, erase=None, batch_episodes=8, label=""):
    """Like exp009.erase.extract_hidden (same forward, same erasure), but also returns the pooled action/proprio coordinates [dz:dz+da] of the last-context-frame tokens.
    Writes tmp/feat_{m}.npy (E,nF,P,dz) float16 for m in depths. Returns dict(pa={m: (E,nF,da) float32}, pooled_mint (E,nF,dz) float32 or None, n_blocks)."""
    import torch
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    L = M.n_blocks(model)
    E_ = len(C["rows"])
    P = M.P_TOK
    nF = len(frames)
    assert min(frames) >= H - 1
    feats = {m: np.lib.format.open_memmap(tmp / ("feat_%d.npy" % m), mode="w+", dtype=np.float16, shape=(E_, nF, P, dz)) for m in depths}
    pa = {m: np.zeros((E_, nF, da), np.float32) for m in depths}
    pooled = np.zeros((E_, nF, dz), np.float32) if erase is not None else None
    t0 = time.time()
    for s in range(0, E_, batch_episodes):
        sl = slice(s, min(E_, s + batch_episodes))
        x = M.frames_to_float(np.asarray(C["frames_u8"][sl])).to(dev)
        acts = M.acts_from_raw(torch.as_tensor(np.asarray(C["actions_raw"][sl])), mean, std).to(dev)
        with torch.no_grad():
            z_all = M.encode_batch(model, x, acts)
            B_ = x.shape[0]
            zw = M.make_windows(z_all, H, frames)
            N, _, _, D = zw.shape
            hid = ER.forward_hidden(model.predictor, zw.reshape(N, H * P, D), erase)
            for m in depths:
                full = M.last_frame_tokens(hid[m], H)
                if full.shape[-1] < dz + da:
                    raise ValueError("hidden dim %d < dz + da = %d" % (full.shape[-1], dz + da))
                tok = full[..., :dz].reshape(B_, nF, P, dz)
                feats[m][sl] = tok.half().cpu().numpy()
                pa[m][sl] = full[..., dz:dz + da].reshape(B_, nF, P, da).float().mean(2).cpu().numpy()
                if pooled is not None and m == erase["m_int"]:
                    pooled[sl] = tok.float().mean(2).cpu().numpy()
        if (s // batch_episodes) % 20 == 0:
            log("%s extract %d/%d (%.0fs)" % (label, sl.stop, E_, time.time() - t0))
    for a in feats.values():
        a.flush()
    return {"pa": pa, "pooled_mint": pooled, "n_blocks": L}


def measure_depth(cfg, cfg2p, tmp, d, pa_d, Yp, rows, Etr, fit_ep, pos_windows, label):
    """Measurements at one depth from tmp/feat_{d}.npy (+ pooled action coordinates)."""
    t0 = time.time()
    F = np.load(Path(tmp) / ("feat_%d.npy" % d), mmap_mode="r")
    E, nF, P, dz = F.shape
    pz = EG.pool_chunked(F.reshape(E * nF, P, dz)).reshape(E, nF, dz)
    rec = MS.measure_all(cfg, cfg2p, pz, pa_d, F, Yp, rows, Etr, fit_ep, pos_windows, cfg["measure"]["probe_seeds"])
    rec["seconds"] = round(time.time() - t0, 1)
    del F
    log("%s depth %d measured in %.0fs" % (label, d, rec["seconds"]))
    return rec


def fit_episodes(cfg, ts, Etr):
    """Same rule as 009: RandomState(fit_seed + train_seed).permutation(Etr)[:round(fit_fraction * Etr)], sorted."""
    ic = cfg["intervention"]
    rng = np.random.RandomState(int(ic["fit_seed"]) + ts)
    return np.sort(rng.permutation(Etr)[: int(round(float(ic["fit_fraction"]) * Etr))])


def run_seed12(cfg, cfg2p, paths, model, C, H, mean, std, dev, frames, pos_windows, Yp, rows, Etr, dz, da, ts, todo, m_ints, fresh=False):
    """All pending units of one train seed with an already loaded model. Returns diagnostics dict."""
    ic = cfg["intervention"]
    bs = int(cfg["predictor"]["batch_episodes"])
    strength = float(cfg["success"]["strength"])
    L = M.n_blocks(model)
    bad = [m for m in m_ints if not (1 <= m <= L)]
    if bad:
        raise ValueError("m_int %s outside 1..%d" % (bad, L))
    tmp = paths["cache"] / ("tmp_s%d" % ts)
    fit_ep = fit_episodes(cfg, ts, Etr)
    MS.probe_split(Etr, len(rows), fit_ep)                       # raises AssertionError if the separation does not hold
    diag = {}
    if ("base", ts, None) in todo:
        bp = part_path(paths, "base", ts)
        part = read_json(bp) if (bp.exists() and not fresh) else {"kind": "base", "train_seed": ts, "strength": 0.0, "n_blocks": L, "depths": {}}
        miss = [d for d in needed_depths(m_ints, L) if str(d) not in part["depths"]]
        if miss:
            out = extract12(model, C, H, tmp, mean, std, dev, frames, dz, da, miss, None, bs, label="s%d base" % ts)
            for d in miss:
                part["depths"][str(d)] = measure_depth(cfg, cfg2p, tmp, d, out["pa"][d], Yp, rows, Etr, fit_ep, pos_windows, "s%d base" % ts)
                write_json(part, bp)
            shutil.rmtree(tmp, ignore_errors=True)
        log("s%d baseline done (depths %s)" % (ts, sorted(int(k) for k in part["depths"])))
    for m in m_ints:
        if ("unit", ts, m) not in todo:
            continue
        dfile = paths["cache"] / "directions" / ("m%d_s%d.npz" % (m, ts))
        if dfile.exists() and not fresh:
            E = ER.load_eraser(dfile)
        else:
            X = ER.pooled_hidden(model, C, H, mean, std, dev, frames, fit_ep, m, dz, bs)
            E = ER.fit_eraser(X.reshape(-1, dz), Yp[fit_ep].reshape(-1), ic["method"], float(ic["ridge_rel"]))
            dfile.parent.mkdir(parents=True, exist_ok=True)
            ER.save_eraser(E, dfile)
            log("s%d m_int %d: direction fitted on %d fit episodes (%d samples), corr(a.x, y) on fit data %.3f" % (ts, m, len(fit_ep), X.shape[0] * X.shape[1], E["corr_fit"]))
        depths = [m] + ([m + 1] if m < L else [])
        er = {"m_int": m, "alpha": strength, "E": E, "dz": dz}
        out = extract12(model, C, H, tmp, mean, std, dev, frames, dz, da, depths, er, bs, label="s%d m%d a%g" % (ts, m, strength))
        part = {"kind": "unit", "train_seed": ts, "m_int": m, "strength": strength, "n_blocks": L,
                "depth_m_int": measure_depth(cfg, cfg2p, tmp, m, out["pa"][m], Yp, rows, Etr, fit_ep, pos_windows, "s%d m%d erased" % (ts, m))}
        part["depth_m_int_plus_1"] = (measure_depth(cfg, cfg2p, tmp, m + 1, out["pa"][m + 1], Yp, rows, Etr, fit_ep, pos_windows, "s%d m%d erased" % (ts, m))
                                      if m < L else "해당 없음 (m_int=%d is the last block: no later block)" % m)
        pm = out["pooled_mint"]
        held, te_idx = MS.probe_split(Etr, pm.shape[0], fit_ep)
        ratio = lambda idx: float((pm[idx].reshape(-1, dz) @ E["a"]).std() / E["readout_std_fit"])
        part["diagnostics"] = {"corr_readout_fit_episodes_pre": E["corr_fit"], "readout_std_ratio_after_over_fit_pre_probe_train": ratio(held), "readout_std_ratio_after_over_fit_pre_test": ratio(te_idx),
                               "n_direction_fit_episodes": int(len(fit_ep))}
        diag["m%d_s%d" % (m, ts)] = part["diagnostics"]
        write_json(part, part_path(paths, "unit", ts, m))
        shutil.rmtree(tmp, ignore_errors=True)
        log("s%d m_int %d strength %g done" % (ts, m, strength))
    return diag


def measure_stage(cfg, c4, cfg2, paths, m_ints, train_seeds, fresh=False, dry_run=False):
    log(plan_text(cfg, m_ints, train_seeds, paths))
    if dry_run:
        return None
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = B.cget(cfg2, "train.num_hist")
    if H is None:
        raise B.ConfigNull("002 config train.num_hist is null")
    dz = int(cfg["predictor"]["feature_part_z"][1])
    a0, a1 = (int(x) for x in cfg["predictor"]["feature_part_action"])
    if cfg["predictor"]["feature_part_z"][0] != 0 or a0 != dz:
        raise ValueError("feature parts must be z=[0,dz) and action=[dz,dz+da)")
    da = a1 - a0
    cfg2p = MS.probe_cfg(cfg, cfg2)
    C = CA.load(B.B4.as002(paths), "")
    frames, pos_windows = S4.frames_and_windows(cfg2)
    rows, Etr, te, Yp = S9.labels_positions(cfg2, C, frames)
    diag_all = {}
    for ts in train_seeds:
        todo = [u for u in units(m_ints, [ts]) if fresh or not unit_done(paths, u, m_ints)]
        if not todo:
            log("train seed %d: all units done" % ts)
            continue
        rdir = B.B4.run_dir002(paths, c4, ts)
        model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
        if M.encoder_fingerprint(model.encoder) != C["meta"]["encoder_fingerprint"]:
            raise RuntimeError("encoder of %s differs from the Q1 cache encoder" % rdir)
        assert int(tcfg.num_hist) == int(H)
        mean, std, _ = M.action_stats_for_run(cfg2, rdir, tcfg)
        log("train seed %d: epoch %d, %d predictor blocks, %d units to run" % (ts, ep, M.n_blocks(model), len(todo)))
        diag_all.update(run_seed12(cfg, cfg2p, paths, model, C, H, mean, std, dev, frames, pos_windows, Yp, rows, Etr, dz, da, ts, todo, m_ints, fresh))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if diag_all:
        p = paths["results_dir"] / "diagnostics.json"
        d = read_json(p) if p.exists() else {}
        d.update(diag_all)
        write_json(d, p)
    return True


# ------------------------------------------------------------------------------------------------ aggregate
def code_version(exp_dir):
    """git HEAD of the checkout containing exp_dir (None if not a git checkout) + env EXP012_CODE_REPO_URL / EXP012_CODE_REF set by the Colab notebook."""
    import subprocess

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(exp_dir)] + list(a), capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    head = git("rev-parse", "HEAD")
    return {"git_head": head, "git_status_dirty": (bool(git("status", "--porcelain", "--", ".")) if head else None),
            "repo_url": os.environ.get("EXP012_CODE_REPO_URL"), "requested_ref": os.environ.get("EXP012_CODE_REF")}


def _get(rec, measure, window, split):
    try:
        v = rec[measure][window][split]["mean"]
        return v
    except (KeyError, TypeError):
        return None


def compact(rec, window):
    """{measure: {split: mean}} of one depth record for the judged window."""
    out = {}
    for mname in MS.MEASURES:
        if isinstance(rec, dict) and isinstance(rec.get(mname), dict):
            out[mname] = {sp: _get(rec, mname, window, sp) for sp in MS.SPLITS if sp in rec[mname].get(window, {})}
    return out


def verdict_for(recs_by_ts, window, b_c_split, success, tseeds):
    a, b, c = {}, {}, {}
    for ts in tseeds:
        r = recs_by_ts.get(ts)
        a[ts] = _get(r, "a_level_logistic", window, "train") if r else None
        b[ts] = {"mlp": _get(r, "b_mlp", window, b_c_split), "attentive": _get(r, "b_attentive", window, b_c_split)} if r else None
        c[ts] = _get(r, "c_action_only_linear", window, b_c_split) if r else None
    return VD.block_verdict(a, b, c, success, tseeds)


def repro_vs_009(res009, m, tseeds, rec_by_strength, window):
    """(d) linear_pooled R2 of 012 vs the 009 curve value at depth m_int (009 results.json blocks[m].strengths[a][ts].curves[split][m]); difference = 012 - 009."""
    if res009 is None:
        return {"status": "not_available", "reason": "009 results.json not found"}
    out = {"status": "computed", "note": "012 trains its probes on the train-split episodes not used for the direction fit (about half), 009 on all train-split episodes; differences are expected to be non-zero", "by_strength": {}}
    for sname, a in (("0", "0"), ("1", "1")):
        out["by_strength"][sname] = {}
        for ts in tseeds:
            r = rec_by_strength[sname].get(ts)
            row = {}
            for sp in ("train", "interp", "extrap"):
                v12 = _get(r, "d_linear_pooled", window, sp) if r else None
                try:
                    v9 = res009["blocks"][str(m)]["strengths"][a][str(ts)]["curves"][sp][str(m)]
                except (KeyError, TypeError):
                    v9 = None
                row[sp] = {"r2_012": v12, "r2_009": v9, "diff_012_minus_009": (None if v12 is None or v9 is None else v12 - v9)}
            out["by_strength"][sname][str(ts)] = row
    return out


def pilot_reasons(cfg, results_dir, m_overridden, seeds_overridden, m_ints, train_seeds):
    """Reasons why results.json is marked pilot: true. Empty list = main run (config pilot false, the experiment's own results dir, no CLI overrides).
    A symlinked results dir (Colab: Drive) counts as the experiment dir when it resolves to the same folder."""
    default_dir = B.default_results_dir().resolve()
    reasons = []
    if cfg["pilot"]:
        reasons.append("config.yaml pilot flag is true")
    if Path(results_dir).resolve() != default_dir:
        reasons.append("results dir %s is not the experiment results dir %s (pilot / separate run)" % (Path(results_dir).resolve(), default_dir))
    if m_overridden:
        reasons.append("m_int overridden on the command line (%s)" % m_ints)
    if seeds_overridden:
        reasons.append("train seeds overridden on the command line (%s)" % train_seeds)
    return reasons


def aggregate(cfg, c4, cfg2, paths, m_ints, m_overridden, source, train_seeds, seeds_overridden, c9, dry_run=False):
    log("aggregate: m_int %s (source %s, overridden=%s), train seeds %s (overridden=%s)" % (m_ints, source, m_overridden, train_seeds, seeds_overridden))
    if dry_run:
        return None
    succ = cfg["success"]
    window = c9["depth"]["primary"]["timepoint"]
    res009 = read_json(paths["results009"]) if Path(paths["results009"]).exists() else None
    bases, units_, missing = {}, {}, []
    for ts in train_seeds:
        p = part_path(paths, "base", ts)
        if p.exists():
            bases[ts] = read_json(p)
        else:
            missing.append(p.name)
        for m in m_ints:
            q = part_path(paths, "unit", ts, m)
            if q.exists():
                units_[(ts, m)] = read_json(q)
            else:
                missing.append(q.name)
    blocks = {}
    for m in m_ints:
        blk = {"m_int": m, "missing_units": [], "strength_1": {}, "strength_0": {}, "plus_1_strength_1": {}, "plus_1_strength_0": {}}
        for ts in train_seeds:
            u, b = units_.get((ts, m)), bases.get(ts)
            if u is None:
                blk["missing_units"].append("unit_m%d_s%d" % (m, ts))
            else:
                blk["strength_1"][ts] = u["depth_m_int"]
                if isinstance(u.get("depth_m_int_plus_1"), dict):
                    blk["plus_1_strength_1"][ts] = u["depth_m_int_plus_1"]
            if b is None:
                blk["missing_units"].append("base_s%d" % ts)
            else:
                if str(m) in b["depths"]:
                    blk["strength_0"][ts] = b["depths"][str(m)]
                else:
                    blk["missing_units"].append("base_s%d_depth%d" % (ts, m))
                if str(m + 1) in b["depths"]:
                    blk["plus_1_strength_0"][ts] = b["depths"][str(m + 1)]
        last = [units_[(ts, m)].get("depth_m_int_plus_1") for ts in train_seeds if (ts, m) in units_]
        is_last = bool(last) and all(isinstance(x, str) for x in last)
        main = verdict_for(blk["strength_1"], window, "all", succ, train_seeds)
        out = {"m_int": m, "missing_units": blk["missing_units"], "judged_window": window,
               "judged_splits": {"a": "train (test episodes of the training levels)", "b_and_c": "all (all test episodes pooled); EXPERIMENTER CHOICE, the config does not name the split. Per-split verdicts below"},
               "verdict": main,
               "verdict_by_split_b_c": {sp: verdict_for(blk["strength_1"], window, sp, succ, train_seeds) for sp in ("train", "interp", "extrap")},
               "compact_depth_m_int": {"strength_1": {str(ts): compact(r, window) for ts, r in blk["strength_1"].items()}, "strength_0": {str(ts): compact(r, window) for ts, r in blk["strength_0"].items()}},
               "compact_depth_m_int_plus_1_auxiliary": ("해당 없음 (m_int=%d is the last block: no later block)" % m) if is_last else
               {"strength_1": {str(ts): compact(r, window) for ts, r in blk["plus_1_strength_1"].items()}, "strength_0": {str(ts): compact(r, window) for ts, r in blk["plus_1_strength_0"].items()},
                "note": "auxiliary observation, no verdict"},
               "records_depth_m_int": {"strength_1": {str(ts): r for ts, r in blk["strength_1"].items()}, "strength_0": {str(ts): r for ts, r in blk["strength_0"].items()}},
               "records_depth_m_int_plus_1": (None if is_last else {"strength_1": {str(ts): r for ts, r in blk["plus_1_strength_1"].items()}, "strength_0": {str(ts): r for ts, r in blk["plus_1_strength_0"].items()}}),
               "d_reproduction_vs_009_at_m_int": repro_vs_009(res009, m, train_seeds, {"1": blk["strength_1"], "0": blk["strength_0"]}, window)}
        blocks[str(m)] = out
    from exp002.common import env_info
    diag_p = paths["results_dir"] / "diagnostics.json"
    default_dir = B.default_results_dir().resolve()
    reasons = pilot_reasons(cfg, paths["results_dir"], m_overridden, seeds_overridden, m_ints, train_seeds)
    res = {"experiment": cfg["name"], "pilot": bool(reasons), "pilot_reasons": reasons, "pilot_config_flag": bool(cfg["pilot"]),
           "m_int_config": [int(x) for x in cfg["intervention"]["m_int"]], "m_int_used": m_ints, "m_int_overridden": bool(m_overridden), "m_int_source": source,
           "train_seeds": train_seeds, "train_seeds_overridden": bool(seeds_overridden), "strengths": [0.0, float(succ["strength"])], "method": cfg["intervention"]["method"],
           "success": dict(succ), "judged_window": window, "logistic_impl_defaults": MS.LOGISTIC_IMPL,
           "measure_config": {k: cfg["measure"][k] for k in ("probe_seeds", "b_mlp", "b_attentive", "c_action_only", "d_linear_repro", "a_level_classifier")},
           "missing_units": missing, "blocks": blocks, "config_consistency_with_009_and_002": [{"check": n, "ok": ok} for n, ok, _ in B.consistency_checks(cfg, c9, cfg2)],
           "diagnostics": (read_json(diag_p) if diag_p.exists() else None), "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(),
           "environment": env_info(), "code_version": code_version(paths["exp"]), "interpretation": None}
    write_json(res, paths["results_dir"] / "results.json")
    if Path(paths["results_dir"]).resolve() == default_dir:
        write_json(res, paths["exp"] / "results.json")
    log("aggregate done (pilot=%s): %s" % (res["pilot"], {m: b["verdict"]["overall"]["status"] for m, b in blocks.items()}))
    return res
