"""Stages of experiment 009: check | intervene | aggregate | selftest. Units = (block m_int, strength alpha, train seed); alpha = 0 is the shared baseline (one per train seed).
Resume: a unit whose part file exists is skipped. Heavy parts need GPU + the 002 cache + the 002 condition-(b) checkpoints; not run on the authoring machine."""
from __future__ import annotations

import csv
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
from exp004 import depthlib as DL
from exp004 import stages as S4
from . import base as B
from . import auxrules as AX
from . import curves as CU
from . import erase as ER
from .base import log, read_json, write_json

FIELDS = DL.CSV_FIELDS + ["m_int", "alpha"]


def resolve_m_int(cfg, cli_value=None):
    """Returns (list, overridden_flag, source). Override order: CLI --m-int, env EXP009_M_INT (comma separated), config."""
    cfgv = [int(x) for x in cfg["intervention"]["m_int"]]
    src, val = "config", cfgv
    env = os.environ.get("EXP009_M_INT")
    if cli_value:
        src, val = "cli", [int(x) for x in str(cli_value).split(",") if x.strip()]
    elif env:
        src, val = "env:EXP009_M_INT", [int(x) for x in env.split(",") if x.strip()]
    val = sorted(set(val))
    if not val:
        raise ValueError("empty m_int list")
    return val, (val != sorted(set(cfgv))), src


def tag(x):
    return ("%g" % float(x)).replace(".", "p")


def part_path(paths, kind, ts, m=None, a=None):
    d = paths["results_dir"] / "parts"
    return d / (("base_s%d.csv" % ts) if kind == "base" else ("unit_m%d_a%s_s%d.csv" % (m, tag(a), ts)))


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    DL.write_rows(rows, path, fields=FIELDS)


def units(m_ints, strengths, train_seeds):
    out = [("base", ts, None, 0.0) for ts in train_seeds]
    out += [("unit", ts, m, float(a)) for ts in train_seeds for m in m_ints for a in strengths if float(a) > 0]
    return out


def unit_done(paths, u):
    kind, ts, m, a = u
    return part_path(paths, kind, ts, m, a).exists()


def plan_text(cfg, m_ints, strengths, train_seeds, paths):
    us = units(m_ints, strengths, train_seeds)
    done = [u for u in us if unit_done(paths, u)]
    return ("intervene plan: m_int %s, strengths %s (0 = shared baseline), train seeds %s, method %s ; units total %d (baseline %d + intervention %d), already done %d -> to run %d ; probe kinds %s seeds %s" % (
        m_ints, strengths, train_seeds, cfg["intervention"]["method"], len(us), len(train_seeds), len(us) - len(train_seeds), len(done), len(us) - len(done), cfg["probe"]["kinds"], cfg["probe"]["seeds"]))


def metadata_missing(manifests):
    """Preflight (no GPU, no attempts): ManifestDataset (via exp002 model_io.action_stats_for_run) reads <episode dir>/metadata.pt for every
    episode dir in each run's train_manifest.json. manifests: list of (label, manifest path). Returns a list of problem strings (empty = ok)."""
    bad = []
    for label, mp in manifests:
        mp = Path(mp)
        if not mp.exists():
            bad.append("%s: train_manifest.json missing (%s)" % (label, mp))
            continue
        dirs = sorted({ep["dir"] for ep in json.loads(mp.read_text())["episodes"]})
        for d in dirs:
            if not (Path(d) / "metadata.pt").exists():
                bad.append("%s: %s/metadata.pt missing" % (label, d))
    return bad


def metadata_preflight(c4, paths, train_seeds):
    return metadata_missing([("train seed %d" % ts, B.B4.run_dir002(paths, c4, ts) / "train_manifest.json") for ts in train_seeds])


def check(cfg, c4, cfg2, paths, train_seeds=None):
    print("cache dir (tmp features, directions):", paths["cache"], "exists=%s" % paths["cache"].exists())
    print("002 Q1 cache DONE:", (CA.cache_dir(B.B4.as002(paths), "") / "DONE").exists())
    ok = True
    for ts in cfg["predictor"]["train_seeds"]:
        rd = B.B4.run_dir002(paths, c4, ts)
        print("002 run %s DONE=%s" % (rd.name, (rd / "DONE").exists()))
        ok &= (rd / "DONE").exists()
    print("004 depth_r2.csv (reproduction check input):", (paths["results004"] / "depth_r2.csv").exists())
    bad = metadata_preflight(c4, paths, train_seeds if train_seeds is not None else cfg["predictor"]["train_seeds"])
    print("train_manifest episode dirs with metadata.pt (needed for action mean/std; mp4 files are NOT needed): %s" % ("all present" if not bad else "MISSING %d" % len(bad)))
    for b in bad[:20]:
        print("  !! " + b)
    return 0 if ok and not bad else 1


def labels_positions(cfg2, C, frames):
    rows, tr, te, Y = S4.labels(cfg2, C)
    return rows, len(tr), te, Y[:, frames]


def probe_rows(cfg2, kinds, seeds, tmp, depths, Yp, Etr, te, levels, pos_windows, base):
    out = []
    for m in depths:
        F = np.load(Path(tmp) / ("feat_%d.npy" % m), mmap_mode="r")
        b = dict(base)
        b.update({"module": "predictor", "depth": m, "fit": "new"})
        out += S4.probe_hidden(cfg2, kinds, seeds, F, Yp[:Etr], Yp[Etr:], Etr, pos_windows, te, levels, b, ctrl_name="real")
        del F
    return out


def intervene_stage(cfg, c4, cfg2, paths, m_ints, strengths, train_seeds, fresh=False, dry_run=False):
    log(plan_text(cfg, m_ints, strengths, train_seeds, paths))
    if dry_run:
        return None
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic = cfg["intervention"]
    H = B.cget(cfg2, "train.num_hist")
    if H is None:
        raise B.ConfigNull("002 config train.num_hist is null")
    dz = 384 if cfg["predictor"]["feature_part"] == "z_only" else 404
    if dz != 384:
        raise ValueError("009 intervenes on the z part only (feature_part must be z_only)")
    C = CA.load(B.B4.as002(paths), "")
    rows, Etr, te, Yp0 = None, None, None, None
    frames, pos_windows = S4.frames_and_windows(cfg2)
    rows, Etr, te, Yp = labels_positions(cfg2, C, frames)
    levels = list(cfg2["data"]["levels"])
    kinds, pseeds = cfg["probe"]["kinds"], cfg["probe"]["seeds"]
    meta_units = {}
    for ts in train_seeds:
        todo = [u for u in units(m_ints, strengths, [ts]) if fresh or not unit_done(paths, u)]
        if not todo:
            log("train seed %d: all units done" % ts)
            continue
        rdir = B.B4.run_dir002(paths, c4, ts)
        if not (rdir / "DONE").exists():
            raise FileNotFoundError("002 run %s has no DONE (condition (b) checkpoint missing)" % rdir)
        model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
        if M.encoder_fingerprint(model.encoder) != C["meta"]["encoder_fingerprint"]:
            raise RuntimeError("encoder of %s differs from the Q1 cache encoder" % rdir)
        assert int(tcfg.num_hist) == int(H)
        mean, std, _ = M.action_stats_for_run(cfg2, rdir, tcfg)
        log("train seed %d: epoch %d, %d predictor blocks, %d units to run" % (ts, ep, M.n_blocks(model), len(todo)))
        meta_units.update(run_seed(cfg, cfg2, paths, model, C, H, mean, std, dev, frames, pos_windows, Yp, Etr, te, levels, dz, ts, todo, m_ints, strengths, fresh))
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if meta_units:
        p = paths["results_dir"] / "diagnostics.json"
        d = read_json(p) if p.exists() else {}
        d.update(meta_units)
        write_json(d, p)
    return True


def run_seed(cfg, cfg2, paths, model, C, H, mean, std, dev, frames, pos_windows, Yp, Etr, te, levels, dz, ts, todo, m_ints, strengths, fresh=False):
    """All pending units of one train seed with an already loaded model. Returns diagnostics dict."""
    ic = cfg["intervention"]
    kinds, pseeds = cfg["probe"]["kinds"], cfg["probe"]["seeds"]
    bs = int(cfg["predictor"]["batch_episodes"])
    meta_units = {}
    L = M.n_blocks(model)
    bad = [m for m in m_ints if not (1 <= m <= L)]
    if bad:
        raise ValueError("m_int %s outside 1..%d" % (bad, L))
    tmp = paths["cache"] / ("tmp_s%d" % ts)
    if ("base", ts, None, 0.0) in todo:
        ER.extract_hidden(model, C, H, tmp, mean, std, dev, frames, dz, list(range(L + 1)), None, bs, label="s%d base" % ts)
        r = probe_rows(cfg2, kinds, pseeds, tmp, range(L + 1), Yp, Etr, te, levels, pos_windows, {"train_seed": ts, "m_int": "NA", "alpha": 0.0})
        write_rows(r, part_path(paths, "base", ts))
        shutil.rmtree(tmp, ignore_errors=True)
        log("s%d baseline done (%d rows)" % (ts, len(r)))
    rng = np.random.RandomState(int(ic["fit_seed"]) + ts)
    fit_ep = np.sort(rng.permutation(Etr)[: int(round(float(ic["fit_fraction"]) * Etr))])
    for m in m_ints:
        pending = [a for a in strengths if float(a) > 0 and ("unit", ts, m, float(a)) in todo]
        if not pending:
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
        for a in pending:
            depths = list(range(m, L + 1))
            er = {"m_int": m, "alpha": float(a), "E": E, "dz": dz}
            out = ER.extract_hidden(model, C, H, tmp, mean, std, dev, frames, dz, depths, er, bs, label="s%d m%d a%g" % (ts, m, a))
            r = probe_rows(cfg2, kinds, pseeds, tmp, depths, Yp, Etr, te, levels, pos_windows, {"train_seed": ts, "m_int": m, "alpha": float(a)})
            # diagnostic: std of the fitted readout a.x after erasure (pooled z at block m_int) relative to its std on the fit data before erasure; held-out train episodes and test episodes
            pm = out["pooled_mint"]
            held = np.setdiff1d(np.arange(Etr), fit_ep)
            te_idx = np.arange(Etr, pm.shape[0])
            ratio = lambda idx: float((pm[idx].reshape(-1, dz) @ E["a"]).std() / E["readout_std_fit"])
            meta_units["m%d_a%s_s%d" % (m, tag(a), ts)] = {"corr_readout_fit_episodes_pre": E["corr_fit"], "readout_std_ratio_after_over_fit_pre_train_heldout": ratio(held) if len(held) else None,
                                                          "readout_std_ratio_after_over_fit_pre_test": ratio(te_idx)}
            write_rows(r, part_path(paths, "unit", ts, m, a))
            shutil.rmtree(tmp, ignore_errors=True)
            log("s%d m_int %d alpha %g done (%d rows)" % (ts, m, a, len(r)))
    return meta_units


def read_part(path):
    return DL.read_rows(path)


def code_version(exp_dir):
    """Code identity for results.json: git HEAD of the checkout containing exp_dir (None if not a git checkout), plus env EXP009_CODE_REPO_URL / EXP009_CODE_REF set by the Colab notebook."""
    import subprocess
    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(exp_dir)] + list(a), capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    return {"git_head": git("rev-parse", "HEAD"), "git_status_dirty": (bool(git("status", "--porcelain", "--", ".")) if git("rev-parse", "HEAD") else None),
            "repo_url": os.environ.get("EXP009_CODE_REPO_URL"), "requested_ref": os.environ.get("EXP009_CODE_REF")}


def aggregate(cfg, c4, cfg2, paths, m_ints, overridden, source, strengths, train_seeds, dry_run=False):
    log("aggregate: m_int %s (source %s, overridden=%s), strengths %s, train seeds %s" % (m_ints, source, overridden, strengths, train_seeds))
    if dry_run:
        return None
    prim, thr, splits = cfg["depth"]["primary"], float(cfg["depth"]["drop_threshold"]), list(cfg["depth"]["splits"])
    succ = cfg["success"]
    base = {}
    missing = []
    for ts in train_seeds:
        p = part_path(paths, "base", ts)
        if p.exists():
            base[ts] = read_part(p)
        else:
            missing.append(str(p.name))
    p004 = paths["results004"] / "depth_r2.csv"
    rows004 = DL.read_rows(p004) if p004.exists() else None      # random-init predictor reference for aux rule A (004 rows, same probe code/splits/seeds; see README)
    rnd_ref = {ts: AX.random_init_reference(rows004, ts, prim, base[ts]) for ts in base}
    slack = int(succ["slack"])
    res_blocks, all_rows = {}, []
    for ts, r in base.items():
        all_rows += r
    for m in m_ints:
        blk = {"m_int": m, "strengths": {}, "missing_units": []}
        stars_succ, complete = {}, False
        for a in strengths:
            a = float(a)
            per_seed = {}
            for ts in train_seeds:
                if a == 0:
                    if ts not in base:
                        continue
                    ur, brows = base[ts], base[ts]
                else:
                    p = part_path(paths, "unit", ts, m, a)
                    if ts not in base or not p.exists():
                        blk["missing_units"].append("m%d_a%g_s%d" % (m, a, ts))
                        continue
                    ur, brows = read_part(p), base[ts]
                    all_rows += ur
                cur = {s: CU.unit_curve(brows, ur, ts, s, prim, m if a > 0 else 0) for s in splits}
                mstar, per = CU.m_star_of(cur, thr)
                mm = m if a > 0 else 0
                rr, info = rnd_ref[ts]
                per_seed[ts] = {"m_star": mstar, "m_star_by_split": per, "curves": {s: {str(d): v for d, v in sorted(c.items())} for s, c in cur.items()},
                                "aux_A_delta_vs_random_init": AX.rule_A(cur, rr, info, splits, thr, mm, slack, ts),
                                "aux_B_common_variance_r2": AX.rule_B(AX.combined_rows(brows, ur, ts, prim, mm), cfg2, splits, thr, mm, slack)}
            blk["strengths"]["%g" % a] = per_seed
            if abs(a - float(succ["strength"])) < 1e-12:
                stars_succ = {ts: v["m_star"] for ts, v in per_seed.items()}
                complete = len(per_seed) == len(train_seeds)
        if stars_succ and complete:
            blk["verdict"] = CU.block_verdict(stars_succ, m, int(succ["slack"]), int(succ["min_seeds"]))
        else:
            blk["verdict"] = {"status": "not_evaluable", "reason": "units_missing_at_success_strength", "missing": [x for x in blk["missing_units"] if "_a%g_" % float(succ["strength"]) in x]}
        # auxiliary (reported only): R2 recovery after the intervention block at the success strength, mean over train seeds, per split
        aux = {}
        sa = blk["strengths"].get("%g" % float(succ["strength"]), {})
        for s in splits:
            vals = {}
            for ts, v in sa.items():
                c = v["curves"][s]
                if str(m) in c:
                    for d, x in c.items():
                        if int(d) > m:
                            vals.setdefault(int(d), []).append(x - c[str(m)])
            aux[s] = {str(d): float(np.mean(v)) for d, v in sorted(vals.items())}
        if all(len(v) == 0 for v in aux.values()):
            aux = "해당 없음 (m_int=%d is the last block: no later block, recovery not computed; detection verdict only)" % m
        blk["aux_recovery_r2_minus_r2_at_m_int"] = aux
        blk["aux_rules_at_success_strength"] = {
            k: {"status_by_train_seed": {str(ts): v[key]["status"] for ts, v in sa.items()},
                "n_seeds_observed": int(sum(v[key]["status"] == AX.OBS for v in sa.values())),
                "n_seeds_observed_within_m_int_plus_slack": int(sum(bool(v[key].get("within_m_int_plus_slack")) for v in sa.values())),
                "n_seeds_not_computable": int(sum(v[key]["status"] == AX.NA for v in sa.values())), "n_seeds": len(sa),
                "note": "auxiliary only; does not enter the primary verdict"}
            for k, key in (("A", "aux_A_delta_vs_random_init"), ("B", "aux_B_common_variance_r2"))}
        res_blocks[str(m)] = blk
    repro = reproduction_check(paths, base, prim, splits)
    from exp002.common import env_info
    diag_p = paths["results_dir"] / "diagnostics.json"
    res = {"experiment": cfg["name"], "pilot": bool(cfg["pilot"]), "m_int_config": [int(x) for x in cfg["intervention"]["m_int"]], "m_int_used": m_ints, "m_int_overridden": bool(overridden),
           "m_int_source": source, "strengths": [float(a) for a in strengths], "train_seeds": train_seeds, "method": cfg["intervention"]["method"],
           "success_rule_draft": dict(succ), "drop_threshold": thr, "primary": dict(prim), "missing_baseline": missing, "blocks": res_blocks,
           "baseline_vs_004": repro, "random_init_reference": {"source": "004 results/depth_r2.csv (module predictor, control random_init, same train seed)", "file_found": rows004 is not None, "by_train_seed": {str(ts): v[1] for ts, v in rnd_ref.items()}}, "diagnostics": (read_json(diag_p) if diag_p.exists() else None), "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(),
           "environment": env_info(), "code_version": code_version(paths["exp"]), "interpretation": None}
    write_json(res, paths["exp"] / "results.json")
    DL.write_rows(all_rows, paths["results_dir"] / "depth_r2_009.csv", fields=FIELDS)
    log("aggregate done: %s" % {m: b["verdict"]["status"] for m, b in res_blocks.items()})
    return res


def reproduction_check(paths, base, prim, splits):
    """The alpha=0 baseline should reproduce 004's predictor curves (same checkpoints, probes, seeds). Reported as max abs difference; no tolerance asserted."""
    p = paths["results004"] / "depth_r2.csv"
    if not p.exists() or not base:
        return {"status": "not_available", "reason": "004 depth_r2.csv missing or no baseline units"}
    r4 = DL.read_rows(p)
    out = {}
    for ts, rows in base.items():
        for s in splits:
            a = CU.seed_curve(rows, ts, s, prim)
            b = CU.seed_curve(r4, ts, s, prim)
            common = sorted(set(a) & set(b))
            out["s%d/%s" % (ts, s)] = float(max(abs(a[d] - b[d]) for d in common)) if common else None
    return {"status": "computed", "max_abs_r2_diff": out, "note": "004 uses probe kinds linear+attentive; only the primary linear rows are compared"}
