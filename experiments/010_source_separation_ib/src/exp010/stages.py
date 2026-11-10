"""Stages of experiment 010: check | heads | report | selftest. Re-uses 005's tasks / heads / metrics code unmodified. Not run on the authoring machine (needs GPU caches)."""
from __future__ import annotations

import csv
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from exp002 import cache as CA
from exp002 import evalgrid as EG
from exp005 import feats as FE
from exp005 import stages as S5
from . import base as B
from . import stats as ST
from .base import log, read_json, write_json

PRED_DIR = "predictions"
MODELS = (("real", "ours"), ("rand", "random"))


def pred_file(paths, task_seed, layer, model, head_seed):
    return paths["results_dir"] / PRED_DIR / ("ts%d__%s__%s__seed%d.npz" % (task_seed, layer, model, head_seed))


def plan_of(cfg):
    names = [r["name"] for r in cfg["representations"]]
    return {"layers": [B.layer_dict(r) for r in cfg["representations"]], "layer_subexps": {n: list(cfg["subexps"]) for n in names}, "hold": False,
            "rows_matched": ["010_fixed"], "unresolved": [], "pilot": bool(cfg["pilot"]), "subexps": list(cfg["subexps"])}


def n_heads(cfg, max_tasks=None):
    n_task = max_tasks or cfg["_c5"]["ib"]["n_tasks"]
    ts, hs = len(cfg["seeds"]["task_a"]), len(cfg["seeds"]["head"])
    reps = len(cfg["representations"]) * 2
    return {"attentive_rep_heads": ts * hs * reps * n_task, "base_heads_attn_area": ts * hs * n_task, "base_small_oracle_heads": ts * hs * n_task * 2}


def missing_inputs(cfg, paths):
    """Preflight list (no GPU, no attempt consumed): the four representation files and the 002 Q1 cache DONE marker. Empty list = all present."""
    suffix = cfg["_c5"]["exp4"]["cache_suffix"]
    bad = []
    for l in plan_of(cfg)["layers"]:
        for ctl, _ in MODELS:
            p = FE.layer_path(paths, l, ctl, suffix)
            if not p.exists():
                bad.append(str(p))
    if not (CA.cache_dir(B.B5.as002(paths), suffix) / "DONE").exists():
        bad.append("002 Q1 cache DONE (%s)" % CA.cache_dir(B.B5.as002(paths), suffix))
    return bad


def code_version(exp_dir):
    """Code identity for results.json: git HEAD of the checkout containing exp_dir (None if not a git checkout), plus env EXP010_CODE_REPO_URL / EXP010_CODE_REF set by the Colab notebook."""
    import subprocess

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(exp_dir)] + list(a), capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    head = git("rev-parse", "HEAD")
    return {"git_head": head, "git_status_dirty": (bool(git("status", "--porcelain", "--", ".")) if head else None),
            "repo_url": os.environ.get("EXP010_CODE_REPO_URL"), "requested_ref": os.environ.get("EXP010_CODE_REF")}


def check(cfg, c2, paths):
    plan = plan_of(cfg)
    suffix = cfg["_c5"]["exp4"]["cache_suffix"]
    ok = True
    for l in plan["layers"]:
        for ctl, _ in MODELS:
            p = FE.layer_path(paths, l, ctl, suffix)
            print("  %-10s %-4s %s exists=%s" % (l["name"], ctl, p, p.exists()))
            ok &= p.exists()
    print("002 Q1 cache DONE:", (CA.cache_dir(B.B5.as002(paths), suffix) / "DONE").exists())
    print("head plan:", n_heads(cfg))
    return 0 if ok else 1


def heads_stage(cfg, c2, paths, task_seeds, head_seeds, max_tasks=None, dry_run=False):
    plan = plan_of(cfg)
    log("heads plan: reps %s x (ours, random) ; task seeds %s ; head seeds %s ; head counts %s ; eval n_inputs %d ; sub-experiments %s (5-A tasks only)" % (
        [l["name"] for l in plan["layers"]], task_seeds, head_seeds, n_heads(cfg, max_tasks), cfg["eval"]["n_inputs"], cfg["subexps"]))
    if dry_run:
        return None
    import torch
    suffix = cfg["_c5"]["exp4"]["cache_suffix"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(B.B5.as002(paths), suffix)
    W = EG.windows(c2)
    frames = list(W["union"])
    Etr = sum(r["part"] == "train" for r in C["rows"])
    nF = len(frames)
    ac, hc = c2["probe_impl"]["attentive"], cfg["_c5"]["head"]
    (paths["results_dir"] / PRED_DIR).mkdir(parents=True, exist_ok=True)
    arrs = {}
    for l in plan["layers"]:
        for ctl, mname in MODELS:
            p = FE.layer_path(paths, l, ctl, suffix)
            if not p.exists():
                raise FileNotFoundError("%s missing (004 enc_cache for encoder layers / 005 feat for predictor layers)" % p)
            arrs[(l["name"], mname)] = np.load(p, mmap_mode="r")
    cost = {"entries": []}
    for ts in task_seeds:
        c5 = B.cfg_for_task_seed(cfg["_c5"], ts)
        T, meta = S5.build_tasks(c5, c2, C["rows"], np.asarray(C["mask_feat"]), frames, [frames.index(f) for f in W["early"]], [frames.index(f) for f in W["late"]])
        np.savez_compressed(paths["results_dir"] / ("tasks_ts%d.npz" % ts), **T)
        write_json(meta, paths["results_dir"] / ("tasks_meta_ts%d.json" % ts))
        bs = S5.base_sources(c5, T, dev)
        for hs in head_seeds:
            for name, src in bs.items():
                f = pred_file(paths, ts, "base", name, hs)
                if not f.exists():
                    t0 = time.time()
                    np.savez_compressed(f, **S5.predict_model(src, T, {"A"}, hs, ac, hc, max_tasks, tag="ts%d/base/%s/s%d" % (ts, name, hs)))
                    cost["entries"].append({"what": f.name, "seconds": time.time() - t0})
        for (lay, mname), arr in arrs.items():
            src = S5._sources(c5, T, arr, Etr, nF, dev)
            for hs in head_seeds:
                f = pred_file(paths, ts, lay, mname, hs)
                if f.exists():
                    continue
                t0 = time.time()
                np.savez_compressed(f, **S5.predict_model(src, T, {"A"}, hs, ac, hc, max_tasks, tag="ts%d/%s/%s/s%d" % (ts, lay, mname, hs)))
                cost["entries"].append({"what": f.name, "seconds": time.time() - t0})
                log("heads ts%d %s %s head seed %d done (%.0fs)" % (ts, lay, mname, hs, time.time() - t0))
            del src
    cost["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    write_json(cost, paths["results_dir"] / "cost.json")
    return True


def load_preds(paths, plan, ts, head_seeds):
    P = {}
    for hs in head_seeds:
        for name in ("oracle_linear", "oracle_mlp", "area"):
            f = pred_file(paths, ts, "base", name, hs)
            if f.exists():
                P[("base", name, hs)] = dict(np.load(f))
        for l in plan["layers"]:
            for _, m in MODELS:
                f = pred_file(paths, ts, l["name"], m, hs)
                if f.exists():
                    P[(l["name"], m, hs)] = dict(np.load(f))
    return P


def collect_ib_rows(cfg, plan, tasks_by_ts, preds_by_ts, head_seeds):
    """Pure: for every task seed call 005's compute_metrics on that task seed's tasks and predictions; tag rows with task_seed."""
    rows = []
    for ts, T in tasks_by_ts.items():
        c5 = B.cfg_for_task_seed(cfg["_c5"], ts)
        R = S5.compute_metrics(c5, T, preds_by_ts[ts], plan, head_seeds)
        for r in R["ib"]:
            r = dict(r)
            r["task_seed"] = ts
            rows.append(r)
    return rows


def analyze(cfg, plan, ib_rows):
    tc = cfg["tost"]
    reps = [(l["name"], m) for l in plan["layers"] for _, m in MODELS]
    return {
        "representation_scores": ST.representation_table(ib_rows),
        "H_f_tost": ST.h_f_tost(ib_rows, reps, tc["pair_group_hf"], tc["delta"], tc["alpha"]),
        "pairwise_differences": ST.pairwise_cmp(ib_rows, reps, tc["cmp_metrics"], tc["pair_group_cmp"], tc["delta"], tc["alpha"]),
    }


IB_FIELDS = ["task_seed", "layer", "model", "metric", "pair_group", "seed", "raw", "score", "score_alt", "ref_mode"]


def report_stage(cfg, c2, paths, head_seeds, dry_run=False):
    plan = plan_of(cfg)
    log("report plan: reads results/tasks_ts*.npz and results/predictions/ts*__*.npz, computes 005 metrics per task seed, TOST delta=%s" % cfg["tost"]["delta"])
    if dry_run:
        return None
    out = paths["results_dir"]
    tasks_by_ts, preds_by_ts, missing = {}, {}, []
    for ts in cfg["seeds"]["task_a"]:
        f = out / ("tasks_ts%d.npz" % ts)
        if not f.exists():
            missing.append(str(f))
            continue
        z = np.load(f)
        tasks_by_ts[ts] = {k: z[k] for k in z.files}
        preds_by_ts[ts] = load_preds(paths, plan, ts, head_seeds)
    if not tasks_by_ts:
        raise FileNotFoundError("no results/tasks_ts*.npz - run `python run.py heads` first")
    rows = collect_ib_rows(cfg, plan, tasks_by_ts, preds_by_ts, head_seeds)
    with open(out / "ib_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=IB_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(r.get(k), float) and not np.isfinite(r.get(k))) else r.get(k)) for k in IB_FIELDS})
    A = analyze(cfg, plan, rows)
    from exp002.common import env_info
    res = {"experiment": cfg["name"], "pilot": bool(cfg["pilot"]), "task_seeds_used": sorted(tasks_by_ts), "task_seeds_missing_files": missing, "head_seeds": list(head_seeds),
           "eval_n_inputs": cfg["eval"]["n_inputs"], "tost_delta": cfg["tost"]["delta"], "tost_alpha": cfg["tost"]["alpha"], "tost_unit": cfg["tost"]["unit"],
           "normalization": cfg["_c5"]["normalization"], "analysis": A, "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(),
           "config005_sha256": hashlib.sha256((B.EXP_DIR / cfg["base"]["config005"]).resolve().read_bytes()).hexdigest(), "environment": env_info(), "code_version": code_version(paths["exp"]), "interpretation": None}
    write_json(res, paths["exp"] / "results.json")
    log("report done: %d ib rows" % len(rows))
    return res
