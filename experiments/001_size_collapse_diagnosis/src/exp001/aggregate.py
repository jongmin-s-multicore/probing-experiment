"""Collect per-run result files into results.json (no interpretation). Also summarises the intervention records."""
from __future__ import annotations

import json
import platform
import subprocess
import time

import numpy as np

from .common import EXP_DIR, read_json, write_json, run_dir
from .diagnose import boot_ci


def _git(args, cwd):
    try:
        return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def summarize_intervention(iv):
    out = {"modes": {}}
    for mode, recs in iv["per_mode"].items():
        by_g = {}
        for r in recs:
            by_g.setdefault(r["group"], []).append(r)
        res = {}
        for g, rs in by_g.items():
            d_in = np.array([np.array(r["in_proj_mod"]) - np.array(r["in_proj_orig"]) for r in rs])          # (E,12)
            d_out = np.array([np.array(r["tf_out_proj_mod"]) - np.array(r["tf_out_proj_orig"]) for r in rs])
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = d_out.sum(1) / d_in.sum(1)
            rnd_in = np.array([np.array(r["random"]["in_shift"]) for r in rs])                               # (E,R,12)
            rnd_out = np.array([np.array(r["random"]["out_shift_along_r"]) for r in rs])
            with np.errstate(divide="ignore", invalid="ignore"):
                gain_r = rnd_out.sum(2) / rnd_in.sum(2)                                                       # (E,R)
            gr = np.nanmean(gain_r, axis=0)
            la = lambda a: np.log(np.maximum(np.array(a, float), 1))
            dl_u = np.array([la(r["tf_area_mod"]) - la(r["tf_area_orig"]) for r in rs])                       # (E,3)
            dl_r = np.array([la(r["random"]["area_mod"]) - la(r["tf_area_orig"])[None] for r in rs])          # (E,R,3)
            ol_dp = np.array([np.array(r["ol_out_proj_mod"]) - np.array(r["ol_out_proj_orig"]) for r in rs])  # (E,12)
            ol_dl = np.array([la(r["ol_area_mod"]) - la(r["ol_area_orig"]) for r in rs])
            res[g] = {
                "n_episodes": len(rs),
                "tf_response_gain_along_u": boot_ci(gain),
                "tf_response_gain_along_random_r": {"per_direction_mean": gr.tolist(), "mean": float(np.nanmean(gr)), "p5": float(np.nanpercentile(gr, 5)), "p95": float(np.nanpercentile(gr, 95))},
                "tf_decoded_dlogarea_u_by_frame": [boot_ci(dl_u[:, j]) for j in range(dl_u.shape[1])],
                "tf_decoded_dlogarea_random_mean_by_frame": np.nanmean(dl_r, axis=(0, 1)).tolist(),
                "ol_out_proj_shift_by_k": ol_dp.mean(0).tolist(),
                "ol_decoded_dlogarea_by_k": ol_dl.mean(0).tolist(),
            }
        out["modes"][mode] = res
    out["definition"] = {"tf_response_gain": "sum over frames of (pooled projection shift of predictor output) / (pooled projection shift of input), per episode",
                         "random": "same applied shift magnitude along unit random directions; output read along the same direction"}
    return out


def run(cfg, paths):
    res = {"experiment": cfg["name"], "created": time.strftime("%F %T"), "hypothesis_ids": cfg["hypothesis"], "cloth_sim_branch": cfg["code"]["cloth_sim_branch"],
           "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "environment": {"python": platform.python_version(), "platform": platform.platform()},
           "experiment_code_git": _git(["rev-parse", "HEAD"], EXP_DIR), "acwm_phys_src_git": _git(["rev-parse", "HEAD"], paths["acwm_src"]),
           "dataset": {"selected": cfg["data"]["active"], "block": cfg["_ds"]}, "maskcheck": None, "runs": {}}
    try:
        import torch
        res["environment"]["torch"] = torch.__version__
        res["environment"]["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            res["environment"]["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        res["environment"]["torch"] = None
    if paths["maskcheck_json"].exists():
        res["maskcheck"] = read_json(paths["maskcheck_json"])
    for spec in cfg["train"]["models_to_diagnose"]:
        dist, seed = spec.split(":")
        name = cfg["train"]["run_name_template"].format(dist=dist, seed=seed)
        r = {"distribution": dist, "seed": int(seed)}
        rd = run_dir(paths, name)
        if (rd / "train_manifest.json").exists():
            m = read_json(rd / "train_manifest.json")
            r["train"] = {"n_episodes": m["n_episodes"], "group_counts": m["group_counts"]}
            lg = rd / "epoch_logs.jsonl"
            if lg.exists():
                lines = [json.loads(x) for x in open(lg)]
                r["train"]["last_epoch_log"] = lines[-1] if lines else None
        d = paths["results_dir"] / name
        for part in ("repro_check", "diagnose", "probe"):
            if (d / f"{part}.json").exists():
                r[part] = read_json(d / f"{part}.json")
        if (d / "intervene.json").exists():
            r["intervene"] = summarize_intervention(read_json(d / "intervene.json"))
        res["runs"][name] = r
    write_json(res, EXP_DIR / "results.json")
    return res
