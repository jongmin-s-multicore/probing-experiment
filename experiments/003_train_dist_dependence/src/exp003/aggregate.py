"""Collect results/<run>/metrics.json into results.json and apply the verdict table (no interpretation text)."""
from __future__ import annotations

import platform
import subprocess
import time

from .common import EXP_DIR, read_json, write_json, all_models, run_dir
from . import verdict


def _git(args, cwd):
    try:
        return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def run(cfg, paths, out_path=None, base_dir=None):
    res = {"experiment": cfg["name"], "created": time.strftime("%F %T"), "hypothesis_ids": cfg["hypothesis"],
           "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "environment": {"python": platform.python_version(), "platform": platform.platform()},
           "experiment_code_git": _git(["rev-parse", "HEAD"], EXP_DIR), "acwm_phys_src_git": _git(["rev-parse", "HEAD"], paths["acwm_src"]),
           "maskcheck": None, "smoke": None, "runs": {}, "missing_runs": []}
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
    sp = paths["results_dir"] / "smoke.json"
    if sp.exists():
        res["smoke"] = read_json(sp)
    by_key = {}
    for spec in all_models(cfg):
        dist, seed = spec.split(":")
        name = cfg["train"]["run_name_template"].format(dist=dist, seed=seed)
        mp = paths["results_dir"] / name / "metrics.json"
        if not mp.exists():
            res["missing_runs"].append(name)
            continue
        m = read_json(mp)
        rd = run_dir(paths, name)
        if (rd / "train_manifest.json").exists():
            tm = read_json(rd / "train_manifest.json")
            m["train"] = {"n_episodes": tm["n_episodes"], "group_counts": tm["group_counts"]}
        res["runs"][name] = m
        by_key[spec] = m
    if not res["runs"]:
        raise RuntimeError("aggregate: no results/<run>/metrics.json found; refusing to write an empty results.json")
    res["verdict"] = verdict.build(cfg, by_key, base_dir or EXP_DIR)
    write_json(res, out_path or EXP_DIR / "results.json")
    return res
