"""Config / path / json helpers. No torch import here so that CPU-only stages work without it."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

EXP_DIR = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else EXP_DIR / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return resolve_dataset(cfg)


def resolve_dataset(cfg):
    """Select the dataset block (config data.dataset, overridable by env EXP001_DATASET) and copy what the stages read:
    train.distributions / total_episodes / hydra env, diagnose.eval_sets / reference_group, probe.class_groups."""
    name = os.environ.get("EXP001_DATASET") or cfg["data"]["dataset"]
    blk = cfg["data"]["datasets"][name]
    cfg["data"]["active"] = name
    cfg["_ds"] = blk
    cfg["train"]["distributions"] = blk["distributions"]
    cfg["train"]["total_episodes"] = blk["total_episodes"]
    cfg["train"]["hydra_overrides"] = {"env": blk["train_hydra_env"], **cfg["train"]["hydra_overrides"]}
    cfg["diagnose"]["eval_sets"] = blk["eval_sets"]
    cfg["diagnose"]["reference_group"] = blk["reference_group"]
    cfg["probe"]["class_groups"] = {g: i for i, g in enumerate(blk["group_order"])}
    return cfg


def area_ratio_of(cfg, group):
    for s in cfg["_ds"]["sizes"]:
        if s["name"] == group:
            return float(s["area_ratio"])
    return None


def size_roles(cfg, train_groups):
    """Role of every size group w.r.t. the sizes a model was trained on (nominal area ratios from config):
    trained | interp (untrained, strictly inside [min,max] of the trained area ratios) | extrap_small | extrap_large."""
    tr = [area_ratio_of(cfg, g) for g in train_groups]
    lo, hi = min(tr), max(tr)
    roles = {}
    for s in cfg["_ds"]["sizes"]:
        g, a = s["name"], float(s["area_ratio"])
        if g in train_groups:
            roles[g] = "trained"
        elif lo < a < hi:
            roles[g] = "interp"
        elif a <= lo:
            roles[g] = "extrap_small"
        else:
            roles[g] = "extrap_large"
    return roles


def approved_distributions(cfg):
    env = os.environ.get("EXP001_APPROVED", "").split()
    return set(cfg["train"].get("approved_distributions") or []) | set(env)


def resolve_paths(cfg):
    """All locations come from environment variables (names in config.paths); nothing is hardcoded."""
    p = cfg["paths"]
    work = Path(os.environ.get(p["work_env"], str(Path.home() / "dinowm"))).expanduser()
    dataset_dir = Path(os.environ.get(p["dataset_env"], str(work / "data"))).expanduser()
    out = {
        "work": work,
        "dataset_dir": dataset_dir,
        "data_root": dataset_dir / cfg["_ds"]["subdir"],
        "acwm_src": Path(os.environ.get(cfg["code"]["src_env"], str(work / "acwm_phys_src"))).expanduser(),
        "train_base": Path(os.environ.get(p["train_env"], str(work / "train_runs"))).expanduser(),
        "cache": Path(os.environ.get(p["cache_env"], str(work / "exp001_cache"))).expanduser(),
        "exp": EXP_DIR,
        "maskcheck_json": EXP_DIR / "results" / "maskcheck.json",
        "logs": EXP_DIR / "logs",
        "manifests": EXP_DIR / "manifests",
        "results_dir": EXP_DIR / "results",
    }
    return out


def run_dir(paths, run_name):
    return paths["train_base"] / "outputs" / run_name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=2, default=_json_default, allow_nan=False)
    os.replace(tmp, path)


def _clean(o):
    """NaN/inf -> null (strict JSON); numpy scalars/arrays -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def group_of(split, name, cfg):
    """Size group of an episode from its split and file name.
    cloth_sim: split '<size>_train' / '<size>_test' -> '<size>'.
    acwm_phys (legacy): ind_train/ind_test -> 'ind', ood_test -> 'smaller'/'larger' by file-name prefix."""
    blk = cfg["_ds"]
    if blk["kind"] == "cloth_sim":
        return split.rsplit("_", 1)[0]
    if split == "ood_test":
        for g, prefix in blk["ood_group_prefix"].items():
            if name.startswith(prefix):
                return g
        raise ValueError(f"cannot infer ood group from file name {name}")
    return "ind"
