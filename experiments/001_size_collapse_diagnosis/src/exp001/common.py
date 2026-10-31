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
    return cfg


def resolve_paths(cfg):
    """All locations come from environment variables (names in config.paths); nothing is hardcoded."""
    p = cfg["paths"]
    work = Path(os.environ.get(p["work_env"], str(Path.home() / "dinowm"))).expanduser()
    dataset_dir = Path(os.environ.get(p["dataset_env"], str(work / "data"))).expanduser()
    out = {
        "work": work,
        "dataset_dir": dataset_dir,
        "clothmove": dataset_dir / p["clothmove_subdir"],
        "acwm_src": Path(os.environ.get(cfg["code"]["src_env"], str(work / "acwm_phys_src"))).expanduser(),
        "train_base": Path(os.environ.get(p["train_env"], str(work / "train_runs"))).expanduser(),
        "cache": Path(os.environ.get(p["cache_env"], str(work / "exp001_cache"))).expanduser(),
        "exp": EXP_DIR,
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
    """Size group of an episode from its split and file name (ind_test/ind_train -> 'ind', ood_test -> 'smaller'/'larger',
    generated split 'gen_<tag>' -> 'gen_<tag>')."""
    if split.startswith("gen_"):
        return split
    if split == "ood_test":
        for g, prefix in cfg["data"]["ood_group_prefix"].items():
            if name.startswith(prefix):
                return g
        raise ValueError(f"cannot infer ood group from file name {name}")
    return "ind"
