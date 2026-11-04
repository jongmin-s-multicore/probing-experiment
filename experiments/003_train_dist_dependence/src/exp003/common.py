"""Config / path / json / logging helpers (adapted from experiments/001 src/exp001/common.py; legacy acwm_phys dataset removed,
002-style retry guard and conda warning added). No torch import here so CPU-only stages work without it. Python 3.9 compatible."""
from __future__ import annotations

import collections
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[2]


class ConfigNull(RuntimeError):
    """A config value that a human must set (null) was needed."""


def _strict_loader():
    import yaml

    class L(yaml.SafeLoader):
        pass

    def cm(loader, node, deep=False):
        ks = [loader.construct_object(k) for k, _ in node.value]
        dup = [k for k, n in collections.Counter(ks).items() if n > 1]
        if dup:
            raise ValueError("duplicate YAML keys (the later one would silently win): %s" % dup)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    L.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, cm)
    return L


def load_config(path=None):
    import yaml
    path = Path(path) if path else EXP_DIR / "config.yaml"
    with open(path) as f:
        cfg = yaml.load(f, Loader=_strict_loader())
    cfg["_config_path"] = str(path)
    return resolve_dataset(cfg)


def resolve_dataset(cfg):
    """Select the dataset block and copy what the stages read (train.distributions as {name: specs}, total_episodes, hydra overrides,
    eval sets, reference group). train.batch_size / train.epochs from config are put into the hydra overrides."""
    name = os.environ.get("EXP003_DATASET") or cfg["data"]["dataset"]
    blk = cfg["data"]["datasets"][name]
    cfg["data"]["active"] = name
    cfg["_ds"] = blk
    names = list(cfg["train"]["distributions"])
    cfg["train"]["distribution_names"] = names
    cfg["train"]["distributions"] = {n: blk["distributions"][n] for n in names}
    cfg["train"]["total_episodes"] = blk["total_episodes"]
    if blk.get("run_name_template"):
        cfg["train"]["run_name_template"] = blk["run_name_template"]
    ov = {"env": blk["train_hydra_env"]}
    ov.update(cfg["train"]["hydra_overrides"])
    ov["training.batch_size"] = cfg["train"]["batch_size"]
    ov["training.epochs"] = cfg["train"]["epochs"]
    cfg["train"]["hydra_overrides"] = ov
    cfg["diagnose"] = {"eval_sets": blk["eval_sets"], "reference_group": blk["reference_group"], "device": cfg["eval"]["device"],
                       "batch_episodes": cfg["eval"]["batch_episodes"], "save_decoded_frames": cfg["eval"]["save_decoded_frames"],
                       "max_episodes_per_group": cfg["eval"]["n_test_episodes_per_size"]}
    return cfg


def all_models(cfg):
    """Every 'distribution:seed' of the experiment (distributions x seeds from config.train)."""
    return ["%s:%d" % (d, s) for d in cfg["train"]["distribution_names"] for s in cfg["train"]["seeds"]]


def cget(cfg, dotted):
    """cfg['a']['b'] by 'a.b'; KeyError if missing."""
    cur = cfg
    for p in dotted.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            raise KeyError("config key missing: " + dotted)
    return cur


def area_ratio_of(cfg, group):
    for s in cfg["_ds"]["sizes"]:
        if s["name"] == group:
            return float(s["area_ratio"])
    return None


def size_roles(cfg, train_groups):
    """Role of every size w.r.t. the trained sizes (nominal area ratios): trained | interp | extrap_small | extrap_large."""
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
    env = os.environ.get("EXP003_APPROVED", "").split()
    return set(cfg["train"].get("approved_distributions") or []) | set(env)


def resolve_paths(cfg):
    """All locations come from environment variables (names in config.paths); nothing is hardcoded."""
    p = cfg["paths"]
    work = Path(os.environ.get(p["work_env"], str(Path.home() / "dinowm"))).expanduser()
    dataset_dir = Path(os.environ.get(p["dataset_env"], str(work / "data"))).expanduser()
    return {
        "work": work, "dataset_dir": dataset_dir, "data_root": dataset_dir / cfg["_ds"]["subdir"],
        "acwm_src": Path(os.environ.get(cfg["code"]["src_env"], str(work / "acwm_phys_src"))).expanduser(),
        "train_base": Path(os.environ.get(p["train_env"], str(work / "train_runs"))).expanduser(),
        "cache": Path(os.environ.get(p["cache_env"], str(work / "exp003_cache"))).expanduser(),
        "exp": EXP_DIR, "maskcheck_json": EXP_DIR / "results" / "maskcheck.json", "logs": EXP_DIR / "logs",
        "manifests": EXP_DIR / "manifests", "results_dir": EXP_DIR / "results",
    }


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


def _clean(o):
    """NaN/inf -> null (strict JSON); numpy scalars/arrays -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=2, allow_nan=False, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def log(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def group_of(split, name, cfg):
    """Size group of an episode from its split: '<size>_train' / '<size>_test' -> '<size>'."""
    return split.rsplit("_", 1)[0]


# ------------------------------------------------------------------------------------------ file logging
class Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


def start_file_log(paths, stage, tag=""):
    paths["logs"].mkdir(parents=True, exist_ok=True)
    lp = paths["logs"] / ("%s%s_%s.log" % (stage, ("_" + tag) if tag else "", time.strftime("%Y%m%d-%H%M%S")))
    fh = open(lp, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    log("log file: %s" % lp)
    return lp


def log_exception(e):
    print("ERROR %s: %s" % (type(e).__name__, e), file=sys.stderr)
    traceback.print_exc()


# ------------------------------------------------------------------------------------------ retry guard (CLAUDE.md: max 3 attempts)
def _attempts_path(paths):
    return paths["results_dir"] / "attempts.json"


def attempts_check(cfg, paths, key):
    p = _attempts_path(paths)
    d = read_json(p) if p.exists() else {}
    n = int(d.get(key, 0))
    mx = int(cfg["run"]["max_attempts"])
    if n >= mx:
        raise RuntimeError("%s failed %d times in a row (limit %d, CLAUDE.md). Stop and report the cause from logs/. "
                           "A human may delete the entry in %s to allow another try." % (key, n, mx, p))
    return n


def attempts_record(paths, key, ok):
    p = _attempts_path(paths)
    d = read_json(p) if p.exists() else {}
    d[key] = 0 if ok else int(d.get(key, 0)) + 1
    write_json(d, p)


def warn_conda():
    if not os.environ.get("CONDA_DEFAULT_ENV"):
        log("WARNING: CONDA_DEFAULT_ENV is empty - not inside an activated conda environment (python=%s). Continuing." % sys.executable)
        return False
    return True


def env_info():
    import platform
    info = {"python": sys.version.split()[0], "python_executable": sys.executable, "platform": platform.platform(),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"), "packages": {}}
    for m in ("torch", "numpy", "scipy", "yaml", "cv2", "decord", "hydra", "omegaconf", "einops"):
        try:
            mod = __import__(m)
            info["packages"][m] = getattr(mod, "__version__", "?")
        except Exception:
            info["packages"][m] = None
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception:
        pass
    return info
