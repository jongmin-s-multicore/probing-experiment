"""Config / path / json / logging helpers (adapted from experiments/001 src/exp001/common.py; rewritten for 002).
No torch import at module level so CPU-only stages work without torch. Python 3.9 compatible."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parents[2]


class ConfigNull(RuntimeError):
    """A config value that must be set by a human (null) was needed."""


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
    return cfg


def config_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cget(cfg, dotted, default="__raise__"):
    """cfg['a']['b'] by 'a.b'. Raises KeyError if missing (unless default given)."""
    cur = cfg
    for p in dotted.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            if default != "__raise__":
                return default
            raise KeyError("config key missing: " + dotted)
    return cur


def require(cfg, dotted, why=""):
    """Value that must not be null. Clear error when a human still has to decide it (batch_size, num_hist, epochs, budget...)."""
    v = cget(cfg, dotted)
    if v is None:
        raise ConfigNull("config.yaml %s is null (TODO, a human decides it after the smoke test)%s" % (dotted, (" - " + why) if why else ""))
    return v


class PreregReader:
    """Every pre-registered value used by a verdict is read through this object, which records
    (name, config path, value) so results.json can list exactly which config values the code used."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.used = {}

    def __call__(self, dotted, name=None):
        v = cget(self.cfg, dotted)
        self.used[name or dotted] = {"config_path": dotted, "value": v}
        return v


def resolve_paths(cfg):
    p = cfg["paths"]
    work = Path(os.environ.get(p["work_env"], str(Path.home() / "dinowm"))).expanduser()
    dataset_dir = Path(os.environ.get(p["dataset_env"], str(work / "data"))).expanduser()
    return {
        "work": work, "dataset_dir": dataset_dir, "data_root": dataset_dir / cfg["data_impl"]["subdir"],
        "acwm_src": Path(os.environ.get(cfg["code"]["src_env"], str(work / "acwm_phys_src"))).expanduser(),
        "train_base": Path(os.environ.get(p["train_env"], str(work / "train_runs"))).expanduser(),
        "cache": Path(os.environ.get(p["cache_env"], str(work / "exp002_cache"))).expanduser(),
        "exp": EXP_DIR, "logs": EXP_DIR / "logs", "manifests": EXP_DIR / "manifests", "results_dir": EXP_DIR / "results",
    }


def run_name(cond, seed):
    return "exp002_%s_s%d" % (cond, int(seed))


def run_dir(paths, name):
    return paths["train_base"] / "outputs" / name


def condition_levels(cfg, cond):
    for c in cfg["conditions"]["list"]:
        if c["name"] == cond:
            return list(c["levels"])
    raise KeyError("unknown condition %s" % cond)


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
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
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


# ------------------------------------------------------------------------------------------ file logging (errors are never swallowed)
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
    """stdout and stderr go to the terminal AND logs/<stage>[_tag]_<time>.log. Returns the log path."""
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
    """Returns consecutive failures recorded for key. Raises if the limit is reached (a human must look at the logs)."""
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


# ------------------------------------------------------------------------------------------ environment info
def env_info():
    info = {"python": sys.version.split()[0], "python_executable": sys.executable, "platform": platform.platform(),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"), "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"), "packages": {}}
    for m in ("torch", "torchvision", "numpy", "scipy", "sklearn", "yaml", "cv2", "decord", "hydra", "omegaconf", "einops"):
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
            info["arch_list"] = torch.cuda.get_arch_list()
    except Exception:
        pass
    return info


def warn_conda():
    if not os.environ.get("CONDA_DEFAULT_ENV"):
        log("WARNING: CONDA_DEFAULT_ENV is empty - not running inside an activated conda environment (python=%s). Continuing." % sys.executable)
        return False
    return True


def disk_free_gb(path):
    import shutil
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return round(shutil.disk_usage(str(p)).free / 1e9, 1)
