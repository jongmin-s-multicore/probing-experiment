"""Experiment 004 glue: puts experiments/002_single_size_sweep/src on sys.path and re-uses exp002 (probes, evalgrid, stats, masks, model_io,
cache, data, common) UNMODIFIED. This file only adds config/path handling for 004 (002's resolve_paths/EXP_DIR point at 002's own folders).
Python 3.9 compatible, no torch import at module level."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[2]
EXP002_DIR = EXP_DIR.parent / "002_single_size_sweep"
if str(EXP002_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXP002_DIR / "src"))

from exp002 import common as C2  # noqa: E402  (re-exported helpers)

log = C2.log
write_json = C2.write_json
read_json = C2.read_json
ConfigNull = C2.ConfigNull
cget = C2.cget
set_seed = C2.set_seed


def attempts_check(cfg, paths, key):
    """Retry guard (CLAUDE.md: max 3). exp002.common.attempts_check was removed in 002 commit 545168e, so it is kept here with the same behaviour:
    returns the consecutive failures recorded for key in <results_dir>/attempts.json, raises when cfg run.max_attempts is reached.
    Recording stays exp002.common.attempts_record."""
    p = paths["results_dir"] / "attempts.json"
    d = C2.read_json(p) if p.exists() else {}
    n = int(d.get(key, 0))
    mx = int(cfg["run"]["max_attempts"])
    if n >= mx:
        raise RuntimeError("%s failed %d times in a row (limit %d, CLAUDE.md). Stop and report the cause from logs/. "
                           "A human may delete the entry in %s to allow another try." % (key, n, mx, p))
    return n


def load_config(path=None):
    """Own config (strict loader: duplicate YAML keys are an error)."""
    return C2.load_config(Path(path) if path else EXP_DIR / "config.yaml")


def load_cfg002(cfg):
    """002 config used by the re-used exp002 functions (probes need probe_impl/frames/q1/data/...). Only probe.kinds / probe.seeds are overridden
    from this experiment's config (explicit seeds live here). The 002 file is read, never written."""
    p = (EXP_DIR / cfg["base"]["config002"]).resolve()
    c = C2.load_config(p)
    c = copy.deepcopy(c)
    c["probe"]["kinds"] = list(cfg["probe"]["kinds"])
    c["probe"]["seeds"] = [int(s) for s in cfg["probe"]["seeds"]]
    for k, v in (cfg["base"].get("override_probe_impl") or {}).items():
        c["probe_impl"][k] = v
    c["_config002_sha256"] = C2.config_sha256(p)
    return c


def resolve_paths(cfg, cfg002):
    p2 = C2.resolve_paths(cfg002)                      # 002 locations: cache (EXP002_CACHE), train_base (DINO_TRAIN), acwm_src, manifests of 002
    work = p2["work"]
    own_cache = Path(os.environ.get(cfg["paths"]["cache_env"], str(work / ("exp%s_cache" % cfg["name"][:3])))).expanduser()
    return {"work": work, "acwm_src": p2["acwm_src"], "train_base": p2["train_base"], "cache002": p2["cache"], "cache": own_cache,
            "manifests002": p2["manifests"], "exp": EXP_DIR, "logs": EXP_DIR / "logs", "results_dir": EXP_DIR / "results"}


def as002(paths):
    """paths dict in the form exp002 functions expect (cache -> 002 cache root, manifests -> 002 manifests)."""
    return dict(paths, cache=paths["cache002"], manifests=paths["manifests002"])


def run_dir002(paths, cfg, seed):
    return C2.run_dir(paths, C2.run_name(cfg["base"]["condition"], seed))
