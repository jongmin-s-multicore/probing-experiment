"""Experiment 005 glue. Puts 002's src (exp002, imported unmodified) and 004's src (exp004: deep.py early-exit/encoder-truncation helpers, depthlib)
on sys.path. Python 3.9 compatible, no torch at module level."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[2]
EXP002_DIR = EXP_DIR.parent / "002_single_size_sweep"
EXP004_DIR = EXP_DIR.parent / "004_depth_truncation"
for p in (EXP002_DIR / "src", EXP004_DIR / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp002 import common as C2  # noqa: E402

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
    return C2.load_config(Path(path) if path else EXP_DIR / "config.yaml")


def load_cfg002(cfg):
    """002 config for the re-used exp002 functions; probe.seeds/kinds are replaced by this experiment's explicit values (head seeds)."""
    p = (EXP_DIR / cfg["base"]["config002"]).resolve()
    c = copy.deepcopy(C2.load_config(p))
    c["probe"]["seeds"] = [int(s) for s in cfg["seeds"]["head"]]
    c["_config002_sha256"] = C2.config_sha256(p)
    return c


def resolve_paths(cfg, cfg2):
    p2 = C2.resolve_paths(cfg2)
    work = p2["work"]
    own = Path(os.environ.get(cfg["paths"]["cache_env"], str(work / "exp005_cache"))).expanduser()
    c004 = Path(os.environ.get(cfg["paths"]["cache004_env"], str(work / "exp004_cache"))).expanduser()
    return {"work": work, "acwm_src": p2["acwm_src"], "train_base": p2["train_base"], "cache002": p2["cache"], "manifests002": p2["manifests"],
            "cache004": c004, "cache": own, "exp": EXP_DIR, "logs": EXP_DIR / "logs", "results_dir": EXP_DIR / "results",
            "results004": (EXP004_DIR / "results")}


def as002(paths):
    return dict(paths, cache=paths["cache002"], manifests=paths["manifests002"])


def run_dir002(paths, cfg, seed):
    return C2.run_dir(paths, C2.run_name(cfg["base"]["condition"], seed))
