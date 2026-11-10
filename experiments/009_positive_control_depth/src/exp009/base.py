"""Experiment 009 glue: re-uses 004's src (exp004: deep.py, depthlib.py, stages.probe_hidden) and 002's src UNMODIFIED. Python 3.9 compatible; torch only inside functions."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[2]
EXP004_DIR = EXP_DIR.parent / "004_depth_truncation"
if str(EXP004_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXP004_DIR / "src"))

from exp004 import base as B4  # noqa: E402  (adds 002 src to sys.path)

C2 = B4.C2
log, write_json, read_json, ConfigNull, set_seed, cget = B4.log, B4.write_json, B4.read_json, B4.ConfigNull, B4.set_seed, B4.cget


def attempts_check(cfg, paths, key):
    return B4.attempts_check(cfg, paths, key)


def load_config(path=None):
    return C2.load_config(Path(path) if path else EXP_DIR / "config.yaml")


def make_cfg004(cfg):
    """004 config (read only) with this experiment's probe kinds/seeds."""
    c4 = copy.deepcopy(B4.load_config((EXP_DIR / cfg["base"]["config004"]).resolve()))
    c4["probe"]["kinds"] = list(cfg["probe"]["kinds"])
    c4["probe"]["seeds"] = [int(s) for s in cfg["probe"]["seeds"]]
    c4["base"]["condition"] = cfg["base"]["condition"]
    c4["predictor"]["train_seeds"] = [int(s) for s in cfg["predictor"]["train_seeds"]]
    return c4


def check_consistency_with_004(cfg, c4):
    """The decision rule must be the one 004 used (hypotheses.md 009: re-use). Raises when drop_threshold/primary differ from 004's config."""
    d4 = c4["depth"]
    d = cfg["depth"]
    if d4["drop_threshold"] != d["drop_threshold"] or dict(d4["primary"]) != dict(d["primary"]):
        raise ValueError("009 depth.drop_threshold/primary differ from 004 config: %s vs %s" % (d, d4))


def resolve_paths(cfg, c4, cfg2):
    p = B4.resolve_paths(c4, cfg2)
    own = Path(os.environ.get(cfg["paths"]["cache_env"], str(p["work"] / "exp009_cache"))).expanduser()
    p.update({"exp": EXP_DIR, "logs": EXP_DIR / "logs", "results_dir": EXP_DIR / "results", "cache": own, "results004": EXP004_DIR / "results"})
    return p
