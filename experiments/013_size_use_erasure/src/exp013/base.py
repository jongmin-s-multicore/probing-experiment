"""Experiment 013 glue: re-uses 009 (erase.py, stages helpers, curves), 004 (deep.random_init_predictor, stages) and 002 (model_io, cache, common) UNMODIFIED
(imports only; none of their files or configs are written). Python 3.9 compatible, no torch import at module level."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[2]
EXP009_DIR = EXP_DIR.parent / "009_positive_control_depth"
if str(EXP009_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXP009_DIR / "src"))

from exp009 import base as B9  # noqa: E402  (adds 004 and 002 src to sys.path)

B4, C2 = B9.B4, B9.C2
log, write_json, read_json, ConfigNull, set_seed, cget = B9.log, B9.write_json, B9.read_json, B9.ConfigNull, B9.set_seed, B9.cget

CACHE_ENV = "EXP013_CACHE"       # not in config.yaml (config values are confirmed and not edited); CHANGES.md item 1
EXIT_GATE = 5                    # stage-0 gate not passed (or missing / stale): measure and readout refuse, NO attempt consumed
EXIT_PREFLIGHT = 4


def attempts_check(cfg, paths, key):
    return B4.attempts_check(cfg, paths, key)


def load_config(path=None):
    return C2.load_config(Path(path) if path else EXP_DIR / "config.yaml")


def load_cfg009(cfg):
    """009 config, read only."""
    return C2.load_config((EXP_DIR / cfg["base"]["config009"]).resolve())


def make_cfg004(cfg):
    """(c4, c9): 004 config in the form 009 builds it (kinds linear_pooled, 009 probe seeds), read only. The condition field is irrelevant here (conditions are loops)."""
    c9 = load_cfg009(cfg)
    return B9.make_cfg004(c9), c9


def resolve_paths(cfg, c4, cfg2):
    p = B4.resolve_paths(c4, cfg2)
    own = Path(os.environ.get(CACHE_ENV, str(p["work"] / "exp013_cache"))).expanduser()
    p.update({"exp": EXP_DIR, "logs": EXP_DIR / "logs", "results_dir": EXP_DIR / "results", "cache": own})
    return p


def default_results_dir():
    return EXP_DIR / "results"


def run_name(cond, ts):
    return C2.run_name(cond, ts)


def run_dir(paths, cond, ts):
    return C2.run_dir(paths, C2.run_name(cond, ts))


def gate_config_sha(cfg):
    """Fingerprint of the config parts that define the stage-0 gate (a changed gate definition makes an old stage0.json stale; flipping `pilot` does not)."""
    d = {"stage0": cfg["stage0"], "data": cfg["data"], "base": {k: v for k, v in cfg["base"].items() if k != "checkpoint"}}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def consistency_checks(cfg, c9, cfg2):
    """List of (name, ok, detail): the erasure parameters that 013 shares with 009 must equal the 009 config; levels, conditions, frames must match 002."""
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    e13, i9 = cfg["erasure"], c9["intervention"]
    for k13, k9 in (("fit_fraction", "fit_fraction"), ("fit_seed", "fit_seed"), ("ridge_rel", "ridge_rel"), ("apply_to", "apply_to")):
        add("erasure.%s == 009 intervention.%s" % (k13, k9), e13[k13] == i9[k9], "013=%r 009=%r" % (e13[k13], i9[k9]))
    add("erasure.methods == [leace_scalar, level_mean_match] (009 method name leace_scalar)", list(e13["methods"]) == ["leace_scalar", "level_mean_match"] and i9["method"] == "leace_scalar")
    add("predictor.train_seeds == 009", cfg["predictor"]["train_seeds"] == c9["predictor"]["train_seeds"])
    names = [c["name"] for c in cfg2["conditions"]["list"]]
    add("base.conditions exist in 002 config", all(c in names for c in cfg["base"]["conditions"]), str(names))
    add("conditions are (a) single_ind and (b) multi_60_ind_160 in this order", list(cfg["base"]["conditions"]) == ["single_ind", "multi_60_ind_160"])
    add("data.judge_levels == stage0.levels == 002 data.train_levels_b", list(cfg["data"]["judge_levels"]) == list(cfg["stage0"]["levels"]) == list(cfg2["data"]["train_levels_b"]), str(cfg2["data"]["train_levels_b"]))
    add("data.transitions == number of 002 input frames (frames.all, frames 0..11 -> next frame)", int(cfg["data"]["transitions"]) == len(cfg2["frames"]["all"]) == 12)
    add("002 train.num_hist == 1 (predictor sees one frame)", cfg2["train"]["num_hist"] == 1)
    add("002 data.levels has 9 levels, train levels a060/ind/a160", len(cfg2["data"]["levels"]) == 9 and list(cfg2["data"]["train_levels_b"]) == ["a060", "ind", "a160"])
    sc = cfg["success"]
    add("success numbers (2026-11-11): min_seeds 2, ci 0.90, 0.10 of persistence gap, b>a, bootstrap 2000 seed 14000 episode",
        sc["min_seeds"] == 2 and sc["ci"] == 0.90 and sc["min_fraction_of_persistence_gap"] == 0.10 and sc["require_b_greater_than_a"] is True
        and sc["bootstrap"]["n"] == 2000 and sc["bootstrap"]["seed"] == 14000 and sc["bootstrap"]["unit"] == "episode")
    add("stage0 numbers (2026-11-11): ratio >= 1.1 in >= 4 pairs", cfg["stage0"]["ratio_threshold"] == 1.1 and cfg["stage0"]["min_pairs"] == 4)
    add("erasure numbers (2026-11-11): 20 random directions, seed 13000, per_token_displacement", e13["random_directions"] == 20 and e13["random_dir_seed"] == 13000 and e13["random_norm_match"] == "per_token_displacement")
    add("predictor: random_init_seed 1234, checkpoint model_latest", cfg["predictor"]["random_init_seed"] == 1234 and cfg["base"]["checkpoint"] == "model_latest")
    add("metrics: token_mse_visual, primary motion_persistence_normalized", cfg["metrics"]["error_space"] == "token_mse_visual" and cfg["metrics"]["primary"] == "motion_persistence_normalized")
    return out


def require_consistent(cfg, c9, cfg2):
    bad = [(n, d) for n, ok, d in consistency_checks(cfg, c9, cfg2) if not ok]
    if bad:
        raise ValueError("config 013 is inconsistent with 009/002 configs: %s" % "; ".join("%s (%s)" % b for b in bad))


def code_version(exp_dir):
    """git HEAD of the checkout containing exp_dir (None if not a git checkout) + env EXP013_CODE_REPO_URL / EXP013_CODE_REF set by the Colab notebook."""
    import subprocess

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(exp_dir)] + list(a), capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    head = git("rev-parse", "HEAD")
    return {"git_head": head, "git_status_dirty": (bool(git("status", "--porcelain", "--", ".")) if head else None),
            "repo_url": os.environ.get("EXP013_CODE_REPO_URL"), "requested_ref": os.environ.get("EXP013_CODE_REF")}
