"""Experiment 015 glue: re-uses the 013 base (which re-uses 009/004/002) UNMODIFIED (imports only; none of their files or configs are written).
Python 3.9 compatible, no torch import at module level."""
from __future__ import annotations

import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[2]
EXP013_DIR = EXP_DIR.parent / "013_size_use_erasure"
if str(EXP013_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXP013_DIR / "src"))

from exp013 import base as B13  # noqa: E402  (adds 009, 004 and 002 src to sys.path)

B9, B4, C2 = B13.B9, B13.B4, B13.C2
log, write_json, read_json, ConfigNull, set_seed = B13.log, B13.write_json, B13.read_json, B13.ConfigNull, B13.set_seed

CACHE_ENV = "EXP015_CACHE"       # not in config.yaml (config values are confirmed and not edited); CHANGES.md item 1
EXIT_PREFLIGHT = 4
TRAIN_LEVELS = ["a060", "ind", "a160"]


def attempts_check(cfg, paths, key):
    return B4.attempts_check(cfg, paths, key)


def load_config(path=None):
    return C2.load_config(Path(path) if path else EXP_DIR / "config.yaml")


def load_cfg013(cfg):
    """013 config, read only (level-mean fit rule, 009 reference)."""
    return C2.load_config((EXP_DIR / cfg["base"]["config013"]).resolve())


def make_cfg004(cfg):
    """(c4, c9, cfg13): 004 config in the form 009 builds it, read only."""
    cfg13 = load_cfg013(cfg)
    c9 = B13.load_cfg009(cfg13)
    return B9.make_cfg004(c9), c9, cfg13


def resolve_paths(cfg, c4, cfg2):
    p = B4.resolve_paths(c4, cfg2)
    own = Path(os.environ.get(CACHE_ENV, str(p["work"] / "exp015_cache"))).expanduser()
    p.update({"exp": EXP_DIR, "logs": EXP_DIR / "logs", "results_dir": EXP_DIR / "results", "cache": own})
    return p


def default_results_dir():
    return EXP_DIR / "results"


def run_dir(paths, cond, ts):
    return B13.run_dir(paths, cond, ts)


def pair_name(s, t):
    return "%s>%s" % (s, t)


def consistency_checks(cfg, cfg13, cfg2):
    """List of (name, ok, detail): the confirmed 2026-11-12 numbers, and agreement with the 013 / 002 configs."""
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    names = [c["name"] for c in cfg2["conditions"]["list"]]
    add("base.conditions exist in 002 config", all(c in names for c in cfg["base"]["conditions"]), str(names))
    add("conditions are (a) single_ind and (b) multi_60_ind_160 in this order", list(cfg["base"]["conditions"]) == ["single_ind", "multi_60_ind_160"])
    add("conditions / checkpoint / train seeds / random-init seed == 013", list(cfg["base"]["conditions"]) == list(cfg13["base"]["conditions"]) and cfg["base"]["checkpoint"] == cfg13["base"]["checkpoint"] == "model_latest"
        and cfg["predictor"]["train_seeds"] == cfg13["predictor"]["train_seeds"] == [0, 1, 2] and cfg["predictor"]["random_init_seed"] == cfg13["predictor"]["random_init_seed"] == 1234)
    add("002 data.train_levels_b == [a060, ind, a160]", list(cfg2["data"]["train_levels_b"]) == TRAIN_LEVELS, str(cfg2["data"]["train_levels_b"]))
    add("data.transitions == number of 002 input frames (frames.all, frames 0..11 -> next frame) == 12", int(cfg["data"]["transitions"]) == len(cfg2["frames"]["all"]) == 12)
    add("002 train.num_hist == 1 (predictor sees one frame)", cfg2["train"]["num_hist"] == 1)
    add("data.eval_split == test; level-mean / target-motion fit on train_split", cfg["data"]["eval_split"] == "test" and cfg["interchange"]["level_mean_fit"] == "train_split" and cfg["interchange"]["target_motion_fit"] == "train_split")
    prs = [list(p) for p in cfg["interchange"]["pairs"]]
    want = [[a, b] for a in TRAIN_LEVELS for b in TRAIN_LEVELS if a != b]
    add("interchange.pairs: all 6 directions among a060/ind/a160", sorted(prs) == sorted(want) and len(prs) == 6, str(prs))
    add("interchange.apply_to == z_part_all_tokens (013 insertion point)", cfg["interchange"]["apply_to"] == "z_part_all_tokens" == cfg13["erasure"]["apply_to"])
    c = cfg["controls"]
    add("controls (2026-11-12): fake swap 20 repeats seed 15000 norm-matched per patch to the real shift (primary), unscaled report only, random directions 20 seed 13000, random_init",
        c["fake_swap"]["n_repeats"] == 20 and c["fake_swap"]["seed"] == 15000 and c["fake_swap"].get("norm_match") == "per_patch_to_real_shift" and c["fake_swap_unscaled"]["report_only"] is True
        and c["primary_comparator"] == "fake_swap_norm_matched" and c["random_directions"] == 20 and c["random_dir_seed"] == 13000 and c["random_init"] is True)
    g = cfg["gate"]["d_reliability"]
    add("gate (2026-11-12): mean over 20 random splits of the split-half cosine >= 0.5", g["metric"] == "split_half_cosine" and g["min"] == 0.5 and g.get("aggregation") == "mean_over_splits" and g.get("n_splits") == 20)
    sc = cfg["success"]
    add("success (2026-11-12): min_seeds 2, ci 0.90, min_moved 0.10, b>a, reverse sign, mean over gated pairs, bootstrap 2000 seed 14000 episode",
        sc["min_seeds"] == 2 and sc["ci"] == 0.90 and sc["min_moved"] == 0.10 and sc["require_b_greater_than_a"] is True and sc["require_reverse_sign"] is True
        and sc["pair_aggregation"] == "mean_over_gated_pairs" and sc["bootstrap"]["n"] == 2000 and sc["bootstrap"]["seed"] == 14000 and sc["bootstrap"]["unit"] == "episode")
    add("seeds explicit: global seed present, pilot flag present", "seed" in cfg and "pilot" in cfg)
    return out


def code_version(exp_dir):
    """git HEAD of the checkout containing exp_dir + env EXP015_CODE_REPO_URL / EXP015_CODE_REF set by the Colab notebook."""
    import subprocess

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(exp_dir)] + list(a), capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    head = git("rev-parse", "HEAD")
    return {"git_head": head, "git_status_dirty": (bool(git("status", "--porcelain", "--", ".")) if head else None),
            "repo_url": os.environ.get("EXP015_CODE_REPO_URL"), "requested_ref": os.environ.get("EXP015_CODE_REF")}
