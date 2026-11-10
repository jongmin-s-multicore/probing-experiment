"""Experiment 010 glue: puts 005's src (and through it 002's and 004's) on sys.path and re-uses exp005 (heads, tasks, ibmetrics, verdicts, stages, feats) UNMODIFIED."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[2]
EXP005_DIR = EXP_DIR.parent / "005_inductive_bias_probe"
if str(EXP005_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXP005_DIR / "src"))

from exp005 import base as B5  # noqa: E402  (adds 002 and 004 src to sys.path)

C2 = B5.C2
log, write_json, read_json, ConfigNull, set_seed = B5.log, B5.write_json, B5.read_json, B5.ConfigNull, B5.set_seed


def attempts_check(cfg, paths, key):
    return B5.attempts_check(cfg, paths, key)


def load_config(path=None):
    return C2.load_config(Path(path) if path else EXP_DIR / "config.yaml")


def make_cfg005(cfg):
    """005 config (read only) with 010's overrides: eval.n_inputs, head seeds. Task seeds are set per task seed by `cfg_for_task_seed`."""
    c5 = copy.deepcopy(B5.load_config((EXP_DIR / cfg["base"]["config005"]).resolve()))
    c5["eval"]["n_inputs"] = int(cfg["eval"]["n_inputs"])
    c5["seeds"]["head"] = [int(s) for s in cfg["seeds"]["head"]]
    c5["seeds"]["eval"] = int(cfg["seeds"]["eval"])
    c5["base"]["condition"] = cfg["base"]["condition"]
    return c5


def cfg_for_task_seed(c5, task_seed):
    c = copy.deepcopy(c5)
    c["seeds"]["tasks_a"] = int(task_seed)
    return c


def resolve_paths(cfg, c5, cfg2):
    p = B5.resolve_paths(c5, cfg2)
    p.update({"exp": EXP_DIR, "logs": EXP_DIR / "logs", "results_dir": EXP_DIR / "results"})
    return p


def load_cfg002(c5):
    return B5.load_cfg002(c5)


def layer_dict(rep):
    return {"name": rep["name"], "module": rep["module"], "depth": int(rep["depth"]), "kind": rep["kind"]}
