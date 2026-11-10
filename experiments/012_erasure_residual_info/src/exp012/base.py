"""Experiment 012 glue: re-uses 009 (erase.py, stages helpers, base), 004 and 002 src UNMODIFIED (imports only; none of their files or configs are written).
Python 3.9 compatible, no torch import at module level."""
from __future__ import annotations

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


def attempts_check(cfg, paths, key):
    return B4.attempts_check(cfg, paths, key)


def load_config(path=None):
    return C2.load_config(Path(path) if path else EXP_DIR / "config.yaml")


def load_cfg009(cfg):
    """009 config, read only."""
    return C2.load_config((EXP_DIR / cfg["base"]["config009"]).resolve())


def make_cfg004(cfg):
    """004 config (read only) in the form 009 builds it (linear_pooled kinds, 009 probe seeds). Returns (c4, cfg009)."""
    c9 = load_cfg009(cfg)
    return B9.make_cfg004(c9), c9


def resolve_paths(cfg, c4, cfg2):
    p = B4.resolve_paths(c4, cfg2)
    own = Path(os.environ.get(cfg["paths"]["cache_env"], str(p["work"] / "exp012_cache"))).expanduser()
    p.update({"exp": EXP_DIR, "logs": EXP_DIR / "logs", "results_dir": EXP_DIR / "results", "cache": own,
              "results009": (EXP_DIR / cfg["base"]["results009"]).resolve()})
    return p


def default_results_dir():
    return EXP_DIR / "results"


def consistency_checks(cfg, c9, cfg2):
    """List of (name, ok, detail). The intervention block must equal the 009 config (same checkpoints, eraser, fit episodes); the attentive probe
    values must equal 002 config probe_impl.attentive; the MLP training settings must equal the attentive training settings (README)."""
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    i12, i9 = cfg["intervention"], c9["intervention"]
    for k in ("m_int", "strengths", "method", "ridge_rel", "fit_fraction", "fit_seed", "apply_to"):
        add("intervention.%s == 009" % k, i12[k] == i9[k], "012=%r 009=%r" % (i12[k], i9[k]))
    add("predictor.train_seeds == 009", cfg["predictor"]["train_seeds"] == c9["predictor"]["train_seeds"])
    add("predictor.batch_episodes == 009", cfg["predictor"]["batch_episodes"] == c9["predictor"]["batch_episodes"])
    add("base.condition == 009", cfg["base"]["condition"] == c9["base"]["condition"])
    add("measure.probe_seeds == 009 probe.seeds", cfg["measure"]["probe_seeds"] == c9["probe"]["seeds"])
    add("feature_part_z == [0, 384] and action part == [384, 404]", list(cfg["predictor"]["feature_part_z"]) == [0, 384] and list(cfg["predictor"]["feature_part_action"]) == [384, 404])
    add("success.strength is one of intervention.strengths and 0 is included", float(cfg["success"]["strength"]) in [float(a) for a in i12["strengths"]] and 0 in [float(a) for a in i12["strengths"]])
    ba, a2 = cfg["measure"]["b_attentive"], cfg2["probe_impl"]["attentive"]
    for k in ("d_model", "n_heads", "mlp_hidden", "epochs", "lr", "weight_decay", "batch_size", "val_fraction"):
        add("b_attentive.%s == 002 probe_impl.attentive" % k, ba[k] == a2[k], "012=%r 002=%r" % (ba[k], a2[k]))
    bm = cfg["measure"]["b_mlp"]
    for k in ("epochs", "lr", "weight_decay", "batch_size"):
        add("b_mlp.%s == 002 attentive training setting" % k, bm[k] == a2[k], "012=%r 002=%r" % (bm[k], a2[k]))
    add("b_mlp.val_fraction == 002 attentive val_fraction (exp002 mlp probe hard-codes 0.1)", bm["val_fraction"] == a2["val_fraction"] == 0.1)
    add("b_mlp: 1 hidden layer relu, early stopping", bm["layers"] == 1 and bm["activation"] == "relu" and bool(bm["early_stopping"]))
    add("b_mlp.init_seeds == b_attentive.init_seeds == probe_seeds", bm["init_seeds"] == ba["init_seeds"] == cfg["measure"]["probe_seeds"])
    sc = cfg["success"]
    add("success numbers (2026-11-10): min_seeds 2, a 0.5333333333, b 0.2, c 0.2",
        sc["min_seeds"] == 2 and sc["a_accuracy_min"] == 0.5333333333 and sc["b_r2_min"] == 0.2 and sc["c_r2_min"] == 0.2)
    return out


def require_consistent(cfg, c9, cfg2):
    res = consistency_checks(cfg, c9, cfg2)
    bad = [(n, d) for n, ok, d in res if not ok]
    if bad:
        raise ValueError("config 012 is inconsistent with 009/002 configs: %s" % "; ".join("%s (%s)" % b for b in bad))
    return res
