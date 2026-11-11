from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from . import analysis as A
from . import data as D
from . import metrics as M

SRC, TGT = ("enc_k12_ours", "enc_k12_random"), ("pred_out6_ours", "pred_out6_random")


def make_logger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        s = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
        print(s, flush=True)
        with open(path, "a") as f:
            f.write(s + "\n")
    return log


def code_version(exp_dir):
    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(exp_dir)] + list(a), capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    head = git("rev-parse", "HEAD")
    return {"git_head": head, "git_status_dirty": (bool(git("status", "--porcelain", "--", ".")) if head else None)}


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable, "numpy": np.__version__, "yaml": yaml.__version__}
    return info


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_main(cfg, cfg_path, paths, inputs_path, out_path, log, expect_pd=(196, 384), smoke=False):
    t0 = time.time()
    if smoke:
        cfg = json.loads(json.dumps(cfg))
        cfg["metrics"]["bootstrap"]["n"], cfg["metrics"]["null_shuffle"]["n"] = 20, 3
    row, fpos, frame, frames = D.load_inputs(inputs_path)
    assert len(row) == cfg["eval"]["n_inputs"], (len(row), cfg["eval"]["n_inputs"])
    log("inputs: %s (sha256 %s) n=%d episodes(rows)=%d frames=%s" % (inputs_path, D.sha256(inputs_path)[:16], len(row), len(np.unique(row)), frames))
    al = D.alignments(row, fpos, frame, frames)
    sel, nxt = al["predicted_frame"]
    log("predicted_frame alignment: %d / %d inputs have frame t+1 cached (frames with t+1 missing: %s)" % (
        len(sel), len(row), sorted(set(frame[np.setdiff1d(np.arange(len(row)), sel)].tolist()))))
    shapes, G, Dim = {}, {}, None
    for name, p in paths.items():
        a = np.load(p, mmap_mode="r")
        shapes[name] = list(a.shape)
        assert a.ndim == 4 and a.shape[1] == len(frames) and tuple(a.shape[2:]) == tuple(expect_pd), (name, a.shape, len(frames), expect_pd)
        assert row.max() < a.shape[0], (name, a.shape, row.max())
    assert len({s[0] for s in shapes.values()}) == 1, "episode axis differs between caches: %s" % shapes
    Dim = int(np.prod(expect_pd))
    for name, p in paths.items():
        a = np.load(p, mmap_mode="r")
        if name.startswith("enc"):
            G[(name, "input")] = M.gram(D.flat(np.asarray(a[row, fpos])))
            if len(sel):
                G[(name, "next")] = M.gram(D.flat(np.asarray(a[row[sel], nxt])))
        else:
            G[(name, "input")] = M.gram(D.flat(np.asarray(a[row, fpos])))
        del a
        log("gram done: %s (%.0fs)" % (name, time.time() - t0))
    pairs = {}
    for aname in ("input_frame", "predicted_frame", "input_frame_matched_subset"):
        for s in SRC:
            for t in TGT:
                key = "%s|%s|%s" % (aname, s, t)
                if aname == "input_frame":
                    Kx, Ky, g, nl = G[(s, "input")], G[(t, "input")], row, True
                elif aname == "predicted_frame":
                    if not len(sel):
                        pairs[key] = {"not_evaluable": "no input has frame t+1 in the encoder cache"}
                        continue
                    Kx, Ky, g, nl = G[(s, "next")], G[(t, "input")][np.ix_(sel, sel)], row[sel], True
                else:
                    if not len(sel):
                        continue
                    Kx, Ky, g, nl = G[(s, "input")][np.ix_(sel, sel)], G[(t, "input")][np.ix_(sel, sel)], row[sel], False
                pairs[key] = A.analyze_pair(Kx, Ky, g, Dim, Dim, cfg, null=nl)
                r = pairs[key]
                log("%s: n=%d CKA=%.4f R2=%.4f R2rev=%.4f (%.0fs)" % (key, r["n_inputs"], r["cka"]["point"], r["r2"]["point"], r["r2_reverse_descriptive"]["point"], time.time() - t0))
    verdict = {a: A.verdict_pass(pairs.get("%s|enc_k12_ours|pred_out6_random" % a), cfg) for a in ("input_frame", "predicted_frame")}
    sec = {}
    for a in ("input_frame", "predicted_frame"):
        for s in SRC:
            o, r = pairs.get("%s|%s|pred_out6_ours" % (a, s)), pairs.get("%s|%s|pred_out6_random" % (a, s))
            if o and r and "not_evaluable" not in o and "not_evaluable" not in r:
                sec["%s|%s" % (a, s)] = {"cka_ours_minus_random": o["cka"]["point"] - r["cka"]["point"], "r2_ours_minus_random": o["r2"]["point"] - r["r2"]["point"]}
    res = {"experiment": cfg["name"], "hypothesis": cfg["hypothesis"], "smoke": bool(smoke), "pilot": cfg["pilot"], "seed": cfg["seed"],
           "config_sha256": D.sha256(cfg_path), "code_version": code_version(D.EXP_DIR), "environment": env_info(),
           "inputs_file": {"path": str(inputs_path), "sha256": D.sha256(inputs_path), "n": int(len(row)), "n_episodes_rows": int(len(np.unique(row)))},
           "cache_files": {k: str(v) for k, v in paths.items()}, "cache_shapes": shapes, "frames_cached": frames,
           "feature_dim_per_input": Dim, "pooling": "none: full (P=196, d=384) token set flattened, as the 005 attentive heads received it (feat[row, fpos])",
           "alignment_counts": {"input_frame": int(len(row)), "predicted_frame": int(len(sel)), "input_frame_matched_subset": int(len(sel))},
           "settings": {"ridge": cfg["metrics"]["ridge_r2"], "inner_folds": M.INNER_FOLDS, "bootstrap": cfg["metrics"]["bootstrap"], "null_shuffle": cfg["metrics"]["null_shuffle"],
                        "feature_scaling": "single global scalar per source so centered features have mean square 1 (alpha grid meaning); R2 = 1 - sum SSE / sum SST pooled over folds, SST around train-fold mean"},
           "pairs": pairs, "verdict_H_pass": verdict, "secondary_ours_minus_random": sec, "interpretation": None, "elapsed_s": time.time() - t0}
    Path(out_path).write_text(json.dumps(_clean(res), indent=2, ensure_ascii=False))
    log("wrote %s (%.0fs)" % (out_path, time.time() - t0))
    return res
