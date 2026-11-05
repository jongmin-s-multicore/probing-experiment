"""Resume / checkpoint helpers for the 9 remaining 003 runs (user decision 2026-11-05). Ported from experiments/002 (colab/colab_tools.py:
prepare_run, repair_latest, prune_run; src/exp002/trainrun.py: train_py_is_patched). Standard library only (torch is imported lazily by the default
loadability test). Python 3.9 compatible. None of this touches pre-registered values.

Facts about the pinned train.py (commit 5a640bed), read from the source: init_models() loads <run>/checkpoints/model_latest.pth when it exists and run() loops
range(init_epoch, init_epoch + training.epochs), so an UNPATCHED train.py would train `epochs` MORE epochs after a restart. With colab/train_resume.patch the loop is
range(init_epoch, total_epochs + 1) (training.epochs = TOTAL target epoch), the encoders and optimizer states are restored.
The runs that already have DONE (the three mix_wide seeds) were trained WITHOUT the patch and are never touched here."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

RESUME_MARKER = "[exp002 operational]"        # written into train.py by colab/train_resume.patch (copied unchanged from experiment 002, so the marker keeps the 002 name)
LOOP_EXPR = "range(init_epoch, self.total_epochs + 1)"
ENV_FILE = "train_env.json"                   # per run: GPU, patch state, one record per start attempt (resume events)
EXPECTED_SAVE_EVERY = 1


def train_py_is_patched(acwm_src):
    try:
        t = (Path(acwm_src) / "train.py").read_text()
    except OSError:
        return False
    return RESUME_MARKER in t and LOOP_EXPR in t


def sha256_file(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def epoch_files(ck):
    fs = []
    ck = Path(ck)
    if not ck.is_dir():
        return fs
    for f in ck.glob("model_*.pth"):
        m = re.fullmatch(r"model_(\d+)\.pth", f.name)
        if m:
            fs.append((int(m.group(1)), f))
    return sorted(fs, reverse=True)


def has_checkpoint(run_dir):
    ck = Path(run_dir) / "checkpoints"
    return (ck / "model_latest.pth").exists() or bool(epoch_files(ck))


def default_loadable(p, acwm_src=None):
    import torch
    src = acwm_src or os.environ.get("ACWM_PHYS_SRC")           # checkpoints pickle whole modules: unpickling imports `models` from the ACWM checkout
    if src and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    torch.load(str(p), map_location="cpu", weights_only=False)  # our own checkpoints (trusted)
    return True


def is_loadable(p, loadable):
    try:
        return bool(loadable(p))
    except ImportError:                                         # environment problem, says nothing about the file: must never lead to quarantine (002 bug)
        raise
    except Exception as e:
        print("cannot load %s: %s: %s" % (p, type(e).__name__, e), file=sys.stderr)
        return False


def _replace_latest(ck, src_file):
    tmp = ck / "model_latest.pth.tmp"
    shutil.copyfile(src_file, tmp)
    os.replace(tmp, ck / "model_latest.pth")


def repair_latest(run_dir, loadable):
    ck = Path(run_dir) / "checkpoints"
    latest = ck / "model_latest.pth"
    if not latest.exists():
        return {"state": "no model_latest.pth"}
    if is_loadable(latest, loadable):
        return {"state": "ok"}
    for e, f in epoch_files(ck):
        if is_loadable(f, loadable):
            _replace_latest(ck, f)
            return {"state": "repaired", "restored_from": f.name}
    return {"state": "unrecoverable"}


STALE_ITEMS = ("checkpoints", "epoch_logs.jsonl")


def quarantine_incomplete(run_dir, backup_root, now=None):
    """MOVE (never delete) checkpoints/ and epoch_logs.jsonl of a run without DONE to <backup_root>/<run>__<time>/ so that train.py starts at epoch 0."""
    run_dir = Path(run_dir)
    if "smoke" in run_dir.name or (run_dir / "DONE").exists():
        return None
    items = [run_dir / n for n in STALE_ITEMS if (run_dir / n).exists()]
    if not items:
        return None
    dst = Path(backup_root) / ("%s__%s" % (run_dir.name, time.strftime("%Y%m%d-%H%M%S", now or time.localtime())))
    dst.mkdir(parents=True, exist_ok=False)
    for it in items:
        shutil.move(str(it), str(dst / it.name))
    return dst


def prepare_run(run_dir, backup_root, loadable=None, now=None):
    """What a (re)start of this run does. Returns {"run", "action": fresh|resume|repaired_resume|quarantined|skip, ...}.
    skip: DONE exists or a smoke run. The caller must have checked that train.py is patched BEFORE calling this when a checkpoint exists."""
    run_dir = Path(run_dir)
    name = run_dir.name
    loadable = loadable or default_loadable
    if "smoke" in name or (run_dir / "DONE").exists():
        return {"run": name, "action": "skip"}
    ck = run_dir / "checkpoints"
    latest_exists = (ck / "model_latest.pth").exists()
    if not latest_exists and not epoch_files(ck):
        return {"run": name, "action": "fresh"}
    if not latest_exists:                                       # epoch files only: rebuild latest from the newest loadable one
        for e, f in epoch_files(ck):
            if is_loadable(f, loadable):
                _replace_latest(ck, f)
                return {"run": name, "action": "repaired_resume", "restored_from": f.name}
        return {"run": name, "action": "quarantined", "to": str(quarantine_incomplete(run_dir, backup_root, now))}
    r = repair_latest(run_dir, loadable)
    if r["state"] == "ok":
        return {"run": name, "action": "resume"}
    if r["state"] == "repaired":
        return {"run": name, "action": "repaired_resume", "restored_from": r["restored_from"]}
    return {"run": name, "action": "quarantined", "to": str(quarantine_incomplete(run_dir, backup_root, now))}


# ------------------------------------------------------------------------------------------------ per-run training record
def gpu_info():
    """GPU of the machine that TRAINS (nvidia-smi, no torch import). {"gpu": name or "unknown", "memory_total_mib": int or None}."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        line = out.strip().splitlines()[0]
        name, mem = [x.strip() for x in line.split(",")[:2]]
        return {"gpu": name, "memory_total_mib": int(mem)}
    except Exception:
        return {"gpu": "unknown", "memory_total_mib": None}


def record_attempt(run_dir, info):
    """Append one start attempt to <run>/train_env.json (created if missing). Returns the attempt index."""
    p = Path(run_dir) / ENV_FILE
    d = json.loads(p.read_text()) if p.exists() else {"save_every_x_epoch": info.get("save_every_x_epoch"), "attempts": []}
    d["attempts"].append(info)
    d["gpus_used"] = sorted({a.get("gpu") or "unknown" for a in d["attempts"]})
    d["n_attempts"] = len(d["attempts"])
    d["n_resumes"] = sum(1 for a in d["attempts"] if a.get("action") in ("resume", "repaired_resume"))
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.replace(tmp, p)
    return len(d["attempts"]) - 1


def finish_attempt(run_dir, idx, rc):
    p = Path(run_dir) / ENV_FILE
    d = json.loads(p.read_text())
    d["attempts"][idx].update(end_time=time.strftime("%F %T"), returncode=int(rc))
    d["finished_ok"] = bool(rc == 0)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2))
    os.replace(tmp, p)


def read_train_env(run_dir):
    """For results.json. A run trained before this record existed (the three mix_wide runs) gets "unknown", not a guess."""
    p = Path(run_dir) / ENV_FILE
    if not p.exists():
        return {"recorded": False, "gpu": "unknown", "gpus_used": ["unknown"], "resumed": None, "n_resumes": None, "attempts": None,
                "note": "trained before the per-run record existed (no %s); GPU, resume state and train.py patch state not recorded" % ENV_FILE}
    d = json.loads(p.read_text())
    d["recorded"] = True
    d["gpu"] = d["gpus_used"][0] if len(d.get("gpus_used", [])) == 1 else ("mixed: " + ", ".join(d.get("gpus_used", [])))
    d["resumed"] = d.get("n_resumes", 0) > 0
    return d


def prune_run(run_dir, keep_epochs, require_done=True):
    """Delete model_<e>.pth with e not in keep_epochs (model_latest.pth is never touched). Only for runs that carry a train_env.json with
    save_every_x_epoch == 1, i.e. runs started by this code. Runs without the record (the mix_wide runs, save_every=5) are never pruned."""
    run_dir = Path(run_dir)
    if require_done and not (run_dir / "DONE").exists():
        return []
    envp = run_dir / ENV_FILE
    if not envp.exists():
        return []
    try:
        if int(json.loads(envp.read_text()).get("save_every_x_epoch")) != EXPECTED_SAVE_EVERY:
            return []
    except (ValueError, TypeError):
        return []
    removed = []
    for e, f in epoch_files(run_dir / "checkpoints"):
        if e not in set(keep_epochs):
            f.unlink()
            removed.append(f.name)
    return removed
