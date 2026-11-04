#!/usr/bin/env python
"""Operational helpers for running experiment 002 on Google Colab with a persistent Drive work root (확정 기록 10). Standard library only (torch is imported lazily, only by the default loadability test).
None of this touches pre-registered values.

  python colab/colab_tools.py setup-dirs --work-root /content/drive/MyDrive/exp002_work
      creates <work>/{train_runs,exp002_cache,results,manifests,logs} and makes experiments/002_single_size_sweep/{results,manifests,logs}
      symlinks to them (existing local content is moved, never overwritten). attempts.json lives in results/, so the retry guard persists.
  python colab/colab_tools.py prepare --train-root <work>/train_runs [--run NAME]
      RESUME from the last checkpoint (user decision 2026-11-04, corrected; needs the patched train.py, colab/train_resume.patch). For every run
      folder without DONE: no checkpoint -> nothing to do (fresh start); model_latest.pth loadable -> keep (resume); model_latest.pth broken ->
      replace it by the newest loadable model_<epoch>.pth (resume); no loadable checkpoint at all -> MOVE checkpoints/ and epoch_logs.jsonl to
      <train_root>/quarantine/<run>__<time>/ (never deleted) and start from epoch 0. Called before every train attempt by overnight.sh and by the
      notebook's last cell (only when overnight.sh is not running). Smoke runs and DONE runs are untouched.
  python colab/colab_tools.py prune --train-root <work>/train_runs
      for finished runs (DONE exists) delete model_<epoch>.pth except epochs in config colab.keep_ckpt_epochs; model_latest.pth is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
LINKED = ("results", "manifests", "logs")
WORK_SUBDIRS = ("train_runs", "exp002_cache") + LINKED


def _move_contents(src, dst):
    for item in src.iterdir():
        t = dst / item.name
        if t.exists():
            print("  keep both: %s exists in the work root, local copy left as %s.local" % (t, item))
            item.rename(item.with_name(item.name + ".local"))
            shutil.move(str(item.with_name(item.name + ".local")), str(dst / (item.name + ".local")))
        else:
            shutil.move(str(item), str(t))


def setup_dirs(work_root, exp_dir=EXP_DIR):
    work_root, exp_dir = Path(work_root), Path(exp_dir)
    for d in WORK_SUBDIRS:
        (work_root / d).mkdir(parents=True, exist_ok=True)
    out = {}
    for name in LINKED:
        link, target = exp_dir / name, work_root / name
        if link.is_symlink():
            if Path(os.readlink(link)) != target:
                link.unlink()
            else:
                out[name] = "already linked"
                continue
        elif link.exists():
            _move_contents(link, target)
            link.rmdir()
        link.symlink_to(target, target_is_directory=True)
        out[name] = "linked -> %s" % target
    return out


def _epoch_files(ck):
    fs = []
    for f in ck.glob("model_*.pth"):
        m = re.fullmatch(r"model_(\d+)\.pth", f.name)
        if m:
            fs.append((int(m.group(1)), f))
    return sorted(fs, reverse=True)


def _default_loadable(p):
    import torch
    torch.load(str(p), map_location="cpu", weights_only=False)      # our own checkpoints (trusted)
    return True


def repair_latest(run_dir, loadable=None):
    """Make checkpoints/model_latest.pth loadable (a session may die while it is written). Returns a dict describing what was done."""
    loadable = loadable or _default_loadable
    ck = Path(run_dir) / "checkpoints"
    latest = ck / "model_latest.pth"
    if not latest.exists():
        return {"run": Path(run_dir).name, "state": "no model_latest.pth"}

    def ok(p):
        try:
            return bool(loadable(p))
        except Exception:
            return False
    if ok(latest):
        return {"run": Path(run_dir).name, "state": "ok"}
    for e, f in _epoch_files(ck):
        if ok(f):
            tmp = ck / "model_latest.pth.tmp"
            shutil.copyfile(f, tmp)
            os.replace(tmp, latest)
            return {"run": Path(run_dir).name, "state": "repaired", "restored_from": f.name}
    return {"run": Path(run_dir).name, "state": "unrecoverable"}


# Items of an unfinished run folder that make the pinned train.py continue from an old state or mix old and new logs.
STALE_ITEMS = ("checkpoints", "epoch_logs.jsonl")


def quarantine_incomplete(run_dir, backup_root, now=None):
    """Move the checkpoints of a run WITHOUT DONE out of the way so that train.py starts from epoch 0. Only used when no loadable checkpoint exists
    (see prepare_run). Nothing is deleted: items go to <backup_root>/<run>__<YYYYmmdd-HHMMSS>/.
    Smoke runs (name contains 'smoke') and runs with DONE are never touched. Returns the backup folder or None."""
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
    """Decide what a restart of this run does. Returns {"run", "action": fresh|resume|repaired_resume|quarantined|skip, ...}.
    Resume is valid only with the patched train.py (trainrun.py refuses an existing checkpoint when the patch marker is missing)."""
    run_dir = Path(run_dir)
    name = run_dir.name
    if "smoke" in name or (run_dir / "DONE").exists():
        return {"run": name, "action": "skip"}
    ck = run_dir / "checkpoints"
    if not (ck / "model_latest.pth").exists() and not _epoch_files(ck):
        return {"run": name, "action": "fresh"}
    if not (ck / "model_latest.pth").exists():                      # epoch files only: rebuild latest from the newest loadable one
        for e, f in _epoch_files(ck):
            try:
                ok = bool((loadable or _default_loadable)(f))
            except Exception:
                ok = False
            if ok:
                shutil.copyfile(f, ck / "model_latest.pth.tmp")
                os.replace(ck / "model_latest.pth.tmp", ck / "model_latest.pth")
                return {"run": name, "action": "repaired_resume", "restored_from": f.name}
        return {"run": name, "action": "quarantined", "to": str(quarantine_incomplete(run_dir, backup_root, now))}
    r = repair_latest(run_dir, loadable)
    if r["state"] == "ok":
        return {"run": name, "action": "resume"}
    if r["state"] == "repaired":
        return {"run": name, "action": "repaired_resume", "restored_from": r["restored_from"]}
    return {"run": name, "action": "quarantined", "to": str(quarantine_incomplete(run_dir, backup_root, now))}


def prune_run(run_dir, keep_epochs, require_done=True):
    run_dir = Path(run_dir)
    if require_done and not (run_dir / "DONE").exists():
        return []
    removed = []
    for e, f in _epoch_files(run_dir / "checkpoints"):
        if e not in set(keep_epochs):
            f.unlink()
            removed.append(f.name)
    return removed


def _runs(train_root):
    out = Path(train_root) / "outputs"
    return sorted(p for p in out.iterdir() if p.is_dir()) if out.is_dir() else []


def _keep_from_config():
    try:
        import yaml
        cfg = yaml.safe_load((EXP_DIR / "config.yaml").read_text())
        return list(cfg["colab"]["keep_ckpt_epochs"])
    except Exception as e:
        print("cannot read colab.keep_ckpt_epochs from config.yaml (%s); keeping every epoch file" % e, file=sys.stderr)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("setup-dirs"); a.add_argument("--work-root", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--train-root", required=True); q.add_argument("--run", default=None, help="only this run folder name (default: every run without DONE)")
    p = sub.add_parser("prune"); p.add_argument("--train-root", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "setup-dirs":
        print(json.dumps(setup_dirs(args.work_root), indent=2))
    elif args.cmd == "prepare":
        for rd in _runs(args.train_root):
            if args.run and rd.name != args.run:
                continue
            r = prepare_run(rd, Path(args.train_root) / "quarantine")
            if r["action"] != "skip":
                print(json.dumps(r))
    else:
        keep = _keep_from_config()
        if keep is None:
            return 0
        for rd in _runs(args.train_root):
            rem = prune_run(rd, keep)
            if rem:
                print("%s: removed %s" % (rd.name, rem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
