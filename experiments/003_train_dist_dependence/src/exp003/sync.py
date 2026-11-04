"""Upload experiment outputs to Google Drive (rclone, or a copy into a locally mounted Drive folder). Never fails the experiment when
rclone / the remote is not configured: it prints how to set it up, writes results/sync_status.json (status "skipped") and returns 0.

Default upload (relative to the experiment dir): results.json, results/, manifests/, logs/, config.yaml, results/run_meta.json.
Opt-in (large): checkpoints of the models ($DINO_TRAIN/outputs/<run>/, ~1.7 GB per run, estimate)  -> SYNC_CKPT=1 / --ckpt
                extraction cache ($EXP003_CACHE/<run>/, ~5 GB per run, estimate)                      -> SYNC_CACHE=1 / --cache
A pattern scan (tokens, private keys, e-mail addresses) runs first; suspicious files are skipped with a warning."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import EXP_DIR, write_json, read_json, log

SETUP_HELP = """\
Drive sync SKIPPED ({reason}). The experiment itself is not affected.
Option A (rclone, default):
  1. install rclone (https://rclone.org/install/)
  2. rclone config            -> n (new remote) -> name e.g. gdrive -> storage "drive" -> follow the prompts
     headless machine (no browser): answer "n" to "Use web browser to automatically authenticate?", then run on a PC WITH a browser:
         rclone authorize "drive"
     and paste the printed token blob back into the headless machine's `rclone config` prompt.
  3. fill config.yaml  sync.drive.rclone_remote (e.g. gdrive) and sync.drive.remote_path (e.g. exp003_results)
     (or env EXP003_RCLONE_REMOTE / EXP003_REMOTE_PATH)
  4. rclone lsd gdrive:        # check that the remote works
Option B (Drive desktop app mounted as a folder): set config.yaml sync.drive.local_mount (or env EXP003_DRIVE_MOUNT) to the mounted path,
  e.g. /mnt/g/MyDrive ; files are copied below <local_mount>/<remote_path>/.
Then:  python run.py sync --dry-run   (lists files and sizes)   and   python run.py sync   (or SYNC=1 bash run_all.sh)"""

# --- secret scan -------------------------------------------------------------------------------------------------
PATTERNS = {
    "hf_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    "api_key": re.compile(rb"sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|xox[abp]-[A-Za-z0-9\-]{10,}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "email": re.compile(rb"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+\.[A-Za-z]{2,}"),
    "secret_assignment": re.compile(rb"(?i)\b(?:hf_token|api[_-]?key|secret|password|passwd|access[_-]?token|refresh_token)\b\s*[:=]\s*[\"']?[A-Za-z0-9_\-./+=]{12,}"),
    "rclone_token": re.compile(rb"\"access_token\"\s*:|\"refresh_token\"\s*:"),
}
SCAN_MAX_BYTES = 8 * 1024 * 1024     # larger (binary: npz, pth) files are not scanned; they are only uploaded when explicitly opted in
EXCLUDE_PARTS = {"__pycache__"}
EXCLUDE_SUFFIX = (".pyc", ".tmp")


def scan_file(path, extra_literals=()):
    """Return the list of pattern names that match (empty = clean, None = not scanned because too large)."""
    try:
        if path.stat().st_size > SCAN_MAX_BYTES:
            return None
        b = path.read_bytes()
    except OSError:
        return ["unreadable"]
    hits = [n for n, rx in PATTERNS.items() if rx.search(b)]
    for lit in extra_literals:
        if lit and lit.encode() in b:
            hits.append("literal_secret_from_env")
    return hits


def _env_secret_values():
    out = []
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "GITHUB_TOKEN"):
        v = os.environ.get(k, "").strip()
        if len(v) >= 8:
            out.append(v)
    return out


def collect(root, rel_items, scan=True):
    """Files below `root` listed by rel_items (files or dirs). Returns (files [(rel, size)], skipped [(rel, hits)], unscanned [rel])."""
    files, skipped, unscanned = [], [], []
    lits = _env_secret_values()
    for it in rel_items:
        p = root / it
        if not p.exists():
            continue
        cands = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
        for q in cands:
            rel = q.relative_to(root).as_posix()
            if EXCLUDE_PARTS & set(q.parts) or rel.endswith(EXCLUDE_SUFFIX) or q.name == "sync_status.json" or q.name.endswith("_sync.log"):  # status/log of the sync itself are still being written
                continue
            if scan:
                hits = scan_file(q, lits)
                if hits is None:
                    unscanned.append(rel)
                elif hits:
                    skipped.append((rel, hits))
                    continue
            files.append((rel, q.stat().st_size))
    return files, skipped, unscanned


def _fmt(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


# --- run meta ----------------------------------------------------------------------------------------------------
def _git(path, *a):
    try:
        return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def build_run_meta(cfg, paths, run_label, models):
    meta = {"experiment": cfg["name"], "run_label": run_label, "models": models, "dataset": cfg["data"]["active"],
            "seeds": {"global": cfg["seed"], "train": cfg["train"].get("seed")},
            "exp_repo_commit": _git(EXP_DIR, "rev-parse", "HEAD"),
            "exp_repo_dirty": bool(_git(EXP_DIR, "status", "--porcelain", "--", ".")),
            "acwm_src_commit": _git(paths["acwm_src"], "rev-parse", "HEAD") if paths["acwm_src"].exists() else None,
            "config_acwm_commit": cfg["code"]["commit"],
            "python": sys.version.split()[0], "platform": platform.platform(), "torch": None, "cuda": None, "gpu": None}
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    if meta["gpu"] is None:
        try:
            meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                                         timeout=20).stdout.strip().splitlines()[0] or None
        except Exception:
            pass
    times = paths["results_dir"] / "run_times.json"          # written by run_all.sh
    t = read_json(times) if times.exists() else {}
    meta["start_time"] = t.get("first_start")
    meta["last_start_time"] = t.get("last_start")
    meta["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")  # time of this sync call
    return meta


# --- upload ------------------------------------------------------------------------------------------------------
def _rclone_ok(remote):
    if not shutil.which("rclone"):
        return "rclone is not installed"
    r = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    if f"{remote}:" not in r.stdout.split():
        return f"rclone remote '{remote}' does not exist (rclone listremotes: {r.stdout.split()})"
    return None


def _upload_group(mode, root, files, dest, remote, tmpdir, tag):
    """Copy `files` (relative to root) to dest and verify. Returns a status dict."""
    st = {"tag": tag, "dest": dest, "n_files": len(files), "bytes": sum(s for _, s in files), "verified": False}
    if not files:
        st["verified"] = True
        return st
    if mode == "rclone":
        lst = tmpdir / f"files_{tag}.txt"
        lst.write_text("\n".join(r for r, _ in files) + "\n")
        full = f"{remote}:{dest}"
        r = subprocess.run(["rclone", "copy", str(root), full, "--files-from", str(lst), "--transfers", "4", "--stats", "60s",
                            "--stats-one-line", "-v"], text=True)
        st["copy_rc"] = r.returncode
        c = subprocess.run(["rclone", "check", str(root), full, "--files-from", str(lst), "--one-way", "--combined", "-"],
                           capture_output=True, text=True)
        st["check_rc"] = c.returncode
        st["check_not_ok"] = [l for l in c.stdout.splitlines() if l and not l.startswith("=")][:20]
        z = subprocess.run(["rclone", "size", full, "--json"], capture_output=True, text=True)
        try:
            sz = json.loads(z.stdout)
            st["remote_count"], st["remote_bytes"] = sz.get("count"), sz.get("bytes")     # remote folder is run-specific, so counts must match
        except Exception:
            st["remote_count"] = st["remote_bytes"] = None
        st["verified"] = (r.returncode == 0 and c.returncode == 0 and st["remote_count"] == st["n_files"] and st["remote_bytes"] == st["bytes"])
    else:                                                       # local_mount: shutil copy + size comparison
        base = Path(dest)
        for rel, _ in files:
            tgt = base / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, tgt)
        bad = [rel for rel, size in files if not (base / rel).exists() or (base / rel).stat().st_size != size]
        st["mismatch"] = bad[:20]
        st["verified"] = not bad
    return st


def run(cfg, paths, dry_run=False, ckpt=None, cache=None, models=None, run_label=None, out_status=None):
    sc = cfg.get("sync", {}).get("drive", {}) or {}
    remote = os.environ.get("EXP003_RCLONE_REMOTE") or sc.get("rclone_remote") or ""
    rpath = (os.environ.get("EXP003_REMOTE_PATH") or sc.get("remote_path") or "").strip("/")
    mount = os.environ.get("EXP003_DRIVE_MOUNT") or sc.get("local_mount") or ""
    ckpt = bool(int(os.environ.get("SYNC_CKPT", "0"))) if ckpt is None else ckpt
    cache = bool(int(os.environ.get("SYNC_CACHE", "0"))) if cache is None else cache
    from .common import all_models
    models = models or all_models(cfg)
    run_names = [cfg["train"]["run_name_template"].format(dist=m.split(":")[0], seed=(m.split(":") + [cfg["train"]["seed"]])[1]) for m in models]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_label = run_label or ("exp003_" + ("+".join(m.replace(":", "_s") for m in models) if len(models) <= 3 else "all%d" % len(models)))
    folder = f"{stamp}_{run_label}"
    status_path = Path(out_status) if out_status else paths["results_dir"] / "sync_status.json"
    exp = paths["exp"]

    # 1. what would be uploaded
    groups = []   # (tag, root, files, subfolder)
    items = ["results.json", "results", "manifests", "logs", "config.yaml"]
    f, sk, un = collect(exp, items)
    groups.append(("main", exp, f, "", sk, un))
    if ckpt:
        for rn in run_names:
            root = paths["train_base"] / "outputs" / rn
            f2, sk2, un2 = collect(root, ["."]) if root.is_dir() else ([], [], [])
            groups.append((f"ckpt_{rn}", root, f2, f"checkpoints/{rn}", sk2, un2))
    if cache:
        for rn in run_names:
            for suffix in ("", "_repro"):
                root = paths["cache"] / (rn + suffix)
                f2, sk2, un2 = collect(root, ["."]) if root.is_dir() else ([], [], [])
                if f2:
                    groups.append((f"cache_{rn}{suffix}", root, f2, f"cache/{rn}{suffix}", sk2, un2))
    log(f"sync: target folder name '{folder}'  (ckpt={ckpt}, cache={cache}, dry_run={dry_run})")
    total = 0
    for tag, root, files, sub, sk2, un2 in groups:
        b = sum(s for _, s in files)
        total += b
        log(f"  [{tag}] {len(files)} files, {_fmt(b)} (root {root})")
        for rel, size in files[: (None if dry_run else 0)]:
            log(f"      {rel}  {_fmt(size)}")
        for rel, hits in sk2:
            log(f"  WARNING skipped (looks like a secret: {','.join(hits)}): {rel}")
        if un2:
            log(f"  note: {len(un2)} large files were not pattern-scanned (binary data)")
    log(f"  total {_fmt(total)}")
    if dry_run:
        log("sync --dry-run: nothing uploaded, nothing written")
        return {"status": "dry_run", "total_bytes": total}

    # 2. destination
    mode, reason = None, None
    if remote:
        reason = _rclone_ok(remote)
        mode = None if reason else "rclone"
    elif mount:
        mode = "mount" if Path(mount).is_dir() else None
        reason = None if mode else f"sync.drive.local_mount '{mount}' is not a directory"
    else:
        reason = "sync.drive.rclone_remote is empty (TODO in config.yaml) and no local_mount is set"
    if mode is None or not rpath:
        reason = reason or "sync.drive.remote_path is empty (TODO in config.yaml)"
        print(SETUP_HELP.format(reason=reason), file=sys.stderr)
        st = {"status": "skipped", "reason": reason, "time": stamp}
        write_json(st, status_path)
        return st

    # 3. run meta (goes into the upload), then upload
    meta = build_run_meta(cfg, paths, run_label, models)
    write_json(meta, paths["results_dir"] / "run_meta.json")
    groups[0] = ("main", exp, collect(exp, items)[0], "", groups[0][4], groups[0][5])
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="exp003_sync_"))
    base = f"{rpath}/{folder}" if mode == "rclone" else str(Path(mount) / rpath / folder)
    sts = []
    for tag, root, files, sub, _, _ in groups:
        dest = f"{base}/{sub}" if (sub and mode == "rclone") else (str(Path(base) / sub) if sub else base)
        sts.append(_upload_group(mode, root, files, dest, remote, tmp, tag))
    ok = all(s["verified"] for s in sts)
    st = {"status": "ok" if ok else "verification_failed", "mode": mode, "folder": folder, "time": stamp, "groups": sts,
          "skipped_secret_like": {tag: [{"file": r, "patterns": h} for r, h in sk2] for tag, _, _, _, sk2, _ in groups if sk2}}
    write_json(st, status_path)
    # status file itself goes up after the check (it cannot be part of its own verification)
    if mode == "rclone":
        subprocess.run(["rclone", "copyto", str(status_path), f"{remote}:{base}/results/sync_status.json"])
    else:
        (Path(base) / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy2(status_path, Path(base) / "results" / "sync_status.json")
    log(f"sync: {st['status']} -> {'rclone ' + remote + ':' if mode == 'rclone' else ''}{base}")
    return st
