"""Dataset download from Google Drive with rclone (data.source = drive), plus the post-download integrity check (any source).

No rclone configuration content, token or credential is read, printed or written here: rclone is only called as a subprocess with a remote
NAME. Which of the two access modes works on a given machine could not be tested by the author (no rclone / credentials on the authoring
machine): both are marked unverified in the README.

  mode "root_folder_id": rclone copy "<remote>,root_folder_id=<ID>:<split>" <dest>   (folder owned by another account; connection string)
                         or, data.drive.root_folder_style = flag:  rclone copy <remote>:<split> <dest> --drive-root-folder-id <ID>
  mode "remote_path"   : rclone copy "<remote>:<remote_path>/<split>" <dest>        (the folder was added as a shortcut / copied into the user's Drive)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

from .common import write_json, log

OK_MARK = ".exp001_download_ok"
QUOTA_RX = re.compile(r"\b403\b|rate.?limit|quota", re.I)       # generic pattern; the exact Drive wording is not assumed
SIZE_NOTE = "sizes are only what `rclone size` reports at download time; nothing was measured by the authoring session"

HELP_SETUP = """\
Drive download cannot start ({reason}).
What to do (on the machine that runs the experiment):
  1. install rclone (https://rclone.org/install/) and make a Google Drive remote:  rclone config  (storage "drive"; headless machine:
     answer "n" to the browser question and run `rclone authorize "drive"` on a PC with a browser). This is the same remote that
     `python run.py sync` uses for uploading results; the credentials stay in rclone's own config and are never read by this code.
  2. put the remote NAME in config.yaml data.drive.rclone_remote (or sync.drive.rclone_remote, or env EXP001_RCLONE_REMOTE)
  3. access to the data folder (it is owned by another account):
     (a) default, no setup in Drive: the folder id in data.drive.root_folder_id is used as the root:
           rclone copy "<remote>,root_folder_id=<ID>:<split>" "$DATASET_DIR/cloth_sim/<split>"
         if your rclone rejects that syntax, set data.drive.root_folder_style: flag  (uses --drive-root-folder-id <ID>).
     (b) in the Drive web UI add the folder to "My Drive" as a shortcut ("Add shortcut to Drive") or make a copy, then set
         data.drive.remote_path to its path in your Drive (e.g. cloth_sim_data); (or env EXP001_DATA_REMOTE_PATH)
  4. check:  python run.py download --dry-run     (lists files and sizes through rclone; downloads nothing)
Both (a) and (b) are untested by the author."""

QUOTA_HELP = """\
The rclone output contains a 403 / rate-limit / quota pattern. If this is Google Drive's download limit for the (shared) folder:
  - wait (hours) and run the same command again; `rclone copy` is restart-safe and finished splits are skipped
  - or make your own copy of the split folders in your Drive (Drive web: select folder -> Make a copy), set data.drive.remote_path
    to the copy and download from there (access mode b)
  - lower the load:  EXP001_RCLONE_FLAGS="--transfers 2 --tpslimit 2"
This pattern match is generic (the exact wording of Drive's message is not assumed); read the lines above to confirm the cause."""


# ----------------------------------------------------------------------------- config helpers
def drive_cfg(cfg):
    d = dict(cfg["data"].get("drive") or {})
    d["rclone_remote"] = (os.environ.get("EXP001_RCLONE_REMOTE") or d.get("rclone_remote")
                          or (cfg.get("sync", {}).get("drive", {}) or {}).get("rclone_remote") or "").strip()
    d["remote_path"] = (os.environ.get("EXP001_DATA_REMOTE_PATH") or d.get("remote_path") or "").strip().strip("/")
    d["root_folder_id"] = (d.get("root_folder_id") or "").strip()
    d["root_folder_style"] = d.get("root_folder_style") or "connection_string"
    d["rclone_bin"] = os.environ.get("EXP001_RCLONE_BIN") or d.get("rclone_bin") or "rclone"
    return d


def data_source(cfg):
    return (os.environ.get("EXP001_DATA_SOURCE") or cfg["data"].get("source") or "drive").strip().lower()


def select_splits(cfg, dists):
    """Default splits (ind_train + 9 *_test) plus the *_train splits that the given distributions use."""
    blk = cfg["_ds"]
    splits = list(blk["download_default_splits"])
    for d in dists:
        for spec in cfg["train"]["distributions"][d]:
            if spec["source"] not in splits:
                splits.append(spec["source"])
    return splits


def expected_n(cfg, split):
    e = cfg["_ds"]["expected_episodes"]
    return e.get(split) if split in e else e.get(split.rsplit("_", 1)[-1])


def rclone_source(d, split):
    """Returns (source string, extra flags, mode). The source contains the remote name and the (non-secret) folder id only."""
    remote = d["rclone_remote"] or "<remote>"
    if d["remote_path"]:
        return f"{remote}:{d['remote_path']}/{split}", [], "remote_path"
    rid = d["root_folder_id"] or "<root_folder_id>"
    if d["root_folder_style"] == "flag":
        return f"{remote}:{split}", ["--drive-root-folder-id", rid], "root_folder_id"
    return f"{remote},root_folder_id={rid}:{split}", [], "root_folder_id"


def _extra_flags():
    return os.environ.get("EXP001_RCLONE_FLAGS", "").split()


def _setup_problem(d):
    if not shutil.which(d["rclone_bin"]):
        return f"{d['rclone_bin']} is not installed"
    if not d["rclone_remote"]:
        return "no rclone remote name: data.drive.rclone_remote / sync.drive.rclone_remote / EXP001_RCLONE_REMOTE is empty"
    if not d["remote_path"] and not d["root_folder_id"]:
        return "neither data.drive.remote_path nor data.drive.root_folder_id is set"
    return None


# ----------------------------------------------------------------------------- local state of a split
def count_mp4(dest):
    return len(list(Path(dest).glob("*.mp4"))) if Path(dest).is_dir() else 0


def meta_len(path):
    """Number of entries of metadata.pt (weights_only load). None when unreadable or torch is missing."""
    try:
        import torch
        return len(torch.load(path, map_location="cpu", weights_only=True))
    except Exception:
        return None


def probe_video(path):
    """Resolution and frame count of one mp4 via OpenCV. Returns a dict with None values when cv2 is missing / the file is unreadable."""
    out = {"file": Path(path).name, "width": None, "height": None, "frames": None, "error": None}
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            out["error"] = "cannot open"
        else:
            out["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))      # container-reported count
        cap.release()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def split_complete(cfg, dest, split, meta_counter=meta_len):
    """Done marker present AND mp4 count == metadata.pt length == expected episodes. Returns (bool, reason)."""
    dest = Path(dest)
    exp = expected_n(cfg, split)
    if not (dest / OK_MARK).exists():
        return False, "no completion marker"
    n_mp4 = count_mp4(dest)
    n_meta = meta_counter(dest / "metadata.pt") if (dest / "metadata.pt").exists() else None
    if n_meta is None:
        return False, "metadata.pt missing or unreadable"
    if n_mp4 != n_meta or (exp is not None and n_mp4 != exp):
        return False, f"mp4={n_mp4} metadata={n_meta} expected={exp}"
    return True, f"mp4={n_mp4} metadata={n_meta}"


# ----------------------------------------------------------------------------- rclone calls
def _run_streaming(cmd, tail=60):
    """Run, echo output lines, return (returncode, last lines). Output is only echoed, never stored in a file."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = deque(maxlen=tail)
    for line in p.stdout:
        sys.stdout.write(line)
        last.append(line.rstrip("\n"))
    return p.wait(), list(last)


def dry_run_split(d, split, dest):
    src, flags, mode = rclone_source(d, split)
    cmd = [d["rclone_bin"], "copy", src, str(dest), *flags, *_extra_flags()]
    log(f"  [{split}] would run: " + " ".join(f'"{c}"' if "," in c else c for c in cmd))
    if _setup_problem(d):
        return {"split": split, "mode": mode, "listed": False}
    r = subprocess.run([d["rclone_bin"], "lsf", "-R", "--files-only", src, *flags], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  [{split}] rclone lsf failed (rc={r.returncode}): " + " ".join(r.stderr.strip().splitlines()[-3:]))
        if QUOTA_RX.search(r.stderr):
            print(QUOTA_HELP, file=sys.stderr)
        return {"split": split, "mode": mode, "listed": False, "rc": r.returncode}
    names = [x for x in r.stdout.splitlines() if x]
    n_mp4 = sum(x.endswith(".mp4") for x in names)
    has_meta = "metadata.pt" in names
    sz = subprocess.run([d["rclone_bin"], "size", "--json", src, *flags], capture_output=True, text=True)
    nbytes = None
    try:
        nbytes = json.loads(sz.stdout).get("bytes")
    except Exception:
        pass
    log(f"  [{split}] remote: {n_mp4} mp4, metadata.pt {'present' if has_meta else 'MISSING'}, "
        f"{'%.2f GB' % (nbytes / 1e9) if nbytes is not None else 'size unknown'}")
    return {"split": split, "mode": mode, "listed": True, "n_mp4": n_mp4, "metadata_pt": has_meta, "bytes": nbytes}


def download_split(d, split, dest):
    """rclone copy of one split. Returns (rc, quota_pattern_seen)."""
    src, flags, mode = rclone_source(d, split)
    Path(dest).mkdir(parents=True, exist_ok=True)
    cmd = [d["rclone_bin"], "copy", src, str(dest), *flags, "--retries", "3", "--low-level-retries", "10",
           "--stats", "60s", "-v", *_extra_flags()]
    log(f"  [{split}] mode={mode}: {d['rclone_bin']} copy {src} {dest} {' '.join(flags)}")
    rc, last = _run_streaming(cmd)
    return rc, any(QUOTA_RX.search(x) for x in last)


# ----------------------------------------------------------------------------- stage
def download(cfg, paths, splits, dry_run=False, force=False, meta_counter=meta_len, video_prober=probe_video):
    """Download the splits (skip complete ones), then run the integrity check. Returns a process exit code."""
    d = drive_cfg(cfg)
    root = Path(paths["data_root"])
    log(f"download: source=drive -> {root}  splits={splits}")
    log(f"  access: {'remote_path' if d['remote_path'] else 'root_folder_id'}  remote='{d['rclone_remote'] or '<unset>'}'  "
        f"(both access modes untested by the author)")
    todo = []
    for s in splits:
        ok, why = split_complete(cfg, root / s, s, meta_counter)
        if ok and not force:
            log(f"  [{s}] complete ({why}), skip")
        else:
            todo.append(s)
    if dry_run:
        prob = _setup_problem(d)
        if prob:
            log(f"  dry-run: cannot query Drive ({prob}); only the planned commands are shown")
        info = [dry_run_split(d, s, root / s) for s in todo]
        tot = sum(i.get("bytes") or 0 for i in info)
        if info and all(i.get("listed") for i in info):
            log(f"  dry-run total to download: {tot / 1e9:.2f} GB in {len(todo)} splits (skipped as complete: {len(splits) - len(todo)})")
        elif prob:
            print(HELP_SETUP.format(reason=prob), file=sys.stderr)
        return 0
    if todo:
        prob = _setup_problem(d)
        if prob:
            print(HELP_SETUP.format(reason=prob), file=sys.stderr)
            return 2
        r = subprocess.run([d["rclone_bin"], "listremotes"], capture_output=True, text=True)
        if f"{d['rclone_remote']}:" not in r.stdout.split():
            print(HELP_SETUP.format(reason=f"rclone remote '{d['rclone_remote']}' not found in `rclone listremotes`"), file=sys.stderr)
            return 2
    failed = []
    for s in todo:
        dest = root / s
        rc, quota = download_split(d, s, dest)
        if rc != 0:
            failed.append(s)
            log(f"  [{s}] rclone exit code {rc}")
            if quota:
                print(QUOTA_HELP, file=sys.stderr)
            continue
        n_mp4 = count_mp4(dest)
        n_meta = meta_counter(dest / "metadata.pt") if (dest / "metadata.pt").exists() else None
        exp = expected_n(cfg, s)
        if n_meta is not None and n_mp4 == n_meta and (exp is None or n_mp4 == exp):
            (dest / OK_MARK).write_text(json.dumps({"n_mp4": n_mp4, "n_metadata": n_meta}) + "\n")
            log(f"  [{s}] done: {n_mp4} episodes")
        else:
            log(f"  [{s}] WARNING copy finished but mp4={n_mp4} metadata={n_meta} expected={exp}; no completion marker, next run retries")
    check(cfg, paths, splits, meta_counter=meta_counter, video_prober=video_prober)
    if failed:
        log(f"download FAILED for splits {failed}; re-run the same command (finished splits are skipped, partial ones resume)")
        return 3
    return 0


def check(cfg, paths, splits, meta_counter=meta_len, video_prober=probe_video, out_path=None):
    """Integrity check of the local splits -> results/download_check.json. Mismatches are warnings, never errors."""
    blk = cfg["_ds"]
    vid = blk.get("video") or {}
    root = Path(paths["data_root"])
    rows, warnings = {}, []
    for s in splits:
        d = root / s
        exp = expected_n(cfg, s)
        row = {"present": d.is_dir(), "expected_episodes": exp, "n_mp4": count_mp4(d), "n_metadata": None, "first_video": None,
               "completion_marker": (d / OK_MARK).exists()}
        if (d / "metadata.pt").exists():
            row["n_metadata"] = meta_counter(d / "metadata.pt")
        mp4 = sorted(d.glob("*.mp4"))[:1] if d.is_dir() else []
        if mp4:
            row["first_video"] = video_prober(mp4[0])
        rows[s] = row
        if not d.is_dir():
            warnings.append(f"{s}: folder missing")
            continue
        if exp is not None and row["n_mp4"] != exp:
            warnings.append(f"{s}: mp4 count {row['n_mp4']} != expected {exp}")
        if row["n_metadata"] is None:
            warnings.append(f"{s}: metadata.pt missing or unreadable (torch needed to read it)")
        elif row["n_metadata"] != row["n_mp4"]:
            warnings.append(f"{s}: metadata entries {row['n_metadata']} != mp4 count {row['n_mp4']}")
        fv = row["first_video"]
        if fv:
            if fv.get("error"):
                warnings.append(f"{s}: first video not probed ({fv['error']})")
            else:
                if vid.get("width") and (fv["width"], fv["height"]) != (vid["width"], vid["height"]):
                    warnings.append(f"{s}: resolution {fv['width']}x{fv['height']} != README {vid['width']}x{vid['height']}")
                if vid.get("frames") and fv["frames"] != vid["frames"]:
                    warnings.append(f"{s}: frames {fv['frames']} != README {vid['frames']}")
    out = {"source": data_source(cfg), "data_root": str(root), "readme_video_description": vid, "splits": rows, "warnings": warnings,
           "note": "mismatches are recorded as warnings, not errors; the README description of the files is the user's text, not verified by the author"}
    out_path = Path(out_path) if out_path else Path(paths["results_dir"]) / "download_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, out_path)
    for w in warnings:
        log("  WARNING " + w)
    log(f"download check: {len(splits)} splits, {len(warnings)} warnings -> {out_path}")
    return out
