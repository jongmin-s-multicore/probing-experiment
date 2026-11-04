"""Source: experiments/001_size_collapse_diagnosis/src/exp001/drive.py (copied; env var prefix EXP001_ renamed to EXP002_, config paths adapted to 002).
Dataset download from Google Drive with rclone (data.source = drive), plus the post-download integrity check (any source).

No rclone configuration content, token or credential is read, printed or written here: rclone is only called as a subprocess with a remote
NAME. Which of the two access modes works on a given machine could not be tested by the author (no rclone / credentials on the authoring
machine): both are marked unverified in the README.

  mode "root_folder_id": rclone copy "<remote>,root_folder_id=<ID>:<split>" <dest>   (folder owned by another account; connection string)
                         or, data_impl.drive.root_folder_style = flag:  rclone copy <remote>:<split> <dest> --drive-root-folder-id <ID>
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

OK_MARK = ".exp001_download_ok"   # same marker as 001 on purpose: splits downloaded by 001 count as complete
QUOTA_RX = re.compile(r"\b403\b|rate.?limit|quota", re.I)       # generic pattern; the exact Drive wording is not assumed
SIZE_NOTE = "sizes are only what `rclone size` reports at download time; nothing was measured by the authoring session"

HELP_SETUP = """\
Drive download cannot start ({reason}).
What to do (on the machine that runs the experiment):
  1. install rclone (https://rclone.org/install/) and make a Google Drive remote:  rclone config  (storage "drive"; headless machine:
     answer "n" to the browser question and run `rclone authorize "drive"` on a PC with a browser). This is the same remote that
     `python run.py sync` uses for uploading results; the credentials stay in rclone's own config and are never read by this code.
  2. put the remote NAME in config.yaml data_impl.drive.rclone_remote (or env EXP002_RCLONE_REMOTE)
  3. access to the data folder (it is owned by another account):
     (a) default, no setup in Drive: the folder id in data_impl.drive.root_folder_id is used as the root:
           rclone copy "<remote>,root_folder_id=<ID>:<split>" "$DATASET_DIR/cloth_sim/<split>"
         if your rclone rejects that syntax, set data_impl.drive.root_folder_style: flag  (uses --drive-root-folder-id <ID>).
     (b) in the Drive web UI add the folder to "My Drive" as a shortcut ("Add shortcut to Drive") or make a copy, then set
         data_impl.drive.remote_path to its path in your Drive (e.g. cloth_sim_data); (or env EXP002_DATA_REMOTE_PATH)
  4. check:  python run.py download --dry-run     (lists files and sizes through rclone; downloads nothing)
Both (a) and (b) are untested by the author."""

QUOTA_HELP = """\
The rclone output contains a 403 / rate-limit / quota pattern. If this is Google Drive's download limit for the (shared) folder:
  - wait (hours) and run the same command again; `rclone copy` is restart-safe and finished splits are skipped
  - or make your own copy of the split folders in your Drive (Drive web: select folder -> Make a copy), set data_impl.drive.remote_path
    to the copy and download from there (access mode b)
  - lower the load:  EXP002_RCLONE_FLAGS="--transfers 2 --tpslimit 2"
This pattern match is generic (the exact wording of Drive's message is not assumed); read the lines above to confirm the cause."""


# ----------------------------------------------------------------------------- config helpers
def drive_cfg(cfg):
    d = dict(cfg["data_impl"].get("drive") or {})
    d["rclone_remote"] = (os.environ.get("EXP002_RCLONE_REMOTE") or d.get("rclone_remote") or "").strip()
    d["remote_path"] = (os.environ.get("EXP002_DATA_REMOTE_PATH") or d.get("remote_path") or "").strip().strip("/")
    d["root_folder_id"] = (d.get("root_folder_id") or "").strip()
    d["root_folder_style"] = d.get("root_folder_style") or "connection_string"
    d["rclone_bin"] = os.environ.get("EXP002_RCLONE_BIN") or d.get("rclone_bin") or "rclone"
    return d


def data_source(cfg):
    return (os.environ.get("EXP002_DATA_SOURCE") or cfg["data"].get("source") or "drive").strip().lower()


def select_splits(cfg):
    """Splits 002 needs: *_train of data.train_levels_b (world-model training + probe training) and *_test of every level."""
    pat = cfg["data_impl"]["split_pattern"]
    out = [pat.format(size=l, part="train") for l in cfg["data"]["train_levels_b"]]
    out += [pat.format(size=l, part="test") for l in cfg["data"]["levels"]]
    return out


def expected_n(cfg, split):
    return cfg["data_impl"]["expected_episodes"].get(split.rsplit("_", 1)[-1])


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
    return os.environ.get("EXP002_RCLONE_FLAGS", "").split()


def _setup_problem(d):
    if not shutil.which(d["rclone_bin"]):
        return f"{d['rclone_bin']} is not installed"
    if not d["rclone_remote"]:
        return "no rclone remote name: data_impl.drive.rclone_remote / EXP002_RCLONE_REMOTE is empty"
    if not d["remote_path"] and not d["root_folder_id"]:
        return "neither data_impl.drive.remote_path nor data_impl.drive.root_folder_id is set"
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


# ----------------------------------------------------------------------------- Colab path: copy from a mounted Drive (no rclone)
# Added 2026-11-04 (확정 기록 10, 실행 환경 변경 서버 -> Colab). Used when env EXP002_DRIVE_MOUNT_SRC is set or a mounted location is found.
# The folder data_impl.drive.root_folder_id belongs to another account, so it is NOT inside MyDrive by itself: the user must add it
# to My Drive ("Add shortcut to Drive" or "Make a copy") before drive.mount can see it. Candidate locations are checked in order.
# UNVERIFIED on a real Colab runtime (the authoring machine has no Drive mount); only the file-copy logic is covered by selftest.
# Expected layout (reported by the user from Drive metadata, 2026-11-04; the mounted form is UNVERIFIED): the user's shortcut 'dino-wm-videos'
# (id 17M0MZGSDPzu7CCT3ez3ABMyZGjavTfx3, owned by another account) contains '[data]/[cloth-sim] cloth_move', which is the folder with
# data_impl.drive.root_folder_id. The names contain brackets and a space: only pathlib iterdir/is_dir are used here (never glob), and the notebook passes
# the path through os.environ (never through a shell command line).
DEFAULT_MOUNT_SRC = "MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move"
SEARCH_ROOT = "MyDrive/dino-wm-videos"


def mount_src_candidates(cfg, mount="/content/drive"):
    """Ordered candidate folders (not yet checked for existence)."""
    rid = (cfg["data_impl"].get("drive") or {}).get("root_folder_id") or ""
    out = []
    if os.environ.get("EXP002_DRIVE_MOUNT_SRC"):
        out.append(Path(os.environ["EXP002_DRIVE_MOUNT_SRC"]).expanduser())
    out.append(Path(mount) / DEFAULT_MOUNT_SRC)
    if rid:
        out.append(Path(mount) / ".shortcut-targets-by-id" / rid)       # path form for a shortcut target (recalled, unverified)
    return out


def _subdirs(d):
    try:
        return sorted(p for p in Path(d).iterdir() if p.is_dir())
    except OSError:
        return []


def find_mount_src(cfg, splits, mount="/content/drive", max_depth=3):
    """Folder that contains every split directory. Order: explicit/default candidates (the folder itself, then its direct children), then a bounded
    search (depth <= max_depth) below <mount>/MyDrive/dino-wm-videos. Returns None when nothing matches."""
    def has_all(b):
        return all((b / s).is_dir() for s in splits)
    for c in mount_src_candidates(cfg, mount):
        if c.is_dir():
            for base in [c] + _subdirs(c):
                if has_all(base):
                    return base
    root = Path(mount) / SEARCH_ROOT
    level = [root] if root.is_dir() else []
    for _ in range(max_depth + 1):
        nxt = []
        for b in level:
            if has_all(b):
                return b
            nxt += _subdirs(b)
        level = nxt
    return None


def _local_bytes(dest):
    return sum(f.stat().st_size for f in Path(dest).iterdir() if f.is_file() and f.name != OK_MARK)


def _copy_one(src, dst):
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def copy_from_mount(cfg, paths, splits, src_root, dry_run=False, force=False, meta_counter=meta_len, workers=8):
    """Copy <src_root>/<split>/* to <data_root>/<split>/ (file copied if missing or size differs). A split is skipped when the completion
    marker exists, mp4 count == metadata length == expected, and (marker written by this function) local total bytes == recorded bytes.
    Returns 0 on success, 3 when a split could not be completed."""
    from concurrent.futures import ThreadPoolExecutor
    root, src_root = Path(paths["data_root"]), Path(src_root)
    log(f"copy_from_mount: {src_root} -> {root} splits={splits}")
    failed = []
    for s in splits:
        dest = root / s
        ok, why = split_complete(cfg, dest, s, meta_counter)
        if ok and not force:
            try:
                rec = json.loads((dest / OK_MARK).read_text() or "{}")
            except Exception:
                rec = {}
            if rec.get("bytes") is None or rec["bytes"] == _local_bytes(dest):
                log(f"  [{s}] complete ({why}, bytes {'checked' if rec.get('bytes') is not None else 'not recorded'}), skip")
                continue
            log(f"  [{s}] marker present but local bytes differ from the recorded value -> re-verify against the source")
        src = src_root / s
        files = sorted(f for f in src.iterdir() if f.is_file()) if src.is_dir() else []
        todo = [f for f in files if not (dest / f.name).exists() or (dest / f.name).stat().st_size != f.stat().st_size]
        log(f"  [{s}] source files={len(files)} to copy={len(todo)}" + (" (dry-run)" if dry_run else ""))
        if dry_run:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        (dest / OK_MARK).unlink(missing_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            errs = [e for e in ex.map(lambda f: _try(lambda: _copy_one(f, dest / f.name)), todo) if e]
        if errs:
            log(f"  [{s}] {len(errs)} copy errors, first: {errs[0]}")
        n_mp4 = count_mp4(dest)
        n_meta = meta_counter(dest / "metadata.pt") if (dest / "metadata.pt").exists() else None
        exp = expected_n(cfg, s)
        same = all((dest / f.name).exists() and (dest / f.name).stat().st_size == f.stat().st_size for f in files) and bool(files)
        if not errs and same and n_meta is not None and n_mp4 == n_meta and (exp is None or n_mp4 == exp):
            (dest / OK_MARK).write_text(json.dumps({"n_mp4": n_mp4, "n_metadata": n_meta, "bytes": _local_bytes(dest), "source": "drive_mount"}) + "\n")
            log(f"  [{s}] done: {n_mp4} episodes, {_local_bytes(dest) / 1e9:.2f} GB")
        else:
            failed.append(s)
            log(f"  [{s}] INCOMPLETE mp4={n_mp4} metadata={n_meta} expected={exp} all_sizes_match={same}; no marker, next run retries")
    if not dry_run:
        check(cfg, paths, splits)
    if failed:
        log(f"copy_from_mount FAILED for {failed}; re-run (finished files are skipped)")
        return 3
    return 0


def _try(fn):
    try:
        fn()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


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
    vid = cfg["data_impl"].get("video") or {}
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
