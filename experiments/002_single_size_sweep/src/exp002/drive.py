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


def _split_done(cfg, dest, split, meta_counter=meta_len, force=False):
    """True when the split is complete: marker + mp4 count == metadata length == expected + (if recorded) total bytes."""
    ok, why = split_complete(cfg, dest, split, meta_counter)
    if not ok or force:
        return False, why
    try:
        rec = json.loads((Path(dest) / OK_MARK).read_text() or "{}")
    except Exception:
        rec = {}
    if rec.get("bytes") is not None and rec["bytes"] != _local_bytes(dest):
        return False, "marker present but local bytes differ from the recorded value"
    return True, f"{why}, bytes {'checked' if rec.get('bytes') is not None else 'not recorded'}"


def _clean_parts(dest):
    if Path(dest).is_dir():
        for f in Path(dest).glob("*.part"):
            f.unlink()


def _finish_split(cfg, dest, split, source, meta_counter, extra_ok=True):
    """Same verification for every source (mount copy, archive): mp4 count == metadata length == expected. Writes the marker with total bytes."""
    n_mp4 = count_mp4(dest)
    n_meta = meta_counter(Path(dest) / "metadata.pt") if (Path(dest) / "metadata.pt").exists() else None
    exp = expected_n(cfg, split)
    if extra_ok and n_meta is not None and n_mp4 == n_meta and (exp is None or n_mp4 == exp):
        b = _local_bytes(dest)
        (Path(dest) / OK_MARK).write_text(json.dumps({"n_mp4": n_mp4, "n_metadata": n_meta, "bytes": b, "source": source}) + "\n")
        log(f"  [{split}] done: {n_mp4} episodes, {b / 1e9:.2f} GB (source: {source})")
        return True
    log(f"  [{split}] INCOMPLETE mp4={n_mp4} metadata={n_meta} expected={exp}; no marker, next run retries")
    return False


class _Progress:
    """Prints 'label: n/total files, GB, elapsed' at most every `every` seconds (and at the end)."""
    def __init__(self, label, total, every=10.0):
        import time
        self.t0 = self.last = time.time()
        self.label, self.total, self.every, self.n, self.bytes = label, total, every, 0, 0

    def report(self):
        import time
        log(f"    {self.label}: {self.n}/{self.total} files, {self.bytes / 1e9:.2f} GB, {time.time() - self.t0:.0f} s (finished)")

    def step(self, nbytes=0, final=False):
        import time
        self.n += 1
        self.bytes += nbytes
        now = time.time()
        if final or now - self.last >= self.every:
            self.last = now
            log(f"    {self.label}: {self.n}/{self.total} files, {self.bytes / 1e9:.2f} GB, {now - self.t0:.0f} s")


def copy_from_mount(cfg, paths, splits, src_root, dry_run=False, force=False, meta_counter=meta_len, workers=None):
    """Parallel copy <src_root>/<split>/* -> <data_root>/<split>/ (file copied if missing or size differs; temp name *.part then rename, so an interrupted
    run leaves no half-written .mp4). Complete splits are skipped (see _split_done). Returns 0 on success, 3 when a split could not be completed."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    workers = int(workers or os.environ.get("EXP002_COPY_WORKERS") or 16)
    root, src_root = Path(paths["data_root"]), Path(src_root)
    log(f"copy_from_mount: {src_root} -> {root} splits={splits} workers={workers}")
    failed = []
    for s in splits:
        dest = root / s
        done, why = _split_done(cfg, dest, s, meta_counter, force)
        if done:
            log(f"  [{s}] complete ({why}), skip")
            continue
        src = src_root / s
        files = sorted(f for f in src.iterdir() if f.is_file()) if src.is_dir() else []
        todo = [f for f in files if not (dest / f.name).exists() or (dest / f.name).stat().st_size != f.stat().st_size]
        log(f"  [{s}] source files={len(files)} to copy={len(todo)}" + (" (dry-run)" if dry_run else ""))
        if dry_run:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        (dest / OK_MARK).unlink(missing_ok=True)
        _clean_parts(dest)
        errs, prog = [], _Progress(s, len(todo))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_try, (lambda f=f: _copy_one(f, dest / f.name))): f for f in todo}
            for fu in as_completed(futs):
                e = fu.result()
                if e:
                    errs.append(e)
                prog.step(futs[fu].stat().st_size if not e else 0)
        prog.report()
        if errs:
            log(f"  [{s}] {len(errs)} copy errors, first: {errs[0]}")
        same = bool(files) and all((dest / f.name).exists() and (dest / f.name).stat().st_size == f.stat().st_size for f in files)
        if not _finish_split(cfg, dest, s, "drive_mount", meta_counter, extra_ok=(not errs and same)):
            failed.append(s)
    if not dry_run:
        check(cfg, paths, splits)
    if failed:
        log(f"copy_from_mount FAILED for {failed}; re-run (finished files are skipped)")
        return 3
    return 0


# ----------------------------------------------------------------------------- archive mode (one big file instead of thousands of small FUSE reads)
ARCHIVE_NAMES = ("cloth_sim_data.tar", "cloth_sim_data.zip")
ARCHIVE_ROOT = "cloth_sim"          # layout written by make_archive: cloth_sim/<split>/<file>


def all_split_names(cfg):
    pat = cfg["data_impl"]["split_pattern"]
    return {pat.format(size=l, part=part) for l in cfg["data"]["levels"] for part in ("train", "test")}


def find_archive(cfg=None):
    """EXP002_DATA_ARCHIVE (explicit file) or <EXP002_WORK_ROOT>/cloth_sim_data.{tar,zip}. None when absent."""
    if os.environ.get("EXP002_DATA_ARCHIVE"):
        p = Path(os.environ["EXP002_DATA_ARCHIVE"]).expanduser()
        return p if p.is_file() else None
    wr = os.environ.get("EXP002_WORK_ROOT")
    if wr:
        for n in ARCHIVE_NAMES:
            if (Path(wr) / n).is_file():
                return Path(wr) / n
    return None


def archive_target(cfg=None):
    wr = os.environ.get("EXP002_DATA_ARCHIVE") or None
    if wr:
        return Path(wr).expanduser()
    wr = os.environ.get("EXP002_WORK_ROOT")
    return Path(wr) / ARCHIVE_NAMES[0] if wr else None


class ArchiveError(Exception):
    pass


def _entries(archive):
    """List of (name, size, kind, handle) of a tar or zip, kind in file|dir|other (links/devices = other). Raises ArchiveError for unreadable archives."""
    import tarfile
    import zipfile
    a = str(archive)
    try:
        if a.lower().endswith(".zip"):
            zf = zipfile.ZipFile(a)
            out = []
            for zi in zf.infolist():
                mode = (zi.external_attr >> 16) & 0o170000
                kind = "dir" if zi.is_dir() else ("file" if mode in (0, 0o100000) else "other")
                out.append((zi.filename, zi.file_size, kind, zi))
            return out, zf
        tf = tarfile.open(a, "r:*")
        return [(m.name, m.size, "file" if m.isreg() else ("dir" if m.isdir() else "other"), m) for m in tf.getmembers()], tf
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError) as e:
        raise ArchiveError(f"cannot read archive {archive}: {type(e).__name__}: {e} (truncated or not a tar/zip file?). Delete or rename it to fall back to copying from the mounted Drive.")


def _plan_archive(cfg, entries):
    """Map archive entries to (split, relative file name). Handles top-level variants (cloth_sim/<split>/..., <split>/..., <anything>/<split>/...).
    Rejects absolute paths, '..', links/devices that could escape. Returns {split: [(entry, relname)]}, prefix tuple."""
    from pathlib import PurePosixPath
    known = all_split_names(cfg)
    plan, prefixes, tops = {}, set(), []
    for name, size, kind, h in entries:
        norm = name.replace("\\", "/")
        pp = PurePosixPath(norm)
        if pp.is_absolute() or ".." in pp.parts or (pp.parts and ":" in pp.parts[0]):
            raise ArchiveError(f"unsafe path in archive (absolute or '..'): {name!r}; refusing to extract anything")
        if len(tops) < 5 and pp.parts and pp.parts[0] not in tops:
            tops.append(pp.parts[0])
        idx = next((i for i, part in enumerate(pp.parts) if part in known), None)
        if idx is None:
            continue
        if kind == "dir":
            continue
        if kind != "file":
            raise ArchiveError(f"non-regular entry (link/device) inside a split folder: {name!r}; refusing")
        if len(pp.parts) == idx + 1:
            raise ArchiveError(f"file named like a split directory: {name!r}")
        prefixes.add(pp.parts[:idx])
        plan.setdefault(pp.parts[idx], []).append(((name, size, kind, h), "/".join(pp.parts[idx + 1:])))
    if not plan:
        raise ArchiveError(f"no split directory ({sorted(known)[:4]} ...) found in the archive; top-level entries: {tops}. Expected cloth_sim/<split>/<file> (or <split>/<file>).")
    if len(prefixes) > 1:
        raise ArchiveError(f"inconsistent top-level layout in the archive: prefixes {sorted('/'.join(p) for p in prefixes)}")
    return plan, (next(iter(prefixes)) if prefixes else ())


def _copy_big(src, dst, label):
    """Single large sequential copy with progress (src on Drive FUSE -> dst on local disk); *.part then rename; reused if the size matches."""
    import time
    src, dst = Path(src), Path(dst)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        log(f"  local archive copy exists ({dst}), reuse")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    total, done, t0, last = src.stat().st_size, 0, time.time(), time.time()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            b = fi.read(64 << 20)
            if not b:
                break
            fo.write(b)
            done += len(b)
            if time.time() - last >= 10:
                last = time.time()
                log(f"    {label}: {done / 1e9:.2f}/{total / 1e9:.2f} GB, {last - t0:.0f} s")
    os.replace(tmp, dst)


def extract_archive(cfg, paths, splits, archive, dry_run=False, force=False, meta_counter=meta_len):
    """Copy the archive to the local disk (one sequential read; EXP002_ARCHIVE_DIRECT=1 reads it in place), extract the needed incomplete splits into
    <data_root>/<split>/, and verify exactly like the mount copy (mp4 count == metadata length == expected, total bytes recorded). Returns 0 / 3."""
    import shutil as _sh
    root = Path(paths["data_root"])
    todo = [s for s in splits if not _split_done(cfg, root / s, s, meta_counter, force)[0]]
    for s in splits:
        if s not in todo:
            log(f"  [{s}] complete, skip")
    if not todo:
        return 0
    log(f"extract_archive: {archive} -> {root} splits to extract={todo}")
    if dry_run:
        log(f"  dry-run: would copy {Path(archive).stat().st_size / 1e9:.2f} GB archive to the local disk and extract splits {todo}")
        return 0
    direct = os.environ.get("EXP002_ARCHIVE_DIRECT") == "1"
    local = Path(archive) if direct else Path(paths["dataset_dir"]) / "_archive_tmp" / Path(archive).name
    try:
        if not direct:
            _copy_big(archive, local, "archive copy")
        entries, h = _entries(local)
        plan, _ = _plan_archive(cfg, entries)
        missing = [s for s in todo if s not in plan]
        if missing:
            raise ArchiveError(f"archive has no split(s) {missing} (it contains {sorted(plan)}); it is not a complete archive. Delete or rename it to fall back to the mount copy.")
    except (ArchiveError, OSError) as e:
        log(f"ERROR {e}")
        return 3
    failed = []
    rroot = root.resolve()
    for s in todo:
        dest = root / s
        dest.mkdir(parents=True, exist_ok=True)
        (dest / OK_MARK).unlink(missing_ok=True)
        _clean_parts(dest)
        items = plan[s]
        prog, err = _Progress(s, len(items)), None
        for (name, size, kind, handle), rel in items:
            out = (dest / rel)
            if not str(out.resolve()).startswith(str(rroot) + os.sep):
                err = f"path escapes the data root: {name!r}"
                break
            if out.exists() and out.stat().st_size == size:
                prog.step(size)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            try:
                src = h.extractfile(handle) if hasattr(h, "extractfile") else h.open(handle)
                with src, open(tmp, "wb") as fo:
                    _sh.copyfileobj(src, fo, 16 << 20)
                if tmp.stat().st_size != size:
                    raise ArchiveError(f"size mismatch for {name!r}: {tmp.stat().st_size} != {size}")
                os.replace(tmp, out)
            except Exception as e:
                err = f"{type(e).__name__}: {e} while extracting {name!r}"
                break
            prog.step(size)
        prog.report()
        if err:
            log(f"  [{s}] ERROR {err}")
            failed.append(s)
            continue
        if not _finish_split(cfg, dest, s, "archive", meta_counter):
            failed.append(s)
    if hasattr(h, "close"):
        h.close()
    if not direct:
        try:
            local.unlink()
        except OSError:
            pass
    if not failed:
        check(cfg, paths, splits)
    if failed:
        log(f"extract_archive FAILED for {failed}. The archive is incomplete or damaged: delete/rename it (it is not modified by this code) and re-run to copy from the mounted Drive.")
        return 3
    return 0


def make_archive(cfg, paths, splits, dest):
    """Uncompressed tar of the verified local splits as cloth_sim/<split>/<file>. Written to <dest>.tmp and renamed only when complete, so a
    half-written archive is never picked up. Returns True/False (failure is only a warning for the caller)."""
    import tarfile
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    root = Path(paths["data_root"])
    try:
        tmp.unlink(missing_ok=True)
        files = [(s, f) for s in splits for f in sorted(p for p in (root / s).iterdir() if p.is_file() and p.name != OK_MARK and not p.name.endswith(".part"))]
        log(f"make_archive: {len(files)} files -> {dest} (via {tmp.name}; one-time, later sessions use the archive)")
        prog = _Progress("archive write", len(files))
        with tarfile.open(tmp, "w") as tf:
            for s, f in files:
                tf.add(str(f), arcname=f"{ARCHIVE_ROOT}/{s}/{f.name}", recursive=False)
                prog.step(f.stat().st_size)
        prog.report()
        os.replace(tmp, dest)
        log(f"make_archive: done {dest.stat().st_size / 1e9:.2f} GB")
        return True
    except Exception as e:
        log(f"WARNING make_archive failed ({type(e).__name__}: {e}); no archive written, the local data is unaffected")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def prepare_data(cfg, paths, splits, dry_run=False, force=False, meta_counter=meta_len):
    """Colab data stage: (1) all splits complete -> nothing; (2) archive present -> extract it; (3) else parallel copy from the mount and, after
    verification, write the archive to Drive once. Returns 0 / 2 (source not found) / 3 (incomplete)."""
    root = Path(paths["data_root"])
    todo = [s for s in splits if not _split_done(cfg, root / s, s, meta_counter, force)[0]]
    if not todo:
        for s in splits:
            log(f"  [{s}] complete, skip")
        return 0
    arc = find_archive(cfg)
    if arc is not None:
        log(f"data source: archive {arc} (one file; fast path)")
        return extract_archive(cfg, paths, splits, arc, dry_run, force, meta_counter)
    src = find_mount_src(cfg, splits)
    if src is None:
        log("data: no archive and no folder containing all splits %s found in %s. Put cloth_sim_data.tar/.zip into the work root or add the shared "
            "folder to My Drive and set EXP002_DRIVE_MOUNT_SRC." % (splits, [str(c) for c in mount_src_candidates(cfg)]))
        return 2
    log(f"data source: mounted Drive folder {src} (no archive found; parallel copy)")
    rc = copy_from_mount(cfg, paths, splits, src, dry_run=dry_run, force=force, meta_counter=meta_counter)
    tgt = archive_target(cfg)
    if rc == 0 and not dry_run and tgt is not None and os.environ.get("EXP002_MAKE_ARCHIVE", "1") != "0" and not tgt.exists():
        if all(_split_done(cfg, root / s, s, meta_counter)[0] for s in splits):
            make_archive(cfg, paths, splits, tgt)
    return rc


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
