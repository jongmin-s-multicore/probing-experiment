"""Experiment 006 glue. Puts experiments/002_single_size_sweep/src on sys.path and re-uses exp002 (probes, evalgrid, stats, masks, model_io,
directions, common) UNMODIFIED (same approach as 004). 002 and 001 files are only read. This file adds: config/path handling for 006, the
READ-ONLY guard for the 002 Drive folder (INPUT_ROOT), checked writes (a failed Drive write must fail loudly), hashing, retry wrapper.
Python 3.9 compatible; no torch import at module level."""
from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True          # never create __pycache__ inside the 002 source folder

EXP_DIR = Path(__file__).resolve().parents[2]


def exp002_dir():
    return Path(os.environ.get("EXP006_EXP002_DIR", str(EXP_DIR.parent / "002_single_size_sweep"))).expanduser().resolve()


_E2 = exp002_dir()
if str(_E2 / "src") not in sys.path:
    sys.path.insert(0, str(_E2 / "src"))

from exp002 import common as C2  # noqa: E402  (re-exported helpers; unmodified 002 module)

log = C2.log
write_json_raw = C2.write_json
read_json = C2.read_json
env_info = C2.env_info
ConfigNull = C2.ConfigNull
cget = C2.cget


class InputWriteRefused(RuntimeError):
    """Something tried to write inside the read-only 002 input folder."""


class OutputCheckError(RuntimeError):
    """A write to the output folder could not be verified (e.g. Drive not mounted / quota)."""


def load_config(path=None):
    return C2.load_config(Path(path) if path else EXP_DIR / "config.yaml")


def sha256_file(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_cfg002(cfg):
    """The 002 config (read only) with probe.kinds / probe.seeds overridden from THIS config (explicit seeds live in 006's config).
    Returns a deep copy; records the 002 file sha256 and refuses (RuntimeError) when it differs from base.config002_sha256."""
    p = (EXP_DIR / cfg["base"]["config002"]).resolve()
    if os.environ.get("EXP006_EXP002_DIR"):
        p = exp002_dir() / "config.yaml"
    sha = sha256_file(p)
    exp = cfg["base"]["config002_sha256"]
    if exp and sha != exp:
        raise RuntimeError("002 config.yaml sha256 %s != expected %s (base.config002_sha256). The 002 settings may have changed; stop." % (sha, exp))
    c = copy.deepcopy(C2.load_config(p))
    c["probe"]["kinds"] = [cfg["task"]["probe_kind"]]
    c["probe"]["seeds"] = [int(s) for s in cfg["seeds"]["probe"]]
    c["_config002_sha256"] = sha
    c["_config002_path"] = str(p)
    return c


def resolve_paths(cfg):
    p = cfg["paths"]
    home = Path.home()
    inp = Path(os.environ.get(p["input_root_env"], str(home / "dinowm_exp002_work"))).expanduser()
    work = Path(os.environ.get(p["work_root_env"], str(home / "dinowm_exp006_work"))).expanduser()
    ds = Path(os.environ.get(p["dataset_env"], str(home / "dinowm" / "data"))).expanduser()
    src = Path(os.environ.get(p["acwm_src_env"], str(home / "dinowm" / "acwm_phys_src"))).expanduser()
    out = {
        "input_root": inp, "work_root": work, "dataset_dir": ds, "data_root": ds / "cloth_sim", "acwm_src": src,
        "train_outputs": inp / p["train_outputs_subdir"], "cache002": inp / p["cache_subdir"], "results002": inp / p["results_subdir"],
        "archive": inp / p["archive_name"],
        "results": work / "results", "parts": work / "results" / "parts", "extract": work / "extract", "cache": work / "cache",
        "logs": EXP_DIR / "logs", "exp": EXP_DIR,
    }
    if _same_or_inside(work, inp) or _same_or_inside(inp, work):
        raise InputWriteRefused("EXP006_WORK_ROOT (%s) and EXP006_INPUT_ROOT (%s) must be different folders and neither inside the other" % (work, inp))
    return out


def _same_or_inside(a, b):
    a, b = Path(os.path.realpath(a)), Path(os.path.realpath(b))
    return a == b or b in a.parents


def run_name(cfg, cond, seed):
    return cfg["runs"]["run_name_pattern"].format(cond=cond, seed=int(seed))


def all_runs(cfg):
    return [(c, int(s), run_name(cfg, c, s)) for c in cfg["runs"]["conditions"] for s in cfg["runs"]["train_seeds"]]


# ------------------------------------------------------------------------------------------ checked writes
def guard_output(paths, target):
    """Raise InputWriteRefused if `target` is inside the read-only input folder or inside the 001/002 experiment folders."""
    t = Path(os.path.realpath(target))
    forbidden = [Path(os.path.realpath(paths["input_root"])), exp002_dir(), EXP_DIR.parent / "001_size_collapse_diagnosis", EXP_DIR.parent / "003_train_dist_dependence"]
    for f in forbidden:
        if t == f or f in t.parents:
            raise InputWriteRefused("refusing to write %s: it is inside the read-only folder %s" % (target, f))


def checked_write_bytes(paths, target, data):
    """Write via tmp + rename, then read back size and sha256. Failure raises OutputCheckError (no silent loss on Drive)."""
    guard_output(paths, target)
    target = Path(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        back = target.read_bytes()
    except OSError as e:
        raise OutputCheckError("write to %s failed: %s: %s" % (target, type(e).__name__, e))
    if len(back) != len(data) or sha256_bytes(back) != sha256_bytes(data):
        raise OutputCheckError("read-back of %s differs from what was written" % target)
    return len(data)


def write_json(paths, obj, target):
    import json
    data = json.dumps(C2._clean(obj), indent=2, allow_nan=False, ensure_ascii=False).encode("utf-8")
    return checked_write_bytes(paths, target, data)


def write_npz(paths, target, **arrays):
    import io
    import numpy as np
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return checked_write_bytes(paths, target, buf.getvalue())


def write_npy(paths, target, arr):
    import io
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, arr)
    return checked_write_bytes(paths, target, buf.getvalue())


def write_marker(paths, target, text="ok\n"):
    return checked_write_bytes(paths, target, text.encode())


def probe_output_dir(paths):
    """Fails loudly if the output folder cannot be written (Drive not mounted, no quota)."""
    d = Path(paths["work_root"])
    guard_output(paths, d / "x")
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OutputCheckError("cannot create output folder %s: %s" % (d, e))
    stamp = ("write-test %s %s\n" % (time.strftime("%F %T"), os.getpid())).encode()
    checked_write_bytes(paths, d / ".write_test", stamp)
    log("output folder OK (write + read-back): %s" % d)


# ------------------------------------------------------------------------------------------ provenance
def git_info():
    def g(args):
        try:
            return subprocess.run(["git"] + args, cwd=str(EXP_DIR), capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    return {"commit": g(["rev-parse", "HEAD"]) or None, "dirty": bool(g(["status", "--porcelain", "--", str(EXP_DIR)])),
            "note": "if the code was unpacked from a notebook snapshot there is no .git: see snapshot_sha256"}


def code_version():
    """Code identity for results.json: git HEAD of the checkout that contains this experiment (None if not a git checkout) and, when the Colab notebook cloned the repo,
    the repo URL and requested ref from EXP006_CODE_REPO_URL / EXP006_CODE_REF."""
    def g(args):
        try:
            r = subprocess.run(["git", "-C", str(EXP_DIR)] + args, capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    head = g(["rev-parse", "HEAD"])
    return {"git_head": head, "git_status_dirty": (bool(g(["status", "--porcelain", "--", "."])) if head else None),
            "repo_url": os.environ.get("EXP006_CODE_REPO_URL"), "requested_ref": os.environ.get("EXP006_CODE_REF")}


def snapshot_sha256():
    """sha256 over the sorted (relative path, file sha256) of this experiment's code (src, run.py, config.yaml, colab/build_notebook.py)."""
    h = hashlib.sha256()
    files = sorted([p for p in (EXP_DIR / "src").rglob("*.py")] + [EXP_DIR / "run.py", EXP_DIR / "config.yaml"])
    for p in files:
        if p.is_file():
            h.update(str(p.relative_to(EXP_DIR)).encode())
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            return {"available": True, "name": torch.cuda.get_device_name(0), "mem_gb": round(pr.total_memory / 1e9, 2), "torch": torch.__version__, "cuda": torch.version.cuda}
        return {"available": False, "torch": torch.__version__}
    except Exception as e:
        return {"available": False, "error": "%s: %s" % (type(e).__name__, e)}


def run_with_retries(fn, max_attempts, name):
    """CLAUDE.md: at most 3 attempts per stage; after the last failure stop and report. Returns (result, attempts_log)."""
    attempts = []
    for i in range(1, int(max_attempts) + 1):
        t0 = time.time()
        try:
            r = fn()
            attempts.append({"attempt": i, "ok": True, "seconds": round(time.time() - t0, 2)})
            return r, attempts
        except (KeyboardInterrupt, SystemExit):
            raise
        except InputWriteRefused:
            raise                                   # a safety stop is not retried
        except Exception as e:                      # noqa: BLE001
            C2.log_exception(e)
            attempts.append({"attempt": i, "ok": False, "error": "%s: %s" % (type(e).__name__, e), "seconds": round(time.time() - t0, 2)})
            log("stage %s attempt %d/%d failed" % (name, i, max_attempts))
    raise StageFailed(name, attempts)


class StageFailed(RuntimeError):
    def __init__(self, name, attempts):
        super().__init__("stage %s failed %d times; stopping (CLAUDE.md retry limit). Last error: %s" % (name, len(attempts), attempts[-1].get("error")))
        self.attempts = attempts
