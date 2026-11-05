"""Make the 001 results.json available for the H-H2 verdict (config criteria.h2_baseline.results_json, relative to this experiment folder).
The Colab snapshot of 003 does not contain experiments/001, so the file has to be found on Drive (size must equal colab.baseline_001_bytes) and copied.
Fails loudly (rc 1) when it is not found or when the baseline runs/slope path of config cannot be read from it. Standard library only."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from .common import EXP_DIR, log
from . import verdict

SKIP_DIRS = {"train_runs", "exp003_cache", "exp002_cache", "outputs", "checkpoints", "cloth_sim", "data", "manifests", "logs", ".git", "quarantine",
             "node_modules", "__pycache__", ".Trash", ".Trash-0", "[data]", "dino-wm-videos"}


def target_path(cfg, base_dir=None):
    return (Path(base_dir or EXP_DIR) / cfg["criteria"]["h2_baseline"]["results_json"]).resolve()


def read_check(cfg, path):
    """The baseline slopes of config runs/slope_path, read through the verdict code. Returns the info dict (baseline.found etc.)."""
    P = verdict.Prereg(cfg)
    vals, info = _slopes_from(cfg, path, P)
    info["values_readable"] = vals is not None
    return info


def _slopes_from(cfg, path, P):
    # verdict.baseline_slopes resolves cfg results_json against a base dir; here the file is given directly, so reuse it with a one-entry cfg
    import copy
    c = copy.deepcopy(cfg)
    c["criteria"]["h2_baseline"]["results_json"] = str(Path(path).resolve())      # absolute path: Path(base) / absolute == absolute
    return verdict.baseline_slopes(c, "/", P)


def _candidates_direct(roots):
    names = ("results.json", "001_results.json", "exp001_results.json")
    subs = ("", "001_size_collapse_diagnosis", "exp001_results", "exp001", "001", "results", "exp001_work", "001_size_collapse_diagnosis/results")
    for r in roots:
        for sub in subs:
            for n in names:
                yield Path(r) / sub / n


def find_candidates(roots, size, deadline_s=240.0, max_depth=6):
    """Files named *results.json with exactly `size` bytes. Direct likely spots first, then a bounded walk (depth, time) that skips data/checkpoint folders."""
    t0 = time.time()
    seen, out = set(), []

    def consider(p):
        try:
            p = Path(p)
            if p.is_file() and p.stat().st_size == size:
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
        except OSError:
            pass
    for p in _candidates_direct(roots):
        consider(p)
    if out:
        return out, "direct"
    timed_out = False
    for r in roots:
        r = Path(r)
        if not r.is_dir():
            continue
        base_depth = len(r.parts)
        for dp, dns, fns in os.walk(str(r)):
            if time.time() - t0 > deadline_s:
                timed_out = True
                break
            if len(Path(dp).parts) - base_depth >= max_depth:
                dns[:] = []
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for fn in fns:
                if fn.endswith("results.json"):
                    consider(Path(dp) / fn)
    return out, ("walk (time limit hit)" if timed_out else "walk")


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def default_roots():
    roots = [p for p in os.environ.get("EXP003_001_SEARCH", "").split(os.pathsep) if p]
    wr = os.environ.get("EXP003_WORK_ROOT")
    if wr:
        roots += [wr, str(Path(wr).parent)]
    roots += ["/content/drive/MyDrive", str(Path.home())]
    out, seen = [], set()
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def run(cfg, base_dir=None, roots=None, deadline_s=240.0):
    """Returns (ok, info). Installs the file at target_path(cfg) when missing or when it does not have the expected size."""
    size = int(cfg["colab"]["baseline_001_bytes"])
    dst = target_path(cfg, base_dir)
    info = {"target": str(dst), "expected_bytes": size}
    if dst.is_file() and dst.stat().st_size == size:
        info.update(action="already_present", source=str(dst))
    else:
        if dst.exists():
            info["note"] = "target exists but has %d bytes (expected %d); looking for the right file, the existing one is kept as %s.bak" % (dst.stat().st_size, size, dst.name)
        cands, how = find_candidates(roots if roots is not None else default_roots(), size, deadline_s=deadline_s)
        info["search"] = how
        info["candidates"] = [str(c) for c in cands]
        if not cands:
            info.update(action="not_found", found=False)
            log("baseline: 001 results.json (%d bytes) NOT FOUND (search: %s). Put the file at %s or set EXP003_001_SEARCH. H-H2 cannot be evaluated without it." % (size, how, dst))
            return False, info
        digests = {sha256_of(c) for c in cands}
        if len(digests) > 1:
            info.update(action="ambiguous", found=False)
            log("baseline: %d files with %d bytes but DIFFERENT contents: %s. Not guessing; copy the right one to %s." % (len(cands), size, [str(c) for c in cands], dst))
            return False, info
        src = cands[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.copyfile(dst, str(dst) + ".bak")
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        info.update(action="copied", source=str(src))
        log("baseline: copied %s -> %s" % (src, dst))
    info["sha256"] = sha256_of(dst)
    try:
        b = read_check(cfg, dst)
    except Exception as e:                      # not JSON etc.
        info.update(found=False, read_error="%s: %s" % (type(e).__name__, e))
        log("baseline: %s cannot be read: %s" % (dst, e))
        return False, info
    info.update(found=bool(b.get("found")), baseline=b)
    ok = bool(b.get("found")) and bool(b.get("values_readable"))
    log("baseline.found=%s values_readable=%s values=%s" % (b.get("found"), b.get("values_readable"), b.get("values")))
    if not ok:
        log("baseline: file present but the config runs %s / slope path cannot be read from it (%s)" % (cfg["criteria"]["h2_baseline"]["runs"], b))
    return ok, info
