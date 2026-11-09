#!/usr/bin/env python
"""Operational helpers for running experiments 004 and 005 on Google Colab (notebook colab/exp004_005_colab.ipynb).
Standard library only (the notebook kernel imports it; the experiments run in the uv venv through subprocesses).
Nothing here touches pre-registered values (seeds, thresholds). 002 files are never written: inputs are read-only, outputs go to
exp004_work / exp005_work on Drive and to local /content caches.

  python colab/exp045_tools.py selftest      # CPU-only checks of the helpers (no GPU, no Drive, no network)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP004 = HERE.parent
CONDITION = "multi_60_ind_160"
Q1_FILES = ("frames_u8.npy", "tokens_dino.npy", "tokens_rand.npy", "mask_feat.npy", "actions_raw.npy", "episodes.json", "meta.json", "DONE")


# ------------------------------------------------------------------------------------------ shell / stage runner
def sh(cmd, cwd=None, env=None):
    """Run a shell command, stream its output into the cell, return the exit code."""
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    return p.wait()


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return max(int(x) for x in out.split() if x.strip().isdigit())
    except Exception:
        return None


class VramSampler(threading.Thread):
    """Samples whole-GPU memory.used every `every` s while a stage runs (nvidia-smi; includes other processes). Peak is a sampled lower bound."""

    def __init__(self, every=2.0, probe=gpu_used_mib):
        super().__init__(daemon=True)
        self.every, self.probe, self.peak, self.base, self._stop_evt = every, probe, None, None, threading.Event()

    def run(self):
        self.base = self.probe()
        while not self._stop_evt.is_set():
            v = self.probe()
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v
            self._stop_evt.wait(self.every)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=10)


def run_stage(py, exp_dir, stage, args, env, timing_file=None, label=None, sample_vram=True):
    """python run.py <stage> <args> in exp_dir. Streams output, records wall time / sampled peak VRAM / log path in timing_file (JSON list).
    Returns the record dict (rc != 0 is NOT raised here: use require_ok)."""
    cmd = [py, "-u", "run.py", stage] + list(args)
    print("$ (cd %s && %s)" % (exp_dir, " ".join(cmd)), flush=True)
    sampler = VramSampler() if sample_vram else None
    if sampler:
        sampler.start()
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(exp_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log_path = None
    for line in p.stdout:
        print(line, end="")
        m = re.search(r"log file: (\S+)", line)
        if m and log_path is None:
            log_path = m.group(1)
    rc = p.wait()
    sec = time.time() - t0
    if sampler:
        sampler.stop()
    rec = {"label": label or stage, "stage": stage, "args": list(args), "rc": rc, "seconds": round(sec, 1), "log": log_path,
           "vram_base_mib": sampler.base if sampler else None, "vram_peak_sampled_mib": sampler.peak if sampler else None,
           "finished_utc": time.strftime("%F %T", time.gmtime())}
    if timing_file:
        append_json_list(timing_file, rec)
    return rec


def append_json_list(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        d = []
    d.append(rec)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, path)


def require_ok(rec, results_dir=None):
    """Stop the cell with the log path (and attempts.json) when a stage failed. The 3-attempt guard itself lives in run.py (results_dir/attempts.json)."""
    if rec["rc"] == 0:
        return
    msg = ["STAGE FAILED: %s (rc=%s). Log: %s" % (rec["label"], rec["rc"], rec["log"] or "(no log line seen; look in logs/)")]
    if results_dir:
        ap = Path(results_dir) / "attempts.json"
        if ap.exists():
            msg.append("attempts.json (%s): %s  -> 3 consecutive failures make run.py refuse the stage (CLAUDE.md). Stop and report the cause." % (ap, ap.read_text().replace("\n", " ")))
    raise SystemExit("\n".join(msg))


# ------------------------------------------------------------------------------------------ directories
def link_dirs(work, exp_dir, names=("results", "logs")):
    """Make <exp_dir>/<name> a symlink to <work>/<name> (created). Existing local content is moved into the target (never overwritten: name clash -> kept as .local)."""
    work, exp_dir = Path(work), Path(exp_dir)
    out = {}
    for n in names:
        target, link = work / n, exp_dir / n
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if Path(os.readlink(link)) == target:
                out[n] = "already linked"
                continue
            link.unlink()
        elif link.exists():
            for item in link.iterdir():
                dst = target / item.name
                shutil.move(str(item), str(dst if not dst.exists() else target / (item.name + ".local")))
            link.rmdir()
        link.symlink_to(target, target_is_directory=True)
        out[n] = "linked -> %s" % target
    return out


def disk_report(paths, need_gb=None):
    """Print free GB for each path; need_gb = {label: gb} estimates (CHANGES.md, 추정). Returns {path: free_gb}."""
    res = {}
    for p in paths:
        q = Path(p)
        while not q.exists() and q != q.parent:
            q = q.parent
        u = shutil.disk_usage(q)
        res[str(p)] = round(u.free / 1e9, 1)
        print("  %-45s free %7.1f GB of %7.1f GB" % (p, u.free / 1e9, u.total / 1e9))
    if need_gb:
        print("  estimates (추정, CHANGES.md 6절): " + ", ".join("%s ~%s GB" % kv for kv in need_gb.items()) + "  -> sum ~%s GB" % sum(need_gb.values()))
    return res


# ------------------------------------------------------------------------------------------ input check (read-only)
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _last_epoch(run_dir):
    p = Path(run_dir) / "epoch_logs.jsonl"
    try:
        return int(json.loads(p.read_text().strip().splitlines()[-1])["epoch"])
    except Exception:
        return None


def check_inputs(work002, seeds, suffixes=("",), expected_epoch=15, expected_cfg_sha=None, cfg002_path=None, need_manifest=False):
    """Read-only check of the 002 artefacts 004/005 depend on. Returns (ok, problems, lines). Never writes."""
    w = Path(work002)
    lines, bad = [], []

    def item(ok, msg, fatal=True):
        lines.append(("[ok]   " if ok else ("[MISSING] " if fatal else "[warn] ")) + msg)
        if not ok and fatal:
            bad.append(msg)

    item(w.is_dir(), "work root exists: %s" % w)
    for s in seeds:
        rd = w / "train_runs" / "outputs" / ("exp002_%s_s%d" % (CONDITION, s))
        item((rd / "DONE").exists(), "%s/DONE" % rd.name)
        item((rd / "hydra.yaml").exists(), "%s/hydra.yaml" % rd.name)
        ck = rd / "checkpoints" / "model_latest.pth"
        item(ck.exists() and ck.stat().st_size > 1e6, "%s/checkpoints/model_latest.pth (%s MB)" % (rd.name, round(ck.stat().st_size / 1e6, 1) if ck.exists() else "-"))
        item((rd / "checkpoints" / ("model_%d.pth" % expected_epoch)).exists(), "%s/checkpoints/model_%d.pth (final-epoch file; kept by prune)" % (rd.name, expected_epoch), fatal=False)
        le = _last_epoch(rd)
        item(le == expected_epoch, "%s last epoch in epoch_logs.jsonl = %s (expected %d)" % (rd.name, le, expected_epoch), fatal=False)
    for sfx in suffixes:
        cd = w / "exp002_cache" / ("q1_tokens" + sfx)
        item((cd / "DONE").exists(), "exp002_cache/q1_tokens%s/DONE" % sfx, fatal=(sfx == ""))
        if (cd / "DONE").exists():
            miss = [f for f in Q1_FILES if not (cd / f).exists()]
            item(not miss, "q1_tokens%s files complete%s" % (sfx, (" (missing %s)" % miss) if miss else ""))
            try:
                m = json.loads((cd / "meta.json").read_text())
                lines.append("        meta: n_episodes=%s source_run=%s epoch=%s fingerprint=%s" % (m.get("n_episodes"), m.get("source_run"), m.get("epoch"), str(m.get("encoder_fingerprint"))[:12]))
            except Exception as e:
                item(False, "q1_tokens%s/meta.json unreadable: %s" % (sfx, e))
    if need_manifest:
        item((w / "manifests" / "cache_episodes.json").exists(), "manifests/cache_episodes.json (needed to build the _smoke cache)")
    if cfg002_path:
        local = sha256_file(cfg002_path)
        exp = expected_cfg_sha
        drive_res = w / "results" / "results.json"
        drive_sha = None
        if drive_res.exists():
            try:
                drive_sha = json.loads(drive_res.read_text()).get("config_sha256")
            except Exception:
                pass
        lines.append("        002 config.yaml sha256 (this code): %s" % local)
        for name, v in (("expected (002/results.json in the repo)", exp), ("Drive exp002_work/results/results.json", drive_sha)):
            if v:
                item(v == local, "002 config fingerprint equals %s: %s" % (name, v[:16]))
            else:
                lines.append("        (no %s to compare)" % name)
    return (not bad), bad, lines


# ------------------------------------------------------------------------------------------ pilot log parsing and cost projection
def parse_enc_probe_units(log_text):
    """[(k, control, cumulative_seconds)] from 'enc_probe k=0 control=real done (12s)'."""
    return [(int(a), b, float(c)) for a, b, c in re.findall(r"enc_probe k=(\d+) control=(\S+) done \((\d+)s\)", log_text)]


def parse_first_heads(log_text):
    """[(tag, seconds, projected_seconds, n_tasks)] from '... 5-A first head: 2.41s -> projected 241s for 100 tasks' and the 5-B equivalent."""
    out = []
    for m in re.finditer(r"(\S+) (5-[AB]) first head: ([\d.]+)s -> projected (\d+)s for (\d+) tasks", log_text):
        out.append({"tag": m.group(1), "sub": m.group(2), "first_head_s": float(m.group(3)), "projected_s": float(m.group(4)), "n_tasks": int(m.group(5))})
    return out


def parse_heads_plan(log_text):
    m = re.search(r"attentive heads \(ours \+ random-init\) = (\d+) ; base heads \(oracle linear/MLP, area\) = (\d+)", log_text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def project_004(pilot_seconds, n_pilot, n_main, n_train_seeds_main=3, n_train_seeds_pilot=1):
    """Linear-in-episodes projection of the pilot stage times to the main run (추정: assumes time ~ number of episodes; fixed overheads make it an overestimate)."""
    r = float(n_main) / float(n_pilot)
    out = {}
    for st in ("enc_cache", "enc_probe", "pred", "aggregate"):
        if st in pilot_seconds:
            f = r * (n_train_seeds_main / float(n_train_seeds_pilot) if st == "pred" else 1.0) if st != "aggregate" else 1.0
            out[st] = round(pilot_seconds[st] * f, 0)
    out["total_s"] = sum(out.values())
    out["ratio_episodes"] = round(r, 3)
    return out


def project_005(first_head_s, n_attn_main):
    """n attentive heads x first-head seconds (the first head includes CUDA warm-up -> upper-leaning 추정; 5-A/5-B only)."""
    return {"heads_total_s": round(first_head_s * n_attn_main, 0), "n_attentive_heads": n_attn_main, "first_head_s": first_head_s}


def hms(sec):
    sec = int(sec)
    return "%dh%02dm" % (sec // 3600, (sec % 3600) // 60)


# ------------------------------------------------------------------------------------------ 004 cache tar (optional, default off in the notebook)
def _safe_member(m):
    parts = m.name.split("/")
    return not m.name.startswith("/") and ".." not in parts and (m.isfile() or m.isdir())


def tar_save(src_dir, dst_tar):
    """Uncompressed tar of src_dir (e.g. EXP004_CACHE/enc) -> dst_tar via .part + rename. Returns bytes."""
    src_dir, dst_tar = Path(src_dir), Path(dst_tar)
    if not (src_dir / "DONE").exists():
        raise SystemExit("%s/DONE missing: the cache is not complete, nothing saved" % src_dir)
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_tar.with_name(dst_tar.name + ".part")
    with tarfile.open(tmp, "w") as tf:
        tf.add(str(src_dir), arcname=src_dir.name)
    os.replace(tmp, dst_tar)
    return dst_tar.stat().st_size


def tar_restore(src_tar, dst_root):
    """Extract a tar made by tar_save into dst_root (refuses absolute paths, '..', links). Skips when dst_root/<name>/DONE already exists."""
    src_tar, dst_root = Path(src_tar), Path(dst_root)
    if not src_tar.exists():
        raise SystemExit("%s not found" % src_tar)
    with tarfile.open(src_tar, "r") as tf:
        mem = tf.getmembers()
        for m in mem:
            if not _safe_member(m):
                raise SystemExit("refused tar member: %s" % m.name)
        top = {m.name.split("/")[0] for m in mem}
        if all((dst_root / t / "DONE").exists() for t in top):
            return "already present: %s" % sorted(top)
        dst_root.mkdir(parents=True, exist_ok=True)
        tf.extractall(dst_root, members=mem)
    return "restored %s into %s" % (sorted(top), dst_root)


# ------------------------------------------------------------------------------------------ final copy info
def sha_table(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if p.exists():
            rows.append((str(p), p.stat().st_size, sha256_file(p)))
        else:
            rows.append((str(p), None, None))
    return rows


def print_sha_table(paths):
    for p, n, h in sha_table(paths):
        print("%-70s %s bytes  sha256 %s" % (p, n if n is not None else "MISSING", h))


def read_yaml_values(path, keys):
    """Read dotted keys from a YAML file (kernel python has PyYAML on Colab; falls back to None per key if yaml is missing)."""
    try:
        import yaml
    except ImportError:
        return {k: "(PyYAML missing)" for k in keys}
    d = yaml.safe_load(Path(path).read_text())
    out = {}
    for k in keys:
        cur = d
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        out[k] = cur
    return out


# ------------------------------------------------------------------------------------------ selftest (CPU)
def selftest():
    n_ok = n_bad = 0

    def check(name, cond):
        nonlocal n_ok, n_bad
        n_ok += bool(cond)
        n_bad += (not cond)
        print("  [%s] %s" % ("ok" if cond else "FAIL", name))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        w2 = td / "exp002_work"
        for s in (0, 1, 2):
            rd = w2 / ("train_runs/outputs/exp002_%s_s%d" % (CONDITION, s))
            (rd / "checkpoints").mkdir(parents=True)
            (rd / "DONE").write_text("ok\n")
            (rd / "hydra.yaml").write_text("a: 1\n")
            (rd / "checkpoints/model_latest.pth").write_bytes(b"0" * 2_000_000)
            (rd / "epoch_logs.jsonl").write_text(json.dumps({"epoch": 15}) + "\n")
        cd = w2 / "exp002_cache/q1_tokens"
        cd.mkdir(parents=True)
        for f in Q1_FILES:
            (cd / f).write_text("{}" if f.endswith("json") else "x")
        (cd / "meta.json").write_text(json.dumps({"n_episodes": 1800, "source_run": "r", "epoch": 15, "encoder_fingerprint": "ab" * 32}))
        cfgp = td / "config.yaml"
        cfgp.write_text("seed: 0\n")
        sha = sha256_file(cfgp)
        ok, bad, _ = check_inputs(w2, [0, 1, 2], cfg002_path=cfgp, expected_cfg_sha=sha)
        check("check_inputs passes on a complete fake 002 work root", ok and not bad)
        ok, bad, _ = check_inputs(w2, [0, 1, 2], cfg002_path=cfgp, expected_cfg_sha="0" * 64)
        check("config fingerprint mismatch is reported", (not ok) and any("fingerprint" in b for b in bad))
        (w2 / ("train_runs/outputs/exp002_%s_s1/DONE" % CONDITION)).unlink()
        ok, bad, _ = check_inputs(w2, [0, 1, 2])
        check("missing DONE is reported", (not ok) and any("s1/DONE" in b for b in bad))
        ok, bad, _ = check_inputs(w2, [0], suffixes=("", "_smoke"))
        check("missing _smoke cache is not fatal", ok)
        before = sorted(str(p) for p in w2.rglob("*"))
        check_inputs(w2, [0, 1, 2], suffixes=("", "_smoke"), need_manifest=True)
        check("check_inputs writes nothing", before == sorted(str(p) for p in w2.rglob("*")))

        ex, w4 = td / "exp", td / "w4"
        (ex / "results").mkdir(parents=True)
        (ex / "results/old.txt").write_text("keep")
        o = link_dirs(w4, ex)
        check("link_dirs links and moves local content", (ex / "results").is_symlink() and (w4 / "results/old.txt").read_text() == "keep" and "logs" in o)
        check("link_dirs idempotent", link_dirs(w4, ex)["results"] == "already linked")

        rec = run_stage(sys.executable, td, "-c", ["print('log file: /x/y.log')"], dict(os.environ), timing_file=td / "t.json", sample_vram=False)
        check("run_stage captures log path and rc (python -u run.py replaced by -c is expected to fail)", rec["rc"] != 0 or rec["log"] == "/x/y.log")
        (td / "run.py").write_text("import sys\nprint('log file: /x/y.log')\nsys.exit(int(sys.argv[2]))\n")
        r0 = run_stage(sys.executable, td, "s", ["0"], dict(os.environ), timing_file=td / "t.json", sample_vram=False)
        r1 = run_stage(sys.executable, td, "s", ["3"], dict(os.environ), timing_file=td / "t.json", sample_vram=False)
        check("run_stage rc, log path, timing file", r0["rc"] == 0 and r1["rc"] == 3 and r1["log"] == "/x/y.log" and len(json.loads((td / "t.json").read_text())) == 3)
        (td / "attempts.json").write_text('{"stage:pred": 3}')
        try:
            require_ok(r1, td)
            check("require_ok stops with log path", False)
        except SystemExit as e:
            check("require_ok stops with log path and attempts", "/x/y.log" in str(e) and "stage:pred" in str(e))

        enc = td / "cache/enc"
        enc.mkdir(parents=True)
        (enc / "tokens_k0.npy").write_bytes(b"1234")
        (enc / "DONE").write_text("ok")
        n = tar_save(enc, td / "tar/enc.tar")
        shutil.rmtree(enc)
        msg = tar_restore(td / "tar/enc.tar", td / "cache")
        check("tar save/restore round trip", n > 0 and (td / "cache/enc/tokens_k0.npy").read_bytes() == b"1234" and msg.startswith("restored"))
        check("tar restore skips when present", tar_restore(td / "tar/enc.tar", td / "cache").startswith("already"))

    check("parse_enc_probe_units", parse_enc_probe_units("x enc_probe k=0 control=real done (12s)\nenc_probe k=0 control=random_init done (30s)") == [(0, "real", 12.0), (0, "random_init", 30.0)])
    fh = parse_first_heads("[t] ibs/ours/s0 5-A first head: 2.41s -> projected 241s for 100 tasks")
    check("parse_first_heads", fh and fh[0]["first_head_s"] == 2.41 and fh[0]["n_tasks"] == 100)
    check("parse_heads_plan", parse_heads_plan("heads plan: layers [] ; seeds [0] ; attentive heads (ours + random-init) = 2040 ; base heads (oracle linear/MLP, area) = 3060 ; x") == (2040, 3060))
    pr = project_004({"enc_cache": 100, "enc_probe": 200, "pred": 300, "aggregate": 5}, 300, 1800)
    check("project_004 scales by episodes and seeds", pr["enc_cache"] == 600 and pr["enc_probe"] == 1200 and pr["pred"] == 5400 and pr["aggregate"] == 5)
    check("project_005", project_005(2.0, 100)["heads_total_s"] == 200)
    check("hms", hms(5400) == "1h30m")
    print("exp045_tools selftest: passed=%d failed=%d" % (n_ok, n_bad))
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    print(__doc__)
