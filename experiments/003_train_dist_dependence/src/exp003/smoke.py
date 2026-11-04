"""Smoke measurement (NOT run on the authoring machine: no GPU). Measures what the 12 GB question needs:
  1. a 1-epoch training run on n_episodes episodes of one distribution (same batch size / hydra settings as the real run),
     while polling nvidia-smi for peak memory used on GPU 0 (whole-GPU number, includes other processes) -> train_peak_mib_nvidia_smi
  2. wall time of that run -> seconds per episode-epoch; an extrapolation to the real run is stored, marked 'estimate'
  3. a small extract (extract_episodes_per_size episodes x 9 sizes) with torch peak allocated memory (extract.py) and wall time
Writes results/smoke.json. Training of the real runs is a separate stage and needs explicit user approval (CLAUDE.md)."""
from __future__ import annotations

import subprocess
import threading
import time

from .common import log, write_json, run_dir


def _poll(stop, out, interval):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                               capture_output=True, text=True, timeout=10)
            used, total = [int(x) for x in r.stdout.strip().splitlines()[0].split(",")]
            out["peak_used_mib"] = max(out.get("peak_used_mib", 0), used)
            out["total_mib"] = total
        except Exception as e:                       # nvidia-smi missing or failing: recorded, never hidden
            out["poll_error"] = "%s: %s" % (type(e).__name__, e)
        stop.wait(interval)


def run(cfg, paths, train_fn, extract_fn, dry_run=False):
    sm = cfg["smoke"]
    out = {"distribution": sm["distribution"], "seed": sm["seed"], "n_episodes": sm["n_episodes"], "epochs": sm["epochs"],
           "batch_size": cfg["train"]["batch_size"], "status": "dry_run" if dry_run else "started"}
    log("smoke: %s" % out)
    if dry_run:
        return out
    poll, stop = {}, threading.Event()
    th = threading.Thread(target=_poll, args=(stop, poll, sm["poll_interval_s"]), daemon=True)
    th.start()
    t0 = time.time()
    rc = train_fn()
    t_train = time.time() - t0
    stop.set()
    th.join(timeout=15)
    out.update({"train_returncode": rc, "train_seconds": round(t_train, 1), "train_peak_mib_nvidia_smi": poll.get("peak_used_mib"),
                "gpu_total_mib": poll.get("total_mib"), "poll_error": poll.get("poll_error")})
    if rc == 0:
        eps_real = cfg["train"]["episodes_per_distribution"]
        # estimate: time scales with the number of episodes x epochs (includes start-up cost, so this overestimates; UNVERIFIED)
        out["estimate_real_train_seconds_per_run"] = round(t_train / sm["n_episodes"] * eps_real * cfg["train"]["epochs"] / sm["epochs"], 0)
        out["estimate_note"] = "linear scaling of the smoke wall time (episodes x epochs) including start-up time; estimate, not measured"
        t1 = time.time()
        extract_fn()
        out["extract_seconds"] = round(time.time() - t1, 1)
        n_eval = len(cfg["_ds"]["eval_sets"]) * sm["extract_episodes_per_size"]
        out["extract_episodes"] = n_eval
        out["estimate_real_extract_seconds_per_run"] = round(out["extract_seconds"] / n_eval * len(cfg["_ds"]["eval_sets"]) * cfg["eval"]["n_test_episodes_per_size"], 0)
    out["status"] = "finished" if rc == 0 else "failed"
    write_json(out, paths["results_dir"] / "smoke.json")
    log("smoke: wrote results/smoke.json")
    return out
