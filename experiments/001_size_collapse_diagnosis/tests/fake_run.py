#!/usr/bin/env python3
"""Fake run.py for tests/test_overnight.sh (no torch, no GPU). Behaviour via env: FAKE_ROOT, FAIL_SPEC="seed:stage:n,..." (fail the first n calls)."""
import json, os, sys, time
from pathlib import Path
stage = sys.argv[1]
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
root = Path(os.environ["FAKE_ROOT"])
ev = root / "events.log"
def event(msg):
    with open(ev, "a") as f:
        f.write(f"{time.time():.2f} {msg}\n")
if model:
    dist, seed = model.split(":"); name = f"exp001_{dist}_s{seed}"
    for item in filter(None, os.environ.get("FAIL_SPEC", "").split(",")):
        s, st, n = item.split(":")
        if s == seed and st == stage:
            c = root / f"cnt_{seed}_{stage}"
            k = int(c.read_text()) if c.exists() else 0
            c.write_text(str(k + 1))
            if k < int(n):
                event(f"{stage} seed{seed} FAIL#{k+1}")
                print("some log line\nTraceback: injected failure in %s seed %s" % (stage, seed), file=sys.stderr)
                sys.exit(3)
event(f"{stage} seed{model.split(':')[1] if model else '-'} start")
time.sleep(float(os.environ.get("FAKE_SLEEP", "1")))
res = Path("results")
if stage == "train":
    d = Path(os.environ["DINO_TRAIN"]) / "outputs" / name; d.mkdir(parents=True, exist_ok=True); (d / "DONE").write_text("ok\n")
elif stage == "repro_check":
    (res / name).mkdir(parents=True, exist_ok=True); (res / name / "repro_check.json").write_text("{}")
elif stage == "extract":
    d = Path(os.environ["EXP001_CACHE"]) / name; d.mkdir(parents=True, exist_ok=True); (d / "DONE").write_text("ok\n")
elif stage in ("diagnose", "probe", "intervene"):
    (res / name).mkdir(parents=True, exist_ok=True); (res / name / f"{stage}.json").write_text("{}")
elif stage == "aggregate":
    Path("results.json").write_text(json.dumps({"fake": True}))
elif stage == "sync":
    res.mkdir(exist_ok=True); (res / "sync_status.json").write_text('{"status": "skipped"}')
elif stage == "listing":
    Path("manifests").mkdir(exist_ok=True); Path("manifests/train_baseline_single.json").write_text("{}")
event(f"{stage} seed{model.split(':')[1] if model else '-'} end")
