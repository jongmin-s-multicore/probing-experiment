#!/usr/bin/env python3
"""Helper of overnight.sh: keeps logs/overnight_status.json (stdlib only, file-locked, atomic writes). No interpretation, only states/times/errors.

  overnight_status.py init  --file F --seeds "0 1 2" [--meta k=v ...]
  overnight_status.py set   --file F SEED STAGE STATE [--retries N] [--error TEXT] [--note TEXT]
  overnight_status.py meta  --file F k=v ...
  overnight_status.py summary --file F      (prints the text summary)
States: pending | running | done | failed | skipped (skipped = not run because an earlier stage of that seed failed)
"""
import argparse, fcntl, json, os, sys, time
from datetime import datetime

STAGES = ["train", "repro_check", "extract", "diagnose", "probe", "intervene", "aggregate_sync"]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Locked:
    def __init__(self, path, write=True):
        self.path = path
        self.write = write
    def __enter__(self):
        self.lf = open(self.path + ".lock", "w")
        fcntl.flock(self.lf, fcntl.LOCK_EX)
        try:
            with open(self.path) as f:
                self.d = json.load(f)
        except Exception:
            self.d = {"seeds": {}, "meta": {}}
        return self.d
    def __exit__(self, *a):
        if a[0] is None and self.write:
            self.d["updated"] = now()
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.d, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        self.lf.close()


def seconds(a, b):
    try:
        return int((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())
    except Exception:
        return None


def hms(s):
    return "-" if s is None else f"{s // 3600}h{(s % 3600) // 60:02d}m"


def kv(items):
    return dict(i.split("=", 1) for i in items or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "set", "meta", "summary"])
    ap.add_argument("--file", required=True)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--meta", nargs="*")
    ap.add_argument("--retries", type=int)
    ap.add_argument("--error")
    ap.add_argument("--note")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    if a.cmd == "init":
        with Locked(a.file) as d:
            d.setdefault("started", now())
            d["last_start"] = now()
            d.setdefault("meta", {}).update(kv(a.meta))
            for s in a.seeds.split():
                sd = d["seeds"].setdefault(s, {"stages": {}})
                for st in STAGES:
                    sd["stages"].setdefault(st, {"state": "pending", "start": None, "end": None, "retries": 0, "last_error": None, "note": None})
    elif a.cmd == "meta":
        with Locked(a.file) as d:
            d.setdefault("meta", {}).update(kv(a.rest))
    elif a.cmd == "set":
        seed, stage, state = a.rest
        with Locked(a.file) as d:
            st = d["seeds"].setdefault(seed, {"stages": {}})["stages"].setdefault(stage, {"state": "pending", "start": None, "end": None, "retries": 0, "last_error": None, "note": None})
            if state == "running" and st["state"] != "running":
                st["start"] = st["start"] or now()
                st["end"] = None
            if state in ("done", "failed", "skipped"):
                st["end"] = now()
            st["state"] = state
            if a.retries is not None:
                st["retries"] = a.retries
            if a.error is not None:
                st["last_error"] = a.error
            if a.note is not None:
                st["note"] = a.note
    else:
        with Locked(a.file, write=False) as d:
            L = [f"overnight summary  (status file updated {d.get('updated')}; first start {d.get('started')}; last start {d.get('last_start')})", ""]
            L.append("meta: " + json.dumps(d.get("meta", {}), ensure_ascii=False))
            for s, sd in sorted(d["seeds"].items()):
                stg = sd["stages"]
                reached = [k for k in STAGES if stg.get(k, {}).get("state") == "done"]
                L += ["", f"seed {s}: stages done = {', '.join(reached) or '(none)'}"]
                for k in STAGES:
                    x = stg.get(k)
                    if not x:
                        continue
                    L.append(f"  {k:15s} {x['state']:8s} retries={x['retries']} start={x['start']} end={x['end']} elapsed={hms(seconds(x['start'], x['end'])) if x['end'] else '-'}"
                             + (f"  note={x['note']}" if x.get("note") else "") + (f"\n      last_error: {x['last_error']}" if x.get("last_error") else ""))
            L += ["", "Numbers only: see results.json (and results.json -> across_seeds). Not judged here."]
            print("\n".join(L))


if __name__ == "__main__":
    main()
