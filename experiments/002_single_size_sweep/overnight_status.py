#!/usr/bin/env python3
"""Helper of overnight.sh: logs/overnight_status.json (stdlib only, file-locked, atomic writes). States/times/errors only, no interpretation.
Adapted from experiments/001 overnight_status.py (uses parse_intermixed_args: options may come before/after the positional ITEM STAGE STATE).
  overnight_status.py init    --file F --items "run1 run2" [--meta k=v ...]
  overnight_status.py set     --file F ITEM STAGE STATE [--retries N] [--error TEXT] [--note TEXT]
  overnight_status.py summary --file F
Errors are NOT swallowed: any exception is printed to stderr and exits 1 (the caller logs stderr)."""
import argparse, fcntl, json, os, sys
from datetime import datetime

STAGES = ["train", "cache", "q1", "q2", "q3", "aggregate"]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Locked:
    def __init__(self, path, write=True):
        self.path, self.write = path, write

    def __enter__(self):
        self.lf = open(self.path + ".lock", "w")
        fcntl.flock(self.lf, fcntl.LOCK_EX)
        try:
            with open(self.path) as f:
                self.d = json.load(f)
        except FileNotFoundError:
            self.d = {"items": {}, "meta": {}}
        return self.d

    def __exit__(self, *a):
        if a[0] is None and self.write:
            self.d["updated"] = now()
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.d, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        self.lf.close()


def new_stage():
    return {"state": "pending", "start": None, "end": None, "retries": 0, "last_error": None, "note": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "set", "summary"])
    ap.add_argument("--file", required=True)
    ap.add_argument("--items", default="")
    ap.add_argument("--meta", nargs="*")
    ap.add_argument("--retries", type=int)
    ap.add_argument("--error")
    ap.add_argument("--note")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_intermixed_args()
    if a.cmd == "init":
        with Locked(a.file) as d:
            d.setdefault("started", now())
            d["last_start"] = now()
            d.setdefault("meta", {}).update(dict(i.split("=", 1) for i in (a.meta or [])))
            for it in a.items.split():
                for st in STAGES:
                    d["items"].setdefault(it, {}).setdefault(st, new_stage())
    elif a.cmd == "set":
        item, stage, state = a.rest
        with Locked(a.file) as d:
            st = d["items"].setdefault(item, {}).setdefault(stage, new_stage())
            if state == "running" and st["state"] != "running":
                st["start"], st["end"] = st["start"] or now(), None
            if state in ("done", "failed", "skipped"):
                st["end"] = now()
            st["state"] = state
            for k in ("retries", "error", "note"):
                v = getattr(a, k)
                if v is not None:
                    st["last_error" if k == "error" else k] = v
    else:
        with Locked(a.file, write=False) as d:
            L = ["overnight summary (updated %s, first start %s)" % (d.get("updated"), d.get("started")), "meta: " + json.dumps(d.get("meta", {}))]
            for it, stg in sorted(d["items"].items()):
                L.append("")
                L.append("%s" % it)
                for k in STAGES:
                    x = stg.get(k)
                    if x:
                        L.append("  %-10s %-8s retries=%s start=%s end=%s" % (k, x["state"], x["retries"], x["start"], x["end"]) + (("\n      last_error: " + x["last_error"]) if x.get("last_error") else ""))
            print("\n".join(L))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("overnight_status.py ERROR: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        sys.exit(1)
