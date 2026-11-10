#!/usr/bin/env python
"""Experiment 010 entry point (source separation of the 005 ours ~ random result).   python run.py <stage> [options]
Stages: check | heads | report | selftest      (every stage supports --dry-run; nothing here has been run on a GPU; approval pending)
heads: 5-A heads for four representations (enc_k12 ours/random, pred_out6 ours/random) x task seeds x head seeds, 005 code re-used unmodified.
report: 005 metrics per task seed + H-f TOST (Delta from config) -> results.json, ib_scores.csv."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp010 import base as B  # noqa: E402
from exp010.base import log, set_seed, ConfigNull  # noqa: E402
from exp002.common import start_file_log, log_exception, attempts_record, warn_conda  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "heads", "report", "selftest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--task-seeds", default=None, help="comma separated (default config seeds.task_a)")
    ap.add_argument("--seeds", default=None, help="comma separated head seeds (default config seeds.head)")
    ap.add_argument("--max-tasks", type=int, default=None, help="heads: only the first N tasks (timing / pilot)")
    args = ap.parse_args()
    cfg = B.load_config(args.config)
    c5 = B.make_cfg005(cfg)
    cfg["_c5"] = c5
    c2 = B.load_cfg002(c5)
    paths = B.resolve_paths(cfg, c5, c2)
    if args.results_dir:
        paths["results_dir"] = Path(args.results_dir).resolve()
    set_seed(cfg["seed"])
    start_file_log(paths, args.stage + ("_dry" if args.dry_run else ""))
    log("stage=%s dry_run=%s python=%s conda_env=%s pilot=%s" % (args.stage, args.dry_run, sys.executable, os.environ.get("CONDA_DEFAULT_ENV"), cfg["pilot"]))
    if args.stage != "selftest":
        warn_conda()
    from exp010 import stages as S
    tseeds = [int(x) for x in args.task_seeds.split(",")] if args.task_seeds else list(cfg["seeds"]["task_a"])
    hseeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(cfg["seeds"]["head"])
    key = "stage:" + args.stage
    guard = args.stage not in ("check", "selftest") and not args.dry_run
    try:
        if guard:
            B.attempts_check(cfg, paths, key)
        rc = 0
        if args.stage == "check":
            rc = S.check(cfg, c2, paths)
        elif args.stage == "heads":
            if len(tseeds) < 3 and not args.dry_run:
                log("WARNING: fewer than 3 task seeds -> the TOST will be not_evaluable")
            S.heads_stage(cfg, c2, paths, tseeds, hseeds, args.max_tasks, dry_run=args.dry_run)
        elif args.stage == "report":
            S.report_stage(cfg, c2, paths, hseeds, dry_run=args.dry_run)
        elif args.stage == "selftest":
            from exp010 import selftest
            rc = selftest.run(cfg, c2)
        if guard:
            attempts_record(paths, key, rc == 0)
    except ConfigNull as e:
        if args.dry_run:
            log("DRY-RUN: a real run of this stage would STOP here: %s" % e)
            sys.exit(0)
        log_exception(e)
        sys.exit(3)
    except Exception as e:
        log_exception(e)
        if guard:
            attempts_record(paths, key, False)
        print("\nFAILED. Paste the log file shown at the top (logs/).", file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
