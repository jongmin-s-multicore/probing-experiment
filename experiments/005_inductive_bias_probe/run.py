#!/usr/bin/env python
"""Experiment 005 entry point.   python run.py <stage> [options]

Stages (run in this order):
  plan      read experiment 4's summary.json and choose layers / sub-experiments by the 4.6 branch table -> results/plan.json
  tasks     evaluation inputs (2000) + 5-A/5-B/5-D tasks -> results/tasks.npz            (CPU, needs the 002 Q1 cache)
  feat      predictor-layer features (real + random-init predictor) from the 002 checkpoint (GPU); encoder layers come from the 004 cache
  heads     train all heads, save per-task predictions -> results/predictions/*.npz        (GPU for the token heads)
  metrics   ib_scores.csv, calibration*.csv, law_fits.csv, figures from the saved predictions (CPU)
  report    verdicts (supported/refuted/not_evaluable) -> REPORT.md, results.json
  selftest  synthetic-data CPU self-test (not a result)
Every stage supports --dry-run. LoRA adaptation is NOT implemented (user decision 2026-11-03, CHANGES.md)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp005 import base as B  # noqa: E402
from exp005.base import log, set_seed, ConfigNull  # noqa: E402
from exp002.common import start_file_log, log_exception, attempts_record, warn_conda  # noqa: E402
from exp005.base import attempts_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "plan", "tasks", "feat", "heads", "metrics", "report", "selftest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", default=None, help="plan: path of experiment 4's summary.json (default config exp4.summary)")
    ap.add_argument("--suffix", default=None, help="cache suffix of partial pilot caches (default config exp4.cache_suffix)")
    ap.add_argument("--seeds", default=None, help="heads/metrics: comma separated head seeds (default config seeds.head)")
    ap.add_argument("--layers", default=None, help="heads: comma separated layer names to run (default: all in plan.json)")
    ap.add_argument("--max-tasks", type=int, default=None, help="heads: only the first N tasks of 5-A and of 5-B (timing / pilot); outputs are marked in tasks only by shape")
    args = ap.parse_args()
    cfg = B.load_config(args.config)
    cfg2 = B.load_cfg002(cfg)
    paths = B.resolve_paths(cfg, cfg2)
    if args.results_dir:
        paths["results_dir"] = Path(args.results_dir).resolve()
    set_seed(cfg["seed"])
    start_file_log(paths, args.stage + ("_dry" if args.dry_run else ""))
    log("stage=%s dry_run=%s python=%s conda_env=%s pilot=%s" % (args.stage, args.dry_run, sys.executable, os.environ.get("CONDA_DEFAULT_ENV"), cfg["pilot"]))
    if args.stage != "selftest":
        warn_conda()
    from exp005 import stages as S
    suffix = cfg["exp4"]["cache_suffix"] if args.suffix is None else args.suffix
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(cfg["seeds"]["head"])
    key = "stage:" + args.stage
    guard = args.stage not in ("check", "selftest") and not args.dry_run
    try:
        if guard:
            attempts_check(cfg, paths, key)
        rc = 0
        if args.stage == "check":
            rc = S.check(cfg, cfg2, paths)
        elif args.stage == "plan":
            S.plan_stage(cfg, paths, args.summary, dry_run=args.dry_run)
        elif args.stage == "selftest":
            from exp005 import selftest
            rc = selftest.run(cfg, cfg2)
        else:
            if args.dry_run and not (paths["results_dir"] / "plan.json").exists():
                plan = S.plan_stage(cfg, paths, args.summary, dry_run=True)                 # plan from the summary if it exists, nothing written
            else:
                plan = S.read_plan(paths)
            if args.stage == "tasks":
                S.tasks_stage(cfg, cfg2, paths, suffix, dry_run=args.dry_run)
            elif args.stage == "feat":
                S.feat_stage(cfg, cfg2, paths, plan, suffix, dry_run=args.dry_run)
            elif args.stage == "heads":
                S.heads_stage(cfg, cfg2, paths, plan, seeds, suffix, args.max_tasks, (args.layers.split(",") if args.layers else None), dry_run=args.dry_run)
            elif args.stage == "metrics":
                S.metrics_stage(cfg, cfg2, paths, plan, seeds, dry_run=args.dry_run)
            elif args.stage == "report":
                S.report_stage(cfg, cfg2, paths, plan, dry_run=args.dry_run)
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
