#!/usr/bin/env python
"""Experiment 012 entry point (residual information after the 009 scalar-LEACE erasure).   python run.py <stage> [options]
Stages: check | measure | aggregate | selftest      (every stage supports --dry-run; NOT RUN on a GPU: approval pending)
measure  : per (train seed) a shared strength-0 baseline unit (depths needed by the requested blocks) and per (train seed, block m_int) an erased unit
           (strength 1.0, depth m_int and m_int+1): (a) level logistic accuracy, (b) MLP / attentive R2, (c) action-coordinate-only R2, (d) linear R2.
           Probes are trained only on train-split episodes NOT used for the eraser direction and evaluated on test episodes. Finished units are skipped (resume).
aggregate: per-block verdict (config success block) and results.json (interpretation stays null).
Exit codes: 0 ok, 1 failure (counts as an attempt, max 3 in <results-dir>/attempts.json), 3 config value null, 4 PREFLIGHT failure (inputs or config; no attempt consumed)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp012 import base as B  # noqa: E402
from exp012.base import log, set_seed, ConfigNull  # noqa: E402
from exp002.common import start_file_log, log_exception, attempts_record, warn_conda  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "measure", "aggregate", "selftest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None, help="separate results folder (pilot). results.json is then written there only and marked pilot: true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--m-int", default=None, help="comma separated blocks (1..L), overrides config intervention.m_int; recorded in results.json (pilot: true)")
    ap.add_argument("--train-seeds", default=None, help="comma separated 002 training seeds (default config predictor.train_seeds); recorded in results.json (pilot: true)")
    ap.add_argument("--fresh", action="store_true", help="redo units even if their part file exists")
    args = ap.parse_args()
    cfg = B.load_config(args.config)
    c4, c9 = B.make_cfg004(cfg)
    cfg2 = B.B4.load_cfg002(c4)
    paths = B.resolve_paths(cfg, c4, cfg2)
    if args.results_dir:
        paths["results_dir"] = Path(args.results_dir).resolve()
    set_seed(cfg["seed"])
    start_file_log(paths, args.stage + ("_dry" if args.dry_run else ""))
    log("stage=%s dry_run=%s python=%s conda_env=%s pilot_config_flag=%s results_dir=%s" % (args.stage, args.dry_run, sys.executable, os.environ.get("CONDA_DEFAULT_ENV"), cfg["pilot"], paths["results_dir"]))
    if args.stage != "selftest":
        warn_conda()
    from exp012 import stages as S
    m_ints, m_over, source = S.resolve_m_int(cfg, args.m_int)
    cfg_seeds = [int(x) for x in cfg["predictor"]["train_seeds"]]
    tseeds = [int(x) for x in args.train_seeds.split(",")] if args.train_seeds else cfg_seeds
    seeds_over = sorted(tseeds) != sorted(cfg_seeds)
    key = "stage:" + args.stage
    guard = args.stage not in ("check", "selftest") and not args.dry_run
    if args.stage in ("measure", "aggregate"):
        # PREFLIGHT (before attempts_check / attempts_record: a preflight failure never consumes an attempt)
        bad = ["config: %s (%s)" % (n, d) for n, ok, d in B.consistency_checks(cfg, c9, cfg2) if not ok]
        if args.stage == "measure":
            bad += S.input_preflight(c4, paths, tseeds)
        if bad:
            msg = "PREFLIGHT ERROR (no attempt consumed): %d problems. First: %s" % (len(bad), "; ".join(bad[:5]))
            log(msg)
            if args.dry_run:
                log("DRY-RUN: continuing the dry run although the preflight failed")
            else:
                print(msg, file=sys.stderr)
                sys.exit(4)
    try:
        if guard:
            B.attempts_check(cfg, paths, key)
        rc = 0
        if args.stage == "check":
            rc = S.check(cfg, c4, cfg2, paths, tseeds, c9)
        elif args.stage == "measure":
            S.measure_stage(cfg, c4, cfg2, paths, m_ints, tseeds, fresh=args.fresh, dry_run=args.dry_run)
        elif args.stage == "aggregate":
            S.aggregate(cfg, c4, cfg2, paths, m_ints, m_over, source, tseeds, seeds_over, c9, dry_run=args.dry_run)
        elif args.stage == "selftest":
            from exp012 import selftest
            rc = selftest.run(cfg, c4, cfg2, c9)
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
