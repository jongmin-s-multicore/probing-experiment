#!/usr/bin/env python
"""Experiment 009 entry point (positive control for 004's depth-truncation pipeline).   python run.py <stage> [options]
Stages: check | intervene | aggregate | selftest      (every stage supports --dry-run; NOT RUN on a GPU: approval pending)
intervene: for each (block m_int, strength, train seed) unit: erase the size direction from the predictor stream after block m_int, forward the later blocks, re-probe.
           Strength 0 (no intervention) is computed once per train seed and shared by all blocks. Finished units are skipped (resume).
aggregate: per-block detection (004 rule) and success verdict -> results.json (records m_int_config / m_int_used / m_int_overridden)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp009 import base as B  # noqa: E402
from exp009.base import log, set_seed, ConfigNull  # noqa: E402
from exp002.common import start_file_log, log_exception, attempts_record, warn_conda  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "intervene", "aggregate", "selftest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--m-int", default=None, help="comma separated blocks (1..L), overrides config intervention.m_int (also env EXP009_M_INT); the override is recorded in results.json")
    ap.add_argument("--strengths", default=None, help="comma separated, overrides config intervention.strengths")
    ap.add_argument("--train-seeds", default=None, help="comma separated 002 training seeds (default config predictor.train_seeds)")
    ap.add_argument("--fresh", action="store_true", help="redo units even if their part file exists")
    args = ap.parse_args()
    cfg = B.load_config(args.config)
    c4 = B.make_cfg004(cfg)
    B.check_consistency_with_004(cfg, c4)
    cfg2 = B.B4.load_cfg002(c4)
    paths = B.resolve_paths(cfg, c4, cfg2)
    if args.results_dir:
        paths["results_dir"] = Path(args.results_dir).resolve()
    set_seed(cfg["seed"])
    start_file_log(paths, args.stage + ("_dry" if args.dry_run else ""))
    log("stage=%s dry_run=%s python=%s conda_env=%s pilot=%s" % (args.stage, args.dry_run, sys.executable, os.environ.get("CONDA_DEFAULT_ENV"), cfg["pilot"]))
    if args.stage != "selftest":
        warn_conda()
    from exp009 import stages as S
    m_ints, overridden, source = S.resolve_m_int(cfg, args.m_int)
    strengths = [float(x) for x in args.strengths.split(",")] if args.strengths else [float(x) for x in cfg["intervention"]["strengths"]]
    tseeds = [int(x) for x in args.train_seeds.split(",")] if args.train_seeds else list(cfg["predictor"]["train_seeds"])
    if 0.0 not in strengths:
        strengths = [0.0] + strengths
    key = "stage:" + args.stage
    guard = args.stage not in ("check", "selftest") and not args.dry_run
    if args.stage == "intervene":
        bad = S.metadata_preflight(c4, paths, tseeds)      # before attempts_check / attempts_record: a preflight error never consumes an attempt
        if bad:
            msg = "PREFLIGHT ERROR (no attempt consumed): metadata.pt missing for the training manifests (%d problems). Run the Colab data cell (metadata.pt only; no mp4 needed). First: %s" % (len(bad), "; ".join(bad[:5]))
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
            rc = S.check(cfg, c4, cfg2, paths, tseeds)
        elif args.stage == "intervene":
            S.intervene_stage(cfg, c4, cfg2, paths, m_ints, strengths, tseeds, fresh=args.fresh, dry_run=args.dry_run)
        elif args.stage == "aggregate":
            S.aggregate(cfg, c4, cfg2, paths, m_ints, overridden, source, strengths, tseeds, dry_run=args.dry_run)
        elif args.stage == "selftest":
            from exp009 import selftest
            rc = selftest.run(cfg, c4, cfg2)
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
