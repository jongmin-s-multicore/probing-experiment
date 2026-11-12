#!/usr/bin/env python
"""Experiment 015 entry point (size interchange: does swapping in another level's mean move the predicted motion toward that level?).   python run.py <stage> [options]
Stages: check | fit | measure | aggregate | selftest      (every stage supports --dry-run; the GPU stage `measure` is NOT RUN: approval pending)
fit      : (CPU) level means m_L, level-mean true motion, d per directed pair, gate (mean split-half cosine over 20 splits >= 0.5), fake-swap shifts (unscaled; norm-matching is per pair at measure), from the TRAIN-split episodes of the Q1 cache.
measure  : (GPU) per (condition, train seed) unit: moved for the real swap, 20 norm-matched fake swaps (primary comparator), 20 unscaled fake swaps (report only), 20 random directions; refused (exit 4) if fit found 0 gated pairs unless --allow-no-gated-pairs; trained and random-init predictor; test episodes of the source level.
aggregate: (CPU) per-pair statistics, episode bootstrap, verdict (condition (b), gated pairs), results.json (interpretation stays null).
Exit codes: 0 ok, 1 failure (counts as an attempt, max 3 in <results-dir>/attempts.json), 3 config value null, 4 PREFLIGHT failure (inputs or config; no attempt consumed)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp015 import base as B  # noqa: E402
from exp015.base import log, set_seed, ConfigNull  # noqa: E402
from exp002.common import start_file_log, log_exception, attempts_record, warn_conda  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "fit", "measure", "aggregate", "selftest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None, help="separate results folder (pilot). results.json is then written there only and marked pilot: true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--conditions", default=None, help="comma separated subset of config base.conditions; recorded (pilot: true)")
    ap.add_argument("--train-seeds", default=None, help="comma separated 002 training seeds (default config predictor.train_seeds); recorded (pilot: true)")
    ap.add_argument("--episodes-per-level", type=int, default=None, help="evaluate only the first N test episodes of each source level (pilot); recorded (pilot: true)")
    ap.add_argument("--batch-episodes", type=int, default=8)
    ap.add_argument("--allow-no-gated-pairs", action="store_true", help="run measure even if fit found 0 pairs passing the d-reliability gate (default: refused with exit code 4, user decision 2026-11-12)")
    ap.add_argument("--fresh", action="store_true", help="redo units even if their part file exists")
    args = ap.parse_args()
    cfg = B.load_config(args.config)
    c4, c9, cfg13 = B.make_cfg004(cfg)
    cfg2 = B.B4.load_cfg002(c4)
    paths = B.resolve_paths(cfg, c4, cfg2)
    if args.results_dir:
        paths["results_dir"] = Path(args.results_dir).resolve()
    set_seed(cfg["seed"])
    start_file_log(paths, args.stage + ("_dry" if args.dry_run else ""))
    log("stage=%s dry_run=%s python=%s conda_env=%s pilot_config_flag=%s results_dir=%s" % (args.stage, args.dry_run, sys.executable, os.environ.get("CONDA_DEFAULT_ENV"), cfg["pilot"], paths["results_dir"]))
    if args.stage != "selftest":
        warn_conda()
    from exp015 import stages as S
    cfg_conds = list(cfg["base"]["conditions"])
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()] if args.conditions else cfg_conds
    bad_c = [c for c in conds if c not in cfg_conds]
    if bad_c:
        print("unknown condition(s) %s (config: %s)" % (bad_c, cfg_conds), file=sys.stderr)
        sys.exit(4)
    cfg_seeds = [int(x) for x in cfg["predictor"]["train_seeds"]]
    seeds = [int(x) for x in args.train_seeds.split(",")] if args.train_seeds else cfg_seeds
    conds_over, seeds_over = conds != cfg_conds, sorted(seeds) != sorted(cfg_seeds)
    key = "stage:" + args.stage
    guard = args.stage not in ("check", "selftest") and not args.dry_run
    if args.stage in ("measure", "aggregate", "fit"):
        # PREFLIGHT (before attempts_check / attempts_record: never consumes an attempt)
        bad = ["config: %s (%s)" % (n, d) for n, ok, d in B.consistency_checks(cfg, cfg13, cfg2) if not ok]
        if args.stage == "measure":
            bad += S.input_preflight(paths, conds, seeds, need_fit=True)
            if not args.allow_no_gated_pairs:
                bad += S.gate_problems(paths)
        elif args.stage == "fit":
            qd = S.CA.cache_dir(B.B4.as002(paths), "") / "DONE"
            if not qd.exists():
                bad.append("002 Q1 cache DONE missing (%s)" % qd)
        if bad:
            msg = "PREFLIGHT ERROR (no attempt consumed): %d problems. First: %s" % (len(bad), "; ".join(bad[:5]))
            log(msg)
            if args.dry_run:
                log("DRY-RUN: continuing the dry run although the preflight failed")
            else:
                print(msg, file=sys.stderr)
                sys.exit(B.EXIT_PREFLIGHT)
    try:
        if guard:
            B.attempts_check(cfg, paths, key)
        rc = 0
        if args.stage == "check":
            rc = S.check(cfg, cfg13, cfg2, paths, conds, seeds)
        elif args.stage == "fit":
            S.fit_stage(cfg, cfg2, paths, dry_run=args.dry_run, fresh=args.fresh)
        elif args.stage == "measure":
            S.measure_stage(cfg, cfg2, paths, conds, seeds, args.episodes_per_level, args.batch_episodes, fresh=args.fresh, dry_run=args.dry_run)
        elif args.stage == "aggregate":
            S.aggregate(cfg, cfg13, cfg2, paths, conds, seeds, conds_over, seeds_over, args.episodes_per_level, dry_run=args.dry_run)
        elif args.stage == "selftest":
            from exp015 import selftest
            rc = selftest.run(cfg, cfg13, cfg2, c9)
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
