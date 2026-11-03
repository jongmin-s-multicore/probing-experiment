#!/usr/bin/env python
"""Experiment 004 entry point.   python run.py <stage> [options]

Stages: check | enc_cache | enc_probe | pred | aggregate | selftest
  enc_cache  4-A  GPU : truncated-encoder patch tokens for every k of encoder.k_grid (+ random-init control, raw-pixel tokens) from the 002 Q1 cache frames
  enc_probe  4-A  CPU/GPU-light : probes per k (linear/attentive x 5 seeds, controls) -> results/parts/encoder.csv
  pred       4-B  GPU : early-exit extraction from 002 condition-(b) checkpoints, hidden probes (i), exp1-probe on early-exit z (ii), next-z MSE (iii)
  aggregate  results/depth_r2.csv, summary.json, next_z_mse.csv, figures, results.json
Every stage supports --dry-run (prints the plan only). Seeds are in config.yaml. Logs go to logs/. Run inside the activated conda env of 002.
4-C (depth retraining) is NOT implemented (user decision 2026-11-03, CHANGES.md)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp004 import base as B  # noqa: E402  (also puts 002's src on sys.path)
from exp004.base import log, set_seed, ConfigNull  # noqa: E402
from exp002.common import start_file_log, log_exception, attempts_check, attempts_record, warn_conda  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "enc_cache", "enc_probe", "pred", "aggregate", "selftest"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--suffix", default="", help="002 Q1 cache suffix (e.g. _smoke for a partial pilot cache); also used for the 004 caches")
    ap.add_argument("--train-seeds", default=None, help="pred: comma separated 002 training seeds (default config predictor.train_seeds)")
    ap.add_argument("--controls", default="real,random_init", help="pred: comma separated: real,random_init")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results/parts rows and redo")
    ap.add_argument("--force", action="store_true", help="enc_cache: overwrite an existing cache")
    ap.add_argument("--keep-tmp", action="store_true")
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
    from exp004 import stages as S
    key = "stage:" + args.stage
    guard = args.stage not in ("check", "selftest") and not args.dry_run
    try:
        if guard:
            attempts_check(cfg, paths, key)
        rc = 0
        if args.stage == "check":
            rc = S.check(cfg, cfg2, paths)
        elif args.stage == "enc_cache":
            S.enc_cache(cfg, cfg2, paths, suffix=args.suffix, overwrite=args.force, dry_run=args.dry_run)
        elif args.stage == "enc_probe":
            S.enc_probe(cfg, cfg2, paths, suffix=args.suffix, fresh=args.fresh, dry_run=args.dry_run)
        elif args.stage == "pred":
            ts = [int(x) for x in args.train_seeds.split(",")] if args.train_seeds else list(cfg["predictor"]["train_seeds"])
            ctl = [c for c in args.controls.split(",") if c]
            if "random_init" in ctl and not cfg["predictor"]["random_init_control"]:
                ctl.remove("random_init")
            S.pred_stage(cfg, cfg2, paths, ts, ctl, suffix=args.suffix, fresh=args.fresh, dry_run=args.dry_run, keep_tmp=args.keep_tmp)
        elif args.stage == "aggregate":
            S.aggregate(cfg, cfg2, paths, dry_run=args.dry_run)
        elif args.stage == "selftest":
            from exp004 import selftest
            rc = selftest.run(cfg, cfg2)
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
