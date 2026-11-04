#!/usr/bin/env python
"""Experiment 002 entry point.   python run.py <stage> [options]      (see README.md for the test order)

Stages: check | download | manifest | smoke | cache | train | q1 | q2 | q3 | aggregate | selftest
Every stage supports --dry-run (prints what it would do; heavy work is not started). Seeds come from config.yaml.
Each stage writes logs/<stage>[_<tag>]_<time>.log (stdout+stderr, tracebacks included). Failures are counted per stage in
results/attempts.json; after run.max_attempts (3) consecutive failures the stage refuses to run (CLAUDE.md retry limit).
Run inside the activated conda environment (see environment.yml / README)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp002.common import (load_config, resolve_paths, run_name, run_dir, write_json, read_json, log, set_seed, start_file_log, log_exception,  # noqa: E402
                           attempts_check, attempts_record, env_info, warn_conda, disk_free_gb, require, ConfigNull)


def conds(cfg, args):
    if args.condition:
        return [args.condition]
    if not args.all_conditions:
        return list(cfg["conditions"]["run_order"][:1])
    opt = set(cfg["conditions"].get("optional") or [])
    return [c for c in cfg["conditions"]["run_order"] if args.include_optional or c not in opt]


def seeds(cfg, args):
    return [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["train"]["seeds"])


def cmd_check(cfg, paths, args):
    from exp002 import drive
    info = env_info()
    info["conda_warning"] = None if warn_conda() else "CONDA_DEFAULT_ENV empty"
    print("python executable :", info["python_executable"])
    print("CONDA_DEFAULT_ENV :", info["CONDA_DEFAULT_ENV"])
    print("packages          :", json.dumps(info["packages"]))
    print("torch cuda        :", info.get("torch_cuda"), "cuda_available:", info.get("cuda_available"), "gpu:", info.get("gpu"), "mem_gb:", info.get("gpu_mem_gb"))
    print("arch_list         :", info.get("arch_list"))
    want = cfg["env"].get("torch_version")
    tv = (info["packages"].get("torch") or "")
    print("torch version     : installed=%s config env.torch_version=%s match=%s ; torch.version.cuda=%s (config expects cu117 -> 11.7)" % (
        tv or None, want, (tv == want) if want else None, info.get("torch_cuda")))
    print("paths:")
    for k in ("work", "dataset_dir", "data_root", "acwm_src", "train_base", "cache"):
        print("  %-12s %s  exists=%s" % (k, paths[k], paths[k].exists()))
    for envn in (cfg["paths"]["work_env"], cfg["paths"]["dataset_env"], cfg["paths"]["train_env"], cfg["paths"]["cache_env"], cfg["code"]["src_env"]):
        print("  env %-16s = %s" % (envn, os.environ.get(envn, "(unset -> default)")))
    src = paths["acwm_src"]
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() if src.exists() else None
    print("branch checkout HEAD:", head, "matches config.code.commit:", head == cfg["code"]["commit"])
    print("disk free GB: cache=%s train=%s data=%s" % (disk_free_gb(paths["cache"]), disk_free_gb(paths["train_base"]), disk_free_gb(paths["dataset_dir"])))
    splits = drive.select_splits(cfg)
    print("data splits present (metadata.pt):")
    for s in splits:
        print("  %-12s %s" % (s, (paths["data_root"] / s / "metadata.pt").exists()))
    print("config nulls that block training (set by a human after smoke):")
    for k in ("train.batch_size", "train.num_hist", "train.epochs", "train.budget", "env.torch_version"):  # budget null is intended
        cur = cfg
        for p in k.split("."):
            cur = cur[p]
        print("  %-20s %s" % (k, cur))
    print("rclone:", shutil.which(cfg["data_impl"]["drive"]["rclone_bin"]))
    print("manifests:", sorted(p.name for p in paths["manifests"].glob("*.json")) if paths["manifests"].exists() else None)
    from exp002 import cache as CA
    print("q1 cache DONE:", (CA.cache_dir(paths) / "DONE").exists())
    for c in cfg["conditions"]["run_order"]:
        for s in cfg["train"]["seeds"]:
            n = run_name(c, s)
            print("  run %-34s DONE=%s" % (n, (run_dir(paths, n) / "DONE").exists()))
    ok = info["packages"].get("torch") is not None and info["packages"].get("numpy") is not None
    return 0 if ok else 1


def cmd_download(cfg, paths, args):
    from exp002 import drive
    splits = drive.select_splits(cfg)
    if os.environ.get("EXP002_DRIVE_MOUNT_SRC") or os.environ.get("EXP002_DATA_MODE", "").lower() == "mount":     # Colab path (확정 기록 10): no rclone
        src = drive.find_mount_src(cfg, splits)
        if src is None:
            log("download (mount mode): no folder containing all splits %s found in %s. Add the shared folder to My Drive (shortcut or copy) and set "
                "EXP002_DRIVE_MOUNT_SRC to the folder that contains the split directories." % (splits, [str(c) for c in drive.mount_src_candidates(cfg)]))
            return 2
        return drive.copy_from_mount(cfg, paths, splits, src, dry_run=args.dry_run, force=args.force)
    return drive.download(cfg, paths, splits, dry_run=args.dry_run, force=args.force)


def cmd_manifest(cfg, paths, args):
    from exp002 import data as D, drive
    splits = drive.select_splits(cfg)
    log("manifest: splits %s ; train %d episodes/level, eval %d/level" % (splits, cfg["data"]["train_episodes_per_level"], cfg["data"]["eval_episodes_per_level"]))
    if args.dry_run:
        for s in splits:
            log("  %-12s metadata.pt present: %s" % (s, (paths["data_root"] / s / "metadata.pt").exists()))
        return 0
    listing, meta = D.list_episodes(paths["data_root"], splits)
    miss = [s for s in splits if s not in listing]
    if miss:
        log("missing splits (run `python run.py download`): %s" % miss)
        return 1
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    write_json(listing, paths["manifests"] / "listing.json")
    write_json(meta, paths["manifests"] / "episode_meta.json")
    for c in cfg["conditions"]["list"]:
        m = D.build_train_manifest(cfg, listing, paths["data_root"], c["name"])
        write_json(m, paths["manifests"] / ("train_%s.json" % c["name"]))
        log("manifest %s: %d episodes %s" % (c["name"], m["n_episodes"], m["group_counts"]))
    rows = D.build_cache_table(cfg, listing, paths["data_root"])
    write_json(rows, paths["manifests"] / "cache_episodes.json")
    log("cache table: %d episodes (train %d, test %d)" % (len(rows), sum(r["part"] == "train" for r in rows), sum(r["part"] == "test" for r in rows)))
    return 0


def cmd_smoke(cfg, paths, args):
    from exp002 import smoke
    warn_conda()
    smoke.run(cfg, paths, dry_run=args.dry_run)
    return 0


def cmd_cache(cfg, paths, args):
    from exp002 import cache as CA
    rn = args.run or run_name(cfg["conditions"]["run_order"][0], cfg["train"]["seeds"][0])
    n = len(read_json(paths["manifests"] / "cache_episodes.json")) if (paths["manifests"] / "cache_episodes.json").exists() else (
        cfg["data"]["train_episodes_per_level"] * len(cfg["data"]["train_levels_b"]) + cfg["data"]["eval_episodes_per_level"] * len(cfg["data"]["levels"]))
    if args.max_eps:
        n = args.max_eps * (len(cfg["data"]["train_levels_b"]) + len(cfg["data"]["levels"]))
    est = CA.disk_estimate_gb(n)
    log("cache: encoder from run %s ; %d episodes ; disk estimate %s ; free at cache path %s GB" % (rn, n, {k: round(v, 2) for k, v in est.items()}, disk_free_gb(paths["cache"])))
    if args.dry_run:
        return 0
    if not (run_dir(paths, rn) / "DONE").exists():
        raise FileNotFoundError("checkpoint run %s has no DONE (train it first, or pass --run <trained run name>)" % rn)
    CA.run(cfg, paths, rn, max_eps=args.max_eps, suffix=args.suffix, overwrite=args.force)
    return 0


def cmd_train(cfg, paths, args):
    from exp002 import trainrun
    rc_all = 0
    for c in conds(cfg, args):
        for s in seeds(cfg, args):
            key = "train:%s" % run_name(c, s)
            if not args.dry_run:
                attempts_check(cfg, paths, key)
            try:
                rc = trainrun.train(cfg, paths, c, s, force=args.force, dry_run=args.dry_run)
            except ConfigNull:
                raise
            except Exception:
                if not args.dry_run:
                    attempts_record(paths, key, False)
                raise
            if not args.dry_run:
                attempts_record(paths, key, rc == 0)
            rc_all = rc_all or rc
    return rc_all


def cmd_q1(cfg, paths, args):
    from exp002 import q1
    q1.run(cfg, paths, dry_run=args.dry_run)
    return 0


def _per_run(mod, cfg, paths, args, stage):
    rc = 0
    for c in conds(cfg, args):
        for s in seeds(cfg, args):
            key = "%s:%s" % (stage, run_name(c, s))
            if not args.dry_run:
                attempts_check(cfg, paths, key)
                if not (run_dir(paths, run_name(c, s)) / "DONE").exists():
                    log("%s: run %s not trained (no DONE) - skipped" % (stage, run_name(c, s)))
                    rc = 1
                    continue
            try:
                getattr(mod, "run")(cfg, paths, c, s, dry_run=args.dry_run)
                if not args.dry_run:
                    attempts_record(paths, key, True)
            except ConfigNull:
                raise
            except Exception:
                if not args.dry_run:
                    attempts_record(paths, key, False)
                raise
    return rc


def cmd_q2(cfg, paths, args):
    from exp002 import q2
    return _per_run(q2, cfg, paths, args, "q2")


def cmd_q3(cfg, paths, args):
    from exp002 import q3
    return _per_run(q3, cfg, paths, args, "q3")


def cmd_aggregate(cfg, paths, args):
    from exp002 import aggregate
    if args.dry_run:
        log("aggregate: would read results/q1, results/<run>/q2*, q3.json and write results.json")
        return 0
    paths["results_json"] = paths["results_dir"].parent / "results.json" if args.results_dir else paths["exp"] / "results.json"
    aggregate.run(cfg, paths)
    log("wrote %s" % paths["results_json"])
    return 0


def cmd_selftest(cfg, paths, args):
    from exp002 import selftest
    return selftest.run(cfg)


STAGES = {"check": cmd_check, "download": cmd_download, "manifest": cmd_manifest, "smoke": cmd_smoke, "cache": cmd_cache, "train": cmd_train,
          "q1": cmd_q1, "q2": cmd_q2, "q3": cmd_q3, "aggregate": cmd_aggregate, "selftest": cmd_selftest}
NO_GUARD = {"check", "selftest"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--config", default=None, help="config.yaml path (default: next to run.py)")
    ap.add_argument("--results-dir", default=None, help="results directory (default: ./results)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--condition", default=None, help="multi_60_ind_160 | single_ind (default: run_order[0])")
    ap.add_argument("--all-conditions", action="store_true", help="every condition in config conditions.run_order")
    ap.add_argument("--include-optional", action="store_true", help="with --all-conditions: also conditions.optional (multi_60_ind_160_small)")
    ap.add_argument("--seeds", default=None, help="comma separated training seeds (default: config train.seeds)")
    ap.add_argument("--run", default=None, help="cache: trained run whose frozen encoder is used")
    ap.add_argument("--max-eps", type=int, default=None, help="cache: episodes per level (smoke); needs --suffix")
    ap.add_argument("--suffix", default="", help="cache directory suffix for partial caches, e.g. _smoke")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.max_eps and not args.suffix:
        ap.error("--max-eps needs --suffix (e.g. _smoke) so that a partial cache cannot be mistaken for the real one")
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    if args.results_dir:
        paths["results_dir"] = Path(args.results_dir).resolve()
    set_seed(cfg["seed"])
    tag = args.condition or ""
    start_file_log(paths, args.stage + ("_dry" if args.dry_run else ""), tag)
    log("stage=%s dry_run=%s python=%s conda_env=%s" % (args.stage, args.dry_run, sys.executable, os.environ.get("CONDA_DEFAULT_ENV")))
    if args.stage not in ("selftest",):
        warn_conda()
    try:
        key = "stage:%s" % args.stage
        guard = args.stage not in NO_GUARD and not args.dry_run and args.stage not in ("train", "q2", "q3")
        if guard:
            attempts_check(cfg, paths, key)
        rc = STAGES[args.stage](cfg, paths, args) or 0
        if guard:
            attempts_record(paths, key, rc == 0)
    except ConfigNull as e:
        if args.dry_run:
            log("DRY-RUN: a real run of this stage would STOP here: %s" % e)
            sys.exit(0)
        log_exception(e)
        print("\nSTOPPED: a config value is still null (human decision needed). Nothing was guessed.", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        log_exception(e)
        if args.stage not in NO_GUARD and not args.dry_run and args.stage not in ("train", "q2", "q3"):
            attempts_record(paths, "stage:%s" % args.stage, False)
        print("\nFAILED. Paste the log file shown at the top (logs/) when asking for help.", file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
