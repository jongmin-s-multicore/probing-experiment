#!/usr/bin/env python
"""Experiment 003 entry point.   python run.py <stage> [options]      (see README.md)

Stages: run_name | check | download | maskcheck | listing | smoke | train | extract | metrics | baseline | aggregate | sync | selftest
Every stage supports --dry-run (prints what it would do; no heavy work). Seeds come from config.yaml (train.seeds, seed).
Run inside the activated conda environment. Each stage logs to logs/<stage>_<time>.log. After run.max_attempts (3) consecutive failures
a stage/run refuses to run (CLAUDE.md retry limit) until a human removes the entry in results/attempts.json.
GPU training (stage `train`, and `smoke`) must not be started without explicit user approval (hypotheses.md 확정 기록 8)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp003.common import (load_config, resolve_paths, run_dir, write_json, read_json, log, set_seed, approved_distributions, all_models,  # noqa: E402
                           start_file_log, log_exception, attempts_check, attempts_record, warn_conda, env_info)


def run_name(cfg, dist, seed, suffix=""):
    return cfg["train"]["run_name_template"].format(dist=dist, seed=seed) + suffix


def parse_model(cfg, spec):
    dist, seed = spec.split(":") if ":" in spec else (spec, str(cfg["train"]["seed"]))
    if dist not in cfg["train"]["distributions"]:
        raise SystemExit("unknown distribution %s (config: %s)" % (dist, list(cfg["train"]["distributions"])))
    return dist, int(seed)


def models(cfg, args):
    return [parse_model(cfg, args.model)] if args.model else [parse_model(cfg, s) for s in all_models(cfg)]


REFUSED = 3          # a refusal (unpatched train.py, retrain of a DONE run) is a setup problem, not a failed attempt: it is not counted in attempts.json


def guarded(cfg, paths, key, fn, dry_run):
    """Run fn() with the 3-attempt guard."""
    if not dry_run:
        attempts_check(cfg, paths, key)
    try:
        rc = fn() or 0
    except Exception:
        if not dry_run:
            attempts_record(paths, key, False)
        raise
    if not dry_run and rc != REFUSED:
        attempts_record(paths, key, rc == 0)
    return rc


def cmd_run_name(cfg, paths, args):
    d, s = args.model.split(":")
    print(run_name(cfg, d, s))
    return 0


def cmd_check(cfg, paths, args):
    info = env_info()
    info["conda_warning"] = None if warn_conda() else "CONDA_DEFAULT_ENV empty"
    src = paths["acwm_src"]
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() if src.exists() else None
    info.update(acwm_src=str(src), acwm_src_exists=src.exists(), acwm_src_head=head, commit_matches_config=head == cfg["code"]["commit"],
                dataset_dir=str(paths["dataset_dir"]), data_root=str(paths["data_root"]), train_base=str(paths["train_base"]), cache=str(paths["cache"]))
    splits = sorted(set(cfg["_ds"]["download_default_splits"]) | set(s["source"] for d in cfg["train"]["distributions"].values() for s in d))
    info["data_present"] = {s: (paths["data_root"] / s / "metadata.pt").exists() for s in splits}
    info["manifests"] = sorted(p.name for p in paths["manifests"].glob("*.json")) if paths["manifests"].exists() else []
    info["runs_done"] = {m: (run_dir(paths, run_name(cfg, *parse_model(cfg, m))) / "DONE").exists() for m in all_models(cfg)}
    info["approved_distributions"] = sorted(approved_distributions(cfg))
    print(json.dumps(info, indent=2))
    ok = info["acwm_src_exists"] and info["commit_matches_config"]
    if not ok:
        log("check: checkout of the acwm-phys commit missing or different (run setup.sh)")
    return 0 if ok else 1


def cmd_download(cfg, paths, args):
    from exp003 import drive
    dists = args.dists.replace(",", " ").split() or list(cfg["train"]["distributions"])     # default: train splits of all four distributions + 9 test splits
    splits = drive.select_splits(cfg, dists)
    log("download splits: %s" % splits)
    if drive.data_source(cfg) != "drive":
        log("only data.source=drive is supported in 003 (hf code was not adapted)")
        return 2
    if drive.mount_mode_requested():
        # Colab mount-copy mode (no rclone): archive in EXP003_WORK_ROOT if present, else parallel copy from the mounted Drive + one-time archive.
        return drive.prepare_data(cfg, paths, splits, dry_run=args.dry_run, force=args.force)
    return drive.download(cfg, paths, splits, dry_run=args.dry_run, force=args.force)


def cmd_maskcheck(cfg, paths, args):
    from exp003 import maskcheck
    maskcheck.run(cfg, paths, dry_run=args.dry_run)
    return 0


def cmd_listing(cfg, paths, args):
    from exp003 import data as D
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    splits = [f"{s['name']}_{part}" for s in cfg["_ds"]["sizes"] for part in ("train", "test")]
    if args.dry_run:
        log("listing: would read metadata.pt from %s splits=%s" % (paths["data_root"], splits))
        return 0
    listing, meta = D.list_episodes(paths["data_root"], splits)
    exp = cfg["_ds"]["expected_episodes"]
    for s, names in listing.items():
        e = exp[s.rsplit("_", 1)[1]]
        if len(names) != e:
            log("WARNING split %s: %d episodes, expected %d" % (s, len(names), e))
    write_json(listing, paths["manifests"] / "listing.json")
    write_json(meta, paths["manifests"] / "episode_meta.json")
    for name, specs in cfg["train"]["distributions"].items():
        try:
            m = D.build_manifests({**cfg, "train": {**cfg["train"], "distributions": {name: specs}}}, listing, paths["data_root"])[name]
        except KeyError as e:
            log("manifest %s: skipped, missing source split %s" % (name, e))
            continue
        write_json(m, paths["manifests"] / ("train_%s.json" % name))
        log("manifest %s: %d episodes %s" % (name, m["n_episodes"], m["group_counts"]))
    return 0


def _train_cmd(cfg, paths, dist, seed, name, smoke=False):
    rd = run_dir(paths, name)
    man = paths["manifests"] / ("train_%s.json" % dist)
    ov = dict(cfg["train"]["hydra_overrides"])
    ov["training.seed"] = seed
    if smoke:
        sm = cfg["smoke"]
        ov.update({"training.epochs": sm["epochs"], "training.save_every_x_epoch": 1, "env.dataset.n_rollout": sm["n_episodes"]})
    cmd = [sys.executable, "train.py", "--config-name", "train.yaml"] + ["%s=%s" % (k, v) for k, v in ov.items()]
    cmd += ["env.dataset._target_=exp003.data.load_manifest_slice_train_val", "+env.dataset.manifest=%s" % man,
            "ckpt_base_path=%s" % paths["train_base"], "hydra.run.dir=%s" % rd]
    env = dict(os.environ)
    env.update({"DATASET_DIR": str(paths["dataset_dir"]), "WANDB_MODE": env.get("WANDB_MODE", "disabled"), "HYDRA_FULL_ERROR": "1",
                "PYTHONPATH": str(HERE / "src") + os.pathsep + env.get("PYTHONPATH", "")})
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and _torch_at_least((2, 1)):
        # expandable_segments is unknown to the torch 2.0.x allocator (RuntimeError: Unrecognized CachingAllocator option). Respect a user-set value.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return cmd, env, rd, man


def _torch_at_least(ver):
    """True only if torch is installed with version >= ver (no import of torch; unknown/missing -> False)."""
    try:
        from importlib.metadata import version
        parts = version("torch").split("+")[0].split(".")
        return tuple(int(x) for x in parts[:2]) >= tuple(ver)
    except Exception:
        return False


def _train_one(cfg, paths, dist, seed, args, smoke=False, suffix=""):
    from exp003 import resume as R
    name = run_name(cfg, dist, seed, suffix)
    cmd, env, rd, man = _train_cmd(cfg, paths, dist, seed, name, smoke)
    if dist not in approved_distributions(cfg) and not (args.dry_run or smoke):
        log("train %s: not approved for GPU training (add to env EXP003_APPROVED after the user approves). Refusing." % dist)
        return 2
    log("train %s cwd=%s\n  %s" % (name, paths["acwm_src"], " ".join(cmd)))
    if args.dry_run:
        return 0
    if not man.exists():
        log("missing %s; run: python run.py listing" % man)
        return 1
    if (rd / "DONE").exists():
        if not args.force:
            log("train %s: DONE exists, skip" % name)
            return 0
        if not smoke and os.environ.get("EXP003_ALLOW_RETRAIN") != "1":
            log("train %s: DONE exists and --force was given. Refusing to retrain a finished run (protects the finished mix_wide runs). "
                "Set EXP003_ALLOW_RETRAIN=1 only if a human decided to retrain it." % name)
            return REFUSED
    rec = None
    if not smoke:
        # Resume guard (user decision 2026-11-05). The pinned train.py would load model_latest.pth and train `epochs` MORE epochs; only the patched
        # train.py (colab/train_resume.patch) treats training.epochs as the total target. The 9 runs of this stage are trained with the patched file.
        patched = R.train_py_is_patched(paths["acwm_src"])
        have_ck = R.has_checkpoint(rd)
        if not patched:
            log("train %s: %s/train.py is NOT the resume-patched file (marker %r / loop %r missing)%s. Refusing to train. Apply colab/train_resume.patch to the "
                "checkout (git apply) first." % (name, paths["acwm_src"], R.RESUME_MARKER, R.LOOP_EXPR,
                                                 ": an unfinished checkpoint exists and the unpatched file would train 15 MORE epochs" if have_ck else ""))
            return REFUSED
        prep = R.prepare_run(rd, paths["train_base"] / "quarantine", loadable=lambda p: R.default_loadable(p, paths["acwm_src"]))
        log("train %s: prepare -> %s" % (name, json.dumps(prep)))
        ck = rd / "checkpoints"
        efs = R.epoch_files(ck)
        rec = {"start_time": time.strftime("%F %T"), "action": prep["action"], "restored_from": prep.get("restored_from"), "quarantined_to": prep.get("to"),
               "epoch_checkpoint_at_start": efs[0][0] if efs and prep["action"] in ("resume", "repaired_resume") else 0,
               "target_total_epochs": int(cfg["train"]["epochs"]), "save_every_x_epoch": int(cfg["train"]["hydra_overrides"]["training.save_every_x_epoch"]),
               "patched_train_py": True, "train_py_sha256": R.sha256_file(Path(paths["acwm_src"]) / "train.py"),
               "patch_sha256": R.sha256_file(HERE / "colab" / "train_resume.patch"), **R.gpu_info()}
        if prep["action"] in ("resume", "repaired_resume"):
            log("train %s: RESUMING from checkpoint epoch %s with the patched train.py (training.epochs = total target %d)" % (name, rec["epoch_checkpoint_at_start"], rec["target_total_epochs"]))
    rd.mkdir(parents=True, exist_ok=True)
    write_json(read_json(man), rd / "train_manifest.json")
    with open(rd / "run_info.txt", "a") as f:
        f.write("START exp003 commit=%s dist=%s seed=%s args=%s\n" % (cfg["code"]["commit"], dist, seed, " ".join(cmd)))
    idx = R.record_attempt(rd, rec) if rec else None
    log("train %s: started. After the first epoch check the time per epoch and GPU memory (nvidia-smi); not measured on this GPU before." % name)
    rc = subprocess.run(cmd, cwd=paths["acwm_src"], env=env).returncode
    if idx is not None:
        R.finish_attempt(rd, idx, rc)
    if rc == 0:
        (rd / "DONE").write_text("ok\n")
        if not smoke:
            keep = (cfg.get("colab") or {}).get("keep_ckpt_epochs")
            if keep is not None:
                removed = R.prune_run(rd, keep)
                if removed:
                    log("train %s: pruned %s (kept %s + model_latest.pth)" % (name, removed, keep))
    return rc


def cmd_train(cfg, paths, args):
    if not args.model:
        raise SystemExit("train needs --model distribution:seed")
    dist, seed = parse_model(cfg, args.model)
    if dist not in approved_distributions(cfg) and not args.dry_run:      # a refusal is not a failed attempt: checked before the retry guard
        log("train %s: not approved for GPU training (set env EXP003_APPROVED after the user approves). Refusing." % dist)
        return 2
    return guarded(cfg, paths, "train:%s" % run_name(cfg, dist, seed), lambda: _train_one(cfg, paths, dist, seed, args), args.dry_run)


def cmd_smoke(cfg, paths, args):
    """12 GB check: 1-epoch train on 60 episodes + small extract, with nvidia-smi peak. NOT executed by the authoring session."""
    from exp003 import smoke, extract
    sm = cfg["smoke"]
    dist, seed, suf = sm["distribution"], sm["seed"], "_smoke"
    name = run_name(cfg, dist, seed, suf)
    args.force = True                                # a smoke run always starts from scratch (run name has the _smoke suffix)
    def tr():
        return _train_one(cfg, paths, dist, seed, args, smoke=True, suffix=suf)
    def ex():
        extract.run(cfg, paths, name, smoke_max_eps=sm["extract_episodes_per_size"], overwrite=True)
    out = smoke.run(cfg, paths, tr, ex, dry_run=args.dry_run)
    if args.dry_run:
        tr()
    return 0 if out["status"] in ("dry_run", "finished") else 1


def cmd_extract(cfg, paths, args):
    from exp003 import extract
    rc = 0
    for dist, seed in models(cfg, args):
        name = run_name(cfg, dist, seed, args.suffix)
        log("extract %s" % name)
        if args.dry_run:
            continue
        if not (run_dir(paths, name) / "DONE").exists():
            log("extract %s: run not trained (no DONE) - skipped" % name)
            rc = 1
            continue
        rc = guarded(cfg, paths, "extract:%s" % name, lambda: (extract.run(cfg, paths, name, smoke_max_eps=args.max_eps, epoch=args.epoch, overwrite=args.force), 0)[1], False) or rc
    return rc


def cmd_metrics(cfg, paths, args):
    from exp003 import metrics
    rc = 0
    for dist, seed in models(cfg, args):
        name = run_name(cfg, dist, seed, args.suffix)
        log("metrics %s" % name)
        if args.dry_run:
            continue
        if not (paths["cache"] / name / "DONE").exists():
            log("metrics %s: no extract cache - skipped" % name)
            rc = 1
            continue
        rc = guarded(cfg, paths, "metrics:%s" % name, lambda: (metrics.run(cfg, paths, name, dist, seed), 0)[1], False) or rc
    return rc


def cmd_baseline(cfg, paths, args):
    """Install/verify the 001 results.json that the H-H2 verdict reads (Colab snapshot has no experiments/001). rc 1 if it cannot be found/read."""
    from exp003 import baseline
    if args.dry_run:
        log("baseline: would look for a %d-byte results.json under %s and copy it to %s" % (
            int(cfg["colab"]["baseline_001_bytes"]), baseline.default_roots(), baseline.target_path(cfg)))
        return 0
    ok, info = baseline.run(cfg)
    write_json(info, paths["results_dir"] / "baseline_001_install.json")
    return 0 if ok else 1


def cmd_aggregate(cfg, paths, args):
    from exp003 import aggregate
    if args.dry_run:
        log("aggregate: would read results/<run>/metrics.json, write results.json and copy it to $EXP003_WORK_ROOT/results/results.json when that variable is set")
        return 0
    res = aggregate.run(cfg, paths)
    out = HERE / "results.json"
    log("wrote %s" % out)
    b = res["verdict"]["H-H2"]["baseline"]
    log("aggregate: baseline.found=%s file=%s" % (b.get("found"), b.get("file")))
    dst = aggregate.persist(out, os.environ.get("EXP003_WORK_ROOT"))     # Colab: HERE is on the local disk and disappears with the runtime; results/ is the Drive link
    if dst:
        log("aggregate: copied results.json to %s (Drive, survives the runtime)" % dst)
    else:
        log("aggregate: EXP003_WORK_ROOT not set, results.json stays only at %s" % out)
    if not b.get("found") or b.get("missing_run_or_path"):
        log("aggregate: FAILED CHECK - the 001 baseline results.json was not found/readable, so H-H2 is not_evaluable. Run `python run.py baseline`, then aggregate again.")
        return 1
    return 0


def cmd_sync(cfg, paths, args):
    from exp003 import sync
    sync.run(cfg, paths, dry_run=args.dry_run, ckpt=True if args.ckpt else None, cache=True if args.cache else None,
             models=[args.model] if args.model else None, run_label=args.run_label)
    return 0


def cmd_selftest(cfg, paths, args):
    from exp003 import selftest
    selftest.run(cfg)
    return 0


STAGES = {"run_name": cmd_run_name, "check": cmd_check, "download": cmd_download, "maskcheck": cmd_maskcheck, "listing": cmd_listing, "smoke": cmd_smoke,
          "train": cmd_train, "extract": cmd_extract, "metrics": cmd_metrics, "baseline": cmd_baseline, "aggregate": cmd_aggregate, "sync": cmd_sync, "selftest": cmd_selftest}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help='"distribution:seed", e.g. single_a026:0 (train requires it; others default to all 12)')
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-eps", type=int, default=None, help="extract: episodes per size (smoke); needs --suffix")
    ap.add_argument("--epoch", type=int, default=0, help="extract: checkpoint epoch (0 = latest)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dists", default="", help="download: distributions whose train splits are fetched (default: all four)")
    ap.add_argument("--ckpt", action="store_true", help="sync: also upload checkpoints (large)")
    ap.add_argument("--cache", action="store_true", help="sync: also upload the extraction cache")
    ap.add_argument("--run-label", default=None)
    ap.add_argument("--suffix", default="", help="appended to run names (smoke runs use _smoke)")
    args = ap.parse_args()
    if args.max_eps and not args.suffix:
        ap.error("--max-eps needs --suffix (e.g. _smoke) so that a partial extraction cannot be mistaken for the real one")
    cfg = load_config(args.config)
    if args.stage == "run_name":
        if not args.model or ":" not in args.model:
            ap.error("run_name needs --model distribution:seed")
        sys.exit(cmd_run_name(cfg, None, args))
    set_seed(cfg["seed"])
    paths = resolve_paths(cfg)
    if args.stage != "selftest":
        start_file_log(paths, args.stage + ("_dry" if args.dry_run else ""), (args.model or "").replace(":", "_s"))
        warn_conda()
    try:
        rc = STAGES[args.stage](cfg, paths, args) or 0
    except Exception as e:
        log_exception(e)
        print("\nFAILED. See the log file shown at the top (logs/).", file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
