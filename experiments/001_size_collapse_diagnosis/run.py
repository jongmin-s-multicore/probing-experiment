#!/usr/bin/env python
"""Experiment 001 entry point.  python run.py <stage> [options]   (see README.md; run_all.sh chains the stages)

Stages: check | download | listing | gen | train | extract | diagnose | probe | intervene | aggregate | selftest
Every stage supports --dry-run (prints what it would do, runs nothing heavy). Config: config.yaml (seeds included).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp001.common import load_config, resolve_paths, run_dir, write_json, read_json, log, set_seed  # noqa: E402


SUFFIX = ""


def run_name(cfg, dist, seed):
    return cfg["train"]["run_name_template"].format(dist=dist, seed=seed) + SUFFIX


def parse_model(cfg, spec):
    dist, seed = spec.split(":") if ":" in spec else (spec, str(cfg["train"]["seed"]))
    return dist, int(seed)


def cmd_check(cfg, paths, args):
    info = {"acwm_src": str(paths["acwm_src"]), "dataset_dir": str(paths["dataset_dir"]), "clothmove": str(paths["clothmove"])}
    src = paths["acwm_src"]
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() if src.exists() else None
    info["acwm_src_exists"] = src.exists()
    info["acwm_src_head"] = head
    info["commit_matches_config"] = head == cfg["code"]["commit"]
    info["data_present"] = {s: (paths["clothmove"] / s / "metadata.pt").exists() for s in ("ind_train", "ind_test", "ood_test")}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        info["torch"] = None
    print(json.dumps(info, indent=2))
    ok = info["acwm_src_exists"] and info["commit_matches_config"]
    if not ok:
        log("check FAILED: checkout of the acwm-phys commit missing or different (run setup.sh)")
    return 0 if ok else 1


def cmd_download(cfg, paths, args):
    script = paths["acwm_src"] / "repro" / "setup" / "download_acwm_phys.py"
    dest = paths["dataset_dir"] / "acwm_phys"
    cmd = [sys.executable, str(script), "--envs", "cloth_move", "--dest", str(dest)]
    log("download: " + " ".join(cmd) + f"   (HF {cfg['data']['hf_repo']} @ {cfg['data']['hf_revision'][:7]}; pinned in the branch script)")
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=paths["acwm_src"]).returncode


def cmd_listing(cfg, paths, args):
    from exp001 import data as D
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    splits = ["ind_train", "ind_test", "ood_test"] + sorted(p.name for p in paths["clothmove"].glob("gen_*") if p.is_dir()) if paths["clothmove"].exists() else []
    if args.dry_run:
        log(f"listing: would read metadata.pt from {paths['clothmove']} splits={splits}")
        return 0
    listing = D.list_episodes(paths["clothmove"], splits)
    for s, exp in cfg["data"]["expected_episodes"].items():
        if len(listing.get(s, [])) != exp:
            log(f"WARNING split {s}: {len(listing.get(s, []))} episodes, expected {exp}")
    write_json(listing, paths["manifests"] / "listing.json")
    mans = {}
    for name, specs in cfg["train"]["distributions"].items():
        try:
            mans.update(D.build_manifests({**cfg, "train": {**cfg["train"], "distributions": {name: specs}}}, listing, paths["clothmove"]))
        except KeyError as e:
            log(f"manifest {name}: skipped, missing source split {e}")
    for name, m in mans.items():
        write_json(m, paths["manifests"] / f"train_{name}.json")
        log(f"manifest {name}: {m['n_episodes']} episodes {m['group_counts']}")
    return 0


def cmd_gen(cfg, paths, args):
    from exp001 import gen_midsize
    rc = gen_midsize.run(cfg, paths, dry_run=args.dry_run)
    return 0 if rc in (0, 3) else rc          # 3 = unavailable/disabled: not an error, status recorded in manifests/gen_status.json


def cmd_train(cfg, paths, args):
    dist, seed = parse_model(cfg, args.model)
    name = run_name(cfg, dist, seed)
    rd = run_dir(paths, name)
    man = paths["manifests"] / f"train_{dist}.json"
    if not man.exists() and not args.dry_run:
        log(f"missing {man}; run: python run.py listing")
        return 1
    if (rd / "DONE").exists() and not args.force:
        log(f"train {name}: DONE exists, skip")
        return 0
    ov = dict(cfg["train"]["hydra_overrides"])
    ov["training.seed"] = seed
    if args.smoke:                                   # 60 episodes, 1 epoch, run name gets the _smoke suffix
        ov.update({"training.epochs": 1, "training.save_every_x_epoch": 1, "env.dataset.n_rollout": 60})
    cmd = [sys.executable, "train.py", "--config-name", "train.yaml"] + [f"{k}={v}" for k, v in ov.items()]
    cmd += ["env.dataset._target_=exp001.data.load_manifest_slice_train_val", f"+env.dataset.manifest={man}",
            f"ckpt_base_path={paths['train_base']}", f"hydra.run.dir={rd}"]
    env = dict(os.environ)
    env.update({"DATASET_DIR": str(paths["dataset_dir"]), "WANDB_MODE": env.get("WANDB_MODE", "disabled"), "HYDRA_FULL_ERROR": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONPATH": str(HERE / "src") + os.pathsep + env.get("PYTHONPATH", "")})
    log("train cwd=" + str(paths["acwm_src"]) + "\n  " + " ".join(cmd))
    if args.dry_run:
        return 0
    rd.mkdir(parents=True, exist_ok=True)
    write_json(read_json(man), rd / "train_manifest.json")
    with open(rd / "run_info.txt", "a") as f:
        f.write(f"START exp001 commit={cfg['code']['commit']} dist={dist} seed={seed} args={' '.join(cmd)}\n")
    rc = subprocess.run(cmd, cwd=paths["acwm_src"], env=env).returncode
    if rc == 0:
        (rd / "DONE").write_text("ok\n")
    return rc


def _models(cfg, args):
    return [parse_model(cfg, args.model)] if args.model else [parse_model(cfg, s) for s in cfg["train"]["models_to_diagnose"]]


def cmd_extract(cfg, paths, args):
    from exp001 import extract
    for dist, seed in _models(cfg, args):
        name = run_name(cfg, dist, seed)
        log(f"extract {name} (smoke_max_eps={args.max_eps})")
        if args.dry_run:
            continue
        extract.run(cfg, paths, name, smoke_max_eps=args.max_eps, epoch=args.epoch, overwrite=args.force)
    return 0


def _cpu_stage(mod, cfg, paths, args, outname):
    import importlib
    m = importlib.import_module(f"exp001.{mod}")
    for dist, seed in _models(cfg, args):
        name = run_name(cfg, dist, seed)
        if (paths["results_dir"] / name / outname).exists() and not args.force:
            log(f"{mod} {name}: {outname} exists, skip")
            continue
        log(f"{mod} {name}")
        if not args.dry_run:
            m.run(cfg, paths, name)
    return 0


def cmd_diagnose(cfg, paths, args):
    return _cpu_stage("diagnose", cfg, paths, args, "diagnose.json")


def cmd_probe(cfg, paths, args):
    return _cpu_stage("probe_stage", cfg, paths, args, "probe.json")


def cmd_intervene(cfg, paths, args):
    if not cfg["intervene"]["enabled"]:
        log("intervene disabled in config")
        return 0
    return _cpu_stage("intervene", cfg, paths, args, "intervene.json")


def cmd_aggregate(cfg, paths, args):
    from exp001 import aggregate
    if args.dry_run:
        log("aggregate: would write results.json")
        return 0
    aggregate.run(cfg, paths)
    log(f"wrote {HERE / 'results.json'}")
    return 0


def cmd_selftest(cfg, paths, args):
    from exp001 import selftest
    selftest.run(cfg)
    return 0


STAGES = {"check": cmd_check, "download": cmd_download, "listing": cmd_listing, "gen": cmd_gen, "train": cmd_train, "extract": cmd_extract,
          "diagnose": cmd_diagnose, "probe": cmd_probe, "intervene": cmd_intervene, "aggregate": cmd_aggregate, "selftest": cmd_selftest}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help='"distribution:seed", e.g. baseline_single:0 (train requires it)')
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="train: 60 episodes, 1 epoch")
    ap.add_argument("--max-eps", type=int, default=None, help="extract: episodes per group (smoke)")
    ap.add_argument("--epoch", type=int, default=0, help="extract: checkpoint epoch (0 = latest)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to run names (smoke tests use _smoke so real results are never overwritten)")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = args.suffix or ("_smoke" if args.smoke else "")
    if args.max_eps and not SUFFIX:
        ap.error("--max-eps needs --suffix (e.g. _smoke) so that a partial extraction cannot be mistaken for the real one")
    cfg = load_config(args.config)
    if os.environ.get("EXP001_MODELS"):          # run_all.sh: space separated "dist:seed" list overrides config.train.models_to_diagnose
        cfg["train"]["models_to_diagnose"] = os.environ["EXP001_MODELS"].split()
    set_seed(cfg["seed"])
    paths = resolve_paths(cfg)
    if args.stage == "train" and not args.model:
        ap.error("train needs --model distribution:seed")
    sys.exit(STAGES[args.stage](cfg, paths, args) or 0)


if __name__ == "__main__":
    main()
