#!/usr/bin/env python
"""Experiment 001 entry point.  python run.py <stage> [options]   (see README.md; run_all.sh chains the stages)

Stages: check | download | maskcheck | listing | train | repro_check | extract | diagnose | probe | intervene | aggregate | sync | selftest
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

from exp001.common import load_config, resolve_paths, run_dir, write_json, read_json, log, set_seed, approved_distributions  # noqa: E402


SUFFIX = ""


def run_name(cfg, dist, seed):
    return cfg["train"]["run_name_template"].format(dist=dist, seed=seed) + SUFFIX


def parse_model(cfg, spec):
    dist, seed = spec.split(":") if ":" in spec else (spec, str(cfg["train"]["seed"]))
    return dist, int(seed)


def cmd_check(cfg, paths, args):
    blk = cfg["_ds"]
    info = {"dataset": cfg["data"]["active"], "acwm_src": str(paths["acwm_src"]), "dataset_dir": str(paths["dataset_dir"]), "data_root": str(paths["data_root"])}
    src = paths["acwm_src"]
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() if src.exists() else None
    info["acwm_src_exists"] = src.exists()
    info["acwm_src_head"] = head
    info["commit_matches_config"] = head == cfg["code"]["commit"]
    info["cloth_sim_branch"] = cfg["code"]["cloth_sim_branch"]
    splits = sorted(set(blk["download_default_splits"]) | set(blk["eval_sets"]))
    info["data_present"] = {s: (paths["data_root"] / s / "metadata.pt").exists() for s in splits}
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


def _dist_train_splits(cfg, dists):
    out = []
    for d in dists:
        for spec in cfg["train"]["distributions"][d]:
            if spec["source"] not in out:
                out.append(spec["source"])
    return out


def cmd_download(cfg, paths, args):
    """cloth_sim: data.source (env EXP001_DATA_SOURCE) = drive (default, rclone) | hf (huggingface_hub). Needed splits only.
    Large: not run by the authoring session. After a real download the integrity check writes results/download_check.json."""
    from exp001 import drive
    blk = cfg["_ds"]
    dists = [d for d in (args.dists or os.environ.get("EXP001_DOWNLOAD_DISTS", "")).replace(",", " ").split() if d]
    if blk["kind"] == "acwm_phys":
        script = paths["acwm_src"] / "repro" / "setup" / "download_acwm_phys.py"
        cmd = [sys.executable, str(script), "--envs", "cloth_move", "--dest", str(paths["dataset_dir"] / "acwm_phys")]
        log("download (legacy acwm_phys): " + " ".join(cmd))
        return 0 if args.dry_run else subprocess.run(cmd, cwd=paths["acwm_src"]).returncode
    splits = drive.select_splits(cfg, dists)
    source = drive.data_source(cfg)
    if source == "drive":
        return drive.download(cfg, paths, splits, dry_run=args.dry_run, force=args.force)
    if source != "hf":
        log(f"unknown data.source '{source}' (use drive | hf)")
        return 2
    rc = _download_hf(cfg, paths, args, splits)
    if rc == 0 and not args.dry_run:
        drive.check(cfg, paths, splits)
    return rc


def _download_hf(cfg, paths, args, splits):
    """data.source = hf. Kept as an option; hanbin5/cloth-sim was reported unusable (technical problem) when the Drive path was added."""
    blk = cfg["_ds"]
    patterns = [f"{s}/*" for s in splits]
    from exp001 import hf
    repo = blk["hf_repo"]
    tok, src = hf.get_token()
    log(f"download: HF {repo} (repo_type={blk['hf_repo_type']}, revision={blk['hf_revision']}) -> {paths['data_root']}")
    log("  allow_patterns=" + json.dumps(patterns))
    log(f"  HF token: {'found (' + src + ')' if tok else 'NOT found (set HF_TOKEN or run huggingface-cli login; the repo returned 401 unauthenticated)'}")
    if args.dry_run and not args.query_size:
        return 0
    if tok is None:
        if args.dry_run:
            log("  dry-run: skipping HF API size query because there is no token")
            return 0
        hf.require_token(repo)                       # exits with the instruction text
    try:
        from huggingface_hub import HfApi, snapshot_download
        if args.query_size:
            import fnmatch
            info = HfApi().dataset_info(repo, files_metadata=True, timeout=30, token=tok)
            files = [(f.rfilename, f.size) for f in info.siblings]
            tot = {p: sum(sz or 0 for n, sz in files if fnmatch.fnmatch(n, p)) for p in patterns}
            log("  sizes from HF API (bytes): " + json.dumps(tot) + f"  total={sum(tot.values())}")
        if args.dry_run:
            return 0
        snapshot_download(repo, repo_type=blk["hf_repo_type"], revision=blk["hf_revision"], local_dir=str(paths["data_root"]),
                          allow_patterns=patterns, max_workers=8, token=tok)
    except SystemExit:
        raise
    except Exception as e:
        msg = hf.explain_error(e, repo)
        if msg is None:
            raise
        log(f"download FAILED: {type(e).__name__}")
        print(msg, file=sys.stderr)
        return 3
    return 0


def cmd_maskcheck(cfg, paths, args):
    from exp001 import maskcheck
    maskcheck.run(cfg, paths, dry_run=args.dry_run)
    return 0


def cmd_listing(cfg, paths, args):
    from exp001 import data as D
    paths["manifests"].mkdir(parents=True, exist_ok=True)
    blk = cfg["_ds"]
    if blk["kind"] == "cloth_sim":
        splits = [f"{s['name']}_{part}" for s in blk["sizes"] for part in ("train", "test")]
    else:
        splits = list(blk["expected_episodes"])
    if args.dry_run:
        log(f"listing: would read metadata.pt from {paths['data_root']} splits={splits}")
        return 0
    listing, meta = D.list_episodes(paths["data_root"], splits)
    exp = blk["expected_episodes"]
    for s, names in listing.items():
        e = exp.get(s) if blk["kind"] != "cloth_sim" else exp[s.rsplit("_", 1)[1]]
        if e is not None and len(names) != e:
            log(f"WARNING split {s}: {len(names)} episodes, expected {e}")
    write_json(listing, paths["manifests"] / "listing.json")
    write_json(meta, paths["manifests"] / "episode_meta.json")
    mans = {}
    for name, specs in cfg["train"]["distributions"].items():
        try:
            mans.update(D.build_manifests({**cfg, "train": {**cfg["train"], "distributions": {name: specs}}}, listing, paths["data_root"]))
        except KeyError as e:
            log(f"manifest {name}: skipped, missing source split {e}")
    for name, m in mans.items():
        write_json(m, paths["manifests"] / f"train_{name}.json")
        log(f"manifest {name}: {m['n_episodes']} episodes {m['group_counts']}")
    return 0


def cmd_train(cfg, paths, args):
    dist, seed = parse_model(cfg, args.model)
    name = run_name(cfg, dist, seed)
    rd = run_dir(paths, name)
    man = paths["manifests"] / f"train_{dist}.json"
    status = cfg["_ds"].get("distribution_status", {}).get(dist)
    if dist not in approved_distributions(cfg) and not (args.dry_run or args.smoke):
        log(f"train {dist}: not approved by a human ({status}). Add it to config train.approved_distributions or env EXP001_APPROVED. Refusing.")
        return 2
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


def cmd_repro_check(cfg, paths, args):
    """Reproduction check, first thing after the baseline is trained: extract only the reported sizes (GPU), then area tables only (CPU).
    Writes results/<run>/repro_check.json. Numbers only; a human decides whether the phenomenon reproduces."""
    from exp001 import extract, diagnose
    rc = cfg["repro_check"]
    for dist, seed in _models(cfg, args):
        name = run_name(cfg, dist, seed)
        out = paths["results_dir"] / name / "repro_check.json"
        if out.exists() and not args.force:
            log(f"repro_check {name}: exists, skip")
            continue
        log(f"repro_check {name}: groups={rc['groups']} max_eps/group={rc['max_episodes_per_group']}")
        if args.dry_run:
            continue
        extract.run(cfg, paths, name, smoke_max_eps=args.max_eps, groups=rc["groups"], cache_suffix="_repro",
                    max_per_group=rc["max_episodes_per_group"], overwrite=args.force, light=True)
        diagnose.run(cfg, paths, name, cache_suffix="_repro", out_name="repro_check.json", area_only=True)
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


def cmd_sync(cfg, paths, args):
    from exp001 import sync
    sync.run(cfg, paths, dry_run=args.dry_run, ckpt=True if args.ckpt else None, cache=True if args.cache else None,
                  models=[args.model] if args.model else None, run_label=args.run_label)
    return 0                                  # a failed/skipped upload never fails the experiment; see results/sync_status.json


def cmd_selftest(cfg, paths, args):
    from exp001 import selftest
    selftest.run(cfg)
    return 0


def cmd_run_name(cfg, paths, args):
    """Print the run name of --model dist:seed (seed is not parsed, so a placeholder like @SEED@ works). Used by overnight.sh / run_all.sh."""
    dist, seed = args.model.split(":")
    print(run_name(cfg, dist, seed))
    return 0


STAGES = {"run_name": cmd_run_name, "check": cmd_check, "download": cmd_download, "maskcheck": cmd_maskcheck, "listing": cmd_listing, "train": cmd_train,
          "repro_check": cmd_repro_check, "extract": cmd_extract,
          "diagnose": cmd_diagnose, "probe": cmd_probe, "intervene": cmd_intervene, "aggregate": cmd_aggregate, "sync": cmd_sync, "selftest": cmd_selftest}


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
    ap.add_argument("--dists", default="", help='download: also fetch the training splits of these distributions (comma/space separated)')
    ap.add_argument("--query-size", action="store_true", help="download (source=hf): ask the HF API for file sizes of the selected splits (best effort). source=drive: --dry-run always lists via rclone")
    ap.add_argument("--ckpt", action="store_true", help="sync: also upload checkpoints (large, ~1.7 GB per run, estimate)")
    ap.add_argument("--cache", action="store_true", help="sync: also upload the extraction cache (large, ~5 GB per run, estimate)")
    ap.add_argument("--run-label", default=None, help="sync: run name used in the Drive folder name (default exp001_<models>)")
    ap.add_argument("--suffix", default="", help="appended to run names (smoke tests use _smoke so real results are never overwritten)")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = args.suffix or ("_smoke" if args.smoke else "")
    if args.max_eps and not SUFFIX:
        ap.error("--max-eps needs --suffix (e.g. _smoke) so that a partial extraction cannot be mistaken for the real one")
    cfg = load_config(args.config)
    if os.environ.get("EXP001_MODELS"):          # run_all.sh: space separated "dist:seed" list overrides config.train.models_to_diagnose
        cfg["train"]["models_to_diagnose"] = os.environ["EXP001_MODELS"].split()
    if args.stage == "run_name":
        if not args.model or ":" not in args.model:
            ap.error("run_name needs --model distribution:seed")
        sys.exit(cmd_run_name(cfg, None, args))
    set_seed(cfg["seed"])
    paths = resolve_paths(cfg)
    if args.stage == "train" and not args.model:
        ap.error("train needs --model distribution:seed")
    sys.exit(STAGES[args.stage](cfg, paths, args) or 0)


if __name__ == "__main__":
    main()
