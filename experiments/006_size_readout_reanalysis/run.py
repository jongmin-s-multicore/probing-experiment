#!/usr/bin/env python
"""Experiment 006 (size readout re-analysis of 002) entry point.   python run.py <stage> [options]

Stages (in order):  check  selftest  data_meta  extract  pool  labels  transfer  repro  analyze_a  analyze_b  analyze_c  analyze_d  analyze_e  aggregate  figures
  check      CPU   inputs of the 002 Drive folder: existence, sizes, sha256 (small files; --hash for all), model-load compatibility facts. Reads only
  selftest   CPU   synthetic data, known answers. The numbers it prints are NOT results
  data_meta  CPU   metadata.pt of the training splits from the 002 archive (action normalisation statistics for extract)
  extract    GPU   recompute rollouts (trained + untrained predictors) from the 002 checkpoints; --smoke measures memory/time only
  pool       CPU   patch-mean of the cached encoder tokens;  labels  CPU  contrast-property labels (D)
  transfer   CPU   the 002 output-space probe fitted once and applied to encoder / persistence / rollouts / untrained rollouts
  repro      CPU   difference to 002's stored numbers (no tolerance applied)
  analyze_a..e CPU analyses A-E ;  aggregate -> results.json (experiment folder and WORK_ROOT) ;  figures -> analysis/*.png (needs matplotlib)
Every stage supports --dry-run (prints the plan, writes nothing). Each stage is attempted at most config run.max_attempts (3) times, then stops.
Environment variables: EXP006_INPUT_ROOT (read only, the 002 Drive folder), EXP006_WORK_ROOT (outputs), ACWM_PHYS_SRC, DATASET_DIR. Seeds are in config.yaml.
Nothing is ever written inside EXP006_INPUT_ROOT or the 001/002/003 folders (guard in src/exp006/base.py)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from exp006 import base as B  # noqa: E402  (puts 002's src on sys.path)
from exp006.base import log  # noqa: E402

STAGES = ["check", "selftest", "data_meta", "extract", "pool", "labels", "transfer", "repro", "analyze_a", "analyze_b", "analyze_c", "analyze_d", "analyze_e", "aggregate", "figures"]
CPU_ALL = ["pool", "labels", "transfer", "repro", "analyze_a", "analyze_b", "analyze_c", "analyze_d", "analyze_e", "aggregate", "figures"]


def plan(stage, cfg, paths):
    P = paths
    d = {
        "check": ("reads the 002 Drive folder (%s)" % P["input_root"], ["%s/input_manifest.json" % P["results"]]),
        "selftest": ("synthetic data only", []),
        "data_meta": ("train_manifest.json of 6 runs + archive %s" % P["archive"], ["%s/<split>/metadata.pt (local)" % P["data_root"], "%s/metadata_fields.json" % P["results"]]),
        "extract": ("checkpoints %s/<run>/checkpoints/model_latest.pth, 002 cache frames/actions" % P["train_outputs"], ["%s/<run>/{rollout_pooled.npy, untrained_*.npy, meta.json, DONE}" % P["extract"]]),
        "pool": ("002 cache tokens_dino.npy", ["%s/pooled_tokens.npy" % P["cache"]]),
        "labels": ("002 cache frames_u8.npy, actions_raw.npy (cv2 mask rule)", ["%s/d_labels.npz" % P["cache"]]),
        "transfer": ("pooled_tokens.npy, extract/*", ["%s/transfer_pred.npz" % P["cache"], "%s/transfer_meta.json" % P["parts"]]),
        "repro": ("transfer_pred.npz, 002 q2_cells.npz, 002 results.json", ["%s/repro.json" % P["parts"]]),
        "analyze_a": ("002 q2_cells.npz (stored) and transfer_pred.npz (reproduced)", ["%s/analyze_a.json" % P["parts"], "%s/A_per_episode_*.npz" % P["results"]]),
        "analyze_b": ("transfer_pred.npz", ["%s/analyze_b.json" % P["parts"]]),
        "analyze_c": ("pooled_tokens.npy, extract/*, transfer_pred.npz", ["%s/analyze_c.json" % P["parts"]]),
        "analyze_d": ("pooled_tokens.npy, d_labels.npz, extract/*", ["%s/analyze_d.json" % P["parts"]]),
        "analyze_e": ("002 q2_cells.npz, 002 results.json", ["%s/analyze_e.json" % P["parts"]]),
        "aggregate": ("parts/*.json", [str(B.EXP_DIR / "results.json"), "%s/results.json" % P["results"]]),
        "figures": ("results.json", [str(B.EXP_DIR / "analysis") + "/fig*.png"]),
    }[stage]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["cpu_all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hash", action="store_true", help="check: sha256 of every input file including checkpoints and big npy")
    ap.add_argument("--smoke", action="store_true", help="extract: small memory/time measurement on one run, writes only results/smoke_extract.json")
    ap.add_argument("--runs", default=None, help="extract: comma separated run names (default all 6)")
    ap.add_argument("--fresh", action="store_true", help="extract: ignore DONE markers")
    ap.add_argument("--no-write-check", action="store_true", help="check: do not write the output folder (manifest not saved)")
    args = ap.parse_args()
    if args.stage == "cpu_all":
        rc = 0
        for st in CPU_ALL:
            r = os.system("%s %s %s%s" % (sys.executable, str(HERE / "run.py"), st, " --dry-run" if args.dry_run else ""))
            if r != 0:
                log("cpu_all: stage %s failed (exit %s); stopping" % (st, r))
                sys.exit(3)
        return
    cfg = B.load_config(args.config)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.stage == "selftest":
        from exp006 import selftest
        sys.exit(0 if selftest.run() else 1)
    cfg2 = B.load_cfg002(cfg)
    paths = B.resolve_paths(cfg)
    B.C2.set_seed(int(cfg["seeds"]["global"]))
    if not args.dry_run:
        B.C2.start_file_log(paths, args.stage)
    log("stage=%s dry_run=%s python=%s started=%s" % (args.stage, args.dry_run, sys.executable, started))
    log("INPUT_ROOT (read only) = %s | WORK_ROOT (outputs) = %s" % (paths["input_root"], paths["work_root"]))
    if args.dry_run:
        inp, outs = plan(args.stage, cfg, paths)
        log("DRY-RUN plan for %s: reads: %s" % (args.stage, inp))
        for o in outs:
            log("  would write: %s" % o)
        log("  max attempts: %d ; seeds: %s" % (cfg["run"]["max_attempts"], cfg["seeds"]))
        return
    from exp006 import stages_pre as SP, stages_an as SA, stages_agg as SG, ctx as CX
    st = args.stage
    info = {"started": started, "host_python": sys.version.split()[0], "git": B.git_info(), "snapshot_sha256": B.snapshot_sha256(), "gpu": B.gpu_info()}

    def work():
        if st in ("data_meta", "extract", "pool", "labels", "transfer", "repro", "analyze_a", "analyze_b", "analyze_c", "analyze_d", "analyze_e", "aggregate"):
            B.probe_output_dir(paths)                       # Drive write failure must stop the stage here
        if st == "check":
            return SP.stage_check(cfg, cfg2, paths, do_hash=args.hash, write=not args.no_write_check)
        if st == "data_meta":
            return SP.stage_data_meta(cfg, cfg2, paths)
        if st == "extract":
            return do_extract(cfg, cfg2, paths, args)
        if st == "pool":
            return SP.stage_pool(cfg, cfg2, paths)
        if st == "labels":
            return SP.stage_labels(cfg, cfg2, paths)
        if st == "transfer":
            return SP.stage_transfer(cfg, cfg2, paths)[0]
        if st == "repro":
            return SA.stage_repro(cfg, cfg2, paths)
        if st in ("analyze_a", "analyze_b", "analyze_c", "analyze_d", "analyze_e"):
            return {"analyze_a": SA.stage_a, "analyze_b": SA.stage_b, "analyze_c": SA.stage_c, "analyze_d": SA.stage_d, "analyze_e": SA.stage_e}[st](cfg, cfg2, paths)
        if st == "aggregate":
            return SG.stage_aggregate(cfg, cfg2, paths, started)
        if st == "figures":
            return do_figures(cfg, paths)
    try:
        _, attempts = B.run_with_retries(work, cfg["run"]["max_attempts"], st)
    except B.StageFailed as e:
        log(str(e))
        for a in e.attempts:
            log("  attempt %(attempt)d: %(error)s" % a)
        sys.exit(3)
    except B.InputWriteRefused as e:
        log("STOP (not retried): %s" % e)
        sys.exit(4)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["attempts"] = attempts
    pn = st if st.startswith("analyze") or st == "repro" else None
    if pn:
        pp = Path(paths["parts"]) / (pn + ".json")
        if pp.exists():
            d = B.read_json(pp)
            d["stage_info"] = info
            B.write_json(paths, d, pp)
    log("stage %s finished after %d attempt(s)" % (st, len(attempts)))


def do_extract(cfg, cfg2, paths, args):
    import torch
    from exp006 import extract as EX
    from exp006.ctx import Ctx
    runs = [r for _, _, r in B.all_runs(cfg)]
    if args.runs:
        runs = [r for r in args.runs.split(",") if r]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        log("WARNING: no CUDA device; the extraction would be very slow on CPU")
    ctx = Ctx(cfg, cfg2, paths)
    if args.smoke:
        run = cfg["extract"]["smoke"]["run"]
        sm = cfg["extract"]["smoke"]
        bs = int(cfg["extract"]["batch_episodes"])
        stats = {}
        t0 = time.time()
        r = EX.extract_run(cfg, cfg2, paths, run, dev, ctx.C, max_rows=(sm["n_train_batches"] * bs, sm["n_test_batches"] * bs), smoke_stats=stats, write=False,
                           out_dir=Path(paths["extract"]) / (run + "_smoke"))
        import numpy as np
        med = {k: float(np.median(v)) for k, v in stats.items()}
        n_runs = len([1 for _ in B.all_runs(cfg)])
        per_run = med["train"] * (ctx.Etr / bs) + med["test"] * (ctx.Ete / bs)
        res = {"run": run, "device": str(dev), "gpu": B.gpu_info(), "batch_seconds_train_batches": stats["train"], "batch_seconds_test_batches": stats["test"],
               "peak_cuda_mem_gib": r["meta"]["peak_cuda_mem_gib"], "median_batch_s": med,
               "estimate_note": "ESTIMATE (추정): median batch time x number of batches; the first batch includes warm-up; model load time and checkpoint hashing are not included",
               "estimated_extract_minutes_per_run": per_run / 60.0, "estimated_extract_minutes_all_%d_runs" % n_runs: per_run * n_runs / 60.0,
               "smoke_total_seconds": time.time() - t0, "rows_used": [r["meta"]["n_train_rows"], r["meta"]["n_test_rows"]]}
        B.write_json(paths, res, Path(paths["results"]) / "smoke_extract.json")
        log("SMOKE: %s" % {k: v for k, v in res.items() if k.startswith(("peak", "estimated", "median"))})
        return res
    out = {}
    for run in runs:
        d = Path(paths["extract"]) / run
        if args.fresh and (d / "DONE").exists():
            (d / "DONE").unlink()
        r = EX.extract_run(cfg, cfg2, paths, run, dev, ctx.C)
        out[run] = {"skipped": bool(r.get("skipped")), "elapsed_s": (r.get("meta") or {}).get("elapsed_s")}
    return out


def do_figures(cfg, paths):
    try:
        os.environ["MPLBACKEND"] = "Agg"      # unconditional (see src/exp006/figs.py)
        import matplotlib  # noqa: F401
    except ImportError:
        log("matplotlib not installed: figures skipped (no error)")
        return {"skipped": "matplotlib missing"}
    from exp006 import figs
    res = B.read_json(B.EXP_DIR / cfg["outputs"]["results_json_exp"])
    made, failed = figs.make_all(res, B.EXP_DIR / cfg["outputs"]["figure_dir"])
    for m in made:
        log("figure: %s" % m)
    for f in failed:
        log("FIGURE FAILED: %s" % f)
    if failed and not made:
        raise RuntimeError("no figure could be made: %s" % failed)
    return {"made": made, "failed": failed}


if __name__ == "__main__":
    main()
