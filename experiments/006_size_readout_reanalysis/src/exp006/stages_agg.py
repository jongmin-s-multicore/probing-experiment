"""aggregate: parts -> results.json (experiment folder AND work root). interpretation is always null."""
from __future__ import annotations

import time
from pathlib import Path

from . import base as B
from .base import log

PARTS = ["analyze_a", "analyze_b", "analyze_c", "analyze_d", "analyze_e", "analyze_f", "repro"]

LIMITATIONS = [
    "Post hoc exploration: planned after the 002 results and reviews were read (hypotheses.md candidate J). Not confirmatory evidence.",
    "All probes are the Q1 linear probe (patch-mean features -> standardized ridge). The attentive probe is not used. Rollout latents are stored patch-mean pooled only.",
    "Rollout latents are recomputed from the 002 checkpoints on this machine; 002's own latents were not stored. Differences to 002's stored numbers are in `reproduction`.",
    "Condition (a) single_ind: its predictor was trained on ind episodes only. In C the probe is trained on a060/ind/a160 training rows, so for (a) two of three training sizes are rollouts of sizes the predictor never saw.",
    "Seed-level intervals use n = 3 training seeds (df = 2); episode-bootstrap intervals cover evaluation-episode sampling only.",
    "Untrained predictor: weights from torch.manual_seed(seed) + the hydra constructor of the trained class; the result depends on that initialisation. Two variants and 3 seeds are reported separately.",
    "D properties: no proprio field exists (the dataset class returns zeros); gripper rows are commanded displacements (cumulative actions), not measured positions. Mask-derived properties depend on the HSV mask rule.",
    "Persistence uses the context frame t = num_hist - 1 = 0, i.e. the first frame, because num_hist = 1.",
    "F (teacher forcing): the teacher-forced latent uses the real encoder latent of the previous frame as context, including its real (encoded) proprio part; the open-loop rollout feeds back the predicted proprio part. The action part is the same action embedding in both. Step 1 is the same computation in both (the stored difference is reported).",
    "Probe-seed variation of the linear probe is only through the GroupKFold split used to pick alpha; seeds listed in config.",
]


def stage_aggregate(cfg, cfg2, paths, started_at, dry_run=False):
    parts = {}
    missing = []
    for n in PARTS:
        p = Path(paths["parts"]) / (n + ".json")
        if p.exists():
            parts[n] = B.read_json(p)
        else:
            missing.append(n)
    extract_meta, smoke = {}, None
    for _, _, run in B.all_runs(cfg):
        mp = Path(paths["extract"]) / run / "meta.json"
        if mp.exists():
            extract_meta[run] = B.read_json(mp)
    sp = Path(paths["results"]) / "smoke_extract.json"
    if sp.exists():
        smoke = B.read_json(sp)
    im = Path(paths["results"]) / "input_manifest.json"
    res = {
        "experiment": cfg["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "started_at": started_at, "status": cfg["status"],
        "hypothesis": cfg["hypothesis"], "interpretation": None,
        "config_path": str(B.EXP_DIR / "config.yaml"), "config_sha256": B.sha256_file(B.EXP_DIR / "config.yaml"), "config002_sha256": cfg2.get("_config002_sha256"),
        "git": B.git_info(), "code_version": B.code_version(), "snapshot_sha256": B.snapshot_sha256(), "environment": B.env_info(), "gpu": B.gpu_info(),
        "seeds": cfg["seeds"], "input_root": str(paths["input_root"]), "work_root": str(paths["work_root"]),
        "input_manifest": B.read_json(im) if im.exists() else None, "extract_meta": extract_meta, "smoke_extract": smoke,
        "stages_present": sorted(parts), "stages_missing": missing,
        "reproduction": parts.get("repro"),
        "A": parts.get("analyze_a"), "B": parts.get("analyze_b"), "C": parts.get("analyze_c"), "D": parts.get("analyze_d"), "E": parts.get("analyze_e"), "F": parts.get("analyze_f"),
        "limitations": LIMITATIONS,
        "note": "No thresholds, pass/fail or p values are produced by this experiment. Synthetic selftest numbers are not results.",
    }
    log("aggregate: parts present %s, missing %s" % (sorted(parts), missing))
    if dry_run:
        return res
    B.probe_output_dir(paths)
    B.write_json(paths, res, B.EXP_DIR / cfg["outputs"]["results_json_exp"])
    B.write_json(paths, res, Path(paths["work_root"]) / cfg["outputs"]["results_json_work"])
    log("results.json written to the experiment folder and to %s" % (Path(paths["work_root"]) / cfg["outputs"]["results_json_work"]))
    return res
