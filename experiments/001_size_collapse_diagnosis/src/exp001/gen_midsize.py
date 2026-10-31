"""OPTIONAL: new intermediate-cloth-size episodes. DISABLED by default (config.gen_midsize.enabled=false).

Status of what is verified (see README "중간 크기 생성"):
  - The ACWM-Phys Cloth Move episode generator is NOT in the acwm-phys branch checkout. The branch README states the dataset has
    no simulator; the only cloth code in the checkout is the DINO-WM FlexEnv cloth scene (env/deformable_env, PyFleX), which is a
    different scene (different renderer, no gripper-balls/sphere task) and cannot reproduce the dataset's 512x512 videos.
  - Therefore this script does not contain a simulator. It only (1) wraps a generator command that YOU provide through the
    environment variable named in config.gen_midsize.generator_cmd_env, (2) validates the output format against the dataset
    layout, (3) writes manifests/gen_status.json. Without the variable it exits with code 3 and records why.

Command template (placeholders filled by this script): {area_scale} {n} {seed} {out_dir}
The command must write <out_dir>/metadata.pt (list of dicts with video_path, actions [T,A], length; same schema as the dataset)
and the mp4 files next to it, 65 frames of 512x512.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .common import log, write_json


def tag_of(scale):
    return "a" + str(scale).replace(".", "p")


def run(cfg, paths, dry_run=False):
    g = cfg["gen_midsize"]
    status = {"enabled": bool(g["enabled"]), "available": False, "generated": [], "reason": None}
    out_json = paths["manifests"] / "gen_status.json"
    if not g["enabled"]:
        status["reason"] = "config.gen_midsize.enabled is false (option)"
        write_json(status, out_json)
        log(status["reason"])
        return 3
    tmpl = os.environ.get(g["generator_cmd_env"])
    if not tmpl:
        status["reason"] = (f"environment variable {g['generator_cmd_env']} is not set. No Cloth Move generator exists in the "
                            "checked-out branch; provide your own generator command (see module docstring).")
        write_json(status, out_json)
        log(status["reason"])
        return 3
    status["available"] = True
    for sc in g["area_scales"]:
        out_dir = Path(paths["clothmove"]) / ("gen_" + tag_of(sc))
        cmd = tmpl.format(area_scale=sc, n=g["episodes_per_scale"], seed=g["seed"], out_dir=out_dir)
        log(f"generate area_scale={sc}: {cmd}")
        if dry_run:
            continue
        if (out_dir / "metadata.pt").exists():
            log(f"{out_dir}/metadata.pt exists, skip generation")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(shlex.split(cmd), check=True)
        status["generated"].append({"area_scale": sc, "tag": tag_of(sc), "dir": str(out_dir), "validation": validate(out_dir, g)})
    write_json(status, out_json)
    return 0


def validate(out_dir, g):
    import torch
    import decord
    meta = torch.load(Path(out_dir) / "metadata.pt", map_location="cpu", weights_only=True)
    rep = {"n_metadata": len(meta), "missing_videos": 0, "frame_count_ok": True, "shape_ok": True, "first_video": None}
    for i, e in enumerate(meta):
        p = Path(out_dir) / Path(e["video_path"]).name
        if not p.exists():
            rep["missing_videos"] += 1
            continue
        if i == 0:
            vr = decord.VideoReader(str(p), num_threads=1)
            fr = vr.get_batch([0]).asnumpy()
            rep["first_video"] = {"frames": len(vr), "shape": list(fr.shape[1:])}
            rep["frame_count_ok"] = len(vr) == g["validate"]["frames"]
            rep["shape_ok"] = list(fr.shape[1:3]) == [g["validate"]["height"], g["validate"]["width"]]
        if tuple(e["actions"].shape)[0] < 60:
            rep["frame_count_ok"] = False
    return rep
