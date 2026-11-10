"""Stages that prepare inputs: check, data_meta, pool, labels, transfer. No interpretation anywhere; numbers and file facts only."""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import base as B
from . import lib as L
from .base import log
from .ctx import Ctx, load_pooled, load_extract, save_part


# ------------------------------------------------------------------------------------------------ names of apply sets
def n_enc(k):
    return "encoder|%d" % k


def n_per(k):
    return "persist|%d" % k


def n_roll(run, k):
    return "rollout|%s|%d" % (run, k)


def n_tf(run, k):
    return "teacher|%s|%d" % (run, k)


def n_unt(run, variant, seed, k):
    return "untrained|%s|%s|%d|%d" % (run, variant, seed, k)


# ------------------------------------------------------------------------------------------------ metadata.pt preflight
def metadata_missing(cfg, paths):
    """Preflight without GPU or attempts: extract.action_stats_local reads <data_root>/<split>/metadata.pt for every episode of each run's train_manifest.json
    (the first 009 Colab pilot failed because these files were missing). Returns a list of problem strings (empty = ok)."""
    bad = []
    for _, _, run in B.all_runs(cfg):
        mp = Path(paths["train_outputs"]) / run / "train_manifest.json"
        if not mp.exists():
            bad.append("%s: train_manifest.json missing (%s)" % (run, mp))
            continue
        for sp in sorted({e["split"] for e in B.read_json(mp)["episodes"]}):
            if not (Path(paths["data_root"]) / sp / "metadata.pt").exists():
                bad.append("%s: %s/%s/metadata.pt missing" % (run, paths["data_root"], sp))
    return bad


# ------------------------------------------------------------------------------------------------ check
def input_files(cfg, paths):
    P = paths
    items = []
    for f in ("cells.npz", "test_rows.json", "q1_summary.json"):
        items.append(("results/q1/" + f, P["results002"] / "q1" / f))
    for cond, seed, run in B.all_runs(cfg):
        for f in ("q2_cells.npz", "q2_summary.json", "q3.json"):
            items.append(("results/%s/%s" % (run, f), P["results002"] / run / f))
        base = P["train_outputs"] / run
        for f in ("checkpoints/model_latest.pth", "hydra.yaml", "train_manifest.json", "DONE"):
            items.append(("train_runs/outputs/%s/%s" % (run, f), base / f))
    for f in ("frames_u8.npy", "tokens_dino.npy", "mask_feat.npy", "actions_raw.npy", "episodes.json", "meta.json", "DONE"):
        items.append(("exp002_cache/q1_tokens/" + f, P["cache002"] / f))
    items.append((cfg["paths"]["archive_name"], P["archive"]))
    return items


def stage_check(cfg, cfg2, paths, do_hash=False, write=True):
    t0 = time.time()
    items = input_files(cfg, paths)
    big = (1 << 28)
    expect = cfg["inputs"].get("expect") or {}
    hash_all = bool(do_hash or cfg["inputs"].get("hash_big_files"))
    files, missing, mismatch = {}, [], []
    for rel, p in items:
        p = Path(p)
        if not p.exists():
            files[rel] = {"exists": False}
            missing.append(rel)
            continue
        size = p.stat().st_size
        d = {"exists": True, "bytes": size}
        key = str(p.relative_to(paths["input_root"])) if str(p).startswith(str(paths["input_root"])) else rel
        if size < big or hash_all or key in expect:
            d["sha256"] = B.sha256_file(p)
        else:
            d["sha256"] = None
        e = expect.get(key)
        if e:
            if e.get("bytes") is not None and e["bytes"] != size:
                mismatch.append(key + ":bytes")
            if e.get("sha256") and d.get("sha256") and e["sha256"] != d["sha256"]:
                mismatch.append(key + ":sha256")
        files[rel] = d
    e2 = B.exp002_dir()
    prov = {
        "config002_sha256": cfg2.get("_config002_sha256"), "config002_expected": cfg["base"]["config002_sha256"],
        "requirements002_sha256": B.sha256_file(e2 / "colab" / "requirements_colab.txt") if (e2 / "colab" / "requirements_colab.txt").exists() else None,
        "requirements002_expected": cfg["base"]["requirements002_sha256"],
        "train_resume_patch_sha256": B.sha256_file(e2 / "colab" / "train_resume.patch") if (e2 / "colab" / "train_resume.patch").exists() else None,
        "train_resume_patch_expected_prefix": cfg["base"]["train_resume_patch_sha256_prefix"],
    }
    src = paths["acwm_src"]
    acwm = {"path": str(src), "exists": src.exists()}
    if src.exists():
        try:
            import subprocess
            acwm["head"] = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
            acwm["head_matches_config"] = acwm["head"] == cfg["base"]["acwm_commit"]
            tp = src / "train.py"
            acwm["train_py_sha256"] = B.sha256_file(tp) if tp.exists() else None
            acwm["train_py_has_resume_patch_marker"] = ("[exp002 operational]" in tp.read_text()) if tp.exists() else None
        except Exception as e:                       # noqa: BLE001
            acwm["error"] = str(e)
    mio = (e2 / "src" / "exp002" / "model_io.py").read_text()
    load_compat = {
        "model_io_imports_train_py": ("import train" in mio or "from train" in mio),
        "model_io_needs": "unpickling needs the classes of the pinned commit (models.*, datasets.* are on sys.path via ACWM_PHYS_SRC) and run_dir/hydra.yaml; model_io.load_world_model reads encoder from the checkpoint if present else instantiates cfg.encoder (torch.hub download of DINOv2)",
        "resume_patch_needed_for_load": False,
        "resume_patch_reason": "the patch changes only train.py (optimizer state_dict saving, resume of action/proprio encoders). Checkpoint keys read by model_io are predictor, decoder, proprio_encoder, action_encoder, epoch (+ encoder if saved); they are the same with or without the patch. Optimizer entries are ignored by model_io.",
    }
    md_bad = metadata_missing(cfg, paths)
    res = {"input_root": str(paths["input_root"]), "work_root": str(paths["work_root"]), "files": files, "missing": missing, "expect_mismatch": mismatch,
           "metadata_pt_missing": md_bad, "metadata_pt_ok": not md_bad,
           "provenance": prov, "acwm_src": acwm, "model_load_compat": load_compat, "python": sys.version.split()[0], "platform": platform.platform(),
           "environment": B.env_info(), "gpu": B.gpu_info(), "hash_computed_for_big_files": hash_all, "seconds": round(time.time() - t0, 1),
           "expect_template_for_config": {k: {"bytes": v["bytes"], "sha256": v.get("sha256")} for k, v in files.items() if v.get("exists") and v.get("sha256") and k.endswith((".pth", ".npz", ".npy"))}}
    log("check: %d files listed, %d missing, %d expectation mismatches" % (len(files), len(missing), len(mismatch)))
    for m in missing:
        log("  MISSING: %s" % m)
    log("check: metadata.pt of the training splits (needed by extract; run `data_meta`): %s" % ("all present" if not md_bad else "MISSING %d" % len(md_bad)))
    for m in md_bad[:20]:
        log("  MISSING metadata.pt: %s" % m)
    for k, v in prov.items():
        log("  %s = %s" % (k, v))
    if write:
        B.probe_output_dir(paths)
        B.write_json(paths, res, Path(paths["results"]) / "input_manifest.json")
    return res


# ------------------------------------------------------------------------------------------------ data_meta
def stage_data_meta(cfg, cfg2, paths, dry_run=False):
    """Only metadata.pt of the splits that the 6 runs trained on (needed for the action normalisation statistics of extract). Reuses the 002 archive
    cloth_sim_data.tar (read only) or an existing dataset folder. Videos are not needed: frames come from the 002 cache."""
    import tarfile
    splits = set()
    for _, _, run in B.all_runs(cfg):
        mp = paths["train_outputs"] / run / "train_manifest.json"
        if not mp.exists():
            raise FileNotFoundError(mp)
        for e in B.read_json(mp)["episodes"]:
            splits.add(e["split"])
    splits = sorted(splits)
    root = Path(paths["data_root"])
    B.guard_output(paths, root / "x")
    todo = [s for s in splits if not (root / s / "metadata.pt").exists()]
    log("data_meta: splits needed %s ; already present %s" % (splits, [s for s in splits if s not in todo]))
    if dry_run:
        return {"splits": splits, "todo": todo}
    if todo:
        arc = Path(paths["archive"])
        if not arc.exists():
            raise FileNotFoundError("%s missing and the dataset folder %s lacks metadata.pt of %s" % (arc, root, todo))
        got = set()
        with tarfile.open(str(arc), "r|*") as tf:            # streaming read, the archive is not modified
            for m in tf:
                parts = [p for p in m.name.replace("\\", "/").split("/") if p]
                if not m.isreg() or ".." in parts or m.name.startswith("/") or parts[-1] != "metadata.pt" or len(parts) < 2:
                    continue
                sp = parts[-2]
                if sp in todo and sp not in got:
                    dest = root / sp / "metadata.pt"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    f = tf.extractfile(m)
                    tmp = dest.with_name("metadata.pt.part")
                    with open(tmp, "wb") as fo:
                        fo.write(f.read())
                    if tmp.stat().st_size != m.size:
                        raise RuntimeError("size mismatch extracting %s" % m.name)
                    os.replace(tmp, dest)
                    got.add(sp)
                    log("  extracted %s (%d bytes)" % (dest, m.size))
                    if got == set(todo):
                        break
        if set(todo) - got:
            raise RuntimeError("metadata.pt not found in the archive for %s" % sorted(set(todo) - got))
    import torch
    fields = {}
    for s in splits:
        m = torch.load(root / s / "metadata.pt", map_location="cpu", weights_only=True)
        e0 = m[0]
        fields[s] = {"n_episodes": len(m), "keys": sorted(e0.keys()),
                     "shapes": {k: (list(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in e0.items()}}
    B.probe_output_dir(paths)
    B.write_json(paths, {"note": "metadata.pt fields (first episode of each split). 'proprio' is not a metadata field: ManifestDataset.get_frames returns zeros as proprio.", "splits": fields},
                 Path(paths["results"]) / "metadata_fields.json")
    return {"splits": splits, "fields": fields}


# ------------------------------------------------------------------------------------------------ pool
def stage_pool(cfg, cfg2, paths, ctx=None):
    """(E,13,196,384) fp16 encoder tokens of the 002 cache -> (E,13,384) float32 patch means (exp002 probes.pool, in chunks)."""
    from exp002 import probes as PR
    ctx = ctx or Ctx(cfg, cfg2, paths)
    tok = ctx.C["tokens_dino"]
    out = np.zeros((tok.shape[0], tok.shape[1], tok.shape[3]), np.float32)
    for s in range(0, tok.shape[0], 32):
        out[s:s + 32] = PR.pool(tok[s:s + 32])
    B.probe_output_dir(paths)
    B.write_npy(paths, Path(paths["cache"]) / "pooled_tokens.npy", out)
    log("pool: %s float32 written" % (out.shape,))
    return {"shape": list(out.shape)}


# ------------------------------------------------------------------------------------------------ labels for D
D_NAMES = ["mask_centroid_x", "mask_centroid_y", "mask_bbox_center_x", "mask_bbox_center_y",
           "cum_action_g1_x", "cum_action_g1_y", "cum_action_g1_z", "cum_action_g2_x", "cum_action_g2_y", "cum_action_g2_z"]


def mask_props(m):
    """bool (H,W) mask -> (cx, cy, bx, by) in pixel coordinates / width (x) or height (y); NaN when the mask is empty."""
    if not m.any():
        return [np.nan] * 4
    ys, xs = np.nonzero(m)
    h, w = m.shape
    return [xs.mean() / w, ys.mean() / h, (xs.min() + xs.max()) / 2.0 / w, (ys.min() + ys.max()) / 2.0 / h]


def cumulative_action(actions_raw, frameskip=5, n_frames=13):
    """actions_raw (E,>=60,6) -> (E,13,6): sum of the raw actions with index < frameskip*f (the actions applied between frame 0 and frame f)."""
    a = np.asarray(actions_raw, np.float64)
    c = np.concatenate([np.zeros((a.shape[0], 1, a.shape[2])), np.cumsum(a, axis=1)], axis=1)
    return c[:, [frameskip * f for f in range(n_frames)]].astype(np.float32)


def stage_labels(cfg, cfg2, paths, ctx=None, mask_fn=None):
    """Contrast-property labels for every cached episode and frame (E,13,10). Mask properties come from the 002 HSV mask rule on the cached frames
    (needs cv2); cumulative action from actions_raw. mask_fn(frame_u8)->bool mask can be injected (selftest)."""
    from exp002 import masks as MK
    ctx = ctx or Ctx(cfg, cfg2, paths)
    fr = ctx.C["frames_u8"]
    mfn = mask_fn or (lambda f: MK.cloth_mask(f, cfg2["mask"]))
    E, K = fr.shape[0], fr.shape[1]
    lab = np.full((E, K, len(D_NAMES)), np.nan, np.float32)
    t0 = time.time()
    for e in range(E):
        frames = np.asarray(fr[e])
        for f in range(K):
            lab[e, f, :4] = mask_props(mfn(frames[f]))
        if e % 200 == 0:
            log("labels %d/%d (%.0fs)" % (e, E, time.time() - t0))
    lab[:, :, 4:] = cumulative_action(ctx.C["actions_raw"], int(cfg2["data_impl"]["frameskip"]), K)
    B.probe_output_dir(paths)
    B.write_npz(paths, Path(paths["cache"]) / "d_labels.npz", labels=lab, names=np.array(D_NAMES))
    n_nan = int(np.isnan(lab[:, :, :4]).any(-1).sum())
    log("labels: (E,K,P)=%s ; frames with an empty mask: %d" % (lab.shape, n_nan))
    return {"shape": list(lab.shape), "names": D_NAMES, "n_frames_empty_mask": n_nan}


# ------------------------------------------------------------------------------------------------ transfer (probe fit once, applied to many sets)
def build_apply_sets(ctx, pooled, extracts, cfg):
    """{name: (X (Ete,1,384), y (Ete,))} for encoder, persistence and every extracted rollout / untrained rollout."""
    sets = {}
    for k in ctx.steps:
        y = ctx.label_te(k)
        sets[n_enc(k)] = (pooled[ctx.Etr:, ctx.t + k][:, None, :], y)
        sets[n_per(k)] = (pooled[ctx.Etr:, ctx.t][:, None, :], y)
    for run, ex in extracts.items():
        for k in ctx.steps:
            y = ctx.label_te(k)
            sets[n_roll(run, k)] = (ex["roll"][ctx.Etr:, k - 1][:, None, :], y)
            if ex.get("tf") is not None:
                sets[n_tf(run, k)] = (ex["tf"][ctx.Etr:, k - 1][:, None, :], y)
            for (v, sd), arr in ex["untrained"].items():
                sets[n_unt(run, v, sd, k)] = (arr[:, k - 1][:, None, :], y)
    return sets


def stage_transfer(cfg, cfg2, paths, ctx=None, extracts=None, pooled=None, runs=None, out_name="transfer"):
    """The 002 output-space measurement, recomputed: the Q1 linear probe is fitted ONLY on encoder tokens of the probe-training rows (frames 0..11, label at the
    same frame) by exp002 evalgrid.fit_apply_cells (same function, same seeds), then applied unchanged to every set. Persistence = encoder tokens of the context frame
    t copied to every step. Saves per-episode predictions (P, Ete) per set."""
    ctx = ctx or Ctx(cfg, cfg2, paths)
    pooled = pooled if pooled is not None else load_pooled(paths)
    if extracts is None:
        extracts = {}
        for _, _, run in (runs or B.all_runs(cfg)):
            extracts[run] = load_extract(paths, run)
    sets = build_apply_sets(ctx, pooled, extracts, cfg)
    Xtr = pooled[:ctx.Etr][:, :, None, :]                                    # (Etr,13,1,384): patch axis of length 1 -> probes.pool is the identity
    t0 = time.time()
    cells, pm, dig = ctx.EG.fit_apply_cells(cfg2, "linear_pooled", Xtr, ctx.Yp[:ctx.Etr], ctx.frames_tr, sets, ctx.probe_seeds)
    log("transfer: %d sets, probe seeds %s, %.0fs" % (len(sets), ctx.probe_seeds, time.time() - t0))
    res = {"n_sets": len(sets), "probe_seeds": ctx.probe_seeds, "prediction_digest_first_set": {str(k): v for k, v in dig.items()},
           "train_frames": ctx.frames_tr, "probe_train_rows": ctx.Etr, "path": "exp002 evalgrid.fit_apply_cells(kind=linear_pooled) on patch-mean features"}
    if paths is not None and out_name:
        B.probe_output_dir(paths)
        B.write_npz(paths, Path(paths["cache"]) / (out_name + "_pred.npz"), **{n: np.asarray(v, np.float64) for n, v in pm.items()})
        B.write_json(paths, dict(res, sets=sorted(pm)), Path(paths["parts"]) / (out_name + "_meta.json"))
    return res, pm
