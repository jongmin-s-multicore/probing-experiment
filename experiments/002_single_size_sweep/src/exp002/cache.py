"""Q1 cache (GPU, needs ONE trained checkpoint only for its frozen DINOv2 encoder + encoder_transform):
for the probe-training episodes (3 levels x 300) and the evaluation episodes (9 levels x 100) store
  frames_u8.npy   (E,13,224,224,3) uint8   decoded + resized (branch transform) frames; Q2/Q3 read these instead of decoding mp4 again
  tokens_dino.npy (E,13,196,384) fp16     encoder patch tokens computed FROM THE STORED uint8 frames
  tokens_rand.npy (E,13,196,384) fp16     random-init DINOv2 ViT-S/14 tokens (control), same input
  mask_feat.npy   (E,13,3) float32        [mask area px, bbox w, bbox h] of the 224x224 frame (HSV rule, config.mask)
  actions_raw.npy (E,60,6) float32        raw (un-normalized) actions of config data_impl.action_dims
  episodes.json, meta.json (encoder fingerprint, source run), DONE"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .common import log, read_json, write_json, run_dir, require
from . import model_io as M
from .masks import frames_mask_features

K = 13


def cache_dir(paths, suffix=""):
    return paths["cache"] / ("q1_tokens" + suffix)


def disk_estimate_gb(n_eps):
    frames = n_eps * K * 224 * 224 * 3
    tok = n_eps * K * 196 * 384 * 2
    other = n_eps * (K * 3 * 4 + 60 * 6 * 4)
    return {"frames_u8_gb": frames / 1e9, "tokens_dino_gb": tok / 1e9, "tokens_rand_gb": tok / 1e9, "total_gb": (frames + 2 * tok + other) / 1e9}


def run(cfg, paths, run_name, max_eps=None, suffix="", overwrite=False):
    import torch
    import hydra
    from .data import make_dataset_class
    cd = cache_dir(paths, suffix)
    if (cd / "DONE").exists() and not overwrite:
        log("cache: %s/DONE exists, skip" % cd)
        return cd
    cd.mkdir(parents=True, exist_ok=True)
    rows = read_json(paths["manifests"] / "cache_episodes.json")
    if max_eps:
        rows = [r for r in rows if r["rank_in_level"] < max_eps]
        for i, r in enumerate(rows):
            r["id"] = i
    E = len(rows)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rdir = run_dir(paths, run_name)
    model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
    fp = M.encoder_fingerprint(model.encoder)
    log("cache from run %s epoch %d; encoder fingerprint %s; episodes %d; disk %s" % (run_name, ep, fp[:12], E, disk_estimate_gb(E)))
    tf = hydra.utils.instantiate(tcfg.env.dataset.transform)
    ds = make_dataset_class()(rows, transform=tf, normalize_action=False, action_dims=list(cfg["data_impl"]["action_dims"]))
    pc = cfg["probe_impl"]["random_init"]
    torch.manual_seed(int(pc["seed"]))
    rnd = torch.hub.load(pc["hub_repo"], pc["hub_name"], pretrained=False).to(dev).eval()
    mm = {n: np.lib.format.open_memmap(cd / (n + ".npy"), mode="w+", dtype=dt, shape=sh) for n, dt, sh in (
        ("frames_u8", np.uint8, (E, K, 224, 224, 3)), ("tokens_dino", np.float16, (E, K, 196, 384)), ("tokens_rand", np.float16, (E, K, 196, 384)),
        ("mask_feat", np.float32, (E, K, 3)), ("actions_raw", np.float32, (E, 60, 6)))}
    bs = int(cfg["q2_impl"]["batch_episodes"])
    t0 = time.time()
    for s in range(0, E, bs):
        idx = list(range(s, min(E, s + bs)))
        vis_f = []
        for i in idx:
            obs, _ = ds.get_frames(i, range(0, 5 * (K - 1) + 1, 5))
            assert ds.n_obs[i] >= 5 * (K - 1) + 1, "%s: only %d frames" % (ds.video_names[i], ds.n_obs[i])
            vis_f.append(obs["visual"])
            mm["actions_raw"][i] = ds.raw_actions[i][:60].numpy()
        vis = torch.stack(vis_f)                                     # (B,K,3,224,224) in [-1,1] (branch transform)
        u8 = np.stack([M.to_u8(v) for v in vis])                     # (B,K,224,224,3)
        mm["frames_u8"][s:s + len(idx)] = u8
        for b, i in enumerate(idx):
            mm["mask_feat"][i] = frames_mask_features(u8[b], cfg["mask"]).astype(np.float32)
        x = M.frames_to_float(u8).to(dev)                            # tokens are computed from the stored uint8 frames
        with torch.no_grad():
            z = M.encode_batch(model, x, torch.zeros(len(idx), K, 30, device=dev))[..., :M.D_VIS]
            mm["tokens_dino"][s:s + len(idx)] = z.half().cpu().numpy()
            xr = model.encoder_transform(x.reshape(-1, *x.shape[2:]))
            r = rnd.forward_features(xr)["x_norm_patchtokens"].reshape(len(idx), K, -1, M.D_VIS)
            mm["tokens_rand"][s:s + len(idx)] = r.half().cpu().numpy()
        if s == 0:
            log("first batch: vis range [%.3f, %.3f]; u8->float round trip max err %.4f (limit %.4f)" % (
                float(vis.min()), float(vis.max()), float((M.frames_to_float(u8) - vis).abs().max()), 1 / 127.5))
        log("cached %d/%d (%.0fs)" % (idx[-1] + 1, E, time.time() - t0))
    for m in mm.values():
        m.flush()
    write_json(rows, cd / "episodes.json")
    write_json({"source_run": run_name, "epoch": ep, "encoder_fingerprint": fp, "K": K, "n_episodes": E, "max_eps": max_eps,
                "random_init": pc, "elapsed_s": time.time() - t0}, cd / "meta.json")
    (cd / "DONE").write_text("ok\n")
    return cd


def load(paths, suffix=""):
    """Returns dict of memmaps + rows + meta (read-only)."""
    cd = cache_dir(paths, suffix)
    if not (cd / "DONE").exists():
        raise FileNotFoundError("%s/DONE missing - run `python run.py cache` first" % cd)
    out = {n: np.load(cd / (n + ".npy"), mmap_mode="r") for n in ("frames_u8", "tokens_dino", "tokens_rand", "mask_feat", "actions_raw")}
    out["rows"] = read_json(cd / "episodes.json")
    out["meta"] = read_json(cd / "meta.json")
    return out
