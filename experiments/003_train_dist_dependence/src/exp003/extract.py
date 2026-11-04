"""GPU stage (slim version of 001 extract.py): run the trained world model on the test episodes and save only what 003 needs.

Per episode (frames are macro steps of frameskip 5: 0,5,...,60 -> K=13 frames, 12 actions):
  area_real (E,13)  cloth pixel area of the real frames (224x224 mask)
  area_ol   (E,12)  cloth pixel area of decode(open-loop z_k), k=1..12, rollout from frame 0 with all actions (same code path as 001)
No latents, no teacher-forced pass, no probes/controls are stored (001 stored them; 003 does not use them).
Model code is imported from the read-only acwm-phys checkout (config code.commit). Requires a CUDA GPU (branch ViT predictor uses .to('cuda')).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .common import log, read_json, write_json, set_seed, run_dir, size_roles
from .data import build_eval_dataset, eval_episode_table
from .masks import frames_area

K = 13           # macro frames 0..12  (65 frames, frameskip 5)
N_ACT = 12       # action groups of 5 steps
F = 5


def add_branch_to_path(acwm_src):
    sys.path.insert(0, str(acwm_src))


def load_world_model(acwm_src, rdir, device, epoch=None):
    """Same assembly as branch plan.load_model, without importing plan.py (it pulls in env/planner code)."""
    import torch
    import hydra
    from omegaconf import OmegaConf
    add_branch_to_path(acwm_src)
    cfg = OmegaConf.load(Path(rdir) / "hydra.yaml")
    ck = Path(rdir) / "checkpoints" / (f"model_{epoch}.pth" if epoch else "model_latest.pth")
    # trusted: only reads checkpoints we trained ourselves (torch>=2.6 defaults to weights_only=True, which fails on pickled modules)
    payload = torch.load(ck, map_location=device, weights_only=False)
    # the frozen DINOv2 encoder is not saved in the checkpoint (train.py saves it only if train_encoder): rebuild it like plan.load_model
    parts = {k: payload[k].to(device) for k in ["predictor", "decoder", "proprio_encoder", "action_encoder"]}
    parts["encoder"] = (payload["encoder"] if "encoder" in payload else hydra.utils.instantiate(cfg.encoder)).to(device)
    model = hydra.utils.instantiate(
        cfg.model, encoder=parts["encoder"], proprio_encoder=parts["proprio_encoder"], action_encoder=parts["action_encoder"],
        predictor=parts["predictor"], decoder=parts["decoder"], proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat,
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, int(payload["epoch"])


def to_u8(x):
    """(N,3,H,W) in [-1,1] -> (N,H,W,3) uint8 (same conversion as branch eval_acwm_phys_quant.py)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def episode_inputs(ds, i, device):
    """Frames at macro steps and action groups for dataset episode i."""
    import torch
    obs, _ = ds.get_frames(i, range(0, F * (K - 1) + 1, F))
    assert ds.n_obs[i] >= F * (K - 1) + 1, f"{ds.video_names[i]}: only {ds.n_obs[i]} frames"
    a = ds.actions[i][: F * N_ACT]
    assert a.shape[0] == F * N_ACT, f"{ds.video_names[i]}: only {a.shape[0]} actions"
    acts = a.reshape(N_ACT, -1)
    acts = torch.cat([acts, torch.zeros(1, acts.shape[1])], 0)
    return obs["visual"], acts


def run_batch(model, vis, acts):
    """vis (B,K,3,H,W), acts (B,K,30) on device. Returns decoded open-loop frames 1..12 as uint8 (B*12,H,W,3)."""
    import torch
    B = vis.shape[0]
    with torch.no_grad():
        z_all = model.encode({"visual": vis, "proprio": torch.zeros(B, K, 1, device=vis.device)}, acts)
        D = model.encoder.emb_dim
        z = z_all[:, :1]
        for j in range(1, K - 1):
            zp1 = model.predict(z[:, -1:])
            z_new = model.replace_actions_from_z(zp1[:, -1:].clone(), acts[:, j : j + 1])
            z = torch.cat([z, z_new], 1)
        zp1 = model.predict(z[:, -1:])
        z = torch.cat([z, zp1[:, -1:]], 1)
        ol = z[:, 1:, :, :D]
        img, _ = model.decoder(ol)
        return to_u8(img.reshape(B * (K - 1), *img.shape[1:]))


def run(cfg, paths, run_name, smoke_max_eps=None, epoch=None, overwrite=False):
    """Writes $EXP003_CACHE/<run_name>/{scalars.npz, episodes.json, frames_sample.npz, DONE}."""
    import torch
    set_seed(cfg["seed"])
    dcfg = cfg["diagnose"]
    device = torch.device(dcfg["device"])
    rdir = run_dir(paths, run_name)
    cache = paths["cache"] / run_name
    if (cache / "DONE").exists() and not overwrite:
        log(f"extract: {cache}/DONE exists, skip")
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, tcfg, ep_loaded = load_world_model(paths["acwm_src"], rdir, device, epoch)
    log(f"loaded {run_name} epoch {ep_loaded}; num_hist={tcfg.num_hist} frameskip={tcfg.frameskip}")
    assert tcfg.num_hist == 1 and tcfg.frameskip == F, "extract assumes num_hist=1, frameskip=5 (config.train.hydra_overrides)"

    listing = read_json(paths["manifests"] / "listing.json")
    table = eval_episode_table(listing, cfg, paths["data_root"])
    train_manifest = rdir / "train_manifest.json"
    tm = read_json(train_manifest)
    train_keys = {(e["split"], e["name"]) for e in tm["episodes"]}
    train_groups = sorted(tm["group_counts"].keys())
    roles = size_roles(cfg, train_groups)
    maxn = smoke_max_eps if smoke_max_eps is not None else dcfg["max_episodes_per_group"]
    keep, cnt = [], {}
    for r in table:
        if r["split"] not in dcfg["eval_sets"]:
            continue
        if maxn and cnt.get(r["group"], 0) >= maxn:
            continue
        cnt[r["group"]] = cnt.get(r["group"], 0) + 1
        r = dict(r)
        r["in_train"] = (r["split"], r["name"]) in train_keys
        r["role"] = roles.get(r["group"])
        keep.append(r)
    assert not any(r["in_train"] for r in keep), "an evaluation episode is also a training episode"
    E = len(keep)
    log(f"episodes to extract: {E}  groups: {cnt}")

    import hydra
    tf = hydra.utils.instantiate(tcfg.env.dataset.transform)
    ds = build_eval_dataset(keep, train_manifest, tf, tcfg.normalize_action, list(tcfg.env.dataset.action_dims))
    mcfg = cfg["mask"]
    area_real = np.zeros((E, K), np.int64)
    area_ol = np.zeros((E, K - 1), np.int64)
    n_save, saved, sample = dcfg["save_decoded_frames"], {}, {}
    bs = dcfg["batch_episodes"]
    for s in range(0, E, bs):
        idx = list(range(s, min(E, s + bs)))
        pairs = [episode_inputs(ds, i, device) for i in idx]
        vis = torch.stack([p[0] for p in pairs]).to(device)
        acts = torch.stack([p[1] for p in pairs]).to(device)
        dec = run_batch(model, vis, acts)
        real_u8 = to_u8(vis.reshape(-1, *vis.shape[2:]))
        for b, i in enumerate(idx):
            area_real[i] = frames_area(real_u8[b * K : (b + 1) * K], mcfg)
            area_ol[i] = frames_area(dec[b * (K - 1) : (b + 1) * (K - 1)], mcfg)
            g = keep[i]["group"]
            if saved.get(g, 0) < n_save:
                saved[g] = saved.get(g, 0) + 1
                sample[f"{g}/{saved[g]}/real"] = real_u8[b * K : (b + 1) * K]
                sample[f"{g}/{saved[g]}/ol"] = dec[b * (K - 1) : (b + 1) * (K - 1)]
        log(f"extracted {idx[-1] + 1}/{E}")
    np.savez_compressed(cache / "scalars.npz", area_real=area_real, area_ol=area_ol)
    if sample:
        np.savez_compressed(cache / "frames_sample.npz", **sample)
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else None
    write_json({"run_name": run_name, "epoch_loaded": ep_loaded, "episodes": keep, "n_episodes": E, "K": K, "code_commit": cfg["code"]["commit"],
                "dataset": cfg["data"]["active"], "train_groups": train_groups, "roles": roles, "reference_group": dcfg["reference_group"],
                "extract_cuda_peak_allocated_gib": peak, "batch_episodes": bs}, cache / "episodes.json")
    (cache / "DONE").write_text("ok\n")
    return cache
