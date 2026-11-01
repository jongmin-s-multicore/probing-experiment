"""GPU stage: run the trained world model on evaluation episodes and save everything the CPU stages need.

Stages measured per episode (frames are macro steps of frameskip 5: 0,5,...,60 -> K=13 frames, 12 actions):
  (a) real frame                      -> cloth pixel area  (area_real)
  (b) decode(encode(real))            -> area_recon        (no predictor)
  (c) decode(predictor(GT z_t, a_t))  -> area_tf1          (teacher-forced 1-step, frames 1..12)
  (d) decode(open-loop z_k)           -> area_ol           (rollout from frame 0 with all actions, frames 1..12)
Latents saved (decoder-free analysis): patch-level fp16 memmaps for gt / tf1 / ol, pooled fp32 for probes, predictor
block hidden states for the teacher-forced pass, random-init ViT features and a raw-pixel grid as controls.

Model code is imported from the read-only acwm-phys checkout (config code.commit). Requires a CUDA GPU: the branch's
ViT predictor builds its attention mask with .to('cuda').
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .common import log, read_json, write_json, set_seed, run_dir, size_roles
from .data import build_eval_dataset, eval_episode_table
from .masks import frames_area, patch_coverage

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


def pool_feats(z, mode):
    """z (..., 196, D) -> (..., F). mean: global mean; grid2: 2x2 grid means of the 14x14 patch grid."""
    import torch
    if mode == "mean":
        return z.mean(-2)
    if mode.startswith("grid"):
        g = int(mode[4:])
        lead = z.shape[:-2]
        zz = z.reshape(*lead, 14, 14, z.shape[-1])
        s = 14 // g
        zz = zz.reshape(*lead, g, s, g, s, z.shape[-1]).mean((-4, -2))     # (..., g, g, D)
        return zz.reshape(*lead, g * g * z.shape[-1])
    raise ValueError(mode)


def to_u8(x):
    """(N,3,H,W) in [-1,1] -> (N,H,W,3) uint8 (same conversion as branch eval_acwm_phys_quant.py)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


class LayerCapture:
    """Residual-stream capture at the input of each predictor block (h0..h5) and before the final norm (h6)."""

    def __init__(self, predictor):
        self.store = {}
        self.handles = []
        tr = predictor.transformer
        for i, (attn, _ff) in enumerate(tr.layers):
            self.handles.append(attn.register_forward_pre_hook(self._hook(f"h{i}")))
        self.handles.append(tr.norm.register_forward_pre_hook(self._hook(f"h{len(tr.layers)}")))

    def _hook(self, name):
        def fn(_mod, args):
            self.store[name] = args[0].detach()
        return fn

    def clear(self):
        self.store = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def episode_inputs(ds, i, device):
    """Frames at macro steps and action groups for dataset episode i."""
    import torch
    obs, _ = ds.get_frames(i, range(0, F * (K - 1) + 1, F))
    assert ds.n_obs[i] >= F * (K - 1) + 1, f"{ds.video_names[i]}: only {ds.n_obs[i]} frames"
    a = ds.actions[i][: F * N_ACT]
    assert a.shape[0] == F * N_ACT, f"{ds.video_names[i]}: only {a.shape[0]} actions"
    acts = a.reshape(N_ACT, -1)                          # (12, 30): same as '(n f) d -> n (f d)'
    acts = torch.cat([acts, torch.zeros(1, acts.shape[1])], 0)   # frame 12 has no following action (unused)
    return obs["visual"], acts


def run_batch(model, vis, acts, pools, cap):
    """vis (B,K,3,H,W), acts (B,K,30) on device. Returns dict of tensors (no grad)."""
    import torch
    B = vis.shape[0]
    with torch.no_grad():
        z_all = model.encode({"visual": vis, "proprio": torch.zeros(B, K, 1, device=vis.device)}, acts)        # (B,K,196,404)
        D = model.encoder.emb_dim                                                                              # 384 visual dims come first
        zv = z_all[..., :D]
        # (c) teacher-forced 1-step: input GT z_t with a_t, predict z_{t+1}
        z_in = z_all[:, : K - 1].reshape(B * (K - 1), 1, z_all.shape[2], z_all.shape[3])
        cap.clear()
        zp = model.predict(z_in)
        tf1 = zp[..., :D].reshape(B, K - 1, -1, D)
        hidden = {k: v[..., :D].reshape(B, K - 1, -1, D) for k, v in cap.store.items()}
        # (d) open loop from frame 0 (replicates VWorldModel._rollout)
        z = z_all[:, :1]
        for j in range(1, K - 1):
            zp1 = model.predict(z[:, -1:])
            z_new = model.replace_actions_from_z(zp1[:, -1:].clone(), acts[:, j : j + 1])
            z = torch.cat([z, z_new], 1)
        zp1 = model.predict(z[:, -1:])
        z = torch.cat([z, zp1[:, -1:]], 1)
        ol = z[:, 1:, :, :D]                                                                                  # frames 1..12
        out = {"gt": zv, "tf1": tf1, "ol": ol}
        # decode: (b) recon of GT, (c) tf1, (d) ol
        dec = {}
        for name, lat in (("recon", zv), ("tf1", tf1), ("ol", ol)):
            n = lat.shape[1]
            img, _ = model.decoder(lat)
            dec[name] = to_u8(img.reshape(B * n, *img.shape[1:]))
        pooled = {}
        for pm in pools:
            for name in ("gt", "tf1", "ol"):
                pooled[f"{pm}/{name}"] = pool_feats(out[name], pm).float().cpu().numpy()
            for hk, hv in hidden.items():
                pooled[f"{pm}/{hk}"] = pool_feats(hv, pm).float().cpu().numpy()
    return out, dec, pooled


def random_vit_features(model, vis, pools, rvit):
    import torch
    B = vis.shape[0]
    with torch.no_grad():
        x = model.encoder_transform(vis.reshape(B * K, *vis.shape[2:]))
        e = rvit.forward_features(x)["x_norm_patchtokens"].reshape(B, K, -1, 384)
        return {f"{pm}/rvit": pool_feats(e, pm).float().cpu().numpy() for pm in pools}


def raw_pixel_grid(vis):
    """(B,K,3,224,224) -> (B,K,588): mean RGB over 16x16 cells (14x14 grid)."""
    import torch
    B = vis.shape[0]
    g = torch.nn.functional.avg_pool2d(vis.reshape(B * K, *vis.shape[2:]).float(), 16)
    return g.reshape(B, K, -1).cpu().numpy()


def run(cfg, paths, run_name, smoke_max_eps=None, epoch=None, overwrite=False, groups=None, cache_suffix="", max_per_group=None, light=False):
    """groups: only these size groups (repro_check uses a subset); cache_suffix: separate cache directory (e.g. "_repro")."""
    import torch
    set_seed(cfg["seed"])
    dcfg = cfg["diagnose"]
    device = torch.device(dcfg["device"])
    rdir = run_dir(paths, run_name)
    cache = paths["cache"] / (run_name + cache_suffix)
    if (cache / "DONE").exists() and not overwrite:
        log(f"extract: {cache}/DONE exists, skip")
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    model, tcfg, ep_loaded = load_world_model(paths["acwm_src"], rdir, device, epoch)
    log(f"loaded {run_name} epoch {ep_loaded}; num_hist={tcfg.num_hist} frameskip={tcfg.frameskip}")
    assert tcfg.num_hist == 1 and tcfg.frameskip == F, "extract assumes num_hist=1, frameskip=5 (config.train.hydra_overrides)"

    listing = read_json(paths["manifests"] / "listing.json")
    table = eval_episode_table(listing, cfg, paths["data_root"])
    meta_p = paths["manifests"] / "episode_meta.json"
    ep_meta = read_json(meta_p) if meta_p.exists() else {}
    train_manifest = rdir / "train_manifest.json"
    train_keys = {(e["split"], e["name"]) for e in read_json(train_manifest)["episodes"]}
    maxn = smoke_max_eps if smoke_max_eps is not None else (max_per_group if max_per_group is not None else dcfg["max_episodes_per_group"])
    tm = read_json(train_manifest)
    train_groups = sorted(tm["group_counts"].keys())
    roles = size_roles(cfg, train_groups)
    keep, cnt = [], {}
    for r in table:
        if r["split"] not in dcfg["eval_sets"] or (groups and r["group"] not in groups):
            continue
        if maxn and cnt.get(r["group"], 0) >= maxn:
            continue
        cnt[r["group"]] = cnt.get(r["group"], 0) + 1
        r = dict(r)
        r["in_train"] = (r["split"], r["name"]) in train_keys
        m_ = ep_meta.get(r["split"], {}).get(r["name"], {})
        r["particles"] = (m_["cloth_dimx"] * m_["cloth_dimy"]) if m_.get("cloth_dimx") and m_.get("cloth_dimy") else None
        r["role"] = roles.get(r["group"])
        keep.append(r)
    E = len(keep)
    log(f"episodes to extract: {E}  groups: {cnt}")

    import hydra
    tf = hydra.utils.instantiate(tcfg.env.dataset.transform)
    ds = build_eval_dataset(keep, train_manifest, tf, tcfg.normalize_action, list(tcfg.env.dataset.action_dims))
    mcfg = cfg["mask"]
    pools = dcfg["pools"]
    cap = LayerCapture(model.predictor)

    rvit = None
    if dcfg["random_vit_control"] and not light:
        torch.manual_seed(dcfg["random_vit_seed"])
        rvit = torch.hub.load("facebookresearch/dinov2:85a24602099d397264d5b30461ad7f3bfd726ca1", "dinov2_vits14", pretrained=False).to(device).eval()

    P, D = 196, 384
    mm = {n: np.lib.format.open_memmap(cache / f"patch_{n}.npy", mode="w+", dtype=np.float16, shape=(E, k, P, D))
          for n, k in (("gt", K), ("tf1", K - 1), ("ol", K - 1))}
    scal = {"area_real": np.zeros((E, K), np.int64), "area_recon": np.zeros((E, K), np.int64),
            "area_tf1": np.zeros((E, K - 1), np.int64), "area_ol": np.zeros((E, K - 1), np.int64),
            "cov_real": np.zeros((E, K, P), np.float16)}
    pooled_all, pix_all = {}, np.zeros((E, K, 588), np.float32)
    n_save = dcfg["save_decoded_frames"]
    saved_per_group, sample = {}, {}
    bs = dcfg["batch_episodes"]
    for s in range(0, E, bs):
        idx = list(range(s, min(E, s + bs)))
        pairs = [episode_inputs(ds, i, device) for i in idx]
        vis = torch.stack([p[0] for p in pairs]).to(device)
        acts = torch.stack([p[1] for p in pairs]).to(device)
        out, dec, pooled = run_batch(model, vis, acts, pools, cap)
        if rvit is not None:
            pooled.update(random_vit_features(model, vis, pools, rvit))
        pix_all[idx[0] : idx[-1] + 1] = raw_pixel_grid(vis)
        for n in ("gt", "tf1", "ol"):
            mm[n][idx[0] : idx[-1] + 1] = out[n].half().cpu().numpy()
        real_u8 = to_u8(vis.reshape(-1, *vis.shape[2:]))
        for b, i in enumerate(idx):
            scal["area_real"][i] = frames_area(real_u8[b * K : (b + 1) * K], mcfg)
            scal["cov_real"][i] = patch_coverage(real_u8[b * K : (b + 1) * K], mcfg)
            scal["area_recon"][i] = frames_area(dec["recon"][b * K : (b + 1) * K], mcfg)
            scal["area_tf1"][i] = frames_area(dec["tf1"][b * (K - 1) : (b + 1) * (K - 1)], mcfg)
            scal["area_ol"][i] = frames_area(dec["ol"][b * (K - 1) : (b + 1) * (K - 1)], mcfg)
            g = keep[i]["group"]
            if saved_per_group.get(g, 0) < n_save:
                saved_per_group[g] = saved_per_group.get(g, 0) + 1
                sample[f"{g}/{saved_per_group[g]}/real"] = real_u8[b * K : (b + 1) * K]
                sample[f"{g}/{saved_per_group[g]}/recon"] = dec["recon"][b * K : (b + 1) * K]
                sample[f"{g}/{saved_per_group[g]}/tf1"] = dec["tf1"][b * (K - 1) : (b + 1) * (K - 1)]
                sample[f"{g}/{saved_per_group[g]}/ol"] = dec["ol"][b * (K - 1) : (b + 1) * (K - 1)]
        for k, v in pooled.items():
            if k not in pooled_all:
                pooled_all[k] = np.zeros((E,) + v.shape[1:], np.float32)
            pooled_all[k][idx[0] : idx[-1] + 1] = v
        log(f"extracted {idx[-1] + 1}/{E}")
    for m in mm.values():
        m.flush()
    cap.remove()
    np.savez_compressed(cache / "scalars.npz", **scal)
    np.savez_compressed(cache / "pix_grid.npz", pix=pix_all)
    for pm in pools:
        np.savez_compressed(cache / f"pooled_{pm}.npz", **{k.split("/", 1)[1]: v for k, v in pooled_all.items() if k.startswith(pm + "/")})
    if sample:
        np.savez_compressed(cache / "frames_sample.npz", **sample)
    write_json({"run_name": run_name, "epoch_loaded": ep_loaded, "episodes": keep, "n_episodes": E, "K": K, "pools": pools,
                "code_commit": cfg["code"]["commit"], "dataset": cfg["data"]["active"], "train_groups": train_groups, "roles": roles,
                "reference_group": cfg["diagnose"]["reference_group"], "groups_filter": groups}, cache / "episodes.json")
    (cache / "DONE").write_text("ok\n")
    return cache
