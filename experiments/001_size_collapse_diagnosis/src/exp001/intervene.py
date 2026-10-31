"""GPU stage: size-direction intervention on predictor inputs (candidate E, "is the size information used?").

direction u = unit ridge direction of log(real cloth area) in the (un-standardised) mean-pooled GT-encoding space
(results/<run>/size_direction.npz, written by the probe stage). The input visual latent (196 x 384) of a world-model step is
modified along u; then the predictor output is read out along u (pooled projection) and decoded (cloth pixel area).
Modes
  remove        z_p <- z_p - (z_p . u) u                        for every patch p (projection removed)
  set_<A>_to_<B> shift all patches by delta*u so that the pooled projection equals the mean pooled projection of group B at the
                 same frame (episodes of group A are modified); B=reference group is the training cloth size
Controls: n_random unit directions r; the same shift magnitude |delta| (or the same removed fraction for 'remove') is applied along r
and the response is read along r (and the decoded area change is measured). Teacher-forced (one step, every input frame) and
open-loop (only the first latent modified, rollout continues). Reported per episode; aggregation is done in aggregate.py.
Not interpreted here.
"""
from __future__ import annotations

import numpy as np

from .common import log, read_json, write_json, set_seed, run_dir
from .data import build_eval_dataset
from .extract import load_world_model, episode_inputs, to_u8, K, N_ACT
from .masks import frames_area


def _proj(z, u):
    """pooled projection: mean over patches of z . u   z (..., P, D)"""
    return (z * u).sum(-1).mean(-1)


def _apply(z, u, mode_kind, target_proj=None):
    """z (..., P, D). Returns modified latent and applied pooled shift (delta)."""
    cur = _proj(z, u)
    if mode_kind == "remove":
        zm = z - (z * u).sum(-1, keepdim=True) * u
    else:
        delta = (target_proj - cur)
        zm = z + delta[..., None, None] * u
    return zm, _proj(zm, u) - cur


def run(cfg, paths, run_name):
    import torch
    icfg = cfg["intervene"]
    dev = torch.device(cfg["diagnose"]["device"])
    set_seed(icfg["seed"])
    cache = paths["cache"] / run_name
    rdir = run_dir(paths, run_name)
    res_dir = paths["results_dir"] / run_name
    dirz = np.load(res_dir / "size_direction.npz")
    model, tcfg, ep_loaded = load_world_model(paths["acwm_src"], rdir, dev)
    epi = read_json(cache / "episodes.json")
    keep = [e for e in epi["episodes"] if not e["in_train"]]
    groups = sorted({e["group"] for e in keep})
    mx = icfg["max_episodes_per_group"]
    if mx:
        cnt, k2 = {}, []
        for e in keep:
            if cnt.get(e["group"], 0) < mx:
                cnt[e["group"]] = cnt.get(e["group"], 0) + 1
                k2.append(e)
        keep = k2
    import hydra
    tf = hydra.utils.instantiate(tcfg.env.dataset.transform)
    ds = build_eval_dataset(keep, rdir / "train_manifest.json", tf, tcfg.normalize_action, list(tcfg.env.dataset.action_dims))
    u = torch.tensor(dirz["u"], dtype=torch.float32, device=dev)
    g = torch.Generator(device="cpu").manual_seed(icfg["seed"])
    rnd = torch.randn(icfg["n_random_directions"], u.numel(), generator=g)
    rnd = (rnd / rnd.norm(dim=1, keepdim=True)).to(dev)
    # group-mean pooled projection per frame from GT encodings (mean pooling is exact for the projection)
    gt_pool = np.load(cache / "pooled_mean.npz")["gt"]
    idx_keep = {(e["split"], e["name"]): i for i, e in enumerate(epi["episodes"])}
    rows = np.array([idx_keep[(e["split"], e["name"])] for e in keep])
    proj_gt = gt_pool[rows] @ dirz["u"]                      # (E,13)
    gnames = np.array([e["group"] for e in keep])
    gmean = {gg: torch.tensor(proj_gt[gnames == gg].mean(0), device=dev, dtype=torch.float32) for gg in groups}
    mcfg = cfg["mask"]
    ref = cfg["diagnose"]["reference_group"]        # the training-size group (baseline); 'set_ind_mean' moves each group to it
    modes = [("remove", None, None)]
    if "set_ind_mean" in icfg["modes"] and ref in groups:
        modes += [(f"set_{a}_to_{ref}", a, ref) for a in groups if a != ref]
    if "set_group_swap" in icfg["modes"]:
        modes += [(f"set_{a}_to_{b}", a, b) for a, b in icfg["swap_pairs"] if a in groups and b in groups and a != b]
    modes = [m for i, m in enumerate(modes) if m not in modes[:i]]
    out = {"run_name": run_name, "epoch_loaded": ep_loaded, "modes": [m[0] for m in modes], "groups": groups, "episodes": [], "per_mode": {}}
    n_r = rnd.shape[0]
    dec_frames = [0, 5, 11]
    for mode, src, tgt in modes:
        out["per_mode"][mode] = []
    bs = 4
    for s0 in range(0, len(keep), bs):
        idx = list(range(s0, min(len(keep), s0 + bs)))
        pairs = [episode_inputs(ds, i, dev) for i in idx]
        vis = torch.stack([p[0] for p in pairs]).to(dev)
        acts = torch.stack([p[1] for p in pairs]).to(dev)
        B = len(idx)
        with torch.no_grad():
            z_all = model.encode({"visual": vis, "proprio": torch.zeros(B, K, 1, device=dev)}, acts)    # (B,K,196,404)
            D = model.encoder.emb_dim
            base_in = z_all[:, : K - 1]                                                                  # (B,12,196,404)
            flat = lambda x: x.reshape(B * (K - 1), 1, *x.shape[2:])

            def tf_predict(zin):
                zz = base_in.clone()
                zz[..., :D] = zin
                return model.predict(flat(zz))[..., :D].reshape(B, K - 1, -1, D)

            p0 = tf_predict(base_in[..., :D])
            area0 = _decode_area(model, p0[:, dec_frames], mcfg)                                         # (B,3)
            for mode, src, tgt in modes:
                for bi, i in enumerate(idx):
                    if src is not None and keep[i]["group"] != src:
                        continue
                    zin = base_in[bi : bi + 1, ..., :D]                                                  # (1,12,196,384)
                    if mode == "remove":
                        zm, shift = _apply(zin, u, "remove")
                    else:
                        zm, shift = _apply(zin, u, "set", gmean[tgt][: K - 1][None])
                    # teacher-forced
                    zb = base_in[bi : bi + 1].clone(); zb[..., :D] = zm
                    pm = model.predict(zb.reshape(K - 1, 1, *zb.shape[2:]))[..., :D].reshape(1, K - 1, -1, D)
                    rec = {"episode": keep[i]["name"], "group": keep[i]["group"],
                           "in_proj_orig": _proj(zin, u)[0].cpu().tolist(), "in_proj_mod": _proj(zm, u)[0].cpu().tolist(),
                           "tf_out_proj_orig": _proj(p0[bi : bi + 1], u)[0].cpu().tolist(), "tf_out_proj_mod": _proj(pm, u)[0].cpu().tolist(),
                           "tf_area_orig": area0[bi].tolist(), "tf_area_mod": _decode_area(model, pm[:, dec_frames], mcfg)[0].tolist(),
                           "random": {"in_shift": [], "out_shift_along_r": [], "area_mod": []}}
                    # open loop with first frame modified
                    rec["ol_out_proj_orig"], rec["ol_out_proj_mod"], rec["ol_area_orig"], rec["ol_area_mod"] = _ol_pair(model, z_all[bi : bi + 1], acts[bi : bi + 1], zm[:, :1], u, mcfg, D)
                    # random-direction controls with the same applied pooled shift magnitude (tf, per input frame)
                    for r in rnd:
                        if mode == "remove":
                            zr = zin - (zin * r).sum(-1, keepdim=True) * r
                        else:
                            zr = zin + (shift[..., None, None] * r)
                        zbr = base_in[bi : bi + 1].clone(); zbr[..., :D] = zr
                        pr = model.predict(zbr.reshape(K - 1, 1, *zbr.shape[2:]))[..., :D].reshape(1, K - 1, -1, D)
                        p_orig_r = _proj(p0[bi : bi + 1], r)[0]
                        rec["random"]["in_shift"].append((_proj(zr, r) - _proj(zin, r))[0].cpu().tolist())
                        rec["random"]["out_shift_along_r"].append((_proj(pr, r)[0] - p_orig_r).cpu().tolist())
                        rec["random"]["area_mod"].append(_decode_area(model, pr[:, dec_frames], mcfg)[0].tolist())
                    out["per_mode"][mode].append(rec)
        log(f"intervene {min(len(keep), s0 + bs)}/{len(keep)}")
    out["decoded_frames_idx"] = [f + 1 for f in dec_frames]
    write_json(out, res_dir / "intervene.json")
    return out


def _decode_area(model, lat, mcfg):
    """lat (B,n,196,384) -> (B,n) cloth pixel area of the decoded frames."""
    B, n = lat.shape[:2]
    img, _ = model.decoder(lat)
    u8 = to_u8(img)
    return np.stack([frames_area(u8[b * n : (b + 1) * n], mcfg) for b in range(B)])


def _ol_pair(model, z_all1, acts1, z0_mod, u, mcfg, D):
    """Open loop from frame 0 for original and with z0 replaced by z0_mod. Returns pooled projection and decoded area per frame."""
    import torch
    res = []
    for mod in (False, True):
        z = z_all1[:, :1].clone()
        if mod:
            z[..., :D] = z0_mod
        for j in range(1, K - 1):
            zp = model.predict(z[:, -1:])
            z_new = model.replace_actions_from_z(zp[:, -1:].clone(), acts1[:, j : j + 1])
            z = torch.cat([z, z_new], 1)
        zp = model.predict(z[:, -1:])
        z = torch.cat([z, zp[:, -1:]], 1)
        lat = z[:, 1:, :, :D]
        res.append((_proj(lat, u)[0].cpu().tolist(), _decode_area(model, lat, mcfg)[0].tolist()))
    return res[0][0], res[1][0], res[0][1], res[1][1]
