"""GPU stage `extract`: for each 002 run (2 conditions x 3 training seeds) load model_latest.pth (read only) and recompute, from the 002 cached uint8
frames (the same inputs 002 used), the autoregressive rollout of the trained predictor for ALL cached episodes (probe-training rows first, then test
rows) and, for the test rows, rollouts of UNTRAINED predictors. Only patch-mean pooled latents (E,12,384) are stored (the linear probe pools over
patches first; patch-level latents would be about 3.3 GB per run). Rollout code path = exp002 model_io.rollout (not modified).

002 stored: q2_cells.npz holds per-episode probe PREDICTIONS (predmean|...OUT|rollout|stepK) but not the predicted latents; the latents of the training
rows were also never kept (q2_tmp is deleted). Everything that needs latents (B untrained, C, D, the reproduction check) therefore needs this stage.

Analysis F (teacher forcing): for the same checkpoint and the same rows the trained predictor is also run ONE step ahead from the REAL encoder latent of the
previous frame (context = z_all[:, t+k-H : t+k], target frame t+k, k = 1..horizon). Same pooling (fp16 cast, float32 patch mean). See README section 3F.

Untrained predictor (README section 3): same class/hyper-parameters as the trained one (hydra.utils.instantiate(tcfg.predictor, num_patches, num_frames, dim) exactly as
train.py init_models), weights from torch.manual_seed(seed). The input encoder (frozen DINOv2) is the original model in all variants (z_all visual part is the
same tensor 002 used). Variant `pred_only`: action_encoder and proprio_encoder = the TRAINED ones of the run's checkpoint. Variant `pred_and_action_proprio`:
action_encoder, proprio_encoder and predictor are all freshly initialised from the same seed (instantiated in train.py order: proprio, action, predictor)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import base as B
from .base import log


def action_stats_local(manifest_path, data_root, action_dims, normalize):
    """mean/std of the raw actions of the run's training manifest, as exp002 data.ManifestDataset computes them (torch.std, + 1e-6). Needs only metadata.pt
    of the training splits (data_meta stage). normalize False -> zeros/ones (ManifestDataset behaviour)."""
    import torch
    dims = list(action_dims)
    if not normalize:
        return torch.zeros(len(dims)), torch.ones(len(dims)), False
    eps = B.read_json(manifest_path)["episodes"]
    cache, raws = {}, []
    for ep in eps:
        d = Path(data_root) / ep["split"]
        if d not in cache:
            mp = d / "metadata.pt"
            if not mp.exists():
                raise FileNotFoundError("%s missing - run `python run.py data_meta` (needs the 002 archive or the dataset)" % mp)
            m = torch.load(mp, map_location="cpu", weights_only=True)
            cache[d] = {Path(e["video_path"]).name: e for e in m}
        raws.append(cache[d][ep["name"]]["actions"][:, dims].float())
    allact = torch.cat(raws)
    return allact.mean(0), allact.std(0) + 1e-6, True


def hydra_factory(model, tcfg, dev, action_in, proprio_in=1):
    """Returns make(seed, variant) -> dict(predictor, action_encoder, proprio_encoder) of freshly initialised modules (train.py init_models order and arguments)."""
    import torch
    import hydra

    def make(seed, variant):
        torch.manual_seed(int(seed))
        proprio = action = None
        if variant == "pred_and_action_proprio":
            proprio = hydra.utils.instantiate(tcfg.proprio_encoder, in_chans=proprio_in, emb_dim=tcfg.proprio_emb_dim)
            action = hydra.utils.instantiate(tcfg.action_encoder, in_chans=action_in, emb_dim=tcfg.action_emb_dim)
            pe, ae = proprio.emb_dim, action.emb_dim
        else:
            pe, ae = model.proprio_encoder.emb_dim, model.action_encoder.emb_dim
        side = int(tcfg.img_size) // 16
        num_patches = side ** 2 + (2 if int(tcfg.concat_dim) == 0 else 0)
        dim = model.encoder.emb_dim + (pe * int(tcfg.num_proprio_repeat) + ae * int(tcfg.num_action_repeat)) * int(tcfg.concat_dim)
        pred = hydra.utils.instantiate(tcfg.predictor, num_patches=num_patches, num_frames=int(tcfg.num_hist), dim=dim)
        mods = {"predictor": pred}
        if proprio is not None:
            mods["proprio_encoder"], mods["action_encoder"] = proprio, action
        if proprio is not None:
            if not same_structure(proprio, model.proprio_encoder) or not same_structure(action, model.action_encoder):
                raise RuntimeError("freshly built action/proprio encoder has a different parameter structure than the checkpoint's (in_chans guess action_in=%s proprio_in=%s)" % (action_in, proprio_in))
        for m in mods.values():
            m.to(dev).eval()
            for p in m.parameters():
                p.requires_grad_(False)
        return mods
    return make


def same_structure(a, b):
    sa = [(k, tuple(v.shape)) for k, v in a.state_dict().items()]
    sb = [(k, tuple(v.shape)) for k, v in b.state_dict().items()]
    return sa == sb


class Swap:
    """Temporarily replace modules of the model (restored on exit)."""

    def __init__(self, model, mods):
        self.model, self.mods, self.old = model, mods, {}

    def __enter__(self):
        for k, v in self.mods.items():
            self.old[k] = getattr(self.model, k)
            setattr(self.model, k, v)
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(self.model, k, v)


def rebuild_aux(model, z_all, acts, mods):
    """z_all with the proprio and action parts recomputed by the encoders in `mods` (visual part untouched). Mirrors VWorldModel.encode for concat_dim == 1."""
    import torch
    z = z_all.clone()
    B_, K, P, D = z.shape
    pe = mods.get("proprio_encoder", model.proprio_encoder)
    ae = mods.get("action_encoder", model.action_encoder)
    nap, nar = int(model.num_proprio_repeat), int(model.num_action_repeat)
    pro = pe(torch.zeros(B_, K, 1, device=z.device))                                  # (B,K,pd)
    act = ae(acts)                                                                     # (B,K,ad)
    pro_t = pro.unsqueeze(2).expand(-1, -1, P, -1).repeat(1, 1, 1, nap)
    act_t = act.unsqueeze(2).expand(-1, -1, P, -1).repeat(1, 1, 1, nar)
    d_vis = model.encoder.emb_dim
    z[..., d_vis:d_vis + pro_t.shape[-1]] = pro_t
    z[..., -act_t.shape[-1]:] = act_t
    return z


def tf_feasible_steps(t, horizon, H, n_frames=13):
    """Steps k for which a teacher-forced one-step prediction exists from the 13 cached frames: the context z_all[:, t+k-H : t+k] must exist (t+k-H >= 0)
    and the target frame t+k must be a cached frame (t+k <= n_frames-1)."""
    return [k for k in range(1, int(horizon) + 1) if t + k - H >= 0 and t + k <= n_frames - 1]


def teacher_forced(model, z_all, t, horizon, H):
    """One-step predictions from REAL encoder latents: step k predicts frame t+k from z_all[:, t+k-H : t+k] (the real, encoded context frames, whose action part is
    the action embedding of that frame, the same one the open-loop rollout writes back with replace_actions_from_z). Returns visual part (B, horizon, 196, 384).
    Mirrors exp002 model_io.rollout except that the context is never replaced by the model's own output."""
    import torch
    from exp002 import model_io as M
    preds = []
    for k in range(1, horizon + 1):
        zp = model.predict(z_all[:, t + k - H: t + k])
        preds.append(zp[:, -1:, :, :M.D_VIS].clone())
    return torch.cat(preds, 1)


def run_pass(cfg, cfg2, model, C, rows_slice, mean, std, dev, t, horizon, H, untrained, bs, label, stats=None, teacher=True, diag=None):
    """Rollouts for rows_slice (start, stop). untrained: None or dict {(variant, seed): modules}. teacher: also the teacher-forced one-step predictions (F).
    Returns (trained_pooled (n,12,384), {key: (n,12,384)}, teacher_forced_pooled (n,12,384) or None). diag (dict) receives the max |teacher step 1 - open-loop step 1| of the first batch."""
    import torch
    from exp002 import model_io as M
    from exp002 import probes as PR
    start, stop = rows_slice
    n = stop - start
    out_t = np.zeros((n, horizon, M.D_VIS), np.float32)
    out_u = {k: np.zeros((n, horizon, M.D_VIS), np.float32) for k in (untrained or {})}
    out_tf = np.zeros((n, horizon, M.D_VIS), np.float32) if teacher else None
    t0 = time.time()
    for s in range(start, stop, bs):
        sl = slice(s, min(stop, s + bs))
        tb = time.time()
        x = M.frames_to_float(np.asarray(C["frames_u8"][sl])).to(dev)
        raw = torch.as_tensor(np.asarray(C["actions_raw"][sl]))
        acts = M.acts_from_raw(raw, mean, std).to(dev)
        with torch.no_grad():
            z_all = M.encode_batch(model, x, acts)
            r = M.rollout(model, z_all, acts, t, horizon, H)                       # (B, horizon, 196, 384), float32
            out_t[sl.start - start: sl.stop - start] = PR.pool(r.half().cpu().numpy())     # same cast + pooling as exp002 (fp16 rollout memmap, probes.pool)
            if teacher:
                rt = teacher_forced(model, z_all, t, horizon, H)
                out_tf[sl.start - start: sl.stop - start] = PR.pool(rt.half().cpu().numpy())
                if diag is not None and "tf_step1_vs_open_loop_step1_max_abs_diff" not in diag:
                    diag["tf_step1_vs_open_loop_step1_max_abs_diff"] = float((rt[:, 0] - r[:, 0]).abs().max())
            if untrained:
                if s == start and label == "test":
                    z_chk = rebuild_aux(model, z_all, acts, {})
                    err = float((z_chk - z_all).abs().max())
                    log("rebuild_aux self-check on the trained encoders: max |diff| = %.3g" % err)
                    if err > 1e-5:
                        raise RuntimeError("rebuild_aux does not reproduce VWorldModel.encode (max diff %.3g)" % err)
                for key, mods in untrained.items():
                    zv = z_all if "action_encoder" not in mods else rebuild_aux(model, z_all, acts, mods)
                    with Swap(model, mods):
                        ru = M.rollout(model, zv, acts, t, horizon, H)
                    out_u[key][sl.start - start: sl.stop - start] = PR.pool(ru.half().cpu().numpy())
        if stats is not None:
            stats.setdefault(label, []).append(time.time() - tb)
        if ((s - start) // bs) % 20 == 0:
            log("extract %s %d/%d (%.0fs)" % (label, sl.stop - start, n, time.time() - t0))
    return out_t, out_u, out_tf


def build_untrained(cfg, factory, model, variants, seeds):
    mods = {}
    for v in variants:
        for sd in seeds:
            m = factory(sd, v)
            if not same_structure(m["predictor"], model.predictor):
                raise RuntimeError("untrained predictor (variant %s seed %s) has a different parameter structure than the trained one" % (v, sd))
            mods[(v, int(sd))] = m
    return mods


def extract_run(cfg, cfg2, paths, run, dev, C, factory=None, max_rows=None, out_dir=None, smoke_stats=None, write=True):
    """Full extraction of one run. max_rows=(n_train_rows, n_test_rows) is for the smoke only."""
    import torch
    from exp002 import model_io as M
    out_dir = Path(out_dir) if out_dir else paths["extract"] / run
    done = out_dir / "DONE"
    use_tf = bool(cfg["extract"].get("teacher_forcing", True))
    if write and done.exists() and use_tf and not (out_dir / "teacher_forced_pooled.npy").exists():
        log("extract %s: DONE marker exists but teacher_forced_pooled.npy (analysis F) is missing -> extraction is redone" % run)
    elif write and done.exists():
        log("extract %s: DONE marker exists, skip" % run)
        return {"run": run, "skipped": True}
    rdir = paths["train_outputs"] / run
    ck = rdir / "checkpoints" / "model_latest.pth"
    if not ck.exists():
        raise FileNotFoundError("checkpoint missing: %s" % ck)
    t_load = time.time()
    model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
    fp = M.encoder_fingerprint(model.encoder)
    if fp != C["meta"]["encoder_fingerprint"]:
        raise RuntimeError("encoder of %s differs from the 002 cache encoder" % run)
    H = int(cfg["task"]["num_hist_expected"])
    assert int(tcfg.num_hist) == H, "checkpoint num_hist %s != config task.num_hist_expected %s" % (tcfg.num_hist, H)
    assert int(tcfg.concat_dim) == 1, "only concat_dim == 1 is supported (z + tiled proprio/action)"
    t = H - 1
    assert t == int(cfg["task"]["context_frame_t"])
    horizon = min(int(cfg["task"]["rollout_horizon"]), 12 - t)
    mean, std, norm = action_stats_local(rdir / "train_manifest.json", paths["data_root"], cfg2["data_impl"]["action_dims"], bool(tcfg.normalize_action))
    log("extract %s: epoch %d, normalize_action %s, num_hist %d, load %.0fs" % (run, ep, norm, H, time.time() - t_load))
    rows = C["rows"]
    E = len(rows)
    Etr = sum(1 for r in rows if r["part"] == "train")
    bs = int(cfg["extract"]["batch_episodes"])
    if factory is None:
        factory = hydra_factory(model, tcfg, dev, int(cfg2["data_impl"]["frameskip"]) * len(cfg2["data_impl"]["action_dims"]))
    variants = list(cfg["extract"]["untrained"]["variants"])
    useeds = [int(s) for s in cfg["seeds"]["untrained_predictor"]]
    untrained = build_untrained(cfg, factory, model, variants, useeds)
    tr_stop = Etr if not max_rows else min(Etr, int(max_rows[0]))
    te_stop = E if not max_rows else min(E, Etr + int(max_rows[1]))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tf_steps = tf_feasible_steps(t, horizon, H)
    if tf_steps != list(range(1, horizon + 1)):
        log("WARNING teacher forcing is possible only for steps %s of 1..%d (cached frames: 13)" % (tf_steps, horizon))
    diag = {}
    tr_roll, _, tr_tf = run_pass(cfg, cfg2, model, C, (0, tr_stop), mean, std, dev, t, horizon, H, None, bs, "train", smoke_stats, teacher=use_tf, diag=diag)
    te_roll, te_unt, te_tf = run_pass(cfg, cfg2, model, C, (Etr, te_stop), mean, std, dev, t, horizon, H, untrained, bs, "test", smoke_stats, teacher=use_tf, diag=diag)
    if use_tf:
        log("teacher forcing: step 1 vs open-loop step 1 max |diff| (first train batch) = %s" % diag.get("tf_step1_vs_open_loop_step1_max_abs_diff"))
    elapsed = time.time() - t0
    peak = float(torch.cuda.max_memory_allocated()) / 2 ** 30 if dev.type == "cuda" else None
    meta = {"run": run, "epoch": ep, "encoder_fingerprint": fp, "num_hist": H, "t": t, "horizon": horizon, "normalize_action": norm,
            "action_mean": mean.tolist(), "action_std": std.tolist(), "n_train_rows": tr_stop, "n_test_rows": te_stop - Etr, "batch_episodes": bs,
            "checkpoint_bytes": ck.stat().st_size, "untrained": [{"variant": k[0], "seed": k[1]} for k in untrained], "elapsed_s": elapsed,
            "peak_cuda_mem_gib": peak, "pooled": "mean over the 196 patches of the fp16-cast rollout latent (visual 384 dims), as exp002 probes.pool",
            "device": str(dev), "finished": time.strftime("%F %T"),
            "teacher_forced": {"enabled": use_tf, "steps_available": tf_steps if use_tf else [], "context": "z_all[:, t+k-H : t+k] (real encoder latents of the cached frames); target frame t+k",
                               "rows": "same rows as the open-loop rollout (probe-training rows first, then test rows)", "file": "teacher_forced_pooled.npy", **diag}}
    if write:
        meta["checkpoint_sha256"] = B.sha256_file(ck)
        write_run_outputs(paths, out_dir, tr_roll, te_roll, te_unt, meta, tr_tf, te_tf)
    return {"run": run, "meta": meta, "tr_roll": tr_roll, "te_roll": te_roll, "te_unt": te_unt, "tr_tf": tr_tf, "te_tf": te_tf}


def write_run_outputs(paths, out_dir, tr_roll, te_roll, te_unt, meta, tr_tf=None, te_tf=None):
    """Checked writes (read-back) of one run's extraction; the DONE marker is written last, so an interrupted run is redone."""
    out_dir = Path(out_dir)
    B.probe_output_dir(paths)
    B.write_npy(paths, out_dir / "rollout_pooled.npy", np.concatenate([tr_roll, te_roll], 0))
    if tr_tf is not None and te_tf is not None:
        B.write_npy(paths, out_dir / "teacher_forced_pooled.npy", np.concatenate([tr_tf, te_tf], 0))
    for (v, sd), arr in te_unt.items():
        B.write_npy(paths, out_dir / ("untrained_%s_s%d.npy" % (v, sd)), arr)
    B.write_json(paths, meta, out_dir / "meta.json")
    B.write_marker(paths, out_dir / "DONE")
