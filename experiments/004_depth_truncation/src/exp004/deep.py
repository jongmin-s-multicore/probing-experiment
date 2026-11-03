"""Torch side of experiment 004 (GPU on the server; also runs on CPU with toy models in selftest).
  * encoder truncation: patch tokens after block k (k=0: patch embedding + positional embedding) with the final LayerNorm applied.
  * predictor early exit: blocks 1..m + final norm. Mirrors branch models/vit.py ViTPredictor.forward @ code.commit 5a640be
    (x + pos_embedding -> blocks `x = attn(x) + x; x = ff(x) + x` -> transformer.norm). The branch predictor has NO separate output projection
    (the spec's 'output projection' does not exist; CHANGES.md).
  * extraction of hidden states / early-exit outputs / early-exit rollouts for all m in one pass.
Dimension constants (196 patches, 384) are read from exp002.model_io at call time (selftest patches them for toy sizes)."""
from __future__ import annotations

import copy
import time

import numpy as np

from exp002 import model_io as M
from .base import log


# ------------------------------------------------------------------------------------------------ encoder truncation
def flat_blocks(base):
    import torch
    out = []
    for b in base.blocks:
        if isinstance(b, torch.nn.ModuleList):               # DINOv2 BlockChunk (block_chunks>0): nested list, Identity placeholders
            out += [x for x in b if not isinstance(x, torch.nn.Identity)]
        elif not isinstance(b, torch.nn.Identity):
            out.append(b)
    return out


def tokens_by_k(base, x, ks):
    """DINOv2 VisionTransformer `base` (hub model), x (N,3,h,w) already resized for the encoder. Returns {k: (N, P, E)} patch tokens after
    block k (1-based; k=0 = patch embedding incl. positional embedding) with base.norm applied (same as get_intermediate_layers(norm=True),
    which cannot return k=0). k=len(blocks) equals forward_features()['x_norm_patchtokens']; stage enc_cache asserts this at run time."""
    ks = sorted(set(int(k) for k in ks))
    blocks = flat_blocks(base)
    if ks[-1] > len(blocks):
        raise ValueError("k=%d > number of blocks %d" % (ks[-1], len(blocks)))
    skip = 1 + int(getattr(base, "num_register_tokens", 0))
    h = base.prepare_tokens_with_masks(x)
    out = {}
    if 0 in ks:
        out[0] = base.norm(h)[:, skip:]
    for i, blk in enumerate(blocks):
        if i + 1 > ks[-1]:
            break
        h = blk(h)
        if (i + 1) in ks:
            out[i + 1] = base.norm(h)[:, skip:]
    return out


# ------------------------------------------------------------------------------------------------ predictor early exit
def ee_forward(predictor, x):
    """x (N, T*P, D) flattened like ViTPredictor.forward. Returns hidden list [h_0 .. h_L] (h_m = stream after m blocks, h_0 = after the positional
    embedding) ; early-exit output after m blocks = predictor.transformer.norm(h_m)."""
    n = x.shape[1]
    x = x + predictor.pos_embedding[:, :n]
    x = predictor.dropout(x)
    hidden = [x]
    for attn, ff in predictor.transformer.layers:
        x = attn(x) + x
        x = ff(x) + x
        hidden.append(x)
    return hidden


def exit_out(predictor, h):
    return predictor.transformer.norm(h)


def predict_exit(model, z, m):
    """z (N,T,P,D) -> early-exit prediction after m blocks, (N,T,P,D). m = n_blocks equals model.predict(z)."""
    N, T, P, D = z.shape
    h = ee_forward(model.predictor, z.reshape(N, T * P, D))
    return exit_out(model.predictor, h[m]).reshape(N, T, P, D)


class ModelView:
    """Read-only view of a world model with a different predictor (does not touch the original module's submodules)."""

    def __init__(self, model, predictor):
        self._m = model
        self.predictor = predictor

    def encode(self, *a, **k):
        return self._m.encode(*a, **k)

    def replace_actions_from_z(self, *a, **k):
        return self._m.replace_actions_from_z(*a, **k)

    def predict(self, z):
        N, T, P, D = z.shape
        return self.predictor(z.reshape(N, T * P, D)).reshape(N, T, P, D)


def random_init_predictor(model, seed):
    """ModelView with a freshly initialized copy of the predictor (control 3.4.5). pos_embedding ~ randn like the branch init."""
    import torch
    p = copy.deepcopy(model.predictor)
    torch.manual_seed(int(seed))
    for mod in p.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    with torch.no_grad():
        p.pos_embedding.copy_(torch.randn_like(p.pos_embedding))
    return ModelView(model, p)


def rollout_exit(model, m, z_all, acts, t, keep_steps, H, n_act_frames=12):
    """Autoregressive rollout from last context frame t, every step through the early-exit path of depth m (mirrors exp002 model_io.rollout
    / branch VWorldModel._rollout). Returns {k: (B,P,D_VIS) visual part of the predicted latent of frame t+k} for k in keep_steps."""
    import torch
    z = z_all[:, t - H + 1: t + 1]
    kept = {}
    for k in range(1, max(keep_steps) + 1):
        zp = predict_exit(model, z[:, -H:], m)
        new = zp[:, -1:].clone()
        if k in keep_steps:
            kept[k] = new[:, 0, :, :M.D_VIS].clone()
        if t + k < n_act_frames:
            new = model.replace_actions_from_z(new, acts[:, t + k: t + k + 1])
        z = torch.cat([z, new], 1)
    return kept


# ------------------------------------------------------------------------------------------------ one-pass extraction
def extract(cfg, cfg2, C, model, H, tmp, mean, std, dev, frames, Etr, keep_steps, dpart, with_outputs=True, batch_episodes=8, label=""):
    """Writes into `tmp`: feat_{m}.npy (E,nF,P,dpart) hidden of the LAST context frame after m blocks (all episodes), and when with_outputs
    out_{m}.npy (Ete,nF,P,D_VIS) early-exit predicted z of frame f+1 (teacher forced), roll_{m}.npy (Ete,len(keep_steps),P,D_VIS),
    mse_tf.npy (Ete,nF,L+1), mse_roll.npy (Ete,len(keep_steps),L+1). Returns meta."""
    import torch
    from pathlib import Path
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    L = M.n_blocks(model)
    E = len(C["rows"])
    Ete = E - Etr
    P, DV = M.P_TOK, M.D_VIS
    nF = len(frames)
    assert min(frames) >= H - 1, "frames %s need %d context frames" % (frames, H)
    assert max(frames) + 1 <= 12, "teacher-forced next-frame target needs frame f+1 <= 12"
    t = H - 1
    horizon = max(keep_steps)
    assert t + horizon <= 12, "rollout horizon %d from t=%d exceeds 12 cached steps" % (horizon, t)
    feats = [np.lib.format.open_memmap(tmp / ("feat_%d.npy" % m), mode="w+", dtype=np.float16, shape=(E, nF, P, dpart)) for m in range(L + 1)]
    if with_outputs:
        outs = [np.lib.format.open_memmap(tmp / ("out_%d.npy" % m), mode="w+", dtype=np.float16, shape=(Ete, nF, P, DV)) for m in range(L + 1)]
        rolls = [np.lib.format.open_memmap(tmp / ("roll_%d.npy" % m), mode="w+", dtype=np.float16, shape=(Ete, len(keep_steps), P, DV)) for m in range(L + 1)]
        mse_tf = np.zeros((Ete, nF, L + 1))
        mse_roll = np.zeros((Ete, len(keep_steps), L + 1))
    t0 = time.time()
    checked = False
    for s in range(0, E, batch_episodes):
        sl = slice(s, min(E, s + batch_episodes))
        x = M.frames_to_float(np.asarray(C["frames_u8"][sl])).to(dev)
        raw = torch.as_tensor(np.asarray(C["actions_raw"][sl]))
        acts = M.acts_from_raw(raw, mean, std).to(dev)
        with torch.no_grad():
            z_all = M.encode_batch(model, x, acts)
            B = x.shape[0]
            zw = M.make_windows(z_all, H, frames)                            # (B*nF, H, P, D)
            N, _, _, D = zw.shape
            hid = ee_forward(model.predictor, zw.reshape(N, H * P, D))
            if not checked and with_outputs:
                ref = model.predict(zw)
                got = exit_out(model.predictor, hid[L]).reshape(N, H, P, D)
                err = float((ref - got).abs().max())
                scale = float(ref.abs().max()) + 1e-6
                log("%s check: early-exit(m=L) vs model.predict max abs diff %.3g (scale %.3g)" % (label, err, scale))
                if err > 1e-3 * scale:
                    raise RuntimeError("early-exit path at m=L differs from model.predict (%.3g): the predictor structure is not what ee_forward assumes" % err)
                checked = True
            for m in range(L + 1):
                tok = M.last_frame_tokens(hid[m], H, dpart).reshape(B, nF, P, dpart)
                feats[m][sl] = tok.half().cpu().numpy()
            if with_outputs:
                idx = np.arange(sl.start, sl.stop)
                te = idx >= Etr
                if te.any():
                    tb = np.where(te)[0]
                    tgt = z_all[:, [f + 1 for f in frames], :, :DV][tb].float()                  # (b,nF,P,DV)
                    for m in range(L + 1):
                        o = exit_out(model.predictor, hid[m]).reshape(B, nF, H, P, D)[tb][:, :, -1, :, :DV].float()
                        outs[m][idx[tb] - Etr] = o.half().cpu().numpy()
                        mse_tf[idx[tb] - Etr, :, m] = ((o - tgt) ** 2).mean((-1, -2)).cpu().numpy()
                        kept = rollout_exit(model, m, z_all[tb], acts[tb], t, list(keep_steps), H)
                        for ki, k in enumerate(keep_steps):
                            rolls[m][idx[tb] - Etr, ki] = kept[k].half().cpu().numpy()
                            mse_roll[idx[tb] - Etr, ki, m] = ((kept[k].float() - z_all[tb][:, t + k, :, :DV].float()) ** 2).mean((-1, -2)).cpu().numpy()
        log("%s extract %d/%d (%.0fs)" % (label, sl.stop, E, time.time() - t0))
    for a in feats + (outs + rolls if with_outputs else []):
        a.flush()
    if with_outputs:
        np.save(tmp / "mse_tf.npy", mse_tf)
        np.save(tmp / "mse_roll.npy", mse_roll)
    return {"n_blocks": L, "num_hist": H, "t": t, "frames": list(frames), "keep_steps": list(keep_steps), "dpart": dpart, "with_outputs": with_outputs}
