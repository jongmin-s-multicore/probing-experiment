"""Torch part of experiment 013: erasure displacements at the predictor input, one-step teacher-forced scores, erased hidden-state extraction (readout check).
Everything takes the dimensions from exp002.model_io (P_TOK, D_VIS) at call time so that the CPU selftest can run toy sizes.

Definitions (per sample = one transition f -> f+1, f = 0..11; z = visual part of the REAL encoder latent of frame f, z1 = of frame f+1, z_e = z + delta the erased input):
  pred            visual part of predictor(z_e with the untouched action/proprio coordinates)
  total           mean over (P tokens x dz dims) of (pred - z1)^2                                  [metric A, next-latent token MSE]
  motion          mean of ((pred - z_e) - (z1 - z))^2   predicted CHANGE (relative to the input the predictor saw) vs real change Delta z   [metric B]
  persist         mean of (z1 - z)^2                    the Delta z = 0 prediction (persistence baseline)
  copy            mean of (z_e - z1)^2                  copying the erased input
  disp_norm_mean  mean over tokens of |delta|           disp_sq_mean  mean over tokens of |delta|^2     (verification of the norm matching)
Without erasure total == motion. The per-episode value is the mean over the 12 transitions."""
from __future__ import annotations

import time

import numpy as np

from exp002 import model_io as M
from exp009 import erase as ER
from .base import log

METRICS = ["total", "motion", "persist", "copy", "disp_norm_mean", "disp_sq_mean"]
MI = {m: i for i, m in enumerate(METRICS)}
NT = 12


def config_keys(K):
    return ["none", "leace:size"] + ["leace:rand:%d" % k for k in range(K)] + ["lmm:size"] + ["lmm:rand:%d" % k for k in range(K)]


class Eraser:
    """Torch copies of the fitted erasers. leace: dict from exp009 fit_eraser (mu, a, c); D (L,P,dz) level-mean shifts; R (K,dz) unit random directions; S (K,L,P) signs."""

    def __init__(self, leace, D, R, S, dev):
        import torch
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
        self.mu, self.a, self.c = f(leace["mu"]), f(leace["a"]), f(leace["c"])
        self.ac = float(np.asarray(leace["a"]) @ np.asarray(leace["c"]))
        self.cn = float(np.linalg.norm(leace["c"]))
        self.D, self.R, self.S = f(D), f(R), f(S)

    def prepare(self, z, lvl):
        """Quantities shared by all keys of a batch. z (N,P,dz), lvl (N,) long."""
        coef = ((z - self.mu) @ self.a) / self.ac                     # (N,P)
        Dl = self.D[lvl]                                              # (N,P,dz)
        return {"coef": coef, "Dl": Dl, "nrm": Dl.norm(dim=-1)}

    def displacement(self, key, pre, lvl):
        if key == "none":
            return None
        parts = key.split(":")
        method, kind = parts[0], parts[1]
        if method == "leace":
            if kind == "size":
                return -pre["coef"][..., None] * self.c
            k = int(parts[2])
            return -pre["coef"][..., None] * self.cn * self.R[k]       # same signed coefficient, |c| along a random unit direction: per-token norm == size erasure
        if kind == "size":
            return pre["Dl"]
        k = int(parts[2])
        return (pre["nrm"] * self.S[k][lvl])[..., None] * self.R[k]    # per (level, patch): |D| along the random direction, random sign: per-token norm == size erasure


def score(pred, z_e, z, z1, delta):
    """All (N,P,dz) -> (N,6) float64 in METRICS order."""
    import torch
    sq = lambda x: (x.double() ** 2).mean((1, 2))
    if delta is None:
        zero = torch.zeros(pred.shape[0], dtype=torch.float64, device=pred.device)
        dn = dsq = zero
    else:
        nr = delta.double().norm(dim=-1)
        dn, dsq = nr.mean(1), (nr ** 2).mean(1)
    return torch.stack([sq(pred - z1), sq((pred - z_e) - (z1 - z)), sq(z1 - z), sq(z_e - z1), dn, dsq], 1)


def eval_batch(z_all, lvl_ep, predict_fns, er, keys):
    """z_all (B,13,P,D) real encoder latents, lvl_ep (B,) long level index. predict_fns {kind: fn(z (N,1,P,D)) -> (N,1,P,D)}.
    Returns {kind: {key: (B,6) float64 numpy}}, {kind: (B,dz) predicted mean change of the UN-erased input}, (B,dz) real mean change."""
    import torch
    dz = M.D_VIS
    B = z_all.shape[0]
    zin_full = z_all[:, :NT].reshape(B * NT, *z_all.shape[2:])
    z = zin_full[..., :dz]
    z1 = z_all[:, 1:NT + 1, :, :dz].reshape(B * NT, -1, dz)
    lvl = lvl_ep.repeat_interleave(NT)
    pre = er.prepare(z, lvl) if er is not None else None
    out, dhat = {}, {}
    for kind, fn in predict_fns.items():
        out[kind] = {}
        for key in keys:
            delta = er.displacement(key, pre, lvl) if key != "none" else None
            z_e = z if delta is None else z + delta
            zin = zin_full.clone()
            zin[..., :dz] = z_e
            pred = fn(zin[:, None])[:, 0, :, :dz]
            out[kind][key] = score(pred, z_e, z, z1, delta).reshape(B, NT, len(METRICS)).mean(1).cpu().numpy()
            if key == "none":
                dhat[kind] = (pred - z).reshape(B, NT, -1, dz).mean((1, 2)).double().cpu().numpy()
    dreal = (z1 - z).reshape(B, NT, -1, dz).mean((1, 2)).double().cpu().numpy()
    return out, dhat, dreal


def extract_erased(model, C, H, tmp, mean, std, dev, frames, dz, depths, er, key, lvl_rows, batch_episodes=8, label=""):
    """Like exp009.erase.extract_hidden but the ERASURE IS APPLIED TO THE PREDICTOR INPUT (z part of every token of every frame) before the positional embedding.
    key: None (no erasure) | 'leace:size' | 'lmm:size'. lvl_rows (E,) long level index of every cache row. Writes tmp/feat_{m}.npy (E,nF,P,dz) float16 for m in depths
    (z part of the tokens of the last context frame after m blocks; m = 0 is the input after the positional embedding)."""
    import torch
    from pathlib import Path
    assert H == 1, "input erasure is implemented for num_hist == 1"
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    E_ = len(C["rows"])
    P = M.P_TOK
    nF = len(frames)
    feats = {m: np.lib.format.open_memmap(tmp / ("feat_%d.npy" % m), mode="w+", dtype=np.float16, shape=(E_, nF, P, dz)) for m in depths}
    t0 = time.time()
    for s in range(0, E_, batch_episodes):
        sl = slice(s, min(E_, s + batch_episodes))
        x = M.frames_to_float(np.asarray(C["frames_u8"][sl])).to(dev)
        acts = M.acts_from_raw(torch.as_tensor(np.asarray(C["actions_raw"][sl])), mean, std).to(dev)
        lv = torch.as_tensor(lvl_rows[sl], dtype=torch.long, device=dev)
        with torch.no_grad():
            z_all = M.encode_batch(model, x, acts)
            B_ = x.shape[0]
            zw = M.make_windows(z_all, H, frames)                          # (B*nF, 1, P, D)
            zw = zw.clone()
            if key is not None:
                zz = zw[:, 0, :, :dz]
                lvw = lv.repeat_interleave(nF)
                pre = er.prepare(zz, lvw)
                zw[:, 0, :, :dz] = zz + er.displacement(key, pre, lvw)
            N, _, _, D = zw.shape
            hid = ER.forward_hidden(model.predictor, zw.reshape(N, H * P, D), None)
            for m in depths:
                tok = M.last_frame_tokens(hid[m], H, dz).reshape(B_, nF, P, dz)
                feats[m][sl] = tok.half().cpu().numpy()
        if (s // batch_episodes) % 20 == 0:
            log("%s extract %d/%d (%.0fs)" % (label, sl.stop, E_, time.time() - t0))
    for a in feats.values():
        a.flush()
