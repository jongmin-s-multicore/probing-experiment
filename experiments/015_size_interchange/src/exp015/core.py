"""Torch part of experiment 015: shifts at the predictor input (z part, all tokens, before the positional embedding) and the projection 'moved' of the predicted-change difference
onto d. Takes the dimensions from exp002.model_io (P_TOK, D_VIS) at call time so that the CPU selftest can run toy sizes.

Per sample = one transition f -> f+1 (f = 0..11): z = z part of the REAL encoder latent of frame f; sh a (P,dz) shift; p0 = pred(z), p1 = pred(z + sh) (visual parts, the
action / proprio coordinates are untouched). Delta_hat(x) = pred(x) - x, so Delta_hat(z + sh) - Delta_hat(z) = p1 - p0 - sh.
  moved  = sum((p1 - p0 - sh) * d) / sum(d * d)        d (P,dz) = DZ_T - DZ_S, per sample; the per-episode value is the mean over the 12 transitions
  resp   = Frobenius norm of (p1 - p0 - sh)             per-episode mean over the 12 transitions (report only)"""
from __future__ import annotations

import numpy as np

from exp002 import model_io as M

NT = 12


def build_shifts(Mn, fake, s_i, t_is, rdirs, dev):
    """Mn (L,P,dz), fake (L,R,P,dz) numpy (unscaled fake shifts), source index s_i, target indices t_is, rdirs (K,dz) unit vectors.
    Returns (shifts {key: torch (P,dz) float32}, specs [(name, key, target index)]).
    keys: real:T | fakeu:r (unscaled, shared by the targets, report only) | fakenm:T:r (norm-matched to |m_T - m_S| per patch, pair specific, PRIMARY comparator) | rand:T:k
    spec names: real:T, fakeu:r:T, fakenm:r:T, rand:T:k. Forwards per sample and predictor: 1 (unedited) + R unscaled + len(t_is) x (1 + R norm-matched + K random) = 103 for R=K=20, 2 targets."""
    import torch
    from .lib import norm_match
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
    shifts, specs = {}, []
    R = fake.shape[1]
    for r in range(R):
        shifts["fakeu:%d" % r] = f(fake[s_i, r])
    for t in t_is:
        sh = Mn[t] - Mn[s_i]
        shifts["real:%d" % t] = f(sh)
        specs.append(("real:%d" % t, "real:%d" % t, t))
        nrm = np.linalg.norm(sh, axis=-1)
        for k in range(len(rdirs)):
            shifts["rand:%d:%d" % (t, k)] = f(nrm[:, None] * np.asarray(rdirs[k])[None, :])
            specs.append(("rand:%d:%d" % (t, k), "rand:%d:%d" % (t, k), t))
        for r in range(R):
            specs.append(("fakeu:%d:%d" % (r, t), "fakeu:%d" % r, t))
        fnm, _ = norm_match(fake[s_i], sh)
        for r in range(R):
            shifts["fakenm:%d:%d" % (t, r)] = f(fnm[r])
            specs.append(("fakenm:%d:%d" % (r, t), "fakenm:%d:%d" % (t, r), t))
    return shifts, specs


def eval_source_batch(z_all, predict_fns, shifts, specs, dvec):
    """z_all (B,13,P,D) real encoder latents (visual + action coordinates). predict_fns {kind: fn(z (N,1,P,D)) -> (N,1,P,D)}. dvec {target index: (d torch (P,dz), |d|^2 float)}.
    Returns {kind: {spec name: (B,) float64 numpy, 'resp:real:T': (B,)}}."""
    import torch
    dz = M.D_VIS
    B = z_all.shape[0]
    zin_full = z_all[:, :NT].reshape(B * NT, *z_all.shape[2:])
    z = zin_full[..., :dz]
    by_key = {}
    for name, key, t in specs:
        by_key.setdefault(key, []).append((name, t))
    dv = {t: (d.double(), float(n2)) for t, (d, n2) in dvec.items()}
    out = {}
    for kind, fn in predict_fns.items():
        o = {}
        p0 = fn(zin_full[:, None])[:, 0, :, :dz]
        for key, sh in shifts.items():
            zin = zin_full.clone()
            zin[..., :dz] = z + sh
            p1 = fn(zin[:, None])[:, 0, :, :dz]
            dd = (p1 - p0 - sh).double()
            if key.startswith("real:"):
                o["resp:" + key] = dd.flatten(1).norm(dim=1).reshape(B, NT).mean(1).cpu().numpy()
            for name, t in by_key.get(key, []):
                d, n2 = dv[t]
                val = (dd * d).sum((1, 2)) / n2 if n2 > 0 else torch.full((B * NT,), float("nan"), dtype=torch.float64, device=dd.device)
                o[name] = val.reshape(B, NT).mean(1).cpu().numpy()
        out[kind] = o
    return out
