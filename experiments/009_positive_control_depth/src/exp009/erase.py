"""Size-direction erasure at predictor block m_int and extraction of hidden states after it (experiment 009).
numpy part: fit_eraser. torch part: erase_stream, forward_hidden, extract_hidden (mirrors exp004.deep.extract's hidden branch, without outputs/rollouts).
The hidden stream h_m = stream after m blocks (h_0 = after positional embedding); the intervention replaces h_{m_int}[..., :dz] and the later blocks run unchanged."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from exp002 import model_io as M
from .base import log


# ------------------------------------------------------------------------------------------------ numpy: fit
def fit_eraser(X, y, method, ridge_rel):
    """X (n, d) pooled z features of the fit episodes, y (n,) size labels. Returns dict(method, mu, c, a, u, corr_fit).
    c = cov(x, y); a = (Sigma + lambda I)^-1 c (ridge probe direction); u = a/|a|.
    leace_scalar: x' = x - c * a.(x - mu) / (a.c)   -> a.x' = a.mu constant and (up to the ridge) cov(x', y) = 0 (LEACE for a scalar target)
    probe_direction: x' = x - u (u.x)               -> orthogonal removal of the probe direction only."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu = X.mean(0)
    Xc = X - mu
    yc = y - y.mean()
    n, d = X.shape
    c = Xc.T @ yc / n
    S = Xc.T @ Xc / n
    lam = float(ridge_rel) * np.trace(S) / d
    a = np.linalg.solve(S + lam * np.eye(d), c)
    if not np.isfinite(a).all() or np.linalg.norm(a) < 1e-12 or abs(float(a @ c)) < 1e-18:
        raise ValueError("degenerate size direction (labels constant in the fit set or zero covariance)")
    u = a / np.linalg.norm(a)
    if method not in ("leace_scalar", "probe_direction"):
        raise ValueError("unknown erase method %r" % method)
    corr = float(np.corrcoef(X @ a, y)[0, 1])
    return {"method": method, "mu": mu, "c": c, "a": a, "u": u, "corr_fit": corr, "readout_std_fit": float((X @ a).std())}


def apply_eraser_np(E, X, alpha):
    """numpy reference of the torch operation, for tests. X (..., d)."""
    X = np.asarray(X, np.float64)
    if E["method"] == "leace_scalar":
        return X - alpha * np.outer((X - E["mu"]) @ E["a"] / float(E["a"] @ E["c"]), E["c"]).reshape(X.shape)
    return X - alpha * np.outer(X @ E["u"], E["u"]).reshape(X.shape)


def save_eraser(E, path):
    np.savez_compressed(path, **{k: (np.asarray(v) if k != "method" else np.array(v)) for k, v in E.items()})


def load_eraser(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["method"] = str(d["method"])
    d["corr_fit"] = float(d["corr_fit"])
    d["readout_std_fit"] = float(d["readout_std_fit"])
    return d


# ------------------------------------------------------------------------------------------------ torch
def erase_stream(x, E, alpha, dz):
    """x (N, T*P, D) torch. Only x[..., :dz] is changed."""
    import torch
    z = x[..., :dz]
    dt = x.dtype
    if E["method"] == "leace_scalar":
        a, c, mu = (torch.as_tensor(E[k], dtype=dt, device=x.device) for k in ("a", "c", "mu"))
        coef = ((z - mu) @ a) / (a @ c)
        z2 = z - alpha * coef[..., None] * c
    else:
        u = torch.as_tensor(E["u"], dtype=dt, device=x.device)
        z2 = z - alpha * (z @ u)[..., None] * u
    return torch.cat([z2, x[..., dz:]], -1)


def forward_hidden(predictor, x, erase=None):
    """Same as exp004.deep.ee_forward, optionally erasing after block erase['m_int']. erase = dict(m_int, alpha, E, dz) or None."""
    n = x.shape[1]
    x = x + predictor.pos_embedding[:, :n]
    x = predictor.dropout(x)
    hidden = [x]
    for i, (attn, ff) in enumerate(predictor.transformer.layers, start=1):
        x = attn(x) + x
        x = ff(x) + x
        if erase is not None and i == erase["m_int"] and erase["alpha"] > 0:
            x = erase_stream(x, erase["E"], erase["alpha"], erase["dz"])
        hidden.append(x)
    return hidden


def pooled_hidden(model, C, H, mean, std, dev, frames, ep_idx, m, dz, batch_episodes=8):
    """Pooled (mean over patches) z part of the last context frame after m blocks, WITHOUT intervention, for the episodes ep_idx.
    Returns (len(ep_idx), nF, dz) float32."""
    import torch
    P = M.P_TOK
    nF = len(frames)
    out = np.zeros((len(ep_idx), nF, dz), np.float32)
    ep_idx = np.asarray(ep_idx)
    for s in range(0, len(ep_idx), batch_episodes):
        ids = ep_idx[s:s + batch_episodes]
        x = M.frames_to_float(np.asarray(C["frames_u8"][np.sort(ids)])).to(dev)
        order = np.argsort(np.argsort(ids))
        raw = torch.as_tensor(np.asarray(C["actions_raw"][np.sort(ids)]))
        acts = M.acts_from_raw(raw, mean, std).to(dev)
        with torch.no_grad():
            z_all = M.encode_batch(model, x, acts)
            B = x.shape[0]
            zw = M.make_windows(z_all, H, frames)
            N, _, _, D = zw.shape
            hid = forward_hidden(model.predictor, zw.reshape(N, H * P, D))
            tok = M.last_frame_tokens(hid[m], H, dz).reshape(B, nF, P, dz).float().mean(2).cpu().numpy()
        out[s:s + len(ids)] = tok[order]
    return out


def extract_hidden(model, C, H, tmp, mean, std, dev, frames, dz, depths, erase=None, batch_episodes=8, label=""):
    """Writes tmp/feat_{m}.npy (E,nF,P,dz) float16 for m in depths (hidden of the last context frame, z part) with the optional intervention applied in the
    forward pass. Returns dict(n_blocks, pooled_mint) where pooled_mint (E,nF,dz) float32 is the pooled z part at erase['m_int'] (after erasure) when erase is given."""
    import torch
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    L = M.n_blocks(model)
    E_ = len(C["rows"])
    P = M.P_TOK
    nF = len(frames)
    assert min(frames) >= H - 1
    feats = {m: np.lib.format.open_memmap(tmp / ("feat_%d.npy" % m), mode="w+", dtype=np.float16, shape=(E_, nF, P, dz)) for m in depths}
    pooled = np.zeros((E_, nF, dz), np.float32) if erase is not None else None
    t0 = time.time()
    for s in range(0, E_, batch_episodes):
        sl = slice(s, min(E_, s + batch_episodes))
        x = M.frames_to_float(np.asarray(C["frames_u8"][sl])).to(dev)
        acts = M.acts_from_raw(torch.as_tensor(np.asarray(C["actions_raw"][sl])), mean, std).to(dev)
        with torch.no_grad():
            z_all = M.encode_batch(model, x, acts)
            B = x.shape[0]
            zw = M.make_windows(z_all, H, frames)
            N, _, _, D = zw.shape
            hid = forward_hidden(model.predictor, zw.reshape(N, H * P, D), erase)
            for m in depths:
                tok = M.last_frame_tokens(hid[m], H, dz).reshape(B, nF, P, dz)
                feats[m][sl] = tok.half().cpu().numpy()
                if pooled is not None and m == erase["m_int"]:
                    pooled[sl] = tok.float().mean(2).cpu().numpy()
        if (s // batch_episodes) % 20 == 0:
            log("%s extract %d/%d (%.0fs)" % (label, sl.stop, E_, time.time() - t0))
    for a in feats.values():
        a.flush()
    return {"n_blocks": L, "pooled_mint": pooled}
