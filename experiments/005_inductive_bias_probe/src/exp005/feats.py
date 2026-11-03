"""Layer features for experiment 005: encoder layers come from the 004 cache (tokens_k<k>.npy / rand_k<k>.npy); predictor layers are extracted here
(GPU) from the 002 condition-(b) checkpoint with exp004.deep.ee_forward (z-part = first D_VIS coordinates)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from exp002 import model_io as M
from exp004 import deep as DP
from .base import log


def enc_cache_dir(paths, suffix):
    return paths["cache004"] / ("enc" + suffix)


def layer_path(paths, layer, control, suffix):
    """npy path of (E,nF,P,d) fp16 features of a layer. control: 'real' | 'rand'."""
    if layer["module"] == "encoder":
        return enc_cache_dir(paths, suffix) / (("tokens_k%d.npy" if control == "real" else "rand_k%d.npy") % layer["depth"])
    return paths["cache"] / ("feat_%s%s%s.npy" % (layer["name"], "" if control == "real" else "_rand", suffix))


def extract_layers(models, C, H, frames, layers, outdir, mean, std, dev, batch_episodes, suffix="", label=""):
    """models: {'real': model, 'rand': ModelView with a random-init predictor} ; layers: predictor layers (dicts from branch.py).
    Writes one file per (layer, control) via layer_path-compatible names under outdir. Returns the written paths."""
    import torch
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    E, nF = len(C["rows"]), len(frames)
    P, DV = M.P_TOK, M.D_VIS
    mm, paths_out = {}, []
    for ctl in models:
        for l in layers:
            p = outdir / ("feat_%s%s%s.npy" % (l["name"], "" if ctl == "real" else "_rand", suffix))
            mm[(ctl, l["name"])] = np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(E, nF, P, DV))
            paths_out.append(p)
    t0 = time.time()
    base_model = models["real"]
    for s in range(0, E, batch_episodes):
        sl = slice(s, min(E, s + batch_episodes))
        x = M.frames_to_float(np.asarray(C["frames_u8"][sl])).to(dev)
        acts = M.acts_from_raw(torch.as_tensor(np.asarray(C["actions_raw"][sl])), mean, std).to(dev)
        with torch.no_grad():
            z_all = M.encode_batch(base_model, x, acts)
            B = x.shape[0]
            zw = M.make_windows(z_all, H, frames)
            N, _, _, D = zw.shape
            for ctl, mdl in models.items():
                hid = DP.ee_forward(mdl.predictor, zw.reshape(N, H * P, D))
                for l in layers:
                    h = hid[l["depth"]]
                    if l["kind"] == "output":
                        h = DP.exit_out(mdl.predictor, h)
                    tok = M.last_frame_tokens(h, H, DV).reshape(B, nF, P, DV)
                    mm[(ctl, l["name"])][sl] = tok.half().cpu().numpy()
        log("%s extract_layers %d/%d (%.0fs)" % (label, sl.stop, E, time.time() - t0))
    for a in mm.values():
        a.flush()
    return paths_out
