"""World-model access: loading, residual-stream hooks, windows, rollout. Based on experiments/001 src/exp001/extract.py
(load_world_model, LayerCapture, rollout loop). Interface assumptions about the branch model (UNVERIFIED on this machine, checked by
`python run.py smoke` step c):
  model.encode(obs, acts) -> (B,K,196,404) ; model.predict(z (N,H,196,404)) -> same shape ; model.replace_actions_from_z(z, acts)
  model.predictor.transformer.layers = [(attn, ff), ...] ; model.predictor.transformer.norm ; model.encoder.emb_dim = 384 ; model.encoder_transform
Residual-stream index j (0..L): j = input of block j (j=0: predictor input, after positional embedding as captured by the pre-hook of the
first attention); j = L: output of the last block (input of the final norm). Hidden tensors are (N, H*196, 404), frame-major token order (assumed)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

P_TOK = 196
D_VIS = 384


def load_world_model(acwm_src, rdir, device, epoch=None):
    import torch
    import hydra
    from omegaconf import OmegaConf
    sys.path.insert(0, str(acwm_src))
    cfg = OmegaConf.load(Path(rdir) / "hydra.yaml")
    ck = Path(rdir) / "checkpoints" / ("model_%s.pth" % epoch if epoch else "model_latest.pth")
    payload = torch.load(ck, map_location=device, weights_only=False)      # trusted: our own checkpoints (torch>=2.6 default weights_only=True fails)
    parts = {k: payload[k].to(device) for k in ["predictor", "decoder", "proprio_encoder", "action_encoder"]}
    parts["encoder"] = (payload["encoder"] if "encoder" in payload else hydra.utils.instantiate(cfg.encoder)).to(device)
    model = hydra.utils.instantiate(
        cfg.model, encoder=parts["encoder"], proprio_encoder=parts["proprio_encoder"], action_encoder=parts["action_encoder"],
        predictor=parts["predictor"], decoder=parts["decoder"], proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, int(payload["epoch"])


def n_blocks(model):
    return len(model.predictor.transformer.layers)


def encoder_fingerprint(encoder):
    """sha256 over all encoder parameters (float32 bytes). Q1 cache and every Q2/Q3 run must share it (frozen DINOv2)."""
    h = hashlib.sha256()
    for name, p in sorted(encoder.named_parameters(), key=lambda t: t[0]):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


class StreamHooks:
    """Hooks on the residual stream at layer index j.
    capture: store the tensor seen at j (after a modification, if any).
    modify : replace the stream by fn(x). The branch blocks are `x = attn(x) + x; x = ff(x) + x`, so a plain pre-hook on attn would change
             only the attention INPUT and leave the residual path untouched (this was a bug in an earlier draft; 001's hooks only captured).
             Here a pre-hook feeds attn(x') and a forward-hook adds delta = x' - x to its output, so that the block output is attn(x') + delta + x = attn(x') + x'.  For j == L (final norm) the stream is the module input itself.
    ASSUMPTION about the branch block structure (checked by `smoke` step c via a residual-path probe)."""

    def __init__(self, model):
        tr = model.predictor.transformer
        self.L = len(tr.layers)
        self.targets = [tr.layers[j][0] for j in range(self.L)] + [tr.norm]
        self.store = {}
        self.modify = {}
        self._delta = {}
        self.handles = []
        self.capture_layers = set()

    def _pre(self, j):
        def fn(_m, args):
            x = args[0]
            changed = False
            if j in self.modify:
                x2 = self.modify[j](x)
                if j < self.L:
                    self._delta[j] = x2 - x
                x, changed = x2, True
            if j in self.capture_layers:
                self.store[j] = x.detach()
            return ((x,) + tuple(args[1:])) if changed else None
        return fn

    def _post(self, j):
        def fn(_m, _args, out):
            d = self._delta.pop(j, None)
            return None if d is None else out + d
        return fn

    def install(self):
        if not self.handles:
            for j, t in enumerate(self.targets):
                self.handles.append(t.register_forward_pre_hook(self._pre(j)))
                if j < self.L:
                    self.handles.append(t.register_forward_hook(self._post(j)))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self):
        self.store, self.modify, self.capture_layers, self._delta = {}, {}, set(), {}


def last_frame_tokens(h, H, d=None):
    """hidden (N, H*196, dim) -> (N,196,dim[:d]) tokens of the last context frame (frame-major order assumed)."""
    out = h[:, (H - 1) * P_TOK: H * P_TOK]
    return out if d is None else out[..., :d]


def encode_batch(model, vis, acts):
    import torch
    B, K = vis.shape[:2]
    return model.encode({"visual": vis, "proprio": torch.zeros(B, K, 1, device=vis.device)}, acts)


def make_windows(z_all, H, frames):
    """z_all (B,K,196,404); frames = list of last-context frame indices (each >= H-1). -> (B*len(frames), H, 196, 404), ordered [b][f]."""
    import torch
    w = [z_all[:, f - H + 1: f + 1] for f in frames]
    return torch.stack(w, 1).reshape(z_all.shape[0] * len(frames), H, *z_all.shape[2:])


def rollout(model, z_all, acts, t, horizon, H, n_act_frames=12):
    """Autoregressive rollout from last context frame t, mirroring 001 extract.run_batch / branch VWorldModel._rollout.
    Returns visual-part predicted latents (B, horizon, 196, 384) for frames t+1 .. t+horizon."""
    import torch
    z = z_all[:, t - H + 1: t + 1]
    preds = []
    for k in range(1, horizon + 1):
        zp = model.predict(z[:, -H:])
        new = zp[:, -1:].clone()
        preds.append(new[..., :D_VIS].clone())
        if t + k < n_act_frames:
            new = model.replace_actions_from_z(new, acts[:, t + k: t + k + 1])
        z = torch.cat([z, new], 1)
    return torch.cat(preds, 1)


def frames_to_float(u8):
    """uint8 (..., H, W, 3) numpy/torch -> float tensor (..., 3, H, W) in [-1, 1] (the 001 convention: to_u8 inverts it)."""
    import torch
    x = torch.as_tensor(np.asarray(u8)).float()
    x = x.permute(*range(x.ndim - 3), x.ndim - 1, x.ndim - 3, x.ndim - 2)
    return x / 127.5 - 1.0


def to_u8(x):
    """(N,3,H,W) in [-1,1] -> (N,H,W,3) uint8 (identical to 001 extract.to_u8)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def action_stats_for_run(cfg, rdir, tcfg):
    """Mean/std of the raw actions of the run's training manifest (as ManifestDataset computes them when normalize_action is true)."""
    import torch
    from .common import read_json
    from .data import make_dataset_class
    man = read_json(Path(rdir) / "train_manifest.json")["episodes"]
    ds = make_dataset_class()(man, normalize_action=bool(tcfg.normalize_action), action_dims=list(tcfg.env.dataset.action_dims))
    return ds.action_mean, ds.action_std, bool(tcfg.normalize_action)


def acts_from_raw(raw_actions, mean, std, n_groups=12, frameskip=5):
    """raw_actions (E,>=60,6) torch float -> (E,13,30) normalized & grouped like 001 episode_inputs (last row zeros)."""
    import torch
    a = (raw_actions[:, : n_groups * frameskip] - mean) / std
    a = a.reshape(a.shape[0], n_groups, -1)
    return torch.cat([a, torch.zeros(a.shape[0], 1, a.shape[2])], 1)
