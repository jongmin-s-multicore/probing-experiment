"""Probes shared by Q1 / Q2 / Q3 (same capacity settings from config.probe_impl everywhere).
  linear_pooled : mean over patch tokens -> standardized ridge (alpha chosen by GroupKFold inside the training episodes)
  attentive     : LayerNorm -> Linear(d_in,d_model) -> 1-query cross-attention over patch tokens -> residual MLP -> linear head (torch)
  mlp           : pooled features -> 1 hidden layer MLP (torch). Used only for the 'was the information really removed?' check.
Hyper-parameters are fixed by config; early stopping uses a validation split of TRAINING episodes only.
torch is imported lazily; ridge parts work without torch."""
from __future__ import annotations

import numpy as np

from . import directions as D


def get_device(cfg):
    import torch
    want = cfg["probe_impl"].get("device", "cuda")
    return torch.device(want if (want != "cuda" or torch.cuda.is_available()) else "cpu")


def pool(tokens):
    """(..., P, d) -> (..., d) float32 mean over patches."""
    return np.asarray(tokens, np.float32).mean(-2)


def linear_fit_predict(Xtr, ytr, eptr, Xte_list, seed, cfg):
    pc = cfg["probe_impl"]
    m = D.ridge_fit(Xtr, ytr, eptr, pc["ridge_alphas"], pc["inner_folds"], seed)
    return [D.ridge_predict(m, X) for X in Xte_list], m


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("torch is required for the attentive / mlp probes (%s)" % e)


def _make_attentive(d_in, ac):
    torch = _torch()
    nn = torch.nn

    class AttentiveProbe(nn.Module):
        def __init__(self):
            super().__init__()
            d = ac["d_model"]
            self.norm = nn.LayerNorm(d_in)
            self.proj = nn.Linear(d_in, d)
            self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.attn = nn.MultiheadAttention(d, ac["n_heads"], batch_first=True)
            self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ac["mlp_hidden"]), nn.GELU(), nn.Linear(ac["mlp_hidden"], d))
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            x = self.proj(self.norm(x))
            o, _ = self.attn(self.q.expand(x.shape[0], -1, -1), x, x, need_weights=False)
            o = o + self.mlp(o)
            return self.head(o[:, 0]).squeeze(-1)

    return AttentiveProbe()


def _train_torch(model, Xtr, ytr, eptr, seed, tc, device, epochs, lr, wd, bs, val_fraction):
    """Generic training loop. Xtr: numpy (N,...) ; returns the model with the best-validation weights (validation = held-out TRAINING episodes)."""
    torch = _torch()
    rng = np.random.RandomState(seed)
    ue = np.unique(eptr)
    rng.shuffle(ue)
    nval = max(1, int(round(len(ue) * val_fraction))) if val_fraction > 0 and len(ue) > 1 else 0
    val_eps = set(ue[:nval].tolist())
    isval = np.array([e in val_eps for e in eptr])
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-8)
    Xt = torch.as_tensor(np.asarray(Xtr)).to(device)
    yt = torch.as_tensor(((ytr - ym) / ys).astype(np.float32)).to(device)
    tr_idx = np.where(~isval)[0]
    va_idx = np.where(isval)[0]
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.as_tensor(tr_idx)[torch.randperm(len(tr_idx), generator=g)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs].to(device)
            loss = ((model(Xt[b].float()) - yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if len(va_idx):
            model.eval()
            with torch.no_grad():
                v = float(sum(((model(Xt[torch.as_tensor(va_idx[i:i + bs]).to(device)].float()) - yt[torch.as_tensor(va_idx[i:i + bs]).to(device)]) ** 2).sum().item()
                              for i in range(0, len(va_idx), bs)) / len(va_idx))
            if v < best:
                best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, ym, ys


def _predict_torch(model, ym, ys, X, device, bs=512):
    torch = _torch()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(np.asarray(X[i:i + bs])).to(device).float()
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) * ys + ym


def attentive_fit_predict(Xtr, ytr, eptr, Xte_list, seed, cfg):
    """Xtr (N,P,d) tokens. Returns [pred for each X in Xte_list]."""
    torch = _torch()
    ac = cfg["probe_impl"]["attentive"]
    dev = get_device(cfg)
    torch.manual_seed(seed)
    model = _make_attentive(Xtr.shape[-1], ac)
    model, ym, ys = _train_torch(model, Xtr, ytr, eptr, seed, ac, dev, ac["epochs"], ac["lr"], ac["weight_decay"], ac["batch_size"], ac["val_fraction"])
    return [_predict_torch(model, ym, ys, X, dev) for X in Xte_list]


def mlp_fit_predict(Xtr, ytr, eptr, Xte_list, seed, cfg):
    """Pooled features (N,d), 1 hidden layer, standardized inputs (train statistics)."""
    torch = _torch()
    mc = cfg["probe_impl"]["mlp_check"]
    dev = get_device(cfg)
    torch.manual_seed(seed)
    mu, sd = D._standardize(np.asarray(Xtr, np.float32))
    nn = torch.nn
    model = nn.Sequential(nn.Linear(Xtr.shape[1], mc["hidden"]), nn.ReLU(), nn.Linear(mc["hidden"], 1))

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).squeeze(-1)

    model, ym, ys = _train_torch(Wrap(model), ((Xtr - mu) / sd).astype(np.float32), ytr, eptr, seed, mc, dev, mc["epochs"], mc["lr"], mc["weight_decay"], mc["batch_size"], 0.1)
    return [_predict_torch(model, ym, ys, ((X - mu) / sd).astype(np.float32), dev) for X in Xte_list]


def fit_predict(kind, Xtr, ytr, eptr, Xte_list, seed, cfg):
    """kind: linear_pooled (X pooled (N,d)) | attentive (X tokens (N,P,d)) | mlp (X pooled)."""
    if kind == "linear_pooled":
        return linear_fit_predict(Xtr, ytr, eptr, Xte_list, seed, cfg)[0]
    if kind == "attentive":
        return attentive_fit_predict(Xtr, ytr, eptr, Xte_list, seed, cfg)
    if kind == "mlp":
        return mlp_fit_predict(Xtr, ytr, eptr, Xte_list, seed, cfg)
    raise ValueError(kind)


def shuffle_episode_labels(Y, seed):
    """Y (E,F) label matrix of the TRAINING episodes -> rows permuted across episodes (episode-level label shuffle)."""
    rng = np.random.RandomState(seed)
    return Y[rng.permutation(len(Y))]


def mask_baseline_features(feat):
    """feat (...,3) [area_px, bbox_w, bbox_h] -> log area, log w, log h, log bbox area, area/bbox_area. Mask-only baseline input."""
    a = np.maximum(feat[..., 0], 1.0)
    w = np.maximum(feat[..., 1], 1.0)
    h = np.maximum(feat[..., 2], 1.0)
    return np.stack([np.log(a), np.log(w), np.log(h), np.log(w * h), a / (w * h)], -1)
