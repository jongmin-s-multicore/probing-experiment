"""Torch heads of experiment 005. The attentive head is exp002.probes._make_attentive (same structure/capacity as the 002 probes, config probe_impl.attentive),
trained per task from scratch: BCE-with-logits for binary tasks (one logit), MSE on standardized targets for continuous tasks.
Oracle heads (true s as input): linear and 2-layer MLP (hidden 5). Area head: the same attentive class applied to one token holding the mask area/bbox
features, with the input LayerNorm replaced by Identity (a LayerNorm over 5 features would remove the feature scale; CHANGES.md)."""
from __future__ import annotations

import numpy as np

from exp002 import probes as PR


def _t():
    import torch
    return torch


class Src:
    """Feature source of one (model, layer): train_t (Ntr_flat, P, d) indexed by flat ids, eval_t (n_eval, P, d), both torch tensors on `dev`."""

    def __init__(self, train_t, eval_t, arch, no_input_norm=False):
        self.train_t, self.eval_t, self.arch, self.no_input_norm = train_t, eval_t, arch, no_input_norm
        self.dev = train_t.device

    def get_train(self, ids):
        torch = _t()
        return self.train_t[torch.as_tensor(np.asarray(ids), device=self.dev)].float()

    def chunks(self, bs=512):
        for i in range(0, self.eval_t.shape[0], bs):
            yield self.eval_t[i:i + bs].float()


def _seed(seed, task):
    return int(np.random.SeedSequence([int(seed), int(task)]).generate_state(1)[0] % (2 ** 31 - 1))


def _loss(kind):
    torch = _t()
    if kind == "bin":
        return lambda out, y: torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    return lambda out, y: ((out - y) ** 2).mean()


def _fit_attn(src, X, y, kind, seed, ac, hc):
    torch = _t()
    torch.manual_seed(seed)
    model = PR._make_attentive(X.shape[-1], ac).to(src.dev)
    if src.no_input_norm:
        model.norm = torch.nn.Identity()
    ym, ys = (0.0, 1.0) if kind == "bin" else (float(y.mean()), float(y.std() + 1e-8))
    yt = torch.as_tensor((y - ym) / ys, dtype=torch.float32, device=src.dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hc["lr"], weight_decay=hc["weight_decay"])
    lf = _loss(kind)
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    bs = int(hc["batch_size"])
    model.train()
    for _ in range(int(hc["epochs"])):
        perm = torch.randperm(n, generator=g).to(src.dev)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = lf(model(X[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    out = []
    with torch.no_grad():
        for xb in src.chunks():
            out.append(model(xb).float().cpu().numpy())
    o = np.concatenate(out)
    return (o > 0).astype(np.float32) if kind == "bin" else (o * ys + ym).astype(np.float32)


def _fit_small(src, X, y, kind, seed, hc):
    torch = _t()
    torch.manual_seed(seed)
    nn = torch.nn
    Xf = X.reshape(X.shape[0], -1)
    mu, sd = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True) + 1e-6
    f = Xf.shape[1]
    model = (nn.Linear(f, 1) if src.arch == "linear" else nn.Sequential(nn.Linear(f, 5), nn.ReLU(), nn.Linear(5, 1))).to(src.dev)
    ym, ys = (0.0, 1.0) if kind == "bin" else (float(y.mean()), float(y.std() + 1e-8))
    yt = torch.as_tensor((y - ym) / ys, dtype=torch.float32, device=src.dev)
    opt = torch.optim.Adam(model.parameters(), lr=hc["small"]["lr"])
    lf = _loss(kind)
    Xs = (Xf - mu) / sd
    for _ in range(int(hc["small"]["steps"])):
        loss = lf(model(Xs).squeeze(-1), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = []
    with torch.no_grad():
        for xb in src.chunks():
            out.append(model((xb.reshape(xb.shape[0], -1) - mu) / sd).squeeze(-1).float().cpu().numpy())
    o = np.concatenate(out)
    return (o > 0).astype(np.float32) if kind == "bin" else (o * ys + ym).astype(np.float32)


def fit_predict(src, ids, y, kind, seed, task, ac, hc):
    """Train one head on the flat ids `ids` with targets y (n,), predict all evaluation inputs. kind: 'bin' | 'reg'. Returns (n_eval,) float32
    (binary: 0/1 predictions; regression: values)."""
    X = src.get_train(ids)
    y = np.asarray(y, np.float64)
    s = _seed(seed, task)
    if src.arch == "attn":
        return _fit_attn(src, X, y, kind, s, ac, hc)
    return _fit_small(src, X, y, kind, s, hc)
