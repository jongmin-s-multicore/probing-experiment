"""numpy-only statistics: per-episode sufficient statistics, metrics, episode cluster bootstrap, Holm, Spearman, percentile.
No torch / scipy / sklearn. Python 3.9 compatible."""
from __future__ import annotations

import numpy as np

STAT_COLS = ["n", "sy", "sh", "syy", "shh", "syh", "sabs"]


def episode_stats(y, h, ep, n_eps):
    """Sufficient statistics per episode. y, h, ep: (N,) arrays, ep in [0, n_eps). Episodes without samples get zeros."""
    y = np.asarray(y, float)
    h = np.asarray(h, float)
    ep = np.asarray(ep, int)
    S = np.zeros((n_eps, len(STAT_COLS)))
    for j, v in enumerate([np.ones_like(y), y, h, y * y, h * h, y * h, np.abs(y - h)]):
        S[:, j] = np.bincount(ep, weights=v, minlength=n_eps)[:n_eps]
    return S


def _tot(S):
    return S[..., 0], S[..., 1], S[..., 2], S[..., 3], S[..., 4], S[..., 5], S[..., 6]


def metric_from_sums(S, kind):
    """S (..., 7) summed over the episode set. kind: r2 | pearson | mae. NaN when undefined."""
    n, sy, sh, syy, shh, syh, sabs = _tot(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "r2":
            sst = syy - sy * sy / n
            return np.where(sst > 1e-12, 1 - (syy - 2 * syh + shh) / sst, np.nan)
        if kind == "pearson":
            cov = syh - sy * sh / n
            den = np.sqrt((syy - sy * sy / n) * (shh - sh * sh / n))
            return np.where(den > 1e-12, cov / den, np.nan)
        if kind == "mae":
            return np.where(n > 0, sabs / n, np.nan)
    raise ValueError(kind)


def r2_within_from_level_sums(SL):
    """SL (..., L, 7): per-level sums. Within-level R2 (level means of y and of prediction removed first)."""
    n, sy, sh, syy, shh, syh, _ = _tot(SL)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = n > 0
        sst = np.where(ok, syy - sy * sy / n, 0.0)
        sse = np.where(ok, (syy - sy * sy / n) - 2 * (syh - sy * sh / n) + (shh - sh * sh / n), 0.0)
        a, b = sst.sum(-1), sse.sum(-1)
        return np.where(a > 1e-12, 1 - b / a, np.nan)


class Bootstrap:
    """Episode cluster bootstrap, resampling episodes within each level (level proportions fixed). The same resamples W are used
    for every statistic, so differences between cells are paired. NOTE: this CI reflects episode sampling only, not training-seed variation."""

    def __init__(self, level_idx, n_boot, seed):
        self.level_idx = np.asarray(level_idx, int)
        self.n = len(self.level_idx)
        self.levels = sorted(set(self.level_idx.tolist()))
        rng = np.random.RandomState(seed)
        self.B = int(n_boot)
        W = np.zeros((self.B, self.n))
        for l in self.levels:
            idx = np.where(self.level_idx == l)[0]
            for b in range(self.B):
                np.add.at(W[b], rng.choice(idx, len(idx), replace=True), 1)
        self.W = W

    def level_sums(self, S, mask, weights=None):
        """S (P,n,7); returns (B or 1, P, L, 7) sums of weights*mask*S per level."""
        W = self.W if weights is None else weights
        Wm = W * mask[None, :]
        out = []
        for l in self.levels:
            idx = np.where(self.level_idx == l)[0]
            out.append(np.einsum("bn,pnc->bpc", Wm[:, idx], S[:, idx, :]))
        return np.stack(out, axis=2)

    def stat(self, S, mask, kind):
        """Returns dict(point, per_probe_seed (P,), boot (B,)) ; probe-seed mean is the statistic."""
        S = np.asarray(S, float)
        if S.ndim == 2:
            S = S[None]
        mask = np.asarray(mask, float)
        ones = np.ones((1, self.n))

        def f(weights):
            SL = self.level_sums(S, mask, weights)                 # (b,P,L,7)
            if kind == "r2_within":
                return r2_within_from_level_sums(SL)               # (b,P)
            return metric_from_sums(SL.sum(2), kind)               # (b,P)

        pt = f(ones)[0]
        bt = f(None)
        return {"point": float(np.nanmean(pt)), "per_probe_seed": pt, "boot": np.nanmean(bt, axis=1)}


def ci95(boot):
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return [None, None]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def p_one_sided_le(boot, thr=0.0):
    """Bootstrap p for H0: stat <= thr vs stat > thr  :  (1 + #{boot <= thr}) / (B + 1)."""
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    return float((1 + (boot <= thr).sum()) / (len(boot) + 1))


def p_one_sided_ge(boot, thr):
    """Bootstrap p for H0: stat >= thr vs stat < thr."""
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    return float((1 + (boot >= thr).sum()) / (len(boot) + 1))


def p_two_sided_zero(boot):
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    lo = (1 + (boot <= 0).sum()) / (len(boot) + 1)
    hi = (1 + (boot >= 0).sum()) / (len(boot) + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvals, alpha):
    """Holm-Bonferroni. pvals: list (None = not evaluable, treated as 1.0 but still counted in the family size m).
    Returns (adjusted list, reject list)."""
    m = len(pvals)
    p = np.array([1.0 if v is None else v for v in pvals], float)
    order = np.argsort(p)
    adj = np.zeros(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    rej = [(a < alpha) and (pv is not None) for a, pv in zip(adj, pvals)]
    return adj.tolist(), rej


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pair_order_accuracy(pred, level_value):
    """Fraction of cross-level episode pairs whose predicted order matches the true level order (ordinal-size accuracy).
    pred (n,), level_value (n,) nominal size per episode. Point estimate only."""
    pred = np.asarray(pred, float)
    lv = np.asarray(level_value, float)
    dl = lv[:, None] - lv[None, :]
    dp = pred[:, None] - pred[None, :]
    m = dl > 0
    return float(((dp > 0) & m).sum() / m.sum()) if m.any() else float("nan")
