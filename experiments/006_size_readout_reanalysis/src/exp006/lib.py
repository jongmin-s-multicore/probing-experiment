"""numpy-only analysis helpers for experiment 006 (no torch). Formulas are written out in README.md section 7.
Nothing here judges anything: no thresholds, no p values, no pass/fail."""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------------------------------------ Student t quantile (no scipy needed)
def t_pdf(x, df):
    c = math.exp(math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0)) / math.sqrt(df * math.pi)
    return c * (1.0 + x * x / df) ** (-(df + 1) / 2.0)


def t_cdf(x, df, n=4000):
    """P(T <= x) by Simpson integration of the density from 0 (x >= 0) ; symmetric for x < 0."""
    if x == 0:
        return 0.5
    if x < 0:
        return 1.0 - t_cdf(-x, df, n)
    n = n if n % 2 == 0 else n + 1
    h = x / n
    s = t_pdf(0.0, df) + t_pdf(x, df)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * t_pdf(i * h, df)
    return 0.5 + s * h / 3.0


def t_ppf(p, df):
    """Quantile of Student t with `df` degrees of freedom (df may be non-integer). p in (0.5, 1)."""
    if not 0.5 < p < 1.0:
        raise ValueError("t_ppf supports 0.5 < p < 1")
    if abs(df - 2.0) < 1e-12:
        return (2 * p - 1) / math.sqrt(2 * p * (1 - p))            # closed form for df = 2
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e7:
            return float("inf")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def t_crit(df, level=0.95):
    return t_ppf(0.5 + level / 2.0, df)


# ------------------------------------------------------------------------------------------------ seed-level summaries
def seed_agg(arr, level=0.95):
    """arr (n_seeds, ...) -> dict(mean, sd, n, se, df, t_crit, ci_lo, ci_hi, per_seed). sd uses ddof=1; CI = mean +- t(df=n-1) * sd / sqrt(n).
    n = number of TRAINING seeds (3 in 002), so df = 2 and the interval is wide. Returned arrays have the trailing shape of arr."""
    a = np.asarray(arr, float)
    n = a.shape[0]
    mean = np.mean(a, axis=0)
    if n > 1:
        sd = np.std(a, axis=0, ddof=1)
        tc = t_crit(n - 1, level)
        se = sd / math.sqrt(n)
        lo, hi = mean - tc * se, mean + tc * se
    else:
        sd = np.full_like(mean, np.nan)
        se, lo, hi, tc = sd, sd, sd, float("nan")
    return {"mean": mean, "sd": sd, "n": n, "se": se, "df": n - 1, "t_crit": tc, "ci_lo": lo, "ci_hi": hi, "per_seed": a}


def diff_agg(a, b, level=0.95):
    """Difference of condition means (a - b) from seed-level values a (na, ...) and b (nb, ...).
    welch: se = sqrt(sa^2/na + sb^2/nb), df = Welch-Satterthwaite, CI = diff +- t(df) * se  (no pairing assumed).
    paired_by_seed_index: d_s = a_s - b_s for s = 0..n-1 ; mean, sd (ddof=1), CI with t(df=n-1). Pairing by seed index is a convention, not shown to be meaningful."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.shape[0], b.shape[0]
    d = a.mean(0) - b.mean(0)
    va, vb = a.var(0, ddof=1) / na, b.var(0, ddof=1) / nb
    se = np.sqrt(va + vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    flat_df = df.reshape(-1)
    lo = np.full(flat_df.shape, np.nan)
    hi = np.full(flat_df.shape, np.nan)
    se_f, d_f = se.reshape(-1), np.asarray(d, float).reshape(-1)
    for i, dv in enumerate(flat_df):
        if np.isfinite(dv) and dv > 0 and np.isfinite(se_f[i]):
            tc = t_crit(float(dv), level)
            lo[i], hi[i] = d_f[i] - tc * se_f[i], d_f[i] + tc * se_f[i]
    out = {"diff_mean": d, "welch_se": se, "welch_df": df, "welch_ci_lo": lo.reshape(d.shape), "welch_ci_hi": hi.reshape(d.shape)}
    if na == nb:
        p = seed_agg(a - b, level)
        out.update({"paired_mean": p["mean"], "paired_sd": p["sd"], "paired_ci_lo": p["ci_lo"], "paired_ci_hi": p["ci_hi"], "paired_df": p["df"], "paired_per_seed": p["per_seed"]})
    return out


# ------------------------------------------------------------------------------------------------ line fits and level tables
def ols_line(x, y):
    """Least squares y = slope * x + intercept. NaN when x has no variance or fewer than 2 points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    vx = ((x - xm) ** 2).sum()
    if vx <= 1e-12:
        return float("nan"), float("nan")
    s = ((x - xm) * (y - ym)).sum() / vx
    return float(s), float(ym - s * xm)


def a_stats(h, true_label, level_idx, n_levels, role_masks):
    """Level-wise readout summary of one prediction vector.
    h (n,) probe predictions (log area ratio), true_label (n,) label (log nominal area ratio, constant per level), level_idx (n,) in [0, n_levels).
    Returns dict of arrays: per_level_{mean_pred,true,bias,mae,sd_pred,n} (n_levels,), and per role: slope/intercept/mae of the
    EPISODE-level fit pred ~ label, plus slope/intercept of the fit of LEVEL MEANS on the level label (levels inside the role only)."""
    h, y, li = np.asarray(h, float), np.asarray(true_label, float), np.asarray(level_idx, int)
    mp, tr, mae, sd, cnt = (np.full(n_levels, np.nan) for _ in range(5))
    for l in range(n_levels):
        m = li == l
        if m.any():
            mp[l], tr[l], mae[l], sd[l], cnt[l] = h[m].mean(), y[m][0], np.abs(h[m] - y[m]).mean(), h[m].std(), m.sum()
    out = {"per_level_mean_pred": mp, "per_level_true": tr, "per_level_bias": mp - tr, "per_level_mae": mae, "per_level_sd_pred": sd, "per_level_n": cnt}
    for role, rm in role_masks.items():
        s, i = ols_line(y[rm], h[rm])
        lv = sorted(set(li[rm].tolist()))
        sl, il = ols_line([tr[l] for l in lv], [mp[l] for l in lv])
        out["slope_episode|" + role] = s
        out["intercept_episode|" + role] = i
        out["slope_levelmeans|" + role] = sl
        out["intercept_levelmeans|" + role] = il
        out["mae|" + role] = float(np.abs(h[rm] - y[rm]).mean()) if rm.any() else float("nan")
        out["bias_mean|" + role] = float((h[rm] - y[rm]).mean()) if rm.any() else float("nan")
    return out


def pair_order_accuracy_levels(h, true_label, level_idx):
    """Same definition as exp002 stats.pair_order_accuracy: fraction of cross-level episode pairs ordered like the true level order."""
    from exp002 import stats as ST
    return ST.pair_order_accuracy(h, true_label)


def metric_point(S, mask, kind):
    """S (P, n, 7) per-episode sufficient statistics (exp002 stats.STAT_COLS). Mean over probe seeds of the metric on the episodes in mask."""
    from exp002 import stats as ST
    S = np.asarray(S, float)
    tot = S[:, np.asarray(mask, bool)].sum(1)
    return float(np.nanmean(ST.metric_from_sums(tot, kind)))


def cells_from_pred(y, H):
    """H (P, n) predictions of P probe seeds for labels y (n,) -> S (P, n, 7)."""
    from exp002 import stats as ST
    y = np.asarray(y, float)
    n = len(y)
    return np.stack([ST.episode_stats(y, h, np.arange(n), n) for h in np.asarray(H, float)])


def r2_within_levels(S, level_idx, n_levels, mask=None):
    """Within-level R2 (exp002 stats.r2_within_from_level_sums) from per-episode cells S (P, n, 7), averaged over probe seeds."""
    from exp002 import stats as ST
    S = np.asarray(S, float)
    li = np.asarray(level_idx, int)
    m = np.ones(len(li), bool) if mask is None else np.asarray(mask, bool)
    SL = np.stack([S[:, (li == l) & m].sum(1) for l in range(n_levels)], axis=1)       # (P, L, 7)
    return float(np.nanmean(ST.r2_within_from_level_sums(SL)))


def digest_arrays(*arrs):
    import hashlib
    h = hashlib.sha1()
    for a in arrs:
        h.update(np.ascontiguousarray(np.round(np.asarray(a, float), 6)).tobytes())
    return h.hexdigest()
