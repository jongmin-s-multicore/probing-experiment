"""Inductive-bias metrics (numpy only). Formulas of spec 4.2-4.5 (Vafa et al. 2025 Eq. 1-3 as transcribed in the spec; the original text was NOT checked,
see CHANGES.md).
  R-IB_raw       = mean_k mean_{(i,j): phi_i = phi_j} 1[yhat_k(x_i) = yhat_k(x_j)]
  D-IB_raw       = 1 - mean_k mean_{(i,j): phi_i != phi_j} 1[yhat_k(x_i) = yhat_k(x_j)]
  D-IB_area=     : D-IB over pairs with different level and the same area decile q;  D-IB_area!= : different level and different q
  R-IB_fold      : R-IB over pairs with the same level and different q
  score = (raw - raw_noninf) / (raw_oracle - raw_noninf)
Pairs are unordered pairs of distinct evaluation inputs (the diagonal is excluded)."""
from __future__ import annotations

import numpy as np

GROUPS = ["all", "ind", "ood_any", "ood_both"]
METRICS = ["R-IB", "D-IB", "D-IB_area=", "D-IB_area≠", "R-IB_fold"]
_SPEC = {"R-IB": (True, None, False), "D-IB": (False, None, True), "D-IB_area=": (False, True, True),
         "D-IB_area≠": (False, False, True), "R-IB_fold": (True, False, False)}      # (state equal?, q equal? (None = any), raw = 1 - mean eq?)


class Ctx:
    """Evaluation-input metadata: lvl (n,) level index, q (n,) area decile, is_ind (n,) bool (level in the trained levels), is_ood (n,) bool."""

    def __init__(self, lvl, q, is_ind, is_ood):
        self.lvl, self.q = np.asarray(lvl), np.asarray(q)
        self.is_ind, self.is_ood = np.asarray(is_ind, bool), np.asarray(is_ood, bool)
        self.n = len(self.lvl)


def group_mask(ctx, group):
    a, b = ctx.is_ind[:, None], ctx.is_ind[None, :]
    o1, o2 = ctx.is_ood[:, None], ctx.is_ood[None, :]
    if group == "all":
        return np.ones((ctx.n, ctx.n), bool)
    if group == "ind":
        return a & b
    if group == "ood_any":
        return o1 | o2
    if group == "ood_both":
        return o1 & o2
    raise ValueError(group)


def pair_mask(ctx, metric, group):
    se, qe, _ = _SPEC[metric]
    m = (ctx.lvl[:, None] == ctx.lvl[None, :]) == se
    if qe is not None:
        m &= (ctx.q[:, None] == ctx.q[None, :]) == qe
    m &= group_mask(ctx, group)
    np.fill_diagonal(m, False)
    return m.astype(np.float32)


def eq_mean_per_task(Y, M):
    """Y (T,n) in {0,1}; M (n,n) float32 symmetric. Mean over pairs of 1[y_i = y_j] per task; NaN if there are no pairs."""
    den = float(M.sum())
    if den <= 0:
        return np.full(Y.shape[0], np.nan)
    Y = np.asarray(Y, np.float32)
    s1 = ((Y @ M) * Y).sum(1)
    Z = 1.0 - Y
    s0 = ((Z @ M) * Z).sum(1)
    return (s1 + s0) / den


def raw_scores_multi(Ys, ctx, metrics=METRICS, groups=GROUPS):
    """Ys {name: binary predictions (T,n)} -> {name: {(metric, group): raw}}. Each pair mask is built once and shared by all models."""
    out = {k: {} for k in Ys}
    for g in groups:
        for m in metrics:
            M = pair_mask(ctx, m, g)
            for k, Y in Ys.items():
                e = eq_mean_per_task(Y, M)
                v = float(np.nanmean(e)) if np.isfinite(e).any() else float("nan")
                out[k][(m, g)] = (1.0 - v) if _SPEC[m][2] else v
    return out


def raw_scores(Y, ctx, metrics=METRICS, groups=GROUPS):
    """{(metric, group): raw} for binary predictions Y (T,n)."""
    return raw_scores_multi({"y": Y}, ctx, metrics, groups)["y"]


def metric_type(metric):
    """'R' for R-IB / R-IB_fold, 'D' for the distinction metrics."""
    return "D" if _SPEC[metric][2] else "R"


def normalize(raw, raw_oracle, raw_ref, min_gap=1e-6):
    """(raw - raw_ref) / (raw_oracle - raw_ref). NaN when the oracle is not separated from the reference by at least min_gap (the score would be noise)."""
    den = raw_oracle - raw_ref
    if not np.isfinite(den) or abs(den) < min_gap or not np.isfinite(raw):
        return float("nan")
    return float((raw - raw_ref) / den)


def noninformative(T, n, seed):
    """Input-independent random predictions: i.i.d. Bernoulli(0.5) per (task, input) (spec 4.2 'non-informative model')."""
    return (np.random.RandomState(seed).rand(T, n) < 0.5).astype(np.float32)


def constant_model(T, n):
    """Alternative reference: the same output for every input (R-IB = 1, D-IB = 0)."""
    return np.zeros((T, n), np.float32)


# ------------------------------------------------------------------------------------------------ calibration (5-B)
def dist_matrix(Y):
    """D_ij = mean_k (y_ki - y_kj)^2 for Y (T,n)."""
    Y = np.asarray(Y, np.float64)
    a = (Y ** 2).mean(0)
    G = (Y.T @ Y) / Y.shape[0]
    return np.maximum(a[:, None] + a[None, :] - 2 * G, 0.0)


def pair_selector(ctx, group, extrap_mask=None):
    """bool (n,n) of the pairs of a calibration group: all | ind | extrap_any (needs extrap_mask (n,) bool)."""
    if group == "all":
        m = np.ones((ctx.n, ctx.n), bool)
    elif group == "ind":
        m = ctx.is_ind[:, None] & ctx.is_ind[None, :]
    elif group == "extrap_any":
        m = extrap_mask[:, None] | extrap_mask[None, :]
    else:
        raise ValueError(group)
    return np.triu(m, 1)


def calibration(Dor, Dmod, sel, n_bins, norm):
    """20 equal-width bins of the oracle pairwise distance over the selected pairs; per bin the mean oracle distance (x) and the mean model distance (y).
    Both are divided by `norm` (max oracle distance over all pairs). Returns (bins list, mean |y_norm - x_norm| over non-empty bins, n_pairs)."""
    x, y = Dor[sel], Dmod[sel]
    if len(x) == 0 or norm <= 0:
        return [], float("nan"), 0
    edges = np.linspace(0.0, float(x.max()), n_bins + 1)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    bins, dev = [], []
    for b in range(n_bins):
        mk = idx == b
        n = int(mk.sum())
        if n == 0:
            bins.append({"bin": b, "n_pairs": 0, "x_oracle_mean": None, "y_model_mean": None, "x_norm": None, "y_norm": None})
            continue
        xm, ym = float(x[mk].mean()), float(y[mk].mean())
        bins.append({"bin": b, "n_pairs": n, "x_oracle_mean": xm, "y_model_mean": ym, "x_norm": xm / norm, "y_norm": ym / norm})
        dev.append(abs(ym - xm) / norm)
    return bins, float(np.mean(dev)), int(len(x))


# ------------------------------------------------------------------------------------------------ 5-D
def ols_line(s, yhat):
    """yhat = slope * s + intercept over the inputs; also R2 of the line. NaN if s is constant."""
    s, yhat = np.asarray(s, float), np.asarray(yhat, float)
    if len(s) < 2 or s.std() < 1e-12:
        return float("nan"), float("nan"), float("nan")
    A = np.stack([s, np.ones_like(s)], 1)
    coef, *_ = np.linalg.lstsq(A, yhat, rcond=None)
    res = yhat - A @ coef
    sst = ((yhat - yhat.mean()) ** 2).sum()
    return float(coef[0]), float(coef[1]), (float(1 - (res ** 2).sum() / sst) if sst > 1e-12 else float("nan"))
