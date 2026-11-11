"""Gram-matrix (dual) implementations of linear CKA and cross-validated ridge R^2. numpy only.
Features of one input are flattened (P*d) token vectors; everything is computed from n x n linear Grams K = X X^T (float64), so no (n, P*d) ops
are needed after the Grams are built. Ridge: Y_c ~ X_c with train-fold centering (intercept), prediction K_ba (K_aa + alpha I)^-1 Y_c."""
from __future__ import annotations

import numpy as np

INNER_FOLDS = 3          # experimenter default: inner folds of the nested alpha selection (config only fixes the 5 outer folds + alpha grid)


def gram(X, chunk=4096):
    """X (n, D) any float dtype -> (n, n) float64 linear Gram, accumulated in column chunks."""
    n, D = X.shape
    G = np.zeros((n, n), np.float64)
    for s in range(0, D, chunk):
        c = np.asarray(X[:, s:s + chunk], dtype=np.float64)
        G += c @ c.T
    return G


def center(K):
    r = K.mean(1, keepdims=True)
    return K - r - r.T + K.mean()


def cka(Kx, Ky):
    a, b = center(Kx), center(Ky)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def unit_scale(K, D):
    """Global scalar s so that the centered features have mean square 1 per element: s = trace(center(K)) / (n*D). K/s is the Gram of X/sqrt(s).
    A single scalar, so linear structure is untouched; it only fixes the meaning of the alpha grid."""
    n = K.shape[0]
    return float(np.trace(center(K)) / (n * D))


def folds_by_group(groups, k, seed):
    """Fold id per sample; whole groups (episodes) go to one fold. Groups are shuffled with RandomState(seed) and dealt round-robin."""
    groups = np.asarray(groups)
    u = np.unique(groups)
    perm = np.random.RandomState(seed).permutation(len(u))
    gf = {g: i % k for i, g in zip(range(len(u)), u[perm])}
    return np.array([gf[g] for g in groups])


def _fold_sse(Kx, Ky, a, b, alphas):
    """Per-test-sample squared error (len(alphas), nb) of ridge Y from X trained on indices a, tested on b, plus per-sample total sum of squares around
    the train mean (nb,). All in the dual (Gram) form."""
    Kaa, Kba = Kx[np.ix_(a, a)], Kx[np.ix_(b, a)]
    ra, rb, s = Kaa.mean(1), Kba.mean(1), Kaa.mean()
    Kaa_c = Kaa - ra[:, None] - ra[None, :] + s
    Kba_c = Kba - rb[:, None] - ra[None, :] + s
    Yaa, Yba = Ky[np.ix_(a, a)], Ky[np.ix_(b, a)]
    ya, yb, sy = Yaa.mean(1), Yba.mean(1), Yaa.mean()
    Yaa_c = Yaa - ya[:, None] - ya[None, :] + sy
    Yba_c = Yba - yb[:, None] - ya[None, :] + sy                  # (nb, m) = Yc_b Yc_a^T
    ydiag = np.diag(Ky)[b] - 2 * yb + sy                          # ||Yc_b,i||^2
    lam, V = np.linalg.eigh(Kaa_c)
    lam = np.clip(lam, 0, None)
    B = Kba_c @ V
    Z = V.T @ Yba_c.T                                             # (m, nb)
    U = V.T @ Yaa_c @ V
    out = np.empty((len(alphas), len(b)))
    for i, al in enumerate(alphas):
        w = 1.0 / (lam + al)
        Bw = B * w
        cross = (Bw * Z.T).sum(1)
        nrm = ((Bw @ U) * Bw).sum(1)
        out[i] = ydiag - 2 * cross + nrm
    return out, ydiag


def ridge_cv(Kx, Ky, groups, D, alphas, k=5, seed=0, inner_k=INNER_FOLDS):
    """Nested-CV ridge of Y from X. Kx, Ky: (n, n) linear Grams; D: number of X features (for unit_scale of X). Folds split by `groups` (episodes).
    Returns dict: sse (n,), sst (n,) per-sample (variance-weighted pooled R^2 = 1 - sum sse / sum sst, SST around the train-fold mean), r2, alphas chosen per outer fold."""
    n = Kx.shape[0]
    Kx = Kx / unit_scale(Kx, D)
    groups = np.asarray(groups)
    fold = folds_by_group(groups, k, seed)
    sse, sst, chosen = np.zeros(n), np.zeros(n), []
    for f in range(k):
        b = np.where(fold == f)[0]
        a = np.where(fold != f)[0]
        ifold = folds_by_group(groups[a], inner_k, seed + 1000 + f)
        tot = np.zeros(len(alphas))
        for g in range(inner_k):
            ia, ib = a[ifold != g], a[ifold == g]
            e, _ = _fold_sse(Kx, Ky, ia, ib, alphas)
            tot += e.sum(1)
        j = int(np.argmin(tot))
        e, yd = _fold_sse(Kx, Ky, a, b, [alphas[j]])
        sse[b], sst[b] = e[0], yd
        chosen.append(float(alphas[j]))
    return {"sse": sse, "sst": sst, "r2": float(1 - sse.sum() / sst.sum()), "alphas": chosen}


def r2_from(sse, sst, idx=None):
    if idx is None:
        return float(1 - sse.sum() / sst.sum())
    return float(1 - sse[idx].sum() / sst[idx].sum())


def episode_counts(groups, n_boot, seed):
    """(n_boot, n_groups) multinomial-style resample counts (episodes drawn with replacement) and the unique group ids."""
    u, inv = np.unique(groups, return_inverse=True)
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(u), size=(n_boot, len(u)))
    C = np.zeros((n_boot, len(u)))
    for i in range(n_boot):
        C[i] = np.bincount(draws[i], minlength=len(u))
    return u, inv, C, draws


def ci(vals, level):
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


def boot_r2(sse, sst, groups, n_boot, level, seed):
    u, inv, C, _ = episode_counts(groups, n_boot, seed)
    se = np.bincount(inv, weights=sse, minlength=len(u))
    st = np.bincount(inv, weights=sst, minlength=len(u))
    v = 1 - (C @ se) / (C @ st)
    return ci(v, level), v


def boot_cka(Kx, Ky, groups, n_boot, level, seed):
    u, inv, C, draws = episode_counts(groups, n_boot, seed)
    members = [np.where(inv == g)[0] for g in range(len(u))]
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([members[g] for g in draws[i]])
        vals[i] = cka(Kx[np.ix_(idx, idx)], Ky[np.ix_(idx, idx)])
    return ci(vals, level), vals


def null_shuffle(Kx, Ky, groups, D, alphas, k, seed, n_perm, with_r2=True):
    """Shuffled-order null: the pairing between inputs is destroyed by permuting the rows/cols of Kx (all inputs, ignoring episodes); folds unchanged."""
    rng = np.random.RandomState(seed)
    c, r = [], []
    for _ in range(n_perm):
        p = rng.permutation(Kx.shape[0])
        Kp = Kx[np.ix_(p, p)]
        c.append(cka(Kp, Ky))
        if with_r2:
            r.append(ridge_cv(Kp, Ky, groups, D, alphas, k, seed)["r2"])
    return np.array(c), np.array(r)


def summarize(v):
    v = np.asarray(v)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None, "q95": float(np.percentile(v, 95)), "max": float(v.max()), "min": float(v.min())}
