"""numpy-only linear algebra for Q3: ridge (inner GroupKFold alpha selection), unit directions, variance-matched random directions,
INLP and LEACE. Implemented directly (no external library).

LEACE: written from the closed form of Belrose et al. 2023 ("LEACE: Perfect linear concept erasure in closed form") as remembered by the
author; NOT compared with the paper or the official implementation (unverified). The selftest only checks the property that matters
here: the linear regression R2 of the label from the erased features is ~0.
INLP: Ravfogel et al. 2020 (Iterative Nullspace Projection), implemented as: fit ridge regressor, project out its direction, repeat.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------------------------------ ridge
def _standardize(X, eps=1e-6):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + eps
    return mu, sd


def group_folds(groups, k, seed):
    """Assign each group id to one of k folds. Returns fold index per sample."""
    rng = np.random.RandomState(seed)
    u = np.unique(groups)
    rng.shuffle(u)
    fold_of = {g: i % k for i, g in enumerate(u)}
    return np.array([fold_of[g] for g in groups])


def ridge_fit(X, y, groups, alphas, k, seed):
    """Ridge on standardized X. alpha chosen by GroupKFold CV INSIDE the given (training) data only.
    Returns dict(mu, sd, w (standardized space), b, alpha, w_raw, u (unit raw-space direction))."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu, sd = _standardize(X)
    Xs = (X - mu) / sd
    ym = y.mean()
    yc = y - ym
    d = X.shape[1]
    folds = group_folds(groups, min(k, len(np.unique(groups))), seed) if k > 1 and len(np.unique(groups)) > 1 else None
    best_a, best_e = alphas[0], np.inf
    if folds is not None and len(alphas) > 1:
        err = np.zeros(len(alphas))
        for f in np.unique(folds):
            tr, va = folds != f, folds == f
            Xt, yt = Xs[tr], yc[tr] - yc[tr].mean()
            lam, V = np.linalg.eigh(Xt.T @ Xt)
            bv = V.T @ (Xt.T @ yt)
            for i, a in enumerate(alphas):
                w = V @ (bv / (np.maximum(lam, 0) + a))
                err[i] += float(((Xs[va] @ w + yc[tr].mean() - yc[va]) ** 2).sum())
        best_a = alphas[int(np.argmin(err))]
    w = np.linalg.solve(Xs.T @ Xs + best_a * np.eye(d), Xs.T @ yc)
    w_raw = w / sd[0]
    nrm = np.linalg.norm(w_raw)
    return {"mu": mu, "sd": sd, "w": w, "b": ym, "alpha": float(best_a), "w_raw": w_raw, "u": w_raw / nrm if nrm > 0 else w_raw}


def ridge_predict(m, X):
    return ((np.asarray(X, np.float64) - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def r2(y, h):
    y = np.asarray(y, float)
    h = np.asarray(h, float)
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - h) ** 2).sum() / sst) if sst > 1e-12 else float("nan")


# ------------------------------------------------------------------------------------------------ variance-matched random directions
def cov_eig(X, rel_tol=1e-10):
    """Eigen-decomposition of the covariance of X (N,d), restricted to the numerical range (the 'data subspace').
    Returns (lam (k,), V (d,k))."""
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / max(len(X) - 1, 1)
    lam, V = np.linalg.eigh(C)
    keep = lam > rel_tol * lam.max()
    return lam[keep], V[:, keep]


def matched_random_direction(lam, V, target_var, rng, orth=None, iters=80):
    """Random unit vector r (in the data subspace spanned by V) with r' C r = target_var (C = V diag(lam) V').
    NOT an isotropic Gaussian: a random direction is drawn, tilted toward the lowest/highest-variance eigen-direction until it brackets
    the target, then bisected along the path between the two. orth: (m,d) orthonormal rows (other directions to stay orthogonal to).
    Returns (r (d,), achieved_var, clipped: bool)."""
    k = len(lam)
    O = None if orth is None or len(orth) == 0 else (np.asarray(orth) @ V)         # coefficients of the constraints in eigenbasis (m,k)

    def prep(v):
        if O is not None:
            v = v - O.T @ (O @ v)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    q = lambda v: float((lam * v * v).sum())
    emin, emax = np.zeros(k), np.zeros(k)
    emin[int(np.argmin(lam))] = 1.0
    emax[int(np.argmax(lam))] = 1.0
    lo_t, hi_t = float(lam.min()), float(lam.max())
    clipped = not (lo_t <= target_var <= hi_t)
    tgt = min(max(target_var, lo_t), hi_t)

    def tilt(e, want_low):
        g = rng.randn(k)
        c = 0.0
        for _ in range(60):
            v = prep(g + c * e)
            if (want_low and q(v) <= tgt) or ((not want_low) and q(v) >= tgt):
                return v
            c = 0.5 if c == 0 else c * 2
        return prep(e)

    va, vb = tilt(emin, True), tilt(emax, False)
    a, b = 0.0, 1.0
    for _ in range(iters):
        t = (a + b) / 2
        v = prep((1 - t) * va + t * vb)
        if q(v) < tgt:
            a = t
        else:
            b = t
    v = prep((1 - (a + b) / 2) * va + ((a + b) / 2) * vb)
    return V @ v, q(v), clipped


# ------------------------------------------------------------------------------------------------ subspace removal
def inlp(X, y, groups, n_iter, alphas, k_folds, seed):
    """Iterative nullspace projection with a ridge regressor. Returns (B (d,n_iter) orthonormal columns, per-iteration train-CV-free
    linear R2 of the current regressor on the (current) data)."""
    Xc = np.asarray(X, np.float64).copy()
    d = Xc.shape[1]
    B = np.zeros((d, 0))
    r2s = []
    for it in range(n_iter):
        m = ridge_fit(Xc, y, groups, alphas, k_folds, seed + it)
        r2s.append(r2(y, ridge_predict(m, Xc)))
        b = m["w_raw"].copy()
        if B.shape[1]:
            b = b - B @ (B.T @ b)
        n = np.linalg.norm(b)
        if n < 1e-12:
            break
        b /= n
        B = np.concatenate([B, b[:, None]], 1)
        Xc = Xc - np.outer(Xc @ b, b)
    return B, r2s


def leace_fit(X, Y, eps=1e-8):
    """Returns (M (d,d), mu (d,)) with erasure x' = x - (x - mu) @ M.T  (M = W+ P_{W Sxy} W)."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    mu = X.mean(0)
    Xc = X - mu
    Yc = Y - Y.mean(0)
    n = len(X)
    Sxx = Xc.T @ Xc / max(n - 1, 1)
    Sxy = Xc.T @ Yc / max(n - 1, 1)
    lam, V = np.linalg.eigh(Sxx)
    keep = lam > eps * lam.max()
    lam, V = lam[keep], V[:, keep]
    W = (V / np.sqrt(lam)) @ V.T
    Wp = (V * np.sqrt(lam)) @ V.T
    A = W @ Sxy
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    U = U[:, s > 1e-10 * max(s.max(), 1e-300)]
    P = U @ U.T
    return Wp @ P @ W, mu


def apply_removal(X, M, mu, strength=1.0):
    """x' = x - strength * (x - mu) @ M.T  (works for (..., d))."""
    return X - strength * ((X - mu) @ M.T)


def projector_from_basis(B):
    return B @ B.T, np.zeros(B.shape[0])
