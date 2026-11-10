"""Residual-information measures (experiment 012) on pooled / token hidden features of ONE depth and ONE intervention setting.
  a_level_logistic     : L2 multinomial logistic regression, input pooled z, target = level of the 3 training levels (one-hot); C chosen by GroupKFold
                         (groups = episodes) INSIDE the probe-train episodes. Evaluated on test episodes of the training levels only (split 'train').
  b_mlp                : exp002 mlp probe (1 hidden layer relu) on pooled z, R2 of log_nominal_area_ratio. Settings from config measure.b_mlp.
  b_attentive          : exp002 attentive probe on token z (same implementation and settings as 002 / 004).
  c_action_only_linear : exp002 ridge (linear_pooled implementation) on the pooled action/proprio coordinates only.
  d_linear_pooled      : exp002 ridge (linear_pooled) on pooled z (reference; compared with 009).
Data separation (skeptic I4): every probe is fitted on rows of probe-train episodes = train-split episodes NOT used for the eraser direction, and evaluated
on test-split episodes. `assert_probe_split` / `assert_rows_in` raise AssertionError (not `assert`, so they survive python -O).
numpy + lazy torch. Python 3.9 compatible."""
from __future__ import annotations

import numpy as np

from exp002 import directions as D
from exp002 import probes as PR

ROLE2SPLIT = {"trained": "train", "interp": "interp", "extrap": "extrap"}
SPLITS = ["train", "interp", "extrap", "all"]
MEASURES = ["a_level_logistic", "b_mlp", "b_attentive", "c_action_only_linear", "d_linear_pooled"]

# Implementation defaults NOT given in config.yaml (the config fixes only "L2 logistic, C by GroupKFold within probe-train episodes"). Recorded in results.json.
LOGISTIC_IMPL = {"objective": "C * sum_i CE_i + 0.5 * ||W||^2 (+ 0.5e-6 * ||b||^2 for conditioning), inputs standardized with probe-train statistics",
                 "C_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], "c_selection_criterion": "lowest summed validation log-loss, ties -> smaller C",
                 "inner_folds": "cfg002 probe_impl.inner_folds (GroupKFold by episode), fold seed 0", "solver": "damped Newton with backtracking, float64 (torch), max_iter 40",
                 "bias_penalty": 1e-6, "tol_rel": 1e-6}


# ------------------------------------------------------------------------------------------------ data separation
def assert_probe_split(train_idx, test_idx, fit_ep, Etr, E):
    """probe-train episodes = train-split episodes not in fit_ep ; test episodes = test split. Raises AssertionError otherwise."""
    tr, te, fit = (np.asarray(x, int) for x in (train_idx, test_idx, fit_ep))
    if len(tr) == 0 or len(te) == 0:
        raise AssertionError("empty probe-train or test episode set")
    if len(np.unique(tr)) != len(tr) or len(np.unique(te)) != len(te):
        raise AssertionError("duplicate episode ids")
    if np.intersect1d(tr, fit).size:
        raise AssertionError("probe-train episodes overlap the eraser direction-fit episodes: %s" % np.intersect1d(tr, fit)[:5])
    if np.intersect1d(tr, te).size or np.intersect1d(fit, te).size:
        raise AssertionError("test episodes overlap probe-train or direction-fit episodes")
    if tr.max() >= Etr or (len(fit) and fit.max() >= Etr):
        raise AssertionError("probe-train / direction-fit episodes must come from the train split (index < %d)" % Etr)
    if te.min() < Etr or te.max() >= E:
        raise AssertionError("test episodes must be the test split (index in [%d, %d))" % (Etr, E))


def probe_split(Etr, E, fit_ep):
    fit_ep = np.asarray(fit_ep, int)
    held = np.setdiff1d(np.arange(Etr), fit_ep)
    te = np.arange(Etr, E)
    assert_probe_split(held, te, fit_ep, Etr, E)
    return held, te


def assert_rows_in(row_episode_ids, allowed, forbidden, what):
    """Row-level check: every training row belongs to an allowed episode and to none of the forbidden ones."""
    ids = np.asarray(row_episode_ids, int)
    if not np.isin(ids, np.asarray(allowed, int)).all():
        raise AssertionError("%s: rows from episodes outside the allowed set" % what)
    if np.isin(ids, np.asarray(forbidden, int)).any():
        raise AssertionError("%s: rows from forbidden (direction-fit or test) episodes" % what)


# ------------------------------------------------------------------------------------------------ logistic regression (torch float64, Newton)
def _tdev(cfg2):
    import torch
    try:
        return PR.get_device(cfg2)
    except Exception:
        return torch.device("cpu")


def _newton(X1, Y1h, C, pen, th0, max_iter=40, tol=1e-6):
    import torch
    N, d1 = X1.shape
    K = Y1h.shape[1]
    th = th0.clone()

    def fobj(t):
        Z = X1 @ t.T
        return C * (torch.logsumexp(Z, 1) - (Z * Y1h).sum(1)).sum() + 0.5 * (pen * t * t).sum()

    f = fobj(th)
    scale = max(1.0, C * N)
    eye = torch.eye(K, dtype=X1.dtype, device=X1.device)
    for _ in range(max_iter):
        Pm = torch.softmax(X1 @ th.T, 1)
        G = C * (Pm - Y1h).T @ X1 + pen * th
        if float(G.abs().max()) < tol * scale:
            break
        H = torch.zeros(K * d1, K * d1, dtype=X1.dtype, device=X1.device)
        for k in range(K):
            for l in range(K):
                w = Pm[:, k] * (eye[k, l] - Pm[:, l])
                H[k * d1:(k + 1) * d1, l * d1:(l + 1) * d1] = C * (X1.T * w) @ X1
        H += torch.diag(pen.repeat(K))
        step = torch.linalg.solve(H, G.reshape(-1)).reshape(K, d1)
        t, gs = 1.0, float((G * step).sum())
        ok = False
        while t > 1e-10:
            fn = fobj(th - t * step)
            if float(fn) <= float(f) - 1e-4 * t * gs:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        th = th - t * step
        f = fn
    return th


def _prep(X, mu, sd, dev):
    import torch
    Xs = (np.asarray(X, np.float64) - mu) / sd
    return torch.cat([torch.as_tensor(Xs, dtype=torch.float64, device=dev), torch.ones(len(Xs), 1, dtype=torch.float64, device=dev)], 1)


def fit_logistic_cv(X, y, groups, K, cfg2, fold_seed=0):
    """Returns predict(X_new)->class ids, info dict(C, cv_logloss by C). Features standardized with probe-train statistics."""
    import torch
    dev = _tdev(cfg2)
    X = np.asarray(X, np.float64)
    y = np.asarray(y, int)
    grid = LOGISTIC_IMPL["C_grid"]
    pen_b = LOGISTIC_IMPL["bias_penalty"]
    d = X.shape[1]
    pen = torch.cat([torch.ones(d, dtype=torch.float64), torch.full((1,), pen_b, dtype=torch.float64)]).to(dev)
    onehot = lambda yy: torch.as_tensor(np.eye(K)[yy], dtype=torch.float64, device=dev)
    k = int(cfg2["probe_impl"]["inner_folds"])
    err = np.zeros(len(grid))
    ng = len(np.unique(groups))
    if ng > 1 and k > 1:
        folds = D.group_folds(groups, min(k, ng), fold_seed)
        for f in np.unique(folds):
            tr, va = folds != f, folds == f
            mu, sd = D._standardize(X[tr])
            Xt, Xv = _prep(X[tr], mu, sd, dev), _prep(X[va], mu, sd, dev)
            Yt, Yv = onehot(y[tr]), onehot(y[va])
            th = torch.zeros(K, d + 1, dtype=torch.float64, device=dev)
            for i, C in enumerate(grid):                      # ascending C: warm start from the previous solution
                th = _newton(Xt, Yt, C, pen, th, tol=LOGISTIC_IMPL["tol_rel"])
                Z = Xv @ th.T
                err[i] += float((torch.logsumexp(Z, 1) - (Z * Yv).sum(1)).sum())
    best = int(np.argmin(err))                                # argmin returns the first (smaller C) on ties
    mu, sd = D._standardize(X)
    Xa, Ya = _prep(X, mu, sd, dev), onehot(y)
    th = _newton(Xa, Ya, grid[best], pen, torch.zeros(K, d + 1, dtype=torch.float64, device=dev), tol=LOGISTIC_IMPL["tol_rel"])
    train_acc = float((torch.argmax(Xa @ th.T, 1).cpu().numpy() == y).mean())

    def predict(Xn):
        return torch.argmax(_prep(Xn, mu, sd, dev) @ th.T, 1).cpu().numpy()

    return predict, {"C": float(grid[best]), "cv_logloss_by_C": {("%g" % c): float(e) for c, e in zip(grid, err)}, "train_accuracy": train_acc}


# ------------------------------------------------------------------------------------------------ helpers
def _num(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _r2(y, h):
    y, h = np.asarray(y, float).ravel(), np.asarray(h, float).ravel()
    sst = ((y - y.mean()) ** 2).sum()
    return float("nan") if sst <= 1e-12 else float(1 - ((y - h) ** 2).sum() / sst)


def _summ(per_seed, n_eps):
    v = np.array([np.nan if x is None else x for x in per_seed], float)
    return {"per_seed": [_num(x) for x in v], "mean": (_num(np.nanmean(v)) if np.isfinite(v).any() else None), "n_test_episodes": int(n_eps)}


def r2_block(preds_by_seed, Yte, te_split, windows):
    """preds_by_seed: list of (Ete,nF) arrays. -> {window: {split: summary}} with r2 pooled over the test episodes of the split and the frames of the window."""
    out = {}
    for w, cols in windows.items():
        out[w] = {}
        for sp in SPLITS:
            mk = np.ones(len(te_split), bool) if sp == "all" else (te_split == sp)
            if not mk.any():
                continue
            vals = [_r2(Yte[mk][:, cols], H[mk][:, cols]) for H in preds_by_seed]
            out[w][sp] = _summ(vals, mk.sum())
    return out


def probe_cfg(cfg, cfg2):
    """cfg2 copy whose exp002 mlp_check block holds the 012 b_mlp settings (hidden, epochs, lr, weight_decay, batch_size; val fraction 0.1 is fixed inside exp002)."""
    import copy
    c = copy.deepcopy(cfg2)
    bm = cfg["measure"]["b_mlp"]
    c["probe_impl"]["mlp_check"] = {"hidden": int(bm["hidden"]), "epochs": int(bm["epochs"]), "lr": float(bm["lr"]), "weight_decay": float(bm["weight_decay"]), "batch_size": int(bm["batch_size"])}
    return c


# ------------------------------------------------------------------------------------------------ main entry
def measure_all(cfg, cfg2p, pz, pa, tok, Yp, rows, Etr, fit_ep, pos_windows, probe_seeds, which=None):
    """pz (E,nF,dz) pooled z ; pa (E,nF,da) pooled action coords ; tok (E,nF,P,dz) array/memmap or None (then b_attentive is skipped);
    Yp (E,nF) labels ; rows: cache rows (train rows first) ; fit_ep: episodes used for the eraser direction (excluded from probe training).
    Returns dict(meta, <measure>: {window: {split: {per_seed, mean, n_test_episodes}}}, extras)."""
    which = list(MEASURES) if which is None else list(which)
    E, nF = pz.shape[:2]
    held, te = probe_split(Etr, E, fit_ep)
    ep_rows = np.repeat(held, nF)
    assert_rows_in(ep_rows, held, np.concatenate([np.asarray(fit_ep, int), te]), "probe-train rows")
    ytr = np.asarray(Yp)[held].reshape(-1)
    Yte = np.asarray(Yp)[te]
    te_split = np.array([ROLE2SPLIT[rows[i]["role"]] for i in te])
    windows = {w: list(c) for w, c in pos_windows.items()}
    out = {"meta": {"n_probe_train_episodes": int(len(held)), "n_probe_train_rows": int(len(ep_rows)), "n_direction_fit_episodes": int(len(fit_ep)), "n_test_episodes": int(len(te)),
                    "train_row_episode_ids_disjoint_from_fit_and_test": True, "windows": windows}}
    seeds = [int(s) for s in probe_seeds]
    Xz_tr, Xz_te = np.asarray(pz)[held].reshape(len(ep_rows), -1), np.asarray(pz)[te].reshape(len(te) * nF, -1)
    if "a_level_logistic" in which:
        classes = list(cfg2p["data"]["train_levels_b"])
        lv_tr = np.array([rows[i]["level"] for i in held])
        if not all(l in classes for l in lv_tr):
            raise AssertionError("a probe-train episode has a level outside the training levels %s" % classes)
        ycls = np.repeat(np.array([classes.index(l) for l in lv_tr]), nF)
        t_mask = np.array([rows[i]["role"] == "trained" for i in te])
        lv_te = np.array([rows[i]["level"] for i in te])
        yte_cls = np.array([classes.index(l) if l in classes else -1 for l in lv_te])
        if (yte_cls[t_mask] < 0).any():
            raise AssertionError("a 'trained'-role test episode has a level outside the training levels")
        pred, info = fit_logistic_cv(Xz_tr, ycls, ep_rows, len(classes), cfg2p, fold_seed=0)
        Hc = pred(Xz_te).reshape(len(te), nF)
        res = {}
        for w, cols in windows.items():
            yy, hh = yte_cls[t_mask][:, None] * np.ones((1, len(cols)), int), Hc[t_mask][:, cols]
            acc = float((yy == hh).mean())
            per_level = {l: _num((hh[yte_cls[t_mask] == classes.index(l)] == classes.index(l)).mean()) for l in classes}
            res[w] = {"train": {"per_seed": [acc], "mean": acc, "n_test_episodes": int(t_mask.sum()), "per_level_recall": per_level, "chance": 1.0 / len(classes),
                                "note": "deterministic given C (fold seed 0); evaluated on test episodes of the training levels only"}}
        res["extras"] = {"classes": classes, "C": info["C"], "cv_logloss_by_C": info["cv_logloss_by_C"], "train_accuracy": info["train_accuracy"]}
        out["a_level_logistic"] = res
    if "b_mlp" in which:
        preds = [PR.fit_predict("mlp", Xz_tr, ytr, ep_rows, [Xz_te], s, cfg2p)[0].reshape(len(te), nF) for s in seeds]
        out["b_mlp"] = r2_block(preds, Yte, te_split, windows)
    if "b_attentive" in which:
        if tok is None:
            out["b_attentive"] = "skipped (no token features given)"
        else:
            Tr = np.asarray(tok[held]).reshape(len(ep_rows), *tok.shape[2:])
            Te = np.asarray(tok[te]).reshape(len(te) * nF, *tok.shape[2:])
            preds = [PR.fit_predict("attentive", Tr, ytr, ep_rows, [Te], s, cfg2p)[0].reshape(len(te), nF) for s in seeds]
            del Tr, Te
            out["b_attentive"] = r2_block(preds, Yte, te_split, windows)
    for name, (Xa, Xb) in (("c_action_only_linear", (np.asarray(pa)[held].reshape(len(ep_rows), -1), np.asarray(pa)[te].reshape(len(te) * nF, -1))),
                           ("d_linear_pooled", (Xz_tr, Xz_te))):
        if name in which:
            preds, alphas = [], []
            for s in seeds:
                p, m = PR.linear_fit_predict(Xa, ytr, ep_rows, [Xb], s, cfg2p)
                preds.append(p[0].reshape(len(te), nF))
                alphas.append(m["alpha"])
            r = r2_block(preds, Yte, te_split, windows)
            r["extras"] = {"ridge_alpha_per_seed": alphas}
            out[name] = r
    return out
