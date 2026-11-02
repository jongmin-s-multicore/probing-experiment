"""Shared Q1 / Q2 machinery (numpy, torch only inside probes): labels, one probe fit -> per-episode statistic cells, summaries.

Cell = array S (P_probe_seeds, n_test_eps, 7) of per-episode sufficient statistics (stats.STAT_COLS) of ONE probe configuration on the
fixed list of evaluation episodes (cache rows with part == 'test', in cache order), restricted to the frames of one window.
All cells share the episode order, so bootstrap differences between cells are paired.

Labels: the label of a frame-f sample is defined at THE SAME frame f as the latent that is read (no one-frame offset, the 001 issue).
  log_nominal_area_ratio : episode constant, log(area_ratio of the level)           (config q1.primary_target)
  log_frame_mask_area    : log(max(mask area px at frame f, 1)), mask of the same 224x224 frame  (config q1.secondary_target)"""
from __future__ import annotations

import numpy as np

from . import probes as PR
from . import stats as ST


def split_rows(rows):
    tr = [r for r in rows if r["part"] == "train"]
    te = [r for r in rows if r["part"] == "test"]
    return tr, te


def make_labels(cfg, rows, mask_feat, target):
    """rows: list of cache rows (len E) ; mask_feat (E,13,3). Returns Y (E,13)."""
    E = len(rows)
    if target == "log_nominal_area_ratio":
        return np.repeat(np.array([r["nominal_log_area"] for r in rows])[:, None], 13, 1)
    if target == "log_frame_mask_area":
        return np.log(np.maximum(np.asarray(mask_feat)[:, :, 0], 1.0))
    raise ValueError(target)


def windows(cfg):
    f = cfg["frames"]
    uni = sorted(set(f["early_flat"]) | set(f["late_deformed"]))
    return {"all": list(f["all"]), "early": list(f["early_flat"]), "late": list(f["late_deformed"]), "union": uni}


def role_masks(rows_test):
    roles = np.array([r["role"] for r in rows_test])
    lv = np.array([r["level"] for r in rows_test])
    m = {"interp": roles == "interp", "extrap": roles == "extrap", "unseen": (roles == "interp") | (roles == "extrap"),
         "trained": roles == "trained", "all": np.ones(len(rows_test), bool)}
    return m


def level_index(rows_test, levels):
    return np.array([levels.index(r["level"]) for r in rows_test])


def fit_predict_cells(cfg, kind, Xtr, Ytr, Xte, Yte, train_frames, eval_frames_by_window, seeds, shuffle_base=None):
    """Fit `kind` on training episodes (frames train_frames) for every probe seed, predict all test samples at the frames in
    eval_frames_by_window, return (cells {window: S (P,n_te,7)}, predmean {window: (P,n_te)}).
    Xtr (Etr,13,...) , Xte (Ete,13,...) : tokens (E,13,196,d) for 'attentive'/'linear_pooled' (pooled inside), or (E,13,f) features for 'ridge_feat'.
    shuffle_base: None or int -> training label rows permuted across episodes (episode-level shuffle; training labels only)."""
    Etr, Ete = Xtr.shape[0], Xte.shape[0]
    all_eval = sorted(set(f for fr in eval_frames_by_window.values() for f in fr))
    ep_tr = np.repeat(np.arange(Etr), len(train_frames))
    cells = {w: [] for w in eval_frames_by_window}
    pm = {w: [] for w in eval_frames_by_window}
    if kind in ("attentive", "linear_pooled"):
        xtr_all = np.asarray(Xtr[:, train_frames])
        xte_all = np.asarray(Xte[:, all_eval])
        if kind == "linear_pooled":
            xtr_all, xte_all = PR.pool(xtr_all), PR.pool(xte_all)
    else:
        xtr_all, xte_all = np.asarray(Xtr[:, train_frames]), np.asarray(Xte[:, all_eval])
    xtr = xtr_all.reshape(Etr * len(train_frames), *xtr_all.shape[2:])
    xte = xte_all.reshape(Ete * len(all_eval), *xte_all.shape[2:])
    pk = "attentive" if kind == "attentive" else "linear_pooled"
    for s in seeds:
        Y = Ytr
        if shuffle_base is not None:
            Y = PR.shuffle_episode_labels(Ytr, shuffle_base + s)
        ytr = Y[:, train_frames].reshape(-1)
        h = PR.fit_predict(pk, xtr, ytr, ep_tr, [xte], s, cfg)[0]
        H = h.reshape(Ete, len(all_eval))
        for w, fr in eval_frames_by_window.items():
            cols = [all_eval.index(f) for f in fr]
            y = Yte[:, fr]
            hh = H[:, cols]
            ep = np.repeat(np.arange(Ete), len(fr))
            cells[w].append(ST.episode_stats(y.reshape(-1), hh.reshape(-1), ep, Ete))
            pm[w].append(hh.mean(1))
    return {w: np.stack(v) for w, v in cells.items()}, {w: np.stack(v) for w, v in pm.items()}


def summarize_cell(S, boot, rmasks, predmean=None, nominal=None):
    """Point estimates + 95% episode-bootstrap CI + per-probe-seed values for every role set and metric."""
    out = {}
    for role, mk in rmasks.items():
        if mk.sum() == 0:
            continue
        d = {}
        for metric in ("r2", "mae", "pearson", "r2_within"):
            b = boot.stat(S, mk, metric)
            d[metric] = {"point": b["point"], "ci95_episode_bootstrap": ST.ci95(b["boot"]), "per_probe_seed": b["per_probe_seed"]}
        if predmean is not None and nominal is not None and role in ("unseen", "interp", "extrap", "all"):
            acc = [ST.pair_order_accuracy(p[mk], nominal[mk]) for p in predmean]
            d["pair_order_accuracy"] = {"point": float(np.nanmean(acc)), "per_probe_seed": acc, "note": "cross-level episode pairs, point estimate only"}
        out[role] = d
    return out


def pool_chunked(X, chunk=256):
    """(N,P,d) array/memmap -> (N,d) float32 mean over patches, in chunks (no full float32 copy)."""
    return np.concatenate([PR.pool(X[i:i + chunk]) for i in range(0, len(X), chunk)]) if len(X) else np.zeros((0, X.shape[-1]), np.float32)


def fit_apply_cells(cfg, kind, Xtr, Ytr, train_frames, apply_sets, seeds):
    """OUTPUT-SPACE measurement (확정 기록 6 item 4a). The probe is fitted ONLY on encoder tokens of the probe-training episodes
    (Xtr (Etr,13,P,d), labels Ytr (Etr,13), frames train_frames = Q1 regime 'all') and then applied unchanged to every set in apply_sets
    {name: (X (Ete,P,d) tokens of one frame / one rollout step of the test episodes, y (Ete,) true label)}. There is no argument through which
    activations of a predictor could enter the fit.
    Returns (cells {name: S (P_seeds, Ete, 7)}, predmean {name: (P_seeds, Ete)}, digest {seed: sha1 of the rounded predictions on the first set})."""
    import hashlib
    Etr = Xtr.shape[0]
    ep_tr = np.repeat(np.arange(Etr), len(train_frames))
    xtr_all = np.asarray(Xtr[:, train_frames])
    if kind == "linear_pooled":
        xtr = PR.pool(xtr_all).reshape(Etr * len(train_frames), -1)
    else:
        xtr = xtr_all.reshape(Etr * len(train_frames), *xtr_all.shape[2:])
    names = list(apply_sets)
    xte = [pool_chunked(apply_sets[n][0]) if kind == "linear_pooled" else apply_sets[n][0] for n in names]
    cells = {n: [] for n in names}
    pm = {n: [] for n in names}
    dig = {}
    for s in seeds:
        ytr = Ytr[:, train_frames].reshape(-1)
        hs = PR.fit_predict("attentive" if kind == "attentive" else "linear_pooled", xtr, ytr, ep_tr, xte, s, cfg)
        for n, h in zip(names, hs):
            y = apply_sets[n][1]
            cells[n].append(ST.episode_stats(y, h, np.arange(len(y)), len(y)))
            pm[n].append(np.asarray(h))
        dig[s] = hashlib.sha1(np.round(np.asarray(hs[0]), 3).astype(np.float64).tobytes()).hexdigest()
    return {n: np.stack(v) for n, v in cells.items()}, {n: np.stack(v) for n, v in pm.items()}, dig
