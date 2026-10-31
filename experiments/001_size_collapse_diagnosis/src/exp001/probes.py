"""Size probes (CPU, numpy + scikit-learn only).

Protocols (config.probe.protocols), all with episode-level splits:
  strat_cv     (i)   train/test on the SAME latent type; GroupKFold over episodes stratified by size group
  logo         (i)   leave-one-group-out (test group never seen in training; extrapolation/interpolation test)
  gt_transfer  (ii)  train on GT encodings only, apply to predicted latents (tf1, ol) of held-out episodes
  upper        (iii) train on GT + predicted latents of all sizes (OoD included), test on held-out predicted latents (upper bound)
Models: ridge (alpha chosen on the first outer fold by inner GroupKFold, then reused) and a small MLP.
Never report R2 alone: every evaluation returns overall R2, within-group R2, 3-way accuracy, slope of predicted vs real
log-area, per-group bias, each with an episode-level cluster-bootstrap CI (resampling episodes within group).
Controls: label_permute (episode-level label shuffle), constant_mean, and extra latent types (random-init ViT, raw pixel grid).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------ sample tables
class Samples:
    """Rows = (episode, frame). x: (N,F) float32; ep, k, grp, cls, y: (N,)."""

    def __init__(self, x, ep, k, grp, cls, y):
        self.x, self.ep, self.k, self.grp, self.cls, self.y = x, ep, k, grp, cls, y

    def __len__(self):
        return len(self.y)

    def subset(self, mask):
        return Samples(self.x[mask], self.ep[mask], self.k[mask], self.grp[mask], self.cls[mask], self.y[mask])


def make_samples(feat, area_real, ep_ids, grp_ids, cls_ids, k_first, k_label_offset=0):
    """feat: (E, T, F) latents for frames k_first .. k_first+T-1 of the E episodes in ep_ids.
    The label is log(area_real at frame k - k_label_offset) (k_label_offset=1: input frame of a teacher-forced pass)."""
    E, T, Fd = feat.shape
    ks = np.arange(k_first, k_first + T)
    y = np.log(np.maximum(area_real[:, ks - k_label_offset], 1).astype(np.float64))
    return Samples(
        feat.reshape(E * T, Fd).astype(np.float32),
        np.repeat(ep_ids, T), np.tile(ks, E), np.repeat(grp_ids, T), np.repeat(cls_ids, T), y.reshape(-1),
    )


def concat_samples(lst):
    return Samples(*[np.concatenate([getattr(s, a) for s in lst]) for a in ("x", "ep", "k", "grp", "cls", "y")])


# ------------------------------------------------------------------------------ splits
def strat_folds(ep_ids, ep_grp, n_folds, seed):
    """Assign each episode to a fold; every fold gets (nearly) every group. Returns {episode id: fold}."""
    rng = np.random.RandomState(seed)
    fold = {}
    for g in np.unique(ep_grp):
        eps = ep_ids[ep_grp == g].copy()
        rng.shuffle(eps)
        for i, e in enumerate(eps):
            fold[int(e)] = i % n_folds
    return fold


def logo_folds(ep_ids, ep_grp):
    return {int(e): int(g) for e, g in zip(ep_ids, ep_grp)}


# ------------------------------------------------------------------------------ probe fitting
def _standardize(xtr, xte_list):
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    return (xtr - mu) / sd, [(x - mu) / sd for x in xte_list]


def pick_alpha(xtr, ytr, eptr, alphas, seed):
    """Inner GroupKFold(3) over train episodes."""
    gkf = GroupKFold(n_splits=3)
    best, best_err = alphas[0], np.inf
    splits = list(gkf.split(xtr, ytr, eptr))
    for a in alphas:
        err = 0.0
        for tr, va in splits:
            m = Ridge(alpha=a).fit(xtr[tr], ytr[tr])
            err += float(((m.predict(xtr[va]) - ytr[va]) ** 2).sum())
        if err < best_err:
            best, best_err = a, err
    return best


class ProbeFactory:
    def __init__(self, pcfg):
        self.pcfg = pcfg
        self.alpha_cache = {}

    def fit_predict(self, kind, key, tr: Samples, te_list, seed):
        """Fit on tr, return [(yhat, chat) for each Samples in te_list]. kind in ridge|mlp|const."""
        xtr_raw = tr.x
        xtr, xte = _standardize(xtr_raw, [t.x for t in te_list])
        cmask = tr.cls >= 0
        outs = []
        if kind == "const":
            maj = np.bincount(tr.cls[cmask]).argmax() if cmask.any() else -1
            return [(np.full(len(t), tr.y.mean()), np.full(len(t), maj)) for t in te_list]
        if kind == "ridge":
            rc = self.pcfg["models"]["ridge"]
            if key not in self.alpha_cache:
                self.alpha_cache[key] = pick_alpha(xtr, tr.y, tr.ep, rc["alphas"], seed)
            a = self.alpha_cache[key]
            reg = Ridge(alpha=a).fit(xtr, tr.y)
            clf = RidgeClassifier(alpha=a).fit(xtr[cmask], tr.cls[cmask]) if cmask.sum() > 0 and len(np.unique(tr.cls[cmask])) > 1 else None
            for x in xte:
                outs.append((reg.predict(x), clf.predict(x) if clf is not None else np.full(len(x), -1)))
            return outs
        if kind == "mlp":
            mc = self.pcfg["models"]["mlp"]
            ym, ys = tr.y.mean(), tr.y.std() + 1e-6
            kw = dict(hidden_layer_sizes=(mc["hidden"],), max_iter=mc["max_iter"], alpha=mc["alpha"],
                      early_stopping=mc["early_stopping"], random_state=seed)
            reg = MLPRegressor(**kw).fit(xtr, (tr.y - ym) / ys)
            clf = MLPClassifier(**kw).fit(xtr[cmask], tr.cls[cmask]) if cmask.sum() > 0 and len(np.unique(tr.cls[cmask])) > 1 else None
            for x in xte:
                outs.append((reg.predict(x) * ys + ym, clf.predict(x) if clf is not None else np.full(len(x), -1)))
            return outs
        raise ValueError(kind)


# ------------------------------------------------------------------------------ metrics from per-episode sufficient statistics
STAT_COLS = ["n", "sy", "sh", "syy", "shh", "syh", "n_cls", "correct"]


def episode_stats(s: Samples, yhat, chat, mask):
    """Rows per episode (for samples with mask): sums needed for every metric, so bootstrap needs no refit."""
    eps = np.unique(s.ep[mask])
    S = np.zeros((len(eps), len(STAT_COLS)))
    G = np.zeros(len(eps), int)
    for i, e in enumerate(eps):
        m = mask & (s.ep == e)
        y, h = s.y[m], yhat[m]
        hasc = s.cls[m] >= 0
        S[i] = [m.sum(), y.sum(), h.sum(), (y * y).sum(), (h * h).sum(), (y * h).sum(), hasc.sum(), ((chat[m] == s.cls[m]) & hasc).sum()]
        G[i] = s.grp[m][0]
    return eps, S, G


def metrics_from_stats(S, G, w, group_ids):
    """S (E,c) stats, G (E,) group, w (E,) episode multiplicities. Returns dict of scalars."""
    Sw = S * w[:, None]
    n, sy, sh, syy, shh, syh = [Sw[:, i].sum() for i in range(6)]
    out = {}
    vy = syy - sy * sy / max(n, 1)
    out["r2_overall"] = 1 - (syy - 2 * syh + shh) / vy if vy > 1e-12 else np.nan
    cov = syh - sy * sh / max(n, 1)
    out["slope"] = cov / vy if vy > 1e-12 else np.nan
    out["acc3"] = Sw[:, 7].sum() / Sw[:, 6].sum() if Sw[:, 6].sum() > 0 else np.nan
    sse_c, sst_c, bias, gm_y, gm_h = 0.0, 0.0, {}, [], []
    for g in group_ids:
        m = G == g
        if not m.any() or Sw[m, 0].sum() == 0:
            continue
        ng, sgy, sgh, sgyy, sgshh, sgyh = [Sw[m][:, i].sum() for i in range(6)]
        # within-group R2: group means of y and of y_hat are removed first (bias is reported separately in bias_by_group)
        sst_c += sgyy - sgy * sgy / ng
        sse_c += (sgyy - sgy * sgy / ng) - 2 * (sgyh - sgy * sgh / ng) + (sgshh - sgh * sgh / ng)
        bias[int(g)] = (sgh - sgy) / ng
        gm_y.append(sgy / ng)
        gm_h.append(sgh / ng)
    out["r2_within_group"] = float(1 - sse_c / sst_c) if sst_c > 1e-12 else np.nan
    out["bias_by_group"] = bias
    if len(gm_y) >= 2 and np.ptp(gm_y) > 1e-9:
        out["slope_group_means"] = float(np.polyfit(gm_y, gm_h, 1)[0])
    else:
        out["slope_group_means"] = np.nan
    return out


def bootstrap_metrics(S, G, group_ids, n_boot, seed):
    """Cluster bootstrap over episodes, resampling within each group. Returns {metric: (lo, hi)} (2.5/97.5 percentiles)."""
    rng = np.random.RandomState(seed)
    E = len(G)
    draws = {k: [] for k in ("r2_overall", "slope", "acc3", "r2_within_group", "slope_group_means")}
    idx_by_g = {g: np.where(G == g)[0] for g in group_ids}
    for _ in range(n_boot):
        w = np.zeros(E)
        for g, idx in idx_by_g.items():
            if len(idx):
                np.add.at(w, rng.choice(idx, len(idx), replace=True), 1)
        m = metrics_from_stats(S, G, w, group_ids)
        for k in draws:
            draws[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if np.isfinite(v).any() else (np.nan, np.nan)
            for k, v in draws.items()}


def summarize(s: Samples, yhat, chat, mask, group_ids, n_boot, seed):
    eps, S, G = episode_stats(s, yhat, chat, mask)
    pt = metrics_from_stats(S, G, np.ones(len(G)), group_ids)
    ci = bootstrap_metrics(S, G, group_ids, n_boot, seed) if n_boot > 0 else {}
    pt["ci"] = ci
    pt["n_episodes"] = int(len(G))
    return pt


# ------------------------------------------------------------------------------ protocol driver
def run_protocol(protocol, ptype_targets, samples, ep_ids, ep_grp, pcfg, kind, pool_key, seed, permute=False, min_k=1):
    """samples: {type: Samples}. ptype_targets: types to evaluate. Returns {type: (yhat, chat)} aligned with samples[type].
    permute: shuffle labels across episodes (episode-level label permutation) in the TRAINING data only."""
    factory = run_protocol.factory.setdefault(pool_key + kind, ProbeFactory(pcfg))
    rng = np.random.RandomState(seed + 1000)
    if protocol == "logo":
        fold = logo_folds(ep_ids, ep_grp)
    else:
        fold = strat_folds(ep_ids, ep_grp, pcfg["n_folds"], seed)
    n_f = len(set(fold.values()))
    preds = {t: (np.full(len(samples[t]), np.nan), np.full(len(samples[t]), -1)) for t in ptype_targets}
    eps_of_fold = {f: np.array([e for e, ff in fold.items() if ff == f]) for f in set(fold.values())}
    for f, test_eps in eps_of_fold.items():
        for t in ptype_targets:
            s = samples[t]
            te_m = np.isin(s.ep, test_eps)
            tr_m = ~te_m
            if protocol in ("strat_cv", "logo"):
                tr = s.subset(tr_m)
            elif protocol == "gt_transfer":
                g = samples["gt"]
                tr = g.subset(~np.isin(g.ep, test_eps))
            elif protocol == "upper":
                parts = [samples[n].subset(~np.isin(samples[n].ep, test_eps)) for n in ("gt", "tf1", "ol") if n in samples]
                tr = concat_samples(parts)
            else:
                raise ValueError(protocol)
            if permute:
                tr = _permute_labels(tr, rng)
            if len(tr) == 0 or te_m.sum() == 0:
                continue
            key = (protocol, t if protocol in ("strat_cv", "logo") else "gt", pool_key, kind, "perm" if permute else "real")
            yh, ch = factory.fit_predict(kind, key, tr, [s.subset(te_m)], seed)[0]
            preds[t][0][te_m] = yh
            preds[t][1][te_m] = ch
    return preds


run_protocol.factory = {}


def _permute_labels(tr: Samples, rng):
    """Shuffle (y, cls) sequences between episodes with the same number of frames (episode-level permutation)."""
    eps = np.unique(tr.ep)
    perm = eps.copy()
    rng.shuffle(perm)
    y, cls = tr.y.copy(), tr.cls.copy()
    for e, pe in zip(eps, perm):
        a, b = np.where(tr.ep == e)[0], np.where(tr.ep == pe)[0]
        n = min(len(a), len(b))
        # align by frame index k
        ka, kb = tr.k[a], tr.k[b]
        common = np.intersect1d(ka, kb)
        for c in common:
            y[a[ka == c]] = tr.y[b[kb == c]]
            cls[a[ka == c]] = tr.cls[b[kb == c]]
    return Samples(tr.x, tr.ep, tr.k, tr.grp, cls, y)
