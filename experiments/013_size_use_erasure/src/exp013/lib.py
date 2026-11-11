"""Numpy-only parts of experiment 013: stage-0 gate regressors, eraser fits (scalar LEACE per train seed, level-mean-match per level), random directions,
displacement norms, bootstrap, Delta_use statistics, verdict, stage-2 correlation. No torch import."""
from __future__ import annotations

import numpy as np

from exp009 import erase as ER

NT = 12   # transitions per episode (frames 0..11 -> 1..12)


# ------------------------------------------------------------------------------------------------ episode sets
def fit_episodes(cfg, ts, Etr):
    """009 rule: RandomState(fit_seed + train_seed).permutation(Etr)[:round(fit_fraction * Etr)], sorted. Indices into the train rows (cache rows 0..Etr-1)."""
    e = cfg["erasure"]
    rng = np.random.RandomState(int(e["fit_seed"]) + int(ts))
    return np.sort(rng.permutation(Etr)[: int(round(float(e["fit_fraction"]) * Etr))])


def ood_halves(cfg, rows_test, levels, train_levels):
    """Level-mean-match split of the test episodes of the OoD levels: per level (config level order) a RandomState(fit_seed)-driven permutation of its
    test episodes; the first half is the FIT half, the rest the EVAL half. Returns (fit_mask, eval_mask) over the test episodes (bool arrays). Train-level episodes:
    fit_mask False (their means are fit on the train split), eval_mask True. One RandomState drawn level by level in `levels` order (reproducible, recorded)."""
    rng = np.random.RandomState(int(cfg["erasure"]["fit_seed"]))
    lv = np.array([r["level"] for r in rows_test])
    fit = np.zeros(len(rows_test), bool)
    ev = np.zeros(len(rows_test), bool)
    for l in levels:
        ids = np.where(lv == l)[0]
        if l in train_levels:
            ev[ids] = True
            continue
        perm = rng.permutation(len(ids))
        h = len(ids) // 2
        fit[ids[np.sort(perm[:h])]] = True
        ev[ids[np.sort(perm[h:])]] = True
    assert not (fit & ev).any()
    return fit, ev


def kfold_groups(n_groups, k, seed=0):
    """Deterministic GroupKFold-like assignment: group g -> fold. (sklearn is not available in the lab env.)"""
    perm = np.random.RandomState(seed).permutation(n_groups)
    fold = np.empty(n_groups, int)
    fold[perm] = np.arange(n_groups) % k
    return fold


# ------------------------------------------------------------------------------------------------ stage 0 gate
def ridge_fit(X, Y, alpha):
    """Standardised ridge. Returns predict(Xnew)."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    ym = Y.mean(0)
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ (Y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ W + ym


def ridge_cv(X, Y, groups, alphas, k, seed=0):
    """alpha chosen by GroupKFold (groups = episode ids) on the training set only; returns (predict, alpha, cv_mse_by_alpha)."""
    ug = np.unique(groups)
    fold_of = dict(zip(ug, kfold_groups(len(ug), k, seed)))
    f = np.array([fold_of[g] for g in groups])
    cv = []
    for a in alphas:
        err = 0.0
        for j in range(k):
            tr, va = f != j, f == j
            p = ridge_fit(X[tr], Y[tr], a)
            err += float(((p(X[va]) - Y[va]) ** 2).sum())
        cv.append(err / Y.size)
    a_best = float(alphas[int(np.argmin(cv))])
    return ridge_fit(X, Y, a_best), a_best, cv


def stage0_samples(pm, act, ids):
    """pm (E,13,d) patch-mean z, act (E,12,da) grouped actions, ids episode indices -> X (n*12, d+da), Y (n*12, d), groups (n*12,)."""
    ids = np.asarray(ids)
    z0, z1 = pm[ids, :NT], pm[ids, 1:NT + 1]
    X = np.concatenate([z0, act[ids]], -1).reshape(-1, pm.shape[-1] + act.shape[-1])
    Y = (z1 - z0).reshape(-1, pm.shape[-1])
    return X, Y, np.repeat(ids, NT)


def stage0_gate(pm, act, rows, levels, alphas, k_inner, ratio_threshold, min_pairs):
    """Per-level ridge regressors (train-split episodes of the level) and the ratio table. For the 6 directed pairs (A, B), A != B:
    ratio = MSE(reg_A on test episodes of B) / MSE(reg_A on test episodes of A)   (config: other-level error / same-level error).
    Diagnostic (no gate): ratio_own = MSE(reg_A on test B) / MSE(reg_B on test B)."""
    lv = np.array([r["level"] for r in rows])
    part = np.array([r["part"] for r in rows])
    reg, alpha = {}, {}
    for l in levels:
        tr = np.where((lv == l) & (part == "train"))[0]
        X, Y, g = stage0_samples(pm, act, tr)
        reg[l], alpha[l], _ = ridge_cv(X, Y, g, alphas, k_inner)
    test = {}
    for l in levels:
        te = np.where((lv == l) & (part == "test"))[0]
        test[l] = stage0_samples(pm, act, te)[:2]
    mse = {(a, b): float(((reg[a](test[b][0]) - test[b][1]) ** 2).mean()) for a in levels for b in levels}
    pairs, npass = [], 0
    for a in levels:
        for b in levels:
            if a == b:
                continue
            r = mse[(a, b)] / mse[(a, a)]
            ok = bool(r >= ratio_threshold)
            npass += ok
            pairs.append({"regressor_level": a, "test_level": b, "ratio_other_over_same": r, "passes": ok, "mse_other": mse[(a, b)], "mse_same": mse[(a, a)],
                          "diagnostic_ratio_vs_own_regressor": mse[(a, b)] / mse[(b, b)]})
    persist = {l: float((test[l][1] ** 2).mean()) for l in levels}
    return {"pairs": pairs, "n_pairs": len(pairs), "n_pairs_pass": int(npass), "ratio_threshold": float(ratio_threshold), "min_pairs": int(min_pairs), "passed": bool(npass >= min_pairs),
            "alpha_by_level": alpha, "mse_matrix": {"%s->%s" % k: v for k, v in mse.items()}, "persistence_mse_by_test_level": persist,
            "n_train_episodes_by_level": {l: int(((lv == l) & (part == "train")).sum()) for l in levels},
            "n_test_episodes_by_level": {l: int(((lv == l) & (part == "test")).sum()) for l in levels}}


# ------------------------------------------------------------------------------------------------ erasers (numpy)
def fit_leace(pm, y_ep, fit_ep, ridge_rel):
    """pm (E,13,dz) patch-mean latents of the Q1 cache rows, y_ep (E,) size label per episode, fit_ep indices. 009 fit_eraser on (episodes x 12 input frames) samples."""
    X = pm[fit_ep, :NT].reshape(-1, pm.shape[-1])
    y = np.repeat(np.asarray(y_ep)[fit_ep], NT)
    return ER.fit_eraser(X, y, "leace_scalar", ridge_rel)


def level_means(token_fn, level_fit_ids, levels, P, dz):
    """Per level the mean over (fit episodes, input frames 0..11) of the tokens, per patch position: (n_levels, P, dz) float64. token_fn(ids) -> (n, >=12, P, dz)."""
    out = np.zeros((len(levels), P, dz))
    cnt = []
    for i, l in enumerate(levels):
        ids = np.asarray(level_fit_ids[l])
        if len(ids) == 0:
            raise ValueError("no fit episodes for level %s" % l)
        s = np.zeros((P, dz))
        for a in range(0, len(ids), 32):
            t = np.asarray(token_fn(ids[a:a + 32]), np.float64)[:, :NT]
            s += t.sum((0, 1))
        out[i] = s / (len(ids) * NT)
        cnt.append(int(len(ids)))
    return out, cnt


def level_mean_shifts(M):
    """D[l, p] = overall_mean[p] - M[l, p]; overall mean = equal-weighted mean of the level means (CHANGES.md item 3). M (n_levels, P, dz)."""
    return M.mean(0, keepdims=True) - M


def random_dirs(n, d, seed):
    R = np.random.RandomState(int(seed)).randn(n, d)
    return R / np.linalg.norm(R, axis=1, keepdims=True)


def random_signs(n, n_levels, P, seed):
    """+-1 per (direction, level, patch) for the level-mean-match random control (keeps the control displacements zero-mean over levels, CHANGES.md item 5)."""
    return np.random.RandomState(int(seed) + 1).choice([-1.0, 1.0], size=(n, n_levels, P))


def leace_disp_np(E, Z, rdir=None):
    """Displacement (n, dz) that LEACE adds to tokens Z (n, dz): -coef * c with coef = a.(z-mu)/(a.c). rdir: unit vector -> control -coef*|c|*rdir (same per-token norm)."""
    coef = (Z - E["mu"]) @ E["a"] / float(E["a"] @ E["c"])
    v = E["c"] if rdir is None else np.linalg.norm(E["c"]) * np.asarray(rdir)
    return -coef[:, None] * v[None]


def lmm_disp_np(D, lvl, rdir=None, signs=None):
    """Displacement for tokens at patch p of level lvl: D[lvl, p]. Control: sign[lvl, p] * |D[lvl, p]| * rdir. D (L,P,dz); lvl int; returns (P, dz)."""
    d = D[lvl]
    if rdir is None:
        return d
    return (np.linalg.norm(d, axis=-1) * signs[lvl])[:, None] * np.asarray(rdir)[None]


# ------------------------------------------------------------------------------------------------ bootstrap and Delta_use
def boot_indices(group_ids, n_boot, seed):
    """Episode bootstrap stratified by group (level): each resample draws, within every group, as many episodes (with replacement) as the group has. (n_boot, n) int."""
    g = np.asarray(group_ids)
    rng = np.random.RandomState(int(seed))
    out = np.empty((n_boot, len(g)), int)
    for u in sorted(set(g.tolist())):
        ids = np.where(g == u)[0]
        out[:, ids] = ids[rng.randint(0, len(ids), size=(n_boot, len(ids)))]
    return out


def norm_motion(motion_sum, persist_sum):
    """persistence-normalised motion error = sum of model motion errors / sum of persistence (Delta z = 0) errors (ratio of means over episodes)."""
    return motion_sum / persist_sum


def use_stats(sel, idx_boot, ci):
    """sel: dict(none (n,), size (n,), rand (K,n), persist (n,)) per-episode motion errors (mean over 12 transitions) of one eval set.
    Point values on the full set; CI of every reported quantity from the bootstrap index matrix (B, n)."""
    def comp(ix):
        den = sel["persist"][ix].sum(-1)
        mn = sel["none"][ix].sum(-1) / den
        ms = sel["size"][ix].sum(-1) / den
        mr = np.stack([sel["rand"][k][ix].sum(-1) for k in range(sel["rand"].shape[0])], 0) / den
        inc_s, inc_r = ms - mn, (mr - mn).mean(0)
        return {"motion_norm_none": mn, "motion_norm_size": ms, "motion_norm_rand_mean": mr.mean(0), "increase_size": inc_s, "increase_rand_mean": inc_r,
                "delta_use": inc_s - inc_r, "gap": 1.0 - mn}
    n = len(sel["persist"])
    pt = comp(np.arange(n)[None, :])
    bt = comp(idx_boot)
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out = {}
    for k in pt:
        out[k] = {"point": float(pt[k][0]), "ci_lo": float(np.percentile(bt[k], lo)), "ci_hi": float(np.percentile(bt[k], hi))}
    out["delta_use_over_gap"] = float(pt["delta_use"][0] / pt["gap"][0]) if pt["gap"][0] != 0 else None
    out["n_episodes"] = int(n)
    out["rand_increase_by_direction"] = [float((sel["rand"][k].sum() - sel["none"].sum()) / sel["persist"].sum()) for k in range(sel["rand"].shape[0])]
    return out


MET, NOT_MET, NOT_EVAL = "criterion_met", "criterion_not_met", "not_evaluable"


def judge_method(stats_b, stats_a, succ, seeds):
    """Confirmed rule (hypotheses.md 013, 2026-11-11) for ONE erasure method, on the train-level eval set.
    stats_b / stats_a: {train_seed: use_stats dict or None}. Conditions:
      1. Delta_use > 0 with the 90% CI excluding 0 (ci_lo > 0) in >= min_seeds of the (b) training seeds;
      2. Delta_use >= 10% of (persistence error - model error), on the seed means of (b) (gap = 1 - normalised model error of the un-erased model; must be > 0);
      3. seed-mean Delta_use of (b) > seed-mean Delta_use of (a).
    Any missing / non-finite input -> not_evaluable (no value substituted)."""
    def ok(s):
        return isinstance(s, dict) and all(np.isfinite(s[k]["point"]) and np.isfinite(s[k]["ci_lo"]) for k in ("delta_use", "gap"))
    miss_b = [ts for ts in seeds if not ok(stats_b.get(ts))]
    miss_a = [ts for ts in seeds if not ok(stats_a.get(ts))]
    if miss_b or (succ.get("require_b_greater_than_a", True) and miss_a):
        return {"status": NOT_EVAL, "missing_b_seeds": miss_b, "missing_a_seeds": miss_a}
    per = {ts: {"delta_use": stats_b[ts]["delta_use"]["point"], "ci_lo": stats_b[ts]["delta_use"]["ci_lo"], "ci_hi": stats_b[ts]["delta_use"]["ci_hi"], "gap": stats_b[ts]["gap"]["point"],
                "positive_and_ci_excludes_0": bool(stats_b[ts]["delta_use"]["point"] > 0 and stats_b[ts]["delta_use"]["ci_lo"] > 0),
                "delta_use_ge_10pct_gap_this_seed": bool(stats_b[ts]["gap"]["point"] > 0 and stats_b[ts]["delta_use"]["point"] >= succ["min_fraction_of_persistence_gap"] * stats_b[ts]["gap"]["point"])}
           for ts in seeds}
    n_ok = sum(p["positive_and_ci_excludes_0"] for p in per.values())
    mb = float(np.mean([p["delta_use"] for p in per.values()]))
    gb = float(np.mean([p["gap"] for p in per.values()]))
    c1 = n_ok >= int(succ["min_seeds"])
    c2 = bool(gb > 0 and mb >= float(succ["min_fraction_of_persistence_gap"]) * gb)
    ma = float(np.mean([stats_a[ts]["delta_use"]["point"] for ts in seeds])) if not miss_a else None
    c3 = bool(mb > ma) if (ma is not None) else None
    if not succ.get("require_b_greater_than_a", True):
        c3 = True
    met = c1 and c2 and bool(c3)
    return {"status": MET if met else NOT_MET,
            "reading": ("이 소거에서 크기 정보 사용 검출" if met else "이 소거에서 검출되지 않음 (크기 정보를 사용하지 않는다는 뜻이 아님: 소거 불완전 가능성, 소거 확인 수치 참고)"),
            "cond1_seeds_positive_ci_excl_0": {"n": int(n_ok), "needed": int(succ["min_seeds"]), "met": bool(c1)},
            "cond2_mean_delta_use_ge_10pct_mean_gap": {"mean_delta_use_b": mb, "mean_gap_b": gb, "threshold": float(succ["min_fraction_of_persistence_gap"]) * gb, "met": c2},
            "cond3_b_greater_than_a": {"mean_delta_use_b": mb, "mean_delta_use_a": ma, "met": c3},
            "per_seed_b": {str(k): v for k, v in per.items()}}


# ------------------------------------------------------------------------------------------------ stage 2
def level_mean_comparison(dhat, dreal, lvl, n_levels):
    """dhat, dreal (n_ep, dz) per-episode mean predicted / real Delta z; lvl (n_ep,) level index. Level means, global-mean removal over the levels, then
    Pearson correlation and OLS slope (pred on real) over the (levels x dz) entries."""
    mh = np.stack([dhat[lvl == i].mean(0) for i in range(n_levels)])
    mr = np.stack([dreal[lvl == i].mean(0) for i in range(n_levels)])
    mh, mr = mh - mh.mean(0, keepdims=True), mr - mr.mean(0, keepdims=True)
    a, b = mh.ravel(), mr.ravel()
    sb = float((b * b).sum())
    corr = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())) if sb > 0 and (a * a).sum() > 0 else None
    slope = float((a * b).sum() / sb) if sb > 0 else None
    return {"corr": corr, "slope_pred_on_real": slope, "n_levels": int(n_levels), "n_entries": int(a.size), "norm_real_residual": float(np.sqrt(sb)), "norm_pred_residual": float(np.sqrt((a * a).sum()))}
