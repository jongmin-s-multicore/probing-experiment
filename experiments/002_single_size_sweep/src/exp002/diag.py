"""4c diagnostics (확정 기록 6 item 4c, rule fixed in 확정 기록 7 A). Numpy only.
Per-episode sufficient statistics of a layer's activations (n, sum_c, sumsq_c over frames x tokens) -> level mean/variance by adding episodes.
activation shift of a level vs ind, per layer j (normalized by ind):
   mean_shift = sqrt(mean_c ((mu_l - mu_ind) / sd_ind)^2)      var_shift = mean_c |var_l / var_ind - 1|      shift = mean_shift + var_shift   (sum: impl choice)
noise floor_j = median over B random splits of the ind episodes into two NON-overlapping halves of shift(half_1, half_2) (no a060/a160 data).
ratio_j = shift_j / floor_j ; class per layer: ratio < small_max -> small ; small_max <= ratio < medium_max -> medium ; ratio >= medium_max -> large.
Level activation class = class of max_j ratio_j. Prediction error: next-frame MSE of level / ind >= err_large_ratio -> large.
Final (over levels in q2_diag.levels): any 'large' -> large, else activation class (worst level). Rule text is stored next to the verdict."""
from __future__ import annotations

import numpy as np

ORDER = {"small": 0, "medium": 1, "large": 2}
RULE_TEXT = {"small": "small: layer-wise results of (a) are interpreted normally as auxiliary results",
             "medium": "medium: interpret, with a caution note about the influence of unfamiliar inputs",
             "large": "large: do NOT interpret the layer-wise results of (a); report numbers only"}


def episode_stats(F_ep):
    """F_ep: (nF, 196, d) activations of one episode -> (n, sum (d,), sumsq (d,)) float64."""
    x = np.asarray(F_ep, np.float64).reshape(-1, F_ep.shape[-1])
    return np.array([x.shape[0]], np.float64), x.sum(0), (x * x).sum(0)


def group(stats, idx):
    n = sum(stats[i][0] for i in idx)
    s = sum(stats[i][1] for i in idx)
    ss = sum(stats[i][2] for i in idx)
    mu = s / n
    var = np.maximum(ss / n - mu * mu, 0.0)
    return mu, var


def shift(g, ref, eps=1e-12):
    mu, var = g
    mu0, var0 = ref
    sd0 = np.sqrt(var0) + eps
    ms = float(np.sqrt(np.mean(((mu - mu0) / sd0) ** 2)))
    vs = float(np.mean(np.abs(var / (var0 + eps) - 1.0)))
    return {"mean_shift": ms, "var_shift": vs, "shift": ms + vs}


def noise_floor(stats_ind, n_boot, seed):
    """Median shift between two disjoint random halves of the ind episodes (B repeats)."""
    rng = np.random.RandomState(int(seed))
    n = len(stats_ind)
    h = n // 2
    vals = []
    for _ in range(int(n_boot)):
        p = rng.permutation(n)
        vals.append(shift(group(stats_ind, p[:h]), group(stats_ind, p[h:2 * h]))["shift"])
    return float(np.median(vals)), len(vals)


def classify_ratio(r, small_max, medium_max):
    if r is None or not np.isfinite(r):
        return "not_evaluable"
    return "small" if r < small_max else ("medium" if r < medium_max else "large")


def classify_err(ratio, large_min):
    if ratio is None or not np.isfinite(ratio):
        return "not_evaluable"
    return "large" if ratio >= large_min else "not_large"


def final_verdict(act_classes, err_classes):
    """act_classes/err_classes: lists over levels. any 'large' -> large; else worst activation class. Missing data -> not_evaluable."""
    allc = list(act_classes) + list(err_classes)
    if not allc or "not_evaluable" in allc:
        return "not_evaluable"
    if "large" in allc:
        return "large"
    return max(act_classes, key=lambda c: ORDER[c])


def evaluate(layer_stats, ind_idx, level_idx, next_err, n_boot, seed, dc):
    """layer_stats: {layer j: list of per-episode stats (index = test-episode order)}; ind_idx: indices of ind eps; level_idx: {level: indices};
    next_err: {level: per-episode next-frame MSE array incl. 'ind'}; dc: config diag_4c. Returns a json-able dict."""
    sm, mm, el = dc["activation_small_max_ratio"], dc["activation_medium_max_ratio"], dc["error_large_min_ratio"]
    layers = sorted(layer_stats)
    floors = {}
    for j in layers:
        f, nb = noise_floor([layer_stats[j][i] for i in ind_idx], n_boot, seed + j)
        floors[j] = f
    out = {"noise_floor_per_layer": floors, "n_bootstrap": int(n_boot), "bootstrap_seed": int(seed), "levels": {}, "rule": RULE_TEXT,
           "thresholds": {"activation_small_max_ratio": sm, "activation_medium_max_ratio": mm, "error_large_min_ratio": el}}
    ref = {j: group(layer_stats[j], ind_idx) for j in layers}
    ind_err = float(np.mean(next_err["ind"]))
    acts, errs = [], []
    for lv, idx in level_idx.items():
        per = {}
        for j in layers:
            s = shift(group(layer_stats[j], idx), ref[j])
            r = s["shift"] / floors[j] if floors[j] > 0 else None
            per[str(j)] = {**s, "ratio_to_noise_floor": r, "class": classify_ratio(r, sm, mm)}
        rmax = [v["ratio_to_noise_floor"] for v in per.values() if v["ratio_to_noise_floor"] is not None]
        ac = classify_ratio(max(rmax), sm, mm) if rmax else "not_evaluable"
        er = float(np.mean(next_err[lv])) / ind_err if ind_err > 0 else None
        ec = classify_err(er, el)
        out["levels"][lv] = {"per_layer": per, "max_ratio_over_layers": max(rmax) if rmax else None, "activation_class": ac,
                             "next_frame_error": float(np.mean(next_err[lv])), "ind_next_frame_error": ind_err, "next_frame_error_ratio_to_ind": er, "error_class": ec}
        acts.append(ac)
        errs.append(ec)
    v = final_verdict(acts, errs)
    out["verdict"] = v
    out["interpretation_rule_for_verdict"] = RULE_TEXT.get(v)
    return out
