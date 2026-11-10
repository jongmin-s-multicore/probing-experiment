"""Numpy-only statistics of experiment 010: per-representation score tables, task-seed-level units, TOST equivalence. No interpretation text."""
from __future__ import annotations

import itertools
import math

import numpy as np

# one-sided t critical values t_{1-alpha, df} for alpha = 0.05 (df 1..30); df > 30 uses the normal quantile 1.645 (slightly liberal)
T95 = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725,
       1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697]
T975 = None


def tcrit(df, alpha=0.05):
    if abs(alpha - 0.05) > 1e-12:
        raise ValueError("t table only for alpha=0.05; extend T95-like table for other alpha")
    return T95[df - 1] if 1 <= df <= 30 else 1.645


def tost(diffs, delta, alpha=0.05):
    """Equivalence of mean(diffs) to 0 within +-delta. diffs: one value per unit. Returns dict with mean, sd, se, df, ci90 (for alpha 0.05), t_low, t_high,
    status: supported (both one-sided tests reject: CI inside (-delta, delta)) | refuted (CI entirely outside [-delta, delta]) | inconclusive | not_evaluable."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if delta is None or n < 3:
        return {"status": "not_evaluable", "reason": "delta_null" if delta is None else "fewer_than_3_units", "n_units": int(n)}
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    tc = tcrit(df, alpha)
    if se <= 1e-15:
        lo = hi = m
        t_low = t_high = float("inf") if abs(m) < delta else float("-inf")
    else:
        lo, hi = m - tc * se, m + tc * se
        t_low, t_high = (m + delta) / se, (delta - m) / se
    if lo > -delta and hi < delta:
        st = "supported"
    elif lo > delta or hi < -delta:
        st = "refuted"
    else:
        st = "inconclusive"
    return {"status": st, "mean": m, "sd": sd, "se": se, "df": df, "n_units": int(n), "t_crit": tc, "ci90": [lo, hi], "t_low": t_low, "t_high": t_high, "delta": delta}


def unit_table(ib_rows, key_fields=("layer", "model", "metric", "pair_group")):
    """ib_rows: dicts with key fields + task_seed, seed (head), score, raw. Returns {key: {task_seed: [head-seed scores in head seed order]}} (duplicates are an error)."""
    out, seen = {}, set()
    for r in ib_rows:
        k = tuple(r[f] for f in key_fields)
        full = k + (r["task_seed"], r["seed"])
        if full in seen:
            raise ValueError("duplicate row %s" % (full,))
        seen.add(full)
        out.setdefault(k, {}).setdefault(r["task_seed"], {})[r["seed"]] = r["score"]
    return out


def unit_means(per_task):
    """{task_seed: {head: score}} -> (task_seeds sorted, per-task-seed mean over head seeds (nanmean), within-task sd over head seeds (mean over tasks))"""
    ts = sorted(per_task)
    mus = np.array([np.nanmean(list(per_task[t].values())) if len(per_task[t]) else np.nan for t in ts])
    w = [np.nanstd(list(per_task[t].values()), ddof=1) for t in ts if len(per_task[t]) > 1]
    return ts, mus, (float(np.mean(w)) if w else float("nan"))


def representation_table(ib_rows):
    T = unit_table(ib_rows)
    out = []
    for k, per in sorted(T.items()):
        ts, mus, w = unit_means(per)
        flat = np.array([v for t in ts for v in per[t].values()], float)
        out.append({"layer": k[0], "model": k[1], "metric": k[2], "pair_group": k[3], "n_task_seeds": len(ts), "mean": float(np.nanmean(mus)) if np.isfinite(mus).any() else float("nan"),
                    "sd_between_task_seeds": float(np.nanstd(mus, ddof=1)) if np.isfinite(mus).sum() > 1 else float("nan"), "sd_within_task_seed_over_head_seeds": w,
                    "sd_total_all_cells": float(np.nanstd(flat, ddof=1)) if np.isfinite(flat).sum() > 1 else float("nan"), "per_task_seed_mean": {str(t): float(m) for t, m in zip(ts, mus)}})
    return out


def h_f_tost(ib_rows, layers_models, group, delta, alpha):
    """Per (layer, model): d_t = score(D-IB_area=) - score(D-IB_area!=) per task seed (head-seed mean); TOST against +-delta."""
    T = unit_table(ib_rows)
    res = {}
    for (lay, mod) in layers_models:
        a, b = T.get((lay, mod, "D-IB_area=", group)), T.get((lay, mod, "D-IB_area≠", group))
        if not a or not b:
            res["%s/%s" % (lay, mod)] = {"status": "not_evaluable", "reason": "missing_rows_for_5C"}
            continue
        ta, ma, _ = unit_means(a)
        tb, mb, _ = unit_means(b)
        common = sorted(set(ta) & set(tb))
        d = np.array([ma[ta.index(t)] - mb[tb.index(t)] for t in common])
        r = tost(d, delta, alpha)
        r["per_task_seed_diff"] = {str(t): float(x) for t, x in zip(common, d)}
        res["%s/%s" % (lay, mod)] = r
    return res


def pairwise_cmp(ib_rows, reps, metrics, groups, delta, alpha):
    """All pairs among representations (layer/model) for each metric and pair group: difference of task-seed means (A - B) with sd over task seeds, and TOST."""
    T = unit_table(ib_rows)
    out = []
    for metric in metrics:
        for g in groups:
            for (l1, m1), (l2, m2) in itertools.combinations(reps, 2):
                a, b = T.get((l1, m1, metric, g)), T.get((l2, m2, metric, g))
                if not a or not b:
                    out.append({"a": "%s/%s" % (l1, m1), "b": "%s/%s" % (l2, m2), "metric": metric, "pair_group": g, "status": "not_evaluable", "reason": "missing_rows"})
                    continue
                ta, ma, _ = unit_means(a)
                tb, mb, _ = unit_means(b)
                common = sorted(set(ta) & set(tb))
                d = np.array([ma[ta.index(t)] - mb[tb.index(t)] for t in common])
                r = tost(d, delta, alpha)
                r.update({"a": "%s/%s" % (l1, m1), "b": "%s/%s" % (l2, m2), "metric": metric, "pair_group": g, "per_task_seed_diff": {str(t): float(x) for t, x in zip(common, d)}})
                out.append(r)
    return out
