"""Verdicts for H-e / H-f / H-g (spec 4.7). supported / refuted / not_evaluable. Every threshold is read from config (null -> not_evaluable).
No interpretive text: only the numbers that enter each condition."""
from __future__ import annotations

import numpy as np


def _vals(rows, **flt):
    """seed -> value (score) for rows matching all filters, sorted by seed."""
    d = {}
    for r in rows:
        if all(r.get(k) == v for k, v in flt.items()):
            d[r["seed"]] = r["score"] if "score" in r else r["value"]
    return np.array([d[k] for k in sorted(d)], float)


def _ms(v):
    v = v[np.isfinite(v)]
    return (float(v.mean()) if len(v) else float("nan")), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)


def _ne(reason, **kw):
    out = {"status": "not_evaluable", "reason": reason}
    out.update(kw)
    return out


def h_e(ib_rows, cal_rows, layer, vc):
    """ib_rows: dicts layer/model/metric/pair_group/seed/score ; cal_rows: dicts layer/model/group/seed/value (mean abs deviation, normalized)."""
    dev_max, mult, sat = vc["h_e"]["dev_max"], vc["sd_mult"], vc["baseline_saturated_score"]
    if dev_max is None or mult is None or sat is None:
        return _ne("threshold_null")
    group = vc["h_e"]["pair_group"]
    out = {"conditions": {}}
    for metric in ("R-IB", "D-IB"):
        o = _vals(ib_rows, layer=layer, model="ours", metric=metric, pair_group=group)
        a = _vals(ib_rows, layer=layer, model="area", metric=metric, pair_group=group)
        r = _vals(ib_rows, layer=layer, model="random", metric=metric, pair_group=group)
        if len(o) == 0 or len(a) == 0 or len(r) == 0:
            return _ne("missing_rows_for_5A", metric=metric)
        (mo, so, no), (ma, sa, na), (mr, sr, nr) = _ms(o), _ms(a), _ms(r)
        if not np.isfinite([mo, ma, mr]).all():
            return _ne("score_nan_oracle_not_separated_from_noninformative", metric=metric)
        if min(no, nr) < 2 or not np.isfinite([so, sr]).all():
            return _ne("single_seed_sd_undefined", metric=metric)
        if ma >= sat:
            return _ne("area_baseline_saturated", metric=metric, area_score=ma)
        out["conditions"][metric] = {"ours": mo, "area": ma, "random": mr, "ours_gt_area": bool(mo > ma), "ours_sd": so, "random_sd": sr,
                                     "abs_diff_ours_random": abs(mo - mr), "outside_seed_sd_of_random": bool(abs(mo - mr) > mult * max(so, sr))}
    cal = _vals(cal_rows, layer=layer, model="ours", group=vc["h_e"]["calibration_group"])
    if len(cal) == 0:
        return _ne("missing_rows_for_5B", conditions=out["conditions"])
    mc, sc, nc = _ms(cal)
    if not np.isfinite(mc):
        return _ne("calibration_nan", conditions=out["conditions"])
    out["conditions"]["calibration"] = {"mean_abs_dev_norm": mc, "sd": sc, "max_allowed": dev_max, "within": bool(mc <= dev_max)}
    ok = all(out["conditions"][m]["ours_gt_area"] and out["conditions"][m]["outside_seed_sd_of_random"] for m in ("R-IB", "D-IB")) and out["conditions"]["calibration"]["within"]
    out["status"] = "supported" if ok else "refuted"
    out["pair_group"] = group
    return out


def h_f(ib_rows, layer, vc):
    mult = vc["sd_mult"]
    if mult is None:
        return _ne("threshold_null")
    g = vc["h_f"]["pair_group"]
    a = _vals(ib_rows, layer=layer, model="ours", metric="D-IB_area=", pair_group=g)
    b = _vals(ib_rows, layer=layer, model="ours", metric="D-IB_area≠", pair_group=g)
    if len(a) == 0 or len(b) == 0:
        return _ne("missing_rows_for_5C")
    (ma, sa, na), (mb, sb, nb) = _ms(a), _ms(b)
    if not np.isfinite([ma, mb]).all():
        return _ne("score_nan_no_pairs_or_oracle_not_separated")
    if min(na, nb) < 2 or not np.isfinite([sa, sb]).all():
        return _ne("single_seed_sd_undefined")
    same = abs(ma - mb) <= mult * max(sa, sb)
    return {"status": "supported" if same else "refuted", "D-IB_area_eq": ma, "D-IB_area_neq": mb, "sd_eq": sa, "sd_neq": sb, "abs_diff": abs(ma - mb), "pair_group": g}


def h_g(law_rows, layer, vc):
    lo, hi = vc["h_g"]["slope_lo"], vc["h_g"]["slope_hi"]
    if lo is None or hi is None:
        return _ne("threshold_null")
    slices = sorted({r["slice"] for r in law_rows if r["layer"] == layer and r["model"] == "ours"})
    if not slices:
        return _ne("missing_rows_for_5D")
    per = {}
    for sl in slices:
        v = np.array([r["slope"] for r in law_rows if r["layer"] == layer and r["model"] == "ours" and r["slice"] == sl], float)
        m, s, n = _ms(v)
        per[sl] = {"slope_mean": m, "slope_sd": s, "n_seeds": n, "within": bool(np.isfinite(m) and lo <= m <= hi)}
    if any(not np.isfinite(p["slope_mean"]) for p in per.values()):
        return _ne("slope_nan_in_some_slice", per_slice=per)
    return {"status": "supported" if all(p["within"] for p in per.values()) else "refuted", "per_slice": per, "range": [lo, hi]}
