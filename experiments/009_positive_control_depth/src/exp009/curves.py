"""Numpy-only: curves per (block m_int, strength, train seed), detection with 004's rule (exp004.depthlib.first_drop), per-block success verdict.
Rows are long-format dicts as written by exp004 (module, depth, probe, fit, level, split, timepoint, seed, r2, control, train_seed, ...) plus m_int, alpha."""
from __future__ import annotations

import numpy as np

from exp004 import depthlib as DL


def seed_curve(rows, train_seed, split, primary, depths_from=None):
    """{depth: mean R2 over probe seeds} of the rows given (already filtered to one unit); primary = dict(probe, fit, control, timepoint)."""
    acc = {}
    for r in rows:
        if (str(r["train_seed"]) == str(train_seed) and r["module"] == "predictor" and r["probe"] == primary["probe"] and r["fit"] == primary["fit"] and
                r["control"] == primary["control"] and r["timepoint"] == primary["timepoint"] and r["level"] == "ALL" and r["split"] == split):
            acc.setdefault(int(r["depth"]), []).append(float(r["r2"]))
    return {d: float(np.nanmean(v)) for d, v in acc.items()}


def unit_curve(base_rows, unit_rows, train_seed, split, primary, m_int):
    """Depth < m_int from the shared baseline, depth >= m_int from the intervention unit."""
    b = seed_curve(base_rows, train_seed, split, primary)
    u = seed_curve(unit_rows, train_seed, split, primary)
    cur = {d: v for d, v in b.items() if d < m_int}
    cur.update({d: v for d, v in u.items() if d >= m_int})
    return cur


def detect(curve, thr):
    ds = sorted(curve)
    return DL.first_drop(ds, [curve[d] for d in ds], thr)


def m_star_of(curves_by_split, thr):
    """earliest vanishing depth over splits (004's m_star definition) for one train seed. Returns (m_star or None, per-split dict)."""
    per = {s: (detect(c, thr) or {}).get("depth") for s, c in curves_by_split.items()}
    vals = [v for v in per.values() if v is not None]
    return (min(vals) if vals else None), per


def block_verdict(m_stars, m_int, slack, min_seeds):
    """m_stars: {train_seed: m* or None} at the success strength. criterion_met = at least `min_seeds` seeds with m* <= m_int + slack."""
    if not m_stars:
        return {"status": "not_evaluable", "reason": "no_units"}
    det = {str(ts): (m is not None and m <= m_int + slack) for ts, m in m_stars.items()}
    n = sum(det.values())
    return {"status": "criterion_met" if n >= min_seeds else "criterion_not_met", "n_seeds_detected": int(n), "n_seeds": len(m_stars), "min_seeds": min_seeds,
            "m_star_by_train_seed": {str(k): v for k, v in m_stars.items()}, "detected_by_train_seed": det, "rule": "m* <= m_int + %d" % slack}
