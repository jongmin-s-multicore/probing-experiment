"""Numpy-free verdict rules of experiment 012 (config success block, user-confirmed 2026-11-10).
Per block, at strength 1.0 and depth m_int, each item holds when at least `min_seeds` of the training seeds satisfy the threshold (>=):
  (a) 3-class accuracy >= a_accuracy_min ; (b) MLP R2 or attentive R2 >= b_r2_min ; (c) action-coordinate-only R2 >= c_r2_min.
A unit that is missing or undefined (None / NaN) makes the item not_evaluable; it is never replaced by an estimate. If none holds and all three are
evaluable and not met, the block is recorded as UNDETECTED (it does not mean that there is no information). The interpretation field stays null."""
from __future__ import annotations

import math

MET, NOT_MET, NOT_EVAL = "criterion_met", "criterion_not_met", "not_evaluable"
UNDETECTED = "이 측정들로는 잔여 정보 미검출"
ITEMS = {"a": "a_class_structure", "b": "b_nonlinear", "c": "c_action_path"}


def _fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def judge_item(values_by_ts, thr, min_seeds, expected_ts, any_of=False):
    """values_by_ts: {train_seed: float|None} or, with any_of, {train_seed: {name: float|None}} (met when ANY named value >= thr; every named value must be finite)."""
    expected_ts = list(expected_ts)
    miss = []
    for ts in expected_ts:
        v = values_by_ts.get(ts)
        ok = (v is not None and len(v) > 0 and all(_fin(x) for x in v.values())) if any_of else _fin(v)
        if not ok:
            miss.append(ts)
    base = {"threshold": thr, "min_seeds": min_seeds, "n_seeds": len(expected_ts), "values_by_train_seed": {str(ts): values_by_ts.get(ts) for ts in expected_ts}}
    if miss:
        base.update({"status": NOT_EVAL, "reason": "missing_or_undefined_unit_values", "missing_train_seeds": miss})
        return base
    met = {str(ts): bool(any(x >= thr for x in values_by_ts[ts].values()) if any_of else values_by_ts[ts] >= thr) for ts in expected_ts}
    n = sum(met.values())
    base.update({"status": MET if n >= min_seeds else NOT_MET, "n_seeds_met": int(n), "met_by_train_seed": met})
    return base


def block_verdict(a_by_ts, b_by_ts, c_by_ts, success, expected_ts):
    """a_by_ts {ts: acc}, b_by_ts {ts: {'mlp': r2, 'attentive': r2}}, c_by_ts {ts: r2}."""
    ms = int(success["min_seeds"])
    items = {ITEMS["a"]: judge_item(a_by_ts, float(success["a_accuracy_min"]), ms, expected_ts),
             ITEMS["b"]: judge_item(b_by_ts, float(success["b_r2_min"]), ms, expected_ts, any_of=True),
             ITEMS["c"]: judge_item(c_by_ts, float(success["c_r2_min"]), ms, expected_ts)}
    st = [v["status"] for v in items.values()]
    met = [k for k, v in items.items() if v["status"] == MET]
    if met:
        overall = {"status": MET, "items_met": met, "items_not_evaluable": [k for k, v in items.items() if v["status"] == NOT_EVAL]}
    elif all(s == NOT_MET for s in st):
        overall = {"status": UNDETECTED, "note": "does not mean that no information is present; other nonlinear forms may exist"}
    else:
        overall = {"status": NOT_EVAL, "reason": "no item met and at least one item not evaluable", "items_not_evaluable": [k for k, v in items.items() if v["status"] == NOT_EVAL]}
    return {"items": items, "overall": overall}
