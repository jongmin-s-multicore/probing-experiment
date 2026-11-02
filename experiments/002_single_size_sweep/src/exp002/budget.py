"""Equal training budget between conditions (확정 기록 6 item 3). Pure python (no torch), testable on CPU.
Definition: budget = total optimizer steps S = epochs x steps_per_epoch, steps_per_epoch = ceil(n_train_slices / batch_size).
The branch train.py is unmodified and only accepts `training.epochs`, so a condition's epochs are converted:
  S_ref = train.budget (if not null) else epochs_ref x steps_per_epoch[ref]       (ref = conditions.budget_reference; train.epochs = its epochs)
  epochs[c] = max(1, round(S_ref / steps_per_epoch[c]));  steps_computed[c] = epochs[c] x steps_per_epoch[c]
rel_diff = (steps_computed - S_ref) / S_ref is recorded for every condition (no threshold is applied: a nonzero value is a fact to report)."""
from __future__ import annotations

import math


def steps_per_epoch(n_slices, batch_size, rounding="ceil"):
    f = {"ceil": math.ceil, "floor": math.floor}[rounding]
    return max(1, int(f(n_slices / batch_size)))


def plan(spe, epochs_ref, ref, budget_steps=None):
    """spe: {condition: steps_per_epoch}. Returns {condition: {epochs, steps_computed, steps_target, rel_diff}} for every key of spe."""
    if ref not in spe:
        raise KeyError("reference condition %s has no steps_per_epoch" % ref)
    target = int(budget_steps) if budget_steps is not None else int(epochs_ref) * int(spe[ref])
    out = {}
    for c, k in spe.items():
        ep = int(epochs_ref) if (c == ref and budget_steps is None) else max(1, int(round(target / k)))
        out[c] = {"epochs": ep, "steps_per_epoch": int(k), "steps_computed": ep * int(k), "steps_target": target, "rel_diff": (ep * int(k) - target) / target}
    return out
