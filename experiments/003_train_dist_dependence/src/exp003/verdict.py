"""Verdict table (pass / fail / not_evaluable) for H-H1 and H-H2. Every threshold is read from config.yaml criteria (recorded in
'criteria_used'); nothing is hard-coded. The status words only apply the pre-registered rule; no interpretation sentence is produced.

status mapping
  H-H1  pass = "학습 크기 수렴" pattern supported;  fail = "가운데 압축" pattern supported;  not_evaluable = everything else
  H-H2  per mix distribution: pass = every (mix seed, baseline seed) pair has slope difference >= h2_slope_gain_min;
        fail = every pair has |difference| < h2_slope_gain_min;  not_evaluable = anything else (mixed / opposite direction / missing data).
        overall: pass if all distributions pass, fail if all fail, else not_evaluable.
        (rule 'all_pairs' and 'all_untrained' are implementer readings of the pre-registered sentences; see README.)"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import area_ratio_of, cget


class Prereg:
    def __init__(self, cfg):
        self.cfg, self.used = cfg, {}

    def __call__(self, dotted):
        v = cget(self.cfg, dotted)
        self.used[dotted] = v
        return v


def dig(d, path):
    for p in path:
        d = d[p]
    return d


def h1_seed(cfg, m, train_group, rule):
    """m: metrics dict of one run. Returns the per-seed record."""
    a_tr = area_ratio_of(cfg, train_group)
    c = m["center"]
    rec = {"c": c["c"], "center_status": c["status"], "center_null_reason": c["reason"], "n_crossings": c["n_crossings"], "crossings": c["crossings"]}
    sizes = [g for g in m["groups_in_order"] if g != train_group]
    flags = {}
    for g in sizes:
        r = m["by_size"][g]["ratio_ol_over_real"]
        toward_smaller = area_ratio_of(cfg, train_group) < area_ratio_of(cfg, g)       # training size is smaller than g -> pull = ratio < 1
        excl = (r["hi"] is not None and r["hi"] < 1.0) if toward_smaller else (r["lo"] is not None and r["lo"] > 1.0)
        flags[g] = {"ci": [r["lo"], r["hi"]], "ci_excludes_1_toward_train_size": bool(excl)}
    rec["opposite_size_ci_flags"] = flags
    if rule != "all_untrained":
        raise NotImplementedError(rule)
    rec["ci_condition_met"] = bool(all(f["ci_excludes_1_toward_train_size"] for f in flags.values())) if flags else None
    rec["n_sizes_ci_excluding_1_toward_train"] = int(sum(f["ci_excludes_1_toward_train_size"] for f in flags.values()))
    if c["c"] is None:
        rec.update(status="not_evaluable", reason="c is null: " + str(c["reason"]), c_closer_to_train_size=None, c_closer_to_ind=None, supports_h1=None, supports_center=None)
        return rec
    lc, ltr = c["log_c"], float(np.log(a_tr))
    d_tr, d_ind = abs(lc - ltr), abs(lc - 0.0)          # ind has nominal area ratio 1.0 -> log 0
    rec.update(dist_log_c_to_train_size=d_tr, dist_log_c_to_ind=d_ind, c_closer_to_train_size=bool(d_tr < d_ind), c_closer_to_ind=bool(d_ind < d_tr))
    rec["supports_h1"] = bool(rec["c_closer_to_train_size"] and rec["ci_condition_met"])
    rec["supports_center"] = bool(rec["c_closer_to_ind"])
    rec["status"] = "evaluated"
    rec["reason"] = None
    return rec


def h1(cfg, runs, P):
    models = P("criteria.h1_single_models")
    rule = P("criteria.h1_opposite_sizes_rule")
    nreq = P("criteria.seeds_required_same_direction")
    out = {"rule_text": "pass: both single models, all required seeds: c closer (log area) to the training size than to ind AND CI of every untrained size excludes 1 toward the training size. "
                        "fail ('가운데 압축'): both models, all required seeds: c closer to ind than to the training size. else not_evaluable.",
           "models": {}}
    pat = {}
    for dist, tg in models.items():
        per = {}
        for s in cfg["train"]["seeds"]:
            r = runs.get("%s:%d" % (dist, s))
            per[str(s)] = h1_seed(cfg, r, tg, rule) if r else {"status": "not_evaluable", "reason": "run %s:%d has no metrics.json" % (dist, s)}
        ev = [p for p in per.values() if p["status"] == "evaluated"]
        if len(ev) < nreq or len(per) < nreq:
            p_ = "not_evaluable"
        elif all(p["supports_h1"] for p in ev):
            p_ = "h1_converge_to_train_size"
        elif all(p["supports_center"] for p in ev):
            p_ = "center_compression"
        else:
            p_ = "mixed"
        pat[dist] = p_
        out["models"][dist] = {"train_group": tg, "pattern": p_, "per_seed": per}
    pats = set(pat.values())
    out["status"] = "pass" if pats == {"h1_converge_to_train_size"} else ("fail" if pats == {"center_compression"} else "not_evaluable")
    out["patterns"] = pat
    return out


def baseline_slopes(cfg, base_dir, P):
    spec = cfg["criteria"]["h2_baseline"]
    P("criteria.h2_baseline")
    p = (Path(base_dir) / spec["results_json"]).resolve()
    if not p.exists():
        return None, {"file": str(p), "found": False}
    d = json.load(open(p))
    vals = {}
    for rn in spec["runs"]:
        try:
            vals[rn] = float(dig(d["runs"][rn], spec["slope_path"]))
        except (KeyError, TypeError):
            return None, {"file": str(p), "found": True, "missing_run_or_path": rn}
    return vals, {"file": str(p), "found": True, "slope_path": spec["slope_path"], "values": vals}


def h2(cfg, runs, base_dir, P):
    gain = float(P("criteria.h2_slope_gain_min"))
    nreq = P("criteria.seeds_required_same_direction")
    P("criteria.h2_rule")
    dists = P("criteria.h2_distributions")
    base, binfo = baseline_slopes(cfg, base_dir, P)
    out = {"slope_gain_min": gain, "baseline": binfo, "distributions": {},
           "rule_text": "per mix distribution: pass if all (mix seed, baseline seed) pairs have mix_slope - baseline_slope >= gain; fail if all pairs |diff| < gain; else not_evaluable. No mean comparison."}
    for d in dists:
        sl = {}
        for s in cfg["train"]["seeds"]:
            r = runs.get("%s:%d" % (d, s))
            if r:
                sl[str(s)] = r["slope_ol"]["all_sizes"]["slope"]
        rec = {"slopes_by_seed": sl}
        if base is None:
            rec.update(status="not_evaluable", reason="baseline_single results not readable (%s)" % binfo)
        elif len(sl) < nreq or any(v is None for v in sl.values()):
            rec.update(status="not_evaluable", reason="fewer than %d seeds with a valid slope" % nreq)
        else:
            diffs = {"%s_vs_%s" % (s, rn): v - bv for s, v in sl.items() for rn, bv in base.items()}
            rec["pair_differences_mix_minus_baseline"] = diffs
            dv = np.array(list(diffs.values()))
            if (dv >= gain).all():
                rec.update(status="pass", reason=None, pattern="all_pairs_ge_gain")
            elif (np.abs(dv) < gain).all():
                rec.update(status="fail", reason=None, pattern="all_pairs_within_gain")
            else:
                rec.update(status="not_evaluable", reason=None, pattern="mixed_or_opposite", n_pairs_ge_gain=int((dv >= gain).sum()), n_pairs_le_minus_gain=int((dv <= -gain).sum()), n_pairs=int(len(dv)))
        out["distributions"][d] = rec
    st = {v["status"] for v in out["distributions"].values()}
    out["status"] = "pass" if st == {"pass"} else ("fail" if st == {"fail"} else "not_evaluable")
    return out


def build(cfg, runs, base_dir):
    """runs: {'dist:seed': metrics dict}."""
    P = Prereg(cfg)
    res = {"H-H1": h1(cfg, runs, P), "H-H2": h2(cfg, runs, base_dir, P)}
    res["criteria_used"] = P.used
    return res
