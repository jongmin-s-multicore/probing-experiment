"""Auxiliary decision rules A and B for 009 (user decision 2026-11-10: apply 008's A and B definitions as AUXILIARY verdicts; the primary success rule is unchanged).
Definitions are copied from experiments/008_posthoc_reaggregate_004_005/run.py (008 files are not modified, run.py is a script and cannot be imported):
  A: delta(d) = R2_intervened_predictor(d) - R2_random_init_predictor(d), 0.2 drop rule on the delta curve (008 `first_drop` = exp004.depthlib.first_drop).
  B: common-variance R2 = 1 - MSE_split / Var_all (Var_all = variance of the label over ALL evaluation episodes, label = log(area_ratio) of 002 config, weighted by n_episodes),
     with the same 0.2 drop rule; MAE (ALL rows) and MSE (log area ratio units) are recorded. Level rows already store r2_level = 1 - MSE_level/Var_all (exp004.depthlib.cell_rows),
     so MSE_split = n-weighted mean of (1 - r2_level) * Var_all. Self-check as in 008: 1 - MSE_split/Var_split must equal the CSV ALL-row split R2 (< 1e-6) else B is 'calculation impossible'.
Status strings: 관측됨 / 관측 안 됨 / 계산 불가. numpy only."""
from __future__ import annotations

import math

import numpy as np

from exp004 import depthlib as DL

OBS, NOT_OBS, NA = "관측됨", "관측 안 됨", "계산 불가"


def _sel(r, ts, primary, control):
    return (str(r["train_seed"]) == str(ts) and r["module"] == "predictor" and r["probe"] == primary["probe"] and r["fit"] == primary["fit"] and
            r["control"] == control and r["timepoint"] == primary["timepoint"])


def combined_rows(base_rows, unit_rows, ts, primary, m_int):
    """Primary-config rows of one (block, strength, train seed): depth < m_int from the shared baseline, depth >= m_int from the unit (m_int 0 = baseline only)."""
    out = [r for r in base_rows if _sel(r, ts, primary, "real") and int(r["depth"]) < m_int]
    out += [r for r in unit_rows if _sel(r, ts, primary, "real") and int(r["depth"]) >= m_int]
    return out


def _mstar(per_split_drop):
    ds = [v["depth"] for v in per_split_drop.values() if v is not None]
    return min(ds) if ds else None


def _verdict(per_split_drop, m_int, slack):
    ms = _mstar(per_split_drop)
    return {"status": OBS if ms is not None else NOT_OBS, "m_star": ms, "m_star_by_split": {s: (v["depth"] if v else None) for s, v in per_split_drop.items()},
            "within_m_int_plus_slack": (None if ms is None else bool(ms <= m_int + slack)), "rule": "0.2 drop from running max; within = m* <= m_int + %d" % slack}


def random_init_reference(rows004, ts, primary, base_rows):
    """random-init predictor rows of 004 for train seed ts; keys (depth, split, probe seed) must equal those of the 009 baseline (real) rows. Returns (rows, info)."""
    rnd = [r for r in (rows004 or []) if _sel(r, ts, primary, "random_init") and r["level"] == "ALL"]
    bas = [r for r in base_rows if _sel(r, ts, primary, "real") and r["level"] == "ALL"]
    kr = {(int(r["depth"]), r["split"], str(r["seed"])) for r in rnd}
    kb = {(int(r["depth"]), r["split"], str(r["seed"])) for r in bas}
    info = {"n_random_init_rows": len(rnd), "n_baseline_rows": len(bas), "keys_match": bool(kr) and kr == kb}
    return rnd, info


def rule_A(cur_real, rnd_rows, info, splits, thr, m_int, slack, ts):
    """cur_real: {split: {depth: mean R2}} of the unit (as in the main verdict). rnd_rows: random-init rows (ALL level). Returns result dict."""
    if not info["keys_match"]:
        return {"status": NA, "reason": "random-init reference rows missing or keys (depth, split, probe seed) differ from the 009 baseline", "reference": info}
    delta, per_split_drop = {}, {}
    for s in splits:
        acc = {}
        for r in rnd_rows:
            if r["split"] == s:
                acc.setdefault(int(r["depth"]), []).append(float(r["r2"]))
        rc = {d: float(np.nanmean(v)) for d, v in acc.items()}
        common = sorted(set(rc) & set(cur_real[s]))
        if sorted(cur_real[s]) != common:
            return {"status": NA, "reason": "depths differ between unit curve and random-init reference (split %s)" % s, "reference": info}
        delta[s] = {d: cur_real[s][d] - rc[d] for d in common}
        per_split_drop[s] = DL.first_drop(common, [delta[s][d] for d in common], thr)
    out = _verdict(per_split_drop, m_int, slack)
    out["delta_curves"] = {s: {str(d): v for d, v in c.items()} for s, c in delta.items()}
    out["reference"] = info
    return out


def common_variance(cfg2):
    """label = log(area_ratio) (002 config data_impl.area_ratio), as in 008. Returns {level: label}."""
    return {k: math.log(v) for k, v in cfg2["data_impl"]["area_ratio"].items()}


def rule_B(rows, cfg2, splits, thr, m_int, slack):
    """rows: primary-config rows of the unit (all levels, all depths, all probe seeds). Common-variance R2 (Var_all of all evaluation episodes) with the 0.2 rule + MAE/MSE."""
    lab = common_variance(cfg2)
    lv = {}
    for r in rows:
        if r["level"] != "ALL":
            lv[r["level"]] = (r["split"], int(r["n_episodes"]))
    if not lv or any(l not in lab for l in lv):
        return {"status": NA, "reason": "no level rows or label missing for a level"}
    Nall = sum(n for _, n in lv.values())
    mu = sum(n * lab[l] for l, (_, n) in lv.items()) / Nall
    var_all = sum(n * (lab[l] - mu) ** 2 for l, (_, n) in lv.items()) / Nall

    def var_split(s):
        ls = [l for l, (sp, _) in lv.items() if sp == s]
        n = sum(lv[l][1] for l in ls)
        m = sum(lv[l][1] * lab[l] for l in ls) / n
        return sum(lv[l][1] * (lab[l] - m) ** 2 for l in ls) / n

    by = {}   # (depth, probe seed) -> {(level, split): row}
    for r in rows:
        by.setdefault((int(r["depth"]), str(r["seed"])), {})[(r["level"], r["split"])] = r
    acc = {s: {} for s in splits}
    chk = []
    for (d, ps), dct in by.items():
        for s in splits:
            ls = [l for l, (sp, _) in lv.items() if sp == s]
            if not ls or any((l, s) not in dct for l in ls) or ("ALL", s) not in dct:
                return {"status": NA, "reason": "level or ALL rows missing at depth %d probe seed %s split %s" % (d, ps, s)}
            n = sum(lv[l][1] for l in ls)
            mse = sum(lv[l][1] * (1 - dct[(l, s)]["r2"]) * var_all for l in ls) / n
            vs = var_split(s)
            chk.append(abs((1 - mse / vs) - dct[("ALL", s)]["r2"]) if vs > 1e-12 else float("nan"))
            a = acc[s].setdefault(d, {"r2_common": [], "mse": [], "mae": []})
            a["r2_common"].append(1 - mse / var_all)
            a["mse"].append(mse)
            a["mae"].append(dct[("ALL", s)]["mae"])
    mx = float(np.nanmax(chk)) if chk else float("nan")
    if not (mx < 1e-6):
        return {"status": NA, "reason": "self-check failed: reconstructed split R2 vs CSV ALL rows max diff %s (>= 1e-6)" % mx}
    curves = {s: {d: {k: float(np.mean(v)) for k, v in a.items()} for d, a in sorted(acc[s].items())} for s in splits}
    per_split_drop = {s: DL.first_drop(sorted(curves[s]), [curves[s][d]["r2_common"] for d in sorted(curves[s])], thr) for s in splits}
    out = _verdict(per_split_drop, m_int, slack)
    out.update({"var_all": var_all, "selfcheck_max_abs_diff": mx, "curves": {s: {str(d): v for d, v in c.items()} for s, c in curves.items()},
                "note": "r2_common = 1 - MSE_split/Var_all; MSE and MAE in log area ratio units; values are means over probe seeds"})
    return out
