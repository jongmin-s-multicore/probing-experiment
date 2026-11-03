"""Numpy-only logic of experiment 004: long-format rows (depth_r2.csv), vanishing point, summary.json, H-d verdict.
No torch. Python 3.9 compatible. Every threshold comes from config (null -> not_evaluable)."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from .base import ConfigNull
from exp002 import stats as ST

ROLE2SPLIT = {"trained": "train", "interp": "interp", "extrap": "extrap"}
SPLITS = ["train", "interp", "extrap"]
# spec 3.5 columns first, then extensions (CHANGES.md)
CSV_FIELDS = ["module", "depth", "probe", "fit", "level", "split", "timepoint", "seed", "r2", "mae", "pearson",
              "control", "train_seed", "pred_mean", "n_episodes"]


# ------------------------------------------------------------------------------------------------ rows
def cell_rows(S, pm, te_rows, levels, base, seeds):
    """S (P, n_eval_eps, 7) per-episode sufficient statistics of ONE probe configuration, pm (P, n) mean prediction per episode (or None),
    te_rows: evaluation episode rows (role/level). Returns long-format rows: one per probe seed x (level | split aggregate).
    level == 'ALL' rows: R2/MAE/Pearson over the episodes of the split (split 'all' = every level).
    level == <name> rows: the label is constant within a level, so the within-level R2 is undefined; r2 = 1 - MSE_level / Var(y over all
    evaluation episodes) (pooled-variance R2), pearson empty. See CHANGES.md."""
    S = np.asarray(S, float)
    lv = np.array([r["level"] for r in te_rows])
    sp = np.array([ROLE2SPLIT[r["role"]] for r in te_rows])
    out = []
    for pi, seed in enumerate(seeds):
        tot_all = S[pi].sum(0)
        n_all = tot_all[0]
        var_all = (tot_all[3] - tot_all[1] ** 2 / n_all) / n_all if n_all > 0 else np.nan
        groups = [(l, sp[lv == l][0], lv == l) for l in levels if (lv == l).any()]
        groups += [("ALL", s, sp == s) for s in SPLITS if (sp == s).any()] + [("ALL", "all", np.ones(len(lv), bool))]
        for name, split, mk in groups:
            t = S[pi][mk].sum(0)
            n = t[0]
            if name == "ALL":
                r2 = float(ST.metric_from_sums(t, "r2"))
                pe = float(ST.metric_from_sums(t, "pearson"))
            else:
                sse = t[3] - 2 * t[5] + t[4]
                r2 = float(1 - (sse / n) / var_all) if (n > 0 and var_all > 1e-12) else float("nan")
                pe = float("nan")
            row = dict(base)
            row.update({"level": name, "split": split, "seed": seed, "r2": r2, "mae": float(ST.metric_from_sums(t, "mae")), "pearson": pe,
                        "pred_mean": (float(np.asarray(pm)[pi][mk].mean()) if pm is not None else float("nan")), "n_episodes": int(mk.sum())})
            out.append(row)
    return out


def write_rows(rows, path, fields=CSV_FIELDS):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(r.get(k), float) and not np.isfinite(r.get(k))) else r.get(k, "")) for k in fields})


def read_rows(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        rows = []
        for r in rd:
            for k in ("r2", "mae", "pearson", "pred_mean"):
                r[k] = float(r[k]) if r.get(k, "") != "" else float("nan")
            r["depth"] = int(r["depth"])
            rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------ curves and vanishing point
def curves(rows, module, probe, fit, control, timepoint, level="ALL"):
    """{split: {depth: np.array of values (probe seeds x train seeds)}} from rows of one configuration."""
    out = {s: defaultdict(list) for s in SPLITS}
    for r in rows:
        if (r["module"], r["probe"], r["fit"], r["control"], r["timepoint"], r["level"]) == (module, probe, fit, control, timepoint, level) and r["split"] in out:
            out[r["split"]][r["depth"]].append(r["r2"])
    return {s: {d: np.array(v, float) for d, v in sorted(dd.items())} for s, dd in out.items() if dd}


def first_drop(depths, values, thr):
    """First depth whose value is >= thr below the maximum of the earlier values. None if no such depth (or thr is None -> ConfigNull).
    NaN values are skipped. Returns dict(depth, value, running_max_before, drop) or None."""
    if thr is None:
        raise ConfigNull("drop_threshold is null")
    runmax = None
    for d, v in sorted(zip(depths, values), key=lambda t: t[0]):
        if not np.isfinite(v):
            continue
        if runmax is not None and (runmax - v) >= thr - 1e-12:
            return {"depth": int(d), "value": float(v), "running_max_before": float(runmax), "drop": float(runmax - v)}
        runmax = v if runmax is None else max(runmax, v)
    return None


def vanishing_by_split(cur, thr):
    res = {}
    for s in SPLITS:
        if s not in cur:
            res[s] = {"evaluated": False, "vanish": None}
            continue
        ds = sorted(cur[s])
        res[s] = {"evaluated": True, "n_depths": len(ds), "vanish": first_drop(ds, [float(np.nanmean(cur[s][d])) for d in ds], thr)}
    return res


def _mean_sd(cur, s, d):
    v = cur[s][d]
    v = v[np.isfinite(v)]
    return (float(v.mean()) if len(v) else float("nan")), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)


def ind_ood_gap(cur, mult, frac):
    """Per depth: |mean(train) - mean(ood split)| vs mult * max(sd_train, sd_ood). overlap = the gap is inside the tolerance at >= `frac` of the depths.
    n_values < 2 (a single seed) -> sd undefined -> evaluable False."""
    out = {}
    for s in ("interp", "extrap"):
        if s not in cur or "train" not in cur:
            continue
        per, ev = [], True
        for d in sorted(set(cur[s]) & set(cur["train"])):
            m0, s0, n0 = _mean_sd(cur, "train", d)
            m1, s1, n1 = _mean_sd(cur, s, d)
            if n0 < 2 or n1 < 2 or not np.isfinite(s0) or not np.isfinite(s1):
                ev = False
                continue
            tol = mult * max(s0, s1)
            inside = abs(m0 - m1) <= tol
            per.append({"depth": int(d), "gap": float(m0 - m1), "tol": float(tol), "inside_seed_sd": bool(inside)})
        fr = float(np.mean([p["inside_seed_sd"] for p in per])) if per else None
        out[s] = {"evaluable": ev and len(per) > 0, "fraction_depths_inside_seed_sd": fr, "overlap": (bool(fr >= frac - 1e-12) if (ev and per) else None), "per_depth": per}
    return out


def l_best(cur_by_module, splits):
    """argmax over (module, depth) of the mean of split-mean R2 over `splits` (default interp, extrap). Ties -> encoder, smaller depth."""
    best = None
    per_module = {}
    for mod in ("encoder", "predictor"):
        cur = cur_by_module.get(mod)
        if not cur or not all(s in cur for s in splits):
            continue
        ds = sorted(set.intersection(*[set(cur[s]) for s in splits]))
        if not ds:
            continue
        vals = [float(np.nanmean([np.nanmean(cur[s][d]) for s in splits])) for d in ds]
        i = int(np.nanargmax(vals))
        per_module[mod] = {"depth": int(ds[i]), "value": float(vals[i])}
        if best is None or vals[i] > best[2] + 1e-12:
            best = (mod, int(ds[i]), vals[i])
    return (None if best is None else {"module": best[0], "depth": best[1], "value": best[2]}), per_module


def verdict_h_d(vanish, gaps, thr, mult, min_depths, frac=0.8):
    """Spec 3 H-d. supported / refuted / not_evaluable. Thresholds come from config (null -> not_evaluable)."""
    if thr is None or mult is None or min_depths is None or frac is None:
        return {"status": "not_evaluable", "reason": "threshold_null"}
    mods = [m for m in vanish if all(vanish[m][s]["evaluated"] and vanish[m][s]["n_depths"] >= min_depths for s in SPLITS)]
    if not mods:
        return {"status": "not_evaluable", "reason": "fewer_than_min_depths_or_missing_splits"}
    per = {}
    for m in mods:
        v = vanish[m]
        vi = v["train"]["vanish"]["depth"] if v["train"]["vanish"] else None
        vo = [v[s]["vanish"]["depth"] for s in ("interp", "extrap") if v[s]["vanish"]]
        vo = min(vo) if vo else None
        any_drop = vi is not None or vo is not None
        earlier = (vo is not None) and (vi is None or vo < vi)
        ovl = [gaps[m][s]["overlap"] for s in ("interp", "extrap") if s in gaps.get(m, {})]
        ev = all(gaps[m][s]["evaluable"] for s in ("interp", "extrap") if s in gaps.get(m, {})) and bool(ovl)
        per[m] = {"any_drop_ge_threshold": any_drop, "ood_vanish_earlier_than_ind": bool(earlier), "ind_vanish_depth": vi, "ood_vanish_depth_min": vo,
                  "curves_overlap_within_seed_sd": (all(ovl) if ev else None), "gap_evaluable": ev}
    if not any(p["any_drop_ge_threshold"] for p in per.values()):
        return {"status": "refuted", "reason": "no_drop_ge_threshold_at_any_depth", "per_module": per}
    if any(p["gap_evaluable"] is False for p in per.values()):
        return {"status": "not_evaluable", "reason": "seed_sd_undefined_single_seed", "per_module": per}
    if all(p["curves_overlap_within_seed_sd"] for p in per.values()):
        return {"status": "refuted", "reason": "ind_ood_curves_overlap_within_seed_sd", "per_module": per}
    if any(p["ood_vanish_earlier_than_ind"] and not p["curves_overlap_within_seed_sd"] for p in per.values()):
        return {"status": "supported", "reason": "drop_ge_threshold_and_ood_vanish_earlier_and_curves_separate", "per_module": per}
    return {"status": "refuted", "reason": "ood_vanish_not_earlier_than_ind", "per_module": per}


# ------------------------------------------------------------------------------------------------ summary.json
def build_summary(rows, cfg, meta):
    """meta: {pilot, encoder_layers, predictor_layers}. All numbers from `rows`; no prose."""
    d = cfg["depth"]
    thr = d["drop_threshold"]
    prim = d["primary"]
    cur, vanish, gaps = {}, {}, {}
    for mod in ("encoder", "predictor"):
        c = curves(rows, mod, prim["probe"], prim["fit"], prim["control"], prim["timepoint"])
        if c:
            cur[mod] = c
            vanish[mod] = vanishing_by_split(c, thr)
            gaps[mod] = ind_ood_gap(c, d["overlap_sd_mult"], d["overlap_depth_fraction"]) if (d["overlap_sd_mult"] is not None and d["overlap_depth_fraction"] is not None) else {}
    kb = {s: (vanish["encoder"][s]["vanish"] or {}).get("depth") for s in SPLITS} if "encoder" in vanish else {}
    mb = {s: (vanish["predictor"][s]["vanish"] or {}).get("depth") for s in SPLITS} if "predictor" in vanish else {}
    first = lambda dd: (min(v for v in dd.values() if v is not None) if any(v is not None for v in dd.values()) else None)
    k_star, m_star = first(kb), first(mb)
    ood_loss = any(v is not None for dd in (kb, mb) for s, v in dd.items() if s != "train")
    ind_loss = any(dd.get("train") is not None for dd in (kb, mb))
    lb, lb_mod = l_best(cur, d["l_best_splits"])
    refine = {}
    for mod, vv in vanish.items():
        for s in SPLITS:
            v = vv[s]["vanish"]
            if v:
                ds = sorted(cur[mod][s])
                prev = [x for x in ds if x < v["depth"]]
                refine["%s/%s" % (mod, s)] = {"between_depths": [int(prev[-1]) if prev else None, v["depth"]], "note": "re-run enc_cache/enc_probe with a finer k_grid in this interval"}
    return {
        "pilot": bool(meta["pilot"]), "encoder_layers": meta["encoder_layers"], "predictor_layers": meta["predictor_layers"],
        "l_best": ({"module": lb["module"], "depth": lb["depth"]} if lb else None), "l_best_value": (lb["value"] if lb else None),
        "l_best_by_module": lb_mod, "l_best_splits": list(d["l_best_splits"]),
        "k_star": k_star, "m_star": m_star, "k_star_by_split": kb, "m_star_by_split": mb,
        "loss_ood_only": (bool(ood_loss and not ind_loss) if (kb or mb) else None),
        "drop_threshold": thr, "overlap_sd_mult": d["overlap_sd_mult"], "overlap_depth_fraction": d["overlap_depth_fraction"], "primary": dict(prim),
        "vanish_detail": vanish, "ind_ood_gap": gaps, "refine_suggestion": refine,
        "verdict_H_d": verdict_h_d(vanish, gaps, thr, d["overlap_sd_mult"], d["min_depths"], d["overlap_depth_fraction"]),
        "definitions": {"k_star": "earliest vanishing depth over splits train/interp/extrap (encoder)", "m_star": "same for predictor hidden probe (fit=new)",
                        "loss_ood_only": "no vanishing depth in split train (either module) and a vanishing depth in interp or extrap (either module)",
                        "vanishing_depth": "first depth whose split-mean R2 is >= drop_threshold below the running maximum of earlier depths; null if none",
                        "l_best": "argmax over (module, depth) of mean R2 over l_best_splits at the primary setting"},
        "notes": "values are computed from results/depth_r2.csv (level == ALL rows); pilot=true marks a subset/partial run",
    }
