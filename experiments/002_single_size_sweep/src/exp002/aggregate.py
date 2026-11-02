"""Collect Q1/Q2/Q3 outputs into results.json and a decision table. Numbers and statuses only - NO interpretation sentences.

Every pre-registered value is read through PreregReader (config path recorded in results.json -> prereg_values_used):
  prereg.q1_margin.delta_r2_vs_mask_area_baseline, prereg.q1_margin.ci_lower_gt, q2.drop_threshold, q3.n_random_directions,
  q3.criterion_percentile, q3.layers, q3.dose_alpha_sigma_units, prereg.holm_alpha, prereg.family_defs.* (family composition).
A null threshold is NOT judged: the row gets status not_evaluable (001 shrink_ratio_tol case).
Statuses: pass | fail | not_evaluable | pending | exploratory.
Bootstrap: episodes resampled within level; probe-seed mean; for several training seeds the statistic is the mean of the per-seed statistics.
The CI therefore does NOT include training-seed variation; per-training-seed values are listed next to every registered test."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from .common import PreregReader, read_json, write_json, env_info, config_sha256, run_name as mk_run_name, log
from . import evalgrid as EG
from . import stats as ST

RULE_4C = {"small": "small: layer-wise results of (a) are interpreted normally as auxiliary results",
           "medium": "medium: interpret, with a caution note about the influence of unfamiliar inputs",
           "large": "large: do NOT interpret the layer-wise results of (a); report numbers only"}
Q2L = ("Layer-wise probes are trained on predictor activations of the 3 sizes a060/ind/a160. In condition (a) (single_ind) the label is constant during predictor training, "
       "so the probe reads activations of sizes the predictor never saw. For (a) only the change relative to layer 0 is reported; no absolute comparison with (b).")
CI_CAVEAT = ("Episode-cluster bootstrap 95% CI: resamples evaluation episodes only. It does NOT include variation across training seeds "
              "(few seeds); per-training-seed values are listed. Probe seeds are averaged inside each statistic.")


def _git(args):
    try:
        return subprocess.run(["git"] + args, cwd=str(Path(__file__).resolve().parents[2]), capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def _load_cells(path):
    if not Path(path).exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files if not k.startswith("predmean|")}


def _row(id_, hyp, test, family, thr, result, status, p=None, p_adj=None, note=None):
    return {"id": id_, "hypothesis": hyp, "test": test, "family": family, "thresholds_used": thr, "result": result, "status": status,
            "p_value": p, "holm_adjusted_p": p_adj, "note": note}


def _thr(P, *names):
    return {n: {"value": P.used[n]["value"], "config_path": P.used[n]["config_path"]} for n in names if n in P.used}


def _holm_apply(rows, alpha):
    adj, rej = ST.holm([r["p_value"] if r["status"] != "not_evaluable" else None for r in rows], alpha)
    for r, a in zip(rows, adj):
        r["holm_adjusted_p"] = a
    return rej


# ------------------------------------------------------------------------------------------------ H_a
def h_a(cfg, P, boot, rmasks, q1):
    fam = P("prereg.family_defs", "family_defs")["H_a"]
    delta = P("prereg.q1_margin.delta_r2_vs_mask_area_baseline", "q1_margin_delta_r2")
    ci_gt = P("prereg.q1_margin.ci_lower_gt", "q1_margin_ci_lower_gt")
    alpha = P("prereg.holm_alpha", "holm_alpha")
    prim, win, sat = cfg["q1"]["primary_target"], cfg["q1"]["primary_window"], cfg["q1"]["baseline_saturated_r2"]
    rows = []
    for role in fam["roles"]:
        for kind in fam["probes"]:
            tid = "H_a:%s:%s" % (role, kind)
            kr, kd, kb = ["%s|%s|%s|all|%s" % (prim, kind, rep, win) for rep in ("dino", "rand")] + ["%s|mask_ridge|mask|all|%s" % (prim, win)]
            if q1 is None or kr not in q1 or kd not in q1 or kb not in q1:
                rows.append(_row(tid, "H-a", "%s %s: real vs mask baseline & random-init" % (role, kind), "H_a", _thr(P, "q1_margin_delta_r2", "q1_margin_ci_lower_gt"), {}, "pending", note="q1 cells missing"))
                continue
            m = rmasks[role]
            sr, sd, sb = boot.stat(q1[kr], m, "r2"), boot.stat(q1[kd], m, "r2"), boot.stat(q1[kb], m, "r2")
            db, dr = sr["boot"] - sb["boot"], sr["boot"] - sd["boot"]
            res = {"r2_real": sr["point"], "r2_random_init": sd["point"], "r2_mask_baseline": sb["point"],
                   "r2_real_per_probe_seed": sr["per_probe_seed"], "r2_random_init_per_probe_seed": sd["per_probe_seed"], "r2_mask_baseline_per_probe_seed": sb["per_probe_seed"],
                   "delta_vs_mask_baseline": sr["point"] - sb["point"], "delta_vs_mask_baseline_ci95": ST.ci95(db),
                   "delta_vs_random_init": sr["point"] - sd["point"], "delta_vs_random_init_ci95": ST.ci95(dr),
                   "ci_note": CI_CAVEAT}
            p = max(ST.p_one_sided_le(db, 0.0), ST.p_one_sided_le(dr, 0.0))
            if delta is None or ci_gt is None:
                status = "not_evaluable"
                note = "threshold null in config"
            elif sb["point"] >= sat:
                status, note = "not_evaluable", "baseline_saturated (mask baseline R2 %.3f >= %.2f)" % (sb["point"], sat)
            else:
                ok = (res["delta_vs_mask_baseline"] >= delta and res["delta_vs_mask_baseline_ci95"][0] > ci_gt and res["delta_vs_random_init_ci95"][0] > ci_gt)
                status, note = ("pending_holm", None) if ok else ("fail", None)
                res["criteria_before_holm"] = bool(ok)
            rows.append(_row(tid, "H-a", "%s %s: R2 real - mask baseline >= margin, CI lower > ci_lower_gt, and CI lower of (real - random-init) > ci_lower_gt" % (role, kind),
                             "H_a", _thr(P, "q1_margin_delta_r2", "q1_margin_ci_lower_gt", "holm_alpha"), res, status, p=p, note=note))
    ev = [r for r in rows if r["status"] in ("pass", "fail", "pending_holm", "not_evaluable")]
    rej = _holm_apply(ev, alpha) if ev else []
    for r, rj in zip(ev, rej):
        if r["status"] == "pending_holm":
            r["status"] = "pass" if rj else "fail"
            r["result"]["holm_rejects"] = bool(rj)
    for r in rows:
        if r["status"] == "pending_holm":
            r["status"] = "fail"
    return rows


def h_a_early_report(cfg, boot, rmasks, q1):
    """early_flat window: NOT judged (no status, no p, not in the Holm family). Reported with the mask baseline R2 (확정 기록 6 item 1)."""
    prim, win, sat = cfg["q1"]["primary_target"], cfg["q1"]["report_window"], cfg["q1"]["baseline_saturated_r2"]
    out = {"status": "reported_not_judged", "window": win, "frames": cfg["frames"]["early_flat"], "rows": []}
    if q1 is None:
        out["status"] = "pending"
        return out
    for role in ("interp", "extrap"):
        for kind in cfg["probe"]["kinds"]:
            ks = {rep: "%s|%s|%s|all|%s" % (prim, kind, rep, win) for rep in ("dino", "rand")}
            kb = "%s|mask_ridge|mask|all|%s" % (prim, win)
            if any(k not in q1 for k in list(ks.values()) + [kb]):
                continue
            r = {rep: boot.stat(q1[k], rmasks[role], "r2")["point"] for rep, k in ks.items()}
            rb = boot.stat(q1[kb], rmasks[role], "r2")["point"]
            out["rows"].append({"role": role, "probe": kind, "r2_real": r["dino"], "r2_random_init": r["rand"], "r2_mask_baseline": rb,
                                "delta_vs_mask_baseline": r["dino"] - rb, "baseline_saturated": bool(rb >= sat), "baseline_saturated_r2": sat})
    return out


# ------------------------------------------------------------------------------------------------ H_b
def _q2_runs(cfg, paths, cond):
    runs = {}
    for s in cfg["train"]["seeds"]:
        d = paths["results_dir"] / mk_run_name(cond, s)
        c = _load_cells(d / "q2_cells.npz")
        if c is not None and (d / "q2_summary.json").exists():
            sm = read_json(d / "q2_summary.json")
            runs[s] = {"cells": c, "meta": sm["meta"], "diag": sm.get("diagnostics_4c"), "output_space": sm.get("output_space")}
    return runs


def _drop_boot(boot, q1, runs, m, kind, prim, last_layer_key, ceil_key):
    per_seed, boots = {}, []
    for s, r in runs.items():
        L = r["meta"]["n_blocks"]
        kq2 = last_layer_key(r["meta"])
        ck = ceil_key(r["meta"])
        src = r["cells"] if ck.startswith(prim + "|" + kind + "|OUT|") else q1       # output-space ceiling = same probe on encoder tokens, stored in the run
        if kq2 not in r["cells"] or ck not in src:
            return None
        a = boot.stat(src[ck], m, "r2")
        b = boot.stat(r["cells"][kq2], m, "r2")
        per_seed[s] = {"r2_q1_ceiling": a["point"], "r2_q2": b["point"], "drop": a["point"] - b["point"]}
        boots.append(a["boot"] - b["boot"])
    return per_seed, np.mean(boots, axis=0)


def h_b_points(cfg, P, paths, boot, rmasks, q1, cond):
    kind = cfg["q2_impl"]["primary_probe"]
    prim = cfg["q1"]["primary_target"]
    m = rmasks["unseen"]
    runs = _q2_runs(cfg, paths, cond)
    pts = {"last_block": (lambda meta: "%s|%s|layer%d|union" % (prim, kind, meta["n_blocks"]), lambda meta: "%s|%s|dino|union|union" % (prim, kind)),
           "last_rollout_step": (lambda meta: "%s|%s|OUT|rollout|step%d" % (prim, kind, meta["horizon"]),       # output-space probe (확정 기록 6 4a)
                                 lambda meta: "%s|%s|OUT|encoder|step%d" % (prim, kind, meta["horizon"]))}
    out = {}
    for name, (lk, ck) in pts.items():
        if not runs or q1 is None:
            out[name] = None
            continue
        r = _drop_boot(boot, q1, runs, m, kind, prim, lk, ck)
        out[name] = None if r is None else {"per_seed": r[0], "boot": r[1], "seeds": sorted(runs)}
    return out, runs


def h_b(cfg, P, paths, boot, rmasks, q1, cond):
    fam = P("prereg.family_defs", "family_defs")["H_b"]
    thr = P("q2.drop_threshold", "q2_drop_threshold")
    alpha = P("prereg.holm_alpha", "holm_alpha")
    pts, _ = h_b_points(cfg, P, paths, boot, rmasks, q1, cond)
    rows = []
    for name in fam["points"]:
        tid = "H_b:%s" % name
        d = pts.get(name)
        if d is None:
            rows.append(_row(tid, "H-b", "R2 drop vs Q1 at %s < drop_threshold" % name, "H_b", _thr(P, "q2_drop_threshold"), {}, "pending", note="q2 runs of condition %s or q1 cells missing" % cond))
            continue
        b = d["boot"]
        res = {"condition": cond, "training_seeds": d["seeds"], "per_training_seed": d["per_seed"], "drop_mean_over_seeds": float(np.mean([v["drop"] for v in d["per_seed"].values()])),
               "drop_ci95_episode_bootstrap": ST.ci95(b), "probe": cfg["q2_impl"]["primary_probe"], "eval_roles": "unseen (interp+extrap)", "ci_note": CI_CAVEAT}
        if thr is None:
            rows.append(_row(tid, "H-b", "R2 drop at %s" % name, "H_b", {"q2_drop_threshold": {"value": None, "config_path": "q2.drop_threshold"}}, res, "not_evaluable", note="threshold null"))
            continue
        res["all_training_seeds_drop_below_threshold"] = bool(all(v["drop"] < thr for v in d["per_seed"].values()))
        rows.append(_row(tid, "H-b", "drop (Q1 ceiling R2 - Q2 R2) at %s < drop_threshold; one-sided bootstrap p for H0 drop >= threshold" % name, "H_b",
                         _thr(P, "q2_drop_threshold", "holm_alpha"), res, "pending_holm", p=ST.p_one_sided_ge(b, thr)))
    ev = [r for r in rows if r["status"] in ("pending_holm", "not_evaluable")]
    rej = _holm_apply(ev, alpha) if ev else []
    for r, rj in zip(ev, rej):
        if r["status"] == "pending_holm":
            r["status"] = "pass" if rj else "fail"
    return rows


# ------------------------------------------------------------------------------------------------ H_c
def _rank_test(block, cfg, primary):
    """Re-derive outside_percentile / p_rank / monotone from the stored arrays with the config values (not from stored flags)."""
    pct = float(cfg["q3"]["criterion_percentile"])
    mono = float(cfg["q3_impl"]["monotonic_min_spearman"])
    key = "T_size" if primary == "delta_predicted_size" else "T_latent"
    size = block.get("size_direction") or block.get("size_removal")
    t = size[key]
    rr = block["random"][key]
    rho = size.get("spearman_signed_dsize_vs_alpha", size.get("spearman_abs_dsize_vs_lambda"))
    return {"T_size_direction": t, "random_percentile_value": float(np.percentile(rr, pct)), "outside_percentile": bool(t > np.percentile(rr, pct)),
            "p_rank": float((1 + sum(1 for x in rr if x >= t)) / (len(rr) + 1)), "n_random": len(rr),
            "spearman": rho, "monotone": bool(rho is not None and np.isfinite(rho) and abs(rho) >= mono)}


def h_c(cfg, P, paths, cond):
    fam = P("prereg.family_defs", "family_defs")["H_c"]
    nrand = P("q3.n_random_directions", "q3_n_random_directions")
    P("q3.criterion_percentile", "q3_criterion_percentile")
    cfg_alphas = P("q3.dose_alpha_sigma_units", "q3_dose_alpha_sigma_units")
    cfg_layers = P("q3.layers", "q3_layers")
    alpha = P("prereg.holm_alpha", "holm_alpha")
    primary = cfg["q3_impl"]["primary_metric"]
    runs = {}
    for s in cfg["train"]["seeds"]:
        f = paths["results_dir"] / mk_run_name(cond, s) / "q3.json"
        if f.exists():
            runs[s] = read_json(f)
    rows = []
    for name in fam["interventions"]:
        tid = "H_c:%s" % name
        if not runs:
            rows.append(_row(tid, "H-c", name, "H_c", _thr(P, "q3_n_random_directions", "q3_criterion_percentile", "q3_layers"), {}, "pending", note="q3 runs of condition %s missing" % cond))
            continue
        cells, not_removed, bad_n, bad_cfg = {}, [], [], []
        for s, r in runs.items():
            if sorted(r["layers"]) != sorted(cfg_layers):
                bad_cfg.append("seed%s layers %s != config q3.layers %s" % (s, sorted(r["layers"]), cfg_layers))
                continue
            for tag, lay in r["layers"].items():
                if name == "steering":
                    blk = lay["steering"][cfg["q3_impl"]["primary_steering_direction"]]
                else:
                    blk = lay["removal"][cfg["q3_impl"]["primary_removal_method"]]
                    if not blk["removed"]:
                        not_removed.append("seed%s/%s" % (s, tag))
                if name == "steering" and list(blk["alphas_sigma_units"]) != list(cfg_alphas):
                    bad_cfg.append("seed%s/%s alphas differ from config q3.dose_alpha_sigma_units" % (s, tag))
                t = _rank_test(blk, cfg, primary)
                if t["n_random"] != nrand:
                    bad_n.append("seed%s/%s:%d" % (s, tag, t["n_random"]))
                cells["seed%s/%s" % (s, tag)] = t
        res = {"condition": cond, "primary_metric": primary, "cells": cells, "ci_note": "rank-based p over independent random directions per training seed; no episode CI; " + CI_CAVEAT.split(".")[0],
               "direction": cfg["q3_impl"]["primary_steering_direction"] if name == "steering" else cfg["q3_impl"]["primary_removal_method"]}
        thr = _thr(P, "q3_n_random_directions", "q3_criterion_percentile", "q3_dose_alpha_sigma_units", "q3_layers", "holm_alpha")
        if bad_cfg:
            rows.append(_row(tid, "H-c", name, "H_c", thr, res, "not_evaluable", note="stored run does not match the config values: %s" % bad_cfg))
            continue
        if not_removed:
            res["flag"] = "not_removed"
            res["not_removed_cells"] = not_removed
            rows.append(_row(tid, "H-c", name, "H_c", thr, res, "not_evaluable", note="not_removed in %s: no conclusion" % not_removed))
            continue
        if bad_n:
            rows.append(_row(tid, "H-c", name, "H_c", thr, res, "not_evaluable", note="number of random directions != q3.n_random_directions: %s" % bad_n))
            continue
        pmax = max(c["p_rank"] for c in cells.values())
        res["p_max_over_seeds_and_layers"] = pmax
        res["all_outside_percentile"] = all(c["outside_percentile"] for c in cells.values())
        res["all_monotone"] = all(c["monotone"] for c in cells.values())
        rows.append(_row(tid, "H-c", "%s: effect outside %sth percentile of %d random and monotone, in every (training seed, fixed layer) cell; p = max rank p" % (name, P.used["q3_criterion_percentile"]["value"], nrand),
                         "H_c", thr, res, "pending_holm", p=pmax))
    ev = [r for r in rows if r["status"] in ("pending_holm", "not_evaluable")]
    rej = _holm_apply(ev, alpha) if ev else []
    for r, rj in zip(ev, rej):
        if r["status"] == "pending_holm":
            res = r["result"]
            r["status"] = "pass" if (rj and res["all_outside_percentile"] and res["all_monotone"]) else "fail"
    return rows


# ------------------------------------------------------------------------------------------------ cond (a) vs (b)
def _layer_rel_boot(boot, runs, m, kind, prim):
    """(R2 at the last block - R2 at layer 0) per run, union window; bootstrap draws averaged over runs."""
    per, bs = {}, []
    for s, r in runs.items():
        L = r["meta"]["n_blocks"]
        k0, kL = "%s|%s|layer0|union" % (prim, kind), "%s|%s|layer%d|union" % (prim, kind, L)
        if k0 not in r["cells"] or kL not in r["cells"]:
            return None
        a, b = boot.stat(r["cells"][k0], m, "r2"), boot.stat(r["cells"][kL], m, "r2")
        per[s] = {"r2_layer0": a["point"], "r2_last_block": b["point"], "change_vs_layer0": b["point"] - a["point"]}
        bs.append(b["boot"] - a["boot"])
    return (per, np.mean(bs, axis=0), sorted(runs)) if bs else None


def cond_a_vs_b(cfg, P, paths, boot, rmasks, q1):
    """(a) vs (b) family (2 tests, declared). last_rollout_step: drop of the SAME output-space probe. last_block: layer-wise probes of (a) are auxiliary
    (확정 기록 6 4b): compared only through the change vs layer 0, and its status is pending_threshold-gated by the 4c verdict: it is judged only when
    the 4c verdict of (a) is small or medium; 'large' -> numbers only (status not_evaluable, no interpretation)."""
    cond_b = cfg["conditions"]["run_order"][0]
    cond_a = cfg["conditions"]["compare_a"]
    alpha = P("prereg.holm_alpha", "holm_alpha")
    kind, prim, m = cfg["q2_impl"]["primary_probe"], cfg["q1"]["primary_target"], rmasks["unseen"]
    pa, _ = h_b_points(cfg, P, paths, boot, rmasks, q1, cond_a)
    pb, _ = h_b_points(cfg, P, paths, boot, rmasks, q1, cond_b)
    ra, rb_ = _q2_runs(cfg, paths, cond_a), _q2_runs(cfg, paths, cond_b)
    verdicts = [r["diag"]["verdict"] for r in ra.values() if r.get("diag")]
    gate = "not_evaluable" if not verdicts else ("large" if "large" in verdicts else ("medium" if "medium" in verdicts else "small"))
    rows = []
    for name in P("prereg.family_defs", "family_defs")["H_b"]["points"]:
        tid = "cond_a_vs_b:%s" % name
        if name == "last_block":
            la, lb = (_layer_rel_boot(boot, ra, m, kind, prim) if ra else None), (_layer_rel_boot(boot, rb_, m, kind, prim) if rb_ else None)
            if la is None or lb is None:
                rows.append(_row(tid, "cond (a) vs (b)", "change_vs_layer0(a) - change_vs_layer0(b) at last block", "cond_a_vs_b", {}, {}, "pending", note="results of one condition missing"))
                continue
            diff = la[1] - lb[1]
            res = {"change_vs_layer0_a_minus_b": float(np.mean([v["change_vs_layer0"] for v in la[0].values()]) - np.mean([v["change_vs_layer0"] for v in lb[0].values()])),
                   "ci95_episode_bootstrap": ST.ci95(diff), "per_training_seed_a": la[0], "per_training_seed_b": lb[0], "diag_4c_verdict_a": gate,
                   "interpretation_rule": RULE_4C.get(gate), "limitation": Q2L, "ci_note": CI_CAVEAT, "unadjusted_two_sided_p": ST.p_two_sided_zero(diff)}
            if gate == "large" or gate == "not_evaluable":
                rows.append(_row(tid, "cond (a) vs (b)", "layer-wise change vs layer 0, auxiliary", "cond_a_vs_b", _thr(P, "holm_alpha"), res, "not_evaluable", note="4c verdict of (a) = %s: numbers only" % gate))
            else:
                rows.append(_row(tid, "cond (a) vs (b)", "two-sided bootstrap p for H0 change_vs_layer0 equal in (a),(b); auxiliary", "cond_a_vs_b", _thr(P, "holm_alpha"), res, "pending_holm", p=ST.p_two_sided_zero(diff),
                                 note=RULE_4C[gate]))
            continue
        a, b = pa.get(name), pb.get(name)
        if a is None or b is None:
            rows.append(_row(tid, "cond (a) vs (b)", "drop(%s) - drop(%s) at %s" % (cond_a, cond_b, name), "cond_a_vs_b", {}, {}, "pending",
                             note="results of one condition missing (a) is retrained later"))
            continue
        diff = a["boot"] - b["boot"]
        res = {"drop_a_minus_drop_b": float(np.mean([v["drop"] for v in a["per_seed"].values()]) - np.mean([v["drop"] for v in b["per_seed"].values()])),
               "ci95_episode_bootstrap": ST.ci95(diff), "per_training_seed_a": a["per_seed"], "per_training_seed_b": b["per_seed"], "ci_note": CI_CAVEAT,
               "probe": "output-space: the same Q1 probe (encoder tokens only) applied to predicted z in both conditions"}
        rows.append(_row(tid, "cond (a) vs (b)", "two-sided bootstrap p for H0 drop_a = drop_b at %s" % name, "cond_a_vs_b", _thr(P, "holm_alpha"), res, "pending_holm", p=ST.p_two_sided_zero(diff)))
    ev = [r for r in rows if r["status"] in ("pending_holm", "not_evaluable")]
    if ev:
        adj, rej = ST.holm([r["p_value"] if r["status"] == "pending_holm" else None for r in ev], alpha)
        for r, a_, rj in zip(ev, adj, rej):
            if r["status"] == "pending_holm":
                r["holm_adjusted_p"] = a_
                r["status"] = "pass" if rj else "fail"
                r["note"] = (r["note"] + "; " if r["note"] else "") + "pass = difference between conditions detected (not a direction claim)"
    return rows


def diagnostics_rows(cfg, paths):
    """4c diagnostics for every condition/seed: verdict from the stored q2 diagnostics (rule fixed in 확정 기록 7 A). Missing data -> not_evaluable."""
    rows = []
    for cond in [c["name"] for c in cfg["conditions"]["list"]]:
        runs = _q2_runs(cfg, paths, cond)
        if not runs:
            rows.append(_row("diag_4c:%s" % cond, "4c", "activation shift / next-frame error vs ind", "diagnostic", {}, {}, "not_evaluable", note="no q2 runs for %s" % cond))
            continue
        per = {s: r["diag"] for s, r in runs.items() if r.get("diag")}
        vs = [d["verdict"] for d in per.values()]
        v = "not_evaluable" if (not vs or "not_evaluable" in vs) else ("large" if "large" in vs else ("medium" if "medium" in vs else "small"))
        rows.append(_row("diag_4c:%s" % cond, "4c", "activation shift (vs noise floor of ind halves) and next-frame error ratio, levels %s" % cfg["q2_diag"]["levels"], "diagnostic",
                         {"diag_4c": {k: cfg["diag_4c"][k] for k in ("activation_small_max_ratio", "activation_medium_max_ratio", "error_large_min_ratio")}},
                         {"verdict_worst_over_seeds": v, "interpretation_rule": RULE_4C.get(v), "per_training_seed": per}, v if v == "not_evaluable" else "decided",
                         note="status decided = verdict given by the fixed rule; verdict value is in result.verdict_worst_over_seeds"))
    return rows


def layerwise_block(cfg, paths, boot, rmasks):
    """Layer-wise probe results: (b) absolute + relative to layer 0; any other condition: relative to layer 0 ONLY (no absolute values)."""
    kind, prim = cfg["q2_impl"]["primary_probe"], cfg["q1"]["primary_target"]
    out = {"limitation": Q2L, "conditions": {}}
    for cond in [c["name"] for c in cfg["conditions"]["list"]]:
        is_b = cond == cfg["conditions"]["run_order"][0]
        runs = _q2_runs(cfg, paths, cond)
        d = {"role": "main" if is_b else "auxiliary (relative to layer 0 only)", "per_seed": {}}
        for s, r in runs.items():
            L = r["meta"]["n_blocks"]
            g = {}
            for j in range(L + 1):
                k = "%s|%s|layer%d|union" % (prim, kind, j)
                if k in r["cells"] and "%s|%s|layer0|union" % (prim, kind) in r["cells"]:
                    for role in ("unseen", "trained"):
                        v, v0 = boot.stat(r["cells"][k], rmasks[role], "r2")["point"], boot.stat(r["cells"]["%s|%s|layer0|union" % (prim, kind)], rmasks[role], "r2")["point"]
                        g.setdefault(str(j), {})[role] = {"change_vs_layer0": float(v - v0)} if not is_b else {"r2": float(v), "change_vs_layer0": float(v - v0)}
            d["per_seed"][str(s)] = g
        out["conditions"][cond] = d
    return out


# ------------------------------------------------------------------------------------------------ exploratory (no p-values)
def exploratory(cfg, paths, boot, rmasks, q1, q1_sum):
    out = {"note": "exploratory: figures/tables only, no p-values and no significance statements"}
    if q1_sum is not None:
        out["q1_cells_summary"] = "see results.json q1.cells (all windows, roles, regimes incl. early->late)"
    grid = {}
    for cond in [c["name"] for c in cfg["conditions"]["list"]]:
        runs = _q2_runs(cfg, paths, cond)
        for s, r in runs.items():
            g = {}
            for k, S in r["cells"].items():
                if ("|layer" in k and k.endswith("|union")) or "|OUT|" in k:
                    g[k] = {role: float(boot.stat(S, rmasks[role], "r2")["point"]) for role in ("unseen", "trained")}
            grid["%s/seed%s" % (cond, s)] = g
    out["q2_layer_time_grid_r2_point_estimates"] = grid
    ex = {}
    for cond in [c["name"] for c in cfg["conditions"]["list"]]:
        for s_, f in ((s_, paths["results_dir"] / mk_run_name(cond, s_) / "q3.json") for s_ in cfg["train"]["seeds"]):
            if not f.exists():
                continue
            for tag, lay in read_json(f)["layers"].items():
                for dn, blk in lay.get("steering", {}).items():
                    if blk.get("tier") == "exploratory":
                        ex["%s/seed%s/%s/steering/%s" % (cond, s_, tag, dn)] = {"T_size": blk["size_direction"]["T_size"], "spearman": blk["size_direction"].get("spearman_signed_dsize_vs_alpha")}
                for mn, blk in lay.get("removal", {}).items():
                    if blk.get("tier") == "exploratory":
                        ex["%s/seed%s/%s/removal/%s" % (cond, s_, tag, mn)] = {"T_size": blk["size_removal"]["T_size"], "removed": blk["removed"]}
    out["q3_exploratory_methods"] = {"note": "meandiff steering and LEACE removal: exploratory, never used in a hypothesis verdict; values only (no p, no percentile flag)", "values": ex}
    out["per_level_note"] = "per-level breakdown: use results/q1/test_rows.json level labels with results/q1/cells.npz (per-episode statistics)"
    return out


def run(cfg, paths):
    P = PreregReader(cfg)
    rd = paths["results_dir"]
    q1 = _load_cells(rd / "q1" / "cells.npz")
    q1_sum = read_json(rd / "q1" / "q1_summary.json") if (rd / "q1" / "q1_summary.json").exists() else None
    te_rows = read_json(rd / "q1" / "test_rows.json") if (rd / "q1" / "test_rows.json").exists() else None
    levels = list(cfg["data"]["levels"])
    if te_rows is None:
        # level order of the evaluation episodes is fixed by the cache table: levels x eval_episodes_per_level
        te_rows = [{"level": l, "role": EG_roles(cfg)[l]} for l in levels for _ in range(cfg["data"]["eval_episodes_per_level"])]
    boot = ST.Bootstrap(EG.level_index(te_rows, levels), cfg["probe_impl"]["n_bootstrap"], cfg["probe_impl"]["bootstrap_seed"])
    rmasks = EG.role_masks(te_rows)
    cond_b = cfg["conditions"]["run_order"][0]
    table = []
    table += h_a(cfg, P, boot, rmasks, q1)
    table += h_b(cfg, P, paths, boot, rmasks, q1, cond_b)
    table += h_c(cfg, P, paths, cond_b)
    table += cond_a_vs_b(cfg, P, paths, boot, rmasks, q1)
    table += diagnostics_rows(cfg, paths)
    table.append(_row("exploratory", "-", "layer x time heatmap, per-level R2, other layers/ranks, behavior contrast", "exploratory", {}, {"see": "exploratory block"}, "exploratory"))
    attempts = read_json(rd / "attempts.json") if (rd / "attempts.json").exists() else {}
    res = {
        "experiment": cfg["name"], "created": time.strftime("%F %T"), "config_path": cfg["_config_path"], "config_sha256": config_sha256(cfg["_config_path"]),
        "git": {"commit": _git(["rev-parse", "HEAD"]), "dirty": bool(_git(["status", "--porcelain", "--", "."]))},
        "environment": env_info(), "ci_caveat": CI_CAVEAT, "prereg_values_used": P.used,
        "decision_table": table, "h_a_early_flat_report": h_a_early_report(cfg, boot, rmasks, q1), "q2_layerwise": layerwise_block(cfg, paths, boot, rmasks),
        "train_budgets": _train_budgets(cfg, paths), "q1": q1_sum, "exploratory": exploratory(cfg, paths, boot, rmasks, q1, q1_sum),
        "attempts": attempts, "smoke": read_json(rd / "smoke.json") if (rd / "smoke.json").exists() else None,
        "interpretation": None,
    }
    write_json(res, Path(paths["exp"]) / "results.json" if not paths.get("results_json") else paths["results_json"])
    log("aggregate: %d decision rows: %s" % (len(table), {s: sum(1 for r in table if r["status"] == s) for s in sorted({r["status"] for r in table})}))
    return res


def _train_budgets(cfg, paths):
    out = {}
    for c in [c["name"] for c in cfg["conditions"]["list"]]:
        for s in cfg["train"]["seeds"]:
            f = Path(paths["train_base"]) / "outputs" / mk_run_name(c, s) / "train_budget.json" if paths.get("train_base") else None
            if f is not None and f.exists():
                out["%s/seed%s" % (c, s)] = read_json(f)
    return out


def EG_roles(cfg):
    from .data import level_roles
    return level_roles(cfg)
