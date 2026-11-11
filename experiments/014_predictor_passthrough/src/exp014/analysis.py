"""Pair analysis + verdict of experiment 014 (no interpretation; numbers and rule outcomes only)."""
from __future__ import annotations

import numpy as np

from . import metrics as M


def analyze_pair(Kx, Ky, groups, Dx, Dy, cfg, null=True, reverse=True):
    m = cfg["metrics"]
    alphas = [float(a) for a in m["ridge_r2"]["alpha_grid"]]
    k, seed, lvl = int(m["ridge_r2"]["folds"]), int(cfg["seed"]), float(m["bootstrap"]["ci"])
    nb, nn = int(m["bootstrap"]["n"]), int(m["null_shuffle"]["n"])
    out = {"n_inputs": int(Kx.shape[0]), "n_episodes": int(len(np.unique(groups)))}
    c = M.cka(Kx, Ky)
    cci, _ = M.boot_cka(Kx, Ky, groups, nb, lvl, seed + 2)
    fw = M.ridge_cv(Kx, Ky, groups, Dx, alphas, k, seed)
    rci, _ = M.boot_r2(fw["sse"], fw["sst"], groups, nb, lvl, seed + 2)
    out["cka"] = {"point": c, "ci90": cci}
    out["r2"] = {"point": fw["r2"], "ci90": rci, "alpha_per_fold": fw["alphas"], "alpha_grid": alphas,
                 "alpha_at_grid_edge_folds": int(sum(a in (alphas[0], alphas[-1]) for a in fw["alphas"]))}
    if null:
        nc, nr = M.null_shuffle(Kx, Ky, groups, Dx, alphas, k, seed + 1, nn)
        out["cka"]["null"] = M.summarize(nc)
        out["r2"]["null"] = M.summarize(nr)
    if reverse:
        bw = M.ridge_cv(Ky, Kx, groups, Dy, alphas, k, seed)
        bci, _ = M.boot_r2(bw["sse"], bw["sst"], groups, nb, lvl, seed + 2)
        out["r2_reverse_descriptive"] = {"point": bw["r2"], "ci90": bci, "alpha_per_fold": bw["alphas"]}
    return out


def verdict_pass(res, cfg, gap_min=0.1):
    """H-pass for (enc_k12 ours -> pred_out6 random). criterion from config (confirmed); invalid rule = experimenter default (recorded):
    invalid if observed CKA - shuffled-null mean CKA < gap_min or observed CKA <= null q95."""
    cr = cfg["criterion"]
    if res is None or "not_evaluable" in res:
        return {"verdict": "not_evaluable", "reason": (res or {}).get("not_evaluable")}
    c, r = res["cka"], res["r2"]
    nm, nq = c["null"]["mean"], c["null"]["q95"]
    invalid = (c["point"] - nm < gap_min) or (c["point"] <= nq)
    met = (c["point"] >= cr["cka_min"]) and (r["point"] >= cr["r2_min"])
    return {"verdict": "invalid" if invalid else ("criterion_met" if met else "criterion_not_met"),
            "point_cka_ge": bool(c["point"] >= cr["cka_min"]), "point_r2_ge": bool(r["point"] >= cr["r2_min"]),
            "ci_lower_cka_ge": bool(c["ci90"][0] >= cr["cka_min"]), "ci_lower_r2_ge": bool(r["ci90"][0] >= cr["r2_min"]),
            "cka_minus_null_mean": c["point"] - nm, "cka_gt_null_q95": bool(c["point"] > nq),
            "invalid_rule": "observed CKA - null mean CKA < %.2f or observed CKA <= null 95th percentile (experimenter default; raw null stats in results)" % gap_min,
            "criterion": {"cka_min": cr["cka_min"], "r2_min": cr["r2_min"]}}
