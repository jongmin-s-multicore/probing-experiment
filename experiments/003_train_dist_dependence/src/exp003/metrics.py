"""CPU stage: the three 003 metrics from the extract cache. No interpretation here.

  ratio_ol_over_real (k=12)  per size: per-episode ratio  area_ol[k=12] / max(area_real[k=12], 1), mean over episodes, episode-bootstrap 95% CI
                              (same definition as 001 diagnose.area_tables ratio_ol_over_real)
  slope ol@k12               group-level log-log slope of mean log(ol area) on mean log(real area) over sizes, with the within-size episode
                              bootstrap of 001 (_boot_slope copied unchanged, so the value is comparable with 001 results.json
                              diagnose/size_curve/slope_model_vs_real_group_means/ol@k12). Also over subsets of sizes (trained+interp, extrap)
  center c                    size where the (size-mean) ratio crosses 1, linear interpolation between adjacent grid sizes in
                              (log nominal area ratio, log ratio) coordinates (config metrics.center_interp). null + reason if no / several crossings.
Per-episode real and open-loop areas are stored so that everything can be recomputed."""
from __future__ import annotations

import numpy as np

from .common import log, read_json, write_json, area_ratio_of


def boot_ci(x, n=2000, seed=0, ci=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.RandomState(seed)
    b = x[rng.randint(0, len(x), (n, len(x)))].mean(1)
    a = (1 - ci) / 2 * 100
    return {"mean": float(x.mean()), "lo": float(np.percentile(b, a)), "hi": float(np.percentile(b, 100 - a)), "n": int(len(x))}


def boot_slope(x_by_g, y_by_g, n=500, seed=0, ci=0.95):
    """Copied from 001 diagnose._boot_slope (percentiles generalised from 2.5/97.5 to the config ci; identical for ci=0.95)."""
    rng = np.random.RandomState(seed)
    gs = list(x_by_g)

    def slope(idx=None):
        mx, my = [], []
        for g in gs:
            x, y = x_by_g[g], y_by_g[g]
            ii = np.arange(len(x)) if idx is None else rng.randint(0, len(x), len(x))
            mx.append(np.mean(np.log(np.maximum(x[ii], 1))))
            my.append(np.mean(np.log(np.maximum(y[ii], 1))))
        return float(np.polyfit(mx, my, 1)[0]) if len(gs) >= 2 and np.ptp(mx) > 1e-9 else float("nan")

    pt = slope()
    d = np.array([slope(True) for _ in range(n)])
    d = d[np.isfinite(d)]
    a = (1 - ci) / 2 * 100
    return {"slope": pt, "lo": float(np.percentile(d, a)) if len(d) else None, "hi": float(np.percentile(d, 100 - a)) if len(d) else None,
            "n_sizes": len(gs)}


def find_center(sizes, log_area, ratio_mean):
    """sizes: names in ascending nominal area; log_area: log nominal area ratio; ratio_mean: size-mean ratio_ol_over_real.
    Returns {c, log_c, status, reason, crossings:[{between, c, direction}]}. status 'ok' only for exactly one crossing."""
    y = [None if (r is None or not np.isfinite(r) or r <= 0) else float(np.log(r)) for r in ratio_mean]
    cr = []
    for i in range(len(sizes) - 1):
        a, b = y[i], y[i + 1]
        if a is None or b is None:
            continue
        if (a < 0 < b) or (a > 0 > b):          # strict sign change between two grid sizes
            xc = log_area[i] + (0.0 - a) * (log_area[i + 1] - log_area[i]) / (b - a)
            cr.append({"between": [sizes[i], sizes[i + 1]], "log_c": float(xc), "c": float(np.exp(xc)),
                       "direction": "ratio_falls_with_size" if a > 0 else "ratio_rises_with_size"})
    for i, v in enumerate(y):                    # a mean ratio of exactly 1.0 at a grid size (practically never) counts as a crossing at that size
        if v == 0.0:
            cr.append({"between": [sizes[i], sizes[i]], "log_c": float(log_area[i]), "c": float(np.exp(log_area[i])), "direction": "exactly_one_at_grid_point"})
    out = {"crossings": cr, "n_crossings": len(cr), "c": None, "log_c": None}
    if any(v is None for v in y):
        out.update(status="null", reason="a size has no valid mean ratio")
    elif len(cr) == 0:
        side = "above" if all(v > 0 for v in y) else ("below" if all(v < 0 for v in y) else "none")
        out.update(status="null", reason="ratio does not cross 1 on the grid (all sizes %s 1)" % side)
    elif len(cr) > 1:
        out.update(status="null", reason="%d crossings of 1 on the grid (ambiguous); candidates listed in 'crossings'" % len(cr))
    else:
        out.update(status="ok", reason=None, c=cr[0]["c"], log_c=cr[0]["log_c"])
    return out


def compute(cfg, ep, sc, k=None):
    """ep: episodes.json dict, sc: {'area_real' (E,13), 'area_ol' (E,12)}. Returns the metrics dict of one run."""
    mc = cfg["metrics"]
    k = int(k or mc["slope_k"])
    ci = float(mc["ci"])
    grp = np.array([e["group"] for e in ep["episodes"]])
    ok = ~np.array([e["in_train"] for e in ep["episodes"]])
    order = [g for g in cfg["_ds"]["group_order"] if (grp[ok] == g).any()]
    real = sc["area_real"][:, k]
    ol = sc["area_ol"][:, k - 1]
    ratio = ol / np.maximum(real, 1)
    roles = ep.get("roles") or {}
    out = {"k": k, "train_groups": ep.get("train_groups"), "roles": roles, "groups_in_order": order, "by_size": {}}
    for g in order:
        m = ok & (grp == g)
        out["by_size"][g] = {
            "role": roles.get(g), "nominal_area_ratio": area_ratio_of(cfg, g), "n": int(m.sum()),
            "ratio_ol_over_real": boot_ci(ratio[m], n=mc["n_bootstrap"], seed=mc["bootstrap_seed"], ci=ci),
            "area_real_mean": float(real[m].mean()), "area_ol_mean": float(ol[m].mean()),
            "area_real_per_episode": real[m].astype(int).tolist(), "area_ol_per_episode": ol[m].astype(int).tolist()}
    xs = {g: real[ok & (grp == g)] for g in order}
    ys = {g: ol[ok & (grp == g)] for g in order}
    sd = mc["bootstrap_seed"]
    out["slope_ol"] = {
        "all_sizes": boot_slope(xs, ys, n=mc["n_bootstrap_slope"], seed=sd, ci=ci),
        "definition": "polyfit slope of group-mean log(max(ol area,1)) on group-mean log(max(real area,1)) at frame k over the listed sizes; CI = 001 within-size episode bootstrap"}
    for name, sel in (("trained_and_interp", lambda r: r in ("trained", "interp")), ("extrap_only", lambda r: r in ("extrap_small", "extrap_large"))):
        gs = [g for g in order if sel(roles.get(g))]
        out["slope_ol"][name] = boot_slope({g: xs[g] for g in gs}, {g: ys[g] for g in gs}, n=mc["n_bootstrap_slope"], seed=sd, ci=ci) if len(gs) >= 2 else None
        out["slope_ol"][name + "_sizes"] = gs
    la = [float(np.log(area_ratio_of(cfg, g))) for g in order]
    out["center"] = find_center(order, la, [out["by_size"][g]["ratio_ol_over_real"]["mean"] for g in order])
    out["center"]["interp"] = cfg["metrics"]["center_interp"]
    return out


def run(cfg, paths, run_name, dist=None, seed=None):
    cache = paths["cache"] / run_name
    assert (cache / "DONE").exists(), "run extract first: %s" % cache
    ep = read_json(cache / "episodes.json")
    z = np.load(cache / "scalars.npz")
    sc = {k_: z[k_] for k_ in z.files}
    out = compute(cfg, ep, sc)
    out.update({"run_name": run_name, "distribution": dist, "seed": seed, "epoch_loaded": ep["epoch_loaded"], "n_episodes_extracted": ep["n_episodes"],
                "code_commit": ep["code_commit"], "extract_cuda_peak_allocated_gib": ep.get("extract_cuda_peak_allocated_gib")})
    write_json(out, paths["results_dir"] / run_name / "metrics.json")
    log("metrics: %s slope_all=%s c=%s" % (run_name, out["slope_ol"]["all_sizes"]["slope"], out["center"]["c"]))
    return out
