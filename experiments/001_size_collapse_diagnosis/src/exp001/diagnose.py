"""CPU stage: pixel-area and decoder-free latent diagnostics from the extract cache (candidates D and E).

Reports, per size group and per step k (macro frames; k-th frame = 5k raw frames):
  area (real / decode(encode(real)) / teacher-forced 1-step / open-loop)   [pixel mask, 224x224]
  ratios decoded/real (per episode, bootstrap CI over episodes)
  threshold-free per-episode shrink indicator: model area closer to the InD reference area than to the episode's own real area
  (+ counts with config.diagnose.thresholds.shrink_ratio_tol if a human has set it)
  decoder-free latent distances (relative error to own GT latent on cloth / non-cloth patches; nearest-neighbour GT distance to
  InD-size scenes vs same-size scenes)
Nothing here interprets the numbers.
"""
from __future__ import annotations

import numpy as np

from .common import log, read_json, write_json, area_ratio_of
from .maskcheck import corr_stats


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.RandomState(seed)
    b = x[rng.randint(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "n": int(len(x))}


def load_cache(cache):
    ep = read_json(cache / "episodes.json")
    sc = np.load(cache / "scalars.npz")
    return ep, {k: sc[k] for k in sc.files}


def area_tables(ep, sc, thresholds, ks, ref="ind"):
    groups = sorted({e["group"] for e in ep["episodes"]})
    in_train = np.array([e["in_train"] for e in ep["episodes"]])
    grp = np.array([e["group"] for e in ep["episodes"]])
    A = {"real": sc["area_real"], "recon": sc["area_recon"],
         "tf1": np.concatenate([np.full((len(grp), 1), np.nan), sc["area_tf1"]], 1),     # frame index 0 undefined
         "ol": np.concatenate([np.full((len(grp), 1), np.nan), sc["area_ol"]], 1)}
    ind_ref = np.median(sc["area_real"][(grp == ref) & ~in_train], axis=0) if ((grp == ref) & ~in_train).any() else None
    out = {"reference_group": ref, "groups": {}, "ind_reference_area_by_frame": None if ind_ref is None else ind_ref.tolist(), "notes": {
        "frame_index": "k = macro frame (5 raw frames each); area in pixels at 224x224",
        "in_train_excluded": "episodes used for training this model are excluded from every statistic"}}
    for g in groups:
        m = (grp == g) & ~in_train
        res = {"role": ep.get("roles", {}).get(g), "n_episodes": int(m.sum()), "n_excluded_in_train": int(((grp == g) & in_train).sum()), "by_k": {}, "last_frame": {}}
        if m.sum() == 0:
            out["groups"][g] = res
            continue
        for k in ks + [12]:
            row = {s: boot_ci(A[s][m, k]) for s in A}
            for s in ("recon", "tf1", "ol"):
                row[f"ratio_{s}_over_real"] = boot_ci(A[s][m, k] / np.maximum(A["real"][m, k], 1))
            res["by_k"][str(k)] = row
        k = 12
        res["last_frame"] = res["by_k"]["12"]
        real_k, ol_k, tf_k = A["real"][m, k], A["ol"][m, k], A["tf1"][m, k]
        if ind_ref is not None:
            closer = lambda a: (np.abs(a - ind_ref[k]) < np.abs(a - real_k)).astype(float)
            res["closer_to_ind_ref_than_to_own_real_k12"] = {"ol": boot_ci(closer(ol_k)), "tf1": boot_ci(closer(tf_k)), "recon": boot_ci(closer(A["recon"][m, k]))}
            res["closer_to_ind_ref_all_k"] = {
                s: boot_ci(np.concatenate([(np.abs(A[s][m, kk] - ind_ref[kk]) < np.abs(A[s][m, kk] - A["real"][m, kk])).astype(float) for kk in range(1, 13)]))
                for s in ("recon", "tf1", "ol")}
        tol = thresholds.get("shrink_ratio_tol")
        if tol is None:
            res["shrink_count"] = "TODO: thresholds.shrink_ratio_tol not set (needs human decision)"
        else:
            r = ol_k / np.maximum(real_k, 1)
            res["shrink_count"] = {"tol": tol, "n_shrunk": int((r < tol).sum()), "n": int(m.sum()), "definition": "open-loop area / real area < tol at k=12"}
        res["per_episode_ratio_ol_k12"] = (ol_k / np.maximum(real_k, 1)).tolist()
        res["per_episode_ratio_recon_k12"] = (A["recon"][m, k] / np.maximum(real_k, 1)).tolist()
        out["groups"][g] = res
    return out


def latent_distances(cache, ep, sc, ks, cloth_cov=0.25, ref="ind"):
    """Decoder-free distances. Memory: one frame-slice of all episodes at a time."""
    grp = np.array([e["group"] for e in ep["episodes"]])
    ok = ~np.array([e["in_train"] for e in ep["episodes"]])
    gt = np.load(cache / "patch_gt.npy", mmap_mode="r")
    pred = {"tf1": np.load(cache / "patch_tf1.npy", mmap_mode="r"), "ol": np.load(cache / "patch_ol.npy", mmap_mode="r")}
    cov = sc["cov_real"].astype(np.float32)
    res = {"definition": {
        "rel_err_self": "||z_hat - z_gt|| / ||z_gt|| per patch (L2 over 384 dims), averaged over cloth patches (real cloth coverage > %.2f) or the rest" % cloth_cov,
        "nn": "min over other episodes of the L2 distance between flattened 196x384 latents at the same step k. ind = episodes of the reference group (%s), " % ref +
              "same = episodes of the same group (excluding self). ratio = d_nn_ind / d_nn_same (<1: closer to InD-size scenes)"}, "groups": {}}
    groups = sorted(set(grp[ok]))
    for kind in ("tf1", "ol"):
        for g in groups:
            res["groups"].setdefault(g, {})[kind] = {}
    for k in ks:
        G = np.asarray(gt[:, k], dtype=np.float32).reshape(len(grp), -1)
        for kind, arr in pred.items():
            P = np.asarray(arr[:, k - 1], dtype=np.float32)     # predicted frame k
            Pf = P.reshape(len(grp), -1)
            Z = np.asarray(gt[:, k], dtype=np.float32)
            pn = np.linalg.norm(P - Z, axis=-1)                  # (E,196)
            zn = np.linalg.norm(Z, axis=-1)
            sq = lambda a: (a * a).sum(1)
            d2 = sq(Pf)[:, None] + sq(G)[None, :] - 2 * Pf @ G.T     # (E_pred, E_gt)
            d = np.sqrt(np.maximum(d2, 0))
            for g in groups:
                m = np.where((grp == g) & ok)[0]
                if len(m) == 0:
                    continue
                c = cov[m, k] > cloth_cov
                rel = pn[m] / np.maximum(zn[m], 1e-6)
                cl = np.array([rel[i][c[i]].mean() if c[i].any() else np.nan for i in range(len(m))])
                bg = np.array([rel[i][~c[i]].mean() if (~c[i]).any() else np.nan for i in range(len(m))])
                ind_idx = np.where((grp == ref) & ok)[0]
                d_ind = np.array([d[i, ind_idx[ind_idx != i]].min() for i in m])
                same_idx = np.where((grp == g) & ok)[0]
                d_same = np.array([d[i, same_idx[same_idx != i]].min() if len(same_idx) > 1 else np.nan for i in m])
                d_self = np.array([d[i, i] for i in m])
                res["groups"][g][kind][str(k)] = {
                    "rel_err_self_cloth_patches": boot_ci(cl), "rel_err_self_other_patches": boot_ci(bg),
                    "d_self": boot_ci(d_self), "d_nn_ind": boot_ci(d_ind), "d_nn_same_group": boot_ci(d_same),
                    "ratio_nn_ind_over_same": boot_ci(d_ind / d_same),
                    "frac_nearest_is_ind_scene": boot_ci((d_ind < d_same).astype(float)),
                }
    return res


def _boot_slope(x_by_g, y_by_g, n=500, seed=0):
    """slope of group-mean(log y) on group-mean(log x) with an episode-level cluster bootstrap (resample within group)."""
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
    return {"slope": pt, "lo": float(np.percentile(d, 2.5)) if len(d) else None, "hi": float(np.percentile(d, 97.5)) if len(d) else None}


def size_curve(cfg, ep, sc, k_list=(3, 6, 12)):
    """Size-vs-area curve over all size groups (not in_train): mean pixel area per group for real / recon / tf1 / ol at frames k,
    measured area ratio at frame 3 against the reference group, and the group-level log-log slope (model vs real) with cluster bootstrap CI."""
    ref = cfg["diagnose"]["reference_group"]
    grp = np.array([e["group"] for e in ep["episodes"]])
    ok = ~np.array([e["in_train"] for e in ep["episodes"]])
    order = [g for g in cfg["_ds"]["group_order"] if (grp[ok] == g).any()]
    A = {"real": sc["area_real"], "recon": sc["area_recon"], "tf1": sc["area_tf1"], "ol": sc["area_ol"]}
    off = {"real": 0, "recon": 0, "tf1": 1, "ol": 1}          # tf1/ol arrays start at frame 1
    out = {"groups_in_order": order, "reference_group": ref, "by_group": {}, "slope_model_vs_real_group_means": {}}
    ref_m = sc["area_real"][ok & (grp == ref), 3].mean() if (ok & (grp == ref)).any() else None
    for g in order:
        m = ok & (grp == g)
        row = {"role": ep.get("roles", {}).get(g), "nominal_area_ratio": area_ratio_of(cfg, g), "n": int(m.sum()),
               "measured_area_ratio_frame3_vs_ref": None if not ref_m else float(sc["area_real"][m, 3].mean() / ref_m)}
        for k in k_list:
            row[f"k{k}"] = {s: float(A[s][m, k - off[s]].mean()) for s in A}
        out["by_group"][g] = row
    for s in ("recon", "tf1", "ol"):
        for k in k_list:
            if k - off[s] < 0:
                continue
            xs = {g: A["real"][ok & (grp == g), k] for g in order}
            ys = {g: A[s][ok & (grp == g), k - off[s]] for g in order}
            out["slope_model_vs_real_group_means"][f"{s}@k{k}"] = _boot_slope(xs, ys, seed=cfg["diagnose"].get("bootstrap_seed", 0))
    return out


def repro_check_summary(cfg, area, curve):
    """Facts for the reproduction check: does the model trained on the baseline size show 'shrinks toward the training size' on the
    reported sizes? Numbers only; no threshold is applied."""
    rc = cfg["repro_check"]
    res = {"note": rc.get("note"), "reference_group": area["reference_group"], "groups": {}}
    for g in rc["groups"]:
        a = area["groups"].get(g)
        if not a or not a.get("by_k"):
            res["groups"][g] = None
            continue
        k12 = a["by_k"]["12"]
        res["groups"][g] = {
            "n_episodes": a["n_episodes"],
            "area_k12_px": {s: k12[s]["mean"] for s in ("real", "recon", "tf1", "ol")},
            "ind_reference_area_k12_px": (area["ind_reference_area_by_frame"] or [None] * 13)[12],
            "ratio_over_real_k12": {s: k12[f"ratio_{s}_over_real"] for s in ("recon", "tf1", "ol")},
            "closer_to_reference_than_to_own_real_k12": a.get("closer_to_ind_ref_than_to_own_real_k12"),
            "closer_to_reference_all_k": a.get("closer_to_ind_ref_all_k"),
            "shrink_count": a.get("shrink_count"),
        }
    res["size_curve_slopes_model_vs_real"] = curve["slope_model_vs_real_group_means"]
    return res


def label_check(cfg, ep, sc):
    """Auxiliary validation of the probe label: correlation of log(cloth_dimx*cloth_dimy) (metadata particle count) with log(mask area).
    The primary label stays the mask area. Overall over non-train episodes; per frame for frames 3 and 12."""
    ok = ~np.array([e["in_train"] for e in ep["episodes"]])
    part = np.array([np.nan if e.get("particles") is None else e["particles"] for e in ep["episodes"]], float)
    res = {"definition": "corr(log particles, log mask area); particles = cloth_dimx*cloth_dimy from metadata.pt", "by_frame": {}}
    if not np.isfinite(part[ok]).any():
        res["status"] = "particle counts unavailable in extract cache (episode_meta.json missing at extract time)"
        return res
    for k in (0, 3, 6, 12):
        res["by_frame"][str(k)] = corr_stats(part[ok], sc["area_real"][ok, k])
    res["all_frames_pooled"] = corr_stats(np.repeat(part[ok], sc["area_real"].shape[1]), sc["area_real"][ok].reshape(-1))
    return res


def run(cfg, paths, run_name, cache_suffix="", out_name="diagnose.json", area_only=False):
    cache = paths["cache"] / (run_name + cache_suffix)
    assert (cache / "DONE").exists(), f"run extract first: {cache}"
    dcfg = cfg["diagnose"]
    ks = [k for k in dcfg["rollout_ks"] if 1 <= k <= 12]
    ep, sc = load_cache(cache)
    ref = dcfg["reference_group"]
    out = {"run_name": run_name, "dataset": ep.get("dataset"), "epoch_loaded": ep["epoch_loaded"], "n_episodes_extracted": ep["n_episodes"],
           "code_commit": ep["code_commit"], "thresholds": dcfg["thresholds"], "train_groups": ep.get("train_groups"), "roles": ep.get("roles")}
    out["area"] = area_tables(ep, sc, dcfg["thresholds"], ks, ref)
    out["size_curve"] = size_curve(cfg, ep, sc)
    out["repro_check"] = repro_check_summary(cfg, out["area"], out["size_curve"])
    out["label_check_particles_vs_mask_area"] = label_check(cfg, ep, sc)
    log("diagnose: area tables, size curve, repro check, label check done")
    if not area_only:
        out["latent_distance"] = latent_distances(cache, ep, sc, ks, ref=ref)
        log("diagnose: latent distances done")
    d = paths["results_dir"] / run_name
    write_json(out, d / out_name)
    return out
