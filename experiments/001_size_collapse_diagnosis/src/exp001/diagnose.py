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

from .common import log, read_json, write_json


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


def area_tables(ep, sc, thresholds, ks):
    groups = sorted({e["group"] for e in ep["episodes"]})
    in_train = np.array([e["in_train"] for e in ep["episodes"]])
    grp = np.array([e["group"] for e in ep["episodes"]])
    A = {"real": sc["area_real"], "recon": sc["area_recon"],
         "tf1": np.concatenate([np.full((len(grp), 1), np.nan), sc["area_tf1"]], 1),     # frame index 0 undefined
         "ol": np.concatenate([np.full((len(grp), 1), np.nan), sc["area_ol"]], 1)}
    ind_ref = np.median(sc["area_real"][(grp == "ind") & ~in_train], axis=0) if ((grp == "ind") & ~in_train).any() else None
    out = {"groups": {}, "ind_reference_area_by_frame": None if ind_ref is None else ind_ref.tolist(), "notes": {
        "frame_index": "k = macro frame (5 raw frames each); area in pixels at 224x224",
        "in_train_excluded": "episodes used for training this model are excluded from every statistic"}}
    for g in groups:
        m = (grp == g) & ~in_train
        res = {"n_episodes": int(m.sum()), "n_excluded_in_train": int(((grp == g) & in_train).sum()), "by_k": {}, "last_frame": {}}
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


def latent_distances(cache, ep, sc, ks, cloth_cov=0.25):
    """Decoder-free distances. Memory: one frame-slice of all episodes at a time."""
    grp = np.array([e["group"] for e in ep["episodes"]])
    ok = ~np.array([e["in_train"] for e in ep["episodes"]])
    gt = np.load(cache / "patch_gt.npy", mmap_mode="r")
    pred = {"tf1": np.load(cache / "patch_tf1.npy", mmap_mode="r"), "ol": np.load(cache / "patch_ol.npy", mmap_mode="r")}
    cov = sc["cov_real"].astype(np.float32)
    res = {"definition": {
        "rel_err_self": "||z_hat - z_gt|| / ||z_gt|| per patch (L2 over 384 dims), averaged over cloth patches (real cloth coverage > %.2f) or the rest" % cloth_cov,
        "nn": "min over other episodes of the L2 distance between flattened 196x384 latents at the same step k. ind = InD-size episodes, "
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
                ind_idx = np.where((grp == "ind") & ok)[0]
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


def run(cfg, paths, run_name):
    cache = paths["cache"] / run_name
    assert (cache / "DONE").exists(), f"run extract first: {cache}"
    dcfg = cfg["diagnose"]
    ks = [k for k in dcfg["rollout_ks"] if 1 <= k <= 12]
    ep, sc = load_cache(cache)
    out = {"run_name": run_name, "epoch_loaded": ep["epoch_loaded"], "n_episodes_extracted": ep["n_episodes"],
           "code_commit": ep["code_commit"], "thresholds": dcfg["thresholds"]}
    out["area"] = area_tables(ep, sc, dcfg["thresholds"], ks)
    log("diagnose: area tables done")
    out["latent_distance"] = latent_distances(cache, ep, sc, ks)
    log("diagnose: latent distances done")
    d = paths["results_dir"] / run_name
    write_json(out, d / "diagnose.json")
    return out
