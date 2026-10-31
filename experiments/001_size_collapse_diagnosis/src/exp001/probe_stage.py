"""CPU stage: run the size-probe protocols on the extract cache -> results/<run>/probe.json (+ size_direction.npz).

Target = log(real cloth pixel area) at the labelled frame. Latent types:
  gt (encoder output of the real frame), tf1 (teacher-forced predictor output for frame k), ol (open-loop output for frame k),
  h0..h6 (predictor residual stream at block inputs / before the final norm, teacher-forced pass; label = area of the INPUT frame),
  rvit (random-init DINOv2 ViT-S/14 on real frames), pix (14x14 mean-RGB grid of the real frame).
"""
from __future__ import annotations

import numpy as np

from .common import log, read_json, write_json
from . import probes as P

METRICS = ["r2_overall", "r2_within_group", "acc_cls", "acc_cls_pm1", "slope", "slope_group_means"]


def load_all(cache, cfg, pool):
    ep = read_json(cache / "episodes.json")
    eps = ep["episodes"]
    ok = np.array([not e["in_train"] for e in eps])
    sc = np.load(cache / "scalars.npz")
    area = sc["area_real"]
    names = np.array([e["group"] for e in eps])
    # groups ordered by the nominal area ratio of the selected dataset (config group_order), only those present among eligible episodes
    groups = [g for g in cfg["_ds"]["group_order"] if (names[ok] == g).any()]
    gid = {g: i for i, g in enumerate(groups)}
    cmap = cfg["probe"]["class_groups"]            # class index = position in the full group_order (9 classes for cloth_sim)
    ep_ids = np.where(ok)[0]
    grp_ids = np.array([gid[names[i]] for i in ep_ids])
    cls_ids = np.array([cmap.get(names[i], -1) for i in ep_ids])
    pooled = np.load(cache / f"pooled_{pool}.npz")
    pix = np.load(cache / "pix_grid.npz")["pix"]
    A = area[ep_ids]
    S = {}
    def mk2(feat, k0, off=0):
        return P.make_samples(feat[ep_ids], A, ep_ids, grp_ids, cls_ids, k0, off)
    S["gt"] = mk2(pooled["gt"], 0)
    S["tf1"] = mk2(pooled["tf1"], 1)
    S["ol"] = mk2(pooled["ol"], 1)
    if cfg["probe"]["layer_probe"]:
        for n in pooled.files:
            if n.startswith("h"):
                S[n] = mk2(pooled[n], 1, 1)      # frame k=t+1 predicted; label = area of input frame t
    if "rvit" in pooled.files:
        S["rvit"] = mk2(pooled["rvit"], 0)
    S["pix"] = mk2(pix, 0)
    return S, ep_ids, grp_ids, groups, ep


def _eval(preds, S, groups, cfg, seed, bins, no_cls=False, role_gids=None):
    nb = cfg["probe"]["n_bootstrap"]
    out = {}
    gids = list(range(len(groups)))
    for t, (yh, ch) in preds.items():
        s = S[t]
        if no_cls:       # leave-one-group-out: the held-out class never appears in training, accuracy is undefined
            s = P.Samples(s.x, s.ep, s.k, s.grp, np.full_like(s.cls, -1), s.y)
        base = np.isfinite(yh) & (s.k >= 1)
        out[t] = {"all": P.summarize(s, yh, ch, base, gids, nb, seed)}
        for rn, rg in (role_gids or {}).items():            # per-role summary (heldout_size protocol)
            m = base & np.isin(s.grp, rg)
            if m.any():
                out[t]["role_" + rn] = P.summarize(s, yh, ch, m, list(rg), max(nb // 2, 0), seed)
        for bn, (lo, hi) in bins.items():
            m = base & (s.k >= lo) & (s.k <= hi)
            if m.any():
                out[t]["bin_" + bn] = P.summarize(s, yh, ch, m, gids, max(nb // 2, 0), seed)
    return out


def _agg(per_seed, groups):
    """per_seed: list of {type:{subset:{metric...}}} -> mean/sd over seeds, CI bounds averaged over seeds."""
    res = {}
    for t in per_seed[0]:
        res[t] = {}
        for sub in per_seed[0][t]:
            row = {}
            for m in METRICS:
                v = np.array([r[t][sub][m] for r in per_seed if sub in r[t]], float)
                lo = np.array([r[t][sub]["ci"][m][0] for r in per_seed if sub in r[t] and r[t][sub]["ci"]], float)
                hi = np.array([r[t][sub]["ci"][m][1] for r in per_seed if sub in r[t] and r[t][sub]["ci"]], float)
                row[m] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else None, "sd": float(np.nanstd(v)) if np.isfinite(v).any() else None,
                          "ci_lo": float(np.nanmean(lo)) if len(lo) and np.isfinite(lo).any() else None,
                          "ci_hi": float(np.nanmean(hi)) if len(hi) and np.isfinite(hi).any() else None, "n_seeds": int(np.isfinite(v).sum())}
            bg = {}
            for r in per_seed:
                for g, b in r[t][sub]["bias_by_group"].items():
                    bg.setdefault(groups[int(g)], []).append(b)
            row["bias_log_area_by_group"] = {g: float(np.mean(v)) for g, v in bg.items()}
            row["n_episodes"] = per_seed[0][t][sub]["n_episodes"]
            res[t][sub] = row
    return res


def logo_heldout_type(cfg, groups):
    """For leave-one-group-out: the held-out size is 'extrap_small'/'extrap_large' if it is the smallest/largest of the 9 (training on the
    other sizes lies entirely above/below it), otherwise 'interp'. Independent of what the world model was trained on."""
    return {g: ("extrap_small" if i == 0 else "extrap_large" if i == len(groups) - 1 else "interp") for i, g in enumerate(groups)}


def run(cfg, paths, run_name):
    cache = paths["cache"] / run_name
    assert (cache / "DONE").exists(), f"run extract first: {cache}"
    pc = cfg["probe"]
    bins = {k: tuple(v) for k, v in pc["probe_step_bins"].items()}
    out = {"run_name": run_name, "seeds": pc["seeds"], "target": pc["target"], "protocol_definitions": {
        "strat_cv": "(i) same-type latent, 5-fold GroupKFold over episodes stratified by size group",
        "logo": "(i) same-type latent, leave-one-group-out",
        "gt_transfer": "(ii) train on GT encodings only (held-out episodes), apply to predicted latents",
        "upper": "(iii) train on GT + tf1 + ol latents of all sizes (held-out episodes), test on predicted latents",
        "heldout_size": "(iv) train on GT encodings of the sizes the world model was trained on (held-out episodes), test on gt/tf1/ol of all sizes, summarised per role"},
        "metrics": {"acc_cls": "exact size-class accuracy (classes = ordered sizes of the dataset; null when the class is absent from training)",
                    "acc_cls_pm1": "accuracy within +-1 neighbouring size class", "r2_within_group": "R2 after removing per-group means of y and y_hat",
                    "slope": "slope of predicted vs real log-area over all samples", "slope_group_means": "same over group means",
                    "ci": "episode-level cluster bootstrap (resampling within group), 2.5/97.5 percentiles, averaged over probe seeds"},
        "label_note": "h0..h6 use the area of the input frame; gt/tf1/ol/rvit/pix use the area of the frame the latent represents",
        "thresholds_todo": cfg["diagnose"]["thresholds"], "pools": {}}
    for pool in cfg["diagnose"]["pools"]:
        S, ep_ids, grp_ids, groups, ep = load_all(cache, cfg, pool)
        ep_grp = grp_ids
        roles = ep.get("roles") or {}
        tgs = [g for g in ep.get("train_groups", []) if g in groups]
        train_gids = [groups.index(g) for g in tgs]
        role_gids = {}
        for g in groups:
            role_gids.setdefault(roles.get(g, "unknown"), []).append(groups.index(g))
        out["train_groups"], out["group_roles"] = tgs, {g: roles.get(g) for g in groups}
        out["logo_heldout_type"] = logo_heldout_type(cfg, groups)
        out["groups_order_by_median_real_area"] = groups
        out["n_eligible_episodes"] = int(len(ep_ids))
        out["group_sizes"] = {g: int((grp_ids == i).sum()) for i, g in enumerate(groups)}
        kinds = ["ridge"] + (["mlp"] if pool in pc["models"]["mlp"]["pools"] else [])
        out["pools"][pool] = {}
        layer_types = [t for t in S if t.startswith("h")]
        for kind in kinds:
            kres = {}
            P.run_protocol.factory = {}
            for protocol in pc["protocols"]:
                if protocol in ("strat_cv", "logo"):
                    targets = ["gt", "tf1", "ol"] + ["rvit", "pix"] * 1 + layer_types
                    targets = [t for t in targets if t in S]
                elif protocol in ("gt_transfer", "heldout_size"):
                    targets = ["gt", "tf1", "ol"]
                    if protocol == "heldout_size" and not train_gids:
                        log("probe heldout_size skipped: no training groups among eligible groups")
                        continue
                else:
                    targets = ["tf1", "ol"]
                per_seed, per_seed_perm = [], []
                for seed in pc["seeds"]:
                    log(f"probe {pool}/{kind}/{protocol} seed {seed}")
                    pr = P.run_protocol(protocol, targets, S, ep_ids, ep_grp, pc, kind, pool, seed, train_gids=train_gids)
                    per_seed.append(_eval(pr, S, groups, cfg, seed, bins, no_cls=(protocol in ("logo", "heldout_size")),
                                          role_gids=role_gids if protocol == "heldout_size" else None))
                    if protocol == "strat_cv" and "label_permute" in pc["controls"]:
                        prp = P.run_protocol(protocol, [t for t in ("gt", "tf1", "ol")], S, ep_ids, ep_grp, pc, kind, pool, seed, permute=True)
                        per_seed_perm.append(_eval(prp, S, groups, cfg, seed, bins))
                kres[protocol] = _agg(per_seed, groups)
                if per_seed_perm:
                    kres["label_permute_" + protocol] = _agg(per_seed_perm, groups)
                    kres["selectivity_real_minus_permute"] = {
                        t: {m: _diff(kres[protocol][t]["all"][m], kres["label_permute_" + protocol][t]["all"][m]) for m in ("r2_overall", "r2_within_group", "acc_cls", "acc_cls_pm1")}
                        for t in kres["label_permute_" + protocol]}
            if "constant_mean" in pc["controls"]:
                per_seed = []
                for seed in pc["seeds"]:
                    pr = P.run_protocol("strat_cv", ["gt"], S, ep_ids, ep_grp, pc, "const", pool, seed)
                    per_seed.append(_eval(pr, S, groups, cfg, seed, bins))
                kres["constant_mean"] = _agg(per_seed, groups)
            out["pools"][pool][kind] = kres
        if pool == pc.get("direction_pool", "mean"):
            _save_direction(cache, paths, run_name, S, pc)
    d = paths["results_dir"] / run_name
    write_json(out, d / "probe.json")
    return out


def _diff(a, b):
    if a["mean"] is None or b["mean"] is None:
        return None
    return a["mean"] - b["mean"]


def _save_direction(cache, paths, run_name, S, pc):
    """Ridge direction of log-area in raw (un-standardised) mean-pooled GT-encoding space, fitted on all eligible episodes."""
    from sklearn.linear_model import Ridge
    s = S["gt"]
    mu, sd = s.x.mean(0), s.x.std(0) + 1e-6
    alpha = float(np.median(pc["models"]["ridge"]["alphas"]))
    m = Ridge(alpha=alpha).fit((s.x - mu) / sd, s.y)
    w_raw = m.coef_ / sd
    u = w_raw / np.linalg.norm(w_raw)
    d = paths["results_dir"] / run_name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "size_direction.npz", u=u, w_raw=w_raw, intercept=m.intercept_ - float((m.coef_ * mu / sd).sum()), alpha=alpha)
