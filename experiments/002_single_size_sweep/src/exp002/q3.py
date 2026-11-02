"""Q3 (H-c): does the predictor USE the size information? Interventions on the predictor residual stream at fixed layers
(config q3.layers 'middle' / 'last' -> concrete indices from the number of blocks, rule recorded; chosen before any data is seen).

Steering: x_j <- x_j + alpha * sigma_u * d on every token, d = unit size direction at layer j estimated from TRAINING episodes only
  (ridge direction of the pooled residual stream; mean-difference vector largest-minus-smallest training level),
  sigma_u = std of the pooled projection onto d over TRAINING samples (never evaluation data), alpha in q3.dose_alpha_sigma_units.
Subspace removal: x_j <- x_j - lam * (x_j - mu) M^T for lam in q3_impl.removal_strengths; M from INLP (ridge, q3_impl.inlp_iterations) or LEACE.
  After removal an MLP probe (trained on removed training features, tested on removed evaluation features) must fail to read size
  (q3_impl.removal_check.mlp_r2_after_max) else the method is flagged `not_removed` and no conclusion is drawn from it.
Controls: q3.n_random_directions (100) random directions/subspaces INSIDE the data subspace with the SAME projection variance and the same
  shift size (directions.matched_random_direction; not isotropic Gaussians). Streams are independent per training seed.
Metrics are ABSOLUTE changes (no ratios): delta_predicted_size = Q1-ridge readout (trained on encoder tokens of the training episodes)
  of the pooled predicted latent after minus before; delta_latent = mean over patches of the L2 change of the predicted visual latent.
Aggregation over evaluation samples = median."""
from __future__ import annotations

import numpy as np

from .common import log, write_json, require, run_dir, run_name as mk_run_name
from . import directions as D
from . import probes as PR
from . import stats as ST

D_VIS = 384


def layer_indices(cfg, L):
    """Replace 'middle'/'last' by residual-stream indices 0..L (L = number of blocks). Rule is returned for the log/results."""
    rule = {"middle": "L // 2 (input of block L//2 = output of block L//2 - 1)", "last": "L (output of the last block, input of the final norm)"}
    idx = {"middle": L // 2, "last": L}
    names = list(cfg["q3"]["layers"])
    for n in names:
        if n not in idx:
            raise ValueError("unknown q3 layer name %s" % n)
    return {n: idx[n] for n in names}, {n: rule[n] for n in names}


def _to_t(x, dev):
    import torch
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


def scan(adapter, Zev, layer, readout, ref, shifts=None, lams=None, M=None, mu=None, bs=128):
    """Run the predictor for A modifications of every evaluation window. Returns dsize (A,S), dlat (A,S) (absolute changes)."""
    import torch
    dev = adapter.dev
    A = len(shifts) if shifts is not None else len(lams)
    S = Zev.shape[0]
    dsize, dlat = np.zeros((A, S)), np.zeros((A, S))
    chunk = max(1, bs // A)
    sh = _to_t(shifts, dev) if shifts is not None else None
    lm = _to_t(lams, dev) if lams is not None else None
    Mt = _to_t(M.T, dev) if M is not None else None
    mut = _to_t(mu, dev) if mu is not None else None
    for s in range(0, S, chunk):
        z = Zev[s:s + chunk].to(dev)
        n = z.shape[0]
        zr = z.repeat(A, *([1] * (z.ndim - 1)))
        shr = sh.repeat_interleave(n, 0)[:, None, :] if sh is not None else None
        lmr = lm.repeat_interleave(n, 0)[:, None, None] if lm is not None else None

        def fn(x):
            if shr is not None:
                x = x + shr
            if lmr is not None:
                x = x - lmr * ((x - mut) @ Mt)
            return x

        with torch.no_grad():
            out = adapter.predict(zr, {layer: fn})
            vis = out[:, -1, :, :D_VIS].float()
            pooled = vis.mean(1)
            r = pooled @ readout["w"] + readout["b"]
            ds = (r.reshape(A, n) - ref["size"][s:s + n][None]).cpu().numpy()
            dl = (vis.reshape(A, n, *vis.shape[1:]) - ref["vis"][s:s + n][None]).norm(dim=-1).mean(-1).cpu().numpy()
        dsize[:, s:s + n], dlat[:, s:s + n] = ds, dl
    return dsize, dlat


def baseline(adapter, Zev, readout, bs=128):
    import torch
    vis_l, size_l = [], []
    with torch.no_grad():
        for s in range(0, Zev.shape[0], bs):
            out = adapter.predict(Zev[s:s + bs].to(adapter.dev), {})
            vis = out[:, -1, :, :D_VIS].float()
            vis_l.append(vis)
            size_l.append(vis.mean(1) @ readout["w"] + readout["b"])
    return {"vis": torch.cat(vis_l), "size": torch.cat(size_l)}


def _effect(dsize, dlat, nonzero):
    """T = median over nonzero doses of (median over samples of |change|), for both metrics."""
    return {"T_size": float(np.median(np.median(np.abs(dsize[nonzero]), axis=1))), "T_latent": float(np.median(np.median(dlat[nonzero], axis=1)))}


def episode_mean(x, epev):
    """x (A,S) per-sample values, epev (S,) episode id of each sample -> (A,) mean over episodes of the within-episode mean (확정 기록 6: alpha curve = episode mean)."""
    ids = np.unique(epev)
    return np.mean([x[:, epev == e].mean(1) for e in ids], axis=0)


def _by_dose(dsize, dlat, epev):
    """Dose-response curves, EPISODE means (not sample medians): signed and absolute change of the predicted size, latent change."""
    return {"episode_mean_signed_dsize": episode_mean(dsize, epev), "episode_mean_abs_dsize": episode_mean(np.abs(dsize), epev), "episode_mean_dlatent": episode_mean(dlat, epev)}


def run_layer(cfg, adapter, layer, tag, Ztr, ytr, eptr, Zev, yev, epev, readout, train_seed, rng_base, layer_id, log_prefix=""):
    """Everything for one residual-stream index. Ztr/Zev: CPU tensors (N,H,196,404) of windows. Returns a json-able dict."""
    qc, qi = cfg["q3"], cfg["q3_impl"]
    alphas = list(qc["dose_alpha_sigma_units"])
    nz = [i for i, a in enumerate(alphas) if a != 0]
    nrand = int(qc["n_random_directions"])
    pct = float(qc["criterion_percentile"])
    mono_min = float(qi["monotonic_min_spearman"])
    primary = qi["primary_metric"]
    pc = cfg["probe_impl"]
    if epev is None:
        raise ValueError("epev (episode id of every evaluation sample) is required: alpha curves are episode means")
    epev = np.asarray(epev)
    prim_dir, prim_rem = qi["primary_steering_direction"], qi["primary_removal_method"]
    Xtr = adapter.pooled_hidden(Ztr, layer)
    Xev = adapter.pooled_hidden(Zev, layer)
    ref = baseline(adapter, Zev, readout, qi["batch_samples"])
    res = {"layer_index": layer, "layer_tag": tag, "dim": int(Xtr.shape[1])}
    chk = scan(adapter, Zev, layer, readout, ref, shifts=np.zeros((1, Xtr.shape[1])), bs=qi["batch_samples"])
    res["identity_check_max_abs_dsize_zero_shift"] = float(np.abs(chk[0]).max())
    lam, V = D.cov_eig(Xtr)
    Cx = np.cov(Xtr.T)
    ridge = D.ridge_fit(Xtr, ytr, eptr, pc["ridge_alphas"], pc["inner_folds"], train_seed)
    lo, hi = ytr.min(), ytr.max()
    md = Xtr[ytr >= hi - 1e-9].mean(0) - Xtr[ytr <= lo + 1e-9].mean(0)
    dirs = {"ridge": ridge["u"], "meandiff": md / np.linalg.norm(md)}
    res["steering"] = {}
    for di, (dname, u) in enumerate(dirs.items()):
        sigma = float(np.std(Xtr @ u, ddof=1))
        rng = np.random.RandomState(int(np.random.SeedSequence([int(qc_seed(cfg)), int(train_seed), int(layer_id), di]).generate_state(1)[0]))
        sh = np.array([a * sigma for a in alphas])[:, None] * u[None, :]
        ds, dl = scan(adapter, Zev, layer, readout, ref, shifts=sh, bs=qi["batch_samples"])
        eff = _effect(ds, dl, nz)
        by = _by_dose(ds, dl, epev)
        rho = ST.spearman(alphas, by["episode_mean_signed_dsize"])
        rT, rTl, errs = [], [], []
        var_t = float(np.var(Xtr @ u, ddof=1))
        for _ in range(nrand):
            r, q, clipped = D.matched_random_direction(lam, V, var_t, rng)
            errs.append(abs(q - var_t) / var_t)
            s2 = np.array([alphas[i] * sigma for i in nz])[:, None] * r[None, :]
            d2, l2 = scan(adapter, Zev, layer, readout, ref, shifts=s2, bs=qi["batch_samples"])
            e2 = _effect(d2, l2, list(range(len(nz))))
            rT.append(e2["T_size"])
            rTl.append(e2["T_latent"])
        T_key, rr = ("T_size", rT) if primary == "delta_predicted_size" else ("T_latent", rTl)
        p95 = float(np.percentile(rr, pct))
        res["steering"][dname] = {
            "sigma_u_train": sigma, "alphas_sigma_units": alphas, "size_direction": {**eff, **by, "spearman_signed_dsize_vs_alpha": rho, "curve": "episode mean"},
            "random": {"T_size": rT, "T_latent": rTl, "variance_rel_err_max": float(np.max(errs)), "n": nrand},
            "tier": "confirmatory" if dname == prim_dir else "exploratory",
            "primary_metric": primary, "p_rank": float((1 + sum(1 for x in rr if x >= eff[T_key])) / (nrand + 1)),
            "percentile": pct, "random_percentile_value": p95, "outside_percentile": bool(eff[T_key] > p95),
            "monotone": bool(np.isfinite(rho) and abs(rho) >= mono_min), "monotonic_min_spearman": mono_min,
            "random_variance_match_ok": bool(np.max(errs) <= qi["variance_match_rel_tol"])}
        log("%sq3 %s layer %s steering %s done" % (log_prefix, tag, layer, dname))
    # ---- subspace removal
    res["removal"] = {}
    lams = list(qi["removal_strengths"])
    nzl = [i for i, l in enumerate(lams) if l != 0]
    tol = float(qi["removal_check"]["mlp_r2_after_max"])
    for mi, meth in enumerate(cfg["q3"]["removal_methods"]):
        if meth == "INLP":
            B, r2s = D.inlp(Xtr, ytr, eptr, int(qi["inlp_iterations"]), pc["ridge_alphas"], pc["inner_folds"], train_seed)
            M, mu = D.projector_from_basis(B)
            basis = B
        elif meth == "LEACE":
            M, mu = D.leace_fit(Xtr, ytr)
            Uu, sv, _ = np.linalg.svd(M)
            basis = Uu[:, sv > 1e-8 * max(sv.max(), 1e-300)]
            r2s = None
        else:
            raise ValueError(meth)
        k = basis.shape[1]
        Xtr_r, Xev_r = D.apply_removal(Xtr, M, mu), D.apply_removal(Xev, M, mu)
        mlp_after = [r2v for r2v in (D.r2(yev, PR.fit_predict("mlp", Xtr_r, ytr, eptr, [Xev_r], s, cfg)[0]) for s in cfg["probe"]["seeds"])]
        mlp_before = [D.r2(yev, PR.fit_predict("mlp", Xtr, ytr, eptr, [Xev], s, cfg)[0]) for s in cfg["probe"]["seeds"]]
        lin_after = D.r2(yev, D.ridge_predict(D.ridge_fit(Xtr_r, ytr, eptr, pc["ridge_alphas"], pc["inner_folds"], train_seed), Xev_r))
        removed = bool(np.mean(mlp_after) <= tol)
        ds, dl = scan(adapter, Zev, layer, readout, ref, lams=lams, M=M, mu=mu, bs=qi["batch_samples"])
        eff = _effect(ds, dl, nzl)
        by = _by_dose(ds, dl, epev)
        rho = ST.spearman(lams, by["episode_mean_abs_dsize"])
        rng = np.random.RandomState(int(np.random.SeedSequence([int(qc_seed(cfg)), int(train_seed), int(layer_id), 10 + mi]).generate_state(1)[0]))
        tv = [float(basis[:, i] @ Cx @ basis[:, i]) for i in range(k)]
        rT, rTl, errs = [], [], []
        for _ in range(nrand):
            dirs_r = np.zeros((0, Xtr.shape[1]))
            ach = []
            for i in range(k):
                r, q, _c = D.matched_random_direction(lam, V, max(tv[i], 1e-12), rng, orth=dirs_r)
                dirs_r = np.concatenate([dirs_r, r[None]], 0)
                ach.append(q)
            errs.append(abs(sum(ach) - sum(tv)) / max(sum(tv), 1e-12))
            Mr = dirs_r.T @ dirs_r
            d2, l2 = scan(adapter, Zev, layer, readout, ref, lams=lams, M=Mr, mu=np.zeros(Xtr.shape[1]), bs=qi["batch_samples"])
            e2 = _effect(d2, l2, nzl)
            rT.append(e2["T_size"])
            rTl.append(e2["T_latent"])
        T_key, rr = ("T_size", rT) if primary == "delta_predicted_size" else ("T_latent", rTl)
        res["removal"][meth] = {
            "tier": "confirmatory" if meth == prim_rem else "exploratory", "rank": int(k), "inlp_linear_r2_by_iteration": r2s, "removed": removed, "flag": None if removed else "not_removed",
            "check": {"mlp_r2_after_removal_per_probe_seed": mlp_after, "mlp_r2_before_per_probe_seed": mlp_before, "linear_ridge_r2_after": lin_after,
                      "threshold_mlp_r2_after_max": tol, "eval_set": "evaluation windows of this run (all levels)"},
            "lambdas": lams, "size_removal": {**eff, **by, "spearman_abs_dsize_vs_lambda": rho, "curve": "episode mean"},
            "random": {"T_size": rT, "T_latent": rTl, "variance_rel_err_max": float(np.max(errs)), "n": nrand},
            "primary_metric": primary, "p_rank": float((1 + sum(1 for x in rr if x >= eff[T_key])) / (nrand + 1)),
            "percentile": pct, "random_percentile_value": float(np.percentile(rr, pct)), "outside_percentile": bool(eff[T_key] > np.percentile(rr, pct)),
            "monotone": bool(np.isfinite(rho) and abs(rho) >= mono_min), "monotonic_min_spearman": mono_min,
            "random_variance_match_ok": bool(np.max(errs) <= qi["variance_match_rel_tol"])}
        log("%sq3 %s layer %s removal %s done (removed=%s)" % (log_prefix, tag, layer, meth, removed))
    return res


def qc_seed(cfg):
    return cfg["q3_impl"]["random_direction_seed"]


# ------------------------------------------------------------------------------------------------ model adapter (real model)
class ModelAdapter:
    def __init__(self, model, hooks, H, dev, dpart=404):
        self.model, self.hooks, self.H, self.dev = model, hooks, H, dev
        self.dpart = dpart

    def predict(self, z, modify):
        self.hooks.reset()
        self.hooks.modify = dict(modify)
        out = self.model.predict(z)
        self.hooks.reset()
        return out

    def pooled_hidden(self, Z, layer, bs=128):
        import torch
        from . import model_io as M
        outs = []
        with torch.no_grad():
            for s in range(0, Z.shape[0], bs):
                self.hooks.reset()
                self.hooks.capture_layers = {layer}
                self.model.predict(Z[s:s + bs].to(self.dev))
                outs.append(M.last_frame_tokens(self.hooks.store[layer], self.H).float().mean(1).cpu().numpy().astype(np.float64))
        self.hooks.reset()
        return np.concatenate(outs)


def behavior_contrast(cfg, adapter, Zev, rows_ev, readout, n_pairs, seed):
    """EXPLORATORY. Pairs of evaluation samples of the same InD level (ind) with the same action stream but different cloth context:
    the visual part (first 384 dims) of sample j is put into sample i (action dims of i kept). No size-only paired data exist, so the pairs
    differ in everything else too (episode content); limitation recorded."""
    import torch
    idx = [i for i, r in enumerate(rows_ev) if r["level"] == "ind"]
    if len(idx) < 2:
        return {"status": "not_run", "reason": "fewer than 2 'ind' evaluation samples"}
    rng = np.random.RandomState(seed)
    pairs = [tuple(rng.choice(idx, 2, replace=False)) for _ in range(n_pairs)]
    with torch.no_grad():
        zi = torch.stack([Zev[i] for i, _ in pairs]).to(adapter.dev)
        zj = torch.stack([Zev[j] for _, j in pairs]).to(adapter.dev)
        zm = zi.clone()
        zm[..., :D_VIS] = zj[..., :D_VIS]
        def rd(z):
            out = adapter.predict(z, {})
            return (out[:, -1, :, :D_VIS].float().mean(1) @ readout["w"] + readout["b"]).cpu().numpy()
        o_i, o_m = rd(zi), rd(zm)
        in_i = (zi[:, -1, :, :D_VIS].float().mean(1) @ readout["w"] + readout["b"]).cpu().numpy()
        in_m = (zm[:, -1, :, :D_VIS].float().mean(1) @ readout["w"] + readout["b"]).cpu().numpy()
    din, dout = in_m - in_i, o_m - o_i
    return {"status": "exploratory", "n_pairs": len(pairs), "mean_abs_input_readout_change": float(np.abs(din).mean()),
            "mean_abs_output_readout_change": float(np.abs(dout).mean()),
            "pearson_input_vs_output_change": float(np.corrcoef(din, dout)[0, 1]) if len(pairs) > 2 and din.std() > 0 and dout.std() > 0 else None,
            "limitation": "pairs are different InD episodes (size identical), not size-only counterfactuals; no p-value"}


def run(cfg, paths, cond, seed, dry_run=False):
    name = mk_run_name(cond, seed)
    H = require(cfg, "train.num_hist")
    qi = cfg["q3_impl"]
    log("q3 plan: run=%s layers=%s directions: ridge+meandiff, removal=%s, random=%d, alphas=%s" % (
        name, cfg["q3"]["layers"], cfg["q3"]["removal_methods"], cfg["q3"]["n_random_directions"], cfg["q3"]["dose_alpha_sigma_units"]))
    if dry_run:
        return None
    import torch
    from . import cache as CA
    from . import model_io as M
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(paths)
    rdir = run_dir(paths, name)
    model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
    if M.encoder_fingerprint(model.encoder) != C["meta"]["encoder_fingerprint"]:
        raise RuntimeError("encoder differs from the Q1 cache encoder")
    mean, std, _ = M.action_stats_for_run(cfg, rdir, tcfg)
    L = M.n_blocks(model)
    layers, rule = layer_indices(cfg, L)
    log("q3 layer rule (fixed before data): n_blocks=%d -> %s ; rule: %s" % (L, layers, rule))
    rows = C["rows"]
    tr_idx = [i for i, r in enumerate(rows) if r["part"] == "train"]
    per_lvl, ev_idx = {}, []
    for i, r in enumerate(rows):
        if r["part"] == "test" and per_lvl.get(r["level"], 0) < qi["episodes_per_level"]:
            per_lvl[r["level"]] = per_lvl.get(r["level"], 0) + 1
            ev_idx.append(i)
    frames = list(qi["frames"])
    assert min(frames) >= H - 1

    def windows_of(idx):
        out = []
        for s in range(0, len(idx), 8):
            ii = idx[s:s + 8]
            x = M.frames_to_float(np.stack([np.asarray(C["frames_u8"][i]) for i in ii])).to(dev)
            acts = M.acts_from_raw(torch.as_tensor(np.stack([np.asarray(C["actions_raw"][i]) for i in ii])), mean, std).to(dev)
            with torch.no_grad():
                out.append(M.make_windows(M.encode_batch(model, x, acts), H, frames).cpu())
        return torch.cat(out)

    Ztr, Zev = windows_of(tr_idx), windows_of(ev_idx)
    ytr = np.repeat([rows[i]["nominal_log_area"] for i in tr_idx], len(frames))
    eptr = np.repeat(np.arange(len(tr_idx)), len(frames))
    yev = np.repeat([rows[i]["nominal_log_area"] for i in ev_idx], len(frames))
    rows_ev = [rows[i] for i in ev_idx for _ in frames]
    epev = np.repeat(np.arange(len(ev_idx)), len(frames))          # windows are ordered [episode][frame] (make_windows)
    tok = C["tokens_dino"]
    Xr = np.stack([np.asarray(tok[i][frames]).astype(np.float32).mean(1) for i in tr_idx]).reshape(len(tr_idx) * len(frames), -1)
    rm = D.ridge_fit(Xr, ytr, eptr, cfg["probe_impl"]["ridge_alphas"], cfg["probe_impl"]["inner_folds"], 0)
    w = rm["w"] / rm["sd"][0]
    readout = {"w": _to_t(w, dev), "b": float(rm["b"] - (rm["mu"][0] * w).sum())}
    hooks = M.StreamHooks(model).install()
    adapter = ModelAdapter(model, hooks, H, dev)
    out = {"run": name, "epoch": ep, "n_blocks": L, "layer_rule": rule, "layers": {}, "frames": frames, "num_hist": H,
           "n_train_samples": len(ytr), "n_eval_samples": len(yev),
           "readout": "ridge on encoder tokens (pooled 384) of training episodes, primary target; absolute change of its output",
           "note": "numbers only; independent random directions per training seed"}
    for tag, j in layers.items():
        out["layers"][tag] = run_layer(cfg, adapter, j, tag, Ztr, ytr, eptr, Zev, yev, epev, readout, int(seed), 0, j, log_prefix=name + " ")
    out["behavior_contrast_exploratory"] = behavior_contrast(cfg, adapter, Zev, rows_ev, readout, qi["behavior_pairs"], int(seed))
    write_json(out, paths["results_dir"] / name / "q3.json")
    return out
