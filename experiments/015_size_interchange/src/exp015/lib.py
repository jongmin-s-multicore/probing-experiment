"""Numpy-only parts of experiment 015: per-episode level statistics, level means m_L and level-mean true motion, split-half gate, fake-swap shifts,
random-direction shifts, projections, bootstrap statistics, verdict. No torch import.

Definitions (per source level S, target level T, all on the z part, shape (P patches, dz)):
  m_L        mean over the train-split episodes of level L and the 12 input frames of the encoder latent (013 level_means rule)
  DZ_L       mean over the same episodes and the 12 transitions of the true change z_{t+1} - z_t
  d          DZ_T - DZ_S
  shift      m_T - m_S                      (real swap)       z' = z + shift on the z part of every token before the positional embedding
  moved      ((pred(z') - pred(z)) - shift) . d / |d|^2     = (Delta_hat(z') - Delta_hat(z)) . d / |d|^2 with Delta_hat(x) = pred(x) - x
  fake swap (unscaled)     per-level random half-split difference of the per-episode mean latents (report only since the 2026-11-12 user decision)
  fake swap (norm-matched) PRIMARY comparator (user decision 2026-11-12): each unscaled fake shift f[p] rescaled per patch to |m_T[p] - m_S[p]|, i.e. f[p] * |shift[p]| / |f[p]|;
             pair-specific (depends on T). Guard: a patch with |f[p]| <= NM_EPS (1e-12, float64) cannot be rescaled; its shift is set to 0 (norm NOT matched there) and
             the number of such patches is recorded in fit.json (expected 0).
  gate       mean over 20 random split-halves (RandomState(seed + 1)) of cos(d half 1, d half 2) >= 0.5 (user decision 2026-11-12); the single-split cosine (RandomState(seed)) is report only.
  copy-through diagnostic  shift . d / |d|^2: how far the added shift itself points along d. Because Delta_hat is taken relative to the EDITED input, a predictor that copies its
             input gives moved == 0 exactly; a predictor that maps the shift into its change output (pred(z') - pred(z) != shift) can give moved != 0 and, if shift and d are aligned
             (this number is large), the effect is not distinguishable from reacting to that direction. Reviewers read moved next to this number and next to the fake-swap value."""
from __future__ import annotations

import numpy as np

from exp013 import lib as L13

NT = 12
boot_indices = L13.boot_indices
random_dirs = L13.random_dirs


# ------------------------------------------------------------------------------------------------ level statistics
def episode_stats(token_fn, ids, P, dz, chunk=16):
    """Per-episode (n,P,dz) float32: zbar = mean over input frames 0..11 of z, dbar = mean over the 12 transitions of z_{t+1}-z_t. token_fn(sorted ids) -> (n, >=13, P, dz)."""
    ids = np.asarray(ids)
    zb = np.zeros((len(ids), P, dz), np.float32)
    db = np.zeros((len(ids), P, dz), np.float32)
    for a in range(0, len(ids), chunk):
        t = np.asarray(token_fn(ids[a:a + chunk]), np.float64)
        zb[a:a + chunk] = t[:, :NT].mean(1)
        db[a:a + chunk] = (t[:, 1:NT + 1] - t[:, :NT]).mean(1)
    return zb, db


def half_split(n, rng):
    perm = rng.permutation(n)
    h = n // 2
    return np.sort(perm[:h]), np.sort(perm[h:2 * h])


def halves_by_level(counts, seed):
    """One RandomState(seed) drawn level by level (config level order): disjoint halves of the train episodes of every level (gate split)."""
    rng = np.random.RandomState(int(seed))
    return [half_split(n, rng) for n in counts]


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def d_halves(db, halves, s, t):
    """d from half 1 and from half 2 of the train episodes: (DZ_T - DZ_S)[half]. db list over levels of (n,P,dz)."""
    out = []
    for k in (0, 1):
        out.append(db[t][halves[t][k]].astype(np.float64).mean(0) - db[s][halves[s][k]].astype(np.float64).mean(0))
    return out


def fake_shifts(zb, n_repeats, seed):
    """(L, R, P, dz) float32: per level (config order) and repeat, a fresh random permutation of the level's train episodes, shift = mean(half 1) - mean(half 2) of zbar (per patch).
    One RandomState(seed) drawn level by level, repeat by repeat. The shift of a source level is shared by all its targets."""
    rng = np.random.RandomState(int(seed))
    out = np.zeros((len(zb), int(n_repeats)) + zb[0].shape[1:], np.float32)
    for li, z in enumerate(zb):
        for r in range(int(n_repeats)):
            h1, h2 = half_split(len(z), rng)
            out[li, r] = z[h1].astype(np.float64).mean(0) - z[h2].astype(np.float64).mean(0)
    return out


NM_EPS = 1e-12


def norm_match(fake, shift):
    """Norm-matched fake swap (user decision 2026-11-12). fake (R,P,dz) unscaled fake shifts of the source level, shift (P,dz) = m_T - m_S of the pair.
    Returns (out (R,P,dz) float64, n_guarded): out[r,p] = fake[r,p] * |shift[p]| / |fake[r,p]|, so |out[r,p]| == |shift[p]| for every patch. Guard: |fake[r,p]| <= NM_EPS -> out[r,p] = 0
    (cannot be rescaled); n_guarded counts those (r,p). If |shift[p]| == 0 the patch shift is 0 (exactly matched)."""
    f = np.asarray(fake, np.float64)
    tgt = np.linalg.norm(np.asarray(shift, np.float64), axis=-1)[None, :]          # (1,P)
    nf = np.linalg.norm(f, axis=-1)                                                # (R,P)
    ok = nf > NM_EPS
    scale = np.where(ok, tgt / np.where(ok, nf, 1.0), 0.0)
    return f * scale[..., None], int((~ok).sum())


def frob(x):
    return float(np.linalg.norm(np.asarray(x, np.float64).ravel()))


def patch_norm_mean(x):
    return float(np.linalg.norm(np.asarray(x, np.float64), axis=-1).mean())


def proj(x, d):
    d = np.asarray(d, np.float64)
    return float((np.asarray(x, np.float64) * d).sum() / (d * d).sum())


def rand_shift_np(norms, r):
    """Random-direction shift (P,dz): per-patch norm = norms[p], the same unit direction r (dz,) at every patch (CHANGES.md)."""
    return np.asarray(norms, np.float64)[:, None] * np.asarray(r, np.float64)[None, :]


def compute_fit_arrays(cfg, levels, zb, db):
    """Pure computation (no I/O). zb, db: lists over `levels` of per-episode (n,P,dz) stats. Returns dict with M, DZ, fake, halves and the per-directed-pair records."""
    L = len(levels)
    counts = [len(z) for z in zb]
    M = np.stack([z.astype(np.float64).mean(0) for z in zb])
    DZ = np.stack([d.astype(np.float64).mean(0) for d in db])
    fcfg = cfg["controls"]["fake_swap"]
    fake = fake_shifts(zb, fcfg["n_repeats"], fcfg["seed"])                          # unscaled (report only); the norm-matched version is pair specific
    halves = halves_by_level(counts, cfg["seed"])                                    # single split (report-only diagnostic)
    gcfg = cfg["gate"]["d_reliability"]
    gmin = float(gcfg["min"])
    n_splits = int(gcfg.get("n_splits", 20))
    rng_x = np.random.RandomState(int(cfg["seed"]) + 1)
    gate_halves = [[half_split(n, rng_x) for n in counts] for _ in range(n_splits)]     # GATE: mean cosine over these splits (user decision 2026-11-12)
    recs = []
    for si, s in enumerate(levels):
        for ti, t in enumerate(levels):
            if si == ti:
                continue
            d = DZ[ti] - DZ[si]
            sh = M[ti] - M[si]
            d1, d2 = d_halves(db, halves, si, ti)
            c_single = cosine(d1, d2)
            cs_all = [cosine(*d_halves(db, hh, si, ti)) for hh in gate_halves]
            cs = float(np.mean(cs_all))                                              # nan if any split is nan -> not gated
            fk = fake[si].astype(np.float64)
            fnm, n_guard = norm_match(fk, sh)
            ct_fake = [proj(f, d) for f in fk]
            ct_nm = [proj(f, d) for f in fnm]
            recs.append({
                "pair": "%s>%s" % (s, t), "source": s, "target": t, "d_frob": frob(d), "d_patch_norm_mean": patch_norm_mean(d),
                "d_split_half_cosine_mean20": cs, "d_split_half_cosine_sd20": float(np.std(cs_all)), "d_split_half_cosine_min20": float(np.min(cs_all)), "n_gate_splits": n_splits,
                "d_split_half_cosine_single_split": c_single,                         # report only (RandomState(seed))
                "gate_min": gmin, "gated": bool(np.isfinite(cs) and cs >= gmin),
                "shift_frob": frob(sh), "shift_patch_norm_mean": patch_norm_mean(sh),
                "fake_unscaled_shift_frob_mean": float(np.mean([frob(f) for f in fk])), "fake_unscaled_shift_patch_norm_mean": float(np.mean([patch_norm_mean(f) for f in fk])),
                "fake_unscaled_over_real_frob": float(np.mean([frob(f) for f in fk]) / frob(sh)) if frob(sh) > 0 else None,
                "fake_normmatched_shift_frob_mean": float(np.mean([frob(f) for f in fnm])), "fake_normmatched_shift_patch_norm_mean": float(np.mean([patch_norm_mean(f) for f in fnm])),
                "fake_normmatched_n_guarded_patches": n_guard,
                "copy_through_real": proj(sh, d),
                "copy_through_fake_unscaled_mean": float(np.mean(ct_fake)), "copy_through_fake_unscaled_sd": float(np.std(ct_fake)),
                "copy_through_fake_normmatched_mean": float(np.mean(ct_nm)), "copy_through_fake_normmatched_sd": float(np.std(ct_nm)),
                "n_train_episodes_source": counts[si], "n_train_episodes_target": counts[ti]})
    return {"M": M, "DZ": DZ, "fake": fake, "halves": halves, "counts": counts, "pairs": recs}


# ------------------------------------------------------------------------------------------------ bootstrap
def ci_of(boot, ci):
    lo, hi = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


def stat_with_ci(x_by_pair, offsets, ix, ci, gated=None):
    """x_by_pair {pair: (n_S,) per-episode values} ; offsets {pair: (start, n)} into the pooled episode index of ix (B, n_total); resampling is stratified by source level so each
    pair's resample stays inside its slice. Returns {pair: {point, ci_lo, ci_hi}} and, if gated is given, the same for the mean over the gated pairs ('gated_mean')."""
    out, pts, bts = {}, [], []
    for p, x in x_by_pair.items():
        a, n = offsets[p]
        x = np.asarray(x, np.float64)
        bt = x[ix[:, a:a + n] - a].mean(1)
        lo, hi = ci_of(bt, ci)
        out[p] = {"point": float(x.mean()), "ci_lo": lo, "ci_hi": hi}
        if gated is not None and p in gated:
            pts.append(float(x.mean()))
            bts.append(bt)
    if gated is not None:
        if pts:
            bt = np.mean(bts, 0)
            lo, hi = ci_of(bt, ci)
            out["gated_mean"] = {"point": float(np.mean(pts)), "ci_lo": lo, "ci_hi": hi, "n_pairs": len(pts)}
        else:
            out["gated_mean"] = None
    return out


# ------------------------------------------------------------------------------------------------ verdict
MET, NOT_MET, NOT_EVAL = "criterion_met", "criterion_not_met", "not_evaluable"


def _ok(s, gated):
    return (isinstance(s, dict) and isinstance(s.get("metric"), dict) and all(np.isfinite(s["metric"].get(k, np.nan)) for k in ("point", "ci_lo", "ci_hi"))
            and all(p in s.get("pairs", {}) and np.isfinite(s["pairs"][p]) for p in gated))


def judge(stats_b, stats_a, gated, succ, seeds):
    """Confirmed rule (hypotheses.md 015, 2026-11-12). stats_x[train_seed] = {'metric': {point, ci_lo, ci_hi} of mean over gated pairs of (moved - NORM-MATCHED fake swap mean; user decision 2026-11-12),
    'pairs': {directed pair: seed point of moved - norm-matched fake swap mean}} or None.
      1. point > 0 and ci_lo > 0 in >= min_seeds of the (b) seeds;  2. (b) seed mean >= min_moved;  3. (b) seed mean > (a) seed mean;
      4. every gated directed pair has (b) seed-mean (moved - norm-matched fake) > 0 (both directions S->T and T->S of every gated pair are in `gated`).
    No gated pair, a missing / non-finite input, or a gated pair whose reverse direction is not in `gated` -> not_evaluable (nothing substituted)."""
    gated = list(gated)
    if len(gated) < 1:
        return {"status": NOT_EVAL, "reason": "no pair passed the d-reliability gate"}
    miss_b = [ts for ts in seeds if not _ok(stats_b.get(ts), gated)]
    miss_a = [ts for ts in seeds if not _ok(stats_a.get(ts), gated)]
    if miss_b or (succ.get("require_b_greater_than_a", True) and miss_a):
        return {"status": NOT_EVAL, "reason": "missing / non-finite input", "missing_b_seeds": miss_b, "missing_a_seeds": miss_a}
    rev_missing = []
    for p in gated:
        s, t = p.split(">")
        if "%s>%s" % (t, s) not in gated:
            rev_missing.append(p)
    if succ.get("require_reverse_sign", True) and rev_missing:
        return {"status": NOT_EVAL, "reason": "reverse direction not available for gated pair(s)", "pairs_without_reverse": rev_missing}
    per = {ts: {"metric": stats_b[ts]["metric"]["point"], "ci_lo": stats_b[ts]["metric"]["ci_lo"], "ci_hi": stats_b[ts]["metric"]["ci_hi"],
                "positive_and_ci_excludes_0": bool(stats_b[ts]["metric"]["point"] > 0 and stats_b[ts]["metric"]["ci_lo"] > 0)} for ts in seeds}
    n_ok = sum(p["positive_and_ci_excludes_0"] for p in per.values())
    mb = float(np.mean([p["metric"] for p in per.values()]))
    c1 = n_ok >= int(succ["min_seeds"])
    c2 = bool(mb >= float(succ["min_moved"]))
    ma = float(np.mean([stats_a[ts]["metric"]["point"] for ts in seeds])) if not miss_a else None
    c3 = True if not succ.get("require_b_greater_than_a", True) else bool(mb > ma)
    pair_means = {p: float(np.mean([stats_b[ts]["pairs"][p] for ts in seeds])) for p in gated}
    if succ.get("require_reverse_sign", True):
        c4 = bool(all(v > 0 for v in pair_means.values()))
    else:
        c4 = True
    met = c1 and c2 and c3 and c4
    return {"status": MET if met else NOT_MET,
            "reading": ("검출: 교체한 레벨 쪽으로 예측 움직임 이동" if met else "검출 안 됨 (크기를 사용하지 않는다는 뜻이 아님; 평균만 바뀌는 교체의 한계)"),
            "cond1_seeds_positive_ci_excl_0": {"n": int(n_ok), "needed": int(succ["min_seeds"]), "met": bool(c1)},
            "cond2_seed_mean_b_ge_min_moved": {"seed_mean_b": mb, "threshold": float(succ["min_moved"]), "met": c2},
            "cond3_b_greater_than_a": {"seed_mean_b": mb, "seed_mean_a": ma, "met": c3},
            "cond4_every_gated_pair_both_directions_positive_in_b": {"seed_mean_by_pair": pair_means, "met": c4},
            "gated_pairs": gated, "per_seed_b": {str(k): v for k, v in per.items()}}
