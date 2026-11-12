"""CPU self-test of experiment 015 on SYNTHETIC data and a toy world model (009's toy builders, 013's planted-token helpers). Numbers printed here are NOT results.
A verdict rule (known answers, boundaries)    B level statistics / gate (20-split mean) / fake swap (unscaled + norm-matched) / random shifts (independent numpy, planted worlds)
C fit/eval separation (poison tests)          D moved definition vs an independent numpy computation, copy predictor, planted oracle, edit placement
E bootstrap + aggregate end to end (planted use, no use, equal use, missing unit, no gated pair, partial gate)
F toy world model (compute_unit shapes, determinism, batch invariance)    G config / attempts / preflight / measure refusal at 0 gated pairs / notebook guard
H verdict metric = moved - NORM-MATCHED fake swap (planted arrays, unscaled value report only)."""
from __future__ import annotations

import copy
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

from . import base as B
from . import core as K
from . import lib as LB
from . import stages as S
from exp013.selftest import planted_tokens, oracle_fn, copy_fn, build_z_all, toy_rows, toy_dims, patched

REC = {"pass": [], "fail": [], "skipped": []}


def check(name, cond, detail=""):
    (REC["pass"] if cond else REC["fail"]).append(name)
    print("  [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))


def _raises(f, exc=Exception):
    try:
        f()
    except exc:
        return True
    return False


LV = ["a060", "ind", "a160"]
PAIRS = [B.pair_name(s, t) for s in LV for t in LV if s != t]


# ------------------------------------------------------------------------------------------------ A
def mkst(metric, lo, pairs):
    return {"metric": {"point": metric, "ci_lo": lo, "ci_hi": metric + 1.0}, "pairs": dict(pairs)}


def t_verdict(cfg):
    print("A  verdict rule (confirmed 2026-11-12)")
    sc = cfg["success"]
    seeds = [0, 1, 2]
    pos = {p: 0.2 for p in PAIRS}
    J = lambda b, a, g=PAIRS: LB.judge(dict(zip(seeds, b)), dict(zip(seeds, a)), g, sc, seeds)
    good = [mkst(0.2, 0.05, pos)] * 3
    a0 = [mkst(0.01, -0.02, {p: 0.01 for p in PAIRS})] * 3
    check("config: min_seeds 2, min_moved 0.10, b>a, reverse sign required", (sc["min_seeds"], sc["min_moved"], sc["require_b_greater_than_a"], sc["require_reverse_sign"]) == (2, 0.10, True, True))
    check("all four conditions met -> criterion_met", J(good, a0)["status"] == LB.MET)
    check("2 of 3 seeds with CI lower > 0 -> met", J([good[0], good[1], mkst(0.2, -0.01, pos)], a0)["status"] == LB.MET)
    check("1 of 3 seeds -> not met", J([good[0], mkst(0.2, -0.01, pos), mkst(0.2, -0.01, pos)], a0)["status"] == LB.NOT_MET)
    check("CI lower exactly 0 does not count", J([mkst(0.2, 0.0, pos)] * 3, a0)["cond1_seeds_positive_ci_excl_0"]["n"] == 0)
    check("seed mean exactly 0.10 counts (>=)", J([mkst(0.10, 0.02, {p: 0.1 for p in PAIRS})] * 3, a0)["cond2_seed_mean_b_ge_min_moved"]["met"] is True)
    check("seed mean 0.0999 does not count", J([mkst(0.0999, 0.02, {p: 0.1 for p in PAIRS})] * 3, a0)["status"] == LB.NOT_MET)
    check("(b) equal to (a) -> cond3 false (strict >)", J(good, good)["cond3_b_greater_than_a"]["met"] is False)
    bad_p = dict(pos)
    bad_p["ind>a160"] = -0.01
    check("one gated pair with non-positive (b) seed mean -> cond4 false -> not met even if the mean is large", J([mkst(0.2, 0.05, bad_p)] * 3, a0)["status"] == LB.NOT_MET and J([mkst(0.2, 0.05, bad_p)] * 3, a0)["cond4_every_gated_pair_both_directions_positive_in_b"]["met"] is False)
    zp = dict(pos)
    zp["ind>a160"] = 0.0
    check("pair seed mean exactly 0 is not > 0", J([mkst(0.2, 0.05, zp)] * 3, a0)["cond4_every_gated_pair_both_directions_positive_in_b"]["met"] is False)
    g2 = ["a060>ind", "ind>a060"]
    check("only gated pairs enter cond4 (a negative value in a non-gated pair is ignored)", J([mkst(0.2, 0.05, bad_p)] * 3, a0, g2)["status"] == LB.MET)
    check("no gated pair -> not_evaluable", J(good, a0, [])["status"] == LB.NOT_EVAL)
    check("gated pair without its reverse -> not_evaluable", J(good, a0, ["a060>ind"])["status"] == LB.NOT_EVAL)
    check("missing (b) seed -> not_evaluable", J([good[0], good[1], None], a0)["status"] == LB.NOT_EVAL)
    check("NaN in (b) -> not_evaluable", J([good[0], good[1], mkst(float("nan"), float("nan"), pos)], a0)["status"] == LB.NOT_EVAL)
    check("NaN pair value in (b) -> not_evaluable", J([good[0], good[1], mkst(0.2, 0.05, dict(pos, **{"ind>a160": float("nan")}))], a0)["status"] == LB.NOT_EVAL)
    check("missing (a) seed -> not_evaluable", J(good, [a0[0], a0[1], None])["status"] == LB.NOT_EVAL)
    check("not-detected reading says it does NOT mean 'unused'", "사용하지 않는다는 뜻이 아님" in J([mkst(0.2, -0.01, pos)] * 3, a0)["reading"])


# ------------------------------------------------------------------------------------------------ planted worlds
def planted2(rows, P, dz, motion, seed=0):
    """tokens (E,13,P,dz): z_t = base + 2 s u + noise + t * motion[level] v ; motion {level: scale}. Same level offsets as 013's planted world."""
    rng = np.random.RandomState(seed)
    s = np.array([r["nominal_log_area"] for r in rows])
    u = rng.randn(dz)
    u /= np.linalg.norm(u)
    v = rng.randn(dz)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)
    base = rng.randn(P, dz) * 0.5
    E = len(rows)
    mo = np.array([motion.get(r["level"], 0.3) for r in rows])
    T = np.zeros((E, 13, P, dz))
    T[:, 0] = base[None] + 2.0 * s[:, None, None] * u[None, None] + 0.1 * rng.randn(E, P, dz)
    for t in range(12):
        T[:, t + 1] = T[:, t] + mo[:, None, None] * v[None, None] + 0.02 * rng.randn(E, P, dz)
    return T.astype(np.float32)


def fit_of(cfg, c2, rows, tokens, seed_override=None):
    c = copy.deepcopy(cfg)
    if seed_override is not None:
        c["seed"] = seed_override
    rows_, Etr, _, _ = S.S13.rows_info(c2, {"rows": rows})
    return S.compute_fit(c, c2, rows, Etr, lambda ids: tokens[np.sort(ids)], tokens.shape[2], tokens.shape[3]), Etr


# ------------------------------------------------------------------------------------------------ B
def t_numpy(cfg, cfg2):
    print("B  level statistics, gate, fake swap, random shifts")
    c2, rows = toy_rows(cfg2)
    P, dz, _ = toy_dims()
    tokens, info = planted_tokens(rows, P, dz)
    F, Etr = fit_of(cfg, c2, rows, tokens)
    lv = np.array([r["level"] for r in rows])
    ok_m, ok_d = True, True
    for i, l in enumerate(LV):
        ids = np.where((lv == l) & (np.arange(len(rows)) < Etr))[0]
        t = tokens[ids].astype(np.float64)
        ok_m &= np.allclose(F["M"][i], t[:, :12].mean((0, 1)), atol=1e-5)
        ok_d &= np.allclose(F["DZ"][i], (t[:, 1:13] - t[:, :12]).mean((0, 1)), atol=1e-5)
    check("m_L == mean over train episodes and frames 0..11; DZ_L == mean true change (independent numpy)", ok_m and ok_d)
    from exp013 import lib as L13
    ids_by = {l: np.where((lv == l) & (np.arange(len(rows)) < Etr))[0] for l in LV}
    M13, _ = L13.level_means(lambda ids: tokens[np.sort(ids)], ids_by, LV, P, dz)
    check("m_L equals the 013 level_means function on the same episodes (reused rule)", np.allclose(F["M"], M13, atol=1e-5))
    check("train episodes per level == 16 (train rows only)", F["counts"] == [16, 16, 16])
    pr = {p["pair"]: p for p in F["pairs"]}
    check("6 directed pairs; |d| of the reverse direction equal; copy-through of reverse equal (both flip sign)", sorted(pr) == sorted(PAIRS) and abs(pr["a060>ind"]["d_frob"] - pr["ind>a060"]["d_frob"]) < 1e-9
          and abs(pr["a060>ind"]["copy_through_real"] - pr["ind>a060"]["copy_through_real"]) < 1e-6)
    check("gate cosine identical for S>T and T>S (bit-equal)", all(pr[B.pair_name(s, t)]["d_split_half_cosine_mean20"] == pr[B.pair_name(t, s)]["d_split_half_cosine_mean20"] for s in LV for t in LV if s != t))
    check("planted level-dependent motion: all 6 pairs pass the gate (cosine >= 0.5)", all(p["gated"] for p in F["pairs"]), str([round(p["d_split_half_cosine_mean20"], 3) for p in F["pairs"]]))
    tok_same = planted2(rows, P, dz, {l: 0.3 for l in LV})
    Fs, _ = fit_of(cfg, c2, rows, tok_same)
    check("identical motion in every level: d is pure noise, cosine low and pairs fail the gate", all((not p["gated"]) and p["d_split_half_cosine_mean20"] < 0.5 for p in Fs["pairs"]), str([round(p["d_split_half_cosine_mean20"], 3) for p in Fs["pairs"]]))
    tok_one = planted2(rows, P, dz, {"a060": 0.0, "ind": 0.3, "a160": 0.3})
    F1, _ = fit_of(cfg, c2, rows, tok_one)
    p1 = {p["pair"]: p for p in F1["pairs"]}
    check("a060 differs, ind == a160 in motion: pairs with a060 gated, ind<->a160 not gated", p1["a060>ind"]["gated"] and p1["ind>a160"]["gated"] is False and p1["a160>ind"]["gated"] is False)
    F2, _ = fit_of(cfg, c2, rows, tokens)
    check("everything reproducible (gate halves, fake swaps) for the same seeds", np.array_equal(F["fake"], F2["fake"]) and [p["d_split_half_cosine_mean20"] for p in F["pairs"]] == [p["d_split_half_cosine_mean20"] for p in F2["pairs"]])
    F3, _ = fit_of(cfg, c2, rows, tokens, seed_override=7)
    check("gate split depends on config seed; fake swaps do not (seed 15000)", [p["d_split_half_cosine_mean20"] for p in F["pairs"]] != [p["d_split_half_cosine_mean20"] for p in F3["pairs"]] and np.array_equal(F["fake"], F3["fake"]))
    fk = LB.fake_shifts([tokens[np.where((lv == l) & (np.arange(len(rows)) < Etr))[0]].mean(1) for l in LV], 20, 15001)
    check("fake swap seed matters", not np.array_equal(fk, F["fake"][:, :, :, :]) )
    check("fake-swap shift array shape (3 levels, 20 repeats, P, dz)", F["fake"].shape == (3, 20, P, dz))
    # fake shift reproduces independent computation for repeat 0 of the first level
    z0 = tokens[ids_by["a060"]].astype(np.float64)[:, :12].mean(1)
    perm = np.random.RandomState(15000).permutation(16)
    ref = z0[np.sort(perm[:8])].mean(0) - z0[np.sort(perm[8:16])].mean(0)
    check("fake swap repeat 0 of a060 == mean(half 1) - mean(half 2) of the per-episode means (independent numpy)", np.allclose(F["fake"][0, 0], ref, atol=1e-5))
    check("planted: fake-swap shift is much smaller than the real swap shift (recorded as fake_unscaled_over_real_frob < 0.5)", all(p["fake_unscaled_over_real_frob"] < 0.5 for p in F["pairs"]), str([round(p["fake_unscaled_over_real_frob"], 3) for p in F["pairs"]]))
    # random shifts: per-patch norm matched
    import torch
    sh, specs = K.build_shifts(F["M"], F["fake"], 0, [1, 2], LB.random_dirs(20, dz, 13000), torch.device("cpu"))
    real = F["M"][1] - F["M"][0]
    okn = all(np.allclose(sh["rand:1:%d" % k].numpy().__pow__(2).sum(-1) ** 0.5, np.linalg.norm(real, axis=-1), atol=1e-4) for k in range(20))
    check("random-direction shifts: per-patch norm == |m_T - m_S| per patch for all 20 directions", okn)
    check("shifts per source: 2 real + 20 unscaled fake + 2 x 20 norm-matched fake + 2 x 20 random = 102 (+1 unedited = 103 forwards per sample); specs 2 x (1 + 20 + 20 + 20)",
          len(sh) == 2 + 20 + 40 + 40 and len(sh) + 1 == 103 and len(specs) == 2 * (1 + 20 + 20 + 20))
    # norm-matched fake swap (user decision 2026-11-12, primary comparator)
    ok_nm, ok_dir = True, True
    for t in (1, 2):
        shr = F["M"][t] - F["M"][0]
        for r in range(20):
            nm = sh["fakenm:%d:%d" % (t, r)].numpy().astype(np.float64)
            ok_nm &= np.allclose(np.linalg.norm(nm, axis=-1), np.linalg.norm(shr, axis=-1), rtol=1e-4, atol=1e-6)
            fu = F["fake"][0, r].astype(np.float64)
            cosp = (nm * fu).sum(-1) / (np.linalg.norm(nm, axis=-1) * np.linalg.norm(fu, axis=-1))
            ok_dir &= np.allclose(cosp, 1.0, atol=1e-5)
    check("norm-matched fake swap: per-patch norm == |m_T[p] - m_S[p]| for both targets and all 20 repeats", ok_nm)
    check("norm-matched fake swap keeps the direction of the unscaled fake shift in every patch (cosine 1); pair specific (target 1 != target 2)", ok_dir and not np.allclose(sh["fakenm:1:0"].numpy(), sh["fakenm:2:0"].numpy()))
    check("unscaled fake swap in the shifts dict is the fit's fake shift, unchanged (same RandomState(15000) draws)", all(np.array_equal(sh["fakeu:%d" % r].numpy(), F["fake"][0, r]) for r in range(20)))
    zf = np.zeros((1, 3, 4)); zf[0, 1] = 1.0
    nmz, ng = LB.norm_match(zf, np.ones((3, 4)) * 2.0)
    check("norm-matched guard: a patch whose fake shift has norm <= 1e-12 is set to 0 and counted; other patches get exactly the target norm", ng == 2 and np.all(nmz[0, 0] == 0) and np.all(nmz[0, 2] == 0) and abs(np.linalg.norm(nmz[0, 1]) - 4.0) < 1e-12)
    nmz2, ng2 = LB.norm_match(np.ones((1, 2, 4)), np.zeros((2, 4)))
    check("norm-matched: target norm 0 gives a 0 shift without error", ng2 == 0 and np.all(nmz2 == 0))
    check("fit records norm-matched diagnostics: shift norm equals the real shift norm, 0 guarded patches, copy-through of norm-matched fake", all(abs(p["fake_normmatched_shift_frob_mean"] - p["shift_frob"]) < 1e-6 * max(1, p["shift_frob"]) and p["fake_normmatched_n_guarded_patches"] == 0
          and np.isfinite(p["copy_through_fake_normmatched_mean"]) and np.isfinite(p["copy_through_fake_unscaled_mean"]) for p in F["pairs"]))
    # gate = mean split-half cosine over 20 random splits (user decision 2026-11-12)
    rng_x = np.random.RandomState(int(cfg["seed"]) + 1)
    db = [(lambda t: (t[:, 1:13] - t[:, :12]).mean(1))(tokens[ids_by[l]].astype(np.float64)) for l in LV]
    splits = [[LB.half_split(16, rng_x) for _ in LV] for _ in range(20)]
    ref_cos = {}
    for si, s_ in enumerate(LV):
        for ti, t_ in enumerate(LV):
            if si != ti:
                cs_ = []
                for hh in splits:
                    d1 = db[ti][hh[ti][0]].mean(0) - db[si][hh[si][0]].mean(0)
                    d2 = db[ti][hh[ti][1]].mean(0) - db[si][hh[si][1]].mean(0)
                    cs_.append(float(d1.ravel() @ d2.ravel() / (np.linalg.norm(d1) * np.linalg.norm(d2))))
                ref_cos[B.pair_name(s_, t_)] = float(np.mean(cs_))
    check("gate value == mean over 20 RandomState(seed + 1) splits of the split-half cosine (independent numpy)", all(abs(pr[k]["d_split_half_cosine_mean20"] - v) < 1e-9 for k, v in ref_cos.items()) and all(pr[k]["n_gate_splits"] == 20 for k in pr))
    check("single-split cosine (RandomState(seed)) kept as report-only field and differs from the 20-split mean", all("d_split_half_cosine_single_split" in p for p in F["pairs"]) and any(abs(p["d_split_half_cosine_single_split"] - p["d_split_half_cosine_mean20"]) > 1e-6 for p in F["pairs"]))
    pk = F["pairs"][0]
    lo, hi = sorted([pk["d_split_half_cosine_single_split"], pk["d_split_half_cosine_mean20"]])
    cg = copy.deepcopy(cfg)
    cg["gate"]["d_reliability"]["min"] = (lo + hi) / 2
    Fg, _ = fit_of(cg, c2, rows, tokens)
    pg = Fg["pairs"][0]
    check("gate decision follows the 20-split mean, not the single split (threshold set between the two values)", pg["gated"] == (pk["d_split_half_cosine_mean20"] >= cg["gate"]["d_reliability"]["min"]) and pg["gated"] != (pk["d_split_half_cosine_single_split"] >= cg["gate"]["d_reliability"]["min"]))
    R = LB.random_dirs(20, dz, 13000)
    check("random directions: unit, reproducible with seed 13000, other seed differs", np.allclose(np.linalg.norm(R, axis=1), 1) and np.array_equal(R, LB.random_dirs(20, dz, 13000)) and not np.array_equal(R, LB.random_dirs(20, dz, 13001)))
    check("copy-through diagnostic == shift . d / |d|^2 (independent)", abs(pr["a060>ind"]["copy_through_real"] - float(((F["M"][1] - F["M"][0]) * (F["DZ"][1] - F["DZ"][0])).sum() / ((F["DZ"][1] - F["DZ"][0]) ** 2).sum())) < 1e-9)


# ------------------------------------------------------------------------------------------------ C
def t_separation(cfg, cfg2):
    print("C  fit / evaluation separation")
    c2, rows = toy_rows(cfg2)
    P, dz, _ = toy_dims()
    tokens, _ = planted_tokens(rows, P, dz)
    rows_, Etr, _, _ = S.S13.rows_info(c2, {"rows": rows})
    seen = []

    def tok_fn(ids):
        seen.extend(np.asarray(ids).tolist())
        return tokens[np.sort(ids)]
    F = S.compute_fit(cfg, c2, rows, Etr, tok_fn, P, dz)
    check("fit reads only train-split rows", len(seen) > 0 and max(seen) < Etr, "(%d rows)" % len(set(seen)))
    p = tokens.copy()
    p[Etr:] = 1e3
    Fp = S.compute_fit(cfg, c2, rows, Etr, lambda ids: p[np.sort(ids)], P, dz)
    check("poison test: garbage in every test row changes neither m_L, DZ, fake swaps nor the gate", np.array_equal(F["M"], Fp["M"]) and np.array_equal(F["DZ"], Fp["DZ"]) and np.array_equal(F["fake"], Fp["fake"]) and F["pairs"] == Fp["pairs"])
    sel = S.select_source_episodes(rows[Etr:], LV, None)
    te_lv = np.array([r["level"] for r in rows[Etr:]])
    check("evaluation episodes = test rows of the source level (10 each in the toy), disjoint from train rows", all((te_lv[sel[l]] == l).all() and len(sel[l]) == 10 for l in LV))
    sel2 = S.select_source_episodes(rows[Etr:], LV, 3)
    check("--episodes-per-level 3 keeps the first 3 by rank_in_level of each source level", all(len(sel2[l]) == 3 and [int(rows[Etr:][j]["rank_in_level"]) for j in sel2[l]] == [0, 1, 2] for l in LV))


# ------------------------------------------------------------------------------------------------ D
def t_moved(cfg, cfg2):
    print("D  moved vs independent numpy, copy predictor, planted oracle, edit placement")
    import torch
    c2, rows = toy_rows(cfg2)
    P, dz, nact = toy_dims()
    tokens, info = planted_tokens(rows, P, dz)
    F, Etr = fit_of(cfg, c2, rows, tokens)
    dev = torch.device("cpu")
    D = dz + nact
    rng = np.random.RandomState(3)
    Wm = rng.randn(D, D).astype(np.float32) * 0.3
    Wt = torch.as_tensor(Wm)
    lin = lambda zin: zin @ Wt
    rdirs = LB.random_dirs(20, dz, 13000)
    si, tis = 1, [0, 2]
    shifts, specs = K.build_shifts(F["M"], F["fake"], si, tis, rdirs, dev)
    dvec = {t: (torch.as_tensor(F["DZ"][t] - F["DZ"][si], dtype=torch.float32), float(((F["DZ"][t] - F["DZ"][si]) ** 2).sum())) for t in tis}
    ids = Etr + np.arange(5)
    z_all = build_z_all(tokens, ids, nact, np.random.RandomState(1))
    out = K.eval_source_batch(z_all, {"m": lin}, shifts, specs, dvec)["m"]
    za = z_all.numpy().astype(np.float64)
    bad = []
    for name, key, t in specs[:6] + specs[-6:] + [s for s in specs if s[0].startswith("fakeu:7") or s[0].startswith("fakenm:7")]:
        sh = shifts[key].numpy().astype(np.float64)
        d = F["DZ"][t] - F["DZ"][si]
        for b in range(5):
            tot = 0.0
            for f in range(12):
                zf = za[b, f]
                p0 = (zf @ Wm.astype(np.float64))[:, :dz]
                z2 = zf.copy()
                z2[:, :dz] += sh
                p1 = (z2 @ Wm.astype(np.float64))[:, :dz]
                tot += (((p1 - p0) - sh) * d).sum() / (d * d).sum()
            bad.append(abs(out[name][b] - tot / 12))
    check("eval_source_batch == independent numpy for real / fake / random shifts and both targets", max(bad) < 2e-4, "(max abs diff %.2g)" % max(bad))
    o_copy = K.eval_source_batch(z_all, {"c": lambda z: z.clone()}, shifts, specs, dvec)["c"]
    check("copy predictor (pred = input): moved == 0 for every shift (Delta_hat is relative to the edited input) and response norm 0", all(np.abs(v).max() < 1e-4 for v in o_copy.values()))
    # edit placement: capture the predictor inputs
    cap = []

    def spy(zin):
        cap.append(zin.clone())
        return zin.clone()
    K.eval_source_batch(z_all, {"s": spy}, {"real:0": shifts["real:0"]}, [("real:0", "real:0", 0)], {0: dvec[0]})
    base_in, edit_in = cap[0], cap[1]
    check("edit: only the z part (first dz coords) of every token changes, by exactly the shift; action/proprio coordinates untouched; the unedited call comes first",
          torch.equal(base_in[..., dz:], edit_in[..., dz:]) and torch.allclose(edit_in[..., :dz] - base_in[..., :dz], shifts["real:0"].expand_as(base_in[..., :dz]), atol=1e-6) and base_in.shape[0] == 5 * 12 and base_in.shape[2] == P)
    check("input transitions are frames 0..11 of the real latents", torch.equal(base_in[:, 0], z_all[:, :12].reshape(60, P, D)))
    # planted oracle: reads the size from the patch mean -> moves to the target's motion
    fo = {"o": oracle_fn(info, dz)}
    full = {t: dvec[t] for t in tis}
    sel_ids = Etr + np.arange(5)
    o = K.eval_source_batch(build_z_all(tokens, sel_ids, nact, np.random.RandomState(1)), fo, shifts, specs, full)["o"]
    real_m = np.mean([o["real:%d" % t].mean() for t in tis])
    fake_m = np.mean([o["fakeu:%d:%d" % (r, t)].mean() for r in range(20) for t in tis])
    fakenm_m = np.mean([o["fakenm:%d:%d" % (r, t)].mean() for r in range(20) for t in tis])
    rand_m = np.mean([o["rand:%d:%d" % (t, k)].mean() for k in range(20) for t in tis])
    print("     planted oracle: real %.3f fake unscaled %.3f fake norm-matched %.3f random %.3f" % (real_m, fake_m, fakenm_m, rand_m))
    check("planted oracle (size read from the patch mean): real swap moved ~ 1; unscaled / norm-matched fake swap and random directions ~ 0", abs(real_m - 1) < 0.15 and abs(fake_m) < 0.15 and abs(fakenm_m) < 0.15 and abs(rand_m) < 0.15)
    ofs = {"o": copy_fn(dz)}
    oc = K.eval_source_batch(build_z_all(tokens, sel_ids, nact, np.random.RandomState(1)), ofs, shifts, specs, full)["o"]
    check("pass-through predictor (second look, other episodes) -> moved 0", abs(np.mean(list(oc.values()))) < 1e-4)
    const = lambda zin: torch.cat([zin[..., :dz] + 0.1, zin[..., dz:]], -1)
    oc2 = K.eval_source_batch(build_z_all(tokens, sel_ids, nact, np.random.RandomState(1)), {"k": const}, shifts, specs, full)["k"]
    check("predictor with input-independent change (pred = z + 0.1) -> moved 0", max(np.abs(v).max() for v in oc2.values()) < 1e-4)
    # predictor that outputs a fixed vector regardless of input (no pass-through): moved = -shift.d/|d|^2 (negative of the copy-through diagnostic)
    vfix = torch.as_tensor(tokens[0, 5], dtype=torch.float32)
    fixed = lambda zin: torch.cat([vfix.expand(zin.shape[0], 1, -1, -1)[:, 0:1].reshape(zin.shape[0], 1, P, dz), zin[..., dz:]], -1)
    of = K.eval_source_batch(build_z_all(tokens, sel_ids, nact, np.random.RandomState(1)), {"f": fixed}, shifts, specs, full)["f"]
    ct = (F["M"][0] - F["M"][si]) * (F["DZ"][0] - F["DZ"][si])
    ct = ct.sum() / ((F["DZ"][0] - F["DZ"][si]) ** 2).sum()
    check("predictor that ignores its input entirely: moved == -(copy-through diagnostic) (shows what the diagnostic measures)", abs(of["real:0"].mean() + ct) < 1e-3, "(%.4f vs %.4f)" % (of["real:0"].mean(), -ct))


# ------------------------------------------------------------------------------------------------ E
def synth_units(cfg, c2, rows, Etr, F, tokens, fns_by_cond, tmp, seeds, nact, per_level=None):
    import torch
    sel = S.select_source_episodes(rows[Etr:], LV, per_level)
    paths = {"results_dir": tmp / "res", "exp": tmp, "cache": tmp / "cache"}
    for cond, fns in fns_by_cond.items():
        for ts in seeds:
            rng = np.random.RandomState(1)
            arrs = S.compute_unit(cfg, c2, Etr, fns, lambda ids: build_z_all(tokens, ids, nact, rng), F, torch.device("cpu"), sel, 4)
            meta = S.unit_meta(LV, sel)
            meta.update({"condition": cond, "train_seed": ts, "run": "toy", "epoch": 0, "checkpoint_sha256": "x", "random_init_seed": 1234 + ts, "encoder_matches_cache": True,
                         "first_batch_max_abs_diff_encoder_vs_cache": 0.0, "episodes_per_level": per_level})
            S.part_npz(paths, cond, ts).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(S.part_npz(paths, cond, ts), **arrs)
            B.write_json(meta, S.part_json(paths, cond, ts))
    return paths


def t_aggregate(cfg, cfg2, cfg13):
    print("E  bootstrap and aggregate end to end (planted worlds)")
    c2, rows = toy_rows(cfg2)
    P, dz, nact = toy_dims()
    tokens, info = planted_tokens(rows, P, dz)
    F, Etr = fit_of(cfg, c2, rows, tokens)
    rec = S.fit_record(cfg, F)
    seeds = [0, 1, 2]
    g = np.repeat(np.arange(3), [10, 20, 30])
    ix = LB.boot_indices(g, 200, 14000)
    check("boot_indices reproducible for seed 14000, different for another seed, stratified", np.array_equal(ix, LB.boot_indices(g, 200, 14000)) and not np.array_equal(ix, LB.boot_indices(g, 200, 14001)) and all((g[ix[b]] == g).all() for b in range(200)))
    off = {"x": (10, 20)}
    st = LB.stat_with_ci({"x": np.full(20, 0.5)}, off, LB.boot_indices(g, 500, 14000), 0.90)
    check("constants: point 0.5 and the CI collapses (slice offset handled)", st["x"]["point"] == 0.5 and st["x"]["ci_lo"] == 0.5 and st["x"]["ci_hi"] == 0.5)
    r = np.random.RandomState(0).randn(20)
    st2 = LB.stat_with_ci({"x": r}, off, ix, 0.90)
    check("bootstrap CI contains the point and uses only the slice (mean of x over the pair's source episodes)", st2["x"]["ci_lo"] <= st2["x"]["point"] <= st2["x"]["ci_hi"] and abs(st2["x"]["point"] - r.mean()) < 1e-12)
    tmp = Path(tempfile.mkdtemp())
    try:
        run = lambda b, a: synth_units(cfg, c2, rows, Etr, F, tokens, {"multi_60_ind_160": {"trained": b, "random_init": copy_fn(dz)}, "single_ind": {"trained": a, "random_init": copy_fn(dz)}}, tmp, seeds, nact)
        conds = ["single_ind", "multi_60_ind_160"]
        agg = lambda paths, fr=rec: S.aggregate(cfg, None, c2, paths, conds, seeds, False, False, None, fit_rec=fr)
        paths = run(oracle_fn(info, dz), copy_fn(dz))
        res = agg(paths)
        v = res["verdict"]
        gm = res["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["gated_mean"]
        print("     planted b: moved %.3f fake norm-matched %.3f diff %.3f CI [%.3f, %.3f]; verdict %s" % (gm["moved"]["point"], gm["fake_normmatched_mean"]["point"], gm["diff_normmatched"]["point"], gm["diff_normmatched"]["ci_lo"], gm["diff_normmatched"]["ci_hi"], v["status"]))
        check("planted use in (b) only: criterion_met, all 4 conditions", v["status"] == LB.MET and v["cond3_b_greater_than_a"]["seed_mean_a"] is not None and abs(v["cond3_b_greater_than_a"]["seed_mean_a"]) < 1e-4, str({k: v[k]["met"] for k in v if k.startswith("cond")}))
        ga = res["stage1"]["conditions"]["single_ind"]["seeds"]["0"]["trained"]["pairs"]["a060>ind"]
        check("copy predictor in (a): moved 0, response 0, fake 0", abs(ga["moved"]["point"]) < 1e-4 and abs(ga["resp"]["point"]) < 1e-3 and abs(ga["fake_unscaled_mean"]["point"]) < 1e-4 and abs(ga["fake_normmatched_mean"]["point"]) < 1e-4)
        check("random-init reference reported per condition and seed (copy -> 0)", abs(res["random_init_gated_mean_moved_minus_fake_normmatched_reference"]["multi_60_ind_160"]["0"]["point"]) < 1e-4)
        pb = res["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["pairs"]["a060>ind"]
        check("pair records: moved / fake (unscaled, norm-matched) / rand_mean / diff_normmatched / diff_unscaled / resp with CI, gated flag, fake repeat spreads", all(k in pb for k in ("moved", "fake_unscaled_mean", "fake_normmatched_mean", "rand_mean", "diff_normmatched", "diff_unscaled", "resp", "gated", "fake_unscaled_repeat_means", "fake_normmatched_repeat_means"))
              and pb["fake_unscaled_repeat_means"]["n_repeats"] == 20 and pb["fake_normmatched_repeat_means"]["n_repeats"] == 20)
        check("results.json fields: pilot true (config flag + separate dir), pilot_reasons, config_sha256, code_version, environment, interpretation null, missing_units empty",
              res["pilot"] is True and len(res["pilot_reasons"]) >= 2 and len(res["config_sha256"]) == 64 and set(res["code_version"]) == {"git_head", "git_status_dirty", "repo_url", "requested_ref"}
              and "python" in res["environment"] and res["interpretation"] is None and res["missing_units"] == [] and (tmp / "res" / "results.json").exists())
        paths = run(copy_fn(dz), copy_fn(dz))
        check("no predictor moves (both copy): criterion_not_met", agg(paths)["verdict"]["status"] == LB.NOT_MET)
        paths = run(oracle_fn(info, dz), oracle_fn(info, dz))
        check("equal use in (a) and (b): cond3 (b > a) fails -> criterion_not_met", agg(paths)["verdict"]["cond3_b_greater_than_a"]["met"] is False and agg(paths)["verdict"]["status"] == LB.NOT_MET)
        paths = run(oracle_fn(info, dz), copy_fn(dz))
        r2 = copy.deepcopy(rec)
        r2["gated_pairs"] = ["a060>ind", "ind>a060"]
        res2 = agg(paths, r2)
        check("partial gate: metric averages only gated pairs (n_pairs 2), others reported not judged", res2["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["gated_mean"]["diff_normmatched"]["n_pairs"] == 2 and res2["verdict"]["status"] == LB.MET
              and res2["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["pairs"]["ind>a160"]["gated"] is False)
        r3 = copy.deepcopy(rec)
        r3["gated_pairs"] = []
        check("no pair passes the gate: not_evaluable (pair values still reported)", agg(paths, r3)["verdict"]["status"] == LB.NOT_EVAL and "a060>ind" in agg(paths, r3)["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["pairs"])
        S.part_json(paths, "single_ind", 1).unlink()
        res4 = agg(paths)
        check("a missing (a) unit -> not_evaluable, listed in missing_units", res4["verdict"]["status"] == LB.NOT_EVAL and "unit_single_ind_s1" in res4["missing_units"])
        paths5 = run(oracle_fn(info, dz), copy_fn(dz))
        res5 = S.aggregate(cfg, None, c2, paths5, conds, seeds, False, False, None, fit_rec=None)
        check("fit record missing -> not_evaluable", res5["verdict"]["status"] == LB.NOT_EVAL)
        cm = copy.deepcopy(cfg)
        cm["pilot"] = False
        check("pilot logic: config pilot false + experiment results dir + no overrides -> no reasons; each override adds a reason",
              S.pilot_reasons(cm, B.default_results_dir(), False, False, None) == [] and len(S.pilot_reasons(cm, B.default_results_dir(), True, True, 10)) == 3 and len(S.pilot_reasons(cm, tmp, False, False, None)) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ F
def t_toy_world(cfg, cfg2):
    print("F  toy world model: compute_unit, determinism, batch invariance")
    try:
        import torch
    except ImportError:
        REC["skipped"].append("F")
        return
    from exp009.selftest import toy_setup
    from exp004 import deep as DP
    M, old, c2, rows, C, w, (nvis, nact, P, L, H), T4 = toy_setup(cfg2)
    try:
        dev = torch.device("cpu")
        mean, std = torch.zeros(6), torch.ones(6)
        E = len(rows)
        Etr = sum(r["part"] == "train" for r in rows)
        toks = np.zeros((E, 13, P, nvis), np.float32)
        with torch.no_grad():
            for s in range(0, E, 16):
                sl = slice(s, min(E, s + 16))
                x = M.frames_to_float(C["frames_u8"][sl])
                acts = M.acts_from_raw(torch.as_tensor(C["actions_raw"][sl]), mean, std)
                toks[sl] = M.encode_batch(w, x, acts)[..., :nvis].numpy()
        F, _ = fit_of(cfg, c2, rows, toks)
        rnd = DP.random_init_predictor(w, 1234)
        fns = {"trained": lambda z: w.predict(z), "random_init": lambda z: rnd.predict(z)}

        def enc(ids):
            x = M.frames_to_float(C["frames_u8"][ids])
            acts = M.acts_from_raw(torch.as_tensor(C["actions_raw"][ids]), mean, std)
            return M.encode_batch(w, x, acts)
        sel = S.select_source_episodes(rows[Etr:], LV, None)
        a1 = S.compute_unit(cfg, c2, Etr, fns, enc, F, dev, sel, 4)
        a2 = S.compute_unit(cfg, c2, Etr, fns, enc, F, dev, sel, 10)
        a3 = S.compute_unit(cfg, c2, Etr, fns, enc, F, dev, sel, 4)
        k = S.okey("trained", "a060", "ind", "real")
        check("compute_unit shapes: real (10,), fake_unscaled (20, 10), fake_normmatched (20, 10), rand (20, 10), resp (10,) per directed pair, 6 pairs x 5 arrays x 2 predictors",
              a1[k].shape == (10,) and a1[S.okey("trained", "a060", "ind", "fake_unscaled")].shape == (20, 10) and a1[S.okey("trained", "a060", "ind", "fake_normmatched")].shape == (20, 10)
              and a1[S.okey("random_init", "a160", "ind", "rand")].shape == (20, 10) and len(a1) == 6 * 5 * 2)
        check("deterministic (same inputs twice) and batch-size invariant (4 vs 10 episodes)", all(np.array_equal(a1[x], a3[x]) for x in a1) and all(np.allclose(a1[x], a2[x], atol=1e-5) for x in a1))
        check("trained and random-init predictors give different values", not np.allclose(a1[k], a1[S.okey("random_init", "a060", "ind", "real")]))
        check("all values finite", all(np.isfinite(v).all() for v in a1.values()))
        print("     toy trained real moved (a060>ind) mean %.3f, norm-matched fake mean %.3f (NOT results)" % (a1[k].mean(), a1[S.okey("trained", "a060", "ind", "fake_normmatched")].mean()))
    finally:
        M.P_TOK, M.D_VIS = old


# ------------------------------------------------------------------------------------------------ G
def t_config(cfg, cfg13, cfg2):
    print("G  config consistency, attempts, preflight")
    res = B.consistency_checks(cfg, cfg13, cfg2)
    check("config consistency with 013/002 (%d checks)" % len(res), all(ok for _, ok, _ in res), str([n for n, ok, _ in res if not ok]))
    bad = copy.deepcopy(cfg)
    bad["success"]["bootstrap"]["seed"] = 5
    bad["controls"]["fake_swap"]["seed"] = 1
    bad["gate"]["d_reliability"]["min"] = 0.4
    names = [n for n, ok, _ in B.consistency_checks(bad, cfg13, cfg2) if not ok]
    check("changed bootstrap seed / fake-swap seed / gate threshold are detected", any("success" in n for n in names) and any("controls" in n for n in names) and any("gate" in n for n in names), str(names))
    check("config: seeds explicit (global 0, train seeds, random-init 1234, fake swap 15000, random directions 13000, bootstrap 14000) and pilot flag true",
          cfg["seed"] == 0 and cfg["predictor"]["train_seeds"] == [0, 1, 2] and cfg["predictor"]["random_init_seed"] == 1234 and cfg["controls"]["fake_swap"]["seed"] == 15000
          and cfg["controls"]["random_dir_seed"] == 13000 and cfg["success"]["bootstrap"]["seed"] == 14000 and cfg["pilot"] is True)
    tmp = Path(tempfile.mkdtemp())
    try:
        paths = {"results_dir": tmp}
        from exp002.common import attempts_record
        for i in range(3):
            attempts_record(paths, "stage:measure", False)
        check("attempts: after 3 consecutive failures the 4th run is refused (max_attempts 3)", _raises(lambda: B.attempts_check(cfg, paths, "stage:measure"), RuntimeError) and cfg["run"]["max_attempts"] == 3)
        B.attempts_check(cfg, paths, "stage:aggregate")
        out = tmp / "train_runs" / "outputs"
        (out / "multi_60_ind_160_s0").mkdir(parents=True)
        pth = {"train_base": tmp / "train_runs", "cache": tmp / "cache"}
        bad_ = S.S13.run_problems(pth, ["multi_60_ind_160"], [0])
        check("preflight: run dir without the exp002_ prefix is not accepted, message names the expected folder", len(bad_) == 1 and "exp002_multi_60_ind_160_s0" in bad_[0])
        pth["results_dir"] = tmp
        check("preflight: missing fit result is reported (measure needs `fit` first)", any("fit result missing" in b for b in S.fit_problems(pth)))
        pth["cache"] = tmp / "cache"
        fj = S.fit_json(pth)
        check("measure refusal helper: no fit.json -> no gate problem (fit_problems covers it)", S.gate_problems(pth) == [])
        B.write_json({"gated_pairs": []}, fj)
        gp = S.gate_problems(pth)
        check("measure refusal helper: 0 gated pairs -> one problem naming --allow-no-gated-pairs", len(gp) == 1 and "--allow-no-gated-pairs" in gp[0])
        B.write_json({"gated_pairs": ["a060>ind", "ind>a060"]}, fj)
        check("measure refusal helper: gated pairs present -> no problem", S.gate_problems(pth) == [])
        rs = (B.EXP_DIR / "run.py").read_text()
        check("run.py: measure checks gate_problems unless --allow-no-gated-pairs", "--allow-no-gated-pairs" in rs and "S.gate_problems(paths)" in rs and "args.allow_no_gated_pairs" in rs)
        nbg = notebook_guard_fn()
        if nbg is None:
            REC["skipped"].append("notebook guard (colab/build_notebook.py not found)")
        else:
            zd = tmp / "nbres"
            zd.mkdir()
            msg = lambda f: _raises(lambda: f(str(zd)), SystemExit)
            check("notebook guard: fit.json missing -> stops (SystemExit)", msg(nbg))
            B.write_json({"gated_pairs": []}, zd / "fit.json")
            try:
                nbg(str(zd))
                stopped, text = False, ""
            except SystemExit as e:
                stopped, text = True, str(e)
            check("notebook guard: 0 gated pairs -> stops with a clear message that measure is not run", stopped and "0 pairs" in text and "measure" in text)
            B.write_json({"gated_pairs": ["a060>ind", "ind>a060"]}, zd / "fit.json")
            check("notebook guard: gated pairs present -> continues and returns them", nbg(str(zd)) == ["a060>ind", "ind>a060"])
            cells = notebook_cells()
            e3 = [c for c in cells if "# E3." in c]
            e1 = [c for c in cells if "# E1." in c]
            check("notebook: main cell E3 calls gate_guard after fit and before measure; pilot cell E1 calls it before measure", len(e3) == 1 and len(e1) == 1 and 0 < e3[0].index("run015('fit'") < e3[0].index("gate_guard(") < e3[0].index("run015('measure'")
                  and 0 < e1[0].index("gate_guard(") < e1[0].index("run015(st, ex"))
        check("part file names distinct per (condition, seed)", len({S.part_json(pth, c, t).name for c in ("single_ind", "multi_60_ind_160") for t in (0, 1, 2)}) == 6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_nb_builder():
    import importlib.util
    pth = B.EXP_DIR / "colab" / "build_notebook.py"
    if not pth.exists():
        return None
    spec = importlib.util.spec_from_file_location("exp015_build_notebook", str(pth))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def notebook_cells():
    m = _load_nb_builder()
    return ["".join(c["source"]) for c in m.C if c["cell_type"] == "code"] if m else []


def notebook_guard_fn():
    """The gate_guard function exactly as defined in the notebook cell A6c (exec'ed, not re-implemented)."""
    src = [c for c in notebook_cells() if "def gate_guard" in c]
    if not src:
        return None
    ns = {}
    exec(src[0], ns)
    return ns["gate_guard"]


def synth_arrays(n_by, moved, fu, fn, rd=0.0, R=20, Kr=20):
    """Planted part-file arrays for aggregate: constant per-episode values, all 6 directed pairs, both predictors."""
    a = {}
    for kind in ("trained", "random_init"):
        for s in LV:
            for t in LV:
                if s == t:
                    continue
                n = n_by[s]
                a[S.okey(kind, s, t, "real")] = np.full(n, moved)
                a[S.okey(kind, s, t, "resp")] = np.full(n, 1.0)
                a[S.okey(kind, s, t, "fake_unscaled")] = np.full((R, n), fu)
                a[S.okey(kind, s, t, "fake_normmatched")] = np.full((R, n), fn)
                a[S.okey(kind, s, t, "rand")] = np.full((Kr, n), rd)
    return a


def t_normmatched_verdict(cfg, cfg2):
    print("H  verdict metric uses the NORM-MATCHED fake swap (planted arrays)")
    c2, rows = toy_rows(cfg2)
    P, dz, nact = toy_dims()
    tokens, info = planted_tokens(rows, P, dz)
    F, Etr = fit_of(cfg, c2, rows, tokens)
    rec = S.fit_record(cfg, F)
    sel = S.select_source_episodes(rows[Etr:], LV, None)
    n_by = {l: len(sel[l]) for l in LV}
    tmp = Path(tempfile.mkdtemp())
    try:
        def make(b_vals, a_vals):
            paths = {"results_dir": tmp / ("res%d" % len(list(tmp.iterdir()))), "exp": tmp, "cache": tmp / "cache"}
            for cond, vals in (("multi_60_ind_160", b_vals), ("single_ind", a_vals)):
                for ts in (0, 1, 2):
                    meta = S.unit_meta(LV, sel)
                    meta.update({"condition": cond, "train_seed": ts, "run": "toy", "epoch": 0, "checkpoint_sha256": "x", "random_init_seed": 1234 + ts, "encoder_matches_cache": True,
                                 "first_batch_max_abs_diff_encoder_vs_cache": 0.0, "episodes_per_level": None})
                    S.part_npz(paths, cond, ts).parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(S.part_npz(paths, cond, ts), **synth_arrays(n_by, *vals))
                    B.write_json(meta, S.part_json(paths, cond, ts))
            return paths
        conds, seeds = ["single_ind", "multi_60_ind_160"], [0, 1, 2]
        agg = lambda paths: S.aggregate(cfg, None, c2, paths, conds, seeds, False, False, None, fit_rec=rec)
        # (b): moved 0.50, unscaled fake 0.00, norm-matched fake 0.45 -> verdict metric 0.05 (< 0.10) although moved - unscaled would be 0.50
        r1 = agg(make((0.50, 0.00, 0.45), (0.00, 0.00, 0.00)))
        gm = r1["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["gated_mean"]
        v1 = r1["verdict"]
        check("verdict metric == moved - norm-matched fake mean (0.05), not moved - unscaled fake (0.50, reported as diff_unscaled)", abs(gm["diff_normmatched"]["point"] - 0.05) < 1e-12 and abs(gm["diff_unscaled"]["point"] - 0.50) < 1e-12
              and abs(v1["cond2_seed_mean_b_ge_min_moved"]["seed_mean_b"] - 0.05) < 1e-12)
        check("planted: unscaled difference would pass (0.50 >= 0.10) but the norm-matched difference 0.05 fails condition 2 -> criterion_not_met", v1["status"] == LB.NOT_MET and v1["cond2_seed_mean_b_ge_min_moved"]["met"] is False)
        r2 = agg(make((0.50, 0.40, 0.00), (0.00, 0.00, 0.00)))
        check("planted: norm-matched fake 0 and unscaled fake 0.40 -> verdict metric 0.50 -> criterion_met (unscaled value does not enter the verdict)", r2["verdict"]["status"] == LB.MET and abs(r2["verdict"]["cond2_seed_mean_b_ge_min_moved"]["seed_mean_b"] - 0.50) < 1e-12)
        check("verdict text names the norm-matched comparator; random-init reference key is the norm-matched one", "NORM-MATCHED" in r2["verdict"]["judged"] and "random_init_gated_mean_moved_minus_fake_normmatched_reference" in r2)
        pair = r1["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["pairs"]["a060>ind"]
        check("pair record keeps the unscaled fake as report-only fields (fake_unscaled_mean 0.00, diff_unscaled 0.50)", abs(pair["fake_unscaled_mean"]["point"]) < 1e-12 and abs(pair["diff_unscaled"]["point"] - 0.5) < 1e-12)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _with(fn, *a):
    with patched():
        return fn(*a)


def run(cfg, cfg13, cfg2, c9):
    for name, fn in (("A", lambda: t_verdict(cfg)), ("B", lambda: _with(t_numpy, cfg, cfg2)), ("C", lambda: _with(t_separation, cfg, cfg2)), ("D", lambda: _with(t_moved, cfg, cfg2)),
                     ("E", lambda: _with(t_aggregate, cfg, cfg2, cfg13)), ("F", lambda: t_toy_world(cfg, cfg2)), ("G", lambda: t_config(cfg, cfg13, cfg2)), ("H", lambda: _with(t_normmatched_verdict, cfg, cfg2))):
        try:
            fn()
        except Exception:
            REC["fail"].append(name + " (exception)")
            traceback.print_exc()
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    if REC["fail"]:
        print("FAILED:", REC["fail"])
    return 1 if REC["fail"] else 0
