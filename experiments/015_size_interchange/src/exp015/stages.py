"""Stages of experiment 015: check | fit | measure | aggregate | selftest.
fit      (CPU) level means m_L, level-mean true motion DZ_L, d per directed pair, split-half gate, fake-swap shifts, all from the TRAIN-split episodes of a060/ind/a160 of the Q1 cache
         -> <cache>/fit/interchange_fit.npz and <results>/fit.json
measure  (GPU) per (condition, train seed) unit: for every source level S and its two targets T: moved of the real swap, of 20 fake swaps and of 20 random directions, for the trained
         and the random-init predictor, on the test episodes of S -> <results>/parts/unit_<cond>_s<ts>.{npz,json}
aggregate(CPU) per pair statistics with episode bootstrap, verdict (condition (b), gated pairs), results.json
Resume: a finished unit (json written last) is skipped."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import base as B
from exp002 import cache as CA
from exp004 import deep as DP
from exp013 import stages as S13
from . import core as K
from . import lib as LB
from .base import log, read_json, write_json

TL = B.TRAIN_LEVELS


# ------------------------------------------------------------------------------------------------ fit
def fit_dir(paths):
    return paths["cache"] / "fit"


def fit_npz(paths):
    return fit_dir(paths) / "interchange_fit.npz"


def fit_json(paths):
    return paths["results_dir"] / "fit.json"


def compute_fit(cfg, cfg2, rows, Etr, token_fn, P, dz):
    """Pure computation (no I/O). Train-split rows of the three training levels only. token_fn(sorted row ids) -> (n, >=13, P, dz)."""
    levels = list(cfg2["data"]["train_levels_b"])
    ids = {l: np.array([i for i in range(Etr) if rows[i]["level"] == l]) for l in levels}
    assert all(len(v) > 0 and v.max() < Etr for v in ids.values()), "level-mean / motion fit episodes must be train-split rows"
    zb, db = [], []
    for l in levels:
        a, b = LB.episode_stats(token_fn, ids[l], P, dz)
        zb.append(a)
        db.append(b)
    F = LB.compute_fit_arrays(cfg, levels, zb, db)
    F.update({"levels": levels, "ids": {l: ids[l].tolist() for l in levels}})
    return F


def fit_record(cfg, F, enc_fp=None):
    halves = {l: {"half1_rows": np.asarray(F["ids"][l])[F["halves"][i][0]].tolist(), "half2_rows": np.asarray(F["ids"][l])[F["halves"][i][1]].tolist()} for i, l in enumerate(F["levels"])}
    gated = [p["pair"] for p in F["pairs"] if p["gated"]]
    return {"levels": F["levels"], "n_train_episodes_by_level": dict(zip(F["levels"], F["counts"])), "pairs": {p["pair"]: p for p in F["pairs"]}, "gated_pairs": gated, "n_gated_directed": len(gated),
            "gate": dict(cfg["gate"]["d_reliability"]), "gate_rule": "mean over %d random split-halves (RandomState(seed + 1)) of cos(d half 1, d half 2) >= min (user decision 2026-11-12)" % int(cfg["gate"]["d_reliability"].get("n_splits", 20)),
            "gate_split_seed": int(cfg["seed"]) + 1, "single_split_seed_report_only": int(cfg["seed"]), "single_split_halves_train_rows_report_only": halves,
            "fake_swap": {"n_repeats": int(cfg["controls"]["fake_swap"]["n_repeats"]), "seed": int(cfg["controls"]["fake_swap"]["seed"]),
                          "note": "per source level: halves of its train episodes, shift = mean(half 1) - mean(half 2) of the per-episode mean latent = UNSCALED fake swap (report only, see fake_unscaled_over_real_frob). "
                                  "PRIMARY comparator (user decision 2026-11-12) = norm-matched fake swap: each unscaled shift rescaled per patch to |m_T[p] - m_S[p]|, pair specific (computed at measure time with lib.norm_match; "
                                  "fake_normmatched_* fields are its diagnostics, fake_normmatched_n_guarded_patches counts patches with |fake| <= 1e-12 that were set to 0)"},
            "level_mean_rule": "013 level_means: per patch mean over the train-split episodes of the level and the 12 input frames (frames 0..11)",
            "target_motion_rule": "per patch mean over the same episodes and the 12 transitions of z_{t+1} - z_t",
            "encoder_fingerprint_cache": enc_fp, "created": time.strftime("%F %T")}


def fit_stage(cfg, cfg2, paths, dry_run=False, fresh=False):
    log("fit plan: level means m_L and level-mean true motion of %s from the train-split episodes of the Q1 cache (12 frames / 12 transitions); gate = mean over %s random split-halves of the cosine of d >= %s "
        "(split seed %s); fake swap %s repeats seed %s (unscaled, then norm-matched per pair)" % (TL, cfg["gate"]["d_reliability"].get("n_splits", 20), cfg["gate"]["d_reliability"]["min"], int(cfg["seed"]) + 1, cfg["controls"]["fake_swap"]["n_repeats"], cfg["controls"]["fake_swap"]["seed"]))
    if dry_run:
        return None
    C = CA.load(B.B4.as002(paths), "")
    rows, Etr, _, _ = S13.rows_info(cfg2, C)
    T = C["tokens_dino"]
    F = compute_fit(cfg, cfg2, rows, Etr, lambda i: T[np.sort(i)], K.M.P_TOK, K.M.D_VIS)
    rec = fit_record(cfg, F, C["meta"]["encoder_fingerprint"])
    fd = fit_dir(paths)
    fd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fit_npz(paths), M=F["M"], DZ=F["DZ"], fake=F["fake"], counts=np.array(F["counts"]))
    write_json(rec, fit_json(paths))
    for p in F["pairs"]:
        log("  %-10s cos(d halves) mean20 %.3f (single split %.3f) gated=%s |d| %.3f |shift| %.3f unscaled-fake/real %.3f copy-through real %.3f norm-matched fake %.3f" % (
            p["pair"], p["d_split_half_cosine_mean20"], p["d_split_half_cosine_single_split"], p["gated"], p["d_frob"], p["shift_frob"], p["fake_unscaled_over_real_frob"] or float("nan"), p["copy_through_real"], p["copy_through_fake_normmatched_mean"]))
    log("fit done: gated directed pairs %s" % rec["gated_pairs"])
    return rec


def load_fit(paths):
    z = np.load(fit_npz(paths))
    return {"M": z["M"], "DZ": z["DZ"], "fake": z["fake"], "counts": z["counts"].tolist()}


# ------------------------------------------------------------------------------------------------ preflight
def fit_problems(paths):
    return [] if fit_npz(paths).exists() else ["fit result missing: %s (run `python run.py fit`)" % fit_npz(paths)]


def gate_problems(paths):
    """User decision 2026-11-12 (#11): if `fit` found 0 gated pairs, `measure` is refused (preflight, exit code 4, no attempt consumed) unless --allow-no-gated-pairs is given.
    A missing fit.json is not reported here (fit_problems covers the fit npz)."""
    fj = fit_json(paths)
    if not fj.exists():
        return []
    g = read_json(fj).get("gated_pairs", [])
    if len(g) == 0:
        return ["0 pairs passed the d-reliability gate in %s: measure would spend GPU time on a verdict that is not_evaluable (user decision 2026-11-12). Report it; to run anyway pass --allow-no-gated-pairs" % fj]
    return []


def input_preflight(paths, conds, seeds, need_fit=True):
    return S13.input_preflight(paths, conds, seeds, need_fit=False) + (fit_problems(paths) if need_fit else [])


# ------------------------------------------------------------------------------------------------ units
def unit_name(cond, ts):
    return "%s_s%d" % (cond, ts)


def part_json(paths, cond, ts):
    return paths["results_dir"] / "parts" / ("unit_%s.json" % unit_name(cond, ts))


def part_npz(paths, cond, ts):
    return paths["results_dir"] / "parts" / ("unit_%s.npz" % unit_name(cond, ts))


def select_source_episodes(rows_test, levels, per_level):
    """{level: test-episode indices (into the test rows) of the level in rank_in_level order; the first `per_level` if given (pilot)}."""
    out = {}
    for l in levels:
        idx = sorted([j for j, r in enumerate(rows_test) if r["level"] == l], key=lambda j: int(rows_test[j]["rank_in_level"]))
        out[l] = np.array(idx[: int(per_level)] if per_level is not None else idx, dtype=int)
    return out


def okey(kind, s, t, what):
    return "%s__%s__%s__%s" % (kind, s, t, what)


def compute_unit(cfg, cfg2, Etr, fns, encode_fn, F, dev, sel, bs, label=""):
    """moved of the real swap, the fake swaps and the random directions for the predictors in fns {kind: fn(z (N,1,P,D)) -> (N,1,P,D)} on the test episodes `sel` {level: indices into the
    test rows}. encode_fn(row ids) -> z_all (B,13,P,D) real encoder latents. Returns the arrays dict for the npz (no I/O)."""
    import torch
    levels = list(cfg2["data"]["train_levels_b"])
    Kr = int(cfg["controls"]["random_directions"])
    R = int(cfg["controls"]["fake_swap"]["n_repeats"])
    rdirs = LB.random_dirs(Kr, K.M.D_VIS, int(cfg["controls"]["random_dir_seed"]))
    arrs = {}
    t0 = time.time()
    for si, s in enumerate(levels):
        tis = [ti for ti in range(len(levels)) if ti != si]
        shifts, specs = K.build_shifts(F["M"], F["fake"], si, tis, rdirs, dev)
        dvec = {}
        for ti in tis:
            d = F["DZ"][ti] - F["DZ"][si]
            dvec[ti] = (torch.as_tensor(d, dtype=torch.float32, device=dev), float((d.astype(np.float64) ** 2).sum()))
        acc = {k: {} for k in fns}
        ep = sel[s]
        for a in range(0, len(ep), bs):
            ids = Etr + ep[a:a + bs]
            with torch.no_grad():
                z_all = encode_fn(ids)
                o = K.eval_source_batch(z_all, fns, shifts, specs, dvec)
            for k in fns:
                for name, v in o[k].items():
                    acc[k].setdefault(name, []).append(v)
            if (a // bs) % 10 == 0:
                log("%s source %s: %d/%d episodes (%.0fs)" % (label, s, min(a + bs, len(ep)), len(ep), time.time() - t0))
        for k in fns:
            cat = {n: np.concatenate(v, 0) for n, v in acc[k].items()}
            for ti in tis:
                t = levels[ti]
                arrs[okey(k, s, t, "real")] = cat["real:%d" % ti]
                arrs[okey(k, s, t, "resp")] = cat["resp:real:%d" % ti]
                arrs[okey(k, s, t, "fake_unscaled")] = np.stack([cat["fakeu:%d:%d" % (r, ti)] for r in range(R)], 0)
                arrs[okey(k, s, t, "fake_normmatched")] = np.stack([cat["fakenm:%d:%d" % (r, ti)] for r in range(R)], 0)
                arrs[okey(k, s, t, "rand")] = np.stack([cat["rand:%d:%d" % (ti, j)] for j in range(Kr)], 0)
    return arrs


def measure_unit(cfg, cfg2, paths, cond, ts, C, dev, per_level, bs, fresh=False):
    jp = part_json(paths, cond, ts)
    if jp.exists() and not fresh:
        log("unit %s done, skip" % unit_name(cond, ts))
        return None
    t0 = time.time()
    levels = list(cfg2["data"]["train_levels_b"])
    rows, Etr, _, _ = S13.rows_info(cfg2, C)
    model, tcfg, ep, fp, mean, std, rdir = S13.load_model_for(paths, cfg2, cond, ts, dev, C)
    F = load_fit(paths)
    rnd = DP.random_init_predictor(model, int(cfg["predictor"]["random_init_seed"]) + ts)
    sel = select_source_episodes(rows[Etr:], levels, per_level)
    log("unit %s: run %s epoch %d, source episodes %s, 6 directed pairs x (1 real + %d unscaled fake (shared per source) + %d norm-matched fake + %d random) x 2 predictors" % (
        unit_name(cond, ts), rdir.name, ep, {l: len(v) for l, v in sel.items()}, cfg["controls"]["fake_swap"]["n_repeats"], cfg["controls"]["fake_swap"]["n_repeats"], cfg["controls"]["random_directions"]))
    fns = {"trained": lambda z: model.predict(z), "random_init": lambda z: rnd.predict(z)}
    fd = {}

    def encode_fn(ids):
        import torch
        x = K.M.frames_to_float(np.asarray(C["frames_u8"][ids])).to(dev)
        acts = K.M.acts_from_raw(torch.as_tensor(np.asarray(C["actions_raw"][ids])), mean, std).to(dev)
        z_all = K.M.encode_batch(model, x, acts)
        if "v" not in fd:
            fd["v"] = float((z_all[..., :K.M.D_VIS].float().cpu() - torch.as_tensor(np.asarray(C["tokens_dino"][ids], np.float32))).abs().max())
            log("unit %s first batch: max |encode_batch visual - cached tokens_dino| = %.3g (cache is fp16)" % (unit_name(cond, ts), fd["v"]))
        return z_all
    arrs = compute_unit(cfg, cfg2, Etr, fns, encode_fn, F, dev, sel, bs, label="unit " + unit_name(cond, ts))
    first_diff = fd.get("v")
    pp = part_npz(paths, cond, ts)
    pp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pp, **arrs)
    meta = unit_meta(levels, sel)
    meta.update({"condition": cond, "train_seed": ts, "run": rdir.name, "epoch": ep, "checkpoint": cfg["base"]["checkpoint"], "checkpoint_sha256": S13.sha256_file(rdir / "checkpoints" / "model_latest.pth"),
                 "random_init_seed": int(cfg["predictor"]["random_init_seed"]) + ts, "encoder_fingerprint": fp, "cache_encoder_fingerprint": C["meta"]["encoder_fingerprint"],
                 "encoder_matches_cache": fp == C["meta"]["encoder_fingerprint"], "first_batch_max_abs_diff_encoder_vs_cache": first_diff, "batch_episodes": bs,
                 "episodes_per_level": per_level, "seconds": round(time.time() - t0, 1), "finished": time.strftime("%F %T"), "device": str(dev)})
    write_json(meta, jp)                                         # written last = done marker
    log("unit %s done in %.0fs" % (unit_name(cond, ts), time.time() - t0))
    return meta


def unit_meta(levels, sel):
    return {"levels": list(levels), "kinds": ["trained", "random_init"], "test_episode_index_by_source": {l: sel[l].tolist() for l in levels}, "n_by_source": {l: int(len(sel[l])) for l in levels}}


def measure_stage(cfg, cfg2, paths, conds, seeds, per_level, bs, fresh=False, dry_run=False):
    done = [(c, t) for c in conds for t in seeds if part_json(paths, c, t).exists()]
    log("unit plan: conditions %s, train seeds %s, units %d (done %d), test episodes per source level %s; per unit forwards per sample: (1 + %d unscaled fake + 2 x (1 real + %d norm-matched fake + %d random)) x 2 predictors = %d x 2" % (
        conds, seeds, len(conds) * len(seeds), len(done), per_level if per_level else "all (100)", cfg["controls"]["fake_swap"]["n_repeats"], cfg["controls"]["fake_swap"]["n_repeats"], cfg["controls"]["random_directions"],
        1 + int(cfg["controls"]["fake_swap"]["n_repeats"]) + 2 * (1 + int(cfg["controls"]["fake_swap"]["n_repeats"]) + int(cfg["controls"]["random_directions"]))))
    if dry_run:
        return None
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(B.B4.as002(paths), "")
    for cond in conds:
        for ts in seeds:
            measure_unit(cfg, cfg2, paths, cond, ts, C, dev, per_level, bs, fresh)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return True


# ------------------------------------------------------------------------------------------------ check
def check(cfg, cfg13, cfg2, paths, conds, seeds):
    print("cache dir (fit):", paths["cache"], "exists=%s" % paths["cache"].exists())
    res = B.consistency_checks(cfg, cfg13, cfg2)
    for n, ok, d in res:
        print("  [%s] %s %s" % ("ok" if ok else "FAIL", n, "" if ok else d))
    bad = input_preflight(paths, conds, seeds, need_fit=False)
    print("inputs (checkpoints of %s x seeds %s, Q1 cache, metadata.pt): %s" % (conds, seeds, "all present" if not bad else "MISSING %d" % len(bad)))
    for b in bad[:20]:
        print("  !! " + b)
    print("fit done:", fit_npz(paths).exists())
    return 0 if (not bad and all(ok_ for _, ok_, _ in res)) else 1


# ------------------------------------------------------------------------------------------------ aggregate
def pilot_reasons(cfg, results_dir, conds_over, seeds_over, per_level, metas=()):
    default_dir = B.default_results_dir().resolve()
    r = []
    if cfg["pilot"]:
        r.append("config.yaml pilot flag is true")
    if Path(results_dir).resolve() != default_dir:
        r.append("results dir %s is not the experiment results dir %s (pilot / separate run)" % (Path(results_dir).resolve(), default_dir))
    if conds_over:
        r.append("conditions overridden on the command line")
    if seeds_over:
        r.append("train seeds overridden on the command line")
    if per_level is not None:
        r.append("only the first %s test episodes per source level were evaluated (--episodes-per-level)" % per_level)
    if per_level is None and any(m.get("episodes_per_level") is not None for m in metas):
        r.append("some units evaluated only the first N test episodes per source level (--episodes-per-level)")
    return r


def pooled_layout(levels, n_by_source):
    """Pooled episode index over source levels in config order: offsets {level: (start, n)}, group ids (n_total,)."""
    off, g, a = {}, [], 0
    for i, l in enumerate(levels):
        n = int(n_by_source[l])
        off[l] = (a, n)
        g += [i] * n
        a += n
    return off, np.array(g, int)


def unit_stats(cfg, meta, Mx, fit_rec, ix, kind):
    """Statistics of one unit and predictor kind. fit_rec['pairs'] provides the gate. Per directed pair: moved, unscaled fake swap mean (report only), norm-matched fake swap mean (PRIMARY
    comparator, user decision 2026-11-12), random-direction mean, diff_normmatched = moved - norm-matched fake mean (VERDICT metric), diff_unscaled = moved - unscaled fake mean (report only),
    diff_rand, response norm, each with point and bootstrap CI; plus the gated mean."""
    ci = float(cfg["success"]["ci"])
    levels = meta["levels"]
    off_l, _ = pooled_layout(levels, meta["n_by_source"])
    gated = [p for p in fit_rec["gated_pairs"]]
    series = {"moved": {}, "fake_unscaled_mean": {}, "fake_normmatched_mean": {}, "rand_mean": {}, "diff_normmatched": {}, "diff_unscaled": {}, "diff_rand": {}, "resp": {}}
    gated_names = tuple(n for n in series if n != "resp")
    offsets = {}
    fake_rep_means = {}
    for s in levels:
        for t in levels:
            if s == t:
                continue
            p = B.pair_name(s, t)
            mv, fu, fn, rd, rp = (Mx[okey(kind, s, t, w)] for w in ("real", "fake_unscaled", "fake_normmatched", "rand", "resp"))
            series["moved"][p] = mv
            series["fake_unscaled_mean"][p] = fu.mean(0)
            series["fake_normmatched_mean"][p] = fn.mean(0)
            series["rand_mean"][p] = rd.mean(0)
            series["diff_normmatched"][p] = mv - fn.mean(0)
            series["diff_unscaled"][p] = mv - fu.mean(0)
            series["diff_rand"][p] = mv - rd.mean(0)
            series["resp"][p] = rp
            offsets[p] = off_l[s]
            fake_rep_means[p] = {"fake_unscaled_repeat_means": fu.mean(1), "fake_normmatched_repeat_means": fn.mean(1)}
    out = {"pairs": {}}
    for name, xs in series.items():
        st = LB.stat_with_ci(xs, offsets, ix, ci, gated=gated if name in gated_names else None)
        for p, v in st.items():
            if p == "gated_mean":
                out.setdefault("gated_mean", {})[name] = v
            else:
                out["pairs"].setdefault(p, {})[name] = v
    for p, dd in fake_rep_means.items():
        for nm, v in dd.items():
            out["pairs"][p][nm] = {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max()), "n_repeats": int(len(v))}
    for p in out["pairs"]:
        out["pairs"][p]["gated"] = p in gated
    return out


def aggregate(cfg, cfg13, cfg2, paths, conds, seeds, conds_over, seeds_over, per_level, dry_run=False, fit_rec=None):
    log("aggregate: conditions %s seeds %s" % (conds, seeds))
    if dry_run:
        return None
    succ = cfg["success"]
    levels = list(cfg2["data"]["train_levels_b"])
    if fit_rec is None:
        fit_rec = read_json(fit_json(paths)) if fit_json(paths).exists() else None
    units, missing = {}, []
    for c in conds:
        for ts in seeds:
            jp = part_json(paths, c, ts)
            if jp.exists() and part_npz(paths, c, ts).exists():
                units[(c, ts)] = (read_json(jp), dict(np.load(part_npz(paths, c, ts))))
            else:
                missing.append("unit_" + unit_name(c, ts))
    nb, bseed = int(succ["bootstrap"]["n"]), int(succ["bootstrap"]["seed"])
    boot_cache = {}
    stage1 = {"conditions": {}}
    for c in conds:
        cres = {"seeds": {}}
        for ts in seeds:
            if (c, ts) not in units or fit_rec is None:
                continue
            meta, Mx = units[(c, ts)]
            _, g = pooled_layout(meta["levels"], meta["n_by_source"])
            bk = tuple(g.tolist())
            if bk not in boot_cache:
                boot_cache[bk] = LB.boot_indices(g, nb, bseed)      # same indices for every unit with the same episodes (all conditions / seeds / predictors)
            sres = {"meta": {k: meta[k] for k in ("run", "epoch", "checkpoint_sha256", "random_init_seed", "encoder_matches_cache", "first_batch_max_abs_diff_encoder_vs_cache", "n_by_source", "episodes_per_level") if k in meta}}
            for kind in meta["kinds"]:
                sres[kind] = unit_stats(cfg, meta, Mx, fit_rec, boot_cache[bk], kind)
            cres["seeds"][str(ts)] = sres
        stage1["conditions"][c] = cres
    b_name, a_name = "multi_60_ind_160", "single_ind"
    gated = fit_rec["gated_pairs"] if fit_rec else []

    def get(cond, kind="trained"):
        out = {}
        for ts in seeds:
            u = stage1["conditions"].get(cond, {}).get("seeds", {}).get(str(ts), {}).get(kind)
            if not u or u["gated_mean"]["diff_normmatched"] is None:
                out[ts] = None
            else:
                out[ts] = {"metric": u["gated_mean"]["diff_normmatched"], "pairs": {p: u["pairs"][p]["diff_normmatched"]["point"] for p in gated}}
        return out
    if fit_rec is None:
        verdict = {"status": LB.NOT_EVAL, "reason": "fit.json missing (run fit)"}
    elif b_name in conds and a_name in conds:
        verdict = LB.judge(get(b_name), get(a_name), gated, succ, seeds)
    else:
        verdict = {"status": LB.NOT_EVAL, "reason": "both conditions are needed"}
    verdict["judged"] = "trained predictor, condition (b) multi_60_ind_160, mean over gated directed pairs of (moved - NORM-MATCHED fake swap mean; user decision 2026-11-12), source-level test episodes; (a) single_ind for condition 3"
    ref = {}
    for c in conds:
        ref[c] = {}
        for ts in seeds:
            u = stage1["conditions"].get(c, {}).get("seeds", {}).get(str(ts), {}).get("random_init")
            ref[c][str(ts)] = (u["gated_mean"]["diff_normmatched"] if u else None)
    from exp002.common import env_info
    reasons = pilot_reasons(cfg, paths["results_dir"], conds_over, seeds_over, per_level, [m for m, _ in units.values()])
    res = {"experiment": cfg["name"], "pilot": bool(reasons), "pilot_reasons": reasons, "pilot_config_flag": bool(cfg["pilot"]),
           "conditions": conds, "train_seeds": seeds, "episodes_per_level_override": per_level,
           "gate": {"rule": dict(cfg["gate"]["d_reliability"]), "gated_pairs": gated, "fit_present": fit_rec is not None},
           "fit": fit_rec,
           "metric_definitions": K.__doc__.split("Per sample")[1].strip() if K.__doc__ else None, "success": dict(succ),
           "stage1": stage1, "verdict": verdict, "random_init_gated_mean_moved_minus_fake_normmatched_reference": ref,
           "missing_units": missing, "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(),
           "environment": env_info(), "code_version": B.code_version(paths["exp"]), "interpretation": None}
    write_json(res, paths["results_dir"] / "results.json")
    if Path(paths["results_dir"]).resolve() == B.default_results_dir().resolve():
        write_json(res, paths["exp"] / "results.json")
    log("aggregate done (pilot=%s): verdict %s" % (res["pilot"], verdict["status"]))
    return res
