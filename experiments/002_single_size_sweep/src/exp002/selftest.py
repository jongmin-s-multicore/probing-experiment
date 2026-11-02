"""CPU self-test on SYNTHETIC data with known answers. The numbers printed here are NOT experiment results.
Sub-tests that need torch are reported as 'skipped(no torch)' when torch is missing (never counted as passed).
Covers: metrics/bootstrap/Holm/Spearman, matched random directions, INLP/LEACE, Q1 probes + 3 controls (end to end through the cache format),
H-a/H-b/H-c verdict logic with known outcomes (incl. null threshold -> not_evaluable, (a) missing -> pending, threshold wiring),
hooks + Q3 run_layer on a toy predictor with the branch's module layout (uses it / ignores it / nonlinear information -> not_removed)."""
from __future__ import annotations

import copy
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

from . import directions as D
from . import stats as ST
from .common import load_config, write_json, read_json, ConfigNull, require, PreregReader

REC = {"pass": [], "fail": [], "skipped": []}


def check(name, cond, detail=""):
    (REC["pass"] if cond else REC["fail"]).append(name)
    print("  [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))


def skip(name, why):
    REC["skipped"].append(name)
    print("  [skipped(%s)] %s" % (why, name))


def have_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------------------------------------ A. numpy parts
def t_stats():
    print("A1 stats")
    rng = np.random.RandomState(0)
    n_eps, per = 40, 5
    ep = np.repeat(np.arange(n_eps), per)
    y = rng.randn(n_eps * per)
    h = y + 0.5 * rng.randn(n_eps * per)
    S = ST.episode_stats(y, h, ep, n_eps)
    tot = S.sum(0)
    check("R2 from sums == direct", abs(ST.metric_from_sums(tot, "r2") - (1 - ((y - h) ** 2).sum() / ((y - y.mean()) ** 2).sum())) < 1e-9)
    check("pearson from sums == np.corrcoef", abs(ST.metric_from_sums(tot, "pearson") - np.corrcoef(y, h)[0, 1]) < 1e-9)
    check("MAE from sums", abs(ST.metric_from_sums(tot, "mae") - np.abs(y - h).mean()) < 1e-9)
    lv = np.repeat(np.arange(4), n_eps // 4)
    b = ST.Bootstrap(lv, 300, 1)
    st = b.stat(S[None], np.ones(n_eps), "r2")
    lo, hi = ST.ci95(st["boot"])
    check("bootstrap CI contains point", lo <= st["point"] <= hi, "(%.3f in [%.3f, %.3f])" % (st["point"], lo, hi))
    sw = b.stat(S[None], np.ones(n_eps), "r2_within")
    ys, hs = y.reshape(-1), h
    ref_sst = sum(((y[(lv[ep] == l)] - y[(lv[ep] == l)].mean()) ** 2).sum() for l in range(4))
    ref_sse = sum((((y - y[lv[ep] == l].mean())[lv[ep] == l] - (h - h[lv[ep] == l].mean())[lv[ep] == l]) ** 2).sum() for l in range(4))
    check("within-level R2 == direct", abs(sw["point"] - (1 - ref_sse / ref_sst)) < 1e-9)
    adj, rej = ST.holm([0.01, 0.04, 0.03], 0.05)
    check("Holm adjusted p", np.allclose(adj, [0.03, 0.06, 0.06]) and rej == [True, False, False], str(adj))
    adj, rej = ST.holm([0.001, None], 0.05)
    check("Holm: None counts in family size (m=2)", abs(adj[0] - 0.002) < 1e-12 and rej == [True, False])
    check("Spearman monotone = 1, reversed = -1", ST.spearman([1, 2, 3, 4], [1, 5, 6, 9]) == 1.0 and ST.spearman([1, 2, 3], [3, 2, 1]) == -1.0)
    check("Spearman ties (average ranks)", abs(ST.spearman([1, 2, 2, 3], [1, 2, 3, 4]) - np.corrcoef(ST.rankdata([1, 2, 2, 3]), ST.rankdata([1, 2, 3, 4]))[0, 1]) < 1e-12)
    check("pair order accuracy perfect", ST.pair_order_accuracy([1, 2, 3, 4], [1, 1, 2, 3]) == 1.0)


def t_directions():
    print("A2 directions / INLP / LEACE")
    rng = np.random.RandomState(0)
    N, d = 600, 20
    A = rng.randn(d, d) @ np.diag(np.linspace(0.2, 3, d))
    base = rng.randn(N, d) @ A.T
    a = rng.randn(d)
    a /= np.linalg.norm(a)
    y = rng.randn(N)
    X = base + 4.0 * y[:, None] * a[None]
    ep = np.arange(N)
    m = D.ridge_fit(X[:400], y[:400], ep[:400], [0.1, 1, 10, 100], 3, 0)
    check("ridge held-out R2 high on linear data", D.r2(y[400:], D.ridge_predict(m, X[400:])) > 0.7, "%.3f" % D.r2(y[400:], D.ridge_predict(m, X[400:])))
    lam, V = D.cov_eig(X)
    tgt = float(np.var(X @ m["u"], ddof=1))
    rs = [D.matched_random_direction(lam, V, tgt, np.random.RandomState(s)) for s in range(30)]
    check("matched random directions: unit norm", all(abs(np.linalg.norm(r) - 1) < 1e-8 for r, _, _ in rs))
    check("matched random directions: variance matched (<1%)", max(abs(q - tgt) / tgt for _, q, _ in rs) < 0.01, "max rel err %.4f" % max(abs(q - tgt) / tgt for _, q, _ in rs))
    iso = np.random.RandomState(1).randn(200, d)
    iso /= np.linalg.norm(iso, axis=1, keepdims=True)
    viso = np.array([np.var(X @ v, ddof=1) for v in iso])
    check("isotropic Gaussian directions would NOT be variance matched (control is meaningful)", np.std(viso) / tgt > 0.2, "iso rel std %.2f" % (np.std(viso) / tgt))
    r0, r1 = rs[0][0], rs[1][0]
    check("different RNG streams give different directions", abs(r0 @ r1) < 0.99)
    ss = [np.random.SeedSequence([1000, ts, 3, 0]).generate_state(1)[0] for ts in (0, 1, 2)]
    check("per-training-seed direction streams are distinct", len(set(ss)) == 3)
    B, r2s = D.inlp(X[:400], y[:400], ep[:400], 20, [1.0, 10.0], 3, 0)
    Xr = X - (X @ B) @ B.T
    check("INLP: train linear R2 decreases over iterations", r2s[-1] < r2s[0] - 0.3, "%.2f -> %.2f" % (r2s[0], r2s[-1]))
    mr = D.ridge_fit(Xr[:400], y[:400], ep[:400], [1.0, 10.0], 3, 0)
    check("INLP: linear info gone (held-out ridge R2 < 0.05)", D.r2(y[400:], D.ridge_predict(mr, Xr[400:])) < 0.05, "r2 %.3f, rank %d" % (D.r2(y[400:], D.ridge_predict(mr, Xr[400:])), B.shape[1]))
    M, mu = D.leace_fit(X[:400], y[:400])
    Xl = D.apply_removal(X, M, mu)
    ml = D.ridge_fit(Xl[:400], y[:400], ep[:400], [1.0, 10.0], 3, 0)
    check("LEACE: linear info gone (held-out ridge R2 < 0.05)", D.r2(y[400:], D.ridge_predict(ml, Xl[400:])) < 0.05)
    check("LEACE: rank-1 change only", np.linalg.matrix_rank(M, tol=1e-6) == 1)
    check("removal strength 0 is identity", np.allclose(D.apply_removal(X, M, mu, 0.0), X))


# ------------------------------------------------------------------------------------------------ B. synthetic cache -> Q1 -> aggregate
def small_cfg(base, mask_noise):
    cfg = copy.deepcopy(base)
    cfg["data"]["train_episodes_per_level"] = 12
    cfg["data"]["eval_episodes_per_level"] = 8
    cfg["probe"]["seeds"] = [0, 1]
    cfg["probe_impl"]["n_bootstrap"] = 200
    cfg["probe_impl"]["attentive"].update({"d_model": 16, "n_heads": 2, "mlp_hidden": 16, "epochs": 15, "batch_size": 64})
    cfg["probe_impl"]["device"] = "cpu"
    cfg["probe_impl"]["mlp_check"].update({"hidden": 32, "epochs": 600, "lr": 3.0e-3})
    cfg["train"]["seeds"] = [0, 1, 2]
    cfg["_mask_noise"] = mask_noise
    return cfg


def make_rows(cfg):
    from .data import level_roles, nominal_log_area
    roles = level_roles(cfg)
    rows = []
    for part, levels, n in (("train", cfg["data"]["train_levels_b"], cfg["data"]["train_episodes_per_level"]), ("test", cfg["data"]["levels"], cfg["data"]["eval_episodes_per_level"])):
        for l in levels:
            for i in range(n):
                rows.append({"id": len(rows), "part": part, "level": l, "role": roles[l], "split": "x", "name": "e%d" % len(rows), "dir": "x", "rank_in_level": i,
                             "nominal_log_area": nominal_log_area(cfg, l)})
    return rows


P_SYN, D_SYN = 6, 16


def feat_from_size(s, rng, scale, noise=0.3, frames=13):
    """tokens (E,frames,P,D): scale * s * a + noise. a fixed direction."""
    a = np.random.RandomState(123).randn(D_SYN)
    a /= np.linalg.norm(a)
    t = scale * s[:, None, None, None] * a[None, None, None, :] + noise * rng.randn(len(s), frames, P_SYN, D_SYN)
    return t.astype(np.float16)


def write_synth_cache(cfg, paths, tokens_scale=1.0, mask_noise=0.6):
    rng = np.random.RandomState(7)
    rows = make_rows(cfg)
    s = np.array([r["nominal_log_area"] for r in rows])
    E = len(rows)
    cd = paths["cache"] / "q1_tokens"
    cd.mkdir(parents=True, exist_ok=True)
    np.save(cd / "tokens_dino.npy", feat_from_size(s, rng, tokens_scale))
    np.save(cd / "tokens_rand.npy", (0.3 * rng.randn(E, 13, P_SYN, D_SYN)).astype(np.float16))
    area = np.exp(5 + s[:, None] + mask_noise * rng.randn(E, 13))
    mf = np.stack([area, np.sqrt(area), np.sqrt(area)], -1).astype(np.float32)
    np.save(cd / "mask_feat.npy", mf)
    np.save(cd / "frames_u8.npy", np.zeros((E, 13, 2, 2, 3), np.uint8))
    np.save(cd / "actions_raw.npy", np.zeros((E, 60, 6), np.float32))
    write_json(rows, cd / "episodes.json")
    write_json({"encoder_fingerprint": "synthetic", "K": 13}, cd / "meta.json")
    (cd / "DONE").write_text("ok\n")
    return rows


def synth_paths(tmp):
    return {"results_dir": Path(tmp) / "results", "exp": Path(tmp), "cache": Path(tmp) / "cache", "logs": Path(tmp) / "logs", "manifests": Path(tmp) / "manifests"}


def t_q1_aggregate(base):
    print("B  synthetic cache -> Q1 (probes + 3 controls) -> H-a verdicts")
    from . import q1 as Q1, aggregate as AG
    kinds = ["linear_pooled", "attentive"] if have_torch() else ["linear_pooled"]
    if not have_torch():
        skip("attentive probe (Q1)", "no torch")
    out = {}
    for case, (scale, mnoise, expect) in {"signal": (1.0, 0.6, "pass"), "no_signal": (0.0, 0.6, "fail"), "saturated_baseline": (1.0, 0.0, "not_evaluable")}.items():
        tmp = tempfile.mkdtemp()
        try:
            cfg = small_cfg(base, mnoise)
            paths = synth_paths(tmp)
            write_synth_cache(cfg, paths, scale, mnoise)
            cfg["probe"]["kinds"] = kinds
            Q1.run(cfg, paths, only_kinds=kinds)
            res = AG.run(cfg, paths)
            rows = [r for r in res["decision_table"] if r["id"].startswith("H_a") and r["id"].split(":")[-1] in kinds]
            check("H-a [%s]: %d rows for the probe kinds that ran" % (case, len(rows)), len(rows) == 2 * len(kinds))
            check("H-a [%s]: all statuses == %s" % (case, expect), all(r["status"] == expect for r in rows), str([(r["id"], r["status"]) for r in rows]))
            er = res["h_a_early_flat_report"]
            check("H-a [%s]: early_flat reported (not judged) with baseline R2 + saturation flag" % case, er["status"] == "reported_not_judged" and len(er["rows"]) == 2 * len(kinds)
                  and all("r2_mask_baseline" in x for x in er["rows"]) and all(x["baseline_saturated"] == (case == "saturated_baseline") for x in er["rows"]))
            if case == "signal":
                cell = np.load(paths["results_dir"] / "q1" / "cells.npz")
                r0 = rows[0]["result"]
                check("signal: real R2 high, shuffle control low, random-init low",
                      r0["r2_real"] > 0.5 and r0["r2_random_init"] < 0.3, "real %.2f rand %.2f base %.2f" % (r0["r2_real"], r0["r2_random_init"], r0["r2_mask_baseline"]))
                sh = json_cell(paths, "%s|linear_pooled|shuffle|all|all" % cfg["q1"]["primary_target"], "unseen")
                check("label-shuffle control R2 near 0 or negative", sh < 0.25, "%.3f" % sh)
                check("results.json has CI caveat + prereg values used", "training-seed" in res["ci_caveat"] and "q2_drop_threshold" in res["prereg_values_used"])
                # threshold wiring: margin 0.99 must flip pass -> fail
                cfg2 = copy.deepcopy(cfg)
                cfg2["prereg"]["q1_margin"]["delta_r2_vs_mask_area_baseline"] = 0.99
                res2 = AG.run(cfg2, paths)
                rows2 = [r for r in res2["decision_table"] if r["id"].startswith("H_a") and r["id"].split(":")[-1] in kinds]
                check("threshold wiring: raising q1 margin to 0.99 changes status to fail", all(r["status"] == "fail" for r in rows2))
                cfg3 = copy.deepcopy(cfg)
                cfg3["prereg"]["q1_margin"]["delta_r2_vs_mask_area_baseline"] = None
                rows3 = [r for r in AG.run(cfg3, paths)["decision_table"] if r["id"].startswith("H_a") and r["id"].split(":")[-1] in kinds]
                check("null margin -> not_evaluable", all(r["status"] == "not_evaluable" for r in rows3))
                out["paths"], out["cfg"], out["tmp"] = paths, cfg, tmp
                tmp = None
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    return out


def json_cell(paths, key, role):
    s = read_json(paths["results_dir"] / "q1" / "q1_summary.json")
    return s["cells"][key][role]["r2"]["point"]


def t_h_b(ctx):
    print("C  H-b verdict logic on synthetic Q2 cells")
    from . import aggregate as AG, evalgrid as EG
    cfg, paths = ctx["cfg"], ctx["paths"]
    from . import q2 as Q2
    prim = cfg["q1"]["primary_target"]
    cfgo = copy.deepcopy(cfg)
    cfgo["probe"]["kinds"] = ["linear_pooled"]
    rows = make_rows(cfg)
    tr, te = EG.split_rows(rows)
    Etr, Ete = len(tr), len(te)
    mf = np.load(paths["cache"] / "q1_tokens" / "mask_feat.npy")
    Y = EG.make_labels(cfg, rows, mf, prim)
    W = EG.windows(cfg)
    frames = W["union"]
    pos = {w: [frames.index(f) for f in W[w]] for w in ("early", "late", "union")}
    for case, (keep_scale, expect) in {"preserved": (0.9, "pass"), "destroyed": (0.0, "fail")}.items():
        for seed in (0, 1, 2):
            rng = np.random.RandomState(100 + seed)
            s = np.array([r["nominal_log_area"] for r in rows])
            cells, L = {}, 3
            for j in range(L + 1):
                sc = 1.0 if j < L else keep_scale
                F = feat_from_size(s, rng, sc, frames=len(frames))
                Yp = Y[:, frames]
                c, _ = EG.fit_predict_cells(cfg, "linear_pooled", F[:Etr], Yp[:Etr], F[Etr:], Yp[Etr:], list(range(len(frames))), pos, [0, 1])
                for w in c:
                    cells["%s|linear_pooled|layer%d|%s" % (prim, j, w)] = c[w]
            tokens = feat_from_size(s, rng, 1.0)                                  # encoder tokens (E,13,P,D)
            R_te = feat_from_size(s[Etr:], rng, keep_scale, frames=12)            # predicted z of TEST episodes only
            oc, _, _ = Q2.output_space(cfgo, tokens, Y, Etr, R_te, 0, 12)
            cells.update(oc)
            d = paths["results_dir"] / ("exp002_multi_60_ind_160_s%d" % seed)
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "q2_cells.npz", **cells)
            write_json({"meta": {"n_blocks": L, "t": 0, "horizon": 12}, "diagnostics_4c": {"verdict": "small"}}, d / "q2_summary.json")
        cfgx = copy.deepcopy(cfg)
        cfgx["q2_impl"]["primary_probe"] = "linear_pooled"
        r = [x for x in AG.run(cfgx, paths)["decision_table"] if x["id"].startswith("H_b")]
        check("H-b [%s]: statuses == %s" % (case, expect), all(x["status"] == expect for x in r), str([(x["id"], x["status"], round(x["result"].get("drop_mean_over_seeds", -9), 3)) for x in r]))
        if case == "preserved":
            cfgy = copy.deepcopy(cfgx)
            cfgy["q2"]["drop_threshold"] = -0.5
            r2 = [x for x in AG.run(cfgy, paths)["decision_table"] if x["id"].startswith("H_b")]
            check("threshold wiring: q2.drop_threshold=-0.5 flips H-b to fail", all(x["status"] == "fail" for x in r2))
            cfgz = copy.deepcopy(cfgx)
            cfgz["q2"]["drop_threshold"] = None
            r3 = [x for x in AG.run(cfgz, paths)["decision_table"] if x["id"].startswith("H_b")]
            check("null q2.drop_threshold -> not_evaluable", all(x["status"] == "not_evaluable" for x in r3))
    r = [x for x in AG.run(cfg, paths)["decision_table"] if x["id"].startswith("cond_a_vs_b")]
    check("cond (a) vs (b) is 'pending' when (a) results are missing", all(x["status"] == "pending" for x in r) and len(r) == 2)
    rows_all = AG.run(cfg, paths)["decision_table"]
    check("exploratory row has no p-value", all(x["p_value"] is None for x in rows_all if x["status"] == "exploratory"))


# ------------------------------------------------------------------------------------------------ F. new in 확정 기록 6/7
def t_budget_manifest(base):
    print("F1 equal budget + manifests (episodes per condition)")
    from . import budget as B, data as DA
    pl = B.plan({"multi_60_ind_160": 1000, "single_ind": 1000, "multi_60_ind_160_small": 334}, 10, "multi_60_ind_160")
    check("equal-episode conditions get the same epochs and steps", pl["single_ind"]["epochs"] == 10 and pl["single_ind"]["steps_computed"] == pl["multi_60_ind_160"]["steps_computed"])
    check("small condition: epochs converted to (nearly) the same total steps, rel_diff recorded", pl["multi_60_ind_160_small"]["epochs"] == 30 and abs(pl["multi_60_ind_160_small"]["rel_diff"]) < 0.01,
          str(pl["multi_60_ind_160_small"]))
    pl2 = B.plan({"a": 100, "b": 33}, 3, "a", budget_steps=600)
    check("train.budget (total steps) overrides: epochs = round(steps/spe)", pl2["a"]["epochs"] == 6 and pl2["b"]["epochs"] == 18 and pl2["b"]["steps_computed"] == 594)
    check("steps_per_epoch ceil/floor", B.steps_per_epoch(101, 24) == 5 and B.steps_per_epoch(101, 24, "floor") == 4)
    listing = {}
    for l in ("a060", "ind", "a160"):
        listing["%s_train" % l] = ["%s_%d.mp4" % (l, i) for i in range(2000)]
    m = {c["name"]: DA.build_train_manifest(base, listing, "/d", c["name"]) for c in base["conditions"]["list"]}
    check("(a) single_ind = 900 ind episodes", m["single_ind"]["n_episodes"] == 900 and m["single_ind"]["group_counts"] == {"ind": 900})
    check("(b) = 3 x 300 = 900", m["multi_60_ind_160"]["n_episodes"] == 900 and set(m["multi_60_ind_160"]["group_counts"].values()) == {300})
    check("small = 3 x 100 = 300", m["multi_60_ind_160_small"]["n_episodes"] == 300 and set(m["multi_60_ind_160_small"]["group_counts"].values()) == {100})
    key = lambda mm: {(e["split"], e["name"]) for e in mm["episodes"]}
    check("small subset of (b); (b)'s ind 300 subset of (a)", key(m["multi_60_ind_160_small"]) <= key(m["multi_60_ind_160"]) and
          {k for k in key(m["multi_60_ind_160"]) if k[0] == "ind_train"} <= key(m["single_ind"]))
    check("run_order: small last and optional; train.budget null; torch_version set", base["conditions"]["run_order"][-1] == "multi_60_ind_160_small" and
          "multi_60_ind_160_small" in base["conditions"]["optional"] and base["train"]["budget"] is None and base["env"]["torch_version"] == "2.0.1+cu117")
    try:
        DA.build_train_manifest(base, {"ind_train": ["x"] * 100}, "/d", "single_ind")
        check("too few episodes -> error", False)
    except ValueError:
        check("too few ind episodes (<900) raises", True)


def t_output_space(base):
    print("F2 output-space probe (Q1 probe applied to predicted z; no refit on predictor outputs)")
    from . import q2 as Q2, evalgrid as EG
    cfg = small_cfg(base, 0.5)
    cfg["probe"]["kinds"] = ["linear_pooled"]
    rows = make_rows(cfg)
    tr, te = EG.split_rows(rows)
    Etr = len(tr)
    s = np.array([r["nominal_log_area"] for r in rows])
    rng = np.random.RandomState(3)
    tokens = feat_from_size(s, rng, 1.0)
    Y = EG.make_labels(cfg, rows, np.zeros((len(rows), 13, 3)), cfg["q1"]["primary_target"])
    R1 = feat_from_size(s[Etr:], rng, 1.0, frames=4)
    R2 = feat_from_size(s[Etr:], rng, 0.0, frames=4)
    c1, p1, d1 = Q2.output_space(cfg, tokens, Y, Etr, R1, 0, 4)
    c2, p2, d2 = Q2.output_space(cfg, tokens, Y, Etr, R2, 0, 4)
    pre = "%s|linear_pooled|OUT|" % cfg["q1"]["primary_target"]
    check("encoder-side reads and probe digest do not depend on the predictor output (probe not fitted on it)", all(np.array_equal(c1[pre + "encoder|step%d" % k], c2[pre + "encoder|step%d" % k]) for k in range(1, 5))
          and d1 == d2)
    rm = EG.role_masks(te)
    boot = ST.Bootstrap(EG.level_index(te, list(cfg["data"]["levels"])), 100, 0)
    r_keep = boot.stat(c1[pre + "rollout|step4"], rm["unseen"], "r2")["point"]
    r_kill = boot.stat(c2[pre + "rollout|step4"], rm["unseen"], "r2")["point"]
    r_base = boot.stat(c1[pre + "rollout|step4"], rm["trained"], "r2_within")["point"]
    check("rollout read: preserved signal high, destroyed signal low", r_keep > 0.5 and r_kill < r_keep - 0.5, "%.2f vs %.2f" % (r_keep, r_kill))
    check("trained-role baseline cell exists for the same probe", np.isfinite(boot.stat(c1[pre + "rollout|step1"], rm["trained"], "mae")["point"]))
    check("predicted-size means stored per level source (pm shape = probe seeds x test eps)", p1[pre + "rollout|step1"].shape == (2, len(te)))


def t_diag(base):
    print("F3 4c diagnostics (rule of 확정 기록 7 A) incl. boundary values")
    from . import diag as DG
    sm, mm, el = base["diag_4c"]["activation_small_max_ratio"], base["diag_4c"]["activation_medium_max_ratio"], base["diag_4c"]["error_large_min_ratio"]
    check("config diag_4c = 2.0 / 5.0 / 2.0 and seed set", (sm, mm, el) == (2.0, 5.0, 2.0) and base["diag_4c"]["bootstrap_seed"] is not None)
    cr = lambda r: DG.classify_ratio(r, sm, mm)
    check("boundaries: 1.999 small, 2.0 medium, 4.999 medium, 5.0 large", (cr(1.999), cr(2.0), cr(4.999), cr(5.0)) == ("small", "medium", "medium", "large"))
    check("error boundary: 1.999 not large, 2.0 large", (DG.classify_err(1.999, el), DG.classify_err(2.0, el)) == ("not_large", "large"))
    check("final: any large -> large; else worst activation class", DG.final_verdict(["small", "small"], ["not_large", "large"]) == "large" and DG.final_verdict(["small", "medium"], ["not_large", "not_large"]) == "medium"
          and DG.final_verdict(["large", "small"], ["not_large"] * 2) == "large" and DG.final_verdict(["small"], ["not_large"]) == "small")
    check("final: missing -> not_evaluable", DG.final_verdict([], []) == "not_evaluable" and DG.final_verdict(["small"], ["not_evaluable"]) == "not_evaluable")
    rng = np.random.RandomState(0)
    mk = lambda n, mu, sd: [DG.episode_stats(mu + sd * rng.randn(2, 3, 5)) for _ in range(n)]
    st = {0: mk(40, 0.0, 1.0) + mk(20, 0.0, 1.0) + mk(20, 3.0, 1.0)}          # 0..39 ind, 40..59 same distribution, 60..79 mean shifted by 3 sd
    ne = {"ind": np.ones(40), "same": np.ones(20), "shifted": np.ones(20), "bigerr": 2.5 * np.ones(20)}
    res = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60)), "shifted": list(range(60, 80))}, {**ne}, 100, 1, base["diag_4c"])
    check("same distribution as ind -> activation class small", res["levels"]["same"]["activation_class"] == "small", "ratio %.2f" % res["levels"]["same"]["max_ratio_over_layers"])
    check("mean shift of 3 sd -> large", res["levels"]["shifted"]["activation_class"] == "large" and res["verdict"] == "large")
    ne2 = {"ind": np.ones(40), "same": 2.5 * np.ones(20)}
    res2 = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60))}, ne2, 100, 1, base["diag_4c"])
    check("same activations but next-frame error 2.5x ind -> large (error criterion)", res2["verdict"] == "large" and res2["levels"]["same"]["error_class"] == "large")
    res3 = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60))}, {"ind": np.ones(40), "same": 1.5 * np.ones(20)}, 100, 1, base["diag_4c"])
    check("verdict follows activation class when error < 2x; rule text stored", res3["verdict"] == res3["levels"]["same"]["activation_class"] and res3["interpretation_rule_for_verdict"])
    r_a = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60))}, ne2, 100, 1, base["diag_4c"])["noise_floor_per_layer"]
    r_b = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60))}, ne2, 100, 1, base["diag_4c"])["noise_floor_per_layer"]
    check("noise floor reproducible for a fixed seed; uses only ind episodes", r_a == r_b)
    from . import aggregate as AG
    tmp = tempfile.mkdtemp()
    try:
        paths = synth_paths(tmp)
        cfg = copy.deepcopy(base)
        r0 = [r for r in AG.diagnostics_rows(cfg, paths) if r["id"] == "diag_4c:single_ind"][0]
        check("diagnostics row without data -> not_evaluable", r0["status"] == "not_evaluable")
        d = paths["results_dir"] / "exp002_single_ind_s0"
        d.mkdir(parents=True)
        np.savez_compressed(d / "q2_cells.npz", x=np.zeros(1))
        write_json({"meta": {"n_blocks": 1}, "diagnostics_4c": res}, d / "q2_summary.json")
        r1 = [r for r in AG.diagnostics_rows(cfg, paths) if r["id"] == "diag_4c:single_ind"][0]
        check("diagnostics row carries the verdict + rule (large: numbers only)", r1["result"]["verdict_worst_over_seeds"] == "large" and "do NOT interpret" in r1["result"]["interpretation_rule"])
        rows = AG.cond_a_vs_b(cfg, PreregReader(cfg), paths, ST.Bootstrap(np.zeros(10, int), 10, 0), EG_masks(), None)
        check("cond_a_vs_b with (a) 4c large and (b) missing -> still pending (no judgment)", all(r["status"] in ("pending",) for r in rows))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def EG_masks():
    return {"unseen": np.ones(10, bool)}


def t_q3_pure(base):
    print("F4 Q3: episode-mean alpha curve, Spearman threshold, tiers")
    from . import q3 as Q3
    ds = np.array([[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0]])
    ep = np.array([0, 0, 1, 1])
    check("episode_mean: mean within episode then over episodes", np.allclose(Q3.episode_mean(ds, ep), [3.0, 4.0]))
    ds2 = np.array([[0.0, 0.0, 0.0, 10.0]])
    check("episode mean differs from sample mean when episodes have unequal sizes", abs(Q3.episode_mean(ds2, np.array([0, 0, 0, 1]))[0] - 5.0) < 1e-12 and abs(ds2.mean() - 2.5) < 1e-12)
    check("monotonic_min_spearman == 0.9 (from config)", base["q3_impl"]["monotonic_min_spearman"] == 0.9)
    r = ST.spearman([-2, -1, -0.5, 0, 0.5, 1, 2], [1, 2, 3, 4, 5, 7, 6])
    check("7-point curve with one adjacent swap: rho >= 0.9 passes, < 1.0 would fail", 0.9 <= abs(r) < 1.0, "rho %.4f" % r)
    check("Q3 confirmatory = ridge + INLP (config); others exploratory", base["q3_impl"]["primary_steering_direction"] == "ridge" and base["q3_impl"]["primary_removal_method"] == "INLP")


# ------------------------------------------------------------------------------------------------ D. torch: hooks + Q3 toy
def build_toy(dim, ptok, nvis, kind):
    import torch
    nn = torch.nn

    class Lin(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.l = nn.Linear(dim, dim, bias=False)
            nn.init.normal_(self.l.weight, std=scale)

        def forward(self, x):
            return self.l(x)

    class Tr(nn.Module):
        def __init__(self, L):
            super().__init__()
            self.layers = nn.ModuleList([nn.ModuleList([Lin(0.0 if kind == "ignore" else 0.02), Lin(0.0 if kind == "ignore" else 0.02)]) for _ in range(L)])
            self.norm = nn.Identity()

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return self.norm(x)

    class Pred(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Tr(4)

        def forward(self, x):
            return self.transformer(x)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = Pred()

        def predict(self, z):
            N, H, P, d = z.shape
            out = self.predictor(z.reshape(N, H * P, d)).reshape(N, H, P, d)
            if kind == "ignore":                       # output = function of action dims only (visual part ignored)
                out = torch.cat([z[..., nvis:].mean(-1, keepdim=True).expand(N, H, P, nvis) * 0.0, out[..., nvis:]], -1)
            return out
    return Toy()


def t_torch_q3(base):
    print("D  hooks + Q3 on a toy predictor (module layout like the branch: transformer.layers[(attn, ff)], transformer.norm)")
    if not have_torch():
        skip("hooks / Q3 run_layer / H-c verdicts", "no torch")
        return
    import torch
    from . import model_io as M, q3 as Q3, aggregate as AG
    M.P_TOK, Q3.D_VIS = 4, 8
    dim, nvis, ptok, H = 12, 8, 4, 1
    dev = torch.device("cpu")
    torch.manual_seed(0)
    model = build_toy(dim, ptok, nvis, "use")
    hooks = M.StreamHooks(model).install()
    z = torch.randn(5, H, ptok, dim)
    hooks.reset()
    hooks.capture_layers = {0, 2, 4}
    out = model.predict(z)
    cap = {k: v.clone() for k, v in hooks.store.items()}
    x = z.reshape(5, ptok, dim)
    manual = {0: x.clone()}
    for i, (a, f) in enumerate(model.predictor.transformer.layers):
        x = a(x) + x
        x = f(x) + x
        manual[i + 1] = x.clone()
    check("hook captures residual stream at layers 0/2/4(=L)", all(torch.allclose(cap[k], manual[k], atol=1e-6) for k in (0, 2, 4)))
    hooks.reset()
    hooks.modify = {2: lambda t: t * 0 + t}
    o2 = model.predict(z)
    check("identity modification leaves the output unchanged", torch.allclose(o2, out, atol=1e-6))
    hooks.reset()
    hooks.modify = {4: lambda t: t + 1.0}
    o3 = model.predict(z)
    check("shift at last layer changes the output by exactly the shift", torch.allclose(o3 - out, torch.ones_like(out), atol=1e-5))
    hooks.reset()
    hooks.modify = {2: lambda t: t + 1.0}
    o4 = model.predict(z)
    check("shift at a MIDDLE layer travels along the residual path (output change ~ shift, not ~0)", abs((o4 - out).mean().item() - 1.0) < 0.15, "mean change %.3f" % (o4 - out).mean().item())
    check("layer index rule: 6 blocks -> middle=3,last=6", Q3.layer_indices(base, 6)[0] == {"middle": 3, "last": 6})
    hooks.remove()

    cfg = small_cfg(base, 0.5)
    cfg["q3"]["n_random_directions"] = 100
    cfg["q3_impl"].update({"batch_samples": 64, "inlp_iterations": 4})
    cfg["probe"]["seeds"] = [0, 1]
    results = {}
    rng = np.random.RandomState(0)
    for kind, nonlinear in (("use", False), ("ignore", False), ("use", True)):
        model = build_toy(dim, ptok, nvis, kind)
        hooks = M.StreamHooks(model).install()

        class Ad(Q3.ModelAdapter):
            pass
        ad = Ad(model, hooks, H, dev)
        a = np.random.RandomState(5).randn(dim)
        a /= np.linalg.norm(a)

        nuis_bulk, nuis_hi = (0.2, 0.0) if nonlinear else (1.0, 4.0)       # high-variance nuisance dims (like action dims) keep matched random directions non-degenerate
        if nonlinear:
            lv = np.array([0.0, 1.0, 3.0])
            ytr, yev = np.repeat(lv, 40), np.repeat(lv, 20)
            b = np.cov(lv ** 2, lv, bias=True)[0, 1] / lv.var()
            f = lambda y: (y ** 2 - b * y)
            fs = f(lv)
            sc = 2.0 / (fs - fs.mean()).std()
            sig_of = lambda y: (f(y) - fs.mean()) * sc          # linearly uncorrelated with y on the training levels, but injective
        else:
            ytr = np.repeat(np.linspace(-1, 1, 5), 24)
            yev = np.repeat(np.linspace(-1.5, 1.5, 6), 10)
            sig_of = lambda y: 2.0 * y

        def mk(n, y):
            nsc = np.array([nuis_bulk] * nvis + [nuis_hi] * (dim - nvis))
            zz = 0.5 * rng.randn(n, H, ptok, dim) + nsc * rng.randn(n, 1, 1, dim) + sig_of(y)[:, None, None, None] * a[None, None, None, :]
            return torch.tensor(zz, dtype=torch.float32)

        Ztr, Zev = mk(len(ytr), ytr), mk(len(yev), yev)
        eptr = np.arange(len(ytr))
        wv = np.zeros(nvis)
        wv[:] = (a[:nvis] / (np.linalg.norm(a[:nvis]) + 1e-9))
        readout = {"w": torch.tensor(wv, dtype=torch.float32), "b": 0.0}
        t0 = __import__("time").time()
        res = Q3.run_layer(cfg, ad, 2, "middle", Ztr, ytr, eptr, Zev, yev, np.arange(len(yev)) // 2, readout, 0, 0, 2)
        name = "%s%s" % (kind, "+nonlinear" if nonlinear else "")
        results[name] = res
        hooks.remove()
        print("  (toy %s: %.0fs)" % (name, __import__("time").time() - t0))
        check("[%s] zero-shift scan equals baseline (identity)" % name, res["identity_check_max_abs_dsize_zero_shift"] < 1e-5)
        check("[%s] random directions variance-matched (<5%%)" % name, res["steering"]["ridge"]["random_variance_match_ok"])
    s_use = results["use"]["steering"]["ridge"]
    s_ign = results["ignore"]["steering"]["ridge"]
    check("toy 'uses size': size direction outside 95th percentile of 100 random, monotone", s_use["outside_percentile"] and s_use["monotone"], "p_rank %.3f rho %.2f" % (s_use["p_rank"], s_use["size_direction"]["spearman_signed_dsize_vs_alpha"]))
    check("toy 'ignores size': NOT outside the random distribution", not s_ign["outside_percentile"], "p_rank %.3f" % s_ign["p_rank"])
    check("tiers: ridge/INLP confirmatory, meandiff/LEACE exploratory", results["use"]["steering"]["ridge"]["tier"] == "confirmatory" and results["use"]["steering"]["meandiff"]["tier"] == "exploratory"
          and results["use"]["removal"]["INLP"]["tier"] == "confirmatory" and results["use"]["removal"]["LEACE"]["tier"] == "exploratory")
    r_use = results["use"]["removal"]["INLP"]
    check("toy linear info: INLP removed (MLP R2 after <= tol)", r_use["removed"], str(np.round(r_use["check"]["mlp_r2_after_removal_per_probe_seed"], 3)))
    r_nl = results["use+nonlinear"]["removal"]["INLP"]
    check("toy nonlinear info: INLP flagged not_removed", (not r_nl["removed"]) and r_nl["flag"] == "not_removed",
          "mlp after %s before %s" % (np.round(r_nl["check"]["mlp_r2_after_removal_per_probe_seed"], 2), np.round(r_nl["check"]["mlp_r2_before_per_probe_seed"], 2)))
    # H-c verdicts from the stored q3.json files
    for case, toy, expect in (("uses", "use", "pass"), ("ignores", "ignore", "fail")):
        tmp = tempfile.mkdtemp()
        try:
            paths = synth_paths(tmp)
            for seed in (0, 1, 2):
                lay = {"middle": results[toy], "last": results[toy]}
                write_json({"layers": lay}, paths["results_dir"] / ("exp002_multi_60_ind_160_s%d" % seed) / "q3.json")
            rows = [r for r in AG.h_c(cfg, PreregReader(cfg), paths, "multi_60_ind_160")]
            st = {r["id"]: r["status"] for r in rows}
            if toy == "use":
                for seed in (0, 1, 2):                      # remove exploratory blocks: verdicts must not change
                    lay2 = {t: {**v, "steering": {"ridge": v["steering"]["ridge"]}, "removal": {"INLP": v["removal"]["INLP"]}} for t, v in lay.items()}
                    write_json({"layers": lay2}, paths["results_dir"] / ("exp002_multi_60_ind_160_s%d" % seed) / "q3.json")
                st2 = {r["id"]: r["status"] for r in AG.h_c(cfg, PreregReader(cfg), paths, "multi_60_ind_160")}
                check("H-c verdicts identical without the exploratory blocks (meandiff, LEACE)", st2 == st)
            print("   H-c [%s]: %s" % (case, st))
            if toy == "use":
                check("H-c steering [uses] == pass", st["H_c:steering"] == "pass")
                check("H-c removal [uses] is pass or fail (removal verified), not not_evaluable", st["H_c:subspace_removal"] in ("pass", "fail"))
            else:
                check("H-c steering [ignores] == fail", st["H_c:steering"] == "fail")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    tmp = tempfile.mkdtemp()
    try:
        paths = synth_paths(tmp)
        for seed in (0, 1, 2):
            write_json({"layers": {"middle": results["use+nonlinear"], "last": results["use+nonlinear"]}}, paths["results_dir"] / ("exp002_multi_60_ind_160_s%d" % seed) / "q3.json")
        st = {r["id"]: r["status"] for r in AG.h_c(cfg, PreregReader(cfg), paths, "multi_60_ind_160")}
        check("H-c removal with not_removed flag -> not_evaluable (no conclusion)", st["H_c:subspace_removal"] == "not_evaluable")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_config(base):
    print("E  config guards")
    try:
        require(base, "train.batch_size")
        check("null batch_size raises ConfigNull", False)
    except ConfigNull as e:
        check("null batch_size raises ConfigNull with a clear message", "TODO" in str(e))
    import yaml
    from .common import _strict_loader
    try:
        yaml.load("a: 1\na: 2\n", Loader=_strict_loader())
        check("duplicate YAML keys rejected", False)
    except ValueError:
        check("duplicate YAML keys rejected", True)
    check("Q1 primary window = late (late_deformed), report window = early", base["q1"]["primary_window"] == "late" and base["q1"]["report_window"] == "early" and base["q1"]["baseline_saturated_r2"] == 0.99)
    check("config seeds present", base["seed"] is not None and base["train"]["seeds"] and base["probe"]["seeds"] and base["q3_impl"]["random_direction_seed"] is not None)


def run(base_cfg=None):
    print("=" * 100)
    print("SELFTEST on synthetic data. The numbers below are NOT experiment results.")
    print("=" * 100)
    base = base_cfg or load_config()
    for name, fn, args in (("stats", t_stats, ()), ("directions", t_directions, ()), ("config", t_config, (base,))):
        try:
            fn(*args)
        except Exception:
            traceback.print_exc()
            REC["fail"].append(name + " (exception)")
    for name, fn in (("budget/manifest", t_budget_manifest), ("output-space probe", t_output_space), ("diag 4c", t_diag), ("q3 pure", t_q3_pure)):
        try:
            fn(base)
        except Exception:
            traceback.print_exc()
            REC["fail"].append(name + " (exception)")
    ctx = {}
    try:
        ctx = t_q1_aggregate(base)
        if ctx:
            t_h_b(ctx)
    except Exception:
        traceback.print_exc()
        REC["fail"].append("q1/aggregate synthetic (exception)")
    finally:
        if ctx.get("tmp"):
            shutil.rmtree(ctx["tmp"], ignore_errors=True)
    try:
        t_torch_q3(base)
    except Exception:
        traceback.print_exc()
        REC["fail"].append("torch/q3 (exception)")
    print("=" * 100)
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    for n in REC["skipped"]:
        print("  skipped:", n)
    for n in REC["fail"]:
        print("  FAILED:", n)
    print("(synthetic numbers are not results; skipped sub-tests were NOT run)")
    return 0 if not REC["fail"] else 1
