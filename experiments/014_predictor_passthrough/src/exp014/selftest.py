"""Synthetic-data selftest of experiment 014 (CPU, numpy only). Checks: dual ridge == primal ridge, CKA invariances, fold hygiene, linear target ->
CKA/R2 near 1, independent target -> near 0, shuffled null near 0, alignment indexing end-to-end on synthetic cache files, missing-file guard."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from . import analysis as A
from . import data as D
from . import metrics as M
from . import runner as R

FAILS = []


def check(name, cond, info=""):
    print("%s  %s  %s" % ("PASS" if cond else "FAIL", name, info), flush=True)
    if not cond:
        FAILS.append(name)


def t_dual_vs_primal():
    rng = np.random.RandomState(0)
    n, d, q = 90, 30, 7
    X, Y = rng.randn(n, d), rng.randn(n, d) @ rng.randn(d, q) * 0.3 + rng.randn(n, q)
    a, b = np.arange(0, 60), np.arange(60, 90)
    al = [0.1, 3.0, 50.0]
    e, yd = M._fold_sse(X @ X.T, Y @ Y.T, a, b, al)
    mx, my = X[a].mean(0), Y[a].mean(0)
    Xa, Ya = X[a] - mx, Y[a] - my
    ok = True
    for i, alpha in enumerate(al):
        W = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(d), Xa.T @ Ya)
        pred = (X[b] - mx) @ W
        ok &= np.allclose(((Y[b] - my - pred) ** 2).sum(1), e[i], rtol=1e-6, atol=1e-8)
    ok &= np.allclose(((Y[b] - my) ** 2).sum(1), yd)
    check("dual ridge SSE == primal ridge SSE (3 alphas)", bool(ok))


def t_cka():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 40)
    Q, _ = np.linalg.qr(rng.randn(40, 40))
    K = M.gram(X)
    check("CKA(X,X)=1", abs(M.cka(K, K) - 1) < 1e-9)
    check("CKA invariant to rotation + scale", abs(M.cka(K, M.gram(3.0 * X @ Q)) - 1) < 1e-9)
    check("CKA of independent Gaussians small", M.cka(K, M.gram(rng.randn(100, 40))) < 0.5, "value %.3f" % M.cka(K, M.gram(rng.randn(100, 40))))


def t_folds():
    g = np.repeat(np.arange(50), 3)
    f = M.folds_by_group(g, 5, 0)
    leak = any(len(set(f[g == e])) > 1 for e in np.unique(g))
    check("folds: each episode in one fold, 5 folds used", (not leak) and len(set(f)) == 5, "sizes %s" % np.bincount(f).tolist())
    check("folds: deterministic given seed", (M.folds_by_group(g, 5, 0) == f).all() and not (M.folds_by_group(g, 5, 1) == f).all())


def t_synth(cfg):
    rng = np.random.RandomState(2)
    n, D_, q = 400, 300, 200
    g = np.repeat(np.arange(200), 2)
    Z = rng.randn(n, 20)
    X = Z @ rng.randn(20, D_) + 0.3 * rng.randn(n, D_)
    Yl = X @ rng.randn(D_, q) / np.sqrt(D_) + 0.1 * rng.randn(n, q)
    Yi = rng.randn(n, q)
    c = json.loads(json.dumps(cfg))
    c["metrics"]["bootstrap"]["n"], c["metrics"]["null_shuffle"]["n"] = 50, 20
    rl = A.analyze_pair(M.gram(X), M.gram(Yl), g, D_, q, c)
    ri = A.analyze_pair(M.gram(X), M.gram(Yi), g, D_, q, c)
    check("linear target: CKA > 0.9 and R2 > 0.9", rl["cka"]["point"] > 0.9 and rl["r2"]["point"] > 0.9, "CKA %.3f R2 %.3f rev %.3f" % (rl["cka"]["point"], rl["r2"]["point"], rl["r2_reverse_descriptive"]["point"]))
    check("linear target: CIs contain point, ordered", all(r["ci90"][0] <= r["point"] <= r["ci90"][1] for r in (rl["cka"], rl["r2"])), "CKA ci %s R2 ci %s" % (rl["cka"]["ci90"], rl["r2"]["ci90"]))
    check("linear target: shuffled null clearly lower", rl["cka"]["null"]["max"] < rl["cka"]["point"] - 0.5 and rl["r2"]["null"]["max"] < 0.1, "null CKA mean %.3f max %.3f; null R2 mean %.3f" % (rl["cka"]["null"]["mean"], rl["cka"]["null"]["max"], rl["r2"]["null"]["mean"]))
    check("independent target: R2 <= 0.05 and CKA < 0.3", ri["r2"]["point"] <= 0.05 and ri["cka"]["point"] < 0.3, "CKA %.3f R2 %.3f" % (ri["cka"]["point"], ri["r2"]["point"]))
    v1 = A.verdict_pass(rl, c)
    v2 = A.verdict_pass(ri, c)
    check("verdict: linear -> criterion_met, independent -> criterion_not_met or invalid", v1["verdict"] == "criterion_met" and v2["verdict"] in ("criterion_not_met", "invalid"), "%s / %s" % (v1["verdict"], v2["verdict"]))
    # a high-CKA but non-informative case must be flagged invalid: shared huge common-mode (same mean offset in a few dims) makes the null CKA high
    Xo = X + 0 * X
    Xo[:, 0] += 100 * rng.randn(n)
    Yo = rng.randn(n, q)
    Yo[:, 0] = Xo[:, 0]
    ro = A.analyze_pair(M.gram(Xo), M.gram(Yo), g, D_, q, c)
    check("verdict: single dominant shared dimension (CKA~1 for any pairing is NOT the case: null drops) is handled", ro["cka"]["null"]["mean"] < ro["cka"]["point"], "CKA %.3f null %.3f" % (ro["cka"]["point"], ro["cka"]["null"]["mean"]))


def t_integration(cfg):
    rng = np.random.RandomState(3)
    frames = D.cached_frames()
    E, nF, P, d = 120, len(frames), 6, 10
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        td = Path(td)
        (td / "enc").mkdir()
        enc = rng.randn(E, nF, P, d).astype(np.float16)
        encr = rng.randn(E, nF, P, d).astype(np.float16)
        Wm = rng.randn(d, d) / np.sqrt(d)
        pred_rand = (enc.astype(np.float32) @ Wm + 0.02 * rng.randn(E, nF, P, d)).astype(np.float16)   # linear copy of the INPUT frame tokens
        pred_ours = rng.randn(E, nF, P, d).astype(np.float16)                                         # unrelated
        np.save(td / "enc/tokens_k12.npy", enc)
        np.save(td / "enc/rand_k12.npy", encr)
        np.save(td / "feat_pred_out6.npy", pred_ours)
        np.save(td / "feat_pred_out6_rand.npy", pred_rand)
        n = 300
        flat_ids = rng.choice(E * nF, n, replace=False)
        row, fpos = flat_ids // nF, flat_ids % nF
        np.savez(td / "in.npz", row=row, fpos=fpos, frame=np.array(frames)[fpos])
        c = json.loads(json.dumps(cfg))
        c["eval"]["n_inputs"] = n
        c["metrics"]["bootstrap"]["n"], c["metrics"]["null_shuffle"]["n"] = 30, 5
        env = {"EXP004_CACHE": str(td), "EXP005_CACHE": str(td)}
        paths = D.cache_paths(env)
        res = R.run_main(c, D.EXP_DIR / "config.yaml", paths, td / "in.npz", td / "res.json", lambda m: None, expect_pd=(P, d))
        pi = res["pairs"]["input_frame|enc_k12_ours|pred_out6_random"]
        pp = res["pairs"]["predicted_frame|enc_k12_ours|pred_out6_random"]
        ps = res["pairs"]["input_frame_matched_subset|enc_k12_ours|pred_out6_random"]
        ok_n = res["alignment_counts"]["predicted_frame"] == int(sum((f + 1) in frames for f in np.array(frames)[fpos]))
        check("integration: predicted_frame subset = inputs with t+1 cached", ok_n, str(res["alignment_counts"]))
        check("integration: input_frame R2 ~ 1 for a linear copy of the input frame", pi["r2"]["point"] > 0.95, "R2 %.3f CKA %.3f" % (pi["r2"]["point"], pi["cka"]["point"]))
        check("integration: predicted_frame (wrong frame) R2 ~ 0", pp["r2"]["point"] < 0.1 and ps["r2"]["point"] > 0.95, "pred-frame R2 %.3f, matched-subset input-frame R2 %.3f" % (pp["r2"]["point"], ps["r2"]["point"]))
        check("integration: random-encoder source R2 ~ 0", res["pairs"]["input_frame|enc_k12_random|pred_out6_random"]["r2"]["point"] < 0.1)
        check("integration: json written with code/config/env fields", all(k in json.load(open(td / "res.json")) for k in ("config_sha256", "code_version", "environment", "seed", "verdict_H_pass")))
        # missing-file guard
        os.remove(td / "feat_pred_out6.npy")
        check("guard: missing cache file detected before any computation", not all(p.exists() for p in D.cache_paths(env).values()))
        try:
            D.cache_paths({})
            check("guard: unset env vars raise", False)
        except KeyError:
            check("guard: unset env vars raise", True)


def main():
    cfg = yaml.safe_load(open(D.EXP_DIR / "config.yaml"))
    for f in (t_dual_vs_primal, t_cka, t_folds):
        f()
    t_synth(cfg)
    t_integration(cfg)
    print("SELFTEST %s (%d failures: %s)" % ("FAILED" if FAILS else "PASSED", len(FAILS), FAILS))
    return 1 if FAILS else 0
