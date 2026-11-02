"""Q1 (H-a): is cloth size readable from the encoder output (DINOv2 patch tokens)? CPU/GPU-light, uses only the cache.
Probes: linear_pooled, attentive (same capacity settings as Q2/Q3, config probe_impl). Controls: random-init encoder, episode-level label
shuffle (training labels only), mask area/bbox ridge baseline. Probes are trained on probe-training episodes (levels data.train_levels_b) and
evaluated on test episodes of every level; unseen levels (interp/extrap) are never used for training. Ridge alpha / early stopping use
only the training episodes. Output: results/q1/cells.npz (per-episode statistics for aggregate) and results/q1/q1_summary.json (numbers only)."""
from __future__ import annotations

import time

import numpy as np

from .common import log, write_json, read_json
from . import cache as CA
from . import evalgrid as EG
from . import probes as PR
from . import stats as ST


def run(cfg, paths, dry_run=False, only_kinds=None):
    outdir = paths["results_dir"] / "q1"
    kinds = only_kinds or cfg["probe"]["kinds"]
    seeds = cfg["probe"]["seeds"]
    log("q1 plan: kinds=%s probe seeds=%s controls=%s targets=[%s, %s]" % (kinds, seeds, cfg["probe"]["controls"], cfg["q1"]["primary_target"], cfg["q1"]["secondary_target"]))
    if dry_run:
        return None
    C = CA.load(paths)
    rows = C["rows"]
    tr_rows, te_rows = EG.split_rows(rows)
    Etr, Ete = len(tr_rows), len(te_rows)
    assert all(r["part"] == "train" for r in rows[:Etr]) and all(r["part"] == "test" for r in rows[Etr:])
    levels = list(cfg["data"]["levels"])
    lvl_idx = EG.level_index(te_rows, levels)
    boot = ST.Bootstrap(lvl_idx, cfg["probe_impl"]["n_bootstrap"], cfg["probe_impl"]["bootstrap_seed"])
    rmasks = EG.role_masks(te_rows)
    nominal = np.array([r["nominal_log_area"] for r in te_rows])
    W = EG.windows(cfg)
    F = cfg["frames"]
    mf = np.asarray(C["mask_feat"])
    toks = {"dino": C["tokens_dino"], "rand": C["tokens_rand"]}
    cells, pms, summary = {}, {}, {}
    shuffle_base = int(cfg["probe_impl"]["shuffle_seed"])
    t0 = time.time()

    def add(key, S, pm):
        cells[key] = S
        pms[key] = pm
        summary[key] = EG.summarize_cell(S, boot, rmasks, pm, nominal)
        log("q1 cell %s done (%.0fs)" % (key, time.time() - t0))

    for target in (cfg["q1"]["primary_target"], cfg["q1"]["secondary_target"]):
        Y = EG.make_labels(cfg, rows, mf, target)
        Ytr, Yte = Y[:Etr], Y[Etr:]
        regimes = [("all", F["all"], {w: W[w] for w in ("early", "late", "all", "union")}),
                   ("early", F["early_flat"], {w: W[w] for w in ("early", "late")}),
                   ("union", W["union"], {"union": W["union"]})]
        for kind in kinds:
            for rep, tk, shuf in (("dino", toks["dino"], None), ("rand", toks["rand"], None), ("shuffle", toks["dino"], shuffle_base)):
                for regime, trf, ev in regimes:
                    if regime == "union" and rep != "dino":
                        continue
                    c, p = EG.fit_predict_cells(cfg, kind, tk[:Etr], Ytr, tk[Etr:], Yte, trf, ev, seeds, shuffle_base=shuf)
                    for w in c:
                        add("%s|%s|%s|%s|%s" % (target, kind, rep, regime, w), c[w], p[w])
            if target == cfg["q1"]["primary_target"]:
                for f in range(1, 13):                       # single-frame matched ceilings for the Q2 rollout steps (frame t+k)
                    c, p = EG.fit_predict_cells(cfg, kind, toks["dino"][:Etr], Ytr, toks["dino"][Etr:], Yte, [f], {"frame%d" % f: [f]}, seeds)
                    add("%s|%s|dino|f%d|frame%d" % (target, kind, f, f), c["frame%d" % f], p["frame%d" % f])
        feats = PR.mask_baseline_features(mf)
        for regime, trf, ev in regimes:
            c, p = EG.fit_predict_cells(cfg, "ridge_feat", feats[:Etr], Ytr, feats[Etr:], Yte, trf, ev, seeds)
            for w in c:
                add("%s|mask_ridge|mask|%s|%s" % (target, regime, w), c[w], p[w])
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "cells.npz", **{k: v for k, v in cells.items()}, **{"predmean|" + k: v for k, v in pms.items()})
    write_json(te_rows, outdir / "test_rows.json")
    write_json({"note": "numbers only; episode-bootstrap CI excludes training-seed variation (the encoder is frozen: no training-seed dimension in Q1)",
                "targets": {"primary": cfg["q1"]["primary_target"], "secondary": cfg["q1"]["secondary_target"]},
                "probe_seeds": seeds, "windows": W, "encoder_fingerprint": C["meta"]["encoder_fingerprint"], "n_train_eps": Etr, "n_test_eps": Ete,
                "cells": summary}, outdir / "q1_summary.json")
    log("q1 done: %d cells" % len(cells))
    return outdir
