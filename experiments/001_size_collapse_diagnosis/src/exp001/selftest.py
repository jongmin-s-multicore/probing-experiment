"""CPU-only self test with synthetic data (no torch, no GPU, no dataset). Checks manifest logic, mask measurement, probe protocols,
diagnose tables and result writing. The synthetic numbers mean nothing; they only show that the code paths run."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np

from .common import write_json, log
from . import data as D, probes as P, probe_stage, diagnose
from .masks import cloth_area


def synth_cache(cache, rng, K=13, Dp=8):
    groups = [("smaller", 12, 0.26), ("ind", 16, 1.0), ("larger", 12, 2.5)]
    eps, scale = [], []
    for g, n, sc in groups:
        for i in range(n):
            eps.append({"dir": "x", "split": "ood_test" if g != "ind" else "ind_test", "name": f"{g}_{i}.mp4", "group": g, "rank_in_group": i + 1, "in_train": False})
            scale.append(sc * np.exp(rng.randn() * 0.03))
    E = len(eps)
    scale = np.array(scale)
    prof = 0.7 + 0.3 * np.sin(np.linspace(0, 2, K))
    area = (2700 * scale[:, None] * prof[None]).astype(np.int64)
    ly = np.log(area)
    W = rng.randn(384)
    W /= np.linalg.norm(W)

    def feat(l, shrink):
        return (rng.randn(E, l.shape[1], 384) * 0.5 + (shrink * (l - 7.9))[..., None] * W * 3).astype(np.float32)

    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / "scalars.npz", area_real=area, area_recon=area, area_tf1=(area[:, 1:] * 0.9).astype(int),
             area_ol=(area[:, 1:] * 0.5 * scale[:, None] ** -0.5).astype(int), cov_real=np.zeros((E, K, 196), np.float16))
    np.savez(cache / "pooled_mean.npz", gt=feat(ly, 1.0), tf1=feat(ly[:, 1:], 0.5), ol=feat(ly[:, 1:], 0.2), h0=feat(ly[:, :-1], 1.0),
             h3=feat(ly[:, :-1], 0.6), rvit=rng.randn(E, K, 384).astype(np.float32))
    np.savez(cache / "pix_grid.npz", pix=(rng.randn(E, K, 588) + ly[..., None] * 0.1).astype(np.float32))
    for n, k in (("gt", K), ("tf1", K - 1), ("ol", K - 1)):
        np.save(cache / f"patch_{n}.npy", rng.randn(E, k, 196, Dp).astype(np.float16))
    write_json({"run_name": "synth", "epoch_loaded": 0, "episodes": eps, "n_episodes": E, "K": K, "pools": ["mean"], "code_commit": "synthetic"}, cache / "episodes.json")
    (cache / "DONE").write_text("ok\n")


def run(cfg):
    rng = np.random.RandomState(0)
    # 1. manifest logic with a fake listing
    listing = {"ind_train": [f"t{i}.mp4" for i in range(2000)], "ind_test": [f"i{i}.mp4" for i in range(100)],
               "ood_test": [f"smaller_{i:03d}.mp4" for i in range(50)] + [f"larger_{i:03d}.mp4" for i in range(50)]}
    man = D.build_manifests(cfg, listing, "/data/clothmove")
    assert man["baseline_single"]["n_episodes"] == 2000 and man["mix_ood3"]["group_counts"] == {"ind": 1950, "smaller": 25, "larger": 25}
    tr_keys = {(e["split"], e["name"]) for e in man["mix_ood3"]["episodes"]}
    tab = D.eval_episode_table(listing, cfg, "/data/clothmove")
    common = [r for r in tab if r["rank_in_group"] <= 25 and r["split"] == "ood_test"]
    assert not any((r["split"], r["name"]) in tr_keys for r in common), "mix training episodes overlap with the common eval half"
    log("selftest: manifest logic OK")
    # 2. mask measurement on a synthetic image (orange patch)
    import cv2
    img = np.zeros((224, 224, 3), np.uint8)
    hsv = np.zeros((224, 224, 3), np.uint8)
    hsv[50:100, 50:100] = (12, 200, 200)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    assert cloth_area(img, cfg["mask"]) == 2500, cloth_area(img, cfg["mask"])
    log("selftest: mask OK")
    # 3. probes + diagnose on synthetic cache
    c2 = copy.deepcopy(cfg)
    c2["diagnose"]["pools"] = ["mean"]
    c2["probe"].update(seeds=[0], n_bootstrap=30, protocols=["strat_cv", "logo", "gt_transfer", "upper"])
    c2["probe"]["models"]["mlp"]["max_iter"] = 30
    c2["diagnose"]["rollout_ks"] = [1, 4, 12]
    tmp = Path(tempfile.mkdtemp(prefix="exp001_selftest_"))
    paths = {"cache": tmp / "cache", "results_dir": tmp / "results"}
    synth_cache(paths["cache"] / "synth", rng)
    out = probe_stage.run(c2, paths, "synth")
    r = out["pools"]["mean"]["ridge"]
    summ = {}
    for proto in ("strat_cv", "logo", "gt_transfer", "upper"):
        for t in ("gt", "tf1", "ol"):
            if t in r[proto]:
                a = r[proto][t]["all"]
                summ[f"{proto}/{t}"] = {m: a[m]["mean"] for m in ("r2_overall", "r2_within_group", "acc3", "slope")}
    diag = diagnose.run(c2, paths, "synth")
    log("selftest: probe + diagnose OK (synthetic numbers, not results)")
    for k, v in summ.items():
        log(f"  {k}: " + ", ".join(f"{m}={'nan' if x is None else round(x, 3)}" for m, x in v.items()))
    return {"ok": True, "tmp": str(tmp)}
