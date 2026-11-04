"""CPU stage without a model: check the cloth-mask rule and the label definition on the raw videos of the *_test splits.

Records (numbers only, no interpretation):
  - per size: mean cloth-mask pixel count at frame `maskcheck.frame_for_area` (224x224), its ratio to the mean of the reference size
    (ind_test here; the user's README defines the ratio against ind_train mean, so this is a proxy), and the nominal README ratio
  - correlation of log(cloth_dimx*cloth_dimy) (particle count, from metadata.pt) with log(mask area) over all episodes:
    Pearson and Spearman, overall; within a size the particle count is constant by design so within-size correlation is undefined
  - fraction of episodes whose mask area is below mask.min_px
Mask thresholds are the HSV rule inherited from ACWM-Phys (config mask); whether it fits the PyFleX-regenerated renderer is exactly
what this stage measures (via the area ratios), not something assumed to be true.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import log, write_json, read_json
from .masks import cloth_area


def read_frame(path, idx, size):
    import cv2
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    return cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)


def corr_stats(particles, area):
    """Pearson / Spearman between log particle count and log area (arrays over episodes)."""
    from scipy.stats import rankdata
    x = np.log(np.asarray(particles, float))
    y = np.log(np.maximum(np.asarray(area, float), 1))
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return {"pearson": None, "spearman": None, "n": int(len(x))}
    return {"pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]), "n": int(len(x))}


def group_area_table(groups, areas, nominal, ref):
    """groups: array of group names per episode, areas: (E,) pixel counts at one frame. Returns per-group means and ratio to ref."""
    out = {}
    ref_mean = float(np.mean(areas[groups == ref])) if (groups == ref).any() else None
    for g in dict.fromkeys(groups.tolist()):
        m = groups == g
        mean = float(np.mean(areas[m]))
        out[g] = {"n": int(m.sum()), "mean_area_px": mean, "sd_area_px": float(np.std(areas[m])),
                  "ratio_to_ref_mean": None if not ref_mean else mean / ref_mean, "nominal_area_ratio": nominal.get(g)}
    return out


def run(cfg, paths, dry_run=False):
    mc = cfg["maskcheck"]
    blk = cfg["_ds"]
    root = Path(paths["data_root"])
    nominal = {s["name"]: s["area_ratio"] for s in blk["sizes"]}
    splits = [s for s in blk["eval_sets"]]
    log(f"maskcheck: splits={splits} root={root} frame={mc['frame_for_area']} episodes/size={mc['episodes_per_size']}")
    if dry_run:
        return None
    import torch
    from .common import group_of
    rows, grp, part = [], [], []
    for sp in splits:
        d = root / sp
        if not (d / "metadata.pt").exists():
            log(f"maskcheck: {d}/metadata.pt missing, skip {sp}")
            continue
        meta = torch.load(d / "metadata.pt", map_location="cpu", weights_only=True)
        if mc["episodes_per_size"]:
            meta = meta[: mc["episodes_per_size"]]
        for e in meta:
            fr = read_frame(d / Path(e["video_path"]).name, mc["frame_for_area"], mc["resize_to"])
            if fr is None:
                continue
            rows.append(cloth_area(fr, cfg["mask"]))
            grp.append(group_of(sp, Path(e["video_path"]).name, cfg))
            part.append(int(e["cloth_dimx"]) * int(e["cloth_dimy"]))
        log(f"maskcheck: {sp} done ({len(rows)} episodes so far)")
    areas, grp, part = np.array(rows, float), np.array(grp), np.array(part, float)
    out = {"frame": mc["frame_for_area"], "resize_to": mc["resize_to"], "n_episodes": int(len(areas)),
           "reference_group": blk["reference_group"],
           "area_by_group": group_area_table(grp, areas, nominal, blk["reference_group"]) if len(areas) else {},
           "frac_below_min_px": float((areas < cfg["mask"]["min_px"]).mean()) if len(areas) else None,
           "particles_vs_mask_area_overall": corr_stats(part, areas) if len(areas) else None,
           "particles_by_group": {g: float(np.mean(part[grp == g])) for g in dict.fromkeys(grp.tolist())},
           "note": "ratio_to_ref_mean uses the reference TEST split mean; the README ratio uses the ind_train mean. Mask = HSV rule inherited "
                   "from ACWM-Phys. Thresholds for 'good enough' are not set (human decision)."}
    write_json(out, paths["maskcheck_json"])
    log(f"maskcheck: wrote {paths['maskcheck_json']}")
    return out
