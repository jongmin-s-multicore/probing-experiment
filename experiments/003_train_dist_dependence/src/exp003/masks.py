"""Cloth pixel-mask measurement. Same HSV rule as branch repro/eval/eval_acwm_phys_quant.py (Cloth Move):
orange (hue 8-16, S>120) + pink back side (hue 158-170, S>60), V>80. Thresholds come from config.mask.
Input frames are uint8 RGB at 224x224 (the model input resolution)."""
from __future__ import annotations

import cv2
import numpy as np


def _hue(hsv, lo, hi, s, v):
    h = hsv[..., 0]
    m = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return m & (hsv[..., 1] > s) & (hsv[..., 2] > v)


def cloth_mask(rgb, mcfg):
    """rgb: (H, W, 3) uint8 -> bool (H, W)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    o, p = mcfg["cloth_orange"], mcfg["cloth_pink"]
    return _hue(hsv, o["hue_lo"], o["hue_hi"], o["sat_min"], o["val_min"]) | _hue(
        hsv, p["hue_lo"], p["hue_hi"], p["sat_min"], p["val_min"]
    )


def cloth_area(rgb, mcfg):
    return int(cloth_mask(rgb, mcfg).sum())


def frames_area(frames_u8, mcfg):
    """(T, H, W, 3) uint8 -> (T,) int pixel counts."""
    return np.array([cloth_area(f, mcfg) for f in frames_u8], dtype=np.int64)


def patch_coverage(frames_u8, mcfg, grid=14):
    """(T, H, W, 3) -> (T, grid*grid) fraction of cloth pixels per patch cell (area pooling)."""
    out = []
    for f in frames_u8:
        m = cloth_mask(f, mcfg).astype(np.float32)
        out.append(cv2.resize(m, (grid, grid), interpolation=cv2.INTER_AREA).reshape(-1))
    return np.stack(out)
