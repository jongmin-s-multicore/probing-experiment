"""Cloth pixel-mask measurement. Source: experiments/001_size_collapse_diagnosis/src/exp001/masks.py (same HSV rule as the branch
repro/eval/eval_acwm_phys_quant.py Cloth Move), extended with bbox features. Thresholds come from config.mask.
Input frames are uint8 RGB at 224x224. cv2 is imported lazily (not available on the authoring machine)."""
from __future__ import annotations

import numpy as np


def _hue(hsv, lo, hi, s, v):
    h = hsv[..., 0]
    m = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return m & (hsv[..., 1] > s) & (hsv[..., 2] > v)


def cloth_mask(rgb, mcfg):
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    o, p = mcfg["cloth_orange"], mcfg["cloth_pink"]
    return _hue(hsv, o["hue_lo"], o["hue_hi"], o["sat_min"], o["val_min"]) | _hue(hsv, p["hue_lo"], p["hue_hi"], p["sat_min"], p["val_min"])


def mask_features(m):
    """bool (H,W) -> [area_px, bbox_w, bbox_h] in pixels (0 when empty)."""
    if not m.any():
        return np.zeros(3)
    ys, xs = np.where(m)
    return np.array([m.sum(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1], dtype=np.float64)


def frames_mask_features(frames_u8, mcfg):
    """(T,H,W,3) uint8 -> (T,3) [area_px, bbox_w, bbox_h]."""
    return np.stack([mask_features(cloth_mask(f, mcfg)) for f in frames_u8])
