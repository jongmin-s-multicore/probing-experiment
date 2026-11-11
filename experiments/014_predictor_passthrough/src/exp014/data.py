"""Cache loading / alignment / Gram building for experiment 014 (see README for the layout derivation with file:line citations)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

EXP_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = EXP_DIR.parent / "010_source_separation_ib" / "results" / "tasks_ts2000.npz"
CFG002 = EXP_DIR.parent / "002_single_size_sweep" / "config.yaml"
FILES = {"enc_k12_ours": ("EXP004_CACHE", "enc/tokens_k12.npy"), "enc_k12_random": ("EXP004_CACHE", "enc/rand_k12.npy"),
         "pred_out6_ours": ("EXP005_CACHE", "feat_pred_out6.npy"), "pred_out6_random": ("EXP005_CACHE", "feat_pred_out6_rand.npy")}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cache_paths(env):
    """env: mapping (os.environ). Returns {name: Path} or raises KeyError listing the unset variables."""
    miss = sorted({v[0] for v in FILES.values() if not env.get(v[0])})
    if miss:
        raise KeyError("env var(s) not set: %s" % ", ".join(miss))
    return {k: Path(env[e]).expanduser() / rel for k, (e, rel) in FILES.items()}


def cached_frames():
    """Frame list of the 004/005 caches = sorted union of early_flat and late_deformed (exp002.evalgrid.windows 'union', 002 config frames)."""
    f = yaml.safe_load(open(CFG002))["frames"]
    return sorted(set(f["early_flat"]) | set(f["late_deformed"]))


def load_inputs(path):
    z = np.load(path)
    frames = cached_frames()
    row, fpos, frame = z["row"].astype(int), z["fpos"].astype(int), z["frame"].astype(int)
    assert (np.array(frames)[fpos] == frame).all(), "fpos/frame mismatch with 002 frames union"
    return row, fpos, frame, frames


def alignments(row, fpos, frame, frames):
    """Returns {name: (sel, fpos_source)} : sel = indices into the 900 inputs usable for the alignment; fpos_source = position in the frame axis of the
    encoder cache used as source for these inputs. input_frame: the last context frame itself (all inputs). predicted_frame: macro frame t+1 of the
    cache; only inputs whose t+1 is a cached frame (t=3 -> 4 and t=11 -> 12 are not cached)."""
    pos = {f: i for i, f in enumerate(frames)}
    ok = np.array([(f + 1) in pos for f in frame])
    nxt = np.array([pos.get(f + 1, -1) for f in frame])
    return {"input_frame": (np.arange(len(row)), fpos), "predicted_frame": (np.where(ok)[0], nxt[ok])}


def load_tokens(path, row, fpos):
    a = np.load(path, mmap_mode="r")
    return a, np.asarray(a[row, fpos])               # (n, P, d) float16 -> as heads: .float() later


def flat(x):
    return x.reshape(x.shape[0], -1)
