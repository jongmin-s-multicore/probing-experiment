"""Evaluation inputs and synthetic tasks of experiment 005 (numpy only, deterministic given config seeds).
Input = one cached frame of one episode (num_hist = 1, so a 'context window' is a single frame). State phi(x) = level; continuous state s = log nominal area ratio.
Index convention: cache rows are ordered [train rows (3 levels) ..., test rows (9 levels) ...] (exp002 cache). A sample is (row, fpos) with fpos the position in the
cached frame list `frames`; flat id = row * nF + fpos."""
from __future__ import annotations

import numpy as np


def eval_inputs(rows, frames, early_pos, late_pos, n_eval, seed):
    """n_eval inputs from test rows, half early frames / half late frames, uniform over test episodes (without replacement over (row, fpos)).
    Returns dict of arrays: row, fpos, frame, kind (0 early / 1 late)."""
    test_rows = np.array([i for i, r in enumerate(rows) if r["part"] == "test"])
    rng = np.random.RandomState(seed)
    out_r, out_f, out_k = [], [], []
    halves = [n_eval // 2, n_eval - n_eval // 2]
    for kind, (pos, n) in enumerate(((early_pos, halves[0]), (late_pos, halves[1]))):
        cand = np.array([(r, p) for r in test_rows for p in pos])
        if n > len(cand):
            raise ValueError("only %d candidate inputs for kind %d, %d requested" % (len(cand), kind, n))
        sel = cand[rng.choice(len(cand), n, replace=False)]
        out_r.append(sel[:, 0]); out_f.append(sel[:, 1]); out_k.append(np.full(n, kind))
    row, fpos, kind = np.concatenate(out_r), np.concatenate(out_f), np.concatenate(out_k)
    return {"row": row, "fpos": fpos, "frame": np.array(frames)[fpos], "kind": kind}


def train_pool(rows, nF):
    """All (row, fpos) of the train part as flat ids."""
    tr = [i for i, r in enumerate(rows) if r["part"] == "train"]
    return np.array([i * nF + p for i in tr for p in range(nF)])


def level_of_flat(rows, flat, nF, levels):
    return np.array([levels.index(rows[f // nF]["level"]) for f in flat])


def tasks_a(rows, nF, levels, n_tasks, n_per_task, pool_levels, seed):
    """5-A: per task a Bernoulli(0.5) label for each of the 9 levels (resampled if constant) and a dataset D_k of n_per_task flat ids from the train pool
    with at least one input per level that exists in the pool. Returns labels (T, 9) uint8, dk (T, n_per_task) flat ids, pool_levels_used."""
    pool = train_pool(rows, nF)
    plv = level_of_flat(rows, pool, nF, levels)
    avail = sorted(set(plv.tolist()) & set(levels.index(l) for l in pool_levels))
    pool = pool[np.isin(plv, avail)]
    plv = level_of_flat(rows, pool, nF, levels)
    labels = np.zeros((n_tasks, len(levels)), np.uint8)
    dk = np.zeros((n_tasks, n_per_task), np.int64)
    for k in range(n_tasks):
        rng = np.random.RandomState(np.random.SeedSequence([seed, k]).generate_state(1)[0])
        while True:
            lab = (rng.rand(len(levels)) < 0.5).astype(np.uint8)
            if 0 < lab.sum() < len(levels):
                break
        labels[k] = lab
        first = [rng.choice(np.where(plv == l)[0]) for l in avail]
        rest = np.setdiff1d(np.arange(len(pool)), first)
        extra = rng.choice(rest, n_per_task - len(first), replace=False)
        dk[k] = pool[np.concatenate([first, extra])]
    return labels, dk, [levels[l] for l in avail]


def sinusoid_params(n_tasks, J, f_lo, f_hi, seed):
    """Random sums of J band-limited sinusoids: g(s) = sum_j a_j sin(2 pi f_j (s - s0) + phi_j); f_j ~ U[f_lo, f_hi] cycles per unit of s."""
    P = []
    for k in range(n_tasks):
        rng = np.random.RandomState(np.random.SeedSequence([seed, k]).generate_state(1)[0])
        P.append({"f": rng.uniform(f_lo, f_hi, J), "phi": rng.uniform(0, 2 * np.pi, J), "a": rng.randn(J)})
    return {"f": np.stack([p["f"] for p in P]), "phi": np.stack([p["phi"] for p in P]), "a": np.stack([p["a"] for p in P])}


def sinusoid_eval(params, s, s_lo, s_hi, n_grid=2001):
    """g_k(s) for all tasks: (T, len(s)). Normalized per task to zero mean / unit std over a uniform grid on [s_lo, s_hi] (so distances are comparable across tasks)."""
    s = np.asarray(s, float)
    grid = np.linspace(s_lo, s_hi, n_grid)

    def g(x):
        arg = 2 * np.pi * params["f"][:, :, None] * (x[None, None, :] - s_lo) + params["phi"][:, :, None]
        return (params["a"][:, :, None] * np.sin(arg)).sum(1)
    gg = g(grid)
    mu, sd = gg.mean(1, keepdims=True), gg.std(1, keepdims=True)
    return (g(s) - mu) / np.maximum(sd, 1e-8)


def tasks_b(rows, nF, levels, train_levels, n_tasks, n_per_task, seed_dk):
    """5-B datasets D_k: n_per_task flat ids from train-pool inputs of the trained levels only."""
    pool = train_pool(rows, nF)
    plv = level_of_flat(rows, pool, nF, levels)
    pool = pool[np.isin(plv, [levels.index(l) for l in train_levels])]
    dk = np.zeros((n_tasks, n_per_task), np.int64)
    for k in range(n_tasks):
        rng = np.random.RandomState(np.random.SeedSequence([seed_dk, k]).generate_state(1)[0])
        dk[k] = pool[rng.choice(len(pool), n_per_task, replace=False)]
    return dk


def label_subset_d(rows, nF, levels, label_levels, n_labels, seed):
    """5-D: n_labels flat ids from the train pool restricted to `label_levels` (size-regression fine-tune labels)."""
    pool = train_pool(rows, nF)
    plv = level_of_flat(rows, pool, nF, levels)
    pool = pool[np.isin(plv, [levels.index(l) for l in label_levels])]
    rng = np.random.RandomState(seed)
    return pool[rng.choice(len(pool), min(n_labels, len(pool)), replace=False)]


def area_quantile_bins(area, n_bins):
    """Pseudo-state q(x): decile (n_bins) index of the visible cloth mask area (pixels) over the evaluation inputs."""
    edges = np.quantile(area, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, area, side="right")
