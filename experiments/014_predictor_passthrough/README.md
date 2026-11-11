# Experiment: 014_predictor_passthrough
- Hypothesis: notes/hypotheses.md "014 Predictor pass-through check" (H-pass). Thresholds, both alignments and the pooling rule are user-confirmed (2026-11-11, before any 014 result). config.yaml was not changed by the experimenter.
- Question: is pred_out6 (005 predictor output) a near-linear copy of enc_k12 (004 encoder last-layer tokens, frozen DINOv2)? Follows up the 010 finding that pred_out6 random is about as good as enc_k12 ours on R-IB / D-IB.
- Success criterion (confirmed, unchanged): for (enc_k12 ours -> pred_out6 random): linear CKA >= 0.9 and CV ridge R2 >= 0.9 (point estimates; CI lower bounds reported alongside). Check is invalid if the shuffled-null CKA is not clearly below the observed CKA. Secondary, descriptive: pred_out6 ours vs pred_out6 random.
- What I (experimenter) claim from this experiment: only the numbers in results.json (CKA, R2, CIs, null stats per source x target x alignment, and the rule outcome for H-pass). No interpretation here; the claim "pred_out6 passes enc_k12 through" or its negation is made only after /review-experiment and a human decision.
- Status (2026-11-11): code written, selftest on synthetic data PASSED (logs/selftest_20261111.log). Main analysis NOT run: the 4 cache files are not on this machine yet (checked: `~/mylab_cache` does not exist, EXP004_CACHE / EXP005_CACHE unset). CPU only, no training.

## Cache layout (derived from the code; file:line, repo-relative to experiments/)
| item | finding | source |
|---|---|---|
| `$EXP004_CACHE/enc/tokens_k12.npy` (ours), `rand_k12.npy` (random) | float16 memmap, shape `(E, nF, 196, 384)` = (episode row, cached frame, patch token, DINOv2 dim); tokens after block k with the final `base.norm` applied, class/register tokens dropped; k=12 equals `x_norm_patchtokens` (checked at cache time against the 002 cache) | 004_depth_truncation/src/exp004/stages.py:130-132 (shape, names), :143-148 (fill), :153-160 (k=L consistency check); deep.py:31-50 (`tokens_by_k`) |
| `enc` dir name | `enc` + cache_suffix (`""` in 005/010) | 004 stages.py:75; 005 feats.py:19-20, 005 config.yaml:15 |
| `$EXP005_CACHE/feat_pred_out6.npy` (ours = 002 condition-b run, train seed 0), `feat_pred_out6_rand.npy` (random-init predictor, seed 1234+0) | float16, same shape `(E, nF, 196, 384)`; first 384 coordinates (visual z part) of the predictor early-exit output after 6 blocks (= final norm applied; equals the model output iff the predictor has 6 blocks, unverified here) | 005 feats.py:23-24 (names), :35 (shape `(E,nF,P,DV)`), :52-57 (`ee_forward`, `exit_out`, `last_frame_tokens`); 005 config.yaml:18-21 (train_seed 0, random_init_seed 1234); 005 stages.py:165 (`random_init_predictor(model, 1234+seed)`) |
| frame axis (nF = 8) | the same for encoder and predictor caches: sorted union of `frames.early_flat=[0,1,2,3]` and `frames.late_deformed=[8,9,10,11]` -> `[0,1,2,3,8,9,10,11]` (macro frame indices, frameskip 5; cached frames 4..7 and 12 do not exist) | 004 stages.py:50-53, :90; 002 evalgrid.py:34-37; 002 config.yaml:112, :126-130 |
| episode axis E | rows of the 002 Q1 cache, train rows first then test rows; both 004 and 005 use the same C["rows"] | 004 stages.py:93-95; 005 feats.py:31; 005 tasks.py docstring; 002 cache.py:97-105 |
| which frame a pred_out6 entry refers to | window = last-context frame t at position `fpos` (`frames[fpos] = t`), history H = `num_hist` (002 config.yaml:42: 1, so the window is the single frame t). The tokens kept are those at the LAST context-frame positions of the predictor output (`last_frame_tokens`: `h[:, (H-1)*196 : H*196]`). A frame-wise causal predictor output at the position of frame t is the prediction of frame t+1 (same slot the rollout stores as the next frame: `zp[:, -1:]`). So the entry `feat[row, fpos]` = predictor INPUT enc(frame t) -> OUTPUT latent for t+1. H is read from the checkpoint config at extraction time (005 stages.py:162); 002 config says 1, not re-verified here | 002 model_io.py:108-111, :120-124; 004 deep.py:114-119; 005 feats.py:46-57 |
| predictor input vs encoder cache | predictor input is the encoder's z (visual 384 dims + action/proprio dims) of frame t; the 384 visual dims are the 002 `tokens_dino` that 004 stages.py:153-160 checks `tokens_by_k(k=12)` against | 004 stages.py:150-160 |
| the 900 eval inputs | `(row, fpos)` pairs from `tasks.eval_inputs(..., n_eval=900, seed=1000)`: test rows only, half early frames [0..3], half late [8..11], without replacement. The pairs used by 010 heads are stored in `010_source_separation_ib/results/tasks_ts2000.npz` (`row`, `fpos`, `frame`; identical in ts2001, ts2002). 014 reads this file (path override env `EXP014_INPUTS`) so it does not need the 002 episodes.json. It has 900 inputs from 590 distinct rows (episodes), rows 900..1798, fpos covers 0..7 | 005 tasks.py:10-25; 005 stages.py:56-66; 010 stages.py:111-121 |
| how the heads saw the features | each head input = `feat[row, fpos]` = the full unpooled token set `(196, 384)` (training pool `feat[:Etr]` reshaped to `(Etr*nF,196,384)`, eval `feat[T["row"], T["fpos"]]`); the attentive head applies LayerNorm, a linear projection and attention pooling itself | 005 stages.py:173-180; 005 heads.py:22-35; 002 probes.py:39-58 |

## Pooling decision (config: token_pooling = same_as_005_heads)
Implemented as: no pooling. For each input the full token set `(196, 384)` is flattened to a 75,264-vector in token order (token position i of the source is paired with token position i of the target; same patch grid). Features are cast float16 -> float64 for Grams (heads cast `.float()`). The heads' internal per-token LayerNorm is part of the head, not of the features, and is not applied. Because the feature dimension (75,264) exceeds n (900), CKA and ridge are computed from n x n linear Grams (exact dual form; selftest checks dual == primal ridge).

## Alignments (config: report both)
- `input_frame`: source = enc tokens of the same (row, fpos) as the pred_out6 entry (frame t, the last context frame). All 900 inputs.
- `predicted_frame`: source = enc tokens of frame t+1 of the same row (position of t+1 in the cached frame list). Only evaluable for inputs whose t+1 is a cached frame: t in {0,1,2,8,9,10}; NOT for t=3 (4 not cached) and t=11 (12 not cached). The count is written to results.json (`alignment_counts`). This is a subset, not the full 900; if it were empty the alignment would be recorded as `not_evaluable`.
- `input_frame_matched_subset` (extra, descriptive, no null): `input_frame` restricted to the same subset as `predicted_frame`, so the two alignments can be compared on the same inputs.
(The 002 cache holds frames 0..12 in `frames_u8`; the 004 enc cache does not, so encoding t+1 is not done here: no GPU/model work.)

## Metrics (per alignment x source in {enc_k12 ours, enc_k12 random} x target in {pred_out6 ours, pred_out6 random})
- Linear CKA on mean-centered Grams (n inputs).
- Cross-validated ridge R2 (target from source): 5 folds, whole episodes (= cache rows) per fold, folds from `RandomState(seed)`; alpha chosen per outer fold by inner CV (3 inner folds, also by episode; the inner fold count is an experimenter default) over config grid [0.1, 1, 10, 100, 1000]; train-fold intercept. R2 = 1 - sum SSE / sum SST over all test samples and all output dims (variance-weighted), SST around the train-fold mean. Source features get one global scalar so that centered features have mean square 1 (fixes the meaning of the alpha grid; does not alter linear structure). Chosen alpha per fold and how many folds sit at a grid edge are reported.
- Shuffled-order null (100 permutations of the source rows over all inputs, folds unchanged): CKA and CV R2, both summarized (mean, sd, q95, min, max).
- Bootstrap 90% percentile CI (1000), unit = episode (rows resampled with replacement, all their inputs taken). R2 CI resamples the per-sample CV errors (no refit); CKA CI recomputes CKA on the resampled inputs (duplicated inputs stay duplicated).
- Reverse-direction R2 (source predicted from target) as a descriptive extra (no null).
- Verdict for H-pass, only for (enc_k12 ours -> pred_out6 random), per alignment: `criterion_met` iff CKA >= 0.9 and R2 >= 0.9 (point estimates; booleans for CI lower bounds >= 0.9 reported). `invalid` if observed CKA - null-mean CKA < 0.10 or observed CKA <= null 95th percentile (the 0.10 gap is an experimenter default for "clearly below"; the criterion thresholds are the confirmed ones). Otherwise `criterion_not_met`. Secondary: ours-minus-random differences of CKA and R2 are listed (descriptive, no test).
- Seeds: config `seed: 0` for folds (seed), inner folds (seed+1000+fold), null permutations (seed+1), bootstrap (seed+2).

## How to run
Environment: `~/miniconda3/envs/lab05` (numpy 2.2.6, PyYAML; no torch/sklearn needed; lab06 has no numpy, lab07 has no scipy). Nothing was installed. Only numpy + yaml are used.
Expected cache locations (suggested): `~/mylab_cache/exp004/enc/{tokens_k12.npy,rand_k12.npy}` and `~/mylab_cache/exp005/{feat_pred_out6.npy,feat_pred_out6_rand.npy}`.
```
cd ~/mylab/experiments/014_predictor_passthrough
PY=~/miniconda3/envs/lab05/bin/python
export EXP004_CACHE=~/mylab_cache/exp004      # contains enc/tokens_k12.npy, enc/rand_k12.npy
export EXP005_CACHE=~/mylab_cache/exp005      # contains feat_pred_out6.npy, feat_pred_out6_rand.npy
$PY run.py selftest          # synthetic data, ~2.5 min, no caches needed
$PY run.py check             # exits 2 and lists missing files if any cache/inputs file is absent
$PY run.py --smoke           # real files, 20 bootstrap / 3 null reps -> results_smoke.json (not a result)
$PY run.py                   # main run -> results.json, log in logs/main_<time>.log
```
Memory: about 2 GB RAM peak (four 900 x 75,264 float32 arrays are made one at a time, Grams kept). Estimated main-run time 30-40 min on 8 CPU cores (estimate from a synthetic n=900 timing: ~2.2 s per nested-CV ridge fit, 100 nulls x 8 source-target-alignment combinations, plus bootstraps); no GPU. Retries: max 3 per CLAUDE.md.
If the run stops with "input files missing" nothing was computed.
