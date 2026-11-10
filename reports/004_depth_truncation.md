# 실험 004_depth_truncation 보고서

- 작성: reporter, 2026-11-10
- 입력: experiments/004_depth_truncation/{README.md, CHANGES.md, config.yaml, results.json, results/summary.json, results/depth_r2.csv, results/colab/timing_main.json, analysis/{stats_out.txt, recompute_stats.py, fig_primary_vs_controls.png}}, notes/reviews/004_prior.md, 004_stats.md, 004_skeptic.md, notes/specs/exp4_5_spec.md 3절, notes/hypotheses.md (후보 I, 확정 기록 9, 12), notes/log.md (2026-11-09 의 004 항목들), reports/003_train_dist_dependence.md (형식).
- 이 보고서는 새 측정을 하지 않았다. 수치 출처 표기:
  - `rj:` = results.json, `sj:` = results/summary.json. 둘의 `summary` 블록은 리포터가 읽어 내용이 같음을 확인했다(`rj: summary` = `sj`). results.json 의 sha256(log.md 는 996b5f40..., 9982 바이트로 적음)은 해시 도구가 없어 계산하지 못했다(미확인).
  - `csv:<줄>` = results/depth_r2.csv 의 줄 번호. 리포터가 Grep 으로 직접 확인한 행은 주 곡선(linear, new, level ALL, late)의 seed 0 행뿐이다: encoder k=0 (76-78), k=12 (4756-4758), predictor m=6 real (10866-10868, 21526-21528), random-init (7486-7488, 18146-18148), 라벨 셔플 (5276-5278, 11126-11128), baseline pixel (5536-5538), mask_area_bbox (5796-5798). **이 행들은 학습 seed 0(또는 1) x probe seed 0 한 줄의 값이며 seed 평균이 아니다.** CSV 전체는 4.7MB 로 통째로 읽지 않았다.
  - `stats_out:§n` = analysis/stats_out.txt 의 절. 리포터가 §1, §2, §3, §12, §13 을 열어 stats 리뷰 1-2절의 서술과 일치함을 확인했다(§2 `mismatches 0`). 나머지 절(§4-§11)은 열지 않았고 stats 리뷰 인용이다.
  - `timing:` = results/colab/timing_main.json (리포터가 읽음).
  - `(prior/stats/skeptic n절)` = 해당 리뷰가 계산했거나 인용했다고 밝힌 값을 옮긴 것. 별도 표시가 없으면 리포터가 대조하지 않았다.
  - 그림(fig_heatmap.png, fig_curves.png, fig_primary_vs_controls.png)은 리포터가 열어보지 않았다. 이미지 서술은 prior 의 서술이다.
  - 코드(stages.py, depthlib.py, deep.py)는 리포터가 읽지 않았다. 코드 줄 번호 인용은 skeptic 의 인용이다.
- 표기: "추정" 또는 "미확인" = 분석가 또는 리포터가 확인하지 못한 주장. 세 분석가는 서로의 파일을 읽지 않고 독립 작성했다.
- 이 보고서의 어떤 제안도 승인이 아니다. 임계값이나 규칙을 바꾸자는 권고는 쓰지 않았다. 사람이 결정할 사항은 8절에 모았다.
- 실행 사실(리드가 전달, 사용자 확인 2026-11-09): 본 실행은 2026-11-09 Colab NVIDIA L4, 학습 시드 3 x probe 시드 5 이다. 결과의 `pilot: true` 는 config 의 `pilot: true` 를 그대로 둔 채 노트북의 `ALLOW_MAIN_WITH_PILOT_TRUE` 를 켜서 본 실행했기 때문이다. `rj: summary.notes` 의 "pilot=true marks a subset/partial run" 문구는 이 실행에는 맞지 않는다(skeptic I1).

---

## 1. 한 문단 요약

DINO-WM 조건 (b)(`multi_60_ind_160`, 002 가 학습)의 인코더를 앞에서 k 층(k = 0, 2, ..., 12)까지, predictor 를 앞에서 m 블록(m = 0..6)까지만 써서 각 깊이마다 천 크기 probe 를 새로 학습하고, R² 가 직전 최고값보다 0.2 이상 처음 떨어지는 "소실 깊이" 를 찾았다. 저장된 판정은 **H-d refuted (`no_drop_ge_threshold_at_any_depth`)** 이다. `k_star`, `m_star`, 모든 split 의 소실 깊이가 null 이고 `loss_ood_only` 는 false 이다(`sj`). 주 곡선(linear, new, real, late)의 최대 하락은 0.033(인코더 interp)로 임계값 0.2 의 약 1/6 이다(`stats_out:§1`). 세 분석가는 이 판정이 규칙대로 맞고 임계값 0.05~0.3 어디서도 바뀌지 않는다는 데 일치한다(주 곡선 한정). 그러나 세 명 모두 이 반증이 말하는 범위를 좁게 본다. 학습 전 predictor(random-init)도 같은 평평한 곡선을 내고(예: m=6 interp 학습 0.5476 대 random-init 0.5084, `csv:10867`, `csv:7487`, seed 0), k=0 의 패치 임베딩만으로도 extrap R² 가 0.788 이며, mask 면적 baseline 이 extrap 0.817 이다. 또 split 마다 R² 분모(라벨 분산)가 달라 `ind_ood_gap` 과 겹침 검정은 해석에 쓸 수 없다. 갈리는 곳은 이 반증의 강도(prior: "정보가 사라지지 않는다 이상의 의미 없음" / stats: "이 측정 설계에서 소실 미관측, 강한 증거 아님" / skeptic: "검정력이 없을 수 있어 양성 대조가 필요"), 저수준 면적 단서가 설명하는 정도, 보조 지표 (ii) 의 해석이다. 가설 H-d 는 이 실험에서 규칙상 반증되었으나, "DINO-WM 이 깊이 전체에서 크기를 보존한다" 로 읽을 근거는 아니다. 결과에는 절차상의 불일치도 남아 있다: pilot 플래그가 본 실행 판정에 붙어 있고, 임계값과 impl default 규칙에 확정 기록이 없고, 파일럿 aggregate 재시도 횟수가 log.md 와 로그가 다르다(6절). 확신 수준: 판정 재현과 "주 곡선에 0.2 이상 하락 없음" 은 높음(stats, skeptic, prior 모두), "학습된 predictor 가 크기를 보존한다" 는 일반화는 낮음, 소실이 없는 원인(잔차 경로, 면적 단서)은 추정이다. 분석가별 전체 확신도는 prior 중간, stats 중간, skeptic 중간~높음이다.

---

## 2. 배경과 가설

- 배경: 002 에서 predictor 출력의 크기가 rollout 후반에 수축하는 현상이 보였고, 004 는 그 정보가 모델의 어느 깊이에서 사라지는지를 본다(hypotheses.md 후보 I, 명세 3절). 002 의 "last_block pass" 가 잔차 연결 때문에 실패하기 어려운 시험이라는 평가는 002 보고서에 있다(prior 2.1 인용; 리포터는 002 보고서를 열지 않았다).
- 가설(hypotheses.md 후보 I; 문장과 제안값은 2026-11-03 사용자 확정, 확정 기록 9):
  - **H-d**: 천 크기 R²(log 명목 면적비)는 어떤 깊이 L* 이후 급격히 줄고, OoD 크기에서 L* 가 InD 보다 앞선다. 소실 지점 = R² 가 직전 최고값보다 0.2 이상 처음 떨어지는 깊이(인코더 k*, predictor m* 따로). 주지표는 predictor 의 (i) hidden probe.
  - 반증: 모든 깊이에서 하락 < 0.2, 또는 InD/OoD 곡선이 seed 편차 안에서 겹침.
- 임계값 0.2 는 명세상 "파일럿에서 확정" 제안값이다(hypotheses.md 후보 I 301행). **확정 기록은 없다**(log.md 2026-11-09 기록의 주의 (b) 도 이를 인정). `overlap_sd_mult` 2.0, `overlap_depth_fraction` 0.8, `min_depths` 3, `primary` 설정은 명세에 값이 없는 impl default 이다(config.yaml:37-40, CHANGES.md 13-14).
- 이 실험이 주장하지 않는 것: probe 정확도는 "읽을 수 있다" 까지만 말한다(README "이 실험으로 내가 주장하려는 것"). "사용된다" 는 주장하지 않는다.
- 4-C(깊이 재학습)는 사용자 결정으로 구현하지 않았다(CHANGES.md 2).

---

## 3. 설정과 방법 (config 기준)

| 항목 | 값 | 출처 |
|---|---|---|
| 대상 모델 | 002 조건 `multi_60_ind_160`(명세의 조건 (b)), 학습 seed 0, 1, 2 체크포인트. 인코더는 동결이라 학습 seed 무관(`train_seed` NA) | config.yaml:6-9, 26; `csv` |
| 인코더 깊이 격자 | k = 0, 2, 4, 6, 8, 10, 12 (거친 격자). `encoder_layers` 12 | config.yaml:19; `sj: encoder_layers` |
| predictor 깊이 | m = 0..6 (`predictor_layers` 6). 모델에서 읽음 | `sj: predictor_layers`; CHANGES.md 3 |
| probe | linear_pooled(ridge), attentive. probe seed 0-4 (5개). 설정은 002 `probe_impl` 그대로 | config.yaml:14-16; CHANGES.md 7 |
| predictor 입력 | `feature_part: z_only`(앞 384 좌표). 002 기본은 full(404). 블록 1 이후 앞 384 좌표가 z 부분공간이라는 보장은 없음 | config.yaml:27; CHANGES.md 10 |
| 통제 | random-init 인코더, random-init predictor(은닉 probe (i) 만, `random_init_seed` 1234 + train_seed), 라벨 셔플. baseline: raw pixel, mask 면적/bbox, reference(인코더 토큰에 실험 1 probe) | config.yaml:21-22, 28-30; CHANGES.md 17-19 |
| 주 곡선 | linear, fit=new, control=real, timepoint=late. 곡선은 probe seed(x 학습 seed) 평균 | config.yaml:37; CHANGES.md 13 |
| 소실 정의 | 평균 R² 가 직전 최고값보다 `drop_threshold` 0.2 이상 낮아지는 첫 깊이, 없으면 null | config.yaml:36; `sj: definitions` |
| 겹침 규칙 | train 과 OoD 평균 곡선의 차가 2.0 x max(sd) 이내인 깊이의 비율이 0.8 이상이면 겹침 | config.yaml:38-39; CHANGES.md 14 |
| 분할 | train = {a060, ind, a160}, interp = {a080, a125}, extrap = {a026, a040, a200, a248} (skeptic C2 가 002 config 14-15 와 명세로 확인). `level == ALL` 행의 R² 는 해당 split 안의 라벨 분산을 분모로 쓴다 | skeptic C2; CHANGES.md 15 |
| 시드 | 전역 seed 0, probe seed [0..4], 학습 seed [0, 1, 2]. 명세에는 "pilot: true 는 학습 seed 0 만" 이 있으나 본 실행은 3개 | config.yaml:3, 16, 26 |
| 실행 환경 | Colab NVIDIA L4 (23.66 GB), Python 3.9.25, torch 2.0.1+cu117, conda 미사용(venv), sklearn null | `rj: environment` |
| 코드 버전 | `rj` 에 코드 commit 해시 없음. config sha256 은 있음(`a3f0d362...`). 실행 코드는 노트북 내장 스냅샷 | `rj: config_sha256`; skeptic I7 |
| 출력 행 수 | depth_r2.csv 37830 행 | `rj: n_rows_depth_r2` |

실행에 관한 주의:
- 단계별 시간(timing): enc_cache 176.6 s, enc_probe 1286.8 s, pred 5345.3 s, aggregate 10.1 s. 모두 rc 0, 본 실행 aggregate 종료는 11:20:44 UTC(`timing: 004:aggregate`). 같은 노트북이 005 를 이어 실행했고 005 plan 시작은 11:20:46 UTC 이다(`timing: 005:plan`).
- config 의 `pilot: true` 가 본 실행에도 그대로 있다(config.yaml:4 의 주석은 "본 실행 후 사람이 false 로 바꾼다" 라고 적는다). `pilot` 필드는 `rj` 3행과 `sj` 모두 true.
- 인코더 절단은 `get_intermediate_layers` 가 아니라 직접 루프(블록 순차 + 최종 norm)이다(CHANGES.md 12). 첫 배치 일치 검사(k=12 대 002 tokens_dino 0.017, get_intermediate_layers 0.003, random-init 0.002, 기준 0.05)는 skeptic S1 이 로그에서 인용했다(리포터는 로그를 열지 않았다).
- 004 의 `summary.json` 은 005 의 분기 입력이다. 리드가 확인한 바로는 005 는 `no_vanishing` 행을 따라 `pred_out6`(predictor 출력)을 층으로 선택했다(log.md 2026-11-09, 005 plan 행).

---

## 4. 결과

CI 가 아니라 seed 평균과 sd 이다. **linear 인코더 곡선의 "seed" 5개는 사실상 반복이 아니다**(4.5절). predictor 의 독립 단위는 학습 seed 3개이다.

### 4.1 저장 판정 (`sj`, `rj: summary`)

| 항목 | 값 | 출처 |
|---|---|---|
| `verdict_H_d.status` | refuted | `sj: verdict_H_d.status` |
| `verdict_H_d.reason` | no_drop_ge_threshold_at_any_depth | 같음 |
| 모듈별 `any_drop_ge_threshold` | encoder false, predictor false | `sj: verdict_H_d.per_module` |
| `k_star`, `m_star` | null, null | `sj` |
| split 별 `k_star_by_split`, `m_star_by_split` | train/interp/extrap 모두 null | `sj` |
| `vanish_detail` | 모듈 x split 6칸 모두 `evaluated: true`, `n_depths: 7`, `vanish: null` | `sj: vanish_detail` |
| `loss_ood_only` | false | `sj` |
| `l_best` | predictor depth 6, `l_best_value` 0.7021583626199643 (interp+extrap 평균 R², 주 설정) | `sj: l_best` |
| `l_best_by_module` | encoder depth 10 = 0.6912602337028997, predictor depth 6 = 0.7021583626199643 (차 0.0109) | `sj: l_best_by_module` |
| `refine_suggestion` | {} (빈 dict) | `sj` |
| `drop_threshold`, `overlap_sd_mult`, `overlap_depth_fraction` | 0.2, 2.0, 0.8 | `sj` |
| `pilot` | true (위 실행 사실 참조) | `sj`, `rj` |

- H-d 의 두 번째 반증 조건(겹침)은 판정 사유로 쓰이지 않았다. `reason` 이 첫 조건(하락 < 0.2)이다. `ind_ood_gap` 의 `overlap` 은 4모듈 x split 4칸 모두 false 이다(`sj: ind_ood_gap`). 후반부("OoD 에서 L* 가 앞당겨짐")는 소실이 없어 검정되지 않았다(stats 8절 3번).

### 4.2 주 곡선과 최대 하락 (linear, new, real, late, level ALL)

평균 R² 는 깊이 순서이며 인코더는 k = 0, 2, 4, 6, 8, 10, 12, predictor 는 m = 0..6 이다. 출처 `stats_out:§1` (리포터가 열어 확인; stats 리뷰 3절 표와 같음).

| 모듈 | split | 깊이별 평균 R² | 최대 하락 (직전 최고 대비) |
|---|---|---|---|
| encoder | train | 0.7329, 0.7925, 0.8256, 0.8404, 0.8373, 0.8462, 0.8342 | 0.012 |
| encoder | interp | 0.2174, 0.4275, 0.4655, 0.4904, 0.5337, 0.5419, 0.5088 | 0.0331 |
| encoder | extrap | 0.7879, 0.8238, 0.8218, 0.8398, 0.8359, 0.8407, 0.8516 | 0.004 |
| predictor | train | 0.8342, 0.8419, 0.8453, 0.8471, 0.8507, 0.8555, 0.8594 | 0 |
| predictor | interp | 0.5088, 0.5193, 0.5310, 0.5371, 0.5457, 0.5430, 0.5526 | 0.0027 |
| predictor | extrap | 0.8516, 0.8449, 0.8420, 0.8426, 0.8437, 0.8466, 0.8518 | 0.0097 |

- `stats_out:§12`: 학습 seed x probe seed 개별 시계열 중 최대 하락이 0.2 이상인 것은 0개(인코더 5개, predictor 15개 시계열 x 3 split; 최대 하락의 최댓값 0.04, 인코더 interp).
- 곡선은 대부분 깊이에 따라 오르거나 평평하다. 가장 큰 변화는 하락이 아니라 상승이다: 인코더 interp k=0 대 k=10 이 +0.3245(`stats_out:§13`, 같은 값 CI [0.3245, 0.3245]).
- 인코더 interp 의 k=10 -> 12 하락(0.5419 -> 0.5088)은 평균 기준 0.033 이다. prior 와 skeptic 은 seed 0 한 줄(0.5419 -> 0.5106; `csv:4757`)로 0.031 이라 적었다. 5개 probe seed 의 평균과 seed 0 의 차이이며 서로 다른 계산이 아니다(`csv:4757` 은 리포터가 확인, 평균은 `stats_out:§1`).

### 4.3 통제와 baseline 대비 (seed 0 한 줄, `csv`; seed 평균은 stats 리뷰 4절)

| 설정 | train | interp | extrap | 출처 |
|---|---|---|---|---|
| encoder k=0 real | 0.7329 | 0.2174 | 0.7879 | `csv:76-78` |
| encoder k=0 random-init | 0.4581 | -0.1474 | 0.6559 | `csv:336-338` |
| encoder k=12 real | 0.8336 | 0.5106 | 0.8524 | `csv:4756-4758` |
| encoder k=12 random-init | 0.6576 | 0.2072 | 0.7685 | `csv:5016-5018` |
| encoder k=12 라벨 셔플 | -0.0360 | -0.0421 | -0.0506 | `csv:5276-5278` |
| predictor m=6 real (학습 seed 0) | 0.8537 | 0.5476 | 0.8487 | `csv:10866-10868` |
| predictor m=6 random-init (학습 seed 0) | 0.8319 | 0.5084 | 0.8525 | `csv:7486-7488` |
| predictor m=6 real (학습 seed 1) | 0.8637 | 0.5619 | 0.8536 | `csv:21526-21528` |
| predictor m=6 random-init (학습 seed 1) | 0.8355 | 0.4948 | 0.8522 | `csv:18146-18148` |
| predictor m=6 라벨 셔플 (학습 seed 0) | -0.0376 | -0.0406 | -0.0433 | `csv:11126-11128` |
| baseline raw pixel (linear) | 0.3725 | -0.2520 | 0.6282 | `csv:5536-5538` |
| baseline mask_area_bbox | 0.5878 | -0.0006 | 0.8171 | `csv:5796-5798` |

- 학습된 predictor 의 깊이 0 대비 m=6 증가(학습 seed 3 x probe seed 5 평균, stats 4절 인용, `stats_out:§13` 로 일부 확인): train +0.025, interp +0.0438 (bootstrap 95% CI [0.0403, 0.0477]), extrap +0.0001 (CI [-0.0012, 0.0015]). stats 는 학습 seed 가 3개라 이 CI 는 좁게 나오므로 신뢰하지 말라고 적었다. predictor m=0 값은 학습 seed 와 무관하다(입력 z 가 동결 인코더 출력; stats 2절).
- predictor m=0 의 R²(0.8336)와 인코더 k=12(0.8336)가 소수 5자리까지 일치한다(`csv:7746` 대 `csv:4756`; skeptic S2 인용, 리포터는 인코더 쪽 값만 직접 확인). z 정렬이 맞다는 skeptic 의 근거이다.
- 같은 깊이에서 mask 면적 baseline 대비 우위(extrap, late, stats 4절): predictor 는 +0.025~+0.035(모든 깊이 양수), 인코더는 k=0 에서 -0.029, k=2, 4 에서 +0.005, k >= 6 에서 +0.02~+0.035. early 시점 mask extrap 은 0.889 이다(stats 4절 인용).

### 4.4 split 간 gap 과 겹침 (`sj: ind_ood_gap`; 재계산 `stats_out:§2` 28행 전부 일치, mismatches 0)

| 모듈, split | gap(train - split) 범위 | `fraction_depths_inside_seed_sd` | `overlap` |
|---|---|---|---|
| encoder, interp | 0.3036~0.5155 | 0.0 | false |
| encoder, extrap | -0.0550~0.0055 | 0.1429 | false |
| predictor, interp | 0.3049~0.3254 | 0.0 | false |
| predictor, extrap | -0.0175~0.0089 | 0.5714 | false |

- 값은 `sj` 의 `per_depth`. 범위는 리포터가 7개 깊이 값에서 읽은 최솟값과 최댓값이다.
- 인코더 interp 의 gap 은 모든 깊이에서 `inside_seed_sd: false` 이다. 이 칸들의 `tol` 이 0 또는 1e-16 이기 때문이다(`sj: ind_ood_gap.encoder.interp.per_depth`: 깊이 0, 2, 6, 10 이 0.0, 1.24e-16, 2.48e-16, 0.0). 이유는 4.5절.
- 해석 주의는 5절: gap 0.3~0.5 는 split 간 R² 분모 차이를 포함한다(prior, stats, skeptic 일치).

### 4.5 시드 구조 (stats 2절, skeptic I4; `stats_out:§2, §3, §3b` 리포터 확인)

- 인코더 linear 곡선은 probe seed 5개의 값이 깊이 0, 2, 10 에서 비트 단위로 같다(`stats_out:§3`: 깊이 0 에서 train 0.732886284192042 이 5개 모두 동일). 깊이 4(`stats_out:§3` 41행)는 5개가 두 종류로 갈린다(train 0.8260 x2, 0.8253 x3). linear 인코더 28개 (깊이 x split) 셀 중 16개가 sd < 1e-9 이다(`stats_out:§3b`: 16 / 28). attentive 는 0 / 28.
- 따라서 인코더 쪽 `tol` 이 0 이거나 1e-16 인 것은 구현 버그가 아니라 probe 가 결정적이기 때문이다(stats 2절). 이 경우 gap 이 정확히 0 이 아닌 한 `inside_seed_sd` 는 false 가 된다.
- predictor 의 sd 는 15개 값(학습 seed 3 x probe seed 5)으로 ddof=1 계산되는데 독립 단위는 학습 seed 3개이다(stats 2절, skeptic I4). skeptic 은 군집 3개, 크기 5 기준으로 sd 가 약 0.85배로 과소추정된다고 계산했다(추정 계산).
- 최종 판정(no_drop)은 sd 를 쓰지 않으므로 판정에 영향이 없다(stats 1절, skeptic I4).

### 4.6 임계값 민감도 (판정 변경 제안이 아니라 보고; stats 7절, `stats_out:§10` 은 열지 않았다)

- 주 곡선(linear/new/real/late): 임계값 0.30, 0.20, 0.15, 0.10, 0.05 모두 인코더, predictor, 3 split 소실 없음. 판정이 바뀌려면 임계값이 위 최대 하락(인코더 interp 0.033 등) 이하여야 한다(stats 7절). skeptic 은 0.05~0.2 범위에서 linear 판정이 같다고 확인했다(I2).
- 보조 곡선(attentive late)은 임계값 0.05 에서만 인코더 interp 깊이 12(하락 0.068), predictor interp 깊이 2(하락 0.079)가 잡히고 0.10 이상에서는 없다(stats 7절). skeptic 은 attentive predictor interp m=0 -> 1 의 0.253 -> 0.172(seed 0, 약 0.08)가 임계값 0.08 이하에서 잡힐 수 있다고 적었다(추정, seed 평균 미확인).

### 4.7 보조 지표 (ii), (iii) (판정 경로 밖)

- (iii) 다음 z MSE 는 깊이에 대해 단조 감소: real late train 1.263 (m=0) -> 0.265 (m=6), extrap 1.265 -> 0.284 (stats 6절, 학습 seed 3 평균, `next_z_mse.csv`; 리포터는 이 파일을 열지 않았다).
- (ii) 실험 1 probe 를 조기 종료 예측에 그대로 적용한 값은 얕은 깊이에서 음수가 크다: m=0 train -4.50(prior 1.3, `csv` seed 0 행 인용), -4.13(stats 6절, linear late, 학습 seed 3 평균 추정). m=6 에서는 train 0.82, interp 0.51, extrap 0.81(stats 6절). 두 값의 차이는 계산 단위(seed 0 한 줄 대 평균)로 보이나 리포터는 확인하지 않았다.
- (ii) linear extrap 의 m=5 -> 6 하락은 late 에서 0.053, rollout t+8 에서 0.196(0.832 -> 0.675)이고 임계값 0.2 에 근접하나 미만이다(stats 6절). attentive (ii) train rollout t+8 은 m=4 0.814 -> m=6 0.517(하락 0.297)로 0.2 를 넘는다. stats 는 이것이 사전 고정 주 곡선 밖이라 post-hoc 보고일 뿐이라고 적었다.
- skeptic I6: (ii) rollout t+8, m=6 의 R²(seed 0)가 train 0.754, interp 0.443, extrap 0.672 이고 같은 시점 인코더 토큰 reference 는 0.890, 0.612, 0.878 이다(`csv:13856-13858`, `csv:11581-11583` 인용; 리포터가 대조하지 않았다).

---

## 5. 분석가 의견 종합

### 5.1 세 명이 일치한 점

1. **저장 판정은 규칙대로이고 주 곡선에 0.2 이상 하락이 없다.** prior 1.1(seed 0 CSV 행), stats 1절(`recompute_stats.py` 로 `ind_ood_gap` 28행 전부 재현, 불일치 0), skeptic 요약과 6절(CSV 행 인용). 최대 하락은 약 0.03(인코더 interp)이다.
2. **임계값 0.2 가 비확정이어도 주 곡선 판정은 바뀌지 않는다.** prior 1.3, 6절(추정), stats 7절(0.05~0.3 확인), skeptic I2(0.05~0.2). 이유는 최대 하락이 임계값보다 훨씬 작기 때문이다.
3. **학습된 predictor 와 random-init predictor 의 곡선이 거의 같아, 이 실험의 "소실 없음" 은 학습의 성질이라기보다 입력 z 가 잔차 경로로 흘러온 것과 일치한다.** prior 2.1(학습 기여는 train +0.02, interp +0.04 수준; 추정), stats 4절(`stats_out:§13` 의 +0.0438, +0.0001), skeptic C1(CSV 행 7746-7748 대 5926-5928 등). 인코더에서도 random-init 곡선이 평평하다(prior 2.1, skeptic C1; 리포터가 `csv:336`, `csv:5016` 로 k=0 0.4581, k=12 0.6576 확인).
4. **판정이 말하는 범위는 "읽을 수 있음" 이고 "사용됨" 이 아니다.** prior 2.2, stats 1절("이 측정 설계에서 소실이 관측되지 않는다" 일 뿐), skeptic I6("teacher-forced 은닉 상태에서 깊이 방향으로 linear 판독이 0.2 이상 떨어진다" 는 좁은 명제만 반증).
5. **R² 의 split 간 비교는 라벨 분산 분모 차이를 포함한다. `ind_ood_gap` 의 interp gap(0.30~0.52)과 `overlap`(false)은 일반화 실패나 곡선 분리의 증거로 읽으면 안 된다.** prior 3.2(002 보고서 인용; 추정), stats 3절(레벨 행의 pooled-variance R² 는 a080 0.95~0.97, a125 0.89~0.95; 추정), skeptic C2(명세 면적비로 split 안 분산 train 약 0.160, interp 약 0.050, extrap 약 0.962 를 계산; MAE 는 interp 0.119, train 0.128, extrap 0.289 로 R² 와 순서가 뒤집힘; `csv:4756-4758`).
6. **저수준 면적 단서가 크기 판독을 상당 부분 설명한다.** k=0 의 train 0.733 / extrap 0.788, mask_area_bbox baseline 의 extrap 0.817. prior 3.1, stats 4절, skeptic I5. 세 분석가는 정도에 대해 의견이 다르다(5.2).
7. **H-d 의 두 번째 반증 조건(겹침)은 이 데이터에서 구조적으로 판별력이 없다.** prior 3.2/6절(분모 효과), stats 8절 2번(인코더 tol ≈ 0, 척도 불일치), skeptic I4(`curves_overlap` 의미 없음). 판정 사유에는 쓰이지 않았다.
8. **002 에서 보인 rollout 수축은 이 실험의 teacher-forcing 한 step 깊이 축으로는 설명되지 않는다.** prior 3.4, skeptic I6/7절 4번. stats 는 직접 쓰지 않았다(8절의 한계에서 rollout 은 (ii) 만으로 다룸). 따라서 이 항목은 prior 와 skeptic 일치, stats 는 미언급이다.

### 5.2 의견이 갈린 점 (하나로 합치지 않음)

| 쟁점 | analyst-prior | analyst-stats | analyst-skeptic |
|---|---|---|---|
| **"refuted" 의 강도** | 규칙상 유효. 의미는 "정보가 깊이에서 사라지지 않는다" 이상이 아니며 "predictor 가 쓴다" 의 뜻이 아니다. 잔차 구조가 있으면 읽힌다는 확인으로 읽는 편이 안전(2.1, 2.2). 확신 높음(7절) | 재현은 확실. "크기 정보가 모든 깊이에서 유지된다는 강한 증거는 아님". mask / random-init 과의 차이가 작아(extrap +0.03) "유지된다" 는 일반화는 약하게만 지지(9절) | **검정에 특이성이 없다(치명, 확인).** random-init 도 "소실 없음" 이므로 H-d 가 참이든 거짓이든 구분하게 설계되지 않았다. 양성 대조(블록 m 에서 크기 방향을 인위로 제거)에서 0.2 소실이 검출되지 않으면 "검정력 없는 결과" 라고 함(C1, 추가 실험 2). 규칙 적용 자체는 맞음 |
| **저수준 면적 단서가 설명하는 정도** | **부분 지지(추정).** pixel baseline(extrap 0.628, train 0.373, interp -0.252)이 낮아 "픽셀 평균만으로는 설명되지 않는다". mask baseline 이 인코더와 거의 같은 점은 일관(3.1). 확신 낮음 | extrap 에서 mask baseline 대비 우위가 R² 단위 최대 +0.035 로 작아 "유지된다는 결론은 면적 baseline 이 이미 대부분 설명하는 범위 안". 인코더 random-init 이 k >= 2 에서 0.64~0.77 이라 R² 의 큰 부분은 사전학습에 의존하지 않는다(4절) | **대부분 설명된다(I5, 확인 + 추정).** "크기 정보가 살아남는다" 는 결과는 "면적 단서가 살아남는다" 와 구별되지 않는다. 그 내용이 면적인지 크기 개념인지는 이 실험으로 알 수 없다 |
| **보조 지표 (ii) 의 해석** | 얕은 깊이의 낮은 R² 는 소실이 아니라 읽기 방향/기저 불일치(조기 종료는 학습 때 없던 경로; tuned lens 구도, 문헌은 미확인). 소실이라고 말하면 안 됨(2.2). 확신 낮음 | 소실이 아니라 반대 방향의 현상(조기 종료 때문에 출력 공간과 어긋남). m=5 -> 6 의 마지막 층 하락(0.053 late, 0.196 rollout t+8)은 "확정된 소실이 아니고 출력 projection 이 마지막에만 쓰인 효과일 수 있다(추정)". attentive (ii) 0.297 하락은 post-hoc, 다중 비교 | m<6 의 (ii) R² 음수(m=0 train -4.50)는 편향 지배(Pearson 0.91, pred_mean 약 -0.94)의 예(S6). 반면 rollout t+8 에서 (ii) 와 reference 의 차가 외삽에서 0.2 이상일 수 있어 "깊이가 아니라 rollout 시간축에 수축 위치가 있을 수 있다(추정, seed 0 한 행)"(I6) |
| **l_best 와 005 의 층 선택** | `l_best_by_module` 의 차 0.011 이 per-depth tol(0.007~0.028) 규모라 사실상 동률. "l_best 가 predictor 라서 predictor 가 크기를 학습했다" 는 쓰면 안 됨(random-init predictor 가 거의 같은 값)(4.1) | l_best 가 predictor m=6 이라는 점을 재현했다고만 적음(3절). 층 선택의 의미는 언급 안 함 | `l_best` 는 interp 와 extrap R² 를 평균하는데 C2 때문에 split 간 척도가 다르다. 0.011 차는 predictor seed tol(0.007~0.014)과 비슷하다. 이 값이 005 의 층 선택 입력으로 들어갔다(C2) |
| **pilot 플래그와 판정의 지위** | 이 관점에서 판단하지 않음(3.6) | 사실로만 언급: "본 통계 결과는 이 둘(pilot, 임계값 비확정)에 민감하지 않음"(8절 6번) | **중요(I1, 확인).** README 10행 "파일럿에서는 점검용" 규칙대로면 저장된 refuted 는 점검용인데 `rj` 에는 확정 판정처럼 저장됨. 005 가 사람 검토 없이 `pilot=true` 로 표시된 004 summary 를 소비했다(`timing`: 004 aggregate 11:20:44, 005 plan 11:20:46) |
| **파일럿 -> 본 실행 사이의 임계값 확정** | 언급 안 함(임계값 비확정만 6절에 표시) | 판정에 영향 없음으로만 보고(7절) | 파일럿 aggregate 가 09:22:44 에 이미 `verdict_H_d=refuted` 를 냈고 본 실행 enc_cache 는 09:27:07 시작. 약 4분 간격, 결정 단계가 없었다. 영향은 절차상이고 수치에는 사소(I2) |
| **마지막 두 층의 소폭 하락 (인코더 interp)** | k=10 -> 12 의 -0.031 이 "마지막 층은 사전학습 목적에 특화" 가설과 방향은 맞지만 작아 **추정**일 뿐, 의미 있는 주장으로 쓰지 말 것(2.4) | k=10 -> 12 의 -0.033 은 probe-seed sd 0.004 에 비해 "통계적으로 분명하나 크기는 임계값의 1/6"(3절). 원인 해석은 하지 않음 | 완만한 감소를 임계 정의 때문에 놓쳤는가에 대해 "아니다, 감소 자체가 없다"(6절). 이 하락은 0.031 로 보고 별도 해석은 안 함 |
| **num_hist=1 의 영향** | predictor 가 한 프레임만 본다. 시간 attention 이 사실상 없어 입력 z 가 그대로 흐르는 것이 더 기대된다(3.5, 추정). DINO-WM 일반의 성질인지 변형의 성질인지 구분 안 됨 | 언급 안 함 | "predictor 의 시간 통합은 검증 대상에서 빠진다(추정)"(6절 마지막). 같은 사실을 다른 맥락으로 적음 |
| **전체 확신도** | 중간(판정과 수치는 확실, seed 평균과 코드 확인이 없고, 문헌은 초록 수준) | 중간(재계산 높음, 실효 시드 수 1 / 3, 일반화 낮음) | 중간~높음(C1, C2, I1, I3, I4, I5 는 직접 확인; 메커니즘, 분산 수치, I6 은 추정이 섞임) |

주의: prior 의 문헌 인용(Raghu 2021, Hewitt & Liang 2019, DINO-WM 초록 3편만 초록/요약 수준으로 확인)은 prior 가 "[확인-초록]" 으로 표시했고 나머지는 [미확인] 이다(prior 상단, 2.1-2.4, 5절). 리포터는 원문을 확인하지 못했으므로 이 보고서는 문헌을 근거로 쓰지 않는다.

### 5.3 한 명만 지적한 점

- prior 만:
  - 문헌 정합(5.2 주의 참조). 월드모델의 층별 소실 곡선을 잰 직접 선행연구를 찾지 못했다("찾지 못함" 은 검색 범위 한정, 2.3).
  - `feature_part: z_only` 가 block 1 이후 z 부분공간이라는 보장이 없어 predictor 의 hidden R² 가 하한에 가까울 수 있다는 우려(6절; 같은 사실은 CHANGES.md 10).
  - 빠진 baseline: persistence baseline, untrained predictor 의 (ii)(iii), 크기와 무관한 속성(그리퍼 위치, 천 중심)의 깊이별 곡선, 시간 정보 포함 mask baseline(5절).
  - 003/008 크기 끌림에 주는 의미: 한 step 안 깊이에서의 정보 손실로 끌림을 설명하기는 어렵고, 후보는 rollout 누적과 출력 공간 판독기 전이 실패(4.3, 추정). 003 분포별 모델에 같은 깊이별 probe 를 rollout step 별 은닉으로 적용하는 후속(4.3).
  - 005 의 D-IB 가 면적 휴리스틱으로 설명될 위험이 커지므로 5-C(`D-IB_area=` 대 `D-IB_area≠`)의 중요성이 올라간다(4.2).
- stats 만:
  - 위치별 시드 구조 계량: 인코더 linear 28셀 중 16셀 sd<1e-9, 깊이 0 의 predictor between-trainseed sd 0.0, 학습된 predictor 의 깊이 0 대비 증가의 bootstrap CI (2, 4절; `stats_out:§3b, §13` 일부 확인).
  - 임계값 격자 민감도 전체(0.30~0.05)와 attentive, early, (ii) 의 보조 곡선별 결과(7절; 리포터는 `stats_out:§10` 을 열지 않았다).
  - 레벨별 곡선(§11): 최대 하락은 a026 predictor 0.056, a026 encoder 0.051 이고 이 레벨은 R² 자체가 0.13~0.19 로 바닥(3절).
  - 다중 비교 경고: 보조 지표 다수(probe 2종 x 시점 5 x split 3 x 모듈 2)를 훑었으므로 보조 곡선의 우연한 큰 하락 가능(8절 7번). 보정은 안 했다.
  - attentive 곡선은 비단조적이고 sd 가 크다. attentive 의 깊이별 sd 는 스크립트가 따로 출력하지 않았다(5절).
- skeptic 만:
  - **I3 재시도 상한 위반과 기록 축소**: 5.4 와 6절 "표시만 하는 불일치" 1번.
  - **I7 재현성**: `rj` 에 코드 commit 해시 없음, 실행 코드는 노트북 내장 스냅샷이고 저장소의 figs.py, exp045_tools.py, 노트북, CHANGES.md, README.md 가 커밋되지 않은 수정 상태이므로(세션 시작 git status와 일치) 같은 코드인지 증거가 없다. 002 코드가 그사이 바뀌었다(커밋 545168e). 완화: log.md 200행에서 parts 로 aggregate 를 다시 계산해 summary 일치(사실 확인: log.md).
  - **S1-S7 (확인 범위에서 버그 없음 포함)**: 인코더 절단과 LayerNorm(버그 아님, 얕은 층에서 분포 어긋난 정규화 가능성은 추정), predictor 조기 종료 일치(m=L 경로와 model.predict 최대 차 0), random-init predictor 가 모든 파라미터를 초기화했는지 미확인(S3), 데이터 누수 발견 없음(S4; 단 probe 학습 라벨의 고유값이 3개뿐이라 사실상 3-클래스 회귀), reference 행이 학습 seed 수만큼 3번 복제(S5), attentive interp R² 가 바닥이라 0.2 하락 관측이 사실상 불가능(S7).
  - 추가 실험 제안 3개(CPU 재집계, 양성 대조, 면적 통제 판독과 rollout 은닉 probe; 7절에 옮김).
- 두 명만 (나머지 한 명은 언급 안 함):
  - stats, skeptic: 인코더 linear 곡선의 probe seed 가 사실상 반복이 아니고 `tol ≈ 0` 이라 `curves_overlap` 이 의미 없다(stats 2절, skeptic I4). prior 는 seed 0 한 줄만 보았고 이 점을 쓰지 않았다.
  - prior, skeptic: l_best 의 두 모듈 값 차가 seed tol 규모라 층 선택 근거로 약하다(5.2 표 4행).

### 5.4 skeptic I3 와 log.md 의 불일치 (표시)

- skeptic I3 은 파일럿 aggregate 실패 5회(09:15:15, 09:18:26, 09:19:20, 09:22:04, 09:22:26)와 `attempts_check` 거부 1회(09:20:35), 이후 성공(09:22:40)을 로그로 확인했다고 적는다. log.md 2026-11-09 항목(194행)은 "attempts aggregate 1" 이라고만 적었다.
- 리포터는 `logs/colab_20261109/` 의 파일 이름 목록에서 aggregate 로그 8개를 확인했다: 091515, 091826, 091920, 092035, 092204, 092226, 092240, 112037(본 실행). skeptic 의 5회 실패 + 1회 거부 + 1회 성공(파일 7개)과 시각이 대응한다. 로그 내용은 열지 않았다. 목록은 skeptic 의 개수와 모순되지 않는다(내용 확인 아님).
- 원인은 matplotlib 백엔드 문제이다(CHANGES.md 7: 커널의 `MPLBACKEND` 가 venv 하위 프로세스에 상속되어 `import matplotlib` ValueError). 코드는 이후 `MPLBACKEND=Agg` 를 무조건 설정하도록 수정되었다(CHANGES.md 7, log.md 194-195행). 본 실행 aggregate 는 rc=0, 10.1초(`timing: 004:aggregate`).
- 영향: skeptic 은 실패가 그림 생성 단계(stages.py 397행)에서 났고 summary.json 은 그 전에 쓰인다(393행)고 보아 summary 값은 오염되지 않았다고 추정한다. 실패한 실행은 results.json 을 쓰지 못했다.

---

## 6. 한계와 위험

- **검정에 특이성이 없다**: random-init predictor 와 random-init 인코더도 평평한 곡선을 낸다. `refuted` 는 규칙상 유효하지만 학습된 구조에 대한 주장을 뒷받침하지 않는다(skeptic C1; prior, stats 와 방향 일치). 통제는 구현됐으나 판정에 쓰이지 않았다(skeptic: depthlib.py 195행에서 판정은 `control = real` 곡선만 사용; 리포터가 코드를 읽지 않았다).
- **split 간 R² 척도 불일치**: `ind_ood_gap`, `curves_overlap_within_seed_sd`, `l_best` 의 interp+extrap 평균이 분모가 다른 값의 비교이다(5.1 5번). 0.2 하락은 MSE 로 환산하면 interp 약 0.010, train 약 0.032, extrap 약 0.19 에 해당한다는 skeptic 의 추정 계산이 있다(C2, 레벨 균등 가정). H-d 가 가장 관심을 두는 외삽에서 검정이 가장 둔하다.
- **실효 시드 수**: 인코더 linear 는 사실상 n=1, predictor 는 학습 seed n=3. "seed 5회, 평균과 표준편차" 는 형식상 채웠고 반복 변동을 거의 담지 않는다(5.3 두 명만).
- **저수준 단서와 천장**: extrap R² 는 y 범위가 넓고 mask baseline 이 0.817 이라 변별력이 낮다(stats 8절 5번).
- **측정 범위가 좁다**: 은닉 probe (i)은 rollout 시점에서 계산하지 않는다(CHANGES.md 6). 002 의 rollout 수축과 직접 연결되는 축은 이 실험의 판정에 들어가지 않았다(skeptic I6). 깊이 격자가 거칠어 짧은 구간 급락을 놓칠 수 있으나 평균 곡선이 완만해 가능성은 낮다(stats 8절 4번, 추정).
- **num_hist=1**: predictor 가 한 프레임만 본다. 004 결과가 DINO-WM 일반의 성질인지 이 변형의 성질인지 구분되지 않는다(prior 3.5, 추정).
- **보조 지표 다중 비교**: 탐색적이며 보정하지 않았다(stats 8절 7번).
- **005 로의 전파**: 004 의 `summary.json` 은 005 의 분기 입력이다. 005 는 `no_vanishing` 행을 따라 `pred_out6` 을 선택했다(리드 확인, log.md). 그 입력이 위 한계(random-init 과 구분되지 않는 곡선, 0.011 차의 l_best, pilot 플래그)를 가진 채로 사람 검토 없이 소비되었다(skeptic I1). 005 결과 해석에서 이 의존을 고려해야 한다. 005 의 결과 자체는 이 보고서에서 다루지 않는다.
- **미확인 목록**:
  - 학습 때의 코드 버전. `rj` 에 commit 해시가 없고, 저장소의 일부 파일이 커밋되지 않은 수정 상태이다. 실행된 노트북 스냅샷과 저장소 코드가 같은지 미확인(skeptic I7).
  - random-init predictor 의 모든 파라미터가 초기화됐는지(skeptic S3).
  - `rj` 의 sha256 (리포터가 계산하지 못함), 그림 3개, depth_r2.csv 의 seed 평균 중 `stats_out` 에 없는 부분.
  - prior 의 문헌 인용 대부분(prior 5절).

### 표시만 하는 불일치 (고치지 않음)

1. **파일럿 aggregate 재시도 횟수**: skeptic I3 은 실패 5회 + 거부 1회, log.md 는 "attempts 1". CLAUDE.md 의 재시도 상한 3회와 맞지 않으며 기록이 축소되었다. 어느 쪽이 맞는지 리포터는 로그 내용으로 대조하지 못했다(5.4).
2. **`pilot: true` 와 본 실행**: `rj`, `sj` 의 `pilot: true` 와 `notes` 의 "subset/partial run" 은 본 실행(전체 캐시 1800 episode, 학습 seed 3)에 맞지 않는다. README 10행 "코드가 만드는 판정은 파일럿에서는 점검용이다" 규칙과 저장된 refuted 의 지위가 충돌한다(skeptic I1). 사용자가 `ALLOW_MAIN_WITH_PILOT_TRUE` 를 켠 것은 확인되었으나 config 의 pilot 값을 바꾸는 시점은 결정되지 않았다.
3. **임계값 및 impl default 의 확정 기록 없음**: drop 0.2, overlap 규칙(2.0, 0.8), `min_depths` 3, `primary` 는 "파일럿 후 확정" 제안값 또는 impl default 이다. 파일럿 결과를 본 뒤 확정 단계가 없었다(skeptic I2, log.md 주의 (b)). 주 곡선 판정은 둔감하다(4.6).
4. **통제 곡선을 판정에 쓰지 않음**: README 10행 "통제 포함" 과 달리 판정은 `control = real` 곡선만 쓴다(skeptic C1). 사전에 정한 규칙이므로 이탈은 아니나, 통제 결과가 판정에 반영되지 않는다.
5. **인코더 interp 최대 하락 표기**: prior, skeptic 의 0.031 은 seed 0 한 줄(`csv:4757`), stats 의 0.033 은 5 probe seed 평균(`stats_out:§1` 0.0331). 같은 현상의 다른 요약이며 불일치로 보지 않으나 인용 시 구분이 필요하다.
6. **(ii) m=0 의 train R² 표기**: prior 의 -4.50(seed 0 행 인용)과 stats 의 -4.13(linear late 평균으로 보임)이 다르다. 계산 단위 차이로 보이나 확인하지 않았다.
7. **README 상태 문구**: README 3행은 "Colab 본 실행 완료(2026-11-09), 결과 복사, 리뷰 전" 으로 갱신되어 있으나 `rj: interpretation` 은 null 이다. `rj: summary.notes` 문구가 실행 범위와 맞지 않는다는 점은 위 2번.

---

## 7. 다음 단계 후보 (제안만, 사전 등록 값은 바꾸지 않는다)

저비용 또는 새 학습 없음:
1. **통제 대비 곡선과 단위 통일 재집계(CPU, 기존 depth_r2.csv 만 사용)**: real 에서 random-init 을 뺀 차이 곡선으로 소실을 정의하고, MAE/MSE 또는 전체 평가 episode 공통 분산 R² 로 InD/OoD 를 비교하고, overlap 의 분산 단위를 학습 seed 3개로 둔다. 임계값은 결과를 보기 전에 기록한다(skeptic 추가 실험 1). 사후 분석으로 표시해야 한다.
2. **보조 곡선의 post-hoc 표시 정리**: (ii)(iii) 와 attentive 곡선을 판정 경로 밖 탐색 결과로 분리해 표기(stats 6, 8절).
3. **문서와 기록 정리**: `rj: summary.notes` 문구와 `pilot` 필드 설명을 실행 사실과 맞게 기록, results.json 에 코드 sha 보강(재계산 아님; skeptic I1, I7). log.md 의 재시도 횟수 정정 여부.

GPU 필요 또는 사람 승인 필요:
4. **양성 대조(검출력 확인)**: 블록 m 에서 크기(면적) 방향을 인위로 제거/상쇄하는 개입을 넣은 predictor 또는 잔차 없는 변형을 같은 파이프라인에 통과시켜 0.2 소실이 검출되는지 확인한다(skeptic 추가 실험 2). 검출되지 않으면 이번 refuted 는 검정력 없는 결과가 된다(skeptic).
5. **면적 통제 판독과 rollout 축 은닉 probe**: 라벨을 mask 면적으로 잔차화한 "면적 외 크기" 타깃, 면적이 같고 크기가 다른 조건의 판독, 은닉 probe (i)을 rollout t+4, t+8 의 은닉 상태에서도 측정(skeptic 추가 실험 3, prior 4.3). prior 는 003 의 분포별 모델에 같은 깊이별 probe 를 rollout step 별 은닉으로 적용하는 것이 가장 정보량이 큰 후속이라고 제안(추정; 004 코드로 새 학습 없이 가능하다고 적음).
6. **빠진 baseline 추가**: persistence baseline, untrained predictor 의 (ii)(iii), 크기와 무관한 속성(그리퍼 위치 등)의 깊이별 곡선, `feature_part: full` 과의 비교(prior 5절, CHANGES.md 10).
7. **새 후속 가설을 세우는 경우**: 소실 정의, 임계값, 단위(split 간 척도)를 결과를 보기 전에 글로 고정하는 새 사전 등록으로만 한다. 지금의 판정 기준을 결과 후에 바꾸지 않는다. 이 항목은 현재 004 판정을 바꾸는 제안이 아니다.

---

## 8. 사람이 결정해야 할 사항

1. config 의 `pilot` 값과 결과의 지위: 이 판정(refuted)을 점검용으로 둘지 본 실행 판정으로 취급할지, `rj` 의 `pilot: true` 와 `notes` 문구를 어떻게 표기할지. 임계값 0.2 와 impl default(overlap 2.0/0.8, `min_depths`, `primary`)의 확정 또는 비확정 유지(005 가 이 summary 를 이미 소비했음).
2. 4번(양성 대조)과 5번(rollout 은닉 probe 등) 후속의 GPU 사용 여부와 범위, 또는 1번(CPU 재집계)만 먼저 할지.
3. log.md 의 "attempts aggregate 1" 을 skeptic 의 5회 실패 + 1회 거부로 정정할지, 그리고 "refuted" 를 논문에서 "정보가 깊이에서 사라지지 않는다" 이상으로 쓰지 않는다는 범위 제한을 둘지.
