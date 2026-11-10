# 실험 003_train_dist_dependence 보고서

- 작성: reporter, 2026-11-07
- 입력: experiments/003_train_dist_dependence/{README.md, CHANGES.md, config.yaml, results.json}, analysis/{stats_review_output.json, per_run_slopes.csv, recon_18runs.csv, decoder_recon_by_epoch.json, 그림 3개, mix_adjacent_s0_real_vs_ol_sheet.png}, notes/reviews/003_prior.md, 003_stats.md, 003_skeptic.md, notes/hypotheses.md (후보 H, 확정 기록 8, 11, 11 변경), notes/log.md (2026-11-07 의 003 항목 3건).
- 이 보고서는 새 측정을 하지 않았다. 수치 출처 표기:
  - `rj:` = results.json 키 경로(줄 번호 병기). 리포터가 파일에서 직접 대조한 것은 `verdict` 블록 전체(25447-26043행), 12개 run 의 `slope_ol.all_sizes`, H-H1 의 시드별 c 와 CI 플래그, H-H2 의 쌍별 차이, `missing_runs` 이다. 크기별 ratio 평균, 부분 기울기(`trained_and_interp`, `extrap_only`), `train_env` 는 리뷰 인용이며 리포터가 직접 대조하지 않았다.
  - `csv:` = analysis/per_run_slopes.csv, analysis/recon_18runs.csv. 리포터가 읽었다. recon_18runs.csv 는 results.json 값이 아니라 Drive epoch_logs.jsonl 에서 온 **운영 로그 값**이다.
  - **운영 로그 값** = analysis/decoder_recon_by_epoch.json 과 그 파생(recon_18runs.csv). 리포터는 decoder_recon_by_epoch.json 을 직접 열지 않았고 csv 와 리뷰 인용, log.md 를 통해 썼다. 이 값은 results.json 이 아니며 판정에 쓰이지 않는다.
  - `(prior/stats/skeptic 절)` = 해당 리뷰가 계산했거나 인용했다고 밝힌 값을 옮긴 것. 별도 표시가 없으면 리포터가 일일이 대조하지 않았다. skeptic 의 `[계산]` 은 손계산이다.
  - 그림 3개(fig_slopes_by_run.png 등)와 시트 이미지는 리포터가 열어보지 않았다. 이미지 서술은 prior/skeptic 의 서술이다.
- 표기: "추정" 또는 "미확인" = 분석가 또는 리포터가 확인하지 못한 주장. 세 분석가는 서로의 파일을 읽지 않고 독립 작성했다.
- 이 보고서의 어떤 제안도 승인이 아니다. 사전 등록 임계값이나 규칙을 바꾸자는 권고는 쓰지 않았다. 사람이 결정할 사항은 8절에 모았다.
- 재학습 결정: 붕괴한 2개 run 을 재학습하지 않고, 문제를 명시한 채 리뷰하기로 한 것은 2026-11-07 사용자 결정이다(리드가 전달; 이 결정이 hypotheses.md 나 log.md 에 이미 적혀 있는지는 리포터가 읽은 범위에서 확인하지 못했고, 이 보고서 작성과 함께 log.md 에 적는다).

---

## 1. 한 문단 요약

끝 크기 하나로만 학습한 모델(single_a026, single_a248)과 크기를 섞어 학습한 모델(mix_adjacent, mix_wide)을 학습 시드 3개씩, 총 12개 학습해 open-loop(k=12) 면적의 크기 곡선 기울기와 "끌림의 중심 c" 를 보았다. 사전 등록 판정은 **H-H1 not_evaluable, H-H2 not_evaluable** 이다. H-H2 안에서 mix_wide 만 pass(9쌍 차이 0.3406~0.6542, 임계 0.2), mix_adjacent 는 not_evaluable 이다(`rj: verdict`). 12개 중 2개(mix_adjacent_s0, single_a026_s1)는 900 에피소드 전부 area_ol 이 0 이고 기울기 0.0, CI [0,0] 이며, 세 분석가는 모두 이를 학습 중 디코더 붕괴로 인한 측정 실패로 본다. 중요한 점은 not_evaluable 이 붕괴 때문만이 아니라는 것이다. 두 run 을 어떤 값으로 바꾸거나 제외해도 두 가설의 전체 판정은 바뀌지 않는다(stats 민감도 격자, skeptic 규칙 계산). 세 분석가가 일치하는 것은 (1) 판정 계산이 규칙대로이고, (2) 붕괴 run 의 수치는 어떤 주장에도 쓰면 안 되며, (3) mix_wide 의 baseline 이 다른 실험(001)의 값이라는 교란이 남고, (4) 시드 3개(정상 run 만 보면 일부 2개)라 시드 단위 유의성은 주장할 수 없다는 것이다. 갈리는 곳은 mix_wide pass 의 의미(prior: 문헌 방향과 맞으나 약한 증거 / stats: 규칙 통과이지 유의 검정이 아님 / skeptic: 혼합 없는 single_a248 도 같은 기준을 통과하고 지표 효과로 설명 가능), 낮은 single_a026 기울기가 실제 신호인지(stats: 실제 신호 / skeptic: 해석 미정 / prior: 상수 출력 쪽으로 읽되 디코더와 분리 불가), 디코더 붕괴의 원인 후보다. 가설 H-H1 과 H-H2 는 이 실험으로 지지도 반증도 되지 않았다. 확신 수준: 판정 재계산의 정확성과 "재학습해도 판정 불변" 은 높음(stats, skeptic), 붕괴 run 이 측정 실패라는 점은 높음, mix_wide 효과의 방향은 중간, 일반화 주장과 붕괴 원인은 낮음. 분석가별 전체 확신도는 prior 중간~낮음, stats 항목별(높음/중간~높음/중간/낮음), skeptic 항목별(높음/중간/낮음)이며 단일 전체 등급은 stats 와 skeptic 이 제시하지 않았다.

---

## 2. 배경과 가설

- 배경: 001 에서 ind 단일 학습 모델의 먼 크기 open-loop 면적이 ind 쪽으로 이동했으나, 학습 크기(ind)가 평가 범위의 가운데라 "학습 크기로 수렴" 과 "가운데로 압축" 을 구별할 수 없었고 크기를 섞은 대조군도 없었다(hypotheses.md 후보 H 첫 줄, 001 보고서 한계 인용).
- 가설(hypotheses.md 후보 H; 문장은 2026-11-03 사용자 확정, 판정 수치는 제안값을 사용자가 그대로 채택한 것으로 근거 수치가 아니다. 확정 기록 8, 결과를 본 뒤 바꾸지 않는다):
  - H-H1: 끝 크기 하나로 학습한 모델의 open-loop 면적(k=12)은 평가 크기를 그 학습 크기 쪽으로 끌어당긴다. 가운데 압축이라면 끌림의 중심이 ind 근처에 남는다.
  - H-H2: 크기를 섞어 학습하면 baseline_single(001, ind 단일)보다 open-loop 수축(ol@k12 기울기 감소)이 줄어든다. 코드의 "기울기"는 log ol 면적 대 log real 면적의 기울기이고 1 에 가까울수록 크기를 보존한다(README).
- 사전 등록 성공 기준(hypotheses.md:246-250): H-H1 지지는 두 단일 모델 모두 3개 시드 전부에서 c 가 ind 보다 학습 크기에 가깝고 반대쪽 크기들의 CI 가 1 을 학습 크기 방향으로 제외. H-H2 지지는 기울기가 baseline_single 보다 [제안 0.2] 이상 크고 3개 시드 모두 같은 방향. 그 밖은 판정 불가.
- 실험이 가르지 않는 것: 원인 층위(디코더/predictor/정보)는 이 실험이 가르지 않는다고 사전에 적혀 있다(hypotheses.md 알려진 한계, README "이 실험으로 주장하려는 것").

---

## 3. 설정과 방법 (config 기준)

| 항목 | 값 | 출처 |
|---|---|---|
| 학습 분포 4종 | single_a026 (a026_train 2000), single_a248 (a248_train 2000), mix_adjacent (a080 667 + ind 667 + a125 666), mix_wide (a040 667 + ind 667 + a160 666). 모두 `*_train` split, 총 2000 에피소드 | config.yaml:61-73 |
| 학습 시드 | [0, 1, 2], 전역 seed 0 | config.yaml:3, 89 |
| 학습 | batch 24, 15 epoch, num_hist 1, frameskip 5, predictor_lr 5.0e-5, 12개 모델 | config.yaml:91-98 |
| 코드 | acwm-phys commit 5a640bed66002eca7bdb31d10b2bde59ad944551 (001 과 같음) | config.yaml:9 |
| 평가 | 9개 크기 x 100 test 에피소드(`*_test` 만), open-loop k=12, 224x224 HSV 마스크 면적 | config.yaml:110-115; README |
| 지표 | 크기별 ratio_ol_over_real(에피소드별 `area_ol/max(area_real,1)` 의 평균, bootstrap 2000회), 기울기(9점 그룹 평균 log-log, CI 는 크기 내 에피소드 bootstrap 500회), c(평균 ratio 가 1 을 가로지르는 크기, log-log 선형 보간) | config.yaml:117-123; README |
| H-H1 규칙 | 두 모델 모두 3시드 전부 (c 가 학습 크기에 더 가깝고 + 학습 크기를 뺀 8개 크기 CI 가 모두 1 을 학습 크기 방향으로 제외) 이면 pass. 전부 c 가 ind 에 더 가까우면 fail. 그 밖(c null 포함) not_evaluable | `rj: verdict.H-H1.rule_text`; config.yaml:129-130 |
| H-H2 규칙 | 분포별 혼합 3시드 x baseline 3시드 = 9쌍 전부 차이 >= 0.2 이면 pass, 9쌍 모두 \|차\| < 0.2 이면 fail, 그 외 not_evaluable. 두 혼합 분포가 모두 pass 여야 전체 pass | `rj: verdict.H-H2.rule_text`; config.yaml:131-136 |
| H-H2 기울기에 쓰는 크기 | 9개 크기 전체(`slope_ol.all_sizes`) | verdict.py:125 (리포터가 Grep 으로 확인); README "구현자 해석" 5번 |
| baseline | 001 results.json 의 baseline_single(ind) ol@k12 기울기 0.2366 / 0.3102 / 0.1500 | `rj: verdict.H-H2.baseline.values` (25957-25961행) |
| 학습 환경이 run 마다 다름 | mix_wide 3개: Colab A100, 고정 commit 의 train.py 무수정, save_every 5, `train_env` 기록 없음(`gpu: "unknown"`). 나머지 9개: 재개 패치 적용 train.py, save_every 1, `train_env` 기록 있음. A100 이 주이고 single_a026_s0 은 L4 와 A100 혼재 | README 방법 절; hypotheses.md 확정 기록 11, 11 변경; `rj: runs.<run>.train_env` (리뷰 인용) |

실행에 관한 주의:
- README 는 이 패치가 재개용이며 새 학습의 epoch 범위는 1..15 로 원본과 같다고 적는다(README). `val()` 위 `@torch.no_grad()` 는 003 에서 유지된다(사용자 결정 2026-11-05, README). 002 는 이 줄을 지운 패치를 썼다.
- 재개 이력: single_a248_s2 는 epoch 1 이후 런타임이 끊겨 재개 1회(log.md 2026-11-07 "single_a248 s2 런타임 끊김과 재개"). single_a026_s0 은 `train_env` 가 action=resume, n_resumes=1 인데 hypotheses.md 확정 기록 11 변경은 실제로는 epoch 1 부터 새로 학습한 것으로 본다고 적는다(추정). **이 두 기록은 일치하지 않으며 어느 쪽이 맞는지 정하지 않았다.** 두 붕괴 run 은 `train_env` 상 fresh, n_resumes 0, A100 이다(prior 2.1, skeptic I4).
- `rj: environment` 는 학습 런타임이 아니라 aggregate 런타임 값이다(`gpu_note`; skeptic I8). 학습 때의 torch 버전은 기록이 없다.
- 사전 등록 이후 변경: 운영 값(save_every 5 -> 1, 재개 패치, GPU L4 -> A100)이 바뀌었고 판정 임계값은 바뀌지 않았다고 기록되어 있다(hypotheses.md 확정 기록 11, 11 변경: "임계값이나 규칙을 바꾸지 않았다"). 이 운영 차이가 결과에 영향이 없다는 문장은 config.yaml:97, 144 의 주장이고 검증되지 않았다(skeptic I3).

---

## 4. 결과

CI 는 모두 run 내부의 에피소드 재표집이며 학습 시드 변동을 포함하지 않는다. 시드 간 표준편차는 CI 반폭보다 크다(stats 4절: CI 반폭 약 0.04-0.05, 시드 SD 0.08-0.09).

### 4.1 판정표 (규칙대로 계산된 status)

| 항목 | status | 근거 | 출처 |
|---|---|---|---|
| H-H1 전체 | not_evaluable | single_a026, single_a248 둘 다 not_evaluable | `rj: verdict.H-H1.status` (25939행) |
| H-H2 전체 | not_evaluable | mix_wide pass, mix_adjacent not_evaluable | `rj: verdict.H-H2.status` (26011행) |
| H-H2 mix_wide | pass | pattern all_pairs_ge_gain, 9쌍 차이 최소 0.3406, 최대 0.6542 | `rj: verdict.H-H2.distributions.mix_wide` |
| H-H2 mix_adjacent | not_evaluable | pattern mixed_or_opposite, n_pairs_ge_gain 3, n_pairs_le_minus_gain 2, n_pairs 9 | `rj: verdict.H-H2.distributions.mix_adjacent` |

`missing_runs` 는 빈 목록이다(`rj: missing_runs`, 25447행).

### 4.2 run 별 기울기 (ol@k12, 9점, 1 이 크기 보존)

| run | s0 | s1 | s2 | 출처 |
|---|---|---|---|---|
| single_a026 | 0.0022 [-0.0844, 0.0906] | **0.0 [0.0, 0.0] (디코더 붕괴, 측정 실패)** | 0.1447 [0.0774, 0.2113] | `rj: runs.<run>.slope_ol.all_sizes` (직접 확인) |
| single_a248 | 0.5269 [0.4878, 0.5646] | 0.5774 [0.5347, 0.6241] | 0.5518 [0.5139, 0.5936] | 같음 |
| mix_adjacent | **0.0 [0.0, 0.0] (디코더 붕괴, 측정 실패)** | 0.5023 [0.4471, 0.5566] | 0.4153 [0.3699, 0.4617] | 같음 |
| mix_wide | 0.8042 [0.7483, 0.8519] | 0.6508 [0.6004, 0.6992] | 0.8005 [0.7561, 0.8424] | 같음 |
| baseline(001, ind) | 0.2366 | 0.3102 | 0.1500 | `rj: verdict.H-H2.baseline.values`. CI 는 csv(0.1875-0.2879, 0.2642-0.3607, 0.0848-0.2113) |

- 붕괴 run 의 0.0 은 `log(max(area,1))` 하한 때문에 모든 크기에서 같은 값이 되어 기울기가 정확히 0.0 이 된 것이다. "크기를 무시하는 평탄 예측"이 아니라 "마스크가 아무것도 못 찾음"이며, 두 경우는 평균 ol 면적 0 대 약 450 으로 구분된다(stats 3.1). 코드 경로는 skeptic C3 가 읽었다(metrics.py:39-41, verdict.py:129).
- 시드 평균(stats 4절, 정상 run 만): baseline 0.2322, single_a026 0.0734(n=2), single_a248 0.5520, mix_adjacent 0.4588(n=2), mix_wide 0.7518. baseline 과의 Welch 차이: single_a248 +0.320, mix_adjacent +0.227, mix_wide +0.520, single_a026 -0.159. 정확 순열 p 의 양측 최소값: 3 대 3 은 0.1, 2 대 3 은 6절 "표시만 하는 불일치" 2번에 적은 불일치가 있다.

### 4.3 H-H2 쌍별 차이 (혼합 - baseline)

- mix_wide (`rj: ...mix_wide.pair_differences_mix_minus_baseline`): s0 가 0.5676 / 0.4940 / 0.6542, s1 이 0.4142 / 0.3406 / 0.5008, s2 가 0.5639 / 0.4903 / 0.6505 (baseline s0 / s1 / s2 순). 전부 0.2 이상. 최소 여유 0.1406(stats 3.2, `stats_review_output.json`).
- mix_adjacent (`rj: ...mix_adjacent.pair_differences_mix_minus_baseline`): s0(붕괴) -0.2366 / -0.3102 / -0.1500, s1 0.2658 / 0.1922 / 0.3524, s2 0.1788 / 0.1052 / 0.2654. 붕괴하지 않은 s1, s2 의 6쌍만으로 3쌍은 0.2 이상이고 3쌍은 \|차\| < 0.2 이다. 붕괴 s0 의 -0.2366, -0.3102 두 쌍이 `n_pairs_le_minus_gain: 2` 로 "반대 방향" 쌍으로 집계되어 있다(skeptic C3 도 지적).
- 두 번째로 큰 변수: single_a248 의 기울기(0.5269 / 0.5774 / 0.5518)를 같은 규칙에 넣으면 baseline 대비 최소 차이가 0.5269 - 0.3102 = 0.2167 이어서 9쌍 전부 0.2 이상이다(skeptic C2 계산; 리포터가 results.json 의 기울기로 최소값 산술을 확인). 이것은 판정표의 항목이 아니다(single_a248 은 H-H2 대상 분포가 아니다).

### 4.4 H-H1 시드별 (`rj: verdict.H-H1.models.<model>.per_seed`, 직접 확인)

| 모델 | 시드 | c | 상태 | CI 조건(8개 크기) | supports_h1 |
|---|---|---|---|---|---|
| single_a026 | 0 | 0.3412 (a026-a040 사이, 학습 크기에 더 가까움) | evaluated | 충족 8/8 | true |
| single_a026 | 1 | null ("a size has no valid mean ratio") | not_evaluable | 8/8 로 표시됨. 단 CI 가 [0.0, 0.0] 이라서 나온 값(붕괴) | null |
| single_a026 | 2 | 0.3335 (a026-a040 사이) | evaluated | 불충족 7/8 (a060 CI [0.4442, 1.0716] 이 1 을 포함) | false |
| single_a248 | 0 | null ("ratio does not cross 1 on the grid (all sizes above 1)") | not_evaluable | 충족 8/8 | null |
| single_a248 | 1 | null (같은 이유) | not_evaluable | 충족 8/8 | null |
| single_a248 | 2 | 2.2093 (a200-a248 사이, 학습 크기에 더 가까움) | evaluated | 충족 8/8 (리포터가 직접 확인; prior 는 이 플래그를 확인하지 못했다고 적었다) | true |

- single_a026 의 패턴은 s0 supports_h1 true, s2 false 여서 s1 의 값과 관계없이 pass 도 fail 도 될 수 없다. single_a248 은 평가된 시드가 1개뿐이라 not_evaluable 이다(skeptic C1, stats 3.2).

### 4.5 붕괴한 두 run 의 운영 로그 값 (results.json 값이 아님)

| 지표 | mix_adjacent_s0 | 같은 분포 형제(s1 / s2) | single_a026_s1 | 같은 분포 형제(s0 / s2) | 출처 |
|---|---|---|---|---|---|
| epoch 15 `val_img_mse_reconstructed` | 0.07856 | 0.002050 / 0.002009 | 0.03305 | 0.001191 / 0.001195 | csv:recon_18runs |
| 최소 대비 최종, 최대 epoch 간 배율, 그 epoch | 26.57, 26.62, epoch 3 | 1.02, 1.03 | 21.41, 21.58, epoch 3 | 1.01, 1.03 | csv |
| 최종 val_loss | 0.2658 | 0.1894 / 0.1890 | 0.1608 | 0.1274 / 0.1275 | csv (log.md 도 같은 값) |
| 최종 val_z_visual_err_pred | 0.2062 | 0.1851 / 0.1815 | 0.1310 | 0.1205 / 0.1329 | csv |
| 최종 val_z_visual_err_rollout | 0.342 | 0.331 / 0.400 | 0.230 | 0.181 / 0.235 | csv (소수 3자리로 반올림) |

- 정상 16개 run 의 epoch 간 최대 배율은 1.39 이다(stats 5절; csv 에서 exp002_single_ind_s0 의 1.391 이 최대로 보인다). 붕괴 정의(최종값이 run 내 최솟값의 5배 초과이고 epoch 간 최대 배율 5 초과)와 임계값 5 는 stats 가 정한 값이고 사전 등록이 아니다. 2/18 = 11.1%(Wilson 95% CI [3.1%, 32.8%], stats 5절), 003 만이면 2/12.
- 두 run 모두 val 과 train 이 함께 높고 12 epoch 동안 거의 변하지 않는다(skeptic I4, prior 2.1). 붕괴 시점은 epoch 3 도중이다: train 지표는 epoch 의 첫 배치 값이라 epoch 3 의 첫 배치는 정상(mix_adjacent_s0 0.00217)이고 epoch 3 의 val 과 epoch 4 의 첫 배치가 높다(skeptic I4 가 `D:` 줄 번호로 인용; prior 도 같은 결론을 추정으로 표시). 이 인용은 리포터가 decoder_recon_by_epoch.json 에서 직접 대조하지 않았다.
- 정상 run 과 비교한 val_loss 상승률은 mix_adjacent_s0 +40.5%, single_a026_s1 +26.1%(stats 5절). 분포를 가로지르면 총 손실만으로는 구분되지 않는다: AUC 0.5625, single_a026_s1 의 val_loss 0.1608 은 정상 run 14개보다 낮다(stats 5절).
- 시각 확인(눈으로 본 서술, 픽셀 측정 아님): mix_adjacent_s0 의 open-loop 복원은 모든 열에서 같은 단색 벽 쪽 장면이다(prior 2.1, skeptic I4). 리포터는 이미지를 열어보지 않았다.

### 4.6 크기별 보조 수치 (prior/stats 인용, 리포터가 대조하지 않음)

- single_a248: a026 에서 ratio 10.74 / 15.95 / 11.04, 학습 크기 a248 에서 1.036 / 1.028 / 0.899 (prior 1.1). a248 s0, s1 은 학습 크기에서 CI 가 [0.989, 1.078], [0.991, 1.065] 로 1 을 포함하는데 평균이 1 을 넘어 c 가 null 이 되었고, s2 는 0.899 라서 c 가 정의되었다(stats 1절, skeptic C4b).
- single_a026: s0 의 크기별 평균 ol 면적이 422-537 px 로 평가 real 면적 488-5476 px 와 무관하다(Spearman -0.13), s2 는 438-601 px (stats 4.1). 학습 크기 real 평균은 487.65 px.
- mix_wide: 지원 안(a040-a160, 6점) 부분 기울기 1.066 / 0.767 / 1.011, 지원 밖(a026, a200, a248, 3점) 0.726 / 0.627 / 0.724. mix_adjacent(s1, s2): 지원 안(3점) 0.588 / 0.658, 지원 밖(6점) 0.495 / 0.403 (prior 1.1).
- c: mix_wide 1.637 / 1.555 / 1.207, mix_adjacent(s1, s2) 1.195 / 1.112 (csv 로 확인). prior 는 평균비가 분모가 작은 에피소드로 부풀려져 c 의미를 두기 어렵다고 본다(1.2).

---

## 5. 분석가 의견 종합

### 5.1 세 명이 일치한 점

1. **판정 계산은 규칙대로다.** stats 는 verdict 를 재구현해 H-H1, H-H2 모두 저장값과 일치함을 보였고 기울기와 CI 를 per-episode 면적에서 재계산해 1e-15 이내로 일치했다(stats 1절, `stats_review_output.json`). skeptic 은 `verdict.py` 와 results.json 을 줄 단위로 대조했다(C1). prior 는 H-H2 산술을 확인했다(1.2).
2. **붕괴한 두 run 은 측정 실패이며 그 수치(기울기 0.0, CI [0,0], c null)는 어떤 주장에도 쓰면 안 된다.** stats 3.1, skeptic C3 마지막 문장, prior 2.3. 세 명 모두 results.json 이 이를 유효값으로 저장하고 집계가 걸러내지 않는다고 지적한다(prior 3절, stats 3.1, skeptic C3). 예: single_a026_s1 의 8개 크기 CI 가 [0,0] 이라 "학습 크기 방향으로 1 을 제외" 플래그가 true 로 찍혔다(`rj`, 25548-25606행; skeptic C3).
3. **not_evaluable 은 붕괴 run 만의 결과가 아니다. 두 run 을 어떤 값으로 바꾸거나 제외해도 H-H1, H-H2 전체 판정은 바뀌지 않는다.**
   - stats 3.2: mix_adjacent s0 슬롯을 -0.5~1.5 의 401점 격자로 바꿔도 H-H2 전체 상태 집합은 {not_evaluable} 하나(`stats_review_output.json: H2_overall_set_over_s0_grid`). 정상 s1, s2 의 6쌍이 이미 3쌍 >= 0.2, 3쌍 < 0.2 로 갈린다. H-H1 은 single_a248 s0, s1 의 c 가 null 이고, single_a026 은 s0 true / s2 false 라서 s1 이 어떤 값이어도 mixed 다(`a026_s1_possible_outcomes_pattern`, `H1_overall_with_variant_and_any_a026_s1`).
   - skeptic C1: 같은 결론을 `verdict.py` 줄 번호와 `R` 줄 번호로 손계산(`verdict.py:135, 137, 143` 등).
   - prior: mix_adjacent 의 not_evaluable 이 "측정 실패뿐 아니라 규칙 때문에도 남는다", a248 s0, s1 의 c null 은 지표 설계 한계라고 적었다(1.2, 0절). prior 는 H-H1 에 대한 대입 격자 계산은 하지 않았다.
   - 따라서 재학습은 판정을 바꾸지 못하고 시드별 수치와 시드 수만 채운다(stats 8절 2번, skeptic C1). 이는 2026-11-07 "재학습하지 않고 문제를 명시한 채 리뷰" 결정과 방향이 같다. 다만 skeptic 과 stats 는 재학습을 기제 규명이나 시드 수 보강 목적으로는 별개로 제안한다(7절).
4. **mix_wide 의 baseline 은 다른 실험(001)의 값이라 같은 파이프라인 대조군이 없다.** prior 4.1 1번(우선순위 가장 높음으로 표시), stats 6.2, skeptic I1. 환경 차이(torch, GPU, 추출 코드)와, 실제 프레임 면적이 001 과 003 에서 약간 다르다는 점(stats 6.2: 크기별 0.02-0.68%, a026 에서 487.65 대 484.35; skeptic I1)이 근거다. 원인은 미확인이다.
5. **디코더 면적 지표는 predictor 편향과 디코더 오독을 분리하지 못한다.** 003 extract 가 복원 면적과 teacher-forced 면적을 저장하지 않는다(README, prior 3절, skeptic I5). H-H1 을 "predictor 가 학습 크기로 끌어당긴다" 로 읽는 것은 데이터 밖이다(skeptic I5).
6. **평균 ratio 추정량의 꼬리 때문에 c 와 CI 조건이 불안정하다.** prior 3절, stats 1-2절, skeptic C4. 예: single_a026_s0 의 학습 크기에서 평균 ratio 는 1.704 인데 평균 면적비는 453.2/487.65 = 0.929(stats 1절), skeptic 계산으로 상위 3개 에피소드가 ratio 합의 46% 를 차지한다.
7. **single_a248 의 c null 은 격자 끝 때문이다.** 학습 크기가 격자 끝이라 가설이 맞을수록 가로지름이 격자 밖으로 나가기 쉽고, a248 s0, s1 은 학습 크기 ratio 가 1 을 약 3% 넘었다는 이유로 null 이 된다(stats 1절, skeptic C4b, prior 0절). prior 는 "반증이 아니라 지표 설계의 한계" 라고 쓴다.
8. **시드는 3개(일부 셀 2개)이고 CI 는 시드 변동을 담지 않는다.** stats 4절, skeptic M2, prior 4.1 6번. 3 대 3 정확 순열 검정의 양측 최소 p 는 0.1 이라 p<0.05 는 구조적으로 불가능하다(stats 4절).
9. **붕괴는 추출이 아니라 학습 중에 생겼고 재개 때문이라는 기록상 근거는 없다.** 두 run 모두 fresh, n_resumes 0, A100(prior 2.1, skeptic I4). stats 도 로그상 epoch 3 급등이 추출 전부터라고 본다(3.1). 단 프로그램상의 원인은 어느 분석가도 확정하지 못했다(5.2).
10. **single_a248(0.53-0.58)이 mix_adjacent(0.42-0.50)보다 기울기가 높으므로 "섞으면 기울기가 오른다" 는 일반 원리로 읽을 수 없다.** prior 1.2, skeptic C2(크기 순서 a026 < ind < a248 로 기울기가 커지는 3점 관측), stats 4절 마지막 항목(a248 이 baseline 보다 크다는 것은 같은 한계 안의 관측 사실).
11. **mix_wide 의 `train_env` 기록이 없고 train.py 가 무수정 원본이어서 나머지 9개와 학습 코드·저장 주기가 다르다.** prior 6절, stats 6.1, skeptic I1, I8. 판정 표에는 반영되지 않는다(hypotheses.md 확정 기록 11).

### 5.2 의견이 갈린 점 (하나로 합치지 않음)

| 쟁점 | analyst-prior | analyst-stats | analyst-skeptic |
|---|---|---|---|
| **mix_wide pass 의 의미** | 규칙대로 계산된 값. "지원이 넓을수록 외삽 거리가 줄어든다" 는 직관(Xu 2021, Balestriero 2021 요약)과 **방향이 같다**. 그러나 **약한 증거**: (a) mix_adjacent 가 s0 붕괴로 유효 2시드, (b) 9점 기울기가 지원 안 크기 개수에 지배(mix_wide 6/9 대 mix_adjacent 3/9)라 혼합 효과와 지원 폭 효과가 분리 안 됨, (c) baseline 이 다른 환경. 차이가 최소 0.340 으로 커서 환경 차이만으로 설명되기는 어렵다고 추정하나 검증 못 함(6절). 확신 낮음~중간 | pass 의 정확한 의미는 "001 의 baseline 3개 값과 비교해 9쌍 모두 +0.2 이상" 이다. **유의 검정이 아니라 규칙 통과**: 3 대 3 순열 최소 p 0.1, 시드 3개. 평균 0.752 대 0.232(Hedges g 4.95). 실험 간 오프셋이 시드 SD 0.08 의 약 1.8배(0.1406 여유)이면 뒤집힌다(추정). "크기 혼합이 수축을 줄인다" 의 일반화는 같은 파이프라인 baseline 없이는 약한 증거. 효과 크기와 방향의 확신은 중간, 일반화는 낮음 | **지표 효과로 설명 가능**. 판정 규칙을 혼합 없는 single_a248 에 적용하면 9쌍 전부 0.2 이상(최소 0.2167)이라 이 기준은 혼합이 없어도 충족된다(C2, 확신 높음). 학습 크기 범위로 자른 단순 모델(예측 = 실제 면적을 학습 범위로 자름)이 mix_wide 기울기 0.69 를 내고 관측은 0.65-0.80(I2, 손계산, 중간). 단 mix_wide 의 범위 안 기울기가 baseline 과 크게 다르므로 실제 효과 가능성은 열어 둔다: "노출된 크기를 재현한다" 에 가깝고 "수축 기제가 줄었다" 의 증거는 아님. 환경 교란과 분리 안 됨 |
| **낮은 single_a026 기울기(s0 0.0022, s2 0.1447)가 실제 신호인가** | 거의 상수 면적 출력에 해당하고 001(ind, 0.15-0.31)보다 더 상수적이다. Kang 2024 의 상수 회귀와 방향은 맞다(설정이 달라 같은 현상이라고는 못 씀). a026 과 a248 의 비대칭은 이 실험으로 설명되지 않는다(후보는 추정). 디코더 오독과 분리 불가 | **측정 실패가 아니라 실제 신호**(확신 중간). 평균 ol 면적이 학습 크기(약 490 px) 근처에서 평탄하고 ol 면적 0 인 에피소드는 900 중 2개. 기울기 0 은 가장 강한 수축을 뜻한다. 다만 정상 시드가 2개뿐이고 s0 는 재개 의심 run, s0 와 s2 의 차이 0.143 은 시드 SD 범위 안 | **해석이 정해지지 않는다**(I5). 이 곡선의 ratio 는 "상수 / 실제 면적" 이라 a026 에서 CI 조건이 산술적으로 충족된다. 기울기 0.0022 와 붕괴의 0.0 은 값만으로는 거의 구별되지 않는데 현상은 전혀 다르다. predictor 가 크기를 잃었는지 a026 만 본 디코더가 큰 천을 못 그리는지 003 데이터로 구분 불가. 반대 근거: 001 의 ind 단일 디코더는 복원 기울기 1.01(`R001:9025`)이라 "디코더가 다른 크기를 못 그린다" 설명은 약하나 끝 크기 단일 디코더는 미확인 |
| **디코더 붕괴 원인 후보** | 모두 **추정**. Adam 손실 스파이크(arXiv 2506.04805, 디코더 붕괴를 직접 다룬 것은 아님), 죽은 뉴런/포화(Lu et al. 2019 로 기억하나 확인 못 함). 한 epoch 안 급등 뒤 12 epoch 동안 고정은 회복 불가한 사건과 맞고 느린 과적합과는 안 맞는다. 같은 설정의 상수 붕괴 보고는 문헌에서 못 찾음(미확인) | 원인 **미확인**. epoch 3 에 두 run 이 모두 급등했다는 사실이 우연인지 공통 원인(같은 Colab 세션, 데이터 로더 이슈)인지 이 데이터로는 구분되지 않는다(추정). 2/18 로 표본이 작아 코드·환경 귀속 불가 | 가장 유력하나 **미확인**: 입력 무관 출력, 흡수 상태, 경계 패턴, 11ba488 에 BatchNorm 없음 + ReLU + clipping 없음을 근거로 한 상류 층 신호 소실(확신 낮음). 보조 후보: 큰 gradient 한 번, 시드x데이터의 결정적 궤적 대 일회성 하드웨어 사건. 약함: 시드 단독, GPU 종류, 재개 패치/save_every 1(확률 0.17), `val()` no_grad 유지(0.43). 기각: 추출 버그, BatchNorm 의 train/eval 차이 설명, VQ 코드북. 고정 commit 코드는 미확인 |
| **붕괴 run 의 predictor 는 정상인가** | 형제 시드 범위 안이라고 요약(0절)했으나 본문 2.1 에서는 mix_adjacent_s0 의 pred 가 형제보다 약 12% 높다고 적음. 디코더 손실이 predictor 로 역전파되는지는 코드를 읽지 않아 모름 | mix_adjacent_s0 는 예측기 쪽도 조금 나빠졌을 가능성(+12%, 추정, 시드 1개), single_a026_s1 은 +3% | 디코더 손실이 `z.detach()` 로 분리되어 있어 일관된다고 봄(`visual_world_model.py:250-252`, 11ba488 기준, 고정 commit 은 미확인). 값은 한 배치 기준 |
| **single_a026 s2 의 CI 조건 실패(a060)가 신호인가** | 언급은 있으나 원인 해석 안 함 | 평균 추정량의 꼬리 때문. 평균의 비율로 다시 계산하면 CI [0.394, 0.473], 중앙값 비율 [0.386, 0.482] 로 1 을 명확히 배제(추정: 다른 시드와 구분할 근거는 없음) | 인접 크기 대비 튀는 값이라 꼬리 에피소드로 의심 |
| **c 를 H-H1 의 판정 변수로 쓸 수 있는가** | c 는 끝 크기에서 구조적으로 정의되지 않는 경우가 있다. 상수성 지표를 후속 가설로 제안(5절 H-OCS 초안) | 추정량 선택에 따라 per-seed 결과가 달라진다. 연속 지표로 바꾸는 후속 가설을 새로 세워야 한다고 함 | 정의 자체가 평균 후 로그인데 사전 등록 문구는 "로그 면적비가 0 을 가로지르는" 이라 에피소드별 로그 후 평균으로도 읽힌다. 구현자 해석 목록에 없음(C4a) |
| **전체 확신도** | 중간~낮음 | 항목별: 판정 재계산 높음, 민감도 중간~높음, a026 신호 중간, mix_wide 일반화 낮음 | 항목별: C1, C2 산술 높음, c 꼬리/지표 효과 중간, 붕괴 기제 낮음 |

주의: 위 표의 prior 문헌 인용(Kang 2024, Xu 2021, Balestriero 2021 등)은 prior 가 웹 초록/요약 수준으로만 확인했고 본문을 정독하지 않았다고 밝혔다(prior 상단). 리포터도 원문을 확인하지 않았으므로 이 보고서는 해당 문헌을 근거로 쓰지 않는다.

### 5.3 한 명만 지적한 점

- prior 만:
  - 문헌: 단일 크기 학습의 학습 크기 쪽 끌림은 Kang 2024 (ICLR, arXiv 2310.00873)의 "분포 밖에서는 학습 분포의 상수로 회귀" 와 방향이 같다. 월드 모델의 객체 크기 분포와 크기 보존을 직접 다룬 보고는 확인 못 함(미확인).
  - 빠진 대조: 학습된 디코더별 GT-latent 재구성비(12개 모델), 학습 평균 면적 상수 출력과 persistence baseline, 같은 크기 집합의 부분 기울기, 가운데 단일 크기 대조(4.1).
  - H-H2 산술: 이 규칙은 baseline 시드 범위(0.15~0.31) 때문에 mixed 가 나오기 쉬운 구조(1.2, README 4번과 같은 지적).
  - 후보 가설 초안 4개: H-OCS, H-Support, H-Decoder-OOD, H-Collapse (5절). 제안이지 사전 등록이 아니다.
  - 발생 빈도 비교: 003 patched 9개 중 2개, 002 patched 6개 0개, 001 3개 0개, mix_wide 3개 0개(2.2). skeptic 은 확률 계산으로 패치 그룹 몰림 확률 0.17(구별력 없음)을 별도로 냈다.
- stats 만:
  - 추정량 민감도: 평균의 비율이나 중앙값 비율로 바꾸면 single_a026 s0, s2 모두 c 가 정의되지 않고(no_crossing_all_below), single_a248 은 평균의 비율로 s2 만 c 가 정의되지만 ind 쪽(0.786)이며 중앙값 비율로는 s2 가 supports_h1 이 아니다(2절). H-H1 의 per-seed 판정은 추정량 선택에 따라 달라진다. 어느 추정량이 옳은지는 판단하지 않았다(stats 9절).
  - 효과 크기와 순열: Welch 95% CI, Hedges g, 정확 순열 p (4절 표). mix_wide 가 뒤집히는 조건(baseline 최댓값 0.4508 이상 또는 혼합 최솟값 0.5102 이하)은 `stats_review_output.json: mix_wide_flip` 로 확인.
  - 붕괴 빈도의 신뢰구간(2/18, Wilson [3.1%, 32.8%], Clopper-Pearson [1.4%, 34.7%])과 총 손실로의 탐지 가능성(분포 내 +26~+40% 로 쉬움, 분포 간 AUC 0.5625 로 불가).
  - 재개 run 의 영향: single_a248_s2, single_a026_s0 모두 증거가 없으나 검정력도 없다(6.1).
  - 001 과 003 의 훈련 손실 수준 비교(001 0.1712-0.1722, 003 mix_wide 0.1864-0.1868). 분포가 달라 당연한 차이일 수 있어 같은 비교가 아니다.
- skeptic 만:
  - **C2 음성 대조**: single_a248 이 H-H2 기준을 통과한다(위 4.3).
  - **C3 시드 품질 게이트 부재**: 사전 등록에도 코드에도 복원 품질 조건이 없다(hypotheses.md:246-254, run.py:266-278, aggregate.py:63-70). 후보 입력으로 최종 `val_img_mse_reconstructed` 와 표본 프레임 간 픽셀 표준편차를 제시했고, 게이트는 기울기와 c 를 보지 않는 값으로만 구성하고 결과를 보기 전에 글로 고정해야 한다고 한다. 임계값은 사람이 정한다.
  - **I3 사전 등록 이탈**: 아래 6절 "표시만 하는 불일치" 1번 참고.
  - **I6 HSV 마스크**: 붕괴 run 의 면적 0 이 색(채도 S>120)에 달린 값이라 "천 없음" 이 아니라는 추정, open-loop 에서 흐림/탈채도가 면적 감소로 보일 수 있다는 의심(둘 다 프레임 근거 없음, 추정).
  - **I7 개별 run 이력**: single_a026_s0 의 `train_env` 와 확정 기록 11 변경의 불일치, `R:2634` action "resume" 과 `resume.py:116-139` 가 맞지 않음(폴더를 옮긴 주체 미확인), mix_wide_s0 의 `n_log_lines` 19(다른 run 은 15, 경위 미확인).
  - **I8 재현성**: `experiment_code_git: null`, 학습 시점 코드 sha 는 9개 run 의 `train_py_sha256` 뿐, `acwm_phys_src_git` 이 HEAD 만 봐서 패치 트리와 무수정 트리가 같은 값, pip 동결 목록 없음.
  - **M3 문서 낡음**: README.md:3 "GPU 학습 ... 하지 않았다", README:77 "미검증" 항목이 실행 후 갱신되지 않았다(리포터가 README 를 읽어 같은 문구를 확인했다).
  - **M4 판정 비대칭**: H-H1 pass 는 CI 조건을 요구하는데 fail 은 요구하지 않는다. 사전 등록대로이지만 fail 이 더 쉽다(verdict.py:59-60).
  - selftest 에 전부 0 면적, 단일 크기 평평 곡선 경우를 판정까지 돌리는 부분을 찾지 못했다(Grep 범위).

---

## 6. 한계와 위험

- **시드와 CI**: 학습 시드 3개(정상 run 만 보면 single_a026, mix_adjacent 는 2개). CI 는 run 내부 에피소드 재표집이다. 3 대 3 정확 순열의 양측 최소 p 는 0.1 이다(stats).
- **베이스라인 교란**: H-H2 baseline 은 001(RTX 5090, torch 2.7.1+cu128)의 값이고 003 은 Colab 의 A100/L4, torch 2.0.1 계열 환경이며 추출 코드도 경량판이다. README 는 "001 숫자와 일치하는지 미확인" 이라 적는다. 003 안에 같은 파이프라인의 ind 단일 대조군이 없다. mix_wide 의 train_env 가 기록되지 않아 A100 사용은 README 의 사실 기록과 hypotheses.md:330 의 "보인다" 사이에 표현 차이가 있다(smoke 의 GPU 메모리와 서버 주소가 근거, skeptic I1).
- **지표 한계**: 평균 비율 추정량의 꼬리, c 의 격자 끝 문제, 9점 기울기가 학습 크기 범위와 크기 집합에 민감함(skeptic I2: 3크기 기울기 0.3088 대 9크기 0.2366, `R001`). 디코더 + HSV 마스크 경로를 거치므로 면적 0 이 "천 없음" 과 구별되지 않는다.
- **원인 층위 미분리**: 복원 면적을 저장하지 않아 predictor 편향과 디코더 오독을 가르지 못한다. 이 실험은 사전에 이를 범위 밖이라 밝혔다.
- **재개와 운영 차이**: 정상 run 중 재개 의심은 single_a026_s0, single_a248_s2 두 개다. 둘 다 정상 시드가 같은 조건에 1-2개뿐이라 영향을 분리할 수 없다(stats 6.1). 운영 값이 결과에 영향이 없다는 주장은 검증되지 않았다(skeptic I3).
- **붕괴 2건의 가능한 의미**: 붕괴 2건이 모두 "패치 + save_every 1 + `val()` no_grad 유지" 그룹(9개 중 2개)에서 나왔고, 002 의 패치 6개와 무패치 mix_wide 3개에서는 0건이다. 단독 원인이라는 근거는 없고 구별력도 없다(skeptic 확률 0.17, 0.43). 원인이 확인되지 않으므로 나머지 정상 10개 run 에 같은 문제가 부분적으로 숨어 있지 않다는 보장은 로그의 재구성 MSE 가 끝까지 0.0012-0.0028 이라는 사실에 의존한다(log.md, csv).
- **미확인 목록**:
  - 고정 commit 5a640bed 의 `models/vqvae.py`, `conf/train.yaml`, `train.py`(decoder_lr, BatchNorm, AMP, clipping, `val()`). 이 머신의 `thread0/` 는 main(11ba488)이고 skeptic 의 모든 `thread0/` 인용은 이 기준이다.
  - 체크포인트 내용, 전체 프레임과 `frames_sample.npz`, 학습 시점 torch 버전, mix_wide 학습 GPU 와 패치 상태, GPU ECC 기록, 001 과 003 의 실제 면적 차이의 원인.
  - results.json 의 sha256: log.md 는 4b86c6de..., 500290 바이트로 적는다. 리포터는 해시 도구가 없어 계산하지 못했다. 파일 크기는 읽기 도구 표시(488.6KB)가 500290 바이트와 대략 맞는 정도로만 확인했다. 현재 저장소 코드와 `rj: experiment_code_sha256` 의 일치도 미확인이다(skeptic).
  - 그림 3개와 시트 이미지는 리포터가 열어보지 않았다.
- **삭제된 중간 체크포인트(리드가 Drive 검색으로 확인한 사실)**: `model_2.pth`, `model_3.pth` 는 Drive 휴지통에 있다(mix_adjacent_s0 폴더로 보이는 파일 포함; 확인 주체는 리드이고 출처는 Drive 검색이다. 리포터가 직접 검색하지 않았다). 이 파일들은 DONE 후 `keep_ckpt_epochs [5,10,15]` 정책으로 삭제된 것이고(config.yaml:146, skeptic I4) 붕괴 직전과 직후를 볼 수 있는 유일한 체크포인트다(남은 5, 10, 15 는 모두 붕괴 후; skeptic I4). Drive 휴지통의 일반 보존 기간 30일은 **추정**이며 실제 삭제 일자와 만료일은 확인되지 않았다. skeptic 은 삭제 시기를 11-05~06 으로 추정했으나 log.md 의 UTC 시각 기록과 날짜 대응을 확인하지 못했다(미확인). 붕괴 부검에 필요하면 기한 전에 복원해야 한다.
- **표시만 하는 불일치 (고치지 않음)**:
  1. **사전 등록 이탈**: hypotheses.md:250 은 "외삽 크기 결과는 따로 보고하고 H-H2 판정에는 쓰지 않는다" 인데 verdict.py:125 는 `all_sizes`(외삽 포함 9점)를 쓴다(리포터가 hypotheses.md 줄과 verdict.py Grep 으로 직접 확인). README "구현자 해석" 5번은 이를 "사용자 확인 필요" 로 적었으나 확인 기록을 리포터도 skeptic 도 찾지 못했다. skeptic 은 baseline 이 학습 크기 1개라 부분 기울기 정의가 불가능해 문구 자체가 구현 불가였을 수 있다고 덧붙인다. **prior 와 stats 는 이 이탈을 지적하지 않았다**(stats 는 README 의 구현자 해석을 그대로 따른다고 적었다). README 5번의 다른 구현자 해석(9쌍 전부, all_untrained)도 같은 "확인 기록 미발견" 상태다(skeptic I3 2번).
  2. **stats 문서 내부**: stats 4절은 "2 vs 3 은 가능한 배정이 10가지라 양측 최소 p 는 0.2 가 될 것으로 예상했으나 코드 결과는 0.1 이다" 라 적고, 표에서 single_a026(n=2)은 0.2, mix_adjacent(n=2)는 0.1 로 같은 설계(2 대 3, 경우의 수 10)에서 서로 다른 값이 나왔다. stats 는 극단 배정 두 방향이 모두 포함되지 않은 이유를 확인하지 않았다고 적었다. 리포터도 원인을 확인하지 않았고 어느 값이 맞는지 판단하지 않았다. 어느 쪽이든 p<0.05 는 불가능하다는 결론은 stats 가 적은 대로다.
  3. **prior 내부**: prior 0절 3번과 2.1 은 "두 붕괴 run 의 val_z_visual_err_pred 는 형제 시드 범위 안" 이라 쓰지만, csv 로 보면 mix_adjacent_s0 의 0.2062 는 형제 0.1851, 0.1815 보다 높고(약 12%), prior 가 비교 대상으로 든 mix_wide 시드 범위(0.1751-0.1958, csv)보다도 위다. prior 2.1 본문의 "mix_wide 시드 간 범위(0.175~0.196)와 겹친다" 는 이 csv 값과 맞지 않는다. 단 single_a026_s1(0.1310)은 형제 범위(0.1205-0.1329) 안이다. 리포터는 수정하지 않고 표시한다.
  4. **prior 가 확인하지 못했다고 적은 항목**: prior 는 이 환경에 셸이 없어 results.json 의 에피소드별 배열을 읽지 않았고 계산을 파일에서 읽은 값의 산술로 한정했다. 코드(src/exp003)도 읽지 않았다. 문헌 서술은 웹 요약 수준이다. single_a248 s2 의 CI 플래그는 prior 가 "확인 못 함" 이라 적었으나 리포터가 results.json 에서 8개 모두 true 임을 확인했다(4.4).
  5. **run 이력 기록 불일치**: single_a026_s0 의 `train_env`(action=resume, n_resumes 1)와 hypotheses.md 확정 기록 11 변경(처음부터 재학습으로 추정)이 다르다(3절).
  7. **stats 문서 내부(사소)**: stats 3.1 은 "정상 15개 run 의 epoch 간 최대 배율 1.39", 5절은 "정상 16개 ... 최대 1.39" 라 적는다. 18개 중 붕괴 2개를 빼면 16개이고 csv 의 최대값도 1.391(exp002_single_ind_s0)이다. 15 와 16 중 어느 쪽이 의도인지 확인하지 않았다.
  6. **README 문서가 낡음**: README.md:3 과 77 의 상태 문구는 실행 전 상태다(skeptic M3).

---

## 7. 다음 단계 후보 (제안만, 사전 등록 값은 바꾸지 않는다)

저비용 또는 새 학습 없음:
1. **붕괴 부검**: Drive 의 `valid/`(epoch 별 검증 이미지), 휴지통의 `model_2.pth`, `model_3.pth`(기한 전 복원), 남은 `model_5.pth` 의 디코더에 forward hook 을 걸어 층별 활성 비율과 배치 간 표준편차 확인, 층별 gradient norm(skeptic 4번 항목, 6절의 삭제 체크포인트 항목). 재학습 없이 가능한 부분은 GPU 거의 불필요(skeptic).
2. **층위와 지표 민감도 분리(새 학습 없음, 사후 분석으로 표시)**: 기존 체크포인트로 GT latent 복원 면적(z_0, z_12)과 open-loop 면적을 같은 에피소드에서 저장, S 임계값 민감도, 집계 방식(평균의 비율, 로그 비 평균, 꼬리 제거)을 병기(skeptic 3번, prior 4.1 2번). 단일 크기 디코더의 크기별 복원비(prior H-Decoder-OOD)가 여기에 포함된다.
3. **상수 출력 하한과 persistence 대비 기울기**(prior 4.1 3번), 같은 크기 집합의 부분 기울기 비교(prior 4.1 4번; baseline 의 부분 기울기는 001 에서 계산되지 않았다).
4. **파이프라인 자동 점검 추가**: 추출 단계에서 `area_ol` 전부 0 이거나 재구성 MSE epoch 간 배율이 큰 run 을 표시(stats 8절 4번). 코드 변경은 experimenter 몫.

GPU 필요 / 사람 승인 필요:
5. **같은 환경의 ind 단일 baseline 3시드 재학습**(또는 003 extract 로 001 체크포인트 재측정): mix_wide 와 single_a248 의 baseline 교란을 직접 측정(stats 8절 3번이 가장 정보량 큼으로 꼽음, skeptic 추가 실험 1번, prior 4.1 1번이 우선순위 가장 높음). 세 분석가가 같은 항목을 제안한다. 가능하면 mix_wide 를 같은 코드로 다시 돌리고 single_a248 을 음성 대조로 함께 표시(skeptic).
6. **붕괴 재현**: mix_adjacent_s0, single_a026_s1 을 같은 설정으로 epoch 4 까지 재학습하며 층별 grad norm 기록(skeptic 추가 실험 2번; run 당 약 74분은 skeptic 추정). 이것은 판정을 바꾸지 못하므로 기제 규명과 데이터 무결성이 목적이다(stats 8절 2번, skeptic C1). 2026-11-07 결정은 "붕괴 2개를 재학습하지 않는다" 이므로 이 항목은 그 결정을 바꾸는 것이어서 별도 승인이 필요하다.
7. **통제 설계 추가**: 가운데 단일 크기(a040, a160 등)로 학습 크기 위치 효과와 혼합 효과를 분리(prior 4.1 5번), 시드 확대(prior 4.1 6번).
8. **새 후속 가설을 세우는 경우**: H-H1 의 비교 가능한 증거가 필요하면 추정량을 사전 등록하고 c 를 교차점이 아닌 연속 지표로 바꾸는 후속 가설을 **새로** 세운다(stats 8절 1번). 시드 품질 게이트도 결과를 보기 전에 글로 고정한다(skeptic C3). 지금의 판정 기준을 결과 후에 바꾸지 않는다. 이 항목들은 현재 003 판정을 바꾸는 제안이 아니다.

문서와 기록 정리(저비용):
9. README.md:3, 77 갱신, results.json 에 학습 시점 코드 sha, 재개 기록(run, epoch), mix_wide 학습 환경 정보 보강(재계산 아님; skeptic M3, I8).

---

## 8. 사람이 결정해야 할 사항

1. 붕괴 부검(휴지통의 model_2/3.pth 복원, Drive 의 valid 이미지, 층별 hook)을 할지, 한다면 만료 추정(일반 보존 30일은 추정, 삭제 일자 미확인) 전에 복원할지. 재학습은 2026-11-07 결정대로 하지 않는 상태를 유지하는지(skeptic 의 epoch 4 재학습은 그 결정을 바꾼다).
2. 같은 환경의 ind 단일 baseline 3시드 재학습 또는 001 체크포인트 재측정을 승인할지(GPU 비용). 승인 전까지 mix_wide pass 는 교란 때문에 약한 증거로 표기할지.
3. 논문 주장 범위와 이탈 표기: hypotheses.md:250 대 verdict.py:125(외삽 포함 9점 기울기 사용)의 사전 등록 이탈을 기록으로 남길지, 규칙이나 지표(c, 추정량, 시드 품질 게이트)를 바꾸는 후속은 결과를 본 뒤의 변경이며 새 사전 등록으로만 한다고 명시하고 진행할지.
