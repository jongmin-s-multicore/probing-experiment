# 실험 001_size_collapse_diagnosis 보고서

- 작성: reporter, 2026-11-02
- 입력: experiments/001_size_collapse_diagnosis/results.json (필요한 키만 부분 읽기), notes/reviews/001_prior.md, 001_stats.md, 001_skeptic.md, notes/hypotheses.md ("후보 D/E 사전 등록 초안", "확정 기록 2"), notes/log.md (2026-11-02 항목), analysis/ (decoder_ratio_table.csv, fig1~fig3).
- 이 보고서는 새 측정을 하지 않았다. 수치는 (a) results.json 키 경로, 또는 (b) 리뷰가 results.json 에서 재계산했다고 밝힌 값(리뷰 파일/절 병기)만 쓴다. 리포터가 results.json 에서 직접 대조한 것은 `across_seeds.baseline_single.metrics` 의 `diagnose/area/groups/a026/by_k/12/ratio_ol_over_real/mean` (mean 3.987, min 3.799, max 4.178)과 해당 키 주변의 구조뿐이다. 나머지는 리뷰가 인용한 값을 옮겼다.
- 표기: "prior / stats / skeptic" = analyst-prior / analyst-stats / analyst-skeptic. 세 분석가는 서로의 파일을 읽지 않고 독립 작성했다. "추정" = 해당 분석가 또는 리포터가 확인하지 못한 주장.
- 이 보고서의 어떤 제안도 승인이 아니다. 사람의 결정이 필요한 항목은 "사람이 결정할 사항" 절에 모았다.

---

## 1. 한 문단 결론

cloth_sim 9종 크기(명목 면적비 0.259~2.479)를 ind 단일 크기로 학습한 baseline(학습 시드 0/1/2)에서, 먼 크기의 open-loop(k=12) 면적은 ind 쪽으로 이동한다(큰 천 a248: 모델/real = 0.434 [시드 0.333~0.549], 작은 천 a026: 3.987 [3.799~4.178]). 이 방향성은 3개 시드에서 일치하고, 기울기 CI 는 ol 의 k=3,6,12 모두 1 을 제외한다(stats). 반면 사전 등록 D/E 판정으로 올라가는 단계에서는 근거가 약하다. (1) 0번 전제 게이트(`shrink_ratio_tol`)는 null 이라 판정 자체가 불가능했고, 승인된 임계값(0.9/1.1, 0.1, slope_margin)은 코드에 연결되지 않아 모든 "판정"이 결과를 본 뒤 사람이 손으로 한 것이다(skeptic). (2) 후보 D("디코더는 원인이 아니다")는 점추정(평균비)으로는 27칸(9크기x3시드) 모두 [0.9, 1.1] 안이지만, 단위(점추정/CI/에피소드)에 따라 판정이 갈리고 그 선택이 사전 등록에 없다. 또 이 시험은 GT latent 를 디코더가 읽는 경우만 본다. (3) 후보 E 의 "인코더 보유"는 random-ViT/픽셀 격자도 거의 같은 수준으로 크기를 읽어 DINOv2 특이적 증거가 아니고, 개입(intervene)은 효과가 작아 "predictor 가 크기 정보를 쓴다/버린다" 어느 쪽도 결론낼 수 없다(stats, skeptic 일치). (4) "학습 크기로 수렴"은 큰 천에서만 잘 맞고, 학습 크기(ind)가 평가 범위의 가운데라 "가운데로 압축"과 이 설계로는 구별되지 않으며, 학습 크기를 섞은 대조군이 없어 원인 분리가 안 된다. 확신 수준: 현상(먼 크기의 open-loop 이동 방향)은 중간~높음, 원인 층위(디코더/predictor/정보 소실)는 낮음. 이전 메인 세션의 일부 요약은 리뷰에서 과장·부정확으로 지적되었고 6절에서 정정한다.

---

## 2. 실험 개요

- 목적: acwm-phys Cloth Move 에서 보고된 "천 크기 수축"이 DINO-WM 구조(동결 DINOv2 + predictor + 디코더)에서 재현되는지, 그리고 원인이 디코더(후보 D)인지 인코더/predictor 정보(후보 E)인지 진단. 근거: notes/hypotheses.md 후보 D/E 와 "후보 D/E 사전 등록 초안".
- 설정(리뷰가 config/README/코드에서 확인한 범위): baseline_single (학습 분포 ind_train 하나), 학습 시드 0/1/2, 평가 9개 크기 x 100 테스트 에피소드, probe 시드 0~4, num_hist=1, 15 epoch, proprio 항상 0 (skeptic I8; prior 4.1-4). ACWM 대비 천 마스크 IoU 0.68 인 재생성본 (log.md; skeptic I7). 환경 torch 2.7.1+cu128 (skeptic I8, `results.json -> environment.torch`).
- 파이프라인: train -> repro_check -> extract -> diagnose -> probe -> intervene -> aggregate. 3 시드 모두 1회 시도, 에러 로그 없음 (log.md 2026-11-02). 체크포인트는 업로드하지 않고 서버에만 있음.
- 실행 절차의 주의(skeptic C1): 본 실행은 overnight.sh 로 했고 repro 게이트가 꺼져 있어(`repro_gate=off`), 사람이 repro_check 를 보기 전에 진단이 끝났다. repro_check 는 같은 100 에피소드를 다시 잰 것이라 독립 재현이 아니다.
- 사전 등록(hypotheses.md 확정 기록 2, 2026-11-01 사용자 승인): 디코더 면적비 하한 0.9/상한 1.1, 선택성 최소 0.1, 기울기는 CI 가 1 을 제외하는지(여유 0), 모델 시드 0/1/2, probe 시드 0~4, 결과를 본 뒤 기준을 바꾸지 않음. 미확정: `shrink_ratio_tol`(null), probe 프로토콜 선택(i/ii/iii; 세 가지 모두 계산됨). 이 승인은 가설 확정이 아니라 측정·판정 규칙 고정이다.

---

## 3. 결과

표기: `AS:` = `results.json -> across_seeds.baseline_single.metrics.`, `R[s]` = `results.json -> runs.exp001_cs_baseline_single_s{s}`. "리뷰 재계산" 은 해당 리뷰가 results.json 에서 다시 계산했다고 밝힌 값이다. 3 시드 평균은 시드 3개(자유도 2)라 방향성만 신뢰할 수 있다(stats 5절).

### 3.1 디코더 재구성 면적비 decode(encode(real))/real (후보 D)

| 항목 | 값 | 출처 |
|---|---|---|
| k=12, 시드 평균, 크기별 | a026 0.936 [시드 0.915~0.974], a040 0.974, a060 0.973, a080 0.979, ind 0.985, a125 0.983, a160 0.977, a200 0.974, a248 0.974 | log.md 2026-11-02 (results.json 에서 옮긴 값); a026 은 `AS: diagnose/area/groups/a026/by_k/12/ratio_recon_over_real/mean` (prior 1.1) |
| 27칸(9크기x3시드) 점추정 최소/최대 (k=12) | 시드별 최소: s0 0.9719 (a248), s1 0.9150 (a026), s2 0.9200 (a026) | stats 1절 (a1.py), analysis/decoder_ratio_table.csv |
| 전 k(1~12) 점추정 범위 | 최소 0.9351 (s0 a248) ~ 최대 1.0620 (s2 a060) | stats 1절 표 |
| 비의 정의 | 파일의 비는 에피소드별 비의 평균(mean-of-ratio)이다. 평균끼리의 비(ratio-of-means)로 보면 27칸 모두 0.970~1.013 | stats 1절; decoder_ratio_table.csv `ratio_of_means` 열 |
| 에피소드 부트스트랩 CI (k=12) | s1 a026 [0.858, 0.971], s2 a026 [0.873, 0.964] 는 하한 0.9 미만 포함. s0 은 전 크기 CI 하한 0.927 이상 | stats 1절; skeptic C2 (s1 a026 mean 0.915, lo 0.858, hi 0.971) |
| 에피소드 단위 비 < 0.9 비율 (k=12) | a026: s0 0.26, s1 0.44, s2 0.35. a248: 0.00~0.02 | stats 1절 (`per_episode_ratio_recon_k12` 재계산) |

### 3.2 모델 출력/real 면적비, k=12 (평균비 = 에피소드별 비의 평균)

| 크기 | teacher-forced 1-step (tf1) | open-loop (ol) | 출처 |
|---|---|---|---|
| a248 | 0.797 [시드 0.778~0.828] | 0.434 [0.333~0.549] | log.md; ol 은 prior 1.1 (L131555) |
| a200 | 0.854 | 0.535 | log.md |
| a160 | 0.872 | 0.636 | log.md |
| a125 | 0.959 | 0.819 | log.md |
| ind (학습 크기) | 0.970 | 1.054 (시드별 0.991/1.313/0.859, std 0.234) | log.md; 시드별은 prior 1.1 (L132323) |
| a080 | - | 1.373 | log.md |
| a060 | - | 2.920 | log.md |
| a040 | - | 2.965 | log.md |
| a026 | 0.965 | 3.987 [3.799~4.178] | log.md; `AS: diagnose/area/groups/a026/by_k/12/ratio_ol_over_real/mean` (리포터 확인) |

- a026 ol/real 을 평균끼리의 비로 보면 시드별 2.23/2.68/2.19, 시드 평균 2.36 (stats 1절; skeptic I1 이 `AS: .../a026/by_k/12/ol/mean` 1144.64 / `.../real/mean` 484.35 로 계산). a248 은 시드별 0.42/0.56/0.35, 평균비 0.44 로 거의 같다(stats 1절; skeptic I1).
- 학습 크기에서도 ol/real@k12 의 시드 간 변동이 크다: ratio-of-means 로 ind 0.95/1.17/0.85 (stats 5절).
- k=1 (tf1 = ol, 같은 계산): a248 tf1/real 0.722 (시드 0.705/0.732/0.729, `AS: diagnose/area/groups/a248/by_k/1/ratio_tf1_over_real/mean`, prior 1.1 L131495), a026 1.098 (skeptic I3; stats 는 s0 만 1.07, prior 는 s0 1.067 로 인용. 시드 평균 vs s0 차이로 추정).

### 3.3 크기 곡선 기울기 (모델 vs real, 군집 평균 log-log, 9점)

출처: stats 2절 표 (`R[s].diagnose.size_curve.slope_model_vs_real_group_means.<모델>@k{3,6,12}`), 에피소드 부트스트랩 95% CI (학습 시드 변동은 CI 에 없음).

| 모델@k | s0 | s1 | s2 | 3시드 평균(sd) | CI 가 1 제외 |
|---|---|---|---|---|---|
| tf1@3 | 0.748 [0.732,0.763] | 0.756 [0.739,0.772] | 0.736 [0.719,0.752] | 0.747 (0.010) | 3/3 |
| tf1@6 | 0.813 [0.792,0.838] | 0.812 [0.789,0.834] | 0.804 [0.781,0.832] | 0.810 (0.005) | 3/3 |
| tf1@12 | 0.978 [0.938,1.028] | 0.985 [0.945,1.032] | 0.998 [0.957,1.046] | 0.987 (0.010) | 0/3 |
| ol@3 | 0.595 [0.578,0.613] | 0.606 [0.588,0.623] | 0.535 [0.516,0.553] | 0.579 (0.038) | 3/3 |
| ol@6 | 0.417 [0.393,0.442] | 0.468 [0.447,0.491] | 0.355 [0.326,0.385] | 0.413 (0.057) | 3/3 |
| ol@12 | 0.237 [0.187,0.288] | 0.310 [0.264,0.361] | 0.150 [0.085,0.211] | 0.232 (0.080) | 3/3 |
| recon@3 / @6 / @12 | 0.978/0.984/1.011 | 0.981/0.988/1.045 | 0.988/0.988/1.025 | - | k=3,6: 3/3 (효과 약 0.02); k=12: 1/3 (s1,s2 는 1 초과) |

- k=1 기울기는 산출되지 않았다. `size_curve` 의 k_list 가 (3, 6, 12) 이기 때문이다(skeptic I3; diagnose.py:147).
- tf1@k12 기울기는 추정량에 민감하다: 군별 산술평균 면적의 log-log 로 재계산하면 0.905/0.912/0.925 (s0/s1/s2). k=3,6 은 두 방식이 일치한다(stats 2절, a2b.py). 에피소드별 tf1 면적이 results.json 에 없어 이 산술평균 기울기의 CI 는 계산 불가.
- k=12 - k=3 의 tf1 기울기 차이는 0.230/0.229/0.262 로 3시드 재현되어 표본 잡음 인공물이 아니다(stats 2절). 원인(시점별 real 면적 변화, 후기 프레임 레이블 오염, 추정량)은 확정되지 않았다(7절).
- 시드 sd 가 CI 반폭과 같거나 큰 곳은 ol (stats 2절). 3시드 t 검정(df=2): ol@12 t=-16.6 p=0.0036, tf1@12 p=0.158 (stats 는 p 값 과신 금지라고 명시).

### 3.4 Probe (후보 E)

정의를 고정해야 하므로 프로토콜별로 분리한다. 모든 값은 ridge, pool=mean.

| 지표 | gt | tf1 | ol | 출처 |
|---|---|---|---|---|
| heldout_size 기울기, 군집 평균 | 0.876 (3시드 동일) | 0.725 (0.727/0.748/0.699) | 0.388 (0.400/0.451/0.313) | prior 1.2 (L287651, L292499, L290075) |
| heldout_size 기울기, 전체(pooled) | 0.825 | 0.709 | 0.453 | prior 1.2; stats 3.2절은 gt 0.825, tf1 0.71, ol 0.455 로 일치 |
| strat_cv 군집 평균 기울기 | - | - | 0.801 | prior 1.2 (L334715); log.md 의 ridge 0.801 과 일치 |
| strat_cv 9-class 정확도 | 0.472 | 0.497 | 0.450 (stats 시드별 0.450/0.421/0.441; prior 0.437) | skeptic I5, stats 3.1 |
| strat_cv r2_within | 0.877 | 0.830 | 0.418 | skeptic I5; stats 3.1 |
| gt_transfer ol 군집 평균 기울기 / 정확도 | - | - | 0.418 / 0.206 | prior 1.2 |

- 같은 프로토콜에서 control 입력 (strat_cv, ridge): rvit 정확도 0.409 · 기울기 0.996, pix(14x14 RGB 격자) 정확도 0.465 · 기울기 0.951, gt 정확도 0.472 · 기울기 0.987 (prior 1.2; skeptic I5; stats 3.1). 시드 간 동일한 값(gt, rvit, pix, constant)은 학습 시드 복제가 아니라 같은 값의 반복이다(stats 3.3, skeptic I4).
- 선택성(실제 - control) >= 0.1 (stats 3.1절 재계산): label_permute 대비 gt r2 0.988 · acc 0.367 등 전부 0.1 이상. random-ViT 대비 r2_overall +0.022, r2_within +0.061, acc +0.063 (모두 0.1 미만, ridge). pixel 격자 대비 acc +0.006. MLP 에서는 pix control 이 붕괴(r2 -320)하고 MLP 선택성 r2 가 1 을 넘어 해석 불가(stats 3.1; skeptic I5 도 MLP pix 학습 실패를 지적).
- 선택성 키는 존재한다: `probe/pools/mean/{ridge,mlp}/selectivity_real_minus_permute/{gt,tf1,ol}/{r2_overall,r2_within_group,acc_cls,acc_cls_pm1}` (skeptic I5). 파일에는 label_permute 대비만 있고 rvit/pix 대비는 stats 가 직접 계산했다(stats 3.1).
- strat_cv 에서 ol 로 재학습한 probe 의 로그면적 편향: ol a026 +0.24/+0.25/+0.31, a248 -0.17/-0.20/-0.18 (gt 는 +0.07, +0.01) (stats 3.2; skeptic I2 가 s0 값 확인).
- 외삽 역할별 ol 점수(heldout_size role_extrap_large r2_overall -0.313/0.299/-0.847)는 시드 변동이 커서 결론 불가(stats 3.2).

### 3.5 개입 (intervene)

- remove 모드, tf_response_gain: u 방향 0.140~0.185 대 무작위 방향 중앙값 0.101~0.149, 27/27 에서 u 가 높지만 차이 0.029~0.057 (비 1.19~1.54). 무작위 20방향 대비 순위 p 는 0.048~0.143 (최소 가능값 0.048) (stats 4절; prior 1.2 와 값 일치: a248 u 0.159 대 무작위 평균 0.105 [p95 0.139]).
- 디코딩 면적 변화: u 제거 시 프레임 0 에서 로그 -0.003~-0.016 (stats 4절). 현상 크기(ol/real 0.4~4)보다 1~2 자릿수 작다(stats 4절).
- set 모드 영 결과: `R[0].intervene.modes.set_a248_to_a026.a248.tf_decoded_dlogarea_u_by_frame[*].mean` = -0.0051, -0.0029, -0.0056 (skeptic I6; 무작위 방향은 약 0). a248 의 u 투영을 a026 평균으로 옮겨도 디코딩 면적이 약 0.5% 만 바뀐다. 이 결과는 log.md 에 인용되지 않았다.
- a125 의 무작위 gain 0.239 는 방향 15 한 개(2.36~2.57)가 만든 이상치이고, 이 방향은 3개 학습 시드에서 동일하게 쓰였으므로 독립 3회 재현이 아니다(stats 4절). 원인(gain = 출력 이동/입력 이동 비의 분모 폭주)은 stats 와 skeptic 모두 추정이며 미확인.

그림: analysis/fig1_decoder_slopes.png (디코더 비와 기울기), analysis/fig2_probe_controls.png (probe control), analysis/fig3_intervene_remove.png (u 대 20 무작위 방향). 표: analysis/decoder_ratio_table.csv. 모두 stats 가 results.json 에서 만든 것이다.

---

## 4. 사전 등록 규칙을 문자 그대로 적용한 판정표

사전 등록(hypotheses.md 확정 기록 2)을 규칙 그대로 적용했을 때 무엇이 판정되고 무엇이 판정 불가인가. "단위 미정"은 사전 등록에 점추정/CI/k/비의 정의가 적혀 있지 않아 결과가 갈린다는 뜻이다. 그 단위를 지금 정하면 결과를 본 뒤의 결정이므로 사후 기준 조정이다.

| 규칙 | 판정 단위 | 적용 결과 | 상태 |
|---|---|---|---|
| 0. 전제 게이트: a026/a248 수축 재현 (`shrink_ratio_tol`) | 임계값 null (미확정) | 코드는 "TODO" 문자열만 기록(skeptic C1). 판정 기준이 없다. 수치로는 a248 ol/real 0.434, a026 3.987 이 있으나 이를 "재현"이라 부를 기준이 없다 | **판정 불가.** 지금 정하면 사후 조정. 규칙상 게이트 실패/미판정 시 D/E 결론을 내리지 않고 보고만 한다 |
| D-비원인: 재구성비 하한 0.9 (9개 크기 모두) | 점추정, mean-of-ratio, k=12 | 27칸 모두 충족 (최소 0.915) | 충족 (단위 미정) |
| 〃 | 점추정, ratio-of-means | 27칸 모두 0.970 이상 | 충족 (단위 미정) |
| 〃 | 에피소드 부트스트랩 CI 하한 | s1 a026 [0.858, 0.971], s2 a026 [0.873, 0.964] 가 0.9 미만 포함. s0 은 충족 | **미충족 칸 있음** (단위 미정) |
| 〃 | 에피소드 단위 비 | a026 에서 에피소드의 0.26~0.44 가 0.9 미만 | 사전 등록에 이 단위는 없음 (참고 수치) |
| D-비원인: 상한 1.1 | 점추정, 전 k | 최대 1.062 | 충족 |
| D-비원인: "모델 출력은 그보다 유의하게 작다" | k 미정 (1-step 또는 k-step) | 큰 천 a248: tf1 0.797, ol 0.434 로 작음. 작은 천 a026: ol 3.987 (큼), tf1 0.965 (recon 0.915~0.974 와 비슷) | **9개 크기 모두에는 성립하지 않음.** 문구가 "수축"만 가정해 작은 천의 확대에는 적용이 불가 |
| D-원인(반증): 어느 크기든 재구성비 < 0.9 또는 > 1.1 | 점추정 | 해당 없음 | 점추정 기준으로는 반증 아님. CI 기준으로는 a026 s1,s2 가 경계에 걸림 (단위 미정) |
| E-인코더 보유: 선택성 >= 0.1 | control 종류 미지정 | label_permute 대비 전부 통과. random-ViT 대비 ridge r2_overall +0.022, r2_within +0.061, acc +0.063 로 미달. pixel 격자 대비 acc +0.006. MLP 는 지표별 부호 갈림 | **control 미지정이라 판정 불가.** permute 만 쓰면 통과하나 취지를 못 보임 |
| E-predictor 수축: 기울기 < 1 이고 CI 가 1 제외 (여유 0) | 어떤 기울기(size_curve/probe), 어떤 k, tf1/ol, 어느 probe 프로토콜인지 미정 | size_curve: ol k=3,6,12 지지 (3/3 시드), tf1 k=3,6 지지, tf1 k=12 불지지(CI 가 1 포함), recon k=3,6 도 규칙상 "지지"(효과 약 0.02) | size_curve 기준으로는 판정됨. recon 0.98 과 tf1 0.75 를 같은 "수축"으로 취급하는 것이 slope_margin=0 의 약점(stats, prior 일치). probe 기울기는 프로토콜 미선택 |
| E-반증/단서: "k=1 에서 정상이고 k 증가로만 감소하면 오차 누적" | k=1 | k=1 에서 이미 a248 0.722 (정상 아님). k=1 기울기는 산출 안 됨 | **적용 불가.** "k=1 이 이미 비정상"인 경우의 해석이 사전 등록에 없음 (공백) |
| E-사용 여부: 개입 없이 "predictor 가 정보를 버린다"를 결론으로 쓰지 않는다 | - | 개입은 수행되었으나 효과가 작음 (디코딩 면적 변화 0.3~1.6%) | "버린다/쓴다" 어느 결론도 쓰지 않는 것이 규칙과 일치 |
| 시드: 평균 하나로 결론 내리지 않고 시드별 값 병기 | - | ol 기울기 시드별 0.237/0.310/0.150, ind ol/real 시드별 0.991/1.313/0.859 | 규칙을 따르면 시드별 병기가 필수 (3절은 병기함) |

---

## 5. 분석가 간 합의와 불일치

### 5.1 세 명 일치

1. tf1 기울기를 "~1" 로 요약할 수 없다. k=3 0.747, k=6 0.810 이고 1-step 인 k=1 에서도 a248 면적비가 0.72 다. 사전 등록의 단서 조항("k=1 정상, k 증가로만 감소 = 오차 누적")은 그대로 적용되지 않는다. (prior 0절 2, 2.2; stats 2절; skeptic I3)
2. GT 에서 크기 probe 가 높은 것은 DINOv2 특이적 증거가 아니다. random-ViT, 픽셀 격자도 같은 수준으로 읽는다. 선택성 0.1 기준은 control 이 label_permute 일 때만 통과하고 변별력이 없다. (prior 2.4; stats 3.1; skeptic I5)
3. 개입 결과로는 "predictor 가 크기 정보를 쓴다/버린다" 어느 쪽도 결론낼 수 없다. (prior 2.4; stats 4절; skeptic I6)
4. 학습 시드 3개로는 ol 지표의 크기(몇 %)가 불확실하다. 에피소드 CI 가 학습 시드 변동을 반영하지 않는다. 방향성(먼 크기의 이동)은 3/3 일치. (prior 3.3; stats 2, 5절; skeptic I4)
5. 학습 크기를 섞은 대조군(또는 다른 단일 크기 학습)이 없어 "학습 크기 때문"이라는 원인을 분리할 수 없다. (prior 4.1-3; stats 5절; skeptic 6절-1, 제안한 대조군 형태는 다름)
6. 사전 등록의 D 판정이 모호하다: "9개 크기 모두"에서 작은 천은 모델 출력이 작지 않고 오히려 크며, 점추정/CI 단위가 없다. (prior 3.2; stats 1절; skeptic C2)
7. 게이트 `shrink_ratio_tol` 이 null 이라 지금 정하면 사후 기준이다. (prior 3.1; stats 8절-8; skeptic C1)

### 5.2 두 명 일치

- D 의 CI 하한 문제: stats 와 skeptic 은 s1 a026 의 CI 하한(0.858)이 0.9 미만임을 각각 확인했다. prior 는 CI 하한을 확인하지 못했다고 쓰고 시드 최소값 0.915 가 임계에 근접하다고만 적었다.
- num_hist=1 이라 "DINO-WM 일반" 결과로 읽을 수 없다: prior 4.1-4 와 skeptic I8.
- MLP pix control 이 붕괴/학습 실패: stats 3.1 과 skeptic I5.
- "3.99배" 표기의 문제(평균끼리의 비로 2.36): stats 1절(시드별 2.23/2.68/2.19)과 skeptic I1. prior 는 3.99 를 그대로 인용하되 절대 면적 ol 1078 대 ind 2335 로 "학습 크기로 완전히 수렴하지 않는다"고 지적했다.
- GT/rvit/pix probe 가 시드 복제가 아님: prior 1.2, stats 3.3, skeptic I4 (이것은 사실상 세 명 일치).

### 5.3 의견이 갈린 점 (하나로 합치지 않고 병기)

| 쟁점 | analyst-prior | analyst-stats | analyst-skeptic |
|---|---|---|---|
| 후보 D "디코더는 주원인이 아니다" | 확인한 범위(k=3,12)에서 지지, 의도 읽기를 지지한다. 신뢰는 "중간 이상". 모든 k, CI 하한, IoU 가 없음 | 점추정이면 27칸 충족, CI 이면 미충족 칸 있음. "충족으로 단정하지 않고 둘 다 보고". 디코더 손실은 평균 -1~-3% 로 모델 출력 격차보다 한 자릿수 이상 작음 | "충족"은 사전 등록 문장의 절반만 확인한 과장. 범위를 "GT latent 를 읽는 디코더는 정상"으로 좁혀야 함. 예측 latent 를 디코더가 읽는 경우는 미시험 (a248 tf1 k=1 에서 latent 최근접 GT 는 100% 같은 크기 군집인데 면적은 0.722) |
| "학습 크기로 수렴" | 큰 천에서는 잘 맞고 작은 천은 부분적 (a026 ol 1078 대 ind 2335). Kang 2024 식 상수 회귀의 사례일 수 있음 [추정] | 먼 크기(a026, a040, a160~a248)의 방향은 3시드 일치. 학습 크기에 가까운 a080~a125 의 "수축"은 시드 변동과 구분 불가 | 큰 천에만 맞음. 작은 천은 에피소드의 29% 만 ind 쪽이 더 가까움. ind 가 평가 범위 가운데라 "가운데로 압축/평균 회귀"와 구별 불가 |
| gt→ol probe 낙폭의 의미 | strat_cv ol probe 가 기울기 0.80 이라 정보가 상당히 남아 있어 "정보 소실"과 어긋남. 그러나 그리퍼 간격 등 우회 단서일 가능성 [추정] | ol 로 재학습해도 편향이 남음: "소실이 아니라 압축"에 부합 [추정]. heldout_size 에서는 probe 외삽 편향과 predictor 효과가 섞임 | gt 0.877 → ol 0.418 낙폭은 기준선과 무관하게 큼. 그러나 ol probe 의 편향(a026 +0.24, a248 -0.17)은 노이즈 특징에 대한 ridge 의 전역 평균 축소로 설명되며(probe 가 9개 크기 전체로 학습됨), predictor 가 ind 로 끌어당긴다는 근거가 못 됨 [추정, 강하게 의심] |
| tf1@k12 기울기 0.987 의 해석 | 두 극단 군집의 상쇄일 수 있음 [추정] (a248 0.80, a026 0.965) | 추정량 의존: 산술평균 기준 0.905~0.925. 후기 프레임 레이블 상관 0.739 (프레임 12)로 "크기"보다 가림/접힘을 반영 | 9점 직선이 큰 쪽 끝의 휘어짐을 못 잡음 (a248 tf1 0.797). k=3 이 더 낮은 이유는 log.md 가 설명 못함, 작은 천 real 면적이 k 에 따라 줄어서일 수 있음 [추정] |
| k=1 편차의 원인 | 모름. 후보: 초기 프레임 형태 차이, real 면적의 시간 변화 [추정] | tf1 은 오차 누적이 없으므로 k 변화는 시점별 군집 면적 분포 효과일 수 있음 [추정] | predictor 문제인지 디코더가 분포 밖 latent 를 잘못 읽는 문제인지 이 데이터로는 갈리지 않음. 그리퍼 위치/행동 분포 밖일 가능성도 있음 [추정] |
| 추가 실험의 최우선 | heldout_size 의 pix/rvit control, 모든 k 의 D 표, 디코더 IoU, 그리퍼 마스킹 probe, F(크기 혼합) 및 num_hist=3 비교 | 에피소드별 tf1/ol 면적 저장, 학습 시드 확대(>=5, 추정), mix_adjacent/mix_wide, 무작위 방향 100개 이상 | 가운데가 아닌 단일 크기 학습 대조군, 디코더를 거치지 않는 면적 판독과 마스크 검증, 공간적으로 맞춘 패치 교체 개입 |

### 5.4 한 명만 지적한 점

- prior 만: (a) 헬드아웃 크기 probe(heldout_size)에 pix/rvit control 이 없어 "GT 외삽 0.83~0.91"이 DINOv2 특이적인지 알 수 없음. 가장 큰 누락 대조군. (b) gt 값이 3시드 동일해 GT 쪽 정보량이 사실상 1. (c) 문헌 정합성: compounding error(Talvitie 2017, Lambert 2022 등)는 보통 편향이 아니라 오차 증가를 말하며, 학습 분포 쪽 편향은 Kang 2024 식 상수 회귀와 방향이 같을 수 있음 [추정]. 서지 확인 상태는 prior 4.2 에 병기(웹 검색 확인 vs 기억/lit_review 2차 정보 구분). (d) 개입의 "응답 이득" 정의 코드는 읽지 않음.
- stats 만: (a) 작은 천 a026 에서 에피소드의 26~44% 가 디코더 단독으로 10% 이상 작아짐(평균비가 이를 가림). (b) mean-of-ratio 와 ratio-of-means 가 재구성비에서도 다름(예: s1 a026 0.915 대 0.989). (c) recon 기울기 0.98 도 "여유 0" 규칙에서 유의로 판정되는 slope_margin=0 의 약점. (d) 후기 프레임(k=12)에서 log(입자 수)와 log(마스크 면적) 상관이 0.739 로 하락. (e) probe 시드는 독립 표본이 아니고 ol 에서는 학습 시드 sd(r2_within 0.0485)가 에피소드 CI 반폭(0.053)과 같은 크기. (f) 파일의 "pix" 는 14x14 RGB 격자이지 사전 등록의 "픽셀 면적 baseline"(마스크 면적 스칼라)이 아님.
- skeptic 만: (a) 승인된 임계값(decoder_ratio_tol, selectivity_min, slope_margin)이 어떤 .py 에서도 참조되지 않음. 게이트는 overnight 에서 꺼져 있었고 repro_check 는 독립 재현이 아님. hypotheses.md:133 의 체크 표시와 :144 확정 기록 2 의 "미확정"이 불일치. (b) 마스크 검증이 실제 프레임에서만, 순환적(명목 면적비가 같은 계열 HSV 규칙으로 만들어짐). 디코딩된 rollout 프레임의 마스크와 frames_sample.npz 육안 점검은 없음. `mask.min_px: 6` 은 masks.py 에서 미사용. (c) 3.99 의 분해: real 면적 k=1 708.6 → k=12 484.4 (x0.68), ol 755.9 → 1077.7 (x1.43). 모델이 키운 효과는 약 1.4~1.5배이고 나머지는 분모 감소 [내 계산, 추정]. (d) 개입 set 모드 영 결과, remove 의 gain 정의 문제(무작위 방향은 제거량이 맞춰지지 않음, 384차원 등방 대조 방향). (e) 행동·위치 분포 밖(A2), 디코더의 분포 밖 latent 오독(A3) 대안 가설. (f) 수렴 증거(학습 곡선) 없음. (g) in_train 제외가 cloth_sim 에서 아무 일도 안 함(S1), 라벨 정의가 층별 곡선과 한 프레임 어긋남(S3).

---

## 6. 정정 사항 (이전 메인 세션 해석 중 리뷰가 지적한 과장·부정확)

출처는 notes/log.md 2026-11-02 항목의 "해석" 절과 이전 세션 요약이다. log.md 원문은 수정하지 않았다(규칙). 아래는 리뷰가 지적한 내용을 그대로 나열한 것이다.

1. "후보 D: 사전 등록 기준 충족" (log.md 해석 첫 항목)
   - 지적(skeptic C2): 사전 등록 문장은 "9개 크기 모두 0.9 이상이고 모델 출력은 그보다 유의하게 작다"인데 앞 절만 확인했다. 작은 천에서는 모델 출력이 오히려 크다(a026 ol 3.987).
   - 지적(stats 1절): 점추정이면 충족, CI 이면 s1/s2 a026 이 미충족. 사전 등록에 단위가 없다.
   - 정정: "수치상 충족"은 점추정 기준에서만 사실이다. 사전 등록 규칙 전체가 충족되었다고 쓸 수 없다. 디코더 시험 범위는 "GT latent 를 읽는 경우"다.
2. "수축이라기보다 학습 크기(ind) 쪽으로의 수렴"
   - 지적(skeptic I2, prior 0절 3): 큰 천에는 맞고 작은 천에는 부분적이다. ind 가 평가 범위의 가운데라 "가운데로 압축/평균 회귀"와 구별되지 않는다. 학습 크기 시드에서도 ol 이 불안정(ind ol/real@k12 0.859~1.313, prior 1.1).
   - 정정: "먼 크기가 ind 쪽으로 이동하는 방향성이 3시드에서 일치"까지가 방어 가능하다(stats 6절).
3. "teacher-forced 1-step 기울기 ~1"
   - 지적(prior 0절 2, stats 2절, skeptic I3): 이것은 tf1@k12 (0.987)에만 해당한다. k=3 0.747, k=6 0.810 이고 k=1 에서 a248 면적비가 이미 0.722 다. k=12 값도 추정량에 따라 0.905~0.925.
   - 정정: 이전 요약이 "1-step 은 정상(~1)"을 함의했다면 부정확하다.
4. "3.99배" (a026 ol/real)
   - 지적(skeptic I1, stats 1절): 에피소드별 비의 평균이며 분모가 작은 에피소드가 부풀린다. 평균끼리의 비는 2.36 (시드별 2.23/2.68/2.19). 분해하면 실제 천이 작아진 효과가 섞여 있다.
   - 정정: 3.99 를 쓰려면 "에피소드별 비의 평균"임과 평균끼리의 비 2.36 을 함께 적는다.
5. probe 수치 "gt 0.83 → tf1 0.73 → ol 0.40"
   - 지적(prior 1.2): ridge 의 군집 평균/pooled 어느 한 정의로도 정확히 재현되지 않고 정의가 섞인 것으로 보인다 (군집 평균: 0.876/0.725/0.388, pooled: 0.825/0.709/0.453).
   - 리포터 확인: log.md 28행은 이 수치를 MLP heldout_size (gt 0.829 / tf1 0.729 / ol 0.401)로 기재했고, 35행 문장은 모델 종류를 명시하지 않았다. prior 는 ridge 와 대조했고 MLP 키는 직접 대조하지 않았다(리포터도 MLP 키 직접 대조는 안 함: 미확인). 어느 경우든 모델(ridge/MLP)과 군집 평균/pooled 정의를 하나로 고정해 인용해야 한다.
6. 선택성 키를 읽지 못했다는 서술 (log.md 30행 "키 경로 확인 못 함")
   - 지적(skeptic I5): 키가 존재한다 (`probe/pools/mean/{ridge,mlp}/selectivity_real_minus_permute/...`). stats 가 값을 읽고 rvit/pix 대비를 직접 계산했다(3.4절).
   - 정정: "읽지 못함"이 아니라 "label_permute 대비로는 전부 0.1 이상이나 random-ViT 대비로는 ridge 에서 미달, control 이 사전 등록에 미지정"이다.
7. 기타 skeptic 이 log.md 에 대해 지적한 것
   - "pixel 기준선 0.588~0.951"은 실패한 MLP 기준선과 정상 ridge 기준선을 한 범위로 묶었다 (skeptic I5; stats 3.1 도 MLP pix 붕괴 확인).
   - 개입 set 모드의 영(null) 결과를 인용하지 않았다 (skeptic I6).
   - 이 항목들은 skeptic 의 지적이며 prior/stats 가 모두 독립적으로 확인한 것은 아니다.
8. 참고: log.md 는 해석을 "추정"으로 표시했고 개입에서 "버린다/쓴다" 결론을 보류했다(skeptic 4절-8 도 인정). 위 정정은 그 표시를 넘어 확정처럼 읽힐 수 있는 문구에 한한다.

---

## 7. 한계와 위험

- 사전 등록 절차: 게이트 null + 임계값 미연결 + 게이트 off 로, 사전 등록 규칙이 코드로 구현된 곳이 없다(skeptic C1). 판정에 쓴 지표·k·통계량이 사전에 고정되지 않았다(forking paths; skeptic I3). 탐색 결과를 골라 쓰면 선택 편향이 생기므로 탐색은 탐색이라 표기해야 한다(stats 5절).
- 시드: 학습 시드 3개(자유도 2). 에피소드 CI 는 학습 시드 변동을 반영하지 않는다. ol@12 기울기 시드 간 범위 0.16 이 단일 시드 CI 폭(약 0.10, s0)보다 크다(skeptic I4). 시드 평균의 t 기반 95% CI: ol@12 [0.033, 0.431], tf1@12 [0.962, 1.012] (stats 5절).
- 설계 교란: 학습 크기 ind 가 평가 범위의 가운데. 대조군 없음. 행동·그리퍼 위치 분포 밖 가능성 미통제(skeptic I8, A2). num_hist=1, proprio 0, 15 epoch, 학습 곡선 없음 (skeptic I8). 일반화 범위는 "acwm-phys 설정의 DINO-WM 변형"이다.
- 측정 타당성: 면적은 HSV 마스크로 측정하고 실제 프레임에서만 검증, 순환적이다(skeptic I7). 디코딩된 rollout 프레임의 마스크 정확도 미검증. frames_sample.npz 육안 점검 없음 (서버 캐시에만 있음). ACWM 대비 IoU 0.68 이라 "ACWM-Phys 현상의 재현"이라고 쓰려면 렌더러·동역학 차이를 병기해야 한다.
- 레이블: 후기 프레임(k=12)에서 마스크 면적과 입자 수 상관 0.739 로, "크기"보다 가림/접힘이 섞인다 (stats 2절).
- probe: random-ViT/pix 와 구분되지 않아 DINOv2 특이적 주장 불가. 9-class 정확도 0.47 이 높은지 낮은지를 판정할 오라클 상한이 없다 (stats 3.3). probe 라벨이 천 면적이 아니라 그리퍼 간격 같은 상관 단서를 읽을 가능성은 어떤 파일에서도 검사되지 않았다 (prior 2.4, 추정).
- 개입: u 는 mean-pool 선형 방향 하나이고 평행이동으로 공간 패턴을 못 바꾼다 (skeptic I6, 추정). 효과가 작아 사용 여부 판단 불가.
- 문헌: prior 가 서지를 직접 확인한 것은 Bengio 2015, Talvitie 2017, Lambert 2022, Kang 2024(ICLR), Kang 2025(ICML), Hewitt & Liang 2019 의 초록/요약 수준이다. DINO-WM, LeWM, SG-JEPA, acwm-phys 는 lit_review 를 통한 2차 정보이고 원문 미독이다 (prior 0절, 7절).
- 재현성 문서: README 가 "실행 전" 등으로 낡았고, results.json 에 작업 트리 dirty 여부가 없으며, overnight_status.json 이 갱신되지 않은 버그가 있다 (skeptic S5, log.md).

---

## 8. 사람이 결정할 사항

에이전트의 제안은 승인이 아니다. 아래는 세 분석가의 요청을 중복을 합쳐 번호로 정리한 것이며, 괄호는 제안한 분석가다. 어느 쪽을 택하든 결과를 본 뒤의 결정이므로 "사후 기준 조정"으로 기록해야 하는 항목에는 (사후)를 붙였다.

1. 0번 전제 게이트 처리 (prior, stats, skeptic 일치). (a) "판정 불가(임계값 미설정, 게이트 off)"로 공식 기록하고 D/E 관찰을 "전제 미확인 조건부"로 둘지, (b) 사후 기준임을 명시하고 임계값을 정할지 (사후). 또 hypotheses.md:133 체크 표시와 :144 확정 기록 2 의 불일치를 바로잡을지 (skeptic).
2. D 규칙의 판정 단위 확정 (prior, stats, skeptic 일치) (사후): 점추정/CI 하한/에피소드 단위, mean-of-ratio vs ratio-of-means, 평가 k(12 만/전 k), "9개 크기 모두 모델 출력이 유의하게 작다"를 작은 천(출력이 커짐)에 문자 그대로 적용할지, 의도("디코더 이후 단계에서 벌어진다")로 읽을지 (prior).
3. E 규칙의 판정 대상 확정 (stats, skeptic) (사후): 선택성 0.1 을 어느 control(label_permute/random-ViT/pixel 격자/마스크 면적 스칼라)에 적용할지 (stats); 기울기 판정에 쓸 기울기(size_curve vs probe 의 slope_group_means), k, tf1/ol, probe 프로토콜 (skeptic); 효과 크기 하한(예: 1 - 기울기 > 0.1)을 추가할지 (stats); tf1 기울기 추정량(로그의 평균 vs 산술평균 로그) (stats).
4. "k=1 에서 이미 수축"인 경우의 해석 규칙 신설 여부 (prior) (사후). 사전 등록에 공백이다.
5. 면적비 주 지표를 에피소드별 비의 평균에서 평균끼리의 비/로그비/중앙값 중 무엇으로 할지, 보고서·log.md 의 "3.99배" 표기를 유지할지 (skeptic, stats 지표 정의 지적).
6. 정정 반영 방식 (skeptic, prior): log.md 2026-11-02 해석 문구를 직접 고칠지, 리뷰 의견으로만 병기할지. probe 수치(gt/tf1/ol)는 모델(ridge/MLP)과 정의(군집 평균/pooled)를 하나로 고정해 재기재할지 (prior). 이 보고서는 log.md 기존 항목을 수정하지 않았다.
7. 저비용 추가 분석 승인 (CPU 위주 추정; GPU 필요 여부는 prior 가 판단하지 못함, skeptic 은 2번이 "GPU 작음"이라 함):
   - heldout_size 의 pix/rvit control, 모든 k 의 D 비율 표, 디코더 IoU, 그리퍼 마스킹 probe (prior).
   - 에피소드별 tf1/ol 면적을 results.json 에 저장 (stats; 산술평균 기울기 CI 계산용).
   - 예측 latent 의 latent 최근접 GT 면적 판독 및 rollout 디코딩 프레임 50~100장 수동 라벨링 마스크 검증, k=1 기울기 추가 (skeptic).
   - 개입의 무작위 방향 20 → 100 이상과 집계를 중앙값/순위로 변경, a125 방향 15 점검 (stats).
   - 행동·x0 를 크기별로 비교하는 분포 점검 (GPU 불필요, skeptic).
8. **GPU/비용이 큰 실험 승인 (사람 승인 필요; 에이전트가 임의로 실행할 수 없음)**:
   - 가운데가 아닌 크기 하나로 학습한 대조군 (예: a040_train 또는 a160_train 단일 크기, 같은 예산, 시드 3개) (skeptic). 학습은 시드당 약 5.5시간 (skeptic 이 overnight.log 의 seed 0 학습 05:20 → 10:55 로 환산한 값; 리포터 직접 확인 안 함).
   - mix_adjacent/mix_wide 크기 혼합 학습 (후보 F) (prior, stats) 및 num_hist=3 비교 (prior).
   - 학습 시드 확대 (stats: ol 근거라면 5 이상 권장 [추정]).
   - 공간적으로 맞춘 패치 교체/합성 확대·축소 개입 (skeptic; 소규모 GPU).
9. 서버 산출물(frames_sample.npz, 추출 캐시, 체크포인트) 확보 여부, `sync --ckpt --cache` (skeptic). 시각 점검과 재분석에 필요.
10. 논문에 들어갈 주장 범위 확정 (prior, 사람 승인 필요 지점). 아래 9절의 제안을 참고하되 확정은 사람이 한다.

---

## 9. 논문 주장 범위에 대한 제안 (확정 아님)

현재 리뷰로 방어 가능한 범위(단, 시드 3개·단일 학습 분포·num_hist=1 설정 한정이라는 단서를 붙여서):
- ind 단일 크기로 학습한 baseline 에서 먼 크기의 open-loop k=12 면적이 ind 쪽으로 이동하는 방향성이 3개 시드에서 일치한다(큰 천 a248 ol/real 0.434, 작은 천은 방향이 반대로 커짐). 기울기 CI 는 ol 에서 모든 k 로 1 을 제외한다. 크기(몇 %)는 시드 의존이 크다. (stats 6절, prior, skeptic 4절-8 "수치 방향은 세 시드 모두 일관")
- GT latent 를 디코더가 재구성할 때 평균 면적 손실은 -1~-3% 수준으로 모델 출력 격차보다 한 자릿수 이상 작다. 단 작은 천에서는 에피소드의 26~44% 가 10% 이상 작아지고 CI 하한이 0.9 에 걸린다. (stats 1절)
- teacher-forced 1-step 에서도 큰 천은 이미 줄어 있다(k=1 a248 0.722; k=3,6 기울기 0.747, 0.810). 따라서 "오차 누적만으로 설명된다"는 해석은 지지되지 않는다. (prior, stats, skeptic 일치)
- 크기 probe 정확도·기울기는 random-ViT 와 픽셀 격자도 비슷하므로 "DINOv2 가 크기를 특이하게 보유"라고 쓸 수 없다. (3명 일치)
- 크기 방향 개입은 효과가 작았다(디코딩 면적 변화 0.3~1.6%, set 모드 약 0.5%). (stats, skeptic)

보류할 주장:
- "디코더는 수축의 원인이 아니다"를 일반적으로 주장하는 것 (GT latent 한정으로만 가능; 예측 latent 의 디코더 오독은 미시험. skeptic C2).
- "학습 크기로 수렴한다"는 기제 주장 ("가운데로 압축"과 구별 불가, 대조군 없음; skeptic I2, stats 5절, prior).
- "predictor 가 크기 정보를 버린다/쓴다" (개입 증거 약함; 규칙상으로도 결론 금지).
- "1-step 은 정상" 또는 "tf1 기울기 ~1".
- "D/E 사전 등록 기준 충족", "전제 게이트 통과".
- "DINO-WM 의 구조적 한계" (num_hist=1 설정, 대조군 없음; prior, stats).
- "작은 천이 3.99배로 커진다"는 문구 (지표 정의 의존; 평균끼리의 비 2.36).
- "ACWM-Phys 수축 현상의 재현" (IoU 0.68, 렌더러·동역학 차이; skeptic I7).

---

## 10. 다음 실험 후보 (제안만; 실행은 8절의 승인 필요)

1. 가운데가 아닌 단일 크기 학습 대조군 (skeptic; GPU, 시드당 약 5.5시간): 큰 천/작은 천 출력이 새 학습 크기로 모이면 "학습 크기로 수렴", 여전히 가운데 쪽이면 "평균 회귀"를 지지. 후보 F(크기 혼합: mix_adjacent/mix_wide)도 같은 목적 (prior, stats).
2. 디코더를 거치지 않는 면적 판독과 마스크 검증 (skeptic; 캐시 기반, GPU 작음): latent 최근접 GT 면적 vs 디코딩 면적, rollout 프레임 수동 라벨링. 재분석: 평균끼리의 비/로그비/중앙값, k=1 기울기.
3. 공간적으로 맞춘 인과 개입 (skeptic): 천 패치 토큰 교체 또는 입력 합성 확대·축소, 데이터 부분공간에서 분산을 맞춘 대조 방향.
4. 부족 control 보강 (prior, 저비용 추정): heldout_size 의 pix/rvit, 마스크 면적 스칼라 baseline (stats), 그리퍼 마스킹 probe, 디코더 IoU.
5. 부분 teacher forcing 으로 편차가 생기는 rollout 길이 특정 (prior 4.1-6), 위치 등 다른 변수의 학습 크기 쪽 편향과 비교 (prior 4.1-2).
6. num_hist=3 baseline 비교 (prior), 학습 곡선 확인 (skeptic I8).
7. 개입 재설계: 무작위 방향 100개 이상, 중앙값/순위 집계 (stats).

---

## 부록. 키 경로 확인 상태

- 리포터가 results.json 에서 직접 확인: `across_seeds.baseline_single.metrics["diagnose/area/groups/a026/by_k/12/ratio_ol_over_real/mean"]` (mean 3.987, min 3.799, max 4.178), 키 `diagnose/area/groups/a026/by_k/12/ratio_recon_over_real/mean` 및 `diagnose/area/groups/a248/by_k/1/ratio_tf1_over_real/mean` 의 존재와 prior 가 적은 행 위치(L126191, L131495).
- 나머지 수치는 log.md(results.json 에서 옮긴 값), 리뷰의 키 경로/재계산값, analysis/decoder_ratio_table.csv 에서 옮겼으며, 리포터가 results.json 과 일일이 대조하지는 않았다.
- 분석가 간 수치 표기 차이(미해결): a026 tf1/real@k=1 (skeptic 1.098, stats 1.07, prior 1.067; 시드 평균 vs s0 로 추정), ol strat_cv 정확도 (prior 0.437, stats 0.450 시드 s0). 리포터는 어느 쪽도 고르지 않았다.
