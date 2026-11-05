# 002_single_size_sweep 통계/데이터 리뷰 (analyst-stats)

- 작성: 2026-11-05. 관점: 통계와 데이터. 다른 분석가의 002 리뷰는 읽지 않았다.
- 입력: experiments/002_single_size_sweep/results.json (유일한 수치 출처), config.yaml, src/exp002/{aggregate,stats,q3}.py, README.md, notes/hypotheses.md(확정 기록 3, 5, 6, 7), notes/log.md(2026-11-04, 05의 002 항목).
- 재계산 코드: experiments/002_single_size_sweep/analysis/stats_recheck.py (표준 라이브러리만, `python3 -I` 로 실행). 전체 출력: analysis/stats_recheck_output.txt. 아래 수치의 경로는 `decision_table[id].result...`, `q1.cells[key][role]`, `q2_layerwise`, `exploratory.q2_layer_time_grid_r2_point_estimates` 이다.
- 재계산하지 못한 것: 에피소드별 충분통계(cells.npz), q3.json 원자료, 학습 로그는 Drive 에만 있다. 따라서 **bootstrap CI 와 bootstrap p 는 results.json 에 저장된 값을 일관성만 점검했고 독립 재생성하지 못했다**. Holm, 판정 규칙, 시드 단위 통계는 재계산했다.

## 0. 요약 (핵심 5줄)

1. 판정 표 14행의 status 와 Holm 조정 p 는 저장된 p 로부터 모두 재현된다(불일치 0). 기계적 오류는 못 찾았다.
2. 그러나 "pass" 중 **cond_a_vs_b:last_block 은 학습 시드 단위로 보면 유의하지 않다**(Welch p=0.155, 시드 단위 95% CI [-0.0212, 0.0049] 가 0 포함). 에피소드 bootstrap p=0.002 는 학습 시드 변동을 담지 않아 과대 확신이다. 효과 크기도 R² 0.008 로 작다.
3. H-a 는 역할에 따라 결과가 갈린다. 보간 linear 는 강하게 pass(ΔR²=0.556, CI 하한 0.352), 보간 attentive 는 점추정만 마진 통과(CI 하한 0.015 < 0.1, 마진을 CI 로는 못 넘음), 외삽은 linear 가 유의하지만 작고(0.034, CI 상한 0.067 < 0.1) attentive 는 mask baseline 보다 나쁘다(-0.125). R² 마진은 타깃 분산(보간 SD 0.22, 외삽 SD 0.98)에 크게 좌우되므로 역할 간 R² 차이를 효과 크기 차이로 읽으면 안 된다.
4. H-b rollout fail 과 (a) 대 (b) rollout 차이(0.422)는 학습 시드 3개 모두에서 같은 방향이며 시드 단위로도 견고하다. H-b last_block pass 는 임계 0.2 대비 여유가 매우 크지만(drop -0.0085) 층별 R² 변화가 전부 ±0.013 이내라 정보량이 적다.
5. H-c steering pass 는 6칸 모두 순위 p 가 하한(1/101)이고 Spearman=1.000 이지만, 읽는 probe 와 방향이 같은 계열이라 구분력이 약하고 단조 기준은 이 설계에서 변별력이 없다. subspace_removal 은 not_evaluable 이며 효과값을 근거로 쓰면 안 된다. 전체 확신도는 중간(아래 11절).

## 1. 판정 표 재계산

재계산: stats_recheck.py "Holm recompute", "H_a recompute of status", "H_b recompute".

| id | 저장 status | 재계산 | 비고 |
|---|---|---|---|
| H_a:interp:linear_pooled | pass | 일치 | ΔR²(mask)=0.5560, CI [0.3517, 0.7777]; Δ(random-init)=0.3399, CI [0.2060, 0.4886] |
| H_a:interp:attentive | pass | 일치 | ΔR²(mask)=0.2246, CI [0.0153, 0.4545]; Δ(random-init)=0.1999, CI [0.0650, 0.3441] |
| H_a:extrap:linear_pooled | fail | 일치 | ΔR²(mask)=0.0339, CI [0.0073, 0.0672] < 마진 0.1 (criteria_before_holm=false) |
| H_a:extrap:attentive | fail | 일치 | ΔR²(mask)=-0.1245, CI [-0.1512, -0.0901] |
| H_b:last_block | pass | 일치 | drop 평균 -0.00851 |
| H_b:last_rollout_step | fail | 일치 | drop 평균 0.30238, 3시드 모두 > 0.2 |
| H_c:steering | pass | 일치(저장 필드) | 6칸 모두 백분위 밖, 단조 |
| H_c:subspace_removal | not_evaluable | 일치 | 6칸 모두 not_removed |
| cond_a_vs_b:last_block | pass | 일치 | 4c 게이트 medium |
| cond_a_vs_b:last_rollout_step | pass | 일치 | |
| diag_4c single_ind / multi | decided (medium / large) | 일치 | 5절 |

- 행 수 집계(pass 6, fail 3, not_evaluable 2, decided 2, exploratory 1 = 14)는 log.md 와 같다.
- 판정 규칙이 aggregate.py 와 일치함을 손으로 다시 적용했다: H_a 는 (Δmask >= 0.1) AND (CI 하한 > 0, 두 대조 모두) AND Holm. 기대와 저장값이 4행 모두 같다.
- 일관성: 각 `r2_real_per_probe_seed` 평균 = 저장 point (예: 0.513667 = 0.513667). 점추정이 모두 해당 CI 안에 있다. Δ = real - baseline 이 점추정 차이와 같다.
- **p 의 정의에 주의**: H_a 의 p 는 "Δ <= 0" 귀무에 대한 것이지 마진 0.1 에 대한 것이 아니다. 마진은 점추정 비교로만 들어간다. 그래서 extrap linear 는 Holm 조정 p=0.021 로 "0 과 다르다"가 기각되지만 마진 미달로 fail 이다(유의하지만 작은 효과). 반대로 interp attentive 는 Holm p=0.030 으로 통과하지만 CI 하한(0.015)이 마진 0.1 아래라 "효과 >= 0.1" 이라는 주장은 CI 가 지지하지 않는다.

## 2. Holm 보정 확인

재계산: stats_recheck.py "Holm recompute" (자체 구현, aggregate 와 별개).

| 패밀리 | 크기 m | 원 p (정렬) | 재계산 조정 p | 저장 조정 p |
|---|---|---|---|---|
| H_a | 4 | 0.000999, 0.00699, 0.01499, 1.0 | 0.003996, 0.020979, 0.029970, 1.0 | 동일 |
| H_b | 2 | 0.000999, 1.0 | 0.001998, 1.0 | 동일 |
| H_c | 2 | 0.00990, 1.0(not_evaluable) | 0.019802, 1.0 | 동일 |
| cond_a_vs_b | 2 | 0.001998, 0.001998 | 0.003996, 0.003996 | 동일 |

- 모두 일치. not_evaluable 을 p=1 로 두고 m 에 포함하는 규칙(사전 등록 크기 유지)도 의도대로 작동. H_c steering 은 m=1 이면 조정 p=0.0099 로 더 작아지므로 m=2 는 보수적 방향이다.
- **p 의 해상도 한계**: bootstrap B=1000 이면 p 하한이 1/1001=0.000999, 양측은 0.001998, 순위 검정 100개는 1/101=0.0099. 보고된 p 중 다수(H_a interp linear, H_b last_block, cond 두 행, H_c 6칸 전부)가 하한에 붙어 있어 "p=0.001" 은 "이 B 로 더 구분 못 함"이지 정밀한 값이 아니다. 효과 크기 해석은 p 가 아니라 추정치와 CI 로 해야 한다.
- 패밀리 간 보정은 없다(사전 등록 사항). 전체로 보면 검정 10개(H_a 4 + H_b 2 + H_c 2 + cond 2)이고 이 중 p 가 하한 근처가 아닌 것은 H_a interp attentive, extrap linear 정도다. 가설 간 다중성은 결론을 바꾸지 않는 수준이지만 보정 없이 통과한 것은 사실이다.

## 3. 효과 크기와 CI

재계산: stats_recheck.py "Q1 late-window table", "H_b recompute". 출처: q1.cells["log_nominal_area_ratio|<probe>|<rep>|all|late"][role].

### 3.1 H-a (late_deformed 프레임, 주 타깃 log 명목 면적비)

| 역할 | probe | R² real [CI] | R² random-init | R² mask baseline | Δmask (CI) | MAE real / mask / rand |
|---|---|---|---|---|---|---|
| interp (200 ep) | linear | 0.514 [0.422, 0.595] | 0.174 | -0.042 | 0.556 [0.352, 0.778] | 0.118 / 0.167 / 0.152 |
| interp | attentive | 0.182 [0.097, 0.255] | -0.018 | -0.042 | 0.225 [0.015, 0.454] | 0.178 / 0.167 / 0.179 |
| extrap (400 ep) | linear | 0.859 [0.848, 0.869] | 0.789 | 0.825 | 0.034 [0.007, 0.067] | 0.281 / 0.319 / 0.377 |
| extrap | attentive | 0.700 [0.697, 0.703] | 0.620 | 0.825 | -0.125 [-0.151, -0.090] | 0.482 / 0.319 / 0.535 |

(MAE 는 q1.cells 의 `mae.point`; mask baseline 의 MAE CI 는 저장되어 있지 않다.)

관찰과 판단:
- **R² 마진은 타깃 분산에 종속적이다.** 보간 레벨(a080, a125)의 log 면적 SD 는 약 0.222, 외삽 4레벨은 약 0.983 이다(config 의 area_ratio 로 계산, 레벨 동일 가중). R² 는 이 SST 로 나뉘므로 같은 절대 오차 개선이 보간에서는 큰 ΔR², 외삽에서는 작은 ΔR² 이 된다. 실제로 linear probe 의 MAE 개선(mask 대비)은 보간 0.049, 외삽 0.038 (log 단위)로 크기가 비슷한데 ΔR² 는 0.556 대 0.034 이다. 즉 "보간은 되고 외삽은 안 된다"는 R² 판정의 대비는 부분적으로 지표의 성질일 수 있다(추정). 임계값(0.1)을 바꾸자는 뜻이 아니라, 사전 등록된 보조 지표 MAE 를 함께 읽어야 한다는 뜻이다.
- **mask baseline 의 R² 가 창에 따라 급변한다.** 보간에서 early 0.772 → late -0.042, 외삽에서 0.903 → 0.825 (h_a_early_flat_report 와 q1.cells). 보간 late 의 pass 는 DINO 가 강해서라기보다 baseline 이 변형 프레임에서 붕괴한 영향이 크다. DINO linear 도 early 0.869 → late 0.514 로 떨어진다.
- **random-init 인코더만으로도 보간 linear 에서 mask baseline 을 넘는다**(0.174 vs -0.042, 차 0.216 > 0.1). random-init 대조는 CI 하한 > 0 만 요구하고 마진이 없으므로, interp pass 는 "사전학습이 있어야만 성립한다"가 아니라 "ViT 패치 특징 + ridge 가 면적 1개 특징 ridge 보다 낫다"에 가깝다. 사전학습의 순수 기여는 real - random-init = 0.340 (linear, CI [0.206, 0.489]) 이다.
- **attentive probe 는 불안정하다.** 보간 attentive 의 probe 시드별 R² 는 0.108~0.271 (SD 0.0697), random-init 은 -0.061~0.046 (SD 0.0501). 보간 attentive 가 linear 보다 낮고(0.18 vs 0.51), early 창에서는 real(0.312) 이 random-init(0.586) 과 mask(0.772) 보다 낮다. MAE 로는 attentive 가 mask baseline 과 같거나 나쁘다(보간 0.178 vs 0.167, 외삽 0.482 vs 0.319). 따라서 보간 attentive "pass" 는 R² 지표에 한정된 약한 증거다. 다만 5개 probe 시드 각각의 Δmask 는 모두 0.1 이상이다(0.246, 0.152, 0.157, 0.255, 0.314, 계산: real per-seed - mask per-seed).
- linear probe 의 5개 probe 시드는 값이 완전히 같다(SD=0, ridge 결정론). "probe 시드 5개"는 linear 에서 독립 반복이 아니다.
- 외삽 linear: CI 상한 0.067 < 마진 0.1. 즉 "유의하지만 마진 미달"일 뿐 아니라 episode 표본 변동 안에서 효과가 0.1 보다 작다고 말할 수 있다(단, 이 CI 는 probe 학습 표본/시드 변동을 담지 않는다).
- 순서 정확도(pair_order_accuracy)는 보간 attentive 0.972 로 높다. R² 가 낮아도 순서 정보는 읽힌다는 뜻이며 R² 단독 판정의 한계를 보여 준다(점추정만, CI 없음).

### 3.2 H-b (조건 (b), 미지 크기 보간+외삽 합, linear probe)

| 점 | 시드별 drop (s0, s1, s2) | 평균 | 시드 SD | 에피소드 bootstrap CI | 시드 단위 t 95% CI (df=2) |
|---|---|---|---|---|---|
| last_block | -0.0035, -0.0137, -0.0083 | -0.00851 | 0.00513 | [-0.0106, -0.0062] | [-0.0212, 0.0042] |
| last_rollout_step | 0.2913, 0.2953, 0.3205 | 0.30238 | 0.01585 | [0.2822, 0.3240] | [0.2630, 0.3418] |

- 두 점 모두 임계 0.2 와의 관계가 시드 단위 CI 로도 바뀌지 않는다(last_block 은 0.2 보다 훨씬 아래, rollout 은 0.2 보다 위). 판정은 견고하다.
- last_block 의 drop 은 음수(Q2 R² 가 Q1 천장보다 높음)이고 시드 단위 CI 는 0 을 포함한다. "정보가 보존된다"는 약한 형태(손실이 0.2 보다 훨씬 작다)로만 지지된다. 층별 R²(unseen)는 layer 0 에서 0.8728 로 거의 같고 이후 0.8649~0.8754 사이에서만 움직인다(q2_layerwise). 변화 폭이 시드 SD 와 비슷해 층별 패턴(예: layer 2-3 에서 소폭 하락, layer 6 에서 상승)을 해석할 근거가 없다.
- rollout drop 은 horizon 에 매우 민감하다(exploratory 격자, 점추정만, p 없음). (b) linear 의 rollout R² 는 step 1 에서 0.853/0.866/0.852 였다가 step 12 에서 0.529/0.525/0.500 으로 단조 감소한다. fail 은 사전 등록된 horizon 12 에서의 결과이며, 다른 horizon 에서 같은 결론이라고 일반화하면 안 된다.
- rollout 의 R² 는 인코더 토큰으로 학습한 probe 를 predictor 출력에 그대로 적용한 값이다(output-space probe). drop 의 일부는 정보 손실이 아니라 분포 이동(예측 latent 가 더 매끈함/편향됨)으로 인한 probe 오보정일 가능성이 있다(추정, 이 머신에서 검증 불가).

### 3.3 조건 (a) 대 (b)

| 비교 | (a) 시드별 | (b) 시드별 | 차이 | 에피소드 bootstrap CI | 시드 단위 Welch (t, df, p) / 95% CI |
|---|---|---|---|---|---|
| rollout drop | 0.7385, 0.7059, 0.7289 | 0.2913, 0.2953, 0.3205 | 0.4221 | [0.3986, 0.4456] | t=31.7, df≈4.0, p<0.0001 / [0.385, 0.459] |
| last_block change_vs_layer0 | 0.0062, -0.0059, -0.0018 | 0.0026, 0.0129, 0.0075 | -0.00817 | [-0.0103, -0.0061] | t=-1.76, df≈3.9, p=0.155 / [-0.0212, 0.0049] |

- rollout: 6개 값이 완전 분리(최소 a 0.706 > 최대 b 0.321)이고 효과가 시드 SD 의 수십 배다. 시드 3+3 의 정확 순열 검정 최소 양측 p 는 0.10 이라 순위 기반 유의성은 원리적으로 불가능하지만, 크기와 분리로 보아 방향과 크기는 신뢰할 만하다. 이 차이는 학습 데이터 분포뿐 아니라 probe 프로토콜(출력 공간)에 의존한다.
- **last_block 의 pass 는 시드 단위로는 뒷받침되지 않는다.** 두 조건의 시드 범위가 겹치고, 시드 단위 CI 는 0 을 포함한다. 에피소드 bootstrap 은 평가 에피소드만 재표집하므로 학습 시드 SD(0.005~0.006)를 반영하지 못해 CI 가 약 6배 좁다(폭 0.0041 vs 시드 단위 Welch 0.0261). 게다가 효과 크기 자체가 R² 0.008 이고 (a) 쪽 probe 는 (a) predictor 가 본 적 없는 크기의 활성에서 학습된 것(한계 문구, Q2L)이다. 이 행의 "pass = 차이 검출"은 결론에 쓰지 않는 것이 맞다.
- 4c 게이트 때문에 이 행이 평가되었는데, 게이트가 임계 바로 아래(5절)였다.

## 4. 학습 시드 간 변동

- 학습 시드는 3개(0, 1, 2). 시드 수가 적어 SD 추정 자체가 불안정하다(df=2).
- 시드 간 SD: rollout 최종 R²(b) = 0.0158 (0.5290, 0.5250, 0.4998), (a) = 약 0.017 (0.0818, 0.1144, 0.0914), last_block drop(b) = 0.0051, Q3 steering T 6칸 CV = 0.19 (평균 0.073, SD 0.014).
- 이 값들은 H-b/H-c/cond 의 판정 임계와의 거리에 비해 작다. 단 last_block 같은 미세 효과(<0.01)는 시드 SD 와 같은 크기다.
- 학습 중단과 재개: log.md 에 따르면 multi s0 는 epoch 5, s1 은 epoch 9 에서 재개되었고 s2 는 재개 없이 진행되었다(로그 기록; 이 머신에서 재확인 못 함). rollout R² 가 s2 에서 가장 낮은 것(0.4998 vs 0.529, 0.525)은 재개 여부와 일치하지 않는 방향이라 재개가 결과를 체계적으로 끌어올린 흔적은 없지만, n=3 으로는 분리할 수 없다. 또 6개 run 모두 `val()` 의 no_grad 가 빠진 패치로 학습되었다는 기록이 있으며 가중치 영향 없음은 추정이다.
- 6개 run 의 총 step 은 모두 27855, 학습 slice 는 모두 44550 으로 동일 예산이 맞다(train_budgets). 단 이 step 수는 계산값이며 optimizer 에서 읽은 값이 아니라고 파일에 명시되어 있다.
- Q1(H-a)에는 학습 시드가 없다(동결 인코더, q1.note). H-a 는 002 의 학습 결과와 무관한 사전학습 DINOv2 의 성질이다.

## 5. 4c 진단 (게이트)

재계산: results.json `decision_table[diag_4c:*].result.per_training_seed`.

- single_ind 의 a060 최대 활성 이동/노이즈 비율: 4.912, 4.661, 4.657 (시드 0,1,2). 임계 5.0 보다 각각 1.8%, 6.8%, 6.9% 아래다. 시드 0 은 임계에 거의 붙어 있다. 이 값이 5 를 넘었다면 게이트가 large 가 되어 cond_a_vs_b:last_block 은 not_evaluable 이 된다. 게이트 결과가 시드 1개의 2% 내 차이에 놓여 있다.
- multi_60_ind_160 에서는 같은 지표가 5.04~5.30 으로 5 를 넘어 large 로 분류되었다. 그런데 (b) 는 a060 과 a160 을 학습에서 본 조건이다. layer 0(입력)의 비율이 두 조건에서 3.82 로 사실상 같아, 이 지표는 predictor 의 낯섦보다 입력 영상 자체의 차이를 주로 재는 것으로 보인다(추정). 따라서 이 4c 진단의 변별력은 낮고, large/medium 구분을 "낯선 입력의 영향"으로 읽을 근거가 약하다. log.md 의 지적 (4)와 같은 문제다.
- 다음 프레임 오차비는 단서가 된다. a160/ind 오차비가 (a) 1.73~1.74 (임계 2.0 의 약 87%), (b) 1.10~1.12 이다. (a) 가 a160 에서 실제로 더 낯설다는 증거인데 이 항목은 임계를 넘지 않아 분류에 반영되지 않았다.

## 6. CI 가 담지 못하는 불확실성

results.json 의 `ci_caveat` 가 명시한 것: 에피소드 재표집만 담고 학습 시드 변동은 없다. 추가로 내가 확인한 것:

1. **학습 시드(n=3)**: 3.3절에서 보았듯 미세 효과의 유의성을 뒤집는다. 큰 효과(rollout)는 영향 없음.
2. **probe 학습 표본/시드**: bootstrap 은 학습된 probe 를 고정하고 평가 에피소드만 재표집한다. probe 는 900 학습 에피소드로 1회 학습한다(레벨 a060/ind/a160). attentive 의 probe 시드 SD 는 R² 0.07 (보간)로 시드 5개 평균의 표준오차가 약 0.03 이다. 이는 보간 attentive ΔR² 의 CI 폭(0.44) 안에 들어가지만 CI 하한이 이미 0.015 라 여유가 없다.
3. **평가 표본이 레벨당 100 에피소드로 고정**: 보간 200, 외삽 400 에피소드. 레벨별 이질성(예: a026 대 a248)은 R² 한 숫자로 합쳐졌다. 레벨별 결과는 results.json 에 없고 Drive 의 `q1/test_rows.json` 과 `cells.npz` 에서만 얻을 수 있다.
4. **Q3 의 평가 표본**: 레벨당 25 에피소드(q3_impl.episodes_per_level)에 프레임 3개. T 는 샘플 중앙값의 중앙값이고 에피소드 불확실성은 계산되지 않는다. 순위 p 는 무작위 방향 100개의 변동만 반영한다. 6칸은 같은 평가 에피소드, 같은 readout(인코더 토큰으로 1회 학습)을 공유하므로 독립 반복이 아니다.
5. **타깃/창/마진 선택(분석자 자유도)**: 주 타깃, late 창, 포화 기준, (a) 에피소드 수, 단조 기준이 결과 확인 전에 사전 등록 후 변경되었다고 문서화되어 있다(확정 기록 6, 7). "결과 미확인" 상태에서의 변경이라는 기록을 그대로 받아들이되 이 머신에서 독립 검증은 못 했다. 001 결과를 본 상태에서 설계된 점은 문서에 명시되어 있다.
6. **프레임 구간 가정**: early_flat/late_deformed 는 인덱스 가정이며 영상으로 확인하지 않았다(config 주석). H-a 의 주 판정이 late 창에 걸려 있으므로 이 가정이 틀리면 H-a 해석이 바뀐다.
7. **측정 구성의 구조적 한계**: output-space probe 의 분포 이동(3.2절), (a) 의 층별 probe 가 미지 크기에서 학습됨, 학습 영상 시뮬레이션 1종(외부 타당성).

## 7. 수치 이상 여부

- 이상 없음(재현됨): Holm, status, 시드 평균(`drop_mean_over_seeds`), per-probe-seed 평균, 점추정이 CI 안, 학습 step 동일, 총 14행.
- 관찰된 특이 사항(오류는 아님):
  - linear probe 의 probe 시드별 값이 모두 동일(결정론). H-a 보간/외삽 linear 의 "5 시드"는 정보가 없다.
  - 보간 mask baseline R² 가 음수(-0.042)이고 early 창에서 0.772. 변동이 크다. mask baseline 의 R² 계산에 쓰인 특징은 면적/bbox 이고 학습 레벨 3개로 적합한 ridge 이다.
  - 보간 attentive: real < random-init 이 early 창에서 관찰(0.312 vs 0.586). attentive probe 의 학습 상태 점검이 필요하다(코드 미수정).
  - Q3 steering Spearman 이 6칸 모두 정확히 1.000, removal 6칸은 0.9999999999999999. 7개 dose 에서 선형에 가까운 응답이면 자연스러운 값이지만, 8절 H-c 항목대로 이 기준은 변별력이 없다.
  - 4c: (b) 가 large 로 표기된 것은 규칙 문구와 맞지 않는다(5절).
  - q1 의 `r2_within` 이 일부 셀에서 null 이다(예: 보간 attentive). 의도된 결측인지 NaN 인지는 이 머신에서 확인하지 못했다.
  - results.json 의 environment 는 aggregate 재실행 런타임 값(python 3.9.25, torch 2.0.1+cu117, CUDA 불가)이며 학습 환경이 아니다(log.md 에도 기록).

## 8. 가설별 평가

**H-a (인코더가 mask baseline 과 random-init 보다 크기를 더 잘 복원)**: 부분 지지. 보간 linear 는 강하게 지지(마진 0.1 을 CI 하한 0.352 로도 넘음, 사전학습 기여 0.340). 보간 attentive 는 사전 등록 기준은 충족하나 마진을 CI 로 넘지 못하고 MAE 로는 baseline 과 비슷하며 probe 시드 변동이 커서 약하다. 외삽은 유의한 작은 효과(linear, 0.034)와 baseline 미달(attentive)로 지지되지 않는다. 역할 간 대비는 타깃 분산에 의존하는 R² 의 성질을 일부 반영한다. 학습 시드가 없는 동결 인코더의 성질이다. 결론 강도: 보간 linear 만 강함.

**H-b (정보가 predictor 마지막 블록과 rollout 끝까지 보존)**: 분리해서 평가한다. last_block 보존은 지지되나 효과가 작고(drop -0.0085) 층별 변화가 시드 노이즈 수준이라 "블록을 통과하며 정보가 거의 변하지 않는다"까지만 말할 수 있다. rollout 끝(horizon 12)은 3시드 모두 임계 0.2 를 넘는 하락(0.29~0.32)으로 지지되지 않는다. 하락은 step 에 따라 점진적이다. 한 가설 안에서 결과가 갈리므로 하나로 합치지 않는다.

**H-c (크기 방향 개입 효과가 무작위 방향보다 크고 단조)**: steering 은 사전 등록 기준을 충족(6칸 모두 순위 p 하한, T/무작위 95백분위 = 5.8~7.9배, Spearman 1.0). 단 (i) 효과 측정에 쓰인 readout(인코더 토큰 ridge)과 개입 방향(ridge 방향)이 같은 크기 회귀에서 나왔고 무작위 방향은 그 readout 과 정렬되지 않으므로 "큰 효과"가 사실상 설계에서 예상되는 값에 가까울 수 있다(추정), (ii) 단조 기준(|rho|>=0.9, 7점)은 선형 응답이면 항상 통과해 변별력이 없고, (iii) 이는 크기 정보가 개입 가능한 방향으로 표현된다는 뜻이지 predictor 가 예측에 그것을 "사용한다"는 행동 수준 증거가 아니다(행동 대조 탐색 항목은 p 없음, 이 보고에서 해석하지 않음). subspace_removal 은 6칸(INLP)과 exploratory LEACE 12칸 모두 not_removed 라 결론 없음이며, 제거 효과값(T=0.38~0.70)은 근거로 쓰면 안 된다(log.md 의 사용자 결정과 동일).

**조건 (a) 대 (b)**: rollout drop 차이는 견고(0.42, 시드 분리). 층별 변화 차이는 시드 단위로 유의하지 않음. (a) 층별 결과는 게이트가 임계 근처의 medium 이라는 점에서도 주의.

## 9. 우연/시드 운으로 설명될 가능성

- rollout drop, (a) 대 (b) rollout: 낮음. 3시드 모두 동방향이고 효과가 시드 SD 의 약 27배(0.422/0.0158)이다.
- H-b last_block pass: 우연 가능성 낮음(임계 대비 여유 큼). 단 "drop 이 음수"라는 세부는 시드 노이즈 수준.
- cond_a_vs_b:last_block pass: 시드 운으로 설명 가능성 높음(시드 단위 p=0.155).
- H-a 보간 linear: 평가 에피소드 표본 변동으로는 설명되지 않음(CI 하한 0.352). 이 평가 세트의 레벨 구성/창 정의에 의존.
- H-a 보간 attentive: probe 시드 운 가능성 중간. 5개 시드 모두 Δ>=0.1 이라는 점은 안정성을 일부 지지.
- H-c steering: 100개 무작위 방향에 대해 6칸 모두 최대 순위라 방향 무작위성만으로는 설명 불가. 하지만 설계 순환성(8절)은 별개 문제.

## 10. 다음 실험 제안 (사전 등록 임계값 변경 없음)

1. 학습 시드를 늘린다(최소 5개, 가능하면 10개). 현재 시드 단위 추론(df=2)은 미세 효과를 판단할 수 없다. 이득이 가장 큰 개선이다. GPU 비용이 크므로 사람 승인이 필요하다.
2. 시드를 분석 단위로 쓰는 2차 보고를 추가한다: 에피소드 bootstrap 과 함께, 학습 시드 단위 평균과 t/순열 CI 를 판정 표에 병기. 그리고 (a)/(b) 비교는 에피소드와 시드를 함께 재표집(2단계 bootstrap)하는 방식을 사전 등록한다. 코드 변경은 experimenter 몫.
3. H-a 는 사전 등록 지표(R², MAE, Pearson)의 MAE 를 로그 단위 절대 오차로 같이 보고하고, 역할별 SST 를 함께 표시한다. 마진 0.1 은 그대로 둔다. 레벨별 결과(Drive 의 cells.npz 로 생성 가능)를 탐색 표로 만든다.
4. H-b 의 rollout 은 horizon 별 drop 곡선(탐색)을 천장(각 step 의 Q1 상한)과 함께 제시하고, 분포 이동 vs 정보 손실을 분리하는 대조(예: 출력 공간에서 probe 를 재학습하는 보조 분석)를 별도 탐색으로 둔다. 이 경우 사전 등록 판정과 섞지 않는다.
5. H-c 는 readout 과 독립인 효과 척도(예: 디코더 출력이나 행동 수준 지표)를 추가하고, 제거 실험은 INLP 반복 수나 비선형 제거를 별도 탐색으로 한다. not_removed 임계(0.1)는 건드리지 않는다.
6. 4c 진단은 "학습 본 크기 대비 낯섦"을 변별하는 기준(예: (b) 의 본 크기와 (a) 의 비교)을 점검한 뒤 쓴다. 지금 형태는 입력 차이를 주로 재는 것으로 보인다(추정).
7. 학습 중단/재개와 val no_grad 패치의 영향은 같은 시드를 재개 없이 한 번 더 돌려 비교하면 분리할 수 있다(추정이 아닌 검증).

## 11. 확신도: 중간

- 높은 부분: 판정 표/Holm 의 기계적 정확성, rollout 관련 결론(H-b rollout fail, (a) 대 (b) rollout 차이)의 방향과 크기, H-c removal 의 not_evaluable.
- 중간 이하인 부분: H-a 전반(지표 의존성, 창 정의 가정, attentive 불안정), H-b last_block 의 해석, H-c steering 의 의미(순환성은 코드 읽기에 근거한 추정), cond_a_vs_b:last_block(유의하지 않음에 가까움).
- 이유: 학습 시드 3개, bootstrap/원자료를 이 머신에서 재생성하지 못함(Drive), CI 가 학습 시드와 probe 학습 표본 변동을 담지 않음, p 가 하한에 포화.
