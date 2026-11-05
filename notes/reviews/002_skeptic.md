# 002_single_size_sweep 회의론자 리뷰 (analyst-skeptic)

- 작성일: 2026-11-05
- 독립성: `notes/reviews/` 의 다른 002 파일은 읽지 않았다.
- 표기: **[확인]** = 코드나 JSON 을 읽어 직접 확인. **[추정]** = 읽고 의심하는 것. 코드는 실행하지 않았다. "(내 계산)" = results.json 값으로 손계산한 값.
- 수치 경로 줄임말
  - `DT[id]` = `results.json -> decision_table` 의 해당 id 행
  - `Q1C[key]` = `results.json -> q1.cells[key]`, 접두 `log_nominal_area_ratio|` 생략
  - `GRID[run]` = `results.json -> exploratory.q2_layer_time_grid_r2_point_estimates[run]`, 접두 `log_nominal_area_ratio|` 생략
  - `EXQ3` = `results.json -> exploratory.q3_exploratory_methods.values`

## 0. 핵심 요약

판정 표의 숫자는 규칙대로 계산되었고, 집계 코드에서 판정을 뒤집는 버그는 찾지 못했다. 문제는 pass 3종이 가설이 묻는 것을 시험하지 못한다는 점이다.

1. **H-c steering pass 는 "predictor 가 크기를 쓴다"의 근거가 아니다.** 6칸 중 `last` 3칸은 개입 뒤에 predictor 블록이 하나도 없다. 측정하는 것은 크기 방향과 크기 판독기의 정렬이다.
2. **H-b last_block pass 는 변별력이 없다.** rollout 에서 크기를 거의 잃는 조건 (a) 도 같은 값을 낸다. 잔차 구조상 거의 자동으로 통과한다.
3. **H-a 보간 pass / 외삽 fail 의 갈림은 상당 부분 R² 분모 차이다.** 절대 MSE 이득은 보간과 외삽이 비슷하다(내 계산). 마스크 baseline 도 프레임 정보 없이 전 프레임으로 학습되어 후반 프레임에서 불리하다.
4. **가장 믿을 만한 관측은 조건 (a) 대 (b) 의 rollout 차이다.** 시드 범위가 겹치지 않는다. 다만 "정보 소실"인지 "판독기 불일치"인지는 구별되지 않고, 인과 해석에는 교란이 남아 있다.
5. **`cond_a_vs_b:last_block` pass 는 인용하면 안 된다.** 효과가 R² 0.008 이고 시드 범위가 겹치며 seed 0 에서는 방향이 반대다.

---

## 1. 치명 (해당 주장을 무효화)

### C1. H-c steering: 시험 구조가 "사용"을 묻지 못한다 [구조는 확인, 크기 해석은 추정]

**무효화 범위:** `H_c:steering` 의 status 는 규칙대로 pass 이지만 "predictor 가 크기 정보를 예측에 쓴다"는 결론은 무효다.

- `last` 층에는 하류 연산이 없다.
  - `q3.py:29-30` 에서 `last = L` 은 마지막 블록 출력, 최종 norm 의 입력이다.
  - `model_io.py:64, 84-88` 에서 j=L 의 hook 대상은 `tr.norm` 이다.
  - `thread0/models/vit.py:93-98` 은 `x = attn(x)+x; x = ff(x)+x; return self.norm(x)` 이고, `ViTPredictor.forward` 도 transformer 출력을 그대로 돌려준다(`vit.py:115-120`).
  - 따라서 `seed*/last` 3칸은 "방향 d 를 더하고 LayerNorm 을 지나 크기 판독기로 읽은 값"이다. 학습된 블록이 관여하지 않는다.
  - 주의: `thread0` 는 `main` 체크아웃이다(`thread0/.git/HEAD`). 고정 commit 5a640bed 의 블록 구조는 `model_io.py:56` 의 서술로만 확인했다.
- `middle` 칸도 분리되지 않는다. `StreamHooks` 는 잔차 경로에 이동량을 그대로 더한다(`model_io.py:56-58`). 출력의 변화에는 "블록이 반응한 양"과 "잔차로 그냥 통과한 양"이 섞여 있고, 저장값으로 나눌 수 없다.
- 방향과 판독기가 같은 라벨에서 나온다.
  - d 는 학습 episode 의 크기 라벨에 맞춘 ridge 방향이다(`q3.py:130`).
  - 판독기는 같은 episode, 같은 라벨로 맞춘 ridge 다(`q3.py:329-332`).
  - 둘이 정렬되는 것은 예상되는 결과이고, 무작위 방향은 이 정렬이 없다. 따라서 p_rank 가 바닥값으로 나오는 것은 사용 여부와 무관하게 예상된다 [추정].
- 수치가 통과 효과를 시사한다.
  - `EXQ3` 의 meandiff steering T_size: (b) `last` 0.397 / 0.402 / 0.386, `middle` 0.185 / 0.171 / 0.169. 블록이 없는 `last` 가 2배 이상 크다.
  - `last` 값은 학습 라벨 3수준의 표준편차 0.401(내 계산)과 거의 같다. 1σ 이동이 판독기에 라벨 1 SD 만큼 그대로 나타난다는 통과 가설의 예측과 맞는다 [추정].
  - 반대 근거: 주 방향(ridge)에서는 `middle`(0.081~0.087)이 `last`(0.054~0.065)보다 크다(`DT[H_c:steering].result.cells`). 블록 기여가 있을 수 있다. 다만 층마다 d 가 달라 직접 비교할 수 없다.
- 단조성은 변별력이 없다. 6칸 모두 Spearman 1.0 이다(같은 경로). 거의 선형인 통과 경로는 자동으로 단조다 [추정].
- 부정 대조(selftest)가 실제 모델에 대응하지 않는다. "ignores" 장난감은 시각 출력을 `* 0.0` 으로 강제로 지운다(`selftest.py:446-447`). 잔차 구조를 유지한 채 크기를 안 쓰는 모델이 fail 하는지는 시험한 적이 없다. 실제 구조의 항등 predictor 는 이 기준을 통과할 것으로 본다 [추정].
- 가설 문장과 지표가 다르다. 후보 G 의 문장은 "다음 latent 가 랜덤 방향보다 더 변한다"이다(`hypotheses.md:157`). 주 지표는 크기 판독 변화다(`config.yaml:185`). 구현 파라미터이므로 위반은 아니지만, latent 변화량 `T_latent` 는 results.json 에 없다(grep 0건).
- 용량 문제: 주 검정의 T_size 0.054~0.087 은 a060 과 a160 의 로그 면적 차 0.98(내 계산)의 약 6~9% 다. 실제 크기 차이에 해당하는 개입이 아니다.

### C2. H-b last_block: 판정에 변별력이 없다 [확인]

**무효화 범위:** `H_b:last_block` pass 를 "크기 정보가 predictor 에 보존된다"의 근거로 쓸 수 없다.

- 조건 (a) 가 같은 값을 낸다.

| 지표 | (a) s0 / s1 / s2 | (b) s0 / s1 / s2 |
|---|---|---|
| 마지막 블록 R² | 0.879 / 0.867 / 0.871 | 0.875 / 0.886 / 0.880 |
| rollout step 12 R² | 0.082 / 0.114 / 0.091 | 0.529 / 0.525 / 0.500 |

  - 경로: `DT[cond_a_vs_b:last_block].result.per_training_seed_{a,b}`, `DT[cond_a_vs_b:last_rollout_step].result.per_training_seed_{a,b}`.
  - 상한 0.872(`DT[H_b:last_block]`) 대비 (a) 의 하락은 -0.007 / +0.005 / +0.001 이다(내 계산). rollout 에서 크기를 거의 잃는 모델도 이 검정을 통과한다는 뜻이다.
- 원인은 설계에 있다.
  - 입력이 teacher-forced GT 이고, 스트림은 `x_L = x_0 + Σf` 형태다. 블록이 입력 정보를 정확히 상쇄하지 않는 한 선형 정보는 남는다.
  - probe 를 그 활성으로 다시 학습한다(`q2.py:177`). layer 0 값 0.8727 과 차이가 거의 없다.
- "상한"이 상한이 아니다. 하락이 -0.0085 로 음수다(`DT[H_b:last_block].result.drop_mean_over_seeds`).
- 두 H_b 점은 프로토콜이 다르다. last_block 은 활성으로 재학습한 probe 와 union 프레임(`aggregate.py:161`)을 쓰고, last_rollout_step 은 재학습 없이 전이한 probe 와 프레임 12 를 쓴다(`aggregate.py:162-163`). 한 패밀리에서 pass 와 fail 이 갈린 것은 층과 시점의 차이만이 아니다.

### C3. H-a 보간 pass: "마스크 이상의 정보"를 보여 주지 못한다 [수치는 확인, 원인은 추정]

**무효화 범위:** status 는 사전 등록대로 유지한다. "DINOv2 토큰이 마스크 면적보다 크기를 더 잘 담는다"는 주장은 지지되지 않는다.

- 보간과 외삽의 갈림은 R² 분모 차이로 대부분 설명된다.
  - 라벨 분산(내 계산, 수준별 episode 수가 같다고 보고): 보간 0.049, 외삽 0.966, 약 20배 차이.
  - R² 로부터 되돌린 MSE(내 계산):

| 역할 | 마스크 | DINO linear | MSE 이득 | R² 차 (`DT`) |
|---|---|---|---|---|
| 보간 | 0.052 | 0.024 | 0.028 | 0.556 |
| 외삽 | 0.169 | 0.136 | 0.033 | 0.034 |

  - 절대 이득은 비슷하다. 보간 pass / 외삽 fail 이 "보간에서만 DINO 가 낫다"는 뜻이 아닐 가능성이 크다.
- R² 외 지표에서 격차는 훨씬 작다. 경로는 `Q1C[linear_pooled|dino|all|late].interp`, `Q1C[attentive|dino|all|late].interp`, `Q1C[mask_ridge|mask|all|late].interp`.

| 보간 후반 프레임 | R² | MAE | Pearson | 순서 정확도 |
|---|---|---|---|---|
| 마스크 | -0.042 | 0.167 | 0.606 | 0.897 |
| DINO linear | 0.514 | 0.118 | 0.762 | 0.966 |
| DINO attentive | 0.182 | **0.178** | 0.707 | 0.972 |

  - 마스크는 R² 가 음수인데 순서는 90% 맞춘다. 정보가 없는 것이 아니라 편향이거나 이상치 문제다 [추정].
  - `H_a:interp:attentive` 는 pass 인데 MAE 는 마스크보다 나쁘다.
- 마스크 baseline 이 시간 정보 없이 전 프레임으로 학습된다.
  - 평평한 프레임에서는 마스크로 크기를 거의 그대로 읽는다. `Q1C[mask_ridge|mask|early|early]`: 외삽 R² 0.987, 보간 R² 0.835, 보간 순서 정확도 1.0.
  - 같은 특징을 프레임 0-11 로 학습하면 후반 보간 R² 가 -0.042 로 떨어진다. 면적은 시간에 따라 변하는데(001 리뷰 I1) 마스크 특징 5개에는 시점 정보가 없다(`probes.py:163-168`).
  - DINO 토큰은 그리퍼나 천 상태에서 시점을 간접적으로 알 수 있다 [추정]. 후반 프레임만으로 학습한 마스크 셀은 없어서 공정한 비교를 할 수 없다.
- 판정은 창에 전적으로 달려 있다. 주 창 고정은 결과 전에 승인되었으므로 위반은 아니지만, 초기 창에서는 네 행 모두 마진 0.1 에 못 미친다.
  - 보간 linear +0.097, 보간 attentive -0.461, 외삽 linear -0.003, 외삽 attentive -0.191 (`h_a_early_flat_report.rows[*].delta_vs_mask_baseline`).
- attentive probe 는 판단 근거로 약하다.
  - 초기 창 보간에서 random-init(0.586)이 실제 DINO(0.312)보다 높다(`h_a_early_flat_report`).
  - 학습 크기에서는 R² 가 거의 1(encoder step 1 trained 0.997, `GRID[multi_60_ind_160/seed0]`)인데 미지 크기에서는 0.70 대다. 학습 3수준을 분류하듯 학습한 것으로 보인다 [추정].
  - probe seed 범위 0.108~0.271 이 CI 에 포함되지 않는다. 그런데 CI 하한은 0.015 로 간신히 0 위다(`DT[H_a:interp:attentive]`).
- 일반화 단위가 작다. 보간은 2수준, 외삽은 4수준뿐이다. episode bootstrap 은 수준 내 episode 만 다시 뽑으므로 크기 수준에 대한 일반화 불확실성은 전혀 반영하지 않는다.
- 001 의 지적이 여전히 남아 있다. random-init 도 외삽 후반 R² 0.789 다(`DT[H_a:extrap:linear_pooled]`). 또한 명목 면적 라벨은 episode 상수라서 수준과 상관된 어떤 단서(그리퍼 위치 등)라도 "크기"로 읽힌다 [추정, 미검증].

---

## 2. 중요 (결론의 범위를 크게 좁힘)

### I1. H-b last_rollout_step fail: 하락은 믿을 수 있으나 "정보 소실"은 말할 수 없다 [확인 + 추정]

- 하락 자체는 안정적이다. 0.291 / 0.295 / 0.321, 모든 시드에서 임계 0.2 초과(`DT[H_b:last_rollout_step]`).
- 원인은 구별되지 않는다. 인코더 토큰으로 학습한 probe 를 예측 z 에 전이하므로 R² 하락에는 정보 소실과 분포 이동(편향, 기울기 축소)이 섞인다. 예측 z 로 재학습한 셀은 삭제되었다(`README.md:206`). 사전 등록상 주 측정은 맞지만, 구별 수단이 사라졌다.
- 미지 크기에 특이적이지 않다. step 12 에서 학습 크기 R² 의 하락이 미지 크기보다 크다.
  - 인코더 0.796 대비 예측 (b): 0.540 / 0.054 / 0.183, 하락 0.256 / 0.741 / 0.613 (내 계산; `GRID[multi_60_ind_160/seed{0,1,2}]["linear_pooled|OUT|{encoder,rollout}|step12"].trained`).
  - 일반적인 rollout 오차 누적만으로 설명될 수 있다 [추정]. 다만 학습 크기는 라벨 범위가 좁아 R² 가 편향에 더 민감하다는 점도 함께 봐야 한다.
- 시드 변동이 크게 숨어 있다. 학습 크기 R² 가 시드에 따라 0.05~0.54 로 움직이는 동안 미지 크기는 0.50~0.53 이다. 미지 크기 R² 는 극단 수준이 분산을 지배해 거친 순서만 맞아도 높게 나온다 [추정]. 사전 등록 지표 하나로는 이 불안정을 볼 수 없다.
- 판정은 probe 에 의존한다. attentive 로 같은 하락을 계산하면 0.147 / 0.231 / 0.183, 평균 0.187 로 0.2 아래다(내 계산; `GRID[multi_60_ind_160/seed*]["attentive|OUT|..."].unseen`). 주 probe 는 결과 전에 linear 로 고정되었으므로 **판정을 바꾸자는 뜻이 아니다.** 탐색 사실로 같이 보고해야 한다는 뜻이다. attentive 는 상한도 0.685 로 낮다.
- 상한 프레임 12 는 probe 학습 범위(프레임 0-11, `q2.py:105`, `config.yaml:128`) 밖이다. 효과는 작을 것이다 [추정].

### I2. CI 와 p 값이 시드 변동을 포함하지 않아 작은 효과가 "유의"로 나온다 [확인]

- `cond_a_vs_b:last_block`: 차이 -0.0082, CI [-0.0103, -0.0061], p 0.002, pass.
  - 시드별 변화량 (a) +0.0062 / -0.0059 / -0.0018, (b) +0.0026 / +0.0129 / +0.0075. 범위가 겹치고 seed 0 은 방향이 반대다(`DT[cond_a_vs_b:last_block].result`).
  - 이 pass 는 episode 표본 크기가 만든 것이다. 조건 차이의 근거가 아니다.
- `H_b:last_block` 의 CI 폭은 0.0044 인데 시드 범위는 0.0103 이다(내 계산).
- 시드 단위 검정은 이 설계로 불가능하다. 3 대 3 정확 순열 검정의 최소 양측 p 는 2/20 = 0.10 이다(내 계산). `ci_caveat` 에 적혀 있지만 status 에는 반영되지 않는다.
- random-init 통제는 초기화 1개다(`config.yaml:153`). `delta_vs_random_init_ci95` 에 초기화 변동이 없다.

### I3. `cond_a_vs_b:last_rollout_step`: 효과는 크지만 인과 해석에 교란이 남는다 [확인 + 추정]

- 효과: (a) 하락 0.739 / 0.706 / 0.729, (b) 0.291 / 0.295 / 0.321. 범위가 겹치지 않고 최소 간격이 0.386 이다(내 계산). 위 I2 의 문제가 있어도 이 관측은 유지된다.
- 초기부터 벌어진다. step 1 미지 크기 R²: (a) 0.702 / 0.643 / 0.713, (b) s0 0.853 (`GRID`). 누적 오차만의 현상이 아니다.
- 남은 교란
  - 재개: (b) 는 3개 중 2개가 재개 run 이다(log.md 2026-11-05). (a) 의 재개 여부는 results.json 에 기록이 없어 확인하지 못했다.
  - 행동 정규화: 조건별 학습 manifest 통계를 쓴다(`data.py:133-137`, `model_io.py:156-163`). 크기별 행동 분포가 다르면 (a) 에서는 행동도 분포 밖이다. 001 의 대안 A2 가 해결되지 않았다.
  - 범위와 다양성: (b) 는 ind 를 가운데 두고 양쪽을 포함한다. "크기 다양성"과 "범위 포함"이 같이 움직인다. 001 의 "가운데로 압축" 대안도 이 대조만으로는 끊기지 않는다. 003 이 이 질문을 맡는 것으로 보인다.
  - probe 정합: 판독기가 (b) 의 학습 수준으로 학습되었다. (b) 의 예측 분포에 유리할 수 있다 [추정].

### I4. 4c 진단: 조건 (a) 의 "medium" 과 (b) 의 "large" 를 "낯섦"으로 읽을 수 없다 [확인 + 추정]

- 구성 타당도가 의심된다. (b) 는 a060 을 학습했는데도 (a) 보다 큰 값을 낸다. a060 최대 비율 (b) 5.04 / 5.05 / 5.30, (a) 4.91 / 4.66 / 4.66 (`DT[diag_4c:*].result.per_training_seed.*.levels.a060.max_ratio_over_layers`).
- predictor 블록 이전에 이미 큰 값이다. layer 0 비율이 (a) s0 3.82, (b) s0 3.82 로 거의 같다. 활성 이동량은 입력의 크기 차이, 즉 재려는 크기 신호 자체를 포함한다.
- 경계에 붙어 있다. (a) s0 4.91 은 임계 5 의 약 2% 아래다. "large" 였다면 `cond_a_vs_b:last_block` 은 not_evaluable 이다(`aggregate.py:324`).
- noise floor 표본 크기가 비교 표본과 다르다. 50 대 50 으로 재는 floor(`diag.py:47-51`)와 100 대 100 비교(`q2.py:124`)의 표집 잡음 크기가 약 √2 배 다르다. 맞추면 (a) s0 은 약 6.9 가 된다(내 계산, 대략값) [추정]. 구현은 사전 등록 문구("두 묶음")를 따랐으므로 **규칙 위반이 아니고 임계값을 바꾸자는 뜻도 아니다.** 판정의 민감도를 보고해야 한다는 뜻이다.
- 다음 프레임 오차 비율은 천 면적과 함께 변할 수 있다. (b) 에서도 a060 0.90, a160 1.12 다(같은 경로). 크기 효과를 빼지 않아 "2배" 기준의 의미가 흐리다 [추정].
- log.md 가 적은 "(4) large 규칙 문구는 (a) 층별 결과에 관한 것인데 large 는 (b) 행에서 나왔다"는 사실이다. `RULE_4C` 텍스트(`aggregate.py:22-24`)가 (b) 행에도 그대로 붙는다. (b) 층별 결과를 해석 금지로 읽으면 안 된다.

### I5. Q3 not_removed: log.md 의 원인 추정은 확인되지 않았고, 다른 후보가 더 있다 [코드는 확인, 원인은 추정]

- 사실: INLP 는 (b) 6칸 모두 not_removed(`DT[H_c:subspace_removal].result.not_removed_cells`), LEACE 는 (a)(b) 12칸 모두 `removed: false` 다(`EXQ3`).
- 판정 근거 수치는 results.json 에 없다. `mlp_r2_after_removal`, `inlp_linear_r2_by_iteration`, `linear_ridge_r2_after` 모두 grep 0건이다. log.md 의 "INLP 10회 부족, LEACE 는 선형만 제거"는 지금 검증할 수 없다.
- 코드를 읽어 찾은 다른 후보
  1. **LEACE 대상이 스칼라다.** `leace_fit(Xtr, ytr)`(`q3.py:175`, `directions.py:166-170`)은 y 와의 공분산 방향 1개만 지운다. 3수준 평균이 일직선이 아니면 나머지 수준 간 차이가 남고 MLP 가 읽을 수 있다. 수준별 분산 차이도 남는다.
  2. **확인 평가셋이 9수준 전체다**(`q3.py:183, 211`). 3수준으로 맞춘 선형 제거가 미지 크기에서도 정보를 없앤다는 보장은 없다. 학습 수준에 한정해 확인한 값이 없어 어느 쪽 실패인지 나눌 수 없다.
  3. **INLP 는 ridge 방향 10개다**(`config.yaml:187`). 404차원에 크기가 넓게 퍼져 있으면 부족하다. 이 부분은 log.md 의 추정과 같은 방향이다.
  4. **확인은 풀링 특징으로 하고 개입은 토큰 단위다**(`q3.py:122-123, 183`, `scan`). 크기가 "어느 패치가 천인가"로 공간에 표상되면(001 리뷰 I6) 풀링 기준 제거가 성공해도 토큰의 크기 정보는 남는다. 즉 removed=True 가 나와도 해석이 어렵다.
- LEACE 식은 내 기억 속 닫힌형과 일치한다. INLP 구현에서도 코드 오류는 찾지 못했다. 논문과 직접 대조하지는 않았다.
- 결정 유지: `H_c:subspace_removal` 은 not_evaluable 이고, 표에 나온 T 0.38~0.70 은 H-c 근거로 쓰지 않는다(log.md 사용자 결정과 동일).

### I6. 무작위 대조의 품질을 검증할 수 없고, 판정에 연결되어 있지 않다 [코드는 확인, 영향은 추정]

- `aggregate.h_c` 와 `_rank_test` 는 `random_variance_match_ok`, `identity_check_max_abs_dsize_zero_shift` 를 보지 않는다(`aggregate.py:204-215, 238-268`). 분산 맞춤이 실패해도, 항등성이 깨져도 pass 가 나올 수 있다. 이 값들은 results.json 에도 없다(grep 0건).
- 목표 분산이 작으면 대조 방향이 서로 닮을 수 있다. `matched_random_direction` 은 임의 방향을 같은 최소 고유방향 쪽으로 기울인다(`directions.py:96-122`). ridge 방향은 분산이 작을 가능성이 크고(`directions.py:54` 의 `w/sd`), 그러면 100개가 최소 고유방향 성분을 공유해 사실상 독립이 아니고 분포도 좁아진다 [추정]. README:185 도 코사인이 미저장이라고 적는다.
- p_rank 는 6칸 모두 하한 0.0099 다. 대조 분포가 얼마나 떨어져 있는지 보여 주지 못한다.

### I7. 결과 파일만으로는 필요한 검증을 할 수 없다 [확인]

- CLAUDE.md 는 수치를 results.json 에서만 가져오라고 하는데, 다음은 Drive 에만 있다: q3.json 세부(`T_latent`, sigma, (a) 의 ridge steering, behavior contrast), `read_by_level`(`q2.py:194`), 수준별 R² (`exploratory.per_level_note`), layer 시드별 메타(loaded epoch 포함), 학습 곡선.
- 재개 여부, 재개 epoch, 패치 sha 도 results.json 에 없다. `train_budgets` 는 재개 run 과 아닌 run 이 같은 형태다.

---

## 3. 사소

- S1. **`val()` 의 `@torch.no_grad()` 제거** [코드 읽기, 추정 유지]. `main` 의 train.py 기준 `.backward` 는 `train()` 안(`thread0/train.py:483`)에만 있고 val 은 eval 모드다. 읽은 범위에서 가중치에 영향을 줄 경로를 찾지 못했다. 고정 commit 과 패치 후 파일은 이 머신에 없으므로 "영향 없음"은 여전히 추정이다.
- S2. **재개와 데이터 순서** [추정]. log.md 는 순서가 복원되지 않는다고 적는다. `main` 에서는 `shuffle=False`(`train.py:130`)이고 슬라이스는 시드 설정 뒤(`train.py:114`) 한 번만 섞인다(`traj_dset.py:75`). 그렇다면 재개해도 슬라이스 순서는 같을 수 있다. 실제 로더 `LazyTrajSlicerDataset`(`data.py:162`)은 `main` 에 없어 읽지 못했다. dropout 등 난수 상태 차이는 남는다.
- S3. **재개 효과는 데이터로 판단할 수 없다.** 비재개 시드가 하나뿐이다. 재개된 s0 과 s1 이 학습 크기 step 12 에서 양 끝이고(0.54, 0.05) 비재개 s2 가 가운데(0.18)라, 일관된 재개 효과는 보이지 않는다. 그렇다고 효과 없음을 뜻하지도 않는다.
- S4. `r2_within` 이 주 타깃에서 무의미하다. 라벨이 수준 내 상수라 정의되지 않는데 CI 에 부동소수점 잡음 값이 찍힌다. 예: `Q1C[linear_pooled|dino|all|late].extrap.r2_within.ci95_episode_bootstrap` = [-4.0e13, -2.4e13]. 판정에는 쓰이지 않는다.
- S5. linear probe 의 "시드 5개"는 값이 전부 같다(`DT[H_a:interp:linear_pooled].result.r2_real_per_probe_seed`). 반복으로 읽으면 안 된다.
- S6. 단계 순서 규칙: "Q1 음성이면 Q2 로 가지 않는다"(`hypotheses.md:159`). 외삽 2행이 fail 인데 무인 실행으로 모두 돌았다. 결과는 있으나 순서 규칙 적용 여부가 기록되지 않았다.
- S7. `attempts` 는 전부 0 이다. 성공하면 0 으로 되돌리므로(`common.py:208`) log.md 의 실패 이력이 results.json 에 남지 않는다.
- S8. aggregate 재실행 대조는 "판정 행 수가 같다"까지만 기록되어 있다(log.md). 값 일치는 기록되지 않았다.
- S9. 문서가 낡았다. `README.md:3`("서버 미실행, 결과 수치 없음"), `README.md:5`("(a) 는 pending"), `config.yaml:2`("미확정"). README 에 실행 결과 절이 없다.
- S10. 시간 창 이름 `early_flat`, `late_deformed` 는 프레임 인덱스 가정이고 영상으로 확인하지 않았다(`README.md:62`). HSV 마스크도 002 에서 재검증하지 않았다(`README.md:187`). C3 의 baseline 평가에 영향을 줄 수 있다.
- S11. 표면상 문제가 없었던 항목: 라벨 셔플 R² 약 0(`Q1C[linear_pooled|shuffle|all|late].interp.r2.point` -0.027, `Q1C[attentive|shuffle|all|late].interp.r2.point` 0.003), probe 학습과 평가 episode 분리(`evalgrid.py`, `q1.py:30`), 행동 채널 혼입 영향 작음(layer 0 0.8727 대 Q1 0.8719), Holm 계산, 조건 간 step 수(tqdm 1857, `train_budgets`).

---

## 4. 가설별 판정을 믿을 수 있는가

| 행 | status | 규칙대로 계산됨 | 가설 주장의 근거가 되는가 | 이유 |
|---|---|---|---|---|
| H_a 보간 linear | pass | 예 | 아니오 | C3. R² 분모와 baseline 학습 방식. 초기 창 +0.097 |
| H_a 보간 attentive | pass | 예 | 아니오 | C3. MAE 가 마스크보다 나쁨, probe 불안정 |
| H_a 외삽 linear | fail | 예 | 제한적으로 예 | 마진 미달(0.034). 다만 R² 비교의 한계는 같음 |
| H_a 외삽 attentive | fail | 예 | 약함 | probe 일반화 실패가 섞임 |
| H_b last_block | pass | 예 | 아니오 | C2. (a) 도 통과, 변별력 없음 |
| H_b last_rollout_step | fail | 예 | 하락은 예, 소실은 미정 | I1 |
| H_c steering | pass | 예 | 아니오 | C1. `last` 칸에 블록 없음 |
| H_c subspace_removal | not_evaluable | 예 | 결론 없음 | I5 |
| cond last_block | pass | 예 | 아니오 | I2. 시드 겹침, 효과 0.008 |
| cond last_rollout_step | pass | 예 | 차이는 예, 원인은 미정 | I3 |
| diag_4c | medium / large | 예 | 약함 | I4 |

종합: 후보 G 의 흐름(인코딩 → 보존 → 사용) 중 이 실험으로 말할 수 있는 것은 다음뿐이다. "인코더 토큰으로 학습한 선형 판독기를 예측 latent 에 적용하면 rollout 이 길어질수록 R² 가 떨어지고, ind 단일 학습에서 훨씬 더 떨어진다." H-b 보존과 H-c 사용은 이 데이터로 확정되지 않았고 반증되지도 않았다. 후보 G 의 "Q1, Q2 양성, Q3 음성이면 정보는 있으나 쓰이지 않음" 같은 분기도 적용할 수 없다.

## 5. 과장 경고 (보고서와 논문에 쓰면 안 되는 문장)

- "predictor 가 크기 정보를 사용한다", "H-c 지지". → C1. 쓸 수 있는 범위: "ridge 크기 방향의 이동은 판독된 크기를 무작위 방향보다 많이 바꿨다. `last` 칸은 블록을 지나지 않으며, `middle` 칸은 잔차 통과와 구분되지 않는다."
- "크기 정보가 predictor 전체 층에서 보존된다". → C2. teacher-forced 입력과 재학습 probe 조건이고, (a) 도 같은 값이다.
- "DINOv2 가 마스크 면적 이상의 크기 정보를 담는다", "보간에서 R² 0.56 우위". → C3. MAE, 순서 정확도, 초기 창 결과를 함께 써야 한다.
- "외삽에서는 인코더가 크기를 못 읽는다". → 외삽 R² 는 0.859 다. fail 은 마진 미달이지 인코딩 실패가 아니다.
- "rollout 에서 크기 정보가 소실된다". → I1. 쓸 수 있는 범위는 "인코더용 판독기의 R² 가 하락한다"이다.
- "다중 학습이 크기 보존의 원인이다". → I3. 관측된 조건 차이까지만 쓸 수 있다.
- "조건 (a), (b) 의 층별 변화가 유의하게 다르다 (p=0.002)". → I2.
- `not_removed` 원인을 사실로 서술하는 문장. → I5.
- "H-c: 제거 효과도 무작위보다 큼". → 표에 수치가 있어도 쓰면 안 된다.

## 6. 재현성 점검

| 항목 | 상태 | 근거 |
|---|---|---|
| 시드 | 양호 | `config.yaml` 에 학습, probe, bootstrap, 방향 시드 명시 |
| 학습 런타임 환경 | **미기록** | `results.json.environment` 는 재실행 런타임 값(`cuda_available: false`). GPU 종류, 세션별 변화 없음 |
| 실험 코드 버전(단계별) | **미기록** | `run_info.txt` 는 브랜치 commit 만 기록(`trainrun.py:146`). q1, q2, q3 출력에도 저장소 commit 없음(`q2.py:97`, `q3.py:335`). `git` 은 aggregate 시점 d8206d6, dirty=true |
| 코드 변경 중 건너뛰기 | **위험** | `overnight.sh:65, 83-90` 은 산출물이 있으면 건너뛴다. 코드 버전을 대조하지 않는다. 세션 사이 코드 변경으로 단계별 버전이 섞였을 수 있다 [추정] |
| 재개 | **results.json 에 없음** | log.md 와 Drive `train.log` 에만 있음 |
| 패치 | 일부 | 패치 파일은 저장소에 있음. 적용 sha 는 Drive 에만 있음. 적용 뒤 train.py 는 이 머신에 없음 |
| 중간 산출물 | **Drive 에만 있음** | q1/q2/q3 셀, 체크포인트, cache. results.json 재생성을 여기서 확인할 수 없음 |
| 재시도 이력 | **소실** | 사후 `attempts` 0 (S7) |
| 환경 재현 | 부분 | `requirements_colab.txt` 고정, conda 규칙 예외는 기록됨. uv 재설치 현상 원인 미확인(log.md) |
| 잔차 경로 점검 | **결과에 없음** | `smoke: null`. `residual_path_probe` 와 항등 점검을 이 run 의 결과로 확인할 수 없음 |
| 인코더 일치 | 양호(코드) | 지문 대조 후 불일치 시 중단(`q2.py:50-52`, `q3.py:296-297`). 값은 results.json `q1.encoder_fingerprint` 에 있음 |

## 7. 다음 실험 전 필요한 수정

002 의 사전 등록값과 판정은 바꾸지 않는다. 아래는 이후 실험과 탐색 재분석에 적용한다. 새 판정 기준은 결과를 보기 전에 새 사전 등록으로 고정해야 한다.

1. **Q3 부정 대조 추가:** 잔차 구조를 유지한 모델(항등 predictor, 무작위 초기화 predictor)에서 기준이 fail 하는지 selftest 와 실제 코드로 확인한다. 통과하면 그 기준은 사용 검정이 아니다.
2. **통과 효과 분리:** 개입별로 직접 효과(블록을 다시 돌리지 않고 이동만 적용)와 전체 효과를 함께 저장한다. 블록이 없는 층은 "사용" 검정에서 제외한다.
3. **검정력 없는 판정 금지:** 어떤 판정이든 음성 사례로 알려진 조건에서 fail 하는지 먼저 본다. last_block 처럼 (a) 도 통과하는 검정은 확증 패밀리에 넣지 않는다.
4. **시드 단위 보고 강제:** status 옆에 시드 범위 겹침 여부를 표시하고, 시드 수를 설계 시 정한다. 3 대 3 은 시드 단위 유의성을 낼 수 없다.
5. **R² 단독 판정 금지:** 라벨 분산이 다른 역할을 비교할 때 MSE, MAE, 순서 정확도를 같은 줄에 둔다. 수준별 값을 results.json 에 넣는다.
6. **baseline 공정성:** 프레임 정보를 포함한 마스크 baseline, 평가 창과 같은 창으로 학습한 baseline 을 추가한다.
7. **전이 대 재학습 병행:** 예측 z 에서 재학습한 probe 셀을 탐색으로 복원해 소실과 분포 이동을 구분한다.
8. **results.json 완전성:** q3 세부, `read_by_level`, 학습 곡선, 재개 기록(run, epoch), 패치 sha, loaded epoch 를 포함한다. 002 도 Drive 의 값을 복사해 보강하기를 권한다(재계산 아님).
9. **단계별 코드 버전 기록:** train, cache, q1, q2, q3 출력에 저장소 commit, dirty, 런타임 환경을 남기고, `overnight.sh` 의 건너뛰기가 버전을 대조하게 한다.
10. **판정 게이트 연결:** `random_variance_match_ok`, 항등 점검, 대조 방향 간 코사인을 저장하고 실패하면 not_evaluable 처리한다.
11. **4c 재설계:** 크기 신호 자체와 낯섦을 분리하는 지표로 바꾸고, floor 와 비교의 표본 크기를 맞춘다. 대상 조건도 분명히 적는다.
12. **문서 갱신:** `README.md:3, 5`, `config.yaml:2` 를 실행 상태에 맞춘다.

## 8. 이 결과를 믿으려면 추가로 필요한 실험 (최대 3개)

1. **Q3 인과 분해 재실행 (기존 체크포인트, GPU, 사람 승인 필요).** 직접 효과와 간접 효과를 분리하고, (a) 모델과 항등, 무작위 predictor 를 대조로 둔다. 지표는 판독 변화와 `T_latent` 를 함께 본다. 평행이동 대신 천 패치 토큰을 다른 크기 episode 의 대응 토큰으로 바꾸는 공간 개입을 추가한다. 간접 효과가 무작위 대조와 대조 모델을 넘을 때만 "사용"을 말할 수 있다.
2. **소실과 판독기 불일치 구분, 판정 재분석 (Drive 산출물 + 소규모 GPU).** 시점별 예측 z 에 재학습한 probe 와 전이 probe 를 비교하고, `read_by_level` 로 수준별 편향과 기울기를 분해한다. Q1 은 후반 프레임 학습 마스크와 마스크+프레임 인덱스 baseline, MAE, 순서 정확도로 재보고한다(모두 탐색 표시, 002 status 불변).
3. **재개 없는 재학습과 시드 확대 (GPU 비용 큼, 사람 승인 필요).** (a), (b) 각 6개 이상, 세션 중단 없이 또는 재개 여부를 양 조건에 균형 있게 둔다. s0, s1 을 먼저 재개 없이 다시 돌려 학습 크기 step 12 R² 의 0.05~0.54 산포가 재개와 관련 있는지 본다. 행동 분포 통계도 크기별로 함께 비교한다(001 A2).

## 9. 확인하지 못한 항목

- 코드 실행 일체. 이 리뷰는 읽기만 했다.
- Drive 전용 파일: q3.json(INLP 수치, 분산 맞춤 오차, 항등 점검, `T_latent`, (a) 의 ridge steering), q2_summary.json(`read_by_level`, probe digest, loaded epoch), train.log, epoch_logs, run_info.txt, 패치 sha, Colab 로그. 이 저장소의 `logs/` 에는 check 와 selftest 로그만 있다.
- 고정 commit 5a640bed 의 코드. `thread0` 는 `main` 이고 `LazyTrajSlicerDataset` 이 없다. train.py 도 패치 전 루프를 가진 다른 판이다. 블록 구조, val 동작, 슬라이스 섞기는 `main` 기준으로만 읽었다.
- single_ind 의 재개 여부, 002 실행 동안의 코드 버전 혼재 여부.
- `config_sha256`(4d5261d6...)과 현재 `config.yaml` 의 일치, results.json 의 sha256(log.md 079ff4c1...). 해시 계산 도구를 쓸 수 없어 대조하지 못했다.
- 영상과 프레임 육안 점검, 마스크 정확도, 행동 분포의 크기 의존성.
- results.json 의 시드별 전체 Q1, Q2 셀은 판정에 쓰이는 키 위주로 읽었다. 전 셀을 읽지는 않았다. rollout 전 구간(step 1~12)은 (b) s0 에서만 전부 읽었고, 다른 run 은 step 1, 6, 12 위주다.

## 10. 확신도

- **높음:** C2((a)(b) 값 직접 대조), I2(시드 범위 대 CI), `last` 칸에 블록이 없다는 C1 의 구조 지적(`q3.py`, `model_io.py`, `vit.py` 가 서로 일치. 고정 commit 원문은 미확인이지만 `model_io.py:56` 서술과 같음), C3 의 수치 사실, I7.
- **중간:** C1 의 "무작위 대조로 구별되지 않는다" 전체 결론. 구조상 강하게 의심되고 meandiff 수치와 맞지만, 직접 효과와 간접 효과의 분해값이 없다. `middle` 칸에는 실제 블록 기여가 있을 수 있다. C3 의 원인(분모, baseline 시간 정보), I1 의 분해, I3 의 교란 크기, I4 의 √2 보정도 중간이다.
- **낮음:** I5 후보들 중 어느 것이 not_removed 의 실제 원인인지, I6 의 대조 방향 유사성, S2 의 재개 후 데이터 순서. 필요한 값이 이 머신에 없다.
- **전체 결론** ("계산은 규칙대로이나 pass 3종은 가설 주장을 지지하지 못하고, 신뢰할 수 있는 것은 rollout 판독 하락과 조건 차이의 존재까지"): **중간~높음**. 뒤집힐 가능성이 가장 큰 곳은 C1 이다. 분해 실험에서 `middle` 의 간접 효과가 대조 모델을 크게 넘으면 H-c 는 더 강한 형태로 되살아날 수 있다.
