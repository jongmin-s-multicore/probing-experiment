# 005_inductive_bias_probe 리뷰: 선행연구/이론 관점 (analyst-prior)

작성일: 2026-11-10. 다른 분석가의 리뷰 파일(004_*, 005_* 의 stats/skeptic)은 읽지 않았다. 내 관점의 003_prior.md 는 003 정합 비교에만 참조했다.

## 0. 읽은 것과 확인 수준

- 읽음: experiments/005_inductive_bias_probe/{results.json, config.yaml, CHANGES.md, results/REPORT.md, results/ib_scores.csv(702행 중 1~493행만; seed 0~3 의 일부. seed 4 행과 const/noninf 일부는 못 봄), results/law_fits.csv(전체), results/calibration_dev.csv(앞 40행)}, 004 results/summary.json(k_star, m_star, loss_ood_only 줄만), notes/specs/exp4_5_spec.md 4절, notes/hypotheses.md 후보 I 와 확정 기록 9·12, notes/log.md 2026-11-09 절.
- 읽지 않음: README.md, plan.json, cost.json(중복은 results.json 에 있음), calibration.csv, fig_*.png. 따라서 calibration 곡선 모양에 대한 주장은 하지 않는다.
- 원문(Vafa et al., arXiv 2507.06952)의 확인 수준은 낮다. PDF 는 바이너리라 읽히지 않았고, arXiv HTML 을 WebFetch(요약 모델 경유, 전체 106,535자 중 앞 100,000자)로 받은 요약만 봤다. 원문을 직접 읽은 것이 아니다. 요약에서 확인된 것은 Eq. 1·2 의 정의, Eq. 3·6 의 calibration 정의, 20구간 등간격 구간, 2,000개 입력, pseudo-state 의 Same/Diff 분해뿐이다. App. B 의 정규화 식은 확인하지 못했다(요약기가 "식과 기준 모델을 찾지 못했다"고 보고). 따라서 "원문 App. B 와 이 구현의 차이"는 확인 불가이고, 아래의 정규화 논의는 정의 수준의 추론이다.
- 이번 세션에서 열지 않은 문헌은 "미확인"으로 표시한다.

## 1. 결과 요약 (파일:키)

- results.json verdicts[0]: H-e = not_evaluable (reason area_baseline_saturated, area_score 1.069). H-f = supported (D-IB_area_eq 0.9278, D-IB_area_neq 1.0150, abs_diff 0.0872, sd_eq 0.0452, sd_neq 0.0306). H-g = refuted (extrap 0.543, early 0.607, late 0.536, interp 0.936, labels_a060+a160 0.541, labels_a060+ind 0.304, labels_ind+a160 0.252).
- pilot true, 임계값은 제안값이고 파일럿 후 확정 기록 없음(log.md 2026-11-09 주의 (b)). 정규화 기준은 결과 전 확정(기록 12).

## 2. 방법 출처(Vafa et al.)와 이 구현의 정합

원문과 일치하는 것(요약 확인): R-IB(Eq. 1)와 D-IB(Eq. 2)의 정의, 두 지표를 precision/recall 처럼 읽는다는 설명, calibration 의 oracle 쌍별 거리 20등간격 구간, 2,000개 입력(spec 4.1 과 일치).

차이 또는 불확실한 것:

1. 정규화(핵심). 요약에 따르면 원문 B.2 는 "0 = perfect accuracy, 1 = random guessing"이라고 쓰고 Table 2 캡션은 "1 = perfect, 0 = noninformative"라고 해 서로 방향이 어긋난다(요약기 보고. 원문 직접 확인 아님). 식은 못 찾았다. 이 구현은 spec 의 `(raw - raw_ref)/(raw_oracle - raw_ref)` 를 썼고(CHANGES 8), 기록 12 에서 D-IB 의 ref 를 상수 출력(D-IB raw 0)으로 바꿨다. 이 선택의 결과는 3절에서 다룬다.
2. 상태 공간과 oracle 의 성격. 원문의 state 는 이산이고 oracle 은 참 state 를 아는 모델이다. 여기서 oracle 은 "참 s 를 받는 hidden 5 짜리 MLP"(config oracle.primary: mlp, CHANGES 14)이고, 9개 레벨에 무작위 이진 라벨을 100개 표본으로 학습한다. 그 결과 oracle 의 D-IB raw 가 0.31~0.33 (ib_scores.csv, oracle, D-IB, all, seed 0~3)로, 무정보 모델의 0.5 (noninf 행)보다 낮다. 즉 이 oracle 은 "state 를 구별하는" 모델이 아니라 "s 에 매끄러운" 모델이다. 이 점이 포화의 이론적 원인이다(3절).
3. 5-B 함수족: 원문은 무작위 선형 사영, 여기는 주파수 제한 사인파 합(CHANGES 3, 사용자 결정). 논문 결과와 직접 비교할 수 없다.
4. 5-D: 원문 Table 1 은 force law 를 PySR 기호회귀로 slice(은하)별로 복원해 "slice 마다 다른 식이 나온다"를 보인다(요약 확인). 여기서는 PySR 이 설치되지 않았다(law_fits.csv, ours, seed 0 행들의 pysr_expr = pysr_not_installed). 따라서 이 실행의 5-D 는 선형 기울기만 있고, 원문이 말하는 "식의 형태가 slice 별로 다른가"는 검증되지 않았다.
5. 5-C 의 해석 방향. 원문은 D-IB_q= < D-IB_q≠ 이면 모델이 합법 수 partition(휴리스틱) 쪽으로 외삽한다고 읽는다(요약 확인). 이 팀의 H-f 는 "두 값이 같다"를 휴리스틱 아님의 근거로 둔다. 같은 논리를 따르되, 원문에서 q 는 state 의 함수이고 state 와 독립적으로 정의되는 coarsening 이다. 여기서 q(면적 10분위)는 s 와 상관이 크다(CHANGES 12, 셀프테스트 사실)는 점이 다르다(아래 3-3).

## 3. 결과별 문헌/이론 검토

### 3-1. 면적 휴리스틱 포화(H-e not_evaluable)는 예상 가능한 결과였고, 실패라기보다 지표의 한계다

- 사실: area 모델의 D-IB ood_any 점수 평균 1.069 (results.json), seed 0~3 행은 1.036, 1.066, 1.113, 1.054 (ib_scores.csv, area, D-IB, ood_any). oracle=1, 그리고 ours 1.014~1.095, random 1.010~1.098 도 비슷하다(ours/random, ood_any, seed 0~3).
- 해석(추정): D-IB 분모를 상수 기준으로 두면 `score = raw / raw_oracle` 가 된다. oracle 의 raw 가 0.3 대로 낮고(무정보 0.5), 신호가 약한 head 일수록 라벨 노이즈에 더 크게 반응해 쌍을 더 자주 "다르게" 예측하므로 raw 가 오르고 점수가 1 을 넘는다. noninf 의 점수가 1.5~1.7 로 나오는 것이 이 구조를 보여 준다(ib_scores.csv, noninf, D-IB, all/ood_any, score 1.52~1.71, score_alt 0). 즉 이 설정에서 D-IB 는 "구별 능력"이 아니라 "예측이 얼마나 들쭉날쭉한가"에 가깝게 움직이고, 최댓값이 oracle 이 아니다. 원문이 D-IB 를 R-IB 와 함께 읽으라고 한 이유(한쪽만 올리면 쉽게 속는다)와 같은 맥락이다(요약 확인).
- score_alt(iid 기준)로 보면 ours 의 ood_any D-IB 는 0.866~0.977 (seed 0~3), random 0.861~0.983, area 0.840~0.939 이다. 순서는 area < ours ≈ random 이지만 간격이 작고, ours 와 random 의 차이는 seed 편차와 구별되기 어렵다(seed 4 와 정확한 sd 는 못 봄: 미확인). 이 기준에서는 면적 모델이 포화되지 않는다는 점이 흥미롭지만, 기록 12 가 주 점수 기준을 결과 전에 const 로 못 박았으므로 주 판정을 바꾸지 말고 "참고용 사후 관찰"로만 둔다. 또 iid 기준 분모는 0.33−0.5 = 약 −0.17 로 음수라서 부호가 반전되고 분모가 작아 불안정하다(분모가 min_gap 0.05 보다 크긴 하나 구조적 취약).
- 결론: "포화"는 모델이 면적 휴리스틱 수준으로 크기를 읽는다는 뜻이 아니라, 이 정규화에서 면적 head 도 oracle 수준의 점수를 받는다는 뜻이다. 5-A/5-B 의 판별력은 이 설정에서 없다. ours 와 random-init 도 R-IB 에서 구별이 약하다(ib_scores.csv R-IB all: ours 0.7380/0.7342/0.7308/0.7336 대 random 0.7113/0.6951/0.7058/0.7050, 차이는 약 0.03). area 의 R-IB all 은 0.755~0.758 로 ours 보다 높다(area, R-IB, all, seed 0~3). 즉 R-IB 에서도 ours 가 area 위가 아니다. (H-e 판정 자체는 not_evaluable 이지만, 만약 평가했어도 "모두 면적보다 높다"는 조건은 R-IB 에서 충족되지 않을 것으로 보인다: 추정, seed 4 미확인.)

### 3-2. 002 와의 정합: probe 판별력 부재는 002 와 같은 방향

- hypotheses.md 확정 기록 12 가 인용한 002 결과: 외삽 mask baseline 0.825, real linear 0.859, attentive 0.700, 보간 R² real 0.514/0.182 대 random-init 0.174/−0.018. 면적 단서가 외삽에서 강하다는 것이다. 005 에서 area head 가 oracle 수준 점수를 얻고 random-init 과 ours 가 구별되지 않는 것은 이 관찰과 일치한다. 같은 원인(면적 ~ 크기 상관)의 두 번째 증거로 보는 것이 자연스럽다(추정). 002 수치는 이번에 원본 results.json 을 다시 열지 않았고 hypotheses.md 의 인용만 사용했다(2차 인용).
- 문헌 맥락: 이는 probe 를 해석할 때 control task/selectivity 가 필요하다는 Hewitt & Liang 2019 (EMNLP, "Designing and interpreting probes with control tasks")의 논점과 같은 계열이다. 이번 세션에서 열어보지 않은 기억 기반 인용이다(미확인). random-init 대조와 area head 가 이 팀의 control 역할인데, 둘 다 ours 와 분리되지 않았다.

### 3-3. H-f supported 는 경계에 있고, 구조적 교란 때문에 약한 증거다

- results.json: abs_diff 0.0872, 판정 임계 `sd_mult × max(sd)` = 2.0 × 0.0452 = 0.0905. 차이가 임계의 96% 로 한 걸음 안쪽이다. sd 는 head seed 5개의 편차이고, 과제/평가 입력/예측기 체크포인트는 고정이다(config seeds, feat.train_seed: 0). 모델 학습 seed 편차는 들어 있지 않으므로 sd 가 과소일 가능성이 있다(추정). 과소라면 supported 쪽으로 기운다.
- 같은 간극이 모든 모델에서 나온다. area head: D-IB_area= 0.864~0.959 대 D-IB_area≠ 1.018~1.090 (seed 0~3, all). random: 0.882~0.977 대 0.982~1.065. 면적을 쓰는 area head 와 학습 안 된 random 에서도 같은 부호의 간극이 있어, 간극이 "모델이 면적 휴리스틱을 쓴다"는 증거로 쓰이기 어렵다. 또 oracle 도 raw 에서 D-IB_area= 0.222 < D-IB_area≠ 0.338 (all, seed 0)로 간극이 있다. CHANGES 12 가 쓴 대로 q 가 같은 쌍은 이웃 크기일 확률이 높아서, 가까운 크기는 매끄러운 함수에서 라벨이 같아지기 쉽다. 이 교란은 원문의 q(next-token set partition, state 의 함수)에는 없던 것이다. 따라서 이 실험의 H-f 는 원문 Sec. 4 와 같은 논리를 그대로 적용할 수 없다.
- 어떤 방향이든 "supported"는 "휴리스틱 아님"으로 읽으면 안 된다. 같은 분해에서 area head 와 random 도 비슷하기 때문이다(귀무 분포가 없음). 필요한 비교는 "ours 의 간극 대 area/random 의 간극"이다(추정: 이 세 간극을 같은 척도로 뽑으면 어느 쪽이든 정보가 생김).

### 3-4. 상대 구조(5-A/5-C) 대 절대 법칙(5-D)의 의미

- 원문의 논지는 "상대 구조는 맞는 것처럼 보이는 모델도 절대 법칙은 slice 마다 다른 식을 쓴다"이다(요약 확인: Table 1 의 은하별 식이 서로 완전히 다르고, oracle 은 모두 같은 식). 여기서 5-A/5-C 는 정보가 약하고(3-1, 3-3), 5-D 에서만 slice 간 이질성이 보인다: 보간 0.936 대 외삽 0.543 (seed sd 0.087, 0.060). 두 값의 차이 0.39 는 sd 보다 훨씬 크다(results.json H-g per_slice).
- 그러나 같은 이질성이 random-init 에서도 나온다: random 보간 0.957, 1.027, 1.015, 0.962, 0.946(평균 약 0.98), 외삽 0.524, 0.512, 0.510, 0.508, 0.551(평균 약 0.52) (law_fits.csv, random, interp/extrap, seed 0~4. 평균은 내가 손으로 합산한 값이므로 파일에서 재계산 필요). 면적 모델도 외삽 0.58, 0.45, 0.38, 0.47, 0.39(평균 약 0.46). 이질성은 "학습된 world model 의 성질"이라기보다 "소량 라벨(약 72개, 학습 레벨 3개)로 학습한 head 가 학습 범위 밖에서 수축한다"는 현상으로 먼저 설명된다(추정).
- 따라서 H-g refuted 는 "ours 가 단일 크기 법칙을 따르지 않는다"로는 맞지만, 그 원인 귀속(world model 대 head/라벨 범위)은 이 실험으로 분리되지 않는다. 학습 범위 밖 수축은 문헌에서 두 갈래로 설명된다: (a) Kang et al. 2024 "Deep neural networks tend to extrapolate predictably"의 OCS(분포 밖에서 예측이 상수 쪽으로 간다, 003_prior.md 에서 요약 확인), (b) Xu et al. 2021 (ReLU MLP 가 지원 밖에서 선형으로 외삽, arXiv 2009.11848, 요약 확인). 둘 다 이 head 의 외삽 기울기 <1 과 방향이 같지만 이 설정을 직접 검증한 문헌은 아니다.
- H-g 기준(0.8~1.2) 자체의 문제. oracle MLP 도 외삽 기울기가 1 이 아니다: law_fits.csv oracle extrap 0.660, 0.820, 0.840, 0.878, 0.836(평균 약 0.81, 손계산). oracle_linear 는 정확히 1.0(자명). 즉 hidden 5 짜리 MLP oracle 이 이미 임계 하한 0.8 근처이고, 라벨 조합 slice 에서는 oracle 이 0.40~1.09 로 넓게 흩어진다(labels_ind+a160: 0.535, 0.646, 1.092, 0.796, 0.396). 기준을 oracle 이 아니라 상수 1 에 둔 것은 CHANGES 14 가 이미 지적한 위험이 실현된 것이다. 더 근거 있는 비교 대상은 "oracle 대비 비율"과 "slice 간 기울기 일치"(원문 Table 1 의 논리)이다. ours/oracle 외삽 비는 0.543/0.81 ≈ 0.67 (추정, 손계산).
- 라벨 조합 slice 는 더 취약하다. a060+ind 와 ind+a160 에서 기울기가 0.30, 0.25 로 가장 낮은데, 이는 라벨이 두 레벨뿐인 head 를 모든 평가 입력(9레벨)에 적합한 것이다. 라벨 두 점에서 선형 목표를 학습한 것과 같아서 외삽 구간에서 수축이 큰 것이 자연스럽다. 이 slice 에서 area head 는 기울기 ≈ 0 (−0.06~0.05, R² 0.01~0.03)이어서 사실상 상수이다. ours 는 area 보다 분명히 신호가 있다(R² 약 0.63~0.75). 이것이 이번 5-D 에서 ours 가 area 와 구별되는 유일한 구간이다. 다만 random 도 같은 값(0.19~0.30)이라 ours 의 고유 기여는 아니다.

### 3-5. 외삽 기울기 약 0.54 와 003 의 open-loop 수축은 "같은 방향이지만 같은 양이 아니다"

- 003 (내 003_prior.md 인용): `slope_ol.all_sizes.slope`(9점 log-log, rollout 출력 면적 대 참 크기)이 0.42~0.58 대의 run 이 많고 mix_wide 의 지원 밖 부분은 0.63~0.73. 005 외삽 기울기 0.543 (ours, extrap, 평균)과 숫자가 비슷하다.
- 비교가 위험한 이유: (a) 003 의 기울기는 rollout 수축을 재는 지표이고, 005 는 단일 스텝 predictor 출력(pred_out6)에서 소량 라벨 head 로 읽은 크기의 선형 적합이다. 측정 대상이 다르다. (b) 005 의 random-init 외삽 기울기가 ours 와 같다(약 0.52). 수축이 학습된 예측기의 동역학에서 나온다면 random-init 에서는 없어야 하는데 있다. 이는 5-D 외삽 기울기 0.54 가 head 와 라벨 범위의 성질에 지배된다는 쪽에 무게를 둔다. 003 의 수축과 같은 현상이라는 증거로는 쓰기 어렵다. (c) 003 은 서로 다른 학습 분포 4종 사이의 비교인데 005 는 002 조건 (b) 한 개, 학습 seed 0 한 개의 체크포인트만 쓴다(config feat.train_seed: 0).
- 정합이라고 말할 수 있는 최대치(추정): 두 실험 모두 학습 크기 밖에서 "참 크기보다 덜 벗어난 값으로 수축"하는 방향이다. 다만 005 에서 이 수축이 random-init 에도 있으므로, 다음 정확한 비교는 003 의 4개 분포에서 같은 5-D 절차를 돌려 기울기가 학습 분포 폭과 함께 오르는지(003 의 H-Support 후보 가설과 연결) 보는 것이다.

### 3-6. 5-B 곡선 이탈 기준(≤ 0.1)의 구조적 문제

- calibration_dev.csv: pred_out6, extrap_any 의 이탈이 ours 0.335(seed 0), 0.351(seed 1); random 0.335, 0.354; area 0.314, 0.327; truth(참 함수를 모델로 쓴 행) 0.333, 0.344. 즉 참 함수 자체의 이탈이 0.33 이다. x축이 oracle MLP 이므로 외삽 구간에서 x축(oracle 거리)이 포화하여 45도선이 성립하지 않는다는 뜻이다(CHANGES 14: "MLP oracle 은 학습 범위 밖에서 포화"). 0.1 이라는 임계는 truth 도 통과하지 못하는 값이어서, 이 설정에서 H-e 의 세 번째 조건은 oracle 선택의 부산물로 항상 불충족이다(추정: calibration.csv 곡선은 못 봄). oracle_linear 의 이탈도 0.224 대로 0이 아니다. 따라서 포화가 없었어도 H-e 는 not_evaluable 이거나 refuted 였을 가능성이 높다.

## 4. 빠진 baseline / 인용

- 빠진 baseline:
  1. 마스크 면적을 쓰되 head 용량을 맞춘 것 외에, "학습 평균 상수 출력"과 "단순 선형 회귀(s←면적)" baseline. 5-D 에서 ours/random/area 가 같은 수축을 보이므로, 외삽 기울기를 설명하려면 head 없이 라벨 범위만으로 정해지는 기준선(예: 라벨 범위로 클리핑한 선형 회귀)이 필요하다.
  2. 인코더 출력(층 0 또는 DINO 인코더 최종)에서 같은 절차를 돌린 ours(spec 4.6 의 분기상 이번 행에는 predictor 출력만 해당). predictor 가 입력보다 외삽을 해치는지 보려면 입력 쪽 대조가 필요하다.
  3. 학습 seed 여러 개(체크포인트 seed 0,1,2). 지금은 head seed 편차뿐이다.
  4. random-init 에서 oracle 대비 비율. 이미 random 행이 있으나 해석이 판정에 연결되지 않았다.
- 인용해야 할 선행연구(열람 수준 표시):
  - Vafa et al., ICML 2025 (arXiv 2507.06952): 정의와 calibration 은 HTML 요약 수준 확인, App. B 식은 미확인. 정규화와 oracle 정의를 원문에서 직접 확인해야 한다.
  - Vafa et al., 2024, "Evaluating the World Model Implicit in a Generative Model" (원문 인용 목록에서 존재만 확인, 미열람): Myhill-Nerode 기반 평가. state 정의 방식이 R-IB/D-IB 의 이산 state 와 연결된다.
  - Hewitt & Liang 2019 (EMNLP): control task, selectivity. 기억 기반, 미열람.
  - Kang et al., 2024 (OCS), Xu et al., 2021 (ReLU 외삽): 003_prior.md 에서 요약 확인.
  - DINO-WM 자체(Zhou et al.)와의 연결은 이 리뷰에서 다루지 않았다(미확인).

## 5. 연구 공백과 후보 가설 (검증 설계 제안, 확정 아님)

1. 후보 가설 J1: 5-D 외삽 기울기의 수축은 head 와 라벨 범위의 성질이다. 반증 가능 예측: 학습 레벨을 더 넓게 둔 head(라벨을 외삽 구간 가까이에서 뽑은 head)에서는 ours, random, area 모두 기울기가 오르고 셋의 차이가 작다. 이미 있는 캐시 위에서 CPU/Colab 소규모로 가능(추정).
2. 후보 가설 J2: D-IB 를 신호로 쓰려면 oracle 을 "참 state 를 아는 lookup"(레벨 정체성 원-핫 입력)으로 바꿔 원문 정의에 맞춰야 한다. 그러면 oracle D-IB raw 가 0.5 에 가까워져 정규화 분모가 맞고, 면적 head 는 레벨 정체성을 몰라서 점수가 내려갈 것이다(추정). 이는 임계/정규화 변경이므로 결과 후 변경에 해당해 별도 기록과 사람 승인이 필요하다.
3. 후보 가설 J3: 5-C 의 간극은 q 와 s 의 상관 때문이다. 반증 가능 예측: s 를 조건부로 통제한 쌍(예: 크기 쌍 거리가 같은 쌍 안에서 q 같음/다름 비교)에서는 ours, random, oracle 모두 간극이 사라진다.
4. 003 연결: 003 의 4개 학습 분포 체크포인트에서 같은 5-D 를 돌려 외삽 기울기가 분포 폭에 따라 오르는지 본다(003 의 H-Support 후보와 직결).
5. 연구 공백: 이 팀 설정(천 크기, 소량 라벨 외삽)에서 inductive bias probe 를 연속 state 에 적용한 선행연구는 내가 확인한 범위에 없다(체계적 검색은 하지 않음: 추정).

## 6. 한계

- 원문 정규화 식 미확인(가장 중요). 원문 수치와의 직접 비교 불가.
- ib_scores.csv 는 일부만 읽었다(seed 4, 일부 pair_group 미확인). 손계산 평균값(random/area/oracle 의 기울기 평균, 간극)은 파일로 재계산해야 한다.
- calibration.csv, 그림, README, plan.json, 004 summary 의 세부는 읽지 않았다.
- 002 수치는 hypotheses.md 의 2차 인용이다.
- 이 실행은 pilot: true 이고 임계값이 확정 기록 없이 제안값이다. 판정 문구의 신뢰도에도 영향을 준다.

## 7. 확신도

- 중간. 근거: (1) 포화 구조, 5-D 의 random-init 동일 수축, H-g 기준 대비 oracle 의 비1 기울기는 CSV 숫자에서 직접 나온 관찰이라 높은 편이다. (2) 반면 원문 App. B 를 확인하지 못했고 정규화 방향이 원문 요약에서도 모호하다. 원인 귀속(왜 D-IB 가 노이즈를 보상하는지, q-s 상관의 영향 크기)은 추정이다. 003 과의 정합 판단은 "측정 대상이 달라서 방향 일치까지만"이 확실한 결론이고 그 이상은 낮은 확신도다.
