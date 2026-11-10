# 008 사후 재집계 통계 리뷰 (analyst-stats)

대상: experiments/008_posthoc_reaggregate_004_005 (results.json, run.py, config.yaml). 원 데이터: experiments/004_depth_truncation/results/depth_r2.csv.
재계산 코드/출력 (008 run.py 와 코드 공유 없음, csv+numpy):
- analysis/recompute_stats.py -> analysis/recompute_stats.txt (lab07 python; 재현, A/B/C)
- analysis/recompute_extra.py -> analysis/recompute_extra.txt (lab05 python, scipy 필요; 낙폭 도달 가능성, t 구간, MSE 환산)
(모든 경로는 /home/jaekyungkim/mylab/experiments/008_posthoc_reaggregate_004_005/ 아래.) 아래 수치는 이 두 출력 또는 results.json 에서만 인용.

## 1. 재현성: results.json 과 일치하는가
- A: encoder/predictor x train/interp/extrap 전 깊이에서 Δ 평균, (평균-2sd) 의 results.json 대비 최대 차 = 0.0 (표시 정밀도 내). Δ>0 깊이 수도 일치(encoder 7/7/7, predictor train 6, interp 2, extrap 0 of 7).
- B: r2_common 곡선 최대 차 <= 3.3e-16. Var_all = 0.49999293541242407 (json 0.49999293541242396).
- C: predictor inside 비율 interp 0/7, extrap 5/7 (=0.714) 재현. 004 원 결과는 4/7 (=0.571).
- 사전 확인: random-init 대응 행 encoder 140/140, predictor 420/420 키 일치 (json precheck). CSV 행 수는 json 의 n_real_rows 와 충돌하지 않음 (별도 검증 안 함).
- 결론: 계산 오류는 찾지 못함. 구현은 정확하다. 아래 문제는 모두 "지표/규칙 설계" 문제이지 산술 문제가 아니다.

## 2. A: 통제 대비 Δ 곡선

관측값(recompute_stats.txt, 평균):
| 모듈 | split | Δ 범위 | 최대 낙폭(규칙 임계 0.2) |
|---|---|---|---|
| encoder | train | 0.150~0.275 | 0.125 |
| encoder | interp | 0.266~0.365 | 0.099 |
| encoder | extrap | 0.056~0.132 | 0.076 |
| predictor | train | 0.000~0.025 | 0.000 |
| predictor | interp | 0.000~0.048 | 0.005 |
| predictor | extrap | -0.010~0.000 | 0.010 |

- "관측 안 됨" 은 산술적으로 맞지만 정보가 거의 없다. predictor 의 Δ 는 최대 0.048 이라 0.2 하락이 구조적으로 도달 불가능하다 (천장 효과). 이 규칙은 predictor Δ 에 대해 검정력이 0 이므로 "소실 안 됨" 증거가 아니라 "규칙이 작동할 수 없음" 이다.
- 효과 크기: 학습된 predictor 가 random-init predictor 에 더하는 정보는 train R² 로 +0.025 (d=6), interp +0.048, extrap -0.0003 (d=6; d=1~5 에서는 -0.006~-0.010). 즉 predictor 층 probe R² 는 거의 전부 (R² 0.83~0.86) 인코더 입력에서 온 것으로 보이며 학습 predictor 의 기여는 작다 (추정; 원인 귀속은 이 데이터만으로 확정 불가).
- 인코더의 Δ 는 d=0 에서 가장 크고 (train 0.275) d=2 에서 0.150 으로 내려간 뒤 거의 평탄 (0.15~0.19). d=0 은 random 의 패치 임베딩이 특히 약한 지점이라 러닝맥스가 d=0 에 고정되는 구조. 0.2 임계 미달이지만 train 에서 0.125 (d=0->2) 낙폭은 있다. 이것은 소실이 아니라 d=0 의 random 기준선 낮음에 기인할 가능성이 높다 (추정).
- 불확실성 단위 혼재 (핵심 지적):
  - encoder: 학습 시드가 없고 단일 동결 네트워크. sd 는 probe seed 5개. 이 sd 는 0.000~0.007 로 사실상 0 (Δ/sd 가 40~ inf). 그래서 encoder 의 "Δ>0 7/7" 은 비교 대상이 "probe 재샘플 잡음" 뿐이어서 자명하다. 네트워크 간 변동(인코더 초기화/학습 시드)은 한 번도 포함되지 않는다. 일반화 가능한 주장으로 쓰면 안 된다 (n=1 네트워크).
  - predictor: 학습 시드 3개 단위 (probe seed 평균 후). df=2 라서 sd 추정 자체가 매우 부정확 (sd 참값/표본 sd 의 95% 구간 [0.52, 6.28]).
  - 두 모듈의 "Δ>0" 은 단위가 달라 같은 표에서 비교하면 안 된다. 현 json 은 unit_note 로 구분하고 있어 정직하지만, 요약 status 에서 나란히 나온다.
- "평균-2sd>0" 규칙의 성격: n=3 일 때 t = 2*sqrt(3) = 3.46 에 해당 (df=2 단측 p = 0.037, 양측 0.074). 다중비교 보정 없음 (7 깊이 x 3 split = 21 검정, Bonferroni 단측 95% 라면 t > 14.44 필요). 예: predictor train d=3 Δ 의 t 95% CI [0.0038, 0.0194], t=6.41; interp d=3 CI [-0.0333, 0.1020] (0 포함); extrap d=3 CI [-0.0195, -0.0001]. 보정을 하면 "Δ>0 깊이" 수는 줄어든다. 특히 predictor interp 의 2/7 와 extrap 의 음의 Δ 는 df=2 로는 가볍게 해석해야 한다.
- extrap 의 Δ 가 약하게 음수 (학습 predictor < random-init) 인 것은 d=3 에서만 CI 가 0 을 간신히 배제 (p2=0.049). 3 시드, 비보정이므로 우연으로 설명 가능. 방향성 주장 금지.

## 3. B: 척도 통일과 공통 분산 R² 복원

타당성:
- 복원 방법 (MSE_level = (1-r2_level)*Var_all) 은 자기검증을 통과했다. 복원한 split R² 와 CSV ALL 행 R² 의 최대 차 8.2e-15. Var_all 도 직접 재계산해 일치. 이 검증은 비자명하다 (레벨 행이 Var_level 로 정규화되어 있었다면 실패했을 것). 따라서 "추정치로 대체하지 않음" 원칙을 지키며 복원은 정확하다고 본다. 남는 가정: 레이블이 에피소드 상수이고 면적비의 자연로그(002 config)라는 것 — 자기검증으로 간접 확인됨.
- 단 유효 표본 주의: 레이블 값이 split 당 3(train), 2(interp), 4(extrap) 개뿐이라 (ln 면적비: train {-0.512, 0, 0.470}, interp {±0.222}, extrap {-1.351, -0.919, 0.693, 0.908}) R² 는 "몇 개 군집을 가르는가" 를 재는 값이다. 100 episode x 레벨은 pseudo-replication 이 있고 episode 단위 CI 는 독립 단위(학습 시드)를 대변하지 못한다.

핵심 발견 (split R² 의 분모 불일치가 InD/OoD 격차의 상당 부분을 만든다):
| split | Var_split | Var_split/Var_all | encoder d=12 MAE | predictor d=6 MAE | predictor d=6 split R² | 공통 R² |
|---|---|---|---|---|---|---|
| train | 0.161 | 0.32 | 0.128 | 0.114 | 0.859 | 0.955 |
| interp | 0.0493 | 0.10 | 0.119 | 0.111 | 0.553 | 0.956 |
| extrap | 0.9655 | 1.93 | 0.290 | 0.283 | 0.852 | 0.714 |
- interp 은 split R² 0.55 로 train 0.86 보다 크게 낮지만 MAE (0.111 vs 0.114) 와 공통 R² (0.956 vs 0.955) 는 같거나 더 낫다. 즉 interp 의 낮은 R² 는 분모(레이블 분산이 train 의 0.31배)가 작아서 생긴 지표 인공물이다. MAE 기준으로 interp 에서 일반화 격차는 관측되지 않는다.
- extrap 은 반대다. split R² 0.852 로 train 과 같아 보이지만 MAE 0.283 vs 0.114 (약 2.5배), MSE 0.143 vs 0.023 (약 6배), 공통 R² 0.714 vs 0.955. 분모가 1.93 배 크기 때문에 R² 가 후하게 나온다. 004 의 "extrap 에서 predictor 겹침" 은 지표가 가린 결과일 가능성이 크다 (아래 C).
- 임계 0.2 의 환산 불균일: split R² 0.2 하락 = ΔMSE 0.0322 (train), 0.0099 (interp), 0.193 (extrap), 공통 R² 0.2 = ΔMSE 0.100. 같은 숫자 0.2 가 split 간 약 20배 다른 오차 증가를 뜻한다. 공통 R² 로 바꾸면 train/interp 은 MSE 가 0.023~0.043 범위라 0.2 낙폭이 도달 불가능 (천장). B 의 "관측 안 됨" 도 규칙 도달 불가능성의 영향을 받는다.
- 깊이 곡선 크기 (공통 R², predictor extrap): 0.713 (d=0) -> 0.695 (d=2) -> 0.714 (d=6). 낙폭 0.018 수준. MAE 는 0.290 -> 0.301 -> 0.283. 이 변동은 시드 sd (아래) 와 비교해야 하나 B 는 sd 를 계산하지 않았다 (평균만). 크기 해석은 불가.
- B 는 MAE/MSE 의 시드 sd 와 CI 를 내지 않는다 (json 에 평균만). 곡선 간 차이를 읽기 전에 이를 추가해야 한다.

## 4. C: 겹침 재계산

- 재계산 결과 predictor inside: interp 0/7, extrap 5/7 (0.714). 004 원 결과 extrap 4/7 (0.571). 둘 다 기준 0.8 미만이므로 상태("관측 안 됨")는 동일. 이동 폭은 1 깊이이며, 6/7 이면 뒤집힌다. 즉 판정은 깊이 1개 경계에 있고 sd 가 df=2 추정이라 안정적이지 않다.
- 검정 설계의 문제: 허용폭 tol = 2*max(sd_train, sd_ood) 이므로 시드 분산이 작을수록 허용폭이 줄어 겹침 판정이 어려워진다. 정밀할수록 "겹침 없음" 이 되는 역설. predictor extrap 의 gap 은 -0.017~0.009 (|gap| <= 0.017) 로 사실상 0 인데 tol 이 0.004~0.010 (d=0 은 0.000) 이라 2 깊이가 탈락한다. 고정 동등 여유 ±0.05 를 적용하면 extrap 7/7, interp 0/7 (recompute_extra.txt). 즉 "겹침 안 됨" 판정은 평균 차이가 아니라 tol 이 작아서 나온 것이다. 단 B 의 결과대로 그 gap≈0 자체가 분모 인공물이므로, 동등 여유를 쓴다고 "InD=OoD" 가 되는 것은 아니다.
- encoder: 학습 시드가 없어 "계산 불가" 는 맞다. 참고로 probe seed sd 기준 tol 은 0.000~0.012 로 의미 없는 수준이며 004 원 결과 (interp 0, extrap 0.143) 는 같은 이유로 "겹침 없음" 이 자명하게 나온다 (재계산 inside: interp 0/7, extrap 1/7).
- depth 0 의 predictor 는 encoder 출력과 동일 입력 (Δ=0.000 정확히) 이므로 시드 sd=0 이 되는 퇴화점. 판정 7 깊이 중 1개가 기계적으로 sd=0.

## 5. 우연/시드 운으로 설명 가능성
- 가능성 낮음(산술 측면): 모든 수치가 CSV 에서 결정적으로 재현됨. 난수 없음.
- 높음(일반화 측면): 학습 시드 3개(df=2), 인코더 n=1 네트워크, 21 검정 비보정, 사후(post hoc) 탐색. 004 판정을 뒤집거나 강화하는 근거로 쓸 수 없다. config 임계(0.2, 2.0, 0.8)가 004 와 동일하게 고정된 것은 좋은 점(사후 임계 튜닝 없음).

## 6. 권고
1. A/B 의 소실 규칙에 "도달 가능성" 점검 추가 (Δ 최대폭 < 임계면 "검정력 없음" 으로 표기). 현재 "관측 안 됨" 은 "소실 없음" 으로 읽히기 쉽다.
2. InD/OoD 비교는 MAE 또는 공통 R² 를 주 지표로, split R² 는 보조로 (interp 의 격차는 분모 인공물; extrap 의 겹침도 인공물).
3. 겹침은 TOST/사전 동등 여유 (예: ±0.05 R² 또는 MAE 단위로 정의) 로 재정의. sd 가 작아질수록 불리한 규칙 폐기.
4. 불확실성 단위를 모듈별로 분리 표기. encoder 의 시드 sd 는 네트워크 변동이 아니다. 인코더 시드 반복이 없으면 일반성 주장 금지.
5. 009 양성 대조 (검출력 확인) 가 이 문제들의 직접 해결책이다. 그 전에는 004 의 refuted 를 "규칙의 검정력 미확인 하에서의 refuted" 로 읽는 것이 안전하다.

## 확신도
- 재현/산술 정합성: 높음 (독립 구현이 1e-15 수준으로 일치, 자기검증 통과).
- 분모 인공물 (interp R² 낮음, extrap R² 후함) 판단: 높음~중간 (MAE/MSE/공통 R² 모두 같은 방향이고 수식으로 설명됨; 단, 레이블 군집 수가 적음).
- 규칙 천장 효과/검정력 부재 판단: 높음 (최대 낙폭 수치가 직접 증거).
- Δ>0, extrap 음의 Δ, C 의 5/7 vs 4/7 의 방향성 해석: 낮음 (3 시드 df=2, 비보정, 1 깊이 차이).
- 전체: 중간. 산술은 확실하나 모든 모든 해석은 n=3 시드, 인코더 n=1, 사후 분석이라 약하다.
