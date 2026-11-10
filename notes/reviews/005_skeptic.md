# 005_inductive_bias_probe 리뷰: 회의론자 관점

- 작성: analyst-skeptic, 2026-11-10. 다른 분석가의 리뷰(004_*, 005_*)는 읽지 않았다.
- 대상: results.json(판정 H-e not_evaluable / H-f supported / H-g refuted, 층 pred_out6), results/{ib_scores.csv, calibration_dev.csv, law_fits.csv, tasks_meta.json}, logs/colab_20261109/, src/exp005/*, config.yaml, CHANGES.md, README.md, 004 deep.py, 004 results/summary.json, 명세 4절, hypotheses.md 확정 기록 9·12, log.md 2026-11-09.
- 코드는 실행하지 않았다. 아래 수치는 모두 CSV/JSON/로그 원문이거나, 그 값을 손으로 평균/표준편차(ddof=1) 낸 것이다. 손 계산 값은 "(계산)"으로 표시한다. 코드 실행 없이 읽어서 추론한 것은 "추정" 또는 "의심"으로 표시한다.

## 요약 (한 문단)
세 판정 모두 모델 표현의 성질보다 판정 장치의 성질을 더 많이 반영한다. (1) H-e 의 "not_evaluable" 은 중립 결과가 아니다. 포화 검사를 빼면 다섯 하위 조건 중 네 개가 실패하는 상태였고(R-IB 와 D-IB 모두 ours < area, D-IB 는 random 과 구분 안 됨, calibration 이탈 0.33 > 0.1), calibration 조건은 참 함수 자신(0.33)과 linear oracle(0.23)도 통과하지 못한다. 그러니 구조적으로 통과할 수 없는 검정이었고, 이것은 실행 전에 예측할 수 있었다. (2) D-IB 에서 ours 와 area 의 순서는 정규화 기준(const 대 iid)에 따라 뒤집힌다. 확정 기록 12 의 근거("oracle D-IB raw 약 0.5")는 실제 실행에서 성립하지 않았다. (3) H-f 의 "supported" 는 여유가 0.0033 뿐이다(0.0872 대 0.0905). 같은 차이(0.0879)를 보인 random-init predictor 는 seed sd 가 작다는 이유만으로 "refuted" 가 된다. 이 기준은 잡음이 클수록 통과하기 쉽다. (4) H-g 의 "refuted" 는 참 s 를 입력받는 MLP oracle 도 한 slice 에서 범위를 벗어나는 기준에서 나왔다. random predictor 의 기울기도 ours 와 같다. (5) 평가 입력 수가 config/명세의 2000 이 아니라 900 으로 실행됐는데, 어디에도 기록되지 않았다.

---

## 치명

### C1. H-e 는 구조적으로 판정 불가이거나 통과 불가였다. "not_evaluable" 이 사실상의 반증을 가린다
- 근거 1 (판정 순서, 확인됨): `src/exp005/verdicts.py:35-47` 는 R-IB, D-IB 순으로 돌면서 area 점수가 `baseline_saturated_score`(0.95) 이상이면 즉시 not_evaluable 을 반환한다. 그래서 그 뒤의 ours>area, random 비교, calibration 조건(`verdicts.py:48-58`)은 results.json 에 남지 않는다.
- 근거 2 (가려진 조건들, ib_scores.csv / calibration_dev.csv 에서 계산, pair_group ood_any, seed 5개):
  - R-IB(score, iid 기준): ours 0.7717, area 0.7920, random 0.7400 (계산). **ours < area** 이다. area 는 포화되지 않았다(0.79 < 0.95).
  - D-IB(score, const 기준): ours 1.0404, area 1.0691(= results.json `area_score`), random 1.0596 (계산). **ours < area** 이다. |ours − random| = 0.019 < 2 x max sd(0.033) = 0.067 이므로 random 과도 **구분되지 않는다**.
  - calibration 이탈(extrap_any): ours 평균 0.335 (seed 값 0.3348, 0.3514, 0.3448, 0.3042, 0.3383; calibration_dev.csv 14, 32, 50, 68, 86행), 기준은 ≤ 0.1.
  - 포화 검사를 빼면 H-e 는 하위 조건 5개 중 4개에서 실패한다(통과는 R-IB 의 random 대비 차이 하나). README 12행의 "판정은 규칙을 적용한 결과일 뿐" 이라는 문장과 명세 5절의 "not_evaluable 은 음성 결과가 아니다" 라는 문장 때문에 이 판정을 "보류" 로 읽기 쉽다. 그러나 데이터에는 ours 가 면적 휴리스틱을 넘는다는 근거가 없다. InD 쌍(ind)에서도 ours 는 area 보다 낮다: R-IB 0.66~0.68 대 0.68, D-IB 0.76~0.82 대 0.82~0.87 (ib_scores.csv 7-8, 47-48행 등).
- 근거 3 (calibration 조건은 통과 불가, 확인됨): x 축이 참 함수가 아니라 3개 레벨로 학습한 MLP oracle 이다(`stages.py:374`, `Dor = Dm["oracle"]`, `oracle.primary: mlp`). 그 결과 참 함수 g_k 자체(`truth` 행)의 이탈이 0.328, linear oracle 이 0.227 이다 (calibration_dev.csv 19, 37, 55, 73, 91행 / 18, 36, 54, 72, 90행, 계산). 기준 0.1 은 참값조차 통과하지 못하는 기준이다. 5-B 의 D_k 는 학습 레벨 3개의 s 값 3개만 본다(`tasks.py:83-92`, 명세 4.3). 주파수 0.3~1.5 의 사인파 합은 점 3개로 외삽이 정해지지 않으므로 어떤 학습자도 참 g_k 를 맞출 수 없다. 결국 이 검정은 "모델이 MLP oracle 의 외삽 습관을 닮았는가" 를 잰다.
- 근거 4 (D-IB "포화" 는 기준의 산물, 확인됨): const 기준에서 D-IB score = raw/raw_oracle 인데, oracle D-IB raw 는 0.29~0.33(all/ood)에 불과하다. 그래서 입력과 무관한 i.i.d. 무작위 모델(`noninf`)이 1.52~1.73 점을 받는다(ib_scores.csv 103, 113, 118행 등). 점수 1 은 상한이 아니다. 잡음이 많을수록 점수가 오른다. 0.95 "포화" 검사는 점수가 [0,1] 에 있다는 가정에서만 의미가 있는데, 이 가정이 깨져 있다. area 가 "너무 좋은 baseline" 이어서 포화된 것이 아니다.
- 사전 예측 가능성: 가능했다. (a) 5-A D_k 가 3개 레벨뿐이라는 점(CHANGES.md 7), (b) MLP oracle 이 학습 범위 밖에서 포화한다는 점(CHANGES.md 14, selftest.py:303 주석)은 실행 전에 이미 기록되어 있었다. (c) selftest 의 정규화 점검(selftest.py:186-203)은 9개 레벨의 라벨을 모두 아는 planted "perfect" 모델을 썼고, 3개 레벨로 학습한 실제 oracle 의 OoD D-IB 는 점검하지 않았다. 이것을 점검했다면 oracle D-IB < 0.5 와 noninf 점수 > 1 이 드러났을 것이다(추정).
- 심각도 이유: H-e 는 후보 I 의 중심 가설인데, 이번 설계로는 어떤 결과가 나와도 "크기 기반 외삽" 을 지지할 수 없었다.

### C2. D-IB 비교의 부호가 정규화 기준 선택에 따라 뒤집힌다. 확정 기록 12 의 전제가 실제 데이터에서 거짓이었다
- 확인됨(ib_scores.csv 의 score_alt = iid 기준, ood_any): ours 0.9390, area 0.8939, random 0.9091 (계산). 이 기준에서는 **ours > area** 이고 area 점수 0.894 < 0.95 라 포화도 아니다. const 기준(주 점수)에서는 **ours < area** 이다. 이유: oracle D-IB raw(약 0.31)가 iid 수준(0.5)보다 낮아 iid 분모가 음수가 되므로, 점수가 raw 에 대해 감소 함수가 된다. 같은 데이터에서 "D-IB 가 높다/낮다" 의 방향 자체가 기준에 달려 있다.
- 확정 기록 12(hypotheses.md:362)의 근거는 "무작위 이진 라벨 과제에서 oracle 의 D-IB raw 와 i.i.d. 기준 raw 가 모두 약 0.5" 였다. 실제 oracle D-IB raw 는 all 0.307~0.329, ood_any 0.293~0.315 이다(ib_scores.csv 63, 73, 203, 213, 343, 353행 등). 약 0.5 인 것은 ind 그룹(0.466~0.492)뿐이다. 원래 명세 기준(iid)으로도 |분모| ≈ 0.18 > min_gap 0.05 이므로 점수가 정의된다. 결정은 결과 전에 내려졌으니 HARKing 은 아니다. 그러나 결정의 전제가 synthetic selftest 의 planted 모델에서만 성립했고, 그 결정이 H-e 의 판정 경로(포화 대 비교)를 정했다.
- 함의: 이 데이터의 D-IB(OoD 쌍)는 "state 구분 능력" 을 단조적으로 재는 지표가 아니다. 어떤 기준으로 보고하든 해석에 쓰면 안 된다(아래 I4 와 연결).

### C3. H-f "supported" 는 과장이다. 경계값 통과이고, 잡음이 클수록 유리한 기준이며, 같은 효과를 보인 대조군은 "refuted" 다
- 여유(확인됨, results.json 52-58행): abs_diff 0.08718, 기준 2.0 x max(0.04525, 0.03058) = 0.0905. 여유는 0.0033(약 4%)이다. sd_mult 가 1.93 이하였다면 refuted 였다(계산). sd_mult 2.0 은 impl default 이고 확정 기록이 없다(config.yaml:68, CHANGES.md 19).
- 대조군과의 비교(ib_scores.csv, pair_group all, 계산):
  - ours: area= 0.9279, area≠ 1.0150, 차 0.0872, sd 0.045 / 0.031 → supported
  - random-init predictor: area= 0.9418, area≠ 1.0297, 차 **0.0879**, sd 0.037 / 0.032 → 같은 규칙을 적용하면 0.0879 > 0.0736 이라 **refuted**
  - area 휴리스틱 head: area= 0.9080, area≠ 1.0502, 차 0.1422 → refuted (예상대로)
  - ours 와 random 은 효과 크기가 사실상 같다. 판정이 갈리는 이유는 ours 의 seed 0(area= 0.8907)과 seed 4(0.8731)가 sd 를 키운 것뿐이다. "seed 편차 안에서 같다" 는 차이가 없다는 증거가 아니라 차이를 검출하지 못했다는 뜻이다. 잡음이 클수록 통과한다.
- 방향(확인됨): ours 의 area= < area≠ 차이는 순수 면적 모델 차이(0.142)의 약 61% 이고 방향도 같다(계산). 명세 4.4 해석 규칙("area= 가 훨씬 낮으면 면적 쪽")을 그대로 적용하면, ours 는 oracle(차 0)과 area(0.142) 사이에서 area 쪽에 더 가깝다. "휴리스틱 아님"(명세 4.7 의 H-f 이름)이라고 쓸 근거가 없다.
- 교란(CHANGES.md 12): q 가 같고 레벨이 다른 쌍은 이웃 크기 쌍에 몰린다. 정규화가 oracle 의 같은 패턴(raw area= 0.21 대 area≠ 0.33)을 나눠서 일부를 상쇄한다. 그러나 그 oracle 은 3개 레벨 MLP 의 외삽이라 OoD 쌍에서 무엇을 기준으로 삼는지 불분명하다(C1 근거 4, I4). pair_group all 은 OoD 쌍이 대부분이다(all 의 calibration 쌍 404,550 중 ind 43,071; calibration_dev.csv 2, 8행. 5-A 쌍 수의 비율도 비슷할 것으로 추정).
- seed sd 의 의미가 좁다(I6): 과제, 라벨, 평가 입력이 모든 seed 에서 같으므로(CHANGES.md 20) sd 는 head 초기화 변동만 담는다. 기준이 "잡음 안에서 같음" 이므로 sd 추정이 바뀌면 판정이 바로 바뀐다.

---

## 중요

### I1. H-g "refuted" 는 표현의 성질이 아니라 기준과 head 의 성질이다
- oracle 도 실패한다(law_fits.csv, MLP oracle = 참 s 입력, 계산): labels_ind+a160 기울기 평균 0.693(seed 값 0.535, 0.646, 1.092, 0.796, 0.396; 28, 63, 98, 133, 168행), extrap 0.807(seed 0 은 0.660). 참 state 를 입력받는 모델이 H-g 를 반증당한다. linear oracle 은 구성상 1.000 이다(목표가 s 의 선형 함수이므로 자명).
- 수학적 감쇠(추정, 표준 결과): MSE 로 학습한 예측기는 E[s|x] 에 가깝다. 예측값 yhat 을 참 s 에 OLS 하면 기울기는 Cov(yhat, s)/Var(s) 이고, 수축된 예측기에서는 1 미만이 된다(완벽하지 않은 예측기는 평균 쪽으로 당겨진다). weight decay 1e-2, 라벨 72개, 학습 레벨 범위 밖 외삽이 감쇠를 키운다. 기울기 < 1 은 "크기 법칙이 일관되지 않다" 를 뜻하지 않을 수 있다.
- 학습 목표값이 2~3개뿐이다(확인됨): 5-D 목표 `s_flat` 은 레벨별 명목값이다(`stages.py:78`, `224`). 3레벨 head 는 서로 다른 목표값 3개, combo head 는 2개로 학습된다(`stages.py:94-96`). 점 2개로 학습한 회귀를 9개 레벨 전체에 OLS 해서 얻은 기울기 0.25~0.30(labels_a060+ind, labels_ind+a160)은 외삽 실패를 다시 확인할 뿐이다.
- random predictor 와 차이가 없다(계산): random 의 slice 별 평균 기울기는 interp 0.981, extrap 0.521, early 0.586, late 0.518, labels_a060+a160 0.533, labels_a060+ind 0.274, labels_ind+a160 0.251 이다. ours(0.936, 0.543, 0.607, 0.536, 0.541, 0.304, 0.252)와 거의 같다. 학습된 predictor 에 귀속시킬 수 있는 차이가 없다.
- 라벨 표본은 한 번만 뽑혔다(`tasks.py:95-101`, seed `seeds.d` 고정): slope_sd(예: labels_a060+a160 0.0056)는 라벨 표본 변동을 포함하지 않는다.

### I2. 기록되지 않은 config 변경: 평가 입력 2000 → 900 (재현성 깨짐)
- 확인됨: 저장소 config.yaml:33 은 `n_inputs: 2000` 이다(명세 4.1 "2,000개", selftest.py:436 도 2000 을 단언). Colab 로그를 보면 `tasks_20261109-092300.log:4-5` 와 `092433.log:4-5` 에서 "eval inputs 2000" 이 `ValueError: only 900 candidate inputs for kind 0, 1000 requested` 로 두 번 실패했다. 그 뒤 `092456.log:4` 와 본 실행 `tasks_20261109-112048.log:4` 에서 "eval inputs 900" 으로 진행했다. 이 값은 `stages.py:111` 이 cfg 에서 읽은 것이므로 Colab 세션의 config.yaml 이 수정된 것이다. tasks_meta.json:26 `n_eval: 900`.
- 기록 누락: CHANGES.md 7절은 "config 변경은 8a 항목 뿐", log.md:190 과 203 은 임계값/시드 변경 없음이라고 적었다. n_inputs 변경과 tasks 단계의 실패 2회는 CHANGES.md 와 log.md 어디에도 없다. results.json 의 `config_sha256`(2fd84b...)은 저장소 config.yaml 의 해시가 아닐 것이다(직접 계산은 못 함, 추정). 저장소 config 로 다시 돌리면 tasks 단계가 ValueError 로 멈춘다.
- 영향: 결과 값에 미친 영향은 작을 것으로 본다(쌍 수는 여전히 많다, 추정). 그러나 "모든 수치는 results.json 에서", "명세와 달라진 점 전부 CHANGES.md" 규칙 위반이고, 사람 승인 없이 실행 중 설정을 바꾼 사례다. 같은 노트북으로 돌린 004 에도 비슷한 수정이 있었는지 확인이 필요하다(이 리뷰 범위 밖).
- 원인(확인됨): 002 캐시의 test 부분은 900 episode 이고 early 후보는 episode 당 1개 위치뿐이다(후보 900). 명세가 가정한 규모를 데이터가 지원하지 않았다.

### I3. random-init 대조는 "학습 안 된 모델" 이 아니라 "실제 DINOv2 인코더 + 무작위 predictor" 다. ours 는 거의 모든 지표에서 이것과 같다
- 확인됨: `stages.py:165` 는 `DP.random_init_predictor(model, ...)` 를 쓰고, `feats.py:47-52` 는 실제 모델의 encoder 로 만든 z 를 두 predictor 에 똑같이 넣는다. 004 `deep.py:54-65` 의 predictor 는 residual(`x = attn(x) + x; x = ff(x) + x`)이므로, 무작위 predictor 의 출력에도 입력 z 의 정보가 대부분 남는다(추정). 명세 4.6 의 "random-init 인코더·predictor" 중 인코더 쪽은 predictor 층에서 쓰이지 않는다.
- 결과: R-IB 0.772 대 0.740, D-IB 1.040 대 1.060, calibration 0.335 대 0.334, 5-D 기울기 전부 비슷하다(계산). 더 단순한 대안 설명은 이것이다: 측정된 구조는 고정된 DINOv2 인코더 특징과 head 의 성질이고, world model(predictor) 학습의 기여는 R-IB 의 약 +0.03 정도다. 이번 실험은 "world model 이 크기를 배웠다" 와 "DINOv2 특징에 크기/면적 단서가 원래 있다" 를 구분하지 못한다.

### I4. 5-A 의 D_k 를 3개 레벨에서만 뽑았기 때문에 OoD 그룹의 R-IB/D-IB 가 Vafa 지표와 다른 것을 잰다
- 확인됨: tasks_meta.json:13-17 `pool_levels_used_5A: [a060, ind, a160]`, `tasks.py:37-58`. 라벨은 9개 레벨에 있지만, OoD 레벨 6개의 라벨은 어떤 모델도 보지 못한다. 그래서 OoD 쌍의 "정답 state 구분" 은 정의되지 않는다. oracle 도 MLP 외삽일 뿐이다(R-IB oracle = 1 은 s 의 결정적 함수라서 자명).
- CHANGES.md 7 은 이 변경을 "(사용자 결정 필요)" 로 표시했는데, hypotheses.md 와 log.md 에서 이 항목의 사용자 결정 기록을 찾지 못했다(grep 확인). 미결 결정을 둔 채 본 실행이 진행됐다.
- C1 근거 4 와 C2 의 근본 원인이다. H-e, H-f 판정은 모두 pair_group ood_any/all 에 걸려 있다.

### I5. 사전 등록 이탈: 판정을 가른 값이 확정되지 않은 impl default 다
- `verdict.sd_mult: 2.0`, `baseline_saturated_score: 0.95`(config.yaml:68-69, "impl default")는 명세 4.7 에 없고 구현자가 정했다. 파일럿 후 확정 기록이 없다(log.md:203 이 명시). 이 두 값이 각각 H-f(여유 0.0033)와 H-e(포화 경로)의 판정을 직접 정했다. dev_max 0.1, 0.8~1.2 도 "파일럿에서 확정" 이 예정이었으나 확정되지 않았다. 파일럿은 heads `--max-tasks 3 --seeds 0` 까지만 돌고 metrics/report 를 돌리지 않아서(README 93행) 확정할 근거 자체가 없었다.
- 층 선택도 확정되지 않은 값에 의존한다: 004 summary.json 의 `k_star`, `m_star` 가 null 인 것(`drop_threshold 0.2`, 역시 미확정 제안값) 때문에 4.6 분기 "no_vanishing" 행이 선택됐고, 그래서 pred_out6 하나만 검사됐다(plan_20261109-112046.log:4).
- results.json `pilot: true`(config 플래그 유지 + ALLOW_MAIN_WITH_PILOT_TRUE). 실행 범위가 본 실행이라는 것은 log.md:203 의 사용자 확인에만 근거한다. 산출물만 보면 파일럿과 구분되지 않는다.

### I6. seed 변동은 head 초기화만 담는다
- 확인됨: 과제 라벨, D_k, 사인파, 5-D 라벨 부분집합, 평가 입력이 seeds.* 로 고정되어 5개 head seed 에서 같다(CHANGES.md 20, `stages.py:82-96`). 판정에 쓰이는 "seed 편차" 는 과제와 표본 추출 변동을 빼고 계산한 값이다. H-f 처럼 "편차 안" 이면 지지하는 규칙에서는 sd 가 작게 추정될수록 반증 쪽으로, 크게 추정될수록 지지 쪽으로 기운다. H-e 의 random 비교처럼 "편차 밖" 이면 지지하는 규칙에서는 반대다. 어느 쪽이든 판정의 통계적 의미가 약하다.

---

## 사소

- S1 (잠재 버그, 이번 실행에는 영향 없음): `stages.py:352-353` 의 `continue` 는 5-A 예측이 하나라도 없으면 그 seed 의 5-B 와 5-D 계산까지 건너뛴다(`for seed` 루프 안). 5-B 와 5-D 블록(368-371, 386-388)도 같은 패턴이다. 이번에는 예측 25개가 모두 있어서 문제가 없었다.
- S2: `verdicts.py:8-14` `_vals` 는 seed 를 키로 덮어쓴다. 같은 조건의 행이 중복되면 경고 없이 마지막 값만 쓴다(이번 CSV 에서는 중복 없음 확인).
- S3 (분포 차이): D_k 는 train episode 의 8개 프레임 전체(`tasks.py:27-30`, 7200개)에서 뽑지만, 평가 입력은 early/late 위치 하나씩에서 뽑는다(후보가 각 900인 점에서 추정). 또 D_k 의 train episode 는 world model 학습에 쓰인 입력이라 predictor 특징이 in-sample 이다. oracle/area 에는 이런 비대칭이 없다. 크기는 작을 것으로 본다(추정).
- S4 (누수 점검 결과, 확인됨): 평가 입력은 `part == "test"` 행(`tasks.py:13`), 모든 D_k 와 5-D 라벨은 `part == "train"` 행(`tasks.py:27-30`, `95-101`)이다. 면적 분위 q 는 평가 입력의 면적으로 정하지만(`stages.py:70`) 라벨을 쓰지 않으므로 누수가 아니다. 면적 특징 표준화는 학습 풀 통계다(`stages.py:192`). 라벨 누수는 찾지 못했다. 다만 002 캐시의 train/test 가 episode 단위로 분리되어 있는지는 002 코드를 확인하지 않았다(미확인). 같은 episode 의 early/late 프레임 쌍은 "같은 state" 쌍으로 R-IB 에 들어간다(CHANGES.md 10). 비중은 작을 것으로 본다(추정).
- S5 (쌍 계산, 확인됨): `ibmetrics.py:52-61` 의 행렬식은 selftest 에서 brute force 와 대조됐다(selftest.py:165-176). 대각선은 제외된다(`ibmetrics.py:48`). 대칭 행렬이라 순서쌍을 두 번 세지만 평균에는 영향이 없다. 버그를 찾지 못했다.
- S6: calibration 의 `norm` 은 seed 별 oracle 최대 쌍 거리(15.46~18.98, calibration_dev.csv)라서 쌍 하나가 정규화를 정한다. 구간은 쌍 수와 무관하게 같은 가중이다(CHANGES.md 13, impl 결정).
- S7: PySR 이 설치되지 않아(`law_fits.csv` 의 `pysr_not_installed`) 명세 4.5 의 "같은 식이 나오는지" 는 보지 못했다. 원문(Vafa et al. App. B)의 정규화식도 여전히 확인하지 않았다(CHANGES.md 8, 5절).
- S8: 실행 환경이 conda 가 아니라 venv 다(results.json environment, `CONDA_DEFAULT_ENV: null`). 002 확정 기록 10 의 Colab 예외를 따른 것으로 보이지만, 005 에 대한 별도 예외 기록은 찾지 못했다.
- S9: pilot 의 attempts.json(`stage:tasks: 0`)에는 Colab 에서 실패한 tasks 2회가 남지 않았다. 연속 실패 카운터가 성공 시 초기화되는 구조로 보인다(추정). 재시도 상한은 지켜졌지만 실패 이력을 감사할 수 없다.

---

## 대안 설명 (같은 결과를 더 단순하게)
1. H-e not_evaluable: "면적 baseline 이 너무 강하다" 가 아니다. OoD 쌍에서 D-IB 의 기준점(3개 레벨 MLP oracle)이 낮아서 잡음이 많은 모델일수록 점수가 1 을 넘는다(C1, C2).
2. H-f supported: 효과가 없어서가 아니다. ours 의 head seed 변동이 random 대조보다 조금 커서 임계 폭이 넓어진 것이다. 효과 크기는 random-init predictor 와 같다(C3).
3. H-g refuted: 표현에 크기 법칙이 없어서가 아니다. MSE 회귀의 감쇠, 목표값 2~3개짜리 학습, 학습 범위 밖 외삽 때문이다. 같은 head 를 쓰는 MLP oracle 과 random predictor 도 같은 양상이다(I1).
4. 전 지표에서 ours ≈ random predictor: 측정 대상은 world model 학습이 아니라 고정 DINOv2 특징이다(I3, 추정).

## 이 결과를 믿으려면 추가로 필요한 실험 (최대 3개)
1. 지표를 다시 정의하고 다시 계산하기(재학습 거의 없음): 5-A 의 D_k 를 9개 레벨 train 에서 뽑는다(002 캐시에 6개 레벨 train episode 추가). 그것이 안 되면 판정을 ind 그룹으로 제한한다. 정규화 기준은 "라벨 lookup 으로 9개 레벨 정답을 아는 perfect-state 모델" 로 하고 iid/const 두 기준을 함께 보고한다. 5-B 는 x 축을 `truth` 로 바꾸고, 포화 검사는 점수 상한이 1 인 경우에만 적용한다. 이 정의와 sd_mult, 포화 기준, dev_max 를 결과 전에 확정 기록으로 남긴다.
2. 원천 분리 대조: 같은 head 와 과제로 (a) 004 캐시에 이미 있는 인코더 최종 층(DINOv2 고정 특징, `tokens_k12`)과 random-init 인코더(`rand_k12`), (b) pred_out6 를 비교한다. 과제/라벨 시드(`seeds.tasks_a`, `seeds.d`)도 3개 이상 바꿔 변동에 포함한다. H-f 는 "seed 편차 안" 대신 사전에 정한 동등성 여유(Δ)를 쓰는 TOST 형태로 판정한다.
3. H-g 의 기준 보정: 같은 attentive head 에 "참 s + 잡음(ours 의 InD R² 에 맞춘 양)" 을 입력으로 준 noise-matched oracle 을 돌려, 감쇠만으로 기대되는 slice 별 기울기를 구한다. ours 의 slice 간 기울기 차이를 그 기준과 비교한다. combo head(목표값 2개)는 판정에서 빼거나 별도로 보고한다.

## 확신도
- **높음**: 수치 기반 지적(C1 의 가려진 조건 값, C2 의 기준에 따른 순서 역전, C3 의 여유 0.0033 과 random 대조의 0.0879, I1 의 oracle 기울기, I2 의 2000→900 변경). 모두 CSV/JSON/로그 원문과 손 계산에서 나왔고, 판정 함수(`verdicts.py`)를 줄 단위로 읽어 경로를 확인했다. 손 계산에 반올림 오차는 있을 수 있지만 판정을 뒤집을 크기는 아니다. 단 H-f 여유는 원래 작으므로 results.json 값(abs_diff 0.08718, sd 0.04525)을 기준으로 삼았다.
- **중간**: I3(무작위 predictor 의 residual 경로가 인코더 정보를 보존한다)와 I1 의 감쇠 설명은 코드 구조와 표준 결과에서 추론했다. 특징 수준에서 직접 확인하지는 않았다(추정). I2 의 config_sha256 불일치는 해시를 직접 계산하지 않았다.
- **낮음**: S3 의 분포 차이 크기와 S4 의 002 episode 분리 여부는 확인하지 않았다.
