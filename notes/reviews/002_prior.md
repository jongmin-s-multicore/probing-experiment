# 002_single_size_sweep 리뷰: 선행연구/이론 관점 (analyst-prior)

- 작성일: 2026-11-05. 관점: 결과가 기존 문헌 및 001 결과와 어떻게 맞는가.
- 읽은 것: experiments/002_single_size_sweep/{results.json(부분 읽기), config.yaml, README.md}, notes/hypotheses.md(후보 G, 확정 기록 3~7, 10), notes/log.md(2026-11-04, 11-05 의 002 항목), notes/research/prior.md, notes/lit_review.md, reports/001_size_collapse_diagnosis.md. 다른 분석가의 002 리뷰와 notes/reviews/002_* 는 읽지 않았다.
- 수치 표기: `rj:` = experiments/002_single_size_sweep/results.json. 이 머신에는 Q1~Q3 중간 파일(cells.npz 등)이 없어 results.json 에 없는 값(레벨별 R², 기울기, 편향)은 계산하지 못했다. 필요한 곳에 "미확인"으로 표시한다.
- 문헌 상태: 이번 세션에서 논문 원문을 새로 읽지 않았다. 아래 논문 서술은 notes/lit_review.md 와 notes/research/prior.md 에 전사된 내용(웹 검색 요약 수준, reporter 가 원문 대조 안 함) 또는 기억에 의존한다. 각 항목에 상태를 적었다. 인용 전 원문 확인이 필요하다.
- 사전 등록 임계값(0.1, 0.2, 95 백분위, 0.9, 0.99 등)은 이 리뷰에서 바꾸자고 하지 않는다. 바꿔야 한다고 보는 부분은 "새 실험으로 제안"에만 썼다.

---

## 0. 한 줄 판정 행이 실제로 뜻하는 것 (먼저 요약)

| 판정 행 | 표면 | 실제로 말해 주는 것 | 말해 주지 않는 것 |
|---|---|---|---|
| H_a interp linear/attentive pass | DINOv2 가 크기를 mask 면적 이상으로 복원 | 변형이 큰 후반 프레임(8-11)에서 mask 면적 ridge 가 무너지고(R² -0.042) DINOv2 가 그보다 낫다 | 초기 평평한 프레임에서는 mask 면적과 거의 동급(아래 1절). "DINOv2 특이적" 보유 증거는 약함 |
| H_a extrap linear/attentive fail | 외삽에서 실패 | mask 면적 baseline 이 이미 R² 0.825 라 마진 0.1 을 못 넘음. attentive 는 baseline 보다 낮음 | 정보가 없다는 뜻이 아님. 사전 등록 마진 규칙이 baseline 이 높은 영역에서 구조적으로 불리 |
| H_b last_block pass | 마지막 블록까지 정보 보존 | 잔차 스트림이 선형 판독 가능한 정보를 거의 그대로 통과시킴. (a) 단일 크기 학습에서도 같은 값(차이 -0.008) | 보존이 학습에 의해 보장된다는 뜻이 아님. 이 점은 실패할 수 없는 시험에 가까움(2절) |
| H_b last_rollout_step fail | rollout 끝에서 정보 손실 | 인코더 토큰으로 학습한 probe 를 rollout 예측 latent 에 적용하면 R² 가 0.30 하락 | 하락이 "크기 정보의 소실"인지 "예측 latent 의 분포 이동으로 probe 가 전이 안 됨"인지 구분 못함. trained 크기에서도 하락 |
| H_c steering pass | predictor 가 크기 방향을 사용 | 크기(ridge) 방향으로 밀면, 같은 probe 로 읽은 예측 크기가 무작위 방향보다 더 변함 | "다음 latent 예측에 크기를 쓴다"는 것. 측정 지표가 개입 방향과 정렬된 probe 의 판독이라 거의 항등적 효과일 수 있음(4절) |
| H_c subspace_removal not_evaluable | 판정 불가 | 6셀 모두 INLP 후 MLP 가 크기를 여전히 읽음(not_removed). 규칙대로 결론 없음 | T 값(0.38-0.70)을 H-c 근거로 쓰면 안 됨 |
| cond_a_vs_b 두 행 pass | 조건 간 차이 검출 | last_rollout_step: (a) 드롭이 (b) 보다 0.422 큼(CI 0.399-0.446), 6개 시드 모두 같은 방향. last_block: 차이 -0.008 (R² 1% 미만) | last_block 의 pass 는 통계적 검출일 뿐 실질 차이 없음(효과 크기 하한이 이 패밀리에 사전 등록 안 됨) |
| 4c single_ind medium / multi large | 진단 | (a) 활성 이동은 noise floor 의 3.2-4.9 배, 오차비 1.05-1.74 로 large 아님 | multi 의 large 는 (b) 에서 나온 것이고 해석 규칙(문구)은 (a) 층별 결과에 대한 것. 5.0 경계에 5.02-5.30 으로 근접(5절) |

---

## 1. H-a (Q1 인코딩)

### 1.1 결과 (rj:decision_table)
- interp linear: R² real 0.514, random-init 0.174, mask baseline -0.042. delta vs mask 0.556 [0.352, 0.778], vs random-init 0.340 [0.206, 0.489]. Holm 조정 p 0.004. pass.
- interp attentive: real 0.182, random-init -0.018, mask -0.042. delta vs mask 0.225 [0.015, 0.454], vs random-init 0.200 [0.065, 0.344]. 조정 p 0.030. pass. probe seed 별 real 은 0.108-0.271 (rj:decision_table[1].result.r2_real_per_probe_seed), 즉 pass 하한(CI 0.015)이 경계에 가까움.
- extrap linear: real 0.859, random-init 0.789, mask 0.825. delta vs mask 0.034 [0.007, 0.067] < 0.1. fail. vs random-init 0.070 [0.053, 0.087].
- extrap attentive: real 0.700, random-init 0.620, mask 0.825. delta vs mask -0.125 [-0.151, -0.090]. fail.
- 초기 평평 프레임 보고(rj:h_a_early_flat_report, 판정 없음): interp linear delta 0.0966 (real 0.869 vs mask 0.772, 임계 0.1 바로 아래), extrap linear delta -0.003 (real 0.901 vs mask 0.903), attentive 는 둘 다 mask 보다 낮음(-0.461, -0.191).

### 1.2 문헌/001 과의 일치
- 001 과 일치: 001 보고서 3.4절(rvit/pixel 격자가 gt 와 비슷한 수준으로 크기를 읽음, random-ViT 대비 선택성 +0.02~0.06)이 보인 "크기 정보는 DINOv2 특이적이 아니다"가 002 의 외삽 행에서 재현된다. random-init 대비 delta 가 extrap linear 0.070 에 그친다. 002 는 001 의 지적(대조군 없음)을 해소했고, 그 결과가 "정보는 있으나 DINOv2 사전학습의 기여는 작다"는 001 의 방향과 같다. 이것은 동결 인코더 + 큰 사물의 면적이 패치 수에 반영된다는 구조적 추정(prior.md 1.2, 추정)과도 모순되지 않는다.
- LeWM Table 1 의 DINO-WM 행(Linear r 0.977-0.997; lit_review 2.2, reporter 전사, 원문 미확인)과 "r 은 높다"는 점에서 방향이 같다. 002 후반 프레임 Pearson r: extrap 0.973, unseen 0.954, trained 0.918 (rj:q1.cells["log_nominal_area_ratio|linear_pooled|dino|all|late"]). 다만 LeWM 쪽 불일치(MSE 대비 r 이 높음; lit_review 2.2, stats/skeptic 지적)와 같은 패턴이 002 에도 있다. extrap 에서 r²=0.947 인데 R²=0.859, interp 에서 r²=0.580 인데 R²=0.514 로, 학습 레벨(trained)에서는 r²=0.843 ≈ R² 0.841 인 것과 대조적이다. 즉 probe 가 안 본 크기에서 상관은 유지하고 크기(스케일/오프셋)는 틀린다. 이것은 R² 만으로 "정보 유무"를 판정하면 안 되고 기울기/편향을 함께 봐야 한다는 뜻이다(추정: R² 정의가 평가 부분집합의 평균 기준이면; stats.py 정의는 확인하지 않음).
- attentive < linear: Punzo et al. (arXiv 2606.09646; prior.md 2.1, 검색 요약 수준)은 V-JEPA 계열에서 attentive/MLP probe 가 linear 보다 크게 높다고 요약된다. 002 는 반대다(interp 0.182 vs 0.514, extrap 0.700 vs 0.859). 설명 후보(추정): Q1 probe 는 학습 라벨이 3개 값(a060/ind/a160 의 log 명목 면적)뿐이라 사실상 3-클래스 문제이고, 비선형 헤드는 라벨 범위 밖 외삽을 못 한다. Xu et al. 2021(신경망의 외삽; lit_review 에는 skeptic 의 기억 기반 인용, 서지 미확인)이 지적한 문제와 같은 구조다. 따라서 "attentive 가 약하다"는 크기 정보 부재가 아니라 probe 의 외삽 한계로 읽어야 한다. 이 경우 Q1 의 외삽 행 두 개는 H-a 의 시험이라기보다 "probe 가 3점에서 외삽할 수 있는가"의 시험이 섞인다(lit_review 의 "이중 외삽" 지적과 일치).

### 1.3 불일치/주의
- interp pass 는 mask baseline 이 후반 프레임에서 음수(-0.042)이기 때문이다. 초기 프레임에서는 mask 면적이 0.772, 후반에서는 -0.042 로 무너진다. 즉 이 pass 는 "DINOv2 가 면적 이상의 정보를 가진다"의 증거로 읽을 수 있지만(접힘/가림으로 mask 면적이 크기와 어긋나는 구간), 동시에 interp 평가 레벨이 2개(a080, a125)뿐이라 라벨 분산이 작아 R² 분모가 작다. 같은 interp 에서 MAE 는 0.118 로 trained MAE 0.124 와 같은 수준이다(rj:q1.cells[...|late].interp.mae.point, trained.mae.point). 즉 오차 절대 크기는 학습 크기와 다르지 않은데 R² 만 0.51 로 낮다. R² 의 레벨 의존성(분모 효과)이 크므로 interp R² 0.51 을 "보간 성능이 낮다"로 읽으면 과장이다(추정; 분모 정의는 코드 미확인).
- 주 타깃이 에피소드 상수인 명목 면적비 로그다. 001 보고서 7절이 남긴 우려(probe 가 천 면적이 아니라 그리퍼 간격 같은 상관 단서를 읽을 가능성; 어떤 파일에서도 검사 안 됨)는 002 에서도 검사되지 않았다. 후반 프레임에서 그리퍼가 천을 끄는 구간이면 그리퍼/팔 위치가 크기와 상관할 수 있다(추정). 이 때문에 "mask 면적 baseline 이상"이 곧 "천 크기 표현"은 아니다.
- 5 probe seed 중 linear_pooled 는 5개 값이 소수점까지 같다(rj:decision_table[0].result.r2_real_per_probe_seed). ridge 는 결정론적이므로 linear 의 probe seed 5개는 독립 정보가 아니다. attentive 쪽만 seed 변동이 있다.
- random-init 대비 CI 하한 > 0 은 4행 모두 충족하지만(interp 0.206/0.065, extrap 0.053/0.065 부근), 크기는 extrap linear 에서 0.07 로 작다. "DINOv2 사전학습이 크기 판독을 가능하게 한다"고 쓰려면 이 효과 크기를 함께 적어야 한다.
- 사전 등록 후 변경(주 타깃, 후반 창)이 pass 를 만든 구조임을 투명하게 적어야 한다. 변경은 결과 확인 전에 기록됐으므로(hypotheses.md 확정 기록 6, 7) 부정은 아니지만, 원래 문서 기준(마스크 면적 기반 크기 타깃)에서는 mask baseline 이 포화되어 H-a 가 평가 불가였을 것이라는 이유도 함께 쓰는 것이 정직하다(config.yaml q1.secondary_target 주석).

### 1.4 빠진 baseline / 인용
- 그리퍼/팔 위치만으로 크기를 예측하는 baseline 또는 마스킹 probe (001 보고서 8절 7번이 이미 제안, 002 에서도 미수행).
- 레벨별 기울기/편향(예: 예측 log 면적 vs 참 log 면적의 레벨별 평균). results.json 에는 없고 q1 원자료(Drive cells.npz)에 있다고 README/results.json 은 말한다(rj:exploratory.per_level_note). 미확인.
- 인용 후보(상태 표시): Hewitt & Liang 2019 control task/선택성(기억 기반, 서지 미확인; lit_review 는 이를 2차 정보로만 기록), Alain & Bengio 2016 linear probe(기억 기반), Belinkov 2022 probing 재검토(skeptic 파일 경유, 검색 확인 표시가 있으나 이번 세션 미확인).

---

## 2. H-b (Q2 보존)

### 2.1 결과
- last_block: 3 학습 시드 모두 Q2 R² 가 Q1 ceiling(0.872)을 오히려 넘는다. drop 평균 -0.0085, CI [-0.0106, -0.0062]. pass (rj:decision_table[4]).
- last_rollout_step: Q1 ceiling 0.820, Q2 R² 0.529/0.525/0.500, drop 0.291/0.295/0.321 (평균 0.302, CI [0.282, 0.324]). 임계 0.2 초과, fail. 3시드 모두 임계 초과(`all_training_seeds_drop_below_threshold: false`).
- 층별(rj:q2_layerwise): (b) 의 layer 6 R² 변화가 layer 0 대비 +0.003~+0.013 (unseen). (a) 도 -0.006~+0.006.
- 출력 공간 rollout 시점별 (rj:exploratory.q2_layer_time_grid_r2_point_estimates, 탐색, 판정 없음), unseen linear: (b) seed0 step1 0.853, step4 0.802, step8 0.669, step12 0.529. (a) seed0 step1 0.702, step4 0.243, step12 0.082. trained 레벨 step12: (b) 0.540/0.054/0.183 (시드 0/1/2), (a) -0.086/-0.196/-0.169.

### 2.2 문헌과의 일치
- SG-JEPA(arXiv 2609.10464; lit_review 2.4, 원문 미확인)의 "excess NMSE = rollout latent 의 probe 오차 - GT 인코딩 latent 의 probe 오차"와 H-b 의 drop 정의는 사실상 같은 구조다. 002 는 이 지표를 R² 로 옮긴 것이다. SG-JEPA 와 lit_review 가 기록한 "DINO-WM 오차가 horizon 에 따라 누적(Fig 4a)"과 (b) 의 step 1→12 단조 하락(0.853→0.529)은 방향이 같다. 단, SG-JEPA 의 DINO-WM 은 원 구현과 달라(LeWM 식 action conditioning; lit_review 2.4) 직접 비교는 안 된다.
- prior.md 3절 가설 3("학습에서 크기 분산이 0이면 predictor 가 크기를 보존할 압력이 없다", 추정)과 (a) 대 (b) 결과가 정합한다. rollout step12 에서 (a) 0.08-0.11 대 (b) 0.50-0.53 이고 6개 시드가 겹치지 않는다. 002 는 이 가설에 대한 첫 정량 지지 후보다. POKEWORLD(arXiv 2607.27017; prior.md 2.2, 요약 페이지 수준)의 "예측 목표가 latent 에 무엇이 남는지 결정한다"와도 같은 방향이다. 다만 POKEWORLD 는 숨은 파라미터이고 002 의 크기는 관측 가능한 양이라는 점이 다르다.
- 001 과의 일치: 001 에서 ol(open-loop) 기울기가 k 에 따라 0.58(k=3), 0.41(k=6), 0.23(k=12)로 떨어졌다(001 보고서 3.3절). 002 의 rollout R² 가 step 에 따라 떨어지는 것과 방향이 같다.

### 2.3 해석상 위험 (중요)
1. last_block 은 실패하기 어려운 시험이다. ViT 블록은 잔차 연결이 있어 입력 스트림이 출력에 더해진다. layer 0 의 R²(0.8727)와 layer 6 의 R²(0.875-0.886)가 거의 같고, 크기 분산이 전혀 없는 (a) 에서도 같다(change_vs_layer0 -0.006~+0.006, rj:q2_layerwise.conditions.single_ind). 즉 이 pass 는 predictor 가 크기를 보존하도록 학습했다는 증거가 아니라 "선형 판독 정보가 잔차 경로로 남는다"는 구조적 사실과 합치한다(잔차 스트림 관점은 Elhage et al. 2021 Transformer Circuits 로 기억하나 서지 미확인). H-b 를 "보존"으로 쓰려면 last_block 행은 약한 증거라고 적어야 한다.
2. last_rollout_step 의 하락은 정보 소실과 probe 전이 실패가 섞여 있다. 이 시험은 인코더 토큰으로 학습한 probe 를 예측 latent 에 그대로 적용한다. 예측 latent 는 인코더 토큰 분포와 다를 수 있어(평활화, 오차 누적 등) probe 입력의 분포 이동만으로도 R² 가 떨어진다. 증거: (b) 의 trained 크기도 같은 방식으로 떨어진다(step12 0.54/0.05/0.18, 시드 간 변동이 매우 큼). trained 크기에서 시드 1 이 0.054 라는 것은 "크기 라벨과 무관하게 rollout 표현이 흐트러진다"는 신호다(추정). 따라서 (b) 의 0.30 하락 전체를 크기 정보 소실로 귀속하면 안 된다.
3. 라벨이 에피소드 상수이므로 가장 단순한 baseline 이 있다. 마지막 context 프레임의 인코더 토큰을 그대로 복사한 예측(persistence/identity)은 크기 라벨이 변하지 않으므로 Q1 ceiling 에 가까운 R² 를 낼 것이다. 이 baseline 이 results.json 에 없다. 이것이 없으면 "rollout 예측이 persistence 보다 크기를 더 잃는다"는 점을 말할 수 없다. 반대로 말하면 (b) 의 step-12 R² 0.52 가 persistence(~0.82)보다 낮다는 것은 predictor 가 크기 정보를 "추가로 잃는다"는 뜻이다(추정, 미측정). prior.md 4절이 이미 identity baseline 을 권고했다(lit_review 5.3: prior 만 지적).
4. 시드 변동이 큼. 에피소드 부트스트랩 CI 는 학습 시드 변동을 포함하지 않는다(rj:ci_caveat). drop CI [0.282, 0.324]는 "시드 3개 평균이 임계 0.2 와 구분되는가"의 근거가 못 된다. 다만 시드 개별 값 3개가 모두 0.29 이상이라 방향은 일관된다(unseen 기준).
5. 평가 대상 "unseen"은 interp+extrap 6개 레벨 혼합이고, extrap(a026, a248) rollout 은 입력이 predictor 에 OOD 이며 (b) 도 마찬가지다. 레벨별 분해는 results.json 에 없다(미확인).

---

## 3. 조건 (a) 대 (b)

- 결과: cond_a_vs_b:last_rollout_step, drop_a - drop_b = 0.422 [0.399, 0.446]. (a) drop 0.739/0.706/0.729, (b) 0.291/0.295/0.321 (rj:decision_table[8]). 6개 시드 값이 겹치지 않는다. Holm 조정 p 0.004.
- 이 행의 "pass"는 방향 주장이 아니라는 코드 주석이 있다(note). 그러나 데이터는 (a) 드롭이 모든 시드 쌍에서 더 크다. 가장 방어 가능한 서술은 "(a) 모든 시드에서 (b) 의 어느 시드보다 큼, 학습 시드 3 대 3" 정도다. 시드 3개의 t/CI 로 일반화하지 않는다(001 보고서 7절도 같은 지적).
- cond_a_vs_b:last_block 은 차이 -0.008 [-0.010, -0.006]. 통계적으로는 검출되지만 R² 0.8% 크기다. 이 패밀리에는 효과 크기 하한이 사전 등록되지 않았다(config prereg.effect_size_floor 는 Q1 마진과 Q2 하락만). 방향도 "(b) 가 layer 0 대비 더 증가"인데, 이 방향을 (b) 가 크기를 더 보존한다는 근거로 쓰면 과장이다(층 0 에서 이미 거의 같고 변화량은 ±0.01).
- 동일 예산은 코드와 로그로 맞았다: 두 조건 모두 epochs 15, steps_per_epoch 1857, steps_computed 27855 (rj:train_budgets). tqdm 총 step 후보 207(valid)/1857(train)이 계산값과 일치(관측은 "눈으로 비교", 공식 optimizer step 카운트 아님).
- 교란 후보:
  1. (a) 대 (b) 는 크기 다양성뿐 아니라 학습 에피소드의 성질도 다르다. ind 900 은 (b) 의 ind 300 을 포함하지만 (b) 의 a060/a160 600 이 추가되고 ind 는 300 으로 줄어든다. 학습 분포의 폭과 ind 데이터 양이 함께 바뀐다. 이것은 이미 설계에 선언된 교환이고 후보 H/003(혼합 학습)이 다른 방식으로 분리하려는 부분이다.
  2. 층별 probe 는 (a) 에서 안 본 크기의 활성으로 학습되므로 절대 비교 불가(rj:q2_layerwise.limitation). 출력 공간 probe 는 인코더 토큰으로 학습해서 (a),(b) 에 같은 probe 가 적용되므로 이 문제가 없다. 그래서 cond_a_vs_b:last_rollout_step 이 (a)(b) 비교의 신뢰할 만한 행이다. last_block 행은 보조(4c medium 주의 문구 포함).
  3. 시드 3개 + 재개 run 포함: log.md 에 따르면 (b) s0 는 epoch 5, s1 은 epoch 9 에서 체크포인트 재개, 6개 run 모두 패치된 train.py(val() 의 `@torch.no_grad()` 줄 누락, 가중치 영향 없다는 것은 추정) 사용. (b) seed 간 변동(예: trained step12 0.54/0.05/0.18)이 재개 효과인지 본질적 변동인지 구분할 수 없다. 재개한 (b) s0, s1 과 재개 안 한 s2 의 값이 크게 다르지 않으나(step12 unseen 0.529, 0.525, 0.500), trained 쪽 변동은 크다.
- 문헌 정합: 001 baseline_single(ind 단일)에서 먼 크기가 ind 쪽으로 이동한 관찰과 002 (a) 의 R² 붕괴(step12 0.08-0.11)는 "정보가 학습 크기 쪽으로 눌린다"에 맞다. 그러나 002 의 R² 는 방향을 말하지 않는다(편향/기울기 미보고). 001 보고서 3.4절의 "ol probe 편향 a026 +0.24, a248 -0.17"처럼 레벨별 부호 있는 편향을 002 에서도 봐야 "ind 로 수렴"과 "무정보로 분산 증가"를 구분할 수 있다(미확인). 001 보고서 5.3은 ind 가 평가 범위의 가운데라 "가운데로 압축"과 구분이 안 된다고 했는데, 002 의 (b) 는 학습 크기가 a060/ind/a160 로 넓어져서 이 교란이 줄지만 완전히 풀리지는 않는다.

---

## 4. H-c (Q3 사용)

### 4.1 결과
- steering(ridge), 조건 (b): 6셀(시드 3 x 층 2) 모두 크기 방향 T 가 무작위 100방향 95 백분위 밖, Spearman 1.0. T: middle 0.087/0.087/0.081 대 95 백분위 0.012/0.011/0.012. last 0.064/0.054/0.065 대 0.010/0.009/0.010. 순위 p 는 모두 0.0099(=1/101, 100개 무작위로 얻을 수 있는 최소값). Holm 조정 p 0.0198. pass (rj:decision_table[6]).
- subspace_removal(INLP): 6셀 모두 not_removed. 참고로 T 는 0.38-0.70 대 무작위 0.16-0.20 이었으나 규칙상 결론 금지 (rj:decision_table[7]). LEACE 도 전 셀에서 removed=false (rj:exploratory.q3_exploratory_methods).
- 탐색: meandiff steering T 는 (b) middle 0.17-0.19, last 0.39-0.40, (a) middle 0.09-0.10, last 0.19-0.22. (a) 에는 ridge steering 의 확증 행이 results.json 에 없다(rj:exploratory.q3_exploratory_methods 에 (a) meandiff/LEACE 만 있음). 패치 평균 L2 변화(README 의 보조 지표 ii)는 results.json 에 저장되어 있지 않다(grep 으로 확인; 미저장 또는 미집계).

### 4.2 문헌과의 관계 및 해석 위험
- 개입으로 "사용"을 확인하겠다는 설계 자체는 문헌 권고와 일치한다. Elazar et al. 2021 amnesic probing(lit_review 4.3, skeptic 가 서지/요지 확인했다고 표시, 이번 세션 미확인), Ravichander et al. 2021, Li et al. 2023 Othello-GPT 개입, Nanda 2023(skeptic 파일 경유, 이번 세션 미확인). 001 보고서의 지적(무작위 20개, 분모 폭주 gain 지표, 평행이동의 한계)을 002 가 해소했다(무작위 100개, 절대 변화, 분산 매칭).
- 그러나 "pass"가 가리키는 바에는 한계가 있다.
  1. 효과 지표가 읽기 probe(Q1 ridge)와 정렬된 방향의 이동이다. 개입 방향 d 는 ridge 방향이고 효과는 같은 계열의 ridge 판독으로 잰다. 무작위 방향은 (분산·이동 크기를 맞췄지만) 판독 가중치와 거의 직교한다. 그러면 d 를 밀었을 때 판독값이 많이 변하는 것은 "판독이 d 와 정렬돼 있다"는 성질에서 거의 자동으로 나온다. predictor 가 크기를 다음 latent 의 계산에 쓰는지와는 다른 질문이다. 층 'last'(마지막 블록 출력)에서는 개입이 거의 출력으로 직결되므로 더 그렇다(추정; q3.py 코드 미확인).
  2. 이 비교는 "크기 방향 대 임의 방향"이다. "크기 방향 대 다른 의미 있는 방향(그리퍼 위치, 프레임 인덱스, 행동 방향 등)"이 아니다. 판독 정렬 효과를 빼려면 같은 크기의 다른 물리 변수 방향과 비교해야 한다.
  3. Spearman 1.0 은 7점 선형 개입의 거의 항등적 결과이고 p_rank 0.0099 는 100개 무작위로 얻을 수 있는 하한이라 변별력이 포화됐다(T 비가 5-8 배라 방향은 확실하지만, "얼마나 유의"는 말할 수 없다).
  4. 효과 크기: T 의 단위가 "Q1 판독 크기(log 면적)의 변화"이므로 0.087 이 의미 있는 값인지 판단하려면 레벨 간 실제 log 면적 차(예: ind 대 a160 은 약 0.47)와 비교해야 한다. 이 비교는 results.json 에 없다. 001 에서 디코딩 면적 변화가 0.3-1.6% 로 작아 "사용 여부 결론 불가"였던 것과 대비해, 002 에서 효과가 "크다"고 말하려면 이 척도 비교가 필요하다(미확인).
  5. 개입이 (b) 에서만 확증 행이고 (a) 는 없다. "(b) 로 학습한 predictor 가 크기 방향에 반응한다"는 말은 가능하지만 "다양성 학습이 사용을 만든다"는 말은 불가능하다.
- 제거 실패는 문헌과 같은 방향의 한계다. INLP 10회(config q3_impl.inlp_iterations)로는 선형 정보도 다 못 지우고, LEACE 는 선형 정보만 지워 MLP 로 읽히는 정보가 남는다(log.md 2026-11-05 추정). lit_review 4.3의 skeptic 조건("제거 후 MLP probe 로 정보가 사라졌는지 확인, 안 사라졌으면 H2 결론 금지")이 그대로 작동한 사례다. LEACE 는 기억 기반 구현이며 원 논문/공식 코드와 대조하지 않았다는 점(README 7절)은 논문에 쓸 때 반드시 밝혀야 한다. 이 결과는 "정보가 비선형으로도 중복 저장되어 있다"는 약한 정보 이상이 못 되고, 그 자체도 추정이다(MLP probe 학습의 과적합 가능성 미배제; 제거 확인 MLP 의 R² 값은 results.json 에 요약되어 있지 않음, 미확인).

---

## 5. 4c 진단

- single_ind: verdict medium. 활성 이동 비율 3.2-4.9 (noise floor 대비), 다음 프레임 오차비 a060 1.05-1.09, a160 1.73-1.74 (모두 2 미만) (rj:decision_table[9]). 해석 규칙 문구: 층별 결과는 해석하되 낯선 입력의 영향 주의 문구를 붙임. 그래서 cond_a_vs_b:last_block 에 이 문구가 붙었다.
- multi_60_ind_160: verdict large(worst over seeds). 한 레벨의 층 최대 비율이 5.037/5.054/5.301(시드 0/1/2), 다른 레벨 4.49/4.56/5.02 (rj:decision_table[10]; 어느 레벨이 어느 값인지는 파일 구조상 두 번째 레벨 블록 순서로 추정). 오차비는 0.89-1.12.
- 해석:
  1. large 판정은 임계 5.0 을 0.04-0.30 넘긴 값이다. 경계 근처이므로 "large" 라는 단어보다 "(a) 와 거의 같은 이동량(3.2-4.9 대 4.5-5.3)"으로 쓰는 것이 정확하다. 임계를 조정하자는 것이 아니라, 이 판정이 임계 바로 위에서 나왔음을 보고에 적자는 것이다.
  2. (b) 에서 a060/a160 은 학습한 크기이므로 "낯선 입력"이 아니다. 이 규칙은 (a) 의 층별 결과 해석용이므로 (b) 의 large 는 (b) 결과를 막지 않는다(README 9절 4c, log.md 2026-11-05 항목 4번). 그런데 (b) 에서도 이동량이 (a) 와 비슷하거나 크다는 것은 이동량 지표가 "학습 여부"가 아니라 "입력 토큰(크기 다른 천)의 차이"를 주로 재고 있음을 시사한다(추정). 층별 비율이 layer 0 부터 3.6-3.8 로 이미 높고 깊이에 따라 소폭 증가할 뿐이다(rj:decision_table[9], 층 0-6의 ratio_to_noise_floor).
  3. 문헌: 이 진단과 직접 대응하는 선행 정량 기준을 찾지 못했다(공분산 이동/OOD 검출 문헌은 일반론으로만 알고 있고 이번 세션에 확인하지 않았다, 미확인). 임계 2배/5배는 사전 등록 값이며 문헌 근거가 아니다(hypotheses.md 확정 기록 7 A).

---

## 6. 기존 문헌/001 과의 종합 대조

| 주제 | 문헌/001 | 002 | 판단 |
|---|---|---|---|
| 인코더 크기 판독이 DINOv2 특이적인가 | 001: rvit/pixel 격자와 비슷 | 외삽 linear 에서 random-init 대비 +0.07, 보간 후반 창에서 +0.34 | 부분 일치. 창과 레벨에 따라 다름 |
| 크기 정보가 predictor 를 통과하는가 | prior.md: 학습 크기 분산 0이면 보존 압력 없음(추정) | (a) rollout step12 R² 0.08-0.11, (b) 0.50-0.53 | 방향 일치. 단 persistence baseline, 분포 이동 통제 없음 |
| rollout 길이에 따른 악화 | SG-JEPA, 001 ol 기울기 | R² step 1→12 단조 하락 | 일치 |
| 층별 보존 | 문헌에 DINO-WM 층별 probing 선행 없음(lit_review 5.1) | layer 0→6 변화 ±0.01 | 새 관찰이지만 잔차 구조상 예상 가능 |
| 사용 여부(개입) | Amnesic probing, Othello 개입 | steering pass, 제거 실패 | 반쪽 증거. 정렬 효과와 제거 실패로 "사용" 주장 불가 |
| attentive 가 linear 보다 높다 | Punzo 2026 요약 | 반대 | 불일치. 3값 라벨 + 외삽 때문일 가능성(추정) |

---

## 7. 대안 설명 (정리)

1. rollout 하락 = 분포 이동으로 인한 probe 전이 실패 (크기 정보 소실이 아님). trained 레벨에서도 하락, (b) trained step12 시드 간 0.05-0.54.
2. R² 는 평가 부분집합의 라벨 분산에 의존. interp 는 분산 작음, MAE 는 trained 와 동급. R² 비교가 레벨 집합에 걸쳐 일관되지 않음.
3. mask baseline 의 후반 붕괴 때문에 interp pass. 초기 프레임에서는 동급.
4. probe 가 그리퍼/팔 위치 같은 상관 단서를 읽음 (001 에서도 미검사).
5. steering 효과는 판독 정렬의 자명한 결과.
6. (a) 의 붕괴는 "크기 정보 소실"이 아니라 "ind 로 수렴한 표현을 주어진 3값 probe 로 읽을 때의 편향". 부호 있는 편향 미보고라 구분 불가.
7. 학습 재개 run 과 시드 3개 때문에 시드 간 변동이 과소/과대 추정될 수 있음(추정).

---

## 8. 다음 실험/분석 제안 (사전 등록 값은 바꾸지 않는다. 새 기록과 새 실험으로만)

저비용(CPU, Drive 캐시 필요, 사람 승인 불필요):
1. 레벨별 예측 크기 평균/기울기(편향 부호) 보고. 출력 공간 probe, 조건 (a),(b), 시점별. cells.npz 필요. "ind 로 수렴"과 "정보 소실"을 구분.
2. persistence(마지막 context 프레임 토큰 복사) 및 untrained(random-init) predictor baseline 의 같은 출력 공간 R². 가장 중요한 누락 baseline.
3. 대조 속성 rollout 감쇠: 크기와 무관한 속성(그리퍼 위치, 천 중심 위치 등)에 대해 같은 출력 공간 probe 감쇠를 측정해 "일반 rollout 품질 저하"와 "크기 특이 손실"을 분리(SG-JEPA excess error 식 해석).
4. steering 재해석: 이동한 판독이 아니라 "다음 latent 전체의 변화"(패치 평균 L2, 이미 코드에 있는 보조 지표가 results.json 에 없음)와 크기 방향 vs 다른 의미 있는 방향(그리퍼, 프레임 인덱스 방향)을 같은 분산으로 비교. 새 분석 스크립트로 별도 기록.
5. 그리퍼-마스킹 probe 또는 그리퍼 위치 baseline (001 보고서 8절 7번의 미이행 항목).
6. T 를 실제 레벨 간 log 면적 차에 대비해 척도화.

GPU/승인 필요:
7. 후보 H/003 (끝 크기 단일 학습, 혼합 학습)과 002 를 같은 출력 공간 probe 로 연결. 002 의 (b)−(a) 차이가 "크기 범위 폭"에 의한 것인지 구분.
8. 학습 시드 5 이상(001 stats 권고, 추정). 현재 CI 에 시드 변동 없음.
9. 크기만 다른 쌍(같은 행동/초기 배치) 생성으로 activation patching. README 가 인정한 한계(InD 크기 내 다른 episode 쌍으로 대체)를 해소. 시뮬레이터 재생성 필요, 비용 큼.
10. 제거 재시도: INLP 반복 수 증가, 비선형 제거 등. 이것은 사전 등록 값(`mlp_r2_after_max`, `inlp_iterations`) 변경이므로 별도 확정 기록과 새 실험 ID 로만 한다.

---

## 9. 논문에 쓸 수 있는 주장 / 쓸 수 없는 주장 (확정은 사람)

쓸 수 있음 (조건: 시드 3개, 단일 학습 분포 설정, num_hist=1, 재생성 데이터 IoU 0.68, 조건 명시):
- 후반(변형) 프레임에서 보간 레벨 a080/a125 의 크기를 DINOv2 패치 토큰 linear probe 가 mask 면적/bbox ridge baseline 과 random-init 인코더보다 높은 R² 로 복원했다(0.514 대 -0.042 / 0.174). 초기 평평한 프레임에서는 mask baseline 과 거의 같았다(0.869 대 0.772, 외삽은 0.901 대 0.903).
- 외삽 레벨에서는 사전 등록 마진(0.1)을 못 넘었다(linear +0.034, attentive -0.125). mask baseline 이 이미 0.825 였다.
- 마지막 predictor 블록에서는 선형 판독 R² 가 layer 0 과 거의 같다((b) +0.003~+0.013, (a) -0.006~+0.006). 조건 간 차이는 0.008 로 실질적으로 같다.
- 인코더 토큰으로 학습한 probe 를 rollout 예측 latent 에 적용했을 때 step 12 에서 R² 하락은 (b) 0.30(시드별 0.291/0.295/0.321), (a) 0.74/0.71/0.73. (a) 가 (b) 의 모든 시드보다 컸다(시드 3 대 3).
- ridge 크기 방향으로 steering 했을 때 같은 probe 로 읽은 예측 크기의 변화가 분산·이동 크기를 맞춘 무작위 100방향의 95 백분위를 6개 셀 모두에서 넘었고 α 에 단조였다. 단, 해석은 "개입 방향과 정렬된 판독이 반응한다"로 제한.
- 제거 실험은 6셀 모두 정보 제거에 실패해 결론이 없다.
- 동일 예산: 두 조건 모두 계산된 27855 step.

쓸 수 없음:
- "predictor 가 크기 정보를 쓴다/버린다" (steering 은 판독 정렬과 구분 안 됨, 제거 실패, (a) 확증 개입 없음).
- "다양한 크기 학습이 크기 보존/사용을 만든다"는 인과 주장 (조건 (a) 대 (b) 는 학습 에피소드 구성이 함께 바뀜, 시드 3개, 재개 run 혼재).
- "rollout 중 크기 정보가 사라진다" (probe 전이 실패와 구분 안 됨, persistence/대조 속성 baseline 없음).
- "H-a 지지" 일반화. 보간 후반 창 한정이며 외삽은 실패.
- "DINOv2 가 크기를 특이하게 보유". 외삽에서 random-init 대비 +0.07, 001 의 rvit/pixel 결과와 합치면 약함.
- "보존이 학습된 것" (last_block 은 잔차 구조로 자명).
- "ind 로 수렴한다" (편향 부호 미보고).
- 4c large 를 근거로 (b) 결과를 해석 불가로 선언하거나, medium 을 근거로 (a) 결과를 정상 해석이라고 쓰는 것. 규칙 문구와 대상이 어긋난다.
- DINO-WM 의 구조적 한계 일반화 (001 과 동일 사유).
- 신뢰구간 문구: 모든 CI 는 에피소드 부트스트랩이며 학습 시드 변동 미포함.

---

## 10. 확신도: 중간-낮음

- 높음: results.json 의 수치와 판정 행의 의미 해석(위 0절, 1~5절에서 직접 읽은 값). 동일 예산 일치. last_block 이 layer 0 과 거의 같다는 점. (a)(b) rollout 차이가 6개 시드에서 겹치지 않는다는 점.
- 중간: 문헌과의 방향 일치 판단. 근거 논문(DINO-WM, LeWM, SG-JEPA, POKEWORLD, Punzo)은 notes 의 2차 전사/요약이며 이번에 원문을 읽지 않았다.
- 낮음: (1) rollout 하락의 원인(분포 이동 대 정보 소실)과 steering 의 판독 정렬 효과는 q3.py/stats.py 코드와 원자료(cells.npz, q3 원자료)를 보지 못해 추정이다. (2) R² 분모 효과, attentive 외삽 한계는 추정이다. (3) 4c 에서 어느 레벨이 5.0 을 넘었는지는 파일 구조 읽기로 추정했다. (4) 인용한 일반 문헌(Hewitt & Liang, Xu 2021, Elhage 2021, Elazar 2021 등)은 기억 또는 타 분석가 파일 경유이며 서지 미확인.
- 한계: 이 머신에는 Q1~Q3 중간 파일이 없고 results.json 의 environment/git.dirty 는 학습 런타임이 아니라 aggregate 재실행 런타임 값이다(cuda_available false, git.dirty true, rj:environment, rj:git). 학습 환경의 재현 정보는 run_info.txt 등 Drive 쪽에서 확인해야 한다(미확인).
