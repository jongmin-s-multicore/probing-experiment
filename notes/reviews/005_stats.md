# 005_inductive_bias_probe 통계·데이터 리뷰 (analyst-stats)

작성: 2026-11-10. 입력은 results.json, results/{ib_scores,calibration,calibration_dev,law_fits}.csv, predictions/*.npz(25), tasks.npz, tasks_meta.json, config.yaml, src/exp005/{ibmetrics,verdicts}.py, notes/specs/exp4_5_spec.md 4절, notes/hypotheses.md 확정 기록 12, notes/log.md 2026-11-09. 다른 분석가 파일은 읽지 않았다.
계산 코드(전부 `experiments/005_inductive_bias_probe/analysis/`): `recompute_stats.py`(저장 예측에서 독립 재구현, 출력 `recompute_stdout.txt`, `recomputed_ib.csv`, `score_table.csv`, `hf_sensitivity.csv`, `law_table.csv`, `recomputed_calibration_dev.csv`, `recomputed_law.csv`), `extra_checks.py`(과제 bootstrap, 라벨 정확도, 모델별 H-f 격차; 출력 `extra_checks_stdout.txt`), `plot_stats.py`(`fig_dib_raw_by_model.png`). 실행: conda lab05 python + pandas 를 임시 경로에 설치한 PYTHONPATH(저장소 환경 변경 없음). 이하 수치의 출처는 이 파일들이며, "[csv]" 는 저장소 결과, "[재계산]" 은 위 스크립트다.

## 0. 요약 (먼저 읽을 것)

1. 판정 재현: ib_scores.csv 와 독립 재구현이 일치한다(700행, raw 최대 차 1.3e-7, score 최대 차 2.4e-6; law_fits 기울기 차 2e-16; calibration_dev 차 9e-17). 판정 3개(H-e not_evaluable, H-f supported, H-g refuted)는 규칙대로 재현된다. 코드 버그는 못 찾았다.
2. 그러나 D-IB 정규화 점수는 이 설정에서 거의 정보가 없다. 모든 모델(ours, random-init, 면적 head, 심지어 선형 oracle)의 D-IB raw 가 0.24~0.33 에 몰려 있고, 분모(MLP oracle raw 0.319)가 "천장"이 아니라 약한 oracle 의 값이다. 점수가 1.0~1.07 인 것은 "oracle 보다 크기를 잘 읽음"이 아니다 (3절).
3. H-f 는 "supported" 이지만 (a) 허용폭 대비 차이가 96%(sd_mult 1.93 이면 뒤집힘), (b) 같은 격차가 random-init 에서도 똑같이 나와(-0.088 vs ours -0.088) 모델 특이적이 아니고, (c) oracle 을 선형으로 바꾸면 refuted(차 0.204, 허용폭의 3.5배), (d) 대응 t 검정은 차이가 0 이 아니라고 말한다(p=0.001). 즉 "동등"을 지지하는 증거가 약하다 (4절).
4. H-e 는 not_evaluable 이지만, 포화 여부와 무관하게 5-B 이탈(ours 0.335 대 기준 0.1)과 "ours > area" 조건이 이미 실패한다. 판정이 아닌 수치로는 ours 는 면적 모델과 random 에 대해 어떤 지표에서도 의미 있게 높지 않다 (5절).
5. H-g refuted 는 slice 7개 중 6개가 범위 밖(기울기 0.25~0.54)으로 통계적으로 확실하다(범위 하한 0.8 대비 t 검정 p<0.003). 단 random-init 도 거의 같은 기울기(0.25~0.59)를 보이고, MLP oracle 도 1개 slice 는 0.69 이다 (6절).
6. 시드 5개는 head 학습 시드만 변한다. 과제(100), 평가 입력(900), 5-D 라벨 부분집합은 고정이라 sd 는 과제/입력 표본 변동을 포함하지 않는다. 과제 bootstrap 은 시드 sd 보다 넓다 (7절).
7. 데이터 불일치 한 건: tasks_meta.json n_eval=900 인데 config.yaml/명세는 2000 이다. 원인은 확인하지 못했다 (8절).

## 1. 재현 결과

| 항목 | 저장 결과 [csv/json] | 재계산 | 비고 |
|---|---|---|---|
| ib_scores raw / score | 700행 | 최대 차 1.3e-7 / 2.4e-6 | float32 반올림 수준 |
| score_alt | 동일 | 최대 차 2.0e-6 | |
| calibration_dev (value) | 90행 | 최대 차 9e-17 | 구간 동일 가중 정의 그대로 |
| law_fits 기울기 | 175행 | 최대 차 2.2e-16 | |
| H-f 값 | eq 0.92784, neq 1.01502, 차 0.08718, sd 0.04525/0.03058 | 같음 (ratio = 차 / (2 x 0.04525) = 0.963) | |
| H-e 면적 점수 | 1.06908 (D-IB, ood_any) | 1.0691 (sd 0.0286) | |

쌍 수(평가 입력 900개의 unordered 쌍, [재계산]): D-IB all 359,706; D-IB_area= all 27,401 (ind 2,429); D-IB_area≠ all 332,305; R-IB all 44,844 (ood_any 30,548 = ood_both 30,548: 같은 레벨 쌍에서 ood_any 는 곧 ood_both).

## 2. 시드 수와 변동원

- 시드 5 = head 초기화/학습 시드(config seeds.head). 평가 입력(seeds.eval 1000), 5-A 과제 라벨(2000)과 D_k, 5-D 라벨 부분집합(5000), 사인파 등은 고정이다. 따라서 보고된 sd 와 모든 "seed 편차 안/밖" 판정은 "고정된 과제 100개와 입력 900개에 대한 head 초기화 변동"만 반영한다. 학습 체크포인트도 1개(feat.train_seed 0)다.
- 5개 시드 sd 추정의 불확실성: n=5 에서 sd 의 95% 구간은 대략 0.6배~2.9배(카이제곱, 표준 공식)다. 따라서 `2 x max(sd)` 허용폭 자체가 매우 불안정하다.
- 과제 변동: ours D-IB(all) 의 과제별 raw 분산 0.0435 (sd 0.21) 대 과제 내 시드 분산 0.0019 [재계산 `extra_checks_stdout.txt`]. 변동의 대부분은 과제(라벨 배정)에서 온다. 과제를 100개 resample 한 bootstrap(시드 평균)이 시드 sd 보다 정직한 불확실성이다 (아래 3, 4절에 사용).
- 5-A 과제 라벨이 Bernoulli(0.5) 이고 D_k 가 레벨 3개(a060, ind, a160)뿐이라, 모델 입장에서 학습 가능한 정보는 라벨 3개(8가지 조합)뿐이다 (`tasks_meta.json pool_levels_used_5A`). 과제의 26% 에서 ours 의 예측이 900개 입력 전체에 대해 상수다 [재계산: 모든 입력 예측이 같은 과제 비율 0.26]. 이런 과제는 D-IB 0, R-IB 1 이다.

## 3. 정규화 점수가 1 을 넘는 이유와 분모 크기 (D-IB)

D-IB raw(all, 시드 평균 [재계산 score_table.csv]): ours 0.3220, random-init 0.3267, 면적 0.3324, oracle MLP 0.3188, oracle linear 0.2419, 무정보(iid) 0.5000, 상수 0.0000. 단위는 "다른 레벨 쌍 중 예측이 다른 쌍의 비율".

- 참 라벨을 그대로 쓰는 완벽한 state 모델의 D-IB raw 는 0.5023 이다(레벨당 Bernoulli(0.5), 같은 계산, 재계산). 즉 이론상 천장은 약 0.5 이고, iid 무정보 모델과 같다. 이진 라벨 + 무작위 배정에서는 완벽한 모델과 무작위 예측이 D-IB raw 에서 구분되지 않는다(CHANGES.md 8번도 같은 문제를 적음).
- MLP oracle(참 s 입력, hidden 5)은 이 천장에 못 미친다: 학습 레벨 입력에서는 정확도 0.9855 로 라벨을 맞히지만 보간/외삽 레벨에서는 0.496/0.459(우연 수준)이고 출력이 s 에 대해 매끄럽고 포화한다. 그 결과 raw 0.319. 선형 oracle 은 0.242.
- 확정 기록 12 이후 D-IB 점수 = raw / 0.3188 (상수 기준, 분모 raw_oracle - 0). 따라서 점수 > 1 은 "raw 가 MLP oracle 보다 크다"는 뜻이고, 그 차이는 ours 0.003, random 0.008, area 0.014 로 oracle 자신의 시드 sd(0.0080)와 같은 크기다. ours/oracle 비의 과제 bootstrap 95% 구간 [0.980, 1.042] ([재계산], 상수 기준 시드 평균).
- 결론: 점수 1.01~1.07 은 "oracle 과 구분되지 않음 + oracle 이 약함"으로 읽어야 한다. random-init 은 1.025, 면적 head 는 1.043, ours 는 1.011 이다. 같은 지표에서 random-init/면적과 ours 의 차이는 ours - random = -0.015 (과제 bootstrap 95% [-0.034, +0.003], 0 포함), ours - area = -0.033 ([-0.053, -0.014]; ours 가 더 낮다).
- 분모 크기(D-IB, const 기준; all / ood_any): 0.3188 / 0.3051; D-IB_area= 0.2119 / 0.1850; D-IB_area≠ 0.3276 / 0.3149. iid 기준이면 분모가 음수(-0.18 ~ -0.31)라 점수의 부호 해석이 뒤집힌다(무정보 점수 = 0, oracle = 1, 그러나 oracle raw < 무정보 raw). R-IB 는 상수 기준 분모가 0(상수 모델도 R-IB 1 = oracle 1)이라 iid 기준만 정의된다(분모 0.4999). 이 때문에 R-IB 의 score_alt 는 모두 NaN 이다.
- 같은 pair_group 별로 분모가 다르다(ind 그룹의 area=: 쌍 2,429개). ind 그룹 D-IB_area= 점수는 0.664 대 area≠ 0.803 으로 크게 갈리지만 쌍 수가 적고 분모가 다르다(해석 주의).
- 분모 선택 민감도 (D-IB all, 상수 기준): MLP oracle 분모 → ours 1.011; 선형 oracle 분모(0.2419) → ours 1.331, random 1.351, area 1.374 (`score_table.csv` 열 orclin_const_mean). 어느 쪽이든 ours 는 random/area 와 구분되지 않거나 낮다.

점수가 1 을 넘는 직접 원인(추정 아님, 산술): 분자 raw 가 분모 raw 보다 크기 때문이다. 왜 큰가에 대한 설명(oracle MLP 가 8가지 레벨 라벨 조합을 낮은 용량으로 표현하고 포화) 중 포화 부분은 oracle 정확도 수치와 일치하지만, oracle 의 각 시드별 학습 곡선은 확인하지 않았다(추정).

## 4. H-f (D-IB_area= 대 D-IB_area≠)

[재계산 `hf_sensitivity.csv`, `extra_checks_stdout.txt`] 시드별 area=: 0.891, 0.939, 0.983, 0.954, 0.873; area≠: 0.990, 1.009, 1.066, 1.017, 0.993. 5시드 모두 area= < area≠.

| 항목 | 값 |
|---|---|
| 평균 차 (eq - neq) | -0.0872, 시드 대응 차 95% t 구간 [-0.1158, -0.0586], dz = -3.8 |
| 허용폭 2 x max(sd) | 0.0905 → 비 0.963 (차이는 허용폭의 96%) |
| 뒤집히는 sd_mult | 1.927 (sd_mult 1.92 이하면 refuted: 1.0 refuted, 1.5 refuted, 2.0 supported) |
| 대응 t 검정 (차=0) | p = 0.0011 ; Welch p = 0.0091 |
| 과제 bootstrap 95% (시드 평균, 100 과제 resample) | 차 -0.0879, [-0.1184, -0.0580] |

해석상 중요한 점:
1. "supported"(동등) 는 허용폭 비교로 나온 것이고, 동등성 검정이 아니다. 대응 t 검정은 차이가 0 이라는 가설을 기각한다(p=0.001). 허용폭은 시드 sd 에서 나오며 sd 추정이 n=5 라 불안정하다 (2절). 이 결과를 "차이가 없다"로 쓰면 과하다. "허용폭(제안값) 안쪽이지만 경계 근처이고 차이는 일관되게 한쪽 방향"이 맞다.
2. 격차는 모델 특이적이 아니다 (같은 점수 정의, MLP oracle 분모, 과제 bootstrap): ours -0.0879 [-0.118, -0.058], random-init -0.0884 [-0.124, -0.054], 면적 head -0.1428 [-0.169, -0.116], 선형 oracle -0.1601 [-0.181, -0.129]. 즉 random-init 이 ours 와 같고, "참 s 를 쓰는 선형 oracle"도 area= < area≠ 이다. 같은 q 인 쌍은 인접 레벨 쌍일 가능성이 높아(CHANGES.md 12번이 같은 교란을 적음) 크기에 매끄러운 모든 함수의 D-IB_area= 가 낮아진다. 그리고 MLP oracle 로 정규화하면 oracle 은 정의상 격차 0 이 되어, 이 교란이 oracle 쪽에서는 지워지고 나머지에서는 남는다. 따라서 H-f 지표는 "면적 휴리스틱 사용"을 모델 간 비교 없이 판단하기 어렵다. 모델 간 비교로는 ours(-0.088) 는 면적 head(-0.143)보다 격차가 작고 random 과는 같다 (추정: 정보 부족, 격차 크기만으로 기전을 구분할 수 없음).
3. score_alt(D-IB 에서 iid 기준) 병기: area= 1.0547, area≠ 0.9744, 차 0.0804, 허용폭 0.1068 → 비 0.753, 여전히 supported. 부호는 반대(eq > neq)다. 판정은 바뀌지 않지만 부호가 바뀌므로 "격차 방향"은 정규화 기준에 의존한다. 이는 3절의 음수 분모 때문이다.
4. oracle 을 선형으로 바꾸면(상수 기준): area= 1.5273, area≠ 1.3230, 차 0.2043, 허용폭 0.0583, 비 3.50 → refuted (재계산 `hf_sensitivity.csv`, 열 score_orclin_const). oracle 선택 하나로 H-f 판정이 뒤집힌다. 사전에 정한 oracle 은 mlp(config oracle.primary)이므로 보고된 판정은 규칙대로이지만, 이 민감도는 보고서에 함께 써야 한다.
5. 다른 쌍 그룹: ood_any 에서는 차 0.047 (허용폭 0.098, 비 0.48, supported), ood_both 0.028 (비 0.24), ind 0.139 (비 2.17 refuted; 쌍 2,429개). 판정은 config 의 pair_group=all 하나만 규칙이다.

## 5. H-e (수치, 판정 아님)

규칙은 첫 지표 R-IB 를 통과한 뒤 D-IB 에서 area 점수 1.069 >= 0.95 로 not_evaluable. 포화 임계 0.95 는 impl default 이고, D-IB 점수가 1 을 넘는 원인이 oracle 약함(3절)이므로 "면적 baseline 포화"라는 사유 이름은 이 지표에서는 부정확하다(점수가 천장을 넘은 것이지 면적 head 가 이미 완벽한 것이 아니다. 추정).

ood_any, 시드 5개 평균(sd) [재계산, 확정 기록 12 의 기준: R-IB iid, D-IB const]:

| 지표 | ours | random-init | area | ours - random | 2 x max(sd) | 비 | ours - area |
|---|---|---|---|---|---|---|---|
| R-IB | 0.7717 (0.0073) | 0.7400 (0.0053) | 0.7920 (0.0021) | +0.0318 | 0.0147 | 2.17 | -0.0203 |
| D-IB | 1.0404 (0.0324) | 1.0596 (0.0333) | 1.0691 (0.0286) | -0.0192 | 0.0667 | 0.29 | -0.0287 |
| D-IB (score_alt, iid 기준) | 0.9390 (0.0444) | 0.9091 (0.0476) | 0.8939 (0.0370) | +0.0299 | 0.0952 | 0.31 | +0.0451 |

- 포화가 아니었다고 가정한 H-e 조건: (i) ours > area: R-IB 거짓(0.772 < 0.792), D-IB 거짓(1.040 < 1.069). (ii) ours 와 random 의 차가 2 x max(sd) 보다 큼: R-IB 참(비 2.17), D-IB 거짓(0.29). (iii) 5-B extrap_any 이탈 0.1 이하: 거짓 (ours 0.3347 sd 0.0182 [재계산; csv 와 일치], random 0.3342, area 0.3116, 선형 oracle 0.2264). 이 가정에서도 H-e 는 refuted 였을 것이다(점수 정의별로 수치만 보면). score_alt 기준 D-IB 에서는 ours > area(+0.045)가 되지만 R-IB 의 alt 가 NaN 이라 H-e 는 어차피 not_evaluable 이다.
- R-IB 의 ours - random 차이(0.032)는 과제 bootstrap 95% [0.018, 0.047] 로 0 을 제외한다(정규화 분모 0.5 가정, 재계산). 효과는 작다(R-IB 점수 0.74 → 0.77). 면적 head 가 ours 보다 R-IB 가 높다(차 -0.020, 구간 [-0.034, -0.007]).
- 5-B calibration (ood 외삽 쌍, 시드 평균 [재계산 + calibration_dev.csv]): 정규화 평균 절대 이탈 ours 0.335, random 0.334, 면적 0.312, 선형 oracle 0.226, "truth"(참 사인파를 모델로 쓴 행) 0.328, 임계 0.1. 모델의 평균 쌍별 거리는 oracle 의 약 1/4 (0.089 대 0.362, extrap_any)로 모든 head 가 예측을 평균 쪽으로 수축한다. truth 행이 oracle MLP 에 대해 0.328 만큼 이탈하므로, x 축으로 쓴 MLP oracle 자체가 참 함수와 크게 다르다(oracle 이 기준으로서 부정확하다). 45도선 이탈 0.1 이라는 임계는 이 기준에서는 참 함수로도 달성되지 않는다 (즉 기준이 이 설정에서 과하다, 추정). InD 쌍(group ind)은 ours 0.038, random 0.043, 면적 0.037 로 모두 0.1 이하이고 서로 구분되지 않는다.

## 6. H-g (5-D 기울기)

[재계산 `law_table.csv`; csv 와 일치] ours 기울기 평균 (시드 sd, n=5, R2 평균), 하한 0.8 에 대한 일표본 t 검정 p, 평균의 95% t 구간:

| slice | 기울기 | sd | R2 | 0.8 대비 p | 95% 구간 |
|---|---|---|---|---|---|
| interp | 0.936 | 0.087 | 0.56 | 0.026 (범위 안, 1.0 과는 p=0.17) | [0.827, 1.044] |
| extrap | 0.543 | 0.060 | 0.96 | 0.0007 | [0.468, 0.618] |
| early | 0.607 | 0.057 | 0.89 | 0.0017 | [0.535, 0.678] |
| late | 0.536 | 0.051 | 0.79 | 0.0003 | [0.473, 0.599] |
| labels_a060+a160 | 0.541 | 0.0056 | 0.68 | <1e-4 | [0.534, 0.548] |
| labels_a060+ind | 0.304 | 0.027 | 0.71 | <1e-4 | [0.271, 0.337] |
| labels_ind+a160 | 0.252 | 0.024 | 0.68 | <1e-4 | [0.222, 0.282] |

- 7개 중 6개가 [0.8, 1.2] 밖이고 구간도 겹치지 않는다. 시드 5개로도 "refuted"는 통계적으로 확실하다. interp 는 s 범위가 좁고(±0.22, 입력 199개) R2 0.56 이라 기울기가 불안정하다(시드 sd 0.087).
- 조합별 기울기 0.25~0.54 는 라벨을 뽑는 레벨 조합에 따라 2배 이상 달라진다 (0.252 대 0.541). 시드 sd 는 작아(0.006~0.027) 조합 차이가 시드 변동으로 설명되지 않는다. 단 라벨 부분집합(72개)은 조합마다 하나뿐이고 시드 간에 고정이다(ids_d 레벨 수: a060+ind = 46+26, ind+a160 = 36+36, a060+a160 = 47+25). 라벨 표본 변동은 측정되지 않았다. 학습 레벨 범위가 s ∈ [-0.51, 0.47] 인데 평가는 [-1.35, 0.91] 이라 외삽 구간에서 기울기가 1 보다 작은 것(수축)은 평균 쪽 수축과 일치한다 (추정).
- 기준 모델: random-init 의 기울기도 비슷하다 (early 0.586, extrap 0.521, late 0.518, 조합 0.253/0.274/0.533). 따라서 이 기울기는 사전학습된 표현이 아니라 head 학습 일정/라벨 수/외삽 구조의 성질일 수 있다 (추정). MLP oracle 도 0.69(ind+a160, sd 0.27)~1.005이고 6/7 slice 가 범위 안, 선형 oracle 은 전부 1.000. 즉 판정 규칙은 oracle MLP 에도 한 slice 에서 실패할 만큼 엄격하다.

## 7. 우연/시드 운으로 설명될 가능성

- H-g: 가능성 낮다. 효과가 sd 의 10배 이상이다 (예: extrap 0.543 대 0.8, 차 0.26 / sd 0.06).
- H-f: 중간 이상. 허용폭 대비 96%, 뒤집히는 sd_mult 1.93, sd 가 5시드 추정. 같은 격차가 random-init 에서도 나온다.
- H-e 관련 수치: ours 와 random 의 D-IB 차이는 시드 sd 안(비 0.29), 과제 bootstrap 구간이 0 을 포함한다. 시드 운과 구분되지 않는다. R-IB 차이는 효과가 작지만 일관된다.
- 다중 비교: 판정 3개 x 쌍 그룹 4 x 지표 5 = 보고된 행이 많고 H-f 만 단일 지정 그룹으로 판정한다. 그룹/oracle/기준을 바꾼 민감도 행은 탐색적 확인이며 p 값은 보정하지 않았다.

## 8. 데이터/설정 점검

- tasks_meta.json 의 n_eval = 900 (초기 450 + 후반 450, 레벨별 86~109). config.yaml eval.n_inputs = 2000, 명세도 2000, 셀프테스트도 2000 을 확인한다. 저장소 코드(`tasks.py eval_inputs`)는 후보 수가 부족하면 ValueError 를 내므로 2000 을 요청하고 900 이 나올 수 없다. 또 tasks_meta 의 config_sha256(2fd84b30...)은 현재 config.yaml 의 sha256(4a6bba0b...)과 다르다. 이 sha 가 무엇을 해시하는지(override 포함 여부)는 확인하지 못했다. 원인 미확인: 실제 본 실행 설정이 저장소 config 와 다를 가능성이 있다(추정). 평가 입력 900개 기준으로 모든 쌍 수와 sd 가 계산되었다. 보고서에 반드시 "n_eval = 900" 임을 쓰고 원인을 experimenter 가 확인해야 한다.
- pilot: true 가 본 실행 결과에 남아 있다(사용자 확인된 사항). 임계값(0.1, 0.8~1.2, 2.0, 0.95)은 제안값/impl default 이며 확정 기록이 없으므로 이 리뷰의 모든 "판정 민감도"는 임계값 선택에 종속된다.
- 5-A 의 D_k 가 학습 레벨 3개에서만 뽑힌다 (CHANGES.md 7번, `pool_levels_used_5A`). 그래서 5-A 의 OoD 쌍 점수는 "처음 보는 레벨의 라벨 외삽"이고, 라벨이 레벨마다 무작위이므로 학습하지 않은 레벨의 라벨은 원리적으로 예측 불가다 (정확도 0.46~0.50, 우연 수준). 이 설정에서 D-IB/R-IB 의 OoD 점수는 모델의 크기 인식을 재는 지표로는 신호 대 잡음이 낮다 (추정).

## 9. 정리: 권고

- H-e: not_evaluable 은 규칙 적용 결과로 맞지만 사유 이름(area_baseline_saturated)은 "점수가 천장을 넘음"으로 쓰고, 분모(MLP oracle D-IB raw 0.319, 천장 0.5)를 같이 쓸 것. 수치상 ours 는 random/area 보다 D-IB 가 낫지 않다.
- H-f: "supported" 로 쓰되 (i) 경계(비 0.96), (ii) 대응 t p=0.001, (iii) random-init 도 같은 격차, (iv) 선형 oracle 로는 refuted 를 함께 보고할 것. 동등성이 확인된 것은 아니다.
- H-g: refuted, 통계적으로 견고. 단 random-init 과 구분되지 않는 기울기이므로 "표현이 크기 법칙을 안 따른다"가 아니라 "이 head/라벨 조건에서 단일 법칙(기울기 1)이 안 나온다"로 한정.
- 다음 확인 후보(실행하지 않음): D-IB 에 라벨 용량이 충분한 oracle(레벨 룩업 등) 사용, 연속 s 과제 기반 지표, 5-A 에서 모든 레벨을 D_k 에 포함, 과제/라벨 부분집합 시드 반복.

## 10. 확신도

- 재현(수치 일치, H-f/H-g 값): 높음. 독립 재구현이 csv 와 1e-6 이내로 일치.
- H-g refuted: 높음 (효과 크고 시드 5개로도 구간이 분리됨).
- H-f supported 의 "실제 의미": 낮음. 경계 근처(비 0.96), sd 추정 n=5, oracle 선택에 민감, random 과 같은 격차, 교란(인접 레벨) 가능성.
- H-e 와 D-IB 점수 해석: 중간. 분모가 약한 oracle 이라는 점은 수치로 확인했지만 oracle MLP 의 학습 상태와 원문 App. B 정규화식은 확인하지 못했다(원문 미확인).
- 전체 결론의 일반화: 낮음. 시드는 head 만, 체크포인트 1개, 과제/입력 고정, n_eval 900 불일치 미해명, pilot 플래그, 임계값 미확정.
