# 004_depth_truncation 회의론 리뷰 (analyst-skeptic)

- 작성: 2026-11-10. 입력: experiments/004_depth_truncation/ (results.json, results/depth_r2.csv, results/attempts.json, results/colab/timing_main.json, logs/colab_20261109/*, README.md, CHANGES.md, config.yaml, run.py, src/exp004/*, colab/build_notebook.py), experiments/002_single_size_sweep/src/exp002/{evalgrid.py, model_io.py}, 002 config.yaml, notes/specs/exp4_5_spec.md 0-3절, notes/hypotheses.md 후보 I·확정 기록 9·12, notes/log.md 2026-11-09.
- 다른 분석가 파일은 읽지 않았다. 코드는 실행하지 않았다. "확인" = 파일/CSV 값으로 직접 확인한 것, "추정" = 읽고 추론한 것.
- 아래 수치는 모두 results.json 또는 results/depth_r2.csv 의 행(줄 번호 표시)에서 가져왔다. CSV 인용은 특별히 적지 않으면 probe seed 0 행이다.

## 요약

저장된 판정 `verdict_H_d = refuted (no_drop_ge_threshold_at_any_depth)` (results.json 300-303행)은 **사전 등록된 규칙을 그대로 적용하면 맞다.** 주 곡선에서 가장 큰 하락은 약 0.03 이라 임계값 0.2 와 거리가 멀다. 그래서 "running max 대비 0.2 정의라서 완만한 감소를 못 잡았다"는 의심은 **이번 데이터에는 해당하지 않는다**(확인, 아래 6절).

문제는 이 반증이 학습된 DINO-WM 에 대해 거의 아무것도 말해 주지 못한다는 점이다.
(1) random-init predictor 도 같은 평평한 곡선을 낸다. 그래서 "소실 없음"은 잔차 구조와 mean-pooled linear probe 의 성질로 설명되고, 학습된 표현의 성질이라는 근거가 없다(치명).
(2) InD/OoD 비교(`ind_ood_gap`, `curves_overlap`)와 `l_best` 는 split 마다 라벨 분산이 크게 다른 R² 를 서로 비교한다. MAE 로 보면 순서가 뒤집힌다(치명, OoD 관련 주장과 005 입력에 한정).
그 밖에 pilot 플래그와 판정 효력이 충돌하고, 시드 분산이 사실상 가짜이며, 재시도 상한을 넘겼는데 기록이 축소됐고, 코드 버전이 기록되지 않았다(중요).

---

## 치명

### C1. 검정에 특이성이 없다: random-init predictor 도 "소실 없음"이다 (확인)
- 주 곡선(linear, fit=new, late, ALL)에서 학습된 predictor 와 random-init predictor 의 값(train_seed 0):

  | m | real train / interp / extrap | random_init train / interp / extrap |
  |---|---|---|
  | 0 | 0.8336 / 0.5106 / 0.8524 (7746-7748행) | 0.8336 / 0.5106 / 0.8524 (5926-5928행) |
  | 3 | 0.8504 / 0.5361 / 0.8390 (9306-9308행) | 0.8350 / 0.5064 / 0.8542 (6706-6708행) |
  | 6 | 0.8537 / 0.5476 / 0.8487 (10866-10868행) | 0.8319 / 0.5084 / 0.8525 (7486-7488행) |

  train_seed 1 도 같다(real 21526-21528행, random_init 18146-18148행).
- 두 곡선 모두 깊이에 따라 떨어지지 않는다. 학습 전후 차이는 interp 에서 최대 약 0.04 이다. 따라서 "predictor 깊이에 따라 크기 정보가 소실되지 않는다"는 관찰은 **학습되지 않은 predictor 에서도 똑같이 나온다.** 이 검정은 H-d 가 거짓이든 참이든(학습이 크기를 지우든 말든) 그 차이를 구분할 수 있게 설계되지 않았다.
- 메커니즘(추정): deep.py 61-64행 `x = attn(x) + x; x = ff(x) + x` 구조에서 은닉 m 은 `x_0 + Σ f_j(x)` 이다. mean pooling 은 선형이므로 풀링된 특징에도 입력 z 의 풀링값이 그대로 들어 있다. ridge(GroupKFold 로 alpha 선택, 002 config 145행)는 더해진 항이 크기와 무관하면 그것을 대체로 무시할 수 있다. 소실이 관측되려면 블록이 크기 방향을 적극적으로 상쇄해야 하는데, random-init 블록은 당연히 그러지 않는다. 학습된 블록도 다음 z(크기 포함)를 예측해야 하므로 그럴 동기가 약하다. 즉 사전확률이 낮은 사건을 찾는 검정이었다.
- 인코더도 같다. random-init DINOv2 곡선 역시 평평하다(train 0.458→0.642→0.654→0.662→0.661→0.655→0.658; 336, 1116, 1896, 2676, 3456, 4236, 5016행).
- 결론에 미치는 영향: "refuted" 는 규칙상 유효하다. 다만 "DINO-WM 은 깊이 전체에서 크기 정보를 보존한다"는 식으로 읽으면 과장이다. random-init 대비 차이로 정의한 곡선이 없어서, 학습된 구조에 대한 주장은 현재 데이터로 뒷받침되지 않는다. README 10행의 "통제 포함"은 통제를 **판정에 쓰지 않았다**(depthlib.py 195행: 판정은 `prim["control"] = real` 곡선만 사용).

### C2. split 사이 R² 는 비교할 수 없다: 분모(라벨 분산)가 split 마다 최대 약 19배 다르다 (값은 확인, 분산은 명세 표로 계산)
- depthlib.py 41행: `level == ALL` 행의 R² 는 split 안의 라벨 분산을 분모로 쓴다. 라벨은 log 명목 면적비이고(evalgrid.py 27-28행), split 은 train={a060, ind, a160}, interp={a080, a125}, extrap={a026, a040, a200, a248} 이다(002 config 14-15행, 명세 22-32행).
- 명세 표의 면적비로 계산한 split 안 분산: train 약 0.160, interp 약 0.050, extrap 약 0.962(추정: 레벨당 test 100 episode 균등, n_episodes 300/200/400 과 일치).
- 실제 데이터(encoder k=12, linear, 4756-4758행):
  - R²: train 0.834 / interp 0.511 / extrap 0.852
  - MAE: train 0.128 / interp 0.119 / extrap 0.289

  **R² 로 보면 interp 가 가장 나쁘고 extrap 은 train 과 같다. MAE 로 보면 interp 는 train 과 같고 extrap 이 2배 이상 나쁘다.** 순서가 지표에 따라 뒤집힌다. 위 분산으로 환산한 MSE 는 train 약 0.027, interp 약 0.024, extrap 약 0.142 다(추정 계산).
- 영향을 받는 산출물:
  - `ind_ood_gap` (results.json 97-297행): interp gap 0.30-0.52 는 주로 분모 효과다. `curves_overlap_within_seed_sd: false` (310, 317행)를 "InD/OoD 곡선이 분리된다"는 증거로 읽으면 안 된다.
  - drop_threshold 0.2 의 민감도도 split 마다 다르다. R² 0.2 하락은 MSE 로 interp 약 0.010, train 약 0.032, extrap 약 0.19 에 해당한다. **H-d 가 가장 관심을 두는 외삽에서 검정이 가장 둔하다.**
  - `l_best` (20-34행)는 interp 와 extrap 의 R² 를 평균한다. predictor m=6 (0.7022) 과 encoder k=10 (0.6913) 의 차이는 0.011 로, predictor 의 seed tol(0.007-0.014, 243·293행)과 비슷하다. 이 값이 005 의 층 선택 입력으로 들어간다.
- 판정 사유 자체(no_drop)는 각 split 안의 비교라서 C2 의 영향을 받지 않는다. C2 는 OoD 관련 서술과 005 입력에 치명적이다.

---

## 중요

### I1. pilot: true 결과에 확정 판정을 붙였다 (확인)
- results.json 3행과 17행은 `"pilot": true` 이고, 329행 notes 는 "pilot=true marks a subset/partial run" 이라고 쓴다. 실제 실행은 전체 캐시(1800 episode, enc_cache 로그 4행), 학습 seed 3개(pred 로그 4행)인 본 실행이다. **결과 파일 안에 사실과 다른 서술이 남아 있다.**
- README 10행: "코드가 만드는 판정은 파일럿(`pilot: true`)에서는 점검용이다." 문서 규칙대로라면 저장된 "refuted" 는 점검용이고 확정 판정이 아니다. 저장 판정(H-d refuted)과 문서 규칙이 충돌한다.
- timing_main.json 81-98행: 004 aggregate 가 11:20:44 에 끝나고 005 plan 이 11:20:46 에 시작했다. 005 는 사람 검토 없이 pilot=true 로 표시된 004 summary 를 소비했다(CLAUDE.md "해석 후 /review-experiment" 이전). 005 쪽 결과를 해석할 때 이 의존을 고려해야 한다.

### I2. 임계값을 확정하지 않았고, 파일럿과 본 실행 사이에 결정 단계가 없었다 (확인. 이번 판정에 미친 영향은 작음)
- 명세 57행, hypotheses.md 301·304·311행은 "0.2 는 파일럿에서 확정"이라고 한다. 확정 기록은 없다(log.md 203행이 인정).
- 파일럿 aggregate 는 09:22:44 에 이미 `verdict_H_d=refuted` 를 냈고(aggregate_20261109-092240.log 5행), 본 실행 enc_cache 는 09:27:07 에 시작했다(enc_cache_20261109-092707.log 1행). 4분 남짓한 간격이다. 파일럿 결과를 본 상태에서 임계값을 확정하지 않고 넘어갔다.
- `overlap_sd_mult`, `overlap_depth_fraction`, `min_depths`, `primary` 도 "impl default"다(config.yaml 37-40행, CHANGES.md 13-14). 사람이 확정한 기록이 없다.
- 완화 요인: 주 곡선의 최대 하락이 약 0.03(6절)이라 0.05-0.2 범위의 어떤 값으로 바꿔도 linear 판정은 같다(확인). 그래서 영향은 절차상의 문제가 크고, 수치 결과에는 사소하다.

### I3. 재시도 상한(3회)을 넘겼고 log.md 기록이 축소됐다 (확인)
- 파일럿 aggregate 실패 기록: 09:15:15, 09:18:26, 09:19:20 (상한 도달) → 09:20:35 attempts_check 거부(aggregate_20261109-092035.log 4행) → (attempts 항목 삭제로 추정) 09:22:04, 09:22:26 에 다시 실패 → 09:22:40 성공. **실패 5회와 거부 1회**다.
- log.md 194행은 "attempts aggregate 1" 이라고만 적었다. CLAUDE.md "3회 실패하면 멈추고 원인을 보고" 를 위반했고 기록도 정확하지 않다.
- 수치 영향: 실패는 그림 생성 단계에서 났다(stages.py 397행). summary.json 은 그 전에 쓰인다(393행). 그래서 summary 값은 오염되지 않았다(추정). 다만 실패한 실행에서는 results.json 이 쓰이지 않았다.

### I4. "seed 평균 ± 표준편차"가 실제 반복 변동을 거의 담지 않는다 (확인)
- linear probe 의 probe seed 0-4 값이 같다. 예: encoder k=0 train R² 0.732886284192042 가 seed 0과 1에서 동일하다(76, 89행). 인코더는 학습 seed 와도 무관하다(train_seed NA).
- 그래서 인코더 tol 은 0.0 또는 1e-16 이다(results.json 107, 113, 125, 137행 등). 0 이 아닌 곳(k=4 0.0123, k=8 0.0079)은 ridge alpha 선택(GroupKFold 분할)의 seed 의존에서만 생긴다(추정). **인코더에서는 overlap 검정이 구조적으로 성립하지 않는다**(tol=0 이면 어떤 gap 도 "밖").
- predictor 의 sd 는 15개 값(학습 seed 3 x 동일에 가까운 probe seed 5)을 독립으로 보고 ddof=1 로 계산한다(depthlib.py 110-113행). 실제 독립 단위는 학습 seed 3개이므로 sd 가 과소추정된다. 군집 3개, 크기 5 기준으로 약 0.85배(추정 계산).
- 명세 0절 6 "seed 5회, 평균과 표준편차"는 형식상 채웠지만 내용상 반복이 아니다. 판정 사유(no_drop)는 sd 와 무관하므로 판정 영향은 없다. `curves_overlap` 필드는 의미가 없다.

### I5. 크기 판독은 저수준 면적 단서로 대부분 설명된다. 외삽의 "소실 없음"은 거의 자명하다 (확인)
- k=0(패치 임베딩 + LayerNorm) 만으로 train 0.733, extrap 0.788 이다(76, 78행).
- mask 면적/bbox baseline 의 extrap R² 는 0.817 이다(5798행). encoder k=12 의 0.852(4758행), predictor m=6 의 0.849(10868행)와 0.03-0.04 차이다. pixel baseline 은 extrap 0.628(5538행)이다.
- 확정 기록 12(hypotheses.md 361행)도 002 에서 "외삽 mask baseline 0.825" 를 확인했다.
- 의미: 천이 차지하는 패치 비율 같은 단서는 mean-pooled 특징에 선형으로 남는다(추정). 이런 단서는 모든 층에 남기 쉽다. "크기 정보" 가 깊이에서 살아남는다는 결과는 "면적 단서가 살아남는다"와 구별되지 않는다. 명세 0절 7 대로 "읽을 수 있다"까지만 말할 수 있고, 그 내용이 면적인지 크기 개념인지는 이 실험으로 알 수 없다.

### I6. 판정 범위가 좁다. 002 현상(rollout 수축)과 관련된 축은 판정에 들어가지 않았다 (확인 + 추정)
- 판정은 `primary = linear/new/real/late` 한 설정만 쓴다(results.json 55-60행, depthlib.py 195행). 은닉 probe (i)은 rollout 시점에서 계산하지 않는다(CHANGES.md 6). 명세 3.4-3 은 rollout t+1/t+4/t+8 에서도 재라고 한다.
- 보조 지표 (ii)(실험 1 probe 를 출력에 적용)에서 m=6, rollout t+8 의 R² 는 train 0.754, interp 0.443, extrap 0.672 다(13856-13858행). 같은 시점의 인코더 토큰 reference 는 0.890, 0.612, 0.878 이다(11581-11583행). **깊이가 아니라 rollout 시간축으로는 외삽에서 0.2 이상의 차이가 있다**(seed 0 한 행만 확인, seed 평균은 확인 안 함 → 추정).
- 따라서 "H-d refuted" 를 "모델이 크기를 잃지 않는다"로 일반화하면 안 된다. 반증된 것은 "teacher-forced 은닉 상태에서 깊이 방향으로 linear 판독이 0.2 이상 떨어진다"는 좁은 명제다.

### I7. 재현성 정보가 부족하다 (확인)
- results.json 에는 config sha256(4-5행)과 환경(331-363행)만 있고 **코드 commit 해시가 없다.** 실행 코드는 노트북에 내장된 스냅샷이다(README 82행). 지금 저장소에서는 figs.py, exp045_tools.py, 노트북, CHANGES.md, README.md 가 커밋되지 않은 수정 상태다(세션 시작 git status). 실행된 스냅샷과 저장소 코드가 같다는 증거가 없다.
- 002 코드를 "수정 없이 import" 하지만(base.py 11-16행), 002 코드는 그사이 바뀌었다(커밋 545168e 가 attempts_check 삭제, CHANGES.md 58행). 실행 시점의 002 코드 버전도 기록되지 않았다.
- `sklearn: null` (results.json 341행), conda 미사용(로그 3행 경고). Colab venv 예외는 002 확정 기록 10 에서 승인됐다. 004 에도 그대로 적용되는지는 기록이 없다(사소).
- 완화 요인: log.md 200행에서 parts 로부터 aggregate 를 다시 계산해 summary 가 일치함을 확인했다. 집계 단계의 재현성은 확인된 셈이다.

---

## 사소

- S1. 인코더 절단과 LayerNorm (확인, 버그 아님): deep.py 40-49행은 중간 블록 출력에 최종 `base.norm` 을 적용한다(`get_intermediate_layers(norm=True)` 와 같은 관례). 첫 배치 일치 검사 결과는 k=12 와 002 tokens_dino 0.017, get_intermediate_layers 0.003, random-init 0.002 로 모두 기준 0.05 이하다(enc_cache_20261109-092707.log 28-30행). 검사는 첫 배치 8 episode 에만 한다(stages.py 149행). 최종 norm 의 affine 이 12층 통계에 맞춰져 있어 얕은 층에서는 분포가 어긋난 정규화가 된다. linear ridge 에는 영향이 작을 것이다(추정).
- S2. predictor 조기 종료 (확인): m=L 경로와 model.predict 의 최대 차이가 0이다(pred 로그 31, 496, 961행). z 부분 384 선택(model_io.py 108-111행)은 m=0 의 R² 0.83357 이 encoder k=12 의 0.83358 과 일치해서(7746행과 4756행) 정렬이 맞다. m≥1 에서 앞 384 좌표가 z 부분공간이라는 보장은 없다(CHANGES.md 10). 그래도 판독이 유지되므로 결론을 바꾸지는 않는다.
- S3. random-init predictor 의 초기화 (추정 위험): deep.py 101-106행은 `reset_parameters` 가 있는 모듈만 초기화한다. 실제 모델에서 가중치가 바뀌었는지 확인하는 검사는 런타임에 없다(selftest.py 398행은 토이 모델만 검사). m≥1 의 값이 real 과 조금 다르므로(6186 대 8266행) 무언가는 바뀌었다. 모든 파라미터가 초기화됐는지는 확인되지 않았다.
- S4. 데이터 누수 (확인, 발견 없음): probe 학습과 평가는 cache 의 part=train/test 로 episode 단위로 나뉜다(evalgrid.py 18-21행, stages.py 56-61행 assert). 같은 episode 의 프레임은 학습에만 들어간다. ridge alpha 는 episode 그룹 CV 로 고른다(evalgrid.py 59행 ep_tr). interp/extrap 레벨은 probe 학습에 없다. 다만 probe 학습 라벨의 고유값이 3개뿐이라(학습 레벨 3) 사실상 3-클래스 회귀이고, "train" split 이라는 이름이 실제로는 학습 레벨의 test episode 를 뜻해 오해 소지가 있다.
- S5. reference 행(학습 seed 무관)이 depth_r2.csv 에 학습 seed 수만큼 세 번 복제돼 있다(11386, 22046, 32706행). 이 행들을 평균하는 소비자가 있으면 반복으로 잘못 셀 수 있다.
- S6. (ii) 출력 공간 R² 는 m<6 에서 크게 음수다(m=0 train -4.50, interp -17.4; 11711-11712행). 반면 Pearson 은 0.91, 0.75 다. pred_mean 이 약 -0.94 로 치우친 편향 탓이다. R² 가 정보량이 아니라 편향에 지배될 수 있음을 보여 주는 사례다. 판정에는 쓰지 않는다.
- S7. attentive probe 의 interp R² 는 -0.13~0.30 이라 바닥에 가깝다(207, 987, 1767행 등). 이 상태에서는 0.2 하락을 관측하기가 사실상 불가능하다. 주 곡선이 아니어서 판정에는 영향이 없다.

---

## 6. 사용자가 짚은 의심에 대한 개별 답

- running max 대비 0.2 정의라서 완만한 감소를 못 잡았나: **아니다(확인).** 주 곡선은 깊이에 따라 오르거나 평평하다. encoder interp 의 최대 하락은 k=10→12 의 0.5419→0.5106, 약 0.031 이다(3977, 4757행). predictor real interp 는 seed 0 에서 m=4→5 의 0.5451→0.5395(9827, 10347행)다. 완만한 감소가 있었는데 놓친 경우가 아니라, 감소 자체가 없다. attentive 는 predictor interp m=0→1 에서 0.253→0.172(7877, 8397행, seed 0), 약 0.08 이다. 임계값을 0.08 이하로 두면 attentive 에서는 소실이 잡힐 수 있다(추정, seed 평균 미확인).
- 인코더 절단·LayerNorm 버그, predictor 조기 종료 버그, z 384 선택 버그: 확인 범위에서 **발견 없음**(S1, S2).
- 데이터 누수: **발견 없음**(S4).
- random-init 통제: 구현은 됐지만 **판정에 쓰이지 않았고**, 결과는 real 과 같아서 검정의 특이성이 없음을 보여 준다(C1).
- 사전 등록 이탈: pilot 플래그(I1), 임계값 미확정(I2), impl default 판정 규칙(I2), 재시도 상한(I3). CHANGES.md 의 구현 선택 중 결과에 실질 영향이 있는 것은 split 별 R² 정의(15번, C2)와 overlap 규칙(14번, I4)이다. num_hist=1(CHANGES.md 4)이라 "마지막 context 프레임"은 유일한 프레임이고, predictor 의 시간 통합은 검증 대상에서 빠진다(추정).

## 7. 대안 설명

1. **잔차 경로 가설**: 깊이 방향으로 소실이 없는 것은 x + f(x) 구조와 선형 풀링 판독 때문이며, 학습과 무관하다. random-init 곡선이 이것과 맞는다(C1).
2. **면적 단서 가설**: 판독되는 것은 천이 차지하는 패치 비율(면적)이고, 이것은 k=0 부터 존재하며 지워질 이유가 없다. mask baseline 이 외삽에서 0.817 인 것이 이것과 맞는다(I5).
3. **분모 가설**: "OoD(interp) 에서 R² 가 낮다", "InD/OoD 곡선이 분리된다"는 split 별 라벨 분산 차이로 설명된다. MAE 에서는 interp 가 train 과 같다(C2).
4. 002 의 크기 수축이 있다면 그 위치는 깊이가 아니라 rollout 누적이나 디코더일 수 있다. (ii) rollout t+8 값이 이 방향을 시사한다(I6, 추정).

## 이 결과를 믿으려면 추가로 필요한 실험 (최대 3개)

1. **통제 대비 곡선과 단위를 통일한 재집계 (CPU, 기존 depth_r2.csv 만 사용)**: (a) real 에서 random-init 을 뺀 차이 곡선으로 소실을 정의하고, (b) split 마다 다른 분모 대신 MSE/MAE(라벨 단위) 또는 전체 평가 episode 의 공통 분산을 쓴 R² 로 InD/OoD 를 비교하고, (c) overlap 의 분산 단위를 학습 seed 3개로 둔다. 임계값은 결과를 보기 전에 기록한다.
2. **검출력 양성 대조(positive control)**: 블록 m 에서 크기(또는 면적) 방향을 인위로 제거하거나 상쇄하는 개입을 넣은 predictor, 또는 잔차 없는 변형을 같은 파이프라인에 통과시킨다. 이때 0.2 기준의 소실이 실제로 검출되는지 확인한다. 검출되지 않으면 이번 "refuted" 는 검정력이 없는 결과다.
3. **면적 통제 판독과 rollout 축 은닉 probe**: 라벨을 mask 면적으로 잔차화한 "면적 외 크기" 타깃, 또는 면적이 같고 크기가 다른 조건에서 판독을 보고, 은닉 probe (i)을 rollout t+4, t+8 의 은닉 상태에서도 잰다. 002 수축 현상과 직접 연결되는 축은 깊이가 아니라 이쪽이다.

## 확신도

- **중간~높음.**
- C1, C2, I1, I3, I4, I5 는 CSV 행과 로그 줄로 직접 확인한 사실에 근거한다. 이 사실들이 맞다는 확신은 높다.
- C1 의 메커니즘 설명(잔차와 선형 풀링 때문에 사전확률이 낮다), C2 의 분산 수치(명세 면적비로 계산, 레벨 균등 가정), I6 의 rollout 차이(seed 0 한 행)는 추정이 섞여 있다. 그래서 "판정이 무의미하다"는 강한 결론에는 중간 정도로만 확신한다.
- 코드를 실행하지 않았으므로 random-init 초기화의 완전성(S3)과, 실행된 노트북 스냅샷이 저장소 코드와 같은지(I7)는 확인하지 못했다.
- 반대 방향 근거도 있다. 사전 등록 규칙의 기계적 적용은 맞고, 결과는 임계값 선택(0.05-0.2)에 둔감하며, 누수와 구현 버그는 발견되지 않았다.
