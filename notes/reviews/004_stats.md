# 004_depth_truncation 통계·데이터 리뷰 (analyst-stats)

작성: 2026-11-10. 입력은 experiments/004_depth_truncation/results/{depth_r2.csv, summary.json, next_z_mse.csv}, results.json, config.yaml, src/exp004/depthlib.py, README.md.
재계산 코드: `experiments/004_depth_truncation/analysis/recompute_stats.py` (출력 `analysis/stats_out.txt`, 아래 "§"는 그 출력의 절 번호), 그림 `analysis/plot_curves.py` -> `analysis/fig_primary_vs_controls.png`. 환경 lab07 (numpy 만; pandas 없음).
이 문서의 임계값 민감도는 판정 변경 제안이 아니라 민감도 보고다. 다른 분석가의 파일은 읽지 않았다.

## 1. 요약 (결론은 약하게)
- 판정 재현: depth_r2.csv 에서 주 곡선(linear, new, real, late, level=ALL)을 직접 다시 계산해 summary.json 의 ind_ood_gap 14+14 행 전부와 gap, tol, inside 판정이 일치한다(§2, mismatch 0). 소실 깊이는 6개 (모듈 x split) 모두 없음. 곡선 최대 하락(직전 최고값 대비)은 인코더 train 0.012, interp 0.033, extrap 0.004, predictor train 0, interp 0.003, extrap 0.010 (§1). 임계값 0.2 의 6배 이상 아래이므로 refuted(no_drop_ge_threshold_at_any_depth) 는 재현된다.
- 이 "하락 없음"은 시드 운으로 설명되지 않는다. 시드 sd 가 0.0~0.014 이고, 개별 (학습 시드 x probe 시드) 시계열 중 최대하락이 0.2 를 넘는 것은 0개다 (§12, 인코더 5개, predictor 15개 시계열 x 3 split; 최대 0.04).
- 그러나 "refuted"의 의미는 좁다. 주 곡선이 깊이에 따라 거의 단조 증가하고, predictor 는 random-init 과 비교하면 실질 변화량이 작다(아래 4절). 즉 "이 측정 설계(잔차 스트림 hidden probe, 분할 풀링 R², 0.2 절대하락)에서는 소실이 관측되지 않는다"이지 "크기 정보가 모든 깊이에서 유지된다는 강한 증거"는 아니다(추정).
- 시드 수는 명목과 실효가 다르다(2절). 인코더 쪽 "seed sd"는 사실상 0 이라 ind_ood_gap 의 tol 이 의미가 없다. 이 판정 경로는 최종 판정에는 영향이 없다(첫 분기 refuted 가 먼저 걸림).
- 확신도: 중간 (판정 재현과 "소실 없음"은 높음, 해석 일반화는 낮음; 맨 끝 참조).

## 2. 시드 구조와 분산 (실효 n)
출처 §2, §3, §3b. 코드 depth_r2.csv 의 train_seed, seed 열.
- 인코더(4-A): 동결 인코더라 train_seed=NA, 시드는 probe seed 5개뿐. linear(ridge, 닫힌 형태로 추정) 에서는 5개 값이 사실상 동일하다. 깊이 0, 2, 6, 10 에서 5개 값이 비트 단위로 같거나 1e-16 차이(sd 0 또는 1.2e-16, 2.5e-16). 깊이 4, 8, 12 에서는 5개 값이 2종류로 갈린다 (예: 깊이 4 train 0.82603 x2, 0.82530 x3; 추정: probe 내부 CV fold 분할 차이). 그래서 linear 인코더 28개 (깊이 x split) 셀 중 16개가 sd<1e-9 (§3b). attentive 는 0/28.
- ind_ood_gap 의 tol=0 또는 1.2e-16 이유: tol = 2*max(sd_train, sd_ood) 이고 위 sd 가 0 이기 때문. 인코더 interp 깊이 0, 2, 6, 10, extrap 깊이 0, 2, 6, 10 이 그렇다(§2). 이 경우 inside_seed_sd 는 gap 이 정확히 0 이 아닌 한 항상 false 가 되어, "겹침 아님"이 자동으로 나온다. 즉 이 검정은 인코더에서 정보가 없다(자명하게 분리). 구현 버그가 아니라 입력(probe 가 결정적)의 성질이다.
- predictor(4-B): 깊이 0 은 학습된 predictor 와 무관한(입력 z, 동결 인코더 출력) 값이라 3 학습 시드가 동일하다(§3, 깊이 0 between-trainseed sd 0.0). 깊이 1 이상에서는 학습 시드 간 변동이 지배적(예: interp 깊이 2 between-trainseed sd 0.0164, probe 내부 sd 0.0). 따라서 독립 표본은 15 가 아니라 학습 시드 3 개다. summary 의 sd(ddof=1, n=15)는 시드 3개에서 나온 값이고 자유도가 2 라 신뢰구간이 매우 넓다. 시드 3개로는 sd 추정 자체가 불안정(추정).
- 개별 시드별 시계열의 최대 하락도 모두 0.2 미만(§12).

## 3. 주 곡선 재계산 (linear, new, real, late, level=ALL)
출처 §1. 평균 R², 깊이 순서.
| 모듈 | split | 깊이별 평균 R² | 최대 하락 |
|---|---|---|---|
| encoder k=0,2,4,6,8,10,12 | train | .733 .793 .826 .840 .837 .846 .834 | 0.012 |
| | interp | .217 .428 .466 .490 .534 .542 .509 | 0.033 |
| | extrap | .788 .824 .822 .840 .836 .841 .852 | 0.004 |
| predictor m=0..6 | train | .834 .842 .845 .847 .851 .856 .859 | 0 |
| | interp | .509 .519 .531 .537 .546 .543 .553 | 0.003 |
| | extrap | .852 .845 .842 .843 .844 .847 .852 | 0.010 |
- 인코더 L_best 는 summary 대로 predictor 깊이 6 (interp+extrap 평균 0.702), 인코더는 깊이 10(0.691). 인코더 12 번째 층에서 interp 0.542 -> 0.509 (-0.033, probe-seed sd 0.004)는 통계적으로 분명하나 크기는 임계값 0.2 의 1/6.
- 해석 주의 (데이터 정의): README/depthlib 에 따르면 level=ALL 행의 R² 는 해당 split 안에서의 R²(split 내 y 분산 기준)이다. interp 는 두 레벨(a080, a125)뿐이라 y 범위가 좁아 R² 가 낮다(0.22~0.55). 같은 모델의 레벨 행(pooled-variance R², 전체 평가 분산 기준)은 a080 0.95~0.97, a125 0.89~0.95 (§11). 즉 "interp R² 가 train 보다 0.3 낮다"는 일반화 실패가 아니라 분산 분모 차이가 주 원인일 수 있다(추정). split 간 R² 절대값이나 gap 비교(ind_ood_gap)는 같은 척도가 아니다. 반대로 extrap 은 4개 레벨(a026~a248)로 y 범위가 넓어 단순 면적 baseline 만으로도 R²=0.817 이다.
- 레벨별 곡선(§11)에서도 최대 하락은 a026 predictor 0.056, a026 encoder 0.051 (이 레벨은 R² 자체가 0.13~0.19 로 바닥).

## 4. 대조군 대비 효과 (linear, new, late; §4, §5)
random-init, 라벨 셔플:
- 라벨 셔플은 모든 깊이에서 R² ≈ 0 (-0.03~0.003). probe 파이프라인 누수 없음.
- 인코더 random-init 은 사전학습 효과를 보여준다. k=12: train 0.834 vs 0.658, interp 0.509 vs 0.207, extrap 0.852 vs 0.768. 단 random-init 도 k≥2 에서 0.64~0.77 로 높아(저수준 특징만으로 면적을 읽음) R² 의 큰 부분이 크기 정보의 사전학습 의존이 아니다.
- predictor random-init 곡선은 깊이 0 값에서 거의 평탄(예: extrap 0.852 -> 0.852, train 0.834 -> 0.835). 학습된 predictor 의 깊이 0 대비 증가는 train +0.025, interp +0.044 (95% bootstrap CI [0.040, 0.048], 학습시드 3 x probe 시드 5 재표집, 시드 3개라 CI 는 좁게 나옴으로 신뢰하지 말 것), extrap +0.000 (CI [-0.001, 0.002]) (§13). 즉 predictor 의 hidden 층은 입력 z 에 이미 있는 크기 정보를 거의 바꾸지 않는다. 잔차 스트림 구조상 정보가 보존되기 쉬워서 "소실 없음"은 설계상 예상 가능한 결과일 수 있다(추정).
mask 면적 baseline 과 pixel baseline (late):
| split | mask_area_bbox | pixel(linear) | predictor 깊이0 / 깊이6 | encoder k=12 |
|---|---|---|---|---|
| train | 0.588 | 0.373 | 0.834 / 0.859 | 0.834 |
| interp | -0.000 | -0.252 | 0.509 / 0.553 | 0.509 |
| extrap | 0.817 | 0.628 | 0.852 / 0.852 | 0.852 |
- extrap 에서의 mask baseline 대비 우위: late 에서 predictor 는 +0.025~+0.035 (모든 깊이 양수), 인코더는 k=0 에서 -0.029 (mask 보다 낮음), k=2,4 에서 +0.005, k≥6 에서 +0.02~+0.035 (§5). 우위는 존재하나 R² 단위로 작다(최대 +0.035). early 시점 baseline 은 mask extrap 0.889 인데 predictor early extrap 은 0.895~0.913(+0.006~+0.024), 인코더 early extrap 은 k≤6 에서 0.835~0.877 로 mask 아래, k≥8 에서 0.899~0.905 로 약간 위 (§4, §7). 따라서 extrap 곡선에서 "크기 정보가 유지된다"는 결론은 면적 baseline 이 이미 대부분 설명하는 범위 안에 있다.
- 표준화 효과 크기(Cohen d): mask baseline 의 시드 sd 가 0.0002 수준(결정적)이라 d 가 수백으로 폭주해 정보가 없다(§5). 차이 크기(R² 단위)로 보고한다.
- 개별 레벨 a026(가장 작은 외삽) 은 R² 0.13~0.19 로 낮다(§11). 이는 split 평균에 가려진다.

## 5. attentive vs linear, early vs late
출처 §6, §7.
- attentive 는 linear 보다 최대하락이 큰 쪽이다: predictor interp 0.079, 인코더 interp 0.069, 인코더 train 0.022, predictor extrap 0.018. 모두 0.2 미만. 그러나 attentive 의 interp 값이 0.10~0.23 (predictor), -0.18~0.23 (encoder) 로 낮고 시드 변동이 크다(참고: pixel baseline attentive interp 의 probe-seed sd 가 0.076, 출처 §4). attentive 의 깊이별 sd 는 이 스크립트가 따로 출력하지 않았다. attentive 곡선은 신뢰구간이 넓고 비단조적(predictor interp 0.156, 0.183, 0.104, 0.154...)이다.
- early 시점은 전체적으로 late 보다 R² 가 높다(예: 인코더 train 0.936~0.965 vs 0.733~0.846). 모든 split 에서 최대하락 <= 0.014 (인코더 interp early 0.0144, predictor early extrap 0.004). 소실 징후 없음.

## 6. predictor 보조 지표 (ii), (iii)
출처 §8, §9 (next_z_mse.csv, level=ALL, 학습 시드 3 평균).
- (iii) 다음 z MSE 는 깊이에 대해 단조 감소: real late train 1.263(m=0) -> 0.265(m=6), extrap 1.265 -> 0.284. rollout t+8 은 6.20 -> 0.49 (train), 6.28 -> 0.58 (extrap). 조기 종료 경로는 학습 때 없던 경로이므로(명세 3.2) 얕은 깊이의 나쁜 MSE 는 소실 증거가 아니다. 소실이 아니라 "m 이 커질수록 예측이 좋아짐"만 보인다.
- (ii) 실험 1 probe 를 조기 종료 예측에 그대로 적용한 값은 얕은 깊이에서 매우 낮고(linear late train -4.13, interp -16.2) 깊이가 커질수록 올라 m=6 에서 train 0.82, interp 0.51, extrap 0.81 이다. 즉 (ii) 는 소실이 아니라 반대 방향(조기 종료 때문에 출력 공간과 어긋남)의 현상이다. 임계값을 (ii) 에 기계적으로 적용하면 rollout t+4/t+8 linear 에서 깊이 1~2 의 "하락"(예: t+4 train 깊이 1, -1.62 -> -3.99 후 상승)이 잡히지만, 시작값이 이미 음수라 의미가 없다(§8). 또한 (ii) linear extrap 은 m=5 에서 0.864 -> m=6 에서 0.811 (late), rollout t+8 에서 0.832 -> 0.675 로 마지막 층에서 하락한다(max 하락 0.053 late, 0.196 rollout t+8; 임계값 0.2 에 근접하나 미만). 시드 3개 평균이며 (ii) 는 판정 경로 밖. 마지막 층의 이 하락은 확정된 소실이 아니고, 출력 projection 이 마지막에만 쓰인 효과일 수 있다(추정).
- attentive (ii) train rollout t+8 에서 m=4 0.814 -> m=6 0.517 (하락 0.297 >= 0.2). 이 값도 주 지표가 아니어서 판정에 쓰이지 않았다. 사전 고정 주 곡선 밖이므로 post-hoc 로 보고만 한다.

## 7. 임계값 민감도 (판정 변경 제안 아님, §10)
- 주 곡선(linear/new/real/late)은 임계값을 0.30, 0.20, 0.15, 0.10, 0.05 중 어느 값으로 해도 인코더, predictor, 3 split 모두 소실 없음. 판정이 바뀌는 임계값은 위 최대하락 이하여야 한다(인코더 interp 0.033, 인코더 train 0.012, predictor extrap 0.010 등). 0.2 대신 0.1, 0.15 를 써도 같은 결론.
- 보조 곡선: attentive late 는 0.05 에서만 인코더 interp 깊이 12(하락 0.068), predictor interp 깊이 2(하락 0.079)가 잡히고 0.10 이상에서는 없음. linear early 는 0.05 에서도 없음. (ii) linear late 는 0.05 에서 extrap 깊이 6(0.053).
- 즉 결론은 임계값 0.1~0.3 범위에서 안정하다. 0.2 가 "파일럿 후 확정" 제안값이고 확정 기록이 없다는 사실은 이 결과에서는 판정에 영향이 없다(주 곡선 한정).

## 8. 한계와 잠재 함정
1. 실효 시드: 인코더는 사실상 n=1 (linear, ridge 결정적), predictor 는 n=3 학습 시드. "5 probe 시드" 는 독립 반복으로 보면 안 된다.
2. ind_ood_gap 검정은 정의상 인코더에서 자명하게 분리를 보고한다(tol≈0). 또 split 간 R² 척도(분산 분모)가 달라 train vs interp gap 0.3 은 모델 차이가 아니라 척도 차이를 포함한다. 해당 검정의 "겹침 없음"은 판정에서 쓰이지 않았지만 해석에 쓰면 안 된다.
3. H-d 의 후반부("OoD 에서 L* 가 앞당겨짐")는 소실이 없어 검정되지 않았다. refuted 는 전반부 반증에 의한 것이다.
4. 깊이 격자: 인코더 k 는 {0,2,4,...,12} 거친 격자(간격 2)라 짧은 구간의 급락은 놓칠 수 있으나 평균 곡선이 완만해 가능성 낮음(추정). refine_suggestion 은 빈 dict.
5. extrap R² 는 y 범위가 넓고 mask baseline 이 0.817 이라 천장 효과로 변별력이 낮다(§4).
6. pilot: true 라벨이 본 실행 결과에 붙어 있음, 임계값 확정 기록 없음 (사실 맥락). 본 통계 결과는 이 둘에 민감하지 않음(7절).
7. 다중 비교: 보조 지표 다수(probe 2종 x 시점 5 x split 3 x 모듈 2)를 훑었으므로 보조 곡선에서 우연히 큰 하락(예: attentive (ii) 0.297)이 나올 수 있다. 보정하지 않았고 탐색적이다.

## 9. 확신도
중간.
- 높음 부분: depth_r2.csv 재계산이 summary.json 과 일치하고, 주 곡선의 최대하락이 임계값의 1/6 이하이며 시드 변동으로는 0.2 를 만들 수 없다. 임계값 0.05~0.3 에서 판정 불변.
- 낮음 부분: 실효 시드 수가 1(인코더), 3(predictor)이고, 분산 분모 차이 때문에 split 간 비교가 척도 의존적이다. "크기 정보가 유지된다"는 일반화는 mask/random-init baseline 과의 차이가 작아(extrap +0.03) 약하게만 지지된다. 소실 부재의 원인(잔차 스트림 구조 등)은 추정이다.
