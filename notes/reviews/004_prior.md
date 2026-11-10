# 004_depth_truncation 리뷰 (analyst-prior: 선행연구/이론 관점)

- 작성: analyst-prior, 2026-11-10. 다른 분석가의 리뷰/연구 파일은 읽지 않았다.
- 읽은 것: experiments/004_depth_truncation/{results.json, results/summary.json, config.yaml, README.md, CHANGES.md}, results/depth_r2.csv(4.7MB, 필요한 행만 grep), results/next_z_mse.csv(앞 40행), results/fig_curves.png, fig_heatmap.png, notes/specs/exp4_5_spec.md 1~3절, notes/hypotheses.md 후보 I/확정 기록 8~12, notes/log.md 2026-11-09 항목(일부 줄은 "omitted"로 못 읽음), reports/002_single_size_sweep.md.
- 못 읽은 것: results/parts/, results/colab/timing_main.json(이 관점에서 필요 없어 생략), 코드(stages.py, depthlib.py). 아래 "미확인" 표시 참조.
- 수치 규칙: summary.json / results.json 값은 `summary:키`. depth_r2.csv 값은 **학습 seed 0 / probe seed 0 한 줄**의 값(도구 제약으로 평균을 계산하지 못했다). seed 평균은 summary.json 의 `l_best_value`, `ind_ood_gap` 에만 있다. 따라서 아래 CSV 수치는 "한 seed 의 점추정"이며 seed 평균이 아니다. 곡선 모양에 대한 주장은 fig_curves.png(seed 평균 + 음영)와 함께 읽었다.
- 문헌 확인 수준 표기: [확인-초록] 이번 세션에 웹 검색으로 초록/요약 수준만 확인, [미확인] 기억에 기반하며 원문/서지 미확인(근거로 쓰지 말고 후보로만).

---

## 1. 결과 요약 (수치)

### 1.1 저장 판정
- `summary:verdict_H_d` = refuted, reason `no_drop_ge_threshold_at_any_depth`. 인코더/predictor 둘 다 `any_drop_ge_threshold: false`. `k_star`, `m_star`, 모든 split 의 `vanish` 는 null, `loss_ood_only: false`.
- `summary:l_best` = predictor depth 6, `l_best_value` 0.7022 (interp+extrap 평균 R², 주 설정 linear/new/real/late). `l_best_by_module`: 인코더 depth 10 = 0.6913, predictor depth 6 = 0.7022 (차 0.0109).
- `drop_threshold` 0.2 는 제안값이며 확정 기록이 없다(config.yaml:36 주석, 사용자 사실 맥락). `pilot: true` 는 본 실행에 남은 플래그다(results.json:3). 아래 1.3 에서 보듯 판정은 임계값에 거의 민감하지 않다.

### 1.2 곡선 (seed 0 점추정, 주 설정 linear/new/late, split=ALL)
| 모듈/깊이 | train | interp | extrap | 출처(depth_r2.csv 행) |
|---|---|---|---|---|
| encoder k=0 real | 0.733 | 0.217 | 0.788 | encoder,0,...,real |
| encoder k=2 | 0.793 | 0.428 | 0.824 | 〃 |
| encoder k=10 | 0.846 | 0.542 | 0.841 | 〃 |
| encoder k=12 | 0.834 | 0.511 | 0.852 | 〃 |
| encoder k=0 random-init | 0.458 | -0.147 | 0.656 | control=random_init |
| encoder k=12 random-init | 0.658 | 0.207 | 0.768 | 〃 |
| encoder 라벨 셔플 (전 깊이) | -0.05 ~ -0.03 | -0.07 ~ -0.03 | -0.05 ~ -0.03 | control=label_shuffle |
| predictor m=0 real | 0.834 | 0.511 | 0.852 | predictor,0,...,real,0 |
| predictor m=6 real | 0.854 | 0.548 | 0.849 | predictor,6,...,real,0 |
| predictor m=0 random-init | 0.834 | 0.511 | 0.852 | control=random_init |
| predictor m=6 random-init | 0.832 | 0.508 | 0.852 | 〃 |
| baseline pixel (linear) | 0.373 | -0.252 | 0.628 | baseline,-1,...,pixel |
| baseline mask_area_bbox | 0.588 | -0.001 | 0.817 | baseline,-1,...,mask_area_bbox |
| reference: 인코더 토큰에 exp1 probe | 0.818 | 0.446 | 0.847 | reference,-1,...,exp1_probe |

### 1.3 소실(하락) 크기
- 인코더 seed 0 에서 "직전 최고값 대비 하락"의 최대는 interp k=10 -> 12 의 0.542 -> 0.511 (-0.031). train 은 0.846 -> 0.834 (-0.012). extrap 은 k=6 0.840 -> 이후 0.836/0.841/0.852 로 사실상 단조.
- predictor real, seed 0: extrap 은 m=0 0.852 에서 m=2 0.838 까지 -0.014 내려갔다가 m=6 0.849. train 과 interp 는 m=0 -> 6 에서 각각 +0.020, +0.037 상승.
- 위는 seed 0 한 줄이라 seed 평균에서는 다를 수 있다(추정). 그러나 가장 큰 하락(-0.03)도 임계값 0.2 의 1/6 이다. 임계값을 0.05 로 낮춰도 같은 판정이 나올 가능성이 높다(추정, seed 평균 미계산). 반대로 가장 큰 변화는 **상승**이다: 인코더 interp k=0 -> 2 에서 +0.21, k=0 -> 10 에서 +0.33.
- (ii) 조기 종료 출력에 exp1 probe 를 그대로 적용: train R² 가 m=0 -4.50, m=3 -0.87, m=5 0.51, m=6 0.82 (predictor,m,linear,exp1_probe,..real,0). 다음 z MSE 는 m=0 에서 m=6 까지 단조 감소(예: a026 early 1.085 -> 0.175, next_z_mse.csv 2~8행).

### 1.4 split 간 차
- `summary:ind_ood_gap` (train - split, 인코더 interp): 0.516(k=0) -> 0.303~0.365(k=2~12), 거의 일정. extrap: 대부분 깊이에서 |gap| < 0.02, 부호 혼재. predictor interp gap 0.305~0.325(전 깊이), extrap gap -0.017 ~ +0.009, `fraction_depths_inside_seed_sd` 0.571.
- `overlap: false` 인 이유는 interp 쪽 gap 이 크기 때문이다(`fraction_depths_inside_seed_sd` 0.0). 이 gap 은 아래 3.2 에서 보듯 R² 분모 효과가 크다.

---

## 2. 문헌과의 정합

수준 표기: [확인-초록] / [미확인]. 모든 문헌 서술은 "이 결과와 방향이 맞는가"에 한정하며, 정량 비교는 하지 않았다(문헌 수치는 인용하지 않는다).

### 2.1 ViT 층별 표현은 깊이에 대해 균질하다, 잔차 연결이 전달한다
- Raghu et al., "Do Vision Transformers See Like CNNs?", NeurIPS 2021, https://arxiv.org/abs/2108.08810 [확인-초록]: ViT 층 표현이 CNN 보다 깊이에 걸쳐 균질하고, 잔차 연결이 앞 층 특징을 뒤로 전달한다는 취지(초록 수준). 본문 실험 설정은 미확인.
- 정합: 인코더 곡선이 k=2 이후 평평하다(train 0.793 -> 0.846, extrap 0.824 -> 0.852). random-init 인코더도 k=2 이후 평평(train 0.642 -> 0.658)하다. 층이 크기 정보를 "지우지 않는" 구조이므로 소실 지점이 없는 것이 자연스럽다(추정).
- Elhage et al. 2021 "A Mathematical Framework for Transformer Circuits"(잔차 스트림을 메모리/통신 채널로 보는 관점) [미확인: 서지, 원문 미확인], nostalgebraist 2020 logit lens [미확인]. 같은 방향의 이론 틀로 후보. 확정 서술은 하지 않는다.
- 핵심 판정 근거: predictor 의 **random-init** 곡선이 real 곡선과 거의 겹친다(m=0 -> 6 에서 random-init 0.834/0.511/0.852 -> 0.832/0.508/0.852; real 은 0.854/0.548/0.849). 즉 predictor hidden probe 가 읽는 크기 정보의 대부분은 **입력 z 가 잔차 스트림으로 그대로 흘러온 것**이고, 학습된 predictor 가 더한 것은 train +0.02, interp +0.04 수준이다(seed 0, 추정: seed 평균 미확인). 이것은 002 보고서에서 세 분석가가 "H-b last_block pass 는 실패하기 어려운 시험(잔차 연결)"이라고 평가한 것(reports/002 §5.2)과 같은 현상의 깊이 축 확장이다. 즉 004 는 "predictor 가 크기를 보존하도록 학습했다"의 증거가 아니라 "잔차 구조가 있으면 읽힌다"의 확인으로 읽는 편이 안전하다.

### 2.2 probe 해석: "읽을 수 있음"과 "사용됨"
- Hewitt & Liang 2019, https://arxiv.org/abs/1909.03368 [확인-요약]: 통제 과제와 selectivity(진짜 과제 - 통제 과제)로 probe 해석, 선형 probe 가 MLP 보다 selectivity 가 높다는 취지.
- 정합: 이 실험의 라벨 셔플 통제는 약 0(-0.05 ~ -0.03)이라 probe 가 라벨을 외우는 것은 아니다. random-init 통제는 selectivity 의 역할을 하며, 인코더에서 real - random-init 은 k=12 에서 train 0.176, interp 0.304, extrap 0.084(seed 0). 이는 002 가 보고한 보간 +0.34, 외삽 +0.07(reports/002 §4.1 H-a 행)과 같은 순서다. 외삽에서 사전학습 이득이 작은 것도 반복된다.
- 단 이 통제들은 "probe 가 표현에 기대는가"를 말할 뿐 "모델 계산이 그 정보를 쓰는가"는 말하지 않는다. Alain & Bengio 2016(선형 probe) [미확인: 서지], Ravichander et al. 2021 "Probing the Probing Paradigm"(읽을 수 있으나 쓰이지 않는 속성) [미확인], Elazar et al. 2021 amnesic probing [미확인]이 이 구분을 다루는 걸로 기억하나 원문은 읽지 않았다. 결론: 004 는 전적으로 "읽을 수 있음"의 측정이다. `refuted` 는 "정보가 깊이에서 사라지지 않는다" 이상의 의미가 없고 "predictor 가 쓴다"는 뜻이 아니다(README 의 "읽을 수 있다고만 쓴다" 원칙과 같다).
- (ii) 결과는 이 구분의 작은 사례다. 같은 hidden 에서 새로 학습한 probe(`fit=new`)는 m=0 부터 높게 읽는데(train 0.834), 인코더 토큰으로 학습한 exp1 probe 를 조기 종료 출력에 그대로 적용하면 m=0 -4.50, m=3 -0.87 이다. 정보는 읽히지만 **인코더 토큰의 읽기 방향/기저에서는** 읽히지 않는다. logit lens 와 tuned lens 의 구분(Belrose et al. 2023 tuned lens [미확인])과 같은 구도이며, 조기 종료 경로는 학습 때 없던 경로라는 명세 3.2 의 경고와도 일치한다. 이 지표로 "소실"을 말하면 안 된다. 다음 z MSE 의 단조 감소도 같은 해석이 가능(조기 종료가 학습된 경로가 아니므로 MSE 가 큰 것이 정보 소실이 아님)하다.

### 2.3 월드모델 표현 연구
- DINO-WM, Zhou et al., arXiv 2411.04983 / ICML 2025, https://proceedings.mlr.press/v267/zhou25t.html [확인-초록]: 사전학습 DINOv2 패치 특징 위에서 동역학 모델을 학습, 평가는 미로/밀기/입자 조작. 천(cloth) 크기 정보 흐름이나 층별 probe 는 이 초록에 없다.
- Vafa et al. (월드모델 평가/inductive bias probe; 005 의 R-IB, D-IB 출처로 repo 노트에 등장) [미확인: 원문 식은 확정 기록 12 가 "원문 식 미확인"이라고 적음]. 004 에는 직접 비교 대상이 아니다.
- 월드모델의 층별 소실 곡선을 잰 직접 선행연구를 이번에 찾지 못했다("찾지 못함" = 검색 범위 한정, 없다는 뜻이 아님). 따라서 이 결과가 "기존 보고와 일치/불일치"인지는 판단 불가이고, 위 ViT/probe 문헌과 방향이 맞는지만 말할 수 있다.

### 2.4 DINOv2 층별 정보 (추정)
- DINOv2 마지막 층 특징이 dense 과제에서 강하다는 것은 DINOv2 논문(Oquab et al. 2023) [미확인: 이번에 원문 미확인]. 중간 층이 마지막 층보다 위치/저수준 정보에 유리하다는 보고가 ViT 일반에 있다고 기억하나(Amir et al. 2021 "Deep ViT Features as Dense Visual Descriptors" [미확인]) 크기 정보에 대해서는 확인하지 못했다.
- 이 결과의 관찰: l_best 인코더 층이 마지막 층이 아닌 10(`l_best_by_module.encoder.depth`)이고 k=10 -> 12 에서 interp -0.031(seed 0)이다. 마지막 두 층에서 약간 줄어드는 경향은 "마지막 층은 사전학습 목적에 맞게 특화" 가설과 방향은 맞지만 크기가 작고 seed 평균/분산 미확인이라 **추정**일 뿐이다. 의미 있는 주장으로 쓰지 말 것.

---

## 3. 대안 설명 (왜 소실이 없는가 / 왜 읽히는가)

3.1 **크기가 저수준 단서(천 면적/색 픽셀 비율)이기 때문.** k=0(패치 임베딩+위치 임베딩)에서 이미 train 0.733, extrap 0.788 이다(random-init 0.458 / 0.656). mean-pooled 선형 probe 는 천이 차지하는 픽셀 비율 같은 저수준 통계를 쉽게 읽을 수 있다(추정: k=0 에서 pooled linear 읽기는 패치 픽셀 평균의 선형 함수라는 점을 코드로 확인하지 않았다). 이 경우 깊이 곡선이 평평한 것은 "의미적 크기 표현이 유지됨"이 아니라 "저수준 단서가 처음부터 있고 지워지지 않음"이다. mask_area_bbox baseline 이 extrap 에서 0.817 로 인코더(0.84~0.85)와 거의 같은 것(baseline 행)이 이 해석과 일관된다. 반대로 pixel baseline 은 0.373/-0.252/0.628 로 낮아 "그냥 픽셀 평균"만으로는 설명되지 않는다(이 baseline 은 4x4 평균 48차원 토큰을 같은 probe 에 넣은 것, CHANGES.md 17). 두 baseline 의 상대 위치로 보아 저수준 단서 가설은 부분적으로만 지지된다(추정).

3.2 **R² 의 분모(라벨 분산) 효과.** 같은 R² 라도 interp 은 레벨 2개(a080, a125), extrap 은 4개(a026, a040, a200, a248)라 라벨 분산이 다르다. 002 보고서는 skeptic 의 계산으로 라벨 분산 보간 0.049, 외삽 0.966 을 적었고(reports/002 §5.2, 그 보고서가 "내 계산"으로 표시한 값), MAE 로는 interp 이 trained 와 비슷하다고 했다. 004 의 interp gap(0.30~0.52, `ind_ood_gap`)도 같은 구조일 가능성이 높다(추정). 따라서 `overlap_depth_fraction` 기준의 "InD/OoD 곡선이 seed 편차 안에서 겹침" 조항은 이 데이터에서 절대 겹치지 않는 지표다. H-d 의 반증 두 조건 중 "하락 < 0.2" 만이 의미 있게 작동했다. 레벨 행의 pooled-variance R²(CHANGES.md 15)와 `level == ALL` R² 가 섞여 있으므로 히트맵(a026 열이 어두움: fig_heatmap.png)은 정보 소실이 아니라 이 분모 효과로 읽어야 한다(추정).

3.3 **판독기가 깊이마다 새로 학습됨(`fit=new`).** 주지표는 깊이별 재학습이라 "정보가 표현에 남아 있는가"의 상한이다. 층이 방향을 회전시켜도 R² 는 유지된다. 반면 (ii) 처럼 고정 판독기는 회전에 취약하다(2.2). 둘을 구분해서 쓴 설계 자체는 문헌의 관행에 맞다.

3.4 **teacher forcing 한 step.** 004 는 early/late 윈도우의 teacher forcing 은닉에서만 hidden probe 를 잰다(CHANGES.md 6: rollout 시점의 은닉 probe 없음). 002 에서 문제가 된 것은 rollout 후반(step 12)에서의 R² 하락이다(reports/002 §4.1 H_b:last_rollout_step: Q1 천장 0.8203, Q2 R² 0.5290/0.5250/0.4998). 004 가 보여 주는 것은 "같은 시점 내 깊이 축에서는 소실이 없다"이며, 002 의 시간 축 하락을 설명하지 못한다. 이 둘은 서로 모순이 아니다.

3.5 **predictor 와 인코더 사이에 위치정보/문맥 혼합이 약하다(num_hist=1).** num_hist=1 이라 predictor 가 한 프레임만 본다(CHANGES.md 4). 시간 attention 이 사실상 없어 predictor 는 프레임 내 patch 간 mixing 과 action MLP 만 한다. 그러면 입력 z 가 거의 그대로 흐르는 것이 더 기대된다(추정). "predictor 깊이에서 정보 유지"가 DINO-WM 일반의 성질인지 num_hist=1 변형의 성질인지는 이 실험으로 구분되지 않는다.

3.6 **Colab 본 실행의 정합성 문제는 이 관점에서 판단하지 않는다.** pilot 플래그, 임계값 비확정, 시드(학습 3, probe 5) 관련 통계 해석은 stats/skeptic 관점이다.

---

## 4. 005 와 후속(003/008 크기 끌림)에 주는 의미

4.1 **005 층 선택.** `l_best` predictor 6 과 인코더 10 의 차는 0.011 이고(`l_best_by_module`), `ind_ood_gap` 의 per-depth tol(seed sd 규모) 은 0.007~0.028(예: predictor interp tol 0.007~0.028)이다. 즉 l_best 는 사실상 동률이고 어느 층을 쓰든 크기 정보량은 거의 같다(추정). 005 의 head 가 읽는 입력으로는 "최종 층을 쓰되 random-init 과 면적 휴리스틱 대조를 우선"하는 편이 층 선택 논쟁보다 중요하다. 005 의 결과 해석에서 "l_best 가 predictor 라서 predictor 가 크기를 학습했다"는 문장은 쓰면 안 된다(random-init predictor 가 거의 같은 값을 내므로, 2.1).
4.2 **005 의 면적 휴리스틱 위험.** 3.1 처럼 크기가 저수준 단서로 읽히면 005 의 D-IB 가 "크기 법칙" 대신 "면적 휴리스틱"으로 설명될 위험이 커진다. 5-C 의 `D-IB_area=` 대 `D-IB_area≠` 가 이 질문의 직접 검증이므로 004 결과는 5-C 의 중요성을 높인다.
4.3 **003/008 크기 끌림.** 이 질문의 핵심은 "predictor/rollout 이 학습 크기 쪽으로 끌려가는가"다. 004 는 (a) teacher forcing 한 step 에서 hidden 크기 정보가 깊이에 따라 줄지 않는다는 것, (b) 그러므로 끌림의 원인을 "한 step 안의 깊이에서 정보 손실"로 두기는 어렵다는 것을 보인다(추정; 레벨별 부호 있는 편향은 004 에서 보지 않았다). 남는 후보는 시간 축 누적(rollout)과 출력 공간 판독기 전이 실패이며, 이는 002 보고서 8절 1~3번 제안과 같다. 따라서 003 의 분포별 모델(single_a026, a248, mix_adjacent 등)에 같은 깊이별 hidden probe 를 적용하되 **rollout step 별 은닉 상태**(004 에서 비어 있는 부분)와 **레벨별 부호 있는 예측 편향**을 추가하는 것이 가장 정보량이 큰 후속(추정)이다. 004 코드를 쓰면 새 학습 없이 가능하다.
4.4 **random-init predictor 를 더 활용.** 003 류의 학습 분포 비교에서 "학습으로 생긴 효과"는 real - random-init 차(+0.02~0.04)에 해당한다. 이 크기를 분포 간 차이(예: gap, slope)와 나란히 놓지 않으면 분포 효과를 과대해석하기 쉽다(추정).

---

## 5. 빠진 baseline / 인용 후보

- persistence baseline(마지막 context 프레임 z 를 그대로 복사한 예측의 (ii)(iii))과 untrained predictor 의 (ii) 값이 없다(CHANGES.md 19: random-init predictor 는 (i) 만). 002 prior 리뷰도 persistence 를 제안했고 006 이 맡는다(hypotheses.md 후보 J). 004 의 "깊이" 결과를 (ii)에 쓰려면 필요하다.
- 크기와 무관한 속성(그리퍼 위치, 천 중심 위치 등)의 같은 깊이별 곡선: "크기 특이적 유지"인지 "모든 정보가 잔차로 유지"인지를 구분한다. 없다(3절 대안 해석의 직접 검증 수단).
- 시간 정보 포함 mask baseline(002 에서 skeptic 이 제기했다고 보고서에 기록): 004 의 mask baseline 도 같은 구조인지는 미확인.
- 인용 후보(원문 미확인, 사용 전 확인): Alain & Bengio 2016; Hewitt & Liang 2019 [확인-요약]; Belinkov 2022 probing survey; Ravichander et al. 2021; Elazar et al. 2021; Elhage et al. 2021; nostalgebraist 2020; Belrose et al. 2023 tuned lens; Raghu et al. 2021 [확인-초록]; Oquab et al. 2023 DINOv2; Zhou et al. DINO-WM [확인-초록].

---

## 6. 한계

- 수치의 상당 부분이 seed 0 한 줄이다. seed 평균과 sd 는 summary.json 의 `l_best_value`, `ind_ood_gap` 외에 직접 계산하지 못했다(도구 제약). 곡선 모양은 fig_curves.png 로 교차 확인.
- 코드(probe 가 어떤 레벨로 학습되는지, `fit=new` 의 train 구성, z_only 부분공간 처리)는 읽지 않았다. "fit=new 가 학습 레벨만으로 학습한다"는 가정은 미확인이다.
- `feature_part: z_only` 는 block 1 이후 앞 384 좌표가 z 부분공간이라는 보장이 없다는 우려가 남아 있다(CHANGES.md 10). 이 경우 predictor 의 hidden R² 는 하한에 가깝다.
- 문헌은 초록/요약 수준 3편만 확인했다. 나머지는 기억이고 서지 미확인이다. 정량 비교는 하지 않았다.
- 임계값 0.2 의 비확정성: refuted 라는 이름은 임계값이 "제안값"이라는 점과 함께 읽어야 하며, 이 데이터에서는 임계값을 낮춰도 같을 가능성이 높다(추정).
- H-d 의 두 번째 반증 조건(겹침)은 R² 분모 효과 때문에 구조적으로 판별력이 약하다(3.2, 추정).

---

## 7. 확신도

- 곡선이 평평하고 소실이 없다(수치로 직접 확인): **높음** (summary 판정, CSV seed 0, 그림 모두 일관).
- "predictor 깊이에서의 유지는 입력 z 가 잔차로 흐르기 때문이고 학습 기여는 작다": **중간** (random-init predictor 곡선이 거의 같다는 직접 증거 + ViT/잔차 문헌 [확인-초록]. 단 seed 0 한 줄, z_only 부분공간 우려).
- "읽을 수 있음이지 사용됨이 아님": **높음** (측정 방법상 자명; 문헌 일부 미확인이나 결론에 의존하지 않음).
- 저수준 단서(면적/색) 가설, num_hist=1 가설, 마지막 두 층 소폭 하락의 원인: **낮음** (모두 추정, 직접 검증 없음).
- 005, 003/008 에 주는 시사점(4절): **중간-낮음** (추정이 섞였고, 레벨별 편향과 rollout 은닉은 이 실험에 없음).
- 전체: **중간**. 이유: 판정과 수치는 확실하나 seed 평균과 코드 확인이 없고, 문헌은 초록 수준이며, 직접 비교할 월드모델 층별 연구를 찾지 못했다.
