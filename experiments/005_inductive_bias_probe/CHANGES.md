# CHANGES: 005_inductive_bias_probe (명세 notes/specs/exp4_5_spec.md 4절 대비 달라진 점)

명세 0절 4 에 따라 원본과 달라진 점을 전부 기록한다. 해석은 하지 않는다.

## 1. 사용자 결정으로 달라진 것
1. **범위: 코드 + 합성 데이터 CPU 셀프테스트 + `--dry-run` 까지.** GPU 실행/다운로드/서버 접속 없음. results/ 는 없다.
2. **LoRA 적응 생략.** 마지막 predictor block 을 LoRA 로 같이 적응하는 선택지는 구현하지 않았다. head 만 학습한다(캐싱된 특징 위).
3. **5-B 함수족 = 주파수를 제한한 무작위 사인파 합** (논문: 무작위 선형 사영). `g_k(s) = sum_j a_j sin(2 pi f_j (s - s_lo) + phi_j)`, j = 1..3, a_j ~ N(0,1), phi_j ~ U(0, 2pi), f_j ~ U(0.3, 1.5) 주기/(로그 면적비 1 단위). 과제마다 격자(`s` 범위 2001점)에서 평균 0, 표준편차 1 로 정규화. 성분 수와 대역은 내가 정한 값(impl default)이고 config `sinusoid.*` 에 있다. 논문의 선형 사영과 달리 과제들이 s 에 대한 affine 변환으로 같아지지 않는다는 점이 이유(명세 4.3).
4. **저장소 관례 경로**: `experiments/005_inductive_bias_probe/` (results/exp5/ 대신). 실험 4 입력은 `../004_depth_truncation/results/summary.json`.

## 2. 기존 설정(002 확정 기록 8)에서 온 것
5. **num_hist = 1, frameskip = 5.** 입력 x = 캐시된 프레임 1장(context 창 = 프레임 1장). 프레임은 002 의 초기 `early_flat`(0-3) / 후반 `late_deformed`(8-11) 8개이며 평가 입력 2000개는 반반(1000 + 1000).
6. head 는 002 `probe_impl.attentive` 의 구조/용량(d_model 128, 4 heads, mlp_hidden 256, 입력 LayerNorm + Linear + 1-query cross-attention + 잔차 MLP + 선형 출력)을 `exp002.probes._make_attentive` 로 그대로 쓴다. 명세 4.1 "§2의 attentive probe와 같은 구조와 용량". 명세 2절의 "학습 query 1개의 cross-attention -> residual -> GeLU·LayerNorm 2층 MLP(hidden 4x dim)" 과 002 구현(hidden 256)이 같은지는 비교하지 않았다 — 002 구현을 따랐다.

## 3. 명세를 그대로 구현할 수 없거나 명세에 정의가 없어서 바꾼/정한 것
7. **5-A 의 D_k 를 "전 레벨 train split" 에서 뽑을 수 없다.** 002 캐시의 train 부분은 학습 레벨 3개(a060, ind, a160)의 train episode 뿐이다(002 가 나머지 6개 레벨의 train split 은 받지 않음). 그래서 D_k 는 캐시의 train 부분에 있는 레벨(3개)에서만 뽑고(각 레벨 최소 1개 보장 유지), 라벨은 9개 레벨 모두에 대해 만든다(평가에는 9개 레벨이 쓰이므로). 코드는 `config ib.pool_levels: null`(= 모든 레벨)이 주어지면 캐시에 있는 레벨 전부를 쓰므로, 나머지 6개 레벨의 train episode 를 002 캐시에 추가하면 코드 수정 없이 명세대로 동작한다. 실제로 쓰인 레벨은 `tasks_meta.json pool_levels_used_5A` 에 기록된다. **이 변경으로 5-A 의 OoD 쌍 점수는 head 가 보지 못한 레벨에 대한 값이 되어 5-B 와 비슷한 외삽 검정이 된다.** (사용자 결정 필요)
8. **정규화식과 무정보 모델.** 명세 문구 그대로 `score = (raw - raw_ref)/(raw_oracle - raw_ref)`, `ref` = 입력과 무관한 i.i.d. Bernoulli(0.5) 예측(`normalization.noninf: iid`). **원문 App. B 의 정확한 식은 확인하지 못했다(원문 미확인).** 셀프테스트가 보인 사실: 이진 과제 + 무작위 라벨에서 참 state 를 아는 모델의 D-IB raw 도 약 0.5 이고 i.i.d. 기준의 D-IB raw 도 약 0.5 이므로 `D-IB` 점수의 분모가 0 에 가깝다. 따라서
   - 분모 `|raw_oracle - raw_ref| < normalization.min_gap`(0.05, impl default)이면 점수를 NaN 으로 쓴다(잡음 점수 방지). 이때 H-e 는 `not_evaluable`(사유 기록).
   - 대안 기준(`const`: 모든 입력에 같은 출력 -> R-IB 1, D-IB 0)을 같이 계산해 `score_alt` 열에 항상 저장하고, config `normalization.noninf.{R,D}` 로 주 점수의 기준을 바꿀 수 있다. 기본은 명세대로 `iid`. 바꾸면 결과를 보기 전에 기록한다.
   - 면적 head 와 random 모델도 같은 분모로 정규화한다. 무정보 모델 행(`noninf`)과 상수 모델 행(`const`)은 `ib_scores.csv` 에 raw 와 함께 들어간다.
9. **OoD 의 정의.** 쌍 그룹의 InD = 학습 레벨(a060, ind, a160), OoD = 학습 레벨이 아닌 모든 레벨(보간 + 외삽; config `ib.ood_roles`). 명세는 "InD 쌍만 / OoD 하나 이상 / 둘 다 OoD" 만 말한다. 5-B 의 "외삽 레벨이 포함된 쌍" 은 외삽 역할(a026, a040, a200, a248)만(`calib.extrap_roles`, 그룹 이름 `extrap_any`).
10. **쌍 정의.** 서로 다른 두 평가 입력의 unordered 쌍(대각 제외). 같은 episode 의 서로 다른 프레임도 "같은 크기"로 센다(명세 4.1). 쌍 수는 약 2백만이고 행렬 연산으로 계산한다.
11. **5-C 쌍 그룹.** `D-IB_area=`, `D-IB_area≠`, `R-IB_fold` 도 pair_group 4가지(all/ind/ood_any/ood_both)별로 계산한다(CSV 열 구조를 같게). H-f 판정은 config `verdict.h_f.pair_group: all`.
12. **면적 10분위 q.** 평가 입력 2000개의 마스크 면적(픽셀, 002 `mask_feat` 의 첫 값)으로 분위수 경계를 정한다. 같은 q 인 쌍과 다른 q 인 쌍을 센다. 면적은 크기와 상관이 크다(레벨이 다르고 q 가 같은 쌍은 이웃 크기일 가능성이 높다): 셀프테스트의 합성 데이터에서 "참 state 를 아는" 토큰의 `D-IB_area=` 도 `D-IB_area≠` 보다 낮았다. H-f 의 해석에 쓸 때 주의(교란).
13. **5-B 정규화 거리와 이탈.** 쌍별 거리 `D_ij = mean_k (yhat_k(x_i) - yhat_k(x_j))^2` (명세의 Î 의 부호를 바꾼 것). 구간 = 선택한 쌍의 oracle 거리 범위 [0, 최댓값] 의 20 등간격. 이탈 = 비어 있지 않은 구간들에서 `|y_model_mean - x_oracle_mean| / norm` 의 평균, `norm` = 전체 쌍의 oracle 거리 최댓값. 구간은 같은 가중(쌍 수 가중 아님). "정규화 거리 기준" 의 구체적 정규화는 명세에 없어서 내가 정했다.
14. **oracle 의 정의.** 참 s 를 입력으로 받는 linear 와 2층 MLP(hidden 5)를 head 와 같은 D_k 로 학습(이진: logit, 연속: 회귀). 정규화와 calibration x 축에 쓰는 것은 config `oracle.primary: mlp`, 다른 하나는 `oracle_linear` 로 별도 저장. 5-B 에는 참 함수 자체를 모델로 쓴 `truth` 행도 calibration 에 추가(참고용). MLP oracle 은 학습 범위 밖에서 포화해 5-D 기울기가 1 보다 작아질 수 있다(셀프테스트에서 관찰; linear oracle 은 1).
15. **면적 휴리스틱 head.** mask 면적/bbox 5개 특징(002 `mask_baseline_features`, 학습 풀 통계로 표준화)을 토큰 1개로 넣는 같은 attentive head. **입력 LayerNorm 은 Identity 로 교체**(5차원 특징에 LayerNorm 을 걸면 스케일 정보가 사라짐). "같은 크기의 head" 해석의 구현 선택.
16. **random-init 모델.** 인코더 층: 004 `enc_cache` 의 `rand_k<k>.npy` (002 `tokens_rand` 와 같은 시드), predictor 층: 실제 인코더 z 를 입력으로 받는 새로 초기화한 predictor (`exp004.deep.random_init_predictor`, 시드 `feat.random_init_seed + train_seed`).
17. **head 학습 일정은 소량 데이터용으로 내가 정함**: 100 epoch, batch 32, AdamW lr 1e-3, wd 1e-2, 검증 분할/조기 종료 없음(100개 중 10%를 떼면 너무 작음). 명세 2절 probe 일정(500 epoch, lr 1e-5, 검증 조기 종료)과 다르다. oracle/면적 소형 head: Adam 400 step, lr 0.02 전체 배치. 모두 impl default 이며 config `head.*`.
18. **5-D 해석.** 라벨 수 = `max(30, ceil(0.01 x 학습 풀 입력 수))`(학습 풀 7200개 가정 시 72). "학습 레벨 조합 3가지(라벨을 뽑는 레벨을 바꿈)" 를 학습 레벨 3개 중 2개씩의 조합 3가지로 해석했다(a060+ind, ind+a160, a060+a160; config `law.combos`). head 는 4개: 3개 레벨 모두에서 라벨을 뽑은 head(slice: 보간 레벨, 외삽 레벨, 초기 프레임, 후반 프레임에서 기울기/절편 적합)와 조합별 head 3개(평가 입력 전체에서 적합). 적합은 평가 입력의 예측값 yhat 을 참 s(log 면적비)에 OLS. slice 별 편차는 `law_fits.csv` 에서 계산한다. PySR 은 `pysr` 가 설치되어 있고 `law.pysr.enabled` 일 때만, ours 의 첫 head seed 에 대해 slice 별로 시도하며 실패 시 `pysr_error: ...` 문자열을 기록한다(조용히 건너뛰지 않음). 이 머신에는 PySR 이 없어 코드 경로를 실행하지 못했다(`pysr_not_installed` 반환만 확인).
19. **판정 규칙의 구체화** (명세 4.7 은 문장뿐): H-e = (OoD 포함 쌍 그룹 `verdict.h_e.pair_group: ood_any` 에서 R-IB, D-IB 점수가 모두 면적 모델보다 높고) 이고 (random-init 과의 차이가 `sd_mult` x max(seed sd) 보다 크고) 이고 (5-B `extrap_any` 이탈 평균 <= 0.1). H-f = |D-IB_area= - D-IB_area≠| <= `sd_mult` x max(seed sd). H-g = ours 의 slice 별 기울기(seed 평균)가 모두 [0.8, 1.2]. `not_evaluable` 사유: 임계값 null, seed 1개, 점수 NaN(분모 문제), 면적 모델 점수가 `baseline_saturated_score`(0.95) 이상, 해당 하위 실험이 이 층에서 선택되지 않음, plan hold. `sd_mult` 2.0 과 0.95 는 impl default 이다.
20. **seed 의 의미.** `seed` 열 = head 학습 seed(5개). 과제(라벨/D_k/사인파)와 평가 입력은 `seeds.*` 로 고정되어 모든 모델/seed 에서 같다. 무정보 모델 예측은 `seeds.noninf + head seed` 로 만들어 저장하지 않는다.
21. **이 실험이 필요로 하는데 002 에 없던 것을 새로 구현**: 평가 입력/과제 생성, IB 지표, calibration, 5-D 적합, 분기 로직, 판정, head 학습 루프(BCE/MSE), predictor 층 추출(`feats.py`: 004 `deep.ee_forward` 사용).
22. **Q1 probe / (b) 체크포인트 / 004 결과가 없을 때**: `plan` 은 summary 가 없으면 FileNotFoundError 로 중단, `feat/heads` 는 체크포인트나 캐시 파일이 없으면 어느 파일인지 적고 중단, `--dry-run` 은 모두 통과. 실험 1 이 음성인지는 config `branch.exp1_negative` 로 사람이 알려 준다(코드가 002 결과를 읽어 판정하지 않음; 002 결과가 저장소에 없음).

## 4. §6 가정 중 이 실험에 해당하는 것 (004 CHANGES.md 의 표 참조)
- 패치 수는 256 이 아니라 196(고정 commit 5a640be `models/visual_world_model.py` 로 확인), predictor 에 출력 projection 없음(`models/vit.py`).
- 실험 1 probe 는 저장되지 않으므로 005 는 probe 를 쓰지 않고 head 를 새로 학습한다(004 의 (ii) 와 달리 필요 없음).
- mask 면적 함수: 002 `masks.py`(HSV) / 캐시 `mask_feat`.

## 5. 못 한 것 / 검증하지 못한 것
- 논문 Eq. 1-3 과 App. B 의 원문 대조(원문 미확인). 정규화식 문제는 위 8번.
- LoRA, 서버 실행 전부.
- PySR 경로(미설치).

## 6. 비용 추정 (추정: 실측 없음)
- head 수 = (5-A 과제 100 + 5-B 과제 100 + 5-D 4개) x seed 5 x 모델 2(ours, random) = 층당 약 2040 개의 attentive head, 레이어 독립 모델(oracle 2 + 면적 1)은 약 3060 개의 소형 head. 층 수는 4.6 분기에 따라 1~4 개(2~3 개가 흔함).
- attentive head 1개: 학습 400 step(100 epoch x 4) + 평가 입력 2000개 순전파. 12GB GPU 에서 약 1.5~3초(추정) -> 층당 1~1.7시간, 층 3개면 3~5시간 + 소형 head 약 0.3시간. 첫 head 의 실측 시간이 로그에 나오므로 길면 seed 3 / 과제 50 으로 줄이고 여기에 기록한다.
- 디스크: predictor 층 특징 파일 층당 real + random-init 각 약 2.2 GB, 평가 입력은 GPU 에 올려 둠(층당 약 0.3 GB, 학습 풀 1.1 GB).
