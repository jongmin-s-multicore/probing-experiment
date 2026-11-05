# 실험 4·5 구현 명세: DINO-WM 천 크기 정보 흐름

> 사용자가 2026-11-03 세션에 붙여 준 명세 원문을 옮긴 것이다. 사용자 결정은 notes/hypotheses.md 후보 I, 확정 기록 9 에 있다.
> **상태: 초안.** 실험 1~3 결과가 나오면 범위, 분할, 기준값이 바뀔 수 있다. 원본 설계 문서는 claude.ai의 「DINO-WM 천 크기 정보 흐름 실험 설계」다. 이 파일은 그중 실험 4·5를 구현용으로 옮긴 것이다.

## 0. 작업 규칙 (먼저 읽을 것)

1. **저장소부터 확인한다.** 실험 1~3의 코드(feature 캐싱, probe, predictor 학습, 평가)가 이미 있으면 그대로 재사용한다. 같은 probe 구현, 분할, 지표 함수를 써야 실험 간 비교가 성립한다. 새 모듈은 기존 디렉터리 관례를 따른다. 관례가 없으면 `exp4/`, `exp5/`, 결과는 `results/exp4/`, `results/exp5/`에 둔다.
2. **실험 1~3 결과가 없으면** 코드를 구현하고 작은 부분집합으로 파일럿만 돌린다. 결과에는 `pilot: true`를 표시한다. 실험 5의 분기(§4.6)는 실험 4 결과가 나와야 정해진다.
3. **§6의 가정은 코드에서 확인하고 기록한다.** 가정이 틀리면 명세의 숫자(층 수, 차원 등)를 실제 값으로 바꾸고 `CHANGES.md`에 적는다.
4. **원본과 달라진 점은 모두 `CHANGES.md`에 기록한다.** 해상도, 프레임 수, context 길이, 사용 레벨, episode 수, 학습 step, 논문 방법과의 차이가 대상이다.
5. **기존 실험 1~3의 코드와 결과 파일은 수정하지 않는다.** 필요한 함수는 import해서 쓴다.
6. **seed를 고정하고 반복한다.** probe와 head는 seed 5회가 기본이고, 비용이 크면 최소 3회다. 평균과 표준편차를 보고한다.
7. **해석을 과장하지 않는다.** probe 정확도만으로는 "읽을 수 있다"고만 쓴다. "사용된다"는 실험 3(개입) 이후에만 쓴다.

## 1. 배경 요약

- **모델:** DINO-WM. frozen DINOv2가 관측을 패치 토큰 z_t ∈ R^{N×E}로 바꾼다. ViT predictor가 과거 H프레임과 action으로 다음 프레임의 z를 예측한다. attention은 프레임 단위 causal이고, action은 MLP를 거쳐 각 패치 벡터에 concat된다.
- **데이터:** cloth-sim. ACWM-Phys Cloth Move를 PyFleX로 재생성했고 Hugging Face `hanbin5/cloth-sim`에 있다. 천 크기는 9레벨이고 크기 외 규칙은 같다. 레벨마다 train 2,000, test 100 episode다. 영상은 512×512, 16fps, 65프레임 mp4이고 action은 (64, 8)이다. 4번째·8번째 값은 상수 1이고 처음 3 step은 0이다.
- **크기 라벨:** `metadata.pt`의 `cloth_dimx`, `cloth_dimy`(입자 수). 주 타깃은 **명목 면적비의 로그(episode마다 상수)**다.

| 레벨 | 면적비 | 입자 (폭×높이) | 역할 |
| --- | --- | --- | --- |
| a026 | 0.26 | 54×51 | 외삽 평가 |
| a040 | 0.40 | 65×62 | 외삽 평가 |
| a060 | 0.60 | 79×75 | 학습 |
| a080 | 0.80 | 89×86 | 보간 평가 |
| ind | 1.00 | 98×94 | 학습 |
| a125 | 1.25 | 108×103 | 보간 평가 |
| a160 | 1.60 | 120×115 | 학습 |
| a200 | 2.00 | 133×127 | 외삽 평가 |
| a248 | 2.48 | 146×139 | 외삽 평가 |

- 학습 레벨당 episode는 300개 안팎, 평가는 전 레벨 test 100개를 쓴다.
- predictor는 다중 InD 크기(학습 레벨 3개)로 학습한 조건 (b)를 기본으로 쓴다.

## 2. 공통 설정 (실험 1~3과 동일하게 유지)

| 항목 | 설정 |
| --- | --- |
| 타깃 | log(명목 면적비), episode 단위 상수 |
| probe | linear(mean-pooled 토큰)와 attentive probe(학습 query 1개의 cross-attention → residual → GeLU·LayerNorm 2층 MLP(hidden 4×dim) → linear). 모든 위치에서 같은 용량 |
| probe 학습 | SparseImagination A.5를 출발점으로 쓴다. Adam, lr 1e-5, batch 128, cosine(warmup 없음), 최대 500 epoch. 검증 손실로 조기 종료 |
| 통제 | random-init 인코더·predictor, 라벨 셔플, mask 면적/bbox baseline |
| 분할 | episode 단위. 같은 episode의 프레임이 train/test에 섞이지 않게 한다 |
| 시간 축 | 초기 평평한 프레임과 변형이 큰 후반 프레임을 따로 보고한다 |
| 지표 | R², MAE, Pearson r, seed 평균 ± 표준편차 |
| 예산 | 비교 대상끼리 같은 step과 wall-clock. 측정한 GPU 모델, 최대 VRAM, 시간을 기록 |

## 3. 실험 4: 레이어를 앞에서부터 일부만 써서 소실 지점 찾기

**목적.** 인코더와 predictor의 transformer 레이어를 앞에서부터 k개만 쓴 모델을 순서대로 만들고, 각 깊이에서 천 크기가 얼마나 남는지 재서 소실 지점을 찾는다.

**실험 2와의 차이.** 실험 2는 완성된 predictor의 block 출력만 읽는다. 실험 4는 인코더 내부 층을 포함하고, 잘린 predictor를 실제 world model로 돌려 예측 출력에서도 크기를 본다.

**가설 H-d.** 천 크기는 어떤 깊이 L* 이후에 급격히 줄어들고, OoD 크기에서는 L*가 InD보다 앞당겨진다.
**반증 조건.** 모든 깊이에서 R² 하락이 0.2 미만이거나, InD와 OoD 곡선이 seed 편차 안에서 겹친다. 0.2는 제안값이며 파일럿에서 확정한다.

### 3.1 4-A: 인코더 절단

- DINOv2 block 1..k만 통과시킨다. k = 0(패치 임베딩)부터 마지막 block까지다(ViT-S/14면 12, §6에서 확인).
- 각 k마다 최종 LayerNorm을 적용한 패치 토큰을 캐싱한다. DINOv2 hub 모델의 `get_intermediate_layers(x, n=..., norm=True)` 같은 API가 있으면 쓰고, 없으면 forward hook을 쓴다.
- 각 k에서 linear·attentive probe를 학습하고 평가한다.
- 기준: k = 0이 하한이다. raw pixel probe와 mask 면적 baseline을 같은 표에 둔다.
- 저장 용량이 크면 층 전체가 아니라 k ∈ {0, 2, 4, 6, 8, 10, 12}로 먼저 돌리고, 하락 구간을 촘촘히 다시 본다.

### 3.2 4-B: predictor 조기 종료

- 실험 2에서 학습한 predictor (b)를 쓴다. block 1..m까지만 통과시키고 최종 norm과 출력 projection을 붙여 다음 z를 낸다. m = 0(입력 z)부터 마지막 block까지다(기본 6층 가정, §6에서 확인).
- 각 m에서 세 값을 기록한다.
  1. **(i) hidden probe (주지표):** block m 출력에서 probe를 새로 학습해 R²를 잰다. action이 concat된 상태이므로 z 부분만 쓴다. 마지막 context 프레임의 패치 토큰을 쓴다.
  2. **(ii) 출력 공간 크기:** 조기 종료 예측 z에 실험 1 probe(인코더 최종 출력에서 학습)를 그대로 적용한 "예측된 크기"의 R²와 MAE.
  3. **(iii) 예측 오차:** 다음 z의 MSE.
- 조기 종료 경로는 학습 때 없던 경로다. 그래서 (ii)·(iii)이 나빠지는 것만으로는 소실의 증거가 아니다. 결론은 (i)로 내고 (ii)·(iii)은 보조로 쓴다.

### 3.3 4-C (선택): 깊이 재학습

- predictor를 깊이 1·2·4·6으로 새로 학습한다. 입력은 4-A의 최선 층 k*와 인코더 최종 층 각각이다.
- 4-B와 같은 값에 rollout 오차를 더한다. 비교 대상은 같은 데이터와 같은 step 수의 기본 DINO-WM이다.
- 비용이 크면 생략하고 `CHANGES.md`에 기록한다. **(사용자 결정 2026-11-03: 생략)**

### 3.4 공통 절차

1. **probe 규칙:** 모든 깊이에서 같은 용량, 같은 학습 설정, seed 5회를 쓴다. "깊이별로 새로 학습한 probe"(읽을 수 있음)와 "실험 1 probe를 그대로 적용"(같은 방향에 남아 있음)을 따로 보고한다.
2. **레벨별 분리:** 학습 레벨, 보간 레벨, 외삽 레벨마다 곡선을 따로 그린다.
3. **시점:** 초기 프레임, 후반 프레임, rollout t+1·t+4·t+8에서 각각 잰다.
4. **소실 지점 정의:** R²가 직전 최고값보다 0.2 이상 처음 떨어지는 깊이를 소실 지점으로 둔다. 인코더는 k*, predictor는 m*로 따로 정한다. 하락이 없으면 `null`로 둔다.
5. **통제:** random-init 인코더·predictor의 같은 곡선, 라벨 셔플 probe.

### 3.5 출력

- `results/exp4/depth_r2.csv`: 열은 `module`(encoder/predictor), `depth`, `probe`(linear/attentive), `fit`(new/exp1_probe), `level`, `split`(train/interp/extrap), `timepoint`, `seed`, `r2`, `mae`, `pearson`.
- `results/exp4/fig_heatmap.png`: 깊이(인코더 0..12 → predictor 1..6을 한 축으로 이음) × 크기 레벨 R² 히트맵.
- `results/exp4/fig_curves.png`: InD/보간/외삽 평균 곡선과 통제 곡선.
- `results/exp4/summary.json`(실험 5 입력):

```json
{
  "pilot": false,
  "encoder_layers": 12,
  "predictor_layers": 6,
  "l_best": {"module": "encoder", "depth": 9},
  "k_star": null,
  "m_star": 4,
  "loss_ood_only": true,
  "drop_threshold": 0.2,
  "notes": "값은 예시"
}
```

## 4. 실험 5: 라벨 없이 천 크기로 외삽하는가 (inductive bias probe)

**출처.** Vafa, Chang, Rambachan, Mullainathan, "What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models", ICML 2025 (arXiv 2507.06952). 이 데이터에 맞게 줄인 버전이다.

**아이디어.** 모델이 천 크기를 내재적으로 배웠다면, 크기로 답이 정해지는 새 과제에 적은 데이터로 적응할 때 크기를 기준으로 외삽해야 한다. 고정 probe 하나로 "읽히는지" 보는 대신, 무작위 과제 여러 개에 적응시킨 뒤 학습된 함수들이 크기 구조를 따르는지 본다.

**논문에서 쓰는 이유.** 표현에서 예측 가능한 state 매핑이 여럿이면, standard probe로는 모델이 실제로 어느 것을 쓰는지 구분하지 못한다. IB probe는 state가 같은지만 쓰므로 state를 어떻게 표현했는지에 둔감하다.

### 4.1 대응 관계

| 논문 | 이 실험 |
| --- | --- |
| foundation model m̂_D (데이터셋 D에 fine-tune) | 실험 4의 층 ℓ에서 자른 DINO-WM + 작은 head. 기본은 캐싱 feature 위 head만 학습. 선택으로 마지막 predictor block을 LoRA로 같이 적응 **(사용자 결정: LoRA 생략)** |
| state φ(x) | 크기 레벨 9개(유한 state). 연속 버전은 log 면적비 s |
| 입력 x | 한 episode의 프레임 1장(또는 context 창). 같은 크기의 다른 episode와 다른 프레임은 같은 state |
| 합성 데이터셋 D_1..D_100, 각 100 입력 | 같은 규모로 시작 |
| oracle (참 state를 받는 linear·2층 MLP) | 참 s를 입력받는 linear와 2층 MLP(hidden 5) |
| pseudo-state (합법 수 집합) | 보이는 천 mask 면적의 구간 |
| force vector fine-tune + PySR | 소량 크기 라벨로 fine-tune한 뒤 slice별 관계식 비교 |

**head 구조.** §2의 attentive probe와 같은 구조와 용량을 쓴다. 출력만 과제에 맞춘다(이진은 logit 1개, 연속은 스칼라 1개). 과제마다 새로 초기화해 학습한다.

**평가 입력.** test split 전 레벨에서 2,000개를 뽑는다(논문 App. B와 같은 규모). 초기 프레임과 후반 프레임을 반씩 섞고, 어느 쪽인지 기록한다.

### 4.2 5-A: 유한 state (R-IB, D-IB)

- 과제 k(=1..100): 9개 레벨에 0/1을 Bernoulli(0.5)로 배정한다. 같은 레벨 입력은 같은 출력을 받는다.
- D_k: 전 레벨 train split에서 100개를 무작위로 뽑는다. 레벨당 최소 1개를 보장한다.
- 각 D_k로 head를 학습하고 평가 입력 2,000개에 대한 예측 ŷ_k(x) ∈ {0,1}을 저장한다.
- 지표 (논문 Eq. 1·2, App. B):

```
R-IB_raw = mean_k mean_{(i,j): φ(x_i)=φ(x_j)} 1[ŷ_k(x_i) = ŷ_k(x_j)]
D-IB_raw = 1 − mean_k mean_{(i,j): φ(x_i)≠φ(x_j)} 1[ŷ_k(x_i) = ŷ_k(x_j)]
```

- **정규화:** 논문 Table 2는 1 = 완벽, 0 = 무정보 모델로 보고한다. 같은 절차를 oracle과 무정보 모델(입력과 무관한 무작위 예측)에 돌려 `score = (raw − raw_noninf) / (raw_oracle − raw_noninf)`로 맞춘다. 논문의 정확한 정규화식은 원문 App. B에서 다시 확인하고, 다르면 원문을 따른다.
- 보고: 전체, InD 쌍만, OoD가 하나 이상 포함된 쌍, 둘 다 OoD인 쌍으로 나눈다.

### 4.3 5-B: 외삽 calibration (상대 크기)

- 과제 k: 무작위 매끄러운 함수 g_k(s)를 쓴다. **(사용자 결정: 주파수를 제한한 무작위 사인파 합)** s가 1차원이라 선형 함수만 쓰면 과제들이 affine 변환으로 같아지므로 비선형을 쓴다. 이 점은 논문(무작위 선형 사영)과의 차이로 `CHANGES.md`에 적는다.
- D_k: **학습 레벨(a060, ind, a160) 입력만**으로 100개를 뽑는다. 평가는 전 레벨에서 한다.
- extrapolative predictability (논문 Eq. 3, H = identity, ℓ = MSE):

```
Î(x_i, x_j) = − mean_k (ŷ_k(x_i) − ŷ_k(x_j))²
I*(x_i, x_j) = 같은 식을 oracle 예측으로 계산
```

- calibration 곡선 (논문 App. B 방식): 평가 입력 쌍에서 oracle 예측의 쌍별 거리를 계산하고, 그 범위를 20개 등간격 구간으로 나눈다. 구간마다 모델 예측의 평균 쌍별 거리를 계산해 oracle 대비로 그린다. oracle과 같으면 45도선이다.
- 보고: 전체 곡선과 외삽 레벨이 포함된 쌍만의 곡선, 45도선에서의 평균 절대 이탈.

### 4.4 5-C: 외형 휴리스틱 진단 (pseudo-state 분해)

- pseudo-state q(x): 프레임의 천 mask 면적(픽셀 수)을 10개 분위 구간으로 나눈 값.
- 5-A의 예측을 그대로 다시 쓴다(추가 학습 없음).
  - `D-IB_area=`: 크기는 다르고 q가 같은 쌍에서 계산한 D-IB.
  - `D-IB_area≠`: 크기도 q도 다른 쌍에서 계산한 D-IB.
  - `R-IB_fold`: 크기는 같고 q가 다른 쌍(평평한 프레임 대 접힌 프레임)에서 계산한 R-IB.
- 해석: `D-IB_area=`가 `D-IB_area≠`보다 훨씬 낮으면 inductive bias가 크기가 아니라 보이는 면적을 향한다. 논문 Sec. 4의 합법 수 partition 결과에 대응한다.

### 4.5 5-D: 크기 법칙 일관성 (절대 크기)

- 학습 레벨의 크기 라벨을 소량(예: 학습 입력의 1%, 최소 30개) 써서 head를 크기 회귀로 fine-tune한다.
- slice별로 예측 크기와 참 크기의 관계를 적합한다. slice는 보간 레벨, 외삽 레벨, 초기 프레임, 후반 프레임, 학습 레벨 조합 3가지(라벨을 뽑는 레벨을 바꿈)다.
- 기본은 선형 적합(기울기, 절편)이다. 선택으로 PySR symbolic regression을 돌린다(설치되면).
- 보고: slice별 기울기·절편과 그 편차, 같은 식이 나오는지. 논문 Table 1처럼 slice마다 다른 식이 나오면 "단일 크기 법칙이 아니라 slice별 휴리스틱"으로 기록한다.

### 4.6 기준 모델과 분기

**기준 모델 (5-A~5-D를 같은 절차로 실행):**
1. oracle: 참 s를 입력받는 linear와 2층 MLP.
2. 면적 휴리스틱 모델: mask 면적과 bbox만 입력받는 같은 크기의 head.
3. random-init 인코더·predictor.

주 판정은 우리 모델의 결과가 oracle 쪽인지 면적 모델 쪽인지다.

**실험 4 `summary.json`에 따른 분기:**

| 실험 4 결과 | IB probe를 돌릴 층 | 주 하위 실험 | 결과의 의미 |
| --- | --- | --- | --- |
| `k_star`가 있음 (인코더 중간에서 소실) | 인코더 `l_best`와 최종 층 | 5-A, 5-B | 중간 층이 크기 기반 외삽을 더 잘하면 world model 입력을 바꾸는 개선의 근거 |
| `k_star` 없음, `m_star` 있음 | 인코더 최종 층, predictor `m_star−1`·`m_star` | 5-A, 5-C | 소실 뒤 inductive bias가 면적 휴리스틱 쪽으로 바뀌는지 |
| `loss_ood_only == true` | `l_best` | 5-B 외삽 구간, 5-D | 상대 구조는 유지되는데 절대 법칙이 slice마다 달라지는지 |
| 소실 없음 (OoD 포함) | 최종 predictor 출력 | 5-A~5-D | IB가 높은데도 OoD 예측이 틀리면 원인은 정보가 아니라 사용(실험 3) |
| 실험 1 음성 | 인코더 최종 층 | 5-C만 점검용 | 실험 5는 보류하고 사용자에게 보고 |

여러 행에 해당하면 위쪽 행을 우선하고 아래 행의 하위 실험을 추가한다.

### 4.7 성공 기준 (제안값, 파일럿에서 확정)

- **H-e 크기 기반 외삽:** OoD 레벨을 포함한 R-IB와 D-IB가 모두 면적 휴리스틱 모델보다 높고, random-init와 seed 편차 밖에서 다르다. 5-B 곡선의 45도선 평균 이탈이 0.1 이하다(정규화 거리 기준).
- **H-f 휴리스틱 아님:** `D-IB_area=`가 `D-IB_area≠`와 seed 편차 안에서 같다.
- **H-g 절대 크기:** 5-D의 slice별 기울기가 모두 0.8~1.2 안에 있다.

### 4.8 출력

- `results/exp5/ib_scores.csv`: 열은 `layer`, `model`(ours/oracle/area/random), `metric`(R-IB/D-IB/D-IB_area=/D-IB_area≠/R-IB_fold), `pair_group`(all/ind/ood_any/ood_both), `seed`, `raw`, `score`.
- `results/exp5/calibration.csv`와 `fig_calibration.png`: 층별로 우리 모델, oracle, 면적 모델의 곡선을 겹친다.
- `results/exp5/fig_dib_decomp.png`: D-IB 분해 막대.
- `results/exp5/law_fits.csv`: slice별 기울기·절편(PySR을 돌렸으면 식도).
- `results/exp5/predictions/`: 과제별 예측 ŷ_k(x)(재계산 없이 지표를 다시 낼 수 있도록).
- `results/exp5/REPORT.md`: 가설별 판정(지지/반증/판정 불가)과 근거 수치, 실행 비용(GPU, 최대 VRAM, 시간).

**비용 메모.** head 학습 횟수는 (5-A 과제 100 + 5-B 과제 100) × 층 수 × seed 수 × 모델 4개다. 캐싱 feature 위라 가벼울 것으로 보지만 한 과제를 먼저 실측하고, 길면 seed를 3으로 줄이거나 과제를 50개로 줄인 뒤 `CHANGES.md`에 기록한다.

## 5. 결과 보고 형식

- 각 실험의 `REPORT.md`에는 결론 한 문장을 먼저 쓰고, 표와 그림 경로를 붙인다.
- 판정은 `supported`, `refuted`, `not_evaluable`(baseline 포화, 데이터 부족 등) 중 하나로 쓴다. `not_evaluable`은 음성 결과가 아니다.
- 기준값(0.2, 0.1, 0.8~1.2)을 파일럿 후 바꿨다면, 바꾼 값과 이유를 결과보다 먼저 적는다.

## 6. 코드에서 확인할 가정

| 가정 | 확인 방법 | 틀리면 |
| --- | --- | --- |
| DINO-WM 인코더는 DINOv2 ViT-S/14(12 block, 384차원) | DINO-WM config와 hub 모델 로드 | 층 수와 차원을 실제 값으로 바꾼다 |
| predictor는 6 block | predictor config | m의 범위를 바꾼다 |
| 입력 해상도 224, 패치 16×16 = 256개 | 전처리 코드 | 캐시 용량 추정을 다시 한다 |
| 실험 1 probe와 실험 2 predictor (b) 체크포인트가 있다 | 저장 경로 확인 | 없으면 실험 1·2 코드로 먼저 만든다 |
| mask 면적 계산 함수가 있다 (실험 1 baseline용) | 기존 baseline 코드 | 없으면 HSV 또는 배경 차분 mask를 구현하고 방식을 기록한다 |

## 7. 사용자에게 물을 것 (막히면)

- 실험 1~3 결과가 없거나 일부만 있을 때 어디까지 진행할지. **(답: 코드 + selftest 만)**
- 5-B 무작위 함수 족을 무엇으로 할지. **(답: 무작위 사인파 합)**
- LoRA 적응 옵션을 돌릴 GPU 예산이 있는지. **(답: LoRA, 4-C 모두 생략)**

## 참고

- Vafa et al., "What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models", ICML 2025. https://arxiv.org/abs/2507.06952 (Sec. 2 정의, Sec. 2.3 calibration, Sec. 3 force law, Sec. 4 pseudo-state 분해, App. B 구현 세부)
- DINO-WM 논문 Sec. 3.1 (구조), SparseImagination App. A.5 (attentive probe 설정), LeWM Sec. 5.1·App. F.2 (probe 보고 방식)
- cloth-sim 데이터: Hugging Face `hanbin5/cloth-sim`, 데이터 폴더 README
