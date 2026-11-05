# 회의론자 관점 자료조사: DINO-WM은 physical attribute를 보존하고 활용하는가

- 작성: analyst-skeptic, 2026-10-30
- 모드: 자료조사. 실험은 하지 않았다. 코드도 실행하지 않았다.
- 읽은 자료: `notes/research/papers/worldmodel_jepa/` 의 5편. PDF 페이지를 직접 렌더링해서 읽었다.
  - DINO-WM (arXiv 2411.04983v2): 본문 p.1-10
  - LeWM (arXiv 2603.19312v3): 본문과 부록 p.1-26 (Table 1, 3, 4, 5, Fig. 8, 14 포함)
  - G-JEPA (arXiv 2610.28483v1): p.1-9
  - SG-JEPA (arXiv 2609.10464v1): p.1-10
  - T2-Z (arXiv 2610.22364v1): p.1-9
- 외부 자료: 5편 바깥의 probing 방법론 문헌은 4절에 따로 적었다. 확인 수준도 함께 적었다.
- 표기: [확인]은 PDF에서 직접 읽은 내용이다. [추정]은 내 추론이다. [미확인]은 기억에 기반했고 원문을 대조하지 않은 내용이다.

---

## 0. 요약 (결론부터)

1. **H1을 encoder 출력 수준의 가설로 읽으면, 이미 있는 자료가 H1에 불리하다.** [확인] LeWM Table 1(PushT)에서 DINO-WM 표현의 probing 결과는 agent location MLP r=0.999, block location linear r=0.997, block angle linear r=0.979다. Table 4(OGBench-Cube)에서는 DINO-WM의 Overall linear r가 0.725, MLP r가 0.799다. 같은 표에서 LeWM은 0.477과 0.584로 더 낮다. 다시 말해 LeWM 논문 자체가 "DINOv2 표현에 물리량이 LeWM과 같거나 더 많이 들어 있다"고 보고한다. "LeWM에는 보존되고 DINO-WM에는 소실된다"는 출발점과 반대다.
2. **H1의 "transformer layer"가 무엇을 가리키는지 정해지지 않았다.** DINO-WM에서 DINOv2 encoder는 frozen이고 [확인, DINO-WM 3.1.1], 학습되는 transformer는 predictor뿐이다. 둘 중 어느 쪽이냐에 따라 H1은 전혀 다른 가설이 된다. encoder 쪽이면 DINO-WM 고유의 문제가 아니라 DINOv2의 성질에 관한 가설이다. predictor 쪽이면 "희석"을 predictor의 중간 layer에서 재는지, 최종 출력(예측 latent)에서 재는지, rollout 몇 step 후에 재는지를 정해야 한다.
3. **H1과 H2는 서로 배타적이지 않고, probing만으로는 H2를 검증할 수 없다.** probing은 정보가 "디코딩 가능하다"는 것만 보여 준다. predictor가 그 정보를 "사용한다"는 것은 보여 주지 않는다. H2를 검증하려면 개입(intervention) 실험이 필요하다.

---

## 1. 가설 자체의 약점

### 1.1 "transformer layer를 통과하면서"가 가리키는 대상이 모호하다 (치명)

- [확인] DINO-WM의 observation model은 사전학습된 DINOv2이고 "remains frozen during training and testing"이다. 출력은 patch embedding z_t ∈ R^{N×E}다 (DINO-WM p.4, 3.1.1).
- [확인] transition model은 ViT를 decoder-only로 바꾼 것이다. 프레임 단위 causal attention을 쓰고, action과 proprio는 각 patch에 concat한다 (DINO-WM p.4-5, 3.1.2).
- [확인] 학습 손실은 teacher forcing 하의 L_pred = ||p_θ(enc(o_{t-H:t}), φ(a)) − enc(o_{t+1})||² 다 (DINO-WM Eq.1).

따라서 해석은 셋으로 갈린다.

| 해석 | 측정 위치 | 문제점 |
|---|---|---|
| (a) DINOv2 내부 layer에서 희석 | DINOv2 block 1..L의 중간 표현 | DINO-WM은 DINOv2를 학습시키지 않는다. 따라서 이것은 "DINO-WM의 결함"이 아니라 "DINOv2 마지막 layer 선택"의 문제다. 이 해석이 맞다면 처방도 다른 layer의 feature를 쓰는 것이 된다. 연구 질문 자체가 바뀐다. |
| (b) predictor 중간 layer에서 희석 | predictor block 1..6 의 hidden state | 중간 layer는 DINOv2 feature 공간에 있을 의무가 없다. 중간 layer에서 probe 성능이 떨어지는 것은 "계산 중"이라는 정상 현상일 수 있다. 무엇과 비교해야 "희석"인지 기준선이 없다. |
| (c) predictor 출력(예측 latent)에서 희석 | ẑ_{t+1}, 또는 rollout k step 후의 ẑ_{t+k} | 가장 의미 있는 해석이다 [추정]. 다만 이때 "희석"은 곧 "예측 오차"와 같다. 그래서 1.3절의 문제가 생긴다. |

**제안:** H1을 (c)로 고정하고, "teacher-forced 1-step"과 "open-loop k-step"을 별도의 조건으로 나눈다. (a)는 별도 가설 H1'로 분리한다.

### 1.2 "physical attribute"의 범위가 정해지지 않았다 (중요)

- 단일 프레임에서 관측 가능한 **운동학적 상태**(위치, 각도)와, 시간 맥락이 필요한 **동역학적 양**(속도), 그리고 상호작용을 통해서만 드러나는 **숨은 물리 파라미터**(질량, 마찰, 중력)는 서로 성격이 다르다.
- [확인] T2-Z는 마찰, 질량 같은 요인이 "neither directly observable nor explicitly specified"라는 점에서 출발한다. 같은 관측과 행동에 대해 다른 결과를 내도록 환경 latent를 따로 최적화한다 (T2-Z p.1-2, Fig.1-2).
- [확인] LeWM Table 4에서 joint velocity의 linear r는 LeWM 0.095, PLDM 0.115, DINO-WM 0.763이다. 단일 embedding에서 속도를 probe하는 설정 자체가 정보이론적으로 불리하다 [추정]. DINO-WM 쪽이 높게 나온 이유도 명시되어 있지 않다. motion blur 같은 시각 단서일 가능성이 있다 [추정].
- [확인] SG-JEPA는 probe에 "short latent window"를 넣는다 (SG-JEPA p.5, Eq.4). 즉 속도를 다룰 때는 창(window) 입력이 사실상 필수다.
- **결과:** 숨은 파라미터(질량, 마찰)에 대해 "DINO-WM이 보존하지 못한다"는 결과가 나와도 이는 frozen 단일 프레임 encoder에서 당연히 예상되는 것이다. H1의 증거가 되지 않는다. 가설에서 attribute 목록을 미리 고정하고, 단일 프레임 / 창 / 상호작용 필요 여부로 분류해야 한다.

### 1.3 predictor 출력에서의 "희석"은 예측 오차와 구분되지 않는다 (중요)

- predictor는 DINOv2 next-frame feature를 MSE로 회귀한다. encoder 출력에서 probe가 잘 된다면, predictor 출력에서 probe가 떨어지는 것은 정의상 예측 오차의 한 성분이다.
- 그러므로 "희석"이라는 표현이 의미를 가지려면 다음 형태여야 한다 [추정]. **"전체 feature MSE 감소량에 비해 물리량 관련 성분의 오차가 불균형하게 크다."**
- 이를 뒷받침할 메커니즘 가설은 이렇다 [추정]. patch feature 196×384 차원의 MSE는 분산이 큰 방향(배경, 외형)이 지배한다. 분산이 작은 물리 관련 방향은 손실에 거의 기여하지 않는다. 그래서 predictor가 그 방향을 덜 정확히 맞춰도 전체 MSE는 낮게 나올 수 있다. 이 메커니즘이 사실이라면 H1은 "transformer layer"의 문제가 아니라 **손실 함수 가중**의 문제다. 이것이 대안 설명 A1이다.

### 1.4 H1과 H2는 상호배타적이지 않고, 둘 다 틀릴 수 있다 (중요)

가능한 상태는 최소 네 가지다.

1. 보존 O, 활용 O: 둘 다 틀렸다. 성능 한계는 planner(CEM), 목표 비용 함수, 데이터 coverage 등 다른 곳에 있다.
2. 보존 O, 활용 X: H2다.
3. 보존 X(predictor 출력에서), 활용 판정 불가: H1이다. 정보가 없으므로 활용 여부는 물을 수 없다.
4. 부분 보존, 부분 활용: attribute마다 다르다. 실제로는 이 경우일 가능성이 높다 [추정]. 근거는 LeWM Table 4다. 같은 모델에서도 위치 r는 0.99 수준이고 yaw r는 0.1~0.3 수준이다.

H1과 H2를 "둘 중 하나"로 설계하면, 결과가 4번일 때 어느 쪽이든 지지된다고 해석할 위험이 있다. **attribute × 측정 위치 별로 따로 판정하는 표**를 미리 설계해 두어야 한다.

### 1.5 1번(둘 다 틀림)을 지지하는 기존 증거가 있다 (중요)

- [확인] LeWM Table 3 캡션에서 저자들은 TwoRoom에서 "the planning gap is not due to a less informative representation but rather to other factors such as the dynamics model or the planning procedure itself"라고 해석한다. probing으로 보존이 확인되어도 성능 차이는 다른 곳에서 온다는 저자 자신의 해석이다.
- [확인] G-JEPA Table 3: 관측 latent에 물리 상태 supervision(S)만 추가하면 PushT-Ada가 56.90에서 55.73으로 오히려 약간 떨어진다. predicted-state grounding(P)을 더해도 PushT-Ada는 79.43으로 변하지 않는다. 즉 latent에 물리 정보를 "더 넣는 것"이 planning 향상으로 바로 이어지지 않는다. 주의할 점이 있다. G-JEPA의 주요 설정은 training seed 1개이고 (p.7), 이 비교에는 nuisance invariance(I) 효과가 섞여 있다.
- [확인] G-JEPA Table 5: action prior(AD)를 더하면 Cube가 72.80에서 100으로 오른다. 이는 성능 병목이 world model 표현보다 **action 탐색**에 있을 수 있다는 근거다.

---

## 2. LeWM 대 DINO-WM 비교의 교란 요인

### 2.1 LeWM의 probing 증거는 DINO-WM에 대해 "보존됨"을 보여 준다 (치명)

- [확인] LeWM Table 1 캡션: "The strong probing performance of DINO-WM on certain properties may stem from its foundation-model pretraining".
- [확인] Table 4 캡션: "DINO-WM retains a clear advantage on dynamic and rotational properties (joint velocity, end-effector yaw)".
- [확인] 반대 방향의 예외도 있다. Table 3(TwoRoom)에서 DINO-WM의 linear r는 0.824이고 LeWM과 PLDM은 0.996이다. MLP probe로는 DINO-WM도 r=0.999다. 즉 정보는 있지만 선형으로 읽히지 않을 뿐이다.
- **함의:** "LeWM은 probing으로 확인되고 DINO-WM은 안 된다"는 배경 서술은 자료와 맞지 않는다. 확인할 수 있는 것은 "LeWM 논문이 DINO-WM 표현을 probing했고 대체로 잘 나왔다"는 사실이다.

### 2.2 LeWM이 probing한 "DINO-WM 표현"이 정확히 무엇인지 적혀 있지 않다 (중요)

- [확인] LeWM F.2는 "learned latent representations"를 probe한다고만 쓴다. DINO-WM의 경우 DINOv2 patch 전체(196×384)를 썼는지, pooling을 했는지, CLS를 썼는지, predictor 출력을 썼는지가 본문과 부록 p.20-21에 명시되어 있지 않다. 내가 읽은 범위에서 그렇다.
- 이 표는 frozen encoder 출력에 대한 결과일 가능성이 높다 [추정]. 그렇다면 predictor에 관한 H1(c)에 대해서는 **아무 정보도 주지 않는다.**
- [확인] SG-JEPA는 control 실험에서 DINO-WM에 "mean-pooled frozen DINOv2 patch features"를 쓴다 (p.5). 그런데 DINO-WM 논문 Table 2에서는 단일 global feature 계열(DINO CLS 등)이 patch feature보다 크게 떨어진다 (PushT 0.44 vs 0.90). pooling 방식만 바꿔도 "DINO-WM이 정보를 잃는다"는 결론을 인위적으로 만들 수 있다.

### 2.3 표현 차원이 다르다: linear probe 비교가 공정하지 않다 (중요)

- [확인] LeWM latent는 192차원이다 (LeWM 3.1, D). DINO-WM은 patch 196개 × E차원이다 (DINO-WM 3.1.1).
- [확인] LeWM Table 1의 수치에 이상한 점이 있다. DINO-WM의 agent location linear 결과는 **MSE 1.888±0.500인데 r=0.977**이다. LeWM은 MSE 0.052, r=0.974다. r가 거의 같은데 MSE가 36배 차이 난다.
- [확인] Table 4에서 DINO-WM block yaw linear는 MSE 4.223, r=0.176이고 end-effector yaw linear는 MSE 3.317, r=0.828이다. 정규화된 target이라면 평균 예측보다 MSE가 나쁜 셈이다.
- **해석 [추정]:** 고차원 입력에서 linear probe가 과적합했거나, 편향/스케일이 어긋나 있거나, train/test 분포가 다른 것으로 보인다. probe 정규화(ridge 강도), 입력 차원, 샘플 수가 통제되지 않은 상태에서 "r" 또는 "MSE" 하나만 보고 보존 여부를 판정하면 결론이 지표 선택에 따라 뒤집힌다.
- **제안:** 모든 모델에 같은 차원으로 축소(PCA k=192 등)한 뒤 probe한다. probe 용량과 정규화를 CV로 고르고, 무작위 초기화 네트워크 baseline을 함께 보고한다.

### 2.4 그 밖의 교란 요인 (중요)

| 요인 | LeWM | DINO-WM | 근거 |
|---|---|---|---|
| encoder 학습 | 처음부터 end-to-end, SIGReg | DINOv2 frozen (LVD 대규모 사전학습) | LeWM 3.1, DINO-WM 3.1.1 |
| encoder 크기 | ViT-Tiny, 약 5M | DINOv2 (크기는 원 논문 설정을 따름) | LeWM 3.1 |
| 표현 형태 | CLS + projector 1개, 192-d | patch token 집합 | 동일 |
| predictor 손실 | pred + λ·SIGReg, encoder로 gradient 전달 | pred만, encoder 고정 | LeWM Eq.3, Appx C.1 |
| proprioception | 없음 | 원 논문은 사용. LeWM 비교에서는 제거 | LeWM 4.1 |
| 학습 epoch | 10 epoch | 원 논문 설정 | LeWM E |
| frameskip / history | frameskip 5, history 3 (PushT) | 원 논문 설정 | LeWM D |

- LeWM은 probing 대상인 encoder가 **예측 가능성을 목표로 함께 학습된다**. 반면 DINO-WM의 encoder는 예측 목표와 무관하다. 따라서 "LeWM의 predictor는 물리량을 활용하고 DINO-WM은 못 한다"는 결과가 나와도, 원인이 predictor 구조인지 encoder와 predictor의 공동 적응인지 구분되지 않는다.
- [확인] SG-JEPA는 encoder와 predictor를 교차 실험해서 이 문제를 분리하려 했다 (Fig.5a). 그런데 논문 안에서 서술이 충돌한다. p.2에는 "Re-training the encoder leaves performance unchanged, whereas re-training the predictor degrades it. The gain is therefore associated with what the GRU predictor learns"라고 되어 있다. 반면 abstract와 p.10에는 "most of the gain comes from the encoder learning better features" 또는 "The advantage lies in the GRU-trained representation"이라고 되어 있다. 이 논문을 "encoder 대 predictor 분리"의 근거로 인용할 때는 주의해야 한다.
- [확인] LeWM이 인용한 DINOv2 사전학습 데이터 규모는 "~124M images"다 (Table 1 캡션). DINOv2 원 논문은 LVD-142M으로 알려져 있다 [미확인]. 수치를 인용할 때는 원문을 대조해야 한다.

### 2.5 환경 선택의 교란 (사소에서 중요 사이)

- [확인] LeWM의 DINO-WM 비교 결과는 환경마다 방향이 바뀐다. PushT에서는 LeWM 96 vs DINO-WM 92 (Table 5, seed 3개)이다. Cube에서는 DINO-WM이 우세하다 (Fig.6). TwoRoom에서도 DINO-WM이 우세하다.
- [확인] VoE(Fig.14): DINO-WM의 surprise는 TwoRoom과 PushT에서 teleport에 반응한다. 그러나 **색 변화(시각 교란)에도 비슷하게 반응**한다. Cube에서는 두 교란 모두에 유의한 반응이 없다.
- 이 결과는 "DINO-WM predictor가 물리 연속성을 쓰는가"에 대한 기존의 유일한 predictor 수준 증거다. 그런데 시각 교란에도 반응하므로 물리 특이성이 없다. 어느 환경을 고르느냐에 따라 H1과 H2의 판정이 달라질 수 있다.

---

## 3. 검증 방법(probing)의 약점

### 3.1 probing은 "존재"만 보여 주고 "활용"은 보여 주지 않는다 (치명, H2에 해당)

- 디코딩 가능하다는 것은 predictor가 그 방향을 계산에 쓴다는 뜻이 아니다. 비선형 MLP probe는 표현에 흩어진 어떤 정보든 끌어낼 수 있다. LeWM Table 3을 보면 MLP probe에서는 모든 모델의 r가 0.999~1.000으로 포화된다 [확인]. 이 상태에서는 모델 간 구분이 사라진다.
- H2를 검증하려면 다음 중 하나 이상이 필요하다 [제안].
  1. **제거 개입(amnesic, INLP류):** 입력 latent에서 물리량 probe 방향을 투영으로 제거한다. 그 뒤 predictor의 예측 오차나 rollout의 물리 오차가 늘어나는지 본다. 늘지 않으면 "활용하지 않음"의 증거다. 단, 정보가 비선형으로 중복 저장되어 있으면 선형 제거만으로는 불완전하다.
  2. **조작 개입:** 입력 latent를 probe 방향으로 Δ만큼 옮긴다. 그 뒤 예측 ẑ_{t+1}의 probe 값이 물리적으로 일관되게 바뀌는지 본다. 예를 들어 위치를 옮기면 다음 위치도 같이 옮겨져야 한다.
  3. **counterfactual 입력:** 시뮬레이터에서 물리량 하나만 다른 초기 상태 쌍을 만들어 predictor 출력 차이를 잰다.
  4. **행동 수준:** 해당 물리량이 성공을 가르는 과제에서 planning 성공률을 측정한다.
- 외부 방법론 근거는 4절에 있다.

### 3.2 데이터 누수 위험 (중요)

- [확인] G-JEPA는 PushT의 train/val을 **window 단위**로 90/10 분할한다. OGBench Cube와 TwoRooms는 episode 단위다 (G-JEPA p.6). 같은 episode에서 나온, 시간적으로 겹치거나 인접한 window가 train과 val에 동시에 들어갈 수 있다. probe 성능이 부풀려질 위험이 있다.
- [확인] LeWM F.2에는 probe의 train/test 분할 방식이 적혀 있지 않다. 내가 읽은 범위에서 그렇다.
- [확인] PushT 데이터는 "20,000 expert episodes"다 (LeWM E). expert 궤적에서는 agent 위치, block 위치, 시간 index가 서로 강하게 상관되어 있을 가능성이 높다 [추정]. 그렇다면 probe가 물리량 자체가 아니라 "궤적 진행도" 같은 대리 변수를 읽을 수 있다.
- **제안:** episode 단위 분할을 쓴다. 가능하면 초기 상태 분포까지 겹치지 않게 나눈다. 시간 index probe와 label shuffle 같은 control task를 함께 보고한다.

### 3.3 teacher forcing과 rollout의 혼동 (중요)

- [확인] DINO-WM은 teacher forcing으로 학습한다 (DINO-WM 3.1.2). 그런데 planning은 autoregressive rollout으로 한다 (3.2).
- [확인] SG-JEPA는 rollout 예측 latent의 probe NMSE에서 GT encoded latent의 probe NMSE를 빼는 "excess error"를 정의했다 (SG-JEPA Eq.4). [확인] Fig.5c에서 teacher-forced 차이는 작고 free-rollout 차이는 크게 벌어진다.
- **함의:** H1의 "희석"이 1-step에서는 없고 rollout에서만 나타난다면, 원인은 transformer layer가 아니라 오차 누적(exposure bias)이다. 이것이 대안 설명 A2다. 측정 위치를 1-step과 k-step으로 반드시 분리해야 한다.

### 3.4 재현성 (중요)

- [확인] G-JEPA는 주요 설정마다 training seed 1개다 (p.7). [확인] LeWM은 학습 분산을 PushT에서 seed 3개로만 보고한다 (Table 5).
- [확인] SG-JEPA의 DINO-WM baseline은 원본과 다르다. "action conditioning performed in the same way as in LeWM"이다 (SG-JEPA p.5). 즉 원 DINO-WM이 아니다.
- [확인] LeWM의 DINO-WM은 proprioception을 제거했다 (4.1). 같은 "DINO-WM"이라는 이름이 논문마다 다른 모델을 가리킨다.
- 우리 실험도 다음을 config.yaml에 고정해야 한다: DINOv2 체크포인트(크기, register 유무), 사용 layer, pooling, probe 정규화, 분할 seed, 체크포인트 출처(공개 가중치인지 재학습인지).

### 3.5 과장될 수 있는 주장 (사전 경고)

- "DINO-WM은 물리를 이해하지 못한다": probe r 몇 개와 VoE 곡선만으로는 "이해"를 주장할 수 없다. 게다가 기존 자료는 오히려 보존을 지지한다.
- "transformer layer가 물리 정보를 희석한다": 모든 layer를 측정하고, 무작위 초기화 predictor와 비교하고, 1-step과 rollout을 분리하기 전에는 위치를 특정할 수 없다.
- "predictor가 활용하지 못해서 planning이 실패한다": planner, 비용 함수, 데이터 coverage를 통제해야 한다. G-JEPA Table 5에서 action prior 하나로 Cube가 72.8에서 100이 된 사례를 보면 병목이 다른 곳에 있을 수 있다.

---

## 4. 외부 방법론 근거 (5편 바깥)

- Elazar, Ravfogel, Jacovi, Goldberg. "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals", TACL 9:160-175, 2021. [확인: 웹 검색으로 서지와 요지 확인] 요지는 두 가지다. 통상적인 probing 성능은 과제 중요도와 상관되지 않는다. 속성의 효용은 그 속성을 표현에서 제거하는 인과적 개입으로 측정해야 한다. https://aclanthology.org/2021.tacl-1.10/
- Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks", EMNLP 2019. [미확인] 고용량 probe는 표현이 아니라 probe 자체가 과제를 학습할 수 있으므로 control task와 selectivity를 보고해야 한다는 주장이다.
- Ravichander, Belinkov, Hovy, "Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?", EACL 2021. [미확인] probe로 디코딩 가능한 속성이 모델의 과제 수행에 쓰이지 않을 수 있다는 주장이다.
- 위 세 편은 NLP 맥락의 연구다. world model latent에 그대로 옮겨도 되는지는 [추정]이다.

---

## 5. 가설을 날카롭게 고치는 제안

### H1 수정안 (H1-c, predictor 출력 기준)

> 같은 frozen DINOv2 feature에서, 사전에 고정한 물리량 집합 S(예: PushT의 agent xy, block xy, block angle)에 대한 probe 오차를 비교한다. 비교 대상은 GT 인코딩 z_{t+k}와 DINO-WM 예측 ẑ_{t+k}(teacher-forced k=1, open-loop k=1..K)이다. 이때 excess error(SG-JEPA Eq.4 방식)가 **전체 feature 예측 오차로부터 기대되는 수준보다** 유의하게 크다.

- 기대 수준은 이렇게 정한다. 같은 크기의 무작위 방향 부분공간에서 잰 excess error, 또는 외형 속성(색 등)의 excess error를 기준선으로 쓴다.
- **H1이 틀렸을 때 나와야 할 결과:**
  1. k=1 teacher-forced에서 excess error가 0에 가깝다. 신뢰구간이 0을 포함한다.
  2. k가 커질 때 물리량 excess error의 증가율이 무작위 부분공간의 증가율과 다르지 않다.
  3. predictor의 layer별 probe에서 마지막 layer의 성능이 입력 layer 이상으로 회복된다.
- 1이 성립하고 rollout에서만 오차가 커지면, H1을 기각하고 A2(오차 누적)로 대체한다.

### H1' (encoder 쪽, 분리해서 다룸)

> DINOv2의 마지막 layer feature는 중간 layer feature보다 S에 대한 probe 성능이 낮다.

- 이 가설은 DINO-WM의 설계 선택(사용 layer)에 관한 것이다. "희석"이 맞다면 처방은 중간 layer feature로 학습하는 것이다. 틀렸다면 마지막 layer의 probe 성능이 다른 layer와 같거나 더 높게 나와야 한다.

### H2 수정안 (활용, 개입 기준)

> predictor 입력에서 S의 선형 probe 방향을 제거하면(amnesic) open-loop 예측의 S 오차가 커진다. 단 동일 차원의 무작위 방향을 제거할 때보다 커져야 한다. 반대로 H2는 "**커지지 않는다**"는 주장이다. 즉 정보는 있지만 predictor가 그 방향에 의존하지 않는다.

- **H2가 틀렸을 때 나와야 할 결과:** 제거 개입 후 S 오차가 무작위 제거보다 유의하게 커진다. 조작 개입(Δ 이동) 후에도 예측의 S 값이 물리적으로 일관되게 따라 움직인다.
- **판정 불가 조건:** 정보가 비선형으로 중복 저장되어 있으면 선형 제거만으로 정보가 사라지지 않는다. 그래서 제거 후 MLP probe로 정보가 실제로 사라졌는지 먼저 확인해야 한다. 사라지지 않았다면 H2에 대한 결론을 내리지 않는다.

### 사전 고정해야 할 것 (hypotheses.md 성공 기준에 반영 권장)

- 물리량 목록과 분류(단일 프레임 / 창 / 상호작용 필요)
- 측정 위치(encoder layer, predictor layer, 출력, rollout step)
- probe 종류, 용량, 정규화, 분할 방식(episode 단위)
- control: 무작위 초기화 encoder와 predictor, label shuffle, 시간 index probe
- 효과 크기 기준과 seed 수(최소 3개)

---

## 6. 대안 설명 목록

- **A1 손실 가중:** patch MSE가 분산이 큰 외형 방향을 우선한다. 물리 관련 저분산 방향의 오차가 상대적으로 방치된다 [추정]. 검증: 방향별 분산과 방향별 예측 오차의 관계를 본다.
- **A2 오차 누적:** teacher forcing으로 학습하고 rollout으로 평가하는 불일치 때문이다. layer와는 무관하다. 근거: SG-JEPA Fig.5c [확인].
- **A3 planner 병목:** 표현과 predictor는 충분하고 CEM 탐색이 병목이다. 근거: G-JEPA Table 5, LeWM Table 3 캡션 [확인].
- **A4 probe 인공물:** 차원, 정규화, 누수 때문에 probe 지표가 흔들린다. 근거: LeWM Table 1과 4의 MSE-r 불일치 [확인. 원인 해석은 추정].
- **A5 관측 불가능성:** 숨은 파라미터는 단일 프레임 encoder에 원래 없다. 근거: T2-Z의 문제 설정 [확인].

---

## 7. 이 가설을 믿으려면 먼저 필요한 실험 (최대 3개)

1. **probe 기준선 재현 및 감사:** 공개된 DINO-WM 체크포인트로 PushT에서 GT DINOv2 feature를 probe하고, LeWM Table 1과 맞는지 확인한다. 조건은 episode 단위 분할, PCA 차원 통일, ridge CV, label shuffle, 무작위 encoder control이다. 이것이 재현되지 않으면 이후 모든 비교의 바닥이 흔들린다.
2. **측정 위치 분리 실험(H1-c):** GT z_{t+k}, teacher-forced ẑ_{t+1}, open-loop ẑ_{t+k} (k=1..5), predictor 각 layer에서 같은 probe의 excess error를 잰다. 무작위 부분공간과 외형 속성을 기준선으로 쓴다.
3. **개입 실험(H2):** predictor 입력에서 물리량 방향을 amnesic하게 제거하는 경우와 무작위 방향을 제거하는 경우를 비교한다. 측정은 open-loop 물리 오차와 PushT planning 성공률이다. 제거 후 MLP probe로 정보가 실제로 사라졌는지 함께 확인한다.

---

## 확신도

- **중간.**
- 높게 보는 부분: 1.1(측정 대상의 모호함), 2.1(LeWM 자료가 DINO-WM의 보존을 지지함), 3.1(probing으로는 활용을 판정할 수 없음)은 PDF 원문에서 직접 확인했다.
- 낮추는 이유 1: LeWM이 DINO-WM의 어떤 표현을 probing했는지 원문에 명시가 없어서, 2.2와 2.3의 해석 일부는 추정이다.
- 낮추는 이유 2: Table 1과 4의 MSE-r 불일치 원인은 코드를 보지 않아 확정할 수 없다.
- 낮추는 이유 3: 5편 모두 본문 위주로 읽었고 일부 부록(G-JEPA, SG-JEPA, T2-Z의 부록)은 읽지 않았다. 그래서 분할 방식이나 probe 세부가 부록에 있을 수 있다.
- 낮추는 이유 4: 외부 probing 문헌 중 두 편은 원문을 대조하지 않았다.
