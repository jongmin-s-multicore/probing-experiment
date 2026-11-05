# 선행연구/이론 관점: DINO-WM의 physical attribute 보존 및 활용 검증 (analyst-prior)

작성일: 2026-10-30 / 관점: 선행연구·이론 / 가설은 미확정, 실험 없음.
표기: [읽음] = PDF를 직접 읽음(본문 앞부분 위주, 부록 미확인 표시), [검색 요약만] = 검색 결과 요약만 봄(원문 미확인).

## 0. 핵심 요약
1. 이미 읽은 문헌에는 H1/H2를 직접 가르는 실험이 없다. LeWM Table 1 에서 DINO-WM 쪽 표현(DINOv2 encoder 출력)은 Push-T 의 agent/block 위치·각도를 linear probe 로 r=0.977~0.997, MLP probe 로 r≈0.995~0.999 수준으로 복원한다. 즉 "frozen encoder 출력 단계의 보존"은 이미 높게 보고되어 있다. 따라서 H1 을 "encoder 단계 소실"로 읽으면 선행 근거는 반대 방향이다. H1 은 "predictor 의 layer 를 지나며 소실"로 좁혀야 의미가 있다.
2. DINO-WM 에서 transformer 는 두 곳이다. (a) frozen DINOv2 ViT encoder (우리가 학습하지 않음), (b) 학습되는 ViT predictor (patch 토큰 입력). H1 의 "transformer layer"가 어느 쪽인지 먼저 정해야 한다.
3. probing 은 "정보가 들어 있음(decodability)"만 보여주고 "사용됨"은 보여주지 않는다 (Amnesic probing 류 문헌 [검색 요약만]). 따라서 H2 는 probing 만으로는 검증 불가하고, 개입(intervention) 실험이 필요하다.

## 1. 5편 정리 [모두 읽음, 앞부분 위주]

### 1.1 DINO-WM (Zhou, Pan, LeCun, Pinto; ICML 2025; arXiv:2411.04983)
- 문제: offline trajectory 로 task-agnostic world model 을 학습하고 test-time MPC 로 planning.
- 구조 (읽음, Sec 3.1): observation model = frozen DINOv2 patch embedding z_t ∈ R^{N×E}. transition model = ViT (tokenization layer 제거, decoder-only transformer), 과거 H 프레임의 patch latent + action 을 받아 다음 프레임 latent 를 frame 단위 causal attention 으로 예측. action 은 MLP 로 K차원→임베딩 후 각 patch 벡터에 concat (proprio 도 같은 방식). 손실 L_pred = ||p_θ(enc(o_{t-H:t}), φ(a)) - enc(o_{t+1})||² (teacher forcing). decoder 는 선택사항이며 predictor 학습과 독립(decoder loss 를 predictor 에 역전파하면 성능이 떨어진다고 Appendix A.4.3 에 있다고 본문에 서술, 부록 미확인).
- planning: CEM 으로 MSE(ẑ_T, z_g) 최소화.
- 결과: Table 1 (PushT SR 0.90, Reach 0.92 등), Table 2: encoder ablation 에서 DINO patch > DINO CLS/R3M/ResNet (PushT SR 0.90 vs 0.44/0.42/0.20). 저자 해석은 "patch 표현이 공간 정보를 보존하기 때문"(가설 수준 표현).
- physical attribute 관련: probing 실험 없음(본문 기준). decoder 로 복원 LPIPS 만 보고 (Table 4).
- 한계: 저자 명시 한계는 데이터 커버리지, action label 의존, 계층적 제어 부재. predictor 가 무엇을 쓰는지는 분석 안 함.

### 1.2 LeWM / LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero; 2026; arXiv:2603.19312)
- 문제: pixel 로부터 end-to-end 로 안정적으로 JEPA 학습. 손실 = next-embedding MSE + SIGReg(Gaussian 정규화) 2항.
- 구조: ViT-tiny encoder(12 layer, hidden 192, ~5M) + 마지막 [CLS] 토큰 → MLP+BatchNorm projection 으로 z_t (192 dim). predictor 는 6 layer, 16 head, 10% dropout 의 transformer, action 은 AdaLN 으로 주입(zero-init). 즉 LeWM 은 프레임당 벡터 1개, DINO-WM 은 프레임당 patch 토큰 N개라는 큰 차이가 있다(LeWM 이 인코딩 토큰 ~200배 적고 planning 최대 48배 빠르다고 주장).
- probing (Sec 5.1, Table 1, Push-T): linear/MLP probe 로 agent location, block location, block angle 예측. MSE 와 Pearson r 보고. 예) Block angle linear: DINO-WM r=0.979, PLDM 0.745, LeWM 0.902; MLP: 0.995 / 0.972 / 0.990. Agent location linear: DINO-WM 0.977, LeWM 0.974. 본문은 DINO-WM 의 강한 probing 성능이 DINOv2 의 대규모 사전학습(~124M images) 덕분일 수 있다고 해석(저자 추정). 다른 환경 probing 은 부록 F.2 (미확인).
- 기타: decoder 로 192-dim 단일 latent 에서 장면 복원, t-SNE, temporal latent path straightening, VoE(순간이동 vs 색 변화에서 surprise). Fig 7 에서 end-effector angle 같은 세부는 완전히 보존되지 않는다고 서술.
- 주의점: Table 1 의 "DINO-WM" 행이 정확히 어떤 pooling/토큰(patch 평균? CLS?)으로 probe 했는지 본문만으로는 불명확 [미확인]. 또한 probe 대상은 encoder 출력이고 predictor 내부/출력은 아님.
- 한계(저자): 단기 horizon planning, 데이터 다양성 필요, 저차원 환경에서 SIGReg 약함.

### 1.3 G-JEPA (Man, Chen, Guo; 2026; arXiv:2610.28483)
- 문제: LeWM 계열이 control-relevant physical info 를 명시적으로 우선하지 않음. 해결: causal S4D 모듈로 observed latent 와 predicted latent 시퀀스 양쪽에서 physical state 를 회귀(L_state, L_pred-state), nuisance invariance(L_inv)를 추가, 그리고 diffusion action prior + CEM planning.
- 서술 중 중요한 대목(Intro): "freezing the encoders prevents interaction data from shaping representations around task-specific physical structure" (DINO-WM/V-JEPA 2-AC 를 겨냥). 또 PhyLatent, PSG-JEPA 를 physical supervision 선행으로 인용 (원문 미확인, G-JEPA 본문 인용만 읽음).
- 결과 (Table 1/3): PushT-Ada 에서 LeWM 56.90 → physical grounding(S+I+P) 79.43, 최종 87.11 (AD 포함). Table 3 에서 S 만 쓰면 PushT 55.73 으로 오히려 개선 없음, S+I 에서 79.43 으로 상승 → 정보 주입 자체보다 invariance 와 결합이 중요했다는 신호(해석은 추정). 이 논문은 DINO-WM 과 직접 비교는 하지 않음(표에 DINO-WM 없음).
- 한계: LeWM 기반이며 DINO-WM 구조 분석 아님. physical state 를 supervision 으로 주는 것은 "보존/활용 진단"이 아니라 "개선책".

### 1.4 SG-JEPA / Semigroup-JEPA (Liu, Sun, Baker, Balestriero, Sous; 2026; arXiv:2609.10464)
- 문제: 장기 rollout 에서 물리 법칙을 학습했는가, gravity OOD 로 일반화하는가. LeWM 에 gravity 를 action 에 concat 하고, K-step rollout loss(역전파를 encoder 와 predictor 양쪽에) + SIGReg.
- DINO-WM 비교 방식(Sec 2.2): DINO-WM 은 pretrained DINOv2 encoder 를 freeze 하고 LeWM 과 같은 방식의 action-conditioned transformer predictor 를 학습(즉 원 논문 구현과 약간 다를 수 있음). control 에서 DINO-WM feature 는 mean-pooled frozen DINOv2 patch.
- 평가: frozen world model 위에 MLP probe 로 position/velocity/rotation 예측, rollout 시 "excess physical-state error"(식 4: rollout latent probe 오차 - 인코딩 latent probe 오차)로 rollout 에 의한 증가분만 분리. 이 지표는 우리 H2 검증에 직접 차용 가능.
- 결과: 2D square 에서 SG-JEPA 가 DINO-WM 대비 오차 31~48% 감소, Original LeWM 은 DINO-WM 근처. 3D Approach Ball 은 DINO-WM 대비 ~34% 감소; DINO-WM 은 짧은 horizon 에서는 잘하지만 horizon 이 늘면 오차가 누적(Fig 4a). control SR 은 Arm Catcher 9.5% (DINO-WM) vs 23.3% 등.
- 핵심 메커니즘 서술(초록/서론): "how well the encoder preserves dynamical information determines the ability to generalize"; encoder 만 재학습해도 성능 불변인데 predictor 재학습은 성능을 저하시키는 교체 실험(주의: 서론 설명이 서로 방향이 다르게 읽힐 수 있어 부록 확인 필요 [미확인]). 초록에서는 "gain 대부분이 predictor 가 더 나은 dynamics 를 배운 것이 아니라 encoder 가 predictor 가 carry forward 할 feature 를 더 잘 보존하는 데서 온다"고 요약.
- 시사점: 이 논문이 우리 H1/H2 와 가장 가깝다. "rollout 에 의한 정보 소실"과 "encoder 의 feature 보존"이 성능을 가른다는 주장이다. 다만 DINO-WM 을 고정 encoder 로 두고 predictor 만 학습시킨 변형이며, DINO-WM 내부 layer 분석은 없다.
- 한계: 합성 MuJoCo gravity 환경, DINO-WM 원 구현 아님, PushT 등 표준 benchmark 아님.

### 1.5 T2-z (Zhou, Hong, Huang, ... Choi; 2026; arXiv:2610.22364)
- 문제: 마찰, 질량 등 관측되지 않는 environment factor 를 world model 이 반영하지 못함. 해법: 학습 가능한 environment latent z_e 를 world model 입력으로 추가하고, 학습 시 world model / z_e 를 번갈아 최적화, test time 에는 model 을 freeze 하고 z_e 만 최적화.
- 백본은 Wan2.2-TI2V-5B 기반 BWM (flow matching 비디오 모델)으로 DINO-WM 계열이 아님.
- 결과: Table 1 에서 Standard World Model 대비 평균 physical error 12.86→4.80, action success 37.9%→74.2% (simulation 7 setting). Friction 에서 Standard 32.0% vs T2-z 72.0%. 학습된 latent 의 PCA 가 mass/friction 등 요인 구조를 반영.
- 시사점: "표준 world model 은 관측으로 구분 불가한 물리 속성(마찰, 질량)을 입력만으로는 구분하지 못한다"는 증거. 즉 이 종류의 attribute 는 정적 이미지 patch 에 애초에 안 들어 있어서 H1/H2 와 다른 문제(관측 불가능성)다. 우리 실험에서 attribute 선택 시 "단일 프레임에서 관측 가능한가 / 시간 정보가 필요한가 / interaction 으로만 드러나는가"를 구분해야 한다는 근거가 된다.
- 한계: 비디오 생성 모델 기반, test-time 최적화 필요, DINO-WM 과 비교 없음.

## 2. 추가 선행연구 (WebSearch)

| 주제 | 문헌 | 상태 | 요지/관련성 |
|---|---|---|---|
| frozen foundation feature 의 physical property | El Banani 계열이 아님. Zhan et al., "A General Protocol to Probe Large Vision Models for 3D Physical Understanding", NeurIPS 2024, arXiv:2310.06836 | 검색 요약만 | 여러 transformer layer 의 frozen feature 에 linear SVM 을 붙여 scene geometry, material, support relation, shadow, occlusion, depth 를 probe. DINOv2 와 Stable Diffusion 이 강함. layer 별로 probe 한다는 점이 우리 설계와 같음. |
| patch 표현 비교 | arXiv:2401.00463 (patch representation 평가 논문, 정확한 제목/저자 미확인) | 검색 요약만 | DINO 계열이 patch 표현을 별도 튜닝 없이 쓰기 좋다는 요약. 인용 전 원문 확인 필요. |
| JEPA 의 intuitive physics | Garrido et al., "Intuitive physics understanding emerges from self-supervised pretraining on natural videos", 2025, arXiv:2502.11831 | 검색 요약만 | V-JEPA 가 IntPhys 등에서 VoE 로 zero-shot 물리 이해. probing 이 아니라 surprise 기반이며 LeWM 의 VoE 와 같은 계열. |
| 물리 파라미터 식별 가능성 | Tan et al., "What Can Latent World Models Know? Physical Parameter Identifiability in Multimodal Predictive Representations", 2026, arXiv:2607.27017 | 검색 요약만 | 숨은 mass/drag/stiffness 가 episode 마다 바뀌는 POKEWORLD 에서 X-JEPA 를 학습. objective 구조가 latent 가 어떤 파라미터를 얻는지 결정, 데이터 증가는 이미 얻은 파라미터만 개선. LeWM 에서 이어지는 probing 계열. |
| PSG-JEPA | Yan et al., "Is Forward Prediction Enough? Physical State Grounding for JEPA World Models", 2026, arXiv:2608.06799 | 검색 요약만 | proprioceptive state 로 latent 를 grounding 하여 probing/planning/policy 개선 보고. |
| rank collapse | Dong, Cordonnier, Loukas, "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth", ICML 2021, arXiv:2103.03404 | 검색 요약만 | skip connection 이 없는 순수 self-attention 은 깊이에 따라 rank-1 로 수렴. skip 이 가장 큰 완화 요인, MLP 도 일부, LayerNorm 은 완화 못 한다고 보고(다른 논문과 상충 소지 있음). 실제 ViT/predictor 는 skip 이 있으므로 이 이론이 그대로 H1 의 근거는 되지 않음(추정). |
| layer 별 정보 | Skean et al., "Layer by Layer: Uncovering Hidden Representations in Language Models", ICML 2025, arXiv:2502.02013 | 검색 요약만 | 언어모델에서 중간 layer 가 마지막 layer 보다 downstream 성능이 최대 16% 높은 경우가 많음. 비전 수치는 미확인. 마지막 layer 가 최선이 아닐 수 있다는 일반 근거. |
| probing 의 한계 | Elazar et al., "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals", arXiv:2006.00995 | 검색 요약만 | probing 성능과 task 에서의 실제 사용(중요도)은 상관되지 않을 수 있음. 정보 제거(counterfactual)로 사용 여부를 검증. |
| probe 통제 | Hewitt & Liang, control tasks / selectivity (2019 로 알고 있음) | 검색 요약만(원문·정확 서지 미확인) | 임의 label control task 로 probe 의 암기 능력을 분리. |

검색에서 "DINO-WM 의 predictor 가 무엇을 쓰는지 분석한 독립 재현 연구"는 찾지 못했다. "world model 이 latent 의 attribute 정보를 무시/복사 shortcut" 연구도 찾지 못했다(없다는 뜻이 아니라 이번 검색에서 미발견).

## 3. 이론적 정리: H1/H2 를 구분하는 틀

### 3.1 가설 정의 정밀화 (제안)
DINO-WM 파이프라인: o_t → [frozen DINOv2 L_enc layers] → z_t (N×E patch) → [action concat] → [predictor L_pred layers] → ẑ_{t+1}.
- H1a: DINOv2 layer 를 지나며 소실 (encoder 단계). 선행 근거: LeWM Table 1 은 DINO-WM 쪽 표현의 높은 probing 을 보고 → H1a 는 약함. 단 attribute 종류와 probe 방식에 따라 다를 수 있음(Block angle linear 0.979 < MLP 0.995, 즉 일부는 비선형).
- H1b: predictor layer 를 지나며 소실. 선행 근거: 직접 보고 없음. SG-JEPA Fig 4a 의 horizon 에 따른 DINO-WM 오차 증가는 rollout 누적 소실과 일치하나 인과 구분은 안 됨(추정).
- H2: ẑ 에는 attribute 정보가 남아 있으나(또는 입력에 있으나) predictor 의 출력/예측이 이에 반응하지 않음. 선행 근거: SG-JEPA 초록의 "encoder 가 preserve 한 feature 가 predictor 가 carry forward 하는 feature" 해석은 오히려 "보존 ↔ 활용" 이 연결된 것으로 보는 입장. G-JEPA 의 Table 3 에서 상태 supervision 만 추가(S)했을 때 PushT 개선이 없었다는 점은 "정보 주입이 곧 활용"이 아님을 시사(해석은 추정).

### 3.2 구분 검증 설계 (제안, 실험 아님)
공통: probe 는 linear 와 MLP 모두, train/val 를 episode 단위로 분리, Hewitt & Liang 식 control task(shuffled label)로 selectivity 보고, attribute 마다 "단일 프레임 관측 가능 / 시간 정보 필요 / 관측 불가"로 분류(T2-z 근거).

(A) 보존(H1) 측정: layer-wise probing profile
1. DINOv2 의 중간 layer ~ 최종 layer patch 토큰 probe (Zhan et al. 프로토콜과 유사). 최종 layer 가 중간보다 낮으면 H1a 신호(Skean et al. 의 일반 경향 근거).
2. predictor 입력(action concat 직후), 각 predictor block 의 hidden state, 출력 ẑ_{t+1} 를 같은 probe 로 평가. probe 대상: Push-T 의 agent/block 위치와 각도 (LeWM 과 동일하게 해서 직접 비교 가능), 속도 등 시간 attribute.
3. 기준선: (i) 같은 probe 를 ground-truth z_{t+1}(encoder 출력)에 적용한 상한, (ii) randomly initialized predictor 통과 후 probe(구조적 정보 손실 하한), (iii) LeWM 의 Table 1 수치.
4. 보조 진단: layer 별 token 간 cosine similarity, effective rank(over-smoothing/rank collapse 지표, Dong et al. 의 정의와는 다를 수 있음, 지표 정의는 따로 확정 필요). 단 skip connection 때문에 이론상 collapse 가 크지 않을 것으로 추정 → 측정으로 확인.
5. 해석: probe 정확도가 layer 를 따라 단조 감소하면 H1b 지지. 거의 일정하면 H1 기각.

(B) 활용(H2) 측정: probing 이 아닌 개입
1. SG-JEPA 식 excess error: rollout 한 ẑ 의 probe 오차 - ground truth z 의 probe 오차를 horizon 별로 보고. horizon 에 따라 증가하면 predictor 가 attribute 를 유지 못 한다는 뜻 (H1b 또는 H2 와 혼재).
2. 입력 개입: 입력 latent 에서 attribute 를 담는 방향(probe 방향 또는 해당 patch 토큰)만 교체/제거(amnesic 식 INLP 류)하고 다음 상태 예측과 planning 성능 변화를 본다. 제거해도 예측이 불변이면 predictor 가 해당 정보를 사용하지 않음(H2 지지). 제거 시 예측 오차가 증가하면 사용 중.
3. 반사실(counterfactual) 쌍: 동일 장면에서 block angle 만 다른 두 입력을 같은 action 으로 예측시켜 출력 차이가 정답 상태 차이와 일치하는지(출력에서 probe 로 복원) 확인.
4. 단순 baseline: "마지막 프레임 복사"(identity) 예측기와 비교. predictor 가 identity 에 가깝게 학습됐다면 작은 attribute 변화를 활용하지 않았을 가능성(추정, 이 shortcut 을 다룬 world-model 문헌은 미발견).
5. action ablation: action 을 0/랜덤으로 바꿨을 때 예측 변화. action 활용 여부 자체가 낮다면 H2 의 일종.
6. planning 수준: goal 프레임에서 attribute 만 바뀐 경우 MSE cost 가 이를 반영해 CEM 이 구분하는지.

(C) 결과별 해석 표
| (A) 보존 | (B) 활용 | 결론 |
|---|---|---|
| layer 따라 감소 | - | H1b 지지 (H1a 인지 H1b 인지 구분) |
| 유지 | 개입에 둔감 | H2 지지 |
| 유지 | 개입에 민감, 그러나 rollout 오차 누적 | 둘 다 아님: 누적 오차 문제(SG-JEPA 의 rollout loss 논지) |
| encoder 출력부터 낮음 | - | H1a 이며 LeWM Table 1 과 불일치 → probe 설정(pooling, 데이터) 차이 점검 |

### 3.3 위험·교란 요인 (선행연구에서 본 것)
- DINO-WM 구현은 논문별로 다르다: 원 논문 구현은 patch 토큰 + proprio, SG-JEPA 는 LeWM 방식 action conditioning 으로 재구현. LeWM 은 proprio 를 제외했다고 서술. 비교 시 프로토콜 일치 필요.
- LeWM 의 DINO-WM probing 은 encoder 출력 기준이라, 우리 결과와 직접 비교하려면 같은 데이터와 pooling 으로 재현해야 함.
- probe capacity: 정보가 선형으로 안 읽히는 것과 소실을 혼동하기 쉽다. LeWM Table 1 에서도 linear 와 MLP 간 격차(block angle 에서 큼)가 존재.
- decoder reconstruction(DINO-WM Table 4)이 잘 되는 것은 보존의 증거이지만 "미세한 physical attribute" 보존의 증거는 아님.

## 4. 연구 공백
1. DINO-WM predictor 내부 layer 별 probing 이 문헌에 없음(이번 조사 범위에서 미발견). LeWM, SG-JEPA 는 encoder 출력 또는 rollout 결과 probe 만 사용.
2. "보존"과 "활용"을 분리한 world-model 연구가 없음. SG-JEPA 의 교체 실험(encoder vs predictor 재학습)이 가장 가깝지만 LeWM 계열이며 attribute 사용 여부를 개입으로 보지 않음.
3. frozen 대형 encoder 의 patch feature 가 Push-T 같은 조작 환경에서 어떤 physical attribute(속도, 질량, 마찰, 접촉)를 담는지 체계적 조사가 없음. 위치/각도 외 속성은 DINO-WM 쪽에 데이터 없음.
4. 관측 불가 attribute(T2-z 의 마찰, 질량)와 관측 가능 attribute 를 구분한 DINO-WM 계열 평가가 없음.

## 5. 이 팀이 검증해 볼 만한 후보 가설 (확정 아님, 사람 승인 필요)
- H1': DINO-WM 의 학습된 predictor 의 layer 를 지나며 단일 프레임 observable attribute(위치/각도)에 대한 probe 성능이 입력 대비 유의하게 감소한다. 성공 기준 예: 최종 layer 의 probe r 이 입력 대비 상대적으로 사전에 정한 임계(예 0.05) 이상 감소 [임계는 팀 합의 필요].
- H1'': frozen DINOv2 encoder 출력의 attribute 보존은 LeWM 의 학습형 encoder 와 대등(LeWM Table 1 과 일치)하나, 시간 의존 attribute(속도, 접촉 상태)는 단일 프레임에서 관측 불가라 모델 전체에서 못 얻는다. (H1 이 부분적으로 틀렸을 가능성)
- H2': 입력에서 attribute 방향을 제거(amnesic 개입)해도 predictor 출력 오차와 planning SR 의 변화가 미미하다 (사용 안 함). 대조군으로 같은 크기의 random direction 제거를 둔다.
- H3 (rollout 누적): 1-step 에서는 보존/활용이 양호하나 horizon 이 늘수록 excess probe error 가 증가한다 (SG-JEPA 와 같은 방식 지표).
- H4: predictor 를 identity/copy 대비로 평가했을 때 개선분이 attribute 변화량과 무관하다 (copy shortcut).

## 6. 출처 목록
- [읽음] DINO-WM: Zhou, Pan, LeCun, Pinto, ICML 2025, arXiv:2411.04983. 파일 notes/research/papers/worldmodel_jepa/DINO-WM.pdf (1~8쪽)
- [읽음] LeWM: Maes et al., 2026, arXiv:2603.19312. LeWM.pdf (1~10쪽, 부록 미확인)
- [읽음] G-JEPA: Man, Chen, Guo, 2026, arXiv:2610.28483. G-JEPA.pdf (1~8쪽)
- [읽음] SG-JEPA: Liu, Sun, Baker, Balestriero, Sous, 2026, arXiv:2609.10464. sg-jepa.pdf (1~7쪽, 부록 미확인)
- [읽음] T2-z: Zhou et al., 2026, arXiv:2610.22364. T2-z.pdf (1~7쪽)
- [검색 요약만] Zhan et al. 2024, arXiv:2310.06836, https://arxiv.org/abs/2310.06836
- [검색 요약만] arXiv:2401.00463, https://arxiv.org/html/2401.00463v2 (제목/저자 미확인)
- [검색 요약만] Garrido et al. 2025, arXiv:2502.11831, https://arxiv.org/pdf/2502.11831
- [검색 요약만] Tan et al. 2026, arXiv:2607.27017, https://www.alphaxiv.org/abs/2607.27017
- [검색 요약만] Yan et al. 2026 (PSG-JEPA), arXiv:2608.06799, https://arxiv.org/pdf/2608.06799
- [검색 요약만] Dong, Cordonnier, Loukas, ICML 2021, arXiv:2103.03404, https://arxiv.org/pdf/2103.03404.pdf
- [검색 요약만] Skean et al., ICML 2025, arXiv:2502.02013, https://arxiv.org/html/2502.02013v2
- [검색 요약만] Elazar et al., Amnesic Probing, arXiv:2006.00995, https://arxiv.org/abs/2006.00995
- [검색 요약만] Hewitt & Liang control tasks: 서지 미확인(검색 요약에서 개념만 확인)

## 7. 확신도
- 5편 구조/probing 내용 정리: 높음 (원문 직접 읽음). 단 LeWM 부록(F.2), DINO-WM 부록, SG-JEPA 부록은 읽지 않아 해당 세부(예: probe 가 쓴 토큰, 교체 실험 상세)는 미확인.
- 추가 문헌 서술: 낮음~중간 (검색 요약만, 원문 미확인). 특히 arXiv:2401.00463 과 Hewitt & Liang 서지는 인용 전 확인 필요.
- 구분 검증 설계 및 후보 가설: 중간. 문헌 근거는 개념적이며(probing 은 사용을 보장하지 않음, rollout excess error 지표 존재), DINO-WM 에 직접 적용한 선례는 없어 실제 결과 예측은 추정.
- H1a 가 약하다는 판단: 중간. LeWM Table 1 한 건에 의존하며 DINO-WM 행의 probe 조건이 불명확함.
