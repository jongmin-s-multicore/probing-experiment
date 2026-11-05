# 선행연구 조사: DINO-WM latent 의 물리 정보 probing + OOD 물리 파라미터(천 크기) 붕괴

작성: analyst-prior (선행연구/이론 관점), 작성일 2026-10-31
범위 주의: 웹 검색 결과(초록/스니펫/요약 페이지)와 로컬 PDF 1~2쪽만 확인했다. 전문을 읽지 못한 논문은 "미확인"으로 표시했다. 2026년 preprint 들은 수치가 버전/요약에 따라 다를 수 있어 인용 전 원문 확인이 필요하다.

## 0. 동기 문제 요약과 데이터셋 확인
- acwm-phys 는 실재한다. ACWM-Phys: Investigating Generalized Physical Interaction in Action-Conditioned Video World Models (Georgia Tech, arXiv 2605.08567, 2026-05). 8개 시뮬 환경, 15k+ 궤적. HF: huggingface.co/datasets/t1an/ACWM-Phys, GitHub: xavihart/ACWM-Phys-dev. (arXiv abs/HTML 요약 확인. 데이터셋 페이지 자체는 미확인. 우리 팀이 쓰는 버전과 같은지도 미확인)
- Cloth Move: 양팔로 천을 고정된 구 위로 미는 과제. OoD split 은 "천 크기와 초기 배치 변경". 논문은 학습/OoD 의 구체적 크기 수치를 주지 않는다(질적 서술만). 학습 1,987 / InD test 100 / OoD test 100 궤적, 행동 3차원(공유 end-effector 변위), horizon 64 (논문 Table 8).
- 중요: 이 benchmark 의 기본 요인은 "크기와 초기 구성이 동시에 바뀜"이라 크기 단일 효과는 분리되지 않는다 (확인된 사실: 두 가지가 함께 바뀐다고 서술됨).

## 1. 핵심 논문별 정리

### 1.1 ACWM-Phys (benchmark, 직접 관련)
- 출처: ACWM-Phys, 2026, arXiv 2605.08567. https://arxiv.org/abs/2605.08567 / https://arxiv.org/html/2605.08567
- 문제: 행동조건 비디오 world model 이 물리 상호작용 OoD 에 일반화하는가.
- 방법: ACWM-DiT(비디오 diffusion transformer)를 InD 학습, OoD test.
- 결과(Cloth Move, Table 1, 요약 페이지 기준): InD MSE 10.667 / OoD 23.820 (x1e-3), M-MSE 63.68 -> 93.67, SSIM 0.920 -> 0.864. 모델 크기를 키우면 OoD MSE 감소(22.4 -> 18.9, 50k step). 행동을 3-D 공유 -> 팔별 8-D 로 바꾸면 OoD MSE 23.8 -> 6.9.
- 실패 서술: OoD 천 크기에서 "hallucinates incorrect cloth extents"(천 범위를 잘못 만들어냄), 구 주변 변형 경계 오배치. 저자 해석은 "외형 통계에 의존, 물리 법칙 미학습".
- 한계/우리와의 연결: DINO-WM 이 아니라 pixel-space DiT 이다. 그래도 "크기 OoD 에서 extent 를 잘못 예측"은 우리가 본 "작은 크기로 회귀"와 같은 계열 현상일 수 있다(추정). 저자들은 probing 을 하지 않았다. 행동 차원이 OoD 에 영향을 준다는 결과는 우리에게 의미 있는 confound 후보다(DINO-WM 에서도 action 설계가 영향을 줄 수 있음, 추정).

### 1.2 DINO-WM (베이스라인)
- 출처: Zhou, Pan, LeCun, Pinto, DINO-WM, ICML 2025, arXiv 2411.04983. https://arxiv.org/abs/2411.04983 (로컬 PDF 있음: notes/research/papers/worldmodel_jepa/DINO-WM.pdf, 이번에 정독하지는 않음)
- 방법: 동결된 DINOv2 의 patch feature 를 인코더로 쓰고, ViT predictor 가 오프라인 궤적에서 다음 patch feature 를 예측. 픽셀 복원 없음. 계획은 test-time 최적화(CEM/GD).
- 핵심 구조적 사실(원리상 확실): 인코더가 동결이므로 "물리 정보가 latent 에 있는가"는 (a) DINOv2 가 크기 등 물리 단서를 patch 에 보존하는가, (b) predictor 가 학습한 동역학이 그 단서를 사용/보존하는가로 분해된다. 크기는 patch 위치/개수로 인코딩되므로 인코더 단계의 정보 손실 가능성은 낮을 것으로 추정(미검증). 반면 predictor 가 학습분포(작은 천)의 prior 로 끌려가는 쪽이 더 의심된다(추정).
- 논문 주장: 미지 maze/물체 형상 등에서 일반화(초록 기준). 천 크기 같은 연속 물리 파라미터 OoD 는 평가 대상 아님(내가 확인한 범위에서). Rope/Granular 같은 deformable 환경은 있음.
- 한계: 물리 파라미터 probing/OoD 파라미터 분석 없음(초록 기준, 미확인 부분 있음).

### 1.3 Semigroup-JEPA (SG-JEPA) -- 물리 OoD 일반화, 로컬 PDF 1~2쪽 확인
- 출처: Liu, Sun, Baker, Balestriero, Sous, arXiv 2609.10464 (2026-09). 로컬: notes/research/papers/worldmodel_jepa/sg-jepa.pdf
- 문제: 중력 값을 학습 범위 밖으로 바꿔도 latent dynamics 가 맞는가.
- 방법: LeWM 에 물리 파라미터(중력 g)를 predictor 조건으로 주고, K-step 자기회귀 rollout loss + SIGReg 로 encoder/predictor 공동 학습.
- 결과(초록): DINO-WM 대비 open-loop 예측 오차 최대 2x 감소(2D), 3D 로봇 제어 성공률 최대 2.5x. 이득 대부분이 predictor 가 아니라 encoder 가 dynamics 에 필요한 feature 를 보존하는 데서 온다(encoder 재학습은 성능 불변, predictor 재학습은 하락한다는 서술도 있음. 이 두 서술의 정합성은 정독 필요).
- 연결: (i) DINO-WM 이 물리 OoD 에서 약하다는 비교 근거, (ii) 파라미터를 조건 입력으로 주는 방향, (iii) multi-step rollout loss 가 오차 누적/붕괴를 줄인다는 주장. 단, 동결 DINOv2 인코더는 (iii) 의 encoder 쪽 이득을 얻을 수 없다(추정). 천 크기를 "조건"으로 줄 수 있느냐는 별도 문제(크기는 이미지에서 관측 가능한 양).
- 한계: 대상이 중력(스칼라 법칙 파라미터)이고 deformable 은 아님. 전문 미확인.

### 1.4 G-JEPA -- 물리 상태 supervision (로컬 PDF 1~2쪽 확인)
- 출처: Man, Chen, Guo, Exploring Temporal Context-Aware Physical Grounding in Latent World Models, arXiv 2610.28483 (2026-10). 로컬: .../G-JEPA.pdf
- 방법: latent 에 물리 상태 supervision(shared causal S4D 모듈)을 붙여 관측/예측 시퀀스에서 물리 상태를 추정하게 함. PushT, OGBench Cube 에서 planning 개선.
- 연결: "latent -> 물리량 head" 를 학습 보조손실로 쓰는 접근. probing 이 음성이면 이런 보조 supervision 이 후보 해결책. 인용된 PhyLatent(Zeng et al., 2026), PSG-JEPA(Yan et al., 2026)도 같은 계열이나 이번에 미확인.
- 한계: OOD 파라미터 일반화 평가 아님(초록 기준). 전문 미확인.

### 1.5 T2-Z 와 LeWM (로컬, 이번에 읽지 않음)
- LeWorldModel (Maes et al., 2026): end-to-end JEPA, SIGReg 사용(1.3, 1.4 의 인용으로만 확인). 미확인.
- T2-Z: Test-Time Optimization of Environment Latents for Robotic World Models. 환경 latent 를 test-time 에 최적화한다는 제목만 확인. 우리 문제에서 "환경 latent 를 추론해 OOD 파라미터에 적응"하는 방향과 연결될 가능성(추정). 내용 미확인.

## 2. Probing 방법론 선행연구

### 2.1 Video foundation model 물리 probing
- How Do Video Foundation Models Encode Intuitive Physics? Probing Across Pretraining Paradigms, Punzo et al., arXiv 2606.09646 (v2 2026-09). https://arxiv.org/abs/2606.09646
  - V-JEPA, VideoMAE, LTX-Video 의 동결 feature 를 IntPhys2, MVP 에서 probing. V-JEPA 가 가장 좋고, 물리 정보는 초기 층에서 가장 약하고 중~후반 층에서 접근 쉬움. 프레임 순서를 섞으면 성능 하락. 검색 스니펫에 따르면 V-JEPA 계열은 linear probe 보다 MLP/attentive probe 에서 크게 오름(수치 43-48% -> 72-88% -> 93%+ 는 2차 요약이므로 원문 확인 필요).
  - 시사점: probe 가 linear 하나만이면 음성 결과가 "정보 부재"가 아니라 "선형 접근 불가"일 수 있다. DINO-WM 에서는 patch 토큰 전체를 입력받는 attentive/MLP probe 를 함께 써야 하고, 층(predictor 블록 깊이)별 probing 도 해야 한다.
- Intuitive physics understanding emerges from self-supervised pretraining on natural videos, Garrido et al., 2025, arXiv 2502.11831 (V-JEPA 의 violation-of-expectation 평가). 검색 요약만 확인, 전문 미확인.
- 위 둘은 "관찰 가능한 직관물리/위반 탐지"이고, 연속 파라미터(크기, 질량) 회귀 probing 은 다음 항목.

### 2.2 latent world model 의 물리 파라미터 식별가능성
- What Can Latent World Models Know? Physical Parameter Identifiability in Multimodal Predictive Representations (POKEWORLD), Tan et al., arXiv 2607.27017 (2026-07). https://www.alphaxiv.org/abs/2607.27017
  - 숨은 질량/drag/접촉 강성을 가진 물체를 찌르는 환경. 학습된 latent 에서 파라미터를 linear probe R^2 로 읽음. 결론: 입력이 알 수 있는 한계를 정하고, 예측 목표가 latent 에 무엇이 남는지 결정한다. 접촉 강성 R^2 약 0.40-0.57(소스 간 불일치), drag 는 예측 목적만으로는 약 0.13 에서 정체, 같은 trunk 의 supervised head 는 0.45 (모두 요약 페이지 기준, 미검증).
  - 시사점(가장 직접적): "예측 loss 만으로 학습한 latent 는 예측에 불필요한 파라미터를 버린다". 크기는 외형으로 관측되므로 이 논문의 숨은 파라미터보다 쉬운 경우다. 그러나 "관측 가능한 양이라도 predictor 가 학습분포 prior 를 쓰면 rollout 에서 정보가 사라진다"는 별도 가설은 이 논문이 직접 다루지 않는다(추정).
- The Observer Effect in World Models: Invasive Adaptation Corrupts Latent Physics (PhyIP), arXiv 2602.12218 (2026-02). https://arxiv.org/abs/2602.12218. 동결 표현 + linear probe 로 유체/궤도 물리를 평가하고, fine-tuning 같은 침습적 적응이 latent 물리를 훼손한다고 주장(검색 요약 기준, 전문 미확인). 시사점: probing 은 동결 표현을 건드리지 않는 비침습 방식이어야 한다.
- CALIPER (preprint, 제목/저자/ID 미확인): 동결 feature 의 linear readout 으로 질량/마찰 기반 슬라이딩 거리 예측. 요약상 linear probe 는 거의 추적 못하고 프레임 aggregation 방식의 영향이 pretraining 보다 크다고 함. 출처 불명확하므로 참고만.

### 2.3 일반 probing 방법론 (내 지식, 이번 검색으로 미확인)
- Alain & Bengio 2016 (linear classifier probes), Hewitt & Liang 2019 (control task / selectivity: probe 용량이 크면 probe 자체가 과제를 학습). 위 2.1 의 "linear 는 과소평가, 고용량은 과대평가" 양면을 보완해서 쓸 수 있는 표준 대조 장치. 이번 세션에서는 URL 검증을 하지 않았다.

## 3. OOD 물리 일반화/학습분포 회귀 관련 이론 (원인 후보)
아래는 선행연구로 직접 확인된 것이 아니라, 위 문헌을 바탕으로 한 원인 가설 정리다. 모두 "추정".
1. predictor prior 로의 수축(regression to training prior): 회귀형(MSE) latent 예측은 불확실한 입력에서 학습분포 평균으로 수축한다. 천 크기는 학습 범위 밖이면 prior 쪽으로 끌려갈 수 있다. ACWM-Phys 의 "hallucinated cloth extent"와 정합적.
2. 오차 누적/자기회귀 drift: SG-JEPA 는 multi-step rollout loss 없이 teacher forcing 으로 학습하면 장기 rollout 이 OOD 에서 오차 증폭한다고 주장(초록 기준). DINO-WM 도 teacher-forcing 학습이므로 해당(DINO-WM 학습 방식은 구현 확인 필요).
3. 동결 인코더 한계: 인코더가 크기 단서는 보존하지만 predictor 가 그것을 상태 변수로 쓰도록 학습되지 않았을 수 있다(크기가 학습에서 상수이므로 shortcut). 학습 데이터에서 크기 분산이 0 이면 predictor 에게 크기를 보존할 압력이 없다 -> 필연적 붕괴 가능(추정, 가장 유력하다고 봄).
4. 행동 표현/조건 결합: ACWM-Phys 에서 action 차원 변경이 OoD 에 큰 영향. action 이 천 상태 변화를 과도하게 설명하는 shortcut 가능성(추정).
5. 시스템 식별/적응 계열 (내 지식, 미검증): RMA (Kumar et al., 2021), UP-OSI(Yu et al., 2017)처럼 환경 파라미터를 context/history 로 추론해 조건화. DINO-WM 에서는 context frame 에서 크기 latent 를 추출해 predictor 에 주는 방식에 대응. T2-Z 와도 유사 방향. 정확한 서지는 미확인.
6. Cloth/deformable 시뮬 world model (내 지식, 미검증): MeshGraphNets (Pfaff et al., 2021), DPI-Net (Li et al., 2019)은 구조적 inductive bias(mesh/particle)로 크기 일반화를 얻는다. 이미지 latent 모델에서 같은 일반화를 기대하기 어려운 이유의 대조군. 서지 세부 미확인.

## 4. 우리 문제와의 연결: probing 설계 시 선행연구가 주는 권고
- 두 위치를 분리해서 probe: (a) DINOv2 encoder 출력 (동결, 학습분포와 무관), (b) predictor 중간층/출력(rollout 후 t+k). 크기 정보가 (a) 에는 있고 (b) 의 장기 step 에서 사라지면 "predictor 수축" 가설 지지.
- probe 용량 비교: linear / MLP / attentive(패치 토큰) 를 모두 쓴다(2.1 근거). 층별 curve 보고.
- 학습 크기 범위 안/밖에서 probe 를 학습하고 평가하는 분리: in-range 로 학습한 probe 가 OOD 크기를 외삽하는가는 probe 자체의 외삽 문제와 섞인다. OOD 크기를 일부 포함해 학습한 probe(상한선)와 학습분포만으로 학습한 probe 를 비교.
- 크기 외 confound 통제: ACWM-Phys 에서 크기와 초기 배치가 함께 바뀜. probing 타깃은 "천 extent" 같은 직접 관측 가능한 양과 크기 파라미터를 따로 둔다.
- 대조군: 무작위 초기화 predictor, DINOv2 raw feature 평균 pooling baseline, 픽셀 면적 같은 단순 handcrafted feature(DINO latent 가 이것보다 더 정보를 갖는지).
- 물리량 목록: cloth 는 크기 외에 질량/강성/마찰이 있으나 ACWM-Phys 에 이 파라미터가 라벨로 제공되는지는 미확인 (HF 데이터셋 카드 확인 필요).

## 5. 연구 공백
1. 동결 foundation encoder 기반 world model(DINO-WM)에서 연속 물리 파라미터(크기)를 encoder/predictor 별로 probing 한 연구는 내가 찾은 범위에서 없다(검색 한계 있음, "없다"의 단정은 불가).
2. "관측 가능한 파라미터가 OOD 일 때 rollout 이 학습분포로 붕괴"를 정량 분해(인코딩 손실 vs 동역학 수축)한 연구 없음(추정, 같은 한계).
3. ACWM-Phys 는 DINO-WM 류 latent 모델을 평가하지 않았다(확인: 평가 모델은 ACWM-DiT).

## 6. 후보 가설 (가설 확정은 사람 승인 필요)
- H1: DINOv2 patch feature 에서 천 크기(예: extent)는 OOD 크기에서도 probe 로 선형/MLP 복원 가능하다 (정보는 인코더에 있다). 성공 기준 예: OOD 에서 R^2 가 InD 대비 크게 떨어지지 않음(구체 기준은 팀 합의).
- H2: predictor 의 rollout 출력 latent 에서는 크기 probe 오차가 step 이 늘수록 학습 크기 쪽으로 편향된다(수축). 지표: 예측 크기 vs 실제 크기의 회귀 기울기 < 1, 학습 평균 쪽 bias.
- H3: 학습 데이터에 크기 다양성을 넣거나(분산 > 0), 크기/extent 를 보조 supervision(G-JEPA 방식) 또는 conditioning(SG-JEPA 방식)으로 주면 H2 의 수축이 줄어든다.
- H4: 행동 표현(공유 3-D vs 팔별)을 바꾸면 OOD 수축이 달라진다(ACWM-Phys 결과로부터의 추정).
- 대안 설명 H0: 붕괴가 아니라 OOD 입력에서 predictor 가 분포 밖 token 에 대해 일반 prior 출력 -> probe 로 구분 가능.

## 7. 출처 목록
- ACWM-Phys: https://arxiv.org/abs/2605.08567 ; https://arxiv.org/html/2605.08567 ; https://xavihart.github.io/ACWM-Phys ; https://huggingface.co/datasets/t1an/ACWM-Phys
- DINO-WM: https://arxiv.org/abs/2411.04983 ; https://proceedings.mlr.press/v267/zhou25t.html
- SG-JEPA: arXiv 2609.10464 (로컬 PDF)
- G-JEPA: arXiv 2610.28483 (로컬 PDF)
- Video FM probing: https://arxiv.org/abs/2606.09646
- POKEWORLD: https://www.alphaxiv.org/abs/2607.27017
- PhyIP Observer Effect: https://arxiv.org/abs/2602.12218
- V-JEPA intuitive physics: arXiv 2502.11831 (검색 요약만)

## 8. 확신도
중간-낮음.
- 높음: acwm-phys 와 Cloth Move OoD(크기 변경)의 존재, DINO-WM/SG-JEPA/G-JEPA 의 기본 주장.
- 중간: probing 방법론 권고(2.1, 2.2 는 초록/요약 기반).
- 낮음: 원인 후보(3절)와 해결 방향 문헌(RMA, MeshGraphNets 등은 기억 기반, 미검증). 어떤 논문도 DINO-WM 의 크기 OOD 붕괴를 직접 분석한 것은 확인하지 못했다. 2026년 preprint 의 수치는 2차 요약이다.
