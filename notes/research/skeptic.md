# 회의론자 관점 자료조사: DINO-WM latent의 물리 정보 probing과 "큰 천이 작아지는" 현상

- 작성: analyst-skeptic, 2026-10-31
- 모드: 자료조사. 실험과 코드 실행은 하지 않았다.
- 다른 분석가 파일(analyst-prior, analyst-stats, lit_review.md)은 읽지 않았다. 내 이전 파일 `notes/research/analyst-skeptic.md`만 참고했다.
- 읽은 것
  - 로컬 PDF: DINO-WM (p.7-8, 4.5절과 Table 3), SG-JEPA (p.1-3)
  - 로컬 코드(`thread0/`, main 브랜치): `models/visual_world_model.py`, `models/vit.py`, `conf/train.yaml`, `datasets/deformable_env_dset.py`, `env/deformable_env/src/sim/sim_env/scenes.py`
  - 웹 검색: 아래 각 항목에 출처를 달았다.
- **읽지 못한 것:** `hanbin5/acwm-phys` 브랜치. 원격 ref만 있고(`thread0/.git/refs/remotes/upstream/hanbin5/acwm-phys`) 파일은 checkout되어 있지 않다. 따라서 acwm-phys에서 천 크기를 어떻게 바꾸는지, 물리 파라미터를 모델 입력으로 주는지, 학습할 때 크기 분포가 어땠는지는 **모른다.** 아래에서 acwm-phys에 대해 쓴 내용은 모두 main 브랜치 코드를 근거로 한 추정이다.
- 표기
  - [확인]: 원문 PDF나 코드에서 직접 읽은 내용
  - [검색확인]: 웹 검색으로 서지와 요지만 확인한 내용. 본문 수치는 대조하지 않았다.
  - [추정]: 내 추론

---

## 0. 요약

1. **"크기가 줄어든다"는 관측이 어느 단계에서 생겼는지부터 확인해야 한다.** DINO-WM의 rollout 영상은 VQ-VAE 디코더가 만든다. 이 디코더는 학습 분포의 이미지(작은 천)만 보고 학습됐다 [확인, `visual_world_model.py:207-218`]. 그러니 latent에는 큰 천이 남아 있는데 디코더가 작은 천으로 "그려 버리는" 경우를 먼저 배제해야 한다. 이것을 배제하기 전에는 인코더나 프리딕터의 문제라고 말할 수 없다.
2. **가장 단순한 설명은 "학습 데이터에서 크기가 변하지 않았으므로 프리딕터가 크기를 입력에서 복사할 이유가 없었다"이다.** 크기 분산이 0인 데이터에서는 "평균 천 템플릿"을 출력해도 손실이 늘지 않는다. 이 해법은 문헌의 case-based generalization(Kang et al., ICML 2025)이나 shortcut learning(Geirhos et al., 2020)과 같은 종류다. 이 설명이 맞다면 "물리 인식 부족"이라는 해석은 과장이다.
3. **probing으로 "정보가 있다"를 보여도 원인 규명은 되지 않는다.** 크기는 한 프레임에서 바로 보이는 외형 속성이다. 그래서 frozen DINOv2 출력에 크기 정보가 있을 가능성은 매우 높다 [추정]. 문제는 프리딕터가 그 정보를 쓰느냐이다. 이것은 probing으로 판정할 수 없고 개입 실험이 필요하다(Ravichander et al., 2021; Elazar et al., 2021).
4. **질량과 마찰 probing에는 구조적 함정이 있다.** upstream cloth scene에서는 stiffness와 dynamicFriction이 같은 난수 `sf` 하나로 정해진다 [확인, `scenes.py:203-207`]. 두 값은 완전히 상관되어 있어서, 이 데이터로는 "마찰을 probe했다"와 "stiffness를 probe했다"를 구분할 수 없다. acwm-phys가 이 결합을 풀었는지는 확인하지 못했다.

---

## 1. "크기 축소" 현상의 대안 원인 (우선순위 순)

원인 후보마다 근거, 예측, 그리고 가장 싼 판별 방법을 적었다. 위에 있을수록 먼저 배제해야 할 후보다.

### A. 디코더(시각화) 문제 (가장 먼저 배제할 것)

- [확인] 디코더는 GT 인코딩 재구성과 `z_pred.detach()` 디코딩 두 가지로 학습된다. predictor 손실에는 영향을 주지 않는다 (`visual_world_model.py:207-218`, 주석 "recon loss should only affect decoder").
- [확인] thread0 README에 따르면 Rope/Granular에서 "encode–decode reconstruction of the target frame is almost as poor as the prediction. The gap is mostly due to the decoder." 즉 이 코드베이스의 deformable 환경에서 디코더 품질이 병목이라는 것은 이미 관찰된 사실이다.
- [추정] 작은 천만 보고 학습한 디코더는 처음 보는 큰 천 latent를 학습 분포의 가장 가까운 이미지, 즉 작은 천으로 투영할 가능성이 있다. 이 경우 latent에는 문제가 없고 "그림"만 틀린다.
- **판별:** 큰 천의 GT 프레임을 인코딩한 다음 바로 디코딩한다(predictor 없이). 이미 여기서 작아지면 원인은 A다. 그때는 인코더나 프리딕터 문제라는 주장 전체를 보류해야 한다. latent 공간에서 직접 측정하는 방법도 있다. 예를 들어 GT 큰 천 latent, GT 작은 천 latent, 예측 latent 사이의 patch별 거리를 비교한다.

### B. 데이터 편향으로 인한 프리딕터의 prior 회귀 (가장 단순한 설명)

- [확인] upstream cloth scene은 `cloth_size = [1.0, 1.0] * 70.0`으로 고정되어 있다 (`scenes.py:196`). acwm-phys가 학습할 때 크기를 다양하게 했는지는 모른다.
- [추정] 학습 데이터의 크기 분산이 0이거나 매우 작으면, "입력의 크기를 다음 프레임으로 정확히 옮기는 함수"와 "학습 크기의 천 템플릿을 출력하는 함수"는 학습 손실에서 구분되지 않는다. 경사하강법이 어느 쪽을 고를지는 보장되지 않는다. 축소는 후자를 골랐다는 증거일 뿐이다. 물리 지식이 없다는 증거는 아니다.
- 문헌 근거
  - Kang et al., "How Far is Video Generation from World Model: A Physical Law Perspective", ICML 2025 (arXiv 2411.02385) [검색확인]. 2D 물리 시뮬레이터 영상으로 학습한 비디오 생성 모델을 시험했다. 분포 안 일반화는 스케일링으로 해결되지만 OOD에서는 실패했다. 모델은 일반 법칙 대신 "가장 가까운 학습 사례를 흉내 내는" case-based generalization을 보였다. 학습 데이터를 참조할 때 속성 우선순위는 color > size > velocity > shape였다고 보고한다. 우리 현상, 즉 크기가 학습 사례 쪽으로 돌아가는 것과 같은 방향의 관찰이다. 다만 모델 계열(픽셀 비디오 생성)이 다르므로 DINO-WM에 그대로 적용된다는 것은 [추정]이다.
  - Geirhos et al., "Shortcut learning in deep neural networks", Nature Machine Intelligence 2(11):665-673, 2020 [검색확인]. 벤치마크에서는 잘 맞지만 조건이 바뀌면 전이되지 않는 결정 규칙에 관한 관점 논문이다.
  - DINO-WM 원 논문 [확인, p.7-8, Table 3]: PushObj(새 물체 모양)에서 SR은 0.34다. 저자들 스스로 "the model was only trained on the four object shapes, which makes it difficult to precisely infer relevant physical parameters"라고 쓴다. GranularRandom의 성공에 대해서는 "making the variance in particle number still within the distribution for each image patch"라는 **가설**을 제시했고, 검증하지는 않았다.
- **판별:** 학습 데이터에 크기를 2-3단계 섞어 재학습한다. 중간 크기나 조금 더 큰 크기에서 축소가 사라지거나 줄면 B가 주원인이다.

### C. 프리딕터의 절대 위치 임베딩과 patch 단위 OOD

- [확인] ViTPredictor는 학습되는 절대 위치 임베딩 `pos_embedding`을 `num_frames * num_patches` 크기로 쓴다 (`vit.py:110, 117`).
- [추정] 작은 천만 학습하면 화면 가장자리 patch 위치에는 항상 바닥이나 배경만 있었다. 그러면 프리딕터가 "이 위치의 다음 latent는 배경"이라는 위치 의존 prior를 배울 수 있다. 큰 천이 그 위치를 덮으면 예측이 배경 쪽으로 끌려가고, 그 결과 가장자리부터 천이 사라지는 축소로 나타날 수 있다.
- DINO-WM 저자의 GranularRandom 설명("patch 단위로는 분포 안")과 비교하면 이렇다. 큰 천은 patch 하나만 보면 천 텍스처라서 분포 안이다. 하지만 **"그 위치에 천이 있다"는 조합**은 분포 밖이다. 저자 가설이 크기 증가 쪽에는 적용되지 않을 수 있다 [추정].
- **판별:** 예측 오차를 patch 위치별 지도로 그린다. 오차가 학습 천 경계 바깥 위치에 몰려 있는지 본다. 같은 크기의 천을 화면의 다른 위치로 옮겨 학습/평가하는 대조 실험도 쓸 수 있다.

### D. 자기회귀 오차 누적 (exposure bias)

- [확인] 학습은 teacher forcing이다(`num_pred: 1`, `num_hist: 3`, `conf/train.yaml:58-59`). rollout은 예측 latent를 다시 입력으로 넣는다 (`visual_world_model.py:298-302`).
- [추정] 1-step 오차가 작아도 매 step마다 학습 분포 쪽으로 조금씩 당겨지면, 몇 step 뒤에는 학습 크기라는 attractor로 수렴할 수 있다. 사용자가 말한 "갑자기"라는 표현이 1-step에서 바로 일어난다는 뜻인지, 몇 step 뒤 급변한다는 뜻인지가 중요하다. 전자라면 B나 C, 후자라면 D가 유력하다.
- 문헌
  - Lambert, Pister, Calandra, "Investigating Compounding Prediction Errors in Learned Dynamics Models", arXiv 2203.09637, 2022 [검색확인]. 오차의 모양과 크기를 가장 강하게 결정하는 것은 시스템 동역학 자체라고 보고한다.
  - Self Forcing (arXiv 2506.08009, NeurIPS 2025) [검색확인]. 자기 출력으로 rollout하면서 학습해 exposure bias를 줄인다.
  - SG-JEPA [확인, p.2-3]. 다단계 rollout 손실을 encoder와 predictor 모두로 역전파한다. 좁은 중력 범위로 학습하고 넓은 범위로 평가해 OOD 일반화를 주장한다. 다만 이전 파일에서 지적했듯이 이 논문은 "이득이 encoder에서 오는지 predictor에서 오는지"에 대한 서술이 p.2와 abstract 사이에서 충돌한다. 또 중력 g를 **명시 입력으로 준다**(Eq.1). 픽셀에서 물리량을 추론한 결과가 아니다.
- **판별:** teacher-forced 1-step 예측(GT 큰 천 latent 3개를 입력)에서 이미 축소되는지, open-loop k-step에서만 축소되는지를 구분해 측정한다.

### E. 전처리, 해상도, 카메라 문제

- [확인] 입력은 `img_size: 224`로 resize된다 (`conf/train.yaml:44`). DINOv2 ViT-S/14 기준으로 16×16 patch다 [추정, 14px patch 가정].
- [추정] 큰 천이 카메라 시야 밖으로 나가거나 resize나 crop 때 잘리면, GT 관측 자체에서 천이 작아 보인다. 시뮬레이터 실제 영상(Real 패널)에서 큰 천이 온전히 보이는지 먼저 확인해야 한다.
- [추정] FleX에서 천의 크기는 particle 개수로 정해질 가능성이 높다(cloth_size 70은 격자 차원으로 보인다. 원문 미대조). 크기가 바뀌면 particle 밀도, 렌더 메시, 늘어짐 거동이 함께 바뀔 수 있다. "시뮬레이터에서 큰 천이 실제로 오그라들거나 접히는" 가능성도 Real 프레임으로 배제해야 한다.

### F. 정규화 통계와 조건화 입력의 OOD (acwm-phys 코드를 봐야 확정)

- [확인] action과 proprio는 학습 데이터의 mean/std로 정규화되고, std에 `1e-6`을 더한다 (`deformable_env_dset.py:72-73, 83-84`).
- [추정, 잠재적 버그] acwm-phys가 천 크기나 물리 파라미터를 proprio나 state 같은 조건화 입력으로 넣고, 학습 데이터에서 그 값이 하나로 고정되어 있었다면 std는 약 1e-6이다. 그러면 큰 천을 넣었을 때 정규화 값이 10^5 단위로 폭주한다. 이런 입력에서 프리딕터의 출력은 정의되지 않으며, 학습 평균 쪽으로 뭉개질 수 있다. **acwm-phys 코드에서 반드시 확인할 항목이다.**
- [추정] 액션 조건화. pusher 궤적이 천 크기를 기준으로 샘플링됐다면 큰 천에서는 액션 분포도 OOD다. 접촉이 일어나는 위치가 달라지기 때문이다. 이 경우 문제는 크기 표현이 아니라 action-contact 조합의 OOD다.
- [확인] proprio token도 예측되어 rollout에 다시 들어간다(concat_dim=1에서 visual, proprio, action을 각 patch에 붙임, `visual_world_model.py:102-108`, `train.yaml:46`). 조건화 입력이 OOD이면 그 영향은 모든 patch에 퍼진다 [추정].

---

## 2. probing으로 "물리 정보가 이미 있는지" 확인할 때의 함정

### 2.1 정보가 있어도 모델이 쓰지 않을 수 있다 (치명)

- Ravichander, Belinkov, Hovy, "Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?", EACL 2021, pp.3363-3377 [검색확인]. 모델은 과제에 필요 없는 속성도 우연 이상의 수준으로 인코딩할 수 있다.
- Elazar et al., "Amnesic Probing", TACL 9:160-175, 2021 [이전 파일에서 검색확인]. probing 성능은 과제 중요도와 상관되지 않는다. 효용은 그 정보를 제거하는 개입으로 재야 한다.
- Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances", Computational Linguistics 48(1):207-219, 2022 [검색확인]. 이 틀의 방법론적 한계(기준선 필요, 상관과 인과의 혼동)를 정리한 리뷰다.
- Vafa et al., "What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models", ICML 2025 (arXiv 2507.06952) [검색확인]. 궤도 데이터로 학습한 모델은 다음 상태 예측은 잘했지만, 새 과제에 적응시키면 뉴턴 역학에 맞는 귀납 편향을 보이지 않았다. 모델은 과제별 휴리스틱처럼 행동했다. **"예측을 잘한다 ≠ 물리 법칙을 담았다"**의 직접 사례다.
- **함의:** frozen DINOv2 출력에서 크기 probe가 잘 되는 것은 거의 확실하다 [추정, 크기는 외형으로 보이기 때문]. 그렇다고 "인코더는 문제없고 프리딕터가 문제다"라고 결론 낼 수는 없다. 프리딕터가 그 방향을 쓰는지는 개입으로 따로 확인해야 한다. 예를 들어 입력 latent에서 크기 방향을 조작했을 때 예측 latent의 크기가 따라 바뀌는지 본다. Othello-GPT에서 probe 방향을 개입해 인과성을 보인 Li et al. 2023과 Nanda et al. 2023 [검색확인]이 방법론의 선례다.

### 2.2 probe 용량과 선택성 (중요)

- Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks", EMNLP 2019 (arXiv 1909.03368) [검색확인]. 흔히 쓰는 설정의 MLP probe는 selectivity가 매우 낮았다. 예로 POS 과제에서 MLP는 97.3 대 control 92.8이었고, linear는 97.2 대 71.2였다. 고용량 probe의 높은 정확도는 표현이 아니라 probe가 배운 것일 수 있다.
- DINO-WM의 patch latent는 196 이상 × 384차원으로 매우 고차원이다. 에피소드 수가 적으면 linear probe도 과적합한다 [추정]. control task(label shuffle, 에피소드 ID 예측)와 차원 축소 기준선이 필요하다.

### 2.3 외형 confound (치명, 질량과 마찰에 해당)

- **크기:** 단일 프레임에서 보인다. probe가 성공해도 물리 이해의 증거가 아니라 외형 인식의 증거다.
- **질량, 마찰, stiffness:** 단일 프레임으로는 원리상 관측할 수 없다. 상호작용 이후의 운동에서만 드러난다.
  - Physion++ (Tung et al., NeurIPS 2023, arXiv 2306.15668) [검색확인]. 표준 방식으로 학습한 모델은 질량, 마찰, 탄성 같은 숨은 속성을 스스로 추론하도록 학습되지 않았다. 사람과의 격차도 크다.
  - Zhan, Ma, Xie, Zisserman, "Inferring Dynamic Physical Properties from Video Foundation Models", arXiv 2510.02311, 2025 [검색확인]. frozen 비디오 모델 위에 가벼운 readout을 붙여 탄성, 점성, 마찰을 추정했다. oracle보다 뒤처지고 도메인 이동에 약했으며, 마찰이 가장 어려웠다. 이 논문의 수치는 대조하지 않았다.
- **[확인] 데이터 생성의 결합:** upstream `cloth_scene()`에서 stiffness와 dynamicFriction은 둘 다 `sf = np.random.rand()` 하나의 함수다 (`scenes.py:203-207`: `stiffness_factor = sf*1.4+0.1`, `dynamicFriction = -sf*0.9+1.0`). stretch stiffness는 [1.0, 1.5]로 clip된다. 따라서 이 생성기로 만든 데이터에서는 마찰 probe의 성공과 stiffness probe의 성공을 구분할 수 없다. 어느 쪽이든 "sf 하나"를 읽는 것이다. acwm-phys가 파라미터를 독립적으로 샘플링하는지 확인해야 한다.
- 파라미터와 외형의 우연한 상관도 점검해야 한다. 예를 들어 크기와 렌더 색, 메시 밀도, 그림자가 함께 바뀌면 probe는 그것을 읽는다 [추정].

### 2.4 OOD probing의 이중 외삽 문제 (중요)

- 학습 데이터에서 크기가 고정이면 크기 probe를 학습 데이터로 학습할 수조차 없다. 분산이 0이기 때문이다. 크기가 다양한 probing용 데이터를 새로 만들면, 그 데이터는 프리딕터 입장에서 OOD다.
- 작은 크기 범위로 학습한 probe를 큰 천에 적용하면 probe 자체가 외삽한다. Xu et al., "How Neural Networks Extrapolate", ICLR 2021 (arXiv 2009.11848) [검색확인]에 따르면 ReLU MLP는 학습 범위 밖에서 원점 방향으로 거의 선형으로 행동한다. 그래서 "예측 latent에서 probe 값이 작게 나왔다"는 결과는 **표현이 작아졌는지, probe가 외삽에 실패했는지** 구분되지 않는다.
- **제안:** probe는 크기 전 범위를 덮는 데이터로, frozen DINOv2 GT 인코딩에 대해서만 학습한다. 그 probe를 예측 latent에 적용한다. 그러면 probe 외삽 문제와 프리딕터 OOD 문제가 분리된다.

### 2.5 데이터 누수 (중요)

- 물리 파라미터는 에피소드마다 하나로 고정된다. 프레임 단위로 train/test를 나누면 probe는 같은 에피소드의 외형(초기 배치 등)을 외워 파라미터를 맞힐 수 있다 [추정]. 분할 단위는 **에피소드, 가능하면 파라미터 값 단위**여야 한다(파라미터 값 held-out).
- DINO-WM 예측 latent를 probing할 때도 같은 원칙이 적용된다. 프리딕터를 학습할 때 쓴 에피소드로 probe를 평가하면 프리딕터 암기와 probe 성공이 섞인다.

---

## 3. 해결 방향 선행연구와 회의적 단서

| 방향 | 대표 문헌 | 단서 (회의론) |
|---|---|---|
| 물리 파라미터 명시 조건화 | SG-JEPA (중력 g를 action에 붙임) [확인]. AdaptiGraph (Zhang et al., RSS 2024, arXiv 2407.07889; 물성 변수를 그래프에 넣고 test-time 역최적화) [검색확인] | 파라미터를 알아야 한다. 외삽 범위는 여전히 보장되지 않는다(Xu et al. 2021). AdaptiGraph는 particle 그래프 기반이라 DINO-WM 같은 이미지 latent와 구조가 다르다. |
| 문맥 latent 추론 (system ID) | CaDM (Lee et al., ICML 2020, arXiv 2005.06800; 최근 경험으로 context latent 추정, 질량 등 변화) [검색확인]. T2-Z (로컬 PDF, 환경 latent test-time 최적화) [이전 파일에서 확인] | 크기는 숨은 파라미터가 아니라 보이는 속성이다. 크기 축소 문제에 context 추론을 처방하는 것은 문제와 처방이 어긋날 수 있다 [추정]. 질량과 마찰에는 더 적합하다. |
| 데이터 다양화 | Kang et al. 2025 [검색확인]: 스케일링은 조합 일반화에는 효과가 있었지만 OOD에서는 실패했다 | 가장 싸고 단순한 대조군이다. 이 처방만으로 해결되면 "physics-aware 구조"의 필요성 주장은 약해진다. |
| rollout 학습 / exposure bias 감소 | Self Forcing 2025, SG-JEPA rollout loss, Lambert et al. 2022 [검색확인] | 원인이 D(오차 누적)일 때만 유효하다. 1-step부터 축소되면 효과가 없을 것이다 [추정]. |
| 디코더 개선 | (해당 문헌 없음, 진단 차원) | 원인이 A이면 world model 성능(planning)과는 무관하다. 시각화만 고치면 된다. |

---

## 4. 과장되기 쉬운 주장 (사전 경고)

1. "DINO-WM은 물리 파라미터 일반화에 실패한다." 분포 밖 크기 하나, 데이터셋 하나, 디코더를 거친 시각 관찰 하나로 일반화 주장을 할 수 없다. 최소한 latent 수준의 정량 지표가 필요하다.
2. "인코더(DINOv2)에는 정보가 있는데 프리딕터가 버린다." probing만으로는 '버린다'가 아니라 '읽을 수 없다'만 말할 수 있다. 원인 위치를 특정하려면 개입이 필요하다.
3. "physics-aware 모듈을 넣으니 크기 축소가 사라졌다." 같은 데이터 다양화와 학습 예산을 준 baseline DINO-WM과 비교하지 않으면 모듈 효과와 데이터 효과를 구분할 수 없다.
4. "질량과 마찰을 latent에서 읽을 수 있다." 단일 프레임 입력에서 나온 결과라면 외형 confound나 파라미터 결합(2.3절)의 산물일 가능성이 높다.
5. "학습 분포로 붕괴(collapse)한다." 이 표현은 표현 붕괴(representation collapse)와 혼동되기 쉽다. 관찰된 것은 "출력이 학습 사례 쪽으로 회귀한다"이다. 용어를 분리해야 한다.

---

## 5. 재현성과 누수 체크리스트

- acwm-phys 브랜치의 commit hash, 학습 크기 분포, 테스트 크기, 파라미터 샘플링 코드를 config.yaml에 기록한다.
- FleX는 프로세스 간 bit 재현이 되지 않는다 [확인, thread0 README "FleX simulation and rendering are not bit-reproducible across processes"]. 단일 영상 사례가 아니라 여러 에피소드와 여러 시드의 통계로 보고해야 한다.
- 축소가 몇 개 에피소드 중 몇 개에서 나타났는지(빈도)를 보고한다. 골라 뽑은 영상(cherry-pick) 하나로 현상을 정의하지 않는다.
- 디코더, 프리딕터, 인코더의 체크포인트 epoch를 고정한다. 디코더와 프리딕터는 학습률이 다르다(3e-4 대 5e-4, `train.yaml:40-41`). 그래서 같은 epoch라도 수렴 정도가 다를 수 있다.
- 정규화 통계(mean/std)를 저장하고 OOD 입력의 정규화 값을 기록한다(1.F).

---

## 6. 이 현상의 원인을 믿으려면 먼저 필요한 실험 (최대 3개)

1. **단계 분리 진단(가장 싸다):** 큰 천 에피소드에서 다음 네 가지를 비교한다. (a) 시뮬레이터 Real 프레임, (b) encode→decode (프리딕터 없음), (c) teacher-forced 1-step 예측의 디코딩과 latent 거리, (d) open-loop k-step 예측. 축소가 처음 나타나는 단계가 원인 후보를 A, E(전처리/시뮬), B·C(1-step), D(rollout) 중 하나로 좁힌다. latent 쪽 지표로는 GT 큰 천 latent와 GT 작은 천 latent에 대한 상대 거리를 쓴다. 디코더를 거치지 않는다.
2. **크기 probe와 개입:** 크기가 다양한 데이터로 frozen DINOv2 GT 인코딩에서 크기 probe를 학습한다(에피소드 단위 분할, label-shuffle control 포함). 이 probe로 예측 latent의 크기 궤적을 잰다. 이어서 입력 latent의 크기 방향을 조작했을 때 예측 latent의 크기가 따라 바뀌는지 본다. 이것으로 "정보 없음"과 "정보 있음, 미사용"을 구분한다.
3. **데이터 다양화 대조군:** 학습 크기를 2-3단계로 섞어 재학습한 baseline DINO-WM에서 보간 크기와 외삽 크기의 축소 정도를 잰다. 이것으로 B(데이터 편향)의 기여를 정량화하고, 이후 physics-aware 방법이 넘어야 할 공정한 기준선을 만든다. GPU 비용이 크므로 사람 승인이 필요하다.

---

## 확신도: 중간

- 높게 보는 부분: 디코더 경유 관찰의 위험(A)과 teacher forcing/rollout 구조(D)는 코드에서 직접 확인했다. 위치 임베딩이 절대 위치 임베딩이라는 점(C)과 cloth scene의 stiffness-friction 결합(2.3)도 코드에서 확인했다. probing 해석의 한계는 여러 독립 문헌이 일관되게 지지한다.
- 낮추는 이유 1: acwm-phys 브랜치 코드를 읽지 못했다. 크기를 바꾸는 방식, 조건화 입력 여부, 학습 크기 분포를 모른다. 1.B, 1.F, 2.3의 acwm-phys 적용은 추정이다.
- 낮추는 이유 2: 현상 자체(영상)를 보지 못했다. "갑자기"가 1-step에서 일어나는지 k-step 뒤에 일어나는지에 따라 유력 원인이 바뀐다.
- 낮추는 이유 3: 웹 검색으로 확인한 문헌들([검색확인])은 서지와 요지만 확인했고 본문 수치는 대조하지 않았다. Kang et al.의 속성 우선순위는 픽셀 비디오 생성 모델의 결과다. DINO-WM 같은 latent 예측 모델에 적용된다는 것은 추정이다.
- 낮추는 이유 4: SG-JEPA가 인용한 "physical parameters probing" 관련 문헌(Xue et al. 2026, Tan et al. 2026)은 찾아 확인하지 않았다.

## 출처

- Kang et al., ICML 2025: https://arxiv.org/html/2411.02385v2 , https://proceedings.mlr.press/v267/kang25g.html
- Geirhos et al., 2020: https://bethgelab.org/publication/2020_11_geirhos/
- Hewitt & Liang, 2019: https://arxiv.org/pdf/1909.03368
- Ravichander et al., 2021: https://aclanthology.org/2021.eacl-main.295
- Belinkov, 2022: https://aclanthology.org/2022.cl-1.7/
- Elazar et al., 2021: https://aclanthology.org/2021.tacl-1.10/
- Vafa et al., ICML 2025: https://arxiv.org/abs/2507.06952
- Li et al. 2023 / Nanda et al. 2023 (Othello-GPT): https://arxiv.org/html/2210.13382v5 , https://arxiv.org/pdf/2309.00941v1
- Physion++: https://arxiv.org/abs/2306.15668v1
- Zhan et al., 2025: https://arxiv.org/abs/2510.02311
- Garrido et al., 2025 (V-JEPA intuitive physics, 참고용): https://arxiv.org/pdf/2502.11831
- Lambert et al., 2022: https://arxiv.org/abs/2203.09637
- Self Forcing, 2025: https://arxiv.org/pdf/2506.08009v1.pdf
- Mathieu et al., ICLR 2016 (MSE 예측의 평균화/흐림, 참고용): https://arxiv.org/abs/1511.05440v6
- Xu et al., ICLR 2021: https://arxiv.org/pdf/2009.11848
- CaDM, ICML 2020: https://arxiv.org/pdf/2005.06800
- AdaptiGraph, RSS 2024: https://arxiv.org/abs/2407.07889
- 로컬: `notes/research/papers/worldmodel_jepa/DINO-WM.pdf` (p.7-8), `sg-jepa.pdf` (p.1-3), `thread0/README.md`, `thread0/models/*.py`, `thread0/conf/train.yaml`, `thread0/datasets/deformable_env_dset.py`, `thread0/env/deformable_env/src/sim/sim_env/scenes.py`
