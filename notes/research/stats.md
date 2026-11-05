# 통계·데이터 관점 자료조사: DINO-WM latent probing 과 OoD 물리 파라미터 붕괴

작성: analyst-stats / 2026-10-31
범위 주의: 웹 검색 없이 기억에 의존한 문헌 목록이다. 저자·연도·제목은 확실한 것만 적었고, 세부 수치(성능, 임계값)는 일부러 적지 않았다. "(추정)" 은 근거가 약한 해석이다. 인용 전에 원문 확인이 필요하다 (특히 arXiv 번호).

## 0. 로컬에서 확인한 사실

- acwm-phys 데이터, 체크포인트, DATASET_DIR 환경변수: 이 머신에는 없다. (`find` 로 acwm/dino_wm 이름 검색, `~` 아래 *.ckpt/*.pth/*.h5 검색, `env` 확인. 결과 없음.) `thread0/data` 도 없다.
- 코드는 있다. `/home/jaekyungkim/mylab/thread0` 의 현재 브랜치 `main` 에는 acwm 코드가 없고, git 원격 브랜치 `upstream/hanbin5/acwm-phys` 에 있다 (`git -C thread0 show upstream/hanbin5/acwm-phys:<경로>` 로 읽음).
  - `datasets/acwm_phys_dset.py`, `conf/env/acwm_cloth_move.yaml` (data_path = `$DATASET_DIR/acwm_phys/deformable/clothmove`), `README.md`, `repro/SETTINGS.md`, `repro/PRIOR_WORK.md`(읽지 않음).
- 데이터 구조 (코드/README 기준): `<data_path>/{ind_train,ind_test,ood_test}/*.mp4 + metadata.pt`. metadata 는 dict 리스트 (video_path, actions [T,A], length, ...). 라벨은 action 뿐이고 물리 파라미터 라벨(질량, 마찰 등)을 쓰는 코드는 없다. SETTINGS.md §8 에 `cloth_dimx`, `cloth_dimy` 키가 metadata 에 있다고 적혀 있으나 "보이는 크기와 상관이 없다 (r ≈ 0.02)". 즉 천 크기의 신뢰할 라벨은 metadata 에 없고, 현재는 색 마스크로 센 천 픽셀 수를 대리 지표로 쓴다.
- Cloth Move 규모 (README): 학습/검증 1800/200, InD test 100, OoD test 100 (면적 0.26× 50개, 2.5× 50개). 65 프레임, frameskip 5, num_hist 1, 15 epoch, 시드 1개. 학습과 InD 의 천 크기는 하나뿐이다 (프레임 3 천 픽셀 13,980–14,330 @512²).
- 이미 있는 현상 수치 (README, 단일 시드, 에피소드 단위 bootstrap CI):
  - 마지막 프레임 천 픽셀: 큰 천 real 4,475 / model 1,836 / decoder-on-real 4,352. InD 는 1,819 / 1,888 / 1,790. 작은 천은 404 / 1,129 / 388.
  - 디코더가 실제 프레임 feature 로는 큰 크기를 그린다 -> 크기 정보가 인코더 feature 에 존재한다는 간접 증거이다 (probe 로 확정한 것은 아님). 붕괴는 예측 feature 쪽에서 생긴다.
  - 학습 8배 늘려도 큰 천 1,836 -> 1,923 (변화 없음). 검증 손실은 계속 하락 (0.166 -> 0.150).
  - OoD 에피소드 약 2.7 % (50개) 를 학습에 넣으면 큰 천 마지막 프레임 1,645 -> 3,524 (real 4,723), rollout MSE -49 % [-56,-42], 25/25 에피소드 개선. 작은 천은 오차 -39 % 이나 크기는 1,189 -> 1,223 (real 365) 로 거의 안 풀림. InD 오차는 +25 % [+16,+34].
  - 한계: 시드 1개, 설정 1개. 검증 손실이 마지막 epoch 에도 계속 하락.
- 해석 메모 (추정): 학습 데이터에 천 크기 변동이 0 이므로, 예측기가 크기에 의존하도록 학습될 신호 자체가 없다 (식별 불가능/underspecification). 인코더가 크기를 담고 있어도, 예측기가 그것을 다음 상태로 전달할 이유가 없다. 이 가설은 probe 로 인코더/예측기를 나눠 검증할 수 있다 (아래 §3).

## 1. Probing 방법론 선행연구

핵심 문헌 (확실한 것만):
- Alain & Bengio 2016, "Understanding intermediate layers using linear classifier probes". 선형 probe 의 출발점.
- Hewitt & Liang 2019 (EMNLP), "Designing and Interpreting Probes with Control Tasks". control task 와 selectivity (= 실제 과제 정확도 - 통제 과제 정확도). 표현력이 큰 probe 는 표현이 아니라 probe 자체가 과제를 외워서 높은 점수를 낼 수 있다는 점.
- Belinkov 2022 (Computational Linguistics), "Probing Classifiers: Promises, Shortcomings, and Advances". 종합 리뷰.
- Pimentel et al. 2020 (ACL), "Information-Theoretic Probing for Linguistic Structure"; Voita & Titov 2020 (EMNLP), "Information-Theoretic Probing with Minimum Description Length". 복잡한 probe 일수록 정보 추정이 더 정확하다는 입장 (data processing 관점)과 MDL 로 정확도와 용량을 함께 보고하는 방법.
- Elazar et al. 2021 (TACL), "Amnesic Probing". probe 가 읽을 수 있는 정보가 모델이 실제 쓰는 정보인지 개입(제거)으로 확인.
- Ravichander, Belinkov, Hovy 2021 (EACL), "Probing the Probing Paradigm". 읽을 수 있음과 사용함은 다르다.
- Gurnee & Tegmark 2023, "Language Models Represent Space and Time". 연속값(좌표, 시간)을 선형 ridge probe 로 읽고 R², 상관을 보고. 비교로 random-init/shuffled 통제, 시드별 변동, 그룹 단위 held-out 분할을 사용했다 (세부는 원문 확인 필요).
- Li et al. 2023 (ICLR), Othello-GPT "Emergent World Representations". 비선형 probe 로 읽히다가 후속(Nanda 등)이 선형으로도 읽힘을 보였다. probe 종류에 따라 결론이 달라질 수 있다는 사례.

### 이 문제에 적용할 설계 (권고, 출처 있는 원칙 + 내 판단)

1. 선형 vs 비선형: 선형(ridge) 을 기본으로 하고 소형 MLP (은닉 1층) 를 함께 보고한다. 두 값의 차이가 작으면 "선형 접근 가능". 차이가 크면 정보가 비선형으로 얽혀 있거나 MLP 가 외우는 것이므로 selectivity 로 구분한다 (Hewitt & Liang).
2. 용량 통제: 입력이 DINO 패치 토큰이면 차원이 매우 크다 (ViT-S 는 토큰당 384차원, 224px/14 = 16x16 = 256 패치. 모델 구성은 README: DINOv2 ViT-S/14). 특징 수 대비 에피소드 수가 적으므로 (OoD 100 에피소드) 정규화 강도를 중첩 CV 로 고르고, 풀링 방식 (평균, CLS, 패치 전체) 을 바꿔서 보고한다. 용량 순서: ridge(풀링) < ridge(패치 전체) < MLP.
3. control task (회귀판, 내 설계): (a) 라벨을 에피소드 간 무작위로 섞은 control (프레임 간 일관성 유지, 즉 에피소드 단위로 permute), (b) 랜덤 초기화 ViT 또는 raw pixel 입력 baseline, (c) "상수 평균 예측" baseline R²=0. selectivity = R²(진짜) - R²(control).
4. 라벨 선택: metadata 의 cloth_dimx/y 는 보이는 크기와 상관 r ≈ 0.02 이므로 라벨로 쓰면 안 된다. 보이는 천 면적(마스크 픽셀)을 쓰되, 마스크 자체의 오류를 보고한다 (README: 마스크 IoU 0.92, 디코더-on-real 과 면적 2–3 % 이내 일치).
5. 라벨 변동 문제: 학습/InD 는 크기가 하나뿐이므로 InD 만으로는 크기 probe 를 학습할 수 없다 (분산 0). probe 는 OoD 100 에피소드 (0.26x, 2.5x) + InD 를 함께 써서 에피소드 단위로 분할해야 한다. 크기가 사실상 3개 값뿐이므로 전체 R² 는 군집 간 분산이 지배한다. 군집 내 R² (같은 크기 안의 면적 변동, 시간에 따른 변동) 와 3-way 분류 정확도를 같이 보고한다.
6. 시간/자세 교란: 천 면적은 프레임 시점(드레이프 진행)에 따라 변한다. 크기와 시점이 섞이므로 같은 프레임 인덱스끼리 비교하거나 프레임 인덱스를 공변량으로 넣어 잔차화한다.
7. 정보 vs 사용: probe 가 읽힌다고 예측기가 쓴다는 뜻이 아니다 (Ravichander 2021, Elazar 2021). 개입 실험(큰 천 feature 의 크기 방향 성분을 투영 제거/치환했을 때 rollout 이 달라지는지)이 필요하다.

## 2. 평가 통계 관행

- 분할 단위는 에피소드이다. 같은 에피소드의 프레임은 강한 자기상관이 있어 프레임 단위 무작위 분할은 누수이다. GroupKFold (그룹 = 에피소드), OoD 는 군집(크기)을 통째로 빼는 leave-one-group-out 도 사용한다. README 의 평가도 에피소드 단위 bootstrap 을 쓴다.
- 신뢰구간: 에피소드를 재표집하는 cluster bootstrap (프레임이 아니라). 같은 에피소드에 두 모델이 평가되면 쌍 bootstrap (README 의 방식과 동일). 프레임 단위 CI 는 과도하게 좁게 나온다.
- 지표: R² 는 라벨 분산에 의존하므로 (분산 0 에서 정의 불가, 군집 간 분산 지배) MSE/RMSE 와 상대오차(|예측 - 실제| / 실제), 그리고 단조성 지표 Spearman ρ 를 함께 보고한다. "붕괴" 지표로는 기울기 (예측 면적을 실제 면적에 회귀한 기울기: 1 이면 보존, 0 이면 학습 크기로 붕괴) 가 가장 직접적이다. README 의 retention r(k) 가 같은 발상이다.
- 시드: 모델 학습 시드 변동과 에피소드 표집 변동은 다르다. 현재 모델은 시드 1개이므로 "OoD 에서 일관되게 붕괴" 라고는 말할 수 있어도 (큰 크기 차이, 8배 학습에도 불변) 효과 크기 비교 (예: 작은 천 -39 % vs 큰 천 -49 %) 는 시드 변동을 모른다. 권고: probe 는 probe 시드 >=5 (분할 + 초기화), 모델 학습은 가능하면 >=3 시드. Agarwal et al. 2021 (NeurIPS), "Deep RL at the Edge of the Statistical Precipice" 와 Henderson et al. 2018, "Deep RL that Matters" 가 소표본 시드 보고 관행의 출처이다. 둘 다 RL 이 대상이지만 원칙은 같다. 시드가 3–5개면 CI 는 넓고 "방향이 같다" 정도로만 쓴다.
- 검정력 (내 계산, Fisher z 근사, 양측 α=0.05, power 0.8): 상관을 검출할 수 있는 최소 |r| 은 n=25 에서 0.535, n=50 에서 0.387, n=100 에서 0.277, n=200 에서 0.197, n=400 에서 0.140 이다 (`python3` 한 줄 계산, tanh(2.8016/sqrt(n-3))). OoD 각 군집이 50 에피소드이므로 군집 내 약한 관계는 검출이 어렵다. 에피소드가 독립 단위이므로 n 은 프레임이 아니라 에피소드 수이다.
- 표본 수 경험칙 (휴리스틱, 출처: Green 1991 의 N >= 50 + 8m 은 다중회귀용으로 기억함, 원문 확인 필요): 풀링된 저차원 feature 로 선형 probe 를 하면 에피소드 수백 개면 충분하고, 패치 전체 (수만 차원) 는 정규화에 의존한다. 즉 차원이 에피소드 수보다 훨씬 크다.
- 다중 비교: 층/풀링/probe 종류/크기군집을 많이 비교하면 우연히 큰 값이 나온다. 비교 축 수를 사전에 정하고 (notes/hypotheses.md 에 먼저 적기) Holm 또는 CI 로 보고한다.

## 3. 인코더 vs 예측기 구분을 위한 probe 배치 (내 제안)

| 대상 | 입력 | 질문 | 비고 |
|---|---|---|---|
| A 인코더 | 실제 프레임의 DINO feature | 크기가 인코더에 있는가 | 상한. README 의 decoder-on-real 이 간접 지지 |
| B 예측기 1-step | 실제 z_t + 행동 으로 예측한 z_{t+1} | 한 스텝 예측이 크기를 보존하는가 | teacher forcing |
| C 예측기 rollout k-step | 열린 루프 예측 z_{t+k} | 몇 스텝에서 크기 정보가 사라지는가 | A 에서 학습한 probe 를 그대로 적용(전이) + 새로 학습한 probe 둘 다 |
| D 통제 | 섞은 라벨, 랜덤 ViT | 용량/외우기 | selectivity |

판독: A 높음, B 낮음 이면 예측기가 한 스텝에서 이미 크기를 지운다. A, B 높음, C 가 k 에 따라 감소하면 오류 누적/드리프트이다 (README 는 큰 천이 rollout 안에서 InD 크기로 수렴한다고 하며 1-step 에서 model/copy 오차비가 InD 0.31 -> OoD 0.74-0.77 로 커진다). 주의: A 에서 학습한 probe 를 예측 feature 에 적용하면 분포 이동 때문에 probe 자체가 틀릴 수 있어 (B, C 에서 새로 학습한 probe 와 비교) 두 방식을 모두 쓴다.

## 4. OoD 평가 방법 (문헌)

- 분포 밖 평가 틀: Koh et al. 2021 (ICML), WILDS; Gulrajani & Lopez-Paz 2021 (ICLR), DomainBed (모델 선택 방식이 OoD 결과를 좌우함. OoD 정보로 하이퍼파라미터를 고르면 안 됨).
- 보간 vs 외삽 구분: 큰 천 2.5x 는 외삽, 작은 천 0.26x 도 학습 범위 밖이다. 신경망의 외삽 거동: Xu et al. 2021 (ICLR), "How Neural Networks Extrapolate". Balestriero et al. 2021, "Learning in High Dimension Always Amounts to Extrapolation". 고차원에서는 대부분의 테스트점이 학습 볼록껍질 밖이므로 "OoD" 의 정의는 파라미터 축 하나로 한정해 보고해야 한다.
- 물리 파라미터 축별 OoD: 이 문제는 축이 하나 (크기) 이고 값이 3개뿐이다. 최소 중간값 (예: 0.5x, 1.5x) 이 있어야 곡선 (예측 대 실제) 을 그릴 수 있다. 현재는 점 3개 (0.26, 1, 2.5) 로 단조 여부밖에 못 본다. 새 데이터를 만들 수 있다면 크기를 연속 샘플링한 OoD sweep 이 통계적으로 가장 정보량이 크다 (추정).
- 누수 점검: 마스크 기반 지표는 디코더 한계를 포함하므로 decoder-on-real 천장 (README 가 이미 사용) 을 항상 함께 낸다. 작은 천 gap 이 디코더 때문인지 구분.

## 5. 물리 파라미터 probing 선행연구

- Bear et al. 2021 (NeurIPS Datasets), Physion: 물리 예측 벤치마크. 학습된 표현에서 선형 readout 으로 물체 접촉 예측.
- Tung et al. 2023 (NeurIPS Datasets), Physion++: 질량, 탄성, 마찰, 변형성 등 물성의 온라인 추론이 필요한 평가 (세부 수치는 원문 확인 필요).
- Piloto et al. 2022 (Nature Human Behaviour), "Intuitive physics learning in a deep-learning model inspired by developmental psychology": violation-of-expectation 평가.
- Garrido et al. 2025, "Intuitive physics understanding emerges from self-supervised pretraining on natural videos" (arXiv, V-JEPA 계열 VoE 평가). 표현 공간 예측이 물리 위반을 감지. 확인 필요: arXiv 번호.
- Li, Wu, Tenenbaum, Torralba et al. 2020 (ICML), "Visual Grounding of Learned Physical Models": 시각에서 물리 파라미터 추정.
- El Banani et al. 2024 (CVPR), "Probing the 3D Awareness of Visual Foundation Models": DINOv2 를 frozen probe 로 평가하는 프로토콜 (깊이/표면 법선). DINOv2 feature 가 기하 정보를 담는다는 사례 (천 크기도 기하이므로 인코더에 있을 가능성이 높다, 추정).
- DINO-WM 원논문 (Zhou et al. 2024, arXiv 2411.04983): 인코더 frozen, 예측기만 학습, 패치 토큰 수준 예측. 물리 파라미터 probing 은 하지 않은 것으로 기억함 (확인 필요). 로컬 PDF: `/home/jaekyungkim/mylab/notes/research/papers/worldmodel_jepa/DINO-WM.pdf` (읽지 않음).
- 확인하지 못한 것: 질량/마찰을 DINO-WM 류 latent 에서 probe 한 직접적 선행연구는 내가 확실히 아는 것이 없다.

## 6. OoD 파라미터에서 학습 분포로 붕괴하는 현상의 원인/해결 문헌

원인 후보 (각각 검증 방법과 함께):
1. 식별 불가능/underspecification: 학습 중 크기가 상수이면 예측기는 크기를 쓸 필요가 없다. D'Amour et al. 2020, "Underspecification Presents Challenges for Credibility in Modern ML"; de Haan et al. 2019 (NeurIPS), "Causal Confusion in Imitation Learning" (인과 아닌 단서에 의존). 검증: 학습 크기 변동을 늘렸을 때 개선되는지 (README 의 OoD 혼합 실험: 큰 천은 개선, 작은 천은 거의 안 풀림 -> 부분 지지).
2. MSE 회귀의 조건부 평균 효과: 입력에서 크기를 추론하기 어려우면 평균으로 수축한다 (추정). num_hist 1 이라 속도/역사로 파라미터를 추정할 수 없다. 검증: num_hist 를 늘려 비교 (Pour Water 는 num_hist 3 와 1 이 같은 horizon 에서 차이 없었다고 README 가 적음).
3. 자기회귀 드리프트: 오류가 누적되며 학습 매니폴드로 끌려간다. 검증: 스텝별 retention (README 에 있음: 12 스텝 후 큰 천 -0.09).
4. feature 정보 부재: 가능성은 낮아 보임 (디코더가 real feature 로 그림). probe 로 확인.

해결 방향 문헌:
- 데이터 다양성: Tobin et al. 2017, "Domain Randomization"; Peng et al. 2018, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization". README 의 2.7 % OoD 혼합 실험이 같은 방향.
- 파라미터/맥락 조건: Yu et al. 2017, UP-OSI; Lee et al. 2020 (ICML), CaDM "Context-aware Dynamics Model"; Kumar et al. 2021, RMA (rapid motor adaptation). 환경 파라미터 추정기와 동역학 모델을 결합. 라벨이 있으면 보조 손실로 파라미터 예측을 붙일 수 있다.
- 적응: Nagabandi et al. 2019 (ICLR), "Learning to Adapt in Dynamic, Real-World Environments through Meta-RL".
- 정리: 위 문헌은 대부분 저차원 상태 기반 RL 이고 픽셀/패치 토큰 world model 에서 천 크기 같은 형상 파라미터에 같은 효과가 있다는 직접 근거는 없다 (추정).

## 7. 권고 실험 규모 (추정 포함)

- Probe 데이터: InD 2000 + OoD 100 에피소드가 있으나 크기 변동은 OoD 100 뿐. 한 군집 50 에피소드. probe 학습/평가 분할은 군집 층화 에피소드 5-fold CV. 에피소드당 프레임은 수 개 (프레임 3, 15, 30, 64 같은 고정 인덱스) 만 써도 정보량은 비슷하다 (같은 에피소드 내 상관, 추정).
- 군집 내 효과 검출에는 에피소드 >= 100 이 필요 (§2 검정력 표). 더 필요하면 새 OoD 에피소드를 생성한다. 크기를 연속 샘플링한 에피소드 200-400 이면 r >= 0.14-0.20 을 검출 가능하다.
- 시드: probe 시드 5, 모델 학습 시드 3 이상 (현재 1).
- 가설과 성공 기준은 실험 전에 `notes/hypotheses.md` 에 적는다 (CLAUDE.md 규칙).

## 8. 한계와 불확실성

- 문헌 항목은 기억 기반이며 검색으로 재확인하지 않았다. 수치(특히 Green 1991 규칙, arXiv 번호) 는 인용 전에 확인 필요.
- 로컬에 데이터/체크포인트가 없어 라벨 분포나 천 크기 분포를 직접 계산하지 못했다. 위의 데이터 수치는 모두 upstream 브랜치 README/SETTINGS 의 서술이다 (재계산 아님). 각 수치의 원자료는 README 가 가리키는 Drive `<env>_h1f5_quant.json` 이다.
- 이 파일에 계산 코드는 검정력 표 한 줄 (§2) 뿐이다.

## 9. 확신도

중간-낮음. 방법론 원칙(에피소드 단위 분할, control task, cluster bootstrap, 시드 보고)은 문헌에 확실한 근거가 있어 확신이 높다. 그러나 (1) 물리 파라미터 probing 의 직접 선행연구 인용이 기억 기반이고 재확인하지 않았으며, (2) 로컬에 데이터가 없어 라벨/크기 분포를 검증하지 못했고, (3) 붕괴 원인에 대한 해석은 단일 시드 README 수치에 의존하는 추정이다.
