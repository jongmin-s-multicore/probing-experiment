# 문헌 종합: DINO-WM의 physical attribute 보존(H1) 및 활용(H2)과 구분 검증 방법

- 작성: reporter, 2026-10-30. 실험은 하지 않았다. 가설은 미확정이다.
- 입력: notes/research/analyst-prior.md, analyst-stats.md, analyst-skeptic.md, sources.md
- 표기 규칙
  - (prior), (stats), (skeptic)은 해당 주장을 한 분석가를 뜻한다. 여러 명이 같은 내용을 쓴 경우 함께 적었다.
  - 분석가가 붙인 [추정], [미확인], [검색 요약만], [기억 기반], [읽음], [확인] 표시는 그대로 유지했다.
  - 이 문서의 수치는 모두 분석가 파일에서 옮긴 것이다. reporter는 PDF를 직접 대조하지 못했다(PDF 추출 도구를 쓸 수 없었다). 따라서 모든 논문 수치는 "분석가 전사, reporter 원문 대조 안 함"이다. 둘 이상의 분석가가 같은 값을 적은 경우는 "일치"로 표시했다.
  - 분석가들의 읽은 범위는 본문 위주이며 일부 부록은 읽지 않았다.

---

## 1. 요약

1. 기존 문헌에는 H1(희석)과 H2(보존되나 비활용)를 직접 가르는 실험이 없다. DINO-WM predictor 내부 layer별 probing 선례도 없다. (prior, stats 모두 "미발견/없음"으로 기록. skeptic은 이를 직접 말하진 않았으나 predictor 수준 증거로 VoE 하나만 있다고 적었다.)
2. LeWM Table 1(PushT)의 DINO-WM 행은 위치와 각도를 높게 복원한다. Linear r 0.977~0.997, MLP r 0.995~0.999이다. 세 분석가 모두 이것을 "frozen encoder 출력 단계의 보존은 높게 보고됨"으로 읽었다. 다만 이 행이 어떤 표현(patch 전체, pooled, CLS, predictor 출력)을 probe한 것인지 원문에 불명확하다는 점도 세 명 모두 지적했다.
3. 세 분석가 모두 H1의 "transformer layer"가 frozen DINOv2 encoder인지 학습되는 6층 predictor인지 먼저 정해야 한다고 썼다. 구분 방식의 세부는 서로 다르다(5절).
4. 세 분석가 모두 probing은 "읽을 수 있음"만 보이고 "사용함"은 보이지 않으므로 H2에는 개입 실험이 필요하다고 썼다. 이 점은 외부 문헌(Amnesic probing 등)에 의존하며 그 문헌은 검색 요약 또는 기억 기반이다.
5. 판정 임계값, 시드 수, 해석 틀(예: 어떤 층 비교를 H1의 증거로 볼지)에서 분석가 의견이 갈린다. 합치지 않고 5절에 나란히 적었다.

---

## 2. 관련 논문별 정리

### 2.1 DINO-WM (Zhou, Pan, LeCun, Pinto; ICML 2025; arXiv:2411.04983)
- 구조: frozen DINOv2 patch embedding z_t 위에 학습되는 ViT predictor를 둔다. frame 단위 causal attention을 쓰고 action은 각 patch 벡터에 concat한다. 손실은 teacher forcing 하의 L_pred (prior [읽음], skeptic [확인], DINO-WM Sec 3.1, Eq.1).
- predictor 규모: depth 6, 16 heads, MLP dim 2048, 약 19M parameters (stats, 부록). SG-JEPA가 쓴 DINO-WM 설정은 H=20, 6 layers, 16 heads, MLP 2048, dropout 0.1 (stats).
- planning: CEM으로 MSE(ẑ_T, z_g)를 최소화 (prior). planning은 autoregressive rollout이다 (skeptic, DINO-WM 3.2).
- encoder ablation (DINO-WM Table 2, prior·skeptic 일치): PushT SR에서 DINO patch 0.90, DINO CLS 0.44 등. prior는 저자 해석("patch가 공간 정보를 보존")을 가설 수준으로 기록했다.
- physical attribute probing은 본문에 없다(prior, stats 일치). decoder 복원 지표(LPIPS 등, Table 4)만 있다(prior). decoder 복원은 보존의 증거이긴 하나 미세한 physical attribute 보존의 증거는 아니다(prior).
- 저자 명시 한계: 데이터 커버리지, action label 의존, 계층적 제어 부재. predictor가 무엇을 쓰는지는 분석하지 않았다 (prior).

### 2.2 LeWM / LeWorldModel (Maes et al.; 2026; arXiv:2603.19312)
- 구조: ViT-tiny encoder(약 5M, 192-d 단일 latent) + 6-layer predictor. action은 AdaLN으로 주입 (prior). DINO-WM과 표현 형태가 다르다(프레임당 벡터 1개 대 patch 토큰 다수) (prior, skeptic 일치).
- probing 설정(stats, LeWM Sec 5.1·App F.2): linear와 MLP 둘 다, 지표는 MSE와 Pearson r. 샘플 수, probe hyperparameter, 분할 방식, probe 시드 수는 stats가 "미확인", skeptic이 "내가 읽은 범위에서 적혀 있지 않음"이라고 적었다. 표의 ± 의미도 불명(stats는 샘플 std로 추정 [추정]).
- 선택성 대조군(random-init, shuffled-label)은 표에 없다 (stats [미확인]).
- 수치 (stats 전사, 논문 Table 1, PushT, DINO-WM 행; prior도 r 값 일치):

| 속성 | Linear MSE | Linear r | MLP MSE | MLP r |
|---|---|---|---|---|
| Agent Location | 1.888 ± 0.500 | 0.977 | 0.003 ± 0.022 | 0.999 |
| Block Location | 0.006 ± 0.007 | 0.997 | 0.002 ± 0.003 | 0.999 |
| Block Angle | 0.050 ± 0.101 | 0.979 | 0.009 ± 0.052 | 0.995 |

  비교 행 예: Block Angle Linear r은 PLDM 0.745, LeWM 0.902 (stats, prior 일치). Agent location Linear MSE는 LeWM 0.052, r 0.974 (stats).
- Table 3 (TwoRoom, agent position, stats·skeptic 일치): DINO-WM Linear r 0.824, MLP r 0.999. LeWM과 PLDM은 Linear r 0.996. 즉 DINO-WM은 정보가 있으나 선형으로는 덜 읽힌다 (skeptic).
- Table 4 (OGBench-Cube, DINO-WM 행, stats): Joint Velocity Linear r 0.763, End-Effector Position 0.996, End-Effector Yaw Linear 0.828 / MLP 0.917, Block Quaternion Linear 0.257 / MLP 0.411, Block Yaw Linear 0.176 / MLP 0.304, Overall Linear 0.725 / MLP 0.799. skeptic도 Overall 0.725 / 0.799를 적었고, 같은 표에서 LeWM Overall은 0.477 / 0.584라고 적었다(skeptic만 기재; stats는 LeWM/PLDM 행을 "파일 참조"로 남김). 논문 본문은 "All three methods struggle to recover block orientation"이라 서술한다 (stats 인용).
- Table 5 (PushT 성공률, 학습 시드 3개, stats·skeptic 일치): DINO-WM 92.0 ± 1.63, PLDM 78.0 ± 5.0, LeWM 96.0 ± 2.83.
- 저자 해석(Table 1 캡션): DINO-WM의 강한 probing은 foundation-model 사전학습 덕일 수 있다. 사전학습 규모를 캡션은 "~124M images"로 적었다 (prior, stats, skeptic 일치). skeptic은 DINOv2 원 논문이 LVD-142M으로 알려져 있다고 적었으나 [미확인]이다. 어느 쪽 수치도 reporter가 대조하지 못했다.
- Table 3 캡션 (skeptic [확인]): TwoRoom의 planning gap은 "덜 informative한 표현" 때문이 아니라 dynamics model이나 planning procedure 같은 다른 요인 때문이라고 저자가 해석한다.
- VoE (Fig.14, skeptic [확인]): DINO-WM의 surprise는 TwoRoom과 PushT에서 teleport에 반응하지만 색 변화에도 비슷하게 반응한다. Cube에서는 두 교란 모두 유의한 반응이 없다.
- 그 밖: LeWM의 DINO-WM 비교에서는 proprioception을 제거했다(prior, skeptic 일치). Fig 7에서 end-effector angle 같은 세부는 완전히 보존되지 않는다고 서술(prior).

### 2.3 G-JEPA (Man, Chen, Guo; 2026; arXiv:2610.28483)
- 방법: observed state(S), nuisance invariance(I), predicted state(P) 보조 손실로 physical grounding. 진단이 아니라 개선책이다 (prior, stats 일치). Intro에서 frozen encoder가 task-specific physical structure 학습을 막는다고 DINO-WM 등을 겨냥한다 (prior).
- Table 3 (PushT-Ada / OGBench Cube 성공률 %, stats 전사): 기준 56.90±2.35 / 72.80±2.53, S만 55.73±1.58 / 75.40±1.65, S+I 79.43±0.90 / 76.20±2.57, S+I+P 79.43±1.58 / 78.60±4.12. prior·skeptic도 PushT-Ada의 56.90, 55.73, 79.43을 동일하게 적었다.
- Table 5 (skeptic [확인]): action prior(AD) 추가 시 Cube 72.80에서 100. prior는 PushT-Ada 최종 87.11(AD 포함)을 적었다(prior만 기재).
- 통계 설정 (stats): 주요 설정은 training seed 1개이고 ± 는 planning seed 간 표본 std. skeptic도 training seed 1개로 일치.
- 데이터 분할 (skeptic [확인], G-JEPA p.6): PushT는 window 단위 90/10 분할이고 Cube와 TwoRooms는 episode 단위이다. 누수 위험을 skeptic이 지적했다.
- DINO-WM과 직접 비교하지 않는다 (prior).

### 2.4 SG-JEPA / Semigroup-JEPA (Liu et al.; 2026; arXiv:2609.10464)
- 평가 틀 (prior, stats, skeptic 일치): frozen 표현 위에 MLP probe(2-layer MLP, NMSE)를 학습하고, rollout latent의 probe 오차에서 GT 인코딩 latent의 probe 오차를 뺀 "excess NMSE"(Eq.4)를 정의한다. 세 분석가 모두 이 지표를 차용할 만하다고 봤다.
- probe 프로토콜 (stats, App C.2): LayerNorm → GELU 512 → GELU 256 → linear, dropout 0.05, AdamW lr 1e-3, wd 1e-5, batch 256, 최대 50 epoch.
- 통계 관행 (stats): Fig 5(b) predictor seed 3개와 95% bootstrap CI, Fig 5(c) 2,800 paired OOD episode의 95% simultaneous interval. 일부 sweep은 1 seed.
- DINO-WM baseline은 원 구현과 다르다. frozen DINOv2 encoder + LeWM 방식 action conditioning (prior, skeptic 일치). control에서는 mean-pooled frozen DINOv2 patch를 쓴다 (prior, skeptic 일치).
- 결과 (prior 전사): 2D square에서 DINO-WM 대비 오차 31~48% 감소, 3D Approach Ball에서 약 34% 감소, Arm Catcher SR 9.5% (DINO-WM) 대 23.3% 등. DINO-WM은 horizon이 길어지면 오차가 누적(Fig 4a)한다 (prior). stats는 teacher-forced far-OOD 중력에서 국소 오차 약 32% 감소(논문 서술)라고 적었다.
- encoder 고정 + predictor 교체 설계 (stats, Fig 5, "fresh predictor"): fresh GRU 1.376 대 1.555, fresh Transformer 1.269 대 1.453. 단위와 어느 쪽이 어느 조건인지는 stats 기록에 명시되지 않아 해석 보류.
- 서술 충돌 (skeptic [확인], prior [미확인]으로 인지): p.2에서는 encoder 재학습은 성능 불변이고 predictor 재학습은 성능을 떨어뜨린다고 쓰고 "이득은 GRU predictor가 배운 것과 관련"이라 쓴다. abstract와 p.10에서는 "most of the gain comes from the encoder learning better features"라고 쓴다. 이 논문을 "encoder 대 predictor 분리"의 근거로 인용할 때 주의가 필요하다.
- skeptic [확인]: Fig.5c에서 teacher-forced 차이는 작고 free-rollout 차이는 크다.
- 한계: 합성 MuJoCo 환경, 부록 미확인 (prior).

### 2.5 T2-z (Zhou et al.; 2026; arXiv:2610.22364)
- 관측되지 않는 환경 요인(마찰, 질량 등)을 learnable environment latent로 추정. 백본은 Wan2.2-TI2V-5B 기반 BWM으로 DINO-WM 계열이 아니다 (prior).
- 수치 (prior 전사, Table 1): 평균 physical error 12.86에서 4.80, action success 37.9%에서 74.2%. Friction에서 Standard 32.0% 대 T2-z 72.0%. stats는 Table 1에 DINOv2 Context baseline이 있다고 적었다. PCA 정성 결과(Fig 1b)의 정량 지표는 stats가 [미확인].
- 세 분석가 모두 같은 함의를 뽑았다. 단일 프레임에서 관측 불가능한 속성(마찰, 질량)은 H1/H2와 다른 문제이므로 attribute를 분류해야 한다.

---

## 3. H1·H2에 대한 기존 근거

### 3.1 H1 (희석/소실)
| 구분 | 근거 | 출처 | 한계 |
|---|---|---|---|
| encoder 출력 단계의 보존 | LeWM Table 1의 DINO-WM 행 (2.2절) | prior, stats, skeptic | probe 대상 표현 불명, 시드/CI 없음, 선택성 대조군 없음. stats: 이 표가 predictor 통과 후인지 불명하므로 H1 근거로 쓰면 안 된다. |
| 보존이 약한 속성 | Cube의 회전 계열(Block Yaw, Quaternion)은 DINO-WM도 r이 낮다 (Table 4) | stats, skeptic | stats: "소실"이 아니라 encoder 단계 부재일 가능성 [추정]. |
| predictor 수준 | DINO-WM의 horizon에 따른 오차 증가 (SG-JEPA Fig 4a) | prior | prior: rollout 누적 소실과 일치하나 인과 구분은 안 됨 [추정]. |
| predictor 수준 | VoE에서 DINO-WM이 teleport와 색 변화에 모두 반응 (LeWM Fig.14) | skeptic | skeptic: 물리 특이성이 없음. 기존의 유일한 predictor 수준 증거이나 H1을 가르지는 못함. |
| 이론적 배경 | rank collapse (Dong et al. 2021), layer별 정보 (Skean et al. 2025) | prior | [검색 요약만]. skip connection이 있는 실제 ViT에서는 collapse가 H1의 근거가 되지 않을 것으로 prior가 [추정]. |
| 직접 근거 | DINO-WM predictor 내부 layer별 probing | 없음 | prior, stats가 "문헌에서 미발견/선행 수치 없음". |

### 3.2 H2 (보존되나 비활용)
| 근거 | 출처 | 한계 |
|---|---|---|
| G-JEPA Table 3에서 상태 supervision(S)만 추가 시 PushT-Ada 56.90에서 55.73으로 개선 없음 | prior, stats, skeptic 모두 인용 | stats: training seed 1개, 차이가 1 sd 미만이라 약함. skeptic: nuisance invariance(I) 효과가 섞인 비교. prior: "정보 주입이 곧 활용이 아님"을 시사하나 해석은 [추정]. |
| LeWM Table 3 캡션: planning gap의 원인이 표현이 아니라 dynamics model/planning 쪽 | skeptic [확인] | 저자 해석이며 DINO-WM에 대한 개입 검증은 아님. |
| G-JEPA Table 5: action prior 추가로 Cube 72.80에서 100 | skeptic [확인] | 병목이 world model 표현이 아니라 action 탐색일 수 있다는 간접 근거 (skeptic). |
| SG-JEPA의 encoder-predictor 교체 실험 | prior, stats, skeptic | skeptic: 서술이 충돌하므로 인용 주의. prior: 보존과 활용을 연결해서 보는 입장에 가깝다. |
| 직접 근거 (DINO-WM에서 attribute 개입 검증) | 없음 | prior: "world model이 latent의 attribute 정보를 무시/복사 shortcut" 연구도 이번 검색에서 미발견. |

---

## 4. 검증 방법 후보

세 분석가의 제안을 방법별로 묶었다. 어느 분석가의 제안인지 병기한다.

### 4.1 Probing
- linear와 MLP를 모두 사용한다 (prior, stats, skeptic 일치). 근거: LeWM도 둘 다 사용하고 Block Angle에서 격차가 있다 (prior).
- 대조군/기준선
  - label shuffle 또는 control task의 selectivity (prior, stats, skeptic). Hewitt & Liang 2019는 prior가 [검색 요약만(서지 미확인)], stats가 기억 기반, skeptic이 [미확인]으로 표시했다.
  - random-init predictor/encoder 기준선 (prior, stats, skeptic).
  - 시간 index probe (skeptic), raw 입력/단순 풀링 기준선과 PCA로 차원을 맞춘 probe (stats). skeptic도 PCA k=192 등으로 차원 통일 제안.
  - GT z_{t+1}(encoder 출력)에 같은 probe를 적용한 상한 (prior).
- 분할: episode 단위 (stats, skeptic). prior도 train/val을 episode 단위로 분리하도록 제안.
- 지표: Pearson r 단독을 쓰지 말고 R²와 MSE를 함께 (stats). LeWM Table 1에서 DINO-WM agent location이 Linear MSE 1.888인데 r 0.977인 불일치를 stats와 skeptic이 모두 지적했다. 원인은 둘 다 [추정]이다.
- 교란: 속성 간 상관(agent 위치, block 위치, 시간 index, action)이 높아 probe가 대리 변수를 읽을 수 있다 (stats, skeptic). 입력 pooling 방식이 결론을 좌우한다 (stats, skeptic, prior 일치). skeptic은 pooling을 바꾸면 "정보를 잃는다"는 결론이 인위적으로 만들어질 수 있다고 썼다.
- 다양한 속성 분류: 단일 프레임 관측 가능(위치, 각도) / 시간 창 필요(속도) / 상호작용으로만 드러남(질량, 마찰) (prior, stats, skeptic 일치).

### 4.2 레이어별 분석
- 측정 지점
  - prior: DINOv2 중간 layer에서 최종 layer, predictor 입력(action concat 직후), predictor 각 block, 출력 ẑ_{t+1}.
  - stats: R0(frozen DINOv2 patch, predictor 입력 직전), R_l (l=1..6, predictor 각 layer 출력), R_pred(예측 latent). 현재 y_t 복원은 보존(H1 축), 미래 y_{t+k} 복원은 활용(H2 축)으로 분리.
  - skeptic: GT 인코딩 z_{t+k}, teacher-forced ẑ_{t+1}, open-loop ẑ_{t+k}, predictor 각 layer.
- 레이어 효과 정의
  - stats: ΔR²(l) = R²(R_l) − R²(R0). SG-JEPA식 차감으로 레이어별 소실량 정의.
  - skeptic: 수정된 H1은 excess error(Eq.4 방식)가 무작위 부분공간 또는 외형 속성(색 등)의 excess error 기준선보다 큰지로 판정.
- 보조 지표: effective rank, participation ratio, 토큰 간 cosine similarity (prior, stats). stats는 rank가 물리 정보량 자체가 아니므로 보조지표로만 쓰라고 경고. y-정렬 부분공간 에너지(CCA/PLS) (stats).
- 외부 근거: Zhan et al. 2024(layer별 frozen feature probe)와 Skean et al. 2025는 prior가 [검색 요약만]으로 표시했다.

### 4.3 개입
- 선형 제거 (INLP/LEACE 계열): 입력 또는 중간 layer에서 속성 방향을 투영 제거하고 예측 오차 변화를 본다. 동일 차원 무작위 부분공간 제거와 비교 (prior, stats, skeptic 일치). 비선형으로 중복 저장되면 선형 제거만으로는 불완전하다 (skeptic). skeptic은 제거 후 MLP probe로 정보가 실제로 사라졌는지 먼저 확인하고, 사라지지 않았으면 H2에 대한 결론을 내리지 않는다는 조건을 둔다.
- 조작/steering: probe 방향으로 이동시켰을 때 예측의 probe 값이 일관되게 바뀌는지 (stats 기울기, skeptic Δ 이동).
- Activation patching, 반사실 쌍: 시뮬레이터에서 한 속성만 다른 쌍을 만들어 출력 차이를 본다 (prior, stats, skeptic). stats는 이 환경이 쌍 생성을 허용하는지 [미확인].
- Fresh predictor: frozen encoder 위에 predictor를 새로 학습해 비교 (stats, SG-JEPA Fig 5 설계 차용).
- 보조 supervision: predictor에 물리 상태 보조 head/손실을 추가했을 때 개선 여부 (stats, G-JEPA 유사). 개선 없으면 한계가 다른 곳에 있다는 간접 증거 (stats).
- 그 밖: action ablation(0/랜덤), identity(마지막 프레임 복사) baseline 비교 (prior만 제안; shortcut을 다룬 world-model 문헌은 미발견). planning 수준 확인(goal 프레임의 속성만 바꿔 CEM이 구분하는지, prior; planning SR 측정, skeptic).
- 외부 근거: Amnesic probing(Elazar et al. 2021)은 skeptic이 웹 검색으로 서지와 요지 확인 [확인], prior는 [검색 요약만]. INLP/LEACE는 stats가 기억 기반이라 표시. 모두 NLP 맥락이며 world model latent에 그대로 옮겨도 되는지는 [추정] (skeptic).

### 4.4 통계 설계
- 단위와 CI: 에피소드 단위 cluster bootstrap (B ≥ 2,000), heavy-tail 때문에 median, trimmed mean, log-MSE 병행 (stats). LeWM 표의 sd가 mean보다 큰 셀이 많다는 점은 stats가 지적.
- 시드 종류 분리: probe 초기화, 분할, world model 학습 시드 (stats). G-JEPA처럼 planning seed만 반복하면 학습 분산이 빠진다 (stats, skeptic이 G-JEPA의 training seed 1개를 같이 지적).
- 등가성 검정(TOST): "보존됨"은 귀무가설 채택이므로 "유의하지 않음"을 "같다"로 읽지 않도록 사전 정의한 ε로 검증 (stats만 제안).
- 다중비교: 속성 × 레이어 × probe 종류로 비교 수가 많으므로 가족을 사전 정의(레이어 기울기 하나로 묶기)하거나 Holm/BH 보정 (stats만 제안).
- 사전 고정: 물리량 목록, 측정 위치, probe 종류/정규화/분할, control, 효과 크기 기준, 시드 수 (skeptic, stats 일치 취지). skeptic은 config.yaml에 DINOv2 체크포인트(크기, register 유무), 사용 layer, pooling, 분할 seed, 체크포인트 출처도 고정하라고 적었다.
- 파일럿: LeWM Table 1 DINO-WM 행의 probe 대상 확인, learning curve로 n 결정, 시드 분산 사전 측정 (stats). skeptic도 공개 DINO-WM 체크포인트로 PushT에서 GT DINOv2 feature를 probe해 LeWM Table 1과 맞는지 재현 확인을 제안.

---

## 5. 의견 차이

### 5.1 세 분석가가 일치한 점
1. "transformer layer"가 frozen DINOv2 encoder인지 학습되는 predictor인지 먼저 정의해야 한다.
2. LeWM Table 1은 DINO-WM 표현이 높은 probing 성능을 가진다고 보고하며, 그 표현이 정확히 무엇인지는 원문에 불명확하다.
3. probing은 활용을 증명하지 못하므로 H2는 개입으로 검증해야 한다.
4. attribute를 단일 프레임 관측 가능 / 시간 필요 / 관측 불가로 분류해야 한다 (T2-z 근거).
5. 선택성 대조군(label shuffle 등)과 random-init 기준선, episode 단위 분할이 필요하다.
6. SG-JEPA의 excess error 지표는 차용할 만하다.

### 5.2 의견이 갈리거나 강조가 다른 점

**(a) H1을 어떻게 구분하는가**
- prior: H1a(DINOv2 layer를 지나며 소실)와 H1b(predictor layer를 지나며 소실)로 나눈다. LeWM Table 1을 근거로 H1a는 약하다고 본다(확신도 "중간", Table 1 한 건에 의존). H1b에서는 probe 정확도가 layer를 따라 단조 감소하면 지지, 거의 일정하면 기각이라는 해석을 제시한다.
- skeptic: (a) DINOv2 내부, (b) predictor 중간 layer, (c) predictor 출력의 세 해석으로 나눈다. (c)로 고정하고 teacher-forced 1-step과 open-loop k-step을 분리하라고 제안한다. (b)는 중간 layer가 DINOv2 feature 공간에 있을 의무가 없어서 probe 성능 저하가 "계산 중"인 정상 현상일 수 있으며 기준선이 없다고 본다. 즉 prior가 단조 감소를 H1b의 증거로 쓰는 것에 대해 skeptic은 중간 layer의 하락만으로는 희석이라 부르기 어렵다는 입장이다. DINOv2 내부(a)는 H1'로 분리한다.
- stats: 지점을 R0, R_l(1..6), R_pred로 정의하고 H1을 "I(R_l; y_t)가 l 증가에 따라 감소 또는 R_pred 복원이 R0보다 유의하게 낮음"으로 쓴다. stats의 운용 정의는 predictor 쪽만 다루며 DINOv2 내부 layer는 포함하지 않는다. H1과 H2는 배타적이지 않다고 보고 2축 표(정보량 곡선 × 인과 기여도)를 권장한다.

**(b) 어떤 증거를 보존의 증거로 인정하는가**
- prior: LeWM Table 1을 근거로 H1a에 불리하다고 판단하되 "probe 설정 불명"이라는 단서를 둔다. 불일치하면 probe 설정 차이를 점검하라고 한다.
- stats: 그 표가 predictor 통과 전인지 후인지 불명이므로 확인되기 전에는 H1의 근거로 쓰면 안 된다고 한다.
- skeptic: LeWM 자료는 DINO-WM의 보존을 지지하지만 이것이 frozen encoder 출력에 대한 것일 가능성이 높고 [추정], 그렇다면 predictor에 관한 H1(c)에는 정보를 주지 않는다고 본다. 또한 LeWM Table 1과 4의 MSE-r 불일치를 근거로 probe 인공물(A4) 가능성을 크게 본다.

**(c) 희석의 대안 설명**
- skeptic만 다음을 대안 설명으로 명시했다. A1 손실 가중([추정]: patch MSE가 분산 큰 외형 방향을 우선), A2 오차 누적, A3 planner 병목, A4 probe 인공물, A5 관측 불가능성. prior는 rollout 누적을 H3로, copy shortcut을 H4로 제시했고 stats는 대안 설명 목록을 두지 않았다.

**(d) 시드 수와 판정 기준**
- stats: 모델 학습 시드 최소 5(가능하면 8~10), probe 시드 레이어×속성당 5 이상. 시드 3 이하는 탐색적으로 표시. 정규근사 손계산이며 코드 검증은 하지 않았다 [추정]. 예시 임계값은 ε1 = 0.10(ΔR²), 선택성 게이트 0.1, 부분공간 에너지 50% 감소 등이다. 모두 "예시, 사람 확정 필요".
- skeptic: 시드 최소 3개. 임계값을 수치로 제시하지 않고 무작위 부분공간 또는 외형 속성 기준선 대비로 판정한다.
- prior: 예시 임계 0.05(probe r 상대 감소), "팀 합의 필요". 구체적 시드 수는 제시하지 않았다.

**(e) H2 판정의 틀**
- prior: 개입에 둔감하면 H2 지지. 보존 유지 + 개입에 민감하나 rollout 오차 누적이면 둘 다 아님(누적 오차 문제).
- stats: H1 기각(TOST) + 무작위 대비 제거 효과 차이 없음 등 4개 비활용 조건(제거, patching, steering, fresh predictor/보조 supervision) 중 하나 이상을 H2 지지 조건으로 제시.
- skeptic: H2를 "제거해도 오차가 커지지 않는다"는 주장으로 정의하고, 제거 후 MLP probe로 정보가 사라졌는지 확인되지 않으면 결론을 내리지 않는다. 또한 "predictor가 활용하지 못해서 planning이 실패한다"는 주장은 planner, 비용 함수, 데이터 coverage를 통제하기 전에는 과장일 수 있다고 사전 경고.

**(f) SG-JEPA 해석**
- prior: 초록의 "encoder가 predictor가 carry forward할 feature를 더 잘 보존"을 "보존 ↔ 활용이 연결된 입장"으로 읽고, 서론 설명의 방향이 다르게 읽혀 부록 확인이 필요하다고 [미확인] 표시.
- skeptic: p.2와 abstract/p.10의 서술이 충돌함을 구체적으로 인용하고 "encoder 대 predictor 분리"의 근거로 인용하는 것에 주의하라고 권고.
- stats: 이 교체 설계 자체를 H2 검증(fresh predictor 대조)에 재사용하자는 입장이며 서술 충돌은 언급하지 않았다.

### 5.3 한 명만 지적한 점
- prior만: identity(copy) predictor baseline과 action ablation, rank collapse 이론과 skip connection 논의, Hewitt & Liang·Zhan 등 추가 문헌 표(검색 요약), PSG-JEPA·POKEWORLD 등 관련 후속 연구 소개.
- stats만: 샘플 독립성과 cluster bootstrap, TOST 등가성 검정, 다중비교 보정, 검정력 계산, MI/MDL 대안, learning curve로 필요 데이터 양 결정, G-JEPA의 ± 는 planning seed 변동이라는 점.
- skeptic만: G-JEPA PushT의 window 단위 분할에 따른 누수 위험, LeWM F.2의 분할 방식 미기재, DINO-WM 이름이 논문마다 다른 모델을 가리킨다는 점의 상세 표(proprio, epoch, frameskip 등), VoE의 색 변화 반응, G-JEPA Table 5의 action prior 효과, 환경에 따라 LeWM 대 DINO-WM 우열이 바뀐다는 점(PushT 96 대 92, Cube와 TwoRoom은 DINO-WM 우세), 가설 정의의 "치명/중요" 등급 평가.

---

## 6. 미확인·한계

- reporter는 PDF 대조를 하지 못했다. 이 문서의 모든 논문 수치는 분석가 파일에서 옮긴 것이다.
- LeWM Table 1의 DINO-WM 행이 어떤 표현을 probe했는지: prior [미확인], stats [미확인], skeptic "읽은 범위에서 명시 없음". 샘플 수, 분할, 시드 수, ± 의미 불명(stats).
- Table 1/4의 MSE-r 불일치 원인: skeptic은 코드를 보지 않아 확정할 수 없다고 했고 stats도 [추정]으로 표시.
- 읽지 않은 부록: LeWM F.2 일부(prior는 부록 미확인, skeptic은 부록 p.1-26 읽음으로 읽은 범위가 다름), DINO-WM 부록, SG-JEPA 부록, G-JEPA·SG-JEPA·T2-Z 부록(skeptic).
- DINOv2 사전학습 규모: LeWM 캡션 ~124M images 대 skeptic 기억의 LVD-142M [미확인]. 어느 쪽도 대조되지 않음.
- 외부 문헌 상태: Zhan 2024, arXiv:2401.00463(제목/저자 미확인), Garrido 2025, Tan 2026, Yan 2026(PSG-JEPA), Dong 2021, Skean 2025, Elazar 2021(prior는 검색 요약만; skeptic은 웹 검색으로 확인), Hewitt & Liang 2019(서지 미확인), Voita & Titov, Ravfogel, Belrose, Belinkov, Ravichander(stats·skeptic 기억 기반/[미확인]). 인용 전 원문 확인 필요.
- 분석가가 본 확신도: prior는 5편 정리 "높음", 추가 문헌 "낮음~중간", 설계 "중간", H1a 약하다는 판단 "중간". stats는 "중간-낮음". skeptic은 "중간".
- 정량 기준(임계값, 시드 수, 표본 수)은 전부 분석가의 예시이며 근거 수치가 아니다. 표본 수식과 검정력은 stats가 코드로 검증하지 않았다 [추정].
- 이 폴더 5편에는 DINO-WM predictor 내부 layer별 probing, "보존과 활용을 분리한" world-model 연구, 관측 불가/가능 attribute를 구분한 DINO-WM 계열 평가가 없다 (prior 연구 공백 4항, stats "공통 공백").
- 5편 외의 범위(sources.md B~E)는 읽지 않았다. DINOv3 등 B 분류 파일은 분석 대상이 아니었다.

---

## 7. 다음 단계 제안 (실험 아님)

1. 사람 결정: H1의 정의를 encoder 단계(DINOv2)와 predictor 단계(layer별, 출력, rollout)로 나눌지, 어느 쪽부터 다룰지 (5.2(a)에 세 분석가의 입장이 갈려 있다).
2. 문서 확인 작업: LeWM 부록 F.2 및 공개 코드에서 Table 1 DINO-WM 행의 probe 대상 표현, 분할 방식, ± 의미, 시드 수를 확인한다. 이 확인이 끝나기 전에는 Table 1을 H1의 근거로 쓰지 않는다 (stats 권고와 일치).
3. 인용 정리: SG-JEPA 부록에서 encoder-predictor 교체 실험의 방향을 확인하고, 외부 문헌(Hewitt & Liang 등) 서지를 원문으로 대조한다.
4. 사전 고정 목록 작성: 속성 목록(단일 프레임 / 창 / 상호작용 분류), 측정 위치, probe 종류와 정규화, 분할 단위, control, 시드 수를 hypotheses.md의 성공 기준에 반영하기 전에 사람이 확정한다 (skeptic, stats 권고).
5. 위 사항이 정해지면 notes/hypotheses.md의 후보 3개 중 사람이 고른 것에 대해 /new-experiment 단계로 넘어간다. GPU/비용이 큰 실행 전에는 사람 승인이 필요하다 (CLAUDE.md).

---
---

# [두 번째 종합] acwm-phys(Cloth Move)에서 DINO-WM의 천 크기 수축: probing 선행연구와 원인/해결 방향

- 작성: reporter, 2026-10-31. 실험은 하지 않았다. 가설은 미확정이다.
- 위쪽 내용(2026-10-30, PushT의 H1/H2 종합)은 다른 주제이며 보존했다. 이 절은 별개 주제다.
- 입력: notes/research/prior.md, notes/research/stats.md, notes/research/skeptic.md (세 분석가가 독립적으로 작성).
- 표기: (prior), (stats), (skeptic)은 주장한 분석가다. 분석가의 [추정], [확인], [검색확인], "미확인" 표시는 그대로 유지했다. 논문 수치는 분석가 파일에서 옮긴 것이며 reporter는 원문과 대조하지 않았다.

## 1. 읽기 전 주의 (반드시 반영할 한계)

1. 문헌 목록의 신뢰도가 분석가마다 다르다. stats는 웹 검색 없이 기억에 의존해 문헌 목록을 썼고, 저자/연도/arXiv 번호는 인용 전에 검증이 필요하다고 명시했다 (stats 범위 주의, 8절). prior는 초록/스니펫 위주로 확인했고 2026년 preprint 수치는 2차 요약이라 원문 확인이 필요하다고 했다 (prior 범위 주의). skeptic은 [검색확인]한 문헌도 서지와 요지만 보고 본문 수치는 대조하지 않았다 (skeptic 확신도).
2. acwm-phys 논문(arXiv 2605.08567)이 평가한 모델은 DINO-WM이 아니라 pixel-space 비디오 DiT(ACWM-DiT)다 (prior 1.1, 5절 3항). 논문의 Cloth Move 수치(InD MSE 10.667, OoD 23.820 등)는 우리 DINO-WM 현상의 직접 증거가 아니다. 저자들은 probing을 하지 않았다 (prior).
3. skeptic은 acwm-phys 브랜치 코드를 직접 읽지 못했다. 원격 ref만 있고 checkout이 없다. skeptic이 acwm-phys에 관해 쓴 내용(학습 크기 분포, 조건화 입력, 정규화)은 main 브랜치 코드에 근거한 추정이다 (skeptic 서두). 반면 stats는 `git show upstream/hanbin5/acwm-phys:<경로>`로 해당 브랜치의 README/SETTINGS를 읽었다 (stats 0절). 두 분석가의 코드 접근 범위가 달라 근거의 층위가 다르다.
4. stats가 인용한 현상 수치는 모두 upstream README 서술이며 재계산이 아니다. 로컬에 데이터와 체크포인트가 없다 (stats 0절, 8절). 단일 시드, 단일 설정이다.

## 2. 배경

- 현상: acwm-phys의 Cloth Move에서 작은(학습 크기) 천으로 학습한 DINO-WM이 큰 천을 받으면 학습 크기로 수축한다. stats가 옮긴 README 수치: 마지막 프레임 천 픽셀이 큰 천 real 4,475 대 model 1,836, InD 1,819 / 1,888 (real / model), 작은 천 real 404 대 model 1,129. 단일 시드이고 에피소드 단위 bootstrap CI를 쓴다 (stats 0절). 출처: stats.
- 수축은 크기가 큰 쪽만의 문제가 아니라 작은 천에서도 학습 크기 쪽으로 커진다는 점이 수치상 보인다(404 대 1,129). 이 해석은 reporter가 stats의 숫자를 읽은 것이며 stats가 직접 쓴 문장은 아니다.
- 데이터 규모: stats(README)는 학습/검증 1800/200, InD test 100, OoD test 100(면적 0.26x 50개, 2.5x 50개)이라 적었다. prior는 논문 Table 8 기준으로 학습 1,987 / InD 100 / OoD 100이라 적었다. 두 값(1800+200 대 1,987)은 같은 버전의 수치인지 확인되지 않았다 (미확인).
- OoD 정의: prior는 논문 서술상 OoD split이 "천 크기와 초기 배치 변경"이라 크기 단일 효과가 분리되지 않는다고 했다. stats가 읽은 README는 면적 0.26x / 2.5x라고 적는다. 두 서술이 어떻게 맞물리는지(초기 배치도 바뀌는지)는 어느 분석가 파일에도 정리되어 있지 않다 (미확인).
- 크기 라벨 문제: metadata의 cloth_dimx/y는 보이는 크기와 상관이 r ≈ 0.02라서 라벨로 쓸 수 없다. 현재는 색 마스크로 센 천 픽셀 수가 대리 지표다 (stats 0절, 마스크 IoU 0.92는 README 서술).
- 학습 데이터의 크기 분산은 0이다 (학습과 InD의 천 크기가 하나뿐, 프레임 3 천 픽셀 13,980-14,330 @512², stats). 따라서 크기 probe는 OoD 3점(0.26x, 1x, 2.5x)에서만 가능하다 (아래 7절).

## 3. 분석가별 핵심 주장

### 3.1 prior (선행연구/이론)
- 크기 정보는 patch 위치/개수로 인코딩되므로 인코더 단계 손실 가능성은 낮고, predictor가 학습분포(작은 천)의 prior로 끌려가는 쪽이 더 의심된다 (추정, 1.2).
- 가장 유력하다고 본 원인(추정): 학습에서 크기 분산이 0이면 predictor가 크기를 보존할 압력이 없다 (3절 3항).
- 그 밖의 원인 후보(모두 추정): MSE 회귀의 학습분포 평균으로의 수축, teacher forcing 오차 누적, 행동 표현(공유 3-D 대 팔별 8-D)의 영향 (ACWM-Phys에서 OoD MSE 23.8 → 6.9, 요약 페이지 기준).
- probing 설계: 인코더 출력과 predictor 출력(rollout 후 t+k)을 분리 probe, linear/MLP/attentive 병용, 층별 curve, 대조군(random-init predictor, 풀링 baseline, 픽셀 면적 같은 handcrafted feature).
- 해결 방향 후보: 크기/extent 보조 supervision(G-JEPA), 조건화(SG-JEPA), 데이터 다양성 (H3). 
- 연구 공백: DINO-WM류 동결 인코더 world model에서 연속 물리 파라미터를 인코더/predictor 별로 probing한 연구를 찾지 못했다 ("없다"의 단정은 불가, 검색 한계).
- 확신도: 중간-낮음.

### 3.2 stats (통계/데이터)
- README 수치로부터: 디코더가 실제 프레임 feature로는 큰 천을 그린다 (큰 천 decoder-on-real 4,352, real 4,475, model 1,836). 크기 정보가 인코더 feature에 존재한다는 간접 증거이며 probe로 확정한 것은 아니다 (0절).
- 학습량을 8배로 늘려도 큰 천 1,836 → 1,923으로 변화가 없고 검증 손실은 계속 하락 (0.166 → 0.150). 
- OoD 에피소드 약 2.7%(50개)를 학습에 넣으면 큰 천 마지막 프레임 1,645 → 3,524(real 4,723), rollout MSE -49% [-56, -42], 25/25 에피소드 개선. 작은 천은 오차 -39%이나 크기는 1,189 → 1,223(real 365)으로 거의 안 풀렸다. InD 오차는 +25% [+16, +34]. 
- 1-step에서 model/copy 오차비가 InD 0.31 → OoD 0.74-0.77로 커지고, 큰 천은 rollout 안에서 InD 크기로 수렴한다 (README 서술, 3절). 12 스텝 후 retention -0.09 (6절).
- 해석(추정): 크기 분산 0이므로 predictor에 크기를 쓸 신호가 없다 (underspecification). 인코더에 크기가 있어도 predictor가 전달할 이유가 없다.
- probing 설계: 인코더(A), 1-step(B), k-step rollout(C), 통제(D)의 4지점. 선형(ridge)과 소형 MLP 병용, control task(에피소드 단위 라벨 permute, 랜덤 ViT/raw pixel, 상수 평균 R²=0), 에피소드 단위 분할(GroupKFold, leave-one-group-out), cluster bootstrap, 기울기 지표(1이면 보존, 0이면 학습 크기로 붕괴), probe 시드 5 이상, 모델 학습 시드 3 이상(현재 1).
- 검정력(stats 자체 계산, Fisher z 근사): 검출 가능한 최소 |r|이 n=50에서 0.387, n=100에서 0.277, n=400에서 0.140. n은 프레임이 아니라 에피소드 수다.
- 해결 문헌 정리: 데이터 다양성(Tobin 2017, Peng 2018), 파라미터/맥락 조건(UP-OSI, CaDM, RMA), 적응(Nagabandi 2019). 모두 기억 기반이며 저상태 RL 위주라 픽셀/patch-token world model에서 형상 파라미터에 같은 효과가 있다는 직접 근거는 없다 (추정).
- 확신도: 중간-낮음. 문헌은 기억 기반이고 현상 수치는 단일 시드 README에 의존.

### 3.3 skeptic (회의론)
- 가장 먼저 배제할 원인은 디코더(A)다. rollout 영상은 VQ-VAE 디코더가 만들며 이 디코더는 학습 분포(작은 천)만 보고 학습됐다 (main 브랜치 `visual_world_model.py:207-218`, 확인). "큰 천이 latent에 남아 있는데 디코더가 작게 그린다"를 배제하기 전에는 인코더/predictor 문제라고 말할 수 없다. 같은 코드베이스의 Rope/Granular에서 decoder 병목이 이미 관찰되었다 (thread0 README 인용).
- 가장 단순한 설명(B)은 학습에서 크기가 안 변해 predictor가 복사할 이유가 없다는 것이다. 이는 "물리 인식 부족"이 아니라 shortcut/case-based generalization이며 후자 해석은 과장이다 (Kang et al. ICML 2025, Geirhos 2020, 검색확인; Kang은 픽셀 비디오 생성 모델 결과라 DINO-WM 적용은 추정).
- 그 밖의 대안 원인(모두 추정 포함): C 절대 위치 임베딩 기반 patch 위치 prior (`vit.py:110, 117`, 코드 확인, 효과는 추정), D 자기회귀 오차 누적 (teacher forcing, `train.yaml:58-59`), E 전처리/해상도/시야 잘림/시뮬레이터 자체의 수축, F 정규화 통계(std에 1e-6을 더함, `deformable_env_dset.py:72-73`)와 조건화 입력의 OOD. F는 acwm-phys 코드를 봐야 확정되며 skeptic은 "잠재적 버그"로만 표기했다 (추정).
- probing 함정: 정보가 있어도 사용하지 않을 수 있다 (Ravichander 2021, Elazar 2021, Belinkov 2022, Vafa 2025, 모두 검색확인). frozen DINOv2에 크기 정보가 있을 가능성은 매우 높다 (추정). 따라서 probing만으로 "인코더는 문제없고 predictor가 문제"라는 결론은 낼 수 없고 개입 실험(입력 latent의 크기 방향 조작 후 예측 크기가 따라 바뀌는지)이 필요하다. 선례: Othello-GPT 개입 (Li 2023, Nanda 2023).
- 이중 외삽: 학습 크기 분산이 0이면 학습 데이터로 크기 probe를 학습할 수 없고, 다른 크기 범위로 학습한 probe를 쓰면 probe 자체가 외삽한다 (Xu 2021). 예측 latent에서 probe 값이 작게 나와도 "표현이 작아졌는지 probe 외삽 실패인지" 구분되지 않는다.
- probe 용량/선택성 (Hewitt & Liang 2019): 고용량 probe의 높은 정확도는 probe가 배운 것일 수 있다.
- 질량/마찰 probing의 함정: 단일 프레임에서 관측 불가, 그리고 main의 cloth_scene에서 stiffness와 dynamicFriction이 같은 난수 sf 하나의 함수라 구분 불가 (`scenes.py:203-207`, 확인). acwm-phys가 이를 풀었는지는 미확인.
- 과장 경고: "DINO-WM은 물리 파라미터 일반화에 실패한다"(크기 하나, 데이터셋 하나, 디코더를 거친 관찰 하나), "인코더에는 있는데 predictor가 버린다", "physics-aware 모듈로 해결"(같은 데이터 다양화 baseline 없이는 구분 불가), "붕괴"라는 용어가 representation collapse와 혼동됨.
- 최대 3개의 선행 실험: (1) 단계 분리 진단(Real / encode→decode / teacher-forced 1-step / open-loop k-step), (2) 크기 probe + 개입, (3) 데이터 다양화 대조군(GPU 비용이 커 사람 승인 필요).
- 확신도: 중간.

## 4. 합의점

1. 학습 데이터에서 크기가 변하지 않아 predictor가 크기를 유지할 신호를 받지 못했다는 설명이 유력 후보라는 점: prior(3절 3항, "가장 유력", 추정), stats(해석 메모, underspecification, 추정), skeptic(B, "가장 단순한 설명", 추정). 세 명 모두 추정으로 표시했다. 확정된 원인이 아니다.
2. 크기는 관측 가능한 외형 속성이라 동결 DINOv2 인코더에 정보가 있을 가능성이 높다: prior(추정), stats(간접 증거 + 추정), skeptic(추정). 어느 쪽도 probe로 확인하지 않았다.
3. 크기 probe는 인코더 / predictor 출력(및 rollout)을 나누어야 한다: prior 4절, stats 3절(A-D), skeptic 6절 실험 1-2.
4. 선형 하나로는 부족하다. 용량을 올리면 probe가 외우는 문제가 생기므로 control task/selectivity가 필요하다: prior(2.1, 2.3), stats(1절), skeptic(2.2).
5. 에피소드 단위 분할이 필수다: stats(2절), skeptic(2.5). prior는 분할 단위를 명시하지 않았고 OoD 포함 학습 probe와 InD만 학습 probe의 비교를 제안.
6. 정보 존재(probing)와 사용 여부는 다르다. stats(1절 7항)와 skeptic(2.1)은 개입을 명시 요구한다. prior도 "정보가 (a)에 있고 (b)에서 사라지는지"로 구분하되, 개입을 명시 요구한 것은 아니다.
7. 학습 데이터 다양화가 가장 단순한 해결 후보이자 대조군이다: prior(H3), stats(6절, README 혼합 실험이 같은 방향), skeptic(3절 표).
8. 해결 방향 문헌(RMA, CaDM, UP-OSI 등)의 직접 근거는 이미지 latent world model에서의 천 크기에는 없다: stats(추정), skeptic(단서), prior(기억 기반, 미검증).
9. 어떤 문헌도 DINO-WM의 크기 OoD 수축을 직접 분석하지 않았다: prior(확신도 절), stats(5절 마지막), skeptic(위 DINO-WM 서술).

## 5. 의견이 갈리는 점

### 5.1 수축의 유력 원인 위치 (핵심 차이)
- 디코더 (skeptic): 디코더 원인(VQ-VAE가 작은 천으로만 학습)을 가장 먼저 배제해야 하며, 배제 전에는 인코더/predictor 결론을 보류한다. 순서가 곧 주장이다.
- 인코더 보유 + predictor 수축 (prior, stats): prior는 "인코더 손실 가능성 낮음, predictor 쪽이 더 의심"(추정). stats는 README의 decoder-on-real 수치(큰 천 4,352 / real 4,475)를 디코더 원인에 불리한 간접 증거로 쓴다.
- 이 차이는 다음과 같이 읽힌다(reporter 정리, 어느 분석가의 문장도 아님). skeptic이 요구한 진단 (b) encode→decode는 stats가 옮긴 README 수치와 같은 종류의 측정으로 보이는데, skeptic은 해당 README를 읽지 못했다. 다만 이 수치가 어떤 체크포인트/디코더로 얻어졌는지, 작은 천 쪽(real 404 대 decoder-on-real 388)도 포함해 같은 조건인지, 그리고 stats 자신이 단일 시드 README 서술이라고 한 한계는 그대로 남는다. 따라서 "디코더 원인이 배제되었다"고 이 문서에서 결론 내리지 않는다. 사람이 판단한다.
- prior는 디코더를 원인 후보로 따로 다루지 않았다(위 prior 파일에 디코더 항목 없음).

### 5.2 학습 설정 서술 차이
- stats(upstream README): num_hist 1, 15 epoch, frameskip 5, 시드 1개, DINOv2 ViT-S/14.
- skeptic(main 브랜치 conf/train.yaml): num_hist 3, num_pred 1, img_size 224.
- 이 둘은 서로 다른 브랜치의 설정이므로 충돌인지 단순한 브랜치 차이인지 판정하지 못한다 (미확인). rollout 오차 누적(D)의 해석은 num_hist에 따라 달라질 수 있다 (stats 6절 2항은 num_hist 1이라 속도/역사로 파라미터를 추정할 수 없다고 봄).

### 5.3 수축 시점의 해석 (D 대 B/C)
- skeptic: 수축이 1-step에서 바로 나타나면 B/C, 몇 step 뒤 급변이면 D가 유력하다. "갑자기"의 의미가 모호하다.
- stats: README에서 1-step model/copy 오차비가 InD 0.31 → OoD 0.74-0.77로 이미 커지고, rollout에서 큰 천이 InD 크기로 수렴한다고 옮겼다. 즉 1-step에서도 증상이 있다는 서술이지만 크기가 1-step에서 수축하는지는 stats 파일에 명시되지 않았다. 해석은 열려 있다.
- prior: 오차 누적(SG-JEPA 근거)도 후보에 두었으나 DINO-WM 학습 방식은 구현 확인이 필요하다고 한다.

### 5.4 probe 학습 프로토콜
- stats: InD + OoD(0.26x, 2.5x)를 합쳐 에피소드 단위, 군집 층화 5-fold 또는 leave-one-group-out으로 probe를 학습/평가하고, A에서 학습한 probe를 예측 feature에 적용하는 방식과 B/C에서 새로 학습하는 방식을 모두 쓰라고 한다. 분포 이동으로 probe가 틀릴 수 있음을 인정한다.
- skeptic: probe는 크기 전 범위를 덮는 데이터로 frozen DINOv2 GT 인코딩에만 학습하고, 그 probe를 예측 latent에 적용해야 probe 외삽과 predictor OOD가 분리된다고 한다. 크기가 다양한 probing 데이터는 predictor 입장에서 OOD라는 점도 지적.
- prior: OoD 크기를 일부 포함해 학습한 probe(상한)와 학습분포만으로 학습한 probe를 비교.
- 세 방식은 완전히 같지 않다. 어느 쪽을 쓸지는 정해지지 않았다.

### 5.5 해결 방향의 적합성
- prior: 환경 latent 추론(T2-Z, RMA류), 보조 supervision(G-JEPA), 조건화(SG-JEPA)를 후보로 둔다.
- skeptic: 크기는 숨은 파라미터가 아니라 보이는 속성이므로 context 추론을 처방하는 것은 문제와 어긋날 수 있다(추정). SG-JEPA는 g를 명시 입력으로 주며 픽셀에서 추론한 것이 아니다. 데이터 다양화 대조군을 넘지 못하면 "physics-aware 구조"의 필요성 주장이 약해진다.
- stats: 보조 손실로 파라미터 예측을 붙일 수 있다고 하되 직접 근거는 없다고 인정(추정).

### 5.6 지표와 표본 규모 권고
- stats: 시드 probe 5 이상, 모델 학습 시드 3 이상, 에피소드 200-400개 연속 샘플링 OoD sweep이 정보량이 가장 크다(추정). R² 대신 MSE/RMSE, 상대오차, Spearman, 기울기 병기.
- skeptic: 시드와 임계값에 숫자를 두지 않고 여러 에피소드/시드의 통계와 빈도(몇 개 중 몇 개에서 축소)를 요구.
- prior: 구체 임계값은 팀 합의로 둔다.

### 5.7 확신도 자기평가 차이
- prior 중간-낮음, stats 중간-낮음, skeptic 중간. 인용 근거의 확인 수준이 서로 다르다는 점(1절)을 고려해 읽을 것.

## 6. 한 명만 지적한 점
- prior만: acwm-phys가 DINO-WM이 아니라 ACWM-DiT를 평가했다는 점(5절 3항), 행동 표현이 OoD에 미치는 영향(3 → 8차원, MSE 23.8 → 6.9, 요약 페이지), PhyIP(Observer Effect)의 비침습 probing 시사점, POKEWORLD의 identifiability 결과(요약 페이지 기준 수치 미검증), Video FM probing의 attentive/MLP probe 필요성(2차 요약).
- stats만: README의 정량 현상 수치 전체와 OoD 2.7% 혼합 실험, 검정력 표, cluster bootstrap/GroupKFold, metadata cloth_dimx/y와 보이는 크기의 낮은 상관 r ≈ 0.02, 크기 연속 샘플링 sweep 제안, 마스크 기반 지표의 디코더 천장(decoder-on-real) 동반 보고 권고.
- skeptic만: 디코더 우선 배제, 절대 위치 임베딩 기반 patch 위치 prior(C), 정규화 std 1e-6 문제(F, 잠재적 버그, 추정), 전처리/시야 잘림/시뮬레이터 자체 수축(E), 질량-마찰 결합(sf 하나), Vafa 2025, Kang 2025, 과장 경고 목록, FleX bit 재현 불가(thread0 README), 골라 뽑은 영상 금지, 체크포인트 epoch 고정.

## 7. 연구 공백
1. DINO-WM류 동결 foundation 인코더 world model에서 연속 물리 파라미터(크기)를 인코더/predictor 별로 probing한 연구: 세 분석가 모두 직접 선행연구를 찾지 못했다 (prior 5절 1항, stats 5절, skeptic). "없다"의 단정은 불가.
2. 관측 가능한 파라미터가 OOD일 때 인코딩 손실과 동역학 수축을 정량 분해한 연구 (prior 5절 2항, 추정).
3. acwm-phys는 DINO-WM류 latent 모델을 평가하지 않았다 (prior 5절 3항).
4. 크기 probe의 데이터 한계: 학습 데이터에 크기 분산이 0이라 크기 probe는 OoD 3점(0.26x, 1x, 2.5x)에서만 가능하다 (stats 5절 5항, skeptic 2.4). 3점뿐이라 전체 R²는 군집 간 분산이 지배하고 단조 여부 이상의 곡선을 볼 수 없다. 군집 내 효과 검출은 군집당 50 에피소드로 검정력이 제한된다(n=50에서 최소 |r| 0.387, stats 자체 계산). 중간값(예: 0.5x, 1.5x)이나 연속 샘플링 OoD sweep은 새 데이터 생성이 필요하다 (stats, 추정).
5. 질량/마찰 같은 숨은 속성: 단일 프레임에서 관측 불가이며 acwm-phys에 해당 파라미터 라벨이 있는지 미확인이다 (prior 4절 마지막, stats 0절 "물리 파라미터 라벨 쓰는 코드 없음").

## 8. 미검증 항목
- 문헌 서지 전반: stats 문헌 목록은 기억 기반이며 인용 전 검증 필요 (stats 8절). prior의 RMA, MeshGraphNets, DPI-Net, CALIPER(제목/ID 미확인)도 미검증 (prior 3절 5-6항, 2.2). skeptic의 [검색확인] 문헌도 본문 수치는 대조하지 않음.
- 2026 preprint 수치: ACWM-Phys Table 1, POKEWORLD R², SG-JEPA 오차 감소율, T2-Z 등은 2차 요약 (prior).
- SG-JEPA의 encoder/predictor 서술 충돌: prior가 정합성 정독 필요라고 했고 skeptic도 같은 충돌을 지적 (skeptic 1.D).
- acwm-phys 데이터셋 카드가 우리 팀 버전과 같은지, 천 크기 외 물성 라벨 유무 (prior).
- 학습 크기 분포, 조건화 입력 유무, 정규화 통계가 acwm-phys에서 어땠는지 (skeptic은 읽지 못함. stats는 일부 README만 읽음).
- 현상 자체의 수축이 1-step 시점인지 k-step 시점인지 (skeptic), 큰 천이 시야/crop에서 온전히 보이는지, 시뮬레이터 Real 프레임에서 수축이 없는지 (skeptic E).
- stats가 옮긴 README 수치(Drive의 `<env>_h1f5_quant.json` 원자료)는 재계산되지 않았다 (stats 8절).
- 한 가지 수치 불일치: 학습 규모 1800/200 (stats, README) 대 1,987 (prior, 논문 Table 8).
- 위에서 "추정"이라고 표시된 모든 원인 후보는 검증된 사실이 아니다. 이 문서는 어느 원인도 확정하지 않는다.
