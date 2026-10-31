# 가설 목록

| ID | 가설 | 성공 기준 | 필요한 실험 | 상태 |
|----|------|-----------|-------------|------|
| H1 |      |           |             | 제안 |

---

## 가설 후보 (미확정, 사람 승인 대기)

- 작성: reporter, 2026-10-30. 출처 종합: notes/lit_review.md (분석가 prior/stats/skeptic).
- 아래 3개는 모두 미확정 후보이며 어느 것도 확정된 가설이 아니다. 가설 확정은 사람이 한다.
- 성공 기준의 임계값, 시드 수는 분석가가 낸 예시이며 사람이 확정한다. 근거 있는 값이 아니다.
- 공통 전제 (lit_review.md 5.1절, 세 분석가 일치): 속성을 "단일 프레임 관측 가능 / 시간 창 필요 / 관측 불가"로 분류하고 관측 가능한 속성(위치, 각도)부터 다룬다. 선택성 대조군, random-init 기준선, episode 단위 분할을 쓴다.
- H1의 encoder 단계 대 predictor 단계 구분에 대하여
  - 합의된 부분: "transformer layer"가 frozen DINOv2인지 학습되는 predictor인지 먼저 정해야 한다는 점은 세 분석가가 모두 지적했다.
  - 단일 의견인 부분: 구체적 구분 방식은 갈린다. prior는 H1a(encoder)/H1b(predictor) 2분, skeptic은 (a) DINOv2 내부 / (b) predictor 중간 layer / (c) predictor 출력 3분이며 (c) 고정을 권고(단일 의견), stats는 R0/R_l/R_pred 지점 정의로 predictor 쪽만 운용 정의(DINOv2 내부 layer는 포함하지 않음). 후보 1과 3은 이 구분을 반영해 나눈 것이며 이 분할 자체가 분석가 간 합의는 아니다.

### 후보 A. predictor 출력 및 layer별 attribute 보존 (H1, predictor 단계)

- 가설 문장 (미확정): DINO-WM의 학습된 predictor를 통과한 표현(layer별 hidden state, teacher-forced 1-step 출력, open-loop k-step 출력)에서, 사전에 정한 관측 가능 속성(PushT agent xy, block xy, block angle)에 대한 probe 오차의 증가량(excess error)이 무작위 부분공간 또는 외형 속성(색 등)에서 기대되는 수준보다 유의하게 크다.
- 검증 방법 개요: GT 인코딩 z_{t+k}, teacher-forced ẑ_{t+1}, open-loop ẑ_{t+k}, predictor 각 layer에서 같은 probe(linear + MLP)를 적용한다. SG-JEPA Eq.4식 excess error와 R0 대비 ΔR²를 쓴다. 1-step과 rollout을 분리한다. 기준선은 random-init predictor, label shuffle, 시간 index probe, GT z_{t+1} 상한이다. 분할은 episode 단위, CI는 episode 단위 cluster bootstrap. 먼저 LeWM Table 1 DINO-WM 행의 probe 조건을 확인하는 기준선 점검을 둔다(stats, skeptic 제안).
- 성공 기준 후보 (임계값은 예시이며 사람이 확정): (1) 선택성 게이트: 실제 label R² − control R² ≥ 0.1 (stats 예시). (2) ΔR²(R_pred 또는 R_6 대 R0)의 95% CI 상한 < −0.10 (stats 예시 ε1). 또는 probe r이 입력 대비 0.05 이상 감소 (prior 예시). (3) 비교 수가 많으므로 사전 정의한 가족에서 Holm 등 보정 (stats). (4) 학습 시드 최소 5 (stats) 또는 최소 3 (skeptic), 의견이 갈리므로 사람이 정한다.
- 근거 출처: LeWM Table 1/3/4 (prior, stats, skeptic), SG-JEPA Eq.4·Fig 4a·Fig 5c (prior, stats, skeptic), G-JEPA, Hewitt & Liang (기억 기반/[미확인]). 직접 선행 근거는 없음(DINO-WM predictor 내부 layer probing은 문헌에서 미발견, prior·stats).
- 반증 조건: (i) teacher-forced k=1에서 excess error의 CI가 0을 포함한다. (ii) k가 커질 때 증가율이 무작위 부분공간과 다르지 않다. (iii) 마지막 layer 성능이 입력 layer 이상으로 회복된다 (이상 skeptic 제안). (iv) ΔR²의 90% CI가 [−ε, +ε] 안에 있어 TOST를 통과한다 (stats 제안). k=1에서는 없고 rollout에서만 커지면 H1이 아니라 오차 누적(skeptic A2)으로 해석한다. 주의: 중간 layer의 하락만으로는 희석이라 부르기 어렵다는 skeptic 의견이 있고, 단조 감소를 H1b의 증거로 보는 prior 의견과 갈린다.

### 후보 B. 보존되나 활용되지 않는가 (H2, 개입 기반)

- 가설 문장 (미확정): 사전에 고정한 속성 집합 S에 대해 predictor 입력 표현에 정보가 남아 있으나(보존 조건), S의 probe 방향을 제거해도 예측 오차와 planning 성능의 변화가 동일 차원의 무작위 방향 제거와 구분되지 않는다(비활용).
- 검증 방법 개요: 보존 조건은 후보 A 또는 TOST로 확인한다. 개입은 (1) 선형 소거(INLP/LEACE 계열)와 무작위 부분공간 소거 비교, 소거 후 MLP probe로 정보가 실제로 사라졌는지 확인(skeptic 조건), (2) probe 방향 steering/조작, (3) 시뮬레이터 반사실 쌍 또는 activation patching(환경이 허용하는지 [미확인], stats), (4) 같은 frozen encoder 위에 fresh predictor 학습(SG-JEPA Fig 5 설계), (5) 보조 supervision(G-JEPA식), (6) action ablation과 identity baseline 비교(prior만 제안). planning SR은 planner, 비용 함수, 데이터 coverage를 통제한 뒤 해석한다(skeptic).
- 성공 기준 후보 (임계값은 예시이며 사람이 확정): 비활용 조건 중 하나 이상. (i) y-부분공간 제거의 예측오차 증가가 무작위 제거와 차이 없음(등가성 ±ε2 또는 경험적 p > 0.05). (ii) patching 정규화 indirect effect < 0.1. (iii) steering 기울기 < 0.2. (iv) fresh predictor 또는 보조 supervision 후 teacher-forced 예측 NMSE가 상대 10~20% 이상 개선되고 CI가 0을 제외 (이상 stats 예시). 정보가 제거되지 않았음이 확인되면 결론을 내리지 않음(skeptic).
- 근거 출처: Amnesic probing (Elazar et al. 2021; skeptic [확인], prior [검색 요약만]), INLP/LEACE (stats 기억 기반), G-JEPA Table 3·5 (prior, stats, skeptic), LeWM Table 3 캡션 (skeptic), SG-JEPA Fig 5 (stats). 모두 NLP 맥락이거나 간접 근거이며 DINO-WM에 직접 적용한 선례는 없다(prior).
- 반증 조건: 제거 개입 후 S 오차가 무작위 제거보다 유의하게 커진다. 조작 개입 후 예측의 S 값이 물리적으로 일관되게 따라 움직인다. patching indirect effect ≥ 0.5이면서 무작위 대비 유의 (stats 예시). 보존 조건 자체가 성립하지 않으면(R0에서도 복원 낮음) 후보 B는 전제 불성립이며 H1 변형도 아님(stats). 둘 다 틀린 경우(보존 O, 활용 O)의 병목은 planner, 비용 함수, 데이터 coverage일 수 있다(skeptic).

### 후보 C. encoder 단계(DINOv2)의 layer별 보존 및 probe 기준선 감사 (H1', encoder 단계)

- 가설 문장 (미확정): frozen DINOv2의 마지막 layer feature는 중간 layer feature보다 사전 고정한 속성 집합 S에 대한 probe 성능이 낮다. 이와 별도로 LeWM Table 1의 DINO-WM 행은 episode 단위 분할, 차원 통일, 선택성 대조군을 적용해도 재현된다.
- 검증 방법 개요: 공개 DINO-WM/DINOv2 체크포인트로 PushT에서 DINOv2 중간 layer에서 최종 layer까지 patch 토큰을 probe한다(Zhan et al. 프로토콜과 유사, prior). 조건은 episode 단위 분할, PCA 차원 통일(skeptic 예: k=192), ridge CV, label shuffle, random-init encoder control, pooling 방식 병기. 속성은 분류(단일 프레임 / 창 / 관측 불가)해 시간 속성은 창 입력으로 따로 다룬다. 지표는 r 단독이 아니라 R²와 MSE 병기, median 병행(stats).
- 성공 기준 후보 (임계값은 예시이며 사람이 확정): (1) 재현 기준: LeWM Table 1 DINO-WM 행(예: Block Angle Linear r 0.979, MLP r 0.995; 전사값, 원문 대조 안 함)과 사전 정한 허용 오차 내에서 일치. 허용 오차는 사람이 정한다. (2) layer 곡선: 마지막 layer의 S_l이 중간 layer 최대값보다 ε(예: 0.10, stats ε1 차용) 이상 낮고 CI가 0을 제외. (3) 학습 시드 개념은 encoder가 frozen이므로 해당 없음, 대신 probe 시드 5 이상과 episode bootstrap (stats).
- 근거 출처: LeWM Table 1 (prior, stats, skeptic), Zhan et al. 2024 (prior [검색 요약만]), Skean et al. 2025 (prior [검색 요약만], 언어모델 결과, 비전 수치 미확인), skeptic의 H1' 정의. DINO-WM 논문 Table 2는 patch가 CLS보다 우수함을 보이나 layer 선택은 다루지 않음(prior, skeptic).
- 반증 조건: 마지막 layer의 probe 성능이 다른 layer와 같거나 더 높다(skeptic). 또는 TOST로 등가. 재현이 안 되면 후속 비교를 보류하고 LeWM의 probe 조건 차이(pooling, 데이터 등)를 점검한다(prior). 회전 계열(Cube의 yaw, quaternion)이 모든 모델에서 낮은 점은 소실이 아니라 encoder 단계 부재일 수 있어(stats) 이 결과만으로 가설을 지지하지 않는다. 이 후보는 DINO-WM 고유의 결함이 아니라 DINOv2 layer 선택에 대한 가설이라는 skeptic의 단서가 있다(skeptic 단일 의견).

---

## 가설 후보: acwm-phys Cloth Move 천 크기 수축 (미확정, 사람 승인 대기)

- 작성: reporter, 2026-10-31. 출처 종합: notes/lit_review.md 두 번째 종합 절 (notes/research/prior.md, stats.md, skeptic.md).
- 아래 D/E/F는 모두 "후보 (미확정, 사람 승인 필요)"이다. 어느 것도 확정되지 않았다. 앞의 후보 A/B/C(PushT)와는 별개 주제다.
- 서로 다른 원인 층위를 다룬다. D 디코더, E 인코더/프리딕터 정보, F 학습 데이터 다양성(개입/해결). 분석가 의견이 갈리는 점(디코더 우선 배제 대 인코더 보유 + predictor 수축)은 D와 E를 모두 후보로 두어 그대로 남겼다. 어느 쪽이 맞는지는 이 문서가 판단하지 않는다.
- 수치 기준(임계값, 시드 수)은 모두 초안 예시이며 근거 수치가 아니다. 사람이 확정한다.
- 공통 한계: 학습 데이터의 크기 분산은 0이라 크기 probe는 OoD 3점(0.26x, 1x, 2.5x)에서만 가능하다 (stats, skeptic). 현상 수치는 단일 시드 README 서술이다 (stats). 로컬에 데이터/체크포인트가 없다 (stats).
- 공통 전제: 평가는 에피소드 단위 분할과 cluster bootstrap을 쓴다 (stats, skeptic). 크기 지표는 마스크 천 픽셀 수와 latent 쪽 지표를 병기한다. cloth_dimx/y 라벨은 보이는 크기와 r ≈ 0.02라 쓰지 않는다 (stats).

### 후보 D. 디코더 층위: 수축은 시각화(디코더) 때문인가

- 상태: 후보 (미확정, 사람 승인 필요)
- 가설 문장 (미확정): 큰 천(2.5x)과 작은 천(0.26x)의 GT 프레임을 predictor 없이 encode→decode 했을 때 천 면적이 real과 가깝게 유지되고(디코더는 수축의 원인이 아니다), 수축은 predictor 출력 이후 단계에서 처음 나타난다. 반대로 encode→decode에서 이미 학습 크기로 수축하면 디코더(작은 천으로만 학습된 VQ-VAE)가 원인이다.
- 검증 방법: skeptic의 단계 분리 진단. (a) 시뮬레이터 Real, (b) encode→decode(predictor 없음), (c) teacher-forced 1-step 예측의 디코딩, (d) open-loop k-step. 디코더를 거치지 않는 latent 지표도 병행한다(GT 큰 천 latent, GT 작은 천 latent, 예측 latent 사이의 patch별 상대 거리). 에피소드 단위 빈도(몇 개 중 몇 개에서 수축)와 체크포인트 epoch를 고정해 보고한다. 큰 천이 시야/crop에서 온전히 보이는지(Real 프레임)도 같이 확인한다 (skeptic E).
- 성공/반증 기준 초안 (임계값은 예시, 사람 확정):
  - 디코더 비원인 지지: decoder-on-real / real 면적비가 두 크기 모두 사전 정한 허용값(예: 0.9 이상) 이상이고 model 출력은 그보다 유의하게 작다. 참고로 stats가 옮긴 README 수치는 큰 천 4,352 / 4,475, 작은 천 388 / 404이다(재계산 아님, 비율은 reporter 나눗셈).
  - 디코더 원인 지지(반증): decoder-on-real 면적도 학습 크기 쪽으로 수축한다. 이 경우 인코더/predictor 원인 주장(후보 E)을 보류한다 (skeptic).
  - 원인 층위가 갈리지 않는 모호한 경우도 그대로 보고한다.
- 필요 자원: 기존 체크포인트와 데이터가 있으면 추론만 필요하다(GPU는 소규모, 추정). 현재 로컬에는 데이터/체크포인트가 없다 (stats). 새 학습은 필요 없음.
- 근거: skeptic(1.A, 6절 실험 1), stats(0절 README 수치, 4절 decoder-on-real 천장 동반 보고), prior는 디코더를 따로 다루지 않았다. 의견 차이는 lit_review 5.1절.

### 후보 E. 인코더/프리딕터 정보 층위: 크기 정보는 어디에 있고 어디서 사라지는가

- 상태: 후보 (미확정, 사람 승인 필요)
- 가설 문장 (미확정): 동결 DINOv2 GT 인코딩에서는 천 크기가 probe로 복원되지만(보존), predictor의 teacher-forced 1-step 또는 open-loop k-step 출력에서는 같은 probe가 학습 크기 쪽으로 편향된다(예측 크기 대 실제 크기 기울기 < 1). 이 정보가 predictor에서 실제로 쓰이는지는 probing이 아니라 개입으로 확인한다.
- 검증 방법: stats의 지점 A(인코더), B(1-step), C(k-step), D(통제). linear(ridge) + MLP + 풀링 방식 병기, 층별 curve(predictor 블록 깊이) (prior). 통제: 에피소드 단위 label permute, 랜덤 ViT/raw pixel, 상수 평균, 픽셀 면적 handcrafted baseline. probe 학습 프로토콜은 의견이 갈리므로 사람이 선택한다: (i) InD+OoD 군집 층화 CV 또는 leave-one-group-out (stats), (ii) 전 범위 GT 인코딩에만 학습해 예측 latent에 적용 (skeptic), (iii) OoD 포함 probe(상한)와 학습분포 probe 비교 (prior). 개입: 입력 latent의 크기 방향을 투영 제거/치환했을 때 예측 latent의 크기가 따라 바뀌는지 (skeptic, stats).
- 성공/반증 기준 초안 (임계값은 예시, 사람 확정):
  - 인코더 보유: GT 인코딩 probe의 선택성(진짜 라벨 − control)이 사전 정한 값 이상, 3-way 분류 정확도와 군집 내 지표를 병기. R² 단독 금지 (군집 간 분산이 지배, stats).
  - predictor 수축 지지: C 또는 B에서 기울기 < 1이고 cluster bootstrap CI가 1을 제외, 학습 평균 쪽 bias.
  - 반증: 인코더 probe가 control과 구분되지 않거나(정보 없음), 기울기 CI가 1을 포함하고 k에 따른 감소도 없다. k=1에서 정상이고 k 증가로만 감소하면 크기 정보 소실이 아니라 오차 누적(D, skeptic)으로 해석한다.
  - 사용 여부: 개입 없이는 "predictor가 버린다"를 결론으로 쓰지 않는다 (skeptic, stats).
  - 데이터 한계: 크기 값이 3점뿐이라 단조 여부 이상은 못 본다. 군집당 50 에피소드에서 검출 가능한 최소 |r|은 0.387 (stats 계산). 곡선을 원하면 중간 크기 또는 연속 샘플링 OoD 에피소드를 새로 생성해야 한다(추정).
- 필요 자원: 체크포인트 추론 + probe 학습은 소형이라 GPU 부담이 크지 않을 것으로 추정(미확인). OoD 에피소드 추가 생성 시 시뮬레이터 비용 발생(승인 필요). probe 시드 5 이상 (stats). 모델 학습 시드 3 이상은 재학습이 필요해 후보 F에 해당 (stats).
- 근거: prior(1.2, 3절 3항, 4절), stats(0절, 1절, 3절, 5절), skeptic(2.1-2.5, 6절 실험 2). 선행 직접 근거 없음(세 분석가). 유력 원인이라는 판단은 prior/stats의 추정이며 skeptic은 디코더 배제를 선행 조건으로 둔다.

### 후보 F. 학습 데이터 다양성 층위: 크기 분산을 넣으면 수축이 줄어드는가 (개입/해결)

- 상태: 후보 (미확정, 사람 승인 필요)
- 가설 문장 (미확정): 학습 데이터에 천 크기 변동을 넣으면(분산 > 0) 같은 학습 예산의 baseline DINO-WM에서 보간 크기와 외삽 크기의 수축이 줄어든다. 수축의 주원인이 크기 분산 0(underspecification)이라면 이 개입만으로 줄고, 줄지 않으면 다른 층위(C 위치 prior, D 오차 누적, F 조건화 OOD 등)를 의심한다.
- 검증 방법: 학습 크기를 2-3단계 섞어 재학습(skeptic 실험 3). 보간(학습 범위 안의 held-out 크기)과 외삽 크기에서 수축 기울기/마지막 프레임 면적을 측정한다. 대조: 데이터량과 학습 epoch를 맞춘 단일 크기 baseline, 그리고 구조적 처방(보조 supervision, 조건화)은 이 데이터 다양화 대조군을 넘는지로만 평가한다 (skeptic 과장 경고 3). 참고 기존 수치: stats가 옮긴 README의 OoD 약 2.7%(50 에피소드) 혼합 실험에서 큰 천 마지막 프레임 1,645 → 3,524(real 4,723), 작은 천은 1,189 → 1,223(real 365)으로 거의 안 풀렸고 InD 오차는 +25% [+16, +34]였다 (단일 시드, 재계산 아님). 학습량 8배 증가로는 큰 천 1,836 → 1,923으로 변화가 없었다.
- 성공/반증 기준 초안 (임계값은 예시, 사람 확정):
  - 지지: 보간/외삽 크기 모두에서 기울기가 단일 크기 baseline보다 사전 정한 폭 이상 1에 가까워지고 에피소드 단위 CI가 0을 제외. InD 오차 악화가 허용 범위 안 (README에서 +25%가 관측되었으므로 비용을 같이 보고).
  - 반증: 개입 후에도 보간 크기에서 수축이 남거나 작은 천(0.26x 쪽)이 README처럼 거의 안 풀린다면 크기 분산 외의 원인이 큼. 이 경우도 그대로 보고한다.
  - 모듈 효과 주장은 같은 데이터 다양화 baseline과 비교하기 전에는 하지 않는다 (skeptic).
- 필요 자원: 재학습이 필요해 GPU와 비용이 크다. 사람 승인 필수 (CLAUDE.md). 시드 3 이상 권고 (stats, 현재 1). 새 크기를 포함한 데이터 생성이 필요할 수 있다. 실행 전 acwm-phys 브랜치의 commit hash, 학습 크기 분포, 정규화 통계를 config.yaml에 기록한다 (skeptic).
- 근거: stats(0절 README 혼합 실험, 6절 Tobin/Peng 등 기억 기반), skeptic(1.B, 3절 표, 6절 실험 3), prior(3절 3항, 6절 H3: 보조 supervision/conditioning도 후보). 의견 차이: prior는 conditioning/보조 supervision을, skeptic은 크기가 관측 가능한 속성이라 context 추론 처방이 어긋날 수 있다고 본다 (lit_review 5.5절).

## 확정 기록 (사람 승인)

- 2026-10-31: 사용자가 "가설 D와 E를 실험하고"라고 지시함에 따라 후보 D(디코더 층위)와 후보 E(인코더/프리딕터 정보 층위)를 실험 001_size_collapse_diagnosis 의 검증 대상으로 선택했다. (사용자 지시를 근거로 한 기록. 문구 확정이 필요하면 사용자가 수정한다.)
- 후보 F(크기 섞어 재학습)는 선택되지 않았다. 별도 승인 전까지 후보(미확정) 상태다.
- 성공/반증 기준의 임계값은 여전히 "사람 확정 필요"이다. 결과를 보기 전에 정하지 않으면 사후 기준 조정이 되므로, 본 진단 실행 전에 정해야 한다.
