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

## 후보 D/E 사전 등록 초안: cloth_sim 9종 버전 (미확정, 사람 확정 대기)

- 작성: 2026-11-01. 출처는 위 후보 D/E 절, experiments/001_size_collapse_diagnosis/README.md 와 config.yaml, 서버에서 확인된 results/maskcheck.json 일부. 새 사실을 추가하지 않았다.
- 왜 다시 쓰는가: 위 D/E 기준은 acwm_phys 3종(0.26x/1x/2.5x) 기준으로 쓰였다. 데이터가 cloth_sim 9종(명목 면적비 0.259 ~ 2.479, 크기마다 train 2,000 / test 100)으로 바뀌었고, 학습 분포는 ind_train 하나(baseline_single)다.
- 이 절의 모든 임계값은 "제안"이다. 근거 수치가 아니다. **본 진단(extract 이후)을 실행하기 전에 사람이 확정해야 하며, 결과를 본 뒤 바꾸면 사후 기준 조정이다.** 확정 칸은 `[ ]` 로 비워 두었다.

### 0. 전제 게이트 (재현 확인)

- 질문: ind_train 으로만 학습한 baseline 이 a026/a248 에서 "학습 크기로 수축"을 재현하는가.
- 이 게이트가 실패하면(수축 없음) D/E 진단의 전제가 성립하지 않으므로 D/E 결론을 내리지 않고 보고만 한다.
- 기준 제안: 마지막 프레임 천 면적 / real 면적이 a026 에서 [ ] 이상, a248 에서 [ ] 이하일 때 수축 재현. (제안 임계값은 사람이 정한다. 값은 비워 둔다. 근거 있는 값이 없다.)

### 1. 후보 D (디코더 층위), cloth_sim 용

- 측정 (README 기준): real / decode(encode(real)) / teacher-forced 1-step 디코딩 / open-loop k-step 디코딩의 천 마스크 면적, 9개 크기 각각. 면적 단위는 마스크 픽셀 수, 크기 곡선은 군집 평균 기울기와 bootstrap CI.
- 디코더 비원인 지지: decode(encode(real)) / real 면적비가 **9개 크기 모두** [제안 0.9] 이상이고, 모델 출력(1-step 또는 k-step)은 그보다 유의하게 작다.
- 디코더 원인 지지(반증): decode(encode(real)) 면적도 학습 크기 쪽으로 수축한다(예: a026 과 a248 중 하나라도 비 < [제안 0.9] 또는 > [제안 1.1]). 이 경우 E 의 인코더/predictor 원인 주장은 보류한다.
- 모호하면 그대로 보고한다.
- 단위/통계: 에피소드 단위, cluster bootstrap CI (config 의 bootstrap_seed 사용).

### 2. 후보 E (인코더/프리딕터 정보 층위), cloth_sim 용

- 측정: 동결 DINOv2 GT 인코딩 대 predictor 출력(1-step, k-step, 블록 깊이별)에서 크기 probe. 9-class 정확도, 인접(+-1) 클래스 정확도, 크기 곡선의 예측 대 실제 기울기. 통제: label permute, 랜덤 ViT, 픽셀 면적 baseline.
- probe 학습 프로토콜(사람이 선택): (i) 전 크기 층화 CV, (ii) 학습 크기(ind)만으로 probe 후 나머지 크기에 적용(보간/외삽, config 의 heldout_size 프로토콜), (iii) 두 가지를 병기. 선택: [ ]
- 인코더 보유: probe 선택성(실제 − control)이 [제안 0.1] 이상.
- predictor 수축 지지: 예측 크기 대 실제 크기 기울기 < 1 이고 cluster bootstrap CI 가 1 을 제외.
- 반증: 인코더 probe 가 control 과 구분되지 않거나, 기울기 CI 가 1 을 포함하고 k 증가에 따른 감소도 없다. k=1 에서 정상이고 k 증가로만 감소하면 크기 정보 소실이 아니라 오차 누적으로 해석한다.
- 사용 여부: 개입(intervene 단계: 크기 방향 제거 대 무작위 방향 제거) 없이 "predictor 가 정보를 버린다"를 결론으로 쓰지 않는다.

### 3. 시드와 반복

- 모델 학습 시드 [제안 3: 0,1,2, config 의 train.seeds]. probe 시드 [제안 5: 0~4, config 의 probe 시드]. 보고는 시드별 값과 시드 간 분포를 함께. 시드 3개는 작아서 평균 하나로 결론 내리지 않는다.

### 4. 사람이 채워야 할 것 (확정 전 체크리스트)

- [ v] 0번 게이트 임계값
- [ ]v D 의 면적비 허용값 (0.9 / 1.1 제안 유지 여부)
- [v ] E 의 선택성 임계값, probe 프로토콜 (i/ii/iii)
- [ v] 기울기 판정 방식 (CI 가 1 을 제외하면 충분한가)
- [ v] 시드 수 (모델 3, probe 5 유지 여부)
- [ v] "결과를 본 뒤 기준을 바꾸지 않는다" 동의

### 확정 기록 2: cloth_sim D/E 사전 등록 (2026-11-01, 사용자 승인)

- 사용자가 위 체크리스트와 본 학습(baseline_single, seed 0/1/2)을 "둘 다 허용"하고 push 를 지시했다. 값은 제안값을 그대로 채택한 것으로 기록한다(제안값은 근거 수치가 아니라 예시임). 사용자가 별도 값을 주지 않았다.
- 확정: 디코더 면적비 허용 하한 0.9(상한 1.1), 선택성 최소 0.1, 기울기 판정은 CI 가 1 을 제외하는지(여유 0), 모델 시드 0/1/2, probe 시드 0~4, 결과를 본 뒤 기준을 바꾸지 않는다.
- **미확정(비어 있음)**: 0번 전제 게이트의 수축 임계값(`shrink_ratio_tol`), probe 학습 프로토콜 선택(i/ii/iii). 제안값이 없고 사용자가 값을 주지 않았으므로 `shrink_ratio_tol` 은 null 로 두어 코드가 수치만 기록하고 판정하지 않는다. 결과를 본 뒤에 이 값을 정하면 사후 기준 조정이므로, 정하려면 본 진단 결과를 열기 전에 정한다. probe 프로토콜은 config 의 병기 설정을 그대로 쓴다(세 가지 모두 계산).
- 이 승인은 가설 확정이 아니라 사전 등록(측정과 판정 규칙 고정)이다. 논문에 들어갈 주장은 결과와 /review-experiment 후 따로 확정한다.

## 후보 G: 정보 흐름 분해 Q1/Q2/Q3, 학습 크기 조건 (a) 단일 / (b) 다중 (미확정, 사람 승인 대기)

- 작성: 메인 세션, 2026-11-02. 출처: 사용자가 제공한 설계 문서 "DINO-WM 천 크기 정보 흐름 실험 설계 (ACWM-Phys Cloth Move)" (Claude Doc, https://claude.ai/artifact/5N3XM9C9ijg8zWmGhL4r2Z, rev 11). 문서의 가설을 옮긴 것이며 새 사실을 추가하지 않았다. 아래 임계값은 모두 문서의 "제안값"이고 근거 수치가 아니다.
- 이 후보는 확정이 아니다. 가설 확정은 사람이 한다. 실험 폴더: experiments/002_single_size_sweep.
- 001(후보 D/E)과의 관계: Q1·Q2·Q3는 001 의 gt probe, h0~h6 층별 probe, 개입과 겹친다. 이 후보에서 새로 들어오는 것은 (1) 학습 크기 조건 비교 (a) 단일 대 (b) 다중 InD, (2) mask 면적/bbox baseline 과 random-init 인코더, 라벨 셔플 3종 통제, (3) attentive probe, (4) 초기 평평한 프레임 학습 대 후반 변형 프레임 평가, (5) INLP/LEACE 제거와 steering dose-response 이다. 001 리뷰의 한계(대조군 없음, 판정 단위 미사전등록, random-ViT/픽셀이 gt 만큼 높음)를 이 실험의 사전 등록에서 해소하는 것이 목적이다.

| 단계 | 질문 | 가설 (미확정) | 반증 조건 |
|---|---|---|---|
| Q1 인코딩 | 인코더 출력(DINOv2 패치 토큰)에 천 크기가 있는가 | H-a: 크기를 mask 면적 baseline 이상으로 복원할 수 있다 | held-out episode 에서 probe R² 가 mask 면적 baseline 또는 random-init 인코더와 구분되지 않음 |
| Q2 보존 | predictor 층들과 rollout 시점에서 그 정보가 유지되는가 | H-b: 마지막 블록과 rollout 후반까지 Q1 성능에 가깝게 유지된다 | 특정 층 또는 시점 이후 R² 가 Q1 대비 사전 정한 폭 이상 하락 (제안값 0.2, 파일럿에서 확정) |
| Q3 사용 | predictor 가 그 정보를 예측에 쓰는가 | H-c: 크기 방향을 이동 또는 제거하면 다음 latent 가 랜덤 방향보다 유의하게 더 변한다 | 크기 방향의 효과가 랜덤 방향 효과의 신뢰구간 안에 들어감 |

- 순서: Q1 이 음성이면 Q2 이후로 가지 않고 probe 형태(attentive, 비선형)부터 재검토. Q1·Q2 양성, Q3 음성이면 "정보는 있으나 쓰이지 않음".
- 성공 기준 (문서의 제안): H-a 는 미지 크기에서 probe R² 가 mask 면적 baseline 보다 높고 random-init 와 seed 편차 밖으로 다를 때 지지. H-c 는 크기 방향 효과가 랜덤 방향 효과의 95% 신뢰구간 밖이고 α 에 대해 단조일 때 지지. mask 면적 baseline 이 이미 R²≈1 이면 Q1 을 "면적 이상의 정보가 있는가"로 바꾸고 변형 후반 프레임 일반화를 주지표로 한다.
- 데이터와 역할 (문서의 제안): cloth-sim 9크기. 학습 a060, ind, a160 (크기당 300 episode 안팎), 보간 평가 a080, a125, 외삽 평가 a026, a040, a200, a248, 평가는 전 크기 test 100. 조건 (a) 는 ind 만, (b) 는 a060+ind+a160. 비용이 크면 (b) 만 먼저.
- 공통 설정 (문서): 타깃은 크기 스칼라(한 변 길이 또는 면적), probe 는 풀링 linear 와 attentive probe, 통제 3종(random-init 인코더, 라벨 셔플, mask 면적/bbox baseline), episode 단위 분할, 시간 축은 초기 평평한 프레임과 변형 큰 후반 프레임 분리 보고, 지표 R², MAE, Pearson r, seed 별 평균±표준편차(3 seed 이상), 동일 예산으로 비교.
- 자원: 12~16GB VRAM 단일 GPU. 이 세션에서 사용자가 12GB 서버를 무제한 사용 가능하다고 알렸고 baseline 도 같은 서버에서 재학습하기로 했다 (비교 조건 통일).
- 이 문서와 001 설정의 차이로 확인이 필요한 점: 문서는 데이터를 Hugging Face(`hanbin5/cloth-sim`)에서 받는다고 쓰지만 001 은 HF 를 사용자가 쓸 수 없다고 해서 Drive(rclone)로 받았다. 문서는 학습 시 IoU 0.68 의 재생성본임을 명시한다(001 과 동일).
- 사람 승인이 필요한 지점: 가설 확정, GPU 학습 실행(조건 (a)/(b) 학습), 논문 주장.

### 확정 기록 3: 후보 G 사전 등록 (2026-11-02, 사용자 선택)

- 사용자가 아래 값을 실행 전에 선택했다 (제안값은 근거 수치가 아니라 예시임). 결과를 본 뒤 바꾸지 않는다.
  - 데이터 출처: Google Drive(rclone). 학습 조건 순서: (b) 다중 a060+ind+a160 먼저, (a) ind 단일은 같은 12GB 서버에서 나중에 재학습. 학습 크기당 300 episode, 학습 시드 3개(0,1,2).
  - Q1(H-a): probe R² − mask 면적 baseline R² >= 0.1 이고 episode bootstrap CI 하한 > 0.
  - Q2(H-b): Q1 대비 R² 하락 >= 0.2 이면 보존 실패 (파일럿 없이 확정).
  - Q3(H-c): 무작위 방향 100개, 크기 방향 효과가 무작위 분포의 95 백분위 밖이고 alpha 에 대해 단조. 집계는 중앙값/순위.
  - 통계: episode bootstrap 95% CI 와 학습 seed 별 값 병기, 판정 패밀리를 미리 정해 Holm 보정.
- **아직 비어 있음/미확정**: 후보 G 자체의 가설 확정(사람 승인), torch 버전 통일, 학습 batch/num_hist/epoch/동일 예산 기준(12GB 스모크 후), probe 시드, Q3 dose alpha, 판정 패밀리의 구성(Holm 대상 비교 목록).
- 이 기록은 사전 등록(측정과 판정 규칙 고정)이며 가설 확정이 아니다. GPU 학습 실행과 논문 주장은 별도 사람 승인이 필요하다.

### 확정 기록 4: 후보 G 가설 확정 (2026-11-02, 사용자 승인)

- 사용자가 "1승인"이라고 답해 후보 G(Q1 H-a, Q2 H-b, Q3 H-c)를 실험 002 의 검증 대상 가설로 확정했다. (직전에 제시한 미결 항목 1번 "후보 G 의 가설 확정"에 대한 답.)
- 확정 범위: 가설 문장과 확정 기록 3 의 사전 등록 값. 확정되지 않은 것: Holm 대상 비교 목록, probe 시드, Q3 dose α, 학습 설정(batch/num_hist/epoch/동일 예산/torch), GPU 학습 실행 승인, 논문 주장.

### 확정 기록 5: 후보 G 사전 등록 보완 (2026-11-02, 사용자 채택)

- 사용자가 "일단 네 제안대로"라고 답해, 직전에 제시한 권고(미결 항목 2번)를 그대로 채택했다. 값은 근거 수치가 아니라 제안값이다. 결과를 본 뒤 바꾸지 않는다.
  - Holm 패밀리: 가설마다 별개 패밀리(각각 0.05), 가설 간 보정 없음. H-a 주 검정 4개 = 역할(보간, 외삽) x probe(linear, attentive), 각각 mask 면적 baseline 대비 R² 차 >= 0.1 (CI 하한 > 0)이고 random-init 인코더도 이김. H-b 주 검정 2개 = 마지막 predictor 블록, 마지막 rollout 시점의 Q1 대비 하락 >= 0.2. H-c 주 검정 2개 = steering, 부분공간 제거, 고정한 층에서 무작위 100방향 분포의 95 백분위 밖이고 alpha 에 단조. 조건 (a) 대 (b) 비교는 별도 패밀리로 지금 선언(실행은 (a) 재학습 후). 나머지(히트맵, 크기별 R², 다른 층/rank, 행동 대조)는 탐색이며 p값/유의성 주장 없이 그림만.
  - probe 시드 5개(0~4). 시드 평균을 "CI 가 좁다"는 근거로 쓰지 않는다 (학습 시드 3개가 불확실성을 지배).
  - Q3: alpha in {-2,-1,-0.5,0,0.5,1,2} x sigma_u (sigma_u 는 학습 에피소드에서 잰 값). 지표는 비율이 아니라 절대 변화량. 무작위 방향은 데이터 부분공간에서 같은 투영 분산과 같은 이동 크기. 층은 중간과 마지막 블록으로 고정(블록 수 확인 후 인덱스 치환, 데이터를 보기 전에).
- 구현 방침 (사용자 지시): 코드만 작성한다. GPU 서버는 별도 컴퓨터이므로 이 머신에서 학습/GPU 실행을 하지 않고, 사용자가 서버에서 직접 테스트할 수 있게 스모크/셀프테스트 명령을 README 에 둔다. GPU 학습 실행은 사용자가 서버에서 직접 하며 별도 승인 사항이다.

### 확정 기록 6: 후보 G 사전 등록 변경 및 보완 (2026-11-02, 사용자 회신) — "사전 등록 후 변경" 포함

시점: 코드 작성 후, GPU 실행과 결과 확인 전. 아래 값은 결과를 보기 전에 확정했으므로 결과 확인 후 바꾸지 않는다. 바꿔야 하면 별도 기록으로 남긴다.

1. **Q1 타깃 (사전 등록 후 변경, 확정 기록 3/5 와 문서 기준 대비)**: 주 타깃 = 명목 면적비의 로그(에피소드 상수). 마스크 면적 로그는 보조 타깃. mask 면적 baseline R² >= 0.99 이면 해당 판정은 not_evaluable. 주 판정은 변형이 큰 후반 프레임(late_deformed)에서의 일반화. 초기 프레임(early_flat) 결과도 baseline R² 와 함께 보고. 0.99 기준은 에이전트 제안이었고 사용자가 승인함.
2. **단조성 기준 (변경)**: q3_impl.monotonic_min_spearman 1.0 -> 0.9. α 곡선은 에피소드 평균으로 계산. 주 기준(크기 방향 효과가 무작위 방향 효과 분포의 95% 밖)은 유지.
3. **동일 예산 (확정 기록 3 의 미정 항목 확정)**: 학습 step 수와 에피소드 수를 모두 맞춘다. (a) = ind 900 에피소드. (b) = 3크기 x 300 에피소드(900). 보조 비교: (b) 를 크기당 100 에피소드(총 300)로 줄인 설정을 가능하면 실행(탐색). 이는 확정 기록 3 의 "크기당 300 episode" 를 조건 (a) 에서 900 으로 바꾸는 것이다.
4. **(a)-(b) 비교와 probe**:
   a) 주 측정 = 출력 공간 probe. Q1 에서 실제 인코더 feature 로만 학습한 probe 를 predictor 의 예측 z 에 그대로 적용. rollout t+1, t+2, ... 시점별로 읽은 크기와 실제 크기를 비교. 기준선 = 학습 크기 에피소드에 같은 probe 를 적용한 값. (a) 와 (b) 에 같은 probe 사용. predictor 활성값으로 probe 를 새로 학습하지 않는다.
   b) 중간 층별 probe 는 (b) 가 주 결과. (a) 의 층별 probe 는 보조: 절댓값 비교 없이 layer 0 대비 상대 변화로만 보고. 한계(ind 에서 라벨이 상수라 3크기 활성으로 probe 를 학습하므로 (a) 에서는 학습 때 본 적 없는 크기의 활성을 읽음)를 결과 문서에 명시.
   c) 진단 지표 추가: (a) 활성값이 a060, a160 에서 ind 대비 이동한 정도(층별 활성 평균·분산 이동량을 ind 기준 정규화)와 다음 프레임 예측 오차가 ind 대비 얼마나 커지는지. 이동이 크면 (a) 의 층별 결과는 해석하지 않는다. "이동이 크다"의 임계값은 미정이므로 결과 보기 전에 별도 기록이 필요하다.
   d) 크기별 활성 정규화는 하지 않는다.
5. **주 검정(확증)**: H-b = linear probe 로 미지 크기 평가. H-c = steering(ridge 방향)과 INLP 제거. LEACE, 평균 차이 방향 등은 탐색으로만 보고하고 가설 지지 근거로 쓰지 않는다.

- 미정으로 남은 것: 4c 의 "이동이 크다" 임계값, 학습 batch/num_hist/epochs/torch (서버 스모크 후), GPU 학습 실행 승인, 논문 주장.

### 확정 기록 7: 4c 진단 기준 확정, 사전 등록 변경 두 건의 상세 기록, torch 버전 (2026-11-02, 사용자 회신)

공통 상태: 확정 일시 2026-11-02 (세션 내, 커밋 전; 002 파일은 아직 커밋되지 않았다). **서버 결과 미확인**: 실험 002 의 학습, 캐시, probe, 개입 결과는 물론 smoke/파일럿도 본 적이 없다 (이 머신에서 본 것은 합성 데이터 selftest 뿐이며 가설 결과가 아니다). 아래 값은 결과 확인 후 변경하지 않는다. 변경이 필요하면 별도 기록으로 남긴다.

**A. 4c "이동이 크다" 판정 규칙 (확정값, pending_threshold 아님)**
- 잡음 기준(noise floor): ind 에피소드를 겹치지 않는 두 묶음으로 나눠 층별 활성 이동량(평균·분산 이동량을 정규화한 값)을 부트스트랩으로 반복 계산하고 그 중앙값을 noise floor 로 한다. a060, a160 데이터는 쓰지 않는다.
- 활성 이동량 판정: noise floor 의 2배 미만 = 작음, 2배 이상 5배 미만 = 중간, 5배 이상 = 큼.
- 예측 오차 판정: a060, a160 의 다음 프레임 예측 오차가 ind held-out 의 2배 이상이면 "큼".
- 최종 판정: 활성 이동량과 예측 오차 중 하나라도 "큼"이면 큼, 아니면 활성 이동량 판정을 따른다.
- 해석 규칙: 작음 = (a) 의 층별 결과를 보조 결과로 정상 해석. 중간 = 해석하되 낯선 입력의 영향 주의 문구 첨부. 큼 = (a) 의 층별 결과는 해석하지 않고 수치만 보고.

**B. 사전 등록 변경 기록 (원래 값 보존)**

| 항목 | 원래 값 | 새 값 | 이유 | 변경 시점 | 결과를 본 적 |
|---|---|---|---|---|---|
| 조건 (a) 에피소드 수 (data.train_episodes_per_level 의 (a) 적용분) | 300 (확정 기록 3: 학습 크기당 300 episode; 문서 제안값 채택) | 900 (ind 단일) | (b) 와 학습 step 및 에피소드 수를 맞추기 위함 | 2026-11-02, 커밋 전 (확정 기록 6 에서 선언, 본 기록에서 상세화) | 없음 (파일럿 포함 본 결과 없음) |
| q3_impl.monotonic_min_spearman | 1.0 (실험 코드 작성 시 에이전트가 정한 구현 기본값. 사용자 사전 등록 문구는 "alpha 에 단조"였고 수치 기준은 없었다) | 0.9 | 7점 곡선에서 인접 한 쌍의 뒤바뀜에 대한 과도한 기각 방지 | 2026-11-02, 커밋 전 (확정 기록 6) | 없음 (파일럿 포함 본 결과 없음) |

- 확정 기록 6 의 Q1 타깃 변경(사전 등록 후 변경)도 같은 상태(2026-11-02, 결과 미확인)다. 원래 값: 문서의 "mask 면적 기반 크기" 타깃. 새 값: 주 타깃 log_nominal_area_ratio, 보조 log_frame_mask_area.
- 001 의 결과(크기 붕괴 진단)는 본 적이 있다. 002 의 설계에 영향을 준 것은 001 리뷰의 한계 사항(대조군 없음, 임계값 코드 미반영 등)이며 이는 확정 기록 3 에 사전 등록된 항목의 배경이다. 002 변경 두 건의 값 결정에 001 의 수치를 쓰지 않았다.

**C. 환경**: PyTorch 2.0.1+cu117 로 통일 (사용자 지정). env.torch_version 에 반영한다.

---

## 후보 H: 학습 분포 의존성 (끝 크기 단일 학습, 혼합 학습) — 1단계 행동 대조

- 작성: 메인 세션, 2026-11-03. 근거: reports/001_size_collapse_diagnosis.md 의 한계 "학습 크기(ind)가 평가 범위의 가운데라 '학습 크기로 수렴'과 '가운데로 압축'을 구별할 수 없고, 크기를 섞은 대조군이 없다".
- 상태: **가설 문장은 사용자가 2026-11-03 확정("1단계 가설 확정"). 아래 판정 기준 수치는 내가 낸 제안값이며 사용자가 값을 직접 확인하지 않았다. 근거 있는 값이 아니다. 학습 실행 전에 확인/수정을 받는다.**

### 가설 문장
- **H-H1 (학습 크기 수렴 vs 가운데 압축)**: 끝 크기 하나로만 학습한 DINO-WM(a026 단일, a248 단일)의 open-loop 면적(k=12)은 평가 크기들을 **그 학습 크기 쪽으로** 끌어당긴다. 가운데 압축이라면 학습 크기를 옮겨도 끌림의 중심은 ind 근처에 남는다.
- **H-H2 (혼합 학습의 효과)**: 크기를 섞어 학습하면(mix_adjacent: a080+ind+a125, mix_wide: a040+ind+a160) 단일 학습(baseline_single)보다 open-loop 수축(크기 곡선 기울기 감소)이 줄어든다.

### 측정 (001 파이프라인과 같은 지표, 학습 분포만 바꿈)
- 학습 분포: a026 단일, a248 단일, mix_adjacent, mix_wide. 대조: 001 의 baseline_single(ind, 시드 0/1/2, 이미 있음).
- 지표: 크기별 open-loop 면적비 ratio_ol_over_real (k=12, 에피소드 평균), 크기 곡선의 로그-로그 기울기(ol@k12), 끌림의 중심 c = 로그 면적비가 0 을 가로지르는 크기(크기 격자 사이는 선형 보간).
- 학습 시드 0/1/2, 평가는 9개 크기 x 100 test 에피소드(001 과 동일). 시드는 config.yaml 에 명시.

### 성공 기준 (제안값, 확인 필요)
- H-H1 지지: a026 단일과 a248 단일 모두에서, 3개 시드 전부 c 가 ind 보다 학습 크기에 더 가깝다(로그 면적 거리). 그리고 학습 크기 반대쪽 크기들의 면적비 CI 가 1 을 학습 크기 방향으로 제외한다(episode bootstrap 95%).
- "가운데 압축" 지지: 두 모델 모두에서 3개 시드 전부 c 가 학습 크기보다 ind 에 더 가깝다.
- 그 밖(모델 간 또는 시드 간 갈림) = 판정 불가, 그대로 보고한다.
- H-H2 지지: ol@k12 기울기가 baseline_single 보다 [제안 0.2] 이상 크고, 3개 시드 모두 같은 방향이다(분포별 3 시드, 시드 간 대응 없이 평균 비교는 하지 않음). 외삽 크기 결과는 따로 보고하고 H-H2 판정에는 쓰지 않는다.

### 반증 조건
- H-H1: 학습 크기를 옮겨도 c 가 ind 근처에 머문다 -> 가운데 압축 쪽 증거.
- H-H2: 혼합 학습의 기울기가 baseline 과 [제안 0.2] 이내로 같다 -> 크기 노출이 수축을 줄이지 못함(혼합 범위 밖 원인 가능).

### 알려진 한계와 주의
- 시드 3개(자유도 2)라 방향성만 신뢰한다. 001 에서 시드 간 변동이 컸다(예: a248 ol 면적비 시드 범위 0.333~0.549).
- 001 의 tf1 k=12 기울기는 fig1(약 0.98~1.0)과 results.json `by_k/12/tf1` 에서 내가 다시 만든 값(0.91~0.92)이 서로 다르다. **원인 미확인**이라 이 실험의 주 지표는 ol 면적비/기울기와 c 로 하고 tf1 기울기는 쓰지 않는다. 정의는 코드(`size_curve` 계산)를 확인한 뒤 고정한다.
- 이 실험은 원인 층위(디코더/predictor/정보)를 가르지 않는다. 학습 분포 의존성만 본다. 층위는 002(후보 G)에서 본다.
- 학습 분포는 001 config 의 TODO "사람 확정 필요" 항목이다. a026/a248 단일 분포는 001 config 에 아직 없다(추가 필요).

### 환경 제약 (사용자 지정, 2026-11-03)
- GPU 는 12GB VRAM 으로 다시 세팅한다. 001 학습 설정(batch 24, 2,000 에피소드, 15 epoch)이 12GB 에 들어가는지는 **확인하지 못했다**. 002 에 12GB 스모크(`run.py smoke`)가 있으나 서버 미실행이다. 학습 전에 스모크로 메모리/시간을 실측하고, 안 맞으면 batch/에피소드 수를 바꾸되 baseline_single 과 같은 예산을 유지한다(바꾸면 별도 기록).

### 체크리스트 (학습 실행 전)
- [x] 가설 문장 확정 (2026-11-03, 사용자)
- [x] H-H1/H-H2 판정 수치(0.2, "3 시드 전부") 제안값 그대로 채택 (2026-11-03, 사용자: "네 제안대로 하고 구현해")
- [x] 학습 분포 4종 승인 (a026 단일, a248 단일, mix_adjacent, mix_wide) (2026-11-03, 같은 발언)
- [ ] 12GB 스모크 결과 확인 후 학습 설정 확정
- [ ] GPU 학습 실행 승인

### 확정 기록 8: 후보 H 사전 등록 (2026-11-03, 사용자)

- 사용자가 후보 H 의 판정 수치(기울기 차 0.2, 시드 3개 전부 같은 방향)와 학습 분포 4종을 제안값 그대로 채택하고 구현을 지시했다. 값은 근거 수치가 아니라 제안값이다. 결과를 본 뒤 바꾸지 않는다.
- **GPU 학습 실행 승인은 받지 않았다.** 구현과 CPU 점검까지만 진행한다. 12GB 스모크 후 학습 설정과 실행은 별도 승인.

### 확정 기록 8: 학습 설정 확정 (2026-11-03, 사용자 결정, GPU 학습 실행 승인 포함)

시점: 12GB 서버 smoke 실행 후, 본 학습 결과를 보기 전. 근거: results/smoke.json (서버 산출물, 이 저장소에는 없음). 사용자가 본 smoke 수치(화면 캡처로 전달)는 아래뿐이며 가설 결과는 본 적이 없다.
- train.batch_size = 24, train.num_hist = 1, train.epochs = 15 ((b) multi_60_ind_160 기준 epoch. 다른 조건은 총 step 이 같아지도록 환산, train.budget 은 null 유지).
- 근거(smoke): num_hist=1 batch 24 는 OOM 없이 peak 9045MiB, 1.14 it/s. num_hist=3 batch 24 는 peak 12103MiB(한도 12287MiB 근접), 18.07 s/it 로 약 20배 느려 제외(원인은 추정). batch 24 와 num_hist=1 은 001·브랜치 학습 job(frameskip=5 num_hist=1 batch_size=24 training.epochs=15)과 같다.
- 실행 방식: overnight.sh 로 6개 학습(조건 2 x 시드 3) 후 cache, q1, q2, q3, aggregate 를 한 번에 실행. 사전에 파이프라인 단독 검증(한 run)은 하지 않기로 했다 — cache/q1/q2/q3/aggregate 는 아직 한 번도 실행된 적이 없어 학습 후에 오류가 날 수 있음을 사용자에게 고지함.
- 시간 추정(추정): epoch 당 약 1856 iteration(smoke 의 48 에피소드=99 iteration 을 900 에피소드로 비례 환산), 1.14 it/s 로 run 당 약 6.8시간, 6 run 약 41시간. 정확한 step 수는 train --dry-run 과 train_budget.json 으로 확인.
- 이 값들은 결과를 보기 전에 확정했으므로 결과 확인 후 바꾸지 않는다.

---

## 후보 I: 깊이별 크기 소실 지점(실험 4)과 inductive bias probe(실험 5) (미확정, 사람 승인 대기)

- 작성: 메인 세션, 2026-11-03. 출처: 사용자가 붙여 준 「실험 4·5 구현 명세」(초안; 원본은 claude.ai 「DINO-WM 천 크기 정보 흐름 실험 설계」). 명세의 "실험 1~3" = 002 의 Q1/Q2/Q3 (후보 G), predictor (b) = 002 조건 `multi_60_ind_160`.
- 상태: **가설 문장과 수치는 명세의 제안값이며 사용자 확정 전이다.** 002 결과(Q1 probe, (b) 체크포인트)는 이 저장소에 아직 없다(2026-11-03 확인: experiments/002_single_size_sweep/results 없음). 명세 0.2절에 따라 결과가 없으면 구현 + 파일럿(`pilot: true`)까지만 한다.
- 명세와 저장소 관례 차이: 출력 위치를 `results/exp4/` 대신 저장소 관례 `experiments/004_depth_truncation/`, `experiments/005_inductive_bias_probe/` 로 둔다(명세 0.1절 "기존 관례를 따른다"). 각 폴더에 CHANGES.md.
- 이미 알려진 설정 차이(002 확정 기록 8): num_hist = 1, frameskip = 5. 명세 4-B 의 "마지막 context 프레임"은 유일한 context 프레임이 된다.

### 가설 (명세 제안값)
- **H-d (실험 4)**: 천 크기 R²(log 명목 면적비)는 어떤 깊이 L* 이후 급격히 줄고, OoD 크기에서 L* 가 InD 보다 앞선다. 소실 지점 = R² 가 직전 최고값보다 0.2 이상 처음 떨어지는 깊이(인코더 k*, predictor m* 따로). 주지표는 predictor 의 (i) hidden probe.
  - 반증: 모든 깊이에서 하락 < 0.2, 또는 InD/OoD 곡선이 seed 편차 안에서 겹침.
- **H-e (5-A/5-B)**: OoD 포함 R-IB, D-IB 가 모두 면적 휴리스틱 모델보다 높고 random-init 과 seed 편차 밖에서 다르며, 5-B calibration 곡선의 45도선 평균 이탈 ≤ 0.1(정규화 거리).
- **H-f (5-C)**: `D-IB_area=` 와 `D-IB_area≠` 가 seed 편차 안에서 같다.
- **H-g (5-D)**: slice 별 기울기가 모두 0.8~1.2.
- 임계값 0.2 / 0.1 / 0.8~1.2 는 명세상 "파일럿에서 확정". 바꾸면 결과보다 먼저 기록한다.

### 체크리스트
- [x] 가설 문장과 제안 임계값 확정 (2026-11-03, 사용자: 제안값 그대로. 임계값은 파일럿 후 최종 확정, 바꾸면 결과 보기 전에 기록)
- [x] 진행 범위: 코드 + 합성 데이터 CPU selftest + dry-run 까지 (2026-11-03, 사용자). 서버 파일럿은 002 결과 후 별도 승인
- [x] 5-B 무작위 함수 족: 무작위 사인파 합 (2026-11-03, 사용자)
- [x] LoRA 옵션, 4-C 깊이 재학습: 둘 다 생략, CHANGES.md 에 기록 (2026-11-03, 사용자)
- [ ] 파일럿 GPU 실행 승인

### 확정 기록 9: 후보 I 확정 (2026-11-03, 사용자)
- 위 체크리스트 네 항목을 사용자가 선택했다. 결과(002 포함)를 본 적 없음. 확정 시점의 값은 결과 확인 후 바꾸지 않는다(파일럿 후 임계값 확정은 명세상 예정된 절차이며 변경 시 별도 기록).

### 확정 기록 10 (후보 G / 실험 002): 실행 환경 변경, 서버 -> Google Colab Pro+ (2026-11-04, 사용자 지시)
(번호 정정: 처음엔 "확정 기록 9" 로 적었으나 후보 I 의 확정 기록 9 와 겹쳐 사용자 지시로 10 으로 바꿨다.)
- 원래 값: 학교 WSL2 GPU 서버(12GB VRAM, SSH), `overnight.sh` 연속 실행, 데이터는 rclone(Google Drive), conda 환경 `dino_wm`, `save_every_x_epoch: 5`.
- 새 값: Google Colab Pro+ 노트북(`experiments/002_single_size_sweep/colab/exp002_colab.ipynb`). 데이터는 마운트한 Drive 에서 `/content` 로컬로 복사(rclone 미사용), 체크포인트/캐시/results/manifests/logs/attempts.json 은 Drive `WORK_ROOT` 에 직접 기록, conda 대신 uv 로 만든 Python 3.9 venv + pip(environment.yml 핀 유지; CLAUDE.md conda 규칙의 예외, 사유: Colab 에 conda 없음/세션 초기화), 재개용 train.py 패치, `save_every_x_epoch` 5 -> 1(사용자 승인, 운영 값, 사전 등록 아님).
- **학습 재개(resume)를 쓴다 (사용자 결정; train.py 패치 승인 포함).** 경위: 2026-11-04 에 사용자가 먼저 "resume patch 를 쓰지 말자"고 지시했으나 이는 사용자의 실수였고, 같은 날 "체크포인트에서 학습을 재개할 수 있게 하라"로 정정했다(정정이 최종 결정). 고정 commit 의 `train.py` 는 `model_latest.pth` 를 읽되 encoder/optimizer 상태를 복원하지 못하고 epochs 만큼 더 학습해 총 예산(27855 step)이 깨지므로(소스 확인), `colab/train_resume.patch`(상위 11ba488 + `training.epochs` 를 총 목표 epoch 로 해석)를 pinned commit 위에 적용한다. 새 학습은 원본과 같이 epoch 1..15. 패치 없는 train.py 로는 체크포인트가 있는 run 을 재개하지 않는다(코드가 거부). 재개 전 `prepare`: 읽히는 체크포인트는 재개, 깨진 `model_latest.pth` 는 최신의 읽히는 epoch 파일로 복구, 쓸 수 있는 체크포인트가 없으면 `DINO_TRAIN/quarantine/<run>__<시각>/` 로 이동(삭제 없음)하고 처음부터. `save_every_x_epoch=1` 은 이 재개에 쓰이므로 의미가 있다.
- 재개의 한계(추정): 재개 후 데이터 순서/난수 상태가 복원되지 않아 중단 없는 run 과 bit 단위로 같지 않을 수 있다. cloth 환경, torch 2.0.1+cu117, Colab, GPU 에서의 재개는 검증하지 못했다(CPU selftest 는 패치 적용과 epoch 범위 1..15 / 6..15 만 확인). 총 optimizer step 수는 같다.
- 데이터 위치(사용자가 Drive 메타데이터로 확인해 전달): MyDrive 의 바로가기 `dino-wm-videos`(id 17M0MZGSDPzu7CCT3ez3ABMyZGjavTfx3, 소유 다른 계정) 안 `[data]/[cloth-sim] cloth_move` 가 config `root_folder_id` 의 폴더. 마운트 형태는 미검증.
- 사유: 서버가 꺼졌고 SSH 사용이 금지됨(사용자 보고).
- 불변(사전 등록 유지): seeds [0,1,2], batch_size 24, num_hist 1, epochs 15, 동일 예산(총 optimizer step 계산값 27855), 모든 사전 등록 임계값, 조건/데이터 구성. VRAM 여유가 생겨도 batch_size 는 바꾸지 않는다.
- 이전 서버 run: 중단됐다(사용자 보고). 사용자 보고 기준으로 가설 결과(Q1/Q2/Q3 수치)를 본 적 없다(이 저장소에 서버 산출물이 없어 실험자는 확인할 수 없음; smoke 수치는 확정 기록 8 에 기록된 것뿐). **서버에서 중단된 run 의 부분 학습 출력은 재사용하지 않고, Colab 에서 처음부터 다시 시작한다**(사용자가 달리 요청하기 전까지).
- 비교 가능성 주의: GPU 종류, 드라이버, venv 의 torch/CUDA 빌드가 서버와 다를 수 있고 재개가 일어난 run 은 중단 없는 run 과 bit 단위로 같지 않을 수 있다(추정). 결과 해석 시 환경 기록(`results.json environment`, `run_info.txt`)을 함께 본다.
- GPU 학습 실행 승인: 확정 기록 8 의 승인은 서버 실행에 대한 것이었다. Colab 본 실행 전에 환경/비용(컴퓨트 유닛) 확인을 사람이 한다. 이 항목 작성 시점에 Colab 실행은 없다.

### 확정 기록 11 (후보 H / 실험 003): 나머지 9개 모델 실행 승인과 운영 조건 (2026-11-05, 사용자 결정)

- 승인: single_a026, single_a248, mix_adjacent 각 시드 0, 1, 2 (9개 모델)의 GPU 학습. 예상 약 42시간(mix_wide 소요 시간 기준 추정, L4 에서 미측정).
- 사용자 결정 1: 세션이 끊기면 체크포인트에서 재개한다(002 의 train_resume.patch 이식). 이유: 기존 003 코드는 체크포인트가 남은 미완료 run 을 다시 실행하면 추가 학습 후 DONE 을 남길 수 있다(코드 확인 중, 003 에서 재현한 것은 아님). 재개가 일어난 run 은 구분해 기록한다.
- 사용자 결정 2: 현재 Colab L4 런타임으로 진행한다. mix_wide 3개는 A100 으로 학습한 것으로 보이므로(smoke 기록 GPU 메모리 40,960MiB, 서버 주소 a100) 분포 간 GPU 가 다르다. 학습 코드도 다르다: mix_wide 는 패치 없음, 나머지 9개는 패치 적용. 이 차이는 판정 표에 반영되지 않으므로 리뷰에서 함께 본다.
- 변경하지 않는 값: 분포 4종, 시드, batch 24, 15 epoch, num_hist 1, frameskip 5, predictor_lr, criteria(기울기 차 0.2, 시드 3개 전부, 9쌍 규칙). `training.save_every_x_epoch` 5 -> 1 은 재개를 위한 운영 값이며 사전 등록 값이 아니다.
- 결정 시점에 본 것: mix_wide 3개의 results.json 판정(H-H2 mix_wide 부분 pass, H-H1 과 H-H2 전체 not_evaluable). 이 값을 보고 임계값이나 규칙을 바꾸지 않았다.

### 확정 기록 11 변경 (후보 H / 실험 003): GPU 를 L4 에서 A100 으로 변경 (2026-11-05, 사용자 결정)

- 이유: L4 첫 epoch 약 74분, mix_wide 의 A100 약 18.5분. 사용자가 런타임을 A100 으로 바꿨다. 확정 기록 11 의 "현재 L4 로 진행" 을 이 기록이 대체한다.
- exp003_single_a026_s0 경위: L4 에서 epoch 1 을 마친 뒤 A100 런타임에서 10번 셀을 시작했고(08:17:58 UTC), 08:18:06 UTC 에 run 폴더가 quarantine/ 으로 이동됐다. 재개 로그 `Resuming from epoch` 이 없고 epoch_logs 에 epoch 1 이 다시 기록됐으며 값도 거의 같아(train_loss 1.9476 대 1.9477, val_loss 1.5289 대 1.5290), 체크포인트를 읽지 못하고 같은 시드로 epoch 1 부터 새로 학습한 것으로 본다(원인은 폴더 이동 시점 때문으로 추정). 10:47 UTC 쯤 폴더를 outputs/ 로 되돌렸고 사용자가 이어 가기로 결정했다.
- 기록 주의: 이 run 의 train_env.json 은 action=resume, n_resumes=1, gpus_used 2개로 남아 있지만 실제 학습은 A100 재시작으로 본다. epoch_logs.jsonl 첫 줄은 L4 의 epoch 1 이다. results.json 에 그대로 복사되므로 해석 때 이 기록을 우선한다. 나머지 8개 모델은 끊기지 않으면 A100 이다. 판정 표에 반영되지 않으므로 리뷰에서 함께 본다. 사전 등록 값은 변경 없음.

### 후보 J / 실험 006: 002 결과의 탐색 재분석 계획 (2026-11-05, 사용자가 재분석 진행 지시. 판정 없는 탐색)

- 목적: 002 리뷰가 미정으로 남긴 질문에 단서를 얻는다. (1) rollout R² 하락이 크기 정보 소실인가 판독기 전이 실패인가, (2) 예측이 특정 크기로 몰리는가, (3) 하락이 크기 특이적인가.
- 분석 5종: (A) 크기 레벨별 예측값 평균과 기울기(편향 부호), rollout step 별. (B) persistence(마지막 context 프레임 토큰 복사)와 untrained predictor 의 같은 output-space R². (C) 예측 latent 로 재학습한 probe 와 전이 probe 비교. (D) 크기와 무관한 속성의 같은 rollout 감쇠. (E) 시드 단위 평균과 CI, MAE, 순서 정확도 병기.
- 지위: 전부 탐색이다. 임계값, pass/fail, p 값 기반 유의성 주장을 만들지 않는다. 002 의 판정과 임계값은 바꾸지 않는다. 결과는 수치와 그림으로만 보고하고 /review-experiment 로 해석한다.
- 입력: 002 의 Drive 산출물과 체크포인트(읽기 전용), 002 학습 시드 그대로. 새 학습 없음. 시드는 config.yaml 에 명시한다.
- 시점: 002 results.json 과 리뷰를 본 뒤 세운 계획이다. 따라서 이 분석은 사후 탐색이며 확증 근거로 쓰지 않는다.
- GPU: 체크포인트 추론이 필요한 항목은 사람이 Colab 에서 실행한다. 실행 전 예상 시간과 함께 다시 확인받는다.

#### 후보 J 추가 (2026-11-10, 사용자 지시): 분석 F 와 실행 승인
- 배경(리드 종합, 추정, 분석가 리뷰 전): 004 는 한 step 안의 깊이 축을 봤고 소실이 없었다. 관측된 하락(002 rollout step 1 -> 12)은 시간 축이다. 따라서 질문 (4) 를 추가한다: rollout 하락이 매 step 의 1-step 예측 손실인가, 자기 출력을 다시 입력하며 생기는 누적인가.
- 분석 (F) teacher-forcing 대 open-loop: 각 step t 에서 실제 인코더 latent context 로 1-step 예측한 latent(teacher-forcing)와 기존 open-loop rollout latent 에 같은 전이 probe(및 C 의 재학습 probe)를 적용해 R², MAE, 레벨별 예측 평균을 step 별로 나란히 보고한다. 같은 checkpoint, 같은 test episode, 같은 시드.
- 지위: A-E 와 같이 탐색이다. 임계값, pass/fail 없음. 확증 근거로 쓰지 않는다.
- 실행: 사용자가 Colab 에서 009 와 병행 실행하기로 함(2026-11-10). 이를 006 GPU 실행 승인으로 기록한다. 노트북은 009 처럼 git clone 방식(코드 임베드 없음).

### 007 진단 질문: 003 디코더 붕괴 run 의 체크포인트 검사 (2026-11-07, 사용자 승인; 가설 확정 아님)
- 지위: 가설이 아니라 진단 질문이다. 재학습과 GPU 사용은 승인되지 않았다. 입력은 Drive 휴지통의 003 체크포인트(읽기 전용)이다. 아래 검사와 판단 기준은 결과를 보기 전에 적는다.
- 질문: exp003_mix_adjacent_s0, exp003_single_a026_s1 의 val_img_mse_reconstructed 가 epoch 2 -> 3 에서 급등(0.0030 -> 0.0787, 0.0015 -> 0.0333)하고 디코딩 프레임이 입력과 무관해진다. 체크포인트(붕괴 run model_2/3/(4), 대조 run mix_adjacent_s1/s2, single_a026_s2 model_2/3)에서 다음 관측이 되는가.
- 검사: A 코드 사실(5a640bed) / B 층별 가중치 통계와 epoch 2->3 변화 / C decoder_optimizer 상태 / D 같은 입력 프레임으로 디코더 층별 활성화(입력 간 std, ReLU 비활성 비율), 출력 std, 실제 프레임 MSE, 복원 시트 PNG / E predictor 출력의 입력 간 std.
- 판단 기준(결과는 이 표현으로만 보고, 항목마다 해당/비해당 + 대조 값 병기):
  (1) NaN/Inf 있음. (2) 입력 의존성이 처음 사라지는 층이 특정됨. (3) 활성화가 전부 비활성인 층이 특정됨. (4) epoch 3 가중치 변화가 대조 대비 10배 이상인 층이 특정됨. (5) 어느 것도 해당 없음.
  체크포인트만으로 구분 불가한 것(epoch 3 도중 정확한 시점, 단일 큰 gradient 여부)은 "체크포인트로 구분 불가"로 쓴다.
- 조작적 정의(사전): "입력 의존성이 사라진 층" = 층 출력의 입력 간 std(샘플 축 std 의 원소 평균)가 같은 층의 대조 run 값의 1/100 미만. "전부 비활성" = ReLU 입력이 모든 입력, 모든 위치에서 <= 0 (비율 1.0). "10배 이상" = 붕괴 run 의 ||W3-W2||/||W2|| 가 대조 3개 run 상대 변화 최댓값의 10배 이상.
- 해석은 쓰지 않는다. 해석은 /review-experiment 를 거친다.

### 확정 기록 12 (후보 I / 실험 005): exp1_negative 와 정규화 기준 (2026-11-09, 사용자 결정: 리드 추천 채택)
- 시점: 004/005 실행 전. 004/005 결과는 아무것도 보지 않았다. 002 results.json 은 본 뒤다(이 값은 002 결과를 입력으로 정하는 항목이다).
- `branch.exp1_negative: false`. 근거(002 results.json decision_table, H-a): 보간 linear_pooled pass(R² real 0.514 / random-init 0.174 / mask baseline -0.042), 보간 attentive pass(0.182 / -0.018 / -0.042). 외삽 두 검정은 fail 이지만 mask 면적 baseline 이 이미 높아서다(외삽 mask baseline 0.825; real linear 0.859, attentive 0.700). 실험 1 음성(hypotheses.md 155행: probe R² 가 mask baseline 또는 random-init 과 구분되지 않음)에 해당하지 않는다. true 로 두면 4.6 분기 5행에 따라 실험 5 전체가 hold 된다. 외삽에서 면적 단서가 강하다는 점은 005 의 면적 휴리스틱 head 와 5-C 가 직접 다룬다.
- `normalization.noninf: {R: iid, D: const}` (기존 {R: iid, D: iid}). 근거(005 CHANGES.md 8번, 셀프테스트 사실): 무작위 이진 라벨 과제에서 oracle 의 D-IB raw 와 i.i.d. 기준의 D-IB raw 가 모두 약 0.5 라 분모가 0 에 가깝고 점수가 NaN -> H-e 가 구조적으로 not_evaluable 이 된다. 상수 출력 기준은 D-IB raw 0 이라 분모가 약 0.5. R-IB 는 반대로 상수 기준이 R-IB 1(oracle 도 약 1)이라 분모가 0 에 가까우므로 i.i.d. 를 유지한다. 즉 각 지표에서 분모가 정의되는 기준을 고른다. 반대쪽 기준 점수는 `score_alt` 열로 계속 저장한다.
- 명세 4.2 문구("i.i.d. 무작위 예측")에서 D-IB 만 벗어난다. 원문(Vafa et al. 2025, App. B) 식은 여전히 미확인이다. 임계값(0.1, 0.8~1.2, min_gap 0.05, sd_mult 2.0)은 바꾸지 않는다.

### 008 사후 재집계 (004/005 후속 1단계, 2026-11-10, 사용자 실행 지시; 가설 확정 아님)
- 지위: 사후 탐색. 004/005 의 결과와 리뷰를 본 뒤 세운 분석이다. 확증 근거로 쓰지 않고, 004 판정(H-d refuted)과 005 판정은 바꾸지 않는다. CPU 만 사용, 새 학습 없음, 입력은 004 results/depth_r2.csv 와 기존 캐시(읽기 전용).
- 아래 기준은 재집계 결과를 보기 전에 적는다.
- A. 통제 대비 곡선: Δ(d) = R²_real(d) - R²_random-init(d) (같은 probe 종류, split, 층 위치, probe seed 대응). 주 곡선은 004 와 같은 (linear, new, late) 설정. 소실 규칙은 004 와 같다: Δ 가 직전 최고값보다 0.2 이상 처음 떨어지는 깊이. 보조로 각 깊이에서 Δ 가 0 보다 큰지(학습 시드 평균 - 2 x 학습 시드 sd > 0)를 보고한다.
- B. 척도 통일: split 별 R² 대신 (i) MAE, MSE (log 면적비 단위), (ii) 전체 평가 episode 를 합친 공통 라벨 분산을 분모로 한 R² 로 InD/OoD 곡선을 비교한다. CSV 와 캐시만으로 계산할 수 없으면 계산하지 않고 그 사실을 보고한다(추정치로 대체하지 않음).
- C. 겹침 검정 재계산: 분산 단위를 학습 시드(3개)의 seed 평균 간 sd 로 둔다. 기준 수치(2.0 x sd, 깊이 비율 0.8)는 004 config 값을 그대로 쓴다.
- 보고 표현: 각 항목을 "관측됨 / 관측 안 됨 / 계산 불가" 로만 쓴다. 해석은 /review-experiment 를 거친다.
- 사용자 결정(2026-11-10, 리뷰 후): 008 결과는 사후 탐색으로만 남긴다. 004 의 H-d refuted 를 보강하는 근거로 인용하지 않는다(리뷰 3명 공통: 규칙의 검정력이 낮음; reports/008_posthoc_reaggregate_004_005.md).

### 009 양성 대조: 004 파이프라인의 검정력 (2단계 초안, 2026-11-10; 승인 대기, 코드 작성만 허용)
- 질문: 004 의 측정 파이프라인(깊이별 probe + 0.2 하락 규칙)이 실제로 크기 정보가 사라지는 predictor 에서 소실을 검출하는가.
- 개입(초안): 조건 (b) 체크포인트의 predictor 블록 m_int(제안 3)의 출력 hidden 에서 크기 방향을 제거한다. 크기 방향 = 학습용 분할에서 적합한 선형 probe 의 가중치 방향(또는 LEACE 류 선형 소거). 제거 강도 0, 0.25, 0.5, 1.0. 개입 이후 블록은 그대로 순전파하고, 004 와 같은 probe 와 규칙으로 깊이별 R² 를 다시 잰다. 개입 방향 적합 데이터와 probe 평가 데이터는 분리한다.
- 성공 기준(초안): 강도 1.0 에서 학습 시드 3개 중 2개 이상이 m* ≤ m_int + 1 로 검출되면 "파이프라인에 검출력 있음". 강도 1.0 에서도 검출되지 않으면 004 의 refuted 는 검정력 없는 결과로 표시한다. 개입 이후 블록에서 R² 가 회복되는지는 보조 관측으로만 보고한다.
- 개입 블록 여러 개(2026-11-10 사용자 요청): m_int 는 리스트(초안 [1, 3, 5]; 1..6 중 부분집합 가능). 성공 기준은 블록마다 따로 판정한다. 강도 0 기준선은 공유한다.
- 사용자 결정(2026-11-10, 008 리뷰 후): m_int = [1, 2, 3, 4, 5, 6] (가능한 모든 블록). 블록 6 은 뒤에 블록이 없으므로 회복 관측은 하지 않고 검출만 판정한다.
- 사용자 결정(2026-11-10): 008 의 A, B 규칙을 009 에도 같은 임계값으로 함께 적용한다. A = 개입 predictor 의 R² 에서 random-init predictor 의 R² 를 뺀 Δ 곡선에 0.2 하락 규칙. B = 공통 라벨 분산 R²(008 과 같은 Var_all 정의)에 0.2 하락 규칙, MAE/MSE 함께 보고. 둘 다 블록 x 강도별로 보고하는 보조 판정이며, 주 성공 기준(위)은 바꾸지 않는다.
- 확정 기록 (2026-11-10, 사용자 승인; 009 결과는 아무것도 보지 않음): 소거 방식 `leace_scalar`. 강도 [0, 1.0] (0.25, 0.5 제외: 방향 성분의 축소는 가역 선형 변환이라 재학습 linear probe 가 복원한다는 근거, CPU toy 에서 강도 0.5 R² 0.996 = 기준선 0.996). 주 성공 기준: 강도 1.0 에서 학습 시드 3개 중 2개 이상이 m* ≤ m_int + 1 이면 해당 블록 "검출력 있음", 블록마다 따로 판정. m_int = [1..6]. A/B 는 보조 판정. 실행은 사용자가 Colab 에서 한다.
- 사람이 정할 것: m_int 리스트, 소거 방식, 성공 기준의 확정, GPU 실행 승인.

### 010 원천 분리 대조: 005 의 ours ≈ random 의 원인 위치 (2단계 초안, 2026-11-10; 승인 대기, 코드 작성만 허용)
- 질문: 005 에서 학습된 predictor 와 random-init predictor 가 비슷한 것이 인코더 표현 때문인가 predictor 때문인가.
- 방법(초안): 005 와 같은 head, 과제, 평가 입력(n_eval 900, 005 실행과 같음)으로 네 표현을 비교: 학습된 인코더 최종 층(004 캐시 tokens_k12), random 인코더(rand_k12), pred_out6 ours, pred_out6 random. 과제/라벨 시드(seeds.tasks_a, seeds.d)를 3개 이상으로 바꿔 변동에 포함한다.
- 판정(초안): H-f 를 사전에 정한 동등성 여유 Δ 로 TOST 판정(Δ 제안값 0.05, 승인 대기). 네 표현 간 D-IB/R-IB 점수 차이를 과제 시드 포함 sd 와 함께 보고한다.
- 사람이 정할 것: Δ, 과제 시드 수, GPU 실행 승인.
