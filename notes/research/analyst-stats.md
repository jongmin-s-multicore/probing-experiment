# 통계·데이터 관점 자료조사: DINO-WM의 physical attribute 보존 / 활용 검증

작성: analyst-stats (자료조사 모드). 가설 H1/H2는 미확정. 실험 없음.
방법: notes/research/papers/worldmodel_jepa/ 의 5편 PDF를 pdftotext -layout 으로 텍스트화해 읽음 (변환본은 scratchpad, 읽기 목적만).
표기: "논문값" = 논문 표에서 그대로 옮김. "추정" = 내 판단. "미확인" = 추출 텍스트에서 찾지 못함 (PDF 전체를 정독하지는 않았으므로 논문에 있을 수도 있음).

---------------------------------------------------------------------------
## 1. 5편의 probing 설정 정리

### 1.1 LeWM (LeWM.pdf) - 핵심 레퍼런스
- 대상 attribute (Sec 5.1, App F.2): TwoRoom = agent 2D position (Tab 3). PushT = agent location, block location, block angle (Tab 1). OGBench-Cube = joint position/velocity, end-effector position/yaw, gripper, block position/quaternion/yaw (Tab 4).
- probe: linear 과 MLP 둘 다. "Linear = 선형 접근 가능성, 비선형 = 정보는 있으나 얽혀(entangled) 있는지" 로 해석한다고 서술 (App F.2).
- 지표: MSE 와 Pearson r. (R^2 아님.) 표기는 "평균 ± 값"; 이 ± 가 샘플 간 std 인지 시드 간 std 인지 본문에 명시 없음 (미확인). 값 형태(std > mean인 경우 다수)로 보아 샘플(예측 오차 분포)의 std 로 추정.
- 샘플 수, probe 학습 hyperparameter, 학습/평가 분할, probe 시드 수: 추출 텍스트에서 미확인.
- 입력: LeWM 임베딩 192차원(단일 latent). DINO-WM 쪽에서 probe 에 넣은 표현이 patch token 인지 pooled 인지 미확인. (DINO-WM 원 구조는 14x14x384 patch.)
- 베이스라인: PLDM, DINO-WM (비교 대상). random-init encoder, shuffled-label, control task 등 선택성 대조군은 없음 (미확인, 표에 없음).
- 학습 시드: 별도로 Tab 5 (Push-T 성공률, 3 training seeds, 50 trajectories): DINO-WM 92.0 ± 1.63, PLDM 78.0 ± 5.0, LeWM 96.0 ± 2.83. probing 표가 시드 반복인지는 불명.

논문값 (Tab 1, Push-T; MSE ± , r):
| Property | Model | Linear MSE | Linear r | MLP MSE | MLP r |
|---|---|---|---|---|---|
| Agent Location | DINO-WM | 1.888 ± 0.500 | 0.977 | 0.003 ± 0.022 | 0.999 |
| Agent Location | PLDM | 0.090 ± 0.311 | 0.955 | 0.014 ± 0.119 | 0.993 |
| Agent Location | LeWM | 0.052 ± 0.149 | 0.974 | 0.004 ± 0.056 | 0.998 |
| Block Location | DINO-WM | 0.006 ± 0.007 | 0.997 | 0.002 ± 0.003 | 0.999 |
| Block Location | PLDM | 0.122 ± 0.341 | 0.938 | 0.011 ± 0.066 | 0.994 |
| Block Location | LeWM | 0.029 ± 0.073 | 0.986 | 0.001 ± 0.006 | 0.999 |
| Block Angle | DINO-WM | 0.050 ± 0.101 | 0.979 | 0.009 ± 0.052 | 0.995 |
| Block Angle | PLDM | 0.446 ± 0.625 | 0.745 | 0.056 ± 0.184 | 0.972 |
| Block Angle | LeWM | 0.187 ± 0.359 | 0.902 | 0.021 ± 0.139 | 0.990 |

Tab 3 (TwoRoom, agent position): DINO-WM Linear 0.488 ± 0.451 / r 0.824, MLP 0.000 ± 0.000 / r 0.999. PLDM Linear 0.008 ± 0.041 / 0.996, MLP 0.000/1.000. LeWM Linear 0.008 ± 0.018 / 0.996, MLP 0.000/1.000.

Tab 4 (OGBench-Cube) 중 DINO-WM 행 (Linear MSE/r ; MLP MSE/r):
- Joint Position 0.960 ± 1.150 / 0.808 ; 0.200 ± 0.967 / 0.870
- Joint Velocity 0.792 ± 0.748 / 0.763 ; 0.263 ± 0.683 / 0.852
- End-Effector Position 0.024 ± 0.010 / 0.996 ; 0.004 ± 0.003 / 0.999
- End-Effector Yaw 3.317 ± 1.016 / 0.828 ; 0.167 ± 0.168 / 0.917
- Gripper 0.114 ± 0.095 / 0.943 ; 0.038 ± 0.060 / 0.982
- Block Position 0.085 ± 0.029 / 0.991 ; 0.007 ± 0.007 / 0.998
- Block Quaternion 1.596 ± 10.457 / 0.257 ; 0.769 ± 8.046 / 0.411
- Block Yaw 4.223 ± 2.530 / 0.176 ; 0.916 ± 0.278 / 0.304
- Overall 1.162 ± 1.579 / 0.725 ; 0.290 ± 1.202 / 0.799
(LeWM/PLDM 행은 파일 Tab 4 참조. LeWM Block Yaw MLP r 0.164, End-Effector Yaw MLP r 0.213, PLDM 0.106/0.117 - 모든 모델이 회전 계열을 못 맞춤. 논문 본문도 "All three methods struggle to recover block orientation" 이라 서술.)

통계적 읽기 (내 해석, 추정 포함):
1. Tab 1 에서 DINO-WM 은 MLP probe 로 agent/block 위치와 각도에서 r >= 0.995. 즉 DINO-WM(인코더 출력)에는 위치·각도 정보가 비선형으로 거의 완전히 존재한다. 이는 H1 중 "인코더 단계에서 소실"에는 불리한 증거이나, 이 표가 "predictor의 transformer layer 통과 후" 표현을 본 것인지는 미확인 (아래 3.1). 
2. Tab 1 DINO-WM agent location: Linear MSE 1.888 인데 r = 0.977. MSE 가 큰데 r 이 높다는 것은 스케일/오프셋 불일치(정규화 방식 차이)나 outlier 때문일 수 있다 (추정). r 과 MSE 는 서로 다른 정보를 주므로 R^2(분산 설명률, 스케일 민감)를 함께 보고해야 한다.
3. "±" 가 mean 보다 큰 셀이 많음(예: DINO-WM agent MLP 0.003 ± 0.022; Block Quaternion 1.596 ± 10.457). 오차 분포가 heavy-tail 이라 평균 MSE 비교의 신뢰성이 낮다. median, 분위수, bootstrap CI 가 필요 (추정).
4. 모델 간 순위 차이(예: LeWM 0.021 vs DINO-WM 0.009 MLP Block Angle)에 대해 논문은 검정·CI 를 제시하지 않음. probe 시드 수 미확인이므로 "차이가 유의하다"는 결론은 논문에서 뒷받침되지 않는다.
5. 저자 스스로 DINO-WM 의 강한 probing 을 대규모 사전학습(약 124M 이미지) 덕으로 해석(Tab 1 캡션). 이는 해석이며 검증되지 않음.

### 1.2 DINO-WM (DINO-WM.pdf)
- 이 논문 자체에는 physical-attribute probing 이 없음 (probe/probing 키워드 미검출). 대신 (a) 학습된 decoder 로 latent 를 복원해 해석 (Sec 4.7), (b) 평가는 planning 성공률(SR)과 LPIPS/SSIM/Chamfer Distance 등 (Table, DINO CLS 베이스라인 포함).
- 구조(App): frozen DINOv2 patch features (14x14, 384). Predictor = ViT, depth 6, 16 heads, MLP dim 2048, 약 19M parameters (App 라인 933-934). Causal attention mask 사용, history h. decoder 는 predictor 학습과 독립(Sec 3.1.3).
- 함의: "transformer layer" = 6층 predictor ViT. frozen DINOv2 인코더(ViT) 내부의 레이어와 predictor 레이어를 구분해야 함 (H1 의 "transformer layer"가 어느 쪽인지 정의 필요).

### 1.3 SG-JEPA (sg-jepa.pdf)
- LeWM 류 사전학습 후 frozen 표현에 대해 MLP state probe 를 사용한 가장 상세한 probe 프로토콜 (App C.2, Sec 2.3).
- 대상: 2D = position, velocity, rotation; 3D = position, velocity (ball state x,y,z,vx,vy,vz 등).
- probe: 2-layer MLP (LayerNorm -> GELU 512 -> GELU 256 -> linear), dropout 0.05, AdamW lr 1e-3, wd 1e-5, batch 256, 최대 50 epoch, validation NMSE 최저 체크포인트. 입력은 latent window(길이 l_r) 연결. 일부 실험은 linear probe 도 사용 (temporal-window-4 linear and MLP probes, SIGReg sweep).
- 지표: NMSE(훈련 split 통계로 z-score 한 좌표별 MSE). 롤아웃 평가 지표 = "excess NMSE" = 롤아웃 latent 에 대한 probe 오차 - GT 인코딩 latent 에 대한 probe 오차 (Eq 4). probe 자체 오차를 뺀다는 점에서 H2 설계에 직접 참고가 됨.
- 통계 관행: Fig 5(b) predictor seed 3개 + pointwise 95% bootstrap CI; Fig 5(c) 2,800 paired OOD episodes 에서 95% simultaneous interval (다중비교 보정 개념에 해당). 일부 sweep 은 1 seed (App 라인 1198 "The sweep uses one seed"). 제어 실험은 5 seeds 평균.
- effective rank (Eq 9-10): 공분산 고유값 p_j 의 엔트로피 exp(H). N=2000, D=256. 예: 138.0 (lambda_SIG=0.72) vs 131.0 (1.44). epoch-5 effective rank 와 이후 linear/MLP probe NMSE 의 Spearman rho = -0.90, -1.00. (체크포인트 수가 작아 rho 의 CI 는 매우 넓음, 추정.)
- DINO-WM 비교: DINO-WM 은 frozen DINOv2 인코더 + 학습된 transformer predictor (H=20, 6 layers, 16 heads, MLP 2048, dropout 0.1). Fig 5: frozen encoder 에 fresh predictor 를 붙여 "표현 vs predictor" 를 분리하는 설계 (fresh GRU: 1.376 vs 1.555; fresh Transformer: 1.269 vs 1.453). teacher-forced 에서 SG-JEPA 가 DINO-WM 대비 far-OOD 중력에서 국소 오차 약 32% 감소 (논문 서술). 이 "encoder 고정 + predictor 교체" 설계는 H2 검증에 재사용 가능.
- 개입 실험 (Fig 상 "Panels (a),(b) report the increase in normalized full-state probe NMSE" 문맥, 라인 ~1303): 채널 교란 후 probe NMSE 증가량을 보는 방식으로 보임 (세부 미확인).

### 1.4 G-JEPA (G-JEPA.pdf)
- probing 이 아니라 "physical state grounding" 보조손실 (observed state S, nuisance invariance I, predicted state P). 평가는 planning 성공률.
- 논문값 Tab 3 (PushT-Ada / OGBench Cube 성공률 %): 기준 56.90±2.35 / 72.80±2.53; S 만 55.73±1.58 / 75.40±1.65; S+I 79.43±0.90 / 76.20±2.57; S+I+P 79.43±1.58 / 78.60±4.12.
- 통계 설정: main configuration 은 training seed 1개; planning seed 3개(PushT, TwoRooms), 10개(Cube); ± 는 planning seed 간 표본 std. 즉 ± 는 학습 시드 변동을 반영하지 않음. temporal-module 비교만 training replicate 3개 (Tab 4: S4D 78.12±3.34, GRU 69.14±14.89).
- 함의: (i) 상태 정보를 predictor 출력에 grounding(P)해 활용을 강제하는 개입이 존재 -> H2 의 "개입 후 성능 변화" 비교군. (ii) S 단독은 PushT-Ada 에서 이득 없음(55.73 vs 56.90, 1 training seed, 차이 < 1 sd) -> "상태 정보를 넣는다고 곧 활용되는 것은 아님"이라는 간접 증거 (추정; 1 training seed 이므로 약함).

### 1.5 T2-z (T2-z.pdf)
- 환경 잠재변수(friction, mass, gravity 등 비관측 물리 요인)를 test-time 에 latent 최적화로 추정. probing 은 아님. 평가: 물리 오차(pixel / lamp-score / bar-tilt degrees) 및 성공률. DINOv2 Context 등 baseline 비교 (Tab 1).
- 함의: "관측으로 식별 불가능한 물리량(mass, friction)"은 단일 프레임 probing 으로 읽을 수 없고 궤적(동역학) 필요. H1/H2 대상 attribute 를 정적(위치/각도) vs 동역학적(속도, mass, friction)으로 분리해야 함. PCA 로 2차원 구조가 나타난다는 정성 결과 (Fig 1b) - 정량 지표는 미확인.

### 1.6 5편 요약 비교
| 논문 | probe 종류 | 지표 | 시드/CI | 선택성 대조군 |
|---|---|---|---|---|
| LeWM | linear + MLP | MSE, Pearson r | 시드 수 미확인, ± 의미 불명, CI 없음 | 없음(미확인) |
| SG-JEPA | MLP (일부 linear) | NMSE, excess NMSE | predictor seed 3 + bootstrap 95% CI, 일부 1 seed | GT 인코딩 대비 차감(롤아웃) |
| G-JEPA | probing 없음 | 성공률 | planning seed 3/10, training seed 대부분 1 | 해당 없음 |
| DINO-WM | probing 없음 | SR, LPIPS 등 | - | - |
| T2-z | probing 없음 | 물리오차 | - | - |

공통 공백 (추정): 5편 어디에서도 layer-wise probing curve, control task(Hewitt & Liang), random-init 인코더 대조, MI/MDL probing, causal patching 은 확인되지 않음. DINO-WM 의 predictor 내부 레이어 probing 은 선행 수치가 없다. 따라서 H1/H2 는 선행 수치 없이 처음부터 설계해야 한다.

---------------------------------------------------------------------------
## 2. H1 / H2 를 구분하는 통계 설계

### 2.0 먼저 정의해야 할 것 (가설의 조작적 정의)
표현 지점을 명시: 
- R0 = frozen DINOv2 patch token (predictor 입력 직전)
- R_l (l=1..6) = predictor ViT 각 레이어 출력 (token 단위; 어떤 token 을 쓸지: 마지막 history frame 의 patch 전체 vs pooled)
- R_pred = 예측된 다음 latent (predictor 최종 출력)
- 비교 대상 정보량 I(R; y) , y = 물리량 (agent/block 위치, 각도, 속도 등) 및 "미래 y_{t+k}".
- H1: I(R_l; y_t) 가 l 증가에 따라 감소 (또는 R_pred 에서 y_t 복원이 R0 보다 유의하게 낮음).
- H2: I(R_l; y) 는 R0 수준으로 유지되지만, predictor 의 출력/다음 상태 예측이 y 에 의존하는 정도(인과적 민감도)가 낮다 (정보가 있어도 다음 latent 예측과 planning 에 쓰이지 않음).
- 주의: H1 과 H2 는 배타적이지 않다(둘 다 부분적으로 참일 수 있음). 이분법이 아니라 레이어별 "정보량 곡선"과 "인과 기여도"의 2축 표로 보고하는 것을 권장 (추정).

### 2.1 Probe 용량 대비 선택성 (Hewitt & Liang 2019 계열)
- 선택성(selectivity) = 실제 label 에서의 probe 정확도 - control task(label 을 입력 토큰/샘플 단위로 무작위 할당) 정확도. 회귀 물리량의 경우 label 을 같은 marginal 분포에서 샘플별로 permute 하거나, 상태 값을 해시해 무작위 함수로 바꾼 control 을 만든다 (회귀 적용은 추정 설계; 원 논문은 품사 태깅이라 분류 과제).
- probe 용량 스윕: linear -> MLP(은닉 1층, 폭 w in {64,256,1024}) 를 같이 보고, 용량이 늘 때 실제-label 성능과 control 성능이 함께 오르면(선택성 하락) "probe 가 스스로 학습한 것" 이라고 해석 (LeWM 의 MLP r=0.999 는 선택성 없이는 해석 한계).
- 필수 baseline (표현 동일 차원 기준):
  (a) random-init predictor/ViT (학습 안 된 레이어) 위에서 같은 probe -> "아키텍처만으로 읽히는" 정보,
  (b) raw 입력 baseline: 픽셀 또는 R0 의 단순 풀링(평균) 에 대한 probe,
  (c) shuffled-label probe,
  (d) 표현의 PCA 로 차원을 맞춘 probe (차원 교란 제거).
- 정규화 정책: ridge 의 lambda 는 validation 에서 선택, 레이어별로 동일 탐색 격자 사용. 차원 큰 레이어(patch 196x384 = 75,264)와 pooled 384 를 비교할 때 입력 차원이 다르면 용량이 달라 불공정 -> PCA 로 d 통일 또는 같은 d 로 pooling 한 결과를 병기.

### 2.2 레이어별 probing curve
- 각 l 에 대해 y 별 R^2 (held-out) 또는 NMSE(훈련 통계 z-score 한 SG-JEPA 식) 를 계산하고 l 축 곡선 + 95% CI. LeWM 처럼 Pearson r 만 쓰지 말 것(스케일 불변이라 오프셋·이득 오류를 숨김). R^2 와 MSE 를 모두 기록.
- 입력 지점이 token 이므로 "어느 token 의 정보인가" 를 분해: agent/block 위치에 해당하는 패치 위치의 token vs 나머지 token. 위치 정보가 token 인덱스(공간 위치)에 인코딩되어 있으면 pooled probe 는 낮고 spatial-aware probe 는 높을 수 있음 -> pooling 방식이 H1 결론을 좌우하는 교란 변수 (추정).
- 동일 데이터로 시점 t (현재 상태) 와 t+k (미래 상태) 두 가지 target 을 분리: 현재 상태 복원 = 보존(H1 축), 미래 상태 복원 = 예측에 필요한 정보/활용(H2 축).
- 판정 통계량 예: ΔR²(l) = R²(R_l) - R²(R0). 다중 레이어/속성 비교이므로 아래 2.6 의 보정 적용.

### 2.3 정보 이론적 / 구조 지표
- MI 추정: 고차원 MI 추정기(MINE, InfoNCE 계열)는 편향·분산이 커서 단독 사용 비추천(추정). 대안으로 "probe 의 held-out 로그우도 향상량" 을 정보량의 하한 대용으로 사용: 정보 하한 ≈ H(y) - CE_probe (분류) / 가우시안 가정 하 0.5·log(Var(y)/MSE) (회귀). 이는 하한이며 probe 용량에 의존한다.
- MDL/online-code probing (Voita & Titov 2020): 데이터 효율(학습곡선)까지 포함해 "얼마나 쉽게 읽히는가" 를 코드 길이로 측정. 샘플 수 대비 성능 곡선(learning curve)으로 근사 가능.
- effective rank (SG-JEPA Eq 10, exp(spectral entropy)), participation ratio = (Σλ)²/Σλ² 를 레이어별로 계산해 차원 붕괴(dimensional collapse)를 확인. 주의: rank 는 "물리 정보량" 이 아니라 전체 분산의 분포일 뿐이다. 물리 정보가 소수 방향에 있으면 rank 변화 없이 정보가 사라질 수 있고 그 반대도 가능하다. 따라서 rank 는 보조지표.
- 물리량 정렬 방향 분석: y 와 공선인 부분공간의 분산 비율 (CCA 또는 PLS 의 canonical correlation, partial R²). 레이어 통과에 따른 이 부분공간 에너지의 감쇠가 "희석" 의 직접 측정이 된다. 단 선형 부분공간만 포착.
- 표현 유사도 (CKA) 는 레이어 간 보존도 참고용; 물리량 별 보존 증거로는 부족.
- SG-JEPA 의 excess NMSE 아이디어 차용: 레이어 l 에서의 probe 오차 - R0 에서의 probe 오차 로 "probe 자체 오차를 제거한 레이어별 소실량" 을 정의.

### 2.4 효과 크기와 신뢰구간
- 보고 항목: ΔR² (레이어/모델 간 차이), 표준화 효과크기 (paired Cohen's d 또는 Hedges' g; 샘플이 에피소드 단위로 쌍을 이룸), 비율 (MSE_l / MSE_R0).
- 샘플 독립성 문제: 같은 에피소드 내 연속 프레임은 상관이 높다. 평가 단위(resampling unit)는 에피소드이며, 프레임 단위 bootstrap 은 CI 를 과소추정한다. 에피소드 단위 cluster bootstrap (B >= 2,000) 사용. train/val/test 분할도 에피소드 단위여야 누수 방지.
- Heavy-tail (LeWM 표에서 sd > mean 다수) -> MSE 평균 대신 median, trimmed mean 병행, log-MSE 로 CI.
- 시드 변동 분리: (i) probe 초기화 시드, (ii) probe 학습/평가 분할 시드, (iii) world model 학습 시드 (DINO-WM predictor 재학습) 를 구분. 모델 간 "차이" 주장에는 (iii) 가 중요하다. G-JEPA 처럼 planning seed 만 반복하면 모델 학습 분산이 빠진다.

### 2.5 시드 수와 검정력
- 두 레이어 또는 두 모델 간 지표 차이를 paired 비교할 때, 시드 간 sd 를 σ, 검출하려는 최소 효과를 δ 라 하면 대략 n ≈ 2·(z_{1-α/2}+z_{1-β})²·(σ/δ)² (독립 두 표본 정규근사; paired 면 σ 를 차이의 sd 로). 예: 효과크기 d=1.5, α=0.05, power 0.8 이면 대략 n≈8 /군 (정규근사, t 보정하면 다소 더 많음). d=1.0 이면 n≈16. 즉 학습 시드 3개로는 d≈2.5 이상(큰 효과)만 검출 가능 (추정; 계산은 위 공식의 근사). 본 조사에서 코드로 검증하지는 않았음.
- 실용 제안 (추정): 모델 학습 시드는 최소 5 (가능하면 8-10), probe 시드는 레이어×속성당 5 이상, 에피소드 단위 bootstrap 병행. 시드가 3 이하이면 CI 가 너무 넓어 "차이 없음"과 "차이 있음" 모두 구분 불가이므로 결론을 약하게 쓴다.
- 등가성(equivalence) 검정: H2 의 "보존됨" 은 귀무가설 "차이 없음" 의 채택이므로, "유의하지 않음" 을 "같다" 로 해석하면 안 된다. TOST 로 사전 정의한 허용 한계 ε (예: ΔR² 가 ±0.05 이내) 를 증명하거나, 90% CI 가 [-ε, +ε] 안에 들어오는지로 판정.
- 베이지안 대안: ΔR² 의 사후분포에서 P(ΔR² > -ε) 보고 (모델: 시드·에피소드 계층 효과).

### 2.6 다중비교 보정
- 비교 수 = 속성 수 × 레이어 수(7: R0 + 6) × probe 종류 (× 모델). 수십~수백 개 -> 보정 필수.
- 가족(family)을 사전에 정의: 예) 속성별로 "레이어 곡선의 단조 감소" 하나의 가설로 묶기(Jonckheere/Page 추세 검정 또는 레이어를 설명변수로 한 혼합효과 회귀의 기울기 1개). 이렇게 하면 비교 수를 줄여 검정력 보존.
- 개별 비교가 필요하면 Holm-Bonferroni(FWER) 또는 Benjamini-Hochberg(FDR, 탐색적). SG-JEPA 의 "simultaneous 95% interval" 처럼 동시 신뢰구간 방식도 가능.
- 가설-성공기준을 데이터를 보기 전에 고정(사전등록 성격)하고 레이어/속성 선택에서 post-hoc 선택(forking paths) 을 금지.

### 2.7 Probing 은 '활용' 을 의미하지 않는다 (한계)
- probe 가 읽을 수 있다 != 모델이 그 정보를 사용한다 (Hewitt & Liang 2019; Belinkov 2022 survey; Ravichander et al. 2021 "Probing the probing paradigm"; Elazar et al. 2021 amnesic probing 등 기존 문헌의 일반 주장. 이 폴더 5편에는 없음 - 인용 시 원문 확인 필요).
- 반대로 읽히지 않는다 != 없다 (probe 용량 부족, 최적화 실패). 그래서 linear/MLP 동시 + 용량 스윕 필요.
- LeWM 의 MLP r≈0.999 는 입력이 이미 충분히 풍부하면 임의 정보도 복원 가능한 용량 때문일 수 있음. DINO-WM 은 patch 수만 개 차원이라 특히 해당 (추정). 선택성 대조군 필수.
- 상관 교란: PushT 에서 agent 위치와 block 위치/각도, 시간 인덱스, action 이 통계적으로 상관. 물리 attribute 와 상관된 시각 특징 (색, 이미지 외형)을 probe 가 읽는 것일 수 있음 -> 편상관(partial) / 다른 변수로 직교화한 target / 데이터 분포를 강제로 교란 해제한 평가 세트 (OOD 물리값) 필요. SG-JEPA 가 OOD 중력값을 쓰는 점이 참고가 됨.
- 정적 vs 동역학: 위치·각도는 단일 프레임에 있으나 속도·질량·마찰은 시계열 필요 (T2-z). probing 입력 윈도우를 H frames 로 맞추고 target 별로 적절한 입력을 정의.

### 2.8 활용 측정: 개입(ablation/patching)
H2 는 상관 분석(probing) 만으로 판정 불가. 개입 필요.
1. 방향 제거(INLP/LEACE 계열 선형 소거; Ravfogel 2020, Belrose 2023 - 일반 문헌, 이 폴더에 없음): R_l 에서 y 에 해당하는 부분공간을 투영 제거 후 predictor 후속 레이어의 다음 latent 예측 오차 증가량 측정. 비교군: 같은 차원의 무작위 부분공간 제거(동일 분산 에너지 제거). 효과 = ΔPredErr(y-subspace) - ΔPredErr(random subspace).
2. Activation patching (인과 매개): 에피소드 A 와 B 가 y 만 다르도록(또는 가장 유사하도록) 쌍을 만들고, A 의 R_l 에 B 의 y-관련 성분을 넣어 출력(예측 next latent 및 그 probe 값)이 B 쪽으로 이동하는 비율(=indirect effect, normalized: (O_patched - O_A)/(O_B - O_A)) 측정. 시뮬레이터에서 y 만 바꾼 반사실 쌍 생성 가능하면 가장 깨끗함 (추정: 이 환경이 허용하는지는 미확인).
3. 입력 단계 개입(counterfactual input): 특정 물리 단서만 마스킹/교체한 입력(예: block patch 위치 교란) -> 예측 변화의 y 민감도 (SG-JEPA 의 채널 교란 후 NMSE 증가 방식과 유사).
4. Probe-guided 개입: 선형 probe 방향으로 R_l 을 이동(steering)해 다음 latent probe 값이 비례 이동하는지(회귀 기울기, 1에 가까울수록 predictor 가 해당 방향을 그대로 전달) 측정. 
5. Fresh predictor 대조 (SG-JEPA Fig 5 설계): 같은 frozen 인코더 위에 predictor 를 새로 학습했을 때 기존 predictor 보다 물리 예측이 좋아지면 "정보는 있었으나 기존 predictor 가 활용 못함" (H2) 지지. 같다면 predictor 학습 문제가 아님. 비교는 predictor 시드 5 이상.
6. Auxiliary-supervision 검증 (G-JEPA 의 S/P 손실 유사): predictor 에 물리 상태 보조 head/손실을 추가했을 때 planning/예측이 개선되면 predictor 가 정보를 쓰지 못하던 것(H2)의 간접 증거. 개선 없으면 한계가 다른 곳(H1 또는 목표 설정).
- 개입 효과 보고: 평균 효과 + 에피소드 cluster bootstrap CI + 무작위 부분공간 분포 대비 퍼센타일(경험적 p).
- 분리 논리 (H1 vs H2 결정 표, 추정):
  | 관측 | 해석 |
  |---|---|
  | R0 에서 y 복원 높음, R_l 로 갈수록 단조 감소, R_pred 낮음 | H1 지지 (희석) |
  | 전 레이어 y 복원 유지 (등가성 통과), 그러나 y-부분공간 제거해도 예측오차 변화 ≈ 무작위 제거 | H2 지지 (보존되나 비활용) |
  | R0 에서도 y 복원 낮음 (선택성 포함) | 인코더 단계 부족: H1 의 변형이라기보다 전제 불성립(둘 다 아님) |
  | 정보 유지 + 제거 시 예측오차 크게 증가 | 둘 다 기각(보존·활용) |

### 2.9 필요한 최소 데이터 양
- 선형 probe 의 표본 복잡도는 입력 차원 d 에 의존. 목표: n_train >= 10·d 이상 또는 ridge+validation 으로 learning curve 가 포화됨을 확인 (경험칙, 추정). 입력이 pooled 384 차원이면 n_train 약 4k 프레임 이상, patch 전체(75,264)를 쓰면 PCA/pooling 으로 차원 축소가 필수.
- 정확한 수치는 pilot learning curve (n = 0.1, 0.25, 0.5, 1.0 × 데이터) 로 R² 가 포화되는 지점을 찾는 방식을 권장. 포화되지 않으면 데이터 부족으로 인한 과소평가 가능성.
- 평가 단위는 에피소드: 시드 반복 × 에피소드 수 (test >= 수백 에피소드 권장, 추정). SG-JEPA 는 5,000 held-out episode, 2,800 paired OOD episode, 2,000 episode 로 effective rank 계산을 사용 (논문 서술). 이는 참고 규모일 뿐 필요 규모 보증은 아님.
- 개입(patching) 은 쌍 샘플이 필요하므로 속성별 수백~수천 쌍 (추정), 반사실 쌍을 시뮬레이터에서 직접 생성하는 것이 데이터 효율적.
- 분포 설계: 물리량이 서로 독립이 되도록 샘플링(위치·각도 상관 제거), 극단/OOD 값 포함, train/test 에서 에피소드·초기조건 분리.

---------------------------------------------------------------------------
## 3. 성공 기준(정량) 후보 (확정 아님)

기호: S_l(y) = 레이어 l 에서 속성 y 의 selectivity-보정 held-out R² (control 차감 후), 쌍 비교는 에피소드 cluster bootstrap 95% CI. ε, 임계값은 사전에 사람이 확정해야 하는 자유 파라미터이며 아래 숫자는 예시(추정).

공통 전제 기준 (게이트):
- G0: 선택성 게이트. 실제 label R² - control R² >= 0.1 (또는 control R² <= 0.1) 이어야 해당 (y, l) 을 "정보 있음" 으로 인정.
- G1: 시드 게이트. world model 학습 시드 >= 5, 에피소드 단위 CI 보고, 주요 결과는 시드 간 방향 일관성(>= 80% 시드에서 동일 부호).

H1 (희석/소실) 지지 기준 후보:
- H1-a: ΔR²(R_pred 또는 R_6 vs R0) 의 95% CI 상한 < -ε1 (예: ε1 = 0.10), 속성 family 에서 Holm 보정 후 유의.
- H1-b: 레이어 l 에 대한 S_l(y) 의 혼합효과 기울기 < 0 (유의, 단조 추세 검정), 효과크기 d >= 0.8.
- H1-c: y-정렬 부분공간 에너지 비율(CCA/partial R²) 이 R0 대비 50% 이상 감소.
- H1 기각: ΔR² 의 90% CI 가 [-ε1, +ε1] 에 포함 (TOST 통과).

H2 (보존되나 비활용) 지지 기준 후보:
- 보존 조건: 위 H1 기각 (TOST) 이 속성 다수에서 성립.
- 비활용 조건 중 하나 이상: (i) y-부분공간 제거의 예측오차 증가가 무작위 부분공간 제거와 차이 없음 (등가성 ±ε2 내, 또는 경험적 p > 0.05), (ii) patching 정규화 indirect effect < 0.1 (예시), (iii) steering 기울기 < 0.2 (예시), (iv) fresh predictor 또는 auxiliary supervision 후 해당 속성의 teacher-forced 예측 NMSE 가 사전 정한 상대 비율(예: 10-20%) 이상 개선되고 CI 가 0 제외.
- 활용 확인(H2 기각) 후보: indirect effect >= 0.5 이면서 무작위 대비 유의.

보고 형식 후보: 속성 × 레이어 히트맵 (S_l, CI), 레이어 곡선, 개입 효과 막대(무작위 부분공간 분포 대비), 위 결정 표(2.8). 유의하지 않은 결과는 "유의하지 않음/결정 불가" 로 명시, 시드 3 이하 결과는 탐색적으로 표시.

사전 파일럿 제안(사람 승인 전 단계, 실행하지 않음): (1) 선행 수치 재현성 점검 = LeWM Tab 1 DINO-WM 행이 어떤 표현에 대한 probe 인지 저자 코드/부록으로 확인, (2) learning curve 로 n 결정, (3) 시드 분산 사전 측정으로 검정력 계산.

---------------------------------------------------------------------------
## 4. 주의 / 논문 해석상 위험 요약
1. LeWM Tab 1/3/4 의 ± 정의 불명, CI·검정·시드 수 없음 -> 모델 간 순위는 탐색적 수준. (미확인 항목은 PDF 부록 정독 필요.)
2. LeWM 의 DINO-WM 행이 predictor 통과 전(인코더 출력)인지 후인지 불명. H1 의 "transformer layer 통과" 와 직접 연결되지 않음. 이 점이 확인되지 않으면 기존 수치를 H1 근거로 쓰면 안 된다.
3. DINO-WM 은 LeWM 과 달리 patch token 표현이므로 차원·pooling 선택이 probe 성능을 좌우 (교란).
4. 회전 계열 attribute (quaternion, yaw) 는 모든 모델에서 r<0.5 (LeWM Tab 4 중 PLDM/LeWM; DINO-WM Block Yaw MLP r 0.304, Quaternion 0.411). 이런 경우 "소실" 이 아니라 인코더 단계 부재일 가능성.
5. Heavy-tail 오차 분포 -> 평균 MSE 단독 비교 금지.
6. 외부 문헌(Hewitt & Liang 2019, Voita & Titov 2020, Ravfogel 2020, Elazar 2021, Belrose 2023, Belinkov 2022 등)은 내 기존 지식 기반이며 이 폴더 5편에 없다. 인용 전 원문 확인 필요(추정 표시).

## 5. 계산 코드 위치
- 이 문서의 모든 논문값은 pdftotext -layout 결과(scratchpad: /tmp/claude-1000/-home-jaekyungkim-mylab/5e599cd6-6cf8-4bff-a06a-193aff48bde4/scratchpad/t/*.txt)에서 직접 옮김. 별도 재계산 코드 없음 (실험 결과 아님).
- 표본수 공식(2.5)은 정규근사 손계산이며 코드로 검증하지 않음 (추정).

## 6. 확신도: 중간-낮음
- 높은 부분: 5편의 probe 설정(LeWM: linear+MLP, MSE+r; SG-JEPA: MLP, NMSE) 및 표 수치 전사(텍스트 추출 기준).
- 낮은 부분: LeWM 의 샘플 수·시드·± 의미·DINO-WM 표현 지점(부록 정독 미완), 외부 문헌 인용(기억 기반), 검정력 수치(근사), 성공 기준 임계값(전부 예시이며 사람 확정 필요).
