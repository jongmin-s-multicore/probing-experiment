# 실험 설명: 013_size_use_erasure

상태: 가설·판정 기준 확정(2026-11-11), **코드 작성 완료(2026-11-11, CPU selftest 통과), GPU 실행 미승인, 미실행**. 결과 수치 없음. config.yaml 값은 모두 사용자 확정값이며 코드 작성 중 바꾸지 않았다(`pilot: true` 유지).

- 검증할 가설: notes/hypotheses.md "### 013 크기 정보의 사용: predictor 입력 소거가 움직임 예측을 망치는가" 절과 그 아래 확정 기록 2개(2026-11-11).
  - 질문: DINO-WM predictor 는 크기 정보를 크기에 따라 다른 동역학(움직임)을 예측하는 데 쓰는가.
  - 설계 전제: 002 는 `num_hist: 1` 이라 predictor 는 한 프레임만 본다. 크기를 지워 오차가 늘면 겉모습 복사 실패일 수 있으므로, 판정은 움직임 오차로 한다.
- 방법
  - 0단계(관문, CPU): 학습 레벨(a060, ind, a160)의 실제 인코더 latent(002 Q1 캐시 `tokens_dino`)로, 레벨별 ridge 회귀기(패치 평균 z_t 384 + 행동 a_t 30 -> 패치 평균 Δz 384, alpha 는 학습 episode 안 episode 단위 3-fold, 002 `probe_impl.ridge_alphas`)를 학습한다(학습 split 300 episode x 12 step). 6개 방향 쌍 (A, B)마다 비 = MSE(A 회귀기, B test) / MSE(A 회귀기, A test). 비 ≥ 1.1 인 쌍이 4개 이상이면 통과. `results/stage0.json` 에 `passed` 와 전체 표를 저장한다. 진단용으로 MSE(A 회귀기, B test) / MSE(B 회귀기, B test) 도 같이 저장한다(관문에는 쓰지 않음).
  - 1단계(주, GPU): 조건 (a) `single_ind`, (b) `multi_60_ind_160` 의 학습 seed 0,1,2 `model_latest` 체크포인트와 random-init predictor(`DP.random_init_predictor(model, 1234 + 학습 seed)`, 인코더·행동/proprio 인코더는 체크포인트 것). test 9 레벨 x 100 episode, episode 당 12 전이(프레임 f -> f+1, f = 0..11)에서 한 step teacher-forced 예측(입력 = 실제 인코더 latent).
    - 소거: predictor 입력 z 부분(앞 384, 196 패치 전부)에 변위 δ 를 더한다. 행동/proprio 좌표는 그대로. 위치 임베딩 더하기 전. 방식 2종: `leace_scalar`(009 `fit_eraser`, fit_fraction 0.5, fit_seed 9000 + 학습 seed, ridge_rel 0.001, 적합은 Q1 캐시의 train 행 절반, 12 입력 프레임의 패치 평균), `level_mean_match`(패치 위치별로 각 레벨 평균을 9개 레벨 평균의 평균으로 이동; 학습 레벨은 train split 300 episode 로, OoD 6개 레벨은 test 100 episode 의 무작위 절반으로 적합하고 나머지 절반으로 평가, 분할은 `results/fit.json` 에 기록).
    - 대조: 무작위 단위 방향 20개(seed 13000), 토큰별 변위 norm 을 크기 소거와 정확히 같게(LEACE: 같은 부호 있는 계수 x |c| 를 무작위 방향으로, level_mean_match: |D[레벨, 패치]| 를 무작위 방향·무작위 부호로).
    - 지표(episode 당 12 전이 평균, 196 패치 x 384 차원 MSE): `total` = 예측 vs 다음 인코더 latent; `motion` = ((예측 − 소거된 입력) − (z_{t+1} − z_t))², 즉 예측된 변화량 vs 실제 Δz; `persist` = |Δz|² (Δz = 0 예측). 주 지표 = Σ motion / Σ persist (persistence 정규화). 소거가 없으면 total == motion. `copy`(소거된 입력 복사 오차)와 변위 norm 도 저장.
    - Δ_use = (크기 소거로 늘어난 정규화 움직임 오차) − (무작위 20개 평균으로 늘어난 값) = size − mean(rand). 불확실성: episode bootstrap 2000회(seed 14000, 레벨별 층화 재표집), 90% CI.
    - 판정(소거 방식마다, 학습된 predictor, 판정 split = 학습 레벨 a060/ind/a160 의 test episode): (1) 조건 (b) 에서 Δ_use > 0 이고 CI 하한 > 0 인 학습 seed 가 3개 중 2개 이상, (2) (b) 의 seed 평균 Δ_use ≥ 0.10 x seed 평균 gap(gap = 1 − 소거 없는 모델의 정규화 움직임 오차 = persistence 오차 − 모델 오차, persistence 오차를 1 로 정규화), (3) (b) 의 seed 평균 Δ_use > (a) 의 seed 평균 Δ_use. 셋 다 맞으면 `criterion_met`("이 소거에서 크기 정보 사용 검출"), 아니면 `criterion_not_met`(사용하지 않는다는 뜻이 아님). 입력이 빠지거나 NaN 이면 `not_evaluable`. 같은 계산을 interp/extrap 에도 보고하지만 판정은 train 레벨만. random-init 의 Δ_use 는 기준선으로 보고.
    - 소거 확인(`readout` 단계): 소거 후 predictor 블록 0..6 의 z 부분 선형 크기 판독 R²(009 probe, `linear_pooled`, late 창, split 별), 소거 없음/LEACE/level_mean_match 3가지.
    - 2단계(보조, 판정 없음): 소거 없는 모델의 레벨 평균 예측 Δẑ 와 레벨 평균 실제 Δz 를 9개 레벨 전체에서 전역 평균 제거 후 상관·기울기.
  - 3단계(반사실 짝 교체): 보류(데이터에 짝 episode 없음, 시뮬레이터 없음).
- 성공 기준(확정 2026-11-11): 위 판정과 0단계 관문(비 ≥ 1.1 이 6쌍 중 4개 이상). 관문 미통과 시 1단계는 실행하지 않고 보고한다.
- 이 실험으로 주장하려는 것: 위 판정과 results.json 의 수치. "사용"은 "이 소거로 크기 정보를 줄이면 움직임 예측이 무작위 소거보다 더 나빠진다"는 좁은 의미로만 쓴다. 검출되지 않아도 "사용하지 않는다"가 아니다(소거 불완전 가능성; `erasure_check_readout_r2` 와 함께 읽는다).

## 실행 방법
코드: `run.py` + `src/exp013/{base,lib,core,stages,selftest}.py`. 009 `exp009.erase/stages/curves`, 004 `exp004.deep/stages`, 002 `exp002.*` 는 import 만 한다(수정 없음).
```
conda activate <002 와 같은 환경>     # CPU 확인은 ~/miniconda3/envs/lab07/bin/python
python run.py selftest                # CPU, 합성 데이터 + toy 모델 (GPU 불필요)
python run.py check                   # config 일치, 체크포인트 폴더, Q1 캐시, metadata.pt, gate/fit 상태
python run.py gate                    # 0단계 (CPU, Q1 캐시만 필요) -> results/stage0.json
python run.py fit                     # 소거기 적합 (CPU) -> $EXP013_CACHE/fit/, results/fit.json
python run.py measure                 # 1단계 (GPU). 관문 미통과/없음/낡음이면 종료코드 5 로 거부 (시도 소비 없음)
python run.py readout                 # 소거 확인 (GPU). 같은 관문 규칙
python run.py aggregate               # CPU: Δ_use, CI, 판정, 2단계, results/results.json
```
모든 단계가 `--dry-run` 을 지원한다. 옵션: `--results-dir`(파일럿 폴더, results.json 은 그곳에만 쓰고 `pilot: true`), `--conditions`, `--train-seeds`, `--episodes-per-level N`(앞 N episode 만 평가), `--override-gate`(관문 미통과 실행, 기록되고 `pilot: true`), `--batch-episodes`(기본 8), `--fresh`. 환경변수: `DINO_TRAIN`(체크포인트, 읽기 전용), `EXP002_CACHE`(Q1 캐시, 읽기 전용), `EXP013_CACHE`(소거기 적합·임시 특징), `ACWM_PHYS_SRC`.
종료코드: 0 정상, 1 실패(시도 1회 소비, `results/attempts.json`, 같은 단계 연속 3회 실패하면 거부), 3 config null, 4 사전 점검 실패(시도 소비 없음), 5 관문 거부(시도 소비 없음).
`pilot` 규칙(012 와 같음): results.json 의 `pilot` 은 config `pilot: true`, 별도 결과 폴더, CLI 덮어쓰기(조건/seed/episode 수/관문)에서 계산되고 이유는 `pilot_reasons`. 본 실행 전에 사람이 config `pilot: false` 로 바꾼다.
Colab: `colab/exp013_colab.ipynb`(`python colab/build_notebook.py` 로 재생성). git clone(REPO_URL/REPO_REF), 별도 폴더(`/content/mylab013` 등, `MyDrive/exp013_work`)라 012 와 다른 세션에서 동시에 돌릴 수 있다. 승인 플래그 `APPROVED_PILOT_013`, `APPROVED_MAIN_013`(기본 False), config `pilot: true` 이면 `ALLOW_MAIN_WITH_PILOT_TRUE` 없이는 본 실행 거부. 0단계·적합 셀(E0)은 CPU 전용이라 승인 플래그가 없다.

## 비용 추정 (추정, 실측 없음)
- 0단계 + 적합: CPU, Q1 캐시(약 3.5 GB 토큰) 읽기 포함 수 분~10분 (추정).
- measure: 단위 6개(2 조건 x 3 seed) x 900 episode x 12 전이 x 86 순전파(소거 설정 43 x 모델 2). predictor 는 깊이 6, dim 404, heads 16, mlp 2048(`conf/predictor/vit.yaml`)이라 샘플당 약 9 GFLOP(추정, 매개변수 수로 계산) -> 단위당 약 8 PFLOP, 합계 약 50 PFLOP. L4 fp32 유효 8~12 TFLOPs 를 가정하면 약 1~1.5시간 (범위 0.7~2.5시간, 추정).
- readout: 단위 6 x 3 방식 x (1800 episode 인코딩 + 7 깊이 특징 저장 약 15 GB + 선형 probe 7개). 009 intervene 3006 s(84 probe 호출, `009 timing`) 기준 호출당 약 25~35 s 로 가정하면 약 1.5~2시간 (추정). `--train-seeds 0` 만 하면 3분의 1.
- 합계 약 2.5~3.5 GPU 시간 (추정). 파일럿(조건 (b), seed 0, 레벨당 10 episode)은 수 분 (추정). 본 실행 전 승인 필요(CLAUDE.md).

## 알려진 한계
- 소거는 predictor 입력 한 곳(z 부분)에서만 한다. 크기가 행동 좌표나 proprio 로 전달되는 경로는 건드리지 않는다. 입력 소거 후 다음 블록에서 크기가 복원될 수 있으므로(009 의 회복) 소거 확인 R²(블록 1..6)와 함께 읽어야 한다.
- 스칼라 LEACE 는 크기 레이블과의 선형 공분산(패치 평균 기준)만 지운다. 토큰별 비선형·공간적 크기 단서(천이를 덮는 패치 수)는 남을 수 있다. level_mean_match 는 레벨 평균만 맞추므로 분산·모양 정보는 남는다. 두 방식 모두 "검출 안 됨"은 "사용 안 함"이 아니다.
- 무작위 방향 대조는 등방 가우시안 단위 방향이라 데이터 분산이 큰 방향과는 다르다. 같은 norm 의 교란이라도 모델이 민감한 방향이 아닐 수 있다(대조가 약해질 수 있음).
- 판정은 train 레벨 test 300 episode 위의 bootstrap CI 이고, 학습 seed 3개는 별도 불확실성 축이다(seed 간 일관성만 조건 (1)로 요구).
- 움직임 오차는 "예측 − 소거된 입력"을 예측된 변화량으로 정의했다(입력 복사형 겉모습 오류를 분리하려는 선택, CHANGES.md 2). 총 오차(`total`)도 함께 저장한다.
- level_mean_match 는 레벨 라벨을 알고 소거한다(개입 실험). OoD 레벨 평가 episode 수는 50개/레벨(절반)이다. LEACE 는 OoD 100 episode 전부와 같은 절반(`*_heldhalf`) 둘 다 저장.
- random-init predictor 는 인코더·행동/proprio 인코더를 체크포인트 것으로 쓰고 predictor 만 새로 초기화한다(004 와 같음). 점수가 매우 커서(정규화 값이 1 을 크게 넘음) 기준선 Δ_use 의 크기는 학습된 모델과 직접 비교하기 어렵다(추정).
- 실제 predictor 는 CPU 에서 실행할 수 없어(`Attention.bias.to('cuda')`) CPU selftest 는 toy 모델과 합성 데이터만 쓴다. 실제 체크포인트·DINOv2 인코더·Colab 환경은 미검증.
- 소거 확인 판독기 학습에는 004/009 와 같이 학습 split 전체 episode 를 쓴다(LEACE 적합 episode 포함). level_mean_match 의 OoD 레벨 test episode 는 적합 절반이 판독 평가에도 포함된다(소거가 더 강하게 보일 수 있음).
