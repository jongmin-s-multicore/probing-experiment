# 008 사후 재집계 (004 depth_r2.csv)

- 검증할 가설: 없음(가설 확정 아님). notes/hypotheses.md "008 사후 재집계"의 사전 기준 A, B, C 를 004 결과에 적용. 사후 탐색이며 004 판정(H-d refuted)을 바꾸지 않는다.
- 방법 (CPU, numpy, 새 학습/GPU 없음; 입력은 읽기 전용):
  - 입력: `../004_depth_truncation/results/depth_r2.csv`, `summary.json`, `config.yaml`, `../002_single_size_sweep/config.yaml`(area_ratio).
  - 사전 확인: encoder, predictor 각각에 random_init 대응 행이 있고 키(depth, split, probe seed, train seed)가 모두 일치함(results.json `precheck_random_init_rows`).
  - A: Δ(d)=R²_real-R²_random_init (linear/new/late/ALL, split 별 train/interp/extrap), 0.2 하락 규칙. 보조: 학습 시드 단위 평균-2sd>0. encoder 는 학습 시드가 없어 probe seed 5개 sd.
  - B: MAE(CSV), MSE 와 공통분산 R². 레벨 행의 r2 는 이미 `1-MSE_level/Var_all` 로 저장돼 있으므로 MSE_level=(1-r2)·Var_all. Var_all 은 002 config area_ratio 의 log(레이블=에피소드 상수)와 레벨별 n_episodes 로 계산. 검증: 이렇게 복원한 split R²가 CSV 의 ALL 행 R²와 1e-6 이내로 일치해야 계산(실패 시 "계산 불가"). 최대 차이는 results.json `selfcheck_max_abs_diff_split_r2_vs_csv_ALL_rows`.
  - C: 겹침 검정을 학습 시드 3개의 (probe seed 평균) 사이 sd 단위로 재계산(2.0·sd, 깊이 비율 0.8). encoder 는 학습 시드 단위가 없어 "계산 불가".
- 성공 기준: 없음. 각 항목을 "관측됨 / 관측 안 됨 / 계산 불가"로 기록.
- 이 실험으로 내가 주장하려는 것: 위 세 재집계를 사전 기준대로 계산해 수치를 results.json 에 남겼다는 사실뿐이다. 해석은 하지 않는다(/review-experiment 대기).
- 실행: `~/miniconda3/envs/lab07/bin/python run.py` (수 초).
- config 값: 시드 0(결정적), 임계값 0.2, overlap 2.0/0.8 은 004 config 값을 그대로 쓰며 run.py 가 일치를 assert 한다.
- 지위: 사후 탐색으로만 남긴다. 004 H-d refuted 의 보강 근거로 인용하지 않는다(사용자 결정 2026-11-10, reports/008_posthoc_reaggregate_004_005.md).
