# 010 원천 분리 대조: 005 의 ours ~ random 의 원인 위치 (코드 작성만, 미실행, 승인 대기)

상태: 가설 확정 전 초안(notes/hypotheses.md "010"). **GPU 실행은 사용자 승인 후에만.** "초안, 승인 대기" 항목은 config.yaml 에도 표시.

- 검증할 질문: 005 에서 학습된 predictor 와 random-init predictor 가 비슷한 것이 인코더 표현 때문인가, predictor 때문인가.
- 방법
  1. 005 의 head(`exp005.heads`), 과제(`exp005.tasks`, `build_tasks`), 지표(`compute_metrics`, 정규화 {R: iid, D: const})를 수정 없이 import 한다. 005 config 는 읽기만 하고 `eval.n_inputs` 만 900 으로 덮어쓴다(005 실행과 같음; 005 CHANGES.md 8절).
  2. 네 표현: `enc_k12` ours = 학습된 인코더 최종 층(004 캐시 `tokens_k12`), `enc_k12` random = random-init 인코더(`rand_k12`), `pred_out6` ours / random(005 feat 의 `feat_pred_out6.npy`, `feat_pred_out6_rand.npy`). 하위 실험은 5-A, 5-C 만(R-IB, D-IB, D-IB_area=, D-IB_area≠, R-IB_fold).
  3. 과제 시드: `seeds.task_a` = [2000, 2001, 2002] (초안, 승인 대기; 첫 값은 005 의 `seeds.tasks_a`). 과제 시드마다 5-A 의 레벨 라벨과 D_k 가 바뀌고 평가 입력(`seeds.eval`)은 고정. head 시드는 005 와 같은 5개.
  4. H-f TOST: 표현(layer/model)마다 d_t = score(D-IB_area=) - score(D-IB_area≠) (pair group `all`)를 과제 시드 t 별로(head 시드 평균) 구하고, 단위 수 3 이상일 때 ±Δ 에 대한 TOST (Δ = 0.05, 초안, 승인 대기; α = 0.05 단측, t 임계값 표 사용). 90% 신뢰구간이 (-Δ, Δ) 안이면 `supported`, 전부 밖이면 `refuted`, 그 외 `inconclusive`, 단위 3 미만이면 `not_evaluable`.
  5. 표현 간 차이: 네 표현 6쌍의 R-IB, D-IB 점수 차이(과제 시드 단위 평균 차이, sd, 같은 TOST)를 pair group all, ood_any 에서 보고. 표현별 점수 표에 과제 시드 간 sd, head 시드 내 sd, 전체 sd 를 모두 기록.
- 성공 기준: 사전 기준 없음. 위 수치와 TOST 상태를 results.json 에 기록한다(해석은 /review-experiment).
- 이 실험으로 내가 주장하려는 것: (실행 후) 네 표현의 5-A/5-C 점수와 과제 시드 변동을 포함한 차이의 수치. 해석은 하지 않는다.

## 실행 방법
```
cd experiments/010_source_separation_ib
conda activate <002 와 같은 환경>        # 이 머신 CPU 확인은 ~/miniconda3/envs/lab07/bin/python
export DINO_WORK=... DINO_TRAIN=... EXP002_CACHE=... ACWM_PHYS_SRC=... EXP004_CACHE=... EXP005_CACHE=...
python run.py selftest
python run.py check                       # 네 특징 파일과 002 캐시 존재 확인
python run.py heads --dry-run
python run.py heads [--task-seeds 2000,2001,2002] [--seeds 0,1,2,3,4] [--max-tasks N]   # 재개: 예측 npz 가 있으면 건너뜀
python run.py report                      # results.json, ib_scores.csv
```
`--max-tasks N` 으로 만든 예측은 과제 수가 적으므로 본 결과와 섞지 말고 `--results-dir` 을 따로 쓴다.

## 필요한 체크포인트/캐시 (읽기 전용)
- 004 enc 캐시: `$EXP004_CACHE/enc/tokens_k12.npy`, `rand_k12.npy` (004 `enc_cache`; k_grid 에 12 포함, random_init_control 켜짐).
- 005 feat 캐시: `$EXP005_CACHE/feat_pred_out6.npy`, `feat_pred_out6_rand.npy` (005 `feat`; 학습 seed 0, random_init_seed 1234).
- 002 Q1 캐시 (`EXP002_CACHE`). 체크포인트는 필요 없다(특징 파일이 있으면).
- 위 캐시는 Colab 세션 로컬에 있어 세션이 끊기면 사라질 수 있다(004 README 6절). 없으면 004 `enc_cache`, 005 `feat` 를 다시 실행해야 한다.

## 예상 GPU 시간 (추정: 실측 없음)
- 근거: 005 Colab 실측(`005/results/cost.json`, L4): 한 모델·head 시드당 5-A 100 + 5-B 100 + 5-D 4 head 약 296 s(약 1.45 s/head), 면적 head(base/area) 시드당 약 240 s, oracle linear/MLP 약 67/87 s.
- 이 실험은 5-A 만: 표현 4개 x 과제 시드 3 x head 시드 5 = 60 묶음 x 약 145 s = 약 2.4시간, 기준 모델(면적/oracle) 과제 시드 3 x head 시드 5 = 15 묶음 x 약 200 s = 약 0.8시간 -> 합계 약 3~3.5시간 (추정, L4 기준). 이때 head 수는 표현 6000 + 면적 1500 + oracle 소형 3000 (`heads --dry-run` 이 출력).
- 줄이려면 head 시드 3개 또는 과제 수를 줄일 수 있으나 005 와의 비교가 약해진다(사용자 결정).
- 디스크/메모리: 특징 파일은 읽기 전용 memmap, 예측 npz 는 작다.

## 알려진 한계
- 과제 시드는 5-A 의 라벨/D_k 만 바꾼다. 평가 입력은 고정이다.
- TOST 의 단위 수가 과제 시드 수(3)라서 자유도 2, t 임계값 2.920 이 넓다.
- 코드는 합성 데이터 selftest 에서만 실행됨(실제 캐시에서는 실행된 적 없음).
- Colab 노트북은 만들지 않았다.
