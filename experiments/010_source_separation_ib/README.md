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
- Colab 노트북은 아래 "Colab 실행" 절에 있다(2026-11-10 추가). Colab 에서는 실행된 적 없다.

## Colab 실행 (2026-11-10 추가)
노트북: `colab/exp010_colab.ipynb` (`python colab/build_notebook.py` 로 생성; 코드 임베드 없음). 셀 A3 가 공개 저장소 `REPO_URL` (https://github.com/jongmin-s-multicore/probing-experiment.git, 009 와 같은 값) 을 `/content/mylab010` 에 git clone 하고 `REPO_REF`(셀 A2, 기본 'master'; commit 해시/태그 가능) 를 detach checkout 한다. HEAD 해시는 출력되고 `exp010_work/code_version.json` 과 `results.json` 의 `code_version`(`git_head`, `git_status_dirty`, `repo_url`, `requested_ref`; 환경변수 `EXP010_CODE_REPO_URL/REF`) 에 기록된다. **이 변경(010 코드, 노트북이 쓰는 파일 포함)은 사용자가 commit/push 해야 Colab 에 반영된다**(이 작업에서는 commit 하지 않음). A3 는 ref 에 새 010 코드(`code_version`, 사전 점검)가 없으면 멈춘다.
- 폴더 분리(006/009 와 병행 가능): Drive 출력 `MyDrive/exp010_work`, 로컬 `/content/mylab010`, `/content/venv010`, `/content/cache010`. 예외 하나: 002 manifest 의 episode dir 이 절대경로(`/content/dinowm/data/cloth_sim/<split>`)라 metadata.pt(읽기 전용, 내용 동일)와 ACWM 소스는 009 와 같은 `/content/dinowm` 아래.
- 셀 순서: A 설정/마운트/clone/환경/ACWM/환경변수/디스크 -> B 입력 확인(B1 002, B2 캐시 4개 복원, B3 dry-run/check/selftest) -> R 캐시 재생성(R0 승인, R1 사전 점검, R2 004 enc_cache, R3 005 feat, R4 검증) -> C 선택 파일럿(C0 승인, C1, C2 시간 추정) -> D 본 실행(D0 승인, D1 heads) -> E report -> F 결과 복사.
- 승인 게이트(모두 기본 False, 사람이 직접 True): `APPROVED_REGEN_010`(R0), `APPROVED_PILOT_010`(C0), `APPROVED_MAIN_010`(D0). 본 실행은 config `pilot: true` 이면 거부(`ALLOW_MAIN_WITH_PILOT_TRUE` 로 해제). 노트북은 config.yaml 값(Delta 0.05, 과제 시드 [2000, 2001, 2002] 초안 등)을 덮어쓰지 않는다. 이 값들과 GPU 승인은 사용자 결정 대기 상태다.
- 캐시 4개는 Drive 에 없다(사용자가 Drive 검색으로 확인). 004 README 6절: 세션 로컬에만 있었다. 재생성(R): 수정 없는 `004 run.py enc_cache`(k_grid [0,2,...,12], random_init_control true, 소스 학습 seed 0) 와 `005 run.py feat`(학습 seed 0, random_init_seed 1234, 층은 저장소의 `005/results/plan.json` = pred_out6 만) 를 각각 자기 config.yaml 로 실행한다. 각각 `--results-dir exp010_work/regen/results_004|005` 로 분리해 004/005 의 attempts 와 결과는 건드리지 않는다. 필요한 4개 파일만 Drive `exp010_work/cache004/enc/`, `exp010_work/cache005/` 에 복사하고 sha256 을 `cache_record.json` 에 기록한다. 다음 세션은 B2 가 복원하고 크기/sha256 을 검사한다.
- 같은 설정인지 확인(R1, R4): R1 은 config 값을 원래 실행 값과 대조(k_grid, random_init_control, source_train_seed, feat.train_seed 0, random_init_seed 1234, 002 random_init seed 0, plan.json 층)해 다르면 GPU 전에 중단한다. R4 는 새 로그의 `enc_cache plan`(k_grid, frames, 에피소드 1800, random_init_control), `feat plan`, 마지막 진행 줄, 004 의 일관성 검사 3줄을 저장소의 원래 로그(`004/logs/colab_20261109/enc_cache_20261109-092707.log`, `005/logs/colab_20261109/feat_20261109-112051.log`)와 비교해 출력하고, 4개 파일의 shape/dtype 과 1800 에피소드 x 8 프레임을 검사한다. 원래 파일이 없어 비트 단위 비교는 할 수 없다(GPU 비결정성 영향은 확인하지 않음). 005 config.yaml 의 sha256 은 005/results.json 의 값과 다르다(원인 미확인; 005 CHANGES.md 는 Colab 에서 n_inputs 를 900 으로 바꿨다고 기록; feat 는 n_inputs 를 쓰지 않음). `eval.n_inputs 900` 은 010 config 가 덮어쓰며 캐시와 무관하다.
- 사전 점검(종료코드 4, attempts 소비 없음): `run.py heads`(--dry-run 제외)는 시작 시 4개 파일과 002 Q1 캐시를 검사하고 없으면 종료코드 4. R1 은 체크포인트(`exp002_multi_60_ind_160_s0/{DONE,hydra.yaml,train_manifest.json,checkpoints/model_latest.pth}`; 실제 Drive 폴더 목록은 B1 이 출력), metadata.pt(009 A6b 와 같은 방식으로 배치), 위 config 값을 GPU 전에 검사한다.
- 시간(추정 표시): 본 heads 약 3-3.5 시간(위 "예상 GPU 시간", 추정, 실측 없음; 파일럿 C2 가 순진한 외삽을 출력). 캐시 재생성: 실측 `004_depth_truncation/results/colab/timing_main.json` 의 Colab L4 값 enc_cache 177 s, 005 feat 137 s(Drive 읽기/복사 제외). Drive 로 복사하는 시간과 두 feat 파일 크기는 측정값 없음(추정). 로컬 디스크: 004 enc_cache 전체 출력 약 30 GB(004 로그 `tokens_total_gb 30.3`).
- 하위 프로세스 환경에 `MPLBACKEND=Agg`(004 Colab 오류 대응). 파이썬은 002 와 같은 uv venv(py3.9).
- 미검증: 노트북은 Colab 에서 실행된 적 없다. 실제 predictor 는 CPU 에서 못 돈다(Attention.bias.to('cuda')). CPU 검증은 `CHANGES.md`.
