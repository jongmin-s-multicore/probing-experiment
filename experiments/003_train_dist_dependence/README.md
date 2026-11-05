# 실험 설명: 003_train_dist_dependence

- 상태 (2026-11-03): **코드 구현 완료, GPU 학습·추출·데이터 다운로드는 하지 않았다.** 이 문서에 실험 결과 수치는 없다(아래 "baseline 값"은 001 results.json 을 읽은 사실이며 003 결과가 아니다).
- 이 머신에서 한 것: `py_compile`, `--dry-run`(train/smoke/extract/metrics/aggregate/listing), 합성 데이터 CPU 셀프테스트(`python run.py selftest`). 이 머신에는 GPU, 데이터, 체크포인트, acwm-phys 체크아웃이 없다. 실행은 Python 3.14 + numpy 2.5 venv(scratchpad)에서 했다. 서버 환경(001: Python 3.9.19, conda env `dino_wm`)에서는 **실행하지 않았다**(코드는 3.9 문법을 쓰도록 작성했으나 미확인).

## 검증할 가설
notes/hypotheses.md 후보 H, 확정 기록 8 (2026-11-03 사용자 승인; 수치 0.2, "3 시드 전부", 학습 분포 4종은 제안값 그대로 채택, 근거 수치 아님. 결과를 본 뒤 바꾸지 않는다).
- **H-H1**: 끝 크기 하나로만 학습한 모델(a026 단일, a248 단일)의 open-loop 면적(k=12)이 평가 크기들을 학습 크기 쪽으로 끌어당기는가, 아니면 끌림의 중심이 ind 근처에 남는가(가운데 압축).
- **H-H2**: 크기를 섞어 학습(mix_adjacent, mix_wide)하면 baseline_single 보다 open-loop 수축(ol@k12 크기 곡선 기울기 감소)이 줄어드는가. (코드의 "기울기"는 log ol 면적 대 log real 면적의 기울기이고, 1 에 가까울수록 크기를 보존한다. 001 results.json 의 `slope_model_vs_real_group_means/ol@k12` 와 같은 정의다.)

## 방법
- 학습 분포 4종 (모두 `*_train` split 에서만, metadata 순서 앞에서부터; 평가는 `*_test` 만 → 학습/평가 에피소드는 겹치지 않음, selftest 로 확인):
  `single_a026` (a026_train 2000), `single_a248` (a248_train 2000), `mix_adjacent` (a080 667 + ind 667 + a125 666), `mix_wide` (a040 667 + ind 667 + a160 666). 총 2000 episode 로 통일.
- 학습: 001 과 같은 acwm-phys commit(`5a640be...`), batch 24, 15 epoch, num_hist 1, frameskip 5, predictor_lr 5e-5, 학습 시드 0/1/2 → 4 분포 x 3 시드 = 12 모델. **`train.py` 는 모델마다 다르다(2026-11-05 사실 기록)**: `mix_wide` 3개는 Colab A100 에서 고정 commit 의 `train.py` **무수정**(패치 없음, save_every 5)으로 학습했고, 나머지 9개(single_a026, single_a248, mix_adjacent x 시드 3)는 `colab/train_resume.patch`(002 의 `colab/train_resume.patch` 에서 `val()` 위 `@torch.no_grad()` 를 삭제하는 마지막 hunk 만 뺀 것, sha256 41f1098b...)를 적용한 `train.py`, save_every 1 로 학습한다. 패치는 재개(인코더/optimizer 상태 복원, `training.epochs` 를 총 목표 epoch 로 해석)용이며 새 학습의 epoch 범위는 1..15 로 원본과 같다(selftest 가 AST 로 확인).  `val()` 위의 `@torch.no_grad()` 는 유지된다(사용자 결정 2026-11-05; 002 패치는 이 줄을 지우므로 그 hunk 를 뺐다).
- 평가: 9개 크기 x 100 test episode, open-loop k=12 (001 extract 와 같은 rollout 경로). 면적은 001 과 같은 HSV 마스크 규칙(224x224).
- 지표 (`metrics.json`, 모델마다):
  1. 크기별 `ratio_ol_over_real` (k=12): episode 별 `area_ol / max(area_real,1)` 의 평균, episode bootstrap 95% CI (2000회).
  2. `slope_ol`: 크기 9점 그룹 평균 log ol 대 log real 의 기울기 + 001 과 같은 크기 내 episode bootstrap CI(500회, `_boot_slope` 복사). 참고로 trained+interp 크기만, extrap 크기만의 기울기도 함께 저장(판정에는 all_sizes 사용).
  3. 끌림의 중심 `c`: 크기별 평균 ratio 가 1 을 가로지르는 크기. 인접 격자 사이를 (log 명목 면적비, log 평균 ratio) 좌표에서 선형 보간. 가로지름이 정확히 1회일 때만 값, 0회면 null(이유: 전 크기가 1 위/아래), 2회 이상이면 null(이유: 모호, 후보 목록은 `crossings` 에 저장).
  4. 에피소드별 `area_real`, `area_ol` (k=12) 를 크기별로 저장 → 재계산 가능.
- probe / intervene / tf1 기울기는 구현하지 않았다 (tf1 기울기는 001 fig1 과 불일치, 원인 미확인).
- 판정 표 (`results.json` → `verdict`): 임계값은 config.yaml `criteria` 를 코드가 읽어(`verdict.Prereg` 가 읽은 값을 `criteria_used` 에 기록) pass / fail / not_evaluable 을 만든다. 해석 문장은 없다.
  - H-H1: 단일 모델 각각, 시드마다 `c_closer_to_train_size` (|log c - log 학습크기| < |log c - log 1.0|), `ci_condition_met`. 두 모델 모두 3 시드 전부 (c 가 학습 크기에 더 가깝고 + CI 조건) → pass. 두 모델 모두 3 시드 전부 c 가 ind 에 더 가까움 → fail(가운데 압축 쪽). 그 밖(c null 포함) → not_evaluable.
  - H-H2: 혼합 분포마다 (혼합 시드 3 x baseline 시드 3 = 9쌍) 기울기 차가 전부 ≥ 0.2 → pass, 9쌍 모두 |차| < 0.2 → fail, 그 외 → not_evaluable. 두 혼합 분포가 모두 pass 면 전체 pass, 모두 fail 이면 fail, 그 외 not_evaluable. baseline 값은 `../001_size_collapse_diagnosis/results.json` 에서 읽고 파일/경로가 없으면 not_evaluable.
  - baseline 기울기(001 results.json `runs/exp001_cs_baseline_single_s{0,1,2}/diagnose/size_curve/slope_model_vs_real_group_means/ol@k12/slope`, 읽은 사실): 0.2366, 0.3102, 0.1500.

### 사전 등록 문구를 구현하면서 내가 고른 해석 (사용자 확인 필요)
1. **"반대쪽 크기 CI"**: 학습 크기를 뺀 8개 크기 모두의 ratio CI 가 1 을 학습 크기 방향으로 제외해야 한다(`h1_opposite_sizes_rule: all_untrained`). 끝 크기라 전부 한쪽에 있기 때문. 크기별 플래그는 저장하므로 "일부만"으로 재계산 가능.
2. **c 의 보간 좌표**: log-log. 선형 좌표(면적비-ratio)로 하면 c 가 달라질 수 있다.
3. **c 가 여러 번 나오는 경우**: null(판정 불가 쪽). 방향이나 학습 크기와의 근접성으로 고르지 않았다(고르면 가설 쪽으로 편향되므로). 노이즈로 1 근처를 오르내리면 not_evaluable 이 많아질 수 있다.
4. **H-H2 "3 시드 같은 방향, 평균 비교 안 함"**: 9쌍 전부 규칙. baseline 시드 간 범위가 0.15~0.31 이므로 이 규칙에서 pass 에는 혼합 모델이 전부 baseline 최대 + 0.2 ≥ 약 0.51 이상이어야 한다(산술 사실).
5. **H-H2 기울기에 쓰는 크기**: 9개 크기 전체(baseline 과 같은 정의). "외삽 크기 결과는 따로 보고"는 trained+interp / extrap 부분 기울기와 크기별 ratio 를 `metrics.json` 에 같이 저장하는 것으로 구현. 부분 기울기에 대한 baseline 값은 001 에서 계산되지 않았다.
6. 혼합 분포 episode 수: 사용자 지정 667/667/666 에 따라 ind 667 (001 의 mix 는 667/666/667 이었다).

## 구조
```
config.yaml   설정(시드, 분포, 학습, 평가, criteria, smoke, sync)
run.py        단계 CLI
setup.sh      환경 설치(001 setup.sh 복사: acwm-phys checkout + conda env). 미실행
run_all.sh    check → selftest → download → maskcheck → listing → (모델별) train → extract → metrics → aggregate
src/exp003/   001 src 개작: common data masks maskcheck drive hf sync (거의 그대로) + extract(경량화) metrics verdict aggregate smoke selftest
results/<run>/metrics.json  모델별 지표      results/smoke.json  smoke 측정      results/attempts.json  단계별 연속 실패 횟수
results.json  aggregate 출력(지표 + 판정 표 + 사용한 criteria 값)      logs/  단계별 로그
```
001 대비 변경: (a) legacy `acwm_phys` 데이터셋 분기 제거, (b) extract 는 teacher-forced, latent, pooled, 랜덤 ViT, 픽셀 격자를 저장하지 않고 area_real/area_ol 만 저장(open-loop 경로 코드는 동일), (c) probe/intervene/diagnose 제거, (d) 002 에서 가져온 3회 재시도 가드·conda 경고·단계별 로그, (e) 학습 분포 4종, (f) smoke 단계. 001, 002 폴더는 수정하지 않았다.

## 실행 방법 (서버에서, 사용자 승인 후)
작업 디렉터리 `experiments/003_train_dist_dependence`, conda env 활성화(001 과 같은 `dino_wm`; 002/환경 문제로 바뀌었다면 그 env). 환경변수: `DINO_WORK DATASET_DIR DINO_TRAIN EXP003_CACHE ACWM_PHYS_SRC`(기본값은 `~/dinowm` 하위).
```
bash setup.sh                        # 최초 1회 (acwm-phys checkout + env). 001 에서 이미 했다면 불필요
python run.py check                  # 환경/경로/체크아웃 commit/데이터 유무
python run.py selftest               # 합성 데이터 CPU 테스트(결과 아님)
python run.py download --dry-run     # rclone 로 목록/크기만. 이어서 python run.py download (*_train 6개 추가 + *_test 9개; ind_train 포함 7개 train split. 용량 미확인)
python run.py maskcheck ; python run.py listing      # 마스크 점검 + manifest(분포별 train_*.json)
python run.py smoke                  # 12GB 실측 (GPU 사용: mix_wide 60 episode 1 epoch + 소규모 extract, nvidia-smi 폴링) → results/smoke.json
# 사용자 승인 후:
EXP003_APPROVED="single_a026 single_a248 mix_adjacent mix_wide" bash run_all.sh --from train
python run.py aggregate              # results.json
```
- **Colab 마운트 복사 모드 (2026-11-04 추가, 미검증)**: rclone 대신 마운트된 Drive 에서 복사한다. `EXP003_DATA_MODE=mount` 또는 `EXP003_DRIVE_MOUNT_SRC=<split 폴더들이 든 폴더>`(또는 `EXP003_DATA_ARCHIVE`)를 지정하고 `python run.py download` 를 실행한다(`--dry-run` 은 복사 계획만 출력). `EXP003_WORK_ROOT/cloth_sim_data.tar|zip` 이 있으면 그 한 파일을 풀고, 없으면 `/content/drive/MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move` 등 후보 경로에서 병렬 복사한 뒤 검증 후 아카이브를 한 번 만든다. 완료 마커(`.exp003_download_ok`)와 검증은 rclone 경로와 같다. 환경변수를 지정하지 않으면 기존 rclone 동작 그대로. 실제 Colab 마운트에서는 한 번도 실행하지 않았다(selftest 는 가짜 마운트 디렉터리). 자세한 내용은 `CHANGES.md`.
- `train` 은 `EXP003_APPROVED` 에 없는 분포를 거부하고(재시도 횟수에 안 셈), `run_all.sh` 는 `results/smoke.json` 이 finished 가 아니면 train 전에 멈춘다(`ALLOW_NO_SMOKE=1` 로만 우회).
- 재시도: 단계별 연속 실패 3회면 거부(`results/attempts.json`), 사람이 항목을 지워야 재개. run_all.sh 는 자체 반복 없음.
- 실패하면 `logs/` 의 로그 파일(traceback 포함)을 보고한다.

## 예상 소요 시간·자원 (추정, 본 실행 전 보고용)
- 001 로그에서 읽은 사실: RTX 5090(32GB)에서 baseline 학습 1건 = 2건 동시 실행 시 약 5h35m(05:20→10:55), 단독 실행 시 약 3h10m(s2: 10:56→14:07). torch 2.7.1+cu128.
- 추정(미확인): 12 모델 순차 학습 ≈ 12 x 3h = 약 38h (RTX 5090 기준). 12GB GPU 는 더 느릴 수 있고 속도 차이는 측정되지 않았다. smoke 의 `estimate_real_train_seconds_per_run` 은 시작 비용을 포함한 선형 외삽이라 과대 추정 가능.
- 12GB: 사용자 보고 "001 약 10GB"만 있고 이 머신에서도 001 로그에서도 확인하지 못했다. torch 버전이 001 과 다를 수 있다(002 environment.yml 은 torch 2.0.1+cu117, 003 은 env 를 정하지 않음 → 사용자 결정). 안 맞으면 batch/에피소드 수 변경이 필요하고, 바꾸면 baseline_single 과의 예산 동일성이 깨지므로 별도 기록·승인이 필요하다.
- 추출: 모델당 900 episode 의 open-loop 추론(GPU). 시간은 smoke 에서 `estimate_real_extract_seconds_per_run` 로 추정. 캐시는 면적 스칼라 + 샘플 프레임뿐이라 001(모델당 약 5GB 추정)보다 훨씬 작다(정량 미측정).
- 디스크: 학습 체크포인트 001 추정치 모델당 약 1.7GB x 12(추정치, 미확인), 데이터는 *_train 6개 추가분 용량 미확인.

## 이 실험으로 내가 주장하려는 것
(주장은 결과가 나온 뒤 /review-experiment 를 거쳐 사람이 확정한다. 지금은 없다.) 이 실험이 답하려는 질문은 두 가지뿐이다: (1) 학습 크기를 a026 / a248 로 옮겼을 때 c 가 학습 크기를 따라가는가(3 시드 전부), (2) 크기를 섞은 학습에서 ol@k12 기울기가 baseline_single 보다 0.2 이상 큰가(9쌍 전부). 원인 층위(디코더/predictor/정보)는 가르지 않는다.

## 검증한 것 / 못한 것
- 검증(이 머신, 합성 데이터): 4 분포 manifest 수/비율/`_train` 한정/평가와 비중복, ratio·기울기·c 가 알려진 합성 곡선에서 복원되는지(c 보간, null 이유 3종), H-H1·H-H2 판정 규칙의 pass/fail/mixed/결측 경우와 config 값 변경 반영, aggregate→results.json 형태, 001 results.json 에서 baseline 기울기 읽기, sync/drive 코드 경로(가짜 rclone), `--dry-run` 명령 구성(`train` 명령은 001 과 같은 override 에 분포별 manifest 만 다름).
- **미검증**: `train.py` 실행, hydra target `exp003.data.load_manifest_slice_train_val`(001 의 `exp001.data...` 이름만 바꿈), `extract.py`(001 에서 teacher-forced 등을 뺀 경량판; open-loop 부분은 복사했지만 GPU 에서 한 번도 돌리지 않음, 001 숫자와 일치하는지 미확인), `smoke.py`(nvidia-smi 폴링), 실제 데이터에서의 `listing`/`maskcheck`/`download`, 12GB VRAM 적합성, 서버 Python 3.9 호환, setup.sh, run_all.sh 의 실제 실행(문법 검사 `bash -n` 과 `--dry-run` 만).
- 알려진 한계: 시드 3개라 방향성만 본다. c 가 null 이면 해당 시드는 판정 불가. 학습 분포 간 총 episode 수는 같지만 크기당 episode 수가 다르다.

## Colab (Pro+, A100) 사용법 (2026-11-04, 노트북 전체는 미검증)
`colab/exp003_colab.ipynb` 를 Colab 에서 위에서 아래로 실행한다(GPU 런타임, Drive 마운트). 003 폴더는 GitHub 에 없으므로 노트북 3번 셀에 코드 스냅샷(tar+gzip+base64)이 내장되어 있고, `MyDrive/exp003_work/exp003_code.tar` 가 있으면 그것이 우선한다. 코드를 바꾸면 `python3 colab/build_notebook.py` 로 노트북을 다시 만든다. 영속 상태는 `MyDrive/exp003_work`(train_runs, exp003_cache, results, logs, manifests; results/logs/manifests 는 심볼릭 링크)이고, 데이터는 `cloth_sim_data.tar` 가 있으면 풀고 없으면 마운트에서 복사한다(`python run.py download` -> `maskcheck` -> `listing`). smoke(9번 셀, `RUN_SMOKE`)와 본 학습(10번 셀, `RUN_TRAIN` + 사용자가 직접 적는 `EXP003_APPROVED`)은 GPU 실행이므로 사람 승인 후에만 True 로 바꾼다. 학습은 `run_all.sh --from train` 이며 DONE 표시가 있는 run 은 건너뛴다. **(2026-11-05 변경)** 새 노트북 `colab/exp003_colab_resume.ipynb` 는 resume 패치를 적용하므로 세션이 끊기면 진행 중이던 run 은 마지막 저장 epoch 다음부터 이어진다(아래 절). 옛 노트북 `colab/exp003_colab.ipynb` 의 "처음부터 다시" 설명은 mix_wide 학습 당시의 것이다. 미검증: 노트북 전체의 Colab 실행(개별 사실 - venv 설치, 체크아웃, Drive 의 tar 존재, 복사 소요 약 10분 - 만 이전 세션에서 확인), `CONDA_ENV=none` 경로, Drive 심볼릭 링크 위에서의 results/logs 쓰기, 모니터의 `epoch_logs.jsonl` 파일명. 승인 없이 GPU 를 쓰는 셀은 없다.

## 재개 패치와 가드 (2026-11-05, 나머지 9개 모델)
- **원래 문제(코드로 확인):** 고정 commit 의 `train.py` 는 `<run>/checkpoints/model_latest.pth` 가 있으면 읽고(`init_models`) `range(init_epoch, init_epoch + training.epochs)` 로 돈다. 이전 `_train_one` 은 DONE 만 보았으므로, epoch 5 저장 후 세션이 끊겨 셀을 다시 실행하면 조용히 15 epoch 를 더 학습하고 DONE 을 남길 수 있었다(epoch 6..20). action/proprio encoder 는 재초기화되고 optimizer 상태는 복원되지 않는다(002 README 10절).
- **변경:** (1) `colab/train_resume.patch`(002 패치에서 no_grad 삭제 hunk 만 뺌; 마커 문자열이 002 이름 `[exp002 operational]` 이므로 003 도 같은 마커를 인식), 노트북 5번 셀이 `git apply --reverse --check` 후 적용하고 sha256 을 기록. (2) `run.py` `_train_one`: 패치되지 않은 `train.py` 이면 학습 거부(rc 3, 실패 횟수에 안 셈); 시작 전 `src/exp003/resume.py` 의 prepare(읽히는 latest 는 재개 / 깨진 latest 는 최신의 읽히는 `model_<e>.pth` 로 복구 / 쓸 수 있는 게 없으면 `train_runs/quarantine/` 로 이동 후 처음부터; ImportError 는 격리하지 않고 실패). (3) `training.save_every_x_epoch` 5 -> 1 (운영 값). DONE 후 `colab.keep_ckpt_epochs` [5,10,15] 외의 `model_<e>.pth` 삭제(`train_env.json` 이 있고 save_every 1 인 run 만; `model_latest.pth` 는 extract 가 읽으므로 항상 유지). (4) 각 시작마다 `<run>/train_env.json` 에 학습 GPU(nvidia-smi), 재개 여부/epoch, 패치 sha256 기록; aggregate 가 `runs.<run>.train_env` 로 results.json 에 포함. mix_wide 는 기록이 없어 `gpu: "unknown"`. (5) `run.py baseline`: 001 results.json(12237220 바이트)을 Drive 에서 찾아 복사, 못 찾거나 baseline run 을 못 읽으면 실패. aggregate 는 `baseline.found` 를 출력하고 못 읽으면 rc 1. aggregate 는 `EXP003_WORK_ROOT/results/results.json` 으로 복사. (6) 이미 DONE 인 run 에 `--force` 는 `EXP003_ALLOW_RETRAIN=1` 없이는 거부.
- **고정 값 불변:** 분포 4종, 시드, batch 24, 15 epoch, num_hist 1, frameskip 5, predictor_lr, criteria. 패치된 `train.py` 에서 `training.epochs=15` 는 총 목표이므로 새 학습 1..15, epoch 5 이후 재개 6..15.
- **한계/미검증(추정 포함):** 실제 cloth 환경·torch 2.0.1·Colab·L4 에서 재개는 실행하지 않았다. 재개한 run 은 중단 없는 run 과 bit 단위로 같지 않을 수 있다(DataLoader 순서/RNG 미복원). 9개 중 재개가 일어난 run 은 `train_env.json`/results.json 의 `n_resumes`, `resumed` 로 구분한다. L4(24GB) 메모리와 epoch 시간은 측정하지 않았다(smoke 는 A100). 옛 Drive `exp003_code.tar` 는 새 노트북에서 기본적으로 무시된다.
