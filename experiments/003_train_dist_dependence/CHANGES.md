# CHANGES (003_train_dist_dependence)

## 2026-11-04 Colab 마운트 복사 모드 추가 (데이터 다운로드)
- 변경 파일: `src/exp003/drive.py`, `run.py`(cmd_download), `src/exp003/selftest.py`, `README.md`, 이 파일. 002 코드는 읽기만 했고 수정하지 않았다.
- 내용: 002 `drive.py` 의 'Colab path' 구간(mount_src_candidates, find_mount_src, copy_from_mount, _split_done, _finish_split, 아카이브 tar/zip 처리, make_archive, prepare_data)을
  003 에 이식했다. 접두사 EXP002_ -> EXP003_, 설정 접근은 003 방식(`cfg["data"]["drive"]["root_folder_id"]`, `cfg["_ds"]["split_pattern"]`, `cfg["_ds"]["sizes"]`)으로 바꿨다.
- 분기: `drive.mount_mode_requested()` 가 참이면(`EXP003_DATA_MODE=mount`, 또는 `EXP003_DRIVE_MOUNT_SRC` 또는 `EXP003_DATA_ARCHIVE` 설정) `run.py download` 가 rclone 대신 `drive.prepare_data` 를 부른다. 그 외에는 기존 `drive.download`(rclone) 그대로.
- 순서: (1) 모든 split 완료면 아무것도 안 함, (2) `EXP003_WORK_ROOT/cloth_sim_data.tar|zip` 있으면 압축 해제, (3) 아니면 마운트에서 병렬 복사(`EXP003_COPY_WORKERS`, 기본 16) 후 검증이 끝나면 아카이브를 한 번 생성(`EXP003_MAKE_ARCHIVE=0` 으로 끔).
- 마운트 후보: `EXP003_DRIVE_MOUNT_SRC` -> `/content/drive/MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move` -> `/content/drive/.shortcut-targets-by-id/<data.drive.root_folder_id>` -> `MyDrive/dino-wm-videos` 아래 깊이 3 탐색.
- 완료 마커/검증: 기존 `.exp003_download_ok` 유지. 마운트/아카이브 경로의 마커 JSON 에는 `bytes`, `source`(drive_mount | archive)가 추가로 기록된다(기존 `split_complete` 는 이 필드를 보지 않아 호환). 검증은 mp4 수 == metadata 길이 == 기대 에피소드 수.
- 003 추가 수정 1건: `make_archive` 가 대상 폴더를 만들도록 `dest.parent.mkdir` 추가(002 원본은 폴더가 있다고 가정; selftest 에서 폴더가 없을 때 아카이브 생성 실패로 발견). 002 는 건드리지 않았다.
- selftest: `_selftest_mount` 추가(가짜 마운트 디렉터리; 대괄호 경로, 후보 순서, 병렬 복사, 마커/바이트, skip/resume, tar/zip, 불완전 아카이브 rc 3, `..` 경로 거부, 소스 없음 rc 2, dry-run).
- 결과(이 머신, python3 3.x): `python3 run.py selftest` 는 numpy 미설치로 import 단계에서 실패(`ModuleNotFoundError: numpy`; conda base 와 시스템 python 모두). 대신 numpy 를 빈 스텁으로 대체한 임시 하네스(저장소 밖)로 기존 `_selftest_drive` 와 새 `_selftest_mount` 만 실행해 통과. 전체 `run.py selftest`(metrics/verdict 포함)는 이 머신에서 **미실행**. `py_compile`(drive.py, selftest.py, run.py) 통과.
- 미검증: 실제 Colab 의 Drive 마운트(`/content/drive`) 경로 구조(대괄호/공백 이름, `.shortcut-targets-by-id`), 실제 데이터 크기/속도, 실제 mp4/metadata.pt 로 한 검증(selftest 는 텍스트 가짜 파일 + 주입 카운터), `run.py download` 의 환경변수 분기 CLI 경로(numpy 없어 실행 못 함, 코드 리뷰만), 아카이브 생성 시간/용량.

## 2026-11-04 selftest `_selftest_mount` 수정 (UnicodeDecodeError)
- 증상: 실제 환경(py3.9, torch 2.0.1)에서 `_finish_split` 의 selftest 카운터 `int(Path(p).read_text())` 가 metadata.pt 의 바이너리(0x80, pickle)를 읽다가 UnicodeDecodeError.
- 코드에서 확인한 원인(추정, 서버 미실행): selftest 가 쓰는 가짜 metadata.pt 는 항상 텍스트("3")다. 바이너리가 나올 수 있는 유일한 경로는 마지막 케이스
  `prepare_data(pathsF, ...) == 2`(마운트 없음 기대)다. 이 호출은 `find_mount_src` 를 기본 `/content/drive` 로 부르기 때문에, 그 위치에 실제 Drive 마운트가 있으면
  실제 cloth_sim 데이터를 임시 폴더 F 로 복사하고(실제 torch.save metadata.pt) 텍스트 카운터가 깨진다. (torch 유무가 아니라 실제 마운트 존재가 원인이라는 판단. 같은 환경이 아니면 재현 안 됨.)
- 수정 1 (`drive.py`, 최소): `prepare_data(..., mount="/content/drive")` 인자 추가, `find_mount_src`/`mount_src_candidates` 에 전달. 기본값이 같아 실제 다운로드 동작은 불변.
- 수정 2 (`selftest.py`): rc 2 케이스에 빈 임시 마운트 `td/"empty_mount"` 전달(실제 /content/drive 를 보지 않음). 또한 `mc` 가 텍스트 파싱 실패 시 `drive.meta_len`(torch)으로 폴백하도록 함.
- 검증: py_compile 통과, numpy 스텁 하네스(`_selftest_drive`+`_selftest_mount`) 통과. 서버(py3.9/torch)에서 `python run.py selftest` 재실행은 **미검증**.

## 2026-11-04 PYTORCH_CUDA_ALLOC_CONF=expandable_segments 조건부로 변경 (smoke 사망 수정)
- 증상(실제 Colab 서버 로그, torch 2.0.1+cu117): smoke 가 모델을 GPU 로 옮기는 단계에서 `RuntimeError: Unrecognized CachingAllocator option: expandable_segments`.
- 원인: `run.py` `_train_cmd` 가 자식 프로세스 env 에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 무조건 넣었다. 이 옵션은 torch >= 2.1 에서만 인식된다.
- 수정(`run.py`): (1) 부모 env 에 `PYTORCH_CUDA_ALLOC_CONF` 가 이미 있으면 그 값을 그대로 상속(건드리지 않음), (2) 없으면 설치된 torch 버전(`importlib.metadata`, torch import 안 함)이 >= 2.1 일 때만 넣음. torch 없음/버전 판독 실패 -> 넣지 않음.
- grep 확인: `expandable_segments` / `PYTORCH_CUDA_ALLOC_CONF` 는 run.py:139 한 곳뿐이었다(smoke.py, run_all.sh, setup.sh, src/, README 에 없음). smoke 는 `_train_one` -> `_train_cmd` 를 공유하므로 같은 수정이 적용된다.
- 같은 날 추가(`run_all.sh`): `CONDA_ENV=none` 이면 conda activate 를 건너뜀(Colab 은 conda 없음; PATH 앞쪽에 venv python). 기본값 dino_wm 동작은 불변.
- 검증: `py_compile`, `bash -n` 통과; 이 머신에는 torch 가 없어 `_torch_at_least((2,1))` 이 False 를 반환하는 것만 확인. torch 2.0.1 서버에서 smoke 재실행은 미검증(사람 승인 필요).

## 2026-11-04 Colab 노트북 추가
- 추가: `colab/exp003_colab.ipynb`, `colab/build_notebook.py`(노트북 재생성; 003 코드 스냅샷을 tar+gzip+base64 로 내장), `colab/requirements_colab.txt`(002 의 파일 복사본). 002 파일은 읽기만 했다.
- resume 패치(002 train_resume.patch)는 의도적으로 넣지 않음: 세션이 끊기면 진행 중이던 run 은 처음부터 다시 학습된다.

## 2026-11-05 나머지 9개 모델: 재개 패치, 가드, 저장 주기, baseline/결과 영속화, GPU 기록, 새 노트북
- 승인: 사용자가 2026-11-05 에 나머지 9개(single_a026, single_a248, mix_adjacent x 시드 0,1,2) GPU 학습, 재개 패치 적용, 현재 Colab L4 런타임 사용을 결정. 이 변경은 코드/노트북 준비까지이며 GPU 실행은 하지 않았다.
- 원래 문제 확인(코드): 고정 commit `train.py` 는 `model_latest.pth` 가 있으면 읽고 `range(init_epoch, init_epoch + training.epochs)` 로 돌며(git show 5a640bed:train.py 직접 확인), 이전 `_train_one` 은 DONE 만 확인했다. 따라서 epoch 5 저장(save_every 5) 후 재실행하면 6..20 을 학습하고 DONE 이 남을 수 있다. 판단이 맞았다.
- 패치: `colab/train_resume.patch` = `experiments/002_single_size_sweep/colab/train_resume.patch` 에서 마지막 hunk(`val()` 위 `@torch.no_grad()  # [repro] ...` 삭제, 24GB OOM 방지 줄)만 뺀 것. 사용자 결정(2026-11-05)으로 그 줄을 살린다. 재개 관련 hunk 5개는 002 와 동일. 새 sha256 41f1098bd54bae7be268be497e12e61d7128d3a4e26f51f174881868158fd5ff (002 원본은 52eb0052...; 002 파일은 수정하지 않음). 마커는 002 이름 `[exp002 operational]` 그대로(`resume.RESUME_MARKER`), 루프식 `range(init_epoch, self.total_epochs + 1)` 도 확인.
- 새 파일: `src/exp003/resume.py`(002 colab_tools/trainrun 이식: prepare_run, repair_latest, quarantine, prune_run, train_py_is_patched, GPU/attempt 기록), `src/exp003/baseline.py`, `colab/train_resume.patch`, `colab/exp003_colab_resume.ipynb`.
- 수정: `run.py`(`_train_one` 가드/prepare/기록/prune, `guarded` 가 rc 3 을 실패로 세지 않음, `baseline` 단계, aggregate 후 WORK_ROOT 복사 및 baseline.found 출력/rc 1), `run_all.sh`(baseline 단계 추가), `config.yaml`(save_every 5 -> 1 운영 값, `colab.keep_ckpt_epochs`, `colab.baseline_001_bytes`), `src/exp003/aggregate.py`(runs.<run>.train_env, `experiment_code_sha256`, `environment.gpu_note`, `persist`), `src/exp003/common.py`(`code_snapshot_sha256`), `src/exp003/selftest.py`(`_selftest_resume`), `colab/build_notebook.py`(출력 파일명 exp003_colab_resume.ipynb, 패치 포함, 셀 변경), README.md.
- 노트북 변경: 5번 셀 패치 적용(`git apply --reverse --check` 후 적용, 패치 sha256 을 `WORK_ROOT/acwm_resume_patch.sha256` 에 기록, 마커/루프식 assert), 3번 셀은 내장 스냅샷을 쓰고 Drive 의 옛 `exp003_code.tar` 는 `USE_DRIVE_TAR=True` 일 때만 사용(그 외 경고 출력), 새 9번 셀 `run.py baseline`, 학습 셀 `EXP003_APPROVED='single_a026 single_a248 mix_adjacent mix_wide'`, `RUN_TRAIN=True`, 2분마다 nvidia-smi 메모리와 run 별 시작 후 경과 분/epoch 줄 수 출력. 옛 `colab/exp003_colab.ipynb` 는 건드리지 않았다(미커밋 실행 출력 포함).
- 기존 결과 보호: mix_wide 3개는 DONE 이면 `_train_one` 과 `run_all.sh` 마커가 건너뛰고, `--force` 는 거부, prune 은 `train_env.json` 이 없는 run 을 건드리지 않으며 prepare 는 DONE/smoke 를 skip. selftest 가 합성 폴더에서 mix_wide/smoke 파일 트리가 바이트 단위로 같음을 확인.
- 검증은 아래 최종 보고와 같다(selftest 수치는 합성 데이터이며 실험 결과가 아니다). 미검증: Colab/L4 에서의 실제 학습, 재개, 메모리, Drive 에서 001 파일 탐색.
