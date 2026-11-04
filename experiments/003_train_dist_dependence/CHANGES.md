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
