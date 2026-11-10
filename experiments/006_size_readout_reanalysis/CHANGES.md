# 006 CHANGES

## 2026-11-10 smoke 실행 이름 수정 (Colab 셀 9 오류)
- 증상: Colab 셀 9(`extract --smoke`)가 `train_runs/outputs/multi_60_ind_160_s0/...` 에서 체크포인트를 찾다 실패. Drive 의 실제 폴더명은 `exp002_multi_60_ind_160_s0` 등 `exp002_` 접두사가 붙어 있다.
- 원인: `config.yaml` `extract.smoke.run` 만 접두사 없이 적혀 있었다. 전체 추출이 쓰는 `runs.run_name_pattern` (`exp002_{cond}_s{seed}`) 은 원래 맞았다.
- 수정: `extract.smoke.run: "exp002_multi_60_ind_160_s0"`. `run.py extract` 사전 점검 추가: smoke/`--runs` 이름이 `run_name_pattern` 의 실행 목록에 없거나 `checkpoints/model_latest.pth` 가 없으면 GPU 로드 전에 종료코드 4 로 멈춘다(시도 횟수 소모 없음).
- 확인(CPU, 가짜 입력 폴더): 옛 이름이면 사전 점검에서 멈추고, 새 이름이면 사전 점검을 통과한다. 결과 수치 없음.

## 2026-11-10 분석 F 추가, 노트북 git clone 전환 (사용자 지시: notes/hypotheses.md "후보 J 추가 (2026-11-10)")
코드 변경은 사실만 적는다. 결과 수치 없음(GPU 미실행). 002/004/009 파일은 수정하지 않았다.

### 분석 F (teacher forcing 대 open-loop)
- `src/exp006/extract.py`: `teacher_forced(model, z_all, t, horizon, H)` 와 `tf_feasible_steps(...)` 추가. `run_pass` 가 (open-loop, untrained, teacher-forced) 3개를 반환(이전 2개). 학습 episode 와 test episode 모두 open-loop 와 같은 범위로 처리, 같은 풀링(fp16 캐스트 후 float32 패치 평균). 저장: `extract/<run>/teacher_forced_pooled.npy` `(1800,12,384)`, `meta.json` 의 `teacher_forced`(가능한 step 목록, step 1 비교 차이). `config.yaml` `extract.teacher_forcing: true`. DONE 마커가 있어도 `teacher_forced_pooled.npy` 가 없으면 다시 계산.
- 가능한 step 범위: `t=0, H=1, horizon=12, 캐시 프레임 13장` 에서 step 1..12 전부(코드 `tf_feasible_steps`, selftest 로 확인). 불가능한 step 없음.
- `src/exp006/ctx.py`: `load_extract` 가 `tf` 를 읽음(없으면 None). `stages_pre.py`: 전이 probe 적용 집합에 `teacher|<run>|<k>` 72개 추가(`n_tf`).
- `src/exp006/stages_f.py`(신규): `stage_f` = 분석 F. `run.py` 에 단계 `analyze_f`, `cpu_all` 에 포함, `--dry-run` 계획. `stages_agg.py`: results.json 에 `F`, `limitations` 한 줄 추가. `figs.py`: `figF_teacher_forcing_vs_open_loop.png`.
- 새 시드 없음(기존 `seeds.probe`, `seeds.retrain_probe`, `seeds.bootstrap`). `config.yaml` 은 `extract.teacher_forcing`, `analysis.f_note` 두 항목만 추가.

### 재현성, 안전
- `base.code_version()`: results.json `code_version` (`git_head`, `git_status_dirty`, `repo_url`, `requested_ref`; 후자 둘은 노트북이 설정하는 환경변수 `EXP006_CODE_REPO_URL`, `EXP006_CODE_REF`).
- `stages_pre.metadata_missing`: 학습 split 의 `metadata.pt` 사전 점검(009 파일럿 실패 교훈). `check` 가 `input_manifest.json` 에 `metadata_pt_ok` 기록, `run.py extract` 가 시작 전에 점검해 없으면 종료코드 4(시도 횟수 소모 없음).

### Colab 노트북
- `colab/build_notebook.py` 재작성. 코드 스냅샷(base64 tar.gz) 임베드 제거. 셀 3 이 공개 저장소를 `/content/mylab006` 에 clone, `REPO_REF` checkout, HEAD 해시를 `exp006_work/code_version.json` 에 기록. 노트북 크기 472,239 -> 23,600 바이트 수준(빌드 후 확인값은 README 9절).
- 009 와 병행 가능하도록 출력 `MyDrive/exp006_work`, 로컬 `/content/mylab006`, `/content/venv006`, `/content/dinowm006` (009 와 겹치지 않음). 009 모듈 import 없음(노트북 안에 `sh`, `bg`, `run_stage` 정의).
- 셀 순서: 설정/마운트, clone, 환경, ACWM commit, 환경변수, `metadata.pt` 배치와 확인, `--dry-run` 전 단계 + check + selftest, smoke, extract(기본 꺼짐), CPU 분석, aggregate/figures, flush.

### 검증 (CPU, 합성 데이터/toy 모델)
- README 8절 "검증 기록" 참조.
