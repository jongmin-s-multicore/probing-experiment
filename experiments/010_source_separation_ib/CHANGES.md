# 010 CHANGES

## 2026-11-10 Colab 노트북 추가, code_version, 사전 점검 (코드 변경, 미실행; GPU 미사용)
- 요청: 사용자 "010 코랩 노트북 만들어줘" (2026-11-10). 010 가설은 초안이고 Delta, 과제 시드, GPU 승인은 사용자 결정 대기. config.yaml 은 바꾸지 않았다.
- 추가: `colab/build_notebook.py` -> `colab/exp010_colab.ipynb` (30셀, 코드 셀 23개). 009/006 패턴(git clone, 승인 게이트, 폴더 분리). 004/005 파일은 수정하지 않고 `run.py enc_cache` / `run.py feat` 호출과 `exp045_tools` import 만 한다.
- `src/exp010/stages.py`: `code_version(exp_dir)` 추가, `results.json` 에 `code_version` 필드(git_head, git_status_dirty, repo_url, requested_ref; 환경변수 EXP010_CODE_REPO_URL/REF) 추가. `missing_inputs(cfg, paths)` 추가.
- `run.py`: `heads`(--dry-run 제외)가 시작 시 `missing_inputs` 를 검사해 캐시 4개 또는 002 Q1 캐시가 없으면 종료코드 4 로 끝낸다(attempts 소비 없음). 이전에는 FileNotFoundError 로 시도 1회를 소모했을 것이다(코드 읽기로 확인, 실행으로는 미확인).
- `src/exp010/selftest.py`: D 블록(노트북 JSON/셀 문법/게이트 기본값/폴더 분리/run 이름 `exp002_multi_60_ind_160_s0`/로그 비교 함수를 저장소의 원래 004/005 로그로 시험/missing_inputs/code_version/004,005 config 값/plan.json) 추가. 이전 PASS 19 -> 37.
- 확인된 사실(코드/로그 읽기): 원래 004 본 실행 `enc_cache` 는 k_grid [0,2,4,6,8,10,12], 프레임 [0,1,2,3,8,9,10,11], 에피소드 1800, random_init_control true, Colab L4 177 s. 원래 005 `feat` 는 pred_out6 만, 1800 에피소드, 137 s (`004/results/colab/timing_main.json`). 005 config.yaml 의 현재 sha256 은 `005/results.json` 의 config_sha256 과 다르다(원인 미확인).
- 하지 않은 것: Colab 실행, GPU 실행, git commit/push, notes/hypotheses.md 수정.
