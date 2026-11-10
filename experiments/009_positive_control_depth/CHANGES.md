# 009 CHANGES

## 2026-11-10 실패 1건 (사용자 Colab 파일럿, 08:21)
- 증상: `intervene` 이 `FileNotFoundError: /content/dinowm/data/cloth_sim/a060_train/metadata.pt` 로 실패 (stages.py `M.action_stats_for_run` -> exp002 `ManifestDataset.__init__` 이 train_manifest.json 의 각 episode dir 의 metadata.pt 를 읽음). results_pilot/attempts.json 의 stage:intervene = 2.
- 원인: 노트북이 metadata.pt 를 로컬에 놓지 않았다. 이전 문구(캐시 토큰만 쓰므로 데이터 다운로드 없음)가 틀렸다.
- 수정: (1) 노트북 셀 A6b (metadata.pt 만 로컬에 배치; 로컬 / DATA_SRC / cloth_sim_data.tar 순; 못 찾으면 split 이름 출력하고 중단), 셀 A2 에 DATA_SRC. (2) `stages.metadata_missing/metadata_preflight`; `run.py check` 가 검사, `run.py intervene` 이 attempts_check 이전에 검사해 없으면 종료 코드 4 (attempts 미소비; attempts 카운터 규칙은 그대로). (3) selftest 점검 6개 추가 (passed=71 failed=0).

## 2026-11-10 노트북 코드 공급 방식 변경 (사용자 결정)
- 코드 임베드(base64 스냅샷) 제거, git clone 만 사용 (REPO_URL, REPO_REF 는 셀 A2 / A3). `stages.aggregate` 가 results.json 에 `code_version` 기록 추가 (git_head 등), 노트북은 `exp009_work/code_version.json` 도 기록.
