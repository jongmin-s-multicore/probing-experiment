# 012 CHANGES

## 2026-11-10 코드와 Colab 노트북 작성 (CPU 점검까지; GPU 실행 없음, 미승인)
- 신규: `run.py` (check | measure | aggregate | selftest, 모두 `--dry-run`), `src/exp012/{base,measure,verdict,stages,selftest}.py`, `colab/build_notebook.py`, `colab/exp012_colab.ipynb`.
- 009 의 `exp009.erase`(fit_eraser, forward_hidden, pooled_hidden, load/save_eraser)와 `exp009.stages`(labels_positions, metadata_missing), 004/002 모듈을 import 만 한다. 009/004/002 파일과 config 는 수정하지 않았다. config.yaml 값은 바꾸지 않았다.
- 실험자가 정한 구현 사항(config 에 없음): README "config 에 없어서 실험자가 정한 것" 1-4. 특히 (b), (c) 의 판정 split 을 `all`(test 전체 합산)로 두고 split 별 판정을 함께 저장한다.
- 사전 점검(종료코드 4, 시도 소비 없음): config 일치(009/002), 체크포인트 폴더 구조 `exp002_` 접두사 포함(006 실패 교훈), Q1 캐시 DONE, metadata.pt(009 파일럿 실패 교훈).
- `pilot` 필드는 config 플래그, 결과 폴더, CLI 덮어쓰기에서 계산하고 이유를 `pilot_reasons` 에 적는다(009 에서 본 실행 결과에 `pilot: true` 가 남은 문제).
- 하위 프로세스 env 에 `MPLBACKEND=Agg` (노트북 A6, exp045_tools 도 강제).
- 검증: `python run.py selftest` passed=82 failed=0 (2026-11-10, CPU, lab07). 노트북은 JSON 로드와 코드 셀 `ast.parse` 만 확인했다. Colab, GPU, 실제 체크포인트 실행은 하지 않았다.
