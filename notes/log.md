# 실험 로그

(날짜 / 실험 ID / 변경점 / 결과 / 해석 / 다음 단계)

## 2026-11-01 / 001_size_collapse_diagnosis / repro_check 수정 (미실행)
- 변경점: src/exp001/extract.py:43 `torch.load(ck, map_location=device)` -> `weights_only=False` 추가 + "자체 학습 체크포인트만 읽음(신뢰)" 주석. 원인은 torch 2.7.1 기본 weights_only=True 의 UnpicklingError (서버 로그, 3 seed x 3회 실패). 다른 체크포인트용 torch.load 는 없음 (나머지는 metadata.pt / drive.py 의 weights_only=True 로 그대로 둠).
- 결과: 코드 수정만 함. repro_check 는 아직 실행하지 않음 (로컬에 GPU/데이터 없음, 서버 접근 불가). 학습 재실행 없음.
- 재시도 카운트: 이번 수정 후 새 시도로 기록 (최대 3회).

## 2026-11-01 / 001_size_collapse_diagnosis / run 이름에 데이터셋 반영 (미실행)
- 배경: 서버에 acwm_phys(레거시) 데이터로 학습된 `$DINO_TRAIN/outputs/exp001_baseline_single_s{0,1,2}/` (DONE 포함)가 있는데, 사용자가 데이터셋을 cloth_sim 으로 정함. 기존 이름 규칙 `exp001_{dist}_s{seed}` 를 그대로 쓰면 overnight.sh/run_all.sh 가 DONE 을 보고 학습을 건너뛰고 results/<run>/ 도 충돌함.
- 변경점: config.yaml 의 데이터셋 블록별 `run_name_template` 추가 (cloth_sim = `exp001_cs_{dist}_s{seed}`, acwm_phys = `exp001_{dist}_s{seed}` 기존과 동일). common.resolve_dataset 이 이를 train.run_name_template 으로 복사. run.py 에 `run_name --model dist:seed` 명령 추가(이름만 출력). overnight.sh 의 name_of 와 DONE 검사(전에는 하드코딩 `exp001_${DIST}_s`), run_all.sh 의 name 을 모두 run.py run_name 으로 얻도록 변경. tests/fake_run.py 도 같은 규칙을 따름. overnight_status.py 는 이름을 만들지 않아 변경 없음. sync.py 는 resolve 된 템플릿을 그대로 쓰므로 변경 없음.
- 결과: 코드 변경만. 학습/추출/진단 실행 없음. 서버의 기존 3개 체크포인트는 건드리지 않음. 로컬 확인은 bash -n, py_compile, run_name 출력(아래 커밋 메시지/보고 참조)만 함.
- 시드: config.yaml 에 seed: 0, train.seed: 0, train.seeds: [0,1,2], probe/bootstrap 시드 등이 명시돼 있음 (확인만).
