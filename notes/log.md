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

## 2026-11-02 / 001_size_collapse_diagnosis / baseline_single seed 0-2 전체 파이프라인 완료 (cloth_sim)
- 변경점
  - 코드 변경 없음 (experiment_code_git = results.json 기록값). 이번 실행: train -> repro_check -> extract -> diagnose -> probe -> intervene -> aggregate, 3 seed (0,1,2), run 이름 exp001_cs_baseline_single_s{0,1,2}. 모두 1회 시도(retries 0), 에러 로그 없음. 서버: RTX 5090.
  - config.yaml sync.drive: rclone_remote="gdrive", remote_path="exp001_results" 채움 (commit c69d1d4). Drive 업로드 77.8 MB, 67 파일. 폴더명은 `20261102-021800_exp001_baseline_single_s0+...` 로 `cs` 가 빠져 있으나 내용은 cloth_sim run 이다.
  - 로컬로 results.json(12,237,220 B)과 results/ 를 받음. 체크포인트는 업로드하지 않았다(서버에만 있음).
- 알려진 문제 (결과 수치와 무관)
  - overnight_status.py `set` 인자 파싱 버그: `set --file F SEED STAGE STATE` 가 argparse 에서 "unrecognized arguments" 로 실패하고 overnight.sh 의 st() 가 에러를 숨겨 logs/overnight_status.json 이 init 시점(05:20)에서 갱신되지 않음. 파이프라인 진행에는 영향 없음. 수정은 별도(미실시).
  - 사전 등록의 shrink_ratio_tol 은 null 이므로 0번 게이트는 판정하지 않고 수치만 기록됨(results.json 에 "TODO ... needs human decision"). 결과를 본 뒤 이 값을 정하면 사후 기준 조정이므로 정하지 않았다.
- 결과 (모두 results.json 에서 옮김; seed 평균 [seed 최솟값, 최댓값], 에피소드 단위 값은 results.json 참조)
  - 후보 D: decode(encode(real))/real 면적비, k=12: a026 0.936 [0.915,0.974], a040 0.974, a060 0.973, a080 0.979, ind 0.985, a125 0.983, a160 0.977, a200 0.974, a248 0.974 (9개 크기 모두 seed 최솟값 ≥ 0.915, 최댓값 0.990). 사전 등록 하한 0.9, 상한 1.1.
  - 모델 출력/real 면적비 k=12 (tf1 = teacher-forced 1-step, ol = open-loop): a248 tf1 0.797 [0.778,0.828], ol 0.434 [0.333,0.549]; a200 tf1 0.854, ol 0.535; a160 tf1 0.872, ol 0.636; a125 tf1 0.959, ol 0.819; ind tf1 0.970, ol 1.054; a080 ol 1.373; a060 ol 2.920; a040 ol 2.965; a026 tf1 0.965, ol 3.987 [3.799,4.178].
  - 모델 대 real 크기 곡선 기울기(군집 평균, 95% CI), k=12, seed 0/1/2: recon 1.011 [0.994,1.028] / 1.045 [1.020,1.079] / 1.025 [1.008,1.048]; tf1 0.978 [0.938,1.028] / 0.985 [0.945,1.032] / 0.998 [0.957,1.046]; ol 0.237 [0.187,0.288] / 0.310 [0.264,0.361] / 0.150 [0.085,0.211]. seed 0 의 k=3 에서는 tf1 0.748 [0.732,0.763], ol 0.595 [0.578,0.613], k=6 tf1 0.813, ol 0.417.
  - 후보 E probe (pool=mean, 크기 로그 면적 대상, 군집 평균 기울기, seed 평균): MLP strat_cv gt 0.995 / tf1 0.984 / ol 0.809 / random-ViT 0.998 / pixel 0.588; ridge strat_cv gt 0.987 / tf1 0.983 / ol 0.801 / rvit 0.996 / pix 0.951. 인접(±1) 클래스 정확도 MLP strat_cv: gt 0.961, tf1 0.931, ol 0.722, rvit 0.888, pix 0.908. heldout_size(학습 크기 GT로만 probe 학습) MLP: gt 0.829 / tf1 0.729 / ol 0.401; ridge: gt 0.876 / tf1 0.725 / ol 0.388. [정정: 이 값들은 군집 평균 기울기이고 "gt 0.83 -> tf1 0.73 -> ol 0.40" 식의 한 줄 요약은 MLP 와 ridge, 군집 평균과 전체 표본 정의를 섞어 쓸 위험이 있음] gt_transfer MLP: tf1 0.803 / ol 0.422.
  - 후보 E 개입(remove): tf_response_gain, u 방향 vs 무작위 방향(seed 평균): a026 0.179 vs 0.143, ind 0.167 vs 0.136, a160 0.161 vs 0.118, a248 0.159 vs 0.105; a125 는 무작위가 0.239 로 u(0.162)보다 큼(이상치, 원인 미확인).
  - 선택성(실제 − label_permute) 값은 이 정리에서 읽지 못했다(키 경로 확인 못 함). 판정하지 않음. [정정: 키는 probe/pools/<mean|...>/{ridge,mlp}/selectivity_real_minus_permute/... 로 존재하며 내가 경로를 틀리게 조회했다. stats/skeptic 리뷰 참조]
- 해석 (**추정, /review-experiment 전의 잠정 해석이며 확정 아님**)
  - [정정 2026-11-02: 리뷰 후 "충족"은 과장. 점추정만 충족하고 seed 1,2 의 a026 은 CI 하한이 0.9 아래이며, "모델 출력이 더 작다" 조건은 작은 천에서 성립하지 않음. reports/001_size_collapse_diagnosis.md 4,6절] D: 사전 등록 기준(9개 크기 모두 0.9 이상)을 수치상 충족한다. 디코더(재구성)는 수축의 주원인으로 보이지 않는다(추정).
  - [정정: 3.99 는 에피소드별 비의 평균이며 평균끼리의 비는 2.36 (skeptic 계산). "학습 크기로 수렴"은 큰 천에만 맞고 ind 가 크기 범위 가운데라 "가운데로 압축"과 구별되지 않음. 보고서 6절] open-loop k=12 에서 큰 천은 ind 쪽으로 줄고(a248 0.43) 작은 천은 오히려 커진다(a026 3.99). 수축이라기보다 학습 크기(ind) 쪽으로의 수렴으로 보인다(추정). ol 기울기 CI 는 세 seed 모두 1 을 제외한다.
  - 1-step: k=12 에서는 tf1 기울기 CI 가 1 을 포함하지만(세 seed) k=3 에서는 0.748 [seed 0] 이고 큰 천 비율이 0.797 까지 내려간다. 사전 등록 규칙(k=1 정상이고 k 증가로 감소하면 오차 누적)과 맞는지는 k 별 값을 더 봐야 한다. k=3 기울기가 k=12 보다 작은 이유는 설명하지 못했다(미확인).
  - E: GT 인코딩에서 크기 정보는 probe 로 복원된다(MLP strat_cv 기울기 0.995). 그러나 random-ViT 와 pixel 기준선도 높아서(0.998, 0.588~0.951) 선택성을 평가해야 하며 지금은 미판정이다. 학습 크기 GT 만으로 학습한 probe 를 적용하면 gt 0.83 -> tf1 0.73 -> ol 0.40 으로 내려간다. 예측 latent 로 갈수록 크기 정보가 줄어든다는 방향과 일치하나(추정), probe 분포 이동의 영향과 분리되지 않았다.
  - 개입: u 방향 gain 이 무작위 방향보다 대체로 크지만(a125 제외) 사전 등록 판정 기준이 없고 gain 의 의미 해석은 보류한다. "predictor 가 정보를 버린다/쓴다"는 결론은 내리지 않는다.
  - 한계: 모델 학습 seed 3개, 단일 학습 분포(ind), 천 마스크 재생성본(ACWM 대비 IoU 0.68, results.json repro_check.note).
- 다음 단계: /review-experiment 001_size_collapse_diagnosis (분석가 3명 독립 리뷰 -> reporter). 리뷰 전에는 위 해석을 논문 주장으로 쓰지 않는다. 별도로 shrink_ratio_tol 처리 방침(사람 결정), selectivity 키 확인, overnight_status.py 버그 수정, 체크포인트 업로드 여부(--ckpt).

## 2026-11-02 / 001_size_collapse_diagnosis / 리뷰 반영 (reporter)
- 보고서 reports/001_size_collapse_diagnosis.md 작성. 위 2026-11-02 항목은 수정하지 않았고, 리뷰(prior/stats/skeptic)가 지적한 정정 사항은 보고서 6절에 따로 정리함.
- 요약: 먼 크기의 open-loop 이동 방향은 3시드 일치. 그러나 D "사전 등록 기준 충족", "학습 크기로 수렴", "tf1 기울기 ~1", "3.99배"(평균끼리의 비 2.36), probe 수치의 정의 혼합, "선택성 키 못 읽음"은 리뷰에서 과장/부정확으로 지적됨. 게이트 shrink_ratio_tol=null 이고 승인 임계값이 코드에 연결되지 않아 사전 등록 판정은 단위 선택(사후 결정)에 달림. 분석가 의견은 합치지 않고 병기. 사람 결정 사항과 GPU 실험(대조군 시드당 약 5.5시간) 승인 필요 여부는 보고서 8절.

## 2026-11-02 / 002_single_size_sweep / 코드 작성 (미실행)
- 변경점: experiments/002_single_size_sweep/ 에 run.py(단계 CLI: check download manifest smoke cache train q1 q2 q3 aggregate selftest), src/exp002/*, config.yaml 구현 파라미터 추가(기존 사전 등록 값은 변경 없음; `prereg.thresholds_wired_in_code` 를 true 로, `prereg.holm_alpha`/`family_defs`/`q3.criterion_percentile` 를 문구에서 숫자/구조로 옮김), environment.yml, overnight.sh, overnight_status.py, README.md 작성. 001 의 drive/masks/data/extract 유틸은 파일 맨 위에 출처를 적어 복사/개작. 사전 등록 값은 PreregReader 로 aggregate.py 가 읽어 판정에 쓴다(README 3절 표). null 임계값은 not_evaluable. 재시도 상한 3회는 results/attempts.json 가드와 overnight.sh 에 반영.
- 이 머신에서 한 것: py_compile / 파이썬 3.9 문법 검사, 모든 단계 `--dry-run`, `python run.py selftest`(합성 데이터, CPU torch 2.15) 60 통과 0 실패 0 건너뜀, torch 없는 경우 시뮬레이션에서 41 통과 0 실패 2 건너뜀(attentive probe, hook/Q3/H-c 는 'skipped(no torch)'). 셀프테스트 수치는 결과가 아니다. 개발 중 셀프테스트가 잡은 오류: 잔차 경로를 거치지 않는 개입 hook(수정함).
- 못 돌린 검증: GPU 실행, 실제 데이터/체크포인트, 브랜치 train.py 호출, cache/q2 추출/q3 run()/smoke 의 실제 경로, rclone 다운로드, conda env create, 서버(python 3.9, 실제 torch)에서의 selftest. 12GB 메모리/시간은 미측정.
- 결과: 없음(코드만). 해석 없음.
- 재시도 카운트: 0 (실행 시도 없음). config 의 batch_size/num_hist/epochs/budget/torch_version 은 null 유지.
- 다음 단계: 사용자가 서버에서 README 5절 순서(selftest -> check -> smoke -> ...)로 실행하고 로그를 붙여 준다. GPU 학습은 별도 사람 승인 필요.

## 2026-11-02 / 002_single_size_sweep / 002 확정 기록 6 반영 (코드만, 미실행; 확정 기록 7 포함)
- 변경점 (사양: notes/hypotheses.md 확정 기록 6, 7). config.yaml: 조건 3개(single_ind 900 / multi_60_ind_160 3x300 / multi_60_ind_160_small 3x100, optional, run_order 마지막), `budget_mode: steps_equalized`, `train.budget` null 유지, `q1.primary_window: late` + `report_window: early`, `q2_diag`, `diag_4c`(2.0/5.0/2.0, 시드 4242, 반복 200은 impl default), 사전 등록 후 변경 항목 옆에 원래 값 주석(300, 1.0). 코드: budget.py(신규), diag.py(신규), trainrun.py(조건별 epoch 환산, train_budget.json, expandable_segments 제거), data.py(조건별 에피소드 수, 슬라이스 수 계산), evalgrid.py(fit_apply_cells), q2.py(출력 공간 probe, 진단 4c, 재학습 rollout probe 삭제), q3.py(에피소드 평균 곡선, tier 표시), aggregate.py(H_a early 보고, H_b rollout 은 출력 공간, cond_a_vs_b 층별은 layer 0 대비 + 4c 게이트, diag_4c 행, q2_layerwise, train_budgets, 탐색 방법 값), run.py(--include-optional, torch 버전 출력), selftest.py(60 -> 98 항목), environment.yml(torch 2.0.1+cu117), README 9절.
- 검증 (이 머신, 임시 venv: python 3.14, numpy 2.5.4, CPU torch 2.15.0): `python run.py selftest` 98 통과 0 실패 0 건너뜀(합성 데이터, 결과 아님); 모든 단계 --dry-run rc=0; py_compile 통과; python 3.9 문법(ast feature_version) 통과. 경계값(2.0/5.0 비율, 오차 2배) 테스트 포함.
- 미검증: 서버 환경(python 3.9, torch 2.0.1+cu117), 실제 브랜치 로더로 slice 수/steps_per_epoch 계산(num_pred=1, drop_last 가정), q2 추출의 next-frame 오차 계산과 활성 모멘트(실제 모델 shape), 브랜치/DINOv2 hub 코드의 torch 2.0.1 호환, GPU 실행 전부, 데이터 다운로드(ind 900).
- 결과: 없음. 해석 없음. 재시도 카운트 0.
