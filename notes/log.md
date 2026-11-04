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
- 2026-11-02 / 002 / smoke.py VramPoller 버그 수정: `self._stop=False`(Thread._stop 덮어씀) -> `self._halt=False`, run() 의 미초기화 _halt 해결; Thread 내부 속성 덮어쓰기는 src/exp002 에서 더 없음(grep); selftest 에 VramPoller 항목 추가. 서버 smoke 재실행 필요(미검증).
- 2026-11-03 / 002 / data.count_train_slices 누락 수정(내 이전 문자열 치환이 조용히 실패했음) + 정적 AST 점검(static_check.py)과 selftest 항목 추가: 모듈 별칭/from-import 이름 전수 점검 302건 누락 없음; selftest 104->105 통과 0 실패. 서버 재확인 필요(미검증: 브랜치 로더로 실제 slice 수).

## 2026-11-04 / 004_depth_truncation / 코드 작성 (미실행)
- 변경점: experiments/004_depth_truncation/ 신규 (config.yaml, run.py, README.md, CHANGES.md, src/exp004/{base,deep,stages,depthlib,figs,selftest}.py). 명세 notes/specs/exp4_5_spec.md 3절(실험 4)의 4-A(인코더 절단 k 격자 {0,2,...,12}, random-init, 라벨 셔플, raw pixel / mask baseline)와 4-B(predictor 조기 종료: (i) z 부분 은닉 probe 주지표, (ii) 실험 1 probe 적용, (iii) 다음 z MSE, rollout t+1/4/8)를 구현. 소실 지점(R² 가 직전 최고값보다 0.2 이상 첫 하락), k*, m*, split 별, `loss_ood_only`, `pilot` 필드를 가진 summary.json 을 만든다(실험 5 입력). 002 의 probes/evalgrid/stats/masks/model_io/cache/data/common 을 수정 없이 import(001/002/003 파일 수정 없음). 4-C(깊이 재학습)는 사용자 결정으로 생략.
- 확인한 것(고정 commit 5a640be 를 `git show` 로 읽음; 명세 §6): 패치 수는 256 이 아니라 196(인코더 입력 196px), predictor 에 출력 projection 없음(조기 종료 = block 1..m + 최종 norm), predictor depth 6(기본값), 인코더 `dinov2_vits14`(12 block / 384 는 hub 모델이 없어 런타임 확인으로 남김), 실험 1 probe 는 디스크에 저장되지 않아 인코더 토큰으로 다시 학습, Attention 이 CUDA 를 하드코딩. 상세는 CHANGES.md.
- 검증 (이 머신, 임시 venv: python 3.14.9, numpy 2.5.4, CPU torch 2.15.0): `python run.py selftest` 72 통과 0 실패 0 건너뜀(합성 데이터; 토이 ViT 와 `get_intermediate_layers` 일치, 토이 predictor 조기 종료가 `model.predict`(m=L)와 002 StreamHooks 캡처(모든 m)와 일치, 계획된 소실 깊이 k*=6 / m*=4 복구, 경계값 0.2, 판정 분기). torch 없는 환경 시뮬레이션: 41 통과 0 실패 2 건너뜀. 모든 단계 `--dry-run` rc=0, py_compile, 파이썬 3.9 문법(ast feature_version) 통과, pyflakes 에서 정의되지 않은 이름 없음. 셀프테스트 수치는 결과가 아니다.
- 미검증: 서버 환경(python 3.9, torch 2.0.1), 실제 DINOv2 hub 모델 경로, 실제 predictor/체크포인트 경로, GPU 실행 전부, 비용(전부 추정).
- 결과: 없음(코드만). 해석 없음. 재시도 카운트 0. 서버 파일럿 GPU 실행은 승인 대기(hypotheses.md 후보 I 체크리스트).

## 2026-11-04 / 005_inductive_bias_probe / 코드 작성 (미실행)
- 변경점: experiments/005_inductive_bias_probe/ 신규 (config.yaml, run.py, README.md, CHANGES.md, src/exp005/{base,branch,tasks,ibmetrics,heads,feats,verdicts,stages,selftest}.py). 명세 4절(실험 5): 실험 4 summary.json 으로 4.6 분기표에 따라 층/하위 실험 선택(`plan`), 평가 입력 2000개와 5-A(이진 라벨 과제 100개 x D_k 100개), 5-B(주파수 제한 무작위 사인파 합 100개; 사용자 결정), 5-D(1% 라벨, 최소 30개) 과제 생성, 002 와 같은 구조의 attentive head 를 과제마다 새로 학습, oracle(linear / 2층 MLP hidden 5)·면적 휴리스틱 head·random-init·무정보 모델 기준, R-IB/D-IB(쌍 그룹 all/ind/ood_any/ood_both), 5-C 분해, 20구간 calibration 과 45도선 평균 이탈, slice 별 선형 적합(PySR 은 설치 시), H-e/H-f/H-g 판정(supported/refuted/not_evaluable, 임계값은 config), 과제별 예측 저장. LoRA 는 사용자 결정으로 생략. 001/002/003 파일 수정 없음, 004 의 deep.py 를 import.
- 확인한 사실/발견: (1) 명세 정규화식에 무정보 모델 = i.i.d. 무작위 예측을 쓰면 이진 과제에서 D-IB 의 분모가 0 에 가깝다(참 state 모델과 무정보 모델의 D-IB raw 가 모두 약 0.5; 셀프테스트의 합성 데이터에서 관찰). 코드는 분모가 min_gap 미만이면 NaN 으로 두고 H-e 를 not_evaluable 로 하며, 상수 출력 모델 기준 점수를 score_alt 에 같이 저장한다. 원문 App. B 식은 확인하지 못했다. (2) 002 캐시의 train 부분에는 학습 레벨 3개뿐이라 5-A 의 D_k 는 "전 레벨 train split" 이 아니라 그 3개 레벨에서만 뽑는다. 상세는 CHANGES.md.
- 검증 (이 머신, 위와 같은 임시 venv): `python run.py selftest` 91 통과 0 실패 0 건너뜀(합성 데이터; 분기표 각 행과 조합, 004 의 summary 로 분기, 지표 brute force 대조, 참 state/면적-only/무작위 토큰을 실제 head·지표 코드에 통과, 판정 규칙, 파일 출력). torch 없는 환경 시뮬레이션: 68 통과 0 실패 1 건너뜀. 모든 단계 `--dry-run` rc=0(합성 summary 예시로 분기 확인), py_compile, 파이썬 3.9 문법 통과. 개발 중 셀프테스트가 잡은 것: 내 기대값 3건이 틀렸고 코드는 맞았음(무정보 토큰의 R-IB 는 높게 나올 수 있음, 크기와 상관된 면적 분위로 인해 참 state 토큰의 D-IB_area= 도 낮음, MLP oracle 은 학습 범위 밖에서 포화) -> 기대값과 설명을 고침. 셀프테스트 수치는 결과가 아니다.
- 미검증: 서버 환경, 실제 토큰 규모의 head 학습 시간(첫 head 실측 필요), 실제 모델에서 predictor 층 추출, 004 캐시 연결, PySR 경로.
- 결과: 없음(코드만). 해석 없음. 재시도 카운트 0.
- 사람 결정이 필요한 것: CHANGES.md 의 목록과 최종 보고서 참조(정규화 기준, 5-A 의 train 레벨 범위, loss_ood_only/k_star 정의, `branch.exp1_negative`, 파일럿 GPU 승인).

## 2026-11-04 / 002_single_size_sweep / 실행 환경 변경 서버 -> Colab (코드만, 미실행; 확정 기록 10)
- 변경점: colab/{exp002_colab.ipynb, colab_tools.py, train_resume.patch, requirements_colab.txt} 신규. src/exp002/drive.py 에 `copy_from_mount`/`find_mount_src`(rclone 없이 마운트된 Drive 에서 /content 로 복사, 완료 마커 + 파일 수 + 총 바이트 검증), run.py `download` 가 `EXP002_DATA_MODE=mount`/`EXP002_DRIVE_MOUNT_SRC` 에서 이를 사용. config.yaml: `train_impl.save_every_x_epoch` 5 -> 1(사용자 승인 운영 값, 사전 등록 아님), `colab.keep_ckpt_epochs` 추가. 사전 등록 값(시드, batch_size 24, num_hist 1, epochs 15, 임계값) 변경 없음. README 10절, hypotheses.md 확정 기록 10(002 환경 변경) 추가.
- 사용자 결정 이력(같은 날, 시각순): (1) 먼저 "resume 패치를 쓰지 말자"는 지시가 있어 패치를 삭제하고 미완 run 을 처음부터 재시작하는 구조(`quarantine`)로 바꿨다. (2) 이후 사용자가 (1)이 실수였다고 정정하고 "체크포인트에서 재개"를 명시적 결정(train.py 패치 승인 포함)으로 내렸다. 이 정정을 최종으로 반영: `colab/train_resume.patch` 복원(상위 11ba488 + `training.epochs` 총 목표 epoch 해석), 노트북 5번 셀에서 `git apply --reverse --check` 후 적용 + 패치 sha256 을 WORK_ROOT 에 기록, `colab_tools.py repair` 복구 로직을 `prepare` 로 통합(읽히는 체크포인트 재개 / 깨진 latest 는 최신의 읽히는 epoch 파일로 복구 / 쓸 수 있는 것이 없을 때만 DINO_TRAIN/quarantine/<run>__<시각>/ 로 이동 후 처음부터), overnight.sh 가 매 train 시도 전 `prepare` 호출(실패 시 rc 97), `trainrun.train` 은 패치 마커가 없는 train.py 에서는 체크포인트가 있는 run 의 학습을 거부. save_every_x_epoch=1 은 재개에 쓰이므로 의미가 있다(앞서 적은 "가치가 낮다" 문구는 철회). (3) 번호 확정 기록 9 -> 10 정정(후보 I 의 9 와 중복). (4) 노트북 repo URL 은 origin 과 같고 ACWM URL 은 config code.repo_url 과 같음; 인증 없는 `git ls-remote` 로 둘 다 조회돼 공개로 보임(저장소 설정은 미확인). (5) 데이터 폴더: 사용자 전달 정보(Drive 메타데이터 확인)로 바로가기 `dino-wm-videos`(id 17M0MZGSDPzu7CCT3ez3ABMyZGjavTfx3) 아래 `[data]/[cloth-sim] cloth_move` 가 config root_folder_id 폴더. 노트북 DATA_SRC 기본값을 `/content/drive/MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move` 로 하고, 없으면 MyDrive/dino-wm-videos 아래 깊이 3 까지 split 디렉터리(ind_train 등)를 가진 폴더를 탐색(마운트 형태 미검증; 대괄호/공백 경로는 pathlib 만 사용, selftest 로 확인).
- 확인한 사실: 고정 commit 5a640bed 의 train.py 는 resume 이 불완전하고(action/proprio encoder 재초기화, optimizer 상태 미복원) `training.epochs` 를 추가 epoch 로 해석한다. 재개는 데이터 순서/난수 상태가 복원되지 않아 중단 없는 run 과 bit 단위로 같지 않을 수 있고, cloth/torch 2.0.1/Colab/GPU 에서 검증하지 못했다(추정).
- 검증(이 PC, CPU, python 3.14, numpy 2.5.4, torch 2.15 CPU): `python run.py selftest` passed=136 failed=0 skipped=0(합성 데이터; 결과 아님; static_check 포함; 패치 적용과 patched `run()` 루프식의 epoch 범위 새 학습 1..15 / epoch 5 후 재개 6..15 포함), py_compile, `bash -n overnight.sh`.
- 미검증: Colab/Drive 마운트 실제 동작, 공유 폴더 접근 경로, pip 환경 설치(Python 3.9 + torch 2.0.1+cu117), overnight.sh 의 `prepare` 호출을 실제 학습 루프에서 돌려 본 것, 패치된 train.py 의 실제 재개, Drive FUSE 속도와 `flock`, 체크포인트 크기, Colab 세션 시간.
- 결과: 없음. 해석 없음. 재시도 카운트 0. 서버 부분 출력은 재사용하지 않음(처음부터).

## 2026-11-04 / 002_single_size_sweep / Colab 데이터 복사 속도 개선: 아카이브 모드 (코드만, 미실행)
- 배경: 사용자 보고 — 마운트된 Drive 에서 작은 mp4 를 하나씩 읽는 복사(`run.py download`, `EXP002_DATA_MODE=mount`)가 너무 느리다(원인은 FUSE 지연으로 추정). 압축 파일 하나를 받아 푸는 방식을 요청.
- 변경점: src/exp002/drive.py — `prepare_data`(완료 split 은 건너뜀 -> `WORK_ROOT/cloth_sim_data.tar|.zip` 있으면 `extract_archive` -> 없으면 16 스레드 병렬 `copy_from_mount` 후 검증 통과 시 `make_archive` 로 Drive 에 한 번만 생성), 공통 검증 `_finish_split`(split 별 mp4 수 == metadata 길이 == 기대값, 총 바이트 마커 기록), 진행률 로그, `*.part` 임시 파일. 아카이브 추출은 최상위 레이아웃 변형(`cloth_sim/<split>/`, `<split>/`, `<임의>/<split>/`)을 처리하고 절대 경로/`..`/링크를 거부, 잘리거나 에피소드/split 이 빠진 아카이브는 완료 처리하지 않음(rc 3). 아카이브는 임시 이름으로 쓰고 rename. run.py `download` 가 `prepare_data` 를 호출. 노트북 6번 셀에 `EXP002_WORK_ROOT`, 7번 셀 설명 갱신. README Colab 절에 사용법과 직접 tar/zip 을 올리는 절차 추가. 사전 등록 값 변경 없음.
- 검증(이 PC, CPU, 임시 디렉터리의 가짜 데이터): `python run.py selftest` passed=158 failed=0 skipped=0(static_check 포함, 359 참조). 아카이브 항목: 생성(.tmp 없음), 추출과 검증 및 바이트 기록, 재실행 건너뜀, 아카이브 우선 사용, 아카이브 없을 때 병렬 복사 후 생성 및 다음 세션은 아카이브만 사용, `EXP002_MAKE_ARCHIVE=0`, 레이아웃 3종(tar/zip), split 없음/접두사 혼합/경로 탈출(`..`, 절대)/심볼릭 링크/잘린 아카이브/에피소드 누락/split 누락/비아카이브 파일 거부, 중단 후 재실행(`.part` 와 짧은 파일). 합성 데이터이며 결과가 아니다.
- 미검증: 실제 Drive FUSE 에서의 속도(향상 정도는 측정하지 못함, 추정), 실제 데이터 크기와 /content 디스크 여유, 실제 아카이브 생성 시간, 마운트 형태.
- 결과: 없음. 해석 없음. 재시도 카운트 0.

## 2026-11-04 / 002_single_size_sweep / Colab 환경 수정: pkg_resources 누락 (코드만, 미실행)
- 배경: exp002_multi_60_ind_160_s0 train 이 3회 모두 5~6초 만에 `ModuleNotFoundError: No module named 'pkg_resources'` 로 실패(07:50:17, 07:51:23, 07:52:28), STOP. 이후 같은 셀을 다시 실행한 07:57:56 시도는 다른 메시지(`FAILED. Paste the log file ...`)로 실패했고 그 로그는 아직 읽지 못했다. 로컬에는 try 로그가 없어 traceback 의 import 위치는 확인하지 못했다.
- 변경점: colab/requirements_colab.txt 에 `setuptools<81` 추가(원인은 추정: uv venv 에는 setuptools 가 기본 설치되지 않고, 새 setuptools 는 pkg_resources 를 제거했을 수 있음 - 후자는 미확인). 노트북 4번 셀: venv 마커를 requirements 해시로 바꿔 requirements 가 바뀌면 기존 venv 에 재설치(venv 가 없을 때만 `uv venv`), import 점검에 `pkg_resources` 추가. 사전 등록 값 변경 없음.
- 검증: 노트북 JSON 파싱 및 4번 셀 구문 컴파일만 확인. Colab 설치/학습은 미검증.
- 결과: 없음. 해석 없음. 재시도 카운트: 위 3회는 소진, 환경 수정 후 새 시도는 사람 승인 뒤 시작.

## 2026-11-04 / 002_single_size_sweep / Colab 재실행 실패 원인 확정, attempts.json 실행 거부 제거 (코드만, 미실행)
- 원인 1(확정, Drive 의 logs/train_multi_60_ind_160_20261104-075013.log traceback): `train.py` line 6 `import wandb` -> wandb 0.13.1 `sdk/internal/internal_api.py` line 30 `from pkg_resources import parse_version` -> ModuleNotFoundError. 앞 항목의 "어느 모듈인지 미확인"을 정정한다. setuptools 상한 `<81` 의 근거(새 setuptools 의 pkg_resources 제거)는 여전히 추정.
- 원인 2(확정, 같은 폴더의 07:59, 08:00, 08:04 로그): 07:57 이후의 `FAILED. Paste the log file ...` 실패는 새 오류가 아니라 `attempts_check` 의 거부다. Drive 의 results/attempts.json 에 `train:exp002_multi_60_ind_160_s0: 3` 이 남아 있어 train.py 를 시작하기 전에 RuntimeError 로 끝났다. 따라서 setuptools 수정은 아직 한 번도 검증되지 않았다.
- 원인 3(근거: 노트북 diff; 덮어쓴 경위는 추정): 저장소는 9a6733e 로 pull 됐지만 실행된 노트북의 4번 셀은 수정 전 코드였고 설치 출력도 없었다. 작업 트리의 노트북도 옛 셀로 되돌아가 있었다(열려 있던 편집기 버퍼 저장으로 추정). 이 세션의 venv 에 setuptools 가 없을 가능성이 높다.
- 변경점: (1) `attempts_check` 와 호출부, config.yaml `run.max_attempts` 삭제. results/attempts.json 은 연속 실패 기록으로만 남고(results.json 에 그대로 포함) 실행을 막지 않는다. 사용자 허용("불필요하면 제거")에 따른 판단: 이 영속 카운터는 원인을 고친 뒤에도 항목을 손으로 지워야 풀려 이번 혼란을 만들었다. (2) overnight.sh 의 단계당 3회 상한(ATTEMPTS)은 유지: 결정적 실패가 무인으로 무한 반복되는 것을 막는다. 상한은 실행 단위이며, 사람이 원인을 본 뒤 다시 시작하면 새로 3회다. CLAUDE.md 문구는 고치지 않았다. (3) overnight.sh: Colab venv(VIRTUAL_ENV 설정 시)에서 시작할 때 pull 된 requirements 를 `uv pip install` 로 동기화하고 `import pkg_resources, wandb` 를 확인, 실패하면 학습 시도 없이 STOP. (4) overnight.sh `lastline` 이 run.py 의 일반 FAILED 줄을 건너뛰어 실제 예외 줄을 보고. (5) 노트북 4번 셀 수정을 다시 적용. README, run.py docstring, colab_tools.py 주석, requirements 주석 갱신. 사전 등록 값 변경 없음. config.yaml sha256 은 바뀐다.
- 검증(이 PC, CPU, python 3.9.13): `python run.py selftest` passed=131 failed=0 skipped=3(건너뜀: resume 패치 epoch 범위, attentive probe, hooks/Q3; 합성 데이터이며 결과 아님), `bash -n overnight.sh`, `lastline` 을 실패 로그 형태로 확인, 동기화 블록을 stub uv/python 으로 4경우(정상, uv 실패, import 실패, venv 없음) 확인, 노트북 JSON 파싱과 4번 셀 구문 컴파일.
- 미검증: wandb 0.13.1 + setuptools<81 의 실제 설치와 import(로컬 설치 테스트는 권한 정책으로 거부되어 하지 않음), `import wandb` 다음 단계의 train.py, 실제 uv 동기화, GPU 학습.
- 결과: 없음. 해석 없음.

## 2026-11-04 / 002_single_size_sweep / 재시도 횟수 제한 제거(사용자 지시), 셀 9 로그 표시 수정 (코드만, 미실행)
- 사용자 결정: "이 retry limit 을 없애줘". overnight.sh 의 단계당 3회 상한(ATTEMPTS, STOP, `fail`)을 제거하고 실패한 단계를 `RETRY_SLEEP` 간격으로 무제한 재시도한다. 앞 항목의 "3회 상한 유지"를 대체한다. 결정적 실패는 저절로 멈추지 않으므로 사람이 `pkill -f "bash overnight.sh"` 로 멈춰야 한다. CLAUDE.md 의 "재시도 최대 3회, 3회 실패하면 멈추고 보고"는 고치지 않았고, 이 실험 002 의 overnight.sh 만 사용자 결정에 따른 예외다. 환경 점검 실패의 즉시 STOP(uv 동기화, import 확인)은 재시도 상한이 아니라서 유지.
- 사용자가 "같은 오류"로 보고한 출력의 판독: 셀 9 출력의 `attempt 3 failed`, `STOP` 줄의 시각은 08:05:08 로, pid 10066 실행(08:12:27 시작)보다 앞선 이전 실행의 기록이다. overnight.log 가 누적 기록(append)이고 셀이 시작 직후 `tail` 을 찍어서 생긴 표시다. 수정: 노트북 9번 셀이 시작 시점의 로그 크기를 기억하고 그 뒤의 줄만 출력(`log_tail`).
- 08:12 실행에서 Drive 로 확인한 것: `import pkg_resources, wandb` 점검 통과(STOP 없음), `train: attempt 1` 시작, try1 로그에 traceback 없음, run 폴더에 `.hydra/` 와 `train/train_e00001_b0.png`(08:12:52) 생성. 노트북 4번 셀 출력에 `- setuptools==82.0.1`, `+ setuptools==80.10.2`, 그리고 pin to Setuptools<81 을 권하는 경고. 조회 시점까지 실패 기록은 없음. 학습이 지금도 진행 중인지, epoch 1 을 완료했는지는 확인하지 못했다(체크포인트 없음, Drive 는 스냅샷). `uv` 동기화 때마다 `~ decord==0.6.0` 재설치가 찍히며 원인은 미확인.
- 진행 중인 실행은 이미 시작된 옛 overnight.sh(3회 상한)로 돈다. 새 규칙은 다음 시작부터 적용.
- 검증(이 PC, CPU, python 3.9.13): `python run.py selftest` passed=131 failed=0 skipped=3(건너뜀: resume 패치 epoch 범위, attentive probe, hooks/Q3; 합성 데이터이며 결과 아님), `bash -n overnight.sh`, `retry_step` 추출 실행(4회 실패 후 5번째 성공, 마커 있으면 건너뜀), 노트북 `log_tail` 실행(이전 실행 줄 제외), 노트북 4, 9번 셀 구문 컴파일.
- 결과: 없음. 해석 없음.

## 2026-11-04 / 002_single_size_sweep / Drive 끊김으로 학습 중단, 재개 판정 버그 수정 (코드 수정, 재시작된 run 은 이전 코드로 진행 중)
- 경과(UTC): 08:12 run s0 시작, 08:21 epoch 1 저장. 노트북 셀 10 `drive.flush_and_unmount()` 가 실행되어(경로는 "모두 실행" 중 셀 9 중단으로 추정) Drive 가 끊겼고, 08:29:06 train.py rc=1, 시도 2, 3 은 logs/ 를 찾지 못해 실패, 08:31:06 STOP. 로컬 폴더로 변한 /content/drive 를 옮기고 다시 마운트.
- 08:39:40 `prepare` 가 `quarantined` 를 반환해 epoch 1 체크포인트와 epoch_logs.jsonl 이 Drive train_runs/quarantine/exp002_multi_60_ind_160_s0__20261104-083940/ 로 이동(삭제 없음). 08:39:42 s0 가 epoch 0 부터 새로 시작(재개 아님). 따라서 현재 s0 는 재개가 섞이지 않은 run 이다.
- 원인(확정, Colab 에서 실제 체크포인트로 확인): `_default_loadable` 이 ACWM 경로 없이 torch.load -> `ModuleNotFoundError: No module named 'models'`, 경로를 넣으면 같은 파일이 정상 로드. 예외를 삼켜 "읽을 수 없음"으로 처리했다. selftest 는 가짜 loadable 만 써서 잡지 못했다. 앞선 "재개 구현" 항목의 재개 경로는 이 버그 때문에 실제로 동작할 수 없었다.
- 변경점: colab_tools.py `_default_loadable` 이 ACWM_PHYS_SRC 를 sys.path 에 추가. `_is_loadable` 추가: ImportError 는 환경 문제이므로 그대로 올려 격리하지 않고(overnight.sh 에서 rc 97), 그 외 로드 실패는 이유를 stderr 에 출력. selftest 항목 2개 추가. 사전 등록 값 변경 없음.
- 알려진 문제: 셀 10 주석과 안전장치는 아직 고치지 않았고, 편집기 저장으로 작업 트리 노트북의 셀 9 수정(`log_tail`)이 되돌아가 있어 이번 커밋에서 노트북은 제외한다. 재개 자체(패치된 train.py 가 체크포인트에서 이어 학습)는 아직 실제로 실행된 적이 없다.
- 결과: 없음. 해석 없음.
