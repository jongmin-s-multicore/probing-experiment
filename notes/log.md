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

## 2026-11-04 / 002_single_size_sweep / run s0 세션 끊김 후 첫 실제 재개 (운영 기록, 결과 아님)
- 경과(UTC): 08:39 시작한 s0 가 epoch 5 저장(09:23:38) 뒤 멈춤. Drive 의 마지막 기록은 09:26:06 이고 overnight.log 에 실패 줄이 없어 세션/런타임 종료로 추정(원인 미확인). 10:55 셀을 다시 실행, `colab_tools.py prepare` -> `"action": "resume"`, trainrun `resuming from checkpoints/model_latest.pth`, 10:56:08 `train_e00006_b0.png` 다시 기록. d8206d6 수정 후 처음 동작한 재개다.
- 주의: s0 는 epoch 5 와 6 사이에 재개가 한 번 들어간 run 이다. 데이터 순서와 난수 상태는 복원되지 않아 중단 없는 run 과 bit 단위로 같지 않을 수 있다(추정). 리뷰와 결과 해석 때 이 사실을 고려해야 한다.
- 미확인: 재개 후 첫 epoch 완료와 그 손실이 이어지는지(조회 시점 11:02 UTC 에 아직 없음). 값은 운영 점검용이며 결과가 아니다.
- 추가 확인(11:09 UTC, epoch_logs.jsonl): 재개 후 epoch 6 완료(11:04:38), train_loss 0.7685, val_loss 0.6784. epoch 1~5 의 train 2.19, 1.78, 1.46, 1.19, 0.96 / val 1.94, 1.60, 1.31, 1.06, 0.85 에서 끊기지 않고 이어져 위의 "재개 후 첫 epoch" 미확인 항목을 해소한다. 옵티마이저 3종 상태 로드도 train.log 에 기록됨. 운영 점검용 값이며 결과가 아니다.

## 2026-11-05 / 002_single_size_sweep / Q3 제거 검정 not_removed 처리 결정 (운영 기록, 결과 아님)
- 관찰: multi_60_ind_160 s0, s1 의 Q3 로그와 q3.json 에서 INLP, LEACE 가 middle, last 층 모두 `removed=False`. s2 와 single_ind 는 이 기록 시점에 미완료. 수치는 Drive `exp002_work/results/<run>/q3.json` 참고(results.json 미생성이라 여기에 옮기지 않음).
- 영향: 주 방법 INLP 가 s0, s1 에서 실패했으므로 aggregate.py 규칙상 H_c:subspace_removal 은 s2 결과와 무관하게 not_evaluable. steering 검정은 이 플래그와 별개.
- 원인 해석(추정, 미검증): INLP 10회로는 선형 정보도 다 지우지 못했고, LEACE 는 선형 정보만 지워 MLP 로 읽히는 비선형 정보가 남은 것으로 보인다. 코드 오류 근거는 찾지 못했다.
- 사용자 결정: `not_evaluable` 그대로 보고하고 후속 분석 여부는 /review-experiment 에서 논의. `q3_impl.removal_check.mlp_r2_after_max`, `inlp_iterations` 는 변경하지 않으며 진행 중인 실행도 건드리지 않는다. 제거 효과값은 H-c 근거로 쓰지 않는다.
- 리뷰 때 함께 볼 것: multi s0 는 epoch 5, s1 은 epoch 9 에서 체크포인트 재개(train.log 확인). optimizer 상태는 복원됐으나 중단 없는 run 과 동일하다는 보장은 없다.

## 2026-11-05 / 002_single_size_sweep / Colab 실행 완료, results.json 확보 (결과 기록, 해석 전)
- 변경점: 학습 6개(multi_60_ind_160, single_ind x 시드 0,1,2), cache, Q1, Q2, Q3, aggregate 완료(overnight 05:49 UTC). 보조 조건 multi_60_ind_160_small 은 실행 안 함. 런타임 해제로 로컬 파일이 사라져 06:04 UTC 에 Drive 입력으로 aggregate 재실행, 판정 행 수는 첫 실행과 같다(14행: pass 6, fail 3, not_evaluable 2, decided 2, exploratory 1). Drive `exp002_work/results/results.json` 을 experiments/002_single_size_sweep/results.json 으로 복사(sha256 079ff4c1...).
- 결과(results.json decision_table): H-a 보간 linear pass, 보간 attentive pass, 외삽 linear fail, 외삽 attentive fail. H-b last_block pass, last_rollout_step fail. H-c steering pass, subspace_removal not_evaluable(not_removed, 6셀 모두). 조건 (a) 대 (b) 패밀리 2개 pass(차이 검출, 방향 주장 아님). 4c: single_ind medium, multi_60_ind_160 large. 수치는 results.json 참고.
- 해석: 하지 않음. /review-experiment 대기.
- 리뷰 때 볼 것: (1) multi s0, s1 재개 run. (2) 002 학습 6개 모두 패치로 `val()` 의 `@torch.no_grad()` 줄이 빠진 train.py 사용(가중치 영향 없음은 추정, 미검증). (3) results.json 의 environment, git dirty 는 학습 런타임이 아니라 aggregate 재실행 런타임 값. (4) 4c large 규칙 문구는 (a) 층별 결과에 관한 것인데 large 는 (b) 행에서 나왔다.

## 2026-11-05 / 002_single_size_sweep / 리뷰 3건 완료, 보고서 작성 (reporter)
- 변경점: notes/reviews/002_prior.md, 002_stats.md, 002_skeptic.md 완료. reports/002_single_size_sweep.md 작성(코드와 실험 파일 수정 없음).
- 결과(results.json 값): 세 분석가 모두 판정 14행의 status 와 Holm 계산이 규칙대로임을 확인. H_b:last_rollout_step 시드별 drop 0.2913/0.2953/0.3205(평균 0.3024, 임계 0.2 초과), (a) 대 (b) rollout drop 차이 0.4221, 조건 (a) drop 0.7385/0.7059/0.7289. H_c:steering 6칸 모두 무작위 100방향 95 백분위 밖(p_rank 0.0099 = 하한). cond_a_vs_b:last_block 차이 -0.00817. 4c: single_ind medium(a060 최대 이동비 4.912/4.662/4.657), multi large(a060 5.037/5.054/5.301).
- 해석(보고서 5, 6절, 추정 포함): 분석가가 갈린 곳은 H-a 보간 pass 의 의미(stats 강한 지지, skeptic 근거 아님, prior 부분), steering/last_block 을 무효로 볼지 약한 증거로 볼지. rollout 하락이 정보 소실인지 판독기 전이 실패인지는 세 명 모두 구분 불가. H_c:subspace_removal 은 사용자 결정대로 not_evaluable, 제거 효과값은 근거로 쓰지 않음. Drive 전용 파일(q3.json, cells.npz 등)로 재계산하지 못한 항목이 있음(보고서 7절). 사람 결정 5건은 보고서 10절.

## 2026-11-07 / 003_train_dist_dependence / single_a248 s2 런타임 끊김과 재개 (운영 기록, 결과 아님)
- 경과(UTC): 07:45 s2 시작, 08:04 epoch 1 저장. 08:12 쯤 이후 Drive 기록이 끊겨 런타임 종료로 추정(원인 미확인). 사용자가 셀을 다시 실행해 08:52:12 train.log 에 `Resuming from epoch 1`, 옵티마이저 3종 상태 로드 기록. 13:15 epoch 15 저장, DONE 생성. 뒤이어 mix_adjacent s0, s1, s2 진행.
- 영향: 저장 안 된 epoch 2 진행분만 다시 학습했고 저장된 epoch 은 잃지 않았다. 약 40분 지연.
- 재개 전후 손실(운영 점검용 값, train.log): epoch 1 train 1.9258 / val 1.5090, epoch 2 train 1.2400 / val 0.9679.
- 리뷰 때 볼 것: s2 는 epoch 1 과 2 사이에 재개가 들어간 run 이다. 데이터 순서와 난수 상태는 복원되지 않아 중단 없는 run 과 bit 단위로 같지 않을 수 있다(추정).
- 결과: 없음. 해석 없음.

## 2026-11-07 / 003_train_dist_dependence / Colab 실행 완료, results.json 확보 (결과 기록, 해석 전)
- 변경점: 12개 run 의 학습, 추출, metrics 와 baseline, aggregate 완료(03:21 UTC). Drive `exp003_work/results/results.json` 을 experiments/003_train_dist_dependence/results.json 으로 복사(sha256 4b86c6de..., 500290 바이트). missing_runs 없음.
- 결과(results.json verdict): H-H1 not_evaluable(single_a026, single_a248 모두 not_evaluable). H-H2 not_evaluable(mix_wide pass, mix_adjacent not_evaluable, pattern mixed_or_opposite).
- 데이터 문제(확인됨): exp003_mix_adjacent_s0, exp003_single_a026_s1 은 900 에피소드 전부 area_ol 0 이고 slope_ol 0.0. 표본 18 에피소드의 frames_sample `ol` 은 에피소드와 시점에 관계없이 같은 이미지였다(픽셀 표준편차 mix_adjacent_s0 0, single_a026_s1 최대 0.5). 천, 공, 바닥이 없다. 두 run 의 epoch_logs 에서 val_img_mse_reconstructed 가 epoch 2 에서 3 사이에 뛰어올라 epoch 15 까지 유지된다(mix_adjacent_s0 0.0030 -> 0.0787, single_a026_s1 0.0015 -> 0.0333, 운영 점검용 값). 따라서 추출 전, 학습 중 decoder 에서 발생한 것으로 보인다(추정). 원인 미확인. 최종 val_loss 는 같은 분포 시드보다 높다(mix_adjacent 0.2658 대 0.1894, 0.1890; single_a026 0.1608 대 0.1274, 0.1275).
- 추가 확인: 003 나머지 10개, 002 6개 run 은 같은 지표가 끝까지 0.0012~0.0028 이고, 001 3개 run 은 마지막 epoch train 지표만 있으며 0.0007 수준이다. epoch 별 값은 experiments/003_train_dist_dependence/analysis/decoder_recon_by_epoch.json.
- 리뷰 때 볼 것: (1) 위 두 run 의 slope 0.0 과 c null 은 측정 실패일 수 있다. (2) 재개 run: single_a026_s0(A100, L4 혼합), single_a248_s2. mix_wide 3개는 train_env 기록 없음.
- 해석: 하지 않음. /review-experiment 대기, 재학습 여부는 사람 결정.

## 2026-11-07 / 003_train_dist_dependence / 리뷰 3건 완료, 보고서 작성 (reporter)
- 변경점: notes/reviews/003_prior.md, 003_stats.md, 003_skeptic.md 완료. reports/003_train_dist_dependence.md 작성(코드와 실험 파일 수정 없음). 사용자 결정(리드 전달): 붕괴 2개 run 은 재학습하지 않고 문제를 명시한 채 리뷰한다.
- 결과(results.json 값): H-H1 not_evaluable, H-H2 not_evaluable(mix_wide pass: 9쌍 차이 0.3406~0.6542, mix_adjacent not_evaluable). 붕괴 run(mix_adjacent_s0, single_a026_s1)은 기울기 0.0, CI [0,0] 측정 실패. 정상 run 기울기: single_a248 0.5269/0.5774/0.5518, mix_wide 0.8042/0.6508/0.8005, mix_adjacent(s1, s2) 0.5023/0.4153, single_a026(s0, s2) 0.0022/0.1447, baseline(001) 0.2366/0.3102/0.1500.
- 해석(보고서 5, 6절, 추정 포함): 세 분석가 일치: 판정 계산은 규칙대로, 붕괴 run 수치는 쓰면 안 됨, 두 run 을 어떤 값으로 바꿔도 전체 판정 불변(stats 민감도, skeptic 규칙 계산), baseline 이 001 값이라 교란. 갈린 곳: mix_wide pass 의 의미(prior 문헌 방향 일치지만 약함 / stats 규칙 통과이지 유의 검정 아님 / skeptic 혼합 없는 single_a248 도 기준 통과, 지표 효과), 낮은 a026 기울기가 실제 신호인가(stats 신호 / skeptic 해석 미정), 붕괴 원인 후보(모두 미확인). 표시만 한 불일치: hypotheses.md:250 대 verdict.py:125(외삽 포함 9점 기울기), stats 의 2 대 3 순열 p 문장, prior 의 val_z_pred 범위 서술 대 csv, single_a026_s0 이력 기록.
- 추가 사실(리드가 Drive 검색으로 확인): 삭제된 model_2.pth, model_3.pth 가 Drive 휴지통에 있음(mix_adjacent_s0 폴더로 보이는 파일 포함). 일반 보존 30일은 추정, 붕괴 부검이 필요하면 기한 전 복원. 사람 결정 3건은 보고서 8절.

## 2026-11-07 / 007_decoder_collapse_autopsy / 체크포인트 검사 실행 완료 (결과 기록, 해석 전)
- 변경점: notes/hypotheses.md 에 "007 진단 질문"(검사와 판단 기준을 결과 보기 전에 기록) 추가. experiments/007_decoder_collapse_autopsy/{config.yaml, run.py, README.md, results.json, smoke_results.json, figures/} 작성. CPU 만 사용(conda `lab07`, torch 2.0.1+cpu), 재학습 없음. Drive 는 `rclone copy/lsf --drive-trashed-only` 로 읽기만 했다(복원, 삭제, 쓰기 없음). 체크포인트 11개(붕괴 5: mix_adjacent_s0 model_2/3, single_a026_s1 model_2/3/4. 대조 6: mix_adjacent_s1/s2, single_a026_s2 의 model_2/3)를 저장소 밖 scratchpad 로 내려받았고, /tmp 가 tmpfs 3.8 GB 라 원본은 predictor, decoder, action/proprio encoder 모듈과 optimizer 요약만 담은 slim 으로 바꾸고 로컬 원본은 지웠다(sha256 은 results.json). 소요: 분석 1691 초. git commit 안 함.
- 코드 사실(5a640bed): 디코더 VQVAE(Conv + ResBlock x4 + ConvT 2단 x2, ReLU, BatchNorm 없음, quantize False). AMP, grad clip, lr scheduler 코드 없음. decoder Adam lr 3e-4. 디코더 입력 z.detach() 라 재구성 손실은 predictor 로 역전파되지 않음. val 의 이미지 지표는 첫 valid 배치 1개. 상세와 줄 번호는 results.json `A_code_facts`.
- 결과(results.json `criteria_table`, 입력 54 프레임 = 9 크기 x 에피소드 2 x 시점 3. 5개 run 의 frames_sample.npz 의 real 이 모두 같아 `own` 과 `common` 집합 값이 동일):
  - (1) NaN/Inf: 비해당. 붕괴 5개, 대조 6개 모두 가중치, optimizer 상태, 활성화, 출력에서 0개.
  - (2) 입력 의존성 소실 층: 해당. 세 체크포인트(mix_adjacent_s0 model_3, single_a026_s1 model_3, model_4)에서 가장 처음 기준 미만이 되는 층은 `upsample_b.blocks.7`(upsample_b 의 첫 ConvTranspose 다음 ReLU). 입력 간 std: mix_adjacent_s0 model_3 0.0, single_a026_s1 model_3 4.4e-6, model_4 4.2e-6, 대조 3개 평균(epoch 3) 4.9e-2. 이 층 이후 27개 층이 기준 미만. 바로 앞 ConvTranspose(`upsample_b.blocks.6`) 출력 std 는 mix_adjacent_s0 7.59, single_a026_s1 11.68 이고 대조 평균 0.266. epoch 2 두 run 은 기준 미만인 층 없음. single_a026_s1 model_4 는 대조 epoch 3 값과 비교(대조에 model_4 없음).
  - (3) 전부 비활성 ReLU: 부분 해당. mix_adjacent_s0 model_3 의 `upsample_b.blocks.7` 입력이 모든 입력, 위치에서 <= 0 (비율 1.0, 채널 전부 비활성). 대조 0.825, 0.847, 0.832. single_a026_s1 model_3, model_4 는 정의(전부 <= 0) 엄격 적용 시 비해당: 비율 0.99998647, 채널 단위 비활성 0.9896 (같은 층, 대조 같은 값).
  - (4) epoch 3 가중치 변화 10배 이상: 비해당. 디코더 최대 비율 6.10 (mix_adjacent_s0), 6.09 (single_a026_s1), 둘 다 `upsample_b.blocks.0.bias`(상대 변화 0.753, 0.752 대 대조 0.100, 0.123, 0.100). predictor 최대 1.17, 1.10, action_encoder 0.85, 0.90, proprio_encoder 1.00. 같은 분포 대조만 써도 해당 층 없음.
  - (5) 어느 것도 해당 없음: 비해당((2), (3)이 해당).
  - 체크포인트로 구분 불가: epoch 3 도중 정확한 시점, 단일 큰 gradient 여부.
- 추가 관측(판단 기준 외): 출력 이미지 입력 간 std (common 54프레임; 실제 프레임 0.1003): mix_adjacent_s0 epoch 2 0.0998 -> epoch 3 0.0000, single_a026_s1 0.0862 -> 0.0002 (epoch 4 0.0003). 실제 프레임 대비 MSE([-1,1] 스케일): mix_adjacent_s0 0.0031 -> 0.0855, single_a026_s1 0.0129 -> 0.0892 (epoch 4 0.0892), 대조 mix_adjacent_s1/s2 epoch 3 0.0029/0.0028, single_a026_s2 epoch 3 0.0112. predictor 출력 입력 간 std(proprio, action 은 0 벡터 고정): epoch 3 mix_adjacent_s0 0.4793 대 대조 0.4793/0.4814, single_a026_s1 0.4140 대 single_a026_s2 0.4118. decoder_optimizer step 은 모든 체크포인트에서 epoch x 4125 (붕괴와 대조 같음), lr 3e-4. mix_adjacent_s0 model_3 의 decoder_optimizer exp_avg 가 21개 파라미터에서 정확히 0 (대조 및 single_a026_s1 은 0개). exp_avg_sq 평균의 대조 대비 배율(epoch 3) 최대: mix_adjacent_s0 `upsample_b.blocks.6.weight` 1861배, single_a026_s1 같은 파라미터 50271배. 두 run 모두 `upsample_b.blocks.6.weight` norm: epoch 2 -> 3 이 15.8 -> 18.2, 15.4 -> 17.1 (대조 s1 15.97 -> 18.07).
- 이상한 점/한계: (a) 두 붕괴 run 의 train_env 는 n_resumes 0, A100, patched train.py. 대조 3개도 같음. (b) 첫 fetch 실행이 lsf 형식 문자열 오타(`pstit`)로 실패해 수정 후 재실행(코드 버그, 3회 재시도 소진 아님). 처음 smoke 용으로 만든 mix_adjacent_s1 model_2 slim 은 메타가 없어 지우고 다시 만들었다. (c) predictor 검사는 행동, proprio 를 0 으로 고정해 실제 행동 입력 결과가 아니다. (d) smoke_results.json 은 smoke(체크포인트 1개, 프레임 6장) 기록으로 남김. (e) hub 캐시를 처음 한 번 잘못된 경로에 받아 DINOv2 가중치를 두 번 내려받음(영향 없음).
- 해석: 하지 않음. /review-experiment 대기.

## 2026-11-07 / 007_decoder_collapse_autopsy / 리뷰 3건 완료, 보고서 작성 (reporter)
- 변경점: notes/reviews/007_prior.md, 007_stats.md, 007_skeptic.md 완료. reports/007_decoder_collapse_autopsy.md 작성(코드와 실험 파일 수정 없음). 진단 질문이며 가설 확정 아님.
- 결과(results.json criteria_table): (1) 비해당, (2) 해당(첫 층 upsample_b.blocks.7), (3) 부분 해당(mix_adjacent_s0 엄격 해당, single_a026_s1 비해당), (4) 비해당(최대 비 6.10, 6.09), (5) 비해당. 체크포인트로 구분 불가: epoch 3 도중 시점, 단일 큰 gradient 여부.
- 해석(보고서 5, 6절, 추정 포함): 리드의 잠정 해석 4문장은 검토 대상이었고 분석가 판정은 갈린다. 일치: 표 재현, ReLU 사멸 관측, (4) 비해당은 큰 gradient 반증 아님, 디코더 손실은 predictor 로 역전파 안 됨, 원인 미확인. 갈림: 큰 gradient 지지 강도(prior 부분 / stats 구분 불가 / skeptic 일시적 사건 강함), blocks.7 원인 대 증상, 디코더만 재학습 타당성(prior 구조상 타당 / skeptic 근거 없음 / stats 판정 없음), 예방책과 비교 가능성. 사람 결정 3건은 보고서 8절.
- 정정(위 007 결과 기록의 176행 "exp_avg 가 21개 파라미터에서 정확히 0" 은 지우지 않고 아래로 정정): results.json 6991-7458행에서 해당 21개 항목의 exp_avg 는 norm 필드만 0.0 이고 absmax 5.605193857299268e-45(float32 최소 비정규수의 4배), zero_frac 0.0 이다. 실제 값은 0 이 아닌 비정규수다(stats, skeptic 지적, 리포터가 results.json 에서 확인). criteria_table 판정은 불변.
- 표시만 한 불일치(수정 안 함): results.json drive_meta 의 modtime_utc/modtime_kst 표기(modtime_kst 17:35:22 가 분석 실행 meta.date_kst 14:47:17 보다 미래이므로 필드 이름이 맞지 않음; 시각을 쓰는 서술에만 영향), action/proprio encoder optimizer 요약의 이름 충돌(n_state 4 대 per_param 2개; C 섹션 action/proprio 수치 사용 불가, criteria_table 영향 없음).

## 2026-11-09 / 004_depth_truncation, 005_inductive_bias_probe / Colab 노트북 작성 (미실행)
- 변경점: experiments/004_depth_truncation/colab/{exp004_005_colab.ipynb, build_notebook.py, exp045_tools.py, requirements_colab_extra.txt}, 005/colab/README.md 추가. 004/005 README 상태 문구와 Colab 절, CHANGES.md 7절 추가. 노트북은 004 -> 005 순서(설정+check, 004 파일럿, 005 파일럿, 004 본 실행, 005 본 실행, 결과 복사). 본 실행은 `APPROVED_MAIN_004/005 = False` 기본이며 False 이면 아무것도 하지 않는다. 파일럿/본 실행은 폴더, 캐시 루트, suffix 분리. 002 파일은 수정하지 않았고 Drive `exp002_work` 에는 쓰지 않는다(파일럿의 002 `_smoke` 캐시는 로컬 기본, 002 attempts 는 `--results-dir` 로 분리).
- 코드 수정(필수): 004/005 `run.py` 가 import 하던 `exp002.common.attempts_check` 가 002 커밋 545168e 에서 삭제되어 004/005 가 ImportError 로 시작하지 못했다. 002 는 건드리지 않고 `src/exp004/base.py`, `src/exp005/base.py` 에 같은 동작의 함수를 복사했다(재시도 상한 3회 동작 동일).
- config 변경(사용자 결정, 확정 기록 12, 결과 보기 전): 005 `branch.exp1_negative` null -> false, `normalization.noninf` {R: iid, D: iid} -> {R: iid, D: const}. 임계값 변경 없음. 005 CHANGES.md 8a. 노트북 005 경고 셀은 "결정됨(확정 기록 12)"과 현재 값을 출력하고, 값이 다르면 경고한다.
- 검증(이 머신, CPU, conda lab07, torch 2.0.1+cpu): 노트북 nbformat 유효, 코드 셀 23개 문법 오류 0. exp045_tools selftest 18 통과 0 실패. 004 selftest 72 통과 0 실패. 005 selftest(확정 기록 12 반영 config) 91 통과 0 실패. 004 `--dry-run` 5/5, 005 `--dry-run` 7/7 단계 통과. 노트북 내장 스냅샷을 풀어 단독 실행한 004 dry-run 5/5, 005 dry-run 7/7, 004 selftest 72/0, tools selftest 18/0.
- 미검증: Colab 에서 노트북 실행 전부, 002 `download`/`cache` 호출 경로, nvidia-smi VRAM 샘플링, tar 보관 셀 대용량, 002 cache_episodes.json 행의 데이터 경로가 `/content/dinowm/data` 와 일치하는지, 시간/디스크 수치(전부 추정).
- 해석: 하지 않음.
- 2026-11-09 추가(Colab 파일럿 후): 004 aggregate 가 MPLBACKEND(커널 값 상속) ValueError 로 실패(attempts aggregate 1). exp045_tools 실행기가 하위 프로세스에 MPLBACKEND=Agg 를 강제하고 004 figs.py, 005 그림 코드도 무조건 Agg 설정. 노트북 재생성, CPU 검증 재실행. 해석 없음.
  - 정정(2026-11-10, 사용자 지시; 위 "attempts aggregate 1" 은 지우지 않고 정정): 004 파일럿 aggregate 는 실제로 **ValueError 실패 5회 + 재시도 상한 거부 1회** 뒤 성공했다. 로그(experiments/004_depth_truncation/logs/colab_20261109/): 09:15:15, 09:18:26, 09:19:20 ValueError(MPLBACKEND) -> 09:20:35 RuntimeError "stage:aggregate failed 3 times in a row (limit 3, CLAUDE.md)" 거부 -> 09:22:04, 09:22:26 ValueError -> 09:22:40 성공(16510 rows). 거부 뒤 재실행은 results_pilot/attempts.json 의 항목이 지워져(누가 지웠는지 기록 없음) 가능했다. CLAUDE.md 재시도 상한(3회)을 넘긴 실행이며 원인은 코드가 아닌 환경 변수 상속(위 수정으로 해결). 파일럿 summary 는 그림 단계 전에 쓰여 수치 영향 없음으로 추정(skeptic I3). 본 실행 aggregate(11:20:37)는 1회 rc=0. 출처: reports/004_depth_truncation.md 5.4절, notes/reviews/004_skeptic.md I3, 리드가 로그 8개 확인.
- 2026-11-09 / 006_size_readout_reanalysis: 004 Colab 파일럿의 MPLBACKEND 상속 오류와 같은 위험을 막기 위해 그림 코드(figs.py, run.py)와 노트북 하위 프로세스 env 에 MPLBACKEND=Agg 를 무조건 설정. 삭제된 002 함수 import(545168e)는 006 에 없음. 노트북 재생성, CPU 검증(잘못된 MPLBACKEND export 상태) 결과는 보고. 해석 없음.

## 2026-11-09 / 004_depth_truncation, 005_inductive_bias_probe / Colab 본 실행 완료, 결과 복사 (결과 기록, 해석 전)
- 변경점: 사용자가 Colab(NVIDIA L4, 23.66 GB)에서 노트북 exp004_005_colab.ipynb 로 004, 005 본 실행(rc=0, 09:06~12:45 UTC). Drive(`exp004_work`, `exp005_work`)에서 rclone 읽기 방향 copy 로 저장소에 복사(Drive 쓰기/삭제 없음): 004 results.json -> experiments/004_depth_truncation/results.json, results/(summary.json, depth_r2.csv, next_z_mse.csv, 그림 2, parts/ 11개, attempts.json), results/colab/{timing_main.json, timing_pilot.json, pilot/, logs_002_pilot/, pilot_002_results/}, logs/colab_20261109/. 005 results.json, results/(REPORT.md, ib_scores.csv, calibration*.csv, law_fits.csv, cost.json, plan.json, tasks.*, 그림 2, predictions/ npz 25개, attempts.json), results/colab/pilot/, logs/colab_20261109/. rclone check: 004 results 17개, 005 results 37개 차이 0. 기존 로컬 파일 덮어쓰기 없음. 004 README, 005 README 상태 문구 갱신.
- sha256 / 바이트: 004 results.json 996b5f40c23ff61f5c9951f5e702dbc30a098ffc48070764be270ebdce6bfe3b 9982; 004 results/summary.json 6038a69e5f305ae170bbfc02dd9ce68ac857232d0891434daefa0aee2636a4a3 8149; 005 results.json 30338df6574b5a4960eac4b1059a0278fbad1754f55db93f6799e0272d922656 6449; 005 results/REPORT.md b20dd6ea2544a1396f1edbd262a725e27c32e46e82d93feb530584ac0a08f929 4586.
- 재계산 확인: parts/ 로 004 `aggregate` 를 CPU(lab07, 임시 --results-dir)에서 다시 돌려 summary.json 이 Drive summary 와 동일(dict 비교 True), 37830 행. (재계산이 저장소의 004/results.json 을 덮어써서 Drive 에서 다시 복사해 sha 996b5f40... 로 복구함; 두 파일의 차이는 environment 필드뿐.) depth_r2.csv: train_seed 고유값 0,1,2(+ 인코더/기준선 NA) = 학습 시드 3개, probe seed 고유값 0~4 = 5개.
- 환경(results.json environment): NVIDIA L4, torch 2.0.1+cu117 (CUDA 11.7), Python 3.9.25. 단계별 시간(timing_main.json, 초) / 샘플링 최대 GPU 메모리(nvidia-smi 전체 GPU, MiB, 하한): 004 enc_cache 176.6 / 1824, enc_probe 1286.8 / 2112, pred 5345.3 / 5118, aggregate 10.1; 005 feat 136.9 / 2508, heads 4949.2 / 2356, plan 2.0, tasks 3.0, metrics 7.4, report 2.5. check 단계는 각 2.3.
- 결과(results.json 의 판정만). 004: summary pilot true, verdict_H_d status refuted, reason no_drop_ge_threshold_at_any_depth, k_star null, m_star null, loss_ood_only false, l_best {module predictor, depth 6}, encoder_layers 12, predictor_layers 6, drop_threshold 0.2. 005: plan 행 no_vanishing, 층 pred_out6(predictor 출력, 하위 실험 5-A~5-D), hold false. H-e not_evaluable(reason area_baseline_saturated, 면적 모델 D-IB 점수 1.0691). H-f supported(D-IB_area= 0.9278, D-IB_area≠ 1.0150, |차| 0.0872, sd 0.0452 / 0.0306). H-g refuted(slope 평균: interp 0.9356, extrap 0.5432, early 0.6066, late 0.5362, labels_a060+a160 0.5409, labels_a060+ind 0.3041, labels_ind+a160 0.2523; 범위 [0.8, 1.2], 범위 안은 interp 뿐).
- 주의 (a): 두 결과의 pilot 은 true 이다. 사용자 확인: 본 실행은 config `pilot: true` 를 그대로 두고 노트북 `ALLOW_MAIN_WITH_PILOT_TRUE` 를 켜서 돌렸다. 결과 파일의 pilot: true 는 config 플래그 때문이며, 실행 범위는 본 실행이다(전체 캐시, 학습 시드 3개, probe seed 5개, head seed 5개). config 의 pilot 값은 바꾸지 않았다. (b): 명세상 임계값(drop 0.2, dev_max 0.1, slope 0.8~1.2, sd_mult 2.0, baseline_saturated 0.95 등)은 "파일럿 후 확정" 제안값인데 확정 기록이 없고, 위 판정은 그 제안값을 그대로 적용한 것이다. 결과를 본 뒤 바꾸지 않는다.
- 해석: 하지 않음. /review-experiment 대기.
- 006 검증(잘못된 MPLBACKEND export 상태): selftest 첫 실행은 selftest.py 자체의 import matplotlib 에서 1건 실패(89/1/1) -> 같은 위치에 Agg 설정 후 재실행 PASS 90 FAIL 0 SKIP 1(scipy 없음). --dry-run 14/14 단계(selftest 제외) 통과, 노트북 JSON 유효, 코드 셀 문법 오류 0.

## 2026-11-10 / 005_inductive_bias_probe / 리뷰 3건 완료, 보고서 작성 (reporter)
- 변경점: notes/reviews/005_prior.md, 005_stats.md, 005_skeptic.md 완료. reports/005_inductive_bias_probe.md 작성(코드와 실험 파일 수정 없음).
- 결과(results.json 값): H-e not_evaluable(area_baseline_saturated, 면적 D-IB 점수 1.0691), H-f supported(|차| 0.0872, 허용폭 0.0905), H-g refuted(slice 7개 중 interp 0.9356 만 범위 안). 층 pred_out6 하나, 체크포인트 1개, head seed 5개, pilot true(사용자 확인: config pilot true 로 본 실행).
- 해석(보고서 5, 6절, 추정 포함): 세 분석가 일치: D-IB 점수 1 초과는 약한 MLP oracle(raw 0.3188 대 무정보 0.5) 때문, 5-B 기준 0.1 은 참 함수도 통과 못함(0.328), ours 와 random-init 이 거의 모든 지표에서 비슷, H-f supported 는 경계값 통과, H-g 기울기 < 1 은 random 과 MLP oracle 에도 있음. 갈린 곳: H-e not_evaluable 의 성격(prior 지표의 한계 / stats 규칙대로이나 사유 이름 부정확 / skeptic 구조적으로 통과 불가), H-f 보고 표현과 격차 방향, 확정 기록 12 처리(skeptic: 전제 "oracle D-IB raw 약 0.5" 가 ind 그룹에서만 성립), ours≈random 원인, H-g 결론 범위.
- 표시만 한 불일치(수정 안 함): 평가 입력 config/명세 2000 대 실행 900(tasks_meta n_eval, config_sha256 2fd84b30... 대 저장소 config 4a6bba0b..., CHANGES.md/log.md 에 기록 없음, 변경 주체 사용자 확인 대기), stats 0절 대 5절의 ours-random 서술, H-f 격차 수치의 계산 방식별 차이, README 낡은 문구. 사람 결정 3건은 보고서 8절.

## 2026-11-10 / 004_depth_truncation / 리뷰 3건 완료 (prior, stats, skeptic)
- 변경점: notes/reviews/004_prior.md, 004_stats.md, 004_skeptic.md 완료. stats 재계산 산출물 experiments/004_depth_truncation/analysis/{recompute_stats.py, stats_out.txt, plot_curves.py, fig_primary_vs_controls.png}. 코드와 실험 파일은 수정하지 않음.
- 결과: 세 분석가 모두 H-d refuted(no_drop_ge_threshold_at_any_depth)가 규칙대로임에 일치(주 곡선 최대 하락 0.033, 임계값 0.05~0.3 에서 불변). stats 재계산에서 ind_ood_gap 28행 불일치 0.

## 2026-11-10 / 004_depth_truncation / 보고서 작성 (reporter)
- 변경점: reports/004_depth_truncation.md 작성(새 측정 없음).
- 요약(해석은 보고서 1, 5절): 판정 refuted 는 규칙상 유효. 일치: random-init predictor 도 평평한 곡선(학습된 predictor 와 interp +0.04 차), split 간 R² 분모 차이로 ind_ood_gap 과 겹침 검정은 해석 불가, 저수준 면적 단서(k=0 extrap 0.788, mask baseline 0.817). 갈림: refuted 의 강도(prior 정보 비소실 이상 아님 / stats 강한 증거 아님 / skeptic 검정력 없을 수 있음), 면적 단서 설명 정도, (ii) 해석, pilot 플래그의 지위(skeptic 만 중요로 지적).
- 표시만 한 불일치(수정 안 함): 파일럿 aggregate 재시도 횟수(skeptic I3 은 실패 5 + 거부 1, 위 2026-11-09 항목은 "attempts 1"; 로그 파일 이름 목록은 skeptic 개수와 모순 없음, 내용은 미확인), `pilot: true` 와 `notes` "subset/partial run" 문구가 본 실행과 안 맞음, 임계값/impl default 확정 기록 없음, 004 summary 를 005 가 사람 검토 없이 소비.
- 사람 결정 3건은 보고서 8절. 가설 확정이나 논문 주장 확정은 하지 않았다.

## 2026-11-10 / 005_inductive_bias_probe / 평가 입력 2000 -> 900 변경 경위 기록
- 변경점(사용자 확인 2026-11-10): Colab 본 실행에서 평가 입력 수(eval.n_inputs)를 사용자가 2000 에서 900 으로 줄였다. 이유: 비용(실행 시간). 저장소 config.yaml 은 2000 그대로이고, 실행 config 의 sha256 은 2fd84b30... 로 저장소 config(4a6bba0b...)와 다르다(reporter 기록). 변경 방법(노트북에서 어떻게 덮어썼는지)은 기록되지 않았다.
- 영향: 005 의 모든 판정은 n_eval 900 기준이다. 사전 등록(명세 4.1, config)에서 벗어난 실행으로 표시한다. 판정은 바꾸지 않는다.

## 2026-11-10 / 008_posthoc_reaggregate_004_005 / 004 depth_r2.csv 사후 재집계 실행 (experimenter)
- 변경점: experiments/008_posthoc_reaggregate_004_005/ 신규 (config.yaml, run.py, README.md, results.json). CPU, numpy, 새 학습 없음. 입력은 004 results/depth_r2.csv, summary.json, 002/004 config (읽기 전용). 임계값 0.2, overlap 2.0, 깊이 비율 0.8 은 004 config 와 같고 run.py 가 일치를 assert 한다. 사전 확인: encoder, predictor 모두 random_init 대응 행이 있고 키(depth, split, probe seed, train seed) 전부 일치(encoder 140/140, predictor 420/420).
- 결과(results.json 에서):
  - A 통제 대비 곡선 Δ(d)=real-random_init (linear/new/late): encoder 관측 안 됨, predictor 관측 안 됨 (어느 split 에서도 Δ 곡선의 0.2 이상 하락 깊이 없음). Δ 곡선 수치(깊이 0,2,4,6,8,10,12): encoder train 0.275, 0.150, 0.172, 0.178, 0.176, 0.191, 0.177 / interp 0.365, 0.266, 0.296, 0.298, 0.333, 0.358, 0.302 / extrap 0.132, 0.068, 0.056, 0.073, 0.066, 0.075, 0.083. predictor(깊이 0..6) train 0.0, 0.008, 0.010, 0.012, 0.016, 0.022, 0.025 / interp 0.0, 0.009, 0.031, 0.034, 0.041, 0.035, 0.048 / extrap -0.0, -0.007, -0.010, -0.010, -0.009, -0.006, -0.0. 보조(평균-2sd>0 인 깊이 수): encoder 7/7(train), 7/7(interp), 7/7(extrap), sd 단위 probe seed 5개. predictor 6/7, 2/7, 0/7, sd 단위 학습 seed 3개.
  - B 척도 통일: 공통분산 R² 와 MSE 는 계산 가능했다. 레벨 레이블을 002 config area_ratio 의 log 로 두고 n_episodes 로 공통 분산 0.49999 를 계산, 복원한 split R² 와 CSV ALL 행 R² 의 최대 차이 8.2e-15. 판정 관측 안 됨(공통분산 R² 곡선에서 0.2 이상 하락 없음, 모든 split, 모듈, real 과 random_init). real 의 깊이별 평균 범위(log 면적비 단위): encoder MAE train 0.1219~0.1612, interp 0.1152~0.1484, extrap 0.2900~0.3701; MSE train 0.0248~0.0430, interp 0.0226~0.0386, extrap 0.1433~0.2048. predictor MAE train 0.1141~0.1281, interp 0.1112~0.1186, extrap 0.2827~0.3013; MSE train 0.0226~0.0267, interp 0.0221~0.0242, extrap 0.1431~0.1526.
  - C 겹침 재계산(학습 seed 3개의 seed 평균 간 sd 단위): encoder 계산 불가(동결 인코더라 학습 seed 단위가 없음). predictor 관측 안 됨: 허용폭 안 깊이 비율 interp 0.0, extrap 0.7143 (기준 0.8). 참고 004 summary 값(probe/학습 seed 합친 sd): predictor interp 0.0, extrap 0.5714; encoder interp 0.0, extrap 0.1429.
- 004 판정(H-d refuted)과 005 판정은 바꾸지 않았다. 해석: 하지 않음. /review-experiment 대기.

## 2026-11-10 / 004_depth_truncation / 문서 정리 (README 7절 추가) (experimenter)
- 변경점: 004 README.md 7절에 기록. results.json 의 `pilot: true` 는 config 플래그 때문이고 실행 범위는 본 실행(학습 시드 3 x probe 시드 5; depth_r2.csv 로 확인), `notes` 의 "subset/partial run" 문구는 이 실행에 맞지 않음. results.json, summary.json 은 수정하지 않았다. log.md 의 재시도 횟수 정정은 하지 않았다(사용자 결정 대기).

## 2026-11-10 / 005_inductive_bias_probe / selftest 보강 (experimenter)
- 변경점: src/exp005/selftest.py 에 단계 H 추가(3레벨 MLP oracle 의 OoD D-IB 점검, `_vals` 중복, `continue` 구조). 판정/지표 코드는 그대로. CHANGES.md 9절에 기록.
- 결과(CPU, 로그 logs/selftest_20261110_skeptic.log): passed=98 failed=3 skipped=0. 실패 3개: C1(c)-2 (합성 무작위 모델의 ood_any D-IB 점수 1.587, 기준 < 1), S1-3, S1-4 (5-A 예측만 없는 seed 에서 5-B/5-D 행도 0개). 코드는 고치지 않았다. 사용자 결정 대기. 해석 없음.

## 2026-11-10 / 009_positive_control_depth, 010_source_separation_ib / 코드 작성 (미실행, 승인 대기) (experimenter)
- 변경점: experiments/009_positive_control_depth/ (config.yaml, run.py, src/exp009, README.md), experiments/010_source_separation_ib/ (동일 구성). 004/005 코드는 import 로만 재사용, 수정 없음. GPU 실행 없음, 실제 체크포인트/캐시에서 실행한 적 없음.
- 009: predictor 블록 m_int(리스트, 기본 [1,3,5], 초안 승인 대기, CLI/환경변수로 덮어쓰기 가능하며 results.json 에 m_int_config/m_int_used/m_int_overridden 기록) 출력 hidden 의 z 부분에서 크기 방향 소거(leace_scalar 기본, probe_direction 대안), 강도 0/0.25/0.5/1.0, 단위 = (블록, 강도, 학습 seed) 독립 실행과 재개, 강도 0 기준선은 학습 seed 당 1회 공유, 블록마다 판정(강도 1.0, seed 3개 중 2개 이상 m* <= m_int+1). 방향 적합은 학습 episode 절반, 평가는 test episode.
- 010: 005 head/과제/지표 재사용, 네 표현(enc_k12 ours/random, pred_out6 ours/random), 과제 시드 [2000, 2001, 2002](초안), eval.n_inputs 900, H-f TOST Δ 0.05(초안).
- 검증(CPU, 합성 데이터/toy 모델): 009 selftest passed=46 failed=0, 010 selftest passed=19 failed=0, 두 실험 --dry-run 통과. selftest 가 확인한 사실: toy 에서 부분 강도(0.5)는 크기 성분을 축소만 하여 다시 학습한 linear probe 의 R² 가 기준선과 같았다(0.996 대 0.996, 강도 1.0 은 -0.022).
- 예상 GPU 시간은 각 README 에 추정으로 표시(009 블록 1개 약 0.5~1시간, 010 약 3~3.5시간, 둘 다 추정). 해석 없음.

## 2026-11-10 / 008_posthoc_reaggregate_004_005 / 리뷰 3건 완료, 보고서 작성 (reporter)
- 변경점: notes/reviews/008_prior.md, 008_stats.md, 008_skeptic.md 완료. reports/008_posthoc_reaggregate_004_005.md 작성(새 측정 없음, 코드와 실험 파일 수정 없음).
- 요약: 사후 탐색이며 004 판정을 바꾸지도 강화하지도 않음. 상태 A/B 관측 안 됨, C predictor 관측 안 됨(extrap 5/7=0.714, 004 원 0.571), C encoder 계산 불가. 일치: 계산 정확(stats 독립 재현), predictor Δ 최대 0.048 이라 0.2 규칙 도달 불가(천장), 공통 R² 도 train/interp 천장, split R² 분모 의존(interp 낮음과 extrap 후함은 인공물), C 는 한 깊이 차이로 불안정. 갈림: random-init 이 통과 통제인가(skeptic), extrap Δ<0 의 의미(prior 가설 / stats 우연 가능 / skeptic 일관된 방향이나 해석 이름), 인코더 보조 판정이 사전 기준 위반인가(skeptic 만), B 판정 규칙이 사전 기준에 없음(skeptic 만). 검정력 문제 해소는 009 필요.
- 사람 결정 3건은 보고서 8절(009/010 GPU 승인과 순서, 009 에 008 의 A/B 규칙 결합 여부와 CPU 재검정 선행 여부, 008 결과 인용 범위).

## 2026-11-10 / 008, 009 / 사용자 결정 (008 리뷰 후)
- 008: 사후 탐색으로만 남긴다. 004 H-d refuted 의 보강 근거로 인용하지 않는다. results.json 재현성 기록 보강은 하지 않음(결정 사항에 포함되지 않음).
- 009: 개입 블록을 가능한 모든 블록 [1, 2, 3, 4, 5, 6] 으로 확정. 008 의 A(random-init 대비 Δ), B(공통 분산 R², MAE/MSE) 규칙을 보조 판정으로 함께 적용. 소거 방식, 강도 구성, 주 성공 기준 확정과 GPU 실행 승인은 아직 대기.

## 2026-11-10 / 009_positive_control_depth / 사용자 결정 반영 (코드, 미실행) (experimenter)
- 변경점(사용자 결정 2026-11-10): (1) config.yaml `intervention.m_int` 기본값 [1, 2, 3, 4, 5, 6], 주석 "사용자 확정 2026-11-10". 블록 6 은 회복 관측을 계산하지 않고 "해당 없음" 으로 기록, 검출 판정만. (2) 008 의 A, B 를 보조 판정으로 추가(`src/exp009/auxrules.py`, 정의는 008 run.py 에서 복사하고 출처 주석, 008 파일 수정 없음; `stages.aggregate` 가 블록 x 강도 x 학습 seed 별로 results.json 에 관측됨/관측 안 됨/계산 불가 저장). 주 성공 기준(강도 1.0, 3개 중 2개 이상 m* <= m_int + 1)은 바꾸지 않았고 보조 판정은 주 판정에 들어가지 않는다. A 의 random-init 기준은 009 에서 새로 측정하지 않고 004 results/depth_r2.csv 의 predictor random_init 대응 행을 쓴다(학습 seed 0,1,2 각 140행, 키 (깊이, split, probe seed) 일치 확인; 집계 시 매번 검사, 불일치나 파일 없음이면 계산 불가). (3) selftest 에 블록 6, A/B 점검 추가. (4) README 예상 시간 갱신. (5) 008 README 에 "지위: 사후 탐색으로만 남긴다" 문구 추가.
- 검증(CPU, 합성/toy): 009 selftest passed=64 failed=0, --dry-run (check/intervene/aggregate) 정상. 실행 단위 57 (기준선 3 + 개입 54). GPU 미사용, 실제 체크포인트 실행 없음.
- 예상 GPU 시간(추정, 실측 없음): 블록 6개 약 3.2~6.3시간(개입 54 단위 3~6시간 + 기준선 10~20분).
- 해석: 하지 않음. 소거 방식, 강도 구성, 주 성공 기준 확정, GPU 실행 승인은 대기.

## 2026-11-10 / 009_positive_control_depth / 설정 확정 (사용자 승인)
- 소거 방식 leace_scalar, 강도 [0, 1.0], 주 성공 기준 초안대로 확정(hypotheses.md 009 확정 기록). Colab 노트북 작성 요청. 실행은 사용자가 Colab 에서 한다.

## 2026-11-10 / 009_positive_control_depth / 확정 설정 반영, Colab 노트북 (코드, 미실행) (experimenter)
- 변경점(사용자 확정 2026-11-10): (1) config.yaml: `intervention.strengths` [0, 1.0] (0.25, 0.5 제거), method leace_scalar, m_int [1..6], success 그대로; "초안/승인 대기" 주석을 "사용자 확정 2026-11-10" 으로 교체. 시드(seed 0, 학습 [0,1,2], probe [0..4], fit_seed 9000) 변화 없음. `pilot: true` 는 그대로(본 실행 전 사람이 변경). (2) selftest: 단위 수 점검 57 -> 21, 단일 블록 점검 3+3, 확정 설정(method, strengths) 점검 추가. 부분 강도 toy 점검들은 코드 기능 점검이라 유지. (3) README: 상태, 실행 단위 21, 예상 GPU 시간(추정) 갱신. (4) Colab: colab/build_notebook.py 와 colab/exp009_colab.ipynb 신규 (004 의 colab/exp045_tools.py 재사용; 코드 스냅샷 임베드; config 값 덮어쓰기 없음, 파일럿 셀만 기존 CLI 플래그 --m-int 6 --train-seeds 0 사용).
- 검증(CPU): 009 selftest passed=65 failed=0, check/intervene/aggregate --dry-run 정상 (intervene 계획: units total 21 = 기준선 3 + 개입 18). 노트북 JSON 유효, 코드 셀 17개 구문 검사 통과, 임베드 스냅샷을 풀어 selftest 재실행 passed=65 failed=0, 스냅샷 config.yaml 이 원본과 동일. 로컬 check 의 002 run DONE=False 는 이 머신에 체크포인트가 없어서이며 Colab 에서 확인 대상.
- 예상 GPU 시간(추정, 실측 없음): 블록당 약 16~30분 x 6 + 기준선 약 10~20분 = 약 1.8~3.3시간 (이전 초안 3.2~6.3시간에서 강도 수 감소 반영; 블록당 시간 분해는 가정).
- GPU 미사용, 실제 체크포인트 실행 없음, git commit 하지 않음. 해석 없음.

## 2026-11-10 / 009_positive_control_depth / 노트북 git clone 전환, Colab 파일럿 실패 1건 수정 (코드, 미실행) (experimenter)
- 변경점: (1) 노트북의 코드 임베드(약 208KB) 제거, A3 가 공개 저장소 https://github.com/jongmin-s-multicore/probing-experiment.git 을 clone 하고 REPO_REF(A2, 기본 master, 브랜치/commit 해시/태그) 로 checkout, HEAD 해시 출력, `exp009_work/code_version.json` 및 results.json `code_version`(aggregate 에 추가) 에 기록. 토큰 선택 사항. (2) 실패 1건 기록: 사용자 Colab 파일럿 2026-11-10 08:21, `intervene` 이 FileNotFoundError `/content/dinowm/data/cloth_sim/a060_train/metadata.pt`; 원인 metadata.pt 로컬 누락(노트북 README 의 "데이터 다운로드 없음" 문구가 틀림); results_pilot attempts stage:intervene = 2. (3) 수정: 노트북 셀 A6b(metadata.pt 만 배치), `run.py check`/`intervene` 의 metadata.pt 사전 점검(attempts 미소비, 종료 코드 4), README 정정, 009/CHANGES.md 신규, selftest 6개 추가.
- 검증(CPU): 009 selftest passed=71 failed=0. 노트북 JSON 유효, 코드 셀 ast 통과. GPU 미사용, git commit 하지 않음. 해석 없음.

## 2026-11-10 / 009_positive_control_depth / 파일럿 attempts 초기화 (사용자 결정)
- results_pilot/attempts.json 의 stage:intervene 2 -> 0 초기화를 사용자가 승인했다. 근거: 앞선 2회 실패는 모두 metadata.pt 누락(데이터 준비 문제, 코드 4728e83 에서 사전 점검 추가)이었다. 초기화는 사용자가 Colab 에서 Drive 파일을 직접 수정한다. 본 실행 attempts 와는 별개다.

## 2026-11-10 / 006_size_readout_reanalysis / 분석 F 추가, 노트북 git clone 전환 (코드, 미실행) (experimenter)
- 근거: notes/hypotheses.md "후보 J 추가 (2026-11-10, 사용자 지시): 분석 F 와 실행 승인". 사용자가 Colab 에서 009 와 병행 실행하기로 했고 이를 006 GPU 실행 승인으로 기록한 상태다.
- 변경점: (1) 분석 F(teacher forcing 대 open-loop): `extract` 가 같은 체크포인트 6개, 같은 행(학습 900 + test 900)에서 실제 인코더 latent context 의 1-step 예측을 step 1-12 마다 추가로 계산해 `teacher_forced_pooled.npy` 로 저장(같은 풀링: fp16 캐스트 후 float32 패치 평균). `analyze_f`(신규 `stages_f.py`): 전이 probe 와 재학습 probe 를 두 latent 에 적용해 step 별 R², MAE, 레벨별 예측 평균/편향, 기울기/절편, tf-ol 차이, 조건별 시드 단위 평균/SD/t CI, (a)-(b) 차이를 results.json `F` 에 저장. 임계값/판정 없음, interpretation null, 새 시드 없음. aggregate, figF, `--dry-run` 15단계. (2) results.json `code_version`(git HEAD 등), `metadata.pt` 사전 점검(check 와 extract 시작 전, 없으면 종료코드 4), DONE 마커가 있어도 TF 파일이 없으면 extract 재계산. (3) Colab 노트북 재작성: 코드 임베드 제거, 공개 저장소 clone + REPO_REF checkout + HEAD 해시를 `exp006_work/code_version.json` 과 results.json 에 기록, 출력/로컬 폴더가 009 와 겹치지 않음(`exp006_work`, `/content/{mylab006,venv006,dinowm006}`), 009 모듈 import 없음, metadata.pt 배치 셀, 전 단계 dry-run + check + selftest, smoke 먼저. 노트북 크기 472,239 -> 23,600 바이트. 002/004/009 파일은 수정하지 않았다.
- teacher forcing 가능한 step: 코드(`tf_feasible_steps`)와 selftest 로 확인. num_hist=1, t=0, 캐시 프레임 13장에서 step 1..12 전부 가능(context 프레임 0..11, 목표 프레임 1..12), 불가능한 step 없음. proprio 부분은 teacher-forced 에서 실제 인코딩 값, open-loop 에서 예측 되먹임 값이라 다르고 action 부분은 같다(README 3F, limitations).
- 검증(CPU, 합성/toy): selftest PASS 117, FAIL 0, SKIP 0 (F 추가 전 같은 환경 PASS 92). 모든 단계 `--dry-run` 15/15 종료코드 0. 노트북 JSON 유효, 코드 셀 13개 ast 통과(전체 14셀, markdown 1).
- 예상 GPU 시간(추정, 실측 없음): 6 run 합계 약 0.35-1.8 GPU 시간(A-E 만의 이전 추정 0.3-1.5 시간 + F 약 18%). Drive 읽기가 지배할 수 있다. CPU 분석 F 는 C 의 약 2배 시간(추정).
- 주의: 노트북은 clone 한 저장소의 코드를 돌리므로 사용자가 이 변경을 commit/push 해야 Colab 에 반영된다. GPU 미사용, git commit/push 하지 않음. 결과 수치 없음, 해석 없음.

## 2026-11-10 / 009_positive_control_depth / Colab 본 실행 완료, 결과 복사 (결과 기록, 해석 전)
- 실행: 사용자 Colab NVIDIA L4, 코드 4728e83 (results.json `code_version.git_status_dirty` true, 무엇이 dirty 였는지 미확인), `pilot: true` 그대로. attempts intervene 0, aggregate 0. 결과를 Drive exp009_work 에서 experiments/009_positive_control_depth/{results.json, results/, results/colab/} 로 사용자가 복사(Windows Zone.Identifier 부속 파일은 삭제).
- 결과(results.json): 블록 1-6 모두 `verdict.status` criterion_met, 학습 seed 3/3 검출, `m_star_by_train_seed` 는 모든 블록에서 m_int 와 같음. 보조 A, B 모두 3/3 관측됨(m_int+1 이내). `baseline_vs_004.max_abs_r2_diff` 전부 0.0. missing_units 없음. diagnostics: 소거 후 판독 방향 std 비 약 2e-7~4e-7.
- 보조 관측(`aux_recovery_r2_minus_r2_at_m_int`, 블록 1): 소거 블록 다음 블록에서 R² 가 train +0.719, interp +0.219, extrap +0.785 회복(seed 평균값 기준인지 키 정의는 리뷰에서 확인). seed 0 곡선: 소거 블록 R² 약 0, 바로 다음 블록 train 0.76~0.82.
- 해석: 하지 않음. /review-experiment 로 넘김.
