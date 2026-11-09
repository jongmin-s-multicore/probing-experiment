# 실험 설명: 005_inductive_bias_probe

- 상태: **코드 작성 완료, Colab 노트북 작성, 미실행.** 이 머신(GPU/데이터/체크포인트 없음)에서는 문법 검사, `--dry-run`, 합성 데이터 CPU 셀프테스트만 했다. 학습, 데이터 다운로드, GPU 실행, 서버 접속은 하지 않았다. 이 문서에 실험 결과 수치는 없다. 명세와 달라진 점은 전부 CHANGES.md 에 있다.
- 검증할 가설: notes/hypotheses.md 후보 I (확정 기록 9), **H-e**(5-A/5-B), **H-f**(5-C), **H-g**(5-D). 임계값 0.1(45도선 평균 이탈), 0.8~1.2(기울기)는 명세의 제안값(파일럿 후 확정, 바꾸면 결과보다 먼저 기록). 명세: notes/specs/exp4_5_spec.md 4절. 출처 방법: Vafa et al., ICML 2025 (arXiv 2507.06952) 의 축소판 — **원문은 확인하지 못했다**(CHANGES.md).
- 방법: 실험 4 의 `summary.json` 으로 4.6 분기표에 따라 층과 하위 실험을 고르고(`branch.py`), 그 층의 동결 특징 위에 과제마다 새로 초기화한 attentive head(002 probe 와 같은 구조/용량)를 소량(100개) 데이터로 학습해, 학습된 함수들이 크기 구조를 따르는지 본다. 기준 모델 4개를 같은 절차로 돌린다: oracle(참 크기 s 를 받는 linear, 2층 MLP hidden 5), 면적 휴리스틱 head(mask 면적/bbox 5개 특징만), random-init(인코더 또는 predictor), 무정보 모델.
  - 5-A 유한 state: 과제 100개 = 9개 레벨에 Bernoulli(0.5) 라벨. R-IB, D-IB (쌍 그룹 all / ind / ood_any / ood_both).
  - 5-B 외삽 calibration: 과제 100개 = 무작위 사인파 합(주파수 제한, 사용자 결정) g_k(s). 학습 레벨 입력만으로 학습, oracle 쌍별 거리의 20개 등간격 구간 calibration, 45도선 평균 절대 이탈.
  - 5-C 외형 휴리스틱 분해: 5-A 예측을 다시 사용(추가 학습 없음). 면적 10분위 q 로 `D-IB_area=`, `D-IB_area≠`, `R-IB_fold`.
  - 5-D 크기 법칙 일관성: 학습 입력의 1%(최소 30개)의 크기 라벨로 head 를 회귀 fine-tune, slice 별(보간 / 외삽 / 초기 / 후반 / 라벨 레벨 조합 3가지) 기울기와 절편. PySR 은 설치되어 있을 때만.
  - LoRA 적응은 구현하지 않았다(사용자 결정).
- 재사용: 002 의 probes(`_make_attentive`), evalgrid(윈도우, 라벨), cache(Q1 캐시 로드), model_io(체크포인트 로드, 윈도우), masks 특징, common(strict config 로더, 로그, 재시도 가드)을 수정 없이 import. 004 의 `deep.py`(조기 종료, random-init predictor)도 import. 001/002/003/004 파일은 수정하지 않았다.
- **이 실험으로 내가 주장하려는 것** (확정 전 초안, 해석은 분석가 리뷰 몫): 4.6 분기로 선택된 층에서, 소량 데이터로 새 과제에 적응시킨 head 들의 예측이 (1) 보이는 면적 휴리스틱 모델보다, 그리고 random-init 모델보다 크기 state 를 구분/유지하는 정도(R-IB, D-IB; OoD 레벨 쌍 포함), (2) 상대 크기의 외삽에서 oracle 과 일치하는 정도(5-B calibration 이탈), (3) 면적 구간이 같은 쌍과 다른 쌍에서 구분하는 정도의 차이(5-C), (4) 절대 크기 회귀의 slice 별 기울기가 일정한 정도(5-D) 를 수치로 보고한다. 판정(supported/refuted/not_evaluable)은 config 임계값을 규칙대로 적용한 결과일 뿐이며, "모델이 크기를 사용한다/세계 모델을 찾았다" 같은 결론은 쓰지 않는다. probe 류 정확도만으로는 "읽을 수 있다"고만 쓰고, "사용된다"는 실험 3(개입) 이후에만 쓴다.
- 성공 기준: config.yaml `verdict.*` 값을 코드가 읽어 판정한다. null 이면 `not_evaluable`.

## 1. 구조

```
experiments/005_inductive_bias_probe/
  config.yaml     모든 시드(head 5개, 평가/과제/사인파/5-D/무정보 시드), 과제 수, 임계값
  run.py          단계 CLI: check plan tasks feat heads metrics report selftest   (모두 --dry-run 지원)
  CHANGES.md      명세와 달라진 점 전부
  src/exp005/
    base.py       002/004 src 를 sys.path 에 올리고 config/경로 처리
    branch.py     4.6 분기표 (summary.json -> 층, 하위 실험, hold)
    tasks.py      평가 입력 2000개, 5-A/5-B/5-D 과제, 사인파 함수족 (numpy)
    ibmetrics.py  R-IB/D-IB 및 5-C 분해, 정규화, calibration, 선형 적합 (numpy)
    heads.py      attentive head(002 구조), oracle linear/MLP, 면적 head (torch)
    feats.py      층 특징 경로(004 캐시) 및 predictor 층 추출 (torch)
    verdicts.py   H-e/H-f/H-g 판정
    stages.py     단계 구현
    selftest.py   합성 데이터 CPU 셀프테스트
  results/        (실행 시 생성) plan.json tasks.npz tasks_meta.json predictions/*.npz ib_scores.csv calibration.csv calibration_dev.csv law_fits.csv cost.json REPORT.md 그림
  results.json    (report 가 생성)
```

## 2. 단계

| 단계 | 하는 일 | GPU |
|---|---|---|
| `check` | 환경, 경로, 004 summary.json/캐시, 002 캐시 존재 출력 | 선택 |
| `plan` | `../004_depth_truncation/results/summary.json` (또는 `--summary`)을 읽어 4.6 분기표로 층/하위 실험 결정 -> `results/plan.json`. summary 가 없으면 중단(dry-run 은 빈 계획 출력). 실험 1 음성 여부는 config `branch.exp1_negative` (null 이면 false 로 취급하고 plan.json 에 기록) | 불필요 |
| `tasks` | 평가 입력 2000개(초기/후반 반반, `eval.seed`), 5-A 라벨/D_k, 5-B 사인파/D_k, 5-D 라벨 부분집합 -> `results/tasks.npz` | 불필요 |
| `feat` | 선택된 predictor 층의 특징(real + random-init predictor)을 002 체크포인트에서 추출 -> `$EXP005_CACHE`. 인코더 층은 004 캐시(`tokens_k<k>.npy`, `rand_k<k>.npy`)를 확인만 한다 | 필요 |
| `heads` | 레이어 독립 모델(oracle linear/MLP, 면적 head)과 층별 ours/random 의 모든 head 학습, 과제별 예측을 `results/predictions/*.npz` 에 저장. 첫 head 의 시간을 로그에 찍고 전체 시간을 투영한다. `--max-tasks N` 으로 시간 측정/파일럿 | 필요 |
| `metrics` | 저장된 예측으로 `ib_scores.csv`, `calibration.csv`, `calibration_dev.csv`, `law_fits.csv`, 그림(재학습 없음) | 불필요 |
| `report` | 판정 규칙 적용 -> `REPORT.md`, `results.json` (해석 없음, 비용 포함) | 불필요 |
| `selftest` | 합성 데이터 CPU 셀프테스트 (결과 아님) | 불필요 |

## 3. 분기 (4.6) 구현
`branch.select(summary, exp1_negative)`; 행은 표 순서로 평가하고 여러 행이 해당하면 위쪽 행의 층이 앞에 오고 아래 행의 하위 실험이 추가된다. 층 이름: `enc_k<k>`, `pred_h<m>`(m 개 block 뒤 은닉 상태, z 부분), `pred_out<m>`(조기 종료 출력, m=block 수는 모델 출력).

| 행 | 조건(summary.json) | 층 | 하위 실험 |
|---|---|---|---|
| 1 | `k_star` 있음 | 인코더 `l_best`(`l_best_by_module.encoder`), 인코더 최종 층 | 5-A, 5-B |
| 2 | `k_star` 없음, `m_star` 있음 | 인코더 최종 층, `pred_h(m_star-1)`, `pred_h(m_star)` | 5-A, 5-C |
| 3 | `loss_ood_only == true` | `l_best` | 5-B, 5-D |
| 4 | `k_star`, `m_star` 모두 없음 | `pred_out(predictor_layers)` | 5-A~5-D |
| 5 | config `branch.exp1_negative == true` | 인코더 최종 층 | 5-C 만, **hold**(heads 는 거부, 판정은 전부 not_evaluable) |
층별로 해당 행의 하위 실험만 돈다(`plan.layer_subexps`). 행마다 셀프테스트로 확인했고, 004 의 `build_summary` 가 만든 summary 로도 확인했다.

## 4. 출력 (명세 4.8)
- `ib_scores.csv`: 열 `layer, model, metric, pair_group, seed, raw, score` (명세 순서) + `score_alt, ref_mode`. `model`: ours, oracle(= config `oracle.primary` 의 head), oracle_linear 또는 oracle_mlp(나머지 하나), area, random, noninf(i.i.d. 무정보), const(상수 출력). `score` 는 config `normalization.noninf` 에 따른 기준, `score_alt` 는 다른 기준.
- `calibration.csv`(구간별 x/y), `calibration_dev.csv`(45도선 평균 절대 이탈, 정규화 거리), `fig_calibration.png`, `fig_dib_decomp.png`, `law_fits.csv`(slice 별 기울기, 절편, R², n, PySR 식), `predictions/`(과제별 예측; 무정보/상수 모델은 시드로 재생성하므로 저장하지 않음), `REPORT.md`, `cost.json`(GPU, 최대 VRAM, 시간), `results.json`.

## 5. 실행 방법

선행 조건: 002 조건 (b) 체크포인트와 Q1 캐시, 004 `summary.json`(파일럿 또는 본 실행)과 004 `enc_cache`(random-init 포함) 캐시. 이 머신에는 없다.

```bash
cd experiments/005_inductive_bias_probe
conda activate dino_wm
export DINO_WORK=... DINO_TRAIN=... EXP002_CACHE=... EXP004_CACHE=... EXP005_CACHE=... ACWM_PHYS_SRC=...
python run.py selftest
python run.py plan --dry-run && python run.py plan            # 004 summary.json 필요
python run.py tasks --dry-run && python run.py tasks
python run.py feat --dry-run && python run.py feat
python run.py heads --dry-run
python run.py heads --max-tasks 3 --seeds 0 --layers <첫 층>   # 시간 실측 (로그의 'first head' 줄), 끝나면 results/predictions 의 시험 파일을 지우고 본 실행
python run.py heads
python run.py metrics && python run.py report
```
(파일럿 캐시이면 `--suffix _smoke` 를 모든 단계에 같이 주거나 config `exp4.cache_suffix` 를 바꾼다.) 시간 투영: 로그에 "first head X s -> projected Y s" 가 찍힌다. 길면 seed 를 3 으로, 과제를 50 으로 줄이고 CHANGES.md 에 기록한다(명세 4.8 비용 메모).

## 6. 알려진 한계와 미검증 사항 (정직하게)
- **서버에서 실행되지 않은 코드**: `feat`(실제 모델 로드/추출), `heads` 의 실제 토큰 규모(196 x 384 토큰 2000개, GPU), 004 캐시 경로 연결. 합성 토이 데이터로 확인한 것: 분기 로직, 과제/평가 입력 생성, 지표(brute force 대조), head 코드 경로, 통계/판정 규칙, 파일 출력.
- **[2026-11-09 갱신: 확정 기록 12 로 사용자가 `normalization.noninf: {R: iid, D: const}`, `branch.exp1_negative: false` 를 결정해 config 에 반영함(결과 보기 전). 아래 서술은 그 이전 상태이며 CHANGES.md 8a 참고.]** 정규화식 문제(당시 미해결): 명세의 식 `(raw - raw_noninf)/(raw_oracle - raw_noninf)` 에 무정보 모델 = i.i.d. 무작위 예측을 쓰면, 이진 과제에서 참 state 를 아는 모델의 D-IB raw 도 약 0.5 이고 무정보 모델도 약 0.5 라서 분모가 0 에 가깝다. 코드는 분모가 `normalization.min_gap`(0.05) 미만이면 점수를 NaN 으로 두고(잡음 점수를 만들지 않기 위해) 그 경우 H-e 는 `not_evaluable` 이 된다. 상수 출력 모델을 기준으로 쓰는 대안은 `score_alt`(또는 config `normalization.noninf.D: const`)로 이미 계산한다. 원문 App. B 식을 확인하지 못했다 (CHANGES.md).
- 5-A 의 D_k 는 명세가 "전 레벨 train split" 이지만 002 캐시에는 학습 레벨 3개의 train 만 있어서 그 3개 레벨에서만 뽑는다(CHANGES.md).
- 비용은 모두 추정이다(CHANGES.md 하단).

## 7. Colab 실행 (2026-11-09 추가)
005 전용 노트북은 없다. `../004_depth_truncation/colab/exp004_005_colab.ipynb` 의 C(파일럿), E(본 실행) 절이 005 를 실행한다. 004 README 6절 참고. 005 와 관련된 점만 적는다.
- 005 는 `exp004_work/results/summary.json`(본 실행) 또는 `results_pilot/summary.json`(파일럿, `plan --summary`)과 004 enc 캐시를 읽는다. 004 enc 캐시는 `/content/main/exp004_cache` 에 있어 **세션이 끊기면 다시 만들어야 한다**(004 README 6절의 선택 tar 셀).
- 출력: `exp005_work/{results,logs,results_pilot}`, 캐시 `/content/main/exp005_cache`(파일럿 `/content/pilot/...`). 파일럿은 `plan`, `tasks`, `feat`, `heads --max-tasks 3 --seeds 0 --layers <첫 층>` 이며 첫 head 시간("first head")과 전체 head 수로 시간을 투영한다. `metrics/report` 는 본 실행에서만.
- 사람 결정 항목은 확정 기록 12(2026-11-09)로 `branch.exp1_negative: false`, `normalization.noninf: {R: iid, D: const}` 가 config 에 반영됐다. 노트북 C1/E0 셀이 현재 값을 출력하고, 값이 이 결정과 다르면 경고한다(노트북은 config 를 바꾸지 않는다).
- `APPROVED_MAIN_005 = False` 이면 본 실행 셀은 아무것도 하지 않는다.
