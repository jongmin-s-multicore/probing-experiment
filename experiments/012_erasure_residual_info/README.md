# 실험 설명: 012_erasure_residual_info

상태: 코드와 Colab 노트북 작성, CPU 점검(selftest, dry-run, 노트북 JSON/문법) 완료(2026-11-10). 가설 확정. **GPU 실행 미승인, 미실행**. 이 문서에 결과 수치는 없다.

- 검증할 가설: notes/hypotheses.md "### 012 소거 후 잔여 정보 진단: 009 의 회복 경로" 절과 그 아래 확정 기록(2026-11-10, 사용자 결정).
  - 질문: 009 에서 스칼라 LEACE 로 블록 m_int 출력의 크기 정보를 소거한 뒤, 지워진 hidden h_{m_int} 에 크기 정보가 어떤 형태로 남아 있었는가. 정보 자체가 제거되었는지, pooled z 의 선형 투영만 제거되었는지를 가른다.
  - 근거: reports/009_positive_control_depth.md 7절 2번, notes/reviews/009_skeptic.md I2, I4.
- 방법(가설 절 요약, 상세는 hypotheses.md):
  - 009 와 같은 체크포인트(002 조건 (b), 학습 seed 0,1,2), 같은 소거(`leace_scalar`, 강도 1.0), 같은 방향 적합 규칙. m_int = [1..6] (확정).
  - 소거 후 h_{m_int} 와 같은 깊이의 강도 0 기준선에서 다음을 측정한다. (a) 레벨 3개 one-hot 의 선형 분류 정확도, (b) pooled MLP probe R² 와 토큰 단위 attentive probe R², (c) 행동/proprio 좌표(384 이후 20차원)만으로 한 linear probe R², (d) 참고용 linear R²(009 재현 점검).
  - probe 는 방향 적합에 쓰지 않은 학습 episode 에서만 학습하고 test episode 에서 평가한다(skeptic I4 반영).
  - 009 의 hidden 캐시와 방향 파일은 남아 있지 않으므로 다시 계산한다. 실제 predictor 는 CPU 로 돌릴 수 없으므로 GPU 가 필요하다.
- 성공 기준(확정 2026-11-10): 블록별로 판정하며, 각 항목은 학습 seed 3개 중 2개 이상에서 다음을 만족할 때 성립한다.
  - (a) "클래스 구조 잔존": 소거 후 3-클래스 정확도 ≥ 0.533 (우연 1/3 + 0.2)
  - (b) "비선형 잔존": 소거 후 MLP 또는 attentive R² ≥ 0.2
  - (c) "행동 경로": 행동 좌표만의 R² ≥ 0.2 (강도 0 의 값을 함께 보고)
  - (a)-(c) 가 모두 아니면 "이 측정들로는 잔여 정보 미검출"로 기록한다. 이는 정보가 없다는 뜻이 아니다.
- 이 실험으로 주장하려는 것: 블록별, 항목별 판정과 수치. 판독 가능성만 보며 "사용"(인과적 역할)이나 회복의 실제 경로는 주장하지 않는다.
- config 결정(2026-11-10, 사용자): MLP probe 은닉 256 1층(002 attentive 와 같은 학습 설정), attentive probe 002/004 설정 그대로, 측정 깊이 m_int 와 m_int+1 (판정은 m_int 만, m_int+1 은 보조 관측), 실행 Colab L4 (009 방식 노트북, APPROVED_MAIN_012 게이트).

## 실행 방법

환경: `~/miniconda3/envs/lab07/bin/python` (로컬 CPU 점검용). 실제 predictor 는 `Attention.bias.to('cuda')` 때문에 CPU 로 못 돌린다(004 README). `measure` 는 GPU(Colab L4) 에서만 돈다. 아래 명령은 `experiments/012_erasure_residual_info/` 에서 실행한다.

| 단계 | 명령 | 하는 일 | GPU |
|---|---|---|---|
| selftest | `python run.py selftest` | 합성/toy 데이터와 toy 월드모델로 판정 경계, 데이터 분리, 측정기, 단위 재개, 집계 점검 | 불필요 |
| check | `python run.py check` | config 일치 검사(009, 002), 체크포인트 폴더 구조, Q1 캐시, metadata.pt 점검. 문제가 있으면 종료코드 1 | 불필요 |
| measure | `python run.py measure [--m-int 6] [--train-seeds 0] [--results-dir DIR] [--fresh]` | 단위별 측정(아래), part 파일을 `results/parts/` 에 저장. 있으면 건너뜀 | 필요 |
| aggregate | `python run.py aggregate [같은 옵션]` | 블록별 판정과 `results.json` | 불필요 |

모든 단계는 `--dry-run` 을 지원한다(계획만 출력). 종료코드: 0 정상, 1 실패(시도 1회 소비, 최대 3회, `<results-dir>/attempts.json`), 3 config 값 null, 4 사전 점검 실패(시도 소비 안 함; 체크포인트 폴더 구조 `outputs/exp002_multi_60_ind_160_s{0,1,2}/{DONE,hydra.yaml,train_manifest.json,checkpoints/model_latest.pth}`, Q1 캐시 DONE, metadata.pt, config 일치).

단위(009 와 같은 21개): 기준선 단위 3개(학습 seed 당 1개; 강도 0, 필요한 모든 깊이 1..6) + 소거 단위 18개(블록 m_int 1..6 x 학습 seed 3; 강도 1.0, 깊이 m_int 와 m_int+1, 블록 6 은 m_int+1 이 "해당 없음"). 깊이별 측정 횟수: 기준선 18 + 소거 33 = 51. 각 깊이에서 측정 5종(a, b-MLP, b-attentive, c, d)을 probe seed 5개(0..4)로 한다.

### 측정과 데이터 분리 (코드에서 확인한 사실)
- 방향 적합: 009 와 같은 규칙. `RandomState(fit_seed + 학습seed).permutation(Etr)[:round(0.5*Etr)]` 정렬, `exp009.erase.fit_eraser`(leace_scalar, ridge_rel 0.001), 009 의 `forward_hidden`/`pooled_hidden` 을 import 해서 사용. 009/004/002 파일은 수정하지 않는다.
- probe 학습 episode = train split 중 방향 적합에 쓰지 않은 episode(약 절반). 평가 = test episode. `measure.assert_probe_split`/`assert_rows_in` 이 `AssertionError` 로 막고, selftest 가 확인한다(적합 episode 의 특징과 라벨을 쓰레기로 바꿔도 결과가 변하지 않는 시험 포함).
- 학습 행은 8개 캐시 프레임 전부, 평가는 window `late` (009 primary timepoint)를 판정에 사용하고 `early` 도 저장한다.
- (a) 레벨 3개(a060, ind, a160) 다항 로지스틱(L2), 입력 pooled z 표준화. C 는 probe 학습 episode 안에서 GroupKFold(episode 단위, 폴드 수는 002 `probe_impl.inner_folds`)로 검증 log-loss 최소값을 고른다. 평가는 test 의 학습 레벨 episode 만(split `train`).
- (b) MLP: `exp002.probes.mlp_fit_predict` 에 config `measure.b_mlp` 값(은닉 256, 1층, relu, epochs 30, lr 1e-3, wd 1e-2, bs 256, val 0.1, validation 최저 가중치 복원)을 넣어 사용. attentive: `exp002.probes.attentive_fit_predict`, 002 `probe_impl.attentive` 와 같은 구현과 값(일치 검사 `check`/`measure` 시작 시).
- (c) `exp002.probes.linear_pooled` ridge(GroupKFold 로 alpha 선택)를 pooled 행동/proprio 좌표 `[384:404]` 만으로. (d) 같은 ridge 를 pooled z 로. 009 results.json 의 깊이 m_int 값과의 차이는 `d_reproduction_vs_009_at_m_int` 에 split 별로 저장한다(probe 학습 episode 수가 달라 0 이 아닐 수 있음).

### config 에 없어서 실험자가 정한 것 (결과를 보기 전에 고정, `results.json` 에 기록)
1. **판정 split**: config 는 (b), (c) 를 어느 split 에서 판정할지 적지 않았다. 코드는 test 전체 episode 를 합친 `all` 을 판정에 쓰고, `train`/`interp`/`extrap` 별 판정을 `verdict_by_split_b_c` 에 함께 저장한다. (a) 는 config 대로 학습 레벨 test episode(`train`)만. **사용자 확정 2026-11-10** (아래 2–4 도 함께 확정, notes/hypotheses.md 012 확정 기록 2).
2. 로지스틱 구현값: C 격자 [1e-4 .. 1e2] 7개, 선택 기준 검증 log-loss, 감쇠 Newton 솔버(float64), 폴드 seed 0. (a) 는 C 가 정해지면 결정적이라 probe seed 별 값이 1개다 (`results.json` 의 `logistic_impl_defaults`).
3. 판정 값 = 해당 학습 seed 의 probe seed 평균(009 와 같은 방식). (b) 는 학습 seed 마다 MLP 또는 attentive 중 하나라도 ≥ 0.2 이면 그 seed 가 성립. 값이 없거나 NaN 이면 그 항목은 `not_evaluable`(추정치로 대체하지 않음). 세 항목이 모두 `criterion_not_met` 일 때만 "이 측정들로는 잔여 정보 미검출".
4. `pilot` 필드: config `pilot` 이 true 이거나, 결과 폴더가 실험 폴더 `results/` 가 아니거나, `--m-int`/`--train-seeds` 로 바꿨으면 `pilot: true` 와 이유 목록 `pilot_reasons` 를 기록한다. 본 실행의 `results.json` 은 config `pilot: false` 로 바꾼 뒤에만 `pilot: false` 가 된다. 파일럿은 `--results-dir` 로 분리하고 그 폴더에만 `results.json` 을 쓴다.

### 비용 추정 (추정, 012 실측 없음)
- 근거 1: 009 본 실행 `intervene` 3005.9 s (21 단위, Colab L4; `experiments/009_positive_control_depth/results/colab/timing_main.json`). 012 의 순전파/추출은 같은 규모이다.
- 근거 2: 004 `enc_probe` 1286.8 s / 21 셀(k 7 x 통제 3), 로그상 셀당 약 57-68 s 가 linear+attentive x probe seed 5 (`experiments/004_depth_truncation/results/colab/timing_main.json`, `logs/colab_20261109/enc_probe_20261109-093004.log`). 012 의 attentive 는 학습 seed 당 17 깊이 측정 x 5 seed 이고 학습 행은 절반이라, 위 셀 시간의 상한으로 환산하면 학습 seed 3개에서 약 52 분 이하.
- 측정 안 한 것: 깊이마다의 MLP 5회와 로지스틱(CPU 8코어에서 C 선택 포함 약 22 s, 합성 데이터 N=3600, d=384 로 로컬 측정; Colab 은 다를 수 있음).
- 합계 추정: 약 2 시간(범위 1.5-3 시간). 임시 디스크는 기준선 단위 최대 약 13 GB(깊이 6개 x 약 2.2 GB, 추정).

## Colab 노트북 (`colab/exp012_colab.ipynb`)

`python3 colab/build_notebook.py` 로 생성한다(코드 변경 후 다시 실행). 코드를 임베드하지 않고 공개 저장소를 git clone 해서 `REPO_REF` 를 checkout 하고 HEAD 를 `exp012_work/code_version.json` 과 `results.json` 의 `code_version` 에 기록한다. 로컬 경로는 `/content/mylab012`, `/content/venv012`, `/content/cache012`, `/content/pilot012`, 출력은 `MyDrive/exp012_work` 로 006/009/010 과 겹치지 않는다(`/content/dinowm/data` 의 metadata.pt 와 ACWM 소스 checkout 은 같은 읽기 전용 내용을 공유).

셀 순서: A1 GPU 확인, A2 설정(Drive 경로, REPO_URL/REPO_REF), A3 clone + HEAD 기록, A4 환경(uv venv 3.9), A5 ACWM 소스, A6 환경변수(`MPLBACKEND=Agg` 포함)와 결과 폴더 연결, A6b metadata.pt 배치, A7 디스크, B1 입력 점검, B2 `check` + dry-run, C1 selftest, D0 승인 플래그, D1 파일럿(선택), D2 파일럿 시간, E1 본 `measure`(`APPROVED_MAIN_012 = True` 필요), F1 `aggregate`, G1 결과 확인, G2 Drive flush.

사람이 할 일: (1) 이 코드를 공개 저장소에 commit/push 한다(노트북은 clone 한다). (2) 노트북 A2 의 Drive 경로와 `REPO_REF` 를 확인한다. (3) 파일럿을 하려면 D0 의 `APPROVED_PILOT_012 = True`. (4) 본 실행 전 config `pilot: false` 로 바꾸고(또는 `ALLOW_MAIN_WITH_PILOT_TRUE`) `APPROVED_MAIN_012 = True`. GPU 실행 승인은 별도로 사람이 한다.
