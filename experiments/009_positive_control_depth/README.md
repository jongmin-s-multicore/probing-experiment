# 009 양성 대조: 004 깊이 probe 파이프라인의 검정력 (설정 확정, 미실행, GPU 실행 승인 대기)

상태: 설정 사용자 확정 2026-11-10 (소거 방식 `leace_scalar`, 강도 [0, 1.0] (0.25, 0.5 제거), 주 성공 기준, m_int [1..6]; notes/hypotheses.md "009" 확정 기록). 보조 판정 A, B 는 사용자 결정 2026-11-10. **GPU 실행은 사용자 승인 후에만.** 본 실행 전까지 config `pilot: true`.

- 검증할 가설: 004 의 측정 파이프라인(깊이별 linear probe + "직전 최고값보다 0.2 이상 하락" 규칙)이, predictor 블록 m_int 의 출력에서 크기 정보를 실제로 제거했을 때 그 소실을 m* ≤ m_int + 1 에서 검출하는가.
- 방법
  1. 004 와 같은 체크포인트(002 조건 (b), 학습 seed 0,1,2), 같은 002 Q1 캐시, 같은 probe 코드(`exp004.stages.probe_hidden`), 같은 판정 함수(`exp004.depthlib.first_drop`)를 import 로 재사용한다. 004 폴더의 파일은 수정하지 않는다.
  2. 개입: hidden h_{m_int}(m_int 개 블록 통과 직후)의 z 부분(앞 384) 모든 토큰에서 크기 방향을 제거하고(강도 α: x - α·(제거 성분)) 나머지 블록은 그대로 순전파. 행동 좌표는 건드리지 않는다.
  3. 소거 방식(사용자 확정 2026-11-10): `leace_scalar` = 크기 레이블에 대한 선형 소거(소거 후 pooled 특징의 레이블 공분산 0). 대안 `probe_direction` = 선형 probe 가중치 방향 하나만 직교 제거(더 약함). config `intervention.method`.
  4. 방향 적합 데이터와 probe 평가 데이터 분리: 방향은 학습 분할 episode 의 절반(`fit_fraction`, `fit_seed`+학습 seed 로 선택)에서 적합. probe 평가는 test episode(방향 적합에 쓰지 않음). probe 학습은 004 와 같이 학습 분할 전체를 쓴다(적합 episode 포함).
  5. 단위: (블록 m_int, 강도 α, 학습 seed) 각각 독립. α = 0(무개입) 기준선은 학습 seed 당 한 번만 계산해 모든 블록이 공유. 깊이 < m_int 의 곡선은 기준선과 같으므로 단위는 깊이 m_int..L 만 계산하고 집계에서 기준선 행을 이어 붙인다.
  6. 블록마다 따로 판정: 강도 1.0 에서 학습 seed 3개 중 2개 이상이 m* ≤ m_int + 1 (m* = split(train/interp/extrap) 별 첫 하락 깊이의 최솟값, 004 의 m_star 정의) 이면 `criterion_met`. 단위가 빠지면 그 블록은 `not_evaluable`(추정치 대체 없음).
  7. 재개: part 파일(`results/parts/`)이 있는 단위는 건너뜀. `--fresh` 로 다시 계산.
  8. 블록 목록 덮어쓰기: config `intervention.m_int` (기본 `[1, 2, 3, 4, 5, 6]`, 사용자 확정 2026-11-10; 1..6 의 임의 부분집합 가능). 블록 6 은 뒤에 블록이 없으므로 회복 관측은 계산하지 않고 `"해당 없음"` 으로 기록하며 검출 판정만 한다(곡선: 깊이 0..5 기준선 + 깊이 6 개입). CLI `--m-int 2,4` 또는 환경변수 `EXP009_M_INT=2,4` (Colab 노트북 셀에서 설정). 덮어쓰면 `results.json` 에 `m_int_config`, `m_int_used`, `m_int_overridden`, `m_int_source` 가 기록된다.
- 성공 기준(사용자 확정 2026-11-10): config `success` (strength 1.0, min_seeds 2, slack 1). 강도 1.0 에서도 검출되지 않으면 004 의 refuted 를 "검정력 없는 결과"로 표시한다는 것이 hypotheses.md 의 확정 기록이며, 코드는 `criterion_met / criterion_not_met` 만 기록한다. 소거 이후 블록에서 R² 가 회복되는지는 보조 관측(`aux_recovery_r2_minus_r2_at_m_int`)으로만 기록한다(블록 6 은 "해당 없음").
- 보조 판정 A, B (사용자 결정 2026-11-10; 주 성공 기준은 바꾸지 않았고, 보조 판정은 주 판정에 들어가지 않는다). 정의는 `experiments/008_posthoc_reaggregate_004_005/run.py` 에서 복사(스크립트라 import 불가, 008 파일은 수정 안 함; 출처는 `src/exp009/auxrules.py` 머리말). 블록 x 강도 x 학습 시드별로 results.json `blocks.<m>.strengths.<a>.<학습 seed>.aux_A_delta_vs_random_init / aux_B_common_variance_r2` 에 `관측됨 / 관측 안 됨 / 계산 불가` 로 저장하고, 강도 1.0 요약은 `blocks.<m>.aux_rules_at_success_strength`.
  - A: Δ(d) = R²_개입 predictor(d) − R²_random-init predictor(d) 곡선에 0.2 하락 규칙(직전 최고값 대비, split 별 첫 하락 깊이의 최솟값 = m*). **random-init 기준은 009 에서 새로 측정하지 않고 004 `results/depth_r2.csv` 의 대응 행(module predictor, control random_init, probe linear/new/late, level ALL, 같은 학습 seed, probe seed 0..4, 깊이 0..6)을 쓴다.** 같은 probe 코드, split, 시드 대응이어야 하므로 집계 시 키 (깊이, split, probe seed) 가 009 기준선 행과 정확히 같은지 학습 seed 별로 검사하고(`random_init_reference`), 다르거나 파일이 없으면 A 는 `계산 불가`. 004 파일의 키 대응은 확인했다(학습 seed 0,1,2 각 140행, 키 일치). random-init predictor 는 개입을 받지 않는다(개입 predictor 의 깊이별 R² 에서 같은 깊이의 004 random-init R² 를 뺀다). 009 의 강도 0 기준선이 004 real 행을 재현하는지는 기존 `baseline_vs_004` 로 기록한다.
  - B: 008 과 같은 Var_all(전체 평가 episode 공통 라벨 분산; 라벨 = 002 config `data_impl.area_ratio` 의 log, 레벨별 n_episodes 가중) 정의의 공통 분산 R² = 1 − MSE_split/Var_all 에 0.2 하락 규칙. MAE(ALL 행)와 MSE(log 면적비 단위)를 곡선에 함께 기록. 레벨 행으로 복원한 split R² 가 ALL 행 R² 와 1e-6 이내로 맞지 않으면 `계산 불가`(008 과 같은 자기검증).
  - 각 보조 판정 항목에는 `within_m_int_plus_slack` (m* ≤ m_int + 1 여부) 도 같이 기록한다.
- 이 실험으로 내가 주장하려는 것: (실행 후) 이 개입에서 004 의 규칙이 블록 m_int 별로 소실을 검출했는지/못 했는지의 수치. 해석은 하지 않는다.

## 실행 방법
```
cd experiments/009_positive_control_depth
conda activate <002 와 같은 환경>        # CLAUDE.md: Conda 환경. 이 머신에서는 ~/miniconda3/envs/lab07/bin/python 으로 CPU selftest 확인
export DINO_WORK=... DINO_TRAIN=... EXP002_CACHE=... ACWM_PHYS_SRC=... EXP009_CACHE=...   # 002/004 와 같은 값 + 임시 캐시(디스크 약 15 GB)
python run.py selftest                       # 합성 데이터 + toy 모델 (CPU)
python run.py check
python run.py intervene --dry-run
python run.py intervene [--m-int 1,2,3,4,5,6] [--strengths 0,1.0] [--train-seeds 0,1,2] [--fresh]
python run.py aggregate [--m-int ...]        # results.json, results/depth_r2_009.csv
```
중단되면 같은 명령을 다시 실행하면 이어서 진행한다. 재시도 상한 3회(results/attempts.json)는 004 와 같다.

## 필요한 체크포인트/캐시 (모두 읽기 전용)
- 002 조건 (b) run 3개 (`multi_60_ind_160`, seed 0,1,2) 의 `DONE`, `checkpoints/model_latest.pth` (`DINO_TRAIN` 아래).
- 002 Q1 캐시 (`EXP002_CACHE`, suffix 없음).
- 004 `results/depth_r2.csv` (기준선 재현 확인용, 선택).

## 예상 GPU 시간 (추정: 실측 없음)
- 근거: 004 Colab pred 단계 5345 s (3 학습 seed, real 과 random-init, linear+attentive, label shuffle 포함; `004/results/colab/timing_main.json`), 005 feat 137 s (모델 2개 x 1층 전체 episode 순전파; 같은 로그).
- 실행 단위 수: 21 = 기준선 3(학습 seed 당 1, 모든 블록 공유) + 개입 18 (블록 6 x 강도 1(1.0) x 학습 seed 3). 방향 적합은 (블록, 학습 seed) 당 1회 = 18회(강도 수와 무관). `python run.py intervene --dry-run` 이 같은 수를 출력한다.
- 기준선 1회(학습 seed 3개, 깊이 0..6 전체, linear 만): 약 10~20분 (추정).
- 블록 1개당(추정): 방향 적합 3회(학습 seed 당 1회) + 개입 단위 3개(강도 1.0 x 학습 seed 3). 이전 초안(강도 3개, 블록당 9 단위 + 적합 3회 = 0.5~1시간 추정)에서 적합 시간을 약 10~15분, 나머지를 단위당 약 2~5분으로 가정하면(가정, 실측 없음) 블록당 약 16~30분 -> 블록 6개 약 1.6~3시간 + 기준선 약 10~20분 = 합계 약 1.8~3.3시간 (추정, 실측 없음). 깊이 m..6 만 계산하므로 뒤쪽 블록 단위는 더 짧을 것이다(추정). 보조 판정 A, B 는 CPU 집계만 추가되고 GPU 시간은 늘지 않는다.
- 임시 디스크: 깊이 하나당 약 2.2 GB (E=1800, 8 프레임, 196 패치, 384, fp16), 단위당 최대 약 15 GB, 단위 끝에 삭제.
- 실측이 아니므로 첫 단위의 로그 시간을 보고 다시 추정해야 한다.

## 알려진 한계 / selftest 에서 확인된 사실 (해석 아님)
- (참고: 강도 0.25, 0.5 는 확정으로 제거됨; 코드는 부분 강도를 여전히 지원) selftest(C 절)에서, 부분 강도(0.5)는 크기 성분을 축소하기만 하므로 다시 학습한 linear probe 의 R² 가 기준선과 거의 같았다(toy). 실제 모델에서도 그런지는 실행되지 않았으므로 모른다.
- 새 probe 를 다시 학습하므로 이 대조는 "선형 소거 후에도 비선형/다른 방향으로 남은 정보"를 보지 않는다(linear probe 만).
- Colab 노트북: `colab/exp009_colab.ipynb` (`colab/build_notebook.py` 로 생성; 코드 임베드 없음). 셀 A3 가 공개 저장소 `REPO_URL` (https://github.com/jongmin-s-multicore/probing-experiment.git) 을 git clone 하고 `REPO_REF` (셀 A2; 기본 'master', 브랜치 이름 또는 commit 해시/태그) 를 detach checkout 한다. 토큰(`GITHUB_TOKEN`)은 선택 사항(기본 None). HEAD commit 해시는 출력되고 `exp009_work/code_version.json` 과 `results.json` 의 `code_version` (`git_head`, `git_status_dirty`, `repo_url`, `requested_ref`) 에 기록된다. 노트북은 저장소 ref 에 002, 004(`colab/exp045_tools.py`, `colab/requirements_colab_extra.txt`, `src`, `run.py`, `config.yaml`), 009 파일이 commit 되어 있어야 동작한다. 셀 순서: 설정 -> clone -> 환경 -> metadata.pt 배치(A6b) -> check -> selftest -> (선택) 파일럿 1단위 -> intervene -> aggregate -> 결과 복사. 본 실행 셀은 `APPROVED_MAIN_009 = True` 를 사람이 설정해야 돈다. config 값은 노트북에서 덮어쓰지 않는다. 입력은 Drive 의 002 체크포인트/캐시, 004 `results/depth_r2.csv`, 그리고 학습 manifest 의 episode dir 별 `metadata.pt`(행동 포함; 행동 mean/std 계산에 필요, mp4 는 불필요, 정정 2026-11-10: 이전 문구 \"데이터 다운로드 없음\"은 틀렸음. 출처 우선순위: 로컬, DATA_SRC, WORK002/cloth_sim_data.tar 의 해당 멤버). `run.py check` 와 `run.py intervene` 은 시작 시 metadata.pt 존재를 검사하고 없으면 attempts 를 소비하지 않는 사전 점검 오류(종료 코드 4)로 끝난다. 노트북은 CPU 머신에서 JSON/문법만 검사했고 Colab 에서는 실행된 적 없다.
- 코드는 toy 모델에서만 실행됨(selftest). 실제 체크포인트에서는 실행된 적 없음.
