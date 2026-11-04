# 실험 설명: 002_single_size_sweep

- 상태: **코드 작성 완료(확정 기록 6, 7 반영 2026-11-02), 서버 미실행.** 이 머신(GPU/데이터/체크포인트 없음)에서는 문법 검사, `--dry-run`, 합성 데이터 CPU 셀프테스트만 했다. 학습, 데이터 다운로드, GPU 실행, 서버 접속은 하지 않았다. 이 문서에 실험 결과 수치는 없다.
- 검증할 가설: notes/hypotheses.md 후보 G (확정 기록 3, 4, 5 로 사용자가 확정/사전 등록). Q1 인코딩 H-a, Q2 보존 H-b, Q3 사용 H-c. 출처 설계 문서 요약은 hypotheses.md 에 있다.
- **이 실험으로 내가 주장하려는 것** (확정 전 초안, 해석은 분석가 리뷰 몫): 조건 (b) a060+ind+a160 로 학습한 DINO-WM 에서 (1) 인코더 출력(DINOv2 패치 토큰)이 mask 면적/bbox baseline 과 random-init 인코더보다 더 천 크기를 복원하는가 (H-a), (2) 그 정보가 predictor 마지막 블록과 rollout 마지막 시점까지 Q1 대비 `q2.drop_threshold` 미만의 R² 하락으로 남는가 (H-b), (3) 크기 방향을 이동/제거하면 같은 분산·같은 이동 크기의 무작위 방향 100개보다 예측이 더 변하고 alpha 에 단조인가 (H-c). 위 세 가지에 대한 판정 표(pass/fail/not_evaluable/pending/exploratory)와 수치만 만든다. "사용된다/버린다" 같은 해석은 개입 결과가 있어도 이 코드가 쓰지 않는다. (a) 단일 크기 조건과의 비교는 별도 패밀리로 선언만 했고 (a) 재학습 전에는 `pending`.
- 성공 기준: config.yaml 의 사전 등록 값(아래 "config 값이 읽히는 위치")을 코드가 읽어 판정한다. 임계값이 null 이면 `not_evaluable`.

## 1. 구조

```
experiments/002_single_size_sweep/
  config.yaml          사전 등록 값 + (파일 아래쪽) 실험자가 추가한 구현 파라미터. 시드 전부 여기
  environment.yml      conda 환경 정의 (미검증)
  run.py               단계 CLI: check download manifest smoke cache train q1 q2 q3 aggregate selftest
  overnight.sh, overnight_status.py   연속 실행기 (단계당 최대 3회 재시도, 실패 이유를 logs/ 에 남김)
  src/exp002/
    common.py     config(중복 키 거부), PreregReader, 경로, 파일 로그(Tee), 재시도 가드, 환경 정보
    drive.py      [001 drive.py 복사] rclone 다운로드 + 무결성 점검 (env EXP002_*)
    masks.py      [001 masks.py 복사+확장] HSV 천 마스크, 면적/bbox
    data.py       [001 data.py 개작] manifest, ManifestDataset(브랜치 클래스 상속), hydra 로더
    model_io.py   [001 extract.py 개작] 체크포인트 로드, StreamHooks(잔차 스트림 hook), 윈도우, rollout
    cache.py      Q1 캐시 (u8 프레임, DINOv2 토큰, random-init 토큰, 마스크 특징, 원시 행동)
    stats.py      numpy: 에피소드 통계, 지표, cluster bootstrap, Holm, Spearman
    directions.py numpy: ridge(GroupKFold), 분산 맞춤 무작위 방향, INLP, LEACE
    probes.py     linear_pooled / attentive / mlp probe (같은 용량 설정)
    evalgrid.py   Q1/Q2 공용: 라벨, 셀(에피소드별 충분통계) 생성, 요약
    q1.py q2.py q3.py   단계 구현
    aggregate.py  results.json 과 판정 표
    smoke.py trainrun.py selftest.py
  manifests/ results/ logs/   (실행 시 생성)
```

### 데이터와 조건
- 데이터: 001 과 같은 Drive 폴더(rclone), `$DATASET_DIR/cloth_sim/<split>`. 001 이 받은 split 의 완료 표시 파일(`.exp001_download_ok`)을 그대로 인정한다(재다운로드 없음). 필요한 split: `a060_train, ind_train, a160_train` 과 9개 `*_test`.
- 조건 (b) `multi_60_ind_160`: a060_train, ind_train, a160_train 각 앞 300 episode (metadata 순서). 조건 (a) `single_ind`: ind_train 의 앞 900 (확정 기록 6, 9절). 평가: 전 크기 `*_test` 앞 100. 분할은 항상 episode 단위.
- 역할: 보간 a080/a125, 외삽 a026/a040/a200/a248 (config `data.roles`), `trained` = a060/ind/a160 (test split 평가용).
- **probe 학습 episode 는 두 조건 모두 (b) 의 학습 episode (3크기 x 300)** 이다. 이유: Q1 은 얼어 있는 DINOv2 라 학습 조건과 무관하고, (a) 의 ind 단일 크기에서는 명목 크기 라벨이 상수라 probe 를 학습할 수 없다. 대신 조건 (a) 의 Q2 probe 는 (a) predictor 가 보지 않은 크기의 활성에서 학습된다(한계).

## 2. 단계별 설명

| 단계 | 하는 일 | GPU |
|---|---|---|
| `check` | 환경/경로/디스크/데이터/torch/arch/GPU/체크포인트 점검. 읽기만 | 선택 |
| `download` | rclone 으로 필요한 split 다운로드 (001 drive.py) | 불필요 |
| `manifest` | `metadata.pt` 목록 -> manifests/{train_<cond>, cache_episodes}.json | 불필요(torch) |
| `smoke` | 12GB 메모리/시간 실측 -> results/smoke.json | 필요 |
| `train` | 브랜치 `train.py` 를 수정 없이 호출 (hydra override: manifest 로더, run dir). 조건/시드 선택 | 필요 |
| `cache` | 학습된 run 하나에서 동결 인코더만 써서 u8 프레임, DINOv2 토큰, random-init 토큰, 마스크 특징, 원시 행동 저장. 디스크 추정 출력 | 필요 |
| `q1` | 캐시만으로 probe/통제 3종 -> results/q1/ | 선택(attentive) |
| `q2` | run 별: predictor 각 블록 출력과 rollout 예측 latent 추출(임시 디스크) 후 probe -> results/<run>/q2_* | 필요 |
| `q3` | run 별: steering, INLP/LEACE 제거, 무작위 대조 100개, 행동 대조(탐색) -> results/<run>/q3.json | 필요 |
| `aggregate` | 위 파일들 -> results.json (판정 표, bootstrap, Holm) | 불필요 |
| `selftest` | 합성 데이터 CPU 셀프테스트 (결과 아님) | 불필요 |

### Q1 (H-a)
- 입력: 패치 토큰. probe: `linear_pooled`(평균 풀링 -> 표준화 ridge, alpha 는 학습 episode 안 GroupKFold), `attentive`(LayerNorm -> Linear -> 1-query cross-attention -> 잔차 MLP -> linear head; 조기 종료는 학습 episode 중 10% 검증). 용량 설정은 `config.probe_impl` 한 곳에서 Q1/Q2/Q3 가 공유.
- 타깃 (config `q1.*`, **사람 확인 필요한 설계 결정**):
  - 주 타깃 `log_nominal_area_ratio`: 크기(레벨)의 명목 면적비 log, episode 상수. 이유: 같은 프레임 마스크 면적의 log 를 타깃으로 하면 "마스크 면적 baseline" 이 설계상 R²≈1 이 되어 H-a 의 `delta >= 0.1` 이 성립할 수 없다(문서의 "baseline 이 R²≈1 이면 면적 이상 정보 여부로 바꾼다" 상황).
  - 보조 타깃 `log_frame_mask_area`: 같은 프레임의 마스크 면적 log. 라벨은 입력 latent 와 같은 프레임(한 프레임 어긋남 없음, 001 지적 해소). mask baseline R²가 `q1.baseline_saturated_r2`(0.99) 이상이면 해당 H-a 행은 `not_evaluable (baseline_saturated)`.
  - 면적비 순서 정확도: 학습한 3개 크기만으로는 9-class 분류가 불가능하므로 서로 다른 크기 episode 쌍의 순서 정확도(`pair_order_accuracy`)를 점추정으로만 병기한다.
- 통제: random-init DINOv2(같은 hub 구조, 시드 config), 라벨 셔플(학습 라벨만, episode 단위로 행 치환, 시드는 `probe_impl.shuffle_seed`+probe seed), mask 면적/bbox ridge baseline(특징: log 면적, log 폭, log 높이, log bbox 넓이, 면적/bbox 넓이).
- 시간 축: 프레임 인덱스로 정의한 `early_flat`[0-3], `late_deformed`[8-11], 전체[0-11], union. **평평함/변형이 큼은 인덱스 가정이며 영상으로 확인하지 않았다.** 학습 방식 `all`(전 프레임), `early`(초기 프레임 학습 -> 후반 평가 포함), `union`.
- 지표: R², MAE, Pearson r, 군집(레벨) 내 R². 역할(보간/외삽/미지/학습/전체)별. 학습 seed 는 Q1 에 없다(인코더가 동결). **probe seed 5개별 값**을 `per_probe_seed` 로 항상 병기한다.

### Q2 (H-b)
- 각 블록 출력 j=0..L (j=0: predictor 입력 스트림, j=L: 마지막 블록 출력)의 마지막 context 프레임 토큰에 Q1 과 같은 probe. rollout: 마지막 context 프레임 t=num_hist-1 에서 autoregressive rollout, t+1..t+K 예측 latent 에 같은 step 으로 학습한 probe.
- 상한(Q1)은 같은 episode, 같은 프레임, 같은 probe 로 인코더 토큰에서 구한 값 (`q1` 의 `union` 셀, 단일 프레임 `f<k>` 셀). layer 0 은 기준선.
- 액션이 concat 된 404 차원: 옵션 `q2_impl.feature_part` = `full`(404 전체, **기본**) | `z_only`(앞 384). 기본이 full 인 이유: 블록 1 이후 잔차 스트림의 앞 384 좌표는 z 의 부분공간이 아니라 임의 좌표라서 z_only 는 정보를 잘라낼 위험이 있고, layer 0 기준선도 같은 404 에서 재야 비교가 같아지기 때문. 단점: layer 0 에 액션 차원이 섞이고 Q1(384)과 입력 차원이 다르다.
- 주 검정은 정확히 2점: (마지막 블록, 마지막 rollout 시점) 의 `Q1 R² - Q2 R²` (probe=`q2_impl.primary_probe`, 평가=미지 크기). 층 x 시점 격자는 `exploratory` 표시의 점추정만 나오고 p값/유의 문구를 만들지 않는다.

### Q3 (H-c)
- 개입 층: `q3.layers` = middle, last. 블록 수 L 은 모델에서 읽고 `middle = L // 2`, `last = L` (= 마지막 블록 출력)로 치환한다. 치환 규칙과 구체 인덱스를 로그와 `q3.json` 의 `layer_rule` 에 쓴다. 데이터를 보고 고르지 않는다.
- Steering: `x_j <- x_j + alpha * sigma_u * d` (모든 토큰). d = 학습 episode 에서 구한 단위 방향(ridge 방향 / 평균 차이(가장 큰 학습 크기 - 가장 작은)). sigma_u = 학습 샘플의 풀링 투영 표준편차(평가 데이터로 계산 안 함). alpha = `q3.dose_alpha_sigma_units`.
- 제거: INLP(ridge 반복, `q3_impl.inlp_iterations`)와 LEACE(numpy 직접 구현). 제거 후 MLP probe(학습 데이터에서 제거한 특징으로 학습, 평가 특징에서 읽기)의 R² 가 `q3_impl.removal_check.mlp_r2_after_max` 이하일 때만 `removed`; 아니면 `flag: not_removed` 이고 해당 결론은 만들지 않는다(판정 `not_evaluable`).
- 지표는 **절대 변화량**: (i) 개입 후 예측 latent 를 Q1 ridge 로 읽은 크기의 변화(주, `q3_impl.primary_metric`), (ii) 예측 visual latent 의 패치 평균 L2 변화. 비율 지표 없음. 효과 T = 0 이 아닌 dose 들에 대한 샘플 중앙값의 중앙값.
- 무작위 대조: 방향 100개 (`q3.n_random_directions`), 데이터 부분공간(공분산의 수치적 rank 범위) 안에서 **같은 풀링 투영 분산, 같은 이동 크기**(`directions.matched_random_direction`: 임의 방향을 가장 낮은/높은 분산 고유방향 쪽으로 기울여 목표 분산을 감싼 뒤 이분법). 등방 가우시안 아님. 제거 대조는 같은 rank 의 부분공간(각 방향 분산을 INLP/LEACE 방향 분산에 맞춤, 총 분산 상대 오차를 기록). 방향 시드 = `SeedSequence([q3_impl.random_direction_seed, 학습 seed, 층, 종류])` 이므로 학습 seed 마다 독립이다(001 에서는 같은 방향이 3 seed 에 쓰였음).
- 판정: 크기 방향 T 가 무작위 T 분포의 `q3.criterion_percentile`(95) 백분위 초과, alpha 에 대해 Spearman |rho| >= `q3_impl.monotonic_min_spearman`(0.9; 원래 1.0, 9절). α 곡선은 에피소드 평균. 집계 T 는 중앙값/순위.
- 행동 수준 대조(탐색): 같은 `ind` 레벨의 다른 episode 쌍에서 시각 부분만 교환(행동 차원 유지)하고 읽은 크기 변화를 기록. **크기만 다른 쌍 데이터가 없어 InD 크기 내 대조 쌍으로 대체**(한계: episode 내용이 함께 다르다). p값 없음.

### 통계와 판정 (aggregate)
- episode cluster bootstrap 95% CI: 레벨 안에서 episode 를 복원추출(레벨 비율 고정), probe seed 평균이 통계량, 학습 seed 가 여럿이면 seed 별 통계량의 평균. **CI 는 학습 seed 변동을 포함하지 않는다** (results.json `ci_caveat` 와 각 행의 `ci_note`). 학습 seed 별 값은 `per_training_seed`, probe seed 별 값은 `per_probe_seed`.
- Holm (alpha=`prereg.holm_alpha`): 가설마다 별개 패밀리. H_a 4개(역할 x probe), H_b 2개(점), H_c 2개(개입). not_evaluable 은 p=1 로 취급하되 패밀리 크기 m 에는 포함한다(사전 등록 크기 유지). 가설 간 보정 없음.
  - H_a 행: pass = (R²real - R²mask baseline >= `q1_margin.delta_r2_vs_mask_area_baseline`) 이고 (그 차의 CI 하한 > `ci_lower_gt`) 이고 (R²real - R²random-init 의 CI 하한 > `ci_lower_gt`) 이고 Holm 조정 p < alpha. p = max(두 차에 대한 단측 bootstrap p).
  - H_b 행: p = H0 `drop >= q2.drop_threshold` 의 단측 bootstrap p. pass = Holm 조정 p < alpha (즉 하락이 임계 미만). 모든 학습 seed 가 임계 미만인지(`all_training_seeds_drop_below_threshold`)는 별도 필드.
  - H_c 행: (학습 seed x 고정 층) 모든 칸에서 백분위 밖 + 단조이고 칸별 순위 p 의 최댓값(교집합 검정)이 Holm 통과.
  - `cond_a_vs_b`: (a) 결과가 없으면 `pending`. 있으면 두 H_b 점에서 `drop_a - drop_b` 의 양측 bootstrap p, 별도 패밀리 Holm.
  - `exploratory`: p값/판정 없음.
- results.json: `decision_table` (가설, 검정, 사용한 임계값과 config 경로, 결과, 상태, p, Holm p), `prereg_values_used`(코드가 실제로 읽은 값과 경로), `environment`(python 경로, conda 환경, 주요 패키지 버전, GPU), git commit/dirty, config sha256, `attempts`, `smoke`. 해석 문장은 없다(`interpretation: null`).

## 3. config 값이 코드에서 읽히는 위치

| 값 | config 경로 | 읽는 곳 | 비고 |
|---|---|---|---|
| Q1 마진 0.1 | `prereg.q1_margin.delta_r2_vs_mask_area_baseline` | `aggregate.h_a` (PreregReader) | null 이면 not_evaluable |
| CI 하한 > 0 | `prereg.q1_margin.ci_lower_gt` | `aggregate.h_a` | 〃 |
| Q2 하락 0.2 | `q2.drop_threshold` | `aggregate.h_b` | 〃 |
| Holm alpha | `prereg.holm_alpha` | `aggregate.h_a/h_b/h_c/cond_a_vs_b` | 사전 등록 문구의 0.05 를 숫자로 옮김 |
| 패밀리 구성 | `prereg.family_defs` | `aggregate.h_a/h_b/h_c` | 문구를 구조화해 옮김 |
| 무작위 방향 수 100 | `q3.n_random_directions` | `q3.run_layer`(생성), `aggregate.h_c`(저장값과 일치 검사) | 불일치면 not_evaluable |
| 95 백분위 | `q3.criterion_percentile` | `q3.run_layer`, `aggregate._rank_test`(저장된 T 로 재계산) | 사전 등록 문구에서 옮김 |
| dose alpha | `q3.dose_alpha_sigma_units` | `q3.run_layer`, `aggregate.h_c`(일치 검사) | |
| 층 middle/last | `q3.layers` | `q3.layer_indices`, `aggregate.h_c`(일치 검사) | 인덱스 치환 규칙 기록 |
| 제거 방법 | `q3.removal_methods` | `q3.run_layer` | |
| probe 종류/seed | `probe.kinds`, `probe.seeds` | `q1.run`, `evalgrid`, `q2.run` | 통제 3종 목록 `probe.controls` 는 문서용(구현은 고정 3종) |
| 학습 seed | `train.seeds` | `run.py`, `aggregate` | |
| 전역 시드 | `seed` | `run.py` set_seed | |
| bootstrap/셔플/random-init/방향 시드 | `probe_impl.bootstrap_seed/shuffle_seed/random_init.seed`, `q3_impl.random_direction_seed` | `aggregate`, `evalgrid`, `cache`, `q3` | |
| 조건/순서 | `conditions.*` | `run.py`, `data`, `aggregate` | 주 검정 조건 = `run_order[0]` |
| 학습 설정 | `train.batch_size/num_hist/epochs/budget` | `trainrun.build_cmd`, `q2`, `q3` (`common.require`) | **null -> ConfigNull 로 중단**(dry-run 은 경고) |
| 그 밖 구현 파라미터 | `q1`, `q2_impl`, `q3_impl`, `probe_impl`, `frames`, `smoke`, `train_impl` | 각 단계 | 사전 등록 값이 아님. 사람 확정 대기 |

결과를 본 뒤 값을 바꾸는 코드/옵션은 없다. (`--config` 로 다른 파일을 줄 수는 있으나 results.json 에 config sha256 이 남는다.)

## 4. 환경 (conda)

- 규칙: CLAUDE.md "Conda 환경에 맞춰서 실행". 모든 단계는 활성화된 conda 환경의 `python` 을 쓴다. 러너(`overnight.sh`)는 활성화된 환경을 가정하고, `CONDA_ENV` 가 주어지면 `conda.sh` 를 source 해서 activate 한다. `conda run -n` 을 쓰지 않은 이유: 출력을 버퍼링해 진행 상황이 로그에 실시간으로 안 남는다.
- **dino_wm 재사용 판단 (의존성 비교)**: 002 가 직접 import 하는 것은 numpy, pyyaml, torch, opencv(cv2), decord, einops, hydra, omegaconf 이고 모두 001 이 이미 쓴 것이다(001 은 sklearn 도 썼지만 002 는 sklearn/scipy 를 import 하지 않는다: ridge/bootstrap 을 numpy 로 직접 구현). 브랜치 `train.py` 의 의존성도 001 과 같다. 따라서 **서버에 001 용 `dino_wm` 이 이미 있고 `python run.py check` 가 위 패키지를 모두 보여 주면 그대로 재사용 가능**하다(추정: 이 판단은 패키지 목록 비교이며 실제 서버 환경을 확인하지 않았다). 12GB 서버가 001 서버와 다른 컴퓨터이면 `dino_wm` 이 없을 수 있으므로 `environment.yml` 로 새로 만든다. 별도 환경이 필요한 추가 의존성은 없다.
- 생성/활성화:
  ```bash
  conda env create -f environment.yml      # dino_wm 이 없을 때. 이미 있으면 이 줄은 건너뛴다
  conda activate dino_wm
  python run.py check                      # python 경로, CONDA_DEFAULT_ENV, torch/cuda 확인
  ```
  기존 `dino_wm` 에 빠진 패키지만 있다면 `conda env update -n dino_wm -f environment.yml` (기존 환경을 수정하므로 주의).
- `environment.yml` 에서 **검증되지 않은 것(서버에서 확인 필요)**: torch/torchvision 버전(의도적으로 무고정; 001 서버는 2.7.1+cu128, 브랜치 pin 은 2.3.0, `config.yaml env.torch_version` 은 null), CUDA wheel index(cu128 은 001 서버 기준; 12GB 서버 GPU 구조는 미확인), opencv-python 4.6.0.66 + numpy 1.26.4 조합, 브랜치 train.py 가 실제로 import 하는 목록(001 `setup.sh --minimal` 에서 복사), `conda env create` 자체(실행해 본 적 없음).
- `run.py check` 와 `smoke` 는 `CONDA_DEFAULT_ENV` 가 비면 경고만 낸다(중단하지 않음). smoke.json 과 results.json 의 `environment` 에 python 경로, conda 환경 이름, torch/numpy/scipy/sklearn 등 버전이 기록된다.
- 하드코딩된 절대 경로 없음: `DINO_WORK`(기본 `~/dinowm`), `DATASET_DIR`, `DINO_TRAIN`, `EXP002_CACHE`, `ACWM_PHYS_SRC` 환경변수. 브랜치 checkout 은 001 의 `setup.sh`(commit 5a640be)로 만든 `$ACWM_PHYS_SRC` 를 그대로 쓴다.

## 5. 사용자 테스트 순서와 정확한 명령 (서버에서 직접)

작업 디렉터리 `experiments/002_single_size_sweep`. 아래 "정상 출력"은 **형태 예시이며 작성자가 서버에서 실행해 본 것이 아니다**(selftest 만 이 머신에서 합성 데이터로 실행). 막히면 각 단계가 `logs/` 에 남긴 파일(`logs/<단계>_<시각>.log`, 터미널 첫 줄에 경로 출력, 학습 smoke 는 `logs/smoke_train_h<H>_b<B>.log`)을 붙여 달라. 에러는 숨기지 않고 traceback 까지 같은 파일에 남는다. 3회 연속 실패하면 해당 단계는 `results/attempts.json` 때문에 실행을 거부한다(원인 보고 후 사람이 항목을 지워야 재개).

```bash
# 0. 환경 (위 4절)
conda activate dino_wm            # 또는 conda env create -f environment.yml 후
export DINO_WORK=...  DATASET_DIR=...  DINO_TRAIN=...  EXP002_CACHE=...  ACWM_PHYS_SRC=...   # 001 과 같은 값 + EXP002_CACHE

# 1. 셀프테스트 (CPU, 합성 데이터, 수 분)
python run.py selftest
```
정상 출력 끝부분: `selftest summary: passed=60 failed=0 skipped=0` (torch 가 없으면 `skipped(no torch)` 항목이 나오고 통과로 세지 않는다). 위 숫자는 작성자 머신(CPU torch 2.15, numpy 2.5, python 3.14)에서 나온 값이고 서버(python 3.9)에서는 미실행. 이 숫자는 결과가 아니다.

```bash
# 2. 환경/경로 점검 (읽기만)
python run.py check
```
확인: `python executable`, `CONDA_DEFAULT_ENV`, `torch cuda ... gpu ... mem_gb`, `arch_list`(GPU sm 이 포함되는지), 경로 존재, 브랜치 `matches config.code.commit: True`, 디스크, split 존재, null 목록. 막히면 `logs/check_*.log`.

```bash
# 3. 스모크: 12GB 메모리/시간 실측 (데이터와 manifest 가 있어야 함)
python run.py download --dry-run        # 필요한 split 과 용량 (rclone 설정 필요, 001 과 동일)
python run.py download                  # 001 에서 받은 split 은 건너뜀. a060_train, a160_train, 일부 test 가 새로 필요할 수 있음
python run.py manifest                  # manifests/*.json
python run.py smoke --dry-run           # 계획만
python run.py smoke                     # results/smoke.json
```
`smoke` 가 하는 일: (b) num_hist 후보 [1,3] x batch [24,16,12,8,4] 순으로 브랜치 train.py 를 1 epoch/48 episode 로 실행, 처음 성공하는 batch 를 기록(OOM 이면 다음 후보). nvidia-smi 폴링으로 최대 VRAM, 전체 wall time. (a) 1 episode 디코딩+224 리사이즈+인코딩 시간과 torch 최대 VRAM, 전체 캐시 시간 추정. (c) hook 포함 Q2 추론 VRAM, hook 항등/잔차 경로 점검 등 인터페이스 가정 확인. 끝에 `RECOMMENDATION (config not changed): ...` 를 출력한다. batch 24 가 12GB 에 안 들어가면 최대 batch 와 gradient accumulation 권고를 출력하지만 **config 는 자동으로 바꾸지 않는다**(브랜치 train.py 의 accumulation 지원은 미확인). smoke.json 에 python 경로/conda/패키지 버전 포함. 막히면 `logs/smoke_*.log`, `logs/smoke_train_*.log`.

```bash
# 4. 사람이 config.yaml 의 null 4개(train.batch_size, num_hist, epochs, budget)와 env.torch_version 을 채운 뒤
python run.py cache --dry-run           # 디스크 추정(약 10.6 GB: 프레임 3.5 + 토큰 3.5 x2, 추정식) 과 여유 공간
python run.py train --dry-run           # 실행할 train.py 명령 출력 (null 이면 어디서 멈추는지 경고)
python run.py train                     # 조건 (b) 3 seed 학습 (GPU, 승인 필요). 단일 seed: --seeds 0
python run.py cache                     # 학습된 run(기본 (b) seed 0)의 동결 인코더로 Q1 캐시
python run.py q1                        # CPU/GPU, results/q1/
python run.py q2 --dry-run && python run.py q2     # run 별. 임시 디스크 약 19 GB/run (추정식, q2 --dry-run 이 출력)
python run.py q3 --dry-run && python run.py q3
python run.py aggregate                 # results.json
bash overnight.sh --dry-run             # 위 전체를 연속 실행하는 러너 계획 (실제는 nohup bash overnight.sh)
```
(a) 단일 크기 조건(ind 900 에피소드)은 나중에 `--condition single_ind` (러너는 `CONDITIONS=single_ind`) 로 같은 서버에서 학습/분석한 뒤 `aggregate` 를 다시 돌리면 `cond_a_vs_b` 행이 `pending` 에서 계산된다.

학습 예산: 9절 (step 기반 동일 예산, `budget_mode: steps_equalized`). 이 줄의 이전 설명(epochs 만, budget 필수)은 폐기.

## 6. 설계 결정 요약
- 001 지적 해소: 사전 등록 값을 코드에 연결(위 표), null 이면 판정 안 함, 라벨은 입력과 같은 프레임, 대조군 3종, 하이퍼파라미터는 학습 fold 안에서만(ridge alpha GroupKFold, attentive 조기 종료는 학습 episode 중 일부), 절대 변화량만, 무작위 방향 100개를 학습 seed 마다 독립 생성, 분산 맞춤, 상태 파일 스크립트의 인자 파싱(`parse_intermixed_args`)과 stderr 로깅.
- **잔차 경로 주의**: 001 의 hook 은 캡처만 했다. 개입은 블록이 `x = attn(x) + x` 구조일 때 attn 입력만 바꾸면 잔차 경로가 그대로라서 효과가 사라진다(셀프테스트의 장난감 모델에서 이 오류를 실제로 발견). `StreamHooks` 는 attn 입력을 바꾸고 출력에 변화량을 더해 `attn(x') + x'` 가 되게 한다. 브랜치 블록이 정말 이 구조인지는 `smoke` (c) 의 `residual_path_probe_ok_if_ge_0.5` 로 서버에서 확인해야 한다.
- 인코더 토큰은 저장한 u8 프레임에서 계산하므로 Q1 과 Q2 layer 0 입력의 픽셀이 동일하다(디코딩 후 uint8 양자화 포함).
- Q1 에는 학습 seed 차원이 없다(동결 인코더). 모든 run 의 인코더 지문(sha256)이 캐시와 같은지 q2/q3 가 확인하고 다르면 중단한다.

## 7. 알려진 한계와 미검증 사항 (정직하게)
- **서버에서 실행되지 않은 코드**: `cache.py`, `q2.py` 의 추출부, `q3.py` 의 `run()`(모델 로드/윈도우 생성), `smoke.py`, `trainrun.py`, `data.py` 의 torch/decord 부분, `download`(실제 rclone), `model_io.load_world_model`. 합성 장난감 모델로 검증한 것은 hook 로직과 Q3 코어, 판정 로직뿐이다.
- 브랜치 인터페이스 가정(미검증): `model.encode(obs, acts) -> (B,K,196,404)`, `model.predict(z (N,H,196,404))`, `model.replace_actions_from_z`, `model.predictor.transformer.layers = [(attn, ff)]`, `transformer.norm`, `encoder.emb_dim=384`, `model.encoder_transform`, 토큰이 프레임 우선 순서, checkpoint 키(`predictor, decoder, proprio_encoder, action_encoder, epoch`), `hydra.yaml`, `train_encoder=false`. 001 코드를 따랐으나 002 의 `num_hist` 가 1 이 아닐 때의 동작은 001 에 없다.
- predictor 블록 수: 코드가 모델에서 읽는다(001 주석은 6 블록을 시사하나 미확인). 임시 디스크 추정식은 6 블록을 가정.
- 장난감 모델 셀프테스트는 크기 정보가 한 방향에 선형으로 박힌 경우다. 실제 데이터에서 크기 방향이 데이터 분산이 가장 큰 방향이면 "같은 투영 분산" 대조가 그 방향과 거의 같은 방향만 허용해 구분력이 없어진다(셀프테스트에서 실제로 관찰). 결과 해석 시 `random.variance_rel_err_max`, 무작위 방향과 크기 방향의 코사인(미저장)을 함께 봐야 한다.
- LEACE 는 기억에 의존한 닫힌형 구현이며 원 논문/공식 구현과 대조하지 않았다(셀프테스트는 선형 정보 소실만 확인). INLP 는 반복 횟수가 모자라면 정보가 남을 수 있고 그 경우 `not_removed` 로 걸러진다.
- 프레임 구간(early/late)은 인덱스 가정, 마스크 규칙(HSV)이 재생성 데이터에 맞는지는 001 `maskcheck` 로 확인한 적이 있으나 002 에서는 재검증하지 않았다.
- 크기 라벨: 명목 면적비 타깃은 episode 상수이므로 "접힘/가림" 영향은 mask baseline 이 받는다. 보조 타깃에서는 baseline 이 포화될 수 있다.
- 12GB 학습 가능 여부, 단계별 소요 시간, 임시 디스크 크기는 모두 미측정(추정식만).
- 조건 (a)의 Q1 은 (b) 와 동일 probe 학습 episode 를 쓴다(위 1절). 같은 서버에서 baseline 재학습 요구는 코드가 조건 선택을 지원할 뿐 실행하지 않았다.
- 결과 해석과 논문 주장 확정은 이 코드의 범위가 아니다 (/review-experiment 와 사람 승인).

## 8. 사람이 정해야 할 값
`train.batch_size`, `train.num_hist`, `train.epochs`, `train.budget`(동일 예산 정의), `env.torch_version`, 그리고 코드 작성자가 임시 기본값을 둔 구현 파라미터(타깃 정의 `q1.primary_target`, 프레임 구간, `q2_impl.primary_probe/feature_part`, `q3_impl.*` 주 방향/제거 방법/지표/단조 기준/제거 확인 임계/에피소드·프레임 수, `probe_impl.*` 용량·bootstrap 횟수) 승인 또는 수정. 이들은 데이터를 보기 전에 고정해야 한다.


## 9. 확정 기록 6, 7 반영 (2026-11-02): 결정 사항, 변경 점, 서버 테스트 순서

아래는 notes/hypotheses.md 확정 기록 6, 7 의 구현이다. 사전 등록 후 변경의 원래 값/새 값은 확정 기록 7 B 표에 있고 config 주석에도 있다 (조건 (a) 에피소드 300 -> 900, `q3_impl.monotonic_min_spearman` 1.0 -> 0.9, Q1 타깃 변경). 해석은 하지 않는다.

### 결정 사항 -> 코드
1. **Q1**: 주 타깃 `log_nominal_area_ratio` 유지. 주 판정 창 = `q1.primary_window: late` (late_deformed 프레임, 학습 regime `all`). `early_flat` 는 `results.json h_a_early_flat_report` 에 판정/p 없이 R²(real, random-init, mask baseline), delta, `baseline_saturated` 로 별도 보고 (Holm 패밀리 4개 불변). mask baseline R² >= `q1.baseline_saturated_r2`(0.99) 이면 H_a 행 `not_evaluable`. 해석 선택: "late_deformed 에서의 일반화"를 "probe 는 전 프레임으로 학습, 평가는 late 프레임"으로 구현했다 (초기 프레임으로만 학습해 후반에서 평가하는 regime `early` 셀도 q1 에 그대로 남아 있고 탐색용).
2. **동일 예산**: (a) `single_ind` = ind 900 에피소드, (b) = 3 x 300, 보조 `multi_60_ind_160_small` = 3 x 100 (탐색, `conditions.optional`, `run_order` 마지막). `--all-conditions` 는 optional 을 빼고, `--include-optional` 또는 `--condition multi_60_ind_160_small` 로만 실행. 에피소드는 각 `*_train` metadata 순서의 앞 N개이므로 small 은 (b) 의 부분집합, (b) 의 ind 300 은 (a) 의 부분집합. 다운로드는 변경 없음 (같은 train split 3개, 각 2000 에피소드 중 앞 900). `manifest` 단계가 조건별 N 을 처리하고 부족하면 오류.
   - **예산 정의** (`budget.py`): 총 optimizer step S = epochs x steps_per_epoch, steps_per_epoch = ceil(학습 slice 수 / batch_size) (slice 수는 브랜치 로더 `LazyTrajSlicerDataset` 로 manifest 의 metadata 만으로 계산, 프레임 디코딩 없음). 브랜치 `train.py` 는 수정하지 않고 epochs 만 받으므로 환산한다: `train.epochs` = 기준 조건(`conditions.budget_reference` = (b))의 epoch 수, S_ref = `train.budget`(null 이면 epochs_ref x steps_per_epoch[ref]; 값이 있으면 총 step 직접 지정), 다른 조건 epochs = round(S_ref / steps_per_epoch[c]). (a) 와 (b) 는 에피소드 수가 같으므로 같은 epoch 가 된다는 것을 코드가 계산해 확인한다 (추정: 에피소드 길이가 같으면). `train.budget` 은 null 유지.
   - 기록: 각 run 폴더 `train_budget.json` (`$DINO_TRAIN/outputs/<run>/`)에 n_train_slices, steps_per_epoch(ceil/floor), epochs, `steps_computed`, `steps_target`, `rel_diff`, 학습 로그의 tqdm 총 step 후보. **steps_computed 는 계산값이지 optimizer 가 센 값이 아니다.** drop_last 여부와 `num_pred`(config `train_impl.num_pred: 1`, 추정)는 미확인이므로 서버에서 tqdm 총 step 과 대조해야 한다. aggregate 가 `results.json train_budgets` 에 모아 둔다.
3. **Q2 출력 공간 probe (주 측정)**: `q2.output_space`: Q1 과 같은 설정(regime `all`)으로 인코더 토큰으로만 학습한 probe 를 predictor 의 rollout 예측 z (t+1.. 시점)에 그대로 적용. 같은 에피소드/프레임의 인코더 토큰 읽기(`OUT|encoder|stepK`)와 학습 크기 role(`trained`) 셀이 기준선. predictor 활성값으로는 이 측정에서 probe 를 학습하지 않는다 (함수에 rollout 은 test 에피소드분만 들어가므로 구조적으로 학습 불가; selftest 로 확인). (a),(b) 는 같은 인코더 토큰/라벨/시드로 학습하므로 같은 probe 이며 `output_space.probe_digest_per_kind_seed` 를 q2_summary.json 에 남긴다 (attentive 의 GPU 비결정성으로 digest 가 달라질 수 있음: 미확인). 기존 "rollout 시점별 probe 재학습" 셀은 삭제했다. H_b 의 `last_rollout_step` 은 이 출력 공간 셀의 R² 하락(= OUT|encoder - OUT|rollout, 미지 크기)이다.
   - 층별 probe (활성으로 학습): (b) 가 주 결과. (a) 는 보조이며 `results.json q2_layerwise` 에 layer 0 대비 변화만 쓴다 (절대 R² 없음). **한계 문구** (q2_summary.json `layerwise_limitation`, results.json `q2_layerwise.limitation`): ind 에서 라벨이 상수이므로 3크기 활성으로 학습한 probe 가 (a) 에서는 본 적 없는 크기의 활성을 읽는다. `cond_a_vs_b:last_block` 은 layer 0 대비 변화의 차이로만 비교하고, (a) 의 4c 판정이 큼/not_evaluable 이면 `not_evaluable`(수치만).
4. **진단 4c** (`diag.py`, 규칙은 확정 기록 7 A, 값은 config `diag_4c`): 층 j 별 활성 이동량 = 평균 이동(ind 표준편차로 정규화한 채널 RMS) + 분산 이동(채널 평균 |var_l/var_ind - 1|) (두 값의 합은 구현 선택, 각각도 저장). noise floor = ind 테스트 에피소드를 겹치지 않는 두 반쪽으로 나눈 이동량의 `noise_floor_bootstrap`(200)회 중앙값 (시드 `diag_4c.bootstrap_seed` + 층 번호; a060/a160 미사용). 비율 = 이동량/noise floor: < 2 작음, 2 이상 5 미만 중간, 5 이상 큼. 다음 프레임 예측 오차(teacher forcing, 마지막 프레임 예측 시각 latent vs 다음 프레임 인코더 latent MSE)가 ind 의 2배 이상이면 큼. 최종: 어느 하나라도 큼 -> 큼, 아니면 활성 이동량 판정(층 최댓값 비율 기준, a060/a160 중 나쁜 쪽). 해석 규칙 문구(작음/중간/큼)는 결과 json 에 같이 저장. `decision_table` 의 `diag_4c:<cond>` 행. 크기별 활성 정규화는 구현하지 않았다.
5. **Q3**: α 곡선 = 에피소드 평균 (`q3.episode_mean`: 에피소드 안 평균 후 에피소드 평균), 단조성 = 그 곡선의 Spearman |rho| >= 0.9 (config, `aggregate._rank_test` 가 config 에서 다시 판정). 효과 크기 T 는 기존대로 샘플 중앙값(무작위 방향과 같은 함수). 주 검정 = steering(ridge) + INLP 만 `tier: confirmatory`; meandiff steering 과 LEACE 는 `tier: exploratory`, 판정/Holm 에 쓰이지 않으며 `results.json exploratory.q3_exploratory_methods` 에 값만 (p, 백분위 플래그 없음). selftest 가 탐색 블록을 지워도 H_c 판정이 같음을 확인.
6. **환경**: torch 2.0.1+cu117 (확정 기록 7 C). `environment.yml` 은 torch==2.0.1+cu117, torchvision==0.15.2+cu117, index cu117. `check` 가 설치 버전과 `env.torch_version` 일치 여부를 출력.

### torch 2.0.1 호환 점검 (코드 읽기 기반, 서버에서 실행해 보지 않음)
- 이 코드의 torch API: `torch.load(weights_only=True/False)`(1.13+), `nn.MultiheadAttention(batch_first, need_weights=False)`, `AdamW`, `torch.hub.load`, `cuda.*` — 2.0.1 에 있음. `torch.compile`, `scaled_dot_product_attention`, 2.1+ 전용 함수 사용 없음 (grep 확인).
- **수정함**: `trainrun.build_cmd` 가 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 를 설정했는데 이 옵션은 2.1 에서 추가되어 2.0.1 의 allocator 가 미인식 옵션으로 오류를 낼 수 있어 제거.
- **못 확인**: 브랜치 `train.py`/모델 코드와 DINOv2 hub 코드(commit 85a2460)가 2.0.1 에서 도는지 (xformers 의존, SDPA 사용 여부), numpy 1.26.4 와 torch 2.0.1 조합, `torchvision==0.15.2` 짝(기억 기반), decord/cv2 와의 조합.

### 서버 테스트 순서 (추가분; 기존 5절 순서에 이어)
1. `python run.py selftest` (서버 환경에서도), `python run.py check` (torch 버전 줄: match=True, cuda 11.7 확인).
2. `download`, `manifest` (조건별 manifest 3개: 900 / 900 / 300 에피소드 확인), `smoke` 후 사람이 `train.batch_size/num_hist/epochs` 확정.
3. `python run.py train --dry-run --all-conditions --include-optional` : 조건별 steps_per_epoch/epochs/steps_computed 출력 확인 (이때 manifest 필요). 실제 학습 후 `train_budget.json` 의 steps_per_epoch 와 로그 tqdm 총 step 대조 (num_pred, drop_last 가정 확인).
4. `cache`, `q1`, `q2`, `q3`, `aggregate` 후 `results.json` 에서 `h_a_early_flat_report`, `q2_layerwise`, `diag_4c:*` 행, `train_budgets` 확인.
5. 보조 조건: `python run.py train --condition multi_60_ind_160_small` 등 (`--include-optional`). 모든 학습은 별도 승인 사항.

### 정적 점검과 slice 수 (2026-11-03 추가)
- `data.count_train_slices` 는 `load_manifest_slice_train_val(transform=None)` 의 train/valid 길이(0.9/0.1 episode 분할, seed 42)를 쓴다 (영상 디코딩 없음). 이전 판은 함수가 없어 서버 `train --dry-run` 이 AttributeError 로 실패했다. 이 머신의 dry-run 이 놓친 이유: manifest 파일이 없어 그 호출 전에 FileNotFoundError 로 빠졌고, selftest 는 이 함수를 부르지 않았다.
- 서버 smoke 관측(num_hist=1, 48 에피소드, batch 24: train 99 / valid 12 iteration)과의 일관성은 selftest 에서 산술로만 확인한다: 가정 "에피소드당 slice = n_obs - (num_hist+num_pred)*frameskip" (관측에서 역산한 가정, 브랜치 슬라이서 코드는 읽지 않음), 65 프레임, 분할 43/5 -> 2365/275 slice -> ceil 로 99/12. 실제 값은 서버에서 `train --dry-run` 이 출력하는 n_train_slices 와 대조.
- `static_check.py`(selftest 항목 H): 모듈 별칭 `X.attr`, `from .mod import name` 이 대상 모듈에 정의됐는지 AST 로 점검. 한계: 변수 이름과 별칭이 같으면(M, B 등) 건너뜀, 실행 없는 휴리스틱.

## 10. Colab 실행 (확정 기록 10, 2026-11-04) — 서버 대신 Google Colab Pro+
- 상태: 코드/노트북만 작성. **Colab 에서 한 번도 실행하지 않았다**(이 PC 에 GPU/Drive 없음). 검증한 것: `python run.py selftest` 136 통과 0 실패 0 건너뜀(Colab 항목 포함: 환경변수 경로, Drive 복사/건너뜀/크기 불일치 복구/누락 감지, 대괄호·공백 경로, 재개 패치 적용과 epoch 범위, `prepare` 분기, 배선 점검), static_check 통과. 셀프테스트 수치는 결과가 아니다.
- 파일: `colab/train_resume.patch`, `colab/exp002_colab.ipynb`(셀 순서: GPU -> Drive 마운트 -> repo clone/pull -> pip 환경 -> ACWM checkout + resume 패치 -> 환경변수 -> 데이터 복사 -> check/manifest/(smoke) -> 전체 실행), `colab/colab_tools.py`, `colab/requirements_colab.txt`. 코드 변경: `src/exp002/drive.py`(`copy_from_mount`, `find_mount_src`), `run.py download`(환경변수 `EXP002_DATA_MODE=mount` 또는 `EXP002_DRIVE_MOUNT_SRC` 이면 rclone 없이 마운트된 Drive 에서 복사), `src/exp002/trainrun.py`(아래 가드), `overnight.sh`(학습 시도 전 `prepare` 호출).
- **Drive 영속화 방식 (택한 것: Drive 에 직접 기록, 동기화 스크립트 없음).** `WORK_ROOT=MyDrive/exp002_work` 아래 `train_runs/`(= `DINO_TRAIN`), `exp002_cache/`(= `EXP002_CACHE`), `results/`(attempts.json 포함), `manifests/`, `logs/` 를 두고, 저장소의 `results|manifests|logs` 를 이 폴더로 가는 심볼릭 링크로 만든다(`colab_tools.py setup-dirs`). 이유: 주기적 rsync 는 동기화 사이에 세션이 끊기면 DONE/체크포인트/attempts.json 이 서로 어긋날 수 있다. 직접 기록하면 각 파일이 쓰인 시점에 영속이고 `resolve_paths` 코드 변경이 없다. 단점: Drive FUSE 는 느리다(측정 못함). 데이터는 읽기 속도 때문에 `/content/dinowm/data/cloth_sim` 로컬 복사본을 쓴다(세션마다 재복사; 검증: 완료 마커 + mp4 수 == metadata == 기대값 + 기록된 총 바이트, 크기가 다른 파일은 다시 복사).
- **학습 재개(resume): 사용한다 (사용자 결정, train.py 패치 승인 포함).** 경위: 2026-11-04 에 사용자가 먼저 "resume 패치를 쓰지 말자"고 지시했으나 이는 사용자의 실수였고, 같은 날 "체크포인트에서 재개할 수 있게 하라"로 정정했다. 이 문서는 정정된 결정을 따른다.
  - 고정 commit 5a640bed 의 `train.py` 소스 확인: `init_models()` 는 `<run 폴더>/checkpoints/model_latest.pth` 가 있으면 읽어 epoch 를 복원하지만 action/proprio encoder 는 새로 초기화되고 optimizer 는 새로 만들어져 상태가 복원되지 않으며(`action_encoder_optimizer` 는 저장도 안 됨), `run()` 이 `range(init_epoch, init_epoch + training.epochs)` 라서 **남은 epoch 가 아니라 epochs 만큼 더** 학습한다. 가중치만 복원하거나 이 상태로 재개하면 총 optimizer step(27855, 계산값) 동일 예산이 깨진다. 학습률 scheduler 는 train.py 에 없다.
  - `colab/train_resume.patch` (pinned train.py 위에 `git apply`): 상위 저장소 main 의 11ba488(`train: resume keeps the action/proprio encoders and the optimizer states`, train.py 49줄)의 diff + 한 줄(`range(init_epoch, self.total_epochs + 1)`, 즉 `training.epochs` 를 **총 목표 epoch** 로 해석, 마커 주석 `[exp002 operational]`). 새 학습(epoch 0)은 원본과 같이 1..15 를 돈다. 노트북 5번 셀이 적용 여부를 `git apply --reverse --check` 로 확인해 없을 때만 적용하고, 적용 후 working tree 가 `train.py` 한 파일만 다른지 assert 하며 패치 sha256 을 `WORK_ROOT/acwm_resume_patch.sha256` 에 기록한다. commit 은 pinned 그대로다.
  - 재개 전 점검 `colab_tools.py prepare`(overnight.sh 가 **매 train 시도 전**, 재시도 포함, 호출하고 실패하면 그 시도를 rc 97 로 처리; 노트북 마지막 셀은 overnight.sh 미실행 시 먼저 호출): DONE 이 없는 run 에 대해 (i) 체크포인트 없음 -> 처음부터, (ii) `model_latest.pth` 읽힘 -> 그대로 재개, (iii) `model_latest.pth` 깨짐 -> 가장 최신의 읽히는 `model_<e>.pth` 로 복구 후 재개, (iv) 읽히는 체크포인트가 하나도 없음 -> `checkpoints/` 와 `epoch_logs.jsonl` 을 `<DINO_TRAIN>/quarantine/<run>__<YYYYmmdd-HHMMSS>/` 로 **이동**(삭제 없음)하고 처음부터. DONE run 과 smoke 이름 run 은 건드리지 않는다. `prune` 은 DONE run 만 보며 격리 폴더는 `outputs/` 밖이라 충돌하지 않는다.
  - 안전장치: `trainrun.train` 은 DONE 이 없는데 `checkpoints/model_latest.pth` 가 있으면, `acwm_src/train.py` 에 패치 마커가 있을 때만 재개를 허용하고(로그에 기록) **패치가 없는 train.py 이면 학습을 시작하지 않고 rc=1 로 거부**한다(epochs 를 더 학습해 예산이 깨지는 것을 막음).
  - 검증한 것(CPU, selftest): 패치가 pinned train.py(`git show 5a640bed:train.py`)에 깨끗이 적용됨, 패치된 소스의 `run()` 루프식을 AST 로 꺼내 평가하면 새 학습 1..15, epoch 5 저장 후 재개 6..15, epoch 15 에서는 0 epoch, (5 + 재개분) x steps_per_epoch == 새 학습 총계(steps_per_epoch 는 예시값), 패치 전 루프는 `init_epoch + self.total_epochs`. `prepare` 의 (i)~(iv) 분기, 마커 검출. **검증하지 못한 것(추정 포함):** 실제 cloth 환경, torch 2.0.1+cu117, Colab, GPU 에서의 재개. 11ba488 커밋 메시지는 Wall 환경에서 재개가 중단 없는 run 과 손실/체크포인트까지 bit 단위로 같았다고 주장하나 우리는 재현하지 못했다. 재개 후 DataLoader 순서와 난수 상태는 복원되지 않으므로(시드는 프로세스 시작 시 다시 설정) 중단 없는 run 과 bit 단위로 같지 않을 수 있다(추정). 총 optimizer step 수는 같다. 중단 때 진행 중이던 epoch 는 버리고 마지막으로 저장된 epoch 뒤부터 다시 하며, `epoch_logs.jsonl` 에는 그 epoch 의 줄이 중복될 수 있다(추정).
- `save_every_x_epoch`: 5 -> **1**(`config.yaml train_impl`, 사용자 승인). 운영 값이며 사전 등록과 무관하다(학습 step/데이터/시드 불변). 재개가 마지막 저장 epoch 부터 이어지므로 세션이 끊겨도 최대 1 epoch 만 잃는다(원래 5 이면 최대 4 epoch). Drive 용량: `config.yaml colab.keep_ckpt_epochs: [5,10,15]` 만 남기고 DONE run 의 나머지 `model_<e>.pth` 를 삭제(`colab_tools.py prune`). 체크포인트 크기는 측정하지 못했다(추정: 수백 MB).
- 환경 (CLAUDE.md "Conda 환경" 규칙의 **예외**): Colab 에는 conda 가 기본으로 없고 런타임이 세션마다 초기화되므로 노트북이 `uv` 로 Python 3.9 venv(`/content/venv`)를 만들고 `colab/requirements_colab.txt`(environment.yml 의 pip 핀: torch 2.0.1+cu117, torchvision 0.15.2+cu117, numpy 1.26.4, hydra-core 1.2.0 등)를 설치한다. 추가: psutil 5.9.8, pillow 10.3.0(고정 commit 의 utils.py 가 import; environment.yml 에는 없음). 이 설치는 해 보지 않았다(미검증, 특히 opencv 4.6 + numpy 1.26, decord 0.6.0 wheel). `USE_SYSTEM_PYTHON=True`(Colab 기본 Python/torch)를 쓰면 torch 버전이 config `env.torch_version` 과 달라 `run.py check` 가 match=False 를 출력하고 CUDA 커널/수치, `torch.load` 기본값(>=2.6), hydra 1.2.0/Python 3.12 호환에서 차이가 생길 수 있다(추정). 서버 결과와 직접 비교 가능하다고 주장하지 않는다. 브랜치 environment.yaml 의 torch 는 2.3.0 이고 config 의 2.0.1 은 사용자 결정(확정 기록 7 C)이라 그대로 따랐다.
- 불변: seeds [0,1,2], batch_size 24, num_hist 1, epochs 15, 동일 예산(총 optimizer step 27855, 계산값), 모든 사전 등록 임계값. VRAM 이 남아도 batch_size 는 바꾸지 않았다. 제안(실행 안 함): 바꾸면 step 환산과 비교 가능성이 달라지므로 002 가 끝난 뒤 별도 실험에서만 고려.
- 이어서 실행: 노트북 마지막 셀을 다시 실행하면 DONE 인 단계는 건너뛰고 미완 run 은 `prepare` 규칙대로 마지막 체크포인트에서 재개한다(체크포인트가 없거나 못 쓰면 처음부터). 세션 중단은 실패로 세지 않는다(attempts.json 은 프로세스가 비정상 종료 코드를 낼 때만 증가). 한계: `overnight.sh` 가 시도별 로그(`overnight_<item>_<stage>_try1.log`)를 덮어쓰므로 이전 세션 로그는 `logs/train_*_<time>.log` 와 run 폴더에서 본다.
- repo/데이터 접근: 노트북 3번 셀의 `REPO_URL` 은 `git remote get-url origin`(https://github.com/jongmin-s-multicore/probing-experiment.git)과 같다. 5번 셀의 ACWM URL(https://github.com/jongmin-s-multicore/thread0.git)은 config `code.repo_url` 과 같고 thread0 의 `upstream` remote 와도 같다. 2026-11-04 에 인증 없이 `git ls-remote --heads` 를 두 URL 에 실행했고 둘 다 브랜치 목록이 나왔다(`hanbin5/acwm-phys` 끝 = 5a640bed). 이 한 번의 테스트로 **공개로 보이며 토큰 불필요**하나, 저장소 설정은 확인하지 못했으므로 공개 여부는 미확인(추정)이다. 비공개로 바뀌면 Colab Secrets `GITHUB_TOKEN`. 원격 `exp002-single-size-sweep` 끝은 10e2833(이번 변경 없음)이었으므로 노트북 실행 전에 커밋/푸시가 필요하다.
- Drive 데이터: 사용자가 계정 davidjk522@gmail.com 의 MyDrive 에 바로가기 `dino-wm-videos`(id 17M0MZGSDPzu7CCT3ez3ABMyZGjavTfx3, 소유는 다른 계정)를 추가했다(사용자가 Drive 메타데이터로 확인해 전달한 내용). config 의 `root_folder_id` 1z4z6WA2G-7UXjbrJtLSWqm6KBpuHrkiI 는 그 안 `[data]/[cloth-sim] cloth_move` 폴더다. 예상 마운트 경로: `/content/drive/MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move`(바로가기 이름이 바뀌었으면 다름, 마운트 형태는 **미검증**). 노트북 2번 셀 `DATA_SRC` 기본값이 이 경로이고, 없으면 코드가 `EXP002_DRIVE_MOUNT_SRC` -> 기본 경로 -> `MyDrive/dino-wm-videos` 아래 깊이 3 까지 탐색(모든 split 디렉터리 `ind_train` 등을 가진 폴더) -> `.shortcut-targets-by-id/<root_folder_id>` 순으로 찾는다. 경로에 대괄호와 공백이 있어 코드는 pathlib `iterdir`/`is_dir` 만 쓰고(glob 미사용) 노트북은 경로를 환경변수로만 넘긴다(셸 명령행에 넣지 않음); selftest 가 대괄호/공백 경로의 탐색과 복사를 확인한다. 공유 폴더 다운로드 할당량 초과 가능성은 미확인.
- 서버에서 중단된 run 의 부분 학습 출력은 재사용하지 않는다. Colab 에서 처음부터 다시 시작한다(사용자가 달리 요청하기 전까지). `WORK_ROOT` 는 새 폴더여야 한다.
