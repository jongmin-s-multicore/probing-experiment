# 실험 설명: 006_size_readout_reanalysis

- 상태: **코드와 Colab 노트북 작성 완료, 합성 데이터 CPU selftest 통과(8절). GPU 실행, Colab 접근, git commit/push 는 하지 않았다. 이 문서에 실험 결과 수치는 없다.** (8절의 selftest 숫자는 합성 데이터의 코드 점검이며 결과가 아니다.)
- 검증할 가설: 없음(탐색). notes/hypotheses.md 의 "후보 J / 실험 006" 을 그대로 따른다. 002 의 판정과 임계값은 바꾸지 않는다. 사용자가 2026-11-05 재분석 진행을 지시했다. 이 분석은 002 결과와 리뷰를 본 뒤 세운 사후 탐색이다(확증 근거로 쓰지 않는다).
- 질문(hypotheses.md 후보 J 목적): (1) rollout R² 하락이 크기 정보 소실인가 판독기 전이 실패인가, (2) 예측이 특정 크기로 몰리는가, (3) 하락이 크기 특이적인가.
- 방법: 002 의 6개 체크포인트(조건 (a) `single_ind`, (b) `multi_60_ind_160` x 학습 시드 0,1,2)와 002 캐시/결과를 읽기 전용으로 쓰고, 분석 A-E 를 계산한다. 새 학습 없음. 주 타깃 `log_nominal_area_ratio`, 002 와 같은 test episode(9 레벨 x 100), 002 와 같은 Q1 linear probe 경로(`probes.linear_fit_predict`, `evalgrid.fit_apply_cells`).
- 성공 기준: **없다.** 임계값, pass/fail, p 값 기반 유의성 주장을 A-E 어디에도 만들지 않는다. results.json 의 `interpretation` 은 null. 코드가 하는 일의 정확성만 selftest 로 점검한다(알려진 답을 복구하는지).
- **이 실험으로 내가 주장하려는 것**(확정 전 초안, 해석은 분석가 리뷰 몫): 002 의 output-space rollout 측정을 같은 코드 경로로 재현했을 때의 차이(재현 차이)와, 그 위에서 (A) 레벨별 전이 probe 예측값 평균/기울기/절편/MAE 가 step 1-12 에서 어떤 값이었는지, (B) persistence 와 학습 안 한 predictor 의 같은 R² 가 얼마였는지, (C) 예측 latent 로 재학습한 probe 의 R² 가 전이 probe 의 R² 와 어떻게 달랐는지, (D) 크기와 무관한 속성의 같은 R² vs step 이 어땠는지, (E) 시드 단위 평균/SD/t 기반 95% CI(n=3, df=2)가 얼마였는지, 를 **수치로 보고**하는 것. 원인(정보 소실, 판독기 전이 실패, 압축 등)에 대한 주장은 하지 않는다.

## 1. 구조

```
experiments/006_size_readout_reanalysis/
  config.yaml      모든 시드(전역, probe, untrained predictor, bootstrap, retrain probe), 입력 경로 환경변수, rollout horizon 12,
                   002 config/results.json/requirements sha256, 입력 파일 기대 크기/sha 자리(inputs.expect)
  run.py           단계 CLI: check selftest data_meta extract pool labels transfer repro analyze_a..analyze_e aggregate figures (+ cpu_all)
  src/exp006/      base(경로, 읽기 전용 guard, 검증되는 쓰기, 재시도 3회) lib(numpy 통계) ctx(입력 로드) extract(GPU) stages_pre stages_an stages_agg figs selftest
  colab/           build_notebook.py -> exp006_colab.ipynb (코드 스냅샷 내장)
  analysis/        make_figures.py, fig*.png (aggregate 후 생성)
  logs/            단계별 로그
  results.json     aggregate 가 생성 (실험 폴더와 WORK_ROOT 둘 다)
```

002 모듈(`probes, evalgrid, stats, masks, model_io, directions, common, q2.output_space`)은 **수정 없이 import** 한다(004 와 같은 방식, `sys.dont_write_bytecode` 로 002 폴더에 `__pycache__` 도 만들지 않는다). 복사한 코드는 없다. 001/002/003 파일은 읽기만 하며, 선택한 selftest 가 파일 크기와 mtime 이 변하지 않았음을 확인한다.

## 2. 002 저장물에서 얻는 값과 추론이 필요한 값 (코드로 확인)

| 002 파일 | 코드에서 실제로 저장되는 것 (출처) | 006 에서 쓰는 곳 |
|---|---|---|
| `results/q1/cells.npz` | Q1 셀 키별 `(probe seed 5, test episode 900, 7)` 에피소드 충분통계 + `predmean|<키>` (`q1.py`). 단일 프레임 천장 셀 `...|dino|f<k>|frame<k>` 포함 | C 의 인코더 재학습 값 교차 확인 |
| `results/q1/test_rows.json` | test 행(레벨, 역할, 명목 log 면적), 캐시 순서 | 레벨 라벨 정렬 확인 |
| `results/<run>/q2_cells.npz` | output-space 셀 `<prim>|linear_pooled|OUT|{encoder,rollout}|step1..12` 의 `(5,900,7)` 충분통계와 **`predmean|<같은 키>` = 에피소드별 probe 예측값 `(5,900)`** (`q2.py` output_space, `evalgrid.fit_apply_cells`) | **A, E 는 이 저장값만으로 계산 가능**(GPU 불필요). 재현 점검의 비교 기준 |
| `results/<run>/q2_summary.json` | meta, `output_space.read_by_level`(레벨별 읽은 크기 평균), 순서 정확도 등 요약 | 교차 확인(참고) |
| `results/<run>/q3.json` | Q3 개입 결과 | **006 은 쓰지 않는다**(목록에만 기록) |
| 002 `results.json` | `exploratory.q2_layer_time_grid_r2_point_estimates[<cond>/seed<s>][키][unseen|trained]`, 판정 표 `H_b:last_rollout_step`, `cond_a_vs_b:last_rollout_step`(시드별 ceiling/q2/drop) | E, 재현 점검 |
| `exp002_cache/q1_tokens/` | `frames_u8`(E,13,224,224,3), `tokens_dino`(E,13,196,384) fp16, `mask_feat`, `actions_raw`(E,60,6), `episodes.json`, `meta.json` | 모든 단계의 입력. (`tokens_rand` 는 006 에서 불필요) |
| `train_runs/outputs/<run>/` | `checkpoints/model_latest.pth`, `hydra.yaml`, `train_manifest.json`, `DONE` | extract |
| `cloth_sim_data.tar` | 데이터 아카이브(`cloth_sim/<split>/...`) | data_meta 가 `metadata.pt` 만 스트리밍으로 읽는다(행동 정규화 통계용) |

**저장되지 않아 추론을 다시 돌려야 얻는 값**(`extract` 단계, GPU): 예측 latent 자체. 002 의 `q2_tmp_<run>/roll.npy` 는 학습 행과 test 행 모두를 rollout 했지만 `keep_tmp: false` 로 삭제됐다. 따라서 **B 의 untrained predictor, C 의 재학습 probe(학습 episode 의 rollout), D 의 rollout 측정, 재현 점검의 rollout 부분**이 추론을 필요로 한다. **A 와 E, B 의 persistence/천장, 재현 점검의 인코더 부분은 추론이 필요 없다**(A 는 저장된 predmean, B 의 persistence 는 캐시 토큰 + 같은 probe 재학습).

저장 형식: 추출은 패치 평균 풀링된 latent `(E,12,384)` float32 만 저장한다(`linear_pooled` 가 패치 평균 후 ridge 이므로 같은 값이다; 패치 단위 latent 는 run 당 약 3.3 GB 라 저장하지 않는다). 풀링은 002 와 같은 순서(rollout 을 fp16 으로 캐스트한 뒤 `probes.pool` 의 float32 평균)로 한다. 한계: attentive probe 는 이 형식으로 쓸 수 없다(쓰지 않는다).

## 3. 단계와 정의

| 단계 | 하는 일 | GPU |
|---|---|---|
| `check` | 환경, 경로, 입력 파일 존재/크기/sha256(작은 파일, `--hash` 면 전부), 002 config sha 대조, 고정 commit 체크아웃/`train.py` 상태, 모델 로드 호환 사실. 읽기만, `input_manifest.json` 기록 | 불필요 |
| `selftest` | 합성 데이터 CPU 점검(8절) | 불필요 |
| `data_meta` | 002 아카이브(읽기 전용)에서 학습 split 의 `metadata.pt` 만 로컬로 꺼냄 + 필드 목록 기록 | 불필요 |
| `extract` | 체크포인트별 rollout(학습된 predictor, 학습 안 한 predictor 6종)을 002 의 캐시 프레임으로 재계산. 단계별 `DONE` 마커, 재개 가능. `--smoke` 는 메모리/시간만 측정 | 필요 |
| `pool`, `labels` | 캐시 토큰 패치 평균, D 의 속성 라벨 | 불필요 |
| `transfer` | 002 와 같은 함수 `evalgrid.fit_apply_cells` 로 Q1 probe 를 인코더 토큰에서만 한 번 학습하고 인코더/persistence/rollout/untrained 집합 528개에 적용 | 불필요 |
| `repro` | 002 저장값과의 차이 보고(허용오차 없음) | 불필요 |
| `analyze_a` .. `analyze_e` | 분석 A-E | 불필요 |
| `aggregate`, `figures` | results.json(실험 폴더와 `WORK_ROOT/results/results.json`), PNG | 불필요 |

모든 단계는 `--dry-run`(계획만, 아무것도 쓰지 않음)을 지원한다. 모든 단계는 최대 3회(config `run.max_attempts`) 시도하고 3회 실패하면 멈추며(`run.py` 종료코드 3), 안전 중단(읽기 전용 폴더 쓰기 시도)은 재시도하지 않는다(종료코드 4).

### 공통 정의
- 라벨: 주 타깃 `log_nominal_area_ratio`(레벨 상수). rollout step k 의 라벨 = 프레임 `t+k` 의 라벨, `t = num_hist-1 = 0`. `num_hist = 1` 이므로 마지막 context 프레임 = 첫 프레임이다.
- Q1 probe: 학습 행 = 캐시의 `part=train` 900행(a060, ind, a160 각 300), 프레임 0..11 전부(`regime all`), 같은 프레임의 라벨, 패치 평균 -> 표준화 ridge, alpha 는 학습 episode 안 GroupKFold(분할 시드 = probe 시드 0..4). 선형 ridge 이므로 시드 5개의 예측은 사실상 같다(분할이 alpha 선택에만 영향). 모든 R² 는 시드 평균.
- R² = 1 - SSE/SST (역할로 묶은 episode 전체, SST 는 역할 내 라벨 평균 기준; 002 `stats.metric_from_sums`). 역할: `unseen`(보간+외삽 6레벨), `trained`(3레벨), `all`.
- 에피소드 bootstrap 95% CI(B, C): 002 `stats.Bootstrap`(레벨 안 재표집, 1000회, 시드 `seeds.bootstrap`). **평가 episode 만 반영**하고 학습 시드/untrained 시드 변동은 반영하지 않는다.

### A. 레벨별 예측값 평균, 기울기, 절편, MAE
- 입력: 두 소스를 따로 보고한다. `stored` = 002 `q2_cells.npz` 의 `predmean`(추론 불필요), `reproduced` = 이 실험의 추출 + 같은 probe 재학습(추출 후).
- 대상: 두 조건 x 시드 3 x {인코더 토큰(천장), rollout} x step 1-12.
- 값: 레벨별(9) 예측 평균, 참값(log 명목 면적비), 편향 `mean_pred - true`, 레벨별 MAE, 예측 SD; 역할별(`all, unseen, trained`) episode 단위 OLS `pred ~ label` 의 **기울기와 절편**, 레벨 평균에 대한 OLS 기울기/절편, MAE, 평균 편향. 에피소드별 예측 `A_per_episode_<source>.npz` 저장(재계산 가능; 레벨 인덱스, 명목 라벨 포함).
- 시드 요약(`seed_summary`) 과 (a)-(b) 차이는 7절 계산식.

### B. 기준선의 같은 output-space R²
- `persistence`: 마지막 context 프레임(= 프레임 0) 인코더 토큰을 모든 step 에 복사해 Q1 probe 적용. `num_hist = 1` 이라 "첫 프레임 persistence" 와 같은 집합이다(`first_frame_equals_last_context_frame: true`). 천장(프레임 `t+k` 인코더 토큰)과의 차이 `persistence_minus_ceiling` 을 step 별 R²/MAE 로 저장.
- `untrained predictor`(같은 구조, 학습 안 한 가중치): 아래 5절.
- 학습된 rollout 의 값(같은 probe 로 재계산한 값)을 나란히 둔다.

### C. 재학습 probe
- 예측 latent(rollout step k)로 학습 행(900)에서 ridge probe 를 학습하고 test 행의 같은 step 예측 latent 로 평가한다. step 별, `unseen`/`trained` 별, 전이 probe R² 와 나란히 저장. 비교용으로 "인코더 토큰 프레임 k 에서 재학습"(단일 프레임 Q1 천장)도 계산하고 002 의 `q1 cells` `f<k>` 와 대조한다.
- 학습 샘플은 step 당 행 1개씩(900개). alpha 와 GroupKFold 분할은 `exp002 directions.ridge_fit` 이 **학습 행 안에서만** 고른다. test 행은 마지막 예측 호출에만 들어간다(selftest: ridge 호출이 항상 학습 행 수만 받는지, test 를 무작위/NaN 으로 바꿔도 alpha 와 적합이 같은지).
- 판독기 학습 크기는 Q1 과 같은 a060, ind, a160. **한계**: 조건 (a) 의 predictor 는 ind 만 학습했으므로, (a) 에서 a060/a160 학습 행의 rollout 은 predictor 가 보지 못한 크기의 rollout 이다. (b) 는 세 크기를 학습했다.

### D. 크기와 무관한 대조 속성 (실제 데이터에서 얻을 수 있는 것만)
코드와 데이터 필드 확인 결과:
- **없음**: proprio(그리퍼 위치) 필드. `ManifestDataset.get_frames` 가 `proprio` 에 0 을 채워 반환한다(`data.py`, `proprio_dim = 1` 자리표시자). `metadata.pt` 의 필드는 `video_path, actions(64,8), length, seed, cloth_dimx, cloth_dimy, x0, n_attempts`(001 config/README 기준; `data_meta` 가 실제 필드를 `metadata_fields.json` 에 기록). `x0` 의 의미는 코드에서 확인하지 못했다(추정: 초기 위치일 수 있으나 미확인이라 쓰지 않는다). 그리퍼 공(빨강/파랑) 색 마스크 규칙은 코드에 없다 -> 쓰지 않는다(`gripper_pixel_position: false`).
- **쓰는 것(10개)**: (1) 천 마스크 중심 x, y (002 HSV 규칙으로 캐시 프레임에서; /224), (2) 마스크 bbox 중심 x, y, (3) 행동 누적합 6개 = 프레임 f 까지 `actions_raw[:, :5f]` 의 합(그리퍼 1 의 dx dy dz, 그리퍼 2 의 dx dy dz). (3) 은 **명령된 이동량의 누적이며 측정된 그리퍼 위치가 아니다**(초기 위치를 모른다). 001 README 는 "action x 15 = 월드 좌표 이동량"이라고 쓰지만 이는 미검증이다. 마스크가 빈 프레임은 제외한다.
- 방법: 속성마다 같은 방식. 인코더 토큰(프레임 0..11, 같은 프레임의 라벨)으로 ridge probe 를 학습해 인코더(프레임 k), persistence, rollout step k 에 적용, R² vs step (역할별 + 레벨 평균을 뺀 within-level R²). 시드 요약 포함.

### E. 시드 단위 보고
- 002 `q2_cells.npz`(Drive)로 step 1-12 의 R²(인코더, rollout), 하락 `drop = R²(인코더 step k) - R²(rollout step k)`, MAE, 순서 정확도(`stats.pair_order_accuracy`, 점추정)를 시드별로 계산하고, 002 `results.json` 값(그리드, 판정 표 H_b/cond_a_vs_b 의 시드별 ceiling/q2/drop)과의 차이도 `consistency_with_results002_json` 에 보고한다.
- step 12 가 중심이지만 모든 step 이 저장된다.

## 4. 모델 로드 호환성 (고정 commit 5a640bed 와 002 패치)
- `model_io.load_world_model` 은 `torch.load(model_latest.pth, weights_only=False)` 로 `predictor, decoder, proprio_encoder, action_encoder`(그리고 있으면 `encoder`)를 읽고, 없으면 `hydra.yaml` 의 `cfg.encoder` 를 `hydra.utils.instantiate` 한다(DINOv2 torch.hub 다운로드, 인터넷 필요). `train.py` 를 import 하지 않는다(`run.py check` 와 selftest 가 소스에서 확인).
- 따라서 **고정 commit 의 모델 클래스(`models.*`, `datasets.*`; unpickle 과 hydra target)는 필요하고, 002 패치(`colab/train_resume.patch`, sha256 52eb0052...)는 로드에 필요하지 않다**(패치는 `train.py` 의 optimizer state_dict 저장과 재개 로직만 바꾼다. 패치 후 체크포인트의 optimizer 항목은 `model_io` 가 읽지 않는다). 노트북은 패치를 적용하지 않고 고정 commit 만 체크아웃한다. 이것은 소스 읽기로 얻은 결론이며 실제 체크포인트 로드는 Colab 에서 확인해야 한다(미검증).
- `ViTPredictor` 의 `Attention.__init__` 이 `self.bias = ....to('cuda')` 로 CUDA 를 하드코딩한다(소스 확인). 체크포인트 로드는 `map_location=device` 라 CPU 로도 읽히지만 predictor 의 마스크 텐서는 장치를 따라간다. 실제 추출은 GPU 에서 한다.

## 5. untrained predictor 와 입력/행동 인코더 처리 (train.py `init_models` 와 `VWorldModel` 코드를 읽음)
- 구조: `hydra.utils.instantiate(tcfg.predictor, num_patches, num_frames=tcfg.num_hist, dim)` 를 train.py `init_models` 와 같은 인자로 호출한다(`dim = 인코더 emb 384 + (proprio_emb*repeat + action_emb*repeat)*concat_dim = 404`, `num_patches = (img_size//16)^2 = 196`). 새 predictor 와 학습된 predictor 의 `state_dict` 키/모양이 같은지 확인하고 다르면 중단한다. 가중치는 `torch.manual_seed(seed)` 직후의 생성자 초기화(여기서 `ViTPredictor.pos_embedding = torch.randn`, Linear/LayerNorm 기본 초기화)이며 학습은 없다. 시드는 `seeds.untrained_predictor` = [0,1,2].
- **입력 인코더(DINOv2)**: 항상 원본 동결 인코더를 쓴다(체크포인트 인코더와 002 캐시의 지문 일치를 확인). 시각 부분 latent(앞 384)는 학습된 모델과 동일한 텐서다.
- **행동 인코더/proprio 인코더**: 두 변형을 따로 계산한다.
  - `pred_only`: 행동/proprio 인코더 = 해당 run 체크포인트의 **학습된** 것. 첫 입력 프레임의 행동 임베딩과 매 step 의 `replace_actions_from_z` 에 학습된 행동 인코더 출력이 들어간다. 바뀌는 것은 predictor 가중치뿐.
  - `pred_and_action_proprio`: 행동 인코더, proprio 인코더, predictor 를 모두 같은 시드로 새로 초기화(train.py 순서: proprio, action, predictor). 입력 z 의 proprio/행동 부분을 `rebuild_aux` 로 다시 만든다(시각 부분은 그대로; 학습된 인코더로 재구성하면 `encode()` 와 일치하는지 첫 배치에서 검사하고 어긋나면 중단).
  - 행동 입력은 항상 해당 run 의 정규화 통계(`train_manifest.json` 의 학습 행동 평균/표준편차)로 정규화한 것.
- 학습 안 한 predictor 의 결과는 초기화에 의존한다. 변형 2 x 시드 3 개를 따로 보고하고, 시드 요약은 untrained 시드 평균 후 학습 시드(run) 간 요약이다.

## 6. 예상 시간과 자원 (모두 **추정**, 측정 없음. smoke 가 첫 단계로 측정한다)
- 근거(산술): predictor 약 20M 파라미터(깊이 6, dim 404, mlp 2048, heads 16 x 64) 이므로 한 번의 rollout 한 배치(episode 8 x 12 step x 토큰 196) 약 0.75 TFLOP. run 당 학습된 rollout 1800 episode(225 배치) + untrained 6종 x 900 episode(113 배치) + DINOv2 ViT-S/14 인코딩 약 23,400 프레임으로 약 950 TFLOP, fp32 유효 처리량을 10-30 TFLOPS 로 가정하면 run 당 약 0.5-2 분의 연산 + 로드/데이터 읽기 시간. Drive 읽기(frames_u8 3.5 GB 를 episode 순서로 읽음)와 체크포인트 로드가 지배할 수 있다.
- 따라서 6 run 합계는 **약 0.3-1.5 GPU 시간(추정, 범위가 넓음)**, GPU 메모리는 배치 8 에서 **약 3-6 GB 이하(추정)**, CPU RAM 은 풀링 결과와 memmap 읽기만이라 수 GB 이하. 출력 용량 약 0.8 GB(`extract/`), `cache/` 약 0.1 GB.
- CPU 단계(전부 Colab CPU 로도 가능): pool/labels 는 Drive 에서 3.5 GB 씩 읽어 수 분(추정), transfer/analyze 는 합쳐 수 분-십수 분(추정; ridge 적합 수백 회 + bootstrap).
- 사람이 Colab 런타임 유형을 선택한다. GPU 는 9-10번 셀(smoke, extract)에만 필요하다.

## 7. 계산식 (시드 단위 보고, `lib.py`)
- 시드 요약(`seed_agg`): 값 `x_1..x_n`(n = 학습 시드 수 = **3**), 평균 `m = mean(x)`, 표본 SD `s = sqrt(sum (x_i - m)^2 / (n-1))`, 표준오차 `s/sqrt(n)`, 95% CI = `m ± t_{0.975, n-1} * s / sqrt(n)`. n=3 이면 자유도 2, `t = 0.95 / sqrt(2 * 0.975 * 0.025) = 4.3027`(df=2 닫힌 식; 그 외 df 는 수치 적분으로 계산, scipy 와 1e-7 이내 일치를 selftest 가 확인). 시드가 3개라 구간이 넓다.
- (a) 대 (b) 차이(`diff_agg`): 평균 차 `d = mean(a) - mean(b)`. `welch`(짝짓기 가정 없음): `se = sqrt(s_a²/n_a + s_b²/n_b)`, 자유도 `(s_a²/n_a + s_b²/n_b)² / ((s_a²/n_a)²/(n_a-1) + (s_b²/n_b)²/(n_b-1))`, CI = `d ± t_{0.975, df} * se`. `paired_by_seed_index`: `d_s = a_s - b_s`, 평균/SD/CI 는 위와 같은 t(df=n-1). 서로 다른 조건의 같은 시드 번호는 짝이라는 근거가 없고(규약일 뿐) 둘 다 병기한다.
- 하락(`drop`) = 같은 probe 의 `R²(인코더 step k) - R²(rollout step k)`. (002 의 H_b 하락은 역할 `unseen`, 마지막 step.)
- 기울기: episode 단위 최소제곱 `slope = sum (x-x̄)(y-ȳ) / sum (x-x̄)²`, `intercept = ȳ - slope x̄` (x = 참값 라벨, y = probe 예측). 레벨 평균 기울기는 9개(또는 역할 내) 레벨 평균점에 같은 식.
- 순서 정확도 = 서로 다른 크기 episode 쌍 중 예측 순서가 참 순서와 같은 비율(002 `stats.pair_order_accuracy`), 점추정.
- 어디에도 임계값, p 값, pass/fail 을 계산하지 않는다.

## 8. 검증 (CPU, 합성 데이터; 합성 수치는 결과가 아니다)
- 환경: 저장소 밖 임시 conda 환경(Python 3.9.23, numpy 1.26.4, scipy, matplotlib, torch 2.0.1+cpu), `python run.py selftest`.
- 확인하는 것: persistence 예측 == 인코더 토큰 복사에 002 경로를 적용한 값(모든 step), 인코더 step 집합 == 해당 프레임 토큰 적용, 재학습 probe 의 ridge 호출이 학습 행만 받음 + test 를 무작위/NaN 으로 바꿔도 alpha/적합 동일, 알려진 기울기/절편 복구(0.5/0.3), 알려진 축소 비율(0.8/0.5)이 A 의 기울기 비로 복구, 레벨 라벨 정렬(test 행을 섞어도 레벨 인덱스로 정렬, 한 레벨만 이동한 편향이 그 인덱스에만 나타남), 시드 단위 평균/SD/CI 가 손 계산과 일치, t 분위수 scipy 일치, Welch CI scipy 일치, 재현 점검이 동일 입력에서 차이 ~0 이고 추출을 교란하면 차이를 보고, 002 저장 형식(`q2.output_space`)과 같은 형식의 가짜 Drive 폴더에서 A-E/aggregate/figures 가 끝까지 도는지, 읽기 전용 입력 폴더/001/002/003 폴더 쓰기 거부, 쓸 수 없는 출력 폴더의 명확한 실패, 3회 재시도 후 중단, toy torch 모델로 추출 경로(002 `model_io.rollout` 참조 구현과 일치, untrained 변형/시드/구조 검사, `rebuild_aux`), `--dry-run` 14개 단계, Python 3.9 문법, 노트북 JSON/셀 컴파일, 001/002/003 파일 크기+mtime 불변.
- 결과: 아래 "검증 기록" 참조(마지막 실행 기준).

### 검증 기록
- 마지막 실행(합성 데이터, Python 3.9.23 / numpy 1.26.4 / torch 2.0.1+cpu): **PASS 92, FAIL 0, SKIP 0** (`python run.py selftest`, 약 6.5 분; 대부분 합성 Drive 폴더 생성과 파이프라인 시간).
- 실행 이력(재시도 한도 3회 이내): 1회째 실패 3건(002 `cache.load` 가 `tokens_rand.npy` 를 요구함 -> 006 전용 로더로 교체, 테스트 오류 2건), 2회째 실패 6건(원인: 합성 테스트에서 결과 파일 탐색 순서 오류로 진짜 002 `results.json` 이 먼저 읽힘 -> 후보 경로에서 제거, `t` 상수 오기, 테스트 계산 오류 -> 수정), 3회째 85/85 통과, 이후 `data_meta`/`check` 테스트와 C 의 의미 없는 within-level 지표 제거 후 4회째 92/92 통과. 숫자는 모두 합성이다.
- 건너뜀 0 이지만 **실행하지 못한 것**: 실제 체크포인트 로드, 실제 predictor 추출(CUDA), 실제 hydra 인자(`img_size, concat_dim, ...`) 검증, 실제 Drive 읽기/쓰기, 실제 데이터에 대한 마스크 규칙, Colab 환경 구성. toy torch 모델로는 추출 경로(002 `model_io.rollout` 과 일치, untrained 변형/시드, `rebuild_aux`)만 확인했다.
- 경고: 합성 D 에서 bbox 중심이 상수(무작위 노이즈 마스크)라 R² 가 정의되지 않는 값이 있어 `Mean of empty slice` RuntimeWarning 이 2건 나온다(합성 인공물; 정의되지 않는 값은 results.json 에 null).

## 9. 실행 방법

로컬(CPU 점검):
```
python run.py selftest
python run.py <stage> --dry-run
```
[2026-11-09] Colab 하위 프로세스에 `MPLBACKEND=Agg` 를 강제하고 그림 코드도 import 전에 Agg 로 설정(004 Colab 파일럿에서 커널의 MPLBACKEND 상속으로 aggregate 실패한 원인).
Colab(권장): `colab/exp006_colab.ipynb` 를 새 런타임에서 위에서 아래로 실행. 입력 `MyDrive/exp002_work`(읽기 전용), 출력 `MyDrive/exp006_work`. 환경변수: `EXP006_INPUT_ROOT`, `EXP006_WORK_ROOT`, `ACWM_PHYS_SRC`, `DATASET_DIR`, `EXP006_EXP002_DIR`.
- 셀 1-8: 환경, 코드 스냅샷(006 + 002 소스 읽기 전용 사본, 002 config sha256 대조), 고정 commit 체크아웃(패치 없음), `data_meta`, `check`, `selftest`.
- 셀 9: **smoke 가 첫 GPU 단계**(`extract --smoke`; 한 run 의 몇 배치로 메모리/시간 추정). 읽고 나서 셀 10 의 `RUN_EXTRACT = True` 로 바꾼다(GPU 사용 확인, 기본 꺼짐).
- 셀 11-12: CPU 분석, `aggregate`(results.json 을 실험 폴더와 `exp006_work/results/` 둘 다에), `figures`.
- Drive 쓰기 실패는 모든 쓰기가 쓰고 다시 읽어 검증하므로 즉시 실패한다(`OutputCheckError`). 런타임이 사라져도 `exp006_work` 에 남는다.

## 10. 한계와 알려진 위험
- 사후 탐색이며 확증 근거가 아니다. 임계값/판정 없음.
- 추출은 이 머신의 GPU 에서 재계산한 값이다. 002 가 같은 값을 저장하지 않았으므로 재현 차이(`repro`)를 먼저 본다. **차이가 크면 B(untrained), C, D 를 믿지 말라**(`repro.banner` 가 숫자를 출력한다; 임의 허용오차로 판정하지 않는다). 배치 구성, cuDNN 커널, GPU 종류가 달라서 생기는 부동소수점 차이는 있을 수 있다(추정).
- 시드 단위 CI 는 n=3(df=2), 에피소드 bootstrap CI 는 평가 episode 만 반영한다.
- 풀링된 latent 와 linear probe 만 쓴다(attentive 불가).
- D 의 "그리퍼" 속성은 명령 누적량이지 측정 위치가 아니다. 마스크 속성은 HSV 규칙과 224 해상도에 의존한다.
- untrained predictor 는 초기화 시드/변형에 따라 달라지며 3x2 를 따로 보고한다.
- 체크포인트 로드, hydra 인자(`tcfg.img_size, concat_dim, ...`), 행동/proprio 인코더 `in_chans` 추정(행동 30 = 5 x 6, proprio 1)은 실제 체크포인트에서 검증하지 못했다. 맞지 않으면 구조 비교(`same_structure`)가 명확한 에러로 중단시킨다.
- 노트북 전체(Drive 마운트, uv 환경, 스냅샷 풀기, 고정 commit 클론, 실제 추출)는 Colab 에서 실행해 보지 못했다.
