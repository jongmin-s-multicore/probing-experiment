# 실험 설명: 004_depth_truncation

- 상태: **코드 작성 완료, 서버 미실행.** 이 머신(GPU/데이터/체크포인트 없음)에서는 문법 검사, `--dry-run`, 합성 데이터 CPU 셀프테스트만 했다. 학습, 데이터 다운로드, GPU 실행, 서버 접속은 하지 않았다. 이 문서에 실험 결과 수치는 없다. 명세와 달라진 점은 전부 CHANGES.md 에 있다.
- 검증할 가설: notes/hypotheses.md 후보 I (확정 기록 9), **H-d**. 천 크기 R²(log 명목 면적비)는 어떤 깊이 L* 이후 급격히 줄고, OoD 크기에서 L* 가 InD 보다 앞선다. 반증: 모든 깊이에서 하락 < 0.2, 또는 InD/OoD 곡선이 seed 편차 안에서 겹침. 임계값 0.2 는 명세의 제안값(파일럿 후 확정, 바꾸면 결과보다 먼저 기록). 명세: notes/specs/exp4_5_spec.md 3절.
- 방법: DINO-WM(조건 (b) `multi_60_ind_160`, 002 가 학습)의 **인코더(4-A)와 predictor(4-B)를 앞에서부터 일부만** 써서 각 깊이에서 크기 probe 를 새로 학습하고, 소실 지점(R² 가 직전 최고값보다 0.2 이상 처음 떨어지는 깊이)을 찾는다.
  - 4-A: DINOv2 block 1..k 통과 + 최종 LayerNorm 한 패치 토큰, k ∈ `encoder.k_grid` (기본 {0,2,...,12}). k=0 은 패치 임베딩(+위치 임베딩). 같은 표에 raw pixel probe, mask 면적/bbox baseline, random-init 인코더, 라벨 셔플.
  - 4-B: predictor block 1..m + 최종 norm(= 조기 종료). m 은 모델에서 읽는다(하드코딩 없음). (i) 마지막 context 프레임의 **z 부분(앞 384)** 은닉 상태에 새로 학습한 probe(주지표), (ii) 조기 종료 예측 z 에 실험 1 probe(인코더 토큰으로만 학습, `fit=exp1_probe`)를 그대로 적용, (iii) 다음 z MSE. random-init predictor 와 라벨 셔플 통제 포함.
  - 4-C(깊이 재학습)는 구현하지 않았다(사용자 결정).
- 재사용: 002 의 probes/evalgrid/stats/masks/model_io/cache/data/common 을 **수정 없이 import** 한다(같은 probe 구현, 분할, 지표). 001/002/003 파일은 수정하지 않았다. 002 설정 파일은 읽기만 하고 `probe.kinds/seeds` 만 이 실험 config 값으로 덮어 쓴다.
- **이 실험으로 내가 주장하려는 것** (확정 전 초안, 해석은 분석가 리뷰 몫): 조건 (b) DINO-WM 의 (1) 인코더 깊이 k 와 (2) predictor 깊이 m 에 따라 천 크기 정보를 probe 로 읽을 수 있는 정도(R², seed 평균 ± 표준편차)가 어떻게 변하는지, 그 값이 R² 하락 0.2 기준으로 어느 깊이에서 처음 꺾이는지(k*, m*), 그 깊이가 학습 레벨/보간 레벨/외삽 레벨에서 다른지(`loss_ood_only`), 그리고 그 결과가 실험 5 의 층 선택(4.6 분기)에 어떤 입력을 주는지. probe 정확도만으로는 "읽을 수 있다"고만 쓰고 "사용된다"는 쓰지 않는다. 숫자와 판정 표(supported/refuted/not_evaluable)만 만들고 해석 문장은 쓰지 않는다. 코드가 만드는 판정은 파일럿(`pilot: true`)에서는 점검용이다.
- 성공 기준: config.yaml `depth.*` 에 있는 값을 코드가 읽어 판정한다. 임계값이 null 이면 `not_evaluable`.

## 1. 구조

```
experiments/004_depth_truncation/
  config.yaml     모든 시드(probe seed 5개, 전역 seed)와 임계값
  run.py          단계 CLI: check enc_cache enc_probe pred aggregate selftest   (모두 --dry-run 지원)
  CHANGES.md      명세와 달라진 점 전부, §6 가정 확인 결과
  src/exp004/
    base.py       002 src 를 sys.path 에 올리고 config/경로 처리 (002 의 resolve_paths 는 002 폴더를 가리키므로 따로)
    deep.py       torch: 인코더 절단 tokens_by_k, predictor 조기 종료 ee_forward/predict_exit, random-init predictor, 한 번에 추출 extract
    stages.py     단계 구현 (4-A cache/probe, 4-B 추출/probe, aggregate, check)
    depthlib.py   numpy: 행(csv) 생성, 소실 지점 first_drop, summary.json, H-d 판정
    figs.py       fig_heatmap.png, fig_curves.png (matplotlib 이 있을 때)
    selftest.py   합성 데이터 CPU 셀프테스트
  results/        (실행 시 생성) parts/*.csv, depth_r2.csv, next_z_mse.csv, summary.json, 그림
  results.json    (aggregate 가 생성) 요약 + summary 사본 + 환경 + config sha256
```

## 2. 단계

| 단계 | 하는 일 | GPU |
|---|---|---|
| `check` | 환경, 경로, 002 캐시/체크포인트 존재, config 값 출력. 읽기만 | 선택 |
| `enc_cache` | 002 Q1 캐시의 frames_u8(초기·후반 8프레임)에서 k 별 절단 토큰(+random-init, raw pixel 토큰)을 `$EXP004_CACHE/enc<suffix>/` 에 저장. 첫 배치에서 k=12 가 002 의 `tokens_dino` 와, random-init k=12 가 `tokens_rand` 와 일치하는지(그리고 `get_intermediate_layers(norm=True)` 와도) 검사하고 틀리면 중단 | 필요 |
| `enc_probe` | k 별 linear/attentive probe x 5 seed (real, random-init, 라벨 셔플), pixel baseline, mask baseline -> `results/parts/encoder.csv` (단위마다 저장, 재실행 시 이어서) | 선택 |
| `pred` | 002 조건 (b) 체크포인트 별: 한 번의 패스로 모든 m 의 은닉 상태, 조기 종료 출력, 조기 종료 rollout(t+1,t+4,t+8) 추출 -> 임시 디스크 -> (i)(ii)(iii) 행 생성 -> 임시 파일 삭제. `--controls real,random_init` | 필요 |
| `aggregate` | parts -> `depth_r2.csv`, `next_z_mse.csv`, `summary.json`, 그림, `results.json` | 불필요 |
| `selftest` | 합성 데이터 CPU 셀프테스트 (결과 아님) | 불필요 |

모든 단계는 `logs/<단계>_<시각>.log` 에 stdout/stderr 를 남기고, 같은 단계가 3회 연속 실패하면 `results/attempts.json` 때문에 실행을 거부한다(CLAUDE.md 재시도 상한).

## 3. 출력 (명세 3.5)

- `results/depth_r2.csv`: 열 `module, depth, probe, fit, level, split, timepoint, seed, r2, mae, pearson` (명세 순서) + 확장 열 `control, train_seed, pred_mean, n_episodes`.
  - `module`: encoder / predictor / baseline(pixel, mask_area_bbox) / reference(인코더 토큰에 실험 1 probe 를 그대로 적용한 값). `fit`: new / exp1_probe. `probe`: linear / attentive. `control`: real, random_init, label_shuffle, pixel, mask_area_bbox, encoder_tokens.
  - `timepoint`: early(프레임 0-3), late(프레임 8-11) = 002 `frames` 와 같은 정의. (ii)(iii)에는 `rollout_t+1`, `rollout_t+4`, `rollout_t+8` 도 있다. early/late 에서 (ii)(iii)의 대상은 teacher forcing 으로 예측한 다음 프레임(f+1)이다.
  - `level == ALL` 행은 split(train/interp/extrap, `all`)에 속한 episode 전체의 R², MAE, Pearson. 레벨 행은 레벨 안에서 라벨이 상수라 R² 가 정의되지 않으므로 `r2 = 1 - MSE_level / Var(전 평가 episode 의 y)`(pooled-variance R²), Pearson 은 빈칸이다. 히트맵은 이 레벨 행을 쓴다.
- `results/summary.json` (실험 5 입력): 명세 필드 `pilot, encoder_layers, predictor_layers, l_best, k_star, m_star, loss_ood_only, drop_threshold, notes` + 추가 `k_star_by_split, m_star_by_split, l_best_by_module, vanish_detail, ind_ood_gap, refine_suggestion, verdict_H_d, definitions`. `pilot` 은 config `pilot: true` 이거나 부분 캐시(`--suffix`)를 썼으면 true.
- `results/next_z_mse.csv`, `results/fig_heatmap.png`, `results/fig_curves.png`, `results.json`.

### 소실 지점과 판정 정의 (코드: depthlib.py)
- 주 곡선(config `depth.primary`): linear probe, fit=new, control=real, timepoint=late. 곡선은 probe seed(x 학습 seed) 평균. **소실 깊이** = 평균 R² 가 직전까지의 최고값보다 `depth.drop_threshold`(0.2) 이상 낮아지는 첫 깊이, 없으면 null. 인코더 `k*`, predictor `m*` 따로, split(train/interp/extrap) 별로 따로.
- `k_star`/`m_star` = split 별 값의 최솟값. `loss_ood_only` = (train split 에서 두 모듈 모두 소실 없음) 이고 (interp/extrap 중 한 곳에서 어느 모듈이든 소실 있음). 둘 다 이 구현의 선택이며 명세에 정의가 없다(CHANGES.md).
- H-d 판정: 모든 깊이/split 에서 하락 < 0.2 -> refuted; train 과 interp/extrap 평균 곡선의 차가 `overlap_sd_mult` x max(seed sd) 이내인 깊이의 비율이 `overlap_depth_fraction` 이상이면 겹침 -> refuted; 소실이 있고 OoD 쪽이 InD 보다 앞서며 곡선이 겹치지 않으면 supported; OoD 가 더 늦으면 refuted; seed 가 1개이거나 깊이가 `min_depths` 미만이면 not_evaluable. 숫자만 기록한다.

## 4. 실행 방법

선행 조건(002): 조건 (b) `multi_60_ind_160` 학습 checkpoint(`DONE` 표시), 002 `cache` 단계의 Q1 캐시(`frames_u8, tokens_dino, tokens_rand, mask_feat, actions_raw`). 이 머신에는 둘 다 없다.

```bash
cd experiments/004_depth_truncation
conda activate dino_wm                       # 002 와 같은 환경 (추가 의존성 없음. figures 에는 matplotlib 이 있으면 쓴다)
export DINO_WORK=... DINO_TRAIN=... EXP002_CACHE=... ACWM_PHYS_SRC=... EXP004_CACHE=...   # 002 와 같은 값 + EXP004_CACHE
python run.py selftest                        # CPU, 합성 데이터
python run.py check
# 파일럿 (부분 캐시: 002 에서 `python run.py cache --max-eps 25 --suffix _smoke` 를 먼저)
python run.py enc_cache --suffix _smoke --dry-run && python run.py enc_cache --suffix _smoke
python run.py enc_probe --suffix _smoke
python run.py pred --suffix _smoke --train-seeds 0
python run.py aggregate                       # summary.json pilot=true
# 본 실행: --suffix 없이 전체 캐시, pred 는 학습 seed 0,1,2, config pilot: false 는 사람이 바꾼다
```

## 5. 알려진 한계와 미검증 사항 (정직하게)
- **서버에서 실행되지 않은 코드**: `enc_cache` 의 실제 DINOv2 hub 모델 경로(`base_model.blocks/norm/prepare_tokens_with_masks`, 블록 청크 구조는 토이로만 확인), `pred` 의 `load_world_model` 이후 전부, 실제 predictor 의 `Attention.bias.to('cuda')` 하드코딩 때문에 CPU 로는 실제 predictor 를 돌릴 수 없음. 합성 장난감 모델로 검증한 것은 추출 로직, 조기 종료(m=L 일치, 002 StreamHooks 와 층 인덱스 일치), probe 행 생성, 소실 지점/판정 로직뿐이다.
- 시간/비용은 모두 추정이다(CHANGES.md 하단 "비용 추정"). attentive probe 한 번의 실측 시간이 없다.
- 자세한 가정 확인 결과(§6): CHANGES.md.
