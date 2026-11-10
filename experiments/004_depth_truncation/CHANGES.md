# CHANGES: 004_depth_truncation (명세 notes/specs/exp4_5_spec.md 3절 대비 달라진 점)

명세 0절 4 에 따라 원본과 달라진 점을 전부 기록한다. 해석은 하지 않는다. "확인" 은 이 저장소/머신에서 한 것만 뜻한다.

## 1. 사용자 결정으로 달라진 것
1. **범위: 코드 + 합성 데이터 CPU 셀프테스트 + `--dry-run` 까지.** GPU 실행, 데이터 다운로드, 서버 접속 없음. 결과 파일(results/)은 아직 없다. (hypotheses.md 후보 I 체크리스트, 확정 기록 9)
2. **4-C(깊이 재학습) 생략.** predictor 를 깊이 1·2·4·6 으로 새로 학습하는 코드는 없다. `summary.json` 의 `l_best` 는 4-A/4-B 의 probe 곡선에서만 정한다.
3. **저장소 관례 경로.** `results/exp4/...` 대신 `experiments/004_depth_truncation/results/...` (config.yaml, run.py, README.md, CHANGES.md, results.json 은 실험 폴더 바로 아래). 명세 0절 1 "기존 관례를 따른다".

## 2. 기존 설정(002 확정 기록 8)에서 온 것
4. **num_hist = 1, frameskip = 5, batch 24, 15 epoch.** 명세의 "마지막 context 프레임"은 유일한 context 프레임이다. 캐시 프레임 0..12(매크로 프레임, 5 프레임 간격). teacher forcing 윈도우는 프레임 1장(H=1). 002 config 에서 읽으며 코드가 `assert int(tcfg.num_hist) == 002 train.num_hist` 로 체크포인트와 일치를 확인한다.
5. **시점 정의.** `early` = 프레임 0-3, `late` = 프레임 8-11 (002 `frames.early_flat/late_deformed`, "평평함/변형이 큼"은 인덱스 가정이며 영상으로 확인하지 않음). 캐시는 이 8프레임만 저장한다(13프레임 전체 대비 용량 8/13).
6. **rollout 시점은 t+1, t+4, t+8 을 첫 프레임(t = H-1 = 0)에서 시작한 것만.** 후반 프레임에서 시작한 rollout 은 없다. rollout 시점에서의 은닉 상태 probe (i)는 계산하지 않는다(은닉 probe 는 early/late teacher forcing 윈도우만). rollout 은 (ii)(iii)에만 있다.
7. **probe 학습 설정은 002 `probe_impl` 그대로** (attentive: d_model 128, 30 epoch, lr 1e-3, AdamW...; linear: 풀링 + ridge). 명세 2절의 SparseImagination A.5 (Adam lr 1e-5, batch 128, 500 epoch)와 다르다. 002 가 이미 같은 이유(실험 간 비교)로 정한 설정을 import 해서 쓴 결과이다.

## 3. 명세와 코드 실제가 달라서 바꾼 것 (§6 가정 확인, 고정 commit 5a640be 를 `git show` 로 읽음)
| §6 가정 | 확인 방법 | 결과 |
|---|---|---|
| DINO-WM 인코더는 DINOv2 ViT-S/14(12 block, 384차원) | `conf/encoder/dino.yaml` (`dinov2_vits14`), `models/dino.py` (`emb_dim = base_model.num_features`) | **부분 확인.** 모델 이름은 코드로 확인. 12 block / 384 는 hub 모델(이 머신에 없음)의 구조이므로 **미확인 -> 런타임 확인**: `enc_cache` 가 block 수를 로그로 남기고 `k_grid` 최댓값이 block 수를 넘으면 중단하며, 384 는 002 가 `D_VIS = 384` 로 쓰고 첫 배치 일치 검사가 틀리면 중단한다. |
| predictor 는 6 block | `conf/predictor/vit.yaml` `depth: 6` | **기본값 확인**(002 가 override 했는지는 미확인). 코드는 모델에서 읽는다(`M.n_blocks`): `m` 범위와 `summary.json predictor_layers` 가 실제 값이 된다. 하드코딩 없음. |
| 입력 해상도 224, 패치 16x16 = 256개 | `models/visual_world_model.py` | **틀림.** `decoder_scale = 16`, `num_side_patches = 224 // 16 = 14`, `encoder_image_size = 14 * 14 = 196` px 로 리사이즈 -> 패치 14x14 = **196개**. 데이터 해상도는 224, 인코더 입력은 196px. 002 의 `P_TOK = 196` 과 같다. 캐시 용량 추정은 196 패치로 했다(토큰 캐시 k 당 약 2.2 GB, 아래). |
| 실험 1 probe, 실험 2 predictor (b) 체크포인트가 있다 | 저장 경로 | **이 머신에 없음.** 체크포인트는 `run_dir002` 가 가리키는 `DINO_TRAIN/outputs/exp002_multi_60_ind_160_s<seed>/DONE` 를 확인하고 없으면 `FileNotFoundError` 로 중단한다(dry-run 은 통과). **실험 1 probe 는 디스크에 저장되지 않는다**(002 `q1` 은 cells.npz 만 저장). 그래서 (ii)의 "실험 1 probe" 는 002 Q2 output_space 와 같은 방식으로 인코더 토큰(regime `all`, 프레임 0..11)만으로 같은 코드(`evalgrid.fit_apply_cells`)와 같은 시드로 **다시 학습**한다. |
| mask 면적 계산 함수가 있다 | 002 `masks.py` | **있음.** HSV 규칙(`frames_mask_features`)이고 002 캐시의 `mask_feat` (면적, bbox 폭, 높이)을 쓴다. 새로 구현하지 않았다. |

추가로 코드에서 확인한 것(명세에 없던 사실):
8. **predictor 에는 출력 projection 이 없다.** `ViTPredictor.forward` = `x + pos_embedding` -> dropout -> `Transformer`(블록 `x = attn(x) + x; x = ff(x) + x` 반복 -> 최종 LayerNorm). 명세 3.2 의 "최종 norm 과 출력 projection" 중 projection 은 없으므로 **조기 종료 = block 1..m + 최종 norm** 이다. 출력 차원은 입력과 같은 404(z 384 + proprio 10 + action 10, `concat_dim=1`).
9. **m = 0 의 정의.** 은닉 상태 m=0 = 위치 임베딩을 더한 입력(002 `StreamHooks` 의 layer 0 과 같음), 은닉 상태 m = block m 개를 지난 잔차 스트림(layer j=m 과 같음; 셀프테스트가 002 `StreamHooks` 캡처와 모든 m 에서 일치함을 확인). 조기 종료 출력 m = `final_norm(은닉 m)`; m=0 의 출력은 위치 임베딩이 더해진 입력에 최종 norm 만 적용한 값이다.
10. **z 부분만(주지표 (i)).** 명세대로 앞 384 좌표만 쓴다 (config `predictor.feature_part: z_only`). 002 의 `q2_impl.feature_part` 기본값은 `full`(404)이다 — 블록 1 이후 앞 384 좌표가 z 의 부분공간이라는 보장이 없다는 002 README 의 우려는 그대로 남는다. 두 값의 비교는 이 코드에서 `feature_part: full` 로 다시 돌리면 된다(`pred` 는 `dpart` 를 바꿈).
11. 실제 predictor `Attention` 이 `self.bias = ....to('cuda')` 로 CUDA 를 하드코딩한다 -> 실제 predictor 는 CPU 로 실행 불가(서버는 GPU 이므로 영향 없음). 합성 셀프테스트는 같은 구조의 토이 predictor 를 쓴다.

## 4. 구현 선택 (명세에 값/정의가 없어 내가 정한 것; config 에 명시, 파일럿 전에 사람이 바꿀 수 있음)
12. **인코더 절단은 `get_intermediate_layers(norm=True)` 가 아니라 직접 루프**(`deep.tokens_by_k`: `prepare_tokens_with_masks` -> block 순차 -> `norm`, cls/register 토큰 제외). 이유: `get_intermediate_layers` 는 k=0(패치 임베딩)을 줄 수 없다. 같은 의미(블록 출력에 최종 LayerNorm)인지는 (a) 토이 ViT 에서 `get_intermediate_layers` 와 일치(셀프테스트), (b) 서버 첫 배치에서 k=12 가 002 `tokens_dino` 와, random-init k=12 가 `tokens_rand` 와, 그리고 hub 모델의 `get_intermediate_layers(n=[11], norm=True)` 와 일치하는지(일치하지 않으면 `enc_cache` 가 중단)로 확인한다. (b)는 서버에서만 확인된다.
13. **소실 지점 정의의 구체화.** 주 곡선 = linear probe, fit=new, control=real, timepoint=late (config `depth.primary`). 곡선은 probe seed(x 학습 seed) 평균. `k_star`/`m_star` = split 별 소실 깊이의 최솟값, `k_star_by_split`/`m_star_by_split` 도 저장. `loss_ood_only` = (두 모듈 모두 train split 에서 소실 없음) 이고 (어느 모듈이든 interp/extrap 에서 소실 있음). `l_best` = (module, depth) 중 `l_best_splits`(interp, extrap) 평균 R² 최대, 동률은 인코더/작은 깊이 우선. `l_best_by_module` 도 저장(실험 5 가 인코더 값을 쓰기 위해).
14. **"곡선이 seed 편차 안에서 겹침" 판정 규칙.** |평균(train) - 평균(OoD split)| <= `overlap_sd_mult`(2.0) x max(sd) 인 깊이의 비율이 `overlap_depth_fraction`(0.8) 이상이면 겹침. 두 값은 impl default 이다(명세에 없음). 모든 깊이를 요구하면 순수 잡음에서도 자주 겹침으로 판정되지 않아 셀프테스트에서 비율 규칙으로 바꿨다. seed 가 1개면 sd 가 정의되지 않아 `not_evaluable`.
15. **레벨 행의 R².** 레벨 안에서 라벨(명목 면적비)이 상수라 레벨별 R² 는 정의되지 않는다. 레벨 행은 `1 - MSE_level / Var(전 평가 episode 의 y)` (pooled-variance R²) 이고 Pearson 은 빈칸. split 행(`level == ALL`)이 일반 R² 이다. 히트맵과 곡선 중 히트맵만 레벨 행을 쓴다.
16. **CSV 확장 열** `control, train_seed, pred_mean, n_episodes` 를 명세 열 뒤에 추가. `module` 값에 `baseline`(pixel, mask_area_bbox)과 `reference`(인코더 토큰에 실험 1 probe 적용, 상한 기준) 추가.
17. **raw pixel baseline.** 프레임(224x224)을 14x14 격자(패치 16px)로 나눠 각 칸을 4x4 로 평균 -> 토큰 196개 x 48차원 (같은 probe 두 종류 적용). 명세는 "raw pixel probe" 만 말한다.
18. **mask baseline = 002 의 5개 특징 ridge**(log 면적, log 폭, log 높이, log bbox 넓이, 면적/bbox 넓이). "mask 면적 baseline" 보다 특징이 많다(002 Q1 과 같은 기준).
19. **통제 범위.** 인코더: random-init(002 의 `probe_impl.random_init`, 같은 시드 = 002 `tokens_rand`), 라벨 셔플(episode 단위, 002 `shuffle_seed`). predictor: random-init predictor(깊이 같음, 실제 인코더 z 입력, 학습 seed 마다 `random_init_seed + train_seed`)는 **은닉 probe (i) 만**, 라벨 셔플도 (i) 만. random-init predictor 의 (ii)(iii)은 계산하지 않는다.
20. **판정 필드 `verdict_H_d`.** supported / refuted / not_evaluable 와 사유(reason)를 `summary.json` 에 쓴다. 파일럿(`pilot: true`)에서는 점검용 값이다.
21. **격자 재탐색 자동화 없음.** 소실 구간이 잡히면 `refine_suggestion` 에 직전/소실 깊이만 기록한다. 더 촘촘한 격자는 `encoder.k_grid` 를 바꿔 `enc_cache` 부터 다시 돌려야 한다(캐시가 k 별 파일이라 새 k 만 추가로 만드는 기능은 없음).
22. **통계.** 명세 3.5/0절 6 대로 seed 평균 ± 표준편차를 쓰고, 002 처럼 episode cluster bootstrap CI 는 만들지 않았다.
23. **파일럿 표시.** `pilot` 은 config `pilot: true` 이거나 부분 캐시(`--suffix`)를 썼으면 true. 본 실행 전 사람이 config 를 false 로 바꾼다.

## 5. 만들지 못한 것 / 검증하지 못한 것
- 4-C, 깊이 재학습 (생략).
- `pred` 의 실제 모델 경로 전부, `enc_cache` 의 실제 hub 모델 경로, 서버(python 3.9, torch 2.0.1)에서의 실행: **미실행**.
- 원본 설계 문서(claude.ai) 는 이 저장소에 없고 명세 파일만 읽었다.

## 6. 비용 추정 (추정: 실측 없음)
- `enc_cache`: 전체 1800 episode x 8프레임, k_grid 7개, random-init 포함 -> 약 30 GB(토큰 k 당 2.17 GB x 7 x 2 + pixel 0.27 GB, `--dry-run` 이 출력). 파일럿(`cache --max-eps 25`)이면 300 episode -> 약 5 GB. 시간: ViT-S 순전파 1800 x 8 x 2(+random-init) 장 -> 12GB GPU 에서 수 분(추정).
- `enc_probe`: 225 fit(linear 포함; attentive 는 105 + pixel 10). attentive 1 fit = 7200 샘플 x 30 epoch 이므로 1 fit 15~30초(추정, 실측 필요) -> 약 0.5~1시간.
- `pred`: 학습 seed 마다 임시 디스크 약 26 GB(6 block 가정, 'real'), 은닉 probe 140 fit (+ exp1 probe 10 fit + 88 개 집합 적용) -> seed 당 약 1~1.5시간(추정). 3 seed x (real + random-init) 약 3~4시간.
- 합계: 파일럿 1시간 안팎, 본 실행 5~6시간(모두 추정, 실측 필요: 첫 attentive fit 시간을 로그로 확인).

## 7. Colab 노트북 (2026-11-09, 운영 변경만; 실험 값 변경 없음)
- 추가: `colab/exp004_005_colab.ipynb`(004 -> 005), `colab/build_notebook.py`(노트북 생성, 코드 스냅샷 내장), `colab/exp045_tools.py`(표준 라이브러리만: stage 실행 + 시간/VRAM 기록, 입력 점검, 링크, tar 보관, 비용 투영), `colab/requirements_colab_extra.txt`(matplotlib). 002 `colab_tools.py`/`requirements_colab.txt` 는 수정 없이 스냅샷에 복사해 사용(요구 파일).
- **코드 수정(운영, 필수): `run.py` 가 `exp002.common.attempts_check` 를 import 했는데 002 의 커밋 545168e("stop attempts.json refusing runs")가 그 함수를 삭제해 004/005 `run.py` 가 ImportError 로 시작하지 못했다**(셀프테스트도 실패). 002 를 건드리지 않고 `src/exp004/base.py`, `src/exp005/base.py` 에 같은 동작(`run.max_attempts` 도달 시 RuntimeError, `<results_dir>/attempts.json` 기록은 그대로 `exp002.common.attempts_record`)의 `attempts_check` 를 복사하고 `run.py` 가 거기서 import 한다. 재시도 상한 3회 동작은 그대로다.
- 운영 값(config.yaml 이 아니라 노트북/환경변수): 경로(`exp002_work`, `exp004_work`, `exp005_work`, `/content/main|pilot`), 파일럿 suffix `_smoke`, `--results-dir`(파일럿 `results_pilot`). config.yaml 의 시드와 임계값은 변하지 않았다(004). 파일럿 결과와 본 결과는 폴더, 캐시 루트, suffix 가 모두 분리된다.
- 수정(Colab 파일럿에서 발견): 004 aggregate 가 `import matplotlib` 에서 ValueError(커널의 MPLBACKEND=module://matplotlib_inline.backend_inline 이 venv 하위 프로세스에 상속)로 실패. `exp045_tools` 의 stage 실행기(002 download/cache 포함)가 하위 프로세스 env 에 MPLBACKEND=Agg 를 무조건 설정하고, 004 `figs.py` 와 005 그림 코드도 import 전에 무조건 Agg 로 설정한다(그림은 파일로만 저장).
- 검증(이 머신, CPU): 노트북 JSON 유효성(nbformat), 코드 셀 문법, `exp045_tools.py selftest`, 004/005 selftest, 모든 단계 `--dry-run`. 미검증: 노트북의 모든 셀(Colab 실행 없음), 002 `download`/`cache` 호출 경로, VRAM 샘플링(nvidia-smi 전체 GPU 사용량 폴링, 최대값은 샘플링 하한), tar 보관 셀의 대용량 동작.
