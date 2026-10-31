# 실험 설명: 001_size_collapse_diagnosis

- 상태: 코드 구현 완료, **실행 전**. 학습·데이터 다운로드·GPU 작업은 하지 않았다. 문법 검사(`py_compile`), `--dry-run`, 합성 데이터 CPU 셀프테스트(`python run.py selftest`, 가짜 rclone 다운로드 시험 포함)만 했다. 데이터 출처는 Google Drive(rclone)로 바꿨다(위 "데이터" 행, 아래 다운로드 절). 이 문서에 실험 결과 수치는 없다.
- 개정 이력: 2026-10-31 데이터셋을 원본 ACWM-Phys(크기 3종)에서 사용자가 PyFleX 로 재생성한 `cloth_sim`(크기 9종)으로 바꿨다. 이전 버전은 commit bfddd9d. 이 개정은 작업 트리에만 있고 commit 하지 않았다.
- 검증할 가설: notes/hypotheses.md 의 후보 D (디코더 층위), 후보 E (인코더/프리딕터 정보 층위).
  - 두 후보 모두 "후보 (미확정, 사람 승인 필요)" 상태다. 이 실험은 확정이 아니라 진단용 첫 단계다.
  - 후보 F (크기 섞어 재학습)는 이 실험의 범위가 아니다. 다만 학습 분포를 config 로 고를 수 있게 대조군 코드를 지원한다. 모든 학습 분포(baseline 포함)는 "사람 확정 필요"이고, 승인되지 않은 분포는 `run.py train` 이 거부한다.
- 배경: acwm-phys Cloth Move 에서 작은 천으로 학습한 DINO-WM 이 큰 천을 받으면 rollout 에서 학습 크기로 수축한다(acwm-phys 브랜치 README 서술, 단일 시드). 사용자가 재생성한 데이터(cloth_sim)에서도 baseline(ind_train 학습)이 a026/a248 에서 같은 현상을 보이는지는 보장되지 않아, 그 확인을 가장 먼저 한다(아래 "재현 확인 단계").

## 개정 요약 (cloth_sim 반영)

| 항목 | 이전 | 지금 |
|---|---|---|
| 데이터 | HF `t1an/ACWM-Phys`, 크기 3종(smaller 0.26 / ind / larger 2.5), 군집당 50 에피소드 | 사용자의 Google Drive 폴더(기본, rclone; HF `hanbin5/cloth-sim` 은 `data.source: hf` 로 선택 가능하나 사용자가 당분간 쓸 수 없다고 함), 크기 9종, 크기마다 train 2,000 / test 100 (평가는 9 x 100 = 900 에피소드). `config data.dataset` 로 선택, `acwm_phys` 는 기존 구현으로 남김 |
| 중간 크기 | 외부 생성기 래퍼 `gen_midsize.py` (GEN=1) | **제거**. 이유: 9종이 데이터에 이미 들어 있어(면적비 0.259 ~ 2.479) 중간 크기를 따로 만들 필요가 없고, 원 생성기를 구하지 못해 래퍼가 동작하지도 않았다. 보간/외삽은 학습 크기를 정해 두고 나머지 크기를 평가하는 방식으로 본다 |
| 학습 분포 | baseline_single, mix_ood3 (ood_test 에서 25+25 차용) | baseline_single(ind_train), mix_adjacent(a080+ind+a125), mix_wide(a040+ind+a160). 모두 `*_train` split 에서만 가져오므로 평가(`*_test`)와 겹치지 않는다. 모두 사람 확정 필요 |
| 크기 곡선 | 3점 | 9점 (면적비 0.259 ~ 2.479) |
| probe | 3-way 분류, 프로토콜 3종 | 9-class 정확도 + 인접(+-1) 클래스 정확도, 프로토콜 4종 + heldout_size(보간/외삽) |
| 재현 확인 | 없음 | `repro_check` 단계 + 마스크 규칙 점검 `maskcheck` 단계 추가, run_all 은 repro_check 뒤에서 멈춤(게이트) |

## 이 실험으로 내가 주장하려는 것

확정된 주장이 아니다. 이 실험은 아래 질문에 대한 **측정값**을 만든다. 주장은 사람이 성공 기준을 확정하고, 결과가 나오고, /review-experiment 를 거친 뒤에만 정한다.

0. (재현) 재생성 데이터에서 baseline(ind_train 학습)이 a026/a248 에서 "학습 크기로 수축"을 보이는지. 수치만 기록하고 판정하지 않는다.
1. (D) 천 마스크 면적에서 `decode(encode(real))`(predictor 없음)이 학습하지 않은 크기에서 real 과 얼마나 다른지, 그리고 teacher-forced 1-step / open-loop k-step 의 디코딩 면적이 어느 단계에서 real 과 벌어지기 시작하는지. 디코더를 거치지 않는 latent 거리로도 같은 질문을 본다. 9개 크기에 대한 크기-면적 곡선(실제 면적 대 모델 면적, 군집 평균 기울기와 bootstrap CI)을 만든다.
2. (E) 동결 DINOv2 GT 인코딩에서 크기가 probe 로 복원되는지(선택성은 control 대비), 같은 probe 가 predictor 출력(1-step, k-step)에서 예측 크기 대 실제 크기 기울기가 어떻게 되는지, 그 정보가 predictor 블록 깊이에 따라 어떻게 변하는지, 크기 방향을 건드렸을 때 predictor 출력이 따라 움직이는지(무작위 방향 대비). 월드 모델이 학습한 크기만으로 만든 probe 를 학습 안 한 크기(보간/외삽)에 적용한 값도 낸다.
3. 위 수치로 "디코더가 원인이다 / 인코더 정보가 없다 / predictor 에서 줄어든다 / 판단 불가" 중 무엇이 데이터와 맞는지는 **결과 후에** 정한다. 지금은 주장하지 않는다.

## 코드 출처와 cloth-sim 브랜치 확인 (확인한 사실 / 확인하지 못한 가정)

### cloth-sim 브랜치 (2026-10-31 확인 시도, 사용자 push 통지 후 2차 시도 포함)

2차 시도(사용자가 push 했다고 알림): `git -C thread0 fetch upstream hanbin5/cloth-sim` 과 `git -C thread0 fetch origin hanbin5/cloth-sim` 모두 `fatal: couldn't find remote ref hanbin5/cloth-sim` 로 **여전히 실패**. `git ls-remote --heads upstream` = 7개 head(위와 동일, `hanbin5/cloth-sim` 없음), `git ls-remote --heads origin`(probing-experiment.git) = main 하나. `git cat-file -t 3e7eb8f` 실패. 원인은 확인 못 함(push 가 이 저장소가 아닌 다른 원격/이름으로 갔거나, 아직 반영 안 됨). 그래서 가정 1~3 검증과 코드 반영은 **하지 않았다**(추측으로 코드를 바꾸지 않음). 브랜치 이름/원격 URL 을 알려 주면 다시 확인한다.


| 확인한 것 | 결과 |
|---|---|
| `git -C thread0 branch -a` | 로컬 ref 에 `hanbin5/cloth-sim` 없음 (있는 것: upstream/{211shyn/ACWM-baseline, hanbin5/acwm-baseline, hanbin5/acwm-phys, hanbin5/local, hanbin5/occlusion, hanbin5/pusht-long-horizon, main}) |
| `git fetch upstream hanbin5/cloth-sim` | **실패**: `fatal: couldn't find remote ref hanbin5/cloth-sim` |
| `git ls-remote --heads upstream` | 원격 head 7개뿐이고 `hanbin5/cloth-sim` 은 없음 (아직 push 되지 않았거나 다른 저장소/이름일 수 있음. 원인은 확인 못 함) |
| 사용자가 준 생성 코드 커밋 `3e7eb8f` | 로컬 객체 DB 에 없음 (`git cat-file -t 3e7eb8f` 실패, `git fetch upstream` 후에도 동일) |
| `datasets/acwm_phys_dset.py` (cloth-sim 판), `conf/env/cloth_sim.yaml`, `repro/cloth_sim/` | **읽지 못함** (브랜치가 없으므로). 존재 여부 미확인 |
| config.yaml 의 cloth-sim commit hash | `code.cloth_sim_branch.commit: null` (브랜치가 push 되면 채울 것) |

따라서 "기존 acwm-phys 브랜치 코드와 달라진 점"(데이터셋 클래스, 설정, num_hist/frameskip 등)은 **비교할 수 없었다**. 깨진 가정은 다음과 같다.
- 가정 1(미검증): cloth-sim 쪽 `ACWMPhysDataset`/`load_acwm_phys_split` 이 acwm-phys 와 같은 인터페이스다. 이 실험은 acwm-phys commit 의 클래스를 상속한 `exp001.data.ManifestDataset` 으로 데이터를 읽으므로, 사용자 README 가 말하는 형식(`<split>/episode_%05d.mp4`, `metadata.pt` 의 video_path/actions(64,8)/length 등)이 acwm-phys 와 같다는 가정에만 의존한다.
- 가정 2(미검증): cloth_sim 학습도 num_hist=1, frameskip=5, 15 epoch, predictor_lr 5e-5, batch 24 이다. cloth_sim.yaml 을 못 읽었으므로 acwm-phys 의 Cloth Move job 값을 그대로 쓴다(config `train.hydra_overrides`, hydra env 는 `acwm_cloth_move`).
- 가정 3(미검증): 데이터 루트가 `$DATASET_DIR/cloth_sim/<split>/` 이고 split 폴더 안에 `episode_%05d.mp4`, `metadata.pt` 가 있다. split 폴더 이름(`<size>_train/_test`, `ind_train/ind_test` 18개)은 Drive 커넥터로 확인했지만, **폴더 안의 파일 구조는 사용자 README 서술이며 열어서 확인하지 못했다.**
- 데이터 출처 변경(2026-10-31): 사용자가 `hanbin5/cloth-sim`(브랜치와 HF 데이터셋 모두)을 기술적 문제로 당분간 쓸 수 없다고 했다. 그래서 코드는 이 브랜치에 의존하지 않고(acwm-phys 5a640be + `exp001.data` manifest 로더), 데이터는 Google Drive 에서 rclone 으로 받는다(`data.source: drive`, 기본). HF 경로는 `data.source: hf`(또는 환경변수 `EXP001_DATA_SOURCE=hf`)로 남겨 두었다.
- 확인한 사실(acwm-phys commit 5a640be 코드): 브랜치 `load_acwm_phys_slice_train_val` 은 학습 split 을 `"ind_train"` 으로 **하드코딩**한다. 따라서 ind_train 이 아닌 split(대조군의 a080_train 등)을 학습에 쓰려면 이 실험의 manifest 로더(`exp001.data.load_manifest_slice_train_val`)가 필요하다. 이 로더는 hydra override 로 연결한다.

### acwm-phys 코드 (이 실험이 실제로 쓰는 코드)

| 항목 | 값 | 확인 방법 |
|---|---|---|
| acwm-phys 브랜치 | `hanbin5/acwm-phys` (remote `upstream` = https://github.com/jongmin-s-multicore/thread0.git) | `git -C thread0 branch -a` |
| commit | `5a640bed66002eca7bdb31d10b2bde59ad944551` ("README: adding OoD episodes to training ...") | `git ls-remote --heads upstream` 로 원격 head 가 같음을 재확인 |
| 읽기용 checkout | `$ACWM_PHYS_SRC` (기본 `$DINO_WORK/acwm_phys_src`, `setup.sh` 가 `git worktree add --detach` 또는 clone). thread0 main 작업 트리는 건드리지 않는다. | setup.sh |
| num_hist | **브랜치 실제 학습 job 은 `num_hist=1`**. 브랜치 `conf/train.yaml` 기본값과 thread0 main 기본값은 3. acwm-phys 의 `repro/jobs/acwm_phys.txt` 는 Cloth Move 를 `frameskip=5 num_hist=1 training.epochs=15 training.predictor_lr=5e-5 training.batch_size=24` 로 돌린다. | 브랜치 파일 직접 읽음 |
| 모델 | frozen DINOv2 ViT-S/14 패치 특징(14x14x384) + ViT predictor(depth 6, heads 16, MLP 2048, dropout 0.1) + VQ-VAE 디코더(quantize=False). 토큰 404차원(시각 384 + proprio 10 + action 10). 디코더는 `z.detach()` 로 학습되어 predictor 와 분리. | `conf/*`, `models/*` |
| 환경 파일 | 브랜치 `environment.yaml` 은 thread0 main 과 동일. python 3.9.19, torch 2.3.0, hydra-core 1.2.0, decord 0.6.0, opencv 4.6.0.66, scikit-learn 1.5.0 등 | `diff` |
| 환경 conf | `conf/env/acwm_cloth_move.yaml`: action_dims [0,1,2,4,5,6] (3,7 은 상수 1.0), split_ratio 0.9, transform `default_transform(img_size)`. cloth_sim 의 action `[dx1,dy1,dz1,1,dx2,dy2,dz2,1]` (사용자 README)과 같은 차원 구성이다. | 파일 읽음 |

### cloth_sim 데이터 (사용자가 제공한 Drive 폴더 README 의 서술. 이 머신에서 받아 확인한 것이 아님)

- ACWM-Phys Cloth Move 를 PyFleX 로 **재생성**한 것(원본 아님). 같은 action 재생 시 ACWM 과 천 마스크 IoU 0.68. 그리퍼 공 2개(빨강, 파랑)가 천 위 두 모서리를 잡고 구 위로 옮긴다. 카메라 고정.
- 형식: `<split>/episode_%05d.mp4` (512x512, 16fps, 65프레임, H.264), `<split>/metadata.pt` (video_path, actions (64,8) float32, length 65, seed, cloth_dimx, cloth_dimy (입자 수), x0, n_attempts). action 처음 3 step 은 0, action x 15 = 월드 좌표 이동량.
- 크기 9종 (모두 크기마다 train 2,000 / test 100, 총 18,900). 면적비는 프레임 3 의 천 픽셀 수를 ind_train 평균과 비교한 값:

| 이름 | 면적비 | 천 폭x높이 | 입자 수 |
|---|---|---|---|
| a026 | 0.259 | 0.328x0.314 | 54x51 (ACWM smaller 0.26 과 같은 면적비) |
| a040 | 0.399 | 0.402x0.384 | 65x62 |
| a060 | 0.599 | 0.486x0.464 | 79x75 |
| a080 | 0.801 | 0.553x0.528 | 89x86 |
| ind | 1.000 | 0.609x0.582 | 98x94 |
| a125 | 1.249 | 0.670x0.641 | 108x103 |
| a160 | 1.600 | 0.746x0.713 | 120x115 |
| a200 | 2.000 | 0.823x0.786 | 133x127 |
| a248 | 2.479 | 0.905x0.865 | 146x139 (ACWM larger 2.48 과 같은 면적비) |

- split 이름 `<size>_train`, `<size>_test` (ind 는 `ind_train`, `ind_test`). seed 규칙: split 기준값 = 10,000,000 + 1,000,000 x 크기순번(a026=0..a248=8), 테스트 +500,000, episode seed = 기준값 + episode 번호 (생성 쪽 규칙; 이 실험은 생성하지 않으며 config 에 기록만 했다).
- 마스크 규칙 가정(미검증): 천 마스크는 ACWM-Phys 와 같은 HSV 규칙(주황 + 분홍 뒷면)이다. 재생성본의 천 색이 같다는 근거를 확인하지 못했다. `python run.py maskcheck` 가 측정 면적비를 README 면적비와 나란히 기록한다.

학습 분포 설정, 평가 에피소드는 모두 시드와 함께 config.yaml 에 있다.

## 방법

### 0. 재현 확인 단계 (가장 먼저)

재생성본이므로 baseline 이 "학습 크기로 수축"을 보이는지는 보장되지 않는다. 순서:
1. `maskcheck` (CPU, 모델 불필요): *_test 영상에서 프레임 3 의 마스크 면적을 재서 크기별 면적비(ind_test 기준)를 README 면적비와 병기, 입자 수(cloth_dimx*cloth_dimy)와 마스크 면적의 상관(Pearson/Spearman)을 기록한다. 학습 전에 마스크 규칙이 이 렌더러에 맞는지 숫자로 본다. 판정 임계값은 없다.
2. baseline 학습 후 `repro_check` (GPU 추출 + CPU): a026, ind, a248 각 100 에피소드만 추출(추출 캐시 `<run>_repro`, 랜덤 ViT 대조군 제외)하고 면적 표만 계산해 `results/<run>/repro_check.json` 에 쓴다. 항목(수치만): k=12 에서 real / recon / tf1 / ol 평균 면적과 ind 기준 면적, 면적비(모델/실제)와 CI, "모델 면적이 ind 기준 면적에 자기 real 면적보다 가까운 에피소드 비율", `shrink_ratio_tol` 이 채워지면 수축 개수, 9개가 아닌 추출 군집만의 크기-면적 기울기.
3. `run_all.sh` 는 repro_check 를 쓴 뒤 멈춘다(`REPRO_GATE=1` 기본). 사람이 `repro_check.json` 을 본 뒤 `--from extract` 로 전체 추출을 이어 간다. 이 결과가 현상을 보이지 않아도 이후 단계는 실행할 수 있고, 그 사실은 results 에 그대로 남는다.

### 1. 학습 (체크포인트가 없으므로 스크립트가 학습한다)

- 브랜치의 `train.py` 를 **수정 없이** 호출한다 (`python run.py train --model 분포:시드`). 데이터 로더만 hydra override (`env.dataset._target_=exp001.data.load_manifest_slice_train_val`, `+env.dataset.manifest=...`)로 바꿔, 에피소드 목록(manifest)을 받는 로더를 쓴다. 로더는 acwm-phys `ACWMPhysDataset` 의 논리를 그대로 가져온 것이고, 0.9/0.1 분할 seed 42 도 같다.
- 분포는 config `data.datasets.cloth_sim.distributions` (모두 **TODO 사람 확정 필요**):

| 분포 | 구성 (총 2000) | 비고 |
|---|---|---|
| `baseline_single` | ind_train 2000 | 원래 문제 재현. 학습 크기 1개 |
| `mix_adjacent` (후보 F 예시 i) | a080_train 667 + ind_train 666 + a125_train 667 | 면적비 0.801/1.0/1.249. 학습 범위 안쪽에 학습 안 한 크기가 없어 보간 대상 없음, 나머지 6개는 외삽 |
| `mix_wide` (후보 F 예시 ii) | a040_train 667 + ind_train 666 + a160_train 667 | 면적비 0.399/1.0/1.6. a060/a080/a125 는 보간, a026/a200/a248 은 외삽 |

- 통제: 총 에피소드 수(2000), epochs(15), batch 24, predictor_lr 5e-5, 학습 시드(0)를 분포 간에 같게 둔다. `build_manifests` 가 총 수 불일치를 오류로 막는다. 시드를 늘리려면 `train.seeds` 와 `models_to_diagnose` 에 추가하되 비용이 배가 되므로 사람 승인이 필요하다.
- **held-out 규칙**: 학습 manifest 에 없는 크기는 모두 held-out 이다. 역할은 면적비로 자동 분류한다: `trained` / `interp`(학습 크기 면적비 범위 [min, max] 안쪽의 학습 안 한 크기) / `extrap_small` / `extrap_large`. 분류는 `common.size_roles`, 결과는 diagnose.json/probe.json 에 군집별 `role` 로 들어간다.
- **학습 split 자동 제외**: 평가 에피소드는 항상 `*_test` 이고 학습은 `*_train` 만 쓰므로 겹치지 않는다. 그래도 기존 안전장치를 유지해, 학습 manifest 에 들어간 에피소드는 모든 통계에서 자동 제외한다(`in_train` 플래그). 학습에 쓴 크기의 test 에피소드는 `role=trained` 로 통계에 남는다(제외 대상은 학습 에피소드 자체이지 크기가 아니다).
- 승인 게이트: `train.approved_distributions` 또는 환경변수 `EXP001_APPROVED` 에 없는 분포는 `train` 이 거부한다(`--dry-run`/`--smoke` 제외).

### 2. 진단 단계 (후보 D/E) — `extract` (GPU) 후 `diagnose` (CPU)

프레임은 frameskip 5 의 매크로 스텝 0..12 (원본 프레임 0,5,...,60), 행동은 12묶음(각 5스텝, 30차원). 평가 군집은 9개 크기(각 100).

| 단계 | 정의 | 코드 |
|---|---|---|
| (a) Real | 실제 프레임의 천 마스크 면적 | `area_real` |
| (b) encode→decode | `decoder(encoder(real))`, predictor 없음 | `area_recon` |
| (c) teacher-forced 1-step | GT `z_t` 와 `a_t` 로 `z_{t+1}` 예측 후 디코딩 | `area_tf1` |
| (d) open-loop k-step | 프레임 0 과 모든 행동으로 rollout(브랜치 `VWorldModel._rollout` 과 같은 절차)한 k 번째 예측을 디코딩 | `area_ol` |

지표 (`diagnose.json`, 군집별, 에피소드 단위 bootstrap 95% CI):
- 마스크 천 면적(픽셀 마스크 직접 측정), 면적 비 decoded/real (에피소드별 비와 평균 CI).
- 에피소드 단위 수축 빈도: (i) 임계값 없는 정의 — 모델 면적이 "기준 군집(`reference_group`, 기본 ind) 면적(같은 프레임 real 면적의 중앙값)"에 자기 real 면적보다 가까운 에피소드의 비율, (ii) `thresholds.shrink_ratio_tol` 이 채워지면 비율 < tol 인 에피소드 수. tol 은 비어 있다(사람 확정 필요). 대조군(mix) 모델에서도 기준 군집은 ind 로 유지한다(학습 크기가 여러 개일 때 "가장 가까운 학습 크기" 기준 지표는 구현하지 않았다).
- **크기-면적 곡선 (`size_curve`)**: 9개 군집 각각의 real/recon/tf1/ol 평균 면적(k=3,6,12), 프레임 3 에서 측정한 면적비(기준 군집 대비)와 명목 면적비, 군집 평균 log-log 기울기(모델 대 real)와 군집 내 복원추출 cluster bootstrap CI.
- 디코더를 거치지 않는 latent 거리: 예측 latent 와 자기 GT latent 의 패치별 상대 오차(천 패치 대 나머지 패치), 평탄화한 196x384 latent 에서 같은 스텝의 기준 군집 GT 장면 최근접 거리 대 같은 군집 GT 장면 최근접 거리(자기 자신 제외).
- 라벨 보조 검증 (`label_check_particles_vs_mask_area`): log(cloth_dimx*cloth_dimy) 와 log(마스크 면적)의 Pearson/Spearman (프레임 0/3/6/12 와 전체 프레임 풀링). 사용자 README 에 따르면 입자 수는 크기마다 설계상 다르다. 원본 ACWM-Phys 에서는 보이는 크기와 r≈0.02 였고 여기서는 다를 수 있으므로 측정한다. **probe 의 주 라벨은 계속 마스크 면적**이고 이 상관은 라벨로 쓰지 않는다. (군집 내에서는 입자 수가 상수라 군집 내 상관은 정의되지 않는다.)
- Real 프레임에서 큰 천이 시야에 온전한지: 프레임이 정사각 512x512 → 224 로 리사이즈+CenterCrop 이라 crop 손실은 없다(브랜치 `default_transform`). 단 가장자리에 걸리는지는 `frames_sample.npz`(군집별 4개 에피소드의 real/recon/tf1/ol 프레임)로 눈으로 확인해야 한다.

### 3. 크기 probe — `probe` (CPU)

목표값: 해당 프레임의 실제 천 마스크 면적의 log. 입력: 패치 풀링 mean(384) / grid2(2x2 격자, 1536).

latent 종류: `gt`(인코더 출력), `tf1`, `ol`(예측), `h0..h6`(predictor 블록 입력 7곳, teacher-forced; 라벨은 입력 프레임 면적), `rvit`(랜덤 초기화 DINOv2 ViT-S/14), `pix`(14x14 평균 RGB 격자 = raw pixel baseline).

프로토콜 4종 + 상한을 병기한다 (에피소드 단위 분할, 시드 5개). 크기 그룹은 9개:

| 프로토콜 | 정의 |
|---|---|
| (i) `strat_cv` / `logo` | 같은 종류 latent 에 학습·평가. 군집(9개 크기) 층화 5-fold GroupKFold 와 leave-one-group-out(9 fold, 한 크기를 통째로 뺌). LOGO 의 held-out 크기가 9개 중 최소/최대면 `extrap_small/large`, 아니면 `interp` 로 `logo_heldout_type` 에 기록 |
| (ii) `gt_transfer` | 홀드아웃 에피소드를 뺀 GT 인코딩에만 학습, 홀드아웃의 예측 latent(tf1, ol)에 적용 |
| (iii) `upper` | 홀드아웃을 뺀 GT + tf1 + ol (모든 크기 포함)에 학습, 홀드아웃의 예측 latent 에서 평가하는 상한 probe |
| (iv) `heldout_size` (신규) | **월드 모델이 학습한 크기의 GT 인코딩에만** 학습한 probe 를 홀드아웃 에피소드의 gt/tf1/ol 에 적용. 결과를 role(trained / interp / extrap_small / extrap_large)별로 요약. baseline 은 학습 크기가 1개라 probe 학습 라벨이 군집 내 변동뿐이다(정의는 되지만 특성상 한계이며 결과로만 기록) |

모델: ridge(alpha 는 내부 GroupKFold(3)으로 선택) + 작은 MLP(은닉 128, mean 풀링에만).
**R² 단독 보고 금지**: 각 평가마다 전체 R², 군집 내 R²(군집 평균을 뺀 뒤), **9-class 정확도(`acc_cls`)와 인접 클래스 허용 정확도(`acc_cls_pm1`, 클래스는 면적비 순서)**, 예측 대 실제 log 면적 기울기(전체, 군집 평균 간), 군집별 bias 를 에피소드 단위 cluster bootstrap CI(군집 내 복원추출, 1000회)와 함께 낸다. 분류 정확도는 해당 클래스가 학습에 있는 프로토콜(strat_cv, gt_transfer, upper)에서만 정의되고 LOGO/heldout_size 에서는 null 이다. 스텝 구간(early 1-3 / mid 4-8 / late 9-12) 별 값도 낸다.
대조군: label-permute(에피소드 단위 라벨 섞기, 학습 라벨만), 랜덤 ViT, raw pixel 격자, 상수 평균. 선택성 = 진짜 − permute 를 `probe.json` 에 함께 기록한다.
probe 시드 5개(분할, MLP 초기화, permutation, bootstrap). 결과는 시드 평균/표준편차와 CI 경계(시드 평균)로 기록한다.

### 4. 크기 방향 개입 — `intervene` (GPU)

`probe` 가 만든 크기 방향 u(GT 인코딩 평균 풀링 ridge 방향)를 predictor **입력 latent** 에서 건드린다: (1) 모든 패치에서 u 성분 제거, (2) 풀링 투영이 기준 군집(학습 크기) 평균이 되도록 u 방향으로 평행이동(`set_<A>_to_ind`, 나머지 8개 크기), (3) config `intervene.swap_pairs` 의 극단 쌍(a248→a026, a026→a248) 치환. 9개 전부 쌍을 만들면 72 모드라 비용 때문에 쌍을 제한했다(TODO 사람 확정). 크기 군집당 25 에피소드(`intervene.max_episodes_per_group`, 비용 때문에 임의로 둔 값, TODO). teacher-forced 1-step(각 입력 프레임)과 open-loop(첫 latent 만)에서 predictor 출력의 u 방향 응답 이득과 디코딩 면적 변화를 기록한다. 같은 크기의 이동/제거를 무작위 단위 방향 20개에 적용한 분포가 대조다. 해석은 하지 않는다. 한계: u 는 mean 풀링 probe 방향 한 개이므로 정보가 비선형·다차원이면 개입이 약하게 보일 수 있다.

### 5. 중간 크기 생성 단계 제거

이전 버전의 `src/exp001/gen_midsize.py`(외부 생성기 래퍼, `GEN=1`, `run.py gen`, config `gen_midsize`)는 **삭제했다**. 이유: (a) cloth_sim 에 면적비 0.259 ~ 2.479 의 9종이 이미 있어 중간 크기를 따로 만들 필요가 없고, (b) 이전 래퍼는 시뮬레이터를 포함하지 않았으며 원 생성기를 구하지 못해 기본 경로에서 쓰인 적이 없다. 중간 크기를 이용한 보간/외삽 평가는 `heldout_size` 프로토콜과 `role` 분류(1절)로 대체했다. 파일은 git 이력(commit bfddd9d)에 남아 있다. 생성 자체가 필요해지면 사용자 README 가 말하는 `repro/cloth_sim/`(커밋 3e7eb8f, 로컬에서 확인 못 함)를 쓰게 된다.

## 성공 기준

사람 확정 후에 채운다. config.yaml 의 `diagnose.thresholds.*` 와 `success_criteria` 는 `null`(TODO 사람 확정 필요)이고, hypotheses.md 의 예시값(0.9, 0.1 등)은 코드·config 어디에도 복사하지 않았다. 임계값이 비어 있는 동안 코드는 판정하지 않고 수치만 기록한다. 재현 확인(0절)의 통과 기준도 정하지 않았다.

## 실행 방법

사전 조건: NVIDIA GPU 가 있는 리눅스 머신, git, conda, 인터넷(GitHub의 DINOv2 hub 코드 `facebookresearch/dinov2:85a2460...` 와 가중치 다운로드; `models/dino.py` 가 호출), **rclone + Google Drive remote**(데이터 다운로드, 아래 "Drive 에서 데이터 받기"). HF 토큰은 `data.source: hf` 일 때만 필요하다.

```bash
cd experiments/001_size_collapse_diagnosis
bash setup.sh --dry-run          # 먼저 명령만 확인
bash setup.sh                    # 코드 checkout(고정 commit) + conda env(dino_wm). 풀 환경 실패 시 --minimal
# 사전 준비(1회, 아래 "Drive 에서 데이터 받기"): rclone 설치 + `rclone config` 로 drive remote 생성 -> 이름을 config.yaml data.drive.rclone_remote 에
python run.py download --dry-run # 받을 split, 명령, (rclone 이 되면) 원격 파일 수/용량. 아직 아무것도 받지 않는다
export EXP001_APPROVED="baseline_single"     # 사람이 승인한 분포만 (TODO 사람 확정 필요)
bash run_all.sh --dry-run        # 단계별 계획 확인
bash run_all.sh                  # check selftest download maskcheck listing train repro_check 까지 실행하고 게이트에서 멈춤
# results/<run>/repro_check.json 을 확인한 뒤
bash run_all.sh --from extract   # extract diagnose probe intervene aggregate
```

`run_all.sh` 옵션: `--from STAGE`, `--only S1,S2`, `--skip S1,S2`, `--force`, `--dry-run`. 단계: `check selftest download maskcheck listing train repro_check extract diagnose probe intervene aggregate` (+ 옵트인 마지막 `sync`). 환경변수: `CONDA_ENV`, `MODELS="baseline_single:0"`, `EXP001_APPROVED`, `REPRO_GATE`(기본 1), `EXP001_DATASET`(cloth_sim|acwm_phys), `DOWNLOAD_DISTS`, `EXP001_DATA_SOURCE`(drive|hf), `EXP001_RCLONE_REMOTE`, `EXP001_DATA_REMOTE_PATH`, `EXP001_RCLONE_FLAGS`, `EXP001_RCLONE_BIN`, `HF_TOKEN`(source=hf 일 때만), `SYNC`, `SYNC_CKPT`, `SYNC_CACHE`, `DINO_WORK`, `DATASET_DIR`, `DINO_TRAIN`, `EXP001_CACHE`, `ACWM_PHYS_SRC`. 로그는 `logs/<시각>_<단계>.log`. **CPU 단계(probe, selftest)는 `OMP_NUM_THREADS=1~4` 권장** (스레드 과다에서 sklearn 이 매우 느려지는 것을 셀프테스트에서 직접 겪었다: 1스레드 2분 49초 vs 기본 설정에서 3분 이상 진행 중 중단).
건너뜀 기준 파일: download=`results/download_check.json`(`DOWNLOAD_DISTS` 를 지정하면 이 단계는 항상 실행되고, split 단위 건너뜀은 `run.py download` 가 한다), maskcheck=`results/maskcheck.json`(download 뒤에 실행), listing=`manifests/listing.json`, train=`$DINO_TRAIN/outputs/<run>/DONE`(중단 시 `model_latest.pth` 에서 재개), repro_check=`results/<run>/repro_check.json`, extract=`$EXP001_CACHE/<run>/DONE`, diagnose/probe/intervene=`results/<run>/{diagnose,probe,intervene}.json`.

단계별 직접 실행:

```bash
python run.py check
python run.py selftest                       # 합성 데이터 CPU 셀프테스트 (GPU/데이터 불필요)
python run.py download --dry-run            # source=drive: rclone lsf/size 로 받을 목록과 용량만 (받기 없음). source=hf 는 --query-size 가 HF API 크기 조회
python run.py download [--dists mix_wide]    # 대용량. 기본 ind_train + 9개 test, --dists 로 대조군 train split 추가. 끝나면 results/download_check.json
python run.py maskcheck                      # 영상에서 면적비/입자수 상관 (CPU, 모델 불필요)
python run.py listing                        # manifests/ 생성 (listing.json, episode_meta.json, train_<dist>.json)
python run.py train --model baseline_single:0
python run.py repro_check --model baseline_single:0   # 재현 확인
python run.py extract   --model baseline_single:0
python run.py diagnose  --model baseline_single:0
python run.py probe     --model baseline_single:0
python run.py intervene --model baseline_single:0
python run.py aggregate                      # results.json
```

### Drive 에서 데이터 받기 (기본, `data.source: drive`; 대용량이므로 이 개정에서는 실행하지 않았다)

**확인한 사실 (이 개정 세션에서 Drive 커넥터로 직접 확인)**: 데이터 루트 폴더 id `1z4z6WA2G-7UXjbrJtLSWqm6KBpuHrkiI`(소유자는 사용자가 아닌 다른 계정, 사용자가 접근 권한이 있다고 함). 그 아래에 split 폴더 18개(`ind_train`, `ind_test`, `a026_train`...`a248_train`, `a026_test`...`a248_test`), `logs/` 폴더, `README.md`.
**확인하지 못한 가정**: 각 split 폴더 안의 `episode_%05d.mp4`, `metadata.pt` 구조, 에피소드 수(train 2000, test 100), 영상 512x512 65프레임은 사용자 README 의 서술이며 열어 보지 못했다. `download_check.json` 이 받은 뒤에 이를 대조한다.

받는 split: 기본 `ind_train` + 9개 `*_test`. 대조군 분포를 쓸 때만 `--dists mix_wide` 처럼 해당 `*_train` 을 추가한다(`DOWNLOAD_DISTS=mix_wide` 도 가능).

**사전 준비(사용자)**
1. rclone 설치, `rclone config` 로 `drive` 타입 remote 생성(결과 업로드용과 같은 remote 를 재사용할 수 있다). 브라우저 없는 머신은 "Use web browser..." 에 `n` 후 브라우저 있는 PC 에서 `rclone authorize "drive"`.
2. remote **이름**을 `config.yaml` 의 `data.drive.rclone_remote` 에 적는다(비어 있으면 `EXP001_RCLONE_REMOTE`, 그다음 `sync.drive.rclone_remote` 를 쓴다; 현재 모두 비어 있음, TODO). 코드는 rclone 설정/토큰을 읽거나 기록하지 않는다.
3. 폴더 소유자가 다른 계정이므로 접근 방식 중 하나를 고른다. **둘 다 이 머신에서 시험하지 못했다(미검증).**
   - (a) `root_folder_id` (기본, `data.drive.remote_path` 가 비어 있을 때): 폴더 id 를 루트로 지정한다.
     `rclone copy "<remote>,root_folder_id=1z4z6WA2G-7UXjbrJtLSWqm6KBpuHrkiI:<split>" "$DATASET_DIR/cloth_sim/<split>"`
     이 connection-string 문법이 안 되는 rclone 이면 `data.drive.root_folder_style: flag` 로 바꾼다: `rclone copy <remote>:<split> <dest> --drive-root-folder-id 1z4z...`. 대안으로 remote 정의(`rclone config`)의 `root_folder_id` 항목에 같은 id 를 넣어도 되지만, 이 remote 를 결과 업로드와 공유하면 업로드 루트가 바뀌므로 권장하지 않는다(추정).
   - (b) `remote_path`: Drive 웹에서 해당 폴더를 "내 드라이브에 바로가기 추가"하거나 사본을 만든 뒤, 내 드라이브 안의 경로를 `data.drive.remote_path`(또는 `EXP001_DATA_REMOTE_PATH`)에 적는다. 그러면 `rclone copy "<remote>:<remote_path>/<split>" ...` 로 받는다. 바로가기가 rclone 에서 따라가지는지는 미검증(사본이 가장 안전하다고 추정).
4. 확인: `python run.py download --dry-run`

동작
- split 마다 `rclone copy`(재시작에 안전, 중단하면 같은 명령을 다시 실행). 옵션: `--retries 3 --low-level-retries 10`, 추가 플래그는 `EXP001_RCLONE_FLAGS`.
- 건너뜀: 완료 마커 `<split>/.exp001_download_ok` 가 있고 mp4 개수 == `metadata.pt` 항목 수 == 기대치(train 2000, test 100)이면 건너뛴다. 아니면 다시 `rclone copy`(이미 받은 파일은 rclone 이 건너뜀). 받은 뒤 개수가 맞지 않으면 마커를 쓰지 않고 경고한다. `--force` 는 마커를 무시한다.
- `--dry-run`: rclone 이 설치되고 remote 가 있으면 split 별로 `rclone lsf -R --files-only` 와 `rclone size --json` 으로 원격 mp4 수, `metadata.pt` 유무, 용량(GB)과 합계를 보여 준다. 없으면 실행할 명령만 보여 준다. 아무것도 받지 않는다.
- 오류: 출력에서 `403` / `rate limit` / `quota` 패턴이 보이면 대처법(시간을 두고 같은 명령 재실행, 폴더 사본을 내 드라이브에 만들고 접근 방식 b 사용, `EXP001_RCLONE_FLAGS="--transfers 2 --tpslimit 2"`)을 출력하고 종료 코드 3 으로 끝난다. 패턴은 일반적인 것이며 Drive 가 쓰는 정확한 문구는 알지 못한다. rclone 미설치, remote 이름 없음/없는 remote 는 안내문과 종료 코드 2.
- 무결성 점검(받기 직후, `maskcheck` 보다 먼저; `run.py download` 끝에서 자동): split 별 mp4 개수, `metadata.pt` 항목 수, 첫 mp4 의 해상도/프레임 수(OpenCV 가 보고하는 컨테이너 값)를 사용자 README 서술(512x512, 65프레임)과 대조해 `results/download_check.json` 에 기록한다. 불일치는 에러가 아니라 `warnings` 에 기록하고 계속한다. `metadata.pt` 를 읽으려면 torch 가 필요하다(없으면 경고).
- 용량: **확인한 값이 없다.** Drive 커넥터로 폴더를 열어 크기를 재지 않았다. `python run.py download --dry-run` 이 rclone 으로 합계를 보여 준다. 아래는 **추정**(근거 약함): acwm-phys 의 download 스크립트 docstring 은 4개 환경(합 약 6,900 에피소드)이 2.6 GB 라고 적는다 -> 에피소드당 약 0.38 MB. 같다고 가정하면 기본 다운로드(2,000 + 900 = 2,900 에피소드)는 약 1.1 GB, 전체 18,900 에피소드는 약 7 GB. 해상도가 512x512 로 서술되어 몇 배 틀릴 수 있다.
- 테스트한 것: `python run.py selftest` 의 가짜 rclone 실행 파일로 명령 구성(접근 방식 a/b/flag), split 선택, 건너뜀/재개, 불일치 경고, 403 패턴 처리, 설정 누락 안내를 검증했다. 진짜 rclone, Drive 접근, 쿼터 동작은 **미실행/미검증**이다.

### 대안: Hugging Face (`data.source: hf`, 현재 사용 불가라고 사용자가 알림)

`EXP001_DATA_SOURCE=hf` 또는 `data.source: hf` 로 선택한다. `huggingface_hub.snapshot_download("hanbin5/cloth-sim", repo_type="dataset", allow_patterns=[...])`, 같은 split 선택 규칙, 끝나면 같은 무결성 점검. 2026-10-31 에 비인증 `GET https://huggingface.co/api/datasets/hanbin5/cloth-sim?blobs=true` 가 `401` 이었다(원인 확인 못 함). **HF 토큰은 이 경로일 때만 필요하다.**

```bash
export HF_TOKEN=hf_xxx         # READ 토큰. 이 셸에서만. 파일에 적지 말 것  (또는 huggingface-cli login)
EXP001_DATA_SOURCE=hf python run.py download --dry-run --query-size
```

토큰이 없거나 401/403/404 이면 안내문(`src/exp001/hf.py`)과 함께 종료한다. 이 머신에는 토큰이 없어 실제 접근은 시험하지 못했다. 코드는 토큰을 huggingface_hub 호출에만 넘기고 로그/파일/config 에 쓰지 않는다.

### Drive 동기화 (결과를 Google Drive 에 저장, 선택)

사용자 GPU 머신에서 직접 올린다(이 세션의 Drive 커넥터는 쓰지 않는다). 이 머신에는 rclone 도 자격증명도 없어 **실제 업로드는 실행하지 않았고**, `--dry-run`(대상 목록/크기)과 CPU 셀프테스트(비밀 검사, 미설정 시 건너뜀, 임시 폴더로의 마운트 복사와 크기 검증)만 확인했다. rclone 경로(`rclone copy/check/size`)는 **미실행**이다.

설정(1회):
1. rclone 설치 후 `rclone config` 로 `drive` 타입 remote 생성(예: `gdrive`). 브라우저가 없는 머신이면 "Use web browser to automatically authenticate?" 에 `n` 을 답하고, 브라우저가 있는 PC 에서 `rclone authorize "drive"` 를 실행해 출력된 토큰을 헤드리스 머신의 프롬프트에 붙여 넣는다. `rclone lsd gdrive:` 로 확인.
2. `config.yaml` 의 `sync.drive.rclone_remote`(예: `gdrive`)와 `sync.drive.remote_path`(예: `exp001_results`)를 채운다(현재 둘 다 비어 있음, TODO). 환경변수 `EXP001_RCLONE_REMOTE`, `EXP001_REMOTE_PATH` 로도 지정 가능.
3. 대안(rclone 없이): Drive 데스크톱 앱이 마운트한 경로를 `sync.drive.local_mount`(또는 `EXP001_DRIVE_MOUNT`)에 적고 `remote_path` 도 채우면 그 아래로 복사한다(`rclone_remote` 가 비어 있을 때만 사용).

실행:
```bash
python run.py sync --dry-run                 # 업로드 대상 파일과 총 크기만 출력 (아무것도 쓰지 않음)
python run.py sync                           # 업로드
python run.py sync --ckpt --cache            # 대용량 포함 (아래)
SYNC=1 bash run_all.sh --from extract        # run_all 의 마지막 단계로 (옵트인). SYNC_CKPT=1 SYNC_CACHE=1 로 대용량 포함
```
- 대상 폴더: `<remote>:<remote_path>/<YYYYmmdd-HHMMSS>_exp001_<모델>/` (호출마다 새 폴더라 덮어쓰지 않음).
- 기본 업로드: `results.json`, `results/`(`run_meta.json` 포함), `manifests/`, `logs/`, `config.yaml`. 제외: `__pycache__`, `*.pyc`, `*.tmp`, 동기화 자신의 로그/상태 파일(`*_sync.log`, 상태 파일은 검증 후 별도 업로드). 이 폴더 안의 `results/*/size_direction.npz` 는 포함된다(크기 미확인).
- 기본 제외(옵트인): 체크포인트 `$DINO_TRAIN/outputs/<run>/`(`SYNC_CKPT=1` / `--ckpt`, run 당 약 1.7 GB 추정), 추출 캐시 `$EXP001_CACHE/<run>/` 와 `<run>_repro/`(`SYNC_CACHE=1` / `--cache`, 약 5 GB + 1.7 GB 추정). 부담: 개인 Drive 무료 용량 15 GB 에 크게 영향(추정), 업로드 시간은 회선에 따라 다름(예: 업로드 10 MB/s 이면 7 GB 에 약 12분, 계산 예시이며 실측 아님). Drive API 일일 업로드 한도(750 GB/일, rclone 문서의 값으로 기억, 미확인)에는 못 미친다. 큰 파일은 비밀 검사를 하지 않는다(8 MB 초과 파일은 스캔 생략, 바이너리).
- 비밀 검사: 업로드 전 각 파일(8 MB 이하)에서 HF/GitHub 토큰, API 키, 개인키 헤더, `password=`/`secret=` 류 대입, rclone 토큰 JSON, **이메일 주소 형태**, 그리고 환경변수(HF_TOKEN 등)의 실제 값을 찾는다. 걸린 파일은 건너뛰고 경고하며 `sync_status.json` 의 `skipped_secret_like` 에 파일명과 패턴 이름(값 아님)을 남긴다. 단순 패턴이라 우회/오탐이 있을 수 있다.
- 검증: rclone 은 `rclone check --one-way`(해시 비교) + `rclone size`(파일 수/바이트가 로컬 목록과 같은지); 마운트 복사는 크기 비교. 결과는 `results/sync_status.json`(`status`: ok / verification_failed / skipped).
- rclone 이 없거나 remote/경로가 비었으면 업로드를 건너뛰고 설정 안내를 출력하며 종료 코드 0(`status: skipped`). 업로드 실패도 실험을 실패시키지 않는다(`run_all.sh` 는 `|| true`).
- `results/run_meta.json`: 실험 저장소 commit(+dirty 여부), acwm-phys 코드 commit, python/torch/CUDA 버전, GPU 이름, 시드, 모델 목록, 시작/종료 시각(시작은 `run_all.sh` 가 `results/run_times.json` 에 기록, 종료는 sync 호출 시각). torch 가 없으면 null.

### 스모크 테스트 (본 실행 전, 사람이 GPU 머신에서)

```bash
python run.py train --model baseline_single:0 --smoke                    # 60 에피소드, 1 epoch, 실행 이름에 _smoke 접미사
python run.py extract --model baseline_single:0 --suffix _smoke --max-eps 4
python run.py diagnose --model baseline_single:0 --suffix _smoke
python run.py probe --model baseline_single:0 --suffix _smoke            # 에피소드가 적어 CV 가 실패할 수 있다(코드 경로 점검용)
python run.py intervene --model baseline_single:0 --suffix _smoke
```

대조군(mix_adjacent, mix_wide)은 사람 승인 후에만: `DOWNLOAD_DISTS=mix_wide EXP001_APPROVED="baseline_single mix_wide" MODELS="baseline_single:0 mix_wide:0" bash run_all.sh`. 대조군 학습은 baseline 과 같은 비용이 한 번 더 든다.

## 밤샘 실행 (`overnight.sh`)

사용자 결정(2026-10-31): baseline_single 시드 1, 2 학습을 추가하고(학습 시드 3개), 후보 D/E 진단을 시드 0, 1, 2 에 대해 무인으로 진행한다. 시드 0 은 사용자가 이미 학습 중이다. 임계값은 정하지 않았으므로(null) 코드는 판정하지 않고 수치만 기록한다. repro_check 게이트는 끈다: repro_check 결과는 기록만 하고 계속 진행한다(수축이 재현되지 않아도 진단은 계속하며, 그 사실은 `results/<run>/repro_check.json` 과 `results.json` 에 그대로 남는다). 후보 F(mix 분포) 대조군은 선택되지 않았으므로 이 스크립트에 없다(`EXP001_APPROVED` 를 `baseline_single` 로 고정).

### 사용법

```bash
cd experiments/001_size_collapse_diagnosis
bash overnight.sh --dry-run                     # 계획과 환경 점검만 (아무것도 실행/기록하지 않음)
# 방법 1: tmux 안에서
bash overnight.sh
# 방법 2: tmux 가 끊겨도 계속 (nohup)
mkdir -p logs && nohup bash overnight.sh > logs/overnight.log 2>&1 &
```

- 시작 시 `~/exp001_env.sh` 가 있으면 읽고(`DATASET_DIR`, `EXP001_CACHE`, `DINO_TRAIN` 등), 환경을 점검해 로그에 남긴다: 세 경로가 설정됐고 공백/한글이 없는지, `df -h` 와 필요량 추정(시드당 체크포인트 약 1.7 GB + 추출 캐시 약 5 GB, 둘 다 추정) 대비 여유, `nvidia-smi`, torch 버전과 `torch.cuda.get_arch_list()` 에 `sm_120` 이 있는지. 문제가 있으면 경고만 하고 계속한다(중단하지 않음).
- conda: `CONDA_ENV`(기본 `dino_wm`, `/venv/dino_wm` 같은 경로도 가능)를 activate 한다. 이미 환경 안이면 `OVERNIGHT_SKIP_ACTIVATE=1`.
- 환경변수(모두 선택): `TRAIN_PARALLEL`(기본 1), `WAIT_MAX_HOURS`(16), `POLL_SEC`(120), `GRACE_SEC`(180), `HEARTBEAT_MIN`(45), `RETRY_SLEEP`(60), `OVERNIGHT_SEEDS`("0 1 2"). 자세한 뜻은 `overnight.sh` 머리말.
- 아침에 볼 것: `logs/overnight_status.json`(시드별 단계 상태 pending/running/done/failed/skipped, 시작/종료 시각, 재시도 횟수, 마지막 에러 줄), `logs/overnight_summary.txt`(끝났을 때 요약; 시드별로 어느 단계까지 됐는지, 걸린 시간), `results.json`(`runs` 와 시드 간 요약 `across_seeds`). 단계별 로그는 `logs/overnight_<run>_<단계>_try<N>.log`. 상태 파일 `retries` 는 그 단계에서 실패한 시도 횟수다.

### 실행 순서와 우선순위

1. 시드마다 (`TS` = 학습 상태): `$DINO_TRAIN/outputs/<run>/DONE` 이 있으면 학습을 건너뛴다. DONE 이 없고 학습이 이미 돌고 있으면(프로세스 `hydra.run.dir=...<run>` 가 보이거나 run 폴더에서 `HEARTBEAT_MIN` 분 안에 바뀐 파일이 있음) **새로 시작하지 않고 DONE 이 생길 때까지 기다린다**(폴링 `POLL_SEC`, 상한 `WAIT_MAX_HOURS`; 상한을 넘으면 그 시드를 failed 로 표시하고 프로세스는 건드리지 않는다). 프로세스도 파일 변화도 없는데 DONE 도 없으면 `GRACE_SEC` 뒤에 우리가 학습을 시작한다(`model_latest.pth` 에서 재개되는 기존 동작).
2. 시드 0 학습이 외부에서 돌고 있으면 그 시드는 건드리지 않는다(프로세스를 죽이거나 그 폴더에 쓰지 않는다. 읽기만: `DONE` 존재, 파일 수정 시각, `pgrep`).
3. **학습 병렬성(`TRAIN_PARALLEL`, 기본 1)**: 동시에 최대 2개 학습(외부에서 도는 시드 0 도 1개로 센다). 기본 동작: 시드 0 이 학습 중이면 시드 1 을 바로 병렬로 시작하고, 시드 2 는 둘 중 하나가 끝나 슬롯이 비면 시작한다(보통 시드 0 이 먼저 끝난다). `TRAIN_PARALLEL=0` 은 한 번에 1개(시드 0 이 끝난 뒤 시드 1, 그다음 시드 2).
   - 기본값을 2개로 정한 이유(**추정**): 32 GB 에 학습 1개가 약 7 GB 라는 값은 사용자가 준 정보이고 이 머신에서 확인하지 않았다. 2개는 메모리에 여유가 있을 것으로 보이며, GPU 가 1개 학습으로 포화되지 않았다면(데이터로더가 병목이면) 합계 처리량이 늘 수 있다. 3개 동시는 속도를 더 나눠 쓰고 CPU/디스크 경쟁이 커서 피했다. 동시 학습이 실제로 처리량을 늘리는지는 **미측정**이다.
4. **우선순위**: 시드 0 의 DONE 이 보이는 즉시 시드 0 의 `repro_check -> extract -> diagnose -> probe -> intervene -> aggregate+sync` 를 시작한다(다른 시드의 학습과 겹쳐 진행. probe/diagnose 는 CPU). GPU 를 쓰는 단계(repro_check, extract, intervene)는 시드 사이에서 잠금(`flock`)으로 한 번에 하나씩 돌고(다른 시드의 학습과는 GPU 를 같이 쓴다), 이미 시작한 다른 시드의 GPU 단계는 끝날 때까지 기다린 뒤 순서가 온다. 시드 1, 2 도 학습이 끝나는 대로 같은 파이프라인이 시작된다.
5. 시드 하나의 파이프라인이 끝날 때마다 `run.py aggregate` 후 `run.py sync` 를 실행해 일부 결과가 아침 전에 Drive 에 올라가게 한다. sync 실패(rclone 미설정 등)는 실험을 실패시키지 않고 `results/sync_status.json` 에 남는다. 마지막에 한 번 더 aggregate+sync.
6. 시드 0 을 `run_all.sh` 로 시작했다면 학습 후 `run_all.sh` 가 같은 시드의 repro_check 를 직접 진행하다가 게이트에서 멈춘다. 겹쳐서 같은 파일을 쓰지 않도록, `run_all.sh` 나 `run.py <단계> --model baseline_single:<시드>` 프로세스가 보이는 동안은 그 시드의 파이프라인 시작을 기다린다(끝난 결과는 건너뜀).

### 실패 처리

- 각 단계는 같은 단계를 최대 3회 시도한다(CLAUDE.md 규칙, 시도 사이 `RETRY_SLEEP` 초). 3회 실패하면 그 시드의 후속 단계는 `skipped` 로 표시하고, 마지막 에러 줄(로그의 마지막 비어 있지 않은 줄)을 `overnight_status.json` 에 기록한 뒤 **다른 시드는 계속**한다. 각 시드의 학습과 파이프라인은 별도 서브셸/프로세스이므로 한 시드의 에러가 다른 시드의 학습을 죽이지 않는다.
- 학습이 재시도되면 `run.py train` 이 `model_latest.pth` 에서 재개한다(기존 동작; 이 환경에서 학습을 돌려 확인한 것이 아님).
- 스크립트를 Ctrl-C 하거나 종료시키면 **이 스크립트가 시작한** 학습/파이프라인만 종료한다(외부 학습은 건드리지 않는다). 다시 `bash overnight.sh` 를 실행하면 결과가 있는 단계는 건너뛰고 이어서 진행한다.

### 예상 소요 시간 (전부 추정이며 실제 측정이 아니다)

| 구성 | 값 | 근거 |
|---|---|---|
| 학습 시드당 | 약 7시간 | RTX 3090 에서의 측정값(사용자 보고). **RTX 5090 은 미측정 = 추정** |
| 동시 학습 시 속도 | 학습 하나당 느려짐, 정도 미측정 | 최악의 경우 합계 처리량이 그대로여서 2개 동시 = 각 약 2배(추정). GPU 여유가 있으면 그보다 덜 느림. 측정 안 함 |
| repro_check / extract / diagnose / probe / intervene | 시드당 수십 분 ~ 수 시간 | 측정 없음. 합성 데이터 셀프테스트(CPU)의 probe 가 수 분 걸린 것이 유일한 참고이고 실제 데이터 규모와 다르다 |
| 디스크 | 시드당 체크포인트 약 1.7 GB + 추출 캐시 약 5 GB | README 의 기존 추정값. 3시드 약 20 GB(추정) |

**아침까지 모든 시드가 끝난다는 보장은 없다.** 시드 0 이 끝난 시점을 모르므로 시드 1, 2 가 언제 끝나는지도 알 수 없다. 단순 추정으로 시드 2 는 시드 0 이 끝난 뒤 학습이 시작되어 최소 수 시간(병렬이면 더 길게)이 필요하므로, 12시간 안에 시드 2 까지 끝나지 않을 수 있다. 시간이 모자라면 예상 동작: 시드 0 의 결과가 가장 먼저(시드 0 DONE 직후부터) 나오고, 시드 1, 2 는 끝난 단계까지만 `overnight_status.json` 에 `done`/`running` 으로 보인다. `results.json` 의 `across_seeds` 는 그 시점에 결과가 있는 시드만으로 계산되며 `seeds_with_results` 에 어느 시드가 들어갔는지 적힌다. 스크립트를 다시 실행하면 이어서 진행한다.

### 시드 간 요약 (`results.json -> across_seeds`)

`python run.py aggregate` 는 분포별로 repro_check / diagnose / probe / intervene 결과의 숫자 필드(신뢰구간 하한/상한/표본 수와 리스트는 제외)마다 시드별 값, 평균, 표준편차(ddof=1, 시드 1개면 null), 최솟값, 최댓값을 `across_seeds.<분포>.metrics.<경로>` 에 담는다. 값만 기록하며 시드 간 변동의 크기를 판정하거나 해석하지 않는다.

### 주의사항

- **확인한 사실**: 오케스트레이션 로직(대기, 중복 학습 방지, 외부 학습이 사라졌을 때 인수, 대기 상한, 3회 재시도, 시드 격리, 상태 파일 갱신, 순차 모드)은 가짜 `run.py` 와 임시 디렉터리로 `bash tests/test_overnight.sh` 에서 확인했다(실제 학습/GPU/rclone 없이). `bash -n`, `py_compile`, `--dry-run` 도 통과했다.
- **확인하지 못한 것**: 실제 `train.py` 의 프로세스 명령줄에서 `hydra.run.dir=<run 폴더>` 가 `pgrep -f` 로 잡히는지(코드상 `run.py train` 이 이 인자를 붙이지만, 사용자가 시드 0 을 다른 방법으로 시작했다면 다를 수 있다. 그 경우 run 폴더의 파일 변화(`HEARTBEAT_MIN`)가 보조 신호가 된다), `sm_120` 지원 torch 에서의 실제 동작, 동시 학습 시 메모리/속도, 실제 단계별 소요 시간, `model_latest.pth` 재개의 정확한 동작.
- 병렬 학습 시 CPU 데이터로더 workers(브랜치 conf 기본 12, **이 머신에서 확인하지 못한 값**: 사용자 보고/브랜치 설정 기준) 경쟁: 학습 2개면 workers 가 합쳐 약 24개가 되어 CPU 코어 수가 모자라면 데이터 로딩이 병목이 되고 병렬의 이득이 줄거나 사라질 수 있다(**추정**). 코어 수는 이 스크립트가 출력하지 않으니 시작 전에 `nproc` 으로 확인하고, 모자라면 `TRAIN_PARALLEL=0` 이 안전하다. probe 는 `OMP_NUM_THREADS`(기본 4)를 쓰며 학습과 CPU 를 나눠 쓴다.
- 시드 간 병렬 학습으로 시드 0 의 학습 속도가 느려질 수 있다(추정). 시드 0 결과를 가장 먼저 보는 것이 중요하면 `TRAIN_PARALLEL=0` 을 쓰면 시드 0 이 끝날 때까지 시드 1, 2 학습이 시작되지 않는다.
- 이 스크립트는 `--force` 를 쓰지 않는다.

## 파일

| 파일 | 역할 |
|---|---|
| `config.yaml` | 모든 설정과 시드. 임계값은 null(TODO). 데이터셋 선택(`data.dataset`), 9개 크기 사양, 학습 분포 |
| `run.py` | 단계별 CLI |
| `setup.sh`, `run_all.sh` | 환경 구성, 전체 실행 (둘 다 작성만 했고 실제 실행은 하지 않았다. `bash -n` 구문 검사와 `run_all.sh --dry-run` 만 함) |
| `overnight.sh`, `overnight_status.py` | 무인(밤샘) 실행과 상태 파일 갱신 도우미 (위 "밤샘 실행" 절) |
| `tests/test_overnight.sh`, `tests/fake_run.py` | 가짜 run.py 로 하는 오케스트레이션 시험 (GPU/데이터 불필요) |
| `src/exp001/common.py` | 설정 로딩, 데이터셋 블록 선택, 크기 역할(trained/interp/extrap) 분류 |
| `src/exp001/data.py` | listing/manifest 생성, manifest 기반 데이터셋/로더 |
| `src/exp001/maskcheck.py` | (CPU) 영상만으로 마스크 면적비, 입자 수-면적 상관 |
| `src/exp001/extract.py` | (GPU) 체크포인트 로드, (a)-(d) 단계, latent/면적/층별 hidden 저장 |
| `src/exp001/diagnose.py` | (CPU) 면적 표, 크기-면적 곡선, 재현 확인 요약, 수축 지표, latent 거리, 라벨 보조 검증 |
| `src/exp001/probes.py`, `probe_stage.py` | (CPU) 4개 프로토콜의 크기 probe, 대조군, bootstrap |
| `src/exp001/intervene.py` | (GPU) 크기 방향 개입 |
| `src/exp001/aggregate.py` | `results.json` 취합(개입 요약, 시드 간 평균/표준편차/범위 `across_seeds` 포함, 해석 없음) |
| `src/exp001/selftest.py` | 합성 데이터 셀프테스트 |
| `src/exp001/drive.py` | Drive 에서 rclone 으로 데이터 받기(split 선택, 건너뜀/재개, 쿼터 패턴 안내, `--dry-run`)와 받은 뒤 무결성 점검(`results/download_check.json`) |
| `src/exp001/hf.py` | (source=hf 일 때만) HF 토큰 조회(env/login)와 401/403 안내 문구 |
| `src/exp001/sync.py` | Google Drive 업로드(rclone 또는 마운트 복사), 비밀 패턴 검사, 검증, `results/run_meta.json`, `results/sync_status.json` |
| 산출물 | `results.json`(취합), `results/{download_check,maskcheck,run_meta,sync_status}.json`, `results/<run>/{repro_check,diagnose,probe,intervene}.json`, `manifests/`, `logs/`, 대용량 특징은 `$EXP001_CACHE/<run>/` (repro 용은 `<run>_repro/`) |

## 필요한 컴퓨터 사양 (근거 구분)

- 이 작업을 한 머신: WSL2, RAM 7 GB, CPU 8코어, GPU 없음(`nvidia-smi` 없음). **이 머신에서는 학습/추출을 할 수 없다**(확인한 사실). 셀프테스트는 임시 venv(numpy, scikit-learn, pyyaml, opencv-python-headless)에서 돌렸다.
- GPU: 브랜치 README 가 Cloth Move 와 같은 설정(num_hist 1, batch 24)의 학습 peak GPU 메모리를 7.1 GB 로 측정했다(확인한 사실, 출처: acwm-phys 브랜치 README 표). VRAM 12 GB 이상 권장은 추정. 브랜치 predictor 가 attention mask 를 `.to('cuda')` 로 만들기 때문에 **추출/개입에도 CUDA GPU 가 필요**하다(코드에서 확인). cloth_sim 에서의 실측은 없다.
- 학습 시간: 브랜치 README "RTX 3090 에서 학습 1회 약 7시간" (확인한 사실, acwm-phys Cloth Move), 4,125 it/epoch x 15 = 61,875 iterations. 데이터 수·설정이 같으므로 분포 하나당 약 7시간(추정; cloth_sim 에서는 미측정). baseline 만이면 약 7시간, 대조군 하나 추가마다 약 7시간.
- 추출/진단: 에피소드 900개(9 크기 x 100; 기존 약 300개의 3배)로 수십 분~수 시간(추정, 미측정). repro_check 는 300 에피소드(추정 기존의 약 1/3). 개입은 군집당 25 에피소드 x 크기 9 x 모드 10개(remove + 8 + 2) x 무작위 방향 20개라 수 시간(추정, 미측정).
- probe: 합성 데이터 90 에피소드(10 x 9 크기), 시드 1, 풀링 mean, bootstrap 30회, 전 프로토콜(5개)+MLP 를 두 번 돌려 **2분 49초**(측정, 1스레드). 실제 규모(에피소드 900, 시드 5, 풀링 2개, bootstrap 1000, 층별 7종 포함)는 이보다 훨씬 오래 걸린다(추정: 수 시간~하루 이상, 미측정). 필요하면 시드/풀링/MLP 를 줄이는 결정이 필요하다(사람 결정).
- 디스크: conda env 약 9 GB + pip 다운로드 약 4 GB (브랜치 README 값), 데이터 기본 다운로드 약 1 GB (위 추정, 확인 못 함; `download --dry-run` 이 실제 합계를 보여 줌), 학습 체크포인트 run 당 약 1.7 GB (브랜치 README "3개 환경 약 5 GB" 에서 환산, 추정), 추출 캐시 에피소드당 약 5.6 MB(코드의 fp16 shape 에서 계산: gt 13x196x384 + tf1/ol 12프레임씩) x 900 = 약 5.0 GB, repro 캐시 300 x 5.6 MB 약 1.7 GB. 여유 30 GB 이상이면 충분(추정).
- RAM: 학습은 데이터로더 workers 12(브랜치 conf) 사용. latent 거리 계산이 한 프레임 슬라이스를 float32 로 메모리에 올린다(900 x 75,264 x 4 B 약 270 MB, 같은 크기 배열이 여러 개). 16 GB 이상 권장(추정, 미측정).

## 알려진 한계와 미검증 사항

- **cloth-sim 브랜치를 확인하지 못했다**(위 표; 사용자가 당분간 쓸 수 없다고 함. 코드는 이 브랜치에 의존하지 않는다). cloth_sim 의 데이터셋 클래스/설정이 acwm-phys 와 같다는 가정 1-3 이 모두 미검증이다. 브랜치가 push 되면 `datasets/acwm_phys_dset.py`, `conf/env/cloth_sim.yaml` 의 num_hist/frameskip/transform/action_dims 를 이 실험 설정과 대조해야 한다.
- **마스크 규칙이 재생성 렌더러에 맞는지 미검증**이다(`maskcheck` 로 숫자만 확인 가능). 프레임 3 의 측정 면적비가 README 면적비와 크게 다르면 모든 면적 지표가 영향을 받는다. 재생성본의 ACWM 마스크 IoU 가 0.68 이라는 점도 원본과의 비교 가능성을 제한한다.
- 실행하지 못한 코드: `extract.py`, `intervene.py`, `data.py` 의 torch/decord 부분, `maskcheck.py` 의 영상 읽기(cv2.VideoCapture 로 프레임 3 읽기, 모델 입력과 픽셀 단위로 같지 않은 cv2 INTER_AREA 리사이즈), `run.py train/listing` 의 실제 경로, 진짜 rclone 을 쓰는 `download`(가짜 rclone 셀프테스트만 함), `setup.sh`, `run_all.sh` 의 실제 실행 경로. 이 머신에 torch/GPU/데이터가 없어 `py_compile`, `--dry-run`, 합성 데이터 CPU 테스트(manifest 논리와 역할 분류, 마스크 측정, probe 4개 프로토콜, diagnose 표, repro_check 요약)만 했다. 첫 GPU 실행 전 스모크 테스트가 필요하다.
- hydra override `+env.dataset.manifest=...`, `env.dataset._target_=...` 가 브랜치 `train.py` 의 `hydra.utils.call` 에서 기대대로 동작하는지는 코드 읽기로만 확인했다.
- `metadata.pt` 를 `weights_only=True` 로 읽는다. 새 데이터의 metadata 가 tensor/int/str 외의 타입(예: numpy 배열)을 쓰면 실패할 수 있다(미확인).
- 체크포인트에서 동결 인코더가 저장되지 않는 경우(`train_encoder=False`)를 가정해 인코더를 hydra 로 다시 만든다(브랜치 `plan.load_model` 과 같은 방식).
- 풀 `environment.yaml`(217개 pip 패키지, mujoco-py·d4rl·tensorflow 포함)이 `conda env create` 로 설치되는지는 실행하지 않았다. 실패하면 `setup.sh --minimal`.
- 합성 데이터에서 `within-group R²` 가 매우 낮게(음수) 나온다(군집 내 분산이 작음). 실제 데이터에서도 그럴 수 있어 R² 는 다른 지표와 함께 읽어야 한다. 합성 숫자는 결과가 아니다.
- probe 라벨 정의: `h0..h6` 은 입력 프레임 면적, `tf1`/`ol` 은 예측 대상 프레임의 면적.
- 기준 군집은 항상 `ind` 다. 대조군(mix)의 학습 크기가 여러 개일 때의 "가장 가까운 학습 크기로 수축" 지표는 구현하지 않았다.
- 학습 시드는 baseline 1개다(단일 시드 한계 유지). 학습 시드 3 이상은 사람 결정.
- `intervene.swap_pairs`, `intervene.max_episodes_per_group=25`, `save_decoded_frames=4` 는 비용 때문에 정한 임의의 값이며 사람 확정 대상이다.
