# 실험 설명: 001_size_collapse_diagnosis

- 상태: 코드 구현 완료, **실행 전**. 학습·데이터 다운로드·GPU 작업은 하지 않았다. 문법 검사, `--dry-run`, 합성 데이터 CPU 셀프테스트(`python run.py selftest`)만 했다. 이 문서에 실험 결과 수치는 없다.
- 검증할 가설: notes/hypotheses.md 의 후보 D (디코더 층위), 후보 E (인코더/프리딕터 정보 층위).
  - 두 후보 모두 "후보 (미확정, 사람 승인 필요)" 상태다. 이 실험은 확정이 아니라 진단용 첫 단계다.
  - 후보 F (크기 섞어 재학습)는 이 실험의 범위가 아니다. 다만 사용자 결정에 따라 **학습 분포를 config 로 고를 수 있게** 해 두었다(F 의 대조군 코드 지원). F 용 학습(mix 분포)은 사람 승인 전에는 돌리지 않는다.
- 배경: acwm-phys Cloth Move 에서 작은 천으로 학습한 DINO-WM 이 큰 천을 받으면 rollout 에서 학습 크기로 수축한다(브랜치 README 서술, 단일 시드). 원인이 디코더(D), 인코더 정보 소실 또는 predictor 수축(E) 중 어디인지 단계별로 분리한다.

## 이 실험으로 내가 주장하려는 것

확정된 주장이 아니다. 이 실험은 아래 질문에 대한 **측정값**을 만든다. 주장은 사람이 성공 기준을 확정하고, 결과가 나오고, /review-experiment 를 거친 뒤에만 정한다.

1. (D) 천 마스크 면적에서 `decode(encode(real))`(predictor 없음)이 두 OoD 크기에서 real 과 얼마나 다른지, 그리고 teacher-forced 1-step / open-loop k-step 의 디코딩 면적이 어느 단계에서 real 과 벌어지기 시작하는지. 디코더를 거치지 않는 latent 거리로도 같은 질문을 본다.
2. (E) 동결 DINOv2 GT 인코딩에서 크기가 probe 로 복원되는지(선택성은 control 대비), 같은 probe 가 predictor 출력(1-step, k-step)에서 예측 크기 대 실제 크기 기울기가 어떻게 되는지, 그 정보가 predictor 블록 깊이에 따라 어떻게 변하는지, 크기 방향을 건드렸을 때 predictor 출력이 따라 움직이는지(무작위 방향 대비).
3. 위 수치로 "디코더가 원인이다 / 인코더 정보가 없다 / predictor 에서 줄어든다 / 판단 불가" 중 무엇이 데이터와 맞는지는 **결과 후에** 정한다. 지금은 주장하지 않는다.

## 코드 출처 (확인한 사실)

| 항목 | 값 | 확인 방법 |
|---|---|---|
| acwm-phys 브랜치 | `hanbin5/acwm-phys` (remote `upstream` = https://github.com/jongmin-s-multicore/thread0.git) | `git -C thread0 branch -a` |
| commit | `5a640bed66002eca7bdb31d10b2bde59ad944551` ("README: adding OoD episodes to training ...") | `git rev-parse upstream/hanbin5/acwm-phys` |
| 읽기용 checkout | `/home/jaekyungkim/acwm_phys_src` (프로젝트 밖, `git worktree add --detach`). thread0 main 작업 트리는 건드리지 않았다(`git status` 변화 없음). | 작업 중 생성 |
| num_hist | **브랜치 실제 학습 job 은 `num_hist=1`**. 브랜치 `conf/train.yaml` 기본값과 thread0 main `conf/train.yaml` 기본값은 3, main README 는 1 이라고 적혀 있어 서로 다르다. acwm-phys 의 `repro/jobs/acwm_phys.txt` 는 Cloth Move 를 `frameskip=5 num_hist=1 training.epochs=15 training.predictor_lr=5e-5 training.batch_size=24` 로 돌리고, 브랜치 README 표도 Cloth Move 를 num_hist 1, 15 epochs 로 적는다. 이 실험은 그 값을 그대로 쓴다(config.yaml `train.hydra_overrides`). | 브랜치 파일 직접 읽음 |
| 모델 | frozen DINOv2 ViT-S/14 패치 특징(14x14x384) + ViT predictor(depth 6, heads 16, MLP 2048, dropout 0.1) + VQ-VAE 디코더(quantize=False). 토큰 404차원(시각 384 + proprio 10 + action 10). 디코더는 `z.detach()` 로 학습되어 predictor 와 분리. | `conf/*`, `models/*` |
| 환경 파일 | 브랜치 `environment.yaml` 은 thread0 main 과 동일 (diff 확인). python 3.9.19, torch 2.3.0, hydra-core 1.2.0, decord 0.6.0, opencv 4.6.0.66, scikit-learn 1.5.0 등 | `diff` |
| 데이터 | HF `t1an/ACWM-Phys` 의 `deformable/clothmove`, 고정 revision `4017e5f5900a7a8590d4e944b6a84f55df75a0c1`. 브랜치 `repro/setup/download_acwm_phys.py` 에서 확인(스크립트에 pin 되어 있음). 기대 에피소드 수 ind_train 2000 / ind_test 100 / ood_test 100. 레이아웃 `<DATASET_DIR>/acwm_phys/deformable/clothmove/{ind_train,ind_test,ood_test}/*.mp4 + metadata.pt`. | 브랜치 파일 직접 읽음 |
| OoD 크기 | ood_test 는 파일 이름이 `smaller_*` 50개(면적 0.26x), `larger_*` 50개(2.5x). 학습/InD 는 크기 하나. | 브랜치 `SETTINGS.md §8`, `eval_acwm_phys.py` |
| 천 마스크 | 주황(hue 8-16, S>120) + 분홍 뒷면(hue 158-170, S>60), V>80. 224x224 프레임 기준 픽셀 수. `cloth_dimx/y` 메타는 r≈0.02 라 쓰지 않는다. | 브랜치 `eval_acwm_phys_quant.py` 의 규칙을 그대로 옮김 |

학습 분포 설정, 평가 에피소드는 모두 시드와 함께 config.yaml 에 있다.

## 방법

### 1. 학습 (체크포인트가 없으므로 스크립트가 학습한다)

- 브랜치의 `train.py` 를 **수정 없이** 호출한다 (`python run.py train --model 분포:시드`). 데이터 로더만 hydra override (`env.dataset._target_=exp001.data.load_manifest_slice_train_val`, `+env.dataset.manifest=...`)로 바꿔, 에피소드 목록(manifest)을 받는 로더를 쓴다. 로더는 브랜치 `ACWMPhysDataset` 의 논리를 그대로 가져온 것이고, 0.9/0.1 분할 seed 42 도 같다.
- 분포는 config.yaml `train.distributions`.
  - `baseline_single`: ind_train 2000개 전부, 원래 순서. (브랜치 기본 학습과 같은 에피소드. 학습 크기는 하나.)
  - `mix_ood3`: ind_train 앞 1950개 + ood_test smaller 25 + larger 25 (정렬한 파일 이름의 **뒤쪽** 25개씩). 크기 3단계(1x, 0.26x, 2.5x). 총 에피소드 2000으로 baseline 과 같다. 구성은 예시이며 "사람 확정 필요".
  - 생성된 중간 크기 에피소드를 넣으려면 `{source: "gen:a0p5", group: ..., n: ..., take: first}` 항목을 추가한다(아래 4절).
- 통제: 에피소드 총수(2000), epochs(15), batch 24, predictor_lr 5e-5, 학습 시드(0)를 분포 간에 같게 둔다. 시드를 늘리려면 `train.seeds` 와 `models_to_diagnose` 에 추가하되 비용이 배가 되므로 사람 승인이 필요하다.
- 평가 에피소드의 일관성: 평가용 ood 에피소드는 정렬 기준 앞 25개씩이 "공통 부분집합"이고 mix 학습은 뒤 25개를 쓴다. 각 모델은 **자기 학습에 쓴 에피소드를 모든 통계에서 자동 제외**한다(`in_train` 플래그).

### 2. 진단 단계 (후보 D/E) — `extract` (GPU) 후 `diagnose` (CPU)

프레임은 frameskip 5 의 매크로 스텝 0..12 (원본 프레임 0,5,...,60), 행동은 12묶음(각 5스텝, 30차원).

| 단계 | 정의 | 코드 |
|---|---|---|
| (a) Real | 실제 프레임의 천 마스크 면적 | `area_real` |
| (b) encode→decode | `decoder(encoder(real))`, predictor 없음 | `area_recon` |
| (c) teacher-forced 1-step | GT `z_t` 와 `a_t` 로 `z_{t+1}` 예측 후 디코딩 | `area_tf1` |
| (d) open-loop k-step | 프레임 0 과 모든 행동으로 rollout(브랜치 `VWorldModel._rollout` 과 같은 절차)한 k 번째 예측을 디코딩 | `area_ol` |

지표 (`diagnose.json`, 군집별, 에피소드 단위 bootstrap 95% CI):
- 마스크 천 면적(픽셀 마스크 직접 측정), 면적 비 decoded/real (에피소드별 비와 평균 CI).
- 에피소드 단위 수축 빈도: (i) 임계값 없는 정의 — 모델 면적이 "InD 기준 면적(InD 에피소드 real 면적의 같은 프레임 중앙값)"에 자기 real 면적보다 가까운 에피소드의 비율, (ii) `thresholds.shrink_ratio_tol` 이 채워지면 비율 < tol 인 에피소드 수. tol 은 비어 있다(사람 확정 필요).
- 디코더를 거치지 않는 latent 거리: 예측 latent 와 자기 GT latent 의 패치별 상대 오차(천 패치 대 나머지 패치), 평탄화한 196x384 latent 에서 같은 스텝의 **InD 크기 GT 장면 최근접 거리 대 같은 군집 GT 장면 최근접 거리**(자기 자신 제외).
- Real 프레임에서 큰 천이 시야에 온전한지: 프레임이 정사각 512x512 → 224 로 리사이즈+CenterCrop 이라 crop 손실은 없다(브랜치 `default_transform`). 단 실제로 가장자리에 걸리는지는 `frames_sample.npz`(군집별 8개 에피소드의 real/recon/tf1/ol 프레임)로 눈으로 확인해야 한다.

### 3. 크기 probe — `probe` (CPU)

목표값: 해당 프레임의 실제 천 마스크 면적의 log. 입력: 패치 풀링 mean(384) / grid2(2x2 격자, 1536).

latent 종류: `gt`(인코더 출력), `tf1`, `ol`(예측), `h0..h6`(predictor 블록 입력 7곳, teacher-forced; 라벨은 입력 프레임 면적), `rvit`(랜덤 초기화 DINOv2 ViT-S/14), `pix`(14x14 평균 RGB 격자 = raw pixel baseline).

세 프로토콜을 모두 구현해 병기한다 (에피소드 단위 분할, 시드 5개):

| 프로토콜 | 정의 |
|---|---|
| (i) `strat_cv` / `logo` | 같은 종류 latent 에 학습·평가. 군집(크기 그룹) 층화 5-fold GroupKFold 와 leave-one-group-out |
| (ii) `gt_transfer` | 홀드아웃 에피소드를 뺀 GT 인코딩에만 학습, 홀드아웃의 예측 latent(tf1, ol)에 적용 |
| (iii) `upper` | 홀드아웃을 뺀 GT + tf1 + ol (모든 크기 포함)에 학습, 홀드아웃의 예측 latent 에서 평가하는 상한 probe |

모델: ridge(alpha 는 내부 GroupKFold(3)으로 선택) + 작은 MLP(은닉 128, mean 풀링에만).
**R² 단독 보고 금지**: 각 평가마다 전체 R², 군집 내 R²(군집 평균을 뺀 뒤), 3-way 분류 정확도(smaller/ind/larger; LOGO 에서는 정의 불가라 null), 예측 대 실제 log 면적 기울기(전체, 군집 평균 간), 군집별 bias 를 에피소드 단위 cluster bootstrap CI(군집 내 복원추출, 1000회)와 함께 낸다. 스텝 구간(early 1-3 / mid 4-8 / late 9-12) 별 값도 낸다.
대조군: label-permute(에피소드 단위 라벨 섞기, 학습 라벨만), 랜덤 ViT, raw pixel 격자, 상수 평균. 선택성 = 진짜 − permute 를 `probe.json` 에 함께 기록한다.
probe 시드 5개(분할, MLP 초기화, permutation, bootstrap). 결과는 시드 평균/표준편차와 CI 경계(시드 평균)로 기록한다.

### 4. 크기 방향 개입 — `intervene` (GPU)

`probe` 가 만든 크기 방향 u(GT 인코딩 평균 풀링 ridge 방향)를 predictor **입력 latent** 에서 건드린다: (1) 모든 패치에서 u 성분 제거, (2) 풀링 투영이 InD 군집 평균(또는 다른 군집 평균)이 되도록 u 방향으로 평행이동. teacher-forced 1-step(각 입력 프레임)과 open-loop(첫 latent 만)에서 predictor 출력의 u 방향 응답 이득과 디코딩 면적 변화를 기록한다. 같은 크기의 이동/제거를 무작위 단위 방향 20개에 적용한 분포가 대조다. 해석은 하지 않는다. 한계: u 는 mean 풀링 probe 방향 한 개이므로 정보가 비선형·다차원이면 개입이 약하게 보일 수 있다(방법의 한계, 결과 해석 때 고려).

### 5. 중간 크기 에피소드 생성 (옵션, 기본 비활성) — 조사 결과

사용자 결정은 "시뮬레이터로 새로 생성하는 스크립트 포함, 불확실하면 옵션으로 분리"였다. 조사 결과:

- **확인한 것**: 브랜치 README 는 ACWM-Phys 데이터셋에 시뮬레이터가 없어서 planning 을 하지 않았다고 적는다(요지 인용). 브랜치에 있는 cloth 코드는 DINO-WM 의 `env/deformable_env`(PyFleX FlexEnv, `scenes.py::cloth_scene`, cloth_size 70x70 고정) 뿐이다. 이것은 데이터셋의 Cloth Move(그리퍼 공 2개로 천 모서리를 잡아 구 위에 올림, 512x512, 16fps)와 다른 장면이다. PyFleX 는 `install_pyflex.sh` 기준 docker(`xingyu/softgym`, CUDA 9.2 이미지) + NVIDIA container runtime 이 필요하고 빌드 난이도가 높다. PyFleX 로 만든 천은 렌더러·색·카메라·과제가 달라 같은 분포에서 크기만 바꾼 데이터가 되지 않는다(교란이 크다).
- **확인하지 못한 것**: ACWM-Phys 데이터를 만든 원 생성기(논문 arXiv:2605.08567 을 가리키는 브랜치 README 외에 코드 위치)가 어디 있는지, 의존성이 무엇인지, `cloth_dimx/y` 같은 크기 인자로 크기를 바꿀 수 있는지. 논문/원 저장소는 열람하지 않았다.
- **결정**: `src/exp001/gen_midsize.py` 는 시뮬레이터를 포함하지 않는다. 사용자가 가진 생성기를 부르는 명령 템플릿(`ACWM_PHYS_GENERATOR_CMD`, 자리표시자 `{area_scale} {n} {seed} {out_dir}`)을 받아 실행하고, 출력 형식(metadata.pt, 65프레임, 512x512)을 검증하고, `manifests/gen_status.json` 에 사실을 기록한다. 환경변수가 없으면 이유를 기록하고 코드 3 으로 끝난다(`run.py gen` 은 이를 오류로 보지 않는다). 생성 결과의 실제 면적 배율(예: 0.5x 요청 → 프레임 3 면적 비)은 검증 보고에 측정해 남기며 허용 오차는 TODO 다. 면적 배율 vs 선형 배율 정의도 생성기 확인 후 정한다.
- 따라서 이 실험의 기본 경로(`run_all.sh`)는 중간 크기 없이 3점(0.26x, 1x, 2.5x)만으로 진행한다. 이 경우 크기 곡선(단조성 이상)은 볼 수 없고, 군집당 50 에피소드의 한계(stats 분석가 계산: 검출 가능한 최소 |r| 0.387)가 그대로 적용된다.

## 성공 기준

사람 확정 후에 채운다. config.yaml 의 `diagnose.thresholds.*` 와 `success_criteria` 는 `null`(TODO 사람 확정 필요)이고, hypotheses.md 의 예시값(0.9, 0.1 등)은 코드·config 어디에도 복사하지 않았다. 임계값이 비어 있는 동안 코드는 판정하지 않고 수치만 기록한다.

## 실행 방법

사전 조건: NVIDIA GPU 가 있는 리눅스 머신, git, conda, 인터넷(HF 데이터, GitHub의 DINOv2 hub 코드 `facebookresearch/dinov2:85a2460...` 와 가중치 다운로드; `models/dino.py` 가 호출).

```bash
cd experiments/001_size_collapse_diagnosis
bash setup.sh --dry-run          # 먼저 명령만 확인
bash setup.sh                    # 코드 checkout(고정 commit) + conda env(dino_wm). 풀 환경 실패 시 --minimal
bash run_all.sh --dry-run        # 단계별 계획 확인
bash run_all.sh                  # 전체. 끝난 단계는 결과 파일이 있으면 건너뜀. 실패하면 같은 명령으로 재개
```

`run_all.sh` 옵션: `--from STAGE`, `--only S1,S2`, `--skip S1,S2`, `--force`, `--dry-run`. 단계: `check selftest download gen listing train extract diagnose probe intervene aggregate`. 환경변수: `CONDA_ENV`, `MODELS="baseline_single:0 mix_ood3:0"`, `GEN=1`, `DINO_WORK`, `DATASET_DIR`, `DINO_TRAIN`, `EXP001_CACHE`, `ACWM_PHYS_SRC`. 로그는 `logs/<시각>_<단계>.log`.
각 단계의 건너뜀 기준 파일: download=`ood_test/metadata.pt`, listing=`manifests/listing.json`, train=`$DINO_TRAIN/outputs/<run>/DONE`(중단 시 `model_latest.pth` 에서 재개), extract=`$EXP001_CACHE/<run>/DONE`, diagnose/probe/intervene=`results/<run>/{diagnose,probe,intervene}.json`.

단계별 직접 실행:

```bash
python run.py check
python run.py selftest                       # 합성 데이터 CPU 셀프테스트 (GPU/데이터 불필요)
python run.py download                       # 대용량: cloth_move 만 받음
python run.py listing                        # manifests/ 생성
python run.py train --model baseline_single:0
python run.py extract   --model baseline_single:0
python run.py diagnose  --model baseline_single:0
python run.py probe     --model baseline_single:0
python run.py intervene --model baseline_single:0
python run.py aggregate                      # results.json
```

### 스모크 테스트 (본 실행 전, 사람이 GPU 머신에서)

```bash
python run.py train --model baseline_single:0 --smoke                    # 60 에피소드, 1 epoch, 실행 이름에 _smoke 접미사
python run.py extract --model baseline_single:0 --suffix _smoke --max-eps 4
python run.py diagnose --model baseline_single:0 --suffix _smoke
python run.py probe --model baseline_single:0 --suffix _smoke            # 에피소드가 적어 CV 가 실패할 수 있다(코드 경로 점검용)
python run.py intervene --model baseline_single:0 --suffix _smoke
```

mix 분포(후보 F 대조군)는 `MODELS="baseline_single:0 mix_ood3:0" bash run_all.sh` 로 돌리되 **사람 승인 전에는 하지 않는다**. mix 학습은 baseline 과 같은 비용이 한 번 더 든다.

## 파일

| 파일 | 역할 |
|---|---|
| `config.yaml` | 모든 설정과 시드. 임계값은 null(TODO) |
| `run.py` | 단계별 CLI |
| `setup.sh`, `run_all.sh` | 환경 구성, 전체 실행 (둘 다 작성만 했고 실제 실행은 하지 않았다) |
| `src/exp001/data.py` | manifest 생성, manifest 기반 데이터셋/로더 |
| `src/exp001/extract.py` | (GPU) 체크포인트 로드, (a)-(d) 단계, latent/면적/층별 hidden 저장 |
| `src/exp001/diagnose.py` | (CPU) 면적 표, 수축 지표, latent 거리 |
| `src/exp001/probes.py`, `probe_stage.py` | (CPU) 세 프로토콜의 크기 probe, 대조군, bootstrap |
| `src/exp001/intervene.py` | (GPU) 크기 방향 개입 |
| `src/exp001/gen_midsize.py` | 옵션: 외부 생성기 래퍼와 검증 |
| `src/exp001/aggregate.py` | `results.json` 취합(개입 요약 포함, 해석 없음) |
| `src/exp001/selftest.py` | 합성 데이터 셀프테스트 |
| 산출물 | `results.json`(취합), `results/<run>/{diagnose,probe,intervene}.json`, `manifests/`, `logs/`, 대용량 특징은 `$EXP001_CACHE/<run>/` |

## 필요한 컴퓨터 사양 (근거 구분)

- 이 작업을 한 머신: WSL2, RAM 7 GB, CPU 8코어, GPU 없음(`nvidia-smi` 없음). **이 머신에서는 학습/추출을 할 수 없다**(확인한 사실).
- GPU: 브랜치 README 가 Cloth Move 와 같은 설정(num_hist 1, batch 24)의 학습 peak GPU 메모리를 7.1 GB 로 측정했다(확인한 사실, 출처: 브랜치 README 표). VRAM 12 GB 이상 권장은 추정. 브랜치 predictor 가 attention mask 를 `.to('cuda')` 로 만들기 때문에 **추출/개입에도 CUDA GPU 가 필요**하다(코드에서 확인).
- 학습 시간: 브랜치 README "RTX 3090 에서 학습 1회 약 7시간" (확인한 사실, 브랜치 README), 브랜치 job 파일 기준 4,125 it/epoch x 15 = 61,875 iterations. 같은 데이터 수·설정이므로 분포 하나당 약 7시간(추정: 데이터 로더/디스크 속도에 따라 다름). baseline 만이면 약 7시간, mix 까지면 약 14시간.
- 추출/진단: 에피소드 약 300개(ind_test 100 + ood 100 + 생성분)로 수 분~수십 분(추정, 미측정). 개입은 무작위 방향 20개 x 모드 수 때문에 수십 분(추정).
- probe: 합성 데이터 56 에피소드, 시드 1, 풀링 mean, 전 프로토콜+MLP 에서 약 28초(측정, 8코어 CPU 1스레드). 실제 규모(에피소드 ~200-300, 시드 5, 풀링 2개, 층별 7종 포함)는 수십 분~수 시간(추정, 미측정). `OMP_NUM_THREADS=1~4` 권장(스레드 과다 시 sklearn 이 오히려 느려지는 현상을 합성 데이터에서 겪었다).
- 디스크: conda env 약 9 GB + pip 다운로드 약 4 GB (브랜치 README 가 같은 environment.yaml 로 적은 값), 데이터는 4개 환경 합쳐 2.6 GB(브랜치 README; cloth_move 만은 그보다 작을 것 - 추정), 학습 체크포인트 run 당 약 1.7 GB (브랜치 README "3개 환경 약 5 GB" 에서 환산, 추정), 추출 캐시 에피소드당 약 5.6 MB x 300 = 약 1.7 GB (코드의 fp16 shape 에서 계산). 여유 30 GB 이상이면 충분(추정).
- RAM: 학습은 데이터로더 workers 12(브랜치 conf) 사용. 16 GB 이상 권장(추정, 미측정).

## 알려진 한계와 미검증 사항

- 실행하지 못한 코드: `extract.py`, `intervene.py`, `data.py` 의 torch/decord 부분, `run.py train` 의 hydra override, `setup.sh`, `run_all.sh` 의 실제 실행 경로. 이 머신에 torch/GPU 가 없어 문법 검사(`py_compile`, pyflakes)와 `--dry-run`, 합성 데이터 CPU 테스트(manifest 논리, 마스크 측정, probe 프로토콜, diagnose 표)만 했다. 첫 GPU 실행 전 스모크 테스트가 필요하다.
- hydra override `+env.dataset.manifest=...`, `env.dataset._target_=...` 가 브랜치 `train.py` 의 `hydra.utils.call` 에서 기대대로 동작하는지는 코드 읽기로만 확인했다.
- 체크포인트에서 동결 인코더가 저장되지 않는 경우(`train_encoder=False`)를 가정해 인코더를 hydra 로 다시 만든다(브랜치 `plan.load_model` 과 같은 방식).
- 풀 `environment.yaml`(217개 pip 패키지, mujoco-py·d4rl·tensorflow 포함)이 `conda env create` 로 설치되는지는 실행하지 않았다. 브랜치 `install_env.sh` 는 micromamba + uv 로 설치했고 pip 경로는 끝까지 돌려 보지 않았다고 적혀 있다(브랜치 파일). 실패하면 `setup.sh --minimal`.
- 합성 데이터에서 `within-group R²` 가 매우 낮게 나온다(군집 내 분산이 작음). 실제 데이터에서도 그럴 수 있어 R² 는 다른 지표와 함께 읽어야 한다(R² 단독 금지 규칙의 이유).
- probe 라벨 정의: `h0..h6` 은 입력 프레임 면적, `tf1`/`ol` 은 예측 대상 프레임의 면적. 정의를 바꾸면 결과가 달라진다.
- 학습 시드는 baseline 1개다(단일 시드 한계 유지). 학습 시드 3 이상은 사람 결정.
- 새 크기 곡선(보간)은 중간 크기 생성기가 없으면 볼 수 없다(위 5절).
