# 007 decoder collapse autopsy

- 지위: 가설 검증이 아니라 진단 질문이다 (notes/hypotheses.md "007 진단 질문", 2026-11-07, 사용자 승인). 재학습과 GPU 사용은 승인되지 않았다. CPU 에서 기존 체크포인트만 읽는다.
- 검증할 질문: 003 의 exp003_mix_adjacent_s0, exp003_single_a026_s1 에서 val_img_mse_reconstructed 가 epoch 2 -> 3 사이에 급등(0.0030 -> 0.0787, 0.0015 -> 0.0333)하고 디코딩 프레임이 입력과 무관해진다. epoch 2, 3(, 4) 체크포인트에서 사전 기록한 5개 판단 기준 중 무엇이 관측되는가.
- 방법 (붕괴 run 과 대조 run 에 같은 코드):
  - A 코드 사실 (commit 5a640bed, 파일과 줄 번호): results.json `A_code_facts`.
  - B 디코더, predictor, action_encoder, proprio_encoder 의 파라미터별 norm/mean/std/NaN/Inf/0 비율, epoch 2->3 변화량과 상대 변화.
  - C decoder_optimizer(및 predictor, action_encoder optimizer)의 step, lr, exp_avg, exp_avg_sq 요약.
  - D 같은 입력 프레임을 DINOv2(dinov2_vits14, 학습 코드의 models/dino.py 그대로, torch hub 고정 commit, CPU)로 인코딩해 디코더에 넣고, 층별 출력의 입력 간 std, ReLU 입력 <= 0 비율을 hook 으로 측정. 출력 이미지의 입력 간 std, 실제 프레임 대비 MSE([-1,1] 스케일, 학습 손실과 같은 스케일). 복원 시트 PNG (`figures/`).
  - E 같은 입력의 DINO 잠재를 predictor 에 넣고(디코더 없이) 출력의 입력 간 std 비교.
  - 입력 집합 2종: `common` = exp003_mix_adjacent_s0 의 frames_sample.npz (모든 체크포인트에 같은 프레임), `own:<run>` = 각 run 자신의 frames_sample.npz. 각 집합은 9개 크기 x 에피소드 2 x 시점 3(t=0,6,12) = 54 프레임.
  - E 의 한계: frames_sample 에 행동, proprio 가 없고 데이터를 내려받지 않으므로 predictor 입력의 proprio 와 action 은 모든 입력에서 0 벡터(각 encoder 를 통과)로 고정했다. 입력 간 차이는 프레임에서만 온다.
- 체크포인트 (Drive 휴지통 읽기 전용, `--drive-trashed-only`, 로컬 방향 copy/lsf 만 사용):
  - 붕괴: exp003_mix_adjacent_s0 model_2, model_3; exp003_single_a026_s1 model_2, model_3, model_4
  - 대조: exp003_mix_adjacent_s1, exp003_mix_adjacent_s2, exp003_single_a026_s2 의 model_2, model_3
  - 다운로드 후 `/tmp` 가 tmpfs 3.8 GB 라 원본(368 MB)은 predictor, decoder, action/proprio encoder 모듈과 optimizer 요약만 담은 slim 파일(약 125 MB)로 바꾸고 원본 로컬 사본은 삭제했다. Drive 파일은 건드리지 않았다. 원본 sha256 은 results.json 의 `ckpts.<run>/model_<e>.drive_meta.sha256` 에 있다.
  - 체크포인트, 코드 사본(git archive of 5a640bed), torch hub 캐시는 저장소 밖 scratchpad (`config.yaml paths`)에 있다.
- 판단 기준 (결과를 보기 전에 notes/hypotheses.md 에 기록): (1) NaN/Inf 있음 (2) 입력 의존성이 처음 사라지는 층이 특정됨 (3) 활성화가 전부 비활성인 층이 특정됨 (4) epoch 3 가중치 변화가 대조 대비 10배 이상인 층이 특정됨 (5) 어느 것도 해당 없음.
  - 조작적 정의: (2) 층 출력의 입력 간 std 가 대조 3개 run 평균의 1/100 미만. (3) ReLU 입력이 모든 입력, 위치에서 <= 0. (4) 붕괴 run 의 ||W3-W2||/||W2|| >= 대조 3개 run 최댓값의 10배 (같은 분포 짝만 쓴 값도 병기).
  - 체크포인트로 구분할 수 없는 것: epoch 3 도중 정확한 시점, 단일 큰 gradient 여부 (→ "체크포인트로 구분 불가").
- 성공 기준: 없음(진단). 5개 기준 각각에 대해 해당/비해당과 대조 run 값을 results.json `criteria` 에 기록하는 것이 산출물이다.
- 실행 방법 (conda 환경 `lab07`: python 3.9, torch 2.0.1+cpu, torchvision 0.15.2+cpu, numpy 1.26.4, einops, matplotlib, pyyaml):
  ```
  conda activate lab07   # 또는 .../envs/lab07/bin/python -I
  python run.py fetch            # rclone 인증이 살아 있어야 함. 대화형 로그인은 하지 않는다
  python run.py analyze --smoke  # smoke_results.json (체크포인트 1개, 프레임 6장)
  python run.py analyze          # results.json, figures/
  ```
- 이 실험으로 내가 주장하려는 것: 위 5개 기준 각각이 해당/비해당이라는 관측 사실과 수치뿐이다. 붕괴의 원인은 주장하지 않는다. 원인 해석은 /review-experiment 를 거친다.

## 실행 기록 (사실만)
- 2026-11-07 KST 실행. fetch 후 analyze 1691 초(CPU), `python run.py summarize` 로 `criteria_table` 추가. 결과 요약은 notes/log.md 2026-11-07 / 007 항목.
- 5개 run 의 frames_sample.npz 의 `real` 배열이 모두 동일해서 입력 집합 `own:<run>` 과 `common` 의 결과가 같다.
- 첫 fetch 가 lsf 형식 문자열 오타로 실패해 고친 뒤 재실행했다. smoke 로 만든 slim 1개는 메타가 없어 지우고 다시 만들었다.
- git commit 하지 않았다 (사용자 요청 시에만).
