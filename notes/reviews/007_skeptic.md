# 007_decoder_collapse_autopsy 회의론자 리뷰

검토 대상: experiments/007_decoder_collapse_autopsy/{README.md, config.yaml, run.py, results.json, figures/}, notes/hypotheses.md:349-357, notes/log.md:166-178, 고정 commit 코드 사본 code5a, experiments/003_train_dist_dependence/colab/train_resume.patch, 003 의 analysis/decoder_recon_by_epoch.json, analysis/recon_18runs.csv.
다른 분석가의 007 리뷰는 읽지 않았다.

표기: [확인] 파일이나 수치를 직접 읽어 확인. [계산] results.json 값으로 내가 손으로 계산한 값(results.json 에 없음. 코드 실행 없음, 인용 전 재계산 필요). [추정] 논리적 추론이며 검증되지 않음. [미확인] 확인하지 못함.

## 결론 요약

잠정 해석을 네 문장으로 나눠 평가한다.

| 문장 | 판정 | 핵심 근거 |
|---|---|---|
| epoch 3 중 아주 큰 gradient | 일시적 큰 gradient 는 강하게 지지됨. 원인인지 붕괴 중 결과인지는 미정 | exp_avg_sq 순수 감쇠(중요 2) |
| ReLU 가 사실상 죽어 회복하지 못함 | mix_adjacent_s0 는 강하게 지지됨. single_a026_s1 은 54프레임에서만 확인, 상태도 다름. 죽음은 회복 실패의 설명이지 붕괴 시작의 설명이 아님 | 치명 1, 중요 1, 중요 6 |
| predictor 는 멀쩡 | 과장. "손실 경로 분리, z 손실 큰 차이 없음"까지만 말할 수 있음 | 치명 2 |
| 디코더만 재학습하면 측정을 살림 | 근거 없음. 코드 경로 없음, 비교 가능성 미검증, 살려도 003 판정은 불변(log.md:163) | 치명 3 |

측정 버그는 결론을 뒤집을 만한 것을 찾지 못했다. 문제는 측정보다 해석에 있다.

## 치명

### 치명 1. 'upsample_b.blocks.7' 은 원인 층이 아니라 증상이 처음 드러나는 층일 가능성이 높다

- [확인] 판단 기준 (2)는 입력 간 std 의 절대값만 본다 (run.py:528-534). 정의상 "포화가 처음 완성된 곳"을 찾으며, 변화가 시작된 곳을 찾지 않는다.
- [확인] blocks.7 앞의 층은 죽지 않고 커졌다. mix_adjacent_s0 의 rms, epoch 2 -> 3 (results.json 의 model_2, model_3 D.common):
  - blocks.0: 7.53 -> 29.28
  - blocks.4: 8.29 -> 87.06
  - blocks.5: 0.93 -> 35.88
  - blocks.6: 0.96 -> 114.77
  - blocks.7: 0.27 -> 0.0
- [확인] blocks.5 입력 <= 0 비율은 대조(0.965~0.966)보다 오히려 낮다. 0.841(mix_adjacent_s0), 0.821(single_a026_s1). 앞단은 더 많이, 더 크게 활성화됐다.
- [확인] blocks.6 의 입력은 blocks.5(ReLU) 출력이라 전부 0 이상이다 (models/vqvae.py:142-148). blocks.6.weight 평균은 대조(mix_adjacent_s1) -0.00181 에 비해 -0.00400, -0.00295 로 더 음수다. 값이 커진 비음수 입력과 평균이 음수인 커널이 곱해지면 출력이 전부 음수가 될 수 있다.
- [추정] 따라서 blocks.7 이 먼저 죽은 까닭은 앞단 증폭과 blocks.6 커널 부호의 조합일 수 있다. blocks.6.bias 변화(평균 약 -0.014)는 활성값 크기(rms 114)에 비해 무시할 만해서 bias 가 ReLU 를 죽였다는 설명은 약하다. 다만 활성화 평균은 저장되지 않아 직접 확인할 수 없다.
- [확인] 큰 변화는 특정 층에 몰리지 않았다. blocks.8.weight, dec.blocks.0.weight 도 상대 변화가 크다 (results.json:62169-62187). 전 층이 함께 움직인 사건에서 "첫 층"을 원인으로 읽을 근거가 없다.
- [미확인] 사건 순서. 체크포인트는 사후 상태만 담는다.
- 구분할 최소 검사: 아래 제안 실험 3의 층 교체.

대안: gradient 가 먼저 커진 것이 아니라, 작은 불안정 하나가 활성값과 gradient 를 키우는 되먹임으로 번졌을 수 있다. 이 경우 큰 gradient 는 원인이 아니라 붕괴 과정의 일부다. 디코더에는 정규화 층과 grad clip 이 없다 (results.json A_code_facts). 현재 자료로는 이 설명과 외부 충격 설명을 구분하지 못한다.

### 치명 2. "predictor 멀쩡" 은 현재 검사로 뒷받침되지 않는다

뒷받침되는 것:
- [확인] 경로 분리. 재구성 손실은 z.detach() 를 받고 예측 경로의 디코더 호출은 no_grad 다 (models/visual_world_model.py:211-214, 254-256). 디코더 폭주가 gradient 로 predictor 에 닿을 길은 없다. optimizer 도 따로이고 grad clip 이 없어 전역 norm 을 통한 간섭도 없다.
- [확인] predictor 가중치 변화 최대 비율 1.17, 1.10 (criteria_table). 단 중요 3 의 이유로 약한 근거다.
- [확인] predictor pos_embedding 의 exp_avg_sq 평균, epoch 3: mix_adjacent_s0 1.74e-10, 대조 1.71e-10, 1.67e-10 / single_a026_s1 7.18e-11, 대조 6.91e-11 (results.json:8038, 36193, 47455, 19300, 58717). 파라미터 하나만 본 값이다.
- [계산] val_loss 에서 첫 배치 재구성 MSE 를 뺀 근사 z 손실, epoch 15: mix_adjacent_s0 0.1873, 대조 0.1874, 0.1870 / single_a026_s1 0.1277, 대조 0.1262, 0.1263 (recon_18runs.csv). 첫 배치 값을 뺀 근사라 오차가 있다.

뒷받침되지 않는 것:
- [확인] 검사 E 는 입력 간 std 하나다 (run.py:341-356). action 0 벡터 고정. 입력을 그대로 복사하는 함수도 통과하므로 예측 품질, action 반응, 크기 정보를 재지 않는다. epoch 3 값이 대조와 비슷하다(log.md:176)는 것은 "상수로 무너지지 않았다"는 뜻뿐이다.
- [확인] val_z_visual_err_pred 는 첫 배치 값이고 표본이 작다. epoch 15 값 0.2062 대 0.1851, 0.1815 (mix_adjacent), 0.1310 대 0.1205, 0.1329 (single_a026). mix_adjacent_s0 은 두 대조보다 높지만 시드 간 범위(예: single_a248 0.2158~0.2500)를 감안하면 판단 불가다. 멀쩡함의 증거도 이상의 증거도 아니다.
- [확인] 003 의 측정 대상인 open-loop 면적은 predictor 와 디코더가 함께 만든다. 두 run 은 이 경로의 값 자체가 측정 실패라 predictor 의 물리 예측 품질은 한 번도 관측되지 않았다.
- [미확인] 잠재 공간 probe, 다른 디코더로 복호한 결과. 어느 것도 수행되지 않았다.

말할 수 있는 최대치: "디코더 붕괴가 gradient 로 predictor 에 전달될 경로는 없고, z 손실은 대조와 큰 차이가 없다." 물리 예측이 정상이라는 말은 할 수 없다.

### 치명 3. "디코더만 재학습하면 측정을 살린다" 는 경로도 비교 가능성도 검증되지 않았다

1. [확인] 바로 쓸 코드 경로가 없다.
   - train.py:207-210 은 model_latest.pth 가 있으면 저장된 decoder 를 그대로 로드한다. :274 의 `if self.decoder is None` 때문에 새 디코더가 만들어지지 않고 env.decoder_path 도 쓰이지 않는다.
   - 003 패치(train_resume.patch:85-92)는 체크포인트에 decoder_optimizer 가 있으면 새 optimizer 에 넣는다. 붕괴 후 상태를 지우지 않으면 새 디코더에 오염된 Adam 상태가 들어간다.
   - train_predictor=False 로 돌리면 _keys_to_save 에서 predictor 가 빠져(train.py:162-166) 이후 저장본에 predictor 가 없다. 별도 스크립트가 필요하다.
2. [확인] 체크포인트에 데이터와 인코더가 없다 (A_code_facts.checkpoint_contents). 학습 데이터를 Drive 에서 다시 받고 GPU 를 써야 한다. 재학습은 승인되지 않았다 (hypotheses.md:350).
3. [미확인] 재학습에 쓸 predictor 체크포인트(model_15 또는 model_latest)의 현존. config 상 남겨야 하지만 (003 config.yaml:146) 007 은 휴지통만 조회했다 (run.py:187).
4. 비교 가능성.
   - [확인] 유리한 점: 원래 학습에서도 디코더는 z_pred 를 학습하지 않는다. 입력은 인코더 z 뿐이므로 predictor 와 독립으로 학습해도 절차가 어긋나지 않는다.
   - [추정] 불리한 점 (a) 같은 시드, 같은 배치 순서로 다시 돌리면 디코더 경로에 난수가 없어(dropout 없음) 같은 붕괴가 재현될 수 있다. 피하려고 시드를 바꾸면 원래 run 과 다른 디코더다. (b) 디코더의 정체가 기울기에 얼마나 기여하는지 모른다. 이 분산을 재지 않으면 살린 값을 해석할 수 없다. (c) 재붕괴 위험이 있다. 관측 빈도 003 에서 2/12 (002 0/6, 001 0/3, log.md:156). 붕괴하면 다시 돌리는 규칙을 사전에 정하지 않으면 선택 편향이 생긴다. (d) 결과를 본 뒤 두 run 만 다른 절차로 만드는 것이라 사전 등록된 12 run 과 동등한 지위가 아니다.
5. [확인] 기대 이득이 작다. log.md:163 은 두 run 을 어떤 값으로 바꿔도 003 전체 판정이 불변이라고 기록한다. 재학습이 바꾸는 것은 run 별 기울기 표뿐일 수 있다.

더 싸고 먼저 해야 할 검사는 재학습 없는 디코더 교체다 (제안 실험 2).

## 중요

### 중요 1. "exp_avg 21개가 정확히 0" 은 log.md 의 오기다. 실제 의미는 더 강하다

- [확인] log.md:176 은 "exp_avg 가 21개 파라미터에서 정확히 0" 이라 쓴다. 실제로는 `norm` 필드만 0.0 이다. 예: results.json:6995-6999 norm 0.0, absmax 5.6e-45, zero_frac 0.0. 값은 0 이 아닌 float32 비정규수이고, norm 은 제곱 과정에서 언더플로우한 것으로 보인다.
- [계산] 5.605e-45 는 최소 비정규수의 정확히 4배다. x*0.9 를 반올림하면 4단위 이하에서 값이 줄지 않으므로 이는 **입력 gradient 가 정확히 0 인 채 지수 감쇠만 하다 멈춘 고정점**이다. 그 값까지 내려가려면 초기값 1e-6~1 에 대해 약 840~970 step 이 필요하다.
- [확인] 21개 목록은 upsample_b 의 blocks.0~6 파라미터 20개와 blocks.8.weight 다 (results.json:6991-7455). [계산] 이것은 blocks.7 출력이 전부 0 일 때 gradient 가 구조적으로 0 이 되는 집합과 정확히 일치한다. blocks.8.bias 는 목록에 없는데 이것도 예측과 맞다.
- 의미: mix_adjacent_s0 는 epoch 3 끝의 약 840 step 이상, 매 배치 전 표본에서 blocks.7 이 완전히 죽어 있었다. 54프레임 hook 측정과 독립인 증거이고 학습 데이터 위의 증거다.
- [확인] 반대로 single_a026_s1 에는 이 흔적이 없다. exp_avg 는 작지만 정상 값이고(results.json:18257 등), 활성 원소 110개(중요 5)를 통한 gradient 가 있는 것으로 보인다. 두 붕괴는 같은 상태가 아니다.
- 조치: log.md 문구 정정이 필요하다. 이 리뷰는 notes/log.md 를 고치지 않았다.

### 중요 2. 큰 gradient 는 일시적 사건으로 지지된다. 단, 다음과 구분해야 한다

- [계산] single_a026_s1 의 exp_avg_sq 는 model_3 -> model_4 에서 순수 감쇠다. 0.999^4125 = 0.01613. blocks.6.weight 평균 2.5804e-6 -> 예측 4.162e-8, 관측 4.1625e-8. absmax 0.08730 -> 예측 1.408e-3, 관측 1.4083e-3. blocks.6.bias, blocks.0.weight 도 같다 (results.json:18682-18684, 24313-24315). 즉 큰 값은 지속되는 gradient 가 아니라 epoch 3 안에 끝난 사건의 잔상이다.
- [계산] 이에 따라 log.md:176 의 50271배, 1861배는 사건 크기가 아니다. 사건 뒤 경과 step 수만큼 줄어든 값이고, 사건 시점을 모르면 환산할 수 없다. 두 run 의 배율 차이를 크기 차이로 읽으면 안 된다.
- [추정] 지속 기간의 상한. Adam 은 고립된 큰 gradient 하나에 원소당 약 lr x 31.6 = 0.0095 만큼 움직인다. 원소당 RMS 변화(delta_norm / sqrt(n)) [계산]:
  - blocks.0.bias 0.0076 (대조 0.0010)
  - blocks.6.bias 0.0122, 0.0145 (대조 0.0023, 0.0034)
  - 정상 학습분을 빼면 단일 step 상한에 가깝거나 약간 넘는다. 한 step 부터 몇 step 정도의 짧은 사건과 맞고, 긴 폭주와는 맞지 않는다. 단일 step 인지는 여전히 구분 불가이며 README:20 의 서술은 유지된다.
- [추정] 0.753 과 0.752 가 두 독립 run 에서 거의 같은 것은 우연이나 공통 원인의 신호로 볼 필요가 없다. Adam 이동 상한에 걸린 값으로 설명된다. 따라서 이 숫자에서 gradient 크기나 원인을 읽을 수 없다.
- 빠진 대조: 대조 run 의 체크포인트는 epoch 2, 3 뿐이다. 다른 epoch 에 큰 gradient 가 있었으나 살아남았는지 모른다. csv 의 recon_max_jump_ratio 는 exp002_single_ind_s0 1.39 (epoch 6), exp003_single_a026_s0 1.17 (epoch 2) 을 기록한다. [미확인] 이것이 작은 사건인지. "큰 gradient 는 붕괴로 이어진다"는 조건부 관계는 관측되지 않았다.

### 중요 3. 판단 기준 (4)는 이 사건을 잡을 수 없는 기준이었다. "비해당" 을 반증으로 읽으면 안 된다

- [계산] Adam 은 step 크기를 정규화한다. 대조의 상대 변화는 blocks.6.weight, blocks.8.weight 에서 0.39~0.58 이다. 10배가 되려면 blocks.6.weight 의 delta_norm 약 69, 원소당 0.064 가 필요하고 이는 단일 사건 상한의 약 7배다. 가중치 층은 짧은 사건으로 기준에 도달하기 어렵다. bias 만 겨우 가능하다.
- [확인] 한 epoch 합산이라 정상 학습 이동(4125 step)에 사건이 묻힌다. 대조 blocks.6.weight delta_norm 6.90 대 붕괴 7.49, 6.49 (results.json:62160, 62755, 63945).
- [추정] 붕괴 후 멈춘 구간이 있어 붕괴 run 의 정상 이동분은 대조보다 적다. 같은 잣대로 비교되지 않는다.
- 결론: (4) 비해당은 "큰 가중치 사건이 없었다"는 뜻이 아니다. 같은 이유로 predictor 의 1.17, 1.10 도 predictor 건강의 증거로 약하다. 임계값 10 은 근거 수치 없이 정해졌고(hypotheses.md:354), 이 Adam 동작을 고려하지 않았다.

### 중요 4. 붕괴 시점은 체크포인트 밖 자료로 좁힐 수 있는데 사용하지 않았다

- [확인] 첫 배치 train_img_mse_reconstructed: epoch 3 mix_adjacent_s0 0.00217, single_a026_s1 0.00128 (정상), epoch 4 0.0729, 0.0342 (붕괴). 첫 배치 뒤 epoch 3 도중에 일어났다 (decoder_recon_by_epoch.json:973, 983, 2065, 2075).
- [확인] train_loss 는 epoch 평균이다 (train.py:724-740).
- [계산] epoch 3 초과분을 epoch 4 초과분으로 나누면 epoch 3 중 붕괴 상태 비율의 상한이 나온다 (같은 분포 대조 기준):
  - mix_adjacent_s0: (0.7932 - 0.7636) / (0.5371 - 0.4600) = 0.38 -> 끝에서 약 1580 step 이내
  - single_a026_s1: (0.7428 - 0.7288) / (0.4385 - 0.4052) = 0.42 -> 끝에서 약 1740 step 이내
  - mix_adjacent_s0 은 중요 1 의 하한과 합치면 epoch 3 시작 후 약 2540~3280 step 구간 (전체 step 약 10800~11500). single_a026_s1 은 하한이 없어 약 2380 step 이후라는 것만 말할 수 있다.
- [추정] 초기 일시 손실 질량을 가정하지 않은 값이라 실제 붕괴는 더 늦을 수 있다. 시드 간 잡음(대조 차이 약 0.0004)은 작다. 일시 손실 질량(손실 x step) 상한은 약 57 (mix_adjacent_s0), 58 (single_a026_s1) 로 작지 않아 단일 큰 손실 step 은 배제되지 않는다.
- 주의: 두 run 이 epoch 3 후반으로 비슷해 보이지만 mix_adjacent_s0 만 구간이 있다. "같은 시점"은 입증되지 않았다.

### 중요 5. 입력 54프레임과 own 집합은 표본 수를 늘리지 않는다

- [확인] 5개 run 의 frames_sample `real` 이 동일하므로 (README:33) `own:<run>` 은 `common` 의 복제다. results.json 의 D.own 값 전체가 독립 정보가 아니다.
- [확인] 프레임은 test split 이고 run 의 학습 분포와 다르다 (extract.py:152, 003 config:58). single_a026 에 9개 크기 중 8개는 학습 밖 크기다. 대조 MSE 0.0112 (single_a026_s2) 와 val 0.0015 의 차이가 이 때문일 수 있다 (추정). log.md:176 의 MSE 값을 val_img_mse 와 직접 비교하면 안 된다.
- [계산] single_a026_s1 의 "입력 > 0" 원소는 1.3533e-5 x 8,128,512 = 110개다. model_3, model_4 동일. 채널로는 2/192. 이 110개가 어떤 프레임에 있는지 저장되지 않았다 (run.py:283 은 채널 합만 저장). 학습 크기(a026) 프레임에서도 살아 있는지 미확인이다. 학습 밖 큰 크기에서만 활성일 수도 있다.
- "모든 입력에서 비활성" 은 54프레임에 대한 진술이다. mix_adjacent_s0 은 중요 1 로 보강되지만 single_a026_s1 은 보강되지 않는다.

### 중요 6. "회복하지 못했다" 는 model_4 까지만 체크포인트로 본 것이다

- [확인] 체크포인트로 본 범위: mix_adjacent_s0 은 model_3 까지, single_a026_s1 은 model_4 까지 (config.yaml:18-20). 003 측정에 실제로 쓰인 것은 model_latest (extract.py:37) 인데 검사되지 않았다.
- [확인] 간접 근거: val_img_mse_reconstructed 가 epoch 15 까지 유지(0.0786, 0.0331, csv), area_ol 0 (log.md:155).
- [확인] 두 run 의 정지 양상이 다르다. single_a026_s1 은 model_3->4 에서 앞단 상대 변화가 blocks.0.weight 0.021, blocks.6.weight 0.00099 로 거의 멈췄다 (results.json:63262, 63352). 큰 exp_avg_sq 가 유효 학습률을 누른 것과 입력 희소성 중 무엇이 주된지는 구분되지 않는다. exp_avg_sq 는 이후 감쇠하므로 epoch 15 에서 같은 상태인지 미확인이다.
- [확인] 그림: 두 복원 시트의 epoch 3 는 가운데 사각형과 테두리 무늬뿐이고, 정적 배경과 공이 없다. [추정] 균일 입력을 받는 conv 망은 가장자리 효과 말고는 공간 구조를 만들 수 없어 이 형태가 설명된다. MSE 최적 상수 이미지(평균 이미지)도 만들지 못하는 이유이기도 하다.
- 확인 방법: 남아 있는 model_5/10/15 에 같은 hook 을 적용 (제안 실험 3).

### 중요 7. 다른 대안 설명과 구분 검사

| 대안 | 현재 자료 | 남은 불확실성 | 최소 구분 검사 |
|---|---|---|---|
| 데이터 이상 배치 | [확인] shuffle=False, slice permutation 은 초기화 때 한 번 (train.py:130, traj_dset.py:75). 같은 배치가 epoch 1, 2 에도 지나갔다면 데이터만으로는 부족하고 디코더 상태와의 상호작용이 있어야 함. [확인] 목표는 [-1,1] 이고 z 는 정규화 토큰(models/dino.py:25, 추정: 값이 무한정 커지지 않음) | 배치 순서가 epoch 간 정말 동일한지 [미확인] (accelerate 준비 뒤 sampler). predictor exp_avg_sq 무변화는 z 손실 큰 사건만 배제, 이미지만의 이상은 배제 못 함 | epoch 3 재현 (제안 1) |
| 내부 수치 불안정 (lr 3e-4, clip 없음, 정규화 없음) | 외부 원인이 불필요. 2/12 의 낮은 빈도, 앞단 증폭과 일관 | epoch 2 에는 전조 없음 [확인]: blocks.5 비율 0.955, 0.957 대 대조 0.959~0.961, blocks.7 0.772, 0.745 대 0.748~0.782 (results.json:69880-69897, 70784-70801). 왜 이 두 run 만 불안정해졌는지 설명 못 함 | 재현 + clip 반사실 (제안 1) |
| Colab 메모리 오류, OOM | [확인] returncode 0, n_attempts 1, n_resumes 0, NaN/Inf 0. OOM 크래시는 아님 | 무증상 비트 오류는 배제 불가. 다른 날 두 번 일어난 점은 불리한 증거지만 반증은 아님 | 재현 실패 여부 (제안 1) |
| 혼합 정밀도 | 코드에 없음 [확인]. 003 의 py, sh, yaml, md, txt 에 mixed_precision, accelerate launch grep 결과 없음 [확인]. 노트북은 grep 범위 밖 | 환경 설정은 체크포인트로 구분 불가 (A_code_facts.amp) | Colab 로그 또는 accelerate 설정 확인 |

세션 관련:
- [확인] 두 run 은 약 하루 차이다. single_a026_s1 2026-11-05 12:57~17:35, mix_adjacent_s0 2026-11-06 13:17~17:55 (train_env. log.md:153 의 완료 시각과 맞춰 UTC 로 판단).
- [추정] 다른 런타임일 가능성이 높다. 그 사이 2026-11-06 약 08:12 UTC 에 런타임 끊김 추정 기록이 있다 (log.md:146). runtime ID 는 기록되지 않아 [미확인].
- [확인] 각 붕괴 run 종료 2~3분 뒤 대조가 시작돼 정상이었다(single_a026_s2 17:37, mix_adjacent_s1 17:58). 같은 세션이었다면 세션 지속 고장에 불리하다 (세션 동일성은 추정).
- 같은 epoch 3 일치: 사건 2개뿐이라 판단 불가. 균등 가정이면 1/15, 학습 초기에 위험이 몰리면 더 흔하다. 공통 원인의 증거로 쓰지 말 것.
- [추정] 시드와 데이터가 달라 배치 순서가 공유되지 않으므로 공통 배치 설명은 해당하기 어렵다.

### 중요 8. 재현성: 지금 상태로 며칠 뒤 재실행이 불가능할 수 있다

- [확인] config.yaml:8-12 의 모든 경로가 세션 전용 scratchpad 에 고정이다. slim, 코드 사본, hub 캐시, common_npz 가 저장소 밖에 있다 (README:16-17). README 는 /tmp 가 tmpfs 라고 적는다.
- [확인] 원본은 로컬 삭제(run.py:218). 슬림에는 optimizer 요약만 있고 텐서가 없어(run.py:133-142) 재현 실험 입력이 될 수 없다. 원본은 Drive 휴지통에만 있다.
- [확인] 휴지통 보존 기간은 추정(log.md:164)이다. 기한이 지나면 B, C, D 의 재계산과 제안 실험 1, 3 의 model_2/3 부분이 모두 불가능하다. 복원 또는 별도 보관은 사람 결정이 필요하며 시간 제약이 있다.
- [확인] git: 007 전체 미커밋이고 hypotheses.md 도 수정 상태(gitStatus). 사전 기록을 commit 으로 증명할 수 없다.
- [확인] Glob 수정 시각 순서(오름차순 가정): hypotheses.md < config.yaml < smoke_results.json < figures < run.py < results.json < log.md < README.md. run.py 가 figures 뒤에 수정됐다. summarize 추가로 보이지만(results.json 도 그 뒤 재작성) analyze 시점 run.py 는 남아 있지 않다. [미확인] 현재 judge() 가 실행 당시와 같은지.
- [확인] 기록 누락: results.json meta 는 torch 버전만 있다 (results.json:75167-75172). numpy, torchvision, rclone 버전, run.py 해시, slim 해시 없음. README 의 버전 서술에 의존한다.
- results.json 만으로 재계산 가능: 판단 기준 (1)~(4), 층별 std, rms, 파라미터별 변화, 중요 1, 2 의 산술. 불가능: 활성화 평균과 부호, 프레임별, 채널별 값, 층 교체, 원소 수준 분석.
- A_code_facts 는 손으로 쓴 문자열이다 (run.py:66-84). 내가 대조한 항목(train.py:61, 336-338, vqvae.py 구조, visual_world_model.py:254-256)은 맞지만 자동 검증은 아니다.
- 프로젝트 규칙: 논문, 노트의 수치는 results.json 에서 가져와야 하므로 이 문서의 [계산] 값은 스크립트로 재계산해 기록한 뒤 인용할 것.

## 사소

1. [확인] run.py:190-193 의 시각 필드 오표기. `modtime_utc` 는 실제로 KST 로 보이고 `modtime_kst` 는 9시간 더 밀렸다. 근거: mix_adjacent_s2 model_3 의 modtime_kst 2026-11-07 17:35:22 는 분석 실행 시각 14:47:17 보다 미래다 (results.json:45182, 75168). `modtime_utc` 를 KST 로 읽으면 5개 run 모두 train_env 시작 시각과 맞는다(시작 후 약 37분에 model_2, 약 18.5분 간격). 이 시각을 인용하면 틀린다.
2. [확인] run.py:139-141 action_encoder_optimizer 요약의 이름 충돌. action, proprio encoder 파라미터 이름이 같아 per_param 이 뒤 값으로 덮인다. n_state 4 인데 per_param 2개다 (results.json:9628-9675). C 의 action 수치는 쓸 수 없다. B 와 B_delta 는 모듈별이라 영향 없음.
3. [추정] patch_embed.weight (n=10, 위 충돌로 proprio 쪽 값으로 보임) 의 exp_avg, exp_avg_sq 가 11개 체크포인트 모두 정확히 0 이다. 학습 proprio 입력이 0 이라면 설명된다. 맞다면 E 의 proprio 0 고정은 실제와 같고, action 0 고정만 비현실적이다. 21개 건과 별개이므로 섞지 말 것.
4. [확인] log.md:171 의 "이 층 이후 27개 층"은 blocks.7 자신을 포함한다 ([계산] 50개 층 중 blocks.7 부터 끝까지 27개). 또한 mix_adjacent_s0 의 하류 소실은 상류 0 의 자동 결과라 독립 증거가 아니다.
5. [확인] 사전 기록과 구현의 작은 차이: 대조 값을 평균으로 씀(hypotheses.md:356 "대조 run 값", run.py:528, README 는 평균 명시), 층 단위가 아닌 파라미터 단위(run.py:557), model_4 를 대조 epoch 3 과 비교(run.py:522, 사전 기록에 없음), 혼합 대조 풀(mix 2, single 1). (2) 는 차이가 4자릿수(4.2e-6 대 4.9e-2)라 이 선택에 흔들리지 않을 것이다 (추정).
6. [확인] (3) "부분 해당"(log.md:172)은 사전 범주가 아니다. single_a026_s1 은 정의상 비해당이고 criteria_table 도 false 다. 8,128,512 원소 중 110개 차이로 run 사이에 해당/비해당이 갈려 엄격 정의의 변별력은 낮다. log 가 수치를 함께 적어 오독 위험은 작다.
7. [확인] single_a026_s0 이 대조에서 빠진 이유가 README, config 에 없다. 재개와 혼합 GPU 이력 때문으로 보인다 (log.md:157, 추정). 대조 수가 분포별로 2, 1 로 불균형이다.

## 측정 버그 점검에서 문제를 찾지 못한 항목

읽어서 확인한 범위이며 실행하지 않았다.

- 층 대응: blocks.0 Conv, 1~4 ResBlock, 5 ReLU, 6 첫 ConvT, 7 ReLU, 8 ConvT (vqvae.py:137-153). "첫 ConvT 뒤 ReLU" 맞음.
- hook: pre-hook 은 inplace ReLU 전에 실행되고, post-hook 은 즉시 복사 합산한다(run.py:276-302). `out += input`, inplace 로 값이 오염되지 않음. 자식 hook 이 부모보다 먼저라 순서도 실행 순서.
- 의존성 계산: 분산 합산은 배치 경계에 안전(run.py:298-309). 평균 std 는 소수 채널 활성을 희석하지만 결론에 영향 없음 (단 single_a026_s1 의 4.4e-6 은 110개 원소에 집중된 값으로 읽을 것).
- epoch 대응: 저장은 train, val 뒤(train.py:376-381). `ck["epoch"] == e` assert (run.py:124). step = epoch x 4125. model_e 는 epoch e 끝 상태 맞음.
- 슬림 사본: 모듈 객체를 그대로 저장(run.py:145-149). sha256 은 원본에만 있고 슬림 해시는 없음. 11개 모두 sha 기록, 크기 일치. 다른 파일을 받았을 가능성은 위 시각 대조로 낮음.
- 모드: 디코더에 dropout, BN 없어 eval/train 동일. 학습 때 encoder 는 train 모드지만(visual_world_model.py:71-72) dinov2 hub 모델의 train/eval 차이는 코드상 확인 못 함 [미확인]. 영향은 작을 것으로 추정.
- 인코더, 정규화: DinoV2Encoder, 같은 hub commit, Resize(196), [-1,1] (run.py:250-253, img_transforms 와 같은 스케일). torch 2.0.1, torchvision 0.15.2 는 Colab 요구 사항과 같음(requirements_colab.txt:8-9). uint8 경유 양자화는 미세함 (추정).
- 파이프라인 교차 검증: 대조 epoch 3 MSE 0.0029, 0.0028 은 학습 로그 val 0.0026, 0.0023 과 같은 규모. 복원 시트 epoch 2 도 정상.
- predictor CPU 실행: Attention.bias 는 버퍼가 아니지만 map_location=cpu 로 CPU 에 옴 (추정). num_hist 1 이라 1프레임 입력은 학습 구조와 맞음(003 config:96).
- 사전 기록 순서: hypotheses.md 의 마지막 수정이 config.yaml 보다 앞선다. 결과 후 수정 흔적 없음. 단 파일 시각 증거뿐이고 commit 증거는 없음.

## 이 결과를 믿으려면 추가로 필요한 실험

모두 사람 승인이 필요하다. 1, 2 는 GPU 또는 데이터가 필요할 수 있다.

1. **epoch 3 재현과 clip 반사실 (원인 구분에 가장 강함).** model_2 원본의 decoder + decoder_optimizer 상태에서 같은 시드의 slice 순서로 디코더만 epoch 3 재생, step 별 손실, 층별 gradient norm, 층별 활성 통계 기록. 붕괴 run 2개와 같은 분포 대조 1개씩. 재현되면 내부 불안정이나 상태와 배치의 상호작용이고, 재현되지 않으면 비결정성이나 하드웨어 쪽이다. 재현되면 grad clip 또는 해당 배치 제외 반사실로 필요조건을 확인한다. 디코더 경로에 난수가 없어(추정) predictor 없이 돌릴 수 있고 비용은 run 당 1 epoch 이하로 추정된다. 원본이 휴지통에 있을 때만 가능하므로 먼저 복원 결정이 필요하다.
2. **디코더 교체 null 검사 (재학습 결정 전 필수).** 재학습 없이 model_15 조합으로 면적 기울기를 잰다. (a) 정상 run 끼리 교체: mix_adjacent_s1 predictor + s2 decoder 대 s1 자기 조합. 디코더 정체가 기울기를 얼마나 바꾸는지가 재학습 값의 오차 하한이다. (b) mix_adjacent_s0 predictor + s1 decoder, single_a026_s1 predictor + s2 decoder. 대조 범위 안이면 처음으로 "predictor 가 물리 예측 면에서 정상"이라는 근거가 생긴다. 교체 차이가 크면 재학습 값은 비교 불가다. 조합 규칙, 판정 범위를 먼저 hypotheses.md 에 적을 것.
3. **CPU 층 교체와 나머지 체크포인트 감사 (기존 자료, 재학습 없음).** model_2 와 model_3 사이에서 파라미터 묶음 교체(앞단만, blocks.6 만, 하류만)로 어느 묶음이 활성을 되살리는지 측정하고, 활성화 평균과 부호, 프레임별 활성 원소도 저장한다. 같은 hook 을 model_5/10/15 와 model_latest 에 적용해 "회복하지 못했다"를 측정에 쓰인 체크포인트까지 확인한다. 층 교체는 사건 순서가 아니라 현재 상태의 필요조건만 알려주므로 원인 주장에는 실험 1 이 필요하다.

## 확신도

**전체: 중간.**

- 높음: 측정 파이프라인에 결론을 뒤집는 버그가 없다는 점 (코드를 직접 대조했고 대조 run MSE 와 복원 시트가 맞음). mix_adjacent_s0 의 blocks.7 완전 사망 (hook, 비정규수 고정점, 21개 집합 일치의 세 근거). exp_avg_sq 가 일시적 사건의 잔상이라는 점 (순수 감쇠가 유효숫자 약 4자리까지 일치). "predictor 멀쩡", "디코더만 재학습"이 현재 근거로 과장이라는 점 (검사 범위와 코드 경로를 직접 읽음).
- 중간: 붕괴가 epoch 3 후반이라는 시점 추정 (손실 일시 질량 가정, 값은 재계산 필요). blocks.7 이 원인이 아니라 증상이라는 판단 (rms 증폭으로 지지되지만 활성화 평균 미저장, 층 교체 미실시).
- 낮음: 내부 불안정이 외부 충격보다 그럴듯하다는 기울기 (배치 재방문, 입력 유계성에 의존한 추론이고 재현 실험 없음). 원인은 현재 미확인이다. Adam 상한 해석도 이전 상태를 무시한 근사다.
- 한계: 코드를 실행하지 못했고, [계산] 값은 손 계산이다. predictor 는 pos_embedding 하나만 직접 비교했다. Colab 로그, train.log, 런타임 ID 는 보지 못했다.
