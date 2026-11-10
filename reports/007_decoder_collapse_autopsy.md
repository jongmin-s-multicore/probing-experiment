# 실험 007_decoder_collapse_autopsy 보고서

- 작성: reporter, 2026-11-07
- 입력: experiments/007_decoder_collapse_autopsy/{README.md, config.yaml, results.json, analysis/stats_007_out.json}, notes/reviews/007_prior.md, 007_stats.md, 007_skeptic.md, notes/hypotheses.md ("007 진단 질문", 349-357행), notes/log.md (2026-11-07 / 007 항목, 166-178행), reports/003_train_dist_dependence.md.
- 이 보고서는 새 측정을 하지 않았다. 수치 출처 표기:
  - `rj:` = results.json 키 경로(줄 번호 병기). 리포터가 파일에서 직접 읽은 것은 `criteria_table` 전체(75531-75696행), `meta`(75167-75172행), 정정 대상 3건의 해당 줄(6991-7458행의 exp_avg 21개 항목 중 첫 항목 전체와 absmax 줄 21개, 45175-45186행, 9628-9675행)이다. 그 밖의 results.json 수치는 log.md 또는 리뷰의 인용이며 리포터가 대조하지 않았다.
  - `log:` = notes/log.md 007 항목. log.md 는 수치가 results.json 에서 왔다고 적었으나 리포터는 `criteria_table` 밖의 값은 대조하지 않았다.
  - `stats:` = 007_stats.md 의 계산값. 계산은 analysis/stats_007.py 가 했고 출력은 analysis/*.csv, stats_007_out.json 이다. 리포터는 stats_007_out.json 앞부분(rms_ratio 5행, crit3 앞 3행, `layers_rms_outside_ctl_range` 45 / 46)만 열어 stats 서술과 일치함을 확인했다. csv, 그림 3개(fig_*.png), 복원 시트 PNG 5장은 열어보지 않았다. 이미지 서술은 prior/skeptic 의 서술이다.
  - `prior:`, `skeptic:` = 해당 리뷰가 적은 값. skeptic 의 `[계산]` 은 skeptic 의 **손계산**(코드 실행 없음, 리뷰 스스로 "인용 전 재계산 필요"라 적음)이고, `[확인]` 은 파일을 직접 읽었다는 뜻, `[추정]`, `[미확인]` 은 리뷰의 표기를 그대로 옮겼다.
  - prior 의 문헌 인용(Lu 2019, Kingma & Ba 2015, Molybog 2023 등)은 prior 가 웹 검색 없이 기억으로 적었고 본문을 확인하지 않았다고 밝혔다. 리포터도 확인하지 못했으므로 이 보고서는 해당 문헌을 근거로 쓰지 않는다.
- 표기: "추정" 또는 "미확인" = 분석가 또는 리포터가 확인하지 못한 주장. 세 분석가는 서로의 파일을 읽지 않고 독립 작성했다.
- 지위: 007 은 가설 검증이 아니라 진단 질문이다(hypotheses.md 349행, 사용자 승인). 성공 기준이 없고 산출물은 5개 판단 기준 각각의 해당/비해당과 대조 값이다. 재학습과 GPU 사용은 승인되지 않았다.
- 이 보고서의 어떤 제안도 승인이 아니다. 사람이 결정할 사항은 8절에 모았다.
- 리드의 잠정 해석 4문장(5.2절 표)은 사실이 아니라 **검토 대상**이었다. 이 보고서는 그 문장을 결과로 쓰지 않는다.

---

## 1. 한 문단 요약

003 에서 val_img_mse_reconstructed 가 epoch 2 에서 3 사이에 급등해 디코딩 프레임이 입력과 무관해진 두 run(exp003_mix_adjacent_s0, exp003_single_a026_s1)의 Drive 휴지통 체크포인트를 CPU 로 검사했다(재학습 없음, 대조 run 3개). 사전 기록한 판단 기준의 결과는 **(1) 비해당, (2) 해당, (3) 부분 해당(mix_adjacent_s0 는 엄격 적용 해당, single_a026_s1 은 비해당), (4) 비해당, (5) 비해당**이다(`rj: criteria_table`, log). (2)에서 입력 의존성이 처음 사라지는 층은 세 체크포인트(mix_adjacent_s0 model_3, single_a026_s1 model_3, model_4) 모두 `upsample_b.blocks.7`(upsample_b 의 첫 ConvTranspose 다음 ReLU)이다. 체크포인트로 구분 불가로 기록된 것은 epoch 3 도중 정확한 시점과 단일 큰 gradient 여부다. 세 분석가가 일치하는 것은 (a) 판단 기준 표가 원값과 사전 정의대로 재현된다는 점, (b) 디코더 쪽 ReLU 층의 사실상 사멸과 이후 비회복이 관측으로 지지된다는 점, (c) 기준 (4) 비해당은 "큰 gradient 가 없었다"의 반증이 아니라는 점, (d) 디코더 손실이 코드상 predictor 로 역전파되지 않는다는 점, (e) 붕괴 원인은 확정되지 않았다는 점이다. 갈리는 곳은 "큰 gradient" 의 지지 강도(prior 부분 지지, 확신 낮음 / stats 체크포인트로 구분 불가 / skeptic 일시적 큰 gradient 는 강하게 지지, 원인인지 결과인지는 미정), blocks.7 이 원인 층인지 증상 층인지, "디코더만 재학습하면 측정을 살린다" 의 타당성(prior 코드 구조상 타당하나 조건 많음 / skeptic 바로 쓸 경로 없고 근거 없음 / stats 판정 없음), 예방책과 비교 가능성이다. 이 보고서는 정정 3건을 표시한다(6절): log.md 의 "exp_avg 21개 정확히 0"(실제로는 비정규수 5.605e-45 이고 norm 필드만 0.0), `modtime_utc`/`modtime_kst` 표기, action/proprio optimizer 요약의 이름 충돌. 세 건 모두 `criteria_table` 의 판정은 바꾸지 않는다. 확신 수준: 관측 사실(ReLU 층 사멸, 입력 무관 출력)은 높음(stats, skeptic, prior), 붕괴 원인(큰 gradient 여부, 내부 불안정 대 외부 충격)은 낮음, predictor 정상 주장의 정량 근거는 낮음. 분석가별 전체 확신도는 prior 중간, stats 중간(부분별로 다름), skeptic 중간이다.

---

## 2. 배경과 가설

- 배경: 003 의 12개 run 중 2개(mix_adjacent_s0, single_a026_s1)는 900 에피소드 전부 area_ol 이 0 이고 기울기 0.0 이다. 두 run 의 val_img_mse_reconstructed 가 epoch 2 에서 3 사이에 뛰어올라 epoch 15 까지 유지된다(mix_adjacent_s0 0.0030 -> 0.0787, single_a026_s1 0.0015 -> 0.0333; log 003 항목, 운영 로그 값이며 results.json 값이 아님). 003 보고서(4.5절, 6절)는 이를 측정 실패로 다루었고 원인은 미확인으로 남겼다.
- 007 은 가설이 아니라 **진단 질문**이다. 질문: 체크포인트(붕괴 run model_2/3/(4), 대조 run mix_adjacent_s1/s2, single_a026_s2 model_2/3)에서 아래 5개 관측이 되는가(hypotheses.md 351-355행).
- 사전 기록한 판단 기준(결과는 이 표현으로만 보고; hypotheses.md 353-356행, README):
  1. NaN/Inf 있음.
  2. 입력 의존성이 처음 사라지는 층이 특정됨. 조작적 정의: 층 출력의 입력 간 std 가 대조 3개 run 평균의 1/100 미만(config `dependence_lost_ratio: 0.01`).
  3. 활성화가 전부 비활성인 층이 특정됨. 조작적 정의: ReLU 입력이 모든 입력, 모든 위치에서 <= 0.
  4. epoch 3 가중치 변화가 대조 대비 10배 이상인 층이 특정됨. 조작적 정의: 붕괴 run 의 ||W3-W2||/||W2|| 가 대조 3개 run 상대 변화 최댓값의 10배 이상(config `weight_change_ratio: 10.0`).
  5. 어느 것도 해당 없음.
- 체크포인트로 구분 불가한 것으로 사전 기록: epoch 3 도중 정확한 시점, 단일 큰 gradient 여부.
- 이 실험이 주장하려는 것은 5개 기준의 해당/비해당과 수치뿐이다. 붕괴 원인은 주장하지 않는다(README).

---

## 3. 설정과 방법 (config 기준)

| 항목 | 값 | 출처 |
|---|---|---|
| 코드 | acwm-phys commit 5a640bed66002eca7bdb31d10b2bde59ad944551 (git archive 로 scratchpad 에 풀어 읽음) | config.yaml:5 |
| 시드 | 0 (추론만 하며 eval 모드, 결과는 시드에 의존하지 않는다고 config 가 적음) | config.yaml:3 |
| 붕괴 run | exp003_mix_adjacent_s0 model_2, 3; exp003_single_a026_s1 model_2, 3, 4 | config.yaml:18-20 |
| 대조 run | exp003_mix_adjacent_s1, exp003_mix_adjacent_s2, exp003_single_a026_s2 의 model_2, 3 | config.yaml:21-24 |
| 같은 분포 짝 | mix_adjacent_s0 의 짝 = s1, s2; single_a026_s1 의 짝 = s2 | config.yaml:25-27 |
| 입력 | 9 크기 x 에피소드 2 x 시점 3(t = 0, 6, 12) = 54 프레임. DINOv2 `dinov2_vits14`, 입력 196, torch hub 고정 commit | config.yaml:28-31; README |
| predictor 입력의 proprio, action | 모두 0 벡터로 고정(frames_sample 에 행동이 없고 데이터를 내려받지 않음) | config.yaml:32-33; README |
| 검사 5종 | A 코드 사실, B 층별 가중치 통계와 epoch 2->3 변화, C optimizer 상태, D 층별 활성화(입력 간 std, ReLU 비활성 비율)와 출력, E predictor 출력의 입력 간 std | README |
| 환경 | conda `lab07`: python 3.9, torch 2.0.1+cpu, torchvision 0.15.2+cpu, numpy 1.26.4. CPU, 분석 1690.9 초 | README; `rj: meta` (75167-75172행) |
| Drive | `--drive-trashed-only` 읽기 전용, 로컬 방향 copy/lsf 만. 원본(368 MB)은 모듈과 optimizer 요약만 담은 slim(약 125 MB)으로 바꾸고 로컬 원본은 삭제. 체크포인트, 코드 사본, hub 캐시는 저장소 밖 scratchpad 에 있음 | README; config.yaml:8-12 |

실행에 관한 주의:
- 5개 run 의 frames_sample.npz 의 `real` 배열이 모두 같아서 입력 집합 `own:<run>` 과 `common` 의 결과는 동일하다(README). 독립 표본이 아니다.
- 입력 프레임은 test split 이며 run 의 학습 분포와 다르다(skeptic 중요 5, `[확인]`). 예: single_a026 은 9개 크기 중 8개가 학습 밖 크기다. 따라서 007 의 실제 프레임 대비 MSE 를 003 의 val_img_mse 와 직접 비교하면 안 된다(skeptic).
- 첫 fetch 가 lsf 형식 문자열 오타로 실패해 수정 후 재실행했다(코드 버그, 재시도 3회 소진 아님). smoke 로 만든 slim 1개는 메타가 없어 지우고 다시 만들었다(README, log).
- git commit 하지 않았다(README). 사전 기록의 시점을 commit 으로 증명할 수 없다(skeptic 중요 8).
- 재현성: config.yaml 의 모든 경로가 세션 전용 scratchpad 에 고정이고 README 가 /tmp 를 tmpfs 라고 적는다. 슬림에는 optimizer 요약만 있고 텐서가 없어 재현 실험 입력이 될 수 없으며 원본은 Drive 휴지통에만 있다(skeptic 중요 8, `[확인]`). `rj: meta` 에는 torch 버전만 있고 run.py 해시, slim 해시, numpy/torchvision 버전이 없다(리포터가 meta 를 직접 확인).

---

## 4. 결과

모든 수치는 5개 체크포인트 + 대조 6개, 입력 54 프레임(독립 아님), 체크포인트당 결정론적 추론 1회에서 나온 것이다. 붕괴 run 은 각 1 시드다.

### 4.1 판단 기준 표 (사전 기록 표현 그대로; `rj: criteria_table` 75531-75696행, 리포터가 직접 읽음)

| 기준 | 결과 | 붕괴 run 값 | 대조 값 | 출처 |
|---|---|---|---|---|
| (1) NaN/Inf 있음 | **비해당** | 붕괴 5개 체크포인트 모두 0개(가중치, optimizer 상태, 활성화, 출력) | 대조 6개 모두 0개 | `rj: 1_nan_inf` (75532-75548행; "optimizer 상태, 활성화, 출력" 범위는 log 서술) |
| (2) 입력 의존성이 처음 사라지는 층이 특정됨 | **해당** | 가장 처음 기준 미만이 되는 층은 `upsample_b.blocks.7`. 입력 간 std: mix_adjacent_s0 model_3 0.0, single_a026_s1 model_3 4.363765e-06, model_4 4.190461e-06. 이 층 이후 27개 층이 기준 미만. 두 run 의 model_2 는 기준 미만인 층 없음 | 대조 3개 평균 std 0.04909 (epoch 3) | `rj: 2_dependence_lost` (75550-75591행) |
| (3) 활성화가 전부 비활성인 층이 특정됨 | **부분 해당** (log 의 표현. `criteria_table` 은 run 별 true/false) | mix_adjacent_s0 model_3: `upsample_b.blocks.7` 입력 <= 0 비율 1.0, 죽은 채널 비율 1.0 (엄격 정의 해당). single_a026_s1 model_3, model_4: 비율 0.9999864673878811, 채널 단위 0.9895833 (엄격 정의 **비해당**). 두 run model_2: 엄격 비해당, 최대 비율 0.95537, 0.95666 | 같은 층 대조 3개 0.82506, 0.84739, 0.83206 | `rj: 3_all_inactive_relu` (75592-75650행) |
| (4) epoch 3 가중치 변화가 대조 대비 10배 이상인 층이 특정됨 | **비해당** | 디코더 최대 비율 6.0992 (mix_adjacent_s0), 6.0927 (single_a026_s1), 둘 다 `upsample_b.blocks.0.bias`, 상대 변화 0.7532, 0.7524. predictor 최대 1.1656, 1.1041. action_encoder 0.8535, 0.8968. proprio_encoder 0.9998, 0.9997. 같은 분포 대조만 써도 해당 층 없음 | 해당 층의 대조 상대 변화 0.10042, 0.12349, 0.10041 | `rj: 4_weight_change_10x` (75651-75688행) |
| (5) 어느 것도 해당 없음 | **비해당** ((2), (3)이 해당) | - | - | `rj: 5_none` (75689-75691행) |

체크포인트로 구분 불가로 기록된 것: epoch 3 도중 정확한 시점, 단일 큰 gradient 여부(`rj: not_distinguishable_from_checkpoints`, 75692-75695행).

주의: "부분 해당" 은 사전 기록된 범주(해당/비해당)에 없는 표현이다(skeptic 사소 6). 이 보고서는 수치를 함께 적어 오독을 막았다.

### 4.2 판단 기준 밖의 관측 (log 인용; 리포터가 results.json 에서 대조하지 않음)

- 출력 이미지 입력 간 std (common 54프레임, 실제 프레임 0.1003): mix_adjacent_s0 epoch 2 0.0998 -> epoch 3 0.0000. single_a026_s1 0.0862 -> 0.0002 (epoch 4 0.0003). (log)
- 실제 프레임 대비 MSE([-1,1] 스케일): mix_adjacent_s0 0.0031 -> 0.0855, single_a026_s1 0.0129 -> 0.0892 (epoch 4 0.0892). 대조 mix_adjacent_s1/s2 epoch 3 0.0029/0.0028, single_a026_s2 epoch 3 0.0112. (log)
- 복원 시트: 붕괴 run 의 epoch 3 는 입력에 상관없이 같은 흐린 회갈색 격자(prior) 또는 가운데 사각형과 테두리 무늬뿐이고 정적 배경과 공이 없다(skeptic 중요 6). 리포터는 이미지를 열어보지 않았다.
- predictor 출력의 입력 간 std (proprio, action 0 고정; epoch 3): mix_adjacent_s0 0.4793 대 대조 0.4793, 0.4814. single_a026_s1 0.4140 대 single_a026_s2 0.4118 (log; stats 4절은 상대 차이 -0.4% ~ +0.5%, epoch 2 는 -0.5%, -0.7%, +1.2%, 대조 두 개 사이 0.4%).
- decoder_optimizer: step 은 모든 체크포인트에서 epoch x 4125, lr 3e-4, betas [0.9, 0.999], eps 1e-8, weight_decay 0, amsgrad false(log; stats 3절).
- exp_avg_sq 평균의 대조 대비 배율(epoch 3, `upsample_b.blocks.6.weight`): mix_adjacent_s0 1861배, single_a026_s1 50271배(log; stats 3절이 대조 3개 평균 5.13e-11 기준으로 재현). prior 는 대조 mix_adjacent_s1 한 개(7.07e-11) 기준으로 약 1.4 x 10^3 배, 약 3.7 x 10^4 배로 적었다. **분모가 다르다**(대조 3개 평균 대 대조 1개). epoch 2 에는 각각 대조의 0.84배, 1.23배였다(stats).
- single_a026_s1 의 exp_avg_sq 는 model_3 -> model_4 에서 순수 감쇠에 일치한다: 0.999^4125 = 0.01613 (stats 0.016130), blocks.6.weight 평균 2.5804e-6 -> 예측 4.162e-8, 관측 4.1625e-8 (skeptic 중요 2, `[계산]`; stats 는 v4/v3 = 0.016131 로 같은 결론). 반면 `dec.blocks.8.bias` 는 0.0663 으로 gradient 가 살아 있다(stats 3절).
- `upsample_b.blocks.6.weight` norm epoch 2 -> 3: 15.8 -> 18.2 (mix_adjacent_s0), 15.4 -> 17.1 (single_a026_s1), 대조 s1 15.97 -> 18.07 (log). 상대 변화 0.474, 0.423 대 대조 0.432, 0.421, 0.391 (stats 1절). 이 층 가중치는 정상 범위다.
- 층별 규모 이상(epoch 3, 붕괴/대조 평균 rms 비): blocks.0 3.14 / 3.28, blocks.4 8.46 / 9.28, blocks.6 107.72 / 109.82 (`analysis/stats_007_out.json: rms_ratio`, 리포터가 확인; 순서는 mix / single). 같은 층 epoch 2 비는 blocks.0 1.027 / 0.912, blocks.4 1.051 / 0.91, blocks.6 1.217 / 1.012. stats 4절은 blocks.5(ReLU) 40~52배도 적었다(리포터 미대조). 50층 중 붕괴 값이 대조 범위 밖인 층 수는 45 / 46 이지만 이 개수 자체는 정보가 거의 없다(stats: 교환 가능한 4개 중 하나가 범위 밖일 확률이 층당 1/2). 정보는 비의 크기에 있다.

### 4.3 재계산과 민감도 (stats)

- (2) 임계값 민감도: 첫 층의 정체는 임계값 1/3 ~ 1e-4 에서 `upsample_b.blocks.7` 로 바뀌지 않는다. single_a026_s1 은 1e-5 에서 해당 층 없음(비 8.9e-5). 두 run 의 model_2 는 모든 임계값에서 없음(`sens_criterion2.csv`).
- (3) 엄격 정의는 극값 통계량이다. mix_adjacent_s0 model_3 은 양수 원소 0 / 8,128,512, single_a026_s1 은 110 / 8,128,512 이고 생존 채널은 0 대 2 / 192 다(`criterion3_counts.csv`, 110 원소는 skeptic 중요 5 의 `[계산]` 과 일치). stats 는 두 run 을 기능적으로 같은 상태로 보는 것이 더 정직하다고 본다. skeptic 은 다르게 본다(5.2절).
- (4) null: 대조끼리 비교한 최대 비 1.47, 1.23, 1.58. 붕괴 run 6.10, 6.09 (같은 분포 짝만 쓰면 6.10, 7.49). 즉 붕괴 쪽은 대조 null 최댓값의 약 4배다. 사전 문턱 10배는 보수적이었고 3배였다면 해당했을 것이다. stats 는 이것이 사후 문턱 변경 제안이 아니라 민감도 정보라고 명시했다(`crit4_null.csv`). 편향 중앙값 비 2.71 / 2.84, 가중치 중앙값 비 1.24 / 1.42.
- exp_avg 감쇠 하한(stats 3절, `m_decay_steps`): m 이 비정규 바닥까지 감쇠하려면 현실적 초기값에서 약 830~920 step 이 필요하다. skeptic 중요 1 도 약 840~970 step 으로 `[계산]`. 즉 mix_adjacent_s0 는 epoch 3 끝 약 850 step 이상 동안 blocks.7 이 완전히 죽어 있었다고 두 리뷰가 본다.

### 4.4 epoch 3 내 붕괴 시점 추정 (003 로그 기반, 가정에 의존; 체크포인트에서 나온 값이 아님)

이 시점 추정은 체크포인트가 아니라 **003 의 epoch 로그**(`decoder_recon_by_epoch.json`)에서 나온 값이다. 체크포인트로는 구분 불가라는 사전 기록은 유효하다.

| 항목 | stats | skeptic |
|---|---|---|
| 방법 | epoch 평균 train_loss 의 epoch 3 대조 대비 초과분을 붕괴 정상상태 초과분(epoch 4, 5 평균 val_loss 차이)으로 나눠 붕괴 상태 비율 f 를 구함 | epoch 3 초과분을 epoch 4 초과분으로 나눠 붕괴 상태 비율의 **상한**을 구함 (같은 분포 대조 기준, `[계산]`) |
| mix_adjacent_s0 | f = 0.39. 사망 시작 step 약 10761 (epoch 3 의 약 61% 지점), 사망 후 경과 약 1614 step | (0.7932 - 0.7636) / (0.5371 - 0.4600) = 0.38. 끝에서 약 1580 step 이내. 위 exp_avg 하한과 합치면 epoch 3 시작 후 약 2540~3280 step 구간(전체 step 약 10800~11500) |
| single_a026_s1 | f = 0.42. 사망 시작 step 약 10650 (약 58% 지점), 사망 후 경과 약 1725 step | 0.42. 끝에서 약 1740 step 이내. 하한이 없어 약 2380 step 이후라는 것만 말할 수 있음 |
| 가정 | (i) 사망 전 손실 정상, 후 정상상태 손실, (ii) 전환 중 일시적 손실 급등 없음(있으면 f 과대 추정), (iii) predictor 쪽 train_loss 는 대조와 같음, (iv) 시드마다 데이터 순서 다름. 오차 범위 대략 f 0.33~0.43 | 초기 일시 손실 질량을 가정하지 않은 값이라 실제 붕괴는 더 늦을 수 있음. 일시 손실 질량 상한 약 57, 58 로 작지 않아 단일 큰 손실 step 은 배제되지 않음 |

- 두 분석가가 같은 값(f 0.38~0.42)을 얻었으나 stats 는 "추정 시점", skeptic 은 "상한" 으로 쓴다. 해석 방식이 다르다.
- 첫 배치 train_img_mse_reconstructed 는 epoch 3 에서 정상(mix_adjacent_s0 0.00217, single_a026_s1 0.00128), epoch 4 에서 붕괴(0.0729, 0.0342) 이다(skeptic 중요 4, `decoder_recon_by_epoch.json` 줄 번호 인용; 리포터 미대조). stats 는 train.py 에서 이 지표가 매 epoch 첫 배치에서만 계산되므로 epoch 3 중 시점 정보가 없다고 본다.
- 두 run 의 f 가 가까운 것(0.39, 0.42)은 n = 2 라서 우연으로도 충분히 설명되며 공통 원인의 증거로 보기 어렵다(stats 3절, skeptic 중요 7). skeptic: mix_adjacent_s0 만 구간이 있어 "같은 시점"은 입증되지 않았다.

### 4.5 predictor 관련 수치

- 코드 사실(`rj: A_code_facts`; stats, skeptic 이 인용): 디코더 입력은 `z.detach()`, 재구성 손실은 predictor 로 역전파되지 않고 optimizer 도 분리돼 있다. skeptic 은 `visual_world_model.py:211-214, 254-256` 을 `[확인]` 했고 고정 commit 사본(code5a)을 읽었다고 적었다. 리포터는 코드를 열어보지 않았다. A_code_facts 는 손으로 쓴 문자열이다(skeptic 중요 8).
- 가중치 변화 최대 비율 predictor 1.1656, 1.1041 (`rj: 4_weight_change_10x`). 대조 leave-one-out 1.18, 1.21, 1.28 안(stats 4절).
- 003 의 `val_z_visual_err_pred` (최종 epoch 15; 003 보고서 4.5절, csv): mix_adjacent_s0 0.2062 대 s1 0.1851, s2 0.1815 (stats: 두 대조 평균 대비 +12.5%). single_a026_s1 0.1310 대 s0 0.1205, s2 0.1329 (+3.4%). 비붕괴 16개 run 의 같은 분포 나머지 평균 대비 상대 편차 sd 7.6%, 최대 절대값 11.4%이므로 +12.5% 는 약 1.6 sd 로 유의하지 않다(stats 4절). 이 값은 첫 배치 기준이다(skeptic).
- 근사 z 손실(val_loss 에서 첫 배치 재구성 MSE 를 뺀 값) epoch 15: mix_adjacent_s0 0.1873, 대조 0.1874, 0.1870. single_a026_s1 0.1277, 대조 0.1262, 0.1263 (skeptic 중요 2 `[계산]`, 첫 배치 값을 뺀 근사라 오차 있음).
- predictor pos_embedding 의 exp_avg_sq 평균(epoch 3, 파라미터 하나): mix_adjacent_s0 1.74e-10, 대조 1.71e-10, 1.67e-10. single_a026_s1 7.18e-11, 대조 6.91e-11 (skeptic).

### 4.6 후속 붕괴 수 예측 (stats 5절, `collapse_forecast.csv`; 설정 불변 가정)

- 붕괴 정의(003 의 collapsed 플래그, 재구성 점프 비)로 2/18 = 0.111. 95% Clopper-Pearson [0.014, 0.347], Wilson [0.031, 0.328]. 003 만이면 2/12 = 0.167, 002 는 0/6. 분포별로는 mix_adjacent 1/3, single_a026 1/3, 나머지 4개 분포 0/3.
- P(1개 이상 붕괴), p = 0.111 적용: N = 3 -> 0.30, 6 -> 0.51, 9 -> 0.65, 12 -> 0.76, 18 -> 0.88 (CP 구간 적용 시 N = 18 에서 0.22~1.00, 범위가 매우 넓음). 설정(lr, 초기화 등)을 바꾸면 이 표는 쓸 수 없다. 두 붕괴 모두 epoch 3 에서 발생했으나 n = 2 이다.
- 한계: 부분 붕괴나 느린 열화는 이 비율에 안 잡힌다. 18 run 은 6개 분포 x 3 시드로 같은 파이프라인이다(stats).

---

## 5. 분석가 의견 종합

### 5.1 세 명이 일치한 점

1. **판단 기준 표는 원값과 사전 정의대로 재현된다.** stats 는 results.json 의 층별 값에서 criteria_table 을 재계산해 (1) 비해당, (2) 해당, (3) 부분 해당, (4) 비해당, (5) 비해당과 일치함을 보였고 사전 정의를 사후에 바꾼 흔적이 없다고 했다(1절). skeptic 은 hook, 층 대응, epoch 대응, 정규화를 코드와 대조해 "결론을 뒤집을 만한 측정 버그를 찾지 못했다" 고 했다. prior 는 results.json 의 criteria_table 과 층별 값을 읽었다(1절). 리포터도 criteria_table 을 직접 읽어 log.md 와 일치함을 확인했다.
2. **처음 입력 의존성이 사라지는 층은 `upsample_b.blocks.7` 이다.** 세 명이 같은 층을 인용하고 구조를 같게 읽는다(upsample_b 의 첫 ConvT(blocks.6) 다음 ReLU). prior 1절 1번, stats 1절, skeptic "측정 버그 점검" 층 대응.
3. **"큰 gradient 가 있었는가" 는 체크포인트로 확정되지 않는다.** prior("단일 스파이크와 지속적 큰 gradient 를 구분할 수 없다"), stats(Adam 은 단발 스파이크와 수십~수백 step 지속 증가를 구분하지 못함; v 는 n 과 gradient 크기를 분리하지 못함), skeptic("단일 step 인지는 여전히 구분 불가"). 사전 기록과 같다.
4. **기준 (4) 비해당은 "큰 gradient 가 없었다" 의 반증이 아니다.** prior(비해당이 리드의 서술 "아주 큰 gradient" 를 직접 지지하지는 않지만 모순도 아님; Adam 갱신은 정규화됨), stats("검정력이 낮은 기준"), skeptic("이 사건을 잡을 수 없는 기준이었다"). 이유는 Adam 갱신 크기가 정규화되기 때문이라는 점이 공통이다.
5. **디코더에서 predictor 로 가는 gradient 경로는 코드상 없다.** prior, stats, skeptic 모두 `z.detach()` 를 인용한다. 또한 **실험 E(입력 간 std)는 약한 증거**라는 점도 stats(한계 1~3)와 skeptic(치명 2)이 일치하고 prior 도 action/proprio 0 고정과 정확도 미측정을 적었다.
6. **붕괴의 원인은 확정되지 않았다.** prior(모두 추정), stats(원인 미확인; 두 run 모두 epoch 3 급등이 우연인지 공통 원인인지 구분 안 됨), skeptic("현재 미확인").
7. **n 이 작아 일반화할 수 없다.** 붕괴 2개 run, 각 1 시드, 대조 3개(stats 표본 요약; skeptic "한계"; prior 3절 대안 D "시드 의존 사건일 가능성이 더 높다(추정)"). 두 run 은 같은 실험 안에서 같은 전처리, 캐시, GPU 환경에서 나왔고 `real` 프레임 배열도 같아 독립 재현이 아니다(stats 2절).
8. **재학습과 GPU 사용은 승인 대상이다.** prior(5절 한계 4), skeptic(치명 3과 "추가로 필요한 실험" 서두), 007 README 와 hypotheses.md:350 도 같다.
9. **epoch 2 체크포인트는 두 run 모두 정상이다.** 기준 (2) 해당 층 없음(`rj: 2_dependence_lost`), 출력 std 0.0998 / 0.0862, MSE 0.0031 / 0.0129 (log). skeptic 은 epoch 2 에 전조가 없다고 덧붙인다(blocks.5 비율 0.955, 0.957 대 대조 0.959~0.961, `[확인]`).

### 5.2 리드의 잠정 해석 4문장 대 분석가 판정 (합치지 않음)

리드의 문장은 검토 대상이었다. 아래는 문장별로 세 분석가의 판정을 나란히 둔 것이다. 이 표의 어느 칸도 보고서가 고른 결론이 아니다.

| 리드의 잠정 해석 | analyst-prior | analyst-stats | analyst-skeptic |
|---|---|---|---|
| **1. "epoch 3 중 아주 큰 gradient"** | **부분 지지.** Adam 2차 모멘트가 대조보다 3~4자리 크다는 간접 단서는 있으나 체크포인트로 단일 스파이크와 지속적 큰 gradient 를 구분할 수 없다. 대안으로 "작은 v 상태에서 평범한 gradient 가 큰 정규화 갱신을 만든 것" 도 같은 지문을 낼 수 있다. 확신 낮음 | **체크포인트로 구분 불가. 일부는 반대 방향 정보 있음.** v 는 사망 후 경과 step 수 n 과 gradient 크기를 분리하지 못한다(단발 환산 1365~10743배, 100 step 지속 환산 140~1101배가 둘 다 가능). 활성값이 상류에서 40~110배 커져 있어 weight gradient 가 커진 것은 "정상적인 큰 gradient" 일 수 있다. 가중치 변화는 정상 범위 | **일시적 큰 gradient 는 강하게 지지됨. 원인인지 붕괴 중 결과인지는 미정.** exp_avg_sq 가 model_3 -> model_4 에서 순수 감쇠(유효숫자 약 4자리까지 일치)이므로 큰 값은 지속되는 gradient 가 아니라 epoch 3 안에 끝난 사건의 잔상이다. 단 1861배, 50271배는 사건 크기가 아니다. 단일 step 인지는 구분 불가. 되먹임으로 번진 불안정이라면 큰 gradient 는 원인이 아니라 과정의 일부 |
| **2. "ReLU 사망과 비회복"** | **강하게 지지.** 입력 무관 출력, 죽은 ReLU 층 특정, 이후 에폭 회복 없음(single 은 epoch 4 까지). 이는 Lu 2019 서술(prior 기억, 본문 미확인)과 일치하나 초기화 시점이 아니라 학습 도중이라 인과가 같다고 할 수 없다(추정). 정상 run 도 ReLU 대부분이 꺼져 있어(0.83~0.96) 소수 채널이 마지막으로 꺼지는 사건일 수 있음(추정) | **지지됨.** ReLU 사멸은 관측(임계값 1/3 ~ 1e-4 에서 같은 층). 비회복은 n = 2 로 지지(v 가 epoch 4 에서 0.999^4125 와 일치, exp_avg 가 float32 바닥, 003 로그에서 epoch 15 까지 고정). 두 run 의 엄격 정의 차이는 54프레임 표본의 우연에 가깝다 | **mix_adjacent_s0 는 강하게 지지됨.** single_a026_s1 은 54프레임에서만 확인되고 상태도 다르다(110개 원소 활성). **죽음은 회복 실패의 설명이지 붕괴 시작의 설명이 아니다.** "회복하지 못했다" 는 model_3(mix) / model_4(single)까지만 체크포인트로 보았고 003 이 쓴 model_latest 는 검사되지 않았다 |
| **3. "predictor 멀쩡"** | **이 실험만으로는 확인되지 않음.** 코드 구조상 영향 없을 것이라는 추론과 가중치 변화 비해당은 지지 요소. 단 003 에 mix_adjacent_s0 의 val_z_visual_err_pred 가 형제보다 약 12% 높다는 기록이 있어 "멀쩡" 을 단정하기 어렵다. "구조상 분리되고 가중치 변화가 작다" 까지만 | **코드 분리로 개연성은 높으나 E 는 약한 증거.** 코드 사실(z.detach())이 가장 강한 근거. E 는 상수 붕괴를 잡을 뿐 정확도를 재지 못한다. +12.5% 는 약 1.6 sd 로 시드 변동 범위로 설명 가능하나 검정력이 낮다. "predictor 가 붕괴 run 에서 더 나쁘다는 증거도 같다는 확증도 아님" | **과장.** 말할 수 있는 최대치: "디코더 붕괴가 gradient 로 predictor 에 전달될 경로는 없고 z 손실은 대조와 큰 차이가 없다." 물리 예측이 정상이라는 말은 할 수 없다. E 는 입력을 복사하는 함수도 통과하며 크기 정보를 재지 않는다. predictor 의 물리 예측 품질은 한 번도 관측되지 않았다 |
| **4. "디코더만 재학습하면 측정을 살릴 수 있다"** | **코드 구조상 타당(조건부).** 입력이 `z.detach()` 이고 encoder 가 동결이라 predictor 를 그대로 두고 디코더만 새로 학습해도 predictor 예측 latent 디코딩은 복구 가능하다고 본다. 조건: 학습 데이터 필요, 비교하려면 정상 run 도 같은 디코더로 재디코딩, 재붕괴 위험, GPU 승인. 선행 사례는 미확인 | **판정 없음.** stats 파일은 이 문장을 다루지 않았다(붕괴 빈도와 후속 붕괴 수 예측만 제시) | **근거 없음.** 바로 쓸 코드 경로가 없고(train.py:207-210, :274 가 저장된 디코더를 그대로 로드), 체크포인트에 데이터와 인코더가 없으며, 비교 가능성이 검증되지 않았고, 재학습으로 얻는 이득이 작다(003 전체 판정 불변). 재학습 전에 디코더 교체 null 검사가 먼저 필요 |

### 5.3 의견이 갈린 점 (하나로 합치지 않음)

| 쟁점 | analyst-prior | analyst-stats | analyst-skeptic |
|---|---|---|---|
| **큰 gradient 지지 강도** | 부분 지지, 확신 낮음. 간접 단서(v 배율 1.4 x 10^3 배(대조 1개 기준), single 약 3.7 x 10^4 배)를 "마지막 약 1000~4000 step 사이의 큰 gradient 구간 후 죽음" 서사의 근거로 쓰되 추정으로 표시. 단일 스파이크 여부는 구분 불가 | 체크포인트로 구분 불가. 1861배, 50271배는 44개 파라미터 중 선택된 최댓값이라 대표값으로 쓰면 과장된다. 배율을 크기로 읽으면 안 되는 이유로 n 과의 혼동을 든다 | 일시적 큰 gradient 는 강하게 지지(순수 감쇠 일치, 확신 높음). 사건 뒤 짧았다는 쪽(원소당 rms 변화 blocks.6.bias 0.0122, 0.0145 가 단일 step 이동 상한 약 0.0095 에 가까움, `[추정]`). 사건 크기는 배율로 환산할 수 없다 |
| **지속 기간 판단의 근거** | 단일 스파이크 대 지속적 진동은 구분 불가 | 실제 이동량(원소당 rms 중앙값 0.0076 / 0.0078 대 대조 0.0044~0.0048)은 정상 학습에서도 Adam 단일 step 상한을 넘는 경우가 있어 단발 스파이크 가설을 판별하는 용도로 쓰지 않는다 | 같은 이동량을 "한 step 부터 몇 step 정도의 짧은 사건과 맞고 긴 폭주와는 맞지 않는다" 로 쓴다. 정상 학습분을 빼면 상한에 가깝다고 본다. stats 와 해석이 다르다 |
| **blocks.7 이 원인 층인가 증상 층인가** | blocks.7 을 "dying ReLU 가 일어난 층" 으로 서술하되 원인 층이라고 단정하지는 않는다. 대안 C: 훨씬 앞의 `upsample_b.blocks.0.bias`(상대 변화 0.75, 대조 약 0.10~0.12)의 큰 이동이 하류 ReLU 입력 평균을 음으로 밀었을 수 있다(미확인) | 기준은 "처음 사라지는 층" 이지 "원인이 있는 층" 이 아니다. blocks.7 이 첫 층인 것은 구조적 결과(ReLU 출력 거의 0 이면 뒤 층 전부 상수). 규모 이상은 blocks.0 에서 이미 시작해 깊이에 따라 단조 증가하므로 "우연히 한 층이 튀었다" 가능성은 낮다. 관측과의 정합성일 뿐 원인 식별이 아니다 | **증상이 처음 드러나는 층일 가능성이 높다**(치명 1, 확신 중간). blocks.7 앞 층은 죽지 않고 커졌다(rms epoch 2 -> 3, blocks.0 7.53 -> 29.28, blocks.4 8.29 -> 87.06, blocks.5 0.93 -> 35.88, blocks.6 0.96 -> 114.77, blocks.7 0.27 -> 0.0, `[확인]`). 앞단 증폭과 blocks.6 커널 부호(평균 -0.00400, -0.00295 대 대조 -0.00181)의 조합일 수 있다(`[추정]`). blocks.6.bias 변화는 무시할 만해 bias 설명은 약하다. 활성화 평균이 저장되지 않아 직접 확인 불가 |
| **두 run 이 같은 상태인가** | 두 run 을 같은 상태로 취급 (single 은 소수 채널만 살아 있음, "엄격 정의에는 미달이나 사실상 같은 상태") | 기능적으로 같은 상태(출력 이미지 std 0.0 / 1.6e-4, 대조 0.099). 엄격 정의로 가르는 것은 54프레임 표본의 우연에 가깝다 | **같은 상태가 아니다**(중요 1). single_a026_s1 에는 exp_avg 비정규수 바닥 흔적이 없고 활성 원소 110개를 통한 gradient 가 있는 것으로 보인다. 두 붕괴는 같은 상태가 아니다 |
| **디코더만 재학습의 타당성** | 코드 구조상 타당(조건부). predictor 는 최종 체크포인트에서 가져오고 디코더만 새로 학습. 단 기존 정상 run 도 같은 디코더로 재디코딩해야 비교 가능. 선행 사례 미확인 | 판정 없음 | 근거 없음. train.py:207-210/:274 때문에 새 디코더가 만들어지지 않고 env.decoder_path 가 쓰이지 않음. 003 패치(train_resume.patch:85-92)는 붕괴 후 decoder_optimizer 상태를 새 optimizer 에 넣음. train_predictor=False 로 돌리면 이후 저장본에 predictor 가 빠짐(train.py:162-166). 같은 시드로 돌리면 같은 붕괴가 재현될 수 있고 시드를 바꾸면 다른 디코더. 두 run 만 다른 절차로 만들면 사전 등록된 12 run 과 동등한 지위가 아님. (`[확인]` 및 `[추정]` 혼재, 파일 경로는 skeptic 이 읽은 고정 commit 사본 기준) |
| **예방책과 비교 가능성** | 예방책 5종(gradient clipping, 활성화 교체, 정규화, lr 하향/웜업, 자동 감지 후 재시작)을 근거 강도(중간/약함, 모두 문헌 본문 미확인)와 함께 나열. 공통 결론: 예방책은 "다음 학습" 에 대한 것이며 디코더 구조나 설정을 바꾸면 디코더 의존 지표(HSV 면적 기울기)는 기존 16개 정상 run 과 다른 판독기를 쓰는 셈이라 같은 표에 넣을 수 없다(추정). clipping 은 Adam 정규화 때문에 갱신 크기를 직접 제한하지 않으며(추정) v 부풀음 효과 완화일 수 있다. 가장 안전한 것은 붕괴 자동 감지와 재시작. 도입 여부는 사람 결정 | 예방책 비교는 하지 않았다. 설정을 바꾸면 후속 붕괴 수 예측표를 쓸 수 없다고만 명시. 같은 설정이면 3 시드 셀 하나에서 붕괴 0개일 확률 약 70% (4.6절). 붕괴 시드는 시드 단위라 셀 내 비교를 오염시킨다 | 예방책을 추천하지 않고 **원인 구분 실험의 일부**로 clip 반사실을 제안(epoch 3 재현 후 grad clip 또는 해당 배치 제외 반사실로 필요조건 확인). 재현되지 않으면 비결정성이나 하드웨어 쪽으로 본다. 재현 실험은 원본이 휴지통에 있을 때만 가능 |
| **내부 불안정 대 외부 충격** | 우연/시드 의존 쪽이 더 높다고 추정(대안 D). Adam 스파이크, 죽은 뉴런/포화는 모두 추정 | 구분 안 됨. 2/18 로 표본이 작아 코드/환경 귀속 불가 | 내부 불안정(lr 3e-4, clip 없음, 정규화 없음)이 외부 충격보다 그럴듯하다는 기울기이나 확신 낮음. epoch 2 에 전조가 없고 왜 이 두 run 만인지 설명 못 함 |

### 5.4 한 명만 지적한 점

- prior 만:
  - 대안 설명 6개(점진적 사멸 A, 작은 v 가 부르는 큰 정규화 갱신 B, blocks.0.bias 의 큰 이동 C, 데이터/배치 특이성 D, 수치 정밀도(AMP 여부는 체크포인트로 구분 불가) E, "predictor 멀쩡" 반론 F)와 각 대안의 구분 단서. 대조 model_2 대 model_3 의 층별 ReLU 비활성 비율 변화는 보지 않았다(미확인).
  - 문헌 후보(Lu 2019, Kingma & Ba 2015, Molybog 2023, Pascanu 2013 등)는 모두 본문 미확인이다. 같은 설정의 상수 붕괴 보고는 문헌에서 찾지 못했다(미확인).
  - 에폭당 4125 step 은 파일 번호와 step 의 대응에서 계산한 추정이라고 적었다. stats 는 decoder_optimizer step 이 epoch x 4125 임을 로그에서 확인했다고 적었다(log 도 같다).
- stats 만:
  - 임계값 민감도(1/3 ~ 1e-5), (4) null(대조 최대 비 1.23~1.58 대 붕괴 6.10, 3배 문턱이었다면 해당), 층 범위 밖 개수의 정보량 없음(45/46 of 50층), 54프레임에서 양수 프레임 비율 95% 상한 약 5.5%(3/54, 독립 가정 불성립).
  - exp_avg 비정규 바닥 4 단위의 정수 고정점 계산(1,2,3,4 단위 고정, 5 -> 4)과 gradient 0 지속 하한 830~920 step.
  - 후속 붕괴 수 예측표와 Wilson/Clopper-Pearson 구간(4.6절).
  - 붕괴 run 안에서 시드 변동과 효과를 분리할 수 없다. 대조 0/3 의 사망 확률 95% 상한 약 63% 로 대조 3개는 거의 정보가 없다.
- skeptic 만:
  - **치명 1~3**: 증상 층 가능성(5.3절), predictor 과장, 재학습 경로 부재.
  - **중요 5**: single_a026_s1 의 활성 원소 110개가 어떤 프레임에 있는지 저장되지 않았다(run.py:283 은 채널 합만 저장). 학습 크기 a026 프레임에서도 살아 있는지 미확인이며 학습 밖 큰 크기에서만 활성일 수도 있다.
  - **중요 6**: 회복 확인 범위가 model_3/4 까지, model_latest 미검사. single_a026_s1 은 model_3 -> 4 에서 앞단 상대 변화가 blocks.0.weight 0.021, blocks.6.weight 0.00099 로 거의 멈췄다. 큰 exp_avg_sq 가 유효 학습률을 누른 것인지 입력 희소성인지 구분 안 됨.
  - **중요 7**: 대안 설명 구분표. 배치 순서 공유(shuffle=False, 시드와 데이터가 달라 해당하기 어려움 `[추정]`), 혼합 정밀도(003 의 py, sh, yaml, md, txt 에 mixed_precision, accelerate launch grep 결과 없음, 노트북은 grep 범위 밖), Colab 메모리 오류(returncode 0, n_attempts 1, n_resumes 0, NaN/Inf 0). 세션: 두 run 은 약 하루 차이이고 각 붕괴 run 종료 2~3분 뒤 정상 대조가 시작됐다(시각 근거는 6절 modtime 표기 문제에 의존, `[추정]`).
  - **사소 3**: `patch_embed.weight` (n = 10)의 exp_avg 와 exp_avg_sq 가 11개 체크포인트 모두 정확히 0 이다(`[추정]` 학습 proprio 입력이 0 이라면 설명됨). 이름 충돌(6절)과 같은 항목이며 21개 건과 별개이므로 섞지 말 것. 리포터는 이 항목 1개를 `rj` 9630-9641행에서 직접 읽었고 11개 체크포인트 전수는 대조하지 않았다.
  - **사소 5**: 사전 기록과 구현의 차이: 대조 값을 평균으로 씀(hypotheses.md:356 은 "대조 run 값", README 는 평균 명시), 층 단위가 아니라 파라미터 단위(run.py:557), model_4 를 대조 epoch 3 과 비교(사전 기록에 없음), 혼합 대조 풀(mix 2, single 1). (2)는 차이가 4자릿수라 이 선택에 흔들리지 않을 것이다(`[추정]`).
  - **사소 7**: single_a026_s0 이 대조에서 빠진 이유가 README, config 에 없다(재개와 혼합 GPU 이력 때문으로 보임 `[추정]`). 대조 수가 분포별로 2, 1 로 불균형.
  - 재현성(중요 8): 경로가 scratchpad 고정, 슬림에 텐서 없음, 휴지통 보존 기간은 추정(log 164행).

---

## 6. 한계와 위험

### 6.1 정정 표시 (고치지 않음; 리포터는 results.json, 코드, 실험 파일을 수정하지 않았다)

| # | 대상 | 확인된 사실 | 결과에 대한 영향 |
|---|---|---|---|
| 1 | **notes/log.md 007 항목 176행의 "mix_adjacent_s0 model_3 의 decoder_optimizer exp_avg 가 21개 파라미터에서 정확히 0"** | stats 3절과 skeptic 중요 1 이 모두 "정확히 0 이 아니다" 라고 지적. 리포터 확인: `rj` 6991-6999행의 첫 항목(`upsample_b.blocks.0.weight`, step 12375)은 exp_avg `norm 0.0`, `mean 0.0`, `std 5.605193857299268e-45`, `absmax 5.605193857299268e-45`, `zero_frac 0.0`. 같은 absmax 값이 6998-7458행에서 23행 간격으로 **21줄** 나타난다(리포터가 grep 으로 확인). 이 값은 float32 최소 비정규수(1.4013e-45)의 4배이며(stats: 4.0000 단위), norm 필드가 0.0 인 것은 제곱합이 float32 에서 언더플로한 결과로 추정(stats)된다. 21개는 `upsample_b.*` 중 `blocks.8.bias` 를 제외한 집합이고 blocks.7 출력이 0 이면 구조적으로 gradient 가 0 이 되는 집합과 일치한다(stats, skeptic 일치). 같은 값은 파일 전체에서 35줄 검색되며 6998-7458행 밖의 14줄은 리포터가 어느 체크포인트인지 확인하지 않았다(미확인). | `criteria_table` 은 exp_avg 를 쓰지 않으므로 판정 (1)~(5)는 불변. 정정된 서술은 오히려 더 강한 증거다: gradient 가 정확히 0 인 채 약 830~970 step 이상 감쇠했다는 하한을 준다(stats 약 850, skeptic 840~970; 모두 계산값). 한편 prior 는 리드 요약과 같은 표현("exp_avg 가 부동소수 하한까지 0", "norm 0.0, 값이 약 1e-45")을 썼으나 본문에서 비정규 하한 근처라고 구체화했다. log.md 의 기존 줄은 지우지 않고 정정 줄을 추가한다(이 보고서 작성과 함께 log.md 에 기록). |
| 2 | **results.json `drive_meta.modtime_utc` / `modtime_kst` 표기** (run.py:190-193) | skeptic 사소 1: `modtime_utc` 가 실제로는 KST 로 보이고 `modtime_kst` 는 9시간 더 밀렸다. 리포터 확인: `rj` 45180 `modtime_utc: 2026-11-07 08:35:22`, 45182 `modtime_kst: 2026-11-07 17:35:22` (mix_adjacent_s2 model_3). 같은 파일의 분석 실행 시각은 `meta.date_kst: 2026-11-07 14:47:17` (75168행)이므로 `modtime_kst` 17:35 는 분석 실행보다 미래라서 KST 로 읽을 수 없다. `modtime_utc` 를 KST 로 읽으면 5개 run 모두 train_env 시작 시각과 맞는다는 것은 skeptic 의 주장이며 리포터는 대조하지 않았다. | `criteria_table` 불변(시각을 쓰지 않음). 시각을 인용하는 서술만 영향을 받는다: skeptic 중요 7 의 세션 관련 항목(두 run 의 시각 간격, 종료 2~3분 뒤 대조 시작). 이 보고서는 해당 시각을 결과로 인용하지 않았다. 다른 문서(003 보고서, log.md)에서 이 두 필드를 인용했는지는 미확인. |
| 3 | **action/proprio encoder optimizer 요약의 이름 충돌** (run.py:139-141) | skeptic 사소 2: action, proprio encoder 파라미터 이름이 같아 per_param 이 뒤 값으로 덮인다. 리포터 확인: `rj` 9628-9675행 `n_state: 4` 인데 `per_param` 은 `patch_embed.weight`, `patch_embed.bias` 2개뿐이다. 같은 요약에서 `patch_embed.weight` 의 exp_avg, exp_avg_sq 는 `norm 0.0`, `zero_frac 1.0` (n = 10). | `criteria_table` 의 `other_modules_max_ratio`(action_encoder, proprio_encoder)는 B/B_delta 에서 오며 모듈별 계산이라 영향 없다(skeptic). **C 섹션의 action/proprio optimizer 수치는 사용할 수 없다.** 이 보고서는 해당 수치를 쓰지 않았다. |

세 건 모두 코드와 results.json 을 수정하지 않았다.

### 6.2 한계와 위험

- **표본**: 붕괴 2개(각 1 시드), 대조 3개(분포 짝은 mix_adjacent 2개, single_a026 1개), 입력 54프레임(독립 아님), 체크포인트당 추론 1회. 효과가 3~5자릿수로 커서 (2)의 층 특정은 표본 크기에 흔들리지 않으나(stats), "붕괴한 run 은 항상 이 층에서 죽는다" 같은 일반화는 불가능하다.
- **엄격 정의의 불안정성**: (3)의 해당/비해당은 54프레임에서 양수 원소가 0 개인지 110개인지로 갈린다(stats, skeptic 사소 6).
- **기준 (4)의 한계**: Adam 정규화 때문에 "10배" 기준은 단일 큰 gradient 에 둔감하다. 한 epoch 합산이라 사건이 정상 학습 이동에 묻히고, 붕괴 후 멈춘 구간 때문에 같은 잣대로 비교되지 않는다(skeptic 중요 3 `[추정]`). 임계값 10 은 근거 수치 없이 정해졌다(hypotheses.md:354; skeptic). 이 보고서는 사전 기록된 임계값을 바꾸자고 권고하지 않는다.
- **원인 미확인**: 세 분석가 모두 붕괴 원인을 확정하지 않았다. 코드 사실은 skeptic 이 고정 commit 사본(code5a)을 읽고 확인했다고 적었다. prior 와 stats 는 A_code_facts(손으로 쓴 문자열)를 인용했다. 리포터는 코드를 열어보지 않았다.
- **predictor 정상 주장**: 코드 분리 외의 정량 근거가 약하다. E 는 action 0, proprio 0 고정이며 skeptic 사소 3 은 proprio 0 이 실제와 같을 수 있고 action 0 만 비현실적이라고 `[추정]` 한다.
- **체크포인트 범위**: 검사한 체크포인트는 epoch 2, 3(, 4)뿐이다. 003 측정에 실제로 쓰인 model_latest 와 남은 model_5/10/15 는 검사되지 않았다(skeptic 중요 6). 대조 run 의 다른 epoch 에 큰 gradient 가 있었는데 살아남았는지 알 수 없다(skeptic 중요 2).
- **접근 위험(시간 제약)**: 원본 체크포인트는 Drive 휴지통에만 있고 로컬 원본은 삭제됐다. 보존 기간은 추정(30일 일반 보존, 삭제 일자와 만료일 미확인; log 164행, 003 보고서 6절). 기한이 지나면 B, C, D 재계산과 층 교체/재현 실험의 model_2, model_3 부분이 불가능해진다(skeptic 중요 8). 슬림에는 텐서가 없어 재현 입력이 될 수 없다.
- **재현성**: 경로가 세션 전용 scratchpad 에 고정이고 tmpfs 일 수 있다. results.json meta 에 torch 버전만 있다. 007 전체가 미커밋이고 hypotheses.md 도 수정 상태라 사전 기록을 commit 으로 증명할 수 없다(skeptic; gitStatus). 파일 수정 시각 순서는 hypotheses.md < config.yaml < ... < results.json < log.md < README.md 라고 skeptic 이 읽었다(`[확인]`, Glob 시각 기준 오름차순 가정). run.py 는 figures 뒤에 수정되었고 현재 `judge()` 가 실행 당시와 같은지는 미확인이다.
- **문서 낡음**: README 의 상태 서술은 실행 기록 절이 갱신되어 있으나 skeptic 이 지적한 사전 기록과 구현의 작은 차이(5.4절 사소 5)는 README 에 없다(리포터가 README 를 읽어 확인한 것은 평균 사용 명시뿐).
- **리뷰 내 불일치(표시만)**: (i) prior 의 v 배율은 대조 1개 기준, stats/log 는 대조 3개 평균 기준이라 숫자가 다르다(4.2절). (ii) stats 와 skeptic 은 같은 f 값(0.38~0.42)을 각각 추정 시점과 상한으로 해석한다(4.4절). (iii) stats 와 skeptic 은 원소당 이동량을 단발 스파이크 판별에 쓸 수 있는지에 대해 다르게 쓴다(5.3절). (iv) skeptic 사소 4: log.md:171 의 "이 층 이후 27개 층" 은 blocks.7 자신을 포함하며 mix_adjacent_s0 의 하류 소실은 상류가 0 이면 자동 결과라 독립 증거가 아니다(stats 1절도 같은 취지).
- **리포터가 확인하지 못한 것**: results.json 의 나머지 약 75,000 줄(B, C, D, E 섹션 대부분), analysis/*.csv, 그림과 복원 시트, run.py, 고정 commit 코드. 해당 수치는 모두 log 또는 리뷰의 인용이다.

---

## 7. 다음 단계 후보 (제안만, 승인 아님; 사전 기록된 임계값은 바꾸지 않는다)

저비용, 새 학습 없음(CPU, 기존 자료):
1. **이미 있는 results.json 으로 대조 model_2 대 model_3 의 층별 ReLU 비활성 비율 변화와 blocks.6/7 주변 bias 부호 이동 비교**(prior 7절 1번). 대조도 같은 방향이면 prior 의 대안 A(점진적 사멸)가 힘을 얻는다.
2. **003 의 운영 로그(train.log)에 step 단위 loss 가 있으면 붕괴 시점 근처 확인**(prior 7절 2번; 있는지는 미확인).
3. **남은 model_5/10/15 와 model_latest 에 같은 hook 적용**, 활성화 평균과 부호, 프레임별 활성 원소 저장, 파라미터 묶음 교체(앞단만, blocks.6 만, 하류만)로 어느 묶음이 활성을 되살리는지 측정(skeptic 실험 3). 층 교체는 현재 상태의 필요조건만 알려주고 원인 주장에는 실험 4가 필요하다.
4. **디코더 교체 null 검사**(재학습 결정 전; skeptic 실험 2): (a) 정상 run 끼리 교체(mix_adjacent_s1 predictor + s2 decoder 대 s1 자기 조합)로 디코더 정체가 기울기를 얼마나 바꾸는지 측정(재학습 값의 오차 하한). (b) mix_adjacent_s0 predictor + s1 decoder, single_a026_s1 predictor + s2 decoder. 대조 범위 안이면 처음으로 predictor 정상의 근거가 생긴다. 조합 규칙과 판정 범위를 먼저 hypotheses.md 에 적을 것(skeptic). 데이터/GPU 필요 여부는 skeptic 이 "1, 2 는 GPU 또는 데이터가 필요할 수 있다" 로 적었다.

GPU 또는 데이터 필요, 사람 승인 필요:
5. **epoch 3 재현과 clip 반사실**(skeptic 실험 1, 원인 구분에 가장 강하다고 skeptic 이 평가): model_2 원본의 decoder + decoder_optimizer 상태에서 같은 시드의 slice 순서로 디코더만 epoch 3 재생, step 별 손실, 층별 gradient norm, 층별 활성 통계 기록. 붕괴 run 2개와 같은 분포 대조 1개씩. 재현되면 grad clip 또는 해당 배치 제외 반사실로 필요조건 확인. 비용은 run 당 1 epoch 이하로 skeptic 이 추정. 원본이 휴지통에 있을 때만 가능.
6. **디코더만 재학습**(prior 5절 대안, skeptic 은 근거 없음): 하게 된다면 재붕괴 시 재시도 규칙을 사전에 정하고, 정상 run 도 같은 방식으로 재디코딩해야 비교 가능하다(prior). 코드 경로를 별도 스크립트로 만들어야 한다(skeptic). 003 전체 판정은 어느 경우에도 바뀌지 않는다(003 보고서 5.1절 3번, log 163행).
7. **예방책을 후속 학습에서 쓰는 경우**(clipping, 활성화 교체, 정규화, lr/웜업, 자동 감지 후 재시작; prior 4절): 디코더 설정이나 구조를 바꾸면 디코더 의존 지표(디코딩 HSV 면적 기반 기울기)가 기존 run 과 직접 비교 불가가 될 수 있다(prior, 추정). predictor 학습은 건드리지 않고 디코더 설정만 별도 후속으로 바꾸는 방식을 prior 가 제시했다. 설정을 바꾸면 stats 의 붕괴 예측표도 쓸 수 없다.
8. **파이프라인 자동 점검**: 추출 단계에서 area_ol 이 전부 0 이거나 재구성 MSE 의 epoch 간 배율이 큰 run 을 표시(003 보고서 7절 4번, stats 8절 4번과 같은 방향; 코드 변경은 experimenter 몫). 007 의 stats 는 총 손실만으로는 분포 간 붕괴 구분이 어렵다고 003 에서 보였다(AUC 0.5625, 003 보고서 4.5절).

문서와 기록 정리(저비용):
9. results.json 에 `modtime` 필드 정정, action/proprio optimizer 요약의 이름 충돌 수정, run.py 와 slim 해시 및 numpy/torchvision 버전 기록, config 경로의 영구 보관(skeptic 중요 8, 사소 1~2). 이 항목은 재계산이 아니라 기록 보강이다. 수정은 experimenter 몫이고 이 보고서는 수정하지 않았다.
10. 후속 가설을 세운다면 새 사전 기록으로만 하고(prior H-Collapse 등 초안은 prior 가 제안; 007 prior 파일에는 후보 가설 초안이 없고 003 prior 에 있다), 이 보고서의 어떤 항목도 가설 확정이 아니다.

---

## 8. 사람이 결정해야 할 사항

1. 휴지통의 model_2/3 원본 보존(복원 또는 별도 보관)을 만료 추정(일반 보존 30일은 추정, 삭제 일자 미확인) 전에 할지. 이 결정이 7절 3, 5, 6번 실험의 선결 조건이다.
2. 원인 규명용 후속 실험(7절 4번 디코더 교체 null 검사 → 5번 epoch 3 재현)을 진행할지와 GPU/데이터 사용 승인. 디코더만 재학습(6번)은 prior 와 skeptic 의 판단이 갈리므로 4번 결과를 본 뒤 정할지.
3. 예방책(clipping 등)을 후속 학습에 넣을지: 넣으면 디코더 의존 지표가 기존 run 과 직접 비교되지 않을 수 있고(prior, 추정) 붕괴 예측표를 쓸 수 없으므로(stats), 넣기 전에 비교 방식(전 run 재디코딩 또는 분리 보고)을 먼저 정할지.
