# 실험 002_single_size_sweep 보고서

- 작성: reporter, 2026-11-05
- 입력: experiments/002_single_size_sweep/results.json (수치의 유일한 출처), config.yaml, README.md, analysis/stats_recheck_output.txt, notes/reviews/002_prior.md, 002_stats.md, 002_skeptic.md, notes/hypotheses.md (후보 G, 확정 기록 3, 5, 6, 7, 10; 참고로 확정 기록 11), notes/log.md (2026-11-04, 2026-11-05 의 002 항목).
- 이 보고서는 새 측정을 하지 않았다. 수치 출처 표기:
  - `rj:` = results.json 키 경로. 리포터가 파일에서 직접 대조한 것은 `decision_table` 14행의 status 와 Holm 조정 p, H-a 4행의 `delta_vs_mask_baseline`, H-b 와 조건 비교의 시드별 값, H-c steering 6칸, 4c 의 `max_ratio_over_layers`, `h_a_early_flat_report` 4행의 delta, `train_budgets.steps_computed` 이다.
  - `stats_recheck_output.txt` = analyst-stats 가 results.json 으로 재계산해 저장한 출력(CI, Q1 R² 표, Holm 재계산). 리포터는 이 파일의 해당 줄을 읽었다.
  - `(prior/stats/skeptic 절)` = 해당 리뷰가 인용했거나 계산했다고 밝힌 값을 옮긴 것이고, 리포터가 results.json 과 일일이 대조하지는 않았다. skeptic 의 "내 계산"은 그대로 표시한다.
- 표기: "추정" = 분석가 또는 리포터가 확인하지 못한 주장. 세 분석가는 서로의 파일을 읽지 않고 독립 작성했다.
- 이 보고서의 어떤 제안도 승인이 아니다. 사전 등록 임계값을 바꾸자는 권고는 쓰지 않았다. 사람이 결정할 사항은 9절에 모았다.

---

## 1. 한 문단 요약

cloth_sim 에서 천 크기 정보의 흐름을 인코딩(H-a), 보존(H-b), 사용(H-c)으로 나눠, 단일 크기 ind 로 학습한 조건 (a)와 a060+ind+a160 을 학습한 조건 (b)를 학습 시드 3개로 비교했다. 판정 표 14행은 pass 6, fail 3, not_evaluable 2, decided 2, exploratory 1 이고, 세 분석가는 모두 status 와 Holm 계산이 규칙대로임을 확인했다(기계적 오류 없음). 그러나 pass 가 곧 가설 지지는 아니다. 세 분석가가 일치하는 것은 다음이다. (1) H-c steering pass 는 "predictor 가 크기를 쓴다"의 근거가 못 된다(강도 표현은 분석가마다 다름). (2) H-b last_block pass 와 cond_a_vs_b:last_block pass 는 효과가 R² 0.01 미만이라 인용 가치가 낮다. (3) H-c subspace_removal 은 6칸 모두 not_removed 로 판정 불가이며 제거 효과값은 근거로 쓰지 않는다. (4) 가장 방어 가능한 관측은 rollout 마지막 시점의 읽힌 크기 R² 하락(조건 (b) 0.291/0.295/0.321, 평균 0.302 > 임계 0.2)과, 조건 (a)의 하락(0.739/0.706/0.729)이 (b)의 어느 시드보다 크다는 것이다. 이 하락이 "정보 소실"인지 "인코더 토큰으로 학습한 판독기가 예측 latent 에 전이되지 않는 것"인지는 세 분석가 모두 구분할 수 없다고 본다. 분석가가 갈리는 곳은 H-a 보간 pass 의 의미(stats 는 linear 에서 강한 지지, skeptic 은 가설 근거 아님, prior 는 부분)와 steering/last_block 을 "무효"로 부를지 "약한 증거"로 부를지다. CI 는 모두 평가 에피소드 재표집이며 학습 시드 변동(시드 3개)을 담지 않는다. 확신 수준: 계산의 정확성 높음, rollout 하락과 (a)-(b) 차이의 존재 중간~높음, 각 가설(보존/사용)의 결론은 낮음. 분석가별 확신도는 prior 중간-낮음, stats 중간, skeptic 중간~높음(전체 결론 "pass 3종은 가설을 지지하지 못한다" 기준)이다.

---

## 2. 배경과 가설

- 배경: 001 에서 ind 단일 학습 baseline 의 먼 크기 open-loop 면적이 ind 쪽으로 이동했으나, 대조군 부재와 판정 단위 미사전등록 때문에 원인 층위를 분리하지 못했다(reports/001_size_collapse_diagnosis.md). 002 는 후보 G 로 이 한계를 사전 등록에서 해소하려는 실험이다(hypotheses.md 후보 G).
- 가설(확정 기록 4, 2026-11-02, 사용자 승인): Q1 인코딩 H-a(크기를 mask 면적 baseline 이상으로 복원), Q2 보존 H-b(마지막 블록과 rollout 후반까지 Q1 성능에 가깝게 유지), Q3 사용 H-c(크기 방향 이동/제거가 무작위 방향보다 효과가 큼, alpha 에 단조).
- 사전 등록(확정 기록 3, 5, 6, 7): 아래 3절에 config 기준으로 정리. 결과를 본 뒤 바꾸지 않는다. "사전 등록 후 변경" 3건(Q1 주 타깃, 단조성 기준 1.0 -> 0.9, 조건 (a) 에피소드 300 -> 900)은 결과를 보기 전에 기록되었다(확정 기록 6, 7; stats 6절 5번과 prior 1.3 이 이 사실을 확인).
- 후보 G 의 순서 규칙: "Q1 이 음성이면 Q2 이후로 가지 않는다"(hypotheses.md:159). 외삽 2행이 fail 이었으나 무인 실행으로 전 단계가 돌았고 이 규칙 적용 여부는 기록되지 않았다(skeptic S6).

---

## 3. 설정과 방법 (config.yaml 기준)

| 항목 | 값 | 출처 |
|---|---|---|
| 조건 (a) single_ind | ind 900 에피소드 | config.yaml:29 (사전 등록 후 변경: 원래 300) |
| 조건 (b) multi_60_ind_160 | a060, ind, a160 각 300 에피소드 | config.yaml:30 |
| 보조 multi_60_ind_160_small | 실행 안 함. diag_4c 행은 not_evaluable | log.md 2026-11-05; `rj: decision_table[diag_4c:multi_60_ind_160_small].status` |
| 학습 시드 | [0, 1, 2] | config.yaml:38 |
| 학습 | batch 24, num_hist 1, 15 epoch, frameskip 5 | config.yaml:41-43, 112 |
| 동일 예산 | 6개 run 모두 steps_computed 27855 (계산값이며 optimizer 가 센 값이 아님) | `rj: train_budgets` (직접 확인); README 9절 |
| 평가 | 크기당 test 100 에피소드. 보간 a080, a125, 외삽 a026, a040, a200, a248 (stats 3.1절: 보간 200, 외삽 400 에피소드) | config.yaml:18; hypotheses.md 후보 G; stats 3.1 |
| Q1 주 타깃/창 | log 명목 면적비(에피소드 상수), 후반 프레임 late_deformed [8-11]. 초기 early_flat [0-3] 은 판정 없이 보고 | config.yaml:129-130, 155-159 |
| Q1 probe | linear_pooled(ridge), attentive. 통제: random-init 인코더, 라벨 셔플, mask 면적/bbox ridge baseline. probe 시드 0-4 | config.yaml:47, 147; README |
| Q1 판정 | R² 가 mask baseline 보다 0.1 이상 높고 episode bootstrap CI 하한 > 0, random-init 도 이김(확정 기록 3, 5). H-a 는 4개 주 검정(보간/외삽 x linear/attentive), Holm | `rj: prereg_values_used`; hypotheses.md 확정 기록 5 |
| Q2 판정 | Q1 천장 R² 대비 하락 >= 0.2 이면 보존 실패. 지점 2개(last_block, last_rollout_step). 출력 공간 probe: Q1 에서 인코더 토큰으로만 학습한 probe 를 예측 z 에 그대로 적용 | `rj: prereg_values_used.q2_drop_threshold`; 확정 기록 6-4a |
| Q3 판정 | 무작위 100방향의 95 백분위 밖, alpha 7점(-2~2 sigma_u)에서 단조(Spearman >= 0.9), 층 middle/last 고정. 주 검정은 steering(ridge 방향)과 INLP 제거. LEACE, 평균차 방향은 탐색 | `rj: prereg_values_used`; 확정 기록 5, 6-5 |
| 제거 확인 | mlp_r2_after_max 0.1, inlp_iterations 10, 평가 25 에피소드/레벨 | config.yaml:187-190 |
| 통계 | 에피소드 단위 bootstrap 95% CI, Holm(패밀리 H_a 4, H_b 2, H_c 2, cond 2, 패밀리 간 보정 없음) | `rj: ci_caveat`, `prereg_values_used.holm_alpha` |
| 4c 진단 | 활성 이동량/noise floor 2배 미만 작음, 2~5배 중간, 5배 이상 큼. 다음 프레임 오차비 2 이상이면 큼. (a)의 층별 결과 해석 규칙(작음: 정상, 중간: 주의 문구, 큼: 수치만) | 확정 기록 7 A |
| 실행 환경 | 서버에서 Google Colab Pro+ 로 변경(확정 기록 10). 학습 재개용 train.py 패치 사용. 사전 등록 값은 불변 | hypotheses.md 확정 기록 10 |

실행에 관한 주의(log.md 2026-11-04, 11-05):
- 학습 6개(multi_60_ind_160 x 3, single_ind x 3)가 Colab 에서 완료되었다. multi s0 는 epoch 5, s1 은 epoch 9 에서 체크포인트 재개가 들어갔고, s2 와 single_ind 의 재개 여부는 이 보고서가 확인한 문서에 기록이 없다(skeptic 9절도 single_ind 재개 여부를 확인하지 못했다고 적었다). 재개 후 데이터 순서와 난수 상태는 복원되지 않아 중단 없는 run 과 bit 단위로 같지 않을 수 있다(추정).
- 6개 run 모두 패치로 `val()` 의 `@torch.no_grad()` 줄이 빠진 train.py 를 썼다. 가중치 영향 없음은 추정이며 검증하지 못했다(skeptic S1 은 `main` 체크아웃 기준으로 영향 경로를 못 찾았으나 고정 commit 과 패치 후 파일을 이 머신에서 읽지 못했다).
- results.json 의 `environment`(python 3.9.25, torch 2.0.1+cu117, cuda_available false)와 `git`(commit d8206d6, dirty true)은 학습 런타임이 아니라 06:04 UTC 에 Drive 입력으로 aggregate 를 재실행한 런타임 값이다(log.md; `rj: environment, git`). 학습 환경의 GPU 종류와 코드 버전은 이 파일로 알 수 없다.

---

## 4. 결과

모든 CI 는 평가 에피소드 재표집 bootstrap 이며 학습 시드 변동을 포함하지 않는다(`rj: ci_caveat`). 학습 시드는 3개(자유도 2)다. 아래 "원 p"의 하한은 bootstrap B=1000 의 1/1001=0.000999, 순위 검정은 1/101=0.0099 다(stats_recheck_output.txt). 하한에 붙은 p 는 정밀한 값이 아니다(stats 2절).

### 4.1 판정표 (규칙대로 계산된 status)

| id | status | 핵심 수치 | 출처 |
|---|---|---|---|
| H_a:interp:linear_pooled | pass | ΔR²(mask) 0.5560 [0.3517, 0.7777], Δ(random-init) 0.3399 [0.2060, 0.4886], Holm p 0.003996 | `rj: decision_table[H_a:interp:linear_pooled]` (delta, p, status 직접 확인); CI 는 stats_recheck_output.txt |
| H_a:interp:attentive | pass | ΔR²(mask) 0.2246 [0.0153, 0.4545], Δ(random-init) 0.1999 [0.0650, 0.3441], Holm p 0.029970 | 같은 방식 |
| H_a:extrap:linear_pooled | fail | ΔR²(mask) 0.0339 [0.0073, 0.0672] (< 마진 0.1), Δ(random-init) 0.0697 [0.0528, 0.0866], Holm p 0.020979 | 같은 방식 |
| H_a:extrap:attentive | fail | ΔR²(mask) -0.1245 [-0.1512, -0.0901], Δ(random-init) 0.0801, Holm p 1.0 | 같은 방식 |
| H_b:last_block | pass | 시드별 drop -0.00348, -0.01373, -0.00831, 평균 -0.00851, CI [-0.01062, -0.00623], Holm p 0.001998 | `rj: decision_table[H_b:last_block]` (직접 확인) |
| H_b:last_rollout_step | fail | Q1 천장 0.8203, Q2 R² 0.5290/0.5250/0.4998, 시드별 drop 0.2913/0.2953/0.3205, 평균 0.3024, CI [0.2822, 0.3240], 3시드 모두 임계 0.2 초과 | `rj: decision_table[H_b:last_rollout_step]` (직접 확인) |
| H_c:steering | pass | 6칸(시드 3 x 층 middle/last) 모두 크기 방향 효과가 무작위 100방향 95 백분위 밖, p_rank 0.0099(하한), Spearman 1.0. T(크기 방향) middle 0.0869/0.0875/0.0806, last 0.0644/0.0541/0.0645. 95 백분위 값 middle 0.0123/0.0111/0.0124, last 0.0104/0.0094/0.0104. Holm p 0.0198 | `rj: decision_table[H_c:steering].result.cells` (직접 확인) |
| H_c:subspace_removal | not_evaluable | INLP 6칸 모두 not_removed. LEACE(탐색)도 전 셀 removed=false (prior 4.1, stats 8절, skeptic I5) | `rj: decision_table[H_c:subspace_removal].status` |
| cond_a_vs_b:last_block | pass | (a)-(b) change_vs_layer0 차 -0.00817 [-0.01026, -0.00613], (a) 시드별 +0.00625/-0.00586/-0.00183, (b) +0.00263/+0.01293/+0.00752, Holm p 0.003996. 4c 게이트 (a) medium | `rj: decision_table[cond_a_vs_b:last_block]` (직접 확인) |
| cond_a_vs_b:last_rollout_step | pass | drop (a)-(b) 0.4221 [0.3986, 0.4456]. (a) Q2 R² 0.0818/0.1144/0.0914, drop 0.7385/0.7059/0.7289. (b) 위와 동일. Holm p 0.003996 | `rj: decision_table[cond_a_vs_b:last_rollout_step]` (직접 확인) |
| diag_4c:single_ind | decided: medium | a060 최대 활성 이동/noise floor 4.912/4.662/4.657 (시드 0/1/2), a160 3.559/3.576/3.570. 다음 프레임 오차비(ind 대비) a160 1.744, 1.727 (시드 0, 1), a060 1.084, 1.053 (시드 0, 1) | `rj: decision_table[diag_4c:single_ind]` (직접 확인; 시드 2 의 오차비는 직접 보지 않음) |
| diag_4c:multi_60_ind_160 | decided: large | a060 5.037/5.054/5.301, a160 4.486/4.564/5.024 (시드 0/1/2). 임계 5.0 을 a060 세 시드와 a160 시드 2 가 넘음. 오차비는 0.89-1.12 (prior 5절) | `rj` (직접 확인; 오차비는 prior) |

- 판정 개수: pass 6, fail 3, not_evaluable 2(H_c:subspace_removal, diag_4c small), decided 2, exploratory 1 = 14행(stats 1절, log.md 와 일치). 이 개수에서 fail 3은 H_a 외삽 2 + H_b rollout 1 이다.
- Holm 재계산: stats 가 자체 구현으로 10개 p 전부 재현(불일치 0)(stats_recheck_output.txt 1-12행). 판정 규칙(H_a: Δmask >= 0.1 AND CI 하한 > 0 AND random-init 도 이김)도 stats 가 4행 모두 재적용해 저장값과 일치함을 확인했다.
- p 의 정의: H_a 의 p 는 "Δ <= 0" 귀무에 대한 것이고 마진 0.1 은 점추정 비교로만 들어간다. 그래서 extrap linear 는 Holm p=0.021 로 0 과 다르다는 기각이 되지만 마진 미달로 fail 이고, interp attentive 는 Holm p=0.030 이지만 CI 하한 0.0153 이 마진 0.1 아래다(stats 1절).

### 4.2 보조 수치

- Q1 후반 창 R² (point, stats_recheck_output.txt 25-53행): 보간 linear real 0.5137, random-init 0.1737, mask baseline -0.0423. 보간 attentive 0.1823, random-init -0.0176. 외삽 linear 0.8588, random-init 0.7892, mask 0.8250. 외삽 attentive 0.7004, random-init 0.6203. 라벨 셔플은 약 0(보간 linear -0.0268, 외삽 linear -0.0117).
- Q1 후반 창 MAE (log 면적 단위, 같은 출력): 보간 linear 0.1180 / mask 0.1666 / random-init 0.1521, 보간 attentive 0.1776, 외삽 linear 0.2806 / mask 0.3185, 외삽 attentive 0.4824.
- 초기 평평 프레임(판정 없음, `rj: h_a_early_flat_report`): ΔR²(mask) 보간 linear 0.0966, 보간 attentive -0.4609, 외삽 linear -0.0028, 외삽 attentive -0.1913 (직접 확인). 즉 초기 창에서는 4행 모두 마진 0.1 에 못 미친다(skeptic C3 도 같은 수치).
- Q2 층별: layer 0 의 R² 는 두 조건 모두 약 0.8727 이고 마지막 블록은 (b) 0.8754/0.8857/0.8803, (a) 0.8790/0.8668/0.8709 이다. Q1 천장 0.8719 (`rj: decision_table[cond_a_vs_b:last_block]`, `[H_b:last_block]`).
- 학습 시드 간 변동(stats 4절): rollout 최종 R² (b) 시드 SD 0.0158. last_block drop (b) SD 0.0051. steering T 6칸 변동계수 0.19 (평균 0.073, SD 0.014).
- 시드 단위 통계(stats 3.2, 3.3절, 3 대 3): H_b last_rollout_step drop 시드 단위 t 95% CI [0.2630, 0.3418]. H_b last_block drop 시드 단위 t 95% CI [-0.0212, 0.0042]. 조건 비교 rollout 은 6값이 완전히 분리(최소 (a) 0.706 > 최대 (b) 0.321, Welch p<0.0001, 95% CI [0.385, 0.459]). 조건 비교 last_block 은 Welch t=-1.76, p=0.155, 95% CI [-0.0212, 0.0049]. 3 대 3 정확 순열 검정의 최소 양측 p 는 0.10 이므로 순위 기반 시드 단위 유의성은 이 설계에서 원리적으로 불가능하다(stats 3.3, skeptic I2 가 같은 값을 독립으로 계산).
- rollout 시점별(탐색, 판정 없음, stats 3.2, prior 2.1): (b) unseen linear step 1 에서 step 12 로 단조 하락(s0 0.853 -> 0.529). (a) s0 step 1 0.702, step 12 0.082. 학습 크기(trained) 레벨의 step 12 R² 는 (b) 0.540/0.054/0.183 (시드 0/1/2).
- 외삽 후반 Pearson r: extrap 0.973, unseen 0.954, trained 0.918(stats_recheck_output.txt). extrap 에서 r²(0.947)와 R²(0.859)가 벌어지는 점을 prior 가 지적했다(1.2절, 해석은 추정).

---

## 5. 분석가 의견 종합

### 5.1 세 명이 일치한 점

1. 판정 표의 status, Holm 조정 p, 시드 평균 등 계산은 규칙대로이며 기계적 오류를 못 찾았다(stats 1, 2절; skeptic 0절; prior 0절). 동일 예산(27855 step)도 확인(prior 3절, stats 4절, skeptic S11).
2. H-c steering pass 는 "predictor 가 크기를 예측에 쓴다"의 근거로 쓸 수 없다. 효과 측정에 쓰는 판독기와 개입 방향이 같은 크기 회귀에서 나와 정렬이 설계상 예상된다(prior 4.2-1, stats 8절 H-c (i), skeptic C1). 그리고 단조 기준(Spearman >= 0.9, 7점)과 순위 p 하한(0.0099)은 변별력이 포화되어 있다(prior 4.2-3, stats 8절 (ii), skeptic C1).
3. H-c subspace_removal 은 not_evaluable 이다. 제거 효과값(T 0.38-0.70)은 H-c 근거로 쓰지 않는다. 사용자 결정(log.md 2026-11-05)과 같다(prior 0절, stats 8절, skeptic I5).
4. H-b last_block 은 효과가 작고 정보량이 적다. 층별 변화가 ±0.013 이내(prior 2.3-1, stats 3.2, skeptic C2).
5. cond_a_vs_b:last_block 의 pass 는 "차이 검출"일 뿐 실질 차이가 아니다. 효과 R² 0.008, 시드 범위가 겹친다(prior 3절, stats 3.3, skeptic I2). 이 행을 결론에 쓰지 않는 것이 맞다는 점에서 일치한다.
6. H-b last_rollout_step 의 하락은 3시드 모두에서 같은 방향이고 임계 0.2 를 넘는다. 그러나 "정보 소실"과 "판독기 전이 실패(분포 이동)"를 구분할 수 없다(prior 2.3-2, stats 3.2, skeptic I1). 예측 z 로 판독기를 재학습한 셀은 삭제되어 구분 수단이 없다(skeptic I1, README:206).
7. (a) 대 (b) rollout 하락 차이(0.4221)는 6개 값이 겹치지 않아 방향과 크기가 신뢰할 만하다(prior 3절, stats 3.3, skeptic I3). 동시에 학습 에피소드 구성이 함께 바뀌는 점(크기 다양성과 범위 포함)과 시드 3개, 재개 run 혼재 때문에 인과("다양한 크기 학습이 보존을 만든다")는 말할 수 없다(prior 3절, skeptic I3, stats 10절 7).
8. 4c 진단은 "낯섦"을 재는 지표로 변별력이 낮다. (b)에서 a060 은 학습한 크기인데도 (a)보다 큰 값을 낸다. layer 0 비율이 두 조건에서 3.82 로 같다(prior 5절, stats 5절, skeptic I4). large 규칙 문구는 (a) 층별 결과에 관한 것인데 large 는 (b) 행에서 나왔다(log.md 도 같은 지적).
9. linear probe 의 probe 시드 5개는 값이 완전히 같아 독립 반복이 아니다(prior 1.3, stats 3.1, skeptic S5).
10. 학습 시드 3개이고 CI 가 시드 변동을 담지 않으므로 미세 효과의 유의성을 시드 단위로 주장할 수 없다(prior 2.3-4, stats 6절, skeptic I2).
11. 크기 라벨이 에피소드 상수라서 그리퍼 위치 같은 상관 단서가 "크기"로 읽혔을 가능성은 001 에서 제기되었고 002 에서도 검사되지 않았다(prior 1.3, 4절 7번; skeptic C3 마지막 항목, 추정).

### 5.2 의견이 갈린 점 (하나로 합치지 않음)

| 쟁점 | analyst-prior | analyst-stats | analyst-skeptic |
|---|---|---|---|
| H-a 보간 pass 의 의미 | mask baseline 이 후반 프레임에서 붕괴(-0.0423)한 영향. "면적 이상의 정보"의 증거로 읽을 수 있으나 "DINOv2 특이적" 증거는 약함(random-init 대비 +0.34, 외삽은 +0.07). 조건부로 쓸 수 있는 주장으로 분류 | 보간 linear 는 "강하게 지지"(CI 하한 0.3517 > 마진 0.1, 사전학습 기여 0.340). 단 random-init 만으로도 mask 를 넘으므로(0.1737 대 -0.0423) "사전학습이 있어야만"은 아님 | 가설 주장의 근거가 못 됨. 보간/외삽 갈림은 R² 분모 차이(보간 라벨 분산 0.049, 외삽 0.966, 내 계산)로 대부분 설명. MSE 이득은 보간 0.028, 외삽 0.033 으로 비슷(내 계산). 마스크 baseline 이 시간 정보 없이 전 프레임으로 학습되어 후반에서 불리 |
| H-a 보간 attentive pass | 하한 0.015 로 경계. attentive < linear 는 3값 라벨과 외삽 한계 때문일 수 있음(추정) | 사전 등록 기준은 충족하나 마진을 CI 로 못 넘고 MAE 로는 mask 와 비슷, probe 시드 R² 0.108-0.271. 단 5개 probe 시드 각각의 Δmask 는 모두 0.1 이상(0.246, 0.152, 0.157, 0.255, 0.314) | MAE 0.178 로 마스크(0.167)보다 나쁜데 pass. 가설 근거가 못 됨 |
| H-a 외삽 fail 의 뜻 | mask baseline 이 이미 0.825 라 마진 0.1 을 못 넘는 구조적 불리. 정보가 없다는 뜻이 아님 | "유의하지만 작은 효과"(Holm p 0.021, CI 상한 0.0672 < 0.1). R² 대비의 일부는 지표 성질(추정) | 외삽 R² 0.859 로 인코딩 실패가 아님. "외삽에서 못 읽는다"는 과장 |
| H-b last_block pass 의 위상 | "실패하기 어려운 시험". 잔차 연결 때문(Elhage 2021 은 기억 기반, 서지 미확인). 약한 증거로 기재 | 지지되나 효과 작음. 시드 단위 CI 가 0 을 포함 | 변별력 없음. 조건 (a)도 같은 값을 내므로((a) 의 마지막 블록 R² 0.879/0.867/0.871) 근거로 쓸 수 없음 |
| H-c steering 의 위상 | "반쪽 증거". 판독 정렬이 거의 항등적으로 효과를 낼 수 있음. 비교 대상이 임의 방향이지 다른 의미 있는 방향이 아님 | 기준 충족(T/95백분위 5.8-7.9배). 설계 순환성은 코드 읽기에 근거한 추정. 변별력 약함. 확신도 중간 | "무효"(C1). last 층 3칸은 개입 뒤 predictor 블록이 없음(q3.py, model_io.py, vit.py 읽음). 구조 확인은 높음, 크기 해석은 추정. middle 칸에는 블록 기여가 있을 수 있어 분해 실험에서 되살아날 수 있다고 인정 |
| steering 효과 크기의 척도 | T 0.087 이 큰지 알려면 실제 레벨 간 log 면적 차(예: ind 대 a160 약 0.47)와 비교해야 함. results.json 에 없음(미확인) | 해당 비교 없음 | T 0.054-0.087 은 a060 과 a160 의 로그 면적 차 0.98(내 계산)의 약 6-9% |
| rollout 하락의 크기 특이성 | (b) 의 학습 크기도 같은 방식으로 하락(step 12 에서 0.54/0.05/0.18)하고 시드 간 변동이 큼. 전체 0.30 하락을 크기 정보 소실로 귀속하면 안 됨 | 분포 이동으로 인한 probe 오보정 가능성(추정). 판정은 시드 단위로도 견고 | 학습 크기 하락(0.256/0.741/0.613, 내 계산)이 미지 크기 하락(0.291/0.295/0.321)보다 큼. 일반적 rollout 오차 누적만으로 설명될 수 있음(추정) |
| probe 종류에 따른 H-b rollout 결론 | 언급 없음 | 언급 없음(기재: 사전 등록은 linear 로 고정) | attentive 로 같은 하락을 계산하면 0.147/0.231/0.183, 평균 0.187 로 0.2 아래(내 계산, 탐색). 판정을 바꾸자는 뜻이 아니라 탐색 사실로 병기하자는 의견 |
| 4c large/medium 를 어떻게 읽을지 | large 는 임계 5.0 을 0.04-0.30 넘긴 경계값. "(a) 와 거의 같은 이동량"으로 표기. 규칙을 바꾸자는 뜻은 아님 | (a) a060 4.912 가 임계 5.0 의 1.8% 아래라 시드 한 개의 2% 안에서 게이트가 갈림. large 였다면 cond last_block 은 not_evaluable. 오차비로는 (a) 가 a160 에서 더 낯선 증거(1.73-1.74) | noise floor 의 표본 크기(50 대 50)와 비교(100 대 100)가 달라 √2 배 차이. 맞추면 (a) s0 약 6.9(내 계산, 대략값, 추정). 구현은 사전 등록 문구를 따랐으므로 규칙 위반이 아니고 임계값을 바꾸자는 뜻도 아님 |
| (a)-(b) 비교의 교란 평가 | 학습 분포 폭과 ind 데이터 양이 같이 바뀜(선언된 교환). 재개 효과는 데이터로 분리 불가 | 재개가 결과를 체계적으로 끌어올린 흔적은 없으나 n=3 으로 분리 불가 | 재개 2/3 이 (b) 에 몰림((a)의 재개 여부는 확인 못함). 행동 정규화가 조건별 학습 manifest 통계라 (a)에서는 행동도 분포 밖일 수 있음(001 의 A2). 판독기가 (b) 학습 수준으로 맞춰져 (b) 에 유리할 수 있음(추정) |
| 전체 확신도 | 중간-낮음 | 중간 | 중간~높음 (전체 결론 기준: 계산은 규칙대로이나 pass 3종은 가설 주장을 지지 못하고, 믿을 수 있는 것은 rollout 판독 하락과 조건 차이의 존재까지) |

### 5.3 한 명만 지적한 점

- prior 만:
  - persistence baseline 부재: 마지막 context 프레임 토큰을 복사한 예측이 크기 라벨을 보존하므로 Q1 천장에 가까울 것인데, 이 baseline 이 results.json 에 없다. 있으면 "rollout 예측이 persistence 보다 크기를 더 잃는다"를 말할 수 있다(추정, 미측정; 2.3-3).
  - attentive < linear 가 문헌(Punzo 2026 요약)과 반대. 3값 라벨 + 외삽 한계로 설명하는 것은 추정(1.2). 문헌 서술은 notes 의 2차 전사/요약이며 이번에 원문을 읽지 않았다(prior 상단, 10절).
  - 보간 R² 는 레벨 집합의 라벨 분산에 의존하며 MAE 는 trained 와 동급(0.118 대 0.124)이라 "보간 성능이 낮다"로 읽으면 과장(추정, 1.3).
  - 대안 설명 7개 정리(7절)와 레벨별 부호 있는 편향 미보고로 "ind 로 수렴"과 "정보 소실"을 구분 못 함(3절).
- stats 만:
  - cond_a_vs_b:last_block 은 시드 단위 Welch p=0.155 로 유의하지 않음(3.3절). 에피소드 bootstrap CI 폭 0.0041 대 시드 단위 0.0261(약 6배).
  - 보간 mask baseline R² 는 early 0.772 에서 late -0.042 로 급변. DINO linear 도 early 0.869 에서 late 0.514 로 하락(3.1절).
  - 순서 정확도(pair_order_accuracy)는 보간 attentive 0.972 로 높아 R² 단독 판정의 한계를 보임(점추정만).
  - 가설 간 다중성: 검정 10개 중 p 가 하한 근처가 아닌 것은 H_a interp attentive, extrap linear 정도이며 패밀리 간 보정은 사전 등록상 없다(2절).
  - `r2_within` 이 일부 셀에서 null(의도된 결측인지 NaN 인지 확인 못함)(7절).
- skeptic 만:
  - H-c: `random_variance_match_ok` 와 항등 점검(`identity_check_max_abs_dsize_zero_shift`)이 판정에 연결되어 있지 않고 results.json 에도 없다(I6). meandiff steering 의 T(b) last 0.397/0.402/0.386 이 middle 0.185/0.171/0.169 의 2배 이상이며 학습 라벨 3수준 표준편차 0.401(내 계산)과 거의 같다는 점이 통과 가설과 맞는다(추정). 반대 근거로 주 방향(ridge)에서는 middle 이 last 보다 크다(C1). 부정 대조(selftest)의 "ignores" 장난감은 시각 출력을 강제로 0 으로 만들어 실제 모델 구조에 대응하지 않는다(C1). 후보 G 의 문장은 "다음 latent 가 더 변한다"인데 주 지표는 크기 판독 변화이고 `T_latent` 는 results.json 에 없다(C1, grep 0건).
  - Q3 not_removed: 판정 근거 수치(`mlp_r2_after_removal` 등)가 results.json 에 없어 log.md 의 원인 추정("INLP 10회 부족, LEACE 는 선형만")을 지금 검증할 수 없다. 코드 읽기로 찾은 다른 후보: LEACE 가 스칼라 대상이라 방향 1개만 제거, 확인 평가셋이 9수준 전체, 확인은 풀링 특징인데 개입은 토큰 단위(I5; 원인은 추정).
  - H-a: 마스크 baseline 이 프레임 정보 없이 전 프레임으로 학습(`probes.py:163-168`). 초기 창 외삽 mask R² 0.987(early 셀)(C3). 보간은 2수준, 외삽은 4수준이라 수준에 대한 일반화 불확실성이 CI 에 없음. attentive 는 학습 크기에서 R² 거의 1(0.997)이고 미지 크기에서 0.70 대로 3수준 분류에 가까운 학습으로 보임(추정).
  - 재현성(6절): 학습 런타임 환경 미기록, 단계별(train, cache, q1, q2, q3) 코드 버전 미기록(`overnight.sh` 가 산출물이 있으면 건너뛰며 버전을 대조하지 않음, 추정), 재시도 이력 소실(`attempts` 전부 0, S7), 재개 정보와 패치 sha 가 results.json 에 없음, `smoke: null`.
  - 문서가 낡았다(README.md:3, 5, config.yaml:2) (S9).

---

## 6. 각 행을 믿을 수 있는 정도 (분석가별)

"규칙대로 계산됨"은 세 분석가 모두 예다. 아래는 "그 status 가 가설 주장의 근거가 되는가"에 대한 각자의 서술을 리포터가 옮긴 것이며, 등급은 각 분석가 문장에서 따온 것이지 리포터의 판단이 아니다. 어느 쪽도 임의로 고르지 않았다.

| 행 | status | prior | stats | skeptic |
|---|---|---|---|---|
| H_a 보간 linear | pass | 조건부로 쓸 수 있음. DINOv2 특이적 증거는 약함 | 강함(이 가설 중 유일하게 "강함") | 아니오 |
| H_a 보간 attentive | pass | 경계(CI 하한 0.015) | 약함 | 아니오 |
| H_a 외삽 linear | fail | 정보 없음의 뜻이 아님(구조적 불리) | 유의하지만 작음. 마진 미달은 신뢰 가능 | 제한적으로 예 |
| H_a 외삽 attentive | fail | probe 외삽 한계가 섞였을 가능성(추정) | baseline 보다 나쁨(-0.125) | 약함(probe 일반화 실패가 섞임) |
| H_b last_block | pass | 약한 증거(실패하기 어려움) | 약하게 지지(정보량 적음) | 아니오 |
| H_b last_rollout_step | fail | 하락 방향은 일관, 원인은 분리 못함 | 높음(시드 단위로도 견고) | 하락은 예, 소실은 미정 |
| H_c steering | pass | 반쪽 증거 | 기준 충족이나 변별력 약함 | 아니오(무효) |
| H_c subspace_removal | not_evaluable | 결론 없음 | 결론 없음 | 결론 없음 |
| cond last_block | pass | 통계적 검출일 뿐 실질 차이 없음 | 시드 운으로 설명될 가능성 높음 | 아니오 |
| cond last_rollout_step | pass | 차이는 방어 가능, 인과는 아님 | 낮은 우연 가능성(견고) | 차이는 예, 원인은 미정 |
| diag_4c single_ind (medium) | decided | (a) 층별 결과 해석 시 주의 문구 필요 | 변별력 낮음. 시드 하나의 2% 안에서 갈림 | 약함 |
| diag_4c multi (large) | decided | 임계 바로 위(0.04-0.30). 규칙 문구와 대상이 어긋남 | 변별력 낮음 | 약함 |

---

## 7. 한계와 위험

- 시드와 CI: 학습 시드 3개(자유도 2). 모든 CI 는 평가 에피소드 재표집이고 학습 시드, probe 학습 표본, random-init 초기화(1개) 변동을 담지 않는다(`rj: ci_caveat`; stats 6절; skeptic I2). 3 대 3 은 시드 단위 순위 유의성이 원리적으로 불가능하다(최소 양측 p 0.10).
- 재계산하지 못한 것(확인하지 못한 항목):
  1. Drive 전용 파일: q3.json(INLP 수치, 분산 맞춤 오차, 항등 점검, `T_latent`, (a) 의 ridge steering), q2_summary.json(`read_by_level`), 레벨별 R²/기울기/편향이 들어 있는 q1/cells.npz, q1/test_rows.json, train.log, epoch_logs, run_info.txt, 패치 sha, 학습 곡선. 세 분석가 모두 이 머신에 없어 재계산하지 못했다(prior 상단, stats 상단, skeptic 9절). 따라서 bootstrap CI 와 bootstrap p 는 results.json 에 저장된 값의 일관성만 점검되었고 독립 재생성되지 않았다(stats). 레벨별 부호 있는 편향(ind 로 수렴하는지)은 어디서도 확인되지 않았다.
  2. 재개 run: multi s0 는 epoch 5, s1 은 epoch 9 에서 재개되었다(log.md). 재개의 영향은 분리되지 않았다. 비재개 (b) 시드는 s2 하나뿐이다(skeptic S3). single_ind 의 재개 여부와 002 실행 중 코드 버전 혼재 여부는 확인하지 못했다(skeptic 9절).
  3. 002 학습 6개의 패치(`val()` 의 `@torch.no_grad()` 줄 누락): 가중치 영향 없음은 추정이며 검증하지 못했다(log.md; stats 4절; skeptic S1).
  4. results.json 의 `environment` 와 `git.dirty` 는 aggregate 재실행 런타임 값이다(cuda_available false, git.dirty true). 학습 때의 GPU, 드라이버, 코드 상태는 results.json 으로 확인할 수 없다(log.md; prior 10절; stats 7절; skeptic 6절).
  5. results.json 의 sha256(log.md 기록 079ff4c1...)과 config_sha256(4d5261d6...)와 현재 config.yaml 의 일치: 리포터는 해시 계산 도구가 없어 계산하지 못했다. skeptic 도 같은 사유로 대조하지 못했다(9절). 해시 자체는 `rj: config_sha256` 로 results.json 에 기록된 값이다.
  6. 고정 commit 5a640bed 의 코드: `thread0` 는 `main` 체크아웃이라 블록 구조와 val 동작을 `main` 기준으로만 읽었다(skeptic 9절). 따라서 skeptic C1 의 구조 지적(last 층에 블록 없음)도 고정 commit 원문 기준 확인은 아니다.
  7. 영상과 프레임 육안 점검, 마스크 정확도(HSV 규칙을 002 에서 재검증 안 함, README:187), early_flat/late_deformed 는 프레임 인덱스 가정(config 주석; stats 6절 6번)이다. H-a 주 판정이 late 창에 걸려 있어 이 가정이 틀리면 H-a 해석이 바뀐다.
  8. 문헌 서술은 notes 의 2차 전사 또는 기억 기반이다. prior 는 이번 세션에 논문 원문을 새로 읽지 않았다고 명시했다(Hewitt & Liang 2019, Xu 2021, Elhage 2021 등 서지 미확인). 이 보고서는 해당 인용을 근거로 쓰지 않았다.
- 설계 교란: (a) 대 (b) 는 크기 다양성뿐 아니라 학습 범위 포함, ind 데이터 양(900 대 300), 행동 분포 정규화가 함께 바뀐다(prior 3절, skeptic I3). 001 의 "가운데로 압축" 대안은 (b) 에서 줄지만 완전히 풀리지 않는다(prior 3절).
- 측정 구조:
  - 출력 공간 probe 는 인코더 토큰으로 학습한 판독기를 예측 z 에 그대로 적용하므로 하락의 일부는 분포 이동일 수 있다(세 분석가 일치).
  - 층별 probe 는 (a) 에서 안 본 크기의 활성으로 학습되어 절대 비교가 불가능하다(`rj: decision_table[cond_a_vs_b:last_block].result.limitation`).
  - Q1 상한 프레임 12 는 probe 학습 범위(프레임 0-11) 밖이다(skeptic I1, 영향은 작을 것으로 추정).
  - 일반화 단위: 크기 수준은 보간 2개, 외삽 4개뿐이다.
- 사전 등록 후 변경 3건(주 타깃, 단조성 기준, (a) 에피소드 수)은 결과를 보기 전 기록이라 부정은 아니다. 다만 mask baseline 이 원래 기준(마스크 면적 기반 타깃)에서 포화되어 H-a 가 평가 불가였을 것이라는 이유도 함께 써야 한다(prior 1.3). H-a 는 001 결과를 본 상태에서 설계되었다(확정 기록 7 B 가 명시).
- 동결 인코더 성질: Q1 에는 학습 시드가 없다(stats 4절).
- 재현성 문서: README 가 "서버 미실행, 결과 수치 없음" 등으로 낡았고, results.json 에 재개 기록과 단계별 코드 버전이 없다(skeptic S9, 6절).
- 일반화 범위: acwm-phys 설정의 DINO-WM 변형(num_hist 1, proprio 0 가능성은 001 기준), 재생성 데이터 IoU 0.68(001 에서 확인된 사실), 학습 시뮬레이션 1종. "DINO-WM 의 구조적 한계"로 일반화하는 것은 prior 가 쓸 수 없는 주장으로 분류했다.

---

## 8. 다음 단계 후보 (제안만, 사전 등록 값은 바꾸지 않는다)

저비용 / Drive 산출물 필요 (GPU 불필요 또는 소규모):
1. 레벨별 예측 크기 평균/기울기(편향 부호)를 출력 공간 probe, 조건 (a), (b), 시점별로 보고 -> "ind 로 수렴"과 "정보 소실" 구분 (prior 8-1, skeptic 8-2, stats 10-3). cells.npz 필요.
2. persistence(마지막 context 프레임 토큰 복사)와 untrained(random-init) predictor baseline 의 같은 출력 공간 R² (prior 8-2). 크기와 무관한 속성(그리퍼 위치, 천 중심 위치)의 같은 rollout 감쇠로 "일반 rollout 품질 저하"와 "크기 특이 손실"을 분리 (prior 8-3).
3. 예측 z 로 재학습한 probe 와 전이 probe 비교(탐색, 판정과 섞지 않음) (skeptic 8-2, stats 10-4). 후반 프레임으로 학습한 마스크 baseline, 마스크+프레임 인덱스 baseline, MAE/순서 정확도 병기 (skeptic 8-2; stats 10-3 은 MAE 와 역할별 SST 병기).
4. 시드를 분석 단위로 쓰는 2차 보고(시드 단위 평균과 CI 를 판정 표에 병기, (a)/(b)는 에피소드와 시드를 함께 재표집하는 2단계 bootstrap 사전 등록) (stats 10-2). 코드 변경은 experimenter 몫.
5. steering 재해석: 다음 latent 전체의 변화(`T_latent`, 패치 평균 L2 는 코드에 있으나 results.json 에 없음)와 크기 방향 대 다른 의미 있는 방향(그리퍼, 프레임 인덱스, 행동) 비교, T 를 레벨 간 log 면적 차에 대비해 척도화 (prior 8-4, 8-6).
6. 그리퍼 마스킹 probe 또는 그리퍼 위치 baseline (prior 8-5; 001 의 미이행 항목).
7. results.json 보강: Drive 의 q3 세부, `read_by_level`, 학습 곡선, 재개 기록(run, epoch), 패치 sha, loaded epoch 를 복사해 넣기(재계산 아님) (skeptic 7-8). 문서 갱신(README.md:3, 5; config.yaml:2) (skeptic 7-12).

GPU 필요 / 사람 승인 필요:
8. Q3 인과 분해 재실행(기존 체크포인트): 직접 효과와 간접 효과 분리, (a) 모델과 항등/무작위 predictor 를 대조로, 판독 변화와 `T_latent` 병기, 천 패치 토큰 교체 같은 공간 개입 추가 (skeptic 8-1). 비용 규모는 분석가가 명시하지 않았다(미확인).
9. 재개 없는 재학습과 시드 확대: (a), (b) 각 6개 이상, 또는 s0, s1 을 재개 없이 재학습해 학습 크기 step 12 R² 의 산포(0.05-0.54)가 재개와 관련 있는지 확인. 행동 분포 통계를 크기별로 비교 (skeptic 8-3, stats 10-1, 10-7, prior 8-8).
10. 제거 재시도(INLP 반복 수 증가, 비선형 제거 등): `mlp_r2_after_max`, `inlp_iterations` 변경이므로 별도 확정 기록과 새 실험 ID 로만 (prior 8-10, stats 10-5).
11. 4c 진단 재설계(크기 신호 자체와 낯섦 분리, floor 와 비교 표본 크기 맞춤) (skeptic 7-11, stats 10-6). 새 판정 기준은 결과를 보기 전 새 사전 등록으로 고정해야 한다(skeptic 7).

관련 진행 중 실험(판정 없음): 실험 003(후보 H, 끝 크기 단일 학습과 혼합 학습)은 skeptic I3 와 prior 3절이 "크기 다양성 대 범위 포함"과 "가운데로 압축" 교란을 맡는 후속으로 언급한 질문과 관련된다. 003 은 002 의 train_resume.patch 를 이식하며 분포 간 GPU 와 학습 코드가 다르다(확정 기록 11). 003 결과는 이 보고서에서 판정하지 않았다.

---

## 9. 논문에 쓸 수 있는 후보 문장과 쓰지 말아야 할 문장 (확정 아님, 사람이 확정)

후보(조건: 학습 시드 3개, 단일 설정, num_hist 1, 재생성 데이터, CI 는 에피소드만 반영):
- 인코더 토큰으로 학습한 linear 크기 판독기를 예측 latent 에 적용했을 때, rollout 마지막 시점(step 12)에서 R² 는 조건 (b) 에서 0.529/0.525/0.500(Q1 천장 0.820), 조건 (a) 에서 0.082/0.114/0.091 이었다. (a)의 하락이 (b)의 모든 시드보다 컸다(시드 3 대 3). 정보 소실인지 판독기 전이 실패인지는 구분되지 않았다. (세 분석가 모두 이 범위까지는 인정; skeptic 4절 종합이 이 문장을 "이 실험으로 말할 수 있는 것"으로 제시)
- 후반 변형 프레임의 보간 레벨(a080, a125)에서 DINOv2 linear 판독기가 mask 면적 baseline 과 random-init 인코더보다 높은 R² 를 냈다(0.5137 대 -0.0423 / 0.1737). 단 초기 프레임에서는 mask baseline 과 거의 같았고(보간 ΔR² 0.0966, 외삽 -0.0028), MAE 와 순서 정확도를 함께 써야 하며(skeptic), 마스크 baseline 의 학습 방식이 불리하다는 지적이 있다(skeptic). 외삽에서는 사전 등록 마진 0.1 을 넘지 못했다(linear +0.0339, attentive -0.1245).
- ridge 크기 방향으로의 이동이 판독된 크기를 분산·이동 크기를 맞춘 무작위 100방향보다 크게 바꿨다(6칸 모두 95 백분위 밖). 해석은 "개입 방향과 정렬된 판독이 반응한다"로 제한한다(prior, skeptic 의 제한 문구; stats 는 설계 순환성을 추정으로 표시).
- 마지막 predictor 블록에서 선형 판독 R² 는 layer 0 과 거의 같고(변화 ±0.013 이내) 조건 간 차이는 0.008 이다. 조건 (a)도 같은 값이므로 "보존이 학습되었다"의 근거는 아니다.

쓰지 말아야 할 문장(분석가 지적):
- "predictor 가 크기 정보를 쓴다/버린다", "H-c 지지" (세 명 일치).
- "다양한 크기 학습이 크기 보존/사용을 만든다" (prior, skeptic).
- "rollout 중 크기 정보가 사라진다" (세 명; 판독기 전이 실패와 구분 안 됨).
- "DINOv2 가 크기를 특이하게 보유한다" (prior; random-init 대비 외삽 +0.07).
- "외삽에서는 인코더가 크기를 못 읽는다" (skeptic; 외삽 R² 0.859).
- "조건 (a), (b) 의 층별 변화가 유의하게 다르다 (p=0.002)" (skeptic, stats).
- "ind 로 수렴한다" (prior; 편향 부호 미보고).
- H-c 제거 효과가 무작위보다 크다는 문장 (사용자 결정; 세 명).
- not_removed 의 원인을 사실로 서술하는 문장 (skeptic; log.md 추정과 skeptic 의 대안 후보는 모두 미검증).
- 4c large 를 근거로 (b) 결과를 해석 불가로 선언하거나, medium 을 근거로 (a) 결과를 정상 해석이라고 쓰는 것 (prior).
- "DINO-WM 의 구조적 한계" 일반화 (prior).

---

## 10. 사람이 결정할 사항

에이전트의 제안은 승인이 아니다. 사전 등록 임계값과 판정을 바꾸는 선택지는 넣지 않았다.

1. 판정 표 표기: (a) status 는 그대로 두고 리뷰의 한정 문구를 병기, (b) 위에 더해 시드 단위 CI 와 R² 외 지표(MAE, 순서 정확도)를 보조 열로 추가하는 새 기록을 만들기(사전 등록 값 불변, stats 10-2, skeptic 7-4, 7-5). 6절의 분석가 불일치(특히 H-a 보간 linear)를 해소할 근거는 이 선택으로 생기지 않는다.
2. H-c 후속: (a) 보류, (b) 기존 체크포인트로 Q3 인과 분해(skeptic 8-1, GPU, 승인 필요), (c) 제거 재시도를 새 확정 기록과 새 실험 ID 로(INLP 반복 수 등 변경, prior 8-10).
3. 저비용 재분석 승인(8절 1-7번, Drive 산출물 필요): 승인 여부, 승인 시 어떤 항목을 먼저 할지. 결과는 탐색으로만 표기하고 002 status 는 불변.
4. 학습 시드 확대 또는 재개 없는 재학습(8절 9번, GPU 비용 큼): 승인 여부와 범위(시드 5 이상, s0/s1 재학습 우선 등).
5. 논문 주장 범위 확정: 9절의 후보 문장 중 무엇을 채택할지, 특히 H-a 보간 linear 를 "지지"로 쓸지(stats 는 강함, skeptic 은 근거 아님). 서지 미확인 인용은 확인 후에만 사용.
