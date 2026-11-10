# 실험 005_inductive_bias_probe 보고서

- 작성: reporter, 2026-11-10
- 입력: experiments/005_inductive_bias_probe/{README.md, CHANGES.md, config.yaml, results.json, results/{REPORT.md, plan.json, tasks_meta.json, attempts.json, cost.json(앞부분)}, analysis/{recompute_stdout.txt, score_table.csv, hf_sensitivity.csv, extra_checks_stdout.txt}, logs/colab_20261109/{tasks_*.log 4개, plan_20261109-112046.log, heads_20261109-112308.log(앞 30줄)}, src/exp005/verdicts.py}, experiments/004_depth_truncation/results/summary.json(앞 80줄, 분기 입력), notes/reviews/005_prior.md, 005_stats.md, 005_skeptic.md, notes/specs/exp4_5_spec.md 4절, notes/hypotheses.md (후보 I, 확정 기록 9, 12), notes/log.md (2026-11-09 항목), reports/003_train_dist_dependence.md, reports/007_decoder_collapse_autopsy.md(형식 참고).
- 이 보고서는 새 측정을 하지 않았다. 수치 출처 표기:
  - `rj:` = results.json 키 경로. 리포터가 파일 전체를 읽었다(272행). 판정 3개, 임계값, 비용, 환경이 들어 있다. **results.json 에는 판정 요약만 있고 H-e 의 하위 조건 값, 5-B 이탈 값, 모델별 점수는 없다**(H-e 가 포화 검사에서 먼저 반환됨; `src/exp005/verdicts.py:46-47`, 리포터가 코드를 읽어 확인).
  - `an:` = experiments/005_inductive_bias_probe/analysis/ 산출물. analyst-stats 가 저장된 예측에서 독립 재구현한 값이다. 리포터가 `recompute_stdout.txt`, `score_table.csv`, `hf_sensitivity.csv`, `extra_checks_stdout.txt` 를 열어 읽었다. stats 는 이 재계산이 results/ib_scores.csv(700행), calibration_dev.csv, law_fits.csv(175행)와 raw 최대 차 1.3e-7, score 최대 차 2.4e-6, 기울기 차 2.2e-16 이내로 일치한다고 적었다(stats 1절). 리포터는 results/*.csv 원본(ib_scores.csv, calibration*.csv, law_fits.csv)을 직접 열지 않았고, 이 일치는 stats 의 서술이다.
  - `(prior/stats/skeptic 절)` = 해당 리뷰가 계산했거나 인용했다고 밝힌 값. 별도 표시가 없으면 리포터가 대조하지 않았다. skeptic 의 `(계산)` 은 코드 실행 없는 손계산이다(skeptic 상단). prior 의 손합산 값도 같다(prior 3-4: "파일에서 재계산 필요").
  - `리포터 산술` = 리포터가 `an:score_table.csv` 의 두 숫자를 빼거나 곱한 것(표시함).
  - 그림(fig_calibration.png, fig_dib_decomp.png, an:fig_dib_raw_by_model.png)과 predictions/*.npz 는 열어보지 않았다.
- 표기: "추정" 또는 "미확인" = 분석가 또는 리포터가 확인하지 못한 주장. 세 분석가는 서로의 파일을 읽지 않고 독립 작성했다. notes/reviews/004_*.md 도 존재하나 이 보고서는 읽지 않았다(004 입력은 summary.json 만).
- 지위: 후보 I 의 가설 H-e, H-f, H-g 의 판정 실험이다(문장은 2026-11-03 사용자 확정, 임계값은 명세 제안값이며 "파일럿 후 확정" 기록은 없다). 판정 문자열(supported/refuted/not_evaluable)은 config 임계값을 코드가 기계적으로 적용한 결과다.
- 이 보고서의 어떤 제안도 승인이 아니다. 사전 등록 값을 바꾸자는 권고는 쓰지 않았다. 사람이 결정할 사항은 8절.
- 사용자 확인 사항(리드 전달): 본 실행은 2026-11-09 Colab NVIDIA L4, head seed 5개, 체크포인트는 002 조건 (b) train_seed 0 하나다. `pilot: true` 는 사용자가 config 의 `pilot: true` 를 둔 채 노트북 `ALLOW_MAIN_WITH_PILOT_TRUE` 로 본 실행했기 때문이다(log.md 2026-11-09 주의 (a)에도 같은 서술).

---

## 1. 한 문단 요약

004 의 `summary.json`(`k_star` null, `m_star` null, `loss_ood_only` false)이 4.6 분기표의 "소실 없음" 행(`no_vanishing`)을 골라, predictor 출력 한 층(`pred_out6`)에서 5-A~5-D 를 돌렸다(`rj: plan`). 모델은 002 조건 (b) 의 train_seed 0 체크포인트 하나이고 변동은 head 초기화 seed 5개뿐이다. 사전 등록 판정은 **H-e not_evaluable**(사유 `area_baseline_saturated`, 면적 head D-IB 점수 1.0691), **H-f supported**(|D-IB_area= - D-IB_area≠| 0.0872 이 허용폭 2.0 x max(sd) = 0.0905 안쪽), **H-g refuted**(7개 slice 중 interp 만 [0.8, 1.2] 안, 나머지 6개는 0.25~0.61)이다(`rj: verdicts[0]`). 실행에는 **사전 등록 이탈이 하나 있다: 평가 입력이 config.yaml 과 명세의 2000개가 아니라 900개**로 실행됐고(tasks_meta.json `n_eval` 900), 누가 어떻게 바꿨는지는 **사용자 확인 대기**이며 CHANGES.md 와 log.md 에 기록이 없다(6절). 세 분석가가 일치하는 점은 (1) 이 설정에서 D-IB 점수 1 은 상한이 아니며 점수가 1 을 넘는 것은 oracle 이 약하기 때문이고(MLP oracle 의 D-IB raw 0.3188 대 무정보 0.5000), (2) 5-B 의 이탈 기준 0.1 은 참 함수 자신(0.328)도 통과하지 못하며, (3) 사전학습된 predictor("ours")와 무작위 초기화 predictor("random")가 거의 모든 지표에서 비슷하고, (4) H-f supported 는 경계값 통과라 "면적 휴리스틱 아님"의 증거로 읽을 수 없고, (5) H-g 의 기울기 < 1 은 ours 와 random 이 같고 참 s 를 받는 MLP oracle 도 한 slice(0.693)에서 범위를 벗어난다는 점이다. 의견이 갈리는 곳은 H-e not_evaluable 의 성격(prior: 지표의 한계 / stats: 규칙대로 맞으나 사유 이름 부정확 / skeptic: 중립 결과가 아니라 구조적으로 통과 불가한 검정), H-f 의 보고 표현과 격차의 방향 해석, 확정 기록 12 의 정규화 기준 처리, ours≈random 의 원인 귀속, H-g 결론의 범위이다. 가설 H-e 는 판정 불가, H-f 와 H-g 는 규칙상 각각 supported/refuted 이지만, 이 실험은 "ours 가 크기 구조를 따르는가"를 지지도 반증도 하지 못했다고 읽는 것이 세 분석가의 공통 방향이다(5.1 의 1, 4, 5번). 확신 수준: 판정 재현과 수치(stats 재구현, skeptic 줄 단위 대조)는 높음, H-g refuted 가 "규칙상" 견고하다는 점은 높음(stats), H-f supported 의 실제 의미는 낮음, D-IB 점수의 해석은 중간(stats), 결과의 일반화는 낮음(1 체크포인트, head seed 만, n_eval 900 미해명, pilot 플래그, 임계값 미확정). 분석가별 전체 확신도는 prior 중간, stats 항목별(높음/중간/낮음), skeptic 항목별(높음/중간/낮음)이며 stats 와 skeptic 은 단일 전체 등급을 제시하지 않았다.

---

## 2. 배경과 가설

- 배경: 실험 5 는 Vafa et al., ICML 2025 (arXiv 2507.06952)의 inductive bias probe 를 줄인 버전이다. 명세는 "새 과제에 적은 데이터로 적응시킨 head 가 크기를 기준으로 외삽하는가"를 고정 probe 1개 대신 무작위 과제 여러 개로 본다(명세 4절). **원문은 확인하지 못했다**(README, CHANGES.md 5절; prior 는 arXiv HTML 을 요약 모델 경유로 앞 100,000자/106,535자만 봤고 App. B 식은 확인하지 못했다. prior 0절). 이 보고서는 원문 내용을 근거로 쓰지 않는다.
- 가설(hypotheses.md 후보 I, 298-300행; 문장은 2026-11-03 사용자 확정(확정 기록 9), 수치는 명세 제안값):
  - H-e (5-A/5-B): OoD 포함 R-IB, D-IB 가 모두 면적 휴리스틱 모델보다 높고 random-init 과 seed 편차 밖에서 다르며, 5-B calibration 곡선의 45도선 평균 이탈이 0.1 이하.
  - H-f (5-C): `D-IB_area=` 와 `D-IB_area≠` 가 seed 편차 안에서 같다(명세 이름: "휴리스틱 아님").
  - H-g (5-D): slice 별 기울기가 모두 0.8~1.2.
  - 임계값 0.1 / 0.8~1.2 는 "파일럿에서 확정, 바꾸면 결과보다 먼저 기록"으로 적혀 있으나 **확정 기록이 없다**(log.md 2026-11-09 주의 (b), skeptic I5). `sd_mult` 2.0 과 `baseline_saturated_score` 0.95 는 명세 4.7 에 없고 구현자가 정한 impl default 다(config.yaml:68-69, CHANGES.md 19).
- 이 실험이 사전에 주장하지 않는 것: README 는 판정이 "규칙을 적용한 결과일 뿐"이고 "모델이 크기를 사용한다/세계 모델을 찾았다"는 결론을 쓰지 않으며, "읽을 수 있다"와 "사용된다"를 구분해 후자는 실험 3(개입) 이후라고 적는다(README 12행).
- 확정 기록 9 (2026-11-03): 5-B 함수족 = 주파수를 제한한 무작위 사인파 합, LoRA 생략, 4-C 깊이 재학습 생략, 진행 범위는 코드+selftest+dry-run(이후 Colab 실행은 별도 단계로 진행됨).
- 확정 기록 12 (2026-11-09, 004/005 결과 보기 전, 리드 추천을 사용자가 채택): `branch.exp1_negative` null -> false, `normalization.noninf` {R: iid, D: iid} -> {R: iid, D: const}. 근거로 적힌 문장: "무작위 이진 라벨 과제에서 oracle 의 D-IB raw 와 i.i.d. 기준의 D-IB raw 가 모두 약 0.5 라 분모가 0 에 가깝다"(hypotheses.md:362). 이 전제가 실제 실행에서 어떠했는지는 4.3, 5.2, 6절.

---

## 3. 설정과 방법 (config 기준)

| 항목 | 값 | 출처 |
|---|---|---|
| 층 선택 | 4.6 분기 행 `no_vanishing` -> 층 `pred_out6`(predictor 출력, 6 block), 하위 실험 5-A~5-D, hold false | `rj: plan`; plan_20261109-112046.log |
| 분기 입력(004) | `summary.json`: pilot true, `k_star` null, `m_star` null, `loss_ood_only` false, `drop_threshold` 0.2, `l_best` = predictor depth 6 | experiments/004_depth_truncation/results/summary.json 1-40행 |
| 체크포인트 | 002 조건 (b) `multi_60_ind_160`, `feat.train_seed` 0 (1개) | config.yaml:7, 19 |
| head seed | [0, 1, 2, 3, 4]. 평가 입력 seed 1000, 5-A 과제 2000, 5-B 3000, 사인파 4000, 5-D 5000, 무정보 6000(모두 고정) | config.yaml:24-30 |
| 평가 입력 | **config.yaml:33 `n_inputs: 2000`(명세 4.1 도 2000). 실행값은 900**(초기 450 + 후반 450, 레벨별 86~109) | tasks_meta.json `n_eval`, `eval_per_level`, `eval_kind_counts`; tasks_20261109-112048.log:4 |
| 5-A 과제 | 100개, 과제당 100 입력, 9개 레벨에 Bernoulli(0.5) 라벨. **D_k 는 학습 레벨 3개(a060, ind, a160)의 train 입력 7,200개에서만** 뽑힘(002 캐시에 나머지 6개 레벨 train 이 없음) | config.yaml:34-38; tasks_meta.json `pool_levels_used_5A`, `n_pool_train_inputs`; CHANGES.md 7 |
| 5-B | 과제 100개, 사인파 3개 합(주파수 0.3~1.5), D_k 는 학습 레벨 입력, 20 등간격 구간, 이탈 집계 = 비어 있지 않은 구간 동일 가중 | config.yaml:39-48; CHANGES.md 3, 13 |
| 5-C | 면적 10분위 q, `D-IB_area=`/`D-IB_area≠`/`R-IB_fold`, 5-A 예측 재사용 | config.yaml:49-50; CHANGES.md 11, 12 |
| 5-D | 라벨 수 72(학습 입력의 1%, 최소 30), head 4개(3레벨 모두 / 레벨 조합 3가지 a060+ind, ind+a160, a060+a160), slice 7개, 평가 입력의 예측을 참 s(log 면적비)에 OLS. PySR 은 미설치로 실행 안 됨 | config.yaml:51-55; tasks_meta.json `n_labels_5D`; CHANGES.md 18; law_fits.csv `pysr_not_installed`(prior, skeptic S7) |
| head | 002 attentive 구조, 100 epoch, lr 1e-3, wd 1e-2, batch 32, 검증/조기 종료 없음(impl default). oracle linear/MLP(hidden 5)는 Adam 400 step | config.yaml:56-61; CHANGES.md 6, 17 |
| 모델 행 | ours(실제 predictor 출력), random(**실제 인코더 z 를 입력받는 새로 초기화한 predictor**, 인코더는 random 이 아님), area(마스크 면적/bbox 5개 특징 head), oracle(MLP, `oracle.primary`), oracle_linear, noninf(i.i.d. 무정보), const(상수 출력) | CHANGES.md 14-16; skeptic I3; config.yaml:65-66 |
| 정규화 | `score = (raw - raw_ref)/(raw_oracle - raw_ref)`, ref 는 R-IB: iid, D-IB: const(확정 기록 12). `min_gap` 0.05. 반대 기준은 `score_alt` 열 | config.yaml:62-64 |
| 판정 규칙 | H-e: (R-IB, D-IB 순으로 면적 점수가 0.95 이상이면 즉시 not_evaluable) 아니면 ours > area 와 |ours - random| > sd_mult x max(sd) 를 두 지표 모두 + 5-B `extrap_any` 이탈 <= 0.1 이면 supported, 아니면 refuted. H-f: `|eq - neq| <= sd_mult x max(sd_eq, sd_neq)` 이면 supported(pair_group `all`). H-g: ours 의 slice 별 기울기 평균이 모두 [0.8, 1.2] 이면 supported | `src/exp005/verdicts.py:28-95`(리포터가 읽음); `rj: thresholds` |
| 임계값 | dev_max 0.1, slope 0.8~1.2, sd_mult 2.0, baseline_saturated 0.95, H-e pair_group `ood_any`, calibration_group `extrap_any` | `rj: thresholds` |
| 실행 환경 | Colab NVIDIA L4(23.66 GB), Python 3.9.25 venv(conda 아님), torch 2.0.1+cu117 | `rj: environment` |
| 비용 | heads 단계 마지막 실행 4946.5 초, 최대 VRAM 1607.2 MB, ours 5 seed 각 약 294~297 초, random 각 약 296~301 초 | `rj: cost` |

실행에 관한 주의:
- **평가 입력 수**: Colab 로그에서 `tasks` 단계는 09:23:00 과 09:24:33 에 "eval inputs 2000" 으로 두 번 `ValueError: only 900 candidate inputs for kind 0, 1000 requested` 로 실패했고, 09:24:56 부터 "eval inputs 900" 으로 진행했다(tasks_20261109-092300.log, 092433.log, 092456.log; 본 실행 tasks_20261109-112048.log:4 도 900). `results/attempts.json` 의 `stage:tasks` 는 0 이라 이 실패 이력이 남지 않는다(리포터가 attempts.json 확인; skeptic S9 가 같은 지적). `rj: config_sha256` 와 tasks_meta.json 의 `config_sha256` 은 `2fd84b30cd8f8e20612f86675f81202d71a6e3ec6b8aa906aca035a74c57e578` 이고, 리드가 계산한 저장소 config.yaml 의 sha256 은 `4a6bba0b...` 로 다르다. 이 sha 가 무엇을 해시하는지(override 포함 여부)는 stats 도 확인하지 못했다(stats 8절). **누가 어떻게 2000 을 900 으로 바꿨는지는 사용자 확인 대기이고, CHANGES.md 7절("config 변경은 8a 항목 뿐")과 log.md 2026-11-09 항목("임계값/시드 변경 없음")에 이 변경이 적혀 있지 않다.** 판정은 900 입력으로 일관되게 계산됐고 stats 재계산이 이를 재현했다(stats 1절).
- 데이터 쪽 원인(skeptic I2): 002 캐시의 test 부분이 900 episode 이고 early 후보가 episode 당 1개 위치뿐이라 후보가 900 이다. skeptic 은 "확인됨"으로 표시했고 리포터는 002 코드와 캐시를 확인하지 않았다.
- 004 의 임계값(drop 0.2 등)도 "파일럿 후 확정" 제안값이고 확정 기록이 없다. 005 의 층 선택(`pred_out6` 한 층)은 004 의 이 미확정 판정(`no_vanishing`)에 의존한다(skeptic I5). 004 summary 의 `pilot` 도 true 다.
- README 와 config 주석의 "평가 입력 2000개" 문구(README 25, 43행; config.yaml:25, 33)는 실행값 900 과 맞지 않는다(리포터가 읽어 확인).

---

## 4. 결과

CI 는 없다. 표의 sd 는 head seed 5개의 표준편차(ddof=1)이며 **과제(100), 평가 입력(900), 5-A/5-D 라벨, 체크포인트는 모든 seed 에서 같다**(CHANGES.md 20; stats 2절). 과제 단위 bootstrap 은 stats 가 따로 계산했다.

### 4.1 판정표 (규칙대로 계산된 status, `rj: verdicts[0]`)

| 가설 | status | 근거 값 |
|---|---|---|
| H-e | not_evaluable | reason `area_baseline_saturated`, metric D-IB, area_score 1.0690845 (임계 0.95 이상) |
| H-f | supported | D-IB_area= 0.92784, D-IB_area≠ 1.01502, abs_diff 0.08718, sd_eq 0.04525, sd_neq 0.03058, pair_group `all`. 허용폭 2.0 x 0.04525 = 0.0905 |
| H-g | refuted | slice 7개 중 `within` true 는 interp 하나 (4.5) |

### 4.2 H-e: 판정 아래에 가려진 수치 (판정이 아님)

판정 함수는 포화 검사에서 반환하므로 아래 값은 results.json 에 없다. stats 재계산(`an:recompute_stdout.txt`, "H-e hypothetical")이고 skeptic 의 손계산이 같은 값을 낸다(skeptic C1).

ood_any, head seed 5개 평균(sd), 확정 기록 12 의 기준(R-IB iid, D-IB const):

| 지표 | ours | random | area | ours - random | 2 x max(sd) | ours - area |
|---|---|---|---|---|---|---|
| R-IB score | 0.7717 (0.0073) | 0.7400 (0.0053) | 0.7920 (0.0021) | +0.0318 | 0.0147 | -0.0203 |
| D-IB score (const 기준) | 1.0404 (0.0324) | 1.0596 (0.0333) | 1.0691 (0.0286) | -0.0192 | 0.0667 | -0.0287 |
| D-IB score_alt (iid 기준) | 0.9390 (0.0444) | 0.9091 (0.0476) | 0.8939 (0.0370) | +0.0299 | 0.0952 | +0.0451 |

- 같은 숫자의 출처: `an:recompute_stdout.txt`, `an:score_table.csv`. D-IB const 기준 area 1.0691 은 `rj: area_score` 와 같다. R-IB 의 `score_alt` 는 const 기준 분모가 0 이라 전부 NaN/-inf 다(`an:score_table.csv`).
- 만약 포화 검사가 없었다면 규칙 조건은 다음과 같다(stats 5절, skeptic C1; 판정이 아닌 수치 서술): R-IB ours > area 거짓, D-IB ours > area 거짓, R-IB |ours - random| 이 허용폭 밖(참, 비 2.17), D-IB |ours - random| 이 허용폭 안(비 0.29), 5-B 이탈 0.1 이하 거짓. 다섯 조건 중 네 개가 실패한다(skeptic C1). 이 가정에서 H-e 는 refuted 였을 것이라고 stats 는 쓴다(점수 정의별 수치만 본 것).
- R-IB 의 ours - random 차이 +0.0318 은 stats 의 과제 bootstrap 95% [0.0178, 0.0471] 로 0 을 제외한다(`an:extra_checks_stdout.txt`). D-IB(all) 에서 ours - random -0.0147 [-0.0338, 0.0033](0 포함), ours - area -0.0325 [-0.0533, -0.0140](`an:extra_checks_stdout.txt`).
- InD 쌍(`ind` 그룹)에서도 ours 는 area 보다 낮다: R-IB 0.6641 대 0.6806, D-IB 0.7909 대 0.8512 (`an:score_table.csv`; skeptic C1 가 같은 방향을 인용).

5-B calibration, `extrap_any`, 정규화 평균 절대 이탈 (임계 0.1; `an:recompute_stdout.txt`, stats 는 calibration_dev.csv 와 일치한다고 적음):

| 모델 | extrap_any 이탈 평균 (sd) | 모델의 평균 쌍거리 |
|---|---|---|
| ours | 0.3347 (0.0182) | 0.0886 |
| random | 0.3342 (0.0200) | 0.0907 |
| area | 0.3116 (0.0156) | 0.1090 |
| oracle_linear | 0.2264 (0.0169) | 0.1842 |
| truth(참 함수를 모델로 쓴 참고 행) | 0.3284 (0.0182) | 0.1233 |
| oracle(x 축 기준, 정의상) | 0 | 0.3617 |

- 즉 참 함수 자신도 기준 0.1 을 통과하지 못한다. x 축으로 쓴 MLP oracle 이 참 함수와 크게 다르기 때문이다(stats 5절). 모든 head 의 모델 쌍거리는 oracle 의 약 1/4 로 예측이 평균 쪽으로 수축한다(stats 5절). `ind` 그룹 이탈은 ours 0.0378, random 0.0431, area 0.0367, truth 0.0082 로 모두 0.1 이하다(`an:recompute_stdout.txt`).

### 4.3 D-IB 분모와 oracle (정규화 점수가 1 을 넘는 이유)

D-IB raw (pair_group `all`, head seed 5개 평균; `an:score_table.csv`, `an:recompute_stdout.txt`):

| 모델 | raw | 모델 | raw |
|---|---|---|---|
| ours | 0.3220 | oracle_linear | 0.2419 |
| random | 0.3267 | noninf (i.i.d.) | 0.5000 |
| area | 0.3324 | const | 0.0000 |
| **oracle (MLP, 분모)** | **0.3188** | 참 라벨을 그대로 쓰는 완벽한 모델 | 0.5023 (`an:extra_checks_stdout.txt`) |

- 분모(const 기준) = oracle raw 0.3188 (`all`), 0.3051 (`ood_any`). iid 기준이면 분모는 0.3188 - 0.5000 = -0.1812 (`all`), -0.1949 (`ood_any`)이고(`an:recompute_stdout.txt` "Denominators") 절댓값이 min_gap 0.05 보다 커서 점수는 정의된다(skeptic C2 도 같은 점을 적음). noninf 의 점수는 1.57 (`all`)/1.64 (`ood_any`)로 oracle(1.0)보다 높다(`an:score_table.csv`).
- MLP oracle 의 라벨 정확도: 학습 레벨 0.9855, 보간 레벨 0.4963, 외삽 레벨 0.4594 (우연 수준 0.5; `an:extra_checks_stdout.txt`). 라벨이 레벨마다 무작위이고 D_k 가 3개 레벨뿐이라 나머지 6개 레벨의 라벨은 원리적으로 예측 불가다(stats 8절, skeptic I4).
- **확정 기록 12 의 전제와 실제**: 기록은 "oracle 의 D-IB raw 약 0.5"를 전제로 했다. 실제 MLP oracle 의 raw 는 `ind` 그룹에서만 0.4761 로 0.5 근처이고(`an:score_table.csv`), `all` 0.3188, `ood_any` 0.3051, `ood_both` 0.3057 이다. skeptic C2 는 이 전제가 selftest 의 planted "perfect" 모델에서만 성립했다고 지적한다(skeptic C1 근거 (c), C2).
- ours 의 예측이 900개 입력 전체에서 상수인 과제가 26% 이다(이런 과제는 D-IB 0, R-IB 1; `an:extra_checks_stdout.txt`, stats 2절).
- 분모 선택 민감도 (D-IB, `all`, const 기준, 선형 oracle 분모 0.2419 사용): ours 1.3313, random 1.3507, area 1.3742 (`an:score_table.csv` 열 `orclin_const_mean`). MLP oracle 분모에서는 ours 1.0106, random 1.0253, area 1.0430.

### 4.4 H-f: D-IB_area= 대 D-IB_area≠ (pair_group `all`)

시드별 D-IB_area=: 0.891, 0.939, 0.983, 0.954, 0.873. D-IB_area≠: 0.990, 1.009, 1.066, 1.017, 0.993. 5 seed 모두 area= 이 낮다(`an:recompute_stdout.txt`).

민감도 (`an:hf_sensitivity.csv`, 모두 stats 재계산; 판정 규칙은 `all` 의 `score_primary` 하나):

| 변형 | eq | neq | 차 | 허용폭 | 차/허용폭 | 판정(규칙 적용 시) |
|---|---|---|---|---|---|---|
| 규칙값 (MLP oracle, D const) | 0.9278 | 1.0150 | 0.0872 | 0.0905 | 0.963 | supported |
| score_alt (D iid) | 1.0547 | 0.9744 | 0.0804 (**부호 반대**: eq > neq) | 0.1068 | 0.753 | supported |
| 선형 oracle 분모 (const) | 1.5273 | 1.3230 | 0.2043 | 0.0583 | 3.50 | refuted |
| pair_group `ood_any` | 0.9957 | 1.0426 | 0.0468 | 0.0982 | 0.477 | supported |
| pair_group `ood_both` | 1.0528 | 1.0805 | 0.0277 | 0.1155 | 0.240 | supported |
| pair_group `ind` (쌍 2,429개) | 0.6639 | 0.8030 | 0.1390 | 0.0640 | 2.17 | refuted |

- 판정이 뒤집히는 `sd_mult`: 1.927 (1.92 이하면 refuted, 2.0 supported; `an:recompute_stdout.txt`). sd_mult 2.0 은 impl default 다.
- seed 대응 t 검정(차 = 0): p = 0.0011, Welch p = 0.0091, dz = -3.79 (`an:hf_sensitivity.csv`). 규칙은 "seed 편차 안에서 같다"를 보는 것이고 동등성 검정이 아니다(stats 4절).
- 다른 모델의 같은 격차(MLP oracle 분모, const 기준; eq - neq):

| 모델 | 격차 | 허용폭 (2 x max sd) | 규칙 적용 시 |
|---|---|---|---|
| ours | -0.0872 | 0.0905 | supported (`rj`) |
| random | -0.0880 (리포터 산술: 0.94174 - 1.02974) | 0.0736 (리포터 산술: 2 x max(0.03682, 0.03155)) | refuted (skeptic C3 의 계산과 같은 방향) |
| area | -0.1423 (리포터 산술: 0.90797 - 1.05024) | 규칙 적용은 skeptic 이 refuted 로 적음 | refuted |
| oracle_linear | -0.1601 (stats bootstrap 점추정, `an:extra_checks_stdout.txt`) | - | - |
| oracle(MLP) | 0 (분모가 oracle 자신이라 정의상) | - | - |

  stats 의 과제 bootstrap 95%: ours -0.0879 [-0.1184, -0.0580], random -0.0884 [-0.1235, -0.0537], area -0.1428 [-0.1690, -0.1162] (`an:extra_checks_stdout.txt`). 점추정이 직접 평균 차와 조금 다르다(6절 표시만 하는 불일치).
- 쌍 수: D-IB_area= `all` 27,401개, D-IB_area≠ 332,305개 (`an:recompute_stdout.txt`).

### 4.5 H-g: slice 별 기울기 (참 s 에 대한 예측 OLS)

| slice | ours 기울기 (sd) | `within` | ours R² | random | oracle MLP | area | 
|---|---|---|---|---|---|---|
| interp | 0.9356 (0.0873) | true | 0.557 | 0.9813 | 1.0051 | 0.7914 |
| extrap | 0.5432 (0.0604) | false | 0.956 | 0.5209 | 0.8068 | 0.4559 |
| early | 0.6066 (0.0574) | false | 0.885 | 0.5855 | 0.8335 | 0.4481 |
| late | 0.5362 (0.0509) | false | 0.791 | 0.5180 | 0.8363 | 0.5011 |
| labels_a060+a160 | 0.5409 (0.0056) | false | 0.683 | 0.5326 | 0.8772 | 0.5579 |
| labels_a060+ind | 0.3041 (0.0266) | false | 0.707 | 0.2736 | 0.8509 | -0.0132 |
| labels_ind+a160 | 0.2523 (0.0243) | false | 0.681 | 0.2512 | 0.6930 | 0.2739 |

- 출처: ours 기울기, sd, `within` 은 `rj: verdicts[0].H-g.per_slice`. R², random, oracle, area 는 `an:recompute_stdout.txt` 의 law table(stats 재계산, law_fits.csv 와 일치한다고 stats 가 적음). oracle_linear 는 7개 slice 모두 1.000(목표가 s 의 선형 함수라 구성상 자명; skeptic I1, `an:recompute_stdout.txt`).
- MLP oracle 은 7개 중 6개가 [0.8, 1.2] 안, `labels_ind+a160` 한 개만 0.693 (seed 값 0.535, 0.646, 1.092, 0.796, 0.396, skeptic I1 계산; sd 0.267 은 `an:recompute_stdout.txt`).
- 0.8 에 대한 일표본 t 검정(stats 6절 `an:recompute_stdout.txt`): extrap p=0.0007, early 0.0017, late 0.0003, 조합 3개 모두 p<1e-4 에 가깝다. interp 는 0.0255 로 0.8 보다 유의하게 크지만 1.0 과는 p=0.174 다.
- 조합 slice 의 학습 라벨 레벨 수(`ids_d`): a060+ind = 46+26, ind+a160 = 36+36, a060+a160 = 47+25. 라벨 부분집합은 조합마다 하나뿐이고 모든 seed 에서 같다(stats 6절). 5-D 의 학습 목표값은 3레벨 head 에서 3개, 조합 head 에서 2개뿐이다(skeptic I1, `stages.py:78, 94-96`).
- 학습 레벨의 s 범위는 [-0.51, 0.47], 평가 범위는 [-1.35, 0.91] (`an:recompute_stdout.txt`).

### 4.6 보조 수치

- 5-C 면적 10분위와 크기의 관계: CHANGES.md 12 는 "레벨이 다르고 q 가 같은 쌍은 이웃 크기일 가능성이 높다"고 셀프테스트 사실을 적는다. 이 때문에 크기에 매끄러운 모델은 모두 D-IB_area= 이 낮다(stats 4절, 선형 oracle 에서도 격차 -0.1601).
- 쌍 수(`an:recompute_stdout.txt`): D-IB `all` 359,706, R-IB `all` 44,844, R-IB `ood_any` 30,548(= `ood_both`; 같은 레벨 쌍에서 ood_any 는 곧 ood_both).
- 과제 단위 변동(stats 2절, `an:recompute_stdout.txt`): ours D-IB `all` 의 과제별 분산 0.0435(sd 약 0.21) 대 과제 안 seed 분산 0.0019. 변동 대부분은 과제(라벨 배정)에서 온다. 보고된 sd 는 과제 변동을 담지 않는다.
- 비용: `rj: cost` (3절).

---

## 5. 분석가 의견 종합

### 5.1 세 명이 일치한 점

1. **D-IB 점수가 1 을 넘는 것은 "크기를 oracle 보다 잘 읽음"도 "면적 baseline 이 천장"도 아니다.** MLP oracle(참 s 입력, hidden 5, 라벨 3레벨로 학습)의 D-IB raw 가 0.3188 로 무정보 모델 0.5000 보다 낮고, 신호가 약하고 잡음이 많은 모델일수록 raw 가 올라 점수가 1 을 넘는다. noninf 가 1.5~1.7 점을 받는 것이 그 증거로 인용된다(prior 3-1, stats 3절, skeptic C1 근거 4).
2. **5-B 이탈 기준 0.1 은 이 설정에서 참 함수도 통과하지 못한다.** x 축이 3레벨로 학습한 MLP oracle 이고, truth 행 이탈이 0.328 이다(prior 3-6, stats 5절, skeptic C1 근거 3). 따라서 H-e 의 세 번째 조건은 포화 여부와 무관하게 불충족이다.
3. **ours 와 random-init predictor 가 거의 모든 지표에서 비슷하다.** R-IB 만 ours 가 약 +0.03 높다(prior 3-1, 3-4; stats 5, 6절; skeptic I3). 5-D 기울기(ours 0.25~0.94 대 random 0.25~0.98)와 5-B 이탈(0.335 대 0.334)이 사실상 같다.
4. **ours 는 R-IB 에서 면적 head 보다 낮다**(ood_any 0.7717 대 0.7920; prior 3-1 는 `all` 0.738 대 0.755~0.758 로 인용, stats 5절, skeptic C1). D-IB(const 기준)에서도 ours < area 는 stats 와 skeptic 이 보였고 prior 는 같은 값을 "비슷하다"로 인용했다.
5. **H-f supported 는 경계값 통과이고 "면적 휴리스틱 아님"의 근거로 읽으면 안 된다.** 차이가 허용폭의 96%(prior 3-3, stats 4절, skeptic C3), 같은 격차가 random-init 과 면적 head 에서도 나온다(prior 3-3 은 area 와 random 의 같은 부호 간극, stats 와 skeptic 은 수치). 세 명의 표현 강도는 5.2.
6. **H-g 의 기울기 < 1 은 ours 의 성질로 귀속하기 어렵다.** random-init 이 같은 기울기를 보이고(prior 3-4, stats 6절, skeptic I1) 참 s 를 받는 MLP oracle 도 한 slice(`labels_ind+a160` 0.693)에서 범위 밖이다(prior 3-4 는 같은 slice 의 seed 값 0.535, 0.646, 1.092, 0.796, 0.396 을 인용, stats 6절, skeptic I1).
7. **seed sd 는 head 초기화 변동만 담는다.** 체크포인트 1개, 과제, 평가 입력, 5-D 라벨 부분집합이 고정이다(prior 3-3, stats 2절, skeptic I6). 이 sd 로 정한 허용폭("seed 편차 안/밖")의 통계적 의미가 약하다.
8. **정규화 기준에 따라 D-IB 에서 ours 와 area 의 순서가 뒤집힌다.** const 기준 ours 1.0404 < area 1.0691, iid 기준(score_alt) ours 0.9390 > area 0.8939 (ood_any; stats 5절, skeptic C2; prior 3-1 은 `all` 의 alt 값을 "area < ours ≈ random" 으로 인용). 이 사실 자체는 세 명이 같다. 처리 방식은 5.2.
9. **임계값이 미확정 제안값이고 `pilot: true` 가 남아 있다**(prior 0절, 6절; stats 8절; skeptic I5). 판정 문구의 신뢰도가 임계값 선택에 종속된다.
10. **원문 App. B 의 정규화식을 확인하지 못했다**(prior 0절, 2절 1번, 4절; stats 10절; skeptic S7, 5절). 원문 수치와 직접 비교할 수 없다.

### 5.2 의견이 갈린 점 (하나로 합치지 않음)

| 쟁점 | analyst-prior | analyst-stats | analyst-skeptic |
|---|---|---|---|
| **H-e not_evaluable 의 성격** | 예상 가능한 결과이고 "실패라기보다 지표의 한계". "포화"는 모델이 면적 휴리스틱 수준으로 크기를 읽는다는 뜻이 아니라 이 정규화에서 면적 head 도 oracle 수준 점수를 받는다는 뜻. 포화가 없었어도 5-B 조건이 항상 불충족이라 "not_evaluable 이거나 refuted 였을 가능성이 높다"(추정, calibration.csv 곡선은 못 봄). R-IB 에서 ours > area 조건도 충족되지 않을 것(추정, seed 4 미확인) | not_evaluable 은 **규칙 적용 결과로 맞다**. 다만 사유 이름 `area_baseline_saturated` 는 이 지표에서 부정확하다(점수가 천장을 넘은 것이지 면적 head 가 완벽한 것이 아님, 추정). 보고서에는 분모(oracle raw 0.319, 천장 0.5)를 같이 쓸 것을 권고. 포화가 없다고 가정한 수치로는 refuted 였을 것 | **중립 결과가 아니다.** 포화 검사가 뒤 조건을 가려, 포화 검사를 빼면 하위 조건 5개 중 4개가 실패한다. 5-B 조건은 참 함수도 통과 못 하므로 "구조적으로 통과 불가였고 실행 전에 예측할 수 있었다"(CHANGES.md 7, 14 가 이미 기록한 사실로 예측 가능, 확신 높음). README 의 "판정은 규칙 적용 결과일 뿐", 명세 5절의 "not_evaluable 은 음성 결과가 아니다" 문장이 이 판정을 "보류"로 읽게 만든다고 지적 |
| **H-f 보고 표현** | "supported" 를 "휴리스틱 아님"으로 읽으면 안 된다. 귀무 분포가 없다(area head, random 도 비슷한 간극). 필요한 비교는 "ours 의 간극 대 area/random 의 간극" | **"supported" 로 쓰되** 경계(비 0.963), 대응 t p=0.0011, random 도 같은 격차, 선형 oracle 로는 refuted 를 함께 보고. 동등성이 확인된 것은 아님 | "supported 는 과장". 여유 0.0033, 잡음이 클수록 통과하기 쉬운 기준. 같은 규칙을 random 에 적용하면 refuted. 판정이 갈리는 것은 ours 의 seed 0, 4 가 sd 를 키웠기 때문 |
| **H-f 격차의 방향 해석** | q 가 s 와 상관이 커서(CHANGES.md 12) 간극이 면적 사용의 증거로 쓰이기 어렵다. 원문의 q 는 state 의 함수라 이 교란이 없었다 | ours 격차(-0.088)는 area(-0.143)보다 작고 random 과 같다. MLP oracle 로 정규화하면 oracle 은 정의상 격차 0 이 되어 교란이 oracle 쪽에서만 지워진다. "격차 크기만으로 기전을 구분할 수 없음"(추정: 정보 부족) | ours 격차는 area 격차의 약 61% 이고 방향이 같다. 명세 4.4 의 해석 규칙("area= 가 훨씬 낮으면 면적 쪽")을 적용하면 ours 는 oracle(0)과 area(0.142) 사이에서 area 쪽에 더 가깝다. "휴리스틱 아님"이라 쓸 근거가 없다. 단 교란(CHANGES.md 12)은 인정하고, 정규화가 그것을 일부만 상쇄한다고 봄 |
| **확정 기록 12 (정규화 기준) 처리** | 기록 12 가 주 점수 기준을 결과 전에 const 로 고정했으므로 **주 판정을 바꾸지 말고 iid 기준은 "참고용 사후 관찰"로만** 둔다. iid 분모(-0.17)는 음수여서 부호가 반전되고 구조적으로 불안정 | const 와 iid(score_alt), 선형 oracle 분모를 병기. 선형 oracle 로 바꾸면 H-f 가 refuted(차 0.204, 허용폭의 3.5배). "보고된 판정은 규칙대로이나 이 민감도를 함께 써야 한다" | **결정의 전제("oracle D-IB raw 약 0.5")가 실제 데이터에서 거짓**(`ind` 그룹 0.466~0.492 에서만 성립). 결정은 결과 전이라 HARKing 은 아니지만 전제가 selftest 의 planted 모델에서만 맞았고 그 결정이 H-e 의 판정 경로를 정했다. 원래 명세 기준(iid)으로도 |분모| 약 0.18 > min_gap 이라 점수가 정의된다. "어떤 기준으로 보고하든 해석에 쓰면 안 된다" |
| **ours≈random 의 원인** | 5-D 외삽 기울기 0.54 는 head 와 라벨 범위의 성질에 지배된다. 수축이 학습된 predictor 의 동역학이라면 random 에서는 없어야 한다(추정). 소량 라벨(72개, 학습 레벨 3개) head 가 학습 범위 밖에서 수축하는 현상 | head 학습 일정/라벨 수/외삽 구조의 성질일 수 있다(추정). "표현이 크기 법칙을 안 따른다"가 아니라 "이 head/라벨 조건에서 단일 법칙(기울기 1)이 안 나온다"로 한정할 것 | MSE 회귀의 감쇠(예측값을 참 s 에 OLS 하면 수축된 예측기의 기울기가 1 미만, 추정·중간 확신)와, 무작위 predictor 도 잔차 경로로 인코더 정보를 보존할 것이라는 점(추정, 중간 확신): 측정된 구조는 고정 DINOv2 특징과 head 의 성질이고 world model 학습의 기여는 R-IB 약 +0.03. 이번 실험은 "world model 이 크기를 배웠다"와 "DINOv2 특징에 면적 단서가 원래 있다"를 구분 못 함 |
| **H-g 결론의 범위와 기준** | refuted 는 "ours 가 단일 크기 법칙을 따르지 않는다"로는 맞지만 원인 귀속(world model 대 head/라벨 범위)은 분리 안 됨. 기준을 상수 1 에 둔 위험(CHANGES.md 14)이 실현됨. 더 근거 있는 비교는 "oracle 대비 비율"과 "slice 간 기울기 일치". ours/oracle 외삽 비 0.543/0.81 ≈ 0.67(손계산) | refuted 는 **통계적으로 견고**(7개 중 6개가 구간이 겹치지 않음, 0.8 대비 p<0.003). 단 규칙이 MLP oracle 에도 한 slice 에서 실패할 만큼 엄격 | 기준과 head 의 성질이다. 2점으로 학습한 combo head(`labels_a060+ind`, `labels_ind+a160`)는 외삽 실패를 다시 확인할 뿐이므로 **판정에서 빼거나 별도 보고**. 감쇠 기준선(noise-matched oracle)이 필요 |
| **전체 확신도** | 중간 (포화 구조, random 과 동일 수축, oracle 비1 기울기는 CSV 에서 직접 나와 높은 편. 원문 App. B 미확인, 003 정합은 방향 일치까지만) | 항목별: 재현 높음, H-g refuted 높음, H-f supported 의 의미 낮음, D-IB 해석 중간, 일반화 낮음 | 항목별: 수치 지적(C1~C3, I1, I2) 높음, I3 와 감쇠 설명 중간, S3·S4 낮음 |

### 5.3 한 명만 지적한 점

- prior 만:
  - 문헌: Kang et al. 2024 (OCS), Xu et al. 2021 (ReLU 외삽), Hewitt & Liang 2019 (control task/selectivity)와의 연결(3-2, 3-4). 앞의 둘은 prior 가 003 리뷰에서 요약 수준으로 확인했다고 하고 Hewitt & Liang 은 기억 기반 미열람이다. 리포터도 원문을 확인하지 못했으므로 이 보고서는 근거로 쓰지 않는다.
  - 원문 요약에서 정규화 방향이 서로 어긋난다는 요약기 보고(B.2 는 "0 = perfect, 1 = random", Table 2 캡션은 "1 = perfect, 0 = noninformative"; 원문 직접 확인 아님, 2절 1번).
  - 003 정합: 5-D 외삽 기울기 0.543 과 003 의 open-loop 기울기(0.42~0.58 대)가 숫자는 비슷하지만 측정 대상이 다르고(rollout 수축 대 단일 스텝 predictor 출력에서 소량 라벨 head 로 읽은 선형 적합), 005 의 random 에도 같은 수축이 있어 "같은 현상이라는 증거로는 쓰기 어렵다". 정합이라 말할 수 있는 최대치는 "같은 방향"(3-5, 추정). stats, skeptic 은 이 비교를 하지 않았다.
  - 빠진 baseline: 학습 평균 상수 출력과 라벨 범위 클리핑 선형 회귀, 인코더 출력 층의 같은 절차(spec 4.6 분기상 이번 행에는 predictor 출력만 해당), 학습 seed 여러 개, random-init 의 oracle 대비 비율(4절).
  - 후보 가설 초안 J1~J3 (5절): J1 = 5-D 수축은 head 와 라벨 범위의 성질, J2 = D-IB 용 oracle 을 레벨 lookup 으로(정규화/임계 변경이라 결과 후 변경이며 사람 승인 필요), J3 = 5-C 간극은 q 와 s 의 상관 때문. 제안이지 사전 등록이 아니다.
  - 연구 공백: 연속 state 에 inductive bias probe 를 적용한 선행연구를 확인한 범위에서 못 찾음(체계적 검색 안 함, 추정).
- stats 만:
  - sd 추정의 불안정: n=5 에서 sd 의 95% 구간이 대략 0.6배~2.9배라 `2 x max(sd)` 허용폭 자체가 불안정(2절). 과제 bootstrap 이 시드 sd 보다 정직한 불확실성(2절).
  - H-f 의 대응 t 검정과 뒤집히는 `sd_mult` 1.927, 선형 oracle 분모에서의 refuted, pair_group 별 민감도(4.4).
  - 5-A 정확도 수치(ours 0.6019 등)와 완벽 라벨 모델의 D-IB raw 0.5023, ours 상수 과제 26%(4.3).
  - 다중 비교: 판정 3개 x 쌍 그룹 4 x 지표 5 중 H-f 만 단일 그룹으로 판정하고 나머지 민감도 행은 탐색이며 p 값을 보정하지 않았다(7절).
  - R-IB 의 `score_alt` 가 모두 NaN 인 이유(const 기준 분모 0)와 H-e 가 어차피 not_evaluable 이라는 점(5절).
  - 권고: 다음 확인 후보로 라벨 용량이 충분한 oracle(레벨 룩업), 연속 s 과제 지표, 5-A 에서 모든 레벨을 D_k 에 포함, 과제/라벨 부분집합 시드 반복(9절).
- skeptic 만:
  - **I4 미결 결정 기록**: CHANGES.md 7 이 5-A 의 D_k 를 3레벨로 제한한 것을 "(사용자 결정 필요)"로 표시했는데 skeptic 은 hypotheses.md 와 log.md 에서 이 항목의 사용자 결정 기록을 찾지 못했다(grep 확인). 리포터가 읽은 범위(후보 I, 확정 기록 9, 12, log.md 2026-11-09)에서도 이 결정 기록을 찾지 못했다.
  - **I5 파일럿 범위**: 파일럿은 heads `--max-tasks 3 --seeds 0` 까지만 돌고 metrics/report 를 돌리지 않아(README 93행) 임계값 확정의 근거 자체가 없었다. 004 의 `k_star`, `m_star` null 도 미확정 drop_threshold 0.2 에 의존해 `no_vanishing` 행과 `pred_out6` 단일 층이 선택됐다.
  - **I3 random 대조의 정의**: random-init 은 실제 DINOv2 인코더 + 무작위 predictor 다(`stages.py:165`, `feats.py:47-52`). 명세 4.6 의 "random-init 인코더·predictor" 중 인코더 쪽은 predictor 층에서 쓰이지 않았다.
  - S1 잠재 버그: `stages.py:352-353` 의 `continue` 가 5-A 예측 하나라도 없으면 해당 seed 의 5-B/5-D 까지 건너뛴다. S2: `verdicts.py:8-14` `_vals` 가 중복 행을 경고 없이 덮어쓴다. 이번 실행에는 영향 없음(예측 25개 모두 존재, 중복 없음).
  - S3: D_k 는 train episode 의 8프레임 전체에서, 평가 입력은 early/late 위치 하나씩에서 뽑고 D_k 의 episode 는 world model 학습에 쓰였다(분포 차이, 크기는 작을 것으로 추정, 확인 안 함).
  - S4 누수 점검: 평가 입력은 `part == "test"`, 모든 D_k 와 5-D 라벨은 `part == "train"` 이고 q 는 라벨을 쓰지 않아 라벨 누수를 찾지 못했다. 단 002 캐시의 train/test 가 episode 단위로 분리되는지는 002 코드를 확인하지 않았다(미확인).
  - S5 쌍 계산이 selftest 에서 brute force 와 대조됨. S6 calibration 의 `norm` 이 seed 별 oracle 최대 쌍 거리라 쌍 하나가 정규화를 정함. S8 실행 환경이 conda 가 아닌 venv(`rj: environment`)이고 005 에 대한 별도 예외 기록을 찾지 못함. S9 `attempts.json` 에 tasks 실패 2회가 남지 않음(리포터 확인, 3절).
  - 추가 실험 제안 3개(최대 3개 제한): 지표 재정의 후 재계산, 원천 분리 대조, noise-matched oracle (7절).
- 두 명이 지적(한 명만이 아님): 평가 입력 900 대 2000 불일치(stats 8절, skeptic I2; prior 는 지적하지 않았다, 6절). PySR 미설치로 "식이 slice 별로 다른가"는 검증되지 않았다(prior 2절 4번, skeptic S7). 5-A 의 D_k 3레벨 한정으로 OoD 레벨 라벨은 예측 불가(stats 8절, skeptic I4).

---

## 6. 한계와 위험

- **체크포인트 1개, seed 는 head 만**: 002 조건 (b) train_seed 0 하나. 학습 seed 변동이 없고 과제/평가 입력/라벨 부분집합이 고정이라 모든 sd 와 "seed 편차 안/밖" 판정은 head 초기화 변동만 반영한다. 일반화 주장은 할 수 없다(세 분석가 일치, 5.1 의 7번).
- **평가 입력 900**: 위 3절. 쌍 수(D-IB `all` 359,706 등)는 900 입력 기준이다. skeptic 은 영향이 작을 것으로 보지만 추정이다(I2). 저장소 config 로 다시 돌리면 `tasks` 단계가 ValueError 로 멈춘다(skeptic I2, 로그로 확인되는 사실).
- **지표 구조**: D-IB 는 이 설정(이진 무작위 라벨, 3레벨 D_k, 약한 MLP oracle)에서 "state 구분 능력"을 단조적으로 재지 않는다(5.1 의 1, 8번). 5-B 의 x 축이 참 함수가 아니다. 5-A 의 OoD 레벨 라벨은 예측 불가(정확도 0.46~0.50)라 OoD 쌍 점수의 신호 대 잡음이 낮다(stats 8절, 추정).
- **판정 규칙의 구조**: H-e 는 포화 검사 때문에 뒤 조건이 results.json 에 남지 않는다. H-f 는 잡음이 클수록 통과하기 쉽다. H-g 는 참 s 를 받는 oracle 도 한 slice 에서 실패한다.
- **원인 귀속 불가**: ours≈random 이라 측정된 수축과 R-IB 수준이 predictor 학습, 고정 DINOv2 특징, head/라벨 범위 중 무엇에서 오는지 분리되지 않는다(5.2).
- **원문 미확인**: 정규화식(App. B)과 oracle 정의를 직접 확인하지 못했다. 원문과의 수치 비교는 불가다. 5-B 함수족(사인파 합)과 5-D(PySR 미설치)도 원문과 다르다(CHANGES.md 3, 18).
- **사전 등록 이탈과 미확정 값** (결과를 보기 전/후 구분):
  1. 평가 입력 2000 -> 900: 변경 주체와 시점 **사용자 확인 대기**. 기록 없음.
  2. 임계값: dev_max 0.1, slope 0.8~1.2 는 명세 제안값, `sd_mult` 2.0 과 `baseline_saturated_score` 0.95 는 impl default, 파일럿 후 확정 기록 없음. 이 두 impl default 가 각각 H-f(여유 0.0033)와 H-e(포화 경로)의 판정을 직접 정했다(skeptic I5). 결과를 본 뒤 바꾸지 않는다(log.md 2026-11-09 주의 (b)).
  3. 정규화 기준 변경(확정 기록 12): 결과 보기 전(004/005 결과는 보지 않았고 002 results.json 은 본 뒤라고 기록에 적힘)에 리드가 추천하고 사용자가 채택. 전제("oracle D-IB raw 약 0.5")가 실제 실행에서는 `ind` 그룹에서만 성립했다(skeptic C2; 4.3 의 수치로 확인). 이 결정이 H-e 의 판정 경로를 정했다. 결정이 결과 전이어도 전제의 오류는 별개의 문제로 남긴다.
  4. 5-A D_k 를 3레벨로 제한(CHANGES.md 7, "사용자 결정 필요" 표시, 결정 기록 미발견).
  5. `pilot: true` 가 본 실행 결과에 남아 있다(사용자 확인된 사항). 산출물만 보면 파일럿과 구분되지 않는다(skeptic I5).
  6. 층 선택이 004 의 미확정 임계값 판정에 의존.
- **실행 환경**: conda 가 아닌 venv(Colab). CLAUDE.md 의 conda 규칙에 대해 002 확정 기록 10 의 Colab 예외가 있으나 005 에 대한 별도 기록은 skeptic 이 찾지 못했다(S8).
- **미확인 목록**: 원문 App. B 식; calibration.csv 곡선 모양(prior 가 읽지 않았고 리포터도 열지 않음); PySR 경로; 학습 seed 별 변동; 002 캐시의 train/test episode 분리 여부(skeptic S4); sha256 `4a6bba0b...`(리드 계산)와 `2fd84b30...` 의 차이가 무엇 때문인지; 리포터가 results/*.csv 원본을 열지 않았다는 점(an: 파일과 stats 서술을 통해서만 확인); 그림 3개.
- **표시만 하는 불일치 (고치지 않음)**:
  1. **n_eval 900 대 config/명세/README/prior**: config.yaml:33, 명세 4.1, README 문구는 2000 이고 실행값은 900 이다. prior 2절은 "2,000개 입력(spec 4.1 과 일치)"라고 쓰고 실행값 900 을 언급하지 않는다(prior 는 tasks_meta.json 과 README 를 읽지 않았다고 0절에 적음). 리포터는 어느 쪽도 바꾸지 않았다.
  2. **stats 문서 내부**: 0절 4번 "수치로는 ours 는 면적 모델과 random 에 대해 어떤 지표에서도 의미 있게 높지 않다"와 5절의 "R-IB 의 ours - random 차이(0.032)는 과제 bootstrap 95% [0.018, 0.047] 로 0 을 제외한다"가 맞지 않는다. R-IB 에서만 ours > random 이 유의하다(효과는 작다고 stats 가 적음). 리포터는 5.1 의 3번에 R-IB 를 예외로 적었다.
  3. **H-f 격차 수치가 계산 방식마다 약간 다름**: ours 직접 평균 차 0.0872(`rj`, `an:hf_sensitivity.csv`) 대 stats 과제 bootstrap 점추정 0.0879; random 은 skeptic 0.0879, 리포터 산술 0.0880, stats bootstrap 점추정 0.0884; area 는 리포터 산술 0.1423 대 stats 0.1428, skeptic 0.1422. bootstrap 점추정이 왜 직접 차와 다른지(과제 단위 집계 방식 추정)는 리포터가 확인하지 못했다. 어느 값이든 random 의 격차는 허용폭 0.0736 을 넘고 판정 결과는 바뀌지 않는다.
  4. **skeptic 인용 범위**: skeptic C1 의 ind 그룹 R-IB "0.66~0.68 대 0.68", D-IB "0.76~0.82 대 0.82~0.87" 은 seed 값 범위이고, 리포터가 `an:score_table.csv` 의 5 seed 평균(R-IB 0.6641 대 0.6806, D-IB 0.7909 대 0.8512)과 대조해 방향이 같음을 확인했다.
  5. **prior 의 읽기 범위**: ib_scores.csv 를 702행 중 1~493행만, calibration_dev.csv 앞 40행만 읽었고(prior 0절) seed 4 행은 못 봤다. prior 의 평균들은 손합산이다(prior 3-4: "파일에서 재계산 필요"). 리포터가 이 손합산 값 중 oracle MLP 외삽 평균(prior 약 0.81)이 `an:recompute_stdout.txt` 의 0.8068 과 같은 범위임을 확인했다.
  6. **README 문구 낡음**: README 6절 "정규화식 문제(당시 미해결)" 와 3절의 "2000개" 서술은 실행 후 갱신되지 않았다(README 86행에 갱신 주석이 붙어 있으나 본문 일부가 이전 상태).
  7. **사유 이름**: `rj: verdicts[0].H-e.reason` = `area_baseline_saturated` 이지만 stats 는 이 이름이 이 지표에서 부정확하다고 본다(5.2). results.json 은 그대로 둔다.

---

## 7. 다음 단계 후보 (제안만, 사전 등록 값은 바꾸지 않는다)

기록과 점검 (저비용, 사람 확인 후 experimenter):
1. n_eval 900 의 경위(누가/어떻게 config 를 바꿨는지)를 확인해 CHANGES.md 와 log.md 에 남긴다. 영향 평가는 stats 가 요청한 대로 experimenter 의 확인 결과를 받은 뒤 정한다(stats 8절 "원인을 experimenter 가 확인해야 한다").
2. selftest 에 3레벨로 학습한 MLP oracle 의 OoD D-IB 점검과 `_vals` 중복 경고, `continue` 구조 점검 추가(skeptic C1 (c), S1, S2).

기존 캐시/산출물 위에서 가능(새 학습 거의 없음, 사후 분석으로 표시):
3. **원천 분리 대조**: 같은 head 와 과제로 004 캐시의 인코더 최종 층(`tokens_k12`)과 random-init 인코더(`rand_k12`)를 pred_out6 와 비교, 과제/라벨 시드(`seeds.tasks_a`, `seeds.d`)를 3개 이상 바꿔 변동에 포함(skeptic 추가 실험 2, prior 4절 빠진 baseline 2·3). H-f 는 사전에 정한 동등성 여유 Δ 를 쓰는 TOST 형태로 판정(skeptic).
4. **H-g 기준 보정**: ours 의 InD R² 에 맞춘 잡음을 참 s 에 더해 같은 head 에 넣는 noise-matched oracle 로 감쇠만으로 기대되는 slice 별 기울기를 구하고, combo head(목표값 2개)는 판정에서 빼거나 별도 보고(skeptic 추가 실험 3). 상수 출력/라벨 범위 클리핑 선형 회귀 baseline(prior 4절 1번).
5. **003 연결**: 003 의 4개 학습 분포 체크포인트에서 같은 5-D 를 돌려 기울기가 분포 폭과 함께 오르는지 확인(prior 3-5, 5절 4번). 체크포인트 접근과 GPU 필요 가능성(미확인).

지표를 바꾸는 후속 (결과 후 변경이므로 **새 사전 등록으로만**):
6. 5-A 의 D_k 를 9레벨 train 에서 뽑도록 002 캐시에 6개 레벨 train episode 를 추가하거나, 판정을 `ind` 그룹으로 제한(skeptic 추가 실험 1, stats 9절). 정규화 기준을 "라벨 lookup 으로 9레벨 정답을 아는 perfect-state 모델"로(prior J2, skeptic 1), 5-B 의 x 축을 `truth` 로, 포화 검사는 점수 상한이 1 인 경우에만 적용. sd_mult, 포화 기준, dev_max 를 결과 전에 확정 기록으로 남긴다.
7. 연속 s 과제 기반 지표(stats 9절)와 5-C 의 간극을 s 를 통제한 쌍에서 비교(prior J3).

GPU/사람 승인 필요:
8. 학습 seed 가 다른 체크포인트(002 조건 (b) seed 1, 2)에서의 같은 절차(prior 4절 3번). 비용은 미확인.

---

## 8. 사람이 결정해야 할 사항

1. 평가 입력 2000 -> 900 변경의 경위를 확인해 기록(CHANGES.md, log.md)으로 남길지, 이 결과를 `pilot: true` 와 임계값 미확정 상태 그대로 "본 실행"으로 인정할지.
2. H-e 를 현재의 not_evaluable 로 두고 지표(D_k 레벨 범위, oracle, 정규화 기준, 5-B x 축, 임계값)를 바꾸는 후속은 새 사전 등록으로만 진행할지, 승인 전까지 H-f supported 와 H-g refuted 를 문서에서 어떤 단서 문구와 함께만 쓸지(세 분석가의 의견이 갈리는 부분은 5.2).
3. 7절 3~5번(원천 분리 대조, noise-matched oracle, 003 연결)을 승인할지, 학습 seed 확대(8번)의 GPU 비용을 승인할지.
