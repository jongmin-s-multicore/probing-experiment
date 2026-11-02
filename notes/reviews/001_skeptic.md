# 001_size_collapse_diagnosis 회의론자 리뷰 (analyst-skeptic)

- 작성일: 2026-11-02
- 관점: 이 결과가 틀렸다고 가정하고 근거를 찾는다. 다른 분석가 파일(notes/reviews/*_prior.md, *_stats.md 등)은 읽지 않았다.
- 읽은 것: experiments/001_size_collapse_diagnosis/ 의 config.yaml, README.md, run.py(일부), overnight.sh(일부), run_all.sh(gate 부분), src/exp001/{diagnose,extract,masks,data,probes,probe_stage,intervene,aggregate}.py, results.json(부분 읽기), results/maskcheck.json, results/exp001_cs_baseline_single_s{0,1}/{repro_check,diagnose,probe}.json(부분), logs/overnight.log, logs/overnight_summary.txt, notes/hypotheses.md(후보 D/E, 사전 등록 초안, 확정 기록 2). notes/log.md 2026-11-02 항목은 수치 출처 대조용으로만 봤다.
- 구분: **[확인]** = 코드나 JSON 을 읽어서 직접 확인한 사실. **[추정]** = 읽고 의심하는 것(코드 실행 안 함).
- 수치는 모두 results.json 이나 results/*.json 에서 옮겼고 키 경로를 붙였다. `AS:` 는 `results.json -> across_seeds.baseline_single.metrics.` 의 줄임이다. 내가 직접 나눗셈한 값은 "(내 계산)"으로 표시했다.

## 0. 요약 판정

1. 측정 파이프라인 자체에서 결과를 뒤집을 만한 치명적 코드 버그는 찾지 못했다. 다만 **사전 등록 절차가 결론을 받쳐 주지 못한다**. 0번 전제 게이트는 판정되지 않았고(임계값 null, 게이트도 꺼져 있었음), 승인된 임계값(0.9 / 0.1 / slope_margin)은 어떤 .py 에서도 쓰이지 않는다. 그래서 지금까지의 "판정"은 모두 사람이 결과를 본 뒤 손으로 한 것이다.
2. log.md 의 "D: 사전 등록 기준 충족"은 사전 등록 문장의 절반만 확인한 것이다. 디코더가 *예측* latent 를 읽을 때 생기는 문제는 시험하지 않았다.
3. "작은 천 3.99배"는 에피소드별 비의 평균이다. 분모(real 면적)가 매우 작은 에피소드 때문에 크게 부풀었다. 평균끼리의 비는 2.36배다(내 계산).
4. "수축이 아니라 학습 크기로 수렴"은 큰 천에만 맞고 작은 천에는 부분적으로만 맞는다. 또 학습 크기 ind 가 평가 범위(로그 면적)의 한가운데에 있어서, "학습 크기로 수렴"과 "가운데로 압축/평균 회귀"를 이 설계로는 구별할 수 없다.
5. 개입(intervene) 결과로는 "predictor 가 크기 정보를 쓴다/버린다" 어느 쪽도 말할 수 없다. 크기 방향 u 를 따라 a248 을 a026 평균으로 옮겨도 디코딩 면적이 약 0.5% 만 바뀐다.

---

## 1. 치명 (사전 등록 기준으로 결론을 낼 수 없게 만드는 문제)

### C1. 0번 전제 게이트가 판정되지 않았고 이제는 사후 설정만 가능하다. 승인된 임계값은 코드에 연결되어 있지 않다 [확인]

- hypotheses.md "0. 전제 게이트": "이 게이트가 실패하면 D/E 결론을 내리지 않고 보고만 한다." 그런데 `diagnose.thresholds.shrink_ratio_tol: null`(config.yaml:226) 이다. 코드는 null 이면 문자열만 기록한다(src/exp001/diagnose.py:67-72, `"TODO: thresholds.shrink_ratio_tol not set"`). 결국 게이트는 통과도 실패도 할 수 없다.
- 게이트가 실제로 멈추게 하는 장치는 run_all.sh:95 의 `REPRO_GATE` 뿐이다. 그런데 본 실행은 overnight.sh 로 했고 거기서 게이트를 껐다(overnight.sh:237 `"repro_gate=off (recorded only, run continues)"`, pipeline() overnight.sh:193-201 은 repro_check 다음에 바로 extract 로 간다). logs/overnight.log 기준으로 seed 0 repro_check 가 10:59:29 에 끝났고 extract 는 같은 초에 시작했다. 결국 사람이 repro_check 를 보기 전에 모든 진단이 끝났다.
- 그래서 지금 임계값을 정하면 정의상 사후 기준 조정이다(확정 기록 2 의 문구 그대로). 사전 등록 규칙을 엄격히 따르면 D/E 결론은 "전제 미확인 상태의 조건부 관찰"을 넘을 수 없다.
- 승인된 임계값 `decoder_ratio_tol`, `selectivity_min`, `slope_margin` 은 `*.py` 어디에서도 참조되지 않는다(Grep 결과 0건). 코드는 판정하지 않는다. log.md 의 판정은 사람이 결과를 본 뒤 고른 지표·k·통계량으로 한 것이고, 그 선택이 사전에 고정되어 있지 않았다(C2, I2 참조).
- 문서 불일치: hypotheses.md:133 체크리스트는 "[ v] 0번 게이트 임계값"으로 체크되어 있다. 그런데 같은 파일 :144 확정 기록 2 는 "미확정(비어 있음)"이라고 쓴다. 어느 쪽이 사실인지 기록이 갈린다.
- repro_check 는 독립된 재현이 아니다. results/exp001_cs_baseline_single_s0/repro_check.json `area.groups.a026.by_k.12.ratio_ol_over_real.mean` = 3.7987… 이 `AS: diagnose/area/groups/a026/by_k/12/ratio_ol_over_real/mean` 의 seed 0 값 3.7987… 과 같다. 같은 100개 에피소드를 다시 잰 것이다.

### C2. "D: 사전 등록 기준을 수치상 충족"은 과장이다 [확인 + 일부 추정]

- 사전 등록 문장(hypotheses.md:113): "decode(encode(real))/real 이 **9개 크기 모두** 0.9 이상**이고, 모델 출력(1-step 또는 k-step)은 그보다 유의하게 작다**." log.md 는 앞부분만 확인했다.
  - 작은 천에서는 모델 출력이 recon 보다 작지 않다. 오히려 크다. open-loop 기준 a026 `AS: diagnose/area/groups/a026/by_k/12/ratio_ol_over_real/mean` mean 3.987. teacher-forced 1-step 기준 a026 `.../ratio_tf1_over_real/mean` 0.965 는 recon 0.915~0.974 와 비슷하다. 그래서 작은 쪽에서는 문장의 두 번째 조건이 성립하지 않는다. 이 사전 등록 문장은 "수축"만 염두에 두고 쓴 것이라 지금 관측(작은 천은 커짐)에 그대로 적용할 수 없다.
  - k, 그리고 점추정과 CI 중 무엇으로 볼지가 사전 등록에 없다. log.md 는 k=12 점추정을 썼다. CI 하한으로 보면 seed 1 의 a026 recon 비가 기준을 넘지 못한다. results/exp001_cs_baseline_single_s1/diagnose.json `area.groups.a026.by_k.12.ratio_recon_over_real` = mean 0.915, lo 0.858, hi 0.971 로 CI 하한이 0.9 보다 낮다. 판정이 해석 선택에 달려 있다.
- 디코더 원인을 배제한 범위가 좁다. D 시험은 "인코더 출력(분포 안)을 디코더가 읽을 때"만 본다. 예측 latent 는 분포 밖일 수 있다. 예를 들어 a248 tf1 k=1 의 천 패치 상대오차는 0.600, ind 는 0.255 다(s0 diagnose.json `latent_distance.groups.a248.tf1.1.rel_err_self_cloth_patches.mean` vs `...ind.tf1.1...`). 이런 latent 를 디코더가 작게 그려도 D 시험에는 잡히지 않는다.
  - 이 가능성을 더 키우는 관측: a248 tf1 k=1 면적비는 0.722 다(`AS: diagnose/area/groups/a248/by_k/1/ratio_tf1_over_real/mean`). 그런데 같은 시점 latent 의 최근접 GT 장면은 100% 같은 크기 군집이다(s0 `latent_distance.groups.a248.tf1.1.frac_nearest_is_ind_scene.mean` = 0.0). latent 공간에서는 아직 a248 쪽에 있는데 디코딩 면적은 30% 작다. predictor 쪽 문제인지, 디코더가 분포 밖 latent 를 잘못 읽는 문제인지 이 데이터로는 갈리지 않는다 [추정].
  - 반대 근거도 있다. GT 로 학습한 probe 를 tf1 에 적용한 gt_transfer 기울기(log.md 인용 MLP 0.803)는 디코딩 면적의 압축 방향과 맞는다. 그래서 디코더만의 문제라고 보기도 어렵다. 결론: "디코더(재구성)는 주원인이 아니다"는 "GT latent 를 읽는 디코더는 정상이다"로 좁혀 써야 한다.

---

## 2. 중요

### I1. "3.99배"는 지표 정의 때문에 부풀었다 [확인]

- 정의: diagnose.py:56 `boot_ci(A[s][m,k] / np.maximum(A["real"][m,k], 1))` 는 에피소드별 비의 산술평균이다. 젠센 부등식 때문에 분모가 작은 에피소드가 평균을 끌어올린다.
- 꼬리가 무겁다. s0 repro_check.json `area.groups.a026.per_episode_ratio_ol_k12` 에 32.0, 28.36, 20.06, 11.65, 10.74, 10.38 같은 값이 있다. 같은 에피소드들의 recon 비 목록에는 0.0, 0.14 도 있다(`per_episode_ratio_recon_k12`). k=12 의 작은 천은 실제 면적이 수십 px 수준까지 줄어드는 에피소드가 있다는 뜻이다(접힘이나 가림으로 추정). `latent_distance.groups.a026.ol.12.rel_err_self_cloth_patches.n` = 82 이니, 100개 중 18개는 천 덮임 25% 넘는 패치가 하나도 없다.
- 평균끼리의 비(내 계산): a026 seed 평균 ol 1144.64 / real 484.35 = **2.36** (`AS: diagnose/area/groups/a026/by_k/12/ol/mean`, `.../real/mean`). seed 별로는 2.23 / 2.68 / 2.19. a248 은 2417.58 / 5474.55 = 0.44 로 평균비 0.434 와 거의 같다. 부풀림은 작은 천 쪽에서만 크다.
- 시간에 따른 분해(s0, repro_check.json `area.groups.a026.by_k`): real 은 k=1 708.6 에서 k=12 484.4 로 줄었다(×0.68, 실제 천이 접히고 가려짐). ol 은 755.9 에서 1077.7 로 늘었다(×1.43). "3.99"는 모델이 커지는 효과와 실제 천이 작아지는 효과를 함께 담고 있다. 모델이 천을 키운 것은 약 1.4~1.5배이고, 나머지는 분모가 줄어서 생긴 것이다(내 계산, 추정).
- 원본 ACWM 에서도 같은 방향이었다. hypotheses.md:84 (stats 가 옮긴 README 수치, results.json 아님): 작은 천 마지막 프레임 1,189(real 365). 그러니 "작은 천이 커진다"는 새 현상이 아니다. 원래 현상 이름인 "수축(size collapse)"부터 작은 천에는 맞지 않았다.

### I2. "학습 크기로 수렴" 해석: 큰 천과 작은 천이 다르고, 설계상 다른 설명과 구별되지 않는다 [확인 + 추정]

- 큰 천: a248 ol 평균 2417.6 이고 ind 기준 면적(k=12 실제 면적 중앙값) 2331.0 과 비슷하다(`AS: diagnose/area/groups/a248/by_k/12/ol/mean`, s0 repro_check.json `repro_check.groups.a248.ind_reference_area_k12_px`). "ind 기준에 자기 real 보다 가까운" 에피소드 비율은 0.85 다(`AS: diagnose/area/groups/a248/closer_to_ind_ref_than_to_own_real_k12/ol/mean`, seed 0.89/0.75/0.91). latent 최근접 GT 장면이 ind 인 비율도 0.95 다(s0 `latent_distance.groups.a248.ol.12.frac_nearest_is_ind_scene.mean`). 큰 천은 수렴 해석과 맞는다.
  - 다만 seed 2 의 a248 ol 은 1893.5 로 ind 기준(2331)보다 작다(`AS: .../a248/by_k/12/ol/mean` by_seed "2"). 학습 크기에서 멈추는 끌개라기보다 "면적 손실 + 압축"일 가능성도 남는다 [추정].
- 작은 천: ind 쪽으로 더 가까운 에피소드는 0.29 뿐이다(`AS: diagnose/area/groups/a026/closer_to_ind_ref_than_to_own_real_k12/ol/mean`, seed 0.24/0.37/0.25). latent 최근접 비 `ratio_nn_ind_over_same` = 1.004 [0.974, 1.034], `frac_nearest_is_ind_scene` = 0.57 이라 판단이 어렵다(s0 `latent_distance.groups.a026.ol.12`). 로그 공간으로 보면 ol 면적은 real 에서 ind 기준까지 거리의 절반쯤(log(1078/484) ≈ 0.80, log(2331/484) ≈ 1.57, 내 계산)까지만 갔다. "작은 천도 학습 크기로 수렴한다"는 데이터와 맞지 않는다. "큰 천은 ind 근처로 붕괴하고, 작은 천은 커지지만 대부분 자기 크기에 더 가깝다"가 데이터에 가까운 서술이다.
- 구별할 수 없는 대안: ind 는 9개 크기의 로그 면적비 범위(ln 0.259 ~ ln 2.479)에서 거의 가운데다. 명목 면적비의 기하평균은 약 0.92 다(내 계산). 그래서 "학습 크기로 수렴", "가운데로 압축", "정보를 잃어 평균으로 회귀"가 이 설계에서는 같은 예측을 낸다. 특히 probe 쪽 bias 양상(ol, ridge, s0 probe.json `pools.mean.ridge.strat_cv.ol.all.bias_log_area_by_group`: a026 +0.24, a248 −0.17)은 노이즈가 많은 특징에 ridge 를 맞출 때 생기는 전역 평균 쪽 축소로 그대로 설명된다. 이 probe 는 9개 크기 모두로 학습했으므로 그 전역 평균이 곧 ind 근처다 [추정, 강하게 의심]. probe bias 는 "predictor 가 ind 로 끌어당긴다"의 근거가 되지 못한다.

### I3. 기울기 하나로 요약하면 비대칭과 k=1 이상이 가려진다. 사전 등록의 "k=1 정상" 규칙은 코드가 계산하지 않는다 [확인]

- tf1@k12 기울기 seed 평균 0.987(`AS: diagnose/size_curve/slope_model_vs_real_group_means/tf1@k12/slope`) 이라 "1-step 은 정상"처럼 보인다. 그러나 같은 k 에서 a248 tf1 비는 0.797 이다(`AS: .../a248/by_k/12/ratio_tf1_over_real/mean`). 9점 로그-로그 직선은 큰 쪽 끝의 휘어짐을 잘 잡지 못한다.
- k=1 에서는 tf1 과 ol 이 같은 계산이다(extract.py:124-137, 첫 스텝). 그런데 이미 a248 은 0.722, a026 은 1.098 이다(`AS: diagnose/area/groups/a248/by_k/1/ratio_tf1_over_real/mean`, `.../a026/by_k/1/...`). 사전 등록(hypotheses.md:124)은 "k=1 에서 정상이고 k 증가로만 감소하면 오차 누적"이라고 했다. k=1 이 정상이 아니므로 사전 등록 문장대로라면 순수한 오차 누적이 아니다. 그런데 size_curve 는 k=3,6,12 만 계산한다(diagnose.py:147 `k_list=(3, 6, 12)`). 사전 등록이 요구하는 k=1 기울기는 산출물에 없다.
- 무엇을 판정할지 사전에 정하지 않았다(forking paths). "기울기 CI 가 1 을 제외"를 어떤 기울기(diagnose size_curve 인지 probe slope / slope_group_means 인지), 어떤 k(1/3/6/12), 어떤 latent(tf1/ol), 어떤 probe 프로토콜(5종 모두 계산, 확정 기록 2 에서 미선택)에 적용할지 정하지 않았다. log.md 는 k=12 와 k=3 을 골라 인용했다. 결과를 본 뒤 고른 것이다.
- k=3 tf1 기울기 0.747 이 k=12 의 0.987 보다 작은 이유는 log.md 도 설명하지 못했다. 작은 천의 real 면적이 k 가 커질수록 줄어(위 I1) 작은 쪽 끝점이 움직이는 것이 한 원인일 수 있다 [추정].

### I4. CI 가 실제 불확실성보다 좁다. GT 계열 수치는 사실상 n=1 이다 [확인]

- `_boot_slope`(diagnose.py:127-144)와 `boot_ci` 는 모델을 고정하고 에피소드만 다시 뽑는다. 모델 학습 시드의 변동은 들어가지 않는다. ol@k12 기울기는 seed 별로 0.237 / 0.310 / 0.150 이다(`AS: diagnose/size_curve/slope_model_vs_real_group_means/ol@k12/slope`). 시드 간 범위(0.16)가 단일 시드 CI 폭(log.md 인용, seed 0 [0.187, 0.288] 로 약 0.10)보다 크다. 사전 등록 판정 "CI 가 1 을 제외"는 시드 하나 기준이고 시드 간 변동을 반영하지 않는다.
- gt / rvit / pix probe 와 real 면적은 세 시드에서 값이 똑같다. 예: `AS: probe/pools/mean/ridge/selectivity_real_minus_permute/gt/r2_within_group` std 1.4e-16, `AS: diagnose/area/groups/a026/by_k/12/real/mean` std 7e-14. 고정된 DINOv2 와 같은 에피소드라서 당연하다. 이 수치들에 "3시드 평균"이라고 쓰면 반복이 있는 것처럼 읽힌다.
- probe CI 는 시드별 CI 경계의 평균이다(probe_stage.py:81-85). 제대로 된 구간이 아니다.

### I5. "인코더가 크기를 보유한다"는 판정은 변별력이 없다. random-ViT 가 거의 같은 수준이다 [확인]

- s0 probe.json ridge strat_cv 군집 내 R²: gt 0.877, rvit 0.816, tf1 0.830, ol 0.418, pix 0.176 (`pools.mean.ridge.strat_cv.{gt,rvit,tf1,ol,pix}.all.r2_within_group.mean`). slope_group_means 는 gt 0.987, rvit 0.996. 무작위 초기화 ViT 도 실제 프레임에서 크기를 거의 똑같이 읽는다. "GT 인코딩에 크기 정보가 있다"는 "이미지에 크기 정보가 있다"와 거의 같은 말이다.
  - 이것이 E 전체를 약화하지는 않는다. E 의 핵심은 gt → ol 에서 정보가 빠지는지인데, gt 0.877 → ol 0.418 의 낙폭은 기준선과 무관하게 크다. 약해지는 것은 "DINOv2 가 크기를 잘 표상한다"는 서술과 사전 등록의 "인코더 보유" 판정이 진단 도구로서 갖는 가치다.
- 선택성 기준 0.1 은 단위가 정해져 있지 않고, 기준 자체가 너무 쉽다. 선택성(R² 차이)은 1 을 넘기도 한다: `AS: probe/pools/mean/mlp/selectivity_real_minus_permute/gt/r2_within_group` = 1.575. permute 쪽 R² 가 음수이기 때문이다. 이런 지표에 "0.1 이상" 기준은 어떤 이미지 인코더라도 통과한다. 키 경로도 있다. log.md 는 "선택성 키 경로 확인 못 함"이라고 했지만 `probe/pools/mean/{ridge,mlp}/selectivity_real_minus_permute/{gt,tf1,ol}/{r2_overall,r2_within_group,acc_cls,acc_cls_pm1}` 로 존재한다.
- 9-class 정확도도 거의 구별하지 못한다. s0 ridge strat_cv `acc_cls.mean`: gt 0.472, tf1 0.497, ol 0.450, pix 0.465, rvit 0.409. tf1 이 gt 보다 높고 pix 가 gt 와 비슷하다. 원인 후보: RidgeClassifier 가 회귀용으로 고른 alpha 를 그대로 쓴다(probes.py:116), 그리고 크기가 순서 있는 클래스인데 one-vs-rest 로 분류한다 [추정].
- MLP pixel 기준선은 학습에 실패한 것으로 보인다. log.md 인용 MLP pix 기울기는 0.588, ridge pix 는 0.951 이다. 같은 특징인데 MLP 가 훨씬 나쁘다. max_iter 200 과 표본 단위 early stopping(probes.py:123-125)이 원인일 수 있다 [추정]. log.md 의 "pixel 기준선 0.588~0.951"은 실패한 기준선과 정상 기준선을 한 범위로 묶은 것이다.
- tf1 probe 가 높게 나오는 것은 teacher forcing 때문에 거의 보장된다. tf1 은 GT z_{k-1} 을 입력으로 받으므로 입력 프레임의 크기 정보가 출력에 남기만 해도 probe 가 읽을 수 있다 [추정].

### I6. 개입(intervene)으로는 어떤 결론도 낼 수 없다. 크기 방향 u 는 인과적으로 효과가 없다 [확인 + 추정]

- set 모드에서 나온 영(null) 결과: `results.json -> runs.exp001_cs_baseline_single_s0.intervene.modes.set_a248_to_a026.a248.tf_decoded_dlogarea_u_by_frame[*].mean` = −0.0051, −0.0029, −0.0056. 가장 큰 천의 u 투영을 가장 작은 천 평균으로 옮겨도 디코딩 면적이 약 0.5% 만 바뀐다. 무작위 방향은 약 0(`tf_decoded_dlogarea_random_mean_by_frame`). log.md 는 이 결과를 인용하지 않았다. u 는 읽기용 probe 방향일 뿐 모델이 쓰는 크기 표상은 아니라는 직접 증거다.
  - 원인 후보 [추정]: (a) u = coef/sd 를 정규화한 것이다(probe_stage.py:192-193). 분산이 작은 차원에 무게가 실린다. (b) 크기는 "어느 패치가 천인가"라는 공간 패턴으로 표상되는데, 모든 패치에 같은 벡터를 더하는 평행이동(intervene.py:36-37)으로는 그 패턴을 바꿀 수 없다.
- remove 모드의 gain 정의(aggregate.py:30-37): 에피소드마다 Σ_t(출력 투영 변화) / Σ_t(입력 투영 변화)다. 문제가 셋 있다.
  1. 무작위 방향에서는 입력 투영 변화(분모)가 0 에 가까울 수 있다. 그러면 비가 폭주하고, 그 값을 에피소드 nanmean 한다. a125 의 무작위 gain 평균 0.240(p5 0.102, p95 0.281; `AS: intervene/modes/remove/a125/tf_response_gain_along_random_r/{mean,p5,p95}`)이 다른 군집(약 0.10~0.14)보다 큰 이상치인 것도 이것으로 설명될 수 있다 [추정, 미확인].
  2. docstring(intervene.py:10)은 "same removed fraction"이라고 쓰지만 코드(intervene.py:130-131)는 무작위 방향의 투영을 전부 제거한다. 제거량이 맞춰져 있지 않다 [확인].
  3. 대조 방향이 384차원 등방 무작위 방향이다. 데이터가 퍼진 부분공간 안의 방향(분산을 맞춘 PCA 방향)이 아니다. 학습된 predictor 는 데이터 다양체 밖의 섭동을 줄이는 경향이 있으므로, 데이터와 정렬된 u 가 무작위보다 gain 이 높은 것은 크기와 무관하게 예상되는 결과다 [추정].
- 크기 자체도 경계선 수준이다. u gain 0.157~0.183 대 무작위 p95 0.134~0.165(`AS: intervene/modes/remove/{a026,ind,a248}/...`). 무작위 20개 방향의 p95 와 비교한 것이지 검정이 아니다. a125 는 u 가 무작위 범위 안에 있다.
- 결론으로 쓸 수 있는 범위: "mean-pooled 선형 방향 하나를 평행이동하거나 제거하는 개입은 출력 면적에 의미 있는 영향을 주지 않았다." 이 결과에서 predictor 가 크기 정보를 쓴다/버린다는 어느 쪽으로도 추론할 수 없다.

### I7. 마스크와 면적 지표의 타당성: 실제 프레임에서만 검증되었다 [확인 + 추정]

- maskcheck 일치는 순환적이다. results/maskcheck.json `area_by_group.*.ratio_to_ref_mean` 이 명목 면적비와 소수 셋째 자리까지 맞는다(a026 0.2588 vs 0.259, a248 2.4812 vs 2.479). 그런데 명목 면적비 자체가 "프레임 3 의 천 픽셀 수"로 만든 값이다(README:69, 사용자 README). 생성 측이 비슷한 HSV 규칙을 썼다면 이 일치는 파이프라인 일관성만 보여 준다. 마스크가 천을 정확히 잡는다는 증거는 아니다 [추정]. 군집 내 표준편차가 6~14 px(`sd_area_px`)로 극히 작은 것은 프레임 3 이 거의 결정적인 초기 상태라는 뜻이다. 이후 프레임(접힘, 가림)에서 마스크가 맞는지는 이것으로 알 수 없다.
- 디코딩된 rollout 프레임에서 마스크가 맞는지는 검증되지 않았다. recon 비 0.94~0.99 는 GT 를 디코딩한 프레임에서만의 값이다. ol 프레임이 흐릿해지면 가장자리 채도가 떨어져 S>120(config.yaml:166) 아래로 내려가므로 면적이 과소 측정될 수 있다. 영향 방향 [추정]: 큰 천 수축은 과장되고 작은 천 증가는 과소평가된다. 반대 근거: ind 의 ol/real 은 1.054(`AS: diagnose/area/groups/ind/by_k/12/ratio_ol_over_real/mean`, log.md 인용과 일치)로 이 편향이 크지 않음을 시사한다.
- frames_sample.npz(군집별 4개 에피소드의 real/recon/tf1/ol 프레임)를 눈으로 본 기록이 없다. 파일은 서버 캐시에 있고 로컬 results/ 에는 없다. 작은 천의 ol 증가가 실제로 천이 커진 것인지, 주황 번짐이나 유령 천 같은 디코더 아티팩트인지 확인되지 않았다.
- `mask.min_px: 6`(config.yaml:168)은 masks.py 에서 쓰이지 않는다. 작은 면적에서 잡음 픽셀을 거르는 장치가 없다 [확인]. 작은 천 분모 문제(I1)를 키운다.
- ACWM 대비 IoU 0.68 은 외적 타당성 문제다. cloth_sim 내부 비교에는 영향이 없다. 하지만 "ACWM-Phys 의 수축 현상을 재현했다"고 쓰려면 이 차이를 함께 적어야 한다. 렌더러와 동역학이 다르다.

### I8. baseline 설정이 "DINO-WM baseline"으로서 충분한지 [확인 + 추정]

- num_hist=1(config.yaml:187), proprio 는 항상 0(data.py:120-143, extract.py:120), 15 epoch. acwm-phys 의 Cloth Move job 설정이고 DINO-WM 기본 설정(num_hist 3)은 아니다. 결과의 일반화 범위는 "acwm-phys 설정의 DINO-WM 변형"이다.
- 수렴 증거가 없다. results.json 에는 마지막 epoch 로그만 있다(`runs.exp001_cs_baseline_single_s0.train.last_epoch_log`: train_z_visual_err_full 0.168, val_z_visual_err_full 0.178, val_z_visual_err_rollout 0.332). 학습 곡선이 평탄해졌는지 알 수 없다. ind 에서 ol/real 비가 약 1.0 이므로 분포 안에서는 모델이 작동한다. 하지만 학습이 부족한 모델일수록 분포 밖에서 prior 로 끌려갈 가능성이 크다 [추정]. epoch_loaded=15 는 설정대로다(s0 diagnose/repro_check `epoch_loaded` 15).
- cloth_sim 쪽 학습 설정(가정 2: num_hist/frameskip/epoch 가 같다)은 미검증이다(README:48). 환경도 다르다. 실행 환경 torch 2.7.1+cu128(`results.json -> environment.torch`), 브랜치 environment.yaml 은 torch 2.3.0(README:62).
- 행동과 위치의 분포 밖 문제가 통제되지 않았다. 행동 정규화는 학습 manifest(ind) 통계를 쓴다(data.py:180-186). 크기가 다르면 그리퍼 공의 시작 위치(천 모서리)가 다르다. 행동 궤적의 분포가 크기별로 다른지는 확인하지 않았다. "시각적 크기"가 아니라 "그리퍼 위치/행동 분포 밖"이 원인일 수 있다 [추정]. metadata.pt 의 actions, x0 를 크기별로 비교하면 확인할 수 있다(GPU 불필요).

---

## 3. 사소

- S1. `in_train` 제외는 cloth_sim 에서는 아무 일도 하지 않는다 [확인]. eval_episode_table 이 `*_train` split 을 아예 빼므로(data.py:64-65) extract.py:205 의 `in_train` 은 항상 False 다. 모든 군집 `n_excluded_in_train` = 0 (s0 repro_check.json). 실제 누수 위험은 다른 데 있다. 생성기 seed 규칙(테스트 +500,000)이 train/test 초기 상태나 행동 열의 중복을 막는지는 미검증이다(README:83 "생성 쪽 규칙").
- S2. probe 의 episode 분할은 올바르다 [확인]. strat_folds / logo_folds 가 episode 를 단위로 쓰고(probes.py:59-72), 학습/평가 마스크도 episode 기준이다(probes.py:229-240). 같은 episode 의 프레임이 학습과 평가로 나뉘지 않는다. 남은 작은 누수:
  - ridge alpha 를 첫 fold, 첫 seed 에서 고른 뒤 `alpha_cache` 로 다른 fold 와 seed 에 재사용한다(probes.py:112-114; factory 는 kind/pool 마다만 새로 만듦, probe_stage.py:138). 다른 fold 의 평가 episode 가 alpha 선택에 들어간다. 후보가 5개뿐이라 영향은 작다 [추정].
  - MLP early_stopping 의 검증 분할이 표본(프레임) 단위다(sklearn 기본). 정지 시점에만 영향을 준다.
  - size_direction u 는 개입 대상 episode 를 포함한 전체로 맞췄다(probe_stage.py:185-196). 예측 평가가 아니므로 영향은 작다.
- S3. 라벨 정의 일관성 [확인]. make_samples(probes.py:42-51): gt/rvit/pix 는 k=0..12 를 각자 프레임 면적으로, tf1/ol 은 k=1..12 를 타깃 프레임 면적으로, h0..h6 은 `mk2(feat,1,1)` 로 입력 프레임(k−1) 면적을 라벨로 쓴다. 코드는 README:377 과 일치한다. 다만 층별 곡선(h6, 라벨은 입력 프레임)과 tf1 출력(라벨은 타깃 프레임)을 같은 축에 놓으면, "predictor 처리"와 "라벨이 한 프레임 이동"이 섞인다. 작은 천은 k=1→2 사이 real 면적이 708.6 → 625.4 로 변하므로(s0 repro_check `a026.by_k`) 이 차이를 무시할 수 없다.
- S4. ind 군집의 latent NN 비는 정의상 1.0 이다(diagnose.py:114-116, ind 에서는 d_ind 와 d_same 이 같은 집합). 정보가 없는 행이다.
- S5. 재현성 문서가 낡았다. README:3 은 아직 "실행 전", README:379 는 "학습 시드는 baseline 1개"라고 쓴다. results.json 에는 `experiment_code_git`(9867860…)만 있고 작업 트리 dirty 여부는 없다(README 는 run_meta.json 에 있다고 함, 로컬 미확인). logs/overnight_status.json 이 갱신되지 않은 버그(log.md 기록)도 있어 단계별 시각은 overnight.log 로만 확인된다. Drive 폴더 이름에 `cs` 가 빠져 있다(log.md).
- S6. diagnose.py:56 과 diagnose.py:71 은 분모에 `np.maximum(real,1)` 하한만 둔다. 분모가 작은 에피소드를 거르거나 로그비로 요약하는 장치가 없다(I1 의 원인).

---

## 4. 질문별 답 (호출자의 8개 점검 항목)

1. 마스크/면적 지표: 실제 프레임에서는 일관되지만 검증이 순환적이다(I7). 디코딩된 rollout 프레임에서는 미검증이다. 면적비 결과를 실제로 좌우하는 것은 마스크보다 비의 정의(I1)다. 작은 천 3.99 → 2.36 처럼 같은 데이터에서 지표 선택만으로 1.7배 차이가 난다. ind 의 ol/real 이 약 1 이라 큰 천의 수축 방향 자체는 마스크 편향만으로는 설명되지 않는다 [추정].
2. episode 누수: probe 분할은 episode 단위로 올바르다(S2). in_train 제외는 cloth_sim 에서 아무 일도 하지 않는다(S1). 라벨 정의는 코드와 README 가 일치하지만 층별 곡선 비교에서 라벨이 한 프레임 어긋난다(S3).
3. open-loop 작은 천 3.99배: 원인 후보를 근거가 강한 순서로 (a) 평균-of-비와 작은 분모(확인), (b) 실제 천은 시간에 따라 작아지는데 모델은 그것을 예측하지 못하고 키움(확인, 분해 I1), (c) ind 쪽 prior(부분적, 에피소드의 29% 만 ind 쪽이 더 가까움), (d) 디코더 아티팩트나 마스크 임계값(미확인, frames_sample 미점검). "수축이 아니라 학습 크기로 수렴"은 큰 천에만 맞고, 설계상 "가운데로 압축"과 구별되지 않는다(I2).
4. repro_check 게이트: 코드는 판정하지 않고(diagnose.py:67-72), overnight 에서 게이트가 꺼져 있었고, 사람이 보기 전에 진단이 끝났다. 승인된 임계값 3개는 코드에 연결되어 있지 않다(C1). 사전 등록 규칙이 코드로 구현된 부분은 없다. 기록만 있다.
5. random-ViT/pixel 기준선: "GT 에 크기 정보가 있다"는 주장을 무너뜨리지는 않지만, 그 주장을 사소한 것으로 만든다. 사전 등록의 인코더 보유 판정과 선택성 0.1 기준은 변별력이 없다. E 의 정보가 쓸모 있는 부분은 gt → ol 낙폭이다(I5).
6. intervene remove gain: 비의 정의 때문에 분모 폭주에 취약하고, 대조가 맞춰져 있지 않다. set 모드의 디코딩 면적은 거의 영이다. 결론으로 쓸 수 있는 범위는 "이 개입은 효과가 없었다"까지다(I6).
7. epoch 15: acwm-phys 설정대로이고 분포 안에서는 작동한다. 수렴 증거(곡선)는 없고 num_hist=1 등으로 DINO-WM 기본 설정과 다르다(I8).
8. log.md 의 과장: (i) "D 사전 등록 기준 충족"은 조건 절반만 확인했고 CI 기준에 따라 판정이 달라진다(C2). (ii) "학습 크기로 수렴"을 작은 천까지 일반화했다(I2). (iii) 3.99 는 부풀려진 지표다(I1). (iv) 선택성 키가 존재하는데 "확인 못 함"이라고 썼다(I5). (v) 개입에서 set 모드 영 결과를 빠뜨렸다(I6). (vi) "pixel 0.588~0.951"은 실패한 MLP 기준선을 섞은 것이다(I5). 반면 log.md 는 해석을 "추정"으로 표시했고 사용 여부 결론을 보류했다.

## 5. 대안 설명 (같은 결과를 더 단순하게 설명하는 가설)

- A1. 가운데로 압축 / 평균 회귀: 분포 밖 입력에 대해 predictor(와 그 latent 를 읽는 디코더)가 학습 분포의 평균 외형 쪽으로 출력을 줄인다. ind 가 평가 범위의 가운데에 있어 "학습 크기로 수렴"과 구별되지 않는다(I2).
- A2. 위치/행동 분포 밖: 크기 자체가 아니라 그리퍼 시작 위치와 행동 궤적이 분포 밖이어서 그렇다(I8).
- A3. 디코더가 분포 밖 latent 를 잘못 읽음: 예측 latent 가 GT 다양체 밖에 있을 때 디코더가 학습한 ind 외형으로 그린다. 특히 tf1 k=1 의 a248 0.72(I-C2).
- A4. 지표 효과: 작은 천의 "3.99배"는 상당 부분 분모가 작아진 효과다(I1).

## 6. 이 결과를 믿으려면 추가로 필요한 실험 (최대 3개)

1. 가운데가 아닌 크기 하나로 학습한 대조군(GPU, 사람 승인 필요). 예: a040_train 2000 또는 a160_train 2000 단일 크기, 같은 예산과 시드 3개. 큰 천과 작은 천의 출력이 *새 학습 크기*로 모이면 "학습 크기로 수렴", 여전히 가운데(ind 근처)나 평균 쪽으로 모이면 A1 을 지지한다. 현재 설계의 가장 큰 교란(I2)을 직접 끊는 실험이다.
2. 디코더를 거치지 않는 면적 판독과 마스크 검증. 이미 있는 캐시로 대부분 가능하고 GPU 는 작다. (a) 예측 latent 마다 latent 공간 최근접 GT 프레임의 실제 마스크 면적을 "검색 면적"으로 정의해 디코딩 면적과 비교한다. predictor 와 디코더의 기여를 분리한다(A3). (b) frames_sample.npz 와 rollout 디코딩 프레임 50~100장을 사람이 직접 라벨링해 HSV 마스크의 정밀도와 재현율을 잰다(I7). (c) 면적비를 평균끼리의 비, 에피소드 로그비의 평균, 중앙값으로 다시 보고하고, k=1 기울기를 추가한다(I1, I3). (c)는 재분석이다.
3. 공간적으로 맞춘 인과 개입. 단일 방향 평행이동 대신, 입력 프레임 0 의 이미지에서 천을 합성으로 확대/축소하거나 큰 천 episode 의 천 패치 토큰을 작은 천 episode 의 대응 토큰으로 바꿔서(패치 단위 교체) 예측 면적이 따라가는지 본다. 대조 방향은 데이터 부분공간에서 분산을 맞춘 방향으로 바꾼다(I6). 이것 없이는 E 의 "사용 여부"에 대해 아무 말도 할 수 없다.

## 7. 사람에게 요청할 결정 사항

1. 0번 게이트 처리: (a) "판정 불가(임계값 미설정, 게이트 꺼짐)"로 공식 기록하고 D/E 관찰을 모두 "전제 미확인 조건부"로 둘지, (b) 사후 기준임을 명시하고 지금 임계값을 정할지. 어느 쪽이든 hypotheses.md:133 체크 표시와 :144 기록의 불일치를 바로잡아야 한다.
2. D 판정 해석의 고정: 점추정과 CI 하한 중 무엇을 볼지, 어떤 k(12 만 / 전 k)로 볼지, "모델 출력이 유의하게 작다"를 작은 천(출력이 커짐)에 어떻게 적용할지. 지금 정하면 사후 결정이라는 점을 함께 기록해야 한다.
3. E 기울기 판정 대상의 고정: diagnose size_curve 인지 probe 인지, k, tf1/ol, probe 프로토콜(확정 기록 2 에서 미선택). 마찬가지로 사후 결정임을 기록해야 한다.
4. 면적비 주 지표 변경 여부: 에피소드별 비의 평균에서 평균끼리의 비, 로그비, 중앙값 중 무엇으로 바꿀지. log.md 와 보고서의 "3.99배" 표기를 유지할지.
5. 추가 실험 승인: 6절 1번(가운데가 아닌 크기 단일 학습, GPU 시드당 약 5.5시간; overnight.log 기준 seed 0 학습 05:20 → 10:55). 2번과 3번은 소규모 GPU 와 사람 라벨링이 필요하다.
6. 서버 산출물 확보: frames_sample.npz 와 추출 캐시, 체크포인트를 받거나 올릴지(`sync --ckpt --cache`). 시각 점검(I7)과 재분석(6절 2번)에 필요하다.
7. log.md 2026-11-02 해석 문구 수정 여부(C2, I1, I2, I5, I6 항목). reporter 종합 전에 원문을 고칠지, 리뷰 의견으로만 병기할지.

## 8. 확신도

- 코드 관련 지적(C1 의 게이트·임계값 미연결, I1 지표 정의, I3 의 k_list, I6 의 gain 정의와 set 모드 영 결과, S1~S3): **높음**. 해당 줄과 JSON 값을 직접 읽어서 확인했다.
- 해석 관련 지적(C2 의 디코더-분포 밖 latent 가능성, I2 의 가운데 압축 교란, I5 의 분류기 문제 원인, I6 의 a125 분모 폭주 원인, I7 의 흐림-채도 편향 방향, I8 의 행동 분포 밖): **중간**. 코드를 실행하지 못했고 frames_sample 과 캐시를 볼 수 없어 메커니즘은 추정이다. 특히 I6 의 a125 원인과 I7 의 편향 크기는 확인하지 못했다.
- 전체 결론("측정은 대체로 믿을 만하지만, 사전 등록 기준으로 D/E 판정을 내릴 근거는 없고 핵심 해석 두 가지('D 충족', '학습 크기로 수렴')는 과장되었다"): **중간~높음**. 수치 방향(큰 천 open-loop 면적 감소, ind 정상)은 세 시드 모두에서 일관되어 의심하지 않는다. 의심하는 것은 그 수치에서 원인 층위로 넘어가는 추론이다.
