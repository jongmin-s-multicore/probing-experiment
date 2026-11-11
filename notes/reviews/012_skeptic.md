# 012_erasure_residual_info 회의론자 리뷰 (analyst-skeptic, 2026-11-11)

입력: experiments/012_erasure_residual_info/{results.json, results/colab/{code_version,timing_main}.json, config.yaml, README.md, CHANGES.md, src/exp012/{measure,stages,verdict}.py}, experiments/009_positive_control_depth/src/exp009/erase.py, experiments/002_single_size_sweep/{config.yaml, src/exp002/probes.py, model_io.py}, notes/hypotheses.md 012 절(397-418행), notes/log.md 2026-11-11 012 항목(360-367행).
다른 분석가의 012 파일은 읽지 않았다. 코드는 실행하지 않았다. "확인"은 코드나 results.json 을 읽어서 확인한 것, "추정"은 읽어서 의심되는 것이다.
모든 수치는 experiments/012_erasure_residual_info/results.json (판정 window late, 깊이 m_int, 강도 1.0, 별도 표시 없으면 split all) 에서 가져왔다.

## 요약

블록별 판정 결과는 (a) 전 블록 not_met, (b) 전 블록 met, (c) 블록 3-6 met 이다. 이 판정 자체는 사전 등록된 규칙대로 계산되었다. 그러나 (c) 는 소거와 구조적으로 무관한 값이다. (a) 는 판정 window 를 바꾸면 결론이 뒤집힌다. (c) 의 met 은 split all 에 포함된 extrap episode 에 기대고 있다. 그래서 "소거 후 잔여 정보가 어떤 형태로 남았는가"라는 질문에 데이터가 실제로 답하는 것은 (b)-MLP 하나뿐이다. 그것도 스칼라(rank-1) LEACE 의 이론적 보장 범위(선형 공분산만 제거)에서 예상되는 결과이다.

---

## 치명

### F1. (c) "행동 경로" 는 깊이 m_int 에서 소거와 무관하다. 강도 0 과 값이 정확히 같다 (확인)
- 코드: `exp009/erase.py` `erase_stream` (63-74행) 은 `x[..., :dz]` 만 바꾸고 `x[..., dz:]` 를 그대로 이어 붙인다. `exp012/stages.py` `extract12` (147행) 는 (c) 의 입력 `pa` 를 같은 깊이 m 의 `full[..., dz:dz+da]` 에서 꺼낸다. 소거는 블록 m_int 의 출력에 적용되고 측정도 같은 텐서에서 하므로, 깊이 m_int 의 행동/proprio 좌표는 강도 0 과 1 에서 비트 단위로 같다.
- results.json: 블록 1 학습 seed 0 의 `c_action_only_linear` all 은 strength_1 0.16113520413340854, strength_0 0.16113520413340854 로 같다. 블록 3 seed 1 도 두 강도 모두 0.2844358022671078 이다. notes/log.md 364행에도 "(c) 는 강도 0 과 1 이 같고" 라고 적혀 있다.
- 결과: 블록 3-6 의 (c) criterion_met 은 "학습된 모델이 소거와 상관없이 블록 m_int 의 행동 좌표에 크기 정보를 선형으로 써 둔다"는 뜻일 뿐이다. 소거 후 정보가 그 경로로 남았다거나 회복 경로라는 증거가 아니다. 그런데 `verdict.overall.items_met` 에 b 와 나란히 `c_action_path` 로 기록되어 있어서 보고서가 "소거 후 행동 경로 검출"로 읽힐 위험이 크다. hypotheses.md 411행도 "행동 좌표는 소거와 무관하게 크기 정보를 가질 수 있다"고 미리 적어 두었다. 따라서 (c) 는 판정 항목이 아니라 기준선 속성으로 보고해야 한다.
- 소거가 (c) 에 미치는 영향은 m_int+1(보조, 판정 아님)에서만 볼 수 있다. 그 값도 일관되지 않다(아래 I5).

### F2. 009 의 "소거"는 블록 출력 전체(404차원)에서 선형 정보를 지우지 않는다. 이것이 009 회복의 더 단순한 설명 후보이다 (확인 + 추정)
- 확인: 소거 대상은 z 부분 [0:384] 뿐이다(config.yaml `apply_to: z_part_all_tokens`, `erase_stream`). 같은 깊이의 행동 좌표 [384:404] 만으로 학습한 linear R² 는 블록 6 에서 all 0.547/0.537/0.580(seed 0/1/2), split train 에서 0.401/0.353/0.425 이다. 블록 m_int 출력 hidden 전체로 보면 소거 후에도 선형 판독이 0 이 아니다.
- 추정: 009 에서 "다음 블록에서 linear R² 회복"을 비선형 잔여 정보로 설명할 필요가 없을 수도 있다. 다음 블록의 attention/MLP 가 소거되지 않은 20차원에서 선형으로 다시 써 넣을 수 있기 때문이다. 다만 블록 1 은 이 설명과 맞지 않는다. 깊이 1 의 (c) split train 이 0.077/-0.007/0.036 으로 거의 0 인데, 깊이 2 의 (d) 는 strength_1 train 0.731/0.713/0.667 로 회복한다. 그러니 행동 좌표 경로만으로는 적어도 블록 1 의 회복을 설명할 수 없다. 어느 쪽인지는 이 실험 설계로 가를 수 없다.

---

## 중요

### I1. (a) "클래스 구조 미잔존"은 판정 window 를 late 로 정한 데 좌우된다 (확인)
- `records_depth_m_int.strength_1` 의 `a_level_logistic.early.train` 은 모든 블록과 학습 seed 에서 0.72-0.81 이다. 예: 블록 1 0.723/0.723/0.750, 블록 3 0.733/0.752/0.753, 블록 6 0.767/0.733/0.814. 모두 임계 0.533 을 넘는다. 판정에 쓴 late 는 0.448-0.523 이다.
- 블록 3 seed 0 early 의 레벨별 recall 은 a060 0.6725, ind 0.92, a160 0.6075 이다. "ind 대 양 끝" 대비가 선형으로 남아 있다는 신호이다. 스칼라 LEACE 는 y 와의 공분산 방향 1개만 지우므로 레벨 3개의 평균 차이 중 y 에 선형이 아닌 성분(대략 y² 방향)은 지우지 않는다. 이론적으로 예상되는 잔존이다.
- 방향은 8프레임을 합쳐 적합했다(`pooled_hidden`, frames 전체). 그래서 프레임 구간별 선형 정보가 남을 수 있다(추정).
- late 는 009 primary timepoint 로 결과를 보기 전에 정했으므로(hypotheses 418행 (3)) p-hacking 은 아니다. 그러나 "(a) 클래스 구조가 남지 않았다"는 일반 주장은 데이터가 뒷받침하지 않는다. 쓸 수 있는 주장은 "late window 에서 0.533 미달"까지이다. late 값도 우연 수준 0.333 보다 0.11-0.19 높다(예: 블록 1 0.455/0.457/0.448). 임계값 미달이 곧 우연 수준이라는 뜻은 아니다.
- 블록 6 late 는 0.514/0.484/0.523 으로 임계 0.533 에 가깝다. 불확실성(CI) 없이 경계 근처에서 판정했다.

### I2. (c) 의 met 은 split all 에 포함된 extrap 에 기대고 있다. split train 에서는 블록 6 만 met 이다 (확인)
- 002 config.yaml 14-21행: extrap = [a026, a040, a200, a248]. 학습 범위 양쪽 바깥의 먼 레벨이라 split 안의 라벨 분산(SST)이 크다. test 900 episode 중 extrap 이 400, train 300, interp 200 이다(`records ... n_test_episodes`). R² 는 split 별로 그 split 의 평균을 기준으로 계산한다(`measure._r2`). 따라서 extrap 과 all 의 R² 는 "작은 크기 대 큰 크기"만 맞혀도 높게 나온다.
- 블록별 (c) (seed 0/1/2):
  - 블록 3: all 0.224/0.284/0.261 (met), train 0.165/0.187/0.153 (not met), extrap 0.236/0.302/0.283 (met), interp -0.411/-0.330/-0.460.
  - 블록 4: all 0.163/0.321/0.316 (2/3 met), train 0.093/0.133/0.162 (not met).
  - 블록 5: all 0.222/0.410/0.343 (met), train 0.139/0.252/0.197 (1/3, not met).
  - 블록 6: all 0.547/0.537/0.580, train 0.401/0.353/0.425 (둘 다 met).
- all 선택은 사용자가 결과를 보기 전에 확정했다(hypotheses 418행 (1)). 사후 선택은 아니다. 그러나 보고서에 "(c) 블록 3-6 met" 을 쓸 때 "split train 기준으로는 블록 6 만"이라는 사실을 같은 문장에 붙여야 한다. 블록 4 seed 0(0.163)과 블록 5 train seed 2(0.197)처럼 임계 근처 값이 판정을 가른다.
- interp 의 (c) 는 모든 블록에서 음수(-0.20 ~ -0.46)이다. 행동 좌표 readout 이 학습 레벨 사이 크기로는 일반화되지 않는다는 뜻이다. "행동 좌표가 크기를 표현한다"는 해석도 extrap 의 거친 대비에 한정된다.

### I3. (b) 의 attentive 는 소거 전후 값이 거의 같다. (b) 판정은 "소거가 attentive 가 읽는 정보를 건드리지 않았다"는 뜻에 가깝다 (확인 + 추정)
- all, seed 평균이 아니라 seed 0 값: 블록 1 strength_0 0.723 → strength_1 0.715, 블록 3 0.700 → 0.714(오히려 증가). log.md 364행 요약도 "attentive 약 0.72-0.73 -> 0.71-0.72" 이다.
- 소거 방향은 pooled(패치 평균) z 에서 적합했고 각 토큰에 같은 affine 사영을 적용한다. 토큰마다 방향 a 위 성분은 상수가 되지만, 크기가 "천이 덮는 패치 수" 같은 공간 분포로 인코딩되어 있으면 다른 방향의 토큰별 정보가 그대로 남는다(추정; hypotheses 423행의 013 설계 전제와 같은 논리). 따라서 attentive 의 ≥0.2 는 사실상 확정된 결과였고, 소거에 대한 정보량이 거의 없다.
- attentive 는 강도 0 에서도 all 기준 linear pooled 보다 낮다(블록 1 seed 0: attentive 0.723, linear 0.826). interp 는 0.24-0.32 로 약하다. split train 에서만 0.86-0.88 로 높다. split 에 따라 성능이 크게 달라지는 도구이다.
- MLP 와 attentive 차이: MLP 는 소거로 약 0.1 떨어진다(블록 1 seed 0 all 0.839 → 0.741). pooled 단위에서 정의된 소거를 평가하는 측정으로는 MLP 쪽이 해석 가능하다. OR 규칙 때문에 attentive 하나만으로 (b) 가 성립할 수 있지만, 이번 데이터에서는 MLP 단독으로도 모든 블록, 모든 split(interp 포함, 예: 블록 1 interp 0.358/0.355/0.342)에서 0.2 를 넘는다. 따라서 (b) 판정은 OR 규칙에 좌우되지 않는다.

### I4. (b) 는 rank-1 소거에서 이론적으로 예상되는 결과라 "정보 자체 제거 대 선형 투영만 제거"를 가르는 힘이 약하다. 보정 대조도 없다 (설계)
- LEACE(Belrose et al. 2023)는 선형 판독에 대한 보장만 준다. 384차원에서 방향 1개를 지운 뒤 비선형 probe 가 회복하는 것은 잘 알려진 한계이다(일반 지식이며 이 저장소 문헌으로 확인한 것은 아님, 추정 표시). 질문의 답("선형 투영만 제거")은 실험 전에도 거의 정해져 있었다.
- 측정 도구 보정이 없다. 정보가 실제로 제거된 표현(예: 라벨 섞기, 또는 레벨 평균 맞추기처럼 강한 소거)에서 MLP R² 가 0.2 아래로 떨어지는지 보여 주는 음성 대조가 없다. 그래서 0.2 임계는 관습값일 뿐이고(hypotheses 408행) 경계 해석에 근거가 없다. 이번 값은 0.74-0.77 로 임계와 멀어서 판정 자체는 이 문제에 강하다.
- "읽을 수 있음"은 "다음 블록이 사용함"이 아니다. README 18행과 hypotheses 415행이 이미 이를 한계로 적었다. 보고서 제목이나 절 이름의 "회복 경로"(hypotheses 397행 "009 의 회복 경로")라는 표현은 이 실험이 보여 주는 것보다 크다.

### I5. m_int+1 보조 관측에서 소거가 행동 좌표에 미친 영향은 seed 마다 방향이 다르다 (확인, 판정 아님)
- 블록 1 → 깊이 2, split train (c): strength_1 0.279/0.229/0.169, 같은 깊이 strength_0(블록 2 의 깊이 m_int 값) 0.146/-0.009/0.223. seed 0, 1 은 증가하고 seed 2 는 감소한다.
- 블록 3 → 깊이 4, all (c): strength_1 0.487/0.314/0.244, strength_0 0.163/0.321/0.316. seed 0 만 크게 증가한다.
- "소거하면 predictor 가 행동 좌표로 우회한다"는 해석은 이 데이터로 지지되지 않는다. 보고서에서 이를 시사하지 않도록 해야 한다.

### I6. 재현성 표기: pilot: true, git_status_dirty: true (확인)
- results.json `pilot: true`, `pilot_reasons: ["config.yaml pilot flag is true"]`. config.yaml 4행 `pilot: true` 가 그대로이다. log.md 361행은 "내용은 본 실행 전체"라고 적었다. CLAUDE.md 규칙상 수치는 results.json 에서만 가져오므로, results.json 이 스스로를 pilot 이라고 표기한 상태에서 논문/보고서 수치로 쓰면 규칙과 충돌한다. 009 에서 지적된 같은 문제가 재발했다(CHANGES.md 8행이 이 문제를 막으려고 만든 장치인데, 사람 단계(config 수정)가 빠졌다). `ALLOW_MAIN_WITH_PILOT_TRUE` 같은 우회를 썼는지, 어떤 승인 기록이 있는지는 확인하지 못했다.
- `code_version.git_status_dirty: true`. `stages.code_version` (276-288행) 은 `git -C <exp_dir> status --porcelain -- .` 로 012 폴더만 본다. 따라서 (1) dirty 의 원인이 코드 변경인지, Colab 에서 생긴 추적 안 되는 파일(results/ 연결, logs/, __pycache__)인지 기록에 없다(log.md 361행 "미확인"). 추정으로는 후자일 가능성이 높다. 로컬 작업 트리에는 012 src 변경이 없고 logs/, results/ 만 untracked 이다(git status 스냅샷). (2) import 하는 009/004/002 폴더의 dirty 여부는 아예 검사하지 않는다.
- Drive 의 `results_pilot`, part 파일, Drive 로그는 받지 않았다(log.md 362행). 파일럿을 본 뒤 설정을 바꿨는지 검증할 수 없다. config_sha256 은 기록되어 있다(현재 config 와 같은지는 실행해 보지 않아 미확인).
- README.md 3행은 여전히 "GPU 실행 미승인, 미실행. 이 문서에 결과 수치는 없다"이다. 실행 후 갱신되지 않은 낡은 상태 표기이다.

---

## 사소

- S1. probe seed 5개는 ridge 측정((c), (d))에서 독립 반복이 아니다. ridge 는 결정적이고 seed 는 GroupKFold 폴드만 바꾼다. 그 결과 alpha 가 두 값 사이를 오가고(블록 3 seed 0 (c) `ridge_alpha_per_seed` [10, 1, 10, 1, 10]), per_seed 값도 두 종류만 반복된다. "probe seed 5개 평균"은 (c), (d) 에서 변동성을 과대 표현한다. (a) 도 seed 별 값이 1개이다(README 46행이 명시함).
- S2. `verdict_by_split_b_c` 의 interp/extrap 항목에도 (a) 가 split train 값 그대로 들어가 overall 계산에 포함된다(stages.py 401행, `verdict_for` 가 a 를 항상 train 으로 읽음). split 별 overall 을 읽을 때 오해할 수 있다.
- S3. hidden 은 float16 으로 저장한 뒤 probe 에 넣는다(`extract12` 146행). 소거 후 방향 a 위 성분의 표준편차 비는 약 2.4e-7-4.5e-7(diagnostics, float32 pooled 기준)인데, float16 양자화 오차는 그보다 훨씬 크다. 소거 후 (d) all 이 0.02-0.06 으로 0 보다 약간 큰 것의 일부가 양자화 잡음 때문일 수 있다(추정, 영향은 작음).
- S4. (d) 재현 점검에서 strength 1 의 interp 차이(012 - 009)가 -0.12 ~ -0.21 로 다른 split 보다 크다(블록 1 seed 0/1/2: -0.120, -0.151, -0.212). interp 는 SST 가 작아 R² 가 불안정한 split 이다. 판정과 무관하지만 "009 재현"을 말할 때 interp 는 빼거나 따로 적어야 한다.
- S5. 데이터 분리는 문제를 찾지 못했다(확인). `probe_split`/`assert_probe_split` 이 probe 학습 episode 와 방향 적합 episode 와 test 의 서로소를 강제한다. MLP/attentive 의 조기 종료 검증 집합도 probe 학습 episode 안에서 episode 단위로 뽑는다(002 probes.py 63-76행). results.json 의 meta 는 probe 학습 450, 적합 450, test 900 episode 로 기록되어 있다. 강도 0 기준선도 같은 분리를 쓴다(같은 `fit_ep` 로 `measure_depth` 호출). 남은 비대칭은 하나다. 소거 방향이 학습 레벨 3개로만 적합되어 extrap/interp 에서는 선형 소거 보장이 없다. (d) extrap 이 0.004-0.056 정도라 영향은 작아 보인다.

---

## 대안 설명 정리

1. (b) MLP 0.74-0.77: 크기 정보가 비선형 형태로 남았다기보다, rank-1 소거가 지우지 못한 다른 선형 방향의 조합(예: ind 대 양 끝 대비, 프레임별 방향)과 약한 비선형의 합일 수 있다(추정). 어느 쪽이든 "선형 투영만 제거"라는 결론은 같다. "비선형으로 인코딩되어 있다"는 더 강한 주장은 지지되지 않는다.
2. (c): 소거가 아니라 모델이 깊이에 따라 행동 좌표에 크기를 써 넣는 기본 성질이다. 강도 0 값이 블록 1→6 에서 all 0.139 → 0.554 로 커진다(log.md 364행, results.json 블록별 strength_0 과 일치). 입력 단계(h_0)의 proprio 자체가 크기와 상관할 가능성도 있다(hypotheses 437행: 레벨마다 초기 위치 x0 가 다름). 깊이 0 의 (c) 를 재지 않아서 가를 수 없다(추정).
3. (a) late 미달: late 에서 클래스 구조가 약해서이지 소거가 그것을 지웠다는 뜻이 아니다. early 에서는 0.72-0.81 이 남는다.

## 이 결과를 믿으려면 추가로 필요한 실험 (최대 3개)

1. 404차원 전체 소거와 행동 좌표 기준선. z 와 행동/proprio 좌표를 함께 소거(방향을 404차원에서 적합)한 뒤 m_int+1 의 linear 회복이 남는지 본다. 같은 측정에서 깊이 0(h_0) 의 (c) 도 잰다. F1, F2, 대안 2 를 가르고, "행동 경로"를 판정 가능한 항목으로 만든다. hidden 재계산이 필요하므로 GPU 대상이다.
2. 측정 도구 보정(음성 대조). 같은 probe 학습 episode 에서 (i) episode 단위 라벨 섞기, (ii) 레벨 평균 맞추기(013 의 소거 (ii))나 다차원 LEACE(one-hot 3클래스 + 프레임별 방향)로 강하게 소거한 표현에 MLP/attentive/로지스틱을 적용한다. 0.2/0.533 임계가 "정보 없음"에서 실제로 미달되는지, 즉 측정이 소거를 검출할 힘이 있는지 확인한다. 특히 attentive 가 토큰 단위 소거에서도 떨어지지 않는다면 attentive 는 이 질문에 쓸 수 없는 도구이다.
3. split 과 window 강건성의 사전 등록 재집계(GPU 불필요). 이미 저장된 records 로 (a) early/late, (b)/(c) train/extrap/all 을 표 하나에 두고 학습 seed 와 episode 재표집 CI 를 붙인다. 임계 근처 판정(블록 4 (c) seed 0 0.163, 블록 5 train (c) seed 2 0.197, 블록 6 (a) 0.484-0.523)이 불확실성 안에서 뒤집히는지 확인한다. 그러려면 part 파일이나 예측값이 필요한데, 현재 받지 않았으므로 Drive 에서 받아야 한다.

## 확신도

- F1 (c 가 소거와 무관): 높음. 코드(`erase_stream` 이 `x[..., dz:]` 를 보존, `extract12` 가 같은 깊이에서 `pa` 추출)와 results.json 의 두 강도 값이 소수점 끝까지 같은 것으로 이중 확인했다.
- I1 (window 의존), I2 (split 의존): 높음. results.json 값을 직접 읽었다. 다만 그 원인(프레임별 방향, 이차 성분, extrap SST)은 추정이다.
- I3, I4 (attentive 무변화와 rank-1 소거의 예상 결과): 무변화 자체는 높음, 원인 해석(공간 인코딩)은 중간이다.
- F2 (009 회복의 대안 설명): 중간 이하. 블록 6 의 행동 좌표 선형 정보는 확인했지만, 블록 1 의 회복은 이것으로 설명되지 않는다. 실제 경로는 이 실험으로 판정할 수 없다.
- I6 (재현성): 표기 문제는 높음. dirty 의 원인은 확인하지 못했다(추정: 추적 안 되는 결과/로그 파일).
- 전체: 사전 등록된 규칙대로 계산된 판정 수치는 신뢰할 만하다(코드 버그나 데이터 누수는 찾지 못했다). 그러나 "소거 후 잔여 정보의 형태"에 대한 해석으로서, (c) 와 (a) 는 현재 판정 라벨 그대로 쓰면 과장이 되거나 반대 결론이 될 수 있다.
