# 실험 설명: 015_size_interchange

상태: 가설·설정 사용자 확정(2026-11-12), **코드 작성 완료(2026-11-12, CPU selftest 96개 통과; 사용자 구현 결정 #4/#5/#11 반영 후 재실행), GPU 실행 미승인·미실행**. 결과 수치 없음. config.yaml 은 사용자 확정값이며, 사용자 결정(2026-11-12, `notes/hypotheses.md` 끝의 "Implementation-choice record")에 따라 `controls`(크기를 맞춘 가짜 교체)와 `gate`(20회 분할 평균)만 바꿨고 `pilot: true`, `seed: 0` 은 유지했다. 구현 선택은 `CHANGES.md`(사람 확인 필요 표시 포함).

- 검증할 가설: notes/hypotheses.md "### 015 Size interchange" 절과 끝의 결정/확정 기록(2026-11-12).
  - 질문: predictor 입력의 크기 단서를 다른 레벨의 것으로 바꾸면(패치별 레벨 평균 교체) 예측된 움직임이 그 레벨의 실제 움직임 쪽으로 이동하는가.
- 방법
  - 조건 (a) `single_ind`, (b) `multi_60_ind_160`, 학습 seed 0,1,2, `model_latest`. 보고용 기준선: random-init predictor(`DP.random_init_predictor(model, 1234 + 학습 seed)`).
  - 레벨 평균 m_L: 학습 레벨(a060, ind, a160)의 train split episode(레벨당 100), 입력 프레임 0..11 에서 패치 위치별 인코더 latent 의 z 부분(384) 평균(196 패치). 013 `level_mean_match` 의 학습 레벨 규칙과 같다.
  - 쌍 (S, T): 세 레벨 사이 6개 방향. 평가는 S 의 test episode(100), episode 당 12 전이, 한 step teacher-forced 예측.
  - 편집: z' = z + (m_T − m_S) (z 부분 전체 토큰, 위치 임베딩 전, 013 과 같은 위치). 행동/proprio 좌표는 그대로.
  - Δ̂(x) = 예측(x) − x. d = Δz̄_T − Δz̄_S (레벨별 실제 움직임 z_{t+1} − z_t 의 train split 평균, 패치 x 384). moved = (Δ̂(z') − Δ̂(z))·d / |d|² (패치 x 384 를 펴서), episode 당 12 전이 평균. 0 = 이동 없음, 1 = T 의 평균 움직임으로 완전 이동.
  - 대조: **주 비교값 = 크기를 맞춘 가짜 교체(사용자 결정 2026-11-12)**: S 의 train episode 를 무작위 반반으로 나눈 평균 차이(20회, seed 15000, 기존 추첨 그대로)를 쌍마다 패치별로 fake[p] × |m_T[p] − m_S[p]| / |fake[p]| 로 재스케일(쌍별이라 두 목표 T 가 공유하지 못함; |fake[p]| ≤ 1e-12 인 패치는 0 으로 두고 fit.json 에 개수를 기록). 재스케일 전 가짜 교체(unscaled)는 보고만. 무작위 방향 20개(seed 13000, 패치별 shift norm 을 |m_T − m_S| 에 맞춤; 보고만), random-init predictor(보고만).
  - 관문(사용자 결정 2026-11-12): 쌍마다 d 의 split-half cosine 을 train episode 무작위 반반 20회 분할(RandomState(seed + 1))에서 구해 평균한 값 ≥ 0.5. 한 번의 분할(RandomState(seed)) cosine 은 보고용 진단. 미달 쌍은 보고만 하고 판정에서 뺀다. 통과 쌍이 0개면 `measure` 는 거부된다(exit 4, `--allow-no-gated-pairs` 로만 무시)/노트북은 멈춘다.
  - 판정 지표: 조건 (b), 관문 통과 쌍 평균 (moved − 크기를 맞춘 가짜 교체 20회 평균), 학습 seed 마다, 90% CI(episode bootstrap 2000회, seed 14000, 원천 레벨 층화, 인덱스는 조건/seed 간 동일).
- 성공 기준(사용자 확정 2026-11-12): `criterion_met` iff (1) CI 하한 > 0 인 학습 seed 3개 중 2개 이상, (2) seed 평균 ≥ 0.10, (3) (b) seed 평균 > (a) seed 평균, (4) 관문 통과 쌍마다 S→T, T→S 모두 (b) seed 평균 (moved − 크기를 맞춘 가짜 교체) > 0. (판정의 "가짜 교체"는 모두 크기를 맞춘 것, 2026-11-12 사용자 결정.) 아니면 `criterion_not_met`(사용 안 함이 아님), 입력 누락/NaN 이면 `not_evaluable`, 관문 통과 쌍이 하나도 없으면 `not_evaluable`.
- 이 실험으로 주장하려는 것: 위 판정과 results.json 의 수치만. "크기 교체 shift 가 예측 움직임을 그 레벨 쪽으로 옮겼다"(검출)와 "검출되지 않았다"만 말한다. 검출 안 됨은 "사용 안 함"이 아니다. 평균만 바뀌는 교체이고 패치 배치는 S 그대로이므로 moved ≈ 0 의 해석은 모호하다(양성 결과만 정보가 있음).

## 실행 방법
코드: `run.py` + `src/exp015/{base,lib,core,stages,selftest}.py`. 013 `exp013.*`(그 아래 009/004/002)는 import 만 한다(수정 없음).
```
conda activate <002 와 같은 환경>     # CPU 확인은 ~/miniconda3/envs/lab07/bin/python
python run.py selftest                # CPU, 합성 데이터 + toy 모델 (GPU 불필요)
python run.py check                   # config 일치, 체크포인트 폴더, Q1 캐시, metadata.pt, fit 상태
python run.py fit                     # CPU (Q1 캐시 필요): m_L, d, 관문(20회 분할 평균), 재스케일 전 가짜 교체 shift -> $EXP015_CACHE/fit/interchange_fit.npz, results/fit.json
python run.py measure                 # GPU: 단위(조건 x 학습 seed) 6개 -> results/parts/unit_<cond>_s<seed>.{npz,json}. fit 에서 통과 쌍이 0개면 거부(exit 4); --allow-no-gated-pairs 로 무시
python run.py aggregate               # CPU: 쌍별 통계, bootstrap, 판정, results/results.json
```
모든 단계가 `--dry-run` 을 지원한다. 옵션: `--results-dir`(파일럿 폴더, results.json 은 그곳에만 쓰고 `pilot: true`), `--conditions`, `--train-seeds`, `--episodes-per-level N`(원천 레벨마다 앞 N episode 만), `--batch-episodes`(기본 8), `--fresh`. 환경변수: `DINO_TRAIN`(체크포인트, 읽기 전용), `EXP002_CACHE`(Q1 캐시, 읽기 전용), `EXP015_CACHE`(fit 결과), `ACWM_PHYS_SRC`.
종료코드: 0 정상, 1 실패(시도 1회 소비, `results/attempts.json`, 같은 단계 연속 3회 실패하면 거부), 3 config null, 4 사전 점검 실패(시도 소비 없음).
`pilot` 규칙(013 과 같음): results.json 의 `pilot` 은 config `pilot: true`, 별도 결과 폴더, CLI 덮어쓰기에서 계산되고 이유는 `pilot_reasons`. 본 실행 전에 사람이 config `pilot: false` 로 바꾼다.

### Colab (사용자가 직접 실행)
`colab/exp015_colab.ipynb`(`python colab/build_notebook.py` 로 재생성; 코드를 바꾸면 다시 생성). 사전 조건: 015·013·009·004·002 코드가 `REPO_REF` 에 커밋/푸시되어 있을 것(노트북 A3 가 확인), Drive 에 `exp002_work`(체크포인트 6개, Q1 캐시) 가 있을 것.
1. Colab 에서 GPU 런타임(L4 또는 A100)으로 노트북을 연다. 셀 A1~A7 순서대로 실행(A2 에서 Drive 경로·`REPO_REF` 확인).
2. B1, B2(입력 점검, `check`, dry-run), C1(selftest, CPU)을 실행. 모두 오류 없이 끝나야 한다.
3. E0 `fit`(CPU, 승인 플래그 없음): 관문 결과(쌍별 cosine, 통과 쌍)와 가짜 교체/실제 shift 크기 비를 출력한다. 통과 쌍이 0개면 파일럿 셀(E1)과 본 실행 셀(E3)이 `gate_guard`(셀 A6c)로 measure 전에 멈춘다(사용자 결정 2026-11-12).
4. (선택) 파일럿: D0 에서 `APPROVED_PILOT_015 = True` 로 바꾸고 E1 실행(조건 (b), seed 0, 레벨당 10 episode; 결과는 `exp015_work/results_pilot`). E2 가 소요 시간을 출력한다. 파일럿 verdict 는 (a) 가 없어 `not_evaluable` 이 정상.
5. 본 실행: 사람이 config `pilot` 을 `false` 로 바꾸고(또는 `ALLOW_MAIN_WITH_PILOT_TRUE = True`) D0 에서 `APPROVED_MAIN_015 = True` 로 바꾼 뒤 E3(`measure`), F1(`aggregate`). 끊기면 A, B 후 E3 부터 다시(끝난 단위는 건너뜀).
6. G1 이 Drive 의 결과 목록을 보여준다. `MyDrive/exp015_work/results/results.json` 이 결과. 노트북은 해석을 하지 않는다.

## 비용 추정 (추정, 실측 없음)
- `fit`: CPU, Q1 캐시에서 학습 split 300 episode 읽기. 013 의 fit 이 12.7 s 였음(`notes/log.md`). 수십 초~수 분 (추정).
- `measure`: 단위 6개 x 원천 episode 300(레벨 3 x 100) x 12 전이 x 순전파 103 (편집 없음 1 + 재스케일 전 가짜 20 + 목표 2 x (실제 1 + 크기를 맞춘 가짜 20 + 무작위 20)) x predictor 2 = 단위당 741,600 회(이전 설계는 순전파 63 으로 단위당 453,600 회). 013 은 단위당 900 x 12 x 43 x 2 = 928,800 회였고 실측 12627 s(6단위, 사용자 L4). 순전파 수 비율 741,600 / 928,800 = 0.80 을 적용하면 약 10,080 s = 약 2.8시간, 범위 2.5~4시간 (추정; 순전파당 비용이 013 과 같다는 가정, 015 실측 없음; 사용자 결정 기록의 "약 1.7 h → 약 2.8 h"와 같은 환산). 파일럿(레벨당 10 episode, 1단위)은 모델 로드 포함 수 분 (추정).
- `aggregate`: CPU, 수 초~수십 초 (추정; 013 aggregate 9.4 s).
- 본 실행 전 승인 필요(CLAUDE.md).

## 알려진 한계
- 평균만 바뀐다: 패치 배치(천이를 덮는 패치)는 S 그대로라 moved ≈ 0 은 해석이 모호하다(양성 결과만 정보가 있음).
- 더한 변위가 출력에 통과하면 shift 와 d 가 정렬될 때 moved 가 오를 수 있다. Δ̂ 를 편집된 입력 기준으로 잡았으므로 완전 복사는 0 이 되지만, 부분 통과·민감한 방향은 가능하다. 쌍별 복사통과 진단값 (m_T − m_S)·d/|d|² 와 가짜 교체의 같은 값을 `fit.json` 에 두었다.
- 주 비교값(크기를 맞춘 가짜 교체)은 방향이 "S 의 episode 반반 표집 잡음 방향"이고 크기만 실제 shift 에 맞춘 것이다. 재스케일 전 가짜 교체는 100개 episode 의 절반끼리 평균 차이라 실제 shift 보다 훨씬 작을 가능성이 높아(`fake_unscaled_over_real_frob` 로 기록) 보고만 한다. 크기를 맞춘 무작위 방향 대조도 보고만 한다(판정 아님).
- 레벨 간 실제 움직임 차이가 작다(013 관문 비 1.04-1.46). d 의 신뢰도는 20회 무작위 분할 평균 split-half cosine 관문으로 보고, 한 번의 분할(seed 0) cosine 은 보고용 진단으로 함께 기록한다.
- 학습 seed 3개와 episode bootstrap 은 서로 다른 불확실성 축이다. 판정 (1) 은 seed 간 일관성만 요구한다.
- 레벨 3개(a060, ind, a160)만 쓴다. 학습 레벨 사이의 교체이며 OoD 레벨은 포함하지 않는다.
- 실제 predictor 는 CPU 에서 실행할 수 없어(013 README 참고) CPU selftest 는 toy 모델과 합성 데이터만 쓴다. 실제 체크포인트, DINOv2 인코더, Colab 환경, GPU 메모리/시간은 미검증.
