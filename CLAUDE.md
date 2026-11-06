# 연구실 규칙

## 연구 주제
Physics Aware World Model. The baseline is DINO-WM

## 작업 방식
- 에이전트끼리의 의존은 "파일"로 한다. 각자 정해진 입력 파일을 읽고 정해진 출력 파일을 쓴다.
- 실험 전에 notes/hypotheses.md 에 가설과 성공 기준을 먼저 적는다.
- 실험은 experiments/NNN_이름/ 에 둔다 (config.yaml, run.py, results.json, README.md).
- 논문/노트에 쓰는 모든 수치는 results.json 에서 가져온다. 기억이나 추정으로 쓰지 않는다.
- 실험이 끝나면 notes/log.md 에 날짜, 변경점, 결과, 해석을 추가한다.
- 확신 없는 주장은 "추정"이라고 표시한다. 없는 논문/수치를 지어내지 않는다.

## 사람 승인이 필요한 지점
- 가설 확정
- GPU/비용이 큰 실험 실행 전
- 논문에 들어갈 주장 확정

## 환경
- Conda 환경에 맞춰서 실행
- 랜덤 시드는 항상 config.yaml 에 명시한다.

## 예산 / 종료 조건
- 한 실험의 재시도는 최대 3회. 3회 실패하면 멈추고 원인을 보고한다.

## 팀 구성
- experimenter (1명): 코드 작성과 실험 실행을 하는 유일한 사람. 해석은 하지 않는다.
- analyst-prior / analyst-stats / analyst-skeptic (3명): 서로 다른 관점의 자료조사자 겸 결과 리뷰어.
  서로의 파일을 읽지 않고 독립적으로 쓴다. 결과는 notes/research/, notes/reviews/ 에 남는다.
- reporter (1명): 모든 자료를 종합해 reports/ 에 보고서를 쓴다. 새 사실을 만들지 않는다.

## 흐름
1. /research 주제  → 분석가 3명 독립 조사 → reporter 종합 → 내가 가설 선택
2. /new-experiment 이름 → experimenter 구현 및 실행
3. /review-experiment ID → 분석가 3명 독립 리뷰 → reporter 보고서 → 내가 결정

## 규칙
- 실험 결과를 해석한 뒤에는 반드시 /review-experiment 를 거친다.
- 분석가들의 의견이 갈리면 하나로 합치지 않고 그대로 보고한다.
