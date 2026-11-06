---
description: 실험 하나를 분석가 3명이 독립 리뷰하고 리포터가 보고서를 쓴다. 사용법 /review-experiment 001_이름
---
대상 실험: experiments/$ARGUMENTS

1. 먼저 experiments/$ARGUMENTS/results.json 이 있는지 확인한다. 없으면 멈추고 알린다.
2. analyst-prior, analyst-stats, analyst-skeptic 을 병렬로 호출해 각자 "결과 분석 모드"로
   이 실험을 리뷰하게 한다. 서로의 리뷰를 보여 주지 않는다.
3. 세 리뷰가 모두 notes/reviews/ 에 생겼는지 확인한 뒤, reporter 에게 실험 보고서를 쓰게 한다.
4. 내게는 reports/ 의 보고서 경로, 한 문단 결론, 분석가 간 의견이 갈린 지점, 내가 결정할 사항을 보고한다.
