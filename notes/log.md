# 실험 로그

(날짜 / 실험 ID / 변경점 / 결과 / 해석 / 다음 단계)

## 2026-11-01 / 001_size_collapse_diagnosis / repro_check 수정 (미실행)
- 변경점: src/exp001/extract.py:43 `torch.load(ck, map_location=device)` -> `weights_only=False` 추가 + "자체 학습 체크포인트만 읽음(신뢰)" 주석. 원인은 torch 2.7.1 기본 weights_only=True 의 UnpicklingError (서버 로그, 3 seed x 3회 실패). 다른 체크포인트용 torch.load 는 없음 (나머지는 metadata.pt / drive.py 의 weights_only=True 로 그대로 둠).
- 결과: 코드 수정만 함. repro_check 는 아직 실행하지 않음 (로컬에 GPU/데이터 없음, 서버 접근 불가). 학습 재실행 없음.
- 재시도 카운트: 이번 수정 후 새 시도로 기록 (최대 3회).
