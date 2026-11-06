#!/usr/bin/env bash
# AI 연구실 초기 세팅 스크립트 (macOS / Linux / WSL)
# 사용법: bash setup_lab.sh my-research
set -euo pipefail

LAB="${1:-my-research}"
if [ -e "$LAB" ]; then
  echo "이미 '$LAB' 가 존재합니다. 다른 이름을 쓰거나 지워 주세요." >&2
  exit 1
fi

mkdir -p "$LAB"/{papers,notes,experiments/_template,.claude/agents,.claude/commands}
cd "$LAB"
git init -q

# ---------------------------------------------------------------- CLAUDE.md
cat > CLAUDE.md <<'EOF'
# 연구실 규칙

## 연구 주제
(여기에 한 줄로 적기: 예) 소규모 LLM의 장문 추론 성능 향상)

## 작업 방식
- 에이전트끼리의 의존은 "파일"로 한다. 각자 정해진 입력 파일을 읽고 정해진 출력 파일을 쓴다.
- 실험 전에 notes/hypotheses.md 에 가설과 성공 기준을 먼저 적는다.
- 실험은 experiments/NNN_이름/ 에 둔다 (config.yaml, run.py, results.json, README.md).
- 논문/노트에 쓰는 모든 수치는 results.json 에서 가져온다. 기억이나 추정으로 쓰지 않는다.
- 실험이 끝나면 notes/log.md 에 날짜, 변경점, 결과, 해석을 추가한다.
- 확신 없는 주장은 "추정"이라고 표시한다. 없는 논문/수치를 지어내지 않는다.
- 결과 해석 후에는 반드시 critic 에이전트의 검증을 받는다.

## 사람 승인이 필요한 지점
- 가설 확정
- GPU/비용이 큰 실험 실행 전
- 논문에 들어갈 주장 확정

## 환경
- (GPU, 파이썬 버전, 패키지 관리 방식 등을 적기)
- 랜덤 시드는 항상 config.yaml 에 명시한다.

## 예산 / 종료 조건
- 한 실험의 재시도는 최대 3회. 3회 실패하면 멈추고 원인을 보고한다.
EOF

# ---------------------------------------------------------------- agents
cat > .claude/agents/literature.md <<'EOF'
---
name: literature
description: 논문을 읽고 요약하거나 관련 연구를 정리할 때 사용. papers/ 폴더와 웹 자료를 다룬다.
model: sonnet
tools: Read, Write, Grep, Glob, WebSearch, WebFetch
---
너는 문헌조사 담당 연구원이다.
- papers/ 의 논문과 웹 자료를 읽고 notes/lit_review.md 에 정리한다.
- 각 논문마다: 문제 설정, 핵심 방법, 데이터셋/지표, 주요 결과, 한계를 적는다.
- 마지막에 "관련 연구 비교표"와 "아직 비어 있는 연구 공백"을 쓴다.
- 읽지 않은 내용을 추측해서 쓰지 않는다. 출처(제목, 연도, 링크)를 반드시 남긴다.
EOF

cat > .claude/agents/experimenter.md <<'EOF'
---
name: experimenter
description: 가설을 실험 코드와 config 로 구현하고 실행할 때 사용.
model: sonnet
tools: Read, Write, Edit, Bash, Grep, Glob
---
너는 실험 엔지니어다.
- notes/hypotheses.md 의 선택된 가설을 읽고 experiments/NNN_이름/ 에 구현한다.
- config.yaml (시드 포함), run.py, README.md(무엇을 왜 하는지)를 만든다.
- 먼저 아주 작은 규모(스모크 테스트)로 돌려 오류를 잡은 뒤 본 실행을 한다.
- 결과는 results.json 에 구조화해서 저장한다. 로그는 같은 폴더에 둔다.
- 본 실행 전에 예상 소요 시간과 자원을 보고하고, CLAUDE.md 의 승인 규칙을 따른다.
- 코드를 바꾸면 git commit 으로 남긴다.
EOF

cat > .claude/agents/analyst.md <<'EOF'
---
name: analyst
description: 실험 결과를 분석하고 표/그래프/통계 검정을 만들 때 사용.
model: sonnet
tools: Read, Write, Edit, Bash, Grep, Glob
---
너는 데이터 분석가다.
- experiments/*/results.json 만을 근거로 분석한다.
- 표와 그래프는 코드로 생성하고 experiments/NNN_이름/analysis/ 에 저장한다.
- 반복 실행이 있으면 평균, 표준편차, 신뢰구간 또는 적절한 검정을 보고한다.
- 해석과 사실을 분리해서 쓴다. 결과가 약하면 약하다고 쓴다.
EOF

cat > .claude/agents/critic.md <<'EOF'
---
name: critic
description: 실험 설계, 결과, 주장을 비판적으로 검증할 때 사용. 결과 해석 후 반드시 호출.
model: opus
tools: Read, Grep, Glob
---
너는 까다로운 리뷰어다. 생성 과정은 모르고 산출물만 본다.
experiments/NNN_이름/ 의 config, 코드, results.json 과 notes/ 의 주장을 읽고 다음을 점검한다.
1. 주장이 데이터로 뒷받침되는가 (과장, 인과 비약)
2. 빠진 baseline / ablation
3. 통계적 유의성과 시드 수
4. 데이터 누수, 평가 오염 가능성
5. 재현 가능성 (시드, 버전, 설정 누락)
칭찬은 생략하고, 문제를 심각도 순으로 정리한다. 마지막에 "다음에 해야 할 실험"을 최대 3개 제안한다.
EOF

# ---------------------------------------------------------------- slash command
cat > .claude/commands/new-experiment.md <<'EOF'
---
description: 새 실험 폴더를 템플릿에서 만든다. 사용법 /new-experiment 이름
---
experiments/ 아래의 기존 폴더 번호를 확인해 다음 번호(NNN)를 정하고,
experiments/_template 을 복사해 experiments/NNN_$ARGUMENTS/ 를 만든다.
그다음 notes/hypotheses.md 에서 이 실험이 검증할 가설을 찾아 README.md 에 연결해 적고,
config.yaml 의 빈칸을 채울 질문을 나에게 한다.
EOF

cat > .claude/commands/review-experiment.md <<'EOF'
---
description: 실험 하나를 분석하고 critic 검증까지 돌린다. 사용법 /review-experiment 001_이름
---
experiments/$ARGUMENTS 에 대해 순서대로 진행한다.
1. analyst 에이전트로 결과를 분석한다.
2. critic 에이전트로 검증받는다.
3. 지적 사항과 분석 요약을 notes/log.md 에 추가한다.
4. 내가 결정해야 할 사항을 3줄 이내로 요약해 보고한다.
EOF

# ---------------------------------------------------------------- notes
cat > notes/hypotheses.md <<'EOF'
# 가설 목록

| ID | 가설 | 성공 기준 | 필요한 실험 | 상태 |
|----|------|-----------|-------------|------|
| H1 |      |           |             | 제안 |
EOF

cat > notes/log.md <<'EOF'
# 실험 로그

(날짜 / 실험 ID / 변경점 / 결과 / 해석 / 다음 단계)
EOF

# ---------------------------------------------------------------- experiment template
cat > experiments/_template/config.yaml <<'EOF'
name: ""
hypothesis: ""      # notes/hypotheses.md 의 ID
seed: 0
# 모델, 데이터, 하이퍼파라미터를 여기에
EOF

cat > experiments/_template/run.py <<'EOF'
"""실험 진입점. config.yaml 을 읽어 실행하고 results.json 을 쓴다."""
import json
import pathlib
import random

here = pathlib.Path(__file__).parent


def main():
    # TODO: config 로드 (예: yaml.safe_load), 시드 고정, 실험 실행
    random.seed(0)
    results = {"metrics": {}, "notes": ""}
    (here / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
EOF

cat > experiments/_template/README.md <<'EOF'
# 실험 설명
- 검증할 가설:
- 방법:
- 성공 기준:
- 실행 방법:
EOF

# ---------------------------------------------------------------- misc
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.venv/
*.ckpt
*.pt
*.safetensors
wandb/
.env
.DS_Store
EOF

cat > README.md <<'EOF'
# AI 연구실
이 폴더에서 `claude` 를 실행하세요. 규칙은 CLAUDE.md, 역할은 .claude/agents/, 명령은 .claude/commands/ 에 있습니다.
EOF

git add -A
git commit -q -m "Initialize AI research lab scaffold" || true

echo
echo "완료: $(pwd)"
echo "다음 순서: cd $LAB && claude"
