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
