from __future__ import annotations

import argparse
import json


def normalize_example(example: dict) -> dict:
    prompt = example.get("prompt", "").strip()
    response = example.get("response", "").strip()
    if not prompt or not response:
        return {}
    return {"prompt": prompt, "response": response}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara dataset para fine-tuning.")
    parser.add_argument("--input", required=True, help="Arquivo JSONL de entrada.")
    parser.add_argument("--output", required=True, help="Arquivo JSONL de saída.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as inp, open(args.output, "w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            example = json.loads(line)
            normalized = normalize_example(example)
            if not normalized:
                continue
            out.write(json.dumps(normalized, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
