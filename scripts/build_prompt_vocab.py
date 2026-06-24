from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dualtsr.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a character vocab from text/prompt/label manifest fields.")
    parser.add_argument("manifests", nargs="+", help="JSON, JSONL, CSV, or TSV manifest files.")
    parser.add_argument("--out", default="data/hr_prompt_json/vocab.txt")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "items", "samples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON manifest shape: {path}")
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        dialect = csv.excel_tab if suffix in {".tsv", ".txt"} else csv.excel
        return list(csv.DictReader(f, dialect=dialect))


def main() -> None:
    args = parse_args()
    texts: list[str] = []
    for item in args.manifests:
        for row in read_rows(Path(item)):
            text = row.get("text", row.get("prompt", row.get("label", "")))
            if text:
                texts.append(str(text))
    if not texts:
        raise RuntimeError("No text/prompt/label values found.")
    CharTokenizer.write_vocab(args.out, texts)
    print(f"wrote {args.out} from {len(texts)} texts")


if __name__ == "__main__":
    main()
