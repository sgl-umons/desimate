import json
from pathlib import Path


def read_input(text_file: Path) -> list[str]:
    with open(text_file, "r", errors="ignore") as f:
        if text_file.name.endswith("json"):
            texts = json.loads(f.read())
        elif text_file.name.endswith("jsonl"):
            texts = [json.loads(line) for line in f]
        else:
            texts = f.readlines()
    texts = [_t.strip() for _t in texts if _t]
    return texts
