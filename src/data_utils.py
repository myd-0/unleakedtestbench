import json
from pathlib import Path


def read_jsonl(path):
    with open(path, "r") as f:
        content = f.read().strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def write_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def add_lineno(code):
    lines = code.splitlines()
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
