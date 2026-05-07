from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Union


def ensure_directory(path: Union[str, Path]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_output_path(base_dir: Union[str, Path], path_value: Union[str, Path]) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def write_json(path: Union[str, Path], payload: object) -> None:
    json_path = Path(path)
    ensure_directory(json_path.parent)
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, object]]) -> None:
    jsonl_path = Path(path)
    ensure_directory(jsonl_path.parent)
    with jsonl_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False))
            file_obj.write("\n")