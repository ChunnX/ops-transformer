"""File I/O utilities — no global config dependency."""

from __future__ import annotations

import json
import os
from typing import Union, Optional


ENCODING = "utf-8"


def safe_read_file(file_path: str, encoding: str = ENCODING) -> Optional[str]:
    """Read file content, return None on failure."""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        return None


def safe_write_file(
    file_path: str,
    content: Union[str, bytes],
    create_dir: bool = True,
    encoding: str = ENCODING,
) -> None:
    """Write content to file, optionally creating parent directories."""
    if create_dir:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    mode = "wb" if isinstance(content, bytes) else "w"
    kwargs = {"mode": mode}
    if "b" not in mode:
        kwargs["encoding"] = encoding

    with open(file_path, **kwargs) as f:
        f.write(content)


def save_json_file(data: dict, file_path: str, encoding: str = ENCODING) -> None:
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json_file(file_path: str, encoding: str = ENCODING) -> Optional[dict]:
    """Load JSON file, return None on failure."""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
