"""Shared case loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import yaml


CASE_FILES = ("cases_smoke.yaml", "cases_typical.yaml", "cases_full.yaml")
DEFAULTS = {
    "n2_index": 1,
    "d": 128,
    "d_index": 128,
    "dr": 64,
    "sparse_mode": 3,
    "pre_tokens": 9223372036854775807,
    "next_tokens": 9223372036854775807,
}


def load_case_file(path: Path) -> List[Dict[str, object]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise ValueError(f"{path}: expected a top-level cases list")
    cases = []
    for raw in document["cases"]:
        case = dict(DEFAULTS)
        case.update(raw)
        case["n2"] = int(case["n1"])
        cases.append(case)
    return cases


def load_all_cases(root: Path) -> List[Dict[str, object]]:
    result = []
    for name in CASE_FILES:
        result.extend(load_case_file(root / name))
    return result


def select_cases(
    root: Path,
    suites: Sequence[str],
    selectors: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    cases = []
    for suite in suites:
        cases.extend(load_case_file(root / f"cases_{suite}.yaml"))
    if not selectors:
        return cases
    wanted = set(selectors)
    selected = [case for case in cases if str(case["id"]) in wanted]
    missing = wanted - {str(case["id"]) for case in selected}
    if missing:
        raise ValueError(f"unknown case IDs: {sorted(missing)}")
    return selected


def validate_case(case: Mapping[str, object]) -> None:
    case_id = str(case["id"])
    if case["layout"] not in ("BSND", "TND"):
        raise ValueError(f"{case_id}: unsupported layout")
    if case["q_dtype"] not in ("float16", "bfloat16"):
        raise ValueError(f"{case_id}: unsupported q_dtype")
    if case["weight_dtype"] not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"{case_id}: unsupported weight_dtype")
    if case["weight_dtype"] != "float32" and case["weight_dtype"] != case["q_dtype"]:
        raise ValueError(f"{case_id}: non-FP32 weight must match q_dtype")
    if int(case["batch"]) not in range(1, 17):
        raise ValueError(f"{case_id}: batch outside 1..16")
    if int(case["s1"]) not in (1024, 2048, 4096):
        raise ValueError(f"{case_id}: unsupported S1")
    if int(case["s2"]) not in (4096, 8192, 16384, 32768, 65536):
        raise ValueError(f"{case_id}: unsupported S2")
    if int(case["s1"]) > int(case["s2"]):
        raise ValueError(f"{case_id}: S1 must be <= S2")
    if int(case["n1"]) not in (32, 64, 128):
        raise ValueError(f"{case_id}: unsupported N1")
    if int(case["n1_index"]) not in (8, 16, 32, 64):
        raise ValueError(f"{case_id}: unsupported N1Index")
    if int(case["n2"]) != int(case["n1"]) or int(case["n2_index"]) != 1:
        raise ValueError(f"{case_id}: invalid N2/N2Index")
    if int(case["d"]) != 128 or int(case["d_index"]) != 128:
        raise ValueError(f"{case_id}: D and DIndex must be 128")
    if bool(case["rope"]) and int(case["dr"]) != 64:
        raise ValueError(f"{case_id}: Dr must be 64")
    if int(case["sparse_mode"]) != 3:
        raise ValueError(f"{case_id}: sparse_mode must be 3")


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
