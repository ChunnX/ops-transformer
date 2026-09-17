# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Load and validate manifest-driven accuracy and performance cases."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


SCHEMA_VERSION = 1
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SUPPORTED_RUN_MODES = {"accuracy", "performance"}
SUPPORTED_DISTRIBUTIONS = {"uniform", "normal"}
SUPPORTED_ROUTES = {"balanced", "unbalanced"}
SUPPORTED_OPERATORS = {
    "all_to_allv_grouped_mat_mul",
    "grouped_mat_mul_all_to_allv",
}


class CaseManifestError(ValueError):
    """Raised when a validation manifest is malformed."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require_positive_int(case_id: str, name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CaseManifestError(f"case {case_id}: {name} must be a positive integer")
    return value


def _require_nonnegative_int(case_id: str, name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CaseManifestError(
            f"case {case_id}: {name} must be a non-negative integer"
        )
    return value


def _validate_distribution(case_id: str, distribution: Any) -> Dict[str, Any]:
    if isinstance(distribution, str):
        distribution = {"kind": distribution}
    if not isinstance(distribution, Mapping):
        raise CaseManifestError(f"case {case_id}: input_distribution must be a mapping or string")
    result = dict(distribution)
    kind = result.get("kind", "uniform")
    if kind not in SUPPORTED_DISTRIBUTIONS:
        raise CaseManifestError(
            f"case {case_id}: input_distribution.kind must be one of "
            f"{sorted(SUPPORTED_DISTRIBUTIONS)}"
        )
    result["kind"] = kind
    if kind == "uniform":
        low = float(result.get("low", -5.0))
        high = float(result.get("high", 5.0))
        if not low < high:
            raise CaseManifestError(f"case {case_id}: uniform low must be smaller than high")
        result.update(low=low, high=high)
    else:
        mean = float(result.get("mean", 0.0))
        std = float(result.get("std", 1.0))
        if std <= 0:
            raise CaseManifestError(f"case {case_id}: normal std must be positive")
        result.update(mean=mean, std=std)
    return result


def _validate_case(raw_case: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_case, Mapping):
        raise CaseManifestError("each case must be a mapping")
    case = copy.deepcopy(dict(raw_case))
    case_id = case.get("id")
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
        raise CaseManifestError(
            "case id must be a non-empty string containing only letters, digits, '.', '_' or '-'"
        )

    shape = case.get("shape")
    if not isinstance(shape, Mapping):
        raise CaseManifestError(f"case {case_id}: shape must be a mapping with m, n and k")
    shape = {
        axis: _require_positive_int(case_id, f"shape.{axis}", shape.get(axis))
        for axis in ("m", "n", "k")
    }
    case["shape"] = shape

    operator = case.get("operator", "all_to_allv_grouped_mat_mul")
    if operator not in SUPPORTED_OPERATORS:
        raise CaseManifestError(
            f"case {case_id}: operator must be one of {sorted(SUPPORTED_OPERATORS)}"
        )
    case["operator"] = operator
    output_all_to_allv = case.get(
        "output_all_to_allv", operator == "all_to_allv_grouped_mat_mul"
    )
    if not isinstance(output_all_to_allv, bool):
        raise CaseManifestError(
            f"case {case_id}: output_all_to_allv must be a boolean"
        )
    if operator != "all_to_allv_grouped_mat_mul" and output_all_to_allv:
        raise CaseManifestError(
            f"case {case_id}: output_all_to_allv is supported only for "
            "all_to_allv_grouped_mat_mul"
        )
    case["output_all_to_allv"] = output_all_to_allv

    world_size = _require_positive_int(case_id, "world_size", case.get("world_size"))
    if world_size > 8:
        raise CaseManifestError(f"case {case_id}: world_size must not exceed 8")
    case["world_size"] = world_size

    expert_num = _require_positive_int(case_id, "expert_num", case.get("expert_num"))
    if expert_num % world_size:
        raise CaseManifestError(f"case {case_id}: expert_num must be divisible by world_size")
    local_expert_limit = 32 if operator == "grouped_mat_mul_all_to_allv" else 64
    if expert_num // world_size > local_expert_limit:
        raise CaseManifestError(
            f"case {case_id}: local expert count must not exceed {local_expert_limit} "
            f"for {operator}"
        )
    case["expert_num"] = expert_num

    if case.get("dtype", "bfloat16") != "bfloat16":
        raise CaseManifestError(f"case {case_id}: the current operator supports only bfloat16")
    case["dtype"] = "bfloat16"
    route = case.get("route", "balanced")
    if route not in SUPPORTED_ROUTES:
        raise CaseManifestError(
            f"case {case_id}: route must be one of {sorted(SUPPORTED_ROUTES)}"
        )
    case["route"] = route

    run = case.get("run", ["accuracy", "performance"])
    if not isinstance(run, Sequence) or isinstance(run, (str, bytes)) or not run:
        raise CaseManifestError(f"case {case_id}: run must be a non-empty list")
    unknown_modes = set(run) - SUPPORTED_RUN_MODES
    if unknown_modes:
        raise CaseManifestError(f"case {case_id}: unsupported run modes {sorted(unknown_modes)}")
    case["run"] = list(dict.fromkeys(run))

    tags = case.get("tags", [])
    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
        raise CaseManifestError(f"case {case_id}: tags must be a list")
    case["tags"] = [str(tag) for tag in tags]
    case["seed"] = int(case.get("seed", 2026))
    case["input_distribution"] = _validate_distribution(
        case_id, case.get("input_distribution", "uniform")
    )

    accuracy = case.get("accuracy", {})
    if not isinstance(accuracy, Mapping):
        raise CaseManifestError(f"case {case_id}: accuracy must be a mapping")
    accuracy = dict(accuracy)
    if accuracy.get("standard", "opbase_experimental") != "opbase_experimental":
        raise CaseManifestError(
            f"case {case_id}: accuracy.standard must be opbase_experimental"
        )
    forbidden = {"rtol", "atol", "required_matched_ratio", "max_abs_error_limit"} & set(accuracy)
    if forbidden:
        raise CaseManifestError(
            f"case {case_id}: official opbase thresholds cannot be overridden: {sorted(forbidden)}"
        )
    accuracy["standard"] = "opbase_experimental"
    accuracy["repeats"] = _require_positive_int(
        case_id, "accuracy.repeats", accuracy.get("repeats", 2)
    )
    # The ordinary launch plus one automatically allocated repeat gives two
    # independent operator invocations by default. Reusable caller-provided
    # storage is an opt-in validation dimension selected by the launcher.
    accuracy["auto_repeats"] = _require_nonnegative_int(
        case_id, "accuracy.auto_repeats", accuracy.get("auto_repeats", 1)
    )
    accuracy["topk"] = _require_positive_int(case_id, "accuracy.topk", accuracy.get("topk", 20))
    accuracy["heatmap_bins"] = _require_positive_int(
        case_id, "accuracy.heatmap_bins", accuracy.get("heatmap_bins", 32)
    )
    accuracy["dump_on_failure"] = bool(accuracy.get("dump_on_failure", True))
    case["accuracy"] = accuracy

    performance = case.get("performance", {})
    if not isinstance(performance, Mapping):
        raise CaseManifestError(f"case {case_id}: performance must be a mapping")
    performance = dict(performance)
    for name, default in (
        ("warmup", 10),
        ("align_iterations", 3),
        ("iterations", 50),
        ("repeats", 3),
    ):
        performance[name] = _require_positive_int(
            case_id, f"performance.{name}", performance.get(name, default)
        )
    performance["max_rank_skew_ratio"] = float(
        performance.get("max_rank_skew_ratio", 0.10)
    )
    performance["max_regression_ratio"] = float(
        performance.get("max_regression_ratio", 0.0)
    )
    if performance["max_rank_skew_ratio"] < 0:
        raise CaseManifestError(
            f"case {case_id}: performance.max_rank_skew_ratio must be non-negative"
        )
    if performance["max_regression_ratio"] < 0:
        raise CaseManifestError(
            f"case {case_id}: performance.max_regression_ratio must be non-negative"
        )
    case["performance"] = performance
    return case


def load_case_manifest(path: Union[Path, str]) -> List[Dict[str, Any]]:
    """Load YAML, merge defaults into cases and return fully resolved cases."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required for validation; install ops_transformer_catccos_udma[validation]"
        ) from error

    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, Mapping):
        raise CaseManifestError("case manifest root must be a mapping")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CaseManifestError(
            f"unsupported schema_version={manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    defaults = manifest.get("defaults", {})
    raw_cases = manifest.get("cases")
    if not isinstance(defaults, Mapping):
        raise CaseManifestError("defaults must be a mapping")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)) or not raw_cases:
        raise CaseManifestError("cases must be a non-empty list")

    if any(not isinstance(raw_case, Mapping) for raw_case in raw_cases):
        raise CaseManifestError("each case must be a mapping")
    cases = [_validate_case(_deep_merge(defaults, raw_case)) for raw_case in raw_cases]
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        raise CaseManifestError(f"duplicate case ids: {duplicates}")
    return cases


def select_cases(
    cases: Iterable[Mapping[str, Any]],
    *,
    mode: Optional[str],
    case_ids: Optional[Sequence[str]] = None,
    include_tags: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Select cases by IDs/tags, optionally restricting their declared run mode.

    Passing mode=None lets stress reuse existing workloads without changing
    their accuracy/performance scheduling metadata.
    """
    if mode is not None and mode not in SUPPORTED_RUN_MODES:
        raise ValueError(f"unsupported mode {mode}")
    requested_ids = set(case_ids or [])
    requested_tags = set(include_tags or [])
    selected = []
    known_ids = set()
    for raw_case in cases:
        case = copy.deepcopy(dict(raw_case))
        known_ids.add(case["id"])
        if mode is not None and mode not in case["run"]:
            continue
        if requested_ids and case["id"] not in requested_ids:
            continue
        if requested_tags and not requested_tags.issubset(set(case["tags"])):
            continue
        selected.append(case)
    missing = requested_ids - known_ids
    if missing:
        raise CaseManifestError(f"unknown case ids: {sorted(missing)}")
    if not selected:
        raise CaseManifestError("case selection is empty")
    return selected


def case_digest(case: Mapping[str, Any]) -> str:
    payload = json.dumps(case, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def performance_workload_digest(case: Mapping[str, Any]) -> str:
    """Hash only fields that affect the timed workload and occurrence layout."""
    payload = {
        "dtype": case["dtype"],
        "expert_num": case["expert_num"],
        "operator": case["operator"],
        "output_all_to_allv": case["output_all_to_allv"],
        "input_distribution": case["input_distribution"],
        "route": case["route"],
        "seed": case["seed"],
        "shape": case["shape"],
        "world_size": case["world_size"],
        "performance": {
            key: case["performance"][key]
            for key in ("warmup", "align_iterations", "iterations", "repeats")
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_resolved_cases(path: Union[Path, str], cases: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump({"schema_version": SCHEMA_VERSION, "cases": list(cases)}, stream, indent=2)
        stream.write("\n")


def read_resolved_cases(path: Union[Path, str]) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(payload.get("cases"), list):
        raise CaseManifestError("invalid resolved case file")
    return [_validate_case(case) for case in payload["cases"]]
