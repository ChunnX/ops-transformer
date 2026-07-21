#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Versioned, shape-checked configuration for live consistency suites."""

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .adapter import CaseAdapter
from .config import ConsistencyMode, ModeRegistry
from .reporting import build_input_summary, input_content_digest
from .schema import get_q_lengths
from .transform import generate_split_groups


SCHEMA_VERSION = 1
CONFIG_KIND = "smla_mqsmla_batch_consistency_suite"
UNMATCHED_POLICIES = ("error", "defaults")
CONFIG_KEYS = {
    "profile",
    "modes",
    "seed",
    "randomize",
    "reorder_order",
    "split_groups",
    "token_split_sizes",
    "mode_batch",
    "common_tokens",
    "derived_extra",
    "require_complete_partition",
}
SELECTOR_KEYS = {
    "operator",
    "case_name",
    "pt_basename",
    "input_content_fingerprint_sha256",
}
LEGACY_ENV_SUFFIXES = (
    "MODES",
    "PROFILE",
    "SEED",
    "RANDOMIZE",
    "SPLIT_GROUPS",
    "TOKEN_SPLIT_SIZES",
    "MODE3_BATCH",
    "COMMON_TOKENS",
    "DERIVED_EXTRA_TOKENS",
)


class SuiteConfigError(ValueError):
    """Raised when a suite config is ambiguous or incompatible with a PT case."""


@dataclass(frozen=True)
class ResolvedCaseConfig:
    """Effective, fully shape-checked settings for one concrete PT case."""

    config_id: str
    source: str
    modes: Tuple[ConsistencyMode, ...]
    seed: int
    randomize: bool
    reorder_order: Optional[Tuple[int, ...]]
    split_groups: Optional[Tuple[Tuple[int, ...], ...]]
    token_split_sizes: Optional[Tuple[int, ...]]
    mode_batch: int
    common_tokens: Optional[int]
    derived_extra: Optional[int]
    require_complete_partition: bool
    selector: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "source": self.source,
            "selector": self.selector,
            "modes": [mode.value for mode in self.modes],
            "seed": self.seed,
            "randomize": self.randomize,
            "reorder_order": list(self.reorder_order) if self.reorder_order else None,
            "split_groups": (
                [list(group) for group in self.split_groups]
                if self.split_groups is not None
                else None
            ),
            "token_split_sizes": (
                list(self.token_split_sizes)
                if self.token_split_sizes is not None
                else None
            ),
            "mode_batch": self.mode_batch,
            "common_tokens": self.common_tokens,
            "derived_extra": self.derived_extra,
            "require_complete_partition": self.require_complete_partition,
        }


def _case_name(input_data: Dict[str, Any]) -> str:
    adapter = CaseAdapter(input_data)
    value = input_data.get("case_name") or input_data.get("Testcase_Name")
    value = value or adapter.params.get("testcase_name") or adapter.params.get("Testcase_Name")
    return str(value or "case")


def case_identity(input_data: Dict[str, Any], pt_path: str) -> Dict[str, str]:
    """Return the exact selector fields available for a loaded baseline PT."""
    summary = build_input_summary(input_data, role="baseline")
    return {
        "operator": CaseAdapter(input_data).name,
        "case_name": _case_name(input_data),
        "pt_basename": os.path.basename(pt_path),
        "input_content_fingerprint_sha256": input_content_digest(summary),
    }


def _reject_unknown(mapping: Mapping[str, Any], allowed: set, scope: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SuiteConfigError(f"Unknown {scope} fields: {unknown}")


def _require_plain_int(value: Any, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SuiteConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise SuiteConfigError(f"{name} must be >= {minimum}")
    return value


def _integer_sequence(value: Any, name: str) -> List[int]:
    if isinstance(value, str):
        try:
            items = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as error:
            raise SuiteConfigError(f"{name} must contain integers") from error
    elif isinstance(value, (list, tuple)):
        items = [_require_plain_int(item, name) for item in value]
    else:
        raise SuiteConfigError(f"{name} must be an integer array")
    if not items:
        raise SuiteConfigError(f"{name} must not be empty")
    return items


def _mode_values(value: Any) -> str:
    if isinstance(value, str):
        tokens = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise SuiteConfigError("modes must contain canonical string names")
        tokens = [item.strip() for item in value if item.strip()]
    else:
        raise SuiteConfigError("modes must be a string array")
    if not tokens or any(token.isdigit() for token in tokens):
        raise SuiteConfigError("modes must use canonical names, not numeric aliases")
    return ",".join(tokens)


def _merge_layers(defaults: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for name, layer in (("defaults", defaults), ("case config", override)):
        _reject_unknown(layer, CONFIG_KEYS, name)
        if "modes" in layer and "profile" in layer:
            raise SuiteConfigError(f"{name} cannot contain both modes and profile")
    merged = dict(defaults)
    if "modes" in override:
        merged.pop("profile", None)
    if "profile" in override:
        merged.pop("modes", None)
    merged.update(override)
    return merged


def resolve_case_values(
    values: Mapping[str, Any],
    input_data: Dict[str, Any],
    *,
    config_id: str,
    source: str,
    selector: Optional[Dict[str, str]] = None,
) -> ResolvedCaseConfig:
    """Resolve defaults and validate every shape-dependent option before NPU use."""
    _reject_unknown(values, CONFIG_KEYS, "config")
    if "modes" in values and "profile" in values:
        raise SuiteConfigError("config cannot contain both modes and profile")
    profile = values.get("profile")
    if profile is not None and not isinstance(profile, str):
        raise SuiteConfigError("profile must be a string")
    try:
        modes = ModeRegistry.resolve(
            modes=_mode_values(values["modes"]) if "modes" in values else None,
            profile=profile,
            warn_legacy=False,
        )
    except ValueError as error:
        raise SuiteConfigError(str(error)) from error

    seed = _require_plain_int(values.get("seed", 20260716), "seed")
    randomize = values.get("randomize", False)
    if not isinstance(randomize, bool):
        raise SuiteConfigError("randomize must be a boolean")
    require_complete = values.get("require_complete_partition", True)
    if not isinstance(require_complete, bool):
        raise SuiteConfigError("require_complete_partition must be a boolean")

    adapter = CaseAdapter(input_data)
    batch_size = adapter.get_batch_size()
    q_lengths = get_q_lengths(input_data, valid_only=True)
    mode_batch = _require_plain_int(values.get("mode_batch", 0), "mode_batch", 0)
    if mode_batch >= batch_size:
        raise SuiteConfigError(f"mode_batch {mode_batch} is outside [0, {batch_size})")

    reorder_order = None
    if "reorder_order" in values:
        reorder_order = tuple(_integer_sequence(values["reorder_order"], "reorder_order"))
        if sorted(reorder_order) != list(range(batch_size)):
            raise SuiteConfigError(
                f"reorder_order must be a permutation of [0, {batch_size})"
            )
        if batch_size >= 2 and reorder_order == tuple(range(batch_size)):
            raise SuiteConfigError("reorder_order must not be the identity permutation")
    if ConsistencyMode.REORDER not in modes and reorder_order is not None:
        raise SuiteConfigError("reorder_order is set but reorder mode is not selected")

    split_groups = None
    if "split_groups" in values:
        raw_groups = values["split_groups"]
        if isinstance(raw_groups, str):
            raw_groups = [group for group in raw_groups.split(";") if group.strip()]
        if not isinstance(raw_groups, (list, tuple)) or len(raw_groups) < 2:
            raise SuiteConfigError("split_groups must contain at least two groups")
        split_groups = tuple(
            tuple(_integer_sequence(group, "split_groups")) for group in raw_groups
        )
    elif ConsistencyMode.SPLIT in modes and batch_size >= 2:
        split_groups = tuple(
            tuple(group)
            for group in generate_split_groups(
                batch_size, seed=seed, randomize=randomize
            )
        )
    if ConsistencyMode.SPLIT in modes:
        flattened = [item for group in split_groups or () for item in group]
        if split_groups is not None and len(flattened) != len(set(flattened)):
            raise SuiteConfigError("split_groups contain duplicate batch ids")
        if split_groups is not None and any(
            item < 0 or item >= batch_size for item in flattened
        ):
            raise SuiteConfigError(f"split_groups contain a batch id outside [0, {batch_size})")
        if (
            split_groups is not None
            and require_complete
            and sorted(flattened) != list(range(batch_size))
        ):
            raise SuiteConfigError(
                "split_groups must form a complete partition when "
                "require_complete_partition=true"
            )
    elif split_groups is not None:
        raise SuiteConfigError("split_groups is set but split mode is not selected")
    if randomize and "split_groups" in values:
        raise SuiteConfigError("randomize cannot be combined with explicit split_groups")
    if randomize and ConsistencyMode.SPLIT not in modes:
        raise SuiteConfigError("randomize is set but split mode is not selected")

    token_sizes = None
    if "token_split_sizes" in values:
        token_sizes = tuple(
            _integer_sequence(values["token_split_sizes"], "token_split_sizes")
        )
    elif ConsistencyMode.TOKEN_SPLIT in modes and q_lengths[mode_batch] >= 2:
        token_count = q_lengths[mode_batch]
        first = 4 * 1024 if token_count == 16 * 1024 else token_count // 2
        token_sizes = (first, token_count - first)
    if ConsistencyMode.TOKEN_SPLIT in modes:
        if token_sizes is not None and (
            len(token_sizes) < 2 or any(item <= 0 for item in token_sizes)
        ):
            raise SuiteConfigError("token_split_sizes must contain at least two positive sizes")
        if token_sizes is not None and sum(token_sizes) != q_lengths[mode_batch]:
            raise SuiteConfigError(
                f"token_split_sizes sum to {sum(token_sizes)}, expected {q_lengths[mode_batch]}"
            )
    elif token_sizes is not None:
        raise SuiteConfigError(
            "token_split_sizes is set but token-split mode is not selected"
        )

    common_tokens = values.get("common_tokens")
    derived_extra = values.get("derived_extra")
    if common_tokens is not None:
        common_tokens = _require_plain_int(common_tokens, "common_tokens", 1)
    if derived_extra is not None:
        derived_extra = _require_plain_int(derived_extra, "derived_extra", 0)
    if ConsistencyMode.SHAPE_CHANGE in modes:
        token_count = q_lengths[mode_batch]
        if token_count > 0:
            common_tokens = (
                common_tokens
                if common_tokens is not None
                else (300 if token_count >= 400 else max(1, token_count // 2))
            )
        if common_tokens is not None and common_tokens > token_count:
            raise SuiteConfigError(
                f"common_tokens {common_tokens} exceeds Q length {token_count}"
            )
        if common_tokens is not None:
            derived_extra = (
                derived_extra
                if derived_extra is not None
                else (
                    200
                    if token_count == 400 and common_tokens == 300
                    else token_count - common_tokens + 1
                )
            )
            if common_tokens + derived_extra == token_count:
                raise SuiteConfigError(
                    "shape-change target Q length must differ from the baseline Q length"
                )
    elif common_tokens is not None or derived_extra is not None:
        raise SuiteConfigError(
            "common_tokens/derived_extra are set but shape-change mode is not selected"
        )

    return ResolvedCaseConfig(
        config_id=config_id,
        source=source,
        modes=modes,
        seed=seed,
        randomize=randomize,
        reorder_order=reorder_order,
        split_groups=split_groups,
        token_split_sizes=token_sizes,
        mode_batch=mode_batch,
        common_tokens=common_tokens,
        derived_extra=derived_extra,
        require_complete_partition=require_complete,
        selector=selector,
    )


def resolve_legacy_environment(
    input_data: Dict[str, Any], pt_path: str, prefix: str
) -> ResolvedCaseConfig:
    """Normalize the deprecated process-wide CLI/environment interface per case."""
    modes_value = os.environ.get(f"{prefix}_MODES", "").strip()
    profile_value = os.environ.get(f"{prefix}_PROFILE", "").strip()
    try:
        modes = ModeRegistry.resolve(
            modes=modes_value or None,
            profile=profile_value or None,
            warn_legacy=True,
        )
    except ValueError as error:
        raise SuiteConfigError(str(error)) from error

    values: Dict[str, Any] = {"modes": [mode.value for mode in modes]}

    def optional_int(suffix: str, target: str) -> None:
        raw = os.environ.get(f"{prefix}_{suffix}", "").strip()
        if not raw:
            return
        try:
            values[target] = int(raw)
        except ValueError as error:
            raise SuiteConfigError(f"{target} must be an integer") from error

    seed = os.environ.get(f"{prefix}_SEED", "20260716").strip()
    try:
        values["seed"] = int(seed)
    except ValueError as error:
        raise SuiteConfigError("seed must be an integer") from error
    randomize = os.environ.get(f"{prefix}_RANDOMIZE", "0").strip()
    if randomize not in ("0", "1"):
        raise SuiteConfigError("randomize environment value must be 0 or 1")
    values["randomize"] = randomize == "1"

    for suffix, target in (
        ("SPLIT_GROUPS", "split_groups"),
        ("TOKEN_SPLIT_SIZES", "token_split_sizes"),
    ):
        raw = os.environ.get(f"{prefix}_{suffix}", "").strip()
        if raw:
            values[target] = raw
    optional_int("MODE3_BATCH", "mode_batch")
    optional_int("COMMON_TOKENS", "common_tokens")
    optional_int("DERIVED_EXTRA_TOKENS", "derived_extra")
    return resolve_case_values(
        values,
        input_data,
        config_id="legacy-cli",
        source="deprecated process-wide CLI/environment",
    )


def resolve_runtime_case_config(
    input_data: Dict[str, Any], pt_path: str, prefix: str
) -> ResolvedCaseConfig:
    """Resolve either a versioned suite file or the mutually exclusive legacy interface."""
    config_path = os.environ.get(f"{prefix}_CONFIG_FILE", "").strip()
    legacy_set = [
        f"{prefix}_{suffix}"
        for suffix in LEGACY_ENV_SUFFIXES
        if f"{prefix}_{suffix}" in os.environ
    ]
    if config_path and legacy_set:
        raise SuiteConfigError(
            "config file cannot be combined with legacy mode/detail environment values: "
            f"{legacy_set}"
        )
    if config_path:
        return SuiteConfig.load(config_path).resolve(input_data, pt_path)
    return resolve_legacy_environment(input_data, pt_path, prefix)


class SuiteConfig:
    """Load one JSON suite and resolve exactly one matching entry per PT case."""

    def __init__(self, path: str, document: Dict[str, Any]):
        self.path = os.path.abspath(path)
        _reject_unknown(
            document,
            {"schema_version", "kind", "unmatched_case_policy", "defaults", "cases"},
            "suite",
        )
        if document.get("schema_version") != SCHEMA_VERSION:
            raise SuiteConfigError(
                f"Unsupported suite schema_version={document.get('schema_version')}"
            )
        if document.get("kind") != CONFIG_KIND:
            raise SuiteConfigError(f"Unexpected suite kind={document.get('kind')!r}")
        self.unmatched_policy = document.get("unmatched_case_policy", "error")
        if self.unmatched_policy not in UNMATCHED_POLICIES:
            raise SuiteConfigError(
                f"unmatched_case_policy must be one of {UNMATCHED_POLICIES}"
            )
        self.defaults = document.get("defaults", {})
        if not isinstance(self.defaults, dict):
            raise SuiteConfigError("defaults must be an object")
        _merge_layers(self.defaults, {})
        self.cases = document.get("cases", [])
        if not isinstance(self.cases, list):
            raise SuiteConfigError("cases must be an array")
        seen_ids = set()
        for index, entry in enumerate(self.cases):
            if not isinstance(entry, dict):
                raise SuiteConfigError(f"cases[{index}] must be an object")
            _reject_unknown(entry, {"id", "match", "config"}, f"cases[{index}]")
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or not entry_id.strip() or entry_id in seen_ids:
                raise SuiteConfigError(f"cases[{index}].id must be unique and non-empty")
            seen_ids.add(entry_id)
            selector = entry.get("match")
            if not isinstance(selector, dict) or not selector:
                raise SuiteConfigError(f"cases[{index}].match must be a non-empty object")
            _reject_unknown(selector, SELECTOR_KEYS, f"cases[{index}].match")
            if not all(isinstance(value, str) and value for value in selector.values()):
                raise SuiteConfigError(f"cases[{index}].match values must be non-empty strings")
            config = entry.get("config", {})
            if not isinstance(config, dict):
                raise SuiteConfigError(f"cases[{index}].config must be an object")
            _merge_layers(self.defaults, config)

    @classmethod
    def load(cls, path: str) -> "SuiteConfig":
        try:
            with open(path, "r", encoding="utf-8") as file:
                document = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise SuiteConfigError(f"Cannot load suite config {path}: {error}") from error
        if not isinstance(document, dict):
            raise SuiteConfigError("Suite config root must be an object")
        return cls(path, document)

    def resolve(self, input_data: Dict[str, Any], pt_path: str) -> ResolvedCaseConfig:
        identity = case_identity(input_data, pt_path)
        matches = [
            entry
            for entry in self.cases
            if all(identity[key] == value for key, value in entry["match"].items())
        ]
        if len(matches) > 1:
            raise SuiteConfigError(
                f"PT {identity['pt_basename']} matches multiple configs: "
                f"{[entry['id'] for entry in matches]}"
            )
        if not matches and self.unmatched_policy == "error":
            raise SuiteConfigError(
                f"PT {identity['pt_basename']} ({identity['case_name']}) has no suite config match"
            )
        if matches:
            entry = matches[0]
            values = _merge_layers(self.defaults, entry.get("config", {}))
            config_id = entry["id"]
            selector = dict(entry["match"])
        else:
            values = dict(self.defaults)
            config_id = "defaults"
            selector = None
        return resolve_case_values(
            values,
            input_data,
            config_id=config_id,
            source=self.path,
            selector=selector,
        )
