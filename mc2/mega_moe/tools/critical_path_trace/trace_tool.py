#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate, audit, and render reviewable operator critical-path traces.

The tool intentionally uses only the Python standard library so it can run in
operator build/test environments without adding a package dependency.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable
import zlib


SPEC_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
POINT_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_S$")
TRACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
COMPILE_SWITCH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REQUIRED_HEADER_FIELDS = {
    "magic",
    "version",
    "schema_id",
    "header_words",
    "record_words",
    "record_count",
    "role_counts",
    "clock_bits",
    "repetition_count",
    "capture_capacity",
    "truncated",
    "flags",
    "rank",
    "runtime_config_hash",
}


class ContractError(RuntimeError):
    """Raised when a spec, source audit, or normalized trace is unsafe."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _spec_sha256(spec: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(spec))


def _expected_schema_id(spec: dict[str, Any]) -> int:
    """Derive a stable uint32 id from wire and timeline semantics, not source lines."""
    trace = spec.get("trace", {})
    protocol = trace.get("protocol", {})
    payload = {
        "trace_name": trace.get("name"),
        "roles": trace.get("roles"),
        "clock": {
            key: trace.get("clock", {}).get(key)
            for key in (
                "unit",
                "bits",
                "timestamp_pipeline",
                "scope",
                "comparable_across_records",
                "missing_value",
            )
        },
        "protocol": {
            key: protocol.get(key)
            for key in (
                "magic",
                "version",
                "word_bits",
                "endianness",
                "header_words",
                "record_words",
                "record_order",
                "record_order_description",
                "header_fields",
                "record_fields",
            )
        },
        "points": [
            {
                key: point.get(key)
                for key in (
                    "id",
                    "stage",
                    "edge",
                    "boundary_quality",
                    "roles",
                    "scope",
                )
            }
            | {
                "producer_pipeline": point.get("dependency", {}).get(
                    "producer_pipeline"
                ),
                "timestamp_pipeline": point.get("dependency", {}).get(
                    "timestamp_pipeline"
                ),
                "bridge": point.get("dependency", {}).get("bridge"),
            }
            for point in spec.get("points", [])
        ],
        "spans": [
            {key: span.get(key) for key in ("id", "begin", "end", "roles", "quality")}
            for span in spec.get("spans", [])
        ],
    }
    return zlib.crc32(_canonical_json_bytes(payload)) & 0xFFFFFFFF


def _contains_todo(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and "TODO" in value.upper():
        found.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_contains_todo(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_contains_todo(child, f"{path}[{index}]"))
    return found


def _parse_uint(value: Any, field: str, bits: int = 32) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be an unsigned integer, not boolean")
    try:
        parsed = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field} must be an unsigned integer: {value!r}") from exc
    if parsed < 0 or parsed >= (1 << bits):
        raise ContractError(f"{field}={parsed} does not fit uint{bits}")
    return parsed


def _required_string(
    obj: Any, key: str, location: str, errors: list[str]
) -> str | None:
    if not isinstance(obj, dict):
        errors.append(f"{location} must be an object")
        return None
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{location}.{key} must be a non-empty string")
        return None
    return value


def _validate_spec(spec: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["spec root must be an object"]
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        errors.append(
            f"$.schema_version must be {SPEC_SCHEMA_VERSION}, got {spec.get('schema_version')!r}"
        )

    todo_paths = _contains_todo(spec)
    if todo_paths:
        preview = ", ".join(todo_paths[:12])
        suffix = " ..." if len(todo_paths) > 12 else ""
        errors.append(f"unresolved TODO placeholders: {preview}{suffix}")

    trace = spec.get("trace")
    if not isinstance(trace, dict):
        errors.append("$.trace must be an object")
        return errors

    trace_name = _required_string(trace, "name", "$.trace", errors)
    if trace_name and not TRACE_NAME_RE.fullmatch(trace_name):
        errors.append("$.trace.name must contain only letters, digits, '.', '_' or '-'")
    compile_switch = _required_string(trace, "compile_switch", "$.trace", errors)
    if compile_switch and not COMPILE_SWITCH_RE.fullmatch(compile_switch):
        errors.append("$.trace.compile_switch must be a C/C++ identifier")

    roles = trace.get("roles")
    if (
        not isinstance(roles, list)
        or not roles
        or any(not isinstance(role, str) or not role.strip() for role in roles)
    ):
        errors.append("$.trace.roles must be a non-empty list of strings")
        roles = []
    elif len(set(roles)) != len(roles):
        errors.append("$.trace.roles must be unique")
    role_set = set(roles)

    clock = trace.get("clock")
    if not isinstance(clock, dict):
        errors.append("$.trace.clock must be an object")
        clock = {}
    for key in ("api", "unit", "timestamp_pipeline", "scope"):
        _required_string(clock, key, "$.trace.clock", errors)
    bits = clock.get("bits")
    if bits not in (32, 64):
        errors.append("$.trace.clock.bits must be 32 or 64")
    if not isinstance(clock.get("comparable_across_records"), bool):
        errors.append("$.trace.clock.comparable_across_records must be boolean")
    missing = clock.get("missing_value")
    if missing is not None and isinstance(bits, int):
        try:
            _parse_uint(missing, "$.trace.clock.missing_value", bits)
        except ContractError as exc:
            errors.append(str(exc))

    protocol = trace.get("protocol")
    if not isinstance(protocol, dict):
        errors.append("$.trace.protocol must be an object")
        protocol = {}
    for key in ("magic", "schema_id", "header_contract"):
        _required_string(protocol, key, "$.trace.protocol", errors)
    for key in ("magic", "schema_id"):
        if key in protocol:
            try:
                _parse_uint(protocol[key], f"$.trace.protocol.{key}")
            except ContractError as exc:
                errors.append(str(exc))
    version = protocol.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        errors.append("$.trace.protocol.version must be a positive integer")
    word_bits = protocol.get("word_bits")
    if word_bits != 32:
        errors.append("$.trace.protocol.word_bits must be 32")
    if protocol.get("endianness") not in ("little", "big"):
        errors.append("$.trace.protocol.endianness must be little or big")
    header_words = protocol.get("header_words")
    if (
        not isinstance(header_words, int)
        or isinstance(header_words, bool)
        or header_words <= 0
    ):
        errors.append("$.trace.protocol.header_words must be a positive integer")
    record_words = protocol.get("record_words")
    if (
        not isinstance(record_words, int)
        or isinstance(record_words, bool)
        or record_words <= 0
    ):
        errors.append("$.trace.protocol.record_words must be a positive integer")
    record_order = _required_string(
        protocol, "record_order", "$.trace.protocol", errors
    )
    if record_order and record_order not in ("role-major", "custom"):
        errors.append("$.trace.protocol.record_order must be role-major or custom")
    _required_string(protocol, "record_order_description", "$.trace.protocol", errors)

    header_fields = protocol.get("header_fields")
    if not isinstance(header_fields, list) or not header_fields:
        errors.append("$.trace.protocol.header_fields must be a non-empty list")
        header_fields = []
    header_names: set[str] = set()
    occupied_header_words: dict[int, str] = {}
    for index, field in enumerate(header_fields):
        loc = f"$.trace.protocol.header_fields[{index}]"
        if not isinstance(field, dict):
            errors.append(f"{loc} must be an object")
            continue
        name = _required_string(field, "name", loc, errors)
        if name:
            if not FIELD_NAME_RE.fullmatch(name):
                errors.append(f"{loc}.name must be a C-like identifier")
            if name in header_names:
                errors.append(f"duplicate header field name: {name}")
            header_names.add(name)
        word = field.get("word")
        width = field.get("width_words")
        if not isinstance(word, int) or isinstance(word, bool) or word < 0:
            errors.append(f"{loc}.word must be a non-negative integer")
            continue
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            errors.append(f"{loc}.width_words must be a positive integer")
            continue
        if isinstance(header_words, int) and word + width > header_words:
            errors.append(f"{loc} exceeds header_words={header_words}")
        for occupied_word in range(word, word + width):
            previous = occupied_header_words.get(occupied_word)
            if previous is not None:
                errors.append(
                    f"{loc} overlaps header word {occupied_word} already used by {previous}"
                )
            elif name:
                occupied_header_words[occupied_word] = name
    missing_header_fields = sorted(REQUIRED_HEADER_FIELDS - header_names)
    if missing_header_fields:
        errors.append(
            "$.trace.protocol.header_fields lacks required fields: "
            + ", ".join(missing_header_fields)
        )
    header_field_by_name = {
        field.get("name"): field for field in header_fields if isinstance(field, dict)
    }
    for required_name in REQUIRED_HEADER_FIELDS - {"role_counts"}:
        field = header_field_by_name.get(required_name)
        if isinstance(field, dict) and field.get("width_words") != 1:
            errors.append(
                f"$.trace.protocol.header_fields field {required_name!r} must have width_words=1"
            )
    role_counts_field = header_field_by_name.get("role_counts")
    if isinstance(role_counts_field, dict) and role_counts_field.get(
        "width_words"
    ) != len(roles):
        errors.append(
            "$.trace.protocol.header_fields field 'role_counts' must have width_words "
            f"equal to len(trace.roles)={len(roles)}"
        )

    output = trace.get("output")
    if not isinstance(output, dict):
        errors.append("$.trace.output must be an object")
        output = {}
    for key in ("carrier", "capacity_check", "write_phase", "rendezvous"):
        _required_string(output, key, "$.trace.output", errors)
    if not isinstance(output.get("overwrites_user_output"), bool):
        errors.append("$.trace.output.overwrites_user_output must be boolean")

    points = spec.get("points")
    if not isinstance(points, list) or not points:
        errors.append("$.points must be a non-empty list")
        points = []
    point_by_id: dict[str, dict[str, Any]] = {}
    for index, point in enumerate(points):
        loc = f"$.points[{index}]"
        if not isinstance(point, dict):
            errors.append(f"{loc} must be an object")
            continue
        point_id = _required_string(point, "id", loc, errors)
        if point_id:
            if not POINT_ID_RE.fullmatch(point_id):
                errors.append(f"{loc}.id must end in _S and be a C-like identifier")
            if point_id in point_by_id:
                errors.append(f"duplicate point id: {point_id}")
            else:
                point_by_id[point_id] = point
        _required_string(point, "stage", loc, errors)
        for key in ("scope", "why", "semantics", "perturbation"):
            _required_string(point, key, loc, errors)

        edge = point.get("edge")
        if edge not in ("begin", "end", "instant"):
            errors.append(f"{loc}.edge must be begin, end, or instant")
        quality = point.get("boundary_quality")
        allowed_quality = {
            "begin": {"causal", "inferred"},
            "end": {"actual", "issue", "inferred"},
            "instant": {"causal", "actual", "issue", "inferred"},
        }
        if edge in allowed_quality and quality not in allowed_quality[edge]:
            errors.append(
                f"{loc}.boundary_quality={quality!r} is invalid for edge={edge!r}"
            )

        point_roles = point.get("roles")
        if not isinstance(point_roles, list) or not point_roles:
            errors.append(f"{loc}.roles must be a non-empty list")
        elif any(role not in role_set for role in point_roles):
            errors.append(f"{loc}.roles must be a subset of $.trace.roles")

        source = point.get("source")
        if not isinstance(source, dict):
            errors.append(f"{loc}.source must be an object")
            source = {}
        source_file = _required_string(source, "file", f"{loc}.source", errors)
        if source_file:
            source_path = Path(source_file)
            if source_path.is_absolute() or ".." in source_path.parts:
                errors.append(f"{loc}.source.file must be a safe repo-relative path")
        _required_string(source, "symbol", f"{loc}.source", errors)
        placement = source.get("placement")
        if placement not in ("before", "after", "at"):
            errors.append(f"{loc}.source.placement must be before, after, or at")
        marker = _required_string(source, "marker", f"{loc}.source", errors)
        if marker and point_id and marker != f"TRACE_POINT: {point_id}":
            errors.append(f"{loc}.source.marker must equal 'TRACE_POINT: {point_id}'")
        distance = source.get("max_line_distance")
        if (
            not isinstance(distance, int)
            or isinstance(distance, bool)
            or distance < 0
            or distance > 200
        ):
            errors.append(
                f"{loc}.source.max_line_distance must be an integer in [0, 200]"
            )
        anchor = source.get("target_anchor")
        if not isinstance(anchor, dict):
            errors.append(f"{loc}.source.target_anchor must be an object")
        else:
            if anchor.get("mode") not in ("literal", "regex"):
                errors.append(
                    f"{loc}.source.target_anchor.mode must be literal or regex"
                )
            _required_string(anchor, "pattern", f"{loc}.source.target_anchor", errors)
            occurrence = anchor.get("occurrence")
            if occurrence is not None and (
                not isinstance(occurrence, int)
                or isinstance(occurrence, bool)
                or occurrence <= 0
            ):
                errors.append(
                    f"{loc}.source.target_anchor.occurrence must be a positive integer when set"
                )

        dependency = point.get("dependency")
        if not isinstance(dependency, dict):
            errors.append(f"{loc}.dependency must be an object")
        else:
            for key in (
                "precondition",
                "producer_pipeline",
                "timestamp_pipeline",
                "bridge",
                "proof",
            ):
                _required_string(dependency, key, f"{loc}.dependency", errors)

    record_fields = protocol.get("record_fields")
    if not isinstance(record_fields, list) or not record_fields:
        errors.append("$.trace.protocol.record_fields must be a non-empty list")
        record_fields = []
    timestamp_coverage = {
        (point_id, role): 0
        for point_id, point in point_by_id.items()
        for role in point.get("roles", [])
    }
    occupied_record_words: dict[str, dict[int, str]] = {role: {} for role in roles}
    clock_width_words = (
        (bits + word_bits - 1) // word_bits
        if isinstance(bits, int) and isinstance(word_bits, int) and word_bits > 0
        else None
    )
    for index, field in enumerate(record_fields):
        loc = f"$.trace.protocol.record_fields[{index}]"
        if not isinstance(field, dict):
            errors.append(f"{loc} must be an object")
            continue
        name = _required_string(field, "name", loc, errors)
        kind = field.get("kind")
        if kind not in ("timestamp", "counter"):
            errors.append(f"{loc}.kind must be timestamp or counter")
        field_roles = field.get("roles")
        if not isinstance(field_roles, list) or not field_roles:
            errors.append(f"{loc}.roles must be a non-empty list")
            field_roles = []
        elif len(set(field_roles)) != len(field_roles):
            errors.append(f"{loc}.roles must be unique")
        elif any(role not in role_set for role in field_roles):
            errors.append(f"{loc}.roles must be a subset of $.trace.roles")
        if kind == "timestamp" and name:
            point = point_by_id.get(name)
            if point is None:
                errors.append(f"{loc}.name references unknown timestamp point {name!r}")
            else:
                unsupported = [
                    role for role in field_roles if role not in point.get("roles", [])
                ]
                if unsupported:
                    errors.append(
                        f"{loc}.roles {unsupported} are not declared by timestamp point {name!r}"
                    )
                for role in field_roles:
                    key = (name, role)
                    if key in timestamp_coverage:
                        timestamp_coverage[key] += 1
        elif kind == "counter" and name and not FIELD_NAME_RE.fullmatch(name):
            errors.append(f"{loc}.name must be a C-like identifier")

        word = field.get("word")
        width = field.get("width_words")
        count = field.get("count")
        stride = field.get("stride_words")
        integers = (
            isinstance(word, int) and not isinstance(word, bool) and word >= 0,
            isinstance(width, int) and not isinstance(width, bool) and width > 0,
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            isinstance(stride, int) and not isinstance(stride, bool) and stride > 0,
        )
        for valid, key, requirement in zip(
            integers,
            ("word", "width_words", "count", "stride_words"),
            (
                "a non-negative integer",
                "a positive integer",
                "a positive integer",
                "a positive integer",
            ),
        ):
            if not valid:
                errors.append(f"{loc}.{key} must be {requirement}")
        if not all(integers):
            continue
        if kind == "counter" and count != 1:
            errors.append(f"{loc}.count must be 1 for counter fields")
        if (
            kind == "timestamp"
            and clock_width_words is not None
            and width < clock_width_words
        ):
            errors.append(
                f"{loc}.width_words={width} cannot hold a {bits}-bit timestamp in {word_bits}-bit words"
            )
        last_end = word + (count - 1) * stride + width
        if isinstance(record_words, int) and last_end > record_words:
            errors.append(f"{loc} exceeds record_words={record_words}")
        for role in field_roles:
            role_occupied = occupied_record_words.setdefault(role, {})
            for repetition in range(count):
                start_word = word + repetition * stride
                for occupied_word in range(start_word, start_word + width):
                    previous = role_occupied.get(occupied_word)
                    if previous is not None:
                        errors.append(
                            f"{loc} overlaps {role} record word {occupied_word} already used by {previous}"
                        )
                    elif name:
                        role_occupied[occupied_word] = f"{name}[{repetition}]"
    for (point_id, role), coverage in timestamp_coverage.items():
        if coverage != 1:
            errors.append(
                f"timestamp point {point_id!r} for role {role!r} must have exactly one "
                f"record_fields mapping, found {coverage}"
            )

    spans = spec.get("spans")
    if not isinstance(spans, list) or not spans:
        errors.append("$.spans must be a non-empty list")
        spans = []
    span_ids: set[str] = set()
    for index, span in enumerate(spans):
        loc = f"$.spans[{index}]"
        if not isinstance(span, dict):
            errors.append(f"{loc} must be an object")
            continue
        span_id = _required_string(span, "id", loc, errors)
        if span_id:
            if span_id in span_ids:
                errors.append(f"duplicate span id: {span_id}")
            span_ids.add(span_id)
        _required_string(span, "label", loc, errors)
        begin = _required_string(span, "begin", loc, errors)
        end = _required_string(span, "end", loc, errors)
        if begin and begin not in point_by_id:
            errors.append(f"{loc}.begin references unknown point {begin!r}")
        if end and end not in point_by_id:
            errors.append(f"{loc}.end references unknown point {end!r}")
        span_roles = span.get("roles")
        if not isinstance(span_roles, list) or not span_roles:
            errors.append(f"{loc}.roles must be a non-empty list")
            span_roles = []
        elif any(role not in role_set for role in span_roles):
            errors.append(f"{loc}.roles must be a subset of $.trace.roles")
        quality = span.get("quality")
        if quality not in ("exact", "issue", "inferred"):
            errors.append(f"{loc}.quality must be exact, issue, or inferred")
        color = _required_string(span, "color", loc, errors)
        if color and not COLOR_RE.fullmatch(color):
            errors.append(f"{loc}.color must be a six-digit #RRGGBB value")
        if quality == "exact" and begin in point_by_id and end in point_by_id:
            begin_point = point_by_id[begin]
            end_point = point_by_id[end]
            if (
                begin_point.get("edge") != "begin"
                or begin_point.get("boundary_quality") != "causal"
            ):
                errors.append(
                    f"{loc} is exact but begin {begin!r} is not a causal begin"
                )
            if (
                end_point.get("edge") != "end"
                or end_point.get("boundary_quality") != "actual"
            ):
                errors.append(
                    f"{loc} is exact but end {end!r} is not an actual-completion end"
                )
            if any(role not in begin_point.get("roles", []) for role in span_roles):
                errors.append(
                    f"{loc}.roles are not all supported by begin point {begin!r}"
                )
            if any(role not in end_point.get("roles", []) for role in span_roles):
                errors.append(f"{loc}.roles are not all supported by end point {end!r}")

    if not todo_paths:
        try:
            actual_schema_id = _parse_uint(
                protocol.get("schema_id"), "$.trace.protocol.schema_id"
            )
            expected_schema_id = _expected_schema_id(spec)
            if actual_schema_id != expected_schema_id:
                errors.append(
                    "$.trace.protocol.schema_id does not match the deterministic contract: "
                    f"actual=0x{actual_schema_id:08x}, expected=0x{expected_schema_id:08x}; "
                    "update the spec, device header, decoder, and fixtures together"
                )
        except ContractError as exc:
            errors.append(str(exc))
    return errors


def _load_valid_spec(path: Path) -> dict[str, Any]:
    spec = _load_json(path)
    errors = _validate_spec(spec)
    if errors:
        raise ContractError("invalid trace spec:\n  - " + "\n  - ".join(errors))
    return spec


def _replace_template_tokens(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_template_tokens(child, replacements) for child in value]
    if isinstance(value, dict):
        return {
            key: _replace_template_tokens(child, replacements)
            for key, child in value.items()
        }
    return value


def _cmd_init(args: argparse.Namespace) -> int:
    name = args.name.strip()
    if not TRACE_NAME_RE.fullmatch(name):
        raise ContractError(
            "--name must start with a letter/digit and use only letters, digits, '.', '_' or '-'"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ContractError(f"refusing to overwrite existing spec: {output}")
    template_path = (
        Path(__file__).resolve().parent.parent / "assets" / "trace-spec.template.json"
    )
    template = _load_json(template_path)
    rendered = _replace_template_tokens(
        template,
        {
            "TODO_TRACE_NAME": name,
            "TODO_UNIQUE_U32_SCHEMA_ID": "0x00000000",
        },
    )
    schema_id = _expected_schema_id(rendered)
    rendered["trace"]["protocol"]["schema_id"] = f"0x{schema_id:08x}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[OK] initialized trace spec: {output}")
    print(
        f"[INFO] template schema_id=0x{schema_id:08x}; validation will report the new "
        "deterministic id after TODO fields/layout are replaced"
    )
    print(
        "[NEXT] replace every TODO, run validate, adopt its expected schema id, then audit --phase plan"
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    spec = _load_valid_spec(args.spec.expanduser().resolve())
    print(
        f"[OK] valid trace spec: {args.spec} "
        f"sha256={_spec_sha256(spec)} points={len(spec['points'])} spans={len(spec['spans'])}"
    )
    return 0


def _anchor_matches(
    lines: list[str], anchor: dict[str, Any], description: str
) -> list[int]:
    mode = anchor["mode"]
    pattern = anchor["pattern"]
    if mode == "literal":
        return [index for index, line in enumerate(lines, start=1) if pattern in line]
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ContractError(f"invalid regex for {description}: {exc}") from exc
    return [index for index, line in enumerate(lines, start=1) if compiled.search(line)]


def _select_match(matches: list[int], anchor: dict[str, Any], description: str) -> int:
    if not matches:
        raise ContractError(f"{description} did not match any source line")
    occurrence = anchor.get("occurrence")
    if occurrence is None:
        if len(matches) != 1:
            raise ContractError(
                f"{description} matched {len(matches)} lines {matches}; make the semantic anchor unique"
            )
        return matches[0]
    if occurrence > len(matches):
        raise ContractError(
            f"{description} occurrence {occurrence} exceeds {len(matches)} matches {matches}"
        )
    return matches[occurrence - 1]


def _run_git(repo_root: Path, arguments: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    revision = _run_git(repo_root, ["rev-parse", "HEAD"])
    branch = _run_git(repo_root, ["branch", "--show-current"])
    status_text = _run_git(repo_root, ["status", "--short"])
    full_status = status_text.splitlines() if status_text else []
    status = full_status[:200]
    diff_raw = _run_git(repo_root, ["diff", "--no-ext-diff", "--raw"])
    is_git_repository = revision is not None
    return {
        "is_git_repository": is_git_repository,
        "revision": revision,
        "branch": branch,
        "dirty": bool(full_status) if is_git_repository else None,
        "status_short": status,
        "status_truncated": len(full_status) > len(status),
        "tracked_diff_raw_sha256": (
            _sha256_bytes(diff_raw.encode("utf-8")) if diff_raw is not None else None
        ),
    }


def _source_context(
    lines: list[str], line_number: int, radius: int = 2
) -> dict[str, Any]:
    start = max(1, line_number - radius)
    end = min(len(lines), line_number + radius)
    return {
        "start_line": start,
        "end_line": end,
        "lines": [
            {"line": number, "text": lines[number - 1]}
            for number in range(start, end + 1)
        ],
    }


def _audit_point(point: dict[str, Any], repo_root: Path, phase: str) -> dict[str, Any]:
    source = point["source"]
    relative_file = Path(source["file"])
    source_path = (repo_root / relative_file).resolve()
    try:
        source_path.relative_to(repo_root)
    except ValueError as exc:
        raise ContractError(f"{point['id']}: source path escapes repo root") from exc
    if not source_path.is_file():
        raise ContractError(
            f"{point['id']}: source file does not exist: {relative_file}"
        )

    raw = source_path.read_bytes()
    try:
        source_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ContractError(
            f"{point['id']}: source file is not UTF-8: {relative_file}"
        ) from exc
    lines = source_text.splitlines()
    target_matches = _anchor_matches(
        lines, source["target_anchor"], f"{point['id']} target anchor"
    )
    target_line = _select_match(
        target_matches, source["target_anchor"], f"{point['id']} target anchor"
    )

    marker_matches = [
        index for index, line in enumerate(lines, start=1) if source["marker"] in line
    ]
    marker_line: int | None = None
    marker_context: dict[str, Any] | None = None
    if phase == "implemented":
        marker_line = _select_match(
            marker_matches,
            {"mode": "literal", "pattern": source["marker"]},
            f"{point['id']} instrumentation marker",
        )
        distance = abs(marker_line - target_line)
        if distance > source["max_line_distance"]:
            raise ContractError(
                f"{point['id']}: marker line {marker_line} is {distance} lines from target "
                f"line {target_line}, exceeding max_line_distance={source['max_line_distance']}"
            )
        placement = source["placement"]
        if placement == "after" and marker_line < target_line:
            raise ContractError(
                f"{point['id']}: marker line {marker_line} must be after target line {target_line}"
            )
        if placement == "before" and marker_line > target_line:
            raise ContractError(
                f"{point['id']}: marker line {marker_line} must be before target line {target_line}"
            )
        if placement == "at" and marker_line != target_line:
            raise ContractError(
                f"{point['id']}: placement=at requires marker and target on line {target_line}"
            )
        marker_context = _source_context(lines, marker_line)

    return {
        "id": point["id"],
        "stage": point["stage"],
        "edge": point["edge"],
        "boundary_quality": point["boundary_quality"],
        "roles": point["roles"],
        "scope": point["scope"],
        "file": relative_file.as_posix(),
        "symbol": source["symbol"],
        "placement": source["placement"],
        "target_anchor": source["target_anchor"],
        "target_match_count": len(target_matches),
        "target_line": target_line,
        "target_context": _source_context(lines, target_line),
        "marker": source["marker"],
        "marker_match_count": len(marker_matches),
        "marker_line": marker_line,
        "marker_context": marker_context,
        "max_line_distance": source["max_line_distance"],
        "source_file_sha256": _sha256_bytes(raw),
        "why": point["why"],
        "semantics": point["semantics"],
        "dependency": point["dependency"],
        "perturbation": point["perturbation"],
    }


def _markdown_cell(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _context_markdown(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    return "\n".join(
        f"{entry['line']:>6}  {entry['text']}".rstrip() for entry in context["lines"]
    )


def _audit_markdown(audit: dict[str, Any]) -> str:
    git = audit["git"]
    lines = [
        "# Trace point audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Phase: `{audit['phase']}`",
        f"- Trace: `{audit['trace_name']}`",
        f"- Repository: `{audit['repo_root']}`",
        f"- Revision: `{git.get('revision') or 'not-a-git-repository'}`",
        f"- Branch: `{git.get('branch') or 'detached-or-unknown'}`",
        f"- Dirty: `{git.get('dirty')}`",
        f"- Spec SHA-256: `{audit['spec']['sha256']}`",
        "",
    ]
    if audit.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in audit["errors"])
        lines.append("")
    lines.extend(
        [
            "## Placement summary",
            "",
            "| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |",
            "|---|---|---|---:|---|---|---|---|",
        ]
    )
    for point in audit["points"]:
        resolved_line = point["marker_line"] or point["target_line"]
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    point["id"],
                    point["roles"],
                    f"{point['file']}:{resolved_line}",
                    point["target_line"],
                    point["placement"],
                    point["boundary_quality"],
                    point["dependency"]["bridge"],
                    point["why"],
                )
            )
            + " |"
        )
    lines.append("")
    lines.extend(["## Point evidence", ""])
    for point in audit["points"]:
        resolved_line = point["marker_line"] or point["target_line"]
        lines.extend(
            [
                f"### `{point['id']}`",
                "",
                f"- Where: `{point['file']}:{resolved_line}` in `{point['symbol']}`; "
                f"`{point['placement']}` target line `{point['target_line']}`.",
                f"- Cardinality: {point['scope']}",
                f"- Boundary claim: `{point['boundary_quality']}` — {point['semantics']}",
                f"- Why: {point['why']}",
                f"- Dependency: {point['dependency']['precondition']}",
                f"- Pipeline proof: `{point['dependency']['producer_pipeline']}` → "
                f"`{point['dependency']['timestamp_pipeline']}` using "
                f"`{point['dependency']['bridge']}` — {point['dependency']['proof']}",
                f"- Perturbation: {point['perturbation']}",
                f"- Source SHA-256: `{point['source_file_sha256']}`",
                "",
                "```text",
                _context_markdown(point["marker_context"] or point["target_context"]),
                "```",
                "",
            ]
        )
    if git.get("status_short"):
        lines.extend(["## Working tree status", "", "```text"])
        lines.extend(git["status_short"])
        lines.extend(["```", ""])
    return "\n".join(lines)


def _cmd_audit(args: argparse.Namespace) -> int:
    spec_path = args.spec.expanduser().resolve()
    spec = _load_valid_spec(spec_path)
    repo_root = args.repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        raise ContractError(f"repo root is not a directory: {repo_root}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audited_points: list[dict[str, Any]] = []
    errors: list[str] = []
    for point in spec["points"]:
        try:
            audited_points.append(_audit_point(point, repo_root, args.phase))
        except ContractError as exc:
            errors.append(str(exc))

    audit = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "status": "fail" if errors else "pass",
        "phase": args.phase,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trace_name": spec["trace"]["name"],
        "repo_root": str(repo_root),
        "git": _git_metadata(repo_root),
        "spec": {
            "path": str(spec_path),
            "sha256": _spec_sha256(spec),
            "protocol_version": spec["trace"]["protocol"]["version"],
            "schema_id": spec["trace"]["protocol"]["schema_id"],
        },
        "points": audited_points,
        "errors": errors,
    }
    json_path = output_dir / "trace_point_audit.json"
    markdown_path = output_dir / "trace_point_audit.md"
    json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_audit_markdown(audit), encoding="utf-8")
    if errors:
        raise ContractError(
            f"{args.phase} audit failed; reports written to {output_dir}:\n  - "
            + "\n  - ".join(errors)
        )
    print(
        f"[OK] {args.phase} audit passed: points={len(audited_points)} "
        f"json={json_path} markdown={markdown_path}"
    )
    return 0


def _timestamp_sequence(value: Any, field: str, bits: int) -> list[int | None]:
    values = value if isinstance(value, list) else [value]
    result: list[int | None] = []
    for index, item in enumerate(values):
        if item is None:
            result.append(None)
            continue
        result.append(_parse_uint(item, f"{field}[{index}]", bits))
    return result


def _validate_normalized_trace(
    trace_data: Any, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(trace_data, dict):
        raise ContractError("normalized trace root must be an object")
    if trace_data.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ContractError(
            f"normalized trace schema_version must be {TRACE_SCHEMA_VERSION}"
        )
    trace_spec = spec["trace"]
    protocol = trace_spec["protocol"]
    if trace_data.get("trace_name") != trace_spec["name"]:
        raise ContractError(
            f"trace_name mismatch: data={trace_data.get('trace_name')!r}, spec={trace_spec['name']!r}"
        )
    if trace_data.get("protocol_version") != protocol["version"]:
        raise ContractError(
            f"protocol_version mismatch: data={trace_data.get('protocol_version')!r}, "
            f"spec={protocol['version']!r}"
        )
    if _parse_uint(trace_data.get("schema_id"), "$.schema_id") != _parse_uint(
        protocol["schema_id"], "$.trace.protocol.schema_id"
    ):
        raise ContractError("schema_id mismatch between normalized trace and spec")

    metadata = trace_data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ContractError("$.metadata must be an object when set")

    bits = trace_spec["clock"]["bits"]
    roles = trace_spec["roles"]
    point_by_id = {point["id"]: point for point in spec["points"]}
    point_ids = set(point_by_id)
    timestamp_field_counts = {
        (role, field["name"]): field["count"]
        for field in trace_spec["protocol"]["record_fields"]
        if field["kind"] == "timestamp"
        for role in field["roles"]
    }
    records = trace_data.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("$.records must be a non-empty list")
    seen: set[tuple[str, int]] = set()
    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        loc = f"$.records[{index}]"
        if not isinstance(record, dict):
            raise ContractError(f"{loc} must be an object")
        role = record.get("role")
        core = record.get("core")
        if role not in roles:
            raise ContractError(f"{loc}.role={role!r} is not declared in the spec")
        if not isinstance(core, int) or isinstance(core, bool) or core < 0:
            raise ContractError(f"{loc}.core must be a non-negative integer")
        key = (role, core)
        if key in seen:
            raise ContractError(f"duplicate role/core record: {role}/{core}")
        seen.add(key)
        label = record.get("label", f"{role}-{core:02d}")
        if not isinstance(label, str) or not label:
            raise ContractError(f"{loc}.label must be a non-empty string when set")
        timestamps = record.get("timestamps")
        if not isinstance(timestamps, dict):
            raise ContractError(f"{loc}.timestamps must be an object")
        unknown = sorted(set(timestamps) - point_ids)
        if unknown:
            raise ContractError(
                f"{loc}.timestamps contains undeclared points: {unknown}"
            )
        wrong_role = sorted(
            point_id
            for point_id in timestamps
            if role not in point_by_id[point_id]["roles"]
        )
        if wrong_role:
            raise ContractError(
                f"{loc}.timestamps contains points not declared for role {role}: {wrong_role}"
            )
        normalized_timestamps = {
            point_id: _timestamp_sequence(value, f"{loc}.timestamps.{point_id}", bits)
            for point_id, value in timestamps.items()
        }
        wrong_counts = [
            f"{point_id}={len(values)} (expected {timestamp_field_counts[(role, point_id)]})"
            for point_id, values in normalized_timestamps.items()
            if len(values) != timestamp_field_counts[(role, point_id)]
        ]
        if wrong_counts:
            raise ContractError(
                f"{loc}.timestamps repetition count disagrees with record_fields: "
                + ", ".join(wrong_counts)
            )
        counters = record.get("counters", {})
        if not isinstance(counters, dict) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in counters.values()
        ):
            raise ContractError(f"{loc}.counters must map strings to numbers")
        normalized_records.append(
            {
                "role": role,
                "core": core,
                "label": label,
                "timestamps": normalized_timestamps,
                "counters": counters,
            }
        )

    reference = trace_data.get("reference")
    if not isinstance(reference, dict):
        raise ContractError("$.reference must be an object")
    reference_point = reference.get("point")
    occurrence = reference.get("occurrence", 0)
    if reference_point not in point_ids:
        raise ContractError("$.reference.point must name a declared point")
    if (
        not isinstance(occurrence, int)
        or isinstance(occurrence, bool)
        or occurrence < 0
    ):
        raise ContractError("$.reference.occurrence must be a non-negative integer")
    reference_normalized = {
        "role": reference.get("role"),
        "core": reference.get("core"),
        "point": reference_point,
        "occurrence": occurrence,
    }
    if trace_spec["clock"]["comparable_across_records"]:
        if reference_normalized["role"] not in roles:
            raise ContractError("$.reference.role must name a declared role")
        if (
            not isinstance(reference_normalized["core"], int)
            or isinstance(reference_normalized["core"], bool)
            or reference_normalized["core"] < 0
        ):
            raise ContractError("$.reference.core must be a non-negative integer")
    return normalized_records, reference_normalized


def _get_timestamp(
    record: dict[str, Any], point_id: str, occurrence: int, missing_value: int | None
) -> int | None:
    values = record["timestamps"].get(point_id, [])
    if occurrence >= len(values):
        return None
    value = values[occurrence]
    if value is None or (missing_value is not None and value == missing_value):
        return None
    return value


def _relative(raw: int, reference: int, bits: int) -> int:
    modulus = 1 << bits
    delta = (raw - reference) & (modulus - 1)
    if delta >= modulus // 2:
        delta -= modulus
    return delta


def _modular_span(begin: int, end: int, bits: int) -> int:
    return (end - begin) & ((1 << bits) - 1)


def _load_audit(path: Path | None, spec: dict[str, Any]) -> dict[str, Any] | None:
    if path is None:
        return None
    audit = _load_json(path.expanduser().resolve())
    if not isinstance(audit, dict):
        raise ContractError("audit root must be an object")
    if audit.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
        raise ContractError("unsupported audit schema version")
    if audit.get("status") != "pass" or audit.get("phase") != "implemented":
        raise ContractError("renderer requires a passing implemented audit")
    audit_spec = audit.get("spec")
    if not isinstance(audit_spec, dict) or audit_spec.get("sha256") != _spec_sha256(
        spec
    ):
        raise ContractError("audit was generated from a different trace spec")
    audited_ids = {point.get("id") for point in audit.get("points", [])}
    expected_ids = {point["id"] for point in spec["points"]}
    if audited_ids != expected_ids:
        raise ContractError("audit point set does not match the trace spec")
    return audit


def _json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _render_html(
    spec: dict[str, Any],
    trace_data: dict[str, Any],
    records: list[dict[str, Any]],
    reference: dict[str, Any],
    audit: dict[str, Any] | None,
) -> str:
    trace_spec = spec["trace"]
    clock = trace_spec["clock"]
    bits = clock["bits"]
    missing_value = clock.get("missing_value")
    comparable = clock["comparable_across_records"]
    point_by_id = {point["id"]: point for point in spec["points"]}
    role_order = {role: index for index, role in enumerate(trace_spec["roles"])}
    records = sorted(
        records, key=lambda record: (role_order[record["role"]], record["core"])
    )

    if comparable:
        reference_record = next(
            (
                record
                for record in records
                if record["role"] == reference["role"]
                and record["core"] == reference["core"]
            ),
            None,
        )
        if reference_record is None:
            raise ContractError("reference role/core record is absent")
        shared_reference = _get_timestamp(
            reference_record, reference["point"], reference["occurrence"], missing_value
        )
        if shared_reference is None:
            raise ContractError("reference timestamp is missing")
        row_references = {
            (record["role"], record["core"]): shared_reference for record in records
        }
        axis_note = (
            f"shared {clock['scope']} {clock['unit']} clock; reference "
            f"{reference_record['label']}/{reference['point']}[{reference['occurrence']}]"
        )
    else:
        row_references: dict[tuple[str, int], int] = {}
        for record in records:
            raw_reference = _get_timestamp(
                record, reference["point"], reference["occurrence"], missing_value
            )
            if raw_reference is None:
                raise ContractError(
                    f"local-clock record {record['label']} lacks reference point {reference['point']}"
                )
            row_references[(record["role"], record["core"])] = raw_reference
        axis_note = (
            f"per-record local {clock['unit']} axes normalized at "
            f"{reference['point']}[{reference['occurrence']}]; cross-record gaps are invalid"
        )

    plotted: list[int] = []
    rendered_spans: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        record_key = (record["role"], record["core"])
        raw_reference = row_references[record_key]
        row_spans: list[dict[str, Any]] = []
        for span in spec["spans"]:
            if record["role"] not in span["roles"]:
                continue
            begins = record["timestamps"].get(span["begin"], [])
            ends = record["timestamps"].get(span["end"], [])
            if len(begins) != len(ends):
                raise ContractError(
                    f"{record['label']} span {span['id']} has {len(begins)} begins and {len(ends)} ends"
                )
            for occurrence, (begin, end) in enumerate(zip(begins, ends)):
                if (
                    begin is None
                    or end is None
                    or (
                        missing_value is not None
                        and (begin == missing_value or end == missing_value)
                    )
                ):
                    continue
                duration = _modular_span(begin, end, bits)
                if duration >= (1 << (bits - 1)):
                    raise ContractError(
                        f"{record['label']} {span['id']}[{occurrence}] has ambiguous/backward span {duration}"
                    )
                begin_delta = _relative(begin, raw_reference, bits)
                end_delta = begin_delta + duration
                plotted.extend((begin_delta, end_delta))
                row_spans.append(
                    {
                        "span": span,
                        "occurrence": occurrence,
                        "begin_raw": begin,
                        "end_raw": end,
                        "begin": begin_delta,
                        "end": end_delta,
                        "duration": duration,
                    }
                )
        rendered_spans[record_key] = row_spans
        for point_id, values in record["timestamps"].items():
            for value in values:
                if value is None or (
                    missing_value is not None and value == missing_value
                ):
                    continue
                plotted.append(_relative(value, raw_reference, bits))

    if not plotted:
        raise ContractError("normalized trace contains no plottable timestamps")
    min_cycle = min(plotted)
    max_cycle = max(plotted)
    if min_cycle == max_cycle:
        max_cycle += 1
    cycle_range = max_cycle - min_cycle

    label_width = 138
    plot_width = 1500
    counter_width = 420
    row_height = 30
    top = 72
    svg_width = label_width + plot_width + counter_width
    svg_height = top + len(records) * row_height + 48

    def x_pos(cycle: int) -> float:
        return label_width + (cycle - min_cycle) * plot_width / cycle_range

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}">',
        "<style>",
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#0f172a}",
        ".label{font-size:12px;font-weight:650}.axis{font-size:10px}.counter{font-size:10px}",
        ".grid{stroke:#cbd5e1;stroke-width:.7}.group{stroke:#334155;stroke-width:1.4}",
        ".marker{stroke:#111827;stroke-width:.65;opacity:.38}.nonexact{stroke:#475569;stroke-dasharray:3 2}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="8" y="20" font-size="16" font-weight="700">{html.escape(trace_spec["name"])} critical path</text>',
        f'<text x="8" y="40" font-size="10">{html.escape(axis_note)}</text>',
    ]

    legend_x = 8
    for span in spec["spans"]:
        label = span["label"]
        opacity = "0.92" if span["quality"] == "exact" else "0.42"
        svg.append(
            f'<rect x="{legend_x}" y="49" width="12" height="9" fill="{span["color"]}" opacity="{opacity}"/>'
            f'<text x="{legend_x + 16}" y="57" font-size="9">{html.escape(label)} ({span["quality"]})</text>'
        )
        legend_x += 34 + len(label) * 6 + len(span["quality"]) * 6

    for tick_index in range(11):
        cycle = min_cycle + cycle_range * tick_index // 10
        x = x_pos(cycle)
        svg.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top - 4}" x2="{x:.2f}" y2="{svg_height - 24}"/>'
            f'<text class="axis" x="{x:.2f}" y="{svg_height - 8}" text-anchor="middle">{cycle}</text>'
        )

    previous_role: str | None = None
    points_in_spans = {
        point_id for span in spec["spans"] for point_id in (span["begin"], span["end"])
    }
    for row_index, record in enumerate(records):
        y = top + row_index * row_height
        if record["role"] != previous_role:
            svg.append(
                f'<line class="group" x1="0" y1="{y}" x2="{svg_width}" y2="{y}"/>'
            )
        previous_role = record["role"]
        shade = "#f8fafc" if row_index % 2 else "#ffffff"
        svg.append(
            f'<rect x="0" y="{y}" width="{svg_width}" height="{row_height}" fill="{shade}"/>'
        )
        svg.append(
            f'<text class="label" x="6" y="{y + 20}">{html.escape(record["label"])}</text>'
        )

        for rendered in rendered_spans[(record["role"], record["core"])]:
            span = rendered["span"]
            begin_point = point_by_id[span["begin"]]
            end_point = point_by_id[span["end"]]
            x_begin = x_pos(rendered["begin"])
            width = max(1.0, x_pos(rendered["end"]) - x_begin)
            exact = span["quality"] == "exact"
            css_class = "" if exact else "nonexact"
            opacity = "0.92" if exact else "0.42"
            title = html.escape(
                f"{span['label']}[{rendered['occurrence']}]: {rendered['begin']} -> "
                f"{rendered['end']} {clock['unit']} (duration {rendered['duration']}); "
                f"begin={begin_point['semantics']}; end={end_point['semantics']}; "
                f"why={begin_point['why']} / {end_point['why']}"
            )
            svg.append(
                f'<rect class="{css_class}" x="{x_begin:.2f}" y="{y + 7}" width="{width:.2f}" '
                f'height="16" rx="2" fill="{span["color"]}" opacity="{opacity}"><title>{title}</title></rect>'
            )

        raw_reference = row_references[(record["role"], record["core"])]
        for point_id, values in record["timestamps"].items():
            if point_id in points_in_spans:
                continue
            point = point_by_id[point_id]
            for occurrence, value in enumerate(values):
                if value is None or (
                    missing_value is not None and value == missing_value
                ):
                    continue
                delta = _relative(value, raw_reference, bits)
                x = x_pos(delta)
                title = html.escape(
                    f"{point_id}[{occurrence}]={delta} {clock['unit']}; "
                    f"{point['boundary_quality']}: {point['semantics']}; why={point['why']}"
                )
                svg.append(
                    f'<line class="marker" x1="{x:.2f}" y1="{y + 4}" x2="{x:.2f}" y2="{y + 26}"><title>{title}</title></line>'
                )

        counter_text = (
            " ".join(
                f"{key}={value}" for key, value in sorted(record["counters"].items())
            )
            or "no counters"
        )
        svg.append(
            f'<text class="counter" x="{label_width + plot_width + 8}" y="{y + 19}">{html.escape(counter_text)}</text>'
        )
    svg.append("</svg>")

    metadata_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(_json_pretty(value))}</code></td></tr>"
        for key, value in sorted(trace_data.get("metadata", {}).items())
    )
    protocol = trace_spec["protocol"]
    overview_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in (
            ("trace", trace_spec["name"]),
            ("protocol version", protocol["version"]),
            ("schema id", protocol["schema_id"]),
            ("roles", ", ".join(trace_spec["roles"])),
            (
                "wire layout",
                f"{protocol['endianness']} uint{protocol['word_bits']}; "
                f"header={protocol['header_words']} words, record={protocol['record_words']} words",
            ),
            ("record order", protocol["record_order"]),
            (
                "clock",
                f"{clock['api']} / {clock['bits']}-bit {clock['unit']} / {clock['scope']}",
            ),
            ("carrier", trace_spec["output"]["carrier"]),
            ("overwrites user output", trace_spec["output"]["overwrites_user_output"]),
            ("capacity check", trace_spec["output"]["capacity_check"]),
            ("write phase", trace_spec["output"]["write_phase"]),
            ("rendezvous", trace_spec["output"]["rendezvous"]),
        )
    )

    audit_by_id = (
        {point["id"]: point for point in audit.get("points", [])} if audit else {}
    )
    audit_rows: list[str] = []
    for point in spec["points"]:
        evidence = audit_by_id.get(point["id"])
        location = (
            f"{evidence['file']}:{evidence['marker_line'] or evidence['target_line']}"
            if evidence
            else f"{point['source']['file']} (unresolved; no implemented audit supplied)"
        )
        dependency = point["dependency"]
        audit_rows.append(
            "<tr>"
            f"<td><code>{html.escape(point['id'])}</code></td>"
            f"<td>{html.escape(', '.join(point['roles']))}</td>"
            f"<td><code>{html.escape(location)}</code><br>{html.escape(point['source']['placement'])} "
            f"in <code>{html.escape(point['source']['symbol'])}</code></td>"
            f"<td><code>{html.escape(point['boundary_quality'])}</code><br>{html.escape(point['semantics'])}</td>"
            f"<td><code>{html.escape(dependency['bridge'])}</code><br>{html.escape(dependency['proof'])}</td>"
            f"<td>{html.escape(point['why'])}</td>"
            f"<td>{html.escape(point['perturbation'])}</td>"
            "</tr>"
        )

    warning = (
        ""
        if comparable
        else '<div class="warning">Clock domains are not comparable across records. Rows are locally normalized; do not infer cross-row handoff gaps.</div>'
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(trace_spec['name'])} critical path</title>"
        "<style>body{margin:0;background:#e2e8f0;color:#0f172a;font-family:system-ui,sans-serif}"
        "main{padding:14px}.panel{background:#fff;border-radius:9px;box-shadow:0 1px 6px #64748b;"
        "padding:12px;margin-bottom:14px;overflow:auto}h1,h2{margin:.2em 0 .6em}"
        "table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #cbd5e1;"
        "padding:6px;text-align:left;vertical-align:top}th{background:#f1f5f9}.warning{background:#fef3c7;"
        "border:1px solid #f59e0b;padding:9px;border-radius:6px;margin-bottom:10px}code{white-space:pre-wrap}"
        "</style></head><body><main>"
        f"<h1>{html.escape(trace_spec['name'])}</h1>{warning}"
        f'<section class="panel"><h2>Contract and runtime metadata</h2><table>{overview_rows}{metadata_rows}</table></section>'
        f'<section class="panel">{"".join(svg)}</section>'
        '<section class="panel"><h2>Trace-point placement audit</h2>'
        "<table><thead><tr><th>Point</th><th>Roles</th><th>Where</th><th>Boundary claim</th>"
        "<th>Pipeline bridge/proof</th><th>Why</th><th>Perturbation</th></tr></thead><tbody>"
        f"{''.join(audit_rows)}</tbody></table></section>"
        "</main></body></html>"
    )


def _combine_protocol_words(words: list[int], word_bits: int, endianness: str) -> int:
    ordered = words if endianness == "little" else list(reversed(words))
    value = 0
    for index, word in enumerate(ordered):
        value |= word << (index * word_bits)
    return value


def _decode_header_field(
    words: list[int], field: dict[str, Any], protocol: dict[str, Any]
) -> int | list[int]:
    start = field["word"]
    width = field["width_words"]
    selected = words[start : start + width]
    if field["name"] == "role_counts":
        return selected
    if width == 1:
        return selected[0]
    return _combine_protocol_words(
        selected, protocol["word_bits"], protocol["endianness"]
    )


def _parse_assignments(values: list[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        name, separator, value = item.partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not name or not value:
            raise ContractError(f"{option} requires NAME=VALUE, got {item!r}")
        if name in parsed:
            raise ContractError(f"{option} repeats {name!r}")
        parsed[name] = value
    return parsed


def _cmd_decode(args: argparse.Namespace) -> int:
    spec_path = args.spec.expanduser().resolve()
    raw_path = args.raw.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    spec = _load_valid_spec(spec_path)
    trace_spec = spec["trace"]
    protocol = trace_spec["protocol"]
    if protocol["record_order"] != "role-major":
        raise ContractError(
            "generic decode supports only record_order=role-major; use a thin target decoder for custom order"
        )
    word_bytes = protocol["word_bits"] // 8
    if args.byte_offset < 0 or args.byte_offset % word_bytes != 0:
        raise ContractError(
            f"--byte-offset must be non-negative and aligned to {word_bytes} bytes"
        )
    try:
        raw = raw_path.read_bytes()
    except FileNotFoundError as exc:
        raise ContractError(f"raw trace file does not exist: {raw_path}") from exc
    if args.byte_offset >= len(raw):
        raise ContractError(
            f"byte offset {args.byte_offset} is outside raw file of {len(raw)} bytes"
        )
    payload = raw[args.byte_offset :]
    header_bytes = protocol["header_words"] * word_bytes
    if len(payload) < header_bytes:
        raise ContractError(
            f"raw trace has {len(payload)} bytes after offset; header requires {header_bytes}"
        )
    words = [
        int.from_bytes(
            payload[offset : offset + word_bytes],
            byteorder=protocol["endianness"],
            signed=False,
        )
        for offset in range(0, len(payload) - len(payload) % word_bytes, word_bytes)
    ]
    header_words = words[: protocol["header_words"]]
    header = {
        field["name"]: _decode_header_field(header_words, field, protocol)
        for field in protocol["header_fields"]
    }

    expected_header_values = {
        "magic": _parse_uint(protocol["magic"], "$.trace.protocol.magic"),
        "version": protocol["version"],
        "schema_id": _parse_uint(protocol["schema_id"], "$.trace.protocol.schema_id"),
        "header_words": protocol["header_words"],
        "record_words": protocol["record_words"],
        "clock_bits": trace_spec["clock"]["bits"],
    }
    mismatches = [
        f"{name}=0x{actual:08x} (expected 0x{expected:08x})"
        if name in ("magic", "schema_id")
        else f"{name}={actual} (expected {expected})"
        for name, expected in expected_header_values.items()
        for actual in (header[name],)
        if actual != expected
    ]
    if mismatches:
        raise ContractError("runtime trace header mismatch: " + ", ".join(mismatches))

    role_counts = header["role_counts"]
    if not isinstance(role_counts, list) or len(role_counts) != len(
        trace_spec["roles"]
    ):
        raise ContractError("runtime role_counts width does not match trace.roles")
    record_count = header["record_count"]
    if not isinstance(record_count, int) or record_count != sum(role_counts):
        raise ContractError(
            f"runtime record_count={record_count!r} does not equal sum(role_counts)={sum(role_counts)}"
        )
    repetition_count = header["repetition_count"]
    capture_capacity = header["capture_capacity"]
    truncated = header["truncated"]
    if not all(
        isinstance(value, int) and value >= 0
        for value in (repetition_count, capture_capacity, truncated)
    ):
        raise ContractError(
            "runtime repetition/capacity/truncated header fields are invalid"
        )
    if truncated not in (0, 1) or truncated != int(repetition_count > capture_capacity):
        raise ContractError(
            f"runtime truncated={truncated} disagrees with repetition_count={repetition_count} "
            f"and capture_capacity={capture_capacity}"
        )
    if truncated and not args.allow_truncated:
        raise ContractError(
            "runtime trace is truncated; rerun with sufficient capture capacity or pass "
            "--allow-truncated only when partial capture is intentional"
        )

    expected_headers = _parse_assignments(args.expect_header, "--expect-header")
    for name, expected_text in expected_headers.items():
        if name not in header:
            raise ContractError(f"--expect-header names unknown header field {name!r}")
        actual = header[name]
        if not isinstance(actual, int):
            raise ContractError(
                f"--expect-header cannot compare aggregate field {name!r}; use --expect-role-count for role_counts"
            )
        expected = _parse_uint(
            expected_text, f"--expect-header {name}", protocol["word_bits"]
        )
        if actual != expected:
            raise ContractError(
                f"runtime header {name}={actual} does not match expected {expected}"
            )

    expected_role_counts = _parse_assignments(
        args.expect_role_count, "--expect-role-count"
    )
    role_count_by_name = dict(zip(trace_spec["roles"], role_counts))
    for role, expected_text in expected_role_counts.items():
        if role not in role_count_by_name:
            raise ContractError(f"--expect-role-count names unknown role {role!r}")
        expected = _parse_uint(expected_text, f"--expect-role-count {role}")
        actual = role_count_by_name[role]
        if actual != expected:
            raise ContractError(
                f"runtime role count {role}={actual} does not match expected {expected}"
            )

    if (args.require_flags_mask is None) != (args.require_flags_value is None):
        raise ContractError(
            "--require-flags-mask and --require-flags-value must be supplied together"
        )
    if args.require_flags_mask is not None:
        mask = _parse_uint(
            args.require_flags_mask, "--require-flags-mask", protocol["word_bits"]
        )
        expected_flags = _parse_uint(
            args.require_flags_value, "--require-flags-value", protocol["word_bits"]
        )
        if expected_flags & ~mask:
            raise ContractError(
                "--require-flags-value contains bits outside --require-flags-mask"
            )
        actual_flags = header["flags"]
        if actual_flags & mask != expected_flags:
            raise ContractError(
                f"runtime flags 0x{actual_flags:08x} masked by 0x{mask:08x} "
                f"do not equal 0x{expected_flags:08x}"
            )

    required_words = protocol["header_words"] + record_count * protocol["record_words"]
    required_bytes = required_words * word_bytes
    if len(payload) < required_bytes:
        raise ContractError(
            f"raw trace has {len(payload)} bytes after offset; decoded layout requires {required_bytes}"
        )

    records: list[dict[str, Any]] = []
    record_index = 0
    for role, role_count in zip(trace_spec["roles"], role_counts):
        for core in range(role_count):
            base = protocol["header_words"] + record_index * protocol["record_words"]
            record_words = words[base : base + protocol["record_words"]]
            timestamps: dict[str, list[int]] = {}
            counters: dict[str, int] = {}
            for field in protocol["record_fields"]:
                if role not in field["roles"]:
                    continue
                values: list[int] = []
                for repetition in range(field["count"]):
                    start = field["word"] + repetition * field["stride_words"]
                    selected = record_words[start : start + field["width_words"]]
                    values.append(
                        _combine_protocol_words(
                            selected, protocol["word_bits"], protocol["endianness"]
                        )
                    )
                if field["kind"] == "timestamp":
                    timestamps[field["name"]] = values
                else:
                    counters[field["name"]] = values[0]
            records.append(
                {
                    "role": role,
                    "core": core,
                    "label": f"{role}-{core:02d}",
                    "timestamps": timestamps,
                    "counters": counters,
                }
            )
            record_index += 1

    normalized = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_name": trace_spec["name"],
        "protocol_version": protocol["version"],
        "schema_id": protocol["schema_id"],
        "metadata": {
            "rank": header["rank"],
            "runtime_config_hash": f"0x{header['runtime_config_hash']:08x}",
            "repetition_count": repetition_count,
            "capture_capacity": capture_capacity,
            "trace_truncated": bool(truncated),
            "flags": f"0x{header['flags']:08x}",
            "raw_artifact": str(raw_path),
            "raw_sha256": _sha256_bytes(raw),
            "byte_offset": args.byte_offset,
            "required_trace_bytes": required_bytes,
            "available_bytes_after_offset": len(payload),
            "header": header,
            "spec_sha256": _spec_sha256(spec),
        },
        "reference": {
            "role": args.reference_role,
            "core": args.reference_core,
            "point": args.reference_point,
            "occurrence": args.reference_occurrence,
        },
        "records": records,
    }
    _validate_normalized_trace(normalized, spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[OK] decoded raw trace: {output_path} records={record_count} "
        f"rank={header['rank']} repetitions={repetition_count}/{capture_capacity} "
        f"truncated={bool(truncated)}"
    )
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    spec_path = args.spec.expanduser().resolve()
    trace_path = args.trace.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    spec = _load_valid_spec(spec_path)
    trace_data = _load_json(trace_path)
    records, reference = _validate_normalized_trace(trace_data, spec)
    audit = _load_audit(args.audit, spec)
    rendered = _render_html(spec, trace_data, records, reference, audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(
        f"[OK] rendered timeline: {output_path} records={len(records)} "
        f"audit={'included' if audit else 'not supplied'}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, validate, source-audit, decode, and render operator critical-path trace contracts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="copy a new trace-spec template")
    init_parser.add_argument("--name", required=True, help="trace contract name")
    init_parser.add_argument(
        "--output", required=True, type=Path, help="new trace-spec JSON path"
    )
    init_parser.set_defaults(handler=_cmd_init)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a completed trace spec"
    )
    validate_parser.add_argument("--spec", required=True, type=Path)
    validate_parser.set_defaults(handler=_cmd_validate)

    audit_parser = subparsers.add_parser(
        "audit",
        help="resolve semantic anchors and optionally implemented TRACE_POINT markers",
    )
    audit_parser.add_argument("--phase", choices=("plan", "implemented"), required=True)
    audit_parser.add_argument("--spec", required=True, type=Path)
    audit_parser.add_argument("--repo-root", required=True, type=Path)
    audit_parser.add_argument("--output-dir", required=True, type=Path)
    audit_parser.set_defaults(handler=_cmd_audit)

    decode_parser = subparsers.add_parser(
        "decode", help="decode a raw binary trace using the spec's checked word layout"
    )
    decode_parser.add_argument("--spec", required=True, type=Path)
    decode_parser.add_argument("--raw", required=True, type=Path)
    decode_parser.add_argument("--output", required=True, type=Path)
    decode_parser.add_argument("--byte-offset", type=int, default=0)
    decode_parser.add_argument("--reference-role", required=True)
    decode_parser.add_argument("--reference-core", required=True, type=int)
    decode_parser.add_argument("--reference-point", required=True)
    decode_parser.add_argument("--reference-occurrence", type=int, default=0)
    decode_parser.add_argument(
        "--expect-header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="require a scalar runtime header field value; repeat as needed",
    )
    decode_parser.add_argument(
        "--expect-role-count",
        action="append",
        default=[],
        metavar="ROLE=COUNT",
        help="require a runtime physical-core count for a role; repeat as needed",
    )
    decode_parser.add_argument("--require-flags-mask")
    decode_parser.add_argument("--require-flags-value")
    decode_parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="accept an explicitly truncated capture; default is to reject it",
    )
    decode_parser.set_defaults(handler=_cmd_decode)

    render_parser = subparsers.add_parser(
        "render", help="render normalized JSON as a self-contained HTML timeline"
    )
    render_parser.add_argument("--spec", required=True, type=Path)
    render_parser.add_argument("--trace", required=True, type=Path)
    render_parser.add_argument(
        "--audit", type=Path, help="passing implemented trace_point_audit.json"
    )
    render_parser.add_argument("--output", required=True, type=Path)
    render_parser.set_defaults(handler=_cmd_render)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
