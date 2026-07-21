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

"""Versioned, bounded relations between baseline and derived consistency cases."""

from collections import Counter, defaultdict
import re
from typing import Any, Dict, Iterable, List, Sequence

from .adapter import CaseAdapter
from .metadata_coverage import CoverageContext, MetadataCoverageAnalyzer, S2Category
from .model import ConsistencyCase, TokenOrigin
from .reporting import stable_json_digest
from .schema import get_q_lengths


SCHEMA_VERSION = 1
MANIFEST_KIND = "smla_mqsmla_batch_consistency_relation"
ALLOWED_MODES = ("reorder", "split", "token-split", "shape-change")
SCENARIO = "same-source-cross-call"
SUPPLEMENTAL_SCENARIOS = ("independent-cross-case", "single-call-intra-case")


class RelationManifestError(ValueError):
    """Raised when a relation cannot safely identify comparable output tokens."""


def _slug(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "case")).strip("_")
    return normalized or "case"


def _case_name(input_data: Dict[str, Any]) -> str:
    adapter = CaseAdapter(input_data)
    value = input_data.get("case_name") or input_data.get("Testcase_Name")
    value = value or adapter.params.get("testcase_name") or adapter.params.get("Testcase_Name")
    return str(value or "case")


def make_baseline_id(input_data: Dict[str, Any], input_content_fingerprint: str) -> str:
    """Build a stable id that changes when the exact baseline input changes."""
    return (
        f"{CaseAdapter(input_data).name}:{_slug(_case_name(input_data))}:"
        f"{input_content_fingerprint[:12]}"
    )


def compact_origins(origins: Sequence[TokenOrigin]) -> List[Dict[str, int]]:
    """Compress logical target-order origins into contiguous source-token spans."""
    spans: List[Dict[str, int]] = []
    for target_index, origin in enumerate(origins):
        if (
            spans
            and spans[-1]["source_batch"] == origin.source_batch
            and spans[-1]["source_token_end"] == origin.source_token
            and spans[-1]["target_end"] == target_index
        ):
            spans[-1]["source_token_end"] += 1
            spans[-1]["target_end"] += 1
            continue
        spans.append(
            {
                "target_start": target_index,
                "target_end": target_index + 1,
                "source_batch": int(origin.source_batch),
                "source_token_start": int(origin.source_token),
                "source_token_end": int(origin.source_token) + 1,
            }
        )
    return spans


def _landmarks(length: int) -> Dict[str, int]:
    if length < 1:
        return {}
    return {"min": 0, "mid": (length - 1) // 2, "max": length - 1}


def _landmark_coverage(
    baseline_lengths: Sequence[int], origins: Sequence[TokenOrigin]
) -> Dict[str, Any]:
    selected_batches = {origin.source_batch for origin in origins}
    selected_tokens: Dict[int, set] = defaultdict(set)
    for origin in origins:
        selected_tokens[origin.source_batch].add(origin.source_token)

    batch_landmarks = _landmarks(len(baseline_lengths))
    token_landmarks = {}
    for batch, length in enumerate(baseline_lengths):
        token_landmarks[str(batch)] = {
            name: {"index": index, "covered": index in selected_tokens[batch]}
            for name, index in _landmarks(int(length)).items()
        }
    return {
        "source_batch": {
            name: {"index": index, "covered": index in selected_batches}
            for name, index in batch_landmarks.items()
        },
        "source_token_by_batch": token_landmarks,
    }


def _coverage_descriptor(input_data: Dict[str, Any]) -> Dict[str, Any]:
    adapter = CaseAdapter(input_data)
    tensors = adapter.tensors
    context = CoverageContext.from_input_data(input_data)
    s2_counts = Counter(
        MetadataCoverageAnalyzer.classify_s2(value).category.value
        for value in context.level1_s2_values()
        if value > 0
    )
    formula = context.formula_audit()
    return {
        "layout_q": adapter.get_layout_q(),
        "layout_kv": adapter.get_layout_kv(),
        "batch_size": adapter.get_batch_size(),
        "q_lengths": list(context.q_lengths),
        "mask": {
            "ori_mask_mode": int(tensors.get("ori_mask_mode", 0)),
            "cmp_mask_mode": int(tensors.get("cmp_mask_mode", 0)),
            "position_sensitive": (
                int(tensors.get("ori_mask_mode", 0)) in (3, 4)
                or int(tensors.get("cmp_mask_mode", 0)) in (3, 4)
            ),
        },
        "visible_s2_categories": dict(sorted(s2_counts.items())),
        "formula_audit": {
            "checked_token_count": formula["checked_token_count"],
            "quotient_formula_conflict_count": formula["quotient_formula_conflict_count"],
            "tile_formula_conflict_count": formula["tile_formula_conflict_count"],
            "zero_tile_guard_count": formula["zero_tile_guard_count"],
        },
    }


def _relation_config(transform: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "mode",
        "mode_name",
        "order",
        "seed",
        "group_index",
        "batch_ids",
        "randomized",
        "complete_partition",
        "part_index",
        "token_range",
        "split_sizes",
        "batch_id",
        "common_tokens",
        "baseline_extra_tokens",
        "derived_extra_tokens",
        "layout_q",
        "layout_kv",
        "operator",
    )
    return {key: transform[key] for key in keys if key in transform}


def _validate_origin_bounds(
    origins: Sequence[TokenOrigin], baseline_lengths: Sequence[int]
) -> None:
    if not origins:
        raise RelationManifestError("A relation must map at least one output token")
    if len(set(origins)) != len(origins):
        raise RelationManifestError("A relation contains duplicate TokenOrigin values")
    for origin in origins:
        if origin.source_batch < 0 or origin.source_batch >= len(baseline_lengths):
            raise RelationManifestError(
                f"source_batch {origin.source_batch} is outside [0, {len(baseline_lengths)})"
            )
        length = int(baseline_lengths[origin.source_batch])
        if origin.source_token < 0 or origin.source_token >= length:
            raise RelationManifestError(
                f"source token ({origin.source_batch}, {origin.source_token}) is outside "
                f"baseline length {length}"
            )


def build_case_relation(
    baseline_data: Dict[str, Any],
    baseline_id: str,
    case: ConsistencyCase,
    derived_pt: str,
    derived_input_content_fingerprint: str,
) -> Dict[str, Any]:
    """Build and precheck one explicit baseline-to-derived comparison relation."""
    mode = str(case.transform_meta.get("mode_name") or "")
    if mode not in ALLOWED_MODES:
        raise RelationManifestError(f"Unsupported relation mode: {mode!r}")

    baseline_lengths = get_q_lengths(baseline_data, valid_only=True)
    _validate_origin_bounds(case.output_origins, baseline_lengths)
    spans = compact_origins(case.output_origins)
    expected_scope = dict(case.input_data.get("_consistency_expected_scope") or {})
    compared_count = sum(int(value) for value in expected_scope.get("compared_q_lengths", []))
    if compared_count != len(case.output_origins):
        raise RelationManifestError(
            f"Compared token count {compared_count} != origin count {len(case.output_origins)}"
        )

    oracle = case.transform_meta.get("semantic_oracle")
    if mode in ("token-split", "shape-change") and oracle is None:
        raise RelationManifestError(f"Mode {mode} requires a semantic oracle")
    if oracle is not None and not oracle.get("pass"):
        raise RelationManifestError(f"Semantic oracle failed for {case.name}")

    warnings = []
    if case.transform_meta.get("complete_partition") is False:
        warnings.append("The selected calls do not form a complete baseline partition")
    identity = {
        "type": "TokenOrigin",
        "target_order": "derived-valid-logical-token-order",
        "mapped_token_count": len(case.output_origins),
        "origin_spans": spans,
        "origin_spans_sha256": stable_json_digest(spans),
    }
    relation = {
        "relation_id": f"{baseline_id}/{_slug(case.name)}",
        "case_id": case.name,
        "baseline_case_id": baseline_id,
        "mode": mode,
        "scenario": SCENARIO,
        "derived_pt": derived_pt,
        "derived_input_content_fingerprint_sha256": derived_input_content_fingerprint,
        "configuration": _relation_config(case.transform_meta),
        "comparison_identity": identity,
        "expected_scope": expected_scope,
        "coverage": _coverage_descriptor(case.input_data),
        "source_landmark_coverage": _landmark_coverage(
            baseline_lengths, case.output_origins
        ),
        "precheck": {
            "status": "PASS",
            "checks": {
                "mode_known": True,
                "origin_bounds_valid": True,
                "origins_unique": True,
                "compared_count_matches_origins": True,
                "semantic_oracle_pass": oracle is None or bool(oracle.get("pass")),
            },
            "warnings": warnings,
        },
    }
    return relation


def _coverage_summary(relations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mode_counts = Counter(relation["mode"] for relation in relations)
    s2_counts = Counter()
    cells = Counter()
    layout_q_counts = Counter()
    layout_kv_counts = Counter()
    mask_class_counts = Counter()
    batch_landmarks = Counter()
    token_landmarks = Counter()
    for relation in relations:
        coverage = relation["coverage"]
        mask = coverage["mask"]
        layout_q_counts[coverage["layout_q"]] += 1
        layout_kv_counts[coverage["layout_kv"]] += 1
        mask_class_counts[
            "position-sensitive" if mask["position_sensitive"] else "no-mask"
        ] += 1
        cell = (
            relation["mode"],
            coverage["layout_q"],
            coverage["layout_kv"],
            str(mask["ori_mask_mode"]),
            str(mask["cmp_mask_mode"]),
        )
        cells[cell] += 1
        s2_counts.update(coverage["visible_s2_categories"])
        landmarks = relation["source_landmark_coverage"]
        for name, item in landmarks["source_batch"].items():
            batch_landmarks[name] += int(item["covered"])
        for per_batch in landmarks["source_token_by_batch"].values():
            for name, item in per_batch.items():
                token_landmarks[name] += int(item["covered"])

    return {
        "scope": "CPU input/relation coverage only; not NPU or kernel-metadata coverage",
        "relation_count": len(relations),
        "mode_counts": {mode: mode_counts.get(mode, 0) for mode in ALLOWED_MODES},
        "missing_modes": [mode for mode in ALLOWED_MODES if not mode_counts.get(mode)],
        "layout_q_counts": dict(sorted(layout_q_counts.items())),
        "missing_layout_q": [name for name in ("BSND", "TND") if not layout_q_counts[name]],
        "layout_kv_counts": dict(sorted(layout_kv_counts.items())),
        "missing_layout_kv": [
            name for name in ("BSND", "TND", "PA_BBND") if not layout_kv_counts[name]
        ],
        "mask_class_counts": dict(sorted(mask_class_counts.items())),
        "missing_mask_classes": [
            name for name in ("no-mask", "position-sensitive") if not mask_class_counts[name]
        ],
        "visible_s2_category_token_counts": {
            category.value: s2_counts.get(category.value, 0) for category in S2Category
        },
        "missing_visible_s2_categories": [
            category.value for category in S2Category if not s2_counts[category.value]
        ],
        "coverage_cells": [
            {
                "mode": key[0],
                "layout_q": key[1],
                "layout_kv": key[2],
                "ori_mask_mode": int(key[3]),
                "cmp_mask_mode": int(key[4]),
                "relation_count": count,
            }
            for key, count in sorted(cells.items())
        ],
        "source_landmark_hits": {
            "batch": dict(sorted(batch_landmarks.items())),
            "token": dict(sorted(token_landmarks.items())),
        },
    }


def build_relation_manifest(
    baseline_data: Dict[str, Any],
    baseline_pt: str,
    baseline_input_content_fingerprint: str,
    baseline_cpu_output_md5: str,
    source_excel: str,
    source_sheet: str,
    relations: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a single-baseline manifest suitable for JSON persistence and CI precheck."""
    baseline_id = make_baseline_id(baseline_data, baseline_input_content_fingerprint)
    if any(relation.get("baseline_case_id") != baseline_id for relation in relations):
        raise RelationManifestError("A relation points at a different baseline id")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "manifest_scope": "single-baseline",
        "operator": CaseAdapter(baseline_data).name,
        "claim_scope": "CPU-only relation precheck; no NPU batch-consistency PASS claim",
        "scenario_coverage": {
            "implemented": [SCENARIO],
            "supplemental_gaps": list(SUPPLEMENTAL_SCENARIOS),
        },
        "baselines": [
            {
                "case_id": baseline_id,
                "case_name": _case_name(baseline_data),
                "source_pt": baseline_pt,
                "source_excel": source_excel,
                "source_sheet": source_sheet,
                "input_content_fingerprint_sha256": baseline_input_content_fingerprint,
                "cpu_output_md5": baseline_cpu_output_md5,
                "valid_q_lengths": get_q_lengths(baseline_data, valid_only=True),
                "coverage": _coverage_descriptor(baseline_data),
            }
        ],
        "relations": list(relations),
        "coverage": _coverage_summary(relations),
        "precheck": {
            "status": "PASS" if relations else "FAIL",
            "baseline_count": 1,
            "relation_count": len(relations),
            "errors": [] if relations else ["No derived relation was generated"],
        },
    }
    validate_relation_manifest(manifest)
    return manifest


def merge_relation_manifests(manifests: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-baseline manifests into one operator-level coverage artifact."""
    items = list(manifests)
    if not items:
        raise RelationManifestError("At least one relation manifest is required")
    for manifest in items:
        validate_relation_manifest(manifest)
    operators = {manifest["operator"] for manifest in items}
    if len(operators) != 1:
        raise RelationManifestError(f"Cannot merge different operators: {sorted(operators)}")
    baselines = [baseline for manifest in items for baseline in manifest["baselines"]]
    relations = [relation for manifest in items for relation in manifest["relations"]]
    merged = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "manifest_scope": "operator-collection",
        "operator": next(iter(operators)),
        "claim_scope": "CPU-only relation precheck; no NPU batch-consistency PASS claim",
        "scenario_coverage": {
            "implemented": [SCENARIO],
            "supplemental_gaps": list(SUPPLEMENTAL_SCENARIOS),
        },
        "baselines": baselines,
        "relations": relations,
        "coverage": _coverage_summary(relations),
        "precheck": {
            "status": "PASS",
            "baseline_count": len(baselines),
            "relation_count": len(relations),
            "errors": [],
        },
    }
    validate_relation_manifest(merged)
    return merged


def validate_relation_manifest(manifest: Dict[str, Any]) -> None:
    """Validate ids, bounds, span digests and comparison counts without JSON Schema."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RelationManifestError(
            f"Unsupported relation schema_version={manifest.get('schema_version')}"
        )
    if manifest.get("kind") != MANIFEST_KIND:
        raise RelationManifestError(f"Unexpected manifest kind={manifest.get('kind')!r}")
    scenario_coverage = manifest.get("scenario_coverage") or {}
    if scenario_coverage.get("implemented") != [SCENARIO]:
        raise RelationManifestError("Manifest scenario classification is missing or invalid")
    if scenario_coverage.get("supplemental_gaps") != list(SUPPLEMENTAL_SCENARIOS):
        raise RelationManifestError("Manifest supplemental scenario gaps are missing or invalid")
    baselines = manifest.get("baselines")
    relations = manifest.get("relations")
    if not isinstance(baselines, list) or not baselines:
        raise RelationManifestError("Manifest must contain at least one baseline")
    if not isinstance(relations, list) or not relations:
        raise RelationManifestError("Manifest must contain at least one relation")

    baseline_by_id = {}
    for baseline in baselines:
        case_id = baseline.get("case_id")
        if not case_id or case_id in baseline_by_id:
            raise RelationManifestError(f"Duplicate or empty baseline case_id={case_id!r}")
        baseline_by_id[case_id] = baseline

    relation_ids = set()
    for relation in relations:
        relation_id = relation.get("relation_id")
        if not relation_id or relation_id in relation_ids:
            raise RelationManifestError(f"Duplicate or empty relation_id={relation_id!r}")
        relation_ids.add(relation_id)
        if relation.get("mode") not in ALLOWED_MODES:
            raise RelationManifestError(f"Unknown relation mode={relation.get('mode')!r}")
        if relation.get("scenario") != SCENARIO:
            raise RelationManifestError(f"Relation {relation_id} scenario is invalid")
        relation_precheck = relation.get("precheck") or {}
        if relation_precheck.get("status") != "PASS":
            raise RelationManifestError(f"Relation {relation_id} precheck is not PASS")
        checks = relation_precheck.get("checks") or {}
        if not all(
            checks.get(name) is True
            for name in (
                "mode_known",
                "origin_bounds_valid",
                "origins_unique",
                "compared_count_matches_origins",
                "semantic_oracle_pass",
            )
        ):
            raise RelationManifestError(f"Relation {relation_id} has a failed precheck")
        baseline = baseline_by_id.get(relation.get("baseline_case_id"))
        if baseline is None:
            raise RelationManifestError(
                f"Relation {relation_id} references an unknown baseline"
            )

        identity = relation.get("comparison_identity") or {}
        spans = identity.get("origin_spans")
        if identity.get("type") != "TokenOrigin" or not isinstance(spans, list) or not spans:
            raise RelationManifestError(f"Relation {relation_id} has no TokenOrigin spans")
        if identity.get("origin_spans_sha256") != stable_json_digest(spans):
            raise RelationManifestError(f"Relation {relation_id} origin span digest mismatch")

        target_cursor = 0
        mapped_count = 0
        source_ranges: Dict[int, List[Any]] = defaultdict(list)
        lengths = baseline["valid_q_lengths"]
        for span in spans:
            if span["target_start"] != target_cursor or span["target_end"] <= target_cursor:
                raise RelationManifestError(
                    f"Relation {relation_id} target spans are not contiguous"
                )
            source_batch = span["source_batch"]
            source_start = span["source_token_start"]
            source_end = span["source_token_end"]
            if source_batch < 0 or source_batch >= len(lengths):
                raise RelationManifestError(f"Relation {relation_id} source batch is out of range")
            if source_start < 0 or source_end <= source_start or source_end > lengths[source_batch]:
                raise RelationManifestError(f"Relation {relation_id} source token span is invalid")
            width = span["target_end"] - span["target_start"]
            if source_end - source_start != width:
                raise RelationManifestError(f"Relation {relation_id} source/target span widths differ")
            source_ranges[source_batch].append((source_start, source_end))
            target_cursor = span["target_end"]
            mapped_count += width

        for ranges in source_ranges.values():
            ordered = sorted(ranges)
            if any(current[0] < previous[1] for previous, current in zip(ordered, ordered[1:])):
                raise RelationManifestError(f"Relation {relation_id} reuses a source token")
        if mapped_count != identity.get("mapped_token_count"):
            raise RelationManifestError(f"Relation {relation_id} mapped token count mismatch")
        expected_scope = relation.get("expected_scope") or {}
        compared_count = sum(int(value) for value in expected_scope.get("compared_q_lengths", []))
        if compared_count != mapped_count:
            raise RelationManifestError(f"Relation {relation_id} expected scope mismatch")

    manifest_precheck = manifest.get("precheck") or {}
    if manifest_precheck.get("status") != "PASS":
        raise RelationManifestError("Manifest precheck is not PASS")
    if manifest_precheck.get("baseline_count") != len(baselines):
        raise RelationManifestError("Manifest baseline count mismatch")
    if manifest_precheck.get("relation_count") != len(relations):
        raise RelationManifestError("Manifest relation count mismatch")
    if manifest_precheck.get("errors"):
        raise RelationManifestError("Manifest precheck contains errors")
