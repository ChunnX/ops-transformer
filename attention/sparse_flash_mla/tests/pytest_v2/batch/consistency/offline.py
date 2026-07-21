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

"""CPU-only PT round-trip checks for consistency modes 1-4."""

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .adapter import CaseAdapter
from .compare import OriginComparator, TensorDigest
from .config import CaseConfigParser, ModeRegistry
from .model import ConsistencyCase
from .relation_manifest import (
    build_case_relation,
    build_relation_manifest,
    make_baseline_id,
)
from .reporting import (
    attach_provenance,
    build_input_summary,
    input_content_digest,
    stable_json_digest,
)
from .schema import build_baseline_origins, get_q_lengths, validate_schema
from .transform import (
    transform_mode1_reorder,
    transform_mode2_split,
    transform_mode3_token_partition,
    transform_mode4_shape_change,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def mode_options_from_mapping(values: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the optional consistency columns carried beside the original Excel schema."""
    result: Dict[str, Any] = {}
    modes = values.get("consistency_modes")
    if isinstance(modes, str) and modes.strip():
        tokens = tuple(item.strip() for item in modes.split(",") if item.strip())
        if any(token.isdigit() for token in tokens):
            raise ValueError("Excel consistency_modes must use canonical names, not numeric aliases")
        result["modes"] = tuple(
            mode.value for mode in ModeRegistry.parse(modes, warn_legacy=False)
        )
    groups = values.get("split_groups")
    if isinstance(groups, str) and groups.strip():
        result["split_groups"] = CaseConfigParser.split_groups(groups)
    sizes = values.get("token_split_sizes")
    if isinstance(sizes, str) and sizes.strip():
        result["token_split_sizes"] = CaseConfigParser.split_sizes(
            sizes, "token_split_sizes"
        )
    for source, target in (
        ("mode_batch", "mode_batch"),
        ("common_tokens", "common_tokens"),
        ("derived_extra", "derived_extra"),
    ):
        value = values.get(source)
        if value is not None and not (isinstance(value, float) and value != value):
            result[target] = int(value)
    if result.get("mode_batch", 0) < 0:
        raise ValueError("mode_batch must be >= 0")
    if "common_tokens" in result and result["common_tokens"] < 1:
        raise ValueError("common_tokens must be >= 1")
    if result.get("derived_extra", 0) < 0:
        raise ValueError("derived_extra must be >= 0")
    return result


def _output_template(input_data: Dict[str, Any]) -> torch.Tensor:
    return torch.zeros_like(CaseAdapter(input_data).tensors["q"])


def _lse_template(
    baseline_lse: torch.Tensor,
    input_data: Dict[str, Any],
) -> torch.Tensor:
    adapter = CaseAdapter(input_data)
    q = adapter.tensors["q"]
    if adapter.get_layout_q() == "TND":
        return baseline_lse.new_zeros((q.shape[0], baseline_lse.shape[1], baseline_lse.shape[2]))
    return baseline_lse.new_zeros(
        (q.shape[0], baseline_lse.shape[1], q.shape[1], baseline_lse.shape[3])
    )


def materialize_expected_output(
    baseline_data: Dict[str, Any],
    case: ConsistencyCase,
) -> Dict[str, Any]:
    """Materialize the origin-mapped CPU expected output used by the NPU comparison."""
    layout_q = CaseAdapter(baseline_data).get_layout_q()
    baseline_physical = get_q_lengths(baseline_data)
    baseline_valid = get_q_lengths(baseline_data, valid_only=True)
    derived_physical = get_q_lengths(case.input_data)
    derived_valid = list(
        case.transform_meta.get("valid_q_lengths")
        or [len(case.output_origins)]
    )
    baseline_origins = build_baseline_origins(baseline_data, valid_only=True)
    expected = OriginComparator.align_to_template(
        baseline_data["cpu_output"],
        baseline_origins,
        baseline_physical,
        case.output_origins,
        derived_physical,
        _output_template(case.input_data),
        layout_q,
        source_valid_q_lengths=baseline_valid,
        target_valid_q_lengths=derived_valid,
    )
    case.input_data["cpu_output"] = expected

    baseline_lse = baseline_data.get("softmax_lse")
    if baseline_lse is not None:
        case.input_data["softmax_lse"] = OriginComparator.align_to_template(
            baseline_lse,
            baseline_origins,
            baseline_physical,
            case.output_origins,
            derived_physical,
            _lse_template(baseline_lse, case.input_data),
            layout_q,
            lse=True,
            source_valid_q_lengths=baseline_valid,
            target_valid_q_lengths=derived_valid,
        )
    case.input_data["_consistency_expected_scope"] = {
        "kind": "origin_mapped_cpu_golden",
        "physical_q_lengths": derived_physical,
        "compared_q_lengths": derived_valid,
        "unmapped_tokens": sum(derived_physical) - sum(derived_valid),
    }
    return case.input_data


def build_mode_cases(
    baseline_data: Dict[str, Any],
    modes: Iterable[str],
    seed: int = 20260716,
    split_groups: Optional[Sequence[Sequence[int]]] = None,
    token_split_sizes: Optional[Sequence[int]] = None,
    mode_batch: int = 0,
    common_tokens: Optional[int] = None,
    derived_extra: Optional[int] = None,
) -> List[ConsistencyCase]:
    """Construct all requested derived calls without touching deterministic/NPU state."""
    requested = tuple(modes)
    supported = {"reorder", "split", "token-split", "shape-change"}
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported offline consistency modes: {unknown}")
    q_lengths = get_q_lengths(baseline_data, valid_only=True)
    batch_size = len(q_lengths)
    cases: List[ConsistencyCase] = []
    if "reorder" in requested:
        cases.append(transform_mode1_reorder(baseline_data, seed=seed))
    if "split" in requested:
        groups = split_groups
        if groups is None:
            pivot = 3 if batch_size == 8 else max(1, batch_size // 2)
            groups = [list(range(pivot)), list(range(pivot, batch_size))]
        cases.extend(transform_mode2_split(baseline_data, groups=groups))
    if "token-split" in requested:
        token_count = q_lengths[mode_batch]
        sizes = list(token_split_sizes or [token_count // 2, token_count - token_count // 2])
        cases.extend(transform_mode3_token_partition(baseline_data, mode_batch, sizes))
    if "shape-change" in requested:
        token_count = q_lengths[mode_batch]
        common = common_tokens if common_tokens is not None else max(1, token_count - 1)
        extra = derived_extra if derived_extra is not None else token_count - common + 1
        cases.append(transform_mode4_shape_change(baseline_data, mode_batch, common, extra))
    return cases


def _roundtrip_case(
    baseline_data: Dict[str, Any],
    baseline_path: str,
    case: ConsistencyCase,
    output_dir: str,
    source_excel: str,
    source_sheet: str,
) -> Dict[str, Any]:
    materialize_expected_output(baseline_data, case)
    attach_provenance(
        case.input_data,
        source_pt=baseline_path,
        source_excel=source_excel,
        source_sheet=source_sheet,
        stage="derived_pt",
    )
    validate_schema(case.input_data)
    before_summary = build_input_summary(
        case.input_data,
        role="derived",
        mode=case.transform_meta.get("mode_name"),
        transform=case.transform_meta,
    )
    before_fingerprint = stable_json_digest(before_summary)
    before_content_fingerprint = input_content_digest(before_summary)
    derived_dir = os.path.join(output_dir, "derived")
    os.makedirs(derived_dir, exist_ok=True)
    path = os.path.join(derived_dir, f"{_safe_name(case.name)}.pt")
    torch.save(case.input_data, path)
    reloaded = torch.load(path, map_location="cpu")
    validate_schema(reloaded)
    expected_digest = TensorDigest.md5(case.input_data["cpu_output"])
    reloaded_digest = TensorDigest.md5(reloaded["cpu_output"])
    summary = build_input_summary(
        reloaded,
        role="derived",
        mode=case.transform_meta.get("mode_name"),
        transform=case.transform_meta,
    )
    after_fingerprint = stable_json_digest(summary)
    after_content_fingerprint = input_content_digest(summary)
    return {
        "case": case.name,
        "mode": case.transform_meta.get("mode_name"),
        "pt_path": os.path.abspath(path),
        "schema_valid": True,
        "cpu_output_md5_before_save": expected_digest,
        "cpu_output_md5_after_reload": reloaded_digest,
        "input_fingerprint_before_save": before_fingerprint,
        "input_fingerprint_after_reload": after_fingerprint,
        "input_content_fingerprint_before_save": before_content_fingerprint,
        "input_content_fingerprint_after_reload": after_content_fingerprint,
        "pt_roundtrip_equal": (
            expected_digest == reloaded_digest
            and before_fingerprint == after_fingerprint
            and before_content_fingerprint == after_content_fingerprint
        ),
        "semantic_oracle": case.transform_meta.get("semantic_oracle"),
        "expected_scope": reloaded.get("_consistency_expected_scope"),
        "input_summary": summary,
    }


def prepare_saved_baseline(
    baseline_path: str,
    output_dir: str,
    source_excel: str,
    source_sheet: str,
    modes: Sequence[str] = ("reorder", "split", "token-split", "shape-change"),
    **mode_options: Any,
) -> Dict[str, Any]:
    """Reload an original golden PT, derive modes 1-4, save/reload, and report."""
    baseline_data = torch.load(baseline_path, map_location="cpu")
    attach_provenance(
        baseline_data,
        source_pt=baseline_path,
        source_excel=source_excel,
        source_sheet=source_sheet,
        stage="baseline_pt_reload",
    )
    validate_schema(baseline_data)
    baseline_digest = TensorDigest.md5(baseline_data["cpu_output"])
    baseline_summary = build_input_summary(baseline_data, role="baseline")
    baseline_fingerprint = stable_json_digest(baseline_summary)
    baseline_content_fingerprint = input_content_digest(baseline_summary)
    baseline_id = make_baseline_id(baseline_data, baseline_content_fingerprint)
    cases = build_mode_cases(baseline_data, modes=modes, **mode_options)
    derived = []
    relations = []
    for case in cases:
        item = _roundtrip_case(
            baseline_data,
            baseline_path,
            case,
            output_dir,
            source_excel,
            source_sheet,
        )
        derived.append(item)
        relations.append(
            build_case_relation(
                baseline_data,
                baseline_id,
                case,
                item["pt_path"],
                item["input_content_fingerprint_after_reload"],
            )
        )

    relation_manifest = build_relation_manifest(
        baseline_data,
        os.path.abspath(baseline_path),
        baseline_content_fingerprint,
        baseline_digest,
        os.path.abspath(source_excel),
        source_sheet,
        relations,
    )
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "consistency_relation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(relation_manifest, file, ensure_ascii=False, indent=2, sort_keys=True)

    report = {
        "schema_version": 1,
        "scope": "CPU-only; no NPU result and no batch-consistency PASS claim",
        "source_excel": os.path.abspath(source_excel),
        "source_sheet": source_sheet,
        "baseline_pt": os.path.abspath(baseline_path),
        "baseline_schema_valid": True,
        "baseline_cpu_output_md5": baseline_digest,
        "baseline_input_fingerprint": baseline_fingerprint,
        "baseline_input_content_fingerprint": baseline_content_fingerprint,
        "baseline_input_summary": baseline_summary,
        "derived": derived,
        "relation_manifest": relation_manifest,
        "relation_manifest_path": os.path.abspath(manifest_path),
        "relation_manifest_sha256": stable_json_digest(relation_manifest),
        "pass": (
            bool(derived)
            and all(item["pt_roundtrip_equal"] for item in derived)
            and relation_manifest["precheck"]["status"] == "PASS"
        ),
    }
    report_path = os.path.join(output_dir, "consistency_prepare_report.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
    report["report_path"] = os.path.abspath(report_path)
    return report
