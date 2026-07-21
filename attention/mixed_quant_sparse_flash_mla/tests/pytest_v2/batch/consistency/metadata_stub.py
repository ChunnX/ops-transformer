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

"""Plan stub candidates and guard an optional, default-off metadata transform."""

from typing import Any, Callable, Dict, List, Optional

import torch

from .metadata_coverage import CoverageContext, MetadataCoverageAnalyzer
from .metadata_layout import MetadataLayout
from .reporting import summarize_tensor


SCHEMA_VERSION = 1
PLAN_KIND = "smla_mqsmla_metadata_stub_candidate_plan"
SAFETY_BLOCKERS = (
    "SUPPORT_FD_DISABLED_IN_CHECKED_SOURCE",
    "INSTALLED_PACKAGE_BUILD_ID_UNVERIFIED",
    "FD_RECORD_SLOT7_CONTRACT_UNVERIFIED",
    "FD_WORKSPACE_CONTRACT_UNVERIFIED",
    "REDUCTION_TILE_FORMULA_UNRESOLVED",
    "DETERMINISTIC_LEVEL3_RUNTIME_UNVERIFIED",
)


class MetadataStubPlanError(ValueError):
    """Raised when natural metadata cannot be safely analyzed for candidate selection."""


MetadataTransform = Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]


def apply_metadata_transform(
    metadata: torch.Tensor,
    input_data: Dict[str, Any],
    transform: Optional[MetadataTransform],
) -> torch.Tensor:
    """Apply an explicitly supplied hook while preserving the natural tensor by default."""
    if transform is None:
        return metadata
    transformed = transform(metadata.clone(), input_data)
    if not isinstance(transformed, torch.Tensor):
        raise MetadataStubPlanError("metadata transform must return a torch.Tensor")
    if transformed.shape != metadata.shape:
        raise MetadataStubPlanError(
            f"metadata transform changed shape {tuple(metadata.shape)} -> "
            f"{tuple(transformed.shape)}"
        )
    if transformed.dtype != metadata.dtype:
        raise MetadataStubPlanError(
            f"metadata transform changed dtype {metadata.dtype} -> {transformed.dtype}"
        )
    if transformed.device != metadata.device:
        raise MetadataStubPlanError(
            f"metadata transform changed device {metadata.device} -> {transformed.device}"
        )
    return transformed.contiguous()


def _covers_complete_token(segments: List[Dict[str, Any]], block_count: int) -> bool:
    ranges = sorted(tuple(segment["s2_block_range"]) for segment in segments)
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            return False
        cursor = end
    return cursor == block_count


def plan_single_to_multi_core_stub(
    metadata: torch.Tensor,
    input_data: Dict[str, Any],
    *,
    max_candidates: int = 16,
) -> Dict[str, Any]:
    """Find auditable natural candidates without constructing synthetic metadata."""
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise MetadataStubPlanError("max_candidates must be an integer")
    if max_candidates < 1:
        raise MetadataStubPlanError("max_candidates must be >= 1")

    parsed = MetadataLayout().parse(metadata)
    context = CoverageContext.from_input_data(input_data)
    coverage = MetadataCoverageAnalyzer.analyze(parsed, context=context)
    tensor_summary = summarize_tensor("kernel_metadata", metadata)
    natural_identity = {
        key: tensor_summary[key]
        for key in ("shape", "dtype", "numel", "byte_size", "sha256")
    }

    candidates = []
    for assignment in coverage.get("token_assignments", []):
        if assignment["cross_core"] or len(assignment["fa_logical_cores"]) != 1:
            continue
        batch = int(assignment["batch"])
        query_token = int(assignment["query_token"])
        block_count = context.s2_block_count(batch, query_token)
        if block_count < 2 or not _covers_complete_token(
            assignment["segments"], block_count
        ):
            continue
        split_at = block_count // 2
        formula = context.reduction_formula(batch, query_token)
        candidates.append(
            {
                "bn2": int(assignment["bn2"]),
                "batch": batch,
                "n2": int(assignment["n2"]),
                "query_token": query_token,
                "natural_fa_logical_core": assignment["fa_logical_cores"][0],
                "natural_segments": assignment["segments"],
                "s2_block_count": block_count,
                "proposed_two_core_s2_block_ranges": [
                    [0, split_at],
                    [split_at, block_count],
                ],
                "visible_s2": {
                    "ori": formula["visible_ori_s2"],
                    "cmp": formula["visible_cmp_s2"],
                    "total": formula["visible_total_s2"],
                },
                "formula_audit": {
                    "pr8737_floor_tile": formula["pr8737_floor_tile"],
                    "requirement_ceil_tile": formula["requirement_ceil_tile"],
                    "quotient_formula_conflict": formula[
                        "quotient_formula_conflict"
                    ],
                    "tile_formula_conflict": formula["tile_formula_conflict"],
                },
            }
        )
        if len(candidates) == max_candidates:
            break

    errors = list(coverage.get("errors", []))
    if errors:
        status = "INVALID_NATURAL_METADATA"
    elif candidates:
        status = "CANDIDATE_FOUND_BLOCKED"
    else:
        status = "NO_CANDIDATE"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "claim_scope": (
            "CPU-only candidate analysis; no metadata mutation, NPU execution, "
            "or natural metadata coverage claim"
        ),
        "status": status,
        "natural_metadata": natural_identity,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "analysis_errors": errors,
        "analysis_truncated": bool(coverage.get("token_assignments_truncated")),
        "mutation": {
            "generated": False,
            "applied": False,
            "allowed": False,
            "synthetic_metadata_must_not_count_as_natural_coverage": True,
        },
        "activation_blockers": list(SAFETY_BLOCKERS),
    }
