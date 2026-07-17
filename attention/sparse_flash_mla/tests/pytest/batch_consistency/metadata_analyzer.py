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

"""Conservative white-box analysis based on fields proven by metadata source."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any, Dict, Iterable, List, Tuple

from .metadata_layout import FaRecord, ParsedMetadata


class S2Category(str, Enum):
    EQUAL_128 = "s2_div_32_eq_128"
    LESS_THAN_128 = "s2_div_32_lt_128"
    GREATER_DIVISIBLE = "s2_div_32_gt_128_divisible"
    GREATER_REMAINDER = "s2_div_32_gt_128_remainder"


@dataclass(frozen=True)
class S2Analysis:
    s2: int
    category: S2Category
    reduction_block_size: int
    reduction_block_count: int


class MetadataCoverageAnalyzer:
    """Reports raw FA ranges and FD reduction tasks without guessing token expansion."""

    @staticmethod
    def classify_s2(s2: int) -> S2Analysis:
        if s2 <= 0:
            raise ValueError(f"S2 must be positive, got {s2}")
        quotient_ceil = (s2 + 31) // 32
        reduction_block = ((quotient_ceil + 127) // 128) * 128
        reduction_count = (s2 + reduction_block - 1) // reduction_block
        if s2 == 4096:
            category = S2Category.EQUAL_128
        elif s2 < 4096:
            category = S2Category.LESS_THAN_128
        elif s2 % 32 == 0:
            category = S2Category.GREATER_DIVISIBLE
        else:
            category = S2Category.GREATER_REMAINDER
        return S2Analysis(s2, category, reduction_block, reduction_count)

    @classmethod
    def analyze(cls, metadata: ParsedMetadata, s2_values: Iterable[int] = ()) -> Dict[str, Any]:
        enabled_fa = [record for record in metadata.fa_records if record.enabled]
        enabled_fd = [record for record in metadata.fd_records if record.enabled]
        fd_by_task: Dict[Tuple[int, int, int, int], List[Any]] = defaultdict(list)
        for record in enabled_fd:
            task_key = (
                record.bn2_index,
                record.m_index,
                record.workspace_index,
                record.workspace_num,
            )
            fd_by_task[task_key].append(record)

        level1 = Counter()
        level1_details = []
        for s2 in s2_values:
            analysis = cls.classify_s2(int(s2))
            level1[analysis.category.value] += 1
            level1_details.append(
                {
                    "s2": analysis.s2,
                    "category": analysis.category.value,
                    "reduction_block_size": analysis.reduction_block_size,
                    "reduction_block_count": analysis.reduction_block_count,
                }
            )

        endpoint_chain_valid = cls.validate_fa_endpoint_chain(enabled_fa)
        analyzer_errors = []
        if not endpoint_chain_valid:
            analyzer_errors.append("Enabled FA ranges do not form a continuous endpoint chain")
        analyzer_errors.extend(
            f"FD core {record.core_id} has workspace_num={record.workspace_num}; expected >= 2"
            for record in enabled_fd
            if record.workspace_num < 2
        )
        analyzer_errors.extend(
            f"FD core {record.core_id} has an empty M range"
            for record in enabled_fd
            if record.m_num == 0
        )

        task_summaries = [
            {
                "bn2_index": key[0],
                "m_index": key[1],
                "workspace_index": key[2],
                "workspace_num": key[3],
                "fd_core_ids": [record.core_id for record in records],
                "m_ranges": [[record.m_start, record.m_start + record.m_num] for record in records],
            }
            for key, records in sorted(fd_by_task.items())
        ]

        return {
            "schema_version": 1,
            "layout_version": metadata.layout_version,
            "fa_record_size": metadata.fa_record_size,
            "enabled_fa_cores": len(enabled_fa),
            "enabled_fd_cores": len(enabled_fd),
            "fa_ranges": [cls.fa_record_to_dict(record) for record in enabled_fa],
            "fd_reduction_tasks": task_summaries,
            "cross_core_reduction_task_count": len(fd_by_task),
            "fd_multi_m_core_count": sum(record.m_num > 1 for record in enabled_fd),
            "max_fd_workspace_split": max((record.workspace_num for record in enabled_fd), default=0),
            "fa_endpoint_chain_valid": endpoint_chain_valid,
            "unknown_fd_slot_nonzero": [
                record.core_id for record in metadata.fd_records if record.unknown_slot != 0
            ],
            "reserved_nonzero_count": sum(value != 0 for value in metadata.reserved),
            "level2_observed": {
                "no_cross_core_reduction": not enabled_fd,
                "has_cross_core_reduction": bool(enabled_fd),
                "has_fd_core_with_multiple_m_rows": any(record.m_num > 1 for record in enabled_fd),
            },
            "level1": dict(level1),
            "level1_details": level1_details,
            "errors": analyzer_errors,
            "analysis_limit": (
                "FD M is an s1g row rather than necessarily one query token. Complete token/core expansion "
                "requires B/N2/S1/group-size/S2 context; this report does not guess that mapping."
            ),
        }

    @staticmethod
    def fa_record_to_dict(record: FaRecord) -> Dict[str, Any]:
        result = asdict(record)
        result["start"] = list(record.start)
        result["end"] = list(record.end)
        return result

    @staticmethod
    def validate_fa_endpoint_chain(records: List[FaRecord]) -> bool:
        if not records:
            return True
        previous = records[0]
        for current in records[1:]:
            if current.start == previous.start and current.end == previous.end:
                previous = current
                continue
            if current.start != previous.end:
                return False
            previous = current
        return True

    @staticmethod
    def dumps(report: Dict[str, Any]) -> str:
        return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
