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
from bisect import bisect_right
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

from .consistency_adapter import CaseAdapter
from .consistency_schema import get_q_lengths
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


@dataclass(frozen=True)
class CoverageContext:
    """Runtime shape context required to expand metadata cursors into tokens."""

    batch_size: int
    num_heads_q: int
    num_heads_kv: int
    q_lengths: Tuple[int, ...]
    ori_kv_lengths: Tuple[int, ...]
    cmp_kv_lengths: Tuple[int, ...]
    cmp_residual: Tuple[int, ...]
    ori_mask_mode: int
    cmp_mask_mode: int
    ori_win_left: int
    ori_win_right: int
    cmp_ratio: int
    ori_topk: int
    cmp_topk: int
    has_ori_kv: bool
    has_cmp_kv: bool
    s2_base_size: int = 128
    batch_consistency: bool = True

    @property
    def group_size(self) -> int:
        return self.num_heads_q // self.num_heads_kv

    @classmethod
    def from_input_data(cls, input_data: Dict[str, Any]) -> "CoverageContext":
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        metadata = adapter.metadata
        batch_size = adapter.get_batch_size()

        def lengths(prefix: str) -> Tuple[int, ...]:
            seqused = tensors.get(f"seqused_{prefix}_kv")
            if seqused is not None:
                return tuple(int(value) for value in seqused.to("cpu", torch.int64).tolist())
            kv = tensors.get(f"{prefix}_kv")
            if kv is None:
                return (0,) * batch_size
            if adapter.get_layout_kv() == "BSND":
                return (int(kv.shape[1]),) * batch_size
            if adapter.get_layout_kv() == "TND":
                cu = tensors[f"cu_seqlens_{prefix}_kv"].to("cpu", torch.int64).tolist()
                return tuple(cu[index + 1] - cu[index] for index in range(batch_size))
            raise ValueError(f"PA_BBND {prefix}_kv requires seqused_{prefix}_kv")

        residual_tensor = tensors.get("cmp_residual_kv")
        residual = (
            tuple(int(value) for value in residual_tensor.to("cpu", torch.int64).tolist())
            if residual_tensor is not None
            else (0,) * batch_size
        )
        num_heads_q = int(adapter.get_metadata("N1"))
        num_heads_kv = int(adapter.get_metadata("N2"))
        if num_heads_kv <= 0 or num_heads_q % num_heads_kv != 0:
            raise ValueError(f"N1={num_heads_q} must be divisible by N2={num_heads_kv}")
        return cls(
            batch_size=batch_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            q_lengths=tuple(get_q_lengths(input_data, valid_only=True)),
            ori_kv_lengths=lengths("ori"),
            cmp_kv_lengths=lengths("cmp"),
            cmp_residual=residual,
            ori_mask_mode=int(tensors.get("ori_mask_mode", metadata.get("ori_mask_mode", 0))),
            cmp_mask_mode=int(tensors.get("cmp_mask_mode", metadata.get("cmp_mask_mode", 0))),
            ori_win_left=int(tensors.get("ori_win_left", metadata.get("ori_win_left", -1))),
            ori_win_right=int(tensors.get("ori_win_right", metadata.get("ori_win_right", 0))),
            cmp_ratio=int(tensors.get("cmp_ratio") or metadata.get("cmp_ratio") or 1),
            ori_topk=(
                int(metadata.get("ori_topk") or metadata.get("K") or 0)
                if tensors.get("ori_sparse_indices") is not None
                else 0
            ),
            cmp_topk=(
                int(metadata.get("cmp_topk") or metadata.get("topk") or metadata.get("K") or 0)
                if tensors.get("cmp_sparse_indices") is not None
                else 0
            ),
            has_ori_kv=tensors.get("ori_kv") is not None,
            has_cmp_kv=tensors.get("cmp_kv") is not None,
        )

    @staticmethod
    def _clip_window(start: int, end: int, length: int) -> Tuple[int, int]:
        return max(0, min(start, length)), max(0, min(end, length))

    def token_windows(self, batch: int, token: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        q_length = self.q_lengths[batch]
        ori_length = self.ori_kv_lengths[batch]
        if not self.has_ori_kv:
            ori_window = (0, 0)
        elif self.ori_mask_mode == 0:
            ori_window = (0, ori_length)
        elif self.ori_mask_mode in (3, 4):
            threshold = ori_length - q_length + token + 1
            start = (
                0
                if self.ori_mask_mode == 3 or self.ori_win_left == -1
                else threshold - self.ori_win_left - 1
            )
            ori_window = self._clip_window(start, threshold + self.ori_win_right, ori_length)
        else:
            raise ValueError(f"Unsupported ori_mask_mode={self.ori_mask_mode}")

        if not self.has_cmp_kv:
            return ori_window, (0, 0)
        cmp_length = self.cmp_kv_lengths[batch]
        reverted = cmp_length * self.cmp_ratio + self.cmp_residual[batch]
        if self.cmp_mask_mode == 0:
            cmp_end = reverted
        elif self.cmp_mask_mode == 3:
            cmp_end = reverted - q_length + token + 1
        else:
            raise ValueError(f"Unsupported cmp_mask_mode={self.cmp_mask_mode}")
        compressed_end = min(cmp_length, max(cmp_end, 0) // self.cmp_ratio)
        return ori_window, (0, compressed_end)

    def visible_s2_lengths(self, batch: int, token: int) -> Tuple[int, int]:
        ori_window, cmp_window = self.token_windows(batch, token)
        ori_tokens = max(ori_window[1] - ori_window[0], 0)
        cmp_tokens = max(cmp_window[1] - cmp_window[0], 0)
        if self.ori_topk:
            ori_tokens = min(ori_tokens, self.ori_topk)
        if self.cmp_topk:
            cmp_tokens = min(cmp_tokens, self.cmp_topk)
        return ori_tokens, cmp_tokens

    def reduction_tile_size(self, batch: int, token: int) -> int:
        ori_tokens, cmp_tokens = self.visible_s2_lengths(batch, token)
        total = ori_tokens + cmp_tokens
        if not self.batch_consistency or total == 0:
            return self.s2_base_size
        quotient = (total + 31) // 32
        return (quotient + self.s2_base_size - 1) // self.s2_base_size * self.s2_base_size

    def s2_block_count(self, batch: int, token: int) -> int:
        ori_tokens, cmp_tokens = self.visible_s2_lengths(batch, token)
        tile_size = self.reduction_tile_size(batch, token)
        return (
            (ori_tokens + tile_size - 1) // tile_size
            + (cmp_tokens + tile_size - 1) // tile_size
        )

    def level1_s2_values(self) -> Tuple[int, ...]:
        values = []
        for batch, q_length in enumerate(self.q_lengths):
            for token in range(q_length):
                total = sum(self.visible_s2_lengths(batch, token))
                if total > 0:
                    values.append(total)
        return tuple(values)


class MetadataCoverageAnalyzer:
    """Analyze raw metadata conservatively, or fully expand it with shape context."""

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
    def analyze(
        cls,
        metadata: ParsedMetadata,
        s2_values: Iterable[int] = (),
        context: Optional[CoverageContext] = None,
    ) -> Dict[str, Any]:
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

        s2_values = tuple(s2_values)
        if context is not None and not s2_values:
            s2_values = context.level1_s2_values()
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

        report = {
            "schema_version": 2,
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
            "analysis_limit": None,
        }
        if context is None:
            report["analysis_limit"] = (
                "Complete token/core expansion requires B/N2/Q/KV/mask context. "
                "Pass CoverageContext.from_input_data(case) to enable it."
            )
            return report

        expanded = cls.expand_level2(metadata, context)
        expansion_errors = expanded.pop("expansion_errors", [])
        report.update(expanded)
        report["errors"].extend(expansion_errors)
        level1_targets = {category.value: level1.get(category.value, 0) > 0 for category in S2Category}
        report["level1_targets"] = level1_targets
        report["level1_missing"] = [name for name, covered in level1_targets.items() if not covered]
        return report

    @classmethod
    def expand_level2(
        cls, metadata: ParsedMetadata, context: CoverageContext
    ) -> Dict[str, Any]:
        """Expand FA cursor intervals and cross-check FD workspace tasks."""
        rows: List[Tuple[int, int, int, int, int]] = []
        row_starts: List[int] = []
        row_ends: List[int] = []
        offset = 0
        for batch, q_length in enumerate(context.q_lengths):
            for n2 in range(context.num_heads_kv):
                bn2 = batch * context.num_heads_kv + n2
                for token in range(q_length):
                    block_count = context.s2_block_count(batch, token)
                    row_starts.append(offset)
                    offset += block_count
                    row_ends.append(offset)
                    rows.append((bn2, token, batch, n2, block_count))
        total_blocks = offset
        row_index = {(bn2, token): index for index, (bn2, token, _, _, _) in enumerate(rows)}

        def cursor_offset(cursor: Tuple[int, int, int]) -> int:
            bn2, token, s2_block = cursor
            total_bn2 = context.batch_size * context.num_heads_kv
            if bn2 == total_bn2 and token == 0 and s2_block == 0:
                return total_blocks
            index = row_index.get((bn2, token))
            if index is None:
                raise ValueError(f"metadata cursor {cursor} is outside the supplied shape context")
            block_count = rows[index][4]
            if s2_block < 0 or s2_block > block_count:
                raise ValueError(
                    f"metadata cursor {cursor} exceeds token block count {block_count}"
                )
            return row_starts[index] + s2_block

        range_groups: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], List[FaRecord]] = defaultdict(list)
        for record in metadata.fa_records:
            if record.enabled:
                range_groups[(record.start, record.end)].append(record)

        token_cores: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        token_segments: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        core_tokens: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        expansion_errors: List[str] = []
        logical_ranges = []
        for logical_core, ((start, end), records) in enumerate(range_groups.items()):
            try:
                start_offset = cursor_offset(start)
                end_offset = cursor_offset(end)
            except ValueError as error:
                expansion_errors.append(str(error))
                continue
            if end_offset < start_offset:
                expansion_errors.append(f"FA range {start}->{end} is reversed")
                continue
            logical_ranges.append(
                {
                    "logical_core": logical_core,
                    "physical_core_ids": [record.core_id for record in records],
                    "start": list(start),
                    "end": list(end),
                    "block_count": end_offset - start_offset,
                }
            )
            if end_offset == start_offset:
                continue
            first_row = bisect_right(row_ends, start_offset)
            for index in range(first_row, len(rows)):
                if row_starts[index] >= end_offset:
                    break
                bn2, token, batch, n2, block_count = rows[index]
                if block_count == 0:
                    continue
                segment_start = max(start_offset, row_starts[index]) - row_starts[index]
                segment_end = min(end_offset, row_ends[index]) - row_starts[index]
                if segment_start >= segment_end:
                    continue
                key = (bn2, token)
                token_cores[key].add(logical_core)
                core_tokens[logical_core].add(key)
                token_segments[key].append(
                    {
                        "logical_core": logical_core,
                        "physical_core_ids": [record.core_id for record in records],
                        "s2_block_range": [segment_start, segment_end],
                        "s2_token_range": [
                            segment_start * context.reduction_tile_size(batch, token),
                            segment_end * context.reduction_tile_size(batch, token),
                        ],
                        "batch": batch,
                        "n2": n2,
                    }
                )

        single_tokens = {token for token, cores in token_cores.items() if len(cores) == 1}
        cross_tokens = {token for token, cores in token_cores.items() if len(cores) > 1}
        fd_by_token: Dict[Tuple[int, int], List[Any]] = defaultdict(list)
        for record in metadata.fd_records:
            if record.enabled:
                fd_by_token[(record.bn2_index, record.m_index)].append(record)

        for token in sorted(cross_tokens):
            records = fd_by_token.get(token, [])
            if not records:
                expansion_errors.append(f"Cross-core token {token} has no enabled FD record")
                continue
            bn2, query_token = token
            batch = bn2 // context.num_heads_kv
            expected_splits = context.s2_block_count(batch, query_token)
            if any(record.workspace_num != expected_splits for record in records):
                actual = sorted({record.workspace_num for record in records})
                expansion_errors.append(
                    f"FD token {token} workspace_num {actual} != FA split count {expected_splits}"
                )
            intervals = sorted((record.m_start, record.m_start + record.m_num) for record in records)
            cursor = 0
            for start, end in intervals:
                if start != cursor or end < start:
                    expansion_errors.append(
                        f"FD token {token} M ranges {intervals} do not form [0,{context.group_size})"
                    )
                    break
                cursor = end
            if cursor != context.group_size:
                expansion_errors.append(
                    f"FD token {token} M coverage ends at {cursor}, expected G={context.group_size}"
                )

        unexpected_fd = sorted(set(fd_by_token) - cross_tokens)
        if unexpected_fd:
            expansion_errors.append(f"FD records exist for non-cross-core tokens: {unexpected_fd[:8]}")

        core_perspective = []
        level2_targets = {
            "all_tokens_single_core": bool(token_cores) and not cross_tokens,
            "has_cross_core_token": bool(cross_tokens),
            "core_one_single_only": False,
            "core_one_cross_only": False,
            "core_multiple_cross": False,
            "core_single_plus_one_cross": False,
            "core_single_plus_multiple_cross": False,
        }
        for core, tokens in sorted(core_tokens.items()):
            singles = sorted(tokens & single_tokens)
            crosses = sorted(tokens & cross_tokens)
            level2_targets["core_one_single_only"] |= len(singles) == 1 and not crosses
            level2_targets["core_one_cross_only"] |= len(crosses) == 1 and not singles
            level2_targets["core_multiple_cross"] |= len(crosses) >= 2
            level2_targets["core_single_plus_one_cross"] |= bool(singles) and len(crosses) == 1
            level2_targets["core_single_plus_multiple_cross"] |= bool(singles) and len(crosses) >= 2
            core_perspective.append(
                {
                    "logical_core": core,
                    "single_core_tokens": [list(token) for token in singles],
                    "cross_core_tokens": [list(token) for token in crosses],
                }
            )

        token_details = []
        for (bn2, token), cores in sorted(token_cores.items())[:256]:
            token_details.append(
                {
                    "bn2": bn2,
                    "batch": bn2 // context.num_heads_kv,
                    "n2": bn2 % context.num_heads_kv,
                    "query_token": token,
                    "fa_logical_cores": sorted(cores),
                    "cross_core": len(cores) > 1,
                    "segments": token_segments[(bn2, token)],
                }
            )

        return {
            "context": asdict(context),
            "fa_logical_ranges": logical_ranges,
            "assigned_token_count": len(token_cores),
            "single_core_token_count": len(single_tokens),
            "cross_core_token_count": len(cross_tokens),
            "token_assignments": token_details,
            "token_assignments_truncated": len(token_cores) > len(token_details),
            "core_perspective": core_perspective,
            "level2_targets": level2_targets,
            "level2_missing": [name for name, covered in level2_targets.items() if not covered],
            "expansion_errors": expansion_errors,
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
