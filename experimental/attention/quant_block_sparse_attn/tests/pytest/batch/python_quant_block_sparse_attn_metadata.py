#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Pure-Python QBSA metadata scheduler for environments without AICPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


_ACTIVE_AIC_CORE_NUM = 64
_AIC_CORE_SLOTS = 72
_AIV_CORE_SLOTS = 72
_HEAD_METADATA_SIZE = 8
_CORE_METADATA_SIZE = 8
_FD_METADATA_SIZE = 8
_L2_BYTES = 120 * 1024 * 1024
_BLOCK_TOLERANCE_RATIO = 2

_CORE_ENABLE_INDEX = 0
_BN1_START_INDEX = 1
_S1_START_INDEX = 2
_S2_START_INDEX = 3
_BN1_END_INDEX = 4
_S1_END_INDEX = 5
_S2_END_INDEX = 6
_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7

_FP8_QUANT_MODE = 1
_MXFP8_QUANT_MODE = 2


@dataclass
class _CoreCache:
    block_limit: int = 0
    block: int = 0


@dataclass
class _AssignContext:
    cur_core_idx: int = 0
    cur_b_idx: int = 0
    cur_n1_idx: int = 0
    cur_bn1_idx: int = 0
    cur_s1_idx: int = 0
    cur_s2_idx: int = 0
    bn1_limit: int = 0
    unassigned_block: int = 0
    bn1_block: int = 0
    core_cache: _CoreCache = field(default_factory=_CoreCache)
    is_finished: bool = False


@dataclass
class _SplitResult:
    used_core_num: int
    max_block: int
    bn1_end: List[int]
    s1_end: List[int]
    s2_end: List[int]
    first_fd_data_workspace_idx: List[int]

    @classmethod
    def create(cls) -> "_SplitResult":
        return cls(
            used_core_num=0,
            max_block=0,
            bn1_end=[0] * _ACTIVE_AIC_CORE_NUM,
            s1_end=[0] * _ACTIVE_AIC_CORE_NUM,
            s2_end=[0] * _ACTIVE_AIC_CORE_NUM,
            first_fd_data_workspace_idx=[0] * _ACTIVE_AIC_CORE_NUM,
        )


@dataclass
class _SectionInfo:
    bn1_start: int
    bn1_end: int
    block_num: int
    cost: int
    split_result: _SplitResult = field(default_factory=_SplitResult.create)


class _MetadataScheduler:
    """Direct translation of QuantBlockSparseAttnMetadataCpuKernel scheduling."""

    def __init__(
        self,
        sparse_seq_len_data: List,
        sparse_seq_len_dim_num: int,
        batch_size: int,
        num_heads_q: int,
        head_dim: int,
        sparse_block_size_q: int,
        sparse_block_size_k: int,
    ) -> None:
        self.sparse_seq_len_data = sparse_seq_len_data
        self.sparse_seq_len_dim_num = sparse_seq_len_dim_num
        self.batch_size = batch_size
        self.num_heads_q = num_heads_q
        self.head_dim = head_dim
        self.sparse_block_size_q = max(sparse_block_size_q, 1)
        self.sparse_block_size_k = max(sparse_block_size_k, 1)
        if sparse_seq_len_dim_num == 3:
            self.qb_max = len(sparse_seq_len_data[0][0])
        else:
            self.qb_max = len(sparse_seq_len_data[0])
        self.bn1_block_num: List[int] = []
        self.bn1_last_row_block_num: List[int] = []
        self._init_bn1_block_info()

    def _get_row_block_num(self, b_idx: int, n1_idx: int, s1_idx: int) -> int:
        if (
            b_idx < 0
            or b_idx >= self.batch_size
            or n1_idx < 0
            or n1_idx >= self.num_heads_q
            or s1_idx < 0
            or s1_idx >= self.qb_max
        ):
            return 0
        if self.sparse_seq_len_dim_num == 3:
            value = self.sparse_seq_len_data[b_idx][n1_idx][s1_idx]
        else:
            value = self.sparse_seq_len_data[b_idx][s1_idx]
        return int(value) if int(value) > 0 else 0

    def _init_bn1_block_info(self) -> None:
        total_bn1 = self.batch_size * self.num_heads_q
        self.bn1_block_num = [0] * total_bn1
        self.bn1_last_row_block_num = [0] * total_bn1
        for b_idx in range(self.batch_size):
            for n1_idx in range(self.num_heads_q):
                bn1_idx = b_idx * self.num_heads_q + n1_idx
                total_block = 0
                last_row_block = 0
                for s1_idx in range(self.qb_max):
                    row_block = self._get_row_block_num(b_idx, n1_idx, s1_idx)
                    total_block += row_block
                    if row_block > 0:
                        last_row_block = row_block
                self.bn1_block_num[bn1_idx] = total_block
                self.bn1_last_row_block_num[bn1_idx] = last_row_block

    def _get_bn1_block_num(self, b_idx: int, n1_idx: int) -> int:
        bn1_idx = b_idx * self.num_heads_q + n1_idx
        if bn1_idx < 0 or bn1_idx >= len(self.bn1_block_num):
            return 0
        return self.bn1_block_num[bn1_idx]

    def _calc_valid_s1_rows(self, b_idx: int, n1_idx: int) -> Tuple[int, int]:
        valid_rows = 0
        max_s2_block_num = 0
        for s1_idx in range(self.qb_max):
            row_block = self._get_row_block_num(b_idx, n1_idx, s1_idx)
            if row_block > 0:
                valid_rows += 1
                max_s2_block_num = max(max_s2_block_num, row_block)
        return valid_rows, max_s2_block_num

    def _calc_bn1_cost(self, b_idx: int, n1_idx: int) -> int:
        valid_s1_rows, max_s2_block_num = self._calc_valid_s1_rows(b_idx, n1_idx)
        if valid_s1_rows == 0:
            return 0
        s1_cost = valid_s1_rows * self.sparse_block_size_q * self.head_dim * 2
        s2_cost = max_s2_block_num * self.sparse_block_size_k * self.head_dim * 2
        return s1_cost + s2_cost

    def _calc_section_boundaries(self) -> List[_SectionInfo]:
        section_results: List[_SectionInfo] = []
        total_bn1 = self.batch_size * self.num_heads_q
        section_start = 0
        section_cost = 0
        section_block_num = 0

        for bn1_idx in range(total_bn1):
            b_idx = bn1_idx // self.num_heads_q
            n1_idx = bn1_idx % self.num_heads_q
            current_cost = self._calc_bn1_cost(b_idx, n1_idx)
            current_blocks = self._get_bn1_block_num(b_idx, n1_idx)
            if section_cost != 0 and section_cost + current_cost > _L2_BYTES:
                section_results.append(
                    _SectionInfo(
                        bn1_start=section_start,
                        bn1_end=bn1_idx,
                        cost=section_cost,
                        block_num=section_block_num,
                    )
                )
                section_start = bn1_idx
                section_cost = 0
                section_block_num = 0
            section_cost += current_cost
            section_block_num += current_blocks

        if section_block_num != 0 or not section_results:
            section_results.append(
                _SectionInfo(
                    bn1_start=section_start,
                    bn1_end=total_bn1,
                    cost=section_cost,
                    block_num=section_block_num,
                )
            )
        return section_results

    def _advance_to_valid_row(self, context: _AssignContext) -> bool:
        if context.is_finished:
            return False
        while context.cur_bn1_idx < context.bn1_limit:
            context.cur_b_idx = context.cur_bn1_idx // self.num_heads_q
            context.cur_n1_idx = context.cur_bn1_idx % self.num_heads_q
            if context.cur_s1_idx == 0 and context.bn1_block == 0:
                context.bn1_block = self._get_bn1_block_num(context.cur_b_idx, context.cur_n1_idx)
            while (
                context.cur_s1_idx < self.qb_max
                and self._get_row_block_num(context.cur_b_idx, context.cur_n1_idx, context.cur_s1_idx) == 0
            ):
                context.cur_s1_idx += 1
            if context.cur_s1_idx < self.qb_max:
                return True
            context.cur_bn1_idx += 1
            context.cur_n1_idx = 0
            context.cur_s1_idx = 0
            context.cur_s2_idx = 0
            context.bn1_block = 0
        context.is_finished = True
        return False

    def _assign_by_batch(self, context: _AssignContext) -> None:
        if context.is_finished:
            return
        while context.cur_bn1_idx < context.bn1_limit:
            if not self._advance_to_valid_row(context):
                return
            if context.bn1_block == 0:
                context.cur_bn1_idx += 1
                context.cur_n1_idx = 0
                context.cur_s1_idx = 0
                context.cur_s2_idx = 0
                continue
            tolerance = 0
            if context.cur_bn1_idx < len(self.bn1_last_row_block_num):
                tolerance = self.bn1_last_row_block_num[context.cur_bn1_idx] // _BLOCK_TOLERANCE_RATIO
            if context.core_cache.block + context.bn1_block > context.core_cache.block_limit + tolerance:
                return
            context.core_cache.block += context.bn1_block
            context.cur_bn1_idx += 1
            context.cur_n1_idx = 0
            context.cur_s1_idx = 0
            context.cur_s2_idx = 0
            context.bn1_block = 0
        context.is_finished = True

    def _assign_by_row(self, context: _AssignContext) -> None:
        while not context.is_finished:
            if not self._advance_to_valid_row(context):
                break
            row_block = self._get_row_block_num(context.cur_b_idx, context.cur_n1_idx, context.cur_s1_idx)
            tolerance = row_block // _BLOCK_TOLERANCE_RATIO
            if (
                context.core_cache.block > 0
                and context.core_cache.block + row_block > context.core_cache.block_limit + tolerance
            ):
                return
            context.core_cache.block += row_block
            context.bn1_block = max(context.bn1_block - row_block, 0)
            context.cur_s1_idx += 1

    def _schedule_fa(self, bn1_start: int, bn1_end: int, block_num: int, result: _SplitResult) -> None:
        context = _AssignContext(
            unassigned_block=block_num,
            cur_bn1_idx=bn1_start,
            cur_b_idx=bn1_start // self.num_heads_q,
            cur_n1_idx=bn1_start % self.num_heads_q,
            bn1_limit=bn1_end,
        )
        for core_idx in range(_ACTIVE_AIC_CORE_NUM):
            if context.is_finished or context.unassigned_block == 0:
                break
            if not self._advance_to_valid_row(context):
                break
            context.cur_core_idx = core_idx
            context.core_cache = _CoreCache()
            remaining_core = _ACTIVE_AIC_CORE_NUM - core_idx
            context.core_cache.block_limit = (
                context.unassigned_block + remaining_core - 1
            ) // remaining_core
            current_row_block = self._get_row_block_num(
                context.cur_b_idx, context.cur_n1_idx, context.cur_s1_idx
            )
            context.core_cache.block_limit = max(context.core_cache.block_limit, current_row_block)

            self._assign_by_batch(context)
            self._assign_by_row(context)

            result.bn1_end[core_idx] = context.cur_bn1_idx
            result.s1_end[core_idx] = context.cur_s1_idx
            result.s2_end[core_idx] = 0
            result.first_fd_data_workspace_idx[core_idx] = 0
            result.max_block = max(result.max_block, context.core_cache.block)
            context.unassigned_block = max(context.unassigned_block - context.core_cache.block, 0)
            result.used_core_num = core_idx + 1
        result.used_core_num = max(result.used_core_num, 1)

    def _schedule_sections(self) -> List[_SectionInfo]:
        section_results = self._calc_section_boundaries()
        if not section_results:
            empty_section = _SectionInfo(
                bn1_start=0,
                bn1_end=self.batch_size * self.num_heads_q,
                block_num=0,
                cost=0,
            )
            empty_section.split_result.used_core_num = 1
            empty_section.split_result.bn1_end[0] = empty_section.bn1_end
            return [empty_section]
        for section in section_results:
            if section.block_num == 0:
                section.split_result.used_core_num = 1
                section.split_result.bn1_end[0] = section.bn1_end
                section.split_result.s1_end[0] = 0
                section.split_result.s2_end[0] = 0
                section.split_result.first_fd_data_workspace_idx[0] = 0
                continue
            self._schedule_fa(
                section.bn1_start,
                section.bn1_end,
                section.block_num,
                section.split_result,
            )
        return section_results

    def _generate_metadata(self, section_results: List[_SectionInfo]) -> List[int]:
        metadata_size = (
            _HEAD_METADATA_SIZE
            + self.batch_size * self.num_heads_q * _AIC_CORE_SLOTS * _CORE_METADATA_SIZE
            + _AIV_CORE_SLOTS * _FD_METADATA_SIZE
        )
        metadata = [0] * metadata_size
        metadata[0] = len(section_results)
        for section_idx, section in enumerate(section_results):
            result = section.split_result
            for core_idx in range(result.used_core_num):
                base = (
                    _HEAD_METADATA_SIZE
                    + section_idx * _AIC_CORE_SLOTS * _CORE_METADATA_SIZE
                    + core_idx * _CORE_METADATA_SIZE
                )
                bn1_start = section.bn1_start if core_idx == 0 else result.bn1_end[core_idx - 1]
                s1_start = 0 if core_idx == 0 else result.s1_end[core_idx - 1]
                metadata[base + _CORE_ENABLE_INDEX] = 1
                metadata[base + _BN1_START_INDEX] = bn1_start
                metadata[base + _S1_START_INDEX] = s1_start
                metadata[base + _S2_START_INDEX] = 0
                metadata[base + _BN1_END_INDEX] = result.bn1_end[core_idx]
                metadata[base + _S1_END_INDEX] = result.s1_end[core_idx]
                metadata[base + _S2_END_INDEX] = result.s2_end[core_idx]
                metadata[base + _FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = (
                    result.first_fd_data_workspace_idx[core_idx]
                )
        return metadata

    def generate(self) -> List[int]:
        return self._generate_metadata(self._schedule_sections())


def _validate_arguments(
    sparse_seq_len: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    batch_size: int,
    sparse_block_size_q: int,
    sparse_block_size_k: int,
    quant_mode: int,
    mask_mode: int,
    layout_sparse_indices: str,
) -> int:
    if not isinstance(sparse_seq_len, torch.Tensor):
        raise TypeError("sparse_seq_len must be a torch.Tensor")
    if sparse_seq_len.dtype != torch.int32:
        raise ValueError("sparse_seq_len dtype must be torch.int32")
    if sparse_seq_len.dim() not in (2, 3):
        raise ValueError("sparse_seq_len must be a 2D or 3D tensor")
    if sparse_seq_len.device.type == "meta":
        raise ValueError("Python metadata backend requires materialized sparse_seq_len data")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be greater than 0, but got {batch_size}")
    if sparse_seq_len.shape[0] != batch_size:
        raise ValueError(
            "sparse_seq_len shape[0] must equal batch_size, "
            f"but got {sparse_seq_len.shape[0]} and {batch_size}"
        )
    if num_heads_q <= 0:
        raise ValueError(f"num_heads_q must be greater than 0, but got {num_heads_q}")
    normalized_num_heads_kv = num_heads_q if num_heads_kv == 0 else num_heads_kv
    if normalized_num_heads_kv <= 0 or num_heads_q % normalized_num_heads_kv != 0:
        raise ValueError(
            "num_heads_q must be divisible by num_heads_kv, "
            f"but got num_heads_q={num_heads_q}, num_heads_kv={num_heads_kv}"
        )
    if head_dim < 0:
        raise ValueError(f"head_dim must be nonnegative, but got {head_dim}")
    if sparse_seq_len.shape[-1] <= 0:
        raise ValueError("sparse_seq_len Qb dimension must be greater than 0")
    if sparse_seq_len.dim() == 3 and sparse_seq_len.shape[1] != num_heads_q:
        raise ValueError(
            "3D sparse_seq_len shape[1] must equal num_heads_q, "
            f"but got {sparse_seq_len.shape[1]} and {num_heads_q}"
        )
    if quant_mode not in (_FP8_QUANT_MODE, _MXFP8_QUANT_MODE):
        raise ValueError(f"quant_mode must be 1 or 2, but got {quant_mode}")
    if mask_mode not in (0, 3):
        raise ValueError(f"mask_mode must be 0 or 3, but got {mask_mode}")
    if quant_mode == _FP8_QUANT_MODE:
        if sparse_block_size_q != 128 or sparse_block_size_k != 128:
            raise ValueError("FP8 sparse block sizes must be 128/128")
    elif sparse_block_size_q not in (64, 128) or sparse_block_size_k not in (64, 128):
        raise ValueError("MXFP8 sparse block sizes must each be 64 or 128")
    if layout_sparse_indices != "B_N_Qb_Kb":
        raise ValueError(
            "layout_sparse_indices must be 'B_N_Qb_Kb', "
            f"but got {layout_sparse_indices!r}"
        )
    return normalized_num_heads_kv


def npu_quant_block_sparse_attn_metadata(
    sparse_seq_len: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: int = 0,
    sparse_block_size_q: int = 128,
    sparse_block_size_k: int = 128,
    quant_mode: int = 1,
    mask_mode: int = 3,
    layout_q: str = "TND",
    layout_kv: str = "PA_BNSD",
    layout_sparse_indices: str = "B_N_Qb_Kb",
) -> torch.Tensor:
    """Generate QBSA metadata with the same scheduling and layout as the AICPU implementation."""
    _validate_arguments(
        sparse_seq_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        sparse_block_size_q,
        sparse_block_size_k,
        quant_mode,
        mask_mode,
        layout_sparse_indices,
    )
    # These arguments are part of the registered Torch interface but are not used by AICPU scheduling.
    _ = (cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, layout_q, layout_kv)
    sparse_seq_len_data = sparse_seq_len.detach().cpu().tolist()
    scheduler = _MetadataScheduler(
        sparse_seq_len_data,
        sparse_seq_len.dim(),
        batch_size,
        num_heads_q,
        head_dim,
        sparse_block_size_q,
        sparse_block_size_k,
    )
    metadata_cpu = torch.tensor(scheduler.generate(), dtype=torch.int32, device="cpu")
    if sparse_seq_len.device.type == "cpu":
        return metadata_cpu
    return metadata_cpu.to(sparse_seq_len.device)


__all__ = ["npu_quant_block_sparse_attn_metadata"]
