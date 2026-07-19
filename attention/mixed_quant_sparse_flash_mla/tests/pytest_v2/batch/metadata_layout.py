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

"""Versioned parser for the shared SMLA/MQSMLA metadata layout."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass(frozen=True)
class FaRecord:
    core_id: int
    enabled: int
    bn2_start: int
    m_start: int
    s2_start: int
    bn2_end: int
    m_end: int
    s2_end: int
    first_fd_workspace_index: int
    s2_max_num: Optional[int]

    @property
    def start(self) -> Tuple[int, int, int]:
        return self.bn2_start, self.m_start, self.s2_start

    @property
    def end(self) -> Tuple[int, int, int]:
        return self.bn2_end, self.m_end, self.s2_end


@dataclass(frozen=True)
class FdRecord:
    core_id: int
    enabled: int
    bn2_index: int
    m_index: int
    workspace_index: int
    workspace_num: int
    m_start: int
    m_num: int
    unknown_slot: int


@dataclass(frozen=True)
class ParsedMetadata:
    layout_version: int
    fa_record_size: int
    fa_records: Tuple[FaRecord, ...]
    fd_records: Tuple[FdRecord, ...]
    reserved: Tuple[int, ...]
    raw: Tuple[int, ...]


class MetadataLayout:
    """Parses 1024-int32 metadata using constants shared by both operators."""

    TOTAL_SIZE = 1024
    FA_CORE_COUNT = 36
    FD_CORE_COUNT = 72
    FD_RECORD_SIZE = 8
    SPLIT_G_FA_RECORD_SIZE = 9
    LEGACY_FA_RECORD_SIZE = 8
    CURRENT_LAYOUT_VERSION = 1
    LEGACY_LAYOUT_VERSION = 0

    def __init__(self, fa_record_size: int = SPLIT_G_FA_RECORD_SIZE):
        if fa_record_size not in (self.LEGACY_FA_RECORD_SIZE, self.SPLIT_G_FA_RECORD_SIZE):
            raise ValueError(f"Unsupported FA record size: {fa_record_size}")
        self.fa_record_size = fa_record_size
        self.fa_end = self.FA_CORE_COUNT * fa_record_size
        self.fd_end = self.fa_end + self.FD_CORE_COUNT * self.FD_RECORD_SIZE
        if self.fd_end > self.TOTAL_SIZE:
            raise ValueError("Metadata records exceed the 1024-int32 buffer")

    def parse(self, metadata: torch.Tensor) -> ParsedMetadata:
        if not isinstance(metadata, torch.Tensor):
            raise TypeError("metadata must be a torch.Tensor")
        if metadata.dtype != torch.int32:
            raise ValueError(f"metadata dtype must be torch.int32, got {metadata.dtype}")
        flat = metadata.detach().cpu().contiguous().reshape(-1)
        if flat.numel() != self.TOTAL_SIZE:
            raise ValueError(f"metadata size {flat.numel()} != {self.TOTAL_SIZE}")
        signed_values = flat.to(torch.int64).tolist()
        values = tuple(value if value >= 0 else value + (1 << 32) for value in signed_values)

        fa_records: List[FaRecord] = []
        for core_id in range(self.FA_CORE_COUNT):
            offset = core_id * self.fa_record_size
            row = values[offset:offset + self.fa_record_size]
            fa_records.append(
                FaRecord(
                    core_id=core_id,
                    enabled=row[0],
                    bn2_start=row[1],
                    m_start=row[2],
                    s2_start=row[3],
                    bn2_end=row[4],
                    m_end=row[5],
                    s2_end=row[6],
                    first_fd_workspace_index=row[7],
                    s2_max_num=row[8] if self.fa_record_size == 9 else None,
                )
            )

        fd_records: List[FdRecord] = []
        for core_id in range(self.FD_CORE_COUNT):
            offset = self.fa_end + core_id * self.FD_RECORD_SIZE
            row = values[offset:offset + self.FD_RECORD_SIZE]
            fd_records.append(
                FdRecord(
                    core_id=core_id,
                    enabled=row[0],
                    bn2_index=row[1],
                    m_index=row[2],
                    workspace_index=row[3],
                    workspace_num=row[4],
                    m_start=row[5],
                    m_num=row[6],
                    unknown_slot=row[7],
                )
            )

        return ParsedMetadata(
            layout_version=(
                self.CURRENT_LAYOUT_VERSION
                if self.fa_record_size == self.SPLIT_G_FA_RECORD_SIZE
                else self.LEGACY_LAYOUT_VERSION
            ),
            fa_record_size=self.fa_record_size,
            fa_records=tuple(fa_records),
            fd_records=tuple(fd_records),
            reserved=values[self.fd_end:],
            raw=values,
        )
