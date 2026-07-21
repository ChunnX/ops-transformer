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

"""SMLA-specific executor plugged into the ordinary pytest batch flow."""

from typing import Any, Dict, Optional

from .consistency.flow import ConsistencyFlow
from .consistency.model import RunResult
from .consistency.metadata_stub import MetadataTransform
from . import sparse_flash_mla_process


class SparseFlashMlaExecutor:
    """Execute metadata then SMLA and retain the exact metadata consumed."""

    def __init__(self, metadata_transform: Optional[MetadataTransform] = None):
        self.metadata_transform = metadata_transform

    def __call__(self, input_data: Dict[str, Any]) -> RunResult:
        captured_metadata = []
        output, softmax_lse = sparse_flash_mla_process.call_npu(
            input_data,
            metadata_sink=captured_metadata.append,
            metadata_transform=self.metadata_transform,
        )
        return RunResult(
            output=output,
            softmax_lse=softmax_lse,
            metadata=captured_metadata[0] if captured_metadata else None,
        )


def create_consistency_flow(
    metadata_transform: Optional[MetadataTransform] = None,
) -> ConsistencyFlow:
    return ConsistencyFlow(SparseFlashMlaExecutor(metadata_transform))
