# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Capture the metadata actually consumed by ESL, without changing NPU math."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from .meta_cpu.mixed_quant_flash_attn_metadata_op import MixedQuantFlashAttnMetadataOp


def metadata_details(metadata):
    words = torch.as_tensor(metadata, dtype=torch.int64).flatten().tolist()
    if len(words) < 16:
        raise ValueError("MQFA metadata must contain its 16-word header")
    sections, fd_enabled, m_size, s2_size, aic, aiv = words[:6]
    if min(sections, aic, aiv) < 0:
        raise ValueError("Negative MQFA metadata dimensions")
    expected = 16 + sections * (aic + aiv) * 16
    if len(words) != expected:
        raise ValueError(f"MQFA metadata length {len(words)} != {expected}")
    splits = []
    fd_base = 16 + sections * aic * 16
    for section in range(sections):
        for core in range(aiv):
            start = fd_base + (section * aiv + core) * 16
            split_count, row_count = words[start + 3], words[start + 5]
            if row_count > 0:
                splits.append(split_count)
    return {
        "fd_enabled": bool(fd_enabled),
        "m_base_size": m_size,
        "s2_base_size": s2_size,
        "fd_split_counts": splits,
    }


def check_fd_metadata(metadata, require_fd=False):
    if metadata is None:
        if require_fd:
            raise AssertionError("FD coverage requires captured runtime metadata")
        print("MQFA_METADATA unavailable (not the ESL metadata path)")
        return
    details = metadata_details(metadata)
    print(f"MQFA_METADATA {details}")
    if require_fd and not (
        details["fd_enabled"] and max(details["fd_split_counts"], default=0) > 1
    ):
        raise AssertionError(f"Case did not exercise multi-partition FD: {details}")


@contextmanager
def record_esl_metadata():
    """Observe the parser return value; never rerun or replace the scheduler."""
    captured = []
    original = MixedQuantFlashAttnMetadataOp._parse_output_to_metadata

    def record(instance, output_json):
        metadata = original(instance, output_json)
        captured.append(torch.tensor(metadata, dtype=torch.int32))
        return metadata

    with patch.object(
        MixedQuantFlashAttnMetadataOp, "_parse_output_to_metadata", record
    ):
        yield captured
