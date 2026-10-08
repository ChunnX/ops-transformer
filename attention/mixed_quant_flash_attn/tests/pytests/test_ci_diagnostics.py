# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest

from core.ci_diagnostics import check_fd_metadata, metadata_details, record_esl_metadata
from core.meta_cpu.mixed_quant_flash_attn_metadata_op import (
    MixedQuantFlashAttnMetadataOp,
)


def _metadata(split_count):
    words = [0] * 48
    words[:6] = [1, int(split_count > 1), 32, 512, 1, 1]
    words[35] = split_count
    words[37] = 6
    return words


def test_metadata_reports_actual_partitions():
    details = metadata_details(_metadata(3))
    assert details["fd_split_counts"] == [3]
    check_fd_metadata(_metadata(3), require_fd=True)


@pytest.mark.parametrize("metadata", [None, _metadata(1)])
def test_required_fd_cannot_silently_skip(metadata):
    with pytest.raises(AssertionError):
        check_fd_metadata(metadata, require_fd=True)


@pytest.mark.parametrize("metadata", [[], [0] * 17, [-1] + [0] * 15])
def test_invalid_metadata_rejected(metadata):
    with pytest.raises(ValueError):
        metadata_details(metadata)


def test_metadata_recorder_is_transparent_and_restores_parser(monkeypatch):
    expected = _metadata(3)
    calls = []

    def parse(instance, payload):
        calls.append(payload)
        return expected

    monkeypatch.setattr(
        MixedQuantFlashAttnMetadataOp, "_parse_output_to_metadata", parse
    )
    op = MixedQuantFlashAttnMetadataOp()
    with record_esl_metadata() as captured:
        assert op._parse_output_to_metadata({"test": True}) is expected
    assert calls == [{"test": True}]
    assert captured[0].tolist() == expected
    assert MixedQuantFlashAttnMetadataOp._parse_output_to_metadata is parse
