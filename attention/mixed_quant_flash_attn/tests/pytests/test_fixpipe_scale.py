# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fixpipe scalar format tests, independent of attention and FP16 conversion."""

import struct

import pytest
import torch

from core.cpu import fixpipe_softmax_scale


@pytest.mark.parametrize(
    "bits, expected",
    [
        (0x3DB504F3, 0x3DB50000),  # FP32(1 / sqrt(128)).
        (0xBDB504F3, 0xBDB50000),
        (0x3F801FFF, 0x3F800000),  # Immediately below a retained-mantissa step.
        (0x3F802000, 0x3F802000),
        (0x3F802001, 0x3F802000),
        (0x00801FFF, 0x00800000),  # Normal FP32, outside FP16 exponent range.
        (0x501502F9, 0x50150000),
        (0x3E800000, 0x3E800000),  # Exactly representable scale is unchanged.
        (0x00000000, 0x00000000),
        (0x80000000, 0x80000000),
    ],
)
def test_fixpipe_scale_clears_only_low_13_bits(bits, expected):
    scale = struct.unpack("<f", struct.pack("<I", bits))[0]
    effective = fixpipe_softmax_scale(scale)
    actual = struct.unpack("<I", struct.pack("<f", effective))[0]
    assert actual == expected


@pytest.mark.parametrize("scale", [1.0009, -1.0009, 1e-10, 1e10])
def test_half_round_trip_is_not_fixpipe_truncation(scale):
    effective = fixpipe_softmax_scale(scale)
    half_round_trip = torch.tensor(scale, dtype=torch.float32).half().float().item()
    assert effective != half_round_trip
    assert torch.isfinite(torch.tensor(effective))
    assert effective != 0


def test_default_scale_matches_documented_fixpipe_value():
    assert fixpipe_softmax_scale(128**-0.5) == 0.08837890625
