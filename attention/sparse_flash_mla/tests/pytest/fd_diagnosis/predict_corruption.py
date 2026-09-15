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

"""Predict the 3+1 failure by overwriting the live CPU accumulator as Vec1 does.

This is a fault model, not a correctness reference.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("saved_result", type=Path)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sparse_flash_mla_golden import GeneralizedSFA
from result_compare_method import check_result

torch.set_num_threads(4)
saved = torch.load(args.saved_result, map_location="cpu", weights_only=False)
assert saved["metadata"][6] == 3 and saved["metadata"][12] == 3
x = saved["input"]["input"]
q = x["q"].reshape(64, 512).float()
kv = x["ori_kv"].reshape(512, 512).float()
sinks = x["sinks"].float().reshape(64)
update = GeneralizedSFA._update_s2_tile

# Compute the unmodified Vec1 max/sum states for the first core's three tiles.
states = []
maximum = sinks.clone()
total = torch.ones(64)
output = torch.zeros(64, 512)
for tile in range(3):
    maximum, total, output = update(
        q,
        kv[tile * 128 : (tile + 1) * 128],
        x["softmax_scale"],
        torch.bfloat16,
        maximum,
        total,
        output,
    )
    states.append((maximum.clone(), total.clone(), output.clone()))

# Vec1 of tile 2 precedes Vec2 of tile 1. Its broadcasts overwrite tile 0's O.
maximum, total, output = [value.clone() for value in states[0]]
for head in (0, 32):
    output[head, :256] = states[2][0][head : head + 32].repeat_interleave(8)
    output[head, 256:] = states[2][1][head : head + 32].repeat_interleave(8)
for tile in (1, 2):
    maximum, total, output = update(
        q,
        kv[tile * 128 : (tile + 1) * 128],
        x["softmax_scale"],
        torch.bfloat16,
        maximum,
        total,
        output,
    )

# The second core owns one tile; sinks are included only in the first partition.
other_max, other_sum, other_output = update(
    q,
    kv[384:],
    x["softmax_scale"],
    torch.bfloat16,
    torch.full((64,), -torch.inf),
    torch.ones(64),
    torch.zeros(64, 512),
)
merged_max = torch.maximum(maximum, other_max)
weight = torch.exp(maximum - merged_max) * total
other_weight = torch.exp(other_max - merged_max) * other_sum
predicted = (
    (
        output / total[:, None] * weight[:, None]
        + other_output / other_sum[:, None] * other_weight[:, None]
    )
    / (weight + other_weight)[:, None]
).to(torch.bfloat16)

actual = saved["output"].reshape(64, 512)
with contextlib.redirect_stdout(io.StringIO()):
    result = check_result(predicted, actual)
error = (predicted.float() - actual.float()).abs()
print(
    json.dumps(
        dict(
            status=result[0],
            fulfill=float(result[1]),
            max_abs=float(error.max()),
            mean_abs=float(error.mean()),
            head_max_abs=error.amax(1).tolist(),
        ),
        indent=2,
    )
)
