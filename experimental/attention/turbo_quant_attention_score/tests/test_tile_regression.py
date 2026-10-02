# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Installed-package coverage for 32-key tiles and grouped output DMA tails."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from aclnn_runner import AclnnRunner
from test_submission_regression import patterned
from test_turbo_quant_attention_score import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    runner = AclnnRunner()
    records = []
    for bits in (2, 3, 4):
        for dtype in (torch.bfloat16, torch.float16):
            for groups in (3, 4):
                configurations = [(1, n) for n in (7, 8, 9, 31, 32, 33, 63, 65)]
                configurations.append((3, 33))
                for batch, length in configurations:
                    tokens = 9 if batch == 1 else batch
                    cpu = patterned(bits, dtype, batch, tokens, groups * 2)
                    cpu[7].fill_(length)
                    if batch > 1:
                        cpu[7][1] = length // 2
                        cpu[7][2] = 0
                    inputs = [value.npu() for value in cpu]
                    out, _ = runner(inputs, bits, length, output_fill=float("nan"))
                    result = dict(
                        bits=bits,
                        dtype=str(dtype),
                        groups=groups,
                        batch=batch,
                        tokens=tokens,
                        length=length,
                        **check(out, cpu, bits),
                    )
                    records.append(result)
                    print(json.dumps(result), flush=True)
    Path(args.output).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
