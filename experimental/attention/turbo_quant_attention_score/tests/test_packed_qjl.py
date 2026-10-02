# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Installed-package packed QJL masks and multi-tile prefetch boundaries."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from aclnn_runner import AclnnRunner
from test_turbo_quant_attention_score import check, make_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20261001)
    torch.npu.set_device(0)
    runner = AclnnRunner()
    records = []
    for bits in (2, 3, 4):
        for dtype in (torch.bfloat16, torch.float16):
            for groups in (1, 3, 4, 5, 8):
                for length in (31, 32, 33, 127, 128, 129, 4095, 4096, 4097):
                    cpu = make_inputs(1, length, bits, 1, 8 * groups, 8, dtype)
                    cpu[3].copy_(
                        (torch.arange(cpu[3].numel()).reshape_as(cpu[3]) % 256).byte()
                    )
                    # Signed basis coefficients exercise both mask halves and bit 127.
                    cpu[1].zero_()
                    for h in range(cpu[1].shape[1]):
                        cpu[1][0, h, h % 128] = -1
                        cpu[1][0, h, 63] = 0.5
                        cpu[1][0, h, 64] = -0.75
                        cpu[1][0, h, 127] = 1.5
                    inputs = [x.npu() for x in cpu]
                    first = None
                    for _ in range(2):
                        out, _ = runner(inputs, bits, length, output_fill=float("nan"))
                        metric = check(out, cpu, bits)
                        if first is None:
                            first = out.cpu()
                        else:
                            assert torch.equal(first, out.cpu())
                    records.append(
                        dict(
                            bits=bits,
                            dtype=str(dtype),
                            groups=groups,
                            length=length,
                            **metric,
                        )
                    )
    Path(args.output).write_text(json.dumps(records, indent=2))
    print(f"Packed QJL / prefetch cases passed: {len(records)}", flush=True)


if __name__ == "__main__":
    main()
