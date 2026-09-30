# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Installed-package coverage for contiguous norm/gamma DMA and fallback."""

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
    torch.manual_seed(20260930)
    torch.npu.set_device(0)
    runner = AclnnRunner()
    records = []
    for bits in (2, 3, 4):
        for dtype in (torch.bfloat16, torch.float16):
            for kv_heads in (1, 2, 3, 7, 8, 15, 16, 17, 32):
                for length in (1, 31, 32, 33, 127, 128, 129):
                    groups = 3 if kv_heads <= 8 else 1
                    cpu = make_inputs(
                        3, length, bits, 3, kv_heads * groups, kv_heads, dtype
                    )
                    for tensor in cpu[4:6]:
                        values = torch.arange(tensor.numel()).reshape(tensor.shape)
                        tensor.copy_((0.25 + (values % 113) / 128).to(dtype))
                    cpu[7][-1] = 0
                    output, _ = runner(
                        [x.npu() for x in cpu], bits, length, output_fill=float("nan")
                    )
                    records.append(
                        dict(
                            bits=bits,
                            dtype=str(dtype),
                            kv_heads=kv_heads,
                            groups=groups,
                            length=length,
                            **check(output, cpu, bits),
                        )
                    )
    Path(args.output).write_text(json.dumps(records, indent=2))
    print(f"norm/gamma boundary cases passed: {len(records)}", flush=True)


if __name__ == "__main__":
    main()
