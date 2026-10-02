# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise reuse of both prefill key-buffer slots through the real aclnn API."""

import argparse
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from aclnn_runner import AclnnRunner
from test_turbo_quant_attention_score import check, make_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="prefill_ring_regression.json")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    torch.manual_seed(20261002)
    runner = AclnnRunner()
    records = []
    # 88 tiles cross both ring slots on 20-AIC hardware, with a partial final
    # wave and a KV-head transition. The 32K official test covers 26 waves.
    for bits in (2, 3, 4):
        for norm_dtype in (torch.bfloat16, torch.float16):
            for length_dtype in (torch.int32, torch.int64):
                groups = (1, 2, 3, 4, 5, 8)[len(records) % 6]
                cpu = make_inputs(
                    128,
                    11264,
                    bits,
                    heads=groups * 2,
                    kv_heads=2,
                    norm_dtype=norm_dtype,
                )
                cpu[-1] = cpu[-1].to(length_dtype)
                inputs = [x.npu() for x in cpu]
                ref = None
                repeats = []
                for repeat in range(3):
                    out, timing = runner(inputs, bits, 11264, output_fill=float("nan"))
                    assert timing["workspace_bytes"] > 0, "Cube path was not selected"
                    host = out.cpu()
                    if ref is None:
                        accuracy = check(host, cpu, bits)
                        ref = host
                    else:
                        assert torch.equal(host, ref), "reused scratch changed output"
                    repeats.append(timing["device_us"])
                records.append(
                    {
                        "bits": bits,
                        "groups": groups,
                        "norm_dtype": str(norm_dtype),
                        "length_dtype": str(length_dtype),
                        "workspace_bytes": timing["workspace_bytes"],
                        "repeats_us": repeats,
                        "accuracy": accuracy,
                    }
                )
                print(json.dumps(records[-1]), flush=True)
                Path(args.output).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
