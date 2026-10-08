#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""B/Q performance grid and one-factor-at-a-time A5 SFA generalization sweeps."""

import argparse
import json

import torch

from bench_sfa_split_kv_arch35 import measure


def cases():
    default = dict(
        batch_count=1,
        query_count=1,
        topk_count=2048,
        heads=8,
        dtype=torch.bfloat16,
        sparse_mode=0,
        return_lse=False,
        rope_dim=64,
        kv_length=4096,
        page_size=128,
    )
    for batch_count in (1, 2, 4, 8):
        for queries_per_batch in (1, 2, 4, 8):
            yield (
                "bq",
                f"B{batch_count}-Q{queries_per_batch}",
                dict(
                    default,
                    batch_count=batch_count,
                    query_count=batch_count * queries_per_batch,
                ),
            )
    sweeps = {
        "heads": range(1, 129),
        "dtype": (torch.bfloat16, torch.float16),
        "sparse_mode": (0, 3),
        "return_lse": (False, True),
        "rope_dim": (64, 0),
        "topk_count": (128, 256, 512, 768, 1000, 1024, 1536, 2048),
        "page_size": (16, 64, 128, 256, 512),
        "kv_length": (4096, 16384, 65536),
    }
    for field, values in sweeps.items():
        for value in values:
            yield field, str(value), dict(default, **{field: value})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=101)
    args = parser.parse_args()
    print(
        json.dumps(
            dict(
                device=torch.npu.get_device_name(0),
                calls=args.calls,
                repetitions=args.repetitions,
                configurations=len(list(cases())),
            )
        ),
        flush=True,
    )
    for sweep, value, config in cases():
        result = measure(calls=args.calls, repetitions=args.repetitions, **config)
        print(json.dumps(dict(result, sweep=sweep, sweep_value=value)), flush=True)


if __name__ == "__main__":
    main()
