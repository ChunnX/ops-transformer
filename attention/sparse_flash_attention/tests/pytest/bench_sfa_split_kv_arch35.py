#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Single-operator A5 SFA latency, excluding indexer and index remapping."""

import argparse
import json
import statistics

import torch
import torch_npu

from sfa_test_operator import get_sfa_operator


def measure(
    batch_count,
    query_count,
    topk_count,
    calls,
    repetitions,
    heads=8,
    dtype=torch.bfloat16,
    sparse_mode=0,
    return_lse=False,
):
    generator = torch.Generator().manual_seed(
        20260928 + query_count + topk_count + (batch_count - 1) * 1000
    )
    kv_length, page_size = 4096, 128
    query_counts = [1] * (batch_count - 1) + [query_count - batch_count + 1]
    query_ends = [sum(query_counts[: index + 1]) for index in range(batch_count)]
    query = torch.randn(query_count, heads, 512, generator=generator).to(dtype).npu()
    query_rope = (
        torch.randn(query_count, heads, 64, generator=generator).to(dtype).npu()
    )
    key = torch.randn(batch_count * kv_length, 512, generator=generator).to(dtype).npu()
    key_rope = (
        torch.randn(batch_count * kv_length, 64, generator=generator).to(dtype).npu()
    )
    indices = (
        torch.stack(
            [
                torch.randperm(kv_length - query_count, generator=generator)[
                    :topk_count
                ].int()
                for _ in range(query_count)
            ]
        )
        .unsqueeze(1)
        .npu()
    )
    block_table = torch.arange(batch_count * kv_length // page_size, dtype=torch.int32)
    block_table = block_table.reshape(batch_count, kv_length // page_size).npu()
    query_lengths = torch.tensor(query_ends, dtype=torch.int32).npu()
    kv_lengths = torch.full((batch_count,), kv_length, dtype=torch.int32).npu()

    sfa_operator = get_sfa_operator()

    def run_operator():
        return sfa_operator(
            query,
            key.reshape(-1, page_size, 1, 512),
            key.reshape(-1, page_size, 1, 512),
            indices,
            1.0 / (576**0.5),
            block_table=block_table,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=query_rope,
            key_rope=key_rope.reshape(-1, page_size, 1, 64),
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=sparse_mode,
            attention_mode=2,
            return_softmax_lse=return_lse,
        )

    for _ in range(4):
        run_operator()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls):
            output = run_operator()
    graph.replay()
    torch.npu.synchronize()
    assert torch.isfinite(output[0]).all()
    if return_lse:
        assert torch.isfinite(output[1]).all() and torch.isfinite(output[2]).all()

    times_us = []
    for _ in range(repetitions):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000 / calls)
    return {
        "b": batch_count,
        "q": query_count,
        "topk": topk_count,
        "g": heads,
        "dtype": str(dtype),
        "sparse_mode": sparse_mode,
        "return_lse": return_lse,
        "min_us": min(times_us),
        "median_us": statistics.median(times_us),
        "samples_us": times_us,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-counts", nargs="+", type=int, default=[1])
    parser.add_argument("--query-counts", nargs="+", type=int, default=[1, 4])
    parser.add_argument(
        "--topk-counts", nargs="+", type=int, default=[256, 512, 768, 1024, 1536, 2048]
    )
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--heads", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["bf16", "fp16"], default=["bf16"]
    )
    parser.add_argument(
        "--sparse-modes", nargs="+", type=int, choices=[0, 3], default=[0]
    )
    parser.add_argument(
        "--lse-modes", nargs="+", choices=["off", "on"], default=["off"]
    )
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(0),
                "calls": args.calls,
                "repetitions": args.repetitions,
            }
        )
    )
    for batch_count in args.batch_counts:
        for query_count in args.query_counts:
            if query_count < batch_count:
                continue
            for topk_count in args.topk_counts:
                for heads in args.heads:
                    for dtype in args.dtypes:
                        for sparse_mode in args.sparse_modes:
                            for lse_mode in args.lse_modes:
                                result = measure(
                                    batch_count,
                                    query_count,
                                    topk_count,
                                    args.calls,
                                    args.repetitions,
                                    heads,
                                    torch.bfloat16
                                    if dtype == "bf16"
                                    else torch.float16,
                                    sparse_mode,
                                    lse_mode == "on",
                                )
                                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
