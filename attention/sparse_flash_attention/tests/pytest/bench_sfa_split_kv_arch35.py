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


def measure(query_count, topk_count, calls, repetitions):
    generator = torch.Generator().manual_seed(20260928 + query_count + topk_count)
    kv_length, page_size, heads = 4096, 128, 8
    query = torch.randn(query_count, heads, 512, generator=generator).bfloat16().npu()
    query_rope = (
        torch.randn(query_count, heads, 64, generator=generator).bfloat16().npu()
    )
    key = torch.randn(kv_length, 512, generator=generator).bfloat16().npu()
    key_rope = torch.randn(kv_length, 64, generator=generator).bfloat16().npu()
    indices = (
        torch.stack(
            [
                torch.randperm(kv_length, generator=generator)[:topk_count].int()
                for _ in range(query_count)
            ]
        )
        .unsqueeze(1)
        .npu()
    )
    block_table = (
        torch.arange(kv_length // page_size, dtype=torch.int32).unsqueeze(0).npu()
    )
    query_lengths = torch.tensor([query_count], dtype=torch.int32).npu()
    kv_lengths = torch.tensor([kv_length], dtype=torch.int32).npu()

    def run_operator():
        return torch_npu.npu_sparse_flash_attention(
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
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=False,
        )[0]

    for _ in range(4):
        run_operator()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls):
            output = run_operator()
    graph.replay()
    torch.npu.synchronize()
    assert torch.isfinite(output).all()

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
        "q": query_count,
        "topk": topk_count,
        "median_us": statistics.median(times_us),
        "samples_us": times_us,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-counts", nargs="+", type=int, default=[1, 4])
    parser.add_argument(
        "--topk-counts", nargs="+", type=int, default=[256, 512, 768, 1024, 1536, 2048]
    )
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=11)
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
    for query_count in args.query_counts:
        for topk_count in args.topk_counts:
            print(
                json.dumps(
                    measure(query_count, topk_count, args.calls, args.repetitions)
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
