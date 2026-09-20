# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression for issue #5333: metadata and FlashAttn must share stream core limits.

Run after installing matching FlashAttn/FlashAttnMetadata and torch extensions:
    python test_stream_core_limit.py --cores 0 1 8 --mode aclgraph
Zero selects hardware defaults. Each stream gets fresh metadata; no TTK is used.
"""

import argparse
import logging
from contextlib import nullcontext

from torch_npu.npu.npugraph_ex import scope

import torch
import torch_npu
from cann_ops_transformer.ops.attention.flash_attn import (
    flash_attn,
    flash_attn_metadata,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, nargs="+", default=[0, 1, 8])
    parser.add_argument(
        "--mode", choices=["eager", "aclgraph", "scope"], default="aclgraph"
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--profile", help="Optional torch_npu profiler output directory"
    )
    parser.add_argument("--q-len", type=int, default=512)
    parser.add_argument("--kv-len", type=int, default=512)
    args = parser.parse_args()
    if min(args.q_len, args.kv_len) <= 0:
        parser.error("sequence lengths must be positive")
    torch.npu.set_device(0)
    props = torch.npu.get_device_properties(0)
    if any(c < 0 or c > props.cube_core_num for c in args.cores):
        parser.error("cores must be zero or within the hardware cube core count")
    torch.manual_seed(5333)
    dtype = getattr(torch, args.dtype)
    # Enough work to use all hardware cores; a tiny shape can hide the regression.
    q_cpu = torch.randn(2, 8, args.q_len, 128).to(dtype)
    k_cpu = torch.randn(2, 2, args.kv_len, 128).to(dtype)
    v_cpu = torch.randn_like(k_cpu)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q_cpu.float(),
        k_cpu.float().repeat_interleave(4, dim=1),
        v_cpu.float().repeat_interleave(4, dim=1),
    )
    q, k, v = (x.npu() for x in (q_cpu, k_cpu, v_cpu))
    torch.npu.synchronize()
    attrs = dict(
        max_seqlen_q=args.q_len,
        max_seqlen_kv=args.kv_len,
        mask_mode=0,
        layout_q="BNSD",
        layout_kv="BNSD",
        layout_out="BNSD",
    )
    torch._dynamo.config.ignore_logger_methods.add(logging.Logger.warning)

    def forward(q, k, v, metadata):
        out = flash_attn(q, k, v, metadata=metadata, softmax_scale=128**-0.5, **attrs)[
            0
        ]
        return out, metadata

    def limited_forward(q, k, v, metadata):
        # The backend may capture on another stream: encode the same limit in the graph.
        with scope.limit_core_num(cube, vector):
            return forward(q, k, v, metadata)

    def scoped_forward(q, k, v, unused_metadata):
        with scope.limit_core_num(cube, vector):
            metadata = flash_attn_metadata(8, 2, 128, batch_size=2, **attrs)
            return forward(q, k, v, metadata)

    for requested in args.cores:
        cube = requested or props.cube_core_num
        vector = 2 * cube if requested else props.vector_core_num
        stream = torch.npu.Stream()
        if requested and args.mode != "scope":
            torch.npu.set_stream_limit(stream, cube_num=cube, vector_num=vector)
        with torch.npu.stream(stream), torch.inference_mode():
            metadata = None
            if args.mode != "scope":
                metadata = flash_attn_metadata(8, 2, 128, batch_size=2, **attrs)
                header = metadata.cpu().tolist()[:16]
                assert header[4:6] == [cube, vector], (requested, header)
            # Scope mode captures metadata too; aclgraph mode generates it before capture.
            model = scoped_forward if args.mode == "scope" else limited_forward
            if args.mode == "eager":
                model = forward
            if args.mode != "eager":
                torch._dynamo.reset()
                model = torch.compile(
                    model,
                    backend="npugraph_ex",
                    fullgraph=True,
                    dynamic=False,
                    options={"static_kernel_compile": False},
                )
            for _ in range(3):
                out, result_metadata = model(q, k, v, metadata)
                stream.synchronize()
                header = result_metadata.cpu().tolist()[:16]
                assert header[4:6] == [cube, vector], (requested, header)
                assert 0 < header[0] <= 4, header
                torch.testing.assert_close(
                    out.cpu().float(), reference, atol=0.005, rtol=0.005
                )
            error = (out.cpu().float() - reference).abs().max().item()
            profiler = nullcontext()
            if args.profile:
                profiler = torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        f"{args.profile}/{cube}c"
                    ),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                    ),
                )
            with profiler:
                for _ in range(3):
                    model(q, k, v, metadata)
                stream.synchronize()
            print(
                f"PASS {args.mode} {args.dtype} cores={cube}/{vector} "
                f"metadata_header={header[:6]} max_abs_error={error:.6g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
