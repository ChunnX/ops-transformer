# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Invoke one complete operator for msprof op with its default replay settings."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from cases import ALL_CASES
from sageattention2 import sageattn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=[case.name for case in ALL_CASES], required=True
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seq", type=int, help="Override context length")
    parser.add_argument("--smooth-v", action="store_true")
    parser.add_argument("--operator", choices=["sage", "fia"], required=True)
    args = parser.parse_args()
    case = next(c for c in ALL_CASES if c.name == args.case)
    batch, heads, hk, dim = case.batch, case.heads, case.hk, case.dim
    dtype_name, causal = case.dtype, case.causal
    seq = args.seq if args.seq is not None else case.seq
    if seq < 1:
        parser.error("invalid context length")
    torch.manual_seed(71)
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    q = torch.randn(batch, heads, seq, dim, device="npu", dtype=dtype)
    k = torch.randn(batch, hk, seq, dim, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    mask = (
        torch.ones(2048, 2048, dtype=torch.bool, device="npu").triu(1)
        if causal
        else None
    )

    def run_sage():
        return sageattn(q, k, v, is_causal=causal, smooth_v=args.smooth_v)

    def run_fia():
        return torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=heads,
            num_key_value_heads=hk,
            input_layout="BNSD",
            scale=dim**-0.5,
            atten_mask=mask,
            sparse_mode=2 if causal else 0,
        )[0]

    # Only one target operator invocation; msprof op owns warm-up and replay.
    torch.npu.synchronize()
    output = run_sage() if args.operator == "sage" else run_fia()
    torch.npu.synchronize()
    assert torch.isfinite(output).all().item()
    report = {
        "case": args.case,
        "operator": args.operator,
        "shape": list(q.shape),
        "kv_heads": hk,
        "dtype": dtype_name,
        "causal": causal,
        "smooth_k": True,
        "smooth_v": False,
        "application_target_calls": 1,
        "application_warmup_calls": 0,
        "path": "public PyTorch -> ACLNN",
        "finite": True,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
