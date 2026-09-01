#!/usr/bin/env python3
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
"""Run the M-RoPE MX Torch Extension directly and compare with its canonical Golden.

This is a standalone single-operator example.  It does not use TTK, TestSpec CSV,
or TTK input customization.  The numerical implementation and acceptance rules
remain in tests/assets/qkv_rms_norm_rope_cache_with_k_scale_*_golden.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


OPERATOR_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = OPERATOR_DIR / "tests" / "assets"
GOLDEN_FILE = GOLDEN_DIR / "qkv_rms_norm_rope_cache_with_k_scale_e2e_golden.py"
KERNEL_GOLDEN_FILE = (
    GOLDEN_DIR / "qkv_rms_norm_rope_cache_with_k_scale_kernel_golden.py"
)
sys.path.insert(0, str(GOLDEN_DIR))
import qkv_rms_norm_rope_cache_with_k_scale_e2e_golden as e2e_golden  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct M-RoPE MX Torch Extension functional/inplace test"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--api", choices=("functional", "inplace"), default="functional"
    )
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--nq", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args) -> None:
    if not 1 <= args.tokens <= 262144:
        raise ValueError("--tokens must satisfy 1 <= T <= 262144")
    if not 0 < args.nq <= 64 or args.nq % 8 != 0:
        raise ValueError("--nq must be one of 8, 16, ..., 64")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")


def make_fp8_zeros(shape, dtype):
    return torch.zeros(shape, dtype=torch.uint8).view(dtype)


def make_inputs(args):
    torch.manual_seed(args.seed)
    token_num = args.tokens
    q_heads = args.nq
    k_heads = q_heads // 8
    v_heads = k_heads
    head_dim = 128
    block_num = (token_num + args.block_size - 1) // args.block_size + 1
    capacity = block_num * args.block_size

    qkv = torch.randn(token_num, q_heads + k_heads + v_heads, head_dim).to(
        torch.bfloat16
    )
    q_gamma = 0.5 + torch.rand(head_dim, dtype=torch.float32)
    k_gamma = 0.5 + torch.rand(head_dim, dtype=torch.float32)
    cos_sin = torch.randn(args.max_seq_len, head_dim, dtype=torch.float32)

    generator = torch.Generator().manual_seed(args.seed + 1)
    slot_mapping = torch.randperm(capacity, generator=generator)[:token_num].to(
        torch.int32
    )
    k_cache = make_fp8_zeros(
        (block_num, k_heads, args.block_size, head_dim), torch.float8_e4m3fn
    )
    v_cache = make_fp8_zeros(
        (block_num, v_heads, args.block_size, head_dim), torch.float8_e4m3fn
    )
    k_scale_cache = torch.full(
        (block_num, k_heads, args.block_size, head_dim // 32),
        127,
        dtype=torch.uint8,
    ).view(torch.float8_e8m0fnu)
    v_scale = 0.5 + torch.rand(v_heads, head_dim, dtype=torch.float32)

    token_index = torch.arange(token_num, dtype=torch.int32)
    mrope_position = torch.stack(
        (
            token_index % args.max_seq_len,
            (token_index * 3 + 1) % args.max_seq_len,
            (token_index * 5 + 2) % args.max_seq_len,
        ),
        dim=1,
    )

    return {
        "qkv": qkv,
        "q_gamma": q_gamma,
        "k_gamma": k_gamma,
        "cos_sin": cos_sin,
        "slot_mapping": slot_mapping,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale_cache": k_scale_cache,
        "query_start_loc": None,
        "seq_lens": None,
        "head_nums": [q_heads, k_heads, v_heads],
        "rotation": None,
        "v_scale": v_scale,
        "layout_qkv": "TND",
        "layout_q_out": "TND",
        "epsilon": 1e-6,
        "mrope_position": mrope_position,
        "mrope_section": [22, 12, 10],
        "q_quant_mode": "Mx",
        "k_quant_mode": "Mx",
        "q_out_dtype": torch.float8_e4m3fn,
    }


def golden_outputs(api, inputs):
    spec = (
        e2e_golden.PtaQkvRmsNormRopeCacheWithKScaleFunctionalTestSpec
        if api == "functional"
        else e2e_golden.PtaQkvRmsNormRopeCacheWithKScaleInplaceTestSpec
    )
    return spec.golden(**inputs)


def move_inputs_to_npu(inputs, device):
    result = {}
    for name, value in inputs.items():
        result[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return result


def resolve_op(package, name):
    if hasattr(package, name):
        return getattr(package, name), f"{package.__name__}.{name}"
    namespace = torch.ops.cann_ops_transformer
    if hasattr(namespace, name):
        return getattr(namespace, name), f"torch.ops.cann_ops_transformer.{name}"
    raise RuntimeError(
        f"the selected package does not register cann_ops_transformer::{name}; "
        "install the Torch Extension wheel built with PR 10557"
    )


def invoke(api, package, inputs):
    common = (
        inputs["qkv"],
        inputs["q_gamma"],
        inputs["k_gamma"],
        inputs["cos_sin"],
        inputs["slot_mapping"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["k_scale_cache"],
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["head_nums"],
    )
    keyword = {
        "rotation": inputs["rotation"],
        "v_scale": inputs["v_scale"],
        "layout_qkv": inputs["layout_qkv"],
        "layout_q_out": inputs["layout_q_out"],
        "epsilon": inputs["epsilon"],
        "mrope_position": inputs["mrope_position"],
        "mrope_section": inputs["mrope_section"],
        "q_quant_mode": inputs["q_quant_mode"],
        "k_quant_mode": inputs["k_quant_mode"],
        "q_out_dtype": inputs["q_out_dtype"],
    }
    if api == "functional":
        op, entry = resolve_op(package, "qkv_rms_norm_rope_cache_with_k_scale")
        return list(op(*common, **keyword)), entry

    op, entry = resolve_op(package, "qkv_rms_norm_rope_cache_with_k_scale_")
    q_out, q_scale = op(*common, **keyword)
    return (
        [
            q_out,
            q_scale,
            inputs["k_cache"],
            inputs["v_cache"],
            inputs["k_scale_cache"],
        ],
        entry,
    )


def compare(api, actual, expected, cpu_inputs, npu_inputs):
    if api == "functional":
        actual.extend(
            [npu_inputs["k_cache"], npu_inputs["v_cache"], npu_inputs["k_scale_cache"]]
        )
    context = SimpleNamespace(
        api_name=(
            "cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale"
            if api == "functional"
            else "cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_"
        ),
        attributes={
            "q_quant_mode": "Mx",
            "k_quant_mode": "Mx",
        },
        input_tensors=[
            cpu_inputs["qkv"],
            cpu_inputs["q_gamma"],
            cpu_inputs["k_gamma"],
            cpu_inputs["cos_sin"],
            cpu_inputs["slot_mapping"],
        ],
    )
    return e2e_golden.pta_compare(*(actual + list(expected)), compare_context=context)


def main() -> int:
    args = parse_args()
    validate_args(args)

    package_spec = importlib.util.find_spec("cann_ops_transformer")
    if package_spec is None:
        raise RuntimeError("cann_ops_transformer is not installed or not on PYTHONPATH")

    import cann_ops_transformer
    import torch_npu

    device = f"npu:{args.device}"
    torch_npu.npu.set_device(args.device)
    cpu_inputs = make_inputs(args)
    expected = golden_outputs(args.api, cpu_inputs)
    npu_inputs = move_inputs_to_npu(cpu_inputs, device)
    actual, entry = invoke(args.api, cann_ops_transformer, npu_inputs)
    torch_npu.npu.synchronize()
    result = compare(args.api, actual, expected, cpu_inputs, npu_inputs)

    print(f"package={package_spec.origin}")
    print(f"entry={entry}")
    print(f"golden={GOLDEN_FILE}")
    print(f"golden_sha256={sha256(GOLDEN_FILE)}")
    print(f"kernel_golden={KERNEL_GOLDEN_FILE}")
    print(f"kernel_golden_sha256={sha256(KERNEL_GOLDEN_FILE)}")
    print(
        f"api={args.api} T={args.tokens} head_nums={cpu_inputs['head_nums']} "
        f"device={device} seed={args.seed}"
    )
    print(result["precision"])
    if result["pass"]:
        print("PASS")
        return 0
    print(f"FAIL: {result['error_info']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
