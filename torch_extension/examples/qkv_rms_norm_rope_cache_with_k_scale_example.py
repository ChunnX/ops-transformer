#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale on several shapes.

Example:
    python3 torch_extension/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py --list-cases
    python3 torch_extension/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py
    python3 torch_extension/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py --case smoke_tnd --mode functional
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Tuple

import torch
import torch_npu

TORCH_EXTENSION_DIR = Path(__file__).resolve().parents[1]
if str(TORCH_EXTENSION_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_EXTENSION_DIR))

HEAD_SIZE = 128
DEFAULT_EPSILON = 1e-6


@dataclass(frozen=True)
class Case:
    name: str
    batch_seq_lens: Tuple[int, ...]
    block_size: int
    block_num: int
    n_q: int
    n_k: int
    n_v: int
    input_layout: str


CASES = (
    Case("smoke_ntd", (1,), 16, 4, 16, 2, 2, "NTD"),
    Case("smoke_tnd", (1,), 16, 4, 16, 2, 2, "TND"),
    Case("multi_batch_ntd", (2, 7), 16, 4, 16, 2, 2, "NTD"),
    Case("multi_batch_tnd", (2, 7), 16, 4, 16, 2, 2, "TND"),
    Case("head_combo", (32,), 16, 8, 32, 4, 4, "NTD"),
)


def make_cu_seq_len(seq_lens: Iterable[int], device: str) -> torch.Tensor:
    prefix = [0]
    for seq_len in seq_lens:
        prefix.append(prefix[-1] + int(seq_len))
    return torch.tensor(prefix, dtype=torch.int32, device=device)


def make_inputs(case: Case, device: str):
    total_tokens = sum(case.batch_seq_lens)
    head_nums = [case.n_q, case.n_k, case.n_v]
    total_heads = case.n_q + case.n_k + case.n_v
    qkv_shape = (total_tokens, total_heads, HEAD_SIZE)
    if case.input_layout == "NTD":
        qkv_shape = (total_heads, total_tokens, HEAD_SIZE)

    max_seq_len = max(case.batch_seq_lens)
    qkv = torch.randn(qkv_shape, dtype=torch.bfloat16, device=device)
    q_gamma = torch.ones((HEAD_SIZE,), dtype=torch.float32, device=device)
    k_gamma = torch.ones((HEAD_SIZE,), dtype=torch.float32, device=device)
    cos_sin = torch.randn((max_seq_len, HEAD_SIZE), dtype=torch.float32, device=device)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int32, device=device) % (case.block_num * case.block_size)
    k_cache = torch.empty((case.block_num, case.n_k, case.block_size, HEAD_SIZE),
                          dtype=torch.float8_e4m3fn, device=device)
    v_cache = torch.empty((case.block_num, case.n_v, case.block_size, HEAD_SIZE),
                          dtype=torch.float8_e4m3fn, device=device)
    k_scale_cache = torch.zeros((case.block_num, case.n_k, case.block_size, 1),
                                dtype=torch.float32, device=device)
    cu_seq_len = make_cu_seq_len(case.batch_seq_lens, device)
    actual_seq_lens = torch.tensor(case.batch_seq_lens, dtype=torch.int32, device=device)
    q_rotation = torch.eye(HEAD_SIZE, dtype=torch.bfloat16, device=device)
    k_rotation = torch.eye(HEAD_SIZE, dtype=torch.bfloat16, device=device)
    v_scale = torch.ones((case.n_v,), dtype=torch.float32, device=device)

    return {
        "qkv": qkv,
        "q_gamma": q_gamma,
        "k_gamma": k_gamma,
        "cos_sin": cos_sin,
        "slot_mapping": slot_mapping,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale_cache": k_scale_cache,
        "cu_seq_len": cu_seq_len,
        "actual_seq_lens": actual_seq_lens,
        "head_nums": head_nums,
        "input_layout": case.input_layout,
        "q_rotation": q_rotation,
        "k_rotation": k_rotation,
        "v_scale": v_scale,
        "epsilon": DEFAULT_EPSILON,
    }


def expected_shapes(case: Case):
    total_tokens = sum(case.batch_seq_lens)
    if case.input_layout == "TND":
        return (total_tokens, case.n_q, HEAD_SIZE), (total_tokens, case.n_q)
    return (case.n_q, total_tokens, HEAD_SIZE), (case.n_q, total_tokens)


def assert_outputs(case: Case, outputs):
    q_out, q_scale = outputs[:2]
    q_out_shape, q_scale_shape = expected_shapes(case)
    assert tuple(q_out.shape) == q_out_shape, f"{case.name}: q_out shape mismatch: {tuple(q_out.shape)} vs {q_out_shape}"
    assert tuple(q_scale.shape) == q_scale_shape, (
        f"{case.name}: q_scale shape mismatch: {tuple(q_scale.shape)} vs {q_scale_shape}")
    assert q_out.dtype == torch.float8_e4m3fn, f"{case.name}: q_out dtype mismatch: {q_out.dtype}"
    assert q_scale.dtype == torch.float32, f"{case.name}: q_scale dtype mismatch: {q_scale.dtype}"


def run_case(case: Case, mode: str, device: str):
    from cann_ops_transformer.ops import (
        qkv_rms_norm_rope_cache_with_k_scale,
        qkv_rms_norm_rope_cache_with_k_scale_functional,
    )

    tensors = make_inputs(case, device)
    with torch.no_grad():
        if mode == "inplace":
            q_out, q_scale = qkv_rms_norm_rope_cache_with_k_scale(
                tensors["qkv"],
                tensors["q_gamma"],
                tensors["k_gamma"],
                tensors["cos_sin"],
                tensors["slot_mapping"],
                tensors["k_cache"],
                tensors["v_cache"],
                tensors["k_scale_cache"],
                tensors["cu_seq_len"],
                tensors["actual_seq_lens"],
                tensors["head_nums"],
                tensors["input_layout"],
                q_rotation=tensors["q_rotation"],
                k_rotation=tensors["k_rotation"],
                v_scale=tensors["v_scale"],
                epsilon=tensors["epsilon"],
            )
            outputs = (q_out, q_scale, tensors["k_cache"], tensors["v_cache"], tensors["k_scale_cache"])
        elif mode == "functional":
            outputs = qkv_rms_norm_rope_cache_with_k_scale_functional(
                tensors["qkv"],
                tensors["q_gamma"],
                tensors["k_gamma"],
                tensors["cos_sin"],
                tensors["slot_mapping"],
                tensors["k_cache"],
                tensors["v_cache"],
                tensors["k_scale_cache"],
                tensors["cu_seq_len"],
                tensors["actual_seq_lens"],
                tensors["head_nums"],
                tensors["input_layout"],
                q_rotation=tensors["q_rotation"],
                k_rotation=tensors["k_rotation"],
                v_scale=tensors["v_scale"],
                epsilon=tensors["epsilon"],
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

    torch.npu.synchronize()
    assert_outputs(case, outputs)
    q_out, q_scale, k_cache, v_cache, k_scale_cache = outputs
    print(
        f"[PASS] case={case.name} mode={mode} layout={case.input_layout} "
        f"q_out={tuple(q_out.shape)}/{q_out.dtype} q_scale={tuple(q_scale.shape)}/{q_scale.dtype} "
        f"k_cache={tuple(k_cache.shape)} v_cache={tuple(v_cache.shape)} "
        f"k_scale_cache={tuple(k_scale_cache.shape)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="case name to run; may be repeated; default: all")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--mode", choices=("inplace", "functional", "both"), default="both")
    parser.add_argument("--device", default="npu")
    return parser.parse_args()


def main():
    args = parse_args()
    case_map = {case.name: case for case in CASES}
    if args.list_cases:
        print("\n".join(case_map))
        return 0

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is required to run this example")

    selected_names: List[str] = args.case or list(case_map)
    unknown = sorted(set(selected_names) - set(case_map))
    if unknown:
        raise ValueError(f"unknown case(s): {unknown}; available: {list(case_map)}")

    modes = ("inplace", "functional") if args.mode == "both" else (args.mode,)
    for case_name in selected_names:
        for mode in modes:
            run_case(case_map[case_name], mode, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
