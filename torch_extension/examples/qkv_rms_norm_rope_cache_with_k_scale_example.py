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
"""Run and compare cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale.

The default run covers generated inplace cases across NTD/TND layouts, Nq/Nk/Nv
combinations, multi-batch prefix sums, non-contiguous slots, actual_seq_lens
history offsets, T axes greater than 5k, and strided cache views. Each case runs
the torch NPU interface and prints match rates against a CPU golden
implementation.

Example:
    python3 torch_extension/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py --list-cases
    python3 torch_extension/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py
    python3 torch_extension/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py --case n16_kv2_offset_b0_ntd
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch_npu

TORCH_EXTENSION_DIR = Path(__file__).resolve().parents[1]
if str(TORCH_EXTENSION_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_EXTENSION_DIR))

HEAD_SIZE = 128
DEFAULT_EPSILON = 1e-6
FP8_E4M3FN_MAX = 448.0


@dataclass(frozen=True)
class Case:
    name: str
    batch_seq_lens: Tuple[int, ...]
    actual_seq_lens: Tuple[int, ...]
    n_q: int
    n_k: int
    n_v: int
    input_layout: str
    slot_pattern: str = "linear"
    seed: int = 0
    block_size: int = 16
    block_num: int = 8
    cache_stride: str = "contiguous"
    cache_pad: int = 0


@dataclass(frozen=True)
class DiffSample:
    index: Tuple[int, ...]
    actual: float
    expected: float
    diff: float


@dataclass(frozen=True)
class MatchResult:
    name: str
    matched: int
    total: int
    rate: float
    max_abs_diff: float
    mean_abs_diff: float
    max_diff: float
    max_actual: float
    max_expected: float
    diff_samples: Tuple[DiffSample, ...]


def build_cases() -> Tuple[Case, ...]:
    n16_specs = (
        ("decode1", (1,), (1,), "linear"),
        ("prefill8", (8,), (8,), "reverse"),
        ("prefill17", (17,), (17,), "stride3"),
        ("prefill33", (33,), (33,), "random"),
        ("mb_2_7", (2, 7), (2, 7), "linear"),
        ("offset_b0", (17, 1), (8209, 1), "stride3"),
        ("offset_b1", (1, 17), (1, 8209), "reverse"),
        ("offset_multi", (3, 5, 2), (4099, 5, 8194), "random"),
    )
    n8_specs = (
        ("decode1", (1,), (1,), "linear"),
        ("prefill8", (8,), (8,), "reverse"),
        ("prefill17", (17,), (17,), "stride3"),
        ("mb_2_7", (2, 7), (2, 7), "linear"),
        ("offset_b0", (17, 1), (8209, 1), "stride3"),
        ("offset_b1", (1, 17), (1, 8209), "reverse"),
        ("offset_multi", (3, 5, 2), (4099, 5, 8194), "random"),
    )
    n16_large_strided_specs = (
        ("long5379_dpad", (5379,), (5379,), "random", 512),
        ("long_offset_mb3217_2891_dpad", (3217, 2891), (11397, 3589), "random", 512),
        ("large96_dpad", (96,), (96,), "reverse", 32),
        ("large127_dpad", (127,), (127,), "random", 32),
        ("large128_stride3_dpad", (128,), (128,), "stride3", 32),
        ("large_mb32_64_dpad", (32, 64), (32, 64), "linear", 32),
        ("large_mb17_79_3_dpad", (17, 79, 3), (17, 79, 3), "random", 32),
        ("large_offset_multi_dpad", (33, 17, 49), (4129, 17, 8241), "random", 32),
    )
    n8_large_strided_specs = (
        ("long5863_dpad", (5863,), (5863,), "random", 512),
        ("long_offset_mb2753_3401_dpad", (2753, 3401), (10945, 7509), "random", 512),
        ("large96_dpad", (96,), (96,), "reverse", 32),
        ("large127_dpad", (127,), (127,), "random", 32),
        ("large_mb32_64_dpad", (32, 64), (32, 64), "linear", 32),
        ("large_mb17_79_3_dpad", (17, 79, 3), (17, 79, 3), "random", 32),
        ("large_offset_multi_dpad", (33, 17, 49), (4129, 17, 8241), "random", 32),
    )
    n64_specs = (
        ("decode1", (1,), (1,), "linear", 8, "contiguous", 0),
        ("prefill19", (19,), (19,), "reverse", 8, "contiguous", 0),
        ("prefill73", (73,), (73,), "random", 8, "contiguous", 0),
        ("mb5_11_23", (5, 11, 23), (5, 11, 23), "stride3", 8, "contiguous", 0),
        ("offset_b0", (17, 1), (8209, 1), "stride3", 8, "contiguous", 0),
        ("offset_b1", (1, 29), (1, 8221), "reverse", 8, "contiguous", 0),
        ("offset_multi", (7, 13, 5), (4103, 13, 8197), "random", 8, "contiguous", 0),
        ("long5819_dpad", (5819,), (5819,), "random", 512, "d_pad", 8),
    )
    n64_long_16k_specs = (
        ("long16384_dpad", (16384,), (16384,), "random", 1024, "d_pad", 8),
    )
    n16_long_16k_specs = (
        ("long16384_dpad", (16384,), (16384,), "random", 1024, "d_pad", 8),
    )
    n8_long_16k_specs = (
        ("long16384_dpad", (16384,), (16384,), "random", 1024, "d_pad", 8),
    )

    cases: List[Case] = []
    seed = 20260623
    for heads_name, specs, heads in (("n16_kv2", n16_specs, (16, 2, 2)), ("n8_kv1", n8_specs, (8, 1, 1))):
        for suffix, seq_lens, actual_lens, slot_pattern in specs:
            for layout in ("NTD", "TND"):
                cases.append(
                    Case(
                        name=f"{heads_name}_{suffix}_{layout.lower()}",
                        batch_seq_lens=seq_lens,
                        actual_seq_lens=actual_lens,
                        n_q=heads[0],
                        n_k=heads[1],
                        n_v=heads[2],
                        input_layout=layout,
                        slot_pattern=slot_pattern,
                        seed=seed,
                    )
                )
                seed += 1
    for heads_name, specs, heads in (
        ("n16_kv2", n16_large_strided_specs, (16, 2, 2)),
        ("n8_kv1", n8_large_strided_specs, (8, 1, 1)),
    ):
        for suffix, seq_lens, actual_lens, slot_pattern, block_num in specs:
            for layout in ("NTD", "TND"):
                cases.append(
                    Case(
                        name=f"{heads_name}_{suffix}_{layout.lower()}",
                        batch_seq_lens=seq_lens,
                        actual_seq_lens=actual_lens,
                        n_q=heads[0],
                        n_k=heads[1],
                        n_v=heads[2],
                        input_layout=layout,
                        slot_pattern=slot_pattern,
                        seed=seed,
                        block_num=block_num,
                        cache_stride="d_pad",
                        cache_pad=8,
                    )
                )
                seed += 1
    for suffix, seq_lens, actual_lens, slot_pattern, block_num, cache_stride, cache_pad in n64_specs:
        for layout in ("NTD", "TND"):
            cases.append(
                Case(
                    name=f"n64_kv8_{suffix}_{layout.lower()}",
                    batch_seq_lens=seq_lens,
                    actual_seq_lens=actual_lens,
                    n_q=64,
                    n_k=8,
                    n_v=8,
                    input_layout=layout,
                    slot_pattern=slot_pattern,
                    seed=seed,
                    block_num=block_num,
                    cache_stride=cache_stride,
                    cache_pad=cache_pad,
                )
            )
            seed += 1
    for suffix, seq_lens, actual_lens, slot_pattern, block_num, cache_stride, cache_pad in n64_long_16k_specs:
        for layout in ("NTD", "TND"):
            cases.append(
                Case(
                    name=f"n64_kv8_{suffix}_{layout.lower()}",
                    batch_seq_lens=seq_lens,
                    actual_seq_lens=actual_lens,
                    n_q=64,
                    n_k=8,
                    n_v=8,
                    input_layout=layout,
                    slot_pattern=slot_pattern,
                    seed=seed,
                    block_num=block_num,
                    cache_stride=cache_stride,
                    cache_pad=cache_pad,
                )
            )
            seed += 1
    for suffix, seq_lens, actual_lens, slot_pattern, block_num, cache_stride, cache_pad in n16_long_16k_specs:
        for layout in ("NTD", "TND"):
            cases.append(
                Case(
                    name=f"n16_kv2_{suffix}_{layout.lower()}",
                    batch_seq_lens=seq_lens,
                    actual_seq_lens=actual_lens,
                    n_q=16,
                    n_k=2,
                    n_v=2,
                    input_layout=layout,
                    slot_pattern=slot_pattern,
                    seed=seed,
                    block_num=block_num,
                    cache_stride=cache_stride,
                    cache_pad=cache_pad,
                )
            )
            seed += 1
    for suffix, seq_lens, actual_lens, slot_pattern, block_num, cache_stride, cache_pad in n8_long_16k_specs:
        for layout in ("NTD", "TND"):
            cases.append(
                Case(
                    name=f"n8_kv1_{suffix}_{layout.lower()}",
                    batch_seq_lens=seq_lens,
                    actual_seq_lens=actual_lens,
                    n_q=8,
                    n_k=1,
                    n_v=1,
                    input_layout=layout,
                    slot_pattern=slot_pattern,
                    seed=seed,
                    block_num=block_num,
                    cache_stride=cache_stride,
                    cache_pad=cache_pad,
                )
            )
            seed += 1
    return tuple(cases)


CASES = build_cases()


def make_cu_seq_len(seq_lens: Iterable[int]) -> torch.Tensor:
    prefix = [0]
    for seq_len in seq_lens:
        prefix.append(prefix[-1] + int(seq_len))
    return torch.tensor(prefix, dtype=torch.int32)


def make_slot_mapping(case: Case, total_tokens: int, generator: torch.Generator) -> torch.Tensor:
    capacity = case.block_num * case.block_size
    if total_tokens > capacity:
        raise ValueError(f"{case.name}: total tokens exceed cache slot capacity")

    token_ids = torch.arange(total_tokens, dtype=torch.int64)
    if case.slot_pattern == "linear":
        slots = token_ids
    elif case.slot_pattern == "reverse":
        slots = torch.arange(total_tokens - 1, -1, -1, dtype=torch.int64)
    elif case.slot_pattern == "stride3":
        slots = token_ids * 3 + 1
    elif case.slot_pattern == "random":
        slots = torch.randperm(capacity, generator=generator, dtype=torch.int64)[:total_tokens]
    else:
        raise ValueError(f"{case.name}: unknown slot pattern {case.slot_pattern}")

    if int(slots.max().item()) >= capacity or torch.unique(slots).numel() != total_tokens:
        raise ValueError(f"{case.name}: slot pattern must produce unique slots within cache capacity")
    return slots.to(torch.int32)


def validate_case(case: Case) -> None:
    if case.input_layout not in ("NTD", "TND"):
        raise ValueError(f"{case.name}: input_layout must be NTD or TND")
    if case.n_k != case.n_v:
        raise ValueError(f"{case.name}: Nk and Nv must match")
    if len(case.actual_seq_lens) != len(case.batch_seq_lens):
        raise ValueError(f"{case.name}: actual_seq_lens length must match batch_seq_lens length")
    if any(actual_len < batch_len for batch_len, actual_len in zip(case.batch_seq_lens, case.actual_seq_lens)):
        raise ValueError(f"{case.name}: actual_seq_lens values must be >= current batch token counts")
    if case.cache_stride not in ("contiguous", "d_pad"):
        raise ValueError(f"{case.name}: cache_stride must be contiguous or d_pad")
    if case.cache_stride == "d_pad" and case.cache_pad <= 0:
        raise ValueError(f"{case.name}: cache_pad must be positive for d_pad cache stride")


def storage_numel(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> int:
    return sum((dim - 1) * step for dim, step in zip(shape, stride)) + 1


def cache_view_stride(shape: Tuple[int, int, int, int], cache_stride: str, cache_pad: int) -> Tuple[int, int, int, int]:
    _, heads, block_size, dim = shape
    if cache_stride == "contiguous":
        token_stride = dim
    elif cache_stride == "d_pad":
        token_stride = dim + cache_pad
    else:
        raise ValueError(f"unknown cache_stride: {cache_stride}")
    return (heads * block_size * token_stride, block_size * token_stride, token_stride, 1)


def make_zero_cache_view(shape: Tuple[int, int, int, int], dtype: torch.dtype, device: str,
                         cache_stride: str, cache_pad: int) -> torch.Tensor:
    if cache_stride == "contiguous":
        if dtype == torch.float8_e4m3fn:
            return torch.zeros(shape, dtype=torch.float32, device=device).to(dtype)
        return torch.zeros(shape, dtype=dtype, device=device)

    stride = cache_view_stride(shape, cache_stride, cache_pad)
    numel = storage_numel(shape, stride)
    if dtype == torch.float8_e4m3fn:
        storage = torch.zeros((numel,), dtype=torch.float32, device=device).to(dtype)
    else:
        storage = torch.zeros((numel,), dtype=dtype, device=device)
    return torch.as_strided(storage, size=shape, stride=stride)


def make_cpu_inputs(case: Case) -> Dict[str, object]:
    validate_case(case)
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    total_tokens = sum(case.batch_seq_lens)
    total_heads = case.n_q + case.n_k + case.n_v
    qkv_shape = (total_tokens, total_heads, HEAD_SIZE)
    if case.input_layout == "NTD":
        qkv_shape = (total_heads, total_tokens, HEAD_SIZE)

    qkv = (torch.randn(qkv_shape, generator=generator, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    q_gamma = torch.rand((HEAD_SIZE,), generator=generator, dtype=torch.float32) + 0.5
    k_gamma = torch.rand((HEAD_SIZE,), generator=generator, dtype=torch.float32) + 0.5
    cos_sin = torch.randn((max(case.actual_seq_lens), HEAD_SIZE), generator=generator, dtype=torch.float32)
    slot_mapping = make_slot_mapping(case, total_tokens, generator)
    k_cache = torch.zeros((case.block_num, case.n_k, case.block_size, HEAD_SIZE), dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    v_cache = torch.zeros((case.block_num, case.n_v, case.block_size, HEAD_SIZE), dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_scale_cache = torch.zeros((case.block_num, case.n_k, case.block_size, 1), dtype=torch.float32)
    cu_seq_len = make_cu_seq_len(case.batch_seq_lens)
    actual_seq_lens = torch.tensor(case.actual_seq_lens, dtype=torch.int32)
    q_rotation = torch.eye(HEAD_SIZE, dtype=torch.float32).to(torch.bfloat16)
    k_rotation = torch.eye(HEAD_SIZE, dtype=torch.float32).to(torch.bfloat16)
    v_scale = torch.rand((case.n_v,), generator=generator, dtype=torch.float32) + 0.5

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
        "head_nums": [case.n_q, case.n_k, case.n_v],
        "input_layout": case.input_layout,
        "q_rotation": q_rotation,
        "k_rotation": k_rotation,
        "v_scale": v_scale,
        "epsilon": DEFAULT_EPSILON,
    }


def to_npu_inputs(cpu_inputs: Dict[str, object], case: Case, device: str) -> Dict[str, object]:
    npu_inputs: Dict[str, object] = {}
    for name, value in cpu_inputs.items():
        if name in ("k_cache", "v_cache", "k_scale_cache"):
            continue
        if isinstance(value, torch.Tensor):
            npu_inputs[name] = value.to(device)
        else:
            npu_inputs[name] = value
    for name in ("k_cache", "v_cache", "k_scale_cache"):
        tensor = cpu_inputs[name]
        npu_inputs[name] = make_zero_cache_view(
            tuple(tensor.shape), tensor.dtype, device, case.cache_stride, case.cache_pad)
    return npu_inputs


def resolve_positions(cu_seq_len: torch.Tensor, actual_seq_lens: torch.Tensor) -> torch.Tensor:
    cu = cu_seq_len.to(torch.int64)
    actual = actual_seq_lens.to(torch.int64)
    total_tokens = int(cu[-1].item())
    positions = torch.empty((total_tokens,), dtype=torch.int64)
    for batch_idx in range(actual.numel()):
        begin = int(cu[batch_idx].item())
        end = int(cu[batch_idx + 1].item())
        local_len = end - begin
        start_pos = int(actual[batch_idx].item()) - local_len
        positions[begin:end] = torch.arange(start_pos, start_pos + local_len, dtype=torch.int64)
    return positions


def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor, cu_seq_len: torch.Tensor, actual_seq_lens: torch.Tensor):
    positions = resolve_positions(cu_seq_len, actual_seq_lens)
    if int(positions.min().item()) < 0 or int(positions.max().item()) >= cos_sin.shape[0]:
        raise ValueError("cos_sin does not cover all RoPE positions")

    half = x.shape[-1] // 2
    cos = cos_sin[positions, :half].to(torch.float32)
    sin = cos_sin[positions, half:].to(torch.float32)
    x_low = x[:, :, :half]
    x_high = x[:, :, half:]
    out_low = x_low * cos[:, None, :] - x_high * sin[:, None, :]
    out_high = x_high * cos[:, None, :] + x_low * sin[:, None, :]
    return torch.cat((out_low, out_high), dim=-1)


def dynamic_quant_fp8(x: torch.Tensor):
    max_abs = torch.amax(torch.abs(x), dim=-1)
    scale = max_abs / FP8_E4M3FN_MAX
    y = x / scale.unsqueeze(-1)
    return y.to(torch.float8_e4m3fn), scale.to(torch.float32)


def unravel_index(flat_idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    index = []
    for dim in reversed(shape):
        index.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(index))


def golden_outputs(cpu_inputs: Dict[str, object]):
    qkv = cpu_inputs["qkv"].to(torch.float32)
    q_gamma = cpu_inputs["q_gamma"].to(torch.float32)
    k_gamma = cpu_inputs["k_gamma"].to(torch.float32)
    cos_sin = cpu_inputs["cos_sin"].to(torch.float32)
    slot_mapping = cpu_inputs["slot_mapping"].to(torch.int64)
    k_cache = cpu_inputs["k_cache"].clone()
    v_cache = cpu_inputs["v_cache"].clone()
    k_scale_cache = cpu_inputs["k_scale_cache"].clone()
    cu_seq_len = cpu_inputs["cu_seq_len"]
    actual_seq_lens = cpu_inputs["actual_seq_lens"]
    q_rotation = cpu_inputs["q_rotation"].to(torch.float32)
    k_rotation = cpu_inputs["k_rotation"].to(torch.float32)
    v_scale = cpu_inputs["v_scale"].to(torch.float32)
    n_q, n_k, n_v = cpu_inputs["head_nums"]
    input_layout = cpu_inputs["input_layout"]
    epsilon = cpu_inputs["epsilon"]

    if input_layout == "TND":
        q = qkv[:, :n_q, :]
        k = qkv[:, n_q:n_q + n_k, :]
        v = qkv[:, n_q + n_k:n_q + n_k + n_v, :]
    else:
        q = qkv[:n_q, :, :].transpose(0, 1).contiguous()
        k = qkv[n_q:n_q + n_k, :, :].transpose(0, 1).contiguous()
        v = qkv[n_q + n_k:n_q + n_k + n_v, :, :].transpose(0, 1).contiguous()

    q_rms = torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) / HEAD_SIZE + epsilon)
    k_rms = torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) / HEAD_SIZE + epsilon)
    q_norm = q / q_rms * q_gamma[None, None, :]
    k_norm = k / k_rms * k_gamma[None, None, :]

    q_rope = apply_rope(q_norm, cos_sin, cu_seq_len, actual_seq_lens).to(torch.bfloat16).to(torch.float32)
    k_rope = apply_rope(k_norm, cos_sin, cu_seq_len, actual_seq_lens).to(torch.bfloat16).to(torch.float32)
    q_after_cube = q_rope @ q_rotation
    k_after_cube = k_rope @ k_rotation
    qk_after_cube = torch.cat((q_after_cube, k_after_cube), dim=1)
    qk_quant, scale = dynamic_quant_fp8(qk_after_cube)

    v_out = (v * v_scale[None, :, None]).to(torch.float8_e4m3fn)
    if input_layout == "TND":
        q_out = qk_quant[:, :n_q, :].contiguous()
        q_scale = scale[:, :n_q].contiguous()
    else:
        q_out = qk_quant[:, :n_q, :].transpose(0, 1).contiguous()
        q_scale = scale[:, :n_q].transpose(0, 1).contiguous()

    block_size = k_cache.shape[2]
    for token_idx, slot in enumerate(slot_mapping.tolist()):
        block_id = slot // block_size
        block_offset = slot % block_size
        k_cache[block_id, :, block_offset, :] = qk_quant[token_idx, n_q:n_q + n_k, :]
        k_scale_cache[block_id, :, block_offset, 0] = scale[token_idx, n_q:n_q + n_k]
        v_cache[block_id, :, block_offset, :] = v_out[token_idx, :, :]

    return {
        "q_out": q_out,
        "q_scale": q_scale,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale_cache": k_scale_cache,
    }


def run_inplace(cpu_inputs: Dict[str, object], case: Case, device: str):
    from cann_ops_transformer.ops import qkv_rms_norm_rope_cache_with_k_scale

    npu_inputs = to_npu_inputs(cpu_inputs, case, device)
    with torch.no_grad():
        q_out, q_scale = qkv_rms_norm_rope_cache_with_k_scale(
            npu_inputs["qkv"],
            npu_inputs["q_gamma"],
            npu_inputs["k_gamma"],
            npu_inputs["cos_sin"],
            npu_inputs["slot_mapping"],
            npu_inputs["k_cache"],
            npu_inputs["v_cache"],
            npu_inputs["k_scale_cache"],
            npu_inputs["cu_seq_len"],
            npu_inputs["actual_seq_lens"],
            npu_inputs["head_nums"],
            npu_inputs["input_layout"],
            q_rotation=npu_inputs["q_rotation"],
            k_rotation=npu_inputs["k_rotation"],
            v_scale=npu_inputs["v_scale"],
            epsilon=npu_inputs["epsilon"],
        )
    torch.npu.synchronize()
    return {
        "q_out": q_out,
        "q_scale": q_scale,
        "k_cache": npu_inputs["k_cache"],
        "v_cache": npu_inputs["v_cache"],
        "k_scale_cache": npu_inputs["k_scale_cache"],
    }


def compare_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor,
                   atol: float, rtol: float, max_diff_lines: int) -> MatchResult:
    actual_f32 = actual.detach().cpu().to(torch.float32)
    expected_f32 = expected.detach().cpu().to(torch.float32)
    if actual_f32.shape != expected_f32.shape:
        return MatchResult(
            name, 0, 1, 0.0, float("inf"), float("inf"),
            float("nan"), float("nan"), float("nan"), tuple())

    close = torch.isclose(actual_f32, expected_f32, atol=atol, rtol=rtol, equal_nan=True)
    matched = int(torch.count_nonzero(close).item())
    total = actual_f32.numel()
    finite = torch.isfinite(actual_f32) & torch.isfinite(expected_f32)
    if torch.any(finite):
        signed_diff = actual_f32[finite] - expected_f32[finite]
        abs_diff = torch.abs(signed_diff)
        max_pos = int(torch.argmax(abs_diff).item())
        max_abs_diff = float(abs_diff[max_pos].item())
        mean_abs_diff = float(torch.mean(abs_diff).item())
        max_diff = float(signed_diff[max_pos].item())
        max_actual = float(actual_f32[finite][max_pos].item())
        max_expected = float(expected_f32[finite][max_pos].item())
    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
        max_diff = 0.0
        max_actual = 0.0
        max_expected = 0.0
    diff_samples: List[DiffSample] = []
    if max_diff_lines > 0 and matched != total:
        actual_flat = actual_f32.reshape(-1)
        expected_flat = expected_f32.reshape(-1)
        mismatch_indices = torch.nonzero((~close).reshape(-1), as_tuple=False).flatten()[:max_diff_lines]
        shape = tuple(actual_f32.shape)
        for flat_idx_tensor in mismatch_indices:
            flat_idx = int(flat_idx_tensor.item())
            actual_value = float(actual_flat[flat_idx].item())
            expected_value = float(expected_flat[flat_idx].item())
            diff_samples.append(
                DiffSample(
                    index=unravel_index(flat_idx, shape),
                    actual=actual_value,
                    expected=expected_value,
                    diff=actual_value - expected_value,
                )
            )
    return MatchResult(
        name, matched, total, matched / total if total else 1.0,
        max_abs_diff, mean_abs_diff, max_diff, max_actual, max_expected, tuple(diff_samples))


def select_touched_cache_rows(outputs: Dict[str, torch.Tensor], slot_mapping: torch.Tensor) -> Dict[str, torch.Tensor]:
    slots = slot_mapping.detach().cpu().to(torch.int64)
    block_size = outputs["k_cache"].shape[2]
    block_ids = slots // block_size
    block_offsets = slots % block_size
    k_cache_f32 = outputs["k_cache"].detach().cpu().to(torch.float32)
    v_cache_f32 = outputs["v_cache"].detach().cpu().to(torch.float32)
    k_scale_cache_f32 = outputs["k_scale_cache"].detach().cpu().to(torch.float32)
    return {
        "k_cache_touched": k_cache_f32[block_ids, :, block_offsets, :],
        "v_cache_touched": v_cache_f32[block_ids, :, block_offsets, :],
        "k_scale_cache_touched": k_scale_cache_f32[block_ids, :, block_offsets, 0],
    }


def compare_outputs(actual: Dict[str, torch.Tensor], expected: Dict[str, torch.Tensor],
                    cpu_inputs: Dict[str, object], args):
    actual_touched = select_touched_cache_rows(actual, cpu_inputs["slot_mapping"])
    expected_touched = select_touched_cache_rows(expected, cpu_inputs["slot_mapping"])

    results = [
        compare_tensor(
            "q_out", actual["q_out"], expected["q_out"], args.fp8_atol, args.fp8_rtol, args.max_diff_lines),
        compare_tensor("k_cache_touched", actual_touched["k_cache_touched"],
                       expected_touched["k_cache_touched"], args.fp8_atol, args.fp8_rtol, args.max_diff_lines),
        compare_tensor("v_cache_touched", actual_touched["v_cache_touched"],
                       expected_touched["v_cache_touched"], args.fp8_atol, args.fp8_rtol, args.max_diff_lines),
        compare_tensor(
            "q_scale", actual["q_scale"], expected["q_scale"], args.float_atol, args.float_rtol, args.max_diff_lines),
        compare_tensor("k_scale_cache_touched", actual_touched["k_scale_cache_touched"],
                       expected_touched["k_scale_cache_touched"],
                       args.float_atol, args.float_rtol, args.max_diff_lines),
    ]
    total = sum(result.total for result in results)
    matched = sum(result.matched for result in results)
    overall = matched / total if total else 1.0
    return results, overall


def expected_shapes(case: Case):
    total_tokens = sum(case.batch_seq_lens)
    if case.input_layout == "TND":
        return (total_tokens, case.n_q, HEAD_SIZE), (total_tokens, case.n_q)
    return (case.n_q, total_tokens, HEAD_SIZE), (case.n_q, total_tokens)


def assert_shapes(case: Case, outputs: Dict[str, torch.Tensor]):
    q_out_shape, q_scale_shape = expected_shapes(case)
    assert tuple(outputs["q_out"].shape) == q_out_shape, (
        f"{case.name}: q_out shape mismatch: {tuple(outputs['q_out'].shape)} vs {q_out_shape}")
    assert tuple(outputs["q_scale"].shape) == q_scale_shape, (
        f"{case.name}: q_scale shape mismatch: {tuple(outputs['q_scale'].shape)} vs {q_scale_shape}")
    assert outputs["q_out"].dtype == torch.float8_e4m3fn, f"{case.name}: q_out dtype mismatch: {outputs['q_out'].dtype}"
    assert outputs["q_scale"].dtype == torch.float32, f"{case.name}: q_scale dtype mismatch: {outputs['q_scale'].dtype}"


def print_case_results(case: Case, results: List[MatchResult]) -> None:
    print(
        f"[CASE] {case.name} layout={case.input_layout} heads=({case.n_q},{case.n_k},{case.n_v}) "
        f"seq={case.batch_seq_lens} actual_seq={case.actual_seq_lens} slot={case.slot_pattern} "
        f"cache_stride={case.cache_stride} cache_pad={case.cache_pad}"
    )
    for result in results:
        print(
            f"  [RESULT] {result.name}: match={result.rate * 100.0:.2f}% "
            f"matched={result.matched}/{result.total} "
            f"max_abs_diff={result.max_abs_diff:.4g} mean_abs_diff={result.mean_abs_diff:.4g} "
            f"max_diff={result.max_diff:.4g} "
            f"max_diff_actual={result.max_actual:.4g} max_diff_expected={result.max_expected:.4g}"
        )
        for diff_idx, sample in enumerate(result.diff_samples):
            print(
                f"    [DIFF {diff_idx:02d}] index={sample.index} "
                f"actual={sample.actual:.4g} expected={sample.expected:.4g} diff={sample.diff:.4g}"
            )


def run_case(case: Case, args) -> float:
    cpu_inputs = make_cpu_inputs(case)
    expected = golden_outputs(cpu_inputs)
    actual = run_inplace(cpu_inputs, case, args.device)
    assert_shapes(case, actual)
    results, overall = compare_outputs(actual, expected, cpu_inputs, args)
    print_case_results(case, results)
    for result in results:
        if result.rate < args.min_match_rate:
            raise AssertionError(
                f"{case.name}: {result.name} golden match rate {result.rate:.6f} "
                f"is below threshold {args.min_match_rate:.6f}")
    return overall


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append",
                        help=f"case name to run; may be repeated; default: all {len(CASES)} cases")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--min-match-rate", type=float, default=0.99)
    parser.add_argument("--fp8-atol", type=float, default=1.0)
    parser.add_argument("--fp8-rtol", type=float, default=0.0)
    parser.add_argument("--float-atol", type=float, default=1e-2)
    parser.add_argument("--float-rtol", type=float, default=1e-2)
    parser.add_argument("--max-diff-lines", type=int, default=20,
                        help="maximum mismatch detail lines to print per output; set 0 to disable")
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

    for case_name in selected_names:
        run_case(case_map[case_name], args)
    print(f"[SUMMARY] cases={len(selected_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
