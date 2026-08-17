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
"""Run and compare cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_.

The default run covers generated inplace cases across NTD/TND input layouts,
TND input to NTD output layout conversion with varied T axes, Nq/Nk/Nv
combinations, multi-batch prefix sums, non-contiguous slots, seq_lens
history offsets, T axes greater than 5k, and strided cache views. Each case
runs the torch NPU interface and prints match rates against a CPU golden
implementation.

Example:
    python3 designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py --list-cases
    python3 designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py
    python3 designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py --case n16_kv2_offset_b0_ntd
"""

import argparse
from dataclasses import dataclass, replace
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch_npu

HEAD_SIZE = 128
DEFAULT_EPSILON = 1e-6
FP8_E4M3FN_MAX = 448.0
INT8_MAX_ABS_DIFF = 1
MROPE_POSITION_SEED_OFFSET = 1000003
ROTATION_SEED_OFFSET = 2000003
MROPE_T_INVARIANT_SECTIONS = ((0, 20, 20), (1, 20, 20), (11, 20, 20), (24, 20, 20))
MROPE_SMOKE_SECTION = MROPE_T_INVARIANT_SECTIONS[2]
MROPE_PRECISION_SECTIONS = (
    (0, 1, 0),
    (0, 0, 1),
    MROPE_SMOKE_SECTION,
    (0, 1, 1),
    (0, 1, 2),
    (0, 2, 1),
    (0, 21, 0),
    (0, 0, 21),
    (0, 21, 21),
    (1, 21, 21),
    (22, 21, 21),
    (22, 21, 0),
    (22, 0, 21),
    (0, 20, 21),
    (0, 21, 20),
    (0, 9, 11),
    (0, 4, 12),
    (0, 12, 4),
    (7, 9, 11),
    (22, 20, 21),
    (22, 21, 20),
)
MROPE_PERFORMANCE_SECTION = (22, 21, 21)
MROPE_T_INVARIANT_BASE_NAME = "n16_kv2_mrope_prefill17_tnd"


@dataclass(frozen=True)
class Case:
    name: str
    batch_seq_lens: Tuple[int, ...]
    seq_lens: Tuple[int, ...]
    n_q: int
    n_k: int
    n_v: int
    layout_qkv: str
    layout_q_out: str = ""
    slot_pattern: str = "linear"
    seed: int = 0
    block_size: int = 16
    block_num: int = 8
    cache_stride: str = "contiguous"
    cache_pad: int = 0
    omit_layout_args: bool = False
    explicit_empty_mrope_section: bool = False
    # A non-empty section selects the M-RoPE scene.  The position tensor is
    # generated from the token count and passed separately as [T, 3].
    mrope_section: Tuple[int, int, int] = ()


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
    strict_all_elements: bool = False


def is_mrope_case(case: Case) -> bool:
    return bool(case.mrope_section)


def with_tnd_to_ntd_cases(cases: List[Case]) -> Tuple[Case, ...]:
    extended = list(cases)
    for index, case in enumerate(cases):
        if case.layout_qkv != "TND" or case.layout_q_out or is_mrope_case(case):
            continue
        if not case.name.endswith("_tnd"):
            raise ValueError(f"{case.name}: TND base case name must end with _tnd")

        varied_batch_seq_lens = make_varied_tnd_to_ntd_batch_seq_lens(case, index)
        history_offsets = tuple(actual - current for current, actual in zip(case.batch_seq_lens, case.seq_lens))
        varied_seq_lens = tuple(current + offset for current, offset in zip(varied_batch_seq_lens, history_offsets))
        extended.append(
            replace(
                case,
                name=f"{case.name[:-4]}_tnd_to_ntd",
                batch_seq_lens=varied_batch_seq_lens,
                seq_lens=varied_seq_lens,
                layout_q_out="NTD",
                seed=case.seed + 100000,
            )
        )
    return tuple(extended)


def is_mrope_t_invariant_case(case: Case) -> bool:
    return case.name == MROPE_T_INVARIANT_BASE_NAME or case.name.startswith(
        f"{MROPE_T_INVARIANT_BASE_NAME[:-4]}_t_section_"
    )


def to_mrope_case(case: Case, mrope_section: Tuple[int, int, int]) -> Case:
    if is_mrope_case(case) or case.explicit_empty_mrope_section:
        raise ValueError(f"{case.name}: only a normal RoPE case can be mirrored to M-RoPE")
    if case.layout_qkv != "TND" or (case.layout_q_out or case.layout_qkv) != "TND":
        raise ValueError(f"{case.name}: M-RoPE mirror requires a RoPE TND->TND case")

    name_parts = case.name.split("_", 2)
    if len(name_parts) != 3:
        raise ValueError(f"{case.name}: case name must start with head-count components")
    return replace(
        case,
        name=f"{name_parts[0]}_{name_parts[1]}_mrope_{name_parts[2]}",
        explicit_empty_mrope_section=False,
        mrope_section=mrope_section,
    )


def max_slot_tokens(case: Case) -> int:
    capacity = case.block_num * case.block_size
    if case.slot_pattern == "stride3":
        return (capacity - 2) // 3 + 1
    return capacity


def make_varied_tnd_to_ntd_batch_seq_lens(case: Case, index: int) -> Tuple[int, ...]:
    old_total = sum(case.batch_seq_lens)
    min_total = len(case.batch_seq_lens)
    max_total = max_slot_tokens(case)
    if min_total > max_total:
        raise ValueError(f"{case.name}: minimum token count exceeds slot capacity")

    if old_total >= 1024:
        deltas = (17, -29, 53, -97, 127, -191, 257, -389)
    elif old_total >= 96:
        deltas = (5, -7, 11, -13, 17, -19)
    else:
        deltas = (1, 3, 5, 7, -1, -3, 9, -5)

    new_total = old_total
    for offset in range(len(deltas)):
        delta = deltas[(index + offset) % len(deltas)]
        candidate = min(max(old_total + delta, min_total), max_total)
        if candidate != old_total:
            new_total = candidate
            break
    if new_total == old_total and old_total < max_total:
        new_total = old_total + 1
    elif new_total == old_total and old_total > min_total:
        new_total = old_total - 1

    varied = list(case.batch_seq_lens)
    delta = new_total - old_total
    cursor = index % len(varied)
    while delta > 0:
        varied[cursor % len(varied)] += 1
        cursor += 1
        delta -= 1
    while delta < 0:
        position = cursor % len(varied)
        if varied[position] > 1:
            varied[position] -= 1
            delta += 1
        cursor += 1
    return tuple(varied)


def performance_t16384_tnd_to_ntd_cases(seed: int) -> Tuple[Case, ...]:
    specs = (
        ("n64_kv8_perf_t16384_dpad_tnd_to_ntd", (64, 8, 8)),
        ("n16_kv2_perf_t16384_dpad_tnd_to_ntd", (16, 2, 2)),
        ("n8_kv1_perf_t16384_dpad_tnd_to_ntd", (8, 1, 1)),
    )
    return tuple(
        Case(
            name=name,
            batch_seq_lens=(16384,),
            seq_lens=(16384,),
            n_q=heads[0],
            n_k=heads[1],
            n_v=heads[2],
            layout_qkv="TND",
            layout_q_out="NTD",
            slot_pattern="random",
            seed=seed + index,
            block_num=1024,
            cache_stride="d_pad",
            cache_pad=8,
        )
        for index, (name, heads) in enumerate(specs)
    )


def performance_t65536_n16_kv2_cases(seed: int) -> Tuple[Case, ...]:
    specs = (
        ("n16_kv2_perf_t65536_dpad_tnd", "TND", "TND"),
        ("n16_kv2_perf_t65536_dpad_ntd", "NTD", "NTD"),
        ("n16_kv2_perf_t65536_dpad_tnd_to_ntd", "TND", "NTD"),
    )
    return tuple(
        Case(
            name=name,
            batch_seq_lens=(65536,),
            seq_lens=(65536,),
            n_q=16,
            n_k=2,
            n_v=2,
            layout_qkv=layout_qkv,
            layout_q_out=layout_q_out,
            slot_pattern="random",
            seed=seed + index,
            block_num=4096,
            cache_stride="d_pad",
            cache_pad=8,
        )
        for index, (name, layout_qkv, layout_q_out) in enumerate(specs)
    )


def performance_t131072_n16_kv2_case(seed: int) -> Case:
    """Return the paired 128K TND performance base case."""
    return Case(
        name="n16_kv2_perf_t131072_dpad_tnd",
        batch_seq_lens=(131072,),
        seq_lens=(131072,),
        n_q=16,
        n_k=2,
        n_v=2,
        layout_qkv="TND",
        layout_q_out="TND",
        slot_pattern="random",
        seed=seed,
        block_num=8192,
        cache_stride="d_pad",
        cache_pad=8,
    )


def mrope_large_output_case(seed: int) -> Case:
    """Keep a large-output M-RoPE precision and multi-tile regression."""
    return Case(
        name="n16_kv2_mrope_perf_t45000_tnd",
        batch_seq_lens=(45000,),
        seq_lens=(45000,),
        n_q=16,
        n_k=2,
        n_v=2,
        layout_qkv="TND",
        layout_q_out="TND",
        slot_pattern="random",
        seed=seed,
        block_num=2813,
        mrope_section=MROPE_PERFORMANCE_SECTION,
    )


def performance_t5379_n16_kv2_cases(seed: int) -> Tuple[Case, ...]:
    """Return the single-batch, approximately-6k TND performance case."""
    return (
        Case(
            name="n16_kv2_perf_t5379_dpad_tnd",
            batch_seq_lens=(5379,),
            seq_lens=(5379,),
            n_q=16,
            n_k=2,
            n_v=2,
            layout_qkv="TND",
            layout_q_out="TND",
            slot_pattern="random",
            seed=seed,
            block_num=512,
            cache_stride="d_pad",
            cache_pad=8,
        ),
    )


def precision_mb20_t16384_cases(seed: int) -> Tuple[Case, ...]:
    batch_seq_lens = (17, 257, 1, 1024, 511, 2048, 33, 4096, 128, 777,
                      64, 1536, 999, 2, 301, 819, 1234, 455, 688, 1394)
    history_offsets = (4096, 0, 8192, 512, 17, 2048, 0, 4096, 128, 3000,
                       0, 1024, 4096, 8192, 513, 0, 256, 2048, 1, 4096)
    seq_lens = tuple(seq_len + offset for seq_len, offset in zip(batch_seq_lens, history_offsets))
    specs = (
        ("n16_kv2_mb20_t16384_dpad_ntd", "NTD", ""),
        ("n16_kv2_mb20_t16384_dpad_tnd", "TND", ""),
        ("n16_kv2_mb20_t16384_dpad_tnd_to_ntd", "TND", "NTD"),
    )
    return tuple(
        Case(
            name=name,
            batch_seq_lens=batch_seq_lens,
            seq_lens=seq_lens,
            n_q=16,
            n_k=2,
            n_v=2,
            layout_qkv=layout_qkv,
            layout_q_out=layout_q_out,
            slot_pattern="random",
            seed=seed + index,
            block_num=1024,
            cache_stride="d_pad",
            cache_pad=8,
        )
        for index, (name, layout_qkv, layout_q_out) in enumerate(specs)
    )


def default_layout_case(seed: int) -> Case:
    return Case(
        name="n16_kv2_default_layout_tnd_to_ntd",
        batch_seq_lens=(19,),
        seq_lens=(19,),
        n_q=16,
        n_k=2,
        n_v=2,
        layout_qkv="TND",
        layout_q_out="NTD",
        slot_pattern="random",
        seed=seed,
        block_num=8,
        omit_layout_args=True,
    )


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
                        seq_lens=actual_lens,
                        n_q=heads[0],
                        n_k=heads[1],
                        n_v=heads[2],
                        layout_qkv=layout,
                        slot_pattern=slot_pattern,
                        seed=seed,
                    )
                )
                seed += 1

    # Exercise the framework convention that an explicitly provided empty
    # mrope_section does not select M-RoPE and remains a normal RoPE call.
    cases.append(
        Case(
            name="n16_kv2_prefill17_empty_mrope_section_tnd",
            batch_seq_lens=(17,),
            seq_lens=(17,),
            n_q=16,
            n_k=2,
            n_v=2,
            layout_qkv="TND",
            layout_q_out="TND",
            slot_pattern="stride3",
            seed=seed,
            block_num=8,
            explicit_empty_mrope_section=True,
        )
    )
    seed += 1

    mrope_smoke_source = next(case for case in cases if case.name == "n16_kv2_prefill17_tnd")
    cases.append(to_mrope_case(mrope_smoke_source, MROPE_SMOKE_SECTION))
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
                        seq_lens=actual_lens,
                        n_q=heads[0],
                        n_k=heads[1],
                        n_v=heads[2],
                        layout_qkv=layout,
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
                    seq_lens=actual_lens,
                    n_q=64,
                    n_k=8,
                    n_v=8,
                    layout_qkv=layout,
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
                    seq_lens=actual_lens,
                    n_q=64,
                    n_k=8,
                    n_v=8,
                    layout_qkv=layout,
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
                    seq_lens=actual_lens,
                    n_q=16,
                    n_k=2,
                    n_v=2,
                    layout_qkv=layout,
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
                    seq_lens=actual_lens,
                    n_q=8,
                    n_k=1,
                    n_v=1,
                    layout_qkv=layout,
                    slot_pattern=slot_pattern,
                    seed=seed,
                    block_num=block_num,
                    cache_stride=cache_stride,
                    cache_pad=cache_pad,
                )
            )
            seed += 1
    cases = list(with_tnd_to_ntd_cases(cases))
    cases.extend(precision_mb20_t16384_cases(seed))
    seed += 3
    rope_precision_tnd_cases = tuple(
        case
        for case in cases
        if not is_mrope_case(case)
        and not case.explicit_empty_mrope_section
        and "_perf_" not in case.name
        and case.layout_qkv == "TND"
        and (case.layout_q_out or case.layout_qkv) == "TND"
    )
    mrope_precision_cases = tuple(
        to_mrope_case(case, MROPE_PRECISION_SECTIONS[index % len(MROPE_PRECISION_SECTIONS)])
        for index, case in enumerate(rope_precision_tnd_cases)
    )
    existing_case_names = {case.name for case in cases}
    cases.extend(case for case in mrope_precision_cases if case.name not in existing_case_names)
    for section in MROPE_T_INVARIANT_SECTIONS:
        if section == MROPE_SMOKE_SECTION:
            continue
        invariant_case = to_mrope_case(mrope_smoke_source, section)
        invariant_name = f"{invariant_case.name[:-4]}_t_section_{section[0]}_tnd"
        cases.append(replace(invariant_case, name=invariant_name))
    cases.extend(performance_t16384_tnd_to_ntd_cases(seed))
    seed += 3
    performance_t5379_cases = performance_t5379_n16_kv2_cases(seed)
    cases.extend(performance_t5379_cases)
    cases.append(to_mrope_case(performance_t5379_cases[0], MROPE_PERFORMANCE_SECTION))
    seed += len(performance_t5379_cases)
    performance_cases = performance_t65536_n16_kv2_cases(seed)
    cases.extend(performance_cases)
    rope_tnd_performance_case = next(
        case
        for case in performance_cases
        if case.layout_qkv == "TND" and (case.layout_q_out or case.layout_qkv) == "TND"
    )
    cases.append(to_mrope_case(rope_tnd_performance_case, MROPE_PERFORMANCE_SECTION))
    seed += 3
    cases.append(mrope_large_output_case(seed))
    seed += 1
    cases.append(default_layout_case(seed))
    seed += 1
    performance_t131072_case = performance_t131072_n16_kv2_case(seed)
    cases.append(performance_t131072_case)
    cases.append(to_mrope_case(performance_t131072_case, MROPE_PERFORMANCE_SECTION))
    return tuple(cases)


CASES = build_cases()


def make_query_start_loc(seq_lens: Iterable[int]) -> torch.Tensor:
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
    if min(case.n_q, case.n_k, case.n_v) <= 0:
        raise ValueError(f"{case.name}: Nq, Nk, and Nv must be positive")
    if case.n_q > 64:
        raise ValueError(f"{case.name}: Nq must not exceed 64")
    if case.n_q != 8 * case.n_k:
        raise ValueError(f"{case.name}: Nq must equal 8 * Nk")
    if case.n_k != case.n_v:
        raise ValueError(f"{case.name}: Nk and Nv must match")
    if case.layout_qkv not in ("NTD", "TND"):
        raise ValueError(f"{case.name}: layout_qkv must be NTD or TND")
    layout_q_out = case.layout_q_out or case.layout_qkv
    if layout_q_out not in ("NTD", "TND"):
        raise ValueError(f"{case.name}: layout_q_out must be NTD or TND")
    if case.layout_qkv == "NTD" and layout_q_out == "TND":
        raise ValueError(f"{case.name}: layout_qkv=NTD with layout_q_out=TND is not supported")
    if len(case.seq_lens) != len(case.batch_seq_lens):
        raise ValueError(f"{case.name}: seq_lens length must match batch_seq_lens length")
    if any(actual_len < batch_len for batch_len, actual_len in zip(case.batch_seq_lens, case.seq_lens)):
        raise ValueError(f"{case.name}: seq_lens values must be >= current batch token counts")
    if case.cache_stride not in ("contiguous", "d_pad"):
        raise ValueError(f"{case.name}: cache_stride must be contiguous or d_pad")
    if case.cache_stride == "d_pad" and case.cache_pad <= 0:
        raise ValueError(f"{case.name}: cache_pad must be positive for d_pad cache stride")
    if is_mrope_case(case):
        if case.layout_qkv != "TND" or (case.layout_q_out or case.layout_qkv) != "TND":
            raise ValueError(f"{case.name}: M-RoPE requires layout_qkv=layout_q_out=TND")
        if len(case.mrope_section) != 3 or any(section < 0 for section in case.mrope_section):
            raise ValueError(f"{case.name}: mrope_section must contain three non-negative values")
        if sum(case.mrope_section) > HEAD_SIZE // 2:
            raise ValueError(f"{case.name}: mrope_section sum must not exceed D/2")
        section_capacities = (21, 21)
        if any(section > capacity for section, capacity in zip(case.mrope_section[1:], section_capacities)):
            raise ValueError(f"{case.name}: H/W mrope_section values exceed D=128 lane capacity")


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


def make_mrope_position(case: Case, total_tokens: int, max_position: int) -> torch.Tensor:
    """Build reproducible, uniformly random token-major [T, 3] positions."""
    generator = torch.Generator(device="cpu").manual_seed(case.seed + MROPE_POSITION_SEED_OFFSET)
    return torch.randint(0, max_position, (total_tokens, 3), generator=generator, dtype=torch.int32)


def make_random_rotation(case: Case) -> torch.Tensor:
    """Build a reproducible, variance-normalized dense BF16 Cube matrix."""
    generator = torch.Generator(device="cpu").manual_seed(case.seed + ROTATION_SEED_OFFSET)
    rotation = torch.randn((HEAD_SIZE, HEAD_SIZE), generator=generator, dtype=torch.float32)
    rotation.mul_(HEAD_SIZE ** -0.5)
    return rotation.to(torch.bfloat16)


def make_cpu_inputs(case: Case) -> Dict[str, object]:
    validate_case(case)
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    total_tokens = sum(case.batch_seq_lens)
    total_heads = case.n_q + case.n_k + case.n_v
    qkv_shape = (total_tokens, total_heads, HEAD_SIZE)
    if case.layout_qkv == "NTD":
        qkv_shape = (total_heads, total_tokens, HEAD_SIZE)

    qkv = (torch.randn(qkv_shape, generator=generator, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    q_gamma = torch.rand((HEAD_SIZE,), generator=generator, dtype=torch.float32) + 0.5
    k_gamma = torch.rand((HEAD_SIZE,), generator=generator, dtype=torch.float32) + 0.5
    cos_sin = torch.randn((max(case.seq_lens), HEAD_SIZE), generator=generator, dtype=torch.float32)
    slot_mapping = make_slot_mapping(case, total_tokens, generator)
    k_cache_dtype = torch.int8 if is_mrope_case(case) else torch.float8_e4m3fn
    k_cache = torch.zeros((case.block_num, case.n_k, case.block_size, HEAD_SIZE), dtype=k_cache_dtype)
    v_cache = torch.zeros((case.block_num, case.n_v, case.block_size, HEAD_SIZE), dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    k_scale_cache = torch.zeros((case.block_num, case.n_k, case.block_size, 1), dtype=torch.float32)
    query_start_loc = None if is_mrope_case(case) else make_query_start_loc(case.batch_seq_lens)
    seq_lens = None if is_mrope_case(case) else torch.tensor(case.seq_lens, dtype=torch.int32)
    mrope_position = (
        make_mrope_position(case, total_tokens, max(case.seq_lens)) if is_mrope_case(case) else None
    )
    rotation = make_random_rotation(case)
    v_scale_shape = (case.n_v, HEAD_SIZE) if is_mrope_case(case) else (case.n_v,)
    v_scale = torch.rand(v_scale_shape, generator=generator, dtype=torch.float32) + 0.5

    return {
        "qkv": qkv,
        "q_gamma": q_gamma,
        "k_gamma": k_gamma,
        "cos_sin": cos_sin,
        "slot_mapping": slot_mapping,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale_cache": k_scale_cache,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "mrope_position": mrope_position,
        "mrope_section": (
            list(case.mrope_section)
            if is_mrope_case(case)
            else ([] if case.explicit_empty_mrope_section else None)
        ),
        "q_quant_mode": "NoQuant" if is_mrope_case(case) else "PerTokenPerHead",
        "q_out_dtype": torch.bfloat16 if is_mrope_case(case) else torch.float8_e4m3fn,
        "head_nums": [case.n_q, case.n_k, case.n_v],
        "layout_qkv": case.layout_qkv,
        "layout_q_out": case.layout_q_out or case.layout_qkv,
        "rotation": rotation,
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


def resolve_positions(query_start_loc: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    cu = query_start_loc.to(torch.int64)
    actual = seq_lens.to(torch.int64)
    total_tokens = int(cu[-1].item())
    positions = torch.empty((total_tokens,), dtype=torch.int64)
    for batch_idx in range(actual.numel()):
        begin = int(cu[batch_idx].item())
        end = int(cu[batch_idx + 1].item())
        local_len = end - begin
        start_pos = int(actual[batch_idx].item()) - local_len
        positions[begin:end] = torch.arange(start_pos, start_pos + local_len, dtype=torch.int64)
    return positions


def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor, query_start_loc: torch.Tensor, seq_lens: torch.Tensor):
    positions = resolve_positions(query_start_loc, seq_lens)
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


def apply_mrope(x: torch.Tensor, cos_sin: torch.Tensor, mrope_position: torch.Tensor,
                mrope_section: Iterable[int]):
    """Apply token-major T/H/W RoPE source selection without expanding D."""
    position = mrope_position.to(torch.int64)
    if position.dim() != 2 or position.shape[1] != 3:
        raise ValueError("mrope_position must have shape [T, 3]")
    if int(position.min().item()) < 0 or int(position.max().item()) >= cos_sin.shape[0]:
        raise ValueError("cos_sin does not cover all M-RoPE positions")

    section = tuple(int(value) for value in mrope_section)
    half = x.shape[-1] // 2
    lane = torch.arange(half, dtype=torch.int64)
    repeat = torch.div(lane, 3, rounding_mode="floor")
    axis_in_group = lane.remainder(3)
    axis = torch.zeros((half,), dtype=torch.int64)
    axis[(axis_in_group == 1) & (repeat < section[1])] = 1
    axis[(axis_in_group == 2) & (repeat < section[2])] = 2

    raw = cos_sin[position]  # [T, 3, D]
    token_index = torch.arange(x.shape[0], dtype=torch.int64)[:, None]
    lane_index = lane[None, :]
    axis_index = axis[None, :]
    cos = raw[token_index, axis_index, lane_index]
    sin = raw[token_index, axis_index, lane_index + half]
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


def dynamic_quant_int8(x: torch.Tensor):
    max_abs = torch.amax(torch.abs(x), dim=-1)
    scale = max_abs / 127.0
    y = torch.round(x / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return y, scale.to(torch.float32)


def quant_v_to_fp8(v: torch.Tensor, v_scale: torch.Tensor, is_mrope: bool) -> torch.Tensor:
    """Apply V scale and cast the result to FP8.

    The scene selects the accepted v_scale shape; the V arithmetic is shared.
    """
    expected_shape = (v.shape[1], HEAD_SIZE) if is_mrope else (v.shape[1],)
    if tuple(v_scale.shape) != expected_shape:
        scene = "M-RoPE" if is_mrope else "RoPE"
        raise ValueError(f"{scene} v_scale must have shape {expected_shape}, got {tuple(v_scale.shape)}")
    scale = v_scale.reshape(1, v.shape[1], -1)
    return (v * scale).to(torch.float8_e4m3fn)


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
    query_start_loc = cpu_inputs["query_start_loc"]
    seq_lens = cpu_inputs["seq_lens"]
    mrope_position = cpu_inputs["mrope_position"]
    mrope_section = cpu_inputs["mrope_section"]
    rotation = cpu_inputs["rotation"].to(torch.float32)
    v_scale = cpu_inputs["v_scale"].to(torch.float32)
    n_q, n_k, n_v = cpu_inputs["head_nums"]
    layout_qkv = cpu_inputs["layout_qkv"]
    layout_q_out = cpu_inputs["layout_q_out"]
    epsilon = cpu_inputs["epsilon"]
    is_mrope = mrope_position is not None or bool(mrope_section)

    if layout_qkv == "TND":
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

    if is_mrope:
        q_rope = apply_mrope(q_norm, cos_sin, mrope_position, mrope_section)
        k_rope = apply_mrope(k_norm, cos_sin, mrope_position, mrope_section)
    else:
        q_rope = apply_rope(q_norm, cos_sin, query_start_loc, seq_lens)
        k_rope = apply_rope(k_norm, cos_sin, query_start_loc, seq_lens)
    q_rope = q_rope.to(torch.bfloat16).to(torch.float32)
    k_rope = k_rope.to(torch.bfloat16).to(torch.float32)
    q_after_cube = q_rope @ rotation
    k_after_cube = k_rope @ rotation

    v_out = quant_v_to_fp8(v, v_scale, is_mrope)
    if is_mrope:
        q_out = q_after_cube[:, :n_q, :].to(torch.bfloat16).contiguous()
        q_scale = None
        k_quant, k_scale = dynamic_quant_int8(k_after_cube)
    else:
        qk_after_cube = torch.cat((q_after_cube, k_after_cube), dim=1)
        qk_quant, scale = dynamic_quant_fp8(qk_after_cube)
        if layout_q_out == "TND":
            q_out = qk_quant[:, :n_q, :].contiguous()
            q_scale = scale[:, :n_q].contiguous()
        else:
            q_out = qk_quant[:, :n_q, :].transpose(0, 1).contiguous()
            q_scale = scale[:, :n_q].transpose(0, 1).contiguous()

    block_size = k_cache.shape[2]
    for token_idx, slot in enumerate(slot_mapping.tolist()):
        block_id = slot // block_size
        block_offset = slot % block_size
        if is_mrope:
            k_cache[block_id, :, block_offset, :] = k_quant[token_idx, :, :]
            k_scale_cache[block_id, :, block_offset, 0] = k_scale[token_idx, :]
        else:
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


def run_inplace(cpu_inputs: Dict[str, object], case: Case, device: str,
                warmup_before_target: bool = False):
    from cann_ops_transformer.ops import qkv_rms_norm_rope_cache_with_k_scale_

    npu_inputs = to_npu_inputs(cpu_inputs, case, device)
    if warmup_before_target:
        warmup_perf_all(device)
    layout_kwargs = (
        {}
        if case.omit_layout_args
        else {"layout_qkv": npu_inputs["layout_qkv"], "layout_q_out": npu_inputs["layout_q_out"]}
    )
    with torch.no_grad():
        q_out, q_scale = qkv_rms_norm_rope_cache_with_k_scale_(
            npu_inputs["qkv"],
            npu_inputs["q_gamma"],
            npu_inputs["k_gamma"],
            npu_inputs["cos_sin"],
            npu_inputs["slot_mapping"],
            npu_inputs["k_cache"],
            npu_inputs["v_cache"],
            npu_inputs["k_scale_cache"],
            npu_inputs["query_start_loc"],
            npu_inputs["seq_lens"],
            npu_inputs["head_nums"],
            rotation=npu_inputs["rotation"],
            v_scale=npu_inputs["v_scale"],
            epsilon=npu_inputs["epsilon"],
            mrope_position=npu_inputs["mrope_position"],
            mrope_section=npu_inputs["mrope_section"],
            q_quant_mode=npu_inputs["q_quant_mode"],
            q_out_dtype=npu_inputs["q_out_dtype"],
            **layout_kwargs,
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
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    if actual_cpu.shape != expected_cpu.shape:
        return MatchResult(
            name, 0, 1, 0.0, float("inf"), float("inf"),
            float("nan"), float("nan"), float("nan"), tuple())

    actual_is_int8 = actual_cpu.dtype == torch.int8
    expected_is_int8 = expected_cpu.dtype == torch.int8
    if actual_is_int8 != expected_is_int8:
        total = actual_cpu.numel()
        return MatchResult(
            name, 0, total, 0.0, float("inf"), float("inf"),
            float("nan"), float("nan"), float("nan"), tuple(),
            strict_all_elements=True)

    is_int8 = actual_is_int8 and expected_is_int8
    if is_int8:
        actual_compare = actual_cpu.to(torch.int16)
        expected_compare = expected_cpu.to(torch.int16)
    else:
        actual_compare = actual_cpu
        expected_compare = expected_cpu
    actual_f32 = actual_compare.to(torch.float32)
    expected_f32 = expected_compare.to(torch.float32)
    if is_int8:
        close = torch.abs(actual_compare - expected_compare) <= INT8_MAX_ABS_DIFF
    else:
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
        max_abs_diff, mean_abs_diff, max_diff, max_actual, max_expected, tuple(diff_samples),
        strict_all_elements=is_int8)


def select_touched_cache_rows(outputs: Dict[str, torch.Tensor], slot_mapping: torch.Tensor) -> Dict[str, torch.Tensor]:
    slots = slot_mapping.detach().cpu().to(torch.int64)
    block_size = outputs["k_cache"].shape[2]
    block_ids = slots // block_size
    block_offsets = slots % block_size
    k_cache_cpu = outputs["k_cache"].detach().cpu()
    v_cache_cpu = outputs["v_cache"].detach().cpu()
    k_scale_cache_cpu = outputs["k_scale_cache"].detach().cpu()
    if k_cache_cpu.dtype == torch.float8_e4m3fn:
        k_cache_cpu = k_cache_cpu.to(torch.float32)
    if v_cache_cpu.dtype == torch.float8_e4m3fn:
        v_cache_cpu = v_cache_cpu.to(torch.float32)
    return {
        "k_cache_touched": k_cache_cpu[block_ids, :, block_offsets, :],
        "v_cache_touched": v_cache_cpu[block_ids, :, block_offsets, :],
        "k_scale_cache_touched": k_scale_cache_cpu[block_ids, :, block_offsets, 0],
    }


def compare_outputs(actual: Dict[str, torch.Tensor], expected: Dict[str, torch.Tensor],
                    cpu_inputs: Dict[str, object], args):
    actual_touched = select_touched_cache_rows(actual, cpu_inputs["slot_mapping"])
    expected_touched = select_touched_cache_rows(expected, cpu_inputs["slot_mapping"])
    is_mrope = cpu_inputs["mrope_position"] is not None or bool(cpu_inputs["mrope_section"])
    q_out_atol = args.float_atol if is_mrope else args.fp8_atol
    q_out_rtol = args.float_rtol if is_mrope else args.fp8_rtol

    results = [
        compare_tensor(
            "q_out", actual["q_out"], expected["q_out"], q_out_atol, q_out_rtol, args.max_diff_lines),
        compare_tensor("k_cache_touched", actual_touched["k_cache_touched"],
                       expected_touched["k_cache_touched"], args.fp8_atol, args.fp8_rtol, args.max_diff_lines),
        compare_tensor("v_cache_touched", actual_touched["v_cache_touched"],
                       expected_touched["v_cache_touched"], args.fp8_atol, args.fp8_rtol, args.max_diff_lines),
    ]
    if actual["q_scale"] is not None or expected["q_scale"] is not None:
        if actual["q_scale"] is None or expected["q_scale"] is None:
            raise AssertionError("q_scale presence differs between actual and golden outputs")
        results.append(
            compare_tensor("q_scale", actual["q_scale"], expected["q_scale"],
                           args.float_atol, args.float_rtol, args.max_diff_lines)
        )
    results.append(
        compare_tensor("k_scale_cache_touched", actual_touched["k_scale_cache_touched"],
                       expected_touched["k_scale_cache_touched"],
                       args.float_atol, args.float_rtol, args.max_diff_lines)
    )
    total = sum(result.total for result in results)
    matched = sum(result.matched for result in results)
    overall = matched / total if total else 1.0
    return results, overall


def exact_output_snapshot(actual: Dict[str, torch.Tensor], cpu_inputs: Dict[str, object]) -> Dict[str, torch.Tensor]:
    snapshot = {"q_out": actual["q_out"].detach().cpu()}
    snapshot.update(select_touched_cache_rows(actual, cpu_inputs["slot_mapping"]))
    return snapshot


def expected_shapes(case: Case):
    total_tokens = sum(case.batch_seq_lens)
    layout_q_out = case.layout_q_out or case.layout_qkv
    if layout_q_out == "TND":
        return (total_tokens, case.n_q, HEAD_SIZE), (total_tokens, case.n_q)
    return (case.n_q, total_tokens, HEAD_SIZE), (case.n_q, total_tokens)


def assert_shapes(case: Case, outputs: Dict[str, torch.Tensor]):
    q_out_shape, q_scale_shape = expected_shapes(case)
    assert tuple(outputs["q_out"].shape) == q_out_shape, (
        f"{case.name}: q_out shape mismatch: {tuple(outputs['q_out'].shape)} vs {q_out_shape}")
    if is_mrope_case(case):
        assert outputs["q_scale"] is None, f"{case.name}: M-RoPE q_scale must be absent"
        assert outputs["q_out"].dtype == torch.bfloat16, (
            f"{case.name}: q_out dtype mismatch: {outputs['q_out'].dtype}")
    else:
        assert tuple(outputs["q_scale"].shape) == q_scale_shape, (
            f"{case.name}: q_scale shape mismatch: {tuple(outputs['q_scale'].shape)} vs {q_scale_shape}")
        assert outputs["q_out"].dtype == torch.float8_e4m3fn, (
            f"{case.name}: q_out dtype mismatch: {outputs['q_out'].dtype}")
        assert outputs["q_scale"].dtype == torch.float32, (
            f"{case.name}: q_scale dtype mismatch: {outputs['q_scale'].dtype}")


def warmup_cpu_golden_sqrt() -> None:
    # Keep MKL VML's first sqrt dispatch on the caller thread before parallel golden computation.
    _ = torch.sqrt(torch.ones(1, dtype=torch.float32, device="cpu"))


def warmup_perf_all(device: str) -> None:
    x = torch.zeros((16384, 16384), dtype=torch.float32, device=device)
    _ = torch.all(x, dim=0)
    torch.npu.synchronize()


def print_case_results(case: Case, results: List[MatchResult]) -> None:
    print(
        f"[CASE] {case.name} layout_qkv={case.layout_qkv} "
        f"layout_q_out={case.layout_q_out or case.layout_qkv} omit_layout_args={case.omit_layout_args} "
        f"scene={'M-RoPE' if is_mrope_case(case) else 'RoPE'} "
        f"mrope_section={case.mrope_section if is_mrope_case(case) else None} "
        f"heads=({case.n_q},{case.n_k},{case.n_v}) "
        f"query_lens={case.batch_seq_lens} seq_lens={case.seq_lens} slot={case.slot_pattern} "
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


def run_case(case: Case, args, capture_outputs: bool = False):
    cpu_inputs = make_cpu_inputs(case)
    expected = golden_outputs(cpu_inputs)
    actual = run_inplace(cpu_inputs, case, args.device, args.perf_warmup_all)
    assert_shapes(case, actual)
    results, overall = compare_outputs(actual, expected, cpu_inputs, args)
    print_case_results(case, results)
    for result in results:
        required_rate = 1.0 if result.strict_all_elements else args.min_match_rate
        if result.rate < required_rate:
            raise AssertionError(
                f"{case.name}: {result.name} golden match rate {result.rate:.6f} "
                f"is below threshold {required_rate:.6f}")
    snapshot = exact_output_snapshot(actual, cpu_inputs) if capture_outputs else None
    return overall, snapshot


def run_selected_cases(selected_names: List[str], case_map: Dict[str, Case], args) -> int:
    success_count = 0
    failed_cases: List[Tuple[str, str]] = []
    invariant_reference = None
    invariant_reference_name = ""
    invariant_pass_count = 0

    for case_name in selected_names:
        try:
            case = case_map[case_name]
            overall, snapshot = run_case(case, args, is_mrope_t_invariant_case(case))
            if snapshot is not None:
                if invariant_reference is None:
                    invariant_reference = snapshot
                    invariant_reference_name = case_name
                else:
                    mismatched_outputs = [
                        name
                        for name in invariant_reference
                        if not torch.equal(snapshot[name], invariant_reference[name])
                    ]
                    if mismatched_outputs:
                        raise AssertionError(
                            f"{case_name}: mrope_section[0] invariant differs from "
                            f"{invariant_reference_name}: {mismatched_outputs}"
                        )
                    invariant_pass_count += 1
                    print(
                        f"[INVARIANT PASS] {case_name} matches {invariant_reference_name} "
                        "for q_out/k_cache/v_cache/k_scale_cache"
                    )
        except Exception as err:
            failed_cases.append((case_name, f"{type(err).__name__}: {err}"))
            print(f"[CASE FAIL] {case_name}: {type(err).__name__}: {err}", file=sys.stderr)
            continue

        success_count += 1
        print(f"[CASE PASS] {case_name}: overall_match={overall * 100.0:.2f}%")

    failed_count = len(failed_cases)
    if invariant_pass_count:
        print(f"[INVARIANT SUMMARY] comparisons={invariant_pass_count} failed=0")
    print(f"[SUMMARY] cases={len(selected_names)} success={success_count} failed={failed_count}")
    for case_name, message in failed_cases:
        print(f"  [FAILED] {case_name}: {message}")
    return 1 if failed_cases else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append",
                        help=f"case name to run; may be repeated; default: all {len(CASES)} cases")
    parser.add_argument("--mrope-precision-all", action="store_true",
                        help="run all M-RoPE precision mirrors and section probes")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--min-match-rate", type=float, default=0.99)
    parser.add_argument("--fp8-atol", type=float, default=1.0)
    parser.add_argument("--fp8-rtol", type=float, default=0.0)
    parser.add_argument("--float-atol", type=float, default=1e-2)
    parser.add_argument("--float-rtol", type=float, default=1e-2)
    parser.add_argument("--max-diff-lines", type=int, default=20,
                        help="maximum mismatch detail lines to print per output; set 0 to disable")
    parser.add_argument("--perf-warmup-all", action="store_true",
                        help="run a large torch.all warmup immediately before each selected target op; intended for msprof runs")
    return parser.parse_args()


def main():
    args = parse_args()
    case_map = {case.name: case for case in CASES}
    if args.list_cases:
        print("\n".join(case_map))
        return 0

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is required to run this example")

    warmup_cpu_golden_sqrt()

    if args.mrope_precision_all and args.case:
        raise ValueError("--mrope-precision-all cannot be combined with --case")
    selected_names: List[str] = (
        [case.name for case in CASES if is_mrope_case(case) and "_perf_" not in case.name]
        if args.mrope_precision_all
        else (args.case or list(case_map))
    )
    unknown = sorted(set(selected_names) - set(case_map))
    if unknown:
        raise ValueError(f"unknown case(s): {unknown}; available: {list(case_map)}")

    return run_selected_cases(selected_names, case_map, args)


if __name__ == "__main__":
    raise SystemExit(main())
