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

"""Generate deterministic, resource-bounded MXFP8 generalized cases."""

import argparse
import ast
import csv
import io
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path


CASE_COUNT = 2_000
LARGE_CASE_COUNT = 0
GENERATOR_SEED = 20260823
HEAD_DIM = 128
QUANT_GROUP_SIZE = 32
MAX_SEQUENCE_LENGTH = 4_096
MAX_ESTIMATED_MIB = 256.0
TOKEN_HEAD_BUDGET = 262_144

SCRIPT_DIR = Path(__file__).resolve().parent
PYTEST_DIR = SCRIPT_DIR.parent
TESTS_DIR = PYTEST_DIR.parents[1]
TTK_DIR = TESTS_DIR / "ttk" / "bsa_fullquant_mxfp8_test"
STC_PATH = PYTEST_DIR / "qbsa_mxfp8_stc.csv"
PYTEST_OUTPUT = PYTEST_DIR / "qbsa_mxfp8_generalized_2000.csv"
TTK_OUTPUT = TTK_DIR / "qbsa_mxfp8_generalized_2000.csv"
COVERAGE_OUTPUT = SCRIPT_DIR / "qbsa_mxfp8_factor_coverage_2000.json"

TTK_FIELDS = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "output_tensor_indexes",
    "golden_api",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
)

TTK_DTYPES = (
    "float8_e4m3fn",
    "float8_e4m3fn",
    "float8_e4m3fn",
    "float8_e8m0",
    "float8_e8m0",
    "float8_e8m0",
    "int32",
    "int32",
    "float8_e8m0",
    "int32",
    "int32",
    "int32",
    "int32",
    "int32",
    "uint8",
    "int32",
)

PRECISION_TOLERANCES = (
    (0.0078125, 0.0001, 0.005, 0.005, 10),
    (0.005, 2.5e-05, 0.005, 0.005, 10),
)
ABSOLUTE_PRECISION = (0.0001, 0.0001)

RANGE_MAGNITUDE_BUCKETS = ("unit", "e4m3", "fp16", "fp32_extreme")
SEQUENCE_BOUNDARIES = (
    1, 63, 64, 65, 127, 128, 129, 255, 256, 257,
    511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096,
)


def _ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)


def _prefix_sums(lengths):
    result = [0]
    for value in lengths:
        result.append(result[-1] + int(value))
    return result


def _schedule(rng, counts):
    values = []
    for value, count in counts.items():
        values.extend([value] * int(count))
    rng.shuffle(values)
    return values


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _parse_list(value):
    if value is None or not str(value).strip():
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected list, got {value!r}")
    return [int(item) for item in parsed]


def _range_value(value):
    text = str(value).strip().lower()
    if text == "inf":
        return math.inf
    if text == "-inf":
        return -math.inf
    if text == "nan":
        return math.nan
    if text.startswith(("[", "(")):
        body = text[1:-1]
        parts = [part.strip() for part in body.split(",")]
        if len(parts) != 2:
            raise ValueError(f"expected two range bounds, got {value!r}")
        special = {"inf": math.inf, "+inf": math.inf, "-inf": -math.inf, "nan": math.nan}
        parsed = [special.get(part, float(part)) if part not in special else special[part] for part in parts]
    else:
        parsed = float(text)
    if isinstance(parsed, (list, tuple)):
        return [float(parsed[0]), float(parsed[1])]
    return float(parsed)


def _format_number(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == math.inf:
            return "inf"
        if value == -math.inf:
            return "-inf"
        return format(value, ".17g")
    return str(value)


def _format_range(value):
    if isinstance(value, (list, tuple)):
        return f"[{_format_number(value[0])},{_format_number(value[1])}]"
    return _format_number(value)


def _tuple_range(value):
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    scalar = float(value)
    if math.isfinite(scalar) and scalar >= 0:
        return -scalar, scalar
    return scalar, scalar


def _random_range(rng, bucket, sign):
    bounds = {
        "unit": (2.0 ** -12, 1.0),
        "e4m3": (1.0, 448.0),
        "fp16": (448.0, 65504.0),
        "fp32_extreme": (1.0e30, 3.3895313892515355e38),
    }
    low_mag, high_mag = bounds[bucket]
    if bucket == "fp32_extreme":
        magnitude = 10.0 ** rng.uniform(math.log10(low_mag), math.log10(high_mag))
    else:
        magnitude = rng.uniform(low_mag, high_mag)
    inner = magnitude * rng.uniform(0.0, 0.95)
    if sign == "positive":
        low, high = inner, magnitude
    elif sign == "negative":
        low, high = -magnitude, -inner
    elif sign == "zero":
        low = high = 0.0
    else:
        low, high = -inner, magnitude
    return [float(f"{low:.10g}"), float(f"{high:.10g}")]


def _runtime_lengths(rng, batch, maximum, mode):
    maximum = max(1, int(maximum))
    if mode == "uniform":
        return [maximum] * batch
    values = [rng.randint(1, maximum) for _ in range(batch)]
    values[rng.randrange(batch)] = maximum
    if mode == "zero_q" or mode == "zero_kv":
        zero_index = rng.randrange(batch)
        values[zero_index] = 0
        if max(values) != maximum:
            values[(zero_index + 1) % batch] = maximum
    return values


def _minimum_sequence_limit(relation, alignment):
    minima = {
        ("lt", "AA"): 512, ("lt", "AU"): 129, ("lt", "UA"): 512, ("lt", "UU"): 2,
        ("eq", "AA"): 512, ("eq", "AU"): 128, ("eq", "UU"): 1,
        ("gt", "AA"): 1024, ("gt", "AU"): 128, ("gt", "UA"): 513, ("gt", "UU"): 2,
    }
    return minima[(relation, alignment)]


def _sample_standard_lengths(rng, relation, alignment, index, maximum=MAX_SEQUENCE_LENGTH):
    q_aligned = alignment[0] == "A"
    kv_aligned = alignment[1] == "A"
    boundaries = [value for value in SEQUENCE_BOUNDARIES if value <= maximum]
    unaligned_maximum = maximum if maximum % 128 and maximum % 512 else maximum - 1
    q_aligned_maximum = (maximum // 128) * 128
    kv_aligned_maximum = (maximum // 512) * 512
    for attempt in range(10_000):
        if attempt == 0:
            q_value = boundaries[(index * 5 + 3) % len(boundaries)]
            kv_value = boundaries[(index * 7 + 11) % len(boundaries)]
        else:
            q_value = rng.randint(1, maximum)
            kv_value = rng.randint(1, maximum)
        if q_aligned:
            q_value = max(128, min(q_aligned_maximum, round(q_value / 128) * 128))
        elif q_value % 128 == 0:
            q_value = min(unaligned_maximum, q_value + 1)
        if kv_aligned:
            kv_value = max(512, min(kv_aligned_maximum, round(kv_value / 512) * 512))
        elif kv_value % 512 == 0:
            kv_value = min(unaligned_maximum, kv_value + 1)
        if relation == "eq":
            if alignment == "AA":
                value = max(512, min(kv_aligned_maximum, round(max(q_value, kv_value) / 512) * 512))
            elif alignment == "AU":
                value = max(128, min(unaligned_maximum, round(max(q_value, kv_value) / 128) * 128))
                if value % 512 == 0:
                    value += 128 if value + 128 <= maximum else -128
            else:
                value = min(unaligned_maximum, max(q_value, kv_value))
                if value % 128 == 0 or value % 512 == 0:
                    value = min(unaligned_maximum, value + 1)
            return value, value
        if relation == "lt" and q_value < kv_value:
            return q_value, kv_value
        if relation == "gt" and q_value > kv_value:
            return q_value, kv_value
    raise RuntimeError(f"cannot sample {relation}/{alignment}")


def _case_from_factors(rng, index, schedules, sequence_anchor=None):
    sparse_size = 64 if index % 2 == 0 else 128
    pa_ratio_limit = 1024 // sparse_size
    pa_ratio = (index // 2) % pa_ratio_limit + 1
    block_size = sparse_size * pa_ratio
    n2 = index % 8 + 1
    ratio = (index // 8) % 16 + 1
    n1 = n2 * ratio
    batch = 1 if index % 17 == 0 else index % 31 + 2
    relation = schedules["relation"][index]
    alignment = schedules["alignment"][index]
    if relation == "eq" and alignment == "UA":
        alignment = "UU"
    resource_level = "standard"
    if sequence_anchor is not None:
        s1, s2 = sequence_anchor
        batch = 1 + index % 2
        n2 = index % 4 + 1
        ratio = index % 8 + 1
        n1 = n2 * ratio
        relation = "eq" if s1 == s2 else "lt" if s1 < s2 else "gt"
        alignment = ("A" if s1 % 128 == 0 else "U") + ("A" if s2 % 512 == 0 else "U")
        resource_level = "boundary"
    else:
        minimum = _minimum_sequence_limit(relation, alignment)
        denominator = n1 + 2 * n2
        batch = min(batch, max(1, TOKEN_HEAD_BUDGET // (denominator * minimum)))
        maximum = min(MAX_SEQUENCE_LENGTH, TOKEN_HEAD_BUDGET // (batch * denominator))
        s1, s2 = _sample_standard_lengths(rng, relation, alignment, index, max(minimum, maximum))

    length_mode = schedules["length_mode"][index]
    if batch == 1 and length_mode in ("ragged", "zero_q", "zero_kv"):
        length_mode = "uniform"
    q_mode = "zero_q" if length_mode == "zero_q" else "ragged" if length_mode == "ragged" else "uniform"
    kv_mode = "zero_kv" if length_mode == "zero_kv" else "ragged" if length_mode == "ragged" else "uniform"
    q_lengths = _runtime_lengths(rng, batch, s1, q_mode)
    kv_lengths = _runtime_lengths(rng, batch, s2, kv_mode)
    s1 = max(q_lengths)
    s2 = max(kv_lengths)
    cu_q = _prefix_sums(q_lengths)

    required_width = max(_ceil_div(value, block_size) for value in kv_lengths)
    max_block_relation = schedules["max_block_relation"][index]
    max_block_per_batch = required_width if max_block_relation == "exact" else required_width + 1 + index % 3
    logical_pages = max(1, sum(_ceil_div(value, block_size) for value in kv_lengths))
    block_relation = schedules["block_relation"][index]
    if block_relation == "under":
        blocknum = max(1, logical_pages - max(1, logical_pages // 4))
    elif block_relation == "equal":
        blocknum = logical_pages
    else:
        blocknum = logical_pages + max(1, min(batch, 8))

    p_mode = schedules["p_mode"][index]
    if p_mode == "none":
        p_scale = None
    elif schedules["p_bucket"][index] == "le1":
        p_scale = float(2 ** rng.randint(-20, 0))
    elif schedules["p_bucket"][index] == "le100":
        p_scale = float(2 ** rng.randint(1, 6))
    else:
        p_scale = float(2 ** rng.randint(7, 12))

    softmax_mode = schedules["softmax"][index]
    softmax_scale = (
        None
        if softmax_mode == "none"
        else 1.0 / math.sqrt(HEAD_DIM)
        if softmax_mode == "default"
        else rng.uniform(1.0e-6, 1.0)
    )
    range_signs = ("cross", "positive", "negative", "zero")
    q_range = _random_range(rng, schedules["q_range"][index], range_signs[index % 4])
    k_range = _random_range(rng, schedules["k_range"][index], range_signs[(index + 1) % 4])
    v_range = _random_range(rng, schedules["v_range"][index], range_signs[(index + 2) % 4])
    sparse_mode = "dense" if index % 4 == 0 else "random"
    mask_mode = schedules["mask"][index]
    return_lse = schedules["lse"][index]
    # Keep the paired pytest/TTK generalized suites semantically identical.
    # Arbitrary non-contiguous views need explicit TTK stride columns and stay
    # in the existing standalone robustness case instead of this paired CSV.
    contiguous = True
    range_categories = (
        schedules["q_range"][index],
        schedules["k_range"][index],
        schedules["v_range"][index],
    )
    name = (
        f"MXGEN{index + 1:05d}_B{batch}_N{n1}_{n2}_"
        f"S{sparse_size}_PA{block_size}_{resource_level.upper()}"
    )
    case = {
        "name": name,
        "enable": True,
        "B": batch,
        "N1": n1,
        "N2": n2,
        "D": HEAD_DIM,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_kv": [],
        "seqused_q": [],
        "seqused_kv": kv_lengths,
        "sparse_mode": sparse_mode,
        "max_block_per_batch": max_block_per_batch,
        "s2_base_size": 512,
        "block_size": block_size,
        "mask_mode": mask_mode,
        "fp8_dtype": "float8_e4m3fn",
        "scale_dtype": "float8_e8m0fnu",
        "quant_group_size": QUANT_GROUP_SIZE,
        "layout_q": "TND",
        "layout_kv": "PA_BNBD",
        "layout_sparse_indices": "B_N_Qb_Kb",
        "layout_out": "TND",
        "kv_cache_layout": "BnNBsD",
        "is_contiguous": contiguous,
        "p_scale_value": p_scale,
        "softmax_scale": softmax_scale,
        "return_softmax_lse": return_lse,
        "seed": 500000 + index,
        "data_range_q": q_range,
        "data_range_k": k_range,
        "data_range_v": v_range,
        "device_id": 0,
        "sparse_q_block_size": sparse_size,
        "sparse_kv_block_size": sparse_size,
        "quant_mode": 2,
        "blocknum": blocknum,
        "case_group": "generalized_boundary" if resource_level == "boundary" else "generalized_2000",
        "shard": f"generalized_shard_{index // 200 + 1:02d}",
        "s1_base_size": 128,
        "S1_max": s1,
        "S2_max": s2,
        "n1_n2_ratio": ratio,
        "s1_s2_relation": relation,
        "s1_alignment": "aligned" if s1 % 128 == 0 else "unaligned",
        "s2_alignment": "aligned" if s2 % 512 == 0 else "unaligned",
        "required_pa_pages": logical_pages,
        "effective_block_table_width": max_block_per_batch,
        "physical_page_relation": block_relation,
        "block_table_relation": max_block_relation,
        "sparse_count": "dense" if sparse_mode == "dense" else "random",
        "range_q_category": range_categories[0],
        "range_k_category": range_categories[1],
        "range_v_category": range_categories[2],
        "random_domain_10pct": False,
        "ttk_mask_shape": [2048, 2048] if mask_mode == 3 else [0],
        "estimated_weight": 1,
        "resource_level": resource_level,
        "runtime_length_mode": length_mode,
    }
    case["estimated_total_mib"] = round(_estimate_case_mib(case), 3)
    return case


def _estimate_case_mib(case):
    """Conservative host/device tensor footprint estimate for generation filtering."""
    batch, n1, n2 = int(case["B"]), int(case["N1"]), int(case["N2"])
    total_q = int(case["cu_seqlens_q"][-1])
    blocknum, block_size = int(case["blocknum"]), int(case["block_size"])
    q_elements = total_q * n1 * HEAD_DIM
    kv_elements = 2 * blocknum * n2 * block_size * HEAD_DIM
    q_scale = total_q * n1 * 4
    kv_scale = 2 * blocknum * n2 * block_size * 4
    v_scale = blocknum * n2 * (block_size // 64) * HEAD_DIM * 2
    sparse = (
        batch * n1 * _ceil_div(int(case["S1_max"]), int(case["sparse_q_block_size"]))
        * _ceil_div(int(case["S2_max"]), int(case["sparse_kv_block_size"])) * 4
    )
    output = q_elements * 2 + total_q * n1 * 4
    metadata = (batch * int(case["max_block_per_batch"]) + total_q + batch * 4) * 4
    return (q_elements + kv_elements + q_scale + kv_scale + v_scale + sparse + output + metadata) / (1024 ** 2)


def _apply_regression(case, tag, q_lengths, kv_lengths, q_range, k_range, v_range, **overrides):
    """Turn a regular case into a compact, named historical/boundary regression anchor."""
    batch = len(q_lengths)
    if len(kv_lengths) != batch:
        raise ValueError(f"{tag}: Q/KV batch mismatch")
    case.update(overrides)
    case.update(
        {
            "name": f"MXREG_{tag}",
            "B": batch,
            "cu_seqlens_q": _prefix_sums(q_lengths),
            "seqused_kv": list(kv_lengths),
            "data_range_q": q_range,
            "data_range_k": k_range,
            "data_range_v": v_range,
            "S1_max": max(q_lengths),
            "S2_max": max(kv_lengths),
            "case_group": "generalized_regression",
            "resource_level": "regression",
            "runtime_length_mode": (
                "zero_q" if 0 in q_lengths else "zero_kv" if 0 in kv_lengths
                else "uniform" if len(set(q_lengths)) == 1 and len(set(kv_lengths)) == 1
                else "ragged"
            ),
        }
    )
    n1, n2 = int(case["N1"]), int(case["N2"])
    if n1 % n2:
        raise ValueError(f"{tag}: invalid GQA ratio")
    block_size = int(case["block_size"])
    logical_pages = max(1, sum(_ceil_div(value, block_size) for value in kv_lengths))
    required_width = max(_ceil_div(value, block_size) for value in kv_lengths)
    case.update(
        {
            "blocknum": logical_pages,
            "max_block_per_batch": required_width,
            "required_pa_pages": logical_pages,
            "effective_block_table_width": required_width,
            "physical_page_relation": "equal",
            "block_table_relation": "exact",
            "n1_n2_ratio": n1 // n2,
            "s1_s2_relation": "eq" if max(q_lengths) == max(kv_lengths) else "lt" if max(q_lengths) < max(kv_lengths) else "gt",
            "s1_alignment": "aligned" if max(q_lengths) % 128 == 0 else "unaligned",
            "s2_alignment": "aligned" if max(kv_lengths) % 512 == 0 else "unaligned",
            "range_q_category": _range_special(q_range),
            "range_k_category": _range_special(k_range),
            "range_v_category": _range_special(v_range),
            "ttk_mask_shape": [2048, 2048] if int(case["mask_mode"]) == 3 else [0],
        }
    )
    case["estimated_total_mib"] = round(_estimate_case_mib(case), 3)
    return case


def _directed_regression_cases(rng, schedules):
    """Pair numerical hazards across one tile and cross-tile online-softmax merge."""
    cases = []
    finite = [-1.0, 1.0]
    specials = (("POS_INF", math.inf), ("NEG_INF", -math.inf), ("NAN", math.nan), ("DUAL_INF", [-math.inf, math.inf]))
    for tensor in ("Q", "K", "V"):
        for special_index, (label, value) in enumerate(specials):
            for tile_label, kv_length in (("SINGLE_TILE", 511), ("MULTI_TILE", 513)):
                index = len(cases)
                case = _case_from_factors(rng, index, schedules)
                ranges = {"Q": finite, "K": finite, "V": finite}
                ranges[tensor] = value
                q_length = 2 if tensor == "Q" and label == "DUAL_INF" else 1
                cases.append(
                    _apply_regression(
                        case, f"{tensor}_{label}_{tile_label}", [q_length], [kv_length],
                        ranges["Q"], ranges["K"], ranges["V"],
                        N1=4 if special_index % 2 else 1, N2=1,
                        sparse_q_block_size=64, sparse_kv_block_size=64, block_size=64,
                        sparse_mode="dense", mask_mode=0, return_softmax_lse=True,
                    )
                )

    extra_specs = (
        ("LSE_EXTREME_NEG_SINGLE_TILE", [1], [511], [3.3895313892515355e38] * 2, [-0.5, 0.0], [-0.5, 0.0], {}),
        ("LSE_EXTREME_NEG_MULTI_TILE", [1], [513], [3.3895313892515355e38] * 2, [-0.5, 0.0], [-0.5, 0.0], {}),
        ("LSE_EXTREME_NEG_1024", [1], [1024], [3.3895313892515355e38] * 2, [-0.5, 0.0], [-0.5, 0.0], {}),
        ("LSE_EXTREME_NEG_1025", [1], [1025], [3.3895313892515355e38] * 2, [-0.5, 0.0], [-0.5, 0.0], {}),
        ("RAGGED_ZERO_Q_GQA", [0, 129], [65, 513], finite, finite, finite, {"N1": 12, "N2": 2, "mask_mode": 3}),
        ("RAGGED_ZERO_KV_GQA", [129, 256], [0, 513], finite, finite, finite, {"N1": 12, "N2": 2, "mask_mode": 3, "p_scale_value": None}),
        ("V_DUAL_INF_MULTI_TILE_GQA", [65, 128], [129, 513], [0.0, 0.0], [0.0, 0.0], [-math.inf, math.inf], {"N1": 12, "N2": 2}),
        ("NAN_SECOND_MERGE_TILE", [1], [1025], finite, math.nan, finite, {"N1": 8, "N2": 2}),
    )
    for tag, q_lengths, kv_lengths, q_range, k_range, v_range, overrides in extra_specs:
        index = len(cases)
        case = _case_from_factors(rng, index, schedules)
        base_overrides = {
            "N1": 1, "N2": 1, "sparse_q_block_size": 64, "sparse_kv_block_size": 64,
            "block_size": 64, "sparse_mode": "dense", "mask_mode": 0,
            "return_softmax_lse": True, "p_scale_value": 96.0,
        }
        base_overrides.update(overrides)
        cases.append(_apply_regression(case, tag, q_lengths, kv_lengths, q_range, k_range, v_range, **base_overrides))
    return cases


def _boundary_anchors():
    anchors = []
    for position, value in enumerate(SEQUENCE_BOUNDARIES):
        partner = SEQUENCE_BOUNDARIES[min(position + 1, len(SEQUENCE_BOUNDARIES) - 1)]
        anchors.extend(((value, partner), (partner, value)))
    return anchors


def build_cases(count=CASE_COUNT):
    if count != CASE_COUNT:
        raise ValueError(f"factor quotas require exactly {CASE_COUNT} cases")
    rng = random.Random(GENERATOR_SEED)
    schedules = {
        "relation": _schedule(rng, {"lt": 667, "eq": 666, "gt": 667}),
        "alignment": _schedule(rng, {"AA": 500, "AU": 500, "UA": 500, "UU": 500}),
        "length_mode": _schedule(rng, {"uniform": 1000, "ragged": 920, "zero_q": 40, "zero_kv": 40}),
        "max_block_relation": _schedule(rng, {"exact": 1500, "extra": 500}),
        "block_relation": _schedule(rng, {"under": 1400, "equal": 400, "over": 200}),
        "p_mode": _schedule(rng, {"none": 200, "value": 1800}),
        "p_bucket": _schedule(rng, {"le1": 1440, "le100": 400, "gt100": 160}),
        "softmax": _schedule(rng, {"none": 200, "default": 1600, "arbitrary": 200}),
        "mask": _schedule(rng, {0: 600, 3: 1400}),
        "lse": _schedule(rng, {True: 1800, False: 200}),
    }
    for tensor_name in ("q", "k", "v"):
        schedules[f"{tensor_name}_range"] = _schedule(
            rng,
            {"unit": 1600, "e4m3": 300, "fp16": 80, "fp32_extreme": 20},
        )
    cases = _directed_regression_cases(rng, schedules)
    for anchor in _boundary_anchors():
        index = len(cases)
        cases.append(_case_from_factors(rng, index, schedules, sequence_anchor=anchor))
    while len(cases) < count:
        index = len(cases)
        cases.append(_case_from_factors(rng, index, schedules))
    return cases


def _validate_case(case):
    name = case["name"]
    batch = int(case["B"])
    n1, n2 = int(case["N1"]), int(case["N2"])
    if batch <= 0 or not (1 <= n2 <= 8) or n1 > 128 or n1 % n2 or n1 // n2 > 16:
        raise ValueError(f"{name}: invalid batch/head factors")
    if case["layout_q"] != "TND" or case["layout_kv"] != "PA_BNBD":
        raise ValueError(f"{name}: invalid MXFP8 layout")
    sparse_q = int(case["sparse_q_block_size"])
    sparse_kv = int(case["sparse_kv_block_size"])
    block_size = int(case["block_size"])
    if sparse_q != sparse_kv or sparse_q not in (64, 128):
        raise ValueError(f"{name}: invalid sparse block size")
    if block_size > 1024 or block_size % sparse_kv:
        raise ValueError(f"{name}: invalid PA block size")
    cu_q = case["cu_seqlens_q"]
    kv_lengths = case["seqused_kv"]
    if len(cu_q) != batch + 1 or cu_q[0] != 0 or any(b < a for a, b in zip(cu_q, cu_q[1:])):
        raise ValueError(f"{name}: invalid cu_seqlens_q")
    if len(kv_lengths) != batch or min(kv_lengths) < 0 or max(kv_lengths) <= 0:
        raise ValueError(f"{name}: invalid seqused_kv")
    required_width = max(_ceil_div(value, block_size) for value in kv_lengths)
    if int(case["max_block_per_batch"]) < required_width:
        raise ValueError(f"{name}: block table is narrower than logical PA pages")
    if int(case["blocknum"]) <= 0:
        raise ValueError(f"{name}: blocknum must be positive")
    if int(case["quant_mode"]) != 2:
        raise ValueError(f"{name}: generalized suite is MXFP8-only")
    if max(int(case["S1_max"]), int(case["S2_max"])) > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"{name}: generalized sequence exceeds {MAX_SEQUENCE_LENGTH}")
    estimated_mib = _estimate_case_mib(case)
    if estimated_mib > MAX_ESTIMATED_MIB:
        raise ValueError(f"{name}: estimated {estimated_mib:.1f} MiB exceeds {MAX_ESTIMATED_MIB:.1f} MiB")


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def render_pytest_csv(cases, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for case in cases:
        _validate_case(case)
        row = {field: _csv_value(case.get(field)) for field in fields}
        for field in ("data_range_q", "data_range_k", "data_range_v"):
            row[field] = _format_range(case[field])
        writer.writerow(row)
    return output.getvalue()


def _ttk_row(case):
    batch, n1, n2 = int(case["B"]), int(case["N1"]), int(case["N2"])
    total_q = int(case["cu_seqlens_q"][-1])
    blocknum = int(case["blocknum"])
    block_size = int(case["block_size"])
    sparse_q = int(case["sparse_q_block_size"])
    sparse_kv = int(case["sparse_kv_block_size"])
    q_blocks = _ceil_div(int(case["S1_max"]), sparse_q)
    kv_blocks = _ceil_div(int(case["S2_max"]), sparse_kv)
    max_block_per_batch = int(case["max_block_per_batch"])
    mask_shape = (2048, 2048) if int(case["mask_mode"]) == 3 else (0,)
    shapes = (
        (total_q, n1, HEAD_DIM),
        (blocknum, n2, block_size, HEAD_DIM),
        (blocknum, n2, block_size, HEAD_DIM),
        (total_q, n1, 2, 2),
        (blocknum, n2, block_size, 2, 2),
        (blocknum, n2, block_size // 64, HEAD_DIM, 2),
        (batch, n1, q_blocks, kv_blocks),
        (batch, n1, q_blocks),
        (1,),
        (len(case["cu_seqlens_q"]),),
        (len(case["cu_seqlens_kv"]),),
        (len(case["seqused_q"]),),
        (len(case["seqused_kv"]),),
        (batch, max_block_per_batch),
        mask_shape,
        (0,),
    )
    softmax_scale = case["softmax_scale"]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    attributes = {
        "p_scale_value": case["p_scale_value"],
        "cu_seqlens_q_value": case["cu_seqlens_q"],
        "cu_seqlens_kv_value": case["cu_seqlens_kv"],
        "seqused_q_value": case["seqused_q"],
        "seqused_kv_value": case["seqused_kv"],
        "quant_mode": 2,
        "softmax_scale": float(softmax_scale),
        "mask_mode": int(case["mask_mode"]),
        "blocksize": block_size,
        "sparse_block_size_q": sparse_q,
        "sparse_block_size_kv": sparse_kv,
        "layout_q": "TND",
        "layout_kv": "PA_BNBD",
        "layout_out": "TND",
        "layout_sparse_indices": "B_N_Qb_Kb",
        "return_softmax_lse": bool(case["return_softmax_lse"]),
        "batch_size": batch,
        "num_heads_q": n1,
        "num_heads_kv": n2,
        "head_dim": HEAD_DIM,
        "sparse_mode": case["sparse_mode"],
        "seed": int(case["seed"]),
    }
    p_range = () if case["p_scale_value"] is None else (case["p_scale_value"], case["p_scale_value"])
    ranges = (
        _tuple_range(case["data_range_q"]),
        _tuple_range(case["data_range_k"]),
        _tuple_range(case["data_range_v"]),
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (0, max(0, kv_blocks - 1)),
        (0, kv_blocks),
        p_range,
        (0, total_q),
        (),
        (),
        (0, max(case["seqused_kv"])),
        (0, blocknum - 1),
        (0, 1) if int(case["mask_mode"]) == 3 else (),
        (0, 1),
    )
    return {
        "testcase_name": case["name"],
        "api_name": "bsa_ttk_ops.quant_block_sparse_attn",
        "tensor_view_shapes": repr(shapes),
        "tensor_dtypes": repr(TTK_DTYPES),
        "tensor_formats": "",
        "attributes": repr(attributes),
        "output_tensor_indexes": "",
        "golden_api": "",
        "input_data_ranges": repr(ranges),
        "precision_tolerances": repr(PRECISION_TOLERANCES),
        "absolute_precision": repr(ABSOLUTE_PRECISION),
    }


def render_ttk_csv(cases):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=TTK_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_ttk_row(case) for case in cases)
    return output.getvalue()


def _standardize_stc(row):
    cu_q = _parse_list(row["cu_seqlens_q"])
    cu_kv = _parse_list(row["cu_seqlens_kv"])
    seqused_q = _parse_list(row["seqused_q"])
    kv_lengths = _parse_list(row["seqused_kv"])
    q_lengths = [right - left for left, right in zip(cu_q, cu_q[1:])]
    normalized = dict(row)
    normalized.update(
        {
            "B": int(row["B"]),
            "N1": int(row["N1"]),
            "N2": int(row["N2"]),
            "cu_seqlens_q": cu_q,
            "cu_seqlens_kv": cu_kv,
            "seqused_q": seqused_q,
            "seqused_kv": kv_lengths,
            "S1_max": int(row.get("S1_max") or max(q_lengths)),
            "S2_max": int(row.get("S2_max") or max(kv_lengths)),
            "block_size": int(row["block_size"]),
            "blocknum": int(row["blocknum"]),
            "max_block_per_batch": int(row["max_block_per_batch"]),
            "sparse_q_block_size": int(row["sparse_q_block_size"]),
            "sparse_kv_block_size": int(row["sparse_kv_block_size"]),
            "mask_mode": int(row["mask_mode"]),
            "return_softmax_lse": str(row["return_softmax_lse"]).lower() == "true",
            "is_contiguous": str(row.get("is_contiguous", "true")).lower() == "true",
            "p_scale_value": None if not row["p_scale_value"].strip() else float(row["p_scale_value"]),
            "softmax_scale": None if not row["softmax_scale"].strip() else float(row["softmax_scale"]),
            "data_range_q": _range_value(row["data_range_q"]),
            "data_range_k": _range_value(row["data_range_k"]),
            "data_range_v": _range_value(row["data_range_v"]),
            "runtime_length_mode": (
                "zero_q" if 0 in q_lengths else "zero_kv" if 0 in kv_lengths
                else "uniform" if len(set(q_lengths)) == 1 and len(set(kv_lengths)) == 1
                else "ragged"
            ),
        }
    )
    return normalized


FACTOR_LEVELS = {
    "B": ("1", "2..31", "32", "64", "128", "256"),
    "N2": tuple(str(value) for value in range(1, 9)),
    "N1/N2": tuple(str(value) for value in range(1, 17)),
    "S bucket": ("<=4096", "4097..8192", "8193..16384", "16385..65536"),
    "S relation": ("lt", "eq", "gt"),
    "alignment": ("AA", "AU", "UA", "UU"),
    "sparse block": ("64", "128"),
    "mask": ("0", "3"),
    "LSE": ("True", "False"),
    "p_scale": ("none", "<=1", "1..100", ">100"),
    "softmax": ("none", "default", "arbitrary"),
    "runtime lengths": ("uniform", "ragged", "zero_q", "zero_kv"),
    "sparse mode": ("dense", "random"),
    "blocknum relation": ("under", "equal", "over"),
    "block table width": ("exact", "extra"),
    "range special": ("finite", "+inf", "-inf", "nan", "[-inf,inf]"),
}

GENERALIZED_FACTOR_LEVELS = dict(FACTOR_LEVELS)
GENERALIZED_FACTOR_LEVELS["B"] = ("1", "2..31", "32")
GENERALIZED_FACTOR_LEVELS["S bucket"] = ("<=4096",)


def _alignment(case):
    return ("A" if int(case["S1_max"]) % 128 == 0 else "U") + ("A" if int(case["S2_max"]) % 512 == 0 else "U")


def _range_special(value):
    if isinstance(value, (list, tuple)):
        low, high = float(value[0]), float(value[1])
        if low == -math.inf and high == math.inf:
            return "[-inf,inf]"
        if math.isnan(low) and math.isnan(high):
            return "nan"
        return "finite" if math.isfinite(low) and math.isfinite(high) else "other"
    scalar = float(value)
    if math.isnan(scalar):
        return "nan"
    if scalar == math.inf:
        return "+inf"
    if scalar == -math.inf:
        return "-inf"
    return "finite"


def _classify(case, factor):
    if factor == "B":
        value = int(case["B"])
        return str(value) if value in (1, 32, 64, 128, 256) else "2..31"
    if factor == "N2":
        return str(case["N2"])
    if factor == "N1/N2":
        return str(int(case["N1"]) // int(case["N2"]))
    if factor == "S bucket":
        value = max(int(case["S1_max"]), int(case["S2_max"]))
        return "<=4096" if value <= 4096 else "4097..8192" if value <= 8192 else "8193..16384" if value <= 16384 else "16385..65536"
    if factor == "S relation":
        s1, s2 = int(case["S1_max"]), int(case["S2_max"])
        return "eq" if s1 == s2 else "lt" if s1 < s2 else "gt"
    if factor == "alignment":
        return _alignment(case)
    if factor == "sparse block":
        return str(case["sparse_q_block_size"])
    if factor == "mask":
        return str(case["mask_mode"])
    if factor == "LSE":
        return str(bool(case["return_softmax_lse"]))
    if factor == "p_scale":
        value = case.get("p_scale_value")
        if value is None:
            return "none"
        value = float(value)
        return "<=1" if value <= 1 else "1..100" if value <= 100 else ">100"
    if factor == "softmax":
        value = case.get("softmax_scale")
        if value is None:
            return "none"
        return "default" if math.isclose(float(value), 1.0 / math.sqrt(HEAD_DIM), rel_tol=1e-9) else "arbitrary"
    if factor == "runtime lengths":
        return case["runtime_length_mode"]
    if factor == "sparse mode":
        return case["sparse_mode"]
    if factor == "blocknum relation":
        logical = max(1, sum(_ceil_div(value, int(case["block_size"])) for value in case["seqused_kv"]))
        value = int(case["blocknum"])
        return "under" if value < logical else "equal" if value == logical else "over"
    if factor == "block table width":
        required = max(_ceil_div(value, int(case["block_size"])) for value in case["seqused_kv"])
        return "exact" if int(case["max_block_per_batch"]) == required else "extra"
    if factor == "range special":
        values = {_range_special(case[name]) for name in ("data_range_q", "data_range_k", "data_range_v")}
        nonfinite = [value for value in values if value != "finite"]
        return nonfinite[0] if len(nonfinite) == 1 else "finite" if not nonfinite else "mixed_nonfinite"
    raise KeyError(factor)


def _coverage_rows(suite, cases, levels=FACTOR_LEVELS):
    rows = []
    hit_levels = 0
    total_levels = 0
    for factor, planned_levels in levels.items():
        counts = Counter(_classify(case, factor) for case in cases)
        for level in planned_levels:
            count = counts.get(level, 0)
            rows.append({"suite": suite, "factor": factor, "category": level, "count": count, "total": len(cases)})
            hit_levels += int(count > 0)
            total_levels += 1
    return rows, {"hit": hit_levels, "total": total_levels, "ratio": hit_levels / total_levels}


PAIRWISE_LEVELS = {
    "S relation × alignment": tuple(
        (left, right)
        for left in ("lt", "eq", "gt")
        for right in ("AA", "AU", "UA", "UU")
        if not (left == "eq" and right == "UA")
    ),
    "mask × LSE": tuple((left, right) for left in ("0", "3") for right in ("True", "False")),
    "N2 × N1/N2": tuple((str(left), str(right)) for left in range(1, 9) for right in range(1, 17)),
    "sparse block × PA ratio": tuple(
        (str(sparse), str(ratio))
        for sparse in (64, 128)
        for ratio in range(1, 1024 // sparse + 1)
    ),
}


def _pair_value(case, pair_name):
    if pair_name == "S relation × alignment":
        return _classify(case, "S relation"), _classify(case, "alignment")
    if pair_name == "mask × LSE":
        return _classify(case, "mask"), _classify(case, "LSE")
    if pair_name == "N2 × N1/N2":
        return _classify(case, "N2"), _classify(case, "N1/N2")
    sparse = int(case["sparse_q_block_size"])
    return str(sparse), str(int(case["block_size"]) // sparse)


def _pairwise_rows(suite, cases):
    rows = []
    hit = total = 0
    for pair_name, planned in PAIRWISE_LEVELS.items():
        counts = Counter(_pair_value(case, pair_name) for case in cases)
        for left, right in planned:
            count = counts.get((left, right), 0)
            rows.append({"suite": suite, "factor_pair": pair_name, "left": left, "right": right, "count": count})
            hit += int(count > 0)
            total += 1
    return rows, {"hit": hit, "total": total, "ratio": hit / total}


def _factor_model():
    return [
        {"id": "F01", "factor": "固定契约", "levels": "D=128; group=32; quant_mode=2", "allocation": "全部", "framework": "pytest/TTK", "note": "仅 MXFP8；不触碰 FP8"},
        {"id": "F02", "factor": "布局", "levels": "TND / PA_BNBD / B_N_Qb_Kb / TND", "allocation": "全部", "framework": "pytest/TTK", "note": "当前算子契约"},
        {"id": "F03", "factor": "B", "levels": "1; 2..31; 32", "allocation": "资源受限循环", "framework": "pytest/TTK", "note": "B=64/128/256 由现有 STC 覆盖"},
        {"id": "F04", "factor": "N2", "levels": "1..8", "allocation": "均衡循环", "framework": "pytest/TTK", "note": ""},
        {"id": "F05", "factor": "N1/N2", "levels": "1..16", "allocation": "均衡循环", "framework": "pytest/TTK", "note": "N1<=128"},
        {"id": "F06", "factor": "S1/S2", "levels": "1..4096", "allocation": "2000 条均受资源上限约束", "framework": "pytest/TTK", "note": "覆盖 64/128/512/1024/2048/4096 前后边界；更大长度留在 STC"},
        {"id": "F07", "factor": "关系/对齐", "levels": "lt/eq/gt × AA/AU/UA/UU", "allocation": "合法组合", "framework": "pytest/TTK", "note": "eq+UA 不可成立"},
        {"id": "F08", "factor": "运行时长度", "levels": "uniform/ragged/zero_q/zero_kv", "allocation": "50/46/2/2%", "framework": "pytest/TTK", "note": "总 Q 与最大 KV 非空"},
        {"id": "F09", "factor": "稀疏块", "levels": "64/128", "allocation": "50/50%", "framework": "pytest/TTK", "note": "Q/KV 相等"},
        {"id": "F10", "factor": "PA block", "levels": "sparse 的整数倍且 <=1024", "allocation": "循环全覆盖", "framework": "pytest/TTK", "note": "PA_BNBD"},
        {"id": "F11", "factor": "sparse_mode", "levels": "dense/random", "allocation": "25/75%", "framework": "pytest/TTK", "note": "旧 sparse_pattern 不再作为公开因子"},
        {"id": "F12", "factor": "blocknum", "levels": "under/equal/over logical pages", "allocation": "70/20/10%", "framework": "pytest/TTK", "note": "允许物理页复用"},
        {"id": "F13", "factor": "block table width", "levels": "exact/extra", "allocation": "75/25%", "framework": "pytest/TTK", "note": "不得小于单 batch 必需页宽"},
        {"id": "F14", "factor": "mask/LSE", "levels": "0/3 × on/off", "allocation": "30/70%; 90/10%", "framework": "pytest/TTK", "note": ""},
        {"id": "F15", "factor": "p_scale", "levels": "none; <=1; 1..100; >100", "allocation": "10%; 72/20/8%", "framework": "pytest/TTK", "note": ""},
        {"id": "F16", "factor": "softmax_scale", "levels": "none/default/arbitrary", "allocation": "10/80/10%", "framework": "pytest/TTK", "note": "none 在执行时解析为默认值"},
        {"id": "F17", "factor": "Q/K/V 有限范围", "levels": "unit/e4m3/fp16/fp32_extreme", "allocation": "80/15/4/1%", "framework": "pytest/TTK", "note": "含正/负/跨零/零"},
        {"id": "F18", "factor": "Q/K/V 非有限范围", "levels": "+inf/-inf/nan/[-inf,inf]", "allocation": "每个张量、每类数值均配单 tile/多 tile", "framework": "pytest/TTK", "note": "定向回归，不随机污染有限值精度分布"},
        {"id": "F19", "factor": "LSE 行状态", "levels": "合法 KV; 全 -inf; 真 NaN; 人工反量化溢出", "allocation": "单 tile/多 tile 与 1024/1025 归并边界", "framework": "pytest/TTK", "note": "覆盖 NaN/0、跨 tile 在线 softmax 合并问题"},
        {"id": "F20", "factor": "连续性", "levels": "paired generalized=contiguous; standalone robustness=non-contiguous", "allocation": "不计入配对覆盖率", "framework": "pytest 定向", "note": "当前 TTK CSV 无 stride 字段，避免伪造 pytest/TTK 等价覆盖"},
        {"id": "F21", "factor": "golden matmul 后端", "levels": "matmul(default)/npu_quant_matmul(--quant_matmul)", "allocation": "执行矩阵", "framework": "pytest/TTK golden", "note": "正常 CSV 不输入 quant_matmul；仅 MXFP8 生效"},
        {"id": "F22", "factor": "执行模式", "levels": "Eager/ACLGraph", "allocation": "同一 CSV 分别运行", "framework": "TTK", "note": "不复制用例"},
        {"id": "F23", "factor": "精度日志", "levels": "摘要 + 超阈值误差明细", "allocation": "全部", "framework": "pytest/TTK", "note": "执行观测因子，不计入数据覆盖率"},
    ]


def _write_atomic(path, content, check):
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"stale generated file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False,
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated files are current")
    args = parser.parse_args()

    stc_raw = _read_csv(STC_PATH)
    stc = [_standardize_stc(row) for row in stc_raw]
    fields = list(stc_raw[0].keys())
    for field in ("resource_level", "runtime_length_mode", "estimated_total_mib"):
        if field not in fields:
            fields.append(field)
    cases = build_cases()
    for case in cases:
        _validate_case(case)
    names = [case["name"] for case in cases]
    if len(names) != CASE_COUNT or len(names) != len(set(names)):
        raise AssertionError("generalized suite must contain 2000 unique cases")
    large_cases = [case for case in cases if case["resource_level"] == "large"]
    if len(large_cases) != LARGE_CASE_COUNT:
        raise AssertionError("generalized suite must not contain large cases")

    stc_suite = f"stc_{len(stc)}"
    stc_coverage, stc_factor_rate = _coverage_rows(stc_suite, stc)
    generalized_coverage, generalized_factor_rate = _coverage_rows("generalized_2000", cases, GENERALIZED_FACTOR_LEVELS)
    combined_coverage, combined_factor_rate = _coverage_rows("combined", stc + cases)
    stc_pairs, stc_pair_rate = _pairwise_rows(stc_suite, stc)
    generalized_pairs, generalized_pair_rate = _pairwise_rows("generalized_2000", cases)
    combined_pairs, combined_pair_rate = _pairwise_rows("combined", stc + cases)
    coverage = {
        "metadata": {
            "generator_seed": GENERATOR_SEED,
            "stc_count": len(stc),
            "generalized_count": len(cases),
            "large_case_count": len(large_cases),
            "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
            "maximum_estimated_mib": MAX_ESTIMATED_MIB,
            "maximum_actual_estimated_mib": max(case["estimated_total_mib"] for case in cases),
            "regression_case_count": sum(case["resource_level"] == "regression" for case in cases),
            "boundary_case_count": sum(case["resource_level"] == "boundary" for case in cases),
            "pytest_output": str(PYTEST_OUTPUT),
            "ttk_output": str(TTK_OUTPUT),
        },
        "factor_model": _factor_model(),
        "coverage_rates": {
            "stc_factor_level": stc_factor_rate,
            "generalized_factor_level": generalized_factor_rate,
            "combined_factor_level": combined_factor_rate,
            "stc_pairwise": stc_pair_rate,
            "generalized_pairwise": generalized_pair_rate,
            "combined_pairwise": combined_pair_rate,
        },
        "coverage": stc_coverage + generalized_coverage + combined_coverage,
        "pairwise": stc_pairs + generalized_pairs + combined_pairs,
        "large_cases": large_cases,
        "stc_recommendation": {
            "update_required": False,
            "reason": (
                "Current STC is complete for the planned 70 single-factor levels. Its main "
                "pairwise gap is N2 x N1/N2; the 2000-case suite fills broad head/shape pairs. "
                "Historical NaN/zero and single-tile-versus-multi-tile regressions are duplicated "
                "in the generalized suite. quant_matmul remains an execution switch, not a CSV factor."
            ),
        },
    }
    outputs = {
        PYTEST_OUTPUT: render_pytest_csv(cases, fields),
        TTK_OUTPUT: render_ttk_csv(cases),
        COVERAGE_OUTPUT: json.dumps(coverage, ensure_ascii=False, indent=2) + "\n",
    }
    for path, content in outputs.items():
        _write_atomic(path, content, args.check)
    action = "validated" if args.check else "generated"
    print(
        f"{action}: STC={len(stc)}, generalized={len(cases)}, large={len(large_cases)}, "
        f"factor={generalized_factor_rate['hit']}/{generalized_factor_rate['total']}, "
        f"pairwise={generalized_pair_rate['hit']}/{generalized_pair_rate['total']}"
    )


if __name__ == "__main__":
    main()
