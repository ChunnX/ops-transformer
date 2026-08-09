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

"""Generate deterministic MXFP8 STC and full-coverage pytest/TTK CSVs."""

import argparse
import csv
import importlib.util
import io
import json
import math
import random
import runpy
from collections import Counter
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
TESTS_DIR = CURRENT_DIR.parents[1]
TTK_DIR = TESTS_DIR / "ttk" / "bsa_fullquant_mxfp8_test"
PYTHON_CASES = CURRENT_DIR / "qbsa_mxfp8_test_cases.py"
DEFAULT_FULL_OUTPUT = CURRENT_DIR / "qbsa_mxfp8_test_cases_generalized.csv"
DEFAULT_TTK_FULL_OUTPUT = TTK_DIR / "qbsa_mxfp8_generalized_cases.csv"
DEFAULT_COVERAGE_JSON = TTK_DIR / "qbsa_mxfp8_factor_coverage.json"

FULL_CASE_COUNT = 10_000
GENERATOR_SEED = 20260810
HEAD_DIM = 128
S1_BASE_SIZE = 128
S2_BASE_SIZE = 512

PYTEST_FIELDS = (
    "name",
    "enable",
    "B",
    "N1",
    "N2",
    "D",
    "actual_seq_q",
    "actual_seq_kv",
    "S1",
    "S2",
    "s1_base_size",
    "s2_base_size",
    "block_size",
    "blocknum",
    "mask_mode",
    "fp8_dtype",
    "scale_dtype",
    "quant_group_size",
    "layout_q",
    "layout_kv",
    "layout_sparse_indices",
    "layout_out",
    "kv_cache_layout",
    "is_contiguous",
    "p_scale_mode",
    "p_scale_value",
    "softmax_scale",
    "return_softmax_lse",
    "empty_actual_seq",
    "seed",
    "data_range_q",
    "data_range_k",
    "data_range_v",
    "device_id",
    "sparse_q_block_size",
    "sparse_kv_block_size",
    "sparse_count",
    "sparse_pattern",
    "block_table_pattern",
    "quant_mode",
    "case_group",
)

TTK_FIELDS = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "input_data_ranges",
    "attributes",
    "is_enabled",
    "remark",
    "priority",
)


def _schedule(rng, counts):
    values = []
    for value, count in counts.items():
        values.extend([value] * int(count))
    rng.shuffle(values)
    return values


def _load_current_stc():
    namespace = runpy.run_path(str(PYTHON_CASES))
    return [dict(case) for case in namespace["TestCases"] if case.get("enable") is True]


def _case_defaults(case):
    normalized = {
        "enable": True,
        "D": HEAD_DIM,
        "s1_base_size": S1_BASE_SIZE,
        "s2_base_size": S2_BASE_SIZE,
        "mask_mode": 3,
        "fp8_dtype": "float8_e4m3fn",
        "scale_dtype": "float8_e8m0fnu",
        "quant_group_size": 32,
        "layout_q": "TND",
        "layout_kv": "PA_BNSD",
        "layout_sparse_indices": "B_N_Qb_Kb",
        "layout_out": "TND",
        "kv_cache_layout": "BnNBsD",
        "is_contiguous": True,
        "p_scale_value": 1.0,
        "p_scale_mode": "value",
        "softmax_scale": None,
        "return_softmax_lse": True,
        "empty_actual_seq": False,
        "device_id": 0,
        "sparse_q_block_size": 128,
        "sparse_kv_block_size": 128,
        "sparse_pattern": "sequential",
        "block_table_pattern": "sequential",
        "quant_mode": 2,
        "case_group": "generated",
    }
    normalized.update(case)
    if "p_scale_mode" not in case:
        normalized["p_scale_mode"] = (
            "none" if normalized.get("p_scale_value") is None else "value"
        )
    normalized["B"] = int(normalized["B"])
    normalized["N1"] = int(normalized["N1"])
    normalized["N2"] = int(normalized["N2"])
    normalized["S1"] = int(normalized["S1"])
    normalized["S2"] = int(normalized["S2"])
    normalized["block_size"] = int(normalized["block_size"])
    normalized["actual_seq_q"] = [int(value) for value in normalized["actual_seq_q"]]
    normalized["actual_seq_kv"] = [int(value) for value in normalized["actual_seq_kv"]]
    if normalized.get("blocknum") is None or int(normalized["blocknum"]) <= 0:
        normalized["blocknum"] = sum(
            math.ceil(value / normalized["block_size"])
            for value in normalized["actual_seq_kv"]
        )
    normalized["blocknum"] = max(1, int(normalized["blocknum"]))
    return normalized


def _sample_aligned(rng, low, high, base, aligned):
    if aligned:
        first = math.ceil(low / base) * base
        last = (high // base) * base
        if first > last:
            raise ValueError(f"no aligned value for [{low}, {high}] base={base}")
        return rng.randrange(first // base, last // base + 1) * base
    for _ in range(1000):
        value = rng.randint(low, high)
        if value % base != 0:
            return value
    raise RuntimeError("failed to sample unaligned length")


def _sample_sequence_pair(rng, bucket, relation, alignment, require_partial_sparse, sparse_size):
    bounds = {
        "le4096": (1, 4096),
        "le8192": (4097, 8192),
        "le16384": (8193, 16384),
        "le65536": (16385, 65536),
    }
    low, high = bounds[bucket]
    q_aligned, kv_aligned = alignment[0] == "A", alignment[1] == "A"
    if relation == "eq":
        if alignment == "UA":
            alignment = "UU"
            q_aligned = kv_aligned = False
        base = 512 if alignment == "AA" else 128 if alignment == "AU" else 1
        for _ in range(1000):
            value = (
                rng.randint(low, high)
                if base == 1
                else _sample_aligned(rng, low, high, base, True)
            )
            if alignment == "AU" and value % 512 == 0:
                continue
            if alignment == "UU" and (value % 128 == 0 or value % 512 == 0):
                continue
            if not require_partial_sparse or math.ceil(value / sparse_size) >= 2:
                return value, value, alignment
        raise RuntimeError("failed to sample equal sequence pair")

    for _ in range(5000):
        q_len = _sample_aligned(rng, low, high, 128, q_aligned)
        kv_len = _sample_aligned(rng, low, high, 512, kv_aligned)
        if relation == "lt" and q_len >= kv_len:
            continue
        if relation == "gt" and q_len <= kv_len:
            continue
        if require_partial_sparse and math.ceil(kv_len / sparse_size) < 2:
            continue
        return q_len, kv_len, alignment
    raise RuntimeError(
        f"failed to sample sequence pair bucket={bucket}, relation={relation}, alignment={alignment}"
    )


def _runtime_lengths(rng, batch, maximum, mode):
    if mode == "uniform":
        return [maximum] * batch
    values = [rng.randint(1, maximum) for _ in range(batch)]
    values[rng.randrange(batch)] = maximum
    if mode == "q_zero" or mode == "kv_zero":
        zero_index = rng.randrange(batch)
        values[zero_index] = 0
        if max(values) != maximum:
            values[(zero_index + 1) % batch] = maximum
    return values


def _data_range(rng, bucket):
    ranges = {
        "le1000": (1.0, 1000.0),
        "le10000": (1001.0, 10000.0),
        "le500000": (10001.0, 500000.0),
        "other": (500001.0, 1000000.0),
    }
    low_mag, high_mag = ranges[bucket]
    magnitude = rng.uniform(low_mag, high_mag)
    sign_mode = rng.choices(
        ("cross", "positive", "negative"), weights=(70, 15, 15), k=1
    )[0]
    asymmetry = rng.uniform(0.01, 1.0)
    if sign_mode == "positive":
        low, high = magnitude * asymmetry, magnitude
    elif sign_mode == "negative":
        low, high = -magnitude, -magnitude * asymmetry
    else:
        low, high = -magnitude * asymmetry, magnitude
    return [round(low, 8), round(high, 8)]


def _pscale_value(rng, category):
    if category == "le1":
        return float(2 ** rng.randint(-20, 0))
    if category == "le100":
        return float(2 ** rng.randint(1, 6))
    return float(2 ** rng.randint(7, 12))


def build_full_cases(count=FULL_CASE_COUNT):
    if count != FULL_CASE_COUNT:
        raise ValueError(f"the factor model is calibrated for exactly {FULL_CASE_COUNT} cases")
    rng = random.Random(GENERATOR_SEED)
    schedules = {
        "bucket": _schedule(rng, {"le4096": 9990, "le8192": 4, "le16384": 4, "le65536": 2}),
        "sparse_size": _schedule(rng, {64: 4000, 128: 6000}),
        "lse": _schedule(rng, {True: 9000, False: 1000}),
        "mask": _schedule(rng, {0: 2000, 3: 8000}),
        "block_relation": _schedule(rng, {"under": 8000, "equal": 1500, "over": 500}),
        "p_mode": _schedule(rng, {"none": 100, "empty": 100, "value": 9800}),
        "softmax": _schedule(rng, {"default": 8000, "none": 1000, "arbitrary": 1000}),
        "length_mode": _schedule(rng, {"uniform": 5000, "ragged": 4800, "q_zero": 100, "kv_zero": 100}),
        "sparse_class": _schedule(rng, {"empty": 100, "dense": 2500, "partial": 7400}),
        "relation": _schedule(rng, {"lt": 3334, "eq": 3333, "gt": 3333}),
        "q_range": _schedule(rng, {"le1000": 8000, "le10000": 1000, "le500000": 500, "other": 500}),
        "k_range": _schedule(rng, {"le1000": 8000, "le10000": 1000, "le500000": 500, "other": 500}),
        "v_range": _schedule(rng, {"le1000": 8000, "le10000": 1000, "le500000": 500, "other": 500}),
    }
    schedules["b_class"] = ["small"] * count
    small_bucket_indexes = [
        index for index, value in enumerate(schedules["bucket"])
        if value == "le4096"
    ]
    rng.shuffle(small_bucket_indexes)
    for offset, value in enumerate(["64", "128", "256"]):
        schedules["b_class"][small_bucket_indexes[offset]] = value

    schedules["p_class"] = ["unused"] * count
    value_indexes = [
        index for index, value in enumerate(schedules["p_mode"])
        if value == "value"
    ]
    p_classes = (["le1"] * 7840) + (["le100"] * 1764) + (["other"] * 196)
    rng.shuffle(p_classes)
    for index, value in zip(value_indexes, p_classes):
        schedules["p_class"][index] = value
    alignments = ["AA", "AU", "UA", "UU"]
    sparse_patterns = ["sequential", "reverse", "random", "tail"]
    block_patterns = ["sequential", "reverse", "random"]
    pa_sizes = {
        64: list(range(64, 1025, 64)),
        128: list(range(128, 1025, 128)),
    }

    cases = []
    for index in range(count):
        sparse_size = schedules["sparse_size"][index]
        bucket = schedules["bucket"][index]
        relation = schedules["relation"][index]
        alignment = rng.choice(
            ["AA", "AU", "UU"] if relation == "eq" else alignments
        )
        sparse_class = schedules["sparse_class"][index]
        s1, s2, alignment = _sample_sequence_pair(
            rng,
            bucket,
            relation,
            alignment,
            sparse_class == "partial",
            sparse_size,
        )

        b_class = schedules["b_class"][index]
        batch = rng.randint(1, 32) if b_class == "small" else int(b_class)
        length_mode = schedules["length_mode"][index]
        if length_mode in ("ragged", "q_zero", "kv_zero") and batch == 1:
            batch = 2
        n2 = index % 8 + 1
        ratio = (index // 8) % 16 + 1
        n1 = n2 * ratio

        # Keep every individual case executable without dropping factor coverage.
        if bucket != "le4096":
            batch = 2 if length_mode in ("q_zero", "kv_zero") else 1
            n2 = min(n2, 2)
            ratio = min(ratio, 4)
            n1 = n2 * ratio
        if batch > 32:
            n2 = 1
            ratio = min(ratio, 2)
            n1 = ratio
            s1 = min(s1, 128)
            s2 = min(s2, 128)
        if n1 > 64:
            batch = min(batch, 2)
            s1 = min(s1, 512)
            s2 = min(s2, 512)

        if sparse_class == "partial" and math.ceil(s2 / sparse_size) < 2:
            s2 = sparse_size + 1
        if relation == "eq":
            value = max(s1, s2)
            s1 = value
            s2 = value
        elif relation == "lt" and s1 >= s2:
            s1 = max(1, s2 - 1)
        elif relation == "gt" and s1 <= s2:
            s1 = s2 + 1

        q_mode = "q_zero" if length_mode == "q_zero" else "ragged" if length_mode == "ragged" else "uniform"
        kv_mode = "kv_zero" if length_mode == "kv_zero" else "ragged" if length_mode == "ragged" else "uniform"
        q_lengths = _runtime_lengths(rng, batch, s1, q_mode)
        kv_lengths = _runtime_lengths(rng, batch, s2, kv_mode)

        page_size = rng.choice(pa_sizes[sparse_size])
        logical_pages = sum(math.ceil(value / page_size) for value in kv_lengths)
        logical_pages = max(1, logical_pages)
        block_relation = schedules["block_relation"][index]
        if block_relation == "under" and logical_pages == 1:
            batch = max(batch, 3 if length_mode == "kv_zero" else 2)
            q_lengths = (q_lengths + [s1] * batch)[:batch]
            kv_lengths = (kv_lengths + [s2] * batch)[:batch]
            kv_lengths[0] = s2
            kv_lengths[1] = s2
            if length_mode == "q_zero":
                q_lengths[-1] = 0
            if length_mode == "kv_zero":
                kv_lengths[-1] = 0
            logical_pages = sum(math.ceil(value / page_size) for value in kv_lengths)
        if block_relation == "under":
            blocknum = rng.randint(1, logical_pages - 1)
        elif block_relation == "equal":
            blocknum = logical_pages
        else:
            blocknum = rng.randint(logical_pages + 1, max(logical_pages + 1, logical_pages * 2))

        max_sparse_blocks = max(1, math.ceil(max(kv_lengths) / sparse_size))
        if sparse_class == "empty":
            sparse_count = 0
        elif sparse_class == "dense":
            sparse_count = max_sparse_blocks
        else:
            sparse_count = rng.randint(1, max(1, max_sparse_blocks - 1))

        p_mode = schedules["p_mode"][index]
        p_value = None if p_mode != "value" else _pscale_value(rng, schedules["p_class"][index])
        softmax_class = schedules["softmax"][index]
        softmax_scale = (
            None
            if softmax_class == "none"
            else 1.0 / math.sqrt(HEAD_DIM)
            if softmax_class == "default"
            else rng.uniform(0.000001, 0.999999)
        )

        case = _case_defaults(
            {
                "name": f"GEN_{index + 1:05d}_B{batch}_N{n1}_{n2}_S{sparse_size}_PA{page_size}",
                "B": batch,
                "N1": n1,
                "N2": n2,
                "S1": max(q_lengths),
                "S2": max(kv_lengths),
                "actual_seq_q": q_lengths,
                "actual_seq_kv": kv_lengths,
                "block_size": page_size,
                "blocknum": blocknum,
                "mask_mode": schedules["mask"][index],
                "p_scale_mode": p_mode,
                "p_scale_value": p_value,
                "softmax_scale": softmax_scale,
                "return_softmax_lse": schedules["lse"][index],
                "seed": 50000 + index,
                "data_range_q": _data_range(rng, schedules["q_range"][index]),
                "data_range_k": _data_range(rng, schedules["k_range"][index]),
                "data_range_v": _data_range(rng, schedules["v_range"][index]),
                "sparse_q_block_size": sparse_size,
                "sparse_kv_block_size": sparse_size,
                "sparse_count": sparse_count,
                "sparse_pattern": sparse_patterns[index % len(sparse_patterns)],
                "block_table_pattern": block_patterns[index % len(block_patterns)],
                "case_group": f"full_{bucket}_{relation}_{alignment}_{sparse_class}_{block_relation}",
            }
        )
        cases.append(case)
    return cases


def validate_case(case):
    name = case["name"]
    if int(case["B"]) <= 0:
        raise ValueError(f"{name}: B must be positive")
    if int(case["N2"]) < 1 or int(case["N2"]) > 8:
        raise ValueError(f"{name}: N2 must be in [1, 8]")
    if int(case["N1"]) > 128 or int(case["N1"]) % int(case["N2"]) != 0:
        raise ValueError(f"{name}: N1 must be <=128 and divisible by N2")
    if int(case["N1"]) // int(case["N2"]) > 16:
        raise ValueError(f"{name}: N1/N2 must be <=16")
    if case["layout_q"] != "TND" or case["layout_kv"] != "PA_BNSD":
        raise ValueError(f"{name}: unsupported layout")
    sparse_q = int(case["sparse_q_block_size"])
    sparse_kv = int(case["sparse_kv_block_size"])
    page_size = int(case["block_size"])
    if sparse_q != sparse_kv or sparse_q not in (64, 128):
        raise ValueError(f"{name}: sparse block size must be equal and 64/128")
    if page_size <= 0 or page_size > 1024 or page_size % sparse_kv != 0:
        raise ValueError(f"{name}: invalid PA block size {page_size}")
    q_lengths = [int(value) for value in case["actual_seq_q"]]
    kv_lengths = [int(value) for value in case["actual_seq_kv"]]
    if len(q_lengths) != int(case["B"]) or len(kv_lengths) != int(case["B"]):
        raise ValueError(f"{name}: runtime length list must have B values")
    if min(q_lengths + kv_lengths) < 0 or sum(q_lengths) <= 0 or max(kv_lengths) <= 0:
        raise ValueError(f"{name}: lengths must be non-negative with non-empty total Q/KV")
    if max(q_lengths) != int(case["S1"]) or max(kv_lengths) != int(case["S2"]):
        raise ValueError(f"{name}: S1/S2 must equal runtime maxima")
    if int(case["sparse_count"]) < 0:
        raise ValueError(f"{name}: invalid sparse_count")
    logical_pages = max(1, sum(math.ceil(value / page_size) for value in kv_lengths))
    if int(case["blocknum"]) < 1 or int(case["blocknum"]) > logical_pages * 2:
        raise ValueError(f"{name}: blocknum must be in [1, 2 * logical pages]")
    p_mode = case.get("p_scale_mode")
    if p_mode not in ("none", "empty", "value"):
        raise ValueError(f"{name}: invalid p_scale_mode {p_mode!r}")
    if p_mode == "value" and (
        case.get("p_scale_value") is None or float(case["p_scale_value"]) <= 0
    ):
        raise ValueError(f"{name}: value p_scale must be positive")
    if p_mode != "value" and case.get("p_scale_value") is not None:
        raise ValueError(f"{name}: absent/empty p_scale must not carry a value")


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def render_pytest_csv(cases):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PYTEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for raw_case in cases:
        case = _case_defaults(raw_case)
        validate_case(case)
        writer.writerow({field: _csv_value(case.get(field)) for field in PYTEST_FIELDS})
    return output.getvalue()


def ttk_row(raw_case):
    case = _case_defaults(raw_case)
    validate_case(case)
    batch = int(case["B"])
    n1 = int(case["N1"])
    n2 = int(case["N2"])
    head_dim = int(case["D"])
    page_size = int(case["block_size"])
    sparse_q = int(case["sparse_q_block_size"])
    sparse_kv = int(case["sparse_kv_block_size"])
    q_lengths = case["actual_seq_q"]
    kv_lengths = case["actual_seq_kv"]
    total_q = sum(q_lengths)
    physical_pages = int(case["blocknum"])
    q_blocks = math.ceil(max(q_lengths) / sparse_q)
    kv_blocks = math.ceil(max(kv_lengths) / sparse_kv)
    p_mode = case["p_scale_mode"]
    p_shape = None if p_mode == "none" else (0,) if p_mode == "empty" else (1,)
    p_dtype = None if p_mode == "none" else "float8_e8m0"
    shapes = (
        (total_q, n1, head_dim),
        (physical_pages, n2, page_size, head_dim),
        (physical_pages, n2, page_size, head_dim),
        (total_q, n1, head_dim // 64, 2),
        (physical_pages, n2, page_size, head_dim // 64, 2),
        (physical_pages, n2, math.ceil(page_size / 64), head_dim, 2),
        (batch, n1, q_blocks, kv_blocks),
        (batch, n1, q_blocks),
        (2048, 2048),
        p_shape,
    )
    dtypes = (
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float8_e8m0",
        "float8_e8m0",
        "float8_e8m0",
        "int32",
        "int32",
        "uint8",
        p_dtype,
    )
    softmax_scale = case.get("softmax_scale")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    attributes = {
        "softmax_scale": float(softmax_scale),
        "sparse_q_block_size": sparse_q,
        "sparse_kv_block_size": sparse_kv,
        "layout_kv": case["layout_kv"],
        "layout_q": case["layout_q"],
        "layout_sparse_indices": case["layout_sparse_indices"],
        "layout_out": case["layout_out"],
        "quant_mode": 2,
        "mask_mode": int(case["mask_mode"]),
        "return_softmax_lse": bool(case["return_softmax_lse"]),
        "actual_seq_q": q_lengths,
        "actual_seq_kv": kv_lengths,
        "empty_actual_seq": bool(case["empty_actual_seq"]),
        "block_table_pattern": case["block_table_pattern"],
        "seed": int(case["seed"]),
        "sparse_count": int(case["sparse_count"]),
        "sparse_pattern": case["sparse_pattern"],
        "p_scale_value": (
            None if p_mode != "value" else float(case["p_scale_value"])
        ),
        "quant_group_size": int(case["quant_group_size"]),
        "s2_base_size": int(case["s2_base_size"]),
        "blocknum": physical_pages,
        "data_range_q": case["data_range_q"],
        "data_range_k": case["data_range_k"],
        "data_range_v": case["data_range_v"],
    }
    return {
        "testcase_name": case["name"],
        "api_name": "bsa_ttk_ops.quant_block_sparse_attn",
        "tensor_view_shapes": repr(shapes),
        "tensor_dtypes": repr(dtypes),
        "input_data_ranges": "",
        "attributes": repr(attributes),
        "is_enabled": "True",
        "remark": f"case_group={case['case_group']}; p_scale_mode={p_mode}",
        "priority": "0",
    }


def render_ttk_csv(cases):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=TTK_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(ttk_row(case) for case in cases)
    return output.getvalue()


def _relation(case):
    return "eq" if case["S1"] == case["S2"] else "lt" if case["S1"] < case["S2"] else "gt"


def _alignment(case):
    return ("A" if case["S1"] % 128 == 0 else "U") + ("A" if case["S2"] % 512 == 0 else "U")


def _range_bucket(value):
    low, high = (-abs(float(value)), abs(float(value))) if isinstance(value, (int, float)) else (float(value[0]), float(value[1]))
    magnitude = max(abs(low), abs(high))
    if magnitude <= 1000:
        return "<=1000"
    if magnitude <= 10000:
        return "1000..10000"
    if magnitude <= 500000:
        return "10000..500000"
    return "other"


def _range_sign(value):
    low, high = (-abs(float(value)), abs(float(value))) if isinstance(value, (int, float)) else (float(value[0]), float(value[1]))
    return "positive" if low >= 0 else "negative" if high <= 0 else "cross_zero"


def _classify(case, factor):
    if factor == "B":
        value = int(case["B"])
        return str(value) if value in (1, 32, 64, 128, 256) else "2..31" if 2 <= value <= 31 else "other"
    if factor == "N2":
        return str(case["N2"])
    if factor == "N1/N2":
        return str(int(case["N1"]) // int(case["N2"]))
    if factor == "S bucket":
        value = max(int(case["S1"]), int(case["S2"]))
        return "<=4096" if value <= 4096 else "4097..8192" if value <= 8192 else "8193..16384" if value <= 16384 else "16385..65536" if value <= 65536 else ">65536"
    if factor == "S relation":
        return _relation(case)
    if factor == "alignment":
        return _alignment(case)
    if factor == "sparse block":
        return str(case["sparse_q_block_size"])
    if factor == "PA block":
        return str(case["block_size"])
    if factor == "mask":
        return str(case["mask_mode"])
    if factor == "LSE":
        return str(bool(case["return_softmax_lse"]))
    if factor == "p_scale mode":
        return case.get("p_scale_mode", "none" if case.get("p_scale_value") is None else "value")
    if factor == "p_scale value":
        mode = case.get("p_scale_mode", "none" if case.get("p_scale_value") is None else "value")
        if mode != "value":
            return "not_applicable"
        value = float(case["p_scale_value"])
        return "<=1" if value <= 1 else "1..100" if value <= 100 else ">100"
    if factor == "softmax":
        value = case.get("softmax_scale")
        if value is None:
            return "none"
        return "default" if math.isclose(float(value), 1.0 / math.sqrt(HEAD_DIM), rel_tol=1e-8) else "arbitrary"
    if factor == "runtime lengths":
        q_values, kv_values = case["actual_seq_q"], case["actual_seq_kv"]
        if 0 in q_values:
            return "partial_zero_q"
        if 0 in kv_values:
            return "partial_zero_kv"
        return "uniform" if len(set(q_values)) == 1 and len(set(kv_values)) == 1 else "ragged"
    if factor == "sparse class":
        maximum = math.ceil(max(case["actual_seq_kv"]) / case["sparse_kv_block_size"])
        count = int(case["sparse_count"])
        return "empty" if count == 0 else "dense" if count >= maximum else "partial"
    if factor == "blocknum relation":
        logical = max(1, sum(math.ceil(value / case["block_size"]) for value in case["actual_seq_kv"]))
        return "under" if case["blocknum"] < logical else "equal" if case["blocknum"] == logical else "over"
    if factor == "sparse pattern":
        return case["sparse_pattern"]
    if factor == "block table pattern":
        return case["block_table_pattern"]
    if factor.endswith(" range"):
        return _range_bucket(case[f"data_range_{factor[0].lower()}"])
    if factor.endswith(" sign"):
        return _range_sign(case[f"data_range_{factor[0].lower()}"])
    raise KeyError(factor)


def coverage_rows(suite_name, cases):
    factors = (
        "B", "N2", "N1/N2", "S bucket", "S relation", "alignment",
        "sparse block", "PA block", "mask", "LSE", "p_scale mode", "p_scale value",
        "softmax", "runtime lengths", "sparse class", "blocknum relation",
        "sparse pattern", "block table pattern", "q range", "k range", "v range",
        "q sign", "k sign", "v sign",
    )
    rows = []
    for factor in factors:
        counts = Counter(_classify(case, factor) for case in cases)
        for category, count in sorted(counts.items(), key=lambda item: item[0]):
            rows.append(
                {
                    "suite": suite_name,
                    "factor": factor,
                    "category": category,
                    "count": count,
                    "total": len(cases),
                    "ratio": count / len(cases),
                }
            )
    return rows


def pairwise_rows(suite_name, cases):
    pair_defs = (
        ("S relation", "alignment"),
        ("sparse block", "PA block"),
        ("mask", "LSE"),
        ("sparse pattern", "block table pattern"),
        ("N2", "N1/N2"),
    )
    rows = []
    for left, right in pair_defs:
        counts = Counter(
            (_classify(case, left), _classify(case, right)) for case in cases
        )
        for (left_value, right_value), count in sorted(counts.items()):
            rows.append(
                {
                    "suite": suite_name,
                    "factor_pair": f"{left} × {right}",
                    "left": left_value,
                    "right": right_value,
                    "count": count,
                }
            )
    return rows


def build_factor_model():
    return [
        {"factor": "q/k/v range", "model": "80% <=1e3; 10% <=1e4; 5% <=5e5; 5% other; asymmetric and one-sided", "constraint": "finite FP32 source; quantized to E4M3 + E8M0 descale", "assessment": "complete after clarifying that 5e5 exceeds FP16 range"},
        {"factor": "B", "model": "9,997 cases use B=1..32; B=64/128/256 each has one generalized boundary case", "constraint": "B>0; large B paired with small S/N", "assessment": "STC remains unchanged; B=256 is full-only"},
        {"factor": "N1/N2", "model": "N2=1..8; ratio=1..16; N1<=128", "constraint": "N1 % N2 == 0", "assessment": "complete"},
        {"factor": "S1/S2", "model": "9,990 cases <=4096; 8 cases are in (4096,16384]; 2 cases are in (16384,65536]", "constraint": "runtime maxima equal S1/S2", "assessment": "large-S buckets are full-only; STC remains unchanged"},
        {"factor": "relation/alignment", "model": "lt/eq/gt × AA/AU/UA/UU", "constraint": "eq+UA is mathematically impossible", "assessment": "all valid combinations already covered"},
        {"factor": "sparse/PA block", "model": "sparse=64/128 (40/60); PA is a positive multiple and <=1024", "constraint": "sparse_q == sparse_kv; PA % sparse == 0", "assessment": "complete"},
        {"factor": "sparse_count", "model": "empty/dense/partial; per-Qb effective count may differ", "constraint": "0 <= sparse_count; runtime uses min(requested, available)", "assessment": "fully empty sparse is full-only; STC remains unchanged"},
        {"factor": "blocknum", "model": "80% under / 15% equal / 5% over logical PA pages", "constraint": "baseline=sum(ceil(seq_kv[b]/PA)); 1 <= blocknum <= 2*baseline", "assessment": "definition clarified; current coverage exists"},
        {"factor": "p_scale", "model": "100 none; 100 empty tensor; remaining values: 80% <=1, 18% <=100, 2% >100", "constraint": "value mode must be positive for CPU log semantics", "assessment": "empty tensor and >100 are full-only; STC remains unchanged"},
        {"factor": "runtime lengths", "model": "uniform/ragged plus partial zero Q/KV batches", "constraint": "B>0; total Q and max KV remain positive", "assessment": "partial zero exists but needs symmetric Q/KV cases"},
        {"factor": "patterns", "model": "sparse sequential/reverse/random/tail; block-table sequential/reverse/random", "constraint": "runtime keyword is reverse, not reversed", "assessment": "complete"},
        {"factor": "execution mode", "model": "Eager and ACLGraph reuse the same case table", "constraint": "execution dimension is measured separately from data factors", "assessment": "must run both suites; not encoded as duplicate cases"},
        {"factor": "layout/stride", "model": "Q/Q-descale TND; KV uses segmented PA_BNSD stride", "constraint": "additional arbitrary non-contiguous views are robustness tests", "assessment": "add as a separate robustness suite, not normal STC/full"},
        {"factor": "fixed contract", "model": "D=128, group=32, s1_base=128, s2_base=512, quant_mode=2", "constraint": "currently fixed by kernel", "assessment": "should be listed explicitly in factor table"},
    ]


def _write_or_check(path, content, check):
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} is stale; regenerate it")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    current = [_case_defaults(case) for case in _load_current_stc()]
    full = build_full_cases()
    if len(current) != 71 or len(full) != FULL_CASE_COUNT:
        raise AssertionError(
            f"unexpected suite sizes: current={len(current)}, full={len(full)}"
        )
    for suite in (current, full):
        names = [case["name"] for case in suite]
        if len(names) != len(set(names)):
            raise ValueError("duplicate testcase name")
        for case in suite:
            validate_case(case)

    outputs = {
        DEFAULT_FULL_OUTPUT: render_pytest_csv(full),
        DEFAULT_TTK_FULL_OUTPUT: render_ttk_csv(full),
    }
    coverage = {
        "metadata": {
            "generator_seed": GENERATOR_SEED,
            "current_stc_count": len(current),
            "added_stc_count": 0,
            "revised_stc_count": len(current),
            "full_count": len(full),
        },
        "factor_model": build_factor_model(),
        "coverage": coverage_rows("current_stc", current)
        + coverage_rows("generalized_10000", full),
        "pairwise": pairwise_rows("current_stc", current)
        + pairwise_rows("generalized_10000", full),
        "added_stc": [],
        "full_sample": full[:100],
        "large_s_cases": [
            case for case in full if max(case["S1"], case["S2"]) > 4096
        ],
        "large_b_cases": [case for case in full if case["B"] > 32],
    }
    outputs[DEFAULT_COVERAGE_JSON] = json.dumps(
        coverage, ensure_ascii=False, indent=2
    ) + "\n"
    for path, content in outputs.items():
        _write_or_check(path, content, args.check)
    action = "validated" if args.check else "generated"
    print(
        f"{action}: current STC={len(current)}, added STC=0, "
        f"full={len(full)}, files={len(outputs)}"
    )


if __name__ == "__main__":
    main()
