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

"""Normalize generated ST rows and canndev seeds into executable, deduplicated TTK suites."""

import argparse
import ast
import copy
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path


API = "aclnnMoeTokenUnpermuteWithEp"
FIELDNAMES = [
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "tensor_storage_shapes",
    "tensor_view_offsets",
    "tensor_view_strides",
    "attributes",
    "output_tensor_indexes",
    "precision_tolerances",
    "absolute_precision",
    "input_data_ranges",
]
FLOAT_DTYPES = ("float16", "bfloat16", "float32")
TOPKS = (1, 2, 8, 511, 512)
HIDDENS = (1, 7, 15, 16, 17, 31, 32, 33, 41, 63, 64, 65, 81, 255, 256, 257, 511, 512, 513, 1024, 2048, 4096, 7619, 7680)
TOKEN_COUNTS = (1, 2, 3, 4, 17, 64, 256, 1024)
RANGE_MODES = ("fallback", "full", "partial", "nonzero_start", "boundary_low", "boundary_high", "outside_zero")


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _literal(value, default):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return default


def _tolerance(dtype):
    value = 1e-4 if dtype == "float32" else 1e-3
    return repr(((value, value),))


def _range_layout(token_num, topk, range_mode, has_probs):
    slots = token_num * topk if (has_probs or range_mode != "fallback") else token_num
    if range_mode == "fallback":
        return slots, slots, None, token_num if has_probs else slots
    if range_mode == "full":
        start, end = 0, slots
    elif range_mode == "partial":
        span = max(1, slots // 2)
        start, end = (slots - span) // 2, (slots - span) // 2 + span
    elif range_mode == "nonzero_start":
        span = max(1, slots // 3)
        start = min(max(1, slots // 4), max(0, slots - span))
        end = start + span
    elif range_mode == "boundary_low":
        start, end = 0, 1
    elif range_mode == "boundary_high":
        start, end = max(0, slots - 1), slots
    else:
        span = max(1, slots // 4)
        start = min(1, max(0, slots - span))
        end = start + span
    return slots, end - start, [start, end], token_num


def _noncontiguous_layout(shapes):
    strides, storages, offsets = [], [], []
    for shape in shapes:
        if shape is None:
            strides.append(None)
            storages.append(None)
            offsets.append(0)
        elif len(shape) == 1:
            strides.append((2,))
            storages.append((max(1, shape[0] * 2),))
            offsets.append(0)
        elif len(shape) == 2:
            strides.append((shape[1] * 2, 2))
            storages.append((shape[0], max(1, shape[1] * 2)))
            offsets.append(0)
        else:
            strides.append(())
            storages.append(shape)
            offsets.append(0)
    return repr(tuple(strides)), repr(tuple(storages)), repr(tuple(offsets))


def make_case(
    name,
    token_dtype="float16",
    probs_dtype="float16",
    topk=2,
    hidden=17,
    token_num=4,
    range_mode="partial",
    profile="random",
    noncontiguous=False,
    token_shape=None,
    sorted_shape=None,
    probs_shape="auto",
    out_shape=None,
    dtypes=None,
    formats=None,
    attrs=None,
):
    has_probs = probs_dtype is not None and probs_shape is not None
    slots, token_rows, range_value, output_rows = _range_layout(token_num, topk, range_mode, has_probs)
    token_shape = tuple(token_shape) if token_shape is not None else (token_rows, hidden)
    sorted_shape = tuple(sorted_shape) if sorted_shape is not None else (slots,)
    if probs_shape == "auto":
        probs_shape = (token_num, topk) if has_probs else None
    elif probs_shape is not None:
        probs_shape = tuple(probs_shape)
    out_shape = tuple(out_shape) if out_shape is not None else (output_rows, hidden)
    shapes = (token_shape, sorted_shape, probs_shape, out_shape)
    if dtypes is None:
        dtypes = (token_dtype, "int32", probs_dtype or token_dtype, token_dtype)
    if formats is None:
        formats = ("ND",)
    if attrs is None:
        attrs = {
            "numTopk": int(topk),
            "rangeOptional": range_value,
            "paddedMode": False,
            "restoreShapeOptional": None,
        }
    token_range = (-1, 1)
    probs_range = (0.1, 1)
    input_ranges = (token_range, (0, max(0, slots - 1)), probs_range if probs_shape is not None else (0, 0))
    storage_shapes = view_offsets = view_strides = ""
    if noncontiguous:
        view_strides, storage_shapes, view_offsets = _noncontiguous_layout(shapes)
    row = {
        "testcase_name": name,
        "api_name": API,
        "tensor_view_shapes": repr(shapes),
        "tensor_dtypes": repr(tuple(dtypes)),
        "tensor_formats": repr(tuple(formats)),
        "tensor_storage_shapes": storage_shapes,
        "tensor_view_offsets": view_offsets,
        "tensor_view_strides": view_strides,
        "attributes": repr(attrs),
        "output_tensor_indexes": "(3,)",
        "precision_tolerances": _tolerance(dtypes[3]),
        "absolute_precision": "0.0001",
        "input_data_ranges": repr(input_ranges),
    }
    meta = {
        "profile": profile,
        "range_mode": range_mode,
        "token_dtype": dtypes[0],
        "probs_dtype": None if probs_shape is None else dtypes[2],
        "topk": attrs.get("numTopk"),
        "hidden": token_shape[-1] if len(token_shape) == 2 else None,
        "token_num": out_shape[0] if len(out_shape) == 2 else None,
        "noncontiguous": noncontiguous,
    }
    return row, meta


def execution_signature(row, meta):
    fields = {key: row.get(key, "") for key in FIELDNAMES if key not in ("testcase_name",)}
    fields["custom_input_profile"] = meta.get("profile", "random")
    return _sha(fields)


def dedupe(entries, forbidden=None):
    forbidden = set(forbidden or ())
    seen, result = set(), []
    for row, meta in entries:
        sig = execution_signature(row, meta)
        if sig in seen or sig in forbidden:
            continue
        seen.add(sig)
        meta = dict(meta, execution_signature=sig)
        result.append((row, meta))
    return result


def read_old_l0(path):
    entries = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {key: row.get(key, "") for key in FIELDNAMES}
            normalized["tensor_storage_shapes"] = ""
            normalized["tensor_view_offsets"] = ""
            normalized["tensor_view_strides"] = ""
            shapes = _literal(normalized["tensor_view_shapes"], ())
            dtypes = _literal(normalized["tensor_dtypes"], ())
            attrs = _literal(normalized["attributes"], {})
            probs_shape = shapes[2] if len(shapes) > 2 else None
            range_value = attrs.get("rangeOptional")
            range_mode = "fallback" if range_value in (None, [0, 0]) else "baseline"
            topk = int(probs_shape[1]) if probs_shape is not None else int(attrs.get("numTopk", 1))
            entries.append((normalized, {
                "profile": "random",
                "range_mode": range_mode,
                "origin": "ascend950_smoke",
                "token_dtype": dtypes[0],
                "probs_dtype": None if probs_shape is None else dtypes[2],
                "topk": topk,
                "hidden": int(shapes[0][-1]),
                "token_num": int(shapes[3][0]),
                "noncontiguous": False,
            }))
    return entries


def _flatten(items):
    for item in items:
        if isinstance(item, list):
            yield from _flatten(item)
        else:
            yield item


def canndev_entries(path):
    data = json.loads(path.read_text())
    entries = []
    for index, case in enumerate(data):
        values = {}
        for item in _flatten(case["inputs"]):
            values.setdefault(item["name"], []).append(item)
        tokens = values["permutedTokens"][0]
        sorted_indices = values["sorted_indices"][0]
        probs = values["probs"][0]
        topk = int(values["topKOptional"][0]["range_values"])
        range_values = [int(v["range_values"]) for v in values["rangeOptional"]]
        hidden = int(tokens["shape"][1])
        source_slots = int(sorted_indices["shape"][0])
        source_token_num = int(probs["shape"][0])
        token_num = min(source_token_num, max(1, 4_000_000 // max(1, hidden)))
        slots = token_num * topk
        token_rows = min(int(tokens["shape"][0]), slots, max(1, 4_000_000 // max(1, hidden)))
        start = min(max(0, range_values[0]), max(0, slots - token_rows))
        normalized_range = [start, start + token_rows]
        name = f"aclnn_moe_token_unpermute_with_ep_l1_canndev_seed_{index:03d}"
        row, meta = make_case(
            name,
            token_dtype="bfloat16",
            probs_dtype="bfloat16",
            topk=topk,
            hidden=hidden,
            token_num=token_num,
            range_mode="partial",
            token_shape=(token_rows, hidden),
            sorted_shape=(slots,),
            probs_shape=(token_num, topk),
            out_shape=(token_num, hidden),
            attrs={
                "numTopk": topk,
                "rangeOptional": normalized_range,
                "paddedMode": False,
                "restoreShapeOptional": None,
            },
        )
        meta.update(
            origin="canndev_200_seed",
            source_case_id=case.get("id", index),
            source_slots=source_slots,
            source_token_num=source_token_num,
            normalized_restore_shape=True,
        )
        entries.append((row, meta))
    return entries, len(data)


def mandatory_positive_entries():
    """Small, executable anchors that make every required axis deterministic."""
    entries = []
    probs_choices = (None, *FLOAT_DTYPES)
    for index, (token_dtype, probs_dtype) in enumerate(itertools.product(FLOAT_DTYPES, probs_choices)):
        entries.append(make_case(
            f"aclnn_moe_token_unpermute_with_ep_l1_dtype_combo_{index:02d}",
            token_dtype, probs_dtype, 2, 17, 4, "partial",
        ))
    for index, hidden in enumerate(HIDDENS):
        entries.append(make_case(
            f"aclnn_moe_token_unpermute_with_ep_l1_hidden_{hidden}",
            FLOAT_DTYPES[index % len(FLOAT_DTYPES)], FLOAT_DTYPES[(index + 1) % len(FLOAT_DTYPES)],
            2, hidden, 2, "partial",
        ))
    for index, token_num in enumerate(TOKEN_COUNTS):
        entries.append(make_case(
            f"aclnn_moe_token_unpermute_with_ep_l1_tokens_{token_num}",
            FLOAT_DTYPES[index % len(FLOAT_DTYPES)], "float32", 2, 7, token_num, "partial",
        ))
    for index, topk in enumerate(TOPKS):
        entries.append(make_case(
            f"aclnn_moe_token_unpermute_with_ep_l1_topk_{topk}",
            FLOAT_DTYPES[index % len(FLOAT_DTYPES)], "bfloat16", topk, 7, 1, "partial",
        ))
    for index, range_mode in enumerate(RANGE_MODES):
        entries.append(make_case(
            f"aclnn_moe_token_unpermute_with_ep_l1_range_{range_mode}",
            FLOAT_DTYPES[index % len(FLOAT_DTYPES)], "float32", 8, 31, 4, range_mode,
        ))
    for _, meta in entries:
        meta["origin"] = "mandatory_coverage_anchor"
    return entries


def semantic_candidates(prefix):
    entries = []
    counter = 0
    prob_choices = (None, *FLOAT_DTYPES)
    for token_dtype, probs_dtype, topk, hidden, token_num, range_mode in itertools.product(
        FLOAT_DTYPES, prob_choices, TOPKS, HIDDENS, TOKEN_COUNTS, RANGE_MODES
    ):
        slots, token_rows, _, _ = _range_layout(token_num, topk, range_mode, probs_dtype is not None)
        if token_rows * hidden > 4_000_000 or token_num * hidden > 4_000_000 or slots > 1_000_000:
            continue
        if range_mode == "nonzero_start" and slots <= 1:
            continue
        name = f"{prefix}_{counter:05d}"
        row, meta = make_case(name, token_dtype, probs_dtype, topk, hidden, token_num, range_mode)
        meta["origin"] = "st_factor_solution"
        entries.append((row, meta))
        counter += 1
    return entries


def build_l0(old_entries, pool):
    entries = list(old_entries)
    for label in (
        "tokens_poszero", "tokens_negzero", "tokens_posinf", "tokens_neginf", "tokens_nan", "tokens_large",
        "probs_poszero", "probs_negzero", "probs_posinf", "probs_neginf", "probs_nan", "probs_large",
    ):
        entries.append(make_case(f"aclnn_moe_token_unpermute_with_ep_l0_{label}", "float32", "float16", 2, 17, 4, "partial", label))
    for dtype in FLOAT_DTYPES:
        entries.append(make_case(f"aclnn_moe_token_unpermute_with_ep_l0_{dtype}_topk511", dtype, "float32", 511, 7, 1, "partial"))
        entries.append(make_case(f"aclnn_moe_token_unpermute_with_ep_l0_{dtype}_noncontiguous", dtype, "bfloat16", 2, 17, 4, "partial", noncontiguous=True))
        entries.append(make_case(f"aclnn_moe_token_unpermute_with_ep_l0_{dtype}_deterministic", dtype, "float16", 8, 15, 2, "full", "deterministic"))
        entries.append(make_case(f"aclnn_moe_token_unpermute_with_ep_l0_{dtype}_fallback_no_probs", dtype, None, 512, 33, 17, "fallback"))
    for mode in RANGE_MODES:
        entries.append(make_case(f"aclnn_moe_token_unpermute_with_ep_l0_range_{mode}", "float16", "float32", 8, 31, 4, mode))
    entries = dedupe(entries)
    pool_iter = iter(pool)
    while len(entries) < 48:
        entries = dedupe(entries + [next(pool_iter)])
    return entries[:48]


def _l2_case(name, mutation, expected_error, **overrides):
    params = dict(
        name=name,
        token_dtype="float16",
        probs_dtype="float16",
        topk=2,
        hidden=4,
        token_num=2,
        range_mode="partial",
        profile=overrides.pop("profile", "random"),
    )
    params.update(overrides)
    row, meta = make_case(**params)
    meta.update(origin="st_l2_derived", mutation=mutation, expected_error=expected_error)
    return row, meta


def build_l2():
    base_attrs = {"numTopk": 2, "rangeOptional": [1, 3], "paddedMode": False, "restoreShapeOptional": None}
    specs = [
        ("tokens_dtype_float64", "tensor_dtypes[0]", ["DTYPE", "not support"], {"dtypes": ("float64", "int32", "float16", "float16")}),
        ("indices_dtype_int64", "tensor_dtypes[1]", ["DTYPE", "int32"], {"dtypes": ("float16", "int64", "float16", "float16")}),
        ("probs_dtype_float64", "tensor_dtypes[2]", ["DTYPE", "not support"], {"dtypes": ("float16", "int32", "float64", "float16")}),
        ("output_dtype_mismatch", "tensor_dtypes[3]", ["DTYPE", "output"], {"dtypes": ("float16", "int32", "float16", "float32")}),
        ("tokens_format_nchw", "tensor_formats[0]", ["FORMAT", "ND"], {"formats": ("NCHW", "ND", "ND", "ND")}),
        ("indices_format_nchw", "tensor_formats[1]", ["FORMAT", "ND"], {"formats": ("ND", "NCHW", "ND", "ND")}),
        ("probs_format_nchw", "tensor_formats[2]", ["FORMAT", "ND"], {"formats": ("ND", "ND", "NCHW", "ND")}),
        ("output_format_nchw", "tensor_formats[3]", ["FORMAT", "ND"], {"formats": ("ND", "ND", "ND", "NCHW")}),
        ("tokens_rank1", "tensor_view_shapes[0]", ["shape", "2D"], {"token_shape": (4,)}),
        ("tokens_rank3", "tensor_view_shapes[0]", ["shape", "2D"], {"token_shape": (1, 2, 4)}),
        ("tokens_rank9", "tensor_view_shapes[0]", ["shape", "2D"], {"token_shape": (1, 1, 1, 1, 1, 1, 1, 1, 4)}),
        ("indices_rank2", "tensor_view_shapes[1]", ["shape", "1D"], {"sorted_shape": (2, 2)}),
        ("indices_rank9", "tensor_view_shapes[1]", ["shape", "1D"], {"sorted_shape": (1, 1, 1, 1, 1, 1, 1, 1, 4)}),
        ("probs_rank1", "tensor_view_shapes[2]", ["shape", "2D"], {"probs_shape": (4,)}),
        ("probs_rank3", "tensor_view_shapes[2]", ["shape", "2D"], {"probs_shape": (1, 2, 2)}),
        ("probs_rank9", "tensor_view_shapes[2]", ["shape", "2D"], {"probs_shape": (1, 1, 1, 1, 1, 1, 1, 2, 2)}),
        ("output_rank1", "tensor_view_shapes[3]", ["shape", "2D"], {"out_shape": (8,)}),
        ("sorted_length_mismatch", "tensor_view_shapes[1]", ["dim(0)", "probs"], {"sorted_shape": (5,)}),
        ("probs_topk_mismatch", "tensor_view_shapes[2]", ["topK", "match"], {"probs_shape": (2, 1)}),
        ("tokens_range_span_mismatch", "tensor_view_shapes[0]", ["shape", "range"], {"token_shape": (3, 4)}),
        ("topk_zero", "attributes.numTopk", ["TopK", "less than 1"], {"attrs": dict(base_attrs, numTopk=0)}),
        ("topk_negative", "attributes.numTopk", ["TopK", "less than 1"], {"attrs": dict(base_attrs, numTopk=-1)}),
        ("topk_513", "attributes.numTopk/tensor_view_shapes[2]", ["TopK", "larger than 512"], {"topk": 513, "probs_shape": (2, 513), "sorted_shape": (1026,), "token_shape": (512, 4), "attrs": {"numTopk": 513, "rangeOptional": [1, 513], "paddedMode": False, "restoreShapeOptional": None}}),
        ("range_length_1", "attributes.rangeOptional", ["range", "size"], {"attrs": dict(base_attrs, rangeOptional=[1])}),
        ("range_length_3", "attributes.rangeOptional", ["range", "size"], {"attrs": dict(base_attrs, rangeOptional=[0, 1, 2])}),
        ("range_start_after_end", "attributes.rangeOptional", ["range", "start"], {"attrs": dict(base_attrs, rangeOptional=[3, 1])}),
        ("range_negative", "attributes.rangeOptional", ["range", "negative"], {"attrs": dict(base_attrs, rangeOptional=[-1, 1])}),
        ("padded_mode_true", "attributes.paddedMode", ["padded", "false"], {"attrs": dict(base_attrs, paddedMode=True)}),
        ("restore_shape_nonempty", "attributes.restoreShapeOptional", ["restore", "nullptr"], {"attrs": dict(base_attrs, restoreShapeOptional=[2, 4])}),
        ("tokens_empty", "tensor_view_shapes[0]", ["empty", "shape"], {"token_shape": (0, 4)}),
        ("indices_empty", "tensor_view_shapes[1]", ["empty", "shape"], {"sorted_shape": (0,), "probs_dtype": None, "probs_shape": None}),
        ("probs_empty", "tensor_view_shapes[2]", ["empty", "shape"], {"probs_shape": (0, 2), "sorted_shape": (0,), "out_shape": (0, 4)}),
        ("output_empty", "tensor_view_shapes[3]", ["empty", "output"], {"out_shape": (0, 4)}),
        ("indices_negative", "sortedIndices.value", ["sortedIndices", "greater than or equal to 0"], {"profile": "l2_indices_negative"}),
        ("indices_upper_bound", "sortedIndices.value", ["sortedIndices", "2134372523"], {"profile": "l2_indices_upper_bound"}),
    ]
    return [
        _l2_case(f"aclnn_moe_token_unpermute_with_ep_l2_{name}", mutation, errors, **options)
        for name, mutation, errors, options in specs
    ]


def write_csv(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(row for row, _ in entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-l0", type=Path, required=True)
    parser.add_argument("--raw-l1", type=Path, required=True)
    parser.add_argument("--raw-l2", type=Path, required=True)
    parser.add_argument("--smoke-l0", type=Path, required=True)
    parser.add_argument("--canndev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_counts = {}
    for level, path in (("L0", args.raw_l0), ("L1", args.raw_l1), ("L2", args.raw_l2)):
        with path.open(newline="") as handle:
            raw_counts[level] = sum(1 for _ in csv.DictReader(handle))

    pool = semantic_candidates("aclnn_moe_token_unpermute_with_ep_l1_generated")
    l0 = build_l0(read_old_l0(args.smoke_l0), pool)
    assert 0 < len(l0) <= 50
    l0_signatures = {meta["execution_signature"] for _, meta in l0}

    seed_entries, seed_count = canndev_entries(args.canndev)
    mandatory_entries = mandatory_positive_entries()
    candidates = dedupe(mandatory_entries + seed_entries + pool, forbidden=l0_signatures)
    rng = random.Random(42)
    mandatory_part = [entry for entry in candidates if entry[1].get("origin") == "mandatory_coverage_anchor"]
    seed_part = [entry for entry in candidates if entry[1].get("origin") == "canndev_200_seed"]
    generated_part = [
        entry for entry in candidates
        if entry[1].get("origin") not in ("mandatory_coverage_anchor", "canndev_200_seed")
    ]
    rng.shuffle(generated_part)
    l1 = dedupe(mandatory_part + seed_part + generated_part, forbidden=l0_signatures)[:600]
    assert len(l1) == 600

    l2 = dedupe(build_l2())
    assert len(l2) == 35

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "L0": args.output_dir / "aclnn_moe_token_unpermute_with_ep_L0.csv",
        "L1": args.output_dir / "aclnn_moe_token_unpermute_with_ep_L1.csv",
        "L2": args.output_dir / "aclnn_moe_token_unpermute_with_ep_L2.csv",
    }
    write_csv(paths["L0"], l0)
    write_csv(paths["L1"], l1)
    write_csv(paths["L2"], l2)

    all_positive = l0 + l1
    positive_meta = [meta for _, meta in all_positive]
    coverage = {
        "schema_version": 1,
        "generator": "ascendc-st-design -> normalized TTK",
        "raw_generated_counts": raw_counts,
        "final_counts": {"L0": len(l0), "L1": len(l1), "L2": len(l2)},
        "canndev_seed": {"input_count": seed_count, "normalized_and_used": sum(m.get("origin") == "canndev_200_seed" for _, m in l1)},
        "positive": {
            "token_dtypes": sorted({m["token_dtype"] for m in positive_meta}),
            "probs_dtypes": sorted({str(m["probs_dtype"]) for m in positive_meta}),
            "topk": sorted({m["topk"] for m in positive_meta if isinstance(m.get("topk"), int)}),
            "hidden": sorted({m["hidden"] for m in positive_meta if isinstance(m.get("hidden"), int)}),
            "token_num": sorted({m["token_num"] for m in positive_meta if isinstance(m.get("token_num"), int)}),
            "dtype_combinations": sorted({
                f"{m['token_dtype']}+{m['probs_dtype']}" for m in positive_meta
            }),
            "range_modes": sorted({m["range_mode"] for m in positive_meta}),
            "special_profiles": sorted({m["profile"] for m in positive_meta if m["profile"] != "random"}),
            "noncontiguous_cases": sum(bool(m.get("noncontiguous")) for m in positive_meta),
        },
        "negative": {
            "count": len(l2),
            "mutations": [meta["mutation"] for _, meta in l2],
            "rank9_cases": sum("rank9" in row["testcase_name"] for row, _ in l2),
            "empty_cases": sum(row["testcase_name"].endswith("_empty") for row, _ in l2),
        },
    }
    assert set(coverage["positive"]["token_dtypes"]) == set(FLOAT_DTYPES)
    assert set(coverage["positive"]["probs_dtypes"]) == {"None", *FLOAT_DTYPES}
    assert len(coverage["positive"]["dtype_combinations"]) == len(FLOAT_DTYPES) * (len(FLOAT_DTYPES) + 1)
    assert set(TOPKS).issubset(coverage["positive"]["topk"])
    assert set(HIDDENS).issubset(coverage["positive"]["hidden"])
    assert set(TOKEN_COUNTS).issubset(coverage["positive"]["token_num"])
    assert set(RANGE_MODES).issubset(coverage["positive"]["range_modes"])
    assert coverage["positive"]["noncontiguous_cases"] >= 3
    assert coverage["negative"]["rank9_cases"] >= 3
    assert coverage["negative"]["empty_cases"] == 4

    (args.output_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "seed": 42,
        "api_name": API,
        "layers": {
            "L0": [{"testcase_name": row["testcase_name"], **meta} for row, meta in l0],
            "L1": [{"testcase_name": row["testcase_name"], **meta} for row, meta in l1],
            "L2": [{"testcase_name": row["testcase_name"], **meta} for row, meta in l2],
        },
    }
    (args.output_dir / "case_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(coverage["final_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
