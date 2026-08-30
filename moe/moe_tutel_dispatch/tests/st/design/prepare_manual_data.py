#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

"""Prepare exact raw-bit inputs and goldens for directed Kernel/GEIR cases."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


P0_GEIR_CASES = {"MTD_REQ_031", "MTD_REQ_032", "MTD_REQ_033"}
UINT_DTYPES = {
    "float16": np.dtype(np.uint16),
    "float32": np.dtype(np.uint32),
    "bfloat16": np.dtype(np.uint16),
}
SIGN_BITS = {"float16": 0x8000, "float32": 0x80000000, "bfloat16": 0x8000}
ABS_MASKS = {"float16": 0x7FFF, "float32": 0x7FFFFFFF, "bfloat16": 0x7FFF}


def concrete_dims(row):
    if row["case_id"] == "MTD_REQ_033":
        return {"S": 4, "H": 8, "K": 3, "E": 2, "capacity": 5}
    return {name: int(row[name]) for name in ("S", "H", "K", "E", "capacity")}


def _dtype(dtype_name, resolver):
    return np.dtype(resolver((dtype_name,))[0])


def _bits_to_values(bits, dtype_name, resolver):
    return np.asarray(bits, dtype=UINT_DTYPES[dtype_name]).view(
        _dtype(dtype_name, resolver)
    )


def _numeric_bits(values, dtype_name, resolver):
    return np.asarray(values, dtype=_dtype(dtype_name, resolver)).view(
        UINT_DTYPES[dtype_name]
    )


def _random_bits(shape, dtype_name, case_id):
    width = UINT_DTYPES[dtype_name].itemsize * 8
    seed = int(case_id.rsplit("_", 1)[1])
    generator = np.random.default_rng(seed)
    limit = 1 << width
    return generator.integers(0, limit, size=shape, dtype=UINT_DTYPES[dtype_name])


def x_bits(row, dims, dtype_name, resolver):
    shape = (dims["S"], dims["H"])
    pattern = row["x_pattern"]
    if np.prod(shape, dtype=np.int64) == 0:
        return np.zeros(shape, dtype=UINT_DTYPES[dtype_name])
    if pattern == "sequential" or pattern == "N/A":
        values = np.arange(1, dims["S"] * dims["H"] + 1, dtype=np.float32).reshape(
            shape
        )
        return _numeric_bits(values, dtype_name, resolver).copy()
    if pattern == "nan_payload_bits":
        special = {
            "float16": [0x7E01, 0x7D55, 0xFE33, 0x3C00],
            "float32": [0x7FC00001, 0x7FA12345, 0xFFC00033, 0x3F800000],
            "bfloat16": [0x7FC1, 0x7FA5, 0xFFC3, 0x3F80],
        }[dtype_name]
        return np.resize(np.asarray(special, dtype=UINT_DTYPES[dtype_name]), shape)
    if pattern == "pos_neg_inf_and_signed_zero":
        special = [
            0x7F800000,
            0xFF800000,
            0x00000000,
            0x80000000,
            0x7FC01234,
            0xFFC05678,
        ]
        return np.resize(np.asarray(special, dtype=np.uint32), shape)
    if pattern == "nan_inf_subnormal_bits":
        special = [0x7FC1, 0xFFC2, 0x7F80, 0xFF80, 0x0001, 0x8001, 0x0000, 0x8000]
        return np.resize(np.asarray(special, dtype=np.uint16), shape)
    bits = _random_bits(shape, dtype_name, row["case_id"])
    if row["route_pattern"] == "duplicate_destination" and dims["S"] > 1:
        bits[1:] = bits[0]
    return bits


def gate_bits(row, dims, dtype_name, resolver):
    shape = (dims["K"], dims["S"])
    pattern = row["gate_pattern"]
    if np.prod(shape, dtype=np.int64) == 0:
        return np.zeros(shape, dtype=UINT_DTYPES[dtype_name])
    one = int(_numeric_bits([1.0], dtype_name, resolver)[0])
    if pattern in {"all_one", "N/A"}:
        return np.full(shape, one, dtype=UINT_DTYPES[dtype_name])
    if pattern == "all_positive_zero":
        return np.zeros(shape, dtype=UINT_DTYPES[dtype_name])
    if pattern == "all_negative_zero":
        return np.full(shape, SIGN_BITS[dtype_name], dtype=UINT_DTYPES[dtype_name])
    if pattern == "mixed_zero_nonzero":
        values = np.resize(
            np.asarray(
                [0, SIGN_BITS[dtype_name], one, one | SIGN_BITS[dtype_name]],
                dtype=UINT_DTYPES[dtype_name],
            ),
            shape,
        )
        return values
    if pattern == "nan_inf_subnormal":
        special = {
            "float16": [0x7E01, 0x7C00, 0x0001, 0xFE01, 0xFC00, 0x8001],
            "float32": [
                0x7FC00001,
                0x7F800000,
                0x00000001,
                0xFFC00001,
                0xFF800000,
                0x80000001,
            ],
            "bfloat16": [0x7FC1, 0x7F80, 0x0001, 0xFFC1, 0xFF80, 0x8001],
        }[dtype_name]
        return np.resize(np.asarray(special, dtype=UINT_DTYPES[dtype_name]), shape)
    raise RuntimeError(f"unsupported gate pattern: {pattern}")


def route_arrays(row, dims):
    count = dims["K"] * dims["S"]
    indices = np.full(count, -1, dtype=np.int32)
    locations = np.full(count, -1, dtype=np.int32)
    pattern = row["route_pattern"]
    if count == 0 or pattern in {"empty", "all_invalid"}:
        return indices.reshape(dims["K"], dims["S"]), locations.reshape(
            dims["K"], dims["S"]
        )
    if pattern == "idx_negative":
        locations.fill(0)
    elif pattern == "idx_equal_E":
        indices.fill(dims["E"])
        locations.fill(0)
    elif pattern == "loc_negative":
        indices.fill(0)
    elif pattern == "loc_equal_capacity":
        indices.fill(0)
        locations.fill(dims["capacity"])
    elif pattern == "duplicate_destination":
        indices.fill(0)
        locations.fill(0)
    elif pattern in {"unique_sparse", "unique_full", "unique_until_full", "N/A"}:
        destination_count = dims["E"] * dims["capacity"]
        for route_index in range(min(count, destination_count)):
            indices[route_index] = route_index // dims["capacity"]
            locations[route_index] = route_index % dims["capacity"]
    else:
        raise RuntimeError(f"unsupported route pattern: {pattern}")
    return indices.reshape(dims["K"], dims["S"]), locations.reshape(
        dims["K"], dims["S"]
    )


def golden_bits(x_raw, gates_raw, indices, locations, dims, dtype_name):
    output = np.zeros(
        (dims["E"], dims["capacity"], dims["H"]), dtype=UINT_DTYPES[dtype_name]
    )
    flat_gates = gates_raw.reshape(-1)
    flat_indices = indices.reshape(-1)
    flat_locations = locations.reshape(-1)
    for route_index in range(flat_gates.size):
        expert = int(flat_indices[route_index])
        location = int(flat_locations[route_index])
        if int(flat_gates[route_index]) & ABS_MASKS[dtype_name] == 0:
            continue
        if (
            expert < 0
            or expert >= dims["E"]
            or location < 0
            or location >= dims["capacity"]
        ):
            continue
        output[expert, location] = x_raw[route_index % dims["S"]]
    return output


def prepare_case(row, resolver):
    dims = concrete_dims(row)
    dtype_name = row["dtype"]
    x_raw = x_bits(row, dims, dtype_name, resolver)
    gates_raw = gate_bits(row, dims, dtype_name, resolver)
    indices, locations = route_arrays(row, dims)
    y_raw = golden_bits(x_raw, gates_raw, indices, locations, dims, dtype_name)
    inputs = [
        _bits_to_values(x_raw, dtype_name, resolver),
        _bits_to_values(gates_raw, dtype_name, resolver),
        indices,
        locations,
    ]
    golden = _bits_to_values(y_raw, dtype_name, resolver)
    case = SimpleNamespace(
        testcase_name=row["case_id"],
        flat_input_shapes=tuple(value.shape for value in inputs),
        flat_input_dtypes=(dtype_name, dtype_name, "int32", "int32"),
        flat_output_shapes=(golden.shape,),
        flat_output_dtypes=(dtype_name,),
    )
    return case, inputs, golden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttk-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()

    ttk_root = args.ttk_root.resolve()
    sys.path.insert(0, str(ttk_root))
    from ttk.core_modules.manual_data import ManualDataStore
    from ttk.utilities import resolve_custom_numpy_dtypes

    st_dir = Path(__file__).resolve().parent.parent
    source = (
        args.source or st_dir / "testcases/MoeTutelDispatch_required_test_cases.csv"
    )
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    runnable = [
        row
        for row in rows
        if row["layer"] in {"kernel", "GEIR"} or row["case_id"] in P0_GEIR_CASES
    ]
    store = ManualDataStore(args.output_dir)
    for row in runnable:
        case, inputs, golden = prepare_case(row, resolve_custom_numpy_dtypes)
        store.write_case(case, "kernel", inputs, [golden], file_format="bin")
    print(
        f"PASS: prepared {len(runnable)} directed cases at {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
