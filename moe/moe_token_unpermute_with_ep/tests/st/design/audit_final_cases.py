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

"""Independent count, uniqueness, contract and shape audit for the final ACLNN suites."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path


FLOAT_DTYPES = {"float16", "bfloat16", "float32"}
EXPECTED_COUNTS = {"L0": 48, "L1": 600, "L2": 35}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def literal(row, field):
    return ast.literal_eval(row[field])


def fingerprint(row, profile="random"):
    ignored = {"testcase_name", "soc_series"}
    value = {key: row[key] for key in sorted(row) if key not in ignored}
    value["custom_input_profile"] = profile
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit_positive(rows):
    dtype_combinations = set()
    topks = set()
    rank_counts = {"tokens": {}, "indices": {}, "probs": {}, "output": {}}
    noncontiguous = 0
    for row in rows:
        assert row["api_name"] == "aclnnMoeTokenUnpermuteWithEp"
        shapes = literal(row, "tensor_view_shapes")
        dtypes = literal(row, "tensor_dtypes")
        formats = literal(row, "tensor_formats")
        attrs = literal(row, "attributes")
        assert len(shapes) == 4 and len(dtypes) == 4
        assert len(shapes[0]) == 2 and len(shapes[1]) == 1 and len(shapes[3]) == 2
        assert shapes[2] is None or len(shapes[2]) == 2
        assert all(int(dim) > 0 for shape in shapes if shape is not None for dim in shape)
        assert dtypes[0] in FLOAT_DTYPES and dtypes[1] == "int32" and dtypes[3] == dtypes[0]
        assert shapes[0][1] == shapes[3][1]
        assert attrs["paddedMode"] is False and attrs["restoreShapeOptional"] is None
        assert set(formats) == {"ND"}
        probs_dtype = None if shapes[2] is None else dtypes[2]
        assert probs_dtype is None or probs_dtype in FLOAT_DTYPES
        dtype_combinations.add((dtypes[0], probs_dtype))

        range_value = attrs.get("rangeOptional")
        is_ep = range_value is not None and list(range_value) != [0, 0]
        if shapes[2] is not None:
            token_num, topk = shapes[2]
            assert shapes[1][0] == token_num * topk
            assert shapes[3][0] == token_num
            topks.add(int(topk))
        elif is_ep:
            topk = int(attrs["numTopk"])
            assert topk > 0 and shapes[1][0] % topk == 0
            assert shapes[3][0] == shapes[1][0] // topk
            topks.add(topk)
        else:
            assert shapes[3][0] == shapes[1][0]
            topks.add(1)

        if is_ep:
            assert len(range_value) == 2
            start, end = map(int, range_value)
            assert 0 <= start < end <= shapes[1][0]
            assert shapes[0][0] == end - start
        else:
            assert shapes[0][0] == shapes[1][0]

        if row.get("tensor_view_strides"):
            noncontiguous += 1
            assert row.get("tensor_storage_shapes") and row.get("tensor_view_offsets")
        for label, shape in zip(rank_counts, shapes):
            rank = -1 if shape is None else len(shape)
            rank_counts[label][str(rank)] = rank_counts[label].get(str(rank), 0) + 1

    assert dtype_combinations == {(token, probs) for token in FLOAT_DTYPES for probs in (None, *FLOAT_DTYPES)}
    assert {1, 2, 8, 511, 512}.issubset(topks)
    assert noncontiguous >= 3
    return {
        "dtype_combinations": sorted(f"{a}+{b}" for a, b in dtype_combinations),
        "topk": sorted(topks),
        "noncontiguous": noncontiguous,
        "rank_counts": rank_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--ttk-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {
        level: read_csv(args.case_dir / f"aclnn_moe_token_unpermute_with_ep_{level}.csv")
        for level in EXPECTED_COUNTS
    }
    manifest = json.loads((args.case_dir / "case_manifest.json").read_text())
    coverage = json.loads((args.case_dir / "coverage_report.json").read_text())
    for level, expected in EXPECTED_COUNTS.items():
        names = [row["testcase_name"] for row in rows[level]]
        manifest_names = [case["testcase_name"] for case in manifest["layers"][level]]
        assert len(rows[level]) == expected == len(manifest_names)
        assert names == manifest_names and len(names) == len(set(names))
        assert len({
            fingerprint(row, case.get("profile", "random"))
            for row, case in zip(rows[level], manifest["layers"][level])
        }) == expected
        assert coverage["final_counts"][level] == expected

    assert not ({row["testcase_name"] for row in rows["L0"]} & {row["testcase_name"] for row in rows["L1"]})
    positive = read_csv(args.ttk_dir / "aclnn_moe_token_unpermute_with_ep_positive.csv")
    negative = read_csv(args.ttk_dir / "aclnn_moe_token_unpermute_with_ep_negative.csv")
    source_positive_names = [row["testcase_name"] for row in rows["L0"] + rows["L1"]]
    assert [row["testcase_name"] for row in positive] == source_positive_names
    assert [row["testcase_name"] for row in negative] == [row["testcase_name"] for row in rows["L2"]]
    assert len(positive) == 648 and len(negative) == 35
    assert all(row.get("soc_series") == "Ascend950" for row in positive + negative)

    l2_manifest = manifest["layers"]["L2"]
    assert len({case["testcase_name"] for case in l2_manifest}) == 35
    assert all(case.get("expected_error") and case.get("mutation") for case in l2_manifest)
    positive_audit = audit_positive(rows["L0"] + rows["L1"])
    report = {
        "status": "PASS",
        "generated_counts": EXPECTED_COUNTS,
        "manifest_counts": {level: len(manifest["layers"][level]) for level in EXPECTED_COUNTS},
        "merged_ttk_counts": {"positive": len(positive), "negative": len(negative)},
        "unique_execution_rows": {
            level: len({
                fingerprint(row, case.get("profile", "random"))
                for row, case in zip(values, manifest["layers"][level])
            })
            for level, values in rows.items()
        },
        "contract_and_shape_audit": positive_audit,
        "l2_mutation_count": len(l2_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
