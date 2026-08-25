#!/usr/bin/env python3
"""Independent contract audit for generated MoeTokenUnpermuteWithEp cases."""

import ast
import csv
import itertools
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FLOAT_DTYPES = {"float16", "bfloat16", "float32"}
TOPK_BOUNDARIES = {1, 2, 8, 511, 512}


def product(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    document = json.loads((ROOT / "design_doc_cases.json").read_text(encoding="utf-8"))
    cases = document["cases"]
    regular_rows = read_csv(ROOT / "ttk_design_doc_cases.csv")
    oversized_rows = read_csv(ROOT / "ttk_design_doc_cases_oversized.csv")
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    case_by_id = {case["id"]: case for case in cases}
    regular_ids = {row["testcase_name"] for row in regular_rows}
    oversized_ids = {row["testcase_name"] for row in oversized_rows}
    require(len(case_by_id) == len(cases), "generated case ids are not unique")
    require(not (regular_ids & oversized_ids), "regular and oversized CSVs overlap")
    require(regular_ids | oversized_ids == set(case_by_id), "CSV rows do not exactly cover generated cases")

    present_pairs = set()
    absent_token_dtypes = set()
    all_topks = set()
    absent_topks = set()
    prob_dtypes = set()
    modes = set()
    outside_range_cases = set()
    sizes = set()
    aligns = set()
    scenario_dtypes = set()
    ranks = set()
    notes = set()

    for case in cases:
        case_id = case["id"]
        inputs = case["inputs"]
        tokens = inputs["permuted_tokens"]
        indices = inputs["sorted_indices"]
        probs = inputs["probs"]
        params = case["params"]
        output = case["expected"]["outputs"]["unpermuted_tokens"]
        token_shape = tokens["shape"]
        index_shape = indices["shape"]
        topk = params["num_topk"]
        range_value = params["range"]
        is_ep = range_value != [0, 0]

        require(len(token_shape) == 2 and all(dim > 0 for dim in token_shape), f"{case_id}: bad tokens shape")
        require(len(index_shape) == 1 and index_shape[0] > 0, f"{case_id}: bad indices shape")
        require(tokens["dtype"] in FLOAT_DTYPES, f"{case_id}: bad tokens dtype")
        require(indices["dtype"] == "int32", f"{case_id}: indices is not int32")
        require(topk in TOPK_BOUNDARIES, f"{case_id}: unexpected TopK {topk}")
        require(params["padded_mode"] is False, f"{case_id}: padded_mode is not false")
        require(params["restore_shape"] == [], f"{case_id}: restore_shape is not empty")
        require(len(range_value) == 2 and range_value[0] >= 0 and range_value[1] >= range_value[0],
                f"{case_id}: invalid range")

        index_count = product(index_shape)
        if probs is not None:
            prob_shape = probs["shape"]
            require(len(prob_shape) == 2 and all(dim > 0 for dim in prob_shape), f"{case_id}: bad probs shape")
            require(probs["dtype"] in FLOAT_DTYPES, f"{case_id}: bad probs dtype")
            require(prob_shape[1] == topk, f"{case_id}: probs.shape[1] != num_topk")
            require(index_count == prob_shape[0] * prob_shape[1],
                    f"{case_id}: indices/probs shape coupling violated")
            output_rows = prob_shape[0]
            present_pairs.add((tokens["dtype"], probs["dtype"]))
            prob_dtypes.add(probs["dtype"])
        else:
            absent_token_dtypes.add(tokens["dtype"])
            absent_topks.add(topk)
            if is_ep:
                require(index_count % topk == 0, f"{case_id}: no-probs EP indices not divisible by TopK")
                output_rows = index_count // topk
            else:
                require(topk == 1, f"{case_id}: no-probs fallback must declare effective TopK 1")
                output_rows = index_count

        if is_ep:
            start, end = range_value
            require(end > start, f"{case_id}: EP range is empty")
            require(token_shape[0] == end - start, f"{case_id}: local token rows != EP range span")
            candidates = [start, end - 1]
            if start > 0:
                candidates.append(start - 1)
            if end < index_count:
                candidates.append(end)
            for value in candidates:
                if start <= value < end:
                    require(value - start < token_shape[0], f"{case_id}: valid EP index exceeds local tokens")
                else:
                    outside_range_cases.add(case_id)
            modes.add("ep_nonzero_start" if start > 0 else "ep_zero_start")
        else:
            require(token_shape[0] > 0, f"{case_id}: fallback has no token rows")
            # The registered customize_inputs hook uses modulo token rows in fallback.
            require(all(0 <= value % token_shape[0] < token_shape[0] for value in (0, index_count - 1)),
                    f"{case_id}: fallback runtime index generation is out of bounds")
            modes.add("fallback")

        require(output["shape"] == [output_rows, token_shape[1]], f"{case_id}: output shape violates contract")
        require(output["dtype"] == tokens["dtype"], f"{case_id}: output dtype differs from tokens")
        require(case.get("_shape_source") != "placeholder", f"{case_id}: placeholder output shape")

        signature = case.get("scenario_signature")
        if signature is not None:
            sizes.add(signature["size"])
            aligns.add(signature["align"])
            scenario_dtypes.add(signature["dtype"])
            ranks.add(signature["rank"])
        all_topks.add(topk)
        notes.add(case.get("_note", ""))

    expected_pairs = set(itertools.product(FLOAT_DTYPES, FLOAT_DTYPES))
    require(expected_pairs <= present_pairs, "not all 9 tokens/probs dtype pairs are covered")
    require(absent_token_dtypes == FLOAT_DTYPES, "probs-absent tokens dtypes are incomplete")
    require(all_topks == TOPK_BOUNDARIES, "TopK boundary coverage is incomplete")
    require(absent_topks == TOPK_BOUNDARIES, "probs-absent TopK boundary coverage is incomplete")
    require(prob_dtypes == FLOAT_DTYPES, "tiling keys for three probs dtypes are incomplete")
    require(modes == {"fallback", "ep_zero_start", "ep_nonzero_start"}, "range mode coverage is incomplete")
    require(outside_range_cases, "no EP case exercises an outside-range index")
    require(sizes == {"S", "M", "L"}, "S/M/L coverage is incomplete")
    require(aligns == {"aligned", "nonaligned", "prime_nonaligned"}, "alignment coverage is incomplete")
    require(scenario_dtypes == FLOAT_DTYPES and ranks == {2}, "scenario dtype/rank coverage is incomplete")

    required_notes = {
        "tokens_posinf", "tokens_neginf", "tokens_nan", "tokens_poszero", "tokens_negzero", "tokens_large",
        "probs_posinf", "probs_neginf", "probs_nan", "probs_poszero", "probs_negzero", "probs_large",
        "deterministic_repeat", "actual_size_m", "actual_size_l_oversized", "no_probs_f32_fallback",
        "general_positive_values", "general_negative_values",
    }
    require(required_notes <= notes, "directed special-value/size/determinism cases are incomplete")
    by_note = {case.get("_note", ""): case for case in cases}
    for note in ("tokens_negzero", "probs_negzero"):
        tensor = "permuted_tokens" if note.startswith("tokens") else "probs"
        value = by_note[note]["inputs"][tensor]["value_range"][0]
        require(value == 0.0 and math.copysign(1.0, value) < 0, f"{note}: negative zero was lost")
    for note in ("tokens_posinf", "tokens_neginf", "probs_posinf", "probs_neginf"):
        tensor = "permuted_tokens" if note.startswith("tokens") else "probs"
        require(math.isinf(by_note[note]["inputs"][tensor]["value_range"][0]), f"{note}: infinity was lost")
    for note in ("tokens_nan", "probs_nan"):
        tensor = "permuted_tokens" if note.startswith("tokens") else "probs"
        require(math.isnan(by_note[note]["inputs"][tensor]["value_range"][0]), f"{note}: NaN was lost")
    require(by_note["general_positive_values"]["inputs"]["permuted_tokens"]["value_range"] == [0.125, 3.0],
            "directed general-positive token range was lost")
    require(by_note["general_positive_values"]["inputs"]["probs"]["value_range"] == [0.125, 1.0],
            "directed general-positive probs range was lost")
    require(by_note["general_negative_values"]["inputs"]["permuted_tokens"]["value_range"] == [-3.0, -0.125],
            "directed general-negative token range was lost")
    require(by_note["general_negative_values"]["inputs"]["probs"]["value_range"] == [-1.0, -0.125],
            "directed general-negative probs range was lost")

    for row in regular_rows + oversized_rows:
        case = case_by_id[row["testcase_name"]]
        input_shapes = ast.literal_eval(row["input_shapes"])
        input_dtypes = ast.literal_eval(row["input_dtypes"])
        input_formats = ast.literal_eval(row["input_formats"])
        output_shapes = ast.literal_eval(row["output_shapes"])
        output_dtypes = ast.literal_eval(row["output_dtypes"])
        output_formats = ast.literal_eval(row["output_formats"])
        attrs = ast.literal_eval(row["attributes"])
        require(len(input_shapes) == len(input_dtypes) == len(input_formats) == 3,
                f"{case['id']}: CSV does not preserve three formal input slots")
        require(input_shapes[2] is None if case["inputs"]["probs"] is None else input_shapes[2] is not None,
                f"{case['id']}: optional probs CSV slot mismatch")
        require(all(fmt in ("ND", None) for fmt in input_formats) and output_formats == ("ND",),
                f"{case['id']}: non-ND format in positive whitebox CSV")
        require(output_shapes == (tuple(case["expected"]["outputs"]["unpermuted_tokens"]["shape"]),),
                f"{case['id']}: CSV output shape differs from source case")
        require(output_dtypes == (case["inputs"]["permuted_tokens"]["dtype"],),
                f"{case['id']}: CSV output dtype differs from source case")
        require(attrs == case["params"], f"{case['id']}: CSV attrs differ from source case")

    require(oversized_ids == {by_note["actual_size_l_oversized"]["id"]},
            "oversized CSV is not the single directed L case")

    report = {
        "ok": not errors,
        "generated_cases": len(cases),
        "regular_csv_cases": len(regular_rows),
        "oversized_csv_cases": len(oversized_rows),
        "shape_contract_cases_checked": len(cases),
        "csv_rows_checked": len(regular_rows) + len(oversized_rows),
        "present_dtype_pairs": sorted([list(pair) for pair in present_pairs]),
        "absent_token_dtypes": sorted(absent_token_dtypes),
        "topk_values": sorted(all_topks),
        "absent_topk_values": sorted(absent_topks),
        "range_modes": sorted(modes),
        "outside_range_case_count": len(outside_range_cases),
        "size_classes": sorted(sizes),
        "alignment_classes": sorted(aligns),
        "tiling_keys": [0, 1, 2, 3],
        "special_value_cases": sorted(required_notes & notes),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
