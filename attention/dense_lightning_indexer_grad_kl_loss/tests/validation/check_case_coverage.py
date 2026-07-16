"""Check value, pairwise, boundary, and TND zero-length coverage."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from case_utils import load_all_cases, load_case_file, validate_case


PAIRWISE_LEVELS = {
    "layout": ("BSND", "TND"),
    "q_dtype": ("float16", "bfloat16"),
    "weight_class": ("same", "float32"),
    "rope": (True, False),
    "batch": (1, 4, 16),
    "s1": (1024, 2048, 4096),
    "s2": (4096, 16384, 65536),
    "n1": (32, 64, 128),
    "n1_index": (8, 16, 32, 64),
}


def factor_value(case, factor):
    if factor == "weight_class":
        return "float32" if case["weight_dtype"] == "float32" else "same"
    return case[factor]


def check_pairwise(cases):
    missing_values = []
    for factor, values in PAIRWISE_LEVELS.items():
        seen = {factor_value(case, factor) for case in cases}
        missing_values.extend((factor, value) for value in values if value not in seen)

    missing_pairs = []
    factors = list(PAIRWISE_LEVELS)
    for index, left in enumerate(factors):
        for right in factors[index + 1 :]:
            seen = {(factor_value(case, left), factor_value(case, right)) for case in cases}
            for pair in itertools.product(PAIRWISE_LEVELS[left], PAIRWISE_LEVELS[right]):
                if pair not in seen:
                    missing_pairs.append((left, pair[0], right, pair[1]))
    return missing_values, missing_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    all_cases = load_all_cases(args.root)
    full_cases = load_case_file(args.root / "cases_full.yaml")
    ids = [str(case["id"]) for case in all_cases]
    if len(ids) != len(set(ids)):
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        raise SystemExit(f"duplicate case IDs: {duplicates}")
    for case in all_cases:
        validate_case(case)
    if len(full_cases) != 48:
        raise SystemExit(f"full suite must contain 48 cases, got {len(full_cases)}")

    pairwise = [case for case in full_cases if case["group"] == "pairwise"]
    if len(pairwise) != 32:
        raise SystemExit(f"pairwise group must contain 32 cases, got {len(pairwise)}")
    missing_values, missing_pairs = check_pairwise(pairwise)
    if missing_values or missing_pairs:
        print("missing values:", missing_values)
        print("missing pairs:", missing_pairs)
        raise SystemExit(1)

    required = {
        "tnd_special": 8,
        "boundary": 4,
        "risk": 4,
    }
    for group, count in required.items():
        actual = sum(case["group"] == group for case in full_cases)
        if actual != count:
            raise SystemExit(f"{group} must contain {count} cases, got {actual}")

    zero_positions = {"first": False, "middle": False, "last": False}
    repeated_cumulative = False
    for case in full_cases:
        if case["layout"] != "TND":
            continue
        if case.get("zero_position") in zero_positions:
            zero_positions[str(case["zero_position"])] = True
        q_lengths = case.get("q_lengths", [])
        if any(int(value) == 0 for value in q_lengths):
            repeated_cumulative = True
            if q_lengths and int(q_lengths[0]) == 0:
                zero_positions["first"] = True
            if q_lengths and int(q_lengths[-1]) == 0:
                zero_positions["last"] = True
            if any(int(value) == 0 for value in q_lengths[1:-1]):
                zero_positions["middle"] = True
    if not all(zero_positions.values()) or not repeated_cumulative:
        raise SystemExit(f"incomplete TND zero-length coverage: {zero_positions}")

    print(
        f"PASS: {len(all_cases)} total cases, 48 full cases, "
        f"{len(missing_pairs)} missing pairwise pairs"
    )
    print(f"TND zero positions covered: {zero_positions}")


if __name__ == "__main__":
    main()
