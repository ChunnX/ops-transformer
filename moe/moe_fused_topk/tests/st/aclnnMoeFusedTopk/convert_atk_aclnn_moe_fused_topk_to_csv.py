#!/usr/bin/env python3
"""Convert the legacy MoeFusedTopk ATK matrix into a TTK ACLNN CSV.

The source matrix describes ranges for both tensors and primitive attributes.
TTK natively supports tensor ranges, while primitive ACLNN attributes must be
concrete values in every CSV row.  This converter materializes the latter with
a stable SHA-256 based selection using the original ATK id and attribute name.
The original JSON remains the source of truth and special NaN/Inf tensor cases
are restored by ``moe_fused_topk_ttk_spec.py`` at test-input generation time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_JSON = HERE / "atk_aclnnMoeFusedTopk.json"
OUTPUT_CSV = HERE / "aclnnMoeFusedTopk.csv"

CSV_FIELDS = (
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_formats",
    "tensor_dtypes",
    "attributes",
    "output_tensor_indexes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "remark",
)

DTYPE_MAP = {
    "fp16": "float16",
    "fp32": "float32",
    "bf16": "bfloat16",
    "int32": "int32",
}

ATTRIBUTE_NAMES = {
    "group_num": "groupNum",
    "group_topk": "groupTopk",
    "top_n": "topN",
    "top_k": "topK",
    "activate_type": "activateType",
    "is_norm": "isNorm",
    "enable_expert_mapping": "enableExpertMapping",
}


def stable_number(case_id: int, attribute_name: str) -> int:
    """Return a cross-version stable pseudo-random integer for one field."""
    digest = hashlib.sha256(f"MoeFusedTopk:{case_id}:{attribute_name}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def materialize_attribute(case_id: int, name: str, value: Any) -> Any:
    """Choose one deterministic value from a legacy ATK attribute range."""
    if not isinstance(value, list):
        return value
    if len(value) != 2 or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Unsupported attribute range for case {case_id}, {name}: {value!r}")

    low, high = value
    if isinstance(low, int) and isinstance(high, int):
        return low + stable_number(case_id, name) % (high - low + 1)

    fraction = stable_number(case_id, name) / ((1 << 64) - 1)
    return round(low + (high - low) * fraction, 8)


def tensor_range(case_id: int, name: str, value: Any) -> tuple[float | int, float | int]:
    """Return the range syntax accepted by TTK for an ATK tensor field."""
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        return tuple(value)
    if isinstance(value, list) and len(value) == 1 and value[0] in {"nan", "inf", "-inf"}:
        # TTK's random range cannot encode these values.  The paired input
        # plugin loads the exact special value from the retained source JSON.
        return (-1.0, 1.0)
    raise ValueError(f"Unsupported tensor range for case {case_id}, {name}: {value!r}")


def input_by_name(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {item["name"]: item for item in case["inputs"]}
    required = {
        "x", "add_num", "mapping_num", "mapping_table", "group_num", "group_topk",
        "top_n", "top_k", "activate_type", "is_norm", "scale", "enable_expert_mapping",
    }
    if set(result) != required:
        raise ValueError(f"Unexpected field set in ATK case {case['id']}: {sorted(result)}")
    return result


def build_row(case: dict[str, Any]) -> dict[str, str]:
    case_id = case["id"]
    values = input_by_name(case)
    x_shape = tuple(values["x"]["shape"])
    token_count, expert_count = x_shape
    top_k = materialize_attribute(case_id, "top_k", values["top_k"]["range_values"])
    attributes = {
        ATTRIBUTE_NAMES.get(name, name): materialize_attribute(case_id, name, values[name]["range_values"])
        for name in (
            "group_num", "group_topk", "top_n", "top_k", "activate_type", "is_norm", "scale",
            "enable_expert_mapping",
        )
    }
    if attributes["topK"] != top_k:
        raise AssertionError("topK materialization must be stable")

    shapes = (
        x_shape,
        tuple(values["add_num"]["shape"]),
        tuple(values["mapping_num"]["shape"]),
        tuple(values["mapping_table"]["shape"]),
        (token_count, top_k),
        (token_count, top_k),
    )
    input_ranges = (
        tensor_range(case_id, "x", values["x"]["range_values"]),
        tensor_range(case_id, "add_num", values["add_num"]["range_values"]),
        tensor_range(case_id, "mapping_num", values["mapping_num"]["range_values"]),
        tensor_range(case_id, "mapping_table", values["mapping_table"]["range_values"]),
        (0, 0),
        (0, 0),
    )
    dtype = DTYPE_MAP[values["x"]["dtype"]]
    special_inputs = [
        f"{name}:{values[name]['range_values'][0]}"
        for name in ("x", "add_num")
        if isinstance(values[name]["range_values"], list)
        and len(values[name]["range_values"]) == 1
        and isinstance(values[name]["range_values"][0], str)
    ]
    special_remark = "; special_inputs=" + ",".join(special_inputs) if special_inputs else ""

    return {
        "testcase_name": f"moe_fused_topk_atk_{case_id:03d}",
        "api_name": "aclnnMoeFusedTopk",
        "tensor_view_shapes": repr(shapes),
        "tensor_formats": repr(("ND",) * len(shapes)),
        "tensor_dtypes": repr((dtype, dtype, "int32", "int32", "float32", "int32")),
        "attributes": repr(attributes),
        "output_tensor_indexes": repr((4, 5)),
        "input_data_ranges": repr(input_ranges),
        "precision_tolerances": repr(((1e-3, 1e-3), (0.0, 0.0))),
        "absolute_precision": repr((1e-3, 0.0)),
        "remark": (
            f"source_atk_id={case_id}; primitive_ranges=sha256_materialized"
            f"; source_json=atk_aclnnMoeFusedTopk.json{special_remark}"
        ),
    }


def render_csv(cases: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for case in cases:
        writer.writerow(build_row(case))
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_JSON)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--check", action="store_true", help="fail if the checked-in CSV is stale")
    args = parser.parse_args()

    cases = json.loads(args.source.read_text(encoding="utf-8"))
    if len(cases) != 201:
        raise ValueError(f"Expected 201 ATK cases, found {len(cases)}")
    rendered = render_csv(cases)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{args.output} is stale; rerun {Path(__file__).name}")
        print(f"Verified {len(cases)} CSV cases: {args.output}")
        return 0

    args.output.write_text(rendered, encoding="utf-8", newline="")
    print(f"Wrote {len(cases)} CSV cases: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
