#!/usr/bin/env python3
"""Create the 200-case direct-Kernel TTK matrix from the ATK CSV migration.

The retained ACLNN CSV contains all 201 converted ATK cases.  Kernel TTK uses
the same four operator inputs, but its CSV has a separate schema and takes the
two outputs out of the input list.  This converter intentionally retains the
first 200 rows in their stable source order, leaving the final ATK row in the
201-case ACLNN regression suite.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
from pathlib import Path


CASE_COUNT = 200
SOURCE = Path(__file__).parents[1] / "aclnnMoeFusedTopk" / "aclnnMoeFusedTopk.csv"
DESTINATION = Path(__file__).with_name("moeFusedTopkKernel.csv")

HEADERS = (
    "testcase_name",
    "network_name",
    "op_name",
    "input_shapes",
    "input_dtypes",
    "input_formats",
    "output_shapes",
    "output_dtypes",
    "output_formats",
    "input_ori_shapes",
    "input_ori_formats",
    "output_ori_shapes",
    "output_ori_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "output_inplace_indexes",
    "output_shape_unknown_indexes",
    "is_enabled",
    "remark",
    "soc_series",
    "priority",
    "dump_file_prefix",
    "manual_input_binaries",
    "manual_golden_binaries",
)

ATTRIBUTES = {
    "groupNum": "group_num",
    "groupTopk": "group_topk",
    "topN": "top_n",
    "topK": "top_k",
    "activateType": "activate_type",
    "isNorm": "is_norm",
    "scale": "scale",
    "enableExpertMapping": "enable_expert_mapping",
}


def _as_tuple(value):
    """Turn nested list-like literal values into the TTK tuple notation."""
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_as_tuple(item) for item in value)
    return value


def _render() -> str:
    with SOURCE.open("r", encoding="utf-8", newline="") as source_file:
        source_rows = list(csv.DictReader(source_file))
    if len(source_rows) < CASE_COUNT:
        raise RuntimeError(f"expected at least {CASE_COUNT} source rows, got {len(source_rows)}")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HEADERS, lineterminator="\n")
    writer.writeheader()
    for position, source in enumerate(source_rows[:CASE_COUNT]):
        shapes = _as_tuple(ast.literal_eval(source["tensor_view_shapes"]))
        dtypes = _as_tuple(ast.literal_eval(source["tensor_dtypes"]))
        ranges = _as_tuple(ast.literal_eval(source["input_data_ranges"]))
        precision = _as_tuple(ast.literal_eval(source["precision_tolerances"]))
        attrs = ast.literal_eval(source["attributes"])
        kernel_attrs = {ATTRIBUTES[key]: attrs[key] for key in ATTRIBUTES}
        source_name = source["testcase_name"]

        if len(shapes) != 6 or len(dtypes) != 6 or len(ranges) != 6:
            raise RuntimeError(f"{source_name}: expected 4 inputs and 2 outputs")

        input_shapes = tuple(shapes[:4])
        input_dtypes = tuple(dtypes[:4])
        output_shapes = tuple(shapes[4:])
        output_dtypes = tuple(dtypes[4:])
        if output_dtypes != ("float32", "int32"):
            raise RuntimeError(f"{source_name}: unexpected output dtypes {output_dtypes}")

        source_id = source_name.rsplit("_", maxsplit=1)[-1]
        writer.writerow(
            {
                "testcase_name": f"moe_fused_topk_kernel_atk_{source_id}",
                "network_name": "atk_moe_fused_topk",
                "op_name": "moe_fused_topk",
                "input_shapes": repr(input_shapes),
                "input_dtypes": repr(input_dtypes),
                "input_formats": repr(("ND",) * 4),
                "output_shapes": repr(output_shapes),
                "output_dtypes": repr(output_dtypes),
                "output_formats": repr(("ND",) * 2),
                "input_ori_shapes": repr(input_shapes),
                "input_ori_formats": repr(("ND",) * 4),
                "output_ori_shapes": repr(output_shapes),
                "output_ori_formats": repr(("ND",) * 2),
                "attributes": repr(kernel_attrs),
                "input_data_ranges": repr(tuple(ranges[:4])),
                "precision_tolerances": repr(precision),
                "absolute_precision": source["absolute_precision"],
                "output_inplace_indexes": "()",
                "output_shape_unknown_indexes": "()",
                "is_enabled": "True",
                "remark": (
                    f"direct_kernel; source_atk_id={int(source_id)}; "
                    f"kernel_matrix_position={position}"
                ),
                "soc_series": "",
                "priority": "0",
                "dump_file_prefix": "",
                "manual_input_binaries": "()",
                "manual_golden_binaries": "()",
            }
        )
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the checked-in CSV is current")
    args = parser.parse_args()
    rendered = _render()
    if args.check:
        if not DESTINATION.is_file() or DESTINATION.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{DESTINATION} is stale; run {Path(__file__).name}")
        print(f"PASS: {DESTINATION.name} has {CASE_COUNT} direct-Kernel cases")
        return
    DESTINATION.write_text(rendered, encoding="utf-8")
    print(f"Wrote {DESTINATION} with {CASE_COUNT} direct-Kernel cases")


if __name__ == "__main__":
    main()
