# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository.
"""Generate matching pytest/TTK CSVs and a reviewable STC worksheet."""
import argparse
import csv
import io
import json
from pathlib import Path

from cases import TestCases, select_cases, tensor_shapes, ttk_attributes

HERE = Path(__file__).resolve().parent
TTK_DIR = HERE.parent / "ttk" / "bsa_fullquant_mxfp8_test"
TTK_FIELDS = ("testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes", "tensor_formats",
              "attributes", "output_tensor_indexes", "golden_api", "input_data_ranges",
              "precision_tolerances", "absolute_precision")


def ttk_row(case):
    ptype = "float32" if case["p_scale_type"] == "float32" or case["p_scale_value"] is None else "float8_e8m0"
    shapes = tensor_shapes(case)
    dtypes = ("float8_e4m3fn",) * 3 + ("float8_e8m0",) * 3 + ("int32",) * 2 + (ptype,) + ("int32",) * 5 + ("uint8", "int32")
    ranges = tuple(tuple(case[k]) for k in ("data_range_q", "data_range_k", "data_range_v")) + (
        (), (), (), (-1, shapes[6][-1] - 1), (0, shapes[6][-1]), (),
        (0, case["B"] * case["S1_max"]), (), (), (0, max(case["seqused_kv"])),
        (0, shapes[1][0] - 1), (0, 1), (),
    )
    return dict(zip(TTK_FIELDS, (
        case["name"], "bsa_ttk_ops.quant_block_sparse_attn", repr(shapes), repr(dtypes), "",
        repr(ttk_attributes(case)), "", "", repr(ranges),
        repr(((0.0078125, 0.0001, 0.005, 0.005, 10), (0.005, 0.000025, 0.005, 0.005, 10))),
        repr((0.0001, 0.0001)),
    )))


def _csv(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def artifacts(include_design=False):
    result = {}
    fields = list(TestCases[0])
    for tier, suffix in (("all", ""), ("P0", "_p0")):
        cases = select_cases(tier)
        normalized = [{k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, bool)) else
                       ("" if v is None else v) for k, v in c.items()} for c in cases]
        result[HERE / f"qbsa_mxfp8_layout_stc{suffix}.csv"] = _csv(fields, normalized)
        result[TTK_DIR / f"qbsa_mxfp8_layout_stc{suffix}.csv"] = _csv(TTK_FIELDS, [ttk_row(c) for c in cases])
    if not include_design:
        return result
    rows = []
    for c in TestCases:
        b, s, n = c["B"], c["S1_max"], c["N1"]
        lse = (b * s, n) if c["layout_q"] == "TND" else (b, n, s)
        if not c["return_softmax_lse"]:
            lse = "TND legacy placeholder" if c["layout_q"] == "TND" else (0,)
        rows.append(dict(case_id=c["name"], priority=c["test_tier"], scenario=c["layout_scenario"],
                         source=c["design_source"], layout=c["layout_q"], q_shape=str(tensor_shapes(c)[0]),
                         qscale_shape=str(tensor_shapes(c)[3]), kv_lengths=str(c["seqused_kv"]),
                         mask=c["mask_mode"], sparse=c["sparse_mode"], lse_shape=str(lse), seed=c["seed"],
                         expected="CPU MXFP8 golden Pass; same-input TND/BSND/BNSD exact; shape/dtype correct"))
    result[HERE / "stc_matrix.csv"] = _csv(list(rows[0]), rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if checked-in artifacts have drifted")
    parser.add_argument("--include-design", action="store_true", help="Also generate/check the optional local STC worksheet")
    args = parser.parse_args()
    for path, content in artifacts(include_design=args.include_design).items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Outdated or missing: {path}; run export_cases.py")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="")
        print(f"{'checked' if args.check else 'wrote'} {path.name}")


if __name__ == "__main__":
    main()
