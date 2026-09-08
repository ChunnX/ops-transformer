#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Generic-case precision test for cla_gate_mx_quant_with_dual_axis.

The case list is loaded from the TTK kernel CSV so that this script covers the
same shapes/dtypes/dst_type/scale_alg/round_mode as the TTK cases.

For each case:
  1. call the transformer fused implementation:
       cann_ops_transformer.cla_gate_mx_quant_with_dual_axis(...)
  2. call the unfused NPU small-op reference:
       sigmoid gate -> broadcast mul/add -> reshape [T, -1]
       -> torch_npu.npu_dynamic_mx_quant_with_dual_axis(...)

The two outputs are compared byte-by-byte.

Examples:
  python test_precision_more_cases.py
  python test_precision_more_cases.py --dtype bf16
  python test_precision_more_cases.py --dtype fp16 --scale-alg 1 --limit 20
  python test_precision_more_cases.py --dtype all --limit 6
"""

import argparse
import ast
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu
import cann_ops_transformer


def _float8_dtype(dst_type):
    if dst_type == 35:
        return getattr(torch, "float8_e5m2", None) or getattr(torch_npu, "float8_e5m2", None)
    if dst_type == 36:
        return getattr(torch, "float8_e4m3fn", None) or getattr(torch_npu, "float8_e4m3fn", None)
    raise ValueError("FP4 should use torch_npu FP4 enum integers")


def _dst_torch_dtype(dst_type):
    if dst_type in (35, 36):
        return _float8_dtype(dst_type)
    if dst_type == 40:
        # torch_npu FP4 packed dtype enum, not GE dst_type 40.
        return 296
    if dst_type == 41:
        return 297
    raise ValueError(f"unsupported dst_type: {dst_type}")


def _small_op(global_out, local_out, global_gate_logits, local_gate_logits, *,
              scale_alg=1, dst_type=36, round_mode="rint"):
    """Unfused NPU small-op reference; this is the only golden used here."""
    orig_dtype = global_out.dtype
    if global_gate_logits.dim() == global_out.dim():
        # [T,N,1] (or legacy [S,B,N,1]) -> [T,N]
        global_gate_logits = global_gate_logits.squeeze(-1)
        local_gate_logits = local_gate_logits.squeeze(-1)
    global_scale = torch.sigmoid(global_gate_logits.to(torch.float32)).unsqueeze(-1)
    local_scale = torch.sigmoid(local_gate_logits.to(torch.float32)).unsqueeze(-1)
    merged = (
        global_out.to(torch.float32) * global_scale
        + local_out.to(torch.float32) * local_scale
    ).to(orig_dtype)
    merged = merged.reshape(merged.shape[0], -1)
    return torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        merged,
        round_mode=round_mode,
        dst_type=_dst_torch_dtype(dst_type),
        scale_alg=scale_alg,
    )


def _rand_tensor(gen, shape, low, high, dtype):
    if low == high:
        return torch.full(shape, low, dtype=dtype)
    return (torch.rand(shape, generator=gen, dtype=torch.float32) * (high - low) + low).to(dtype)


def _load_ttk_cases(case_file):
    """Load the TTK kernel CSV cases used by run_ttk_kernel.sh."""
    with open(case_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cases = []
        for row in reader:
            if str(row.get("is_enabled", "True")).strip().lower() == "false":
                continue
            input_dtypes = ast.literal_eval(row["input_dtypes"])
            shapes = ast.literal_eval(row["input_ori_shapes"])
            attrs = ast.literal_eval(row["attributes"])
            ranges = ast.literal_eval(row["input_data_ranges"])
            cases.append({
                "name": row["testcase_name"],
                "input_dtype": input_dtypes[0],
                "global_out_shape": tuple(shapes[0]),
                "local_out_shape": tuple(shapes[1]),
                "global_gate_shape": tuple(shapes[2]),
                "local_gate_shape": tuple(shapes[3]),
                "dst_type": int(attrs["dst_type"]),
                "scale_alg": int(attrs["scale_alg"]),
                "round_mode": attrs.get("round_mode", "rint"),
                "quant_mode": int(attrs.get("quant_mode", 1)),
                "ranges": [tuple(r) for r in ranges],
            })
        return cases


def _make_case_inputs(case, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dtype_name = case["input_dtype"]
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"unsupported TTK input dtype: {dtype_name}")

    go_low, go_high = case["ranges"][0]
    lo_low, lo_high = case["ranges"][1]
    gg_low, gg_high = case["ranges"][2]
    lg_low, lg_high = case["ranges"][3]

    global_out = _rand_tensor(gen, case["global_out_shape"], go_low, go_high, dtype)
    local_out = _rand_tensor(gen, case["local_out_shape"], lo_low, lo_high, dtype)
    global_gate = _rand_tensor(gen, case["global_gate_shape"], gg_low, gg_high, dtype)
    local_gate = _rand_tensor(gen, case["local_gate_shape"], lg_low, lg_high, dtype)
    return global_out, local_out, global_gate, local_gate


def _tensor_bytes(tensor):
    return tensor.detach().cpu().view(torch.uint8).numpy().tobytes()


def _outputs_match(actual, golden, max_preview=5):
    names = ("row_data", "row_scale", "col_data", "col_scale")
    mismatches = {}
    for name, actual_t, golden_t in zip(names, actual, golden):
        if actual_t.shape != golden_t.shape:
            mismatches[name] = {
                "shape": (tuple(actual_t.shape), tuple(golden_t.shape)),
            }
            continue

        actual_bytes = _tensor_bytes(actual_t)
        golden_bytes = _tensor_bytes(golden_t)
        if actual_bytes == golden_bytes:
            continue

        actual_arr = np.frombuffer(actual_bytes, dtype=np.uint8).reshape(actual_t.shape)
        golden_arr = np.frombuffer(golden_bytes, dtype=np.uint8).reshape(golden_t.shape)
        diff = actual_arr != golden_arr
        mismatch_count = int(np.sum(diff))
        total = int(actual_arr.size)
        first_indices = np.flatnonzero(diff.ravel())[:max_preview]
        first = []
        for flat in first_indices:
            index = tuple(int(x) for x in np.unravel_index(int(flat), actual_arr.shape))
            first.append({
                "index": index,
                "actual": int(actual_arr[index]),
                "golden": int(golden_arr[index]),
            })
        mismatches[name] = {
            "count": mismatch_count,
            "total": total,
            "rate": mismatch_count / total,
            "first": first,
        }
    return mismatches


def _format_mismatch_detail(detail):
    if isinstance(detail, str):
        return detail
    if "shape" in detail:
        actual_shape, golden_shape = detail["shape"]
        return f"shape {actual_shape} != {golden_shape}"
    parts = [f"count={detail['count']}/{detail['total']}", f"rate={detail['rate']:.6%}"]
    if detail["first"]:
        samples = []
        for item in detail["first"]:
            samples.append(
                f"@{item['index']} actual=0x{item['actual']:02x} golden=0x{item['golden']:02x}"
            )
        parts.append("first=" + " | ".join(samples))
    return " ".join(parts)


def main():
    default_case_file = str(
        Path(__file__).resolve().parents[1] / "arch35/ttk_kernel_cla_gate_mx_quant_with_dual_axis_general.csv"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "all"), default="all",
                        help="filter by input dtype; default runs both bf16 and fp16")
    parser.add_argument("--scale-alg", type=int, choices=(0, 1), default=None,
                        help="filter by scale_alg; default follows the TTK CSV")
    parser.add_argument("--case-file", default=default_case_file,
                        help="TTK kernel CSV used as the case list")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit number of cases (0 means all)")
    parser.add_argument("--device", type=int, default=6, help="NPU device id")
    parser.add_argument("--seed", type=int, default=20260905, help="random seed base")
    args = parser.parse_args()

    cases = _load_ttk_cases(args.case_file)
    if args.dtype != "all":
        dtype_map = {"bf16": "bfloat16", "fp16": "float16"}
        cases = [c for c in cases if c["input_dtype"] == dtype_map[args.dtype]]
    if args.scale_alg is not None:
        cases = [c for c in cases if c["scale_alg"] == args.scale_alg]
    if args.limit > 0:
        cases = cases[:args.limit]
    if not cases:
        parser.error("No TTK case matched the current filters")

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"

    failures = []
    total = 0
    print("=" * 100)
    print("CLA gate MX quant precision test (TTK-aligned cases)")
    print(f"device      : {device}")
    print(f"case file   : {args.case_file}")
    print(f"total cases : {len(cases)}")
    print("=" * 100)

    for case_idx, case in enumerate(cases):
        total += 1
        seed = args.seed + case_idx * 1000
        global_out, local_out, global_gate, local_gate = _make_case_inputs(case, seed)

        go = global_out.to(device)
        lo = local_out.to(device)
        gg = global_gate.to(device)
        lg = local_gate.to(device)

        try:
            fused = cann_ops_transformer.cla_gate_mx_quant_with_dual_axis(
                go, lo, gg, lg,
                round_mode=case["round_mode"],
                scale_alg=case["scale_alg"],
                dst_type=case["dst_type"],
                quant_mode=case["quant_mode"],
            )
            golden = _small_op(
                go, lo, gg, lg,
                scale_alg=case["scale_alg"],
                dst_type=case["dst_type"],
                round_mode=case["round_mode"],
            )
            torch.npu.synchronize()
        except Exception as exc:  # noqa: BLE001
            failures.append((case, f"exception: {exc}"))
            print(f"[FAIL] {case['name']}: {exc}")
            continue

        mismatches = _outputs_match(fused, golden, max_preview=10)
        if mismatches:
            failures.append((case, mismatches))
            names = ", ".join(mismatches.keys())
            print(f"[FAIL] {case['name']} mismatched_outputs={names}")
            for name, detail in mismatches.items():
                print(f"      {name}: {_format_mismatch_detail(detail)}")
        else:
            print(f"[PASS] {case['name']}")

    print("=" * 100)
    if failures:
        print(f"Failed {len(failures)}/{total} cases.")
        for case, reason in failures[:50]:
            if isinstance(reason, str):
                print(f"  - {case['name']}: {reason}")
            else:
                print(f"  - {case['name']}: mismatched_outputs={list(reason.keys())}")
        return 1
    print(f"All {total} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
