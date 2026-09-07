#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Generic-case precision test for cla_gate_mx_quant_with_dual_axis.

For the same random inputs:
  1. call the transformer fused implementation:
       cann_ops_transformer.cla_gate_mx_quant_with_dual_axis(...)
  2. call the unfused NPU small-op reference:
       sigmoid gate -> broadcast mul/add -> reshape [T, -1]
       -> torch_npu.npu_dynamic_mx_quant_with_dual_axis(...)

The two outputs are compared byte-by-byte.

Examples:
  python test_precision_more_cases.py
  python test_precision_more_cases.py --dtype bf16
  python test_precision_more_cases.py --dtype fp16 --scale-alg 1 --num-cases 20
  python test_precision_more_cases.py --dtype all --num-cases 6
"""

import argparse
import sys

import numpy as np
import torch
import torch_npu
import cann_ops_transformer


def _float8_e4m3fn():
    return getattr(torch, "float8_e4m3fn", None) or getattr(torch_npu, "float8_e4m3fn", None)


def _small_op(global_out, local_out, global_gate_logits, local_gate_logits, scale_alg):
    """Unfused NPU small-op reference; this is the only golden used here."""
    orig_dtype = global_out.dtype
    global_scale = torch.sigmoid(global_gate_logits.to(torch.float32)).unsqueeze(-1)
    local_scale = torch.sigmoid(local_gate_logits.to(torch.float32)).unsqueeze(-1)
    merged = (
        global_out.to(torch.float32) * global_scale
        + local_out.to(torch.float32) * local_scale
    ).to(orig_dtype)
    merged = merged.reshape(merged.shape[0], -1)
    return torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        merged,
        dst_type=_float8_e4m3fn(),
        scale_alg=scale_alg,
    )


def _rand_tensor(gen, shape, low, high, dtype):
    return (torch.rand(shape, generator=gen, dtype=torch.float32) * (high - low) + low).to(dtype)


def _make_case_inputs(shape, dtype, seed, pattern):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t, n, d = shape
    gate_shape = (t, n)

    if pattern == "random":
        global_out = _rand_tensor(gen, shape, -2.0, 2.0, dtype)
        local_out = _rand_tensor(gen, shape, -2.0, 2.0, dtype)
        global_gate = _rand_tensor(gen, gate_shape, -8.0, 8.0, dtype)
        local_gate = _rand_tensor(gen, gate_shape, -8.0, 8.0, dtype)
    elif pattern == "saturate":
        global_out = _rand_tensor(gen, shape, -1.0, 1.0, dtype)
        local_out = _rand_tensor(gen, shape, -1.0, 1.0, dtype)
        global_gate = _rand_tensor(gen, gate_shape, -30.0, 30.0, dtype)
        local_gate = _rand_tensor(gen, gate_shape, -30.0, 30.0, dtype)
    elif pattern == "zero":
        global_out = torch.zeros(shape, dtype=dtype)
        local_out = torch.zeros(shape, dtype=dtype)
        global_gate = _rand_tensor(gen, gate_shape, -4.0, 4.0, dtype)
        local_gate = _rand_tensor(gen, gate_shape, -4.0, 4.0, dtype)
    elif pattern == "one":
        global_out = torch.ones(shape, dtype=dtype)
        local_out = torch.ones(shape, dtype=dtype)
        global_gate = _rand_tensor(gen, gate_shape, -4.0, 4.0, dtype)
        local_gate = _rand_tensor(gen, gate_shape, -4.0, 4.0, dtype)
    elif pattern == "small":
        global_out = _rand_tensor(gen, shape, -0.01, 0.01, dtype)
        local_out = _rand_tensor(gen, shape, -0.01, 0.01, dtype)
        global_gate = _rand_tensor(gen, gate_shape, -2.0, 2.0, dtype)
        local_gate = _rand_tensor(gen, gate_shape, -2.0, 2.0, dtype)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return global_out, local_out, global_gate, local_gate


def _case_descriptors():
    """(shape, pattern) pairs are expanded according to --num-cases."""
    descriptors = [
        ((1023, 96, 256), "random"),
        ((64, 96, 256), "random"),
        ((128, 128, 128), "random"),
        ((64, 4, 256), "random"),
        ((64, 96, 256), "saturate"),
        ((64, 96, 256), "zero"),
        ((64, 96, 256), "one"),
        ((64, 96, 256), "small"),
    ]
    return descriptors


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "all"), default="all",
                        help="input dtype; default runs both bf16 and fp16")
    parser.add_argument("--scale-alg", type=int, choices=(0, 1), default=1,
                        help="scale_alg passed to both implementations (default: 1)")
    parser.add_argument("--num-cases", type=int, default=8,
                        help="number of cases per dtype (default: 8)")
    parser.add_argument("--device", type=int, default=0, help="NPU device id")
    parser.add_argument("--seed", type=int, default=20260905, help="random seed base")
    args = parser.parse_args()

    dtypes = []
    if args.dtype in ("bf16", "all"):
        dtypes.append(("bf16", torch.bfloat16))
    if args.dtype in ("fp16", "all"):
        dtypes.append(("fp16", torch.float16))
    if not dtypes:
        parser.error("--dtype must be one of bf16/fp16/all")

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"

    descriptors = _case_descriptors()
    if args.num_cases <= 0:
        parser.error("--num-cases must be positive")

    failures = []
    total = 0
    print("=" * 100)
    print("CLA gate MX quant precision test")
    print(f"device      : {device}")
    print(f"dtypes      : {[d[0] for d in dtypes]}")
    print(f"scale_alg   : {args.scale_alg}")
    print(f"num_cases   : {args.num_cases} per dtype")
    print("=" * 100)

    for dtype_name, dtype in dtypes:
        for case_idx in range(args.num_cases):
            total += 1
            shape, pattern = descriptors[case_idx % len(descriptors)]
            seed = args.seed + case_idx * 1000 + len(dtypes) * case_idx
            global_out, local_out, global_gate, local_gate = _make_case_inputs(
                shape, dtype, seed, pattern
            )

            go = global_out.to(device)
            lo = local_out.to(device)
            gg = global_gate.to(device)
            lg = local_gate.to(device)

            # Fused transformer implementation.
            fused = cann_ops_transformer.cla_gate_mx_quant_with_dual_axis(
                go, lo, gg, lg,
                round_mode="rint",
                scale_alg=args.scale_alg,
                dst_type=36,
            )

            # Unfused NPU small-op golden.
            golden = _small_op(go, lo, gg, lg, args.scale_alg)
            torch.npu.synchronize()

            mismatches = _outputs_match(fused, golden, max_preview=10)
            status = "PASS" if not mismatches else "FAIL"
            if mismatches:
                failures.append((dtype_name, case_idx, shape, pattern, mismatches))
            if mismatches:
                names = ", ".join(mismatches.keys())
                print(f"[{status}] {dtype_name:5s} case={case_idx:<2d} shape={shape} "
                      f"pattern={pattern:<8s} scale_alg={args.scale_alg} "
                      f"mismatched_outputs={names}")
                for name, detail in mismatches.items():
                    print(f"      {name}: {_format_mismatch_detail(detail)}")
            else:
                print(f"[{status}] {dtype_name:5s} case={case_idx:<2d} shape={shape} "
                      f"pattern={pattern:<8s} scale_alg={args.scale_alg}")

    print("=" * 100)
    if failures:
        print(f"Failed {len(failures)}/{total} cases.")
        for item in failures:
            dtype_name, case_idx, shape, pattern, mismatches = item
            print(f"  - {dtype_name} case={case_idx} shape={shape} pattern={pattern} "
                  f"mismatched_outputs={list(mismatches.keys())}")
            for name, detail in mismatches.items():
                print(f"      {name}: {_format_mismatch_detail(detail)}")
        return 1
    print(f"All {total} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())