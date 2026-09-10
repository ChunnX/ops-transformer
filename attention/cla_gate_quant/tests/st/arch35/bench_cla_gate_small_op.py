#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Small-op baseline benchmark for the fixed CLA-gate dual-axis quant case.

It runs the exact unfused reference path:
    global_gate  = sigmoid(global_gate_logits)  broadcast over D
    local_gate   = sigmoid(local_gate_logits)   broadcast over D
    merged       = global_out * global_gate + local_out * local_gate
    row_data, row_scale, col_data, col_scale =
        torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            merged.reshape(merged.shape[0], merged.shape[1] * merged.shape[2]),
            dst_type=torch.float8_e4m3fn, scale_alg=1)

Use this to compare against the fused npu_cla_gate_quant
kernel on the same [256, 4, 96, 256] shape.

Example:
    source /home/g00940185/work/CANN9.2.0/cann-9.2.0/set_env.sh
    python3 bench_cla_gate_small_op.py --dtype bf16 --device 0 --warmup 10 --repeat 100
"""

import argparse
import statistics
import time

import torch
import torch_npu


def _get_float8_e4m3fn():
    """Return the FP8 E4M3FN dtype accepted by the local torch/torch_npu."""
    for candidate in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch_npu, "float8_e4m3fn", None),
    ):
        if candidate is not None:
            return candidate
    return 36  # ge::DT_FLOAT8_E4M3FN enum fallback

FLOAT8_E4M3FN = _get_float8_e4m3fn()


def cla_gate(global_out, local_out, global_logits, local_logits):
    orig_dtype = global_out.dtype
    global_scale = torch.sigmoid(global_logits.to(torch.float32)).unsqueeze(-1)
    local_scale = torch.sigmoid(local_logits.to(torch.float32)).unsqueeze(-1)
    merged = (
        global_out.to(torch.float32) * global_scale
        + local_out.to(torch.float32) * local_scale
    )
    return merged.to(orig_dtype)


def cla_gate_quant_dual(global_out, local_out, global_logits, local_logits):
    merged = cla_gate(global_out, local_out, global_logits, local_logits)
    merged = merged.reshape(merged.shape[0], merged.shape[1] * merged.shape[2])
    return torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        merged, dst_type=FLOAT8_E4M3FN, scale_alg=1
    )


def create_inputs(shape, dtype, device, seed):
    """Random inputs, generated on CPU to keep the timed NPU path clean."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def rand_tensor(size, low, high):
        return (torch.rand(size, generator=generator, dtype=torch.float32) *
                (high - low) + low).to(dtype).to(device)

    gate_shape = shape[:2]
    g_out = rand_tensor(shape, -2.0, 2.0)
    l_out = rand_tensor(shape, -2.0, 2.0)
    # Gate logits are the sigmoid inputs; this range exercises both tails.
    g_logits = rand_tensor(gate_shape, -8.0, 8.0)
    l_logits = rand_tensor(gate_shape, -8.0, 8.0)
    return g_out, l_out, g_logits, l_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"],
                        help="input dtype, default bf16")
    parser.add_argument("--device", type=int, default=0, help="NPU device id")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for input data")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="timed iterations")
    args = parser.parse_args()

    device_name = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # TND shape used by the CLA gate operator.
    shape = (64, 96, 256)

    global_out, local_out, global_logits, local_logits = create_inputs(
        shape, dtype, device_name, args.seed
    )

    # Run once before timing so framework init / graph build cost is excluded.
    _ = cla_gate_quant_dual(global_out, local_out, global_logits, local_logits)
    torch.npu.synchronize()

    latencies = []
    for i in range(args.warmup + args.repeat):
        start = time.perf_counter()
        row_data, row_scale, col_data, col_scale = cla_gate_quant_dual(
            global_out, local_out, global_logits, local_logits
        )
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        if i >= args.warmup:
            latencies.append(elapsed_ms)

    latencies.sort()
    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)

    # IO approximation for the small-op baseline: fused sub-ops materialize the
    # gated tensors and the full BF16/FP16 merged tensor in HBM before quant.
    # Exact intermediate traffic depends on graph optimization, so only the
    # logical sizes are reported below.
    numel = shape[0] * shape[1] * shape[2] * shape[3]
    bytes_per_elem = 2  # both fp16 and bf16 are 2 bytes
    logical_merged_bytes = numel * bytes_per_elem
    print("=" * 78)
    print("CLA gate + dual-axis MXFP8 small-op baseline")
    print("=" * 78)
    print(f"device           : {device_name}")
    print(f"input dtype      : {args.dtype}")
    print(f"input shape      : {shape}  (global_out / local_out TND)")
    print(f"gate shape       : {shape[:3]}  (global/local gate logits)")
    print(f"row/col data     : {tuple(row_data.shape)}")
    print(f"row_scale        : {tuple(row_scale.shape)}")
    print(f"col_scale        : {tuple(col_scale.shape)}")
    print(f"iterations       : warmup={args.warmup}, repeat={args.repeat}")
    print("-" * 78)
    print(f"latency (ms)     : min={min_ms:.4f}, median={median_ms:.4f}, "
          f"mean={mean_ms:.4f}, max={max_ms:.4f}")
    print(f"QPS (1/s)        : mean={1.0 / (mean_ms * 1e-3):.2f}")
    print(f"logical merged   : {logical_merged_bytes / 1024 / 1024:.2f} MiB "
          f"({args.dtype} merged tensor only)")
    print("=" * 78)


if __name__ == "__main__":
    main()
