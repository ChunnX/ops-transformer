#!/usr/bin/env python3
"""Independent S8S4 golden composed from small torch_npu operations.

The formal ACLNN test exports one case with S8S4_DUMP_DIR. This program reads
that dump, reconstructs ND/NZ INT4 weight, computes a CPU reference, and builds
an NPU reference from INT8 quant-matmul/FixPipe, add and cast operations.
"""
import argparse
from pathlib import Path
import sys

import numpy as np


def read_meta(root: Path):
    result = {}
    for line in (root / "meta.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split("=", 1)
            result[key] = value
    return result


def unpack_all_int4(path: Path):
    packed = np.fromfile(path, dtype=np.uint8)
    values = np.empty(packed.size * 2, dtype=np.int8)
    low = (packed & 0x0F).astype(np.int8)
    high = (packed >> 4).astype(np.int8)
    low[low >= 8] -= 16
    high[high >= 8] -= 16
    values[0::2] = low
    values[1::2] = high
    return values


def restore_weight(root, meta, e_count, k_size, n_size):
    physical = unpack_all_int4(root / "weight.i4.bin")
    if meta["weight_format"] == "ND":
        expected = e_count * k_size * n_size
        if physical.size != expected:
            raise ValueError(f"ND weight scalar count mismatch: {physical.size} != {expected}")
        return physical.reshape(e_count, k_size, n_size)

    if meta["weight_format"] != "NZ":
        raise ValueError(f"unsupported weight_format={meta['weight_format']}")
    if int(meta.get("special_nz", "0")):
        k_blocks, n_blocks = k_size // 64, n_size // 16
        expected = e_count * k_blocks * n_blocks * 16 * 64
        if physical.size != expected:
            raise ValueError(f"special NZ scalar count mismatch: {physical.size} != {expected}")
        # Physical axes: [E, K/64, N/16, n-inner, k-inner].
        return physical.reshape(e_count, k_blocks, n_blocks, 16, 64).transpose(
            0, 1, 4, 2, 3
        ).reshape(e_count, k_size, n_size)

    n_blocks, k_blocks = n_size // 64, k_size // 16
    expected = e_count * n_blocks * k_blocks * 16 * 64
    if physical.size != expected:
        raise ValueError(f"NZ scalar count mismatch: {physical.size} != {expected}")
    # Physical axes: [E, N/64, K/16, k-inner, n-inner].
    return physical.reshape(e_count, n_blocks, k_blocks, 16, 64).transpose(
        0, 2, 3, 1, 4
    ).reshape(e_count, k_size, n_size)


def bf16_to_float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def compare(name, lhs, rhs):
    lhs32, rhs32 = lhs.astype(np.float32), rhs.astype(np.float32)
    abs_error = np.abs(lhs32 - rhs32)
    rel_error = abs_error / np.maximum(np.abs(rhs32), 1.0e-6)
    abs_flat, rel_flat = int(np.argmax(abs_error)), int(np.argmax(rel_error))
    abs_m, abs_n = np.unravel_index(abs_flat, abs_error.shape)
    rel_m, rel_n = np.unravel_index(rel_flat, rel_error.shape)
    print(
        f"{name}: max_abs_error={abs_error[abs_m, abs_n]:.9g} "
        f"abs_at=[{abs_m},{abs_n}] lhs={lhs32[abs_m, abs_n]:.9g} "
        f"rhs={rhs32[abs_m, abs_n]:.9g} max_rel_error={rel_error[rel_m, rel_n]:.9g} "
        f"rel_at=[{rel_m},{rel_n}]"
    )


SMALL_VALUE_RULES = {
    "fp16": (2.0 ** -11, 2.0 ** -16),
    "bf16": (2.0 ** -8, 2.0 ** -16),
}


def error_metrics(actual, golden, normal_mask):
    actual64 = actual.astype(np.float64)[normal_mask]
    golden64 = golden.astype(np.float64)[normal_mask]
    if golden64.size == 0:
        return {"MARE": 0.0, "MERE": 0.0, "RMSE": 0.0}
    difference = actual64 - golden64
    relative = np.abs(difference) / (np.abs(golden64) + 1.0e-7)
    return {
        "MARE": float(np.max(relative)),
        "MERE": float(np.mean(relative)),
        "RMSE": float(np.sqrt(np.mean(np.square(difference)))),
    }


def small_value_error_count(actual, golden, small_mask, error_limit):
    difference = np.abs(actual.astype(np.float64) - golden.astype(np.float64))
    return int(np.count_nonzero(small_mask & (difference > error_limit)))


def metric_ratio(numerator, denominator):
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--save-npu", type=Path)
    args = parser.parse_args()

    root, meta = args.dump_dir, read_meta(args.dump_dir)
    e_count, m_total, n_size, k_size, group_count = (
        int(meta[key]) for key in ("E", "M", "N", "K", "G")
    )
    group_size = int(meta["group_size"])
    mode = meta["dequant_mode"]
    if mode not in ("perchannel", "pergroup"):
        raise ValueError(f"unsupported dequant_mode={mode}")

    x = np.fromfile(root / "x.i8.bin", dtype=np.int8).reshape(m_total, k_size)
    weight = restore_weight(root, meta, e_count, k_size, n_size)
    scale = np.fromfile(root / "scale.f32.bin", dtype=np.float32).reshape(
        e_count, group_count, n_size
    )
    groups = np.fromfile(root / "groups.i64.bin", dtype=np.int64)
    if groups.size != e_count or int(groups.sum()) != m_total:
        raise ValueError(f"invalid groups={groups.tolist()} for E={e_count}, M={m_total}")
    bias = np.fromfile(root / "bias.f32.bin", dtype=np.float32).reshape(e_count, n_size)
    per_token = np.fromfile(root / "per_token.f32.bin", dtype=np.float32)
    offset_raw = np.fromfile(root / "offset.f32.bin", dtype=np.float32)
    offset = offset_raw.reshape(e_count, n_size) if mode == "perchannel" else None
    kernel_bits = np.fromfile(root / "kernel.u16.bin", dtype=np.uint16).reshape(m_total, n_size)
    kernel = bf16_to_float(kernel_bits) if meta["output_dtype"] == "bf16" else \
        kernel_bits.view(np.float16).astype(np.float32)
    k_ranges = [(0, k_size)] if mode == "perchannel" else [
        (g * group_size, min((g + 1) * group_size, k_size)) for g in range(group_count)
    ]

    cpu = np.zeros((m_total, n_size), dtype=np.float32)
    row_start = 0
    for expert, rows in enumerate(groups.tolist()):
        row_end = row_start + rows
        for group, (k_start, k_end) in enumerate(k_ranges):
            dot = x[row_start:row_end, k_start:k_end].astype(np.int32) @ \
                  weight[expert, k_start:k_end, :].astype(np.int32)
            cpu[row_start:row_end] += dot.astype(np.float32) * scale[expert, group]
        if mode == "perchannel":
            row_sum = x[row_start:row_end].astype(np.int32).sum(axis=1, keepdims=True)
            cpu[row_start:row_end] += row_sum.astype(np.float32) * offset[expert]
        cpu[row_start:row_end] += bias[expert]
        row_start = row_end
    cpu *= per_token[:, None]

    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("torch and torch_npu are required for NPU stitched golden") from error

    if not hasattr(torch_npu, "npu_quant_matmul") or not hasattr(torch_npu, "npu_trans_quant_param"):
        raise RuntimeError(
            "installed torch_npu must provide npu_quant_matmul and npu_trans_quant_param"
        )
    output_torch_dtype = torch.bfloat16 if meta["output_dtype"] == "bf16" else torch.float16
    fixpipe_dtype = torch.float16
    npu_parts, row_start = [], 0
    with torch.no_grad():
        for expert, rows in enumerate(groups.tolist()):
            row_end = row_start + rows
            if rows == 0:
                npu_parts.append(np.empty((0, n_size), dtype=np.float32))
                continue
            # Keep Cube inputs as INT8. Each group goes through QuantMatmul and
            # FixPipe scale conversion to FP16. The formal BlockMmad workspace
            # CType is half and SetAtomicAdd<half> rounds every group addition.
            x_npu = torch.from_numpy(x[row_start:row_end]).to(args.device, dtype=torch.int8)
            weight_npu = torch.from_numpy(np.ascontiguousarray(weight[expert])).to(
                args.device, dtype=torch.int8
            )
            scale_npu = torch.from_numpy(np.ascontiguousarray(scale[expert])).to(
                args.device, dtype=torch.float32
            )
            quant_scale_npu = torch_npu.npu_trans_quant_param(scale_npu, None)
            result = torch.zeros((rows, n_size), device=args.device, dtype=fixpipe_dtype)
            for group, (k_start, k_end) in enumerate(k_ranges):
                partial = torch_npu.npu_quant_matmul(
                    x_npu[:, k_start:k_end].contiguous(),
                    weight_npu[k_start:k_end, :].contiguous(),
                    quant_scale_npu[group].contiguous(),
                    output_dtype=fixpipe_dtype,
                )
                result.add_(partial)
            # Epilogue widens the FP16 workspace to FP32, applies offset and
            # per-token scale, then casts to the requested output dtype.
            result = result.to(torch.float32)
            if mode == "perchannel":
                offset_npu = torch.from_numpy(offset[expert]).to(args.device)
                row_sum = x_npu.to(torch.int32).sum(dim=1, keepdim=True).to(torch.float32)
                result.add_(row_sum * offset_npu)
            result.add_(torch.from_numpy(bias[expert]).to(args.device))
            result.mul_(torch.from_numpy(per_token[row_start:row_end, None]).to(args.device))
            result = result.to(output_torch_dtype)
            npu_parts.append(result.to(torch.float32).cpu().numpy())
            row_start = row_end
    npu = np.concatenate(npu_parts, axis=0)

    if args.save_npu:
        args.save_npu.parent.mkdir(parents=True, exist_ok=True)
        npu.astype(np.float32).tofile(args.save_npu)
    compare("kernel_vs_cpu", kernel, cpu)
    compare("kernel_vs_npu", kernel, npu)
    compare("npu_vs_cpu", npu, cpu)

    dtype = meta["output_dtype"]
    if dtype not in SMALL_VALUE_RULES:
        raise ValueError(f"no small-value rule configured for output_dtype={dtype}")
    small_threshold, small_error_limit = SMALL_VALUE_RULES[dtype]
    small_mask = np.abs(cpu.astype(np.float64)) < small_threshold
    normal_mask = ~small_mask
    formal_metrics = error_metrics(kernel, cpu, normal_mask)
    stitched_metrics = error_metrics(npu, cpu, normal_mask)
    formal_small_count = small_value_error_count(kernel, cpu, small_mask, small_error_limit)
    stitched_small_count = small_value_error_count(npu, cpu, small_mask, small_error_limit)
    error_count_ratio = formal_small_count / max(stitched_small_count, 1)
    thresholds = {"MARE": 2.0, "MERE": 1.2, "RMSE": 1.2}
    ratios = {
        name: metric_ratio(formal_metrics[name], stitched_metrics[name])
        for name in thresholds
    }
    for name in ("MARE", "MERE", "RMSE"):
        print(
            f"{name}_formal={formal_metrics[name]:.9g} "
            f"{name}_stitched_npu={stitched_metrics[name]:.9g} "
            f"{name}_ratio={ratios[name]:.9g} threshold={thresholds[name]:.9g}"
        )
    print(
        f"small_value_threshold={small_threshold:.9g} small_value_error={small_error_limit:.9g} "
        f"small_value_count={int(np.count_nonzero(small_mask))} "
        f"ErrorCount_formal={formal_small_count} "
        f"ErrorCount_stitched_npu={stitched_small_count} "
        f"ErrorCount_ratio={error_count_ratio:.9g} threshold=2"
    )
    normal_passed = all(ratios[name] <= thresholds[name] for name in thresholds)
    small_passed = error_count_ratio <= 2.0
    passed = normal_passed and small_passed
    print("TRIPLE_COMPARE_PASS" if passed else "TRIPLE_COMPARE_FAIL")
    print(
        f"mode={mode} dtype={meta['output_dtype']} weight_format={meta['weight_format']} "
        f"shape=E{e_count} M{m_total} N{n_size} K{k_size} groups={groups.tolist()}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)