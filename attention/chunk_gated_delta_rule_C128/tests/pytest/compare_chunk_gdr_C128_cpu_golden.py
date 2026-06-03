#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch_npu

REPO_ROOT = Path(__file__).resolve().parents[4]
TORCH_EXTENSION_DIR = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_DIR.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_DIR))

from npu_ops_transformer.ops import npu_chunk_gated_delta_rule  # noqa: E402
from npu_ops_transformer.ops import npu_chunk_gated_delta_rule_C128  # noqa: E402

from chunk_gated_delta_rule_C128_golden import cgdr_golden, cgdr_npu  # noqa: E402
from compare_chunk_gdr_C128_cpu_golden_paramset import CPU_GOLDEN_CASES, case_id  # noqa: E402

_ = npu_chunk_gated_delta_rule
_ = npu_chunk_gated_delta_rule_C128

DEFAULT_THRESHOLD = 0.005
DTYPE = torch.bfloat16


def make_case_args(case):
    defaults = {
        "case_name": "custom",
        "device": int(os.getenv("NPU_DEVICE_ID", "0")),
        "seed": 21,
        "batch_size": 1,
        "seqlen": 128,
        "seqlens": "",
        "nk": 4,
        "nv": 4,
        "dk": 128,
        "dv": 128,
        "chunk_size": 128,
        "atol": DEFAULT_THRESHOLD,
        "rtol": DEFAULT_THRESHOLD,
    }
    defaults.update(case)
    defaults["case_name"] = defaults.pop("name", defaults["case_name"])
    return argparse.Namespace(**defaults)


def parse_seqlens(seqlens, batch_size, seqlen):
    if not seqlens:
        return [seqlen] * batch_size
    parsed = [int(item) for item in seqlens.split(",") if item.strip()]
    if len(parsed) != batch_size:
        raise ValueError(f"--seqlens length should equal B={batch_size}, but got {len(parsed)}.")
    return parsed


def make_inputs(args):
    torch.manual_seed(args.seed)
    torch_npu.npu.set_device(args.device)
    device = f"npu:{args.device}"
    actual_lengths = parse_seqlens(args.seqlens, args.batch_size, args.seqlen)
    total_length = sum(actual_lengths)

    query = F.normalize(torch.rand((total_length, args.nk, args.dk), dtype=torch.float32), p=2, dim=-1).to(DTYPE)
    key = F.normalize(torch.rand((total_length, args.nk, args.dk), dtype=torch.float32), p=2, dim=-1).to(DTYPE)
    value = torch.rand((total_length, args.nv, args.dv), dtype=DTYPE)
    beta = torch.rand((total_length, args.nv), dtype=DTYPE)
    g = torch.rand((total_length, args.nv), dtype=torch.float32) * -1.0
    initial_state = torch.rand((args.batch_size, args.nv, args.dv, args.dk), dtype=DTYPE)

    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    beta = beta.to(device)
    g = g.to(device)
    initial_state = initial_state.to(device)
    actual_seq_lengths = torch.tensor(actual_lengths, dtype=torch.int32, device=device)

    scale = 1.0 / math.sqrt(args.dk)
    return query, key, value, g, beta, scale, initial_state, actual_seq_lengths


def calc_metrics(expect, actual):
    expect = expect.cpu().to(torch.float32)
    actual = actual.cpu().to(torch.float32)
    diff = actual - expect
    denom = torch.clamp(expect.abs(), min=1e-6)
    rel = diff.abs() / denom
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rmse": torch.sqrt(torch.mean(diff * diff)).item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
    }


def print_tensor_check(name, expect, actual, atol, rtol):
    expect = expect.cpu().to(torch.float32)
    actual = actual.cpu().to(torch.float32)
    metrics = calc_metrics(expect, actual)
    passed = torch.allclose(actual, expect, atol=atol, rtol=rtol)
    print(
        f"{name}: {'PASS' if passed else 'FAIL'} "
        f"atol={atol:.6g}, rtol={rtol:.6g}, "
        f"max_abs={metrics['max_abs']:.6e}, mean_abs={metrics['mean_abs']:.6e}, "
        f"rmse={metrics['rmse']:.6e}, max_rel={metrics['max_rel']:.6e}, "
        f"mean_rel={metrics['mean_rel']:.6e}"
    )
    if not passed:
        abs_diff = (actual - expect).abs().flatten()
        flat_idx = torch.argmax(abs_diff).item()
        print(
            f"  worst_idx={flat_idx}, expect={expect.flatten()[flat_idx].item():.8e}, "
            f"actual={actual.flatten()[flat_idx].item():.8e}, "
            f"abs_diff={abs_diff[flat_idx].item():.8e}"
        )
    return passed


def run_gdn_npu(query, key, value, g, beta, scale, initial_state, actual_seq_lengths):
    out, state = torch_npu.npu_chunk_gated_delta_rule(
        query,
        key,
        value,
        beta=beta,
        initial_state=initial_state.clone(),
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g,
    )
    return out.to(torch.float32), state.to(torch.float32)


def run_cpu_golden_case(args):
    inputs = make_inputs(args)
    query, key, value, g, beta, scale, initial_state, actual_seq_lengths = inputs
    print(
        f"case={args.case_name}: B={args.batch_size}, "
        f"seqlens={actual_seq_lengths.cpu().tolist()}, nk={args.nk}, nv={args.nv}, "
        f"dk={args.dk}, dv={args.dv}, chunk_size={args.chunk_size}, seed={args.seed}, "
        f"atol={args.atol}, rtol={args.rtol}"
    )

    golden_out, golden_state = cgdr_golden(
        query,
        key,
        value,
        g,
        beta,
        scale,
        initial_state,
        actual_seq_lengths,
        chunk_size=args.chunk_size,
        use_float64=True,
    )
    torch_npu.npu.synchronize()
    gdn_out, gdn_state = run_gdn_npu(query, key, value, g, beta, scale, initial_state, actual_seq_lengths)
    torch_npu.npu.synchronize()
    c128_out, c128_state = cgdr_npu(query, key, value, g, beta, scale, initial_state, actual_seq_lengths)
    torch_npu.npu.synchronize()

    checks = [
        print_tensor_check("CPU golden vs GDN NPU out", golden_out, gdn_out, args.atol, args.rtol),
        print_tensor_check("CPU golden vs GDN NPU state", golden_state, gdn_state, args.atol, args.rtol),
        print_tensor_check("CPU golden vs C128 NPU out", golden_out, c128_out, args.atol, args.rtol),
        print_tensor_check("CPU golden vs C128 NPU state", golden_state, c128_state, args.atol, args.rtol),
        print_tensor_check("GDN NPU vs C128 NPU out", gdn_out, c128_out, args.atol, args.rtol),
        print_tensor_check("GDN NPU vs C128 NPU state", gdn_state, c128_state, args.atol, args.rtol),
    ]
    print("PASSED" if all(checks) else "FAILED")
    return all(checks)


@pytest.mark.ci
@pytest.mark.parametrize("case", CPU_GOLDEN_CASES, ids=case_id)
def test_cpu_golden_precision(case):
    args = make_case_args(case)
    assert run_cpu_golden_case(args), f"three-way precision failed: {args.case_name}"


def main():
    parser = argparse.ArgumentParser(description="Three-way compare CPU golden, GDN NPU, and C128 NPU.")
    parser.add_argument("--device", type=int, default=int(os.getenv("NPU_DEVICE_ID", "0")))
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch-size", "-B", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--seqlens", type=str, default="", help="Comma separated variable sequence lengths.")
    parser.add_argument("--nk", type=int, default=4)
    parser.add_argument("--nv", type=int, default=4)
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--atol", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--rtol", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    args.case_name = "cli"
    return 0 if run_cpu_golden_case(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
