import argparse
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

from npu_ops_transformer.ops import npu_chunk_gated_delta_rule, npu_chunk_gated_delta_rule_neumann

CURRENT_DIR = Path(__file__).resolve().parent

from chunk_gated_delta_rule_neumann_golden import cgdr_golden  # noqa: E402
from compare_chunk_gdr_three_way_paramset import THREE_WAY_CASES, case_id  # noqa: E402

_ = (npu_chunk_gated_delta_rule, npu_chunk_gated_delta_rule_neumann)


def make_case_args(case):
    defaults = {
        "case_name": "custom",
        "device": int(os.getenv("NPU_DEVICE_ID", "0")),
        "seed": 21,
        "batch_size": 1,
        "seqlen": 64,
        "seqlens": "",
        "nk": 4,
        "nv": 4,
        "dk": 128,
        "dv": 128,
        "golden_float64": False,
        "atol": 8e-2,
        "rtol": 8e-2,
    }
    defaults.update(case)
    defaults["case_name"] = defaults.pop("name", defaults["case_name"])
    return argparse.Namespace(**defaults)


def parse_seqlens(seqlens, batch_size, seqlen):
    if seqlens:
        parsed = [int(item) for item in seqlens.split(",") if item.strip()]
        if len(parsed) != batch_size:
            raise ValueError(f"--seqlens length should equal B={batch_size}, but got {len(parsed)}.")
        return parsed
    return [seqlen] * batch_size


def make_inputs(args):
    torch.manual_seed(args.seed)
    torch_npu.npu.set_device(args.device)
    device = f"npu:{args.device}"
    actual_lengths = parse_seqlens(args.seqlens, args.batch_size, args.seqlen)
    total_length = sum(actual_lengths)
    dtype = torch.bfloat16

    query = torch.rand((total_length, args.nk, args.dk), dtype=dtype, device=device)
    key = torch.rand((total_length, args.nk, args.dk), dtype=dtype, device=device)
    value = torch.rand((total_length, args.nv, args.dv), dtype=dtype, device=device)
    beta = torch.rand((total_length, args.nv), dtype=dtype, device=device)
    g = torch.rand((total_length, args.nv), dtype=torch.float32, device=device) * -1.0
    initial_state = torch.rand((args.batch_size, args.nv, args.dv, args.dk), dtype=dtype, device=device)
    actual_seq_lengths = torch.tensor(actual_lengths, dtype=torch.int32, device=device)

    query = F.normalize(query, p=2, dim=-1)
    key = F.normalize(key, p=2, dim=-1)
    scale = 1 / (args.dk ** 0.5)
    return query, key, value, g, beta, scale, initial_state, actual_seq_lengths


def run_cpu_golden(query, key, value, g, beta, scale, initial_state, actual_seq_lengths, use_float64):
    out, final_state = cgdr_golden(
        query,
        key,
        value,
        g,
        beta,
        scale,
        initial_state,
        actual_seq_lengths,
        use_float64=use_float64,
    )
    torch.npu.synchronize()
    return out.cpu().to(torch.float32), final_state.cpu().to(torch.float32)


def run_normal_op(query, key, value, g, beta, scale, initial_state, actual_seq_lengths):
    out, final_state = torch_npu.npu_chunk_gated_delta_rule(
        query,
        key,
        value,
        beta=beta,
        initial_state=initial_state.clone(),
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g,
    )
    torch.npu.synchronize()
    return out.cpu().to(torch.float32), final_state.cpu().to(torch.float32)


def run_neumann_op(query, key, value, g, beta, scale, initial_state, actual_seq_lengths):
    out, final_state = torch_npu.npu_chunk_gated_delta_rule_neumann(
        query,
        key,
        value,
        beta=beta,
        initial_state=initial_state.clone(),
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g,
    )
    torch.npu.synchronize()
    return out.cpu().to(torch.float32), final_state.cpu().to(torch.float32)


def calc_metrics(expect, actual):
    diff = (actual - expect).to(torch.float32)
    denom = torch.clamp(expect.abs().to(torch.float32), min=1e-3)
    rel = diff.abs() / denom
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rmse": torch.sqrt(torch.mean(diff * diff)).item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
    }


def print_metrics(name, expect, actual, atol, rtol):
    metrics = calc_metrics(expect, actual)
    passed = torch.allclose(actual, expect, atol=atol, rtol=rtol)
    print(
        f"{name}: {'PASS' if passed else 'FAIL'} "
        f"max_abs={metrics['max_abs']:.6e}, mean_abs={metrics['mean_abs']:.6e}, "
        f"rmse={metrics['rmse']:.6e}, max_rel={metrics['max_rel']:.6e}, mean_rel={metrics['mean_rel']:.6e}"
    )
    if not passed:
        flat_idx = torch.argmax((actual - expect).abs()).item()
        print(
            f"  worst_idx={flat_idx}, expect={expect.flatten()[flat_idx].item():.8e}, "
            f"actual={actual.flatten()[flat_idx].item():.8e}"
        )
    return passed


def run_three_way_case(args):
    inputs = make_inputs(args)
    query, key, value, g, beta, scale, initial_state, actual_seq_lengths = inputs
    print(
        f"case={getattr(args, 'case_name', 'custom')}: "
        f"B={args.batch_size}, seqlens={actual_seq_lengths.cpu().tolist()}, "
        f"nk={args.nk}, nv={args.nv}, dk={args.dk}, dv={args.dv}, "
        f"golden_float64={args.golden_float64}, seed={args.seed}, scale={scale:.6f}"
    )

    golden_out, golden_state = run_cpu_golden(*inputs, use_float64=args.golden_float64)
    normal_out, normal_state = run_normal_op(*inputs)
    neumann_out, neumann_state = run_neumann_op(*inputs)

    checks = [
        print_metrics("CPU vs ChunkGDR out", golden_out, normal_out, args.atol, args.rtol),
        print_metrics("CPU vs ChunkGDR state", golden_state, normal_state, args.atol, args.rtol),
        print_metrics("CPU vs ChunkGDRNeumann out", golden_out, neumann_out, args.atol, args.rtol),
        print_metrics("CPU vs ChunkGDRNeumann state", golden_state, neumann_state, args.atol, args.rtol),
        print_metrics("ChunkGDR vs ChunkGDRNeumann out", normal_out, neumann_out, args.atol, args.rtol),
        print_metrics("ChunkGDR vs ChunkGDRNeumann state", normal_state, neumann_state, args.atol, args.rtol),
    ]
    print("PASSED" if all(checks) else "FAILED")
    return all(checks)


@pytest.mark.ci
@pytest.mark.parametrize("case", THREE_WAY_CASES, ids=case_id)
def test_three_way_precision(case):
    args = make_case_args(case)
    assert run_three_way_case(args), f"three-way precision failed: {args.case_name}"


def main():
    parser = argparse.ArgumentParser(description="Compare CPU golden, ChunkGatedDeltaRule, and Neumann NPU ops.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch-size", "-B", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--seqlens", type=str, default="", help="Comma separated variable sequence lengths.")
    parser.add_argument("--nk", type=int, default=4)
    parser.add_argument("--nv", type=int, default=4)
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--golden-float64", action="store_true", help="Use original golden's float64 path.")
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    args = parser.parse_args()
    args.case_name = "cli"
    passed = run_three_way_case(args)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
