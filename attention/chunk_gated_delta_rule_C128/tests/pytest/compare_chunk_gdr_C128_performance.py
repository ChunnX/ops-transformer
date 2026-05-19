#!/usr/bin/env python3
"""
msprof payload for ChunkGDR normal vs C128 prefill profiling.

Timing is collected by msprof outside this process. This script emits one warmup
operator launch and one measurement operator launch, with cache flush work in
between, so the summarizer can pick the last target op as the measurement row.
"""
import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

import torch
import torch.nn.functional as F
import torch_npu

REPO_ROOT = Path(__file__).resolve().parents[4]
TORCH_EXTENSION_DIR = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_DIR.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_DIR))

DTYPE = torch.bfloat16


def make_inputs(args, seed):
    torch.manual_seed(seed)
    total_length = args.B * args.S
    device = f"npu:{args.device}"

    query = torch.rand((total_length, args.Hkv, args.D), dtype=DTYPE, device=device)
    key = torch.rand((total_length, args.Hkv, args.D), dtype=DTYPE, device=device)
    value = torch.rand((total_length, args.Hq, args.D), dtype=DTYPE, device=device)
    beta = torch.rand((total_length, args.Hq), dtype=DTYPE, device=device)
    initial_state = torch.rand((args.B, args.Hq, args.D, args.D), dtype=DTYPE, device=device)
    actual_seq_lengths = torch.full((args.B,), args.S, dtype=torch.int32, device=device)

    query = F.normalize(query, p=2, dim=-1)
    key = F.normalize(key, p=2, dim=-1)
    if args.g_mode == "random":
        g = torch.rand((total_length, args.Hq), dtype=torch.float32, device=device) * -1.0
    else:
        g = None
    return query, key, value, beta, initial_state, actual_seq_lengths, g


def load_runner(ops):
    if ops == "normal":
        from npu_ops_transformer.ops.chunk_gated_delta_rule import npu_chunk_gated_delta_rule

        return npu_chunk_gated_delta_rule
    from npu_ops_transformer.ops.chunk_gated_delta_rule_C128 import npu_chunk_gated_delta_rule_C128

    return npu_chunk_gated_delta_rule_C128


def run_operator(args, runner, inputs):
    query, key, value, beta, initial_state, actual_seq_lengths, g = inputs
    return runner(
        query,
        key,
        value,
        beta=beta,
        initial_state=initial_state,
        actual_seq_lengths=actual_seq_lengths,
        scale=1.0 / math.sqrt(args.D),
        g=g,
    )


def empty_cache():
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()


def flush_l2_with_matmul(args):
    print(f"[Phase 2] MatMul {args.flush_mm}x{args.flush_mm} to flush cache ...")
    a = torch.randn(args.flush_mm, args.flush_mm, dtype=torch.bfloat16).npu()
    b = torch.randn(args.flush_mm, args.flush_mm, dtype=torch.bfloat16).npu()
    torch.npu.synchronize()
    c = torch.mm(a, b)
    torch.npu.synchronize()
    del a, b, c
    empty_cache()


def flush_l2_with_reducemax(args):
    print(
        f"[Phase 3] ReduceMax {args.flush_reduce_outer}x{args.flush_reduce_mid}x{args.flush_reduce_inner} "
        "to flush cache ..."
    )
    r = torch.randn(
        args.flush_reduce_outer,
        args.flush_reduce_mid,
        args.flush_reduce_inner,
        dtype=torch.bfloat16,
    ).npu()
    torch.npu.synchronize()
    reduced = torch.max(r, dim=-1)
    torch.npu.synchronize()
    del r, reduced
    empty_cache()


def main():
    parser = argparse.ArgumentParser(description="msprof payload for ChunkGDR normal/C128 performance profiling.")
    parser.add_argument("--ops", choices=["normal", "c128"], required=True)
    parser.add_argument("--g-mode", choices=["none", "random"], required=True)
    parser.add_argument("--device", type=int, default=int(os.getenv("NPU_DEVICE_ID", "0")))
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--S", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--Hq", type=int, required=True)
    parser.add_argument("--Hkv", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--flush-mm", type=int, default=10240)
    parser.add_argument("--flush-reduce-outer", type=int, default=96)
    parser.add_argument("--flush-reduce-mid", type=int, default=1024)
    parser.add_argument("--flush-reduce-inner", type=int, default=1024)
    args = parser.parse_args()

    if args.Hq % args.Hkv != 0:
        raise ValueError(f"Hq must be divisible by Hkv, got Hq={args.Hq}, Hkv={args.Hkv}.")

    torch.npu.set_device(args.device)
    print(
        f"[Config] ops={args.ops} g_mode={args.g_mode} "
        f"B={args.B} S={args.S} D={args.D} Hq={args.Hq} Hkv={args.Hkv}"
    )
    runner = load_runner(args.ops)

    print("[Phase 1] Warmup operator to load kernel and warm instruction cache ...")
    warm_inputs = make_inputs(args, args.seed + 57)
    torch.npu.synchronize()
    warm_out, warm_state = run_operator(args, runner, warm_inputs)
    torch.npu.synchronize()
    del warm_inputs, warm_out, warm_state
    empty_cache()

    flush_l2_with_matmul(args)
    flush_l2_with_reducemax(args)

    print("[Phase 4] Measurement run (instruction-warm, cache-cold) ...")
    measure_inputs = make_inputs(args, args.seed)
    torch.npu.synchronize()
    out, final_state = run_operator(args, runner, measure_inputs)
    torch.npu.synchronize()
    print(f"[Phase 4] Done: out={list(out.shape)} final_state={list(final_state.shape)} dtype={out.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
