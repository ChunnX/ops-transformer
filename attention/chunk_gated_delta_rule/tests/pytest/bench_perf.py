#!/usr/bin/env python3
"""Performance comparison: baseline vs refactored."""
import os
import time
import torch
import torch_npu
import cann_ops_transformer

CASES = [
    ("S=2048 nv=32",  1, 2048,   32, 32, 128, 128, True),
    ("S=4096 nv=32",  1, 4096,   32, 32, 128, 128, True),
    ("S=8192 nv=32",  1, 8192,   32, 32, 128, 128, True),
    ("S=32768 nv=32", 1, 32768,  32, 32, 128, 128, True),
    ("S=65536 nv=32", 1, 65536,  32, 32, 128, 128, True),
    ("S=131072 nv=32",1, 131072, 32, 32, 128, 128, True),
    ("S=2048 nv=2",   1, 2048,    2, 2, 128, 128, True),
    ("S=2048 nv=32 GQA", 1, 2048, 16, 32, 128, 128, True),
    ("S=10240 nv=64 GQA", 1, 10240, 16, 64, 128, 128, True),
    ("S=2048 nv=32 nog", 1, 2048, 32, 32, 128, 128, False),
    ("B=4 S=32768 nv=32", 4, 32768, 32, 32, 128, 128, True),
    ("S=2048 FP32-state", 1, 2048, 32, 32, 128, 128, True),
]

WARMUP = 3
REPEAT = 10

def bench(label, B, seqlen, nk, nv, dk, dv, has_g):
    dtype = torch.bfloat16
    sdt = torch.float32 if "FP32" in label else torch.bfloat16
    T = B * seqlen
    q = torch.randn(T, nk, dk, dtype=dtype, device="npu:0")
    k = torch.randn(T, nk, dk, dtype=dtype, device="npu:0")
    v = torch.randn(T, nv, dv, dtype=dtype, device="npu:0")
    beta = torch.rand(T, nv, dtype=dtype, device="npu:0")
    g = torch.randn(T, nv, dtype=torch.float32, device="npu:0") * -0.1 if has_g else None
    scale = 1.0 / (dk ** 0.5)
    init_state = torch.randn(B, nv, dv, dk, dtype=sdt, device="npu:0")
    asl = torch.tensor([seqlen] * B, dtype=torch.int32, device="npu:0")

    # warmup
    for _ in range(WARMUP):
        torch.ops.cann_ops_transformer.chunk_gated_delta_rule(
            q, k, v, beta=beta, initial_state=init_state,
            actual_seq_lengths=asl, scale=scale, g=g)

    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(REPEAT):
        torch.ops.cann_ops_transformer.chunk_gated_delta_rule(
            q, k, v, beta=beta, initial_state=init_state,
            actual_seq_lengths=asl, scale=scale, g=g)
    torch.npu.synchronize()
    elapsed = (time.time() - t0) / REPEAT * 1000  # ms
    return elapsed

print(f"{'Case':<25} {'Time(ms)':>10}")
print("-" * 37)
results = []
for label, B, s, nk, nv, dk, dv, has_g in CASES:
    t = bench(label, B, s, nk, nv, dk, dv, has_g)
    results.append((label, t))
    print(f"{label:<25} {t:>10.2f}")
    torch.npu.empty_cache()

# save
import csv
with open("output/perf_refactored2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "time_ms"])
    w.writerows(results)
print("\nSaved to output/perf_refactored2.csv")
