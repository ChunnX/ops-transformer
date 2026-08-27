"""ps%8==0 主路径展开性能探针: 固定配置多次迭代取中位耗时。

用法: python _perf_probe.py [label]
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch_npu
from _defect_case_runner import build_case_inputs
from cann_ops_transformer import pool_key_indexer

torch_npu.npu.set_device(0)
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"

CONFIGS = [
    # (name, ps, topk, q_len, s2pools, mask)
    ("ps16_topk2048", 16, 2048, 64, 512, 3),
    ("ps64_topk2048", 64, 2048, 32, 128, 3),
    ("ps128_topk8192", 128, 8192, 16, 128, 3),
]

ITERS = 30
WARMUP = 5

for name, ps, topk, s1, s2p, mask in CONFIGS:
    cfg = dict(id=f"perf_{name}", layout_q="BSND", layout_k="BSND",
               dtype=torch.float16, q_lens=[s1], k_pools=[s2p], n1=8,
               topk=topk, pool_size=ps, mask_mode=mask, return_value=True,
               tail=[ps - 1], pa_block_size=16)
    inputs = build_case_inputs(cfg)
    dev = "npu:0"
    q = inputs["query"].to(dev); k = inputs["pool_key"].to(dev)
    w = inputs["weights"].to(dev); tk = inputs["pool_tail_k"].to(dev)

    def call():
        return pool_key_indexer(q, k, w, tk, topk=topk, pool_size=ps,
                                mask_mode=mask, return_value=True)

    for _ in range(WARMUP):
        call()
    torch.npu.synchronize()
    times = []
    for _ in range(ITERS):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end))
    med = statistics.median(times)
    print(f"[perf {LABEL}] {name}: median={med:.3f} ms  min={min(times):.3f}  (n={ITERS})")
