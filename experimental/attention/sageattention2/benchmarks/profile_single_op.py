"""Single-op profiling recipe for the SageAttention2 PyTorch path on Atlas 950PR.

取 ./benchmarks/cases.py 的一个 case（默认 P-01），经 sageattn_qk_int8_pv_fp8_asc
（fp32+fp16 交付默认档）在真机上跑 warmup + 受测循环，用 torch_npu.profiler（kineto）
采集算子级耗时，导出 chrome trace 并打印 op 级汇总表。本脚本是 PyTorch 路径的
单算子 profile 配方，与 C++ aclnn 基准工具 ./benchmarks/benchmark_aclnn.cpp 互补，
不导入、不依赖该二进制。

真机用法（算子目录执行）：

    python3 benchmarks/profile_single_op.py --case P-01 --warmup 10 --steps 20 \
        --trace build/profile_P-01.json

msprof CLI 整进程附加（可选；受测循环保持小步数，采集细节以验收环境 msprof 文档为准）：

    msprof --application="python3 benchmarks/profile_single_op.py --case P-01 \
        --warmup 5 --steps 10 --no-trace" --output=build/msprof_p01

无 torch/torch_npu 时打印跳过信息并以退出码 0 结束（CI-off 机器冒烟）；
在 NPU 上为严格模式，任何失败非零退出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
OP_DIR = BENCH_DIR.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from cases import BenchmarkCase, get_case  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile one benchmark case of sageattn_qk_int8_pv_fp8_asc with torch_npu.profiler."
    )
    parser.add_argument("--case", type=str, default="P-01", help="Case name from benchmarks/cases.py (default P-01)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations outside the profiler (default 10)")
    parser.add_argument("--steps", type=int, default=20, help="Profiled iterations (default 20)")
    parser.add_argument("--trace", type=str, default="", help="Chrome trace output path (default build/profile_<case>.json)")
    parser.add_argument("--no-trace", action="store_true", help="Skip chrome trace export (for msprof attach runs)")
    return parser


def build_inputs(case: BenchmarkCase, torch):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[case.dtype]
    q = torch.randn(case.batch, case.heads, case.seq, case.dim, dtype=dtype, device="npu")
    k = torch.randn(case.batch, case.kv_heads, case.seq, case.dim, dtype=dtype, device="npu")
    v = torch.randn(case.batch, case.kv_heads, case.seq, case.dim, dtype=dtype, device="npu")
    return q, k, v


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        import torch
        import torch_npu
    except ImportError as err:
        print(f"[profile_single_op] torch/torch_npu unavailable ({err}); skipping (exit 0).")
        return 0

    if not torch.npu.is_available():
        print("[profile_single_op] torch.npu.is_available() is False (no NPU device attached); skipping (exit 0).")
        return 0

    if str(OP_DIR / "python") not in sys.path:
        sys.path.insert(0, str(OP_DIR / "python"))
    from sageattention2 import sageattn_qk_int8_pv_fp8_asc

    case = get_case(args.case)
    q, k, v = build_inputs(case, torch)

    def call():
        return sageattn_qk_int8_pv_fp8_asc(
            q, k, v, tensor_layout="HND", is_causal=case.is_causal, pv_accum_dtype="fp32+fp16"
        )

    for _ in range(args.warmup):
        call()
    torch.npu.synchronize()

    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]
    with torch_npu.profiler.profile(activities=activities) as prof:
        for _ in range(args.steps):
            call()
        torch.npu.synchronize()

    print(f"[profile_single_op] case={case.name} shape=[{case.batch}, {case.heads}, {case.seq}, {case.dim}] "
          f"kv_heads={case.kv_heads} dtype={case.dtype} causal={case.is_causal} tier={case.tier} "
          f"device={torch.npu.get_device_name()}")
    print(prof.key_averages().table(row_limit=30))

    if not args.no_trace:
        trace = Path(args.trace) if args.trace else OP_DIR / "build" / f"profile_{case.name}.json"
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))
        print(f"[profile_single_op] chrome trace written to {trace}")
    print("[profile_single_op] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
