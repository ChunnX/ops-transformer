"""PyTorch-path performance validation: sageattn_qk_int8_pv_fp8_asc vs FIA baseline.

Times the delivered PyTorch API against the FIA baseline on Atlas 950PR for the
required cases P-01..P-07 (task doc 基准用例；唯一用例表 ./benchmarks/cases.py，
本脚本不复制第二份)，逐 case 判 Speedup >= 1.6 门禁并给出几何平均
（SageAttention2_task_doc.md 性能要求：PyTorch 路径硬性指标）。

计时口径（task/06_benchmark.md）：torch.npu.Event 事件对逐次预建，warmup 不计入
统计，循环后 synchronize 再取逐迭代 ms 均值。

FIA 基线调用配方固化自任务书「最小调用示例」与 task/06_benchmark.md「FIA 基线
采集」：input_layout="BNSD"（与本算子 HND 同构，双方共享同一物理张量，无转置拷贝）、
pre_tokens=65535、next_tokens=65535、scale=1/sqrt(D)；因果 case P-03 叠加官方
perf harness 因果配方 sparse_mode=2 + 2048x2048 triu bool 掩码（task/06_benchmark.md
因果配方条目，与 sparse_mode=3 配方语义等价）。

无 torch/torch_npu 或无 NPU 设备时打印明确跳过信息并以退出码 0 结束（CI-off
机器冒烟）；在 NPU 上则严格模式，任何失败直接抛出。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
OP_DIR = BENCH_DIR.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from cases import CASES, BenchmarkCase, get_case  # noqa: E402

REQUIRED_CASES: List[BenchmarkCase] = [get_case(entry[0]) for entry in CASES]

SPEEDUP_GATE = 1.6
WARMUP_FLOOR = 10
REPEATS_FLOOR = 20

SAGE_API = "sageattention2.sageattn_qk_int8_pv_fp8_asc"
FIA_API = "torch_npu.npu_fused_infer_attention_score"
FIA_CAUSAL_RECIPE = "sparse_mode=2 + 2048x2048 triu bool mask (task/06_benchmark.md 因果配方, P-03 only)"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyTorch-path performance gate: sageattn_qk_int8_pv_fp8_asc vs FIA (P-01..P-07, >=1.6x)."
    )
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations, excluded from stats (>=10)")
    parser.add_argument("--repeats", type=int, default=100, help="Measured iterations (>=20)")
    parser.add_argument("--gate", type=float, default=SPEEDUP_GATE, help="Per-case speedup gate threshold")
    parser.add_argument(
        "--l2-flush-mb", type=int, default=0,
        help="L2 flush buffer (MB) zeroed before each measured iteration; 0=off here "
             "(task/06_benchmark.md protocol default: 256MB; pass --l2-flush-mb 256 "
             "for a protocol-conform run)",
    )
    parser.add_argument("--output", default="pyperf_report.json", help="Output JSON report path")
    return parser


def probe_npu_environment() -> bool:
    """Return True only when torch + torch_npu + a real NPU device are usable."""
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as err:  # absent or broken torch_npu on CI-off machines
        print(f"[SKIP] torch/torch_npu unavailable ({err}); "
              f"PyTorch-path performance harness requires a real Atlas 950PR environment.")
        return False
    if not torch.npu.is_available():
        print("[SKIP] torch.npu.is_available() is False (no NPU device attached); skipping (exit 0).")
        return False
    return True


def time_mean_ms(fn: Callable[[], object], warmup: int, repeats: int, flush_buf: Optional[object]) -> float:
    """Warmup excluded, per-iteration pre-created Event pairs, sync before stats, mean ms."""
    import torch

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
    stops = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        if flush_buf is not None:
            flush_buf.zero_()
        starts[i].record()
        fn()
        stops[i].record()
    torch.npu.synchronize()

    return statistics.fmean(s.elapsed_time(e) for s, e in zip(starts, stops))


def run_case(case: BenchmarkCase, warmup: int, repeats: int, l2_flush_mb: int, gate: float) -> Dict[str, object]:
    import torch
    import torch_npu
    from sageattention2 import sageattn_qk_int8_pv_fp8_asc

    dtype = torch.bfloat16 if case.dtype == "bf16" else torch.float16
    # HND [B, Hq, Sq, D]；FIA 以 input_layout="BNSD" 接收同构 [B, N, S, D]，
    # 双方直接共享同一物理张量（task/06_benchmark.md 布局换算口径）
    q = torch.randn(case.batch, case.heads, case.seq, case.dim, dtype=dtype, device="npu")
    k = torch.randn(case.batch, case.kv_heads, case.seq, case.dim, dtype=dtype, device="npu")
    v = torch.randn(case.batch, case.kv_heads, case.seq, case.dim, dtype=dtype, device="npu")
    scale = 1.0 / math.sqrt(case.dim)

    sparse_mode = 0
    atten_mask = None
    if case.is_causal:
        sparse_mode = 2
        atten_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device="npu"), diagonal=1)

    def run_fia() -> object:
        return torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=case.heads,
            num_key_value_heads=case.kv_heads,
            input_layout="BNSD",
            scale=scale,
            pre_tokens=65535,
            next_tokens=65535,
            sparse_mode=sparse_mode,
            atten_mask=atten_mask,
        )[0]

    def run_sage() -> object:
        return sageattn_qk_int8_pv_fp8_asc(
            q, k, v,
            tensor_layout="HND",
            is_causal=case.is_causal,
            sm_scale=scale,
            pv_accum_dtype="fp32+fp16",
        )

    flush_buf = None
    if l2_flush_mb > 0:
        flush_buf = torch.empty(l2_flush_mb * 1024 * 1024 // 2, dtype=torch.float16, device="npu")

    t_fia = time_mean_ms(run_fia, warmup, repeats, flush_buf)
    t_sage = time_mean_ms(run_sage, warmup, repeats, flush_buf)
    speedup = t_fia / t_sage

    del q, k, v, atten_mask, flush_buf
    torch.npu.empty_cache()

    return {
        "case_id": case.name,
        "tier": case.tier,
        "shape": [case.batch, case.heads, case.seq, case.dim],
        "kv_heads": case.kv_heads,
        "dtype": case.dtype,
        "is_causal": case.is_causal,
        "t_fia_ms": t_fia,
        "t_sage_ms": t_sage,
        "speedup": speedup,
        "gate": gate,
        "passed": speedup >= gate,
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.warmup < WARMUP_FLOOR or args.repeats < REPEATS_FLOOR:
        raise SystemExit(
            f"warmup must be >={WARMUP_FLOOR} and repeats >={REPEATS_FLOOR} "
            f"(got warmup={args.warmup}, repeats={args.repeats})"
        )

    if not probe_npu_environment():
        return 0

    python_pkg_dir = str(OP_DIR / "python")
    if python_pkg_dir not in sys.path:
        sys.path.insert(0, python_pkg_dir)

    import torch
    import torch_npu

    case_reports: List[Dict[str, object]] = []
    for case in REQUIRED_CASES:
        report = run_case(case, args.warmup, args.repeats, args.l2_flush_mb, args.gate)
        verdict = "PASS" if report["passed"] else "FAIL"
        print(
            f"[{case.name}] T_FIA={report['t_fia_ms']:.4f} ms | T_sage={report['t_sage_ms']:.4f} ms | "
            f"Speedup={report['speedup']:.3f}x | gate>={args.gate}: {verdict}"
        )
        case_reports.append(report)

    speedups = [float(r["speedup"]) for r in case_reports]
    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    all_pass = all(bool(r["passed"]) for r in case_reports)
    print(
        f"[geomean] Speedup={geomean:.3f}x over {len(case_reports)} cases | "
        f"gate>={args.gate}: {'PASS' if all_pass else 'FAIL'}"
    )

    report_doc = {
        "path": "PyTorch (torch_npu)",
        "sage_api": SAGE_API,
        "fia_api": FIA_API,
        "gate": args.gate,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "l2_flush_mb": args.l2_flush_mb,
        "fia_causal_recipe": FIA_CAUSAL_RECIPE,
        "meta": {
            "torch_version": torch.__version__,
            "torch_npu_version": torch_npu.__version__,
            "device": torch.npu.get_device_name(),
        },
        "cases": case_reports,
        "geomean_speedup": geomean,
        "all_cases_pass": all_pass,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[REPORT] JSON written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
