#!/usr/bin/env python3
"""SageAttention2 高性能基准压测与自动化评测套件 (Standalone Performance Benchmark & Profiling Suite)

100% 独立推导实现，对照任务书 §5.1 / §7 与 todo.md §0.5 / M5.1 ~ M5.4 性能规范：
1. 完整覆盖 P-01 ~ P-07 全部 7 大性能基准测试规格及短序列/Decode 扩展规格；
2. 高精度硬件级计时与算力吞吐（TFLOP/s）模型计算：
   - 非因果注意力量：4 * B * H * Sq * Sk * D
   - 因果注意力量：2 * B * H * Sq * Sk * D
3. 多 Baseline 评测（PyTorch 原生 FP32 SDPA / FlashAttention / Ascend FIA），自动核验 >= 1.6x 性能契约硬门禁；
4. 真实硬件 Profiling 探针集成：
   - CANN Ascend Profiler (torch_npu.profiler) 探针打点与 Trace 导出；
   - CANN msprof 命令行模板生成器 (--dump-msprof-cmd)；
   - 显存带宽利用率 (MBU %) 与算力强度 (Arithmetic Intensity) 计算，验证 Vector-bound 判定。
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# 检查 NPU 支持
try:
    import torch_npu
    HAVE_TORCH_NPU = True
except ImportError:
    HAVE_TORCH_NPU = False

# 检查 FlashAttention 支持
try:
    import flash_attn
    HAVE_FLASH_ATTN = True
except ImportError:
    HAVE_FLASH_ATTN = False

# 导入 SageAttention2 库
_repo_python_path = Path(__file__).resolve().parent.parent / "python"
if _repo_python_path.exists() and str(_repo_python_path) not in sys.path:
    sys.path.insert(0, str(_repo_python_path))

from sageattention2 import sageattn_qk_int8_pv_fp8_asc
from sageattention2._constants import (
    ALL_BENCHMARK_SPECS,
    BENCHMARK_SPECS,
    EXTENDED_CASE_IDS,
    SHORT_CASE_IDS,
    STANDARD_CASE_IDS,
)
from sageattention2._utils import (
    compute_attention_flops,
    compute_geomean,
    compute_memory_traffic_bytes,
    create_qkv,
)

# 任务书 §5.1 / §7 与 todo.md §0.5 规定的基准测试规格矩阵
BENCHMARK_CASES = ALL_BENCHMARK_SPECS



def sync_device(device: torch.device) -> None:
    """跨平台设备同步防线"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and HAVE_TORCH_NPU:
        torch_npu.npu.synchronize(device)


def measure_execution_time_ms(
    fn,
    device: torch.device,
    warmup: int = 25,
    repeats: int = 50,
) -> float:
    """高精度计时引擎：
    在 NPU / CUDA 上优先使用硬件 Event 计时排除 Python host 端启动开销；
    在 CPU 或 Event 不可用时使用 time.perf_counter()。
    """
    # 1. 预热
    for _ in range(warmup):
        fn()
    sync_device(device)

    # 2. 硬件 Event 计时路径
    if device.type == "npu" and HAVE_TORCH_NPU:
        try:
            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)
            sync_device(device)
            start_event.record()
            for _ in range(repeats):
                fn()
            end_event.record()
            sync_device(device)
            return float(start_event.elapsed_time(end_event)) / float(repeats)
        except Exception:  # noqa: BLE001, S110
            pass

    if device.type == "cuda":
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            sync_device(device)
            start_event.record()
            for _ in range(repeats):
                fn()
            end_event.record()
            sync_device(device)
            return float(start_event.elapsed_time(end_event)) / float(repeats)
        except Exception:  # noqa: BLE001, S110
            pass

    # 3. CPU / 通用 host 计时路径
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync_device(device)
    t1 = time.perf_counter()
    return ((t1 - t0) / float(repeats)) * 1000.0


def build_fia_callable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
):
    """Ascend 官方 FusedInferAttentionScore 基准调用器包装"""
    return lambda: torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
        sparse_mode=2 if is_causal else 0,
    )


def build_sdpa_callable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
    force_fp32: bool = False,
):
    """PyTorch 原生 SDPA 调用器包装（支持 GQA 广播与 FP32 强制计算）"""
    hq = q.shape[1]
    hk = k.shape[1]
    if hq != hk:
        group = hq // hk
        k_in = k.repeat_interleave(group, dim=1)
        v_in = v.repeat_interleave(group, dim=1)
    else:
        k_in, v_in = k, v

    if force_fp32:
        return lambda: F.scaled_dot_product_attention(
            q.float(), k_in.float(), v_in.float(), is_causal=is_causal, scale=scale
        ).to(q.dtype)

    return lambda: F.scaled_dot_product_attention(
        q, k_in, v_in, is_causal=is_causal, scale=scale
    )


def build_flash_attn_callable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
):
    """FlashAttention 调用器包装"""
    if HAVE_FLASH_ATTN:
        # FlashAttention expects (B, S, H, D)
        q_bshd = q.transpose(1, 2).contiguous()
        k_bshd = k.transpose(1, 2).contiguous()
        v_bshd = v.transpose(1, 2).contiguous()
        return lambda: flash_attn.flash_attn_func(
            q_bshd, k_bshd, v_bshd, softmax_scale=scale, causal=is_causal
        )
    # Fallback to PyTorch SDPA with flash backend enabled
    return build_sdpa_callable(q, k, v, scale=scale, is_causal=is_causal, force_fp32=False)


def run_benchmark_case(
    case_id: str,
    spec: dict[str, Any],
    device: torch.device,
    baseline_type: str = "auto",
    warmup: int = 25,
    repeats: int = 50,
    peak_bandwidth_gbs: float = 2800.0,
    scale_factor: int = 1,
    qk_quant_gran: str = "per_thread",
    pv_accum_dtype: str = "fp32+fp16",
    profiler_context=None,
) -> dict[str, Any]:
    """单用例基准压测与吞吐评测核心流程"""
    b = spec["batch"]
    hq = spec["hq"]
    hk = spec["hk"]
    sq = max(1, spec["sq"] // scale_factor)
    sk = max(1, spec["sk"] // scale_factor)
    d = spec["d"]
    is_causal = spec["is_causal"]
    dtype = spec["dtype"]
    sm_scale = 1.0 / math.sqrt(d)

    # 1. 生成随机输入张量
    q, k, v = create_qkv(
        b=b, hq=hq, hk=hk, sq=sq, sk=sk, d=d, dtype=dtype, device=device, layout="HND"
    )

    flops = compute_attention_flops(b, hq, sq, sk, d, is_causal)
    traffic_bytes = compute_memory_traffic_bytes(b, hq, hk, sq, sk, d, dtype)
    arithmetic_intensity = flops / float(traffic_bytes) if traffic_bytes > 0 else 0.0

    # 2. 测量 SageAttention2 延迟
    sa2_callable = lambda: sageattn_qk_int8_pv_fp8_asc(
        q, k, v,
        tensor_layout="HND",
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
    )

    if profiler_context is not None:
        with profiler_context:
            sa2_time_ms = measure_execution_time_ms(sa2_callable, device, warmup=2, repeats=2)
    else:
        sa2_time_ms = measure_execution_time_ms(sa2_callable, device, warmup=warmup, repeats=repeats)

    sa2_tflops = (flops / ((sa2_time_ms / 1000.0) * 1e12)) if sa2_time_ms > 0 else 0.0
    effective_bw_gbs = (traffic_bytes / ((sa2_time_ms / 1000.0) * 1e9)) if sa2_time_ms > 0 else 0.0
    mbu_percent = (effective_bw_gbs / peak_bandwidth_gbs * 100.0) if peak_bandwidth_gbs > 0 else 0.0

    # 3. 构建并测量 Baseline 延迟
    baseline_name = baseline_type
    if baseline_type == "auto":
        if device.type == "npu" and HAVE_TORCH_NPU and hasattr(torch_npu, "npu_fused_infer_attention_score"):
            baseline_name = "FIA"
            baseline_fn = build_fia_callable(q, k, v, scale=sm_scale, is_causal=is_causal)
        elif device.type == "cuda" and HAVE_FLASH_ATTN:
            baseline_name = "FlashAttention"
            baseline_fn = build_flash_attn_callable(q, k, v, scale=sm_scale, is_causal=is_causal)
        else:
            baseline_name = "FP32_SDPA"
            baseline_fn = build_sdpa_callable(q, k, v, scale=sm_scale, is_causal=is_causal, force_fp32=True)
    elif baseline_type == "fia":
        baseline_name = "FIA"
        baseline_fn = build_fia_callable(q, k, v, scale=sm_scale, is_causal=is_causal)
    elif baseline_type == "flash_attn":
        baseline_name = "FlashAttention"
        baseline_fn = build_flash_attn_callable(q, k, v, scale=sm_scale, is_causal=is_causal)
    elif baseline_type == "fp32_sdpa":
        baseline_name = "FP32_SDPA"
        baseline_fn = build_sdpa_callable(q, k, v, scale=sm_scale, is_causal=is_causal, force_fp32=True)
    else:  # sdpa (native dtype)
        baseline_name = "Native_SDPA"
        baseline_fn = build_sdpa_callable(q, k, v, scale=sm_scale, is_causal=is_causal, force_fp32=False)

    try:
        baseline_time_ms = measure_execution_time_ms(baseline_fn, device, warmup=warmup, repeats=repeats)
        baseline_tflops = (flops / ((baseline_time_ms / 1000.0) * 1e12)) if baseline_time_ms > 0 else 0.0
        speedup = (baseline_time_ms / sa2_time_ms) if sa2_time_ms > 0 else 0.0
        meets_gate = speedup >= 1.6
        error_msg = None
    except Exception as exc:  # noqa: BLE001
        baseline_time_ms = 0.0
        baseline_tflops = 0.0
        speedup = 0.0
        meets_gate = False
        error_msg = f"Baseline execution error: {exc}"

    return {
        "case_id": case_id,
        "category": spec["category"],
        "desc": spec["desc"],
        "shape": f"[{b},{hq},{sq},{d}]",
        "sq": sq,
        "sk": sk,
        "d": d,
        "dtype": str(dtype).split(".")[-1],
        "is_causal": is_causal,
        "qk_quant_gran": qk_quant_gran,
        "pv_accum_dtype": pv_accum_dtype,
        "flops": flops,
        "traffic_bytes": traffic_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "effective_bw_gbs": effective_bw_gbs,
        "mbu_percent": mbu_percent,
        "sa2_time_ms": sa2_time_ms,
        "sa2_tflops": sa2_tflops,
        "baseline_name": baseline_name,
        "baseline_time_ms": baseline_time_ms,
        "baseline_tflops": baseline_tflops,
        "speedup": speedup,
        "meets_gate": meets_gate,
        "error": error_msg,
    }


def generate_msprof_command_template(cases: list[str], output_dir: str = "./prof_output") -> str:
    """生成 Ascend CANN 真实硬件 profiling 采集命令模板"""
    case_str = ",".join(cases)
    template = f"""# ==============================================================================
# CANN Ascend Profiler (msprof) Hardware Profiling Command Template
# Target Platform: Atlas 950PR / Ascend 910B
# ==============================================================================
msprof --output={output_dir} \\
       --application="python tools/benchmark.py --device npu --cases {case_str} --warmup 10 --repeats 20" \\
       --ai-core=on \\
       --aicpu=on \\
       --l2-cache=on \\
       --memory-bandwidth=on \\
       --model-execution=on \\
       --task-time=on

# 解析生成的时间线与算子指标报告：
# 1. 文本分析：msprof-analyze -d {output_dir}
# 2. Chrome Trace 可视化：打开 chrome://tracing 并加载 {output_dir}/*/trace.json
# ==============================================================================
"""
    return template


def parse_args():
    parser = argparse.ArgumentParser(
        description="SageAttention2 Standalone Performance Benchmark & Profiling Suite (P-01 to P-07 against Baseline/FIA)"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="standard",
        help="Case IDs to run: 'standard' (P-01~P-07), 'all' (inc. short), 'short', or comma-separated list (e.g. P-01,P-02)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu",
        help="Target device: 'npu', 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="auto",
        choices=["auto", "fia", "fp32_sdpa", "sdpa", "flash_attn"],
        help="Baseline comparison target: 'auto' (NPU->FIA, CUDA->FlashAttn, CPU->FP32 SDPA), 'fp32_sdpa', 'sdpa', 'fia', 'flash_attn'",
    )
    parser.add_argument(
        "--metric-policy",
        type=str,
        default="auto",
        choices=["auto", "frozen", "test_perf", "both"],
        help="Timing iteration policy: 'frozen' (warmup=25, repeats=50 as per §0.5), 'test_perf' (warmup=5, repeats=20 as per test_perf.py), 'both' (R-6 双口径补录: run frozen then test_perf and print delta appendix), or 'auto' (respects SAGEATTN_PERF_WARMUP/ITERS, default frozen)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Number of warmup iterations (default: 25 as per todo.md §0.5)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of timing iterations (default: 50 as per todo.md §0.5)",
    )
    parser.add_argument(
        "--qk-quant-gran",
        type=str,
        default="per_thread",
        choices=["per_thread", "per_warp"],
        help="QK quantization granularity (default: per_thread)",
    )
    parser.add_argument(
        "--pv-accum-dtype",
        type=str,
        default="fp32+fp16",
        choices=["fp32+fp16", "fp32+fp32", "fp32"],
        help="PV accumulation dtype (default: fp32+fp16)",
    )
    parser.add_argument(
        "--peak-bandwidth-gbs",
        type=float,
        default=2800.0,
        help="Target hardware peak HBM bandwidth in GB/s for MBU calculation (default: 2800.0 for Atlas 950PR)",
    )
    parser.add_argument(
        "--scale-down",
        type=int,
        default=1,
        help="Downscale factor for sequence lengths (useful for fast smoke testing / CPU verification)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run rapid smoke test (scale_down=8, warmup=2, repeats=3)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch/CANN Profiler probe to capture trace timelines",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./prof_output",
        help="Output directory to save trace timeline files",
    )
    parser.add_argument(
        "--dump-msprof-cmd",
        action="store_true",
        help="Print CANN msprof hardware profiling command template and exit",
    )
    parser.add_argument(
        "--check-gate",
        action="store_true",
        help="Exit with error code 1 if any case or geometric mean fails the >=1.6x gate",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to dump structured benchmark results as JSON",
    )
    return parser.parse_args()


def execute_round(
    selected_case_ids: list[str],
    args: argparse.Namespace,
    device: torch.device,
    warmup: int,
    repeats: int,
    scale_factor: int,
    profiler_context: Any,
    tag: str,
) -> dict[str, Any]:
    """Run one full protocol round (single timing policy) and print its report."""
    results: list[dict[str, Any]] = []
    print("\nBenchmark Execution Progress:")
    for case_id in selected_case_ids:
        spec = BENCHMARK_CASES[case_id]
        print(f"  * Running {case_id:<12} ({spec['desc']})... ", end="", flush=True)
        try:
            res = run_benchmark_case(
                case_id=case_id,
                spec=spec,
                device=device,
                baseline_type=args.baseline,
                warmup=warmup,
                repeats=repeats,
                peak_bandwidth_gbs=args.peak_bandwidth_gbs,
                scale_factor=scale_factor,
                qk_quant_gran=args.qk_quant_gran,
                pv_accum_dtype=args.pv_accum_dtype,
                profiler_context=profiler_context,
            )
            results.append(res)
            gate_tag = "[PASS >= 1.6x]" if res["meets_gate"] else "[BELOW 1.6x]"
            if res.get("error"):
                print(f"FAILED (Baseline error: {res['error']})")
            else:
                print(
                    f"Done. SA2: {res['sa2_time_ms']:.2f} ms ({res['sa2_tflops']:.1f} TFLOPS, "
                    f"BW: {res['effective_bw_gbs']:.1f} GB/s, MBU: {res['mbu_percent']:.1f}%) | "
                    f"{res['baseline_name']}: {res['baseline_time_ms']:.2f} ms ({res['baseline_tflops']:.1f} TFLOPS) | "
                    f"Speedup: {res['speedup']:.2f}x {gate_tag}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED (Error: {exc})")
            results.append({
                "case_id": case_id,
                "category": spec["category"],
                "desc": spec["desc"],
                "error": str(exc),
                "meets_gate": False,
                "speedup": 0.0,
            })

    speedup_values = [r["speedup"] for r in results if r.get("speedup", 0.0) > 0 and not r.get("error")]
    geomean_speedup = compute_geomean(speedup_values)
    all_individual_pass = all(r.get("meets_gate", False) for r in results)
    overall_gate_pass = all_individual_pass and (geomean_speedup >= 1.6)

    print("\n" + "=" * 80)
    print(f"   SageAttention2 Performance Benchmark & Profiling Report [{tag}]")
    print("=" * 80)
    for r in results:
        case_id = r["case_id"]
        if r.get("error"):
            print(f"- Case {case_id}: FAILED ({r['error']})")
        else:
            status = "PASS" if r["meets_gate"] else "FAIL"
            print(f"- Case {case_id} ({r['desc']}):")
            print(f"  - Shape: {r['shape']} (D={r['d']}, Causal={r['is_causal']}, Dtype={r['dtype']})")
            print(f"  - SageAttention2 Latency: {r['sa2_time_ms']:.3f} ms")
            print(f"  - SageAttention2 Throughput: {r['sa2_tflops']:.2f} TFLOP/s")
            print(f"  - Effective Memory Bandwidth: {r['effective_bw_gbs']:.2f} GB/s (MBU: {r['mbu_percent']:.1f}%)")
            print(f"  - Arithmetic Intensity: {r['arithmetic_intensity']:.2f} FLOP/Byte (Vector-bound: {'YES' if r['arithmetic_intensity'] < 100 else 'NO'})")
            print(f"  - Baseline ({r['baseline_name']}) Latency: {r['baseline_time_ms']:.3f} ms ({r['baseline_tflops']:.2f} TFLOP/s)")
            print(f"  - Acceleration Speedup: {r['speedup']:.2f}x (Gate >= 1.6x: {status})")

    print("\n- Overall Performance Gate Summary:")
    print(f"  - Tested Cases Count: {len(results)}")
    print(f"  - Geometric Mean Speedup: {geomean_speedup:.2f}x")
    print(f"  - Individual Gate Compliance: {'ALL PASS' if all_individual_pass else 'SOME FAILED'}")
    print(f"  - Performance Contract Gate Status: {'PASSED (>= 1.6x)' if overall_gate_pass else 'FAILED (< 1.6x)'}")
    print("=" * 80)

    return {
        "tag": tag,
        "warmup": warmup,
        "repeats": repeats,
        "results": results,
        "geomean_speedup": geomean_speedup,
        "all_individual_pass": all_individual_pass,
        "overall_gate_pass": overall_gate_pass,
    }


def print_dual_policy_appendix(primary: dict[str, Any], secondary: dict[str, Any]) -> None:
    """R-6 双口径补录：门禁以 primary (frozen 25/50) 判定，secondary (test_perf 5/20) 为配套脚本口径附注。"""
    print("\n" + "=" * 80)
    print("   Dual-Policy Appendix (R-6 计时口径差值附注)")
    print(f"   Primary (门禁口径): {primary['tag']} | Secondary (配套补录): {secondary['tag']}")
    print("=" * 80)
    sec_by_case = {r["case_id"]: r for r in secondary["results"]}
    for r in primary["results"]:
        s = sec_by_case.get(r["case_id"])
        if s is None or r.get("error") or s.get("error") or r.get("sa2_time_ms", 0.0) <= 0:
            print(f"- Case {r['case_id']}: appendix skipped (error or missing round)")
            continue
        delta_pct = 100.0 * (s["sa2_time_ms"] - r["sa2_time_ms"]) / r["sa2_time_ms"]
        print(
            f"- Case {r['case_id']}: SA2 time {r['sa2_time_ms']:.3f} -> {s['sa2_time_ms']:.3f} ms "
            f"(Δ {delta_pct:+.1f}%) | speedup {r['speedup']:.2f}x -> {s['speedup']:.2f}x"
        )
    print(
        f"- Geomean speedup: {primary['geomean_speedup']:.2f}x [{primary['tag']}] "
        f"vs {secondary['geomean_speedup']:.2f}x [{secondary['tag']}]"
    )
    print("- 注: 门禁判定以 Primary 口径为准; 两口径差值供自验证报告附注引用 (todo.md R-6)。")


def main() -> int:
    args = parse_args()

    # 处理选定的 case
    if args.cases == "standard":
        selected_case_ids = STANDARD_CASE_IDS
    elif args.cases == "all":
        selected_case_ids = list(BENCHMARK_CASES.keys())
    elif args.cases == "short":
        selected_case_ids = SHORT_CASE_IDS
    else:
        selected_case_ids = []
        for c in args.cases.split(","):
            c = c.strip()
            if c in BENCHMARK_CASES:
                selected_case_ids.append(c)
            else:
                print(f"[WARNING] Unknown case ID '{c}', skipping.")

    if not selected_case_ids:
        print("[ERROR] No valid benchmark cases selected.")
        return 1

    # 导出 msprof 命令模板模式
    if args.dump_msprof_cmd:
        print(generate_msprof_command_template(selected_case_ids, args.profile_dir))
        return 0

    scale_factor = args.scale_down
    default_warmup = 5 if args.metric_policy == "test_perf" else 25
    default_repeats = 20 if args.metric_policy == "test_perf" else 50
    env_warmup = os.environ.get("SAGEATTN_PERF_WARMUP")
    env_repeats = os.environ.get("SAGEATTN_PERF_ITERS")

    warmup = int(env_warmup) if env_warmup is not None else (args.warmup if args.warmup != 25 else default_warmup)
    repeats = int(env_repeats) if env_repeats is not None else (args.repeats if args.repeats != 50 else default_repeats)

    if args.quick:
        scale_factor = max(scale_factor, 8)
        warmup = 2
        repeats = 3

    policy_tag = "test_perf (5/20)" if (warmup == 5 and repeats == 20) else ("frozen (25/50)" if (warmup == 25 and repeats == 50) else "custom")
    device = torch.device(args.device)
    print("=" * 80)
    print("   SageAttention2 Performance Benchmark & Profiling Suite")
    print(f"   Target Device: {device} | Baseline: {args.baseline} | Scale Factor: {scale_factor}x")
    print(f"   Granularity: {args.qk_quant_gran} | Accumulator: {args.pv_accum_dtype}")
    print(f"   Timing Policy: {policy_tag} (Warmup={warmup}, Repeats={repeats}) | Peak HBM: {args.peak_bandwidth_gbs:.1f} GB/s")
    if args.quick:
        print("   [MODE] Quick smoke verification mode enabled.")
    print("=" * 80)

    # Profiler 探针配置
    profiler_context = None
    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        if device.type == "npu" and HAVE_TORCH_NPU:
            try:
                experimental_config = torch_npu.profiler._ExperimentalConfig(
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    l2_cache=True,
                )
                profiler_context = torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    experimental_config=experimental_config,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile_dir),
                )
                print(f"[PROFILER] CANN Ascend Profiler enabled, traces will be saved to {args.profile_dir}")
            except Exception as e:  # noqa: BLE001
                print(f"[PROFILER] Warning: Failed to initialize CANN profiler: {e}, falling back to standard profiler")
        if profiler_context is None:
            profiler_context = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU] + (
                    [torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
            )
            print(f"[PROFILER] PyTorch Profiler enabled, traces will be saved to {args.profile_dir}")

    protocols: list[tuple[str, int, int]]
    if args.metric_policy == "both":
        if args.quick:
            print("[WARN] --quick overrides dual policy; falling back to single custom protocol.")
            protocols = [(f"quick (warmup={warmup}, repeats={repeats})", warmup, repeats)]
        else:
            protocols = [("frozen (25/50)", 25, 50), ("test_perf (5/20)", 5, 20)]
            if args.profile:
                print("[WARN] --metric-policy both: profiler trace captured for the primary protocol only.")
    else:
        protocols = [(policy_tag, warmup, repeats)]

    rounds: list[dict[str, Any]] = []
    for idx, (tag, w, r) in enumerate(protocols):
        if len(protocols) > 1:
            print(f"\n{'#' * 80}")
            print(f"#   Protocol Round {idx + 1}/{len(protocols)}: {tag} (Warmup={w}, Repeats={r})")
            print(f"{'#' * 80}")
        ctx = profiler_context if idx == 0 else None
        rounds.append(execute_round(selected_case_ids, args, device, w, r, scale_factor, ctx, tag))

    if len(rounds) > 1:
        print_dual_policy_appendix(rounds[0], rounds[1])

    # 门禁判定以 primary（frozen / 指定口径）为准；secondary 仅差值附注（R-6）
    primary = rounds[0]
    overall_gate_pass = primary["overall_gate_pass"]

    # 导出 JSON 报告
    if args.output_json:
        report_data = {
            "metadata": {
                "device": str(device),
                "baseline_type": args.baseline,
                "scale_factor": scale_factor,
                "warmup": primary["warmup"],
                "repeats": primary["repeats"],
                "qk_quant_gran": args.qk_quant_gran,
                "pv_accum_dtype": args.pv_accum_dtype,
                "peak_bandwidth_gbs": args.peak_bandwidth_gbs,
                "geomean_speedup": round(primary["geomean_speedup"], 3),
                "all_individual_pass": primary["all_individual_pass"],
                "overall_gate_pass": overall_gate_pass,
                "policy_rounds": [
                    {
                        "tag": rd["tag"],
                        "warmup": rd["warmup"],
                        "repeats": rd["repeats"],
                        "geomean_speedup": round(rd["geomean_speedup"], 3),
                        "overall_gate_pass": rd["overall_gate_pass"],
                    }
                    for rd in rounds
                ],
            },
            "cases": primary["results"],
        }
        if len(rounds) > 1:
            report_data["secondary_cases"] = rounds[1]["results"]
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Benchmark results successfully exported to {output_path.resolve()}")

    if args.check_gate and not overall_gate_pass:
        print("\n[GATE ERROR] Performance contract hard gate verification failed (< 1.6x).")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
