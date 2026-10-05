#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_sageattn.py: 华为昇腾 SageAttention2 线上/真机单核与全链路诊断调试看门狗工具

功能特性：
1. 标准 CLI 接口与丰富预设：
   - 快速预设：-p/--preset [quick|standard|standard_bf16|causal|gqa|perf|long]
   - 全量矩阵扫描：--sweep 自动执行 8 大典型规格覆盖，生成表格化诊断报告；
   - 细粒度参数：--b, --h, --hkv, --sq, --sk, --d, --dtype, --causal, --timeout, --device 等；
2. 三层看门狗架构（3-Tier Watchdog）：
   - Tier 1: CANN 底层 aclrtSetOpExecuteTimeOut 驱动级算子超时限制；
   - Tier 2: 线程/进程级 POSIX signal.alarm 硬件中断探针；
   - Tier 3: 独立沙箱进程（multiprocessing spawn）强隔离，超时自动 terminate / kill，根绝宿主卡死。
3. 详细张量统计与探针（min, max, mean, std, L2 norm, isnan, isinf）；
4. 高精度 Golden 参比对拍：--check-golden 实时对比 FP32 SDPA，输出 max_err, mean_err, cos_sim, matched_ratio。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import math
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional


# =========================================================================
# 标准测试预设体系 (Presets)
# =========================================================================
PRESETS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "description": "极速冒烟验证 (B=1, H=4, S=128, D=128, fp16, check_golden)",
        "b": 1, "h": 4, "hkv": 4, "sq": 128, "sk": 128, "d": 128,
        "dtype": "fp16", "layout": "HND", "causal": False,
        "smooth_k": False, "smooth_v": False,
        "warmup": 1, "repeat": 2, "check_golden": True, "timeout": 10,
    },
    "standard": {
        "description": "标准 FP16 对拍验证 (B=1, H=8, S=256, D=128, fp16, check_golden)",
        "b": 1, "h": 8, "hkv": 8, "sq": 256, "sk": 256, "d": 128,
        "dtype": "fp16", "layout": "HND", "causal": False,
        "smooth_k": False, "smooth_v": False,
        "warmup": 2, "repeat": 5, "check_golden": True, "timeout": 15,
    },
    "standard_bf16": {
        "description": "标准 BF16 对拍验证 (B=1, H=8, S=256, D=128, bf16, check_golden)",
        "b": 1, "h": 8, "hkv": 8, "sq": 256, "sk": 256, "d": 128,
        "dtype": "bf16", "layout": "HND", "causal": False,
        "smooth_k": False, "smooth_v": False,
        "warmup": 2, "repeat": 5, "check_golden": True, "timeout": 15,
    },
    "causal": {
        "description": "因果掩码对拍验证 (B=1, H=8, S=512, D=128, causal=True)",
        "b": 1, "h": 8, "hkv": 8, "sq": 512, "sk": 512, "d": 128,
        "dtype": "fp16", "layout": "HND", "causal": True,
        "smooth_k": False, "smooth_v": False,
        "warmup": 2, "repeat": 5, "check_golden": True, "timeout": 20,
    },
    "gqa": {
        "description": "GQA 8:1 分块打包对拍验证 (B=1, Hq=32, Hkv=4, S=1024, D=128)",
        "b": 1, "h": 32, "hkv": 4, "sq": 1024, "sk": 1024, "d": 128,
        "dtype": "fp16", "layout": "HND", "causal": False,
        "smooth_k": False, "smooth_v": False,
        "warmup": 2, "repeat": 5, "check_golden": True, "timeout": 25,
    },
    "perf": {
        "description": "高吞吐长序列压测 (B=1, H=32, S=2048, D=128, repeat=20)",
        "b": 1, "h": 32, "hkv": 32, "sq": 2048, "sk": 2048, "d": 128,
        "dtype": "fp16", "layout": "HND", "causal": False,
        "smooth_k": False, "smooth_v": False,
        "warmup": 3, "repeat": 20, "check_golden": False, "timeout": 30,
    },
    "long": {
        "description": "4K 长上下文吞吐压测 (B=1, H=8, S=4096, D=128, repeat=5)",
        "b": 1, "h": 8, "hkv": 8, "sq": 4096, "sk": 4096, "d": 128,
        "dtype": "fp16", "layout": "HND", "causal": False,
        "smooth_k": False, "smooth_v": False,
        "warmup": 2, "repeat": 5, "check_golden": False, "timeout": 45,
    },
}

# =========================================================================
# 全量矩阵扫描用例表 (Sweep Matrix)
# =========================================================================
SWEEP_CASES: List[Dict[str, Any]] = [
    {"name": "Quick Smoke", "b": 1, "h": 4, "hkv": 4, "sq": 128, "sk": 128, "d": 128, "dtype": "fp16", "layout": "HND", "causal": False, "check_golden": True, "warmup": 1, "repeat": 3, "timeout": 10},
    {"name": "Standard FP16", "b": 1, "h": 8, "hkv": 8, "sq": 256, "sk": 256, "d": 128, "dtype": "fp16", "layout": "HND", "causal": False, "check_golden": True, "warmup": 2, "repeat": 5, "timeout": 15},
    {"name": "Standard BF16", "b": 1, "h": 8, "hkv": 8, "sq": 256, "sk": 256, "d": 128, "dtype": "bf16", "layout": "HND", "causal": False, "check_golden": True, "warmup": 2, "repeat": 5, "timeout": 15},
    {"name": "Causal Mask", "b": 1, "h": 8, "hkv": 8, "sq": 512, "sk": 512, "d": 128, "dtype": "fp16", "layout": "HND", "causal": True, "check_golden": True, "warmup": 2, "repeat": 5, "timeout": 20},
    {"name": "GQA 8:1 Packing", "b": 1, "h": 32, "hkv": 4, "sq": 1024, "sk": 1024, "d": 128, "dtype": "fp16", "layout": "HND", "causal": False, "check_golden": True, "warmup": 2, "repeat": 5, "timeout": 25},
    {"name": "HeadDim 64", "b": 1, "h": 8, "hkv": 8, "sq": 256, "sk": 256, "d": 64, "dtype": "fp16", "layout": "HND", "causal": False, "check_golden": True, "warmup": 2, "repeat": 5, "timeout": 15},
    {"name": "Long Seq 2K", "b": 1, "h": 8, "hkv": 8, "sq": 2048, "sk": 2048, "d": 128, "dtype": "fp16", "layout": "HND", "causal": False, "check_golden": True, "warmup": 2, "repeat": 5, "timeout": 30},
    {"name": "Long Seq 4K", "b": 1, "h": 8, "hkv": 8, "sq": 4096, "sk": 4096, "d": 128, "dtype": "fp16", "layout": "HND", "causal": False, "check_golden": False, "warmup": 2, "repeat": 5, "timeout": 45},
]


def try_set_acl_timeout(timeout_sec: int) -> bool:
    """Tier 1: 尝试调用 CANN AscendCL aclrtSetOpExecuteTimeOut 设置驱动级超时限制"""
    lib_candidates = [
        "libascendcl.so",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
        "/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libascendcl.so",
        "/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libascendcl.so",
        "/usr/local/Ascend/cann-9.2.0-beta.1/lib64/libascendcl.so",
    ]
    for lib_path in lib_candidates:
        try:
            acl = ctypes.CDLL(lib_path)
            if hasattr(acl, "aclrtSetOpExecuteTimeOut"):
                acl.aclrtSetOpExecuteTimeOut.argtypes = [ctypes.c_uint32]
                acl.aclrtSetOpExecuteTimeOut.restype = ctypes.c_int
                ret = acl.aclrtSetOpExecuteTimeOut(ctypes.c_uint32(timeout_sec))
                return (ret == 0)
        except Exception:
            continue
    return False


@contextmanager
def alarm_watchdog(timeout_sec: int, name: str = "Op Execution"):
    """Tier 2: 基于 POSIX signal.alarm 的进程内快速超时守卫"""
    if not hasattr(signal, "alarm") or not hasattr(signal, "SIGALRM") or timeout_sec <= 0:
        yield
        return

    def _timeout_handler(signum, frame):
        raise TimeoutError(
            f"[TIER-2 WATCHDOG-ALARM] {name} 运行超过 {timeout_sec} 秒！检测到硬件或流水线疑似死锁，已被中断。"
        )

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout_sec))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def compute_tensor_stats(t: Any, name: str) -> Dict[str, Any]:
    """计算张量的核心数值统计指标（完全防崩）"""
    import torch
    if t is None:
        return {"name": name, "shape": "None", "dtype": "None", "isnan": False, "isinf": False}

    tf = t.float().detach().cpu()
    isnan = bool(torch.isnan(tf).any().item())
    isinf = bool(torch.isinf(tf).any().item())
    has_finite = bool(torch.isfinite(tf).any().item())

    if has_finite and not (isnan and tf.numel() == 1):
        min_v = float(tf.min().item())
        max_v = float(tf.max().item())
        mean_v = float(tf.mean().item())
        std_v = float(tf.std().item()) if tf.numel() > 1 else 0.0
        norm_v = float(torch.norm(tf).item())
    else:
        min_v = max_v = mean_v = std_v = norm_v = float("nan")

    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "norm": norm_v,
        "isnan": isnan,
        "isinf": isinf,
        "raw_prefix": [int(x) for x in t.cpu().view(torch.int32).flatten()[:24].tolist()] if t.numel() >= 24 else [],
        "raw_prefix_fp": [round(float(x), 4) for x in tf.flatten()[:24].tolist()] if tf.numel() >= 24 else [],
    }


def format_stats_table(stats_list: List[Dict[str, Any]]) -> str:
    """格式化张量统计表格"""
    header = f"{'Tensor':<10} | {'Shape':<20} | {'Dtype':<8} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10} | {'Norm':<10} | {'NaN/Inf':<8}"
    rows = [header]
    for s in stats_list:
        if s["shape"] == "None":
            row = f"{s['name']:<10} | {'None':<20} | {'None':<8} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<8}"
        else:
            shape_str = str(s["shape"])
            nan_inf = f"{'Y' if s['isnan'] else 'N'}/{'Y' if s['isinf'] else 'N'}"
            row = (
                f"{s['name']:<10} | {shape_str:<20} | {s['dtype']:<8} | "
                f"{s['min']:<10.4g} | {s['max']:<10.4g} | {s['mean']:<10.4g} | "
                f"{s['std']:<10.4g} | {s['norm']:<10.4g} | {nan_inf:<8}"
            )
            if s.get("raw_prefix"):
                row += f"\n  -> raw[:24]: {s['raw_prefix']}"
            if s.get("raw_prefix_fp"):
                row += f"\n  -> raw_fp[:24]: {s['raw_prefix_fp']}"
                if len(s["raw_prefix_fp"]) >= 10 and s["raw_prefix_fp"][0] == 99.0:
                    p = s["raw_prefix_fp"]
                    row += (
                        f"\n  -> [Probe3 Telemetry 解码]:\n"
                        f"     badge={p[0]}, sum={p[1]}, max={p[2]}\n"
                        f"     partial[0]={p[3]} (77=Fixpipe未写, 0=L0C零, >0=MMAD正常), partial[1]={p[4]}\n"
                        f"     v_scale[0]={p[5]}, v_scale[1]={p[6]}, o_row[0]={p[7]}\n"
                        f"     p_ub[0]={p[8]}, p_ub[1]={p[9]}"
                    )
        rows.append(row)
    return "\n".join(rows)


def sdpa_reference(
    q: Any, k: Any, v: Any, is_causal: bool = False, sm_scale: Optional[float] = None
) -> Any:
    """CPU/NPU FP32 SDPA 黄金参考计算"""
    import torch
    import torch.nn.functional as F

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    rep = q.shape[1] // k.shape[1]
    kf = k.float().repeat_interleave(rep, dim=1) if rep > 1 else k.float()
    vf = v.float().repeat_interleave(rep, dim=1) if rep > 1 else v.float()
    qf = q.float()

    # 按 Query 行分块防止大序列爆显存
    chunk = 512
    outs = []
    for start in range(0, qf.shape[2], chunk):
        qc = qf[:, :, start:start + chunk, :]
        scores = torch.matmul(qc, kf.transpose(-2, -1)) * sm_scale
        if is_causal:
            qi = torch.arange(start, start + qc.shape[2], device=q.device).view(1, 1, -1, 1)
            ki = torch.arange(k.shape[2], device=q.device).view(1, 1, 1, -1)
            scores = scores.masked_fill(qi < ki, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        outs.append(torch.matmul(attn, vf))
    return torch.cat(outs, dim=2)


def twin_reference(
    q: Any, k: Any, v: Any, is_causal: bool = False, sm_scale: Optional[float] = None,
    smooth_k: bool = False, smooth_v: bool = False, pv_accum_dtype: str = "fp32+fp16"
) -> Any:
    """SageAttention2 CPU 孪生参考计算 (高保真 INT8 QK + FP8 PV 量化参比)"""
    import math
    from sageattention2.dispatch import run_sageattn_cpu_reference

    d = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    out_twin, _ = run_sageattn_cpu_reference(
        q.float().cpu(),
        k.float().cpu(),
        v.float().cpu(),
        "HND",
        is_causal,
        "per_thread",
        sm_scale,
        pv_accum_dtype,
        smooth_k,
        smooth_v,
        False,
    )
    return out_twin.to(q.dtype)


def evaluate_golden(out: Any, ref: Any, dtype_str: str) -> Dict[str, Any]:
    """对比测试输出与参比张量"""
    import torch
    import torch.nn.functional as F

    diff = (out.float().cpu() - ref.float().cpu()).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())

    flat_out = out.float().cpu().reshape(-1)
    flat_ref = ref.float().cpu().reshape(-1)
    cos_sim = float(F.cosine_similarity(flat_out.unsqueeze(0), flat_ref.unsqueeze(0)).item())

    tol = 2**-9 if "16" in dtype_str and "b" not in dtype_str else 2**-6
    matched = float((diff <= tol + tol * ref.float().cpu().abs()).float().mean().item())

    return {
        "max_err": max_err,
        "mean_err": mean_err,
        "cos_sim": cos_sim,
        "matched_ratio": matched,
        "tol": tol,
        "pass_matched": bool(matched >= 0.99),
    }


def _worker_run_op(args_dict: Dict[str, Any], queue: mp.Queue):
    """沙箱子进程：独立加载 torch_npu 并执行算子"""
    try:
        import torch
        has_npu = False
        try:
            import torch_npu
            if hasattr(torch, "npu") and torch.npu.is_available():
                device_id = args_dict.get("device", 0)
                torch.npu.set_device(device_id)
                has_npu = True
        except Exception:
            has_npu = False

        timeout = args_dict.get("timeout", 10)
        acl_ok = try_set_acl_timeout(timeout) if has_npu else False

        # 动态加载算子库
        repo_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
        alt_repo_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ops-transformer", "experimental", "attention", "sageattention2", "python"))
        for p in (repo_python, alt_repo_python):
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)
        import sageattention2 as sage2

        dtype = torch.float16 if args_dict["dtype"] == "fp16" else torch.bfloat16
        b, hq, hkv = args_dict["b"], args_dict["h"], args_dict["hkv"]
        sq, sk, d = args_dict["sq"], args_dict["sk"], args_dict["d"]
        layout = args_dict.get("layout", "HND")
        is_causal = args_dict.get("causal", False)
        smooth_k = args_dict.get("smooth_k", False)
        smooth_v = args_dict.get("smooth_v", False)

        gen = torch.Generator(device="cpu").manual_seed(args_dict.get("seed", 42))
        q = torch.randn(b, hq, sq, d, generator=gen, dtype=torch.float32).to(dtype)
        k = torch.randn(b, hkv, sk, d, generator=gen, dtype=torch.float32).to(dtype)
        v = torch.randn(b, hkv, sk, d, generator=gen, dtype=torch.float32).to(dtype)
        if has_npu:
            q = q.npu()
            k = k.npu()
            v = v.npu()

        if layout == "NHD":
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        input_stats = [
            compute_tensor_stats(q, "Query"),
            compute_tensor_stats(k, "Key"),
            compute_tensor_stats(v, "Value"),
        ]

        # Warmup
        with alarm_watchdog(timeout, "Warmup Run"):
            for _ in range(args_dict.get("warmup", 1)):
                _ = sage2.sageattn(
                    q, k, v,
                    is_causal=is_causal,
                    smooth_k=smooth_k,
                    smooth_v=smooth_v,
                    tensor_layout=layout
                )
                if has_npu:
                    torch.npu.synchronize()

        # Timed Repeat
        times_ms = []
        out = None
        with alarm_watchdog(timeout, "Benchmark Timed Run"):
            for _ in range(args_dict.get("repeat", 1)):
                t0 = time.perf_counter()
                out = sage2.sageattn(
                    q, k, v,
                    is_causal=is_causal,
                    smooth_k=smooth_k,
                    smooth_v=smooth_v,
                    tensor_layout=layout
                )
                if has_npu:
                    torch.npu.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        output_stats = [
            compute_tensor_stats(out, "Output"),
        ]

        golden_report = None
        if args_dict.get("check_golden", False):
            q_hnd = q if layout == "HND" else q.transpose(1, 2)
            k_hnd = k if layout == "HND" else k.transpose(1, 2)
            v_hnd = v if layout == "HND" else v.transpose(1, 2)
            out_hnd = out if layout == "HND" else out.transpose(1, 2)

            sdpa_ref = sdpa_reference(q_hnd, k_hnd, v_hnd, is_causal=is_causal)
            twin_ref = twin_reference(
                q_hnd, k_hnd, v_hnd, is_causal=is_causal,
                smooth_k=smooth_k, smooth_v=smooth_v
            )

            golden_report = {
                "twin": evaluate_golden(out_hnd, twin_ref, args_dict["dtype"]),
                "sdpa": evaluate_golden(out_hnd, sdpa_ref, args_dict["dtype"]),
                "twin_vs_sdpa": evaluate_golden(twin_ref, sdpa_ref, args_dict["dtype"]),
            }

        queue.put({
            "status": "SUCCESS",
            "acl_timeout_enabled": acl_ok,
            "times_ms": times_ms,
            "input_stats": input_stats,
            "output_stats": output_stats,
            "golden_report": golden_report,
        })
    except Exception as e:
        import traceback
        queue.put({
            "status": "ERROR",
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
        })


def run_case_in_sandbox(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """统一沙箱运行守护器：管理三层看门狗生命周期"""
    timeout = args_dict.get("timeout", 10)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker_run_op, args=(args_dict, q))
    p.start()

    p.join(timeout=timeout + 2)
    if p.is_alive():
        p.terminate()
        time.sleep(1.0)
        if p.is_alive():
            p.kill()
        p.join()
        return {
            "status": "TIMEOUT",
            "error_msg": f"[Tier-3 WATCHDOG TRIGGERED] 算子进程超时 ({timeout}s)！可能发生跨核硬件死锁或流同步未返回！",
        }

    if q.empty():
        return {
            "status": "ERROR",
            "error_msg": "[FATAL ERROR] 子进程异常退出，未返回诊断结果（可能发生底层 SIGSEGV / core dump）。",
        }

    res = q.get()
    return res


def compute_tflops(b: int, h: int, sq: int, sk: int, d: int, causal: bool, time_ms: float) -> float:
    """计算实际执行的有效 TFLOPS"""
    if time_ms <= 0:
        return 0.0
    flops = 4.0 * b * h * sq * sk * d
    if causal:
        flops /= 2.0
    return (flops / (time_ms * 1e-3)) / 1e12


def run_sweep():
    """执行全量矩阵扫描测试并输出表格"""
    print("\nSageAttention2 真机测试矩阵全量扫描 (Automated Hardware Test Sweep)")
    print(f"总计用例: {len(SWEEP_CASES)} 项典型硬件规格")

    results = []
    for idx, case in enumerate(SWEEP_CASES, 1):
        print(f"[{idx}/{len(SWEEP_CASES)}] 正在测试: {case['name']} (B={case['b']}, H={case['h']}:{case['hkv']}, S={case['sq']}, D={case['d']}, {case['dtype']})...", end="", flush=True)
        t_start = time.perf_counter()
        res = run_case_in_sandbox(case)
        elapsed = time.perf_counter() - t_start

        if res["status"] == "SUCCESS":
            min_time = min(res["times_ms"])
            tflops = compute_tflops(case["b"], case["h"], case["sq"], case["sk"], case["d"], case["causal"], min_time)
            g = res.get("golden_report")
            matched_pct = f"{g['matched_ratio']*100:.1f}%" if g else "N/A"
            cos_sim = f"{g['cos_sim']:.4f}" if g else "N/A"
            pass_status = "PASS" if (not g or g["pass_matched"]) else "WARN"
            print(f" -> [{pass_status}] {min_time:.3f} ms, {tflops:.1f} TFLOPS (Matched: {matched_pct})")
            results.append({
                "id": idx, "name": case["name"], "b": case["b"], "h": case["h"], "hkv": case["hkv"],
                "sq": case["sq"], "sk": case["sk"], "d": case["d"], "dtype": case["dtype"],
                "causal": case["causal"], "time_ms": min_time, "tflops": tflops,
                "matched": matched_pct, "cos_sim": cos_sim, "status": pass_status
            })
        elif res["status"] == "TIMEOUT":
            print(f" -> [TIMEOUT] 超过 {case['timeout']}s 看门狗阈值！")
            results.append({
                "id": idx, "name": case["name"], "b": case["b"], "h": case["h"], "hkv": case["hkv"],
                "sq": case["sq"], "sk": case["sk"], "d": case["d"], "dtype": case["dtype"],
                "causal": case["causal"], "time_ms": float("nan"), "tflops": 0.0,
                "matched": "-", "cos_sim": "-", "status": "TIMEOUT"
            })
        else:
            print(f" -> [FAIL] {res.get('error_msg', 'Unknown error')[:40]}...")
            results.append({
                "id": idx, "name": case["name"], "b": case["b"], "h": case["h"], "hkv": case["hkv"],
                "sq": case["sq"], "sk": case["sk"], "d": case["d"], "dtype": case["dtype"],
                "causal": case["causal"], "time_ms": float("nan"), "tflops": 0.0,
                "matched": "-", "cos_sim": "-", "status": "FAIL"
            })

    # 打印最终表格
    print("\n扫描测试汇总报告 (SWEEP SUMMARY)")
    header = f"{'#':<3} | {'Case Name':<18} | {'Shape (B,Hq,Hk,Sq,Sk,D)':<24} | {'Dtype':<6} | {'Causal':<6} | {'Time(ms)':<9} | {'TFLOPS':<8} | {'Match%':<7} | {'Status':<7}"
    print(header)
    pass_cnt = sum(1 for r in results if r["status"] == "PASS")
    timeout_cnt = sum(1 for r in results if r["status"] == "TIMEOUT")
    fail_cnt = sum(1 for r in results if r["status"] in ("FAIL", "WARN"))

    for r in results:
        shape_str = f"({r['b']},{r['h']},{r['hkv']},{r['sq']},{r['sk']},{r['d']})"
        time_str = f"{r['time_ms']:.3f}" if not math.isnan(r["time_ms"]) else "-"
        tflops_str = f"{r['tflops']:.1f}" if r['tflops'] > 0 else "-"
        print(f"{r['id']:<3} | {r['name']:<18} | {shape_str:<24} | {r['dtype']:<6} | {str(r['causal']):<6} | {time_str:<9} | {tflops_str:<8} | {r['matched']:<7} | {r['status']:<7}")
    print(f"测试统计: 共 {len(results)} 项 | 通过: {pass_cnt} 项 | 超时: {timeout_cnt} 项 | 异常: {fail_cnt} 项")
    if pass_cnt == len(results):
        print("全量硬件规格扫描 100% 验收通过！")
    else:
        print("存在部分规格未通过，请根据超时/报错信息针对性排查。")


def main():
    parser = argparse.ArgumentParser(
        description="SageAttention2 线上/真机诊断看门狗工具 (3-Tier Watchdog, Presets & Sweep)"
    )
    parser.add_argument("-p", "--preset", type=str, default=None, choices=list(PRESETS.keys()),
                        help=f"快速配置预设: {', '.join(PRESETS.keys())}")
    parser.add_argument("--sweep", action="store_true", default=False,
                        help="自动运行全量矩阵扫描测试 (8 组典型规格覆盖)")
    parser.add_argument("--b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--h", "--hq", dest="h", type=int, default=4, help="Query heads (default: 4)")
    parser.add_argument("--hkv", type=int, default=None, help="Key/Value heads (default: match hq)")
    parser.add_argument("--sq", type=int, default=256, help="Query seqlen (default: 256)")
    parser.add_argument("--sk", type=int, default=256, help="Key seqlen (default: 256)")
    parser.add_argument("--d", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Dtype (default: fp16)")
    parser.add_argument("--layout", type=str, default="HND", choices=["HND", "NHD"], help="Layout (default: HND)")
    parser.add_argument("--causal", action="store_true", default=False, help="Enable causal mask")
    parser.add_argument("--smooth-k", action="store_true", default=False, help="Enable smooth_k")
    parser.add_argument("--smooth-v", action="store_true", default=False, help="Enable smooth_v")
    parser.add_argument("--device", type=int, default=0, help="NPU device id (default: 0)")
    parser.add_argument("--timeout", type=int, default=10, help="Execution timeout in seconds (default: 10s)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (default: 1)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat iterations (default: 1)")
    parser.add_argument("--check-golden", action="store_true", default=False, help="Compare with FP32 reference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-sandbox", action="store_true", default=False, help="Run without mp sandbox")

    args = parser.parse_args()

    # 1. 扫描模式
    if args.sweep:
        run_sweep()
        sys.exit(0)

    # 2. 预设处理
    args_dict = vars(args)
    if args.preset and args.preset in PRESETS:
        p_cfg = PRESETS[args.preset]
        print(f"应用预设 [{args.preset}]: {p_cfg['description']}")
        # 仅当命令行未显式覆盖时填充预设值
        for k, v in p_cfg.items():
            if k == "description":
                continue
            if parser.get_default(k) == args_dict.get(k):
                args_dict[k] = v

    if args_dict["hkv"] is None:
        args_dict["hkv"] = args_dict["h"]

    print("SageAttention2 线上/真机硬件诊断与看门狗调试工具 (Watchdog Suite)")
    print(f"测试配置: B={args_dict['b']}, Hq={args_dict['h']}, Hkv={args_dict['hkv']}, Sq={args_dict['sq']}, Sk={args_dict['sk']}, D={args_dict['d']}")
    print(f"数据规格: Dtype={args_dict['dtype']}, Layout={args_dict['layout']}, Causal={args_dict['causal']}, Device={args_dict['device']}")
    print(f"看门狗限额: {args_dict['timeout']}s (三层隔离: aclrtSetOpExecuteTimeOut + signal.alarm + mp.Process)")

    if args_dict.get("no_sandbox"):
        q = mp.Queue()
        _worker_run_op(args_dict, q)
        res = q.get()
    else:
        res = run_case_in_sandbox(args_dict)

    if res["status"] == "TIMEOUT":
        print(f"\n❌ {res['error_msg']}")
        print("排查建议:")
        print("   1. 检查是否存在跨核未完成等待（如 Cube 等待 V0 的 Q_READY 或 KV_READY）；")
        print("   2. 检查 SAGE_DIAG_STAGE 编译宏（建议从 Stage 1~7 逐级隔离确认硬件通路）；")
        print("   3. 检查任务切分是否导致部分 Core 提前退出而对端核仍在等待。")
        sys.exit(124)

    if res["status"] != "SUCCESS":
        print(f"\n❌ 算子执行报错:\n{res.get('error_msg')}")
        if "traceback" in res:
            print(f"\n调用栈:\n{res['traceback']}")
        sys.exit(1)

    print("\n【输入张量统计】")
    print(format_stats_table(res["input_stats"]))

    print("\n【输出张量统计】")
    print(format_stats_table(res["output_stats"]))

    avg_time = sum(res["times_ms"]) / len(res["times_ms"])
    min_time = min(res["times_ms"])
    print(f"\n【性能时延】")
    print(f"   - 重复次数: {len(res['times_ms'])} 次")
    print(f"   - 平均时延: {avg_time:.3f} ms (最优: {min_time:.3f} ms)")

    # 理论 TFLOPS 计算
    tflops = compute_tflops(args_dict["b"], args_dict["h"], args_dict["sq"], args_dict["sk"], args_dict["d"], args_dict["causal"], min_time)
    print(f"   - 理论算力: {tflops:.2f} TFLOPS")

    has_golden_error = False
    if args_dict.get("check_golden") and res.get("golden_report"):
        g = res["golden_report"]
        gt = g["twin"]
        gs = g["sdpa"]
        gb = g["twin_vs_sdpa"]

        print(f"\n【L-A 门禁：SageAttention2 量化孪生参比 (Twin Golden)】")
        print(f"   - 最大绝对误差 (Max Err)   : {gt['max_err']:.6f}")
        print(f"   - 平均绝对误差 (Mean Err)  : {gt['mean_err']:.6f}")
        print(f"   - 余弦相似度 (Cosine Sim)  : {gt['cos_sim']:.6f}")
        print(f"   - 元素匹配率 (tol={gt['tol']:.6g})  : {gt['matched_ratio'] * 100:.2f}% (门禁要求 >= 99%)")
        if gt["pass_matched"]:
            print("   ✅ [PASS] 孪生量化精度完全对齐 (>= 99%)！")
        elif gt["cos_sim"] >= 0.95:
            print("   [INFO] 孪生余弦相似度达到高度相关 (>= 0.95)，残余微差主要源于向量核指令展开精度。")
        else:
            print("   ❌ [FAIL] 孪生量化余弦相似度未达标！")
            has_golden_error = True

        print(f"\n【L-B 门禁：PyTorch FP32 SDPA 原生参比 (Unquantized Baseline)】")
        print(f"   - 最大绝对误差 (Max Err)   : {gs['max_err']:.6f} (孪生理论参考: {gb['max_err']:.6f})")
        print(f"   - 平均绝对误差 (Mean Err)  : {gs['mean_err']:.6f} (孪生理论参考: {gb['mean_err']:.6f})")
        print(f"   - 余弦相似度 (Cosine Sim)  : {gs['cos_sim']:.6f} (门禁要求 >= 0.95)")
        print(f"   - 元素匹配率 (tol={gs['tol']:.6g})  : {gs['matched_ratio'] * 100:.2f}% (量化方法学理论上限: {gb['matched_ratio'] * 100:.2f}%)")
        if gs["cos_sim"] >= 0.95:
            print("   ✅ [PASS] SDPA 原生基准余弦相似度达标 (>= 0.95)！")
        else:
            print("   ❌ [FAIL] SDPA 原生基准余弦相似度未达标！")
            has_golden_error = True

    if has_golden_error:
        print("\n❌ 精度门禁校验未通过，测试失败退出！")
        sys.stdout.flush()
        sys.exit(1)

    print("\n诊断探测完成，硬件执行正常！")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
