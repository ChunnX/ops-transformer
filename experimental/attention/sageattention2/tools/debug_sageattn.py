#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_sageattn.py: 华为昇腾 SageAttention2 线上/真机单核与全链路诊断调试看门狗工具

功能特性：
1. 标准 CLI 接口：--b, --h, --sq, --sk, --d, --dtype (fp16/bf16), --causal, --timeout, --device 等；
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
from typing import Any, Dict, Optional, Tuple


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
    }


def format_stats_table(stats_list: list[Dict[str, Any]]) -> str:
    """格式化张量统计表格"""
    header = f"{'Tensor':<10} | {'Shape':<20} | {'Dtype':<8} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10} | {'Norm':<10} | {'NaN/Inf':<8}"
    sep = "-" * len(header)
    rows = [header, sep]
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


def evaluate_golden(out: Any, ref: Any, dtype_str: str) -> Dict[str, Any]:
    """对比测试输出与 Golden 参比"""
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
        import torch_npu

        device_id = args_dict["device"]
        torch.npu.set_device(device_id)

        timeout = args_dict["timeout"]
        acl_ok = try_set_acl_timeout(timeout)

        # 动态加载算子库
        repo_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
        if repo_python not in sys.path:
            sys.path.insert(0, repo_python)
        import sageattention2 as sage2

        dtype = torch.float16 if args_dict["dtype"] == "fp16" else torch.bfloat16
        b, hq, hkv = args_dict["b"], args_dict["h"], args_dict["hkv"]
        sq, sk, d = args_dict["sq"], args_dict["sk"], args_dict["d"]
        layout = args_dict["layout"]
        is_causal = args_dict["causal"]
        smooth_k = args_dict["smooth_k"]
        smooth_v = args_dict["smooth_v"]

        gen = torch.Generator(device="cpu").manual_seed(args_dict["seed"])
        q = torch.randn(b, hq, sq, d, generator=gen, dtype=torch.float32).to(dtype).npu()
        k = torch.randn(b, hkv, sk, d, generator=gen, dtype=torch.float32).to(dtype).npu()
        v = torch.randn(b, hkv, sk, d, generator=gen, dtype=torch.float32).to(dtype).npu()

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
            for _ in range(args_dict["warmup"]):
                _ = sage2.sageattn(
                    q, k, v,
                    is_causal=is_causal,
                    smooth_k=smooth_k,
                    smooth_v=smooth_v,
                    tensor_layout=layout
                )
                torch.npu.synchronize()

        # Timed Repeat
        times_ms = []
        out = None
        with alarm_watchdog(timeout, "Benchmark Timed Run"):
            for _ in range(args_dict["repeat"]):
                t0 = time.perf_counter()
                out = sage2.sageattn(
                    q, k, v,
                    is_causal=is_causal,
                    smooth_k=smooth_k,
                    smooth_v=smooth_v,
                    tensor_layout=layout
                )
                torch.npu.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        output_stats = [
            compute_tensor_stats(out, "Output"),
        ]

        golden_report = None
        if args_dict["check_golden"]:
            q_hnd = q if layout == "HND" else q.transpose(1, 2)
            k_hnd = k if layout == "HND" else k.transpose(1, 2)
            v_hnd = v if layout == "HND" else v.transpose(1, 2)
            ref = sdpa_reference(q_hnd, k_hnd, v_hnd, is_causal=is_causal)
            out_hnd = out if layout == "HND" else out.transpose(1, 2)
            golden_report = evaluate_golden(out_hnd, ref, args_dict["dtype"])

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


def main():
    parser = argparse.ArgumentParser(
        description="SageAttention2 在线/真机诊断看门狗工具 (3-Tier Watchdog & Multi-Metric Diagnostics)"
    )
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
    if args.hkv is None:
        args.hkv = args.h

    print("================================================================================")
    print("      🚀 SageAttention2 线上/真机硬件诊断与看门狗调试工具 (Watchdog Suite)")
    print("================================================================================")
    print(f"🔧 测试配置: B={args.b}, Hq={args.h}, Hkv={args.hkv}, Sq={args.sq}, Sk={args.sk}, D={args.d}")
    print(f"🔧 数据规格: Dtype={args.dtype}, Layout={args.layout}, Causal={args.causal}, Device={args.device}")
    print(f"🛡️  看门狗限额: {args.timeout}s (三层隔离: aclrtSetOpExecuteTimeOut + signal.alarm + mp.Process)")
    print("--------------------------------------------------------------------------------")

    args_dict = vars(args)

    if args.no_sandbox:
        q = mp.Queue()
        _worker_run_op(args_dict, q)
        res = q.get()
    else:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_worker_run_op, args=(args_dict, q))
        p.start()
        
        # 等待并在超时后由 Tier 3 强杀
        p.join(timeout=args.timeout + 2)
        if p.is_alive():
            print(f"\n❌ [TIER-3 WATCHDOG TRIGGERED] 算子进程超时 ({args.timeout}s)！可能发生跨核硬件死锁！")
            p.terminate()
            time.sleep(1.0)
            if p.is_alive():
                p.kill()
            p.join()
            print("🛑 沙箱子进程已彻底终止并隔离清理，宿主终端安全返回。")
            sys.exit(124)

        if q.empty():
            print("\n❌ [FATAL ERROR] 子进程异常退出，未返回诊断结果（可能发生底层 SIGSEGV / core dump）。")
            sys.exit(1)
        res = q.get()

    if res["status"] != "SUCCESS":
        print(f"\n❌ 算子执行报错:\n{res.get('error_msg')}")
        if "traceback" in res:
            print(f"\n调用栈:\n{res['traceback']}")
        sys.exit(1)

    print("\n📊 【输入张量统计】")
    print(format_stats_table(res["input_stats"]))

    print("\n📊 【输出张量统计】")
    print(format_stats_table(res["output_stats"]))

    avg_time = sum(res["times_ms"]) / len(res["times_ms"])
    min_time = min(res["times_ms"])
    print(f"\n⏱️  【性能时延】")
    print(f"   - 重复次数: {len(res['times_ms'])} 次")
    print(f"   - 平均时延: {avg_time:.3f} ms (最优: {min_time:.3f} ms)")

    # 理论 TFLOPS 计算
    # 标准注意力 FLOPs: 4 * B * Hq * Sq * Sk * D (非 Causal) / 2 (Causal)
    flops = 4.0 * args.b * args.h * args.sq * args.sk * args.d
    if args.causal:
        flops /= 2.0
    tflops = (flops / (min_time * 1e-3)) / 1e12
    print(f"   - 理论算力: {tflops:.2f} TFLOPS")

    if args.check_golden and res["golden_report"]:
        g = res["golden_report"]
        print(f"\n🎯 【Golden 参比校验结果】")
        print(f"   - 最大绝对误差 (Max Err)   : {g['max_err']:.6f}")
        print(f"   - 平均绝对误差 (Mean Err)  : {g['mean_err']:.6f}")
        print(f"   - 余弦相似度 (Cosine Sim)  : {g['cos_sim']:.6f}")
        print(f"   - 元素匹配率 (Matched >= 99%): {g['matched_ratio'] * 100:.2f}% (容差阈值: {g['tol']})")
        if g["pass_matched"]:
            print("   ✅ [PASS] 精度验收通过！")
        else:
            print("   ⚠️  [WARN] 元素匹配率未达 99% 门禁线，请检查数值反量化与缩放系数！")

    print("\n================================================================================")
    print("      🎉 诊断探测完成，硬件执行正常！")
    print("================================================================================")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

