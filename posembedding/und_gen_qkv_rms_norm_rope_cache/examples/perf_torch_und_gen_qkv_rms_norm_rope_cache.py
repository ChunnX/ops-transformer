#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UndGenQkvRmsNormRopeCache torch 接口性能测量（NPU 上板）。

输入构造复用 ../tests/assets/golden.py 的 build_case，只是把 shape 从命令行给进来，
默认就是本次要测的 case：und_qkv [6652,10,128] / gen_qkv [4100,10,128] /
(Hq,Hk,Hv)=(8,1,1) / k_cache = v_cache = [23568,128,1,128]。

前置条件同 test_torch_*.py：自定义算子包 + torch_extension 已安装，且
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH

冷/稳态分开报：kv cache 有 1.5GB，装不进 128MB L2，但 und_qkv/gen_qkv(共 27MB)
在连续迭代里会常驻 L2，稳态数字因此偏乐观，首拍(cold)才接近真实首次调用。
"""

import argparse
import glob
import os
import statistics
import sys

import torch
import torch_npu

# torch 2.12 删掉了 torch.fx.experimental.symbolic_shapes.hint_int，torch_npu 自带的
# torchair 仍在 import 它 —— cann_ops_transformer 的 `from .ops import *` 会连带把
# graph_convert 里的 torchair 拉起来，于是整包 import 直接失败。这里只补一个同名 shim
# 让 import 过去；本脚本走 eager，不碰图模式，shim 不参与实际计算。
import torch.fx.experimental.symbolic_shapes as _ss  # noqa: E402
if not hasattr(_ss, "hint_int"):
    def _hint_int(a, fallback=None):
        return int(a)
    _ss.hint_int = _hint_int

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests"))
from assets.golden import build_case, gather_cache_rows, golden_dense  # noqa: E402

import cann_ops_transformer  # noqa: E402

RTOL = 2 ** -7
ATOL = 2 ** -13


def _to_npu(t):
    return None if t is None else t.npu()


def make_inputs(args):
    """按命令行 shape 造一套 NPU 上的入参，返回 (kwargs, meta, cpu_case)。"""
    bs = args.block_size
    total = args.und_len + args.gen_len
    need_blocks = (total + bs - 1) // bs
    if args.num_blocks < need_blocks:
        raise SystemExit(f"num_blocks={args.num_blocks} 不足以放下 T={total}（至少 {need_blocks}）")
    spec = dict(
        name="perf",
        heads=(args.hq, args.hk, args.hv),
        und_len=args.und_len,
        gen_len=args.gen_len,
        cat=args.cat,
        slot_mode=args.slot_mode,
        mrope_section=[16, 16, 16],
        big=False,
        block_size=bs,
        extra_blocks=args.num_blocks - need_blocks,
        max_pos=args.max_pos,
    )
    case = build_case(spec)
    meta = case.pop("_meta")
    npu_case = dict(
        und_qkv=case["und_qkv"].npu(),
        und_weights_q=case["und_weights_q"].npu(),
        und_weights_k=case["und_weights_k"].npu(),
        cos_sin_cache=case["cos_sin_cache"].npu(),
        k_cache=case["k_cache"].npu(),
        v_cache=case["v_cache"].npu(),
        slot_mapping=case["slot_mapping"].npu(),
        positions=case["positions"].npu(),
        gen_qkv=_to_npu(case["gen_qkv"]),
        gen_weights_q=_to_npu(case["gen_weights_q"]),
        gen_weights_k=_to_npu(case["gen_weights_k"]),
        cat_indices=_to_npu(case["cat_indices"]),
    )
    torch.npu.synchronize()
    return npu_case, meta, case


def call(npu_case, args):
    return cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
        npu_case["und_qkv"], npu_case["und_weights_q"], npu_case["und_weights_k"],
        npu_case["cos_sin_cache"], npu_case["k_cache"], npu_case["v_cache"],
        npu_case["slot_mapping"], npu_case["positions"],
        npu_case["gen_qkv"], npu_case["gen_weights_q"], npu_case["gen_weights_k"],
        npu_case["cat_indices"],
        num_heads_q=args.hq, num_heads_k=args.hk, num_heads_v=args.hv,
        norm_eps=1e-6, mrope_section=[16, 16, 16])


def timed_once(npu_case, args):
    """单次调用的 device 时间（ms），用 npu Event 掐，不含 host 侧下发抖动。"""
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    call(npu_case, args)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end)


def make_l2_flusher(mib):
    """返回一个把 L2 冲干净的闭包：写一片 >L2 容量的 buffer，逼所有脏行回 HBM。

    950 的 L2 是 128MiB，本 case 的 und_qkv+gen_qkv 只有 27MiB，连续迭代里会整片
    常驻，稳态数字因此偏乐观。真实模型里每次调用之间夹着别的算子，L2 早被冲掉，
    所以"冲一遍再测"这一档才是单次调用的可用估计。
    """
    buf = torch.empty(mib * 2 ** 20 // 4, dtype=torch.float32, device="npu")
    def flush():
        buf.fill_(1.0)
    return flush


def check_precision(npu_case, cpu_case, meta, args):
    """跑一次并对 golden，确认测的是真在算而不是空转。"""
    hq, hk, hv = args.hq, args.hk, args.hv
    ref_q, ref_k, ref_v = golden_dense(
        cpu_case["und_qkv"], cpu_case["gen_qkv"], cpu_case["und_weights_q"], cpu_case["und_weights_k"],
        cpu_case["gen_weights_q"], cpu_case["gen_weights_k"], cpu_case["cos_sin_cache"],
        cpu_case["positions"], cpu_case["cat_indices"], hq, hk, hv,
        cpu_case["norm_eps"], cpu_case["mrope_section"])
    k_cache = cpu_case["k_cache"].clone().npu()
    v_cache = cpu_case["v_cache"].clone().npu()
    saved_k, saved_v = npu_case["k_cache"], npu_case["v_cache"]
    npu_case["k_cache"], npu_case["v_cache"] = k_cache, v_cache
    q = call(npu_case, args)
    torch.npu.synchronize()
    npu_case["k_cache"], npu_case["v_cache"] = saved_k, saved_v
    slot = cpu_case["slot_mapping"]
    ok = True
    for name, got, ref in (("q", q.cpu(), ref_q),
                           ("k_cache", gather_cache_rows(k_cache.cpu(), slot), ref_k),
                           ("v_cache", gather_cache_rows(v_cache.cpu(), slot), ref_v)):
        ref_f = ref.to(torch.bfloat16).float()
        diff = (got.float() - ref_f).abs()
        bad = (diff > (ATOL + RTOL * ref_f.abs())).sum().item()
        ok = ok and bad == 0
        print(f"  {name}: max_abs={diff.max().item():.5f} 不达标元素={bad}/{ref_f.numel()}")
    print(f"  -> 精度 {'PASS' if ok else 'FAIL'}")
    return ok


def profile(npu_case, args):
    """采一段 profiling，回读 kernel_details.csv 里本算子的 Task Duration。"""
    out_dir = args.prof_dir
    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    # 冲 L2 的 kernel 排在被测 kernel 前面，profiling 按 Name 过滤，不会把它算进来
    flush = make_l2_flusher(args.flush_mib) if args.prof_flush else None
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exp,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
    ):
        for _ in range(args.prof_iters):
            if flush is not None:
                flush()
            call(npu_case, args)
        torch.npu.synchronize()
    csvs = glob.glob(os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True)
    if not csvs:
        print(f"  [warn] {out_dir} 下没找到 kernel_details.csv")
        return
    import csv as _csv
    latest = max(csvs, key=os.path.getmtime)
    rows = list(_csv.DictReader(open(latest)))
    key = None
    for r in rows:
        if "und_gen_qkv" in r.get("Name", "").lower() or "UndGenQkv" in r.get("Name", ""):
            key = r["Name"]
            break
    if key is None:
        print(f"  [warn] {latest} 里没有本算子的行，Name 取值样例：{[r.get('Name') for r in rows[:5]]}")
        return
    mine = [r for r in rows if r["Name"] == key]
    durs = [float(r["Duration(us)"]) for r in mine]
    print(f"  kernel={key}  n={len(durs)}  Task Duration(us): "
          f"min={min(durs):.2f} median={statistics.median(durs):.2f} max={max(durs):.2f}")
    extra = ("Block Num", "aiv_time(us)", "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
             "aiv_scalar_ratio", "aiv_fixpipe_ratio", "aiv_icache_miss_rate")
    r0 = mine[len(mine) // 2]
    shown = {k: r0[k] for k in extra if k in r0 and r0[k] not in ("", "N/A")}
    if shown:
        print("  " + "  ".join(f"{k}={v}" for k, v in shown.items()))
    print(f"  csv={latest}")


def main():
    p = argparse.ArgumentParser(description="UndGenQkvRmsNormRopeCache 性能测量")
    p.add_argument("--und-len", type=int, default=6652)
    p.add_argument("--gen-len", type=int, default=4100)
    p.add_argument("--hq", type=int, default=8)
    p.add_argument("--hk", type=int, default=1)
    p.add_argument("--hv", type=int, default=1)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=23568)
    p.add_argument("--max-pos", type=int, default=32768)
    p.add_argument("--cat", default="shuffled")
    p.add_argument("--slot-mode", default="shuffled")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--flush-l2", action="store_true", help="每拍前冲 L2，量 L2 脏条件下的时延")
    p.add_argument("--flush-mib", type=int, default=512, help="冲 L2 用的 buffer 大小（950 L2=128MiB）")
    p.add_argument("--check", action="store_true", help="先对 golden 验一次精度")
    p.add_argument("--prof", action="store_true", help="采 profiling 取 Task Duration")
    p.add_argument("--prof-iters", type=int, default=20)
    p.add_argument("--prof-flush", action="store_true", help="profiling 时每拍前冲 L2")
    p.add_argument("--prof-dir", default="./prof_und_gen_qkv")
    args = p.parse_args()

    npu_case, meta, cpu_case = make_inputs(args)
    total = meta["total"]
    kv = npu_case["k_cache"]
    print(f"case: und_qkv={tuple(npu_case['und_qkv'].shape)} gen_qkv={tuple(npu_case['gen_qkv'].shape)} "
          f"T={total} heads=({args.hq},{args.hk},{args.hv}) "
          f"k_cache={tuple(kv.shape)} v_cache={tuple(npu_case['v_cache'].shape)} "
          f"cache={kv.numel() * 2 / 2**20:.0f}MiB x2")

    if args.check:
        print("精度自检（对 CPU golden）：")
        check_precision(npu_case, cpu_case, meta, args)

    cold = timed_once(npu_case, args)
    print(f"cold（首拍，L2 未预热）: {cold * 1000:.2f} us")

    for _ in range(args.warmup):
        call(npu_case, args)
    torch.npu.synchronize()

    samples = [timed_once(npu_case, args) * 1000 for _ in range(args.iters)]
    samples_sorted = sorted(samples)
    print(f"steady（{args.iters} 次单发，npu Event 计时，us）: "
          f"min={samples_sorted[0]:.2f} p50={statistics.median(samples):.2f} "
          f"mean={statistics.fmean(samples):.2f} p90={samples_sorted[int(0.9 * (len(samples) - 1))]:.2f} "
          f"max={samples_sorted[-1]:.2f} stdev={statistics.stdev(samples):.2f}")

    # 连发：不在每次之间同步，量的是背靠背吞吐（含 host 下发被隐藏后的真实占用）
    torch.npu.synchronize()
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(args.iters):
        call(npu_case, args)
    t1.record()
    torch.npu.synchronize()
    print(f"back-to-back（{args.iters} 次连发）: {t0.elapsed_time(t1) * 1000 / args.iters:.2f} us/次")

    if args.flush_l2:
        flush = make_l2_flusher(args.flush_mib)
        dirty = []
        for _ in range(args.iters):
            flush()
            dirty.append(timed_once(npu_case, args) * 1000)
        dirty_sorted = sorted(dirty)
        print(f"L2 冲脏后（每拍前写 {args.flush_mib}MiB，{args.iters} 次，us）: "
              f"min={dirty_sorted[0]:.2f} p50={statistics.median(dirty):.2f} "
              f"mean={statistics.fmean(dirty):.2f} max={dirty_sorted[-1]:.2f}")

    if args.prof:
        print("profiling：")
        profile(npu_case, args)


if __name__ == "__main__":
    main()
