#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roofline_mfu.py —— arch35 FA/FAG roofline 理论上限 (import fa_core)

同时报两层"计算访存比", 直面"理想 roofline 会高估"的问题:
  理想 AI = FLOPs / 理想一遍访存   -> 绝对天花板, 长 S 恒 compute-bound, MFU 上限 100%
  工程 AI = FLOPs / 真实重取访存   -> 真实 bound; 重取会把 AI 打塌, 可能跌到 memory-bound

memory-bound 时 MFU 上限 = AI / ridge。所以"roofline 说 compute、实测却 bandwidth"
的真相是: 用理想 AI 判 bound 高估了, 工程 AI 才是实际工作点(细看 traffic_gap.py)。

用法:
  python roofline_mfu.py
  python roofline_mfu.py --custom --op fa_fwd --S1 16384 ...
"""
from __future__ import annotations
import argparse
from fa_core import (HW, PRESETS, Shape, Split, peak_flops,
                     flops, ideal_hbm_bytes, eng_traffic)

def analyze(s: Shape, op: str, hw: HW) -> dict:
    F = flops(s, op); peak = peak_flops(s, hw); ridge = peak / (hw.hbm_bw_tbs*1e12)
    # 工程 ceiling 用"实现带宽"η×HBM 算有效 ridge(默认未融合/打HBM), 才反映真实 memory-bound
    bw_real = hw.mem_efficiency * hw.hbm_bw_tbs * 1e12
    ridge_real = peak / bw_real
    b_ideal = ideal_hbm_bytes(s, op); ai_ideal = F / b_ideal
    tr = eng_traffic(s, Split(op)); b_eng = tr["mte2_bytes"]; ai_eng = F / b_eng
    return dict(flops=F, peak_tflops=peak/1e12, ridge=ridge, ridge_real=ridge_real,
                ai_ideal=ai_ideal, mfu_ideal=min(1.0, ai_ideal/ridge),
                ai_eng=ai_eng, mfu_eng=min(1.0, ai_eng/ridge_real),
                bw_real_tbs=bw_real/1e12, t_compute_us=F/peak*1e6)

def report(name, s, op, hw, actual_us=None):
    r = analyze(s, op, hw)
    print(f"\n{'='*74}\n{name}  [{op}]  dtype={s.dtype} sparse={s.sparse}")
    print(f"  shape: B={s.B} N1={s.N1} N2={s.N2}(G={s.G}) S1={s.S1} S2={s.S2} D={s.D} Dv={s.Dv}")
    print(f"  HW: 峰值 {r['peak_tflops']:.0f} TFLOPS, HBM {hw.hbm_bw_tbs} TB/s, "
          f"ridge {r['ridge']:.0f}(峰值) / {r['ridge_real']:.0f}(实现带宽 η×HBM={r['bw_real_tbs']:.1f}T)")
    print(f"  {'-'*70}")
    print(f"  计算量 FLOPs            : {r['flops']/1e12:8.3f} TFLOP")
    print(f"  理想 AI (一遍访存)      : {r['ai_ideal']:9.1f} FLOP/byte -> "
          f"{'compute' if r['ai_ideal']>=r['ridge'] else 'memory'}-bound, MFU 上限 {r['mfu_ideal']*100:.0f}% (绝对天花板, vs 峰值ridge)")
    print(f"  工程 AI (真实重取/L2级) : {r['ai_eng']:9.1f} FLOP/byte -> "
          f"{'compute' if r['ai_eng']>=r['ridge_real'] else 'memory'}-bound, MFU 上限 {r['mfu_eng']*100:.0f}% (真实工作点, vs 实现ridge)")
    if r['ai_ideal'] >= r['ridge'] and r['ai_eng'] < r['ridge_real']:
        print(f"  [!] AI 塌缩: 理想说 compute-bound, 但重取把 AI 从 {r['ai_ideal']:.0f} 打到 "
              f"{r['ai_eng']:.0f}(< 实现ridge {r['ridge_real']:.0f}) -> 实际 memory-bound = 'roofline 乐观'的根源")
    if actual_us is not None:
        mfu = r['flops'] / (actual_us*1e-6 * r['peak_tflops']*1e12)
        print(f"  profiling: 实测 {actual_us:.0f}us -> MFU {mfu*100:.1f}% "
              f"(在工程 AI 上限 {r['mfu_eng']*100:.0f}% 之下, 差额=重叠/调度, 见 traffic_gap)")

def builtin():
    hw = PRESETS["950DT"]
    report("例1: FA 前向 (真实 profiling)", Shape(1,16,2,16384,16384,256,256,"bf16","causal"), "fa_fwd", hw, 7771.0)
    report("例2: FAG 反向 (真实 profiling)", Shape(1,16,2,16384,16384,256,256,"bf16","causal"), "fag_bwd", hw, 15453.0)
    report("例3: decode S1=1 (memory-bound)", Shape(1,32,4,1,8192,128,128,"bf16","full"), "fa_fwd", hw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom", action="store_true")
    ap.add_argument("--op", choices=["fa_fwd","fag_bwd"], default="fa_fwd")
    for k,d in [("B",1),("N1",16),("N2",2),("S1",16384),("S2",16384),("D",256),("Dv",256)]:
        ap.add_argument(f"--{k}", type=int, default=d)
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--sparse", default="causal")
    ap.add_argument("--band_w", type=int, default=0); ap.add_argument("--hw", default="950DT")
    ap.add_argument("--actual_us", type=float, default=None)
    a = ap.parse_args()
    if not a.custom: builtin(); return
    report("自定义 shape", Shape(a.B,a.N1,a.N2,a.S1,a.S2,a.D,a.Dv,a.dtype,a.sparse,a.band_w),
           a.op, PRESETS[a.hw], a.actual_us)

if __name__ == "__main__":
    main()