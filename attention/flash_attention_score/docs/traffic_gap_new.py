#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_gap.py —— arch35 FA/FAG 工程流量 + profiling 对账 (import fa_core)

相对上一版的三处升级 (回答 L2/CV/带宽 三个疑问):
  1) MTE2 改成 L2-aware: K/V 重取按"工作集 vs 128MB + 是否有局部性"分两档带宽
       L2-resident: η·16 TB/s  |  HBM-spill: η·4 TB/s   (η 由 profiling 校准)
     -> 能区分"重取命中 L2(便宜) vs 打到 HBM(贵)", 而不是套同一个实测带宽。
  2) 片上 CV 通路显式排除: S/P/dS 走 L0C/UB/L1, 不进 MTE2/FixPipe 的 GM 预算
       (eng_traffic 里单列 on_chip_cv_bytes 仅作信息)。
  3) 输出给"场景带": MTE2 在 L2-resident / HBM-spill 两档下的耗时, 让 L2 优化空间一眼可见。

工作流:
  calibrate(已知 profiling) -> η + 各 pipe 有效带宽
  predict(自定义 shape)     -> per-pipe 时间(MTE2 自动选 L2/HBM 档)+ bottleneck + 可达 MFU 上限
  gap(predict, profiling)   -> 实现离可达上限差多少 / bottleneck / 重叠余量
"""
from __future__ import annotations
import argparse
from fa_core import (HW, PRESETS, Shape, Split, peak_flops, flops,
                     eng_traffic, mte2_regime, kv_working_set_bytes, DTYPE_PEAK_MULT)

# ---------------------------------------------------------------- 校准
def calibrate(s: Shape, sp: Split, prof: dict, hw: HW) -> dict:
    """prof = {duration_us, mac_ratio?, mte2_ratio?, fixpipe_ratio?}"""
    t = prof["duration_us"] * 1e-6
    tr = eng_traffic(s, sp)
    eff = {}
    if prof.get("mac_ratio"):
        eff["mac_eff_tflops"] = tr["mac_flops"] / (prof["mac_ratio"]*t) / 1e12
    if prof.get("mte2_ratio"):
        realized = tr["mte2_bytes"] / (prof["mte2_ratio"]*t)          # B/s
        reg = mte2_regime(s, sp, hw)
        peak = (hw.l2_bw_tbs if reg["l2_served"] else hw.hbm_bw_tbs) * 1e12
        eff["mte2_realized_bw_tbs"] = realized/1e12
        eff["mem_efficiency"] = realized / peak                      # η, 喂回 hw
        eff["mte2_regime_at_calib"] = reg["regime"]
    if prof.get("fixpipe_ratio"):
        eff["fixpipe_bw_tbs"] = tr["fixpipe_bytes"] / (prof["fixpipe_ratio"]*t) / 1e12
    return eff

# ---------------------------------------------------------------- 预测 (L2-aware)
def predict(s: Shape, sp: Split, eff: dict, hw: HW) -> dict:
    tr = eng_traffic(s, sp)
    peak = peak_flops(s, hw)
    reg = mte2_regime(s, sp, hw)                                      # 用 hw.mem_efficiency
    t_mac  = tr["mac_flops"] / (eff["mac_eff_tflops"]*1e12)
    t_mte2 = tr["mte2_bytes"] / (reg["bw_used_tbs"]*1e12)
    t_fixp = tr["fixpipe_bytes"] / (eff["fixpipe_bw_tbs"]*1e12)
    # MTE2 两档场景(给优化空间)
    t_mte2_l2  = tr["mte2_bytes"] / (reg["bw_l2_tbs"]*1e12)
    t_mte2_hbm = tr["mte2_bytes"] / (reg["bw_hbm_tbs"]*1e12)
    times = {"mac":t_mac, "mte2":t_mte2, "fixpipe":t_fixp}
    floor = max(times.values()); bott = max(times, key=times.get)
    return dict(times_us={k:v*1e6 for k,v in times.items()}, floor_us=floor*1e6,
                bottleneck=bott, mfu_ceiling=tr["mac_flops"]/(floor*peak),
                regime=reg, traffic=tr,
                mte2_l2_us=t_mte2_l2*1e6, mte2_hbm_us=t_mte2_hbm*1e6)

def gap(s, sp, pred, prof, hw):
    t = prof["duration_us"]*1e-6; peak = peak_flops(s, hw)
    mfu = pred["traffic"]["mac_flops"]/(t*peak)
    return dict(mfu_ceiling=pred["mfu_ceiling"], mfu_actual=mfu,
                gap_pts=(pred["mfu_ceiling"]-mfu)*100,
                overlap_eff=pred["floor_us"]/prof["duration_us"],
                bottleneck=pred["bottleneck"])

# ---------------------------------------------------------------- 打印
DEFAULT_EFF = {}

def show(name, s, sp, eff, hw, prof=None):
    pr = predict(s, sp, eff, hw); tr = pr["traffic"]; reg = pr["regime"]
    cfg = sp.split_axis if sp.op=="fag_bwd" else ("fused" if sp.kv_fused else "unfused")
    print(f"\n  [{name}] op={sp.op} cfg={cfg}")
    print(f"    工程流量: MAC {tr['mac_flops']/1e12:.3f} TFLOP | MTE2读 {tr['mte2_bytes']/1e9:.2f} GB | "
          f"FixPipe写 {tr['fixpipe_bytes']/1e9:.2f} GB | 片上CV(不占GM) {tr['on_chip_cv_bytes']/1e9:.1f} GB")
    print(f"    K/V 工作集 {reg['working_set_MB']:.0f} MB {'<=' if reg['fits_l2'] else '>'} L2 128MB"
          f"  -> MTE2 走 [{reg['regime']}] @ {reg['bw_used_tbs']:.1f} TB/s")
    print(f"    per-pipe: mac {pr['times_us']['mac']:.0f}us | mte2 {pr['times_us']['mte2']:.0f}us "
          f"(L2档 {pr['mte2_l2_us']:.0f} / HBM档 {pr['mte2_hbm_us']:.0f}) | fixpipe {pr['times_us']['fixpipe']:.0f}us")
    print(f"    bottleneck={pr['bottleneck']} -> floor {pr['floor_us']:.0f}us -> 可达 MFU 上限 {pr['mfu_ceiling']*100:.1f}%")
    if prof:
        g = gap(s, sp, pr, prof, hw)
        print(f"    实测 {prof['duration_us']:.0f}us -> MFU {g['mfu_actual']*100:.1f}% | "
              f"gap {g['gap_pts']:.1f} pts | 重叠效率 {g['overlap_eff']*100:.0f}%")
    return pr

def builtin():
    hw = PRESETS["950DT"]

    # A) FA 前向校准 (未融合 -> HBM-spill, 校准 η)
    fa = Shape(1,16,2,16384,16384,256,256,"bf16","causal"); fa_sp = Split("fa_fwd", kv_fused=False)
    fa_eff = calibrate(fa, fa_sp, {"duration_us":7771.0,"mac_ratio":0.702,"mte2_ratio":0.801}, hw)
    print("="*80)
    print(f"步骤A — FA 前向校准: MTE2 实测 {fa_eff['mte2_realized_bw_tbs']:.2f} TB/s "
          f"(regime={fa_eff['mte2_regime_at_calib']}) -> η={fa_eff['mem_efficiency']:.2f}, "
          f"MAC {fa_eff['mac_eff_tflops']:.0f} TFLOPS")
    hw.mem_efficiency = fa_eff["mem_efficiency"]                       # 喂回 η

    # B) FAG 反向校准 (FixPipe-bound)
    fag = Shape(1,16,2,16384,16384,256,256,"bf16","causal"); fag_sp = Split("fag_bwd", split_axis="BN2GS1S2")
    fag_eff = calibrate(fag, fag_sp, {"duration_us":15453.0,"mac_ratio":0.885,"fixpipe_ratio":0.977}, hw)
    print(f"步骤B — FAG 反向校准: FixPipe 实测 {fag_eff['fixpipe_bw_tbs']:.2f} TB/s, MAC {fag_eff['mac_eff_tflops']:.0f} TFLOPS")

    EFF = {"mac_eff_tflops": (fa_eff["mac_eff_tflops"]+fag_eff["mac_eff_tflops"])/2,
           "fixpipe_bw_tbs": fag_eff["fixpipe_bw_tbs"]}
    DEFAULT_EFF.update(EFF)
    print(f"  >>> 合并: MAC {EFF['mac_eff_tflops']:.0f} TFLOPS, FixPipe {EFF['fixpipe_bw_tbs']:.2f} TB/s, "
          f"η={hw.mem_efficiency:.2f} (L2档={hw.mem_efficiency*hw.l2_bw_tbs:.1f}T / HBM档={hw.mem_efficiency*hw.hbm_bw_tbs:.1f}T)")

    print("\n"+"="*80+"\n步骤C — FA 前向: 未融合(打HBM) vs 融合(落L2) —— 直接回答'roofline乐观'")
    show("FA 未融合(实跑)", fa, Split("fa_fwd", kv_fused=False), EFF, hw,
         {"duration_us":7771.0,"mac_ratio":0.702,"mte2_ratio":0.801})
    show("FA 融合GQA->落L2", fa, Split("fa_fwd", kv_fused=True), EFF, hw)
    print("  解读: 未融合重取打 HBM(2.8T)->MTE2-bound 82%; 融合后 K/V 落 L2 且体积/G->MTE2 塌到 MAC 以下")
    print("        -> compute-bound, 上限回到 ~100%。这就是 roofline 理想值真正能拿到的前提。")

    print("\n"+"="*80+"\n步骤D — FAG 反向: splitAxis 扫(FixPipe 决定上限)")
    for ax in ["BN2GS1S2","BN2S2","BN2"]:
        show(f"FAG {ax}", fag, Split("fag_bwd", split_axis=ax, swizzle=(ax=="BN2GS1S2")), EFF, hw,
             {"duration_us":15453.0,"mac_ratio":0.885,"fixpipe_ratio":0.977} if ax=="BN2GS1S2" else None)

    print("\n"+"="*80+"\n步骤E — 自定义 shape 预测 (大 B*N2=32, BN2)")
    show("FAG big-BN2", Shape(16,16,2,16384,16384,256,256,"bf16","causal"),
         Split("fag_bwd", split_axis="BN2"), EFF, hw)
    print("\n用法: predict(Shape(...), Split(...), DEFAULT_EFF, hw)  [先 hw.mem_efficiency=η]")
    return hw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom", action="store_true")
    ap.add_argument("--op", choices=["fa_fwd","fag_bwd"], default="fag_bwd")
    for k,d in [("B",1),("N1",16),("N2",2),("S1",16384),("S2",16384),("D",256),("Dv",256),("Br",128),("Bc",128)]:
        ap.add_argument(f"--{k}", type=int, default=d)
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--sparse", default="causal"); ap.add_argument("--band_w", type=int, default=0)
    ap.add_argument("--split_axis", default="BN2GS1S2"); ap.add_argument("--kv_fused", action="store_true"); ap.add_argument("--swizzle", action="store_true")
    ap.add_argument("--actual_us", type=float, default=None)
    ap.add_argument("--mac_ratio", type=float, default=None); ap.add_argument("--mte2_ratio", type=float, default=None); ap.add_argument("--fixpipe_ratio", type=float, default=None)
    a = ap.parse_args()
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        hw = builtin()
    if not a.custom:
        builtin(); return
    s = Shape(a.B,a.N1,a.N2,a.S1,a.S2,a.D,a.Dv,a.dtype,a.sparse,a.band_w)
    sp = Split(a.op,a.Br,a.Bc,a.kv_fused,a.split_axis,a.swizzle)
    prof = None
    if a.actual_us:
        prof = {"duration_us":a.actual_us}
        for k in ("mac_ratio","mte2_ratio","fixpipe_ratio"):
            if getattr(a,k) is not None: prof[k]=getattr(a,k)
    print("="*80+"\n自定义 shape 预测 (校准常数来自内置两个 profiling):")
    show("custom", s, sp, DEFAULT_EFF, hw, prof)

if __name__ == "__main__":
    main()