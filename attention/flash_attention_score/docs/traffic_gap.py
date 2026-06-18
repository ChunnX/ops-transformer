#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_gap.py —— arch35 FA/FAG "真实数据流" 工程流量 + profiling 对账 (gap 分析)

roofline_mfu.py 给的是绝对天花板 (K/V 读一遍、输出写一遍)。但真实实现里:
  - FA 前向: K/V 按 S1 外块被重取多次 (未融合 x N1, 融合 x N2) -> MTE2(L2读) 暴涨
  - FAG 反向: 输出按 splitAxis 走 atomic/FP32 workspace, 贡献体积 ∝ S^2 -> FixPipe 暴涨
这个脚本把这些算成"每条 pipe 的工程流量",并做三件事:

  eng_traffic(shape, split)         按 split 配置算 mac_flops / mte2_bytes / fixpipe_bytes
  calibrate(shape, split, prof)     用一个真实 profiling 反推每条 pipe 的"有效带宽/算力"
  predict(shape, split, eff)        用校准后的常数预测任意 shape 的 per-pipe 时间 + 可达 MFU 上限
  gap(predict, prof)                预测(完美重叠) vs 实测 -> gap = 重叠/调度损失

核心工作流 (正是你要的):
  1) 拿已知 profiling 跑 calibrate() -> 得到这块硅的 MTE2/FixPipe 有效带宽、MAC 有效算力
  2) 对自定义 shape 跑 predict() -> 得到它的可达 MFU 上限 + bottleneck
  3) 等你拿到自定义 shape 的 profiling, 跑 gap() -> 看实现离可达上限还差多少、差在哪条 pipe

非循环性说明: per-pipe 流量(由 shape+split 决定)和 bottleneck 判定是真值;
"完美重叠 floor = max(per-pipe)" 是可达下界(天花板); overlap_eff = floor/实测 是拟合量,
只用来读"还有多少重叠余量",不参与预测 floor。
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse

# ---------------- 硬件 / dtype (与 roofline_mfu.py 一致, 此处自带以便独立运行) -----------
@dataclass
class HW:
    peak_cube_tflops: float = 432.0
    hbm_bw_tbs: float = 4.0
    aic_num: int = 32

DTYPE_BYTES     = {"bf16":2,"fp16":2,"fp32":4,"fp8":1,"hif8":1,"mxfp4":0.5}
DTYPE_PEAK_MULT = {"bf16":1,"fp16":1,"fp32":0.5,"fp8":2,"hif8":2,"mxfp4":4}
DACC = 4  # FP32 累加/workspace

@dataclass
class Shape:
    B:int; N1:int; N2:int; S1:int; S2:int; D:int; Dv:int
    dtype:str="bf16"; sparse:str="causal"; band_w:int=0
    @property
    def G(self): return self.N1 // self.N2
    def valid_ratio(self):
        return {"full":1.0,"causal":0.5,
                "band":min(1.0,self.band_w/self.S2)}.get(self.sparse,1.0)

@dataclass
class Split:
    """分核/循环序配置 (决定工程流量)。"""
    op: str                      # fa_fwd | fag_bwd
    Br: int = 128; Bc: int = 128
    kv_fused: bool = False       # FA: GQA 是否融合 K/V 加载 (融合 -> K/V 流量 /G)
    split_axis: str = "BN2GS1S2" # FAG: BN2 | BN2S2 | BN2GS1S2 (决定哪些输出走 partial/atomic)

def flops(s: Shape, op: str) -> float:
    if op == "fa_fwd":
        per = 2*s.S1*s.S2*(s.D+s.Dv)
    else:
        per = 2*s.S1*s.S2*(3*s.D+2*s.Dv)
    return s.B*s.N1*per*s.valid_ratio()

# ---------------------------------------------------------------------------
# 工程流量: 每条 pipe 真实搬/算多少
#   mac_flops    : Cube 计算量 (= 理论 FLOPs, 与循环序无关)
#   mte2_bytes   : MTE2 从 L2 读的字节 (输入重取, 这是 FA 的 bound)
#   fixpipe_bytes: FixPipe 从 L0C 往 GM 写的字节 (输出, 这是 FAG 的 bound)
# ---------------------------------------------------------------------------
def eng_traffic(s: Shape, sp: Split) -> dict:
    dt = DTYPE_BYTES[s.dtype]; vr = s.valid_ratio()
    M  = s.S1 / sp.Br        # S1 外块数
    Nn = s.S2 / sp.Bc        # S2 外块数
    F  = flops(s, sp.op)

    if sp.op == "fa_fwd":
        # 外 Q 内 KV: Q 驻留读一遍; K/V 每个 S1 外块再取一次 (融合则 /G)
        kvdiv = s.G if sp.kv_fused else 1
        kv_refetch = s.B * s.N1 * M * s.S2 * (s.D + s.Dv) * dt * vr / kvdiv
        q_once     = s.B * s.N1 * s.S1 * s.D * dt
        mte2 = kv_refetch + q_once
        fixp = s.B * s.N1 * s.S1 * s.Dv * dt           # O 写一遍 (小)
        extra = {"kv_refetch_GB": kv_refetch/1e9, "kv_refetch_x": M/kvdiv}

    else:  # fag_bwd
        # MTE2: 内层 2 张量重取 (外KV: Q+dO 取 Nn 次; 与外Q 对称, 不是区分点)
        qdo_refetch = s.B * s.N1 * Nn * s.S1 * (s.D + s.Dv) * dt * vr
        kv_once     = s.B * s.N2 * s.S2 * (s.D + s.Dv) * dt
        mte2 = qdo_refetch + kv_once

        # FixPipe: 各输出写出体积. 驻留 -> 写一遍(∝S); partial/atomic -> 贡献体积(∝S^2)
        dq_resident = s.B * s.N1 * s.S1 * s.D
        dk_resident = s.B * s.N2 * s.S2 * s.D
        dv_resident = s.B * s.N2 * s.S2 * s.Dv
        dq_partial  = s.B * s.N1 * s.S1 * s.S2 * s.D  / sp.Bc * vr   # dQ 沿 S2 归约
        dk_partial  = s.B * s.N1 * s.S1 * s.S2 * s.D  / sp.Br * vr   # dK 沿(G,S1)归约
        dv_partial  = s.B * s.N1 * s.S1 * s.S2 * s.Dv / sp.Br * vr
        if sp.split_axis == "BN2":          # 0 跨核归约: 三个都驻留写一遍
            fout = dq_resident + dk_resident + dv_resident; ncross = 0
        elif sp.split_axis == "BN2S2":      # 只 dQ 跨核
            fout = dq_partial + dk_resident + dv_resident; ncross = 1
        else:                               # BN2GS1S2: 三个都 partial
            fout = dq_partial + dk_partial + dv_partial;   ncross = 3
        fixp = fout * DACC
        extra = {"cross_core_tensors": ncross,
                 "dq_partial_GB": dq_partial*DACC/1e9,
                 "dkv_partial_GB": (dk_partial+dv_partial)*DACC/1e9}

    return dict(mac_flops=F, mte2_bytes=mte2, fixpipe_bytes=fixp, **extra)

# ---------------------------------------------------------------------------
# 校准: 用真实 profiling 的 (耗时, 各 pipe 占空比) 反推有效带宽/算力
#   prof = {duration_us, mac_ratio?, mte2_ratio?, fixpipe_ratio?}
# ---------------------------------------------------------------------------
def calibrate(s: Shape, sp: Split, prof: dict, hw: HW = HW()) -> dict:
    t = prof["duration_us"] * 1e-6
    tr = eng_traffic(s, sp)
    eff = {}
    if prof.get("mac_ratio"):
        eff["mac_eff_tflops"]    = tr["mac_flops"]    / (prof["mac_ratio"]    * t) / 1e12
    if prof.get("mte2_ratio"):
        eff["mte2_eff_bw_tbs"]   = tr["mte2_bytes"]   / (prof["mte2_ratio"]   * t) / 1e12
    if prof.get("fixpipe_ratio"):
        eff["fixpipe_eff_bw_tbs"]= tr["fixpipe_bytes"]/ (prof["fixpipe_ratio"]* t) / 1e12
    return eff

# ---------------------------------------------------------------------------
# 预测: 用校准常数算 per-pipe 时间 + 完美重叠 floor + 可达 MFU 上限
# ---------------------------------------------------------------------------
def predict(s: Shape, sp: Split, eff: dict, hw: HW = HW()) -> dict:
    tr   = eng_traffic(s, sp)
    peak = hw.peak_cube_tflops * 1e12 * DTYPE_PEAK_MULT[s.dtype]
    t_mac  = tr["mac_flops"]     / (eff["mac_eff_tflops"]    * 1e12)
    t_mte2 = tr["mte2_bytes"]    / (eff["mte2_eff_bw_tbs"]   * 1e12)
    t_fixp = tr["fixpipe_bytes"] / (eff["fixpipe_eff_bw_tbs"]* 1e12)
    times = {"mac":t_mac, "mte2":t_mte2, "fixpipe":t_fixp}
    floor = max(times.values())                      # 完美重叠下界 (可达天花板)
    bott  = max(times, key=times.get)
    mfu_ceiling = tr["mac_flops"] / (floor * peak)    # 若耗时=floor 的 MFU
    return dict(times_us={k:v*1e6 for k,v in times.items()},
                floor_us=floor*1e6, bottleneck=bott,
                mfu_ceiling=mfu_ceiling, traffic=tr)

def gap(s: Shape, sp: Split, pred: dict, prof: dict, hw: HW = HW()) -> dict:
    t_act = prof["duration_us"]*1e-6
    peak  = hw.peak_cube_tflops*1e12*DTYPE_PEAK_MULT[s.dtype]
    mfu_act = pred["traffic"]["mac_flops"] / (t_act*peak)
    return dict(mfu_ceiling=pred["mfu_ceiling"], mfu_actual=mfu_act,
                gap_pts=(pred["mfu_ceiling"]-mfu_act)*100,
                overlap_eff=pred["floor_us"]/(prof["duration_us"]),
                bottleneck=pred["bottleneck"])

# ---------------------------------------------------------------------------
# 默认校准常数 (从你两个 profiling 反推, 见 __main__; 自定义预测开箱即用)
# ---------------------------------------------------------------------------
DEFAULT_EFF = {}  # 运行内置例子时填充

def _fmt_traffic(tr):
    s = f"mac {tr['mac_flops']/1e12:.3f} TFLOP | MTE2 {tr['mte2_bytes']/1e9:.2f} GB | FixPipe {tr['fixpipe_bytes']/1e9:.2f} GB"
    if "cross_core_tensors" in tr: s += f" | 跨核张量 {tr['cross_core_tensors']}"
    if "kv_refetch_x" in tr: s += f" | K/V 重取 x{tr['kv_refetch_x']:.0f}"
    return s

def show_predict(name, s, sp, eff, hw, prof=None):
    pr = predict(s, sp, eff, hw)
    print(f"\n  [{name}]  split={sp.split_axis if sp.op=='fag_bwd' else ('fused' if sp.kv_fused else 'unfused')}")
    print(f"    工程流量: {_fmt_traffic(pr['traffic'])}")
    print(f"    per-pipe: mac {pr['times_us']['mac']:.0f}us | mte2 {pr['times_us']['mte2']:.0f}us | fixpipe {pr['times_us']['fixpipe']:.0f}us")
    print(f"    bottleneck = {pr['bottleneck']}  -> 完美重叠 floor {pr['floor_us']:.0f}us  -> 可达 MFU 上限 {pr['mfu_ceiling']*100:.1f}%")
    if prof:
        g = gap(s, sp, pr, prof, hw)
        print(f"    实测 {prof['duration_us']:.0f}us -> MFU {g['mfu_actual']*100:.1f}% | "
              f"距可达上限 gap {g['gap_pts']:.1f} pts | 重叠效率 {g['overlap_eff']*100:.0f}%")
    return pr

# ---------------------------------------------------------------------------
def builtin():
    hw = HW(432.0, 4.0, 32)

    # ===== 1) FA 前向: 校准 MTE2 有效带宽 (MTE2-bound) =====
    fa  = Shape(1,16,2,16384,16384,256,256,"bf16","causal")
    fa_sp = Split("fa_fwd", kv_fused=False)             # profiler 证明未融合
    fa_prof = {"duration_us":7771.0, "mac_ratio":0.702, "mte2_ratio":0.801}
    fa_eff = calibrate(fa, fa_sp, fa_prof, hw)
    print("="*78)
    print("步骤A — 用 FA 前向 profiling 校准 (MTE2-bound):")
    print(f"  反推: MAC 有效算力 {fa_eff['mac_eff_tflops']:.0f} TFLOPS, "
          f"MTE2(L2读) 有效带宽 {fa_eff['mte2_eff_bw_tbs']:.2f} TB/s")
    show_predict("FA 前向 自校验", fa, fa_sp, {**fa_eff,"fixpipe_eff_bw_tbs":99}, hw, fa_prof)

    # ===== 2) FAG 反向: 校准 FixPipe 有效带宽 (FixPipe-bound) =====
    fag = Shape(1,16,2,16384,16384,256,256,"bf16","causal")
    fag_sp = Split("fag_bwd", split_axis="BN2GS1S2")    # B*N2=2 逼到最碎
    fag_prof = {"duration_us":15453.0, "mac_ratio":0.885, "fixpipe_ratio":0.977}
    fag_eff = calibrate(fag, fag_sp, fag_prof, hw)
    print("\n"+"="*78)
    print("步骤B — 用 FAG 反向 profiling 校准 (FixPipe-bound):")
    print(f"  反推: MAC 有效算力 {fag_eff['mac_eff_tflops']:.0f} TFLOPS, "
          f"FixPipe 有效带宽 {fag_eff['fixpipe_eff_bw_tbs']:.2f} TB/s")
    # 合并校准常数 (MTE2 来自 FA, FixPipe 来自 FAG, MAC 取均值)
    EFF = {"mac_eff_tflops": (fa_eff['mac_eff_tflops']+fag_eff['mac_eff_tflops'])/2,
           "mte2_eff_bw_tbs": fa_eff['mte2_eff_bw_tbs'],
           "fixpipe_eff_bw_tbs": fag_eff['fixpipe_eff_bw_tbs']}
    DEFAULT_EFF.update(EFF)
    print(f"\n  >>> 合并校准常数 (供自定义 shape 预测): "
          f"MAC {EFF['mac_eff_tflops']:.0f} TFLOPS, MTE2 {EFF['mte2_eff_bw_tbs']:.2f} TB/s, "
          f"FixPipe {EFF['fixpipe_eff_bw_tbs']:.2f} TB/s")
    show_predict("FAG 反向 自校验 (BN2GS1S2)", fag, fag_sp, EFF, hw, fag_prof)

    # ===== 3) 同 shape 换 splitAxis -> 看 FixPipe 流量怎么决定可达上限 =====
    print("\n"+"="*78)
    print("步骤C — 同 shape 换 splitAxis (反事实: 若能走更粗的切法):")
    for ax in ["BN2GS1S2","BN2S2","BN2"]:
        show_predict(f"FAG {ax}", fag, Split("fag_bwd", split_axis=ax), EFF, hw,
                     fag_prof if ax=="BN2GS1S2" else None)
    print("  解读: 3 跨核张量(BN2GS1S2) FixPipe-bound ~84%; 退到 1(BN2S2)/0(BN2) 回到 MAC-bound ~93%。")
    print("        本 shape B*N2=2、S1=S2 -> 填不满也切不出长S2 -> 被迫 BN2GS1S2,这就是 82% 的根因。")

    # ===== 4) 自定义 shape 预测 (大 B*N2, 可走 BN2) =====
    print("\n"+"="*78)
    print("步骤D — 自定义 shape 预测 (大 B*N2=32, FAG 可走 BN2 -> 0 跨核):")
    big = Shape(16,16,2,16384,16384,256,256,"bf16","causal")   # B*N2=32 填满
    show_predict("FAG big-BN2", big, Split("fag_bwd", split_axis="BN2"), EFF, hw)
    print("  -> 预测可达上限 ~93% (MAC-bound); 比 82% 那档 +~10 pts, 印证 §13.6.2 的 splitAxis 价值。")

    print("\n"+"="*78)
    print("用法: 把你新 shape 填进 Shape(...), 选 op/split, 调 predict(s, sp, DEFAULT_EFF);")
    print("      等你有了该 shape 的 profiling, 调 gap() 看实现离可达上限的差距和 bottleneck。")

def main():
    ap = argparse.ArgumentParser(description="arch35 FA/FAG 工程流量 + profiling 对账")
    ap.add_argument("--custom", action="store_true")
    ap.add_argument("--op", choices=["fa_fwd","fag_bwd"], default="fag_bwd")
    ap.add_argument("--B",type=int,default=1); ap.add_argument("--N1",type=int,default=16)
    ap.add_argument("--N2",type=int,default=2); ap.add_argument("--S1",type=int,default=16384)
    ap.add_argument("--S2",type=int,default=16384); ap.add_argument("--D",type=int,default=256)
    ap.add_argument("--Dv",type=int,default=256); ap.add_argument("--dtype",default="bf16")
    ap.add_argument("--sparse",default="causal"); ap.add_argument("--band_w",type=int,default=0)
    ap.add_argument("--Br",type=int,default=128); ap.add_argument("--Bc",type=int,default=128)
    ap.add_argument("--split_axis",default="BN2GS1S2"); ap.add_argument("--kv_fused",action="store_true")
    ap.add_argument("--actual_us",type=float,default=None)
    ap.add_argument("--mac_ratio",type=float,default=None)
    ap.add_argument("--mte2_ratio",type=float,default=None)
    ap.add_argument("--fixpipe_ratio",type=float,default=None)
    a = ap.parse_args()
    if not a.custom:
        builtin(); return
    # 自定义: 先用内置例子算出 DEFAULT_EFF, 再预测/对账
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        builtin()
    s = Shape(a.B,a.N1,a.N2,a.S1,a.S2,a.D,a.Dv,a.dtype,a.sparse,a.band_w)
    sp = Split(a.op,a.Br,a.Bc,a.kv_fused,a.split_axis)
    prof = None
    if a.actual_us:
        prof = {"duration_us":a.actual_us}
        for k in ("mac_ratio","mte2_ratio","fixpipe_ratio"):
            if getattr(a,k) is not None: prof[k]=getattr(a,k)
    print("="*78); print("自定义 shape 预测 (校准常数来自内置两个 profiling):")
    show_predict("custom", s, sp, DEFAULT_EFF, HW(), prof)

if __name__ == "__main__":
    main()