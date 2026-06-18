#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roofline_mfu.py —— arch35 (Ascend 950) FA / FAG 理论性能上限 (roofline) 计算器

给定一个 shape,输出:
  1) 计算量 FLOPs (FA 前向 2 matmul / FAG 反向 5 matmul,带 causal/band 折扣)
  2) 理想一遍访存字节 (Q/K/V[/dO] 各读一遍 + 输出各写一遍 —— 完美融合下界)
  3) 算术强度 AI = FLOPs / bytes  (= "计算访存比")
  4) ridge 点 = 峰值算力 / HBM 带宽
  5) bound 判定 + 理论 MFU 上限:
        compute-bound (AI >= ridge)  -> MFU 上限 = 100% (能喂满 Cube)
        memory-bound  (AI <  ridge)  -> MFU 上限 = AI / ridge
  6) 最短耗时 = max(FLOPs/峰值, bytes/带宽)

这是"绝对天花板"(假设 K/V 只从 HBM 读一遍、输出只写一遍)。真实数据流里
的 K/V 重取、输出 atomic/workspace 会把它压低 —— 那部分在 traffic_gap.py。

用法:
  python roofline_mfu.py                 # 跑内置的三个例子 (FA fwd / FAG bwd / 短S memory-bound)
  python roofline_mfu.py --custom ...    # 自定义 shape,见 --help
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse

# ----------------------------------------------------------------------------
# 硬件配置 (默认 = 你两个 profiling 用的 950DT full: 峰值 432 TFLOPS BF16, HBM 4 TB/s)
# ----------------------------------------------------------------------------
@dataclass
class HW:
    peak_cube_tflops: float = 432.0   # BF16 峰值 (32 核 x 13.5)
    hbm_bw_tbs: float = 4.0           # 950DT 4 TB/s (950PR = 1.6)
    aic_num: int = 32

PRESETS = {
    "950DT": HW(432.0, 4.0, 32),
    "950PR": HW(432.0, 1.6, 32),
}

DTYPE_BYTES      = {"bf16":2, "fp16":2, "fp32":4, "fp8":1, "hif8":1, "mxfp4":0.5}
# Cube 峰值相对 BF16 的倍率 (白皮书: FP8/HiF8=2x, MXFP4=4x; FP32 约 0.5x,待核实)
DTYPE_PEAK_MULT  = {"bf16":1, "fp16":1, "fp32":0.5, "fp8":2, "hif8":2, "mxfp4":4}

# ----------------------------------------------------------------------------
# shape
# ----------------------------------------------------------------------------
@dataclass
class Shape:
    B:int; N1:int; N2:int; S1:int; S2:int; D:int; Dv:int
    dtype:str = "bf16"
    sparse:str = "causal"     # full | causal | band
    band_w:int = 0            # sparse=band 时的窗口宽 (token)

    @property
    def G(self) -> int:
        return self.N1 // self.N2

    def valid_ratio(self) -> float:
        """有效基本块占满网格的比例 (causal 下三角 ~0.5)。"""
        if self.sparse == "full":   return 1.0
        if self.sparse == "causal": return 0.5
        if self.sparse == "band":   return min(1.0, self.band_w / self.S2)
        return 1.0

# ----------------------------------------------------------------------------
# 计算量
# ----------------------------------------------------------------------------
def flops(s: Shape, op: str) -> float:
    """理论 FLOPs (含 causal/band 折扣)。MAC=2 FLOP。"""
    if op == "fa_fwd":
        per_head = 2 * s.S1 * s.S2 * (s.D + s.Dv)            # QK^T + PV
    elif op == "fag_bwd":
        per_head = 2 * s.S1 * s.S2 * (3 * s.D + 2 * s.Dv)    # MM1..MM5 = 2.5x fwd
    else:
        raise ValueError(op)
    return s.B * s.N1 * per_head * s.valid_ratio()

# ----------------------------------------------------------------------------
# 理想一遍访存 (完美融合下界): 输入各读一遍 + 输出各写一遍
# 注意: causal 下你仍要把 K/V/Q 各读一遍、输出各写一遍 -> 不打 valid_ratio 折扣
# ----------------------------------------------------------------------------
def ideal_hbm_bytes(s: Shape, op: str) -> float:
    dt = DTYPE_BYTES[s.dtype]
    q  = s.B * s.N1 * s.S1 * s.D  * dt
    k  = s.B * s.N2 * s.S2 * s.D  * dt
    v  = s.B * s.N2 * s.S2 * s.Dv * dt
    if op == "fa_fwd":
        o = s.B * s.N1 * s.S1 * s.Dv * dt
        return q + k + v + o
    else:  # fag_bwd: 还要读 dO, 写 dQ/dK/dV
        do = s.B * s.N1 * s.S1 * s.Dv * dt
        dq = s.B * s.N1 * s.S1 * s.D  * dt
        dk = s.B * s.N2 * s.S2 * s.D  * dt
        dv = s.B * s.N2 * s.S2 * s.Dv * dt
        return q + k + v + do + dq + dk + dv

# ----------------------------------------------------------------------------
# roofline
# ----------------------------------------------------------------------------
def roofline(s: Shape, op: str, hw: HW = HW()) -> dict:
    F     = flops(s, op)
    peak  = hw.peak_cube_tflops * 1e12 * DTYPE_PEAK_MULT[s.dtype]   # FLOP/s
    bw    = hw.hbm_bw_tbs * 1e12                                    # B/s
    byts  = ideal_hbm_bytes(s, op)
    AI    = F / byts                                               # FLOP/byte
    ridge = peak / bw                                             # FLOP/byte
    compute_bound = AI >= ridge
    mfu_ceiling   = 1.0 if compute_bound else AI / ridge
    t_compute = F / peak
    t_mem     = byts / bw
    min_dur   = max(t_compute, t_mem)
    # 达到上限时的有效算力
    achieved_tflops = (F / min_dur) / 1e12
    return dict(
        op=op, flops=F, peak_tflops=peak/1e12, hbm_bytes=byts,
        AI=AI, ridge=ridge, bound=("compute" if compute_bound else "memory(HBM)"),
        mfu_ceiling=mfu_ceiling, min_dur_us=min_dur*1e6,
        t_compute_us=t_compute*1e6, t_mem_us=t_mem*1e6,
        achieved_tflops=achieved_tflops,
        # 每核理论分摊 (看 B*N1 能不能填满 aicNum)
        bn1=s.B*s.N1, bn2=s.B*s.N2, aic_num=hw.aic_num,
        fill_bn2=min(1.0, (s.B*s.N2)/hw.aic_num),
    )

# ----------------------------------------------------------------------------
# 打印
# ----------------------------------------------------------------------------
def report(name: str, s: Shape, op: str, hw: HW, actual_us: float | None = None):
    r = roofline(s, op, hw)
    print(f"\n{'='*72}\n{name}  [{op}]  dtype={s.dtype} sparse={s.sparse}")
    print(f"  shape: B={s.B} N1={s.N1} N2={s.N2}(G={s.G}) S1={s.S1} S2={s.S2} D={s.D} Dv={s.Dv}")
    print(f"  HW: 峰值 {r['peak_tflops']:.0f} TFLOPS, HBM {hw.hbm_bw_tbs} TB/s, aicNum {hw.aic_num}")
    print(f"  {'-'*68}")
    print(f"  计算量 FLOPs          : {r['flops']/1e12:8.3f} TFLOP")
    print(f"  理想一遍访存          : {r['hbm_bytes']/1e9:8.3f} GB")
    print(f"  算术强度 AI(HBM一遍)  : {r['AI']:8.1f} FLOP/byte   <- 计算访存比")
    print(f"  ridge = 峰值/带宽     : {r['ridge']:8.1f} FLOP/byte")
    print(f"  bound                 : {r['bound']}")
    print(f"  >>> 理论 MFU 上限     : {r['mfu_ceiling']*100:6.1f}%   "
          f"({'compute-bound: 能喂满 Cube,上限 100%' if r['bound']=='compute' else 'memory-bound: 上限=AI/ridge'})")
    print(f"  最短耗时(完美)        : {r['min_dur_us']:8.1f} us  "
          f"(compute {r['t_compute_us']:.0f} / mem {r['t_mem_us']:.0f})")
    print(f"  达到上限时有效算力    : {r['achieved_tflops']:8.1f} TFLOPS")
    print(f"  填核: B*N2={r['bn2']} / aicNum={r['aic_num']} -> BN2 填核率 {r['fill_bn2']*100:.0f}%"
          f"  ({'BN2 够填,可 0 跨核归约' if r['fill_bn2']>=1 else '不够填,需切到 G/S1/S2 -> 跨核归约 (见 traffic_gap.py)'})")
    if actual_us is not None:
        mfu_act = r['flops'] / (actual_us*1e-6 * r['peak_tflops']*1e12)
        print(f"  --- 真实 profiling ---")
        print(f"  实测耗时              : {actual_us:8.1f} us")
        print(f"  实测 MFU              : {mfu_act*100:6.1f}%")
        print(f"  距理论上限 gap        : {(r['mfu_ceiling']-mfu_act)*100:6.1f} pts")
        print(f"      (注: 这是对'绝对天花板'的 gap; 真实数据流的可达天花板见 traffic_gap.py)")

# ----------------------------------------------------------------------------
# 内置例子 + CLI
# ----------------------------------------------------------------------------
def builtin_examples():
    hw = PRESETS["950DT"]
    # 例1: 你的 FA 前向 profiling (Duration 7771us, 实测 MFU 65.5%, MTE2-bound)
    fa = Shape(B=1, N1=16, N2=2, S1=16384, S2=16384, D=256, Dv=256, dtype="bf16", sparse="causal")
    report("例1: FA 前向 (真实 profiling)", fa, "fa_fwd", hw, actual_us=7771.0)

    # 例2: 你的 FAG 反向 profiling (Duration 15453us, 实测 MFU 82.3%, FixPipe-bound)
    fag = Shape(B=1, N1=16, N2=2, S1=16384, S2=16384, D=256, Dv=256, dtype="bf16", sparse="causal")
    report("例2: FAG 反向 (真实 profiling)", fag, "fag_bwd", hw, actual_us=15453.0)

    # 例3: decode (S1=1) -> 干净演示 memory-bound: AI<<ridge, MFU 上限 = AI/ridge
    #   注意力在 HBM 一遍口径下,长 prefill 几乎总是 compute-bound;
    #   真正 memory-bound 的是 decode(算一个 query 却要读整段 K/V)。
    dec = Shape(B=1, N1=32, N2=4, S1=1, S2=8192, D=128, Dv=128, dtype="bf16", sparse="full")
    report("例3: decode S1=1 (演示 memory-bound MFU = AI/ridge)", dec, "fa_fwd", hw)

def main():
    ap = argparse.ArgumentParser(description="arch35 FA/FAG roofline 理论上限计算器")
    ap.add_argument("--custom", action="store_true", help="使用下面参数自定义 shape (否则跑内置例子)")
    ap.add_argument("--op", choices=["fa_fwd","fag_bwd"], default="fa_fwd")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--N1", type=int, default=16)
    ap.add_argument("--N2", type=int, default=2)
    ap.add_argument("--S1", type=int, default=16384)
    ap.add_argument("--S2", type=int, default=16384)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--Dv", type=int, default=256)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE_BYTES))
    ap.add_argument("--sparse", default="causal", choices=["full","causal","band"])
    ap.add_argument("--band_w", type=int, default=0)
    ap.add_argument("--hw", default="950DT", choices=list(PRESETS))
    ap.add_argument("--actual_us", type=float, default=None, help="可选: 实测耗时 us,出 gap")
    a = ap.parse_args()
    if not a.custom:
        builtin_examples(); return
    s = Shape(a.B,a.N1,a.N2,a.S1,a.S2,a.D,a.Dv,a.dtype,a.sparse,a.band_w)
    report("自定义 shape", s, a.op, PRESETS[a.hw], actual_us=a.actual_us)

if __name__ == "__main__":
    main()