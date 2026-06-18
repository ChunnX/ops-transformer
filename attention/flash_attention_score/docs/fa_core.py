#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fa_core.py —— arch35 FA/FAG 性能建模工具链的共享核心 (单一真源)

放统一的: HW/Shape/Split 数据结构, dtype 表, FLOPs, 理想访存, 工程流量,
L2 工作集/命中模型, 有效带宽两档(L2 / HBM), causal 有效块枚举器与窗口模拟。

被 roofline_mfu.py / traffic_gap.py / fag_viz.py 复用。

关键口径约定 (回答两个常见疑问):
  * "计算访存比" 有两层:
      - 理想 AI  = FLOPs / 理想一遍访存(Q/K/V 各读一遍)  -> 给"绝对天花板", 长 S 恒 compute-bound
      - 工程 AI  = FLOPs / 真实重取访存(L2 级)            -> 真实 bound, 会被重取打塌
    用理想 AI 判 bound 会高估; 必须看工程 AI(见 traffic_gap)。
  * 片上 CV 通路: arch35 的 S(C->V, 经 UB)/ P(V->C, 经 L1)/ dS(经 L1) 是片上中间量,
    走 L0C/UB/L1, 不碰 HBM/L2 -> 不计入 MTE2/FixPipe 的 GM 带宽预算(下面 eng_traffic 只数 GM 张量,
    并单列 on_chip_cv_bytes 仅作信息)。
"""
from __future__ import annotations
from dataclasses import dataclass
import math

# ---------------------------------------------------------------- 硬件
@dataclass
class HW:
    peak_cube_tflops: float = 432.0   # BF16 峰值 (950, 32 核 x 13.5)
    hbm_bw_tbs: float = 4.0           # 950DT HBM 4 TB/s (PR=1.6)
    l2_bw_tbs: float = 16.0           # 950DT L2 有效带宽 ~16 TB/s
    l2_size_mb: float = 128.0
    aic_num: int = 32
    mem_efficiency: float = 0.70      # 实测/峰值 的实现效率 (可由 profiling 校准)

PRESETS = {
    "950DT": HW(432.0, 4.0, 16.0, 128.0, 32, 0.70),
    "950PR": HW(432.0, 1.6, 16.0, 128.0, 32, 0.70),
}

DTYPE_BYTES     = {"bf16":2, "fp16":2, "fp32":4, "fp8":1, "hif8":1, "mxfp4":0.5}
DTYPE_PEAK_MULT = {"bf16":1, "fp16":1, "fp32":0.5, "fp8":2, "hif8":2, "mxfp4":4}
DACC = 4   # FP32 累加 / workspace

# ---------------------------------------------------------------- shape / split
@dataclass
class Shape:
    B:int; N1:int; N2:int; S1:int; S2:int; D:int; Dv:int
    dtype:str = "bf16"
    sparse:str = "causal"     # full | causal | band
    band_w:int = 0
    @property
    def G(self): return self.N1 // self.N2
    def valid_ratio(self):
        return {"full":1.0, "causal":0.5,
                "band":min(1.0, self.band_w/max(1,self.S2))}.get(self.sparse, 1.0)

@dataclass
class Split:
    op: str                       # fa_fwd | fag_bwd
    Br: int = 128; Bc: int = 128
    kv_fused: bool = False        # FA: GQA 是否融合 K/V 加载
    split_axis: str = "BN2GS1S2"  # FAG: BN2 | BN2S2 | BN2GS1S2
    swizzle: bool = False         # 是否走 swizzle 枚举(影响 L2 局部性)
    def good_locality(self):
        """K/V 重取是否有"落 L2"的局部性(决定走 L2 档还是 HBM 档)。"""
        if self.op == "fa_fwd":
            return self.kv_fused or self.swizzle
        return self.swizzle or self.split_axis in ("BN2", "BN2S2")

# ---------------------------------------------------------------- 计算量
def peak_flops(s: Shape, hw: HW) -> float:
    return hw.peak_cube_tflops * 1e12 * DTYPE_PEAK_MULT[s.dtype]

def flops(s: Shape, op: str) -> float:
    if op == "fa_fwd":
        per = 2 * s.S1 * s.S2 * (s.D + s.Dv)              # QK^T + PV
    elif op == "fag_bwd":
        per = 2 * s.S1 * s.S2 * (3 * s.D + 2 * s.Dv)      # MM1..MM5 = 2.5x fwd
    else:
        raise ValueError(op)
    return s.B * s.N1 * per * s.valid_ratio()

# ---------------------------------------------------------------- 理想一遍访存 (绝对天花板)
def ideal_hbm_bytes(s: Shape, op: str) -> float:
    dt = DTYPE_BYTES[s.dtype]
    q = s.B*s.N1*s.S1*s.D*dt; k = s.B*s.N2*s.S2*s.D*dt; v = s.B*s.N2*s.S2*s.Dv*dt
    if op == "fa_fwd":
        return q + k + v + s.B*s.N1*s.S1*s.Dv*dt
    do = s.B*s.N1*s.S1*s.Dv*dt
    dq = s.B*s.N1*s.S1*s.D*dt; dk = s.B*s.N2*s.S2*s.D*dt; dv = s.B*s.N2*s.S2*s.Dv*dt
    return q+k+v+do+dq+dk+dv

# ---------------------------------------------------------------- 工程流量 (真实数据流)
def eng_traffic(s: Shape, sp: Split) -> dict:
    """每条 pipe 真实搬/算多少。只数 GM 上的张量进 MTE2/FixPipe;
       片上 CV 中间量(S/P/dS)单列 on_chip_cv_bytes, 不进 GM 带宽预算。"""
    dt = DTYPE_BYTES[s.dtype]; vr = s.valid_ratio()
    M = math.ceil(s.S1 / sp.Br); Nn = math.ceil(s.S2 / sp.Bc)
    F = flops(s, sp.op)
    extra = {}

    if sp.op == "fa_fwd":
        kvdiv = s.G if sp.kv_fused else 1
        kv_refetch = s.B*s.N1*M*s.S2*(s.D+s.Dv)*dt*vr/kvdiv      # K/V 每个 S1 外块再取
        q_once     = s.B*s.N1*s.S1*s.D*dt                        # Q 读一遍
        mte2 = kv_refetch + q_once
        fixp = s.B*s.N1*s.S1*s.Dv*dt                             # O 写一遍 (GM)
        on_chip = 2 * s.B*s.N1*s.S1*s.S2*dt*vr                   # S(C->V)+P(V->C) 经 UB/L1
        extra = {"kv_refetch_GB": kv_refetch/1e9, "kv_refetch_x": M/kvdiv}
    else:  # fag_bwd
        qdo_refetch = s.B*s.N1*Nn*s.S1*(s.D+s.Dv)*dt*vr          # Q+dO 内层再取
        kv_once     = s.B*s.N2*s.S2*(s.D+s.Dv)*dt                # K/V 读一遍
        mte2 = qdo_refetch + kv_once
        dq_res = s.B*s.N1*s.S1*s.D; dk_res = s.B*s.N2*s.S2*s.D; dv_res = s.B*s.N2*s.S2*s.Dv
        dq_par = s.B*s.N1*s.S1*s.S2*s.D /sp.Bc*vr                # dQ 沿 S2 归约
        dk_par = s.B*s.N1*s.S1*s.S2*s.D /sp.Br*vr                # dK 沿(G,S1)归约
        dv_par = s.B*s.N1*s.S1*s.S2*s.Dv/sp.Br*vr
        if sp.split_axis == "BN2":     fout, nc = dq_res+dk_res+dv_res, 0
        elif sp.split_axis == "BN2S2": fout, nc = dq_par+dk_res+dv_res, 1
        else:                          fout, nc = dq_par+dk_par+dv_par, 3
        fixp = fout * DACC                                       # FP32 workspace 写 (GM)
        on_chip = 2 * s.B*s.N1*s.S1*s.S2*dt*vr                   # P/dS 经 L1
        extra = {"cross_core_tensors": nc,
                 "dq_partial_GB": dq_par*DACC/1e9,
                 "dkv_partial_GB": (dk_par+dv_par)*DACC/1e9}
    return dict(mac_flops=F, mte2_bytes=mte2, fixpipe_bytes=fixp,
                on_chip_cv_bytes=on_chip, **extra)

# ---------------------------------------------------------------- L2 工作集 / 命中模型
def kv_working_set_bytes(s: Shape, sp: Split, hw: HW) -> float:
    """需要并发驻留 L2 的 K/V 数据足迹 (proxy)。
       K/V 数据按 N2 头唯一; 受 aicNum 并发约束。"""
    per_head_kv = s.S2 * (s.D + s.Dv) * DTYPE_BYTES[s.dtype]
    distinct = min(hw.aic_num, s.B * s.N2)
    return distinct * per_head_kv

def mte2_regime(s: Shape, sp: Split, hw: HW) -> dict:
    """判断 K/V 重取走 L2 档(命中)还是 HBM 档(spill), 并给两档有效带宽。
       L2 档需要: 足迹<=L2 且 有局部性(融合/swizzle); 否则 spill 到 HBM。"""
    ws = kv_working_set_bytes(s, sp, hw)
    fits = ws <= hw.l2_size_mb * 1e6
    l2_served = fits and sp.good_locality()
    eff = hw.mem_efficiency
    bw_l2  = eff * hw.l2_bw_tbs  * 1e12
    bw_hbm = eff * hw.hbm_bw_tbs * 1e12
    return dict(working_set_MB=ws/1e6, fits_l2=fits, l2_served=l2_served,
                bw_l2_tbs=bw_l2/1e12, bw_hbm_tbs=bw_hbm/1e12,
                bw_used_tbs=(bw_l2 if l2_served else bw_hbm)/1e12,
                regime=("L2-resident" if l2_served else "HBM-spill"))

# ---------------------------------------------------------------- causal 有效块枚举 (供 viz)
def valid_linear(N: int):
    """线性 decode 顺序: idx=s2o*N+s1o (s1o 最快); 连续=同 s2o 列。causal 下三角。"""
    return [(s1o, s2o) for s2o in range(N) for s1o in range(N) if s2o <= s1o]

def valid_macro(N: int, MS: int):
    """宏块遍历: 逐 MS×MS 宏块(causal), 块在宏块内连续。"""
    seq = []
    for mS1 in range(N // MS):
        for mS2 in range(mS1 + 1):
            diag = (mS1 == mS2)
            for ls2 in range(MS):
                for ls1 in range(MS):
                    s1o, s2o = mS1*MS+ls1, mS2*MS+ls2
                    if s2o <= s1o and not (diag and s2o > s1o):
                        seq.append((s1o, s2o))
    return seq

def assign_linear(seq, aic):
    per = math.ceil(len(seq) / aic)
    return [seq[c*per:(c+1)*per] for c in range(aic)]

def assign_stride(seq, aic, cbn):
    cs = [[] for _ in range(aic)]
    for pos, blk in enumerate(seq):
        cs[(pos // cbn) % aic].append(blk)
    return cs

def simulate_window(core_seq, aic):
    """每 tick 各核当前 s2o 列, 算活跃 K/V 列跨度(=要并发驻留 L2 的列数 proxy)。"""
    import numpy as np
    T = max(len(s) for s in core_seq)
    per_core = [[(t, core_seq[c][t][1]) for t in range(len(core_seq[c]))] for c in range(aic)]
    span = np.full(T, np.nan); lo = np.full(T, np.nan); hi = np.full(T, np.nan); dist = np.full(T, np.nan)
    for t in range(T):
        cols = [core_seq[c][t][1] for c in range(aic) if t < len(core_seq[c])]
        if cols:
            lo[t], hi[t] = min(cols), max(cols)
            span[t] = hi[t]-lo[t]+1; dist[t] = len(set(cols))
    return dict(T=T, per_core=per_core, span=span, distinct=dist, lo=lo, hi=hi)