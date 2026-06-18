#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_bw.py —— 多 shape 自动拟合有效带宽 + 量化"线性够不够 / 争抢重不重" (import fa_core)

给一批 (shape, split, profiling) 自动反推有效带宽/算力, 并把你担心的两件事变成可测量:

  模式1 [有 pipe 占空比 mac/mte2/fixpipe_ratio]:
    对每条 pipe 做"过原点线性回归"  pipe_time = traffic / BW
    -> 输出 BW + R² + 每点残差。R² 就是"单一线性带宽够不够"的直接答案:
       R²≈1 -> 该 pipe 用一个常数带宽足够; R² 低/残差随并发增大 -> 非线性/争抢不可忽略。

  模式2 [只有耗时 duration]:
    瓶颈-EM 拟合三条 BW(不同 shape 被不同 pipe bound, 才能解耦);并对比两种"内存子系统"假设:
      A 独立流水:  dur ≈ max(t_mac, t_mte2, t_fixpipe) / overlap        (roofline 式, 假设互不抢)
      B 共享总线:  dur ≈ max(t_mac, (mte2+fixpipe)/BW_hbm_shared)/overlap (读写共享 HBM, 会相加抢)
    比较 A/B 的拟合残差(RMSE): B 明显更优 -> 带宽争抢显著, 不能用独立线性叠加。

依赖: numpy, scipy。把下面 SYNTHETIC_BATCH 换成你真实的一批测量即可。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from fa_core import HW, PRESETS, Shape, Split, peak_flops, eng_traffic

@dataclass
class Meas:
    shape: Shape
    split: Split
    duration_us: float
    mac_ratio: float = None
    mte2_ratio: float = None
    fixpipe_ratio: float = None

# ============================================================ 模式1: 每 pipe 线性回归
def _fit_through_origin(traffic, time):
    """time = traffic / BW 过原点最小二乘; 返回 BW 与(过原点)R²。"""
    traffic = np.asarray(traffic, float); time = np.asarray(time, float)
    k = np.sum(traffic*time) / np.sum(traffic*traffic)   # k = 1/BW
    pred = k*traffic
    ss_res = np.sum((time-pred)**2); ss_tot = np.sum(time**2)   # 过原点用未中心化
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    resid_pct = (time-pred)/np.where(time==0,1,time)*100
    return 1.0/k, r2, resid_pct

def fit_per_pipe(ms, hw):
    print("="*78)
    print("模式1 — 每条 pipe 过原点线性回归 (需 pipe 占空比)。R² 即'单一线性带宽够不够'。")
    pipes = {"mac":("mac_flops","mac_ratio"), "mte2":("mte2_bytes","mte2_ratio"),
             "fixpipe":("fixpipe_bytes","fixpipe_ratio")}
    out = {}
    for name,(tkey,rkey) in pipes.items():
        pts = [(eng_traffic(m.shape,m.split)[tkey], getattr(m,rkey)*m.duration_us*1e-6)
               for m in ms if getattr(m,rkey) is not None]
        if len(pts) < 2:
            print(f"  [{name:7}] 点数 {len(pts)} < 2, 跳过(需要至少 2 个该 pipe 有占空比的 shape)"); continue
        tr = np.array([p[0] for p in pts]); tm = np.array([p[1] for p in pts])
        bw, r2, resid = _fit_through_origin(tr, tm)
        unit = "TFLOPS" if name=="mac" else "TB/s"
        val = bw/1e12
        print(f"  [{name:7}] 拟合 {val:7.2f} {unit:6} | R²={r2:6.3f} | n={len(pts)} | "
              f"残差% {np.array2string(resid, precision=1, floatmode='fixed')}")
        if r2 < 0.9 and len(pts) >= 3:
            print(f"            ^ R²<0.9: 单一线性带宽不足, 可能有争抢/非线性(看残差是否随流量/并发增大)")
        out[name] = dict(bw=val, r2=r2, n=len(pts))
    return out

# ============================================================ 模式2: 只有耗时 + 争抢诊断
def _times(m, hw, bw_mac, bw_mte2, bw_fix):
    tr = eng_traffic(m.shape, m.split)
    return (tr["mac_flops"]/(bw_mac*1e12),
            tr["mte2_bytes"]/(bw_mte2*1e12),
            tr["fixpipe_bytes"]/(bw_fix*1e12), tr)

def _resid_indep(p, ms, hw):
    bw_mac, bw_mte2, bw_fix, ov = p
    r = []
    for m in ms:
        tmac,tmte2,tfix,_ = _times(m, hw, bw_mac, bw_mte2, bw_fix)
        r.append((max(tmac,tmte2,tfix)/ov - m.duration_us*1e-6)*1e6)
    return r

def fit_durations(ms, hw):
    """只有耗时、且带宽当自由参数拟合: 给每 pipe 的'有效带宽'+overlap。
       关键警告: 这样拟合出的有效带宽会把'争抢/L2-miss/调度'全部吸收进去 ——
       对'同 co-load 区间'的预测够用, 但无法把争抢单独分离出来(见 diagnose_contention)。"""
    print("="*78)
    print("模式2a — 拟合有效带宽(自由参数)。注意: 争抢会被吸收成'看起来更低的带宽', 无法单独识别。")
    a = least_squares(_resid_indep, x0=[400,3,3,0.9], args=(ms,hw),
                      bounds=([50,0.5,0.5,0.3],[600,20,20,1.0]))
    bw_mac, bw_mte2, bw_fix, ov = a.x
    rmse = np.sqrt(np.mean(np.array(a.fun)**2))
    print(f"  有效带宽: BW_mac={bw_mac:.0f}T  BW_mte2={bw_mte2:.2f}  BW_fix={bw_fix:.2f}  overlap={ov:.2f}  RMSE={rmse:.0f}us")
    print(f"  (若这些有效带宽明显低于微基准实测峰值, 差额里就含争抢/miss —— 但分不清各占多少)")
    return dict(bw=a.x, rmse=rmse)

def diagnose_contention(ms, hw, bw_mac, bw_mte2, bw_fix):
    """用'独立已知'的 per-pipe 带宽(来自微基准 或 mode-1 占空比拟合, 不是拿这批耗时回归来的),
       回归 [实测/独立预测 floor] vs [co-load 双流水并发度]:
         截距 ≈ 1/overlap(基线重叠); 斜率>0 => co-load 越高越慢 = 争抢, 斜率/截距 ≈ 齐载额外开销。
       这是从耗时分离争抢的唯一稳妥前提: 带宽必须独立给定, 否则争抢被吸收(见 fit_durations)。"""
    print("="*78)
    print(f"模式2b — 争抢诊断(带宽独立给定: mac={bw_mac:.0f}T mte2={bw_mte2:.2f} fix={bw_fix:.2f})")
    rows = []
    for m in ms:
        tmac,tmte2,tfix,_ = _times(m, hw, bw_mac, bw_mte2, bw_fix)
        ts = sorted([tmac,tmte2,tfix], reverse=True)
        floor = ts[0]; coload = ts[1]/ts[0] if ts[0] > 0 else 0.0
        rows.append((coload, (m.duration_us*1e-6)/floor))
    cl = np.array([r[0] for r in rows]); rt = np.array([r[1] for r in rows])
    slope, intercept = np.linalg.lstsq(np.vstack([cl, np.ones_like(cl)]).T, rt, rcond=None)[0]
    print(f"  co-load 范围 [{cl.min():.2f}, {cl.max():.2f}] | 回归 实测/独立预测 = {intercept:.2f} + {slope:.2f}·co-load")
    if cl.max()-cl.min() < 0.2:
        print(f"  >>> co-load 跨度太小(shape 都单流水主导)=> 耗时识别不出争抢, 需齐载 shape 或 pipe 占空比")
    elif slope > 0.12:
        print(f"  >>> 斜率 {slope:+.2f} 显著 => 存在带宽争抢, 齐载(co-load≈1)时比独立预测额外慢 ~{slope/intercept*100:.0f}%")
    else:
        print(f"  >>> 斜率 {slope:+.2f} ≈ 0 => 争抢不显著, 独立线性带宽够用; 基线 1/overlap≈{intercept:.2f}")
    return dict(slope=slope, intercept=intercept, coload=(cl.min(),cl.max()))

# ============================================================ 合成 demo (换成你的真实批次)
def make_synthetic(hw, contention=0.30, noise=0.03, seed=0):
    """造一批 shape, 真值: 每 pipe 独立带宽 mac=403T/mte2=2.8/fix=3.4, overlap=0.9,
       并注入'随 co-load(双流水并发度)上升的争抢': dur = floor/ov · (1 + contention·coload)。
       contention=0 表示无争抢。混入 FA(mte2主导)/BN2S2(读写齐载)/BN2GS1S2(fix主导)使 coload 有跨度。"""
    rng = np.random.default_rng(seed)
    grid = []
    for S in [8192, 12288, 16384, 20480]:
        grid += [("fa_fwd",S,None), ("fag_bwd",S,"BN2S2"), ("fag_bwd",S,"BN2GS1S2")]
    ms = []
    for op,S,ax in grid:
        s = Shape(1,16,2,S,S,256,256,"bf16","causal")
        sp = Split(op, kv_fused=False) if op=="fa_fwd" else Split(op, split_axis=ax)
        tr = eng_traffic(s, sp)
        tmac = tr["mac_flops"]/403e12; tmte2 = tr["mte2_bytes"]/2.8e12; tfix = tr["fixpipe_bytes"]/3.4e12
        ts = sorted([tmac,tmte2,tfix], reverse=True)
        floor = ts[0]; coload = ts[1]/ts[0]
        dur = floor/0.9 * (1 + contention*coload) * (1+rng.normal(0,noise))
        ms.append(Meas(s, sp, dur*1e6))
    return ms

def main():
    hw = PRESETS["950DT"]
    print("【演示用合成批次(12 个 shape): 真值每 pipe 独立带宽 mac=403T/mte2=2.8/fix=3.4 + overlap 0.9,")
    print(" 注入随 co-load 上升的争抢。把 make_synthetic 换成你真实的一批即可】\n")

    ms = make_synthetic(hw, contention=0.30)
    print(">>> 批次注入了 30% 争抢(齐载时慢 30%)")
    fit_durations(ms, hw)
    print("    ^ 看: 拟合的有效 BW_mte2/overlap 都被压低了 —— 争抢被吸收, 这条路看不见它。\n")
    print(">>> 改用独立已知带宽(微基准/mode-1 的 2.8/3.4/403)来诊断:")
    diagnose_contention(ms, hw, 403, 2.8, 3.4)

    print("\n>>> 对照: 同样独立带宽, 但批次本身无争抢(contention=0):")
    diagnose_contention(make_synthetic(hw, contention=0.0), hw, 403, 2.8, 3.4)

    print("\n>>> 模式1 演示(用你两个真实 profiling 的 pipe 占空比; n=2 仅示意, 真用每 pipe 需 5~8 个 shape):")
    real = [
        Meas(Shape(1,16,2,16384,16384,256,256,"bf16","causal"), Split("fa_fwd",kv_fused=False),
             7771.0, mac_ratio=0.702, mte2_ratio=0.801),
        Meas(Shape(1,16,2,16384,16384,256,256,"bf16","causal"), Split("fag_bwd",split_axis="BN2GS1S2"),
             15453.0, mac_ratio=0.885, fixpipe_ratio=0.977),
    ]
    fit_per_pipe(real, hw)
    print("\n小结:")
    print("  · 模式1(占空比): R² 直接判'单一线性带宽够不够'; 每 pipe 独立拟合, 不受其他 pipe 干扰。")
    print("  · 模式2a(拟合带宽): 方便, 但争抢/miss 会被吸收成'更低的有效带宽', 分不清成分。")
    print("  · 模式2b(独立带宽诊断): 只有 per-pipe 带宽独立给定, 才能从耗时缺口随 co-load 的斜率分离出争抢。")
    print("  => 想真正量化争抢: 优先拿 pipe 占空比(模式1)或微基准定带宽, 再看 co-load 斜率。光有耗时分不清。")

if __name__ == "__main__":
    main()