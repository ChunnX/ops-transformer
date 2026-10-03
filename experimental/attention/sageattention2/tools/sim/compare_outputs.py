#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2 门禁数值比对：设备侧（camodel 仿真）输出 vs 宿主孪生 golden。

用法：python3 compare_outputs.py [目录]（缺省 "."，内含 golden/device bin）。

判定口径（任务书冻结）：
- O（fp16 输出）：报告位级一致率与最大 ulp 差；
- LSE（fp32）：相对差 ≤ 1e-3 量级；
- FastExp2（设备侧位构造多项式）vs std::exp2（宿主）的预期 ≤1ulp 偏差据此判读。
"""
import sys

import numpy as np

B, HQ, HKV, SQ, SK, D = 1, 1, 1, 128, 128, 64


def read_u16(path, n):
    with open(path, "rb") as f:
        raw = f.read()
    assert len(raw) == n * 2, f"{path}: 期望 {n * 2} 字节，实得 {len(raw)}"
    return np.frombuffer(raw, dtype="<u2").reshape(n)


def read_f32(path, n):
    with open(path, "rb") as f:
        raw = f.read()
    assert len(raw) == n * 4, f"{path}: 期望 {n * 4} 字节，实得 {len(raw)}"
    return np.frombuffer(raw, dtype="<f4").reshape(n)


def fp16_view(bits):
    """uint16 位型 → float16 数值视图"""
    return bits.astype(np.float16)


def ulp_diff_fp16(a_bits, b_bits):
    """fp16 位型的有序整数差 = ulp 距离（NaN 记为 32768）"""
    a = a_bits.astype(np.int32)
    b = b_bits.astype(np.int32)

    # 负数位型转有序整数（符号位映射后单调）
    def to_order(x):
        return np.where(x < 0, -32768 + (x & 0x7FFF), x)

    ulp = np.abs(to_order(a) - to_order(b))
    nan_mask = ((a_bits & 0x7C00) == 0x7C00) | ((b_bits & 0x7C00) == 0x7C00)
    ulp[nan_mask] = 32768
    return ulp


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    n_o = B * SQ * HQ * D
    n_lse = B * HQ * SQ

    g_bits = read_u16(f"{d}/golden_o_fp16.bin", n_o)
    e_bits = read_u16(f"{d}/device_o_fp16.bin", n_o)
    g_lse = read_f32(f"{d}/golden_lse_f32.bin", n_lse)
    e_lse = read_f32(f"{d}/device_lse_f32.bin", n_lse)

    print("=" * 62)
    print("O 比对（fp16 输出，NHD [b,s,h,d]）")
    print("=" * 62)
    exact = int(np.sum(g_bits == e_bits))
    print(f"  位级一致率 : {exact}/{n_o} = {exact / n_o:.6f}")
    ulp = ulp_diff_fp16(g_bits, e_bits)
    print(f"  ulp 分布   : max={ulp.max()} 均值={ulp.mean():.4f} "
          f"|≤1ulp={int(np.sum(ulp <= 1))} ≤2ulp={int(np.sum(ulp <= 2))}")
    g_v = fp16_view(g_bits).astype(np.float64)
    e_v = fp16_view(e_bits).astype(np.float64)
    denom = np.maximum(np.abs(g_v), 1e-6)
    rel = np.abs(e_v - g_v) / denom
    print(f"  最大相对差 : {rel.max():.6f}   最大绝对差: {np.abs(e_v - g_v).max():.6f}")
    print(f"  golden 范围: [{g_v.min():.4f}, {g_v.max():.4f}]  "
          f"device 范围: [{e_v.min():.4f}, {e_v.max():.4f}]")

    print("=" * 62)
    print("LSE 比对（fp32）")
    print("=" * 62)
    absdiff = np.abs(e_lse.astype(np.float64) - g_lse.astype(np.float64))
    rel_lse = absdiff / np.maximum(np.abs(g_lse.astype(np.float64)), 1e-6)
    print(f"  golden : {np.array2string(g_lse, precision=6, max_line_width=100)}")
    print(f"  device : {np.array2string(e_lse, precision=6, max_line_width=100)}")
    print(f"  绝对差 : {np.array2string(absdiff, precision=6, max_line_width=100)}")
    print(f"  相对差 : max={rel_lse.max():.3e}")

    print("=" * 62)
    o_bit_ok = exact == n_o
    o_ulp_ok = bool(ulp.max() <= 1)
    lse_ok = bool(rel_lse.max() <= 1e-3)
    print(f"判定: O 位级全一致={o_bit_ok}  O ≤1ulp={o_ulp_ok}  LSE 相对差≤1e-3={lse_ok}")
    if not lse_ok:
        print("  LSE 超差提示：smooth_k=1 用例下 LSE 含 q·km 校正项（×sm_scale），")
        print("  该项经 lseCorrection GM 入参两侧同源传入（V1 FinalizeAndStoreOutput 直读），")
        print("  超差不属预期域，须查 harness/kernel 数值轨迹。")
    print(f"结论: {'PASS' if (o_bit_ok or o_ulp_ok) and lse_ok else 'FAIL'}"
          f"（FastExp2 多项式 vs std::exp2 的 ≤1ulp 预期 {'成立' if o_ulp_ok else '不成立'}）")
    sys.exit(0 if (o_bit_ok or o_ulp_ok) and lse_ok else 1)


if __name__ == "__main__":
    main()
