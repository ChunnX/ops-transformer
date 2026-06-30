#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
golden_combine.py —— CombineTeardown(mxfp8) 输入造数 + CPU golden。
用法: golden_combine.py <bs> <topk> <h> <E> <R> <datadir> [seed]
mxfp8：combine_data=fp8(e4m3fn)，combine_scale=e8m0(u8 字节)，group=32。gather k 份 → 逐组反量化 → 加权和 → x bf16。
造（写 datadir）:
  combine_data.bin  fp8   [R*E*bs, h]      每全局专家一块(bs 槽)，块内 expand_idx 定位
  combine_scale.bin u8    [R*E*bs, h/32]   e8m0 字节，取值 [120,134]→2^-7..2^7
  expert_scales.bin fp32  [bs, topk]
  expand_idx.bin    int32 [bs, topk]       出现次数(块内位置)
  expert_ids.bin    int32 [bs, topk]
golden:
  x.golden.bin      bf16  [bs, h]  Σ_j payload(fp8→f32) × (2^(e8m0-127) × scale[i,j])，fp32 累加末 cast bf16
"""
import sys, os
import numpy as np
import ml_dtypes
BF16 = ml_dtypes.bfloat16
FP8 = ml_dtypes.float8_e4m3fn


def main():
    bs, topk, h, E, R = (int(sys.argv[i]) for i in range(1, 6))
    datadir = sys.argv[6]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 1234
    os.makedirs(datadir, exist_ok=True)
    rng = np.random.default_rng(seed)
    GE = R * E
    scaleH = h // 32

    expert_ids = np.zeros((bs, topk), dtype=np.int32)
    for i in range(bs):
        expert_ids[i] = rng.choice(GE, size=topk, replace=False).astype(np.int32)
    expand_idx = np.zeros((bs, topk), dtype=np.int32)
    counter = np.zeros(GE, dtype=np.int32)
    for i in range(bs):
        for j in range(topk):
            ge = int(expert_ids[i, j])
            expand_idx[i, j] = counter[ge]
            counter[ge] += 1
    assert counter.max() <= bs, "出现次数超过 bs，块容量不足"
    expert_scales = rng.uniform(0.1, 1.0, size=(bs, topk)).astype(np.float32)
    combine_data = rng.standard_normal((GE * bs, h)).astype(np.float32).astype(FP8)       # fp8 payload
    combine_scale = rng.integers(120, 135, size=(GE * bs, scaleH)).astype(np.uint8)       # e8m0 字节

    combine_data.tofile(os.path.join(datadir, "combine_data.bin"))
    combine_scale.tofile(os.path.join(datadir, "combine_scale.bin"))
    expert_scales.tofile(os.path.join(datadir, "expert_scales.bin"))
    expand_idx.tofile(os.path.join(datadir, "expand_idx.bin"))
    expert_ids.tofile(os.path.join(datadir, "expert_ids.bin"))

    # golden x：逐组反量化 fp8×2^(e8m0-127)，乘 expert_scale，fp32 累加，末 cast bf16（与 kernel 同序同精度）
    cd_f32 = combine_data.astype(np.float32)
    x = np.zeros((bs, h), dtype=np.float32)
    for i in range(bs):
        acc = np.zeros(h, dtype=np.float32)
        for j in range(topk):
            ge = int(expert_ids[i, j])
            pos = int(expand_idx[i, j])
            w = np.float32(expert_scales[i, j])
            srcRow = ge * bs + pos
            sc_row = (2.0 ** (combine_scale[srcRow].astype(np.int32) - 127)).astype(np.float32)  # [scaleH]
            combined = (np.repeat(sc_row, 32) * w).astype(np.float32)                              # [h] = 2^(E-127)*w
            acc += (cd_f32[srcRow] * combined).astype(np.float32)
        x[i] = acc
    x.astype(BF16).tofile(os.path.join(datadir, "x.golden.bin"))
    print(f"[golden] bs={bs} topk={topk} h={h} E={E} R={R} GE={GE} scaleH={scaleH} seed={seed} max_occ={int(counter.max())} (mxfp8)")


if __name__ == "__main__":
    main()
